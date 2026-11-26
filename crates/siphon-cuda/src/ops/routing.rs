//! MoE expert routing and depth attention (`kernels/routing.cu`).

use std::{mem::size_of, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig,
    PushKernelArg as _,
    sys::{CUdevice_attribute, CUfunction_attribute},
};
use half::bf16;

use crate::{
    gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module},
    ops::QuantDest,
};

/// Hidden width of the model.
const HIDDEN: usize = 2304;
/// Routed experts.
pub const EXPERTS: usize = 128;
/// Experts chosen per token.
pub const TOP_K: usize = 8;
/// Default floor on kept experts when probability-threshold pruning is active.
pub const DEFAULT_MIN_K: usize = 4;

/// Opt-in MoE dynamic top-k and probability-threshold expert pruning.
///
/// The default (`top_k = 8`, `threshold = 0.0`, `min_k = 4`) keeps every slot and runs
/// the exact stock top-8 normalization loop. When active, a ranked slot `k` (`0..8`,
/// highest biased score first) is kept iff `k < top_k` and either `k < min_k` or its
/// normalized share of the top-8 score sum is at least `threshold`. Pruned slots receive
/// index `-1` and weight `0.0`, and kept slots are renormalized to sum to `route_scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoePruning {
    /// Maximum experts kept per token (`1..=8`).
    pub top_k: usize,
    /// Minimum normalized weight share (`0.0..=1.0`) required to keep a slot at or above
    /// `min_k`. `0.0` disables threshold pruning.
    pub threshold: f32,
    /// Minimum experts always kept per token (`1..=top_k`), regardless of `threshold`.
    pub min_k: usize,
}

impl Default for MoePruning {
    fn default() -> Self {
        Self {
            top_k: TOP_K,
            threshold: 0.0,
            min_k: DEFAULT_MIN_K,
        }
    }
}

impl MoePruning {
    /// Checks that `1 <= min_k <= top_k <= 8` and `0.0 <= threshold <= 1.0`.
    pub fn validate(self) -> anyhow::Result<()> {
        ensure!(
            (1..=TOP_K).contains(&self.top_k),
            "moe top_k must be in 1..={TOP_K}, got {}",
            self.top_k
        );
        ensure!(
            (1..=self.top_k).contains(&self.min_k),
            "moe min_k must be in 1..={}, got {}",
            self.top_k,
            self.min_k
        );
        ensure!(
            self.threshold.is_finite() && (0.0..=1.0).contains(&self.threshold),
            "moe prune threshold must be in 0.0..=1.0, got {}",
            self.threshold
        );
        Ok(())
    }

    /// Whether pruning alters routing from stock top-8.
    #[must_use]
    pub fn is_active(self) -> bool {
        self.top_k < TOP_K || self.threshold > 0.0
    }

    fn kernel_args(self) -> anyhow::Result<(i32, i32, f32)> {
        self.validate()?;
        Ok((
            i32::try_from(self.top_k)?,
            i32::try_from(self.min_k)?,
            self.threshold,
        ))
    }
}

impl std::fmt::Display for MoePruning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "top_k={}, min_k={}, threshold={:.4}",
            self.top_k, self.min_k, self.threshold
        )
    }
}
/// Rows of one cluster of [`Routing::moe_router_step_rows`], `kStepRouterRows` in the
/// kernel.
const STEP_ROUTER_GROUP: usize = 8;
/// Most rows [`Routing::moe_router_step_rows`] takes: 32 clusters of 8 rows.
pub const STEP_ROUTER_ROWS: usize = 32 * STEP_ROUTER_GROUP;
/// Blocks of `moe_router_step_2304`, one cluster, `kRouterCluster` in the kernel.
const STEP_ROUTER_CLUSTER: u32 = 16;
/// Threads per block of `moe_router_step_rows_2304`, `kRowsRouterThreads` in the kernel:
/// one warp per virtual warp of the step router's 288-thread sum order.
const STEP_ROUTER_ROWS_THREADS: u32 = 288;
/// Threads per block of `router_logits_2304`, which fixes its summation order.
const LOGITS_THREADS: u32 = 256;
/// Threads per block of `expert_routing_128_top8`: one warp per row.
const ROUTING_THREADS: u32 = 256;
/// Threads per row of `depth_attention_2304`, which fixes its summation order.
const DEPTH_THREADS: u32 = 288;
/// Candidates and queries `depth_attention_fused_2304` holds in registers.
pub const FUSED_MAX_CANDIDATES: usize = 6;
const _: () = assert!(FUSED_MAX_CANDIDATES == siphon::memory::FUSED_MAX_CANDIDATES);
pub const FUSED_MAX_QUERIES: usize = 4;
/// Normalized copies `depth_attention_fused_2304` can write.
pub const FUSED_DESTS: usize = 4;
/// Blocks and threads of `moe_router_2304`.
const ROUTER_BLOCKS: u32 = 64;
const ROUTER_THREADS: u32 = 288;
/// Slices of the inner dimension that `moe_router_logits_rows_2304` sums separately, of
/// 256 columns each.
const ROUTER_SPLITS: usize = 9;
const _: () = assert!(ROUTER_SPLITS == siphon::memory::ROUTER_SPLITS);
/// Rows per block of `moe_router_logits_rows_2304`, which covers all 128 experts.
const ROUTER_TILE_ROWS: usize = 64;
const ROUTER_TILE_THREADS: u32 = 256;
/// Query counts with a `depth_attention_rows_2304_<nb>_<nq>` kernel.
const ROWS_QUERY_COUNTS: [usize; 3] = [1, 3, 4];
/// Query counts with a `depth_attention_rows_quant_2304_<nb>_<nq>` kernel.
const ROWS_QUANT_QUERY_COUNTS: [usize; 2] = [1, 3];

/// The partial candidate of [`Routing::depth_attention_fused`].
pub enum Partial<'a> {
    /// An F32 row that is already normalized.
    Normalized(CudaView<'a, f32>),
    /// An F32 row that gets an unweighted RMS norm first.
    F32(CudaView<'a, f32>),
    /// A BF16 row that is widened and gets an unweighted RMS norm first.
    Bf16(CudaView<'a, bf16>),
}

/// The partial candidate of [`Routing::depth_attention_step`].
pub enum StepPartial<'a> {
    /// An F32 row that is already normalized.
    Normalized(CudaView<'a, f32>),
    /// An F32 row that gets an unweighted RMS norm first.
    F32(CudaView<'a, f32>),
    /// A BF16 row that is widened and gets an unweighted RMS norm first.
    Bf16(CudaView<'a, bf16>),
    /// The post-attention conv and residual add of [`crate::ops::conv::Conv::short_conv_residual`],
    /// computed in the kernel and written to `residual`, then an unweighted RMS norm.
    ConvResidual {
        /// The attention output, BF16 `[2304]`.
        x: CudaView<'a, bf16>,
        /// The conv weight, BF16 `[2304, 4]`.
        weight: &'a CudaSlice<bf16>,
        /// The conv state, F32 `[3, 2304]`, updated in place.
        state: CudaViewMut<'a, f32>,
        /// The visit input added to the conv output, or `None` at a block start.
        hidden: Option<CudaView<'a, bf16>>,
        /// Receives the residual, F32 `[2304]`.
        residual: CudaViewMut<'a, f32>,
    },
}

/// What [`Routing::depth_attention_step`] reads and writes.
pub struct DepthStep<'a> {
    /// The block states, F32 `[nb, 2304]`.
    pub blocks: CudaView<'a, f32>,
    pub partial: StepPartial<'a>,
    /// Receives the partial as used, F32 `[2304]`.
    pub normed_partial: Option<CudaViewMut<'a, f32>>,
    /// `batch` sets of `nq` queries, F32 `[batch, nq, 2304]`.
    pub queries: CudaView<'a, f32>,
    pub batch: usize,
    /// Receives the output rows, F32 `[batch, nq, 2304]`.
    pub out: Option<CudaViewMut<'a, f32>>,
    /// Destination `k` receives output row `min(k, nq - 1)` of each set after an RMS norm,
    /// the sets `dst_stride` elements apart.
    pub dests: [Option<NormDest<'a>>; FUSED_DESTS],
    pub dst_stride: usize,
}

/// The partial candidate of [`Routing::depth_attention_rows`], `[rows, 2304]`.
pub enum RowsPartial<'a> {
    /// F32 rows that are already normalized.
    Normalized(CudaView<'a, f32>),
    /// F32 rows that get an unweighted RMS norm first.
    F32(CudaView<'a, f32>),
    /// BF16 rows that are widened and get an unweighted RMS norm first.
    Bf16(CudaView<'a, bf16>),
    /// The post-attention conv and residual add of
    /// [`crate::ops::conv::Conv::short_conv_residual_rows`], computed in the kernel and
    /// written to `residual`, then an unweighted RMS norm.
    ConvResidual {
        /// The attention output, BF16 `[rows, 2304]`.
        x: CudaView<'a, bf16>,
        /// The conv weight, BF16 `[2304, 4]`.
        weight: &'a CudaSlice<bf16>,
        /// The conv state before the chunk, F32 `[3, 2304]`. The kernel only reads it.
        state: CudaView<'a, f32>,
        /// The visit input added to the conv output, or `None` at a block start.
        hidden: Option<CudaView<'a, bf16>>,
        /// Receives the residual, F32 `[rows, 2304]`.
        residual: CudaViewMut<'a, f32>,
    },
    /// The post-attention conv and residual add of
    /// [`crate::ops::conv::Conv::conv_residual_batch`], computed in the kernel and
    /// written to `residual`, then an unweighted RMS norm: each sequence of
    /// `rows_per_seq` rows reads its post-attention conv state from word 4 of `states`.
    ConvResidualBatch {
        /// The attention output, BF16 `[rows, 2304]`.
        x: CudaView<'a, bf16>,
        /// The conv weight, BF16 `[2304, 4]`.
        weight: &'a CudaSlice<bf16>,
        /// Per-sequence state table, `[seqs * BATCH_STATE_WORDS]` device addresses
        /// (word 4 is the post-attention conv state, F32 `[3, 2304]`, only read).
        states: CudaView<'a, u64>,
        /// Consecutive rows belonging to each sequence (`1..=8`).
        rows_per_seq: usize,
        /// The visit input added to the conv output, or `None` at a block start.
        hidden: Option<CudaView<'a, bf16>>,
        /// Receives the residual, F32 `[rows, 2304]`.
        residual: CudaViewMut<'a, f32>,
    },
}

/// What [`Routing::depth_attention_rows`] reads and writes for a chunk of `rows` rows.
pub struct DepthRows<'a> {
    /// The block states, 1 to 5 runs of F32 `[rows, 2304]`.
    pub blocks: CudaView<'a, f32>,
    pub partial: RowsPartial<'a>,
    /// Receives the partial as used, F32 `[rows, 2304]`.
    pub normed_partial: Option<CudaViewMut<'a, f32>>,
    /// The `nq` queries, F32 `[nq, 2304]`.
    pub queries: &'a CudaSlice<f32>,
    /// Receives the output rows, F32 `[rows, nq, 2304]`.
    pub out: Option<CudaViewMut<'a, f32>>,
    /// Receives the weights of each output, F32 `[rows, nq, nb + 1]`, from which
    /// [`crate::ops::misc::MessageMix`] rebuilds it: the block states' softmax weights, then
    /// the partial's, which also carries the partial's norm scale.
    pub probs: Option<CudaViewMut<'a, f32>>,
    /// Destination `k` receives output row `min(k, nq - 1)` of each row after an RMS norm,
    /// the rows `dst_stride` elements apart, so a destination may be one half of a
    /// `[rows, 2 * 2304]` buffer: its view starts at row 0 and reaches the end of the last row.
    pub dests: [Option<NormDest<'a>>; FUSED_DESTS],
    pub dst_stride: usize,
    pub rows: usize,
}

/// The sets of [`Routing::depth_attention_rows_batch`]: how many, and how far apart each
/// set's weights and destinations are, in elements. The default is one set.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowsBatch {
    pub count: usize,
    pub probs_stride: usize,
    pub dst_stride: usize,
}

/// A BF16 copy of one depth-attention output row after an RMS norm.
pub struct NormDest<'a> {
    pub out: CudaViewMut<'a, bf16>,
    /// The weight of a weighted RMS norm, or `None` for an unweighted one.
    pub weight: Option<&'a CudaSlice<bf16>>,
}

/// A depth-attention output destination with fused MXFP8 or NVFP4 activation quantization
/// (and an optional BF16 copy when still read downstream).
pub struct QuantNormDest<'a> {
    pub out: Option<CudaViewMut<'a, bf16>>,
    /// The weight of a weighted RMS norm, or `None` for an unweighted one.
    pub weight: Option<&'a CudaSlice<bf16>>,
    /// Quantized destination (`values`, `scales`, optional `outer`).
    pub quant: QuantDest<'a>,
}

/// What [`Routing::depth_attention_rows_quant`] reads and writes for a chunk of `rows` rows.
pub struct DepthRowsQuant<'a> {
    /// The block states, 1 to 5 runs of F32 `[rows, 2304]`.
    pub blocks: CudaView<'a, f32>,
    pub partial: RowsPartial<'a>,
    /// Receives the partial as used, F32 `[rows, 2304]`.
    pub normed_partial: Option<CudaViewMut<'a, f32>>,
    /// The `nq` (1 or 3) queries, F32 `[nq, 2304]`.
    pub queries: &'a CudaSlice<f32>,
    /// Receives the output rows, F32 `[rows, nq, 2304]`.
    pub out: Option<CudaViewMut<'a, f32>>,
    /// Receives the weights of each output, F32 `[rows, nq, nb + 1]`.
    pub probs: Option<CudaViewMut<'a, f32>>,
    /// Destinations `0..3`, all using the same [`crate::ops::QuantFormat`].
    pub dests: [Option<QuantNormDest<'a>>; 3],
    pub dst_stride: usize,
    pub rows: usize,
}

/// How `Routing::router_logits` treats the F32 router weight.
///
/// MLX on the M5 reference machine computes an F32 matmul with one row in a GEMV at full
/// precision, but with two or more rows in a GEMM that truncates both operands' mantissas
/// to TF32's 10 bits. Matching MLX means picking `Tf32` exactly when MLX had more than one
/// row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterMatmul {
    /// Full F32 products, summed in the order of MLX's GEMV.
    F32,
    /// Weight mantissas truncated to 10 bits. The hidden states are BF16, which TF32 holds
    /// exactly.
    Tf32,
}

/// Scratch of [`Routing::moe_router_rows`] for chunks of up to `rows` rows.
pub struct RouterRowsScratch {
    rows: usize,
    /// The sums of each logit over each slice of the inner dimension, F32 `[9, rows, 128]`.
    partials: CudaSlice<f32>,
}

impl RouterRowsScratch {
    pub fn new(stream: &Arc<CudaStream>, rows: usize) -> anyhow::Result<Self> {
        Ok(Self {
            rows,
            partials: stream.alloc_zeros(ROUTER_SPLITS * rows * EXPERTS)?,
        })
    }
}

#[derive(Debug)]
pub struct Routing {
    router_logits_2304: CudaFunction,
    expert_routing_128_top8: CudaFunction,
    depth_attention_2304: CudaFunction,
    depth_attention_fused_2304: CudaFunction,
    moe_router_2304: CudaFunction,
    moe_router_logits_rows_2304: CudaFunction,
    moe_router_route_rows_2304: CudaFunction,
    depth_attention_step_2304: CudaFunction,
    moe_router_step_2304: CudaFunction,
    moe_router_step_rows_2304: CudaFunction,
    /// `depth_attention_rows_2304_<nb>_<nq>` at `[nq index][nb - 1]`, with the most blocks
    /// that run at once.
    depth_attention_rows_2304: Vec<Vec<(CudaFunction, u32)>>,
    /// `depth_attention_rows_quant_2304_<nb>_<nq>` at `[nq index][nb - 1]` for `nq` in `[1, 3]`.
    depth_attention_rows_quant_2304: Vec<Vec<(CudaFunction, u32)>>,
}

impl Routing {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "routing", cubin!("routing"))?;
        let sms = u32::try_from(
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        )?;
        Ok(Self {
            router_logits_2304: module.load_function("router_logits_2304")?,
            expert_routing_128_top8: module.load_function("expert_routing_128_top8")?,
            depth_attention_2304: module.load_function("depth_attention_2304")?,
            depth_attention_fused_2304: module.load_function("depth_attention_fused_2304")?,
            moe_router_2304: module.load_function("moe_router_2304")?,
            moe_router_logits_rows_2304: module.load_function("moe_router_logits_rows_2304")?,
            moe_router_route_rows_2304: module.load_function("moe_router_route_rows_2304")?,
            depth_attention_step_2304: module.load_function("depth_attention_step_2304")?,
            moe_router_step_2304: {
                let f = module.load_function("moe_router_step_2304")?;
                // Its compiled cluster of 16 blocks is above the portable limit of 8. The
                // sm_89 build has no clusters.
                if !cfg!(sm89) {
                    f.set_attribute(
                        CUfunction_attribute::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                        1,
                    )?;
                }
                f
            },
            moe_router_step_rows_2304: {
                let f = module.load_function("moe_router_step_rows_2304")?;
                if !cfg!(sm89) {
                    f.set_attribute(
                        CUfunction_attribute::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                        1,
                    )?;
                }
                f
            },
            depth_attention_rows_2304: ROWS_QUERY_COUNTS
                .iter()
                .map(|nq| {
                    (1..FUSED_MAX_CANDIDATES)
                        .map(|nb| {
                            let f = module
                                .load_function(&format!("depth_attention_rows_2304_{nb}_{nq}"))?;
                            let per_sm = f.occupancy_max_active_blocks_per_multiprocessor(
                                DEPTH_THREADS,
                                0,
                                None,
                            )?;
                            Ok((f, per_sm.max(1) * sms))
                        })
                        .collect::<anyhow::Result<Vec<_>>>()
                })
                .collect::<anyhow::Result<_>>()?,
            depth_attention_rows_quant_2304: ROWS_QUANT_QUERY_COUNTS
                .iter()
                .map(|nq| {
                    (1..FUSED_MAX_CANDIDATES)
                        .map(|nb| {
                            let f = module.load_function(&format!(
                                "depth_attention_rows_quant_2304_{nb}_{nq}"
                            ))?;
                            let per_sm = f.occupancy_max_active_blocks_per_multiprocessor(
                                DEPTH_THREADS,
                                0,
                                None,
                            )?;
                            Ok((f, per_sm.max(1) * sms))
                        })
                        .collect::<anyhow::Result<Vec<_>>>()
                })
                .collect::<anyhow::Result<_>>()?,
        })
    }

    /// Router logits `F32(hidden) @ weight.T` for BF16 `hidden` (`[rows, 2304]`) and the
    /// F32 router weight (`[128, 2304]`) into `logits` (`[rows, 128]`).
    pub fn router_logits(
        &self,
        stream: &CudaStream,
        hidden: &CudaSlice<bf16>,
        weight: &CudaSlice<f32>,
        logits: &mut CudaSlice<f32>,
        matmul: RouterMatmul,
    ) -> anyhow::Result<()> {
        ensure!(
            weight.len() == EXPERTS * HIDDEN,
            "weight has {} elements, expected {EXPERTS} x {HIDDEN}",
            weight.len()
        );
        ensure!(
            hidden.len().is_multiple_of(HIDDEN),
            "hidden has {} elements, not a multiple of {HIDDEN}",
            hidden.len()
        );
        let rows = hidden.len() / HIDDEN;
        ensure!(
            logits.len() == rows * EXPERTS,
            "logits has {} elements, expected {rows} x {EXPERTS}",
            logits.len()
        );
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(rows).context("too many rows")?,
                EXPERTS as u32,
                1,
            ),
            block_dim: (LOGITS_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let truncate_tf32 = i32::from(matmul == RouterMatmul::Tf32);
        let mut launch = stream.launch_builder(&self.router_logits_2304);
        launch
            .arg(hidden)
            .arg(weight)
            .arg(logits)
            .arg(&truncate_tf32);
        // SAFETY: the arguments match `router_logits_2304(const bf16*, const float*, float*,
        // int)`, and the checks above keep every row inside `hidden` and `logits`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Top-8 experts of each row of `logits` (`[rows, 128]`) by sigmoid score plus `bias`
    /// (`[128]`), into `indices` and `weights` (both `[rows, 8]`). The weights are the
    /// unbiased scores over their sum plus 1e-20, times `route_scale`. Experts come in
    /// MLX's order: highest biased score first, ties to the lower index.
    pub fn expert_routing(
        &self,
        stream: &CudaStream,
        logits: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        indices: &mut CudaSlice<i32>,
        weights: &mut CudaSlice<f32>,
        route_scale: f32,
    ) -> anyhow::Result<()> {
        self.expert_routing_pruned(
            stream,
            logits,
            bias,
            indices,
            weights,
            route_scale,
            MoePruning::default(),
        )
    }

    /// [`Self::expert_routing`] with optional dynamic top-k and probability-threshold
    /// pruning.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn expert_routing_pruned(
        &self,
        stream: &CudaStream,
        logits: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        indices: &mut CudaSlice<i32>,
        weights: &mut CudaSlice<f32>,
        route_scale: f32,
        pruning: MoePruning,
    ) -> anyhow::Result<()> {
        let (max_k, min_k, threshold) = pruning.kernel_args()?;
        ensure!(
            bias.len() == EXPERTS,
            "bias has {} elements, expected {EXPERTS}",
            bias.len()
        );
        ensure!(
            logits.len().is_multiple_of(EXPERTS),
            "logits has {} elements, not a multiple of {EXPERTS}",
            logits.len()
        );
        let rows = logits.len() / EXPERTS;
        ensure!(
            indices.len() == rows * TOP_K && weights.len() == rows * TOP_K,
            "indices has {} and weights {} elements, expected {rows} x {TOP_K}",
            indices.len(),
            weights.len()
        );
        let rows = i32::try_from(rows).context("too many rows")?;
        let rows_per_block = ROUTING_THREADS / 32;
        let config = LaunchConfig {
            grid_dim: (rows.unsigned_abs().div_ceil(rows_per_block), 1, 1),
            block_dim: (ROUTING_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.expert_routing_128_top8);
        launch
            .arg(logits)
            .arg(bias)
            .arg(indices)
            .arg(weights)
            .arg(&rows)
            .arg(&route_scale)
            .arg(&max_k)
            .arg(&min_k)
            .arg(&threshold);
        // SAFETY: the arguments match `expert_routing_128_top8(const float*, const float*,
        // int32_t*, float*, int, float, int, int, float)`, the kernel skips warps past
        // `rows`, and the checks above keep every row inside the buffers.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Depth attention: for each row, the `nq` F32 `queries` (`[nq, 2304]`) attend over
    /// the `nb + 1` candidate states `blocks` (`[nb, rows, 2304]`) and `partial`
    /// (`[rows, 2304]`) with scores scaled by `2304^-0.5`, into `out` (`[rows, nq, 2304]`).
    pub fn depth_attention(
        &self,
        stream: &CudaStream,
        blocks: &impl DeviceIn<f32>,
        partial: &CudaSlice<f32>,
        queries: &CudaSlice<f32>,
        out: &mut impl DeviceOut<f32>,
    ) -> anyhow::Result<()> {
        let (blocks, mut out) = (blocks.view(), out.view_mut());
        ensure!(
            !partial.is_empty() && partial.len().is_multiple_of(HIDDEN),
            "partial has {} elements, not a positive multiple of {HIDDEN}",
            partial.len()
        );
        ensure!(
            blocks.len().is_multiple_of(partial.len()),
            "blocks has {} elements, not a multiple of partial's {}",
            blocks.len(),
            partial.len()
        );
        ensure!(
            !queries.is_empty() && queries.len().is_multiple_of(HIDDEN),
            "queries has {} elements, not a positive multiple of {HIDDEN}",
            queries.len()
        );
        let rows = partial.len() / HIDDEN;
        let nb = blocks.len() / partial.len();
        let nq = queries.len() / HIDDEN;
        ensure!(
            out.len() == rows * nq * HIDDEN,
            "out has {} elements, expected {rows} x {nq} x {HIDDEN}",
            out.len()
        );
        // Metal computes the scale in F64 and passes it as F32; 1 / 48 rounds the same way.
        let scale = (HIDDEN as f32).sqrt().recip();
        let shared_floats = (nb + 1) * nq * (DEPTH_THREADS as usize / 32 + 1);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: (DEPTH_THREADS, 1, 1),
            shared_mem_bytes: u32::try_from(shared_floats * size_of::<f32>())
                .context("too many candidates")?,
        };
        let nb = i32::try_from(nb).context("too many blocks")?;
        let nq = i32::try_from(nq).context("too many queries")?;
        let mut launch = stream.launch_builder(&self.depth_attention_2304);
        launch
            .arg(&blocks)
            .arg(partial)
            .arg(queries)
            .arg(&mut out)
            .arg(&nb)
            .arg(&nq)
            .arg(&scale);
        // SAFETY: the arguments match `depth_attention_2304(const float*, const float*,
        // const float*, float*, int, int, float)`, the checks above keep every row inside the
        // buffers, and the dynamic shared memory holds the (nb + 1) * nq * 10 floats the
        // kernel uses.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Depth attention for one row, as [`Routing::depth_attention`] with `rows = 1`, with the
    /// RMS norms around it fused in. Sums in the order of the separate kernels; the softmax
    /// uses CUDA's `exp2f`.
    ///
    /// The `nb` rows of `blocks` (`[nb, 2304]`) and the `partial` row are the candidates;
    /// `normed_partial` receives the partial as used. `out` receives the `nq` output rows
    /// (`[nq, 2304]`). Each `dests[k]` receives output row `min(k, nq - 1)` after an RMS
    /// norm, rounded to BF16.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn depth_attention_fused(
        &self,
        stream: &CudaStream,
        blocks: &impl DeviceIn<f32>,
        partial: Partial<'_>,
        mut normed_partial: Option<CudaViewMut<'_, f32>>,
        queries: &CudaSlice<f32>,
        mut out: Option<CudaViewMut<'_, f32>>,
        mut dests: [Option<NormDest<'_>>; FUSED_DESTS],
        eps: f32,
    ) -> anyhow::Result<()> {
        let blocks = blocks.view();
        ensure!(
            blocks.len().is_multiple_of(HIDDEN),
            "blocks has {} elements, not a multiple of {HIDDEN}",
            blocks.len()
        );
        let nb = blocks.len() / HIDDEN;
        ensure!(
            nb < FUSED_MAX_CANDIDATES,
            "{nb} blocks; at most {} fit",
            FUSED_MAX_CANDIDATES - 1
        );
        ensure!(
            !queries.is_empty() && queries.len().is_multiple_of(HIDDEN),
            "queries has {} elements, not a positive multiple of {HIDDEN}",
            queries.len()
        );
        let nq = queries.len() / HIDDEN;
        ensure!(
            nq <= FUSED_MAX_QUERIES,
            "{nq} queries; at most {FUSED_MAX_QUERIES} fit"
        );
        let (partial_f32, partial_bf16, normalize) = match partial {
            Partial::Normalized(x) => (Some(x), None, 0i32),
            Partial::F32(x) => (Some(x), None, 1),
            Partial::Bf16(x) => (None, Some(x), 1),
        };
        let partial_len = partial_f32.as_ref().map_or_else(
            || partial_bf16.as_ref().map_or(0, CudaView::len),
            CudaView::len,
        );
        ensure!(
            partial_len == HIDDEN,
            "partial has {partial_len} elements, expected {HIDDEN}"
        );
        ensure!(
            normed_partial.as_ref().is_none_or(|x| x.len() == HIDDEN),
            "normed_partial must hold {HIDDEN} elements"
        );
        ensure!(
            out.as_ref().is_none_or(|x| x.len() == nq * HIDDEN),
            "out must hold {nq} x {HIDDEN} elements"
        );
        for dest in dests.iter().flatten() {
            ensure!(
                dest.out.len() == HIDDEN && dest.weight.is_none_or(|w| w.len() == HIDDEN),
                "destinations and norm weights must hold {HIDDEN} elements"
            );
        }

        let scale = (HIDDEN as f32).sqrt().recip();
        let mut args = KernelArgs::default();
        args.ptr(&blocks, stream)
            .i32(i32::try_from(nb)?)
            .opt(partial_f32.as_ref(), stream)
            .opt(partial_bf16.as_ref(), stream)
            .i32(normalize)
            .opt_mut(normed_partial.as_mut(), stream)
            .ptr(queries, stream)
            .i32(i32::try_from(nq)?)
            .opt_mut(out.as_mut(), stream);
        for dest in &mut dests {
            args.opt_mut(dest.as_mut().map(|d| &mut d.out), stream);
        }
        for dest in &dests {
            args.opt(dest.as_ref().and_then(|d| d.weight), stream);
        }
        args.f32(eps).f32(scale);
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (DEPTH_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `depth_attention_fused_2304`'s signature, every non-null
        // pointer comes from a live buffer of the length checked above, the kernel reads `nb`
        // block rows and `nq` query rows, and it reads only the queries before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.depth_attention_fused_2304, config) }
    }

    /// The MoE input norm and router for one row: `x_out = bf16(rms_norm(x) * norm_weight)`,
    /// router logits from `x_out`, then the top-8 routing of [`Routing::expert_routing`].
    /// The norm and logits sum in the order of the separate kernels; the scores use CUDA's
    /// `expf` rather than an emulation of Apple's exp.
    ///
    /// `counter` must hold 0; the kernel leaves it at 0.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn moe_router(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<f32>,
        norm_weight: &CudaSlice<bf16>,
        eps: f32,
        router: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        x_out: &mut CudaSlice<bf16>,
        logits: &mut CudaSlice<f32>,
        indices: &mut CudaSlice<i32>,
        weights: &mut CudaSlice<f32>,
        counter: &mut CudaSlice<u32>,
        route_scale: f32,
    ) -> anyhow::Result<()> {
        self.moe_router_pruned(
            stream,
            x,
            norm_weight,
            eps,
            router,
            bias,
            x_out,
            logits,
            indices,
            weights,
            counter,
            route_scale,
            MoePruning::default(),
        )
    }

    /// [`Self::moe_router`] with optional dynamic top-k and probability-threshold pruning.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn moe_router_pruned(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<f32>,
        norm_weight: &CudaSlice<bf16>,
        eps: f32,
        router: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        x_out: &mut CudaSlice<bf16>,
        logits: &mut CudaSlice<f32>,
        indices: &mut CudaSlice<i32>,
        weights: &mut CudaSlice<f32>,
        counter: &mut CudaSlice<u32>,
        route_scale: f32,
        pruning: MoePruning,
    ) -> anyhow::Result<()> {
        let (max_k, min_k, threshold) = pruning.kernel_args()?;
        let x = x.view();
        ensure!(
            x.len() == HIDDEN,
            "x has {} elements, expected {HIDDEN}",
            x.len()
        );
        ensure!(
            norm_weight.len() == HIDDEN && x_out.len() == HIDDEN,
            "norm weight or x_out length"
        );
        ensure!(
            router.len() == EXPERTS * HIDDEN,
            "router has {} elements",
            router.len()
        );
        ensure!(
            bias.len() == EXPERTS && logits.len() == EXPERTS,
            "bias or logits length"
        );
        ensure!(
            indices.len() == TOP_K && weights.len() == TOP_K,
            "indices or weights length"
        );
        ensure!(!counter.is_empty(), "counter is empty");
        let config = LaunchConfig {
            grid_dim: (ROUTER_BLOCKS, 1, 1),
            block_dim: (ROUTER_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .ptr(norm_weight, stream)
            .f32(eps)
            .ptr(router, stream)
            .ptr(bias, stream)
            .ptr_mut(x_out, stream)
            .ptr_mut(logits, stream)
            .ptr_mut(indices, stream)
            .ptr_mut(weights, stream)
            .ptr_mut(counter, stream)
            .f32(route_scale)
            .i32(max_k)
            .i32(min_k)
            .f32(threshold);
        // SAFETY: the arguments match `moe_router_2304(const float*, const bf16*, float,
        // const float*, const float*, bf16*, float*, int32_t*, float*, unsigned*, float,
        // int, int, float)`, every buffer has the length the kernel reads or writes, and it
        // reads only weights before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.moe_router_2304, config) }
    }

    /// Depth attention for decode, as [`Routing::depth_attention_fused`], summed in the
    /// order of `depth_attention_step_2304` rather than the separate kernels'. `d.batch`
    /// sets of queries attend over the same candidates in one launch. A
    /// [`StepPartial::ConvResidual`] partial needs a batch of 1.
    pub fn depth_attention_step(
        &self,
        stream: &CudaStream,
        d: DepthStep<'_>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let DepthStep {
            blocks,
            partial,
            mut normed_partial,
            queries,
            batch,
            mut out,
            mut dests,
            dst_stride,
        } = d;
        ensure!(
            blocks.len().is_multiple_of(HIDDEN),
            "blocks has {} elements, not a multiple of {HIDDEN}",
            blocks.len()
        );
        let nb = blocks.len() / HIDDEN;
        ensure!(
            nb < FUSED_MAX_CANDIDATES,
            "{nb} blocks; at most {} fit",
            FUSED_MAX_CANDIDATES - 1
        );
        ensure!(batch > 0, "empty batch");
        ensure!(
            !queries.is_empty() && queries.len().is_multiple_of(batch * HIDDEN),
            "queries has {} elements, not a positive multiple of {batch} x {HIDDEN}",
            queries.len()
        );
        let nq = queries.len() / (batch * HIDDEN);
        ensure!(
            nq <= FUSED_MAX_QUERIES,
            "{nq} queries; at most {FUSED_MAX_QUERIES} fit"
        );
        ensure!(
            normed_partial.as_ref().is_none_or(|x| x.len() == HIDDEN),
            "normed_partial must hold {HIDDEN} elements"
        );
        ensure!(
            out.as_ref().is_none_or(|x| x.len() == batch * nq * HIDDEN),
            "out must hold {batch} x {nq} x {HIDDEN} elements"
        );
        ensure!(
            batch == 1 || dst_stride >= HIDDEN,
            "dst_stride is below {HIDDEN}"
        );
        let dest_len = (batch - 1) * dst_stride + HIDDEN;
        for dest in dests.iter().flatten() {
            ensure!(
                dest.out.len() >= dest_len && dest.weight.is_none_or(|w| w.len() == HIDDEN),
                "destinations must reach {dest_len} elements and norm weights hold {HIDDEN}"
            );
        }

        let mut args = KernelArgs::default();
        args.ptr(&blocks, stream).i32(i32::try_from(nb)?);
        push_step_partial(&mut args, partial, batch, stream)?;
        args.opt_mut(normed_partial.as_mut(), stream)
            .ptr(&queries, stream)
            .i32(i32::try_from(nq)?)
            .i32(i32::try_from(nq * HIDDEN)?)
            .opt_mut(out.as_mut(), stream)
            .i32(i32::try_from(nq * HIDDEN)?);
        for dest in &mut dests {
            args.opt_mut(dest.as_mut().map(|d| &mut d.out), stream);
        }
        args.i32(i32::try_from(dst_stride).context("dst_stride too large")?);
        for dest in &dests {
            args.opt(dest.as_ref().and_then(|d| d.weight), stream);
        }
        args.f32(eps).f32((HIDDEN as f32).sqrt().recip());
        let config = LaunchConfig {
            grid_dim: (u32::try_from(batch).context("batch too large")?, 1, 1),
            block_dim: (DEPTH_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `depth_attention_step_2304`'s signature, every non-null
        // pointer comes from a live buffer of the length checked above, the grid has one
        // block per query set, only block 0 touches the conv state, and the kernel reads
        // only weights (queries, norm weights, conv taps) before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.depth_attention_step_2304, config) }
    }

    /// Depth attention for the rows of a chunk, summed in the order of
    /// `depth_attention_step_2304` (see [`Routing::depth_attention_step`]), so results
    /// differ from [`Routing::depth_attention_fused`] by F32 rounding. A
    /// [`RowsPartial::ConvResidual`] partial also computes the post-attention conv and
    /// residual of each row.
    pub fn depth_attention_rows(
        &self,
        stream: &CudaStream,
        d: DepthRows<'_>,
        eps: f32,
    ) -> anyhow::Result<()> {
        self.depth_attention_rows_batch(stream, d, RowsBatch::default(), eps)
    }

    /// [`Routing::depth_attention_rows`] for `b.count` sets of queries over the same
    /// candidates in one launch: `d.queries` holds the sets one after another, and set `i`
    /// writes its weights `i * b.probs_stride` and its destinations `i * b.dst_stride`
    /// elements after set 0's. Every set sums as a launch of its own would. With more than
    /// one set, `out` and `normed_partial` must be `None` and the partial not a
    /// [`RowsPartial::ConvResidual`].
    pub fn depth_attention_rows_batch(
        &self,
        stream: &CudaStream,
        d: DepthRows<'_>,
        b: RowsBatch,
        eps: f32,
    ) -> anyhow::Result<()> {
        let DepthRows {
            blocks,
            partial,
            mut normed_partial,
            queries,
            mut out,
            mut probs,
            mut dests,
            dst_stride,
            rows,
        } = d;
        ensure!(rows > 0, "no rows");
        let block_len = rows * HIDDEN;
        ensure!(
            blocks.len().is_multiple_of(block_len),
            "blocks has {} elements, not a multiple of {rows} x {HIDDEN}",
            blocks.len()
        );
        let nb = blocks.len() / block_len;
        ensure!(
            (1..FUSED_MAX_CANDIDATES).contains(&nb),
            "{nb} blocks; 1 to {} fit",
            FUSED_MAX_CANDIDATES - 1
        );
        let count = b.count.max(1);
        ensure!(
            queries.len().is_multiple_of(count * HIDDEN),
            "queries has {} elements, not {count} sets of whole rows of {HIDDEN}",
            queries.len()
        );
        ensure!(
            count == 1
                || (out.is_none()
                    && normed_partial.is_none()
                    && !matches!(
                        partial,
                        RowsPartial::ConvResidual { .. } | RowsPartial::ConvResidualBatch { .. }
                    )),
            "a batch of sets writes only weights and destinations"
        );
        let nq = queries.len() / (count * HIDDEN);
        let q_index = ROWS_QUERY_COUNTS
            .iter()
            .position(|&n| n == nq)
            .with_context(|| format!("no rows kernel for {nq} queries"))?;
        ensure!(
            normed_partial.as_ref().is_none_or(|x| x.len() == block_len),
            "normed_partial must hold {rows} x {HIDDEN} elements"
        );
        ensure!(
            out.as_ref().is_none_or(|x| x.len() == rows * nq * HIDDEN),
            "out must hold {rows} x {nq} x {HIDDEN} elements"
        );
        let probs_len = rows * nq * (nb + 1);
        ensure!(
            probs.as_ref().is_none_or(|x| if count == 1 {
                x.len() == probs_len
            } else {
                b.probs_stride >= probs_len && x.len() >= (count - 1) * b.probs_stride + probs_len
            }),
            "probs must hold {count} sets of {rows} x {nq} x {} elements",
            nb + 1
        );
        ensure!(dst_stride >= HIDDEN, "dst_stride is below {HIDDEN}");
        let dest_len = (count - 1) * b.dst_stride + (rows - 1) * dst_stride + HIDDEN;
        for dest in dests.iter().flatten() {
            ensure!(
                dest.out.len() >= dest_len && dest.weight.is_none_or(|w| w.len() == HIDDEN),
                "destinations must reach {dest_len} elements and norm weights hold {HIDDEN}"
            );
        }

        let mut args = KernelArgs::default();
        args.ptr(&blocks, stream)
            .i32(i32::try_from(block_len).context("too many rows")?);
        push_rows_partial(&mut args, partial, block_len, stream)?;
        args.opt_mut(normed_partial.as_mut(), stream)
            .ptr(queries, stream)
            .opt_mut(out.as_mut(), stream)
            .opt_mut(probs.as_mut(), stream);
        for dest in &mut dests {
            args.opt_mut(dest.as_mut().map(|d| &mut d.out), stream);
        }
        args.i32(i32::try_from(dst_stride).context("dst_stride too large")?);
        for dest in &dests {
            args.opt(dest.as_ref().and_then(|d| d.weight), stream);
        }
        let rows_u32 = u32::try_from(rows).context("too many rows")?;
        args.f32(eps)
            .f32((HIDDEN as f32).sqrt().recip())
            .i32(i32::try_from(rows)?)
            .i32(i32::try_from(b.probs_stride).context("probs stride too large")?)
            .i32(i32::try_from(b.dst_stride).context("destination stride too large")?);
        let (function, resident) = &self.depth_attention_rows_2304[q_index][nb - 1];
        let config = LaunchConfig {
            grid_dim: (rows_u32.min(*resident), u32::try_from(count)?, 1),
            block_dim: (DEPTH_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `depth_attention_rows_2304_<nb>_<nq>`'s signature,
        // every non-null pointer comes from a live buffer that holds the rows of every set
        // checked above, the blocks loop over the rows and the grid's y dimension covers
        // the sets, and the kernel reads only weights (queries, norm weights, conv taps)
        // before `pdl_wait`.
        unsafe { args.launch_pdl(stream, function, config) }
    }

    /// [`Routing::depth_attention_rows`] with fused MXFP8 or NVFP4 activation quantization
    /// directly in the RMSNorm epilogue (`nq` in `[1, 3]`). Every non-`None` destination
    /// writes quantized activations from registers (and BF16 only when `dest.out` is `Some`),
    /// matching [`Self::depth_attention_rows`] followed by the standalone quantizer bit for
    /// bit.
    #[expect(
        clippy::too_many_lines,
        reason = "validates buffers and mirrors the kernel signature"
    )]
    pub fn depth_attention_rows_quant(
        &self,
        stream: &CudaStream,
        d: DepthRowsQuant<'_>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let DepthRowsQuant {
            blocks,
            partial,
            mut normed_partial,
            queries,
            mut out,
            mut probs,
            mut dests,
            dst_stride,
            rows,
        } = d;
        ensure!(rows > 0, "no rows");
        let block_len = rows * HIDDEN;
        ensure!(
            blocks.len().is_multiple_of(block_len),
            "blocks has {} elements, not a multiple of {rows} x {HIDDEN}",
            blocks.len()
        );
        let nb = blocks.len() / block_len;
        ensure!(
            (1..FUSED_MAX_CANDIDATES).contains(&nb),
            "{nb} blocks; 1 to {} fit",
            FUSED_MAX_CANDIDATES - 1
        );
        ensure!(
            !queries.is_empty() && queries.len().is_multiple_of(HIDDEN),
            "queries has {} elements, not whole rows of {HIDDEN}",
            queries.len()
        );
        let nq = queries.len() / HIDDEN;
        let q_index = ROWS_QUANT_QUERY_COUNTS
            .iter()
            .position(|&n| n == nq)
            .with_context(|| format!("no quant rows kernel for {nq} queries"))?;
        ensure!(
            normed_partial.as_ref().is_none_or(|x| x.len() == block_len),
            "normed_partial must hold {rows} x {HIDDEN} elements"
        );
        ensure!(
            out.as_ref().is_none_or(|x| x.len() == rows * nq * HIDDEN),
            "out must hold {rows} x {nq} x {HIDDEN} elements"
        );
        let probs_len = rows * nq * (nb + 1);
        ensure!(
            probs.as_ref().is_none_or(|x| x.len() == probs_len),
            "probs must hold {rows} x {nq} x {} elements",
            nb + 1
        );
        let dest_len = (rows - 1) * dst_stride + HIDDEN;
        let mut q_mode = 0i32;
        for dest in dests.iter().flatten() {
            if let Some(bf16_out) = &dest.out {
                ensure!(dst_stride >= HIDDEN, "dst_stride is below {HIDDEN}");
                ensure!(
                    bf16_out.len() >= dest_len,
                    "destination BF16 buffer must reach {dest_len} elements"
                );
            }
            ensure!(
                dest.weight.is_none_or(|w| w.len() == HIDDEN),
                "norm weights must hold {HIDDEN} elements"
            );
            dest.quant.check(rows, HIDDEN)?;
            let mode = dest.quant.format.mode();
            if q_mode == 0 {
                q_mode = mode;
            } else {
                ensure!(
                    q_mode == mode,
                    "all depth_attention_rows_quant destinations must share one QuantFormat"
                );
            }
        }

        let mut args = KernelArgs::default();
        args.ptr(&blocks, stream)
            .i32(i32::try_from(block_len).context("too many rows")?);
        push_rows_partial(&mut args, partial, block_len, stream)?;
        args.opt_mut(normed_partial.as_mut(), stream)
            .ptr(queries, stream)
            .opt_mut(out.as_mut(), stream)
            .opt_mut(probs.as_mut(), stream);
        for dest in &mut dests {
            args.opt_mut(dest.as_mut().and_then(|d| d.out.as_mut()), stream);
        }
        // dst3 is always null for the quant variant.
        args.u64(0)
            .i32(i32::try_from(dst_stride).context("dst_stride too large")?);
        for dest in &dests {
            args.opt(dest.as_ref().and_then(|d| d.weight), stream);
        }
        // weight3 is always null for the quant variant.
        args.u64(0);
        let rows_u32 = u32::try_from(rows).context("too many rows")?;
        args.f32(eps)
            .f32((HIDDEN as f32).sqrt().recip())
            .i32(i32::try_from(rows)?)
            .i32(0)
            .i32(0);
        for dest in &mut dests {
            args.opt_mut(dest.as_mut().map(|d| &mut d.quant.values), stream);
        }
        for dest in &mut dests {
            args.opt_mut(dest.as_mut().map(|d| &mut d.quant.scales), stream);
        }
        for dest in &mut dests {
            args.opt_mut(dest.as_mut().and_then(|d| d.quant.outer.as_mut()), stream);
        }
        args.i32(q_mode);
        let (function, resident) = &self.depth_attention_rows_quant_2304[q_index][nb - 1];
        let config = LaunchConfig {
            grid_dim: (rows_u32.min(*resident), 1, 1),
            block_dim: (DEPTH_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `depth_attention_rows_quant_2304_<nb>_<nq>`'s signature,
        // every non-null pointer comes from a live buffer sized for `rows` as checked above,
        // and the kernel reads only read-only weights (queries, conv taps) before `pdl_wait`.
        unsafe { args.launch_pdl(stream, function, config) }
    }

    /// The router for one BF16 row `x` that is already normalized: logits from `x`, then the
    /// top-8 routing of [`Routing::moe_router`], in one launch of one 16-block cluster. The
    /// logits sum in the order of [`Self::moe_router_step_rows`].
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn moe_router_step(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        router: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        logits: &mut CudaSlice<f32>,
        indices: &mut CudaSlice<i32>,
        weights: &mut CudaSlice<f32>,
        route_scale: f32,
    ) -> anyhow::Result<()> {
        self.moe_router_step_pruned(
            stream,
            x,
            router,
            bias,
            logits,
            indices,
            weights,
            route_scale,
            MoePruning::default(),
        )
    }

    /// [`Self::moe_router_step`] with optional dynamic top-k and probability-threshold
    /// pruning.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn moe_router_step_pruned(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        router: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        logits: &mut CudaSlice<f32>,
        indices: &mut CudaSlice<i32>,
        weights: &mut CudaSlice<f32>,
        route_scale: f32,
        pruning: MoePruning,
    ) -> anyhow::Result<()> {
        let (max_k, min_k, threshold) = pruning.kernel_args()?;
        ensure!(
            x.len() == HIDDEN,
            "x has {} elements, expected {HIDDEN}",
            x.len()
        );
        ensure!(
            router.len() == EXPERTS * HIDDEN,
            "router has {} elements",
            router.len()
        );
        ensure!(
            bias.len() == EXPERTS && logits.len() == EXPERTS,
            "bias or logits length"
        );
        ensure!(
            indices.len() == TOP_K && weights.len() == TOP_K,
            "indices or weights length"
        );
        let config = LaunchConfig {
            grid_dim: (STEP_ROUTER_CLUSTER, 1, 1),
            block_dim: (EXPERTS as u32 / STEP_ROUTER_CLUSTER * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(x, stream)
            .ptr(router, stream)
            .ptr(bias, stream)
            .ptr_mut(logits, stream)
            .ptr_mut(indices, stream)
            .ptr_mut(weights, stream)
            .f32(route_scale)
            .i32(max_k)
            .i32(min_k)
            .f32(threshold);
        // SAFETY: the arguments match `moe_router_step_2304(const bf16*, const float*, const
        // float*, float*, int32_t*, float*, float, int, int, float)`, every buffer has the
        // length the kernel reads or writes, the grid is the kernel's one cluster of 16
        // blocks of one warp per expert, and the kernel reads only the router and the bias
        // before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.moe_router_step_2304, config) }
    }

    /// [`Self::moe_router_step`] for `rows` rows of a step, 1 to [`STEP_ROUTER_ROWS`]: `x`
    /// `[rows, 2304]`, `logits` `[rows, 128]`, `indices` and `weights` `[rows, 8]`. Each
    /// row's results equal a one-row [`Self::moe_router_step`] call's bit for bit. Each
    /// cluster reads the router weights once for its 8 rows.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn moe_router_step_rows(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        router: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        logits: &mut impl DeviceOut<f32>,
        indices: &mut impl DeviceOut<i32>,
        weights: &mut impl DeviceOut<f32>,
        route_scale: f32,
        rows: usize,
    ) -> anyhow::Result<()> {
        self.moe_router_step_rows_pruned(
            stream,
            x,
            router,
            bias,
            logits,
            indices,
            weights,
            route_scale,
            rows,
            MoePruning::default(),
        )
    }

    /// [`Self::moe_router_step_rows`] with optional dynamic top-k and probability-threshold
    /// pruning.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn moe_router_step_rows_pruned(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        router: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        logits: &mut impl DeviceOut<f32>,
        indices: &mut impl DeviceOut<i32>,
        weights: &mut impl DeviceOut<f32>,
        route_scale: f32,
        rows: usize,
        pruning: MoePruning,
    ) -> anyhow::Result<()> {
        let (max_k, min_k, threshold) = pruning.kernel_args()?;
        let x = x.view();
        let (mut logits, mut indices, mut weights) =
            (logits.view_mut(), indices.view_mut(), weights.view_mut());
        ensure!(
            (1..=STEP_ROUTER_ROWS).contains(&rows),
            "{rows} rows, expected 1 to {STEP_ROUTER_ROWS}"
        );
        ensure!(
            x.len() == rows * HIDDEN,
            "x must hold {rows} x {HIDDEN} elements"
        );
        ensure!(
            router.len() == EXPERTS * HIDDEN,
            "router has {} elements",
            router.len()
        );
        ensure!(bias.len() == EXPERTS, "bias length");
        ensure!(logits.len() == rows * EXPERTS, "logits length");
        ensure!(
            indices.len() == rows * TOP_K && weights.len() == rows * TOP_K,
            "indices or weights length"
        );
        let config = LaunchConfig {
            grid_dim: (
                STEP_ROUTER_CLUSTER,
                u32::try_from(rows.div_ceil(STEP_ROUTER_GROUP))?,
                1,
            ),
            block_dim: (STEP_ROUTER_ROWS_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .i32(i32::try_from(rows)?)
            .ptr(router, stream)
            .ptr(bias, stream)
            .ptr_mut(&mut logits, stream)
            .ptr_mut(&mut indices, stream)
            .ptr_mut(&mut weights, stream)
            .f32(route_scale)
            .i32(max_k)
            .i32(min_k)
            .f32(threshold);
        // SAFETY: the arguments match `moe_router_step_rows_2304(const bf16*, int, const
        // float*, const float*, float*, int32_t*, float*, float, int, int, float)`, every
        // buffer has the length the kernel reads or writes for `rows` rows, the grid is one
        // of the kernel's clusters of 16 blocks of 288 threads per 8 rows, and the kernel
        // reads only the router and the bias before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.moe_router_step_rows_2304, config) }
    }

    /// The router for each of the `rows` rows of a chunk from rows `x` (`[rows, 2304]`) that
    /// are already normalized, as [`Routing::moe_router`] writes `x_out`: `logits`
    /// `[rows, 128]`, `indices` and `weights` `[rows, 8]`, with `scratch` for at least
    /// `rows` rows. The logits sum in their own order (a tiled matrix product split along
    /// the inner dimension), so they differ from a single-row call's by F32 rounding; the
    /// routing is [`Routing::moe_router`]'s.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signatures")]
    pub fn moe_router_rows(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        router: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        logits: &mut impl DeviceOut<f32>,
        indices: &mut impl DeviceOut<i32>,
        weights: &mut impl DeviceOut<f32>,
        scratch: &mut RouterRowsScratch,
        route_scale: f32,
        rows: usize,
    ) -> anyhow::Result<()> {
        self.moe_router_rows_pruned(
            stream,
            x,
            router,
            bias,
            logits,
            indices,
            weights,
            scratch,
            route_scale,
            rows,
            MoePruning::default(),
        )
    }

    /// [`Self::moe_router_rows`] with optional dynamic top-k and probability-threshold
    /// pruning.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signatures")]
    pub fn moe_router_rows_pruned(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        router: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        logits: &mut impl DeviceOut<f32>,
        indices: &mut impl DeviceOut<i32>,
        weights: &mut impl DeviceOut<f32>,
        scratch: &mut RouterRowsScratch,
        route_scale: f32,
        rows: usize,
        pruning: MoePruning,
    ) -> anyhow::Result<()> {
        let (max_k, min_k, threshold) = pruning.kernel_args()?;
        let x = x.view();
        let (mut logits, mut indices, mut weights) =
            (logits.view_mut(), indices.view_mut(), weights.view_mut());
        ensure!(rows > 0, "no rows");
        ensure!(
            x.len() == rows * HIDDEN,
            "x must hold {rows} x {HIDDEN} elements"
        );
        ensure!(
            router.len() == EXPERTS * HIDDEN,
            "router has {} elements",
            router.len()
        );
        ensure!(bias.len() == EXPERTS, "bias length");
        ensure!(logits.len() == rows * EXPERTS, "logits length");
        ensure!(
            indices.len() == rows * TOP_K && weights.len() == rows * TOP_K,
            "indices or weights length"
        );
        ensure!(
            rows <= scratch.rows,
            "scratch holds {} rows, not {rows}",
            scratch.rows
        );
        let mut partials = scratch.partials.slice_mut(..ROUTER_SPLITS * rows * EXPERTS);
        let rows_i32 = i32::try_from(rows).context("too many rows")?;

        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .ptr(router, stream)
            .ptr_mut(&mut partials, stream)
            .i32(rows_i32)
            .i32(i32::try_from(HIDDEN / ROUTER_SPLITS)?);
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(rows.div_ceil(ROUTER_TILE_ROWS))?,
                u32::try_from(ROUTER_SPLITS)?,
                1,
            ),
            block_dim: (ROUTER_TILE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `moe_router_logits_rows_2304(const bf16*, const float*,
        // float*, int, int)`, x holds `rows` rows and partials 9 x `rows` x 128 floats, the
        // grid covers the rows in tiles of 64 (the kernel skips rows past `rows`) and the
        // columns in 9 slices of 256, and the kernel reads only the router before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.moe_router_logits_rows_2304, config) }?;

        let mut args = KernelArgs::default();
        args.ptr(&partials, stream)
            .i32(i32::try_from(ROUTER_SPLITS)?)
            .ptr(bias, stream)
            .ptr_mut(&mut logits, stream)
            .ptr_mut(&mut indices, stream)
            .ptr_mut(&mut weights, stream)
            .i32(rows_i32)
            .f32(route_scale)
            .i32(max_k)
            .i32(min_k)
            .f32(threshold);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: (u32::try_from(EXPERTS)?, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `moe_router_route_rows_2304(const float*, int, const
        // float*, float*, int*, float*, int, float, int, int, float)`, every buffer holds
        // `rows` rows as checked above, the grid has one block of 128 threads per row, and
        // the kernel reads nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.moe_router_route_rows_2304, config) }
    }
}

/// The partial's arguments of `depth_attention_rows_2304_<nb>_<nq>`: its F32 and BF16
/// pointers, its mode, the conv's weight, state, hidden state and residual, and the batched
/// conv's state table and rows per sequence, for partials of `len` elements.
#[expect(
    clippy::too_many_lines,
    reason = "pushes kernel arguments for each RowsPartial variant"
)]
fn push_rows_partial(
    args: &mut KernelArgs,
    partial: RowsPartial<'_>,
    len: usize,
    stream: &CudaStream,
) -> anyhow::Result<()> {
    let check = |n: usize| -> anyhow::Result<()> {
        ensure!(n == len, "the partial must hold {len} elements, not {n}");
        Ok(())
    };
    match partial {
        RowsPartial::Normalized(x) => {
            check(x.len())?;
            args.ptr(&x, stream)
                .u64(0)
                .i32(0)
                .u64(0)
                .u64(0)
                .u64(0)
                .u64(0)
                .u64(0)
                .i32(0);
        }
        RowsPartial::F32(x) => {
            check(x.len())?;
            args.ptr(&x, stream)
                .u64(0)
                .i32(1)
                .u64(0)
                .u64(0)
                .u64(0)
                .u64(0)
                .u64(0)
                .i32(0);
        }
        RowsPartial::Bf16(x) => {
            check(x.len())?;
            args.u64(0)
                .ptr(&x, stream)
                .i32(2)
                .u64(0)
                .u64(0)
                .u64(0)
                .u64(0)
                .u64(0)
                .i32(0);
        }
        RowsPartial::ConvResidual {
            x,
            weight,
            state,
            hidden,
            mut residual,
        } => {
            check(x.len())?;
            check(residual.len())?;
            ensure!(
                weight.len() == 4 * HIDDEN
                    && state.len() == 3 * HIDDEN
                    && hidden.as_ref().is_none_or(|h| h.len() == len),
                "conv partial lengths"
            );
            args.u64(0)
                .ptr(&x, stream)
                .i32(3)
                .ptr(weight, stream)
                .ptr(&state, stream)
                .opt(hidden.as_ref(), stream)
                .ptr_mut(&mut residual, stream)
                .u64(0)
                .i32(0);
        }
        RowsPartial::ConvResidualBatch {
            x,
            weight,
            states,
            rows_per_seq,
            hidden,
            mut residual,
        } => {
            check(x.len())?;
            check(residual.len())?;
            let rows = len / HIDDEN;
            ensure!(
                (1..=crate::ops::spec::VERIFY_ROWS).contains(&rows_per_seq)
                    && rows.is_multiple_of(rows_per_seq),
                "{rows} rows is not a multiple of rows_per_seq {rows_per_seq}"
            );
            let seqs = rows / rows_per_seq;
            ensure!(
                weight.len() == 4 * HIDDEN
                    && states.len() == seqs * crate::ops::spec::BATCH_STATE_WORDS
                    && hidden.as_ref().is_none_or(|h| h.len() == len),
                "batched conv partial lengths"
            );
            args.u64(0)
                .ptr(&x, stream)
                .i32(4)
                .ptr(weight, stream)
                .u64(0)
                .opt(hidden.as_ref(), stream)
                .ptr_mut(&mut residual, stream)
                .ptr(&states, stream)
                .i32(i32::try_from(rows_per_seq)?);
        }
    }
    Ok(())
}

/// The partial's arguments of `depth_attention_step_2304`: its F32 and BF16 pointers, its
/// mode, and the conv's weight, state, hidden state and residual.
fn push_step_partial(
    args: &mut KernelArgs,
    partial: StepPartial<'_>,
    batch: usize,
    stream: &CudaStream,
) -> anyhow::Result<()> {
    match partial {
        StepPartial::Normalized(x) => {
            ensure!(x.len() == HIDDEN, "partial must hold {HIDDEN} elements");
            args.ptr(&x, stream)
                .u64(0)
                .i32(0)
                .u64(0)
                .u64(0)
                .u64(0)
                .u64(0);
        }
        StepPartial::F32(x) => {
            ensure!(x.len() == HIDDEN, "partial must hold {HIDDEN} elements");
            args.ptr(&x, stream)
                .u64(0)
                .i32(1)
                .u64(0)
                .u64(0)
                .u64(0)
                .u64(0);
        }
        StepPartial::Bf16(x) => {
            ensure!(x.len() == HIDDEN, "partial must hold {HIDDEN} elements");
            args.u64(0)
                .ptr(&x, stream)
                .i32(2)
                .u64(0)
                .u64(0)
                .u64(0)
                .u64(0);
        }
        StepPartial::ConvResidual {
            x,
            weight,
            mut state,
            hidden,
            mut residual,
        } => {
            ensure!(batch == 1, "the conv partial needs a batch of 1");
            ensure!(
                x.len() == HIDDEN
                    && weight.len() == 4 * HIDDEN
                    && state.len() == 3 * HIDDEN
                    && residual.len() == HIDDEN
                    && hidden.as_ref().is_none_or(|h| h.len() == HIDDEN),
                "conv partial lengths"
            );
            args.u64(0)
                .ptr(&x, stream)
                .i32(3)
                .ptr(weight, stream)
                .ptr_mut(&mut state, stream)
                .opt(hidden.as_ref(), stream)
                .ptr_mut(&mut residual, stream);
        }
    }
    Ok(())
}
