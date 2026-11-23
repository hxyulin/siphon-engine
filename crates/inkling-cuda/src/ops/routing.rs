//! MoE expert routing and depth attention (`kernels/routing.cu`).

use std::{mem::size_of, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig,
    PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module};

/// Hidden width of the model.
const HIDDEN: usize = 2304;
/// Routed experts.
pub const EXPERTS: usize = 128;
/// Experts chosen per token.
pub const TOP_K: usize = 8;
/// Threads per block of `router_logits_2304`, which fixes its summation order.
const LOGITS_THREADS: u32 = 256;
/// Threads per block of `expert_routing_128_top8`: one warp per row.
const ROUTING_THREADS: u32 = 256;
/// Threads per row of `depth_attention_2304`, which fixes its summation order.
const DEPTH_THREADS: u32 = 288;
/// Candidates and queries `depth_attention_fused_2304` holds in registers.
pub const FUSED_MAX_CANDIDATES: usize = 6;
pub const FUSED_MAX_QUERIES: usize = 4;
/// Normalized copies `depth_attention_fused_2304` can write.
pub const FUSED_DESTS: usize = 4;
/// Blocks and threads of `moe_router_2304`.
const ROUTER_BLOCKS: u32 = 64;
const ROUTER_THREADS: u32 = 288;
/// Partial sums per logit of the chunked router: 4 subtrees of the shuffle tree of each
/// of the 8 warps of `router_logits_2304`.
const ROUTER_SUBTREES: usize = 32;
/// Tile of `moe_router_logits_rows_2304`: rows by experts, with 256 threads.
const ROUTER_TILE_ROWS: usize = 64;
const ROUTER_TILE_EXPERTS: usize = 32;
const ROUTER_TILE_THREADS: u32 = 128;
/// Rows per block of `moe_router_norm_rows_2304`, one per warp.
const ROUTER_NORM_ROWS: usize = 8;

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

/// A BF16 copy of one depth-attention output row after an RMS norm.
pub struct NormDest<'a> {
    pub out: CudaViewMut<'a, bf16>,
    /// The weight of a weighted RMS norm, or `None` for an unweighted one.
    pub weight: Option<&'a CudaSlice<bf16>>,
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
    /// The subtree sums of each logit, F32 `[32, rows, 128]`.
    partials: CudaSlice<f32>,
    /// The normed rows, permuted for the logits kernel.
    x_perm: CudaSlice<bf16>,
}

impl RouterRowsScratch {
    pub fn new(stream: &Arc<CudaStream>, rows: usize) -> anyhow::Result<Self> {
        Ok(Self {
            rows,
            partials: stream.alloc_zeros(ROUTER_SUBTREES * rows * EXPERTS)?,
            x_perm: stream.alloc_zeros(rows * HIDDEN)?,
        })
    }
}

#[derive(Debug)]
pub struct Routing {
    router_logits_2304: CudaFunction,
    expert_routing_128_top8: CudaFunction,
    depth_attention_2304: CudaFunction,
    depth_attention_fused_2304: CudaFunction,
    depth_attention_fused_rows_2304: CudaFunction,
    moe_router_2304: CudaFunction,
    moe_router_norm_rows_2304: CudaFunction,
    moe_router_logits_rows_2304: CudaFunction,
    moe_router_route_rows_2304: CudaFunction,
    depth_attention_step_2304: CudaFunction,
    moe_router_step_2304: CudaFunction,
}

impl Routing {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "routing", cubin!("routing"))?;
        Ok(Self {
            router_logits_2304: module.load_function("router_logits_2304")?,
            expert_routing_128_top8: module.load_function("expert_routing_128_top8")?,
            depth_attention_2304: module.load_function("depth_attention_2304")?,
            depth_attention_fused_2304: module.load_function("depth_attention_fused_2304")?,
            depth_attention_fused_rows_2304: module
                .load_function("depth_attention_fused_rows_2304")?,
            moe_router_2304: module.load_function("moe_router_2304")?,
            moe_router_norm_rows_2304: module.load_function("moe_router_norm_rows_2304")?,
            moe_router_logits_rows_2304: module.load_function("moe_router_logits_rows_2304")?,
            moe_router_route_rows_2304: module.load_function("moe_router_route_rows_2304")?,
            depth_attention_step_2304: module.load_function("depth_attention_step_2304")?,
            moe_router_step_2304: module.load_function("moe_router_step_2304")?,
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
            .arg(&route_scale);
        // SAFETY: the arguments match `expert_routing_128_top8(const float*, const float*,
        // int32_t*, float*, int, float)`, the kernel skips warps past `rows`, and the checks
        // above keep every row inside the buffers.
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
            .f32(route_scale);
        // SAFETY: the arguments match `moe_router_2304(const float*, const bf16*, float,
        // const float*, const float*, bf16*, float*, int32_t*, float*, unsigned*, float)`,
        // every buffer has the length the kernel reads or writes, and it reads only weights
        // before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.moe_router_2304, config) }
    }

    /// [`Routing::depth_attention_fused`] for each of the `rows` rows of a chunk, with the
    /// same results per row.
    ///
    /// `blocks` holds `nb` block states of `[rows, 2304]` each, one after another. The
    /// partial, `normed_partial` and `out` hold `[rows, 2304]`, `[rows, 2304]` and
    /// `[rows, nq, 2304]`. The rows of each destination are `dst_stride` elements apart, so
    /// a destination may be one half of a `[rows, 2 * 2304]` buffer: its view starts at
    /// row 0 and must reach the end of the last row.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn depth_attention_fused_rows(
        &self,
        stream: &CudaStream,
        blocks: &impl DeviceIn<f32>,
        partial: Partial<'_>,
        mut normed_partial: Option<CudaViewMut<'_, f32>>,
        queries: &CudaSlice<f32>,
        mut out: Option<CudaViewMut<'_, f32>>,
        mut dests: [Option<NormDest<'_>>; FUSED_DESTS],
        dst_stride: usize,
        eps: f32,
        rows: usize,
    ) -> anyhow::Result<()> {
        let blocks = blocks.view();
        ensure!(rows > 0, "no rows");
        let block_len = rows * HIDDEN;
        ensure!(
            blocks.len().is_multiple_of(block_len),
            "blocks has {} elements, not a multiple of {rows} x {HIDDEN}",
            blocks.len()
        );
        let nb = blocks.len() / block_len;
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
            partial_len == block_len,
            "partial has {partial_len} elements, expected {rows} x {HIDDEN}"
        );
        ensure!(
            normed_partial.as_ref().is_none_or(|x| x.len() == block_len),
            "normed_partial must hold {rows} x {HIDDEN} elements"
        );
        ensure!(
            out.as_ref().is_none_or(|x| x.len() == rows * nq * HIDDEN),
            "out must hold {rows} x {nq} x {HIDDEN} elements"
        );
        ensure!(dst_stride >= HIDDEN, "dst_stride is below {HIDDEN}");
        let dest_len = (rows - 1) * dst_stride + HIDDEN;
        for dest in dests.iter().flatten() {
            ensure!(
                dest.out.len() >= dest_len && dest.weight.is_none_or(|w| w.len() == HIDDEN),
                "destinations must reach {dest_len} elements and norm weights hold {HIDDEN}"
            );
        }

        let scale = (HIDDEN as f32).sqrt().recip();
        let mut args = KernelArgs::default();
        args.ptr(&blocks, stream)
            .i32(i32::try_from(block_len).context("too many rows")?)
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
        args.i32(i32::try_from(dst_stride).context("dst_stride too large")?);
        for dest in &dests {
            args.opt(dest.as_ref().and_then(|d| d.weight), stream);
        }
        args.f32(eps).f32(scale);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: (DEPTH_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `depth_attention_fused_rows_2304`'s signature, every
        // non-null pointer comes from a live buffer that holds the rows checked above, the
        // grid has one block per row, and the kernel reads only the queries before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.depth_attention_fused_rows_2304, config) }
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

    /// The router for one BF16 row `x` that is already normalized: logits from `x`, then the
    /// top-8 routing of [`Routing::moe_router`], in one launch. The logits sum in the order
    /// of `moe_router_step_2304`.
    ///
    /// `counter` must hold 0; the kernel leaves it at 0.
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
        counter: &mut CudaSlice<u32>,
        route_scale: f32,
    ) -> anyhow::Result<()> {
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
        ensure!(!counter.is_empty(), "counter is empty");
        let config = LaunchConfig {
            grid_dim: (EXPERTS as u32, 1, 1),
            block_dim: (ROUTER_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(x, stream)
            .ptr(router, stream)
            .ptr(bias, stream)
            .ptr_mut(logits, stream)
            .ptr_mut(indices, stream)
            .ptr_mut(weights, stream)
            .ptr_mut(counter, stream)
            .f32(route_scale);
        // SAFETY: the arguments match `moe_router_step_2304(const bf16*, const float*, const
        // float*, float*, int32_t*, float*, unsigned*, float)`, every buffer has the length
        // the kernel reads or writes, the grid has one block per expert, and the kernel reads
        // only the router before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.moe_router_step_2304, config) }
    }

    /// [`Routing::moe_router`] for each of the `rows` rows of a chunk: `x` and `x_out`
    /// `[rows, 2304]`, `logits` `[rows, 128]`, `indices` and `weights` `[rows, 8]`, and
    /// `scratch` for at least `rows` rows. Each row's results equal a single-row call's bit
    /// for bit: the logits run as a tiled matrix product that keeps the single-row sums.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signatures")]
    pub fn moe_router_rows(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<f32>,
        norm_weight: &CudaSlice<bf16>,
        eps: f32,
        router: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        x_out: &mut impl DeviceOut<bf16>,
        logits: &mut impl DeviceOut<f32>,
        indices: &mut impl DeviceOut<i32>,
        weights: &mut impl DeviceOut<f32>,
        scratch: &mut RouterRowsScratch,
        route_scale: f32,
        rows: usize,
    ) -> anyhow::Result<()> {
        let x = x.view();
        let (mut x_out, mut logits, mut indices, mut weights) = (
            x_out.view_mut(),
            logits.view_mut(),
            indices.view_mut(),
            weights.view_mut(),
        );
        ensure!(rows > 0, "no rows");
        ensure!(
            x.len() == rows * HIDDEN && x_out.len() == rows * HIDDEN,
            "x and x_out must hold {rows} x {HIDDEN} elements"
        );
        ensure!(norm_weight.len() == HIDDEN, "norm weight length");
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
        let mut partials = scratch
            .partials
            .slice_mut(..ROUTER_SUBTREES * rows * EXPERTS);
        let mut x_perm = scratch.x_perm.slice_mut(..rows * HIDDEN);
        let grid_rows = u32::try_from(rows).context("too many rows")?;
        let rows_i32 = i32::try_from(rows).context("too many rows")?;

        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .ptr(norm_weight, stream)
            .f32(eps)
            .ptr_mut(&mut x_out, stream)
            .ptr_mut(&mut x_perm, stream)
            .i32(rows_i32);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows.div_ceil(ROUTER_NORM_ROWS))?, 1, 1),
            block_dim: (u32::try_from(ROUTER_NORM_ROWS * 32)?, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `moe_router_norm_rows_2304(const float*, const bf16*,
        // float, bf16*, uint2*, int)`, x, x_out and x_perm hold `rows` rows, the grid has a
        // warp per row in blocks of 8 (the kernel skips rows past `rows`), and the kernel
        // reads nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.moe_router_norm_rows_2304, config) }?;

        let mut args = KernelArgs::default();
        args.ptr(&x_perm, stream)
            .ptr(router, stream)
            .ptr_mut(&mut partials, stream)
            .i32(rows_i32);
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(rows.div_ceil(ROUTER_TILE_ROWS))?,
                u32::try_from(EXPERTS / ROUTER_TILE_EXPERTS)?,
                u32::try_from(ROUTER_SUBTREES)?,
            ),
            block_dim: (ROUTER_TILE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `moe_router_logits_rows_2304(const uint2*, const float*,
        // float*, int)`, x_perm holds `rows` rows and partials 32 x `rows` x 128 floats, the
        // grid covers the rows in tiles of 64 (the kernel skips rows past `rows`), and the
        // kernel reads only the router before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.moe_router_logits_rows_2304, config) }?;

        let mut args = KernelArgs::default();
        args.ptr(&partials, stream)
            .ptr(bias, stream)
            .ptr_mut(&mut logits, stream)
            .ptr_mut(&mut indices, stream)
            .ptr_mut(&mut weights, stream)
            .i32(rows_i32)
            .f32(route_scale);
        let config = LaunchConfig {
            grid_dim: (grid_rows, 1, 1),
            block_dim: (u32::try_from(EXPERTS)?, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `moe_router_route_rows_2304(const float*, const float*,
        // float*, int*, float*, int, float)`, every buffer holds `rows` rows as checked
        // above, the grid has one block of 128 threads per row, and the kernel reads nothing
        // before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.moe_router_route_rows_2304, config) }
    }
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
