//! NVFP4 and MXFP8 expert products: matrix-vector for decode (`kernels/experts.cu`) and
//! expert-grouped matrix products for prefill (`kernels/expert_matmul.cu`).

use std::sync::Arc;

use anyhow::{Context as _, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
    sys::CUfunction_attribute,
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, cubin, load_module};

/// Experts each token routes to.
const SLOTS: usize = 8;
/// Output rows per block, one warp each.
const WARPS: u32 = 8;
/// `expert_matmul`: output columns per block, pairs per tile, input values per pipeline stage,
/// and threads per block.
const GEMM_COLUMNS: usize = 128;
const GEMM_PAIRS: usize = 64;
const GEMM_K: usize = 64;
const GEMM_THREADS: u32 = 256;
/// `expert_matmul` needs k a multiple of this: two pipeline stages, so that each scale copy is
/// 4 or 8 aligned bytes.
const GEMM_K_MULTIPLE: usize = 2 * GEMM_K;
/// `expert_group`: threads, and the largest row and expert counts it supports.
const GROUP_THREADS: u32 = 1024;
const GROUP_MAX_ROWS: usize = 2048;
const GROUP_MAX_EXPERTS: usize = 1024;

/// Quantization format of one expert projection.
#[derive(Debug, Clone, Copy)]
pub enum ExpertFormat<'a> {
    /// Layers 0 to 19: E2M1 weights, one E4M3 scale per 16 values, and one F32 global scale per
    /// expert (`[experts]`), computed as `amax * 6 * 448 / outer_scale_denominator`.
    Nvfp4 { global_scale: &'a CudaSlice<f32> },
    /// Layers 20 to 23: E4M3 weights and one E8M0 scale per 32 values.
    Mxfp8,
}

impl ExpertFormat<'_> {
    fn group(self) -> usize {
        match self {
            Self::Nvfp4 { .. } => 16,
            Self::Mxfp8 => 32,
        }
    }

    fn row_bytes(self, k: usize) -> usize {
        match self {
            Self::Nvfp4 { .. } => k / 2,
            Self::Mxfp8 => k,
        }
    }

    /// Dynamic shared memory of an `expert_gemm_*` block: per pipeline stage, the weight bytes
    /// and the BF16 inputs, then 2 slots of scales for a pair of stages. Both formats run 3
    /// stages, so that two blocks fit in an SM.
    fn gemm_shared_bytes(self) -> usize {
        let stages = 3;
        let scale_slots = 2;
        stages * (GEMM_COLUMNS * self.row_bytes(GEMM_K) + GEMM_PAIRS * GEMM_K * 2)
            + scale_slots * GEMM_COLUMNS * 2 * GEMM_K / self.group()
    }
}

/// One expert projection (`w13` or `w2`) as raw checkpoint bytes on the GPU.
#[derive(Debug, Clone, Copy)]
pub struct ExpertWeights<'a> {
    pub format: ExpertFormat<'a>,
    /// `*_weight`: `[experts, n, k / 2]` for NVFP4, `[experts, n, k]` for MXFP8.
    pub weight: &'a CudaSlice<u8>,
    /// `*_weight_scale`: `[experts, n, k / 16]` for NVFP4, `[experts, n, k / 32]` for MXFP8.
    pub scales: &'a CudaSlice<u8>,
    /// Output rows per expert: 2304 for both projections. Tests pass a slice.
    pub n: usize,
    /// Input width: 2304 for `w13`, 1152 for `w2`.
    pub k: usize,
}

/// Checked shapes of one `expert_matvec` or `expert_matmul` call.
struct Shape {
    experts: usize,
    rows: usize,
    /// 1 for one input shared by the 8 slots, 8 for one input per slot.
    inputs_per_row: i32,
}

impl ExpertWeights<'_> {
    fn check(&self, x_len: usize, indices_len: usize, out_len: usize) -> anyhow::Result<Shape> {
        let &ExpertWeights {
            format,
            weight,
            scales,
            n,
            k,
        } = self;
        let group = format.group();
        ensure!(n > 0, "n must be positive");
        ensure!(
            k > 0 && k.is_multiple_of(group),
            "k = {k} is not a positive multiple of {group}"
        );
        let row_bytes = n * format.row_bytes(k);
        ensure!(
            weight.len().is_multiple_of(row_bytes) && !weight.is_empty(),
            "weight has {} bytes, not a whole number of {n} x {} experts",
            weight.len(),
            format.row_bytes(k)
        );
        let experts = weight.len() / row_bytes;
        ensure!(
            scales.len() == experts * n * (k / group),
            "scales has {} bytes, expected {experts} x {n} x {}",
            scales.len(),
            k / group
        );
        if let ExpertFormat::Nvfp4 { global_scale } = format {
            ensure!(
                global_scale.len() == experts,
                "global_scale has {} elements, expected {experts}",
                global_scale.len()
            );
        }
        ensure!(
            indices_len.is_multiple_of(SLOTS),
            "indices has {indices_len} elements, not a multiple of {SLOTS}"
        );
        let rows = indices_len / SLOTS;
        let inputs_per_row = if x_len == rows * k {
            1
        } else if x_len == rows * SLOTS * k {
            8
        } else {
            bail!("x has {x_len} elements, expected {rows} x 1 or {rows} x {SLOTS} inputs of {k}");
        };
        ensure!(
            out_len == rows * SLOTS * n,
            "out has {out_len} elements, expected {rows} x {SLOTS} x {n}"
        );
        Ok(Shape {
            experts,
            rows,
            inputs_per_row,
        })
    }
}

/// Device scratch for [`Experts::expert_matmul`]: the (row, slot) pairs sorted by expert and
/// the list of tiles. Create it once for the largest call.
#[derive(Debug)]
pub struct ExpertScratch {
    /// Flat pair index `row * 8 + slot` of each pair, in expert order, then the pairs with
    /// unknown ids.
    perm: CudaSlice<i32>,
    /// `{expert, first position in perm, pairs, 0}` per tile of at most 64 pairs.
    tiles: CudaSlice<i32>,
    /// The number of tiles, of pairs with known ids, and of all pairs.
    group_counts: CudaSlice<i32>,
    max_rows: usize,
    max_experts: usize,
    /// The grouping that [`Experts::expert_group`] last queued.
    grouping: Option<Grouping>,
}

/// What [`Experts::expert_group`] grouped: the row and expert counts, and the stream it ran on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Grouping {
    rows: usize,
    experts: usize,
    stream: usize,
}

impl ExpertScratch {
    /// Scratch for calls with up to `max_rows` rows and `max_experts` experts, at most 2048
    /// and 1024.
    pub fn new(
        stream: &Arc<CudaStream>,
        max_rows: usize,
        max_experts: usize,
    ) -> anyhow::Result<Self> {
        ensure!(
            max_rows > 0 && max_experts > 0,
            "max_rows and max_experts must be positive"
        );
        ensure!(
            max_rows <= GROUP_MAX_ROWS && max_experts <= GROUP_MAX_EXPERTS,
            "{max_rows} rows and {max_experts} experts, at most {GROUP_MAX_ROWS} and \
             {GROUP_MAX_EXPERTS} supported"
        );
        let pairs = max_rows * SLOTS;
        Ok(Self {
            perm: stream.alloc_zeros(pairs)?,
            tiles: stream.alloc_zeros(4 * max_tiles(pairs, max_experts))?,
            group_counts: stream.alloc_zeros(3)?,
            max_rows,
            max_experts,
            grouping: None,
        })
    }
}

/// Most tiles `pairs` pairs over `experts` experts can need: one partial tile per expert
/// besides the full ones, and never more tiles than pairs.
fn max_tiles(pairs: usize, experts: usize) -> usize {
    pairs.min(pairs / GEMM_PAIRS + experts)
}

#[derive(Debug)]
pub struct Experts {
    nvfp4_8: CudaFunction,
    nvfp4_16: CudaFunction,
    mxfp8_4: CudaFunction,
    mxfp8_8: CudaFunction,
    group: CudaFunction,
    gemm_nvfp4: CudaFunction,
    gemm_mxfp8: CudaFunction,
}

impl Experts {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "experts", cubin!("experts"))?;
        let matmul = load_module(ctx, "expert_matmul", cubin!("expert_matmul"))?;
        let gemm_nvfp4 = matmul.load_function("expert_gemm_nvfp4")?;
        let gemm_mxfp8 = matmul.load_function("expert_gemm_mxfp8")?;
        // MXFP8 needs the most: 3 stages of 16 KiB and 1 KiB of scales against NVFP4's 3 of
        // 12 KiB and 2 KiB of scales.
        let shared = i32::try_from(ExpertFormat::Mxfp8.gemm_shared_bytes())?;
        for function in [&gemm_nvfp4, &gemm_mxfp8] {
            function.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            )?;
        }
        Ok(Self {
            nvfp4_8: module.load_function("expert_matvec_nvfp4_8")?,
            nvfp4_16: module.load_function("expert_matvec_nvfp4_16")?,
            mxfp8_4: module.load_function("expert_matvec_mxfp8_4")?,
            mxfp8_8: module.load_function("expert_matvec_mxfp8_8")?,
            group: matmul.load_function("expert_group")?,
            gemm_nvfp4,
            gemm_mxfp8,
        })
    }

    /// Multiplies each row's input by the weights of its 8 selected experts, writing BF16
    /// `[rows, 8, n]` to `out`.
    ///
    /// `indices` is `[rows, 8]`. `x` is BF16 `[rows, 1, k]`, one input shared by all 8 slots
    /// (`w13`), or `[rows, 8, k]`, one input per slot (`w2`). Each id must be below the expert
    /// count of `weights`; an id outside it gives a NaN row.
    ///
    /// Equals MLX's `mx.gather_qmm` on one row per expert, bit for bit.
    pub fn expert_matvec(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        indices: &impl DeviceIn<i32>,
        weights: &ExpertWeights<'_>,
        out: &mut impl DeviceOut<bf16>,
    ) -> anyhow::Result<()> {
        let (x, indices, mut out) = (x.view(), indices.view(), out.view_mut());
        let Shape {
            experts,
            rows,
            inputs_per_row: inputs_arg,
        } = weights.check(x.len(), indices.len(), out.len())?;
        if rows == 0 {
            return Ok(());
        }
        let &ExpertWeights {
            format,
            weight,
            scales,
            n,
            k,
        } = weights;

        // MLX's rule for its fast kernel, which gives each lane twice as many values.
        let (fast_k, function_fast, function) = match format {
            ExpertFormat::Nvfp4 { .. } => (512, &self.nvfp4_16, &self.nvfp4_8),
            ExpertFormat::Mxfp8 => (256, &self.mxfp8_8, &self.mxfp8_4),
        };
        let fast = n.is_multiple_of(WARPS as usize) && k.is_multiple_of(fast_k);
        let function = if fast { function_fast } else { function };

        let slot_rows = u32::try_from(rows * SLOTS).context("too many rows")?;
        let tiles = u32::try_from(n.div_ceil(WARPS as usize)).context("n is too large")?;
        let n_arg = i32::try_from(n).context("n is too large")?;
        let k_arg = i32::try_from(k).context("k is too large")?;
        let experts_arg = i32::try_from(experts).context("too many experts")?;
        // A null pointer stands in for MXFP8's missing global scale.
        let null: u64 = 0;
        let config = LaunchConfig {
            grid_dim: (slot_rows, tiles, 1),
            block_dim: (WARPS * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&x).arg(&indices).arg(weight).arg(scales);
        match format {
            ExpertFormat::Nvfp4 { global_scale } => launch.arg(global_scale),
            ExpertFormat::Mxfp8 => launch.arg(&null),
        };
        launch
            .arg(&mut out)
            .arg(&n_arg)
            .arg(&k_arg)
            .arg(&inputs_arg)
            .arg(&experts_arg);
        // SAFETY: the arguments match `expert_matvec_*(const bf16*, const int*, const u8*,
        // const u8*, const float*, bf16*, int, int, int, int)`. The checks above keep every input
        // row inside `x`, every output inside `out`, and every weight and scale row of an expert
        // below `experts` inside `weight` and `scales`; the kernel skips other expert ids. Only
        // the NVFP4 kernels read `global_scale`, which then holds `experts` values.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// [`Self::expert_matvec`] for many rows: the same arguments, shapes and results, up to
    /// F32 summation order.
    ///
    /// Runs [`Self::expert_group`], then [`Self::expert_matmul_grouped`]. To multiply several
    /// projections with the same `indices`, group once and call the latter for each.
    pub fn expert_matmul(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        indices: &impl DeviceIn<i32>,
        weights: &ExpertWeights<'_>,
        out: &mut impl DeviceOut<bf16>,
        scratch: &mut ExpertScratch,
    ) -> anyhow::Result<()> {
        let Shape { experts, .. } =
            weights.check(x.view().len(), indices.view().len(), out.view_mut().len())?;
        self.expert_group(stream, indices, experts, scratch)?;
        self.expert_matmul_grouped(stream, x, weights, out, scratch)
    }

    /// Groups the (row, slot) pairs of `indices` (`[rows, 8]`) by expert into `scratch`, for
    /// [`Self::expert_matmul_grouped`] calls on the same stream. Ids at or above `experts`
    /// give NaN rows there. The grouping depends only on `indices`, so one serves every
    /// projection routed by them.
    ///
    /// `scratch` must have room for `rows` rows and `experts` experts. The call allocates
    /// nothing and does not wait for the GPU.
    pub fn expert_group(
        &self,
        stream: &CudaStream,
        indices: &impl DeviceIn<i32>,
        experts: usize,
        scratch: &mut ExpertScratch,
    ) -> anyhow::Result<()> {
        let indices = indices.view();
        ensure!(
            indices.len().is_multiple_of(SLOTS),
            "indices has {} elements, not a multiple of {SLOTS}",
            indices.len()
        );
        let rows = indices.len() / SLOTS;
        ensure!(experts > 0, "experts must be positive");
        ensure!(
            rows <= scratch.max_rows && experts <= scratch.max_experts,
            "scratch holds {} rows and {} experts, the call has {rows} and {experts}",
            scratch.max_rows,
            scratch.max_experts
        );
        scratch.grouping = None;
        if rows > 0 {
            let pairs_arg = i32::try_from(rows * SLOTS).context("too many rows")?;
            let experts_arg = i32::try_from(experts).context("too many experts")?;
            let mut launch = stream.launch_builder(&self.group);
            launch
                .arg(&indices)
                .arg(&pairs_arg)
                .arg(&experts_arg)
                .arg(&mut scratch.perm)
                .arg(&mut scratch.tiles)
                .arg(&mut scratch.group_counts);
            let config = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (GROUP_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the arguments match `expert_group(const int*, int, int, int*, int4*,
            // int*)`. `indices` holds `pairs` ids. The scratch was sized for at least `rows`
            // rows and `experts` experts, at most 2048 and 1024 as the kernel needs, so `perm`
            // holds `pairs` entries, `tiles` every tile the kernel can write, and
            // `group_counts` 3 values.
            unsafe { launch.launch(config) }?;
        }
        scratch.grouping = Some(Grouping {
            rows,
            experts,
            stream: stream_id(stream),
        });
        Ok(())
    }

    /// [`Self::expert_matmul`] with the grouping that the last [`Self::expert_group`] call on
    /// `scratch` queued on this stream: `weights` must have that call's expert count, and `x`
    /// and `out` its row count.
    ///
    /// Multiplies each expert's pairs as one matrix product, so each expert's weights are read
    /// once per 64 pairs instead of once per pair. The call allocates nothing and does not
    /// wait for the GPU.
    ///
    /// Needs n a multiple of 16 and k a multiple of 128.
    pub fn expert_matmul_grouped(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        weights: &ExpertWeights<'_>,
        out: &mut impl DeviceOut<bf16>,
        scratch: &ExpertScratch,
    ) -> anyhow::Result<()> {
        let (x, mut out) = (x.view(), out.view_mut());
        let grouping = scratch
            .grouping
            .context("expert_matmul_grouped needs a grouping from expert_group")?;
        let Shape {
            experts,
            rows,
            inputs_per_row,
        } = weights.check(x.len(), grouping.rows * SLOTS, out.len())?;
        ensure!(
            experts == grouping.experts,
            "weights have {experts} experts, the grouping {}",
            grouping.experts
        );
        ensure!(
            grouping.stream == stream_id(stream),
            "the grouping was queued on another stream"
        );
        if rows == 0 {
            return Ok(());
        }
        let &ExpertWeights {
            format,
            weight,
            scales,
            n,
            k,
        } = weights;
        ensure!(
            n.is_multiple_of(16) && k.is_multiple_of(GEMM_K_MULTIPLE),
            "expert_matmul needs n a multiple of 16 and k of {GEMM_K_MULTIPLE}, got n = {n}, \
             k = {k}"
        );

        let pairs = rows * SLOTS;
        let n_arg = i32::try_from(n).context("n is too large")?;
        let k_arg = i32::try_from(k).context("k is too large")?;
        let function = match format {
            ExpertFormat::Nvfp4 { .. } => &self.gemm_nvfp4,
            ExpertFormat::Mxfp8 => &self.gemm_mxfp8,
        };
        let null: u64 = 0;
        // One more row of blocks than tiles: the row at the tile count writes the NaN rows.
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(n.div_ceil(GEMM_COLUMNS))?,
                u32::try_from(max_tiles(pairs, experts) + 1)?,
                1,
            ),
            block_dim: (GEMM_THREADS, 1, 1),
            shared_mem_bytes: u32::try_from(format.gemm_shared_bytes())?,
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&x).arg(weight).arg(scales);
        match format {
            ExpertFormat::Nvfp4 { global_scale } => launch.arg(global_scale),
            ExpertFormat::Mxfp8 => launch.arg(&null),
        };
        launch
            .arg(&mut out)
            .arg(&scratch.perm)
            .arg(&scratch.tiles)
            .arg(&scratch.group_counts)
            .arg(&n_arg)
            .arg(&k_arg)
            .arg(&inputs_per_row);
        // SAFETY: the arguments match `expert_gemm_*(const bf16*, const u8*, const u8*,
        // const float*, bf16*, const int*, const int4*, const int*, int, int, int)`.
        // `expert_group` ran earlier on this stream for `rows` rows and `experts` experts, so
        // `perm` holds a permutation of the `pairs` pair indices, its first `group_counts[1]`
        // with ids below `experts`, and `tiles` holds at most `max_tiles` tiles of those. So
        // every input row is inside `x`, every output row inside `out`, and every weight and
        // scale row inside `weight` and `scales`; the kernel reads weight rows only below n.
        // n and k meet the kernel's alignment needs, checked above. Only the NVFP4 kernel
        // reads `global_scale`, which then holds `experts` values.
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}

/// Identifies a stream for [`Grouping`].
fn stream_id(stream: &CudaStream) -> usize {
    stream.cu_stream().addr()
}
