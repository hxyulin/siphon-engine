//! NVFP4 and MXFP8 expert products: matrix-vector for decode (`kernels/experts.cu`) and
//! expert-grouped matrix products for prefill (`kernels/expert_matmul.cu`).

use std::{fmt, str::FromStr, sync::Arc};

use anyhow::{Context as _, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg as _,
    sys::{CUdevice_attribute, CUfunction_attribute},
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module};

/// Experts each token routes to.
const SLOTS: usize = 8;
/// The hidden size, which the end of the MoE block in `expert_mlp` needs as `w2.n`.
const HIDDEN: usize = 2304;
/// Taps of the post-MLP conv.
const CONV_TAPS: usize = 4;

/// The end of the MoE block for one row, which [`Experts::expert_mlp`] can run after its
/// products, as `Conv::moe_conv_residual` does: `hidden = bf16(residual + conv(combine))`,
/// where `combine` sums the 8 expert outputs times `weights` and `conv` is the post-MLP
/// short conv with taps `conv_weight` (`[2304, 1, 4]`) and state `conv_state`
/// (`[3, 2304]`, updated).
pub struct MoeEnd<'a> {
    pub weights: &'a CudaSlice<f32>,
    pub conv_weight: &'a CudaSlice<bf16>,
    pub conv_state: &'a mut CudaSlice<f32>,
    pub residual: &'a CudaSlice<f32>,
    pub hidden: &'a mut CudaSlice<bf16>,
}
/// `expert_matvec` and `expert_mlp`: warps per block, and 16-byte weight chunks per lane.
const MATVEC_WARPS: usize = 4;
const MATVEC_CHUNKS: usize = 9;
/// Input widths the `expert_matvec` kernels support, in the order of [`MATVEC_KERNELS`].
const MATVEC_K: [usize; 2] = [2304, 1152];
/// `expert_matvec` kernels by format and input width.
const MATVEC_KERNELS: [&str; 4] = [
    "expert_matvec_nvfp4_2304",
    "expert_matvec_nvfp4_1152",
    "expert_matvec_mxfp8_2304",
    "expert_matvec_mxfp8_1152",
];
/// `expert_mlp`: the input widths of `w13` and `w2`, and the blocks it runs per SM for
/// NVFP4 and MXFP8, the fastest measured on an RTX 5090.
const MLP_K13: usize = 2304;
const MLP_K2: usize = 1152;
const MLP_BLOCKS_PER_SM: [usize; 2] = [4, 7];
/// `expert_matmul`: output columns per block, pairs per tile, input values per pipeline stage,
/// and threads per block.
const GEMM_COLUMNS: usize = 128;
const GEMM_PAIRS: usize = 64;
/// Pairs per tile of the quantized products, `kQTileT` in the kernel.
const QUANTIZED_PAIRS: usize = 128;
const GEMM_K: usize = 64;
const GEMM_THREADS: u32 = 256;
/// Threads per block of `expert_gemm_q8_*`.
const FP8_GEMM_THREADS: u32 = 256;
/// `expert_matmul` needs k a multiple of this: two pipeline stages, so that each scale copy is
/// 4 or 8 aligned bytes.
const GEMM_K_MULTIPLE: usize = 2 * GEMM_K;
/// `expert_group`: threads, and the largest row and expert counts it supports.
const GROUP_THREADS: u32 = 1024;
const GROUP_MAX_ROWS: usize = 2048;
const GROUP_MAX_EXPERTS: usize = 1024;

/// How prefill's expert products take their activations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExpertActivations {
    /// BF16, as MLX computes them: the weights are decoded to BF16.
    #[default]
    Bf16,
    /// Quantized as the checkpoint's `hf_quant_config.json` declares for each expert group,
    /// NVFP4 or MXFP8 (see [`Experts::quantize_rows`]), and multiplied on the block-scaled
    /// FP4 or FP8 tensor cores. Lowers precision on purpose; decode stays BF16.
    Quantized,
    /// Quantized to MXFP8 for every expert group (see [`Experts::quantize_rows_fp8`]) and
    /// multiplied on the block-scaled FP8 tensor cores, the NVFP4 weights with their E4M3
    /// scales applied exactly. Lowers precision on purpose; decode stays BF16.
    Fp8,
}

impl FromStr for ExpertActivations {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "quantized" => Ok(Self::Quantized),
            "fp8" => Ok(Self::Fp8),
            _ => bail!("unknown expert activations {s:?}; expected bf16, quantized or fp8"),
        }
    }
}

impl fmt::Display for ExpertActivations {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Quantized => "quantized",
            Self::Fp8 => "fp8",
        })
    }
}

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

    /// Output rows one `expert_matvec` or `expert_mlp` block covers for input width k:
    /// `MATVEC_WARPS` warps, each with whole rows of `MATVEC_CHUNKS` 16-byte chunks per lane.
    fn matvec_block_rows(self, k: usize) -> usize {
        let chunk_values = match self {
            Self::Nvfp4 { .. } => 32,
            Self::Mxfp8 => 16,
        };
        MATVEC_WARPS * 32 * MATVEC_CHUNKS * chunk_values / k
    }

    fn index(self) -> usize {
        match self {
            Self::Nvfp4 { .. } => 0,
            Self::Mxfp8 => 1,
        }
    }

    fn row_bytes(self, k: usize) -> usize {
        match self {
            Self::Nvfp4 { .. } => k / 2,
            Self::Mxfp8 => k,
        }
    }

    /// Dynamic shared memory of an `expert_gemm_q_*` block, `kQShared` in
    /// `kernels/expert_matmul.cu`: 2 stages of 64 bytes for each of the 128 weight rows and
    /// 128 pairs, then 2 scale slots of 16 bytes per row. The same for both formats.
    fn quantized_shared_bytes() -> usize {
        let rows = GEMM_COLUMNS + QUANTIZED_PAIRS;
        2 * rows * 64 + 2 * rows * 16
    }

    /// Dynamic shared memory of an `expert_gemm_q8_*` block, `kQ8Shared` in
    /// `kernels/expert_matmul.cu`: 3 stages of 32 bytes per weight row and 64 per pair, the
    /// same scale slots, then the decode table of 176 entries of 16 bytes.
    fn fp8_shared_bytes() -> usize {
        let rows = GEMM_COLUMNS + QUANTIZED_PAIRS;
        3 * (GEMM_COLUMNS * 32 + QUANTIZED_PAIRS * 64) + 2 * rows * 16 + 176 * 16
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

/// Activations quantized by [`Experts::quantize_rows`] or [`Experts::quantize_rows_fp8`] for
/// the quantized expert products.
///
/// NVFP4 rows hold k / 2 bytes of E2M1 pairs (low nibble first), k / 16 E4M3 block scales
/// and one F32 outer scale; MXFP8 rows k E4M3 bytes and k / 32 UE8M0 block scales. The MXFP8
/// rows that `quantize_rows_fp8` writes for NVFP4 weights store each block of 32 values in
/// the order the FP8 kernel loads them.
#[derive(Debug)]
pub struct QuantizedRows {
    values: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    outer: CudaSlice<f32>,
    /// What the last [`Experts::quantize_rows`] call queued.
    contents: Option<Quantized>,
}

/// What [`Experts::quantize_rows`] wrote: the rows, their width and format, and the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Quantized {
    rows: usize,
    k: usize,
    format: RowFormat,
    stream: usize,
}

/// The format of quantized rows: one per `quantize_rows_*` kernel, in [`Experts::quantize`]'s
/// order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowFormat {
    Nvfp4 = 0,
    Mxfp8 = 1,
    /// MXFP8 in the order of `expert_gemm_q8_*`, for NVFP4 weights.
    Fp8Paired = 2,
}

impl RowFormat {
    /// The row format of the quantized products of `weights`: their checkpoint format, or
    /// with `fp8` MXFP8, paired for NVFP4 weights.
    fn of(weights: ExpertFormat<'_>, fp8: bool) -> Self {
        match (weights, fp8) {
            (ExpertFormat::Nvfp4 { .. }, false) => Self::Nvfp4,
            (ExpertFormat::Nvfp4 { .. }, true) => Self::Fp8Paired,
            (ExpertFormat::Mxfp8, _) => Self::Mxfp8,
        }
    }
}

impl QuantizedRows {
    /// Room for up to `max_values` values in up to `max_rows` rows, in either format.
    pub fn new(
        stream: &Arc<CudaStream>,
        max_values: usize,
        max_rows: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            values: stream.alloc_zeros(max_values.max(1))?,
            scales: stream.alloc_zeros((max_values / 16).max(1))?,
            outer: stream.alloc_zeros(max_rows.max(1))?,
            contents: None,
        })
    }

    /// The packed values, `[rows, k / 2]` E2M1 pairs or `[rows, k]` E4M3.
    #[must_use]
    pub fn values(&self) -> &CudaSlice<u8> {
        &self.values
    }

    /// The block scales, `[rows, k / 16]` E4M3 or `[rows, k / 32]` UE8M0.
    #[must_use]
    pub fn scales(&self) -> &CudaSlice<u8> {
        &self.scales
    }

    /// Each NVFP4 row's outer scale, `amax / 1536`.
    #[must_use]
    pub fn outer(&self) -> &CudaSlice<f32> {
        &self.outer
    }
}

/// Device scratch for [`Experts::expert_matmul`]: the (row, slot) pairs sorted by expert and
/// the list of tiles. Create it once for the largest call.
#[derive(Debug)]
pub struct ExpertScratch {
    /// Flat pair index `row * 8 + slot` of each pair, in expert order, then the pairs with
    /// unknown ids.
    perm: CudaSlice<i32>,
    /// `{expert, first position in perm, pairs, 0}` per tile of at most 64 pairs, or 128 for
    /// the quantized products.
    tiles: CudaSlice<i32>,
    /// The number of tiles, of pairs with known ids, and of all pairs.
    group_counts: CudaSlice<i32>,
    max_rows: usize,
    max_experts: usize,
    /// The grouping that [`Experts::expert_group`] last queued.
    grouping: Option<Grouping>,
}

/// What [`Experts::expert_group`] grouped: the row and expert counts, the pairs per tile,
/// and the stream it ran on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Grouping {
    rows: usize,
    experts: usize,
    tile_pairs: usize,
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
    /// The kernels of [`MATVEC_KERNELS`], in its order.
    matvec: Vec<CudaFunction>,
    /// `expert_mlp` for NVFP4 and MXFP8, and the blocks each runs: all must fit on the GPU
    /// at once.
    mlp: [CudaFunction; 2],
    mlp_blocks: [usize; 2],
    group: CudaFunction,
    /// `expert_gemm_*` for NVFP4 and MXFP8, each plain and with the SwiGLU epilogue.
    gemm: [[CudaFunction; 2]; 2],
    /// The same for `expert_gemm_q_*`, the products with quantized activations.
    gemm_q: [[CudaFunction; 2]; 2],
    /// `expert_gemm_q8_nvfp4`, plain and with the SwiGLU epilogue: NVFP4 weights with FP8
    /// activations.
    gemm_q8: [CudaFunction; 2],
    /// `quantize_rows_*` for NVFP4, MXFP8 and paired MXFP8, in [`RowFormat`] order.
    quantize: [CudaFunction; 3],
    /// `nvfp4_scale_spread`.
    scale_spread: CudaFunction,
}

/// The input of an `expert_gemm_*` kernel.
enum GemmInput<'a, 'b> {
    Bf16(CudaView<'b, bf16>),
    Quantized(&'a QuantizedRows),
}

impl GemmInput<'_, '_> {
    /// The number of input values, after checking that quantized inputs suit `weights`.
    fn len(&self, weights: &ExpertWeights<'_>, stream: &CudaStream) -> anyhow::Result<usize> {
        match self {
            Self::Bf16(x) => Ok(x.len()),
            Self::Quantized(xq) => {
                let contents = xq
                    .contents
                    .context("the quantized products need inputs from quantize_rows")?;
                let suits = contents.format == RowFormat::of(weights.format, false)
                    || contents.format == RowFormat::of(weights.format, true);
                ensure!(
                    contents.k == weights.k && suits,
                    "the inputs were quantized for k = {} or another format than the weights' \
                     k = {}",
                    contents.k,
                    weights.k
                );
                ensure!(
                    contents.stream == stream_id(stream),
                    "the inputs were quantized on another stream"
                );
                Ok(contents.rows * contents.k)
            }
        }
    }
}

/// What an `expert_gemm_*` kernel writes.
#[derive(Debug, Clone, Copy)]
enum Epilogue {
    /// The BF16 products, `[pairs, n]`.
    Plain,
    /// The softcapped SwiGLU of each (gate, up) column pair, `[pairs, n / 2]`.
    Swiglu { gate_softcap: f32, up_softcap: f32 },
}

impl Experts {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "experts", cubin!("experts"))?;
        let matmul = load_module(ctx, "expert_matmul", cubin!("expert_matmul"))?;
        let gemm = [
            [
                matmul.load_function("expert_gemm_nvfp4")?,
                matmul.load_function("expert_gemm_nvfp4_swiglu")?,
            ],
            [
                matmul.load_function("expert_gemm_mxfp8")?,
                matmul.load_function("expert_gemm_mxfp8_swiglu")?,
            ],
        ];
        let gemm_q = [
            [
                matmul.load_function("expert_gemm_q_nvfp4")?,
                matmul.load_function("expert_gemm_q_nvfp4_swiglu")?,
            ],
            [
                matmul.load_function("expert_gemm_q_mxfp8")?,
                matmul.load_function("expert_gemm_q_mxfp8_swiglu")?,
            ],
        ];
        // MXFP8 needs the most: 3 stages of 16 KiB and 1 KiB of scales against NVFP4's 3 of
        // 12 KiB and 2 KiB of scales, and the quantized kernels' 2 of 16 KiB and 8 KiB.
        let shared = i32::try_from(
            ExpertFormat::Mxfp8
                .gemm_shared_bytes()
                .max(ExpertFormat::quantized_shared_bytes()),
        )?;
        for function in gemm.iter().chain(&gemm_q).flatten() {
            function.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            )?;
        }
        let gemm_q8 = [
            matmul.load_function("expert_gemm_q8_nvfp4")?,
            matmul.load_function("expert_gemm_q8_nvfp4_swiglu")?,
        ];
        for function in &gemm_q8 {
            function.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(ExpertFormat::fp8_shared_bytes())?,
            )?;
        }
        let mlp = [
            module.load_function("expert_mlp_nvfp4")?,
            module.load_function("expert_mlp_mxfp8")?,
        ];
        Ok(Self {
            matvec: MATVEC_KERNELS
                .iter()
                .map(|name| module.load_function(name))
                .collect::<Result<_, _>>()?,
            mlp_blocks: [
                mlp_blocks(ctx, &mlp[0], MLP_BLOCKS_PER_SM[0])?,
                mlp_blocks(ctx, &mlp[1], MLP_BLOCKS_PER_SM[1])?,
            ],
            mlp,
            group: matmul.load_function("expert_group")?,
            gemm,
            gemm_q,
            gemm_q8,
            quantize: [
                matmul.load_function("quantize_rows_nvfp4")?,
                matmul.load_function("quantize_rows_mxfp8")?,
                matmul.load_function("quantize_rows_fp8")?,
            ],
            scale_spread: matmul.load_function("nvfp4_scale_spread")?,
        })
    }

    /// Multiplies each row's input by the weights of its 8 selected experts, writing BF16
    /// `[rows, 8, n]` to `out`.
    ///
    /// `indices` is `[rows, 8]`. `x` is BF16 `[rows, 1, k]`, one input shared by all 8 slots
    /// (`w13`), or `[rows, 8, k]`, one input per slot (`w2`). Each id must be below the expert
    /// count of `weights`; an id outside it gives a NaN row. k must be 2304 or 1152, and n a
    /// multiple of 16 (NVFP4, k = 2304) to 32 (NVFP4, k = 1152).
    ///
    /// Equals MLX's `mx.gather_qmm` up to F32 summation order.
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
            inputs_per_row,
        } = weights.check(x.len(), indices.len(), out.len())?;
        let &ExpertWeights {
            format,
            weight,
            scales,
            n,
            k,
        } = weights;
        let k_index = MATVEC_K
            .iter()
            .position(|&supported| supported == k)
            .with_context(|| format!("expert_matvec supports k in {MATVEC_K:?}, got {k}"))?;
        let block_rows = format.matvec_block_rows(k);
        ensure!(
            n.is_multiple_of(block_rows),
            "expert_matvec needs n a multiple of {block_rows} for k = {k}, got {n}"
        );
        if rows == 0 {
            return Ok(());
        }
        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .ptr(&indices, stream)
            .ptr(weight, stream)
            .ptr(scales, stream);
        global_scale_arg(&mut args, format, stream);
        args.ptr_mut(&mut out, stream)
            .i32(i32::try_from(n).context("n is too large")?)
            .i32(inputs_per_row)
            .i32(i32::try_from(experts).context("too many experts")?);
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(n / block_rows).context("n is too large")?,
                u32::try_from(rows * SLOTS).context("too many rows")?,
                1,
            ),
            block_dim: (u32::try_from(MATVEC_WARPS * 32)?, 1, 1),
            shared_mem_bytes: 0,
        };
        let function = &self.matvec[2 * format.index() + k_index];
        // SAFETY: the arguments match `expert_matvec_*(const bf16*, const int*, const u8*,
        // const u8*, const float*, bf16*, int, int, int)`, and the kernel's k is the weights'
        // k. `check` kept every input row inside `x`, every output inside `out`, and every
        // weight and scale row of an expert below `experts` inside `weight` and `scales`; the
        // kernel skips other ids, and the grid covers exactly n rows. Only the NVFP4 kernels
        // read `global_scale`, which then holds `experts` values. The kernel reads nothing
        // before `pdl_wait`.
        unsafe { args.launch_pdl(stream, function, config) }
    }

    /// The decode expert MLP: `act = softcapped_swiglu(x @ w13)` for each row and slot, then
    /// `out = act @ w2`, in one kernel.
    ///
    /// `x` is BF16 `[rows, 2304]`, `indices` `[rows, 8]`, `act` `[rows, 8, 1152]` and `out`
    /// `[rows, 8, w2.n]`. `w13` has 2304 rows of input width 2304, interleaving gate (even)
    /// and up (odd) rows; `w2` has input width 1152 and n a multiple of 32 (NVFP4) or 16
    /// (MXFP8). Both use the same format and expert count. `sync` holds at least
    /// `1 + rows * 8` zeros, which the kernel leaves zero. Unknown ids give NaN rows in `act`
    /// and `out`.
    ///
    /// Equals [`Self::expert_matvec`], `Elementwise::softcapped_swiglu` and
    /// [`Self::expert_matvec`] again, bit for bit. The `w2` products of each slot start as
    /// soon as its activations are ready, so DRAM streams both projections without a gap.
    ///
    /// With `end`, which needs one row and `w2.n` = 2304, the kernel also ends the MoE block
    /// as `Conv::moe_conv_residual` does from `out`, bit for bit, and `sync` holds at least
    /// `1 + 8 + w2.n / 32` (NVFP4) or `1 + 8 + w2.n / 16` (MXFP8) zeros.
    #[expect(clippy::too_many_arguments, reason = "two projections and two caps")]
    pub fn expert_mlp(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        indices: &impl DeviceIn<i32>,
        w13: &ExpertWeights<'_>,
        w2: &ExpertWeights<'_>,
        act: &mut impl DeviceOut<bf16>,
        out: &mut impl DeviceOut<bf16>,
        sync: &mut impl DeviceOut<u32>,
        gate_softcap: f32,
        up_softcap: f32,
        end: Option<MoeEnd<'_>>,
    ) -> anyhow::Result<()> {
        let (x, indices, mut act, mut out, mut sync) = (
            x.view(),
            indices.view(),
            act.view_mut(),
            out.view_mut(),
            sync.view_mut(),
        );
        ensure!(
            w13.k == MLP_K13 && w13.n == 2 * MLP_K2 && w2.k == MLP_K2,
            "expert_mlp needs w13 of {} x {MLP_K13} and w2 of input width {MLP_K2}, got {} x {} \
             and {}",
            2 * MLP_K2,
            w13.n,
            w13.k,
            w2.k
        );
        let gate_up = w13.check(x.len(), indices.len(), 2 * act.len())?;
        ensure!(gate_up.inputs_per_row == 1, "x must hold one input per row");
        let down = w2.check(act.len(), indices.len(), out.len())?;
        ensure!(
            w13.format.index() == w2.format.index() && gate_up.experts == down.experts,
            "w13 and w2 must share format and expert count"
        );
        let pairs = gate_up.rows * SLOTS;
        ensure!(
            sync.len() > pairs,
            "sync has {} elements, expected at least {}",
            sync.len(),
            1 + pairs
        );
        let block_rows2 = w2.format.matvec_block_rows(MLP_K2);
        ensure!(
            w2.n.is_multiple_of(block_rows2),
            "expert_mlp needs w2's n a multiple of {block_rows2}, got {}",
            w2.n
        );
        if let Some(end) = &end {
            ensure!(
                pairs == SLOTS && w2.n == HIDDEN,
                "ending the MoE block needs one row and w2.n = {HIDDEN}, got {pairs} pairs \
                 and {}",
                w2.n
            );
            ensure!(
                sync.len() > pairs + w2.n / block_rows2,
                "sync has {} elements, expected at least {}",
                sync.len(),
                1 + pairs + w2.n / block_rows2
            );
            ensure!(
                end.weights.len() == SLOTS
                    && end.conv_weight.len() == CONV_TAPS * HIDDEN
                    && end.conv_state.len() == (CONV_TAPS - 1) * HIDDEN
                    && end.residual.len() == HIDDEN
                    && end.hidden.len() == HIDDEN,
                "the MoE block's end needs 8 weights, [2304, 4] conv taps, a [3, 2304] conv \
                 state, and 2304-wide residual and hidden"
            );
        }
        if pairs == 0 {
            return Ok(());
        }
        let tasks = pairs * (w13.n / w13.format.matvec_block_rows(MLP_K13) + w2.n / block_rows2);
        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .ptr(&indices, stream)
            .ptr(w13.weight, stream)
            .ptr(w13.scales, stream);
        global_scale_arg(&mut args, w13.format, stream);
        args.ptr(w2.weight, stream).ptr(w2.scales, stream);
        global_scale_arg(&mut args, w2.format, stream);
        args.ptr_mut(&mut act, stream)
            .ptr_mut(&mut out, stream)
            .i32(i32::try_from(w2.n).context("n is too large")?)
            .i32(i32::try_from(pairs).context("too many rows")?)
            .i32(i32::try_from(gate_up.experts).context("too many experts")?)
            .f32(gate_softcap)
            .f32(up_softcap)
            .ptr_mut(&mut sync, stream);
        match end {
            Some(end) => args
                .ptr(end.weights, stream)
                .ptr(end.conv_weight, stream)
                .ptr_mut(end.conv_state, stream)
                .ptr(end.residual, stream)
                .ptr_mut(end.hidden, stream),
            None => args.u64(0).u64(0).u64(0).u64(0).u64(0),
        };
        let blocks = tasks.min(self.mlp_blocks[w13.format.index()]);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(blocks)?, 1, 1),
            block_dim: (u32::try_from(MATVEC_WARPS * 32)?, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `expert_mlp_*(const bf16*, const int*, const u8*,
        // const u8*, const float*, const u8*, const u8*, const float*, bf16*, bf16*, int, int,
        // int, float, float, unsigned*, const float*, const bf16*, float*, const float*,
        // bf16*)`. The checks above give `x` one 2304-wide input per row, `act` 1152 values
        // and `out` w2.n values per pair, the weights the kernel's shapes and `experts`
        // experts each, and `sync` room for its 1 + pairs counters, and with `end` its
        // per-tile counters, zero as the kernel leaves them; `end`'s buffers have the lengths
        // checked above, and a null `hidden` makes the kernel read none of them. The grid has at most one block per task, and all
        // its blocks fit on the GPU at once (`mlp_blocks`), which the kernel's waits
        // need. Only the NVFP4 kernel reads the global scales, which then hold `experts`
        // values. The kernel reads nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.mlp[w13.format.index()], config) }
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
        self.group(stream, indices, experts, GEMM_PAIRS, scratch)
    }

    /// [`Self::expert_group`] for the quantized products
    /// ([`Self::expert_matmul_quantized_grouped`] and
    /// [`Self::expert_matmul_quantized_swiglu_grouped`]), whose tiles take 128 pairs.
    pub fn expert_group_quantized(
        &self,
        stream: &CudaStream,
        indices: &impl DeviceIn<i32>,
        experts: usize,
        scratch: &mut ExpertScratch,
    ) -> anyhow::Result<()> {
        self.group(stream, indices, experts, QUANTIZED_PAIRS, scratch)
    }

    fn group(
        &self,
        stream: &CudaStream,
        indices: &impl DeviceIn<i32>,
        experts: usize,
        tile_pairs: usize,
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
            let tile_arg = i32::try_from(tile_pairs)?;
            let mut launch = stream.launch_builder(&self.group);
            launch
                .arg(&indices)
                .arg(&pairs_arg)
                .arg(&experts_arg)
                .arg(&tile_arg)
                .arg(&mut scratch.perm)
                .arg(&mut scratch.tiles)
                .arg(&mut scratch.group_counts);
            let config = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (GROUP_THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the arguments match `expert_group(const int*, int, int, int, int*,
            // int4*, int*)`. `indices` holds `pairs` ids. The scratch was sized for at least
            // `rows` rows and `experts` experts, at most 2048 and 1024 as the kernel needs, so
            // `perm` holds `pairs` entries, `tiles` every tile the kernel can write with tiles
            // of 64 pairs or more, and `group_counts` 3 values.
            unsafe { launch.launch(config) }?;
        }
        scratch.grouping = Some(Grouping {
            rows,
            experts,
            tile_pairs,
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
        let input = GemmInput::Bf16(x.view());
        self.gemm(stream, &input, weights, out, scratch, Epilogue::Plain)
    }

    /// [`Self::expert_matmul_grouped`] on `w13` followed by `Elementwise::softcapped_swiglu`,
    /// in one kernel: `act` `[rows, 8, w13.n / 2]` from `x` `[rows, 1, k]`, where `w13`
    /// interleaves gate (even) and up (odd) rows.
    ///
    /// Equals the two calls bit for bit: the products round to BF16 before the SwiGLU, as
    /// the separate kernels store them.
    #[expect(
        clippy::too_many_arguments,
        reason = "the product's inputs and the two caps"
    )]
    pub fn expert_matmul_swiglu_grouped(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        w13: &ExpertWeights<'_>,
        act: &mut impl DeviceOut<bf16>,
        scratch: &ExpertScratch,
        gate_softcap: f32,
        up_softcap: f32,
    ) -> anyhow::Result<()> {
        let epilogue = Epilogue::Swiglu {
            gate_softcap,
            up_softcap,
        };
        self.gemm(
            stream,
            &GemmInput::Bf16(x.view()),
            w13,
            act,
            scratch,
            epilogue,
        )
    }

    /// Quantizes each row of `x` (BF16 `[rows, k]`) into `out` for the quantized products
    /// with weights of `format`: NVFP4 or MXFP8 as the checkpoint's activation quantization
    /// prescribes, following vLLM's Inkling kernels bit for bit (`kernels/expert_matmul.cu`
    /// describes the rules). k must be a multiple of 128.
    pub fn quantize_rows(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        k: usize,
        format: ExpertFormat<'_>,
        out: &mut QuantizedRows,
    ) -> anyhow::Result<()> {
        self.quantize(stream, x, k, RowFormat::of(format, false), out)
    }

    /// Quantizes each row of `x` (BF16 `[rows, k]`) to MXFP8 into `out` for the products
    /// with FP8 activations on weights of `format`: E4M3 values with a UE8M0 scale per 32, as
    /// [`Self::quantize_rows`] does for MXFP8 weights. For NVFP4 weights each block of 32
    /// values is stored in the order `expert_gemm_q8_*` loads it. k must be a multiple of
    /// 128.
    pub fn quantize_rows_fp8(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        k: usize,
        format: ExpertFormat<'_>,
        out: &mut QuantizedRows,
    ) -> anyhow::Result<()> {
        self.quantize(stream, x, k, RowFormat::of(format, true), out)
    }

    /// Counts the NVFP4 group scales in `scales` (E4M3 bytes, two per 32 weights) for which
    /// the products with FP8 activations would not be exact. They split each weight in two
    /// E4M3 values under one power-of-two scale per 32 weights, which is exact while the
    /// block's two scales are within a factor of 2^10 (`kernels/expert_matmul.cu` gives the
    /// exact rule). Waits for the GPU.
    pub fn fp8_inexact_scales(
        &self,
        stream: &Arc<CudaStream>,
        scales: &CudaSlice<u8>,
    ) -> anyhow::Result<u32> {
        ensure!(
            scales.len().is_multiple_of(2),
            "{} scale bytes, not whole blocks of two",
            scales.len()
        );
        let mut bad = stream.alloc_zeros::<u32>(1)?;
        let count = scales.len() / 2;
        if count > 0 {
            let count_arg = u64::try_from(count)?;
            let mut launch = stream.launch_builder(&self.scale_spread);
            launch.arg(scales).arg(&count_arg).arg(&mut bad);
            let config = LaunchConfig {
                grid_dim: (u32::try_from(count.div_ceil(256).min(4096))?, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the arguments match `nvfp4_scale_spread(const uint16_t*, size_t,
            // unsigned*)`; `scales` holds `count` pairs of bytes, which the kernel reads as
            // 16-bit values (device allocations are aligned), and `bad` one value.
            unsafe { launch.launch(config) }?;
        }
        Ok(stream.clone_dtoh(&bad)?[0])
    }

    fn quantize(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        k: usize,
        format: RowFormat,
        out: &mut QuantizedRows,
    ) -> anyhow::Result<()> {
        let x = x.view();
        ensure!(
            k > 0 && k.is_multiple_of(GEMM_K_MULTIPLE),
            "k = {k} is not a positive multiple of {GEMM_K_MULTIPLE}"
        );
        ensure!(
            x.len().is_multiple_of(k),
            "x has {} values, not whole rows of {k}",
            x.len()
        );
        let rows = x.len() / k;
        ensure!(
            x.len() <= out.values.len() && rows <= out.outer.len(),
            "{rows} rows of {k} do not fit in {} values and {} rows",
            out.values.len(),
            out.outer.len()
        );
        out.contents = None;
        if rows > 0 {
            let rows_arg = i32::try_from(rows).context("too many rows")?;
            let k_arg = i32::try_from(k).context("k is too large")?;
            let mut launch = stream.launch_builder(&self.quantize[format as usize]);
            launch
                .arg(&x)
                .arg(&rows_arg)
                .arg(&k_arg)
                .arg(&mut out.values)
                .arg(&mut out.scales)
                .arg(&mut out.outer);
            let config = LaunchConfig {
                grid_dim: (u32::try_from(rows.div_ceil(8))?, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the arguments match `quantize_rows_*(const bf16*, int, int, u8*, u8*,
            // float*)`. `x` holds `rows` rows of k values; `values` holds at least rows * k
            // bytes, `scales` rows * k / 16 and `outer` `rows` values, as much as either
            // kernel writes or more.
            unsafe { launch.launch(config) }?;
        }
        out.contents = Some(Quantized {
            rows,
            k,
            format,
            stream: stream_id(stream),
        });
        Ok(())
    }

    /// [`Self::expert_matmul_grouped`] with inputs quantized by [`Self::quantize_rows`] or
    /// [`Self::quantize_rows_fp8`]: multiplies the E2M1 or E4M3 inputs with the raw weights
    /// on the block-scaled FP4 or FP8 tensor cores, sums in F32, and rounds to BF16. `xq` must
    /// hold the grouping's rows (one input per row) or 8 times as many (one per pair), of the
    /// weights' k and format, queued on this stream. The grouping must come from
    /// [`Self::expert_group_quantized`].
    pub fn expert_matmul_quantized_grouped(
        &self,
        stream: &CudaStream,
        xq: &QuantizedRows,
        weights: &ExpertWeights<'_>,
        out: &mut impl DeviceOut<bf16>,
        scratch: &ExpertScratch,
    ) -> anyhow::Result<()> {
        let input = GemmInput::Quantized(xq);
        self.gemm(stream, &input, weights, out, scratch, Epilogue::Plain)
    }

    /// [`Self::expert_matmul_swiglu_grouped`] with quantized inputs, as
    /// [`Self::expert_matmul_quantized_grouped`] multiplies them.
    #[expect(
        clippy::too_many_arguments,
        reason = "the product's inputs and the two caps"
    )]
    pub fn expert_matmul_quantized_swiglu_grouped(
        &self,
        stream: &CudaStream,
        xq: &QuantizedRows,
        w13: &ExpertWeights<'_>,
        act: &mut impl DeviceOut<bf16>,
        scratch: &ExpertScratch,
        gate_softcap: f32,
        up_softcap: f32,
    ) -> anyhow::Result<()> {
        let epilogue = Epilogue::Swiglu {
            gate_softcap,
            up_softcap,
        };
        self.gemm(
            stream,
            &GemmInput::Quantized(xq),
            w13,
            act,
            scratch,
            epilogue,
        )
    }

    /// Launches an `expert_gemm_*` kernel for [`Self::expert_matmul_grouped`] and
    /// [`Self::expert_matmul_swiglu_grouped`].
    fn gemm(
        &self,
        stream: &CudaStream,
        input: &GemmInput<'_, '_>,
        weights: &ExpertWeights<'_>,
        out: &mut impl DeviceOut<bf16>,
        scratch: &ExpertScratch,
        epilogue: Epilogue,
    ) -> anyhow::Result<()> {
        let mut out = out.view_mut();
        let x_len = input.len(weights, stream)?;
        let grouping = scratch
            .grouping
            .context("expert_matmul_grouped needs a grouping from expert_group")?;
        // `check` takes the product's width; the SwiGLU halves it.
        let (out_per_value, swiglu, (gate_softcap, up_softcap)) = match epilogue {
            Epilogue::Plain => (1, 0, (0.0, 0.0)),
            Epilogue::Swiglu {
                gate_softcap,
                up_softcap,
            } => (2, 1, (gate_softcap, up_softcap)),
        };
        let Shape {
            experts,
            rows,
            inputs_per_row,
        } = weights.check(x_len, grouping.rows * SLOTS, out_per_value * out.len())?;
        ensure!(
            swiglu == 0 || inputs_per_row == 1,
            "the SwiGLU product takes one input per row"
        );
        ensure!(
            experts == grouping.experts,
            "weights have {experts} experts, the grouping {}",
            grouping.experts
        );
        ensure!(
            grouping.stream == stream_id(stream),
            "the grouping was queued on another stream"
        );
        let tile_pairs = match input {
            GemmInput::Bf16(_) => GEMM_PAIRS,
            GemmInput::Quantized(_) => QUANTIZED_PAIRS,
        };
        ensure!(
            grouping.tile_pairs == tile_pairs,
            "the grouping has tiles of {} pairs, the product needs {tile_pairs}; group with \
             expert_group for BF16 inputs and expert_group_quantized for quantized ones",
            grouping.tile_pairs
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
        let (function, shared, threads) = self.gemm_function(input, format, swiglu);
        let null: u64 = 0;
        // One more row of blocks than tiles: the row at the tile count writes the NaN rows.
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(n.div_ceil(GEMM_COLUMNS))?,
                u32::try_from(max_tiles(pairs, experts) + 1)?,
                1,
            ),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: u32::try_from(shared)?,
        };
        let mut launch = stream.launch_builder(function);
        match input {
            GemmInput::Bf16(x) => launch.arg(x),
            GemmInput::Quantized(xq) => launch.arg(&xq.values).arg(&xq.scales).arg(&xq.outer),
        };
        launch.arg(weight).arg(scales);
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
            .arg(&inputs_per_row)
            .arg(&gate_softcap)
            .arg(&up_softcap);
        // SAFETY: the arguments match `expert_gemm_*(const bf16*, const u8*, const u8*,
        // const float*, bf16*, const int*, const int4*, const int*, int, int, int, float,
        // float)`, or for quantized inputs `expert_gemm_q_*` and `expert_gemm_q8_*(const u8*,
        // const u8*, const float*, ...)` with the same remaining parameters; the paired rows
        // go only to `expert_gemm_q8_nvfp4*`, with NVFP4 weights, as `len` checked. `quantize_rows` wrote
        // `x_len / k` rows of k values of the weights' format into `xq`, so its values,
        // scales and outer scales cover every input row. The `_swiglu` kernels write n / 2
        // values per pair, which `out` holds.
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
    /// The `expert_gemm_*` kernel for `input` and weights of `format`, plain (`swiglu` 0) or
    /// with the SwiGLU epilogue (1), with its dynamic shared memory and threads per block.
    fn gemm_function(
        &self,
        input: &GemmInput<'_, '_>,
        format: ExpertFormat<'_>,
        swiglu: usize,
    ) -> (&CudaFunction, usize, u32) {
        match *input {
            GemmInput::Bf16(_) => (
                &self.gemm[format.index()][swiglu],
                format.gemm_shared_bytes(),
                GEMM_THREADS,
            ),
            GemmInput::Quantized(xq) => {
                if xq.contents.map(|c| c.format) == Some(RowFormat::Fp8Paired) {
                    (
                        &self.gemm_q8[swiglu],
                        ExpertFormat::fp8_shared_bytes(),
                        FP8_GEMM_THREADS,
                    )
                } else {
                    (
                        &self.gemm_q[format.index()][swiglu],
                        ExpertFormat::quantized_shared_bytes(),
                        GEMM_THREADS,
                    )
                }
            }
        }
    }
}

/// The blocks an `expert_mlp` kernel runs: `per_sm` per SM, and never more than fit on the
/// GPU at once.
fn mlp_blocks(
    ctx: &Arc<CudaContext>,
    kernel: &CudaFunction,
    per_sm: usize,
) -> anyhow::Result<usize> {
    let sms = ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let resident = kernel.occupancy_max_active_blocks_per_multiprocessor(
        u32::try_from(MATVEC_WARPS * 32)?,
        0,
        None,
    )?;
    let per_sm = per_sm.min(usize::try_from(resident)?);
    ensure!(per_sm > 0, "expert_mlp does not fit on an SM");
    Ok(per_sm * usize::try_from(sms)?)
}

/// The global scale argument of an expert kernel: the NVFP4 scales, or a null pointer for
/// MXFP8, which has none.
fn global_scale_arg(args: &mut KernelArgs, format: ExpertFormat<'_>, stream: &CudaStream) {
    match format {
        ExpertFormat::Nvfp4 { global_scale } => args.ptr(global_scale, stream),
        ExpertFormat::Mxfp8 => args.u64(0),
    };
}

/// Identifies a stream for [`Grouping`].
fn stream_id(stream: &CudaStream) -> usize {
    stream.cu_stream().addr()
}
