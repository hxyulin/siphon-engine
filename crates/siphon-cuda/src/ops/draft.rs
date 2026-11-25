//! The DSpark draft's small kernels (`kernels/draft.cu`): the tap mix, norms, RoPE, the
//! draft cache, block attention, the Markov head and the LM-head fold. The products run
//! through [`crate::ops::linear::Linear`]. [`crate::draft::Draft`] puts them together.

use std::sync::Arc;

use anyhow::{Context as _, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _,
    DevicePtrMut as _, LaunchConfig, PushKernelArg as _, sys::CUfunction_attribute,
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module};
use crate::ops::linear::launch_by_value;
use crate::weights::draft::{HEAD_DIM, HIDDEN, KV_HEADS, RANK, TAPS};

/// Rows of a draft block: the anchor and 7 mask rows.
pub const BLOCK: usize = 8;
/// Width of the keys or values of one position: 8 heads of 128.
pub const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
/// Query heads per KV head of the bundled draft and of 2DSD's DFlash draft.
const GROUP: usize = 3;
/// Query heads per KV head of DFlash 2, the most a block attention takes.
const MAX_GROUP: usize = 4;
/// Most splits of one block attention.
pub const MAX_ATTENTION_SPLITS: usize = 128;
/// Fewest keys one attention split takes, a multiple of the kernel's chunk of 32.
const MIN_KEYS_PER_SPLIT: usize = 128;
/// Blocks of the first argmax pass over the vocabulary in [`DraftKernels::markov_logits`].
pub const LOGIT_BLOCKS: usize = 148;
/// Blocks of [`DraftKernels::markov_fold`]: two per SM of the RTX 5090.
pub const FOLD_BLOCKS: usize = 340;
/// Partials an [`ArgmaxPartials`] holds.
const MAX_PARTIALS: usize = 512;
/// Dynamic shared memory of the block attention with `group` query heads per KV head,
/// `attention_smem` in the kernel.
const fn attention_smem(group: usize) -> u32 {
    (4 * BLOCK * group * HEAD_DIM
        + 2 * 2 * 32 * (HEAD_DIM + 8)
        + 2 * 2 * 32 * HEAD_DIM
        + 16 * BLOCK * 32) as u32
}
/// Candidates the DFlash 2 selector keeps per draft row, `kTopK` in the kernel.
pub const TOP_K: usize = 16;
/// Values per block of the first top-k pass, `kTopkSpan` in the kernel.
const TOPK_SPAN: usize = 2048;
/// Rows of DFlash 2's convolution coefficients: 2 sides of 2 taps of 144 groups.
const CONV_COEFFICIENTS: usize = 576;
/// Products one [`DraftKernels::gemm_fp8`] launch takes, `kF8MaxSegments` in the kernel.
pub const F8_MAX_PRODUCTS: usize = 10;
/// Input rows of [`DraftKernels::gemm_fp8`], `kF8MaxM` in the kernel.
const F8_MAX_M: usize = 8;
/// Weight rows per group of `draft_gemm_fp8`: 8 warps of `kF8Rows`.
const F8_ROWS_PER_BLOCK: usize = 16;
/// Dynamic shared memory of `draft_gemm_fp8`, `kF8Smem` in the kernel.
const F8_SMEM: u32 = 49152;
/// Blocks of `draft_gemm_fp8` at most: two per SM of the RTX 5090.
const F8_BLOCKS: usize = 340;
/// Blocks of [`DraftKernels::markov_step`], `kStepBlocks` in the kernel.
const STEP_BLOCKS: u32 = 32;
/// Rows the fold kernel covers per block, `kFoldRows` in the kernel.
const FOLD_ROWS: usize = 16;
/// Threads of the row kernels, `kRowThreads` in the kernel.
const ROW_THREADS: u32 = 256;

#[derive(Debug)]
pub struct DraftKernels {
    mix_rms_norm: CudaFunction,
    add_rms_norm: CudaFunction,
    block_input: CudaFunction,
    head_norm_rope: CudaFunction,
    store_kv: CudaFunction,
    attention: CudaFunction,
    attention_g4: CudaFunction,
    attention_fp8: CudaFunction,
    attention_combine: CudaFunction,
    silu_mul: CudaFunction,
    markov_context_input: CudaFunction,
    markov_init: CudaFunction,
    markov_gate: CudaFunction,
    markov_logits: CudaFunction,
    markov_pick: CudaFunction,
    fold_weight: CudaFunction,
    fold_bias: CudaFunction,
    set_control: CudaFunction,
    set_control_kept: CudaFunction,
    commit_mix: CudaFunction,
    commit_rope_store: CudaFunction,
    block_rope_store: CudaFunction,
    markov_step: CudaFunction,
    markov_fold: CudaFunction,
    quantize_rows: CudaFunction,
    gather_rows: CudaFunction,
    set_i32: CudaFunction,
    fold_error: CudaFunction,
    markov_bound: CudaFunction,
    markov_exact: CudaFunction,
    gemm_fp8: CudaFunction,
    markov_fold_any: CudaFunction,
    markov_fold_raw: CudaFunction,
    dflash_commit_rope_store: CudaFunction,
    dflash_step: CudaFunction,
    conv: CudaFunction,
    conv_add_rms_norm: CudaFunction,
    topk_partial: CudaFunction,
    topk_merge: CudaFunction,
    select: CudaFunction,
}

/// A matrix `[rows, k]` of E4M3 values with one F32 scale per row, `amax / 448`, as
/// [`crate::ops::linear_fp8::Fp8Matrix`] holds them, in two buffers.
pub struct Fp8Rows {
    pub values: CudaSlice<u8>,
    pub scales: CudaSlice<f32>,
    pub k: usize,
}

impl Fp8Rows {
    /// Zeroed room for `rows` rows of `k` values.
    pub fn new(stream: &Arc<CudaStream>, rows: usize, k: usize) -> anyhow::Result<Self> {
        ensure!(k.is_multiple_of(16), "k = {k} is not a multiple of 16");
        Ok(Self {
            values: stream.alloc_zeros(rows * k)?,
            scales: stream.alloc_zeros(rows)?,
            k,
        })
    }

    /// Rows the buffers hold.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.scales.len()
    }

    /// Device bytes the matrix takes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.values.len() + 4 * self.scales.len()
    }
}

/// One product of [`DraftKernels::gemm_fp8`]: `out[i, n] = scale[n] * x[i] · w[n]` for the
/// first `rows` rows of `w`, with output rows `ld` elements apart.
pub struct Fp8Product<'a> {
    pub w: &'a Fp8Rows,
    pub rows: usize,
    pub out: CudaViewMut<'a, bf16>,
    pub ld: usize,
}

/// The fold weights [`DraftKernels::markov_fold_any`] reads.
#[derive(Clone, Copy)]
pub enum FoldRows<'a> {
    Bf16(&'a CudaSlice<bf16>),
    Fp8(&'a Fp8Rows),
}

/// Kernel parameter: one product of `draft_gemm_fp8`, as `F8Segment` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct F8Segment {
    w: u64,
    scale: u64,
    out: u64,
    rows: i32,
    ld: i32,
    block_end: i32,
}

/// Kernel parameter of `draft_gemm_fp8`, as `F8Batch` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct F8Batch {
    seg: [F8Segment; F8_MAX_PRODUCTS],
    x: u64,
    count: i32,
    m: i32,
    k: i32,
}

/// Most layers of a DFlash draft, `kDflashMaxLayers` in the kernel.
pub const DFLASH_MAX_LAYERS: usize = 16;

/// Kernel parameter of `draft_dflash_commit_rope_store`, as `DflashCommit` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DflashCommit {
    ctx_k: u64,
    ctx_v: u64,
    control: u64,
    inv_freq: u64,
    k_norm: [u64; DFLASH_MAX_LAYERS],
    cache_k: [u64; DFLASH_MAX_LAYERS],
    cache_v: [u64; DFLASH_MAX_LAYERS],
    slots: [i32; DFLASH_MAX_LAYERS],
    eps: f32,
    ring: i32,
    rope: i32,
    fp8_layers: i32,
}

/// The RoPE arithmetic of the draft kernels, `kRope*` in the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rope {
    /// F32 cosine, sine and arithmetic, rounded once (vLLM's bundled draft).
    F32 = 0,
    /// BF16 cosine and sine with F32 arithmetic (the 2DSD runner).
    Bf16Table = 1,
    /// BF16 cosine and sine, each product and the sum rounded to BF16 (SGLang's
    /// `table_qk_norm_rope` with a BF16 table, and Transformers in BF16).
    Bf16Ops = 2,
}

/// Where a block attention takes its keys from.
#[derive(Clone, Copy)]
pub enum AttentionKeys<'a> {
    /// Positions `first..first + keys`.
    Host { first: usize, keys: usize },
    /// The block of the round whose inputs `control` holds (see
    /// [`DraftKernels::set_control`]), after `window` positions (0: every position).
    /// `max_keys` bounds the key count, which sizes the grid. With `row_window` above
    /// zero, block row `j` at `P + j` sees only keys at or after `P + j - (row_window -
    /// 1)`, a sliding window of `row_window` positions that ends at the row.
    Device {
        control: &'a CudaSlice<i32>,
        window: usize,
        row_window: usize,
        max_keys: usize,
    },
}

/// The split outputs of [`DraftKernels::attention`].
pub struct AttentionScratch {
    out: CudaSlice<f32>,
    stats: CudaSlice<f32>,
}

impl AttentionScratch {
    pub fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let rows = MAX_ATTENTION_SPLITS * KV_HEADS * BLOCK * MAX_GROUP;
        Ok(Self {
            out: stream.alloc_zeros(rows * HEAD_DIM)?,
            stats: stream.alloc_zeros(rows * 2)?,
        })
    }
}

/// The first-pass partials of [`DraftKernels::topk`] for up to 7 rows of up to
/// 262,144 values each.
pub struct TopkScratch {
    values: CudaSlice<f32>,
    ids: CudaSlice<i32>,
}

impl TopkScratch {
    pub fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let n = (BLOCK - 1) * TOPK_SPAN;
        Ok(Self {
            values: stream.alloc_zeros(n)?,
            ids: stream.alloc_zeros(n)?,
        })
    }
}

/// The bounds of the pruned exact Markov argmax (see [`DraftKernels::markov_bound`]).
pub struct LogitBounds {
    /// Each row's upper bound, F32 `[vocab]`.
    upper: CudaSlice<f32>,
    /// Each block's largest lower bound, F32 `[FOLD_BLOCKS]`.
    lower: CudaSlice<f32>,
}

impl LogitBounds {
    pub fn new(stream: &Arc<CudaStream>, vocab: usize) -> anyhow::Result<Self> {
        Ok(Self {
            upper: stream.alloc_zeros(vocab)?,
            lower: stream.alloc_zeros(FOLD_BLOCKS)?,
        })
    }
}

/// The first-pass results of the Markov argmax, and each block's sum of
/// `exp(logit - softcap)` for the confidence.
pub struct ArgmaxPartials {
    value: CudaSlice<f32>,
    index: CudaSlice<i32>,
    sum: CudaSlice<f32>,
}

impl ArgmaxPartials {
    pub fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        Ok(Self {
            value: stream.alloc_zeros(MAX_PARTIALS)?,
            index: stream.alloc_zeros(MAX_PARTIALS)?,
            sum: stream.alloc_zeros(MAX_PARTIALS)?,
        })
    }
}

/// One layer's draft cache: keys and values, `[slots, 8, 128]` each, as BF16 or with
/// `fp8` as E4M3 of the BF16 values (see `put` in the kernel). Position `p` sits in slot
/// `p % slots`.
pub struct KvCache {
    /// The keys' and values' bytes.
    pub k: CudaSlice<u8>,
    pub v: CudaSlice<u8>,
    pub slots: usize,
    pub fp8: bool,
}

impl KvCache {
    /// A BF16 cache.
    pub fn new(stream: &Arc<CudaStream>, slots: usize) -> anyhow::Result<Self> {
        Self::with_format(stream, slots, false)
    }

    /// A cache of BF16 values, or of E4M3 values with `fp8`.
    pub fn with_format(stream: &Arc<CudaStream>, slots: usize, fp8: bool) -> anyhow::Result<Self> {
        ensure!(slots > 0, "a cache needs at least one slot");
        let bytes = slots * KV_WIDTH * if fp8 { 1 } else { 2 };
        Ok(Self {
            k: stream.alloc_zeros(bytes)?,
            v: stream.alloc_zeros(bytes)?,
            slots,
            fp8,
        })
    }

    /// Device bytes of the keys and values.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.k.len() + self.v.len()
    }
}

fn row_config(rows: usize) -> anyhow::Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
        block_dim: (ROW_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// One block of `threads` threads.
fn one_block(threads: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (threads as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn check_len(name: &str, len: usize, want: usize) -> anyhow::Result<()> {
    ensure!(len == want, "{name} has {len} elements, expected {want}");
    Ok(())
}

fn rows_of(name: &str, len: usize, width: usize) -> anyhow::Result<usize> {
    ensure!(
        len.is_multiple_of(width),
        "{name} has {len} elements, not a multiple of {width}"
    );
    Ok(len / width)
}

fn i32_arg(name: &str, v: usize) -> anyhow::Result<i32> {
    i32::try_from(v).with_context(|| format!("{name} = {v} does not fit an i32"))
}

/// `start` as a kernel argument, checking that every position up to `start + rows` fits.
fn start_arg(start: usize, rows: usize) -> anyhow::Result<i32> {
    i32_arg("last position", start + rows)?;
    i32_arg("start", start)
}

/// Elements of a round's control buffer: the first position the round commits, the
/// number of positions it commits and the anchor token.
pub const CONTROL: usize = 3;

/// Splits of the block attention over `keys` keys, and the keys of each split.
#[must_use]
pub fn attention_splits(keys: usize) -> (usize, usize) {
    let per_split = keys
        .div_ceil(MAX_ATTENTION_SPLITS)
        .max(MIN_KEYS_PER_SPLIT)
        .next_multiple_of(32);
    (keys.div_ceil(per_split), per_split)
}

impl DraftKernels {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "draft", cubin!("draft"))?;
        let f = |name: &str| module.load_function(name);
        let attention = f("draft_attention")?;
        let attention_g4 = f("draft_attention_g4")?;
        let attention_fp8 = f("draft_attention_fp8")?;
        for (func, group) in [
            (&attention, GROUP),
            (&attention_g4, MAX_GROUP),
            (&attention_fp8, GROUP),
        ] {
            func.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(attention_smem(group))?,
            )
            .context("raising the shared memory of the block attention")?;
        }
        Ok(Self {
            mix_rms_norm: f("draft_mix_rms_norm")?,
            add_rms_norm: f("draft_add_rms_norm")?,
            block_input: f("draft_block_input")?,
            head_norm_rope: f("draft_head_norm_rope")?,
            store_kv: f("draft_store_kv")?,
            attention,
            attention_g4,
            attention_fp8,
            attention_combine: f("draft_attention_combine")?,
            silu_mul: f("draft_silu_mul")?,
            markov_context_input: f("draft_markov_context_input")?,
            markov_init: f("draft_markov_init")?,
            markov_gate: f("draft_markov_gate")?,
            markov_logits: f("draft_markov_logits")?,
            markov_pick: f("draft_markov_pick")?,
            fold_weight: f("draft_fold_weight")?,
            fold_bias: f("draft_fold_bias")?,
            set_control: f("draft_set_control")?,
            set_control_kept: f("draft_set_control_kept")?,
            commit_mix: f("draft_commit_mix")?,
            commit_rope_store: f("draft_commit_rope_store")?,
            block_rope_store: f("draft_block_rope_store")?,
            markov_step: f("draft_markov_step")?,
            markov_fold: f("draft_markov_fold")?,
            quantize_rows: f("draft_quantize_rows")?,
            set_i32: f("draft_set_i32")?,
            fold_error: f("draft_fold_error")?,
            markov_bound: f("draft_markov_bound")?,
            markov_exact: f("draft_markov_exact")?,
            gather_rows: f("draft_gather_rows")?,
            gemm_fp8: f("draft_gemm_fp8")?,
            markov_fold_any: f("draft_markov_fold_any")?,
            markov_fold_raw: f("draft_markov_fold_raw")?,
            dflash_commit_rope_store: f("draft_dflash_commit_rope_store")?,
            dflash_step: f("draft_dflash_step")?,
            conv: f("draft_conv")?,
            conv_add_rms_norm: f("draft_conv_add_rms_norm")?,
            topk_partial: f("draft_topk_partial")?,
            topk_merge: f("draft_topk_merge")?,
            select: f("draft_select")?,
        })
    }

    /// `out[r] = rms_norm(Σ_j alpha[j] taps[r, j]; weight)` for taps BF16 `[rows, 4, 2304]`
    /// and `alpha` F32 `[4]`, into `out` BF16 `[rows, 2304]`. Sums and normalizes in F32.
    pub fn mix_rms_norm(
        &self,
        stream: &CudaStream,
        taps: &impl DeviceIn<bf16>,
        alpha: &CudaView<'_, f32>,
        weight: &CudaSlice<bf16>,
        out: &mut impl DeviceOut<bf16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let (taps, mut out) = (taps.view(), out.view_mut());
        let rows = rows_of("taps", taps.len(), TAPS * HIDDEN)?;
        check_len("alpha", alpha.len(), TAPS)?;
        check_len("weight", weight.len(), HIDDEN)?;
        check_len("out", out.len(), rows * HIDDEN)?;
        if rows == 0 {
            return Ok(());
        }
        let mut launch = stream.launch_builder(&self.mix_rms_norm);
        launch
            .arg(&taps)
            .arg(alpha)
            .arg(weight)
            .arg(&mut out)
            .arg(&eps);
        // SAFETY: the arguments match `draft_mix_rms_norm(const bf16*, const float*, const
        // bf16*, bf16*, float)`, and the checks above size every buffer for `rows` blocks.
        unsafe { launch.launch(row_config(rows)?) }?;
        Ok(())
    }

    /// For each row of BF16 `[rows, 2304]` matrices: `sum = residual + x` in F32,
    /// `residual = bf16(sum)` and `out = rms_norm(sum; weight)`. Without `x`, normalizes
    /// `residual` and leaves it as it is.
    pub fn add_rms_norm(
        &self,
        stream: &CudaStream,
        x: Option<&CudaSlice<bf16>>,
        residual: &mut CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let rows = rows_of("residual", residual.len(), HIDDEN)?;
        if let Some(x) = x {
            check_len("x", x.len(), residual.len())?;
        }
        check_len("weight", weight.len(), HIDDEN)?;
        check_len("out", out.len(), residual.len())?;
        let null = 0u64;
        let mut launch = stream.launch_builder(&self.add_rms_norm);
        match x {
            Some(x) => launch.arg(x),
            None => launch.arg(&null),
        };
        launch.arg(residual).arg(weight).arg(out).arg(&eps);
        // SAFETY: the arguments match `draft_add_rms_norm(const bf16*, bf16*, const bf16*,
        // bf16*, float)`, with a null `x` passed as a zero pointer, which the kernel checks.
        // The buffers hold `rows` rows.
        unsafe { launch.launch(row_config(rows)?) }?;
        Ok(())
    }

    /// The block's input rows, BF16 `[8, 2304]`: row 0 is `rms_norm(embed[anchor];
    /// embed_norm)` and rows 1 to 7 are `mask_embed`. `anchor` holds one token id.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn block_input(
        &self,
        stream: &CudaStream,
        anchor: &impl DeviceIn<i32>,
        embed: &CudaSlice<bf16>,
        embed_norm: &CudaSlice<bf16>,
        mask_embed: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let anchor = anchor.view();
        check_len("anchor", anchor.len(), 1)?;
        let vocab = i32_arg("vocab", rows_of("embed", embed.len(), HIDDEN)?)?;
        check_len("embed_norm", embed_norm.len(), HIDDEN)?;
        check_len("mask_embed", mask_embed.len(), HIDDEN)?;
        check_len("out", out.len(), BLOCK * HIDDEN)?;
        let mut launch = stream.launch_builder(&self.block_input);
        launch
            .arg(&anchor)
            .arg(embed)
            .arg(&vocab)
            .arg(embed_norm)
            .arg(mask_embed)
            .arg(out)
            .arg(&eps);
        // SAFETY: the arguments match `draft_block_input(const int*, const bf16*, int, const
        // bf16*, const bf16*, bf16*, float)`. The kernel bounds-checks the anchor against
        // `vocab` and writes the 8 rows of `out`, one block each.
        unsafe { launch.launch(row_config(BLOCK)?) }?;
        Ok(())
    }

    /// In place on the leading `heads` heads of each of the `rows` rows of `x` (row stride
    /// `row_stride` elements): per-head RMSNorm with `weight` (`[128]`), rounded to BF16,
    /// then NeoX RoPE at position `start + row` with the frequencies `inv_freq` (F32
    /// `[64]`).
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn head_norm_rope(
        &self,
        stream: &CudaStream,
        x: &mut impl DeviceOut<bf16>,
        rows: usize,
        heads: usize,
        row_stride: usize,
        weight: &CudaSlice<bf16>,
        inv_freq: &CudaSlice<f32>,
        start: usize,
        eps: f32,
    ) -> anyhow::Result<()> {
        self.head_norm_rope_table(
            stream,
            x,
            rows,
            heads,
            row_stride,
            weight,
            inv_freq,
            start,
            eps,
            Rope::F32,
        )
    }

    /// [`Self::head_norm_rope`] with the RoPE arithmetic `rope`.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn head_norm_rope_table(
        &self,
        stream: &CudaStream,
        x: &mut impl DeviceOut<bf16>,
        rows: usize,
        heads: usize,
        row_stride: usize,
        weight: &CudaSlice<bf16>,
        inv_freq: &CudaSlice<f32>,
        start: usize,
        eps: f32,
        rope: Rope,
    ) -> anyhow::Result<()> {
        let mut x = x.view_mut();
        ensure!(
            heads * HEAD_DIM <= row_stride,
            "{heads} heads exceed the row stride"
        );
        if rows == 0 {
            return Ok(());
        }
        ensure!(
            x.len() >= (rows - 1) * row_stride + heads * HEAD_DIM,
            "x has {} elements, too few for {rows} rows of stride {row_stride}",
            x.len()
        );
        check_len("weight", weight.len(), HEAD_DIM)?;
        check_len("inv_freq", inv_freq.len(), HEAD_DIM / 2)?;
        let row_stride = i32_arg("row_stride", row_stride)?;
        let start = start_arg(start, rows)?;
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(rows).context("too many rows")?,
                u32::try_from(heads).context("too many heads")?,
                1,
            ),
            block_dim: ((HEAD_DIM / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let rope = rope as i32;
        let mut launch = stream.launch_builder(&self.head_norm_rope);
        launch
            .arg(&mut x)
            .arg(&row_stride)
            .arg(weight)
            .arg(inv_freq)
            .arg(&start)
            .arg(&eps)
            .arg(&rope);
        // SAFETY: the arguments match `draft_head_norm_rope(bf16*, int, const bf16*, const
        // float*, int, float, int)`. The check above keeps every head of every row inside `x`, and
        // every position fits an i32.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Copies the `rows` rows of `k` and `v` (BF16, row stride `row_stride` elements, the
    /// first 1024 of each row) into `cache` at positions `start..start + rows`.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn store_kv(
        &self,
        stream: &CudaStream,
        k: &impl DeviceIn<bf16>,
        v: &impl DeviceIn<bf16>,
        rows: usize,
        row_stride: usize,
        cache: &mut KvCache,
        start: usize,
    ) -> anyhow::Result<()> {
        let (k, v) = (k.view(), v.view());
        if rows == 0 {
            return Ok(());
        }
        ensure!(
            row_stride >= KV_WIDTH && row_stride.is_multiple_of(8),
            "row stride {row_stride} must be a multiple of 8 of at least {KV_WIDTH}"
        );
        let need = (rows - 1) * row_stride + KV_WIDTH;
        ensure!(
            k.len() >= need && v.len() >= need,
            "k and v have {} and {} elements, need {need}",
            k.len(),
            v.len()
        );
        ensure!(
            rows <= cache.slots,
            "{rows} rows overflow {} slots",
            cache.slots
        );
        let slots = i32_arg("slots", cache.slots)?;
        let start = start_arg(start, rows)?;
        let row_stride = i32_arg("row_stride", row_stride)?;
        let fp8 = i32::from(cache.fp8);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: ((KV_WIDTH / 8) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.store_kv);
        launch
            .arg(&k)
            .arg(&v)
            .arg(&row_stride)
            .arg(&mut cache.k)
            .arg(&mut cache.v)
            .arg(&start)
            .arg(&slots)
            .arg(&fp8);
        // SAFETY: the arguments match `draft_store_kv(const bf16*, const bf16*, int, void*,
        // void*, int, int, int)`. The checks above keep the 16-byte row reads inside `k` and
        // `v`, and every slot index is below `slots`, inside the caches of `fp8`'s element.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Bidirectional attention of the block's queries `q` (BF16 `[8, heads * 128]`, 24 or
    /// 32 heads) over the keys at positions `first..first + keys` of `cache`, into `out`
    /// (the shape of `q`). Scale 128^-0.5, F32 softmax.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn attention(
        &self,
        stream: &CudaStream,
        q: &CudaSlice<bf16>,
        cache: &KvCache,
        first: usize,
        keys: usize,
        scratch: &mut AttentionScratch,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        self.attention_from(
            stream,
            q,
            cache,
            AttentionKeys::Host { first, keys },
            scratch,
            out,
        )
    }

    /// [`Self::attention`] with the keys from `keys`. The split count depends only on the
    /// number of keys, so the result does not depend on where that number comes from.
    pub fn attention_from(
        &self,
        stream: &CudaStream,
        q: &CudaSlice<bf16>,
        cache: &KvCache,
        keys: AttentionKeys<'_>,
        scratch: &mut AttentionScratch,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        let group = q.len() / (BLOCK * KV_WIDTH);
        ensure!(
            q.len() == BLOCK * group * KV_WIDTH && (group == GROUP || group == MAX_GROUP),
            "q has {} elements, not 8 rows of 24 or 32 heads",
            q.len()
        );
        check_len("out", out.len(), q.len())?;
        let slots = i32_arg("slots", cache.slots)?;
        let (control, window, row_window, first, count, max_keys) = match keys {
            AttentionKeys::Host { first, keys } => {
                ensure!(keys > 0, "attention needs at least one key");
                ensure!(
                    keys <= cache.slots,
                    "{keys} keys do not fit {} slots",
                    cache.slots
                );
                (
                    None,
                    0,
                    0,
                    start_arg(first, keys)?,
                    i32_arg("keys", keys)?,
                    keys,
                )
            }
            AttentionKeys::Device {
                control,
                window,
                row_window,
                max_keys,
            } => {
                check_len("control", control.len(), CONTROL)?;
                ensure!(
                    max_keys <= cache.slots,
                    "{max_keys} keys do not fit {} slots",
                    cache.slots
                );
                (
                    Some(control),
                    i32_arg("window", window)?,
                    i32_arg("row_window", row_window)?,
                    0,
                    0,
                    max_keys,
                )
            }
        };
        // The splits of the largest key count bound the splits of every smaller one.
        let grid_splits = max_keys
            .div_ceil(MIN_KEYS_PER_SPLIT)
            .clamp(1, MAX_ATTENTION_SPLITS);
        let config = LaunchConfig {
            grid_dim: (KV_HEADS as u32, u32::try_from(grid_splits)?, 1),
            block_dim: ((BLOCK * 32) as u32, 1, 1),
            shared_mem_bytes: attention_smem(group),
        };
        let mut args = KernelArgs::default();
        args.ptr(q, stream)
            .ptr(&cache.k, stream)
            .ptr(&cache.v, stream)
            .i32(slots)
            .opt(control, stream)
            .i32(window)
            .i32(row_window)
            .i32(first)
            .i32(count)
            .ptr_mut(&mut scratch.out, stream)
            .ptr_mut(&mut scratch.stats, stream);
        let attention = match (group == GROUP, cache.fp8) {
            (true, false) => &self.attention,
            (true, true) => &self.attention_fp8,
            (false, false) => &self.attention_g4,
            (false, true) => bail!("no E4M3 block attention with 4 query heads per KV head"),
        };
        // SAFETY: the arguments match `draft_attention(const bf16*, const void*, const void*,
        // int, const int*, int, int, int, int, float*, float*)`, `draft_attention_fp8` and
        // `draft_attention_g4`, picked by the head count of `q` and the cache's element. Every key position maps to a slot below
        // `slots`; the grid covers the splits of `max_keys` keys, at most
        // MAX_ATTENTION_SPLITS, which the scratch holds for 4 query heads per KV head, and
        // blocks past the split count of the actual keys exit. The kernel reads earlier
        // outputs only after `pdl_wait`.
        unsafe { args.launch_pdl(stream, attention, config) }?;

        let config = LaunchConfig {
            grid_dim: (KV_HEADS as u32, (BLOCK * group) as u32, 1),
            block_dim: (HEAD_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&scratch.out, stream)
            .ptr(&scratch.stats, stream)
            .opt(control, stream)
            .i32(window)
            .i32(first)
            .i32(count)
            .i32(i32_arg("group", group)?)
            .ptr_mut(out, stream);
        // SAFETY: the arguments match `draft_attention_combine(const float*, const float*,
        // const int*, int, int, int, int, bf16*)`; the first launch wrote all the splits the
        // combine reads, and `out` holds the 8 rows of 8 * group heads.
        unsafe { args.launch_pdl(stream, &self.attention_combine, config) }?;
        Ok(())
    }

    /// `out = bf16(silu(gate) * up)` elementwise, in F32.
    pub fn silu_mul(
        &self,
        stream: &CudaStream,
        gate: &CudaSlice<bf16>,
        up: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        let n = out.len();
        check_len("gate", gate.len(), n)?;
        check_len("up", up.len(), n)?;
        let blocks = n.div_ceil(256).clamp(1, 1 << 16);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(blocks)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.silu_mul);
        launch.arg(gate).arg(up).arg(out).arg(&n);
        // SAFETY: the arguments match `draft_silu_mul(const bf16*, const bf16*, bf16*,
        // size_t)`, and all three buffers hold `n` elements.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// The inputs of the Markov context product, F32 `[7, 4608]`: row `i` is
    /// `[y[i + 1]; f]` widened, for `y` BF16 `[8, 2304]` and `f` BF16 `[2304]`.
    pub fn markov_context_input(
        &self,
        stream: &CudaStream,
        y: &CudaSlice<bf16>,
        f: &CudaSlice<bf16>,
        out: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        check_len("y", y.len(), BLOCK * HIDDEN)?;
        check_len("f", f.len(), HIDDEN)?;
        check_len("out", out.len(), (BLOCK - 1) * 2 * HIDDEN)?;
        let mut launch = stream.launch_builder(&self.markov_context_input);
        launch.arg(y).arg(f).arg(out);
        // SAFETY: the arguments match `draft_markov_context_input(const bf16*, const bf16*,
        // float*)`, with one block per output row, inside the checked sizes.
        unsafe { launch.launch(row_config(BLOCK - 1)?) }?;
        Ok(())
    }

    /// Starts the Markov loop: `step_in` (F32 `[512]`) becomes `[0; w1[anchor]]`.
    pub fn markov_init(
        &self,
        stream: &CudaStream,
        anchor: &impl DeviceIn<i32>,
        w1: &CudaSlice<bf16>,
        step_in: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        let anchor = anchor.view();
        check_len("anchor", anchor.len(), 1)?;
        check_len("step_in", step_in.len(), 2 * RANK)?;
        let vocab = i32_arg("vocab", rows_of("w1", w1.len(), RANK)?)?;
        let mut launch = stream.launch_builder(&self.markov_init);
        launch.arg(&anchor).arg(w1).arg(&vocab).arg(step_in);
        // SAFETY: the arguments match `draft_markov_init(const int*, const bf16*, int,
        // float*)`; the kernel bounds-checks the anchor and writes the 512 inputs.
        unsafe { launch.launch(one_block(RANK)) }?;
        Ok(())
    }

    /// One Markov gate from the F32 `[768]` products `raw` (`[state; w1 token]`) and
    /// `context` (`[y; f]`) and the bias: replaces the state in `step_in[..256]` and
    /// writes the latent (F32 `[256]`).
    pub fn markov_gate(
        &self,
        stream: &CudaStream,
        raw: &CudaSlice<f32>,
        context: &CudaView<'_, f32>,
        bias: &CudaSlice<f32>,
        step_in: &mut CudaSlice<f32>,
        latent: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        for (name, len) in [
            ("raw", raw.len()),
            ("context", context.len()),
            ("bias", bias.len()),
        ] {
            check_len(name, len, 3 * RANK)?;
        }
        check_len("step_in", step_in.len(), 2 * RANK)?;
        check_len("latent", latent.len(), RANK)?;
        let mut launch = stream.launch_builder(&self.markov_gate);
        launch
            .arg(raw)
            .arg(context)
            .arg(bias)
            .arg(step_in)
            .arg(latent);
        // SAFETY: the arguments match `draft_markov_gate(const float*, const float*, const
        // float*, float*, float*)`, and every buffer has the checked size.
        unsafe { launch.launch(one_block(RANK)) }?;
        Ok(())
    }

    /// The Markov logits of one step over the first `vocab` columns and the first pass of
    /// their argmax: `bf16(cap tanh(bf16(base + bf16(fold_bias + acc)) / cap))`, written to
    /// `logits` if given. `base` is BF16, `fold_bias` BF16 and `acc` F32, each with at least
    /// `vocab` elements.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn markov_logits(
        &self,
        stream: &CudaStream,
        base: &CudaView<'_, bf16>,
        fold_bias: &CudaSlice<bf16>,
        acc: &CudaSlice<f32>,
        vocab: usize,
        softcap: f32,
        logits: Option<CudaViewMut<'_, bf16>>,
        partials: &mut ArgmaxPartials,
    ) -> anyhow::Result<()> {
        ensure!(
            base.len() >= vocab && fold_bias.len() >= vocab && acc.len() >= vocab,
            "base, fold_bias and acc have {}, {} and {} elements, fewer than {vocab}",
            base.len(),
            fold_bias.len(),
            acc.len()
        );
        let mut logits = logits;
        if let Some(logits) = logits.as_ref() {
            check_len("logits", logits.len(), vocab)?;
        }
        let vocab_arg = i32_arg("vocab", vocab)?;
        let null = 0u64;
        let config = LaunchConfig {
            grid_dim: (LOGIT_BLOCKS as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.markov_logits);
        launch
            .arg(base)
            .arg(fold_bias)
            .arg(acc)
            .arg(&vocab_arg)
            .arg(&softcap);
        match logits.as_mut() {
            Some(logits) => launch.arg(logits),
            None => launch.arg(&null),
        };
        launch.arg(&mut partials.value).arg(&mut partials.index);
        // SAFETY: the arguments match `draft_markov_logits(const bf16*, const bf16*, const
        // float*, int, float, bf16*, float*, int*)`, with no logits passed as a zero pointer.
        // The kernel reads and writes columns below `vocab` and one partial per block.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// The second argmax pass: writes the step's draft id to `id` (one element) and puts
    /// its `w1` row into `step_in[256..]`.
    pub fn markov_pick(
        &self,
        stream: &CudaStream,
        partials: &ArgmaxPartials,
        id: &mut impl DeviceOut<i32>,
        w1: &CudaSlice<bf16>,
        step_in: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        let mut id = id.view_mut();
        check_len("id", id.len(), 1)?;
        check_len("step_in", step_in.len(), 2 * RANK)?;
        let count = i32_arg("partials", LOGIT_BLOCKS)?;
        let mut launch = stream.launch_builder(&self.markov_pick);
        launch
            .arg(&partials.value)
            .arg(&partials.index)
            .arg(&count)
            .arg(&mut id)
            .arg(w1)
            .arg(step_in);
        // SAFETY: the arguments match `draft_markov_pick(const float*, const int*, int, int*,
        // const bf16*, float*)`. The picked index is a column of the logits, below the
        // vocabulary, so its `w1` row exists.
        unsafe { launch.launch(one_block(RANK)) }?;
        Ok(())
    }

    /// `fold_weight = bf16(lm_head · hidden_out)` and `fold_bias = bf16(lm_head ·
    /// hidden_out_bias)`, summed in F64 and rounded through F32 as vLLM folds. `lm_head` is
    /// BF16 `[rows, 2304]`, `hidden_out` `[2304, 256]`.
    pub fn fold(
        &self,
        stream: &CudaStream,
        lm_head: &CudaSlice<bf16>,
        hidden_out: &CudaSlice<bf16>,
        hidden_out_bias: &CudaSlice<bf16>,
        fold_weight: &mut CudaSlice<bf16>,
        fold_bias: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        let rows = rows_of("lm_head", lm_head.len(), HIDDEN)?;
        check_len("hidden_out", hidden_out.len(), HIDDEN * RANK)?;
        check_len("hidden_out_bias", hidden_out_bias.len(), HIDDEN)?;
        check_len("fold_weight", fold_weight.len(), rows * RANK)?;
        check_len("fold_bias", fold_bias.len(), rows)?;
        let rows_arg = i32_arg("rows", rows)?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows.div_ceil(FOLD_ROWS))?, 1, 1),
            block_dim: (RANK as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.fold_weight);
        launch
            .arg(lm_head)
            .arg(&rows_arg)
            .arg(hidden_out)
            .arg(fold_weight);
        // SAFETY: the arguments match `draft_fold_weight(const bf16*, int, const bf16*,
        // bf16*)`; the kernel skips rows at or past `rows`.
        unsafe { launch.launch(config) }?;

        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows.div_ceil(8))?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.fold_bias);
        launch
            .arg(lm_head)
            .arg(&rows_arg)
            .arg(hidden_out_bias)
            .arg(fold_bias);
        // SAFETY: the arguments match `draft_fold_bias(const bf16*, int, const bf16*, bf16*)`;
        // one warp per row, skipping rows at or past `rows`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Writes a round's inputs to `control` (I32 `[3]`): `[start, rows, anchor]`, with the
    /// anchor read from `anchor` (one I32 on the device).
    pub fn set_control(
        &self,
        stream: &CudaStream,
        control: &mut CudaSlice<i32>,
        start: usize,
        rows: usize,
        anchor: &impl DeviceIn<i32>,
    ) -> anyhow::Result<()> {
        let anchor = anchor.view();
        check_len("control", control.len(), CONTROL)?;
        check_len("anchor", anchor.len(), 1)?;
        ensure!(rows <= BLOCK, "{rows} rows exceed a block");
        let start = start_arg(start, rows + BLOCK)?;
        let rows = i32_arg("rows", rows)?;
        let mut launch = stream.launch_builder(&self.set_control);
        launch.arg(control).arg(&start).arg(&rows).arg(&anchor);
        // SAFETY: the arguments match `draft_set_control(int*, int, int, const int*)`; one
        // thread writes the three elements.
        unsafe { launch.launch(one_block(1)) }?;
        Ok(())
    }

    /// Writes a round's inputs to `control` after a verify of `max_rows` rows that kept
    /// `kept[0]` drafts: `[start, min(kept + 1, max_rows), anchor]`, with `kept` and the
    /// anchor read from the device.
    pub fn set_control_kept(
        &self,
        stream: &CudaStream,
        control: &mut CudaSlice<i32>,
        start: usize,
        kept: &CudaView<'_, i32>,
        max_rows: usize,
        anchor: &impl DeviceIn<i32>,
    ) -> anyhow::Result<()> {
        let anchor = anchor.view();
        check_len("control", control.len(), CONTROL)?;
        check_len("kept", kept.len(), 1)?;
        check_len("anchor", anchor.len(), 1)?;
        ensure!(
            (1..=BLOCK).contains(&max_rows),
            "{max_rows} rows, expected 1 to {BLOCK}"
        );
        let start = start_arg(start, max_rows + BLOCK)?;
        let max_rows = i32_arg("rows", max_rows)?;
        let mut args = KernelArgs::default();
        args.ptr_mut(control, stream)
            .i32(start)
            .ptr(kept, stream)
            .i32(max_rows)
            .ptr(&anchor, stream);
        // SAFETY: the arguments match `draft_set_control_kept(int*, int, const int*, int,
        // const int*)`; one thread reads the two inputs after `pdl_wait` and writes the three
        // elements.
        unsafe { args.launch_pdl(stream, &self.set_control_kept, one_block(1)) }?;
        Ok(())
    }

    /// The tap mixes of a round's committed rows (see `draft_commit_mix`): for each of the
    /// first `control[1]` rows of `taps` (BF16 `[8, 4, 2304]`), the mixes of the 3 layers
    /// into `mixed` (BF16 `[3, 8, 2304]`), and the last row's Markov mix into `f`. `alpha`
    /// is F32 `[4, 4]` and `norms` the 4 RMSNorm weights.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn commit_mix(
        &self,
        stream: &CudaStream,
        taps: &CudaSlice<bf16>,
        control: &CudaSlice<i32>,
        alpha: &CudaSlice<f32>,
        norms: [&CudaSlice<bf16>; TAPS],
        mixed: &mut CudaSlice<bf16>,
        f: &mut CudaSlice<bf16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        check_len("taps", taps.len(), BLOCK * TAPS * HIDDEN)?;
        check_len("control", control.len(), CONTROL)?;
        check_len("alpha", alpha.len(), TAPS * TAPS)?;
        for norm in norms {
            check_len("norm", norm.len(), HIDDEN)?;
        }
        check_len("mixed", mixed.len(), 3 * BLOCK * HIDDEN)?;
        check_len("f", f.len(), HIDDEN)?;
        let mut args = KernelArgs::default();
        args.ptr(taps, stream)
            .ptr(control, stream)
            .ptr(alpha, stream);
        for norm in norms {
            args.ptr(norm, stream);
        }
        args.ptr_mut(mixed, stream).ptr_mut(f, stream).f32(eps);
        let config = LaunchConfig {
            grid_dim: (BLOCK as u32, TAPS as u32, 1),
            block_dim: (ROW_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `draft_commit_mix(const bf16*, const int*, const
        // float*, const bf16* x4, bf16*, bf16*, float)`; the buffers have the checked sizes,
        // and the kernel clamps the row count to 8.
        unsafe { args.launch_pdl(stream, &self.commit_mix, config) }?;
        Ok(())
    }

    /// Norm, RoPE and cache stores of a round's context keys and values (see
    /// `draft_commit_rope_store`): `ctx_k` and `ctx_v` are BF16 `[3, 8, 1024]`, the
    /// projections of `mixed` by each layer's `k_proj_ctx` and `v_proj_ctx`.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn commit_rope_store(
        &self,
        stream: &CudaStream,
        ctx_k: &CudaSlice<bf16>,
        ctx_v: &CudaSlice<bf16>,
        control: &CudaSlice<i32>,
        inv_freq: &CudaSlice<f32>,
        eps: f32,
        k_norms: [&CudaSlice<bf16>; 3],
        caches: &mut [KvCache],
    ) -> anyhow::Result<()> {
        check_len("ctx_k", ctx_k.len(), 3 * BLOCK * KV_WIDTH)?;
        check_len("ctx_v", ctx_v.len(), 3 * BLOCK * KV_WIDTH)?;
        check_len("control", control.len(), CONTROL)?;
        check_len("inv_freq", inv_freq.len(), HEAD_DIM / 2)?;
        for norm in k_norms {
            check_len("k_norm", norm.len(), HEAD_DIM)?;
        }
        ensure!(caches.len() == 3, "{} caches, expected 3", caches.len());
        let mut args = KernelArgs::default();
        args.ptr(ctx_k, stream)
            .ptr(ctx_v, stream)
            .ptr(control, stream)
            .ptr(inv_freq, stream)
            .f32(eps);
        for norm in k_norms {
            args.ptr(norm, stream);
        }
        let mut fp8_layers = 0;
        for (layer, cache) in caches.iter_mut().enumerate() {
            let slots = i32_arg("slots", cache.slots)?;
            args.ptr_mut(&mut cache.k, stream)
                .ptr_mut(&mut cache.v, stream)
                .i32(slots);
            fp8_layers |= i32::from(cache.fp8) << layer;
        }
        args.i32(fp8_layers);
        let config = LaunchConfig {
            grid_dim: (BLOCK as u32, KV_HEADS as u32, 3),
            block_dim: ((HEAD_DIM / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `draft_commit_rope_store`'s parameters in order. The
        // kernel writes cache slots `position mod slots` of each cache, inside it, and skips
        // positions past layer 1's slots and below zero.
        unsafe { args.launch_pdl(stream, &self.commit_rope_store, config) }?;
        Ok(())
    }

    /// Norm and RoPE of the block's queries `q` (BF16 `[8, heads * 128]`, in place) and
    /// keys `k` (BF16 `[8, 1024]`) at the round's position, and the cache stores of the
    /// keys and the values `v` (see `draft_block_rope_store`). With `limit`, positions at
    /// or past the cache's slots are skipped, for a cache that is not a ring. `rope` picks
    /// the RoPE arithmetic.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn block_rope_store(
        &self,
        stream: &CudaStream,
        q: &mut CudaSlice<bf16>,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
        control: &CudaSlice<i32>,
        inv_freq: &CudaSlice<f32>,
        eps: f32,
        q_norm: &CudaSlice<bf16>,
        k_norm: &CudaSlice<bf16>,
        cache: &mut KvCache,
        limit: bool,
        rope: Rope,
    ) -> anyhow::Result<()> {
        let q_heads = rows_of("q", q.len(), BLOCK * HEAD_DIM)?;
        ensure!(
            q_heads == KV_HEADS * GROUP || q_heads == KV_HEADS * MAX_GROUP,
            "{q_heads} query heads, expected 24 or 32"
        );
        check_len("k", k.len(), BLOCK * KV_WIDTH)?;
        check_len("v", v.len(), BLOCK * KV_WIDTH)?;
        check_len("control", control.len(), CONTROL)?;
        check_len("inv_freq", inv_freq.len(), HEAD_DIM / 2)?;
        check_len("q_norm", q_norm.len(), HEAD_DIM)?;
        check_len("k_norm", k_norm.len(), HEAD_DIM)?;
        let slots = i32_arg("slots", cache.slots)?;
        let mut args = KernelArgs::default();
        args.ptr_mut(q, stream)
            .ptr(k, stream)
            .ptr(v, stream)
            .ptr(control, stream)
            .ptr(inv_freq, stream)
            .f32(eps)
            .ptr(q_norm, stream)
            .ptr(k_norm, stream)
            .ptr_mut(&mut cache.k, stream)
            .ptr_mut(&mut cache.v, stream)
            .i32(slots)
            .i32(i32::from(limit))
            .i32(rope as i32)
            .i32(i32_arg("q_heads", q_heads)?)
            .i32(i32::from(cache.fp8));
        let config = LaunchConfig {
            grid_dim: (BLOCK as u32, u32::try_from(q_heads + KV_HEADS)?, 1),
            block_dim: ((HEAD_DIM / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `draft_block_rope_store(bf16*, const bf16*, const
        // bf16*, const int*, const float*, float, const bf16*, const bf16*, void*, void*, int,
        // int, int, int, int)`. The buffers have the checked sizes, the grid has one block per
        // query head of `q` and per KV head, and every cache slot index is below `slots`.
        unsafe { args.launch_pdl(stream, &self.block_rope_store, config) }?;
        Ok(())
    }

    /// One Markov step of a round of `steps` (see `draft_markov_step`), or with
    /// `step == steps` only the pick of the last step's id. `context` is the step's row of
    /// the `[y; f]` product, F32 `[768]`. `state_in` and `state_out` are F32 `[256]`. The
    /// pick's softmax probability goes to `confidence`, F32 like `ids`.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn markov_step(
        &self,
        stream: &CudaStream,
        step: usize,
        steps: usize,
        partials: &ArgmaxPartials,
        softcap: f32,
        control: &CudaSlice<i32>,
        ids: &mut CudaSlice<i32>,
        confidence: &mut CudaSlice<f32>,
        w1: &CudaSlice<bf16>,
        rnn_state: &CudaSlice<bf16>,
        context: &CudaView<'_, f32>,
        bias: &CudaSlice<f32>,
        state_in: &CudaSlice<f32>,
        state_out: &mut CudaSlice<f32>,
        latent: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        ensure!(
            step <= steps && steps <= ids.len() && confidence.len() == ids.len(),
            "step {step} of {steps} with {} ids and {} confidences",
            ids.len(),
            confidence.len()
        );
        check_len("control", control.len(), CONTROL)?;
        let vocab = i32_arg("vocab", rows_of("w1", w1.len(), RANK)?)?;
        check_len("rnn_state", rnn_state.len(), 3 * RANK * 2 * RANK)?;
        check_len("context", context.len(), 3 * RANK)?;
        check_len("bias", bias.len(), 3 * RANK)?;
        check_len("state_in", state_in.len(), RANK)?;
        check_len("state_out", state_out.len(), RANK)?;
        check_len("latent", latent.len(), RANK)?;
        let mut args = KernelArgs::default();
        args.i32(1)
            .i32(i32_arg("step", step)?)
            .i32(i32_arg("steps", steps)?)
            .ptr(&partials.value, stream)
            .ptr(&partials.index, stream)
            .ptr(&partials.sum, stream)
            .i32(i32_arg("partials", FOLD_BLOCKS)?)
            .f32(softcap)
            .ptr_mut(confidence, stream)
            .ptr(control, stream)
            .ptr_mut(ids, stream)
            .ptr(w1, stream)
            .i32(vocab)
            .ptr(rnn_state, stream)
            .ptr(context, stream)
            .ptr(bias, stream)
            .ptr(state_in, stream)
            .ptr_mut(state_out, stream)
            .ptr_mut(latent, stream);
        let config = LaunchConfig {
            grid_dim: (STEP_BLOCKS, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `draft_markov_step(int, int, int, const float*, const
        // int*, const float*, int, float, float*, const int*, int*, const bf16*, int, const
        // bf16*, const float*, const float*, const float*, float*, float*)`. Before
        // `pdl_wait` it reads only `rnn_state`, which no kernel writes. At step 0 the anchor
        // is bounds-checked; later tokens are argmax columns below the vocabulary. It writes
        // ids and confidences below `steps`.
        unsafe { args.launch_pdl(stream, &self.markov_step, config) }?;
        Ok(())
    }

    /// One Markov step's logits and the first pass of their argmax, fused with the product
    /// `fold_weight · latent` (see `draft_markov_fold`). `base` and `fold_bias` hold at least
    /// `vocab` elements, `fold_weight` at least `vocab` rows of 256. With `reverse`, the
    /// kernel reads the rows from the end. Writes the logits to `logits` if given.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn markov_fold(
        &self,
        stream: &CudaStream,
        reverse: bool,
        fold_weight: &CudaSlice<bf16>,
        fold_bias: &CudaSlice<bf16>,
        base: &CudaView<'_, bf16>,
        latent: &CudaSlice<f32>,
        vocab: usize,
        softcap: f32,
        logits: Option<CudaViewMut<'_, bf16>>,
        partials: &mut ArgmaxPartials,
    ) -> anyhow::Result<()> {
        ensure!(
            base.len() >= vocab && fold_bias.len() >= vocab && fold_weight.len() >= vocab * RANK,
            "base, fold_bias and fold_weight are too short for {vocab} columns"
        );
        check_len("latent", latent.len(), RANK)?;
        let mut logits = logits;
        if let Some(logits) = logits.as_ref() {
            check_len("logits", logits.len(), vocab)?;
        }
        let mut args = KernelArgs::default();
        args.i32(1)
            .i32(i32::from(reverse))
            .ptr(fold_weight, stream)
            .ptr(fold_bias, stream)
            .ptr(base, stream)
            .ptr(latent, stream)
            .i32(i32_arg("vocab", vocab)?)
            .f32(softcap)
            .opt_mut(logits.as_mut(), stream)
            .ptr_mut(&mut partials.value, stream)
            .ptr_mut(&mut partials.index, stream)
            .ptr_mut(&mut partials.sum, stream);
        let config = LaunchConfig {
            grid_dim: (FOLD_BLOCKS as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `draft_markov_fold(int, int, const bf16*, const bf16*,
        // const bf16*, const float*, int, float, bf16*, float*, int*, float*)`. Before `pdl_wait` it
        // reads only `fold_weight`. It reads rows and columns below `vocab` and writes one
        // partial per block, FOLD_BLOCKS of them, which the partials hold.
        unsafe { args.launch_pdl(stream, &self.markov_fold, config) }?;
        Ok(())
    }

    /// Quantizes `rows` rows of `src` (BF16 `[*, k]`) into rows `first..` of `dst`: row
    /// `r` comes from source row `ids[r]`, or `r` without `ids`.
    pub fn quantize_rows(
        &self,
        stream: &CudaStream,
        src: &CudaSlice<bf16>,
        ids: Option<&CudaView<'_, i32>>,
        rows: usize,
        first: usize,
        dst: &mut Fp8Rows,
    ) -> anyhow::Result<()> {
        let k = dst.k;
        let src_rows = rows_of("src", src.len(), k)?;
        ensure!(first + rows <= dst.rows(), "rows past the destination");
        match ids {
            Some(ids) => check_len("ids", ids.len(), rows)?,
            None => ensure!(rows <= src_rows, "rows past the source"),
        }
        if rows == 0 {
            return Ok(());
        }
        let null = 0u64;
        let k_arg = i32_arg("k", k)?;
        let first = i32_arg("first", first)?;
        let mut launch = stream.launch_builder(&self.quantize_rows);
        launch.arg(src).arg(&k_arg);
        match ids {
            Some(ids) => launch.arg(ids),
            None => launch.arg(&null),
        };
        launch.arg(&first).arg(&mut dst.values).arg(&mut dst.scales);
        // SAFETY: the arguments match `draft_quantize_rows(const bf16*, int, const int*, int,
        // uint8_t*, float*)`. One block per row writes destination rows below `first + rows`,
        // inside `dst`; the caller's ids are source rows (checked without ids).
        unsafe { launch.launch(row_config(rows)?) }?;
        Ok(())
    }

    /// `dst[first + r] = src[ids[r]]` for rows of `k` BF16 elements.
    pub fn gather_rows(
        &self,
        stream: &CudaStream,
        src: &CudaSlice<bf16>,
        k: usize,
        ids: &CudaView<'_, i32>,
        first: usize,
        dst: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        let rows = ids.len();
        rows_of("src", src.len(), k)?;
        ensure!((first + rows) * k <= dst.len(), "rows past the destination");
        if rows == 0 {
            return Ok(());
        }
        let k_arg = i32_arg("k", k)?;
        let first = i32_arg("first", first)?;
        let mut launch = stream.launch_builder(&self.gather_rows);
        launch.arg(src).arg(&k_arg).arg(ids).arg(&first).arg(dst);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows)?, 1, 1),
            block_dim: (u32::try_from(k.clamp(1, 256))?, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `draft_gather_rows(const bf16*, int, const int*, int,
        // bf16*)`; destination rows stay inside `dst`, and the caller's ids are rows of `src`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Up to 10 products with E4M3 weights that share the input `x` (BF16 `[m, k]`, `m` up
    /// to 8) in one launch: BF16 outputs, F32 accumulation, each row's scale applied to
    /// its sum (see `draft_gemm_fp8`).
    pub fn gemm_fp8(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, bf16>,
        m: usize,
        products: &mut [Fp8Product<'_>],
    ) -> anyhow::Result<()> {
        ensure!(
            (1..=F8_MAX_PRODUCTS).contains(&products.len()),
            "{} products, expected 1 to {F8_MAX_PRODUCTS}",
            products.len()
        );
        ensure!(
            (1..=F8_MAX_M).contains(&m),
            "m = {m}, expected 1 to {F8_MAX_M}"
        );
        let k = products[0].w.k;
        check_len("x", x.len(), m * k)?;
        let mut batch = F8Batch {
            x: x.device_ptr(stream).0,
            count: i32::try_from(products.len())?,
            m: i32::try_from(m)?,
            k: i32_arg("k", k)?,
            ..F8Batch::default()
        };
        let mut blocks = 0;
        for (i, p) in products.iter_mut().enumerate() {
            ensure!(p.w.k == k, "product {i} has k = {}, expected {k}", p.w.k);
            ensure!(
                p.rows > 0 && p.rows <= p.w.rows(),
                "product {i}: bad row count"
            );
            ensure!(
                p.ld >= p.rows && p.out.len() >= (m - 1) * p.ld + p.rows,
                "product {i}: output too short"
            );
            blocks += p.rows.div_ceil(F8_ROWS_PER_BLOCK);
            batch.seg[i] = F8Segment {
                w: p.w.values.device_ptr(stream).0,
                scale: p.w.scales.device_ptr(stream).0,
                out: p.out.device_ptr_mut(stream).0,
                rows: i32_arg("rows", p.rows)?,
                ld: i32_arg("ld", p.ld)?,
                block_end: i32_arg("blocks", blocks)?,
            };
        }
        let config = LaunchConfig {
            grid_dim: (u32::try_from(blocks.min(F8_BLOCKS))?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: F8_SMEM,
        };
        // SAFETY: `batch` has the kernel's `F8Batch` layout. The checks above size each
        // weight as at least [rows, k], x as [m, k] and each output for m rows `ld` apart;
        // the kernel reads only weight rows below `rows` and writes only rows below `rows`
        // of the first m output rows. It reads x only after `pdl_wait`.
        unsafe { launch_by_value(stream, &self.gemm_fp8, config, &batch, true) }?;
        Ok(())
    }

    /// One Markov step's logits and the first pass of their argmax over FP8 or shortlist
    /// head rows (see `draft_markov_fold_any`): `rows` rows, or `*count` with `count`, row
    /// `n` standing for token `map[n]` with `map`.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn markov_fold_any(
        &self,
        stream: &CudaStream,
        weight: FoldRows<'_>,
        fold_bias: &CudaSlice<bf16>,
        base: &CudaView<'_, bf16>,
        latent: &CudaSlice<f32>,
        rows: usize,
        count: Option<&CudaSlice<i32>>,
        map: Option<&CudaSlice<i32>>,
        softcap: f32,
        partials: &mut ArgmaxPartials,
    ) -> anyhow::Result<()> {
        let capacity = match weight {
            FoldRows::Bf16(w) => rows_of("fold weight", w.len(), RANK)?,
            FoldRows::Fp8(w) => {
                ensure!(w.k == RANK, "FP8 fold rows have k = {}", w.k);
                w.rows()
            }
        };
        ensure!(
            rows <= capacity && rows <= fold_bias.len() && rows <= base.len(),
            "{rows} rows exceed the fold, its bias or the base logits"
        );
        if let Some(map) = map {
            ensure!(map.len() >= rows, "the map is too short");
        }
        let mut args = KernelArgs::default();
        match weight {
            FoldRows::Bf16(w) => args.ptr(w, stream).u64(0),
            FoldRows::Fp8(w) => args.ptr(&w.values, stream).ptr(&w.scales, stream),
        };
        args.ptr(fold_bias, stream)
            .ptr(base, stream)
            .ptr(latent, stream)
            .i32(i32_arg("rows", rows)?)
            .opt(count, stream)
            .opt(map, stream)
            .f32(softcap)
            .ptr_mut(&mut partials.value, stream)
            .ptr_mut(&mut partials.index, stream)
            .ptr_mut(&mut partials.sum, stream);
        let config = LaunchConfig {
            grid_dim: (FOLD_BLOCKS as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `draft_markov_fold_any(const void*, const float*, const
        // bf16*, const bf16*, const float*, int, const int*, const int*, float, float*,
        // int*, float*)`. With `count`, the caller keeps *count at most `rows`, which the buffers
        // hold; the kernel writes FOLD_BLOCKS partials.
        unsafe { args.launch_pdl(stream, &self.markov_fold_any, config) }?;
        Ok(())
    }

    /// [`Self::markov_fold_any`] without the softcap (see `draft_markov_fold_raw`): the
    /// partials' values are the uncapped logits and their sums zero.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn markov_fold_raw(
        &self,
        stream: &CudaStream,
        weight: FoldRows<'_>,
        fold_bias: &CudaSlice<bf16>,
        base: &CudaView<'_, bf16>,
        latent: &CudaSlice<f32>,
        rows: usize,
        count: Option<&CudaSlice<i32>>,
        map: Option<&CudaSlice<i32>>,
        partials: &mut ArgmaxPartials,
    ) -> anyhow::Result<()> {
        let capacity = match weight {
            FoldRows::Bf16(w) => rows_of("fold weight", w.len(), RANK)?,
            FoldRows::Fp8(w) => {
                ensure!(w.k == RANK, "FP8 fold rows have k = {}", w.k);
                w.rows()
            }
        };
        ensure!(
            rows <= capacity && rows <= fold_bias.len() && rows <= base.len(),
            "{rows} rows exceed the fold, its bias or the base logits"
        );
        check_len("latent", latent.len(), RANK)?;
        if let Some(map) = map {
            ensure!(map.len() >= rows, "the map is too short");
        }
        let mut args = KernelArgs::default();
        match weight {
            FoldRows::Bf16(w) => args.ptr(w, stream).u64(0),
            FoldRows::Fp8(w) => args.ptr(&w.values, stream).ptr(&w.scales, stream),
        };
        args.ptr(fold_bias, stream)
            .ptr(base, stream)
            .ptr(latent, stream)
            .i32(i32_arg("rows", rows)?)
            .opt(count, stream)
            .opt(map, stream)
            .ptr_mut(&mut partials.value, stream)
            .ptr_mut(&mut partials.index, stream)
            .ptr_mut(&mut partials.sum, stream);
        let config = LaunchConfig {
            grid_dim: (FOLD_BLOCKS as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `draft_markov_fold_raw(const void*, const float*, const
        // bf16*, const bf16*, const float*, int, const int*, const int*, float*, int*,
        // float*)`. With `count`, the caller keeps *count at most `rows`, which the buffers
        // hold; the kernel writes FOLD_BLOCKS partials.
        unsafe { args.launch_pdl(stream, &self.markov_fold_raw, config) }?;
        Ok(())
    }

    /// Norm, RoPE and cache stores of a DFlash draft round's context keys and values (see
    /// `draft_dflash_commit_rope_store`): `ctx_k` and `ctx_v` are BF16 `[layers, 8, 1024]`,
    /// the projections of the round's context rows by each layer's `k_proj` and `v_proj`.
    /// With `ring` the caches are rings; otherwise positions past a cache are skipped.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn dflash_commit_rope_store(
        &self,
        stream: &CudaStream,
        ctx_k: &CudaSlice<bf16>,
        ctx_v: &CudaSlice<bf16>,
        control: &CudaSlice<i32>,
        inv_freq: &CudaSlice<f32>,
        eps: f32,
        k_norms: &[&CudaSlice<bf16>],
        caches: &mut [KvCache],
        ring: bool,
        rope: Rope,
    ) -> anyhow::Result<()> {
        let layers = caches.len();
        ensure!(
            (1..=DFLASH_MAX_LAYERS).contains(&layers) && k_norms.len() == layers,
            "{layers} caches and {} norms, expected 1 to {DFLASH_MAX_LAYERS} of each",
            k_norms.len()
        );
        check_len("ctx_k", ctx_k.len(), layers * BLOCK * KV_WIDTH)?;
        check_len("ctx_v", ctx_v.len(), layers * BLOCK * KV_WIDTH)?;
        check_len("control", control.len(), CONTROL)?;
        check_len("inv_freq", inv_freq.len(), HEAD_DIM / 2)?;
        let mut c = DflashCommit {
            ctx_k: ctx_k.device_ptr(stream).0,
            ctx_v: ctx_v.device_ptr(stream).0,
            control: control.device_ptr(stream).0,
            inv_freq: inv_freq.device_ptr(stream).0,
            eps,
            ring: i32::from(ring),
            rope: rope as i32,
            ..DflashCommit::default()
        };
        for (l, (norm, cache)) in k_norms.iter().zip(caches.iter_mut()).enumerate() {
            check_len("k_norm", norm.len(), HEAD_DIM)?;
            c.k_norm[l] = norm.device_ptr(stream).0;
            c.cache_k[l] = cache.k.device_ptr_mut(stream).0;
            c.cache_v[l] = cache.v.device_ptr_mut(stream).0;
            c.slots[l] = i32_arg("slots", cache.slots)?;
            c.fp8_layers |= i32::from(cache.fp8) << l;
        }
        let config = LaunchConfig {
            grid_dim: (BLOCK as u32, KV_HEADS as u32, u32::try_from(layers)?),
            block_dim: ((HEAD_DIM / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `c` has the kernel's `DflashCommit` layout. The buffers have the checked
        // sizes; the kernel writes cache slots below each cache's `slots` (positions mod
        // `slots` with `ring`, else skipping positions past them), skips positions below
        // zero, and reads `ctx_k` and `ctx_v` after `pdl_wait`.
        unsafe { launch_by_value(stream, &self.dflash_commit_rope_store, config, &c, true) }?;
        Ok(())
    }

    /// One step of a DFlash draft's vanilla Markov head (see `draft_dflash_step`), or with
    /// `step == steps` only the pick of the last step's id. Reads the partials of
    /// [`Self::markov_fold_raw`]; writes the step's latent (F32 `[256]`) and the
    /// confidence of the draft it picks.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn dflash_step(
        &self,
        stream: &CudaStream,
        step: usize,
        steps: usize,
        partials: &ArgmaxPartials,
        control: &CudaSlice<i32>,
        ids: &mut CudaSlice<i32>,
        w1: &CudaSlice<bf16>,
        y: &CudaSlice<bf16>,
        conf_w: &CudaSlice<bf16>,
        conf_b: f32,
        latent: &mut CudaSlice<f32>,
        confidence: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        ensure!(
            step <= steps && steps < BLOCK && steps <= ids.len() && confidence.len() == ids.len(),
            "step {step} of {steps} with {} ids and {} confidences",
            ids.len(),
            confidence.len()
        );
        check_len("control", control.len(), CONTROL)?;
        let vocab = i32_arg("vocab", rows_of("w1", w1.len(), RANK)?)?;
        check_len("y", y.len(), BLOCK * HIDDEN)?;
        check_len("conf_w", conf_w.len(), HIDDEN + RANK)?;
        check_len("latent", latent.len(), RANK)?;
        let mut args = KernelArgs::default();
        args.i32(i32_arg("step", step)?)
            .i32(i32_arg("steps", steps)?)
            .ptr(&partials.value, stream)
            .ptr(&partials.index, stream)
            .i32(i32_arg("partials", FOLD_BLOCKS)?)
            .ptr(control, stream)
            .ptr_mut(ids, stream)
            .ptr(w1, stream)
            .i32(vocab)
            .ptr(y, stream)
            .ptr(conf_w, stream)
            .f32(conf_b)
            .ptr_mut(latent, stream)
            .ptr_mut(confidence, stream);
        // SAFETY: the arguments match `draft_dflash_step(int, int, const float*, const int*,
        // int, const int*, int*, const bf16*, int, const bf16*, const bf16*, float, float*,
        // float*)`. The kernel reads everything after `pdl_wait`, bounds-checks the token
        // against `vocab`, reads row step + 1 of the 8 rows of `y`, and writes ids and
        // confidences below `steps`.
        unsafe { args.launch_pdl(stream, &self.dflash_step, one_block(256)) }?;
        Ok(())
    }

    /// DFlash 2's convolution of the 8 block rows of `x` (BF16 `[8, 2304]`) with side
    /// `side` (0 or 1) of the coefficients, into `out`: tap `j` of channel `c` in row `r`
    /// weighs `base[side, j, c] + coef[r, side, j, c / 16]` and reads row `r - j`, zero
    /// before the block. `coef` is BF16 `[8, 576]`, `base` BF16 `[2, 2, 2304]`. Sums in
    /// F32 and rounds once.
    pub fn conv(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        coef: &CudaSlice<bf16>,
        side: usize,
        base: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        ensure!(side < 2, "side {side}, expected 0 or 1");
        check_len("x", x.len(), BLOCK * HIDDEN)?;
        check_len("coef", coef.len(), BLOCK * CONV_COEFFICIENTS)?;
        check_len("base", base.len(), 4 * HIDDEN)?;
        check_len("out", out.len(), BLOCK * HIDDEN)?;
        let mut args = KernelArgs::default();
        args.ptr(x, stream)
            .ptr(coef, stream)
            .i32(i32_arg("side", side)?)
            .ptr(base, stream)
            .ptr_mut(out, stream);
        // SAFETY: the arguments match `draft_conv(const bf16*, const bf16*, int, const
        // bf16*, bf16*)`, with one block per row of the checked buffers. The kernel reads
        // earlier outputs only after `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.conv, row_config(BLOCK)?) }?;
        Ok(())
    }

    /// DFlash 2's output convolution (side 1 of `coef`, see [`Self::conv`]) of the
    /// sublayer output `x`, fused with the residual add and the next norm:
    /// `sum = residual + bf16(conv(x))`, `residual = bf16(sum)`, `out = rms_norm(sum;
    /// weight)`.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn conv_add_rms_norm(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        coef: &CudaSlice<bf16>,
        base: &CudaSlice<bf16>,
        residual: &mut CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        check_len("x", x.len(), BLOCK * HIDDEN)?;
        check_len("coef", coef.len(), BLOCK * CONV_COEFFICIENTS)?;
        check_len("base", base.len(), 4 * HIDDEN)?;
        check_len("residual", residual.len(), BLOCK * HIDDEN)?;
        check_len("weight", weight.len(), HIDDEN)?;
        check_len("out", out.len(), BLOCK * HIDDEN)?;
        let mut args = KernelArgs::default();
        args.ptr(x, stream)
            .ptr(coef, stream)
            .ptr(base, stream)
            .ptr_mut(residual, stream)
            .ptr(weight, stream)
            .ptr_mut(out, stream)
            .f32(eps);
        // SAFETY: the arguments match `draft_conv_add_rms_norm(const bf16*, const bf16*,
        // const bf16*, bf16*, const bf16*, bf16*, float)`, with one block per row of the
        // checked buffers. The kernel reads earlier outputs only after `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.conv_add_rms_norm, row_config(BLOCK)?) }?;
        Ok(())
    }

    /// The 16 largest of each of the `rows` rows of BF16 `logits` (row stride `ld`), best
    /// first and ties to the smaller token, into `values` (F32) and `ids` (I32), each
    /// `[rows, 16]`. A row holds `n` values, or `min(n, *count)` with `count`; value `i`
    /// is token `map[i]` with `map`, else token `i`. Slots past the row's values hold
    /// `-inf` and `i32::MAX`.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn topk(
        &self,
        stream: &CudaStream,
        logits: &CudaView<'_, bf16>,
        rows: usize,
        n: usize,
        ld: usize,
        count: Option<&CudaSlice<i32>>,
        map: Option<&CudaSlice<i32>>,
        scratch: &mut TopkScratch,
        values: &mut CudaSlice<f32>,
        ids: &mut CudaSlice<i32>,
    ) -> anyhow::Result<()> {
        ensure!(
            (1..BLOCK).contains(&rows) && n > 0 && ld >= n,
            "{rows} rows of {n} values {ld} apart"
        );
        ensure!(
            logits.len() >= (rows - 1) * ld + n,
            "logits have {} elements, too few for {rows} rows",
            logits.len()
        );
        if let Some(map) = map {
            ensure!(map.len() >= n, "the map has {} of {n} entries", map.len());
        }
        if let Some(count) = count {
            check_len("count", count.len(), 1)?;
        }
        let blocks = n.div_ceil(TOPK_SPAN);
        let parts = blocks * TOP_K;
        ensure!(
            parts <= TOPK_SPAN && rows * parts <= scratch.values.len(),
            "{n} values exceed the top-k scratch"
        );
        check_len("values", values.len(), (BLOCK - 1) * TOP_K)?;
        check_len("ids", ids.len(), (BLOCK - 1) * TOP_K)?;
        let mut args = KernelArgs::default();
        args.ptr(logits, stream)
            .i32(i32_arg("ld", ld)?)
            .i32(i32_arg("n", n)?)
            .opt(count, stream)
            .opt(map, stream)
            .ptr_mut(&mut scratch.values, stream)
            .ptr_mut(&mut scratch.ids, stream);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(blocks)?, u32::try_from(rows)?, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `draft_topk_partial(const bf16*, int, int, const int*,
        // const int*, float*, int*)`. The kernel reads values below n (and below *count) of
        // the `rows` rows `ld` apart, the map below n, and writes 16 entries per block and
        // row, which the scratch holds. It reads earlier outputs only after `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.topk_partial, config) }?;
        let mut args = KernelArgs::default();
        args.ptr(&scratch.values, stream)
            .ptr(&scratch.ids, stream)
            .i32(i32_arg("parts", parts)?)
            .ptr_mut(values, stream)
            .ptr_mut(ids, stream);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `draft_topk_merge(const float*, const int*, int,
        // float*, int*)`: the first pass wrote `parts` entries per row, at most the 2048 the
        // kernel takes, and the outputs hold 16 per row.
        unsafe { args.launch_pdl(stream, &self.topk_merge, config) }?;
        Ok(())
    }

    /// DFlash 2's greedy walk over the candidates of [`Self::topk`] (see `draft_select`):
    /// `hp` is BF16 `[steps.max(2), 256]`, the hidden projection of block rows 1 to
    /// `steps`; the anchor is `control[2]`. Writes `steps` draft ids and confidences.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn select(
        &self,
        stream: &CudaStream,
        cand: &CudaSlice<i32>,
        unary: &CudaSlice<f32>,
        hp: &CudaSlice<bf16>,
        predecessor: &CudaSlice<bf16>,
        successor: &CudaSlice<bf16>,
        control: &CudaSlice<i32>,
        steps: usize,
        softcap: f32,
        ids: &mut CudaSlice<i32>,
        confidence: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        ensure!(
            (1..BLOCK).contains(&steps) && ids.len() >= steps && confidence.len() >= steps,
            "{steps} steps for {} ids and {} confidences",
            ids.len(),
            confidence.len()
        );
        check_len("cand", cand.len(), (BLOCK - 1) * TOP_K)?;
        check_len("unary", unary.len(), (BLOCK - 1) * TOP_K)?;
        ensure!(
            hp.len() >= steps * RANK,
            "hp has {} elements, too few for {steps} rows",
            hp.len()
        );
        let vocab = rows_of("predecessor", predecessor.len(), RANK)?;
        check_len("successor", successor.len(), vocab * RANK)?;
        check_len("control", control.len(), CONTROL)?;
        let mut args = KernelArgs::default();
        args.ptr(cand, stream)
            .ptr(unary, stream)
            .ptr(hp, stream)
            .ptr(predecessor, stream)
            .ptr(successor, stream)
            .i32(i32_arg("vocab", vocab)?)
            .ptr(control, stream)
            .i32(i32_arg("steps", steps)?)
            .f32(softcap)
            .ptr_mut(ids, stream)
            .ptr_mut(confidence, stream);
        // SAFETY: the arguments match `draft_select(const int*, const float*, const bf16*,
        // const bf16*, const bf16*, int, const int*, int, float, int*, float*)`. The kernel
        // reads rows below `steps` of the checked inputs, bounds-checks every token against
        // `vocab` before reading a codebook row, and writes `steps` ids and confidences. It
        // reads everything after `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.select, one_block(TOP_K * 32)) }?;
        Ok(())
    }

    /// Writes `value` to the one element of `dst`.
    pub fn set_i32(
        &self,
        stream: &CudaStream,
        dst: &mut CudaSlice<i32>,
        value: i32,
    ) -> anyhow::Result<()> {
        check_len("dst", dst.len(), 1)?;
        let mut launch = stream.launch_builder(&self.set_i32);
        launch.arg(dst).arg(&value);
        // SAFETY: the arguments match `draft_set_i32(int*, int)`; one thread writes the
        // element.
        unsafe { launch.launch(one_block(1)) }?;
        Ok(())
    }

    /// `err[c]`, the bound on `|fold_weight[c] · z - fp8[c] · z|` for any latent with
    /// `|z_k| <= 1`, including the F32 rounding of both products (see `draft_fold_error`).
    pub fn fold_error(
        &self,
        stream: &CudaStream,
        fold_weight: &CudaSlice<bf16>,
        fp8: &Fp8Rows,
        err: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        let rows = err.len();
        ensure!(
            fp8.k == RANK && fp8.rows() >= rows && fold_weight.len() >= rows * RANK,
            "the fold copies are too short for {rows} rows"
        );
        let rows_arg = i32_arg("rows", rows)?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows.div_ceil(8))?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.fold_error);
        launch
            .arg(fold_weight)
            .arg(&fp8.values)
            .arg(&fp8.scales)
            .arg(&rows_arg)
            .arg(err);
        // SAFETY: the arguments match `draft_fold_error(const bf16*, const uint8_t*, const
        // float*, int, float*)`; one warp per row below `rows`, which every buffer holds.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// One step's exact Markov argmax, pruned: bounds from the E4M3 fold `fp8` and `err`
    /// (`draft_markov_bound`), then the BF16 product for the rows the bounds leave in the
    /// running (`draft_markov_exact`). Writes the same argmax partials as
    /// [`Self::markov_fold`] without logits. With `reverse`, the bounds read the rows from
    /// the end.
    #[expect(clippy::too_many_arguments, reason = "one argument per kernel input")]
    pub fn markov_pruned(
        &self,
        stream: &CudaStream,
        reverse: bool,
        fold_weight: &CudaSlice<bf16>,
        fp8: &Fp8Rows,
        err: &CudaSlice<f32>,
        fold_bias: &CudaSlice<bf16>,
        base: &CudaView<'_, bf16>,
        latent: &CudaSlice<f32>,
        vocab: usize,
        softcap: f32,
        bounds: &mut LogitBounds,
        partials: &mut ArgmaxPartials,
    ) -> anyhow::Result<()> {
        ensure!(
            fp8.k == RANK
                && fp8.rows() >= vocab
                && err.len() >= vocab
                && fold_weight.len() >= vocab * RANK
                && fold_bias.len() >= vocab
                && base.len() >= vocab
                && bounds.upper.len() >= vocab,
            "a buffer is too short for {vocab} rows"
        );
        check_len("latent", latent.len(), RANK)?;
        let vocab = i32_arg("vocab", vocab)?;
        let config = LaunchConfig {
            grid_dim: (FOLD_BLOCKS as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.i32(i32::from(reverse))
            .ptr(&fp8.values, stream)
            .ptr(&fp8.scales, stream)
            .ptr(err, stream)
            .ptr(fold_bias, stream)
            .ptr(base, stream)
            .ptr(latent, stream)
            .i32(vocab)
            .f32(softcap)
            .ptr_mut(&mut bounds.upper, stream)
            .ptr_mut(&mut bounds.lower, stream)
            .ptr_mut(&mut partials.sum, stream);
        // SAFETY: the arguments match `draft_markov_bound(int, const uint8_t*, const float*,
        // const float*, const bf16*, const bf16*, const float*, int, float, float*, float*,
        // float*)`. Before `pdl_wait` it reads only the E4M3 fold. It reads rows below
        // `vocab`, writes `upper` below `vocab` and one lower bound and one sum per block,
        // FOLD_BLOCKS of each.
        unsafe { args.launch_pdl(stream, &self.markov_bound, config) }?;
        let mut args = KernelArgs::default();
        args.ptr(fold_weight, stream)
            .ptr(fold_bias, stream)
            .ptr(base, stream)
            .ptr(latent, stream)
            .ptr(&bounds.upper, stream)
            .ptr(&bounds.lower, stream)
            .i32(i32_arg("bounds", FOLD_BLOCKS)?)
            .i32(vocab)
            .f32(softcap)
            .ptr_mut(&mut partials.value, stream)
            .ptr_mut(&mut partials.index, stream);
        // SAFETY: the arguments match `draft_markov_exact(const bf16*, const bf16*, const
        // bf16*, const float*, const float*, const float*, int, int, float, float*, int*)`.
        // It reads rows below `vocab` and FOLD_BLOCKS lower bounds, and writes FOLD_BLOCKS
        // partials.
        unsafe { args.launch_pdl(stream, &self.markov_exact, config) }?;
        Ok(())
    }
}
