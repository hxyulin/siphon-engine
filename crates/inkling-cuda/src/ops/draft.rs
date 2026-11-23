//! The DSpark draft's small kernels (`kernels/draft.cu`): the tap mix, norms, RoPE, the
//! draft cache, block attention, the Markov head and the LM-head fold. The products run
//! through [`crate::ops::linear::Linear`]. [`crate::draft::Draft`] puts them together.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig,
    PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, cubin, load_module};
use crate::weights::draft::{HEAD_DIM, HIDDEN, KV_HEADS, RANK, TAPS};

/// Rows of a draft block: the anchor and 7 mask rows.
pub const BLOCK: usize = 8;
/// Width of the keys or values of one position: 8 heads of 128.
pub const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
/// Query heads per KV head.
const GROUP: usize = 3;
/// Most splits of one block attention.
pub const MAX_ATTENTION_SPLITS: usize = 128;
/// Fewest keys one attention split takes, a multiple of the kernel's chunk of 32.
const MIN_KEYS_PER_SPLIT: usize = 128;
/// Blocks of the first argmax pass over the vocabulary.
pub const LOGIT_BLOCKS: usize = 148;
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
    attention_combine: CudaFunction,
    silu_mul: CudaFunction,
    markov_context_input: CudaFunction,
    markov_init: CudaFunction,
    markov_gate: CudaFunction,
    markov_logits: CudaFunction,
    markov_pick: CudaFunction,
    fold_weight: CudaFunction,
    fold_bias: CudaFunction,
}

/// The split outputs of [`DraftKernels::attention`].
pub struct AttentionScratch {
    out: CudaSlice<f32>,
    stats: CudaSlice<f32>,
}

impl AttentionScratch {
    pub fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let rows = MAX_ATTENTION_SPLITS * KV_HEADS * BLOCK * GROUP;
        Ok(Self {
            out: stream.alloc_zeros(rows * HEAD_DIM)?,
            stats: stream.alloc_zeros(rows * 2)?,
        })
    }
}

/// The first-pass results of the Markov argmax.
pub struct ArgmaxPartials {
    value: CudaSlice<f32>,
    index: CudaSlice<i32>,
}

impl ArgmaxPartials {
    pub fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        Ok(Self {
            value: stream.alloc_zeros(LOGIT_BLOCKS)?,
            index: stream.alloc_zeros(LOGIT_BLOCKS)?,
        })
    }
}

/// One layer's draft cache: keys and values, BF16 `[slots, 8, 128]` each. Position `p`
/// sits in slot `p % slots`.
pub struct KvCache {
    pub k: CudaSlice<bf16>,
    pub v: CudaSlice<bf16>,
    pub slots: usize,
}

impl KvCache {
    pub fn new(stream: &Arc<CudaStream>, slots: usize) -> anyhow::Result<Self> {
        ensure!(slots > 0, "a cache needs at least one slot");
        Ok(Self {
            k: stream.alloc_zeros(slots * KV_WIDTH)?,
            v: stream.alloc_zeros(slots * KV_WIDTH)?,
            slots,
        })
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
        Ok(Self {
            mix_rms_norm: f("draft_mix_rms_norm")?,
            add_rms_norm: f("draft_add_rms_norm")?,
            block_input: f("draft_block_input")?,
            head_norm_rope: f("draft_head_norm_rope")?,
            store_kv: f("draft_store_kv")?,
            attention: f("draft_attention")?,
            attention_combine: f("draft_attention_combine")?,
            silu_mul: f("draft_silu_mul")?,
            markov_context_input: f("draft_markov_context_input")?,
            markov_init: f("draft_markov_init")?,
            markov_gate: f("draft_markov_gate")?,
            markov_logits: f("draft_markov_logits")?,
            markov_pick: f("draft_markov_pick")?,
            fold_weight: f("draft_fold_weight")?,
            fold_bias: f("draft_fold_bias")?,
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
        let mut launch = stream.launch_builder(&self.head_norm_rope);
        launch
            .arg(&mut x)
            .arg(&row_stride)
            .arg(weight)
            .arg(inv_freq)
            .arg(&start)
            .arg(&eps);
        // SAFETY: the arguments match `draft_head_norm_rope(bf16*, int, const bf16*, const
        // float*, int, float)`. The check above keeps every head of every row inside `x`, and
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
            .arg(&slots);
        // SAFETY: the arguments match `draft_store_kv(const bf16*, const bf16*, int, bf16*,
        // bf16*, int, int)`. The checks above keep the 16-byte row reads inside `k` and `v`,
        // and every slot index is below `slots`, inside the caches.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Bidirectional attention of the block's queries `q` (BF16 `[8, 24 * 128]`) over the
    /// keys at positions `first..first + keys` of `cache`, into `out` (BF16
    /// `[8, 24 * 128]`). Scale 128^-0.5, F32 softmax.
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
        let width = KV_HEADS * GROUP * HEAD_DIM;
        check_len("q", q.len(), BLOCK * width)?;
        check_len("out", out.len(), BLOCK * width)?;
        ensure!(keys > 0, "attention needs at least one key");
        ensure!(
            keys <= cache.slots,
            "{keys} keys do not fit {} slots",
            cache.slots
        );
        let (splits, per_split) = attention_splits(keys);
        let slots = i32_arg("slots", cache.slots)?;
        let first_arg = start_arg(first, keys)?;
        let keys_arg = i32_arg("keys", keys)?;
        let per_split = i32_arg("keys per split", per_split)?;
        let splits_arg = i32_arg("splits", splits)?;
        let config = LaunchConfig {
            grid_dim: (KV_HEADS as u32, u32::try_from(splits)?, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.attention);
        launch
            .arg(q)
            .arg(&cache.k)
            .arg(&cache.v)
            .arg(&slots)
            .arg(&first_arg)
            .arg(&keys_arg)
            .arg(&per_split)
            .arg(&mut scratch.out)
            .arg(&mut scratch.stats);
        // SAFETY: the arguments match `draft_attention(const bf16*, const bf16*, const bf16*,
        // int, int, int, int, float*, float*)`. Every key position maps to a slot below
        // `slots`, and the scratch holds MAX_ATTENTION_SPLITS splits, at least `splits`.
        unsafe { launch.launch(config) }?;

        let config = LaunchConfig {
            grid_dim: (KV_HEADS as u32, (BLOCK * GROUP) as u32, 1),
            block_dim: (HEAD_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.attention_combine);
        launch
            .arg(&scratch.out)
            .arg(&scratch.stats)
            .arg(&splits_arg)
            .arg(out);
        // SAFETY: the arguments match `draft_attention_combine(const float*, const float*,
        // int, bf16*)`; the first launch wrote all `splits` partials, and `out` holds the
        // 8 rows of 24 heads.
        unsafe { launch.launch(config) }?;
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
}
