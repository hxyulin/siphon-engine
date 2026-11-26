//! Chunked prefill: a prompt runs through the model a chunk of rows at a time, every
//! layer visit for the whole chunk before the next visit.
//!
//! Each chunk fills the KV caches and conv states as stepping its tokens one at a time
//! would, so [`Model::step`] continues after it. The small kernels have multi-row forms.
//! Most give the single-row results bit for bit; depth attention, the router logits and
//! the message gates' products sum in their own orders and differ by F32 rounding. The
//! three heavy operations go through [`Ctx::gemm_rows`], [`Ctx::moe_rows`] and
//! [`Ctx::attention_rows`].
//!
//! A chunk's buffers hold `[rows, ...]` per intermediate. The block states of one chunk
//! are `[blocks, rows, 2304]`, so the states in use are one contiguous run.

use std::{fmt, str::FromStr, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _};
use half::bf16;
use siphon::TextConfig;

use super::{
    Blocks, Ctx, MESSAGE_SLOTS, Model, RowProduct, Sampling, Scratch, Upcoming, VisitState, Window,
    batch::BatchBuffers, verify::VerifyBuffers, visit_plan, zeros,
};
use crate::{
    DeviceIn,
    ops::{
        Products, attention,
        conv::VisitConvs,
        experts::{ExpertScratch, QuantizedRows},
        linear_fp8::{Fp8Matrix, Fp8Rows},
        linear_rows,
        misc::{GATE_SLOTS, GateRowsSlot, MessageMix},
        norm::{AttentionPostRows, KvConvWeights},
        routing::{
            DepthRows, FUSED_MAX_CANDIDATES, NormDest, RouterRowsScratch, RowsBatch, RowsPartial,
        },
        sample::SampleBuffers,
        spec::{
            BATCH_ROWS, BATCH_STATE_WORDS, BATCH_VERIFY_MAX_ROWS, BatchRows, RESULT_LEN, TapLayout,
            VERIFY_ROWS, VISIT_CHANNELS,
        },
    },
    weights::{InputNorms, KvConv, LayerWeights, MessageGateWeights},
};

/// Most rows one chunk may hold.
pub const MAX_PREFILL_CHUNK: usize = attention::MAX_PREFILL_ROWS;
const _: () = assert!(MAX_PREFILL_CHUNK == siphon::memory::MAX_PREFILL_CHUNK);
/// The fastest chunk size measured: each expert weight read serves the most rows. Time to
/// first token for 32K tokens was 2.93 s at 512 rows, 2.55 s at 1024 and 2.39 s at 2048.
pub const DEFAULT_PREFILL_CHUNK: usize = 2048;
const _: () = assert!(DEFAULT_PREFILL_CHUNK == siphon::memory::DEFAULT_PREFILL_CHUNK);
/// The attention products [`Model::prefill`] runs over an FP8 cache unless
/// [`Model::set_prefill_attention`] says otherwise.
pub const DEFAULT_PREFILL_ATTENTION: PrefillAttention = PrefillAttention::Auto;
/// Keys past which [`PrefillAttention::Auto`] runs a chunk's global attention on FP8 tensor
/// cores. Rounding the softmax weights to E4M3 moves the output most where few keys share
/// them, while attention's time grows with the square of the length: FP8 saves 2% of time to
/// first token at 8,192 tokens, and keys below this mark are 3% of global attention's work at
/// 92,160 tokens.
pub const FP8_ATTENTION_KEYS: usize = 16_384;

/// Which products [`Model::prefill`]'s global attention runs over an FP8 cache. FP8 passes
/// the quality gate (see `docs/engine.md`) and halves global attention's time, but changes
/// the cache prefill leaves, and with it greedy decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillAttention {
    /// BF16 for chunks whose last row reads up to [`FP8_ATTENTION_KEYS`] keys, FP8 after,
    /// so prompts up to that length prefill exactly as with `Bf16`. Always BF16 without
    /// [`crate::BLOCK_SCALED_MMA`].
    Auto,
    Bf16,
    Fp8,
}

impl PrefillAttention {
    /// The products for a chunk whose last row sits at position `end - 1`.
    fn products(self, end: usize) -> Products {
        match self {
            Self::Auto if end > FP8_ATTENTION_KEYS && crate::BLOCK_SCALED_MMA => Products::Fp8,
            Self::Auto | Self::Bf16 => Products::Bf16,
            Self::Fp8 => Products::Fp8,
        }
    }
}

impl FromStr for PrefillAttention {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "auto" => Ok(Self::Auto),
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            _ => anyhow::bail!("unknown prefill attention {s:?}; expected auto, bf16 or fp8"),
        }
    }
}

impl fmt::Display for PrefillAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        })
    }
}
/// Most rows whose visit ends in one kernel that also shifts every conv state
/// (`Conv::moe_conv_residual_step_rows`); larger chunks shift each state separately.
const STEP_ROWS: usize = VERIFY_ROWS;
/// Rows the LM head runs at once when every row's logits are wanted: 32 rows of 201,024
/// logits take 13 MB in BF16 and 26 MB after the F32 softcap.
const HEAD_ROWS: usize = 32;
const _: () = assert!(HEAD_ROWS == siphon::memory::HEAD_ROWS);
/// Smallest batched/invariant row count that uses native FP8 tensor-core dense GEMMs
/// (`LinearFp8::gemm`) when `prefill_dense == Products::Fp8`.
pub(super) const BATCH_FP8_DENSE_MIN_ROWS: usize = 48;
/// Receives each chunk's feature taps and the position of its first row.
pub(super) type TapsCallback<'a> = &'a mut dyn FnMut(usize, CudaView<'_, bf16>) -> anyhow::Result<()>;

/// The embedding rows of one image in a prompt. Prefill writes them over the rows of the
/// image's placeholder tokens after `embed_norm`, as Transformers' Inkling 2 scatters the
/// vision tower's features into `inputs_embeds`, which hold normed embeddings. Positions
/// stay sequential through the image.
#[derive(Clone, Debug)]
pub struct ImageRows {
    /// The prompt position of the image's first placeholder.
    pub position: usize,
    /// BF16 `[placeholders, hidden]`, written on the model's stream or synchronized with it.
    pub rows: Arc<CudaSlice<bf16>>,
}

impl ImageRows {
    /// Placeholders the image covers.
    #[must_use]
    pub fn len(&self, hidden: usize) -> usize {
        self.rows.len() / hidden
    }
}

/// Gives the images whose placeholders overlap prompt positions `start..end`, a chunk that
/// prefill is about to run. Prefill asks once per chunk, so the caller can encode each image
/// only when its first chunk comes and drop it after its last.
pub type ImageSource<'a> = &'a mut dyn FnMut(usize, usize) -> anyhow::Result<Vec<ImageRows>>;

/// One sequence slice packed into [`Model::prefill_packed`].
pub struct PackedPrefill<'a> {
    /// Target sequence slot (must be distinct across slices in one call).
    pub slot: usize,
    /// Prompt tokens to prefill for this slice (non-empty).
    pub tokens: &'a [u32],
    /// Image rows overlapping this slice's prompt positions.
    pub images: Vec<ImageRows>,
    /// Whether this slice finishes its prompt and needs its first output token.
    pub need_first_token: bool,
    /// Sampling parameters used to draw the first token when `need_first_token` is true.
    pub sampling: Sampling,
}

/// Callback receiving each packed slice's `(slot, start_position, taps)` on the model stream.
pub(crate) type PackedTapsCallback<'a> =
    &'a mut dyn FnMut(usize, usize, CudaView<'_, bf16>) -> anyhow::Result<()>;

/// Host-side metadata for one slice inside [`Ctx::prefill_packed_enqueue`].
struct PackedSliceMeta<'a> {
    slot: usize,
    offset: usize,
    len: usize,
    start: usize,
    images: &'a [ImageRows],
    need_first_token: bool,
    sampling: Sampling,
}

/// Which logits [`Model::prefill`] produces.
pub enum PrefillLogits<'a> {
    /// The last row's only, which [`Model::logits`] then returns.
    Last,
    /// Also every row's F32 logits after the softcap (`[unpadded_vocab]`), passed with the
    /// row's position in order. [`Model::logits`] returns the last row's afterwards.
    All(&'a mut dyn FnMut(usize, &[f32]) -> anyhow::Result<()>),
    /// [`PrefillLogits::All`] for the rows whose position is a multiple of the count only.
    Every(
        usize,
        &'a mut dyn FnMut(usize, &[f32]) -> anyhow::Result<()>,
    ),
}

/// Intermediates of the message gates for a chunk.
struct GateRows {
    /// Whether the products run in a fixed order for any row count (see
    /// [`PrefillScratch::new`]).
    invariant: bool,
    /// Per slot, the normalized state and message side by side, BF16 `[rows, 2 * hidden]`.
    inputs: [CudaSlice<bf16>; MESSAGE_SLOTS],
    /// The down projections, BF16 `[slots, rows, rank]`.
    down: CudaSlice<bf16>,
    /// A gate's output, F32 `[rows, hidden]`, and in invariant chunks one per slot of a launch.
    out: CudaSlice<f32>,
    /// The MXFP8 inputs of `Products::Fp8` dense products: up to three inputs of
    /// `2 * hidden` values per row in prefill, or `max(3 * hidden, q_dims)` per row in
    /// invariant chunks with at least [`BATCH_FP8_DENSE_MIN_ROWS`] rows. The attention
    /// projections and batched LM head borrow it too.
    dense: Fp8Rows,
}

impl GateRows {
    fn new(
        stream: &Arc<CudaStream>,
        rows: usize,
        hidden: usize,
        q_dims: usize,
        rank: usize,
        invariant: bool,
    ) -> anyhow::Result<Self> {
        let dense_len = if !invariant {
            dense_values(rows, hidden)
        } else if rows >= BATCH_FP8_DENSE_MIN_ROWS {
            rows * (3 * hidden).max(q_dims)
        } else {
            0
        };
        Ok(Self {
            invariant,
            inputs: [
                zeros(stream, rows * 2 * hidden)?,
                zeros(stream, rows * 2 * hidden)?,
                zeros(stream, rows * 2 * hidden)?,
                zeros(stream, rows * 2 * hidden)?,
            ],
            down: zeros(stream, MESSAGE_SLOTS * rows * rank)?,
            // Invariant chunks run up to three gates at once, each with its own output.
            out: zeros(
                stream,
                if invariant { GATE_SLOTS } else { 1 } * rows * hidden,
            )?,
            dense: Fp8Rows::new(stream, dense_len)?,
        })
    }
}

/// Intermediates of attention for a chunk, `[rows, ...]` versions of `AttentionTemps`.
struct AttentionRows {
    /// Whether the products and attention run in a fixed order for any row count (see
    /// [`PrefillScratch::new`]).
    invariant: bool,
    /// The partials of the multi-row attention, which invariant chunks run.
    rows: attention::RowsAttentionScratch,
    k_raw: CudaSlice<bf16>,
    v_raw: CudaSlice<bf16>,
    k_dynamic: CudaSlice<bf16>,
    v_dynamic: CudaSlice<bf16>,
    q_raw: CudaSlice<bf16>,
    r_raw: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    sinks: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    r: CudaSlice<bf16>,
    bias: CudaSlice<bf16>,
    partials: attention::PrefillScratch,
    out: CudaSlice<bf16>,
    gated: CudaSlice<bf16>,
    proj: CudaSlice<bf16>,
}

/// Intermediates of the MoE block for a chunk.
struct MoeRows {
    /// Whether the router and experts run the multi-row step kernels, whose rows are the
    /// same bit for bit for any row count (see [`PrefillScratch::new`]).
    invariant: bool,
    /// The counters of the multi-row expert kernel, left zero by each call.
    expert_sync: CudaSlice<u32>,
    logits: CudaSlice<f32>,
    router: RouterRowsScratch,
    indices: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    act: CudaSlice<bf16>,
    down: CudaSlice<bf16>,
    /// The combined expert outputs, the post-MLP conv's input.
    combined: CudaSlice<bf16>,
    /// Grouping of the (row, slot) pairs by expert, shared by w13 and w2.
    experts: ExpertScratch,
    /// The quantized expert inputs of `ExpertActivations::Quantized`, one projection's at a
    /// time: up to `rows * 8` rows of 1152 values.
    quantized: QuantizedRows,
}

/// Every intermediate of a chunk of up to `rows` rows. Names follow [`Scratch`].
pub(super) struct PrefillScratch {
    pub(super) rows: usize,
    /// Whether a chunk records its feature taps into `taps`.
    pub(super) record_taps: bool,
    /// Which feature taps `taps` holds.
    pub(super) tap_layout: TapLayout,
    /// The feature taps of every row, BF16 `[rows, taps, hidden]`: the partial block
    /// entering each visit of the layout, then the final normed hidden state if it asks.
    pub(super) taps: Option<CudaSlice<bf16>>,
    /// The verify pass's own state, in the verify scratch only.
    pub(super) verify: Option<VerifyBuffers>,
    /// A batched step's own state, in the batch scratch only: each row is then the next
    /// token of its own sequence (see `model/batch.rs`).
    pub(super) batch: Option<BatchBuffers>,
    /// With sampling on ([`Model::enable_sampling`]), the settings of each row, by which
    /// the verify and batch heads pick tokens instead of taking the argmax.
    pub(super) sample: Option<SampleBuffers>,
    pub(super) tokens: CudaSlice<i32>,
    /// The position of the chunk's first row.
    pub(super) start: CudaSlice<u32>,
    /// Starting positions of each slice in [`Model::prefill_packed`], `[BATCH_ROWS]`.
    packed_starts: CudaSlice<u32>,
    /// First-token argmax/sample outputs of [`Model::prefill_packed`], `[BATCH_ROWS]`.
    packed_argmax: CudaSlice<i32>,
    embedded: CudaSlice<bf16>,
    hidden: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    prelude_hidden: CudaSlice<bf16>,
    blocks: CudaSlice<f32>,
    bank: CudaSlice<f32>,
    bank_normed: CudaSlice<f32>,
    /// Block states in the bank, the candidates of the messages besides `bank_normed`.
    bank_rows: usize,
    /// The weights of the messages over their candidates, F32 `[rows, 4, bank_rows + 1]`;
    /// the message gates rebuild the messages from them. Invariant chunks hold every looped
    /// layer's, one after another (see `message_x`).
    message_probs: CudaSlice<f32>,
    /// In invariant chunks, every looped layer's normalized messages, BF16
    /// `[4 slots, looped layers, rows, hidden]`: one launch per pass computes them all
    /// instead of one per visit. Other chunks put them into the gate inputs' second halves.
    message_x: Option<CudaSlice<bf16>>,
    /// F32 `[rows, 1 or 3, hidden]`.
    attention_inputs: CudaSlice<f32>,
    gate: GateRows,
    qkv_in: [CudaSlice<bf16>; 3],
    attention: AttentionRows,
    residual: CudaSlice<f32>,
    mlp_input: CudaSlice<f32>,
    mlp_x: CudaSlice<bf16>,
    moe: MoeRows,
    final_hidden: CudaSlice<bf16>,
    head_logits: CudaSlice<bf16>,
    pub(super) head_capped: CudaSlice<f32>,
    /// Host copy of `head_capped`.
    head_host: Vec<f32>,
}

impl PrefillScratch {
    /// Buffers for chunks of up to `rows` rows. With `invariant`, each row's results depend
    /// only on its position, its token and the state before it, bit for bit: not on the
    /// chunk's start, its row count (up to 8 in a verify, 32 in a batched step) or its other
    /// rows. The dense products then run
    /// the multi-row kernels of `ops::linear_rows` (on the FP8 copies when the model has
    /// them), the router and experts those of `Routing::moe_router_step_rows` and
    /// `ops::experts_rows`, and attention those of `Attention::attention_rows`. The verify
    /// pass sets it, so that each of its rows equals a verify of that row alone.
    #[expect(
        clippy::too_many_lines,
        reason = "allocates prefill scratch buffers together"
    )]
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        cfg: &TextConfig,
        max_blocks: usize,
        rows: usize,
        invariant: bool,
    ) -> anyhow::Result<Self> {
        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let q_dims = heads * cfg.head_dim;
        let kv_dims = cfg.num_key_value_heads * cfg.head_dim;
        let dynamic = cfg.num_key_value_heads * cfg.sconv_kernel_size;
        let top_k = cfg.num_experts_per_tok;
        let width = cfg.intermediate_size;
        let looped = cfg.loop_config.core_end - cfg.loop_config.core_start;
        Ok(Self {
            rows,
            record_taps: false,
            tap_layout: TapLayout::stock(),
            taps: None,
            verify: None,
            batch: None,
            sample: None,
            tokens: zeros(stream, rows)?,
            start: zeros(stream, 1)?,
            packed_starts: zeros(stream, BATCH_ROWS)?,
            packed_argmax: zeros(stream, BATCH_ROWS)?,
            embedded: zeros(stream, rows * hidden)?,
            hidden: zeros(stream, rows * hidden)?,
            out: zeros(stream, rows * hidden)?,
            prelude_hidden: zeros(stream, rows * hidden)?,
            blocks: zeros(stream, rows * max_blocks * hidden)?,
            bank: zeros(stream, rows * max_blocks * hidden)?,
            bank_normed: zeros(stream, rows * hidden)?,
            bank_rows: 0,
            message_probs: zeros(
                stream,
                if invariant { looped } else { 1 } * rows * MESSAGE_SLOTS * FUSED_MAX_CANDIDATES,
            )?,
            message_x: invariant
                .then(|| zeros(stream, MESSAGE_SLOTS * looped * rows * hidden))
                .transpose()?,
            attention_inputs: zeros(stream, rows * 3 * hidden)?,
            gate: GateRows::new(
                stream,
                rows,
                hidden,
                q_dims,
                cfg.loop_config.gate_rank,
                invariant,
            )?,
            qkv_in: [
                zeros(stream, rows * hidden)?,
                zeros(stream, rows * hidden)?,
                zeros(stream, rows * hidden)?,
            ],
            attention: AttentionRows {
                invariant,
                rows: attention::RowsAttentionScratch::new(
                    stream,
                    if invariant { rows.min(BATCH_VERIFY_MAX_ROWS) } else { 1 },
                )?,
                k_raw: zeros(stream, rows * kv_dims)?,
                v_raw: zeros(stream, rows * kv_dims)?,
                k_dynamic: zeros(stream, rows * dynamic)?,
                v_dynamic: zeros(stream, rows * dynamic)?,
                q_raw: zeros(stream, rows * q_dims)?,
                r_raw: zeros(stream, rows * heads * cfg.d_rel)?,
                gate: zeros(stream, rows * q_dims)?,
                sinks: zeros(stream, rows * heads)?,
                q: zeros(stream, rows * q_dims)?,
                k: zeros(stream, rows * kv_dims)?,
                v: zeros(stream, rows * kv_dims)?,
                r: zeros(stream, rows * heads * cfg.d_rel)?,
                bias: zeros(stream, rows * heads * cfg.rel_extent)?,
                partials: attention::PrefillScratch::new(stream, rows)?,
                out: zeros(stream, rows * q_dims)?,
                gated: zeros(stream, rows * q_dims)?,
                proj: zeros(stream, rows * hidden)?,
            },
            residual: zeros(stream, rows * hidden)?,
            mlp_input: zeros(stream, rows * hidden)?,
            mlp_x: zeros(stream, rows * hidden)?,
            moe: MoeRows {
                invariant,
                expert_sync: zeros(
                    stream,
                    crate::ops::experts_rows::ExpertsRows::sync_len(rows),
                )?,
                logits: zeros(stream, rows * cfg.n_routed_experts)?,
                router: RouterRowsScratch::new(stream, rows)?,
                indices: zeros(stream, rows * top_k)?,
                weights: zeros(stream, rows * top_k)?,
                act: zeros(stream, rows * top_k * width)?,
                down: zeros(stream, rows * top_k * hidden)?,
                combined: zeros(stream, rows * hidden)?,
                experts: ExpertScratch::new(stream, rows, cfg.n_routed_experts)?,
                quantized: QuantizedRows::new(
                    stream,
                    rows * top_k * width.max(hidden / top_k),
                    rows * top_k,
                )?,
            },
            final_hidden: zeros(stream, rows * hidden)?,
            head_logits: zeros(
                stream,
                if invariant { rows.max(HEAD_ROWS) } else { HEAD_ROWS } * cfg.vocab_size,
            )?,
            head_capped: zeros(
                stream,
                if invariant { rows.max(HEAD_ROWS) } else { HEAD_ROWS } * cfg.unpadded_vocab_size,
            )?,
            head_host: vec![0.0; HEAD_ROWS * cfg.unpadded_vocab_size],
        })
    }
}

/// The first `rows` rows of `width` elements.
/// Values of [`GateRows::dense`] for `rows` rows: three inputs of `2 * hidden`, the most
/// one FP8 dense call takes (three message gates).
pub(crate) fn dense_values(rows: usize, hidden: usize) -> usize {
    rows * 3 * 2 * hidden
}

fn rows<T>(x: &CudaSlice<T>, rows: usize, width: usize) -> CudaView<'_, T> {
    x.slice(..rows * width)
}

fn rows_mut<T>(x: &mut CudaSlice<T>, rows: usize, width: usize) -> CudaViewMut<'_, T> {
    x.slice_mut(..rows * width)
}

/// Column block `offset..offset + width` of `rows` rows that are `stride` apart: a view
/// from the first row's block to the end of the last row's.
fn strided_mut<T>(
    x: &mut CudaSlice<T>,
    rows: usize,
    stride: usize,
    offset: usize,
    width: usize,
) -> CudaViewMut<'_, T> {
    x.slice_mut(offset..offset + (rows - 1) * stride + width)
}

impl Model {
    /// Runs `tokens` from the current position in chunks of up to `chunk` rows and returns
    /// the next token after the last one, greedy or drawn with [`Model::sampling`]. Fills every KV cache and conv state as
    /// stepping the tokens one at a time would, so [`Model::step`] continues after it, and
    /// advances [`Model::position`] by `tokens.len()`. `chunk` is 1 to
    /// [`MAX_PREFILL_CHUNK`]; its buffers are allocated on the first call that needs them.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        chunk: usize,
        logits: PrefillLogits<'_>,
    ) -> anyhow::Result<u32> {
        self.prefill_chunks(tokens, chunk, logits, None, None)
    }

    /// [`Model::prefill`] that also records every row's feature taps for the draft, BF16
    /// `[rows, taps, 2304]` in the order of [`Model::tap_layout`]. After enqueueing each chunk it passes the chunk's taps and the
    /// position of its first row to `taps`. Work that `taps` enqueues on the model's stream
    /// runs before the next chunk overwrites them.
    pub fn prefill_with_taps(
        &mut self,
        tokens: &[u32],
        chunk: usize,
        logits: PrefillLogits<'_>,
        taps: TapsCallback<'_>,
    ) -> anyhow::Result<u32> {
        self.prefill_chunks(tokens, chunk, logits, Some(taps), None)
    }

    /// [`Model::prefill_with_taps`], with taps if `taps` is `Some`, for a prompt with
    /// images: before each chunk it asks `images` for the images that overlap the chunk's
    /// positions, and writes their rows over the chunk's normed embeddings. The tokens at
    /// those positions are the images' placeholders; their own embeddings are discarded.
    pub fn prefill_with_images(
        &mut self,
        tokens: &[u32],
        chunk: usize,
        logits: PrefillLogits<'_>,
        taps: Option<TapsCallback<'_>>,
        images: ImageSource<'_>,
    ) -> anyhow::Result<u32> {
        self.prefill_chunks(tokens, chunk, logits, taps, Some(images))
    }

    /// The normed embedding rows of `tokens`, BF16 `[tokens, hidden]`: the rows prefill
    /// starts from. Image rows replace these, so a test can stand an image in for text.
    pub fn embedding_rows(&self, tokens: &[u32]) -> anyhow::Result<CudaSlice<bf16>> {
        ensure!(!tokens.is_empty(), "no tokens to embed");
        let ctx = &self.ctx;
        let hidden = ctx.hidden();
        let ids = tokens
            .iter()
            .map(|&t| Ok(i32::try_from(t)?))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let ids = ctx.stream.clone_htod(&ids)?;
        let mut embedded = zeros::<bf16>(&ctx.stream, tokens.len() * hidden)?;
        let mut normed = zeros::<bf16>(&ctx.stream, tokens.len() * hidden)?;
        ctx.kernels.elementwise.embedding(
            &ctx.stream,
            &ids,
            ctx.weights.embed.slice(),
            &mut embedded,
        )?;
        ctx.kernels.norm.rms_norm(
            &ctx.stream,
            &embedded,
            &ctx.weights.embed_norm,
            &mut normed,
            ctx.eps(),
        )?;
        Ok(normed)
    }

    /// Allocates the prefill buffers for chunks of `rows` rows now, unless buffers at least
    /// that large exist. Prefill otherwise allocates them on first use, sized by its first
    /// chunk; a server reserves the full chunk at startup so that its memory is known.
    pub fn reserve_prefill(&mut self, rows: usize) -> anyhow::Result<()> {
        ensure!(
            (1..=MAX_PREFILL_CHUNK).contains(&rows),
            "rows is {rows}, expected 1 to {MAX_PREFILL_CHUNK}"
        );
        if self.prefill.as_ref().is_none_or(|p| p.rows < rows) {
            // Free the smaller buffers before allocating the new ones.
            self.prefill = None;
            self.prefill = Some(PrefillScratch::new(
                &self.ctx.stream,
                &self.ctx.config,
                self.max_blocks,
                rows,
                false,
            )?);
        }
        Ok(())
    }

    fn prefill_chunks(
        &mut self,
        tokens: &[u32],
        chunk: usize,
        mut logits: PrefillLogits<'_>,
        mut taps: Option<TapsCallback<'_>>,
        mut images: Option<ImageSource<'_>>,
    ) -> anyhow::Result<u32> {
        ensure!(!tokens.is_empty(), "nothing to prefill");
        ensure!(
            (1..=MAX_PREFILL_CHUNK).contains(&chunk),
            "chunk is {chunk}, expected 1 to {MAX_PREFILL_CHUNK}"
        );
        let last = self.position + tokens.len() - 1;
        ensure!(
            self.visits.iter().all(|v| v.kv.holds(last)),
            "the global KV caches cannot hold position {last}"
        );
        super::map_visits(&mut self.visits, last + 1, &Arc::clone(&self.ctx.stream))?;
        u32::try_from(last).context("position does not fit the kernels")?;
        let chunk_rows = chunk.min(tokens.len());
        self.reserve_prefill(chunk_rows)?;
        let hidden = self.ctx.hidden();
        let p = self
            .prefill
            .as_mut()
            .context("the prefill buffers are missing")?;
        p.record_taps = taps.is_some();
        if p.record_taps && (p.taps.is_none() || p.tap_layout != self.tap_layout) {
            p.taps = None;
            p.tap_layout = self.tap_layout.clone();
            p.taps = Some(zeros(
                &self.ctx.stream,
                p.rows * p.tap_layout.len() * hidden,
            )?);
        }
        let row_len = p.tap_layout.len() * hidden;
        let pieces = tokens.chunks(chunk);
        let count = pieces.len();
        for (i, piece) in pieces.enumerate() {
            let start = self.position;
            let runs = match images.as_mut() {
                Some(images) => images(start, start + piece.len())?,
                None => Vec::new(),
            };
            for run in &runs {
                ensure!(
                    !run.rows.is_empty() && run.rows.len().is_multiple_of(hidden),
                    "image rows hold {} values, not whole rows of {hidden}",
                    run.rows.len()
                );
            }
            self.prefill_chunk(piece, &mut logits, i + 1 == count, &runs)?;
            if let Some(taps) = taps.as_mut() {
                let p = self
                    .prefill
                    .as_ref()
                    .context("the prefill buffers are missing")?;
                let buffer = p.taps.as_ref().context("the taps buffer is missing")?;
                taps(start, rows(buffer, piece.len(), row_len))?;
            }
        }
        self.next_token()
    }

    /// Enqueues one chunk, with the rows of the `images` that overlap it. Only the final
    /// chunk computes the last row's logits.
    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        logits: &mut PrefillLogits<'_>,
        final_chunk: bool,
        images: &[ImageRows],
    ) -> anyhow::Result<()> {
        let m = tokens.len();
        let start = self.position;
        let Self {
            ctx,
            passes,
            visits,
            scratch,
            max_blocks,
            prefill,
            rounded_below,
            ..
        } = self;
        let rounding = ctx.kv_rounding;
        if rounding.recent > 0 {
            // Rows that left the recent window since the last chunk take the 4-bit format.
            let mut end = start.saturating_sub(rounding.recent);
            if rounding.keys_per_channel {
                end -= end % rounding.channel_rows();
            }
            if end > *rounded_below {
                for v in visits.iter_mut().filter(|v| v.round) {
                    ctx.kernels.attention.age_rows(
                        &ctx.stream,
                        &mut v.kv,
                        *rounded_below,
                        end - *rounded_below,
                        rounding,
                    )?;
                }
                *rounded_below = end;
            }
        }
        let p = prefill
            .as_mut()
            .context("the prefill buffers are missing")?;
        let ids = tokens
            .iter()
            .map(|&t| Ok(i32::try_from(t)?))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let first = u32::try_from(start)?;
        ctx.chunk_attention = ctx.prefill_attention.products(start + m);
        ctx.stream
            .memcpy_htod(&ids, &mut rows_mut(&mut p.tokens, m, 1))?;
        ctx.stream.memcpy_htod(&[first], &mut p.start)?;
        ctx.prefill_enqueue(
            p,
            visits,
            *passes,
            *max_blocks,
            scratch,
            m,
            start,
            logits,
            final_chunk,
            images,
        )?;
        self.position += m;
        Ok(())
    }

    /// Prefills multiple independent sequence slices packed into a single pass of up to
    /// `chunk` (`<= MAX_PREFILL_CHUNK`) total rows. All row-independent heavy ops (dense
    /// projections, depth attention, message gates, MoE router + experts, and batched LM
    /// head for finishing slices) run once across all `m = sum(tokens.len())` rows while
    /// attention and short convolutions execute per slice against each slot's own KV cache
    /// and conv states.
    pub fn prefill_packed(
        &mut self,
        slices: &[PackedPrefill<'_>],
        chunk: usize,
    ) -> anyhow::Result<Vec<Option<u32>>> {
        self.prefill_packed_with_taps(slices, chunk, None)
    }

    /// [`Self::prefill_packed`] that also records feature taps for every packed row and
    /// passes each slice's `(slot, start_position, taps)` to `taps` on the model stream.
    #[expect(clippy::too_many_lines, reason = "validates slices, enqueues packed pass, reads tokens")]
    pub(crate) fn prefill_packed_with_taps(
        &mut self,
        slices: &[PackedPrefill<'_>],
        chunk: usize,
        mut taps: Option<PackedTapsCallback<'_>>,
    ) -> anyhow::Result<Vec<Option<u32>>> {
        ensure!(!slices.is_empty(), "nothing to prefill");
        ensure!(
            slices.len() <= BATCH_ROWS,
            "at most {BATCH_ROWS} slices can be packed, got {}",
            slices.len()
        );
        ensure!(
            (1..=MAX_PREFILL_CHUNK).contains(&chunk),
            "chunk is {chunk}, expected 1 to {MAX_PREFILL_CHUNK}"
        );
        ensure!(
            self.ctx.kv_rounding.recent == 0,
            "packed prefill does not support recent-window kv rounding"
        );
        let hidden = self.ctx.hidden();
        let mut total_rows = 0usize;
        let mut meta = Vec::with_capacity(slices.len());
        let mut starts_u32 = Vec::with_capacity(slices.len());
        let mut ids = Vec::new();
        for (i, s) in slices.iter().enumerate() {
            ensure!(!s.tokens.is_empty(), "packed slice {i} is empty");
            ensure!(
                slices[..i].iter().all(|prev| prev.slot != s.slot),
                "slot {} appears twice in packed prefill",
                s.slot
            );
            let start = self.slot_position(s.slot)?;
            let last = start + s.tokens.len() - 1;
            ensure!(
                self.visits.iter().all(|v| v.kv.holds(last)),
                "slot {}: the global KV caches cannot hold position {last}",
                s.slot
            );
            self.map_slot(s.slot, last + 1)?;
            u32::try_from(last).context("position does not fit the kernels")?;
            for run in &s.images {
                ensure!(
                    !run.rows.is_empty() && run.rows.len().is_multiple_of(hidden),
                    "image rows hold {} values, not whole rows of {hidden}",
                    run.rows.len()
                );
            }
            starts_u32.push(u32::try_from(start)?);
            meta.push(PackedSliceMeta {
                slot: s.slot,
                offset: total_rows,
                len: s.tokens.len(),
                start,
                images: &s.images,
                need_first_token: s.need_first_token,
                sampling: s.sampling,
            });
            total_rows += s.tokens.len();
            for &t in s.tokens {
                ids.push(i32::try_from(t)?);
            }
        }
        ensure!(
            total_rows <= chunk,
            "packed prefill has {total_rows} rows, exceeding chunk {chunk}"
        );
        self.reserve_prefill(total_rows)?;
        let p = self
            .prefill
            .as_mut()
            .context("the prefill buffers are missing")?;
        p.record_taps = taps.is_some();
        if p.record_taps && (p.taps.is_none() || p.tap_layout != self.tap_layout) {
            p.taps = None;
            p.tap_layout = self.tap_layout.clone();
            p.taps = Some(zeros(
                &self.ctx.stream,
                p.rows * p.tap_layout.len() * hidden,
            )?);
        }
        let row_len = p.tap_layout.len() * hidden;
        self.ctx
            .stream
            .memcpy_htod(&ids, &mut rows_mut(&mut p.tokens, total_rows, 1))?;
        self.ctx
            .stream
            .memcpy_htod(&starts_u32, &mut p.packed_starts.slice_mut(..slices.len()))?;
        let Self {
            ctx,
            passes,
            visits,
            slots,
            max_blocks,
            prefill,
            sample_one,
            sampler,
            ..
        } = self;
        let p = prefill
            .as_mut()
            .context("the prefill buffers are missing")?;
        let mut by_slice: Vec<Option<&mut [VisitState]>> =
            (0..slices.len()).map(|_| None).collect();
        if let Some(pos) = slices.iter().position(|s| s.slot == 0) {
            by_slice[pos] = Some(visits.as_mut_slice());
        }
        for (idx, slot_state) in slots.iter_mut().enumerate() {
            if let Some(pos) = slices.iter().position(|s| s.slot == idx + 1) {
                by_slice[pos] = Some(slot_state.visits.as_mut_slice());
            }
        }
        let mut slice_visits: Vec<&mut [VisitState]> = by_slice
            .into_iter()
            .map(|v| v.context("missing visit states for packed slot"))
            .collect::<anyhow::Result<_>>()?;
        ctx.prefill_packed_enqueue(
            p,
            &mut slice_visits,
            *passes,
            *max_blocks,
            &meta,
            total_rows,
            sample_one,
            *sampler,
        )?;
        for s in slices {
            if s.slot == 0 {
                self.position += s.tokens.len();
            } else {
                self.slots[s.slot - 1].position += s.tokens.len();
            }
        }
        if let Some(taps) = taps.as_mut() {
            let p = self
                .prefill
                .as_ref()
                .context("the prefill buffers are missing")?;
            let buffer = p.taps.as_ref().context("the taps buffer is missing")?;
            for m_info in &meta {
                let view =
                    buffer.slice(m_info.offset * row_len..(m_info.offset + m_info.len) * row_len);
                taps(m_info.slot, m_info.start, view)?;
            }
        }
        let finishing: Vec<usize> = meta
            .iter()
            .enumerate()
            .filter(|(_, s)| s.need_first_token)
            .map(|(i, _)| i)
            .collect();
        let mut out = vec![None; slices.len()];
        if !finishing.is_empty() {
            let p = self
                .prefill
                .as_ref()
                .context("the prefill buffers are missing")?;
            let raw = self
                .ctx
                .stream
                .clone_dtoh(&p.packed_argmax.slice(..finishing.len()))?;
            for (idx, tok) in finishing.into_iter().zip(raw) {
                out[idx] = Some(u32::try_from(tok).context("argmax returned a negative index")?);
            }
        }
        Ok(out)
    }
}

/// The messages of a chunk, as their candidates and weights: `bank_rows` block states of
/// `[rows, hidden]` and the normalized bank, with the weights `[rows, 4, bank_rows + 1]`.
struct Messages<'a> {
    bank: CudaView<'a, f32>,
    normed: CudaView<'a, f32>,
    probs: CudaView<'a, f32>,
    /// In invariant chunks, the normalized message of each slot, BF16 `[rows, hidden]`;
    /// other chunks hold them in the gate inputs.
    x: Option<[CudaView<'a, bf16>; MESSAGE_SLOTS]>,
}

impl<'a> Messages<'a> {
    /// The messages of looped layer `i` of `looped` for a chunk of `m` rows, from
    /// [`PrefillScratch`]'s `bank`, `bank_normed`, `message_probs` and `message_x`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the scratch fields they come from"
    )]
    fn new(
        bank: &'a CudaSlice<f32>,
        bank_rows: usize,
        normed: &'a CudaSlice<f32>,
        probs: &'a CudaSlice<f32>,
        x: Option<&'a CudaSlice<bf16>>,
        m: usize,
        hidden: usize,
        (i, looped): (usize, usize),
    ) -> Self {
        let mh = m * hidden;
        let set = m * MESSAGE_SLOTS * (bank_rows + 1);
        let first = if x.is_some() { i * set } else { 0 };
        Self {
            bank: bank.slice(..bank_rows * mh),
            normed: rows(normed, m, hidden),
            probs: probs.slice(first..first + set),
            x: x.map(|x| {
                std::array::from_fn(|k| x.slice((k * looped + i) * mh..(k * looped + i + 1) * mh))
            }),
        }
    }
}

/// What [`Ctx::attention_inputs_rows`] reads and writes, as `AttentionInputs` for a chunk.
struct AttentionInputsRows<'a> {
    x: CudaView<'a, bf16>,
    states: CudaView<'a, f32>,
    /// Receives the normalized visit input at a block start.
    normed: Option<CudaViewMut<'a, f32>>,
    inputs: &'a mut CudaSlice<f32>,
    qkv_in: &'a mut [CudaSlice<bf16>; 3],
    gate: &'a mut GateRows,
    messages: Option<(&'a [MessageGateWeights; MESSAGE_SLOTS], Messages<'a>)>,
}

impl Ctx {
    /// One chunk of `m` rows through the model, as `Model::enqueue` for one row. On the
    /// final chunk, the last row's logits and greedy token go to the decode scratch's
    /// `capped` and `next`. Other chunks skip the final norm and the LM head unless
    /// `logits` asks for every row or the chunk records its taps.
    ///
    /// In the verify scratch, the chunk is a verify pass instead: it backs up the conv
    /// states and the ring slots it will overwrite, records each visit's conv inputs, and
    /// ends with every row's logits and greedy token, the acceptance and the rollback (see
    /// `kernels/spec.cu`). The decode scratch is left alone.
    ///
    /// In the batch scratch, the chunk is a batched step instead: row r is the next token of
    /// its own sequence, whose caches, conv states and position the batch's state tables
    /// give, and `visits` only gives each visit's cache sizes and formats. It ends with
    /// every row's logits and greedy token. The decode scratch is left alone.
    ///
    /// The rows of `images` that fall in the chunk replace its normed embeddings. Only a
    /// prompt's prefill has images; verify passes and batched steps pass none.
    #[expect(clippy::too_many_arguments, reason = "the parts of Model a chunk uses")]
    #[expect(
        clippy::too_many_lines,
        reason = "one chunk reads top to bottom like _run_loop"
    )]
    pub(super) fn prefill_enqueue(
        &self,
        p: &mut PrefillScratch,
        visits: &mut [VisitState],
        passes: usize,
        max_blocks: usize,
        s: &mut Scratch,
        m: usize,
        start: usize,
        logits: &mut PrefillLogits<'_>,
        final_chunk: bool,
        images: &[ImageRows],
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        let (core_start, core_end) = (cfg.loop_config.core_start, cfg.loop_config.core_end);
        let hidden = self.hidden();
        let mh = m * hidden;
        // A verify pass with FP8 dense weights prefetches each visit's dense weights into L2
        // beside the main stream, as a decode step does (see `Ctx::visit`). With BF16
        // weights the prefetch made every width slower, paced or not: at 10,674 tokens and
        // 1, 4 and 8 rows, 5.61, 7.68 and 9.89 ms with it against 5.38, 7.20 and 9.29
        // without.
        let prefetch =
            (p.verify.is_some() || p.batch.is_some()) && self.weights.lm_head_fp8.is_some();
        let plan = visit_plan(cfg, passes);
        if prefetch {
            let (layer, messages) = plan.first().context("no visits")?.0;
            self.kernels.linear.prefetch(
                &self.stream,
                &self.visit_prefetch(layer, messages, true),
                self.prefetch_pace(layer),
            )?;
        }
        // The sliding rings as decode prefetches them (see `Ctx::prefetch_rest`), from the
        // first row's position. A batched step's rows read rings of their own slots, which
        // `visits` does not hold, so it prefetches none.
        let position = p.start.device_ptr(&self.stream).0;
        let windows: Vec<Option<Window>> = visits
            .iter()
            .filter(|_| p.batch.is_none())
            .map(|v| {
                v.kv.ring_buffers(&self.stream)
                    .map(|(buffers, slot_bytes)| Window {
                        buffers,
                        slot_bytes,
                        position,
                    })
            })
            .collect();
        let mut plan = plan.iter().enumerate().map(|(v, &(_, upcoming))| {
            prefetch.then_some(Upcoming {
                next_window: windows.get(v + 1).copied().flatten(),
                after_window: windows.get(v + 2).copied().flatten(),
                ..upcoming
            })
        });
        let num_visits = visits.len();
        let mut visits = visits.iter_mut().enumerate();
        let mut next_visit = || {
            let visit = visits.next().context("fewer visit caches than visits")?;
            Ok::<_, anyhow::Error>((visit, plan.next().flatten()))
        };
        if let Some(v) = p.verify.as_mut() {
            self.kernels
                .spec
                .conv_backup(&self.stream, &v.states, &mut v.conv_backup)?;
            self.kernels.spec.ring_backup(
                &self.stream,
                &v.rings,
                v.ring_slots,
                &p.start,
                m,
                &mut v.ring_backup,
                v.ring_row_bytes,
            )?;
        }
        if let Some(b) = p.batch.as_mut()
            && let Some(bv) = b.verify.as_mut()
        {
            let width = bv.width;
            let seqs = m / width;
            self.kernels.spec.conv_backup_batch(
                &self.stream,
                &b.tables,
                num_visits,
                b.max_seqs,
                seqs,
                &mut bv.conv_backup,
            )?;
            self.kernels.spec.ring_backup_batch(
                &self.stream,
                &b.tables,
                &bv.sliding_visits,
                num_visits,
                b.max_seqs,
                seqs,
                bv.ring_slots,
                &b.positions.slice(..seqs),
                width,
                &mut bv.ring_backup,
                bv.ring_row_bytes,
            )?;
        }
        self.kernels.elementwise.embedding(
            &self.stream,
            &rows(&p.tokens, m, 1),
            self.weights.embed.slice(),
            &mut rows_mut(&mut p.embedded, m, hidden),
        )?;
        self.kernels.norm.rms_norm(
            &self.stream,
            &rows(&p.embedded, m, hidden),
            &self.weights.embed_norm,
            &mut rows_mut(&mut p.hidden, m, hidden),
            self.eps(),
        )?;
        for image in images {
            let first = image.position.max(start);
            let end = (image.position + image.len(hidden)).min(start + m);
            if first < end {
                let from = (first - image.position) * hidden;
                let to = (first - start) * hidden;
                let len = (end - first) * hidden;
                self.stream.memcpy_dtod(
                    &image.rows.slice(from..from + len),
                    &mut p.hidden.slice_mut(to..to + len),
                )?;
            }
        }

        let mut blocks = Blocks {
            len: 0,
            max: max_blocks,
        };
        for layer in 0..core_start {
            self.visit_rows(p, next_visit()?, layer, &mut blocks, false, m)?;
        }
        self.stream.memcpy_dtod(
            &rows(&p.hidden, m, hidden),
            &mut rows_mut(&mut p.prelude_hidden, m, hidden),
        )?;
        let prelude_blocks = blocks.len;
        for pass in 0..passes {
            if pass > 0 {
                self.kernels.norm.unweighted_rms_norm_bf16(
                    &self.stream,
                    &rows(&p.hidden, m, hidden),
                    &mut rows_mut(&mut p.bank_normed, m, hidden),
                    self.eps(),
                )?;
                p.bank_rows = blocks.len.saturating_sub(prelude_blocks + 1);
                ensure!(p.bank_rows > 0, "the message bank needs a block state");
                ensure!(
                    p.bank_rows < FUSED_MAX_CANDIDATES,
                    "{} bank states; at most {} fit",
                    p.bank_rows,
                    FUSED_MAX_CANDIDATES - 1
                );
                self.stream.memcpy_dtod(
                    &p.blocks.slice((prelude_blocks + 1) * mh..blocks.len * mh),
                    &mut p.bank.slice_mut(..p.bank_rows * mh),
                )?;
                self.stream.memcpy_dtod(
                    &rows(&p.prelude_hidden, m, hidden),
                    &mut rows_mut(&mut p.hidden, m, hidden),
                )?;
                blocks.len = prelude_blocks;
            }
            if pass > 0 && p.message_x.is_some() {
                // Every looped layer's messages in one launch: they attend over the bank,
                // which stays the same for the whole pass.
                let looped = core_end - core_start;
                let bank_rows = p.bank_rows;
                let x = p.message_x.as_mut().context("message_x")?;
                let (x0, mut rest) = x.split_at_mut(looped * mh);
                let (x1, mut rest) = rest.split_at_mut(looped * mh);
                let (x2, x3) = rest.split_at_mut(looped * mh);
                let dests = [x0, x1, x2, x3].map(|out| Some(NormDest { out, weight: None }));
                let probs_set = m * MESSAGE_SLOTS * (bank_rows + 1);
                self.kernels.routing.depth_attention_rows_batch(
                    &self.stream,
                    DepthRows {
                        blocks: p.bank.slice(..bank_rows * mh),
                        partial: RowsPartial::Normalized(rows(&p.bank_normed, m, hidden)),
                        normed_partial: None,
                        queries: &s.message_queries,
                        out: None,
                        probs: Some(p.message_probs.slice_mut(..looped * probs_set)),
                        dests,
                        dst_stride: hidden,
                        rows: m,
                    },
                    RowsBatch {
                        count: looped,
                        probs_stride: probs_set,
                        dst_stride: mh,
                    },
                    self.eps(),
                )?;
            }
            for (i, layer) in (core_start..core_end).enumerate() {
                if pass > 0 && p.message_x.is_none() {
                    // The messages' weights, and their normalized halves of the gate inputs.
                    let bank_rows = p.bank_rows;
                    let dests = p.gate.inputs.each_mut().map(|x| {
                        Some(NormDest {
                            out: strided_mut(x, m, 2 * hidden, hidden, hidden),
                            weight: None,
                        })
                    });
                    self.kernels.routing.depth_attention_rows(
                        &self.stream,
                        DepthRows {
                            blocks: p.bank.slice(..bank_rows * mh),
                            partial: RowsPartial::Normalized(rows(&p.bank_normed, m, hidden)),
                            normed_partial: None,
                            queries: &self.weights.message_queries[i],
                            out: None,
                            probs: Some(rows_mut(
                                &mut p.message_probs,
                                m,
                                MESSAGE_SLOTS * (bank_rows + 1),
                            )),
                            dests,
                            dst_stride: 2 * hidden,
                            rows: m,
                        },
                        self.eps(),
                    )?;
                }
                self.visit_rows(p, next_visit()?, layer, &mut blocks, pass > 0, m)?;
            }
        }
        for layer in core_end..cfg.num_hidden_layers {
            self.visit_rows(p, next_visit()?, layer, &mut blocks, false, m)?;
        }
        let verify = p.verify.is_some();
        if prefetch {
            self.kernels.linear.join_prefetch(&self.stream)?;
        }
        let skip_logits = !final_chunk && matches!(logits, PrefillLogits::Last);
        if skip_logits && !verify && !p.record_taps {
            return Ok(());
        }

        self.kernels.routing.depth_attention_rows(
            &self.stream,
            DepthRows {
                blocks: blocks.rows(&p.blocks, mh)?,
                partial: RowsPartial::Bf16(rows(&p.hidden, m, hidden)),
                normed_partial: None,
                queries: &self.weights.final_query,
                out: None,
                probs: None,
                dests: [
                    Some(NormDest {
                        out: rows_mut(&mut p.final_hidden, m, hidden),
                        weight: Some(&self.weights.norm),
                    }),
                    None,
                    None,
                    None,
                ],
                dst_stride: hidden,
                rows: m,
            },
            self.eps(),
        )?;
        if p.record_taps && p.tap_layout.final_hidden {
            let count = p.tap_layout.len();
            let taps = p.taps.as_mut().context("the taps buffer is missing")?;
            self.kernels.spec.copy_tap(
                &self.stream,
                &rows(&p.final_hidden, m, hidden),
                &mut rows_mut(taps, m, count * hidden),
                count - 1,
                count,
            )?;
        }
        if verify {
            return self.verify_head(p, m);
        }
        if p.batch.is_some() {
            return self.batch_head(p, m, num_visits);
        }
        if skip_logits {
            return Ok(());
        }

        // The last row's logits and greedy token, in the decode buffers. With FP8 weights
        // the BF16 LM head is in host memory, so the head reads its FP8 copy through the
        // verify pass's products: the token a one-row verify at that position would give.
        let fp8 = self.weights.lm_head_fp8.as_ref();
        self.gemm_rows(
            &mut [RowProduct {
                weight: self.weights.lm_head.slice(),
                fp8,
                x: p.final_hidden.slice((m - 1) * hidden..mh),
                out: s.logits.as_view_mut(),
            }],
            1,
            fp8.is_some(),
            None,
        )?;
        self.kernels.elementwise.logit_softcap(
            &self.stream,
            &s.logits,
            &mut s.capped,
            cfg.final_logit_softcapping,
        )?;
        self.kernels.misc.argmax_f32(
            &self.stream,
            &s.capped,
            &mut s.next,
            cfg.unpadded_vocab_size,
        )?;

        let (every, callback) = match logits {
            PrefillLogits::Last => return Ok(()),
            PrefillLogits::All(callback) => (1, callback),
            PrefillLogits::Every(every, callback) => (*every, callback),
        };
        ensure!(every > 0, "logits every 0 rows");
        let (vocab, padded) = (cfg.unpadded_vocab_size, cfg.vocab_size);
        let step = if fp8.is_some() {
            HEAD_ROWS.min(linear_rows::MAX_ROWS)
        } else {
            HEAD_ROWS
        };
        for first in (0..m).step_by(step) {
            let n = step.min(m - first);
            let wanted = (first..first + n).filter(|&i| (start + i).is_multiple_of(every));
            if wanted.clone().next().is_none() {
                continue;
            }
            self.gemm_rows(
                &mut [RowProduct {
                    weight: self.weights.lm_head.slice(),
                    fp8,
                    x: p.final_hidden.slice(first * hidden..(first + n) * hidden),
                    out: rows_mut(&mut p.head_logits, n, padded),
                }],
                n,
                fp8.is_some(),
                None,
            )?;
            self.kernels.elementwise.logit_softcap(
                &self.stream,
                &rows(&p.head_logits, n, padded),
                &mut rows_mut(&mut p.head_capped, n, vocab),
                cfg.final_logit_softcapping,
            )?;
            if every == 1 {
                let host = &mut p.head_host[..n * vocab];
                self.stream
                    .memcpy_dtoh(&rows(&p.head_capped, n, vocab), host)?;
                for (i, row) in host.chunks_exact(vocab).enumerate() {
                    callback(start + first + i, row)?;
                }
            } else {
                for i in wanted {
                    let row = i - first;
                    let host = &mut p.head_host[..vocab];
                    self.stream
                        .memcpy_dtoh(&p.head_capped.slice(row * vocab..(row + 1) * vocab), host)?;
                    callback(start + i, host)?;
                }
            }
        }
        Ok(())
    }

    /// The end of a verify pass of `m` rows: every row's logits and token, the
    /// acceptance, and the rollback of the conv states and ring slots of rejected rows.
    fn verify_head(&self, p: &mut PrefillScratch, m: usize) -> anyhow::Result<()> {
        let cfg = &self.config;
        let (vocab, padded) = (cfg.unpadded_vocab_size, cfg.vocab_size);
        self.gemm_rows(
            &mut [RowProduct {
                weight: self.weights.lm_head.slice(),
                fp8: self.weights.lm_head_fp8.as_ref(),
                x: rows(&p.final_hidden, m, self.hidden()),
                out: rows_mut(&mut p.head_logits, m, padded),
            }],
            m,
            true,
            None,
        )?;
        self.kernels.elementwise.logit_softcap(
            &self.stream,
            &rows(&p.head_logits, m, padded),
            &mut rows_mut(&mut p.head_capped, m, vocab),
            cfg.final_logit_softcapping,
        )?;
        let v = p
            .verify
            .as_mut()
            .context("the verify buffers are missing")?;
        let spec = &self.kernels.spec;
        match p.sample.as_mut() {
            Some(sample) => self.kernels.sample.run(
                &self.stream,
                &rows(&p.head_capped, m, vocab),
                vocab,
                sample,
                &mut v.argmax.slice_mut(..m),
            )?,
            None => self.kernels.misc.argmax_f32(
                &self.stream,
                &rows(&p.head_capped, m, vocab),
                &mut v.argmax.slice_mut(..m),
                vocab,
            )?,
        }
        spec.accept_greedy(
            &self.stream,
            &rows(&p.tokens, m, 1),
            &v.argmax.slice(..m),
            &v.keep,
            m,
            &mut v.result,
            &mut v.anchor,
        )?;
        spec.conv_rollback(
            &self.stream,
            &v.states,
            &v.conv_backup,
            &v.inputs,
            &v.result,
        )?;
        spec.ring_restore(
            &self.stream,
            &v.rings,
            v.ring_slots,
            &p.start,
            m,
            &v.ring_backup,
            &v.result,
            v.ring_row_bytes,
        )
    }

    /// The end of a batched step or batched verify of `m` rows: every row's logits and
    /// token, plus per-sequence acceptance and rollback for a batched verify.
    fn batch_head(
        &self,
        p: &mut PrefillScratch,
        m: usize,
        num_visits: usize,
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        let (vocab, padded) = (cfg.unpadded_vocab_size, cfg.vocab_size);
        self.gemm_rows(
            &mut [RowProduct {
                weight: self.weights.lm_head.slice(),
                fp8: self.weights.lm_head_fp8.as_ref(),
                x: rows(&p.final_hidden, m, self.hidden()),
                out: rows_mut(&mut p.head_logits, m, padded),
            }],
            m,
            true,
            Some(&mut p.gate.dense),
        )?;
        self.kernels.elementwise.logit_softcap(
            &self.stream,
            &rows(&p.head_logits, m, padded),
            &mut rows_mut(&mut p.head_capped, m, vocab),
            cfg.final_logit_softcapping,
        )?;
        let b = p.batch.as_mut().context("the batch buffers are missing")?;
        match p.sample.as_mut() {
            Some(sample) => self.kernels.sample.run(
                &self.stream,
                &rows(&p.head_capped, m, vocab),
                vocab,
                sample,
                &mut b.argmax.slice_mut(..m),
            )?,
            None => self.kernels.misc.argmax_f32(
                &self.stream,
                &rows(&p.head_capped, m, vocab),
                &mut b.argmax.slice_mut(..m),
                vocab,
            )?,
        }
        if let Some(bv) = b.verify.as_mut() {
            let width = bv.width;
            let seqs = m / width;
            let spec = &self.kernels.spec;
            spec.accept_greedy_batch(
                &self.stream,
                &rows(&p.tokens, m, 1),
                &b.argmax.slice(..m),
                &bv.keep.slice(..seqs),
                seqs,
                width,
                &mut bv.results.slice_mut(..seqs * RESULT_LEN),
                &mut bv.anchors.slice_mut(..seqs),
            )?;
            spec.conv_rollback_batch(
                &self.stream,
                &b.tables,
                num_visits,
                b.max_seqs,
                seqs,
                &bv.conv_backup,
                &bv.inputs,
                &bv.results.slice(..seqs * RESULT_LEN),
            )?;
            spec.ring_restore_batch(
                &self.stream,
                &b.tables,
                &bv.sliding_visits,
                num_visits,
                b.max_seqs,
                seqs,
                bv.ring_slots,
                &b.positions.slice(..seqs),
                width,
                &bv.ring_backup,
                &bv.results.slice(..seqs * RESULT_LEN),
                bv.ring_row_bytes,
            )?;
        }
        Ok(())
    }

    /// One decoder-layer visit for a chunk, as [`Ctx::visit`] for one row, from
    /// `p.hidden` to `p.out`; the two then swap.
    #[expect(
        clippy::too_many_lines,
        reason = "follows DecoderLayer.__call__ op by op"
    )]
    fn visit_rows(
        &self,
        p: &mut PrefillScratch,
        ((index, state), upcoming): ((usize, &mut VisitState), Option<Upcoming>),
        layer: usize,
        blocks: &mut Blocks,
        messages: bool,
        m: usize,
    ) -> anyhow::Result<()> {
        let w = &self.weights.layers[layer];
        let sliding = self.config.is_sliding(layer);
        let hidden = self.hidden();
        let kv_dims = self.config.num_key_value_heads * self.config.head_dim;
        let mh = m * hidden;
        let batch = p.batch.as_ref().map(|b| b.rows(index, m));
        ensure!(
            batch.is_none() || m <= BATCH_VERIFY_MAX_ROWS,
            "a batched step takes at most {BATCH_VERIFY_MAX_ROWS} rows"
        );
        if p.record_taps
            && let Some(tap) = p.tap_layout.visits.iter().position(|&v| v == index)
        {
            let count = p.tap_layout.len();
            let taps = p.taps.as_mut().context("the taps buffer is missing")?;
            self.kernels.spec.copy_tap(
                &self.stream,
                &rows(&p.hidden, m, hidden),
                &mut rows_mut(taps, m, count * hidden),
                tap,
                count,
            )?;
        }
        let block_start = layer.is_multiple_of(self.config.n_layers_per_attn_res_block);
        ensure!(
            !block_start || blocks.len < blocks.max,
            "more block states than planned"
        );
        let gates = if messages {
            w.message_gates.as_ref()
        } else {
            None
        };
        // This layer's index among the looped layers, and their count.
        let loops = self.config.loop_config;
        let looped = (
            layer.saturating_sub(loops.core_start),
            loops.core_end - loops.core_start,
        );
        // The first visit starts prefetching the rest of its own weights; later visits had
        // theirs started by the visit before.
        if let Some(next) = upcoming
            && index == 0
        {
            self.prefetch_rest(layer, gates.is_some(), next.next, true, next.next_window)?;
        }

        // The normalized visit input goes straight into the new block state at a block start.
        // Elsewhere nothing reads it.
        let (states, mut rest) = p.blocks.split_at_mut(blocks.len * mh);
        let normed = block_start.then(|| rest.slice_mut(..mh));
        let one_input = self.attention_inputs_rows(
            w,
            AttentionInputsRows {
                x: rows(&p.hidden, m, hidden),
                states: states.slice(..),
                normed,
                inputs: &mut p.attention_inputs,
                qkv_in: &mut p.qkv_in,
                gate: &mut p.gate,
                messages: gates.map(|g| {
                    (
                        g,
                        Messages::new(
                            &p.bank,
                            p.bank_rows,
                            &p.bank_normed,
                            &p.message_probs,
                            p.message_x.as_ref(),
                            m,
                            hidden,
                            looped,
                        ),
                    )
                }),
            },
            sliding,
            m,
        )?;
        if block_start {
            blocks.len += 1;
        }
        let [q_in, k_in, v_in] = &p.qkv_in;
        let (k_in, v_in) = if one_input {
            (q_in, q_in)
        } else {
            (k_in, v_in)
        };
        self.attention_block_rows(
            state,
            &mut p.attention,
            &mut p.gate.dense,
            &p.start,
            w,
            sliding,
            [q_in, k_in, v_in].map(|x| rows(x, m, hidden)),
            m,
            batch.as_ref(),
        )?;
        // Up to STEP_ROWS rows, and in a batched step, the kernel that ends the visit shifts
        // every conv state.
        let step = m <= STEP_ROWS || batch.is_some();

        // The post-attention conv and residual add run inside the MLP's depth attention.
        // Without message gates, its output goes straight to the router input after the
        // MLP norm; with them, the gate normalizes its output into the router input.
        let (mlp_out, dests) = match gates {
            Some(_) => (
                Some(rows_mut(&mut p.mlp_input, m, hidden)),
                [
                    None,
                    None,
                    None,
                    Some(NormDest {
                        out: strided_mut(&mut p.gate.inputs[3], m, 2 * hidden, 0, hidden),
                        weight: None,
                    }),
                ],
            ),
            None => (
                None,
                [
                    Some(NormDest {
                        out: rows_mut(&mut p.mlp_x, m, hidden),
                        weight: Some(&w.mlp_norm),
                    }),
                    None,
                    None,
                    None,
                ],
            ),
        };
        let partial = if let Some(batch) = batch {
            RowsPartial::ConvResidualBatch {
                x: rows(&p.attention.proj, m, hidden),
                weight: &w.post_attention_conv,
                states: batch.states,
                rows_per_seq: batch.rows_per_seq,
                hidden: (!block_start).then(|| rows(&p.hidden, m, hidden)),
                residual: rows_mut(&mut p.residual, m, hidden),
            }
        } else {
            RowsPartial::ConvResidual {
                x: rows(&p.attention.proj, m, hidden),
                weight: &w.post_attention_conv,
                state: state.conv[2].as_view(),
                hidden: (!block_start).then(|| rows(&p.hidden, m, hidden)),
                residual: rows_mut(&mut p.residual, m, hidden),
            }
        };
        self.kernels.routing.depth_attention_rows(
            &self.stream,
            DepthRows {
                blocks: blocks.rows(&p.blocks, mh)?,
                partial,
                normed_partial: None,
                queries: &w.mlp_query,
                out: mlp_out,
                probs: None,
                dests,
                dst_stride: if gates.is_some() { 2 * hidden } else { hidden },
                rows: m,
            },
            self.eps(),
        )?;
        if !step {
            self.kernels.conv.shift_state(
                &self.stream,
                &rows(&p.attention.proj, m, hidden),
                &mut state.conv[2],
                hidden,
                m,
            )?;
        }
        if let Some(gates) = gates {
            self.message_gate_rows(
                &gates[3..],
                &mut p.gate,
                3,
                (rows(&p.mlp_input, m, hidden), hidden, 0),
                &Messages::new(
                    &p.bank,
                    p.bank_rows,
                    &p.bank_normed,
                    &p.message_probs,
                    p.message_x.as_ref(),
                    m,
                    hidden,
                    looped,
                ),
                vec![(&w.mlp_norm, rows_mut(&mut p.mlp_x, m, hidden))],
                m,
            )?;
        }
        self.moe_block_rows(w, &mut p.moe, &p.mlp_x, m)?;
        // The experts are the visit's last DRAM stream, so the next visits' weights start
        // as soon as they finish.
        if let Some(Upcoming {
            next: Some((layer, messages)),
            after,
            after_window,
            ..
        }) = upcoming
        {
            self.kernels.linear.join_prefetch(&self.stream)?;
            self.prefetch_rest(layer, messages, after, true, after_window)?;
        }

        if let Some(b) = p.batch.as_mut() {
            let max_seqs = b.max_seqs;
            let at = index * max_seqs * BATCH_STATE_WORDS;
            if let Some(bv) = b.verify.as_mut() {
                let width = bv.width;
                let seqs = m / width;
                let visit_len = max_seqs * VERIFY_ROWS * VISIT_CHANNELS;
                self.kernels.conv.moe_conv_residual_batch_verify(
                    &self.stream,
                    &rows(&p.moe.down, m, 8 * hidden),
                    &rows(&p.moe.weights, m, 8),
                    &w.post_mlp_conv,
                    &rows(&p.residual, m, hidden),
                    &mut rows_mut(&mut p.out, m, hidden),
                    [
                        rows(&p.attention.proj, m, hidden),
                        rows(&p.attention.k_raw, m, kv_dims),
                        rows(&p.attention.v_raw, m, kv_dims),
                    ],
                    &b.tables.slice(at..at + seqs * BATCH_STATE_WORDS),
                    Some(bv.inputs.slice_mut(index * visit_len..(index + 1) * visit_len)),
                    seqs,
                    width,
                )?;
            } else {
                self.kernels.conv.moe_conv_residual_batch(
                    &self.stream,
                    &rows(&p.moe.down, m, 8 * hidden),
                    &rows(&p.moe.weights, m, 8),
                    &w.post_mlp_conv,
                    &rows(&p.residual, m, hidden),
                    &mut rows_mut(&mut p.out, m, hidden),
                    [
                        rows(&p.attention.proj, m, hidden),
                        rows(&p.attention.k_raw, m, kv_dims),
                        rows(&p.attention.v_raw, m, kv_dims),
                    ],
                    &b.tables.slice(at..at + m * BATCH_STATE_WORDS),
                    m,
                )?;
            }
            std::mem::swap(&mut p.hidden, &mut p.out);
            return Ok(());
        }
        if step {
            let [k_state, v_state, proj_state, moe_state] = &mut state.conv;
            let len = VERIFY_ROWS * VISIT_CHANNELS;
            self.kernels.conv.moe_conv_residual_step_rows(
                &self.stream,
                &rows(&p.moe.down, m, 8 * hidden),
                &rows(&p.moe.weights, m, 8),
                &w.post_mlp_conv,
                moe_state,
                &rows(&p.residual, m, hidden),
                &mut rows_mut(&mut p.out, m, hidden),
                VisitConvs {
                    proj: rows(&p.attention.proj, m, hidden),
                    proj_state,
                    k_raw: rows(&p.attention.k_raw, m, kv_dims),
                    v_raw: rows(&p.attention.v_raw, m, kv_dims),
                    k_state,
                    v_state,
                    inputs: p
                        .verify
                        .as_mut()
                        .map(|v| v.inputs.slice_mut(index * len..(index + 1) * len)),
                },
                m,
            )?;
            std::mem::swap(&mut p.hidden, &mut p.out);
            return Ok(());
        }
        self.kernels.conv.moe_conv_residual_rows(
            &self.stream,
            &rows(&p.moe.down, m, 8 * hidden),
            &rows(&p.moe.weights, m, 8),
            &w.post_mlp_conv,
            &mut state.conv[3],
            &rows(&p.residual, m, hidden),
            &mut rows_mut(&mut p.moe.combined, m, hidden),
            &mut rows_mut(&mut p.out, m, hidden),
            m,
        )?;
        std::mem::swap(&mut p.hidden, &mut p.out);
        Ok(())
    }

    /// [`Ctx::attention_inputs`] for a chunk of `m` rows.
    #[expect(clippy::too_many_lines, reason = "follows attention_inputs")]
    fn attention_inputs_rows(
        &self,
        w: &LayerWeights,
        a: AttentionInputsRows<'_>,
        sliding: bool,
        m: usize,
    ) -> anyhow::Result<bool> {
        let hidden = self.hidden();
        let mh = m * hidden;
        let norms: [&CudaSlice<bf16>; 3] = match &w.input_norms {
            InputNorms::Shared(norm) => [norm; 3],
            InputNorms::Separate { q, k, v } => [q, k, v],
        };
        let shared = matches!(w.input_norms, InputNorms::Shared(_));
        let Some(query) = &w.attention_query else {
            ensure!(
                a.messages.is_none(),
                "layer without depth attention got messages"
            );
            let mut normed = a
                .normed
                .context("a layer without depth attention must start a block")?;
            self.kernels.norm.unweighted_rms_norm_bf16(
                &self.stream,
                &a.x,
                &mut normed,
                self.eps(),
            )?;
            self.kernels.norm.rms_norm(
                &self.stream,
                &a.x,
                norms[0],
                &mut a.qkv_in[0].slice_mut(..mh),
                self.eps(),
            )?;
            return Ok(true);
        };
        let count = if sliding { 1 } else { 3 };
        ensure!(
            shared == (count == 1),
            "shared input norms must go with one depth-attention query"
        );
        let [g0, g1, g2, _] = &mut a.gate.inputs;
        let [q0, q1, q2] = &mut *a.qkv_in;
        let (dests, dst_stride) = if a.messages.is_some() {
            (
                [g0, g1, g2].map(|g| {
                    Some(NormDest {
                        out: strided_mut(g, m, 2 * hidden, 0, hidden),
                        weight: None,
                    })
                }),
                2 * hidden,
            )
        } else if shared {
            (
                [
                    Some(NormDest {
                        out: q0.slice_mut(..mh),
                        weight: Some(norms[0]),
                    }),
                    None,
                    None,
                ],
                hidden,
            )
        } else {
            (
                [(q0, norms[0]), (q1, norms[1]), (q2, norms[2])].map(|(q, norm)| {
                    Some(NormDest {
                        out: q.slice_mut(..mh),
                        weight: Some(norm),
                    })
                }),
                hidden,
            )
        };
        let [d0, d1, d2] = dests;
        self.kernels.routing.depth_attention_rows(
            &self.stream,
            DepthRows {
                blocks: a.states,
                partial: RowsPartial::Bf16(a.x),
                normed_partial: a.normed,
                queries: query,
                out: a
                    .messages
                    .is_some()
                    .then(|| a.inputs.slice_mut(..m * count * hidden)),
                probs: None,
                dests: [d0, d1, d2, None],
                dst_stride,
                rows: m,
            },
            self.eps(),
        )?;
        let Some((gates, messages)) = a.messages else {
            return Ok(shared);
        };
        // The three gates of the attention inputs, each into its input norm.
        let [q0, q1, q2] = &mut *a.qkv_in;
        self.message_gate_rows(
            &gates[..GATE_SLOTS],
            a.gate,
            0,
            (
                a.inputs.slice(..),
                count * hidden,
                if shared { 0 } else { hidden },
            ),
            &messages,
            [norms[0], norms[1], norms[2]]
                .into_iter()
                .zip([q0, q1, q2])
                .map(|(norm, q)| (norm, q.slice_mut(..mh)))
                .collect(),
            m,
        )?;
        Ok(false)
    }

    /// [`Ctx::attention`] for a chunk of `m` rows. `start` holds the first row's position.
    /// With `batch`, each row is the next token of its own sequence instead, with the
    /// caches, conv states and position `batch` gives; `state` then only gives the cache's
    /// size and format.
    #[expect(
        clippy::too_many_arguments,
        reason = "the attention inputs and outputs"
    )]
    #[expect(clippy::too_many_lines, reason = "follows Ctx::attention op by op")]
    fn attention_block_rows(
        &self,
        state: &mut VisitState,
        t: &mut AttentionRows,
        dense: &mut Fp8Rows,
        start: &CudaSlice<u32>,
        w: &LayerWeights,
        sliding: bool,
        [q_in, k_in, v_in]: [CudaView<'_, bf16>; 3],
        m: usize,
        batch: Option<&BatchRows<'_>>,
    ) -> anyhow::Result<()> {
        fn product<'a>(
            (weight, fp8): (&'a CudaSlice<bf16>, Option<&'a Fp8Matrix>),
            x: &'a CudaView<'_, bf16>,
            out: CudaViewMut<'a, bf16>,
        ) -> RowProduct<'a> {
            RowProduct {
                weight,
                fp8,
                x: x.slice(..),
                out,
            }
        }
        let cfg = &self.config;
        let a = &w.attention;
        let hidden = self.hidden();
        let q_dims = cfg.num_attention_heads * cfg.head_dim;
        let kv_dims = cfg.num_key_value_heads * cfg.head_dim;
        let rel_dims = cfg.num_attention_heads * cfg.d_rel;
        let heads = cfg.num_attention_heads;
        let taps = cfg.num_key_value_heads * cfg.sconv_kernel_size;
        let invariant = t.invariant;
        // Up to STEP_ROWS rows, and in a batched step, `attention_post_rows` also computes
        // the relative bias, and the kernel that ends the visit shifts the conv states.
        let step = m <= STEP_ROWS || batch.is_some();
        // The FP8 copies in `input_projections` order, which the multi-row products read.
        let fp8 = |i: usize| w.fp8.as_ref().map(|f| &f.inputs[i]);

        let mut products = vec![
            product((&a.q, fp8(0)), &q_in, rows_mut(&mut t.q_raw, m, q_dims)),
            product((&a.r, fp8(1)), &q_in, rows_mut(&mut t.r_raw, m, rel_dims)),
            product((&a.gate, fp8(2)), &q_in, rows_mut(&mut t.gate, m, q_dims)),
            product((&a.sink, fp8(3)), &q_in, rows_mut(&mut t.sinks, m, heads)),
            product((&a.k, fp8(4)), &k_in, rows_mut(&mut t.k_raw, m, kv_dims)),
            product((&a.v, fp8(5)), &v_in, rows_mut(&mut t.v_raw, m, kv_dims)),
        ];
        if let KvConv::Dynamic { k_proj, v_proj, .. } = &a.kv_conv {
            products.push(product(
                (k_proj, fp8(6)),
                &k_in,
                rows_mut(&mut t.k_dynamic, m, taps),
            ));
            products.push(product(
                (v_proj, fp8(7)),
                &v_in,
                rows_mut(&mut t.v_dynamic, m, taps),
            ));
        }
        self.gemm_rows(&mut products, m, invariant, Some(&mut *dense))?;
        drop(products);

        let conv = match &a.kv_conv {
            KvConv::Static { k, v } => KvConvWeights::Static { k, v },
            KvConv::Dynamic {
                k_static, v_static, ..
            } => KvConvWeights::Dynamic {
                k_dynamic: rows(&t.k_dynamic, m, taps),
                v_dynamic: rows(&t.v_dynamic, m, taps),
                k_static,
                v_static,
            },
        };
        self.kernels.norm.attention_post_rows(
            &self.stream,
            AttentionPostRows {
                k_raw: rows(&t.k_raw, m, kv_dims),
                v_raw: rows(&t.v_raw, m, kv_dims),
                conv,
                k_state: &state.conv[0],
                v_state: &state.conv[1],
                q_raw: rows(&t.q_raw, m, q_dims),
                q_scale: &a.q_clip_scale,
                r_raw: rows(&t.r_raw, m, rel_dims),
                r_scale: &a.rel_clip_scale,
                k: rows_mut(&mut t.k, m, kv_dims),
                v: rows_mut(&mut t.v, m, kv_dims),
                q: rows_mut(&mut t.q, m, q_dims),
                r: rows_mut(&mut t.r, m, rel_dims),
                log_scale: (!sliding).then_some((
                    start.as_view(),
                    cfg.log_scaling_alpha,
                    cfg.log_scaling_n_floor as f32,
                )),
                qk_threshold: cfg.qk_l2_clip_threshold,
                rel_threshold: cfg.rel_l2_clip_threshold,
                eps: self.eps(),
                bias: step.then(|| {
                    (
                        &a.rel_table,
                        rows_mut(&mut t.bias, m, heads * cfg.rel_extent),
                    )
                }),
                batch,
            },
            m,
        )?;
        if !step {
            let [k_state, v_state, ..] = &mut state.conv;
            self.kernels.conv.shift_state(
                &self.stream,
                &rows(&t.k_raw, m, kv_dims),
                k_state,
                kv_dims,
                m,
            )?;
            self.kernels.conv.shift_state(
                &self.stream,
                &rows(&t.v_raw, m, kv_dims),
                v_state,
                kv_dims,
                m,
            )?;
        }
        if !step {
            self.kernels.attention.relative_bias_rows(
                &self.stream,
                &rows(&t.r, m, rel_dims),
                &a.rel_table,
                &mut rows_mut(&mut t.bias, m, heads * cfg.rel_extent),
            )?;
        }

        if state.round {
            self.kernels.attention.round_rows(
                &self.stream,
                &mut rows_mut(&mut t.k, m, kv_dims),
                &mut rows_mut(&mut t.v, m, kv_dims),
                m,
                self.kv_rounding,
            )?;
        }
        if let Some(batch) = batch {
            ensure!(invariant, "a batched step runs the invariant kernels");
            self.kernels.attention.attention_batch(
                &self.stream,
                &rows(&t.q, m, q_dims),
                &rows(&t.k, m, kv_dims),
                &rows(&t.v, m, kv_dims),
                &state.kv,
                batch,
                m,
                &rows(&t.bias, m, heads * cfg.rel_extent),
                &rows(&t.sinks, m, heads),
                Some(&rows(&t.gate, m, q_dims)),
                &mut t.rows,
                &mut rows_mut(&mut t.gated, m, q_dims),
            )?;
        } else if invariant {
            self.kernels.attention.attention_rows(
                &self.stream,
                &rows(&t.q, m, q_dims),
                &rows(&t.k, m, kv_dims),
                &rows(&t.v, m, kv_dims),
                &mut state.kv,
                start,
                m,
                &rows(&t.bias, m, heads * cfg.rel_extent),
                &rows(&t.sinks, m, heads),
                Some(&rows(&t.gate, m, q_dims)),
                &mut t.rows,
                &mut rows_mut(&mut t.gated, m, q_dims),
            )?;
        } else {
            self.attention_rows(
                &rows(&t.q, m, q_dims),
                &rows(&t.k, m, kv_dims),
                &rows(&t.v, m, kv_dims),
                &mut state.kv,
                start,
                m,
                &rows(&t.bias, m, heads * cfg.rel_extent),
                &rows(&t.sinks, m, heads),
                &mut t.partials,
                &mut rows_mut(&mut t.out, m, q_dims),
            )?;
            self.kernels.elementwise.attention_gate(
                &self.stream,
                &rows(&t.out, m, q_dims),
                &rows(&t.gate, m, q_dims),
                &mut rows_mut(&mut t.gated, m, q_dims),
            )?;
        }
        self.gemm_rows(
            &mut [RowProduct {
                weight: &a.o,
                fp8: w.fp8.as_ref().and_then(|f| f.o.as_ref()),
                x: rows(&t.gated, m, q_dims),
                out: rows_mut(&mut t.proj, m, hidden),
            }],
            m,
            invariant,
            // In prefill (`!invariant`) the output projection stays BF16 under
            // `Products::Fp8` for the prefill quality gate; in batched/invariant passes
            // with `OutputProjection::Fp8` (`w.fp8.o.is_some()`) and `m >= 48`, it uses
            // native FP8 tensor cores when `prefill_dense == Products::Fp8`.
            invariant.then_some(&mut *dense),
        )
    }

    /// [`Ctx::message_gate`] for a chunk of `m` rows and the gates of slots
    /// `first .. first + g.len()`, one, or three at once. `h` holds the gates' states with
    /// its row and gate strides. Each gate normalizes its output with its `norms` weight into
    /// its destination.
    #[expect(clippy::too_many_arguments, reason = "the gates' inputs and outputs")]
    fn message_gate_rows(
        &self,
        weights: &[MessageGateWeights],
        t: &mut GateRows,
        first: usize,
        (state, h_stride, h_slot_stride): (CudaView<'_, f32>, usize, usize),
        messages: &Messages<'_>,
        norms: Vec<(&CudaSlice<bf16>, CudaViewMut<'_, bf16>)>,
        m: usize,
    ) -> anyhow::Result<()> {
        let hidden = self.hidden();
        let rank = self.config.loop_config.gate_rank;
        let mr = m * rank;
        let n = weights.len();
        let invariant = t.invariant;
        ensure!(n == norms.len(), "{n} gates with {} norms", norms.len());
        if invariant {
            // One launch for the gates: down projections, outputs and norms, as below.
            return self.message_gate_launch(
                weights,
                t,
                first,
                (state, h_stride, h_slot_stride),
                messages,
                norms,
                m,
            );
        }
        let mut down = t.down.slice_mut(first * mr..(first + n) * mr);
        match n {
            1 => self.gemm_rows(
                &mut [RowProduct {
                    weight: &weights[0].down,
                    fp8: weights[0].fp8.as_ref().map(|f| &f[0]),
                    x: rows(&t.inputs[first], m, 2 * hidden),
                    out: down.slice_mut(..),
                }],
                m,
                invariant,
                Some(&mut t.dense),
            )?,
            3 => {
                let (d0, mut rest) = down.split_at_mut(mr);
                let (d1, d2) = rest.split_at_mut(mr);
                let [x0, x1, x2] = [0, 1, 2].map(|i| rows(&t.inputs[first + i], m, 2 * hidden));
                self.gemm_rows(
                    &mut [
                        RowProduct {
                            weight: &weights[0].down,
                            fp8: weights[0].fp8.as_ref().map(|f| &f[0]),
                            x: x0,
                            out: d0,
                        },
                        RowProduct {
                            weight: &weights[1].down,
                            fp8: weights[1].fp8.as_ref().map(|f| &f[0]),
                            x: x1,
                            out: d1,
                        },
                        RowProduct {
                            weight: &weights[2].down,
                            fp8: weights[2].fp8.as_ref().map(|f| &f[0]),
                            x: x2,
                            out: d2,
                        },
                    ],
                    m,
                    invariant,
                    Some(&mut t.dense),
                )?;
            }
            _ => anyhow::bail!("{n} gates; one or three run at once"),
        }
        // One gate at a time, so that every gate rewrites the same output rows in L2.
        let candidates = messages.bank.len() / (m * hidden) + 1;
        for (i, (w, (norm, mut normed))) in weights.iter().zip(norms).enumerate() {
            let slot = first + i;
            self.kernels.misc.message_gate_out_rows(
                &self.stream,
                &t.down.slice(slot * mr..(slot + 1) * mr),
                &w.forget,
                &w.gain,
                (&state.slice(i * h_slot_stride..), h_stride),
                (&messages.normed, hidden),
                Some(&MessageMix {
                    blocks: messages.bank.slice(..),
                    probs: messages.probs.slice(slot * candidates..),
                    probs_stride: MESSAGE_SLOTS * candidates,
                }),
                &mut rows_mut(&mut t.out, m, hidden),
                Some((norm, &mut normed)),
                self.eps(),
                m,
            )?;
        }
        Ok(())
    }

    /// [`Ctx::message_gate_rows`] for an invariant chunk: every gate's down projection,
    /// outputs and norm in one launch of [`crate::ops::misc::Misc::message_gate_rows`],
    /// which sums as the separate launches do.
    #[expect(clippy::too_many_arguments, reason = "the gates' inputs and outputs")]
    fn message_gate_launch(
        &self,
        weights: &[MessageGateWeights],
        t: &mut GateRows,
        first: usize,
        (state, h_stride, h_slot_stride): (CudaView<'_, f32>, usize, usize),
        messages: &Messages<'_>,
        norms: Vec<(&CudaSlice<bf16>, CudaViewMut<'_, bf16>)>,
        m: usize,
    ) -> anyhow::Result<()> {
        let hidden = self.hidden();
        let n = weights.len();
        let mh = m * hidden;
        let candidates = messages.bank.len() / mh + 1;
        let fp8 = weights.iter().all(|w| w.fp8.is_some());
        let (o0, mut rest) = t.out.split_at_mut(mh);
        let (o1, mut rest) = rest.split_at_mut(mh);
        let mut outs = vec![o0, o1, rest.slice_mut(..mh)];
        outs.truncate(n);
        let inputs = &t.inputs;
        let x_message_stride = if messages.x.is_some() {
            hidden
        } else {
            2 * hidden
        };
        let mut gates: Vec<GateRowsSlot<'_>> = weights
            .iter()
            .zip(norms)
            .zip(outs)
            .enumerate()
            .map(|(i, ((w, norm), out))| {
                let slot = first + i;
                GateRowsSlot {
                    down: &w.down,
                    down_fp8: w.fp8.as_ref().filter(|_| fp8).map(|f| &f[0]),
                    forget: &w.forget,
                    gain: &w.gain,
                    x_state: rows(&inputs[slot], m, 2 * hidden),
                    x_message: match &messages.x {
                        Some(x) => x[slot].slice(..),
                        None => inputs[slot].slice(hidden..m * 2 * hidden),
                    },
                    h: state.slice(i * h_slot_stride..),
                    probs: messages.probs.slice(slot * candidates..),
                    norm,
                    out,
                }
            })
            .collect();
        self.kernels.misc.message_gate_rows(
            &self.stream,
            &mut gates,
            [2 * hidden, x_message_stride],
            h_stride,
            (&messages.normed, hidden),
            &messages.bank,
            MESSAGE_SLOTS * candidates,
            self.eps(),
            m,
        )
    }

    /// [`Ctx::moe`] for a chunk of `m` rows from the MLP input `x` after its norm: the
    /// router, then the expert products and SwiGLU into `t.down`.
    fn moe_block_rows(
        &self,
        w: &LayerWeights,
        t: &mut MoeRows,
        x: &CudaSlice<bf16>,
        m: usize,
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let width = cfg.intermediate_size;
        let top_k = cfg.num_experts_per_tok;
        if t.invariant {
            // Each row as a step routes it, and each distinct expert streams once.
            self.kernels.routing.moe_router_step_rows(
                &self.stream,
                &rows(x, m, hidden),
                &w.router,
                &w.router_bias,
                &mut rows_mut(&mut t.logits, m, cfg.n_routed_experts),
                &mut rows_mut(&mut t.indices, m, top_k),
                &mut rows_mut(&mut t.weights, m, top_k),
                cfg.route_scale,
                m,
            )?;
            let (w13, w2) = self.expert_weights(w);
            return self.kernels.experts_rows.expert_mlp_rows(
                &self.stream,
                &rows(x, m, hidden),
                &rows(&t.indices, m, top_k),
                &w13,
                &w2,
                &mut rows_mut(&mut t.act, m, top_k * width),
                &mut rows_mut(&mut t.down, m, top_k * hidden),
                &mut t.expert_sync,
                cfg.mlp_gate_softcap,
                cfg.mlp_up_softcap,
            );
        }
        self.kernels.routing.moe_router_rows(
            &self.stream,
            &rows(x, m, hidden),
            &w.router,
            &w.router_bias,
            &mut rows_mut(&mut t.logits, m, cfg.n_routed_experts),
            &mut rows_mut(&mut t.indices, m, top_k),
            &mut rows_mut(&mut t.weights, m, top_k),
            &mut t.router,
            cfg.route_scale,
            m,
        )?;
        let (w13, w2) = self.expert_weights(w);
        self.group_rows(&rows(&t.indices, m, top_k), &mut t.experts, m)?;
        self.moe_rows(
            &rows(x, m, hidden),
            &w13,
            &mut rows_mut(&mut t.act, m, top_k * width),
            &t.experts,
            true,
            &mut t.quantized,
        )?;
        self.moe_rows(
            &rows(&t.act, m, top_k * width),
            &w2,
            &mut rows_mut(&mut t.down, m, top_k * hidden),
            &t.experts,
            false,
            &mut t.quantized,
        )
    }

    /// Packed multi-prompt prefill forward pass: executes all row-independent ops once across
    /// all `m` packed rows and per-slice attention and short convolutions against each slot's
    /// own KV cache and conv states.
    #[expect(clippy::too_many_arguments, reason = "packed prefill state and slices")]
    #[expect(clippy::too_many_lines, reason = "mirrors prefill_enqueue top to bottom")]
    fn prefill_packed_enqueue(
        &self,
        p: &mut PrefillScratch,
        slice_visits: &mut [&mut [VisitState]],
        passes: usize,
        max_blocks: usize,
        meta: &[PackedSliceMeta<'_>],
        m: usize,
        sample_one: &mut SampleBuffers,
        sampler: bool,
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        let (core_start, core_end) = (cfg.loop_config.core_start, cfg.loop_config.core_end);
        let hidden = self.hidden();
        let mh = m * hidden;
        self.kernels.elementwise.embedding(
            &self.stream,
            &rows(&p.tokens, m, 1),
            self.weights.embed.slice(),
            &mut rows_mut(&mut p.embedded, m, hidden),
        )?;
        self.kernels.norm.rms_norm(
            &self.stream,
            &rows(&p.embedded, m, hidden),
            &self.weights.embed_norm,
            &mut rows_mut(&mut p.hidden, m, hidden),
            self.eps(),
        )?;
        for s in meta {
            for image in s.images {
                let first = image.position.max(s.start);
                let end = (image.position + image.len(hidden)).min(s.start + s.len);
                if first < end {
                    let from = (first - image.position) * hidden;
                    let to = (s.offset + first - s.start) * hidden;
                    let len = (end - first) * hidden;
                    self.stream.memcpy_dtod(
                        &image.rows.slice(from..from + len),
                        &mut p.hidden.slice_mut(to..to + len),
                    )?;
                }
            }
        }
        let mut blocks = Blocks {
            len: 0,
            max: max_blocks,
        };
        let mut visit_index = 0usize;
        for layer in 0..core_start {
            self.visit_packed_rows(p, slice_visits, visit_index, layer, &mut blocks, false, meta, m)?;
            visit_index += 1;
        }
        self.stream.memcpy_dtod(
            &rows(&p.hidden, m, hidden),
            &mut rows_mut(&mut p.prelude_hidden, m, hidden),
        )?;
        let prelude_blocks = blocks.len;
        for pass in 0..passes {
            if pass > 0 {
                self.kernels.norm.unweighted_rms_norm_bf16(
                    &self.stream,
                    &rows(&p.hidden, m, hidden),
                    &mut rows_mut(&mut p.bank_normed, m, hidden),
                    self.eps(),
                )?;
                p.bank_rows = blocks.len.saturating_sub(prelude_blocks + 1);
                ensure!(p.bank_rows > 0, "the message bank needs a block state");
                ensure!(
                    p.bank_rows < FUSED_MAX_CANDIDATES,
                    "{} bank states; at most {} fit",
                    p.bank_rows,
                    FUSED_MAX_CANDIDATES - 1
                );
                self.stream.memcpy_dtod(
                    &p.blocks.slice((prelude_blocks + 1) * mh..blocks.len * mh),
                    &mut p.bank.slice_mut(..p.bank_rows * mh),
                )?;
                self.stream.memcpy_dtod(
                    &rows(&p.prelude_hidden, m, hidden),
                    &mut rows_mut(&mut p.hidden, m, hidden),
                )?;
                blocks.len = prelude_blocks;
            }
            for (i, layer) in (core_start..core_end).enumerate() {
                if pass > 0 {
                    let bank_rows = p.bank_rows;
                    let dests = p.gate.inputs.each_mut().map(|x| {
                        Some(NormDest {
                            out: strided_mut(x, m, 2 * hidden, hidden, hidden),
                            weight: None,
                        })
                    });
                    self.kernels.routing.depth_attention_rows(
                        &self.stream,
                        DepthRows {
                            blocks: p.bank.slice(..bank_rows * mh),
                            partial: RowsPartial::Normalized(rows(&p.bank_normed, m, hidden)),
                            normed_partial: None,
                            queries: &self.weights.message_queries[i],
                            out: None,
                            probs: Some(rows_mut(
                                &mut p.message_probs,
                                m,
                                MESSAGE_SLOTS * (bank_rows + 1),
                            )),
                            dests,
                            dst_stride: 2 * hidden,
                            rows: m,
                        },
                        self.eps(),
                    )?;
                }
                self.visit_packed_rows(
                    p,
                    slice_visits,
                    visit_index,
                    layer,
                    &mut blocks,
                    pass > 0,
                    meta,
                    m,
                )?;
                visit_index += 1;
            }
        }
        for layer in core_end..cfg.num_hidden_layers {
            self.visit_packed_rows(p, slice_visits, visit_index, layer, &mut blocks, false, meta, m)?;
            visit_index += 1;
        }
        let finishing: Vec<usize> = meta
            .iter()
            .enumerate()
            .filter(|(_, s)| s.need_first_token)
            .map(|(i, _)| i)
            .collect();
        if finishing.is_empty() && !p.record_taps {
            return Ok(());
        }
        self.kernels.routing.depth_attention_rows(
            &self.stream,
            DepthRows {
                blocks: blocks.rows(&p.blocks, mh)?,
                partial: RowsPartial::Bf16(rows(&p.hidden, m, hidden)),
                normed_partial: None,
                queries: &self.weights.final_query,
                out: None,
                probs: None,
                dests: [
                    Some(NormDest {
                        out: rows_mut(&mut p.final_hidden, m, hidden),
                        weight: Some(&self.weights.norm),
                    }),
                    None,
                    None,
                    None,
                ],
                dst_stride: hidden,
                rows: m,
            },
            self.eps(),
        )?;
        if p.record_taps && p.tap_layout.final_hidden {
            let count = p.tap_layout.len();
            let taps = p.taps.as_mut().context("the taps buffer is missing")?;
            self.kernels.spec.copy_tap(
                &self.stream,
                &rows(&p.final_hidden, m, hidden),
                &mut rows_mut(taps, m, count * hidden),
                count - 1,
                count,
            )?;
        }
        if finishing.is_empty() {
            return Ok(());
        }
        // Gather each finishing slice's final hidden row into contiguous `p.out[..finishing.len()]`.
        for (k, &si) in finishing.iter().enumerate() {
            let last_row = meta[si].offset + meta[si].len - 1;
            self.stream.memcpy_dtod(
                &p.final_hidden.slice(last_row * hidden..(last_row + 1) * hidden),
                &mut p.out.slice_mut(k * hidden..(k + 1) * hidden),
            )?;
        }
        let fp8 = self.weights.lm_head_fp8.as_ref();
        let (vocab, padded) = (cfg.unpadded_vocab_size, cfg.vocab_size);
        let step = if fp8.is_some() {
            HEAD_ROWS.min(linear_rows::MAX_ROWS)
        } else {
            HEAD_ROWS
        };
        for first in (0..finishing.len()).step_by(step) {
            let n = step.min(finishing.len() - first);
            self.gemm_rows(
                &mut [RowProduct {
                    weight: self.weights.lm_head.slice(),
                    fp8,
                    x: p.out.slice(first * hidden..(first + n) * hidden),
                    out: rows_mut(&mut p.head_logits, n, padded),
                }],
                n,
                fp8.is_some(),
                None,
            )?;
            self.kernels.elementwise.logit_softcap(
                &self.stream,
                &rows(&p.head_logits, n, padded),
                &mut rows_mut(&mut p.head_capped, n, vocab),
                cfg.final_logit_softcapping,
            )?;
            let all_greedy = !sampler
                || finishing[first..first + n]
                    .iter()
                    .all(|&si| meta[si].sampling.is_greedy());
            if all_greedy {
                self.kernels.misc.argmax_f32(
                    &self.stream,
                    &rows(&p.head_capped, n, vocab),
                    &mut p.packed_argmax.slice_mut(first..first + n),
                    vocab,
                )?;
            } else {
                for (j, &si) in finishing[first..first + n].iter().enumerate() {
                    let s = &meta[si];
                    let row_logits = p.head_capped.slice(j * vocab..(j + 1) * vocab);
                    let mut dst = p.packed_argmax.slice_mut((first + j)..=(first + j));
                    if s.sampling.is_greedy() {
                        self.kernels
                            .misc
                            .argmax_f32(&self.stream, &row_logits, &mut dst, vocab)?;
                    } else {
                        sample_one.upload(&self.stream, &[(s.sampling, s.start + s.len)])?;
                        self.kernels.sample.run(
                            &self.stream,
                            &row_logits,
                            vocab,
                            sample_one,
                            &mut dst,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// One decoder-layer visit for a packed multi-prompt prefill pass.
    #[expect(clippy::too_many_arguments, reason = "packed visit state")]
    #[expect(clippy::too_many_lines, reason = "mirrors visit_rows with per-slice conv/attention")]
    fn visit_packed_rows(
        &self,
        p: &mut PrefillScratch,
        slice_visits: &mut [&mut [VisitState]],
        index: usize,
        layer: usize,
        blocks: &mut Blocks,
        messages: bool,
        meta: &[PackedSliceMeta<'_>],
        m: usize,
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        let layer_w = &self.weights.layers[layer];
        let attn_w = &layer_w.attention;
        let sliding = cfg.is_sliding(layer);
        let hidden = self.hidden();
        let heads = cfg.num_attention_heads;
        let q_dims = heads * cfg.head_dim;
        let kv_dims = cfg.num_key_value_heads * cfg.head_dim;
        let rel_dims = heads * cfg.d_rel;
        let bias_w = heads * cfg.rel_extent;
        let taps_w = cfg.num_key_value_heads * cfg.sconv_kernel_size;
        let mh = m * hidden;

        if p.record_taps
            && let Some(tap) = p.tap_layout.visits.iter().position(|&v| v == index)
        {
            let count = p.tap_layout.len();
            let taps = p.taps.as_mut().context("the taps buffer is missing")?;
            self.kernels.spec.copy_tap(
                &self.stream,
                &rows(&p.hidden, m, hidden),
                &mut rows_mut(taps, m, count * hidden),
                tap,
                count,
            )?;
        }
        let block_start = layer.is_multiple_of(cfg.n_layers_per_attn_res_block);
        ensure!(
            !block_start || blocks.len < blocks.max,
            "more block states than planned"
        );
        let gates = if messages {
            layer_w.message_gates.as_ref()
        } else {
            None
        };
        let loops = cfg.loop_config;
        let looped = (
            layer.saturating_sub(loops.core_start),
            loops.core_end - loops.core_start,
        );

        let (states, mut rest) = p.blocks.split_at_mut(blocks.len * mh);
        let normed = block_start.then(|| rest.slice_mut(..mh));
        let one_input = self.attention_inputs_rows(
            layer_w,
            AttentionInputsRows {
                x: rows(&p.hidden, m, hidden),
                states: states.slice(..),
                normed,
                inputs: &mut p.attention_inputs,
                qkv_in: &mut p.qkv_in,
                gate: &mut p.gate,
                messages: gates.map(|g| {
                    (
                        g,
                        Messages::new(
                            &p.bank,
                            p.bank_rows,
                            &p.bank_normed,
                            &p.message_probs,
                            p.message_x.as_ref(),
                            m,
                            hidden,
                            looped,
                        ),
                    )
                }),
            },
            sliding,
            m,
        )?;
        if block_start {
            blocks.len += 1;
        }
        let [q_in, k_in, v_in] = &p.qkv_in;
        let (k_in, v_in) = if one_input {
            (q_in, q_in)
        } else {
            (k_in, v_in)
        };
        let [q_in, k_in, v_in] = [q_in, k_in, v_in].map(|x| rows(x, m, hidden));

        // 1. Attention input projections once across all `m` packed rows.
        let attn = &mut p.attention;
        let fp8 = |i: usize| layer_w.fp8.as_ref().map(|f| &f.inputs[i]);
        let mut products = vec![
            RowProduct {
                weight: &attn_w.q,
                fp8: fp8(0),
                x: q_in.slice(..),
                out: rows_mut(&mut attn.q_raw, m, q_dims),
            },
            RowProduct {
                weight: &attn_w.r,
                fp8: fp8(1),
                x: q_in.slice(..),
                out: rows_mut(&mut attn.r_raw, m, rel_dims),
            },
            RowProduct {
                weight: &attn_w.gate,
                fp8: fp8(2),
                x: q_in.slice(..),
                out: rows_mut(&mut attn.gate, m, q_dims),
            },
            RowProduct {
                weight: &attn_w.sink,
                fp8: fp8(3),
                x: q_in.slice(..),
                out: rows_mut(&mut attn.sinks, m, heads),
            },
            RowProduct {
                weight: &attn_w.k,
                fp8: fp8(4),
                x: k_in.slice(..),
                out: rows_mut(&mut attn.k_raw, m, kv_dims),
            },
            RowProduct {
                weight: &attn_w.v,
                fp8: fp8(5),
                x: v_in.slice(..),
                out: rows_mut(&mut attn.v_raw, m, kv_dims),
            },
        ];
        if let KvConv::Dynamic { k_proj, v_proj, .. } = &attn_w.kv_conv {
            products.push(RowProduct {
                weight: k_proj,
                fp8: fp8(6),
                x: k_in.slice(..),
                out: rows_mut(&mut attn.k_dynamic, m, taps_w),
            });
            products.push(RowProduct {
                weight: v_proj,
                fp8: fp8(7),
                x: v_in.slice(..),
                out: rows_mut(&mut attn.v_dynamic, m, taps_w),
            });
        }
        self.gemm_rows(&mut products, m, false, Some(&mut p.gate.dense))?;
        drop(products);

        // 2. Per-slice attention_post_rows + K/V conv state shift.
        for (si, slice) in meta.iter().enumerate() {
            let (off, len) = (slice.offset, slice.len);
            let state = &mut slice_visits[si][index];
            let conv = match &attn_w.kv_conv {
                KvConv::Static { k, v } => KvConvWeights::Static { k, v },
                KvConv::Dynamic {
                    k_static, v_static, ..
                } => KvConvWeights::Dynamic {
                    k_dynamic: attn.k_dynamic.slice(off * taps_w..(off + len) * taps_w),
                    v_dynamic: attn.v_dynamic.slice(off * taps_w..(off + len) * taps_w),
                    k_static,
                    v_static,
                },
            };
            self.kernels.norm.attention_post_rows(
                &self.stream,
                AttentionPostRows {
                    k_raw: attn.k_raw.slice(off * kv_dims..(off + len) * kv_dims),
                    v_raw: attn.v_raw.slice(off * kv_dims..(off + len) * kv_dims),
                    conv,
                    k_state: &state.conv[0],
                    v_state: &state.conv[1],
                    q_raw: attn.q_raw.slice(off * q_dims..(off + len) * q_dims),
                    q_scale: &attn_w.q_clip_scale,
                    r_raw: attn.r_raw.slice(off * rel_dims..(off + len) * rel_dims),
                    r_scale: &attn_w.rel_clip_scale,
                    k: attn.k.slice_mut(off * kv_dims..(off + len) * kv_dims),
                    v: attn.v.slice_mut(off * kv_dims..(off + len) * kv_dims),
                    q: attn.q.slice_mut(off * q_dims..(off + len) * q_dims),
                    r: attn.r.slice_mut(off * rel_dims..(off + len) * rel_dims),
                    log_scale: (!sliding).then_some((
                        p.packed_starts.slice(si..=si),
                        cfg.log_scaling_alpha,
                        cfg.log_scaling_n_floor as f32,
                    )),
                    qk_threshold: cfg.qk_l2_clip_threshold,
                    rel_threshold: cfg.rel_l2_clip_threshold,
                    eps: self.eps(),
                    bias: None,
                    batch: None,
                },
                len,
            )?;
            let [k_state, v_state, ..] = &mut state.conv;
            self.kernels.conv.shift_state(
                &self.stream,
                &attn.k_raw.slice(off * kv_dims..(off + len) * kv_dims),
                k_state,
                kv_dims,
                len,
            )?;
            self.kernels.conv.shift_state(
                &self.stream,
                &attn.v_raw.slice(off * kv_dims..(off + len) * kv_dims),
                v_state,
                kv_dims,
                len,
            )?;
        }

        // 3. Relative bias and optional KV rounding once across all `m` rows.
        self.kernels.attention.relative_bias_rows(
            &self.stream,
            &rows(&attn.r, m, rel_dims),
            &attn_w.rel_table,
            &mut rows_mut(&mut attn.bias, m, bias_w),
        )?;
        if slice_visits[0][index].round {
            self.kernels.attention.round_rows(
                &self.stream,
                &mut rows_mut(&mut attn.k, m, kv_dims),
                &mut rows_mut(&mut attn.v, m, kv_dims),
                m,
                self.kv_rounding,
            )?;
        }

        // 4. Per-slice flash prefill attention against each slot's KV cache.
        for (si, slice) in meta.iter().enumerate() {
            let (off, len) = (slice.offset, slice.len);
            let state = &mut slice_visits[si][index];
            self.kernels.attention.prefill_attention(
                &self.stream,
                &attn.q.slice(off * q_dims..(off + len) * q_dims),
                &attn.k.slice(off * kv_dims..(off + len) * kv_dims),
                &attn.v.slice(off * kv_dims..(off + len) * kv_dims),
                &mut state.kv,
                &p.packed_starts.slice(si..=si),
                len,
                &attn.bias.slice(off * bias_w..(off + len) * bias_w),
                &attn.sinks.slice(off * heads..(off + len) * heads),
                &mut attn.partials,
                &mut attn.out.slice_mut(off * q_dims..(off + len) * q_dims),
                self.prefill_attention.products(slice.start + len),
            )?;
        }

        // 5. Attention gate + output projection once across all `m` rows.
        self.kernels.elementwise.attention_gate(
            &self.stream,
            &rows(&attn.out, m, q_dims),
            &rows(&attn.gate, m, q_dims),
            &mut rows_mut(&mut attn.gated, m, q_dims),
        )?;
        self.gemm_rows(
            &mut [RowProduct {
                weight: &attn_w.o,
                fp8: layer_w.fp8.as_ref().and_then(|f| f.o.as_ref()),
                x: rows(&attn.gated, m, q_dims),
                out: rows_mut(&mut attn.proj, m, hidden),
            }],
            m,
            false,
            None,
        )?;

        // 6. Per-slice post-attention conv + residual add (shifts conv[2] per slice),
        //    then one depth_attention_rows over all `m` rows.
        for (si, slice) in meta.iter().enumerate() {
            let (off, len) = (slice.offset, slice.len);
            let state = &mut slice_visits[si][index];
            let hidden_slice =
                (!block_start).then(|| p.hidden.slice(off * hidden..(off + len) * hidden));
            self.kernels.conv.short_conv_residual_rows(
                &self.stream,
                &p.attention.proj.slice(off * hidden..(off + len) * hidden),
                &layer_w.post_attention_conv,
                &mut state.conv[2],
                hidden_slice.as_ref().map(|v| v as &dyn DeviceIn<bf16>),
                &mut p.residual.slice_mut(off * hidden..(off + len) * hidden),
                len,
            )?;
        }
        let (mlp_out, dests) = match gates {
            Some(_) => (
                Some(rows_mut(&mut p.mlp_input, m, hidden)),
                [
                    None,
                    None,
                    None,
                    Some(NormDest {
                        out: strided_mut(&mut p.gate.inputs[3], m, 2 * hidden, 0, hidden),
                        weight: None,
                    }),
                ],
            ),
            None => (
                None,
                [
                    Some(NormDest {
                        out: rows_mut(&mut p.mlp_x, m, hidden),
                        weight: Some(&layer_w.mlp_norm),
                    }),
                    None,
                    None,
                    None,
                ],
            ),
        };
        self.kernels.routing.depth_attention_rows(
            &self.stream,
            DepthRows {
                blocks: blocks.rows(&p.blocks, mh)?,
                partial: RowsPartial::F32(rows(&p.residual, m, hidden)),
                normed_partial: None,
                queries: &layer_w.mlp_query,
                out: mlp_out,
                probs: None,
                dests,
                dst_stride: if gates.is_some() { 2 * hidden } else { hidden },
                rows: m,
            },
            self.eps(),
        )?;
        if let Some(gates) = gates {
            self.message_gate_rows(
                &gates[3..],
                &mut p.gate,
                3,
                (rows(&p.mlp_input, m, hidden), hidden, 0),
                &Messages::new(
                    &p.bank,
                    p.bank_rows,
                    &p.bank_normed,
                    &p.message_probs,
                    p.message_x.as_ref(),
                    m,
                    hidden,
                    looped,
                ),
                vec![(&layer_w.mlp_norm, rows_mut(&mut p.mlp_x, m, hidden))],
                m,
            )?;
        }

        // 7. MoE router + grouping + expert GEMMs once across all `m` rows.
        self.moe_block_rows(layer_w, &mut p.moe, &p.mlp_x, m)?;

        // 8. Per-slice post-MLP conv + residual add (shifts conv[3] per slice).
        for (si, slice) in meta.iter().enumerate() {
            let (off, len) = (slice.offset, slice.len);
            let state = &mut slice_visits[si][index];
            self.kernels.conv.moe_conv_residual_rows(
                &self.stream,
                &p.moe.down.slice(off * 8 * hidden..(off + len) * 8 * hidden),
                &p.moe.weights.slice(off * 8..(off + len) * 8),
                &layer_w.post_mlp_conv,
                &mut state.conv[3],
                &p.residual.slice(off * hidden..(off + len) * hidden),
                &mut p.moe.combined.slice_mut(off * hidden..(off + len) * hidden),
                &mut p.out.slice_mut(off * hidden..(off + len) * hidden),
                len,
            )?;
        }
        std::mem::swap(&mut p.hidden, &mut p.out);
        Ok(())
    }
}
