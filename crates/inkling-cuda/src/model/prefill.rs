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

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaSlice, CudaStream, CudaView, CudaViewMut};
use half::bf16;
use inkling::TextConfig;

use super::{
    Blocks, Ctx, MESSAGE_SLOTS, Model, RowProduct, Scratch, VisitState, verify::VerifyBuffers,
    zeros,
};
use crate::{
    ops::{
        attention,
        experts::{ExpertScratch, QuantizedRows},
        linear_fp8::Fp8Matrix,
        misc::{GATE_SLOTS, MessageMix},
        norm::{AttentionPostRows, KvConvWeights},
        routing::{DepthRows, FUSED_MAX_CANDIDATES, NormDest, RouterRowsScratch, RowsPartial},
        spec::{TAPS, VERIFY_ROWS, VISIT_CHANNELS},
    },
    weights::{InputNorms, KvConv, LayerWeights, MessageGateWeights},
};

/// Most rows one chunk may hold.
pub const MAX_PREFILL_CHUNK: usize = attention::MAX_PREFILL_ROWS;
/// The fastest chunk size measured: each expert weight read serves the most rows. Time to
/// first token for 32K tokens was 2.93 s at 512 rows, 2.55 s at 1024 and 2.39 s at 2048.
pub const DEFAULT_PREFILL_CHUNK: usize = 2048;
/// Rows the LM head runs at once when every row's logits are wanted: 32 rows of 201,024
/// logits take 13 MB in BF16 and 26 MB after the F32 softcap.
const HEAD_ROWS: usize = 32;
/// The visits whose input partial block is a feature tap of the draft, in tap order:
/// `target_layer_ids` 5, 23 and 41 of the checkpoint's `mtp_config`. The fourth tap is the
/// final normed hidden state.
const TAP_VISITS: [usize; TAPS - 1] = [5, 23, 41];

/// Receives each chunk's feature taps and the position of its first row.
type TapsCallback<'a> = &'a mut dyn FnMut(usize, CudaView<'_, bf16>) -> anyhow::Result<()>;

/// Which logits [`Model::prefill`] produces.
pub enum PrefillLogits<'a> {
    /// The last row's only, which [`Model::logits`] then returns.
    Last,
    /// Also every row's F32 logits after the softcap (`[unpadded_vocab]`), passed with the
    /// row's position in order. [`Model::logits`] returns the last row's afterwards.
    All(&'a mut dyn FnMut(usize, &[f32]) -> anyhow::Result<()>),
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
    /// A gate's output, F32 `[rows, hidden]`.
    out: CudaSlice<f32>,
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
    /// The counters of the multi-row router and expert kernels, left zero by each call.
    router_counter: CudaSlice<u32>,
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
    /// The feature taps of every row, BF16 `[rows, 4, hidden]`: the partial block entering
    /// each visit of [`TAP_VISITS`], then the final normed hidden state.
    pub(super) taps: Option<CudaSlice<bf16>>,
    /// The verify pass's own state, in the verify scratch only.
    pub(super) verify: Option<VerifyBuffers>,
    pub(super) tokens: CudaSlice<i32>,
    /// The position of the chunk's first row.
    pub(super) start: CudaSlice<u32>,
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
    /// the message gates rebuild the messages from them.
    message_probs: CudaSlice<f32>,
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
    /// chunk's start, its row count up to 8 or its other rows. The dense products then run
    /// the multi-row kernels of `ops::linear_rows` (on the FP8 copies when the model has
    /// them), the router and experts those of `Routing::moe_router_step_rows` and
    /// `ops::experts_rows`, and attention those of `Attention::attention_rows`. The verify
    /// pass sets it, so that each of its rows equals a verify of that row alone.
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
        Ok(Self {
            rows,
            record_taps: false,
            taps: None,
            verify: None,
            tokens: zeros(stream, rows)?,
            start: zeros(stream, 1)?,
            embedded: zeros(stream, rows * hidden)?,
            hidden: zeros(stream, rows * hidden)?,
            out: zeros(stream, rows * hidden)?,
            prelude_hidden: zeros(stream, rows * hidden)?,
            blocks: zeros(stream, rows * max_blocks * hidden)?,
            bank: zeros(stream, rows * max_blocks * hidden)?,
            bank_normed: zeros(stream, rows * hidden)?,
            bank_rows: 0,
            message_probs: zeros(stream, rows * MESSAGE_SLOTS * FUSED_MAX_CANDIDATES)?,
            attention_inputs: zeros(stream, rows * 3 * hidden)?,
            gate: GateRows {
                invariant,
                inputs: [
                    zeros(stream, rows * 2 * hidden)?,
                    zeros(stream, rows * 2 * hidden)?,
                    zeros(stream, rows * 2 * hidden)?,
                    zeros(stream, rows * 2 * hidden)?,
                ],
                down: zeros(stream, MESSAGE_SLOTS * rows * cfg.loop_config.gate_rank)?,
                out: zeros(stream, rows * hidden)?,
            },
            qkv_in: [
                zeros(stream, rows * hidden)?,
                zeros(stream, rows * hidden)?,
                zeros(stream, rows * hidden)?,
            ],
            attention: AttentionRows {
                invariant,
                rows: attention::RowsAttentionScratch::new(stream)?,
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
                router_counter: zeros(stream, 1)?,
                expert_sync: zeros(stream, 1 + rows * top_k)?,
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
            head_logits: zeros(stream, HEAD_ROWS * cfg.vocab_size)?,
            head_capped: zeros(stream, HEAD_ROWS * cfg.unpadded_vocab_size)?,
            head_host: vec![0.0; HEAD_ROWS * cfg.unpadded_vocab_size],
        })
    }
}

/// The first `rows` rows of `width` elements.
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
    /// the greedy next token after the last one. Fills every KV cache and conv state as
    /// stepping the tokens one at a time would, so [`Model::step`] continues after it, and
    /// advances [`Model::position`] by `tokens.len()`. `chunk` is 1 to
    /// [`MAX_PREFILL_CHUNK`]; its buffers are allocated on the first call that needs them.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
        chunk: usize,
        logits: PrefillLogits<'_>,
    ) -> anyhow::Result<u32> {
        self.prefill_chunks(tokens, chunk, logits, None)
    }

    /// [`Model::prefill`] that also records every row's feature taps for the draft, BF16
    /// `[rows, 4, 2304]`: the partial block entering visits 5, 23 and 41, and the final
    /// normed hidden state. After enqueueing each chunk it passes the chunk's taps and the
    /// position of its first row to `taps`. Work that `taps` enqueues on the model's stream
    /// runs before the next chunk overwrites them.
    pub fn prefill_with_taps(
        &mut self,
        tokens: &[u32],
        chunk: usize,
        logits: PrefillLogits<'_>,
        taps: TapsCallback<'_>,
    ) -> anyhow::Result<u32> {
        self.prefill_chunks(tokens, chunk, logits, Some(taps))
    }

    fn prefill_chunks(
        &mut self,
        tokens: &[u32],
        chunk: usize,
        mut logits: PrefillLogits<'_>,
        mut taps: Option<TapsCallback<'_>>,
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
        u32::try_from(last).context("position does not fit the kernels")?;
        let chunk_rows = chunk.min(tokens.len());
        if self.prefill.as_ref().is_none_or(|p| p.rows < chunk_rows) {
            // Free the smaller buffers before allocating the new ones.
            self.prefill = None;
            self.prefill = Some(PrefillScratch::new(
                &self.ctx.stream,
                &self.ctx.config,
                self.max_blocks,
                chunk_rows,
                false,
            )?);
        }
        let hidden = self.ctx.hidden();
        let p = self
            .prefill
            .as_mut()
            .context("the prefill buffers are missing")?;
        p.record_taps = taps.is_some();
        if p.record_taps && p.taps.is_none() {
            p.taps = Some(zeros(&self.ctx.stream, p.rows * TAPS * hidden)?);
        }
        let pieces = tokens.chunks(chunk);
        let count = pieces.len();
        for (i, piece) in pieces.enumerate() {
            let start = self.position;
            self.prefill_chunk(piece, &mut logits, i + 1 == count)?;
            if let Some(taps) = taps.as_mut() {
                let p = self
                    .prefill
                    .as_ref()
                    .context("the prefill buffers are missing")?;
                let buffer = p.taps.as_ref().context("the taps buffer is missing")?;
                taps(start, rows(buffer, piece.len(), TAPS * hidden))?;
            }
        }
        let next = self.ctx.stream.clone_dtoh(&self.scratch.next)?[0];
        u32::try_from(next).context("argmax returned a negative index")
    }

    /// Enqueues one chunk. Only the final chunk computes the last row's logits.
    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        logits: &mut PrefillLogits<'_>,
        final_chunk: bool,
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
            ..
        } = self;
        let p = prefill
            .as_mut()
            .context("the prefill buffers are missing")?;
        let ids = tokens
            .iter()
            .map(|&t| Ok(i32::try_from(t)?))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let first = u32::try_from(start)?;
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
        )?;
        self.position += m;
        Ok(())
    }
}

/// The messages of a chunk, as their candidates and weights: `bank_rows` block states of
/// `[rows, hidden]` and the normalized bank, with the weights `[rows, 4, bank_rows + 1]`.
struct Messages<'a> {
    bank: CudaView<'a, f32>,
    normed: CudaView<'a, f32>,
    probs: CudaView<'a, f32>,
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
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        let (core_start, core_end) = (cfg.loop_config.core_start, cfg.loop_config.core_end);
        let hidden = self.hidden();
        let mh = m * hidden;
        let mut visits = visits.iter_mut().enumerate();
        let mut next_visit = || visits.next().context("fewer visit caches than visits");

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
            )?;
        }
        self.kernels.elementwise.embedding(
            &self.stream,
            &rows(&p.tokens, m, 1),
            &self.weights.embed,
            &mut rows_mut(&mut p.embedded, m, hidden),
        )?;
        self.kernels.norm.rms_norm(
            &self.stream,
            &rows(&p.embedded, m, hidden),
            &self.weights.embed_norm,
            &mut rows_mut(&mut p.hidden, m, hidden),
            self.eps(),
        )?;

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
            for (i, layer) in (core_start..core_end).enumerate() {
                if pass > 0 {
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
        if p.record_taps {
            let taps = p.taps.as_mut().context("the taps buffer is missing")?;
            self.kernels.spec.copy_tap(
                &self.stream,
                &rows(&p.final_hidden, m, hidden),
                &mut rows_mut(taps, m, TAPS * hidden),
                TAPS - 1,
            )?;
        }
        if verify {
            return self.verify_head(p, m);
        }
        if skip_logits {
            return Ok(());
        }

        // The last row's logits and greedy token, in the decode buffers.
        self.gemm_rows(
            &mut [RowProduct {
                weight: &self.weights.lm_head,
                fp8: None,
                x: p.final_hidden.slice((m - 1) * hidden..mh),
                out: s.logits.as_view_mut(),
            }],
            1,
            false,
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

        let PrefillLogits::All(callback) = logits else {
            return Ok(());
        };
        let (vocab, padded) = (cfg.unpadded_vocab_size, cfg.vocab_size);
        for first in (0..m).step_by(HEAD_ROWS) {
            let n = HEAD_ROWS.min(m - first);
            self.gemm_rows(
                &mut [RowProduct {
                    weight: &self.weights.lm_head,
                    fp8: None,
                    x: p.final_hidden.slice(first * hidden..(first + n) * hidden),
                    out: rows_mut(&mut p.head_logits, n, padded),
                }],
                n,
                false,
            )?;
            self.kernels.elementwise.logit_softcap(
                &self.stream,
                &rows(&p.head_logits, n, padded),
                &mut rows_mut(&mut p.head_capped, n, vocab),
                cfg.final_logit_softcapping,
            )?;
            let host = &mut p.head_host[..n * vocab];
            self.stream
                .memcpy_dtoh(&rows(&p.head_capped, n, vocab), host)?;
            for (i, row) in host.chunks_exact(vocab).enumerate() {
                callback(start + first + i, row)?;
            }
        }
        Ok(())
    }

    /// The end of a verify pass of `m` rows: every row's logits and greedy token, the
    /// acceptance, and the rollback of the conv states and ring slots of rejected rows.
    fn verify_head(&self, p: &mut PrefillScratch, m: usize) -> anyhow::Result<()> {
        let cfg = &self.config;
        let (vocab, padded) = (cfg.unpadded_vocab_size, cfg.vocab_size);
        self.gemm_rows(
            &mut [RowProduct {
                weight: &self.weights.lm_head,
                fp8: self.weights.lm_head_fp8.as_ref(),
                x: rows(&p.final_hidden, m, self.hidden()),
                out: rows_mut(&mut p.head_logits, m, padded),
            }],
            m,
            true,
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
        self.kernels.misc.argmax_f32(
            &self.stream,
            &rows(&p.head_capped, m, vocab),
            &mut v.argmax.slice_mut(..m),
            vocab,
        )?;
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
        )
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
        (index, state): (usize, &mut VisitState),
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
        if p.record_taps
            && let Some(tap) = TAP_VISITS.iter().position(|&v| v == index)
        {
            let taps = p.taps.as_mut().context("the taps buffer is missing")?;
            self.kernels.spec.copy_tap(
                &self.stream,
                &rows(&p.hidden, m, hidden),
                &mut rows_mut(taps, m, TAPS * hidden),
                tap,
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
                        Messages {
                            bank: p.bank.slice(..p.bank_rows * mh),
                            normed: rows(&p.bank_normed, m, hidden),
                            probs: rows(&p.message_probs, m, MESSAGE_SLOTS * (p.bank_rows + 1)),
                        },
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
            &p.start,
            w,
            sliding,
            [q_in, k_in, v_in].map(|x| rows(x, m, hidden)),
            m,
        )?;

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
        self.kernels.routing.depth_attention_rows(
            &self.stream,
            DepthRows {
                blocks: blocks.rows(&p.blocks, mh)?,
                partial: RowsPartial::ConvResidual {
                    x: rows(&p.attention.proj, m, hidden),
                    weight: &w.post_attention_conv,
                    state: state.conv[2].as_view(),
                    hidden: (!block_start).then(|| rows(&p.hidden, m, hidden)),
                    residual: rows_mut(&mut p.residual, m, hidden),
                },
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
        self.kernels.conv.shift_state(
            &self.stream,
            &rows(&p.attention.proj, m, hidden),
            &mut state.conv[2],
            hidden,
            m,
        )?;
        if let Some(gates) = gates {
            self.message_gate_rows(
                &gates[3..],
                &mut p.gate,
                3,
                (rows(&p.mlp_input, m, hidden), hidden, 0),
                &Messages {
                    bank: p.bank.slice(..p.bank_rows * mh),
                    normed: rows(&p.bank_normed, m, hidden),
                    probs: rows(&p.message_probs, m, MESSAGE_SLOTS * (p.bank_rows + 1)),
                },
                vec![(&w.mlp_norm, rows_mut(&mut p.mlp_x, m, hidden))],
                m,
            )?;
        }
        self.moe_block_rows(w, &mut p.moe, &p.mlp_x, m)?;

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
        if let Some(v) = p.verify.as_mut() {
            let len = VERIFY_ROWS * VISIT_CHANNELS;
            self.kernels.spec.conv_inputs_rows(
                &self.stream,
                &rows(&p.attention.k_raw, m, kv_dims),
                &rows(&p.attention.v_raw, m, kv_dims),
                &rows(&p.attention.proj, m, hidden),
                &rows(&p.moe.down, m, 8 * hidden),
                &rows(&p.moe.weights, m, 8),
                &mut v.inputs.slice_mut(index * len..(index + 1) * len),
                m,
            )?;
        }
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
    #[expect(
        clippy::too_many_arguments,
        reason = "the attention inputs and outputs"
    )]
    #[expect(clippy::too_many_lines, reason = "follows Ctx::attention op by op")]
    fn attention_block_rows(
        &self,
        state: &mut VisitState,
        t: &mut AttentionRows,
        start: &CudaSlice<u32>,
        w: &LayerWeights,
        sliding: bool,
        [q_in, k_in, v_in]: [CudaView<'_, bf16>; 3],
        m: usize,
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
        self.gemm_rows(&mut products, m, invariant)?;
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
                    start,
                    cfg.log_scaling_alpha,
                    cfg.log_scaling_n_floor as f32,
                )),
                qk_threshold: cfg.qk_l2_clip_threshold,
                rel_threshold: cfg.rel_l2_clip_threshold,
                eps: self.eps(),
            },
            m,
        )?;
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
        self.kernels.attention.relative_bias_rows(
            &self.stream,
            &rows(&t.r, m, rel_dims),
            &a.rel_table,
            &mut rows_mut(&mut t.bias, m, heads * cfg.rel_extent),
        )?;

        if invariant {
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
                &mut t.rows,
                &mut rows_mut(&mut t.out, m, q_dims),
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
        }
        self.kernels.elementwise.attention_gate(
            &self.stream,
            &rows(&t.out, m, q_dims),
            &rows(&t.gate, m, q_dims),
            &mut rows_mut(&mut t.gated, m, q_dims),
        )?;
        self.gemm_rows(
            &mut [RowProduct {
                weight: &a.o,
                fp8: w.fp8.as_ref().map(|f| &f.o),
                x: rows(&t.gated, m, q_dims),
                out: rows_mut(&mut t.proj, m, hidden),
            }],
            m,
            invariant,
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
                &mut t.router_counter,
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
}
