//! Chunked prefill: a prompt runs through the model a chunk of rows at a time, every
//! layer visit for the whole chunk before the next visit.
//!
//! Each chunk fills the KV caches and conv states as stepping its tokens one at a time
//! would, so [`Model::step`] continues after it. The small kernels have multi-row forms
//! that give the single-row results bit for bit. The three heavy operations go through
//! [`Ctx::gemm_rows`], [`Ctx::moe_rows`] and [`Ctx::attention_rows`].
//!
//! A chunk's buffers hold `[rows, ...]` per intermediate. The block states of one chunk
//! are `[blocks, rows, 2304]`, so the states in use are one contiguous run.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaSlice, CudaStream, CudaView, CudaViewMut};
use half::bf16;
use inkling::TextConfig;

use super::{Blocks, Ctx, MESSAGE_SLOTS, Model, RowProduct, Scratch, VisitState, zeros};
use crate::{
    ops::{
        attention,
        experts::ExpertScratch,
        norm::{AttentionPostRows, KvConvWeights},
        routing::{NormDest, Partial},
    },
    weights::{InputNorms, KvConv, LayerWeights, MessageGateWeights},
};

/// Most rows one chunk may hold.
pub const MAX_PREFILL_CHUNK: usize = 1024;
/// The chunk size MLX's chunked captures use.
pub const DEFAULT_PREFILL_CHUNK: usize = 512;
/// Rows the LM head runs at once when every row's logits are wanted: 32 rows of 201,024
/// logits take 13 MB in BF16 and 26 MB after the F32 softcap.
const HEAD_ROWS: usize = 32;

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
    /// Per slot, the normalized state and message side by side, BF16 `[rows, 2 * hidden]`.
    inputs: [CudaSlice<bf16>; MESSAGE_SLOTS],
    down: CudaSlice<bf16>,
    counters: CudaSlice<u32>,
}

/// Intermediates of attention for a chunk, `[rows, ...]` versions of `AttentionTemps`.
struct AttentionRows {
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
    logits: CudaSlice<f32>,
    counters: CudaSlice<u32>,
    indices: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    gate_up: CudaSlice<bf16>,
    act: CudaSlice<bf16>,
    down: CudaSlice<bf16>,
    /// The combined expert outputs, the post-MLP conv's input.
    combined: CudaSlice<bf16>,
    /// Grouping of the (row, slot) pairs by expert, shared by w13 and w2.
    experts: ExpertScratch,
}

/// Every intermediate of a chunk of up to `rows` rows. Names follow [`Scratch`].
pub(super) struct PrefillScratch {
    rows: usize,
    tokens: CudaSlice<i32>,
    /// The position of the chunk's first row.
    start: CudaSlice<u32>,
    embedded: CudaSlice<bf16>,
    hidden: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    prelude_hidden: CudaSlice<bf16>,
    blocks: CudaSlice<f32>,
    bank: CudaSlice<f32>,
    bank_normed: CudaSlice<f32>,
    /// F32 `[rows, 4, hidden]`.
    messages: CudaSlice<f32>,
    normed_residual: CudaSlice<f32>,
    /// F32 `[rows, 1 or 3, hidden]`.
    attention_inputs: CudaSlice<f32>,
    gated: CudaSlice<f32>,
    gate: GateRows,
    qkv_in: [CudaSlice<bf16>; 3],
    attention: AttentionRows,
    residual: CudaSlice<f32>,
    mlp_input: CudaSlice<f32>,
    mlp_x: CudaSlice<bf16>,
    moe: MoeRows,
    final_hidden: CudaSlice<bf16>,
    head_logits: CudaSlice<bf16>,
    head_capped: CudaSlice<f32>,
    /// Host copy of `head_capped`.
    head_host: Vec<f32>,
}

impl PrefillScratch {
    fn new(
        stream: &Arc<CudaStream>,
        cfg: &TextConfig,
        max_blocks: usize,
        rows: usize,
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
            tokens: zeros(stream, rows)?,
            start: zeros(stream, 1)?,
            embedded: zeros(stream, rows * hidden)?,
            hidden: zeros(stream, rows * hidden)?,
            out: zeros(stream, rows * hidden)?,
            prelude_hidden: zeros(stream, rows * hidden)?,
            blocks: zeros(stream, rows * max_blocks * hidden)?,
            bank: zeros(stream, rows * max_blocks * hidden)?,
            bank_normed: zeros(stream, rows * hidden)?,
            messages: zeros(stream, rows * MESSAGE_SLOTS * hidden)?,
            normed_residual: zeros(stream, rows * hidden)?,
            attention_inputs: zeros(stream, rows * 3 * hidden)?,
            gated: zeros(stream, rows * hidden)?,
            gate: GateRows {
                inputs: [
                    zeros(stream, rows * 2 * hidden)?,
                    zeros(stream, rows * 2 * hidden)?,
                    zeros(stream, rows * 2 * hidden)?,
                    zeros(stream, rows * 2 * hidden)?,
                ],
                down: zeros(stream, rows * cfg.loop_config.gate_rank)?,
                counters: zeros(stream, rows)?,
            },
            qkv_in: [
                zeros(stream, rows * hidden)?,
                zeros(stream, rows * hidden)?,
                zeros(stream, rows * hidden)?,
            ],
            attention: AttentionRows {
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
                logits: zeros(stream, rows * cfg.n_routed_experts)?,
                counters: zeros(stream, rows)?,
                indices: zeros(stream, rows * top_k)?,
                weights: zeros(stream, rows * top_k)?,
                gate_up: zeros(stream, rows * top_k * 2 * width)?,
                act: zeros(stream, rows * top_k * width)?,
                down: zeros(stream, rows * top_k * hidden)?,
                combined: zeros(stream, rows * hidden)?,
                experts: ExpertScratch::new(stream, rows, cfg.n_routed_experts)?,
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
fn strided<T>(
    x: &CudaSlice<T>,
    rows: usize,
    stride: usize,
    offset: usize,
    width: usize,
) -> CudaView<'_, T> {
    x.slice(offset..offset + (rows - 1) * stride + width)
}

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
        mut logits: PrefillLogits<'_>,
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
        let rows = chunk.min(tokens.len());
        if self.prefill.as_ref().is_none_or(|p| p.rows < rows) {
            // Free the smaller buffers before allocating the new ones.
            self.prefill = None;
            self.prefill = Some(PrefillScratch::new(
                &self.ctx.stream,
                &self.ctx.config,
                self.max_blocks,
                rows,
            )?);
        }
        let pieces = tokens.chunks(chunk);
        let count = pieces.len();
        for (i, piece) in pieces.enumerate() {
            self.prefill_chunk(piece, &mut logits, i + 1 == count)?;
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

/// What [`Ctx::attention_inputs_rows`] reads and writes, as `AttentionInputs` for a chunk.
struct AttentionInputsRows<'a> {
    x: CudaView<'a, bf16>,
    states: CudaView<'a, f32>,
    normed: CudaViewMut<'a, f32>,
    inputs: &'a mut CudaSlice<f32>,
    qkv_in: &'a mut [CudaSlice<bf16>; 3],
    gate: &'a mut GateRows,
    gated: &'a mut CudaSlice<f32>,
    messages: Option<(&'a [MessageGateWeights; MESSAGE_SLOTS], &'a CudaSlice<f32>)>,
}

impl Ctx {
    /// One chunk of `m` rows through the model, as `Model::enqueue` for one row. On the
    /// final chunk, the last row's logits and greedy token go to the decode scratch's
    /// `capped` and `next`. Other chunks skip the final norm and the LM head unless
    /// `logits` asks for every row.
    #[expect(clippy::too_many_arguments, reason = "the parts of Model a chunk uses")]
    #[expect(
        clippy::too_many_lines,
        reason = "one chunk reads top to bottom like _run_loop"
    )]
    fn prefill_enqueue(
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
        let mut visits = visits.iter_mut();
        let mut next_visit = || visits.next().context("fewer visit caches than visits");

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
        let mut bank_rows = 0;
        for pass in 0..passes {
            if pass > 0 {
                self.kernels.norm.unweighted_rms_norm_bf16(
                    &self.stream,
                    &rows(&p.hidden, m, hidden),
                    &mut rows_mut(&mut p.bank_normed, m, hidden),
                    self.eps(),
                )?;
                bank_rows = blocks.len.saturating_sub(prelude_blocks + 1);
                ensure!(bank_rows > 0, "the message bank needs a block state");
                self.stream.memcpy_dtod(
                    &p.blocks.slice((prelude_blocks + 1) * mh..blocks.len * mh),
                    &mut p.bank.slice_mut(..bank_rows * mh),
                )?;
                self.stream.memcpy_dtod(
                    &rows(&p.prelude_hidden, m, hidden),
                    &mut rows_mut(&mut p.hidden, m, hidden),
                )?;
                blocks.len = prelude_blocks;
            }
            for (i, layer) in (core_start..core_end).enumerate() {
                if pass > 0 {
                    // The messages, and their normalized halves of the gate inputs.
                    let dests = p.gate.inputs.each_mut().map(|x| {
                        Some(NormDest {
                            out: strided_mut(x, m, 2 * hidden, hidden, hidden),
                            weight: None,
                        })
                    });
                    self.kernels.routing.depth_attention_fused_rows(
                        &self.stream,
                        &p.bank.slice(..bank_rows * mh),
                        Partial::Normalized(rows(&p.bank_normed, m, hidden)),
                        None,
                        &self.weights.message_queries[i],
                        Some(rows_mut(&mut p.messages, m, MESSAGE_SLOTS * hidden)),
                        dests,
                        2 * hidden,
                        self.eps(),
                        m,
                    )?;
                }
                self.visit_rows(p, next_visit()?, layer, &mut blocks, pass > 0, m)?;
            }
        }
        for layer in core_end..cfg.num_hidden_layers {
            self.visit_rows(p, next_visit()?, layer, &mut blocks, false, m)?;
        }
        if !final_chunk && matches!(logits, PrefillLogits::Last) {
            return Ok(());
        }

        self.kernels.routing.depth_attention_fused_rows(
            &self.stream,
            &blocks.rows(&p.blocks, mh)?,
            Partial::Bf16(rows(&p.hidden, m, hidden)),
            None,
            &self.weights.final_query,
            None,
            [
                Some(NormDest {
                    out: rows_mut(&mut p.final_hidden, m, hidden),
                    weight: Some(&self.weights.norm),
                }),
                None,
                None,
                None,
            ],
            hidden,
            self.eps(),
            m,
        )?;

        // The last row's logits and greedy token, in the decode buffers.
        self.gemm_rows(
            &mut [RowProduct {
                weight: &self.weights.lm_head,
                x: p.final_hidden.slice((m - 1) * hidden..mh),
                out: s.logits.as_view_mut(),
            }],
            1,
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
                    x: p.final_hidden.slice(first * hidden..(first + n) * hidden),
                    out: rows_mut(&mut p.head_logits, n, padded),
                }],
                n,
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

    /// One decoder-layer visit for a chunk, as [`Ctx::visit`] for one row, from
    /// `p.hidden` to `p.out`; the two then swap.
    #[expect(
        clippy::too_many_lines,
        reason = "follows DecoderLayer.__call__ op by op"
    )]
    fn visit_rows(
        &self,
        p: &mut PrefillScratch,
        state: &mut VisitState,
        layer: usize,
        blocks: &mut Blocks,
        messages: bool,
        m: usize,
    ) -> anyhow::Result<()> {
        let w = &self.weights.layers[layer];
        let sliding = self.config.is_sliding(layer);
        let hidden = self.hidden();
        let mh = m * hidden;
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
        let (states, mut rest) = p.blocks.split_at_mut(blocks.len * mh);
        let normed = if block_start {
            rest.slice_mut(..mh)
        } else {
            rows_mut(&mut p.normed_residual, m, hidden)
        };
        let one_input = self.attention_inputs_rows(
            w,
            AttentionInputsRows {
                x: rows(&p.hidden, m, hidden),
                states: states.slice(..),
                normed,
                inputs: &mut p.attention_inputs,
                qkv_in: &mut p.qkv_in,
                gate: &mut p.gate,
                gated: &mut p.gated,
                messages: gates.map(|g| (g, &p.messages)),
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

        let block_hidden = rows(&p.hidden, m, hidden);
        self.kernels.conv.short_conv_residual_rows(
            &self.stream,
            &rows(&p.attention.proj, m, hidden),
            &w.post_attention_conv,
            &mut state.conv[2],
            (!block_start).then_some(&block_hidden as &dyn crate::DeviceIn<bf16>),
            &mut rows_mut(&mut p.residual, m, hidden),
            m,
        )?;

        let gate_dest = gates.map(|_| NormDest {
            out: strided_mut(&mut p.gate.inputs[3], m, 2 * hidden, 0, hidden),
            weight: None,
        });
        self.kernels.routing.depth_attention_fused_rows(
            &self.stream,
            &blocks.rows(&p.blocks, mh)?,
            Partial::F32(rows(&p.residual, m, hidden)),
            None,
            &w.mlp_query,
            Some(rows_mut(&mut p.mlp_input, m, hidden)),
            [None, None, None, gate_dest],
            2 * hidden,
            self.eps(),
            m,
        )?;
        let mlp_input = match gates {
            Some(gates) => {
                self.message_gate_rows(
                    &gates[3],
                    &mut p.gate,
                    3,
                    (&rows(&p.mlp_input, m, hidden), hidden),
                    (
                        &strided(&p.messages, m, MESSAGE_SLOTS * hidden, 3 * hidden, hidden),
                        MESSAGE_SLOTS * hidden,
                    ),
                    rows_mut(&mut p.gated, m, hidden),
                    None,
                    m,
                )?;
                &p.gated
            }
            None => &p.mlp_input,
        };
        self.moe_block_rows(w, &mut p.moe, &rows(mlp_input, m, hidden), &mut p.mlp_x, m)?;

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
            let mut normed = a.normed;
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
        self.kernels.routing.depth_attention_fused_rows(
            &self.stream,
            &a.states,
            Partial::Bf16(a.x),
            Some(a.normed),
            query,
            a.messages
                .is_some()
                .then(|| a.inputs.slice_mut(..m * count * hidden)),
            [d0, d1, d2, None],
            dst_stride,
            self.eps(),
            m,
        )?;
        let Some((gates, messages)) = a.messages else {
            return Ok(shared);
        };
        for (i, norm) in norms.into_iter().enumerate() {
            let slot = if shared { 0 } else { i };
            self.message_gate_rows(
                &gates[i],
                &mut *a.gate,
                i,
                (
                    &strided(a.inputs, m, count * hidden, slot * hidden, hidden),
                    count * hidden,
                ),
                (
                    &strided(messages, m, MESSAGE_SLOTS * hidden, i * hidden, hidden),
                    MESSAGE_SLOTS * hidden,
                ),
                rows_mut(a.gated, m, hidden),
                Some((norm, a.qkv_in[i].slice_mut(..mh))),
                m,
            )?;
        }
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
            weight: &'a CudaSlice<bf16>,
            x: &'a CudaView<'_, bf16>,
            out: CudaViewMut<'a, bf16>,
        ) -> RowProduct<'a> {
            RowProduct {
                weight,
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

        let mut products = vec![
            product(&a.q, &q_in, rows_mut(&mut t.q_raw, m, q_dims)),
            product(&a.r, &q_in, rows_mut(&mut t.r_raw, m, rel_dims)),
            product(&a.gate, &q_in, rows_mut(&mut t.gate, m, q_dims)),
            product(&a.sink, &q_in, rows_mut(&mut t.sinks, m, heads)),
            product(&a.k, &k_in, rows_mut(&mut t.k_raw, m, kv_dims)),
            product(&a.v, &v_in, rows_mut(&mut t.v_raw, m, kv_dims)),
        ];
        if let KvConv::Dynamic { k_proj, v_proj, .. } = &a.kv_conv {
            products.push(product(k_proj, &k_in, rows_mut(&mut t.k_dynamic, m, taps)));
            products.push(product(v_proj, &v_in, rows_mut(&mut t.v_dynamic, m, taps)));
        }
        self.gemm_rows(&mut products, m)?;
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
        self.kernels.attention.relative_bias(
            &self.stream,
            &rows(&t.r, m, rel_dims),
            &a.rel_table,
            &mut rows_mut(&mut t.bias, m, heads * cfg.rel_extent),
        )?;

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
        self.gemm_rows(
            &mut [RowProduct {
                weight: &a.o,
                x: rows(&t.gated, m, q_dims),
                out: rows_mut(&mut t.proj, m, hidden),
            }],
            m,
        )
    }

    /// [`Ctx::message_gate`] for a chunk of `m` rows: `h` and `message` are rows with
    /// their strides, and `norm` the norm weight and destination rows.
    #[expect(clippy::too_many_arguments, reason = "the gate's inputs and outputs")]
    fn message_gate_rows(
        &self,
        g: &MessageGateWeights,
        t: &mut GateRows,
        slot: usize,
        h: (&CudaView<'_, f32>, usize),
        message: (&CudaView<'_, f32>, usize),
        mut out: CudaViewMut<'_, f32>,
        norm: Option<(&CudaSlice<bf16>, CudaViewMut<'_, bf16>)>,
        m: usize,
    ) -> anyhow::Result<()> {
        let hidden = self.hidden();
        let rank = self.config.loop_config.gate_rank;
        self.gemm_rows(
            &mut [RowProduct {
                weight: &g.down,
                x: rows(&t.inputs[slot], m, 2 * hidden),
                out: rows_mut(&mut t.down, m, rank),
            }],
            m,
        )?;
        let mut counters = rows_mut(&mut t.counters, m, 1);
        let mut norm = norm;
        let norm = norm.as_mut().map(|(weight, normed)| {
            (
                *weight,
                normed as &mut dyn crate::DeviceOut<bf16>,
                &mut counters as &mut dyn crate::DeviceOut<u32>,
            )
        });
        self.kernels.misc.message_gate_out_rows(
            &self.stream,
            &rows(&t.down, m, rank),
            &g.forget,
            &g.gain,
            h,
            message,
            &mut out,
            norm,
            self.eps(),
            m,
        )
    }

    /// [`Ctx::moe`] for a chunk of `m` rows: the router, then the expert products and
    /// SwiGLU into `t.down`.
    fn moe_block_rows(
        &self,
        w: &LayerWeights,
        t: &mut MoeRows,
        input: &CudaView<'_, f32>,
        x: &mut CudaSlice<bf16>,
        m: usize,
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let width = cfg.intermediate_size;
        let top_k = cfg.num_experts_per_tok;
        self.kernels.routing.moe_router_rows(
            &self.stream,
            input,
            &w.mlp_norm,
            self.eps(),
            &w.router,
            &w.router_bias,
            &mut rows_mut(x, m, hidden),
            &mut rows_mut(&mut t.logits, m, cfg.n_routed_experts),
            &mut rows_mut(&mut t.indices, m, top_k),
            &mut rows_mut(&mut t.weights, m, top_k),
            &mut rows_mut(&mut t.counters, m, 1),
            cfg.route_scale,
            m,
        )?;
        let (w13, w2) = self.expert_weights(w);
        let indices = rows(&t.indices, m, top_k);
        self.moe_rows(
            &rows(x, m, hidden),
            &indices,
            &w13,
            &mut rows_mut(&mut t.gate_up, m, top_k * 2 * width),
            &mut t.experts,
            m,
        )?;
        self.kernels.elementwise.softcapped_swiglu(
            &self.stream,
            &rows(&t.gate_up, m, top_k * 2 * width),
            &mut rows_mut(&mut t.act, m, top_k * width),
            cfg.mlp_gate_softcap,
            cfg.mlp_up_softcap,
        )?;
        self.moe_rows(
            &rows(&t.act, m, top_k * width),
            &indices,
            &w2,
            &mut rows_mut(&mut t.down, m, top_k * hidden),
            &mut t.experts,
            m,
        )
    }
}
