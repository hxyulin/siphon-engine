//! Single-stream decode, one token per step, following MLX-LM's `inkling2` model.
//!
//! The structure follows `TextModel._run_loop` and `DecoderLayer.__call__` in MLX-LM so
//! the two can be read side by side.
//!
//! Every intermediate lives in a [`Scratch`] buffer allocated in [`Model::new`]; visits
//! run one after another, so one set serves all of them. The kernels read the token and
//! its position from device memory, so a step launches the same work at every position
//! and never waits on the host. An untraced step replays that work as one CUDA graph,
//! captured on the first untraced step.

use std::{ops::Range, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaGraph, CudaSlice, CudaStream, CudaView, DeviceRepr, ValidAsZeroBits,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
};
use half::bf16;
use inkling::TextConfig;

use crate::{
    DeviceIn, Gpu,
    ops::{
        attention::{Attention, AttentionScratch, KvCache},
        conv::Conv,
        elementwise::Elementwise,
        experts::{self, Experts},
        linear::{Gemv, Linear},
        misc::Misc,
        norm::Norm,
        routing::{RouterMatmul, Routing},
    },
    weights::{ExpertFormat, InputNorms, KvConv, LayerWeights, MessageGateWeights, Weights},
};

/// Queries, keys, values and the MLP each get a message on looped layers.
const MESSAGE_SLOTS: usize = 4;

#[derive(Debug)]
struct Kernels {
    attention: Attention,
    conv: Conv,
    elementwise: Elementwise,
    experts: Experts,
    linear: Linear,
    misc: Misc,
    norm: Norm,
    routing: Routing,
}

/// The caches one layer visit keeps across tokens.
#[derive(Debug)]
struct VisitState {
    kv: KvCache,
    /// Short-conv input history, F32 `[kernel - 1, dims]`: keys, values, post-attention
    /// and post-MLP, as in MLX-LM's `ArraysCache(4)`.
    conv: [CudaSlice<f32>; 4],
}

/// Inputs and outputs of each visit, read back for parity checks.
#[derive(Debug, Default)]
pub struct Trace {
    pub visits: Vec<VisitTrace>,
    /// F32 logits after the softcap, `[unpadded_vocab]`.
    pub logits: Vec<f32>,
}

#[derive(Debug)]
pub struct VisitTrace {
    pub input: Vec<bf16>,
    pub output: Vec<bf16>,
    /// The block states after the visit, F32 `[blocks, hidden]`.
    pub blocks: Vec<f32>,
}

/// Every intermediate of a step. Names follow the MLX-LM code that produces them.
struct Scratch {
    /// The step's token, uploaded before the step.
    token: CudaSlice<i32>,
    /// The step's position, uploaded before the step and read by the KV append, decode
    /// attention and log scaling.
    position: CudaSlice<u32>,
    /// The greedy next token, read back after the step.
    next: CudaSlice<i32>,
    embedded: CudaSlice<bf16>,
    /// The input of the current visit; `out` is its output, and the two swap after it.
    hidden: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    /// The hidden state after the prelude, the input of every pass.
    prelude_hidden: CudaSlice<bf16>,
    /// Block states for depth attention, F32 `[max_blocks, hidden]`: the normalized residual
    /// at each block boundary. The host counts the rows in use.
    blocks: CudaSlice<f32>,
    /// The message bank, F32 `[max_blocks, hidden]`: the block states the previous pass
    /// added after its first one. The host counts the rows in use.
    bank: CudaSlice<f32>,
    /// The normalized hidden state at the end of the previous pass.
    bank_normed: CudaSlice<f32>,
    /// Depth attention over the bank, F32 `[4, hidden]`.
    messages: CudaSlice<f32>,
    /// F32 copy of a BF16 row, before normalization.
    wide: CudaSlice<f32>,
    normed_residual: CudaSlice<f32>,
    /// Depth attention for the attention inputs, F32 `[1 or 3, hidden]`.
    attention_inputs: CudaSlice<f32>,
    /// A message gate's output.
    gated: CudaSlice<f32>,
    gate: GateScratch,
    /// Normalized query, key and value inputs of attention. Layer 0 uses only the first.
    qkv_in: [CudaSlice<bf16>; 3],
    attention: AttentionTemps,
    attn_conv: CudaSlice<bf16>,
    residual: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    mlp_input: CudaSlice<f32>,
    mlp_x: CudaSlice<bf16>,
    moe: MoeScratch,
    mlp_conv: CudaSlice<bf16>,
    final_normed: CudaSlice<f32>,
    final_input: CudaSlice<f32>,
    final_hidden: CudaSlice<bf16>,
    logits: CudaSlice<bf16>,
    /// F32 logits after the softcap, `[unpadded_vocab]`.
    capped: CudaSlice<f32>,
}

/// Intermediates of `MessageGate`.
struct GateScratch {
    /// The normalized state and message side by side, F32 `[2 * hidden]`.
    both: CudaSlice<f32>,
    both_bf16: CudaSlice<bf16>,
    down: CudaSlice<bf16>,
    down_f32: CudaSlice<f32>,
    forget: CudaSlice<f32>,
    gain: CudaSlice<f32>,
}

/// Intermediates of `SlidingAttention` and `FullAttention`.
struct AttentionTemps {
    k_raw: CudaSlice<bf16>,
    v_raw: CudaSlice<bf16>,
    k_dynamic: CudaSlice<bf16>,
    v_dynamic: CudaSlice<bf16>,
    k_conv: CudaSlice<bf16>,
    v_conv: CudaSlice<bf16>,
    q_raw: CudaSlice<bf16>,
    r_raw: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    sinks: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    r: CudaSlice<bf16>,
    bias: CudaSlice<bf16>,
    partials: AttentionScratch,
    out: CudaSlice<bf16>,
    gated: CudaSlice<bf16>,
    /// The output projection, BF16 `[hidden]`.
    proj: CudaSlice<bf16>,
}

/// Intermediates of `SparseMoeBlock`.
struct MoeScratch {
    logits: CudaSlice<f32>,
    indices: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    gate_up: CudaSlice<bf16>,
    act: CudaSlice<bf16>,
    down: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
}

fn zeros<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> anyhow::Result<CudaSlice<T>> {
    Ok(stream.alloc_zeros(len)?)
}

impl Scratch {
    fn new(stream: &Arc<CudaStream>, cfg: &TextConfig, max_blocks: usize) -> anyhow::Result<Self> {
        let hidden = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim;
        let kv_dims = cfg.num_key_value_heads * head_dim;
        let dynamic = cfg.num_key_value_heads * cfg.sconv_kernel_size;
        let rank = cfg.loop_config.gate_rank;
        let top_k = cfg.num_experts_per_tok;
        let width = cfg.intermediate_size;
        Ok(Self {
            token: zeros(stream, 1)?,
            position: zeros(stream, 1)?,
            next: zeros(stream, 1)?,
            embedded: zeros(stream, hidden)?,
            hidden: zeros(stream, hidden)?,
            out: zeros(stream, hidden)?,
            prelude_hidden: zeros(stream, hidden)?,
            blocks: zeros(stream, max_blocks * hidden)?,
            bank: zeros(stream, max_blocks * hidden)?,
            bank_normed: zeros(stream, hidden)?,
            messages: zeros(stream, MESSAGE_SLOTS * hidden)?,
            wide: zeros(stream, hidden)?,
            normed_residual: zeros(stream, hidden)?,
            attention_inputs: zeros(stream, 3 * hidden)?,
            gated: zeros(stream, hidden)?,
            gate: GateScratch {
                both: zeros(stream, 2 * hidden)?,
                both_bf16: zeros(stream, 2 * hidden)?,
                down: zeros(stream, rank)?,
                down_f32: zeros(stream, rank)?,
                forget: zeros(stream, hidden)?,
                gain: zeros(stream, hidden)?,
            },
            qkv_in: [
                zeros(stream, hidden)?,
                zeros(stream, hidden)?,
                zeros(stream, hidden)?,
            ],
            attention: AttentionTemps {
                k_raw: zeros(stream, kv_dims)?,
                v_raw: zeros(stream, kv_dims)?,
                k_dynamic: zeros(stream, dynamic)?,
                v_dynamic: zeros(stream, dynamic)?,
                k_conv: zeros(stream, kv_dims)?,
                v_conv: zeros(stream, kv_dims)?,
                q_raw: zeros(stream, heads * head_dim)?,
                r_raw: zeros(stream, heads * cfg.d_rel)?,
                gate: zeros(stream, heads * head_dim)?,
                sinks: zeros(stream, heads)?,
                q: zeros(stream, heads * head_dim)?,
                k: zeros(stream, kv_dims)?,
                v: zeros(stream, kv_dims)?,
                r: zeros(stream, heads * cfg.d_rel)?,
                bias: zeros(stream, heads * cfg.rel_extent)?,
                partials: AttentionScratch::new(stream)?,
                out: zeros(stream, heads * head_dim)?,
                gated: zeros(stream, heads * head_dim)?,
                proj: zeros(stream, hidden)?,
            },
            attn_conv: zeros(stream, hidden)?,
            residual: zeros(stream, hidden)?,
            normed: zeros(stream, hidden)?,
            mlp_input: zeros(stream, hidden)?,
            mlp_x: zeros(stream, hidden)?,
            moe: MoeScratch {
                logits: zeros(stream, cfg.n_routed_experts)?,
                indices: zeros(stream, top_k)?,
                weights: zeros(stream, top_k)?,
                gate_up: zeros(stream, top_k * 2 * width)?,
                act: zeros(stream, top_k * width)?,
                down: zeros(stream, top_k * hidden)?,
                out: zeros(stream, hidden)?,
            },
            mlp_conv: zeros(stream, hidden)?,
            final_normed: zeros(stream, hidden)?,
            final_input: zeros(stream, hidden)?,
            final_hidden: zeros(stream, hidden)?,
            logits: zeros(stream, cfg.vocab_size)?,
            capped: zeros(stream, cfg.unpadded_vocab_size)?,
        })
    }
}

/// The captured untraced step.
struct StepGraph(CudaGraph);

// SAFETY: a graph exec may be launched from any thread with its context bound, which
// `CudaGraph::launch` does. The runner launches it only through `&mut Model`, so launches
// never overlap.
unsafe impl Send for StepGraph {}

pub struct Model {
    ctx: Ctx,
    passes: usize,
    visits: Vec<VisitState>,
    scratch: Scratch,
    /// Rows of `scratch.blocks` a step can fill.
    max_blocks: usize,
    graph: Option<StepGraph>,
    /// Position of the next token.
    position: usize,
}

/// Everything a step reads but does not change.
struct Ctx {
    config: TextConfig,
    weights: Weights,
    kernels: Kernels,
    stream: Arc<CudaStream>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("passes", &self.passes)
            .field("position", &self.position)
            .field("graph", &self.graph.is_some())
            .finish_non_exhaustive()
    }
}

/// Layers in `layers` that start a block state.
fn block_starts(config: &TextConfig, layers: Range<usize>) -> usize {
    layers
        .filter(|l| l.is_multiple_of(config.n_layers_per_attn_res_block))
        .count()
}

impl Model {
    /// Builds a runner with the looped core run `passes` times and caches for
    /// `capacity` tokens in the global layers.
    pub fn new(
        gpu: &Gpu,
        config: TextConfig,
        weights: Weights,
        passes: usize,
        capacity: usize,
    ) -> anyhow::Result<Self> {
        let ctx = gpu.context();
        let stream = Arc::clone(gpu.stream());
        let kernels = Kernels {
            attention: Attention::load(ctx)?,
            conv: Conv::load(ctx)?,
            elementwise: Elementwise::load(ctx)?,
            experts: Experts::load(ctx)?,
            linear: Linear::load(ctx)?,
            misc: Misc::load(ctx)?,
            norm: Norm::load(ctx)?,
            routing: Routing::load(ctx)?,
        };
        let history = config.sconv_kernel_size - 1;
        let kv_dims = config.num_key_value_heads * config.head_dim;
        let hidden = config.hidden_size;
        let visits = config
            .visit_layers(passes)
            .into_iter()
            .map(|layer| {
                let kv = if config.is_sliding(layer) {
                    KvCache::sliding(&stream, config.sliding_window_size)?
                } else {
                    KvCache::global(&stream, capacity)?
                };
                Ok(VisitState {
                    kv,
                    conv: [
                        stream.alloc_zeros(history * kv_dims)?,
                        stream.alloc_zeros(history * kv_dims)?,
                        stream.alloc_zeros(history * hidden)?,
                        stream.alloc_zeros(history * hidden)?,
                    ],
                })
            })
            .collect::<anyhow::Result<_>>()?;
        let loops = config.loop_config;
        // Each pass starts again from the prelude's block states.
        let max_blocks = block_starts(&config, 0..loops.core_start)
            + block_starts(&config, loops.core_start..loops.core_end)
            + block_starts(&config, loops.core_end..config.num_hidden_layers);
        let scratch = Scratch::new(&stream, &config, max_blocks)?;
        Ok(Self {
            ctx: Ctx {
                config,
                weights,
                kernels,
                stream,
            },
            passes,
            visits,
            scratch,
            max_blocks,
            graph: None,
            position: 0,
        })
    }

    /// Position of the next token.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Moves every cache to `position` without computing the skipped tokens, for decode
    /// benchmarks at long contexts. Outputs after this are not meaningful.
    pub fn skip_to_for_benchmark(&mut self, position: usize) -> anyhow::Result<()> {
        ensure!(
            self.visits.iter().all(|v| v.kv.holds(position)),
            "the global KV caches cannot hold position {position}"
        );
        self.position = position;
        Ok(())
    }

    /// The F32 logits of the last step after the softcap, `[unpadded_vocab]`.
    pub fn logits(&self) -> anyhow::Result<Vec<f32>> {
        Ok(self.ctx.stream.clone_dtoh(&self.scratch.capped)?)
    }

    /// Runs one token and returns the greedy next token. Fills `trace` if given; otherwise
    /// replays the step's CUDA graph, capturing it first if needed.
    pub fn step(&mut self, token: u32, trace: Option<&mut Trace>) -> anyhow::Result<u32> {
        let position = self.position;
        ensure!(
            self.visits.iter().all(|v| v.kv.holds(position)),
            "the global KV caches are full at {position} tokens"
        );
        let stream = Arc::clone(&self.ctx.stream);
        let s = &mut self.scratch;
        stream.memcpy_htod(&[i32::try_from(token)?], &mut s.token)?;
        stream.memcpy_htod(
            &[u32::try_from(position).context("position does not fit the kernels")?],
            &mut s.position,
        )?;
        if let Some(trace) = trace {
            *trace = Trace::default();
            self.enqueue(Some(trace))?;
        } else {
            if self.graph.is_none() {
                self.graph = Some(self.capture()?);
            }
            let graph = self.graph.as_ref().context("the step graph is missing")?;
            graph.0.launch()?;
        }
        let next = stream.clone_dtoh(&self.scratch.next)?[0];
        self.position += 1;
        u32::try_from(next).context("argmax returned a negative index")
    }

    /// Records the untraced step as a CUDA graph.
    fn capture(&mut self) -> anyhow::Result<StepGraph> {
        let stream = Arc::clone(&self.ctx.stream);
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let enqueued = self.enqueue(None);
        // End the capture even when enqueueing failed, so the stream is usable again.
        let graph = stream.end_capture(CUgraphInstantiate_flags(0));
        enqueued.context("capturing the decode step")?;
        let graph = graph?.context("the step capture recorded no work")?;
        graph.upload()?;
        Ok(StepGraph(graph))
    }

    /// Enqueues one step on the stream, for the token and position already uploaded.
    /// Synchronizes only to fill `trace`.
    #[expect(
        clippy::too_many_lines,
        reason = "one step reads top to bottom like _run_loop"
    )]
    fn enqueue(&mut self, mut trace: Option<&mut Trace>) -> anyhow::Result<()> {
        let ctx = &self.ctx;
        let s = &mut self.scratch;
        let cfg = &ctx.config;
        let (core_start, core_end) = (cfg.loop_config.core_start, cfg.loop_config.core_end);
        let hidden = ctx.hidden();
        let mut visits = self.visits.iter_mut();
        let mut next_visit = || visits.next().context("fewer visit caches than visits");

        ctx.kernels.elementwise.embedding(
            &ctx.stream,
            &s.token,
            &ctx.weights.embed,
            &mut s.embedded,
        )?;
        ctx.kernels.norm.rms_norm(
            &ctx.stream,
            &s.embedded,
            &ctx.weights.embed_norm,
            &mut s.hidden,
            ctx.eps(),
        )?;

        let mut blocks = Blocks {
            len: 0,
            max: self.max_blocks,
        };
        for layer in 0..core_start {
            ctx.visit(
                s,
                next_visit()?,
                layer,
                &mut blocks,
                false,
                trace.as_deref_mut(),
            )?;
        }
        ctx.stream.memcpy_dtod(&s.hidden, &mut s.prelude_hidden)?;
        let prelude_blocks = blocks.len;
        let mut bank_rows = 0;
        for pass in 0..self.passes {
            if pass > 0 {
                // The message bank: block states the pass added after its first one, and
                // the normalized hidden state at the end of the pass.
                ctx.normed_f32(&s.hidden, &mut s.wide, &mut s.bank_normed)?;
                bank_rows = blocks.len.saturating_sub(prelude_blocks + 1);
                ensure!(bank_rows > 0, "the message bank needs a block state");
                ctx.stream.memcpy_dtod(
                    &s.blocks
                        .slice((prelude_blocks + 1) * hidden..blocks.len * hidden),
                    &mut s.bank.slice_mut(..bank_rows * hidden),
                )?;
                ctx.stream.memcpy_dtod(&s.prelude_hidden, &mut s.hidden)?;
                blocks.len = prelude_blocks;
            }
            for (i, layer) in (core_start..core_end).enumerate() {
                if pass > 0 {
                    ctx.kernels.routing.depth_attention(
                        &ctx.stream,
                        &s.bank.slice(..bank_rows * hidden),
                        &s.bank_normed,
                        &ctx.weights.message_queries[i],
                        &mut s.messages,
                    )?;
                }
                ctx.visit(
                    s,
                    next_visit()?,
                    layer,
                    &mut blocks,
                    pass > 0,
                    trace.as_deref_mut(),
                )?;
            }
        }
        for layer in core_end..cfg.num_hidden_layers {
            ctx.visit(
                s,
                next_visit()?,
                layer,
                &mut blocks,
                false,
                trace.as_deref_mut(),
            )?;
        }

        ctx.normed_f32(&s.hidden, &mut s.wide, &mut s.final_normed)?;
        ctx.kernels.routing.depth_attention(
            &ctx.stream,
            &blocks.rows(&s.blocks, hidden)?,
            &s.final_normed,
            &ctx.weights.final_query,
            &mut s.final_input,
        )?;
        ctx.kernels.norm.rms_norm_f32(
            &ctx.stream,
            &s.final_input,
            &ctx.weights.norm,
            &mut s.final_hidden,
            ctx.eps(),
        )?;
        ctx.kernels.linear.gemv_bf16(
            &ctx.stream,
            &mut [Gemv::new(
                &ctx.weights.lm_head,
                &s.final_hidden,
                &mut s.logits,
            )],
        )?;
        ctx.kernels.elementwise.logit_softcap(
            &ctx.stream,
            &s.logits,
            &mut s.capped,
            cfg.final_logit_softcapping,
        )?;
        ctx.kernels.misc.argmax_f32(
            &ctx.stream,
            &s.capped,
            &mut s.next,
            cfg.unpadded_vocab_size,
        )?;
        if let Some(t) = trace {
            t.logits = ctx.stream.clone_dtoh(&s.capped)?;
        }
        Ok(())
    }
}

/// The count of block states in use, the leading rows of `Scratch::blocks`.
struct Blocks {
    len: usize,
    max: usize,
}

impl Blocks {
    /// The block states in use, F32 `[len, hidden]`.
    fn rows<'a>(
        &self,
        states: &'a CudaSlice<f32>,
        hidden: usize,
    ) -> anyhow::Result<CudaView<'a, f32>> {
        ensure!(self.len > 0, "depth attention needs a block state");
        Ok(states.slice(..self.len * hidden))
    }
}

impl Ctx {
    /// One decoder-layer visit, as MLX-LM's `DecoderLayer.__call__`, from `s.hidden` to
    /// `s.out`; the two then swap.
    #[expect(
        clippy::too_many_lines,
        reason = "follows DecoderLayer.__call__ op by op"
    )]
    fn visit(
        &self,
        s: &mut Scratch,
        state: &mut VisitState,
        layer: usize,
        blocks: &mut Blocks,
        messages: bool,
        trace: Option<&mut Trace>,
    ) -> anyhow::Result<()> {
        let w = &self.weights.layers[layer];
        let sliding = self.config.is_sliding(layer);
        let hidden = self.hidden();
        // The weights of the attention projections, and of the message gates that build
        // their inputs, stream into L2 while those inputs are built.
        let mut prefetch = Vec::new();
        if let (true, Some(gates)) = (messages, &w.message_gates) {
            prefetch.extend(gates[..3].iter().flat_map(MessageGateWeights::matrices));
        }
        prefetch.extend(w.attention.input_projections());
        self.kernels.linear.prefetch(&self.stream, &prefetch)?;
        self.normed_f32(&s.hidden, &mut s.wide, &mut s.normed_residual)?;

        let has_inputs = match &w.attention_query {
            Some(query) => {
                let count = if sliding { 1 } else { 3 };
                self.kernels.routing.depth_attention(
                    &self.stream,
                    &blocks.rows(&s.blocks, hidden)?,
                    &s.normed_residual,
                    query,
                    &mut s.attention_inputs.slice_mut(..count * hidden),
                )?;
                true
            }
            None => false,
        };
        let one_input = self.attention_inputs(w, s, has_inputs, messages)?;
        let [q_in, k_in, v_in] = &s.qkv_in;
        let (k_in, v_in) = if one_input {
            (q_in, q_in)
        } else {
            (k_in, v_in)
        };
        self.attention(
            state,
            &mut s.attention,
            &s.position,
            w,
            sliding,
            [q_in, k_in, v_in],
        )?;
        if let (true, Some(gates)) = (messages, &w.message_gates) {
            self.kernels
                .linear
                .prefetch(&self.stream, &gates[3].matrices())?;
        }

        self.kernels.conv.short_conv(
            &self.stream,
            &s.attention.proj,
            &w.post_attention_conv,
            &mut state.conv[2],
            &mut s.attn_conv,
        )?;

        if layer.is_multiple_of(self.config.n_layers_per_attn_res_block) {
            ensure!(blocks.len < blocks.max, "more block states than planned");
            self.stream.memcpy_dtod(
                &s.normed_residual,
                &mut s
                    .blocks
                    .slice_mut(blocks.len * hidden..(blocks.len + 1) * hidden),
            )?;
            blocks.len += 1;
            self.kernels
                .misc
                .cast_bf16_to_f32(&self.stream, &s.attn_conv, &mut s.residual)?;
        } else {
            self.kernels.elementwise.add_bf16_to_f32(
                &self.stream,
                &s.hidden,
                &s.attn_conv,
                &mut s.residual,
            )?;
        }

        self.kernels.norm.unweighted_rms_norm(
            &self.stream,
            &s.residual,
            &mut s.normed,
            self.eps(),
        )?;
        self.kernels.routing.depth_attention(
            &self.stream,
            &blocks.rows(&s.blocks, hidden)?,
            &s.normed,
            &w.mlp_query,
            &mut s.mlp_input,
        )?;
        let mlp_input = match (messages, &w.message_gates) {
            (true, Some(gates)) => {
                self.kernels.linear.join_prefetch(&self.stream)?;
                self.message_gate(
                    &gates[3],
                    &mut s.gate,
                    &s.mlp_input,
                    &s.messages.slice(3 * hidden..4 * hidden),
                    &mut s.gated,
                )?;
                &s.gated
            }
            _ => &s.mlp_input,
        };
        self.kernels.norm.rms_norm_f32(
            &self.stream,
            mlp_input,
            &w.mlp_norm,
            &mut s.mlp_x,
            self.eps(),
        )?;
        self.moe(w, &mut s.moe, &s.mlp_x)?;

        self.kernels.conv.short_conv(
            &self.stream,
            &s.moe.out,
            &w.post_mlp_conv,
            &mut state.conv[3],
            &mut s.mlp_conv,
        )?;
        self.kernels
            .elementwise
            .add_to_bf16(&self.stream, &s.residual, &s.mlp_conv, &mut s.out)?;

        if let Some(t) = trace {
            t.visits.push(VisitTrace {
                input: self.stream.clone_dtoh(&s.hidden)?,
                output: self.stream.clone_dtoh(&s.out)?,
                blocks: self.stream.clone_dtoh(&blocks.rows(&s.blocks, hidden)?)?,
            });
        }
        std::mem::swap(&mut s.hidden, &mut s.out);
        Ok(())
    }

    /// The normalized query, key and value inputs of attention into `s.qkv_in`, as
    /// `_self_attention` in MLX-LM's sliding and full decoder layers. Returns whether the
    /// layer has one shared input, in `s.qkv_in[0]`.
    fn attention_inputs(
        &self,
        w: &LayerWeights,
        s: &mut Scratch,
        has_inputs: bool,
        messages: bool,
    ) -> anyhow::Result<bool> {
        let hidden = self.hidden();
        let norms: [&CudaSlice<bf16>; 3] = match &w.input_norms {
            InputNorms::Shared(norm) => [norm; 3],
            InputNorms::Separate { q, k, v } => [q, k, v],
        };
        if !has_inputs {
            // Layer 0: no depth attention and no messages, one shared BF16 input.
            ensure!(!messages, "layer without depth attention got messages");
            self.kernels.norm.rms_norm(
                &self.stream,
                &s.hidden,
                norms[0],
                &mut s.qkv_in[0],
                self.eps(),
            )?;
            return Ok(true);
        }
        let shared = matches!(w.input_norms, InputNorms::Shared(_));
        for (i, norm) in norms.into_iter().enumerate() {
            let row = if shared { 0 } else { i };
            let state = s.attention_inputs.slice(row * hidden..(row + 1) * hidden);
            let x = &mut s.qkv_in[i];
            match (messages, &w.message_gates) {
                (true, Some(gates)) => {
                    self.message_gate(
                        &gates[i],
                        &mut s.gate,
                        &state,
                        &s.messages.slice(i * hidden..(i + 1) * hidden),
                        &mut s.gated,
                    )?;
                    self.kernels
                        .norm
                        .rms_norm_f32(&self.stream, &s.gated, norm, x, self.eps())?;
                }
                _ => self
                    .kernels
                    .norm
                    .rms_norm_f32(&self.stream, &state, norm, x, self.eps())?,
            }
        }
        Ok(false)
    }

    /// Attention from the normalized query, key and value inputs to the output projection
    /// in `t.proj`, as MLX-LM's `SlidingAttention` and `FullAttention`.
    #[expect(clippy::too_many_lines, reason = "follows Attention._attend op by op")]
    fn attention(
        &self,
        state: &mut VisitState,
        t: &mut AttentionTemps,
        position: &CudaSlice<u32>,
        w: &LayerWeights,
        sliding: bool,
        [q_in, k_in, v_in]: [&CudaSlice<bf16>; 3],
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        let a = &w.attention;
        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        // Every projection of the three inputs in one launch.
        let mut products = vec![
            Gemv::new(&a.q, q_in, &mut t.q_raw),
            Gemv::new(&a.r, q_in, &mut t.r_raw),
            Gemv::new(&a.gate, q_in, &mut t.gate),
            Gemv::new(&a.sink, q_in, &mut t.sinks),
            Gemv::new(&a.k, k_in, &mut t.k_raw),
            Gemv::new(&a.v, v_in, &mut t.v_raw),
        ];
        if let KvConv::Dynamic { k_proj, v_proj, .. } = &a.kv_conv {
            products.push(Gemv::new(k_proj, k_in, &mut t.k_dynamic));
            products.push(Gemv::new(v_proj, v_in, &mut t.v_dynamic));
        }
        self.kernels.linear.gemv_bf16(&self.stream, &mut products)?;
        drop(products);
        // The output projection's weight streams into L2 while attention runs.
        self.kernels.linear.prefetch(&self.stream, &[&a.o])?;

        let [k_state, v_state, ..] = &mut state.conv;
        match &a.kv_conv {
            KvConv::Static { k, v } => {
                self.kernels
                    .conv
                    .short_conv(&self.stream, &t.k_raw, k, k_state, &mut t.k_conv)?;
                self.kernels
                    .conv
                    .short_conv(&self.stream, &t.v_raw, v, v_state, &mut t.v_conv)?;
            }
            KvConv::Dynamic {
                k_static, v_static, ..
            } => {
                self.kernels.conv.dynamic_short_conv(
                    &self.stream,
                    &t.k_raw,
                    &t.k_dynamic,
                    k_static,
                    k_state,
                    &mut t.k_conv,
                )?;
                self.kernels.conv.dynamic_short_conv(
                    &self.stream,
                    &t.v_raw,
                    &t.v_dynamic,
                    v_static,
                    v_state,
                    &mut t.v_conv,
                )?;
            }
        }

        let threshold = cfg.qk_l2_clip_threshold;
        self.kernels.norm.l2_clip(
            &self.stream,
            &t.q_raw,
            Some(&a.q_clip_scale),
            &mut t.q,
            heads,
            head_dim,
            threshold,
        )?;
        self.kernels.norm.l2_clip(
            &self.stream,
            &t.k_conv,
            None,
            &mut t.k,
            kv_heads,
            head_dim,
            threshold,
        )?;
        self.kernels.norm.unweighted_rms_norm_heads(
            &self.stream,
            &t.v_conv,
            &mut t.v,
            self.eps(),
        )?;
        self.kernels.norm.l2_clip(
            &self.stream,
            &t.r_raw,
            Some(&a.rel_clip_scale),
            &mut t.r,
            heads,
            cfg.d_rel,
            cfg.rel_l2_clip_threshold,
        )?;
        if !sliding {
            let alpha = cfg.log_scaling_alpha;
            let floor = cfg.log_scaling_n_floor as f32;
            self.kernels.elementwise.log_scale(
                &self.stream,
                &mut t.q,
                heads * head_dim,
                position,
                alpha,
                floor,
            )?;
            self.kernels.elementwise.log_scale(
                &self.stream,
                &mut t.r,
                heads * cfg.d_rel,
                position,
                alpha,
                floor,
            )?;
        }
        self.kernels
            .attention
            .relative_bias(&self.stream, &t.r, &a.rel_table, &mut t.bias)?;

        self.kernels
            .attention
            .append(&self.stream, &mut state.kv, position, &t.k, &t.v)?;
        self.kernels.attention.decode_attention(
            &self.stream,
            &t.q,
            &state.kv,
            position,
            &t.bias,
            &t.sinks,
            &mut t.partials,
            &mut t.out,
        )?;
        self.kernels
            .elementwise
            .attention_gate(&self.stream, &t.out, &t.gate, &mut t.gated)?;
        self.kernels.linear.join_prefetch(&self.stream)?;
        self.kernels
            .linear
            .gemv_bf16(&self.stream, &mut [Gemv::new(&a.o, &t.gated, &mut t.proj)])
    }

    /// MLX-LM's `MessageGate`: mixes `message` into the F32 state `h`, into `out`.
    fn message_gate(
        &self,
        g: &MessageGateWeights,
        t: &mut GateScratch,
        h: &impl DeviceIn<f32>,
        message: &impl DeviceIn<f32>,
        out: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        let hidden = self.hidden();
        self.kernels.norm.unweighted_rms_norm(
            &self.stream,
            h,
            &mut t.both.slice_mut(..hidden),
            self.eps(),
        )?;
        self.kernels.norm.unweighted_rms_norm(
            &self.stream,
            message,
            &mut t.both.slice_mut(hidden..),
            self.eps(),
        )?;
        self.kernels
            .misc
            .cast_f32_to_bf16(&self.stream, &t.both, &mut t.both_bf16)?;
        self.kernels.linear.gemv_bf16(
            &self.stream,
            &mut [Gemv::new(&g.down, &t.both_bf16, &mut t.down)],
        )?;
        self.kernels
            .misc
            .cast_bf16_to_f32(&self.stream, &t.down, &mut t.down_f32)?;
        self.kernels.linear.gemv_f32(
            &self.stream,
            &mut [
                Gemv::new(&g.forget, &t.down_f32, &mut t.forget),
                Gemv::new(&g.gain, &t.down_f32, &mut t.gain),
            ],
        )?;
        self.kernels
            .misc
            .message_gate_combine(&self.stream, h, message, &t.forget, &t.gain, out)
    }

    /// MLX-LM's `SparseMoeBlock` for one BF16 row, into `t.out`.
    fn moe(&self, w: &LayerWeights, t: &mut MoeScratch, x: &CudaSlice<bf16>) -> anyhow::Result<()> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let width = cfg.intermediate_size;

        self.kernels.routing.router_logits(
            &self.stream,
            x,
            &w.router,
            &mut t.logits,
            RouterMatmul::F32,
        )?;
        self.kernels.routing.expert_routing(
            &self.stream,
            &t.logits,
            &w.router_bias,
            &mut t.indices,
            &mut t.weights,
            cfg.route_scale,
        )?;

        let e = &w.experts;
        let (w13_format, w2_format) = match &e.format {
            ExpertFormat::Nvfp4 {
                w13_global_scale,
                w2_global_scale,
            } => (
                experts::ExpertFormat::Nvfp4 {
                    global_scale: w13_global_scale,
                },
                experts::ExpertFormat::Nvfp4 {
                    global_scale: w2_global_scale,
                },
            ),
            ExpertFormat::Mxfp8 => (experts::ExpertFormat::Mxfp8, experts::ExpertFormat::Mxfp8),
        };
        let w13 = experts::ExpertWeights {
            format: w13_format,
            weight: &e.w13,
            scales: &e.w13_scale,
            n: 2 * width,
            k: hidden,
        };
        let w2 = experts::ExpertWeights {
            format: w2_format,
            weight: &e.w2,
            scales: &e.w2_scale,
            n: hidden,
            k: width,
        };
        self.kernels
            .experts
            .expert_matvec(&self.stream, x, &t.indices, &w13, &mut t.gate_up)?;
        self.kernels.elementwise.softcapped_swiglu(
            &self.stream,
            &t.gate_up,
            &mut t.act,
            cfg.mlp_gate_softcap,
            cfg.mlp_up_softcap,
        )?;
        self.kernels
            .experts
            .expert_matvec(&self.stream, &t.act, &t.indices, &w2, &mut t.down)?;
        self.kernels
            .elementwise
            .moe_combine(&self.stream, &t.down, &t.weights, &mut t.out)
    }

    /// `unweighted_rms_norm(x.astype(mx.float32))` for a BF16 row, through `wide`.
    fn normed_f32(
        &self,
        x: &CudaSlice<bf16>,
        wide: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        self.kernels.misc.cast_bf16_to_f32(&self.stream, x, wide)?;
        self.kernels
            .norm
            .unweighted_rms_norm(&self.stream, wide, out, self.eps())
    }

    fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    fn eps(&self) -> f32 {
        self.config.rms_norm_eps
    }
}
