//! Single-stream decode, one token per step, following MLX-LM's `inkling2` model.
//!
//! This is the correctness-first runner: every op allocates its output, and rows are
//! copied out of multi-row buffers instead of read in place. The structure follows
//! `TextModel._run_loop` and `DecoderLayer.__call__` in MLX-LM so the two can be read
//! side by side.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::bf16;
use inkling::TextConfig;

use crate::{
    Gpu,
    ops::{
        attention::{Attention, AttentionScratch, KvCache},
        conv::Conv,
        elementwise::Elementwise,
        experts::{self, Experts},
        linear::Linear,
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
    /// Per-visit so the visit borrows nothing else mutably; 48 copies of about 400 KB.
    scratch: AttentionScratch,
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

/// Block states for depth attention: the normalized residual at each block boundary.
#[derive(Debug, Clone, Default)]
struct Blocks(Vec<Arc<CudaSlice<f32>>>);

pub struct Model {
    ctx: Ctx,
    passes: usize,
    visits: Vec<VisitState>,
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
            .finish_non_exhaustive()
    }
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
                    scratch: AttentionScratch::new(&stream)?,
                    conv: [
                        stream.alloc_zeros(history * kv_dims)?,
                        stream.alloc_zeros(history * kv_dims)?,
                        stream.alloc_zeros(history * hidden)?,
                        stream.alloc_zeros(history * hidden)?,
                    ],
                })
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self {
            ctx: Ctx {
                config,
                weights,
                kernels,
                stream,
            },
            passes,
            visits,
            position: 0,
        })
    }

    /// Position of the next token.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Runs one token and returns the greedy next token. Fills `trace` if given.
    #[expect(
        clippy::too_many_lines,
        reason = "one step reads top to bottom like _run_loop"
    )]
    pub fn step(&mut self, token: u32, mut trace: Option<&mut Trace>) -> anyhow::Result<u32> {
        let ctx = &self.ctx;
        let cfg = &ctx.config;
        let (core_start, core_end) = (cfg.loop_config.core_start, cfg.loop_config.core_end);
        let layers = cfg.num_hidden_layers;
        let passes = self.passes;
        if let Some(t) = trace.as_deref_mut() {
            *t = Trace::default();
        }

        let ids = ctx.stream.clone_htod(&[i32::try_from(token)?])?;
        let mut embedded = ctx.zeros::<bf16>(ctx.hidden())?;
        ctx.kernels
            .elementwise
            .embedding(&ctx.stream, &ids, &ctx.weights.embed, &mut embedded)?;
        let mut hidden = ctx.zeros::<bf16>(ctx.hidden())?;
        ctx.kernels.norm.rms_norm(
            &ctx.stream,
            &embedded,
            &ctx.weights.embed_norm,
            &mut hidden,
            ctx.eps(),
        )?;

        let mut blocks = Blocks::default();
        let mut visit = 0;
        for layer in 0..core_start {
            hidden = ctx.visit(
                &mut self.visits[visit],
                self.position,
                layer,
                &hidden,
                &mut blocks,
                None,
                trace.as_deref_mut(),
            )?;
            visit += 1;
        }
        let prelude_hidden = hidden.clone();
        let prelude_blocks = blocks.clone();
        for pass in 0..passes {
            let mut bank = None;
            if pass > 0 {
                // The message bank: block states the pass added after its first one, and
                // the normalized hidden state at the end of the pass.
                let normed = ctx.normed_f32(&hidden)?;
                let states = Blocks(blocks.0[prelude_blocks.0.len() + 1..].to_vec());
                bank = Some((ctx.contiguous(&states)?, normed));
                hidden = prelude_hidden.clone();
                blocks = prelude_blocks.clone();
            }
            for (i, layer) in (core_start..core_end).enumerate() {
                let messages = match &bank {
                    Some((states, normed)) => Some(ctx.depth_attention(
                        states,
                        normed,
                        &ctx.weights.message_queries[i],
                        MESSAGE_SLOTS,
                    )?),
                    None => None,
                };
                hidden = ctx.visit(
                    &mut self.visits[visit],
                    self.position,
                    layer,
                    &hidden,
                    &mut blocks,
                    messages.as_ref(),
                    trace.as_deref_mut(),
                )?;
                visit += 1;
            }
        }
        for layer in core_end..layers {
            hidden = ctx.visit(
                &mut self.visits[visit],
                self.position,
                layer,
                &hidden,
                &mut blocks,
                None,
                trace.as_deref_mut(),
            )?;
            visit += 1;
        }

        let normed = ctx.normed_f32(&hidden)?;
        let states = ctx.contiguous(&blocks)?;
        let final_input = ctx.depth_attention(&states, &normed, &ctx.weights.final_query, 1)?;
        let mut final_hidden = ctx.zeros::<bf16>(ctx.hidden())?;
        ctx.kernels.norm.rms_norm_f32(
            &ctx.stream,
            &final_input,
            &ctx.weights.norm,
            &mut final_hidden,
            ctx.eps(),
        )?;
        let mut logits = ctx.zeros::<bf16>(ctx.config.vocab_size)?;
        ctx.kernels.linear.gemv_bf16(
            &ctx.stream,
            &ctx.weights.lm_head,
            &final_hidden,
            &mut logits,
            ctx.config.vocab_size,
            ctx.config.hidden_size,
        )?;
        let mut capped = ctx.zeros::<f32>(ctx.config.unpadded_vocab_size)?;
        ctx.kernels.elementwise.logit_softcap(
            &ctx.stream,
            &logits,
            &mut capped,
            ctx.config.final_logit_softcapping,
        )?;
        let mut next = ctx.zeros::<i32>(1)?;
        ctx.kernels.misc.argmax_f32(
            &ctx.stream,
            &capped,
            &mut next,
            ctx.config.unpadded_vocab_size,
        )?;
        if let Some(t) = trace {
            t.logits = ctx.stream.clone_dtoh(&capped)?;
        }
        let next = ctx.stream.clone_dtoh(&next)?[0];
        self.position += 1;
        u32::try_from(next).context("argmax returned a negative index")
    }
}

impl Ctx {
    /// One decoder-layer visit, as MLX-LM's `DecoderLayer.__call__`.
    #[expect(clippy::too_many_arguments, reason = "mirrors DecoderLayer.__call__")]
    fn visit(
        &self,
        state: &mut VisitState,
        position: usize,
        layer: usize,
        hidden: &CudaSlice<bf16>,
        blocks: &mut Blocks,
        messages: Option<&CudaSlice<f32>>,
        trace: Option<&mut Trace>,
    ) -> anyhow::Result<CudaSlice<bf16>> {
        let w = &self.weights.layers[layer];
        let sliding = self.config.is_sliding(layer);
        let normed_residual = self.normed_f32(hidden)?;

        let attention_inputs = match &w.attention_query {
            Some(query) => {
                let count = if sliding { 1 } else { 3 };
                let states = self.contiguous(blocks)?;
                Some(self.depth_attention(&states, &normed_residual, query, count)?)
            }
            None => None,
        };
        let [q_in, k_in, v_in] =
            self.attention_inputs(w, hidden, attention_inputs.as_ref(), messages)?;
        let attn = self.attention(state, position, w, sliding, &q_in, &k_in, &v_in)?;

        let mut attn_conv = self.stream.alloc_zeros::<bf16>(self.config.hidden_size)?;
        self.kernels.conv.short_conv(
            &self.stream,
            &attn,
            &w.post_attention_conv,
            &mut state.conv[2],
            &mut attn_conv,
        )?;

        let residual = if layer.is_multiple_of(self.config.n_layers_per_attn_res_block) {
            blocks.0.push(Arc::new(normed_residual));
            self.to_f32(&attn_conv)?
        } else {
            let mut sum = self.zeros::<f32>(self.hidden())?;
            self.kernels
                .elementwise
                .add_bf16_to_f32(&self.stream, hidden, &attn_conv, &mut sum)?;
            sum
        };

        let normed = self.unweighted_norm(&residual)?;
        let states = self.contiguous(blocks)?;
        let mut mlp_input = self.depth_attention(&states, &normed, &w.mlp_query, 1)?;
        if let (Some(messages), Some(gates)) = (messages, &w.message_gates) {
            let message = self.row(messages, 3)?;
            mlp_input = self.message_gate(&gates[3], &mlp_input, &message)?;
        }
        let mut mlp_x = self.zeros::<bf16>(self.hidden())?;
        self.kernels.norm.rms_norm_f32(
            &self.stream,
            &mlp_input,
            &w.mlp_norm,
            &mut mlp_x,
            self.eps(),
        )?;
        let moe = self.moe(w, &mlp_x)?;

        let mut mlp_conv = self.stream.alloc_zeros::<bf16>(self.config.hidden_size)?;
        self.kernels.conv.short_conv(
            &self.stream,
            &moe,
            &w.post_mlp_conv,
            &mut state.conv[3],
            &mut mlp_conv,
        )?;
        let mut out = self.zeros::<bf16>(self.hidden())?;
        self.kernels
            .elementwise
            .add_to_bf16(&self.stream, &residual, &mlp_conv, &mut out)?;

        if let Some(t) = trace {
            let states = self.contiguous(blocks)?;
            t.visits.push(VisitTrace {
                input: self.stream.clone_dtoh(hidden)?,
                output: self.stream.clone_dtoh(&out)?,
                blocks: self.stream.clone_dtoh(&states)?,
            });
        }
        Ok(out)
    }

    /// The normalized query, key and value inputs of attention, as `_self_attention` in
    /// MLX-LM's sliding and full decoder layers.
    fn attention_inputs(
        &self,
        w: &LayerWeights,
        hidden: &CudaSlice<bf16>,
        attention_inputs: Option<&CudaSlice<f32>>,
        messages: Option<&CudaSlice<f32>>,
    ) -> anyhow::Result<[CudaSlice<bf16>; 3]> {
        let norms: [&CudaSlice<bf16>; 3] = match &w.input_norms {
            InputNorms::Shared(norm) => [norm; 3],
            InputNorms::Separate { q, k, v } => [q, k, v],
        };
        let Some(inputs) = attention_inputs else {
            // Layer 0: no depth attention and no messages, one shared BF16 input.
            ensure!(
                messages.is_none(),
                "layer without depth attention got messages"
            );
            let mut x = self.zeros::<bf16>(self.hidden())?;
            self.kernels
                .norm
                .rms_norm(&self.stream, hidden, norms[0], &mut x, self.eps())?;
            return Ok([x.clone(), x.clone(), x]);
        };
        let shared = matches!(w.input_norms, InputNorms::Shared(_));
        let mut out = Vec::with_capacity(3);
        for (i, norm) in norms.into_iter().enumerate() {
            let mut state = self.row(inputs, if shared { 0 } else { i })?;
            if let (Some(messages), Some(gates)) = (messages, &w.message_gates) {
                let message = self.row(messages, i)?;
                state = self.message_gate(&gates[i], &state, &message)?;
            }
            let mut x = self.zeros::<bf16>(self.hidden())?;
            self.kernels
                .norm
                .rms_norm_f32(&self.stream, &state, norm, &mut x, self.eps())?;
            out.push(x);
        }
        Ok(out.try_into().expect("three inputs"))
    }

    /// Attention from normalized inputs to the output projection, as MLX-LM's
    /// `SlidingAttention` and `FullAttention`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the three attention inputs stay separate"
    )]
    #[expect(clippy::too_many_lines, reason = "follows Attention._attend op by op")]
    #[expect(
        clippy::many_single_char_names,
        reason = "q, k, v and r are MLX-LM's names"
    )]
    fn attention(
        &self,
        state: &mut VisitState,
        position: usize,
        w: &LayerWeights,
        sliding: bool,
        q_in: &CudaSlice<bf16>,
        k_in: &CudaSlice<bf16>,
        v_in: &CudaSlice<bf16>,
    ) -> anyhow::Result<CudaSlice<bf16>> {
        let cfg = &self.config;
        let a = &w.attention;
        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let kv_dims = kv_heads * head_dim;

        let k_raw = self.gemv(&a.k, k_in, kv_dims)?;
        let v_raw = self.gemv(&a.v, v_in, kv_dims)?;
        let mut k_conv = self.zeros::<bf16>(kv_dims)?;
        let mut v_conv = self.zeros::<bf16>(kv_dims)?;
        let conv_k = cfg.sconv_kernel_size;
        match &a.kv_conv {
            KvConv::Static { k, v } => {
                let [k_state, v_state, ..] = &mut state.conv;
                self.kernels
                    .conv
                    .short_conv(&self.stream, &k_raw, k, k_state, &mut k_conv)?;
                self.kernels
                    .conv
                    .short_conv(&self.stream, &v_raw, v, v_state, &mut v_conv)?;
            }
            KvConv::Dynamic {
                k_static,
                k_proj,
                v_static,
                v_proj,
            } => {
                let k_dynamic = gemv(
                    &self.kernels.linear,
                    &self.stream,
                    k_proj,
                    k_in,
                    kv_heads * conv_k,
                )?;
                let v_dynamic = gemv(
                    &self.kernels.linear,
                    &self.stream,
                    v_proj,
                    v_in,
                    kv_heads * conv_k,
                )?;
                let [k_state, v_state, ..] = &mut state.conv;
                self.kernels.conv.dynamic_short_conv(
                    &self.stream,
                    &k_raw,
                    &k_dynamic,
                    k_static,
                    k_state,
                    &mut k_conv,
                )?;
                self.kernels.conv.dynamic_short_conv(
                    &self.stream,
                    &v_raw,
                    &v_dynamic,
                    v_static,
                    v_state,
                    &mut v_conv,
                )?;
            }
        }

        let q_raw = self.gemv(&a.q, q_in, heads * head_dim)?;
        let r_raw = self.gemv(&a.r, q_in, heads * cfg.d_rel)?;
        let gate = self.gemv(&a.gate, q_in, heads * head_dim)?;
        let sinks = self.gemv(&a.sink, q_in, heads)?;

        let threshold = cfg.qk_l2_clip_threshold;
        let mut q = self.zeros::<bf16>(heads * head_dim)?;
        self.kernels.norm.l2_clip(
            &self.stream,
            &q_raw,
            Some(&a.q_clip_scale),
            &mut q,
            heads,
            head_dim,
            threshold,
        )?;
        let mut k = self.zeros::<bf16>(kv_dims)?;
        self.kernels.norm.l2_clip(
            &self.stream,
            &k_conv,
            None,
            &mut k,
            kv_heads,
            head_dim,
            threshold,
        )?;
        let mut v = self.zeros::<bf16>(kv_dims)?;
        self.kernels
            .norm
            .unweighted_rms_norm_heads(&self.stream, &v_conv, &mut v, self.eps())?;
        let mut r = self.zeros::<bf16>(heads * cfg.d_rel)?;
        self.kernels.norm.l2_clip(
            &self.stream,
            &r_raw,
            Some(&a.rel_clip_scale),
            &mut r,
            heads,
            cfg.d_rel,
            cfg.rel_l2_clip_threshold,
        )?;
        let position_u32 = u32::try_from(position)?;
        if !sliding {
            let alpha = cfg.log_scaling_alpha;
            let floor = cfg.log_scaling_n_floor as f32;
            self.kernels.elementwise.log_scale(
                &self.stream,
                &mut q,
                heads * head_dim,
                position_u32,
                alpha,
                floor,
            )?;
            self.kernels.elementwise.log_scale(
                &self.stream,
                &mut r,
                heads * cfg.d_rel,
                position_u32,
                alpha,
                floor,
            )?;
        }
        let mut bias = self.zeros::<bf16>(heads * cfg.rel_extent)?;
        self.kernels
            .attention
            .relative_bias(&self.stream, &r, &a.rel_table, &mut bias)?;

        let mut out = self.zeros::<bf16>(heads * head_dim)?;
        let appended = self
            .kernels
            .attention
            .append(&self.stream, &mut state.kv, &k, &v)?;
        ensure!(
            appended == position,
            "visit cache is at position {appended}, the step at {position}"
        );
        self.kernels.attention.decode_attention(
            &self.stream,
            &q,
            &state.kv,
            &bias,
            &sinks,
            &mut state.scratch,
            &mut out,
        )?;
        let mut gated = self.zeros::<bf16>(heads * head_dim)?;
        self.kernels
            .elementwise
            .attention_gate(&self.stream, &out, &gate, &mut gated)?;
        self.gemv(&a.o, &gated, cfg.hidden_size)
    }

    /// MLX-LM's `MessageGate`: mixes `message` into the F32 state `h`.
    fn message_gate(
        &self,
        g: &MessageGateWeights,
        h: &CudaSlice<f32>,
        message: &CudaSlice<f32>,
    ) -> anyhow::Result<CudaSlice<f32>> {
        let hidden = self.hidden();
        let mut both = self.zeros::<f32>(2 * hidden)?;
        let normed_h = self.unweighted_norm(h)?;
        let normed_m = self.unweighted_norm(message)?;
        self.stream
            .memcpy_dtod(&normed_h, &mut both.slice_mut(..hidden))?;
        self.stream
            .memcpy_dtod(&normed_m, &mut both.slice_mut(hidden..))?;
        let mut both_bf16 = self.zeros::<bf16>(2 * hidden)?;
        self.kernels
            .misc
            .cast_f32_to_bf16(&self.stream, &both, &mut both_bf16)?;
        let rank = self.config.loop_config.gate_rank;
        let down = self.gemv(&g.down, &both_bf16, rank)?;
        let down = self.to_f32(&down)?;
        let mut forget = self.zeros::<f32>(hidden)?;
        let mut gain = self.zeros::<f32>(hidden)?;
        self.kernels
            .linear
            .gemv_f32(&self.stream, &g.forget, &down, &mut forget, hidden, rank)?;
        self.kernels
            .linear
            .gemv_f32(&self.stream, &g.gain, &down, &mut gain, hidden, rank)?;
        let mut out = self.zeros::<f32>(hidden)?;
        self.kernels.misc.message_gate_combine(
            &self.stream,
            h,
            message,
            &forget,
            &gain,
            &mut out,
        )?;
        Ok(out)
    }

    /// MLX-LM's `SparseMoeBlock` for one BF16 row.
    fn moe(&self, w: &LayerWeights, x: &CudaSlice<bf16>) -> anyhow::Result<CudaSlice<bf16>> {
        let cfg = &self.config;
        let experts = cfg.n_routed_experts;
        let top_k = cfg.num_experts_per_tok;
        let hidden = cfg.hidden_size;
        let width = cfg.intermediate_size;

        let mut logits = self.zeros::<f32>(experts)?;
        self.kernels.routing.router_logits(
            &self.stream,
            x,
            &w.router,
            &mut logits,
            RouterMatmul::F32,
        )?;
        let mut indices = self.zeros::<i32>(top_k)?;
        let mut weights = self.zeros::<f32>(top_k)?;
        self.kernels.routing.expert_routing(
            &self.stream,
            &logits,
            &w.router_bias,
            &mut indices,
            &mut weights,
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
        let mut gate_up = self.zeros::<bf16>(top_k * 2 * width)?;
        self.kernels
            .experts
            .expert_matvec(&self.stream, x, &indices, &w13, &mut gate_up)?;
        let mut act = self.zeros::<bf16>(top_k * width)?;
        self.swiglu(&gate_up, &mut act)?;
        let mut down = self.zeros::<bf16>(top_k * hidden)?;
        self.kernels
            .experts
            .expert_matvec(&self.stream, &act, &indices, &w2, &mut down)?;
        let mut out = self.zeros::<bf16>(hidden)?;
        self.kernels
            .elementwise
            .moe_combine(&self.stream, &down, &weights, &mut out)?;
        Ok(out)
    }

    fn swiglu(&self, gate_up: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>) -> anyhow::Result<()> {
        self.kernels.elementwise.softcapped_swiglu(
            &self.stream,
            gate_up,
            out,
            self.config.mlp_gate_softcap,
            self.config.mlp_up_softcap,
        )
    }

    /// Depth attention of `count` queries over `states` (`[blocks, hidden]`) and `partial`,
    /// giving F32 `[count, hidden]`.
    fn depth_attention(
        &self,
        states: &CudaSlice<f32>,
        partial: &CudaSlice<f32>,
        queries: &CudaSlice<f32>,
        count: usize,
    ) -> anyhow::Result<CudaSlice<f32>> {
        let mut out = self.zeros::<f32>(count * self.hidden())?;
        self.kernels
            .routing
            .depth_attention(&self.stream, states, partial, queries, &mut out)?;
        Ok(out)
    }

    /// Block states as one contiguous `[blocks, hidden]` buffer.
    fn contiguous(&self, blocks: &Blocks) -> anyhow::Result<CudaSlice<f32>> {
        let hidden = self.hidden();
        ensure!(!blocks.0.is_empty(), "depth attention needs a block state");
        let mut out = self.zeros::<f32>(blocks.0.len() * hidden)?;
        for (i, block) in blocks.0.iter().enumerate() {
            self.stream.memcpy_dtod(
                block.as_ref(),
                &mut out.slice_mut(i * hidden..(i + 1) * hidden),
            )?;
        }
        Ok(out)
    }

    /// Row `i` of a `[rows, hidden]` F32 buffer.
    fn row(&self, rows: &CudaSlice<f32>, i: usize) -> anyhow::Result<CudaSlice<f32>> {
        let hidden = self.hidden();
        let mut out = self.zeros::<f32>(hidden)?;
        self.stream
            .memcpy_dtod(&rows.slice(i * hidden..(i + 1) * hidden), &mut out)?;
        Ok(out)
    }

    /// `unweighted_rms_norm(x.astype(mx.float32))` for a BF16 row.
    fn normed_f32(&self, x: &CudaSlice<bf16>) -> anyhow::Result<CudaSlice<f32>> {
        let wide = self.to_f32(x)?;
        self.unweighted_norm(&wide)
    }

    fn unweighted_norm(&self, x: &CudaSlice<f32>) -> anyhow::Result<CudaSlice<f32>> {
        let mut out = self.zeros::<f32>(x.len())?;
        self.kernels
            .norm
            .unweighted_rms_norm(&self.stream, x, &mut out, self.eps())?;
        Ok(out)
    }

    fn to_f32(&self, x: &CudaSlice<bf16>) -> anyhow::Result<CudaSlice<f32>> {
        let mut out = self.zeros::<f32>(x.len())?;
        self.kernels
            .misc
            .cast_bf16_to_f32(&self.stream, x, &mut out)?;
        Ok(out)
    }

    fn gemv(
        &self,
        weight: &CudaSlice<bf16>,
        x: &CudaSlice<bf16>,
        n: usize,
    ) -> anyhow::Result<CudaSlice<bf16>> {
        gemv(&self.kernels.linear, &self.stream, weight, x, n)
    }

    fn zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> anyhow::Result<CudaSlice<T>> {
        Ok(self.stream.alloc_zeros(len)?)
    }

    fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    fn eps(&self) -> f32 {
        self.config.rms_norm_eps
    }
}

/// A free function so callers can hold a mutable borrow of the visit caches meanwhile.
fn gemv(
    linear: &Linear,
    stream: &Arc<CudaStream>,
    weight: &CudaSlice<bf16>,
    x: &CudaSlice<bf16>,
    n: usize,
) -> anyhow::Result<CudaSlice<bf16>> {
    let mut out = stream.alloc_zeros(n)?;
    linear.gemv_bf16(stream, weight, x, &mut out, n, x.len())?;
    Ok(out)
}
