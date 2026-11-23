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
//!
//! [`Model::prefill`] runs a prompt in chunks of many rows instead; see [`prefill`].

mod prefill;

pub use prefill::{DEFAULT_PREFILL_CHUNK, MAX_PREFILL_CHUNK, PrefillLogits};

use std::{ops::Range, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaGraph, CudaSlice, CudaStream, CudaView, CudaViewMut, DeviceRepr, ValidAsZeroBits,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
};
use half::bf16;
use inkling::TextConfig;

use crate::{
    DeviceIn, Gpu,
    ops::{
        attention::{self, Attention, AttentionScratch, KvCache, KvData, KvFormat},
        conv::Conv,
        elementwise::Elementwise,
        experts::{self, Experts},
        linear::{Gemm, Gemv, Linear},
        misc::{ArgmaxScratch, GATE_SLOTS, GateStep, GateStepScratch, Misc},
        norm::{AttentionPost, KvConvWeights, Norm},
        routing::{DepthStep, NormDest, Routing, StepPartial},
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

/// One visit's caches from [`Model::cache_snapshot`].
#[derive(Debug)]
pub struct VisitSnapshot {
    /// Every cache slot, `[slots, 8, 128]`.
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    /// The key, value, post-attention and post-MLP conv states, `[3, dims]` each.
    pub conv: Vec<Vec<f32>>,
}

fn widen(x: &[bf16]) -> Vec<f32> {
    x.iter().map(|v| v.to_f32()).collect()
}

/// The value of an E4M3 byte: 1 sign, 4 exponent (bias 7) and 3 mantissa bits.
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((b >> 3) & 0xf);
    let mantissa = f32::from(b & 7);
    if exponent == 0xf && b & 7 == 7 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    sign * magnitude
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
    /// Every looped layer's message queries, F32 `[core layers, 4, hidden]`, copied from
    /// the weights so one launch can run them all.
    message_queries: CudaSlice<f32>,
    /// Depth attention over the bank for every looped layer, F32 `[core layers, 4, hidden]`.
    messages: CudaSlice<f32>,
    /// The normalized visit input, when it does not start a block state.
    normed_residual: CudaSlice<f32>,
    /// Depth attention for the attention inputs, F32 `[1 or 3, hidden]`.
    attention_inputs: CudaSlice<f32>,
    /// The message gates' outputs, F32 `[3, hidden]`: the attention gates', then the MLP
    /// gate's in the first row.
    gated: CudaSlice<f32>,
    gate: GateScratch,
    /// Normalized query, key and value inputs of attention. Layer 0 uses only the first.
    qkv_in: [CudaSlice<bf16>; 3],
    attention: AttentionTemps,
    residual: CudaSlice<f32>,
    mlp_input: CudaSlice<f32>,
    mlp_x: CudaSlice<bf16>,
    moe: MoeScratch,
    final_hidden: CudaSlice<bf16>,
    logits: CudaSlice<bf16>,
    /// F32 logits after the softcap, `[unpadded_vocab]`.
    capped: CudaSlice<f32>,
    argmax: ArgmaxScratch,
}

/// Intermediates of `MessageGate`.
struct GateScratch {
    /// For each message slot, every looped layer's normalized state and message side by
    /// side, BF16 `[core layers, 2 * hidden]`: the down projections' inputs. Depth
    /// attention writes both halves, the messages for all layers at the start of a pass.
    inputs: [CudaSlice<bf16>; MESSAGE_SLOTS],
    /// The gate kernel's cross-block sums and barrier counters.
    step: GateStepScratch,
}

/// Intermediates of `SlidingAttention` and `FullAttention`.
struct AttentionTemps {
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
    partials: AttentionScratch,
    out: CudaSlice<bf16>,
    gated: CudaSlice<bf16>,
    /// The output projection, BF16 `[hidden]`.
    proj: CudaSlice<bf16>,
}

/// Intermediates of `SparseMoeBlock`.
struct MoeScratch {
    logits: CudaSlice<f32>,
    /// The router kernel's block counter.
    counter: CudaSlice<u32>,
    indices: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    act: CudaSlice<bf16>,
    down: CudaSlice<bf16>,
    /// The expert MLP kernel's counters.
    mlp_sync: CudaSlice<u32>,
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
        let top_k = cfg.num_experts_per_tok;
        let width = cfg.intermediate_size;
        let looped = cfg.loop_config.core_end - cfg.loop_config.core_start;
        Ok(Self {
            token: zeros(stream, 1)?,
            position: zeros(stream, 1)?,
            next: zeros(stream, 1)?,
            embedded: zeros(stream, hidden)?,
            hidden: zeros(stream, hidden)?,
            out: zeros(stream, hidden)?,
            prelude_hidden: zeros(stream, hidden)?,
            blocks: zeros(stream, max_blocks * hidden)?,
            message_queries: zeros(stream, looped * MESSAGE_SLOTS * hidden)?,
            messages: zeros(stream, looped * MESSAGE_SLOTS * hidden)?,
            normed_residual: zeros(stream, hidden)?,
            attention_inputs: zeros(stream, 3 * hidden)?,
            gated: zeros(stream, GATE_SLOTS * hidden)?,
            gate: GateScratch {
                inputs: [
                    zeros(stream, looped * 2 * hidden)?,
                    zeros(stream, looped * 2 * hidden)?,
                    zeros(stream, looped * 2 * hidden)?,
                    zeros(stream, looped * 2 * hidden)?,
                ],
                step: GateStepScratch::new(stream)?,
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
            residual: zeros(stream, hidden)?,
            mlp_input: zeros(stream, hidden)?,
            mlp_x: zeros(stream, hidden)?,
            moe: MoeScratch {
                logits: zeros(stream, cfg.n_routed_experts)?,
                counter: zeros(stream, 1)?,
                indices: zeros(stream, top_k)?,
                weights: zeros(stream, top_k)?,
                act: zeros(stream, top_k * width)?,
                down: zeros(stream, top_k * hidden)?,
                mlp_sync: zeros(stream, 1 + top_k)?,
            },
            final_hidden: zeros(stream, hidden)?,
            logits: zeros(stream, cfg.vocab_size)?,
            capped: zeros(stream, cfg.unpadded_vocab_size)?,
            argmax: ArgmaxScratch::new(stream)?,
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
    /// Buffers of [`Model::prefill`], allocated on its first call.
    prefill: Option<prefill::PrefillScratch>,
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
    /// Builds a runner with the looped core run `passes` times and BF16 caches for
    /// `capacity` tokens in the global layers.
    pub fn new(
        gpu: &Gpu,
        config: TextConfig,
        weights: Weights,
        passes: usize,
        capacity: usize,
    ) -> anyhow::Result<Self> {
        Self::with_kv_format(gpu, config, weights, passes, capacity, KvFormat::Bf16)
    }

    /// [`Model::new`] with the global-layer KV caches in `kv_format`.
    pub fn with_kv_format(
        gpu: &Gpu,
        config: TextConfig,
        weights: Weights,
        passes: usize,
        capacity: usize,
        kv_format: KvFormat,
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
                    KvCache::global_with_format(&stream, capacity, kv_format)?
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
        let mut scratch = Scratch::new(&stream, &config, max_blocks)?;
        ensure!(
            weights.message_queries.len() == loops.core_end - loops.core_start,
            "one set of message queries per looped layer"
        );
        for (i, queries) in weights.message_queries.iter().enumerate() {
            let len = MESSAGE_SLOTS * hidden;
            ensure!(queries.len() == len, "message queries of layer {i}");
            stream.memcpy_dtod(
                queries,
                &mut scratch.message_queries.slice_mut(i * len..(i + 1) * len),
            )?;
        }
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
            prefill: None,
        })
    }

    /// Starts a new sequence: zeroes every KV cache and conv state and moves to position 0.
    pub fn reset(&mut self) -> anyhow::Result<()> {
        for visit in &mut self.visits {
            visit.kv.clear(&self.ctx.stream)?;
            for state in &mut visit.conv {
                self.ctx.stream.memset_zeros(state)?;
            }
        }
        self.position = 0;
        Ok(())
    }

    /// Every visit's KV cache and conv states, widened to F32, for tests that compare two
    /// ways of filling them.
    pub fn cache_snapshot(&self) -> anyhow::Result<Vec<VisitSnapshot>> {
        let stream = &self.ctx.stream;
        self.visits
            .iter()
            .map(|v| {
                let (keys, values) = match v.kv.data() {
                    KvData::Bf16 { keys, values } => (
                        widen(&stream.clone_dtoh(keys)?),
                        widen(&stream.clone_dtoh(values)?),
                    ),
                    KvData::Fp8 { keys, values } => (
                        stream
                            .clone_dtoh(keys)?
                            .into_iter()
                            .map(e4m3_to_f32)
                            .collect(),
                        stream
                            .clone_dtoh(values)?
                            .into_iter()
                            .map(e4m3_to_f32)
                            .collect(),
                    ),
                };
                Ok(VisitSnapshot {
                    keys,
                    values,
                    conv: v
                        .conv
                        .iter()
                        .map(|c| Ok(stream.clone_dtoh(c)?))
                        .collect::<anyhow::Result<_>>()?,
                })
            })
            .collect()
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
        // Each visit's layer and whether it takes messages: the next visit's weights stream
        // into L2 during the MoE half of the current one.
        let looped = core_end - core_start;
        let plan: Vec<(usize, bool)> = cfg
            .visit_layers(self.passes)
            .into_iter()
            .enumerate()
            .map(|(v, layer)| {
                (
                    layer,
                    v >= core_start + looped && v < core_start + self.passes * looped,
                )
            })
            .collect();
        let mut upcoming = plan.iter().skip(1).map(|&p| Some(p)).chain([None]);
        let mut next_plan = || upcoming.next().flatten();

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
        ctx.kernels
            .linear
            .prefetch(&ctx.stream, &ctx.visit_prefetch(plan[0].0, plan[0].1))?;
        for layer in 0..core_start {
            ctx.visit(
                s,
                next_visit()?,
                VisitPlan {
                    layer,
                    looped: None,
                    next: next_plan(),
                },
                &mut blocks,
                trace.as_deref_mut(),
            )?;
        }
        ctx.stream.memcpy_dtod(&s.hidden, &mut s.prelude_hidden)?;
        let prelude_blocks = blocks.len;
        for pass in 0..self.passes {
            if pass > 0 {
                // Every looped layer's messages, and their normalized halves of the gate
                // inputs, in one launch. They attend over the message bank: the block
                // states the previous pass added after its first one, and the normalized
                // hidden state at its end.
                let bank_rows = blocks.len.saturating_sub(prelude_blocks + 1);
                ensure!(bank_rows > 0, "the message bank needs a block state");
                let dests = s.gate.inputs.each_mut().map(|x| {
                    Some(NormDest {
                        out: x.slice_mut(hidden..),
                        weight: None,
                    })
                });
                ctx.kernels.routing.depth_attention_step(
                    &ctx.stream,
                    DepthStep {
                        blocks: s
                            .blocks
                            .slice((prelude_blocks + 1) * hidden..blocks.len * hidden),
                        partial: StepPartial::Bf16(s.hidden.as_view()),
                        normed_partial: None,
                        queries: s.message_queries.as_view(),
                        batch: core_end - core_start,
                        out: Some(s.messages.as_view_mut()),
                        dests,
                        dst_stride: 2 * hidden,
                    },
                    ctx.eps(),
                )?;
                ctx.stream.memcpy_dtod(&s.prelude_hidden, &mut s.hidden)?;
                blocks.len = prelude_blocks;
            }
            for (i, layer) in (core_start..core_end).enumerate() {
                ctx.visit(
                    s,
                    next_visit()?,
                    VisitPlan {
                        layer,
                        looped: (pass > 0).then_some(i),
                        next: next_plan(),
                    },
                    &mut blocks,
                    trace.as_deref_mut(),
                )?;
            }
        }
        for layer in core_end..cfg.num_hidden_layers {
            ctx.visit(
                s,
                next_visit()?,
                VisitPlan {
                    layer,
                    looped: None,
                    next: next_plan(),
                },
                &mut blocks,
                trace.as_deref_mut(),
            )?;
        }

        ctx.kernels.linear.join_prefetch(&ctx.stream)?;
        ctx.kernels.routing.depth_attention_step(
            &ctx.stream,
            DepthStep {
                blocks: blocks.rows(&s.blocks, hidden)?,
                partial: StepPartial::Bf16(s.hidden.as_view()),
                normed_partial: None,
                queries: ctx.weights.final_query.as_view(),
                batch: 1,
                out: None,
                dests: [
                    Some(NormDest {
                        out: s.final_hidden.as_view_mut(),
                        weight: Some(&ctx.weights.norm),
                    }),
                    None,
                    None,
                    None,
                ],
                dst_stride: 0,
            },
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
        ctx.kernels
            .misc
            .argmax_wide_f32(&ctx.stream, &s.capped, &mut s.next, &mut s.argmax)?;
        if let Some(t) = trace {
            t.logits = ctx.stream.clone_dtoh(&s.capped)?;
        }
        Ok(())
    }
}

/// Which layer a visit runs, and whose weights it prefetches next.
#[derive(Clone, Copy)]
struct VisitPlan {
    layer: usize,
    /// The layer's index among the looped layers, on passes after the first, which take
    /// messages.
    looped: Option<usize>,
    /// The next visit's layer and whether it takes messages.
    next: Option<(usize, bool)>,
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

/// One product of [`Ctx::gemm_rows`]: `out [m, n] = x [m, k] @ weight [n, k]^T`.
struct RowProduct<'a> {
    weight: &'a CudaSlice<bf16>,
    x: CudaView<'a, bf16>,
    out: CudaViewMut<'a, bf16>,
}

/// What [`Ctx::attention_inputs`] reads and writes.
struct AttentionInputs<'a> {
    /// The visit input.
    x: &'a CudaSlice<bf16>,
    /// The block states so far, F32 `[blocks, hidden]`.
    states: CudaView<'a, f32>,
    /// Receives the normalized visit input.
    normed: CudaViewMut<'a, f32>,
    /// Depth attention's output rows, F32 `[1 or 3, hidden]`, written only for the gates.
    inputs: &'a mut CudaSlice<f32>,
    qkv_in: &'a mut [CudaSlice<bf16>; 3],
    gate: &'a mut GateScratch,
    gated: &'a mut CudaSlice<f32>,
    /// On looped passes after the first: the layer's message gates, its index among the
    /// looped layers, and every looped layer's messages.
    messages: Option<(
        &'a [MessageGateWeights; MESSAGE_SLOTS],
        usize,
        &'a CudaSlice<f32>,
    )>,
}

impl Ctx {
    /// The weights a visit of `layer` reads first: the attention projections and, with
    /// `messages`, the message gates that build their inputs.
    fn visit_prefetch(&self, layer: usize, messages: bool) -> Vec<&CudaSlice<bf16>> {
        let w = &self.weights.layers[layer];
        let mut weights = Vec::new();
        if let Some(gates) = w.message_gates.as_ref().filter(|_| messages) {
            weights.extend(gates[..3].iter().flat_map(MessageGateWeights::matrices));
        }
        weights.extend(w.attention.input_projections());
        weights
    }

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
        plan: VisitPlan,
        blocks: &mut Blocks,
        trace: Option<&mut Trace>,
    ) -> anyhow::Result<()> {
        let VisitPlan {
            layer,
            looped,
            next,
        } = plan;
        let w = &self.weights.layers[layer];
        let sliding = self.config.is_sliding(layer);
        let hidden = self.hidden();
        let block_start = layer.is_multiple_of(self.config.n_layers_per_attn_res_block);
        ensure!(
            !block_start || blocks.len < blocks.max,
            "more block states than planned"
        );
        let gates = w.message_gates.as_ref().zip(looped);
        // The previous visit (or the step, for the first) started prefetching this visit's
        // projection weights. The output projection's and the MLP gate's weights follow,
        // then the next visit's projection weights, which stream during the MoE half of
        // this visit while its experts leave DRAM bandwidth to spare. The first visit does
        // not wait for its prefetch: its projections read from DRAM alongside it.
        if layer > 0 {
            self.kernels.linear.join_prefetch(&self.stream)?;
        }
        let mut prefetch = vec![&w.attention.o];
        if let Some((gates, _)) = gates {
            prefetch.extend(gates[3].matrices());
        }
        if let Some((layer, messages)) = next {
            prefetch.extend(self.visit_prefetch(layer, messages));
        }
        self.kernels.linear.prefetch(&self.stream, &prefetch)?;

        // The normalized visit input goes straight into the new block state at a block start.
        let (states, mut rest) = s.blocks.split_at_mut(blocks.len * hidden);
        let normed = if block_start {
            rest.slice_mut(..hidden)
        } else {
            s.normed_residual.as_view_mut()
        };
        let one_input = self.attention_inputs(
            w,
            AttentionInputs {
                x: &s.hidden,
                states: states.slice(..),
                normed,
                inputs: &mut s.attention_inputs,
                qkv_in: &mut s.qkv_in,
                gate: &mut s.gate,
                gated: &mut s.gated,
                messages: gates.map(|(g, i)| (g, i, &s.messages)),
            },
            sliding,
        )?;
        if block_start {
            blocks.len += 1;
        }
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
        // The post-attention conv and residual add, then depth attention for the MLP input.
        // Without a gate, the kernel also applies the MoE input norm. With one, it writes
        // the MLP state's normalized half of the gate input, and the gate applies the norm.
        let gate_row = gates.map(|(_, i)| i * 2 * hidden..i * 2 * hidden + hidden);
        let mlp_dest = match &gate_row {
            Some(row) => NormDest {
                out: s.gate.inputs[3].slice_mut(row.clone()),
                weight: None,
            },
            None => NormDest {
                out: s.mlp_x.as_view_mut(),
                weight: Some(&w.mlp_norm),
            },
        };
        let [.., conv_state, _] = &mut state.conv;
        self.kernels.routing.depth_attention_step(
            &self.stream,
            DepthStep {
                blocks: blocks.rows(&s.blocks, hidden)?,
                partial: StepPartial::ConvResidual {
                    x: s.attention.proj.as_view(),
                    weight: &w.post_attention_conv,
                    state: conv_state.as_view_mut(),
                    hidden: (!block_start).then(|| s.hidden.as_view()),
                    residual: s.residual.as_view_mut(),
                },
                normed_partial: None,
                queries: w.mlp_query.as_view(),
                batch: 1,
                out: gates.is_some().then(|| s.mlp_input.as_view_mut()),
                dests: [None, None, None, Some(mlp_dest)],
                dst_stride: 0,
            },
            self.eps(),
        )?;
        if let Some((gates, i)) = gates {
            let message = (i * MESSAGE_SLOTS + 3) * hidden;
            self.message_gates(
                &gates[3..],
                &mut s.gate,
                3,
                i,
                (&s.mlp_input, 0),
                &s.messages.slice(message..message + hidden),
                &mut s.gated,
                vec![Some((&w.mlp_norm, s.mlp_x.as_view_mut()))],
            )?;
        }
        self.moe(w, &mut s.moe, &s.mlp_x)?;

        self.kernels.conv.moe_conv_residual(
            &self.stream,
            &s.moe.down,
            &s.moe.weights,
            &w.post_mlp_conv,
            &mut state.conv[3],
            &s.residual,
            &mut s.out,
        )?;

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

    /// The normalized query, key and value inputs of attention into `a.qkv_in`, as
    /// `_self_attention` in MLX-LM's sliding and full decoder layers, and the normalized
    /// visit input into `a.normed`. Returns whether the layer has one shared input, in
    /// `a.qkv_in[0]`.
    fn attention_inputs(
        &self,
        w: &LayerWeights,
        a: AttentionInputs<'_>,
        sliding: bool,
    ) -> anyhow::Result<bool> {
        let hidden = self.hidden();
        let norms: [&CudaSlice<bf16>; 3] = match &w.input_norms {
            InputNorms::Shared(norm) => [norm; 3],
            InputNorms::Separate { q, k, v } => [q, k, v],
        };
        let shared = matches!(w.input_norms, InputNorms::Shared(_));
        let Some(query) = &w.attention_query else {
            // Layer 0: no depth attention and no messages, one shared BF16 input.
            ensure!(
                a.messages.is_none(),
                "layer without depth attention got messages"
            );
            let mut normed = a.normed;
            self.kernels.norm.unweighted_rms_norm_bf16(
                &self.stream,
                a.x,
                &mut normed,
                self.eps(),
            )?;
            self.kernels.norm.rms_norm(
                &self.stream,
                a.x,
                norms[0],
                &mut a.qkv_in[0],
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
        // Without messages, the depth-attention kernel applies the input norms. With them, it
        // writes the normalized states into the gate inputs, and each gate applies its norm.
        let dests = if let Some((_, looped, _)) = a.messages {
            let row = looped * 2 * hidden;
            [g0, g1, g2].map(|g| {
                Some(NormDest {
                    out: g.slice_mut(row..row + hidden),
                    weight: None,
                })
            })
        } else if shared {
            [
                Some(NormDest {
                    out: q0.as_view_mut(),
                    weight: Some(norms[0]),
                }),
                None,
                None,
            ]
        } else {
            [(q0, norms[0]), (q1, norms[1]), (q2, norms[2])].map(|(q, norm)| {
                Some(NormDest {
                    out: q.as_view_mut(),
                    weight: Some(norm),
                })
            })
        };
        let [d0, d1, d2] = dests;
        self.kernels.routing.depth_attention_step(
            &self.stream,
            DepthStep {
                blocks: a.states,
                partial: StepPartial::Bf16(a.x.as_view()),
                normed_partial: Some(a.normed),
                queries: query.as_view(),
                batch: 1,
                out: a
                    .messages
                    .is_some()
                    .then(|| a.inputs.slice_mut(..count * hidden)),
                dests: [d0, d1, d2, None],
                dst_stride: 0,
            },
            self.eps(),
        )?;
        let Some((gates, looped, messages)) = a.messages else {
            return Ok(shared);
        };
        // The three gates in one launch; with a shared input they all read its one row.
        let message = looped * MESSAGE_SLOTS * hidden;
        let [q0, q1, q2] = &mut *a.qkv_in;
        self.message_gates(
            &gates[..3],
            &mut *a.gate,
            0,
            looped,
            (&*a.inputs, if shared { 0 } else { hidden }),
            &messages.slice(message..message + 3 * hidden),
            &mut *a.gated,
            [q0, q1, q2]
                .into_iter()
                .zip(norms)
                .map(|(q, norm)| Some((norm, q.as_view_mut())))
                .collect(),
        )?;
        Ok(false)
    }

    /// Attention from the normalized query, key and value inputs to the output projection
    /// in `t.proj`, as MLX-LM's `SlidingAttention` and `FullAttention`.
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

        let conv = match &a.kv_conv {
            KvConv::Static { k, v } => KvConvWeights::Static { k, v },
            KvConv::Dynamic {
                k_static, v_static, ..
            } => KvConvWeights::Dynamic {
                k_dynamic: t.k_dynamic.as_view(),
                v_dynamic: t.v_dynamic.as_view(),
                k_static,
                v_static,
            },
        };
        let [k_state, v_state, ..] = &mut state.conv;
        self.kernels.norm.attention_post(
            &self.stream,
            AttentionPost {
                k_raw: &t.k_raw,
                v_raw: &t.v_raw,
                conv,
                k_state,
                v_state,
                q_raw: &t.q_raw,
                q_scale: &a.q_clip_scale,
                r_raw: &t.r_raw,
                r_scale: &a.rel_clip_scale,
                k: &mut t.k,
                v: &mut t.v,
                q: &mut t.q,
                r: &mut t.r,
                log_scale: (!sliding).then_some((
                    position,
                    cfg.log_scaling_alpha,
                    cfg.log_scaling_n_floor as f32,
                )),
                qk_threshold: cfg.qk_l2_clip_threshold,
                rel_threshold: cfg.rel_l2_clip_threshold,
                eps: self.eps(),
            },
        )?;
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
        self.kernels
            .linear
            .gemv_bf16(&self.stream, &mut [Gemv::new(&a.o, &t.gated, &mut t.proj)])
    }

    /// MLX-LM's `MessageGate` for the gates `gates` of message slots `first_slot..`, in one
    /// launch. Depth attention has already written each slot's normalized state and
    /// message into row `looped` of `t.inputs[slot]`. Gate `k` mixes `messages[k]` into the
    /// F32 state `h[k * h_stride..]`, into `out[k]`; with `norms[k]`, it also writes
    /// `bf16(rms_norm(out[k]) * weight)`.
    #[expect(clippy::too_many_arguments, reason = "the gates' inputs and outputs")]
    fn message_gates(
        &self,
        gates: &[MessageGateWeights],
        t: &mut GateScratch,
        first_slot: usize,
        looped: usize,
        (h, h_stride): (&impl DeviceIn<f32>, usize),
        messages: &impl DeviceIn<f32>,
        out: &mut CudaSlice<f32>,
        norms: Vec<Option<(&CudaSlice<bf16>, CudaViewMut<'_, bf16>)>>,
    ) -> anyhow::Result<()> {
        let n = gates.len();
        let hidden = self.hidden();
        ensure!(norms.len() == n, "one norm entry per gate");
        let row = looped * 2 * hidden..(looped + 1) * 2 * hidden;
        let mut steps: Vec<GateStep<'_>> = gates
            .iter()
            .zip(&t.inputs[first_slot..first_slot + n])
            .zip(norms)
            .map(|((g, input), norm)| GateStep {
                down_weight: &g.down,
                input: input.slice(row.clone()),
                forget: &g.forget,
                gain: &g.gain,
                norm,
            })
            .collect();
        self.kernels.misc.message_gate_step(
            &self.stream,
            &mut steps,
            h,
            h_stride,
            messages,
            &mut out.slice_mut(..n * hidden),
            None,
            &mut t.step,
            self.eps(),
        )
    }

    /// MLX-LM's `SparseMoeBlock` for one row from its normalized BF16 input `x`, up to the
    /// expert outputs in `t.down` and their weights in `t.weights`.
    fn moe(&self, w: &LayerWeights, t: &mut MoeScratch, x: &CudaSlice<bf16>) -> anyhow::Result<()> {
        let cfg = &self.config;
        self.kernels.routing.moe_router_step(
            &self.stream,
            x,
            &w.router,
            &w.router_bias,
            &mut t.logits,
            &mut t.indices,
            &mut t.weights,
            &mut t.counter,
            cfg.route_scale,
        )?;

        let (w13, w2) = self.expert_weights(w);
        self.kernels.experts.expert_mlp(
            &self.stream,
            x,
            &t.indices,
            &w13,
            &w2,
            &mut t.act,
            &mut t.down,
            &mut t.mlp_sync,
            cfg.mlp_gate_softcap,
            cfg.mlp_up_softcap,
        )
    }

    /// The layer's `w13` and `w2` expert projections as the expert kernels take them.
    fn expert_weights<'a>(
        &self,
        w: &'a LayerWeights,
    ) -> (experts::ExpertWeights<'a>, experts::ExpertWeights<'a>) {
        let (hidden, width) = (self.config.hidden_size, self.config.intermediate_size);
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
        (w13, w2)
    }

    // The three heavy multi-row operations of prefill. Each is one place to swap a fallback
    // built from the decode kernels for a fast multi-row kernel.

    /// `out [m, n] = x [m, k] @ weight [n, k]^T` for every product, all with the same `m`
    /// and `k`, as one batch.
    ///
    /// Runs the tensor-core GEMM, which hands `m = 1` to the batched GEMV.
    fn gemm_rows(&self, products: &mut [RowProduct<'_>], m: usize) -> anyhow::Result<()> {
        ensure!(m > 0, "no rows");
        for (i, p) in products.iter().enumerate() {
            ensure!(
                p.x.len().is_multiple_of(m) && p.out.len().is_multiple_of(m),
                "product {i}: x has {} and out {} elements, not whole rows of {m}",
                p.x.len(),
                p.out.len()
            );
        }
        let mut gemms: Vec<Gemm<'_, bf16>> = products
            .iter_mut()
            .map(|p| Gemm::new(p.weight, &p.x, &mut p.out))
            .collect();
        self.kernels.linear.gemm_bf16(&self.stream, &mut gemms, m)?;
        Ok(())
    }

    /// Groups the routed `indices` `[m, 8]` of `m` rows by expert into `scratch`, for the
    /// [`Ctx::moe_rows`] calls that follow.
    fn group_rows(
        &self,
        indices: &CudaView<'_, i32>,
        scratch: &mut experts::ExpertScratch,
        m: usize,
    ) -> anyhow::Result<()> {
        ensure!(indices.len() == m * 8, "indices must hold {m} x 8 ids");
        self.kernels.experts.expert_group(
            &self.stream,
            indices,
            self.config.n_routed_experts,
            scratch,
        )
    }

    /// The expert products of `m` rows: `out` `[m, 8, n]` from `x` `[m, 1, k]` (one input
    /// for all 8 experts) or `[m, 8, k]` (one per expert), routed by the indices that
    /// [`Ctx::group_rows`] grouped into `scratch`.
    ///
    /// Runs [`Experts::expert_matmul_grouped`], which multiplies each expert's pairs as one
    /// matrix product so each expert's weights are read once per call.
    fn moe_rows(
        &self,
        x: &CudaView<'_, bf16>,
        weights: &experts::ExpertWeights<'_>,
        out: &mut CudaViewMut<'_, bf16>,
        scratch: &experts::ExpertScratch,
    ) -> anyhow::Result<()> {
        self.kernels
            .experts
            .expert_matmul_grouped(&self.stream, x, weights, out, scratch)
    }

    /// Appends the keys `k` and values `v` (`[m, 8, 128]`) of `m` rows to `cache` and
    /// attends causally from each row's query (`q` `[m, 24, 128]`) into `out`
    /// (`[m, 24, 128]`). Row `i` sits at position `start + i`, where `start` holds one
    /// value. `bias` `[m, 24, extent]` and `sinks` `[m, 24]` are each row's relative bias
    /// and sink logits. The caller checks that `cache` holds the chunk's last position.
    ///
    /// Runs [`Attention::prefill_attention`], a tensor-core flash attention over the cache
    /// and the chunk. A single row goes to the decode kernels.
    #[expect(
        clippy::too_many_arguments,
        reason = "the attention inputs and outputs"
    )]
    fn attention_rows(
        &self,
        q: &CudaView<'_, bf16>,
        k: &CudaView<'_, bf16>,
        v: &CudaView<'_, bf16>,
        cache: &mut KvCache,
        start: &CudaSlice<u32>,
        m: usize,
        bias: &CudaView<'_, bf16>,
        sinks: &CudaView<'_, bf16>,
        scratch: &mut attention::PrefillScratch,
        out: &mut CudaViewMut<'_, bf16>,
    ) -> anyhow::Result<()> {
        self.kernels.attention.prefill_attention(
            &self.stream,
            q,
            k,
            v,
            cache,
            start,
            m,
            bias,
            sinks,
            scratch,
            out,
        )
    }

    fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    fn eps(&self) -> f32 {
        self.config.rms_norm_eps
    }
}
