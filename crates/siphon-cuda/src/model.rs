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
//! [`Model::verify`] runs the anchor and drafts of a speculative round through the same
//! multi-row path; see [`verify`]. [`Model::step_batch`] runs the next token of several
//! sequences through it, each in its own slot; see [`batch`].

mod batch;
mod prefill;
mod prefix;
mod verify;

pub use batch::BATCH_ROWS;
pub use prefill::{DEFAULT_PREFILL_CHUNK, MAX_PREFILL_CHUNK, PrefillLogits};
pub use prefix::LocalState;
pub use verify::{Accept, PendingVerify, Verdict, VerifyTokens};

use std::{ops::Range, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaGraph, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _, DeviceRepr,
    ValidAsZeroBits,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
};
use half::bf16;
use siphon::TextConfig;

use crate::{
    DeviceIn, Gpu,
    ops::{
        attention::{
            self, Attention, AttentionScratch, DecodeStep, KvCache, KvData, KvFormat,
            MIN_SPLIT_SLOTS, ONE_TILE_SPLIT_SLOTS,
        },
        conv::Conv,
        elementwise::Elementwise,
        experts::{self, ExpertActivations, Experts, MoeEnd},
        experts_rows::ExpertsRows,
        linear::{Gemm, Gemv, Linear, Pace, Prefetch, Start},
        linear_fp8::{Fp8Gemv, Fp8Matrix, LinearFp8},
        linear_rows::{self, LinearRows, RowsProduct},
        misc::{ArgmaxScratch, GATE_SLOTS, GateStep, Misc},
        norm::{KvConvWeights, Norm},
        routing::{DepthStep, NormDest, Routing, StepPartial},
        spec::{Spec, VERIFY_ROWS},
    },
    weights::{
        ExpertFormat, InputNorms, KvConv, LayerWeights, MessageGateWeights, WeightFormat, Weights,
    },
};

/// Queries, keys, values and the MLP each get a message on looped layers.
const MESSAGE_SLOTS: usize = 4;

#[derive(Debug)]
struct Kernels {
    attention: Attention,
    conv: Conv,
    elementwise: Elementwise,
    experts: Experts,
    experts_rows: ExpertsRows,
    linear: Linear,
    linear_fp8: LinearFp8,
    linear_rows: LinearRows,
    misc: Misc,
    norm: Norm,
    routing: Routing,
    spec: Spec,
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

/// A device copy of everything a sequence keeps across tokens, from [`Model::save_state`]:
/// the filled slots of every global KV cache, every sliding ring, every conv state and the
/// position.
pub struct SavedState {
    visits: Vec<(KvData, Vec<CudaSlice<f32>>)>,
    position: usize,
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
    partials: AttentionScratch,
    gated: CudaSlice<bf16>,
    /// The output projection, BF16 `[hidden]`.
    proj: CudaSlice<bf16>,
}

/// Intermediates of `SparseMoeBlock`.
struct MoeScratch {
    logits: CudaSlice<f32>,
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
                partials: AttentionScratch::new(stream)?,
                gated: zeros(stream, heads * head_dim)?,
                proj: zeros(stream, hidden)?,
            },
            residual: zeros(stream, hidden)?,
            mlp_input: zeros(stream, hidden)?,
            mlp_x: zeros(stream, hidden)?,
            moe: MoeScratch {
                logits: zeros(stream, cfg.n_routed_experts)?,
                indices: zeros(stream, top_k)?,
                weights: zeros(stream, top_k)?,
                act: zeros(stream, top_k * width)?,
                down: zeros(stream, top_k * hidden)?,
                // The pair counters, then one per row tile of `w2` (at most 16 rows).
                mlp_sync: zeros(stream, 1 + top_k + hidden / 16)?,
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
    /// The step that also records the draft's taps ([`Model::step_with_taps`]), and its
    /// taps, BF16 `[taps, hidden]`.
    tap_graph: Option<StepGraph>,
    step_taps: Option<CudaSlice<bf16>>,
    /// Position of the next token.
    position: usize,
    /// Counts the sequences slot 0 has held: [`Model::reset`], [`Model::restore_state`]
    /// and [`Model::move_slot`] into it start a new one (see [`LocalState`]).
    sequence: u64,
    /// Buffers of [`Model::prefill`], allocated on its first call.
    prefill: Option<prefill::PrefillScratch>,
    /// Buffers of [`Model::verify`], allocated on its first call.
    verify: Option<prefill::PrefillScratch>,
    /// The captured verify pass of each width from 1 to [`VERIFY_ROWS`].
    verify_graphs: Vec<Option<StepGraph>>,
    /// Rows of the last verify.
    verify_rows: usize,
    /// The feature taps that prefill with taps and verify record for the draft.
    tap_layout: crate::ops::spec::TapLayout,
    /// Sequence slots besides slot 0, the state above.
    slots: Vec<batch::Slot>,
    /// Buffers of [`Model::step_batch`], allocated on its first call.
    batch: Option<prefill::PrefillScratch>,
    /// The captured batched step of each size from 1 to [`BATCH_ROWS`].
    batch_graphs: Vec<Option<StepGraph>>,
    /// Whether batched steps record each row's taps ([`Model::set_batch_taps`]).
    batch_taps: bool,
}

/// Everything a step reads but does not change.
struct Ctx {
    config: TextConfig,
    weights: Weights,
    kernels: Kernels,
    stream: Arc<CudaStream>,
    /// How prefill's expert products take their activations.
    expert_activations: ExpertActivations,
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
        Self::with_formats(
            gpu,
            config,
            weights,
            passes,
            capacity,
            kv_format,
            WeightFormat::Bf16,
        )
    }

    /// [`Model::with_kv_format`] with decode reading the attention projections and the LM
    /// head in `weight_format`. [`WeightFormat::Fp8`] quantizes them first, keeping the
    /// BF16 weights for prefill.
    pub fn with_formats(
        gpu: &Gpu,
        config: TextConfig,
        mut weights: Weights,
        passes: usize,
        capacity: usize,
        kv_format: KvFormat,
        weight_format: WeightFormat,
    ) -> anyhow::Result<Self> {
        let ctx = gpu.context();
        let stream = Arc::clone(gpu.stream());
        let kernels = Kernels {
            attention: Attention::load(ctx)?,
            conv: Conv::load(ctx)?,
            elementwise: Elementwise::load(ctx)?,
            experts: Experts::load(ctx)?,
            experts_rows: ExpertsRows::load(ctx)?,
            linear: Linear::load(ctx)?,
            linear_fp8: LinearFp8::load(ctx)?,
            linear_rows: LinearRows::load(ctx)?,
            misc: Misc::load(ctx)?,
            norm: Norm::load(ctx)?,
            routing: Routing::load(ctx)?,
            spec: Spec::load(ctx)?,
        };
        if weight_format == WeightFormat::Fp8 {
            weights.quantize_fp8(&kernels.linear_fp8, &stream)?;
        }
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
                expert_activations: ExpertActivations::Bf16,
            },
            passes,
            visits,
            scratch,
            max_blocks,
            graph: None,
            tap_graph: None,
            step_taps: None,
            position: 0,
            sequence: 0,
            prefill: None,
            verify: None,
            verify_graphs: (0..VERIFY_ROWS).map(|_| None).collect(),
            verify_rows: 0,
            tap_layout: crate::ops::spec::TapLayout::stock(),
            slots: Vec::new(),
            batch: None,
            batch_graphs: (0..BATCH_ROWS).map(|_| None).collect(),
            batch_taps: false,
        })
    }

    /// Replaces every sliding-window ring with one of `slots` slots, at least the window,
    /// and starts a new sequence. The kernels still see only the window. Captured graphs
    /// are dropped, since they hold the old rings.
    pub fn set_ring_slots(&mut self, slots: usize) -> anyhow::Result<()> {
        ensure!(
            self.slots.is_empty(),
            "the ring size changes only without other sequence slots"
        );
        let window = self.ctx.config.sliding_window_size;
        for v in &mut self.visits {
            if v.kv.window().is_some() {
                v.kv = KvCache::sliding_ring(&self.ctx.stream, window, slots)?;
            }
        }
        self.graph = None;
        self.tap_graph = None;
        self.verify = None;
        self.verify_graphs.iter_mut().for_each(|g| *g = None);
        self.batch = None;
        self.batch_graphs.iter_mut().for_each(|g| *g = None);
        self.reset()
    }

    /// The feature taps that [`Model::prefill_with_taps`] and verify passes record for the
    /// draft: by default the bundled draft's ([`crate::ops::spec::TapLayout::stock`]).
    #[must_use]
    pub fn tap_layout(&self) -> &crate::ops::spec::TapLayout {
        &self.tap_layout
    }

    /// Records the taps of `layout` from now on. Drops the verify and batch buffers and
    /// graphs when the layout changes, since they hold the old taps.
    pub fn set_tap_layout(&mut self, layout: crate::ops::spec::TapLayout) -> anyhow::Result<()> {
        let visits = self.visits.len();
        ensure!(
            !layout.is_empty() && layout.len() <= crate::ops::spec::MAX_TAPS,
            "{} taps, expected 1 to {}",
            layout.len(),
            crate::ops::spec::MAX_TAPS
        );
        ensure!(
            layout.visits.iter().all(|&v| v < visits),
            "tap visits {:?} reach past the {visits} visits",
            layout.visits
        );
        if layout != self.tap_layout {
            self.tap_layout = layout;
            self.tap_graph = None;
            self.step_taps = None;
            self.verify = None;
            self.verify_graphs.iter_mut().for_each(|g| *g = None);
            self.batch = None;
            self.batch_graphs.iter_mut().for_each(|g| *g = None);
        }
        Ok(())
    }

    #[must_use]
    pub fn config(&self) -> &TextConfig {
        &self.ctx.config
    }

    /// The stream every kernel of the model runs on.
    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.ctx.stream
    }

    /// The target tensors the draft shares: the embedding, `embed_norm` and the LM head.
    #[must_use]
    pub fn draft_shared(&self) -> crate::weights::draft::SharedWeights<'_> {
        crate::weights::draft::SharedWeights::from(&self.ctx.weights)
    }

    /// Starts a new sequence: zeroes every KV cache and conv state and moves to position 0.
    /// Makes [`Model::prefill`] run its expert products with `activations`. Decode always
    /// takes BF16 activations.
    pub fn set_expert_activations(&mut self, activations: ExpertActivations) {
        self.ctx.expert_activations = activations;
    }

    pub fn reset(&mut self) -> anyhow::Result<()> {
        for visit in &mut self.visits {
            visit.kv.clear(&self.ctx.stream)?;
            for state in &mut visit.conv {
                self.ctx.stream.memset_zeros(state)?;
            }
        }
        self.position = 0;
        self.sequence += 1;
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

    /// Copies the slots of every global KV cache below the position, every sliding ring,
    /// every conv state and the position on the device, so that [`Model::restore_state`]
    /// can return to them exactly. A global cache's slots from the position on are left
    /// out: no kernel reads a slot there before writing it, and copying the whole cache
    /// would double its memory at long capacities.
    pub fn save_state(&self) -> anyhow::Result<SavedState> {
        let stream = &self.ctx.stream;
        let visits = self
            .visits
            .iter()
            .map(|v| {
                // At least one slot, so that no copy is empty.
                let slots = if v.kv.window().is_some() {
                    v.kv.slots()
                } else {
                    self.position.clamp(1, v.kv.slots())
                };
                let kv = match v.kv.data() {
                    KvData::Bf16 { keys, values } => {
                        let len = keys.len() / v.kv.slots() * slots;
                        KvData::Bf16 {
                            keys: stream.clone_dtod(&keys.slice(..len))?,
                            values: stream.clone_dtod(&values.slice(..len))?,
                        }
                    }
                    KvData::Fp8 { keys, values } => {
                        let len = keys.len() / v.kv.slots() * slots;
                        KvData::Fp8 {
                            keys: stream.clone_dtod(&keys.slice(..len))?,
                            values: stream.clone_dtod(&values.slice(..len))?,
                        }
                    }
                };
                let conv = v
                    .conv
                    .iter()
                    .map(|c| stream.clone_dtod(c))
                    .collect::<Result<_, _>>()?;
                Ok((kv, conv))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(SavedState {
            visits,
            position: self.position,
        })
    }

    /// Returns every KV cache, conv state and the position to `state`, which
    /// [`Model::save_state`] took from this model.
    pub fn restore_state(&mut self, state: &SavedState) -> anyhow::Result<()> {
        let stream = &self.ctx.stream;
        ensure!(
            state.visits.len() == self.visits.len(),
            "the state has {} visits, the model {}",
            state.visits.len(),
            self.visits.len()
        );
        for (v, (kv, conv)) in self.visits.iter_mut().zip(&state.visits) {
            // A ring is saved whole, a global cache up to the saved position.
            let ring = v.kv.window().is_some();
            let fits = |saved: usize, cache: usize| saved == cache || (!ring && saved < cache);
            match (v.kv.data_mut(), kv) {
                (KvData::Bf16 { keys, values }, KvData::Bf16 { keys: k, values: x }) => {
                    ensure!(fits(k.len(), keys.len()), "the cache sizes differ");
                    stream.memcpy_dtod(k, &mut keys.slice_mut(..k.len()))?;
                    stream.memcpy_dtod(x, &mut values.slice_mut(..x.len()))?;
                }
                (KvData::Fp8 { keys, values }, KvData::Fp8 { keys: k, values: x }) => {
                    ensure!(fits(k.len(), keys.len()), "the cache sizes differ");
                    stream.memcpy_dtod(k, &mut keys.slice_mut(..k.len()))?;
                    stream.memcpy_dtod(x, &mut values.slice_mut(..x.len()))?;
                }
                _ => anyhow::bail!("the cache formats differ"),
            }
            for (dst, src) in v.conv.iter_mut().zip(conv) {
                stream.memcpy_dtod(src, dst)?;
            }
        }
        self.position = state.position;
        self.sequence += 1;
        Ok(())
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
        self.run_step(token, trace, false)
    }

    /// [`Self::step`] that also records the token's feature taps for the draft, which
    /// [`Self::step_taps`] returns until the next step. The taps are copies, so the step's
    /// results equal [`Self::step`]'s.
    pub fn step_with_taps(&mut self, token: u32) -> anyhow::Result<u32> {
        if self.step_taps.is_none() {
            let len = self.tap_layout.len() * self.ctx.hidden();
            self.step_taps = Some(self.ctx.stream.alloc_zeros(len)?);
        }
        self.run_step(token, None, true)
    }

    /// The feature taps of the last [`Self::step_with_taps`], BF16 `[taps, 2304]` in the
    /// order of [`Self::tap_layout`].
    pub fn step_taps(&self) -> anyhow::Result<CudaView<'_, bf16>> {
        Ok(self
            .step_taps
            .as_ref()
            .context("no step recorded taps")?
            .as_view())
    }

    fn run_step(
        &mut self,
        token: u32,
        trace: Option<&mut Trace>,
        taps: bool,
    ) -> anyhow::Result<u32> {
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
            self.enqueue(Some(trace), false)?;
        } else {
            let captured = if taps { &self.tap_graph } else { &self.graph };
            if captured.is_none() {
                let graph = self.capture(taps)?;
                *(if taps {
                    &mut self.tap_graph
                } else {
                    &mut self.graph
                }) = Some(graph);
            }
            let graph = if taps { &self.tap_graph } else { &self.graph };
            graph
                .as_ref()
                .context("the step graph is missing")?
                .0
                .launch()?;
        }
        let next = stream.clone_dtoh(&self.scratch.next)?[0];
        self.position += 1;
        u32::try_from(next).context("argmax returned a negative index")
    }

    /// Records the untraced step as a CUDA graph.
    fn capture(&mut self, taps: bool) -> anyhow::Result<StepGraph> {
        let stream = Arc::clone(&self.ctx.stream);
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let enqueued = self.enqueue(None, taps);
        // End the capture even when enqueueing failed, so the stream is usable again.
        let graph = stream.end_capture(CUgraphInstantiate_flags(0));
        enqueued.context("capturing the decode step")?;
        let graph = graph?.context("the step capture recorded no work")?;
        graph.upload()?;
        Ok(StepGraph(graph))
    }

    /// Enqueues one step on the stream, for the token and position already uploaded, and
    /// copies the draft's taps into `step_taps` if `taps`. Synchronizes only to fill
    /// `trace`.
    #[expect(
        clippy::too_many_lines,
        reason = "one step reads top to bottom like _run_loop"
    )]
    fn enqueue(&mut self, mut trace: Option<&mut Trace>, taps: bool) -> anyhow::Result<()> {
        let ctx = &self.ctx;
        let s = &mut self.scratch;
        let mut taps = StepTaps {
            layout: &self.tap_layout,
            taps: if taps { self.step_taps.as_mut() } else { None },
            visit: 0,
        };
        let cfg = &ctx.config;
        let (core_start, core_end) = (cfg.loop_config.core_start, cfg.loop_config.core_end);
        let hidden = ctx.hidden();
        let position = s.position.device_ptr(&ctx.stream).0;
        let windows: Vec<Option<Window>> = self
            .visits
            .iter()
            .map(|v| {
                v.kv.ring_buffers(&ctx.stream)
                    .map(|(buffers, slot_bytes)| Window {
                        buffers,
                        slot_bytes,
                        position,
                    })
            })
            .collect();
        let mut visits = self.visits.iter_mut();
        let mut next_visit = || visits.next().context("fewer visit caches than visits");
        let plan = visit_plan(cfg, self.passes);
        let mut upcoming = plan.iter().enumerate().map(|(v, &(_, u))| Upcoming {
            next_window: windows.get(v + 1).copied().flatten(),
            after_window: windows.get(v + 2).copied().flatten(),
            ..u
        });
        let mut next_plan = || upcoming.next().unwrap_or_default();

        // The embedding, its norm, and the first visit's attention inputs (see
        // `attention_inputs`), which go into the first block state and `qkv_in[0]`.
        let first = &ctx.weights.layers[0];
        let InputNorms::Shared(input_norm) = &first.input_norms else {
            anyhow::bail!("the first layer needs one shared input norm");
        };
        ensure!(
            first.attention_query.is_none() && cfg.is_sliding(0),
            "the first layer must have no depth attention and one shared input"
        );
        ctx.kernels.misc.decode_prologue(
            &ctx.stream,
            &s.token,
            &ctx.weights.embed,
            &ctx.weights.embed_norm,
            input_norm,
            &mut s.hidden,
            &mut s.blocks.slice_mut(..hidden),
            &mut s.qkv_in[0],
            ctx.eps(),
        )?;

        let mut blocks = Blocks {
            len: 0,
            max: self.max_blocks,
        };
        let first_window = windows[0]
            .filter(|_| ctx.weights.layers[plan[0].0.0].fp8.is_some())
            .map(Window::prefetch);
        let mut first = ctx.visit_prefetch(plan[0].0.0, plan[0].0.1, false);
        first.extend(first_window.iter().flatten().map(|r| r as &dyn Prefetch));
        ctx.kernels
            .linear
            .prefetch(&ctx.stream, &first, ctx.prefetch_pace(plan[0].0.0))?;
        for layer in 0..core_start {
            taps.visit(ctx, &s.hidden)?;
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
        // Each pass of the looped layers starts from the prelude's output, which stays in
        // `prelude_hidden`. Instead of copying it into `hidden` before each pass, a graph
        // memcpy node that also stops the next kernel from starting early, the buffers trade
        // roles: the first visit of a pass reads `prelude_hidden` as its input, and the
        // buffer it read becomes `prelude_hidden` again once its output is in `hidden`. With
        // the copies, a step took 5 us longer at context 0 and 23 us longer at 32K.
        std::mem::swap(&mut s.hidden, &mut s.prelude_hidden);
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
                blocks.len = prelude_blocks;
            }
            for (i, layer) in (core_start..core_end).enumerate() {
                if i == 0 {
                    std::mem::swap(&mut s.hidden, &mut s.prelude_hidden);
                }
                taps.visit(ctx, &s.hidden)?;
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
                if i == 0 {
                    // The visit swapped the buffer it read into `out`.
                    std::mem::swap(&mut s.out, &mut s.prelude_hidden);
                }
            }
        }
        for layer in core_end..cfg.num_hidden_layers {
            taps.visit(ctx, &s.hidden)?;
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
        taps.final_hidden(ctx, &s.final_hidden)?;
        if let Some(lm_head) = &ctx.weights.lm_head_fp8 {
            ctx.kernels.linear_fp8.gemv(
                &ctx.stream,
                &mut [Fp8Gemv::new(lm_head, &s.final_hidden, &mut s.logits)],
                Start::Serial,
            )?;
        } else {
            ctx.kernels.linear.gemv_bf16(
                &ctx.stream,
                &mut [Gemv::new(
                    &ctx.weights.lm_head,
                    &s.final_hidden,
                    &mut s.logits,
                )],
                Start::Serial,
            )?;
        }
        // Folding the softcap into the FP8 product's epilogue made a step 4 us slower: its
        // F64 tanh holds each block of the product longer.
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

/// Where a step copies the draft's feature taps, as prefill and verify record them: the
/// partial block entering each tap visit, then the final normed hidden state.
struct StepTaps<'a> {
    layout: &'a crate::ops::spec::TapLayout,
    taps: Option<&'a mut CudaSlice<bf16>>,
    /// Visits enqueued so far.
    visit: usize,
}

impl StepTaps<'_> {
    /// Copies `hidden`, the input of the next visit, if that visit is tapped.
    fn visit(&mut self, ctx: &Ctx, hidden: &CudaSlice<bf16>) -> anyhow::Result<()> {
        let index = self.visit;
        self.visit += 1;
        let Some(taps) = self.taps.as_deref_mut() else {
            return Ok(());
        };
        if let Some(tap) = self.layout.visits.iter().position(|&v| v == index) {
            ctx.kernels
                .spec
                .copy_tap(&ctx.stream, hidden, taps, tap, self.layout.len())?;
        }
        Ok(())
    }

    fn final_hidden(&mut self, ctx: &Ctx, hidden: &CudaSlice<bf16>) -> anyhow::Result<()> {
        let count = self.layout.len();
        match self.taps.as_deref_mut() {
            Some(taps) if self.layout.final_hidden => {
                ctx.kernels
                    .spec
                    .copy_tap(&ctx.stream, hidden, taps, count - 1, count)
            }
            _ => Ok(()),
        }
    }
}

/// Bytes of the LM head the last visit prefetches. 32 MiB took a step from 3.865 to 3.853 ms
/// with BF16 weights and from 2.935 to 2.931 ms with FP8 weights; 16 and 48 MiB gained no
/// more.
const LM_HEAD_PREFETCH_BYTES: usize = 32 << 20;

/// The rate of the paced prefetches with FP8 weights, in TB/s. Once the expert kernel ended
/// the MoE block and the router ran as one cluster, the chain between prefetches got shorter
/// and a faster pace paid. FP8 cache, ms per step at contexts 0, 32K and 128K: 1.1 took
/// 2.857, 3.374 and 4.356; 1.3 took 2.816, 3.329 and 4.312; 1.4 took 2.795, 3.318 and 4.303;
/// 1.5 took 2.780, 3.315 and 4.298; 1.6 took 2.780, 3.327 and 4.315; 1.7 took 2.810, 3.370
/// and 4.356.
const PREFETCH_TB_PER_S: f32 = 1.5;

/// The first bytes of a buffer to prefetch, at most its whole length.
struct PrefetchStart<'a>(&'a dyn Prefetch, usize);

impl Prefetch for PrefetchStart<'_> {
    fn device_bytes(&self, stream: &CudaStream) -> (u64, usize) {
        let (p, n) = self.0.device_bytes(stream);
        (p, n.min(self.1))
    }
}

/// A visit's sliding KV ring, for the prefetch: the key and value buffers, the bytes of one
/// slot, and the device address of the step's position.
#[derive(Clone, Copy)]
struct Window {
    buffers: [(u64, usize); 2],
    slot_bytes: usize,
    position: u64,
}

/// The filled slots of one buffer of a [`Window`].
struct RingPrefetch {
    ptr: u64,
    bytes: usize,
    slot_bytes: usize,
    position: u64,
}

impl Prefetch for RingPrefetch {
    fn device_bytes(&self, _stream: &CudaStream) -> (u64, usize) {
        (self.ptr, self.bytes)
    }

    fn ring(&self, _stream: &CudaStream) -> Option<(usize, u64)> {
        Some((self.slot_bytes, self.position))
    }
}

impl Window {
    /// The filled slots of the keys and the values.
    fn prefetch(self) -> [RingPrefetch; 2] {
        self.buffers.map(|(ptr, bytes)| RingPrefetch {
            ptr,
            bytes,
            slot_bytes: self.slot_bytes,
            position: self.position,
        })
    }
}

/// Which layer a visit runs, and whose weights it prefetches next.
#[derive(Clone, Copy)]
struct VisitPlan {
    layer: usize,
    /// The layer's index among the looped layers, on passes after the first, which take
    /// messages.
    looped: Option<usize>,
    next: Upcoming,
}

/// Each visit's layer and whether it takes messages, with the visits after it: each visit
/// starts prefetching the next one's weights, which need the visit after that.
fn visit_plan(cfg: &TextConfig, passes: usize) -> Vec<((usize, bool), Upcoming)> {
    let (core_start, core_end) = (cfg.loop_config.core_start, cfg.loop_config.core_end);
    let looped = core_end - core_start;
    let plan: Vec<(usize, bool)> = cfg
        .visit_layers(passes)
        .into_iter()
        .enumerate()
        .map(|(v, layer)| {
            (
                layer,
                v >= core_start + looped && v < core_start + passes * looped,
            )
        })
        .collect();
    (0..plan.len())
        .map(|v| {
            (
                plan[v],
                Upcoming {
                    next_window: None,
                    after_window: None,
                    next: plan.get(v + 1).copied(),
                    after: plan.get(v + 2).copied(),
                },
            )
        })
        .collect()
}

/// The visits after one, whose weights it prefetches.
#[derive(Clone, Copy, Default)]
struct Upcoming {
    /// The sliding KV rings of the next visit and of the one after it, if they slide.
    next_window: Option<Window>,
    after_window: Option<Window>,
    /// The next visit's layer and whether it takes messages.
    next: Option<(usize, bool)>,
    /// The same for the visit after the next.
    after: Option<(usize, bool)>,
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
    /// The weight's FP8 copy, which the multi-row kernel reads instead when every product
    /// of a batch has one.
    fp8: Option<&'a Fp8Matrix>,
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
    /// `messages`, the message gates that build their inputs, then the fused attention
    /// step's small weights, which nothing has read since the previous step. The
    /// projections are their FP8 copies when the layer has them. With `rows`, the gates'
    /// matrices are those a verify pass reads.
    fn visit_prefetch(&self, layer: usize, messages: bool, rows: bool) -> Vec<&dyn Prefetch> {
        let w = &self.weights.layers[layer];
        let mut weights: Vec<&dyn Prefetch> = Vec::new();
        if let Some(gates) = w.message_gates.as_ref().filter(|_| messages) {
            for g in &gates[..3] {
                weights.extend(if rows {
                    g.rows_matrices()
                } else {
                    g.decode_matrices()
                });
            }
        }
        let a = &w.attention;
        if let Some(fp8) = &w.fp8 {
            weights.extend(fp8.inputs.iter().map(|m| m as &dyn Prefetch));
        } else {
            weights.extend(
                a.input_projections()
                    .into_iter()
                    .map(|m| m as &dyn Prefetch),
            );
        }
        weights
            .extend([&a.q_clip_scale, &a.rel_clip_scale, &a.rel_table].map(|m| m as &dyn Prefetch));
        if let KvConv::Static { k, v } = &a.kv_conv {
            weights.extend([k as &dyn Prefetch, v]);
        }
        weights
    }

    /// Starts prefetching what a visit of `layer` reads after its projections: the output
    /// projection, the MLP message gate if `gated`, then the projections of the visit
    /// after it, `next`, or the start of the LM head after the last visit. With `rows`,
    /// the gates' matrices are those a verify pass reads.
    fn prefetch_rest(
        &self,
        layer: usize,
        gated: bool,
        next: Option<(usize, bool)>,
        rows: bool,
        window: Option<Window>,
    ) -> anyhow::Result<()> {
        let w = &self.weights.layers[layer];
        let mut prefetch: Vec<&dyn Prefetch> = match &w.fp8 {
            Some(fp8) => vec![&fp8.o],
            None => vec![&w.attention.o],
        };
        if let Some(gates) = w.message_gates.as_ref().filter(|_| gated) {
            prefetch.extend(if rows {
                gates[3].rows_matrices()
            } else {
                gates[3].decode_matrices()
            });
        }
        if rows {
            // The verify router loads its F32 weights only once its previous kernel ends.
            prefetch.push(&w.router);
        }
        let lm_head_start;
        // With FP8 weights, the next visit's sliding KV ring too, once every slot is filled
        // (see `Prefetch::ring`): at 32K its attention took 8.1 us with the window in DRAM.
        let window = window.filter(|_| w.fp8.is_some()).map(Window::prefetch);
        if let Some((layer, messages)) = next {
            prefetch.extend(self.visit_prefetch(layer, messages, rows));
            prefetch.extend(window.iter().flatten().map(|r| r as &dyn Prefetch));
        } else {
            // The last visit's experts leave DRAM bandwidth to spare for the start of the
            // LM head, which the step reads next.
            let lm_head: &dyn Prefetch = match &self.weights.lm_head_fp8 {
                Some(fp8) => fp8,
                None => &self.weights.lm_head,
            };
            lm_head_start = PrefetchStart(lm_head, LM_HEAD_PREFETCH_BYTES);
            prefetch.push(&lm_head_start);
        }
        self.kernels
            .linear
            .prefetch(&self.stream, &prefetch, self.prefetch_pace(layer))
    }

    /// The pace of a prefetch that runs beside a visit of `layer`. With FP8 weights a
    /// visit prefetches half the bytes, so a fixed rate leaves DRAM bandwidth to the main
    /// stream. With BF16 weights the prefetch uses loads, which yield to the main stream:
    /// paced at 1.5 TB/s beside sliding layers, a step took 3.669 instead of 3.767 ms at
    /// context 0 but 4.934 instead of 4.777 ms at 32K, where it slowed the attention's
    /// reads of the window's cache. At 1.2 TB/s it took 3.831 ms at context 0.
    fn prefetch_pace(&self, layer: usize) -> Pace {
        if self.weights.layers[layer].fp8.is_some() {
            Pace::Rate(PREFETCH_TB_PER_S)
        } else {
            Pace::Loads
        }
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
        // The previous visit started prefetching this visit's weights once its experts
        // finished, and the step started the first visit's projections. The first visit
        // starts the rest of its own here, and does not wait for its projections: they read
        // from DRAM alongside the prefetch.
        if layer == 0 {
            self.prefetch_rest(layer, gates.is_some(), next.next, false, next.next_window)?;
        }

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
        // With FP8 weights the expert kernel also runs the MoE combine, the post-MLP conv and
        // the residual add into `s.out`, which a separate kernel after it took 2.4 us of: a
        // step took 2.862 instead of 2.883 ms at context 0. With BF16 weights the step waits
        // on the prefetch, which forks after the expert kernel, and the fused end delayed it:
        // 3.752 instead of 3.715 ms.
        let fused = w.fp8.is_some();
        self.moe(
            w,
            &mut s.moe,
            &s.mlp_x,
            fused.then_some((&mut state.conv[3], &s.residual, &mut s.out)),
        )?;
        // The experts are the visit's last DRAM stream, so the next visit's weights start
        // as soon as they finish, beside the conv and the next visit's depth attention.
        // Forking after a separate conv kernel took a step from 3.767 to 3.829 ms with BF16
        // weights and from 2.883 to 2.897 ms with FP8 weights. Forking before the router, so
        // that the prefetch overlaps the experts, took 4.02 and 3.33 ms. The main stream first
        // waits for this visit's prefetch, which has usually finished by then.
        if let Some((layer, messages)) = next.next {
            self.kernels.linear.join_prefetch(&self.stream)?;
            self.prefetch_rest(layer, messages, next.after, false, next.after_window)?;
        }
        if !fused {
            self.kernels.conv.moe_conv_residual(
                &self.stream,
                &s.moe.down,
                &s.moe.weights,
                &w.post_mlp_conv,
                &mut state.conv[3],
                &s.residual,
                &mut s.out,
            )?;
        }

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
            // Layer 0: no depth attention and no messages, one shared BF16 input. The step's
            // first kernel, `Misc::decode_prologue`, has written its unweighted norm into the
            // first block state, which is `a.normed`, and its normed input into `qkv_in[0]`.
            ensure!(
                a.messages.is_none(),
                "layer without depth attention got messages"
            );
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

        // Every projection of the three inputs in one launch, in the order of
        // `input_projections`.
        let mut outs = vec![
            (q_in, &mut t.q_raw),
            (q_in, &mut t.r_raw),
            (q_in, &mut t.gate),
            (q_in, &mut t.sinks),
            (k_in, &mut t.k_raw),
            (v_in, &mut t.v_raw),
        ];
        if matches!(a.kv_conv, KvConv::Dynamic { .. }) {
            outs.push((k_in, &mut t.k_dynamic));
            outs.push((v_in, &mut t.v_dynamic));
        }
        // Early starts measured faster with FP8 weights (2.94 ms per step against 3.08) and
        // slower with BF16 weights (4.00 against 3.87), whose early blocks take the SMs the
        // prefetch at the start of the visit needs.
        if let Some(fp8) = &w.fp8 {
            let mut products: Vec<_> = fp8
                .inputs
                .iter()
                .zip(&mut outs)
                .map(|(weight, (x, out))| Fp8Gemv::new(weight, *x, *out))
                .collect();
            self.kernels
                .linear_fp8
                .gemv(&self.stream, &mut products, Start::Early)?;
        } else {
            let mut products: Vec<_> = a
                .input_projections()
                .into_iter()
                .zip(&mut outs)
                .map(|(weight, (x, out))| Gemv::new(weight, *x, *out))
                .collect();
            self.kernels
                .linear
                .gemv_bf16(&self.stream, &mut products, Start::Serial)?;
        }
        drop(outs);

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
        // Post-processing, relative bias, KV append, attention and output gate in one
        // launch.
        let [k_state, v_state, ..] = &mut state.conv;
        self.kernels.attention.decode_step(
            &self.stream,
            &mut state.kv,
            position,
            &mut t.partials,
            DecodeStep {
                k_raw: &t.k_raw,
                v_raw: &t.v_raw,
                conv,
                k_state,
                v_state,
                q_raw: &t.q_raw,
                q_scale: &a.q_clip_scale,
                r_raw: &t.r_raw,
                r_scale: &a.rel_clip_scale,
                table: &a.rel_table,
                sinks: &t.sinks,
                gate: &t.gate,
                log_scale: (!sliding)
                    .then_some((cfg.log_scaling_alpha, cfg.log_scaling_n_floor as f32)),
                qk_threshold: cfg.qk_l2_clip_threshold,
                rel_threshold: cfg.rel_l2_clip_threshold,
                eps: self.eps(),
                out: &mut t.gated,
                // With FP8 weights, a full sliding window in 32 one-tile splits took a step at
                // 32K from 3.367 to 3.328 ms and at context 0 from 2.814 to 2.818 ms. With
                // BF16 weights, whose unpaced prefetch then competes with the window's reads,
                // it took 4.770 to 4.834 ms at 32K and 3.736 to 3.755 at context 0.
                min_split_slots: if w.fp8.is_some() {
                    ONE_TILE_SPLIT_SLOTS
                } else {
                    MIN_SPLIT_SLOTS
                },
            },
        )?;
        // decode_step triggers early, and no prefetch starts here.
        if let Some(fp8) = &w.fp8 {
            self.kernels.linear_fp8.gemv(
                &self.stream,
                &mut [Fp8Gemv::new(&fp8.o, &t.gated, &mut t.proj)],
                Start::Early,
            )
        } else {
            self.kernels.linear.gemv_bf16(
                &self.stream,
                &mut [Gemv::new(&a.o, &t.gated, &mut t.proj)],
                Start::Early,
            )
        }
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
                fp8: g.fp8.as_ref(),
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
            self.eps(),
        )
    }

    /// MLX-LM's `SparseMoeBlock` for one row from its normalized BF16 input `x`, up to the
    /// expert outputs in `t.down` and their weights in `t.weights`. With `end`, also the
    /// post-MLP conv with that state and the residual add: `hidden = bf16(residual +
    /// conv(moe(x)))`.
    fn moe(
        &self,
        w: &LayerWeights,
        t: &mut MoeScratch,
        x: &CudaSlice<bf16>,
        end: Option<(&mut CudaSlice<f32>, &CudaSlice<f32>, &mut CudaSlice<bf16>)>,
    ) -> anyhow::Result<()> {
        let cfg = &self.config;
        self.kernels.routing.moe_router_step(
            &self.stream,
            x,
            &w.router,
            &w.router_bias,
            &mut t.logits,
            &mut t.indices,
            &mut t.weights,
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
            end.map(|(conv_state, residual, hidden)| MoeEnd {
                weights: &t.weights,
                conv_weight: &w.post_mlp_conv,
                conv_state,
                residual,
                hidden,
            }),
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
    /// Runs the tensor-core GEMM, which hands `m = 1` to the batched GEMV, unless
    /// `invariant`. With `invariant`, `m` is at most 8 and the multi-row products of
    /// [`LinearRows`] run: each row's result is the same, bit for bit, for every `m` and
    /// whatever the other rows hold, and the FP8 copies stream when every product has one.
    fn gemm_rows(
        &self,
        products: &mut [RowProduct<'_>],
        m: usize,
        invariant: bool,
    ) -> anyhow::Result<()> {
        ensure!(m > 0, "no rows");
        for (i, p) in products.iter().enumerate() {
            ensure!(
                p.x.len().is_multiple_of(m) && p.out.len().is_multiple_of(m),
                "product {i}: x has {} and out {} elements, not whole rows of {m}",
                p.x.len(),
                p.out.len()
            );
        }
        if invariant {
            ensure!(
                m <= linear_rows::MAX_ROWS,
                "{m} rows; the invariant products take at most {}",
                linear_rows::MAX_ROWS
            );
            let fp8 = products.iter().all(|p| p.fp8.is_some());
            let mut rows: Vec<RowsProduct<'_>> = products
                .iter_mut()
                .map(|p| match p.fp8 {
                    Some(q) if fp8 => RowsProduct::fp8(q, &p.x, &mut p.out),
                    _ => RowsProduct::bf16(p.weight, &p.x, &mut p.out),
                })
                .collect();
            return self
                .kernels
                .linear_rows
                .run(&self.stream, &mut rows, m, Start::Early);
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
        let experts = &self.kernels.experts;
        let n = self.config.n_routed_experts;
        match self.expert_activations {
            ExpertActivations::Bf16 => experts.expert_group(&self.stream, indices, n, scratch),
            ExpertActivations::Quantized => {
                experts.expert_group_quantized(&self.stream, indices, n, scratch)
            }
        }
    }

    /// The expert products of `m` rows: `out` `[m, 8, n]` from `x` `[m, 1, k]` (one input
    /// for all 8 experts) or `[m, 8, k]` (one per expert), routed by the indices that
    /// [`Ctx::group_rows`] grouped into `scratch`.
    ///
    /// With `swiglu`, `weights` is `w13` and `out` `[m, 8, n / 2]` receives the softcapped
    /// SwiGLU of the products instead.
    ///
    /// Runs [`Experts::expert_matmul_grouped`] or [`Experts::expert_matmul_swiglu_grouped`],
    /// which multiply each expert's pairs as one matrix product so each expert's weights are
    /// read once per call. With [`ExpertActivations::Quantized`], it first quantizes `x` into
    /// `quantized` and runs the quantized forms instead.
    fn moe_rows(
        &self,
        x: &CudaView<'_, bf16>,
        weights: &experts::ExpertWeights<'_>,
        out: &mut CudaViewMut<'_, bf16>,
        scratch: &experts::ExpertScratch,
        swiglu: bool,
        quantized: &mut experts::QuantizedRows,
    ) -> anyhow::Result<()> {
        let experts = &self.kernels.experts;
        let caps = (self.config.mlp_gate_softcap, self.config.mlp_up_softcap);
        let stream = &self.stream;
        match (self.expert_activations, swiglu) {
            (ExpertActivations::Bf16, true) => experts
                .expert_matmul_swiglu_grouped(stream, x, weights, out, scratch, caps.0, caps.1),
            (ExpertActivations::Bf16, false) => {
                experts.expert_matmul_grouped(stream, x, weights, out, scratch)
            }
            (ExpertActivations::Quantized, _) => {
                experts.quantize_rows(stream, x, weights.k, weights.format, quantized)?;
                if swiglu {
                    experts.expert_matmul_quantized_swiglu_grouped(
                        stream, quantized, weights, out, scratch, caps.0, caps.1,
                    )
                } else {
                    experts
                        .expert_matmul_quantized_grouped(stream, quantized, weights, out, scratch)
                }
            }
        }
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
