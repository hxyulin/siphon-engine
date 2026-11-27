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

pub use batch::{BATCH_ROWS, BATCH_VERIFY_MAX_ROWS, BatchVerifyTokens, PendingBatchVerify};
pub use prefill::{
    DEFAULT_PREFILL_ATTENTION, DEFAULT_PREFILL_CHUNK, FP8_ATTENTION_KEYS, HybridStepResult,
    HybridVerifyResult, ImageRows, ImageSource, MAX_PREFILL_CHUNK, PackedPrefill,
    PendingHybridVerify, PrefillAttention, PrefillLogits,
};
pub use prefix::{DeviceSpan, GlobalKv, LocalState};
pub use verify::{
    Accept, AcceptTree, PendingTreeVerify, PendingVerify, TreeVerdict, Verdict, VerifyTokens,
    VerifyTreeTokens,
};

pub use crate::ops::{routing::MoePruning, sample::Sampling};

use std::{collections::HashMap, ops::Range, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaGraph, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _, DeviceRepr,
    ValidAsZeroBits,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
};
use half::bf16;
use siphon::{KvQuant, TextConfig};

use crate::{
    DeviceIn, Gpu,
    ops::{
        Products,
        attention::{
            self, Attention, AttentionScratch, DecodeStep, KvCache, KvData, KvFormat, KvRounding,
            MIN_SPLIT_SLOTS, ONE_TILE_SPLIT_SLOTS, RingSlab,
        },
        conv::Conv,
        elementwise::Elementwise,
        experts::{self, ExpertActivations, Experts, MoeEnd},
        experts_rows::ExpertsRows,
        linear::{Gemm, Gemv, Linear, Pace, Prefetch, Start},
        linear_fp8::{
            Fp8Gemm, Fp8Gemv, Fp8Matrix, Fp8Rows, LinearFp8, Mxfp8Gemm, Mxfp8Gemv, Mxfp8Matrix,
        },
        linear_nvfp4::{ARGMAX_MAX_BLOCKS as NVFP4_ARGMAX_MAX_BLOCKS, LinearNvfp4, Nvfp4Gemv},
        linear_rows::{self, LinearRows, RowsProduct},
        misc::{ArgmaxScratch, DepthGateSlot, Misc},
        norm::{KvConvWeights, Norm},
        routing::{DepthStep, NormDest, Routing, StepPartial},
        sample::{Sample, SampleBuffers},
        spec::{Spec, VERIFY_ROWS},
    },
    vmm::PagePool,
    weights::{
        ExpertFormat, InputNorms, KvConv, LayerWeights, LmHeadFormat, MessageGateWeights,
        OutputProjection, WeightFormat, Weights,
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
    linear_nvfp4: LinearNvfp4,
    linear_rows: LinearRows,
    misc: Misc,
    norm: Norm,
    routing: Routing,
    sample: Sample,
    spec: Spec,
}

/// Positions a pooled global cache maps at a time, the default prefill chunk. Each step,
/// prefill, verify and batched step maps what it writes before it runs, whatever its
/// chunk (see [`Model::map_slot`]).
pub const MAP_POSITIONS: usize = prefill::DEFAULT_PREFILL_CHUNK;

/// Maps the global caches of `visits` and its shared sliding-ring slab (if any) to hold
/// every position below `end`, rounded up to [`MAP_POSITIONS`]. Does nothing for caches
/// that are not pooled.
fn map_visits(
    visits: &mut [VisitState],
    ring_slab: Option<&mut RingSlab>,
    end: usize,
    stream: &CudaStream,
) -> anyhow::Result<()> {
    let end = end.next_multiple_of(MAP_POSITIONS);
    let more = pages_to(visits, ring_slab.as_deref(), end);
    if more == 0 {
        return Ok(());
    }
    let free = ring_slab
        .as_deref()
        .map(|s| s.pool().free_pages())
        .or_else(|| {
            visits
                .iter()
                .find_map(|v| v.kv.pooled().map(|p| p[0].pool().free_pages()))
        })
        .unwrap_or(0);
    ensure!(
        more <= free,
        "the KV pool is out of pages: {more} more needed to reach {end} positions, {free} free"
    );
    if let Some(slab) = ring_slab
        && slab.pages_needed() > 0
    {
        slab.map(stream)?;
    }
    for v in visits.iter_mut() {
        let positions = if v.kv.window().is_some() {
            v.kv.slots()
        } else {
            end
        };
        v.kv.map(positions, stream)?;
    }
    Ok(())
}

/// Pages that [`map_visits`] of `end` would take: the global caches' to `end`, and the
/// rings' whole, which a slot maps as soon as it holds a sequence.
fn pages_to(visits: &[VisitState], ring_slab: Option<&RingSlab>, end: usize) -> usize {
    let end = end.next_multiple_of(MAP_POSITIONS);
    ring_slab.map_or(0, RingSlab::pages_needed)
        + visits
            .iter()
            .map(|v| {
                if v.kv.window().is_some() {
                    v.kv.pages_to(v.kv.slots())
                } else {
                    v.kv.pages_to(end)
                }
            })
            .sum::<usize>()
}

/// The caches one layer visit keeps across tokens.
#[derive(Debug)]
struct VisitState {
    kv: KvCache,
    /// Short-conv input history, F32 `[kernel - 1, dims]`: keys, values, post-attention
    /// and post-MLP, as in MLX-LM's `ArraysCache(4)`.
    conv: [CudaSlice<f32>; 4],
    /// Whether the visit rounds the keys and values it writes ([`Model::set_kv_rounding`]).
    round: bool,
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

/// The values of NVFP4 cache slots (`[slots, FP4_ROW]` bytes), as `[slots, 8, 128]`.
fn fp4_to_f32(bytes: &[u8]) -> Vec<f32> {
    const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let heads = attention::KV_HEADS;
    let dims = attention::HEAD_DIM;
    let mut out = Vec::with_capacity(bytes.len() / attention::FP4_ROW * heads * dims);
    for slot in bytes.as_chunks::<{ attention::FP4_ROW }>().0 {
        let (data, scales) = slot.split_at(heads * dims / 2);
        for i in 0..heads * dims {
            let nibble = (data[i / 2] >> (4 * (i % 2))) & 0xf;
            let magnitude = E2M1[usize::from(nibble & 7)];
            let value = if nibble & 8 == 0 {
                magnitude
            } else {
                -magnitude
            };
            out.push(value * e4m3_to_f32(scales[i / 16]));
        }
    }
    out
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
    gate: GateScratch,
    /// Normalized query, key and value inputs of attention. Layer 0 uses only the first.
    qkv_in: [CudaSlice<bf16>; 3],
    attention: AttentionTemps,
    residual: CudaSlice<f32>,
    mlp_x: CudaSlice<bf16>,
    moe: MoeScratch,
    final_hidden: CudaSlice<bf16>,
    logits: CudaSlice<bf16>,
    /// F32 logits after the softcap, `[unpadded_vocab]`.
    capped: CudaSlice<f32>,
    argmax: ArgmaxScratch,
    /// Partials for fused `B = 1` NVFP4 LM-head argmax (`<= NVFP4_ARGMAX_MAX_BLOCKS`).
    nvfp4_partials_vals: CudaSlice<f32>,
    nvfp4_partials_idxs: CudaSlice<i32>,
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
            nvfp4_partials_vals: zeros(stream, NVFP4_ARGMAX_MAX_BLOCKS)?,
            nvfp4_partials_idxs: zeros(stream, NVFP4_ARGMAX_MAX_BLOCKS)?,
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
    /// Slot 0's global cache rows below this have aged out of the rounding's recent window
    /// and been rounded to 4 bits ([`KvRounding::recent`]).
    rounded_below: usize,
    visits: Vec<VisitState>,
    /// Contiguous VMM slab backing slot 0's sliding rings when `kv_pool` is present.
    /// Declared after `visits` so `KvCache::drop` leaks its slices before the slab frees
    /// its virtual address range.
    ring_slab: Option<RingSlab>,
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
    /// The sequence slot 0 holds, unique among all slots' sequences: [`Model::reset`],
    /// [`Model::restore_state`] and [`Model::move_slot`] into a slot start a new one (see
    /// [`Model::slot_sequence`]).
    sequence: u64,
    /// The last sequence number handed out.
    sequences: u64,
    /// Buffers of [`Model::prefill`], allocated on its first call.
    prefill: Option<prefill::PrefillScratch>,
    /// Buffers of [`Model::verify`], allocated on its first call.
    verify: Option<prefill::PrefillScratch>,
    /// The captured verify pass of each width from 1 to [`VERIFY_ROWS`].
    verify_graphs: Vec<Option<StepGraph>>,
    /// The captured tree-verify pass of each node count from 1 to [`VERIFY_ROWS`].
    verify_tree_graphs: Vec<Option<StepGraph>>,
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
    /// Buffers of [`Model::verify_batch`], allocated on its first call.
    batch_verify: Option<prefill::PrefillScratch>,
    /// The captured batched verify pass of each `(seqs, width)` pair.
    batch_verify_graphs: HashMap<(usize, usize), StepGraph>,
    /// Whether batched steps record each row's taps ([`Model::set_batch_taps`]).
    batch_taps: bool,
    /// How slot 0's next tokens are picked ([`Model::set_sampling`]).
    sampling: Sampling,
    /// Whether the verify and batch heads sample ([`Model::enable_sampling`]).
    sampler: bool,
    /// The settings of a step's or a prefill's draw.
    sample_one: SampleBuffers,
    /// Whether `scratch.capped` holds the last step's/prefill's softcapped logits.
    /// `false` after an untraced `B = 1` step with fused NVFP4 LM-head argmax until
    /// materialized lazily for sampling or [`Model::logits`].
    capped_valid: bool,
    /// The pages that back the global caches of every slot, if they are pooled.
    kv_pool: Option<PagePool>,
}

/// Everything a step reads but does not change.
struct Ctx {
    config: TextConfig,
    weights: Weights,
    kernels: Kernels,
    stream: Arc<CudaStream>,
    /// How prefill's expert products take their activations.
    expert_activations: ExpertActivations,
    /// How prefill's global attention multiplies over an FP8 cache.
    prefill_attention: PrefillAttention,
    /// What `prefill_attention` picks for the chunk being enqueued, set per chunk.
    chunk_attention: Products,
    /// How prefill's (and large batched passes') dense products multiply.
    prefill_dense: Products,
    /// How `W_o` (the attention output projection) multiplies when `prefill_dense == Fp8`:
    /// `Fp8` uses MXFP8 block-32 tensor cores (`mxfp8_o`), `Bf16` keeps BF16 (`Linear::gemm_bf16`).
    prefill_dense_o: Products,
    /// The 4-bit rounding of the global caches' writes, for the visits with `round` set.
    kv_rounding: KvRounding,
    /// Opt-in MoE dynamic top-k and probability-threshold expert pruning.
    moe_pruning: MoePruning,
    /// Opt-in logit margin for speculative acceptance (`0.0` for exact lossless acceptance).
    accept_margin: f32,
    /// Opt-in beta-relaxed / maximal-coupling rejection sampling parameter (`(0, 1]`).
    /// `None` uses default coupled Gumbel-Max race; `Some(1.0)` is exact lossless
    /// Leviathan/Chen rejection; `Some(0.70..0.95)` is Pareto-optimal controlled-lossy.
    spec_beta: Option<f32>,
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
    ///
    /// With `SIPHON_KV_POOL_GIB` set, the global caches come from a page pool of that many
    /// GiB ([`Model::with_kv_pool`]), so that tests run pooled memory unchanged.
    pub fn with_formats(
        gpu: &Gpu,
        config: TextConfig,
        weights: Weights,
        passes: usize,
        capacity: usize,
        kv_format: KvFormat,
        weight_format: WeightFormat,
    ) -> anyhow::Result<Self> {
        let pool = match std::env::var("SIPHON_KV_POOL_GIB") {
            Ok(gib) => {
                let gib: f64 = gib.parse().context("SIPHON_KV_POOL_GIB takes GiB")?;
                let (page, _) = crate::vmm::granularity(gpu.context())?;
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a test switch of a few GiB"
                )]
                let pages = (gib * f64::from(1u32 << 30) / page as f64) as usize;
                Some(PagePool::new(gpu.context(), pages)?)
            }
            Err(_) => None,
        };
        Self::with_kv_pool(
            gpu,
            config,
            weights,
            passes,
            capacity,
            kv_format,
            weight_format,
            pool,
        )
    }

    /// [`Model::with_formats`] with every slot's global caches at addresses reserved for
    /// `capacity` positions and backed by pages of `pool` as sequences grow, if given.
    /// Kernels and graphs keep the addresses, so nothing else changes.
    #[expect(clippy::too_many_arguments, reason = "the model's formats and memory")]
    #[expect(clippy::too_many_lines, reason = "every buffer of the model")]
    pub fn with_kv_pool(
        gpu: &Gpu,
        config: TextConfig,
        mut weights: Weights,
        passes: usize,
        capacity: usize,
        kv_format: KvFormat,
        weight_format: WeightFormat,
        kv_pool: Option<PagePool>,
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
            linear_nvfp4: LinearNvfp4::load(ctx)?,
            linear_rows: LinearRows::load(ctx)?,
            misc: Misc::load(ctx)?,
            norm: Norm::load(ctx)?,
            routing: Routing::load(ctx)?,
            sample: Sample::load(ctx)?,
            spec: Spec::load(ctx)?,
        };
        let lm_head_format = weights.lm_head_format.unwrap_or(match weight_format {
            WeightFormat::Bf16 => LmHeadFormat::Bf16,
            WeightFormat::Fp8 => LmHeadFormat::Fp8,
        });
        if weight_format == WeightFormat::Fp8 {
            weights.quantize_fp8(&kernels.linear_fp8, &stream)?;
        }
        weights.quantize_lm_head(
            &kernels.linear_fp8,
            &kernels.linear_nvfp4,
            &stream,
            lm_head_format,
        )?;
        let history = config.sconv_kernel_size - 1;
        let kv_dims = config.num_key_value_heads * config.head_dim;
        let hidden = config.hidden_size;
        let sliding_format = kv_format.sliding_format();
        let formats = config.cache_formats(passes, kv_format);
        let visit_layers = config.visit_layers(passes);
        let sliding_count = visit_layers.iter().filter(|&&l| config.is_sliding(l)).count();
        let ring_slab = match (&kv_pool, sliding_count) {
            (Some(pool), n) if n > 0 => Some(RingSlab::reserve(
                pool,
                n,
                config.sliding_window_size,
                sliding_format,
            )?),
            _ => None,
        };
        let mut ring_idx = 0usize;
        let mut visits: Vec<VisitState> = Vec::new();
        for (v, layer) in visit_layers.into_iter().enumerate() {
            let kv = if config.is_sliding(layer) {
                let window = config.sliding_window_size;
                match ring_slab.as_ref() {
                    Some(slab) => {
                        let idx = ring_idx;
                        ring_idx += 1;
                        KvCache::sliding_in_slab(&stream, window, window, sliding_format, slab, idx)?
                    }
                    None => {
                        KvCache::sliding_ring_with_format(&stream, window, window, sliding_format)?
                    }
                }
            } else {
                let format = formats[v].context("a global visit has a format")?;
                match &kv_pool {
                    Some(pool) => KvCache::global_pooled(&stream, capacity, format, pool)?,
                    None => KvCache::global_with_format(&stream, capacity, format)?,
                }
            };
            // NVFP4 caches take rows already rounded to their values (`kv_round_bf16`).
            let round = kv.format() == KvFormat::Nvfp4;
            visits.push(VisitState {
                kv,
                conv: [
                    stream.alloc_zeros(history * kv_dims)?,
                    stream.alloc_zeros(history * kv_dims)?,
                    stream.alloc_zeros(history * hidden)?,
                    stream.alloc_zeros(history * hidden)?,
                ],
                round,
            });
        }
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
        let sample_one = SampleBuffers::new(&stream, 1)?;
        let has_nvfp4 = visits.iter().any(|v| v.kv.format() == KvFormat::Nvfp4);
        Ok(Self {
            ctx: Ctx {
                config,
                weights,
                kernels,
                stream,
                expert_activations: ExpertActivations::Bf16,
                prefill_attention: DEFAULT_PREFILL_ATTENTION,
                chunk_attention: Products::Bf16,
                prefill_dense: Products::Bf16,
                prefill_dense_o: Products::Bf16,
                kv_rounding: if has_nvfp4 {
                    KvRounding {
                        quant: KvQuant::Nvfp4,
                        ..KvRounding::default()
                    }
                } else {
                    KvRounding::default()
                },
                moe_pruning: MoePruning::default(),
                accept_margin: 0.0,
                spec_beta: None,
            },
            passes,
            rounded_below: 0,
            visits,
            ring_slab,
            scratch,
            max_blocks,
            graph: None,
            tap_graph: None,
            step_taps: None,
            position: 0,
            sequence: 0,
            sequences: 0,
            prefill: None,
            verify: None,
            verify_graphs: (0..VERIFY_ROWS).map(|_| None).collect(),
            verify_tree_graphs: (0..VERIFY_ROWS).map(|_| None).collect(),
            verify_rows: 0,
            tap_layout: crate::ops::spec::TapLayout::stock(),
            slots: Vec::new(),
            batch: None,
            batch_graphs: (0..BATCH_ROWS).map(|_| None).collect(),
            batch_verify: None,
            batch_verify_graphs: HashMap::new(),
            batch_taps: false,
            sampling: Sampling::GREEDY,
            sampler: false,
            sample_one,
            capped_valid: true,
            kv_pool,
        })
    }

    /// Positions each slot's global caches hold at most.
    #[must_use]
    pub fn slot_capacity(&self) -> usize {
        self.visits
            .iter()
            .filter(|v| v.kv.window().is_none())
            .map(|v| v.kv.slots())
            .min()
            .unwrap_or(0)
    }

    /// The page pool of the global caches, if they are pooled.
    #[must_use]
    pub fn kv_pool(&self) -> Option<&PagePool> {
        self.kv_pool.as_ref()
    }

    /// Maps slot `slot`'s global caches to hold every position below `end`, rounded up to
    /// [`MAP_POSITIONS`], from the page pool. The model maps what each step, prefill,
    /// verify and batched step writes before it runs, so callers need this only to keep
    /// memory for a sequence ahead of time, or before copying into a slot. Does nothing
    /// without a pool.
    pub fn map_slot(&mut self, slot: usize, end: usize) -> anyhow::Result<()> {
        let stream = Arc::clone(&self.ctx.stream);
        if slot == 0 {
            return map_visits(&mut self.visits, self.ring_slab.as_mut(), end, &stream);
        }
        self.slot_state(slot)?;
        let s = &mut self.slots[slot - 1];
        map_visits(&mut s.visits, s.ring_slab.as_mut(), end, &stream)
    }

    /// Pages more that [`Model::map_slot`] of `end` would take.
    pub fn slot_pages_to(&self, slot: usize, end: usize) -> anyhow::Result<usize> {
        let (visits, ring_slab) = if slot == 0 {
            (&self.visits, self.ring_slab.as_ref())
        } else {
            let s = self.slot_state(slot)?;
            (&s.visits, s.ring_slab.as_ref())
        };
        Ok(pages_to(visits, ring_slab, end))
    }

    /// Pages of the pool that one slot's rings take while it holds a sequence, and that a
    /// prefix cache entry's copies of them take.
    #[must_use]
    pub fn ring_pages(&self) -> usize {
        self.pages_for(0)
    }

    /// Pages that a sequence of `positions` positions takes from the pool in one slot: its
    /// global caches to that length and its rings.
    #[must_use]
    pub fn pages_for(&self, positions: usize) -> usize {
        let end = positions.next_multiple_of(MAP_POSITIONS);
        self.ring_slab.as_ref().map_or(0, RingSlab::pages_total)
            + self
                .visits
                .iter()
                .map(|v| {
                    let held = if v.kv.window().is_some() {
                        v.kv.slots()
                    } else {
                        end
                    };
                    v.kv.pages_for(held)
                })
                .sum::<usize>()
    }

    /// Pages of the pool that slot `slot`'s global caches and rings hold.
    pub fn slot_pages(&self, slot: usize) -> anyhow::Result<usize> {
        let (visits, ring_slab) = if slot == 0 {
            (&self.visits, self.ring_slab.as_ref())
        } else {
            let s = self.slot_state(slot)?;
            (&s.visits, s.ring_slab.as_ref())
        };
        Ok(ring_slab.map_or(0, RingSlab::held_pages)
            + visits
                .iter()
                .filter_map(|v| v.kv.pooled())
                .map(|p| p.iter().map(|b| b.pages().len()).sum::<usize>())
                .sum::<usize>())
    }

    /// Positions of slot `slot`'s global caches that are mapped: the capacity unless
    /// pooled.
    pub fn slot_mapped(&self, slot: usize) -> anyhow::Result<usize> {
        let visits = if slot == 0 {
            &self.visits
        } else {
            &self.slot_state(slot)?.visits
        };
        Ok(visits
            .iter()
            .filter(|v| v.kv.window().is_none())
            .map(|v| v.kv.mapped())
            .min()
            .unwrap_or(0))
    }

    /// Returns the pages of slot `slot`'s global caches past its first `positions`
    /// positions, and its rings, to the pool, after the model's stream has done its work,
    /// and moves the slot back to `positions` if it was past them. Does nothing without a
    /// pool.
    pub fn unmap_slot_from(&mut self, slot: usize, positions: usize) -> anyhow::Result<()> {
        if self.kv_pool.is_none() {
            return Ok(());
        }
        self.ctx.stream.synchronize()?;
        self.with_slot(slot, |m| {
            let keep = positions.next_multiple_of(MAP_POSITIONS);
            if let Some(slab) = m.ring_slab.as_mut() {
                slab.unmap()?;
            }
            for v in &mut m.visits {
                // A ring is of use only while the slot decodes; an entry keeps a copy.
                let keep = if v.kv.window().is_some() { 0 } else { keep };
                v.kv.unmap_from(keep)?;
            }
            m.position = m.position.min(positions);
            Ok(())
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
                v.kv = KvCache::sliding_ring_with_format(
                    &self.ctx.stream,
                    window,
                    slots,
                    v.kv.format(),
                )?;
            }
        }
        self.ring_slab = None;
        self.graph = None;
        self.tap_graph = None;
        self.verify = None;
        self.verify_graphs.iter_mut().for_each(|g| *g = None);
        self.verify_tree_graphs.iter_mut().for_each(|g| *g = None);
        self.batch = None;
        self.batch_graphs.iter_mut().for_each(|g| *g = None);
        self.batch_verify = None;
        self.batch_verify_graphs.clear();
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
        self.verify_tree_graphs.iter_mut().for_each(|g| *g = None);
            self.batch = None;
            self.batch_graphs.iter_mut().for_each(|g| *g = None);
            self.batch_verify = None;
            self.batch_verify_graphs.clear();
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
        crate::weights::draft::SharedWeights {
            kv_pool: self.kv_pool.as_ref(),
            ..crate::weights::draft::SharedWeights::from(&self.ctx.weights)
        }
    }

    /// Starts a new sequence: zeroes every KV cache and conv state and moves to position 0.
    /// Makes [`Model::prefill`] run its expert products with `activations`. Decode always
    /// takes BF16 activations.
    ///
    /// [`ExpertActivations::Fp8`] first checks that the NVFP4 weight scales allow the FP8
    /// products' exact split (see [`Experts::fp8_inexact_scales`]), and fails otherwise.
    pub fn set_expert_activations(&mut self, activations: ExpertActivations) -> anyhow::Result<()> {
        ensure!(
            crate::BLOCK_SCALED_MMA || activations == ExpertActivations::Bf16,
            "{activations:?} expert activations need the block-scaled MMA, which this sm_89 build lacks"
        );
        if activations == ExpertActivations::Fp8 {
            for (layer, w) in self.ctx.weights.layers.iter().enumerate() {
                if matches!(w.experts.format, ExpertFormat::Nvfp4 { .. }) {
                    for scales in [&w.experts.w13_scale, &w.experts.w2_scale] {
                        let bad = self
                            .ctx
                            .kernels
                            .experts
                            .fp8_inexact_scales(&self.ctx.stream, scales)?;
                        ensure!(
                            bad == 0,
                            "layer {layer}: {bad} NVFP4 scales are too far from their block's \
                             other scale for the FP8 expert products"
                        );
                    }
                }
            }
        }
        self.ctx.expert_activations = activations;
        Ok(())
    }

    /// Makes [`Model::prefill`] run attention over an FP8 or NVFP4 cache as `attention` says
    /// ([`DEFAULT_PREFILL_ATTENTION`] until set). Decode and BF16 caches are unaffected.
    /// [`PrefillAttention::Fp8`] and [`PrefillAttention::Fp4`] need
    /// [`crate::BLOCK_SCALED_MMA`].
    pub fn set_prefill_attention(&mut self, attention: PrefillAttention) -> anyhow::Result<()> {
        ensure!(
            crate::BLOCK_SCALED_MMA
                || !matches!(attention, PrefillAttention::Fp8 | PrefillAttention::Fp4),
            "{attention} prefill attention needs the block-scaled MMA, which this sm_89 build lacks"
        );
        self.ctx.prefill_attention = attention;
        Ok(())
    }

    /// Current dense product setting for prefill and wide batched passes (`m >= 48`).
    #[must_use]
    pub fn prefill_dense(&self) -> Products {
        self.ctx.prefill_dense
    }

    /// Makes [`Model::prefill`] — and batched passes ([`Model::step_batch`],
    /// [`Model::verify_batch`]) of at least 48 rows — run their dense products with
    /// `products`: with [`Products::Fp8`], the attention input projections, the attention
    /// output projection (`W_o`, quantized to MXFP8 with block-32 scales), the message
    /// gates' down projections (in prefill), and in batched passes the LM head run on FP8
    /// tensor cores with MXFP8 inputs. It reads the FP8 weight copies, so it needs a model
    /// loaded with [`WeightFormat::Fp8`]; decode and single-sequence verify are unaffected.
    /// Drops the captured batched graphs when changed.
    pub fn set_prefill_dense(&mut self, products: Products) -> anyhow::Result<()> {
        ensure!(
            products != Products::Fp4,
            "dense prefill products are bf16 or fp8, not fp4"
        );
        ensure!(
            crate::BLOCK_SCALED_MMA || products == Products::Bf16,
            "FP8 dense products need the block-scaled MMA, which this sm_89 build lacks"
        );
        ensure!(
            products == Products::Bf16 || self.ctx.weights.layers.iter().all(|l| l.fp8.is_some()),
            "FP8 dense products need the FP8 weight copies (WeightFormat::Fp8)"
        );
        let changed = self.ctx.prefill_dense != products || self.ctx.prefill_dense_o != products;
        if products == Products::Fp8 {
            let hidden = self.ctx.hidden();
            let q_dims = self.ctx.config.num_attention_heads * self.ctx.config.head_dim;
            let mut quantized = false;
            for layer in &mut self.ctx.weights.layers {
                if layer.mxfp8_o.is_none() {
                    layer.mxfp8_o = Some(self.ctx.kernels.linear_fp8.quantize_mxfp8(
                        &self.ctx.stream,
                        &layer.attention.o,
                        hidden,
                        q_dims,
                    )?);
                    quantized = true;
                }
            }
            if quantized {
                self.ctx.stream.synchronize()?;
            }
        } else if self.ctx.weights.output != OutputProjection::Mxfp8 {
            for layer in &mut self.ctx.weights.layers {
                layer.mxfp8_o = None;
            }
        }
        self.ctx.prefill_dense = products;
        self.ctx.prefill_dense_o = products;
        if changed {
            self.batch_graphs.iter_mut().for_each(|g| *g = None);
            self.batch_verify_graphs.clear();
        }
        Ok(())
    }

    /// Overrides the attention output projection (`W_o`) product format under
    /// [`Products::Fp8`] dense dispatch: `Products::Fp8` uses MXFP8 block-32 tensor cores,
    /// while `Products::Bf16` keeps `W_o` in BF16 (using `Linear::gemm_bf16` for prefill
    /// and wide batched passes `m >= 48`). Has no effect when `prefill_dense` is `Bf16`.
    pub fn set_prefill_dense_output(&mut self, products: Products) -> anyhow::Result<()> {
        ensure!(
            crate::BLOCK_SCALED_MMA || products == Products::Bf16,
            "MXFP8 output projection needs the block-scaled MMA, which this sm_89 build lacks"
        );
        if self.ctx.prefill_dense_o != products {
            if products == Products::Fp8 && self.ctx.prefill_dense == Products::Fp8 {
                let hidden = self.ctx.hidden();
                let q_dims = self.ctx.config.num_attention_heads * self.ctx.config.head_dim;
                for layer in &mut self.ctx.weights.layers {
                    if layer.mxfp8_o.is_none() {
                        layer.mxfp8_o = Some(self.ctx.kernels.linear_fp8.quantize_mxfp8(
                            &self.ctx.stream,
                            &layer.attention.o,
                            hidden,
                            q_dims,
                        )?);
                    }
                }
                self.ctx.stream.synchronize()?;
            } else if products == Products::Bf16
                && self.ctx.weights.output != OutputProjection::Mxfp8
            {
                for layer in &mut self.ctx.weights.layers {
                    layer.mxfp8_o = None;
                }
            }
            self.ctx.prefill_dense_o = products;
            self.batch_graphs.iter_mut().for_each(|g| *g = None);
            self.batch_verify_graphs.clear();
        }
        Ok(())
    }

    /// Current MoE expert pruning configuration.
    #[must_use]
    pub fn moe_pruning(&self) -> MoePruning {
        self.ctx.moe_pruning
    }

    /// Sets MoE dynamic top-k and probability-threshold expert pruning. Drops captured
    /// decode, verify and batched graphs when the setting changes, since they hold the
    /// old router arguments.
    pub fn set_moe_pruning(&mut self, pruning: MoePruning) -> anyhow::Result<()> {
        pruning.validate()?;
        if pruning != self.ctx.moe_pruning {
            self.ctx.moe_pruning = pruning;
            self.graph = None;
            self.tap_graph = None;
            self.verify_graphs.iter_mut().for_each(|g| *g = None);
            self.batch_graphs.iter_mut().for_each(|g| *g = None);
            self.batch_verify_graphs.clear();
        }
        Ok(())
    }

    /// Rounds the keys and values that the global visits in `rounding.visits` write through
    /// a 4-bit format before they are stored, in every slot, to measure what a 4-bit cache
    /// costs in quality. Needs BF16 global caches. Prefill, verify and batched steps round
    /// the rows before their attention reads them; a decode step rounds the row it
    /// appended after its attention, so only its own key and value reach that attention
    /// unrounded. Drops the captured graphs.
    pub fn set_kv_rounding(&mut self, rounding: KvRounding) -> anyhow::Result<()> {
        let on = rounding.quant != KvQuant::None;
        ensure!(
            self.visits.iter().all(|v| v.kv.format() != KvFormat::Nvfp4),
            "NVFP4 caches store 4-bit values already; the rounding is for BF16 caches"
        );
        let layers = self.ctx.config.visit_layers(self.passes);
        let mut slots: Vec<&mut Vec<VisitState>> = vec![&mut self.visits];
        slots.extend(self.slots.iter_mut().map(|s| &mut s.visits));
        for visits in slots {
            for (v, state) in visits.iter_mut().enumerate() {
                state.round = on && !self.ctx.config.is_sliding(layers[v]) && rounding.applies(v);
                ensure!(
                    !state.round || state.kv.format() == KvFormat::Bf16,
                    "4-bit rounding needs BF16 global caches"
                );
            }
        }
        self.ctx.kv_rounding = rounding;
        self.graph = None;
        self.tap_graph = None;
        self.verify_graphs.iter_mut().for_each(|g| *g = None);
        self.verify_tree_graphs.iter_mut().for_each(|g| *g = None);
        self.batch_graphs.iter_mut().for_each(|g| *g = None);
        self.batch_verify_graphs.clear();
        Ok(())
    }

    pub fn reset(&mut self) -> anyhow::Result<()> {
        let bulk_rings = if let Some(slab) = self.ring_slab.as_ref() {
            slab.zero(&self.ctx.stream)?;
            true
        } else {
            false
        };
        for visit in &mut self.visits {
            if !(bulk_rings && visit.kv.window().is_some()) {
                visit.kv.clear(&self.ctx.stream)?;
            }
            for state in &mut visit.conv {
                self.ctx.stream.memset_zeros(state)?;
            }
        }
        self.position = 0;
        self.rounded_below = 0;
        self.sequence = self.new_sequence();
        Ok(())
    }

    /// A sequence number no slot has held.
    fn new_sequence(&mut self) -> u64 {
        self.sequences += 1;
        self.sequences
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
                    KvData::Nvfp4 { keys, values } => (
                        fp4_to_f32(&stream.clone_dtoh(keys)?),
                        fp4_to_f32(&stream.clone_dtoh(values)?),
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
                    KvData::Nvfp4 { keys, values } => {
                        let len = keys.len() / v.kv.slots() * slots;
                        KvData::Nvfp4 {
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
        map_visits(&mut self.visits, self.ring_slab.as_mut(), state.position.max(1), stream)?;
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
                (KvData::Fp8 { keys, values }, KvData::Fp8 { keys: k, values: x })
                | (KvData::Nvfp4 { keys, values }, KvData::Nvfp4 { keys: k, values: x }) => {
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
        self.sequence = self.new_sequence();
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
        map_visits(
            &mut self.visits,
            self.ring_slab.as_mut(),
            position + 1,
            &Arc::clone(&self.ctx.stream),
        )?;
        self.position = position;
        Ok(())
    }

    /// The F32 logits of the last step after the softcap, `[unpadded_vocab]`.
    pub fn logits(&self) -> anyhow::Result<Vec<f32>> {
        if self.capped_valid {
            return Ok(self.ctx.stream.clone_dtoh(&self.scratch.capped)?);
        }
        let cfg = &self.ctx.config;
        let lm_head = self
            .ctx
            .weights
            .lm_head_nvfp4
            .as_ref()
            .context("NVFP4 LM head missing when materializing lazy step logits")?;
        let mut tmp_logits = zeros::<bf16>(&self.ctx.stream, cfg.vocab_size)?;
        let mut tmp_capped = zeros::<f32>(&self.ctx.stream, cfg.unpadded_vocab_size)?;
        self.ctx.kernels.linear_nvfp4.gemv(
            &self.ctx.stream,
            &mut [Nvfp4Gemv::new(
                lm_head,
                &self.scratch.final_hidden,
                &mut tmp_logits,
            )],
            Start::Serial,
        )?;
        self.ctx.kernels.elementwise.logit_softcap(
            &self.ctx.stream,
            &tmp_logits,
            &mut tmp_capped,
            cfg.final_logit_softcapping,
        )?;
        Ok(self.ctx.stream.clone_dtoh(&tmp_capped)?)
    }

    /// Runs one token and returns the next token, greedy or drawn with [`Self::sampling`].
    /// Fills `trace` if given; otherwise replays the step's CUDA graph, capturing it first
    /// if needed.
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
        map_visits(&mut self.visits, self.ring_slab.as_mut(), position + 1, &stream)?;
        let s = &mut self.scratch;
        stream.memcpy_htod(&[i32::try_from(token)?], &mut s.token)?;
        stream.memcpy_htod(
            &[u32::try_from(position).context("position does not fit the kernels")?],
            &mut s.position,
        )?;
        if let Some(trace) = trace {
            *trace = Trace::default();
            self.enqueue(Some(trace), false)?;
            self.capped_valid = true;
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
            self.capped_valid = self.ctx.weights.lm_head_nvfp4.is_none();
        }
        self.position += 1;
        self.next_token()
    }

    /// How [`Self::step`], [`Self::prefill`] and the verify pick tokens from now on, and,
    /// with sampling on, [`Self::prefill_slot`]. Starts greedy; a reset keeps it.
    pub fn set_sampling(&mut self, sampling: Sampling) {
        self.sampling = sampling;
    }

    #[must_use]
    pub fn sampling(&self) -> Sampling {
        self.sampling
    }

    /// Lets the verify pass and batched steps sample: their heads then pick each row's
    /// token with `ops::sample`, which takes the argmax of greedy rows by the same rule,
    /// instead of the argmax kernel. Drops the captured verify and batch graphs. Without
    /// this, the verify and batched steps stay greedy whatever [`Self::set_sampling`] says.
    pub fn enable_sampling(&mut self) -> anyhow::Result<()> {
        self.sampler = true;
        let stream = Arc::clone(&self.ctx.stream);
        for p in [
            self.verify.as_mut(),
            self.batch.as_mut(),
            self.batch_verify.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            if p.sample.is_none() {
                p.sample = Some(SampleBuffers::new(&stream, p.rows)?);
            }
            if p.head_logits.is_none() {
                p.head_logits = Some(zeros(&stream, p.rows * self.ctx.config.vocab_size)?);
            }
            p.head_partials = None;
            p.head_fp8 = None;
        }
        self.verify_graphs.iter_mut().for_each(|g| *g = None);
        self.verify_tree_graphs.iter_mut().for_each(|g| *g = None);
        self.batch_graphs.iter_mut().for_each(|g| *g = None);
        self.batch_verify_graphs.clear();
        Ok(())
    }

    /// Whether [`Self::enable_sampling`] was called.
    #[must_use]
    pub fn sampling_enabled(&self) -> bool {
        self.sampler
    }

    /// Sets the opt-in logit margin for speculative acceptance (`0.0` by default, exact
    /// lossless acceptance). When positive, a draft token whose softcapped target logit is
    /// within `margin` of the target's chosen token is accepted as well. Drops any captured
    /// verify and batched-verify graphs when the value changes.
    pub fn set_accept_margin(&mut self, margin: f32) -> anyhow::Result<()> {
        ensure!(
            margin.is_finite() && margin >= 0.0,
            "accept_margin must be finite and non-negative, got {margin}"
        );
        ensure!(
            !(margin > 0.0 && self.ctx.spec_beta.is_some()),
            "accept_margin and spec_beta cannot both be enabled"
        );
        if self.ctx.accept_margin.to_bits() != margin.to_bits() {
            self.ctx.accept_margin = margin;
            self.verify_graphs.iter_mut().for_each(|g| *g = None);
            self.batch_verify_graphs.clear();
        }
        Ok(())
    }

    /// The logit margin used for speculative acceptance (`0.0` for exact lossless acceptance).
    #[must_use]
    pub fn accept_margin(&self) -> f32 {
        self.ctx.accept_margin
    }

    /// Enables or disables beta-relaxed / maximal-coupling rejection sampling (`0 < beta <= 1`)
    /// for DFlash 2 speculation. `Some(1.0)` is exact lossless Leviathan/Chen maximal-coupling
    /// rejection; `Some(0.70..0.90)` relaxes acceptance with 0% out-of-support mass. Drops any
    /// captured verify and batched-verify graphs when changed.
    pub fn set_spec_beta(&mut self, beta: Option<f32>) -> anyhow::Result<()> {
        if let Some(b) = beta {
            ensure!(
                b.is_finite() && b > 0.0 && b <= 1.0,
                "spec_beta must be in (0, 1], got {b}"
            );
            ensure!(
                self.ctx.accept_margin == 0.0,
                "spec_beta and accept_margin cannot both be enabled"
            );
        }
        let changed = self.ctx.spec_beta.map(f32::to_bits) != beta.map(f32::to_bits);
        if changed {
            self.ctx.spec_beta = beta;
            self.verify_graphs.iter_mut().for_each(|g| *g = None);
            self.batch_verify_graphs.clear();
        }
        Ok(())
    }

    /// Active beta-relaxed / maximal-coupling rejection parameter, or `None` for coupled race.
    #[must_use]
    pub fn spec_beta(&self) -> Option<f32> {
        self.ctx.spec_beta
    }

    /// The token after a step or prefill, whose logits are in the decode buffers: the
    /// argmax the step computed, or a draw with [`Self::sampling`] for the token at the
    /// current position.
    fn next_token(&mut self) -> anyhow::Result<u32> {
        let stream = Arc::clone(&self.ctx.stream);
        if !self.sampling.is_greedy() {
            let s = &mut self.scratch;
            if !self.capped_valid {
                let lm_head = self
                    .ctx
                    .weights
                    .lm_head_nvfp4
                    .as_ref()
                    .context("NVFP4 LM head missing when materializing sampled logits")?;
                self.ctx.kernels.linear_nvfp4.gemv(
                    &stream,
                    &mut [Nvfp4Gemv::new(lm_head, &s.final_hidden, &mut s.logits)],
                    Start::Serial,
                )?;
                self.ctx.kernels.elementwise.logit_softcap(
                    &stream,
                    &s.logits,
                    &mut s.capped,
                    self.ctx.config.final_logit_softcapping,
                )?;
                self.capped_valid = true;
            }
            self.sample_one
                .upload(&stream, &[(self.sampling, self.position)])?;
            self.ctx.kernels.sample.run(
                &stream,
                &s.capped,
                self.ctx.config.unpadded_vocab_size,
                &mut self.sample_one,
                &mut s.next,
            )?;
        }
        let next = stream.clone_dtoh(&self.scratch.next)?[0];
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
            ctx.weights.embed.slice(),
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
        if let Some(lm_head) = &ctx.weights.lm_head_nvfp4 {
            if let Some(t) = trace {
                ctx.kernels.linear_nvfp4.gemv(
                    &ctx.stream,
                    &mut [Nvfp4Gemv::new(lm_head, &s.final_hidden, &mut s.logits)],
                    Start::Serial,
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
                t.logits = ctx.stream.clone_dtoh(&s.capped)?;
            } else {
                let n_partials = ctx.kernels.linear_nvfp4.gemv_argmax(
                    &ctx.stream,
                    lm_head,
                    &s.final_hidden,
                    cfg.unpadded_vocab_size,
                    &mut s.nvfp4_partials_vals,
                    &mut s.nvfp4_partials_idxs,
                    Start::Early,
                )?;
                ctx.kernels.misc.argmax_partials(
                    &ctx.stream,
                    &s.nvfp4_partials_vals.slice(..n_partials),
                    &s.nvfp4_partials_idxs.slice(..n_partials),
                    &mut s.next,
                    n_partials,
                    n_partials,
                )?;
            }
            return Ok(());
        }
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
                    ctx.weights.lm_head.slice(),
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
///
/// The RTX 4090 reads DRAM at 1.0 TB/s. Measured there while its expert kernels still
/// decoded E2M1 through CUDA's software fallback, ms per step at contexts 0 and 32K: 1.5
/// took 9.09 and 10.47; 1.1 took 9.12 and 10.51; 0.9 took 8.79 and 10.41; 0.7 took 8.78
/// and 10.32; no prefetch took 9.21 and 10.41.
const PREFETCH_TB_PER_S: f32 = if cfg!(sm89) { 0.8 } else { 1.5 };

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
    /// The weight's per-row FP8 copy, which the multi-row kernel (or FP8 GEMM) reads
    /// instead when every product of a batch has one.
    fp8: Option<&'a Fp8Matrix>,
    /// Optional MXFP8 block-32 copy (for `W_o`), preferred by the FP8 tensor-core GEMM
    /// when `prefill_dense_o == Products::Fp8`.
    mxfp8: Option<&'a Mxfp8Matrix>,
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
    qkv_in: &'a mut [CudaSlice<bf16>; 3],
    gate: &'a mut GateScratch,
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
        let mut prefetch: Vec<&dyn Prefetch> = if let Some(o) =
            w.fp8.as_ref().and_then(|f| f.o.as_ref())
        {
            vec![o]
        } else if self.weights.output == OutputProjection::Mxfp8
            && let Some(o) = w.mxfp8_o.as_ref()
        {
            vec![o]
        } else {
            vec![&w.attention.o]
        };
        if let Some(gates) = w.message_gates.as_ref().filter(|_| gated) {
            prefetch.extend(if rows {
                gates[3].rows_matrices()
            } else {
                gates[3].decode_matrices()
            });
        }
        if rows || w.fp8.is_some() {
            // Prefetch F32 router weights for verify passes and FP8 decode so
            // moe_router_step_2304 hits L2 while overlapping the fused pre-MLP kernel.
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
            let lm_head: &dyn Prefetch =
                match (&self.weights.lm_head_nvfp4, &self.weights.lm_head_fp8) {
                    (Some(nvfp4), _) => nvfp4,
                    (_, Some(fp8)) => fp8,
                    (None, None) => self.weights.lm_head.slice(),
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
                qkv_in: &mut s.qkv_in,
                gate: &mut s.gate,
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
        // The post-attention conv and residual add, depth attention for the MLP input,
        // optional MLP message gate, and MoE router.
        let [.., conv_state, _] = &mut state.conv;
        let fused_end = w.fp8.is_some();
        if let Some((gates, i)) = gates {
            let msg_off = (i * MESSAGE_SLOTS + 3) * hidden;
            let msg_in_row = i * 2 * hidden + hidden..i * 2 * hidden + 2 * hidden;
            let g = &gates[3];
            let mut slot = DepthGateSlot {
                down_weight: &g.down,
                input_msg: s.gate.inputs[3].slice(msg_in_row),
                forget: &g.forget,
                gain: &g.gain,
                fp8: g.fp8.as_ref(),
                norm_weight: &w.mlp_norm,
                normed: s.mlp_x.as_view_mut(),
            };
            let block_rows = blocks.rows(&s.blocks, hidden)?;
            let proj_view = s.attention.proj.as_view();
            let hid_view = (!block_start).then(|| s.hidden.as_view());
            let mlp_q_view = w.mlp_query.as_view();
            self.kernels.misc.depth_gate_mlp_step(
                &self.stream,
                &block_rows,
                &proj_view,
                &w.post_attention_conv,
                conv_state.as_view_mut(),
                hid_view.as_ref(),
                s.residual.as_view_mut(),
                &mlp_q_view,
                &mut slot,
                &s.messages.slice(msg_off..msg_off + hidden),
                None,
                self.eps(),
            )?;
        } else {
            let mlp_dest = NormDest {
                out: s.mlp_x.as_view_mut(),
                weight: Some(&w.mlp_norm),
            };
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
                    out: None,
                    dests: [None, None, None, Some(mlp_dest)],
                    dst_stride: 0,
                },
                self.eps(),
            )?;
        }
        // With FP8 weights the expert kernel also runs the MoE combine, the post-MLP conv and
        // the residual add into `s.out`, which a separate kernel after it took 2.4 us of: a
        // step took 2.862 instead of 2.883 ms at context 0. With BF16 weights the step waits
        // on the prefetch, which forks after the expert kernel, and the fused end delayed it:
        // 3.752 instead of 3.715 ms.
        let fused = fused_end;
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
    #[expect(clippy::too_many_lines, reason = "fused and unfused attention-input paths")]
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
        if let Some((gates, looped, messages)) = a.messages {
            let msg_in_row = looped * 2 * hidden + hidden..looped * 2 * hidden + 2 * hidden;
            let message = looped * MESSAGE_SLOTS * hidden;
            let mut slots = [
                DepthGateSlot {
                    down_weight: &gates[0].down,
                    input_msg: g0.slice(msg_in_row.clone()),
                    forget: &gates[0].forget,
                    gain: &gates[0].gain,
                    fp8: gates[0].fp8.as_ref(),
                    norm_weight: norms[0],
                    normed: q0.as_view_mut(),
                },
                DepthGateSlot {
                    down_weight: &gates[1].down,
                    input_msg: g1.slice(msg_in_row.clone()),
                    forget: &gates[1].forget,
                    gain: &gates[1].gain,
                    fp8: gates[1].fp8.as_ref(),
                    norm_weight: norms[1],
                    normed: q1.as_view_mut(),
                },
                DepthGateSlot {
                    down_weight: &gates[2].down,
                    input_msg: g2.slice(msg_in_row),
                    forget: &gates[2].forget,
                    gain: &gates[2].gain,
                    fp8: gates[2].fp8.as_ref(),
                    norm_weight: norms[2],
                    normed: q2.as_view_mut(),
                },
            ];
            let x_view = a.x.as_view();
            let q_view = query.as_view();
            self.kernels.misc.depth_gate_step(
                &self.stream,
                &a.states,
                &x_view,
                Some(a.normed),
                &q_view,
                if shared { 0 } else { hidden },
                &mut slots,
                &messages.slice(message..message + 3 * hidden),
                None,
                self.eps(),
            )?;
            return Ok(false);
        }
        // Without messages, the depth-attention kernel applies the input norms directly.
        let dests = if shared {
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
                out: None,
                dests: [d0, d1, d2, None],
                dst_stride: 0,
            },
            self.eps(),
        )?;
        Ok(shared)
    }

    /// Attention from the normalized query, key and value inputs to the output projection
    /// in `t.proj`, as MLX-LM's `SlidingAttention` and `FullAttention`.
    #[expect(
        clippy::too_many_lines,
        reason = "follows SlidingAttention and FullAttention op by op"
    )]
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
        // The NVFP4 decode step rounds its own row.
        if state.round && state.kv.format() == KvFormat::Bf16 {
            self.kernels.attention.round_cache_row(
                &self.stream,
                &mut state.kv,
                position.device_ptr(&self.stream).0,
                self.kv_rounding,
            )?;
        }
        // decode_step triggers early, and no prefetch starts here.
        if let Some(o) = w.fp8.as_ref().and_then(|f| f.o.as_ref()) {
            self.kernels.linear_fp8.gemv(
                &self.stream,
                &mut [Fp8Gemv::new(o, &t.gated, &mut t.proj)],
                Start::Early,
            )
        } else if self.weights.output == OutputProjection::Mxfp8
            && let Some(o) = w.mxfp8_o.as_ref()
        {
            self.kernels.linear_fp8.gemv_mxfp8(
                &self.stream,
                &mut [Mxfp8Gemv::new(o, &t.gated, &mut t.proj)],
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
        self.kernels.routing.moe_router_step_pruned(
            &self.stream,
            x,
            &w.router,
            &w.router_bias,
            &mut t.logits,
            &mut t.indices,
            &mut t.weights,
            cfg.route_scale,
            self.moe_pruning,
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
    /// `invariant`. With `invariant`, `m` is at most 32 and the multi-row products of
    /// [`LinearRows`] run: each row's result is the same, bit for bit, for every `m` and
    /// whatever the other rows hold, and the FP8 copies stream when every product has one.
    ///
    /// With [`Products::Fp8`] and `dense` scratch, more than one row, and an FP8 copy of
    /// every weight, it runs [`LinearFp8::gemm`] instead: the inputs round to MXFP8.
    fn gemm_rows(
        &self,
        products: &mut [RowProduct<'_>],
        m: usize,
        invariant: bool,
        dense: Option<&mut Fp8Rows>,
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
        let use_fp8_gemm = self.prefill_dense == Products::Fp8
            && crate::BLOCK_SCALED_MMA
            && if invariant {
                m >= prefill::BATCH_FP8_DENSE_MIN_ROWS
            } else {
                m > 1
            };
        if let Some(scratch) = dense
            && use_fp8_gemm
        {
            if self.prefill_dense_o == Products::Fp8 && products.iter().all(|p| p.mxfp8.is_some()) {
                let mut gemms: Vec<Mxfp8Gemm<'_>> = products
                    .iter_mut()
                    .filter_map(|p| {
                        Some(Mxfp8Gemm {
                            weight: p.mxfp8?,
                            x: p.x.slice(..),
                            out: p.out.slice_mut(..),
                        })
                    })
                    .collect();
                return self
                    .kernels
                    .linear_fp8
                    .gemm_mxfp8(&self.stream, &mut gemms, m, scratch);
            }
            if products.iter().all(|p| p.fp8.is_some()) {
                let mut gemms: Vec<Fp8Gemm<'_>> = products
                    .iter_mut()
                    .filter_map(|p| {
                        Some(Fp8Gemm {
                            weight: p.fp8?,
                            x: p.x.slice(..),
                            out: p.out.slice_mut(..),
                        })
                    })
                    .collect();
                return self
                    .kernels
                    .linear_fp8
                    .gemm(&self.stream, &mut gemms, m, scratch);
            }
        }
        // Under opt-in `--prefill-dense fp8`, BF16 projections without an FP8/MXFP8 copy
        // (such as BF16 `W_o` when `prefill_dense_o == Bf16`) at `m >= 48` dispatch to
        // BF16 tensor-core GEMM (`Linear::gemm_bf16`) instead of `LinearRows`.
        let use_wide_bf16_gemm = invariant
            && self.prefill_dense == Products::Fp8
            && m >= prefill::BATCH_BF16_GEMM_MIN_ROWS;
        if invariant && !use_wide_bf16_gemm {
            ensure!(
                m <= linear_rows::MAX_ROWS,
                "{m} rows; the invariant products take at most {}",
                linear_rows::MAX_ROWS
            );
            let mxfp8 = self.weights.output == OutputProjection::Mxfp8
                && products.iter().all(|p| p.mxfp8.is_some());
            let fp8 = !mxfp8 && products.iter().all(|p| p.fp8.is_some());
            let mut rows: Vec<RowsProduct<'_>> = products
                .iter_mut()
                .map(|p| match (p.mxfp8, p.fp8) {
                    (Some(q), _) if mxfp8 => RowsProduct::mxfp8(q, &p.x, &mut p.out),
                    (_, Some(q)) if fp8 => RowsProduct::fp8(q, &p.x, &mut p.out),
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
            ExpertActivations::Quantized | ExpertActivations::Fp8 => {
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
    /// read once per call. With [`ExpertActivations::Quantized`] or [`ExpertActivations::Fp8`],
    /// it first quantizes `x` into `quantized` (unless `prequantized` is true, when a fused
    /// norm-quant epilogue has already populated `quantized`) and runs the quantized forms
    /// instead.
    #[expect(clippy::too_many_arguments, reason = "expert projection inputs and flags")]
    fn moe_rows(
        &self,
        x: &CudaView<'_, bf16>,
        weights: &experts::ExpertWeights<'_>,
        out: &mut CudaViewMut<'_, bf16>,
        scratch: &experts::ExpertScratch,
        swiglu: bool,
        prequantized: bool,
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
            (activations @ (ExpertActivations::Quantized | ExpertActivations::Fp8), _) => {
                if !prequantized {
                    if activations == ExpertActivations::Fp8 {
                        experts.quantize_rows_fp8(
                            stream,
                            x,
                            weights.k,
                            weights.format,
                            quantized,
                        )?;
                    } else {
                        experts.quantize_rows(stream, x, weights.k, weights.format, quantized)?;
                    }
                }
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
        start: &impl DeviceIn<u32>,
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
            self.chunk_attention,
        )
    }

    fn hidden(&self) -> usize {
        self.config.hidden_size
    }

    fn eps(&self) -> f32 {
        self.config.rms_norm_eps
    }
}
