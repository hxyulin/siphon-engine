//! GPU memory plans: the device bytes a configuration allocates, worked out from the
//! model config and the safetensors headers before any weight is loaded, and the longest
//! context that fits a budget.
//!
//! Every buffer that grows with the context (the global KV caches of the target and the
//! draft) or with the prefill chunk is counted from the shapes `siphon-cuda` allocates, and
//! so is every fixed buffer of about a megabyte or more. The rest, the module images, the
//! CUDA graphs, the small scratch buffers and the allocator's slack, is one measured
//! constant, [`RUNTIME_BYTES`]. The CUDA context itself is [`CONTEXT_BYTES`].
//!
//! The constants below mirror `siphon-cuda`'s; that crate asserts at compile time that they
//! agree.

use std::{fmt, path::Path};

use anyhow::{Context as _, ensure};
use safetensors::Dtype;

use crate::{Checkpoint, KvFormat, TextConfig, WeightFormat};

/// Device memory the CUDA context takes, rounded up: 507 MiB on an RTX 5090 with driver
/// 620, measured as `nvidia-smi`'s free memory before `siphon-serve` started less the free
/// memory it read right after creating the context. That free memory already excludes it.
pub const CONTEXT_BYTES: usize = 512 << 20;

/// Device memory a loaded model takes beyond the buffers the plan counts: the module
/// images of every kernel, the CUDA graphs of the step, the verify widths and the draft
/// rounds, scratch buffers under a megabyte, and the stream-ordered allocator's rounding.
///
/// Measured on an RTX 5090 with driver 620. A thread sampled `cuMemGetInfo` every
/// millisecond while `siphon-serve` loaded, warmed up, prefilled a prompt of 3,820 tokens
/// (more than a chunk) and generated 256 tokens speculatively. The free memory right after
/// creating the context, less the lowest sample, less the counted buffers, came to 69 to
/// 87 MiB over FP8 and BF16 weights and caches, the bundled draft and DFlash 2, and
/// capacities from 65,536 to the largest that fit. This is the largest, rounded up.
pub const RUNTIME_BYTES: usize = 128 << 20;

/// What each stream past the first adds beyond its counted buffers: the CUDA graph of one
/// more batch size and the allocator's rounding of its caches' buffers. Measured on an RTX
/// PRO 6000 Blackwell Max-Q with `siphon-serve --context 16384`: the modules and graphs
/// took 0.03 GiB at 1 stream and 0.11 GiB at 32, and the peak after a request came 48 MiB
/// under the plan without this at 1 stream and 11 to 43 MiB over it at 32.
pub const STREAM_RUNTIME_BYTES: usize = 4 << 20;

/// Free memory a plan leaves unclaimed unless `--margin` says otherwise, for other
/// processes on the GPU and the allocator's growth past the plan. Filled to their largest
/// contexts, the peak came at most 4 MiB past the plan on an RTX 4090 and stayed 44 to
/// 65 MiB under it on an RTX 5090 (`docs/engine.md`).
pub const DEFAULT_MARGIN: usize = 256 << 20;

/// Rows of a prefill chunk unless `--prefill-chunk` says otherwise, and the most it takes.
pub const DEFAULT_PREFILL_CHUNK: usize = 2048;
pub const MAX_PREFILL_CHUNK: usize = 2048;

/// Rows of the largest verify pass.
pub const VERIFY_ROWS: usize = 8;
/// Rows of the largest batched step: one per stream.
pub const BATCH_ROWS: usize = 32;
/// Draft tokens per round; a draft's capacity is the target's plus this and the anchor.
pub const DRAFT_STEPS: usize = 7;
/// Slots of the bundled draft's sliding rings: a window of 1024 and a block of 8.
pub const DRAFT_RING_SLOTS: usize = 1024 + 8;
/// Slots of DFlash 2's rings: a window of 2048 and a block of 8.
pub const DFLASH2_RING_SLOTS: usize = 2048 + 8;
/// Prompt rows whose taps the speculator keeps for a restart, twice: DFlash 2's ring.
pub const TAIL_ROWS: usize = DFLASH2_RING_SLOTS;
/// Plain steps whose taps the speculator keeps at most.
pub const PLAIN_RUN_ROWS: usize = 32;
/// Ranked tokens and context tokens a shortlist head holds at most, and the first special
/// token id, from which on it keeps every token.
pub const SHORTLIST_ROWS: usize = 32768;
pub const SHORTLIST_CONTEXT_ROWS: usize = 16384;
pub const FIRST_SPECIAL: usize = 199_990;
/// The bundled draft's layers, taps, query width and MLP width; a 2DSD DFlash draft has
/// the same widths.
pub const DRAFT_LAYERS: usize = 3;
pub const DRAFT_TAPS: usize = 4;
pub const DRAFT_Q_WIDTH: usize = 24 * 128;
pub const DRAFT_MLP_WIDTH: usize = 9216;
/// DFlash 2's query width and MLP width.
pub const DFLASH2_Q_WIDTH: usize = 32 * 128;
pub const DFLASH2_MLP_WIDTH: usize = 6912;
/// Rank of the Markov heads, the width of their fold rows.
pub const MARKOV_RANK: usize = 256;
/// Target positions one draft commit projects at once.
pub const COMMIT_ROWS: usize = 512;
/// Rows the LM head of a prefill or verify runs at once.
pub const HEAD_ROWS: usize = 32;
/// The FP8 copies of the target pad their rows to a multiple of this.
pub const FP8_ROW_PAD: usize = 32;
/// Floats of one prefill attention partial slot: 128 rows of 128 values, a maximum and a sum.
pub const PREFILL_SLOT: usize = 128 * (128 + 2);
/// Partial rows of the multi-row attention: 8 rows of 24 heads, 33 partials each.
pub const ROWS_ATTENTION_ROWS: usize = 8 * 24 * 33;
/// Slices of the prefill router's inner dimension.
pub const ROUTER_SPLITS: usize = 9;
/// Message candidates the fused depth attention weighs per slot.
pub const FUSED_MAX_CANDIDATES: usize = 6;
/// Capacities are multiples of this.
pub const CAPACITY_STEP: usize = 1024;

/// Approximations of the draft that allocate copies (`siphon_cuda::draft::DraftOptions`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(clippy::struct_excessive_bools, reason = "one flag per approximation")]
pub struct DraftApprox {
    pub fp8_backbone: bool,
    pub fp8_head: bool,
    pub fp8_fold: bool,
    pub fp8_kv: bool,
    pub shortlist: bool,
}

/// Which draft runs beside the target.
#[derive(Clone, Copy, Debug)]
pub enum DraftChoice<'a> {
    /// No draft and no verify passes: plain decode.
    None,
    /// The DSpark draft in the checkpoint's `mtp.safetensors`.
    Bundled(DraftApprox),
    /// The drafter exported to this directory: DFlash 2 or a 2DSD DFlash draft.
    Dflash(&'a Path, DraftApprox),
}

/// What a plan depends on.
#[derive(Clone, Copy, Debug)]
pub struct PlanInputs<'a> {
    pub config: &'a TextConfig,
    /// The target checkpoint, of which only the headers are read.
    pub checkpoint: &'a Checkpoint,
    pub weights: WeightFormat,
    pub kv: KvFormat,
    pub draft: DraftChoice<'a>,
    /// Rows of the largest prefill chunk, or 0 if nothing prefills.
    pub prefill_chunk: usize,
    /// Sequence slots, each with its own caches of the capacity, 1 to [`BATCH_ROWS`].
    /// More than one adds the batched step's buffers, sized for one row per slot.
    pub streams: usize,
    /// Entries of the speculator's prefix cache, 0 for none: each copies one slot's
    /// sliding rings and conv states and the draft's rings.
    pub prefix_entries: usize,
    /// Streaming multiprocessors of the device. The prefill attention keeps two slots of
    /// split partials per SM.
    pub sms: usize,
}

/// Device bytes of one configuration at one capacity, by what holds them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Positions of the global KV caches.
    pub capacity: usize,
    /// Sequence slots, each with caches of `capacity` positions.
    pub streams: usize,
    /// The target's checkpoint tensors.
    pub weights: usize,
    /// The FP8 copies decode reads ([`WeightFormat::Fp8`]).
    pub fp8_copies: usize,
    /// The target's global KV caches, of every slot.
    pub kv: usize,
    /// The sliding rings and the conv states of every slot, and the decode step's buffers.
    pub rings: usize,
    /// The batched step's buffers, with more than one slot.
    pub batch: usize,
    /// The prefill chunk's buffers.
    pub prefill: usize,
    /// The verify pass's buffers and the speculator's.
    pub verify: usize,
    /// The draft's weights, its copies, its rings and its buffers.
    pub draft: usize,
    /// The draft's global KV caches.
    pub draft_kv: usize,
    /// The prefix cache's copies of the rings and conv states.
    pub prefix: usize,
    /// [`RUNTIME_BYTES`], and [`STREAM_RUNTIME_BYTES`] per stream past the first.
    pub runtime: usize,
}

impl Plan {
    /// Bytes allocated after the CUDA context: what must fit the free memory read right
    /// after creating it.
    #[must_use]
    pub fn total(&self) -> usize {
        self.weights
            + self.fp8_copies
            + self.kv
            + self.rings
            + self.batch
            + self.prefill
            + self.verify
            + self.draft
            + self.draft_kv
            + self.prefix
            + self.runtime
    }
}

/// The bytes a plan may take: `free`, the free memory read right after creating the CUDA
/// context, less `margin` (siphon-serve's `--margin`, [`DEFAULT_MARGIN`] by default), and
/// with a `limit` on the process's memory at most the limit less [`CONTEXT_BYTES`].
#[must_use]
pub fn budget(free: usize, limit: Option<usize>, margin: usize) -> usize {
    let free = free.saturating_sub(margin);
    limit.map_or(free, |l| free.min(l.saturating_sub(CONTEXT_BYTES)))
}

/// Bytes as GiB with two decimals.
#[must_use]
pub fn gib(bytes: usize) -> String {
    format!("{:.2}", bytes as f64 / f64::from(1u32 << 30))
}

impl fmt::Display for Plan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts = [
            ("weights", self.weights),
            ("FP8 copies", self.fp8_copies),
            ("KV", self.kv),
            ("rings", self.rings),
            ("batch", self.batch),
            ("prefill", self.prefill),
            ("verify", self.verify),
            ("draft", self.draft),
            ("draft KV", self.draft_kv),
            ("prefix", self.prefix),
            ("runtime", self.runtime),
        ];
        let listed: Vec<String> = parts
            .iter()
            .filter(|(_, bytes)| *bytes > 0)
            .map(|(name, bytes)| format!("{name} {}", gib(*bytes)))
            .collect();
        let streams = if self.streams > 1 {
            format!(" x {} streams", self.streams)
        } else {
            String::new()
        };
        write!(
            f,
            "{} GiB for {} positions{streams} ({})",
            gib(self.total()),
            self.capacity,
            listed.join(", ")
        )
    }
}

/// The plan of a configuration at any capacity.
#[derive(Clone, Debug)]
pub struct Planner {
    /// The plan at capacity 0.
    fixed: Plan,
    /// Bytes of one position in the target's and the draft's global caches.
    kv_per_position: usize,
    draft_kv_per_position: usize,
    max_positions: usize,
}

/// The draft's shapes that its memory depends on.
struct DraftShape {
    /// Tensor bytes of its weights.
    weights: usize,
    taps: usize,
    layers: usize,
    global_layers: usize,
    ring_slots: usize,
    q_width: usize,
    mlp_width: usize,
    /// Whether it has a Markov head, whose fold rows a shortlist or `fp8_fold` copies.
    markov: bool,
    kind: DraftKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DraftKind {
    Bundled,
    Dflash,
    Dflash2,
}

impl Planner {
    /// Reads the shapes `inputs` names: the target's from its config and headers, the
    /// draft's from its headers and, for a DFlash drafter, its `config.json`.
    pub fn new(inputs: &PlanInputs<'_>) -> anyhow::Result<Self> {
        let c = inputs.config;
        let hidden = c.hidden_size;
        let kv_dims = c.num_key_value_heads * c.head_dim;
        let visits = c.visit_layers(c.loop_config.num_passes);
        let global_visits = visits.iter().filter(|&&l| !c.is_sliding(l)).count();
        let sliding_visits = visits.len() - global_visits;

        let draft = match inputs.draft {
            DraftChoice::None => None,
            DraftChoice::Bundled(approx) => Some((bundled_shape(inputs.checkpoint)?, approx)),
            DraftChoice::Dflash(dir, approx) => Some((dflash_shape(dir, hidden)?, approx)),
        };
        let taps = draft.as_ref().map_or(0, |(d, _)| d.taps);

        let streams = inputs.streams;
        ensure!(
            (1..=BATCH_ROWS).contains(&streams),
            "{streams} streams; 1 to {BATCH_ROWS} fit a batched step"
        );
        // One slot's sliding rings and conv states.
        let slot_rings = sliding_visits * c.sliding_window_size * 2 * kv_dims * 2
            + visits.len() * (c.sconv_kernel_size - 1) * (2 * kv_dims + 2 * hidden) * 4;
        let rings = streams * slot_rings + c.vocab_size * 2 + c.unpadded_vocab_size * 4;
        let prefill = if inputs.prefill_chunk == 0 {
            0
        } else {
            rows_scratch(c, inputs.prefill_chunk, inputs.sms)
                + inputs.prefill_chunk * taps * hidden * 2
                + dense_scratch(c, inputs.prefill_chunk)
        };
        let (verify, draft_bytes, draft_kv_per_position) = match &draft {
            None => (0, 0, 0),
            Some((d, approx)) => {
                let verify = rows_scratch(c, VERIFY_ROWS, inputs.sms)
                    + invariant_scratch(c, VERIFY_ROWS)
                    + VERIFY_ROWS * taps * hidden * 2
                    + visits.len() * VERIFY_ROWS * (2 * kv_dims + 2 * hidden) * 2
                    + visits.len() * (c.sconv_kernel_size - 1) * (2 * kv_dims + 2 * hidden) * 4
                    // The speculator's prompt tail, twice, and its run of plain steps.
                    + (2 * TAIL_ROWS + PLAIN_RUN_ROWS) * taps * hidden * 2;
                // A device copy of the BF16 LM head for the exact base logits, which the
                // target keeps in host memory with FP8 weights.
                let head_copy =
                    if inputs.weights == WeightFormat::Fp8 && !approx.fp8_head && !approx.shortlist
                    {
                        c.vocab_size * hidden * 2
                    } else {
                        0
                    };
                let kv_bytes = if approx.fp8_kv { 1 } else { 2 };
                (
                    verify,
                    draft_fixed_bytes(c, d, *approx) + head_copy,
                    d.global_layers * 2 * kv_dims * kv_bytes,
                )
            }
        };
        let kv_per_position = streams * global_visits * 2 * kv_dims * inputs.kv.element_bytes();
        // The batched step's chunk buffers for one row per stream, its state tables,
        // positions and tokens, and its rows' taps for the draft.
        let batch = if streams > 1 {
            rows_scratch(c, streams, inputs.sms)
                + invariant_scratch(c, streams)
                + visits.len() * streams * 6 * 8
                + streams * 8
                + streams * taps * hidden * 2
        } else {
            0
        };
        let prefix = match &draft {
            Some((d, _)) => inputs.prefix_entries * (slot_rings + draft_ring_bytes(c, d)),
            _ => 0,
        };
        let fixed = Plan {
            capacity: 0,
            streams,
            weights: target_weight_bytes(inputs.checkpoint, inputs.weights),
            fp8_copies: match inputs.weights {
                WeightFormat::Bf16 => 0,
                WeightFormat::Fp8 => fp8_copy_bytes(c),
            },
            kv: 0,
            rings,
            batch,
            prefill,
            verify,
            draft: draft_bytes,
            draft_kv: (DRAFT_STEPS + 1) * draft_kv_per_position,
            prefix,
            runtime: RUNTIME_BYTES + (streams - 1) * STREAM_RUNTIME_BYTES,
        };
        Ok(Self {
            fixed,
            kv_per_position,
            draft_kv_per_position,
            max_positions: c.model_max_length,
        })
    }

    /// The plan at `capacity` positions.
    #[must_use]
    pub fn plan(&self, capacity: usize) -> Plan {
        Plan {
            capacity,
            kv: capacity * self.kv_per_position,
            draft_kv: self.fixed.draft_kv + capacity * self.draft_kv_per_position,
            ..self.fixed
        }
    }

    /// Bytes each position adds: the target's and the draft's global KV caches.
    #[must_use]
    pub fn bytes_per_position(&self) -> usize {
        self.kv_per_position + self.draft_kv_per_position
    }

    /// The model's longest sequence (`model_max_length`).
    #[must_use]
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// The largest capacity whose plan fits `budget` bytes: a multiple of
    /// [`CAPACITY_STEP`], at most [`Self::max_positions`], or `None` if not even one step
    /// fits.
    #[must_use]
    pub fn max_capacity(&self, budget: usize) -> Option<usize> {
        let room = budget.checked_sub(self.fixed.total())?;
        let capacity = (room / self.bytes_per_position())
            .min(self.max_positions)
            .next_multiple_of(CAPACITY_STEP);
        // `next_multiple_of` rounded up; step back to a multiple that fits.
        let capacity = if self.plan(capacity).total() <= budget && capacity <= self.max_positions {
            capacity
        } else {
            capacity.saturating_sub(CAPACITY_STEP)
        };
        (capacity > 0).then_some(capacity)
    }
}

/// Tensors the target keeps in mapped host memory: the embedding table, and with FP8
/// weights the BF16 LM head, whose FP8 copy every product reads.
fn host_tensor(name: &str, weights: WeightFormat) -> bool {
    name == "model.llm.embed.weight"
        || (weights == WeightFormat::Fp8 && name == "model.llm.unembed.weight")
}

/// Device bytes of the target's text model as uploaded: every `model.llm.` tensor but those
/// in host memory ([`host_tensor`]), with the BF16 tensors the loader widens to F32 counted
/// twice. The vision tower and the draft's tensors are not the target's.
fn target_weight_bytes(checkpoint: &Checkpoint, weights: WeightFormat) -> usize {
    checkpoint
        .tensors()
        .filter(|(name, _)| name.starts_with("model.llm.") && !host_tensor(name, weights))
        .map(|(name, entry)| {
            let widened = [
                "attn_res.",
                ".mlp.gate.",
                "sconv.static_w",
                "middle_cycle_cross_pass_",
            ]
            .iter()
            .any(|p| name.contains(p));
            if widened && entry.dtype == Dtype::BF16 {
                2 * entry.byte_len()
            } else {
                entry.byte_len()
            }
        })
        .sum()
}

/// Bytes of an FP8 copy of the target, `rows` padded, with an F32 scale per padded row.
fn fp8_matrix(rows: usize, k: usize) -> usize {
    let padded = rows.next_multiple_of(FP8_ROW_PAD);
    padded * k + 4 * padded
}

/// Bytes of a draft's FP8 rows: values and an F32 scale per row.
fn fp8_rows(rows: usize, k: usize) -> usize {
    rows * k + 4 * rows
}

/// `Weights::quantize_fp8`: every layer's attention input and output projections, the
/// looped layers' message gates and the LM head.
fn fp8_copy_bytes(c: &TextConfig) -> usize {
    let hidden = c.hidden_size;
    let q_dims = c.num_attention_heads * c.head_dim;
    let kv_dims = c.num_key_value_heads * c.head_dim;
    let rank = c.loop_config.gate_rank;
    let layers: usize = (0..c.num_hidden_layers)
        .map(|layer| {
            let mut inputs = vec![
                q_dims,
                c.num_attention_heads * c.d_rel,
                q_dims,
                c.num_attention_heads,
                kv_dims,
                kv_dims,
            ];
            if !c.is_sliding(layer) {
                inputs.extend([c.num_key_value_heads * c.sconv_kernel_size; 2]);
            }
            let looped = (c.loop_config.core_start..c.loop_config.core_end).contains(&layer);
            let gates = if looped {
                4 * (fp8_matrix(rank, 2 * hidden) + 2 * fp8_matrix(hidden, rank))
            } else {
                0
            };
            inputs
                .iter()
                .map(|&rows| fp8_matrix(rows, hidden))
                .sum::<usize>()
                + fp8_matrix(hidden, q_dims)
                + gates
        })
        .sum();
    layers + fp8_matrix(c.vocab_size, hidden)
}

/// Rows of block states a step keeps: the block starts of the prelude, of one pass of the
/// core and of the coda.
fn max_blocks(c: &TextConfig) -> usize {
    let starts = |r: std::ops::Range<usize>| {
        r.filter(|l| l.is_multiple_of(c.n_layers_per_attn_res_block))
            .count()
    };
    let l = c.loop_config;
    starts(0..l.core_start)
        + starts(l.core_start..l.core_end)
        + starts(l.core_end..c.num_hidden_layers)
}

/// `PrefillScratch::new` for `rows` rows: the buffers of a prefill chunk or a verify pass,
/// without the taps and the verify pass's own state.
fn rows_scratch(c: &TextConfig, rows: usize, sms: usize) -> usize {
    let hidden = c.hidden_size;
    let heads = c.num_attention_heads;
    let q_dims = heads * c.head_dim;
    let kv_dims = c.num_key_value_heads * c.head_dim;
    let dynamic = c.num_key_value_heads * c.sconv_kernel_size;
    let top_k = c.num_experts_per_tok;
    let width = c.intermediate_size;
    let experts = c.n_routed_experts;
    let blocks = max_blocks(c);
    let bf16_per_row = 9 * hidden // embedded, hidden, out, prelude, qkv_in, mlp_x, final, proj
        + 4 * 2 * hidden // the gates' inputs
        + 4 * c.loop_config.gate_rank
        + 4 * kv_dims // k_raw, v_raw, k, v
        + 2 * dynamic
        + 5 * q_dims // q_raw, gate, q, out, gated
        + 2 * heads * c.d_rel
        + heads
        + heads * c.rel_extent
        + top_k * width
        + top_k * hidden
        + hidden; // combined
    let f32_per_row = 2 * blocks * hidden // blocks, bank
        + hidden // bank_normed
        + 4 * FUSED_MAX_CANDIDATES
        + 3 * hidden // attention_inputs
        + hidden // the gates' output
        + 2 * hidden // residual, mlp_input
        + experts
        + ROUTER_SPLITS * experts
        + 2 * top_k; // indices, weights
    let pairs = rows * top_k;
    let quantized = pairs * width.max(hidden / top_k);
    let per_rows = rows * (2 * bf16_per_row + 4 * f32_per_row + 4)
        + 4 * (1 + pairs + rows) // expert counters
        + 4 * pairs // perm
        + 16 * pairs.min(pairs / 64 + experts) // tiles
        + quantized + quantized / 16 + 4 * pairs
        + 4 * (rows * (heads / c.num_key_value_heads)).div_ceil(128) * c.num_key_value_heads;
    let fixed = 4 * 2 * sms * PREFILL_SLOT
        + 4 * ROWS_ATTENTION_ROWS * (c.head_dim + 2)
        + HEAD_ROWS * (c.vocab_size * 2 + c.unpadded_vocab_size * 4);
    per_rows + fixed
}

/// The MXFP8 inputs of a prefill chunk's FP8 dense products (`GateRows::dense`): three
/// inputs of `2 * hidden` values per row and a scale byte per 32 values. Invariant chunks
/// allocate none.
fn dense_scratch(c: &TextConfig, rows: usize) -> usize {
    let values = rows * 3 * 2 * c.hidden_size;
    values + values / 32
}

/// What `PrefillScratch::new` adds for an invariant chunk, the verify pass: every looped
/// layer's message weights and normalized messages, and an output for each of the 3 gates
/// one launch runs (`misc::GATE_SLOTS`).
fn invariant_scratch(c: &TextConfig, rows: usize) -> usize {
    let hidden = c.hidden_size;
    let looped = c.loop_config.core_end - c.loop_config.core_start;
    // The multi-row attention's partials and counters past the 8 rows `rows_scratch`
    // counts, for a batched step's rows.
    let attention = rows.saturating_sub(VERIFY_ROWS)
        * (ROWS_ATTENTION_ROWS / VERIFY_ROWS * (c.head_dim + 2) * 4
            + 2 * c.num_key_value_heads * 4);
    rows * ((looped - 1) * 4 * FUSED_MAX_CANDIDATES * 4 + 4 * looped * hidden * 2 + 2 * hidden * 4)
        + attention
}

/// The bundled draft's shape: the checkpoint's tensors outside the text and vision models
/// are `mtp.safetensors`.
fn bundled_shape(checkpoint: &Checkpoint) -> anyhow::Result<DraftShape> {
    let weights: usize = checkpoint
        .tensors()
        .filter(|(name, _)| !name.starts_with("model."))
        .map(|(_, e)| e.byte_len())
        .sum();
    ensure!(weights > 0, "the checkpoint has no draft (mtp.safetensors)");
    Ok(DraftShape {
        weights,
        taps: DRAFT_TAPS,
        layers: DRAFT_LAYERS,
        global_layers: 1,
        ring_slots: DRAFT_RING_SLOTS,
        q_width: DRAFT_Q_WIDTH,
        mlp_width: DRAFT_MLP_WIDTH,
        markov: true,
        kind: DraftKind::Bundled,
    })
}

/// A DFlash drafter's shape, from its `config.json` and headers.
fn dflash_shape(dir: &Path, hidden: usize) -> anyhow::Result<DraftShape> {
    let path = dir.join("config.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let config: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let dflash2 =
        config.pointer("/architectures/0").and_then(|v| v.as_str()) == Some("DFlash2DraftModel");
    let checkpoint = Checkpoint::open(dir)?;
    let weights = checkpoint.tensors().map(|(_, e)| e.byte_len()).sum();
    let fc = checkpoint
        .get("fc.weight")
        .context("the drafter has no fc.weight")?;
    ensure!(
        fc.shape.len() == 2 && fc.shape[1].is_multiple_of(hidden),
        "fc.weight has shape {:?}",
        fc.shape
    );
    let layers = (0..checkpoint.tensors().count())
        .take_while(|l| {
            checkpoint
                .get(&format!("layers.{l}.input_layernorm.weight"))
                .is_some()
        })
        .count();
    ensure!(layers > 0, "the drafter in {} has no layers", dir.display());
    Ok(DraftShape {
        weights,
        taps: fc.shape[1] / hidden,
        layers,
        global_layers: if dflash2 { 0 } else { layers },
        ring_slots: DFLASH2_RING_SLOTS,
        q_width: if dflash2 {
            DFLASH2_Q_WIDTH
        } else {
            DRAFT_Q_WIDTH
        },
        mlp_width: if dflash2 {
            DFLASH2_MLP_WIDTH
        } else {
            DRAFT_MLP_WIDTH
        },
        markov: !dflash2,
        kind: if dflash2 {
            DraftKind::Dflash2
        } else {
            DraftKind::Dflash
        },
    })
}

/// The draft's sliding rings.
fn draft_ring_bytes(c: &TextConfig, d: &DraftShape) -> usize {
    let kv_dims = c.num_key_value_heads * c.head_dim;
    (d.layers - d.global_layers) * d.ring_slots * 2 * kv_dims * 2
}

/// The draft's bytes that do not grow with the capacity: its weights less the BF16
/// projections that FP8 copies replace, the bundled draft's fold, the copies its
/// approximations make, its rings and its larger buffers.
fn draft_fixed_bytes(c: &TextConfig, d: &DraftShape, approx: DraftApprox) -> usize {
    let hidden = c.hidden_size;
    let kv_dims = c.num_key_value_heads * c.head_dim;
    let vocab = c.unpadded_vocab_size;
    let padded = c.vocab_size;
    let mut bytes = d.weights;
    if approx.fp8_backbone {
        // The BF16 projections only the block reads, freed for their FP8 copies.
        bytes -= d.layers * 2 * (2 * d.q_width * hidden + 3 * d.mlp_width * hidden);
    }
    bytes += match d.kind {
        // The fold, its bias, its E4M3 copy and the copy's error bounds.
        DraftKind::Bundled => {
            padded * MARKOV_RANK * 2 + padded * 2 + fp8_rows(vocab, MARKOV_RANK) + vocab * 4
        }
        // The zero bias of the vanilla Markov head.
        DraftKind::Dflash => padded * 2,
        DraftKind::Dflash2 => 0,
    };
    if approx.fp8_backbone {
        let per_layer = fp8_rows(d.q_width, hidden)
            + 2 * fp8_rows(kv_dims, hidden)
            + fp8_rows(hidden, d.q_width)
            + 2 * fp8_rows(d.mlp_width, hidden)
            + fp8_rows(hidden, d.mlp_width);
        let context = match d.kind {
            DraftKind::Bundled => 2 * fp8_rows(kv_dims, hidden),
            DraftKind::Dflash | DraftKind::Dflash2 => 0,
        };
        bytes += d.layers * (per_layer + context);
        if d.kind != DraftKind::Bundled {
            bytes += fp8_rows(hidden, d.taps * hidden);
        }
    }
    // The approximate head's copies of the LM head and fold rows.
    if approx.shortlist {
        let rows = SHORTLIST_ROWS + vocab.saturating_sub(FIRST_SPECIAL) + SHORTLIST_CONTEXT_ROWS;
        bytes += if approx.fp8_head {
            fp8_rows(rows, hidden)
        } else {
            rows * hidden * 2
        };
        if d.markov {
            bytes += if approx.fp8_fold {
                fp8_rows(rows, MARKOV_RANK)
            } else {
                rows * MARKOV_RANK * 2
            } + rows * 2;
        }
        bytes += rows * 4;
    } else {
        if approx.fp8_head {
            bytes += fp8_rows(vocab, hidden);
        }
        if approx.fp8_fold && d.markov {
            bytes += fp8_rows(vocab, MARKOV_RANK);
        }
    }
    bytes += draft_ring_bytes(c, d);
    // Base logits, the commit's mixes and context keys and values, and a DFlash draft's
    // context rows.
    bytes += DRAFT_STEPS * padded * 2 + COMMIT_ROWS * (hidden + 2 * kv_dims) * 2;
    if d.kind != DraftKind::Bundled {
        bytes += 2 * COMMIT_ROWS * hidden * 2;
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TextConfig {
        serde_json::from_str(include_str!("../tests/fixtures/text_config.json"))
            .expect("the fixture parses")
    }

    #[test]
    fn global_kv_per_position_matches_the_known_sizes() {
        let c = config();
        let visits = c.visit_layers(c.loop_config.num_passes);
        let global = visits.iter().filter(|&&l| !c.is_sliding(l)).count();
        assert_eq!(global, 8);
        let kv_dims = c.num_key_value_heads * c.head_dim;
        assert_eq!(
            global * 2 * kv_dims * KvFormat::Bf16.element_bytes(),
            32 << 10
        );
        assert_eq!(
            global * 2 * kv_dims * KvFormat::Fp8.element_bytes(),
            16 << 10
        );
    }

    #[test]
    fn fp8_copies_take_about_1_14_gb() {
        let bytes = fp8_copy_bytes(&config());
        assert!((1.13e9..1.15e9).contains(&(bytes as f64)), "{bytes}");
    }

    #[test]
    fn max_capacity_fits_and_the_next_step_does_not() {
        let planner = Planner {
            fixed: Plan {
                weights: 20 << 30,
                runtime: RUNTIME_BYTES,
                draft_kv: 8 << 12,
                ..Plan::default()
            },
            kv_per_position: 16 << 10,
            draft_kv_per_position: 4 << 10,
            max_positions: 1 << 20,
        };
        let budget = 30 << 30;
        let capacity = planner.max_capacity(budget).expect("something fits");
        assert_eq!(capacity % CAPACITY_STEP, 0);
        assert!(planner.plan(capacity).total() <= budget);
        assert!(planner.plan(capacity + CAPACITY_STEP).total() > budget);
        assert_eq!(planner.max_capacity(1 << 30), None);
        assert_eq!(planner.max_capacity(usize::MAX / 2), Some(1 << 20));
    }

    /// Writes `tensors` (name, dtype, shape) as `model.safetensors` in a new directory.
    fn checkpoint(name: &str, tensors: &[(&str, Dtype, &[usize])]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("siphon-memory-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("creating the test directory");
        let bytes: Vec<Vec<u8>> = tensors
            .iter()
            .map(|&(_, dtype, shape)| {
                vec![0; shape.iter().product::<usize>() * dtype.bitsize() / 8]
            })
            .collect();
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .zip(&bytes)
            .map(|(&(name, dtype, shape), data)| {
                let view = safetensors::tensor::TensorView::new(dtype, shape.to_vec(), data)
                    .expect("a valid view");
                (name.to_owned(), view)
            })
            .collect();
        safetensors::serialize_to_file(views, None, &dir.join("model.safetensors"))
            .expect("writing");
        dir
    }

    #[test]
    fn target_weights_skip_the_vision_tower_and_host_tensors_and_count_widened_tensors_twice() {
        let dir = checkpoint(
            "target",
            &[
                ("model.llm.embed.weight", Dtype::BF16, &[100]),
                ("model.llm.unembed.weight", Dtype::BF16, &[100]),
                ("model.llm.norm.weight", Dtype::BF16, &[25]),
                ("model.llm.layers.0.mlp.gate.weight", Dtype::BF16, &[10]),
                (
                    "model.llm.layers.0.mlp.experts.w13_weight",
                    Dtype::U8,
                    &[1000],
                ),
                ("model.visual.proj_in.weight", Dtype::BF16, &[5000]),
                ("layers.0.mlp.up_proj.weight", Dtype::BF16, &[300]),
            ],
        );
        let checkpoint = Checkpoint::open(&dir).expect("opening");
        assert_eq!(
            target_weight_bytes(&checkpoint, WeightFormat::Bf16),
            200 + 40 + 1000 + 50
        );
        assert_eq!(
            target_weight_bytes(&checkpoint, WeightFormat::Fp8),
            50 + 40 + 1000
        );
        assert_eq!(bundled_shape(&checkpoint).expect("a draft").weights, 600);
        std::fs::remove_dir_all(dir).expect("cleaning up");
    }

    #[test]
    fn dflash2_drafters_keep_rings_and_dflash_drafters_global_caches() {
        let hidden = 16;
        let dir = checkpoint(
            "dflash",
            &[
                ("fc.weight", Dtype::BF16, &[hidden, 5 * hidden]),
                ("layers.0.input_layernorm.weight", Dtype::BF16, &[hidden]),
                ("layers.1.input_layernorm.weight", Dtype::BF16, &[hidden]),
            ],
        );
        let config = dir.join("config.json");
        std::fs::write(&config, r#"{"architectures": ["DFlash2DraftModel"]}"#).expect("writing");
        let d = dflash_shape(&dir, hidden).expect("a DFlash 2 shape");
        assert_eq!((d.taps, d.layers, d.global_layers), (5, 2, 0));
        std::fs::write(&config, r#"{"architectures": ["DFlashDraftModel"]}"#).expect("writing");
        let d = dflash_shape(&dir, hidden).expect("a DFlash shape");
        assert_eq!((d.taps, d.layers, d.global_layers), (5, 2, 2));
        std::fs::remove_dir_all(dir).expect("cleaning up");
    }

    #[test]
    fn prefill_scratch_grows_with_the_chunk() {
        let c = config();
        let small = rows_scratch(&c, 8, 170);
        let large = rows_scratch(&c, 2048, 170);
        // About 0.3 MB per row: the relative bias, the expert activations and the blocks.
        let per_row = (large - small) / 2040;
        assert!((250_000..400_000).contains(&per_row), "{per_row}");
    }
}
