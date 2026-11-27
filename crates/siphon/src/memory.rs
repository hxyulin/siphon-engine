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

use crate::{Checkpoint, KvFormat, OutputProjection, TextConfig, VisionFormat, WeightFormat};

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
pub const MAX_PREFILL_CHUNK: usize = 4096;

/// What a KV pool adds beyond its pages: the page tables of every slot's reservations and
/// copies between them. On an RTX PRO 6000 Blackwell, two sequences of 300,000 positions
/// that took turns in a pool of 524,288 peaked 17 MiB past a plan without it.
pub const POOL_RUNTIME_BYTES: usize = 64 << 20;

/// Rows of the largest verify pass.
pub const VERIFY_ROWS: usize = 8;
/// Rows of the largest batched step: one per stream.
pub const BATCH_ROWS: usize = 64;
/// Most total rows (`seqs * width`) one batched verify takes.
pub const BATCH_VERIFY_MAX_ROWS: usize = 256;
/// Most concurrent sequences one batched DFlash 2 speculation call handles.
pub const MAX_BATCH_SPEC_SEQS: usize = 32;
/// Adaptive batched verify widths chosen by `Speculator::choose_batch_width`.
pub const BATCH_SPEC_WIDTHS: [usize; 4] = [2, 4, 6, 8];
/// Tight padded sequence-count buckets for batched verify and batched DFlash 2 CUDA graphs.
/// Uses step 1 for `1..=8` (0% padding) and step 2 for `10..=32` (at most 1 dummy slot,
/// <= 5.9% overhead for `seqs >= 17`), bounding warmup graphs to at most 20 sequence
/// counts across `1..=32` (12 for `C <= 16`) so changing concurrency never triggers live
/// `cuGraphInstantiateWithFlags` stalls.
pub const BATCH_SEQ_BUCKETS: [usize; 20] = [
    1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
];

/// Smallest sequence-count bucket `>= seqs`, clamped to `max_seqs` (`max_seqs >= seqs`).
/// When `seqs > 32` (e.g. a non-DFlash direct `verify_batch` with narrow width), rounds
/// up to a multiple of 8 clamped to `max_seqs`.
#[must_use]
pub fn batch_seq_bucket(seqs: usize, max_seqs: usize) -> usize {
    let cap = max_seqs.max(seqs);
    for &b in &BATCH_SEQ_BUCKETS {
        if b >= seqs {
            return b.min(cap);
        }
    }
    seqs.next_multiple_of(8).min(cap)
}

/// All padded sequence-count buckets reachable for `1..=max_seqs` (capped at
/// [`MAX_BATCH_SPEC_SEQS`]), ascending and deduplicated.
#[must_use]
pub fn batch_seq_buckets(max_seqs: usize) -> Vec<usize> {
    let cap = max_seqs.min(MAX_BATCH_SPEC_SEQS);
    if cap == 0 {
        return Vec::new();
    }
    let mut out: Vec<usize> = BATCH_SEQ_BUCKETS
        .iter()
        .copied()
        .filter(|&b| b <= cap)
        .collect();
    if out.last().copied() != Some(cap) {
        out.push(cap);
    }
    out
}

/// All `(bucket_seqs, width)` batched verify + draft CUDA graph shapes reachable for
/// `1..=max_seqs` sequences under [`BATCH_SPEC_WIDTHS`].
#[must_use]
pub fn batch_spec_graph_shapes(max_seqs: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for seqs in batch_seq_buckets(max_seqs) {
        let max_w = (BATCH_VERIFY_MAX_ROWS / seqs).min(VERIFY_ROWS);
        for w in BATCH_SPEC_WIDTHS {
            if w <= max_w {
                out.push((seqs, w));
            }
        }
    }
    out
}

/// Fixed shortlist head row capacity for `ranked` base+special tokens and `vocab` total
/// tokens. Keeping the base-logit GEMM row count fixed at this capacity avoids CUDA graph
/// invalidation as `note_tokens` appends context tokens.
#[must_use]
pub fn shortlist_capacity(ranked: usize, vocab: usize) -> usize {
    (ranked + SHORTLIST_CONTEXT_ROWS).min(vocab)
}

/// Contiguous virtual-address layout for packing all sliding rings of one slot (or one
/// prefix-cache `LocalState`) into a single VMM reservation, so mapping, zeroing, and
/// unmapping all `ring_count` rings requires one `cuMemSetAccess` / `cuMemUnmap` call
/// instead of `ring_count` separate driver calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RingSlabLayout {
    /// Number of sliding rings packed into the slab.
    pub ring_count: usize,
    /// Bytes of one ring's key buffer (and equally its value buffer).
    pub buffer_bytes: usize,
    /// Stride in bytes between consecutive rings (`2 * buffer_bytes`, keys then values).
    pub ring_stride: usize,
    /// Total virtual reservation size in bytes, rounded up to whole `page_bytes`.
    pub total_bytes: usize,
    /// Physical pages of `page_bytes` backing the entire slab.
    pub pages: usize,
}

impl RingSlabLayout {
    /// Computes the contiguous slab layout for `ring_count` identical sliding rings, each
    /// with `buffer_bytes` for keys and `buffer_bytes` for values, backed by VMM pages of
    /// `page_bytes`.
    #[must_use]
    pub fn new(ring_count: usize, buffer_bytes: usize, page_bytes: usize) -> Self {
        let ring_stride = 2 * buffer_bytes;
        let raw_bytes = ring_count * ring_stride;
        let total_bytes = if raw_bytes == 0 {
            0
        } else {
            raw_bytes.next_multiple_of(page_bytes.max(1))
        };
        let pages = if raw_bytes == 0 {
            0
        } else {
            total_bytes / page_bytes.max(1)
        };
        Self {
            ring_count,
            buffer_bytes,
            ring_stride,
            total_bytes,
            pages,
        }
    }

    /// Byte offsets `(keys_offset, values_offset)` of sliding ring `index` inside the slab.
    #[must_use]
    pub fn ring_offsets(&self, index: usize) -> (usize, usize) {
        let keys = index * self.ring_stride;
        (keys, keys + self.buffer_bytes)
    }
}

/// Target position to pre-map a pooled slot up to so decoding avoids mid-round
/// `cuMemSetAccess` boundary stalls: advances past `position` by `grow_chunks` multiples
/// of `chunk`, capped at `final_position` (rounded up to `chunk`).
#[must_use]
pub fn decode_premap_target(
    position: usize,
    final_position: usize,
    chunk: usize,
    grow_chunks: usize,
) -> usize {
    let chunk = chunk.max(1);
    let desired = position.saturating_add(chunk.saturating_mul(grow_chunks.max(1)));
    desired
        .min(final_position.max(position))
        .next_multiple_of(chunk)
}
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
/// Smallest batched/invariant row count that allocates MXFP8 dense GEMM scratch.
pub const BATCH_FP8_DENSE_MIN_ROWS: usize = 48;
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
/// Positions of the global caches that a pooled KV cache maps at a time: the default prefill
/// chunk, at whose multiples the prefix cache saves.
pub const POOL_CHUNK: usize = 2048;
/// Bytes of a page of the KV pool: the allocation granularity of the RTX 5090 and the RTX
/// PRO 6000 Blackwell. `siphon-serve` checks it against the device's.
pub const KV_PAGE_BYTES: usize = 2 << 20;

/// Approximations of the draft that allocate copies (`siphon_cuda::draft::DraftOptions`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(clippy::struct_excessive_bools, reason = "one flag per approximation")]
pub struct DraftApprox {
    pub fp8_backbone: bool,
    pub nvfp4_backbone: bool,
    pub fp8_head: bool,
    pub nvfp4_head: bool,
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

/// The vision tower of `--vision`, and the largest image a request may send.
#[derive(Clone, Copy, Debug)]
pub struct VisionInputs {
    pub format: VisionFormat,
    /// Image tokens of the largest image ([`max_image_tokens`]).
    pub max_image_tokens: usize,
    /// Native patches per image token: the adapter's merge squared, 4.
    pub patches_per_token: usize,
}

/// Image tokens of the largest image the server takes when it shrinks images to at most
/// `max_pixels` pixels: a square of `max_pixels` covers `ceil(sqrt(max_pixels) / side)`
/// tokens a side, `side` being the pixels of one token (40), and one more row and column
/// leave room for the padding of images that are not square. 1,369 for 1920 x 1080.
#[must_use]
pub fn max_image_tokens(side: usize, max_pixels: usize) -> usize {
    let tokens = max_pixels.isqrt().div_ceil(side) + 1;
    tokens * tokens
}

/// What a plan depends on.
#[derive(Clone, Copy, Debug)]
pub struct PlanInputs<'a> {
    pub config: &'a TextConfig,
    /// The target checkpoint, of which only the headers are read.
    pub checkpoint: &'a Checkpoint,
    pub weights: WeightFormat,
    /// The output projection's format under [`WeightFormat::Fp8`].
    pub output: OutputProjection,
    /// Whether opt-in `--prefill-dense fp8` is enabled (adds per-layer MXFP8 copies of `W_o`).
    pub prefill_dense_fp8: bool,
    pub kv: KvFormat,
    pub draft: DraftChoice<'a>,
    /// Rows of the largest prefill chunk, or 0 if nothing prefills.
    pub prefill_chunk: usize,
    /// Sequence slots, each with its own caches of the capacity, 1 to [`BATCH_ROWS`].
    /// More than one adds the batched step's buffers, sized for one row per slot.
    pub streams: usize,
    /// Whether the global caches of every slot and of the draft share one pool of pages,
    /// as `siphon-serve` runs them: the capacity is then the positions the pool holds over
    /// every sequence, and one sequence may take all of them.
    pub pooled: bool,
    /// Entries of the speculator's prefix cache, 0 for none: each copies one slot's
    /// sliding rings and conv states and the draft's rings.
    pub prefix_entries: usize,
    /// Streaming multiprocessors of the device. The prefill attention keeps two slots of
    /// split partials per SM.
    pub sms: usize,
    /// The vision tower, with `--vision`.
    pub vision: Option<VisionInputs>,
}

/// Device bytes of one configuration at one capacity, by what holds them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// Positions of the global KV caches: of each slot, or with a pool, of the pool.
    pub capacity: usize,
    /// Whether the global caches share one pool of pages ([`PlanInputs::pooled`]).
    pub pooled: bool,
    /// Sequence slots, each with caches of `capacity` positions.
    pub streams: usize,
    /// The target's checkpoint tensors.
    pub weights: usize,
    /// The FP8 copies decode reads ([`WeightFormat::Fp8`]).
    pub fp8_copies: usize,
    /// The target's global KV caches, of every slot, or the pool of pages that the target's
    /// and the draft's global caches share.
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
    /// The vision tower ([`vision_bytes`]) for the largest image, and the rows of the other
    /// images one prefill chunk overlaps.
    pub vision: usize,
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
            + self.vision
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
            (if self.pooled { "KV pool" } else { "KV" }, self.kv),
            ("rings", self.rings),
            ("batch", self.batch),
            ("prefill", self.prefill),
            ("verify", self.verify),
            ("draft", self.draft),
            ("draft KV", self.draft_kv),
            ("prefix", self.prefix),
            ("vision", self.vision),
            ("runtime", self.runtime),
        ];
        let listed: Vec<String> = parts
            .iter()
            .filter(|(_, bytes)| *bytes > 0)
            .map(|(name, bytes)| format!("{name} {}", gib(*bytes)))
            .collect();
        let streams = match (self.pooled, self.streams) {
            (true, n) => format!(" shared by up to {n} streams"),
            (false, 1) => String::new(),
            (false, n) => format!(" x {n} streams"),
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
    /// The pool's shape, with [`PlanInputs::pooled`].
    pool: Option<PoolShape>,
}

/// What a pool holds for one sequence ([`Planner::plan`]).
#[derive(Clone, Debug)]
struct PoolShape {
    /// Bytes of one position in each key and value buffer of a slot's global caches, which
    /// differ in a mixed layout, and in each of the draft's.
    target_rows: Vec<usize>,
    draft_rows: Vec<usize>,
    /// Bytes of one sequence's rings and of a prefix cache entry's copies of them.
    rings: usize,
}

impl PoolShape {
    /// Bytes of the pages that buffers of `rows` bytes per position take for `positions`,
    /// in whole chunks of [`POOL_CHUNK`], each buffer rounded up to whole pages.
    fn pages_bytes(rows: &[usize], positions: usize) -> usize {
        let positions = positions.next_multiple_of(POOL_CHUNK);
        rows.iter()
            .map(|r| (positions * r).next_multiple_of(KV_PAGE_BYTES))
            .sum()
    }
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
    #[expect(clippy::too_many_lines, reason = "one term per buffer")]
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
        // One slot's sliding rings and conv states. With a pool, a slot's rings and an
        // entry's copies of them come from the pool while in use, and only the conv states
        // stay fixed.
        let ring_row_bytes = inputs.kv.sliding_format().row_bytes(kv_dims);
        let ring_bytes = sliding_visits * c.sliding_window_size * 2 * ring_row_bytes;
        let conv_bytes = visits.len() * (c.sconv_kernel_size - 1) * (2 * kv_dims + 2 * hidden) * 4;
        let slot_rings = ring_bytes + conv_bytes;
        let fixed_rings = if inputs.pooled {
            conv_bytes
        } else {
            slot_rings
        };
        let rings = streams * fixed_rings + c.vocab_size * 2 + c.unpadded_vocab_size * 4;
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
                let verify = invariant_rows_scratch(c, VERIFY_ROWS, inputs.sms)
                    + VERIFY_ROWS * taps * hidden * 2
                    + visits.len() * VERIFY_ROWS * (2 * kv_dims + 2 * hidden) * 2
                    // The speculator's prompt tail, twice, and its run of plain steps.
                    + (2 * TAIL_ROWS + PLAIN_RUN_ROWS) * taps * hidden * 2;
                // A device copy of the BF16 LM head for the exact base logits, which the
                // target keeps in host memory with FP8 weights.
                let head_copy = if inputs.weights == WeightFormat::Fp8
                    && !approx.fp8_head
                    && !approx.nvfp4_head
                    && !approx.shortlist
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
        let slot_kv_per_position = c.global_cache_bytes(c.loop_config.num_passes, inputs.kv);
        let kv_per_position = if inputs.pooled {
            slot_kv_per_position
        } else {
            streams * slot_kv_per_position
        };
        let pool = inputs.pooled.then(|| PoolShape {
            // Each global cache's keys and values in its own format: a mixed layout has
            // FP8 and NVFP4 caches.
            target_rows: c
                .cache_formats(c.loop_config.num_passes, inputs.kv)
                .into_iter()
                .flatten()
                .flat_map(|f| [f.row_bytes(kv_dims); 2])
                .collect(),
            draft_rows: match &draft {
                Some((d, approx)) => {
                    vec![kv_dims * if approx.fp8_kv { 1 } else { 2 }; 2 * d.global_layers]
                }
                None => Vec::new(),
            },
            // Every active stream's rings plus one prefix cache entry's copies of them. A
            // ring whose buffers are not whole pages packs its keys and values together, so
            // an FP8 ring of two 1 MiB buffers takes one page.
            rings: (streams + 1)
                * sliding_visits
                * (2 * c.sliding_window_size * ring_row_bytes).next_multiple_of(KV_PAGE_BYTES),
        });
        // The batched step's chunk buffers for one row per stream, its state tables,
        // positions and tokens, and its rows' taps for the draft. With a DFlash 2 drafter,
        // also count batched verify scratch/backups and per-slot DFlash 2 rings + scratch.
        let batch_spec = if streams > 1
            && let Some((d, approx)) = &draft
            && d.kind == DraftKind::Dflash2
        {
            let spec_seqs = streams.min(32);
            let spec_rows = (spec_seqs * VERIFY_ROWS).min(256);
            let draft_kv_bytes = if approx.fp8_kv { 1 } else { 2 };
            let extra_draft_rings =
                (streams - 1) * d.layers * d.ring_slots * 2 * kv_dims * draft_kv_bytes;
            let verify_bufs = invariant_rows_scratch(c, spec_rows, inputs.sms)
                + spec_rows * taps * hidden * 2
                + visits.len() * spec_seqs * VERIFY_ROWS * (2 * kv_dims + 2 * hidden) * 2
                + sliding_visits * spec_seqs * VERIFY_ROWS * 2 * ring_row_bytes;
            // Batched DFlash 2 activations, split-KV partials, shortlist/vocab logits, and
            // pre-allocated MXFP8 dense GEMM scratch (`BatchDflash2::dense`, ~2.3 MiB).
            let draft_scratch = 224 << 20;
            extra_draft_rings + verify_bufs + draft_scratch
        } else {
            0
        };
        let batch = if streams > 1 {
            invariant_rows_scratch(c, streams, inputs.sms)
                + visits.len() * streams * 6 * 8
                + streams * 8
                + streams * taps * hidden * 2
                + batch_spec
        } else {
            0
        };
        let prefix = match &draft {
            Some((d, _)) => inputs.prefix_entries * (fixed_rings + draft_ring_bytes(c, d)),
            _ => 0,
        };
        // Prefill holds the rows of the images that overlap its chunk: the chunk's rows, and
        // at most one image on each side that runs past it. `vision_bytes` counts one
        // image's rows, its output.
        let vision = inputs.vision.map_or(0, |v| {
            let tokens = v.max_image_tokens;
            vision_bytes(v.format, tokens * v.patches_per_token)
                + (inputs.prefill_chunk + tokens) * hidden * 2
        });
        let fixed = Plan {
            capacity: 0,
            pooled: inputs.pooled,
            streams,
            weights: target_weight_bytes(inputs.checkpoint, inputs.weights),
            fp8_copies: match inputs.weights {
                WeightFormat::Bf16 => 0,
                WeightFormat::Fp8 => {
                    fp8_copy_bytes(c, inputs.output)
                        + if inputs.prefill_dense_fp8 && inputs.output != OutputProjection::Mxfp8 {
                            mxfp8_output_bytes(c)
                        } else {
                            0
                        }
                }
            },
            kv: 0,
            rings,
            batch,
            prefill,
            verify,
            draft: draft_bytes,
            draft_kv: if inputs.pooled {
                0
            } else {
                (DRAFT_STEPS + 1) * draft_kv_per_position
            },
            prefix,
            vision,
            runtime: RUNTIME_BYTES
                + (streams - 1) * STREAM_RUNTIME_BYTES
                + if inputs.pooled { POOL_RUNTIME_BYTES } else { 0 },
        };
        Ok(Self {
            fixed,
            kv_per_position,
            draft_kv_per_position,
            max_positions: c.model_max_length,
            pool,
        })
    }

    /// The plan at `capacity` positions.
    #[must_use]
    pub fn plan(&self, capacity: usize) -> Plan {
        if let Some(pool) = &self.pool {
            return Plan {
                capacity,
                kv: self.pool_bytes(capacity, pool) + pool.rings,
                ..self.fixed
            };
        }
        Plan {
            capacity,
            kv: capacity * self.kv_per_position,
            draft_kv: self.fixed.draft_kv + capacity * self.draft_kv_per_position,
            ..self.fixed
        }
    }

    /// The global caches' part of the pool: `capacity` positions of target caches, in
    /// whole chunks, and the draft's caches for one sequence (at most
    /// [`Self::max_positions`]). Sequences map whole chunks, so several together hold up to
    /// a chunk each less than `capacity`.
    fn pool_bytes(&self, capacity: usize, pool: &PoolShape) -> usize {
        let draft_capacity = capacity.min(self.max_positions);
        PoolShape::pages_bytes(&pool.target_rows, capacity)
            + PoolShape::pages_bytes(&pool.draft_rows, draft_capacity + DRAFT_STEPS + 1)
    }

    /// Pages of [`KV_PAGE_BYTES`] in the plan's pool, 0 without one.
    #[must_use]
    pub fn pool_pages(&self, capacity: usize) -> usize {
        if self.pool.is_some() {
            self.plan(capacity).kv / KV_PAGE_BYTES
        } else {
            0
        }
    }

    /// Bytes each position adds: the target's and the draft's global KV caches.
    #[must_use]
    pub fn bytes_per_position(&self) -> usize {
        match &self.pool {
            Some(pool) => pool.target_rows.iter().chain(&pool.draft_rows).sum(),
            None => self.kv_per_position + self.draft_kv_per_position,
        }
    }

    /// The model's longest single sequence (`model_max_length`).
    #[must_use]
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// The largest total capacity this configuration can use: for a pooled multi-stream
    /// cache, up to `streams * max_positions` (each individual sequence stays capped at
    /// [`Self::max_positions`]); otherwise [`Self::max_positions`].
    #[must_use]
    pub fn max_pool_positions(&self) -> usize {
        if self.pool.is_some() {
            self.max_positions
                .saturating_mul(self.fixed.streams.max(1))
        } else {
            self.max_positions
        }
    }

    /// The largest capacity whose plan fits `budget` bytes: a multiple of
    /// [`CAPACITY_STEP`], at most [`Self::max_pool_positions`], or `None` if not even one
    /// step fits.
    #[must_use]
    pub fn max_capacity(&self, budget: usize) -> Option<usize> {
        let max_steps = self.max_pool_positions() / CAPACITY_STEP;
        if max_steps == 0 || self.plan(CAPACITY_STEP).total() > budget {
            return None;
        }
        let (mut lo, mut hi) = (1usize, max_steps);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if self.plan(mid * CAPACITY_STEP).total() <= budget {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Some(lo * CAPACITY_STEP)
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

/// Bytes of an MXFP8 copy of the target, `rows` padded, with one UE8M0 scale per 32 values.
fn mxfp8_matrix(rows: usize, k: usize) -> usize {
    let padded = rows.next_multiple_of(FP8_ROW_PAD);
    padded * k + padded * (k / 32)
}

/// Bytes of a draft's FP8 rows: values and an F32 scale per row.
fn fp8_rows(rows: usize, k: usize) -> usize {
    rows * k + 4 * rows
}

/// `Weights::quantize_fp8`: every layer's attention input projections, its output
/// projection with [`OutputProjection::Fp8`], the looped layers' message gates and the LM
/// head.
fn fp8_copy_bytes(c: &TextConfig, output: OutputProjection) -> usize {
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
                + match output {
                    OutputProjection::Fp8 => fp8_matrix(hidden, q_dims),
                    OutputProjection::Mxfp8 => mxfp8_matrix(hidden, q_dims),
                    OutputProjection::Bf16 => 0,
                }
                + gates
        })
        .sum();
    layers + fp8_matrix(c.vocab_size, hidden)
}

/// Device bytes of the per-layer MXFP8 block-32 copies of the attention output projection
/// (`W_o`), allocated when `--output-projection mxfp8` or `--prefill-dense fp8` is enabled.
#[must_use]
pub fn mxfp8_output_bytes(c: &TextConfig) -> usize {
    let hidden = c.hidden_size;
    let q_dims = c.num_attention_heads * c.head_dim;
    c.num_hidden_layers * mxfp8_matrix(hidden, q_dims)
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

/// Core per-chunk buffers of `PrefillScratch::new` without the LM head outputs.
fn rows_scratch_core(c: &TextConfig, rows: usize, sms: usize) -> usize {
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
    let quantized_in = rows * hidden;
    let quantized = pairs * width.max(hidden / top_k);
    let per_rows = rows * (2 * bf16_per_row + 4 * f32_per_row + 4)
        + 4 * (1 + pairs + rows) // expert counters
        + 4 * pairs // perm
        + 16 * pairs.min(pairs / 64 + experts) // tiles
        + quantized_in + quantized_in / 16 + 4 * rows
        + quantized + quantized / 16 + 4 * pairs
        + 4 * (rows * (heads / c.num_key_value_heads)).div_ceil(128) * c.num_key_value_heads;
    let fixed = 4 * 2 * sms * PREFILL_SLOT + 4 * ROWS_ATTENTION_ROWS * (c.head_dim + 2);
    per_rows + fixed
}

/// `PrefillScratch::new` for a non-invariant prefill chunk of `rows` rows, including
/// `HEAD_ROWS` of BF16 `head_logits` and F32 `head_capped` for `PrefillLogits::All`/`Every`.
fn rows_scratch(c: &TextConfig, rows: usize, sms: usize) -> usize {
    rows_scratch_core(c, rows, sms) + HEAD_ROWS * (c.vocab_size * 2 + c.unpadded_vocab_size * 4)
}

/// `PrefillScratch::new` for an invariant chunk (`verify`, `batch`, `batch_verify`):
/// never allocates F32 `head_capped` (sampling reads BF16 `head_logits` with inline
/// softcap; greedy runs fused tile-argmax into tiny partial buffers).
fn invariant_rows_scratch(c: &TextConfig, rows: usize, sms: usize) -> usize {
    rows_scratch_core(c, rows, sms) + invariant_scratch(c, rows)
}

/// The MXFP8 inputs of a prefill chunk's FP8 dense products (`GateRows::dense`): four
/// inputs of `2 * hidden` values per row (slots 0..3 for attention message gates and slot 3
/// for the MLP message gate) and a scale byte per 32 values. Invariant chunks allocate none.
fn dense_scratch(c: &TextConfig, rows: usize) -> usize {
    let values = rows * 4 * 2 * c.hidden_size;
    values + values / 32
}

/// What `PrefillScratch::new` adds for an invariant chunk (verify / batch / batch_verify):
/// every looped layer's message weights and normalized messages, an output for each of the
/// 3 gates one launch runs (`misc::GATE_SLOTS`), BF16 `head_logits` (worst case when
/// sampling is enabled; greedy uses even smaller tile-argmax partials), and wide-row sync
/// headroom.
fn invariant_scratch(c: &TextConfig, rows: usize) -> usize {
    let hidden = c.hidden_size;
    let looped = c.loop_config.core_end - c.loop_config.core_start;
    // The multi-row attention's partials and counters past the 8 rows `rows_scratch_core`
    // counts, for a batched step's rows.
    let attention = rows.saturating_sub(VERIFY_ROWS)
        * (ROWS_ATTENTION_ROWS / VERIFY_ROWS * (c.head_dim + 2) * 4
            + 2 * c.num_key_value_heads * 4);
    // BF16 `head_logits` when sampling is enabled (0 bytes of F32 `head_capped`!).
    let head_logits = rows * c.vocab_size * 2;
    // Above 8 rows, `ExpertsRows::sync_len` adds grouped routing metadata (`ExpertRowsMeta`).
    let extra_sync = if rows > 8 {
        let base = 1 + rows * (c.num_experts_per_tok + 1);
        let pairs = rows * c.num_experts_per_tok;
        let groups = pairs.min(c.n_routed_experts) + rows;
        let wide = (2 + groups).next_multiple_of(2) + 2056 / 4 + pairs.div_ceil(2);
        wide.saturating_sub(base) * 4
    } else {
        0
    };
    // At 48+ rows, invariant chunks also allocate MXFP8 scratch for opt-in FP8 dense GEMMs.
    let dense = if rows >= BATCH_FP8_DENSE_MIN_ROWS {
        let values = rows * (3 * hidden).max(c.num_attention_heads * c.head_dim);
        values + values / 32
    } else {
        0
    };
    rows * ((looped - 1) * 4 * FUSED_MAX_CANDIDATES * 4 + 4 * looped * hidden * 2 + 2 * hidden * 4)
        + attention
        + head_logits
        + extra_sync
        + dense
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

/// Bytes of one `Nvfp4Matrix` (`[padded_rows, k / 2 + k / 16 + 4]`).
fn nvfp4_matrix(rows: usize, k: usize) -> usize {
    let padded = rows.next_multiple_of(FP8_ROW_PAD);
    padded * (k / 2 + k / 16 + 4)
}

/// The draft's bytes that do not grow with the capacity: its weights less the BF16
/// projections that FP8/NVFP4 copies replace, the bundled draft's fold, the copies its
/// approximations make, its rings and its larger buffers.
fn draft_fixed_bytes(c: &TextConfig, d: &DraftShape, approx: DraftApprox) -> usize {
    let hidden = c.hidden_size;
    let kv_dims = c.num_key_value_heads * c.head_dim;
    let vocab = c.unpadded_vocab_size;
    let padded = c.vocab_size;
    let mut bytes = d.weights;
    if approx.fp8_backbone || approx.nvfp4_backbone {
        // The BF16 projections only the block reads, freed for their quantized copies.
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
    if approx.nvfp4_backbone {
        let per_layer = nvfp4_matrix(d.q_width, hidden)
            + 2 * nvfp4_matrix(kv_dims, hidden)
            + nvfp4_matrix(hidden, d.q_width)
            + 2 * nvfp4_matrix(d.mlp_width, hidden)
            + nvfp4_matrix(hidden, d.mlp_width);
        let context = match d.kind {
            DraftKind::Bundled => 2 * nvfp4_matrix(kv_dims, hidden),
            DraftKind::Dflash | DraftKind::Dflash2 => 0,
        };
        bytes += d.layers * (per_layer + context);
        if d.kind != DraftKind::Bundled {
            bytes += nvfp4_matrix(hidden, d.taps * hidden);
        }
    } else if approx.fp8_backbone {
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
        bytes += if approx.nvfp4_head {
            nvfp4_matrix(rows, hidden)
        } else if approx.fp8_head {
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
        if approx.nvfp4_head {
            bytes += nvfp4_matrix(vocab, hidden);
        } else if approx.fp8_head {
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

/// The vision tower's width, layers, MLP width, values per patch, position-table rows,
/// projector width and output width, which `siphon_cuda::vision` checks at load.
pub const VISION_DIM: usize = 1024;
pub const VISION_LAYERS: usize = 23;
pub const VISION_MLP: usize = 4096;
pub const VISION_PATCH_VALUES: usize = 1200;
pub const VISION_POSITIONS: usize = 32 * 32;
pub const VISION_PROJECTION: usize = 4096;
pub const VISION_OUT: usize = 2304;

/// Device bytes of the vision tower with its linear layers in `format`: the weights, plus
/// the buffers of one image of up to `max_patches` 20-pixel patches and its output
/// embeddings. The buffers live from the first encode (or `VisionTower::reserve`) until
/// `VisionTower::release`; the output lives as long as the caller keeps it.
#[must_use]
pub fn vision_bytes(format: VisionFormat, max_patches: usize) -> usize {
    vision_weight_bytes(format) + vision_buffer_bytes(format, max_patches)
}

/// Device bytes of the vision tower's weights in `format`.
#[must_use]
pub fn vision_weight_bytes(format: VisionFormat) -> usize {
    let (d, mlp) = (VISION_DIM, VISION_MLP);
    let matrix = |rows: usize, k: usize| match format {
        VisionFormat::Bf16 => 2 * rows * k,
    };
    let layer = 4 * matrix(d, d) + matrix(mlp, d) + matrix(d, mlp) + 2 * (11 * d + mlp);
    let embeddings = 2 * (d * VISION_PATCH_VALUES + VISION_POSITIONS * d + 4 * d);
    let projector = matrix(VISION_PROJECTION, 4 * d) + matrix(VISION_OUT, VISION_PROJECTION);
    embeddings + VISION_LAYERS * layer + projector
}

/// Device bytes of the vision tower's buffers for images of up to `max_patches` patches,
/// the output embeddings of one image included.
#[must_use]
pub fn vision_buffer_bytes(format: VisionFormat, max_patches: usize) -> usize {
    if max_patches == 0 {
        return 0;
    }
    let rows = max_patches + 1;
    let d = VISION_DIM;
    // The F32 residual stream, the BF16 product inputs and outputs, and the wide scratch
    // that holds q, k and v, the MLP's hidden rows and, first, the patches.
    let rows_bytes = match format {
        VisionFormat::Bf16 => rows * d * (4 + 2) + rows * VISION_MLP * 2,
    };
    let output = max_patches / 4 * VISION_OUT * 2;
    rows_bytes + output
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
        assert_eq!(global * 2 * KvFormat::Bf16.row_bytes(kv_dims), 32 << 10);
        assert_eq!(global * 2 * KvFormat::Fp8.row_bytes(kv_dims), 16 << 10);
        // NVFP4: 576 bytes per buffer and position, 9 KiB per position over 8 caches.
        assert_eq!(KvFormat::Nvfp4.row_bytes(kv_dims), 576);
        assert_eq!(KvFormat::Nvfp4All.row_bytes(kv_dims), 576);
        assert_eq!(KvFormat::Nvfp4.sliding_format(), KvFormat::Fp8);
        assert_eq!(KvFormat::Nvfp4All.sliding_format(), KvFormat::Nvfp4);
        let passes = c.loop_config.num_passes;
        assert_eq!(c.global_cache_bytes(passes, KvFormat::Fp8), 16 << 10);
        assert_eq!(c.global_cache_bytes(passes, KvFormat::Nvfp4), 9 << 10);
        assert_eq!(c.global_cache_bytes(passes, KvFormat::Nvfp4All), 9 << 10);
        // Mixed: visits 5 and 11 in FP8, 2 KiB each, the other 6 in NVFP4.
        assert_eq!(
            c.global_cache_bytes(passes, KvFormat::Mixed),
            2 * 2048 + 6 * 1152
        );
        let formats = c.cache_formats(passes, KvFormat::Mixed);
        assert_eq!(formats[5], Some(KvFormat::Fp8));
        assert_eq!(formats[11], Some(KvFormat::Fp8));
        assert_eq!(formats[17], Some(KvFormat::Nvfp4));
        assert_eq!(formats[47], Some(KvFormat::Nvfp4));
        let all_formats = c.cache_formats(passes, KvFormat::Nvfp4All);
        assert_eq!(all_formats[5], Some(KvFormat::Nvfp4));
        assert_eq!(all_formats[47], Some(KvFormat::Nvfp4));
    }

    #[test]
    fn fp8_copies_take_about_1_14_gb_or_0_97_without_the_output_projection() {
        let bytes = fp8_copy_bytes(&config(), OutputProjection::Fp8);
        assert!((1.13e9..1.15e9).contains(&(bytes as f64)), "{bytes}");
        // Without the copies of the 24 output projections of 2304 x 3072.
        let bf16 = fp8_copy_bytes(&config(), OutputProjection::Bf16);
        assert_eq!(bytes - bf16, 24 * (2304 * 3072 + 4 * 2304));
        let mxfp8 = fp8_copy_bytes(&config(), OutputProjection::Mxfp8);
        assert_eq!(mxfp8 - bf16, mxfp8_output_bytes(&config()));
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
            pool: None,
        };
        let budget = 30 << 30;
        let capacity = planner.max_capacity(budget).expect("something fits");
        assert_eq!(capacity % CAPACITY_STEP, 0);
        assert!(planner.plan(capacity).total() <= budget);
        assert!(planner.plan(capacity + CAPACITY_STEP).total() > budget);
        let planner = Planner {
            pool: Some(PoolShape {
                target_rows: vec![1024; 32],
                draft_rows: vec![1024; 4],
                rings: 320 << 20,
            }),
            ..planner
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
    fn vision_adds_the_tower_and_the_rows_of_one_chunk_and_two_images() {
        assert_eq!(max_image_tokens(40, 1920 * 1080), 37 * 37);
        assert_eq!(max_image_tokens(40, 1600), 4);
        let dir = checkpoint("vision", &[("model.llm.norm.weight", Dtype::BF16, &[25])]);
        let checkpoint = Checkpoint::open(&dir).expect("opening");
        let c = config();
        let plan = |vision| {
            Planner::new(&PlanInputs {
                config: &c,
                checkpoint: &checkpoint,
                weights: WeightFormat::Fp8,
                output: OutputProjection::default(),
                prefill_dense_fp8: false,
                kv: KvFormat::Fp8,
                draft: DraftChoice::None,
                prefill_chunk: 2048,
                streams: 1,
                pooled: false,
                prefix_entries: 0,
                sms: 170,
                vision,
            })
            .expect("a plan")
            .plan(4096)
        };
        let off = plan(None);
        let on = plan(Some(VisionInputs {
            format: VisionFormat::Bf16,
            max_image_tokens: 100,
            patches_per_token: 4,
        }));
        assert_eq!(off.vision, 0);
        assert_eq!(on.total() - off.total(), on.vision);
        assert_eq!(
            on.vision,
            vision_bytes(VisionFormat::Bf16, 400) + (2048 + 100) * 2304 * 2
        );
        std::fs::remove_dir_all(dir).expect("cleaning up");
    }

    #[test]
    fn pool_pages_follow_each_caches_format() {
        let dir = checkpoint("pool", &[("model.llm.norm.weight", Dtype::BF16, &[25])]);
        let checkpoint = Checkpoint::open(&dir).expect("opening");
        let c = config();
        let pool = |kv| {
            Planner::new(&PlanInputs {
                config: &c,
                checkpoint: &checkpoint,
                weights: WeightFormat::Fp8,
                output: OutputProjection::default(),
                prefill_dense_fp8: false,
                kv,
                draft: DraftChoice::None,
                prefill_chunk: 2048,
                streams: 1,
                pooled: true,
                prefix_entries: 0,
                sms: 170,
                vision: None,
            })
            .expect("a plan")
        };
        let mib = 1 << 20;
        // 8 global caches of keys and values, then one sequence's rings and its entry's
        // copies of them: 40 FP8 rings of 1 MiB keys and 1 MiB values, which share a
        // page, twice, and BF16 rings of a page per buffer.
        let rings = pool(KvFormat::Fp8).plan(0).kv;
        assert_eq!(rings, 160 * mib);
        assert_eq!(pool(KvFormat::Bf16).plan(0).kv, 320 * mib);
        // FP8: 1,024 bytes per position, 2 MiB per buffer and chunk of 2,048 positions.
        assert_eq!(pool(KvFormat::Fp8).plan(4096).kv - rings, 16 * 4 * mib);
        // NVFP4: 576 bytes per position and buffer; 4,096 positions take 2.25 MiB, 2 pages
        // of each buffer, 6,144 take 3.375 MiB, 2 pages, not 3.
        assert_eq!(pool(KvFormat::Nvfp4).plan(4096).kv - rings, 16 * 4 * mib);
        assert_eq!(pool(KvFormat::Nvfp4).plan(6144).kv - rings, 16 * 4 * mib);
        // Mixed: 4 FP8 buffers of 3 chunks, 12 NVFP4 buffers in 2 pages each.
        assert_eq!(
            pool(KvFormat::Mixed).plan(6144).kv - rings,
            (4 * 6 + 12 * 4) * mib
        );
        // NVFP4-all: 40 packed NVFP4 rings of 1.125 MiB each also share one 2 MiB page each.
        assert_eq!(pool(KvFormat::Nvfp4All).plan(0).kv, rings);
        assert_eq!(pool(KvFormat::Nvfp4All).plan(4096).kv - rings, 16 * 4 * mib);
        assert_eq!(pool(KvFormat::Nvfp4All).plan(6144).kv - rings, 16 * 4 * mib);
        assert_eq!(pool(KvFormat::Nvfp4All).bytes_per_position(), 9 << 10);
        assert_eq!(pool(KvFormat::Nvfp4).bytes_per_position(), 9 << 10);
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

    #[test]
    fn invariant_scratch_omits_f32_head_capped() {
        let c = config();
        // Verify (8 rows) + batch (32 rows) + batch_verify (256 rows) no longer allocate
        // F32 head_capped (which previously took (32 + 32 + 256) * unpadded_vocab * 4 ~ 256 MB).
        let old_head = |rows: usize| {
            rows.max(HEAD_ROWS) * (c.vocab_size * 2 + c.unpadded_vocab_size * 4)
        };
        let new_head = |rows: usize| rows * c.vocab_size * 2;
        let saved: usize = [8usize, 32, 256]
            .into_iter()
            .map(|r| old_head(r) - new_head(r))
            .sum();
        assert!(saved > 250_000_000, "expected >250 MB saved, got {saved}");
        // Verify invariant_rows_scratch reflects this exact difference vs rows_scratch_core.
        assert_eq!(
            invariant_rows_scratch(&c, 256, 170) - rows_scratch_core(&c, 256, 170),
            invariant_scratch(&c, 256)
        );
    }

    #[test]
    fn batch_seq_bucketing_covers_all_sizes_without_misses() {
        assert_eq!(batch_seq_buckets(0), Vec::<usize>::new());
        assert_eq!(batch_spec_graph_shapes(0), Vec::<(usize, usize)>::new());

        for max_seqs in [1, 2, 3, 4, 5, 8, 9, 10, 12, 16, 17, 20, 24, 32] {
            let buckets = batch_seq_buckets(max_seqs);
            assert!(buckets.windows(2).all(|w| w[0] < w[1]), "{buckets:?}");
            assert_eq!(buckets.first().copied(), Some(1));
            assert_eq!(buckets.last().copied(), Some(max_seqs));
            // Every live seqs in 1..=max_seqs maps into the pre-captured bucket set.
            for seqs in 1..=max_seqs {
                let padded = batch_seq_bucket(seqs, max_seqs);
                assert!(padded >= seqs && padded <= max_seqs);
                // Step 1 for 1..=8 (0 dummy), step 2 for 9..=32 (at most 1 dummy slot).
                assert!(
                    padded - seqs <= usize::from(seqs > 8),
                    "seqs={seqs} padded to {padded}"
                );
                assert!(
                    buckets.contains(&padded),
                    "seqs={seqs} (max={max_seqs}) mapped to {padded}, not in {buckets:?}"
                );
            }
            let shapes = batch_spec_graph_shapes(max_seqs);
            for seqs in 1..=max_seqs {
                let padded = batch_seq_bucket(seqs, max_seqs);
                for w in BATCH_SPEC_WIDTHS {
                    if padded * w <= BATCH_VERIFY_MAX_ROWS {
                        assert!(
                            shapes.contains(&(padded, w)),
                            "missing shape ({padded}, {w}) for seqs={seqs} max={max_seqs}"
                        );
                    }
                }
            }
        }
        // C=16: 12 tight buckets x 4 widths = 48 verify shapes for all seqs 1..=16.
        assert_eq!(
            batch_seq_buckets(16),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]
        );
        assert_eq!(batch_spec_graph_shapes(16).len(), 48);
        // C=32: 20 tight buckets x 4 widths = 80 verify shapes for all seqs 1..=32.
        assert_eq!(
            batch_seq_buckets(32),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32
            ]
        );
        assert_eq!(batch_spec_graph_shapes(32).len(), 80);
    }

    #[test]
    fn ring_slab_coalesces_all_sliding_rings_without_extra_pages() {
        let mib = 1 << 20;
        // 40 FP8 sliding rings of 1 MiB K + 1 MiB V = 80 MiB = 40 pages in one slab.
        let fp8 = RingSlabLayout::new(40, mib, KV_PAGE_BYTES);
        assert_eq!(fp8.ring_stride, 2 * mib);
        assert_eq!(fp8.total_bytes, 80 * mib);
        assert_eq!(fp8.pages, 40);
        assert_eq!(fp8.ring_offsets(0), (0, mib));
        assert_eq!(fp8.ring_offsets(39), (78 * mib, 79 * mib));

        // 40 BF16 sliding rings of 2 MiB K + 2 MiB V = 160 MiB = 80 pages in one slab.
        let bf16 = RingSlabLayout::new(40, 2 * mib, KV_PAGE_BYTES);
        assert_eq!(bf16.ring_stride, 4 * mib);
        assert_eq!(bf16.total_bytes, 160 * mib);
        assert_eq!(bf16.pages, 80);
        assert_eq!(bf16.ring_offsets(1), (4 * mib, 6 * mib));

        // 40 NVFP4 sliding rings of 576 KiB K + 576 KiB V = 45 MiB = 23 pages in one slab.
        let nvfp4_buf = 1024 * KvFormat::Nvfp4All.sliding_format().row_bytes(1024);
        let nvfp4 = RingSlabLayout::new(40, nvfp4_buf, KV_PAGE_BYTES);
        assert_eq!(nvfp4.ring_stride, 2 * nvfp4_buf);
        assert_eq!(nvfp4.pages, 23);

        // Empty slab takes zero bytes and zero pages.
        let empty = RingSlabLayout::new(0, mib, KV_PAGE_BYTES);
        assert_eq!((empty.total_bytes, empty.pages), (0, 0));
    }

    #[test]
    fn shortlist_capacity_is_stable_and_capped() {
        let vocab = 200_000;
        let ranked = SHORTLIST_ROWS + (vocab - FIRST_SPECIAL);
        let cap = shortlist_capacity(ranked, vocab);
        assert_eq!(cap, ranked + SHORTLIST_CONTEXT_ROWS);
        assert!(cap < vocab);
        // Small vocabularies clamp cleanly to `vocab`.
        assert_eq!(shortlist_capacity(100, 128), 128);
        assert_eq!(shortlist_capacity(128, 128), 128);
    }

    #[test]
    fn mxfp8_output_and_batch_dense_scratch_match_shapes() {
        let c = config();
        // 24 layers x 2304 x (3072 + 96) = 175,177,728 bytes (~167 MiB).
        assert_eq!(mxfp8_output_bytes(&c), 24 * 2304 * (3072 + 96));
        // Below BATCH_FP8_DENSE_MIN_ROWS (48), no MXFP8 dense scratch is added.
        let inv47 = invariant_scratch(&c, 47);
        let inv48 = invariant_scratch(&c, 48);
        let dense48 = 48 * (3 * c.hidden_size) + (48 * 3 * c.hidden_size) / 32;
        assert!(inv48 - inv47 > dense48);
    }

    #[test]
    fn pooled_multi_stream_capacity_scales_past_model_max_length() {
        let dir = checkpoint("pool-multi", &[("model.llm.norm.weight", Dtype::BF16, &[25])]);
        let checkpoint = Checkpoint::open(&dir).expect("opening");
        let c = config();
        let make = |streams, pooled, kv, draft| {
            Planner::new(&PlanInputs {
                config: &c,
                checkpoint: &checkpoint,
                weights: WeightFormat::Fp8,
                output: OutputProjection::default(),
                prefill_dense_fp8: false,
                kv,
                draft,
                prefill_chunk: 2048,
                streams,
                pooled,
                prefix_entries: 0,
                sms: 188,
                vision: None,
            })
            .expect("a plan")
        };

        let budget_96g = 94usize << 30;
        let single_pooled = make(1, true, KvFormat::Fp8, DraftChoice::None);
        assert_eq!(single_pooled.max_positions(), 1 << 20);
        assert_eq!(single_pooled.max_pool_positions(), 1 << 20);
        assert_eq!(single_pooled.max_capacity(budget_96g), Some(1 << 20));

        // 2 pooled streams can use up to 2 * model_max_length, fitting in 94 GiB FP8.
        let two_pooled = make(2, true, KvFormat::Fp8, DraftChoice::None);
        assert_eq!(two_pooled.max_positions(), 1 << 20);
        assert_eq!(two_pooled.max_pool_positions(), 2 << 20);
        assert_eq!(two_pooled.max_capacity(budget_96g), Some(2 << 20));

        // 32 pooled streams scale to fill the 94 GiB budget well past 1M positions.
        let many_pooled = make(32, true, KvFormat::Fp8, DraftChoice::None);
        assert_eq!(many_pooled.max_pool_positions(), 32 << 20);
        let cap32 = many_pooled.max_capacity(budget_96g).expect("fits");
        assert!(cap32 > (4 << 20), "expected >4M FP8 positions, got {cap32}");
        assert_eq!(cap32 % CAPACITY_STEP, 0);
        assert!(many_pooled.plan(cap32).total() <= budget_96g);
        assert!(many_pooled.plan(cap32 + CAPACITY_STEP).total() > budget_96g);

        // NVFP4 pooled streams reach >8M positions on a 94 GiB budget.
        let nvfp4_pooled = make(32, true, KvFormat::Nvfp4, DraftChoice::None);
        let cap_nvfp4 = nvfp4_pooled.max_capacity(budget_96g).expect("fits");
        assert!(
            cap_nvfp4 > (8 << 20),
            "expected >8M NVFP4 positions, got {cap_nvfp4}"
        );
        assert!(nvfp4_pooled.plan(cap_nvfp4).total() <= budget_96g);
        assert!(nvfp4_pooled.plan(cap_nvfp4 + CAPACITY_STEP).total() > budget_96g);

        // Unpooled multi-stream stays capped at model_max_length per stream.
        let unpooled = make(4, false, KvFormat::Fp8, DraftChoice::None);
        assert_eq!(unpooled.max_pool_positions(), 1 << 20);
        assert_eq!(unpooled.max_capacity(usize::MAX / 2), Some(1 << 20));

        // Draft global KV in a pooled multi-stream plan caps at max_positions + DRAFT_STEPS + 1.
        let draft_planner = Planner {
            fixed: Plan {
                streams: 4,
                pooled: true,
                weights: 16 << 30,
                runtime: RUNTIME_BYTES,
                ..Plan::default()
            },
            kv_per_position: 16 << 10,
            draft_kv_per_position: 2 << 10,
            max_positions: 1 << 20,
            pool: Some(PoolShape {
                target_rows: vec![1024; 16],
                draft_rows: vec![1024; 2],
                rings: 320 << 20,
            }),
        };
        let at_1m = draft_planner.plan(1 << 20).kv;
        let at_2m = draft_planner.plan(2 << 20).kv;
        // Going from 1M to 2M adds only 1M target positions (16 GiB), not draft KV.
        assert_eq!(at_2m - at_1m, 16 << 30);

        std::fs::remove_dir_all(dir).expect("cleaning up");
    }

    #[test]
    fn decode_premap_target_batches_chunks_and_caps_at_final_position() {
        // Prompt 1900 + max_tokens 512 => final 2421; 2 chunks ahead from 1900 caps at 4096.
        assert_eq!(decode_premap_target(1900, 2421, POOL_CHUNK, 2), 4096);
        // Short request prompt 200 + max_tokens 256 => final 465; maps just 1 chunk (2048).
        assert_eq!(decode_premap_target(200, 465, POOL_CHUNK, 2), 2048);
        // Long request prompt 1000 + max_tokens 16384 => pre-maps 2 chunks ahead (6144).
        assert_eq!(decode_premap_target(1000, 17384, POOL_CHUNK, 2), 6144);
    }
}
