//! The DSpark draft bundled with the checkpoint: a 3-layer dense transformer that drafts
//! one block of 8 rows in one pass, and a recurrent Markov head that turns the 7 mask rows
//! into 7 greedy draft tokens.
//!
//! A round anchored at position `P` runs the block at positions `P..P + 8`: the anchor
//! token's normalized target embedding, then 7 copies of `mask_embed`. The block attends
//! bidirectionally to itself and to the draft cache below `P`, the last 1024 positions in
//! layers 0 and 2 and every position in layer 1. [`Draft::commit`] fills the cache below
//! `P` from the target's four feature taps of each committed position: each layer mixes
//! them, normalizes them and projects them through `k_proj_ctx` and `v_proj_ctx`. A round
//! writes its own block's keys and values at `P..P + 8`, which the next commit or round
//! overwrites, so the cache needs no rollback.
//!
//! A round can also take every input from device memory, for a CUDA graph that replays
//! right after the target's verify pass: [`Draft::round_inputs`] holds the taps of up to 8
//! positions to commit, the first of those positions, their count and the anchor, and
//! [`Draft::run_round`] or [`Draft::replay_round`] commits them and runs the block right
//! after them. The host never learns the position. The round runs the Markov head and the
//! base logits only for the drafts that will be verified.
//!
//! The numerics follow vLLM's draft, which `reference/dspark.py` reproduces in MLX: BF16
//! weights and activations, F32 accumulation, and rounding where vLLM rounds (see
//! `kernels/draft.cu`).

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaGraph, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
};
use half::bf16;
use inkling::TextConfig;

use crate::{
    DeviceIn,
    ops::{
        draft::{
            ArgmaxPartials, AttentionKeys, AttentionScratch, BLOCK, CONTROL, DraftKernels,
            FoldRows, Fp8Product, Fp8Rows, KV_WIDTH, KvCache, LogitBounds,
        },
        linear::{Gemm, Linear},
    },
    weights::draft::{
        DraftWeights, HEAD_DIM, HEADS, HIDDEN, LAYERS, MLP_WIDTH, RANK, SharedWeights, TAPS,
    },
};

/// Draft tokens per round: one per mask row.
pub const STEPS: usize = BLOCK - 1;
/// Positions before the block that the local layers (0 and 2) see.
pub const WINDOW: usize = 1024;
/// Slots of the local layers' rings: the window and the block.
const LOCAL_SLOTS: usize = WINDOW + BLOCK;
/// RoPE base of the draft (`rope_theta` in `mtp_config`).
const ROPE_THETA: f64 = 1e6;
/// RMSNorm epsilon of the draft (`rms_norm_eps` in `mtp_config`).
const EPS: f32 = 1e-6;
/// Target positions one [`Draft::commit`] chunk projects at once.
const COMMIT_ROWS: usize = 512;

/// Ranked tokens a shortlist head keeps, as Ingine's drafter.
pub const SHORTLIST_ROWS: usize = 32768;
/// Context tokens a shortlist head adds at most.
pub const CONTEXT_ROWS: usize = 16384;
/// The first special token id; a shortlist head keeps every token from here on.
const FIRST_SPECIAL: usize = 199_990;
/// A shortlist head's base logits cover its rows in use rounded up to this.
const HEAD_BUCKET: usize = 1024;

/// Approximations of the draft. Each changes the draft's tokens and so how often the
/// target accepts them, never the target's output. The default is the exact draft.
#[derive(Clone, Debug, Default)]
pub struct DraftOptions {
    /// FP8 copies of the block's projections and of the round's context projections.
    pub fp8_backbone: bool,
    /// Base logits from an FP8 copy of the LM head (or of its shortlist rows).
    pub fp8_head: bool,
    /// An FP8 copy of the fold weight (or of its shortlist rows).
    pub fp8_fold: bool,
    /// Token ids by rank, as Ingine's `draft_shortlist.bin`: the base and Markov logits
    /// then cover only the first [`SHORTLIST_ROWS`] of them, the special tokens and up to
    /// [`CONTEXT_ROWS`] tokens passed to [`Draft::note_tokens`].
    pub shortlist: Option<Vec<u32>>,
}

/// The RoPE frequencies `1 / theta^(2i/128)` as vLLM's F32 table builds them: a correctly
/// rounded power, then a correctly rounded reciprocal.
#[must_use]
pub fn rope_inv_freq() -> Vec<f32> {
    (0..HEAD_DIM / 2)
        .map(|i| {
            #[expect(clippy::cast_precision_loss, reason = "i is below 64")]
            let exponent = (2 * i) as f64 / HEAD_DIM as f64;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "rounding the power to F32 is the intent"
            )]
            let power = ROPE_THETA.powf(exponent) as f32;
            1.0 / power
        })
        .collect()
}

/// The draft's weights, its LM-head fold, its cache and its scratch buffers.
pub struct Draft {
    kernels: DraftKernels,
    weights: DraftWeights,
    /// `lm_head · hidden_out` and its copies for the pruned argmax.
    fold: Fold,
    /// `softmax(fuse_weight)`, F32 `[4, 4]`.
    alpha: CudaSlice<f32>,
    /// RoPE frequencies `theta^(-2i/128)`, F32 `[64]`.
    inv_freq: CudaSlice<f32>,
    caches: Vec<KvCache>,
    /// Positions the global layer holds.
    capacity: usize,
    /// One past the last committed position.
    committed: usize,
    vocab: usize,
    padded_vocab: usize,
    softcap: f32,
    s: Scratch,
    inputs: RoundInputs,
    /// The captured round per draft count, and the LM head it was captured with.
    graphs: [Option<RoundGraph>; STEPS],
    /// FP8 copies of each layer's projections, with [`DraftOptions::fp8_backbone`].
    backbone: Option<Vec<Fp8Layer>>,
    /// The approximate Markov head, or `None` for the exact one.
    head: Option<Head>,
}

/// FP8 copies of one layer's projections.
struct Fp8Layer {
    q: Fp8Rows,
    k: Fp8Rows,
    v: Fp8Rows,
    o: Fp8Rows,
    gate: Fp8Rows,
    up: Fp8Rows,
    down: Fp8Rows,
    k_ctx: Fp8Rows,
    v_ctx: Fp8Rows,
}

enum Matrix {
    Bf16(CudaSlice<bf16>),
    Fp8(Fp8Rows),
}

/// An approximate Markov head: FP8 weights, a shortlist of tokens, or both.
struct Head {
    /// The base logits' weights, or `None` for the shared BF16 LM head.
    lm: Option<Matrix>,
    /// The fold rows, or `None` for the exact BF16 fold.
    fold: Option<Matrix>,
    /// The shortlist's rows of the fold bias.
    fold_bias: Option<CudaSlice<bf16>>,
    shortlist: Option<Shortlist>,
}

/// The tokens of a shortlist head: rows `0..ranked` are the ranked and special tokens in
/// ascending id order, then context tokens.
struct Shortlist {
    ranked: usize,
    used: usize,
    member: Vec<bool>,
    context: Vec<i32>,
    /// Row to token id, I32 `[ranked + CONTEXT_ROWS]`.
    map: CudaSlice<i32>,
    /// Rows in use, I32 `[1]`.
    count: CudaSlice<i32>,
    /// Whether `count` lags `used`.
    stale: bool,
}

impl Shortlist {
    fn capacity(&self) -> usize {
        self.ranked + CONTEXT_ROWS
    }

    /// Rows the base logits cover.
    fn bucket(&self) -> usize {
        self.used.next_multiple_of(HEAD_BUCKET).min(self.capacity())
    }
}

/// A round's inputs in device memory, for [`Draft::run_round`] and [`Draft::replay_round`].
pub struct RoundInputs {
    /// Taps of the positions to commit, BF16 `[8, 4, 2304]`; rows past the count are unused.
    pub taps: CudaSlice<bf16>,
    /// I32 `[3]`: the first position to commit, the number of positions to commit (0 to 8)
    /// and the anchor token. The block sits right after the committed positions.
    pub control: CudaSlice<i32>,
}

/// A captured round and the device address of the LM head it reads.
struct RoundGraph {
    graph: CudaGraph,
    lm_head: u64,
}

// SAFETY: a graph exec may be launched from any thread with its context bound, which
// `CudaGraph::launch` does, and the draft launches it only through `&mut Draft`.
unsafe impl Send for RoundGraph {}

struct Scratch {
    mixed: CudaSlice<bf16>,
    ctx_k: CudaSlice<bf16>,
    ctx_v: CudaSlice<bf16>,
    /// A round's commit: the 3 layers' mixes, BF16 `[3, 8, 2304]`, and their context keys
    /// and values, BF16 `[3, 8, 1024]` each.
    round_mixed: CudaSlice<bf16>,
    round_k: CudaSlice<bf16>,
    round_v: CudaSlice<bf16>,
    /// The Markov head's mixed taps of the last committed position, BF16 `[2304]`.
    f: CudaSlice<bf16>,
    residual: CudaSlice<bf16>,
    h: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    o: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    up: CudaSlice<bf16>,
    act: CudaSlice<bf16>,
    mlp: CudaSlice<bf16>,
    /// The decoder's output for the block, BF16 `[8, 2304]`.
    y: CudaSlice<bf16>,
    attention: AttentionScratch,
    /// Rows 1 to 7 of `y` through the LM head, BF16 `[7, padded vocab]`.
    base: CudaSlice<bf16>,
    context_in: CudaSlice<f32>,
    context: CudaSlice<f32>,
    /// The Markov state, F32 `[256]`, alternating between two buffers by step.
    state: [CudaSlice<f32>; 2],
    latent: CudaSlice<f32>,
    partials: ArgmaxPartials,
    bounds: LogitBounds,
    ids: CudaSlice<i32>,
    /// Each step's Markov logits, BF16 `[7, vocab]`, when recorded.
    logits: Option<CudaSlice<bf16>>,
}

impl Draft {
    /// Uploads the fold and allocates the cache and scratch. The global layer's cache holds
    /// `capacity` positions, 4 KiB each; the local layers' rings hold 1032.
    pub fn new(
        stream: &Arc<CudaStream>,
        config: &TextConfig,
        weights: DraftWeights,
        shared: SharedWeights<'_>,
        capacity: usize,
    ) -> anyhow::Result<Self> {
        Self::with_options(
            stream,
            config,
            weights,
            shared,
            capacity,
            &DraftOptions::default(),
        )
    }

    /// [`Self::new`] with the approximations in `options`.
    pub fn with_options(
        stream: &Arc<CudaStream>,
        config: &TextConfig,
        weights: DraftWeights,
        shared: SharedWeights<'_>,
        capacity: usize,
        options: &DraftOptions,
    ) -> anyhow::Result<Self> {
        ensure!(
            capacity >= LOCAL_SLOTS,
            "capacity {capacity} is below the {LOCAL_SLOTS} positions of a window and a block"
        );
        let kernels = DraftKernels::load(stream.context())?;
        let padded_vocab = config.vocab_size;
        ensure!(
            shared.lm_head.len() == padded_vocab * HIDDEN,
            "the LM head has {} elements, expected {padded_vocab} x {HIDDEN}",
            shared.lm_head.len()
        );
        let fold = fold(
            stream,
            &kernels,
            &weights,
            shared,
            padded_vocab,
            config.unpadded_vocab_size,
        )?;
        let alpha: Vec<f32> = weights.alpha.iter().flatten().copied().collect();
        let inv_freq = rope_inv_freq();
        let caches = (0..LAYERS)
            .map(|layer| KvCache::new(stream, if layer == 1 { capacity } else { LOCAL_SLOTS }))
            .collect::<anyhow::Result<_>>()?;
        let z = |n: usize| stream.alloc_zeros::<bf16>(n);
        let zf = |n: usize| stream.alloc_zeros::<f32>(n);
        let s = Scratch {
            mixed: z(COMMIT_ROWS * HIDDEN)?,
            ctx_k: z(COMMIT_ROWS * KV_WIDTH)?,
            ctx_v: z(COMMIT_ROWS * KV_WIDTH)?,
            round_mixed: z(LAYERS * BLOCK * HIDDEN)?,
            round_k: z(LAYERS * BLOCK * KV_WIDTH)?,
            round_v: z(LAYERS * BLOCK * KV_WIDTH)?,
            f: z(HIDDEN)?,
            residual: z(BLOCK * HIDDEN)?,
            h: z(BLOCK * HIDDEN)?,
            q: z(BLOCK * HEADS * HEAD_DIM)?,
            k: z(BLOCK * KV_WIDTH)?,
            v: z(BLOCK * KV_WIDTH)?,
            attn: z(BLOCK * HEADS * HEAD_DIM)?,
            o: z(BLOCK * HIDDEN)?,
            gate: z(BLOCK * MLP_WIDTH)?,
            up: z(BLOCK * MLP_WIDTH)?,
            act: z(BLOCK * MLP_WIDTH)?,
            mlp: z(BLOCK * HIDDEN)?,
            y: z(BLOCK * HIDDEN)?,
            attention: AttentionScratch::new(stream)?,
            base: z(STEPS * padded_vocab)?,
            context_in: zf(STEPS * 2 * HIDDEN)?,
            context: zf(STEPS * 3 * RANK)?,
            state: [zf(RANK)?, zf(RANK)?],
            latent: zf(RANK)?,
            partials: ArgmaxPartials::new(stream)?,
            bounds: LogitBounds::new(stream, config.unpadded_vocab_size)?,
            ids: stream.alloc_zeros(STEPS)?,
            logits: None,
        };
        let backbone = options
            .fp8_backbone
            .then(|| quantize_backbone(stream, &kernels, &weights))
            .transpose()?;
        let head = Head::new(
            stream,
            &kernels,
            shared,
            &fold.weight,
            &fold.bias,
            config.unpadded_vocab_size,
            options,
        )?;
        Ok(Self {
            kernels,
            weights,
            fold,
            alpha: stream.clone_htod(&alpha)?,
            inv_freq: stream.clone_htod(&inv_freq)?,
            caches,
            capacity,
            committed: 0,
            vocab: config.unpadded_vocab_size,
            padded_vocab,
            softcap: config.final_logit_softcapping,
            s,
            inputs: RoundInputs {
                taps: z(BLOCK * TAPS * HIDDEN)?,
                control: stream.alloc_zeros(CONTROL)?,
            },
            graphs: Default::default(),
            backbone,
            head,
        })
    }

    /// Positions the global layer's cache holds.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// One past the last committed position: the position of the next round's anchor.
    #[must_use]
    pub fn committed(&self) -> usize {
        self.committed
    }

    /// Forgets every committed position and the shortlist's context tokens, for a new
    /// sequence.
    pub fn reset(&mut self) {
        self.committed = 0;
        let Some(list) = self.head.as_mut().and_then(|h| h.shortlist.as_mut()) else {
            return;
        };
        let bucket = list.bucket();
        for &t in &list.context {
            if let Ok(t) = usize::try_from(t) {
                list.member[t] = false;
            }
        }
        list.context.clear();
        list.used = list.ranked;
        list.stale = true;
        if list.bucket() != bucket {
            self.graphs = Default::default();
        }
    }

    /// Uploads the shortlist's row count after a [`Self::reset`].
    fn sync_shortlist(&mut self, stream: &CudaStream) -> anyhow::Result<()> {
        if let Some(list) = self.head.as_mut().and_then(|h| h.shortlist.as_mut())
            && list.stale
        {
            self.kernels
                .set_i32(stream, &mut list.count, i32::try_from(list.used)?)?;
            list.stale = false;
        }
        Ok(())
    }

    /// Adds the tokens of the context (prompt and generated) that a shortlist head lacks to
    /// it, up to [`CONTEXT_ROWS`] since the last [`Self::reset`], so the draft can propose
    /// them. Does nothing without a shortlist.
    pub fn note_tokens(
        &mut self,
        stream: &Arc<CudaStream>,
        shared: SharedWeights<'_>,
        tokens: &[i32],
    ) -> anyhow::Result<()> {
        let Some(head) = self.head.as_mut() else {
            return Ok(());
        };
        let Some(list) = head.shortlist.as_mut() else {
            return Ok(());
        };
        let bucket = list.bucket();
        let first = list.used;
        let mut ids = Vec::new();
        for &t in tokens {
            let Ok(i) = usize::try_from(t) else { continue };
            if i < self.vocab && !list.member[i] && first + ids.len() < list.capacity() {
                list.member[i] = true;
                list.context.push(t);
                ids.push(t);
            }
        }
        if ids.is_empty() {
            return Ok(());
        }
        list.used += ids.len();
        stream.memcpy_htod(&ids, &mut list.map.slice_mut(first..list.used))?;
        stream.memcpy_htod(&[i32::try_from(list.used)?], &mut list.count)?;
        list.stale = false;
        let changed = list.bucket() != bucket;
        let ids = stream.clone_htod(&ids)?;
        head.gather(
            stream,
            &self.kernels,
            shared,
            &self.fold.weight,
            &self.fold.bias,
            &ids.slice(..),
            first,
        )?;
        if changed {
            self.graphs = Default::default();
        }
        Ok(())
    }

    /// Keeps each step's Markov logits from the next round on, for tests.
    pub fn record_logits(&mut self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        if self.s.logits.is_none() {
            self.s.logits = Some(stream.alloc_zeros(STEPS * self.vocab)?);
            self.graphs = Default::default();
        }
        Ok(())
    }

    /// The device inputs of [`Self::run_round`] and [`Self::replay_round`], which a verify
    /// pass may write directly.
    pub fn round_inputs(&mut self) -> &mut RoundInputs {
        &mut self.inputs
    }

    /// Fills [`Self::round_inputs`] from the host: the taps of positions
    /// `start..start + rows` (BF16 `[rows, 4, 2304]`, `rows` up to 8, possibly none) and
    /// the anchor token in `anchor` (one I32 on the device). The next round commits those
    /// positions and runs the block at `start + rows`. Like [`Self::commit`], `start` must
    /// not skip positions.
    pub fn set_round_inputs(
        &mut self,
        stream: &Arc<CudaStream>,
        taps: &impl DeviceIn<bf16>,
        start: usize,
        anchor: &impl DeviceIn<i32>,
    ) -> anyhow::Result<()> {
        let taps = taps.view();
        let row_len = TAPS * HIDDEN;
        ensure!(
            taps.len().is_multiple_of(row_len) && taps.len() <= BLOCK * row_len,
            "taps have {} elements, not a multiple of {row_len} up to {BLOCK} rows",
            taps.len()
        );
        let rows = taps.len() / row_len;
        ensure!(
            start <= self.committed,
            "committing from {start} would skip positions {}..{start}",
            self.committed
        );
        ensure!(
            start + rows + BLOCK <= self.capacity,
            "the block at {} exceeds the capacity {}",
            start + rows,
            self.capacity
        );
        self.sync_shortlist(stream)?;
        if rows > 0 {
            let mut dst = self.inputs.taps.slice_mut(..rows * row_len);
            stream.memcpy_dtod(&taps, &mut dst)?;
        }
        self.kernels
            .set_control(stream, &mut self.inputs.control, start, rows, anchor)?;
        self.committed = start + rows;
        Ok(())
    }

    /// Fills [`Self::round_inputs`] after a verify pass without reading its verdict: `taps`
    /// holds the verify's rows at positions `start..` (BF16 `[rows, 4, 2304]`, up to 8 rows),
    /// `kept` the number of drafts it kept (one I32 on the device) and `anchor` the next
    /// anchor. The next round commits the kept rows `0..=kept` and runs the block right
    /// after them. The host does not learn the new position; [`Self::committed`] becomes
    /// `start + rows`, an upper bound, until the next [`Self::commit`] or
    /// [`Self::set_round_inputs`].
    pub fn set_round_inputs_kept(
        &mut self,
        stream: &Arc<CudaStream>,
        taps: &impl DeviceIn<bf16>,
        start: usize,
        kept: &CudaView<'_, i32>,
        anchor: &impl DeviceIn<i32>,
    ) -> anyhow::Result<()> {
        let taps = taps.view();
        let row_len = TAPS * HIDDEN;
        ensure!(
            taps.len().is_multiple_of(row_len) && (1..=BLOCK).contains(&(taps.len() / row_len)),
            "taps have {} elements, not 1 to {BLOCK} rows of {row_len}",
            taps.len()
        );
        let rows = taps.len() / row_len;
        ensure!(
            start <= self.committed,
            "committing from {start} would skip positions {}..{start}",
            self.committed
        );
        ensure!(
            start + rows + BLOCK <= self.capacity,
            "the block at {} exceeds the capacity {}",
            start + rows,
            self.capacity
        );
        self.sync_shortlist(stream)?;
        let mut dst = self.inputs.taps.slice_mut(..rows * row_len);
        stream.memcpy_dtod(&taps, &mut dst)?;
        self.kernels.set_control_kept(
            stream,
            &mut self.inputs.control,
            start,
            kept,
            rows,
            anchor,
        )?;
        self.committed = start + rows;
        Ok(())
    }

    /// The last round's draft token ids, I32 `[7]`; a round of `k` drafts writes the first
    /// `k`.
    #[must_use]
    pub fn draft_ids(&self) -> &CudaSlice<i32> {
        &self.s.ids
    }

    /// The last round's decoder output for the 8 block rows, BF16 `[8, 2304]`.
    #[must_use]
    pub fn y(&self) -> &CudaSlice<bf16> {
        &self.s.y
    }

    /// The last round's base logits, rows 1 to 7 of [`Self::y`] through the LM head, BF16
    /// `[7, padded vocab]`.
    #[must_use]
    pub fn base_logits(&self) -> &CudaSlice<bf16> {
        &self.s.base
    }

    /// The last round's Markov logits, BF16 `[7, vocab]`, after [`Self::record_logits`].
    #[must_use]
    pub fn markov_logits(&self) -> Option<&CudaSlice<bf16>> {
        self.s.logits.as_ref()
    }

    /// Writes the draft cache entries of target positions `start..start + rows` from their
    /// taps, BF16 `[rows, 4, 2304]`: the partial block entering visits 5, 23 and 41, and the
    /// final normed hidden state. The last row's taps also become the Markov head's context
    /// for the round anchored right after it. `start` may go back to rewrite positions, but
    /// must not skip any.
    pub fn commit(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        taps: &impl DeviceIn<bf16>,
        start: usize,
    ) -> anyhow::Result<()> {
        let taps = taps.view();
        let row_len = TAPS * HIDDEN;
        ensure!(
            taps.len().is_multiple_of(row_len) && !taps.is_empty(),
            "taps have {} elements, not a positive multiple of {row_len}",
            taps.len()
        );
        let rows = taps.len() / row_len;
        ensure!(
            start <= self.committed,
            "committing from {start} would skip positions {}..{start}",
            self.committed
        );
        ensure!(
            start + rows <= self.capacity,
            "positions up to {} exceed the capacity {}",
            start + rows,
            self.capacity
        );
        let s = &mut self.s;
        for offset in (0..rows).step_by(COMMIT_ROWS) {
            let n = COMMIT_ROWS.min(rows - offset);
            let chunk = taps.slice(offset * row_len..(offset + n) * row_len);
            let position = start + offset;
            for (layer, (w, cache)) in self.weights.layers.iter().zip(&mut self.caches).enumerate()
            {
                let mut mixed = s.mixed.slice_mut(..n * HIDDEN);
                let alpha = self.alpha.slice(layer * TAPS..(layer + 1) * TAPS);
                self.kernels.mix_rms_norm(
                    stream,
                    &chunk,
                    &alpha,
                    &w.dflare_norm,
                    &mut mixed,
                    EPS,
                )?;
                let mixed = s.mixed.slice(..n * HIDDEN);
                let mut k = s.ctx_k.slice_mut(..n * KV_WIDTH);
                let mut v = s.ctx_v.slice_mut(..n * KV_WIDTH);
                linear.gemm_bf16(
                    stream,
                    &mut [
                        Gemm::new(&w.k_ctx, &mixed, &mut k),
                        Gemm::new(&w.v_ctx, &mixed, &mut v),
                    ],
                    n,
                )?;
                self.kernels.head_norm_rope(
                    stream,
                    &mut s.ctx_k,
                    n,
                    KV_WIDTH / HEAD_DIM,
                    KV_WIDTH,
                    &w.k_norm_ctx,
                    &self.inv_freq,
                    position,
                    EPS,
                )?;
                self.kernels
                    .store_kv(stream, &s.ctx_k, &s.ctx_v, n, KV_WIDTH, cache, position)?;
            }
        }
        let last = taps.slice((rows - 1) * row_len..);
        let alpha = self.alpha.slice(LAYERS * TAPS..);
        self.kernels.mix_rms_norm(
            stream,
            &last,
            &alpha,
            &self.weights.markov.dflare_norm,
            &mut s.f,
            EPS,
        )?;
        self.committed = start + rows;
        Ok(())
    }

    /// Runs one round anchored at the token in `anchor` (one I32 on the device) at position
    /// [`Self::committed`]: the block forward, the base logits and the 7 greedy Markov
    /// steps. The ids land in [`Self::draft_ids`].
    pub fn round(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        shared: SharedWeights<'_>,
        anchor: &impl DeviceIn<i32>,
    ) -> anyhow::Result<()> {
        let position = self.committed;
        ensure!(
            position > 0,
            "a round needs at least one committed position"
        );
        ensure!(
            position + BLOCK <= self.capacity,
            "the block at {position} exceeds the capacity {}",
            self.capacity
        );
        self.sync_shortlist(stream)?;
        self.kernels
            .set_control(stream, &mut self.inputs.control, position, 0, anchor)?;
        self.block(stream, linear, shared)?;
        self.markov(stream, linear, shared, STEPS)
    }

    /// Runs a round from [`Self::round_inputs`]: commits its positions, then runs the block
    /// right after them and `steps` Markov steps (1 to 7), which leave `steps` ids in
    /// [`Self::draft_ids`]. Every position comes from device memory. The caller keeps the
    /// block inside the capacity: positions past it are not stored.
    pub fn run_round(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        shared: SharedWeights<'_>,
        steps: usize,
    ) -> anyhow::Result<()> {
        ensure!(
            (1..=STEPS).contains(&steps),
            "{steps} drafts, expected 1 to {STEPS}"
        );
        self.commit_round(stream, linear)?;
        self.block(stream, linear, shared)?;
        self.markov(stream, linear, shared, steps)
    }

    /// [`Self::run_round`] as a CUDA graph, captured on the first call for each `steps`.
    /// `shared` must be the same tensors on every call.
    pub fn replay_round(
        &mut self,
        stream: &Arc<CudaStream>,
        linear: &Linear,
        shared: SharedWeights<'_>,
        steps: usize,
    ) -> anyhow::Result<()> {
        ensure!(
            (1..=STEPS).contains(&steps),
            "{steps} drafts, expected 1 to {STEPS}"
        );
        self.sync_shortlist(stream)?;
        let lm_head = shared.lm_head.device_ptr(stream).0;
        if self.graphs[steps - 1].is_none() {
            stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let enqueued = self.run_round(stream, linear, shared, steps);
            // End the capture even when enqueueing failed, so the stream is usable again.
            let graph = stream.end_capture(CUgraphInstantiate_flags(0));
            enqueued?;
            let graph = graph?.context("the round capture recorded no work")?;
            graph.upload()?;
            self.graphs[steps - 1] = Some(RoundGraph { graph, lm_head });
        }
        let round = self.graphs[steps - 1]
            .as_ref()
            .context("the round graph is missing")?;
        ensure!(
            round.lm_head == lm_head,
            "the round graph was captured with other shared weights"
        );
        round.graph.launch()?;
        Ok(())
    }

    /// The commit of a round's positions from [`Self::round_inputs`].
    fn commit_round(&mut self, stream: &CudaStream, linear: &Linear) -> anyhow::Result<()> {
        let k = &self.kernels;
        let s = &mut self.s;
        let w = &self.weights;
        let control = &self.inputs.control;
        k.commit_mix(
            stream,
            &self.inputs.taps,
            control,
            &self.alpha,
            [
                &w.layers[0].dflare_norm,
                &w.layers[1].dflare_norm,
                &w.layers[2].dflare_norm,
                &w.markov.dflare_norm,
            ],
            &mut s.round_mixed,
            &mut s.f,
            EPS,
        )?;
        for (layer, lw) in w.layers.iter().enumerate() {
            let mixed = s
                .round_mixed
                .slice(layer * BLOCK * HIDDEN..(layer + 1) * BLOCK * HIDDEN);
            let kv = layer * BLOCK * KV_WIDTH..(layer + 1) * BLOCK * KV_WIDTH;
            let ck = s.round_k.slice_mut(kv.clone());
            let cv = s.round_v.slice_mut(kv);
            let f = self.backbone.as_ref().map(|b| &b[layer]);
            project(
                linear,
                k,
                stream,
                &mixed,
                &mut [
                    (&lw.k_ctx, f.map(|f| &f.k_ctx), ck),
                    (&lw.v_ctx, f.map(|f| &f.v_ctx), cv),
                ],
            )?;
        }
        k.commit_rope_store(
            stream,
            &s.round_k,
            &s.round_v,
            control,
            &self.inv_freq,
            EPS,
            [
                &w.layers[0].k_norm_ctx,
                &w.layers[1].k_norm_ctx,
                &w.layers[2].k_norm_ctx,
            ],
            &mut self.caches,
        )
    }

    /// The block forward at the position in [`Self::round_inputs`].
    fn block(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        shared: SharedWeights<'_>,
    ) -> anyhow::Result<()> {
        let k = &self.kernels;
        let s = &mut self.s;
        let control = &self.inputs.control;
        k.block_input(
            stream,
            &control.slice(2..3),
            shared.embed,
            shared.embed_norm,
            &self.weights.mask_embed,
            &mut s.residual,
            EPS,
        )?;
        for (layer, (w, cache)) in self.weights.layers.iter().zip(&mut self.caches).enumerate() {
            let x = (layer > 0).then_some(&s.mlp);
            k.add_rms_norm(stream, x, &mut s.residual, &w.input_norm, &mut s.h, EPS)?;
            let f = self.backbone.as_ref().map(|b| &b[layer]);
            project(
                linear,
                k,
                stream,
                &s.h.slice(..),
                &mut [
                    (&w.q, f.map(|f| &f.q), s.q.slice_mut(..)),
                    (&w.k, f.map(|f| &f.k), s.k.slice_mut(..)),
                    (&w.v, f.map(|f| &f.v), s.v.slice_mut(..)),
                ],
            )?;
            let global = layer == 1;
            k.block_rope_store(
                stream,
                &mut s.q,
                &s.k,
                &s.v,
                control,
                &self.inv_freq,
                EPS,
                &w.q_norm,
                &w.k_norm,
                cache,
                global,
            )?;
            let keys = AttentionKeys::Device {
                control,
                window: if global { 0 } else { WINDOW },
                max_keys: if global { cache.slots } else { LOCAL_SLOTS },
            };
            k.attention_from(stream, &s.q, cache, keys, &mut s.attention, &mut s.attn)?;
            project(
                linear,
                k,
                stream,
                &s.attn.slice(..),
                &mut [(&w.o, f.map(|f| &f.o), s.o.slice_mut(..))],
            )?;
            k.add_rms_norm(
                stream,
                Some(&s.o),
                &mut s.residual,
                &w.post_attention_norm,
                &mut s.h,
                EPS,
            )?;
            project(
                linear,
                k,
                stream,
                &s.h.slice(..),
                &mut [
                    (&w.gate, f.map(|f| &f.gate), s.gate.slice_mut(..)),
                    (&w.up, f.map(|f| &f.up), s.up.slice_mut(..)),
                ],
            )?;
            k.silu_mul(stream, &s.gate, &s.up, &mut s.act)?;
            project(
                linear,
                k,
                stream,
                &s.act.slice(..),
                &mut [(&w.down, f.map(|f| &f.down), s.mlp.slice_mut(..))],
            )?;
        }
        k.add_rms_norm(
            stream,
            Some(&s.mlp),
            &mut s.residual,
            &self.weights.norm,
            &mut s.y,
            EPS,
        )
    }

    /// The base logits and `steps` greedy Markov steps of the block in [`Self::y`].
    fn markov(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        shared: SharedWeights<'_>,
        steps: usize,
    ) -> anyhow::Result<()> {
        let k = &self.kernels;
        let mw = &self.weights.markov;
        let s = &mut self.s;
        let control = &self.inputs.control;
        let head = self.head.as_ref();
        let list = head.and_then(|h| h.shortlist.as_ref());
        // At least two rows for the BF16 GEMMs: a single row would run as a GEMV, which
        // sums in another order.
        let m_rows = steps.max(2);
        // Rows of the base logits, each `ld` elements.
        let (rows, ld) = match list {
            Some(list) => (list.bucket(), list.bucket()),
            None => (self.vocab, self.padded_vocab),
        };
        match head.and_then(|h| h.lm.as_ref()) {
            None => {
                let y = s.y.slice(HIDDEN..(m_rows + 1) * HIDDEN);
                let mut base = s.base.slice_mut(..m_rows * ld);
                linear.gemm_bf16(
                    stream,
                    &mut [Gemm::new(shared.lm_head, &y, &mut base)],
                    m_rows,
                )?;
            }
            Some(Matrix::Bf16(w)) => {
                let y = s.y.slice(HIDDEN..(m_rows + 1) * HIDDEN);
                let w = w.slice(..rows * HIDDEN);
                let mut base = s.base.slice_mut(..m_rows * rows);
                linear.gemm_bf16(stream, &mut [Gemm::new(&w, &y, &mut base)], m_rows)?;
            }
            Some(Matrix::Fp8(w)) => {
                let y = s.y.slice(HIDDEN..(steps + 1) * HIDDEN);
                let base = s.base.slice_mut(..);
                k.gemm_fp8(stream, &y, steps, &mut [fp8_rows(w, rows, base, ld)])?;
            }
        }
        k.markov_context_input(stream, &s.y, &s.f, &mut s.context_in)?;
        let context_in = s.context_in.slice(..m_rows * 2 * HIDDEN);
        let mut context = s.context.slice_mut(..m_rows * 3 * RANK);
        linear.gemm_f32(
            stream,
            &mut [Gemm::new(&mw.rnn_context, &context_in, &mut context)],
            m_rows,
        )?;
        for step in 0..=steps {
            let [even, odd] = &mut s.state;
            let (state_in, state_out) = if step % 2 == 0 {
                (&*even, odd)
            } else {
                (&*odd, even)
            };
            let row = step.min(steps - 1);
            let context = s.context.slice(row * 3 * RANK..(row + 1) * 3 * RANK);
            k.markov_step(
                stream,
                step,
                steps,
                &s.partials,
                control,
                &mut s.ids,
                &mw.w1,
                &mw.rnn_state,
                &context,
                &mw.rnn_bias,
                state_in,
                state_out,
                &mut s.latent,
            )?;
            if step < steps {
                let base = s.base.slice(step * ld..step * ld + rows);
                let logits = s
                    .logits
                    .as_mut()
                    .map(|l| l.slice_mut(step * self.vocab..(step + 1) * self.vocab));
                let step_argmax = StepArgmax {
                    kernels: k,
                    fold: &self.fold,
                    head,
                    vocab: self.vocab,
                    softcap: self.softcap,
                };
                step_argmax.run(
                    stream,
                    step,
                    &base,
                    &s.latent,
                    logits,
                    &mut s.bounds,
                    &mut s.partials,
                )?;
            }
        }
        Ok(())
    }
}

/// The LM-head fold and its copies for the pruned argmax.
struct Fold {
    /// `lm_head · hidden_out`, BF16 `[padded vocab, 256]`.
    weight: CudaSlice<bf16>,
    /// `lm_head · hidden_out.bias`, BF16 `[padded vocab]`.
    bias: CudaSlice<bf16>,
    /// An E4M3 copy of the weight over the vocabulary.
    fp8: Fp8Rows,
    /// The bound on each row's error in the E4M3 copy's products, F32 `[vocab]`.
    error: CudaSlice<f32>,
}

/// `lm_head · hidden_out` and `lm_head · hidden_out.bias` over the padded vocabulary,
/// and for the pruned argmax an E4M3 copy of the fold and each row's error bound over the
/// vocabulary.
fn fold(
    stream: &Arc<CudaStream>,
    kernels: &DraftKernels,
    weights: &DraftWeights,
    shared: SharedWeights<'_>,
    padded_vocab: usize,
    vocab: usize,
) -> anyhow::Result<Fold> {
    let mut fold_weight = stream.alloc_zeros(padded_vocab * RANK)?;
    let mut fold_bias = stream.alloc_zeros(padded_vocab)?;
    kernels.fold(
        stream,
        shared.lm_head,
        &weights.markov.hidden_out,
        &weights.markov.hidden_out_bias,
        &mut fold_weight,
        &mut fold_bias,
    )?;
    let mut fold_fp8 = Fp8Rows::new(stream, vocab, RANK)?;
    kernels.quantize_rows(stream, &fold_weight, None, vocab, 0, &mut fold_fp8)?;
    let mut fold_error = stream.alloc_zeros(vocab)?;
    kernels.fold_error(stream, &fold_weight, &fold_fp8, &mut fold_error)?;
    Ok(Fold {
        weight: fold_weight,
        bias: fold_bias,
        fp8: fold_fp8,
        error: fold_error,
    })
}

/// FP8 copies of every layer's projections.
fn quantize_backbone(
    stream: &Arc<CudaStream>,
    kernels: &DraftKernels,
    weights: &DraftWeights,
) -> anyhow::Result<Vec<Fp8Layer>> {
    let quantize = |w: &CudaSlice<bf16>, k: usize| -> anyhow::Result<Fp8Rows> {
        let rows = w.len() / k;
        let mut out = Fp8Rows::new(stream, rows, k)?;
        kernels.quantize_rows(stream, w, None, rows, 0, &mut out)?;
        Ok(out)
    };
    weights
        .layers
        .iter()
        .map(|w| {
            Ok(Fp8Layer {
                q: quantize(&w.q, HIDDEN)?,
                k: quantize(&w.k, HIDDEN)?,
                v: quantize(&w.v, HIDDEN)?,
                o: quantize(&w.o, HEADS * HEAD_DIM)?,
                gate: quantize(&w.gate, HIDDEN)?,
                up: quantize(&w.up, HIDDEN)?,
                down: quantize(&w.down, MLP_WIDTH)?,
                k_ctx: quantize(&w.k_ctx, HIDDEN)?,
                v_ctx: quantize(&w.v_ctx, HIDDEN)?,
            })
        })
        .collect()
}

impl Head {
    /// The approximate head `options` ask for, or `None` for the exact one.
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &DraftKernels,
        shared: SharedWeights<'_>,
        fold_weight: &CudaSlice<bf16>,
        fold_bias: &CudaSlice<bf16>,
        vocab: usize,
        options: &DraftOptions,
    ) -> anyhow::Result<Option<Self>> {
        let Some(ranking) = &options.shortlist else {
            if !options.fp8_head && !options.fp8_fold {
                return Ok(None);
            }
            let quantize = |w: &CudaSlice<bf16>, k: usize| -> anyhow::Result<Matrix> {
                let mut out = Fp8Rows::new(stream, vocab, k)?;
                kernels.quantize_rows(stream, w, None, vocab, 0, &mut out)?;
                Ok(Matrix::Fp8(out))
            };
            return Ok(Some(Self {
                lm: options
                    .fp8_head
                    .then(|| quantize(shared.lm_head, HIDDEN))
                    .transpose()?,
                fold: options
                    .fp8_fold
                    .then(|| quantize(fold_weight, RANK))
                    .transpose()?,
                fold_bias: None,
                shortlist: None,
            }));
        };
        let mut member = vec![false; vocab];
        let mut taken = 0;
        for &t in ranking {
            let t = t as usize;
            if taken == SHORTLIST_ROWS {
                break;
            }
            if t < vocab && !member[t] {
                member[t] = true;
                taken += 1;
            }
        }
        for m in member.iter_mut().skip(FIRST_SPECIAL) {
            *m = true;
        }
        let ids: Vec<i32> = (0..vocab)
            .filter(|&t| member[t])
            .map(i32::try_from)
            .collect::<Result<_, _>>()?;
        let ranked = ids.len();
        let capacity = ranked + CONTEXT_ROWS;
        let mut map = stream.alloc_zeros::<i32>(capacity)?;
        stream.memcpy_htod(&ids, &mut map.slice_mut(..ranked))?;
        let count = stream.clone_htod(&[i32::try_from(ranked)?])?;
        let mut head = Self {
            lm: Some(if options.fp8_head {
                Matrix::Fp8(Fp8Rows::new(stream, capacity, HIDDEN)?)
            } else {
                Matrix::Bf16(stream.alloc_zeros(capacity * HIDDEN)?)
            }),
            fold: Some(if options.fp8_fold {
                Matrix::Fp8(Fp8Rows::new(stream, capacity, RANK)?)
            } else {
                Matrix::Bf16(stream.alloc_zeros(capacity * RANK)?)
            }),
            fold_bias: Some(stream.alloc_zeros(capacity)?),
            shortlist: Some(Shortlist {
                ranked,
                used: ranked,
                member,
                context: Vec::new(),
                map,
                count,
                stale: false,
            }),
        };
        let ids = stream.clone_htod(&ids)?;
        head.gather(
            stream,
            kernels,
            shared,
            fold_weight,
            fold_bias,
            &ids.slice(..),
            0,
        )?;
        Ok(Some(head))
    }

    /// Copies the head, fold and fold-bias rows of the tokens in `ids` into rows
    /// `first..` of the shortlist's tables.
    #[expect(clippy::too_many_arguments, reason = "the source tables and the rows")]
    fn gather(
        &mut self,
        stream: &CudaStream,
        kernels: &DraftKernels,
        shared: SharedWeights<'_>,
        fold_weight: &CudaSlice<bf16>,
        fold_bias: &CudaSlice<bf16>,
        ids: &CudaView<'_, i32>,
        first: usize,
    ) -> anyhow::Result<()> {
        let rows = ids.len();
        let copy = |src: &CudaSlice<bf16>, k: usize, dst: &mut Matrix| match dst {
            Matrix::Bf16(dst) => kernels.gather_rows(stream, src, k, ids, first, dst),
            Matrix::Fp8(dst) => kernels.quantize_rows(stream, src, Some(ids), rows, first, dst),
        };
        if let Some(lm) = self.lm.as_mut() {
            copy(shared.lm_head, HIDDEN, lm)?;
        }
        if let Some(fold) = self.fold.as_mut() {
            copy(fold_weight, RANK, fold)?;
        }
        if let Some(bias) = self.fold_bias.as_mut() {
            kernels.gather_rows(stream, fold_bias, 1, ids, first, bias)?;
        }
        Ok(())
    }
}

/// `out = x · weight^T` over the 8 block rows for each product `(weight, fp8, out)`:
/// through the FP8 copies when every product has one, else through the BF16 weights.
fn project(
    linear: &Linear,
    kernels: &DraftKernels,
    stream: &CudaStream,
    x: &CudaView<'_, bf16>,
    products: &mut [(&CudaSlice<bf16>, Option<&Fp8Rows>, CudaViewMut<'_, bf16>)],
) -> anyhow::Result<()> {
    let fp8: Option<Vec<&Fp8Rows>> = products.iter().map(|p| p.1).collect();
    if let Some(fp8) = fp8 {
        let mut fp8: Vec<Fp8Product<'_>> = fp8
            .into_iter()
            .zip(products.iter_mut())
            .map(|(w, p)| {
                let ld = p.2.len() / BLOCK;
                fp8_rows(w, w.rows(), p.2.slice_mut(..), ld)
            })
            .collect();
        return kernels.gemm_fp8(stream, x, BLOCK, &mut fp8);
    }
    let mut gemms: Vec<Gemm<'_, bf16>> = products
        .iter_mut()
        .map(|p| Gemm::new(p.0, x, &mut p.2))
        .collect();
    linear.gemm_bf16(stream, &mut gemms, BLOCK)
}

/// An [`Fp8Product`] over the first `rows` rows of `w`.
fn fp8_rows<'a>(
    w: &'a Fp8Rows,
    rows: usize,
    out: CudaViewMut<'a, bf16>,
    ld: usize,
) -> Fp8Product<'a> {
    Fp8Product { w, rows, out, ld }
}

/// One Markov step's argmax over its base logits and latent.
struct StepArgmax<'a> {
    kernels: &'a DraftKernels,
    fold: &'a Fold,
    head: Option<&'a Head>,
    vocab: usize,
    softcap: f32,
}

impl StepArgmax<'_> {
    /// Writes the step's argmax partials: over every row with recorded `logits`, over the
    /// rows that can win for the exact fold, otherwise through the approximate head.
    #[expect(clippy::too_many_arguments, reason = "the step's buffers")]
    fn run(
        &self,
        stream: &CudaStream,
        step: usize,
        base: &CudaView<'_, bf16>,
        latent: &CudaSlice<f32>,
        logits: Option<CudaViewMut<'_, bf16>>,
        bounds: &mut LogitBounds,
        partials: &mut ArgmaxPartials,
    ) -> anyhow::Result<()> {
        let k = self.kernels;
        let fold = self.fold;
        let approx_fold = self
            .head
            .is_some_and(|h| h.fold.is_some() || h.shortlist.is_some());
        if logits.is_some() && self.head.is_none() {
            return k.markov_fold(
                stream,
                step % 2 == 1,
                &fold.weight,
                &fold.bias,
                base,
                latent,
                self.vocab,
                self.softcap,
                logits,
                partials,
            );
        }
        if !approx_fold {
            return k.markov_pruned(
                stream,
                step % 2 == 1,
                &fold.weight,
                &fold.fp8,
                &fold.error,
                &fold.bias,
                base,
                latent,
                self.vocab,
                self.softcap,
                bounds,
                partials,
            );
        }
        let head = self.head.context("no approximate head")?;
        let list = head.shortlist.as_ref();
        let weight = match &head.fold {
            None => FoldRows::Bf16(&fold.weight),
            Some(Matrix::Bf16(w)) => FoldRows::Bf16(w),
            Some(Matrix::Fp8(w)) => FoldRows::Fp8(w),
        };
        k.markov_fold_any(
            stream,
            weight,
            head.fold_bias.as_ref().unwrap_or(&fold.bias),
            base,
            latent,
            base.len(),
            list.map(|l| &l.count),
            list.map(|l| &l.map),
            self.softcap,
            partials,
        )
    }
}
