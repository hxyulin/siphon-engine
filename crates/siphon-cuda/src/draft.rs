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

mod batch;
pub use batch::MAX_BATCH_DRAFT_SEQS;

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaGraph, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
};
use half::bf16;
use siphon::TextConfig;

use crate::{
    DeviceIn,
    model::{DeviceSpan, GlobalKv},
    ops::spec::TapLayout,
    ops::{
        draft::{
            ArgmaxPartials, AttentionKeys, AttentionScratch, BLOCK, CONTROL, DraftKernels,
            F8_MAX_PRODUCTS, FoldRows, Fp8Product, Fp8Rows, KV_WIDTH, KvCache, LogitBounds,
            RACE_WORDS, Rope, StepRace, TOP_K, TopkScratch, race_words,
        },
        linear::{Gemm, Linear, MAX_PRODUCTS, Start},
        linear_rows::{LinearRows, MAX_PRODUCTS as ROWS_MAX_PRODUCTS, RowsProduct},
        sample::Sampling,
    },
    weights::{
        dflash::{DflashLayerWeights, DflashWeights},
        dflash2::{
            CONV_COEFFICIENTS, ConvWeights, DFLASH2_HEADS, DFLASH2_MLP_WIDTH, DFLASH2_ROPE_THETA,
            DFLASH2_WINDOW, Dflash2LayerWeights, Dflash2Weights,
        },
        draft::{
            DraftLayerWeights, DraftWeights, HEAD_DIM, HEADS, HIDDEN, LAYERS, MLP_WIDTH, RANK,
            SharedWeights, TAPS,
        },
    },
};

/// Draft tokens per round: one per mask row.
pub const STEPS: usize = BLOCK - 1;
const _: () = assert!(STEPS == siphon::memory::DRAFT_STEPS);
/// Positions before the block that the local layers (0 and 2) see.
pub const WINDOW: usize = 1024;
/// Slots of the local layers' rings: the window and the block.
const LOCAL_SLOTS: usize = WINDOW + BLOCK;
const _: () = assert!(LOCAL_SLOTS == siphon::memory::DRAFT_RING_SLOTS);
/// Slots of DFlash 2's rings: the window before the block's first row and the block.
const DFLASH2_SLOTS: usize = DFLASH2_WINDOW + BLOCK;
const _: () = assert!(DFLASH2_SLOTS == siphon::memory::DFLASH2_RING_SLOTS);
/// RoPE base of the draft (`rope_theta` in `mtp_config`).
const ROPE_THETA: f64 = 1e6;
/// RMSNorm epsilon of the draft (`rms_norm_eps` in `mtp_config`).
const EPS: f32 = 1e-6;
/// Target positions one [`Draft::commit`] chunk projects at once.
const COMMIT_ROWS: usize = 512;
const _: () = assert!(COMMIT_ROWS == siphon::memory::COMMIT_ROWS);

/// Ranked tokens a shortlist head keeps, as Ingine's drafter.
pub const SHORTLIST_ROWS: usize = 32768;
const _: () = assert!(SHORTLIST_ROWS == siphon::memory::SHORTLIST_ROWS);
/// Context tokens a shortlist head adds at most.
pub const CONTEXT_ROWS: usize = 16384;
const _: () = assert!(CONTEXT_ROWS == siphon::memory::SHORTLIST_CONTEXT_ROWS);
/// The first special token id; a shortlist head keeps every token from here on.
const FIRST_SPECIAL: usize = 199_990;
const _: () = assert!(FIRST_SPECIAL == siphon::memory::FIRST_SPECIAL);
/// A shortlist head's base logits cover its rows in use rounded up to this.
pub const HEAD_BUCKET: usize = 1024;

/// Validates a configured shortlist ranked-token budget: a multiple of [`HEAD_BUCKET`] in
/// `HEAD_BUCKET..=SHORTLIST_ROWS`.
pub fn validate_shortlist_rows(n: usize) -> anyhow::Result<usize> {
    ensure!(
        (HEAD_BUCKET..=SHORTLIST_ROWS).contains(&n) && n.is_multiple_of(HEAD_BUCKET),
        "draft shortlist rows must be a multiple of {HEAD_BUCKET} in {HEAD_BUCKET}..={SHORTLIST_ROWS}, not {n}"
    );
    Ok(n)
}

/// Validates a configured shortlist context-token budget: a multiple of [`HEAD_BUCKET`] in
/// `0..=CONTEXT_ROWS`.
pub fn validate_context_rows(n: usize) -> anyhow::Result<usize> {
    ensure!(
        n <= CONTEXT_ROWS && n.is_multiple_of(HEAD_BUCKET),
        "draft context rows must be a multiple of {HEAD_BUCKET} in 0..={CONTEXT_ROWS}, not {n}"
    );
    Ok(n)
}

/// Approximations of the draft. Each changes the draft's tokens and so how often the
/// target accepts them, never the target's output. The default is the exact draft.
#[derive(Clone, Debug, Default)]
#[expect(clippy::struct_excessive_bools, reason = "one flag per approximation")]
pub struct DraftOptions {
    /// FP8 copies of the block's projections and of the round's context projections.
    pub fp8_backbone: bool,
    /// Base logits from an FP8 copy of the LM head (or of its shortlist rows).
    pub fp8_head: bool,
    /// An FP8 copy of the fold weight (or of its shortlist rows).
    pub fp8_fold: bool,
    /// E4M3 global caches, 2 KiB per position and layer instead of 4 KiB, without a
    /// scale: keys are RMS-normed per head and values stay within E4M3's range.
    pub fp8_kv: bool,
    /// Token ids by rank, most frequent first: the base and Markov logits
    /// then cover only the first `shortlist_rows` (default [`SHORTLIST_ROWS`]) of them,
    /// the special tokens and up to `context_rows` (default [`CONTEXT_ROWS`]) tokens
    /// passed to [`Draft::note_tokens`].
    pub shortlist: Option<Vec<u32>>,
    /// Ranked tokens a shortlist head keeps (`HEAD_BUCKET..=SHORTLIST_ROWS`, multiple of
    /// [`HEAD_BUCKET`]); `None` uses [`SHORTLIST_ROWS`].
    pub shortlist_rows: Option<usize>,
    /// Context tokens a shortlist head adds at most (`0..=CONTEXT_ROWS`, multiple of
    /// [`HEAD_BUCKET`]); `None` uses [`CONTEXT_ROWS`].
    pub context_rows: Option<usize>,
}

impl DraftOptions {
    /// The three FP8 approximations without a shortlist. Its round takes about 0.96 ms
    /// against 0.51 ms for [`Self::fast`], for about the same tokens per round.
    #[must_use]
    pub fn fp8() -> Self {
        Self {
            fp8_backbone: true,
            fp8_head: true,
            fp8_fold: true,
            fp8_kv: true,
            shortlist: None,
            shortlist_rows: None,
            context_rows: None,
        }
    }

    /// Every approximation, with the shortlist ranked by [`counted_ranking`]: the draft
    /// speculation uses unless told otherwise.
    #[must_use]
    pub fn fast() -> Self {
        Self {
            fp8_backbone: true,
            fp8_head: true,
            fp8_fold: true,
            fp8_kv: true,
            shortlist: Some(counted_ranking()),
            shortlist_rows: None,
            context_rows: None,
        }
    }

    /// Applies optional shortlist ranked-token and context-token budgets after validating
    /// them against [`HEAD_BUCKET`], [`SHORTLIST_ROWS`] and [`CONTEXT_ROWS`].
    pub fn with_shortlist_budget(
        mut self,
        shortlist_rows: Option<usize>,
        context_rows: Option<usize>,
    ) -> anyhow::Result<Self> {
        if let Some(n) = shortlist_rows {
            self.shortlist_rows = Some(validate_shortlist_rows(n)?);
        }
        if let Some(n) = context_rows {
            self.context_rows = Some(validate_context_rows(n)?);
        }
        Ok(self)
    }

    /// The options of a variant `spec`: `exact`, `fp8` ([`Self::fp8`]), `fast`
    /// ([`Self::fast`]), or a `+`-joined set of `fp8-backbone`, `fp8-head`, `fp8-fold`,
    /// `fp8-kv`, `shortlist`, `shortlist=<N>` and `context=<N>`. `shortlist` takes
    /// `ranking`, or without one [`counted_ranking`]; `fp8` and `fast` with a `ranking`
    /// use it as their shortlist.
    pub fn parse(spec: &str, ranking: Option<&[u32]>) -> anyhow::Result<Self> {
        let mut options = Self::default();
        for part in spec.split('+') {
            match part {
                "exact" => options = Self::default(),
                "fp8" => {
                    options = Self::fp8();
                    if let Some(r) = ranking {
                        options.shortlist = Some(r.to_vec());
                    }
                }
                "fast" => {
                    options = Self::fast();
                    if let Some(r) = ranking {
                        options.shortlist = Some(r.to_vec());
                    }
                }
                "fp8-backbone" => options.fp8_backbone = true,
                "fp8-head" => options.fp8_head = true,
                "fp8-fold" => options.fp8_fold = true,
                "fp8-kv" => options.fp8_kv = true,
                "shortlist" => {
                    options.shortlist =
                        Some(ranking.map_or_else(counted_ranking, <[u32]>::to_vec));
                }
                other => {
                    if let Some(v) = other.strip_prefix("shortlist=") {
                        let n: usize = v
                            .parse()
                            .with_context(|| format!("bad shortlist size {v:?}"))?;
                        options.shortlist_rows = Some(validate_shortlist_rows(n)?);
                        options.shortlist =
                            Some(ranking.map_or_else(counted_ranking, <[u32]>::to_vec));
                    } else if let Some(v) = other.strip_prefix("context=") {
                        let n: usize = v
                            .parse()
                            .with_context(|| format!("bad context size {v:?}"))?;
                        options.context_rows = Some(validate_context_rows(n)?);
                    } else {
                        anyhow::bail!("unknown draft variant part {other:?}");
                    }
                }
            }
        }
        Ok(options)
    }

    /// The approximations as a memory plan counts them.
    #[must_use]
    pub fn approx(&self) -> siphon::memory::DraftApprox {
        siphon::memory::DraftApprox {
            fp8_backbone: self.fp8_backbone,
            fp8_head: self.fp8_head,
            fp8_fold: self.fp8_fold,
            fp8_kv: self.fp8_kv,
            shortlist: self.shortlist.is_some(),
        }
    }
}

/// The [`SHORTLIST_ROWS`] tokens the target generated most often, most frequent first, a
/// ranking for [`DraftOptions::shortlist`]. `scripts/shortlist/` counted them over 1.0
/// million tokens that the target generated greedily for open-perfectblend prompts:
/// output tokens by count, then prompt tokens by count, special tokens left out (the
/// shortlist keeps them all). They cover 98.9% of the tokens the target generated for
/// held-out prompts.
#[must_use]
pub fn counted_ranking() -> Vec<u32> {
    let ranking = ranking_from_bytes(include_bytes!("../data/shortlist.bin"));
    debug_assert_eq!(ranking.len(), SHORTLIST_ROWS);
    ranking
}

/// Reads a token ranking for [`DraftOptions::shortlist`]: little-endian u32 token ids,
/// most frequent first, as [`counted_ranking`]'s file or Ingine's `draft_shortlist.bin`.
pub fn read_ranking(path: &std::path::Path) -> anyhow::Result<Vec<u32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len() % 4 == 0, "the ranking is not whole u32s");
    Ok(ranking_from_bytes(&bytes))
}

fn ranking_from_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

/// The RoPE frequencies `1 / theta^(2i/128)` of the bundled draft (theta 1e6) as vLLM's
/// F32 table builds them: a correctly rounded power, then a correctly rounded reciprocal.
#[must_use]
pub fn rope_inv_freq() -> Vec<f32> {
    rope_inv_freq_theta(ROPE_THETA)
}

/// [`rope_inv_freq`] for the base `theta`.
#[must_use]
pub fn rope_inv_freq_theta(theta: f64) -> Vec<f32> {
    (0..HEAD_DIM / 2)
        .map(|i| {
            #[expect(clippy::cast_precision_loss, reason = "i is below 64")]
            let exponent = (2 * i) as f64 / HEAD_DIM as f64;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "rounding the power to F32 is the intent"
            )]
            let power = theta.powf(exponent) as f32;
            1.0 / power
        })
        .collect()
}

/// The draft's weights, its LM-head fold, its cache and its scratch buffers.
pub struct Draft {
    kernels: DraftKernels,
    linear_rows: LinearRows,
    arch: Arch,
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
    /// A device copy of the BF16 LM head for the base logits, when they read it and the
    /// target keeps it in mapped host memory.
    head_copy: Option<CudaSlice<bf16>>,
    /// Per-slot sliding rings and batched buffers for DFlash 2.
    batch: Option<batch::BatchDflash2>,
}

/// Copies of a draft's sliding rings at one committed position, from
/// [`Draft::save_rings`]. Rounds and commits after that position overwrite ring slots,
/// but not the global layers' slots below it.
pub struct DraftRings {
    /// Each ring layer's keys and values; `None` for a global layer.
    caches: Vec<Option<KvCache>>,
    committed: usize,
}

impl DraftRings {
    /// Each ring layer's keys and values, in the order of [`Draft::ring_spans`]. Copies
    /// into these buffers change the rings, whose committed position
    /// [`Self::set_committed`] then gives.
    pub(crate) fn spans(&self, stream: &CudaStream) -> Vec<DeviceSpan> {
        self.caches
            .iter()
            .flatten()
            .flat_map(|c| [DeviceSpan::of(&*c.k, stream), DeviceSpan::of(&*c.v, stream)])
            .collect()
    }

    /// Makes `committed` the committed position of copies that [`Self::spans`] filled.
    pub(crate) fn set_committed(&mut self, committed: usize) {
        self.committed = committed;
    }

    /// Device bytes the copies take.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.caches.iter().flatten().map(KvCache::bytes).sum()
    }
}

/// The weights of one of the two draft architectures.
#[expect(
    clippy::large_enum_variant,
    reason = "one per draft, never moved after load"
)]
enum Arch {
    /// The DSpark draft bundled with the checkpoint (`mtp.safetensors`).
    Stock {
        weights: DraftWeights,
        /// `lm_head · hidden_out` and its copies for the pruned argmax.
        fold: Fold,
        /// `softmax(fuse_weight)`, F32 `[4, 4]`.
        alpha: CudaSlice<f32>,
    },
    /// A DFlash draft with DSpark heads ([`DflashWeights`]): `fc` over the concatenated
    /// taps, every layer global, a vanilla Markov head and a confidence head.
    Dflash {
        weights: DflashWeights,
        /// Zeros over the vocabulary: the vanilla head's bias has no constant term.
        zero_bias: CudaSlice<bf16>,
        /// An FP8 copy of `fc`, with [`DraftOptions::fp8_backbone`].
        fc_fp8: Option<Fp8Rows>,
    },
    /// Inco AI's DFlash 2 drafter ([`Dflash2Weights`]): `fc` over the taps, sliding layers
    /// with block convolutions, and a candidate selector over the top 16 tokens.
    Dflash2 {
        weights: Dflash2Weights,
        /// An FP8 copy of `fc`, with [`DraftOptions::fp8_backbone`].
        fc_fp8: Option<Fp8Rows>,
        /// The mask token's normed target embedding, BF16 `[2304]`.
        mask_embed: CudaSlice<bf16>,
    },
}

/// The context rows of a DFlash or DFlash 2 draft: `fc` and `hidden_norm` of the taps,
/// then each layer's `k_proj`, `v_proj` and key norm.
struct ContextRefs<'a> {
    fc: &'a CudaSlice<bf16>,
    fc_fp8: Option<&'a Fp8Rows>,
    hidden_norm: &'a CudaSlice<bf16>,
    /// Each layer's `(k, v, k_norm)`.
    layers: Vec<(
        &'a CudaSlice<bf16>,
        &'a CudaSlice<bf16>,
        &'a CudaSlice<bf16>,
    )>,
}

/// The weights of one decoder layer's block forward, which both architectures share.
/// The projections only the block reads are `None` once [`Arch::release_block_bf16`]
/// has freed them for their FP8 copies.
struct LayerRefs<'a> {
    input_norm: &'a CudaSlice<bf16>,
    post_attention_norm: &'a CudaSlice<bf16>,
    q: Option<&'a CudaSlice<bf16>>,
    k: &'a CudaSlice<bf16>,
    v: &'a CudaSlice<bf16>,
    o: Option<&'a CudaSlice<bf16>>,
    q_norm: &'a CudaSlice<bf16>,
    k_norm: &'a CudaSlice<bf16>,
    gate: Option<&'a CudaSlice<bf16>>,
    up: Option<&'a CudaSlice<bf16>>,
    down: Option<&'a CudaSlice<bf16>>,
    /// DFlash 2's convolutions around the attention and the MLP.
    convs: Option<(&'a ConvWeights, &'a ConvWeights)>,
}

impl<'a> From<&'a DraftLayerWeights> for LayerRefs<'a> {
    fn from(w: &'a DraftLayerWeights) -> Self {
        Self {
            input_norm: &w.input_norm,
            post_attention_norm: &w.post_attention_norm,
            q: w.q.as_ref(),
            k: &w.k,
            v: &w.v,
            o: w.o.as_ref(),
            q_norm: &w.q_norm,
            k_norm: &w.k_norm,
            gate: w.gate.as_ref(),
            up: w.up.as_ref(),
            down: w.down.as_ref(),
            convs: None,
        }
    }
}

impl<'a> From<&'a DflashLayerWeights> for LayerRefs<'a> {
    fn from(w: &'a DflashLayerWeights) -> Self {
        Self {
            input_norm: &w.input_norm,
            post_attention_norm: &w.post_attention_norm,
            q: w.q.as_ref(),
            k: &w.k,
            v: &w.v,
            o: w.o.as_ref(),
            q_norm: &w.q_norm,
            k_norm: &w.k_norm,
            gate: w.gate.as_ref(),
            up: w.up.as_ref(),
            down: w.down.as_ref(),
            convs: None,
        }
    }
}

impl<'a> From<&'a Dflash2LayerWeights> for LayerRefs<'a> {
    fn from(w: &'a Dflash2LayerWeights) -> Self {
        Self {
            input_norm: &w.input_norm,
            post_attention_norm: &w.post_attention_norm,
            q: w.q.as_ref(),
            k: &w.k,
            v: &w.v,
            o: w.o.as_ref(),
            q_norm: &w.q_norm,
            k_norm: &w.k_norm,
            gate: w.gate.as_ref(),
            up: w.up.as_ref(),
            down: w.down.as_ref(),
            convs: Some((&w.attention_conv, &w.mlp_conv)),
        }
    }
}

impl Arch {
    fn layers(&self) -> usize {
        match self {
            Self::Stock { weights, .. } => weights.layers.len(),
            Self::Dflash { weights, .. } => weights.layers.len(),
            Self::Dflash2 { weights, .. } => weights.layers.len(),
        }
    }

    /// Frees every layer's BF16 projections that only the block reads (queries, output,
    /// gate, up and down), once the FP8 copies of [`DraftOptions::fp8_backbone`] stand in
    /// for them. The commit keeps reading the BF16 keys, values and context projections.
    fn release_block_bf16(&mut self) {
        macro_rules! release {
            ($layers:expr) => {
                for w in $layers {
                    w.q = None;
                    w.o = None;
                    w.gate = None;
                    w.up = None;
                    w.down = None;
                }
            };
        }
        match self {
            Self::Stock { weights, .. } => release!(&mut weights.layers),
            Self::Dflash { weights, .. } => release!(&mut weights.layers),
            Self::Dflash2 { weights, .. } => release!(&mut weights.layers),
        }
    }

    fn layer(&self, layer: usize) -> LayerRefs<'_> {
        match self {
            Self::Stock { weights, .. } => (&weights.layers[layer]).into(),
            Self::Dflash { weights, .. } => (&weights.layers[layer]).into(),
            Self::Dflash2 { weights, .. } => (&weights.layers[layer]).into(),
        }
    }

    /// Whether `layer` keeps every position up to the capacity rather than a ring.
    fn global(&self, layer: usize) -> bool {
        match self {
            Self::Stock { .. } => layer == 1,
            Self::Dflash { .. } => true,
            Self::Dflash2 { .. } => false,
        }
    }

    /// Slots of `layer`'s cache for a draft of `capacity` positions.
    fn slots(&self, layer: usize, capacity: usize) -> usize {
        match self {
            Self::Dflash2 { .. } => DFLASH2_SLOTS,
            _ if self.global(layer) => capacity,
            _ => LOCAL_SLOTS,
        }
    }

    /// The keys of `layer`'s block attention: the positions before the block it reads
    /// (0: all) and the window of each row (0: none), as [`AttentionKeys::Device`] takes
    /// them.
    fn attention_window(&self, layer: usize) -> (usize, usize) {
        match self {
            Self::Dflash2 { .. } => (DFLASH2_WINDOW - 1, DFLASH2_WINDOW),
            _ if self.global(layer) => (0, 0),
            _ => (WINDOW, 0),
        }
    }

    /// Width of the queries of a block row.
    fn q_width(&self) -> usize {
        match self {
            Self::Dflash2 { .. } => DFLASH2_HEADS * HEAD_DIM,
            _ => HEADS * HEAD_DIM,
        }
    }

    /// Width of the MLP.
    fn mlp_width(&self) -> usize {
        match self {
            Self::Dflash2 { .. } => DFLASH2_MLP_WIDTH,
            _ => MLP_WIDTH,
        }
    }

    /// The RoPE arithmetic of the reference the draft follows.
    fn rope(&self) -> Rope {
        match self {
            Self::Stock { .. } => Rope::F32,
            Self::Dflash { .. } => Rope::Bf16Table,
            Self::Dflash2 { .. } => Rope::Bf16Ops,
        }
    }

    fn rope_theta(&self) -> f64 {
        match self {
            Self::Dflash2 { .. } => DFLASH2_ROPE_THETA,
            _ => ROPE_THETA,
        }
    }

    fn mask_embed(&self) -> &CudaSlice<bf16> {
        match self {
            Self::Stock { weights, .. } => &weights.mask_embed,
            Self::Dflash { weights, .. } => &weights.mask_embed,
            Self::Dflash2 { mask_embed, .. } => mask_embed,
        }
    }

    fn norm(&self) -> &CudaSlice<bf16> {
        match self {
            Self::Stock { weights, .. } => &weights.norm,
            Self::Dflash { weights, .. } => &weights.norm,
            Self::Dflash2 { weights, .. } => &weights.norm,
        }
    }

    /// The rows a shortlist head gathers for its Markov bias, and their bias, if the
    /// draft has a Markov head.
    fn fold_rows(&self) -> Option<(&CudaSlice<bf16>, &CudaSlice<bf16>)> {
        match self {
            Self::Stock { fold, .. } => Some((&fold.weight, &fold.bias)),
            Self::Dflash {
                weights, zero_bias, ..
            } => Some((&weights.w2, zero_bias)),
            Self::Dflash2 { .. } => None,
        }
    }

    /// The context projections of a DFlash or DFlash 2 draft.
    fn context(&self) -> Option<ContextRefs<'_>> {
        match self {
            Self::Stock { .. } => None,
            Self::Dflash {
                weights, fc_fp8, ..
            } => Some(ContextRefs {
                fc: &weights.fc,
                fc_fp8: fc_fp8.as_ref(),
                hidden_norm: &weights.hidden_norm,
                layers: weights
                    .layers
                    .iter()
                    .map(|l| (&l.k, &l.v, &l.k_norm))
                    .collect(),
            }),
            Self::Dflash2 {
                weights, fc_fp8, ..
            } => Some(ContextRefs {
                fc: &weights.fc,
                fc_fp8: fc_fp8.as_ref(),
                hidden_norm: &weights.hidden_norm,
                layers: weights
                    .layers
                    .iter()
                    .map(|l| (&l.k, &l.v, &l.k_norm))
                    .collect(),
            }),
        }
    }

    fn taps(&self) -> TapLayout {
        match self {
            Self::Stock { .. } => TapLayout::stock(),
            Self::Dflash { weights, .. } => weights.taps.clone(),
            Self::Dflash2 { weights, .. } => weights.taps.clone(),
        }
    }

    /// Taps per position.
    fn tap_count(&self) -> usize {
        self.taps().len()
    }
}

/// FP8 copies of one layer's projections. Only the stock draft has context projections.
struct Fp8Layer {
    q: Fp8Rows,
    k: Fp8Rows,
    v: Fp8Rows,
    o: Fp8Rows,
    gate: Fp8Rows,
    up: Fp8Rows,
    down: Fp8Rows,
    k_ctx: Option<Fp8Rows>,
    v_ctx: Option<Fp8Rows>,
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
/// ascending id order, then up to `context_cap` context tokens.
struct Shortlist {
    ranked: usize,
    context_cap: usize,
    used: usize,
    member: Vec<bool>,
    context: Vec<i32>,
    /// Row to token id, I32 `[ranked + context_cap]`.
    map: CudaSlice<i32>,
    /// Rows in use, I32 `[1]`.
    count: CudaSlice<i32>,
    /// Whether `count` lags `used`.
    stale: bool,
}

impl Shortlist {
    fn capacity(&self) -> usize {
        self.ranked + self.context_cap
    }

    /// Rows the base logits cover.
    fn bucket(&self) -> usize {
        self.used.next_multiple_of(HEAD_BUCKET).min(self.capacity())
    }
}

/// A round's inputs in device memory, for [`Draft::run_round`] and [`Draft::replay_round`].
pub struct RoundInputs {
    /// Taps of the positions to commit, BF16 `[8, taps, 2304]`; rows past the count are unused.
    pub taps: CudaSlice<bf16>,
    /// I32 `[3]`: the first position to commit, the number of positions to commit (0 to 8)
    /// and the anchor token. The block sits right after the committed positions.
    pub control: CudaSlice<i32>,
    /// U32 `[4]`, the request's sampling as the draft reads it: the temperature's bits and
    /// the seed's low and high words (see [`Draft::set_sampling`]).
    pub race: CudaSlice<u32>,
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
    /// Each draft's softmax probability under the Markov logits, F32 `[7]`.
    confidence: CudaSlice<f32>,
    /// Each step's Markov logits, BF16 `[7, vocab]`, when recorded.
    logits: Option<CudaSlice<bf16>>,
    /// The context rows of a DFlash or DFlash 2 draft.
    dflash: Option<DflashScratch>,
    /// DFlash 2's convolutions and selector.
    dflash2: Option<Dflash2Scratch>,
}

/// DFlash 2's convolution coefficients and outputs, and the selector's inputs.
struct Dflash2Scratch {
    /// The coefficient projections of the attention's and the MLP's input, BF16
    /// `[8, 576]` each.
    coef_attention: CudaSlice<bf16>,
    coef_mlp: CudaSlice<bf16>,
    /// The input convolution of a sublayer, BF16 `[8, 2304]`.
    conv: CudaSlice<bf16>,
    topk: TopkScratch,
    /// The top 16 candidates of each draft row and their logits, `[7, 16]`.
    cand: CudaSlice<i32>,
    unary: CudaSlice<f32>,
    /// The hidden projection of block rows 1 to 7, BF16 `[7, 256]`.
    hp: CudaSlice<bf16>,
}

/// The context rows of a DFlash draft: `fc` of the taps, then `hidden_norm`.
struct DflashScratch {
    /// [`Draft::commit`]'s, BF16 `[512, 2304]` each.
    raw: CudaSlice<bf16>,
    normed: CudaSlice<bf16>,
    /// A round's, BF16 `[8, 2304]` each.
    round_raw: CudaSlice<bf16>,
    round_normed: CudaSlice<bf16>,
}

impl Scratch {
    fn new(stream: &Arc<CudaStream>, config: &TextConfig, arch: &Arch) -> anyhow::Result<Self> {
        let z = |n: usize| stream.alloc_zeros::<bf16>(n);
        let zf = |n: usize| stream.alloc_zeros::<f32>(n);
        let dflash = match arch {
            Arch::Stock { .. } => None,
            Arch::Dflash { .. } | Arch::Dflash2 { .. } => Some(DflashScratch {
                raw: z(COMMIT_ROWS * HIDDEN)?,
                normed: z(COMMIT_ROWS * HIDDEN)?,
                round_raw: z(BLOCK * HIDDEN)?,
                round_normed: z(BLOCK * HIDDEN)?,
            }),
        };
        let dflash2 = match arch {
            Arch::Dflash2 { .. } => Some(Dflash2Scratch {
                coef_attention: z(BLOCK * CONV_COEFFICIENTS)?,
                coef_mlp: z(BLOCK * CONV_COEFFICIENTS)?,
                conv: z(BLOCK * HIDDEN)?,
                topk: TopkScratch::new(stream)?,
                cand: stream.alloc_zeros(STEPS * TOP_K)?,
                unary: zf(STEPS * TOP_K)?,
                hp: z(STEPS * RANK)?,
            }),
            _ => None,
        };
        let q_width = arch.q_width();
        let mlp_width = arch.mlp_width();
        Ok(Self {
            mixed: z(COMMIT_ROWS * HIDDEN)?,
            ctx_k: z(COMMIT_ROWS * KV_WIDTH)?,
            ctx_v: z(COMMIT_ROWS * KV_WIDTH)?,
            round_mixed: z(LAYERS * BLOCK * HIDDEN)?,
            round_k: z(arch.layers() * BLOCK * KV_WIDTH)?,
            round_v: z(arch.layers() * BLOCK * KV_WIDTH)?,
            f: z(HIDDEN)?,
            residual: z(BLOCK * HIDDEN)?,
            h: z(BLOCK * HIDDEN)?,
            q: z(BLOCK * q_width)?,
            k: z(BLOCK * KV_WIDTH)?,
            v: z(BLOCK * KV_WIDTH)?,
            attn: z(BLOCK * q_width)?,
            o: z(BLOCK * HIDDEN)?,
            gate: z(BLOCK * mlp_width)?,
            up: z(BLOCK * mlp_width)?,
            act: z(BLOCK * mlp_width)?,
            mlp: z(BLOCK * HIDDEN)?,
            y: z(BLOCK * HIDDEN)?,
            attention: AttentionScratch::new(stream)?,
            base: z(STEPS * config.vocab_size)?,
            context_in: zf(STEPS * 2 * HIDDEN)?,
            context: zf(STEPS * 3 * RANK)?,
            state: [zf(RANK)?, zf(RANK)?],
            latent: zf(RANK)?,
            partials: ArgmaxPartials::new(stream)?,
            bounds: LogitBounds::new(stream, config.unpadded_vocab_size)?,
            ids: stream.alloc_zeros(STEPS)?,
            confidence: stream.alloc_zeros(STEPS)?,
            logits: None,
            dflash,
            dflash2,
        })
    }
}

impl Draft {
    /// Uploads the fold and allocates the cache and scratch. The global layer's cache holds
    /// `capacity` positions, 4 KiB each (2 KiB with [`DraftOptions::fp8_kv`]); the local
    /// layers' rings hold 1032.
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
        let kernels = DraftKernels::load(stream.context())?;
        check_lm_head(config, shared)?;
        let fold = fold(
            stream,
            &kernels,
            &weights,
            shared,
            config.vocab_size,
            config.unpadded_vocab_size,
        )?;
        let alpha: Vec<f32> = weights.alpha.iter().flatten().copied().collect();
        let alpha = stream.clone_htod(&alpha)?;
        let arch = Arch::Stock {
            weights,
            fold,
            alpha,
        };
        Self::build(stream, config, kernels, arch, shared, capacity, options)
    }

    /// A DFlash draft with DSpark heads ([`DflashWeights`]) in place of the bundled one,
    /// with the approximations in `options`. Every layer keeps `capacity` positions, 4 KiB
    /// each (2 KiB with [`DraftOptions::fp8_kv`]). Its taps are [`Self::tap_layout`], which
    /// the target must record.
    pub fn dflash(
        stream: &Arc<CudaStream>,
        config: &TextConfig,
        weights: DflashWeights,
        shared: SharedWeights<'_>,
        capacity: usize,
        options: &DraftOptions,
    ) -> anyhow::Result<Self> {
        let kernels = DraftKernels::load(stream.context())?;
        check_lm_head(config, shared)?;
        let fc_fp8 = if options.fp8_backbone {
            let k = weights.taps.len() * HIDDEN;
            let mut out = Fp8Rows::new(stream, HIDDEN, k)?;
            kernels.quantize_rows(stream, &weights.fc, None, HIDDEN, 0, &mut out)?;
            Some(out)
        } else {
            None
        };
        let arch = Arch::Dflash {
            weights,
            zero_bias: stream.alloc_zeros(config.vocab_size)?,
            fc_fp8,
        };
        Self::build(stream, config, kernels, arch, shared, capacity, options)
    }

    /// Inco AI's DFlash 2 drafter ([`Dflash2Weights`]) in place of the bundled draft,
    /// with the approximations in `options` (`fp8_fold` does nothing: the drafter has no
    /// Markov head). Every layer keeps a ring of the last 2056 positions, 4 KiB each. Its
    /// taps are [`Self::tap_layout`], which the target must record.
    pub fn dflash2(
        stream: &Arc<CudaStream>,
        config: &TextConfig,
        weights: Dflash2Weights,
        shared: SharedWeights<'_>,
        capacity: usize,
        options: &DraftOptions,
    ) -> anyhow::Result<Self> {
        let kernels = DraftKernels::load(stream.context())?;
        check_lm_head(config, shared)?;
        let fc_fp8 = if options.fp8_backbone {
            let k = weights.taps.len() * HIDDEN;
            let mut out = Fp8Rows::new(stream, HIDDEN, k)?;
            kernels.quantize_rows(stream, &weights.fc, None, HIDDEN, 0, &mut out)?;
            Some(out)
        } else {
            None
        };
        // The mask rows are the mask token's normed target embedding: row 0 of a block
        // anchored at the mask token.
        let mask = stream.clone_htod(&[i32::try_from(weights.mask_token)?])?;
        let mut rows = stream.alloc_zeros::<bf16>(BLOCK * HIDDEN)?;
        let unused = stream.alloc_zeros::<bf16>(HIDDEN)?;
        kernels.block_input(
            stream,
            &mask,
            shared.embed,
            shared.embed_norm,
            &unused,
            &mut rows,
            EPS,
        )?;
        let mut mask_embed = stream.alloc_zeros::<bf16>(HIDDEN)?;
        stream.memcpy_dtod(&rows.slice(..HIDDEN), &mut mask_embed)?;
        let arch = Arch::Dflash2 {
            weights,
            fc_fp8,
            mask_embed,
        };
        Self::build(stream, config, kernels, arch, shared, capacity, options)
    }

    fn build(
        stream: &Arc<CudaStream>,
        config: &TextConfig,
        kernels: DraftKernels,
        mut arch: Arch,
        shared: SharedWeights<'_>,
        capacity: usize,
        options: &DraftOptions,
    ) -> anyhow::Result<Self> {
        ensure!(
            capacity >= LOCAL_SLOTS,
            "capacity {capacity} is below the {LOCAL_SLOTS} positions of a window and a block"
        );
        ensure!(
            config.final_logit_softcapping > 0.0,
            "the draft's confidences need a positive logit softcap"
        );
        let padded_vocab = config.vocab_size;
        let inv_freq = rope_inv_freq_theta(arch.rope_theta());
        let layers = arch.layers();
        let caches = (0..layers)
            .map(|layer| {
                let fp8 = options.fp8_kv && (arch.global(layer) || matches!(arch, Arch::Dflash2 { .. }));
                let slots = arch.slots(layer, capacity);
                match shared.kv_pool {
                    Some(pool) if arch.global(layer) => KvCache::pooled(stream, slots, fp8, pool),
                    _ => KvCache::with_format(stream, slots, fp8),
                }
            })
            .collect::<anyhow::Result<_>>()?;
        let s = Scratch::new(stream, config, &arch)?;
        let z = |n: usize| stream.alloc_zeros::<bf16>(n);
        let backbone = options
            .fp8_backbone
            .then(|| quantize_backbone(stream, &kernels, &arch))
            .transpose()?;
        if backbone.is_some() {
            arch.release_block_bf16();
        }
        let round_taps = z(BLOCK * arch.tap_count() * HIDDEN)?;
        let head = Head::new(
            stream,
            &kernels,
            shared,
            arch.fold_rows(),
            config.unpadded_vocab_size,
            options,
        )?;
        let head_copy = if shared.lm_head_on_host && head.as_ref().is_none_or(|h| h.lm.is_none()) {
            let mut copy = stream.alloc_zeros(shared.lm_head.len())?;
            stream.memcpy_dtod(shared.lm_head, &mut copy)?;
            Some(copy)
        } else {
            None
        };
        let linear_rows = LinearRows::load(stream.context())?;
        Ok(Self {
            kernels,
            linear_rows,
            arch,
            inv_freq: stream.clone_htod(&inv_freq)?,
            caches,
            capacity,
            committed: 0,
            vocab: config.unpadded_vocab_size,
            padded_vocab,
            softcap: config.final_logit_softcapping,
            s,
            inputs: RoundInputs {
                taps: round_taps,
                control: stream.alloc_zeros(CONTROL)?,
                race: stream.alloc_zeros(RACE_WORDS)?,
            },
            graphs: Default::default(),
            backbone,
            head,
            head_copy,
            batch: None,
        })
    }

    /// The target hidden states the draft reads per position, in order.
    #[must_use]
    pub fn tap_layout(&self) -> TapLayout {
        self.arch.taps()
    }

    /// Elements of one position's taps: taps times the hidden width.
    fn row_len(&self) -> usize {
        self.arch.tap_count() * HIDDEN
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

    /// Treats every position below `position` as committed without computing it, for
    /// benchmarks at long contexts. Drafts after this are not meaningful.
    pub fn skip_to_for_benchmark(&mut self, position: usize) -> anyhow::Result<()> {
        ensure!(
            position <= self.capacity,
            "the draft's cache cannot hold position {position}"
        );
        if let Some(stream) = self.caches.first().map(|c| Arc::clone(c.k.stream())) {
            self.map(position + BLOCK, &stream)?;
        }
        self.committed = position;
        Ok(())
    }

    /// Maps the global layers' caches to hold every position below `end`, rounded up to
    /// [`crate::model::MAP_POSITIONS`], from the target's page pool. The draft maps what
    /// its commits and rounds write before they run; this is for copies into the caches.
    /// Does nothing for caches that are not pooled.
    pub fn map(&mut self, end: usize, stream: &CudaStream) -> anyhow::Result<()> {
        let end = end.next_multiple_of(crate::model::MAP_POSITIONS);
        for c in &mut self.caches {
            c.map(end, stream)?;
        }
        Ok(())
    }

    /// Pages more that [`Self::map`] of `end` would take.
    #[must_use]
    pub fn pages_to(&self, end: usize) -> usize {
        let end = end.next_multiple_of(crate::model::MAP_POSITIONS);
        self.caches.iter().map(|c| c.pages_to(end)).sum()
    }

    /// Pages of the pool that the global layers' caches hold.
    #[must_use]
    pub fn held_pages(&self) -> usize {
        self.caches.iter().map(KvCache::held_pages).sum()
    }

    /// Positions of the global layers' caches that are mapped.
    #[must_use]
    pub fn mapped(&self) -> usize {
        self.caches
            .iter()
            .filter(|c| c.slots >= self.capacity)
            .map(KvCache::mapped)
            .min()
            .unwrap_or(0)
    }

    /// Returns the global layers' pages past the first `positions` to the pool, after
    /// `stream` has done its work, and forgets the positions past them.
    pub fn unmap_from(&mut self, positions: usize, stream: &CudaStream) -> anyhow::Result<()> {
        if !self.caches.iter().any(KvCache::is_pooled) {
            return Ok(());
        }
        stream.synchronize()?;
        let keep = positions.next_multiple_of(crate::model::MAP_POSITIONS);
        for c in &mut self.caches {
            c.unmap_from(keep)?;
        }
        self.committed = self.committed.min(positions);
        Ok(())
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
            if let Some(b) = self.batch.as_mut() {
                b.clear_graphs();
            }
        }
    }

    /// Whether the draft has global layers, whose caches hold every committed position of
    /// one sequence. [`DraftRings`] can then return the draft only to a sequence whose
    /// positions those caches still hold.
    #[must_use]
    pub fn has_global_layers(&self) -> bool {
        self.caches.iter().any(|c| c.slots >= self.capacity)
    }

    /// The sliding rings' keys and values, in the order of [`DraftRings::spans`].
    pub(crate) fn ring_spans(&self, stream: &CudaStream) -> Vec<DeviceSpan> {
        self.caches
            .iter()
            .filter(|c| c.slots < self.capacity)
            .flat_map(|c| [DeviceSpan::of(&*c.k, stream), DeviceSpan::of(&*c.v, stream)])
            .collect()
    }

    /// The global layers' caches, none for a draft of rings alone such as DFlash 2.
    pub(crate) fn global_kv(&self, stream: &CudaStream) -> GlobalKv {
        let globals: Vec<&KvCache> = self
            .caches
            .iter()
            .filter(|c| c.slots >= self.capacity)
            .collect();
        GlobalKv {
            caches: globals
                .iter()
                .flat_map(|c| {
                    [
                        DeviceSpan::of(&*c.k, stream).ptr,
                        DeviceSpan::of(&*c.v, stream).ptr,
                    ]
                })
                .collect(),
            position_bytes: globals
                .iter()
                .flat_map(|c| [KV_WIDTH * if c.fp8 { 1 } else { 2 }; 2])
                .collect(),
            capacity: self.capacity,
        }
    }

    /// Takes every position below `committed` as committed, once the global layers hold
    /// them and the sliding rings hold the state at `committed`, copied from elsewhere
    /// (the prefix cache's host tier). A commit must come before the next round, as after
    /// [`Self::restore_rings`].
    pub(crate) fn set_loaded(&mut self, committed: usize) -> anyhow::Result<()> {
        ensure!(
            committed <= self.capacity,
            "the draft's cache cannot hold position {committed}"
        );
        self.committed = committed;
        Ok(())
    }

    /// Allocates a [`DraftRings`] for the sliding rings, holding nothing yet.
    pub fn rings(&self, stream: &Arc<CudaStream>) -> anyhow::Result<DraftRings> {
        let caches = self
            .caches
            .iter()
            .map(|c| {
                (c.slots < self.capacity)
                    .then(|| KvCache::with_format(stream, c.slots, c.fp8))
                    .transpose()
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(DraftRings {
            caches,
            committed: 0,
        })
    }

    /// Copies the sliding rings and the committed position into `rings`, on the stream.
    pub fn save_rings(
        &self,
        stream: &Arc<CudaStream>,
        rings: &mut DraftRings,
    ) -> anyhow::Result<()> {
        ensure!(
            rings.caches.len() == self.caches.len(),
            "the copies have {} layers, the draft {}",
            rings.caches.len(),
            self.caches.len()
        );
        for (cache, copy) in self.caches.iter().zip(&mut rings.caches) {
            if let Some(copy) = copy {
                stream.memcpy_dtod(&*cache.k, &mut *copy.k)?;
                stream.memcpy_dtod(&*cache.v, &mut *copy.v)?;
            }
        }
        rings.committed = self.committed;
        Ok(())
    }

    /// Returns to the committed position of `rings`: copies back the sliding rings. The
    /// global layers must still hold the positions below it, so the draft must not have
    /// committed or drafted below it since, nor served another sequence. The Markov head's
    /// context stays the last commit's, so a commit must come before the next round.
    pub fn restore_rings(
        &mut self,
        stream: &Arc<CudaStream>,
        rings: &DraftRings,
    ) -> anyhow::Result<()> {
        ensure!(
            rings.caches.len() == self.caches.len(),
            "the copies have {} layers, the draft {}",
            rings.caches.len(),
            self.caches.len()
        );
        for (cache, copy) in self.caches.iter_mut().zip(&rings.caches) {
            if let Some(copy) = copy {
                stream.memcpy_dtod(&*copy.k, &mut *cache.k)?;
                stream.memcpy_dtod(&*copy.v, &mut *cache.v)?;
            }
        }
        self.committed = rings.committed;
        Ok(())
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

    /// Maximum context tokens the shortlist head can add (0 without a shortlist).
    #[must_use]
    pub fn context_capacity(&self) -> usize {
        self.head
            .as_ref()
            .and_then(|h| h.shortlist.as_ref())
            .map_or(0, |s| s.context_cap)
    }

    /// Adds the tokens of the context (prompt and generated) that a shortlist head lacks to
    /// it, up to [`Self::context_capacity`] since the last [`Self::reset`], so the draft can
    /// propose them. Does nothing without a shortlist.
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
            self.arch.fold_rows(),
            &ids.slice(..),
            first,
        )?;
        if changed {
            self.graphs = Default::default();
            if let Some(b) = self.batch.as_mut() {
                b.clear_graphs();
            }
        }
        Ok(())
    }

    /// Keeps each step's Markov logits from the next round on, for tests. Only the
    /// bundled draft records them.
    pub fn record_logits(&mut self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        ensure!(
            matches!(self.arch, Arch::Stock { .. }),
            "only the bundled draft records its Markov logits"
        );
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

    /// How the verify picks tokens, from the next round on. Greedy (the default) drafts
    /// each step's argmax. A sampling request makes the bundled and DFlash 2 drafts take
    /// the coupled pick: the best of logit / T plus the noise the verify's draw uses for
    /// the same seed and position (`kernels/race.cuh`). The verify keeps a draft only when
    /// it equals the draw, so the output stays the target's draw; the coupled pick is
    /// that draw whenever the draft's logits are close enough to the target's. The
    /// draft ignores the top-k, top-p and min-p filters, so it can propose a token the
    /// verify filters out. The settings live in [`RoundInputs::race`], so captured
    /// rounds replay with them. 2DSD's DFlash drafts stay greedy.
    pub fn set_sampling(
        &mut self,
        stream: &Arc<CudaStream>,
        sampling: Sampling,
    ) -> anyhow::Result<()> {
        let words = if sampling.is_greedy() {
            race_words(0.0, 0)
        } else {
            race_words(sampling.temperature, sampling.seed)
        };
        stream.memcpy_htod(&words, &mut self.inputs.race)?;
        Ok(())
    }

    /// Fills [`Self::round_inputs`] from the host: the taps of positions
    /// `start..start + rows` (BF16 `[rows, taps, 2304]` in the order of [`Self::tap_layout`], `rows` up to 8, possibly none) and
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
        let row_len = self.row_len();
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
        self.map(start + rows + BLOCK, stream)?;
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
    /// holds the verify's rows at positions `start..` (BF16 `[rows, taps, 2304]`, up to 8 rows),
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
        let row_len = self.row_len();
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
        self.map(start + rows + BLOCK, stream)?;
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

    /// The confidence of each of the last round's drafts, F32 `[7]`: its probability under
    /// the softmax of the Markov logits, an estimate that may be off by a few percent (it
    /// sums the E4M3 fold's logits unless the logits are recorded). With a sampling
    /// request ([`Self::set_sampling`]), the largest probability under the softmax of the
    /// logits over T instead: a coupled pick is kept at least about as often as the
    /// argmax would be if the draft matched the target. A round of `k` drafts writes the
    /// first `k`.
    #[must_use]
    pub fn confidences(&self) -> &CudaSlice<f32> {
        &self.s.confidence
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
    /// taps, BF16 `[rows, taps, 2304]` in the order of [`Self::tap_layout`]. For the bundled
    /// draft these are the partial block entering visits 5, 23 and 41 and the final normed
    /// hidden state, and the last row's taps also become the Markov head's context for the
    /// round anchored right after it. `start` may go back to rewrite positions, but
    /// must not skip any.
    pub fn commit(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        taps: &impl DeviceIn<bf16>,
        start: usize,
    ) -> anyhow::Result<()> {
        let taps = taps.view();
        let row_len = self.row_len();
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
        self.map(start + rows, stream)?;
        match &self.arch {
            Arch::Stock { weights, alpha, .. } => {
                let s = &mut self.s;
                for offset in (0..rows).step_by(COMMIT_ROWS) {
                    let n = COMMIT_ROWS.min(rows - offset);
                    let chunk = taps.slice(offset * row_len..(offset + n) * row_len);
                    let position = start + offset;
                    for (layer, (w, cache)) in
                        weights.layers.iter().zip(&mut self.caches).enumerate()
                    {
                        let mut mixed = s.mixed.slice_mut(..n * HIDDEN);
                        let alpha = alpha.slice(layer * TAPS..(layer + 1) * TAPS);
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
                let alpha = alpha.slice(LAYERS * TAPS..);
                self.kernels.mix_rms_norm(
                    stream,
                    &last,
                    &alpha,
                    &weights.markov.dflare_norm,
                    &mut s.f,
                    EPS,
                )?;
            }
            Arch::Dflash { .. } | Arch::Dflash2 { .. } => {
                self.commit_dflash(stream, linear, &taps, start, rows)?;
            }
        }
        self.committed = start + rows;
        Ok(())
    }

    /// [`Self::commit`]'s work for a DFlash or DFlash 2 draft: `fc` and `hidden_norm` of
    /// each position's taps, then every layer's keys and values of that context row.
    /// DFlash 2 skips positions that no later block's window reaches.
    fn commit_dflash(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        taps: &CudaView<'_, bf16>,
        start: usize,
        rows: usize,
    ) -> anyhow::Result<()> {
        let context = self
            .arch
            .context()
            .context("not a DFlash or DFlash 2 draft")?;
        let rope = self.arch.rope();
        let row_len = self.arch.tap_count() * HIDDEN;
        // The next block sits at start + rows or later, and its rows see at most the
        // DFLASH2_WINDOW - 1 positions before it.
        let skip = match self.arch {
            Arch::Dflash2 { .. } => (start + rows)
                .saturating_sub(DFLASH2_WINDOW - 1)
                .saturating_sub(start)
                .min(rows),
            _ => 0,
        };
        let (taps, start, rows) = (taps.slice(skip * row_len..), start + skip, rows - skip);
        let s = &mut self.s;
        let dfl = s
            .dflash
            .as_mut()
            .context("the DFlash buffers are missing")?;
        for offset in (0..rows).step_by(COMMIT_ROWS) {
            let n = COMMIT_ROWS.min(rows - offset);
            let chunk = taps.slice(offset * row_len..(offset + n) * row_len);
            let position = start + offset;
            let mut raw = dfl.raw.slice_mut(..n * HIDDEN);
            linear.gemm_bf16(stream, &mut [Gemm::new(context.fc, &chunk, &mut raw)], n)?;
            self.kernels.add_rms_norm(
                stream,
                None,
                &mut dfl.raw,
                context.hidden_norm,
                &mut dfl.normed,
                EPS,
            )?;
            let normed = dfl.normed.slice(..n * HIDDEN);
            for (&(k, v, k_norm), cache) in context.layers.iter().zip(&mut self.caches) {
                let mut ck = s.ctx_k.slice_mut(..n * KV_WIDTH);
                let mut cv = s.ctx_v.slice_mut(..n * KV_WIDTH);
                linear.gemm_bf16(
                    stream,
                    &mut [
                        Gemm::new(k, &normed, &mut ck),
                        Gemm::new(v, &normed, &mut cv),
                    ],
                    n,
                )?;
                self.kernels.head_norm_rope_table(
                    stream,
                    &mut s.ctx_k,
                    n,
                    KV_WIDTH / HEAD_DIM,
                    KV_WIDTH,
                    k_norm,
                    &self.inv_freq,
                    position,
                    EPS,
                    rope,
                )?;
                self.kernels
                    .store_kv(stream, &s.ctx_k, &s.ctx_v, n, KV_WIDTH, cache, position)?;
            }
        }
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
        self.map(position + BLOCK, stream)?;
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
        let control = &self.inputs.control;
        match &self.arch {
            Arch::Stock {
                weights: w, alpha, ..
            } => {
                k.commit_mix(
                    stream,
                    &self.inputs.taps,
                    control,
                    alpha,
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
                        &self.linear_rows,
                        k,
                        stream,
                        &mixed,
                        &mut [
                            (Some(&lw.k_ctx), f.and_then(|f| f.k_ctx.as_ref()), ck),
                            (Some(&lw.v_ctx), f.and_then(|f| f.v_ctx.as_ref()), cv),
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
            Arch::Dflash { .. } | Arch::Dflash2 { .. } => self.commit_round_dflash(stream, linear),
        }
    }

    /// [`Self::commit_round`]'s work for a DFlash or DFlash 2 draft.
    fn commit_round_dflash(&mut self, stream: &CudaStream, linear: &Linear) -> anyhow::Result<()> {
        let context = self
            .arch
            .context()
            .context("not a DFlash or DFlash 2 draft")?;
        let ring = matches!(self.arch, Arch::Dflash2 { .. });
        let rope = self.arch.rope();
        let k = &self.kernels;
        let s = &mut self.s;
        let control = &self.inputs.control;
        let dfl = s
            .dflash
            .as_mut()
            .context("the DFlash buffers are missing")?;
        // Every row of the 8 runs; the store keeps only the committed ones.
        project(
            linear,
            &self.linear_rows,
            k,
            stream,
            &self.inputs.taps.slice(..),
            &mut [(
                Some(context.fc),
                context.fc_fp8,
                dfl.round_raw.slice_mut(..),
            )],
        )?;
        k.add_rms_norm(
            stream,
            None,
            &mut dfl.round_raw,
            context.hidden_norm,
            &mut dfl.round_normed,
            EPS,
        )?;
        // Every layer's keys and values of the same context rows, in as few launches as
        // the product limits allow.
        let layer_rows = BLOCK * KV_WIDTH;
        let mut products = Vec::with_capacity(2 * context.layers.len());
        for ((layer, &(lk, lv, _)), (ck, cv)) in context.layers.iter().enumerate().zip(
            s.round_k
                .as_view_mut()
                .chunks_exact_mut(layer_rows)
                .zip(s.round_v.as_view_mut().chunks_exact_mut(layer_rows)),
        ) {
            let fp8 = self.backbone.as_ref().map(|b| &b[layer]);
            products.push((Some(lk), fp8.map(|f| &f.k), ck));
            products.push((Some(lv), fp8.map(|f| &f.v), cv));
        }
        project(
            linear,
            &self.linear_rows,
            k,
            stream,
            &dfl.round_normed.slice(..),
            &mut products,
        )?;
        let norms: Vec<&CudaSlice<bf16>> = context.layers.iter().map(|l| l.2).collect();
        k.dflash_commit_rope_store(
            stream,
            &s.round_k,
            &s.round_v,
            control,
            &self.inv_freq,
            EPS,
            &norms,
            &mut self.caches,
            ring,
            rope,
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
        let arch = &self.arch;
        k.block_input(
            stream,
            &control.slice(2..3),
            shared.embed,
            shared.embed_norm,
            arch.mask_embed(),
            &mut s.residual,
            EPS,
        )?;
        // The MLP convolution of the previous layer, which the next norm applies.
        let mut pending: Option<&ConvWeights> = None;
        for (layer, cache) in self.caches.iter_mut().enumerate() {
            let lw = arch.layer(layer);
            if layer == 0 {
                k.add_rms_norm(stream, None, &mut s.residual, lw.input_norm, &mut s.h, EPS)?;
            } else {
                s.add_norm(k, stream, Sublayer::Mlp, pending, lw.input_norm, false)?;
            }
            let f = self.backbone.as_ref().map(|b| &b[layer]);
            let use_rows = f.is_some();
            let (attention_conv, mlp_conv) = lw.convs.unzip();
            let input = sublayer_input(
                linear,
                &self.linear_rows,
                use_rows,
                k,
                stream,
                &s.h,
                s.dflash2.as_mut(),
                Sublayer::Attention,
                attention_conv,
            )?;
            project(
                linear,
                &self.linear_rows,
                k,
                stream,
                &input,
                &mut [
                    (lw.q, f.map(|f| &f.q), s.q.slice_mut(..)),
                    (Some(lw.k), f.map(|f| &f.k), s.k.slice_mut(..)),
                    (Some(lw.v), f.map(|f| &f.v), s.v.slice_mut(..)),
                ],
            )?;
            attend(k, stream, arch, s, control, &self.inv_freq, layer, cache)?;
            project(
                linear,
                &self.linear_rows,
                k,
                stream,
                &s.attn.slice(..),
                &mut [(lw.o, f.map(|f| &f.o), s.o.slice_mut(..))],
            )?;
            s.add_norm(
                k,
                stream,
                Sublayer::Attention,
                attention_conv,
                lw.post_attention_norm,
                false,
            )?;
            let input = sublayer_input(
                linear,
                &self.linear_rows,
                use_rows,
                k,
                stream,
                &s.h,
                s.dflash2.as_mut(),
                Sublayer::Mlp,
                mlp_conv,
            )?;
            project(
                linear,
                &self.linear_rows,
                k,
                stream,
                &input,
                &mut [
                    (lw.gate, f.map(|f| &f.gate), s.gate.slice_mut(..)),
                    (lw.up, f.map(|f| &f.up), s.up.slice_mut(..)),
                ],
            )?;
            k.silu_mul(stream, &s.gate, &s.up, &mut s.act)?;
            project(
                linear,
                &self.linear_rows,
                k,
                stream,
                &s.act.slice(..),
                &mut [(lw.down, f.map(|f| &f.down), s.mlp.slice_mut(..))],
            )?;
            pending = mlp_conv;
        }
        s.add_norm(k, stream, Sublayer::Mlp, pending, arch.norm(), true)
    }

    /// The base logits and `steps` Markov steps of the block in [`Self::y`], greedy or
    /// coupled ([`Self::set_sampling`]).
    fn markov(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        shared: SharedWeights<'_>,
        steps: usize,
    ) -> anyhow::Result<()> {
        let (weights, fold) = match &self.arch {
            Arch::Stock { weights, fold, .. } => (weights, fold),
            Arch::Dflash { .. } => return self.markov_dflash(stream, linear, shared, steps),
            Arch::Dflash2 { .. } => return self.select_dflash2(stream, linear, shared, steps),
        };
        let k = &self.kernels;
        let mw = &weights.markov;
        let s = &mut self.s;
        let control = &self.inputs.control;
        let race = &self.inputs.race;
        let head = self.head.as_ref();
        // At least two rows for the BF16 GEMMs: a single row would run as a GEMV, which
        // sums in another order.
        let m_rows = steps.max(2);
        let (rows, ld) = base_logits(
            linear,
            &self.linear_rows,
            k,
            stream,
            self.head_copy.as_ref().unwrap_or(shared.lm_head),
            head,
            s,
            steps,
            self.vocab,
            self.padded_vocab,
        )?;
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
                self.softcap,
                control,
                &mut s.ids,
                &mut s.confidence,
                &mw.w1,
                &mw.rnn_state,
                &context,
                &mw.rnn_bias,
                state_in,
                state_out,
                &mut s.latent,
                race,
            )?;
            if step < steps {
                let base = s.base.slice(step * ld..step * ld + rows);
                let logits = s
                    .logits
                    .as_mut()
                    .map(|l| l.slice_mut(step * self.vocab..(step + 1) * self.vocab));
                let step_argmax = StepArgmax {
                    kernels: k,
                    fold,
                    head,
                    vocab: self.vocab,
                    softcap: self.softcap,
                    race: StepRace {
                        race,
                        control,
                        step,
                    },
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

impl Draft {
    /// The base logits and `steps` greedy steps of a DFlash draft's vanilla Markov head
    /// over the block in [`Self::y`]. Step `j` adds the bias `w2 · w1[t]` of the token `t`
    /// before draft `j` to the base logits of row `j + 1` and takes the argmax of the
    /// uncapped sum, as the 2DSD runner does; its confidence is the confidence head's.
    fn markov_dflash(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        shared: SharedWeights<'_>,
        steps: usize,
    ) -> anyhow::Result<()> {
        let Arch::Dflash {
            weights, zero_bias, ..
        } = &self.arch
        else {
            anyhow::bail!("not a DFlash draft");
        };
        let k = &self.kernels;
        let s = &mut self.s;
        let control = &self.inputs.control;
        let head = self.head.as_ref();
        let list = head.and_then(|h| h.shortlist.as_ref());
        let (rows, ld) = base_logits(
            linear,
            &self.linear_rows,
            k,
            stream,
            self.head_copy.as_ref().unwrap_or(shared.lm_head),
            head,
            s,
            steps,
            self.vocab,
            self.padded_vocab,
        )?;
        let (weight, bias) = match head {
            None => (FoldRows::Bf16(&weights.w2), zero_bias),
            Some(h) => (
                match &h.fold {
                    None => FoldRows::Bf16(&weights.w2),
                    Some(Matrix::Bf16(w)) => FoldRows::Bf16(w),
                    Some(Matrix::Fp8(w)) => FoldRows::Fp8(w),
                },
                h.fold_bias.as_ref().unwrap_or(zero_bias),
            ),
        };
        for step in 0..=steps {
            k.dflash_step(
                stream,
                step,
                steps,
                &s.partials,
                control,
                &mut s.ids,
                &weights.w1,
                &s.y,
                &weights.confidence,
                weights.confidence_bias,
                &mut s.latent,
                &mut s.confidence,
            )?;
            if step < steps {
                let base = s.base.slice(step * ld..step * ld + rows);
                k.markov_fold_raw(
                    stream,
                    weight,
                    bias,
                    &base,
                    &s.latent,
                    rows,
                    list.map(|l| &l.count),
                    list.map(|l| &l.map),
                    &mut s.partials,
                )?;
            }
        }
        Ok(())
    }
}

impl Draft {
    /// DFlash 2's drafts over the block in [`Self::y`]: the top 16 tokens of rows 1 to
    /// `steps` through the LM head (or the approximate head), then the selector's greedy
    /// walk from the anchor. A draft's confidence is its softmax probability over its
    /// row's 16 capped scores.
    fn select_dflash2(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        shared: SharedWeights<'_>,
        steps: usize,
    ) -> anyhow::Result<()> {
        let Arch::Dflash2 { weights, .. } = &self.arch else {
            anyhow::bail!("not a DFlash 2 draft");
        };
        let k = &self.kernels;
        let s = &mut self.s;
        let control = &self.inputs.control;
        let head = self.head.as_ref();
        let list = head.and_then(|h| h.shortlist.as_ref());
        let (rows, ld) = base_logits(
            linear,
            &self.linear_rows,
            k,
            stream,
            self.head_copy.as_ref().unwrap_or(shared.lm_head),
            head,
            s,
            steps,
            self.vocab,
            self.padded_vocab,
        )?;
        let d2 = s
            .dflash2
            .as_mut()
            .context("the DFlash 2 buffers are missing")?;
        k.topk(
            stream,
            &s.base.slice(..),
            steps,
            rows,
            ld,
            list.map(|l| &l.count),
            list.map(|l| &l.map),
            &mut d2.topk,
            &mut d2.unary,
            &mut d2.cand,
        )?;
        // At least two rows for the BF16 GEMM, as in base_logits.
        if self.backbone.is_some() {
            let y = s.y.slice(HIDDEN..(steps + 1) * HIDDEN);
            let mut hp = d2.hp.slice_mut(..steps * RANK);
            self.linear_rows.run(
                stream,
                &mut [RowsProduct::bf16(
                    &weights.selector.hidden_projection,
                    &y,
                    &mut hp,
                )],
                steps,
                Start::Early,
            )?;
        } else {
            let m_rows = steps.max(2);
            let y = s.y.slice(HIDDEN..(m_rows + 1) * HIDDEN);
            let mut hp = d2.hp.slice_mut(..m_rows * RANK);
            linear.gemm_bf16(
                stream,
                &mut [Gemm::new(&weights.selector.hidden_projection, &y, &mut hp)],
                m_rows,
            )?;
        }
        k.select(
            stream,
            &d2.cand,
            &d2.unary,
            &d2.hp,
            &weights.selector.predecessor,
            &weights.selector.successor,
            control,
            steps,
            self.softcap,
            &mut s.ids,
            &mut s.confidence,
            &self.inputs.race,
        )
    }
}

/// The input convolution of a DFlash 2 sublayer: `coef = conv.projection · x`, then side 0
/// of the convolution of `x` into `out`.
#[expect(clippy::too_many_arguments, reason = "convolution inputs and kernels")]
fn conv_input(
    linear: &Linear,
    linear_rows: &LinearRows,
    use_rows: bool,
    k: &DraftKernels,
    stream: &CudaStream,
    conv: &ConvWeights,
    x: &CudaSlice<bf16>,
    coef: &mut CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
) -> anyhow::Result<()> {
    if use_rows {
        let x_view = x.slice(..);
        let mut coef_out = coef.slice_mut(..);
        linear_rows.run(
            stream,
            &mut [RowsProduct::bf16(&conv.projection, &x_view, &mut coef_out)],
            BLOCK,
            Start::Early,
        )?;
    } else {
        linear.gemm_bf16(
            stream,
            &mut [Gemm::new(
                &conv.projection,
                &x.slice(..),
                &mut coef.slice_mut(..),
            )],
            BLOCK,
        )?;
    }
    k.conv(stream, x, coef, 0, &conv.base, out)
}

/// The block attention of `layer`: norm and RoPE of the block's queries and keys, the
/// stores of its keys and values in `cache`, and the attention over `cache` into
/// [`Scratch::attn`].
#[expect(clippy::too_many_arguments, reason = "the draft's parts and the layer")]
fn attend(
    k: &DraftKernels,
    stream: &CudaStream,
    arch: &Arch,
    s: &mut Scratch,
    control: &CudaSlice<i32>,
    inv_freq: &CudaSlice<f32>,
    layer: usize,
    cache: &mut KvCache,
) -> anyhow::Result<()> {
    let lw = arch.layer(layer);
    k.block_rope_store(
        stream,
        &mut s.q,
        &s.k,
        &s.v,
        control,
        inv_freq,
        EPS,
        lw.q_norm,
        lw.k_norm,
        cache,
        arch.global(layer),
        arch.rope(),
    )?;
    let (window, row_window) = arch.attention_window(layer);
    let keys = AttentionKeys::Device {
        control,
        window,
        row_window,
        max_keys: if window == 0 {
            cache.slots
        } else {
            (window + BLOCK).min(cache.slots)
        },
    };
    k.attention_from(stream, &s.q, cache, keys, &mut s.attention, &mut s.attn)
}

/// A decoder layer's sublayer, for DFlash 2's convolution coefficients.
#[derive(Clone, Copy)]
enum Sublayer {
    Attention,
    Mlp,
}

impl Scratch {
    /// `residual += x` and `out = rms_norm(residual; weight)`, where `x` is the sublayer's
    /// output (`o` or `mlp`) and `out` is `y` with `last`, else `h`. With `conv`, `x` first
    /// goes through DFlash 2's output convolution with that sublayer's coefficients.
    fn add_norm(
        &mut self,
        k: &DraftKernels,
        stream: &CudaStream,
        sublayer: Sublayer,
        conv: Option<&ConvWeights>,
        weight: &CudaSlice<bf16>,
        last: bool,
    ) -> anyhow::Result<()> {
        let x = match sublayer {
            Sublayer::Attention => &self.o,
            Sublayer::Mlp => &self.mlp,
        };
        let out = if last { &mut self.y } else { &mut self.h };
        let Some(conv) = conv else {
            return k.add_rms_norm(stream, Some(x), &mut self.residual, weight, out, EPS);
        };
        let d2 = self
            .dflash2
            .as_ref()
            .context("the DFlash 2 buffers are missing")?;
        let coef = match sublayer {
            Sublayer::Attention => &d2.coef_attention,
            Sublayer::Mlp => &d2.coef_mlp,
        };
        k.conv_add_rms_norm(
            stream,
            x,
            coef,
            &conv.base,
            &mut self.residual,
            weight,
            out,
            EPS,
        )
    }
}

/// A sublayer's input: its normed input `h`, or with `conv` DFlash 2's input convolution
/// of `h` (see [`conv_input`]) into `d2`'s buffers.
#[expect(clippy::too_many_arguments, reason = "sublayer input selection")]
fn sublayer_input<'a>(
    linear: &Linear,
    linear_rows: &LinearRows,
    use_rows: bool,
    k: &DraftKernels,
    stream: &CudaStream,
    h: &'a CudaSlice<bf16>,
    d2: Option<&'a mut Dflash2Scratch>,
    sublayer: Sublayer,
    conv: Option<&ConvWeights>,
) -> anyhow::Result<CudaView<'a, bf16>> {
    let Some(conv) = conv else {
        return Ok(h.slice(..));
    };
    let d2 = d2.context("the DFlash 2 buffers are missing")?;
    let coef = match sublayer {
        Sublayer::Attention => &mut d2.coef_attention,
        Sublayer::Mlp => &mut d2.coef_mlp,
    };
    conv_input(
        linear,
        linear_rows,
        use_rows,
        k,
        stream,
        conv,
        h,
        coef,
        &mut d2.conv,
    )?;
    Ok(d2.conv.slice(..))
}

/// Rows 1 to `steps` of the block output [`Scratch::y`] through the BF16 LM head
/// `lm_head`, or through the approximate head's rows, into [`Scratch::base`]. Returns the rows of each step's
/// base logits and the elements between steps.
#[expect(clippy::too_many_arguments, reason = "the head and the buffers")]
fn base_logits(
    linear: &Linear,
    linear_rows: &LinearRows,
    k: &DraftKernels,
    stream: &CudaStream,
    lm_head: &CudaSlice<bf16>,
    head: Option<&Head>,
    s: &mut Scratch,
    steps: usize,
    vocab: usize,
    padded_vocab: usize,
) -> anyhow::Result<(usize, usize)> {
    let list = head.and_then(|h| h.shortlist.as_ref());
    // At least two rows for the BF16 GEMMs: a single row would run as a GEMV, which sums
    // in another order.
    let m_rows = steps.max(2);
    // Rows of the base logits, each `ld` elements.
    let (rows, ld) = match list {
        Some(list) => (list.bucket(), list.bucket()),
        None => (vocab, padded_vocab),
    };
    match head.and_then(|h| h.lm.as_ref()) {
        None => {
            let y = s.y.slice(HIDDEN..(m_rows + 1) * HIDDEN);
            let mut base = s.base.slice_mut(..m_rows * ld);
            linear.gemm_bf16(stream, &mut [Gemm::new(lm_head, &y, &mut base)], m_rows)?;
        }
        Some(Matrix::Bf16(w)) => {
            let y = s.y.slice(HIDDEN..(m_rows + 1) * HIDDEN);
            let w = w.slice(..rows * HIDDEN);
            let mut base = s.base.slice_mut(..m_rows * rows);
            linear.gemm_bf16(stream, &mut [Gemm::new(&w, &y, &mut base)], m_rows)?;
        }
        Some(Matrix::Fp8(w)) => {
            let y = s.y.slice(HIDDEN..(steps + 1) * HIDDEN);
            if rows == ld && linear_rows.supports(w.k) {
                let mut base = s.base.slice_mut(..steps * rows);
                linear_rows.run(
                    stream,
                    &mut [RowsProduct::fp8_raw(
                        w.values.slice(..rows * w.k),
                        w.scales.slice(..rows),
                        rows,
                        w.k,
                        &y,
                        &mut base,
                    )],
                    steps,
                    Start::Early,
                )?;
            } else {
                let base = s.base.slice_mut(..);
                k.gemm_fp8(stream, &y, steps, &mut [fp8_rows(w, rows, base, ld)])?;
            }
        }
    }
    Ok((rows, ld))
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

/// Checks that the target's LM head covers the padded vocabulary.
fn check_lm_head(config: &TextConfig, shared: SharedWeights<'_>) -> anyhow::Result<()> {
    ensure!(
        shared.lm_head.len() == config.vocab_size * HIDDEN,
        "the LM head has {} elements, expected {} x {HIDDEN}",
        shared.lm_head.len(),
        config.vocab_size
    );
    Ok(())
}

/// A BF16 projection that [`Arch::release_block_bf16`] may have freed.
fn freed(w: Option<&CudaSlice<bf16>>) -> anyhow::Result<&CudaSlice<bf16>> {
    w.context("a BF16 projection was freed")
}

/// FP8 copies of every layer's projections.
fn quantize_backbone(
    stream: &Arc<CudaStream>,
    kernels: &DraftKernels,
    arch: &Arch,
) -> anyhow::Result<Vec<Fp8Layer>> {
    let quantize = |w: &CudaSlice<bf16>, k: usize| -> anyhow::Result<Fp8Rows> {
        let rows = w.len() / k;
        let mut out = Fp8Rows::new(stream, rows, k)?;
        kernels.quantize_rows(stream, w, None, rows, 0, &mut out)?;
        Ok(out)
    };
    (0..arch.layers())
        .map(|layer| {
            let w = arch.layer(layer);
            let (k_ctx, v_ctx) = match arch {
                Arch::Stock { weights, .. } => {
                    let c = &weights.layers[layer];
                    (
                        Some(quantize(&c.k_ctx, HIDDEN)?),
                        Some(quantize(&c.v_ctx, HIDDEN)?),
                    )
                }
                Arch::Dflash { .. } | Arch::Dflash2 { .. } => (None, None),
            };

            Ok(Fp8Layer {
                q: quantize(freed(w.q)?, HIDDEN)?,
                k: quantize(w.k, HIDDEN)?,
                v: quantize(w.v, HIDDEN)?,
                o: quantize(freed(w.o)?, arch.q_width())?,
                gate: quantize(freed(w.gate)?, HIDDEN)?,
                up: quantize(freed(w.up)?, HIDDEN)?,
                down: quantize(freed(w.down)?, arch.mlp_width())?,
                k_ctx,
                v_ctx,
            })
        })
        .collect()
}

impl Head {
    /// The approximate head `options` ask for, or `None` for the exact one. `fold` holds
    /// the fold weight and bias of a draft with a Markov head.
    fn new(
        stream: &Arc<CudaStream>,
        kernels: &DraftKernels,
        shared: SharedWeights<'_>,
        fold: Option<(&CudaSlice<bf16>, &CudaSlice<bf16>)>,
        vocab: usize,
        options: &DraftOptions,
    ) -> anyhow::Result<Option<Self>> {
        let fp8_fold = options.fp8_fold && fold.is_some();
        let Some(ranking) = &options.shortlist else {
            if !options.fp8_head && !fp8_fold {
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
                fold: fold
                    .filter(|_| fp8_fold)
                    .map(|(w, _)| quantize(w, RANK))
                    .transpose()?,
                fold_bias: None,
                shortlist: None,
            }));
        };
        let shortlist_rows = match options.shortlist_rows {
            Some(n) => validate_shortlist_rows(n)?,
            None => SHORTLIST_ROWS,
        };
        let context_cap = match options.context_rows {
            Some(n) => validate_context_rows(n)?,
            None => CONTEXT_ROWS,
        };
        let mut member = vec![false; vocab];
        let mut taken = 0;
        for &t in ranking {
            let t = t as usize;
            if taken == shortlist_rows {
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
        let capacity = ranked + context_cap;
        let mut map = stream.alloc_zeros::<i32>(capacity)?;
        stream.memcpy_htod(&ids, &mut map.slice_mut(..ranked))?;
        let count = stream.clone_htod(&[i32::try_from(ranked)?])?;
        let mut head = Self {
            lm: Some(if options.fp8_head {
                Matrix::Fp8(Fp8Rows::new(stream, capacity, HIDDEN)?)
            } else {
                Matrix::Bf16(stream.alloc_zeros(capacity * HIDDEN)?)
            }),
            fold: match (fold, fp8_fold) {
                (None, _) => None,
                (Some(_), true) => Some(Matrix::Fp8(Fp8Rows::new(stream, capacity, RANK)?)),
                (Some(_), false) => Some(Matrix::Bf16(stream.alloc_zeros(capacity * RANK)?)),
            },
            fold_bias: fold.map(|_| stream.alloc_zeros(capacity)).transpose()?,
            shortlist: Some(Shortlist {
                ranked,
                context_cap,
                used: ranked,
                member,
                context: Vec::new(),
                map,
                count,
                stale: false,
            }),
        };
        let ids = stream.clone_htod(&ids)?;
        head.gather(stream, kernels, shared, fold, &ids.slice(..), 0)?;
        Ok(Some(head))
    }

    /// Copies the head rows of the tokens in `ids`, and with `fold` the fold and fold-bias
    /// rows, into rows `first..` of the shortlist's tables.
    fn gather(
        &mut self,
        stream: &CudaStream,
        kernels: &DraftKernels,
        shared: SharedWeights<'_>,
        fold: Option<(&CudaSlice<bf16>, &CudaSlice<bf16>)>,
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
        let Some((fold_weight, fold_bias)) = fold else {
            return Ok(());
        };
        if let Some(fold) = self.fold.as_mut() {
            copy(fold_weight, RANK, fold)?;
        }
        if let Some(bias) = self.fold_bias.as_mut() {
            kernels.gather_rows(stream, fold_bias, 1, ids, first, bias)?;
        }
        Ok(())
    }
}

/// One product of [`project`]: the BF16 weight, unless freed, its FP8 copy, if any, and
/// the output.
type Projection<'a> = (
    Option<&'a CudaSlice<bf16>>,
    Option<&'a Fp8Rows>,
    CudaViewMut<'a, bf16>,
);

/// `out = x · weight^T` over the 8 block rows for each product `(weight, fp8, out)`:
/// through the FP8 copies when every product has one, else through the BF16 weights,
/// which must then all be there. As few launches as the kernels' product limits allow.
fn project(
    linear: &Linear,
    linear_rows: &LinearRows,
    kernels: &DraftKernels,
    stream: &CudaStream,
    x: &CudaView<'_, bf16>,
    products: &mut [Projection<'_>],
) -> anyhow::Result<()> {
    if products.iter().all(|p| p.1.is_some()) {
        let width = x.len() / BLOCK;
        if linear_rows.supports(width) {
            for chunk in products.chunks_mut(ROWS_MAX_PRODUCTS) {
                let mut batch: Vec<RowsProduct<'_>> = chunk
                    .iter_mut()
                    .map(|p| {
                        let w = p.1.context("FP8 weight missing")?;
                        Ok(RowsProduct::fp8_raw(
                            w.values.slice(..),
                            w.scales.slice(..),
                            w.rows(),
                            w.k,
                            x,
                            &mut p.2,
                        ))
                    })
                    .collect::<anyhow::Result<_>>()?;
                linear_rows.run(stream, &mut batch, BLOCK, Start::Early)?;
            }
            return Ok(());
        }
        for chunk in products.chunks_mut(F8_MAX_PRODUCTS) {
            let mut fp8: Vec<Fp8Product<'_>> = chunk
                .iter_mut()
                .filter_map(|p| {
                    let ld = p.2.len() / BLOCK;
                    p.1.map(|w| fp8_rows(w, w.rows(), p.2.slice_mut(..), ld))
                })
                .collect();
            kernels.gemm_fp8(stream, x, BLOCK, &mut fp8)?;
        }
        return Ok(());
    }
    for chunk in products.chunks_mut(MAX_PRODUCTS) {
        let mut gemms: Vec<Gemm<'_, bf16>> = chunk
            .iter_mut()
            .map(|p| {
                Ok(Gemm::new(
                    p.0.context("a BF16 projection was freed")?,
                    x,
                    &mut p.2,
                ))
            })
            .collect::<anyhow::Result<_>>()?;
        linear.gemm_bf16(stream, &mut gemms, BLOCK)?;
    }
    Ok(())
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

/// One Markov step's argmax over its base logits and latent, or its coupled pick when
/// the race samples.
struct StepArgmax<'a> {
    kernels: &'a DraftKernels,
    fold: &'a Fold,
    head: Option<&'a Head>,
    vocab: usize,
    softcap: f32,
    race: StepRace<'a>,
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
                Some(self.race),
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
                Some(self.race),
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
            Some(self.race),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortlist_budget_validation() {
        assert_eq!(validate_shortlist_rows(HEAD_BUCKET).unwrap(), HEAD_BUCKET);
        assert_eq!(validate_shortlist_rows(8192).unwrap(), 8192);
        assert_eq!(
            validate_shortlist_rows(SHORTLIST_ROWS).unwrap(),
            SHORTLIST_ROWS
        );
        assert!(validate_shortlist_rows(0).is_err());
        assert!(validate_shortlist_rows(512).is_err());
        assert!(validate_shortlist_rows(8193).is_err());
        assert!(validate_shortlist_rows(SHORTLIST_ROWS + HEAD_BUCKET).is_err());

        assert_eq!(validate_context_rows(0).unwrap(), 0);
        assert_eq!(validate_context_rows(4096).unwrap(), 4096);
        assert_eq!(validate_context_rows(CONTEXT_ROWS).unwrap(), CONTEXT_ROWS);
        assert!(validate_context_rows(512).is_err());
        assert!(validate_context_rows(CONTEXT_ROWS + HEAD_BUCKET).is_err());
    }

    #[test]
    fn parse_shortlist_and_context_specs() {
        let fast = DraftOptions::parse("fast", None).unwrap();
        assert!(fast.shortlist.is_some());
        assert_eq!(fast.shortlist_rows, None);
        assert_eq!(fast.context_rows, None);

        let shrunk = DraftOptions::parse("fast+shortlist=8192+context=4096", None).unwrap();
        assert!(shrunk.fp8_backbone && shrunk.fp8_head && shrunk.fp8_fold && shrunk.fp8_kv);
        assert!(shrunk.shortlist.is_some());
        assert_eq!(shrunk.shortlist_rows, Some(8192));
        assert_eq!(shrunk.context_rows, Some(4096));

        let custom = DraftOptions::parse("shortlist=16384+context=0", Some(&[1, 2, 3])).unwrap();
        assert_eq!(custom.shortlist.as_deref(), Some(&[1, 2, 3][..]));
        assert_eq!(custom.shortlist_rows, Some(16384));
        assert_eq!(custom.context_rows, Some(0));

        let overridden = fast
            .with_shortlist_budget(Some(8192), Some(0))
            .unwrap();
        assert_eq!(overridden.shortlist_rows, Some(8192));
        assert_eq!(overridden.context_rows, Some(0));

        assert!(DraftOptions::parse("fast+shortlist=1000", None).is_err());
        assert!(DraftOptions::parse("fast+context=1000", None).is_err());
        assert!(DraftOptions::parse("fast+shortlist=65536", None).is_err());
        assert!(DraftOptions::parse("fast+context=32768", None).is_err());
    }
}
