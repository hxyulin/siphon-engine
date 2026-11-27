//! Speculative decoding with the DSpark draft: each round verifies the anchor and up to 7
//! drafts in one pass and emits the kept drafts and the target's next token.
//!
//! A round, after the prompt's prefill has committed its taps to the draft and the draft
//! has run its first block:
//!
//! 1. [`Model::start_verify`] runs the anchor and the round's leading drafts as one pass,
//!    of the width [`Width`] chose, and keeps the longest matching run of `a` drafts.
//! 2. The draft commits the taps of the kept rows `0..=a` and runs its block from the new
//!    anchor, both of which the verify left on the device, and drafts the next round.
//! 3. [`Model::finish_verify`] synchronizes once to read the verdict, and the host reads
//!    the new drafts' confidences to choose the next width.
//!
//! Acceptance keeps only drafts equal to the target's own token, and the verify pass gives
//! every row the same result at every width, so the output equals greedy decoding through
//! the verify pass one row at a time, whatever the draft proposes and whatever widths the
//! rounds verify ([`greedy_by_verify`]).
//!
//! With the model sampling ([`Model::set_sampling`]), each verify row draws its token from
//! the target's filtered distribution with the request's seed and the token's position,
//! and a draft is kept when it equals that draw. The draft proposes one token per
//! position, so this is the standard rule for a point-mass proposal: keep the draft with
//! probability p(x), and after a rejection draw from p without it. The output follows the
//! target's distribution exactly and equals sampling through one-row verifies with the
//! same seed (see `docs/engine.md`). The draft's proposal is its coupled pick
//! ([`Draft::set_sampling`]): the best of its logits over T plus the noise of the
//! verify's draw at the same position, which is the target's draw whenever the two
//! distributions are close enough. It changes how often drafts are kept, not the output.
//!
//! When no width is expected to emit tokens faster than plain decoding, a policy with
//! plain prices ([`WidthPolicy::with_plain`]) runs plain steps instead
//! ([`Model::step_with_taps`]), which record the draft's taps. After a run of steps the
//! draft commits their taps and drafts from the new anchor, and the policy decides again.
//! Plain steps sum in other orders than the verify pass, so the output then equals
//! greedy decoding along the same path one token at a time ([`greedy_by_path`]).

use std::{
    ops::ControlFlow,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, CudaView, sys::CUevent_flags};
use half::bf16;
use siphon::Checkpoint;
pub use siphon::tree::{BranchStrategy, DEFAULT_MIN_BRANCH_ENTROPY, SpecTree, SpecTreeConfig};
use siphon::tree::{
    choose_branch_parent, choose_branch_parent_by_entropy, entropy_nats,
    sample_without_replacement, secondary_draw_uniform,
};

mod host;
mod prefix;

pub use host::{DISK_FORMAT, DiskLoaded, DiskSaved, HOST_CHUNK_POSITIONS, ParkSaved};
use prefix::PrefixCache;
pub use prefix::{PrefixHit, PrefixTier};

use crate::{
    draft::{Draft, DraftOptions, STEPS, WINDOW},
    gpu::{PinnedBuf, sync_event},
    model::{
        Accept, AcceptTree, BATCH_VERIFY_MAX_ROWS, BatchVerifyTokens, HybridStepResult,
        HybridVerifyResult, ImageSource, Model, PackedPrefill, PrefillLogits, Sampling, SavedState,
        Verdict, VerifyTokens, VerifyTreeTokens,
    },
    ops::{
        draft::MAX_BATCH_SEQS,
        linear::Linear,
        spec::{RESULT_LEN, Spec, VERIFY_ROWS},
    },
    weights::{
        dflash::DflashWeights,
        dflash2::{DFLASH2_WINDOW, Dflash2Weights, is_dflash2},
        draft::{DraftWeights, HIDDEN},
    },
};

/// Prompt rows whose taps [`Speculator::restart`] commits again: the slots of the draft's
/// rings, the window and a block. DFlash 2's rings, 2056 slots, are the longest.
const TAIL: usize = DFLASH2_WINDOW + STEPS + 1;
const _: () = assert!(TAIL == siphon::memory::TAIL_ROWS);
const _: () = assert!(
    DFLASH2_WINDOW >= WINDOW,
    "TAIL covers the bundled draft's rings"
);

/// A prompt as the speculator prefills it: the tokens the model runs, the key the prefix
/// cache matches it by, and the rows of its images.
///
/// The key has one id per token. For a prompt of text it is the tokens. Two prompts whose
/// images differ have the same placeholder tokens, so the caller replaces each image's
/// placeholders in the key with ids that its image's hash decides, above every token id.
/// The cache saves and compares keys only, so an entry serves a prompt only when every
/// image below its boundary is the same.
pub struct Prompt<'a> {
    pub tokens: &'a [u32],
    pub key: &'a [u32],
    /// Gives the image rows of each chunk ([`Model::prefill_with_images`]); `None` for a
    /// prompt without images.
    pub images: Option<ImageSource<'a>>,
}

impl<'a> Prompt<'a> {
    /// A prompt of text, its tokens as its key.
    #[must_use]
    pub fn text(tokens: &'a [u32]) -> Self {
        Self {
            tokens,
            key: tokens,
            images: None,
        }
    }

    /// The image source for one prefill call of several, borrowed for that call.
    fn images<'b>(images: &'b mut Option<ImageSource<'_>>) -> Option<ImageSource<'b>> {
        match images {
            Some(images) => Some(&mut **images),
            None => None,
        }
    }

    fn check(&self) -> anyhow::Result<()> {
        ensure!(!self.tokens.is_empty(), "nothing to prefill");
        ensure!(
            self.key.len() == self.tokens.len(),
            "the prompt's key has {} ids for {} tokens",
            self.key.len(),
            self.tokens.len()
        );
        Ok(())
    }
}

/// Plain steps in the first run after a draft round prefers them. Each run whose draft
/// round still prefers plain steps doubles the next run, up to [`MAX_PLAIN_RUN`]. One
/// step and a draft round emit a token faster than a verify of one row.
const FIRST_PLAIN_RUN: usize = 1;
const MAX_PLAIN_RUN: usize = 32;
const _: () = assert!(MAX_PLAIN_RUN == siphon::memory::PLAIN_RUN_ROWS);

/// The share of drafts the target kept so far in a generation, for each tenth of the
/// confidence range: the calibration by which a policy with plain prices decides between
/// plain steps and a verify. On prose
/// the bundled draft's confidences run high, so verifies the raw confidences chose ran
/// slower than plain steps. One share over the last 20 or so rounds cost code with the
/// bundled draft 10%: it stayed low after a stretch of new tokens, into the repetitive
/// stretch that followed.
#[derive(Clone, Debug, Default)]
struct Calibration {
    /// Drafts kept and drafts verified, each bin a tenth of the confidence range. A draft
    /// counts as verified when the drafts before it were kept and the round reached it.
    kept: [f64; CALIBRATION_BINS],
    verified: [f64; CALIBRATION_BINS],
    /// The raw confidences of the round being verified.
    round: Vec<f32>,
}

const CALIBRATION_BINS: usize = 10;
/// Drafts each bin starts from, kept at their own confidence.
const CALIBRATION_PRIOR: f64 = 4.0;

impl Calibration {
    fn bin(confidence: f32) -> usize {
        // Truncation picks the tenth; the clamp keeps 1.0 in the last bin.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "0 to 10"
        )]
        let bin = (confidence.clamp(0.0, 1.0) * CALIBRATION_BINS as f32) as usize;
        bin.min(CALIBRATION_BINS - 1)
    }

    /// The confidences with each replaced by its bin's share kept, which starts at the
    /// confidence itself.
    fn apply(&self, confidence: &[f32]) -> Vec<f32> {
        confidence
            .iter()
            .map(|&c| {
                let b = Self::bin(c);
                let share = (self.kept[b] + CALIBRATION_PRIOR * f64::from(c))
                    / (self.verified[b] + CALIBRATION_PRIOR);
                #[expect(clippy::cast_possible_truncation, reason = "a share from 0 to 1")]
                let share = share as f32;
                share
            })
            .collect()
    }

    /// Records a verify of `width` rows that kept `kept` drafts of [`Self::round`].
    fn record(&mut self, width: usize, kept: usize) {
        let reached = (kept + 1).min(width - 1).min(self.round.len());
        for (j, &c) in self.round[..reached].iter().enumerate() {
            let b = Self::bin(c);
            self.verified[b] += 1.0;
            if j < kept {
                self.kept[b] += 1.0;
            }
        }
    }

    /// Records one sequence's verify of `width` rows with confidences `conf` that kept
    /// `kept` drafts, with bounded history so long-running batched servers adapt quickly.
    fn record_bounded(&mut self, conf: &[f32], width: usize, kept: usize) {
        let reached = (kept + 1).min(width.saturating_sub(1)).min(conf.len());
        for (j, &c) in conf[..reached].iter().enumerate() {
            let b = Self::bin(c);
            self.verified[b] += 1.0;
            if j < kept {
                self.kept[b] += 1.0;
            }
            if self.verified[b] > 200.0 {
                self.verified[b] *= 0.5;
                self.kept[b] *= 0.5;
            }
        }
    }
}

/// What the round loop runs next.
#[derive(Clone, Copy, Debug)]
enum Next {
    /// A verify of this width.
    Verify(usize),
    /// Up to this many plain steps, then a draft round that decides again.
    Plain(usize),
}

/// Where a round's drafts come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drafts {
    /// The DSpark draft.
    Dspark,
    /// Uniformly random token ids from a seeded generator, which the target almost never
    /// accepts. The output must not change. An adaptive [`Width`] cycles through its widths.
    Random(u64),
}

/// How many rows a round verifies: the anchor and its leading drafts.
#[derive(Clone, Debug)]
pub enum Width {
    /// The anchor and `k` drafts every round, `k` from 1 to 7.
    Fixed(usize),
    /// A width per round, from the confidences of the round's drafts.
    Adaptive(WidthPolicy),
}

impl Width {
    fn check(&self) -> anyhow::Result<()> {
        if let Self::Fixed(k) = self {
            ensure!((1..=STEPS).contains(k), "k is {k}, expected 1 to {STEPS}");
        }
        Ok(())
    }

    /// The drafts each round proposes.
    fn drafts(&self) -> usize {
        match self {
            Self::Fixed(k) => *k,
            Self::Adaptive(policy) => policy.widest() - 1,
        }
    }
}

/// Picks a round's verify width from its drafts' confidences.
///
/// Draft `j`'s confidence `c_j` is its softmax probability under the draft's Markov
/// logits. On the greedy continuations of three texts, the share of drafts the target
/// accepts after accepting the drafts before them is within a few points of `c_j` at
/// every draft position, so the policy takes `c_j` as that chance. Draft `j` then
/// survives with `s_j = c_1 ... c_j`, and a width `w` emits `1 + s_1 + ... + s_{w-1}`
/// tokens on average. The policy picks the width with the most tokens per millisecond of
/// its round, `round_ms[w - 1]`, which the caller measures.
///
/// With plain prices ([`Plain`]) it may instead pick a run of plain steps, when the run
/// and the draft round after it emit more tokens per millisecond than the best width
/// (see [`Self::decide`]).
#[derive(Clone, Debug)]
pub struct WidthPolicy {
    /// The widths it may pick, ascending, each 1 to 8.
    widths: Vec<usize>,
    /// The time of a round of each width from 1 to 8: verify, draft and host.
    round_ms: [f64; VERIFY_ROWS],
    /// The prices of plain steps, when the speculator may fall back to them.
    plain: Option<Plain>,
}

/// The prices of a run of plain steps, which emits a token per step and ends in a draft
/// round that decides again: `run` steps take `run * step_ms + probe_ms` milliseconds.
#[derive(Clone, Copy, Debug)]
pub struct Plain {
    /// A plain step that records the draft's taps.
    pub step_ms: f64,
    /// The draft's commit of the steps' taps and its round.
    pub probe_ms: f64,
}

impl WidthPolicy {
    /// A policy over `widths` (ascending, 1 to 8, the widest at least 2) that prices a
    /// round of width `w` at `round_ms[w - 1]` milliseconds.
    pub fn new(widths: &[usize], round_ms: [f64; VERIFY_ROWS]) -> anyhow::Result<Self> {
        ensure!(
            widths.windows(2).all(|w| w[0] < w[1])
                && widths.iter().all(|w| (1..=VERIFY_ROWS).contains(w))
                && widths.last().is_some_and(|&w| w >= 2),
            "widths {widths:?} are not ascending from 1 to {VERIFY_ROWS} up to at least 2"
        );
        ensure!(
            widths.iter().all(|&w| round_ms[w - 1] > 0.0),
            "a round time is not positive"
        );
        Ok(Self {
            widths: widths.to_vec(),
            round_ms,
            plain: None,
        })
    }

    /// The policy that also falls back to runs of plain steps priced at `plain` when no
    /// width is expected to emit tokens faster, or never with `None`.
    pub fn with_plain(mut self, plain: Option<Plain>) -> anyhow::Result<Self> {
        ensure!(
            plain.is_none_or(|p| p.step_ms > 0.0 && p.probe_ms >= 0.0),
            "the plain step time is not positive"
        );
        self.plain = plain;
        Ok(self)
    }

    #[must_use]
    pub fn widths(&self) -> &[usize] {
        &self.widths
    }

    /// The time of a round of each width from 1 to 8, in ms.
    #[must_use]
    pub fn round_ms(&self) -> &[f64; VERIFY_ROWS] {
        &self.round_ms
    }

    /// The prices of plain steps, if the policy falls back to them.
    #[must_use]
    pub fn plain(&self) -> Option<Plain> {
        self.plain
    }

    fn widest(&self) -> usize {
        self.widths.last().copied().unwrap_or(2)
    }

    /// The width with the most expected tokens per millisecond for drafts with these
    /// confidences, the first one first.
    #[must_use]
    pub fn choose(&self, confidence: &[f32]) -> usize {
        self.best(confidence).0
    }

    /// `None` for a run of `run` plain steps and the draft round after it, if that is
    /// expected to emit tokens faster than every width by the `calibrated` confidences,
    /// and otherwise the width [`Self::choose`] picks by the raw `confidence`, so that
    /// the rounds verify the widths they would without plain prices.
    #[must_use]
    pub fn decide(&self, confidence: &[f32], calibrated: &[f32], run: usize) -> Option<usize> {
        if let Some(p) = self.plain {
            let run = run.max(1) as f64;
            if self.best(calibrated).1 * (run * p.step_ms + p.probe_ms) < run {
                return None;
            }
        }
        Some(self.choose(confidence))
    }

    /// [`Self::choose`]'s width and its expected tokens per millisecond.
    fn best(&self, confidence: &[f32]) -> (usize, f64) {
        let mut best = (self.widths[0], f64::NEG_INFINITY);
        let mut survival = 1.0;
        let mut tokens = 1.0;
        let mut drafts = 0;
        for &w in &self.widths {
            while drafts + 1 < w {
                let c = confidence.get(drafts).copied().unwrap_or(0.0);
                survival *= f64::from(c.clamp(0.0, 1.0));
                tokens += survival;
                drafts += 1;
            }
            let rate = tokens / self.round_ms[w - 1];
            if rate > best.1 {
                best = (w, rate);
            }
        }
        best
    }
}

/// Counts and times of the rounds since the last [`Speculator::generate`] started.
#[derive(Clone, Debug, Default)]
pub struct SpecStats {
    /// Verify rounds.
    pub rounds: usize,
    /// Tokens emitted, the first token from the prefill excluded.
    pub emitted: usize,
    /// The indices in the output of the tokens that plain steps emitted.
    pub stepped: Vec<usize>,
    /// Draft rounds after plain steps that decided whether to verify again.
    pub probes: usize,
    /// Rounds in which draft `i` was kept, for `i` from 0.
    pub accepted_at: Vec<usize>,
    /// Rounds that verified `w` rows, at index `w - 1`.
    pub widths: [usize; VERIFY_ROWS],
    /// Wall time of the whole generation.
    pub wall: Duration,
    /// GPU time of the verify passes, from the start of each pass to its end.
    pub verify: Duration,
    /// GPU time of the draft's commits and rounds, from the first launch to the end.
    pub draft: Duration,
    /// Wall time of the plain steps.
    pub plain: Duration,
    /// Rounds that verified a 2-branch speculative tree.
    pub tree_rounds: usize,
    /// Tree rounds in which the winning path entered the secondary branch.
    pub secondary_accepted: usize,
}

impl SpecStats {
    /// Mean tokens per verify round, the plain steps' tokens excluded.
    #[must_use]
    pub fn tokens_per_round(&self) -> f64 {
        (self.emitted - self.stepped.len()) as f64 / self.rounds.max(1) as f64
    }

    /// Mean verify rows per round.
    #[must_use]
    pub fn rows_per_round(&self) -> f64 {
        let rows: usize = self
            .widths
            .iter()
            .enumerate()
            .map(|(i, n)| (i + 1) * n)
            .sum();
        rows as f64 / self.rounds.max(1) as f64
    }

    /// The share of rounds that kept draft `i`, unconditionally.
    #[must_use]
    pub fn acceptance(&self) -> Vec<f64> {
        self.accepted_at
            .iter()
            .map(|&n| n as f64 / self.rounds.max(1) as f64)
            .collect()
    }

    /// Wall time outside the verify passes, the draft and the plain steps.
    #[must_use]
    pub fn host(&self) -> Duration {
        self.wall
            .saturating_sub(self.verify + self.draft + self.plain)
    }
}

/// One anchor of [`Speculator::sweep`]: the draft's 7 proposals after it and their
/// confidences.
#[derive(Clone, Debug)]
pub struct SweepRow {
    pub drafts: Vec<u32>,
    pub confidence: Vec<f32>,
}

/// The draft and the round loop for one sequence of a [`Model`].
#[expect(clippy::struct_excessive_bools, reason = "in-flight event flags and coupled mode")]
pub struct Speculator {
    draft: Draft,
    linear: Linear,
    spec: Spec,
    width: Width,
    /// Dedicated copy stream for overlapping verify D2H transfers with draft execution.
    copy_stream: Arc<CudaStream>,
    /// Two ping-pong sets of reusable `[before, verified, drafted]` timing events for
    /// single-stream verify rounds, eliminating per-round CUDA event allocations.
    round_events: [[CudaEvent; 3]; 2],
    round_event_pending: [bool; 2],
    round_event_slot: usize,
    /// Reusable `[before, after]` timing events for plain-step probes.
    probe_events: [CudaEvent; 2],
    probe_event_pending: bool,
    /// Lightweight synchronization events (timing disabled) for cross-stream overlap.
    sync_verified: CudaEvent,
    sync_drafted: CudaEvent,
    results_copied: CudaEvent,
    conf_copied: CudaEvent,
    /// Whether `copy_stream` has an in-flight async D2H of single-stream draft confidences.
    conf_in_flight: bool,
    /// Dedicated event and pinned host buffers for zero-sync single-stream tree draft reads
    /// (`cand`, `q_probs`, `draft_ids`, and winning tree `path`).
    tree_dist_copied: CudaEvent,
    tree_dist_in_flight: bool,
    pinned_tree_path: PinnedBuf<i32>,
    pinned_tree_cand: PinnedBuf<i32>,
    pinned_tree_q: PinnedBuf<f32>,
    pinned_tree_ids: PinnedBuf<i32>,
    /// Persistent page-locked host buffers for single-stream verify results, draft
    /// confidences, and random drafts.
    pinned_verify_result: PinnedBuf<i32>,
    pinned_conf: PinnedBuf<f32>,
    pinned_random: PinnedBuf<i32>,
    random_h2d_done: CudaEvent,
    /// Random drafts of [`Drafts::Random`], uploaded each round.
    random: CudaSlice<i32>,
    /// The taps of the prompt's last rows, up to [`TAIL`] of them, oldest first, for
    /// [`Self::restart`]; `spare` takes the next copy while one is built.
    tail: CudaSlice<bf16>,
    spare: CudaSlice<bf16>,
    tail_rows: usize,
    /// The taps of the current run of plain steps, up to [`MAX_PLAIN_RUN`] rows.
    plain_taps: CudaSlice<bf16>,
    calibration: Calibration,
    /// Elements of one position's taps.
    row_len: usize,
    prompt_len: usize,
    stats: SpecStats,
    /// The sequence the draft follows through batched steps, if any.
    batched: Option<Batched>,
    /// The prefix cache, with [`Self::enable_prefix_cache`].
    prefix: Option<PrefixCache>,
    /// The sequence whose committed positions the draft's global layers hold: changes
    /// whenever the draft starts a sequence from position 0.
    draft_sequence: u64,
    /// Called after each verify round, for measurements ([`Self::set_round_hook`]).
    round_hook: Option<RoundHook>,
    /// Whether a sampling request drafts coupled picks ([`Self::set_coupled_drafts`]).
    coupled: bool,
    /// Rows of the prefill chunks [`Self::rebuild`] runs ([`Self::set_prefill_chunk`]).
    prefill_chunk: usize,
    /// Pre-allocated batched speculation buffers (packed token cache slots + pinned D2H).
    batch_spec: Option<BatchSpecBuffers>,
    /// Running per-confidence-bin acceptance calibration for batched DFlash 2 width choice.
    batch_calibration: Calibration,
    /// Configuration for opt-in 2-branch speculative tree verification (disabled by default).
    tree_config: SpecTreeConfig,
}

impl Drop for Speculator {
    fn drop(&mut self) {
        let _ = self.copy_stream.synchronize();
        let _ = self.random_h2d_done.synchronize();
    }
}

/// Number of pre-allocated packed-token cache slots for active batched micro-batches.
const BATCH_DRAFT_CACHE_SLOTS: usize = 4;

/// One pre-allocated packed verify-token slot for batched DFlash 2.
struct BatchDraftedSlot {
    slots: Vec<usize>,
    width: usize,
    /// Pre-allocated device scratch (`BATCH_VERIFY_MAX_ROWS` ints) reused across cache misses.
    tokens: CudaSlice<i32>,
    /// Cached 16-sparse draft candidates and filtered probabilities (`[MAX_BATCH_SEQS * 7 * 16]`)
    /// for batched `--spec-beta` rejection sampling.
    cand: CudaSlice<i32>,
    q_probs: CudaSlice<f32>,
    has_dist: bool,
    confidences: Vec<f32>,
    valid: bool,
    tick: u64,
}

/// In-flight batched draft whose confidences are copying to `pinned_conf` on `copy_stream`
/// while the host processes the current round's verdicts and returns them to the scheduler.
struct PendingBatchDraft {
    slots: Vec<usize>,
    max_width: usize,
    slot_idx: usize,
}

/// Pre-allocated device and pinned-host buffers for zero-sync batched speculation.
struct BatchSpecBuffers {
    cache: [BatchDraftedSlot; BATCH_DRAFT_CACHE_SLOTS],
    tick: u64,
    /// Persistent page-locked buffer for batched verify results `[MAX_BATCH_SEQS * RESULT_LEN]`.
    pinned_results: PinnedBuf<i32>,
    /// Persistent page-locked buffer for batched draft confidences `[MAX_BATCH_SEQS * STEPS]`.
    pinned_conf: PinnedBuf<f32>,
    /// Pending async draft from the previous `speculate_batch_round`.
    pending: Option<PendingBatchDraft>,
}

impl BatchSpecBuffers {
    fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let mut slots_vec = Vec::with_capacity(BATCH_DRAFT_CACHE_SLOTS);
        let dist_elems =
            crate::ops::draft::MAX_BATCH_SEQS * STEPS * crate::ops::sample::BETA_CAND_K;
        for _ in 0..BATCH_DRAFT_CACHE_SLOTS {
            slots_vec.push(BatchDraftedSlot {
                slots: Vec::new(),
                width: 0,
                tokens: stream.alloc_zeros(BATCH_VERIFY_MAX_ROWS)?,
                cand: stream.alloc_zeros(dist_elems)?,
                q_probs: stream.alloc_zeros(dist_elems)?,
                has_dist: false,
                confidences: Vec::new(),
                valid: false,
                tick: 0,
            });
        }
        let cache: [BatchDraftedSlot; BATCH_DRAFT_CACHE_SLOTS] = slots_vec
            .try_into()
            .ok()
            .context("exact batch draft cache slot count")?;
        Ok(Self {
            cache,
            tick: 0,
            pinned_results: PinnedBuf::zeros(stream, MAX_BATCH_SEQS * RESULT_LEN)?,
            pinned_conf: PinnedBuf::zeros(stream, MAX_BATCH_SEQS * STEPS)?,
            pending: None,
        })
    }

    fn alloc_slot(&mut self) -> usize {
        if let Some(i) = self.cache.iter().position(|s| !s.valid) {
            return i;
        }
        self.cache
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.tick)
            .map_or(0, |(i, _)| i)
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }
}

/// Relative round time model for batched DFlash 2 (`seqs` sequences x `width` verify rows),
/// fitted to measured Blackwell `verify_batch + batch_commit_and_draft` timings.
fn batch_round_cost_ms(seqs: usize, width: usize) -> f64 {
    5.4 + 0.165 * (seqs * width) as f64 + 0.10 * seqs as f64
}

/// What a verify round saw, for [`Speculator::set_round_hook`].
pub struct RoundRecord<'a> {
    /// The anchor's position: verify row `j` draws the token at `start + j + 1`.
    pub start: usize,
    /// The drafts the round verified, after the anchor.
    pub drafts: &'a [u32],
    /// The draft's logits for each of `drafts`, F32 `[drafts, vocab]`, after
    /// [`Speculator::record_draft_logits`]; empty otherwise.
    pub draft_logits: &'a [f32],
    /// The target's logits for each verified row, the anchor's first, F32
    /// `[drafts + 1, vocab]`.
    pub target_logits: &'a [f32],
    /// Each verified row's token, greedy or drawn, the anchor's row first.
    pub drawn: &'a [u32],
    /// Drafts kept.
    pub accepted: usize,
}

/// A callback that sees every verify round.
pub type RoundHook = Box<dyn FnMut(&RoundRecord<'_>) -> anyhow::Result<()> + Send>;

/// A sequence that left speculation for batched steps ([`Speculator::follow`]): the draft
/// holds its state below `start`, and `plain_taps` the taps of the `tokens.len()` rows
/// from `start` on, which emitted `tokens`.
#[derive(Debug)]
struct Batched {
    start: usize,
    tokens: Vec<u32>,
}

/// A rebuild of the draft's state in chunks, from [`Speculator::start_rebuild`].
#[derive(Debug)]
pub struct Rebuild {
    /// The slot it prefills in.
    slot: usize,
    /// [`Model::slot_sequence`] of the slot after its reset.
    slot_sequence: u64,
    /// The draft sequence ([`Speculator::draft_sequence`]) it builds.
    sequence: u64,
    /// The key of the positions it prefilled.
    key: Vec<u32>,
}

impl Rebuild {
    /// The slot it prefills in.
    #[must_use]
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// The key of the positions it prefilled, from position 0.
    #[must_use]
    pub fn key(&self) -> &[u32] {
        &self.key
    }
}

/// Which draft a [`Speculator`] runs.
#[derive(Clone, Copy, Debug)]
pub enum DraftSource<'a> {
    /// The DSpark draft bundled with the checkpoint (`mtp.safetensors`).
    Bundled,
    /// A drafter in this directory: Inco AI's DFlash 2 drafter (see [`Dflash2Weights`])
    /// when its `config.json` names `DFlash2DraftModel`, else a DFlash draft with DSpark
    /// heads exported by the 2DSD project (see [`DflashWeights`]).
    Dflash(&'a std::path::Path),
}

/// Loads the draft `source` names for `model`, with room for `capacity` positions and the
/// approximations in `options`, and makes the model record the draft's taps.
pub fn load_draft(
    model: &mut Model,
    checkpoint: &Checkpoint,
    source: DraftSource<'_>,
    capacity: usize,
    options: &DraftOptions,
) -> anyhow::Result<Draft> {
    let stream = Arc::clone(model.stream());
    let draft = match source {
        DraftSource::Bundled => {
            let weights = DraftWeights::load(checkpoint, model.config(), &stream)?;
            Draft::with_options(
                &stream,
                model.config(),
                weights,
                model.draft_shared(),
                capacity + STEPS + 1,
                options,
            )?
        }
        DraftSource::Dflash(dir) if is_dflash2(dir)? => {
            let weights = Dflash2Weights::load(dir, model.config(), &stream)?;
            Draft::dflash2(
                &stream,
                model.config(),
                weights,
                model.draft_shared(),
                capacity + STEPS + 1,
                options,
            )?
        }
        DraftSource::Dflash(dir) => {
            let weights = DflashWeights::load(dir, model.config(), &stream)?;
            Draft::dflash(
                &stream,
                model.config(),
                weights,
                model.draft_shared(),
                capacity + STEPS + 1,
                options,
            )?
        }
    };
    model.set_tap_layout(draft.tap_layout())?;
    Ok(draft)
}

impl Speculator {
    /// Sets DFlash 2's draft temperature calibration multiplier `alpha = T_draft / T_target`
    /// (`> 0.0`, default `0.85`).
    pub fn set_draft_temp_scale(&mut self, scale: f32) -> anyhow::Result<()> {
        self.draft.set_temp_scale(scale)
    }

    /// Loads the draft from `checkpoint` for `model`, with room for `capacity` positions,
    /// verifying `k` drafts per round (1 to 7), with the approximations of
    /// [`DraftOptions::fast`].
    pub fn new(
        model: &Model,
        checkpoint: &Checkpoint,
        capacity: usize,
        k: usize,
    ) -> anyhow::Result<Self> {
        Self::with_options(
            model,
            checkpoint,
            capacity,
            Width::Fixed(k),
            &DraftOptions::fast(),
        )
    }

    /// [`Self::new`] with a [`Width`] and the draft's approximations in `options`.
    pub fn with_options(
        model: &Model,
        checkpoint: &Checkpoint,
        capacity: usize,
        width: Width,
        options: &DraftOptions,
    ) -> anyhow::Result<Self> {
        width.check()?;
        let stream = model.stream();
        let weights = DraftWeights::load(checkpoint, model.config(), stream)?;
        let draft = Draft::with_options(
            stream,
            model.config(),
            weights,
            model.draft_shared(),
            capacity + STEPS + 1,
            options,
        )?;
        Self::with_draft(model, draft, width)
    }

    /// A speculator around `draft` (see [`load_draft`]), whose taps `model` must record.
    pub fn with_draft(model: &Model, draft: Draft, width: Width) -> anyhow::Result<Self> {
        width.check()?;
        ensure!(
            *model.tap_layout() == draft.tap_layout(),
            "the model records taps {:?}, the draft reads {:?}",
            model.tap_layout(),
            draft.tap_layout()
        );
        let stream = model.stream();
        let row_len = draft.tap_layout().len() * HIDDEN;
        let copy_stream = stream.context().new_stream()?;
        let round_events = [
            [
                timing_event(stream)?,
                timing_event(stream)?,
                timing_event(stream)?,
            ],
            [
                timing_event(stream)?,
                timing_event(stream)?,
                timing_event(stream)?,
            ],
        ];
        let probe_events = [timing_event(stream)?, timing_event(stream)?];
        Ok(Self {
            draft,
            linear: Linear::load(stream.context())?,
            spec: Spec::load(stream.context())?,
            width,
            copy_stream,
            round_events,
            round_event_pending: [false; 2],
            round_event_slot: 0,
            probe_events,
            probe_event_pending: false,
            sync_verified: sync_event(stream)?,
            sync_drafted: sync_event(stream)?,
            results_copied: sync_event(stream)?,
            conf_copied: sync_event(stream)?,
            conf_in_flight: false,
            tree_dist_copied: sync_event(stream)?,
            tree_dist_in_flight: false,
            pinned_tree_path: PinnedBuf::zeros(stream, VERIFY_ROWS)?,
            pinned_tree_cand: PinnedBuf::zeros(
                stream,
                STEPS * crate::ops::sample::BETA_CAND_K,
            )?,
            pinned_tree_q: PinnedBuf::zeros(
                stream,
                STEPS * crate::ops::sample::BETA_CAND_K,
            )?,
            pinned_tree_ids: PinnedBuf::zeros(stream, STEPS)?,
            pinned_verify_result: PinnedBuf::zeros(stream, RESULT_LEN)?,
            pinned_conf: PinnedBuf::zeros(stream, STEPS)?,
            pinned_random: PinnedBuf::zeros(stream, STEPS)?,
            random_h2d_done: sync_event(stream)?,
            random: stream.alloc_zeros(STEPS)?,
            tail: stream.alloc_zeros(TAIL * row_len)?,
            spare: stream.alloc_zeros(TAIL * row_len)?,
            tail_rows: 0,
            plain_taps: stream.alloc_zeros(MAX_PLAIN_RUN * row_len)?,
            calibration: Calibration::default(),
            row_len,
            prompt_len: 0,
            stats: SpecStats::default(),
            batched: None,
            prefix: None,
            draft_sequence: 0,
            round_hook: None,
            coupled: true,
            prefill_chunk: crate::model::DEFAULT_PREFILL_CHUNK,
            batch_spec: None,
            batch_calibration: Calibration::default(),
            tree_config: SpecTreeConfig::default(),
        })
    }

    /// Enables or configures 2-branch speculative tree verification (disabled by default so
    /// linear speculation stays bit-for-bit untouched).
    pub fn set_spec_tree(&mut self, config: SpecTreeConfig) -> anyhow::Result<()> {
        if config.enabled {
            config.check()?;
        }
        self.draft.set_tree_top2(config.enabled);
        self.tree_config = config;
        Ok(())
    }

    /// Current 2-branch speculative tree configuration.
    #[must_use]
    pub fn spec_tree(&self) -> &SpecTreeConfig {
        &self.tree_config
    }

    /// Whether this speculator's drafter supports batched speculation (DFlash 2).
    #[must_use]
    pub fn supports_batch_spec(&self) -> bool {
        self.draft.is_dflash2()
    }

    /// Maximum sequences one batched speculation call verifies at once.
    #[must_use]
    pub fn max_batch_spec_seqs(&self) -> usize {
        MAX_BATCH_SEQS
    }

    /// Ensures the pre-allocated batched speculation scratch and pinned host buffers exist.
    fn ensure_batch_spec(&mut self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        if self.batch_spec.is_none() {
            self.batch_spec = Some(BatchSpecBuffers::new(stream)?);
        }
        Ok(())
    }

    /// Drops any pre-drafted batched DFlash 2 verify tokens without freeing their
    /// pre-allocated device buffers, waiting for any in-flight async confidence D2H first.
    pub fn invalidate_batch_draft(&mut self) {
        if let Some(bs) = self.batch_spec.as_mut() {
            if bs.pending.take().is_some() {
                let _ = self.conf_copied.synchronize();
            }
            for slot in &mut bs.cache {
                slot.valid = false;
            }
        }
    }

    /// Returns the mean calibrated draft confidence of `slot` from the most recent
    /// pre-drafted batched speculation state, or `None` if `slot` has no cached draft
    /// confidences yet.
    #[must_use]
    pub fn batch_slot_confidence(&self, slot: usize) -> Option<f32> {
        let bs = self.batch_spec.as_ref()?;
        let mut best: Option<(u64, &[f32])> = None;
        for entry in &bs.cache {
            if !entry.valid {
                continue;
            }
            if let Some(pos) = entry.slots.iter().position(|&s| s == slot) {
                let start = pos * STEPS;
                let end = start + STEPS;
                if let Some(row) = entry.confidences.get(start..end)
                    && best.is_none_or(|(tick, _)| entry.tick >= tick)
                {
                    best = Some((entry.tick, row));
                }
            }
        }
        let (_, row) = best?;
        let cal = self.batch_calibration.apply(row);
        let sum: f32 = cal.iter().copied().sum();
        Some(sum / STEPS as f32)
    }

    /// Pre-allocates batched DFlash 2 rings, packed verify-token scratch slots, and
    /// page-locked host buffers for up to `max_seqs` concurrent sequences and slots
    /// `0..=max_slot`.
    pub fn reserve_batch_spec(
        &mut self,
        model: &Model,
        max_seqs: usize,
        max_slot: usize,
    ) -> anyhow::Result<()> {
        self.ensure_batch_spec(model.stream())?;
        self.draft
            .ensure_batch_dflash2(model.stream(), max_seqs.min(MAX_BATCH_SEQS), max_slot)
    }

    /// Number of captured `(single_stream, batched)` draft CUDA graphs currently cached.
    #[must_use]
    pub fn cached_draft_graph_counts(&self) -> (usize, usize) {
        self.draft.cached_graph_counts()
    }

    /// Pre-captures every bucketed `(seqs, width)` target verify and DFlash 2 draft CUDA
    /// graph pair (plus initial `commit_width = 0` draft graphs) for `1..=max_seqs` during
    /// startup warmup. Combined with bucket-padding in `Model::start_verify_batch` and
    /// `Draft::batch_*`, this eliminates live `cuGraphInstantiateWithFlags` stalls for every
    /// concurrency level `1..=max_seqs` and adaptive verify width `{2, 4, 6, 8}`.
    /// Returns the number of `(seqs, width)` verify+draft graph pairs captured.
    #[expect(
        clippy::too_many_lines,
        reason = "pre-captures bucketed batched verify + draft graph pairs"
    )]
    pub fn warmup_batch_spec(
        &mut self,
        model: &mut Model,
        max_seqs: usize,
        token: u32,
    ) -> anyhow::Result<usize> {
        ensure!(
            self.supports_batch_spec(),
            "batched speculation warmup needs a DFlash 2 drafter"
        );
        let max_seqs = max_seqs
            .clamp(1, self.max_batch_spec_seqs())
            .min(model.slots());
        let stream = Arc::clone(model.stream());
        self.ensure_batch_spec(&stream)?;
        self.invalidate_batch_draft();
        for slot in 0..max_seqs {
            model.reset_slot(slot)?;
            self.draft.reset_slot(slot);
        }
        // Populate 1 committed position per slot in one packed forward pass so every
        // `batch_draft_initial` / `verify_batch` bucket has valid non-empty context.
        let tok_slice = [token];
        let slices: Vec<PackedPrefill<'_>> = (0..max_seqs)
            .map(|slot| PackedPrefill {
                slot,
                tokens: &tok_slice,
                images: Vec::new(),
                need_first_token: false,
                sampling: Sampling::GREEDY,
            })
            .collect();
        let flags = vec![true; max_seqs];
        self.prefill_packed_batch(model, &slices, &flags, &flags, max_seqs)?;

        let mut captured = 0usize;
        for seqs in siphon::memory::batch_seq_buckets(max_seqs) {
            let slots: Vec<usize> = (0..seqs).collect();
            let anchors = vec![token; seqs];
            let samplings = vec![Sampling::GREEDY; seqs];
            let accept = vec![Accept::Rows(1); seqs];
            self.draft.batch_draft_initial(
                &stream,
                model.draft_shared(),
                &slots,
                &anchors,
                &samplings,
                STEPS,
            )?;
            let max_w = (BATCH_VERIFY_MAX_ROWS / seqs).min(VERIFY_ROWS);
            for w in siphon::memory::BATCH_SPEC_WIDTHS {
                if w > max_w {
                    continue;
                }
                let starts: Vec<usize> = slots
                    .iter()
                    .map(|&s| model.slot_position(s))
                    .collect::<anyhow::Result<_>>()?;
                {
                    let src = self.draft.batch_verify_tokens(seqs, VERIFY_ROWS)?;
                    let spec_beta = model.spec_beta().is_some();
                    let bs = self
                        .batch_spec
                        .as_mut()
                        .context("batch spec buffers missing")?;
                    self.spec.pack_verify_tokens_batch(
                        &stream,
                        &src,
                        &mut bs.cache[0].tokens.slice_mut(..seqs * w),
                        seqs,
                        w,
                    )?;
                    if spec_beta {
                        let (src_cand, src_q) =
                            self.draft.batch_dflash2_distributions(seqs, STEPS)?;
                        let dist_n = src_cand.len();
                        stream.memcpy_dtod(&src_cand, &mut bs.cache[0].cand.slice_mut(..dist_n))?;
                        stream.memcpy_dtod(&src_q, &mut bs.cache[0].q_probs.slice_mut(..dist_n))?;
                    }
                    bs.cache[0].has_dist = spec_beta;
                }
                if model.spec_beta().is_some() {
                    let dist_n = seqs * STEPS * crate::ops::sample::BETA_CAND_K;
                    let bs = self
                        .batch_spec
                        .as_ref()
                        .context("batch spec buffers missing")?;
                    model.set_batch_verify_draft_dist(
                        seqs,
                        w,
                        &bs.cache[0].cand.slice(..dist_n),
                        &bs.cache[0].q_probs.slice(..dist_n),
                    )?;
                }
                let pending = {
                    let bs = self
                        .batch_spec
                        .as_ref()
                        .context("batch spec buffers missing")?;
                    let tokens = bs.cache[0].tokens.slice(..seqs * w);
                    model.start_verify_batch(
                        &slots,
                        BatchVerifyTokens::Device { tokens, width: w },
                        &accept,
                        &samplings,
                    )?
                };
                self.draft.batch_commit_and_draft(
                    &stream,
                    model.draft_shared(),
                    &slots,
                    &starts,
                    &model.batch_verify_taps_padded()?,
                    &model.batch_verify_results_padded()?,
                    &model.batch_verify_anchors_padded()?,
                    &samplings,
                    w,
                    STEPS,
                )?;
                let verdicts = model.finish_verify_batch(pending)?;
                for (&slot, (&start, v)) in slots.iter().zip(starts.iter().zip(&verdicts)) {
                    self.draft.set_slot_committed(slot, start + v.accepted + 1)?;
                }
                captured += 1;
            }
        }
        stream.synchronize()?;
        for slot in 0..max_seqs {
            model.reset_slot(slot)?;
            self.draft.reset_slot(slot);
        }
        self.draft.reset();
        self.invalidate_batch_draft();
        Ok(captured)
    }

    /// Synchronizes slot 0's host-tracked DFlash 2 committed position with `model.position()`
    /// after single-stream speculation stops.
    pub fn sync_slot0_draft(&mut self, model: &Model) -> anyhow::Result<()> {
        self.invalidate_batch_draft();
        self.draft.set_slot_committed(0, model.position())
    }

    /// Returns slot `slot`'s DFlash 2 committed position.
    #[must_use]
    pub fn slot_draft_committed(&self, slot: usize) -> usize {
        self.draft.slot_committed(slot)
    }

    /// Copies slot `from`'s DFlash 2 rings and committed position into slot `to`.
    pub fn copy_slot_draft(
        &mut self,
        model: &Model,
        from: usize,
        to: usize,
    ) -> anyhow::Result<()> {
        self.invalidate_batch_draft();
        self.draft.copy_slot_rings(model.stream(), from, to)
    }

    /// Moves a DFlash 2-ready sequence from slot `from` to slot 0 and prepares it to resume
    /// single-stream speculation from `anchor` without rebuilding the draft.
    pub fn resume_dflash2_slot(
        &mut self,
        model: &mut Model,
        from: usize,
        anchor: u32,
    ) -> anyhow::Result<()> {
        self.invalidate_batch_draft();
        let pos = model.slot_position(from)?;
        ensure!(
            self.draft.slot_committed(from) == pos,
            "slot {from}'s DFlash 2 rings are at {}, model is at {pos}",
            self.draft.slot_committed(from)
        );
        self.move_to_slot0(model, from)?;
        if from != 0 {
            self.draft.copy_slot_rings(model.stream(), from, 0)?;
        }
        model.set_verify_anchor(anchor)
    }

    /// Prefills `tokens` in `slot` with feature taps and commits them into `slot`'s DFlash 2
    /// sliding rings. When `reset_draft` is true, resets `slot`'s DFlash 2 committed count
    /// first.
    pub fn prefill_slot_batch(
        &mut self,
        model: &mut Model,
        slot: usize,
        tokens: &[u32],
        images: Option<ImageSource<'_>>,
        chunk: usize,
        reset_draft: bool,
    ) -> anyhow::Result<u32> {
        self.invalidate_batch_draft();
        let stream = Arc::clone(model.stream());
        if reset_draft {
            self.draft.reset_slot(slot);
        }
        // Note a bounded tail of each prompt chunk for shortlist heads without thrashing the
        // shared context table across many concurrent streams.
        let noted_cap = (self.draft.context_capacity() / 2).min(256);
        let latest: Vec<i32> = tokens
            .iter()
            .rev()
            .take(noted_cap)
            .map(|&t| i32::try_from(t))
            .collect::<Result<_, _>>()?;
        if !latest.is_empty() {
            self.draft
                .note_tokens(&stream, model.draft_shared(), &latest)?;
        }
        let Self { draft, linear, .. } = self;
        let mut commit = |start: usize, taps: CudaView<'_, bf16>| {
            draft.commit_slot(&stream, linear, slot, &taps, start)
        };
        match images {
            Some(images) => model.prefill_slot_with_images_and_taps(
                slot,
                tokens,
                chunk,
                Some(&mut commit),
                images,
            ),
            None => model.prefill_slot_with_taps(slot, tokens, chunk, &mut commit),
        }
    }

    /// Prefills multiple independent prompt slices packed into a single pass of up to `chunk`
    /// total rows, committing feature taps into each DFlash 2-ready slot's sliding rings.
    pub fn prefill_packed_batch(
        &mut self,
        model: &mut Model,
        slices: &[PackedPrefill<'_>],
        commit_draft: &[bool],
        reset_draft: &[bool],
        chunk: usize,
    ) -> anyhow::Result<Vec<Option<u32>>> {
        ensure!(
            commit_draft.len() == slices.len() && reset_draft.len() == slices.len(),
            "mismatched packed prefill draft flags"
        );
        self.invalidate_batch_draft();
        let stream = Arc::clone(model.stream());
        let any_draft = commit_draft.iter().any(|&c| c);
        if any_draft {
            let mut noted = Vec::new();
            let noted_cap = (self.draft.context_capacity() / 2).min(256);
            let per_slice = if noted_cap == 0 {
                0
            } else {
                (noted_cap / slices.len().max(1)).max(32).min(noted_cap)
            };
            for ((s, &commit), &reset) in slices.iter().zip(commit_draft).zip(reset_draft) {
                if !commit {
                    continue;
                }
                if reset {
                    self.draft.reset_slot(s.slot);
                }
                for &t in s.tokens.iter().rev().take(per_slice) {
                    noted.push(i32::try_from(t)?);
                }
            }
            if !noted.is_empty() {
                self.draft
                    .note_tokens(&stream, model.draft_shared(), &noted)?;
            }
            let Self { draft, linear, .. } = self;
            let mut commit_fn = |slot: usize, start: usize, taps: CudaView<'_, bf16>| {
                if let Some(pos) = slices.iter().position(|s| s.slot == slot)
                    && commit_draft[pos]
                {
                    draft.commit_slot(&stream, linear, slot, &taps, start)?;
                }
                Ok(())
            };
            model.prefill_packed_with_taps(slices, chunk, Some(&mut commit_fn))
        } else {
            model.prefill_packed(slices, chunk)
        }
    }

    /// Runs a hybrid plain-decode + piggybacked-prefill pass via [`Model::step_hybrid`],
    /// committing feature taps into each DFlash 2-ready prefill slot's sliding rings.
    #[expect(
        clippy::too_many_arguments,
        reason = "batched decode rows + piggybacked prefill slices + draft flags"
    )]
    pub fn step_hybrid_batch(
        &mut self,
        model: &mut Model,
        decode_rows: &[(usize, u32)],
        decode_sampling: &[Sampling],
        prefill_slices: &[PackedPrefill<'_>],
        commit_prefill_draft: &[bool],
        reset_prefill_draft: &[bool],
        chunk: usize,
    ) -> anyhow::Result<HybridStepResult> {
        ensure!(
            commit_prefill_draft.len() == prefill_slices.len()
                && reset_prefill_draft.len() == prefill_slices.len(),
            "mismatched hybrid prefill draft flags"
        );
        self.invalidate_batch_draft();
        let stream = Arc::clone(model.stream());
        let any_draft = commit_prefill_draft.iter().any(|&c| c);
        if any_draft {
            let mut noted = Vec::new();
            let noted_cap = (self.draft.context_capacity() / 2).min(256);
            let per_slice = if noted_cap == 0 {
                0
            } else {
                (noted_cap / prefill_slices.len().max(1))
                    .max(32)
                    .min(noted_cap)
            };
            for ((s, &commit), &reset) in prefill_slices
                .iter()
                .zip(commit_prefill_draft)
                .zip(reset_prefill_draft)
            {
                if !commit {
                    continue;
                }
                if reset {
                    self.draft.reset_slot(s.slot);
                }
                for &t in s.tokens.iter().rev().take(per_slice) {
                    noted.push(i32::try_from(t)?);
                }
            }
            if !noted.is_empty() {
                self.draft
                    .note_tokens(&stream, model.draft_shared(), &noted)?;
            }
            let Self { draft, linear, .. } = self;
            let mut commit_fn = |slot: usize, start: usize, taps: CudaView<'_, bf16>| {
                if let Some(pos) = prefill_slices.iter().position(|s| s.slot == slot)
                    && commit_prefill_draft[pos]
                {
                    draft.commit_slot(&stream, linear, slot, &taps, start)?;
                }
                Ok(())
            };
            model.step_hybrid_with_taps(
                decode_rows,
                decode_sampling,
                prefill_slices,
                chunk,
                Some(&mut commit_fn),
            )
        } else {
            model.step_hybrid(decode_rows, decode_sampling, prefill_slices, chunk)
        }
    }

    /// Runs one batched DFlash 2 speculation round for `verify_slots` while piggybacking
    /// `prefill_slices` onto the same target forward pass, preserving the pre-drafted
    /// `batch_spec` pipeline for `verify_slots` and committing DFlash 2 taps for each
    /// draft-ready prefill slice on the same stream.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "pipelined adaptive batch verify + piggybacked prefill"
    )]
    pub fn speculate_hybrid_round(
        &mut self,
        model: &mut Model,
        verify_slots: &[usize],
        anchors: &[u32],
        verify_samplings: &[Sampling],
        max_width: usize,
        prefill_slices: &[PackedPrefill<'_>],
        commit_prefill_draft: &[bool],
        reset_prefill_draft: &[bool],
        chunk: usize,
    ) -> anyhow::Result<HybridVerifyResult> {
        let seqs = verify_slots.len();
        ensure!(
            (1..=MAX_BATCH_SEQS).contains(&seqs),
            "batched speculation takes 1 to {MAX_BATCH_SEQS} sequences, not {seqs}"
        );
        ensure!(
            (2..=VERIFY_ROWS).contains(&max_width) && seqs * max_width <= BATCH_VERIFY_MAX_ROWS,
            "invalid batched verify shape {seqs} x {max_width}"
        );
        ensure!(
            anchors.len() == seqs && verify_samplings.len() == seqs,
            "mismatched batched speculation lengths"
        );
        ensure!(
            commit_prefill_draft.len() == prefill_slices.len()
                && reset_prefill_draft.len() == prefill_slices.len(),
            "mismatched hybrid prefill draft flags"
        );
        let stream = Arc::clone(model.stream());
        let spec_beta = model.spec_beta().is_some();
        self.ensure_batch_spec(&stream)?;
        self.resolve_pending_batch_draft(&stream, verify_slots, max_width, spec_beta)?;
        self.draft
            .set_fp8_dense(model.prefill_dense() == crate::ops::Products::Fp8)?;
        let starts: Vec<usize> = verify_slots
            .iter()
            .map(|&s| model.slot_position(s))
            .collect::<anyhow::Result<_>>()?;
        let draft_samplings: Vec<Sampling> = if self.coupled && model.sampling_enabled() {
            verify_samplings.to_vec()
        } else {
            vec![Sampling::GREEDY; seqs]
        };
        let cached_idx = self.batch_spec.as_ref().and_then(|bs| {
            bs.cache.iter().position(|e| {
                e.valid
                    && e.slots.as_slice() == verify_slots
                    && e.width <= max_width
                    && (!spec_beta || e.has_dist)
            })
        });
        let accept = vec![Accept::Greedy; seqs];
        let (verify_w, used_conf, slot_idx) = if let Some(idx) = cached_idx {
            let bs = self
                .batch_spec
                .as_mut()
                .context("batch spec buffers missing")?;
            let tick = bs.next_tick();
            let entry = &mut bs.cache[idx];
            entry.tick = tick;
            (entry.width, entry.confidences.clone(), idx)
        } else {
            self.draft.batch_draft_initial(
                &stream,
                model.draft_shared(),
                verify_slots,
                anchors,
                &draft_samplings,
                STEPS,
            )?;
            self.sync_drafted.record(&stream)?;
            self.copy_stream.wait(&self.sync_drafted)?;
            {
                let bs = self
                    .batch_spec
                    .as_mut()
                    .context("batch spec buffers missing")?;
                bs.pinned_conf.copy_from_device(
                    &self.copy_stream,
                    &self.draft.batch_confidences_device(seqs, STEPS)?,
                )?;
            }
            self.conf_copied.record(&self.copy_stream)?;
            self.conf_copied.synchronize()?;
            let init_conf = {
                let bs = self
                    .batch_spec
                    .as_ref()
                    .context("batch spec buffers missing")?;
                bs.pinned_conf.as_slice()[..seqs * STEPS].to_vec()
            };
            let w = self.choose_batch_width(seqs, &init_conf, max_width);
            let src = self.draft.batch_verify_tokens(seqs, VERIFY_ROWS)?;
            let bs = self
                .batch_spec
                .as_mut()
                .context("batch spec buffers missing")?;
            let idx = bs.alloc_slot();
            let tick = bs.next_tick();
            let entry = &mut bs.cache[idx];
            self.spec.pack_verify_tokens_batch(
                &stream,
                &src,
                &mut entry.tokens.slice_mut(..seqs * w),
                seqs,
                w,
            )?;
            if spec_beta {
                let (src_cand, src_q) = self.draft.batch_dflash2_distributions(seqs, STEPS)?;
                let dist_n = src_cand.len();
                stream.memcpy_dtod(&src_cand, &mut entry.cand.slice_mut(..dist_n))?;
                stream.memcpy_dtod(&src_q, &mut entry.q_probs.slice_mut(..dist_n))?;
            }
            entry.has_dist = spec_beta;
            entry.slots = verify_slots.to_vec();
            entry.width = w;
            entry.confidences.clone_from(&init_conf);
            entry.valid = true;
            entry.tick = tick;
            (w, init_conf, idx)
        };
        let any_prefill_draft = commit_prefill_draft.iter().any(|&c| c);
        if any_prefill_draft {
            for ((s, &commit), &reset) in prefill_slices
                .iter()
                .zip(commit_prefill_draft)
                .zip(reset_prefill_draft)
            {
                if commit && reset {
                    self.draft.reset_slot(s.slot);
                }
            }
        }
        if model.spec_beta().is_some() {
            let dist_n = seqs * STEPS * crate::ops::sample::BETA_CAND_K;
            let bs = self
                .batch_spec
                .as_ref()
                .context("batch spec buffers missing")?;
            model.set_batch_verify_draft_dist(
                seqs,
                verify_w,
                &bs.cache[slot_idx].cand.slice(..dist_n),
                &bs.cache[slot_idx].q_probs.slice(..dist_n),
            )?;
        }
        let pending = {
            let Self {
                draft,
                linear,
                batch_spec,
                ..
            } = self;
            let bs = batch_spec
                .as_ref()
                .context("batch spec buffers missing")?;
            let tokens = bs.cache[slot_idx].tokens.slice(..seqs * verify_w);
            if any_prefill_draft {
                let mut commit_fn = |slot: usize, start: usize, taps: CudaView<'_, bf16>| {
                    if let Some(pos) = prefill_slices.iter().position(|s| s.slot == slot)
                        && commit_prefill_draft[pos]
                    {
                        draft.commit_slot(&stream, linear, slot, &taps, start)?;
                    }
                    Ok(())
                };
                model.start_verify_hybrid(
                    verify_slots,
                    BatchVerifyTokens::Device {
                        tokens,
                        width: verify_w,
                    },
                    &accept,
                    verify_samplings,
                    prefill_slices,
                    chunk,
                    Some(&mut commit_fn),
                )?
            } else {
                model.start_verify_hybrid(
                    verify_slots,
                    BatchVerifyTokens::Device {
                        tokens,
                        width: verify_w,
                    },
                    &accept,
                    verify_samplings,
                    prefill_slices,
                    chunk,
                    None,
                )?
            }
        };
        // Pipeline commit + next-round draft for `verify_slots` on the GPU before syncing.
        self.draft.batch_commit_and_draft(
            &stream,
            model.draft_shared(),
            verify_slots,
            &starts,
            &model.batch_verify_taps()?,
            &model.batch_verify_results()?,
            &model.batch_verify_anchors()?,
            &draft_samplings,
            verify_w,
            STEPS,
        )?;
        self.sync_drafted.record(&stream)?;
        self.copy_stream.wait(&self.sync_drafted)?;
        {
            let bs = self
                .batch_spec
                .as_mut()
                .context("batch spec buffers missing")?;
            bs.pinned_conf.copy_from_device(
                &self.copy_stream,
                &self.draft.batch_confidences_device(seqs, STEPS)?,
            )?;
        }
        self.conf_copied.record(&self.copy_stream)?;
        // Single host synchronization for the whole hybrid verify + prefill round.
        let result = model.finish_verify_hybrid(pending)?;
        for (s, (&slot, (&start, v))) in verify_slots
            .iter()
            .zip(starts.iter().zip(&result.verdicts))
            .enumerate()
        {
            self.draft.set_slot_committed(slot, start + v.accepted + 1)?;
            self.batch_calibration.record_bounded(
                &used_conf[s * STEPS..(s + 1) * STEPS],
                verify_w,
                v.accepted,
            );
        }
        let mut noted: Vec<i32> = result
            .verdicts
            .iter()
            .flat_map(Verdict::emitted)
            .map(i32::try_from)
            .collect::<Result<_, _>>()?;
        if any_prefill_draft {
            let noted_cap = (self.draft.context_capacity() / 2).min(256);
            let per_slice = if noted_cap == 0 {
                0
            } else {
                (noted_cap / prefill_slices.len().max(1))
                    .max(32)
                    .min(noted_cap)
            };
            for (s, &commit) in prefill_slices.iter().zip(commit_prefill_draft) {
                if !commit {
                    continue;
                }
                for &t in s.tokens.iter().rev().take(per_slice) {
                    noted.push(i32::try_from(t)?);
                }
            }
        }
        if !noted.is_empty() {
            self.draft
                .note_tokens(&stream, model.draft_shared(), &noted)?;
        }
        if let Some(bs) = self.batch_spec.as_mut() {
            bs.cache[slot_idx].valid = false;
            bs.pending = Some(PendingBatchDraft {
                slots: verify_slots.to_vec(),
                max_width,
                slot_idx,
            });
        }
        Ok(result)
    }

    /// Chooses the verify width in `[2, 4, 6, 8]` (at most `max_width`) that maximizes
    /// expected tokens per millisecond for a batch of `seqs` sequences given their calibrated
    /// per-position draft confidences `[seqs, STEPS]`.
    fn choose_batch_width(&self, seqs: usize, confidences: &[f32], max_width: usize) -> usize {
        let mut best_w = 2usize.min(max_width);
        let mut best_rate = f64::NEG_INFINITY;
        for &w in &[2usize, 4, 6, 8] {
            if w > max_width {
                continue;
            }
            let mut total_tok = 0.0f64;
            for s in 0..seqs {
                let row = &confidences[s * STEPS..(s + 1) * STEPS];
                let cal = self.batch_calibration.apply(row);
                let mut surv = 1.0f64;
                let mut tok = 1.0f64;
                for &c in &cal[..w - 1] {
                    surv *= f64::from(c.clamp(0.0, 1.0));
                    tok += surv;
                }
                total_tok += tok;
            }
            let mean_tok = total_tok / seqs as f64;
            let rate = mean_tok / batch_round_cost_ms(seqs, w);
            if rate > best_rate {
                best_rate = rate;
                best_w = w;
            }
        }
        best_w
    }

    /// Finishes any in-flight batched draft from the previous round: waits for `conf_copied`
    /// on `copy_stream`, chooses the next verify width from the pinned confidences, and packs
    /// `[seqs, VERIFY_ROWS]` drafted tokens into `[seqs, next_w]` on the device via
    /// `pack_verify_tokens_batch`.
    fn resolve_pending_batch_draft(
        &mut self,
        stream: &Arc<CudaStream>,
        next_slots: &[usize],
        next_max_width: usize,
        spec_beta: bool,
    ) -> anyhow::Result<()> {
        let Some(pending) = self
            .batch_spec
            .as_mut()
            .and_then(|bs| bs.pending.take())
        else {
            return Ok(());
        };
        let seqs = pending.slots.len();
        self.conf_copied.synchronize()?;
        let next_conf = {
            let bs = self
                .batch_spec
                .as_ref()
                .context("batch spec buffers missing")?;
            bs.pinned_conf.as_slice()[..seqs * STEPS].to_vec()
        };
        let eff_max = if pending.slots.as_slice() == next_slots {
            pending.max_width.min(next_max_width)
        } else {
            pending.max_width
        };
        let next_w = self.choose_batch_width(seqs, &next_conf, eff_max);
        let next_tokens = self.draft.batch_verify_tokens(seqs, VERIFY_ROWS)?;
        let bs = self
            .batch_spec
            .as_mut()
            .context("batch spec buffers missing")?;
        let tick = bs.next_tick();
        let entry = &mut bs.cache[pending.slot_idx];
        self.spec.pack_verify_tokens_batch(
            stream,
            &next_tokens,
            &mut entry.tokens.slice_mut(..seqs * next_w),
            seqs,
            next_w,
        )?;
        if spec_beta {
            let (src_cand, src_q) = self.draft.batch_dflash2_distributions(seqs, STEPS)?;
            let dist_n = src_cand.len();
            stream.memcpy_dtod(&src_cand, &mut entry.cand.slice_mut(..dist_n))?;
            stream.memcpy_dtod(&src_q, &mut entry.q_probs.slice_mut(..dist_n))?;
        }
        entry.has_dist = spec_beta;
        entry.slots = pending.slots;
        entry.width = next_w;
        entry.confidences = next_conf;
        entry.valid = true;
        entry.tick = tick;
        Ok(())
    }

    /// Runs one batched DFlash 2 speculation round for `slots` (`1..=32`) of up to `max_width`
    /// rows per sequence (`2..=8`, `seqs * max_width <= 256`), adapting the verify width from
    /// calibrated per-position confidences and pipelining `Model::start_verify_batch` with
    /// `Draft::batch_commit_and_draft`, while overlapping verify D2H and host verdict/scheduler
    /// work with draft execution on a dedicated copy stream.
    #[expect(clippy::too_many_lines, reason = "pipelined adaptive batch verify + draft")]
    pub fn speculate_batch_round(
        &mut self,
        model: &mut Model,
        slots: &[usize],
        anchors: &[u32],
        samplings: &[Sampling],
        max_width: usize,
    ) -> anyhow::Result<Vec<Verdict>> {
        let seqs = slots.len();
        ensure!(
            (1..=MAX_BATCH_SEQS).contains(&seqs),
            "batched speculation takes 1 to {MAX_BATCH_SEQS} sequences, not {seqs}"
        );
        ensure!(
            (2..=VERIFY_ROWS).contains(&max_width) && seqs * max_width <= BATCH_VERIFY_MAX_ROWS,
            "invalid batched verify shape {seqs} x {max_width}"
        );
        ensure!(
            anchors.len() == seqs && samplings.len() == seqs,
            "mismatched batched speculation lengths"
        );
        let stream = Arc::clone(model.stream());
        let spec_beta = model.spec_beta().is_some();
        self.ensure_batch_spec(&stream)?;
        self.resolve_pending_batch_draft(&stream, slots, max_width, spec_beta)?;
        self.draft
            .set_fp8_dense(model.prefill_dense() == crate::ops::Products::Fp8)?;
        let starts: Vec<usize> = slots
            .iter()
            .map(|&s| model.slot_position(s))
            .collect::<anyhow::Result<_>>()?;
        let draft_samplings: Vec<Sampling> = if self.coupled && model.sampling_enabled() {
            samplings.to_vec()
        } else {
            vec![Sampling::GREEDY; seqs]
        };
        let cached_idx = self.batch_spec.as_ref().and_then(|bs| {
            bs.cache.iter().position(|e| {
                e.valid
                    && e.slots.as_slice() == slots
                    && e.width <= max_width
                    && (!spec_beta || e.has_dist)
            })
        });
        let accept = vec![Accept::Greedy; seqs];
        let (verify_w, used_conf, slot_idx) = if let Some(idx) = cached_idx {
            let bs = self
                .batch_spec
                .as_mut()
                .context("batch spec buffers missing")?;
            let tick = bs.next_tick();
            let entry = &mut bs.cache[idx];
            entry.tick = tick;
            (entry.width, entry.confidences.clone(), idx)
        } else {
            self.draft.batch_draft_initial(
                &stream,
                model.draft_shared(),
                slots,
                anchors,
                &draft_samplings,
                STEPS,
            )?;
            self.sync_drafted.record(&stream)?;
            self.copy_stream.wait(&self.sync_drafted)?;
            {
                let bs = self
                    .batch_spec
                    .as_mut()
                    .context("batch spec buffers missing")?;
                bs.pinned_conf.copy_from_device(
                    &self.copy_stream,
                    &self.draft.batch_confidences_device(seqs, STEPS)?,
                )?;
            }
            self.conf_copied.record(&self.copy_stream)?;
            self.conf_copied.synchronize()?;
            let init_conf = {
                let bs = self
                    .batch_spec
                    .as_ref()
                    .context("batch spec buffers missing")?;
                bs.pinned_conf.as_slice()[..seqs * STEPS].to_vec()
            };
            let w = self.choose_batch_width(seqs, &init_conf, max_width);
            let src = self.draft.batch_verify_tokens(seqs, VERIFY_ROWS)?;
            let bs = self
                .batch_spec
                .as_mut()
                .context("batch spec buffers missing")?;
            let idx = bs.alloc_slot();
            let tick = bs.next_tick();
            let entry = &mut bs.cache[idx];
            self.spec.pack_verify_tokens_batch(
                &stream,
                &src,
                &mut entry.tokens.slice_mut(..seqs * w),
                seqs,
                w,
            )?;
            if spec_beta {
                let (src_cand, src_q) = self.draft.batch_dflash2_distributions(seqs, STEPS)?;
                let dist_n = src_cand.len();
                stream.memcpy_dtod(&src_cand, &mut entry.cand.slice_mut(..dist_n))?;
                stream.memcpy_dtod(&src_q, &mut entry.q_probs.slice_mut(..dist_n))?;
            }
            entry.has_dist = spec_beta;
            entry.slots = slots.to_vec();
            entry.width = w;
            entry.confidences.clone_from(&init_conf);
            entry.valid = true;
            entry.tick = tick;
            (w, init_conf, idx)
        };
        if model.spec_beta().is_some() {
            let dist_n = seqs * STEPS * crate::ops::sample::BETA_CAND_K;
            let bs = self
                .batch_spec
                .as_ref()
                .context("batch spec buffers missing")?;
            model.set_batch_verify_draft_dist(
                seqs,
                verify_w,
                &bs.cache[slot_idx].cand.slice(..dist_n),
                &bs.cache[slot_idx].q_probs.slice(..dist_n),
            )?;
        }
        let pending = {
            let bs = self
                .batch_spec
                .as_ref()
                .context("batch spec buffers missing")?;
            let tokens = bs.cache[slot_idx].tokens.slice(..seqs * verify_w);
            model.start_verify_batch(
                slots,
                BatchVerifyTokens::Device {
                    tokens,
                    width: verify_w,
                },
                &accept,
                samplings,
            )?
        };
        // Record `sync_verified` right after verify_batch and kick off async pinned D2H of
        // verify results on `copy_stream` before launching the draft graph.
        self.sync_verified.record(&stream)?;
        self.copy_stream.wait(&self.sync_verified)?;
        {
            let bs = self
                .batch_spec
                .as_mut()
                .context("batch spec buffers missing")?;
            bs.pinned_results
                .copy_from_device(&self.copy_stream, &model.batch_verify_results()?)?;
        }
        self.results_copied.record(&self.copy_stream)?;
        // Immediately pipeline commit + next-round draft (always 7 drafts) on the main stream.
        self.draft.batch_commit_and_draft(
            &stream,
            model.draft_shared(),
            slots,
            &starts,
            &model.batch_verify_taps_padded()?,
            &model.batch_verify_results_padded()?,
            &model.batch_verify_anchors_padded()?,
            &draft_samplings,
            verify_w,
            STEPS,
        )?;
        // Queue async pinned D2H of next round's confidences on `copy_stream` right after draft.
        self.sync_drafted.record(&stream)?;
        self.copy_stream.wait(&self.sync_drafted)?;
        {
            let bs = self
                .batch_spec
                .as_mut()
                .context("batch spec buffers missing")?;
            bs.pinned_conf.copy_from_device(
                &self.copy_stream,
                &self.draft.batch_confidences_device(seqs, STEPS)?,
            )?;
        }
        self.conf_copied.record(&self.copy_stream)?;
        // Wait only for verify results (not draft!), so host verdict parsing, calibration,
        // `note_tokens` enqueue, and engine token streaming overlap with the draft graph.
        self.results_copied.synchronize()?;
        let verdicts = {
            let bs = self
                .batch_spec
                .as_ref()
                .context("batch spec buffers missing")?;
            model.complete_verify_batch(pending, &bs.pinned_results.as_slice()[..seqs * RESULT_LEN])?
        };
        for (s, (&slot, (&start, v))) in slots
            .iter()
            .zip(starts.iter().zip(&verdicts))
            .enumerate()
        {
            self.draft.set_slot_committed(slot, start + v.accepted + 1)?;
            self.batch_calibration.record_bounded(
                &used_conf[s * STEPS..(s + 1) * STEPS],
                verify_w,
                v.accepted,
            );
        }
        let emitted_ids: Vec<i32> = verdicts
            .iter()
            .flat_map(Verdict::emitted)
            .map(i32::try_from)
            .collect::<Result<_, _>>()?;
        if !emitted_ids.is_empty() {
            self.draft
                .note_tokens(&stream, model.draft_shared(), &emitted_ids)?;
        }
        if let Some(bs) = self.batch_spec.as_mut() {
            bs.cache[slot_idx].valid = false;
            bs.pending = Some(PendingBatchDraft {
                slots: slots.to_vec(),
                max_width,
                slot_idx,
            });
        }
        Ok(verdicts)
    }

    /// The rows of the prefill chunks the caller runs, and that [`Self::rebuild`] runs:
    /// 1 to [`crate::model::MAX_PREFILL_CHUNK`], [`crate::model::DEFAULT_PREFILL_CHUNK`]
    /// until set. Prefix cache entries are found by the chunk they were saved with.
    pub fn set_prefill_chunk(&mut self, rows: usize) -> anyhow::Result<()> {
        ensure!(
            (1..=crate::model::MAX_PREFILL_CHUNK).contains(&rows),
            "a prefill chunk of {rows} rows, expected 1 to {}",
            crate::model::MAX_PREFILL_CHUNK
        );
        self.prefill_chunk = rows;
        Ok(())
    }

    /// The rows of [`Self::set_prefill_chunk`].
    #[must_use]
    pub fn prefill_chunk(&self) -> usize {
        self.prefill_chunk
    }

    /// Calls `hook` after each verify round with the round's drafts and logits, or stops
    /// with `None`. Each round then synchronizes twice more and copies the logits to the
    /// host, so this is for measurements only.
    pub fn set_round_hook(&mut self, hook: Option<RoundHook>) {
        self.round_hook = hook;
    }

    /// Whether a sampling request's drafts take the coupled pick (the default) or the
    /// draft's argmax, from the next generation on. For measurements: the output is the
    /// same either way, only acceptance changes.
    pub fn set_coupled_drafts(&mut self, coupled: bool) {
        self.coupled = coupled;
    }

    /// Gives the draft the model's sampling, which the verify uses, or greedy when the
    /// verify does not sample or coupled drafts are off.
    fn sync_draft_sampling(&mut self, model: &Model) -> anyhow::Result<()> {
        let sampling = if self.coupled && model.sampling_enabled() {
            model.sampling()
        } else {
            Sampling::GREEDY
        };
        self.draft.set_sampling(model.stream(), sampling)
    }

    /// Makes the draft keep its logits from the next round on, which
    /// [`RoundRecord::draft_logits`] then holds. Only the bundled draft without an
    /// approximate head records them ([`Draft::record_logits`]).
    pub fn record_draft_logits(&mut self, model: &Model) -> anyhow::Result<()> {
        self.draft.record_logits(model.stream())
    }

    #[must_use]
    pub fn width(&self) -> &Width {
        &self.width
    }

    /// Verifies rounds of `width` from now on.
    pub fn set_width(&mut self, width: Width) -> anyhow::Result<()> {
        width.check()?;
        self.width = width;
        Ok(())
    }

    #[must_use]
    pub fn stats(&self) -> &SpecStats {
        &self.stats
    }

    /// Prefills `prompt` from position 0 in chunks of `chunk` rows, commits every row's
    /// taps to the draft and returns the greedy next token, the first anchor.
    pub fn prefill(
        &mut self,
        model: &mut Model,
        prompt: &[u32],
        chunk: usize,
    ) -> anyhow::Result<u32> {
        ensure!(
            model.position() == 0,
            "the speculator prefills from position 0"
        );
        self.prefill_whole(model, Prompt::text(prompt), chunk)
    }

    /// [`Self::prefill`] of a [`Prompt`].
    fn prefill_whole(
        &mut self,
        model: &mut Model,
        prompt: Prompt<'_>,
        chunk: usize,
    ) -> anyhow::Result<u32> {
        prompt.check()?;
        self.fence_host()?;
        // The draft's global caches now take another sequence.
        self.draft_sequence += 1;
        let tokens = prompt.tokens;
        self.start_prompt(model, tokens)?;
        let first = self.prefill_rows(model, tokens, prompt.images, chunk)?;
        self.end_prompt(model, tokens.len(), first)
    }

    /// Readies the draft for a new prompt: forgets its committed positions and the
    /// sequence it followed, and notes the prompt's last tokens for a shortlist head.
    fn start_prompt(&mut self, model: &Model, prompt: &[u32]) -> anyhow::Result<()> {
        self.invalidate_batch_draft();
        self.draft.reset();
        self.batched = None;
        self.tail_rows = 0;
        self.prompt_len = 0;
        // A shortlist head takes the tokens of the prompt's last positions, latest first,
        // and keeps half of its context budget free for the tokens the rounds emit.
        let noted = self.draft.context_capacity() / 2;
        if noted == 0 {
            return Ok(());
        }
        let latest: Vec<i32> = prompt
            .iter()
            .rev()
            .take(noted)
            .map(|&t| i32::try_from(t))
            .collect::<Result<_, _>>()?;
        self.draft
            .note_tokens(model.stream(), model.draft_shared(), &latest)
    }

    /// Prefills `tokens` from the model's position in chunks of `chunk` rows, commits every
    /// row's taps to the draft and keeps the last [`TAIL`] rows' taps. Returns the greedy
    /// next token.
    fn prefill_rows(
        &mut self,
        model: &mut Model,
        tokens: &[u32],
        images: Option<ImageSource<'_>>,
        chunk: usize,
    ) -> anyhow::Result<u32> {
        let stream = Arc::clone(model.stream());
        let Self {
            draft,
            linear,
            tail,
            spare,
            tail_rows,
            row_len,
            ..
        } = self;
        let row = *row_len;
        let mut commit = |start: usize, taps: CudaView<'_, bf16>| {
            draft.commit(&stream, linear, &taps, start)?;
            // Keep the last TAIL rows: the newest of the old tail, then this chunk's.
            let rows = taps.len() / row;
            let new = rows.min(TAIL);
            let old = (*tail_rows).min(TAIL - new);
            if old > 0 {
                stream.memcpy_dtod(
                    &tail.slice((*tail_rows - old) * row..*tail_rows * row),
                    &mut spare.slice_mut(..old * row),
                )?;
            }
            stream.memcpy_dtod(
                &taps.slice((rows - new) * row..),
                &mut spare.slice_mut(old * row..(old + new) * row),
            )?;
            std::mem::swap(tail, spare);
            *tail_rows = old + new;
            Ok(())
        };
        match images {
            Some(images) => model.prefill_with_images(
                tokens,
                chunk,
                PrefillLogits::Last,
                Some(&mut commit),
                images,
            ),
            None => model.prefill_with_taps(tokens, chunk, PrefillLogits::Last, &mut commit),
        }
    }

    /// Ends a prompt of `len` tokens whose prefill returned `first`: makes it the anchor.
    fn end_prompt(&mut self, model: &mut Model, len: usize, first: u32) -> anyhow::Result<u32> {
        model.set_verify_anchor(first)?;
        self.prompt_len = len;
        Ok(first)
    }

    /// Returns the draft to the end of the prompt of the last [`Self::prefill`], with
    /// `first` as the anchor again, after the model's state was restored to that point
    /// ([`Model::restore_state`]). Rounds after the prompt overwrite the ring slots of
    /// the prompt's last positions in the draft's ring layers, but nothing of its global
    /// layers below the prompt's end, so committing the prompt's last [`TAIL`] rows again
    /// restores the draft. A shortlist head keeps the tokens noted since the prefill.
    pub fn restart(&mut self, model: &mut Model, first: u32) -> anyhow::Result<()> {
        ensure!(self.prompt_len > 0, "nothing was prefilled");
        ensure!(
            self.tail_rows == TAIL.min(self.prompt_len),
            "the draft kept {} of the prompt's last rows, too few to restart after a \
             cached prefill",
            self.tail_rows
        );
        ensure!(
            model.position() == self.prompt_len,
            "the model is at {}, the prompt ends at {}",
            model.position(),
            self.prompt_len
        );
        let stream = Arc::clone(model.stream());
        let row = self.row_len;
        self.draft.commit(
            &stream,
            &self.linear,
            &self.tail.slice(..self.tail_rows * row),
            self.prompt_len - self.tail_rows,
        )?;
        model.set_verify_anchor(first)
    }

    /// Follows the sequence in slot 0 into batched steps ([`Model::step_batch`]) from its
    /// position, through which the draft holds its state after a round or a run of plain
    /// steps. [`Self::keep_batched`] then keeps the taps of its rows, and
    /// [`Self::resume`] returns it to speculation.
    pub fn follow(&mut self, model: &Model) {
        self.batched = Some(Batched {
            start: model.position(),
            tokens: Vec::new(),
        });
    }

    /// [`Draft::skip_to_for_benchmark`]: the draft takes every position below `position`
    /// as committed, for benchmarks at long contexts.
    pub fn skip_to_for_benchmark(&mut self, position: usize) -> anyhow::Result<()> {
        self.draft.skip_to_for_benchmark(position)
    }

    /// Stops following a sequence, whose draft state is then lost.
    pub fn forget(&mut self) {
        self.batched = None;
    }

    /// Whether the draft follows a sequence ([`Self::follow`]).
    #[must_use]
    pub fn following(&self) -> bool {
        self.batched.is_some()
    }

    /// After a batched step in which the followed sequence was row `row` and emitted
    /// `token`: keeps the row's taps, which batched steps must record
    /// ([`Model::set_batch_taps`]). Commits them to the draft every [`MAX_PLAIN_RUN`] rows.
    /// Each row of a batched step equals a one-row verify, so the draft gets the taps that
    /// speculation would have committed.
    pub fn keep_batched(&mut self, model: &Model, row: usize, token: u32) -> anyhow::Result<()> {
        let row_len = self.row_len;
        let b = self
            .batched
            .as_mut()
            .context("the draft follows no sequence")?;
        let taps = model.batch_taps(row)?;
        ensure!(
            taps.len() == row_len,
            "the batched step's taps differ from the draft's"
        );
        let at = b.tokens.len() * row_len;
        model
            .stream()
            .memcpy_dtod(&taps, &mut self.plain_taps.slice_mut(at..at + row_len))?;
        b.tokens.push(token);
        if b.tokens.len() == MAX_PLAIN_RUN {
            let start = b.start;
            b.start += MAX_PLAIN_RUN;
            let tokens = std::mem::take(&mut b.tokens);
            self.commit_plain(&Arc::clone(model.stream()), model, start, &tokens)?;
        }
        Ok(())
    }

    /// Returns the followed sequence to speculation: commits the taps of its batched rows
    /// and makes `anchor`, its next token, the verify's anchor. The sequence must be in
    /// slot 0 ([`Model::move_slot`]) at the position after its last kept row.
    /// [`Self::generate_with`] then continues it.
    pub fn resume(&mut self, model: &mut Model, anchor: u32) -> anyhow::Result<()> {
        let b = self
            .batched
            .take()
            .context("the draft follows no sequence")?;
        ensure!(
            model.position() == b.start + b.tokens.len(),
            "the model is at {}, the draft's rows end at {}",
            model.position(),
            b.start + b.tokens.len()
        );
        self.commit_plain(&Arc::clone(model.stream()), model, b.start, &b.tokens)?;
        model.set_verify_anchor(anchor)
    }

    /// Builds the draft's state for a sequence that ran without it in slot `from`, whose
    /// positions 0 .. `tokens.len()` took `tokens`, in one go: [`Self::start_rebuild`] in
    /// slot `slot`, whose sequence is lost, [`Self::rebuild_step`] until the draft holds
    /// every position, then [`Self::follow_rebuilt`]. The draft then follows the sequence
    /// from the end of `tokens`, for [`Self::resume`].
    pub fn rebuild(
        &mut self,
        model: &mut Model,
        slot: usize,
        from: usize,
        mut tokens: Prompt<'_>,
    ) -> anyhow::Result<()> {
        ensure!(
            slot != from,
            "the draft is rebuilt in another slot than the sequence's"
        );
        tokens.check()?;
        let mut rebuild = self.start_rebuild(model, slot)?;
        loop {
            let step = Prompt {
                tokens: tokens.tokens,
                key: tokens.key,
                images: Prompt::images(&mut tokens.images),
            };
            if self.rebuild_step(model, &mut rebuild, step)? {
                break;
            }
        }
        self.follow_rebuilt(model, rebuild, tokens.tokens)
    }

    /// Starts building the draft's state for a sequence that ran without it, in chunks
    /// that other work may come between ([`Self::rebuild_step`]): resets slot `slot`, whose
    /// sequence is lost, for the rebuild's prefill, and readies the draft for another
    /// sequence. Entries whose draft state lies in the draft's global caches lose it.
    pub fn start_rebuild(&mut self, model: &mut Model, slot: usize) -> anyhow::Result<Rebuild> {
        ensure!(!self.following(), "the draft follows a sequence");
        self.fence_host()?;
        model.reset_slot(slot)?;
        // The draft's global caches now take another sequence.
        self.draft_sequence += 1;
        self.invalidate_batch_draft();
        self.draft.reset();
        self.tail_rows = 0;
        self.prompt_len = 0;
        Ok(Rebuild {
            slot,
            slot_sequence: model.slot_sequence(slot)?,
            sequence: self.draft_sequence,
            key: Vec::new(),
        })
    }

    /// Whether `rebuild` can go on: nothing else used its slot or the draft since its last
    /// step.
    #[must_use]
    pub fn rebuild_live(&self, model: &Model, rebuild: &Rebuild) -> bool {
        let done = rebuild.key.len();
        rebuild.sequence == self.draft_sequence
            && !self.following()
            && self.draft.committed() == done
            && model.slot_sequence(rebuild.slot).ok() == Some(rebuild.slot_sequence)
            && model.slot_position(rebuild.slot).ok() == Some(done)
    }

    /// The position the next [`Self::rebuild_step`] of `rebuild` for a sequence whose
    /// positions took `key` prefills up to: a chunk more, or the boundary of a prefix cache
    /// entry that `key` starts with, or the end of `key`.
    #[must_use]
    pub fn rebuild_end(&self, model: &Model, rebuild: &Rebuild, key: &[u32]) -> usize {
        let done = rebuild.key.len();
        let mut end = (done + self.prefill_chunk).min(key.len());
        if let Some(boundary) = self.entry_after(model, key, done, self.prefill_chunk) {
            end = end.min(boundary);
        }
        end
    }

    /// Pages of the KV pool that the next [`Self::rebuild_step`] of `rebuild` maps for
    /// `key`, in its slot and the draft.
    #[must_use]
    pub fn rebuild_pages(&self, model: &Model, rebuild: &Rebuild, key: &[u32]) -> usize {
        let end = self.rebuild_end(model, rebuild, key);
        model.slot_pages_to(rebuild.slot, end).unwrap_or(0) + self.draft.pages_to(end + STEPS + 1)
    }

    /// Prefills the next rows of `rebuild` for the sequence whose positions so far took
    /// `tokens`, whose key must start with the rows it prefilled, up to
    /// [`Self::rebuild_end`], in its slot, and commits their taps. At the boundary of a
    /// live prefix cache entry that holds the rebuilt positions and lacks the draft's
    /// state, copies the draft's rings into it, so that a later prompt that starts with
    /// it can speculate from the cache. Prefill rows follow the chunks of a prefill from
    /// position 0 up to the entry, which so gets the rings that prefilling its prompt with
    /// the draft gives. Prefill sums in other orders than the verify pass, so past the
    /// prompt the draft gets slightly different taps than speculation would have
    /// committed; that changes its drafts, never the target's tokens. Returns whether the
    /// draft holds every position of `tokens`.
    pub fn rebuild_step(
        &mut self,
        model: &mut Model,
        rebuild: &mut Rebuild,
        mut tokens: Prompt<'_>,
    ) -> anyhow::Result<bool> {
        tokens.check()?;
        ensure!(
            self.rebuild_live(model, rebuild),
            "something else used the rebuild's slot or the draft"
        );
        let done = rebuild.key.len();
        ensure!(
            tokens.key.starts_with(&rebuild.key),
            "the sequence differs from the rebuilt one"
        );
        let end = self.rebuild_end(model, rebuild, tokens.key);
        if end > done {
            let chunk = self.prefill_chunk;
            let images = Prompt::images(&mut tokens.images);
            model.with_slot(rebuild.slot, |m| {
                self.prefill_rows(m, &tokens.tokens[done..end], images, chunk)
            })?;
            rebuild.key.extend_from_slice(&tokens.key[done..end]);
        }
        for i in self.entries_without_draft(model, &rebuild.key) {
            self.save_entry_rings(model, i)?;
        }
        Ok(end == tokens.key.len())
    }

    /// Makes the draft follow the sequence that `rebuild` holds every position of, whose
    /// positions took `tokens`, from its end, for [`Self::resume`].
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taking the rebuild by value ends it"
    )]
    pub fn follow_rebuilt(
        &mut self,
        model: &Model,
        rebuild: Rebuild,
        tokens: &[u32],
    ) -> anyhow::Result<()> {
        let len = rebuild.key.len();
        ensure!(tokens.len() == len, "the rebuild holds {len} positions");
        ensure!(
            rebuild.sequence == self.draft_sequence && self.draft.committed() == len,
            "something else used the draft"
        );
        let noted = self.draft.context_capacity() / 2;
        if noted > 0 {
            let latest: Vec<i32> = tokens
                .iter()
                .rev()
                .take(noted)
                .map(|&t| i32::try_from(t))
                .collect::<Result<_, _>>()?;
            self.draft
                .note_tokens(model.stream(), model.draft_shared(), &latest)?;
        }
        self.prompt_len = len;
        self.batched = Some(Batched {
            start: len,
            tokens: Vec::new(),
        });
        Ok(())
    }

    /// Generates at least `n` tokens after [`Self::prefill`], `first` (its result)
    /// excluded, and returns them; the last round may emit a few more. `drafts` chooses
    /// the drafts.
    ///
    /// Each round launches the verify, then the draft's next round from the verify's
    /// results on the device, and synchronizes once, to read the verdict and the new
    /// drafts' confidences, from which [`Width`] picks the next round's width. A policy
    /// with a plain step time ([`WidthPolicy::with_plain`]) runs plain steps instead when
    /// no width is expected to be faster, then a draft round that decides again.
    pub fn generate(
        &mut self,
        model: &mut Model,
        n: usize,
        drafts: Drafts,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_with(model, n, drafts, |_| ControlFlow::Continue(()))
    }

    /// [`Self::generate`], calling `on_round` with each round's tokens, or each plain
    /// step's token, as soon as it is known. Generation stops early when `on_round`
    /// breaks.
    pub fn generate_with(
        &mut self,
        model: &mut Model,
        n: usize,
        drafts: Drafts,
        on_round: impl FnMut(&[u32]) -> ControlFlow<()>,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_with_room(model, n, drafts, |_, _, _| true, on_round)
    }

    /// [`Self::generate_with`], calling `room` with the model, the speculator and the
    /// position below which the model writes before the draft's first round and before
    /// each verify round or run of plain steps, so that it can free pages of the KV pool
    /// for them. Generation stops before the work when `room` returns false.
    #[expect(
        clippy::too_many_lines,
        reason = "main speculative generation loop with plain-step fallback and tree verify"
    )]
    pub fn generate_with_room(
        &mut self,
        model: &mut Model,
        n: usize,
        drafts: Drafts,
        mut room: impl FnMut(&mut Model, &mut Self, usize) -> bool,
        mut on_round: impl FnMut(&[u32]) -> ControlFlow<()>,
    ) -> anyhow::Result<Vec<u32>> {
        let stream = Arc::clone(model.stream());
        let mut rng = match drafts {
            Drafts::Random(seed) => seed,
            Drafts::Dspark => 0,
        };
        let proposals = self.width.drafts();
        self.stats = SpecStats {
            accepted_at: vec![0; STEPS],
            ..SpecStats::default()
        };
        if !room(model, self, model.position() + 1) {
            return Ok(Vec::new());
        }
        if self.conf_in_flight {
            self.conf_copied.synchronize()?;
            self.conf_in_flight = false;
        }
        if self.tree_dist_in_flight {
            self.tree_dist_copied.synchronize()?;
            self.tree_dist_in_flight = false;
        }
        self.round_event_pending = [false; 2];
        self.round_event_slot = 0;
        self.probe_event_pending = false;
        let started = Instant::now();
        let mut out = Vec::with_capacity(n + STEPS + 1);
        let mut run = FIRST_PLAIN_RUN;
        self.calibration = Calibration::default();
        let sampling_active = model.sampling_enabled() && !model.sampling().is_greedy();
        let mut next = Next::Verify(proposals + 1);
        self.sync_draft_sampling(model)?;
        if drafts == Drafts::Dspark {
            // Prefill committed the prompt, so the first round commits no rows.
            self.draft.set_round_inputs(
                &stream,
                &self.tail.slice(..0),
                model.position(),
                model.verify_anchor()?,
            )?;
            self.draft
                .replay_round(&stream, &self.linear, model.draft_shared(), proposals)?;
            next = self.next_round(&stream, 0, &mut run, sampling_active)?;
        }
        // The anchor on the host, once a round or step has emitted it.
        let mut anchor = None;
        while out.len() < n {
            let writes = match next {
                Next::Verify(width) => width,
                Next::Plain(steps) => steps.min(n - out.len()),
            };
            if !room(model, self, model.position() + writes) {
                break;
            }
            let width = match next {
                Next::Verify(width) => width,
                Next::Plain(steps) => {
                    if self.tree_dist_in_flight {
                        self.tree_dist_copied.synchronize()?;
                        self.tree_dist_in_flight = false;
                    }
                    let start = model.position();
                    let token = match anchor {
                        Some(t) => t,
                        None => u32::try_from(stream.clone_dtoh(model.verify_anchor()?)?[0])?,
                    };
                    let first_step = out.len();
                    let steps = steps.min(n - first_step);
                    let (token, stop) =
                        self.plain_steps(model, token, steps, drafts, &mut out, &mut on_round)?;
                    anchor = Some(token);
                    let done = stop || out.len() >= n;
                    if drafts == Drafts::Dspark {
                        self.probe(model, start, &out[first_step..], !done)?;
                    }
                    if done {
                        break;
                    }
                    next = match drafts {
                        Drafts::Dspark => {
                            self.stats.probes += 1;
                            self.next_round(&stream, self.stats.rounds, &mut run, sampling_active)?
                        }
                        Drafts::Random(_) => Next::Verify(self.next_width_blind(self.stats.rounds)),
                    };
                    continue;
                }
            };
            let emitted = self.verify_round(model, width, drafts, &mut rng)?;
            anchor = emitted.last().copied();
            let stop = on_round(&emitted).is_break();
            out.extend(emitted);
            if stop {
                break;
            }
            next = match drafts {
                Drafts::Dspark => {
                    self.next_round(&stream, self.stats.rounds, &mut run, sampling_active)?
                }
                Drafts::Random(_) => self.next_blind(self.stats.rounds),
            };
        }
        if self.conf_in_flight {
            self.conf_copied.synchronize()?;
            self.conf_in_flight = false;
        }
        if self.tree_dist_in_flight {
            self.tree_dist_copied.synchronize()?;
            self.tree_dist_in_flight = false;
        }
        stream.synchronize()?;
        self.stats.wall = started.elapsed();
        self.stats.emitted = out.len();
        self.flush_round_event(0)?;
        self.flush_round_event(1)?;
        self.flush_probe_event()?;
        Ok(out)
    }

    fn flush_round_event(&mut self, slot: usize) -> anyhow::Result<()> {
        if self.round_event_pending[slot] {
            let [before, verified, drafted] = &self.round_events[slot];
            self.stats.verify += ms(before.elapsed_ms(verified)?);
            self.stats.draft += ms(verified.elapsed_ms(drafted)?);
            self.round_event_pending[slot] = false;
        }
        Ok(())
    }

    fn flush_probe_event(&mut self) -> anyhow::Result<()> {
        if self.probe_event_pending {
            let [before, after] = &self.probe_events;
            self.stats.draft += ms(before.elapsed_ms(after)?);
            self.probe_event_pending = false;
        }
        Ok(())
    }

    /// Runs a verify round of `width` rows: the verify, then the draft's next round when
    /// `drafts` is the draft's, overlapping verify D2H and host verdict/callback work with
    /// the GPU draft graph via `copy_stream`. Reuses pre-allocated ping-pong timing events.
    #[expect(
        clippy::too_many_lines,
        reason = "overlapped verify, draft, and pinned D2H round"
    )]
    fn verify_round(
        &mut self,
        model: &mut Model,
        width: usize,
        drafts: Drafts,
        rng: &mut u64,
    ) -> anyhow::Result<Vec<u32>> {
        if self.tree_config.enabled
            && drafts == Drafts::Dspark
            && width >= 3
            && let Some(emitted) = self.try_verify_tree_round(model, width)?
        {
            return Ok(emitted);
        }
        if self.tree_dist_in_flight {
            self.tree_dist_copied.synchronize()?;
            self.tree_dist_in_flight = false;
        }
        let stream = Arc::clone(model.stream());
        let vocab = model.config().unpadded_vocab_size as u64;
        let proposals = self.width.drafts();
        let slot = self.round_event_slot;
        self.flush_round_event(slot)?;
        self.round_events[slot][0].record(&stream)?;
        let start = model.position();
        let tokens = match drafts {
            Drafts::Dspark => self.draft.draft_ids().slice(..width - 1),
            Drafts::Random(_) => {
                if width > 1 {
                    self.random_h2d_done.synchronize()?;
                    {
                        let dst = self.pinned_random.as_mut_slice();
                        for slot_id in &mut dst[..width - 1] {
                            *slot_id = i32::try_from(splitmix64(rng) % vocab)?;
                        }
                    }
                    self.pinned_random.copy_to_device(
                        &stream,
                        width - 1,
                        &mut self.random.slice_mut(..width - 1),
                    )?;
                    self.random_h2d_done.record(&stream)?;
                }
                self.random.slice(..width - 1)
            }
        };
        // The drafts and their logits before the draft's next round replaces them.
        let observed = if self.round_hook.is_some() {
            let ids = if width > 1 {
                stream.clone_dtoh(&tokens)?
            } else {
                Vec::new()
            };
            let logits = match self.draft.markov_logits() {
                Some(l) if width > 1 => {
                    let vocab = model.config().unpadded_vocab_size;
                    stream
                        .clone_dtoh(&l.slice(..(width - 1) * vocab))?
                        .into_iter()
                        .map(f32::from)
                        .collect()
                }
                _ => Vec::new(),
            };
            Some((ids, logits))
        } else {
            None
        };
        if model.spec_beta().is_some() {
            ensure!(
                self.draft.is_dflash2(),
                "--spec-beta requires a DFlash 2 drafter"
            );
            if drafts == Drafts::Dspark
                && let Some((cand, q_probs)) = self.draft.dflash2_distributions()
            {
                model.set_verify_draft_dist(&cand, &q_probs)?;
            }
        }
        let pending = model.start_verify(VerifyTokens::Drafts(tokens), Accept::Greedy)?;
        self.round_events[slot][1].record(&stream)?;
        // Enqueue the draft graph on the compute stream immediately after recording `verified`
        // so it launches with zero host-side delay, then pull the verify result on `copy_stream`
        // in parallel.
        if drafts == Drafts::Dspark {
            self.draft.set_round_inputs_kept(
                &stream,
                &model.verify_taps()?,
                start,
                &model.verify_kept()?,
                model.verify_anchor()?,
            )?;
            self.draft
                .replay_round(&stream, &self.linear, model.draft_shared(), proposals)?;
        }
        self.round_events[slot][2].record(&stream)?;
        self.round_event_pending[slot] = true;
        self.round_event_slot ^= 1;
        self.copy_stream.wait(&self.round_events[slot][1])?;
        self.pinned_verify_result
            .copy_from_device(&self.copy_stream, &model.verify_result_device()?)?;
        self.results_copied.record(&self.copy_stream)?;
        if drafts == Drafts::Dspark && matches!(self.width, Width::Adaptive(_)) {
            self.copy_stream.wait(&self.round_events[slot][2])?;
            self.pinned_conf.copy_from_device(
                &self.copy_stream,
                &self.draft.confidences().slice(..proposals),
            )?;
            self.conf_copied.record(&self.copy_stream)?;
            self.conf_in_flight = true;
        }
        if drafts == Drafts::Dspark
            && self.tree_config.enabled
            && let Some((cand, q_probs)) = self.draft.dflash2_distributions()
        {
            self.copy_stream.wait(&self.round_events[slot][2])?;
            let dist_n = proposals * crate::ops::sample::BETA_CAND_K;
            self.pinned_tree_cand
                .copy_from_device(&self.copy_stream, &cand.slice(..dist_n))?;
            self.pinned_tree_q
                .copy_from_device(&self.copy_stream, &q_probs.slice(..dist_n))?;
            self.pinned_tree_ids.copy_from_device(
                &self.copy_stream,
                &self.draft.draft_ids().slice(..proposals),
            )?;
            self.tree_dist_copied.record(&self.copy_stream)?;
            self.tree_dist_in_flight = true;
        }
        // Wait only for verify results (not draft!), overlapping verdict processing,
        // `note_tokens`, and the caller's `on_round` callback with the GPU draft graph.
        self.results_copied.synchronize()?;
        let verdict = model.complete_verify(pending, self.pinned_verify_result.as_slice())?;
        let a = verdict.accepted;
        if let (Some((ids, draft_logits)), Some(hook)) = (observed, self.round_hook.as_mut()) {
            let drafts: Vec<u32> = ids
                .iter()
                .map(|&t| u32::try_from(t))
                .collect::<Result<_, _>>()?;
            hook(&RoundRecord {
                start,
                drafts: &drafts,
                draft_logits: &draft_logits,
                target_logits: &model.verify_logits()?,
                drawn: &verdict.argmax,
                accepted: a,
            })?;
        }
        let emitted = verdict.emitted();
        self.stats.rounds += 1;
        self.stats.widths[width - 1] += 1;
        for kept in &mut self.stats.accepted_at[..a] {
            *kept += 1;
        }
        self.calibration.record(width, a);
        if drafts == Drafts::Dspark {
            let ids: Vec<i32> = emitted
                .iter()
                .map(|&t| i32::try_from(t))
                .collect::<Result<_, _>>()?;
            self.draft
                .note_tokens(&stream, model.draft_shared(), &ids)?;
        }
        Ok(emitted)
    }

    /// Attempts a 2-branch speculative tree verify round within `width` (`3..=VERIFY_ROWS`)
    /// total verify rows. Returns `Ok(None)` to fall back to the linear verify path if the
    /// gating threshold is not crossed (confidence threshold for greedy/coupled-race, or
    /// entropy threshold `H(q) > min_branch_entropy` for `--spec-beta` sampling) or if no
    /// distinct secondary token is available.
    #[expect(
        clippy::too_many_lines,
        reason = "handles both greedy/coupled-race and entropy-gated WOR beta-residual trees"
    )]
    fn try_verify_tree_round(
        &mut self,
        model: &mut Model,
        width: usize,
    ) -> anyhow::Result<Option<Vec<u32>>> {
        let stream = Arc::clone(model.stream());
        let proposals = self.width.drafts();
        let max_k2 = self.tree_config.secondary_len.min(width - 2);
        if max_k2 == 0 || proposals < width - 1 {
            return Ok(None);
        }
        if self.conf_in_flight {
            self.conf_copied.synchronize()?;
            self.conf_in_flight = false;
        }
        let had_tree_dist = if self.tree_dist_in_flight {
            self.tree_dist_copied.synchronize()?;
            self.tree_dist_in_flight = false;
            true
        } else {
            false
        };

        let start = model.position();
        let beta_sampling = model.spec_beta().is_some() && !model.sampling().is_greedy();
        let (k1, k2, branch_parent, primary_i32, alt_tokens) = if beta_sampling {
            ensure!(
                self.draft.is_dflash2(),
                "--spec-beta requires a DFlash 2 drafter"
            );
            let k_cand = crate::ops::sample::BETA_CAND_K;
            let dist_n = proposals * k_cand;
            let (cand_host, q_host, ids_host) = if had_tree_dist {
                (
                    self.pinned_tree_cand.as_slice()[..dist_n].to_vec(),
                    self.pinned_tree_q.as_slice()[..dist_n].to_vec(),
                    self.pinned_tree_ids.as_slice()[..proposals].to_vec(),
                )
            } else {
                let (cand_d, q_d) = self
                    .draft
                    .dflash2_distributions()
                    .context("DFlash 2 distributions missing")?;
                let c = stream.clone_dtoh(&cand_d.slice(..dist_n))?;
                let q = stream.clone_dtoh(&q_d.slice(..dist_n))?;
                let ids = stream.clone_dtoh(&self.draft.draft_ids().slice(..proposals))?;
                (c, q, ids)
            };

            let k1_max = width - 2;
            let mut entropies = [0.0f32; STEPS];
            let mut top1_q = [0.0f32; STEPS];
            for j in 0..proposals {
                let row = &q_host[j * k_cand..(j + 1) * k_cand];
                entropies[j] = entropy_nats(row);
                top1_q[j] = row.iter().copied().fold(0.0f32, f32::max);
            }

            let (k1, k2, b) = match self.tree_config.branch_strategy {
                BranchStrategy::Fixed(_) => {
                    let k2 = max_k2;
                    let k1 = width - 1 - k2;
                    let Some(b) = choose_branch_parent_by_entropy(
                        &self.tree_config,
                        k1,
                        &entropies[..k1],
                        None,
                    ) else {
                        return Ok(None);
                    };
                    if b + k2 > proposals {
                        return Ok(None);
                    }
                    (k1, k2, b)
                }
                BranchStrategy::LowestConfidence { .. } => {
                    let Some(b) = choose_branch_parent_by_entropy(
                        &self.tree_config,
                        k1_max,
                        &entropies[..k1_max],
                        Some(&top1_q[..k1_max]),
                    ) else {
                        return Ok(None);
                    };
                    // Match offline study (553fd32): choose 5+2 over 6+1 only when the
                    // next step is confident enough to sustain a 2-node secondary branch.
                    let next_top1 = top1_q.get(b + 1).copied().unwrap_or(0.0);
                    let k2 = if max_k2 >= 2
                        && b < width - 1 - max_k2
                        && b + max_k2 <= proposals
                        && next_top1 > 0.65
                    {
                        max_k2
                    } else {
                        1
                    };
                    let k1 = width - 1 - k2;
                    (k1, k2, b)
                }
            };

            let pos = u32::try_from(start + b + 1).context("position does not fit kernels")?;
            let u_sec = secondary_draw_uniform(model.sampling().seed, pos);
            let cand_b = &cand_host[b * k_cand..(b + 1) * k_cand];
            let q_b = &q_host[b * k_cand..(b + 1) * k_cand];
            let Some(x2) = sample_without_replacement(cand_b, q_b, ids_host[b], u_sec) else {
                return Ok(None);
            };
            let (alt_tokens, branch_q) =
                self.draft
                    .rollout_secondary_branch_from(&stream, b, x2, k2, proposals)?;

            // Assemble per-tree-node 16-sparse draft distributions `[m, 16]`.
            let m = 1 + k1 + k2;
            let mut tree_cand = vec![-1i32; m * k_cand];
            let mut tree_q = vec![0.0f32; m * k_cand];
            for r in 0..k1 {
                tree_cand[r * k_cand..(r + 1) * k_cand]
                    .copy_from_slice(&cand_host[r * k_cand..(r + 1) * k_cand]);
                tree_q[r * k_cand..(r + 1) * k_cand]
                    .copy_from_slice(&q_host[r * k_cand..(r + 1) * k_cand]);
            }
            for t in 0..k2.saturating_sub(1) {
                let node = 1 + k1 + t;
                let step = b + 1 + t;
                tree_cand[node * k_cand..(node + 1) * k_cand]
                    .copy_from_slice(&cand_host[step * k_cand..(step + 1) * k_cand]);
                tree_q[node * k_cand..(node + 1) * k_cand]
                    .copy_from_slice(&branch_q[(t + 1) * k_cand..(t + 2) * k_cand]);
            }
            model.set_verify_tree_draft_dist(&tree_cand, &tree_q)?;
            (k1, k2, b, ids_host[..k1].to_vec(), alt_tokens)
        } else {
            let k2 = max_k2;
            let k1 = width - 1 - k2;
            if k1 == 0 {
                return Ok(None);
            }
            let confidences = stream.clone_dtoh(&self.draft.confidences().slice(..k1))?;
            let Some(branch_parent) =
                choose_branch_parent(&self.tree_config, k1, &confidences, None)
            else {
                return Ok(None);
            };
            let primary_i32 = stream.clone_dtoh(&self.draft.draft_ids().slice(..k1))?;
            let Some(alt_tokens) = self.draft.rollout_secondary_branch(
                &stream,
                branch_parent,
                primary_i32[branch_parent],
                k2,
                proposals,
            )?
            else {
                return Ok(None);
            };
            (k1, k2, branch_parent, primary_i32, alt_tokens)
        };

        let tree = SpecTree::two_branch(k1, branch_parent, k2)?;
        let mut tree_drafts = Vec::with_capacity(k1 + k2);
        for &tok in &primary_i32 {
            tree_drafts.push(u32::try_from(tok).context("negative primary draft token")?);
        }
        tree_drafts.extend_from_slice(&alt_tokens);

        let slot = self.round_event_slot;
        self.flush_round_event(slot)?;
        self.round_events[slot][0].record(&stream)?;
        let pending = model.start_verify_tree(
            VerifyTreeTokens::WithDeviceAnchor(&tree_drafts),
            &tree,
            AcceptTree::Auto,
        )?;
        self.round_events[slot][1].record(&stream)?;
        self.draft.set_round_inputs_kept(
            &stream,
            &model.verify_tree_taps()?,
            start,
            &model.verify_kept()?,
            model.verify_anchor()?,
        )?;
        self.draft
            .replay_round(&stream, &self.linear, model.draft_shared(), proposals)?;
        self.round_events[slot][2].record(&stream)?;
        self.round_event_pending[slot] = true;
        self.round_event_slot ^= 1;
        self.copy_stream.wait(&self.round_events[slot][1])?;
        self.pinned_verify_result
            .copy_from_device(&self.copy_stream, &model.verify_result_device()?)?;
        self.pinned_tree_path
            .copy_from_device(&self.copy_stream, &model.verify_tree_path_device()?)?;
        self.results_copied.record(&self.copy_stream)?;
        if matches!(self.width, Width::Adaptive(_)) {
            self.copy_stream.wait(&self.round_events[slot][2])?;
            self.pinned_conf.copy_from_device(
                &self.copy_stream,
                &self.draft.confidences().slice(..proposals),
            )?;
            self.conf_copied.record(&self.copy_stream)?;
            self.conf_in_flight = true;
        }
        if let Some((cand, q_probs)) = self.draft.dflash2_distributions() {
            self.copy_stream.wait(&self.round_events[slot][2])?;
            let dist_n = proposals * crate::ops::sample::BETA_CAND_K;
            self.pinned_tree_cand
                .copy_from_device(&self.copy_stream, &cand.slice(..dist_n))?;
            self.pinned_tree_q
                .copy_from_device(&self.copy_stream, &q_probs.slice(..dist_n))?;
            self.pinned_tree_ids.copy_from_device(
                &self.copy_stream,
                &self.draft.draft_ids().slice(..proposals),
            )?;
            self.tree_dist_copied.record(&self.copy_stream)?;
            self.tree_dist_in_flight = true;
        }
        self.results_copied.synchronize()?;
        let verdict = model.complete_verify_tree(
            pending,
            self.pinned_verify_result.as_slice(),
            self.pinned_tree_path.as_slice(),
        )?;
        let a = verdict.accepted;
        if let Some(hook) = self.round_hook.as_mut() {
            hook(&RoundRecord {
                start,
                drafts: &tree_drafts,
                draft_logits: &[],
                target_logits: &model.verify_logits()?,
                drawn: &verdict.argmax,
                accepted: a,
            })?;
        }
        let emitted = verdict.emitted();
        self.stats.rounds += 1;
        self.stats.tree_rounds += 1;
        if verdict.took_secondary {
            self.stats.secondary_accepted += 1;
        }
        self.stats.widths[tree.num_nodes - 1] += 1;
        for kept in &mut self.stats.accepted_at[..a] {
            *kept += 1;
        }
        let primary_kept = if verdict.took_secondary {
            branch_parent
        } else {
            a
        };
        self.calibration.record(k1 + 1, primary_kept);
        let ids: Vec<i32> = emitted
            .iter()
            .map(|&t| i32::try_from(t))
            .collect::<Result<_, _>>()?;
        self.draft
            .note_tokens(&stream, model.draft_shared(), &ids)?;
        Ok(Some(emitted))
    }

    /// Runs `steps` plain steps from `token`, the anchor, appends their tokens to `out`
    /// and calls `on_round` with each, and makes the last token the verify's anchor.
    /// Keeps each step's taps for the draft. Returns the last token and whether
    /// `on_round` broke, which ends the run early.
    fn plain_steps(
        &mut self,
        model: &mut Model,
        mut token: u32,
        steps: usize,
        drafts: Drafts,
        out: &mut Vec<u32>,
        on_round: &mut impl FnMut(&[u32]) -> ControlFlow<()>,
    ) -> anyhow::Result<(u32, bool)> {
        let started = Instant::now();
        let row = self.row_len;
        let mut stop = false;
        for i in 0..steps {
            token = model.step_with_taps(token)?;
            if drafts == Drafts::Dspark {
                model.stream().memcpy_dtod(
                    &model.step_taps()?,
                    &mut self.plain_taps.slice_mut(i * row..(i + 1) * row),
                )?;
            }
            self.stats.stepped.push(out.len());
            out.push(token);
            if on_round(&[token]).is_break() {
                stop = true;
                break;
            }
        }
        self.stats.plain += started.elapsed();
        model.set_verify_anchor(token)?;
        Ok((token, stop))
    }

    /// After plain steps from `start` that emitted `tokens`: commits their taps to the
    /// draft and notes the tokens, then, if `draft`, runs the draft's round from the new
    /// anchor. Records timing in the reusable `probe_events`.
    fn probe(
        &mut self,
        model: &Model,
        start: usize,
        tokens: &[u32],
        draft: bool,
    ) -> anyhow::Result<()> {
        let stream = Arc::clone(model.stream());
        self.flush_probe_event()?;
        self.probe_events[0].record(&stream)?;
        self.commit_plain(&stream, model, start, tokens)?;
        if draft {
            self.draft.set_round_inputs(
                &stream,
                &self.tail.slice(..0),
                model.position(),
                model.verify_anchor()?,
            )?;
            self.draft.replay_round(
                &stream,
                &self.linear,
                model.draft_shared(),
                self.width.drafts(),
            )?;
        }
        self.probe_events[1].record(&stream)?;
        self.probe_event_pending = true;
        Ok(())
    }

    /// Commits the taps of the plain steps from `start` to the draft, which emitted
    /// `tokens`, and notes the tokens.
    fn commit_plain(
        &mut self,
        stream: &Arc<CudaStream>,
        model: &Model,
        start: usize,
        tokens: &[u32],
    ) -> anyhow::Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let row = self.row_len;
        self.draft.commit(
            stream,
            &self.linear,
            &self.plain_taps.slice(..tokens.len() * row),
            start,
        )?;
        let ids: Vec<i32> = tokens
            .iter()
            .map(|&t| i32::try_from(t))
            .collect::<Result<_, _>>()?;
        self.draft.note_tokens(stream, model.draft_shared(), &ids)
    }

    /// What follows the draft's last round, the `round`th verify round: a verify of the
    /// width the confidences pick, or a run of plain steps. `run` is the length of the
    /// next run of plain steps: it doubles after each run whose draft round still
    /// prefers plain steps, up to [`MAX_PLAIN_RUN`], and starts again from
    /// [`FIRST_PLAIN_RUN`] once a verify is chosen.
    fn next_round(
        &mut self,
        stream: &Arc<CudaStream>,
        round: usize,
        run: &mut usize,
        sampling_active: bool,
    ) -> anyhow::Result<Next> {
        let count = match &self.width {
            Width::Adaptive(policy) => policy.widest() - 1,
            Width::Fixed(_) => return Ok(self.next_blind(round)),
        };
        if self.conf_in_flight {
            self.conf_copied.synchronize()?;
            self.conf_in_flight = false;
        } else {
            self.sync_drafted.record(stream)?;
            self.copy_stream.wait(&self.sync_drafted)?;
            self.pinned_conf
                .copy_from_device(&self.copy_stream, &self.draft.confidences().slice(..count))?;
            self.conf_copied.record(&self.copy_stream)?;
            self.conf_copied.synchronize()?;
        }
        let confidence = self.pinned_conf.as_slice()[..count].to_vec();
        let calibrated = self.calibration.apply(&confidence);
        let Width::Adaptive(policy) = &self.width else {
            unreachable!("checked above");
        };
        let decision = if sampling_active {
            policy.decide(&calibrated, &calibrated, *run)
        } else {
            policy.decide(&confidence, &calibrated, *run)
        };
        Ok(match decision {
            None => {
                let steps = *run;
                *run = (*run * 2).min(MAX_PLAIN_RUN);
                Next::Plain(steps)
            }
            Some(width) => {
                *run = FIRST_PLAIN_RUN;
                self.calibration.round = confidence;
                Next::Verify(width)
            }
        })
    }

    /// What follows verify round `round` without confidences: the fixed width, or a
    /// cycle through the adaptive widths and, if the policy has a plain step time, a run
    /// of plain steps.
    fn next_blind(&self, round: usize) -> Next {
        match &self.width {
            Width::Adaptive(policy) if policy.plain.is_some() => {
                let i = round % (policy.widths.len() + 1);
                policy
                    .widths
                    .get(i)
                    .map_or(Next::Plain(FIRST_PLAIN_RUN), |&w| Next::Verify(w))
            }
            _ => Next::Verify(self.next_width_blind(round)),
        }
    }

    /// The width of round `round` without confidences: the fixed one, or a cycle through
    /// the adaptive widths.
    fn next_width_blind(&self, round: usize) -> usize {
        match &self.width {
            Width::Fixed(k) => k + 1,
            Width::Adaptive(policy) => policy.widths[round % policy.widths.len()],
        }
    }

    /// Greedy decoding through the verify pass one row at a time, `n` tokens after the
    /// `first` that [`Self::prefill`] returned (after [`Self::restart`] if needed), with a
    /// draft round of 7 drafts at each anchor. Returns the tokens and, for each anchor
    /// (`first`, then each token but the last), the drafts that follow it with their
    /// confidences. A round sees the same drafts at a given anchor whatever the rounds
    /// before it verified, so these rows price any width policy offline.
    pub fn sweep(
        &mut self,
        model: &mut Model,
        n: usize,
    ) -> anyhow::Result<(Vec<u32>, Vec<SweepRow>)> {
        let stream = Arc::clone(model.stream());
        self.sync_draft_sampling(model)?;
        self.draft.set_round_inputs(
            &stream,
            &self.tail.slice(..0),
            model.position(),
            model.verify_anchor()?,
        )?;
        let mut tokens = Vec::with_capacity(n);
        let mut rows = Vec::with_capacity(n);
        while tokens.len() < n {
            self.draft
                .replay_round(&stream, &self.linear, model.draft_shared(), STEPS)?;
            let drafts = stream
                .clone_dtoh(self.draft.draft_ids())?
                .into_iter()
                .map(u32::try_from)
                .collect::<Result<_, _>>()?;
            let confidence = stream.clone_dtoh(self.draft.confidences())?;
            rows.push(SweepRow { drafts, confidence });
            let start = model.position();
            let verdict = model.verify(
                VerifyTokens::Drafts(self.draft.draft_ids().slice(..0)),
                Accept::Greedy,
            )?;
            tokens.push(verdict.next());
            let next = [i32::try_from(verdict.next())?];
            self.draft
                .note_tokens(&stream, model.draft_shared(), &next)?;
            self.draft.set_round_inputs(
                &stream,
                &model.verify_taps()?,
                start,
                model.verify_anchor()?,
            )?;
        }
        Ok((tokens, rows))
    }
}

/// Timed verify passes per width that [`price`] takes the fastest of.
const PRICE_REPEATS: usize = 8;

/// Times `repeats` verify passes of `m` rows from `saved`, each keeping one row, after
/// three untimed ones (the first captures the graph), in ms. `rows` holds the tokens,
/// at least `repeats + 2 + m` of them.
pub fn verify_times(
    model: &mut Model,
    saved: &SavedState,
    rows: &[u32],
    m: usize,
    repeats: usize,
) -> anyhow::Result<Vec<f64>> {
    ensure!(
        rows.len() >= repeats + 2 + m,
        "{} rows cannot time {repeats} verify passes of {m} rows",
        rows.len()
    );
    model.restore_state(saved)?;
    let mut times = Vec::with_capacity(repeats);
    for i in 0..repeats + 3 {
        let window = &rows[i..i + m];
        model.stream().synchronize()?;
        let t = Instant::now();
        model.verify(VerifyTokens::Host(window), Accept::Rows(1))?;
        if i >= 3 {
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    Ok(times)
}

/// Times `repeats` plain steps that record the draft's taps from `saved`, each with the
/// copy of its taps the round loop makes, after three untimed ones (the first captures the
/// graph), in ms. `rows` holds the tokens, at least `repeats + 3` of them.
pub fn plain_times(
    model: &mut Model,
    saved: &SavedState,
    rows: &[u32],
    repeats: usize,
) -> anyhow::Result<Vec<f64>> {
    ensure!(
        rows.len() >= repeats + 3,
        "{} rows cannot time {repeats} plain steps",
        rows.len()
    );
    model.restore_state(saved)?;
    let stream = Arc::clone(model.stream());
    let mut copy = stream.alloc_zeros::<bf16>(model.tap_layout().len() * HIDDEN)?;
    let mut times = Vec::with_capacity(repeats);
    for (i, &token) in rows[..repeats + 3].iter().enumerate() {
        stream.synchronize()?;
        let t = Instant::now();
        model.step_with_taps(token)?;
        stream.memcpy_dtod(&model.step_taps()?, &mut copy)?;
        if i >= 3 {
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    Ok(times)
}

/// Prices an adaptive `speculator` at the state `saved`, which [`Speculator::prefill`]
/// left with `first` as its result, and returns its policy with the prices: each width at
/// the fastest of a few verify passes of that width, plus the draft and host time per
/// round of a short warmup generation, and a plain step with taps at the fastest of a few.
/// `rows` holds tokens for the verify passes (see [`verify_times`]). Leaves the model at
/// `saved`.
pub fn price(
    model: &mut Model,
    speculator: &mut Speculator,
    saved: &SavedState,
    rows: &[u32],
    first: u32,
) -> anyhow::Result<WidthPolicy> {
    let Width::Adaptive(policy) = speculator.width().clone() else {
        anyhow::bail!("only an adaptive width is priced");
    };
    let mut verify = [f64::INFINITY; VERIFY_ROWS];
    for &w in policy.widths() {
        let times = verify_times(model, saved, rows, w, PRICE_REPEATS)?;
        verify[w - 1] = times.into_iter().fold(f64::INFINITY, f64::min);
    }
    let plain = plain_times(model, saved, rows, PRICE_REPEATS)?
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    model.restore_state(saved)?;
    speculator.restart(model, first)?;
    speculator.set_width(Width::Adaptive(policy.clone().with_plain(None)?))?;
    let warmup = speculator.generate(model, 64, Drafts::Dspark);
    speculator.set_width(Width::Adaptive(policy.clone()))?;
    warmup?;
    let stats = speculator.stats();
    let rounds = stats.rounds.max(1) as f64;
    let other = (stats.draft + stats.host()).as_secs_f64() * 1e3 / rounds;
    model.restore_state(saved)?;
    WidthPolicy::new(
        policy.widths(),
        verify.map(|v| if v.is_finite() { v + other } else { 1.0 }),
    )?
    .with_plain(Some(Plain {
        step_ms: plain,
        probe_ms: other,
    }))
}

/// Greedy decoding through the verify pass one row at a time: `n` tokens after `first`,
/// from the model's current position. The reference that speculative output must equal.
pub fn greedy_by_verify(model: &mut Model, first: u32, n: usize) -> anyhow::Result<Vec<u32>> {
    let mut out = Vec::with_capacity(n);
    let mut token = first;
    while out.len() < n {
        token = model
            .verify(VerifyTokens::Host(&[token]), Accept::Greedy)?
            .next();
        out.push(token);
    }
    Ok(out)
}

/// Greedy decoding one token at a time from the model's current position, `n` tokens
/// after `first`: a plain step ([`Model::step`]) for each token whose index `stepped`
/// holds (ascending), a verify pass of one row for the others. Plain steps sum in other
/// orders than the verify pass, so the output of a speculator that fell back to plain
/// steps must equal this for its [`SpecStats::stepped`].
pub fn greedy_by_path(
    model: &mut Model,
    first: u32,
    n: usize,
    stepped: &[usize],
) -> anyhow::Result<Vec<u32>> {
    let mut out = Vec::with_capacity(n);
    let mut token = first;
    let mut steps = stepped.iter().peekable();
    while out.len() < n {
        token = if steps.next_if_eq(&&out.len()).is_some() {
            model.step(token, None)?
        } else {
            model
                .verify(VerifyTokens::Host(&[token]), Accept::Greedy)?
                .next()
        };
        out.push(token);
    }
    Ok(out)
}

fn timing_event(stream: &CudaStream) -> anyhow::Result<CudaEvent> {
    stream
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .context("creating a timing event")
}

fn ms(milliseconds: f32) -> Duration {
    Duration::from_secs_f64(f64::from(milliseconds) / 1e3)
}

/// One step of SplitMix64.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
