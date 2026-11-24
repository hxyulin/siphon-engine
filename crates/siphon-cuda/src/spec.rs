//! Greedy speculative decoding with the DSpark draft: each round verifies the anchor and
//! up to 7 drafts in one pass and emits the kept drafts and the target's next token.
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
//! Greedy acceptance keeps only drafts equal to the target's own greedy token, and the
//! verify pass gives every row the same result at every width, so the output equals
//! greedy decoding through the verify pass one row at a time, whatever the draft proposes
//! and whatever widths the rounds verify ([`greedy_by_verify`]).
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

use crate::{
    draft::{CONTEXT_ROWS, Draft, DraftOptions, STEPS, WINDOW},
    model::{Accept, Model, PrefillLogits, SavedState, VerifyTokens},
    ops::{linear::Linear, spec::VERIFY_ROWS},
    weights::{
        dflash::DflashWeights,
        dflash2::{DFLASH2_WINDOW, Dflash2Weights, is_dflash2},
        draft::{DraftWeights, HIDDEN},
    },
};

/// Prompt rows whose taps [`Speculator::restart`] commits again: the slots of the draft's
/// rings, the window and a block. DFlash 2's rings, 2056 slots, are the longest.
const TAIL: usize = DFLASH2_WINDOW + STEPS + 1;
const _: () = assert!(
    DFLASH2_WINDOW >= WINDOW,
    "TAIL covers the bundled draft's rings"
);

/// Prompt positions whose tokens a shortlist head adds to its context rows, the last
/// ones: at most half of [`CONTEXT_ROWS`], so the tokens that generation emits keep room.
/// A prompt of random ids that filled every row left the draft unable to propose the new
/// tokens of the continuation.
const NOTED_PROMPT: usize = CONTEXT_ROWS / 2;

/// Plain steps in the first run after a draft round prefers them. Each run whose draft
/// round still prefers plain steps doubles the next run, up to [`MAX_PLAIN_RUN`]. One
/// step and a draft round emit a token faster than a verify of one row.
const FIRST_PLAIN_RUN: usize = 1;
const MAX_PLAIN_RUN: usize = 32;

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
pub struct Speculator {
    draft: Draft,
    linear: Linear,
    width: Width,
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
        Ok(Self {
            draft,
            linear: Linear::load(stream.context())?,
            width,
            random: stream.alloc_zeros(STEPS)?,
            tail: stream.alloc_zeros(TAIL * row_len)?,
            spare: stream.alloc_zeros(TAIL * row_len)?,
            tail_rows: 0,
            plain_taps: stream.alloc_zeros(MAX_PLAIN_RUN * row_len)?,
            calibration: Calibration::default(),
            row_len,
            prompt_len: 0,
            stats: SpecStats::default(),
        })
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
        self.draft.reset();
        let stream = Arc::clone(model.stream());
        // A shortlist head takes the tokens of the prompt's last positions, latest first,
        // and keeps room for the tokens the rounds emit.
        let latest: Vec<i32> = prompt
            .iter()
            .rev()
            .take(NOTED_PROMPT)
            .map(|&t| i32::try_from(t))
            .collect::<Result<_, _>>()?;
        self.draft
            .note_tokens(&stream, model.draft_shared(), &latest)?;
        self.tail_rows = 0;
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
        let first = model.prefill_with_taps(prompt, chunk, PrefillLogits::Last, &mut commit)?;
        model.set_verify_anchor(first)?;
        self.prompt_len = prompt.len();
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
        let mut timings: Vec<[CudaEvent; 3]> = Vec::new();
        let mut probes: Vec<[CudaEvent; 2]> = Vec::new();
        let started = Instant::now();
        let mut out = Vec::with_capacity(n + STEPS + 1);
        let mut run = FIRST_PLAIN_RUN;
        self.calibration = Calibration::default();
        let mut next = Next::Verify(proposals + 1);
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
            next = self.next_round(&stream, 0, &mut run)?;
        }
        // The anchor on the host, once a round or step has emitted it.
        let mut anchor = None;
        while out.len() < n {
            let width = match next {
                Next::Verify(width) => width,
                Next::Plain(steps) => {
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
                        probes.push(self.probe(model, start, &out[first_step..], !done)?);
                    }
                    if done {
                        break;
                    }
                    next = match drafts {
                        Drafts::Dspark => {
                            self.stats.probes += 1;
                            self.next_round(&stream, self.stats.rounds, &mut run)?
                        }
                        Drafts::Random(_) => Next::Verify(self.next_width_blind(self.stats.rounds)),
                    };
                    continue;
                }
            };
            let (emitted, events) = self.verify_round(model, width, drafts, &mut rng)?;
            timings.push(events);
            anchor = emitted.last().copied();
            let stop = on_round(&emitted).is_break();
            out.extend(emitted);
            if stop {
                break;
            }
            next = match drafts {
                Drafts::Dspark => self.next_round(&stream, self.stats.rounds, &mut run)?,
                Drafts::Random(_) => self.next_blind(self.stats.rounds),
            };
        }
        stream.synchronize()?;
        self.stats.wall = started.elapsed();
        self.stats.emitted = out.len();
        for [before, verified, drafted] in &timings {
            self.stats.verify += ms(before.elapsed_ms(verified)?);
            self.stats.draft += ms(verified.elapsed_ms(drafted)?);
        }
        for [before, after] in &probes {
            self.stats.draft += ms(before.elapsed_ms(after)?);
        }
        Ok(out)
    }

    /// Runs a verify round of `width` rows: the verify, then the draft's next round when
    /// `drafts` is the draft's. Returns the round's tokens and timing events around the
    /// verify and the draft. `rng` makes random drafts.
    fn verify_round(
        &mut self,
        model: &mut Model,
        width: usize,
        drafts: Drafts,
        rng: &mut u64,
    ) -> anyhow::Result<(Vec<u32>, [CudaEvent; 3])> {
        let stream = Arc::clone(model.stream());
        let vocab = model.config().unpadded_vocab_size as u64;
        let proposals = self.width.drafts();
        let before = timing_event(&stream)?;
        let verified = timing_event(&stream)?;
        let drafted = timing_event(&stream)?;
        before.record(&stream)?;
        let start = model.position();
        let tokens = match drafts {
            Drafts::Dspark => self.draft.draft_ids().slice(..width - 1),
            Drafts::Random(_) => {
                let ids: Vec<i32> = (1..width)
                    .map(|_| i32::try_from(splitmix64(rng) % vocab))
                    .collect::<Result<_, _>>()?;
                if !ids.is_empty() {
                    stream.memcpy_htod(&ids, &mut self.random.slice_mut(..ids.len()))?;
                }
                self.random.slice(..width - 1)
            }
        };
        let pending = model.start_verify(VerifyTokens::Drafts(tokens), Accept::Greedy)?;
        verified.record(&stream)?;
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
        drafted.record(&stream)?;
        let verdict = model.finish_verify(pending)?;
        let a = verdict.accepted;
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
        Ok((emitted, [before, verified, drafted]))
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
    /// anchor. Returns timing events around the work.
    fn probe(
        &mut self,
        model: &Model,
        start: usize,
        tokens: &[u32],
        draft: bool,
    ) -> anyhow::Result<[CudaEvent; 2]> {
        let stream = Arc::clone(model.stream());
        let before = timing_event(&stream)?;
        let after = timing_event(&stream)?;
        before.record(&stream)?;
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
        after.record(&stream)?;
        Ok([before, after])
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
    ) -> anyhow::Result<Next> {
        let Width::Adaptive(policy) = &self.width else {
            return Ok(self.next_blind(round));
        };
        let confidence =
            stream.clone_dtoh(&self.draft.confidences().slice(..policy.widest() - 1))?;
        let calibrated = self.calibration.apply(&confidence);
        Ok(match policy.decide(&confidence, &calibrated, *run) {
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
