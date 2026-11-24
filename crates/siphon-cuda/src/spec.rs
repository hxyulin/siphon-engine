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
#[derive(Clone, Debug)]
pub struct WidthPolicy {
    /// The widths it may pick, ascending, each 1 to 8.
    widths: Vec<usize>,
    /// The time of a round of each width from 1 to 8: verify, draft and host.
    round_ms: [f64; VERIFY_ROWS],
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
        })
    }

    #[must_use]
    pub fn widths(&self) -> &[usize] {
        &self.widths
    }

    fn widest(&self) -> usize {
        self.widths.last().copied().unwrap_or(2)
    }

    /// The width with the most expected tokens per millisecond for drafts with these
    /// confidences, the first one first.
    #[must_use]
    pub fn choose(&self, confidence: &[f32]) -> usize {
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
        best.0
    }
}

/// Counts and times of the rounds since the last [`Speculator::generate`] started.
#[derive(Clone, Debug, Default)]
pub struct SpecStats {
    pub rounds: usize,
    /// Tokens emitted, the first token from the prefill excluded.
    pub emitted: usize,
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
}

impl SpecStats {
    /// Mean tokens per round.
    #[must_use]
    pub fn tokens_per_round(&self) -> f64 {
        self.emitted as f64 / self.rounds.max(1) as f64
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

    /// Wall time outside the verify passes and the draft.
    #[must_use]
    pub fn host(&self) -> Duration {
        self.wall.saturating_sub(self.verify + self.draft)
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
    /// drafts' confidences, from which [`Width`] picks the next round's width.
    pub fn generate(
        &mut self,
        model: &mut Model,
        n: usize,
        drafts: Drafts,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_with(model, n, drafts, |_| ControlFlow::Continue(()))
    }

    /// [`Self::generate`], calling `on_round` with each round's tokens as soon as the
    /// round ends. Generation stops early when `on_round` breaks.
    pub fn generate_with(
        &mut self,
        model: &mut Model,
        n: usize,
        drafts: Drafts,
        mut on_round: impl FnMut(&[u32]) -> ControlFlow<()>,
    ) -> anyhow::Result<Vec<u32>> {
        let stream = Arc::clone(model.stream());
        let vocab = model.config().unpadded_vocab_size as u64;
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
        let started = Instant::now();
        let mut out = Vec::with_capacity(n + STEPS + 1);
        let mut width = proposals + 1;
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
            width = self.next_width(&stream, 0)?;
        }
        while out.len() < n {
            let before = timing_event(&stream)?;
            let verified = timing_event(&stream)?;
            let drafted = timing_event(&stream)?;
            before.record(&stream)?;
            let start = model.position();
            let tokens = match drafts {
                Drafts::Dspark => self.draft.draft_ids().slice(..width - 1),
                Drafts::Random(_) => {
                    let ids: Vec<i32> = (1..width)
                        .map(|_| i32::try_from(splitmix64(&mut rng) % vocab))
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
            timings.push([before, verified, drafted]);
            let a = verdict.accepted;
            let emitted = verdict.emitted();
            self.stats.rounds += 1;
            self.stats.widths[width - 1] += 1;
            for kept in &mut self.stats.accepted_at[..a] {
                *kept += 1;
            }
            if drafts == Drafts::Dspark {
                let ids: Vec<i32> = emitted
                    .iter()
                    .map(|&t| i32::try_from(t))
                    .collect::<Result<_, _>>()?;
                self.draft
                    .note_tokens(&stream, model.draft_shared(), &ids)?;
            }
            let stop = on_round(&emitted).is_break();
            out.extend(emitted);
            if stop {
                break;
            }
            width = match drafts {
                Drafts::Dspark => self.next_width(&stream, self.stats.rounds)?,
                Drafts::Random(_) => self.next_width_blind(self.stats.rounds),
            };
        }
        stream.synchronize()?;
        self.stats.wall = started.elapsed();
        self.stats.emitted = out.len();
        for [before, verified, drafted] in &timings {
            self.stats.verify += ms(before.elapsed_ms(verified)?);
            self.stats.draft += ms(verified.elapsed_ms(drafted)?);
        }
        Ok(out)
    }

    /// The width of the round after the draft's last round, the `round`th.
    fn next_width(&self, stream: &Arc<CudaStream>, round: usize) -> anyhow::Result<usize> {
        let Width::Adaptive(policy) = &self.width else {
            return Ok(self.next_width_blind(round));
        };
        let confidence =
            stream.clone_dtoh(&self.draft.confidences().slice(..policy.widest() - 1))?;
        Ok(policy.choose(&confidence))
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

/// Prices each width of an adaptive `speculator` at the state `saved`, which
/// [`Speculator::prefill`] left with `first` as its result: the fastest of a few verify
/// passes of that width, plus the draft and host time per round of a short warmup
/// generation. `rows` holds tokens for the verify passes (see [`verify_times`]). Leaves
/// the model at `saved`.
pub fn price(
    model: &mut Model,
    speculator: &mut Speculator,
    saved: &SavedState,
    rows: &[u32],
    first: u32,
) -> anyhow::Result<[f64; VERIFY_ROWS]> {
    let Width::Adaptive(policy) = speculator.width().clone() else {
        anyhow::bail!("only an adaptive width is priced");
    };
    let mut verify = [f64::INFINITY; VERIFY_ROWS];
    for &w in policy.widths() {
        let times = verify_times(model, saved, rows, w, PRICE_REPEATS)?;
        verify[w - 1] = times.into_iter().fold(f64::INFINITY, f64::min);
    }
    model.restore_state(saved)?;
    speculator.restart(model, first)?;
    speculator.generate(model, 64, Drafts::Dspark)?;
    let stats = speculator.stats();
    let rounds = stats.rounds.max(1) as f64;
    let other = (stats.draft + stats.host()).as_secs_f64() * 1e3 / rounds;
    model.restore_state(saved)?;
    Ok(verify.map(|v| if v.is_finite() { v + other } else { 1.0 }))
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
