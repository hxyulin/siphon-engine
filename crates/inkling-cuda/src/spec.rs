//! Greedy speculative decoding with the DSpark draft: each round verifies the anchor and
//! `k` drafts in one pass of `k + 1` rows and emits the kept drafts and the target's next
//! token.
//!
//! A round, after the prompt's prefill has committed its taps to the draft and the draft
//! has run its first block:
//!
//! 1. [`Model::verify`] runs the anchor and the draft's first `k` tokens, keeps the
//!    longest matching run of `a` drafts and synchronizes to read the verdict.
//! 2. The draft commits the taps of the kept rows `0..=a` and runs its block from the new
//!    anchor, which the verify left on the device.
//!
//! Greedy acceptance keeps only drafts equal to the target's own greedy token, and the
//! verify pass gives every row the same result at every width, so the output equals
//! greedy decoding through the verify pass one row at a time, whatever the draft proposes
//! ([`greedy_by_verify`]).

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaEvent, CudaSlice, CudaStream, CudaView, sys::CUevent_flags};
use half::bf16;
use inkling::Checkpoint;

use crate::{
    draft::{Draft, STEPS, WINDOW},
    model::{Accept, Model, PrefillLogits, VerifyTokens},
    ops::{linear::Linear, spec::TAPS},
    weights::draft::{DraftWeights, HIDDEN},
};

/// Prompt rows whose taps [`Speculator::restart`] commits again: the slots of the draft's
/// local rings, the window and a block.
const TAIL: usize = WINDOW + STEPS + 1;

/// Where a round's drafts come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drafts {
    /// The DSpark draft.
    Dspark,
    /// Uniformly random token ids from a seeded generator, which the target almost never
    /// accepts. The output must not change.
    Random(u64),
}

/// Counts and times of the rounds since the last [`Speculator::generate`] started.
#[derive(Clone, Debug, Default)]
pub struct SpecStats {
    pub rounds: usize,
    /// Tokens emitted, the first token from the prefill excluded.
    pub emitted: usize,
    /// Rounds in which draft `i` was kept, for `i` from 0.
    pub accepted_at: Vec<usize>,
    /// Wall time of the whole generation.
    pub wall: Duration,
    /// GPU time of the verify passes, from the start of each pass to its verdict on the
    /// host.
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

/// The draft and the round loop for one sequence of a [`Model`].
pub struct Speculator {
    draft: Draft,
    linear: Linear,
    /// Drafts verified per round.
    k: usize,
    /// Random drafts of [`Drafts::Random`], uploaded each round.
    random: CudaSlice<i32>,
    /// The taps of the prompt's last rows, up to [`TAIL`] of them, oldest first, for
    /// [`Self::restart`]; `spare` takes the next copy while one is built.
    tail: CudaSlice<bf16>,
    spare: CudaSlice<bf16>,
    tail_rows: usize,
    prompt_len: usize,
    stats: SpecStats,
}

impl Speculator {
    /// Loads the draft from `checkpoint` for `model`, with room for `capacity` positions,
    /// verifying `k` drafts per round (1 to 7).
    pub fn new(
        model: &Model,
        checkpoint: &Checkpoint,
        capacity: usize,
        k: usize,
    ) -> anyhow::Result<Self> {
        ensure!((1..=STEPS).contains(&k), "k is {k}, expected 1 to {STEPS}");
        let stream = model.stream();
        let weights = DraftWeights::load(checkpoint, model.config(), stream)?;
        let draft = Draft::new(
            stream,
            model.config(),
            weights,
            model.draft_shared(),
            capacity + STEPS + 1,
        )?;
        Ok(Self {
            draft,
            linear: Linear::load(stream.context())?,
            k,
            random: stream.alloc_zeros(STEPS)?,
            tail: stream.alloc_zeros(TAIL * TAPS * HIDDEN)?,
            spare: stream.alloc_zeros(TAIL * TAPS * HIDDEN)?,
            tail_rows: 0,
            prompt_len: 0,
            stats: SpecStats::default(),
        })
    }

    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Verifies `k` drafts per round from now on, 1 to 7.
    pub fn set_k(&mut self, k: usize) -> anyhow::Result<()> {
        ensure!((1..=STEPS).contains(&k), "k is {k}, expected 1 to {STEPS}");
        self.k = k;
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
        self.tail_rows = 0;
        let Self {
            draft,
            linear,
            tail,
            spare,
            tail_rows,
            ..
        } = self;
        let row = TAPS * HIDDEN;
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
    /// the prompt's last positions in the draft's local layers, but nothing of its global
    /// layer below the prompt's end, so committing the prompt's last [`TAIL`] rows again
    /// restores the draft.
    pub fn restart(&mut self, model: &mut Model, first: u32) -> anyhow::Result<()> {
        ensure!(self.prompt_len > 0, "nothing was prefilled");
        ensure!(
            model.position() == self.prompt_len,
            "the model is at {}, the prompt ends at {}",
            model.position(),
            self.prompt_len
        );
        let stream = Arc::clone(model.stream());
        let row = TAPS * HIDDEN;
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
    pub fn generate(
        &mut self,
        model: &mut Model,
        n: usize,
        drafts: Drafts,
    ) -> anyhow::Result<Vec<u32>> {
        let stream = Arc::clone(model.stream());
        let vocab = model.config().unpadded_vocab_size as u64;
        let mut rng = match drafts {
            Drafts::Random(seed) => seed,
            Drafts::Dspark => 0,
        };
        self.stats = SpecStats {
            accepted_at: vec![0; self.k],
            ..SpecStats::default()
        };
        let mut timings: Vec<[CudaEvent; 3]> = Vec::new();
        let started = Instant::now();
        let mut out = Vec::with_capacity(n + self.k + 1);
        if drafts == Drafts::Dspark {
            self.draft.round(
                &stream,
                &self.linear,
                model.draft_shared(),
                model.verify_anchor()?,
            )?;
        }
        while out.len() < n {
            let before = timing_event(&stream)?;
            let verified = timing_event(&stream)?;
            let drafted = timing_event(&stream)?;
            before.record(&stream)?;
            let start = model.position();
            let verdict = match drafts {
                Drafts::Dspark => model.verify(
                    VerifyTokens::Drafts(self.draft.draft_ids().slice(..self.k)),
                    Accept::Greedy,
                )?,
                Drafts::Random(_) => {
                    let ids: Vec<i32> = (0..self.k)
                        .map(|_| i32::try_from(splitmix64(&mut rng) % vocab))
                        .collect::<Result<_, _>>()?;
                    stream.memcpy_htod(&ids, &mut self.random.slice_mut(..self.k))?;
                    model.verify(
                        VerifyTokens::Drafts(self.random.slice(..self.k)),
                        Accept::Greedy,
                    )?
                }
            };
            verified.record(&stream)?;
            let a = verdict.accepted;
            out.extend(verdict.emitted());
            self.stats.rounds += 1;
            for kept in &mut self.stats.accepted_at[..a] {
                *kept += 1;
            }
            if drafts == Drafts::Dspark {
                let taps = model.verify_taps()?;
                let kept = taps.slice(..(a + 1) * TAPS * HIDDEN);
                self.draft.commit(&stream, &self.linear, &kept, start)?;
                self.draft.round(
                    &stream,
                    &self.linear,
                    model.draft_shared(),
                    model.verify_anchor()?,
                )?;
            }
            drafted.record(&stream)?;
            timings.push([before, verified, drafted]);
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
