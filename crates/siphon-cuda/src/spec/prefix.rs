//! The prefix cache: the state of recent prompts at their last chunk boundary, one entry
//! per sequence slot, so that a prompt that starts with an entry's tokens prefills only
//! the rest.
//!
//! The state of a sequence at position `p` lies in its slot's global KV caches, whose
//! positions below `p` later tokens leave alone, and in the slot's sliding rings and conv
//! states, which later tokens overwrite. An entry copies the second kind at a chunk
//! boundary (a [`LocalState`]) and leaves the first in the slot. It also copies the draft's
//! rings when the prompt was prefilled with the draft (`draft_sequence`). The draft keeps
//! one sequence: a bundled draft's global layer holds only the positions of the sequence it
//! last started, so its rings are of use only while that sequence is the entry's, while a
//! DFlash 2 draft has rings alone and any entry's rings restore it.
//!
//! An entry is live while its slot holds the same sequence ([`Model::slot_sequence`]) and
//! it was used within the time to live. Anything that starts another sequence in the slot
//! (a reset, a [`Model::move_slot`] into it) so drops the entry without the cache's help.

use std::time::{Duration, Instant};

use anyhow::{Context as _, ensure};

use super::Speculator;
use crate::{
    draft::DraftRings,
    model::{LocalState, Model},
};

/// The entries of a [`Speculator`]'s prefix cache.
pub(super) struct PrefixCache {
    entries: Vec<Entry>,
    /// How long an unused entry lives.
    ttl: Duration,
    /// Whether the draft has global layers ([`crate::draft::Draft::has_global_layers`]).
    draft_global: bool,
}

/// One prompt's state at its last chunk boundary.
struct Entry {
    /// The slot whose global caches hold the prompt's positions, `None` when unused.
    slot: Option<usize>,
    /// [`Model::slot_sequence`] of `slot` when the entry was saved or moved there.
    sequence: u64,
    /// The prompt's tokens below the boundary.
    tokens: Vec<u32>,
    /// The prefill chunk the state was prefilled with.
    chunk: usize,
    used: Instant,
    model: LocalState,
    draft: DraftRings,
    /// The draft sequence ([`Speculator::draft_sequence`]) whose rings `draft` holds, `None`
    /// when the prompt was prefilled without the draft.
    draft_sequence: Option<u64>,
}

/// The longest live entry a prompt starts with, from [`Speculator::prefix_lookup`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixHit {
    /// The slot that holds the entry.
    pub slot: usize,
    /// Prompt tokens whose state the entry holds.
    pub tokens: usize,
    /// Whether the entry also holds the draft's state, so that the prompt can speculate
    /// right after its prefill.
    pub draft: bool,
}

impl PrefixCache {
    fn live(&self, e: &Entry, model: &Model, now: Instant) -> bool {
        let Some(slot) = e.slot else {
            return false;
        };
        !e.tokens.is_empty()
            && now.saturating_duration_since(e.used) <= self.ttl
            && model.slot_sequence(slot).is_ok_and(|s| s == e.sequence)
            && model.slot_position(slot).is_ok_and(|p| p >= e.tokens.len())
    }

    fn has_draft(&self, e: &Entry, draft_sequence: u64) -> bool {
        e.draft_sequence
            .is_some_and(|s| !self.draft_global || s == draft_sequence)
    }

    /// The longest live entry outside `busy` slots that `prompt` starts with and goes
    /// past, with the draft's state if `draft` is `Some` (the current draft sequence).
    fn lookup(
        &self,
        model: &Model,
        prompt: &[u32],
        chunk: usize,
        busy: &[usize],
        draft: Option<u64>,
        now: Instant,
    ) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                self.live(e, model, now)
                    && e.slot.is_some_and(|s| !busy.contains(&s))
                    && e.chunk == chunk
                    && prompt.len() > e.tokens.len()
                    && prompt.starts_with(&e.tokens)
                    && draft.is_none_or(|d| self.has_draft(e, d))
            })
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, _)| i)
    }

    /// The live entry of slot `slot`.
    fn of_slot(&self, model: &Model, slot: usize, now: Instant) -> Option<usize> {
        self.entries
            .iter()
            .position(|e| e.slot == Some(slot) && self.live(e, model, now))
    }

    /// An entry to save a new prompt into: one that is not live, else the least recently
    /// used outside `busy` slots.
    fn free_entry(&self, model: &Model, busy: &[usize], now: Instant) -> Option<usize> {
        if let Some(i) = self.entries.iter().position(|e| !self.live(e, model, now)) {
            return Some(i);
        }
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.slot.is_some_and(|s| !busy.contains(&s)))
            .min_by_key(|(_, e)| e.used)
            .map(|(i, _)| i)
    }

    /// A slot outside `busy` for a new sequence: one without a live entry, else the one
    /// whose entry was used least recently. Slot 0 comes last among equals, since a request
    /// that runs alone speculates there.
    fn victim_slot(&self, model: &Model, busy: &[usize], now: Instant) -> Option<usize> {
        let candidates = || (1..model.slots()).chain([0]).filter(|s| !busy.contains(s));
        if let Some(slot) = candidates().find(|&s| self.of_slot(model, s, now).is_none()) {
            return Some(slot);
        }
        candidates().min_by_key(|&s| self.of_slot(model, s, now).map(|i| self.entries[i].used))
    }

    /// Drops the entry of slot `slot`, if any.
    fn drop_slot(&mut self, slot: usize) {
        for e in &mut self.entries {
            if e.slot == Some(slot) {
                e.slot = None;
            }
        }
    }
}

impl Speculator {
    /// Keeps a prefix cache of `entries` entries from now on, each dropped after `ttl`
    /// unused. Allocates each entry's copies of a slot's sliding rings and conv states and
    /// of the draft's rings, and returns their bytes.
    pub fn enable_prefix_cache(
        &mut self,
        model: &Model,
        entries: usize,
        ttl: Duration,
    ) -> anyhow::Result<usize> {
        ensure!(
            (1..=model.slots()).contains(&entries),
            "{entries} prefix cache entries; 1 to the {} sequence slots fit",
            model.slots()
        );
        let now = Instant::now();
        let entries = (0..entries)
            .map(|_| {
                Ok(Entry {
                    slot: None,
                    sequence: 0,
                    tokens: Vec::new(),
                    chunk: 0,
                    used: now,
                    model: model.local_state()?,
                    draft: self.draft.rings(model.stream())?,
                    draft_sequence: None,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let bytes = entries
            .iter()
            .map(|e| e.model.bytes() + e.draft.bytes())
            .sum();
        self.prefix = Some(PrefixCache {
            entries,
            ttl,
            draft_global: self.draft.has_global_layers(),
        });
        Ok(bytes)
    }

    /// Drops entries after `ttl` unused from now on.
    pub fn set_prefix_ttl(&mut self, ttl: Duration) {
        if let Some(cache) = self.prefix.as_mut() {
            cache.ttl = ttl;
        }
    }

    /// The longest live entry outside `busy` slots that `prompt` starts with, if the cache
    /// is on. `prompt` must go past the entry's tokens.
    #[must_use]
    pub fn prefix_lookup(
        &self,
        model: &Model,
        prompt: &[u32],
        chunk: usize,
        busy: &[usize],
    ) -> Option<PrefixHit> {
        let cache = self.prefix.as_ref()?;
        let i = cache.lookup(model, prompt, chunk, busy, None, Instant::now())?;
        let e = &cache.entries[i];
        Some(PrefixHit {
            slot: e.slot?,
            tokens: e.tokens.len(),
            draft: cache.has_draft(e, self.draft_sequence),
        })
    }

    /// [`Self::prefill`] of `prompt` into slot 0 through the prefix cache
    /// ([`Self::enable_prefix_cache`]), for a request that runs alone: no slot runs a
    /// request. Returns the first anchor and the prompt tokens whose state came from the
    /// cache.
    ///
    /// The longest live entry that holds the draft's state and that `prompt` starts with
    /// moves to slot 0 ([`Self::move_to_slot0`]), and its copies return slot 0 and the
    /// draft to the entry's boundary; then only the rest is prefilled, in the same chunks
    /// as a whole prompt from position 0, so the results equal [`Self::prefill`]'s bit for
    /// bit. Without such an entry, the entry of slot 0 moves to a free slot and slot 0
    /// prefills from position 0. Either way slot 0's entry then takes `prompt`'s state at
    /// its own last multiple of `chunk` below its end. Without a cache this prefills the
    /// whole prompt.
    pub fn prefill_cached(
        &mut self,
        model: &mut Model,
        prompt: &[u32],
        chunk: usize,
    ) -> anyhow::Result<(u32, usize)> {
        ensure!(!prompt.is_empty(), "nothing to prefill");
        let now = Instant::now();
        let Some(cache) = self.prefix.as_ref() else {
            model.reset()?;
            return Ok((self.prefill(model, prompt, chunk)?, 0));
        };
        let hit = cache.lookup(model, prompt, chunk, &[], Some(self.draft_sequence), now);
        let start = if let Some(i) = hit {
            let e = &cache.entries[i];
            let (slot, len) = (e.slot.context("a live entry has a slot")?, e.tokens.len());
            self.move_to_slot0(model, slot)?;
            len
        } else {
            self.vacate_slot0(model, &[])?;
            0
        };
        self.start_prompt(model, prompt)?;
        let stream = std::sync::Arc::clone(model.stream());
        let cache = self.prefix.as_mut().context("the cache is on")?;
        if let Some(i) = hit {
            let e = &mut cache.entries[i];
            model.restore_local_state(0, &e.model)?;
            self.draft.restore_rings(&stream, &e.draft)?;
            e.used = now;
        } else {
            model.reset()?;
            // The draft's global caches now take another sequence.
            self.draft_sequence += 1;
        }
        // The last chunk boundary below the end, so at least one row remains for the
        // logits. `start` is such a boundary of a shorter prompt, so it is at most this.
        let boundary = (prompt.len() - 1) / chunk * chunk;
        if boundary > start {
            // Prefill splits a chunk only at the end of its tokens, so the chunks match a
            // prefill of the whole prompt.
            self.prefill_rows(model, &prompt[start..boundary], chunk)?;
            let cache = self.prefix.as_ref().context("the cache is on")?;
            if let Some(i) = hit.or_else(|| cache.free_entry(model, &[], now)) {
                self.save_entry(model, i, 0, &prompt[..boundary], chunk)?;
                self.save_entry_rings(model, i)?;
            }
        }
        let first = self.prefill_rows(model, &prompt[boundary..], chunk)?;
        Ok((self.end_prompt(model, prompt.len(), first)?, start))
    }

    /// Picks a slot outside `busy` for a request that runs batched, and readies it: the
    /// slot of the longest live entry that `prompt` starts with, returned to that entry's
    /// boundary, else a slot without a live entry, else the slot of the least recently used
    /// entry, reset. Returns the slot and the prompt tokens whose state it holds, from
    /// which the request prefills the rest in chunks of `chunk` ([`Model::prefill_slot`]),
    /// as a prefill from position 0 would. Without a cache, takes the first slot outside
    /// `busy`, slot 0 last.
    pub fn claim_slot(
        &mut self,
        model: &mut Model,
        prompt: &[u32],
        chunk: usize,
        busy: &[usize],
    ) -> anyhow::Result<(usize, usize)> {
        let now = Instant::now();
        let slot = if let Some(cache) = self.prefix.as_mut() {
            if let Some(i) = cache.lookup(model, prompt, chunk, busy, None, now) {
                let e = &mut cache.entries[i];
                let slot = e.slot.context("a live entry has a slot")?;
                model.restore_local_state(slot, &e.model)?;
                e.used = now;
                return Ok((slot, e.tokens.len()));
            }
            let slot = cache
                .victim_slot(model, busy, now)
                .context("every slot runs a request")?;
            cache.drop_slot(slot);
            slot
        } else {
            (1..model.slots())
                .chain([0])
                .find(|s| !busy.contains(s))
                .context("every slot runs a request")?
        };
        model.reset_slot(slot)?;
        Ok((slot, 0))
    }

    /// After a batched request in slot `slot` has prefilled `prompt` up to its last
    /// multiple of `chunk` below its end, and no further, keeps that state in the slot's
    /// entry, or in a free or least recently used one outside `busy` slots. The entry holds
    /// no draft state. Does nothing without a cache or below the first boundary.
    pub fn save_prefix(
        &mut self,
        model: &mut Model,
        slot: usize,
        prompt: &[u32],
        chunk: usize,
        busy: &[usize],
    ) -> anyhow::Result<()> {
        let now = Instant::now();
        let Some(cache) = self.prefix.as_ref() else {
            return Ok(());
        };
        let boundary = prompt.len().saturating_sub(1) / chunk * chunk;
        if boundary == 0 {
            return Ok(());
        }
        ensure!(
            model.slot_position(slot)? == boundary,
            "slot {slot} is at {}, not at the prompt's boundary {boundary}",
            model.slot_position(slot)?
        );
        let entry = cache
            .of_slot(model, slot, now)
            .or_else(|| cache.free_entry(model, busy, now));
        if let Some(i) = entry {
            self.save_entry(model, i, slot, &prompt[..boundary], chunk)?;
        }
        Ok(())
    }

    /// A slot outside `busy` whose sequence may be lost, for [`Self::rebuild`]: one without
    /// a live entry, else the one whose entry was used least recently, whose entry is then
    /// dropped. Without a cache, the first slot outside `busy`.
    pub fn idle_slot(&mut self, model: &Model, busy: &[usize]) -> Option<usize> {
        let Some(cache) = self.prefix.as_mut() else {
            return (0..model.slots()).find(|s| !busy.contains(s));
        };
        let slot = cache.victim_slot(model, busy, Instant::now())?;
        cache.drop_slot(slot);
        Some(slot)
    }

    /// Moves the sequence in slot `from` to slot 0, where speculation runs, with its entry.
    /// No slot but `from` may run a request. Slot 0's entry first moves to a slot without a
    /// live entry, or to the slot of an entry used less recently, if there is one.
    pub fn move_to_slot0(&mut self, model: &mut Model, from: usize) -> anyhow::Result<()> {
        if from == 0 {
            return Ok(());
        }
        self.vacate_slot0(model, &[from])?;
        let now = Instant::now();
        let entry = self
            .prefix
            .as_ref()
            .and_then(|c| c.of_slot(model, from, now));
        model.move_slot(from, 0)?;
        if let (Some(cache), Some(i)) = (self.prefix.as_mut(), entry) {
            let e = &mut cache.entries[i];
            e.slot = Some(0);
            e.sequence = model.slot_sequence(0)?;
        }
        Ok(())
    }

    /// Moves slot 0's sequence and entry to another slot outside `busy`, before slot 0
    /// takes another sequence: to a slot without a live entry, else to the slot of the
    /// least recently used entry if that one was used before slot 0's.
    fn vacate_slot0(&mut self, model: &mut Model, busy: &[usize]) -> anyhow::Result<()> {
        let now = Instant::now();
        let Some(cache) = self.prefix.as_mut() else {
            return Ok(());
        };
        let Some(i) = cache.of_slot(model, 0, now) else {
            return Ok(());
        };
        let mut busy = busy.to_vec();
        busy.push(0);
        let Some(to) = cache.victim_slot(model, &busy, now) else {
            return Ok(());
        };
        if let Some(j) = cache.of_slot(model, to, now)
            && cache.entries[j].used >= cache.entries[i].used
        {
            return Ok(());
        }
        cache.drop_slot(to);
        model.move_slot(0, to)?;
        let e = &mut cache.entries[i];
        e.slot = Some(to);
        e.sequence = model.slot_sequence(to)?;
        Ok(())
    }

    /// Saves slot `slot`'s state and `tokens`, the prompt below its boundary, into entry
    /// `i`, without the draft's state.
    fn save_entry(
        &mut self,
        model: &mut Model,
        i: usize,
        slot: usize,
        tokens: &[u32],
        chunk: usize,
    ) -> anyhow::Result<()> {
        let cache = self.prefix.as_mut().context("the cache is on")?;
        // Nothing is kept until the copies are whole, in case this fails midway.
        cache.entries[i].slot = None;
        cache.drop_slot(slot);
        let e = &mut cache.entries[i];
        model.save_local_state(slot, &mut e.model)?;
        e.tokens.clear();
        e.tokens.extend_from_slice(tokens);
        e.chunk = chunk;
        e.used = Instant::now();
        e.draft_sequence = None;
        e.sequence = model.slot_sequence(slot)?;
        e.slot = Some(slot);
        Ok(())
    }

    /// Copies the draft's rings into entry `i`, whose boundary the draft has just
    /// committed up to.
    pub(super) fn save_entry_rings(&mut self, model: &Model, i: usize) -> anyhow::Result<()> {
        let sequence = self.draft_sequence;
        let cache = self.prefix.as_mut().context("the cache is on")?;
        ensure!(
            self.draft.committed() == cache.entries[i].tokens.len(),
            "the draft committed {} positions, the entry holds {}",
            self.draft.committed(),
            cache.entries[i].tokens.len()
        );
        // The draft's global layers hold one sequence's positions, so only one entry may
        // claim them.
        if cache.draft_global {
            for e in &mut cache.entries {
                if e.draft_sequence == Some(sequence) {
                    e.draft_sequence = None;
                }
            }
        }
        let e = &mut cache.entries[i];
        self.draft.save_rings(model.stream(), &mut e.draft)?;
        e.draft_sequence = Some(sequence);
        Ok(())
    }

    /// The live entry of slot `slot` whose tokens `tokens` starts with and goes past.
    pub(super) fn entry_starting(
        &self,
        model: &Model,
        slot: usize,
        tokens: &[u32],
        chunk: usize,
    ) -> Option<usize> {
        let cache = self.prefix.as_ref()?;
        let i = cache.of_slot(model, slot, Instant::now())?;
        let e = &cache.entries[i];
        (e.chunk == chunk && tokens.len() > e.tokens.len() && tokens.starts_with(&e.tokens))
            .then_some(i)
    }

    /// Tokens of entry `i`.
    pub(super) fn entry_len(&self, i: usize) -> usize {
        self.prefix
            .as_ref()
            .map_or(0, |c| c.entries[i].tokens.len())
    }
}
