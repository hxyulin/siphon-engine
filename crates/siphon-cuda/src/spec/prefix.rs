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
//!
//! With a host tier ([`Speculator::enable_host_cache`], [`super::host`]), every saved entry
//! is also copied to host memory, global caches included, and a prompt whose entry died
//! on the device restores from there into the slot it takes.
//!
//! Entries, device and host, hold and match a prompt's key ([`Prompt::key`]), not its
//! tokens: the tokens with each image's placeholders replaced by ids of the image's hash.
//! Every function here that takes a `prompt: &[u32]` takes that key.

use std::time::{Duration, Instant};

use anyhow::{Context as _, ensure};
use siphon::prefix_hash::{PrefixHashes, prefix_hash};

use super::{
    Prompt, Speculator,
    host::{DiskLoaded, DiskSaved, DiskTierConfig, DiskTierStats, HOST_CHUNK_POSITIONS, HostCache},
};
use crate::{
    draft::{Draft, DraftRings},
    model::{LocalState, Model},
};

/// The entries of a [`Speculator`]'s prefix cache.
pub(super) struct PrefixCache {
    entries: Vec<Entry>,
    /// How long an unused entry lives.
    ttl: Duration,
    /// Whether the draft has global layers ([`crate::draft::Draft::has_global_layers`]).
    draft_global: bool,
    /// The host tier, with [`Speculator::enable_host_cache`].
    host: Option<HostCache>,
    /// Where the last [`Speculator::prefill_cached`] or [`Speculator::claim_slot`] found
    /// its prefix, `None` for a miss.
    last_tier: Option<PrefixTier>,
    last_restore_seconds: f64,
    /// Lines for the server's log about dropped entries, drained by
    /// [`Speculator::cache_notes`].
    notes: Vec<String>,
}

/// The tier a prompt's cached prefix came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixTier {
    /// A device entry, whose global caches stay in its slot's pages of the KV pool.
    Slot,
    /// A host entry in page-locked RAM.
    Host,
    /// An entry promoted from the active L3 NVMe disk tier.
    Disk,
}

impl PrefixTier {
    /// The tier's name in the server's log.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Slot => "slot",
            Self::Host => "host",
            Self::Disk => "disk",
        }
    }
}

/// Where a prompt's longest cached prefix lies.
#[derive(Clone, Copy, Debug)]
enum Source {
    /// A device entry.
    Device(usize),
    /// A host entry, and its tokens.
    Host(usize, usize),
    /// An L3 disk entry id, and its tokens.
    Disk(u64, usize),
}

/// One prompt's state at its last chunk boundary.
struct Entry {
    /// The slot whose global caches hold the prompt's positions, `None` when unused.
    slot: Option<usize>,
    /// [`Model::slot_sequence`] of `slot` when the entry was saved or moved there.
    sequence: u64,
    /// The prompt's key ([`Prompt::key`]) below the boundary.
    tokens: Vec<u32>,
    /// Cumulative hash of `tokens` ([`prefix_hash`]), for O(1) mismatch rejection.
    hash: u64,
    /// The prefill chunk the state was prefilled with.
    chunk: usize,
    used: Instant,
    model: LocalState,
    draft: DraftRings,
    /// The draft sequence ([`Speculator::draft_sequence`]) whose rings `draft` holds, `None`
    /// when the prompt was prefilled without the draft.
    draft_sequence: Option<u64>,
}

/// The longest entry a prompt starts with, from [`Speculator::prefix_lookup`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixHit {
    /// The slot that holds the entry, `None` for an entry in host memory.
    pub slot: Option<usize>,
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
    #[expect(clippy::too_many_arguments, reason = "prefix cache lookup criteria")]
    fn lookup(
        &self,
        model: &Model,
        prompt: &[u32],
        chunk: usize,
        busy: &[usize],
        draft: Option<u64>,
        now: Instant,
        hashes: &PrefixHashes,
    ) -> Option<usize> {
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let len = e.tokens.len();
            if len == 0
                || best.is_some_and(|(_, best_len)| len <= best_len)
                || prompt.len() <= len
                || e.chunk != chunk
                || !draft.is_none_or(|d| self.has_draft(e, d))
                || e.slot.is_none_or(|s| busy.contains(&s))
                || !self.live(e, model, now)
            {
                continue;
            }
            if hashes.hash_at_len(len) == Some(e.hash) && prompt.starts_with(&e.tokens) {
                best = Some((i, len));
            }
        }
        best.map(|(i, _)| i)
    }

    /// The longest entry outside `busy` slots that `prompt` starts with and goes past: the
    /// live device entry of [`Self::lookup`], or a host entry if one is longer. With
    /// `draft`, only entries with the draft's state count.
    fn pick(
        &self,
        model: &Model,
        prompt: &[u32],
        chunk: usize,
        busy: &[usize],
        draft: Option<u64>,
        now: Instant,
    ) -> Option<Source> {
        let device_max = self
            .entries
            .iter()
            .filter(|e| e.chunk == chunk && e.tokens.len() < prompt.len())
            .map(|e| e.tokens.len())
            .max()
            .unwrap_or(0);
        let host_max = self
            .host
            .as_ref()
            .filter(|_| chunk == HOST_CHUNK_POSITIONS)
            .map_or(0, |h| {
                h.max_len_below(prompt.len())
                    .max(h.disk_max_len_below(prompt.len()))
            });
        let max_len = device_max.max(host_max);
        if max_len == 0 || chunk == 0 {
            return None;
        }
        let hashes = PrefixHashes::new(prompt, chunk, max_len / chunk);
        let device = self.lookup(model, prompt, chunk, busy, draft, now, &hashes);
        let device_len = device.map_or(0, |i| self.entries[i].tokens.len());
        let host = self
            .host
            .as_ref()
            .filter(|_| chunk == HOST_CHUNK_POSITIONS)
            .and_then(|h| {
                let j = h.lookup(prompt, draft.is_some(), &hashes)?;
                (h.len(j) > device_len).then(|| Source::Host(j, h.len(j)))
            });
        let best_so_far = host.or(device.map(Source::Device));
        let best_len = match best_so_far {
            Some(Source::Device(_)) => device_len,
            Some(Source::Host(_, len) | Source::Disk(_, len)) => len,
            None => 0,
        };
        let disk = self
            .host
            .as_ref()
            .filter(|_| chunk == HOST_CHUNK_POSITIONS)
            .and_then(|h| {
                let (id, len) = h.lookup_disk(prompt, draft.is_some(), &hashes)?;
                (len > best_len).then_some(Source::Disk(id, len))
            });
        disk.or(best_so_far)
    }

    /// Makes the model's stream wait for the host tier's copies before it overwrites what
    /// they read.
    fn fence(&self) -> anyhow::Result<()> {
        self.host.as_ref().map_or(Ok(()), HostCache::fence)
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

    /// Returns the pages of the copies of every entry that is not live to the pool, once
    /// the host tier's copies from them are done. Returns the pages freed.
    fn release_dead(&mut self, model: &Model) -> anyhow::Result<usize> {
        let now = Instant::now();
        let dead: Vec<usize> = (0..self.entries.len())
            .filter(|&i| {
                self.entries[i].model.pages() > 0 && !self.live(&self.entries[i], model, now)
            })
            .collect();
        if dead.is_empty() {
            return Ok(0);
        }
        self.fence()?;
        let mut freed = 0;
        for i in dead {
            let e = &mut self.entries[i];
            freed += e.model.pages();
            e.model.release(model.stream())?;
            e.slot = None;
        }
        Ok(freed)
    }

    /// The slot outside `busy` that a parked sequence returns to: the idle slot without a
    /// live entry that holds the most pages, which it then maps the fewest of, else the
    /// slot of [`Self::victim_slot`].
    fn return_slot(&self, model: &Model, busy: &[usize], now: Instant) -> Option<usize> {
        (0..model.slots())
            .filter(|s| !busy.contains(s) && self.of_slot(model, *s, now).is_none())
            .max_by_key(|&s| model.slot_pages(s).unwrap_or(0))
            .or_else(|| self.victim_slot(model, busy, now))
    }

    /// Notes for the log that entry `i`, live until now, is dropped, and `why`. Says
    /// nothing for an empty entry.
    fn note_drop(&mut self, i: usize, now: Instant, why: &str) {
        let e = &self.entries[i];
        if e.tokens.is_empty() {
            return;
        }
        let slot = e.slot.map_or_else(String::new, |s| format!(" in slot {s}"));
        self.notes.push(format!(
            "prefix cache: dropped the entry of {} tokens{slot}, unused for {:.0} s: {why}",
            e.tokens.len(),
            now.saturating_duration_since(e.used).as_secs_f64()
        ));
    }

    /// [`Self::drop_slot`] of a slot that [`Self::victim_slot`] picked for a new sequence,
    /// noting for the log the live entry it drops: every slot held one.
    fn drop_victim(&mut self, model: &Model, slot: usize, now: Instant) {
        if let Some(i) = self.of_slot(model, slot, now) {
            self.note_drop(
                i,
                now,
                "every free slot holds a cached prompt, and it was the least recently used",
            );
        }
        self.drop_slot(slot);
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
                    hash: 0,
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
            host: None,
            last_tier: None,
            last_restore_seconds: 0.0,
            notes: Vec::new(),
        });
        Ok(bytes)
    }

    /// Adds a host tier of `bytes` of page-locked host memory to the prefix cache
    /// ([`Self::enable_prefix_cache`]), which every saved entry is copied to. Returns the
    /// chunks the memory holds and the bytes of each.
    pub fn enable_host_cache(
        &mut self,
        model: &Model,
        bytes: usize,
    ) -> anyhow::Result<(usize, usize)> {
        let cache = self
            .prefix
            .as_mut()
            .context("the host cache needs the prefix cache")?;
        let host = HostCache::new(model, &self.draft, bytes)?;
        let chunks = host.chunks();
        cache.host = Some(host);
        Ok(chunks)
    }

    /// Copies every live device entry to the host tier, so that the conversations the
    /// slots hold are there too, then writes every host entry to files in `dir` with a
    /// manifest that records `identity` ([`HostCache::save_dir`]).
    pub fn save_host_cache(
        &mut self,
        model: &Model,
        dir: &std::path::Path,
        identity: &serde_json::Value,
    ) -> anyhow::Result<DiskSaved> {
        let now = Instant::now();
        let cache = self.prefix.as_ref().context("the prefix cache is off")?;
        let live: Vec<(usize, bool)> = (0..cache.entries.len())
            .filter(|&i| cache.live(&cache.entries[i], model, now))
            .map(|i| (i, cache.has_draft(&cache.entries[i], self.draft_sequence)))
            .collect();
        for (i, draft) in live {
            self.save_host(model, i, draft);
        }
        let host = self
            .prefix
            .as_mut()
            .and_then(|c| c.host.as_mut())
            .context("the host cache is off")?;
        host.save_dir(dir, identity)
    }

    /// Reads the host entries that [`Self::save_host_cache`] wrote to `dir`, if its
    /// manifest matches `identity` and this host tier's layout, and otherwise removes them.
    ///
    /// # Errors
    ///
    /// Returns an error if the host tier is disabled or the files cannot be listed or
    /// discarded. Individual entries that cannot be restored are counted as skipped.
    pub fn load_host_cache(
        &mut self,
        dir: &std::path::Path,
        identity: &serde_json::Value,
    ) -> anyhow::Result<DiskLoaded> {
        self.prefix
            .as_mut()
            .and_then(|c| c.host.as_mut())
            .context("the host cache is off")?
            .load_dir(dir, identity)
    }

    /// Enables an active L3 NVMe disk cache tier backed by `config.dir`, indexing any
    /// existing compatible entries and pre-warming pinned host RAM with the newest ones
    /// that fit.
    ///
    /// # Errors
    ///
    /// Returns an error if the host tier is disabled or the cache directory cannot be
    /// initialized.
    pub fn enable_disk_cache(
        &mut self,
        config: DiskTierConfig,
        identity: &serde_json::Value,
    ) -> anyhow::Result<DiskLoaded> {
        self.prefix
            .as_mut()
            .and_then(|c| c.host.as_mut())
            .context("the disk cache needs the host cache")?
            .enable_disk_cache(config, identity)
    }

    /// Returns statistics for the active L3 disk cache tier, if enabled.
    #[must_use]
    pub fn disk_cache_stats(&self) -> Option<DiskTierStats> {
        self.prefix
            .as_ref()
            .and_then(|c| c.host.as_ref())
            .and_then(HostCache::disk_stats)
    }

    /// Makes the model's stream wait for the host tier's copies before it overwrites what
    /// they read: a slot's caches or the draft's.
    pub(super) fn fence_host(&self) -> anyhow::Result<()> {
        self.prefix.as_ref().map_or(Ok(()), PrefixCache::fence)
    }

    /// Lines about the prefix cache's dropped entries and the host tier's saves, restores
    /// and evictions since the last call, for a log.
    pub fn cache_notes(&mut self) -> Vec<String> {
        let Some(cache) = self.prefix.as_mut() else {
            return Vec::new();
        };
        let mut notes = std::mem::take(&mut cache.notes);
        notes.extend(cache.host.as_mut().map_or_else(Vec::new, HostCache::notes));
        notes
    }

    /// Where the last [`Self::prefill_cached`] or [`Self::claim_slot`] found its prompt's
    /// prefix, `None` for a miss or without a cache.
    #[must_use]
    pub fn last_prefix_tier(&self) -> Option<PrefixTier> {
        self.prefix.as_ref().and_then(|c| c.last_tier)
    }

    /// The last prefix hit's restore time in seconds. Includes disk reads and host
    /// transfers when applicable. Device copies are timed through submission.
    #[must_use]
    pub fn last_prefix_restore_seconds(&self) -> f64 {
        self.prefix.as_ref().map_or(0.0, |c| c.last_restore_seconds)
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
        match cache.pick(model, prompt, chunk, busy, None, Instant::now())? {
            Source::Device(i) => {
                let e = &cache.entries[i];
                Some(PrefixHit {
                    slot: Some(e.slot?),
                    tokens: e.tokens.len(),
                    draft: cache.has_draft(e, self.draft_sequence),
                })
            }
            Source::Host(j, tokens) => Some(PrefixHit {
                slot: None,
                tokens,
                draft: cache.host.as_ref()?.has_draft(j),
            }),
            Source::Disk(id, tokens) => Some(PrefixHit {
                slot: None,
                tokens,
                draft: cache.host.as_ref()?.has_disk_draft(id),
            }),
        }
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
    /// bit. A longer host entry with the draft's state ([`Self::enable_host_cache`]) is
    /// copied into slot 0 and the draft instead, after slot 0's entry moves aside, and slot
    /// 0's entry takes it at once. Without either, the entry of slot 0 moves to a free slot
    /// and slot 0 prefills from position 0. Either way slot 0's entry then takes `prompt`'s
    /// state at its own last multiple of `chunk` below its end. Without a cache this
    /// prefills the whole prompt.
    pub fn prefill_cached(
        &mut self,
        model: &mut Model,
        prompt: Prompt<'_>,
        chunk: usize,
    ) -> anyhow::Result<(u32, usize)> {
        self.prefill_cached_with_progress(model, prompt, chunk, |_| {})
    }

    /// Prefills like [`Self::prefill_cached`] and reports the number of new tokens
    /// processed in each submitted chunk. Restoring cached state does not report progress.
    pub fn prefill_cached_with_progress(
        &mut self,
        model: &mut Model,
        prompt: Prompt<'_>,
        chunk: usize,
        mut progress: impl FnMut(usize),
    ) -> anyhow::Result<(u32, usize)> {
        prompt.check()?;
        let now = Instant::now();
        let Some(cache) = self.prefix.as_ref() else {
            model.reset()?;
            return Ok((self.prefill_whole(model, prompt, chunk, &mut progress)?, 0));
        };
        let Prompt {
            tokens,
            key,
            mut images,
        } = prompt;
        cache.fence()?;
        let source = cache.pick(model, key, chunk, &[], Some(self.draft_sequence), now);
        let tier = source.map(|s| match s {
            Source::Device(_) => PrefixTier::Slot,
            Source::Host(..) => PrefixTier::Host,
            Source::Disk(..) => PrefixTier::Disk,
        });
        let mut move_seconds = 0.0;
        let (mut hit, mut start) = match source {
            Some(Source::Device(i)) => {
                let e = &cache.entries[i];
                let (slot, len) = (e.slot.context("a live entry has a slot")?, e.tokens.len());
                let started = Instant::now();
                self.move_to_slot0(model, slot)?;
                move_seconds = started.elapsed().as_secs_f64();
                (Some(i), len)
            }
            Some(Source::Host(_, len) | Source::Disk(_, len)) => {
                self.vacate_slot0(model, &[])?;
                (None, len)
            }
            None => {
                self.vacate_slot0(model, &[])?;
                (None, 0)
            }
        };
        self.start_prompt(model, tokens)?;
        let stream = std::sync::Arc::clone(model.stream());
        let cache = self.prefix.as_mut().context("the cache is on")?;
        cache.last_tier = tier;
        cache.last_restore_seconds = 0.0;
        let mut from_host = false;
        match source {
            Some(Source::Device(i)) => {
                let started = Instant::now();
                let e = &mut cache.entries[i];
                model.restore_local_state(0, &e.model)?;
                self.draft.restore_rings(&stream, &e.draft)?;
                e.used = now;
                cache.last_restore_seconds = move_seconds + started.elapsed().as_secs_f64();
            }
            Some(Source::Host(j, len)) => {
                model.reset()?;
                // The draft's global caches now take another sequence.
                self.draft_sequence += 1;
                let host = cache.host.as_mut().context("a host hit has a host cache")?;
                cache.last_restore_seconds = restore_solo(host, model, &mut self.draft, j, len)?;
                from_host = true;
            }
            Some(Source::Disk(disk_id, len)) => {
                model.reset()?;
                self.draft_sequence += 1;
                let host = cache.host.as_mut().context("a disk hit has a host cache")?;
                if let Some((j, read_bytes, read_secs)) = host.promote_disk(disk_id)? {
                    host.note(promoted(len, read_bytes, read_secs));
                    cache.last_restore_seconds =
                        read_secs + restore_solo(host, model, &mut self.draft, j, len)?;
                    from_host = true;
                } else {
                    cache.last_tier = None;
                    start = 0;
                }
            }
            None => {
                model.reset()?;
                // The draft's global caches now take another sequence.
                self.draft_sequence += 1;
            }
        }
        // The last chunk boundary below the end, so at least one row remains for the
        // logits. `start` is such a boundary of a shorter prompt, so it is at most this.
        let boundary = (tokens.len() - 1) / chunk * chunk;
        if boundary > start || from_host {
            // Prefill splits a chunk only at the end of its tokens, so the chunks match a
            // prefill of the whole prompt.
            if boundary > start {
                let images = Prompt::images(&mut images);
                self.prefill_rows(
                    model,
                    &tokens[start..boundary],
                    images,
                    chunk,
                    &mut progress,
                )?;
            }
            // A state restored from the host goes into a device entry too, so that the
            // next turn finds it there.
            let cache = self.prefix.as_ref().context("the cache is on")?;
            hit = hit.or_else(|| cache.free_entry(model, &[], now));
            if let Some(i) = hit {
                self.save_entry(model, i, 0, &key[..boundary], chunk)?;
                self.save_entry_rings(model, i)?;
            }
        }
        let first = self.prefill_rows(model, &tokens[boundary..], images, chunk, &mut progress)?;
        Ok((self.end_prompt(model, tokens.len(), first)?, start))
    }

    /// Picks a slot outside `busy` for a request that runs batched, and readies it: the
    /// slot of the longest live entry that `prompt` starts with, returned to that entry's
    /// boundary, else a slot without a live entry, else the slot of the least recently used
    /// entry, reset. A longer host entry is copied into that reset slot, without the
    /// draft's state but for a draft of rings alone, and kept in a device entry outside
    /// `busy` slots too. Returns the slot and the prompt tokens whose state it holds, from
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
        self.invalidate_batch_draft();
        let now = Instant::now();
        let Some(cache) = self.prefix.as_mut() else {
            let slot = (1..model.slots())
                .chain([0])
                .find(|s| !busy.contains(s))
                .context("every slot runs a request")?;
            model.reset_slot(slot)?;
            self.draft.reset_slot(slot);
            return Ok((slot, 0));
        };
        cache.fence()?;
        let source = cache.pick(model, prompt, chunk, busy, None, now);
        cache.last_tier = source.map(|s| match s {
            Source::Device(_) => PrefixTier::Slot,
            Source::Host(..) => PrefixTier::Host,
            Source::Disk(..) => PrefixTier::Disk,
        });
        cache.last_restore_seconds = 0.0;
        if let Some(Source::Device(i)) = source {
            let started = Instant::now();
            let e = &mut cache.entries[i];
            let slot = e.slot.context("a live entry has a slot")?;
            model.restore_local_state(slot, &e.model)?;
            e.used = now;
            let len = e.tokens.len();
            if self.draft.is_dflash2() {
                if e.draft_sequence.is_some() {
                    self.draft
                        .restore_slot_rings(model.stream(), slot, &e.draft)?;
                } else {
                    self.draft.reset_slot(slot);
                }
            }
            cache.last_restore_seconds = started.elapsed().as_secs_f64();
            return Ok((slot, len));
        }
        let slot = cache
            .victim_slot(model, busy, now)
            .context("every slot runs a request")?;
        cache.drop_victim(model, slot, now);
        model.reset_slot(slot)?;
        if self.draft.is_dflash2() {
            self.draft.reset_slot(slot);
        }
        let (j, len, disk_seconds) = match source {
            Some(Source::Host(j, len)) => (j, len, 0.0),
            Some(Source::Disk(disk_id, len)) => {
                let host = cache.host.as_mut().context("a disk hit has a host cache")?;
                let Some((j, read_bytes, read_secs)) = host.promote_disk(disk_id)? else {
                    cache.last_tier = None;
                    return Ok((slot, 0));
                };
                host.note(promoted(len, read_bytes, read_secs));
                (j, len, read_secs)
            }
            Some(Source::Device(_)) | None => return Ok((slot, 0)),
        };
        let host = cache.host.as_mut().context("a host hit has a host cache")?;
        let started = host.start_restore()?;
        let bytes = host.restore_slot(model, j, slot)?;
        let seconds = host.finish_restore(started)?;
        cache.last_restore_seconds = disk_seconds + seconds;
        let rings = !cache.draft_global && host.has_rings(j);
        host.note(restored(
            len,
            slot,
            bytes,
            seconds,
            if rings {
                "with the draft's rings"
            } else {
                "without the draft's state"
            },
        ));
        // Into a device entry too, so that the next turn finds it there. A draft of rings
        // alone takes any entry's rings, so they come along.
        if let Some(i) = cache.free_entry(model, busy, now) {
            self.save_entry(model, i, slot, &prompt[..len], chunk)?;
            if rings {
                let sequence = self.draft_sequence;
                let cache = self.prefix.as_mut().context("the cache is on")?;
                let host = cache.host.as_ref().context("a host hit has a host cache")?;
                let e = &mut cache.entries[i];
                host.restore_rings(j, &mut e.draft)?;
                e.draft_sequence = Some(sequence);
                if self.draft.is_dflash2() {
                    self.draft
                        .restore_slot_rings(model.stream(), slot, &e.draft)?;
                }
            }
        }
        Ok((slot, len))
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
        cache.fence()?;
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
            let has_draft = self.draft.is_dflash2() && self.draft.slot_committed(slot) == boundary;
            if has_draft {
                let sequence = self.draft_sequence;
                let cache = self.prefix.as_mut().context("the cache is on")?;
                let e = &mut cache.entries[i];
                self.draft
                    .save_slot_rings(model.stream(), slot, &mut e.draft)?;
                e.draft_sequence = Some(sequence);
            }
            self.save_host(model, i, has_draft);
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
        cache.fence().ok()?;
        let now = Instant::now();
        let slot = cache.victim_slot(model, busy, now)?;
        cache.drop_victim(model, slot, now);
        Some(slot)
    }

    /// Moves the sequence in slot `from` to slot 0, where speculation runs, with its entry.
    /// No slot but `from` may run a request. Slot 0's entry first moves to a slot without a
    /// live entry, or to the slot of an entry used less recently, if there is one.
    pub fn move_to_slot0(&mut self, model: &mut Model, from: usize) -> anyhow::Result<()> {
        if from == 0 {
            return Ok(());
        }
        if let Some(cache) = self.prefix.as_ref() {
            cache.fence()?;
        }
        self.vacate_slot0(model, &[from])?;
        let now = Instant::now();
        let entry = self
            .prefix
            .as_ref()
            .and_then(|c| c.of_slot(model, from, now));
        model.transfer_slot(from, 0)?;
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
        cache.fence()?;
        let mut busy = busy.to_vec();
        busy.push(0);
        let Some(to) = cache.victim_slot(model, &busy, now) else {
            cache.note_drop(
                i,
                now,
                "slot 0 takes another sequence, and no other slot is free",
            );
            return Ok(());
        };
        if let Some(j) = cache.of_slot(model, to, now)
            && cache.entries[j].used >= cache.entries[i].used
        {
            cache.note_drop(
                i,
                now,
                "slot 0 takes another sequence, and every other slot holds a more recent prompt",
            );
            return Ok(());
        }
        cache.drop_victim(model, to, now);
        model.transfer_slot(0, to)?;
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
        let now = Instant::now();
        let old = &cache.entries[i];
        // A conversation that grows replaces its own entry: nothing is lost.
        if !old.tokens.is_empty() && !tokens.starts_with(&old.tokens) {
            let same_sequence = old
                .slot
                .is_some_and(|s| model.slot_sequence(s).is_ok_and(|q| q == old.sequence));
            let expired = same_sequence && now.saturating_duration_since(old.used) > cache.ttl;
            if cache.live(old, model, now) {
                cache.note_drop(
                    i,
                    now,
                    "every entry holds a cached prompt, and it was the least recently used",
                );
            } else if expired {
                let ttl = cache.ttl.as_secs_f64();
                cache.note_drop(
                    i,
                    now,
                    &format!("expired, unused past the {ttl:.0} s time to live"),
                );
            }
        }
        // Nothing is kept until the copies are whole, in case this fails midway.
        cache.entries[i].slot = None;
        cache.drop_slot(slot);
        let e = &mut cache.entries[i];
        model.save_local_state(slot, &mut e.model)?;
        e.tokens.clear();
        e.tokens.extend_from_slice(tokens);
        e.hash = prefix_hash(tokens);
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
        self.save_host(model, i, true);
        Ok(())
    }

    /// Copies entry `i`, just saved, to the host tier, with the draft's rings and global
    /// caches if `draft`. A failure only costs the host copy, so it goes to the log.
    ///
    /// After a rebuild step ([`Self::rebuild_step`]), the copies read the entry's slot and
    /// the draft's global caches, which the rebuild filled from the entry's tokens.
    fn save_host(&mut self, model: &Model, i: usize, draft: bool) {
        let Some(PrefixCache {
            entries,
            host: Some(host),
            ..
        }) = self.prefix.as_mut()
        else {
            return;
        };
        let e = &entries[i];
        if e.chunk != HOST_CHUNK_POSITIONS {
            return;
        }
        let saved = e.slot.context("a saved entry has a slot").and_then(|slot| {
            host.save(
                model,
                &self.draft,
                slot,
                &e.tokens,
                &e.model,
                draft.then_some(&e.draft),
            )
        });
        let line = match saved {
            Ok(Some(s)) if s.bytes == 0 => return,
            Ok(Some(s)) => format!(
                "host cache: saving {} tokens{}: {} MiB, {} chunks copied, {} shared; {} of {} \
                 chunks in use by {} entries",
                e.tokens.len(),
                if draft { " with the draft's state" } else { "" },
                s.bytes >> 20,
                s.copied,
                s.shared,
                host.used_chunks(),
                host.chunks().0,
                host.entries()
            ),
            Ok(None) => format!(
                "host cache: {} tokens do not fit the host cache",
                e.tokens.len()
            ),
            Err(err) => format!(
                "host cache: saving {} tokens failed: {err:#}",
                e.tokens.len()
            ),
        };
        host.note(line);
    }

    /// Whether the host tier is on, which [`Self::park`] needs.
    #[must_use]
    pub fn can_park(&self) -> bool {
        self.prefix.as_ref().is_some_and(|c| c.host.is_some())
    }

    /// Copies the sequence in slot `slot` to the host tier ([`HostCache::park`]) and frees
    /// the slot: drops its entry and resets it. `key` is the prompt's key
    /// ([`Prompt::key`]) and the tokens fed after it, of which the first `prompt_len` came
    /// from the prompt. The slot keeps its pages, which a returning sequence of about its
    /// length maps no more, until the pool needs them ([`Self::release_slot`]). Returns the
    /// id that [`Self::unpark`] takes, or `None`, leaving the slot as it was, when the host
    /// tier is off or the sequence does not fit it.
    pub fn park(
        &mut self,
        model: &mut Model,
        slot: usize,
        key: &[u32],
        prompt_len: usize,
    ) -> anyhow::Result<Option<usize>> {
        self.invalidate_batch_draft();
        let draft_rings = if self.draft.is_dflash2() && self.draft.slot_committed(slot) == key.len()
        {
            Some((self.draft.slot_ring_spans(model.stream(), slot)?, key.len()))
        } else {
            None
        };
        let Some(host) = self.prefix.as_mut().and_then(|c| c.host.as_mut()) else {
            return Ok(None);
        };
        let started = Instant::now();
        let Some((id, saved)) = host.park(
            model,
            slot,
            key,
            prompt_len,
            draft_rings.as_ref().map(|(s, c)| (s.as_slice(), *c)),
        )?
        else {
            host.note(format!(
                "host cache: a sequence of {} tokens does not fit to park",
                key.len()
            ));
            return Ok(None);
        };
        let draft_note = if draft_rings.is_some() {
            " with draft rings"
        } else {
            ""
        };
        host.note(format!(
            "host cache: parked {} tokens from slot {slot}{draft_note}: {} MiB, {} chunks \
             copied, {} shared, {:.1} ms to queue",
            key.len(),
            saved.bytes >> 20,
            saved.copied,
            saved.shared,
            started.elapsed().as_secs_f64() * 1e3
        ));
        if let Some(cache) = self.prefix.as_mut() {
            cache.fence()?;
            cache.drop_slot(slot);
        }
        model.reset_slot(slot)?;
        if self.draft.is_dflash2() {
            self.draft.reset_slot(slot);
        }
        Ok(Some(id))
    }

    /// Returns parked sequence `id` ([`Self::park`]) into a slot outside `busy`, which
    /// loses its sequence and entry, and returns the slot and the sequence's position.
    pub fn unpark(
        &mut self,
        model: &mut Model,
        id: usize,
        busy: &[usize],
    ) -> anyhow::Result<(usize, usize)> {
        self.invalidate_batch_draft();
        let now = Instant::now();
        let cache = self
            .prefix
            .as_mut()
            .context("parking needs the prefix cache")?;
        cache.fence()?;
        let slot = cache
            .return_slot(model, busy, now)
            .context("every slot runs a request")?;
        cache.drop_slot(slot);
        model.reset_slot(slot)?;
        if self.draft.is_dflash2() {
            self.draft.reset_slot(slot);
        }
        let host = cache
            .host
            .as_mut()
            .context("parking needs the host cache")?;
        let started = host.start_restore()?;
        let (position, bytes, draft_restored) = host.unpark(model, &mut self.draft, id, slot)?;
        let seconds = host.finish_restore(started)?;
        let draft_note = if draft_restored {
            " with draft rings"
        } else {
            ""
        };
        host.note(format!(
            "host cache: returned {position} parked tokens into slot {slot}{draft_note}: {} MiB \
             in {:.1} ms, {:.1} GB/s",
            bytes >> 20,
            seconds * 1e3,
            bytes as f64 / seconds.max(1e-9) / 1e9
        ));
        Ok((slot, position))
    }

    /// The slot outside `busy` that [`Self::unpark`] would return a sequence to.
    #[must_use]
    pub fn return_slot(&self, model: &Model, busy: &[usize]) -> Option<usize> {
        let now = Instant::now();
        self.prefix
            .as_ref()
            .and_then(|c| c.return_slot(model, busy, now))
    }

    /// Frees parked sequence `id`'s host memory, for a request that ended while parked.
    pub fn drop_parked(&mut self, id: usize) {
        if let Some(host) = self.prefix.as_mut().and_then(|c| c.host.as_mut()) {
            host.drop_parked(id);
        }
    }

    /// Drops slot `slot`'s entry and returns all of its pages to the pool, once the host
    /// tier's copies from it are done, for a caller that needs `need` free pages.
    pub fn evict_slot(
        &mut self,
        model: &mut Model,
        slot: usize,
        need: usize,
    ) -> anyhow::Result<()> {
        if let Some(cache) = self.prefix.as_mut() {
            cache.fence()?;
            let now = Instant::now();
            if let Some(i) = cache.of_slot(model, slot, now) {
                let pages = model.slot_pages(slot).unwrap_or(0) + cache.entries[i].model.pages();
                let why = format!("the KV pool needs {need} free pages, and it held {pages}");
                cache.note_drop(i, now, &why);
            }
            cache.drop_slot(slot);
            cache.release_dead(model)?;
        }
        model.unmap_slot_from(slot, 0)?;
        model.reset_slot(slot)
    }

    /// Returns the pages of the copies of entries that are no longer live to the pool.
    /// Returns the pages freed.
    pub fn release_dead_entries(&mut self, model: &Model) -> anyhow::Result<usize> {
        self.prefix
            .as_mut()
            .map_or(Ok(0), |c| c.release_dead(model))
    }

    /// Returns the pages of idle slot `slot` that its live entry does not need to the pool:
    /// past the entry's boundary, and its rings, or all of them without a live entry, once
    /// the host tier's copies are done. The entry keeps the pages below its boundary until
    /// [`Self::evict_slot`]. A slot keeps its pages after its request ends, so that the
    /// next sequence there maps none; the server calls this when the pool runs short.
    pub fn release_slot(&mut self, model: &mut Model, slot: usize) -> anyhow::Result<()> {
        let now = Instant::now();
        let keep = self.prefix.as_ref().map_or(Ok(None), |c| {
            c.fence().map(|()| {
                c.of_slot(model, slot, now)
                    .map(|i| c.entries[i].tokens.len())
            })
        })?;
        if let Some(boundary) = keep {
            return model.unmap_slot_from(slot, boundary);
        }
        if let Some(cache) = self.prefix.as_mut() {
            cache.drop_slot(slot);
        }
        model.unmap_slot_from(slot, 0)?;
        model.reset_slot(slot)
    }

    /// The slots outside `busy` that hold pages, least recently used entry first (a slot
    /// without a live entry before any), with the pages each holds: what
    /// [`Self::evict_slot`] can return to the pool.
    #[must_use]
    pub fn evictable_slots(&self, model: &Model, busy: &[usize]) -> Vec<(usize, usize)> {
        let now = Instant::now();
        let mut slots: Vec<(Option<Instant>, usize, usize)> = (0..model.slots())
            .filter(|s| !busy.contains(s))
            .filter_map(|s| {
                let entry = self
                    .prefix
                    .as_ref()
                    .and_then(|c| c.of_slot(model, s, now).map(|i| c.entries[i].model.pages()));
                let pages = model.slot_pages(s).ok()? + entry.unwrap_or(0);
                (pages > 0).then(|| {
                    let used = self
                        .prefix
                        .as_ref()
                        .and_then(|c| c.of_slot(model, s, now).map(|i| c.entries[i].used));
                    (used, s, pages)
                })
            })
            .collect();
        slots.sort_by_key(|&(used, s, _)| (used, s == 0));
        slots.into_iter().map(|(_, s, p)| (s, p)).collect()
    }

    /// Returns the draft's pages to the pool when it follows no sequence, so that no entry
    /// keeps its global caches any more. Returns the pages freed.
    pub fn trim_draft(&mut self, model: &Model) -> anyhow::Result<usize> {
        if self.following() {
            return Ok(0);
        }
        let held = self.draft.held_pages();
        if held == 0 {
            return Ok(0);
        }
        self.fence_host()?;
        self.draft.unmap_from(0, model.stream())?;
        // The draft's global caches no longer hold any sequence.
        self.draft_sequence += 1;
        Ok(held)
    }

    /// Pages more that the draft takes to commit every position below `end`.
    #[must_use]
    pub fn draft_pages_to(&self, end: usize) -> usize {
        self.draft.pages_to(end)
    }

    /// The boundary of the shortest live prefix cache entry of prefill chunks of `chunk`
    /// whose tokens `key` starts with and that goes past position `done`.
    pub(super) fn entry_after(
        &self,
        model: &Model,
        key: &[u32],
        done: usize,
        chunk: usize,
    ) -> Option<usize> {
        let cache = self.prefix.as_ref()?;
        let now = Instant::now();
        let max_len = cache
            .entries
            .iter()
            .filter(|e| e.chunk == chunk && e.tokens.len() > done && e.tokens.len() <= key.len())
            .map(|e| e.tokens.len())
            .max()
            .unwrap_or(0);
        if max_len == 0 || chunk == 0 {
            return None;
        }
        let hashes = PrefixHashes::new(key, chunk, max_len / chunk);
        let mut best: Option<usize> = None;
        for e in &cache.entries {
            let len = e.tokens.len();
            if len <= done
                || best.is_some_and(|b| len >= b)
                || e.chunk != chunk
                || len > key.len()
                || !cache.live(e, model, now)
            {
                continue;
            }
            if hashes.hash_at_len(len) == Some(e.hash) && key.starts_with(&e.tokens) {
                best = Some(len);
            }
        }
        best
    }

    /// The live entries of prefill chunks of [`Self::prefill_chunk`] whose tokens are
    /// `key` and that lack the draft's state.
    pub(super) fn entries_without_draft(&self, model: &Model, key: &[u32]) -> Vec<usize> {
        let Some(cache) = self.prefix.as_ref() else {
            return Vec::new();
        };
        let now = Instant::now();
        let key_hash = prefix_hash(key);
        (0..cache.entries.len())
            .filter(|&i| {
                let e = &cache.entries[i];
                e.tokens.len() == key.len()
                    && e.hash == key_hash
                    && e.chunk == self.prefill_chunk
                    && !cache.has_draft(e, self.draft_sequence)
                    && cache.live(e, model, now)
                    && e.tokens == key
            })
            .collect()
    }

    /// The slot and key of the live prefix cache entry with the most tokens, when it lacks
    /// the draft's state and holds text alone, ids below `vocab`: the entry that a rebuild
    /// on an idle server gives the draft's state ([`Self::start_rebuild`]).
    #[must_use]
    pub fn entry_to_rebuild(&self, model: &Model, vocab: usize) -> Option<(usize, Vec<u32>)> {
        let cache = self.prefix.as_ref()?;
        let now = Instant::now();
        let e = cache
            .entries
            .iter()
            .filter(|e| cache.live(e, model, now) && e.chunk == self.prefill_chunk)
            .max_by_key(|e| e.tokens.len())?;
        let text = e
            .tokens
            .iter()
            .all(|&t| usize::try_from(t).is_ok_and(|t| t < vocab));
        if !text || cache.has_draft(e, self.draft_sequence) {
            return None;
        }
        Some((e.slot?, e.tokens.clone()))
    }

    /// A slot outside `busy` without a live entry, for a rebuild that may not drop one.
    #[must_use]
    pub fn empty_slot(&self, model: &Model, busy: &[usize]) -> Option<usize> {
        let now = Instant::now();
        (0..model.slots()).rev().find(|s| {
            !busy.contains(s)
                && self
                    .prefix
                    .as_ref()
                    .is_none_or(|c| c.of_slot(model, *s, now).is_none())
        })
    }
}

/// Restores both caches for a request prefilling and decoding alone in slot 0.
fn restore_solo(
    host: &mut HostCache,
    model: &mut Model,
    draft: &mut Draft,
    entry: usize,
    tokens: usize,
) -> anyhow::Result<f64> {
    let started = host.start_restore()?;
    let bytes = host.restore_slot(model, entry, 0)? + host.restore_draft(draft, entry)?;
    let seconds = host.finish_restore(started)?;
    host.note(restored(
        tokens,
        0,
        bytes,
        seconds,
        "with the draft's state",
    ));
    Ok(seconds)
}

/// A log line for a restore from the host tier.
fn restored(tokens: usize, slot: usize, bytes: usize, seconds: f64, what: &str) -> String {
    format!(
        "host cache: restored {tokens} tokens into slot {slot} {what}: {} MiB in {:.1} ms, \
         {:.1} GB/s",
        bytes >> 20,
        seconds * 1e3,
        bytes as f64 / seconds.max(1e-9) / 1e9
    )
}

/// A log line for promoting an entry from the L3 disk tier into pinned host memory.
fn promoted(tokens: usize, bytes: u64, seconds: f64) -> String {
    format!(
        "disk cache: promoted {tokens} tokens into host memory: {} MiB in {:.1} ms, {:.1} GB/s",
        bytes >> 20,
        seconds * 1e3,
        bytes as f64 / seconds.max(1e-9) / 1e9
    )
}
