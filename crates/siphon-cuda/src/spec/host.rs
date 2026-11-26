//! The prefix cache's host tier: whole prompt states in page-locked host memory, so that a
//! conversation whose slot another sequence took returns over PCIe instead of prefilling
//! again.
//!
//! A device entry of the prefix cache ([`super::prefix`]) keeps a slot's rings and conv
//! states and leaves the global KV caches in the slot, so it dies when the slot starts
//! another sequence. A host entry copies all of it: the global caches below the entry's
//! boundary, the rings and conv states, and the draft's state when the device entry has it.
//!
//! The memory is one pool of fixed-size chunks, page-locked once at startup. A chunk holds
//! [`HOST_CHUNK_POSITIONS`] positions of every global cache of the target, then the same
//! positions of the draft's global caches. The prefix cache saves at multiples of the
//! prefill chunk, so an entry's global caches fill whole chunks. An entry's rings and conv
//! states, then the draft's rings, take a run of chunks of their own.
//!
//! Chunk `j` of a prompt's global caches depends only on the prompt's tokens below
//! `(j + 1) * HOST_CHUNK_POSITIONS`: prefill runs the same chunks from position 0 whatever
//! follows, and the cache's restores keep it bit-identical. So two entries whose tokens
//! agree that far share chunk `j`, counted by references. A conversation that grows shares
//! every chunk of its last entry and copies only the new ones; the old entry, a prefix of
//! the new, is dropped. Agents that start with the same system prompt share its chunks.
//!
//! Copies to the host run on a stream of their own, beside serving. Every save first waits
//! for the model's stream, so it copies what the model has written, and records an event
//! that the model's stream waits for ([`HostCache::fence`]) before anything that could
//! overwrite what the copies read: a slot's reset or a move into it, the device entries'
//! copies, the draft's global caches. Restores run on the model's stream.
//!
//! When the pool is full, whole entries go, least recently used first. [`disk`] writes the
//! entries to files and reads them back, for a restart; a disk tier could keep entries in
//! the same files, since chunks are opaque numbers and an entry needs only its chunks back.

mod disk;

use std::{ffi::c_void, sync::Arc, time::Instant};

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaEvent, CudaStream, PinnedHostSlice, sys};

use crate::{
    draft::{Draft, DraftRings},
    model::{DEFAULT_PREFILL_CHUNK, DeviceSpan, GlobalKv, LocalState, Model},
};

/// Positions of the global caches per chunk: the prefill chunk, at whose multiples the
/// prefix cache saves.
pub const HOST_CHUNK_POSITIONS: usize = DEFAULT_PREFILL_CHUNK;

pub use disk::{DISK_FORMAT, Loaded as DiskLoaded, Saved as DiskSaved};

/// Most bytes one page-locked allocation takes. Chunks never straddle two.
const SLAB_BYTES: usize = 1 << 30;

/// Chunks of page-locked host memory, allocated once, with a reference count each.
struct ChunkPool {
    /// Owns the memory.
    _slabs: Vec<PinnedHostSlice<u8>>,
    /// Each chunk's host address.
    addresses: Vec<usize>,
    chunk_bytes: usize,
    refs: Vec<u32>,
    free: Vec<u32>,
}

impl ChunkPool {
    fn new(stream: &Arc<CudaStream>, chunk_bytes: usize, chunks: usize) -> anyhow::Result<Self> {
        let ctx = stream.context();
        let per_slab = (SLAB_BYTES / chunk_bytes).max(1);
        let mut slabs = Vec::new();
        let mut addresses = Vec::with_capacity(chunks);
        while addresses.len() < chunks {
            let n = (chunks - addresses.len()).min(per_slab);
            // SAFETY: the copies write a chunk before any copy reads it, and nothing reads
            // the memory on the host.
            let mut slab = unsafe { ctx.alloc_pinned_with_flags::<u8>(n * chunk_bytes, 0) }
                .with_context(|| {
                    format!(
                        "page-locking {} MiB of host memory after {} MiB",
                        (n * chunk_bytes) >> 20,
                        (addresses.len() * chunk_bytes) >> 20
                    )
                })?;
            let base = slab.as_mut_ptr()? as usize;
            addresses.extend((0..n).map(|i| base + i * chunk_bytes));
            slabs.push(slab);
        }
        let count = u32::try_from(chunks).context("too many host cache chunks")?;
        Ok(Self {
            _slabs: slabs,
            addresses,
            chunk_bytes,
            refs: vec![0; chunks],
            free: (0..count).rev().collect(),
        })
    }

    fn take(&mut self) -> Option<u32> {
        let c = self.free.pop()?;
        self.refs[c as usize] = 1;
        Some(c)
    }

    fn retain(&mut self, c: u32) {
        self.refs[c as usize] += 1;
    }

    fn release(&mut self, c: u32) {
        let refs = &mut self.refs[c as usize];
        *refs -= 1;
        if *refs == 0 {
            self.free.push(c);
        }
    }

    fn address(&self, c: u32) -> usize {
        self.addresses[c as usize]
    }

    /// Copies `spans` one after another between the device and the bytes of `chunks` read
    /// as one run, from byte `offset` of the run on, on `stream`.
    fn copy_run(
        &self,
        stream: &CudaStream,
        chunks: &[u32],
        offset: usize,
        spans: &[DeviceSpan],
        direction: Direction,
    ) -> anyhow::Result<()> {
        let total: usize = spans.iter().map(|s| s.bytes).sum();
        ensure!(
            offset + total <= chunks.len() * self.chunk_bytes,
            "{} bytes do not fit {} chunks from byte {offset}",
            total,
            chunks.len()
        );
        let mut at = offset;
        for s in spans {
            let mut done = 0;
            while done < s.bytes {
                let (c, within) = (at / self.chunk_bytes, at % self.chunk_bytes);
                let n = (s.bytes - done).min(self.chunk_bytes - within);
                let host = self.address(chunks[c]) + within;
                copy(stream, direction, host, s.ptr + done as u64, n)?;
                done += n;
                at += n;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    ToHost,
    ToDevice,
}

/// Copies `bytes` between host address `host` and device address `device` on `stream`.
fn copy(
    stream: &CudaStream,
    direction: Direction,
    host: usize,
    device: u64,
    bytes: usize,
) -> anyhow::Result<()> {
    // SAFETY: `host` lies in a chunk of the pool, which lives until both streams are done
    // (`HostCache::drop`), and `device` in a live device buffer, with `bytes` inside both
    // (`ChunkPool::copy_run` and the callers' layouts). The streams' order and the fence
    // keep the copies from racing the work that reads or writes either side.
    unsafe {
        match direction {
            Direction::ToHost => {
                sys::cuMemcpyDtoHAsync_v2(host as *mut c_void, device, bytes, stream.cu_stream())
            }
            Direction::ToDevice => {
                sys::cuMemcpyHtoDAsync_v2(device, host as *const c_void, bytes, stream.cu_stream())
            }
        }
    }
    .result()
    .context("copying between host and device memory")
}

/// One prompt's state in host memory.
struct HostEntry {
    /// The prompt's key ([`super::Prompt::key`]) below its boundary, a multiple of
    /// [`HOST_CHUNK_POSITIONS`]: its tokens, with image placeholders replaced by ids of the
    /// images' hashes.
    tokens: Vec<u32>,
    /// The chunks of the global caches, by position.
    kv: Vec<u32>,
    /// The chunks of the rings and conv states, then the draft's rings.
    local: Vec<u32>,
    /// Whether `local` holds the draft's rings.
    rings: bool,
    /// When it was last saved or restored, in [`HostCache::tick`]s.
    used: u64,
}

/// What [`HostCache::save`] copied.
#[derive(Debug)]
pub(super) struct Saved {
    /// Global cache chunks copied, and shared with an earlier entry.
    pub copied: usize,
    pub shared: usize,
    pub bytes: usize,
}

/// The host tier's entries and memory.
pub(super) struct HostCache {
    pool: ChunkPool,
    /// The model's stream, on which restores run.
    main: Arc<CudaStream>,
    /// The stream that copies to the host.
    copy: Arc<CudaStream>,
    /// Recorded on `copy` after the last save's copies.
    pending: Option<CudaEvent>,
    entries: Vec<HostEntry>,
    /// Per chunk, whether its draft part holds the draft's global caches.
    drafted: Vec<bool>,
    /// The target's global caches and the draft's, for their layouts.
    target: GlobalKv,
    draft: GlobalKv,
    /// Bytes of a slot's rings and conv states, and of the draft's rings.
    local_bytes: usize,
    rings_bytes: usize,
    tick: u64,
    /// Lines for the server's log, drained by [`Self::notes`].
    notes: Vec<String>,
}

impl HostCache {
    /// Page-locks `bytes` of host memory, or less to whole chunks, for entries of `model`
    /// and `draft`.
    pub fn new(model: &Model, draft: &Draft, bytes: usize) -> anyhow::Result<Self> {
        let main = Arc::clone(model.stream());
        let target = model.global_kv(0)?;
        let draft_kv = draft.global_kv(&main);
        let chunk_bytes =
            HOST_CHUNK_POSITIONS * (target.bytes_per_position() + draft_kv.bytes_per_position());
        let local_bytes: usize = model.local_spans(0)?.iter().map(|s| s.bytes).sum();
        let rings_bytes: usize = draft.ring_spans(&main).iter().map(|s| s.bytes).sum();
        let chunks = bytes / chunk_bytes;
        let local_chunks = (local_bytes + rings_bytes).div_ceil(chunk_bytes);
        ensure!(
            chunks > local_chunks,
            "a host cache of {} MiB holds {chunks} chunks of {} MiB; an entry takes at least {}",
            bytes >> 20,
            chunk_bytes >> 20,
            local_chunks + 1
        );
        let pool = ChunkPool::new(&main, chunk_bytes, chunks)?;
        let copy = main.context().new_stream()?;
        Ok(Self {
            pool,
            main,
            copy,
            pending: None,
            entries: Vec::new(),
            drafted: vec![false; chunks],
            target,
            draft: draft_kv,
            local_bytes,
            rings_bytes,
            tick: 0,
            notes: Vec::new(),
        })
    }

    /// Chunks in the pool, and the bytes of each.
    pub fn chunks(&self) -> (usize, usize) {
        (self.pool.addresses.len(), self.pool.chunk_bytes)
    }

    /// Makes the model's stream wait for the copies to the host issued so far, before it
    /// overwrites anything they read.
    pub fn fence(&self) -> anyhow::Result<()> {
        if let Some(event) = &self.pending {
            self.main.wait(event)?;
        }
        Ok(())
    }

    pub fn notes(&mut self) -> Vec<String> {
        std::mem::take(&mut self.notes)
    }

    fn local_chunks(&self) -> usize {
        (self.local_bytes + self.rings_bytes).div_ceil(self.pool.chunk_bytes)
    }

    /// Whether entry `e` holds the draft's state: its rings, and the draft's global caches
    /// in every chunk if the draft has global layers.
    fn complete(&self, e: &HostEntry) -> bool {
        e.rings && (self.draft.caches.is_empty() || e.kv.iter().all(|&c| self.drafted[c as usize]))
    }

    /// The longest entry that `prompt` starts with and goes past, with the draft's state
    /// if `draft`.
    pub fn lookup(&self, prompt: &[u32], draft: bool) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                prompt.len() > e.tokens.len()
                    && prompt.starts_with(&e.tokens)
                    && (!draft || self.complete(e))
            })
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, _)| i)
    }

    /// Tokens of entry `h`.
    pub fn len(&self, h: usize) -> usize {
        self.entries[h].tokens.len()
    }

    /// Whether entry `h` holds the draft's state ([`Self::restore_draft`]).
    pub fn has_draft(&self, h: usize) -> bool {
        self.complete(&self.entries[h])
    }

    /// Whether entry `h` holds the draft's rings ([`Self::restore_rings`]).
    pub fn has_rings(&self, h: usize) -> bool {
        self.entries[h].rings
    }

    fn release(&mut self, e: &HostEntry) {
        for &c in e.kv.iter().chain(&e.local) {
            self.pool.release(c);
        }
    }

    /// Copies the state of a prompt of `tokens` into an entry, from the global caches of
    /// slot `slot` below `tokens.len()`, the device entry's copies `local` of the slot's
    /// rings and conv states, and with `rings` the device entry's copies of the draft's
    /// rings and the draft's global caches. Copies only what no entry holds yet. Returns
    /// `None` when the entry does not fit the pool even without every other entry.
    pub fn save(
        &mut self,
        model: &Model,
        draft: &Draft,
        slot: usize,
        tokens: &[u32],
        local: &LocalState,
        rings: Option<&DraftRings>,
    ) -> anyhow::Result<Option<Saved>> {
        let p = HOST_CHUNK_POSITIONS;
        ensure!(
            !tokens.is_empty() && tokens.len().is_multiple_of(p),
            "{} tokens are not a positive multiple of {p}",
            tokens.len()
        );
        ensure!(
            model.slot_position(slot)? >= tokens.len() && local.position() == tokens.len(),
            "slot {slot} and its copies do not hold the {} tokens",
            tokens.len()
        );
        self.tick += 1;
        let n = tokens.len() / p;
        let target = model.global_kv(slot)?;
        let draft_kv = draft.global_kv(&self.main);
        // The copies read what the model's stream has written by now.
        let ready = self
            .main
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))?;
        self.copy.wait(&ready)?;

        let existing = self.entries.iter().position(|e| e.tokens == tokens);
        let (kv, local_chunks, shared) = if let Some(h) = existing {
            // The same prompt again: only the draft's state may be new.
            self.entries[h].used = self.tick;
            if rings.is_none() || self.complete(&self.entries[h]) {
                return Ok(Some(Saved {
                    copied: 0,
                    shared: n,
                    bytes: 0,
                }));
            }
            let e = &self.entries[h];
            (e.kv.clone(), e.local.clone(), n)
        } else {
            let Some(chunks) = self.allocate(tokens)? else {
                return Ok(None);
            };
            chunks
        };
        let copied = n - shared;
        let mut bytes = 0;
        for (j, &c) in kv.iter().enumerate().skip(n - copied) {
            for (cache, pb, before) in target.layout() {
                let at = (j * p) as u64 * pb as u64;
                let host = self.pool.address(c) + p * before;
                copy(&self.copy, Direction::ToHost, host, cache + at, p * pb)?;
                bytes += p * pb;
            }
            self.drafted[c as usize] = false;
        }
        if existing.is_none() {
            self.pool.copy_run(
                &self.copy,
                &local_chunks,
                0,
                &local.spans(&self.main),
                Direction::ToHost,
            )?;
            bytes += self.local_bytes;
        }
        if let Some(rings) = rings {
            bytes += self.save_draft(&kv, &local_chunks, &draft_kv, rings)?;
        }
        if let Some(h) = existing {
            self.entries[h].rings = true;
        } else {
            self.entries.push(HostEntry {
                tokens: tokens.to_vec(),
                kv,
                local: local_chunks,
                rings: rings.is_some(),
                used: self.tick,
            });
        }
        self.pending = Some(
            self.copy
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))?,
        );
        Ok(Some(Saved {
            copied,
            shared,
            bytes,
        }))
    }

    /// Copies the draft's global caches into the chunks `kv` that lack them, and the draft's
    /// rings `rings` into `local`. Returns the bytes copied.
    fn save_draft(
        &mut self,
        kv: &[u32],
        local: &[u32],
        draft_kv: &GlobalKv,
        rings: &DraftRings,
    ) -> anyhow::Result<usize> {
        let p = HOST_CHUNK_POSITIONS;
        let base = p * self.target.bytes_per_position();
        let mut bytes = 0;
        for (j, &c) in kv.iter().enumerate() {
            if self.drafted[c as usize] {
                continue;
            }
            for (cache, dpb, before) in draft_kv.layout() {
                let at = (j * p) as u64 * dpb as u64;
                let host = self.pool.address(c) + base + p * before;
                copy(&self.copy, Direction::ToHost, host, cache + at, p * dpb)?;
                bytes += p * dpb;
            }
            self.drafted[c as usize] = true;
        }
        self.pool.copy_run(
            &self.copy,
            local,
            self.local_bytes,
            &rings.spans(&self.main),
            Direction::ToHost,
        )?;
        Ok(bytes + self.rings_bytes)
    }

    /// Chunks for a new entry of `tokens`: the global cache chunks it shares with the entry
    /// that agrees with it longest, new ones for the rest, and new ones for its rings and
    /// conv states, and how many chunks it shares. Drops the entries that `tokens` extends,
    /// then the least recently used, until the new chunks fit. Returns `None`, and keeps
    /// every entry it has not dropped, when they do not fit at all.
    #[expect(clippy::type_complexity, reason = "the chunks and their count")]
    fn allocate(&mut self, tokens: &[u32]) -> anyhow::Result<Option<(Vec<u32>, Vec<u32>, usize)>> {
        let p = HOST_CHUNK_POSITIONS;
        let n = tokens.len() / p;
        let longest = self
            .entries
            .iter()
            .map(|e| {
                let same = e
                    .tokens
                    .iter()
                    .zip(tokens)
                    .take_while(|(a, b)| a == b)
                    .count();
                (same / p).min(e.kv.len())
            })
            .enumerate()
            .max_by_key(|&(_, k)| k);
        let mut kv = match longest {
            Some((i, k)) if k > 0 => self.entries[i].kv[..k].to_vec(),
            _ => Vec::new(),
        };
        let shared = kv.len();
        for &c in &kv {
            self.pool.retain(c);
        }
        // An entry whose tokens start the new one's holds nothing the new one lacks but
        // its rings, conv states and draft rings.
        let mut i = 0;
        while i < self.entries.len() {
            if tokens.starts_with(&self.entries[i].tokens) {
                let e = self.entries.swap_remove(i);
                self.release(&e);
            } else {
                i += 1;
            }
        }
        let need = n - kv.len() + self.local_chunks();
        while self.pool.free.len() < need {
            let Some(lru) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i)
            else {
                for &c in &kv {
                    self.pool.release(c);
                }
                return Ok(None);
            };
            let e = self.entries.swap_remove(lru);
            self.release(&e);
        }
        while kv.len() < n {
            kv.push(self.pool.take().context("a free chunk was counted")?);
        }
        let local = (0..self.local_chunks())
            .map(|_| self.pool.take().context("a free chunk was counted"))
            .collect::<anyhow::Result<_>>()?;
        Ok(Some((kv, local, shared)))
    }

    /// Copies entry `h`'s global caches, rings and conv states into slot `slot`, just
    /// reset, and moves the slot to the entry's end. Returns the bytes copied.
    pub fn restore_slot(
        &mut self,
        model: &mut Model,
        h: usize,
        slot: usize,
    ) -> anyhow::Result<usize> {
        self.fence()?;
        self.tick += 1;
        let p = HOST_CHUNK_POSITIONS;
        let target = model.global_kv(slot)?;
        let e = &self.entries[h];
        ensure!(
            target.caches.len() == self.target.caches.len()
                && target.position_bytes == self.target.position_bytes,
            "the slot's global caches differ from the host cache's"
        );
        for (j, &c) in e.kv.iter().enumerate() {
            for (cache, pb, before) in target.layout() {
                let at = (j * p) as u64 * pb as u64;
                let host = self.pool.address(c) + p * before;
                copy(&self.main, Direction::ToDevice, host, cache + at, p * pb)?;
            }
        }
        let spans = model.local_spans(slot)?;
        self.pool
            .copy_run(&self.main, &e.local, 0, &spans, Direction::ToDevice)?;
        let len = e.tokens.len();
        self.entries[h].used = self.tick;
        model.set_loaded_position(slot, len)?;
        Ok(len * target.bytes_per_position() + self.local_bytes)
    }

    /// Copies entry `h`'s draft state into the draft: its rings, and its global caches if
    /// it has global layers. The draft then takes the entry's positions as committed.
    /// Returns the bytes copied.
    pub fn restore_draft(&mut self, draft: &mut Draft, h: usize) -> anyhow::Result<usize> {
        ensure!(self.has_draft(h), "host entry {h} lacks the draft's state");
        self.fence()?;
        let p = HOST_CHUNK_POSITIONS;
        let e = &self.entries[h];
        let draft_kv = draft.global_kv(&self.main);
        let base = p * self.target.bytes_per_position();
        for (j, &c) in e.kv.iter().enumerate() {
            for (cache, dpb, before) in draft_kv.layout() {
                let at = (j * p) as u64 * dpb as u64;
                let host = self.pool.address(c) + base + p * before;
                copy(&self.main, Direction::ToDevice, host, cache + at, p * dpb)?;
            }
        }
        let spans = draft.ring_spans(&self.main);
        self.pool.copy_run(
            &self.main,
            &e.local,
            self.local_bytes,
            &spans,
            Direction::ToDevice,
        )?;
        let len = e.tokens.len();
        draft.set_loaded(len)?;
        Ok(len * draft_kv.bytes_per_position() + self.rings_bytes)
    }

    /// Copies entry `h`'s draft rings into `rings`, a device entry's copies.
    pub fn restore_rings(&self, h: usize, rings: &mut DraftRings) -> anyhow::Result<()> {
        let e = &self.entries[h];
        ensure!(e.rings, "host entry {h} lacks the draft's rings");
        self.fence()?;
        let spans = rings.spans(&self.main);
        self.pool.copy_run(
            &self.main,
            &e.local,
            self.local_bytes,
            &spans,
            Direction::ToDevice,
        )?;
        rings.set_committed(e.tokens.len());
        Ok(())
    }

    /// Adds a line for the server's log.
    pub fn note(&mut self, line: String) {
        self.notes.push(line);
    }

    /// Chunks in use.
    pub fn used_chunks(&self) -> usize {
        self.pool.addresses.len() - self.pool.free.len()
    }

    /// Entries held.
    pub fn entries(&self) -> usize {
        self.entries.len()
    }

    /// Starts timing a restore: waits for the work queued on the model's stream, so that
    /// [`Self::finish_restore`] times the restore's copies alone.
    pub fn start_restore(&self) -> anyhow::Result<Instant> {
        self.main.synchronize()?;
        Ok(Instant::now())
    }

    /// The seconds since [`Self::start_restore`] once the model's stream has done its work.
    pub fn finish_restore(&self, started: Instant) -> anyhow::Result<f64> {
        self.main.synchronize()?;
        Ok(started.elapsed().as_secs_f64())
    }
}

impl Drop for HostCache {
    fn drop(&mut self) {
        // Copies may still read or write the pool's memory.
        let _ = self.copy.synchronize();
        let _ = self.main.synchronize();
    }
}
