//! Batched decode: one row for each of up to [`BATCH_ROWS`] sequences per step.
//!
//! The model keeps sequence slots. Slot 0 is the state that [`Model::step`],
//! [`Model::prefill`] and the verify pass use; [`Model::add_slots`] adds more, each with
//! its own KV caches and conv states of the same sizes. [`Model::step_batch`] runs the
//! next token of several slots as one pass through the verify pass's multi-row kernels,
//! with each row reading and writing its own slot's caches through a table of device
//! addresses (see `docs/batching.md`).
//!
//! Each row's results equal, bit for bit, a one-row verify of its sequence at its
//! position, whatever the other rows hold. A sequence decoded in batches therefore emits
//! the tokens of greedy decoding through one-row verifies, as greedy speculation does.
//!
//! Each batch size has its own CUDA graph, captured on first use. The tokens, the
//! positions and the state tables live in device buffers, so a graph replays for any
//! assignment of slots to rows.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaEvent, CudaSlice, CudaStream, CudaView, DevicePtr as _,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
};
use half::bf16;

use super::{
    Accept, ImageSource, Model, PrefillLogits, Sampling, StepGraph, Verdict, VisitState,
    prefill::PrefillScratch, zeros,
};
use crate::{
    gpu::{PinnedBuf, sync_event},
    ops::{
        attention::{KvCache, KvData, RingSlab},
        sample::SampleBuffers,
        spec::{
            BATCH_STATE_WORDS, BatchRows, RESULT_LEN, STATE_ROWS, STREAM_DIMS, VERIFY_ROWS,
            VISIT_CHANNELS,
        },
    },
    vmm::PagePool,
};

/// Slots of each dummy verify buffer: enough for any verify width `1..=VERIFY_ROWS` at
/// position 0, without touching the KV pool.
const DUMMY_KV_SLOTS: usize = 16;
/// Max bytes per KV slot across all supported cache formats (BF16: 1024 * 2 bytes).
const DUMMY_KV_ROW_BYTES: usize = 1024 * 2;

/// Tiny scratch KV + conv buffers for one inactive padded sequence in a bucketed
/// [`Model::verify_batch`] launch. Consumes ~144 KiB and zero KV pool pages.
struct DummyVerifySlot {
    keys: CudaSlice<u8>,
    values: CudaSlice<u8>,
    conv: [CudaSlice<f32>; 4],
}

impl DummyVerifySlot {
    fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let kv_bytes = DUMMY_KV_SLOTS * DUMMY_KV_ROW_BYTES;
        Ok(Self {
            keys: stream.alloc_zeros(kv_bytes)?,
            values: stream.alloc_zeros(kv_bytes)?,
            conv: [
                stream.alloc_zeros(STATE_ROWS * STREAM_DIMS[0])?,
                stream.alloc_zeros(STATE_ROWS * STREAM_DIMS[1])?,
                stream.alloc_zeros(STATE_ROWS * STREAM_DIMS[2])?,
                stream.alloc_zeros(STATE_ROWS * STREAM_DIMS[3])?,
            ],
        })
    }

    fn state_words(&self, stream: &CudaStream) -> [u64; BATCH_STATE_WORDS] {
        [
            self.keys.device_ptr(stream).0,
            self.values.device_ptr(stream).0,
            self.conv[0].device_ptr(stream).0,
            self.conv[1].device_ptr(stream).0,
            self.conv[2].device_ptr(stream).0,
            self.conv[3].device_ptr(stream).0,
        ]
    }
}

/// Most sequences one batched step runs.
pub use crate::ops::spec::{BATCH_ROWS, BATCH_VERIFY_MAX_ROWS};

/// Where the tokens of a batched verify come from.
pub enum BatchVerifyTokens<'a> {
    /// Host tokens for each sequence, all of the same width `1..=VERIFY_ROWS`.
    Host(&'a [&'a [u32]]),
    /// Device tokens in sequence-major order `[seqs, width]` (each sequence's anchor
    /// followed by `width - 1` drafts).
    Device {
        tokens: CudaView<'a, i32>,
        width: usize,
    },
}

/// A batched verify pass that [`Model::start_verify_batch`] launched and whose verdicts
/// [`Model::finish_verify_batch`] has not read yet.
#[must_use = "the slots' positions advance only in finish_verify_batch"]
#[derive(Debug)]
pub struct PendingBatchVerify {
    slots: Vec<(usize, usize)>,
    width: usize,
}

/// Per-sequence backup, rollback and acceptance buffers for [`Model::verify_batch`].
pub(super) struct BatchVerifyState {
    /// Indices of the sliding visits in `visits`.
    pub(super) sliding_visits: CudaSlice<i32>,
    /// Slots of every sliding ring.
    pub(super) ring_slots: usize,
    /// Bytes per slot of each sliding ring's key or value buffer.
    pub(super) ring_row_bytes: usize,
    /// Recorded conv inputs, BF16 `[visits, max_seqs, VERIFY_ROWS, VISIT_CHANNELS]`.
    pub(super) inputs: CudaSlice<bf16>,
    /// Sliding ring slots overwritten by the pass, `[sliding, max_seqs, VERIFY_ROWS, 2, row_bytes]`.
    pub(super) ring_backup: CudaSlice<u8>,
    /// Per-sequence rows to keep (0 for greedy, else exact count), `[max_seqs]`.
    pub(super) keep: CudaSlice<u32>,
    /// Per-sequence `[accepted, tokens, argmax]`, `[max_seqs * RESULT_LEN]`.
    pub(super) results: CudaSlice<i32>,
    /// Next anchor per sequence, `[max_seqs]`.
    pub(super) anchors: CudaSlice<i32>,
    /// Persistent page-locked host staging buffers for `keep` (H2D) and `results` (D2H).
    pub(super) host_keep: PinnedBuf<u32>,
    pub(super) host_results: PinnedBuf<i32>,
    /// Rows per sequence of the active or last pass (`1..=VERIFY_ROWS`).
    pub(super) width: usize,
    /// Sequences of the active or last pass (unpadded real sequence count).
    pub(super) last_seqs: usize,
    /// Padded sequence bucket count of the active or last pass (`>= last_seqs`).
    pub(super) padded_seqs: usize,
    /// Whether a pass was launched and its verdicts not read yet.
    pub(super) pending: bool,
    /// Scratch slots backing inactive padded sequences when `seqs < padded_seqs`.
    dummy_slots: Vec<DummyVerifySlot>,
    /// Fixed-address 16-sparse DFlash 2 draft candidates and normalized probabilities
    /// (`[max_seqs * 7 * 16]` each) for batched `--spec-beta` rejection sampling.
    pub(super) draft_cand: CudaSlice<i32>,
    pub(super) draft_q: CudaSlice<f32>,
}

impl BatchVerifyState {
    fn ensure_dummies(&mut self, stream: &Arc<CudaStream>, needed: usize) -> anyhow::Result<()> {
        while self.dummy_slots.len() < needed {
            self.dummy_slots.push(DummyVerifySlot::new(stream)?);
        }
        Ok(())
    }

    fn new(
        stream: &Arc<CudaStream>,
        visits: &[VisitState],
        max_seqs: usize,
    ) -> anyhow::Result<Self> {
        let mut sliding_visits = Vec::new();
        let mut ring_slots = None;
        let mut ring_row_bytes = None;
        for (i, v) in visits.iter().enumerate() {
            for (state, dims) in v.conv.iter().zip(STREAM_DIMS) {
                ensure!(
                    state.len() == STATE_ROWS * dims,
                    "visit {i}: a conv state has {} elements, expected {STATE_ROWS} x {dims}",
                    state.len()
                );
            }
            if v.kv.window().is_some() {
                let row_bytes = match v.kv.data() {
                    KvData::Bf16 { .. } => 1024 * 2,
                    KvData::Fp8 { .. } => 1024,
                    KvData::Nvfp4 { .. } => crate::ops::attention::FP4_ROW,
                    KvData::Nvfp4G32 { .. } => crate::ops::attention::FP4_G32_ROW,
                };
                sliding_visits.push(i32::try_from(i)?);
                ensure!(
                    *ring_slots.get_or_insert(v.kv.slots()) == v.kv.slots(),
                    "the sliding rings differ in size"
                );
                ensure!(
                    *ring_row_bytes.get_or_insert(row_bytes) == row_bytes,
                    "the sliding rings differ in format"
                );
            }
        }
        let sliding = sliding_visits.len();
        ensure!(sliding > 0, "no sliding visits");
        let ring_row_bytes = ring_row_bytes.unwrap_or(1024 * 2);
        Ok(Self {
            sliding_visits: stream.clone_htod(&sliding_visits)?,
            ring_slots: ring_slots.unwrap_or(0),
            ring_row_bytes,
            inputs: zeros(
                stream,
                visits.len() * max_seqs * VERIFY_ROWS * VISIT_CHANNELS,
            )?,
            ring_backup: zeros(
                stream,
                sliding * max_seqs * VERIFY_ROWS * 2 * ring_row_bytes,
            )?,
            keep: zeros(stream, max_seqs)?,
            results: zeros(stream, max_seqs * RESULT_LEN)?,
            anchors: zeros(stream, max_seqs)?,
            host_keep: PinnedBuf::zeros(stream, max_seqs)?,
            host_results: PinnedBuf::zeros(stream, max_seqs * RESULT_LEN)?,
            width: 1,
            last_seqs: 0,
            padded_seqs: 0,
            pending: false,
            dummy_slots: Vec::new(),
            draft_cand: zeros(
                stream,
                max_seqs * crate::draft::STEPS * crate::ops::sample::BETA_CAND_K,
            )?,
            draft_q: zeros(
                stream,
                max_seqs * crate::draft::STEPS * crate::ops::sample::BETA_CAND_K,
            )?,
        })
    }
}

/// A sequence slot besides slot 0: its caches and position.
#[derive(Debug)]
pub(super) struct Slot {
    pub(super) visits: Vec<VisitState>,
    /// Contiguous VMM slab backing this slot's sliding rings when pooled.
    /// Declared after `visits` so `KvCache::drop` leaks its slices before the slab frees
    /// its virtual address range.
    pub(super) ring_slab: Option<RingSlab>,
    pub(super) position: usize,
    /// The slot's [`Model::slot_sequence`], swapped with slot 0's while the slot stands in
    /// for it ([`Model::with_slot`]).
    sequence: u64,
}

/// What a batched step or batched verify keeps besides a chunk's intermediates.
pub(super) struct BatchBuffers {
    /// Most sequences the tables hold, up to [`BATCH_ROWS`].
    pub(super) max_seqs: usize,
    /// Each visit's state table, `[visits, max_seqs, BATCH_STATE_WORDS]` device addresses
    /// (see [`BATCH_STATE_WORDS`]).
    pub(super) tables: CudaSlice<u64>,
    /// Each sequence's starting position, `[max_seqs]`.
    pub(super) positions: CudaSlice<u32>,
    /// Each row's token, greedy or drawn, `[max_total_rows]`.
    pub(super) argmax: CudaSlice<i32>,
    /// The slots whose addresses `tables` holds, by sequence index.
    pub(super) members: Vec<usize>,
    /// Persistent page-locked host staging buffers for async H2D uploads.
    host_tables: PinnedBuf<u64>,
    host_positions: PinnedBuf<u32>,
    host_tokens: PinnedBuf<i32>,
    /// Recorded on the stream after each async pinned H2D upload.
    h2d_done: CudaEvent,
    /// Total sequence rows (`real + padded`) currently populated in `tables`.
    pub(super) padded_seqs: usize,
    /// Present for [`Model::verify_batch`]; `None` for [`Model::step_batch`].
    pub(super) verify: Option<BatchVerifyState>,
}

impl Drop for BatchBuffers {
    fn drop(&mut self) {
        let _ = self.h2d_done.synchronize();
    }
}

impl BatchBuffers {
    fn new(
        stream: &Arc<CudaStream>,
        visits: usize,
        max_seqs: usize,
        max_rows: usize,
        verify: Option<BatchVerifyState>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            max_seqs,
            tables: zeros(stream, visits * max_seqs * BATCH_STATE_WORDS)?,
            positions: zeros(stream, max_seqs)?,
            argmax: zeros(stream, max_rows)?,
            members: Vec::new(),
            host_tables: PinnedBuf::zeros(stream, visits * max_seqs * BATCH_STATE_WORDS)?,
            host_positions: PinnedBuf::zeros(stream, max_seqs)?,
            host_tokens: PinnedBuf::zeros(stream, max_rows)?,
            h2d_done: sync_event(stream)?,
            padded_seqs: 0,
            verify,
        })
    }

    /// The sequences' states in visit `visit`, for a pass of `rows` total rows.
    pub(super) fn rows(&self, visit: usize, rows: usize) -> BatchRows<'_> {
        let rows_per_seq = self.verify.as_ref().map_or(1, |v| v.width);
        let seqs = rows / rows_per_seq;
        let at = visit * self.max_seqs * BATCH_STATE_WORDS;
        BatchRows {
            states: self.tables.slice(at..at + seqs * BATCH_STATE_WORDS),
            positions: self.positions.slice(..seqs),
            rows_per_seq,
        }
    }
}

/// Allocates a full set of visits and optional shared sliding-ring slab matching `template`.
fn empty_slot_visits(
    stream: &Arc<CudaStream>,
    template: &[VisitState],
    pool: Option<&PagePool>,
) -> anyhow::Result<(Vec<VisitState>, Option<RingSlab>)> {
    let sliding_count = template
        .iter()
        .filter(|v| v.kv.window().is_some() && v.kv.is_pooled())
        .count();
    let ring_slab = if let (Some(pool), Some(first)) = (
        pool,
        template
            .iter()
            .find(|v| v.kv.window().is_some() && v.kv.is_pooled()),
    ) && sliding_count > 0
    {
        Some(RingSlab::reserve(
            pool,
            sliding_count,
            first.kv.slots(),
            first.kv.format(),
        )?)
    } else {
        None
    };
    let mut ring_idx = 0usize;
    let visits = template
        .iter()
        .map(|v| {
            let format = v.kv.format();
            let v_pool = pool.filter(|_| v.kv.is_pooled());
            let kv = match (v.kv.window(), v_pool, ring_slab.as_ref()) {
                (Some(window), Some(_), Some(slab)) => {
                    let idx = ring_idx;
                    ring_idx += 1;
                    KvCache::sliding_in_slab(stream, window, v.kv.slots(), format, slab, idx)?
                }
                (Some(window), Some(pool), None) => {
                    KvCache::sliding_pooled_with_format(stream, window, v.kv.slots(), format, pool)?
                }
                (Some(window), None, _) => {
                    KvCache::sliding_ring_with_format(stream, window, v.kv.slots(), format)?
                }
                (None, Some(pool), _) => {
                    KvCache::global_pooled(stream, v.kv.slots(), format, pool)?
                }
                (None, None, _) => KvCache::global_with_format(stream, v.kv.slots(), format)?,
            };
            Ok(VisitState {
                kv,
                conv: [
                    stream.alloc_zeros(v.conv[0].len())?,
                    stream.alloc_zeros(v.conv[1].len())?,
                    stream.alloc_zeros(v.conv[2].len())?,
                    stream.alloc_zeros(v.conv[3].len())?,
                ],
                round: v.round,
            })
        })
        .collect::<anyhow::Result<_>>()?;
    Ok((visits, ring_slab))
}

/// Slot `slot`'s visit states and optional ring slab, which must exist.
fn slot_visits(m: &mut Model, slot: usize) -> (&mut Vec<VisitState>, Option<&mut RingSlab>) {
    if slot == 0 {
        (&mut m.visits, m.ring_slab.as_mut())
    } else {
        let s = &mut m.slots[slot - 1];
        (&mut s.visits, s.ring_slab.as_mut())
    }
}

/// A visit's line of a state table: its caches' and conv states' device addresses.
pub(super) fn state_words(stream: &CudaStream, v: &VisitState) -> [u64; BATCH_STATE_WORDS] {
    let (keys, values) = match v.kv.data() {
        KvData::Bf16 { keys, values } => (keys.device_ptr(stream).0, values.device_ptr(stream).0),
        KvData::Fp8 { keys, values }
        | KvData::Nvfp4 { keys, values }
        | KvData::Nvfp4G32 { keys, values } => {
            (keys.device_ptr(stream).0, values.device_ptr(stream).0)
        }
    };
    let [k, v_state, proj, moe] = &v.conv;
    [
        keys,
        values,
        k.device_ptr(stream).0,
        v_state.device_ptr(stream).0,
        proj.device_ptr(stream).0,
        moe.device_ptr(stream).0,
    ]
}

impl Model {
    /// Adds `count` sequence slots, each with KV caches and conv states of slot 0's sizes
    /// and formats, starting empty at position 0.
    pub fn add_slots(&mut self, count: usize) -> anyhow::Result<()> {
        let stream = Arc::clone(&self.ctx.stream);
        for _ in 0..count {
            let (visits, ring_slab) =
                empty_slot_visits(&stream, &self.visits, self.kv_pool.as_ref())?;
            let sequence = self.new_sequence();
            self.slots.push(Slot {
                visits,
                ring_slab,
                position: 0,
                sequence,
            });
        }
        Ok(())
    }

    /// Sequence slots, slot 0 included.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots.len() + 1
    }

    /// The sequence slot `slot` holds: a number that changes whenever the slot starts
    /// another sequence ([`Model::reset_slot`], [`Model::restore_state`], a
    /// [`Model::move_slot`] into it), and that no other slot's sequence has had. While it
    /// stays the same, the slot's global caches keep every position below its position.
    pub fn slot_sequence(&self, slot: usize) -> anyhow::Result<u64> {
        if slot == 0 {
            return Ok(self.sequence);
        }
        Ok(self.slot_state(slot)?.sequence)
    }

    /// Position of the next token of slot `slot`.
    pub fn slot_position(&self, slot: usize) -> anyhow::Result<usize> {
        if slot == 0 {
            return Ok(self.position);
        }
        Ok(self.slot_state(slot)?.position)
    }

    pub(super) fn slot_state(&self, slot: usize) -> anyhow::Result<&Slot> {
        slot.checked_sub(1)
            .and_then(|i| self.slots.get(i))
            .with_context(|| format!("no slot {slot}; the model has {}", self.slots()))
    }

    /// Runs `f` with slot `slot`'s caches and position in slot 0's place. Only code that
    /// launches its kernels directly may run there: the step and verify graphs hold slot
    /// 0's addresses.
    pub(crate) fn with_slot<T>(
        &mut self,
        slot: usize,
        f: impl FnOnce(&mut Self) -> anyhow::Result<T>,
    ) -> anyhow::Result<T> {
        if slot == 0 {
            return f(self);
        }
        self.slot_state(slot)?;
        let swap = |m: &mut Self| {
            let s = &mut m.slots[slot - 1];
            std::mem::swap(&mut m.visits, &mut s.visits);
            std::mem::swap(&mut m.ring_slab, &mut s.ring_slab);
            std::mem::swap(&mut m.position, &mut s.position);
            std::mem::swap(&mut m.sequence, &mut s.sequence);
        };
        swap(self);
        let out = f(self);
        swap(self);
        out
    }

    /// Moves the sequence in slot `from` to slot `to`: copies its KV caches (a global
    /// cache up to its position, a ring whole), its conv states and its position. The
    /// sequence in `to` is lost, and `from` keeps a copy until it is reset. A sequence that
    /// ran in a batch from any slot can so continue in slot 0, the slot that speculation
    /// and plain steps use, whose graphs hold slot 0's addresses.
    pub fn move_slot(&mut self, from: usize, to: usize) -> anyhow::Result<()> {
        self.slot_position(from)?;
        self.slot_position(to)?;
        if from == to {
            return Ok(());
        }
        let stream = Arc::clone(&self.ctx.stream);
        let position = self.slot_position(from)?;
        self.map_slot(to, position.max(1))?;
        let (src, src_slab, dst, dst_slab) = match (from, to) {
            (0, t) => {
                let d = &mut self.slots[t - 1];
                (
                    &self.visits,
                    self.ring_slab.as_ref(),
                    &mut d.visits,
                    d.ring_slab.as_mut(),
                )
            }
            (f, 0) => {
                let s = &self.slots[f - 1];
                (
                    &s.visits,
                    s.ring_slab.as_ref(),
                    &mut self.visits,
                    self.ring_slab.as_mut(),
                )
            }
            (f, t) if f < t => {
                let (low, high) = self.slots.split_at_mut(t - 1);
                let (s, d) = (&low[f - 1], &mut high[0]);
                (
                    &s.visits,
                    s.ring_slab.as_ref(),
                    &mut d.visits,
                    d.ring_slab.as_mut(),
                )
            }
            (f, t) => {
                let (low, high) = self.slots.split_at_mut(f - 1);
                let (s, d) = (&high[0], &mut low[t - 1]);
                (
                    &s.visits,
                    s.ring_slab.as_ref(),
                    &mut d.visits,
                    d.ring_slab.as_mut(),
                )
            }
        };
        let bulk_rings = match (src_slab, dst_slab) {
            (Some(s), Some(d)) => {
                if s.is_mapped() {
                    s.copy_to(d, &stream)?;
                }
                true
            }
            _ => false,
        };
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            // A ring whole, a global cache up to the position, but only what is mapped: an
            // idle slot may have returned its rings and the pages past its prefix cache
            // entry to the pool, and the driver faults on a copy from
            // unmapped addresses. What it returned holds nothing an entry needs.
            let slots = if s.kv.window().is_some() {
                if bulk_rings || s.kv.mapped() < s.kv.slots() {
                    0
                } else {
                    s.kv.slots()
                }
            } else {
                position.min(s.kv.mapped())
            };
            if slots == 0 {
                for (from, to) in s.conv.iter().zip(d.conv.iter_mut()) {
                    stream.memcpy_dtod(from, to)?;
                }
                continue;
            }
            let len = slots * s.kv.row_len();
            match (s.kv.data(), d.kv.data_mut()) {
                (KvData::Bf16 { keys, values }, KvData::Bf16 { keys: k, values: v }) => {
                    stream.memcpy_dtod(&keys.slice(..len), &mut k.slice_mut(..len))?;
                    stream.memcpy_dtod(&values.slice(..len), &mut v.slice_mut(..len))?;
                }
                (KvData::Fp8 { keys, values }, KvData::Fp8 { keys: k, values: v })
                | (KvData::Nvfp4 { keys, values }, KvData::Nvfp4 { keys: k, values: v })
                | (KvData::Nvfp4G32 { keys, values }, KvData::Nvfp4G32 { keys: k, values: v }) => {
                    stream.memcpy_dtod(&keys.slice(..len), &mut k.slice_mut(..len))?;
                    stream.memcpy_dtod(&values.slice(..len), &mut v.slice_mut(..len))?;
                }
                _ => anyhow::bail!("the slots' cache formats differ"),
            }
            for (from, to) in s.conv.iter().zip(d.conv.iter_mut()) {
                stream.memcpy_dtod(from, to)?;
            }
        }
        let sequence = self.new_sequence();
        if to == 0 {
            self.position = position;
            self.sequence = sequence;
        } else {
            let slot = &mut self.slots[to - 1];
            slot.position = position;
            slot.sequence = sequence;
        }
        Ok(())
    }

    /// Moves the sequence in slot `from` to slot `to`. It copies, as [`Model::move_slot`]
    /// does, when `to` holds the pages for it or the pool has them free: a copy runs at
    /// device bandwidth, about 1 ms for 40,000 positions, while remapping a page costs about
    /// 150 µs. Otherwise it maps `from`'s pages into `to`, which needs no second copy of a
    /// long sequence, and leaves `from` empty at position 0. The sequence in `to` is lost.
    pub fn transfer_slot(&mut self, from: usize, to: usize) -> anyhow::Result<()> {
        self.slot_position(from)?;
        self.slot_position(to)?;
        if from == to {
            return Ok(());
        }
        let missing = self.slot_pages_to(to, self.slot_position(from)?.max(1))?;
        let free = self
            .kv_pool
            .as_ref()
            .map_or(usize::MAX, PagePool::free_pages);
        if missing <= free {
            return self.move_slot(from, to);
        }
        let stream = Arc::clone(&self.ctx.stream);
        // Nothing queued may touch the pages while they move.
        stream.synchronize()?;
        let position = self.slot_position(from)?;
        let mut moved = Vec::new();
        let moved_rings = {
            let (from_visits, from_slab) = slot_visits(self, from);
            for v in from_visits.iter_mut() {
                // Each reservation's pages: the keys' and the values', or a packed ring's.
                let mut pages = Vec::new();
                for b in v.kv.pooled_mut().into_iter().flatten() {
                    pages.push(b.pages().to_vec());
                    b.unmap_from(0)?;
                }
                moved.push(pages);
            }
            from_slab.map(RingSlab::take_pages).transpose()?
        };
        {
            let (to_visits, to_slab) = slot_visits(self, to);
            if let (Some(dst_slab), Some(ring_pages)) = (to_slab, moved_rings.as_deref()) {
                dst_slab.map_shared_pages(ring_pages)?;
            }
            for (v, pages) in to_visits.iter_mut().zip(&moved) {
                // The slots' caches have the same shapes, so their reservations pair up.
                for (b, p) in v.kv.pooled_mut().into_iter().flatten().zip(pages) {
                    b.unmap_from(0)?;
                    b.map_shared(p)?;
                }
            }
        }
        // Pooled rings moved with their pages; rings of fixed memory and the conv states
        // are copied.
        let (src, dst) = match (from, to) {
            (0, t) => (&self.visits, &mut self.slots[t - 1].visits),
            (f, 0) => (&self.slots[f - 1].visits, &mut self.visits),
            (f, t) if f < t => {
                let (low, high) = self.slots.split_at_mut(t - 1);
                (&low[f - 1].visits, &mut high[0].visits)
            }
            (f, t) => {
                let (low, high) = self.slots.split_at_mut(f - 1);
                (&high[0].visits, &mut low[t - 1].visits)
            }
        };
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            if s.kv.window().is_some() && !s.kv.is_pooled() {
                let len = s.kv.slots() * s.kv.row_len();
                match (s.kv.data(), d.kv.data_mut()) {
                    (KvData::Bf16 { keys, values }, KvData::Bf16 { keys: k, values: v }) => {
                        stream.memcpy_dtod(&keys.slice(..len), &mut k.slice_mut(..len))?;
                        stream.memcpy_dtod(&values.slice(..len), &mut v.slice_mut(..len))?;
                    }
                    (KvData::Fp8 { keys, values }, KvData::Fp8 { keys: k, values: v })
                    | (KvData::Nvfp4 { keys, values }, KvData::Nvfp4 { keys: k, values: v })
                    | (
                        KvData::Nvfp4G32 { keys, values },
                        KvData::Nvfp4G32 { keys: k, values: v },
                    ) => {
                        stream.memcpy_dtod(&keys.slice(..len), &mut k.slice_mut(..len))?;
                        stream.memcpy_dtod(&values.slice(..len), &mut v.slice_mut(..len))?;
                    }
                    _ => anyhow::bail!("the slots' cache formats differ"),
                }
            }
            for (from, to) in s.conv.iter().zip(d.conv.iter_mut()) {
                stream.memcpy_dtod(from, to)?;
            }
        }
        let (moved_to, emptied) = (self.new_sequence(), self.new_sequence());
        let mut set = |slot: usize, position: usize, sequence: u64| {
            if slot == 0 {
                self.position = position;
                self.sequence = sequence;
            } else {
                let s = &mut self.slots[slot - 1];
                s.position = position;
                s.sequence = sequence;
            }
        };
        set(to, position, moved_to);
        set(from, 0, emptied);
        Ok(())
    }

    /// Makes batched steps record each row's feature taps for the draft, in the order of
    /// [`Model::tap_layout`] ([`Model::batch_taps`]), or stop recording them. Drops the
    /// batched step's buffers and graphs when this changes.
    pub fn set_batch_taps(&mut self, on: bool) {
        if on != self.batch_taps {
            self.batch_taps = on;
            self.batch = None;
            self.batch_graphs.iter_mut().for_each(|g| *g = None);
        }
    }

    /// The feature taps of row `row` of the last batched step, BF16 `[taps, 2304]` in the
    /// order of [`Model::tap_layout`], if batched steps record them
    /// ([`Model::set_batch_taps`]).
    pub fn batch_taps(&self, row: usize) -> anyhow::Result<CudaView<'_, bf16>> {
        let p = self.batch.as_ref().context("no batched step has run")?;
        let taps = p
            .taps
            .as_ref()
            .context("batched steps do not record taps")?;
        ensure!(row < p.rows, "a batch has at most {} rows", p.rows);
        let row_len = p.tap_layout.len() * self.ctx.hidden();
        Ok(taps.slice(row * row_len..(row + 1) * row_len))
    }

    /// Starts a new sequence in slot `slot`: zeroes its caches and conv states and moves it
    /// to position 0.
    pub fn reset_slot(&mut self, slot: usize) -> anyhow::Result<()> {
        self.with_slot(slot, Self::reset)
    }

    /// [`Model::skip_to_for_benchmark`] for slot `slot`.
    pub fn skip_slot_to_for_benchmark(
        &mut self,
        slot: usize,
        position: usize,
    ) -> anyhow::Result<()> {
        self.with_slot(slot, |m| m.skip_to_for_benchmark(position))
    }

    /// [`Model::prefill`] into slot `slot`, from its position, returning the next token,
    /// greedy or drawn with [`Model::sampling`]. The other slots are left alone.
    pub fn prefill_slot(
        &mut self,
        slot: usize,
        tokens: &[u32],
        chunk: usize,
    ) -> anyhow::Result<u32> {
        self.with_slot(slot, |m| m.prefill(tokens, chunk, PrefillLogits::Last))
    }

    /// [`Model::prefill_slot`] for a prompt with images ([`Model::prefill_with_images`]).
    pub fn prefill_slot_with_images(
        &mut self,
        slot: usize,
        tokens: &[u32],
        chunk: usize,
        images: ImageSource<'_>,
    ) -> anyhow::Result<u32> {
        self.with_slot(slot, |m| {
            m.prefill_with_images(tokens, chunk, PrefillLogits::Last, None, images)
        })
    }

    /// [`Model::prefill_with_taps`] into slot `slot`, passing each chunk's feature taps to
    /// `taps` on the model's stream.
    pub fn prefill_slot_with_taps(
        &mut self,
        slot: usize,
        tokens: &[u32],
        chunk: usize,
        taps: super::prefill::TapsCallback<'_>,
    ) -> anyhow::Result<u32> {
        self.with_slot(slot, |m| {
            m.prefill_with_taps(tokens, chunk, PrefillLogits::Last, taps)
        })
    }

    /// [`Model::prefill_with_images`] with taps into slot `slot`.
    pub fn prefill_slot_with_images_and_taps(
        &mut self,
        slot: usize,
        tokens: &[u32],
        chunk: usize,
        taps: Option<super::prefill::TapsCallback<'_>>,
        images: ImageSource<'_>,
    ) -> anyhow::Result<u32> {
        self.with_slot(slot, |m| {
            m.prefill_with_images(tokens, chunk, PrefillLogits::Last, taps, images)
        })
    }

    /// The token after slot `slot`'s last prefill, picked with `sampling` instead of
    /// [`Model::sampling`]: the token that prefill would have returned had it run with
    /// `sampling`. It reads the last row's logits that the prefill left in the decode
    /// buffers, so nothing may run between the two. With [`Model::move_slot`], one prefill
    /// serves several sequences of the same prompt that sample differently.
    pub fn redraw_slot(&mut self, slot: usize, sampling: Sampling) -> anyhow::Result<u32> {
        if sampling.is_greedy() {
            // A sampled draw replaces the argmax the prefill computed.
            let s = &mut self.scratch;
            self.ctx.kernels.misc.argmax_f32(
                &self.ctx.stream,
                &s.capped,
                &mut s.next,
                self.ctx.config.unpadded_vocab_size,
            )?;
        }
        let kept = std::mem::replace(&mut self.sampling, sampling);
        let next = self.with_slot(slot, Self::next_token);
        self.sampling = kept;
        next
    }

    /// Runs the next token of each of `rows`' slots, `(slot, token)` with distinct slots,
    /// 1 to [`BATCH_ROWS`] of them, as one batched step at each slot's position. Returns
    /// each row's greedy next token and advances each slot by one. Each row's results
    /// equal a one-row verify of its slot (see the module docs). Synchronizes once, to read
    /// the tokens.
    pub fn step_batch(&mut self, rows: &[(usize, u32)]) -> anyhow::Result<Vec<u32>> {
        self.step_batch_sampled(rows, &vec![Sampling::GREEDY; rows.len()])
    }

    /// [`Self::step_batch`] with each row's token picked by its entry of `sampling`, with
    /// sampling on ([`Model::enable_sampling`]). A row's draw depends only on its logits,
    /// its sampling and its position, so it equals a one-row verify's.
    pub fn step_batch_sampled(
        &mut self,
        rows: &[(usize, u32)],
        sampling: &[Sampling],
    ) -> anyhow::Result<Vec<u32>> {
        let m = rows.len();
        ensure!(sampling.len() == m, "one sampling per row");
        ensure!(
            self.sampler || sampling.iter().all(Sampling::is_greedy),
            "sampling in a batched step needs enable_sampling"
        );
        ensure!(
            (1..=BATCH_ROWS).contains(&m),
            "a batched step takes 1 to {BATCH_ROWS} rows, not {m}"
        );
        let mut positions = Vec::with_capacity(m);
        let mut tokens = Vec::with_capacity(m);
        for (i, &(slot, token)) in rows.iter().enumerate() {
            ensure!(
                rows[..i].iter().all(|&(s, _)| s != slot),
                "slot {slot} appears twice in a batch"
            );
            let position = self.slot_position(slot)?;
            ensure!(
                self.visits.iter().all(|v| v.kv.holds(position)),
                "slot {slot}: the global KV caches are full at {position} tokens"
            );
            self.map_slot(slot, position + 1)?;
            positions.push(u32::try_from(position).context("position does not fit the kernels")?);
            tokens.push(i32::try_from(token)?);
        }
        self.ensure_batch(m)?;
        let stream = Arc::clone(&self.ctx.stream);
        let members: Vec<usize> = rows.iter().map(|&(slot, _)| slot).collect();
        let p = self
            .batch
            .as_mut()
            .context("the batch buffers are missing")?;
        let b = p.batch.as_mut().context("the batch buffers are missing")?;
        if b.members != members {
            // Every visit's table, max_seqs lines each.
            let mut table = vec![0u64; self.visits.len() * b.max_seqs * BATCH_STATE_WORDS];
            for (r, &slot) in members.iter().enumerate() {
                let visits = if slot == 0 {
                    &self.visits
                } else {
                    &self.slots[slot - 1].visits
                };
                for (i, v) in visits.iter().enumerate() {
                    let at = (i * b.max_seqs + r) * BATCH_STATE_WORDS;
                    table[at..at + BATCH_STATE_WORDS].copy_from_slice(&state_words(&stream, v));
                }
            }
            stream.memcpy_htod(&table, &mut b.tables)?;
            b.members = members;
        }
        stream.memcpy_htod(&positions, &mut b.positions.slice_mut(..m))?;
        stream.memcpy_htod(&tokens, &mut p.tokens.slice_mut(..m))?;
        if let Some(sample) = p.sample.as_mut() {
            let settings: Vec<(Sampling, usize)> = sampling
                .iter()
                .zip(&positions)
                .map(|(&s, &position)| (s, position as usize + 1))
                .collect();
            sample.upload(&stream, &settings)?;
        }
        self.launch_batch(m)?;
        let p = self
            .batch
            .as_ref()
            .context("the batch buffers are missing")?;
        let b = p.batch.as_ref().context("the batch buffers are missing")?;
        let next = stream.clone_dtoh(&b.argmax.slice(..m))?;
        for &(slot, _) in rows {
            if slot == 0 {
                self.position += 1;
            } else {
                self.slots[slot - 1].position += 1;
            }
        }
        next.into_iter()
            .map(|t| u32::try_from(t).context("argmax returned a negative index"))
            .collect()
    }

    /// The F32 logits after the softcap of the last batched step's first `rows` rows,
    /// `[rows, unpadded_vocab]`.
    pub fn batch_logits(&self, rows: usize) -> anyhow::Result<Vec<f32>> {
        let p = self.batch.as_ref().context("no batched step has run")?;
        ensure!(rows <= p.rows, "a batch has at most {} rows", p.rows);
        self.ctx.compute_capped_logits(&p.final_hidden, rows)
    }

    /// Makes the batched step's buffers for one row per slot, or for at least `m` rows,
    /// unless large enough ones exist. New buffers drop the graphs, which hold the old
    /// ones' addresses.
    pub(super) fn ensure_batch(&mut self, m: usize) -> anyhow::Result<()> {
        let rows = self.slots().max(m).min(BATCH_ROWS);
        if self.batch.as_ref().is_some_and(|p| p.rows >= rows) {
            return Ok(());
        }
        self.batch = None;
        self.batch_graphs.iter_mut().for_each(|g| *g = None);
        let stream = &self.ctx.stream;
        let mut p = PrefillScratch::new(
            stream,
            &self.ctx.config,
            self.max_blocks,
            rows,
            true,
            self.sampler,
            self.ctx.fp8_wide_head(),
            false,
        )?;
        if self.batch_taps {
            p.record_taps = true;
            p.tap_layout = self.tap_layout.clone();
            p.taps = Some(zeros(
                stream,
                rows * p.tap_layout.len() * self.ctx.hidden(),
            )?);
        }
        p.batch = Some(BatchBuffers::new(
            stream,
            self.visits.len(),
            rows,
            rows,
            None,
        )?);
        if self.sampler {
            p.sample = Some(SampleBuffers::new(stream, rows)?);
        }
        self.batch = Some(p);
        Ok(())
    }

    /// Runs a batched verify pass over `slots` (distinct slots, `1..=BATCH_ROWS`), each
    /// verifying `width` (`1..=VERIFY_ROWS`) tokens with `slots.len() * width <= 256`,
    /// keeping the rows `accept[s]` chooses and rolling back the rest for each sequence.
    /// Every sequence's verdict, logits and post-rollback caches equal single-sequence
    /// [`Model::verify`] bit for bit.
    pub fn verify_batch(
        &mut self,
        slots: &[usize],
        tokens: BatchVerifyTokens<'_>,
        accept: &[Accept],
        sampling: &[Sampling],
    ) -> anyhow::Result<Vec<Verdict>> {
        let pending = self.start_verify_batch(slots, tokens, accept, sampling)?;
        self.finish_verify_batch(pending)
    }

    /// Launches [`Self::verify_batch`] without waiting for it. Work enqueued on the
    /// model's stream before [`Self::finish_verify_batch`] runs after the pass and can
    /// read [`Self::batch_verify_taps`], [`Self::batch_verify_results`] and
    /// [`Self::batch_verify_anchors`].
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid batch layout, unavailable cache space, an
    /// outstanding verify pass, or a CUDA launch failure.
    #[expect(
        clippy::too_many_lines,
        reason = "validates slots, uploads inputs and launches graph"
    )]
    pub fn start_verify_batch(
        &mut self,
        slots: &[usize],
        tokens: BatchVerifyTokens<'_>,
        accept: &[Accept],
        sampling: &[Sampling],
    ) -> anyhow::Result<PendingBatchVerify> {
        let seqs = slots.len();
        ensure!(
            (1..=BATCH_ROWS).contains(&seqs),
            "a batched verify takes 1 to {BATCH_ROWS} sequences, not {seqs}"
        );
        ensure!(accept.len() == seqs, "one accept mode per sequence");
        ensure!(sampling.len() == seqs, "one sampling per sequence");
        ensure!(
            self.sampler || sampling.iter().all(Sampling::is_greedy),
            "sampling in a batched verify needs enable_sampling"
        );
        let width = match &tokens {
            BatchVerifyTokens::Host(rows) => {
                ensure!(rows.len() == seqs, "one token slice per sequence");
                let w = rows[0].len();
                ensure!(
                    rows.iter().all(|r| r.len() == w),
                    "every sequence in a batched verify must have the same width"
                );
                w
            }
            BatchVerifyTokens::Device { tokens: d, width } => {
                ensure!(
                    d.len() == seqs * *width,
                    "device tokens has {} elements, expected {seqs} x {width}",
                    d.len()
                );
                *width
            }
        };
        ensure!(
            (1..=VERIFY_ROWS).contains(&width),
            "a batched verify takes width 1 to {VERIFY_ROWS}, not {width}"
        );
        let total_rows = seqs * width;
        ensure!(
            total_rows <= BATCH_VERIFY_MAX_ROWS,
            "{seqs} sequences x {width} rows = {total_rows} exceeds {BATCH_VERIFY_MAX_ROWS}"
        );

        let mut positions = Vec::with_capacity(seqs);
        let mut slot_starts = Vec::with_capacity(seqs);
        let mut keeps = Vec::with_capacity(seqs);
        for (i, &slot) in slots.iter().enumerate() {
            ensure!(
                slots[..i].iter().all(|&s| s != slot),
                "slot {slot} appears twice in a batched verify"
            );
            let start = self.slot_position(slot)?;
            let last = start + width - 1;
            ensure!(
                self.visits.iter().all(|v| v.kv.holds(last)),
                "slot {slot}: the global KV caches cannot hold position {last}"
            );
            self.map_slot(slot, last + 1)?;
            positions.push(u32::try_from(start).context("position does not fit the kernels")?);
            slot_starts.push((slot, start));
            let keep = match accept[i] {
                Accept::Greedy => 0,
                Accept::Rows(n) => {
                    ensure!((1..=width).contains(&n), "cannot keep {n} of {width} rows");
                    u32::try_from(n)?
                }
            };
            keeps.push(keep);
        }

        let max_allowed = self
            .slots()
            .min(BATCH_ROWS)
            .min(BATCH_VERIFY_MAX_ROWS / width);
        let padded_seqs = siphon::memory::batch_seq_bucket(seqs, max_allowed);
        let padded_rows = padded_seqs * width;

        self.ensure_batch_verify(padded_seqs, padded_rows)?;
        let stream = Arc::clone(&self.ctx.stream);
        let p = self
            .batch_verify
            .as_mut()
            .context("the batch verify buffers are missing")?;
        let b = p
            .batch
            .as_mut()
            .context("the batch verify buffers are missing")?;
        let bv = b
            .verify
            .as_mut()
            .context("the batch verify state is missing")?;
        ensure!(
            !bv.pending,
            "the last batched verify's verdicts were not read"
        );
        b.h2d_done.synchronize()?;
        bv.ensure_dummies(&stream, padded_seqs - seqs)?;

        if b.members.as_slice() != slots || b.padded_seqs != padded_seqs {
            let table_len = self.visits.len() * b.max_seqs * BATCH_STATE_WORDS;
            {
                let table = b.host_tables.as_mut_slice();
                table[..table_len].fill(0);
                for (r, &slot) in slots.iter().enumerate() {
                    let visits = if slot == 0 {
                        &self.visits
                    } else {
                        &self.slots[slot - 1].visits
                    };
                    for (i, v) in visits.iter().enumerate() {
                        let at = (i * b.max_seqs + r) * BATCH_STATE_WORDS;
                        table[at..at + BATCH_STATE_WORDS].copy_from_slice(&state_words(&stream, v));
                    }
                }
                for (d, dummy) in bv.dummy_slots[..padded_seqs - seqs].iter().enumerate() {
                    let words = dummy.state_words(&stream);
                    let r = seqs + d;
                    for i in 0..self.visits.len() {
                        let at = (i * b.max_seqs + r) * BATCH_STATE_WORDS;
                        table[at..at + BATCH_STATE_WORDS].copy_from_slice(&words);
                    }
                }
            }
            b.host_tables
                .copy_to_device(&stream, table_len, &mut b.tables)?;
            b.members = slots.to_vec();
            b.padded_seqs = padded_seqs;
        }

        if padded_seqs > seqs {
            positions.resize(padded_seqs, 0);
            keeps.resize(padded_seqs, u32::try_from(width)?);
        }
        b.host_positions.as_mut_slice()[..padded_seqs].copy_from_slice(&positions);
        b.host_positions.copy_to_device(
            &stream,
            padded_seqs,
            &mut b.positions.slice_mut(..padded_seqs),
        )?;
        bv.host_keep.as_mut_slice()[..padded_seqs].copy_from_slice(&keeps);
        bv.host_keep
            .copy_to_device(&stream, padded_seqs, &mut bv.keep.slice_mut(..padded_seqs))?;
        match tokens {
            BatchVerifyTokens::Host(rows) => {
                {
                    let ids = b.host_tokens.as_mut_slice();
                    let mut idx = 0;
                    for row in rows {
                        for &tok in *row {
                            ids[idx] = i32::try_from(tok)?;
                            idx += 1;
                        }
                    }
                    ids[idx..padded_rows].fill(0);
                }
                b.host_tokens.copy_to_device(
                    &stream,
                    padded_rows,
                    &mut p.tokens.slice_mut(..padded_rows),
                )?;
            }
            BatchVerifyTokens::Device { tokens: d, .. } => {
                stream.memcpy_dtod(&d, &mut p.tokens.slice_mut(..total_rows))?;
                if padded_rows > total_rows {
                    stream.memset_zeros(&mut p.tokens.slice_mut(total_rows..padded_rows))?;
                }
            }
        }
        b.h2d_done.record(&stream)?;
        if let Some(sample) = p.sample.as_mut() {
            let mut settings = Vec::with_capacity(padded_rows);
            for (seq, &pos) in positions[..seqs].iter().enumerate() {
                for j in 0..width {
                    settings.push((sampling[seq], pos as usize + j + 1));
                }
            }
            settings.resize(padded_rows, (Sampling::GREEDY, 1));
            sample.upload(&stream, &settings)?;
        }
        bv.width = width;
        bv.last_seqs = seqs;
        bv.padded_seqs = padded_seqs;
        self.launch_batch_verify(padded_seqs, width)?;
        self.batch_verify
            .as_mut()
            .and_then(|p| p.batch.as_mut())
            .and_then(|b| b.verify.as_mut())
            .context("the batch verify state is missing")?
            .pending = true;
        Ok(PendingBatchVerify {
            slots: slot_starts,
            width,
        })
    }

    /// Waits for `pending`, reads each sequence's verdict and advances each slot's
    /// position past its kept rows.
    pub fn finish_verify_batch(
        &mut self,
        pending: PendingBatchVerify,
    ) -> anyhow::Result<Vec<Verdict>> {
        let seqs = pending.slots.len();
        let width = pending.width;
        for &(slot, start) in &pending.slots {
            ensure!(
                self.slot_position(slot)? == start,
                "slot {slot}'s position moved before finish_verify_batch"
            );
        }
        let stream = Arc::clone(&self.ctx.stream);
        let raw = {
            let bv = self
                .batch_verify
                .as_mut()
                .and_then(|p| p.batch.as_mut())
                .and_then(|b| b.verify.as_mut())
                .context("the batch verify state is missing")?;
            ensure!(
                bv.pending && bv.last_seqs == seqs && bv.width == width,
                "the verdicts belong to another batched verify"
            );
            let n = seqs * RESULT_LEN;
            bv.host_results
                .copy_from_device(&stream, &bv.results.slice(..n))?;
            stream.synchronize()?;
            bv.host_results.as_slice()[..n].to_vec()
        };
        self.complete_verify_batch(pending, &raw)
    }

    /// Completes `pending` from batched verify result words (`[seqs * RESULT_LEN]`) that the
    /// caller already copied to the host asynchronously (for example on a copy stream
    /// overlapped with draft execution), advancing each slot's position past its kept rows.
    pub fn complete_verify_batch(
        &mut self,
        pending: PendingBatchVerify,
        raw: &[i32],
    ) -> anyhow::Result<Vec<Verdict>> {
        let PendingBatchVerify { slots, width } = pending;
        let seqs = slots.len();
        ensure!(
            raw.len() >= seqs * RESULT_LEN,
            "batched verify results have {} elements, expected at least {}",
            raw.len(),
            seqs * RESULT_LEN
        );
        for &(slot, start) in &slots {
            ensure!(
                self.slot_position(slot)? == start,
                "slot {slot}'s position moved before complete_verify_batch"
            );
        }
        let bv = self
            .batch_verify
            .as_mut()
            .and_then(|p| p.batch.as_mut())
            .and_then(|b| b.verify.as_mut())
            .context("the batch verify state is missing")?;
        ensure!(
            bv.pending && bv.last_seqs == seqs && bv.width == width,
            "the verdicts belong to another batched verify"
        );
        bv.pending = false;
        let ids = |r: &[i32]| -> anyhow::Result<Vec<u32>> {
            r.iter()
                .map(|&t| u32::try_from(t).context("a negative token id"))
                .collect()
        };
        let mut verdicts = Vec::with_capacity(seqs);
        for (seq, &(slot, start)) in slots.iter().enumerate() {
            let result = &raw[seq * RESULT_LEN..(seq + 1) * RESULT_LEN];
            let accepted = usize::try_from(result[0])?;
            ensure!(
                accepted < width,
                "slot {slot}: accepted {accepted} of {} drafts",
                width - 1
            );
            if slot == 0 {
                self.position = start + accepted + 1;
            } else {
                self.slots[slot - 1].position = start + accepted + 1;
            }
            verdicts.push(Verdict {
                accepted,
                tokens: ids(&result[1..=width])?,
                argmax: ids(&result[1 + VERIFY_ROWS..1 + VERIFY_ROWS + width])?,
            });
        }
        Ok(verdicts)
    }

    /// Feature taps of the last batched verify including bucket-padded sequences, BF16
    /// `[padded_seqs * width, taps, 2304]`.
    pub fn batch_verify_taps_padded(&self) -> anyhow::Result<CudaView<'_, bf16>> {
        let p = self
            .batch_verify
            .as_ref()
            .context("no batched verify has run")?;
        let bv = p
            .batch
            .as_ref()
            .and_then(|b| b.verify.as_ref())
            .context("the batch verify state is missing")?;
        let taps = p.taps.as_ref().context("the taps buffer is missing")?;
        let row_len = p.tap_layout.len() * self.ctx.hidden();
        Ok(taps.slice(..bv.padded_seqs * bv.width * row_len))
    }

    /// Acceptance results of the last batched verify including bucket-padded sequences,
    /// `[padded_seqs * RESULT_LEN]`.
    pub fn batch_verify_results_padded(&self) -> anyhow::Result<CudaView<'_, i32>> {
        let bv = self
            .batch_verify
            .as_ref()
            .and_then(|p| p.batch.as_ref())
            .and_then(|b| b.verify.as_ref())
            .context("no batched verify has run")?;
        Ok(bv.results.slice(..bv.padded_seqs * RESULT_LEN))
    }

    /// Next anchors of the last batched verify including bucket-padded sequences,
    /// `[padded_seqs]`.
    pub fn batch_verify_anchors_padded(&self) -> anyhow::Result<CudaView<'_, i32>> {
        let bv = self
            .batch_verify
            .as_ref()
            .and_then(|p| p.batch.as_ref())
            .and_then(|b| b.verify.as_ref())
            .context("no batched verify has run")?;
        Ok(bv.anchors.slice(..bv.padded_seqs))
    }

    /// Feature taps of the last batched verify, BF16 `[seqs * width, taps, 2304]`.
    pub fn batch_verify_taps(&self) -> anyhow::Result<CudaView<'_, bf16>> {
        let p = self
            .batch_verify
            .as_ref()
            .context("no batched verify has run")?;
        let bv = p
            .batch
            .as_ref()
            .and_then(|b| b.verify.as_ref())
            .context("the batch verify state is missing")?;
        let taps = p.taps.as_ref().context("the taps buffer is missing")?;
        let row_len = p.tap_layout.len() * self.ctx.hidden();
        Ok(taps.slice(..bv.last_seqs * bv.width * row_len))
    }

    /// Acceptance results of the last batched verify on the device, `[seqs * RESULT_LEN]`.
    pub fn batch_verify_results(&self) -> anyhow::Result<CudaView<'_, i32>> {
        let bv = self
            .batch_verify
            .as_ref()
            .and_then(|p| p.batch.as_ref())
            .and_then(|b| b.verify.as_ref())
            .context("no batched verify has run")?;
        Ok(bv.results.slice(..bv.last_seqs * RESULT_LEN))
    }

    /// Next anchors of the last batched verify on the device, `[seqs]`.
    pub fn batch_verify_anchors(&self) -> anyhow::Result<CudaView<'_, i32>> {
        let bv = self
            .batch_verify
            .as_ref()
            .and_then(|p| p.batch.as_ref())
            .and_then(|b| b.verify.as_ref())
            .context("no batched verify has run")?;
        Ok(bv.anchors.slice(..bv.last_seqs))
    }

    /// F32 logits after softcap of the last batched verify, `[seqs * width, unpadded_vocab]`.
    pub fn batch_verify_logits(&self) -> anyhow::Result<Vec<f32>> {
        let p = self
            .batch_verify
            .as_ref()
            .context("no batched verify has run")?;
        let bv = p
            .batch
            .as_ref()
            .and_then(|b| b.verify.as_ref())
            .context("the batch verify state is missing")?;
        self.ctx
            .compute_capped_logits(&p.final_hidden, bv.last_seqs * bv.width)
    }

    /// Number of captured `(bucket_seqs, width)` batched verify CUDA graphs currently cached.
    #[must_use]
    pub fn batch_verify_graph_count(&self) -> usize {
        self.batch_verify_graphs.len()
    }

    /// Reserves [`Model::verify_batch`] scratch and backup buffers for up to `max_seqs`
    /// concurrent sequences (capped at `BATCH_VERIFY_MAX_ROWS / VERIFY_ROWS`), so startup
    /// memory checks observe them immediately.
    pub fn reserve_batch_verify(&mut self, max_seqs: usize) -> anyhow::Result<()> {
        let max_seqs = max_seqs.clamp(1, BATCH_VERIFY_MAX_ROWS / VERIFY_ROWS);
        self.ensure_batch_verify(max_seqs, max_seqs * VERIFY_ROWS)
    }

    /// Copies batched DFlash 2 16-candidate ids and normalized filtered probabilities
    /// (`[seqs * 7 * 16]` each) into fixed-address batch-verify scratch for `--spec-beta`.
    pub fn set_batch_verify_draft_dist(
        &mut self,
        seqs: usize,
        width: usize,
        cand: &CudaView<'_, i32>,
        q_probs: &CudaView<'_, f32>,
    ) -> anyhow::Result<()> {
        let stride_elems = crate::draft::STEPS * crate::ops::sample::BETA_CAND_K;
        let expected = seqs * stride_elems;
        ensure!(
            cand.len() == expected && q_probs.len() == expected,
            "batch draft dist len ({}, {}), expected {expected}",
            cand.len(),
            q_probs.len()
        );
        self.ensure_batch_verify(seqs, seqs * width)?;
        let stream = Arc::clone(&self.ctx.stream);
        let bv = self
            .batch_verify
            .as_mut()
            .and_then(|p| p.batch.as_mut())
            .and_then(|b| b.verify.as_mut())
            .context("the batch verify state is missing")?;
        stream.memcpy_dtod(cand, &mut bv.draft_cand.slice_mut(..expected))?;
        stream.memcpy_dtod(q_probs, &mut bv.draft_q.slice_mut(..expected))?;
        Ok(())
    }

    pub(super) fn ensure_batch_verify(
        &mut self,
        seqs: usize,
        total_rows: usize,
    ) -> anyhow::Result<()> {
        let max_seqs = self.slots().max(seqs).min(BATCH_ROWS);
        let max_rows = (max_seqs * VERIFY_ROWS)
            .max(total_rows)
            .min(BATCH_VERIFY_MAX_ROWS);
        if self.batch_verify.as_ref().is_some_and(|p| {
            p.rows >= max_rows && p.batch.as_ref().is_some_and(|b| b.max_seqs >= max_seqs)
        }) {
            if self.ctx.accept_margin > 0.0
                && let Some(p) = self.batch_verify.as_mut()
            {
                p.ensure_capped_head(&self.ctx.stream, &self.ctx.config)?;
            }
            return Ok(());
        }
        self.batch_verify = None;
        self.batch_verify_graphs.clear();
        let stream = &self.ctx.stream;
        let mut p = PrefillScratch::new(
            stream,
            &self.ctx.config,
            self.max_blocks,
            max_rows,
            true,
            self.sampler,
            self.ctx.fp8_wide_head(),
            false,
        )?;
        if self.ctx.accept_margin > 0.0 {
            p.ensure_capped_head(stream, &self.ctx.config)?;
        }
        p.record_taps = true;
        p.tap_layout = self.tap_layout.clone();
        p.taps = Some(zeros(
            stream,
            max_rows * p.tap_layout.len() * self.ctx.hidden(),
        )?);
        let bv = BatchVerifyState::new(stream, &self.visits, max_seqs)?;
        p.batch = Some(BatchBuffers::new(
            stream,
            self.visits.len(),
            max_seqs,
            max_rows,
            Some(bv),
        )?);
        if self.sampler {
            p.sample = Some(SampleBuffers::new(stream, max_rows)?);
        }
        self.batch_verify = Some(p);
        Ok(())
    }

    fn launch_batch_verify(&mut self, seqs: usize, width: usize) -> anyhow::Result<()> {
        let key = (seqs, width);
        if !self.batch_verify_graphs.contains_key(&key) {
            let stream = Arc::clone(&self.ctx.stream);
            stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let enqueued = self.enqueue_batch_verify(seqs * width);
            let graph = stream.end_capture(CUgraphInstantiate_flags(0));
            enqueued.context("capturing the batched verify pass")?;
            let graph = graph?.context("the batched verify capture recorded no work")?;
            graph.upload()?;
            self.batch_verify_graphs.insert(key, StepGraph(graph));
        }
        let graph = self
            .batch_verify_graphs
            .get(&key)
            .context("the batched verify graph is missing")?;
        graph.0.launch()?;
        Ok(())
    }

    fn enqueue_batch_verify(&mut self, total_rows: usize) -> anyhow::Result<()> {
        let Self {
            ctx,
            passes,
            visits,
            scratch,
            max_blocks,
            batch_verify,
            ..
        } = self;
        let p = batch_verify
            .as_mut()
            .context("the batch verify buffers are missing")?;
        ctx.prefill_enqueue(
            p,
            visits,
            *passes,
            *max_blocks,
            scratch,
            total_rows,
            0,
            &mut PrefillLogits::Last,
            true,
            &[],
        )
    }

    /// Replays the batched step graph of `m` rows, capturing it first if needed.
    fn launch_batch(&mut self, m: usize) -> anyhow::Result<()> {
        if self.batch_graphs[m - 1].is_none() {
            let stream = Arc::clone(&self.ctx.stream);
            stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let enqueued = self.enqueue_batch(m);
            // End the capture even when enqueueing failed, so the stream is usable again.
            let graph = stream.end_capture(CUgraphInstantiate_flags(0));
            enqueued.context("capturing the batched step")?;
            let graph = graph?.context("the batched step capture recorded no work")?;
            graph.upload()?;
            self.batch_graphs[m - 1] = Some(StepGraph(graph));
        }
        let graph = self.batch_graphs[m - 1]
            .as_ref()
            .context("the batched step graph is missing")?;
        graph.0.launch()?;
        Ok(())
    }

    fn enqueue_batch(&mut self, m: usize) -> anyhow::Result<()> {
        let Self {
            ctx,
            passes,
            visits,
            scratch,
            max_blocks,
            batch,
            ..
        } = self;
        let p = batch.as_mut().context("the batch buffers are missing")?;
        // Slot 0's visits give each visit's cache sizes and formats; the rows' own caches
        // come from the state tables.
        ctx.prefill_enqueue(
            p,
            visits,
            *passes,
            *max_blocks,
            scratch,
            m,
            0,
            &mut PrefillLogits::Last,
            true,
            &[],
        )
    }
}
