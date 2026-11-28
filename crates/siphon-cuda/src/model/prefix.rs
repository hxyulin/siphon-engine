//! Copies of the per-position state of a slot that later tokens overwrite, for the prefix
//! cache (`spec::PrefixCache`).
//!
//! Tokens after position `p` write global KV cache slots from `p` on, so the slots below
//! `p` stay as they were. The sliding rings and the conv states do not: later tokens
//! overwrite ring slots and shift the conv windows. A [`LocalState`] keeps copies of those
//! at one position, so that the sequence can return there while the global caches still
//! hold its positions below it.

use std::{ffi::c_void, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr as _, PinnedHostSlice, sys};

use super::{Model, VisitState};
use crate::ops::attention::KvData;

/// A device buffer's address and length in bytes, for copies to and from host memory
/// (the prefix cache's host tier, `spec::HostCache`).
#[derive(Clone, Copy, Debug)]
pub struct DeviceSpan {
    /// Raw CUDA device address, whose backing storage belongs to the model.
    pub ptr: u64,
    /// Size of the span in bytes.
    pub bytes: usize,
}

impl DeviceSpan {
    pub(crate) fn of<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> Self {
        Self {
            ptr: slice.device_ptr(stream).0,
            bytes: slice.num_bytes(),
        }
    }
}

/// The global KV caches of a slot or of the draft, from [`Model::global_kv`]: position `p`
/// of cache `c` takes the `position_bytes[c]` bytes at `caches[c] + p * position_bytes[c]`.
/// The caches of a mixed layout differ in size (FP8 and NVFP4).
#[derive(Clone, Debug)]
pub struct GlobalKv {
    /// Each global cache's keys, then its values, by visit or layer.
    pub caches: Vec<u64>,
    /// Bytes of one position in each cache.
    pub position_bytes: Vec<usize>,
    /// Positions each cache holds.
    pub capacity: usize,
}

impl GlobalKv {
    /// Bytes of one position over every cache.
    #[must_use]
    pub fn bytes_per_position(&self) -> usize {
        self.position_bytes.iter().sum()
    }

    /// Each cache's address, bytes per position, and the bytes per position of the caches
    /// before it: a block of `n` positions of every cache, one cache after another, holds
    /// cache `c` from `n` times that last number on.
    pub fn layout(&self) -> impl Iterator<Item = (u64, usize, usize)> + '_ {
        let mut before = 0;
        self.caches
            .iter()
            .zip(&self.position_bytes)
            .map(move |(&cache, &bytes)| {
                before += bytes;
                (cache, bytes, before - bytes)
            })
    }
}

/// The ring and conv state buffers of `visits`, in the order of [`LocalState::spans`].
fn visit_spans(visits: &[VisitState], stream: &CudaStream) -> Vec<DeviceSpan> {
    let mut spans = Vec::new();
    for v in visits {
        if v.kv.window().is_some() {
            match v.kv.data() {
                KvData::Bf16 { keys, values } => {
                    spans.push(DeviceSpan::of(keys, stream));
                    spans.push(DeviceSpan::of(values, stream));
                }
                KvData::Fp8 { keys, values }
                | KvData::Nvfp4 { keys, values }
                | KvData::Nvfp4G32 { keys, values } => {
                    spans.push(DeviceSpan::of(keys, stream));
                    spans.push(DeviceSpan::of(values, stream));
                }
            }
        }
        spans.extend(v.conv.iter().map(|c| DeviceSpan::of(c, stream)));
    }
    spans
}

/// A slot's sliding rings, conv states and position at one point of a sequence, from
/// [`Model::save_local_state`]. Stored in page-locked host memory so prefix-cache
/// snapshots consume zero GPU VRAM and zero KV pool pages.
pub struct LocalState {
    /// Page-locked host buffer holding every sliding ring and conv state in
    /// [`visit_spans`] order.
    _buf: PinnedHostSlice<u8>,
    /// Base host address of `_buf`.
    base: usize,
    /// `(byte_offset, byte_len)` of each buffer in [`visit_spans`] order.
    spans: Vec<(usize, usize)>,
    /// Number of visits this snapshot was sized for.
    visit_count: usize,
    /// Stream used for async saves/restores, synchronized on drop.
    stream: Arc<CudaStream>,
    position: usize,
    valid: bool,
}

impl Drop for LocalState {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

impl LocalState {
    /// The position the copies were saved at.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Each saved buffer's `(host_ptr, bytes)` in [`Model::local_spans`] order.
    #[must_use]
    pub(crate) fn host_spans(&self) -> Vec<(usize, usize)> {
        self.spans
            .iter()
            .map(|&(off, len)| (self.base + off, len))
            .collect()
    }

    /// Whether the snapshot currently holds saved state.
    ///
    /// Call [`Model::save_local_state`] to fill the buffers before restoring them.
    #[must_use]
    pub fn holds(&self) -> bool {
        self.valid
    }

    /// Pages of the KV pool that the copies of the rings hold (always 0 for host-backed
    /// snapshots).
    #[must_use]
    pub fn pages(&self) -> usize {
        0
    }

    /// Marks the snapshot empty (host-backed snapshots hold no KV pool pages).
    ///
    /// # Errors
    ///
    /// Never fails for host-backed snapshots.
    pub fn release(&mut self, _stream: &CudaStream) -> anyhow::Result<()> {
        self.valid = false;
        Ok(())
    }

    /// Device bytes the copies take (always 0 for host-backed snapshots).
    #[must_use]
    pub fn bytes(&self) -> usize {
        0
    }
}

impl Model {
    /// The global KV caches of slot `slot`.
    ///
    /// Pooled addresses can be reserved beyond the mapped range. Use
    /// [`Self::slot_mapped`] before copying memory from these addresses.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot is unknown.
    pub fn global_kv(&self, slot: usize) -> anyhow::Result<GlobalKv> {
        let stream = &self.ctx.stream;
        let visits = if slot == 0 {
            &self.visits
        } else {
            &self.slot_state(slot)?.visits
        };
        let mut caches = Vec::new();
        let mut position_bytes = Vec::new();
        let mut capacity = 0;
        for v in visits.iter().filter(|v| v.kv.window().is_none()) {
            let (keys, values) = match v.kv.data() {
                KvData::Bf16 { keys, values } => {
                    (DeviceSpan::of(keys, stream), DeviceSpan::of(values, stream))
                }
                KvData::Fp8 { keys, values }
                | KvData::Nvfp4 { keys, values }
                | KvData::Nvfp4G32 { keys, values } => {
                    (DeviceSpan::of(keys, stream), DeviceSpan::of(values, stream))
                }
            };
            caches.extend([keys.ptr, values.ptr]);
            position_bytes.extend([v.kv.row_bytes(); 2]);
            capacity = v.kv.slots();
        }
        Ok(GlobalKv {
            caches,
            position_bytes,
            capacity,
        })
    }

    /// Slot `slot`'s rings and conv states, in the order of [`LocalState::spans`].
    pub fn local_spans(&self, slot: usize) -> anyhow::Result<Vec<DeviceSpan>> {
        let visits = if slot == 0 {
            &self.visits
        } else {
            &self.slot_state(slot)?.visits
        };
        Ok(visit_spans(visits, &self.ctx.stream))
    }

    /// Moves slot `slot`, just reset, to `position`, once its global caches below
    /// `position` and its rings and conv states hold a sequence's state at that position,
    /// copied from elsewhere (the prefix cache's host tier).
    pub fn set_loaded_position(&mut self, slot: usize, position: usize) -> anyhow::Result<()> {
        self.with_slot(slot, |m| {
            ensure!(
                m.position == 0,
                "slot {slot} is at {}, not reset",
                m.position
            );
            ensure!(
                m.visits
                    .iter()
                    .all(|v| v.kv.holds(position.saturating_sub(1))),
                "the global KV caches cannot hold position {position}"
            );
            ensure!(
                m.visits
                    .iter()
                    .filter(|v| v.kv.window().is_none())
                    .all(|v| v.kv.mapped() >= position),
                "slot {slot}'s global caches are not mapped up to {position}"
            );
            m.position = position;
            Ok(())
        })
    }

    /// Allocates a [`LocalState`] in page-locked host memory for a slot's rings and conv
    /// states, holding nothing yet.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA cannot allocate the pinned host buffer.
    pub fn local_state(&self) -> anyhow::Result<LocalState> {
        let stream = &self.ctx.stream;
        let dev_spans = visit_spans(&self.visits, stream);
        let mut total = 0usize;
        let mut spans = Vec::with_capacity(dev_spans.len());
        for s in &dev_spans {
            spans.push((total, s.bytes));
            total += s.bytes;
        }
        // SAFETY: bytes are written by async D2H before any H2D or host copy reads them.
        let mut buf = unsafe {
            stream
                .context()
                .alloc_pinned_with_flags::<u8>(total.max(1), 0)
        }
        .context("allocating pinned host memory for prefix local state")?;
        let base = buf.as_mut_ptr()? as usize;
        Ok(LocalState {
            _buf: buf,
            base,
            spans,
            visit_count: self.visits.len(),
            stream: Arc::clone(stream),
            position: 0,
            valid: false,
        })
    }

    /// Copies slot `slot`'s rings, conv states and position into `state`, on the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot or saved-state layout is invalid, or if CUDA cannot
    /// copy the buffers.
    pub fn save_local_state(&mut self, slot: usize, state: &mut LocalState) -> anyhow::Result<()> {
        self.with_slot(slot, |m| m.save_slot0_state(state))
    }

    fn save_slot0_state(&self, state: &mut LocalState) -> anyhow::Result<()> {
        let stream = &self.ctx.stream;
        ensure!(
            state.visit_count == self.visits.len(),
            "the state has {} visits, the model {}",
            state.visit_count,
            self.visits.len()
        );
        let dev_spans = visit_spans(&self.visits, stream);
        ensure!(
            dev_spans.len() == state.spans.len(),
            "the state has {} spans, the model {}",
            state.spans.len(),
            dev_spans.len()
        );
        for (src, &(off, len)) in dev_spans.iter().zip(&state.spans) {
            ensure!(src.bytes == len, "the local state buffer sizes differ");
            // SAFETY: `state._buf` is a live pinned host allocation sized for `off + len`
            // and `src.ptr` is a valid device slice; `stream` orders all accesses.
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    (state.base + off) as *mut c_void,
                    src.ptr,
                    len,
                    stream.cu_stream(),
                )
            }
            .result()
            .context("saving local state to pinned host memory")?;
        }
        state.position = self.position;
        state.valid = true;
        Ok(())
    }

    /// Returns slot `slot` to `state`: copies back its rings and conv states and moves to
    /// its position. The caller makes sure that the slot holds the sequence `state` was
    /// saved from, with the same [`Model::slot_sequence`] as when it was saved or copied
    /// there: its global caches then keep the slots below that position, which the
    /// sequence wrote before `state` was saved and has not written since.
    pub fn restore_local_state(&mut self, slot: usize, state: &LocalState) -> anyhow::Result<()> {
        self.with_slot(slot, |m| m.restore_slot0_state(state))
    }

    fn restore_slot0_state(&mut self, state: &LocalState) -> anyhow::Result<()> {
        ensure!(
            self.position >= state.position,
            "the slot is at {}, before the saved state's {}",
            self.position,
            state.position
        );
        ensure!(state.holds(), "the saved state's copies were released");
        let stream = Arc::clone(&self.ctx.stream);
        // A pooled ring slab of a slot that held no sequence takes its pages now.
        super::map_visits(&mut self.visits, self.ring_slab.as_mut(), 0, &stream)?;
        let dev_spans = visit_spans(&self.visits, &stream);
        ensure!(
            dev_spans.len() == state.spans.len(),
            "the state has {} spans, the model {}",
            state.spans.len(),
            dev_spans.len()
        );
        for (dst, &(off, len)) in dev_spans.iter().zip(&state.spans) {
            ensure!(dst.bytes == len, "the local state buffer sizes differ");
            // SAFETY: `state._buf` is a live pinned host allocation sized for `off + len`
            // and `dst.ptr` is a valid device slice; `stream` orders all accesses.
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    dst.ptr,
                    (state.base + off) as *const c_void,
                    len,
                    stream.cu_stream(),
                )
            }
            .result()
            .context("restoring local state from pinned host memory")?;
        }
        self.position = state.position;
        Ok(())
    }
}
