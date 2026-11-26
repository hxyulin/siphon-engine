//! Copies of the per-position state of a slot that later tokens overwrite, for the prefix
//! cache (`spec::PrefixCache`).
//!
//! Tokens after position `p` write global KV cache slots from `p` on, so the slots below
//! `p` stay as they were. The sliding rings and the conv states do not: later tokens
//! overwrite ring slots and shift the conv windows. A [`LocalState`] keeps copies of those
//! at one position, so that the sequence can return there while the global caches still
//! hold its positions below it.

use std::sync::Arc;

use anyhow::ensure;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr as _};

use super::{Model, VisitState};
use crate::ops::attention::{KvCache, KvData};

/// A device buffer's address and length in bytes, for copies to and from host memory
/// (the prefix cache's host tier, `spec::HostCache`).
#[derive(Clone, Copy, Debug)]
pub struct DeviceSpan {
    pub ptr: u64,
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
                KvData::Fp8 { keys, values } | KvData::Nvfp4 { keys, values } => {
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
/// [`Model::save_local_state`]. The copies belong to no slot: they can return any slot that
/// holds the sequence's global caches to that point.
pub struct LocalState {
    /// Each visit's ring, `None` for a global cache, and its four conv states. With a KV
    /// pool the rings take pages only while they hold a copy.
    visits: Vec<(Option<KvCache>, Vec<CudaSlice<f32>>)>,
    position: usize,
}

impl LocalState {
    /// The position the copies were saved at.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Each copy's device buffer: per visit its ring's keys and values, if it has a ring,
    /// then its four conv states. [`Model::local_spans`] lists a slot's buffers in the same
    /// order.
    pub(crate) fn spans(&self, stream: &CudaStream) -> Vec<DeviceSpan> {
        let mut spans = Vec::new();
        for (ring, conv) in &self.visits {
            match ring.as_ref().map(KvCache::data) {
                Some(KvData::Bf16 { keys, values }) => {
                    spans.push(DeviceSpan::of(keys, stream));
                    spans.push(DeviceSpan::of(values, stream));
                }
                Some(KvData::Fp8 { keys, values } | KvData::Nvfp4 { keys, values }) => {
                    spans.push(DeviceSpan::of(keys, stream));
                    spans.push(DeviceSpan::of(values, stream));
                }
                None => {}
            }
            spans.extend(conv.iter().map(|c| DeviceSpan::of(c, stream)));
        }
        spans
    }

    /// Whether the copies hold a state: saved, and not released since.
    #[must_use]
    pub fn holds(&self) -> bool {
        self.visits
            .iter()
            .filter_map(|(ring, _)| ring.as_ref())
            .all(|r| r.mapped() == r.slots())
    }

    /// Pages of the KV pool that the copies of the rings hold.
    #[must_use]
    pub fn pages(&self) -> usize {
        self.visits
            .iter()
            .filter_map(|(ring, _)| ring.as_ref()?.pooled())
            .map(|p| p.iter().map(|b| b.pages().len()).sum::<usize>())
            .sum()
    }

    /// Returns the pages of the copies of pooled rings to the pool, once `stream` has done
    /// its work: the copies hold nothing until saved again. Work on other streams that
    /// reads them must be done.
    pub fn release(&mut self, stream: &CudaStream) -> anyhow::Result<()> {
        if self.pages() == 0 {
            return Ok(());
        }
        stream.synchronize()?;
        for ring in self.visits.iter_mut().filter_map(|(ring, _)| ring.as_mut()) {
            ring.unmap_from(0)?;
        }
        Ok(())
    }

    /// Device bytes the copies take, but for copies of rings in pooled memory, which take
    /// pages only while they hold a state ([`Self::pages`]).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.visits
            .iter()
            .map(|(ring, conv)| {
                let ring = match ring.as_ref().filter(|r| !r.is_pooled()).map(KvCache::data) {
                    Some(KvData::Bf16 { keys, values }) => (keys.len() + values.len()) * 2,
                    Some(KvData::Fp8 { keys, values } | KvData::Nvfp4 { keys, values }) => {
                        keys.len() + values.len()
                    }
                    None => 0,
                };
                ring + conv.iter().map(|c| c.len() * 4).sum::<usize>()
            })
            .sum()
    }
}

impl Model {
    /// The global KV caches of slot `slot`.
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
                KvData::Fp8 { keys, values } | KvData::Nvfp4 { keys, values } => {
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

    /// Allocates a [`LocalState`] for a slot's rings and conv states, holding nothing yet.
    pub fn local_state(&self) -> anyhow::Result<LocalState> {
        let stream = &self.ctx.stream;
        let visits = self
            .visits
            .iter()
            .map(|v| {
                let ring = match (v.kv.window(), &self.kv_pool) {
                    (None, _) => None,
                    (Some(window), Some(pool)) => Some(KvCache::sliding_pooled_with_format(
                        stream,
                        window,
                        v.kv.slots(),
                        v.kv.format(),
                        pool,
                    )?),
                    (Some(window), None) => Some(KvCache::sliding_ring_with_format(
                        stream,
                        window,
                        v.kv.slots(),
                        v.kv.format(),
                    )?),
                };
                let conv = v
                    .conv
                    .iter()
                    .map(|c| stream.alloc_zeros(c.len()))
                    .collect::<Result<_, _>>()?;
                Ok((ring, conv))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(LocalState {
            visits,
            position: 0,
        })
    }

    /// Copies slot `slot`'s rings, conv states and position into `state`, on the stream.
    pub fn save_local_state(&mut self, slot: usize, state: &mut LocalState) -> anyhow::Result<()> {
        self.with_slot(slot, |m| m.save_slot0_state(state))
    }

    fn save_slot0_state(&self, state: &mut LocalState) -> anyhow::Result<()> {
        let stream = &self.ctx.stream;
        ensure!(
            state.visits.len() == self.visits.len(),
            "the state has {} visits, the model {}",
            state.visits.len(),
            self.visits.len()
        );
        for (v, (ring, conv)) in self.visits.iter().zip(&mut state.visits) {
            if let Some(ring) = ring.as_mut() {
                ring.map(ring.slots(), stream)?;
            }
            match (v.kv.data(), ring.as_mut().map(KvCache::data_mut)) {
                (_, None) => ensure!(v.kv.window().is_none(), "the state lacks a ring"),
                (KvData::Bf16 { keys, values }, Some(KvData::Bf16 { keys: k, values: x })) => {
                    ensure!(k.len() == keys.len(), "the ring sizes differ");
                    stream.memcpy_dtod(keys, k)?;
                    stream.memcpy_dtod(values, x)?;
                }
                (KvData::Fp8 { keys, values }, Some(KvData::Fp8 { keys: k, values: x }))
                | (
                    KvData::Nvfp4 { keys, values },
                    Some(KvData::Nvfp4 { keys: k, values: x }),
                ) => {
                    ensure!(k.len() == keys.len(), "the ring sizes differ");
                    stream.memcpy_dtod(keys, k)?;
                    stream.memcpy_dtod(values, x)?;
                }
                _ => anyhow::bail!("the ring formats differ"),
            }
            for (src, dst) in v.conv.iter().zip(conv.iter_mut()) {
                stream.memcpy_dtod(src, dst)?;
            }
        }
        state.position = self.position;
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
        // A pooled ring of a slot that held no sequence takes its pages now.
        super::map_visits(&mut self.visits, 0, &stream)?;
        for (v, (ring, conv)) in self.visits.iter_mut().zip(&state.visits) {
            match (v.kv.data_mut(), ring.as_ref().map(KvCache::data)) {
                (_, None) => {}
                (KvData::Bf16 { keys, values }, Some(KvData::Bf16 { keys: k, values: x })) => {
                    stream.memcpy_dtod(k, keys)?;
                    stream.memcpy_dtod(x, values)?;
                }
                (KvData::Fp8 { keys, values }, Some(KvData::Fp8 { keys: k, values: x }))
                | (
                    KvData::Nvfp4 { keys, values },
                    Some(KvData::Nvfp4 { keys: k, values: x }),
                ) => {
                    stream.memcpy_dtod(k, keys)?;
                    stream.memcpy_dtod(x, values)?;
                }
                _ => anyhow::bail!("the ring formats differ"),
            }
            for (src, dst) in conv.iter().zip(v.conv.iter_mut()) {
                stream.memcpy_dtod(src, dst)?;
            }
        }
        self.position = state.position;
        Ok(())
    }
}
