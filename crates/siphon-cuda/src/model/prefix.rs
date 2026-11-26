//! Copies of the per-position state of a slot that later tokens overwrite, for the prefix
//! cache (`spec::PrefixCache`).
//!
//! Tokens after position `p` write global KV cache slots from `p` on, so the slots below
//! `p` stay as they were. The sliding rings and the conv states do not: later tokens
//! overwrite ring slots and shift the conv windows. A [`LocalState`] keeps copies of those
//! at one position, so that the sequence can return there while the global caches still
//! hold its positions below it.

use anyhow::ensure;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr as _};

use super::{Model, VisitState};
use crate::ops::attention::{HEAD_DIM, KV_HEADS, KvData};

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
/// of cache `c` takes the `position_bytes` bytes at `caches[c] + p * position_bytes`.
#[derive(Clone, Debug)]
pub struct GlobalKv {
    /// Each global cache's keys, then its values, by visit or layer.
    pub caches: Vec<u64>,
    /// Bytes of one position in one cache.
    pub position_bytes: usize,
    /// Positions each cache holds.
    pub capacity: usize,
}

impl GlobalKv {
    /// Bytes of one position over every cache.
    #[must_use]
    pub fn bytes_per_position(&self) -> usize {
        self.caches.len() * self.position_bytes
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
                KvData::Fp8 { keys, values } => {
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
    /// Each visit's ring, `None` for a global cache, and its four conv states.
    visits: Vec<(Option<KvData>, Vec<CudaSlice<f32>>)>,
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
            match ring {
                Some(KvData::Bf16 { keys, values }) => {
                    spans.push(DeviceSpan::of(keys, stream));
                    spans.push(DeviceSpan::of(values, stream));
                }
                Some(KvData::Fp8 { keys, values }) => {
                    spans.push(DeviceSpan::of(keys, stream));
                    spans.push(DeviceSpan::of(values, stream));
                }
                None => {}
            }
            spans.extend(conv.iter().map(|c| DeviceSpan::of(c, stream)));
        }
        spans
    }

    /// Device bytes the copies take.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.visits
            .iter()
            .map(|(ring, conv)| {
                let ring = match ring {
                    Some(KvData::Bf16 { keys, values }) => (keys.len() + values.len()) * 2,
                    Some(KvData::Fp8 { keys, values }) => keys.len() + values.len(),
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
        let (mut position_bytes, mut capacity) = (0, 0);
        for v in visits.iter().filter(|v| v.kv.window().is_none()) {
            let (keys, values, elem) = match v.kv.data() {
                KvData::Bf16 { keys, values } => (
                    DeviceSpan::of(keys, stream),
                    DeviceSpan::of(values, stream),
                    2,
                ),
                KvData::Fp8 { keys, values } => (
                    DeviceSpan::of(keys, stream),
                    DeviceSpan::of(values, stream),
                    1,
                ),
            };
            caches.extend([keys.ptr, values.ptr]);
            position_bytes = KV_HEADS * HEAD_DIM * elem;
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
                let ring = match (v.kv.window(), v.kv.data()) {
                    (None, _) => None,
                    (Some(_), KvData::Bf16 { keys, values }) => Some(KvData::Bf16 {
                        keys: stream.alloc_zeros(keys.len())?,
                        values: stream.alloc_zeros(values.len())?,
                    }),
                    (Some(_), KvData::Fp8 { keys, values }) => Some(KvData::Fp8 {
                        keys: stream.alloc_zeros(keys.len())?,
                        values: stream.alloc_zeros(values.len())?,
                    }),
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
            match (v.kv.data(), ring) {
                (_, None) => ensure!(v.kv.window().is_none(), "the state lacks a ring"),
                (KvData::Bf16 { keys, values }, Some(KvData::Bf16 { keys: k, values: x })) => {
                    ensure!(k.len() == keys.len(), "the ring sizes differ");
                    stream.memcpy_dtod(keys, k)?;
                    stream.memcpy_dtod(values, x)?;
                }
                (KvData::Fp8 { keys, values }, Some(KvData::Fp8 { keys: k, values: x })) => {
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
        let stream = &self.ctx.stream;
        for (v, (ring, conv)) in self.visits.iter_mut().zip(&state.visits) {
            match (v.kv.data_mut(), ring) {
                (_, None) => {}
                (KvData::Bf16 { keys, values }, Some(KvData::Bf16 { keys: k, values: x })) => {
                    stream.memcpy_dtod(k, keys)?;
                    stream.memcpy_dtod(x, values)?;
                }
                (KvData::Fp8 { keys, values }, Some(KvData::Fp8 { keys: k, values: x })) => {
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
