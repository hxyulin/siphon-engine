//! Copies of the per-position state of slot 0 that later tokens overwrite, for a prompt
//! cache (`Speculator::prefill_cached`).
//!
//! Tokens after position `p` write global KV cache slots from `p` on, so the slots below
//! `p` stay as they were. The sliding rings and the conv states do not: later tokens
//! overwrite ring slots and shift the conv windows. A [`LocalState`] keeps copies of those
//! at one position, so that the sequence can return there while the global caches still
//! hold its positions below it.

use anyhow::ensure;
use cudarc::driver::CudaSlice;

use super::Model;
use crate::ops::attention::KvData;

/// Slot 0's sliding rings, conv states and position at one point of a sequence, from
/// [`Model::save_local_state`].
pub struct LocalState {
    /// Each visit's ring, `None` for a global cache, and its four conv states.
    visits: Vec<(Option<KvData>, Vec<CudaSlice<f32>>)>,
    position: usize,
    /// [`Model::sequence`] when saved; `None` before the first save.
    sequence: Option<u64>,
}

impl LocalState {
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
    /// Allocates a [`LocalState`] for slot 0's rings and conv states, holding nothing yet.
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
            sequence: None,
        })
    }

    /// Copies slot 0's rings, conv states and position into `state`, on the stream.
    pub fn save_local_state(&self, state: &mut LocalState) -> anyhow::Result<()> {
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
        state.sequence = Some(self.sequence);
        Ok(())
    }

    /// Whether [`Model::restore_local_state`] can return slot 0 to `state`: slot 0 still
    /// holds the sequence `state` was saved from, at `state`'s position or later.
    #[must_use]
    pub fn can_restore_local_state(&self, state: &LocalState) -> bool {
        state.sequence == Some(self.sequence) && self.position >= state.position
    }

    /// Returns slot 0 to `state`: copies back its rings and conv states and moves to its
    /// position. The global caches keep the slots below that position, which the sequence
    /// wrote before `state` was saved and has not written since.
    pub fn restore_local_state(&mut self, state: &LocalState) -> anyhow::Result<()> {
        ensure!(
            self.can_restore_local_state(state),
            "slot 0 no longer holds the sequence of the saved state"
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
