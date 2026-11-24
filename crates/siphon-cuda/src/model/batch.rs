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
    CudaSlice, CudaStream, CudaView, DevicePtr as _,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
};
use half::bf16;

use super::{
    Model, PrefillLogits, Sampling, StepGraph, VisitState, prefill::PrefillScratch, zeros,
};
use crate::ops::{
    attention::{HEAD_DIM, KV_HEADS, KvCache, KvData, KvFormat},
    sample::SampleBuffers,
    spec::{BATCH_STATE_WORDS, BatchRows, VERIFY_ROWS},
};

/// Most sequences one batched step runs.
pub const BATCH_ROWS: usize = VERIFY_ROWS;

/// A sequence slot besides slot 0: its caches and position.
#[derive(Debug)]
pub(super) struct Slot {
    visits: Vec<VisitState>,
    position: usize,
    /// [`Model::sequence`] while the slot stands in for slot 0 ([`Model::with_slot`]).
    sequence: u64,
}

/// What a batched step keeps besides a chunk's intermediates.
pub(super) struct BatchBuffers {
    /// Each visit's state table, `[visits, BATCH_ROWS, BATCH_STATE_WORDS]` device addresses
    /// (see [`BATCH_STATE_WORDS`]).
    tables: CudaSlice<u64>,
    /// Each row's position.
    positions: CudaSlice<u32>,
    /// Each row's token, greedy or drawn.
    argmax: CudaSlice<i32>,
    /// The slots whose addresses `tables` holds, by row.
    members: Vec<usize>,
}

impl BatchBuffers {
    fn new(stream: &Arc<CudaStream>, visits: usize) -> anyhow::Result<Self> {
        Ok(Self {
            tables: zeros(stream, visits * BATCH_ROWS * BATCH_STATE_WORDS)?,
            positions: zeros(stream, BATCH_ROWS)?,
            argmax: zeros(stream, BATCH_ROWS)?,
            members: Vec::new(),
        })
    }

    /// The rows' states in visit `visit`, for a step of `rows` rows.
    pub(super) fn rows(&self, visit: usize, rows: usize) -> BatchRows<'_> {
        let at = visit * BATCH_ROWS * BATCH_STATE_WORDS;
        BatchRows {
            states: self.tables.slice(at..at + rows * BATCH_STATE_WORDS),
            positions: self.positions.slice(..rows),
        }
    }

    /// Each row's token, greedy or drawn, `[BATCH_ROWS]`.
    pub(super) fn argmax_mut(&mut self) -> &mut CudaSlice<i32> {
        &mut self.argmax
    }
}

/// A visit's caches and conv states for a new sequence, of the sizes and formats of `v`.
fn empty_like(stream: &Arc<CudaStream>, v: &VisitState) -> anyhow::Result<VisitState> {
    let kv = match (v.kv.window(), v.kv.data()) {
        (Some(window), _) => KvCache::sliding_ring(stream, window, v.kv.slots())?,
        (None, KvData::Bf16 { .. }) => {
            KvCache::global_with_format(stream, v.kv.slots(), KvFormat::Bf16)?
        }
        (None, KvData::Fp8 { .. }) => {
            KvCache::global_with_format(stream, v.kv.slots(), KvFormat::Fp8)?
        }
    };
    Ok(VisitState {
        kv,
        conv: [
            stream.alloc_zeros(v.conv[0].len())?,
            stream.alloc_zeros(v.conv[1].len())?,
            stream.alloc_zeros(v.conv[2].len())?,
            stream.alloc_zeros(v.conv[3].len())?,
        ],
    })
}

/// A visit's line of a state table: its caches' and conv states' device addresses.
fn state_words(stream: &CudaStream, v: &VisitState) -> [u64; BATCH_STATE_WORDS] {
    let (keys, values) = match v.kv.data() {
        KvData::Bf16 { keys, values } => (keys.device_ptr(stream).0, values.device_ptr(stream).0),
        KvData::Fp8 { keys, values } => (keys.device_ptr(stream).0, values.device_ptr(stream).0),
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
            let visits = self
                .visits
                .iter()
                .map(|v| empty_like(&stream, v))
                .collect::<anyhow::Result<_>>()?;
            self.slots.push(Slot {
                visits,
                position: 0,
                sequence: 0,
            });
        }
        Ok(())
    }

    /// Sequence slots, slot 0 included.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots.len() + 1
    }

    /// Position of the next token of slot `slot`.
    pub fn slot_position(&self, slot: usize) -> anyhow::Result<usize> {
        if slot == 0 {
            return Ok(self.position);
        }
        Ok(self.slot(slot)?.position)
    }

    fn slot(&self, slot: usize) -> anyhow::Result<&Slot> {
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
        self.slot(slot)?;
        let swap = |m: &mut Self| {
            let s = &mut m.slots[slot - 1];
            std::mem::swap(&mut m.visits, &mut s.visits);
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
            // A ring whole, a global cache up to the position.
            let slots = if s.kv.window().is_some() {
                s.kv.slots()
            } else {
                position.min(s.kv.slots())
            };
            let len = slots * KV_HEADS * HEAD_DIM;
            match (s.kv.data(), d.kv.data_mut()) {
                (KvData::Bf16 { keys, values }, KvData::Bf16 { keys: k, values: v }) => {
                    stream.memcpy_dtod(&keys.slice(..len), &mut k.slice_mut(..len))?;
                    stream.memcpy_dtod(&values.slice(..len), &mut v.slice_mut(..len))?;
                }
                (KvData::Fp8 { keys, values }, KvData::Fp8 { keys: k, values: v }) => {
                    stream.memcpy_dtod(&keys.slice(..len), &mut k.slice_mut(..len))?;
                    stream.memcpy_dtod(&values.slice(..len), &mut v.slice_mut(..len))?;
                }
                _ => anyhow::bail!("the slots' cache formats differ"),
            }
            for (from, to) in s.conv.iter().zip(d.conv.iter_mut()) {
                stream.memcpy_dtod(from, to)?;
            }
        }
        if to == 0 {
            self.position = position;
            self.sequence += 1;
        } else {
            self.slots[to - 1].position = position;
        }
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
        ensure!(row < BATCH_ROWS, "a batch has at most {BATCH_ROWS} rows");
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
            positions.push(u32::try_from(position).context("position does not fit the kernels")?);
            tokens.push(i32::try_from(token)?);
        }
        self.ensure_batch()?;
        let stream = Arc::clone(&self.ctx.stream);
        let members: Vec<usize> = rows.iter().map(|&(slot, _)| slot).collect();
        let p = self
            .batch
            .as_mut()
            .context("the batch buffers are missing")?;
        let b = p.batch.as_mut().context("the batch buffers are missing")?;
        if b.members != members {
            // Every visit's table, BATCH_ROWS lines each.
            let mut table = vec![0u64; self.visits.len() * BATCH_ROWS * BATCH_STATE_WORDS];
            for (r, &slot) in members.iter().enumerate() {
                let visits = if slot == 0 {
                    &self.visits
                } else {
                    &self.slots[slot - 1].visits
                };
                for (i, v) in visits.iter().enumerate() {
                    let at = (i * BATCH_ROWS + r) * BATCH_STATE_WORDS;
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
        ensure!(rows <= BATCH_ROWS, "a batch has at most {BATCH_ROWS} rows");
        let n = rows * self.ctx.config.unpadded_vocab_size;
        Ok(self.ctx.stream.clone_dtoh(&p.head_capped.slice(..n))?)
    }

    fn ensure_batch(&mut self) -> anyhow::Result<()> {
        if self.batch.is_some() {
            return Ok(());
        }
        let stream = &self.ctx.stream;
        let mut p =
            PrefillScratch::new(stream, &self.ctx.config, self.max_blocks, BATCH_ROWS, true)?;
        if self.batch_taps {
            p.record_taps = true;
            p.tap_layout = self.tap_layout.clone();
            p.taps = Some(zeros(
                stream,
                BATCH_ROWS * p.tap_layout.len() * self.ctx.hidden(),
            )?);
        }
        p.batch = Some(BatchBuffers::new(stream, self.visits.len())?);
        if self.sampler {
            p.sample = Some(SampleBuffers::new(stream, BATCH_ROWS)?);
        }
        self.batch = Some(p);
        Ok(())
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
        )
    }
}
