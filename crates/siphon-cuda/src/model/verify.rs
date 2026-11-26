//! The verify pass of speculative decoding: the anchor and up to 7 drafts as one chunk of
//! up to [`VERIFY_ROWS`] rows through chunked prefill, then every row's token (greedy, or
//! drawn with the model's sampling), the acceptance and the rollback of rejected rows on
//! the device.
//!
//! Each width has its own CUDA graph, captured on first use. The tokens, the start
//! position and the acceptance mode live in device buffers, so a graph replays at any
//! position and with tokens the draft left on the device.
//!
//! The pass runs in [`PrefillScratch::invariant`] mode: each row's results are the same,
//! bit for bit, for every width and whatever the later rows hold. Greedy speculation then
//! emits exactly the tokens that verifying one row at a time would.
//!
//! State after accepting `a` drafts of a pass at position `P`: the position is
//! `P + a + 1`, the global caches keep their stale rows past it (attention never reads
//! them, and later rows overwrite them), the sliding rings get back the slots that
//! rejected rows overwrote, and the conv states are rebuilt from their backup and the
//! recorded inputs of rows `0..=a`.

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
    attention::KvData,
    sample::SampleBuffers,
    spec::{RESULT_LEN, STATE_ROWS, STREAM_DIMS, STREAMS, VERIFY_ROWS, VISIT_CHANNELS},
};

/// Where the tokens of a verify come from.
pub enum VerifyTokens<'a> {
    /// Host tokens, 1 to [`VERIFY_ROWS`] of them.
    Host(&'a [u32]),
    /// The current anchor ([`Model::verify_anchor`]) and up to 7 drafts on the device.
    Drafts(CudaView<'a, i32>),
}

/// Which rows a verify keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accept {
    /// The anchor and the longest run of drafts that each equal the previous row's token:
    /// the greedy one, or with sampling on, the row's draw ([`Model::enable_sampling`]).
    Greedy,
    /// The first `n` rows, whatever the tokens: teacher forcing, or a chosen rollback.
    Rows(usize),
}

/// The outcome of a verify.
#[derive(Clone, Debug)]
pub struct Verdict {
    /// Drafts kept; rows `0..=accepted` stay in the caches.
    pub accepted: usize,
    /// The verified tokens, the anchor first.
    pub tokens: Vec<u32>,
    /// Each row's token: the greedy one, or with sampling on, the row's draw.
    pub argmax: Vec<u32>,
}

impl Verdict {
    /// The row's token after the kept rows: the correction, or the bonus when every draft
    /// was kept. It is the next anchor.
    #[must_use]
    pub fn next(&self) -> u32 {
        self.argmax[self.accepted]
    }

    /// The tokens the round emits: the kept drafts and [`Self::next`].
    #[must_use]
    pub fn emitted(&self) -> Vec<u32> {
        let mut out = self.tokens[1..=self.accepted].to_vec();
        out.push(self.next());
        out
    }
}

/// A verify pass that [`Model::start_verify`] launched and whose verdict
/// [`Model::finish_verify`] has not read yet.
#[must_use = "the model's position advances only in finish_verify"]
#[derive(Debug)]
pub struct PendingVerify {
    rows: usize,
    start: usize,
}

/// What a verify pass keeps besides a chunk's intermediates (see `kernels/spec.cu`).
pub(super) struct VerifyBuffers {
    /// Device addresses of every visit's four conv states, in visit order.
    pub states: CudaSlice<u64>,
    /// The conv states before the pass, F32 `[visits, 3 * 6656]`.
    pub conv_backup: CudaSlice<f32>,
    /// Each visit's conv inputs of the pass's rows, BF16 `[visits, 8, 6656]`.
    pub inputs: CudaSlice<bf16>,
    /// Device addresses of the sliding visits' keys and values, a pair per visit.
    pub rings: CudaSlice<u64>,
    /// Slots of every sliding ring.
    pub ring_slots: usize,
    /// Bytes per slot of each sliding ring's key or value buffer (`2048` for BF16, `1024` for FP8).
    pub ring_row_bytes: usize,
    /// The ring slots the pass overwrites, `[sliding visits, 8, 2, ring_row_bytes]` bytes.
    pub ring_backup: CudaSlice<u8>,
    /// Each row's token, greedy or drawn.
    pub argmax: CudaSlice<i32>,
    /// Rows to keep whatever the drafts, or 0 for greedy acceptance.
    pub keep: CudaSlice<u32>,
    /// `[accepted, tokens, argmax]`, read back after the pass.
    pub result: CudaSlice<i32>,
    /// The next anchor, `argmax[accepted]`.
    pub anchor: CudaSlice<i32>,
    /// Whether a pass was launched and its verdict not read yet.
    pub pending: bool,
}

impl VerifyBuffers {
    fn new(stream: &Arc<CudaStream>, visits: &[VisitState]) -> anyhow::Result<Self> {
        let mut states = Vec::with_capacity(visits.len() * STREAMS);
        let mut rings = Vec::new();
        let mut ring_slots = None;
        let mut ring_row_bytes = None;
        for (i, v) in visits.iter().enumerate() {
            for (state, dims) in v.conv.iter().zip(STREAM_DIMS) {
                ensure!(
                    state.len() == STATE_ROWS * dims,
                    "visit {i}: a conv state has {} elements, expected {STATE_ROWS} x {dims}",
                    state.len()
                );
                states.push(state.device_ptr(stream).0);
            }
            if v.kv.window().is_some() {
                let (keys, values, row_bytes) = match v.kv.data() {
                    KvData::Bf16 { keys, values } => (
                        keys.device_ptr(stream).0,
                        values.device_ptr(stream).0,
                        1024 * 2,
                    ),
                    KvData::Fp8 { keys, values } => {
                        (keys.device_ptr(stream).0, values.device_ptr(stream).0, 1024)
                    }
                    KvData::Nvfp4 { .. } => {
                        anyhow::bail!("visit {i}: sliding caches are BF16 or FP8")
                    }
                };
                rings.push(keys);
                rings.push(values);
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
        let sliding = rings.len() / 2;
        ensure!(sliding > 0, "no sliding visits");
        let ring_row_bytes = ring_row_bytes.unwrap_or(1024 * 2);
        Ok(Self {
            states: stream.clone_htod(&states)?,
            conv_backup: zeros(stream, visits.len() * STATE_ROWS * VISIT_CHANNELS)?,
            inputs: zeros(stream, visits.len() * VERIFY_ROWS * VISIT_CHANNELS)?,
            rings: stream.clone_htod(&rings)?,
            ring_slots: ring_slots.unwrap_or(0),
            ring_row_bytes,
            ring_backup: zeros(stream, sliding * VERIFY_ROWS * 2 * ring_row_bytes)?,
            argmax: zeros(stream, VERIFY_ROWS)?,
            keep: zeros(stream, 1)?,
            result: zeros(stream, RESULT_LEN)?,
            anchor: zeros(stream, 1)?,
            pending: false,
        })
    }
}

impl Model {
    /// Runs the anchor and drafts in `tokens` at the current position as one verify pass of
    /// their count, 1 to [`VERIFY_ROWS`] rows, keeps the rows `accept` chooses and rolls back
    /// the rest. Advances the position past the kept rows. The next anchor stays on the
    /// device for [`VerifyTokens::Drafts`]. Synchronizes once, to read the verdict.
    pub fn verify(&mut self, tokens: VerifyTokens<'_>, accept: Accept) -> anyhow::Result<Verdict> {
        let pending = self.start_verify(tokens, accept)?;
        self.finish_verify(pending)
    }

    /// Launches the verify pass of [`Self::verify`] without waiting for it. Work enqueued on
    /// the model's stream before [`Self::finish_verify`] runs after the pass and before the
    /// verdict is read: [`Self::verify_taps`], [`Self::verify_kept`] and
    /// [`Self::verify_anchor`] hold the pass's results on the device. The position stays at
    /// the pass's start until then.
    pub fn start_verify(
        &mut self,
        tokens: VerifyTokens<'_>,
        accept: Accept,
    ) -> anyhow::Result<PendingVerify> {
        let m = match &tokens {
            VerifyTokens::Host(t) => t.len(),
            VerifyTokens::Drafts(d) => d.len() + 1,
        };
        ensure!(
            (1..=VERIFY_ROWS).contains(&m),
            "a verify takes 1 to {VERIFY_ROWS} rows, not {m}"
        );
        let start = self.position;
        let last = start + m - 1;
        ensure!(
            self.visits.iter().all(|v| v.kv.holds(last)),
            "the global KV caches cannot hold position {last}"
        );
        super::map_visits(&mut self.visits, last + 1, &Arc::clone(&self.ctx.stream))?;
        let keep = match accept {
            Accept::Greedy => 0,
            Accept::Rows(n) => {
                ensure!((1..=m).contains(&n), "cannot keep {n} of {m} rows");
                u32::try_from(n)?
            }
        };
        self.ensure_verify()?;
        let stream = Arc::clone(&self.ctx.stream);
        let p = self
            .verify
            .as_mut()
            .context("the verify buffers are missing")?;
        let v = p
            .verify
            .as_mut()
            .context("the verify buffers are missing")?;
        ensure!(!v.pending, "the last verify's verdict was not read");
        match tokens {
            VerifyTokens::Host(t) => {
                let ids = t
                    .iter()
                    .map(|&t| Ok(i32::try_from(t)?))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                stream.memcpy_htod(&ids, &mut p.tokens.slice_mut(..m))?;
            }
            VerifyTokens::Drafts(d) => {
                stream.memcpy_dtod(&v.anchor, &mut p.tokens.slice_mut(..1))?;
                if m > 1 {
                    stream.memcpy_dtod(&d, &mut p.tokens.slice_mut(1..m))?;
                }
            }
        }
        stream.memcpy_htod(
            &[u32::try_from(start).context("position does not fit the kernels")?],
            &mut p.start,
        )?;
        stream.memcpy_htod(&[keep], &mut v.keep)?;
        if let Some(sample) = p.sample.as_mut() {
            // Row j picks the token at position start + j + 1.
            let settings: Vec<(Sampling, usize)> =
                (0..m).map(|j| (self.sampling, start + j + 1)).collect();
            sample.upload(&stream, &settings)?;
        }
        self.launch_verify(m, start)?;
        self.verify_rows = m;
        self.verify_buffers_mut()?.pending = true;
        Ok(PendingVerify { rows: m, start })
    }

    /// Waits for the pass of `pending` and the work enqueued after it, reads the verdict
    /// and advances the position past the kept rows.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "taking the pass by value reads each verdict once"
    )]
    pub fn finish_verify(&mut self, pending: PendingVerify) -> anyhow::Result<Verdict> {
        let PendingVerify { rows: m, start } = pending;
        ensure!(
            self.position == start && self.verify_rows == m,
            "the verdict belongs to another pass"
        );
        let stream = Arc::clone(&self.ctx.stream);
        let v = self.verify_buffers_mut()?;
        v.pending = false;
        let result = stream.clone_dtoh(&v.result)?;
        let accepted = usize::try_from(result[0])?;
        ensure!(accepted < m, "accepted {accepted} of {} drafts", m - 1);
        let ids = |r: &[i32]| -> anyhow::Result<Vec<u32>> {
            r.iter()
                .map(|&t| u32::try_from(t).context("a negative token id"))
                .collect()
        };
        self.position = start + accepted + 1;
        Ok(Verdict {
            accepted,
            tokens: ids(&result[1..=m])?,
            argmax: ids(&result[1 + VERIFY_ROWS..1 + VERIFY_ROWS + m])?,
        })
    }

    /// Sets the anchor that the next [`VerifyTokens::Drafts`] verify starts with, after a
    /// prefill. A verify leaves its own next anchor there.
    pub fn set_verify_anchor(&mut self, token: u32) -> anyhow::Result<()> {
        self.ensure_verify()?;
        let stream = Arc::clone(&self.ctx.stream);
        let v = self
            .verify
            .as_mut()
            .and_then(|p| p.verify.as_mut())
            .context("the verify buffers are missing")?;
        stream.memcpy_htod(&[i32::try_from(token)?], &mut v.anchor)?;
        Ok(())
    }

    /// The number of drafts the last verify kept, on the device, one I32.
    pub fn verify_kept(&self) -> anyhow::Result<CudaView<'_, i32>> {
        self.verify
            .as_ref()
            .and_then(|p| p.verify.as_ref())
            .map(|v| v.result.slice(..1))
            .context("no verify has run")
    }

    fn verify_buffers_mut(&mut self) -> anyhow::Result<&mut VerifyBuffers> {
        self.verify
            .as_mut()
            .and_then(|p| p.verify.as_mut())
            .context("the verify buffers are missing")
    }

    /// The anchor of the next round on the device, one I32.
    pub fn verify_anchor(&self) -> anyhow::Result<&CudaSlice<i32>> {
        self.verify
            .as_ref()
            .and_then(|p| p.verify.as_ref())
            .map(|v| &v.anchor)
            .context("no verify has run")
    }

    /// The F32 logits after the softcap of the last verify's rows, `[rows, unpadded_vocab]`.
    pub fn verify_logits(&self) -> anyhow::Result<Vec<f32>> {
        let p = self.verify.as_ref().context("no verify has run")?;
        let n = self.verify_rows * self.ctx.config.unpadded_vocab_size;
        Ok(self.ctx.stream.clone_dtoh(&p.head_capped.slice(..n))?)
    }

    /// The feature taps of the last verify's rows, BF16 `[rows, taps, 2304]` in the order
    /// of [`Model::tap_layout`], for the draft.
    pub fn verify_taps(&self) -> anyhow::Result<CudaView<'_, bf16>> {
        let p = self.verify.as_ref().context("no verify has run")?;
        let taps = p.taps.as_ref().context("the taps buffer is missing")?;
        let row_len = p.tap_layout.len() * self.ctx.hidden();
        Ok(taps.slice(..self.verify_rows * row_len))
    }

    fn ensure_verify(&mut self) -> anyhow::Result<()> {
        if self.verify.is_some() {
            return Ok(());
        }
        let stream = &self.ctx.stream;
        let mut p =
            PrefillScratch::new(stream, &self.ctx.config, self.max_blocks, VERIFY_ROWS, true)?;
        p.record_taps = true;
        p.tap_layout = self.tap_layout.clone();
        p.taps = Some(zeros(
            stream,
            VERIFY_ROWS * p.tap_layout.len() * self.ctx.hidden(),
        )?);
        p.verify = Some(VerifyBuffers::new(stream, &self.visits)?);
        if self.sampler {
            p.sample = Some(SampleBuffers::new(stream, VERIFY_ROWS)?);
        }
        self.verify = Some(p);
        Ok(())
    }

    /// Replays the verify graph of `m` rows, capturing it first if needed.
    fn launch_verify(&mut self, m: usize, start: usize) -> anyhow::Result<()> {
        if self.verify_graphs[m - 1].is_none() {
            let stream = Arc::clone(&self.ctx.stream);
            stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let enqueued = self.enqueue_verify(m, start);
            // End the capture even when enqueueing failed, so the stream is usable again.
            let graph = stream.end_capture(CUgraphInstantiate_flags(0));
            enqueued.context("capturing the verify pass")?;
            let graph = graph?.context("the verify capture recorded no work")?;
            graph.upload()?;
            self.verify_graphs[m - 1] = Some(StepGraph(graph));
        }
        let graph = self.verify_graphs[m - 1]
            .as_ref()
            .context("the verify graph is missing")?;
        graph.0.launch()?;
        Ok(())
    }

    fn enqueue_verify(&mut self, m: usize, start: usize) -> anyhow::Result<()> {
        let Self {
            ctx,
            passes,
            visits,
            scratch,
            max_blocks,
            verify,
            ..
        } = self;
        let p = verify.as_mut().context("the verify buffers are missing")?;
        ctx.prefill_enqueue(
            p,
            visits,
            *passes,
            *max_blocks,
            scratch,
            m,
            start,
            &mut PrefillLogits::Last,
            false,
            &[],
        )
    }
}
