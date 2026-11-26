//! Batched DFlash 2 drafting: runs 1 to [`MAX_BATCH_SEQS`] (`32`) independent DFlash 2
//! blocks (`8 * B <= 256` rows) in one CUDA-graph-captured pass on FP8/BF16 tensor cores
//! (`LinearRows`), with each sequence slot owning its own 5 sliding-window KV rings.

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaEvent, CudaGraph, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _,
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
};
use half::bf16;

use super::{Arch, DFLASH2_SLOTS, Draft, DraftRings, EPS, Matrix, STEPS};
use crate::{
    DeviceIn,
    gpu::{PinnedBuf, sync_event},
    model::Sampling,
    ops::{
        draft::{
            BLOCK, BatchAttentionScratch, CONTROL, DFLASH2_MAX_SPLITS, Fp8Rows, KV_WIDTH, KvCache,
            MAX_BATCH_SEQS, RACE_WORDS, TOP_K, TopkScratch, race_words,
        },
        linear::Start,
        linear_rows::{LinearRows, MAX_PRODUCTS as ROWS_MAX_PRODUCTS, RowsProduct},
        spec::RESULT_LEN,
    },
    weights::{
        dflash2::{CONV_COEFFICIENTS, ConvWeights, DFLASH2_HEADS, DFLASH2_WINDOW},
        draft::{HEAD_DIM, HIDDEN, RANK, SharedWeights},
    },
};

/// Maximum sequences one batched DFlash 2 round drafts.
pub const MAX_BATCH_DRAFT_SEQS: usize = MAX_BATCH_SEQS;

/// Captured batched DFlash 2 round and the external device addresses baked into it.
/// Per-round varying metadata (`controls`, `starts`, `races`, `cache_table`) lives inside
/// [`BatchDflash2`] at stable addresses; external inputs (`shared` weights and, for
/// `commit_width > 0`, `Model::batch_verify_*` buffers) are checked on every launch so any
/// reallocation or alternate caller buffer transparently recaptures.
struct BatchRoundGraph {
    graph: CudaGraph,
    lm_head: u64,
    embed: u64,
    embed_norm: u64,
    taps: u64,
    results: u64,
    anchors: u64,
}

// SAFETY: a graph exec may be launched from any thread with its context bound, which
// `CudaGraph::launch` does, and the draft launches it only through `&mut Draft`.
unsafe impl Send for BatchRoundGraph {}

/// External device buffers from `Model::start_verify_batch` consumed inside a captured
/// `batch_commit_and_draft` graph.
struct BatchCommitInputs<'a> {
    taps: CudaView<'a, bf16>,
    results: CudaView<'a, i32>,
    anchors: CudaView<'a, i32>,
    width: usize,
}

/// Ensures no slot appears twice in one batched DFlash 2 call (two active sequences cannot
/// share one KV ring).
fn check_distinct_slots(slots: &[usize]) -> anyhow::Result<()> {
    for (i, &slot) in slots.iter().enumerate() {
        ensure!(
            !slots[..i].contains(&slot),
            "duplicate slot {slot} in batched DFlash 2 call"
        );
    }
    Ok(())
}

/// Per-slot sliding rings and batched intermediates for DFlash 2.
pub(super) struct BatchDflash2 {
    /// Sliding rings for slots `1..` (slot 0 reuses `Draft::caches`).
    extra_slots: Vec<Vec<KvCache>>,
    /// Committed position of slots `1..` (slot 0 uses `Draft::committed`).
    extra_committed: Vec<usize>,
    /// Whether the rings use FP8 E4M3 storage.
    fp8_kv: bool,
    /// Maximum sequences the scratch buffers hold (`<= MAX_BATCH_SEQS`).
    max_seqs: usize,
    /// Elements per row of `base` logits (`capacity` with shortlist, else `vocab`).
    base_ld: usize,
    /// Per-sequence `[start, commit_rows, anchor]` controls, `[max_seqs * 3]`.
    controls: CudaSlice<i32>,
    /// Per-sequence start positions for `set_control_kept_batch`, `[max_seqs]`.
    starts: CudaSlice<i32>,
    /// Per-sequence sampling race descriptors, `[max_seqs * 4]`.
    races: CudaSlice<u32>,
    /// Host mirror of `races` so unchanged sampling descriptors skip H2D uploads.
    last_races: Vec<u32>,
    /// Active slot list whose ring pointers `cache_table` holds.
    members: Vec<usize>,
    /// `[max_seqs, layers, 2]` device addresses `(cache_k, cache_v)`.
    cache_table: CudaSlice<u64>,
    /// Persistent page-locked staging buffers for async H2D metadata uploads.
    host_controls: PinnedBuf<i32>,
    host_starts: PinnedBuf<i32>,
    host_races: PinnedBuf<u32>,
    host_cache_table: PinnedBuf<u64>,
    /// Recorded on the stream after each async pinned H2D upload.
    h2d_done: CudaEvent,
    /// Commit intermediates.
    ctx_raw: CudaSlice<bf16>,
    ctx_normed: CudaSlice<bf16>,
    ctx_k: CudaSlice<bf16>,
    ctx_v: CudaSlice<bf16>,
    /// Block intermediates (`[max_seqs * BLOCK, ...]`).
    residual: CudaSlice<bf16>,
    h: CudaSlice<bf16>,
    coef_attention: CudaSlice<bf16>,
    coef_mlp: CudaSlice<bf16>,
    conv: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    o: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    up: CudaSlice<bf16>,
    act: CudaSlice<bf16>,
    mlp: CudaSlice<bf16>,
    y: CudaSlice<bf16>,
    attention: BatchAttentionScratch,
    /// Head & selector intermediates (`[max_seqs * STEPS, ...]`).
    y_steps: CudaSlice<bf16>,
    base: CudaSlice<bf16>,
    topk: TopkScratch,
    cand: CudaSlice<i32>,
    unary: CudaSlice<f32>,
    hp: CudaSlice<bf16>,
    ids: CudaSlice<i32>,
    confidence: CudaSlice<f32>,
    /// Ready-to-verify tokens `[max_seqs * BLOCK]` (`[anchor, d0..d_{steps-1}]` per seq).
    verify_tokens: CudaSlice<i32>,
    /// Captured graphs keyed by `(seqs, commit_width, draft_steps)`.
    graphs: HashMap<(usize, usize, usize), BatchRoundGraph>,
}

impl BatchDflash2 {
    #[expect(
        clippy::too_many_arguments,
        reason = "sizes batched DFlash 2 scratch and adaptive attention buffers"
    )]
    fn new(
        stream: &Arc<CudaStream>,
        layers: usize,
        q_width: usize,
        mlp_width: usize,
        base_ld: usize,
        fp8_kv: bool,
        max_seqs: usize,
        attention_batch_slots: usize,
    ) -> anyhow::Result<Self> {
        let max_rows = max_seqs * BLOCK;
        let max_steps = max_seqs * STEPS;
        let z = |n: usize| stream.alloc_zeros::<bf16>(n);
        let zf = |n: usize| stream.alloc_zeros::<f32>(n);
        let zi = |n: usize| stream.alloc_zeros::<i32>(n);
        Ok(Self {
            extra_slots: Vec::new(),
            extra_committed: Vec::new(),
            fp8_kv,
            max_seqs,
            base_ld,
            controls: zi(max_seqs * CONTROL)?,
            starts: zi(max_seqs)?,
            races: stream.alloc_zeros(max_seqs * RACE_WORDS)?,
            last_races: vec![0; max_seqs * RACE_WORDS],
            members: Vec::new(),
            cache_table: stream.alloc_zeros(max_seqs * layers * 2)?,
            host_controls: PinnedBuf::zeros(stream, max_seqs * CONTROL)?,
            host_starts: PinnedBuf::zeros(stream, max_seqs)?,
            host_races: PinnedBuf::zeros(stream, max_seqs * RACE_WORDS)?,
            host_cache_table: PinnedBuf::zeros(stream, max_seqs * layers * 2)?,
            h2d_done: sync_event(stream)?,
            ctx_raw: z(max_rows * HIDDEN)?,
            ctx_normed: z(max_rows * HIDDEN)?,
            ctx_k: z(layers * max_rows * KV_WIDTH)?,
            ctx_v: z(layers * max_rows * KV_WIDTH)?,
            residual: z(max_rows * HIDDEN)?,
            h: z(max_rows * HIDDEN)?,
            coef_attention: z(max_rows * CONV_COEFFICIENTS)?,
            coef_mlp: z(max_rows * CONV_COEFFICIENTS)?,
            conv: z(max_rows * HIDDEN)?,
            q: z(max_rows * q_width)?,
            k: z(max_rows * KV_WIDTH)?,
            v: z(max_rows * KV_WIDTH)?,
            attn: z(max_rows * q_width)?,
            o: z(max_rows * HIDDEN)?,
            gate: z(max_rows * mlp_width)?,
            up: z(max_rows * mlp_width)?,
            act: z(max_rows * mlp_width)?,
            mlp: z(max_rows * HIDDEN)?,
            y: z(max_rows * HIDDEN)?,
            attention: BatchAttentionScratch::for_adaptive(
                stream,
                attention_batch_slots,
                max_seqs,
                DFLASH2_SLOTS,
                DFLASH2_MAX_SPLITS,
            )?,
            y_steps: z(max_steps * HIDDEN)?,
            base: z(max_steps * base_ld)?,
            topk: TopkScratch::new_for_rows(stream, max_steps)?,
            cand: zi(max_steps * TOP_K)?,
            unary: zf(max_steps * TOP_K)?,
            hp: z(max_steps * RANK)?,
            ids: zi(max_steps)?,
            confidence: zf(max_steps)?,
            verify_tokens: zi(max_rows)?,
            graphs: HashMap::new(),
        })
    }

    pub(super) fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    fn ensure_slot(&mut self, stream: &Arc<CudaStream>, layers: usize, slot: usize) -> anyhow::Result<()> {
        while self.extra_slots.len() < slot {
            let caches = (0..layers)
                .map(|_| KvCache::with_format(stream, DFLASH2_SLOTS, self.fp8_kv))
                .collect::<anyhow::Result<Vec<_>>>()?;
            self.extra_slots.push(caches);
            self.extra_committed.push(0);
        }
        Ok(())
    }
}

impl Drop for BatchDflash2 {
    fn drop(&mut self) {
        let _ = self.h2d_done.synchronize();
    }
}

/// One product entry for [`run_rows_products`]: optional BF16 weight, optional FP8 raw
/// weight, and output slice.
type BatchProduct<'a> = (
    Option<&'a CudaSlice<bf16>>,
    Option<&'a Fp8Rows>,
    CudaViewMut<'a, bf16>,
);

/// Runs one or more `out = x @ W^T` products through `LinearRows`, using FP8 raw weights
/// when every product supplies one and BF16 otherwise.
fn run_rows_products(
    linear_rows: &LinearRows,
    stream: &CudaStream,
    x: &CudaView<'_, bf16>,
    m: usize,
    products: &mut [BatchProduct<'_>],
) -> anyhow::Result<()> {
    let use_fp8 = products.iter().all(|p| p.1.is_some());
    for chunk in products.chunks_mut(ROWS_MAX_PRODUCTS) {
        if use_fp8 {
            let mut batch: Vec<RowsProduct<'_>> = chunk
                .iter_mut()
                .map(|(_, fp8, out)| {
                    let w = fp8.context("FP8 weight missing")?;
                    Ok(RowsProduct::fp8_raw(
                        w.values.slice(..),
                        w.scales.slice(..),
                        w.rows(),
                        w.k,
                        x,
                        out,
                    ))
                })
                .collect::<anyhow::Result<_>>()?;
            linear_rows.run(stream, &mut batch, m, Start::Early)?;
        } else {
            let mut batch: Vec<RowsProduct<'_>> = chunk
                .iter_mut()
                .map(|(bf, _, out)| {
                    let w = bf.context("BF16 weight was freed")?;
                    Ok(RowsProduct::bf16(w, x, out))
                })
                .collect::<anyhow::Result<_>>()?;
            linear_rows.run(stream, &mut batch, m, Start::Early)?;
        }
    }
    Ok(())
}

impl Draft {
    /// Whether this draft is a DFlash 2 drafter that supports batched speculation.
    #[must_use]
    pub fn is_dflash2(&self) -> bool {
        matches!(self.arch, Arch::Dflash2 { .. })
    }

    /// Ensures batched DFlash 2 buffers exist for at least `max_seqs` concurrent sequences
    /// and per-slot sliding rings exist up to `max_slot` (inclusive).
    pub fn ensure_batch_dflash2(
        &mut self,
        stream: &Arc<CudaStream>,
        max_seqs: usize,
        max_slot: usize,
    ) -> anyhow::Result<()> {
        ensure!(self.is_dflash2(), "batched drafting needs a DFlash 2 drafter");
        let max_seqs = max_seqs.clamp(1, MAX_BATCH_SEQS);
        let layers = self.arch.layers();
        let base_ld = self
            .head
            .as_ref()
            .and_then(|h| h.shortlist.as_ref())
            .map_or(self.vocab, super::Shortlist::capacity);
        let fp8_kv = self.caches.first().is_some_and(|c| c.fp8);
        let need_alloc = self
            .batch
            .as_ref()
            .is_none_or(|b| b.max_seqs < max_seqs || b.base_ld < base_ld);
        if need_alloc {
            let prev_extra = self.batch.take();
            let mut b = BatchDflash2::new(
                stream,
                layers,
                self.arch.q_width(),
                self.arch.mlp_width(),
                base_ld,
                fp8_kv,
                max_seqs,
                self.kernels.attention_batch_slots(),
            )?;
            if let Some(mut prev) = prev_extra {
                b.extra_slots = std::mem::take(&mut prev.extra_slots);
                b.extra_committed = std::mem::take(&mut prev.extra_committed);
            }
            self.batch = Some(b);
        }
        if let Some(b) = self.batch.as_mut() {
            b.ensure_slot(stream, layers, max_slot)?;
        }
        Ok(())
    }

    /// Committed position of slot `slot` in this DFlash 2 drafter.
    #[must_use]
    pub fn slot_committed(&self, slot: usize) -> usize {
        if slot == 0 {
            self.committed
        } else {
            self.batch
                .as_ref()
                .and_then(|b| b.extra_committed.get(slot - 1))
                .copied()
                .unwrap_or(0)
        }
    }

    /// Updates slot `slot`'s recorded committed position after a batched verify finishes.
    pub fn set_slot_committed(&mut self, slot: usize, pos: usize) -> anyhow::Result<()> {
        if slot == 0 {
            self.committed = pos;
        } else {
            let b = self
                .batch
                .as_mut()
                .context("batched DFlash 2 buffers are missing")?;
            let entry = b
                .extra_committed
                .get_mut(slot - 1)
                .with_context(|| format!("no DFlash 2 rings for slot {slot}"))?;
            *entry = pos;
        }
        Ok(())
    }

    /// Resets slot `slot`'s DFlash 2 committed position to 0.
    pub fn reset_slot(&mut self, slot: usize) {
        if slot == 0 {
            self.committed = 0;
        } else if let Some(b) = self.batch.as_mut()
            && let Some(c) = b.extra_committed.get_mut(slot - 1)
        {
            *c = 0;
        }
    }

    /// Commits target feature taps into slot `slot`'s DFlash 2 sliding rings, starting at
    /// position `start`.
    pub fn commit_slot(
        &mut self,
        stream: &Arc<CudaStream>,
        linear: &crate::ops::linear::Linear,
        slot: usize,
        taps: &impl DeviceIn<bf16>,
        start: usize,
    ) -> anyhow::Result<()> {
        if slot == 0 {
            return self.commit(stream, linear, taps, start);
        }
        self.ensure_batch_dflash2(stream, 1, slot)?;
        // Temporarily swap slot `slot`'s rings and committed count into slot 0 to reuse
        // `commit_dflash` directly.
        {
            let b = self.batch.as_mut().context("batch DFlash 2 missing")?;
            std::mem::swap(&mut self.caches, &mut b.extra_slots[slot - 1]);
            std::mem::swap(&mut self.committed, &mut b.extra_committed[slot - 1]);
        }
        let res = self.commit(stream, linear, taps, start);
        {
            let b = self.batch.as_mut().context("batch DFlash 2 missing")?;
            std::mem::swap(&mut self.caches, &mut b.extra_slots[slot - 1]);
            std::mem::swap(&mut self.committed, &mut b.extra_committed[slot - 1]);
        }
        res
    }

    /// Copies slot `from`'s DFlash 2 sliding rings and committed position into slot `to`
    /// (for shared prefills).
    pub fn copy_slot_rings(
        &mut self,
        stream: &Arc<CudaStream>,
        from: usize,
        to: usize,
    ) -> anyhow::Result<()> {
        if from == to {
            return Ok(());
        }
        self.ensure_batch_dflash2(stream, 1, from.max(to))?;
        let b = self.batch.as_mut().context("batch DFlash 2 missing")?;
        let layers = self.caches.len();
        let copy_pair = |src: &[KvCache], dst: &mut [KvCache]| -> anyhow::Result<()> {
            for l in 0..layers {
                stream.memcpy_dtod(&*src[l].k, &mut *dst[l].k)?;
                stream.memcpy_dtod(&*src[l].v, &mut *dst[l].v)?;
            }
            Ok(())
        };
        match (from, to) {
            (0, t) => {
                copy_pair(&self.caches, &mut b.extra_slots[t - 1])?;
                b.extra_committed[t - 1] = self.committed;
            }
            (f, 0) => {
                copy_pair(&b.extra_slots[f - 1], &mut self.caches)?;
                self.committed = b.extra_committed[f - 1];
            }
            (f, t) if f < t => {
                let (low, high) = b.extra_slots.split_at_mut(t - 1);
                copy_pair(&low[f - 1], &mut high[0])?;
                b.extra_committed[t - 1] = b.extra_committed[f - 1];
            }
            (f, t) => {
                let (low, high) = b.extra_slots.split_at_mut(f - 1);
                copy_pair(&high[0], &mut low[t - 1])?;
                b.extra_committed[t - 1] = b.extra_committed[f - 1];
            }
        }
        Ok(())
    }

    /// Copies slot `slot`'s DFlash 2 sliding rings and committed position into `rings`.
    pub fn save_slot_rings(
        &self,
        stream: &Arc<CudaStream>,
        slot: usize,
        rings: &mut DraftRings,
    ) -> anyhow::Result<()> {
        if slot == 0 {
            return self.save_rings(stream, rings);
        }
        let b = self
            .batch
            .as_ref()
            .context("batched DFlash 2 buffers are missing")?;
        let caches = b
            .extra_slots
            .get(slot - 1)
            .with_context(|| format!("no DFlash 2 rings for slot {slot}"))?;
        ensure!(
            rings.caches.len() == caches.len(),
            "the copies have {} layers, the draft {}",
            rings.caches.len(),
            caches.len()
        );
        for (cache, copy) in caches.iter().zip(&mut rings.caches) {
            if let Some(copy) = copy {
                stream.memcpy_dtod(&*cache.k, &mut *copy.k)?;
                stream.memcpy_dtod(&*cache.v, &mut *copy.v)?;
            }
        }
        rings.committed = b.extra_committed[slot - 1];
        Ok(())
    }

    /// Restores `rings` into slot `slot`'s DFlash 2 sliding rings and committed position.
    pub fn restore_slot_rings(
        &mut self,
        stream: &Arc<CudaStream>,
        slot: usize,
        rings: &DraftRings,
    ) -> anyhow::Result<()> {
        if slot == 0 {
            return self.restore_rings(stream, rings);
        }
        self.ensure_batch_dflash2(stream, 1, slot)?;
        let b = self
            .batch
            .as_mut()
            .context("batched DFlash 2 buffers are missing")?;
        let caches = b
            .extra_slots
            .get_mut(slot - 1)
            .with_context(|| format!("no DFlash 2 rings for slot {slot}"))?;
        ensure!(
            rings.caches.len() == caches.len(),
            "the copies have {} layers, the draft {}",
            rings.caches.len(),
            caches.len()
        );
        for (cache, copy) in caches.iter_mut().zip(&rings.caches) {
            if let Some(copy) = copy {
                stream.memcpy_dtod(&*copy.k, &mut *cache.k)?;
                stream.memcpy_dtod(&*copy.v, &mut *cache.v)?;
            }
        }
        b.extra_committed[slot - 1] = rings.committed;
        Ok(())
    }

    /// Runs an initial batched DFlash 2 round (no context commit) for `slots`, each anchored
    /// at `anchors[i]` with sampling `samplings[i]`, producing `steps` (`1..=7`) drafts per
    /// sequence and populating [`Self::batch_verify_tokens`] with `[seqs, steps + 1]`.
    pub fn batch_draft_initial(
        &mut self,
        stream: &Arc<CudaStream>,
        shared: SharedWeights<'_>,
        slots: &[usize],
        anchors: &[u32],
        samplings: &[Sampling],
        steps: usize,
    ) -> anyhow::Result<()> {
        let seqs = slots.len();
        ensure!(
            (1..=MAX_BATCH_SEQS).contains(&seqs),
            "batched DFlash 2 takes 1 to {MAX_BATCH_SEQS} sequences, not {seqs}"
        );
        ensure!(anchors.len() == seqs && samplings.len() == seqs, "mismatched batch lengths");
        ensure!((1..=STEPS).contains(&steps), "{steps} drafts, expected 1 to {STEPS}");
        check_distinct_slots(slots)?;
        let max_slot = slots.iter().copied().max().unwrap_or(0);
        self.ensure_batch_dflash2(stream, seqs, max_slot)?;
        self.sync_shortlist(stream)?;
        self.upload_batch_meta(stream, slots, samplings)?;
        let committed_positions: Vec<i32> = slots
            .iter()
            .map(|&slot| {
                let pos = self.slot_committed(slot);
                ensure!(pos > 0, "slot {slot} has no committed DFlash 2 context");
                Ok(i32::try_from(pos)?)
            })
            .collect::<anyhow::Result<_>>()?;
        let b = self.batch.as_mut().context("batch DFlash 2 missing")?;
        let controls = b.host_controls.as_mut_slice();
        for (i, (&pos, &anchor)) in committed_positions.iter().zip(anchors).enumerate() {
            controls[i * CONTROL] = pos;
            controls[i * CONTROL + 1] = 0;
            controls[i * CONTROL + 2] = i32::try_from(anchor)?;
        }
        b.host_controls.copy_to_device(
            stream,
            seqs * CONTROL,
            &mut b.controls.slice_mut(..seqs * CONTROL),
        )?;
        b.h2d_done.record(stream)?;
        self.launch_batch_graph(stream, shared, seqs, None, steps)
    }

    /// Commits each sequence's kept verify taps and drafts the next `draft_steps` (`1..=7`)
    /// tokens for all `slots` concurrently, reading `taps`, `results` and `anchors` directly
    /// from `Model::start_verify_batch`'s device buffers inside the captured CUDA graph
    /// without host synchronization or extra tap copies.
    #[expect(clippy::too_many_arguments, reason = "explicit device inputs from verify_batch")]
    pub fn batch_commit_and_draft(
        &mut self,
        stream: &Arc<CudaStream>,
        shared: SharedWeights<'_>,
        slots: &[usize],
        starts: &[usize],
        taps: &CudaView<'_, bf16>,
        results: &CudaView<'_, i32>,
        anchors: &CudaView<'_, i32>,
        samplings: &[Sampling],
        verify_width: usize,
        draft_steps: usize,
    ) -> anyhow::Result<()> {
        let seqs = slots.len();
        ensure!(
            (1..=MAX_BATCH_SEQS).contains(&seqs),
            "batched DFlash 2 takes 1 to {MAX_BATCH_SEQS} sequences, not {seqs}"
        );
        ensure!(starts.len() == seqs && samplings.len() == seqs, "mismatched batch lengths");
        ensure!((1..=BLOCK).contains(&verify_width), "bad verify_width {verify_width}");
        ensure!((1..=STEPS).contains(&draft_steps), "bad draft_steps {draft_steps}");
        check_distinct_slots(slots)?;
        let row_len = self.row_len();
        ensure!(
            taps.len() == seqs * verify_width * row_len,
            "taps has {} elements, expected {} x {row_len}",
            taps.len(),
            seqs * verify_width
        );
        ensure!(
            results.len() == seqs * RESULT_LEN,
            "results has {} elements, expected {}",
            results.len(),
            seqs * RESULT_LEN
        );
        ensure!(
            anchors.len() == seqs,
            "anchors has {} elements, expected {seqs}",
            anchors.len()
        );
        let mut starts_i32 = [0i32; MAX_BATCH_SEQS];
        for (dst, &s) in starts_i32[..seqs].iter_mut().zip(starts) {
            let end = s
                .checked_add(verify_width + BLOCK)
                .context("start position overflow")?;
            let _ = i32::try_from(end)?;
            *dst = i32::try_from(s)?;
        }
        let max_slot = slots.iter().copied().max().unwrap_or(0);
        self.ensure_batch_dflash2(stream, seqs, max_slot)?;
        self.sync_shortlist(stream)?;
        self.upload_batch_meta(stream, slots, samplings)?;
        let b = self.batch.as_mut().context("batch DFlash 2 missing")?;
        b.host_starts.as_mut_slice()[..seqs].copy_from_slice(&starts_i32[..seqs]);
        b.host_starts
            .copy_to_device(stream, seqs, &mut b.starts.slice_mut(..seqs))?;
        b.h2d_done.record(stream)?;
        let commit = BatchCommitInputs {
            taps: taps.view(),
            results: results.view(),
            anchors: anchors.view(),
            width: verify_width,
        };
        self.launch_batch_graph(stream, shared, seqs, Some(&commit), draft_steps)
    }

    /// Ready-to-verify tokens `[seqs * width]` produced by the last batched DFlash 2 round
    /// (`width = draft_steps + 1`), sequence-major `[anchor, d0..d_{steps-1}]`.
    pub fn batch_verify_tokens(&self, seqs: usize, width: usize) -> anyhow::Result<CudaView<'_, i32>> {
        let b = self
            .batch
            .as_ref()
            .context("no batched DFlash 2 round has run")?;
        Ok(b.verify_tokens.slice(..seqs * width))
    }

    /// Confidences `[seqs * steps]` of the last batched DFlash 2 round on the device.
    pub fn batch_confidences_device(
        &self,
        seqs: usize,
        steps: usize,
    ) -> anyhow::Result<CudaView<'_, f32>> {
        let b = self
            .batch
            .as_ref()
            .context("no batched DFlash 2 round has run")?;
        Ok(b.confidence.slice(..seqs * steps))
    }

    /// Confidences `[seqs * steps]` of the last batched DFlash 2 round, read to host.
    pub fn batch_confidences(
        &self,
        stream: &Arc<CudaStream>,
        seqs: usize,
        steps: usize,
    ) -> anyhow::Result<Vec<f32>> {
        let b = self
            .batch
            .as_ref()
            .context("no batched DFlash 2 round has run")?;
        Ok(stream.clone_dtoh(&b.confidence.slice(..seqs * steps))?)
    }

    /// Uploads `cache_table` (when `slots` changed) and `races` (when `samplings` changed)
    /// into stable device buffers using persistent page-locked staging buffers. Synchronizes
    /// `h2d_done` first so the caller's subsequent pinned H2D (`controls` or `starts`) may
    /// overwrite its staging buffer safely too.
    fn upload_batch_meta(
        &mut self,
        stream: &Arc<CudaStream>,
        slots: &[usize],
        samplings: &[Sampling],
    ) -> anyhow::Result<()> {
        let seqs = slots.len();
        let layers = self.arch.layers();
        let b = self.batch.as_mut().context("batch DFlash 2 missing")?;
        b.h2d_done.synchronize()?;
        if b.members.as_slice() != slots {
            let table_len = seqs * layers * 2;
            let table = b.host_cache_table.as_mut_slice();
            let mut idx = 0;
            for &slot in slots {
                let caches: &[KvCache] = if slot == 0 {
                    &self.caches
                } else {
                    &b.extra_slots[slot - 1]
                };
                for c in caches {
                    table[idx] = c.k.device_ptr(stream).0;
                    table[idx + 1] = c.v.device_ptr(stream).0;
                    idx += 2;
                }
            }
            b.host_cache_table
                .copy_to_device(stream, table_len, &mut b.cache_table.slice_mut(..table_len))?;
            b.members = slots.to_vec();
        }
        let race_len = seqs * RACE_WORDS;
        {
            let races = b.host_races.as_mut_slice();
            for (i, &s) in samplings.iter().enumerate() {
                let words = if s.is_greedy() {
                    race_words(0.0, 0)
                } else {
                    race_words(s.temperature, s.seed)
                };
                races[i * RACE_WORDS..(i + 1) * RACE_WORDS].copy_from_slice(&words);
            }
        }
        if b.last_races[..race_len] != b.host_races.as_slice()[..race_len] {
            b.host_races
                .copy_to_device(stream, race_len, &mut b.races.slice_mut(..race_len))?;
            b.last_races[..race_len].copy_from_slice(&b.host_races.as_slice()[..race_len]);
        }
        Ok(())
    }

    /// Replays (or captures) the CUDA graph for `(seqs, commit_width, draft_steps)`.
    fn launch_batch_graph(
        &mut self,
        stream: &Arc<CudaStream>,
        shared: SharedWeights<'_>,
        seqs: usize,
        commit: Option<&BatchCommitInputs<'_>>,
        draft_steps: usize,
    ) -> anyhow::Result<()> {
        let lm_head = shared.lm_head.device_ptr(stream).0;
        let embed = shared.embed.device_ptr(stream).0;
        let embed_norm = shared.embed_norm.device_ptr(stream).0;
        let (commit_width, taps_ptr, results_ptr, anchors_ptr) = match commit {
            Some(c) => (
                c.width,
                c.taps.device_ptr(stream).0,
                c.results.device_ptr(stream).0,
                c.anchors.device_ptr(stream).0,
            ),
            None => (0, 0, 0, 0),
        };
        let key = (seqs, commit_width, draft_steps);
        let needs_capture = self
            .batch
            .as_ref()
            .context("batch DFlash 2 missing")?
            .graphs
            .get(&key)
            .is_none_or(|g| {
                g.lm_head != lm_head
                    || g.embed != embed
                    || g.embed_norm != embed_norm
                    || g.taps != taps_ptr
                    || g.results != results_ptr
                    || g.anchors != anchors_ptr
            });
        if needs_capture {
            stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let enqueued =
                self.enqueue_batch_dflash2(stream, shared, seqs, commit, draft_steps);
            // End the capture even when enqueueing failed, so the stream is usable again.
            let graph = stream.end_capture(CUgraphInstantiate_flags(0));
            enqueued.context("capturing batched DFlash 2 round")?;
            let graph = graph?.context("batched DFlash 2 capture recorded no work")?;
            graph.upload()?;
            let b = self.batch.as_mut().context("batch DFlash 2 missing")?;
            b.graphs.insert(
                key,
                BatchRoundGraph {
                    graph,
                    lm_head,
                    embed,
                    embed_norm,
                    taps: taps_ptr,
                    results: results_ptr,
                    anchors: anchors_ptr,
                },
            );
        }
        let b = self.batch.as_ref().context("batch DFlash 2 missing")?;
        let round = b
            .graphs
            .get(&key)
            .context("batched DFlash 2 graph missing")?;
        round.graph.launch()?;
        Ok(())
    }

    /// Enqueues the batched DFlash 2 commit (when `commit` is `Some`), 5-layer block forward,
    /// LM head, top-16, and selector walk.
    #[expect(clippy::too_many_lines, reason = "linear top-to-bottom DFlash 2 round")]
    fn enqueue_batch_dflash2(
        &mut self,
        stream: &CudaStream,
        shared: SharedWeights<'_>,
        seqs: usize,
        commit: Option<&BatchCommitInputs<'_>>,
        draft_steps: usize,
    ) -> anyhow::Result<()> {
        let Arch::Dflash2 {
            weights,
            fc_fp8,
            mask_embed,
        } = &self.arch
        else {
            anyhow::bail!("not a DFlash 2 draft");
        };
        let layers = weights.layers.len();
        let q_width = DFLASH2_HEADS * HEAD_DIM;
        let mlp_width = self.arch.mlp_width();
        let rope = self.arch.rope();
        let k = &self.kernels;
        let b = self.batch.as_mut().context("batch DFlash 2 missing")?;
        let fp8_kv = b.fp8_kv;
        let lr = &self.linear_rows;
        let cache_table = b.cache_table.slice(..seqs * layers * 2);

        // 1. Optional context commit (`seqs * commit.width` rows).
        if let Some(c) = commit {
            k.set_control_kept_batch(
                stream,
                &mut b.controls.slice_mut(..seqs * CONTROL),
                &b.starts.slice(..seqs),
                &c.results,
                RESULT_LEN,
                c.width,
                &c.anchors,
                seqs,
            )?;
            let cm = seqs * c.width;
            run_rows_products(
                lr,
                stream,
                &c.taps,
                cm,
                &mut [(
                    Some(&weights.fc),
                    fc_fp8.as_ref(),
                    b.ctx_raw.slice_mut(..cm * HIDDEN),
                )],
            )?;
            k.add_rms_norm_rows(
                stream,
                None,
                &mut b.ctx_raw.slice_mut(..cm * HIDDEN),
                &weights.hidden_norm,
                &mut b.ctx_normed.slice_mut(..cm * HIDDEN),
                EPS,
            )?;
            let normed = b.ctx_normed.slice(..cm * HIDDEN);
            let layer_kv = cm * KV_WIDTH;
            let mut prods = Vec::with_capacity(2 * layers);
            for (l, (ck, cv)) in b
                .ctx_k
                .slice_mut(..layers * layer_kv)
                .chunks_exact_mut(layer_kv)
                .zip(
                    b.ctx_v
                        .slice_mut(..layers * layer_kv)
                        .chunks_exact_mut(layer_kv),
                )
                .enumerate()
            {
                let lw = &weights.layers[l];
                let fp8 = self.backbone.as_ref().map(|bb| &bb[l]);
                prods.push((Some(&lw.k), fp8.map(|f| &f.k), ck));
                prods.push((Some(&lw.v), fp8.map(|f| &f.v), cv));
            }
            run_rows_products(lr, stream, &normed, cm, &mut prods)?;
            let k_norms: Vec<&CudaSlice<bf16>> =
                weights.layers.iter().map(|l| &l.k_norm).collect();
            let controls = b.controls.slice(..seqs * CONTROL);
            k.dflash_commit_rope_store_batch(
                stream,
                &b.ctx_k.slice(..layers * layer_kv),
                &b.ctx_v.slice(..layers * layer_kv),
                &controls,
                &self.inv_freq,
                EPS,
                &k_norms,
                &cache_table,
                DFLASH2_SLOTS,
                c.width,
                seqs,
                rope,
                fp8_kv,
            )?;
        }
        let controls = b.controls.slice(..seqs * CONTROL);

        // 2. Block input (`seqs * 8` rows).
        let bm = seqs * BLOCK;
        k.block_input_batch(
            stream,
            &controls,
            shared.embed,
            shared.embed_norm,
            mask_embed,
            &mut b.residual.slice_mut(..bm * HIDDEN),
            seqs,
            EPS,
        )?;

        // 3. Five DFlash 2 layers over `bm = seqs * 8` rows.
        let mut pending: Option<&ConvWeights> = None;
        for (layer, lw) in weights.layers.iter().enumerate() {
            if layer == 0 {
                k.add_rms_norm_rows(
                    stream,
                    None,
                    &mut b.residual.slice_mut(..bm * HIDDEN),
                    &lw.input_norm,
                    &mut b.h.slice_mut(..bm * HIDDEN),
                    EPS,
                )?;
            } else {
                let prev = pending.context("previous layer MLP conv missing")?;
                k.conv_add_rms_norm_batch(
                    stream,
                    &b.mlp.slice(..bm * HIDDEN),
                    &b.coef_mlp.slice(..bm * CONV_COEFFICIENTS),
                    &prev.base,
                    &mut b.residual.slice_mut(..bm * HIDDEN),
                    &lw.input_norm,
                    &mut b.h.slice_mut(..bm * HIDDEN),
                    seqs,
                    EPS,
                )?;
            }
            let f = self.backbone.as_ref().map(|bb| &bb[layer]);
            // Attention input conv.
            {
                let h_view = b.h.slice(..bm * HIDDEN);
                let mut coef_out = b.coef_attention.slice_mut(..bm * CONV_COEFFICIENTS);
                lr.run(
                    stream,
                    &mut [RowsProduct::bf16(
                        &lw.attention_conv.projection,
                        &h_view,
                        &mut coef_out,
                    )],
                    bm,
                    Start::Early,
                )?;
            }
            k.conv_batch(
                stream,
                &b.h.slice(..bm * HIDDEN),
                &b.coef_attention.slice(..bm * CONV_COEFFICIENTS),
                0,
                &lw.attention_conv.base,
                &mut b.conv.slice_mut(..bm * HIDDEN),
                seqs,
            )?;
            // Q, K, V projections.
            {
                let conv_in = b.conv.slice(..bm * HIDDEN);
                let q_out = b.q.slice_mut(..bm * q_width);
                let k_out = b.k.slice_mut(..bm * KV_WIDTH);
                let v_out = b.v.slice_mut(..bm * KV_WIDTH);
                run_rows_products(
                    lr,
                    stream,
                    &conv_in,
                    bm,
                    &mut [
                        (lw.q.as_ref(), f.map(|f| &f.q), q_out),
                        (Some(&lw.k), f.map(|f| &f.k), k_out),
                        (Some(&lw.v), f.map(|f| &f.v), v_out),
                    ],
                )?;
            }
            // RoPE + KV ring store + batched G=4 sliding attention.
            k.block_rope_store_batch(
                stream,
                &mut b.q.slice_mut(..bm * q_width),
                &b.k.slice(..bm * KV_WIDTH),
                &b.v.slice(..bm * KV_WIDTH),
                &controls,
                &self.inv_freq,
                EPS,
                &lw.q_norm,
                &lw.k_norm,
                &cache_table,
                layer,
                layers,
                DFLASH2_SLOTS,
                seqs,
                rope,
                DFLASH2_HEADS,
                fp8_kv,
            )?;
            k.attention_g4_batch(
                stream,
                &b.q.slice(..bm * q_width),
                &cache_table,
                layer,
                layers,
                DFLASH2_SLOTS,
                &controls,
                DFLASH2_WINDOW - 1,
                DFLASH2_WINDOW,
                DFLASH2_SLOTS,
                seqs,
                fp8_kv,
                &mut b.attention,
                &mut b.attn.slice_mut(..bm * q_width),
            )?;
            // O projection.
            {
                let attn_in = b.attn.slice(..bm * q_width);
                let o_out = b.o.slice_mut(..bm * HIDDEN);
                run_rows_products(
                    lr,
                    stream,
                    &attn_in,
                    bm,
                    &mut [(lw.o.as_ref(), f.map(|f| &f.o), o_out)],
                )?;
            }
            // Attention output conv + residual + post_attention_norm.
            k.conv_add_rms_norm_batch(
                stream,
                &b.o.slice(..bm * HIDDEN),
                &b.coef_attention.slice(..bm * CONV_COEFFICIENTS),
                &lw.attention_conv.base,
                &mut b.residual.slice_mut(..bm * HIDDEN),
                &lw.post_attention_norm,
                &mut b.h.slice_mut(..bm * HIDDEN),
                seqs,
                EPS,
            )?;
            // MLP input conv.
            {
                let h_view = b.h.slice(..bm * HIDDEN);
                let mut coef_out = b.coef_mlp.slice_mut(..bm * CONV_COEFFICIENTS);
                lr.run(
                    stream,
                    &mut [RowsProduct::bf16(
                        &lw.mlp_conv.projection,
                        &h_view,
                        &mut coef_out,
                    )],
                    bm,
                    Start::Early,
                )?;
            }
            k.conv_batch(
                stream,
                &b.h.slice(..bm * HIDDEN),
                &b.coef_mlp.slice(..bm * CONV_COEFFICIENTS),
                0,
                &lw.mlp_conv.base,
                &mut b.conv.slice_mut(..bm * HIDDEN),
                seqs,
            )?;
            // Gate + Up projections, SiLU*mul, Down projection.
            {
                let conv_in = b.conv.slice(..bm * HIDDEN);
                let gate_out = b.gate.slice_mut(..bm * mlp_width);
                let up_out = b.up.slice_mut(..bm * mlp_width);
                run_rows_products(
                    lr,
                    stream,
                    &conv_in,
                    bm,
                    &mut [
                        (lw.gate.as_ref(), f.map(|f| &f.gate), gate_out),
                        (lw.up.as_ref(), f.map(|f| &f.up), up_out),
                    ],
                )?;
            }
            k.silu_mul_rows(
                stream,
                &b.gate.slice(..bm * mlp_width),
                &b.up.slice(..bm * mlp_width),
                &mut b.act.slice_mut(..bm * mlp_width),
            )?;
            {
                let act_in = b.act.slice(..bm * mlp_width);
                let mlp_out = b.mlp.slice_mut(..bm * HIDDEN);
                run_rows_products(
                    lr,
                    stream,
                    &act_in,
                    bm,
                    &mut [(lw.down.as_ref(), f.map(|f| &f.down), mlp_out)],
                )?;
            }
            pending = Some(&lw.mlp_conv);
        }
        // Final MLP output conv + residual + final norm into `y`.
        let last_conv = pending.context("DFlash 2 has no layers")?;
        k.conv_add_rms_norm_batch(
            stream,
            &b.mlp.slice(..bm * HIDDEN),
            &b.coef_mlp.slice(..bm * CONV_COEFFICIENTS),
            &last_conv.base,
            &mut b.residual.slice_mut(..bm * HIDDEN),
            &weights.norm,
            &mut b.y.slice_mut(..bm * HIDDEN),
            seqs,
            EPS,
        )?;

        // 4. Head & Candidate Selector (`sm = seqs * draft_steps` rows).
        let sm = seqs * draft_steps;
        k.gather_mask_rows(
            stream,
            &b.y.slice(..bm * HIDDEN),
            &mut b.y_steps.slice_mut(..sm * HIDDEN),
            seqs,
            draft_steps,
        )?;
        let head = self.head.as_ref();
        let list = head.and_then(|h| h.shortlist.as_ref());
        let rows = match list {
            Some(list) => list.capacity(),
            None => self.vocab,
        };
        let ld = rows;
        {
            let y_in = b.y_steps.slice(..sm * HIDDEN);
            let mut base_out = b.base.slice_mut(..sm * ld);
            match head.and_then(|h| h.lm.as_ref()) {
                None => {
                    let lm = self.head_copy.as_ref().unwrap_or(shared.lm_head);
                    let w = lm.slice(..rows * HIDDEN);
                    lr.run(
                        stream,
                        &mut [RowsProduct::bf16(&w, &y_in, &mut base_out)],
                        sm,
                        Start::Early,
                    )?;
                }
                Some(Matrix::Bf16(w)) => {
                    let w = w.slice(..rows * HIDDEN);
                    lr.run(
                        stream,
                        &mut [RowsProduct::bf16(&w, &y_in, &mut base_out)],
                        sm,
                        Start::Early,
                    )?;
                }
                Some(Matrix::Fp8(w)) => {
                    lr.run(
                        stream,
                        &mut [RowsProduct::fp8_raw(
                            w.values.slice(..rows * HIDDEN),
                            w.scales.slice(..rows),
                            rows,
                            HIDDEN,
                            &y_in,
                            &mut base_out,
                        )],
                        sm,
                        Start::Early,
                    )?;
                }
            }
        }
        k.topk(
            stream,
            &b.base.slice(..sm * ld),
            sm,
            rows,
            ld,
            list.map(|l| &l.count),
            list.map(|l| &l.map),
            &mut b.topk,
            &mut b.unary.slice_mut(..sm * TOP_K),
            &mut b.cand.slice_mut(..sm * TOP_K),
        )?;
        {
            let y_in = b.y_steps.slice(..sm * HIDDEN);
            let mut hp_out = b.hp.slice_mut(..sm * RANK);
            lr.run(
                stream,
                &mut [RowsProduct::bf16(
                    &weights.selector.hidden_projection,
                    &y_in,
                    &mut hp_out,
                )],
                sm,
                Start::Early,
            )?;
        }
        k.select_batch(
            stream,
            &b.cand.slice(..sm * TOP_K),
            &b.unary.slice(..sm * TOP_K),
            &b.hp.slice(..sm * RANK),
            &weights.selector.predecessor,
            &weights.selector.successor,
            &controls,
            seqs,
            draft_steps,
            self.softcap,
            &mut b.ids.slice_mut(..sm),
            &mut b.confidence.slice_mut(..sm),
            Some(b.verify_tokens.slice_mut(..seqs * (draft_steps + 1))),
            &b.races.slice(..seqs * RACE_WORDS),
        )
    }
}
