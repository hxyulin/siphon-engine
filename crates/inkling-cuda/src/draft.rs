//! The DSpark draft bundled with the checkpoint: a 3-layer dense transformer that drafts
//! one block of 8 rows in one pass, and a recurrent Markov head that turns the 7 mask rows
//! into 7 greedy draft tokens.
//!
//! A round anchored at position `P` runs the block at positions `P..P + 8`: the anchor
//! token's normalized target embedding, then 7 copies of `mask_embed`. The block attends
//! bidirectionally to itself and to the draft cache below `P`, the last 1024 positions in
//! layers 0 and 2 and every position in layer 1. [`Draft::commit`] fills the cache below
//! `P` from the target's four feature taps of each committed position: each layer mixes
//! them, normalizes them and projects them through `k_proj_ctx` and `v_proj_ctx`. A round
//! writes its own block's keys and values at `P..P + 8`, which the next commit or round
//! overwrites, so the cache needs no rollback.
//!
//! The numerics follow vLLM's draft, which `reference/dspark.py` reproduces in MLX: BF16
//! weights and activations, F32 accumulation, and rounding where vLLM rounds (see
//! `kernels/draft.cu`).

use std::sync::Arc;

use anyhow::ensure;
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use inkling::TextConfig;

use crate::{
    DeviceIn,
    ops::{
        draft::{ArgmaxPartials, AttentionScratch, BLOCK, DraftKernels, KV_WIDTH, KvCache},
        linear::{Gemm, Gemv, Linear, Start},
    },
    weights::draft::{
        DraftWeights, HEAD_DIM, HEADS, HIDDEN, LAYERS, MLP_WIDTH, RANK, SharedWeights, TAPS,
    },
};

/// Draft tokens per round: one per mask row.
pub const STEPS: usize = BLOCK - 1;
/// Positions before the block that the local layers (0 and 2) see.
pub const WINDOW: usize = 1024;
/// Slots of the local layers' rings: the window and the block.
const LOCAL_SLOTS: usize = WINDOW + BLOCK;
/// RoPE base of the draft (`rope_theta` in `mtp_config`).
const ROPE_THETA: f64 = 1e6;
/// RMSNorm epsilon of the draft (`rms_norm_eps` in `mtp_config`).
const EPS: f32 = 1e-6;
/// Target positions one [`Draft::commit`] chunk projects at once.
const COMMIT_ROWS: usize = 512;

/// The RoPE frequencies `1 / theta^(2i/128)` as vLLM's F32 table builds them: a correctly
/// rounded power, then a correctly rounded reciprocal.
#[must_use]
pub fn rope_inv_freq() -> Vec<f32> {
    (0..HEAD_DIM / 2)
        .map(|i| {
            #[expect(clippy::cast_precision_loss, reason = "i is below 64")]
            let exponent = (2 * i) as f64 / HEAD_DIM as f64;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "rounding the power to F32 is the intent"
            )]
            let power = ROPE_THETA.powf(exponent) as f32;
            1.0 / power
        })
        .collect()
}

/// The draft's weights, its LM-head fold, its cache and its scratch buffers.
pub struct Draft {
    kernels: DraftKernels,
    weights: DraftWeights,
    /// `lm_head · hidden_out`, BF16 `[padded vocab, 256]`, and `lm_head · hidden_out.bias`.
    fold_weight: CudaSlice<bf16>,
    fold_bias: CudaSlice<bf16>,
    /// `softmax(fuse_weight)`, F32 `[4, 4]`.
    alpha: CudaSlice<f32>,
    /// RoPE frequencies `theta^(-2i/128)`, F32 `[64]`.
    inv_freq: CudaSlice<f32>,
    caches: Vec<KvCache>,
    /// Positions the global layer holds.
    capacity: usize,
    /// One past the last committed position.
    committed: usize,
    vocab: usize,
    padded_vocab: usize,
    softcap: f32,
    s: Scratch,
}

struct Scratch {
    mixed: CudaSlice<bf16>,
    ctx_k: CudaSlice<bf16>,
    ctx_v: CudaSlice<bf16>,
    /// The Markov head's mixed taps of the last committed position, BF16 `[2304]`.
    f: CudaSlice<bf16>,
    residual: CudaSlice<bf16>,
    h: CudaSlice<bf16>,
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    attn: CudaSlice<bf16>,
    o: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
    up: CudaSlice<bf16>,
    act: CudaSlice<bf16>,
    mlp: CudaSlice<bf16>,
    /// The decoder's output for the block, BF16 `[8, 2304]`.
    y: CudaSlice<bf16>,
    attention: AttentionScratch,
    /// Rows 1 to 7 of `y` through the LM head, BF16 `[7, padded vocab]`.
    base: CudaSlice<bf16>,
    context_in: CudaSlice<f32>,
    context: CudaSlice<f32>,
    step_in: CudaSlice<f32>,
    raw: CudaSlice<f32>,
    latent: CudaSlice<f32>,
    acc: CudaSlice<f32>,
    partials: ArgmaxPartials,
    ids: CudaSlice<i32>,
    /// Each step's Markov logits, BF16 `[7, vocab]`, when recorded.
    logits: Option<CudaSlice<bf16>>,
}

impl Draft {
    /// Uploads the fold and allocates the cache and scratch. The global layer's cache holds
    /// `capacity` positions, 4 KiB each; the local layers' rings hold 1032.
    pub fn new(
        stream: &Arc<CudaStream>,
        config: &TextConfig,
        weights: DraftWeights,
        shared: SharedWeights<'_>,
        capacity: usize,
    ) -> anyhow::Result<Self> {
        ensure!(
            capacity >= LOCAL_SLOTS,
            "capacity {capacity} is below the {LOCAL_SLOTS} positions of a window and a block"
        );
        let kernels = DraftKernels::load(stream.context())?;
        let padded_vocab = config.vocab_size;
        ensure!(
            shared.lm_head.len() == padded_vocab * HIDDEN,
            "the LM head has {} elements, expected {padded_vocab} x {HIDDEN}",
            shared.lm_head.len()
        );
        let mut fold_weight = stream.alloc_zeros(padded_vocab * RANK)?;
        let mut fold_bias = stream.alloc_zeros(padded_vocab)?;
        kernels.fold(
            stream,
            shared.lm_head,
            &weights.markov.hidden_out,
            &weights.markov.hidden_out_bias,
            &mut fold_weight,
            &mut fold_bias,
        )?;
        let alpha: Vec<f32> = weights.alpha.iter().flatten().copied().collect();
        let inv_freq = rope_inv_freq();
        let caches = (0..LAYERS)
            .map(|layer| KvCache::new(stream, if layer == 1 { capacity } else { LOCAL_SLOTS }))
            .collect::<anyhow::Result<_>>()?;
        let z = |n: usize| stream.alloc_zeros::<bf16>(n);
        let zf = |n: usize| stream.alloc_zeros::<f32>(n);
        let s = Scratch {
            mixed: z(COMMIT_ROWS * HIDDEN)?,
            ctx_k: z(COMMIT_ROWS * KV_WIDTH)?,
            ctx_v: z(COMMIT_ROWS * KV_WIDTH)?,
            f: z(HIDDEN)?,
            residual: z(BLOCK * HIDDEN)?,
            h: z(BLOCK * HIDDEN)?,
            q: z(BLOCK * HEADS * HEAD_DIM)?,
            k: z(BLOCK * KV_WIDTH)?,
            v: z(BLOCK * KV_WIDTH)?,
            attn: z(BLOCK * HEADS * HEAD_DIM)?,
            o: z(BLOCK * HIDDEN)?,
            gate: z(BLOCK * MLP_WIDTH)?,
            up: z(BLOCK * MLP_WIDTH)?,
            act: z(BLOCK * MLP_WIDTH)?,
            mlp: z(BLOCK * HIDDEN)?,
            y: z(BLOCK * HIDDEN)?,
            attention: AttentionScratch::new(stream)?,
            base: z(STEPS * padded_vocab)?,
            context_in: zf(STEPS * 2 * HIDDEN)?,
            context: zf(STEPS * 3 * RANK)?,
            step_in: zf(2 * RANK)?,
            raw: zf(3 * RANK)?,
            latent: zf(RANK)?,
            acc: zf(padded_vocab)?,
            partials: ArgmaxPartials::new(stream)?,
            ids: stream.alloc_zeros(STEPS)?,
            logits: None,
        };
        Ok(Self {
            kernels,
            weights,
            fold_weight,
            fold_bias,
            alpha: stream.clone_htod(&alpha)?,
            inv_freq: stream.clone_htod(&inv_freq)?,
            caches,
            capacity,
            committed: 0,
            vocab: config.unpadded_vocab_size,
            padded_vocab,
            softcap: config.final_logit_softcapping,
            s,
        })
    }

    /// Positions the global layer's cache holds.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// One past the last committed position: the position of the next round's anchor.
    #[must_use]
    pub fn committed(&self) -> usize {
        self.committed
    }

    /// Forgets every committed position, for a new sequence.
    pub fn reset(&mut self) {
        self.committed = 0;
    }

    /// Keeps each step's Markov logits from the next round on, for tests.
    pub fn record_logits(&mut self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        if self.s.logits.is_none() {
            self.s.logits = Some(stream.alloc_zeros(STEPS * self.vocab)?);
        }
        Ok(())
    }

    /// The last round's 7 draft token ids, I32.
    #[must_use]
    pub fn draft_ids(&self) -> &CudaSlice<i32> {
        &self.s.ids
    }

    /// The last round's decoder output for the 8 block rows, BF16 `[8, 2304]`.
    #[must_use]
    pub fn y(&self) -> &CudaSlice<bf16> {
        &self.s.y
    }

    /// The last round's base logits, rows 1 to 7 of [`Self::y`] through the LM head, BF16
    /// `[7, padded vocab]`.
    #[must_use]
    pub fn base_logits(&self) -> &CudaSlice<bf16> {
        &self.s.base
    }

    /// The last round's Markov logits, BF16 `[7, vocab]`, after [`Self::record_logits`].
    #[must_use]
    pub fn markov_logits(&self) -> Option<&CudaSlice<bf16>> {
        self.s.logits.as_ref()
    }

    /// Writes the draft cache entries of target positions `start..start + rows` from their
    /// taps, BF16 `[rows, 4, 2304]`: the partial block entering visits 5, 23 and 41, and the
    /// final normed hidden state. The last row's taps also become the Markov head's context
    /// for the round anchored right after it. `start` may go back to rewrite positions, but
    /// must not skip any.
    pub fn commit(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        taps: &impl DeviceIn<bf16>,
        start: usize,
    ) -> anyhow::Result<()> {
        let taps = taps.view();
        let row_len = TAPS * HIDDEN;
        ensure!(
            taps.len().is_multiple_of(row_len) && !taps.is_empty(),
            "taps have {} elements, not a positive multiple of {row_len}",
            taps.len()
        );
        let rows = taps.len() / row_len;
        ensure!(
            start <= self.committed,
            "committing from {start} would skip positions {}..{start}",
            self.committed
        );
        ensure!(
            start + rows <= self.capacity,
            "positions up to {} exceed the capacity {}",
            start + rows,
            self.capacity
        );
        let s = &mut self.s;
        for offset in (0..rows).step_by(COMMIT_ROWS) {
            let n = COMMIT_ROWS.min(rows - offset);
            let chunk = taps.slice(offset * row_len..(offset + n) * row_len);
            let position = start + offset;
            for (layer, (w, cache)) in self.weights.layers.iter().zip(&mut self.caches).enumerate()
            {
                let mut mixed = s.mixed.slice_mut(..n * HIDDEN);
                let alpha = self.alpha.slice(layer * TAPS..(layer + 1) * TAPS);
                self.kernels.mix_rms_norm(
                    stream,
                    &chunk,
                    &alpha,
                    &w.dflare_norm,
                    &mut mixed,
                    EPS,
                )?;
                let mixed = s.mixed.slice(..n * HIDDEN);
                let mut k = s.ctx_k.slice_mut(..n * KV_WIDTH);
                let mut v = s.ctx_v.slice_mut(..n * KV_WIDTH);
                linear.gemm_bf16(
                    stream,
                    &mut [
                        Gemm::new(&w.k_ctx, &mixed, &mut k),
                        Gemm::new(&w.v_ctx, &mixed, &mut v),
                    ],
                    n,
                )?;
                self.kernels.head_norm_rope(
                    stream,
                    &mut s.ctx_k,
                    n,
                    KV_WIDTH / HEAD_DIM,
                    KV_WIDTH,
                    &w.k_norm_ctx,
                    &self.inv_freq,
                    position,
                    EPS,
                )?;
                self.kernels
                    .store_kv(stream, &s.ctx_k, &s.ctx_v, n, KV_WIDTH, cache, position)?;
            }
        }
        let last = taps.slice((rows - 1) * row_len..);
        let alpha = self.alpha.slice(LAYERS * TAPS..);
        self.kernels.mix_rms_norm(
            stream,
            &last,
            &alpha,
            &self.weights.markov.dflare_norm,
            &mut s.f,
            EPS,
        )?;
        self.committed = start + rows;
        Ok(())
    }

    /// Runs one round anchored at the token in `anchor` (one I32 on the device) at position
    /// [`Self::committed`]: the block forward, the base logits and the 7 greedy Markov
    /// steps. The ids land in [`Self::draft_ids`].
    pub fn round(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        shared: SharedWeights<'_>,
        anchor: &impl DeviceIn<i32>,
    ) -> anyhow::Result<()> {
        let position = self.committed;
        ensure!(
            position > 0,
            "a round needs at least one committed position"
        );
        ensure!(
            position + BLOCK <= self.capacity,
            "the block at {position} exceeds the capacity {}",
            self.capacity
        );
        self.block(stream, linear, shared, anchor, position)?;
        self.markov(stream, linear, shared, anchor)
    }

    fn block(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        shared: SharedWeights<'_>,
        anchor: &impl DeviceIn<i32>,
        position: usize,
    ) -> anyhow::Result<()> {
        let k = &self.kernels;
        let s = &mut self.s;
        k.block_input(
            stream,
            anchor,
            shared.embed,
            shared.embed_norm,
            &self.weights.mask_embed,
            &mut s.residual,
            EPS,
        )?;
        for (layer, (w, cache)) in self.weights.layers.iter().zip(&mut self.caches).enumerate() {
            let x = (layer > 0).then_some(&s.mlp);
            k.add_rms_norm(stream, x, &mut s.residual, &w.input_norm, &mut s.h, EPS)?;
            linear.gemm_bf16(
                stream,
                &mut [
                    Gemm::new(&w.q, &s.h, &mut s.q),
                    Gemm::new(&w.k, &s.h, &mut s.k),
                    Gemm::new(&w.v, &s.h, &mut s.v),
                ],
                BLOCK,
            )?;
            let q_width = HEADS * HEAD_DIM;
            let kv_heads = KV_WIDTH / HEAD_DIM;
            k.head_norm_rope(
                stream,
                &mut s.q,
                BLOCK,
                HEADS,
                q_width,
                &w.q_norm,
                &self.inv_freq,
                position,
                EPS,
            )?;
            k.head_norm_rope(
                stream,
                &mut s.k,
                BLOCK,
                kv_heads,
                KV_WIDTH,
                &w.k_norm,
                &self.inv_freq,
                position,
                EPS,
            )?;
            k.store_kv(stream, &s.k, &s.v, BLOCK, KV_WIDTH, cache, position)?;
            let first = if layer == 1 {
                0
            } else {
                position.saturating_sub(WINDOW)
            };
            let keys = position + BLOCK - first;
            k.attention(
                stream,
                &s.q,
                cache,
                first,
                keys,
                &mut s.attention,
                &mut s.attn,
            )?;
            linear.gemm_bf16(stream, &mut [Gemm::new(&w.o, &s.attn, &mut s.o)], BLOCK)?;
            k.add_rms_norm(
                stream,
                Some(&s.o),
                &mut s.residual,
                &w.post_attention_norm,
                &mut s.h,
                EPS,
            )?;
            linear.gemm_bf16(
                stream,
                &mut [
                    Gemm::new(&w.gate, &s.h, &mut s.gate),
                    Gemm::new(&w.up, &s.h, &mut s.up),
                ],
                BLOCK,
            )?;
            k.silu_mul(stream, &s.gate, &s.up, &mut s.act)?;
            linear.gemm_bf16(stream, &mut [Gemm::new(&w.down, &s.act, &mut s.mlp)], BLOCK)?;
        }
        k.add_rms_norm(
            stream,
            Some(&s.mlp),
            &mut s.residual,
            &self.weights.norm,
            &mut s.y,
            EPS,
        )
    }

    fn markov(
        &mut self,
        stream: &CudaStream,
        linear: &Linear,
        shared: SharedWeights<'_>,
        anchor: &impl DeviceIn<i32>,
    ) -> anyhow::Result<()> {
        let k = &self.kernels;
        let m = &self.weights.markov;
        let s = &mut self.s;
        let rows = s.y.slice(HIDDEN..);
        linear.gemm_bf16(
            stream,
            &mut [Gemm::new(shared.lm_head, &rows, &mut s.base)],
            STEPS,
        )?;
        k.markov_context_input(stream, &s.y, &s.f, &mut s.context_in)?;
        linear.gemm_f32(
            stream,
            &mut [Gemm::new(&m.rnn_context, &s.context_in, &mut s.context)],
            STEPS,
        )?;
        k.markov_init(stream, anchor, &m.w1, &mut s.step_in)?;
        for step in 0..STEPS {
            linear.gemv_f32(
                stream,
                &mut [Gemv::new(&m.rnn_state, &s.step_in, &mut s.raw)],
                Start::Serial,
            )?;
            let context = s.context.slice(step * 3 * RANK..(step + 1) * 3 * RANK);
            k.markov_gate(
                stream,
                &s.raw,
                &context,
                &m.rnn_bias,
                &mut s.step_in,
                &mut s.latent,
            )?;
            linear.gemv_f32(
                stream,
                &mut [Gemv::new(&self.fold_weight, &s.latent, &mut s.acc)],
                Start::Serial,
            )?;
            let base = s
                .base
                .slice(step * self.padded_vocab..(step + 1) * self.padded_vocab);
            let logits = s
                .logits
                .as_mut()
                .map(|l| l.slice_mut(step * self.vocab..(step + 1) * self.vocab));
            k.markov_logits(
                stream,
                &base,
                &self.fold_bias,
                &s.acc,
                self.vocab,
                self.softcap,
                logits,
                &mut s.partials,
            )?;
            let mut id = s.ids.slice_mut(step..=step);
            k.markov_pick(stream, &s.partials, &mut id, &m.w1, &mut s.step_in)?;
        }
        Ok(())
    }
}
