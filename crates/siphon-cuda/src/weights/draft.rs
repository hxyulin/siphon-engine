//! Device copies of the DSpark draft's weights (`mtp.safetensors`).
//!
//! Tensor names are the draft file's own, without a prefix. The shapes come from the
//! checkpoint's `mtp_config`, which `siphon::TextConfig` does not parse; every tensor's
//! shape is checked on load. The draft shares the target's embedding, `embed_norm` and
//! LM head ([`SharedWeights`]) and has no copies of them.

use std::sync::Arc;

use anyhow::ensure;
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use siphon::{Checkpoint, TextConfig};

use super::{Loader, Weights};

/// Hidden width of the draft, equal to the target's.
pub const HIDDEN: usize = 2304;
/// Query heads, KV heads and head dimension of the draft's attention.
pub const HEADS: usize = 24;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
/// Width of the SiLU-gated MLP.
pub const MLP_WIDTH: usize = 9216;
/// Decoder layers. Layers 0 and 2 attend to a window of 1024 positions, layer 1 to all.
pub const LAYERS: usize = 3;
/// Rank of the Markov head's state and token embedding.
pub const RANK: usize = 256;
/// Target features the draft mixes: the partial block entering visits 5, 23 and 41, and
/// the final normed hidden state (`target_layer_ids [5, 23, 41, 48]`).
pub const TAPS: usize = 4;

/// The draft's weights. The LM-head fold lives in [`crate::draft::Draft`], since it needs
/// the target's LM head.
#[derive(Debug)]
pub struct DraftWeights {
    /// `softmax(fuse_weight, axis -1)` in F32, `[4 rows, 4 taps]`: row l mixes the taps
    /// for draft layer l, row 3 for the Markov head.
    pub alpha: [[f32; TAPS]; LAYERS + 1],
    pub mask_embed: CudaSlice<bf16>,
    /// Final norm of the decoder.
    pub norm: CudaSlice<bf16>,
    pub layers: Vec<DraftLayerWeights>,
    pub markov: MarkovWeights,
}

#[derive(Debug)]
pub struct DraftLayerWeights {
    /// RMSNorm of the layer's mixed taps.
    pub dflare_norm: CudaSlice<bf16>,
    pub input_norm: CudaSlice<bf16>,
    pub post_attention_norm: CudaSlice<bf16>,
    /// `[3072, 2304]`.
    pub q: CudaSlice<bf16>,
    /// `[1024, 2304]` each: block keys and values.
    pub k: CudaSlice<bf16>,
    pub v: CudaSlice<bf16>,
    /// `[1024, 2304]` each: context keys and values from the mixed taps.
    pub k_ctx: CudaSlice<bf16>,
    pub v_ctx: CudaSlice<bf16>,
    /// `[128]` each: per-head norms of queries, block keys and context keys.
    pub q_norm: CudaSlice<bf16>,
    pub k_norm: CudaSlice<bf16>,
    pub k_norm_ctx: CudaSlice<bf16>,
    /// `[2304, 3072]`.
    pub o: CudaSlice<bf16>,
    /// `[9216, 2304]` each.
    pub gate: CudaSlice<bf16>,
    pub up: CudaSlice<bf16>,
    /// `[2304, 9216]`.
    pub down: CudaSlice<bf16>,
}

#[derive(Debug)]
pub struct MarkovWeights {
    /// RMSNorm of the head's mixed taps.
    pub dflare_norm: CudaSlice<bf16>,
    /// Token embedding of the previous draft token, `[201024, 256]`.
    pub w1: CudaSlice<bf16>,
    /// Columns 0 to 511 of `rnn_proj.weight`, for `[state; w1 token]`: `[768, 512]`.
    pub rnn_state: CudaSlice<bf16>,
    /// Columns 512 to 5119 of `rnn_proj.weight`, for `[y; f]`: `[768, 4608]`.
    pub rnn_context: CudaSlice<bf16>,
    /// `rnn_proj.bias` widened to F32, `[768]`.
    pub rnn_bias: CudaSlice<f32>,
    /// `hidden_out`, `[2304, 256]` and `[2304]`: folded into the LM head at load.
    pub hidden_out: CudaSlice<bf16>,
    pub hidden_out_bias: CudaSlice<bf16>,
}

/// The target tensors the draft reads: the embedding and `embed_norm` for the anchor row,
/// and the LM head for the base logits and the fold.
#[derive(Clone, Copy, Debug)]
pub struct SharedWeights<'a> {
    /// `[vocab, 2304]`.
    pub embed: &'a CudaSlice<bf16>,
    pub embed_norm: &'a CudaSlice<bf16>,
    /// `[vocab, 2304]`.
    pub lm_head: &'a CudaSlice<bf16>,
}

impl<'a> From<&'a Weights> for SharedWeights<'a> {
    fn from(weights: &'a Weights) -> Self {
        Self {
            embed: &weights.embed,
            embed_norm: &weights.embed_norm,
            lm_head: &weights.lm_head,
        }
    }
}

/// Device copies of only the target tensors in [`SharedWeights`], for running the draft
/// without the rest of the target.
#[derive(Debug)]
pub struct SharedTensors {
    pub embed: CudaSlice<bf16>,
    pub embed_norm: CudaSlice<bf16>,
    pub lm_head: CudaSlice<bf16>,
}

impl SharedTensors {
    pub fn load(
        checkpoint: &Checkpoint,
        config: &TextConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let loader = Loader { checkpoint, stream };
        let shape = [config.vocab_size, config.hidden_size];
        Ok(Self {
            embed: loader.bf16("model.llm.embed.weight", &shape)?,
            embed_norm: loader.bf16("model.llm.embed_norm.weight", &[config.hidden_size])?,
            lm_head: loader.bf16("model.llm.unembed.weight", &shape)?,
        })
    }
}

impl<'a> From<&'a SharedTensors> for SharedWeights<'a> {
    fn from(tensors: &'a SharedTensors) -> Self {
        Self {
            embed: &tensors.embed,
            embed_norm: &tensors.embed_norm,
            lm_head: &tensors.lm_head,
        }
    }
}

impl DraftWeights {
    /// Uploads the draft's weights, about 0.64 GB.
    pub fn load(
        checkpoint: &Checkpoint,
        config: &TextConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        ensure!(
            config.hidden_size == HIDDEN,
            "the draft expects hidden size {HIDDEN}, the target has {}",
            config.hidden_size
        );
        let loader = Loader { checkpoint, stream };
        let fuse = loader.f32_host("fuse_weight", &[LAYERS + 1, TAPS])?;
        let mut alpha = [[0.0; TAPS]; LAYERS + 1];
        for (row, weights) in alpha.iter_mut().zip(fuse.as_chunks::<TAPS>().0) {
            *row = softmax(weights);
        }
        let layers = (0..LAYERS)
            .map(|layer| DraftLayerWeights::load(&loader, layer))
            .collect::<anyhow::Result<_>>()?;
        Ok(Self {
            alpha,
            mask_embed: loader.bf16("mask_embed", &[HIDDEN])?,
            norm: loader.bf16("norm.weight", &[HIDDEN])?,
            layers,
            markov: MarkovWeights::load(&loader, config.vocab_size)?,
        })
    }
}

/// Softmax in F64, rounded to F32: the same values as torch's F32 softmax up to the last
/// bit, since the inputs are four BF16 numbers.
fn softmax(x: &[f32; TAPS]) -> [f32; TAPS] {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = x.iter().map(|&v| f64::from(v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let mut out = [0.0; TAPS];
    for (o, e) in out.iter_mut().zip(&exps) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "rounding the F64 softmax to F32 is the intent"
        )]
        let value = (e / sum) as f32;
        *o = value;
    }
    out
}

impl DraftLayerWeights {
    fn load(loader: &Loader<'_>, layer: usize) -> anyhow::Result<Self> {
        let p = format!("layers.{layer}.");
        let attn = format!("{p}self_attn.");
        let kv = KV_HEADS * HEAD_DIM;
        let q = HEADS * HEAD_DIM;
        Ok(Self {
            dflare_norm: loader.bf16(&format!("{p}dflare_norm.weight"), &[HIDDEN])?,
            input_norm: loader.bf16(&format!("{p}input_layernorm.weight"), &[HIDDEN])?,
            post_attention_norm: loader
                .bf16(&format!("{p}post_attention_layernorm.weight"), &[HIDDEN])?,
            q: loader.bf16(&format!("{attn}q_proj.weight"), &[q, HIDDEN])?,
            k: loader.bf16(&format!("{attn}k_proj.weight"), &[kv, HIDDEN])?,
            v: loader.bf16(&format!("{attn}v_proj.weight"), &[kv, HIDDEN])?,
            k_ctx: loader.bf16(&format!("{attn}k_proj_ctx.weight"), &[kv, HIDDEN])?,
            v_ctx: loader.bf16(&format!("{attn}v_proj_ctx.weight"), &[kv, HIDDEN])?,
            q_norm: loader.bf16(&format!("{attn}q_norm.weight"), &[HEAD_DIM])?,
            k_norm: loader.bf16(&format!("{attn}k_norm.weight"), &[HEAD_DIM])?,
            k_norm_ctx: loader.bf16(&format!("{attn}k_norm_ctx.weight"), &[HEAD_DIM])?,
            o: loader.bf16(&format!("{attn}o_proj.weight"), &[HIDDEN, q])?,
            gate: loader.bf16(&format!("{p}mlp.gate_proj.weight"), &[MLP_WIDTH, HIDDEN])?,
            up: loader.bf16(&format!("{p}mlp.up_proj.weight"), &[MLP_WIDTH, HIDDEN])?,
            down: loader.bf16(&format!("{p}mlp.down_proj.weight"), &[HIDDEN, MLP_WIDTH])?,
        })
    }
}

impl MarkovWeights {
    fn load(loader: &Loader<'_>, vocab: usize) -> anyhow::Result<Self> {
        const STATE_COLS: usize = 2 * RANK;
        const COLS: usize = 2 * RANK + 2 * HIDDEN;
        let name = "markov_head.rnn_proj.weight";
        let (dtype, bytes) = loader.entry(name, &[3 * RANK, COLS])?;
        ensure!(
            dtype == safetensors::Dtype::BF16,
            "{name} is {dtype:?}, expected BF16"
        );
        let values: Vec<bf16> = bytes
            .as_chunks()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        let mut state = Vec::with_capacity(3 * RANK * STATE_COLS);
        let mut context = Vec::with_capacity(3 * RANK * (COLS - STATE_COLS));
        for row in values.as_chunks::<COLS>().0 {
            state.extend_from_slice(&row[..STATE_COLS]);
            context.extend_from_slice(&row[STATE_COLS..]);
        }
        Ok(Self {
            dflare_norm: loader.bf16("markov_head.dflare_norm.weight", &[HIDDEN])?,
            w1: loader.bf16("markov_head.markov_w1.weight", &[vocab, RANK])?,
            rnn_state: loader.upload(name, &state)?,
            rnn_context: loader.upload(name, &context)?,
            rnn_bias: loader.f32("markov_head.rnn_proj.bias", &[3 * RANK])?,
            hidden_out: loader.bf16("markov_head.hidden_out.weight", &[HIDDEN, RANK])?,
            hidden_out_bias: loader.bf16("markov_head.hidden_out.bias", &[HIDDEN])?,
        })
    }
}
