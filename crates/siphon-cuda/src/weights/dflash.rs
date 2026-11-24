//! Device copies of a DFlash-style draft with DSpark heads, as the 2DSD project exports it
//! (`architectures: DFlashDraftModel` with a `loopedtf_drafter.dspark` section).
//!
//! The export directory holds `config.json`, `model.safetensors` and the learned mask
//! vector as `mask_embedding.safetensors` (one BF16 tensor `mask_embedding`, `[2304]`),
//! converted once from the export's `mask_embedding.pt`. The draft concatenates its
//! taps, projects them with `fc` and normalizes them with `hidden_norm` into one context
//! row per position, which every layer projects with its own `k_proj` and `v_proj`, as
//! the block's rows. Its Markov head is the vanilla one: the draft token after token `t`
//! gets the logit bias `markov_w2 · markov_w1[t]`. Its confidence head reads the block
//! row and the previous token's `markov_w1` row. It shares the target's embedding,
//! `embed_norm` and LM head.

use std::{path::Path, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use serde_json::Value;
use siphon::{Checkpoint, TextConfig};

use super::Loader;
use super::draft::{HEAD_DIM, HEADS, HIDDEN, KV_HEADS, MLP_WIDTH, RANK};
use crate::ops::spec::{MAX_TAPS, TapLayout};

/// Rows of a draft block, the anchor and 7 mask rows.
const BLOCK: usize = 8;

/// The draft's weights.
#[derive(Debug)]
pub struct DflashWeights {
    /// The target hidden states the draft reads, in the order `fc` takes them.
    pub taps: TapLayout,
    /// The learned mask vector, `[2304]`, which stands for the normed embedding of rows 1
    /// to 7 of a block.
    pub mask_embed: CudaSlice<bf16>,
    /// `[2304, taps * 2304]`.
    pub fc: CudaSlice<bf16>,
    pub hidden_norm: CudaSlice<bf16>,
    /// Final norm of the decoder.
    pub norm: CudaSlice<bf16>,
    pub layers: Vec<DflashLayerWeights>,
    /// The previous token's Markov embedding, `[vocab, 256]`.
    pub w1: CudaSlice<bf16>,
    /// The Markov bias of each token, `[vocab, 256]`.
    pub w2: CudaSlice<bf16>,
    /// `confidence_head.proj.weight`, `[2304 + 256]`, over `[block row; w1 row]`.
    pub confidence: CudaSlice<bf16>,
    /// `confidence_head.proj.bias`, a BF16 value widened to F32.
    pub confidence_bias: f32,
}

/// One decoder layer. The context rows and the block rows share `k` and `v`.
#[derive(Debug)]
pub struct DflashLayerWeights {
    pub input_norm: CudaSlice<bf16>,
    pub post_attention_norm: CudaSlice<bf16>,
    /// `[3072, 2304]`.
    pub q: CudaSlice<bf16>,
    /// `[1024, 2304]` each.
    pub k: CudaSlice<bf16>,
    pub v: CudaSlice<bf16>,
    /// `[128]` each.
    pub q_norm: CudaSlice<bf16>,
    pub k_norm: CudaSlice<bf16>,
    /// `[2304, 3072]`.
    pub o: CudaSlice<bf16>,
    /// `[9216, 2304]` each.
    pub gate: CudaSlice<bf16>,
    pub up: CudaSlice<bf16>,
    /// `[2304, 9216]`.
    pub down: CudaSlice<bf16>,
}

impl DflashWeights {
    /// Loads the export in `dir` for the target `config`, after checking that its
    /// `config.json` describes a draft these kernels run: the target's widths, blocks of
    /// 8 with the anchor in row 0, full attention, a vanilla Markov head of rank 256 and
    /// a confidence head over the block row and the Markov embedding.
    pub fn load(dir: &Path, config: &TextConfig, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let c: Value = serde_json::from_str(&text).context("parsing the draft's config.json")?;
        let layers = check_config(&c, config)?;
        let taps = tap_layout(&c, config)?;
        let checkpoint = Checkpoint::open(dir)?;
        let loader = Loader {
            checkpoint: &checkpoint,
            stream,
        };
        let vocab = config.unpadded_vocab_size;
        let fc = loader.bf16("fc.weight", &[HIDDEN, taps.len() * HIDDEN])?;
        let bias = loader.f32_host("confidence_head.proj.bias", &[1])?;
        Ok(Self {
            mask_embed: loader.bf16("mask_embedding", &[HIDDEN])?,
            fc,
            hidden_norm: loader.bf16("hidden_norm.weight", &[HIDDEN])?,
            norm: loader.bf16("norm.weight", &[HIDDEN])?,
            layers: (0..layers)
                .map(|l| DflashLayerWeights::load(&loader, l))
                .collect::<anyhow::Result<_>>()?,
            w1: loader.bf16("markov_head.markov_w1.weight", &[vocab, RANK])?,
            w2: loader.bf16("markov_head.markov_w2.weight", &[vocab, RANK])?,
            confidence: loader.bf16("confidence_head.proj.weight", &[1, HIDDEN + RANK])?,
            confidence_bias: bias[0],
            taps,
        })
    }
}

impl DflashLayerWeights {
    fn load(loader: &Loader<'_>, layer: usize) -> anyhow::Result<Self> {
        let p = format!("layers.{layer}.");
        let attn = format!("{p}self_attn.");
        let kv = KV_HEADS * HEAD_DIM;
        let q = HEADS * HEAD_DIM;
        Ok(Self {
            input_norm: loader.bf16(&format!("{p}input_layernorm.weight"), &[HIDDEN])?,
            post_attention_norm: loader
                .bf16(&format!("{p}post_attention_layernorm.weight"), &[HIDDEN])?,
            q: loader.bf16(&format!("{attn}q_proj.weight"), &[q, HIDDEN])?,
            k: loader.bf16(&format!("{attn}k_proj.weight"), &[kv, HIDDEN])?,
            v: loader.bf16(&format!("{attn}v_proj.weight"), &[kv, HIDDEN])?,
            q_norm: loader.bf16(&format!("{attn}q_norm.weight"), &[HEAD_DIM])?,
            k_norm: loader.bf16(&format!("{attn}k_norm.weight"), &[HEAD_DIM])?,
            o: loader.bf16(&format!("{attn}o_proj.weight"), &[HIDDEN, q])?,
            gate: loader.bf16(&format!("{p}mlp.gate_proj.weight"), &[MLP_WIDTH, HIDDEN])?,
            up: loader.bf16(&format!("{p}mlp.up_proj.weight"), &[MLP_WIDTH, HIDDEN])?,
            down: loader.bf16(&format!("{p}mlp.down_proj.weight"), &[HIDDEN, MLP_WIDTH])?,
        })
    }
}

fn usize_at(c: &Value, pointer: &str) -> anyhow::Result<usize> {
    let v = c
        .pointer(pointer)
        .and_then(Value::as_u64)
        .with_context(|| format!("config.json lacks the integer {pointer}"))?;
    Ok(usize::try_from(v)?)
}

fn f64_at(c: &Value, pointer: &str) -> anyhow::Result<f64> {
    c.pointer(pointer)
        .and_then(Value::as_f64)
        .with_context(|| format!("config.json lacks the number {pointer}"))
}

/// Checks the draft's `config.json` against what the kernels run, and returns its layer
/// count.
fn check_config(c: &Value, target: &TextConfig) -> anyhow::Result<usize> {
    let arch = c.pointer("/architectures/0").and_then(Value::as_str);
    ensure!(
        arch == Some("DFlashDraftModel"),
        "the draft's architecture is {arch:?}, expected DFlashDraftModel"
    );
    for (pointer, want) in [
        ("/hidden_size", HIDDEN),
        ("/num_attention_heads", HEADS),
        ("/num_key_value_heads", KV_HEADS),
        ("/head_dim", HEAD_DIM),
        ("/intermediate_size", MLP_WIDTH),
        ("/block_size", BLOCK),
        ("/vocab_size", target.unpadded_vocab_size),
        ("/loopedtf_drafter/dspark/markov_rank", RANK),
    ] {
        let got = usize_at(c, pointer)?;
        ensure!(
            got == want,
            "the draft's {pointer} is {got}, expected {want}"
        );
    }
    ensure!(
        target.hidden_size == HIDDEN,
        "the draft expects hidden size {HIDDEN}, the target has {}",
        target.hidden_size
    );
    #[expect(clippy::float_cmp, reason = "the configs store these exact values")]
    {
        ensure!(
            f64_at(c, "/rope_theta")? == 1e6,
            "the draft's rope_theta is not 1e6"
        );
        ensure!(
            f64_at(c, "/rms_norm_eps")? == 1e-6,
            "the draft's rms_norm_eps is not 1e-6"
        );
    }
    let dspark = c
        .pointer("/loopedtf_drafter/dspark")
        .context("the draft has no DSpark heads (loopedtf_drafter.dspark)")?;
    ensure!(
        dspark.get("markov_head_type").and_then(Value::as_str) == Some("vanilla"),
        "only the vanilla Markov head is supported"
    );
    ensure!(
        dspark.get("sample_from_anchor").and_then(Value::as_bool) == Some(false)
            && dspark
                .get("confidence_head_with_markov")
                .and_then(Value::as_bool)
                == Some(true),
        "only drafts whose anchor row does not predict, with the confidence head over the \
         Markov embedding, are supported"
    );
    let layers = usize_at(c, "/num_hidden_layers")?;
    ensure!(
        (1..=16).contains(&layers),
        "the draft has {layers} layers, expected 1 to 16"
    );
    let full = c
        .get("layer_types")
        .and_then(Value::as_array)
        .is_some_and(|t| t.len() == layers && t.iter().all(|x| x == "full_attention"));
    ensure!(full, "every layer of the draft must use full attention");
    Ok(layers)
}

/// The taps of `dflash_config.target_layer_ids`: visit ids of the target, whose input
/// partial block is the tap, or the visit count for the final normed hidden state, last.
pub(super) fn tap_layout(c: &Value, target: &TextConfig) -> anyhow::Result<TapLayout> {
    let ids = c
        .pointer("/dflash_config/target_layer_ids")
        .and_then(Value::as_array)
        .context("config.json lacks dflash_config.target_layer_ids")?;
    let visits = target.layer_visits();
    let mut layout = TapLayout {
        visits: Vec::new(),
        final_hidden: false,
    };
    for (i, id) in ids.iter().enumerate() {
        let id = usize::try_from(id.as_u64().context("a target layer id is not an integer")?)?;
        if id == visits && i + 1 == ids.len() {
            layout.final_hidden = true;
        } else {
            ensure!(
                id < visits,
                "target layer id {id} is past the {visits} visits"
            );
            layout.visits.push(id);
        }
    }
    ensure!(
        (1..=MAX_TAPS).contains(&layout.len()),
        "{} taps, expected 1 to {MAX_TAPS}",
        layout.len()
    );
    Ok(layout)
}
