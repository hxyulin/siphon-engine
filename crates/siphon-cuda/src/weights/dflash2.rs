//! Device copies of Inco AI's DFlash 2 drafter for Inkling 2 Mini
//! (`incoai/Inkling-2-Mini-DFlash2`, `architectures: DFlash2DraftModel`).
//!
//! The drafter reads the same four taps as the bundled draft: the partial block entering
//! visits 5, 23 and 41 and the final normed hidden state (`target_layer_ids [5, 23, 41,
//! 48]`). `fc` and `hidden_norm` turn each position's taps into one context row, which
//! every layer projects with the same `k_proj` and `v_proj` as the block's rows. Its 5
//! Qwen3-style layers attend to the last 2048 positions and to the whole block, and each
//! sublayer is wrapped in a two-tap dynamic convolution over the block's rows. A
//! candidate selector picks one path through the top 16 tokens of each block row. The
//! drafter has no LM head or embedding: it reads the target's.

use std::{path::Path, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use serde_json::Value;
use siphon::{Checkpoint, TextConfig};

use super::Loader;
use super::dflash::tap_layout;
use super::draft::{HEAD_DIM, HIDDEN, KV_HEADS};
use crate::ops::spec::TapLayout;

/// Rows of a draft block: the anchor and 7 mask rows.
const BLOCK: usize = 8;
/// Query heads.
pub const DFLASH2_HEADS: usize = 32;
/// Width of the SiLU-gated MLP.
pub const DFLASH2_MLP_WIDTH: usize = 6912;
const _: () = assert!(
    DFLASH2_HEADS * HEAD_DIM == siphon::memory::DFLASH2_Q_WIDTH
        && DFLASH2_MLP_WIDTH == siphon::memory::DFLASH2_MLP_WIDTH
);
/// Channels that share one dynamic convolution coefficient.
pub const CONV_GROUP: usize = 16;
/// Taps of each convolution: the row itself and the row before it in the block.
pub const CONV_TAPS: usize = 2;
/// Outputs of a convolution's coefficient projection: 2 sides (input and output of the
/// sublayer) of 2 taps of 144 groups, in that order.
pub const CONV_COEFFICIENTS: usize = 2 * CONV_TAPS * HIDDEN / CONV_GROUP;
/// Positions each layer attends to, the row's own included (`sliding_window`).
pub const DFLASH2_WINDOW: usize = 2048;
/// Rank of the selector's codebooks.
pub const SELECTOR_RANK: usize = 256;
/// Candidates the selector keeps per block row.
pub const SELECTOR_TOP_K: usize = 16;
/// RoPE base (`rope_parameters.rope_theta`).
pub const DFLASH2_ROPE_THETA: f64 = 1e4;

/// The drafter's weights.
#[derive(Debug)]
pub struct Dflash2Weights {
    /// The target hidden states the drafter reads, in the order `fc` takes them.
    pub taps: TapLayout,
    /// The token whose normed target embedding fills rows 1 to 7 of a block.
    pub mask_token: usize,
    /// `[2304, taps * 2304]`.
    pub fc: CudaSlice<bf16>,
    pub hidden_norm: CudaSlice<bf16>,
    /// Final norm of the decoder.
    pub norm: CudaSlice<bf16>,
    pub layers: Vec<Dflash2LayerWeights>,
    pub selector: SelectorWeights,
}

/// One decoder layer. The context rows and the block rows share `k` and `v`.
#[derive(Debug)]
pub struct Dflash2LayerWeights {
    pub input_norm: CudaSlice<bf16>,
    pub post_attention_norm: CudaSlice<bf16>,
    /// `[4096, 2304]`.
    pub q: Option<CudaSlice<bf16>>,
    /// `[1024, 2304]` each.
    pub k: CudaSlice<bf16>,
    pub v: CudaSlice<bf16>,
    /// `[128]` each.
    pub q_norm: CudaSlice<bf16>,
    pub k_norm: CudaSlice<bf16>,
    /// `[2304, 4096]`.
    pub o: Option<CudaSlice<bf16>>,
    /// `[6912, 2304]` each.
    pub gate: Option<CudaSlice<bf16>>,
    pub up: Option<CudaSlice<bf16>>,
    /// `[2304, 6912]`.
    pub down: Option<CudaSlice<bf16>>,
    pub attention_conv: ConvWeights,
    pub mlp_conv: ConvWeights,
}

/// A grouped dynamic convolution around a sublayer. The coefficient of tap `j`, side `s`
/// and channel `c` in row `r` is `base[s, j, c] + delta[r, s, j, c / 16]`, where
/// `delta = projection · x` is computed from the sublayer's normed input `x`. Side 0
/// convolves the sublayer's input and side 1 its output.
#[derive(Debug)]
pub struct ConvWeights {
    /// `[2, 2, 2304]`.
    pub base: CudaSlice<bf16>,
    /// `[576, 2304]`.
    pub projection: CudaSlice<bf16>,
}

/// The candidate selector: the score of candidate `c` after token `t` adds
/// `<predecessor[t] * hidden_projection · y, successor[c]>` to `c`'s logit.
#[derive(Debug)]
pub struct SelectorWeights {
    /// `[256, 2304]`.
    pub hidden_projection: CudaSlice<bf16>,
    /// `[padded vocab, 256]` each.
    pub predecessor: CudaSlice<bf16>,
    pub successor: CudaSlice<bf16>,
}

impl Dflash2Weights {
    /// Loads the drafter in `dir` for the target `config`, after checking that its
    /// `config.json` describes a drafter these kernels run.
    pub fn load(dir: &Path, config: &TextConfig, stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let c: Value = serde_json::from_str(&text).context("parsing the drafter's config.json")?;
        let (layers, mask_token) = check_config(&c, config)?;
        let taps = tap_layout(&c, config)?;
        let checkpoint = Checkpoint::open(dir)?;
        let loader = Loader {
            checkpoint: &checkpoint,
            stream,
        };
        let vocab = config.vocab_size;
        Ok(Self {
            fc: loader.bf16("fc.weight", &[HIDDEN, taps.len() * HIDDEN])?,
            hidden_norm: loader.bf16("hidden_norm.weight", &[HIDDEN])?,
            norm: loader.bf16("norm.weight", &[HIDDEN])?,
            layers: (0..layers)
                .map(|l| Dflash2LayerWeights::load(&loader, l))
                .collect::<anyhow::Result<_>>()?,
            selector: SelectorWeights {
                hidden_projection: loader.bf16(
                    "candidate_selector.hidden_projection.weight",
                    &[SELECTOR_RANK, HIDDEN],
                )?,
                predecessor: loader.bf16(
                    "candidate_selector.predecessor_codebook",
                    &[vocab, SELECTOR_RANK],
                )?,
                successor: loader.bf16(
                    "candidate_selector.successor_codebook",
                    &[vocab, SELECTOR_RANK],
                )?,
            },
            taps,
            mask_token,
        })
    }
}

impl Dflash2LayerWeights {
    fn load(loader: &Loader<'_>, layer: usize) -> anyhow::Result<Self> {
        let p = format!("layers.{layer}.");
        let attn = format!("{p}self_attn.");
        let kv = KV_HEADS * HEAD_DIM;
        let q = DFLASH2_HEADS * HEAD_DIM;
        let conv = |name: &str| -> anyhow::Result<ConvWeights> {
            Ok(ConvWeights {
                base: loader.bf16(&format!("{p}{name}.base_kernel"), &[2, CONV_TAPS, HIDDEN])?,
                projection: loader.bf16(
                    &format!("{p}{name}.kernel_projection.weight"),
                    &[CONV_COEFFICIENTS, HIDDEN],
                )?,
            })
        };
        Ok(Self {
            input_norm: loader.bf16(&format!("{p}input_layernorm.weight"), &[HIDDEN])?,
            post_attention_norm: loader
                .bf16(&format!("{p}post_attention_layernorm.weight"), &[HIDDEN])?,
            q: Some(loader.bf16(&format!("{attn}q_proj.weight"), &[q, HIDDEN])?),
            k: loader.bf16(&format!("{attn}k_proj.weight"), &[kv, HIDDEN])?,
            v: loader.bf16(&format!("{attn}v_proj.weight"), &[kv, HIDDEN])?,
            q_norm: loader.bf16(&format!("{attn}q_norm.weight"), &[HEAD_DIM])?,
            k_norm: loader.bf16(&format!("{attn}k_norm.weight"), &[HEAD_DIM])?,
            o: Some(loader.bf16(&format!("{attn}o_proj.weight"), &[HIDDEN, q])?),
            gate: Some(loader.bf16(
                &format!("{p}mlp.gate_proj.weight"),
                &[DFLASH2_MLP_WIDTH, HIDDEN],
            )?),
            up: Some(loader.bf16(
                &format!("{p}mlp.up_proj.weight"),
                &[DFLASH2_MLP_WIDTH, HIDDEN],
            )?),
            down: Some(loader.bf16(
                &format!("{p}mlp.down_proj.weight"),
                &[HIDDEN, DFLASH2_MLP_WIDTH],
            )?),
            attention_conv: conv("attention_conv")?,
            mlp_conv: conv("mlp_conv")?,
        })
    }
}

/// Whether the drafter in `dir` is a DFlash 2 drafter: its `config.json` names
/// `DFlash2DraftModel` as its architecture.
pub fn is_dflash2(dir: &Path) -> anyhow::Result<bool> {
    let path = dir.join("config.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let c: Value = serde_json::from_str(&text).context("parsing the drafter's config.json")?;
    Ok(c.pointer("/architectures/0").and_then(Value::as_str) == Some("DFlash2DraftModel"))
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

/// Checks the drafter's `config.json` against what the kernels run, and returns its layer
/// count and mask token.
fn check_config(c: &Value, target: &TextConfig) -> anyhow::Result<(usize, usize)> {
    let arch = c.pointer("/architectures/0").and_then(Value::as_str);
    ensure!(
        arch == Some("DFlash2DraftModel"),
        "the drafter's architecture is {arch:?}, expected DFlash2DraftModel"
    );
    for (pointer, want) in [
        ("/hidden_size", HIDDEN),
        ("/num_attention_heads", DFLASH2_HEADS),
        ("/num_key_value_heads", KV_HEADS),
        ("/head_dim", HEAD_DIM),
        ("/intermediate_size", DFLASH2_MLP_WIDTH),
        ("/sliding_window", DFLASH2_WINDOW),
        ("/vocab_size", target.vocab_size),
        ("/dflash_config/block_size", BLOCK),
        ("/dflash_config/conv_group_size", CONV_GROUP),
        ("/dflash_config/conv_kernel_size", CONV_TAPS),
        ("/dflash_config/selector_rank", SELECTOR_RANK),
        ("/dflash_config/selector_top_k", SELECTOR_TOP_K),
    ] {
        let got = usize_at(c, pointer)?;
        ensure!(
            got == want,
            "the drafter's {pointer} is {got}, expected {want}"
        );
    }
    ensure!(
        target.hidden_size == HIDDEN,
        "the drafter expects hidden size {HIDDEN}, the target has {}",
        target.hidden_size
    );
    #[expect(clippy::float_cmp, reason = "the configs store these exact values")]
    {
        ensure!(
            f64_at(c, "/rope_parameters/rope_theta")? == DFLASH2_ROPE_THETA,
            "the drafter's rope_theta is not {DFLASH2_ROPE_THETA}"
        );
        ensure!(
            c.pointer("/rope_parameters/rope_type")
                .and_then(Value::as_str)
                == Some("default"),
            "only the default RoPE is supported"
        );
        ensure!(
            f64_at(c, "/rms_norm_eps")? == 1e-6,
            "the drafter's rms_norm_eps is not 1e-6"
        );
        let softcap = f64_at(c, "/dflash_config/final_logit_softcapping")?;
        ensure!(
            softcap == f64::from(target.final_logit_softcapping),
            "the drafter's softcap {softcap} differs from the target's {}",
            target.final_logit_softcapping
        );
    }
    ensure!(
        c.pointer("/dflash_config/selector_softcap_mode")
            .and_then(Value::as_str)
            == Some("after_correction"),
        "only selector_softcap_mode after_correction is supported"
    );
    ensure!(
        c.get("is_causal").and_then(Value::as_bool) == Some(false),
        "only drafters whose block attends to itself in both directions are supported"
    );
    ensure!(
        c.pointer("/dflash_config/output_multiplier")
            .and_then(Value::as_f64)
            .is_none_or(|m| (m - 1.0).abs() < f64::EPSILON),
        "only drafters without an output multiplier are supported"
    );
    let layers = usize_at(c, "/num_hidden_layers")?;
    ensure!(
        (1..=16).contains(&layers),
        "the drafter has {layers} layers, expected 1 to 16"
    );
    let sliding = c
        .get("layer_types")
        .and_then(Value::as_array)
        .is_some_and(|t| t.len() == layers && t.iter().all(|x| x == "sliding_attention"));
    ensure!(
        sliding,
        "every layer of the drafter must use sliding attention"
    );
    let mask_token = usize_at(c, "/dflash_config/mask_token_id")?;
    ensure!(
        mask_token < target.unpadded_vocab_size,
        "the mask token {mask_token} is outside the vocabulary"
    );
    Ok((layers, mask_token))
}
