use std::{fs, path::Path};

use anyhow::Context as _;
use serde::Deserialize;

/// Text-model hyperparameters read from the checkpoint's `config.json`.
///
/// Only the fields the engine uses are listed; serde ignores the rest.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    /// Per-expert FFN width.
    pub intermediate_size: usize,
    pub sliding_window_size: usize,
    /// Layers that use sliding-window attention; the rest attend globally.
    pub local_layer_ids: Vec<usize>,
    pub final_logit_softcapping: f32,
    pub loop_config: LoopConfig,
}

/// The block of layers `core_start..core_end` runs `num_passes` times with shared weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
}

#[derive(Deserialize)]
struct ConfigFile {
    text_config: TextConfig,
}

impl TextConfig {
    /// Reads `text_config` from `<model_dir>/config.json`.
    pub fn load(model_dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let file: ConfigFile =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Ok(file.text_config)
    }

    /// Decoder-layer visits per token, counting each pass through the looped core.
    #[must_use]
    pub fn layer_visits(&self) -> usize {
        let core = self.loop_config.core_end - self.loop_config.core_start;
        self.num_hidden_layers + core * (self.loop_config.num_passes - 1)
    }
}
