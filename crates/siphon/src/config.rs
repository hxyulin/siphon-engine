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
    /// Width of the per-head relative-position features.
    pub d_rel: usize,
    /// Distances with a learned relative bias; the bias is zero at and beyond it.
    pub rel_extent: usize,
    /// Global layers scale queries up past this position.
    pub log_scaling_n_floor: usize,
    pub log_scaling_alpha: f32,
    pub qk_l2_clip_threshold: f32,
    pub rel_l2_clip_threshold: f32,
    pub sconv_kernel_size: usize,
    /// A new block state starts at every layer index divisible by this.
    pub n_layers_per_attn_res_block: usize,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
    pub route_scale: f32,
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
    /// Rank of the message gates' down projection.
    pub gate_rank: usize,
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
        self.visit_layers(self.loop_config.num_passes).len()
    }

    /// The layer index of each visit when the looped core runs `passes` times.
    #[must_use]
    pub fn visit_layers(&self, passes: usize) -> Vec<usize> {
        let LoopConfig {
            core_start,
            core_end,
            ..
        } = self.loop_config;
        let core = (0..passes).flat_map(|_| core_start..core_end);
        (0..core_start)
            .chain(core)
            .chain(core_end..self.num_hidden_layers)
            .collect()
    }

    /// Whether `layer` uses sliding-window attention.
    #[must_use]
    pub fn is_sliding(&self, layer: usize) -> bool {
        self.local_layer_ids.contains(&layer)
    }
}
