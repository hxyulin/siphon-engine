//! Device copies of the checkpoint's text-model weights.
//!
//! Tensor names are the checkpoint's own (`model.llm.*`). The conversions mirror
//! MLX-LM's `sanitize` in `models/inkling2/checkpoint.py`:
//! - the router weight, the message gates' forget and gain weights and the dynamic
//!   convolutions' static weights are widened from BF16 to F32;
//! - the NVFP4 global scale of each expert is `amax * 6 * 448 / denominator`;
//! - the loop-message queries are reordered to query, key, value, MLP per looped layer.
//!
//! The depthwise convolution weights need no transpose: the checkpoint's `[dims, 1, 4]`
//! and MLX's `[dims, 4, 1]` have the same bytes.

use std::sync::Arc;

use anyhow::{Context as _, bail, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
use half::bf16;
use inkling::{Checkpoint, TextConfig};
use safetensors::Dtype;

/// Largest E2M1 value, from MLX-LM's `sanitize`.
const NVFP4_E2M1_MAX: f32 = 6.0;
/// Largest E4M3 value, from MLX-LM's `sanitize`.
const NVFP4_E4M3_MAX: f32 = 448.0;

#[derive(Debug)]
pub struct Weights {
    /// `[vocab, hidden]`.
    pub embed: CudaSlice<bf16>,
    pub embed_norm: CudaSlice<bf16>,
    pub layers: Vec<LayerWeights>,
    /// Query of the final depth attention, F32 `[hidden]`.
    pub final_query: CudaSlice<f32>,
    pub norm: CudaSlice<bf16>,
    /// `[vocab, hidden]`.
    pub lm_head: CudaSlice<bf16>,
    /// Per looped layer, F32 `[4, hidden]`: the query, key, value and MLP message queries.
    pub message_queries: Vec<CudaSlice<f32>>,
}

#[derive(Debug)]
pub struct LayerWeights {
    /// Depth-attention queries for the attention input, F32 `[1 or 3, hidden]`.
    /// Layer 0 has none.
    pub attention_query: Option<CudaSlice<f32>>,
    /// Depth-attention query for the MLP input, F32 `[hidden]`.
    pub mlp_query: CudaSlice<f32>,
    pub input_norms: InputNorms,
    pub attention: AttentionWeights,
    /// Depthwise conv weights `[hidden, 4]`.
    pub post_attention_conv: CudaSlice<bf16>,
    pub post_mlp_conv: CudaSlice<bf16>,
    pub mlp_norm: CudaSlice<bf16>,
    /// F32 `[experts, hidden]`.
    pub router: CudaSlice<f32>,
    /// F32 `[experts]`.
    pub router_bias: CudaSlice<f32>,
    pub experts: ExpertWeights,
    /// Query, key, value and MLP gates, present on looped layers.
    pub message_gates: Option<[MessageGateWeights; 4]>,
}

#[derive(Debug)]
pub enum InputNorms {
    /// Sliding layers normalize one input for queries, keys and values.
    Shared(CudaSlice<bf16>),
    /// Global layers normalize three separate inputs.
    Separate {
        q: CudaSlice<bf16>,
        k: CudaSlice<bf16>,
        v: CudaSlice<bf16>,
    },
}

#[derive(Debug)]
pub struct AttentionWeights {
    pub q: CudaSlice<bf16>,
    pub k: CudaSlice<bf16>,
    pub v: CudaSlice<bf16>,
    /// Relative-feature projection `[heads * d_rel, hidden]`.
    pub r: CudaSlice<bf16>,
    pub o: CudaSlice<bf16>,
    pub gate: CudaSlice<bf16>,
    pub sink: CudaSlice<bf16>,
    /// `[heads, head_dim]`.
    pub q_clip_scale: CudaSlice<bf16>,
    /// `[heads, d_rel]`.
    pub rel_clip_scale: CudaSlice<bf16>,
    /// `[d_rel, rel_extent]`.
    pub rel_table: CudaSlice<bf16>,
    pub kv_conv: KvConv,
}

#[derive(Debug)]
pub enum KvConv {
    /// Sliding layers: depthwise conv weights `[kv_dims, 4]` for keys and values.
    Static {
        k: CudaSlice<bf16>,
        v: CudaSlice<bf16>,
    },
    /// Global layers: F32 static weights `[4, kv_dims]` plus a BF16 projection
    /// `[kv_heads * 4, hidden]` to per-token weights, for keys and values.
    Dynamic {
        k_static: CudaSlice<f32>,
        k_proj: CudaSlice<bf16>,
        v_static: CudaSlice<f32>,
        v_proj: CudaSlice<bf16>,
    },
}

/// One layer's 128 experts as raw checkpoint bytes.
#[derive(Debug)]
pub struct ExpertWeights {
    pub format: ExpertFormat,
    /// Gate and up rows interleaved, `[experts, 2 * width, hidden]` in the packed format.
    pub w13: CudaSlice<u8>,
    pub w13_scale: CudaSlice<u8>,
    /// `[experts, hidden, width]` in the packed format.
    pub w2: CudaSlice<u8>,
    pub w2_scale: CudaSlice<u8>,
}

#[derive(Debug)]
pub enum ExpertFormat {
    /// E2M1 pairs per byte, E4M3 scales per 16 values, and an F32 global scale per expert.
    Nvfp4 {
        w13_global_scale: CudaSlice<f32>,
        w2_global_scale: CudaSlice<f32>,
    },
    /// E4M3 values with E8M0 scales per 32 values.
    Mxfp8,
}

#[derive(Debug)]
pub struct MessageGateWeights {
    /// `[rank, 2 * hidden]`.
    pub down: CudaSlice<bf16>,
    /// F32 `[hidden, rank]`.
    pub forget: CudaSlice<f32>,
    /// F32 `[hidden, rank]`.
    pub gain: CudaSlice<f32>,
}

/// Reads tensors from the checkpoint and uploads them.
struct Loader<'a> {
    checkpoint: &'a Checkpoint,
    stream: &'a Arc<CudaStream>,
}

impl Loader<'_> {
    fn entry(&self, name: &str, shape: &[usize]) -> anyhow::Result<(Dtype, &[u8])> {
        let entry = self
            .checkpoint
            .get(name)
            .with_context(|| format!("{name} missing from the checkpoint"))?;
        ensure!(
            entry.shape == shape,
            "{name} has shape {:?}, expected {shape:?}",
            entry.shape
        );
        Ok((entry.dtype, self.checkpoint.bytes(entry)))
    }

    fn upload<T: DeviceRepr>(&self, name: &str, values: &[T]) -> anyhow::Result<CudaSlice<T>> {
        self.stream
            .clone_htod(values)
            .with_context(|| format!("uploading {name}"))
    }

    fn bf16(&self, name: &str, shape: &[usize]) -> anyhow::Result<CudaSlice<bf16>> {
        let (dtype, bytes) = self.entry(name, shape)?;
        ensure!(dtype == Dtype::BF16, "{name} is {dtype:?}, expected BF16");
        let values: Vec<bf16> = bytes
            .as_chunks()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        self.upload(name, &values)
    }

    /// Loads an F32 tensor, widening it first if the checkpoint stores BF16.
    fn f32(&self, name: &str, shape: &[usize]) -> anyhow::Result<CudaSlice<f32>> {
        let values = self.f32_host(name, shape)?;
        self.upload(name, &values)
    }

    fn f32_host(&self, name: &str, shape: &[usize]) -> anyhow::Result<Vec<f32>> {
        let (dtype, bytes) = self.entry(name, shape)?;
        Ok(match dtype {
            Dtype::F32 => bytes
                .as_chunks()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            Dtype::BF16 => bytes
                .as_chunks()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b).to_f32())
                .collect(),
            other => bail!("{name} is {other:?}, expected F32 or BF16"),
        })
    }

    fn raw(&self, name: &str, dtype: Dtype, shape: &[usize]) -> anyhow::Result<CudaSlice<u8>> {
        let (found, bytes) = self.entry(name, shape)?;
        ensure!(found == dtype, "{name} is {found:?}, expected {dtype:?}");
        self.upload(name, bytes)
    }
}

impl Weights {
    /// Uploads every text-model weight. Takes about 19 GB of device memory.
    pub fn load(
        checkpoint: &Checkpoint,
        config: &TextConfig,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<Self> {
        let loader = Loader { checkpoint, stream };
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let layers = (0..config.num_hidden_layers)
            .map(|layer| {
                LayerWeights::load(&loader, config, layer)
                    .with_context(|| format!("loading layer {layer}"))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self {
            embed: loader.bf16("model.llm.embed.weight", &[vocab, hidden])?,
            embed_norm: loader.bf16("model.llm.embed_norm.weight", &[hidden])?,
            layers,
            final_query: loader.f32("model.llm.attn_res.output_query", &[hidden])?,
            norm: loader.bf16("model.llm.norm.weight", &[hidden])?,
            lm_head: loader.bf16("model.llm.unembed.weight", &[vocab, hidden])?,
            message_queries: load_message_queries(&loader, config)?,
        })
    }
}

/// The checkpoint stores `[query, mlp]` and `[key, value]` rows per looped layer; MLX-LM
/// and the runner use `[query, key, value, mlp]`.
fn load_message_queries(
    loader: &Loader<'_>,
    config: &TextConfig,
) -> anyhow::Result<Vec<CudaSlice<f32>>> {
    let hidden = config.hidden_size;
    let looped = config.loop_config.core_end - config.loop_config.core_start;
    let shape = [2 * looped, hidden];
    let q = loader.f32_host("model.llm.middle_cycle_cross_pass_queries", &shape)?;
    let kv = loader.f32_host("model.llm.middle_cycle_cross_pass_kv_queries", &shape)?;
    let row = |rows: &[f32], layer: usize, i: usize| {
        let start = (2 * layer + i) * hidden;
        rows[start..start + hidden].to_vec()
    };
    (0..looped)
        .map(|layer| {
            let ordered = [
                row(&q, layer, 0),
                row(&kv, layer, 0),
                row(&kv, layer, 1),
                row(&q, layer, 1),
            ]
            .concat();
            loader.upload("message queries", &ordered)
        })
        .collect()
}

impl LayerWeights {
    fn load(loader: &Loader<'_>, config: &TextConfig, layer: usize) -> anyhow::Result<Self> {
        let hidden = config.hidden_size;
        let sliding = config.is_sliding(layer);
        let p = format!("model.llm.layers.{layer}");
        let conv_k = config.sconv_kernel_size;

        let attention_query = match layer {
            0 => None,
            _ => {
                let name = format!("model.llm.attn_res.attn_queries.{}", layer - 1);
                let shape: &[usize] = if sliding { &[hidden] } else { &[3, hidden] };
                Some(loader.f32(&name, shape)?)
            }
        };
        let input_norms = if sliding {
            InputNorms::Shared(loader.bf16(&format!("{p}.attn_norm.weight"), &[hidden])?)
        } else {
            InputNorms::Separate {
                q: loader.bf16(&format!("{p}.q_input_norm.weight"), &[hidden])?,
                k: loader.bf16(&format!("{p}.k_input_norm.weight"), &[hidden])?,
                v: loader.bf16(&format!("{p}.v_input_norm.weight"), &[hidden])?,
            }
        };
        let looped = (config.loop_config.core_start..config.loop_config.core_end).contains(&layer);
        let message_gates = if looped {
            let gate = |name: &str| {
                MessageGateWeights::load(loader, config, &format!("{p}.controllers.{name}"))
            };
            Some([gate("attn")?, gate("k")?, gate("v")?, gate("mlp")?])
        } else {
            None
        };

        Ok(Self {
            attention_query,
            mlp_query: loader.f32(
                &format!("model.llm.attn_res.mlp_queries.{layer}"),
                &[hidden],
            )?,
            input_norms,
            attention: AttentionWeights::load(loader, config, &p, sliding)?,
            post_attention_conv: loader
                .bf16(&format!("{p}.attn_sconv.weight"), &[hidden, 1, conv_k])?,
            post_mlp_conv: loader.bf16(&format!("{p}.mlp_sconv.weight"), &[hidden, 1, conv_k])?,
            mlp_norm: loader.bf16(&format!("{p}.mlp_norm.weight"), &[hidden])?,
            router: loader.f32(
                &format!("{p}.mlp.gate.weight"),
                &[config.n_routed_experts, hidden],
            )?,
            router_bias: loader.f32(&format!("{p}.mlp.gate.bias"), &[config.n_routed_experts])?,
            experts: ExpertWeights::load(loader, config, &format!("{p}.mlp.experts"))?,
            message_gates,
        })
    }
}

impl AttentionWeights {
    fn load(
        loader: &Loader<'_>,
        config: &TextConfig,
        p: &str,
        sliding: bool,
    ) -> anyhow::Result<Self> {
        let hidden = config.hidden_size;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let conv_k = config.sconv_kernel_size;
        let a = format!("{p}.attn");
        let kv_conv = if sliding {
            KvConv::Static {
                k: loader.bf16(&format!("{a}.k_sconv.weight"), &[kv_dim, 1, conv_k])?,
                v: loader.bf16(&format!("{a}.v_sconv.weight"), &[kv_dim, 1, conv_k])?,
            }
        } else {
            let proj_shape = [kv_heads * conv_k, hidden];
            KvConv::Dynamic {
                k_static: loader.f32(&format!("{a}.k_sconv.static_w"), &[conv_k, kv_dim])?,
                k_proj: loader.bf16(&format!("{a}.k_sconv.weight_proj"), &proj_shape)?,
                v_static: loader.f32(&format!("{a}.v_sconv.static_w"), &[conv_k, kv_dim])?,
                v_proj: loader.bf16(&format!("{a}.v_sconv.weight_proj"), &proj_shape)?,
            }
        };
        Ok(Self {
            q: loader.bf16(&format!("{a}.wq_du.weight"), &[q_dim, hidden])?,
            k: loader.bf16(&format!("{a}.wk_dv.weight"), &[kv_dim, hidden])?,
            v: loader.bf16(&format!("{a}.wv_dv.weight"), &[kv_dim, hidden])?,
            r: loader.bf16(
                &format!("{a}.wr_du.weight"),
                &[heads * config.d_rel, hidden],
            )?,
            o: loader.bf16(&format!("{a}.wo_ud.weight"), &[hidden, q_dim])?,
            gate: loader.bf16(&format!("{a}.gate_attn.weight"), &[q_dim, hidden])?,
            sink: loader.bf16(&format!("{a}.attn_sink_proj.weight"), &[heads, hidden])?,
            q_clip_scale: loader.bf16(&format!("{a}.q_l2_clip.scale"), &[heads, head_dim])?,
            rel_clip_scale: loader
                .bf16(&format!("{a}.rel_l2_clip.scale"), &[heads, config.d_rel])?,
            rel_table: loader.bf16(
                &format!("{a}.rel_logits_proj.proj"),
                &[config.d_rel, config.rel_extent],
            )?,
            kv_conv,
        })
    }
}

impl ExpertWeights {
    fn load(loader: &Loader<'_>, config: &TextConfig, p: &str) -> anyhow::Result<Self> {
        let experts = config.n_routed_experts;
        let hidden = config.hidden_size;
        let width = config.intermediate_size;
        let name = format!("{p}.w13_weight");
        let entry = loader
            .checkpoint
            .get(&name)
            .with_context(|| format!("{name} missing from the checkpoint"))?;
        match entry.dtype {
            Dtype::U8 => {
                // Two E2M1 values per byte and one E4M3 scale per 16 values.
                let global_scale = |w: &str| -> anyhow::Result<CudaSlice<f32>> {
                    let amax = loader.f32_host(&format!("{p}.{w}_weight_amax"), &[experts])?;
                    let denominator = loader
                        .f32_host(&format!("{p}.{w}_weight_outer_scale_denominator"), &[1])?[0];
                    let scale = NVFP4_E2M1_MAX * NVFP4_E4M3_MAX / denominator;
                    let values: Vec<f32> = amax.iter().map(|a| a * scale).collect();
                    loader.upload("global scale", &values)
                };
                Ok(Self {
                    format: ExpertFormat::Nvfp4 {
                        w13_global_scale: global_scale("w13")?,
                        w2_global_scale: global_scale("w2")?,
                    },
                    w13: loader.raw(&name, Dtype::U8, &[experts, 2 * width, hidden / 2])?,
                    w13_scale: loader.raw(
                        &format!("{p}.w13_weight_scale"),
                        Dtype::F8_E4M3,
                        &[experts, 2 * width, hidden / 16],
                    )?,
                    w2: loader.raw(
                        &format!("{p}.w2_weight"),
                        Dtype::U8,
                        &[experts, hidden, width / 2],
                    )?,
                    w2_scale: loader.raw(
                        &format!("{p}.w2_weight_scale"),
                        Dtype::F8_E4M3,
                        &[experts, hidden, width / 16],
                    )?,
                })
            }
            Dtype::F8_E4M3 => Ok(Self {
                // One E8M0 scale per 32 values.
                format: ExpertFormat::Mxfp8,
                w13: loader.raw(&name, Dtype::F8_E4M3, &[experts, 2 * width, hidden])?,
                w13_scale: loader.raw(
                    &format!("{p}.w13_weight_scale"),
                    Dtype::U8,
                    &[experts, 2 * width, hidden / 32],
                )?,
                w2: loader.raw(
                    &format!("{p}.w2_weight"),
                    Dtype::F8_E4M3,
                    &[experts, hidden, width],
                )?,
                w2_scale: loader.raw(
                    &format!("{p}.w2_weight_scale"),
                    Dtype::U8,
                    &[experts, hidden, width / 32],
                )?,
            }),
            other => bail!("{name} is {other:?}; expected NVFP4 (U8) or MXFP8 (F8_E4M3)"),
        }
    }
}

impl MessageGateWeights {
    fn load(loader: &Loader<'_>, config: &TextConfig, p: &str) -> anyhow::Result<Self> {
        let hidden = config.hidden_size;
        let rank = config.loop_config.gate_rank;
        Ok(Self {
            down: loader.bf16(&format!("{p}.down_weight"), &[rank, 2 * hidden])?,
            forget: loader.f32(&format!("{p}.forget_weight"), &[hidden, rank])?,
            gain: loader.f32(&format!("{p}.gain_weight"), &[hidden, rank])?,
        })
    }
}
