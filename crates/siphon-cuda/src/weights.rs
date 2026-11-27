//! Device copies of the checkpoint's text-model weights.
//!
//! Tensor names are the checkpoint's own (`model.llm.*`). The conversions mirror
//! MLX-LM's `sanitize` in `models/inkling2/checkpoint.py`:
//! - the router weight and the dynamic convolutions' static weights are widened from
//!   BF16 to F32. The message gates' forget and gain weights stay BF16: MLX widens them
//!   too, but the kernel widens each weight exactly as it reads it, which halves the bytes;
//! - the NVFP4 global scale of each expert is `amax * 6 * 448 / denominator`;
//! - the loop-message queries are reordered to query, key, value, MLP per looped layer.
//!
//! The depthwise convolution weights need no transpose: the checkpoint's `[dims, 1, 4]`
//! and MLX's `[dims, 4, 1]` have the same bytes.
//!
//! [`Weights::quantize_fp8`] adds FP8 copies of the large matrices decode reads, and of the
//! message gates, for [`WeightFormat::Fp8`]; the BF16 originals stay for prefill, except
//! the LM head's, which moves to mapped host memory.
//!
//! The embedding table lives in mapped host memory ([`HostMapped`]): kernels gather only
//! the rows of the tokens they embed, over PCIe, so it takes no device memory.

use std::sync::Arc;

use anyhow::{Context as _, bail, ensure};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
use half::bf16;
use safetensors::Dtype;
use siphon::{Checkpoint, TextConfig};

use crate::{
    gpu::HostMapped,
    ops::{
        linear::Prefetch,
        linear_fp8::{Fp8Matrix, LinearFp8, Mxfp8Matrix},
    },
};

pub mod dflash;
pub mod dflash2;
pub mod draft;

/// Largest E2M1 value, from MLX-LM's `sanitize`.
const NVFP4_E2M1_MAX: f32 = 6.0;
/// Largest E4M3 value, from MLX-LM's `sanitize`.
const NVFP4_E4M3_MAX: f32 = 448.0;

#[derive(Debug)]
pub struct Weights {
    /// `[vocab, hidden]`, in mapped host memory.
    pub embed: HostMapped<bf16>,
    pub embed_norm: CudaSlice<bf16>,
    pub layers: Vec<LayerWeights>,
    /// Query of the final depth attention, F32 `[hidden]`.
    pub final_query: CudaSlice<f32>,
    pub norm: CudaSlice<bf16>,
    /// `[vocab, hidden]`: on the device, or after [`Weights::quantize_fp8`] in mapped host
    /// memory, where only the bundled draft's load and its shortlist read it.
    pub lm_head: Table,
    /// Per looped layer, F32 `[4, hidden]`: the query, key, value and MLP message queries.
    pub message_queries: Vec<CudaSlice<f32>>,
    /// FP8 copy of `lm_head`, after [`Weights::quantize_fp8`], which every LM-head product
    /// then reads.
    pub lm_head_fp8: Option<Fp8Matrix>,
    /// The output projection's format in [`Weights::quantize_fp8`].
    pub output: OutputProjection,
}

/// A BF16 tensor in device memory or in mapped host memory.
#[derive(Debug)]
pub enum Table {
    Device(CudaSlice<bf16>),
    Host(HostMapped<bf16>),
}

impl Table {
    /// The tensor as a device slice: its device memory or its mapped host address.
    #[must_use]
    pub fn slice(&self) -> &CudaSlice<bf16> {
        match self {
            Self::Device(t) => t,
            Self::Host(t) => t.slice(),
        }
    }

    /// Whether kernels read it over PCIe.
    #[must_use]
    pub fn is_host(&self) -> bool {
        matches!(self, Self::Host(_))
    }
}

pub use siphon::{OutputProjection, WeightFormat};

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
    /// FP8 copies of the attention projections for decode, after
    /// [`Weights::quantize_fp8`].
    pub fp8: Option<Fp8Projections>,
    /// MXFP8 block-32 copy of `attention.o` (`W_o`), allocated lazily when opt-in
    /// `--prefill-dense fp8` is enabled.
    pub mxfp8_o: Option<Mxfp8Matrix>,
}

/// FP8 copies of one layer's attention projections.
#[derive(Debug)]
pub struct Fp8Projections {
    /// The input projections, in the order of [`AttentionWeights::input_projections`].
    pub inputs: Vec<Fp8Matrix>,
    /// The output projection with [`OutputProjection::Fp8`]; decode reads the BF16 weights
    /// otherwise.
    pub o: Option<Fp8Matrix>,
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
    /// `[hidden, rank]`, multiplied with an F32 input.
    pub forget: CudaSlice<bf16>,
    /// `[hidden, rank]`, multiplied with an F32 input.
    pub gain: CudaSlice<bf16>,
    /// FP8 copies of `down`, `forget` and `gain` for decode, after
    /// [`Weights::quantize_fp8`].
    pub fp8: Option<[Fp8Matrix; 3]>,
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
        let values = self.bf16_host(name, shape)?;
        self.upload(name, &values)
    }

    fn bf16_host(&self, name: &str, shape: &[usize]) -> anyhow::Result<Vec<bf16>> {
        let (dtype, bytes) = self.entry(name, shape)?;
        ensure!(dtype == Dtype::BF16, "{name} is {dtype:?}, expected BF16");
        Ok(bytes
            .as_chunks()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect())
    }

    /// A BF16 tensor in mapped host memory.
    fn bf16_mapped(&self, name: &str, shape: &[usize]) -> anyhow::Result<HostMapped<bf16>> {
        let values = self.bf16_host(name, shape)?;
        HostMapped::new(self.stream, &values).with_context(|| format!("mapping {name}"))
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
    /// Uploads every text-model weight but the embedding table, which goes to mapped host
    /// memory. Takes about 18 GB of device memory.
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
            embed: loader.bf16_mapped("model.llm.embed.weight", &[vocab, hidden])?,
            embed_norm: loader.bf16("model.llm.embed_norm.weight", &[hidden])?,
            layers,
            final_query: loader.f32("model.llm.attn_res.output_query", &[hidden])?,
            norm: loader.bf16("model.llm.norm.weight", &[hidden])?,
            lm_head: Table::Device(loader.bf16("model.llm.unembed.weight", &[vocab, hidden])?),
            message_queries: load_message_queries(&loader, config)?,
            lm_head_fp8: None,
            output: OutputProjection::default(),
        })
    }

    /// Quantizes every layer's attention input projections, its message gates and the LM
    /// head to FP8 copies, and its output projection too with [`OutputProjection::Fp8`],
    /// which decode then reads, and moves the BF16 LM head to mapped host memory. Returns
    /// the bytes the copies take.
    pub fn quantize_fp8(
        &mut self,
        linear: &LinearFp8,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<usize> {
        let quantize = |w: &CudaSlice<bf16>, k: usize| {
            ensure!(w.len().is_multiple_of(k), "weight is not [rows, {k}]");
            linear.quantize(stream, w, w.len() / k, k)
        };
        let hidden = self.embed_norm.len();
        let mut bytes = 0;
        ensure!(
            self.lm_head_fp8.is_none(),
            "the weights are quantized already"
        );
        for layer in &mut self.layers {
            let a = &layer.attention;
            let inputs = a
                .input_projections()
                .into_iter()
                .map(|w| quantize(w, hidden))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let o = match self.output {
                OutputProjection::Fp8 => Some(quantize(&a.o, a.o.len() / hidden)?),
                OutputProjection::Bf16 => None,
            };
            bytes += inputs.iter().map(Fp8Matrix::bytes).sum::<usize>()
                + o.as_ref().map_or(0, Fp8Matrix::bytes);
            layer.fp8 = Some(Fp8Projections { inputs, o });
            for gate in layer.message_gates.iter_mut().flatten() {
                let quantize_gate =
                    |w: &CudaSlice<bf16>, k: usize| linear.quantize_any(stream, w, w.len() / k, k);
                let rank = gate.forget.len() / hidden;
                let copies = [
                    quantize_gate(&gate.down, 2 * hidden)?,
                    quantize_gate(&gate.forget, rank)?,
                    quantize_gate(&gate.gain, rank)?,
                ];
                bytes += copies.iter().map(Fp8Matrix::bytes).sum::<usize>();
                gate.fp8 = Some(copies);
            }
        }
        let lm_head = quantize(self.lm_head.slice(), hidden)?;
        bytes += lm_head.bytes();
        self.lm_head_fp8 = Some(lm_head);
        if let Table::Device(head) = &self.lm_head {
            let host = stream.clone_dtoh(head)?;
            self.lm_head = Table::Host(HostMapped::new(stream, &host)?);
        }
        stream.synchronize()?;
        Ok(bytes)
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

        // Layer 0 has no attention input, so query i belongs to layer i + 1.
        let attention_query = if layer == 0 {
            None
        } else {
            let name = format!("model.llm.attn_res.attn_queries.{}", layer - 1);
            let shape: &[usize] = if sliding { &[hidden] } else { &[3, hidden] };
            Some(loader.f32(&name, shape)?)
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
            fp8: None,
            mxfp8_o: None,
        })
    }
}

impl AttentionWeights {
    /// The projections of the attention inputs, in the runner's launch order.
    #[must_use]
    pub fn input_projections(&self) -> Vec<&CudaSlice<bf16>> {
        let mut weights = vec![&self.q, &self.r, &self.gate, &self.sink, &self.k, &self.v];
        if let KvConv::Dynamic { k_proj, v_proj, .. } = &self.kv_conv {
            weights.extend([k_proj, v_proj]);
        }
        weights
    }

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
    /// The down, forget and gain weights, in the order the gate reads them.
    #[must_use]
    pub fn matrices(&self) -> [&CudaSlice<bf16>; 3] {
        [&self.down, &self.forget, &self.gain]
    }

    /// The matrices decode reads, in the order the gate reads them: the FP8 copies if
    /// there are any.
    #[must_use]
    pub fn decode_matrices(&self) -> [&dyn Prefetch; 3] {
        match &self.fp8 {
            Some([down, forget, gain]) => [down, forget, gain],
            None => [&self.down, &self.forget, &self.gain],
        }
    }

    /// The matrices a verify pass reads, in the order the gate reads them: the FP8 copy of
    /// the down projection if there is one, and the BF16 forget and gain weights.
    #[must_use]
    pub fn rows_matrices(&self) -> [&dyn Prefetch; 3] {
        match &self.fp8 {
            Some([down, ..]) => [down, &self.forget, &self.gain],
            None => [&self.down, &self.forget, &self.gain],
        }
    }

    fn load(loader: &Loader<'_>, config: &TextConfig, p: &str) -> anyhow::Result<Self> {
        let hidden = config.hidden_size;
        let rank = config.loop_config.gate_rank;
        Ok(Self {
            down: loader.bf16(&format!("{p}.down_weight"), &[rank, 2 * hidden])?,
            forget: loader.bf16(&format!("{p}.forget_weight"), &[hidden, rank])?,
            gain: loader.bf16(&format!("{p}.gain_weight"), &[hidden, rank])?,
            fp8: None,
        })
    }
}
