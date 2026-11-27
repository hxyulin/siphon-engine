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
    Gpu,
    gpu::HostMapped,
    ops::{
        linear::Prefetch,
        linear_fp8::{Fp8Matrix, LinearFp8, Mxfp8Matrix},
        linear_nvfp4::{LinearNvfp4, Nvfp4Matrix},
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
    /// FP8 copy of `lm_head`, after [`Weights::quantize_fp8`] or [`Weights::quantize_lm_head`],
    /// which every LM-head product then reads (unless `lm_head_nvfp4` is set).
    pub lm_head_fp8: Option<Fp8Matrix>,
    /// NVFP4 copy of `lm_head` (opt-in `--lm-head nvfp4`, ~261 MB vs 463 MB FP8).
    pub lm_head_nvfp4: Option<Nvfp4Matrix>,
    /// Optional override for the LM head's format (`None` follows [`WeightFormat`]).
    pub lm_head_format: Option<LmHeadFormat>,
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

pub use siphon::{LmHeadFormat, OutputProjection, WeightFormat};

/// Options for [`Weights::load_with_options`]: streaming layer-by-layer quantization that
/// drops unused BF16 weights as each layer is quantized and loads quantized LM heads
/// directly from mapped host memory to avoid startup VRAM spikes.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    pub weights: WeightFormat,
    pub output: OutputProjection,
    pub lm_head: Option<LmHeadFormat>,
    /// When true under [`WeightFormat::Fp8`], every path reads the FP8 attention input
    /// projections and message-gate `down` projections (and `W_o` when its quantized copy
    /// covers prefill too), so their BF16 originals are dropped per layer right after
    /// quantization.
    pub prefill_dense_fp8: bool,
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
    pub q: Option<CudaSlice<bf16>>,
    pub k: Option<CudaSlice<bf16>>,
    pub v: Option<CudaSlice<bf16>>,
    /// Relative-feature projection `[heads * d_rel, hidden]`.
    pub r: Option<CudaSlice<bf16>>,
    pub o: Option<CudaSlice<bf16>>,
    pub gate: Option<CudaSlice<bf16>>,
    pub sink: Option<CudaSlice<bf16>>,
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
    /// `[kv_heads * 4, hidden]` to per-token weights, for keys and values (dropped when
    /// FP8 copies cover every path).
    Dynamic {
        k_static: CudaSlice<f32>,
        k_proj: Option<CudaSlice<bf16>>,
        v_static: CudaSlice<f32>,
        v_proj: Option<CudaSlice<bf16>>,
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
    /// `[rank, 2 * hidden]`, dropped when FP8 copies cover every path.
    pub down: Option<CudaSlice<bf16>>,
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
            lm_head_nvfp4: None,
            lm_head_format: None,
            output: OutputProjection::default(),
        })
    }

    /// Loads the checkpoint with streaming per-layer quantization according to `options`:
    ///
    /// - Under [`WeightFormat::Fp8`], quantizes each layer's attention projections and
    ///   message gates right after uploading that layer; when `options.prefill_dense_fp8`
    ///   is also set, immediately drops the BF16 attention input projections, message-gate
    ///   `down` projections, and (when its quantized copy covers prefill too) `attention.o`
    ///   before moving to the next layer so peak VRAM stays near the final footprint.
    /// - Whenever the effective LM head format is `Fp8` or `Nvfp4`, loads
    ///   `model.llm.unembed.weight` directly into mapped host memory and quantizes from
    ///   there, avoiding the 885 MiB BF16 LM-head device peak.
    pub fn load_with_options(
        checkpoint: &Checkpoint,
        config: &TextConfig,
        gpu: &Gpu,
        options: LoadOptions,
    ) -> anyhow::Result<Self> {
        let stream = gpu.stream();
        let loader = Loader { checkpoint, stream };
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let q_dims = config.num_attention_heads * config.head_dim;
        let effective_lm_head = options.lm_head.unwrap_or(match options.weights {
            WeightFormat::Bf16 => LmHeadFormat::Bf16,
            WeightFormat::Fp8 => LmHeadFormat::Fp8,
        });
        let linear_fp8 =
            (options.weights == WeightFormat::Fp8 || effective_lm_head == LmHeadFormat::Fp8)
                .then(|| LinearFp8::load(gpu.context()))
                .transpose()?;
        let linear_nvfp4 = (effective_lm_head == LmHeadFormat::Nvfp4)
            .then(|| LinearNvfp4::load(gpu.context()))
            .transpose()?;
        let release_inputs = options.weights == WeightFormat::Fp8 && options.prefill_dense_fp8;
        let release_o = release_inputs
            && matches!(options.output, OutputProjection::Fp8 | OutputProjection::Mxfp8);
        let need_mxfp8_o = options.weights == WeightFormat::Fp8
            && (options.output == OutputProjection::Mxfp8 || options.prefill_dense_fp8);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let mut layer = LayerWeights::load(&loader, config, layer_idx)
                .with_context(|| format!("loading layer {layer_idx}"))?;
            if options.weights == WeightFormat::Fp8 {
                let linear = linear_fp8.as_ref().context("LinearFp8 missing")?;
                quantize_layer_fp8(
                    &mut layer,
                    linear,
                    stream,
                    hidden,
                    q_dims,
                    options.output,
                    need_mxfp8_o,
                )?;
                if release_inputs {
                    layer.attention.release_input_bf16();
                    for gate in layer.message_gates.iter_mut().flatten() {
                        gate.down = None;
                    }
                }
                if release_o {
                    layer.attention.o = None;
                }
            }
            layers.push(layer);
        }

        let (lm_head, lm_head_fp8, lm_head_nvfp4) = match effective_lm_head {
            LmHeadFormat::Bf16 => (
                Table::Device(loader.bf16("model.llm.unembed.weight", &[vocab, hidden])?),
                None,
                None,
            ),
            LmHeadFormat::Fp8 => {
                let mapped = loader.bf16_mapped("model.llm.unembed.weight", &[vocab, hidden])?;
                let linear = linear_fp8.as_ref().context("LinearFp8 missing")?;
                let fp8 = linear.quantize(stream, mapped.slice(), vocab, hidden)?;
                (Table::Host(mapped), Some(fp8), None)
            }
            LmHeadFormat::Nvfp4 => {
                let mapped = loader.bf16_mapped("model.llm.unembed.weight", &[vocab, hidden])?;
                let linear = linear_nvfp4.as_ref().context("LinearNvfp4 missing")?;
                let nvfp4 = linear.quantize(stream, mapped.slice(), vocab, hidden)?;
                (Table::Host(mapped), None, Some(nvfp4))
            }
        };

        let weights = Self {
            embed: loader.bf16_mapped("model.llm.embed.weight", &[vocab, hidden])?,
            embed_norm: loader.bf16("model.llm.embed_norm.weight", &[hidden])?,
            layers,
            final_query: loader.f32("model.llm.attn_res.output_query", &[hidden])?,
            norm: loader.bf16("model.llm.norm.weight", &[hidden])?,
            lm_head,
            message_queries: load_message_queries(&loader, config)?,
            lm_head_fp8,
            lm_head_nvfp4,
            lm_head_format: options.lm_head,
            output: options.output,
        };
        stream.synchronize()?;
        if release_inputs || effective_lm_head != LmHeadFormat::Bf16 {
            gpu.trim_mem_pool()?;
        }
        Ok(weights)
    }

    /// Whether every layer still retains its BF16 attention input projections and
    /// message-gate `down` projections.
    #[must_use]
    pub fn has_bf16_inputs(&self) -> bool {
        self.layers.iter().all(|l| {
            l.attention.has_bf16_inputs()
                && l.message_gates
                    .iter()
                    .flatten()
                    .all(|g| g.down.is_some())
        })
    }

    /// Whether every layer still retains its BF16 attention output projection `W_o`.
    #[must_use]
    pub fn has_bf16_output(&self) -> bool {
        self.layers.iter().all(|l| l.attention.o.is_some())
    }

    /// Quantizes every layer's attention input projections, its message gates and (unless
    /// [`Self::lm_head_format`] overrides it) the LM head to FP8 copies, and its output
    /// projection too with [`OutputProjection::Fp8`], which decode then reads, and moves
    /// the BF16 LM head to mapped host memory when quantized. Returns the bytes the copies
    /// take.
    pub fn quantize_fp8(
        &mut self,
        linear: &LinearFp8,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<usize> {
        let hidden = self.embed_norm.len();
        let mut bytes = 0;
        ensure!(
            self.layers.first().is_none_or(|l| l.fp8.is_none()),
            "the weights are quantized already"
        );
        let q_dims = self
            .layers
            .first()
            .and_then(|l| l.attention.o.as_ref().map(|o| o.len() / hidden))
            .context("BF16 attention output projection missing")?;
        let need_mxfp8_o = self.output == OutputProjection::Mxfp8;
        for layer in &mut self.layers {
            bytes += quantize_layer_fp8(
                layer,
                linear,
                stream,
                hidden,
                q_dims,
                self.output,
                need_mxfp8_o,
            )?;
        }
        if self.lm_head_format.is_none_or(|f| f == LmHeadFormat::Fp8)
            && self.lm_head_fp8.is_none()
            && self.lm_head_nvfp4.is_none()
        {
            let rows = self.lm_head.slice().len() / hidden;
            ensure!(
                self.lm_head.slice().len() == rows * hidden,
                "lm_head is not [rows, {hidden}]"
            );
            let lm_head = linear.quantize(stream, self.lm_head.slice(), rows, hidden)?;
            bytes += lm_head.bytes();
            self.lm_head_fp8 = Some(lm_head);
            self.move_lm_head_to_host(stream)?;
        }
        stream.synchronize()?;
        Ok(bytes)
    }

    /// Quantizes `lm_head` to `format` (if not already quantized) and moves the BF16 original
    /// to mapped host memory. Returns the bytes of any newly allocated quantized copy.
    pub fn quantize_lm_head(
        &mut self,
        linear_fp8: &LinearFp8,
        linear_nvfp4: &LinearNvfp4,
        stream: &Arc<CudaStream>,
        format: LmHeadFormat,
    ) -> anyhow::Result<usize> {
        let hidden = self.embed_norm.len();
        let rows = self.lm_head.slice().len() / hidden;
        ensure!(
            self.lm_head.slice().len() == rows * hidden,
            "lm_head is not [rows, {hidden}]"
        );
        let bytes = match format {
            LmHeadFormat::Bf16 => 0,
            LmHeadFormat::Fp8 => {
                if self.lm_head_fp8.is_some() {
                    return Ok(0);
                }
                let q = linear_fp8.quantize(stream, self.lm_head.slice(), rows, hidden)?;
                let b = q.bytes();
                self.lm_head_fp8 = Some(q);
                self.move_lm_head_to_host(stream)?;
                b
            }
            LmHeadFormat::Nvfp4 => {
                if self.lm_head_nvfp4.is_some() {
                    return Ok(0);
                }
                let q = linear_nvfp4.quantize(stream, self.lm_head.slice(), rows, hidden)?;
                let b = q.bytes();
                self.lm_head_nvfp4 = Some(q);
                self.move_lm_head_to_host(stream)?;
                b
            }
        };
        if bytes > 0 {
            stream.synchronize()?;
        }
        Ok(bytes)
    }

    fn move_lm_head_to_host(&mut self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        if let Table::Device(head) = &self.lm_head {
            let host = stream.clone_dtoh(head)?;
            self.lm_head = Table::Host(HostMapped::new(stream, &host)?);
        }
        Ok(())
    }
}

/// Quantizes one layer's attention input projections, output projection (according to
/// `output` and `need_mxfp8_o`), and message gates to FP8 / MXFP8.
fn quantize_layer_fp8(
    layer: &mut LayerWeights,
    linear: &LinearFp8,
    stream: &Arc<CudaStream>,
    hidden: usize,
    q_dims: usize,
    output: OutputProjection,
    need_mxfp8_o: bool,
) -> anyhow::Result<usize> {
    let quantize = |w: &CudaSlice<bf16>, k: usize| {
        ensure!(w.len().is_multiple_of(k), "weight is not [rows, {k}]");
        linear.quantize(stream, w, w.len() / k, k)
    };
    let a = &layer.attention;
    let inputs = a
        .input_projections()?
        .into_iter()
        .map(|w| quantize(w, hidden))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let o_bf16 = a
        .o
        .as_ref()
        .context("BF16 attention output projection was released")?;
    let o = match output {
        OutputProjection::Fp8 => Some(quantize(o_bf16, q_dims)?),
        OutputProjection::Bf16 | OutputProjection::Mxfp8 => None,
    };
    let mut bytes = 0;
    if need_mxfp8_o && layer.mxfp8_o.is_none() {
        let mxfp8 = linear.quantize_mxfp8(stream, o_bf16, hidden, q_dims)?;
        bytes += mxfp8.bytes();
        layer.mxfp8_o = Some(mxfp8);
    }
    bytes += inputs.iter().map(Fp8Matrix::bytes).sum::<usize>()
        + o.as_ref().map_or(0, Fp8Matrix::bytes);
    layer.fp8 = Some(Fp8Projections { inputs, o });
    for gate in layer.message_gates.iter_mut().flatten() {
        let quantize_gate =
            |w: &CudaSlice<bf16>, k: usize| linear.quantize_any(stream, w, w.len() / k, k);
        let rank = gate.forget.len() / hidden;
        let down = gate
            .down
            .as_ref()
            .context("BF16 message gate down projection was released")?;
        let copies = [
            quantize_gate(down, 2 * hidden)?,
            quantize_gate(&gate.forget, rank)?,
            quantize_gate(&gate.gain, rank)?,
        ];
        bytes += copies.iter().map(Fp8Matrix::bytes).sum::<usize>();
        gate.fp8 = Some(copies);
    }
    Ok(bytes)
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
    /// Whether every BF16 attention input projection is still present on device.
    #[must_use]
    pub fn has_bf16_inputs(&self) -> bool {
        self.q.is_some()
            && self.r.is_some()
            && self.gate.is_some()
            && self.sink.is_some()
            && self.k.is_some()
            && self.v.is_some()
            && match &self.kv_conv {
                KvConv::Static { .. } => true,
                KvConv::Dynamic { k_proj, v_proj, .. } => k_proj.is_some() && v_proj.is_some(),
            }
    }

    /// Drops the BF16 attention input projections (`q, r, gate, sink, k, v` and dynamic
    /// `k_proj, v_proj`) once their FP8 copies cover every code path.
    pub fn release_input_bf16(&mut self) {
        self.q = None;
        self.k = None;
        self.v = None;
        self.r = None;
        self.gate = None;
        self.sink = None;
        if let KvConv::Dynamic { k_proj, v_proj, .. } = &mut self.kv_conv {
            *k_proj = None;
            *v_proj = None;
        }
    }

    /// The projections of the attention inputs, in the runner's launch order.
    pub fn input_projections(&self) -> anyhow::Result<Vec<&CudaSlice<bf16>>> {
        let missing = "BF16 attention input projections were released for FP8 copies";
        let mut weights = vec![
            self.q.as_ref().context(missing)?,
            self.r.as_ref().context(missing)?,
            self.gate.as_ref().context(missing)?,
            self.sink.as_ref().context(missing)?,
            self.k.as_ref().context(missing)?,
            self.v.as_ref().context(missing)?,
        ];
        if let KvConv::Dynamic { k_proj, v_proj, .. } = &self.kv_conv {
            weights.extend([
                k_proj.as_ref().context(missing)?,
                v_proj.as_ref().context(missing)?,
            ]);
        }
        Ok(weights)
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
                k_proj: Some(loader.bf16(&format!("{a}.k_sconv.weight_proj"), &proj_shape)?),
                v_static: loader.f32(&format!("{a}.v_sconv.static_w"), &[conv_k, kv_dim])?,
                v_proj: Some(loader.bf16(&format!("{a}.v_sconv.weight_proj"), &proj_shape)?),
            }
        };
        Ok(Self {
            q: Some(loader.bf16(&format!("{a}.wq_du.weight"), &[q_dim, hidden])?),
            k: Some(loader.bf16(&format!("{a}.wk_dv.weight"), &[kv_dim, hidden])?),
            v: Some(loader.bf16(&format!("{a}.wv_dv.weight"), &[kv_dim, hidden])?),
            r: Some(loader.bf16(
                &format!("{a}.wr_du.weight"),
                &[heads * config.d_rel, hidden],
            )?),
            o: Some(loader.bf16(&format!("{a}.wo_ud.weight"), &[hidden, q_dim])?),
            gate: Some(loader.bf16(&format!("{a}.gate_attn.weight"), &[q_dim, hidden])?),
            sink: Some(loader.bf16(&format!("{a}.attn_sink_proj.weight"), &[heads, hidden])?),
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
    pub fn matrices(&self) -> anyhow::Result<[&CudaSlice<bf16>; 3]> {
        let down = self
            .down
            .as_ref()
            .context("BF16 message gate down projection was released")?;
        Ok([down, &self.forget, &self.gain])
    }

    /// The matrices decode reads, in the order the gate reads them: the FP8 copies if
    /// there are any.
    #[must_use]
    pub fn decode_matrices(&self) -> Vec<&dyn Prefetch> {
        if let Some([down, forget, gain]) = &self.fp8 {
            vec![down, forget, gain]
        } else {
            let mut v: Vec<&dyn Prefetch> = Vec::with_capacity(3);
            if let Some(down) = &self.down {
                v.push(down);
            }
            v.push(&self.forget);
            v.push(&self.gain);
            v
        }
    }

    /// The matrices a verify pass reads, in the order the gate reads them: the FP8 copy of
    /// the down projection if there is one, and the BF16 forget and gain weights.
    #[must_use]
    pub fn rows_matrices(&self) -> Vec<&dyn Prefetch> {
        if let Some([down, ..]) = &self.fp8 {
            vec![down, &self.forget, &self.gain]
        } else {
            let mut v: Vec<&dyn Prefetch> = Vec::with_capacity(3);
            if let Some(down) = &self.down {
                v.push(down);
            }
            v.push(&self.forget);
            v.push(&self.gain);
            v
        }
    }

    fn load(loader: &Loader<'_>, config: &TextConfig, p: &str) -> anyhow::Result<Self> {
        let hidden = config.hidden_size;
        let rank = config.loop_config.gate_rank;
        Ok(Self {
            down: Some(loader.bf16(&format!("{p}.down_weight"), &[rank, 2 * hidden])?),
            forget: loader.bf16(&format!("{p}.forget_weight"), &[hidden, rank])?,
            gain: loader.bf16(&format!("{p}.gain_weight"), &[hidden, rank])?,
            fp8: None,
        })
    }
}
