//! The checkpoint's vision tower on the GPU: patch embedding, 23 bidirectional transformer
//! layers over one image, the 2x2 merge and the projector, giving one 2304-wide BF16
//! embedding per image token.
//!
//! The forward pass is Transformers' `Inkling2VisionModel` (5.24.0); `docs/vision.md`
//! describes it. The image's rows are the class token, then the patches in
//! [`PreparedImage`]'s order. Each layer runs
//! `x += ls_1 * attn(ln_1(x))`, then `x += ls_2 * mlp(ln_2(x))`, with an F32 residual
//! stream and BF16 products. The merge needs no copy: four consecutive patch rows are one
//! token, so rows 1.. of the last layer's BF16 output are the projector's `[tokens, 4096]`
//! input.
//!
//! The linear layers stay BF16. FP8 weights with MXFP8 activations were tried and removed:
//! they changed greedy tokens and one image description (see docs/vision.md).

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use safetensors::Dtype;
use siphon::Checkpoint;
use siphon::memory::{
    VISION_DIM, VISION_LAYERS, VISION_MLP, VISION_OUT, VISION_PATCH_VALUES, VISION_POSITIONS,
    VISION_PROJECTION,
};
pub use siphon::{VisionFormat, vision::PreparedImage};

use crate::{
    Gpu,
    ops::{
        linear::{Gemm, Linear},
        vision::{DIM, EmbedWeights, HEAD_DIM, HEADS, POS_GRID, Residual, Vision},
    },
};

const _: () = assert!(
    DIM == VISION_DIM && HEADS * HEAD_DIM == DIM && POS_GRID * POS_GRID == VISION_POSITIONS
);

/// `layer_norm_eps` of the checkpoint's `vision_config`.
const LAYER_NORM_EPS: f32 = 1e-5;
/// Patches per image token: the 2x2 merge.
const MERGE: usize = 4;

#[derive(Debug)]
struct Layer {
    norm1: (CudaSlice<bf16>, CudaSlice<bf16>),
    norm2: (CudaSlice<bf16>, CudaSlice<bf16>),
    /// Layer scales of the attention and MLP outputs.
    gamma1: CudaSlice<bf16>,
    gamma2: CudaSlice<bf16>,
    qkv: [CudaSlice<bf16>; 3],
    qkv_bias: [CudaSlice<bf16>; 3],
    out: CudaSlice<bf16>,
    out_bias: CudaSlice<bf16>,
    fc1: CudaSlice<bf16>,
    fc1_bias: CudaSlice<bf16>,
    fc2: CudaSlice<bf16>,
    fc2_bias: CudaSlice<bf16>,
}

/// Activation buffers for images of up to `max_patches` patches, `rows = max_patches + 1`.
#[derive(Debug)]
struct Buffers {
    max_patches: usize,
    /// F32 `[rows, 1024]`, the residual stream.
    x: CudaSlice<f32>,
    /// BF16 `[rows, 1024]`: each product's input, the attention output and the MLP's
    /// second product.
    xn: CudaSlice<bf16>,
    /// BF16 `[rows, 4096]`: first the patches (BF16, then the F32 upload after them),
    /// then per layer q, k and v, the attention projection's output, and the MLP's hidden
    /// rows.
    scratch: CudaSlice<bf16>,
}

/// The vision tower's weights, kernels and buffers.
#[derive(Debug)]
pub struct VisionTower {
    stream: Arc<CudaStream>,
    format: VisionFormat,
    linear: Linear,
    vision: Vision,
    patch_embedding: CudaSlice<bf16>,
    position_table: CudaSlice<bf16>,
    class: CudaSlice<bf16>,
    class_position: CudaSlice<bf16>,
    pre_norm: (CudaSlice<bf16>, CudaSlice<bf16>),
    layers: Vec<Layer>,
    proj_in: CudaSlice<bf16>,
    proj_out: CudaSlice<bf16>,
    buffers: Option<Buffers>,
}

/// Reads `model.visual.*` tensors and uploads them.
struct Loader<'a> {
    checkpoint: &'a Checkpoint,
    stream: &'a Arc<CudaStream>,
}

impl Loader<'_> {
    fn bf16(&self, name: &str, shape: &[usize]) -> anyhow::Result<CudaSlice<bf16>> {
        let name = format!("model.visual.{name}");
        let entry = self
            .checkpoint
            .get(&name)
            .with_context(|| format!("{name} missing from the checkpoint"))?;
        ensure!(
            entry.shape == shape,
            "{name} has shape {:?}, expected {shape:?}",
            entry.shape
        );
        ensure!(
            entry.dtype == Dtype::BF16,
            "{name} is {:?}, expected BF16",
            entry.dtype
        );
        let values: Vec<bf16> = self
            .checkpoint
            .bytes(entry)
            .as_chunks()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect();
        self.stream
            .clone_htod(&values)
            .with_context(|| format!("uploading {name}"))
    }

    fn vector(&self, name: &str, len: usize) -> anyhow::Result<CudaSlice<bf16>> {
        self.bf16(name, &[len])
    }

    /// A linear layer's weight `[rows, k]`.
    fn matrix(&self, name: &str, rows: usize, k: usize) -> anyhow::Result<CudaSlice<bf16>> {
        self.bf16(name, &[rows, k])
    }

    fn norm(&self, prefix: &str) -> anyhow::Result<(CudaSlice<bf16>, CudaSlice<bf16>)> {
        Ok((
            self.vector(&format!("{prefix}.weight"), DIM)?,
            self.vector(&format!("{prefix}.bias"), DIM)?,
        ))
    }

    fn layer(&self, i: usize) -> anyhow::Result<Layer> {
        let p = format!("backbone.layers.{i}");
        let attn = |w: &str| format!("{p}.self_attn.{w}_proj");
        Ok(Layer {
            norm1: self.norm(&format!("{p}.ln_1"))?,
            norm2: self.norm(&format!("{p}.ln_2"))?,
            gamma1: self.vector(&format!("{p}.ls_1.gamma"), DIM)?,
            gamma2: self.vector(&format!("{p}.ls_2.gamma"), DIM)?,
            qkv: [
                self.matrix(&format!("{}.weight", attn("q")), DIM, DIM)?,
                self.matrix(&format!("{}.weight", attn("k")), DIM, DIM)?,
                self.matrix(&format!("{}.weight", attn("v")), DIM, DIM)?,
            ],
            qkv_bias: [
                self.vector(&format!("{}.bias", attn("q")), DIM)?,
                self.vector(&format!("{}.bias", attn("k")), DIM)?,
                self.vector(&format!("{}.bias", attn("v")), DIM)?,
            ],
            out: self.matrix(&format!("{}.weight", attn("out")), DIM, DIM)?,
            out_bias: self.vector(&format!("{}.bias", attn("out")), DIM)?,
            fc1: self.matrix(&format!("{p}.mlp.c_fc.weight"), VISION_MLP, DIM)?,
            fc1_bias: self.vector(&format!("{p}.mlp.c_fc.bias"), VISION_MLP)?,
            fc2: self.matrix(&format!("{p}.mlp.c_proj.weight"), DIM, VISION_MLP)?,
            fc2_bias: self.vector(&format!("{p}.mlp.c_proj.bias"), DIM)?,
        })
    }
}

/// Checks the checkpoint's `vision_config` against the shapes the kernels run.
fn check_config(checkpoint: &Checkpoint) -> anyhow::Result<()> {
    let entry = checkpoint
        .get("model.visual.proj_out.weight")
        .context("the checkpoint has no vision tower")?;
    let dir = checkpoint
        .file_path(entry)
        .parent()
        .context("checkpoint file has no directory")?;
    let path = dir.join("config.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&text)?;
    let vision = &config["vision_config"];
    let expect = [
        ("hidden_size", VISION_DIM),
        ("num_hidden_layers", VISION_LAYERS),
        ("num_attention_heads", HEADS),
        ("native_patch_size", 20),
        ("adapter_merge_size", 2),
        ("posemb_grid_size", POS_GRID),
        ("projection_hidden_dim", VISION_PROJECTION),
        ("text_hidden_size", VISION_OUT),
    ];
    for (key, value) in expect {
        ensure!(
            vision[key].as_u64() == Some(value as u64),
            "vision_config.{key} is {}, the kernels need {value}",
            vision[key]
        );
    }
    ensure!(
        vision["projector_bias"].as_bool() == Some(false),
        "the kernels need a projector without biases"
    );
    let eps = vision["layer_norm_eps"].as_f64().unwrap_or(f64::NAN);
    ensure!(
        (eps - f64::from(LAYER_NORM_EPS)).abs() < 1e-12,
        "vision_config.layer_norm_eps is {eps}, the kernels use {LAYER_NORM_EPS}"
    );
    Ok(())
}

impl VisionTower {
    /// Uploads the checkpoint's `model.visual.*` tensors (0.59 GiB) and loads the kernels.
    /// Activation buffers are allocated by the first [`Self::encode`] or by
    /// [`Self::reserve`]. Every launch goes to `gpu`'s stream.
    pub fn load(gpu: &Gpu, checkpoint: &Checkpoint, format: VisionFormat) -> anyhow::Result<Self> {
        check_config(checkpoint)?;
        let stream = Arc::clone(gpu.stream());
        let ctx = gpu.context();
        let loader = Loader {
            checkpoint,
            stream: &stream,
        };
        let layers = (0..VISION_LAYERS)
            .map(|i| {
                loader
                    .layer(i)
                    .with_context(|| format!("loading vision layer {i}"))
            })
            .collect::<anyhow::Result<_>>()?;
        ensure!(
            checkpoint
                .get(&format!(
                    "model.visual.backbone.layers.{VISION_LAYERS}.ln_1.weight"
                ))
                .is_none(),
            "the checkpoint's vision tower has more than {VISION_LAYERS} layers"
        );
        let e = "backbone.embeddings";
        let tower = Self {
            linear: Linear::load(ctx)?,
            vision: Vision::load(ctx)?,
            patch_embedding: loader.bf16(
                &format!("{e}.patch_embedding.weight"),
                &[DIM, VISION_PATCH_VALUES],
            )?,
            position_table: loader
                .bf16(&format!("{e}.position_embedding"), &[VISION_POSITIONS, DIM])?,
            class: loader.vector(&format!("{e}.class_embedding"), DIM)?,
            class_position: loader.vector(&format!("{e}.cls_position_embedding"), DIM)?,
            pre_norm: loader.norm("backbone.ln_pre")?,
            layers,
            proj_in: loader.matrix("proj_in.weight", VISION_PROJECTION, MERGE * DIM)?,
            proj_out: loader.matrix("proj_out.weight", VISION_OUT, VISION_PROJECTION)?,
            buffers: None,
            format,
            stream,
        };
        tower.stream.synchronize()?;
        Ok(tower)
    }

    #[must_use]
    pub fn format(&self) -> VisionFormat {
        self.format
    }

    /// Allocates the buffers for images of up to `max_patches` patches now, so that
    /// encoding such images allocates nothing but its output. Device bytes:
    /// [`siphon::memory::vision_buffer_bytes`], less the output.
    pub fn reserve(&mut self, max_patches: usize) -> anyhow::Result<()> {
        ensure!(max_patches > 0, "no patches to reserve for");
        if self
            .buffers
            .as_ref()
            .is_some_and(|b| b.max_patches >= max_patches)
        {
            return Ok(());
        }
        // Free the smaller buffers before allocating the new ones.
        self.buffers = None;
        let rows = max_patches + 1;
        self.buffers = Some(Buffers {
            max_patches,
            x: self.stream.alloc_zeros(rows * DIM)?,
            xn: self.stream.alloc_zeros(rows * DIM)?,
            scratch: self.stream.alloc_zeros(rows * VISION_MLP)?,
        });
        Ok(())
    }

    /// Frees the activation buffers. The next [`Self::encode`] allocates them again.
    pub fn release(&mut self) {
        self.buffers = None;
    }

    /// Patches the current buffers hold room for, 0 without buffers.
    #[must_use]
    pub fn reserved_patches(&self) -> usize {
        self.buffers.as_ref().map_or(0, |b| b.max_patches)
    }

    /// Embeddings for one image: `[image.tokens(), 2304]` BF16, ready to replace the
    /// placeholder rows. Enqueued on the stream without waiting for it; the buffers grow
    /// to the image's patches if they are smaller.
    #[expect(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        reason = "follows the forward pass step by step"
    )]
    pub fn encode(&mut self, image: &PreparedImage) -> anyhow::Result<CudaSlice<bf16>> {
        let [frames, h, w] = image.grid;
        let patches = frames * h * w;
        ensure!(
            patches > 0 && h.is_multiple_of(2) && w.is_multiple_of(2),
            "grid {:?} must have positive even sides",
            image.grid
        );
        ensure!(
            image.patches.len() == patches * VISION_PATCH_VALUES,
            "the image has {} patch values, its grid {:?} needs {}",
            image.patches.len(),
            image.grid,
            patches * VISION_PATCH_VALUES
        );
        self.reserve(patches)?;
        let rows = patches + 1;
        let tokens = patches / MERGE;
        let grid = image.grid;
        let Self {
            stream,
            linear,
            vision,
            patch_embedding,
            position_table,
            class,
            class_position,
            pre_norm,
            layers,
            proj_in,
            proj_out,
            buffers,
            ..
        } = self;
        let stream: &Arc<CudaStream> = stream;
        let b = buffers.as_mut().context("buffers were just reserved")?;
        let mut x = b.x.slice_mut(..rows * DIM);
        let mut xn = b.xn.slice_mut(..rows * DIM);
        let values = patches * VISION_PATCH_VALUES;

        // The patches: F32 uploaded after the BF16 copy they round to, which the patch
        // embedding reads.
        {
            let (mut head, mut tail) = b.scratch.split_at_mut(values);
            // SAFETY: `tail` holds rows * 4096 - values >= 2 * values BF16 elements, room for
            // `values` F32 values, at a byte offset of 2 * values, a multiple of 16 since 1200
            // is a multiple of 8; any bit pattern is a valid F32.
            let mut upload = unsafe { tail.transmute_mut::<f32>(values) }
                .context("the scratch cannot hold the patches")?;
            stream.memcpy_htod(image.patches.as_slice(), &mut upload)?;
            vision.f32_to_bf16(stream, &upload.as_view(), &mut head)?;
            let head = head.slice(..values);
            let mut emb = xn.slice_mut(DIM..);
            linear.gemm_bf16(
                stream,
                &mut [Gemm::new(patch_embedding, &head, &mut emb)],
                patches,
            )?;
        }
        vision.embed(
            stream,
            &xn.as_view(),
            &EmbedWeights {
                table: position_table,
                class,
                class_position,
                norm_weight: &pre_norm.0,
                norm_bias: &pre_norm.1,
            },
            grid,
            LAYER_NORM_EPS,
            &mut x,
        )?;
        let first = &layers[0].norm1;
        vision.residual_norm(
            stream,
            &mut x,
            None,
            Some((&first.0, &first.1)),
            LAYER_NORM_EPS,
            &mut xn,
        )?;

        let width = rows * DIM;
        for (i, layer) in layers.iter().enumerate() {
            let scratch = &mut b.scratch;
            {
                let (mut q, mut rest) = scratch.split_at_mut(width);
                let (mut k, mut rest) = rest.split_at_mut(width);
                let (mut v, _) = rest.split_at_mut(width);
                let [wq, wk, wv] = &layer.qkv;
                let x_in = xn.as_view();
                linear.gemm_bf16(
                    stream,
                    &mut [
                        Gemm::new(wq, &x_in, &mut q),
                        Gemm::new(wk, &x_in, &mut k),
                        Gemm::new(wv, &x_in, &mut v),
                    ],
                    rows,
                )?;
                let [bq, bk, bv] = &layer.qkv_bias;
                vision.qkv_rope(stream, &mut q, &mut k, &mut v, [bq, bk, bv], grid)?;
                vision.attention(stream, &q.as_view(), &k.as_view(), &v.as_view(), &mut xn)?;
                // The attention projection's output goes to q's rows, which are spent.
                let attended = xn.as_view();
                linear.gemm_bf16(
                    stream,
                    &mut [Gemm::new(&layer.out, &attended, &mut q)],
                    rows,
                )?;
                vision.residual_norm(
                    stream,
                    &mut x,
                    Some(&Residual {
                        delta: Some(q.as_view()),
                        bias: &layer.out_bias,
                        gamma: &layer.gamma1,
                    }),
                    Some((&layer.norm2.0, &layer.norm2.1)),
                    LAYER_NORM_EPS,
                    &mut xn,
                )?;
            }
            let mut hidden = scratch.slice_mut(..rows * VISION_MLP);
            let normed = xn.as_view();
            linear.gemm_bf16(
                stream,
                &mut [Gemm::new(&layer.fc1, &normed, &mut hidden)],
                rows,
            )?;
            vision.gelu(stream, &mut hidden, Some(&layer.fc1_bias), VISION_MLP)?;
            let activated = hidden.as_view();
            linear.gemm_bf16(
                stream,
                &mut [Gemm::new(&layer.fc2, &activated, &mut xn)],
                rows,
            )?;
            // The next layer's first norm, or after the last layer the plain BF16 rows the
            // projector takes.
            let next = layers.get(i + 1).map(|l| (&l.norm1.0, &l.norm1.1));
            vision.residual_norm(
                stream,
                &mut x,
                Some(&Residual {
                    delta: None,
                    bias: &layer.fc2_bias,
                    gamma: &layer.gamma2,
                }),
                next,
                LAYER_NORM_EPS,
                &mut xn,
            )?;
        }

        // The 2x2 merge: patch rows 1.. are [tokens, 4096].
        let merged = xn.slice(DIM..);
        let mut hidden = b.scratch.slice_mut(..tokens * VISION_PROJECTION);
        linear.gemm_bf16(
            stream,
            &mut [Gemm::new(proj_in, &merged, &mut hidden)],
            tokens,
        )?;
        vision.gelu(stream, &mut hidden, None, VISION_PROJECTION)?;
        let mut out = stream.alloc_zeros::<bf16>(tokens * VISION_OUT)?;
        let activated = hidden.as_view();
        linear.gemm_bf16(
            stream,
            &mut [Gemm::new(proj_out, &activated, &mut out)],
            tokens,
        )?;
        Ok(out)
    }
}
