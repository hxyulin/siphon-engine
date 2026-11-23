//! RMS normalization and L2 clipping (`kernels/norm.cu`).

use std::sync::Arc;

use anyhow::{Context as _, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module};

/// Hidden width of the model.
const HIDDEN: usize = 2304;
/// Width of one attention head.
const HEAD_DIM: usize = 128;
/// Width of one head of relative-position states.
const REL_DIM: usize = 16;
/// Elements each thread of a row or head kernel owns, as in MLX.
const VEC: usize = 4;
/// Threads per block of the one-thread-per-head kernel.
const REL_BLOCK: u32 = 128;
/// Key/value heads, query heads, and conv taps of the attention projections.
const KV_HEADS: usize = 8;
const Q_HEADS: usize = 24;
const CONV_TAPS: usize = 4;
/// Blocks and threads of `attention_post_bf16`: one warp per key, value and query head,
/// and one more block for the relative-position heads.
const POST_THREADS: u32 = 128;
const POST_BLOCKS: u32 = ((2 * KV_HEADS + Q_HEADS) / 4 + 1) as u32;

/// The key and value convolutions of [`Norm::attention_post`].
pub enum KvConvWeights<'a> {
    /// `ShortConvolution` weights, BF16 `[1024, 4]`.
    Static {
        k: &'a CudaSlice<bf16>,
        v: &'a CudaSlice<bf16>,
    },
    /// `DynamicShortConvolution`: the row's BF16 taps `[8 heads * 4]` and the F32 static
    /// weights `[4, 1024]`.
    Dynamic {
        k_dynamic: &'a CudaSlice<bf16>,
        v_dynamic: &'a CudaSlice<bf16>,
        k_static: &'a CudaSlice<f32>,
        v_static: &'a CudaSlice<f32>,
    },
}

/// Inputs and outputs of [`Norm::attention_post`] for one row.
pub struct AttentionPost<'a> {
    /// Key and value projections, BF16 `[1024]`.
    pub k_raw: &'a CudaSlice<bf16>,
    pub v_raw: &'a CudaSlice<bf16>,
    pub conv: KvConvWeights<'a>,
    /// Conv states, F32 `[3, 1024]`.
    pub k_state: &'a mut CudaSlice<f32>,
    pub v_state: &'a mut CudaSlice<f32>,
    /// Query projection `[24 * 128]` and relative-position projection `[24 * 16]`, with
    /// their L2-clip scales of the same shapes.
    pub q_raw: &'a CudaSlice<bf16>,
    pub q_scale: &'a CudaSlice<bf16>,
    pub r_raw: &'a CudaSlice<bf16>,
    pub r_scale: &'a CudaSlice<bf16>,
    pub k: &'a mut CudaSlice<bf16>,
    pub v: &'a mut CudaSlice<bf16>,
    pub q: &'a mut CudaSlice<bf16>,
    pub r: &'a mut CudaSlice<bf16>,
    /// For full attention, the position buffer, alpha and n_floor of the query log scaling.
    pub log_scale: Option<(&'a CudaSlice<u32>, f32, f32)>,
    pub qk_threshold: f32,
    pub rel_threshold: f32,
    pub eps: f32,
}

#[derive(Debug)]
pub struct Norm {
    rms_norm_bf16_2304: CudaFunction,
    rms_norm_f32_2304: CudaFunction,
    unweighted_rms_norm_f32_2304: CudaFunction,
    unweighted_rms_norm_bf16_f32_2304: CudaFunction,
    unweighted_rms_norm_bf16_128: CudaFunction,
    l2_clip_bf16_128: CudaFunction,
    l2_clip_bf16_16: CudaFunction,
    attention_post_bf16: CudaFunction,
}

/// Row count of `x` as a grid dimension, after checking that `x` holds whole rows of `width`.
fn rows(len: usize, width: usize) -> anyhow::Result<u32> {
    ensure!(
        len.is_multiple_of(width),
        "x has {len} elements, not a multiple of {width}"
    );
    u32::try_from(len / width).context("too many rows")
}

fn check_out(out: usize, x: usize) -> anyhow::Result<()> {
    ensure!(out == x, "out has {out} elements, x has {x}");
    Ok(())
}

fn check_weight(weight: usize) -> anyhow::Result<()> {
    ensure!(
        weight == HIDDEN,
        "weight has {weight} elements, expected {HIDDEN}"
    );
    Ok(())
}

/// One block of `HIDDEN / VEC` threads per row.
fn row_config(rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: ((HIDDEN / VEC) as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Norm {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "norm", cubin!("norm"))?;
        Ok(Self {
            rms_norm_bf16_2304: module.load_function("rms_norm_bf16_2304")?,
            rms_norm_f32_2304: module.load_function("rms_norm_f32_2304")?,
            unweighted_rms_norm_f32_2304: module.load_function("unweighted_rms_norm_f32_2304")?,
            unweighted_rms_norm_bf16_f32_2304: module
                .load_function("unweighted_rms_norm_bf16_f32_2304")?,
            unweighted_rms_norm_bf16_128: module.load_function("unweighted_rms_norm_bf16_128")?,
            l2_clip_bf16_128: module.load_function("l2_clip_bf16_128")?,
            l2_clip_bf16_16: module.load_function("l2_clip_bf16_16")?,
            attention_post_bf16: module.load_function("attention_post_bf16")?,
        })
    }

    /// Weighted RMSNorm of each BF16 row of `x` (shape `[rows, 2304]`) into `out`.
    pub fn rms_norm(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        check_weight(weight.len())?;
        let rows = rows(x.len(), HIDDEN)?;
        check_out(out.len(), x.len())?;
        let mut launch = stream.launch_builder(&self.rms_norm_bf16_2304);
        launch.arg(x).arg(weight).arg(out).arg(&eps);
        // SAFETY: the arguments match `rms_norm_bf16_2304(const bf16*, const bf16*, bf16*, float)`,
        // and the checks above keep every row inside `x` and `out`.
        unsafe { launch.launch(row_config(rows)) }?;
        Ok(())
    }

    /// Weighted RMSNorm of each F32 row of `x` (shape `[rows, 2304]`), written to `out` as BF16.
    ///
    /// Equals MLX's `norm(x).astype(mx.bfloat16)` for an `nn.RMSNorm` with a BF16 weight.
    pub fn rms_norm_f32(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<f32>,
        weight: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let x = x.view();
        check_weight(weight.len())?;
        let rows = rows(x.len(), HIDDEN)?;
        check_out(out.len(), x.len())?;
        let mut launch = stream.launch_builder(&self.rms_norm_f32_2304);
        launch.arg(&x).arg(weight).arg(out).arg(&eps);
        // SAFETY: the arguments match `rms_norm_f32_2304(const float*, const bf16*, bf16*, float)`,
        // and the checks above keep every row inside `x` and `out`.
        unsafe { launch.launch(row_config(rows)) }?;
        Ok(())
    }

    /// Unweighted RMSNorm of each F32 row of `x` (shape `[rows, 2304]`) into `out`.
    pub fn unweighted_rms_norm(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<f32>,
        out: &mut impl DeviceOut<f32>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let (x, mut out) = (x.view(), out.view_mut());
        let rows = rows(x.len(), HIDDEN)?;
        check_out(out.len(), x.len())?;
        let mut launch = stream.launch_builder(&self.unweighted_rms_norm_f32_2304);
        launch.arg(&x).arg(&mut out).arg(&eps);
        // SAFETY: the arguments match `unweighted_rms_norm_f32_2304(const float*, float*, float)`,
        // and the checks above keep every row inside `x` and `out`.
        unsafe { launch.launch(row_config(rows)) }?;
        Ok(())
    }

    /// Unweighted RMSNorm of each BF16 row of `x` (shape `[rows, 2304]`), computed in and
    /// written as F32: [`Norm::unweighted_rms_norm`] of `x` widened.
    pub fn unweighted_rms_norm_bf16(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        out: &mut impl DeviceOut<f32>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let mut out = out.view_mut();
        let rows = rows(x.len(), HIDDEN)?;
        check_out(out.len(), x.len())?;
        let mut args = KernelArgs::default();
        args.ptr(x, stream).ptr_mut(&mut out, stream).f32(eps);
        // SAFETY: the arguments match `unweighted_rms_norm_bf16_f32_2304(const bf16*, float*,
        // float)`, the checks above keep every row inside `x` and `out`, and the kernel reads
        // nothing before `pdl_wait`.
        unsafe {
            args.launch_pdl(
                stream,
                &self.unweighted_rms_norm_bf16_f32_2304,
                row_config(rows),
            )
        }
    }

    /// Unweighted RMSNorm of each 128-wide BF16 head of `x` (shape `[..., 128]`) into `out`,
    /// computed in F32 as MLX does for the value states.
    pub fn unweighted_rms_norm_heads(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let heads = rows(x.len(), HEAD_DIM)?;
        check_out(out.len(), x.len())?;
        let config = LaunchConfig {
            grid_dim: (heads, 1, 1),
            block_dim: ((HEAD_DIM / VEC) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.unweighted_rms_norm_bf16_128);
        launch.arg(x).arg(out).arg(&eps);
        // SAFETY: the arguments match `unweighted_rms_norm_bf16_128(const bf16*, bf16*, float)`,
        // and the checks above keep every head inside `x` and `out`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// L2 clip of each head of `x` (shape `[rows, heads, head_dim]`) into `out`: scales the head
    /// by `scale`, then shrinks it to norm `threshold` if it is longer.
    ///
    /// `head_dim` is 128 (queries and keys) or 16 (relative-position states). `scale` has shape
    /// `[heads, head_dim]`; `None` means a scale of 1, as in MLX's `L2Clip`.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn l2_clip(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        scale: Option<&CudaSlice<bf16>>,
        out: &mut CudaSlice<bf16>,
        heads: usize,
        head_dim: usize,
        threshold: f32,
    ) -> anyhow::Result<()> {
        ensure!(heads > 0, "heads must be positive");
        rows(x.len(), heads * head_dim)?;
        check_out(out.len(), x.len())?;
        if let Some(scale) = scale {
            ensure!(
                scale.len() == heads * head_dim,
                "scale has {} elements, expected {heads} x {head_dim}",
                scale.len()
            );
        }
        let total_heads = rows(x.len(), head_dim)?;
        let heads_arg = i32::try_from(heads).context("too many heads")?;
        // A null pointer tells the kernel there is no scale.
        let null: u64 = 0;
        let (function, config) = match head_dim {
            HEAD_DIM => (
                &self.l2_clip_bf16_128,
                LaunchConfig {
                    grid_dim: (total_heads, 1, 1),
                    block_dim: ((HEAD_DIM / VEC) as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
            ),
            REL_DIM => (
                &self.l2_clip_bf16_16,
                LaunchConfig {
                    grid_dim: (total_heads.div_ceil(REL_BLOCK), 1, 1),
                    block_dim: (REL_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                },
            ),
            _ => bail!("l2_clip supports head_dim {HEAD_DIM} or {REL_DIM}, not {head_dim}"),
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(x);
        match scale {
            Some(scale) => launch.arg(scale),
            None => launch.arg(&null),
        };
        launch.arg(out).arg(&heads_arg).arg(&threshold);
        if head_dim == REL_DIM {
            launch.arg(&total_heads);
        }
        // SAFETY: the arguments match `l2_clip_bf16_128(const bf16*, const bf16*, bf16*, int,
        // float)` or, for 16-wide heads, `l2_clip_bf16_16` with a trailing `unsigned` head count.
        // The checks above keep every head inside `x` and `out` and every scale row inside
        // `scale`; the 16-wide kernel skips threads past `total_heads`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Everything between attention's projections and the KV append for one row: the key
    /// and value convs, the key L2 clip, the value head norm, the query and relative-position
    /// L2 clips and, for full attention, their log scaling. Equals [`Conv::short_conv`] or
    /// [`Conv::dynamic_short_conv`], [`Norm::l2_clip`], [`Norm::unweighted_rms_norm_heads`]
    /// and `Elementwise::log_scale` in sequence, summing in their order.
    ///
    /// [`Conv::short_conv`]: crate::ops::conv::Conv::short_conv
    /// [`Conv::dynamic_short_conv`]: crate::ops::conv::Conv::dynamic_short_conv
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the struct only bundles borrows for this call"
    )]
    pub fn attention_post(&self, stream: &CudaStream, a: AttentionPost<'_>) -> anyhow::Result<()> {
        let kv = KV_HEADS * HEAD_DIM;
        let check = |name: &str, len: usize, want: usize| {
            ensure!(len == want, "{name} has {len} elements, expected {want}");
            Ok(())
        };
        check("k_raw", a.k_raw.len(), kv)?;
        check("v_raw", a.v_raw.len(), kv)?;
        check("k_state", a.k_state.len(), (CONV_TAPS - 1) * kv)?;
        check("v_state", a.v_state.len(), (CONV_TAPS - 1) * kv)?;
        check("k", a.k.len(), kv)?;
        check("v", a.v.len(), kv)?;
        for (name, x) in [("q_raw", a.q_raw), ("q_scale", a.q_scale)] {
            check(name, x.len(), Q_HEADS * HEAD_DIM)?;
        }
        check("q", a.q.len(), Q_HEADS * HEAD_DIM)?;
        for (name, x) in [("r_raw", a.r_raw), ("r_scale", a.r_scale)] {
            check(name, x.len(), Q_HEADS * REL_DIM)?;
        }
        check("r", a.r.len(), Q_HEADS * REL_DIM)?;
        let config = LaunchConfig {
            grid_dim: (POST_BLOCKS, 1, 1),
            block_dim: (POST_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(a.k_raw, stream).ptr(a.v_raw, stream);
        match &a.conv {
            KvConvWeights::Static { k, v } => {
                check("k conv weight", k.len(), kv * CONV_TAPS)?;
                check("v conv weight", v.len(), kv * CONV_TAPS)?;
                args.ptr(*k, stream)
                    .ptr(*v, stream)
                    .u64(0)
                    .u64(0)
                    .u64(0)
                    .u64(0);
            }
            KvConvWeights::Dynamic {
                k_dynamic,
                v_dynamic,
                k_static,
                v_static,
            } => {
                check("k_dynamic", k_dynamic.len(), KV_HEADS * CONV_TAPS)?;
                check("v_dynamic", v_dynamic.len(), KV_HEADS * CONV_TAPS)?;
                check("k_static", k_static.len(), CONV_TAPS * kv)?;
                check("v_static", v_static.len(), CONV_TAPS * kv)?;
                args.u64(0)
                    .u64(0)
                    .ptr(*k_dynamic, stream)
                    .ptr(*v_dynamic, stream)
                    .ptr(*k_static, stream)
                    .ptr(*v_static, stream);
            }
        }
        let (position, log_scale, alpha, n_floor) = match a.log_scale {
            Some((position, alpha, n_floor)) => (Some(position), 1, alpha, n_floor),
            None => (None, 0, 0.0, 1.0),
        };
        args.ptr_mut(a.k_state, stream)
            .ptr_mut(a.v_state, stream)
            .ptr(a.q_raw, stream)
            .ptr(a.q_scale, stream)
            .ptr(a.r_raw, stream)
            .ptr(a.r_scale, stream)
            .ptr_mut(a.k, stream)
            .ptr_mut(a.v, stream)
            .ptr_mut(a.q, stream)
            .ptr_mut(a.r, stream)
            .opt(position, stream)
            .i32(log_scale)
            .f32(alpha)
            .f32(n_floor)
            .f32(a.qk_threshold)
            .f32(a.rel_threshold)
            .f32(a.eps);
        // SAFETY: the arguments match `attention_post_bf16`'s signature, every buffer has the
        // length checked above, the kernel reads no null pointer (the static conv weights only
        // without dynamic taps, the position only with log scaling), and it reads nothing
        // before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.attention_post_bf16, config) }
    }
}
