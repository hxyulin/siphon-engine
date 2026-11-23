//! RMS normalization and L2 clipping (`kernels/norm.cu`).

use std::sync::Arc;

use anyhow::{Context as _, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, cubin, load_module};

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

#[derive(Debug)]
pub struct Norm {
    rms_norm_bf16_2304: CudaFunction,
    rms_norm_f32_2304: CudaFunction,
    unweighted_rms_norm_f32_2304: CudaFunction,
    unweighted_rms_norm_bf16_128: CudaFunction,
    l2_clip_bf16_128: CudaFunction,
    l2_clip_bf16_16: CudaFunction,
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
            unweighted_rms_norm_bf16_128: module.load_function("unweighted_rms_norm_bf16_128")?,
            l2_clip_bf16_128: module.load_function("l2_clip_bf16_128")?,
            l2_clip_bf16_16: module.load_function("l2_clip_bf16_16")?,
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
}
