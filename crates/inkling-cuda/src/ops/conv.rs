//! Causal short convolutions with a carried conv state (`kernels/conv.cu`).
//!
//! Both ops process the rows of one sequence and update `state` in place to the
//! last three rows of the conv window, so a decode step continues where the
//! previous call stopped. A fresh sequence starts from a zeroed state.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{KernelArgs, cubin, load_module};

/// Taps per channel.
const KERNEL: usize = 4;
/// Window rows carried between calls.
const STATE_ROWS: usize = KERNEL - 1;
/// Channel width of the K/V projections that use the dynamic convolution.
const DYNAMIC_DIMS: usize = 1024;
/// Heads that each get their own dynamic taps.
const DYNAMIC_HEADS: usize = 8;
/// Threads per block; each thread owns one channel.
const BLOCK: u32 = 256;
/// Hidden width, the channels of the post-attention and post-MLP convs.
const HIDDEN: usize = 2304;
/// Experts per token in the MoE combine.
const TOP_K: usize = 8;

/// One thread per hidden channel.
fn hidden_config() -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((HIDDEN as u32).div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn check_hidden_conv(weight: &CudaSlice<bf16>, state: &CudaSlice<f32>) -> anyhow::Result<()> {
    ensure!(
        weight.len() == HIDDEN * KERNEL && state.len() == STATE_ROWS * HIDDEN,
        "weight has {} and state {} elements, expected [{HIDDEN}, {KERNEL}] and [{STATE_ROWS}, {HIDDEN}]",
        weight.len(),
        state.len()
    );
    Ok(())
}

#[derive(Debug)]
pub struct Conv {
    short_conv_bf16: CudaFunction,
    dynamic_short_conv_bf16: CudaFunction,
    short_conv_residual_2304: CudaFunction,
    moe_conv_residual_2304: CudaFunction,
}

impl Conv {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "conv", cubin!("conv"))?;
        Ok(Self {
            short_conv_bf16: module.load_function("short_conv_bf16")?,
            dynamic_short_conv_bf16: module.load_function("dynamic_short_conv_bf16")?,
            short_conv_residual_2304: module.load_function("short_conv_residual_2304")?,
            moe_conv_residual_2304: module.load_function("moe_conv_residual_2304")?,
        })
    }

    /// MLX-LM `ShortConvolution` of the BF16 rows `x` (shape `[rows, dims]`) into `out`.
    ///
    /// `weight` is the checkpoint's depthwise `conv1d.weight` as stored after `sanitize`,
    /// shape `[dims, 4, 1]`: tap 0 multiplies the oldest row and tap 3 the current one.
    /// `state` is F32 `[3, dims]`.
    pub fn short_conv(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        ensure!(
            !weight.is_empty() && weight.len().is_multiple_of(KERNEL),
            "weight has {} elements, expected [dims, {KERNEL}]",
            weight.len()
        );
        let dims = weight.len() / KERNEL;
        ensure!(
            state.len() == STATE_ROWS * dims,
            "state has {} elements, expected [{STATE_ROWS}, {dims}]",
            state.len()
        );
        let rows = check_rows(x, out, dims)?;
        let mut launch = stream.launch_builder(&self.short_conv_bf16);
        let dims_arg = i32::try_from(dims).context("too many channels")?;
        launch
            .arg(x)
            .arg(weight)
            .arg(state)
            .arg(out)
            .arg(&dims_arg)
            .arg(&rows);
        // SAFETY: the arguments match `short_conv_bf16(const bf16*, const bf16*, float*, bf16*,
        // int, int)`, and the checks above keep every channel and row inside `x`, `weight`,
        // `state` and `out`.
        unsafe { launch.launch(channel_config(dims)?) }?;
        Ok(())
    }

    /// MLX-LM `DynamicShortConvolution` of the BF16 rows `x` (shape `[rows, 1024]`) into `out`.
    ///
    /// `dynamic` is the BF16 `weight_proj` output, shape `[rows, 8 heads, 4 lags]`.
    /// `static_weight` is the F32 `static_weight`, shape `[4 lags, 1024]`; lag 0 multiplies
    /// the current row. `state` is F32 `[3, 1024]`.
    pub fn dynamic_short_conv(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        dynamic: &CudaSlice<bf16>,
        static_weight: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        ensure!(
            static_weight.len() == KERNEL * DYNAMIC_DIMS,
            "static_weight has {} elements, expected [{KERNEL}, {DYNAMIC_DIMS}]",
            static_weight.len()
        );
        ensure!(
            state.len() == STATE_ROWS * DYNAMIC_DIMS,
            "state has {} elements, expected [{STATE_ROWS}, {DYNAMIC_DIMS}]",
            state.len()
        );
        let rows = check_rows(x, out, DYNAMIC_DIMS)?;
        ensure!(
            dynamic.len() == x.len() / DYNAMIC_DIMS * DYNAMIC_HEADS * KERNEL,
            "dynamic has {} elements, expected [{rows}, {}]",
            dynamic.len(),
            DYNAMIC_HEADS * KERNEL
        );
        let mut launch = stream.launch_builder(&self.dynamic_short_conv_bf16);
        launch
            .arg(x)
            .arg(dynamic)
            .arg(static_weight)
            .arg(state)
            .arg(out)
            .arg(&rows);
        // SAFETY: the arguments match `dynamic_short_conv_bf16(const bf16*, const bf16*,
        // const float*, float*, bf16*, int)`, and the checks above keep every row inside `x`,
        // `dynamic` and `out`; the kernel's channel count is fixed at 1024.
        unsafe { launch.launch(channel_config(DYNAMIC_DIMS)?) }?;
        Ok(())
    }

    /// The post-attention conv and residual add for one row: `residual = f32(hidden) +
    /// f32(short_conv(x))`, or `f32(short_conv(x))` without `hidden`. Equals
    /// [`Conv::short_conv`] followed by the elementwise add or cast.
    pub fn short_conv_residual(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        state: &mut CudaSlice<f32>,
        hidden: Option<&CudaSlice<bf16>>,
        residual: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        check_hidden_conv(weight, state)?;
        ensure!(
            x.len() == HIDDEN
                && residual.len() == HIDDEN
                && hidden.is_none_or(|h| h.len() == HIDDEN),
            "x, hidden and residual must hold {HIDDEN} elements"
        );
        let mut args = KernelArgs::default();
        args.ptr(x, stream)
            .ptr(weight, stream)
            .ptr_mut(state, stream)
            .opt(hidden, stream)
            .ptr_mut(residual, stream);
        // SAFETY: the arguments match `short_conv_residual_2304(const bf16*, const bf16*, float*,
        // const bf16*, float*)`; every buffer has the length checked above, a null `hidden` is
        // never read, and the kernel reads nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.short_conv_residual_2304, hidden_config()) }
    }

    /// The end of the MoE block and the layer for one row: the combine of the expert outputs
    /// `experts` (`[8, 2304]`) with `weights` (`[8]`), the post-MLP conv, and `out =
    /// bf16(residual + conv)`.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn moe_conv_residual(
        &self,
        stream: &CudaStream,
        experts: &CudaSlice<bf16>,
        weights: &CudaSlice<f32>,
        weight: &CudaSlice<bf16>,
        state: &mut CudaSlice<f32>,
        residual: &CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        check_hidden_conv(weight, state)?;
        ensure!(
            experts.len() == TOP_K * HIDDEN && weights.len() == TOP_K,
            "experts has {} and weights {} elements",
            experts.len(),
            weights.len()
        );
        ensure!(
            residual.len() == HIDDEN && out.len() == HIDDEN,
            "residual and out must hold {HIDDEN} elements"
        );
        let mut args = KernelArgs::default();
        args.ptr(experts, stream)
            .ptr(weights, stream)
            .ptr(weight, stream)
            .ptr_mut(state, stream)
            .ptr(residual, stream)
            .ptr_mut(out, stream);
        // SAFETY: the arguments match `moe_conv_residual_2304(const bf16*, const float*,
        // const bf16*, float*, const float*, bf16*)`, every buffer has the length checked
        // above, and the kernel reads nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.moe_conv_residual_2304, hidden_config()) }
    }
}

/// Checks that `x` holds whole rows of `dims` channels and `out` matches it, and returns the
/// row count as the kernels' `int`.
fn check_rows(x: &CudaSlice<bf16>, out: &CudaSlice<bf16>, dims: usize) -> anyhow::Result<i32> {
    ensure!(
        x.len().is_multiple_of(dims),
        "x has {} elements, not a multiple of {dims}",
        x.len()
    );
    ensure!(
        out.len() == x.len(),
        "out has {} elements, x has {}",
        out.len(),
        x.len()
    );
    i32::try_from(x.len() / dims).context("too many rows")
}

/// One thread per channel.
fn channel_config(dims: usize) -> anyhow::Result<LaunchConfig> {
    let blocks = u32::try_from(dims.div_ceil(BLOCK as usize)).context("too many channels")?;
    Ok(LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}
