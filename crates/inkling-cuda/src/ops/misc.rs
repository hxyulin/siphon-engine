//! Message-gate combine, casts and argmax (`kernels/misc.cu`).

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{cubin, load_module};

/// Threads per block for the elementwise kernels.
const BLOCK: u32 = 256;
/// Upper bound on blocks for the grid-stride kernels.
const MAX_BLOCKS: usize = 1 << 16;
/// Threads per block of the argmax kernel, one block per row.
const ARGMAX_THREADS: u32 = 1024;

#[derive(Debug)]
pub struct Misc {
    message_gate_combine_f32: CudaFunction,
    cast_f32_to_bf16: CudaFunction,
    cast_bf16_to_f32: CudaFunction,
    argmax_f32: CudaFunction,
}

/// Launch shape for a grid-stride loop over `n` elements.
fn grid_stride(n: usize) -> LaunchConfig {
    let blocks = n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS);
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn check_len(name: &str, len: usize, want: usize) -> anyhow::Result<()> {
    ensure!(len == want, "{name} has {len} elements, expected {want}");
    Ok(())
}

impl Misc {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "misc", cubin!("misc"))?;
        Ok(Self {
            message_gate_combine_f32: module.load_function("message_gate_combine_f32")?,
            cast_f32_to_bf16: module.load_function("cast_f32_to_bf16")?,
            cast_bf16_to_f32: module.load_function("cast_bf16_to_f32")?,
            argmax_f32: module.load_function("argmax_f32")?,
        })
    }

    /// `out = (1 - tanh(f_pre)) * h + tanh(g_pre) * m`, the last step of `MessageGate`.
    ///
    /// All five slices have the same length, normally `[rows, 2304]`. `f_pre` and `g_pre`
    /// are the forget and gain projections before their tanh.
    pub fn message_gate_combine(
        &self,
        stream: &CudaStream,
        h: &CudaSlice<f32>,
        m: &CudaSlice<f32>,
        f_pre: &CudaSlice<f32>,
        g_pre: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        let n = h.len();
        check_len("m", m.len(), n)?;
        check_len("f_pre", f_pre.len(), n)?;
        check_len("g_pre", g_pre.len(), n)?;
        check_len("out", out.len(), n)?;
        let mut launch = stream.launch_builder(&self.message_gate_combine_f32);
        launch.arg(h).arg(m).arg(f_pre).arg(g_pre).arg(out).arg(&n);
        // SAFETY: the arguments match `message_gate_combine_f32(const float*, const float*,
        // const float*, const float*, float*, size_t)`, and every slice holds `n` elements.
        unsafe { launch.launch(grid_stride(n)) }?;
        Ok(())
    }

    /// Rounds each F32 value of `x` to BF16 in `out`, as MLX's `astype(mx.bfloat16)` does.
    pub fn cast_f32_to_bf16(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        let n = x.len();
        check_len("out", out.len(), n)?;
        let mut launch = stream.launch_builder(&self.cast_f32_to_bf16);
        launch.arg(x).arg(out).arg(&n);
        // SAFETY: the arguments match `cast_f32_to_bf16(const float*, uint16_t*, size_t)`; BF16
        // is passed as its 16 bits, and both slices hold `n` elements.
        unsafe { launch.launch(grid_stride(n)) }?;
        Ok(())
    }

    /// Widens each BF16 value of `x` to F32 in `out`.
    pub fn cast_bf16_to_f32(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        out: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        let n = x.len();
        check_len("out", out.len(), n)?;
        let mut launch = stream.launch_builder(&self.cast_bf16_to_f32);
        launch.arg(x).arg(out).arg(&n);
        // SAFETY: the arguments match `cast_bf16_to_f32(const uint16_t*, float*, size_t)`; BF16
        // is passed as its 16 bits, and both slices hold `n` elements.
        unsafe { launch.launch(grid_stride(n)) }?;
        Ok(())
    }

    /// Writes the index of the largest value in each row of `x` (shape `[rows, n]`) to `out`
    /// (shape `[rows]`). Ties go to the first index and NaN counts as the largest value, as in
    /// `mx.argmax`.
    pub fn argmax_f32(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<i32>,
        n: usize,
    ) -> anyhow::Result<()> {
        ensure!(n > 0, "n must be positive");
        ensure!(
            x.len().is_multiple_of(n),
            "x has {} elements, not a multiple of {n}",
            x.len()
        );
        let rows = x.len() / n;
        check_len("out", out.len(), rows)?;
        let n_arg = i32::try_from(n).context("n too large")?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: (ARGMAX_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.argmax_f32);
        launch.arg(x).arg(out).arg(&n_arg);
        // SAFETY: the arguments match `argmax_f32(const float*, int*, int)`. `x` holds `rows`
        // rows of `n` and `out` holds `rows` elements, one per block.
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}
