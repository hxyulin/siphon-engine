//! Message-gate combine, casts and argmax (`kernels/misc.cu`).

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module};

/// Threads per block for the elementwise kernels.
const BLOCK: u32 = 256;
/// Upper bound on blocks for the grid-stride kernels.
const MAX_BLOCKS: usize = 1 << 16;
/// Threads per block of the argmax kernel, one block per row.
const ARGMAX_THREADS: u32 = 1024;
/// Hidden width and message-gate rank of the model.
const HIDDEN: usize = 2304;
const GATE_RANK: usize = 64;
/// Blocks and threads of `message_gate_out_2304`: 72 output rows per block.
const GATE_OUT_BLOCKS: u32 = (HIDDEN / 72) as u32;
const GATE_OUT_THREADS: u32 = 288;

/// The norm of [`Misc::message_gate_out_rows`]: its weight, the normalized rows and one
/// counter per row.
pub type GateNorm<'a> = (
    &'a CudaSlice<bf16>,
    &'a mut dyn DeviceOut<bf16>,
    &'a mut dyn DeviceOut<u32>,
);

#[derive(Debug)]
pub struct Misc {
    message_gate_combine_f32: CudaFunction,
    cast_f32_to_bf16: CudaFunction,
    cast_bf16_to_f32: CudaFunction,
    argmax_f32: CudaFunction,
    message_gate_out_2304: CudaFunction,
    message_gate_out_rows_2304: CudaFunction,
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
            message_gate_out_2304: module.load_function("message_gate_out_2304")?,
            message_gate_out_rows_2304: module.load_function("message_gate_out_rows_2304")?,
        })
    }

    /// `out = (1 - tanh(f_pre)) * h + tanh(g_pre) * m`, the last step of `MessageGate`.
    ///
    /// All five slices have the same length, normally `[rows, 2304]`. `f_pre` and `g_pre`
    /// are the forget and gain projections before their tanh.
    pub fn message_gate_combine(
        &self,
        stream: &CudaStream,
        h: &impl DeviceIn<f32>,
        m: &impl DeviceIn<f32>,
        f_pre: &CudaSlice<f32>,
        g_pre: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        let (h, m) = (h.view(), m.view());
        let n = h.len();
        check_len("m", m.len(), n)?;
        check_len("f_pre", f_pre.len(), n)?;
        check_len("g_pre", g_pre.len(), n)?;
        check_len("out", out.len(), n)?;
        let mut launch = stream.launch_builder(&self.message_gate_combine_f32);
        launch
            .arg(&h)
            .arg(&m)
            .arg(f_pre)
            .arg(g_pre)
            .arg(out)
            .arg(&n);
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
        x: &impl DeviceIn<f32>,
        out: &mut impl DeviceOut<i32>,
        n: usize,
    ) -> anyhow::Result<()> {
        let (x, mut out) = (x.view(), out.view_mut());
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
        launch.arg(&x).arg(&mut out).arg(&n_arg);
        // SAFETY: the arguments match `argmax_f32(const float*, int*, int)`. `x` holds `rows`
        // rows of `n` and `out` holds `rows` elements, one per block.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// The end of `MessageGate` for one row from its BF16 down projection `down` (`[64]`):
    /// the forget and gain projections (BF16 weights `[2304, 64]` with an F32 input) and
    /// the combine into `out`, as [`Misc::message_gate_combine`] after the F32 GEMVs. Each
    /// row sums in the order of MLX's GEMV for k = 64; tanh is CUDA's `tanhf`.
    ///
    /// With `norm` set to `(weight, normed, counter)`, also writes `normed =
    /// bf16(rms_norm(out) * weight)`. `counter` must hold 0; the kernel leaves it at 0.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn message_gate_out(
        &self,
        stream: &CudaStream,
        down: &CudaSlice<bf16>,
        forget: &CudaSlice<bf16>,
        gain: &CudaSlice<bf16>,
        h: &impl DeviceIn<f32>,
        m: &impl DeviceIn<f32>,
        out: &mut CudaSlice<f32>,
        norm: Option<(&CudaSlice<bf16>, &mut CudaSlice<bf16>, &mut CudaSlice<u32>)>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let (h, m) = (h.view(), m.view());
        check_len("down", down.len(), GATE_RANK)?;
        check_len("forget", forget.len(), HIDDEN * GATE_RANK)?;
        check_len("gain", gain.len(), HIDDEN * GATE_RANK)?;
        check_len("h", h.len(), HIDDEN)?;
        check_len("m", m.len(), HIDDEN)?;
        check_len("out", out.len(), HIDDEN)?;
        let config = LaunchConfig {
            grid_dim: (GATE_OUT_BLOCKS, 1, 1),
            block_dim: (GATE_OUT_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(down, stream)
            .ptr(forget, stream)
            .ptr(gain, stream)
            .ptr(&h, stream)
            .ptr(&m, stream)
            .ptr_mut(out, stream);
        match norm {
            Some((weight, normed, counter)) => {
                check_len("norm weight", weight.len(), HIDDEN)?;
                check_len("normed", normed.len(), HIDDEN)?;
                ensure!(!counter.is_empty(), "counter is empty");
                args.ptr(weight, stream)
                    .ptr_mut(normed, stream)
                    .ptr_mut(counter, stream);
            }
            None => {
                args.u64(0).u64(0).u64(0);
            }
        }
        args.f32(eps);
        // SAFETY: the arguments match `message_gate_out_2304(const bf16*, const bf16*,
        // const bf16*, const float*, const float*, float*, const bf16*, bf16*, unsigned*,
        // float)`, every buffer has the length checked above, a null `normed` makes the
        // kernel skip the norm and its buffers, and it reads only weights before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.message_gate_out_2304, config) }
    }

    /// [`Misc::message_gate_out`] for each of the `rows` rows of a chunk: `down`
    /// `[rows, 64]`, `out` `[rows, 2304]`, and the rows of `h` and `m` `h_stride` and
    /// `m_stride` elements apart. With `norm` set to `(weight, normed, counters)`, `normed`
    /// is `[rows, 2304]` and `counters` `[rows]`, each holding 0 and left at 0. Each row's
    /// results equal a single-row call's bit for bit.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn message_gate_out_rows(
        &self,
        stream: &CudaStream,
        down: &impl DeviceIn<bf16>,
        forget: &CudaSlice<bf16>,
        gain: &CudaSlice<bf16>,
        (h, h_stride): (&impl DeviceIn<f32>, usize),
        (m, m_stride): (&impl DeviceIn<f32>, usize),
        out: &mut impl DeviceOut<f32>,
        norm: Option<GateNorm<'_>>,
        eps: f32,
        rows: usize,
    ) -> anyhow::Result<()> {
        let (down, h, m, mut out) = (down.view(), h.view(), m.view(), out.view_mut());
        ensure!(rows > 0, "no rows");
        check_len("down", down.len(), rows * GATE_RANK)?;
        check_len("forget", forget.len(), HIDDEN * GATE_RANK)?;
        check_len("gain", gain.len(), HIDDEN * GATE_RANK)?;
        for (name, len, stride) in [("h", h.len(), h_stride), ("m", m.len(), m_stride)] {
            ensure!(
                stride >= HIDDEN && len >= (rows - 1) * stride + HIDDEN,
                "{name} has {len} elements, too few for {rows} rows {stride} apart"
            );
        }
        check_len("out", out.len(), rows * HIDDEN)?;
        let config = LaunchConfig {
            grid_dim: (
                GATE_OUT_BLOCKS,
                u32::try_from(rows).context("too many rows")?,
                1,
            ),
            block_dim: (GATE_OUT_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&down, stream)
            .ptr(forget, stream)
            .ptr(gain, stream)
            .ptr(&h, stream)
            .i32(i32::try_from(h_stride)?)
            .ptr(&m, stream)
            .i32(i32::try_from(m_stride)?)
            .ptr_mut(&mut out, stream);
        match norm {
            Some((weight, normed, counters)) => {
                let (mut normed, mut counters) = (normed.view_mut(), counters.view_mut());
                check_len("norm weight", weight.len(), HIDDEN)?;
                check_len("normed", normed.len(), rows * HIDDEN)?;
                check_len("counters", counters.len(), rows)?;
                args.ptr(weight, stream)
                    .ptr_mut(&mut normed, stream)
                    .ptr_mut(&mut counters, stream);
            }
            None => {
                args.u64(0).u64(0).u64(0);
            }
        }
        args.f32(eps);
        // SAFETY: the arguments match `message_gate_out_rows_2304(const bf16*, const bf16*,
        // const bf16*, const float*, int, const float*, int, float*, const bf16*, bf16*,
        // unsigned*, float)`, every buffer holds `rows` rows as checked above, the grid has
        // one y index per row, a null `normed` makes the kernel skip the norm and its
        // buffers, and it reads only weights before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.message_gate_out_rows_2304, config) }
    }
}
