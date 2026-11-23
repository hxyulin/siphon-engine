//! Message-gate combine, casts and argmax (`kernels/misc.cu`).

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig,
    PushKernelArg as _,
};
use half::bf16;

use crate::{
    gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module},
    ops::linear_fp8::Fp8Matrix,
};

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
/// Tile of `message_gate_out_rows_2304`: rows by output columns, with 256 threads.
const GATE_TILE_ROWS: usize = 32;
const GATE_TILE_COLS: usize = 64;
const GATE_TILE_THREADS: u32 = 256;

/// Gates [`Misc::message_gate_out`] runs in one launch.
pub const GATE_SLOTS: usize = 3;
/// Threads, values per thread and most blocks of `argmax_wide_f32`.
const ARGMAX_WIDE_THREADS: u32 = 256;
const ARGMAX_WIDE_PER_THREAD: usize = 2;
const ARGMAX_WIDE_BLOCKS: usize = 256;

/// One gate of [`Misc::message_gate_out`]: its forget and gain weights, and optionally the
/// weight of the norm after it and the normalized output.
pub struct GateSlot<'a> {
    pub forget: &'a CudaSlice<bf16>,
    pub gain: &'a CudaSlice<bf16>,
    pub norm: Option<(&'a CudaSlice<bf16>, CudaViewMut<'a, bf16>)>,
}

/// One gate of [`Misc::message_gate_step`]: its weights, its down projection's input, and
/// optionally the weight of the norm after it and the normalized output.
pub struct GateStep<'a> {
    /// `[64, 4608]`.
    pub down_weight: &'a CudaSlice<bf16>,
    /// The normalized state and message side by side, BF16 `[4608]`.
    pub input: CudaView<'a, bf16>,
    pub forget: &'a CudaSlice<bf16>,
    pub gain: &'a CudaSlice<bf16>,
    /// FP8 copies of the down, forget and gain weights, which the gate then reads instead.
    /// Every gate of a launch has them or none does.
    pub fp8: Option<&'a [Fp8Matrix; 3]>,
    pub norm: Option<(&'a CudaSlice<bf16>, CudaViewMut<'a, bf16>)>,
}

/// The cross-block sums and barrier counters of [`Misc::message_gate_step`].
pub struct GateStepScratch {
    partials: CudaSlice<f32>,
    /// One per gate, a multiple of the blocks per gate between launches.
    counters: CudaSlice<u32>,
}

impl GateStepScratch {
    pub fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        Ok(Self {
            partials: stream
                .alloc_zeros(GATE_SLOTS * GATE_OUT_BLOCKS as usize * (GATE_RANK + 1))?,
            counters: stream.alloc_zeros(GATE_SLOTS)?,
        })
    }
}

/// The partial results of [`Misc::argmax_wide_f32`].
pub struct ArgmaxScratch {
    vals: CudaSlice<f32>,
    idxs: CudaSlice<i32>,
    /// The kernel's block counter, 0 between launches.
    counter: CudaSlice<u32>,
}

impl ArgmaxScratch {
    pub fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        Ok(Self {
            vals: stream.alloc_zeros(ARGMAX_WIDE_BLOCKS)?,
            idxs: stream.alloc_zeros(ARGMAX_WIDE_BLOCKS)?,
            counter: stream.alloc_zeros(1)?,
        })
    }
}

/// The norm of [`Misc::message_gate_out_rows`]: its weight and the normalized rows.
pub type GateNorm<'a> = (&'a CudaSlice<bf16>, &'a mut dyn DeviceOut<bf16>);

#[derive(Debug)]
pub struct Misc {
    message_gate_combine_f32: CudaFunction,
    cast_f32_to_bf16: CudaFunction,
    cast_bf16_to_f32: CudaFunction,
    argmax_f32: CudaFunction,
    message_gate_out_2304: CudaFunction,
    message_gate_out_rows_2304: CudaFunction,
    message_gate_norm_rows_2304: CudaFunction,
    argmax_wide_f32: CudaFunction,
    message_gate_step_2304: CudaFunction,
    message_gate_step_fp8_2304: CudaFunction,
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
            message_gate_norm_rows_2304: module.load_function("message_gate_norm_rows_2304")?,
            argmax_wide_f32: module.load_function("argmax_wide_f32")?,
            message_gate_step_2304: module.load_function("message_gate_step_2304")?,
            message_gate_step_fp8_2304: module.load_function("message_gate_step_fp8_2304")?,
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

    /// The end of `MessageGate` for one row from its BF16 down projection: the forget and
    /// gain projections (BF16 weights `[2304, 64]` with an F32 input) and the combine into
    /// `out`, as [`Misc::message_gate_combine`] after the F32 GEMVs. Each row sums in the
    /// order of MLX's GEMV for k = 64; tanh is CUDA's `tanhf`.
    ///
    /// Runs up to three gates in one launch, one per entry of `slots`. Gate `s` reads
    /// `down[64 s..]`, `h[s * h_stride..]` and `m[2304 s..]`, and writes `out[2304 s..]`.
    /// With a norm, it also writes `normed = bf16(rms_norm(out) * weight)`, using
    /// `counter[s]`, which must hold 0; the kernel leaves it at 0.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn message_gate_out(
        &self,
        stream: &CudaStream,
        down: &impl DeviceIn<bf16>,
        slots: &mut [GateSlot<'_>],
        h: &impl DeviceIn<f32>,
        h_stride: usize,
        m: &impl DeviceIn<f32>,
        out: &mut impl DeviceOut<f32>,
        counter: &mut CudaSlice<u32>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let (down, h, m, mut out) = (down.view(), h.view(), m.view(), out.view_mut());
        let n = slots.len();
        ensure!(
            (1..=GATE_SLOTS).contains(&n),
            "{n} gates; one launch takes 1 to {GATE_SLOTS}"
        );
        check_len("down", down.len(), n * GATE_RANK)?;
        ensure!(
            h.len() >= (n - 1) * h_stride + HIDDEN,
            "h has {} elements, too few for {n} rows {h_stride} apart",
            h.len()
        );
        check_len("m", m.len(), n * HIDDEN)?;
        check_len("out", out.len(), n * HIDDEN)?;
        ensure!(counter.len() >= n, "counter needs {n} elements");
        for slot in &*slots {
            check_len("forget", slot.forget.len(), HIDDEN * GATE_RANK)?;
            check_len("gain", slot.gain.len(), HIDDEN * GATE_RANK)?;
            if let Some((weight, normed)) = &slot.norm {
                check_len("norm weight", weight.len(), HIDDEN)?;
                check_len("normed", normed.len(), HIDDEN)?;
            }
        }
        let config = LaunchConfig {
            grid_dim: (GATE_OUT_BLOCKS, u32::try_from(n)?, 1),
            block_dim: (GATE_OUT_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&down, stream);
        for k in 0..GATE_SLOTS {
            args.ptr(slots[k.min(n - 1)].forget, stream);
        }
        for k in 0..GATE_SLOTS {
            args.ptr(slots[k.min(n - 1)].gain, stream);
        }
        args.ptr(&h, stream)
            .i32(i32::try_from(h_stride).context("h_stride too large")?)
            .ptr(&m, stream)
            .ptr_mut(&mut out, stream);
        for k in 0..GATE_SLOTS {
            args.opt(
                slots.get(k).and_then(|s| s.norm.as_ref()).map(|n| n.0),
                stream,
            );
        }
        for k in 0..GATE_SLOTS {
            args.opt_mut(
                slots
                    .get_mut(k)
                    .and_then(|s| s.norm.as_mut())
                    .map(|n| &mut n.1),
                stream,
            );
        }
        args.ptr_mut(counter, stream).f32(eps);
        // SAFETY: the arguments match `message_gate_out_2304`'s signature, every buffer has
        // the length checked above for the `n` gates of the grid's y dimension, a null
        // `normed` makes a gate skip the norm, and the kernel reads only weights before
        // `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.message_gate_out_2304, config) }
    }

    /// The whole `MessageGate` for one row and up to three gates in one launch: the down
    /// projection of each gate's input, rounded to BF16, then the end of
    /// [`Misc::message_gate_out`] from it. The down projection sums in another order than
    /// [`crate::ops::linear::Linear::gemv_bf16`], and the norm's sum of squares in another
    /// order than [`Misc::message_gate_out`]'s.
    ///
    /// Gate `k` reads `h[k * h_stride..]` and `m[2304 k..]` and writes `out[2304 k..]`.
    /// `down`, if given, receives the BF16 down projections, `[gates, 64]`.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn message_gate_step(
        &self,
        stream: &CudaStream,
        gates: &mut [GateStep<'_>],
        h: &impl DeviceIn<f32>,
        h_stride: usize,
        m: &impl DeviceIn<f32>,
        out: &mut impl DeviceOut<f32>,
        down: Option<&mut CudaSlice<bf16>>,
        scratch: &mut GateStepScratch,
        eps: f32,
    ) -> anyhow::Result<()> {
        let (h, m, mut out) = (h.view(), m.view(), out.view_mut());
        let n = gates.len();
        ensure!(
            (1..=GATE_SLOTS).contains(&n),
            "{n} gates; one launch takes 1 to {GATE_SLOTS}"
        );
        ensure!(
            h.len() >= (n - 1) * h_stride + HIDDEN,
            "h has {} elements, too few for {n} rows {h_stride} apart",
            h.len()
        );
        check_len("m", m.len(), n * HIDDEN)?;
        check_len("out", out.len(), n * HIDDEN)?;
        if let Some(down) = &down {
            check_len("down", down.len(), n * GATE_RANK)?;
        }
        for gate in &*gates {
            check_len(
                "down weight",
                gate.down_weight.len(),
                GATE_RANK * 2 * HIDDEN,
            )?;
            check_len("input", gate.input.len(), 2 * HIDDEN)?;
            check_len("forget", gate.forget.len(), HIDDEN * GATE_RANK)?;
            check_len("gain", gate.gain.len(), HIDDEN * GATE_RANK)?;
            if let Some([down, forget, gain]) = gate.fp8 {
                ensure!(
                    (down.rows(), down.k()) == (GATE_RANK, 2 * HIDDEN)
                        && [forget, gain]
                            .iter()
                            .all(|m| (m.rows(), m.k()) == (HIDDEN, GATE_RANK)),
                    "FP8 gate weight shapes"
                );
            }
            if let Some((weight, normed)) = &gate.norm {
                check_len("norm weight", weight.len(), HIDDEN)?;
                check_len("normed", normed.len(), HIDDEN)?;
            }
        }
        let config = LaunchConfig {
            grid_dim: (GATE_OUT_BLOCKS, u32::try_from(n)?, 1),
            block_dim: (GATE_OUT_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        let fp8 = push_gate_weights(&mut args, gates, stream)?;
        args.ptr(&h, stream)
            .i32(i32::try_from(h_stride).context("h_stride too large")?)
            .ptr(&m, stream)
            .ptr_mut(&mut out, stream);
        for k in 0..GATE_SLOTS {
            args.opt(
                gates.get(k).and_then(|g| g.norm.as_ref()).map(|n| n.0),
                stream,
            );
        }
        for k in 0..GATE_SLOTS {
            args.opt_mut(
                gates
                    .get_mut(k)
                    .and_then(|g| g.norm.as_mut())
                    .map(|n| &mut n.1),
                stream,
            );
        }
        args.opt_mut(down, stream)
            .ptr_mut(&mut scratch.partials, stream)
            .ptr_mut(&mut scratch.counters, stream)
            .f32(eps);
        let function = if fp8 {
            &self.message_gate_step_fp8_2304
        } else {
            &self.message_gate_step_2304
        };
        // SAFETY: the arguments match `message_gate_step_2304`'s signature, or with FP8
        // weights `message_gate_step_fp8_2304`'s, every buffer has the length checked above
        // for the `n` gates of the grid's y dimension, the scratch holds 32 x 65 partial sums
        // and a barrier counter per gate, a null `normed` makes a gate skip the norm, and the
        // kernel reads only weights before `pdl_wait`. Its barriers need every block
        // resident, which 32 blocks of 288 threads per gate are.
        unsafe { args.launch_pdl(stream, function, config) }
    }

    /// The index of the largest of the values of one long F32 row `x`, as
    /// [`Misc::argmax_f32`] with one row, over many blocks.
    pub fn argmax_wide_f32(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<f32>,
        out: &mut impl DeviceOut<i32>,
        scratch: &mut ArgmaxScratch,
    ) -> anyhow::Result<()> {
        let (x, mut out) = (x.view(), out.view_mut());
        ensure!(!x.is_empty(), "x is empty");
        check_len("out", out.len(), 1)?;
        let n = i32::try_from(x.len()).context("x too long")?;
        let blocks = x
            .len()
            .div_ceil(4 * ARGMAX_WIDE_PER_THREAD * ARGMAX_WIDE_THREADS as usize)
            .min(ARGMAX_WIDE_BLOCKS);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(blocks)?, 1, 1),
            block_dim: (ARGMAX_WIDE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .i32(n)
            .ptr_mut(&mut scratch.vals, stream)
            .ptr_mut(&mut scratch.idxs, stream)
            .ptr_mut(&mut scratch.counter, stream)
            .ptr_mut(&mut out, stream);
        // SAFETY: the arguments match `argmax_wide_f32(const float*, int, float*, int*,
        // unsigned*, int*)`, the scratch holds a partial for each of the at most 256 blocks,
        // a fresh buffer is 256-byte aligned, and the kernel reads nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.argmax_wide_f32, config) }
    }

    /// [`Misc::message_gate_out`] for each of the `rows` rows of a chunk: `down`
    /// `[rows, 64]`, `out` `[rows, 2304]`, and the rows of `h` and `m` `h_stride` and
    /// `m_stride` elements apart. With `norm` set to `(weight, normed)`, `normed` is
    /// `[rows, 2304]`. Each row's results equal a single-row call's bit for bit.
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
            // The kernel reads pairs of floats.
            ensure!(stride.is_multiple_of(2), "{name} stride {stride} is odd");
        }
        check_len("out", out.len(), rows * HIDDEN)?;
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(HIDDEN / GATE_TILE_COLS)?,
                u32::try_from(rows.div_ceil(GATE_TILE_ROWS)).context("too many rows")?,
                1,
            ),
            block_dim: (GATE_TILE_THREADS, 1, 1),
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
            .ptr_mut(&mut out, stream)
            .i32(i32::try_from(rows).context("too many rows")?);
        // SAFETY: the arguments match `message_gate_out_rows_2304(const bf16*, const bf16*,
        // const bf16*, const float*, int, const float*, int, float*, int)`, every buffer
        // holds `rows` rows as checked above, the grid covers the rows in tiles of 32 (the
        // kernel skips rows past `rows`), and it reads only weights before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.message_gate_out_rows_2304, config) }?;

        let Some((weight, normed)) = norm else {
            return Ok(());
        };
        let mut normed = normed.view_mut();
        check_len("norm weight", weight.len(), HIDDEN)?;
        check_len("normed", normed.len(), rows * HIDDEN)?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: (GATE_OUT_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&out, stream)
            .ptr(weight, stream)
            .ptr_mut(&mut normed, stream)
            .f32(eps);
        // SAFETY: the arguments match `message_gate_norm_rows_2304(const float*, const bf16*,
        // bf16*, float)`, out and normed hold `rows` rows of 2304, the grid has one block per
        // row, and the kernel reads nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.message_gate_norm_rows_2304, config) }
    }
}

/// The weight and input arguments of `message_gate_step_2304`, or of
/// `message_gate_step_fp8_2304` if the gates have FP8 weights, which it returns. Slots past
/// the last gate repeat it.
fn push_gate_weights(
    args: &mut KernelArgs,
    gates: &[GateStep<'_>],
    stream: &CudaStream,
) -> anyhow::Result<bool> {
    let n = gates.len();
    let fp8 = gates[0].fp8.is_some();
    ensure!(
        gates.iter().all(|g| g.fp8.is_some() == fp8),
        "every gate of a launch has FP8 weights or none does"
    );
    let slot = |k: usize| &gates[k.min(n - 1)];
    if fp8 {
        // Each matrix's E4M3 values, then its row scales; the inputs follow the down
        // projection's.
        for matrix in 0..3 {
            for k in 0..GATE_SLOTS {
                args.ptr(&slot(k).fp8.context("FP8 gate")?[matrix].values(), stream);
            }
            for k in 0..GATE_SLOTS {
                args.ptr(&slot(k).fp8.context("FP8 gate")?[matrix].scales(), stream);
            }
            if matrix == 0 {
                for k in 0..GATE_SLOTS {
                    args.ptr(&slot(k).input, stream);
                }
            }
        }
    } else {
        for k in 0..GATE_SLOTS {
            args.ptr(slot(k).down_weight, stream);
        }
        for k in 0..GATE_SLOTS {
            args.ptr(&slot(k).input, stream);
        }
        for k in 0..GATE_SLOTS {
            args.ptr(slot(k).forget, stream);
        }
        for k in 0..GATE_SLOTS {
            args.ptr(slot(k).gain, stream);
        }
    }
    Ok(fp8)
}
