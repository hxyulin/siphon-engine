//! Message-gate combine, casts and argmax (`kernels/misc.cu`).

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    DevicePtr as _, DevicePtrMut as _, LaunchConfig, PushKernelArg as _, sys::CUfunction_attribute,
};
use half::bf16;

use crate::{
    gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module},
    ops::{QuantDest, linear::launch_by_value, linear_fp8::Fp8Matrix},
};

/// Threads per block for the elementwise kernels.
const BLOCK: u32 = 256;
/// Upper bound on blocks for the grid-stride kernels.
const MAX_BLOCKS: usize = 1 << 16;
/// Threads per block of the argmax kernel, one block per row.
const ARGMAX_THREADS: u32 = 1024;
/// Threads per block of `argmax_partials`, one block per row.
const ARGMAX_PARTIALS_THREADS: u32 = 256;
/// Hidden width and message-gate rank of the model.
const HIDDEN: usize = 2304;
const GATE_RANK: usize = 64;
/// Blocks and threads of `message_gate_out_2304`: 72 output rows per block.
const GATE_OUT_BLOCKS: u32 = (HIDDEN / 72) as u32;
const GATE_OUT_THREADS: u32 = 288;
/// Blocks per gate of `message_gate_step`, one cluster, `kGateCluster` in the kernel.
const GATE_CLUSTER: u32 = 16;
/// Tile of `message_gate_out_rows_2304`: rows by output columns, with 256 threads.
const GATE_TILE_ROWS: usize = 64;
const GATE_TILE_COLS: usize = 64;
const GATE_TILE_THREADS: u32 = 256;

/// Block states a message rebuilt by [`Misc::message_gate_out_rows`] can mix.
const GATE_MAX_BLOCKS: usize = 5;
/// Rows of one cluster of [`Misc::message_gate_rows`], `gate_rows::kMaxRows` in the kernel.
const GATE_CLUSTER_ROWS: usize = 8;
/// Most rows of [`Misc::message_gate_rows`]: 32 clusters per gate.
pub const GATE_ROWS_MAX: usize = 32 * GATE_CLUSTER_ROWS;
/// Dynamic shared memory of `message_gate_rows_2304` and `message_gate_rows_fp8_2304`,
/// `gate_rows::smem_bytes` in the kernel: the down stages of two warp splits (5 of 4 KiB
/// each for BF16 weights, 3 for FP8), then the block's forget and gain rows.
const GATE_ROWS_SMEM_BF16: u32 = 2 * 5 * 4096 + 2 * 144 * 64 * 2;
const GATE_ROWS_SMEM_FP8: u32 = 2 * 3 * 4096 + 2 * 144 * 64 * 2;
/// Gates [`Misc::message_gate_out`] runs in one launch.
pub const GATE_SLOTS: usize = 3;
const _: () = assert!(
    GATE_SLOTS == 3,
    "siphon::memory counts 3 gate outputs per verify launch"
);
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

/// One gate of [`Misc::message_gate_rows`].
pub struct GateRowsSlot<'a> {
    /// `[64, 4608]`.
    pub down: &'a CudaSlice<bf16>,
    /// The down weight's FP8 copy, which the gate reads instead. Every gate of a launch has
    /// one or none does.
    pub down_fp8: Option<&'a Fp8Matrix>,
    /// `[2304, 64]` each.
    pub forget: &'a CudaSlice<bf16>,
    pub gain: &'a CudaSlice<bf16>,
    /// The down projection's input: the normalized state and the normalized message, BF16
    /// rows of 2304 with the row strides of [`Misc::message_gate_rows`].
    pub x_state: CudaView<'a, bf16>,
    pub x_message: CudaView<'a, bf16>,
    /// The gate's state, rows `h_stride` apart.
    pub h: CudaView<'a, f32>,
    /// Each row's message weights, as in [`MessageMix`].
    pub probs: CudaView<'a, f32>,
    /// The norm weight and the normalized output, `[rows, 2304]`.
    pub norm: (&'a CudaSlice<bf16>, CudaViewMut<'a, bf16>),
    /// Receives the gate's output before the norm, F32 `[rows, 2304]`.
    pub out: CudaViewMut<'a, f32>,
}

/// The launch shape [`GateRowsSlot::push`] checks a gate's buffers against.
struct GateRowsShape {
    rows: usize,
    nb: usize,
    probs_stride: usize,
    x_strides: [usize; 2],
    h_stride: usize,
}

impl GateRowsSlot<'_> {
    /// Checks the gate's buffers against `shape` and writes their pointers into slot `k`
    /// of `args`.
    fn push(
        &mut self,
        args: &mut GateRowsArgs,
        k: usize,
        shape: &GateRowsShape,
        stream: &CudaStream,
    ) -> anyhow::Result<()> {
        let GateRowsShape {
            rows,
            nb,
            probs_stride,
            x_strides,
            h_stride,
        } = *shape;
        check_len("down weight", self.down.len(), GATE_RANK * 2 * HIDDEN)?;
        check_len("forget", self.forget.len(), HIDDEN * GATE_RANK)?;
        check_len("gain", self.gain.len(), HIDDEN * GATE_RANK)?;
        for (name, x, stride) in [
            ("x_state", &self.x_state, x_strides[0]),
            ("x_message", &self.x_message, x_strides[1]),
        ] {
            ensure!(
                x.len() >= (rows - 1) * stride + HIDDEN,
                "{name} has {} elements, too few for {rows} rows {stride} apart",
                x.len()
            );
        }
        ensure!(
            self.h.len() >= (rows - 1) * h_stride + HIDDEN,
            "h has {} elements, too few for {rows} rows {h_stride} apart",
            self.h.len()
        );
        ensure!(
            self.probs.len() > (rows - 1) * probs_stride + nb,
            "probs has {} elements, too few for {rows} rows {probs_stride} apart",
            self.probs.len()
        );
        check_len("norm weight", self.norm.0.len(), HIDDEN)?;
        check_len("normed", self.norm.1.len(), rows * HIDDEN)?;
        check_len("out", self.out.len(), rows * HIDDEN)?;
        let (down, scale) = match self.down_fp8 {
            Some(w) => {
                ensure!(
                    (w.rows(), w.k()) == (GATE_RANK, 2 * HIDDEN),
                    "FP8 down weight shape"
                );
                (
                    w.values().device_ptr(stream).0,
                    w.scales().device_ptr(stream).0,
                )
            }
            None => (self.down.device_ptr(stream).0, 0),
        };
        let x_state = self.x_state.device_ptr(stream).0;
        let x_message = self.x_message.device_ptr(stream).0;
        ensure!(
            (down | x_state | x_message).is_multiple_of(16),
            "the down weight and x must be 16-byte aligned"
        );
        args.g[k] = GateRowsGate {
            down,
            down_scale: scale,
            forget: self.forget.device_ptr(stream).0,
            gain: self.gain.device_ptr(stream).0,
            x_state,
            x_message,
            h: self.h.device_ptr(stream).0,
            probs: self.probs.device_ptr(stream).0,
            norm_weight: self.norm.0.device_ptr(stream).0,
            normed: self.norm.1.device_ptr_mut(stream).0,
            out: self.out.device_ptr_mut(stream).0,
        };
        Ok(())
    }
}

/// One gate of [`GateRowsArgs`], as `gate_rows::Gate` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GateRowsGate {
    down: u64,
    down_scale: u64,
    forget: u64,
    gain: u64,
    x_state: u64,
    x_message: u64,
    h: u64,
    probs: u64,
    norm_weight: u64,
    normed: u64,
    out: u64,
}

/// Kernel parameter of `message_gate_rows_2304`, as `gate_rows::GateRowsArgs` in the
/// kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GateRowsArgs {
    g: [GateRowsGate; GATE_SLOTS],
    m: u64,
    blocks: u64,
    x_state_stride: i32,
    x_message_stride: i32,
    h_stride: i32,
    m_stride: i32,
    block_stride: i32,
    nb: i32,
    probs_stride: i32,
    rows: i32,
    eps: f32,
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

/// A message that [`Misc::message_gate_out_rows`] rebuilds from its depth-attention
/// candidates and the weights that [`crate::ops::routing::DepthRows::probs`] received:
/// the `nb` block states of `[rows, 2304]` each, one after another, and the weights of
/// each row `probs_stride` elements apart. The partial candidate is the `m` argument.
pub struct MessageMix<'a> {
    pub blocks: CudaView<'a, f32>,
    pub probs: CudaView<'a, f32>,
    pub probs_stride: usize,
}

/// The norm of [`Misc::message_gate_out_rows`]: its weight and the normalized rows.
pub type GateNorm<'a> = (&'a CudaSlice<bf16>, &'a mut dyn DeviceOut<bf16>);

/// Fused norm + quantization epilogue of [`Misc::message_gate_out_rows_quant`]: optional
/// BF16 normalized rows (skip when only quantized rows are read downstream) and the
/// quantized destination.
pub struct GateNormQuant<'a> {
    pub weight: &'a CudaSlice<bf16>,
    pub normed: Option<CudaViewMut<'a, bf16>>,
    pub quant: QuantDest<'a>,
}

#[derive(Debug)]
pub struct Misc {
    message_gate_combine_f32: CudaFunction,
    cast_f32_to_bf16: CudaFunction,
    cast_bf16_to_f32: CudaFunction,
    argmax_f32: CudaFunction,
    argmax_partials: CudaFunction,
    message_gate_out_2304: CudaFunction,
    message_gate_out_rows_2304: CudaFunction,
    message_gate_norm_rows_2304: CudaFunction,
    message_gate_norm_quant_rows_2304: CudaFunction,
    argmax_wide_f32: CudaFunction,
    message_gate_step_2304: CudaFunction,
    message_gate_step_fp8_2304: CudaFunction,
    message_gate_rows_2304: CudaFunction,
    message_gate_rows_fp8_2304: CudaFunction,
    decode_prologue_2304: CudaFunction,
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

/// Loads a kernel compiled with a cluster of 16 blocks, above the portable limit of 8. The
/// sm_89 build compiles it without clusters.
fn cluster_function(module: &Arc<CudaModule>, name: &str) -> anyhow::Result<CudaFunction> {
    let f = module.load_function(name)?;
    if !cfg!(sm89) {
        f.set_attribute(
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
            1,
        )?;
    }
    Ok(f)
}

/// [`cluster_function`] with `smem` bytes of dynamic shared memory.
fn gate_rows_function(
    module: &Arc<CudaModule>,
    name: &str,
    smem: u32,
) -> anyhow::Result<CudaFunction> {
    let f = cluster_function(module, name)?;
    f.set_attribute(
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        i32::try_from(smem)?,
    )
    .with_context(|| format!("raising the shared memory of {name}"))?;
    Ok(f)
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
            argmax_partials: module.load_function("argmax_partials")?,
            message_gate_out_2304: module.load_function("message_gate_out_2304")?,
            message_gate_out_rows_2304: module.load_function("message_gate_out_rows_2304")?,
            message_gate_norm_rows_2304: module.load_function("message_gate_norm_rows_2304")?,
            message_gate_norm_quant_rows_2304: module
                .load_function("message_gate_norm_quant_rows_2304")?,
            argmax_wide_f32: module.load_function("argmax_wide_f32")?,
            message_gate_step_2304: cluster_function(&module, "message_gate_step_2304")?,
            message_gate_step_fp8_2304: cluster_function(&module, "message_gate_step_fp8_2304")?,
            message_gate_rows_2304: gate_rows_function(
                &module,
                "message_gate_rows_2304",
                GATE_ROWS_SMEM_BF16,
            )?,
            message_gate_rows_fp8_2304: gate_rows_function(
                &module,
                "message_gate_rows_fp8_2304",
                GATE_ROWS_SMEM_FP8,
            )?,
            decode_prologue_2304: module.load_function("decode_prologue_2304")?,
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
    /// `mx.argmax`. Vectorized and launched with PDL.
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
        if rows == 0 {
            return Ok(());
        }
        let n_arg = i32::try_from(n).context("n too large")?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: (ARGMAX_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&x, stream).ptr_mut(&mut out, stream).i32(n_arg);
        // SAFETY: the arguments match `argmax_f32(const float*, int*, int)`. `x` holds `rows`
        // rows of `n`, `out` holds `rows` elements, one per block, and the kernel reads
        // nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.argmax_f32, config) }
    }

    /// Second stage of the fused LM head greedy argmax: reduces `n` `(val, idx)` partials
    /// per row (row stride `stride`) into `out` (`[rows]`), launched with PDL.
    pub fn argmax_partials(
        &self,
        stream: &CudaStream,
        vals: &impl DeviceIn<f32>,
        idxs: &impl DeviceIn<i32>,
        out: &mut impl DeviceOut<i32>,
        n: usize,
        stride: usize,
    ) -> anyhow::Result<()> {
        let (vals, idxs, mut out) = (vals.view(), idxs.view(), out.view_mut());
        ensure!(n > 0, "n must be positive");
        ensure!(stride >= n, "stride {stride} < n {n}");
        let rows = out.len();
        ensure!(rows > 0, "no rows");
        ensure!(
            vals.len() >= rows * stride && idxs.len() >= rows * stride,
            "partials ({}, {}) need at least {} elements ({rows} x {stride})",
            vals.len(),
            idxs.len(),
            rows * stride
        );
        let n_arg = i32::try_from(n).context("n too large")?;
        let stride_arg = i32::try_from(stride).context("stride too large")?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: (ARGMAX_PARTIALS_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&vals, stream)
            .ptr(&idxs, stream)
            .ptr_mut(&mut out, stream)
            .i32(n_arg)
            .i32(stride_arg);
        // SAFETY: the arguments match `argmax_partials(const float*, const int*, int*, int,
        // int)`. `vals` and `idxs` hold at least `rows * stride` elements, `out` holds `rows`
        // elements, and the kernel reads nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.argmax_partials, config) }
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
            grid_dim: (GATE_CLUSTER, u32::try_from(n)?, 1),
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
        args.opt_mut(down, stream).f32(eps);
        let function = if fp8 {
            &self.message_gate_step_fp8_2304
        } else {
            &self.message_gate_step_2304
        };
        // SAFETY: the arguments match `message_gate_step_2304`'s signature, or with FP8
        // weights `message_gate_step_fp8_2304`'s, every buffer has the length checked above
        // for the `n` gates of the grid's y dimension, the grid is one cluster of 16 blocks
        // per gate as the kernel is compiled, a null `normed` makes a gate skip the norm, and
        // the kernel reads only weights before `pdl_wait`.
        unsafe { args.launch_pdl(stream, function, config) }
    }

    /// The start of a decode step in one launch: the embedding row of token `ids[0]` from
    /// `table` (`[vocab, 2304]`; a zero row for an id outside it), its RMS norm with
    /// `embed_norm` into `hidden`, and from `hidden` the first layer's attention inputs: its
    /// unweighted RMS norm into `normed` (F32) and its RMS norm with `input_norm` into
    /// `input`. Equals `Elementwise::embedding`, then `Norm::rms_norm`,
    /// `Norm::unweighted_rms_norm_bf16` and `Norm::rms_norm` again, bit for bit.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn decode_prologue(
        &self,
        stream: &CudaStream,
        ids: &CudaSlice<i32>,
        table: &CudaSlice<bf16>,
        embed_norm: &CudaSlice<bf16>,
        input_norm: &CudaSlice<bf16>,
        hidden: &mut CudaSlice<bf16>,
        normed: &mut impl DeviceOut<f32>,
        input: &mut CudaSlice<bf16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let mut normed = normed.view_mut();
        ensure!(!ids.is_empty(), "no token id");
        ensure!(
            table.len().is_multiple_of(HIDDEN) && !table.is_empty(),
            "table has {} elements, not whole rows of {HIDDEN}",
            table.len()
        );
        let vocab = i32::try_from(table.len() / HIDDEN).context("table has too many rows")?;
        for (name, len) in [
            ("embed_norm", embed_norm.len()),
            ("input_norm", input_norm.len()),
            ("hidden", hidden.len()),
            ("normed", normed.len()),
            ("input", input.len()),
        ] {
            check_len(name, len, HIDDEN)?;
        }
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: ((HIDDEN / 4) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(ids, stream)
            .ptr(table, stream)
            .i32(vocab)
            .ptr(embed_norm, stream)
            .ptr(input_norm, stream)
            .ptr_mut(hidden, stream)
            .ptr_mut(&mut normed, stream)
            .ptr_mut(input, stream)
            .f32(eps);
        // SAFETY: the arguments match `decode_prologue_2304(const int*, const bf16*, int, const
        // bf16*, const bf16*, bf16*, float*, bf16*, float)`, every buffer has the length
        // checked above, the kernel reads only rows of `table` below `vocab`, and it reads
        // only the norm weights before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.decode_prologue_2304, config) }
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

    /// The whole `MessageGate` for the `rows` rows (1 to [`GATE_ROWS_MAX`]) of a chunk and
    /// 1 to 3 gates in one launch: each gate's down projection of its state and message
    /// side by side (their rows `x_state_stride` and `x_message_stride` apart), its outputs with the
    /// message rebuilt from `mix` (`m` being the partial candidate, rows `m_stride` apart),
    /// and their norm. It sums as the multi-row down product of
    /// [`crate::ops::linear_rows::LinearRows`] followed by [`Misc::message_gate_out_rows`]
    /// with its norm, so the results are the same bit for bit, and each row's result
    /// depends on that row only.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn message_gate_rows(
        &self,
        stream: &CudaStream,
        gates: &mut [GateRowsSlot<'_>],
        [x_state_stride, x_message_stride]: [usize; 2],
        h_stride: usize,
        (m, m_stride): (&impl DeviceIn<f32>, usize),
        blocks: &CudaView<'_, f32>,
        probs_stride: usize,
        eps: f32,
        rows: usize,
    ) -> anyhow::Result<()> {
        let m = m.view();
        let n = gates.len();
        ensure!(
            (1..=GATE_SLOTS).contains(&n),
            "{n} gates; one launch takes 1 to {GATE_SLOTS}"
        );
        ensure!(
            (1..=GATE_ROWS_MAX).contains(&rows),
            "{rows} rows, expected 1 to {GATE_ROWS_MAX}"
        );
        let fp8 = gates[0].down_fp8.is_some();
        let block_len = rows * HIDDEN;
        ensure!(
            blocks.len().is_multiple_of(block_len),
            "blocks has {} elements, not a multiple of {rows} x {HIDDEN}",
            blocks.len()
        );
        let nb = blocks.len() / block_len;
        ensure!(
            nb <= GATE_MAX_BLOCKS,
            "{nb} block states; at most {GATE_MAX_BLOCKS} fit"
        );
        ensure!(
            m_stride >= HIDDEN
                && m_stride.is_multiple_of(2)
                && m.len() >= (rows - 1) * m_stride + HIDDEN,
            "m has {} elements, too few for {rows} even rows {m_stride} apart",
            m.len()
        );
        ensure!(
            h_stride >= HIDDEN && h_stride.is_multiple_of(2),
            "h_stride {h_stride} is odd or below {HIDDEN}"
        );
        ensure!(
            probs_stride > nb,
            "probs_stride {probs_stride} holds no {} weights",
            nb + 1
        );
        ensure!(
            [x_state_stride, x_message_stride]
                .iter()
                .all(|&s| s >= HIDDEN && s.is_multiple_of(8)),
            "x strides must be multiples of 8, at least {HIDDEN}"
        );
        let mut args = GateRowsArgs {
            m: m.device_ptr(stream).0,
            blocks: blocks.device_ptr(stream).0,
            x_state_stride: i32::try_from(x_state_stride)?,
            x_message_stride: i32::try_from(x_message_stride)?,
            h_stride: i32::try_from(h_stride)?,
            m_stride: i32::try_from(m_stride)?,
            block_stride: i32::try_from(block_len)?,
            nb: i32::try_from(nb)?,
            probs_stride: i32::try_from(probs_stride)?,
            rows: i32::try_from(rows)?,
            eps,
            ..GateRowsArgs::default()
        };
        let shape = GateRowsShape {
            rows,
            nb,
            probs_stride,
            x_strides: [x_state_stride, x_message_stride],
            h_stride,
        };
        for (k, gate) in gates.iter_mut().enumerate() {
            ensure!(
                gate.down_fp8.is_some() == fp8,
                "every gate of a launch has an FP8 down weight or none does"
            );
            gate.push(&mut args, k, &shape, stream)?;
        }
        let (function, smem) = if fp8 {
            (&self.message_gate_rows_fp8_2304, GATE_ROWS_SMEM_FP8)
        } else {
            (&self.message_gate_rows_2304, GATE_ROWS_SMEM_BF16)
        };
        let config = LaunchConfig {
            grid_dim: (
                GATE_CLUSTER,
                u32::try_from(n)?,
                u32::try_from(rows.div_ceil(GATE_CLUSTER_ROWS))?,
            ),
            block_dim: (GATE_OUT_THREADS, 1, 1),
            shared_mem_bytes: smem,
        };
        // SAFETY: `args` has the layout of `gate_rows::GateRowsArgs`. The checks above size every
        // buffer for `rows` rows of the `n` gates of the grid's y dimension, the message
        // weights for `nb` block states, and the grid is one cluster of 16 blocks per gate
        // and 8 rows as the kernel is compiled, with its dynamic shared memory. The kernel reads only
        // weights before `pdl_wait`.
        unsafe { launch_by_value(stream, function, config, &args, true) }
    }

    /// Launches `message_gate_out_rows_2304` (without the trailing RMSNorm).
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    fn launch_message_gate_out_rows(
        &self,
        stream: &CudaStream,
        down: &CudaView<'_, bf16>,
        forget: &CudaSlice<bf16>,
        gain: &CudaSlice<bf16>,
        (h, h_stride): (&CudaView<'_, f32>, usize),
        (m, m_stride): (&CudaView<'_, f32>, usize),
        mix: Option<&MessageMix<'_>>,
        out: &mut CudaViewMut<'_, f32>,
        rows: usize,
    ) -> anyhow::Result<()> {
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
        let block_len = rows * HIDDEN;
        if let Some(mix) = mix {
            ensure!(
                !mix.blocks.is_empty() && mix.blocks.len().is_multiple_of(block_len),
                "blocks has {} elements, not a positive multiple of {rows} x {HIDDEN}",
                mix.blocks.len()
            );
            let nb = mix.blocks.len() / block_len;
            ensure!(
                nb <= GATE_MAX_BLOCKS,
                "{nb} block states; at most {GATE_MAX_BLOCKS} fit"
            );
            ensure!(
                mix.probs_stride > nb && mix.probs.len() > (rows - 1) * mix.probs_stride + nb,
                "probs has {} elements, too few for {rows} rows of {} weights {} apart",
                mix.probs.len(),
                nb + 1,
                mix.probs_stride
            );
        }
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
        args.ptr(down, stream)
            .ptr(forget, stream)
            .ptr(gain, stream)
            .ptr(h, stream)
            .i32(i32::try_from(h_stride)?)
            .ptr(m, stream)
            .i32(i32::try_from(m_stride)?);
        match mix {
            Some(mix) => args
                .ptr(&mix.blocks, stream)
                .i32(i32::try_from(block_len)?)
                .i32(i32::try_from(mix.blocks.len() / block_len)?)
                .ptr(&mix.probs, stream)
                .i32(i32::try_from(mix.probs_stride)?),
            None => args.u64(0).i32(0).i32(0).u64(0).i32(0),
        };
        args.ptr_mut(out, stream)
            .i32(i32::try_from(rows).context("too many rows")?);
        // SAFETY: the arguments match `message_gate_out_rows_2304(const bf16*, const bf16*,
        // const bf16*, const float*, int, const float*, int, const float*, int, int, const
        // float*, int, float*, int)`, every buffer holds `rows` rows as checked above, a null
        // `probs` leaves `blocks` unread, the grid covers the rows in tiles of 64 (the kernel
        // skips rows past `rows`), and it reads only weights before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.message_gate_out_rows_2304, config) }
    }

    /// [`Misc::message_gate_out`] for each of the `rows` rows of a chunk: `down`
    /// `[rows, 64]`, `out` `[rows, 2304]`, and the rows of `h` and `m` `h_stride` and
    /// `m_stride` elements apart. With `norm` set to `(weight, normed)`, `normed` is
    /// `[rows, 2304]`. With `mix`, the message is rebuilt from its candidates, `m` being the
    /// partial one. The products with the down projection run on tensor cores, so results
    /// differ from a single-row call's by F32 rounding.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn message_gate_out_rows(
        &self,
        stream: &CudaStream,
        down: &impl DeviceIn<bf16>,
        forget: &CudaSlice<bf16>,
        gain: &CudaSlice<bf16>,
        (h, h_stride): (&impl DeviceIn<f32>, usize),
        (m, m_stride): (&impl DeviceIn<f32>, usize),
        mix: Option<&MessageMix<'_>>,
        out: &mut impl DeviceOut<f32>,
        norm: Option<GateNorm<'_>>,
        eps: f32,
        rows: usize,
    ) -> anyhow::Result<()> {
        let (down, h, m, mut out) = (down.view(), h.view(), m.view(), out.view_mut());
        self.launch_message_gate_out_rows(
            stream,
            &down,
            forget,
            gain,
            (&h, h_stride),
            (&m, m_stride),
            mix,
            &mut out,
            rows,
        )?;
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
        // row, and the kernel reads only the norm weight before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.message_gate_norm_rows_2304, config) }
    }

    /// [`Misc::message_gate_out_rows`] whose trailing RMSNorm fuses MXFP8 or NVFP4 activation
    /// quantization directly from registers into `norm.quant` (and optionally writes BF16
    /// `norm.normed`). Matches [`Self::message_gate_out_rows`] followed by the standalone
    /// quantizer bit for bit.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel signature")]
    pub fn message_gate_out_rows_quant(
        &self,
        stream: &CudaStream,
        down: &impl DeviceIn<bf16>,
        forget: &CudaSlice<bf16>,
        gain: &CudaSlice<bf16>,
        (h, h_stride): (&impl DeviceIn<f32>, usize),
        (m, m_stride): (&impl DeviceIn<f32>, usize),
        mix: Option<&MessageMix<'_>>,
        out: &mut impl DeviceOut<f32>,
        mut norm: GateNormQuant<'_>,
        eps: f32,
        rows: usize,
    ) -> anyhow::Result<()> {
        let (down, h, m, mut out) = (down.view(), h.view(), m.view(), out.view_mut());
        check_len("norm weight", norm.weight.len(), HIDDEN)?;
        if let Some(normed) = &norm.normed {
            check_len("normed", normed.len(), rows * HIDDEN)?;
        }
        norm.quant.check(rows, HIDDEN)?;
        self.launch_message_gate_out_rows(
            stream,
            &down,
            forget,
            gain,
            (&h, h_stride),
            (&m, m_stride),
            mix,
            &mut out,
            rows,
        )?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: (GATE_OUT_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&out, stream)
            .ptr(norm.weight, stream)
            .opt_mut(norm.normed.as_mut(), stream)
            .ptr_mut(&mut norm.quant.values, stream)
            .ptr_mut(&mut norm.quant.scales, stream)
            .opt_mut(norm.quant.outer.as_mut(), stream)
            .i32(norm.quant.format.mode())
            .f32(eps);
        // SAFETY: the arguments match `message_gate_norm_quant_rows_2304(const float*, const
        // bf16*, bf16*, u8*, u8*, float*, int, float)`, all buffers hold `rows` rows of 2304
        // as checked above, and the kernel reads only `norm_weight` before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.message_gate_norm_quant_rows_2304, config) }
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
