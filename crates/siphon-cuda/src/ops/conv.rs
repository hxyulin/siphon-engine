//! Causal short convolutions with a carried conv state (`kernels/conv.cu`).
//!
//! Both ops process the rows of one sequence and update `state` in place to the
//! last three rows of the conv window, so a decode step continues where the
//! previous call stopped. A fresh sequence starts from a zeroed state.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig,
    PushKernelArg as _,
};
use half::bf16;

use crate::{
    gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module},
    ops::spec::{
        BATCH_ROWS, BATCH_STATE_WORDS, BATCH_VERIFY_MAX_ROWS, PACKED_SLICE_WORDS, PackedRows,
        VERIFY_ROWS, VISIT_CHANNELS,
    },
};

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
/// Rows per block of `moe_conv_residual_rows_packed_2304`, and its threads and columns.
pub const MOE_CONV_ROWS: usize = 8;
/// Maximum 8-row tiles across `BATCH_ROWS` packed slices totaling at most `MAX_PREFILL_CHUNK`
/// rows (`2048 / 8 + 64 = 320`).
pub const MOE_PACKED_MAX_TILES: usize = 320;
const MOE_CONV_THREADS: u32 = 192;
const MOE_CONV_BLOCK_COLS: usize = 768;
/// Tile of `moe_conv_residual_rows_2304`: 29 output rows x 128 cols, 128 threads.
const MOE_CONV_WIDE_ROWS: usize = 29;
const MOE_CONV_WIDE_THREADS: u32 = 128;
const MOE_CONV_WIDE_BLOCK_COLS: usize = 128;
/// Max rows of `moe_conv_residual_step_rows_2304` and `moe_conv_residual_batch_verify_2304`.
const STEP_MAX_ROWS: usize = 8;
/// Channels per block of `moe_conv_residual_step_rows_2304`, `kStepBlockCols` in the
/// kernel; each row takes a warp of the block.
const STEP_BLOCK_COLS: usize = 32;
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
    short_conv_residual_rows_2304: CudaFunction,
    moe_conv_residual_rows_2304: CudaFunction,
    moe_conv_residual_step_rows_2304: CudaFunction,
    residual_batch_2304: CudaFunction,
    moe_conv_residual_batch_2304: CudaFunction,
    moe_conv_residual_batch_verify_2304: CudaFunction,
    shift_state_rows: CudaFunction,
    short_conv_residual_rows_packed_2304: CudaFunction,
    moe_conv_residual_rows_packed_2304: CudaFunction,
    shift_visit_states_packed_2304: CudaFunction,
}

/// The other conv streams of a visit, which [`Conv::moe_conv_residual_step_rows`] shifts
/// into their states.
pub struct VisitConvs<'a> {
    /// The attention output projection, the post-attention conv's input, `[rows, 2304]`.
    pub proj: CudaView<'a, bf16>,
    pub proj_state: &'a mut CudaSlice<f32>,
    /// The raw keys and values, the inputs of the key and value convs, `[rows, 1024]`.
    pub k_raw: CudaView<'a, bf16>,
    pub v_raw: CudaView<'a, bf16>,
    pub k_state: &'a mut CudaSlice<f32>,
    pub v_state: &'a mut CudaSlice<f32>,
    /// Where a verify records the visit's conv inputs, BF16 `[8, 6656]`: per row the raw
    /// keys and values, the projection and the MoE combine.
    pub inputs: Option<CudaViewMut<'a, bf16>>,
}

/// One block of [`BLOCK`] channels per (channel block, row) of a `[rows, 2304]` chunk.
fn hidden_rows_config(rows: usize) -> anyhow::Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (
            (HIDDEN as u32).div_ceil(BLOCK),
            u32::try_from(rows).context("too many rows")?,
            1,
        ),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn check_len(name: &str, len: usize, want: usize) -> anyhow::Result<()> {
    ensure!(len == want, "{name} has {len} elements, expected {want}");
    Ok(())
}

impl Conv {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "conv", cubin!("conv"))?;
        Ok(Self {
            short_conv_bf16: module.load_function("short_conv_bf16")?,
            dynamic_short_conv_bf16: module.load_function("dynamic_short_conv_bf16")?,
            short_conv_residual_2304: module.load_function("short_conv_residual_2304")?,
            moe_conv_residual_2304: module.load_function("moe_conv_residual_2304")?,
            short_conv_residual_rows_2304: module.load_function("short_conv_residual_rows_2304")?,
            moe_conv_residual_rows_2304: module.load_function("moe_conv_residual_rows_2304")?,
            moe_conv_residual_step_rows_2304: module
                .load_function("moe_conv_residual_step_rows_2304")?,
            residual_batch_2304: module.load_function("conv_residual_batch_2304")?,
            moe_conv_residual_batch_2304: module.load_function("moe_conv_residual_batch_2304")?,
            moe_conv_residual_batch_verify_2304: module
                .load_function("moe_conv_residual_batch_verify_2304")?,
            shift_state_rows: module.load_function("shift_state_rows")?,
            short_conv_residual_rows_packed_2304: module
                .load_function("short_conv_residual_rows_packed_2304")?,
            moe_conv_residual_rows_packed_2304: module
                .load_function("moe_conv_residual_rows_packed_2304")?,
            shift_visit_states_packed_2304: module
                .load_function("shift_visit_states_packed_2304")?,
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

impl Conv {
    /// [`Conv::short_conv_residual`] for the `rows` rows of a chunk, `[rows, 2304]` each,
    /// in order: row `t` convolves rows `t - 3 ..= t` of `x`, reaching into `state` before
    /// row 0. Afterwards `state` holds the chunk's last three inputs, as after `rows`
    /// single-row calls, whose results this gives bit for bit.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn short_conv_residual_rows(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        weight: &CudaSlice<bf16>,
        state: &mut CudaSlice<f32>,
        hidden: Option<&dyn DeviceIn<bf16>>,
        residual: &mut impl DeviceOut<f32>,
        rows: usize,
    ) -> anyhow::Result<()> {
        let (x, mut residual) = (x.view(), residual.view_mut());
        let hidden = hidden.map(|h| h.view());
        check_hidden_conv(weight, state)?;
        ensure!(rows > 0, "no rows");
        check_len("x", x.len(), rows * HIDDEN)?;
        check_len("residual", residual.len(), rows * HIDDEN)?;
        if let Some(h) = &hidden {
            check_len("hidden", h.len(), rows * HIDDEN)?;
        }
        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .ptr(weight, stream)
            .ptr(&*state, stream)
            .opt(hidden.as_ref(), stream)
            .ptr_mut(&mut residual, stream);
        // SAFETY: the arguments match `short_conv_residual_rows_2304(const bf16*, const bf16*,
        // const float*, const bf16*, float*)`, every row buffer holds `rows` rows of 2304 as
        // checked above, the grid has one y index per row, and a null `hidden` is never read.
        unsafe {
            args.launch_pdl(
                stream,
                &self.short_conv_residual_rows_2304,
                hidden_rows_config(rows)?,
            )
        }?;
        self.shift_state(stream, &x, state, HIDDEN, rows)
    }

    /// [`Conv::moe_conv_residual`] for the `rows` rows of a chunk: `experts`
    /// `[rows, 8, 2304]`, `weights` `[rows, 8]`, `residual` and `out` `[rows, 2304]`.
    /// `combined` `[rows, 2304]` receives the MoE combine, the conv's input, of the last
    /// three rows, which the state shift reads; its other rows are left alone. Updates
    /// `state` as [`Conv::short_conv_residual_rows`] does, and gives the single-row results
    /// bit for bit.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn moe_conv_residual_rows(
        &self,
        stream: &CudaStream,
        experts: &impl DeviceIn<bf16>,
        weights: &impl DeviceIn<f32>,
        weight: &CudaSlice<bf16>,
        state: &mut CudaSlice<f32>,
        residual: &impl DeviceIn<f32>,
        combined: &mut impl DeviceOut<bf16>,
        out: &mut impl DeviceOut<bf16>,
        rows: usize,
    ) -> anyhow::Result<()> {
        let (experts, weights, residual) = (experts.view(), weights.view(), residual.view());
        let (mut combined, mut out) = (combined.view_mut(), out.view_mut());
        check_hidden_conv(weight, state)?;
        ensure!(rows > 0, "no rows");
        check_len("experts", experts.len(), rows * TOP_K * HIDDEN)?;
        check_len("weights", weights.len(), rows * TOP_K)?;
        for (name, len) in [
            ("residual", residual.len()),
            ("combined", combined.len()),
            ("out", out.len()),
        ] {
            check_len(name, len, rows * HIDDEN)?;
        }
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(HIDDEN / MOE_CONV_WIDE_BLOCK_COLS)?,
                u32::try_from(rows.div_ceil(MOE_CONV_WIDE_ROWS)).context("too many rows")?,
                1,
            ),
            block_dim: (MOE_CONV_WIDE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&experts, stream)
            .ptr(&weights, stream)
            .ptr(weight, stream)
            .ptr(&*state, stream)
            .ptr(&residual, stream)
            .ptr_mut(&mut combined, stream)
            .ptr_mut(&mut out, stream)
            .i32(i32::try_from(rows).context("too many rows")?);
        // SAFETY: the arguments match `moe_conv_residual_rows_2304(const bf16*, const float*,
        // const bf16*, const float*, const float*, bf16*, bf16*, int)`, the buffers hold
        // `rows` rows as checked above, and the grid covers the row in 18 blocks of 128
        // columns and the rows in tiles of 29 (the kernel stops at `rows`).
        unsafe { args.launch_pdl(stream, &self.moe_conv_residual_rows_2304, config) }?;
        self.shift_state(stream, &combined.slice(..), state, HIDDEN, rows)
    }

    /// [`Conv::moe_conv_residual_rows`] for the 1 to 8 rows of a step. With `convs.inputs`
    /// set to `None` (a small prefill chunk), it also shifts the visit's other conv inputs in
    /// `convs` into their states, as [`Conv::shift_state`] would after the kernels that read
    /// them. With `convs.inputs` set to `Some(...)` (verify mode), it records every stream's
    /// inputs into `convs.inputs` and leaves all four conv states untouched so
    /// [`crate::ops::spec::Spec::conv_rollback`] can shift them in place. One launch instead
    /// of six.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn moe_conv_residual_step_rows(
        &self,
        stream: &CudaStream,
        experts: &impl DeviceIn<bf16>,
        weights: &impl DeviceIn<f32>,
        weight: &CudaSlice<bf16>,
        state: &mut CudaSlice<f32>,
        residual: &impl DeviceIn<f32>,
        out: &mut impl DeviceOut<bf16>,
        convs: VisitConvs<'_>,
        rows: usize,
    ) -> anyhow::Result<()> {
        let (experts, weights, residual) = (experts.view(), weights.view(), residual.view());
        let mut out = out.view_mut();
        check_hidden_conv(weight, state)?;
        ensure!(
            (1..=STEP_MAX_ROWS).contains(&rows),
            "{rows} rows, expected 1 to {STEP_MAX_ROWS}"
        );
        check_len("experts", experts.len(), rows * TOP_K * HIDDEN)?;
        check_len("weights", weights.len(), rows * TOP_K)?;
        check_len("residual", residual.len(), rows * HIDDEN)?;
        check_len("out", out.len(), rows * HIDDEN)?;
        check_len("proj", convs.proj.len(), rows * HIDDEN)?;
        check_len("proj state", convs.proj_state.len(), STATE_ROWS * HIDDEN)?;
        for (name, len) in [("k_raw", convs.k_raw.len()), ("v_raw", convs.v_raw.len())] {
            check_len(name, len, rows * DYNAMIC_DIMS)?;
        }
        for (name, len) in [
            ("k state", convs.k_state.len()),
            ("v state", convs.v_state.len()),
        ] {
            check_len(name, len, STATE_ROWS * DYNAMIC_DIMS)?;
        }
        if let Some(inputs) = &convs.inputs {
            check_len(
                "inputs",
                inputs.len(),
                STEP_MAX_ROWS * (2 * DYNAMIC_DIMS + 2 * HIDDEN),
            )?;
        }
        let config = LaunchConfig {
            grid_dim: (u32::try_from(HIDDEN / STEP_BLOCK_COLS)?, 1, 1),
            block_dim: (u32::try_from(32 * rows)?, 1, 1),
            shared_mem_bytes: 0,
        };
        let VisitConvs {
            proj,
            proj_state,
            k_raw,
            v_raw,
            k_state,
            v_state,
            mut inputs,
        } = convs;
        let mut args = KernelArgs::default();
        args.ptr(&experts, stream)
            .ptr(&weights, stream)
            .ptr(weight, stream)
            .ptr_mut(state, stream)
            .ptr(&residual, stream)
            .ptr_mut(&mut out, stream)
            .i32(i32::try_from(rows)?)
            .ptr(&proj, stream)
            .ptr_mut(proj_state, stream)
            .ptr(&k_raw, stream)
            .ptr(&v_raw, stream)
            .ptr_mut(k_state, stream)
            .ptr_mut(v_state, stream)
            .opt_mut(inputs.as_mut(), stream);
        // SAFETY: the arguments match `moe_conv_residual_step_rows_2304(const bf16*, const
        // float*, const bf16*, float*, const float*, bf16*, int, const bf16*, float*, const
        // bf16*, const bf16*, float*, float*, bf16*)`, every buffer has the length checked
        // above for `rows` rows, and the grid gives each thread one channel of the 2304 (and
        // of the 1024 key and value channels below 1024) in one row, with every row of a
        // channel in the same block.
        unsafe { args.launch_pdl(stream, &self.moe_conv_residual_step_rows_2304, config) }
    }

    /// The post-attention short conv and residual add of a batched step: row `r` of `proj`
    /// (`[rows, 2304]`, the attention output projection) convolves with its own
    /// post-attention conv state, which `states` names, adds row `r` of `hidden` if given,
    /// and goes to `residual` (`[rows, 2304]`, F32). The state is only read. Each row
    /// rounds as `Routing::depth_attention_rows` computes its `ConvResidual` partial for a
    /// chunk of that one row.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn conv_residual_batch(
        &self,
        stream: &CudaStream,
        proj: &impl DeviceIn<bf16>,
        weight: &CudaSlice<bf16>,
        states: &CudaView<'_, u64>,
        hidden: Option<&CudaView<'_, bf16>>,
        residual: &mut impl DeviceOut<f32>,
        rows: usize,
        rows_per_seq: usize,
    ) -> anyhow::Result<()> {
        let proj = proj.view();
        let mut residual = residual.view_mut();
        ensure!(
            (1..=BATCH_VERIFY_MAX_ROWS).contains(&rows),
            "{rows} rows, expected 1 to {BATCH_VERIFY_MAX_ROWS}"
        );
        ensure!(
            (1..=VERIFY_ROWS).contains(&rows_per_seq) && rows.is_multiple_of(rows_per_seq),
            "{rows} rows is not a multiple of rows_per_seq {rows_per_seq}"
        );
        let seqs = rows / rows_per_seq;
        check_len("weight", weight.len(), HIDDEN * KERNEL)?;
        check_len("proj", proj.len(), rows * HIDDEN)?;
        check_len("residual", residual.len(), rows * HIDDEN)?;
        check_len("states", states.len(), seqs * BATCH_STATE_WORDS)?;
        if let Some(hidden) = hidden {
            check_len("hidden", hidden.len(), rows * HIDDEN)?;
        }
        let mut args = KernelArgs::default();
        args.ptr(&proj, stream)
            .ptr(weight, stream)
            .ptr(states, stream)
            .opt(hidden, stream)
            .ptr_mut(&mut residual, stream)
            .i32(i32::try_from(rows_per_seq)?);
        // SAFETY: the arguments match `conv_residual_batch_2304(const bf16*, const bf16*,
        // const unsigned long long*, const bf16*, float*, int)`. Every buffer holds `rows` rows
        // across `seqs` sequences as checked above, and each sequence's post-attention state
        // (word 4) is a live `[3, 2304]` F32 buffer.
        unsafe { args.launch_pdl(stream, &self.residual_batch_2304, hidden_rows_config(rows)?) }
    }

    /// [`Conv::moe_conv_residual_step_rows`] for the rows of a batched step, each with its
    /// own conv states, which `states` names: each row combines its experts, convolves
    /// its own post-MLP state and the combine, adds its residual into `out`, and shifts its
    /// `k_raw`, `v_raw`, `proj` and combine rows into its own four states. Each row rounds
    /// as a step of that one row.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn moe_conv_residual_batch(
        &self,
        stream: &CudaStream,
        experts: &impl DeviceIn<bf16>,
        weights: &impl DeviceIn<f32>,
        weight: &CudaSlice<bf16>,
        residual: &impl DeviceIn<f32>,
        out: &mut impl DeviceOut<bf16>,
        [proj, k_raw, v_raw]: [CudaView<'_, bf16>; 3],
        states: &CudaView<'_, u64>,
        rows: usize,
    ) -> anyhow::Result<()> {
        let (experts, weights, residual) = (experts.view(), weights.view(), residual.view());
        let mut out = out.view_mut();
        ensure!(
            (1..=BATCH_ROWS).contains(&rows),
            "{rows} rows, expected 1 to {BATCH_ROWS}"
        );
        check_len("weight", weight.len(), HIDDEN * KERNEL)?;
        check_len("experts", experts.len(), rows * TOP_K * HIDDEN)?;
        check_len("weights", weights.len(), rows * TOP_K)?;
        check_len("residual", residual.len(), rows * HIDDEN)?;
        check_len("out", out.len(), rows * HIDDEN)?;
        check_len("proj", proj.len(), rows * HIDDEN)?;
        check_len("k_raw", k_raw.len(), rows * DYNAMIC_DIMS)?;
        check_len("v_raw", v_raw.len(), rows * DYNAMIC_DIMS)?;
        check_len("states", states.len(), rows * BATCH_STATE_WORDS)?;
        let mut args = KernelArgs::default();
        args.ptr(&experts, stream)
            .ptr(&weights, stream)
            .ptr(weight, stream)
            .ptr(&residual, stream)
            .ptr_mut(&mut out, stream)
            .ptr(&proj, stream)
            .ptr(&k_raw, stream)
            .ptr(&v_raw, stream)
            .ptr(states, stream);
        // SAFETY: the arguments match `moe_conv_residual_batch_2304(const bf16*, const float*,
        // const bf16*, const float*, bf16*, const bf16*, const bf16*, const bf16*, const
        // unsigned long long*)`. Every buffer holds `rows` rows as checked above, the grid
        // has one y index per row and 9 blocks of 256 channels, and the caller keeps each
        // row's four conv states, words 2 to 5 of its state table, live buffers of
        // `[3, 1024]`, `[3, 1024]`, `[3, 2304]` and `[3, 2304]` F32, distinct between rows.
        // Before `pdl_wait` the kernel reads only the weight.
        unsafe {
            args.launch_pdl(
                stream,
                &self.moe_conv_residual_batch_2304,
                hidden_rows_config(rows)?,
            )
        }
    }

    /// Batched multi-row form of [`Self::moe_conv_residual_step_rows`]: `seqs` sequences
    /// each running `rows_per_seq` (`1..=8`) verify rows. With `inputs` set to `Some(...)`
    /// (verify mode), it records each sequence's conv inputs into
    /// `inputs` (`[max_seqs, 8, VISIT_CHANNELS]`) and leaves all four conv states untouched
    /// for [`crate::ops::spec::Spec::conv_rollback_batch`]; with `inputs` set to `None`, it
    /// shifts all four conv states of every sequence.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn moe_conv_residual_batch_verify(
        &self,
        stream: &CudaStream,
        experts: &impl DeviceIn<bf16>,
        weights: &impl DeviceIn<f32>,
        weight: &CudaSlice<bf16>,
        residual: &impl DeviceIn<f32>,
        out: &mut impl DeviceOut<bf16>,
        [proj, k_raw, v_raw]: [CudaView<'_, bf16>; 3],
        states: &CudaView<'_, u64>,
        mut inputs: Option<CudaViewMut<'_, bf16>>,
        seqs: usize,
        rows_per_seq: usize,
    ) -> anyhow::Result<()> {
        let (experts, weights, residual) = (experts.view(), weights.view(), residual.view());
        let mut out = out.view_mut();
        ensure!(
            seqs > 0
                && (1..=STEP_MAX_ROWS).contains(&rows_per_seq)
                && seqs * rows_per_seq <= BATCH_VERIFY_MAX_ROWS,
            "{seqs} seqs x {rows_per_seq} rows exceeds {BATCH_VERIFY_MAX_ROWS}"
        );
        let rows = seqs * rows_per_seq;
        check_len("weight", weight.len(), HIDDEN * KERNEL)?;
        check_len("experts", experts.len(), rows * TOP_K * HIDDEN)?;
        check_len("weights", weights.len(), rows * TOP_K)?;
        check_len("residual", residual.len(), rows * HIDDEN)?;
        check_len("out", out.len(), rows * HIDDEN)?;
        check_len("proj", proj.len(), rows * HIDDEN)?;
        check_len("k_raw", k_raw.len(), rows * DYNAMIC_DIMS)?;
        check_len("v_raw", v_raw.len(), rows * DYNAMIC_DIMS)?;
        check_len("states", states.len(), seqs * BATCH_STATE_WORDS)?;
        if let Some(inputs) = &inputs {
            ensure!(
                inputs.len() >= seqs * VERIFY_ROWS * VISIT_CHANNELS,
                "inputs has {} elements, need at least {}",
                inputs.len(),
                seqs * VERIFY_ROWS * VISIT_CHANNELS
            );
        }
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(HIDDEN / STEP_BLOCK_COLS)?,
                u32::try_from(seqs)?,
                1,
            ),
            block_dim: (u32::try_from(32 * rows_per_seq)?, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&experts, stream)
            .ptr(&weights, stream)
            .ptr(weight, stream)
            .ptr(&residual, stream)
            .ptr_mut(&mut out, stream)
            .i32(i32::try_from(rows_per_seq)?)
            .ptr(&proj, stream)
            .ptr(&k_raw, stream)
            .ptr(&v_raw, stream)
            .ptr(states, stream)
            .opt_mut(inputs.as_mut(), stream);
        // SAFETY: matches `moe_conv_residual_batch_verify_2304`; every per-row buffer holds
        // `seqs * rows_per_seq` rows, every sequence's four conv states in `states` are live
        // and distinct, and `inputs` (if present) holds at least `seqs * 8 * VISIT_CHANNELS`.
        unsafe { args.launch_pdl(stream, &self.moe_conv_residual_batch_verify_2304, config) }
    }

    /// Shifts the conv inputs `x` (`[rows, dims]`) of a chunk into `state` (`[3, dims]`),
    /// as `rows` single-row convs would leave it. The multi-row kernels only read the
    /// state, so this runs after them.
    pub fn shift_state(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        state: &mut CudaSlice<f32>,
        dims: usize,
        rows: usize,
    ) -> anyhow::Result<()> {
        let x = x.view();
        ensure!(dims > 0 && rows > 0, "no channels or no rows");
        check_len("x", x.len(), rows * dims)?;
        check_len("state", state.len(), STATE_ROWS * dims)?;
        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .ptr_mut(state, stream)
            .i32(i32::try_from(dims).context("too many channels")?)
            .i32(i32::try_from(rows).context("too many rows")?);
        // SAFETY: the arguments match `shift_state_rows(const bf16*, float*, int, int)`, `x`
        // holds `rows` rows of `dims` and `state` three, and the kernel skips threads past
        // `dims`.
        unsafe { args.launch_pdl(stream, &self.shift_state_rows, channel_config(dims)?) }
    }

    /// Ragged packed multi-slice form of [`Conv::short_conv_residual_rows`]: runs across all
    /// `rows` packed rows of `packed.num_slices` slices in one launch. Each row reads its
    /// slice's post-attention conv state (word 4 of `packed.states`) without modifying it;
    /// [`Conv::shift_visit_states_packed`] updates all conv states at the end of the visit.
    /// Every output equals per-slice [`Conv::short_conv_residual_rows`] bit for bit.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn short_conv_residual_rows_packed(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        weight: &CudaSlice<bf16>,
        packed: &PackedRows<'_>,
        hidden: Option<&CudaView<'_, bf16>>,
        residual: &mut impl DeviceOut<f32>,
        rows: usize,
    ) -> anyhow::Result<()> {
        let (x, mut residual) = (x.view(), residual.view_mut());
        ensure!(rows > 0, "no rows");
        ensure!(packed.num_slices > 0, "no packed slices");
        check_len("weight", weight.len(), HIDDEN * KERNEL)?;
        check_len("x", x.len(), rows * HIDDEN)?;
        check_len("residual", residual.len(), rows * HIDDEN)?;
        check_len("slices", packed.slices.len(), packed.num_slices * PACKED_SLICE_WORDS)?;
        ensure!(
            !packed.states.is_empty() && packed.states.len().is_multiple_of(BATCH_STATE_WORDS),
            "states has {} elements, not a positive multiple of {BATCH_STATE_WORDS}",
            packed.states.len()
        );
        if let Some(h) = hidden {
            check_len("hidden", h.len(), rows * HIDDEN)?;
        }
        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .ptr(weight, stream)
            .ptr(&packed.states, stream)
            .ptr(&packed.slices, stream)
            .i32(i32::try_from(packed.num_slices)?)
            .opt(hidden, stream)
            .ptr_mut(&mut residual, stream);
        // SAFETY: the arguments match `short_conv_residual_rows_packed_2304(const bf16*,
        // const bf16*, const unsigned long long*, const PackedSlice*, int, const bf16*,
        // float*)`. Every buffer holds `rows` rows as checked above, `slices` covers `0..rows`,
        // each slice's word 4 in `states` is a live `[3, 2304]` F32 conv state, and before
        // `pdl_wait` the kernel reads only `weight`.
        unsafe {
            args.launch_pdl(
                stream,
                &self.short_conv_residual_rows_packed_2304,
                hidden_rows_config(rows)?,
            )
        }
    }

    /// Ragged packed multi-slice form of [`Conv::moe_conv_residual_rows`]: runs across all
    /// `tiles.len()` 8-row tiles covering `packed.num_slices` slices in one launch without
    /// crossing slice boundaries. Reads each slice's post-MLP conv state (word 5 of
    /// `packed.states`) and writes the combines of each slice's last `min(len, 3)` rows into
    /// `combined` for [`Conv::shift_visit_states_packed`]. Every output equals per-slice
    /// [`Conv::moe_conv_residual_rows`] bit for bit.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn moe_conv_residual_rows_packed(
        &self,
        stream: &CudaStream,
        experts: &impl DeviceIn<bf16>,
        weights: &impl DeviceIn<f32>,
        weight: &CudaSlice<bf16>,
        packed: &PackedRows<'_>,
        tiles: &CudaView<'_, u32>,
        residual: &impl DeviceIn<f32>,
        combined: &mut impl DeviceOut<bf16>,
        out: &mut impl DeviceOut<bf16>,
        rows: usize,
    ) -> anyhow::Result<()> {
        let (experts, weights, residual) = (experts.view(), weights.view(), residual.view());
        let (mut combined, mut out) = (combined.view_mut(), out.view_mut());
        ensure!(rows > 0, "no rows");
        ensure!(packed.num_slices > 0 && !tiles.is_empty(), "no packed slices or tiles");
        check_len("weight", weight.len(), HIDDEN * KERNEL)?;
        check_len("experts", experts.len(), rows * TOP_K * HIDDEN)?;
        check_len("weights", weights.len(), rows * TOP_K)?;
        check_len("slices", packed.slices.len(), packed.num_slices * PACKED_SLICE_WORDS)?;
        ensure!(
            !packed.states.is_empty() && packed.states.len().is_multiple_of(BATCH_STATE_WORDS),
            "states has {} elements, not a positive multiple of {BATCH_STATE_WORDS}",
            packed.states.len()
        );
        for (name, len) in [
            ("residual", residual.len()),
            ("combined", combined.len()),
            ("out", out.len()),
        ] {
            check_len(name, len, rows * HIDDEN)?;
        }
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(HIDDEN / MOE_CONV_BLOCK_COLS)?,
                u32::try_from(tiles.len()).context("too many tiles")?,
                1,
            ),
            block_dim: (MOE_CONV_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&experts, stream)
            .ptr(&weights, stream)
            .ptr(weight, stream)
            .ptr(&packed.states, stream)
            .ptr(&packed.slices, stream)
            .ptr(tiles, stream)
            .ptr(&residual, stream)
            .ptr_mut(&mut combined, stream)
            .ptr_mut(&mut out, stream);
        // SAFETY: the arguments match `moe_conv_residual_rows_packed_2304(const bf16*,
        // const float*, const bf16*, const unsigned long long*, const PackedSlice*,
        // const uint32_t*, const float*, bf16*, bf16*)`. Every buffer holds `rows` rows as
        // checked above, each `tiles` entry names a valid `(local_tile, slice_idx)` pair,
        // each slice's word 5 in `states` is a live `[3, 2304]` F32 conv state, and before
        // `pdl_wait` the kernel reads only `weight`.
        unsafe {
            args.launch_pdl(stream, &self.moe_conv_residual_rows_packed_2304, config)
        }
    }

    /// Shifts up to all four conv streams (`k_raw`, `v_raw`, `proj`, `combined`) of a visit
    /// into every packed slice's conv states (words 2..5 of `packed.states`) in one launch.
    /// Any `None` stream is skipped. Exact bit-for-bit match with per-slice
    /// [`Conv::shift_state`].
    pub fn shift_visit_states_packed(
        &self,
        stream: &CudaStream,
        [k_raw, v_raw, proj, combined]: [Option<CudaView<'_, bf16>>; 4],
        packed: &PackedRows<'_>,
        rows: usize,
    ) -> anyhow::Result<()> {
        ensure!(rows > 0, "no rows");
        ensure!(packed.num_slices > 0, "no packed slices");
        check_len("slices", packed.slices.len(), packed.num_slices * PACKED_SLICE_WORDS)?;
        ensure!(
            !packed.states.is_empty() && packed.states.len().is_multiple_of(BATCH_STATE_WORDS),
            "states has {} elements, not a positive multiple of {BATCH_STATE_WORDS}",
            packed.states.len()
        );
        if let Some(x) = &k_raw {
            check_len("k_raw", x.len(), rows * DYNAMIC_DIMS)?;
        }
        if let Some(x) = &v_raw {
            check_len("v_raw", x.len(), rows * DYNAMIC_DIMS)?;
        }
        if let Some(x) = &proj {
            check_len("proj", x.len(), rows * HIDDEN)?;
        }
        if let Some(x) = &combined {
            check_len("combined", x.len(), rows * HIDDEN)?;
        }
        let config = LaunchConfig {
            grid_dim: (
                (HIDDEN as u32).div_ceil(BLOCK),
                u32::try_from(packed.num_slices)?,
                1,
            ),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.opt(k_raw.as_ref(), stream)
            .opt(v_raw.as_ref(), stream)
            .opt(proj.as_ref(), stream)
            .opt(combined.as_ref(), stream)
            .ptr(&packed.states, stream)
            .ptr(&packed.slices, stream);
        // SAFETY: the arguments match `shift_visit_states_packed_2304(const bf16*, const bf16*,
        // const bf16*, const bf16*, const unsigned long long*, const PackedSlice*)`. Each
        // non-null stream holds `rows` rows as checked above, every slice's conv states in
        // `states` (words 2..5) are live and distinct across slices, and the kernel reads
        // nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.shift_visit_states_packed_2304, config) }
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
