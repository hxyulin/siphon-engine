//! RMS normalization and L2 clipping (`kernels/norm.cu`).

use std::sync::Arc;

use anyhow::{Context as _, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig,
    PushKernelArg as _,
};
use half::bf16;

use crate::{
    gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module},
    ops::spec::{BATCH_STATE_WORDS, BatchRows},
};

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
/// Blocks per row of `attention_post_rows_bf16` that compute the relative bias,
/// `kBiasBlocks` in the kernel.
const BIAS_BLOCKS: u32 = 24;

/// The key and value convolutions of [`Norm::attention_post`].
pub enum KvConvWeights<'a> {
    /// `ShortConvolution` weights, BF16 `[1024, 4]`.
    Static {
        k: &'a CudaSlice<bf16>,
        v: &'a CudaSlice<bf16>,
    },
    /// `DynamicShortConvolution`: each row's BF16 taps `[rows, 8 heads * 4]` and the F32
    /// static weights `[4, 1024]`.
    Dynamic {
        k_dynamic: CudaView<'a, bf16>,
        v_dynamic: CudaView<'a, bf16>,
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

/// Inputs and outputs of [`Norm::attention_post_rows`] for the rows of a chunk: the
/// per-row buffers of [`AttentionPost`] with `rows` rows each.
pub struct AttentionPostRows<'a> {
    pub k_raw: CudaView<'a, bf16>,
    pub v_raw: CudaView<'a, bf16>,
    pub conv: KvConvWeights<'a>,
    /// Conv states, F32 `[3, 1024]`, only read.
    pub k_state: &'a CudaSlice<f32>,
    pub v_state: &'a CudaSlice<f32>,
    pub q_raw: CudaView<'a, bf16>,
    pub q_scale: &'a CudaSlice<bf16>,
    pub r_raw: CudaView<'a, bf16>,
    pub r_scale: &'a CudaSlice<bf16>,
    pub k: CudaViewMut<'a, bf16>,
    pub v: CudaViewMut<'a, bf16>,
    pub q: CudaViewMut<'a, bf16>,
    pub r: CudaViewMut<'a, bf16>,
    /// For full attention, the position of row 0 (one value), alpha and n_floor.
    pub log_scale: Option<(&'a CudaSlice<u32>, f32, f32)>,
    pub qk_threshold: f32,
    pub rel_threshold: f32,
    pub eps: f32,
    /// The layer's relative bias table `[16, extent]` and the bias `[rows, 24, extent]` to
    /// compute from the rows' relative states, as `Attention::relative_bias_rows` does.
    pub bias: Option<(&'a CudaSlice<bf16>, CudaViewMut<'a, bf16>)>,
    /// For a batched step, each row's conv states and position: every row is then the first
    /// of its own chunk, `k_state` and `v_state` go unread, and the log scale takes the
    /// row's position instead of `start + row`.
    pub batch: Option<&'a BatchRows<'a>>,
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
    attention_post_rows_bf16: CudaFunction,
}

/// The extent of a relative bias table `[16, extent]`, after checking that the bias holds
/// `rows` rows of 24 heads of it; 0 without one.
fn bias_extent(
    bias: Option<&(&CudaSlice<bf16>, CudaViewMut<'_, bf16>)>,
    rows: usize,
) -> anyhow::Result<usize> {
    let Some((table, bias)) = bias else {
        return Ok(0);
    };
    ensure!(
        table.len().is_multiple_of(4 * REL_DIM) && !table.is_empty(),
        "table has {} elements, not [16, a multiple of 4]",
        table.len()
    );
    let extent = table.len() / REL_DIM;
    ensure!(
        bias.len() == rows * Q_HEADS * extent,
        "bias has {} elements, expected {rows} x {Q_HEADS} x {extent}",
        bias.len()
    );
    Ok(extent)
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
            attention_post_rows_bf16: module.load_function("attention_post_rows_bf16")?,
        })
    }

    /// Weighted RMSNorm of each BF16 row of `x` (shape `[rows, 2304]`) into `out`.
    pub fn rms_norm(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        weight: &CudaSlice<bf16>,
        out: &mut impl DeviceOut<bf16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let (x, mut out) = (x.view(), out.view_mut());
        check_weight(weight.len())?;
        let rows = rows(x.len(), HIDDEN)?;
        check_out(out.len(), x.len())?;
        let mut launch = stream.launch_builder(&self.rms_norm_bf16_2304);
        launch.arg(&x).arg(weight).arg(&mut out).arg(&eps);
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
        x: &impl DeviceIn<bf16>,
        out: &mut impl DeviceOut<f32>,
        eps: f32,
    ) -> anyhow::Result<()> {
        let (x, mut out) = (x.view(), out.view_mut());
        let rows = rows(x.len(), HIDDEN)?;
        check_out(out.len(), x.len())?;
        let mut args = KernelArgs::default();
        args.ptr(&x, stream).ptr_mut(&mut out, stream).f32(eps);
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
                    .ptr(k_dynamic, stream)
                    .ptr(v_dynamic, stream)
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

    /// [`Norm::attention_post`] for each of the `rows` rows of a chunk, whose row 0 sits at
    /// the position in `log_scale`. Row `t`'s convs read rows `t - 3 ..= t` of `k_raw` and
    /// `v_raw`, reaching into the states before row 0. The states are only read: the caller
    /// shifts `k_raw` and `v_raw` into them afterwards with
    /// [`Conv::shift_state`](crate::ops::conv::Conv::shift_state). Each row's results equal
    /// a single-row call's bit for bit.
    #[expect(
        clippy::too_many_lines,
        reason = "checks and pushes every kernel argument"
    )]
    pub fn attention_post_rows(
        &self,
        stream: &CudaStream,
        a: AttentionPostRows<'_>,
        rows: usize,
    ) -> anyhow::Result<()> {
        let kv = KV_HEADS * HEAD_DIM;
        let check = |name: &str, len: usize, want: usize| {
            ensure!(len == want, "{name} has {len} elements, expected {want}");
            Ok(())
        };
        ensure!(rows > 0, "no rows");
        for (name, len) in [
            ("k_raw", a.k_raw.len()),
            ("v_raw", a.v_raw.len()),
            ("k", a.k.len()),
            ("v", a.v.len()),
        ] {
            check(name, len, rows * kv)?;
        }
        check("k_state", a.k_state.len(), (CONV_TAPS - 1) * kv)?;
        check("v_state", a.v_state.len(), (CONV_TAPS - 1) * kv)?;
        check("q_raw", a.q_raw.len(), rows * Q_HEADS * HEAD_DIM)?;
        check("q", a.q.len(), rows * Q_HEADS * HEAD_DIM)?;
        check("q_scale", a.q_scale.len(), Q_HEADS * HEAD_DIM)?;
        check("r_raw", a.r_raw.len(), rows * Q_HEADS * REL_DIM)?;
        check("r", a.r.len(), rows * Q_HEADS * REL_DIM)?;
        check("r_scale", a.r_scale.len(), Q_HEADS * REL_DIM)?;
        let extent = bias_extent(a.bias.as_ref(), rows)?;
        let config = LaunchConfig {
            grid_dim: (
                POST_BLOCKS + if a.bias.is_some() { BIAS_BLOCKS } else { 0 },
                u32::try_from(rows).context("too many rows")?,
                1,
            ),
            block_dim: (POST_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut args = KernelArgs::default();
        args.ptr(&a.k_raw, stream).ptr(&a.v_raw, stream);
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
                check("k_dynamic", k_dynamic.len(), rows * KV_HEADS * CONV_TAPS)?;
                check("v_dynamic", v_dynamic.len(), rows * KV_HEADS * CONV_TAPS)?;
                check("k_static", k_static.len(), CONV_TAPS * kv)?;
                check("v_static", v_static.len(), CONV_TAPS * kv)?;
                args.u64(0)
                    .u64(0)
                    .ptr(k_dynamic, stream)
                    .ptr(v_dynamic, stream)
                    .ptr(*k_static, stream)
                    .ptr(*v_static, stream);
            }
        }
        let (start, log_scale, alpha, n_floor) = match a.log_scale {
            Some((start, alpha, n_floor)) => {
                check("start", start.len(), 1)?;
                (Some(start), 1, alpha, n_floor)
            }
            None => (None, 0, 0.0, 1.0),
        };
        let (mut k_out, mut v_out, mut q_out, mut r_out) = (a.k, a.v, a.q, a.r);
        let (table, mut bias) = a.bias.unzip();
        args.ptr(a.k_state, stream)
            .ptr(a.v_state, stream)
            .ptr(&a.q_raw, stream)
            .ptr(a.q_scale, stream)
            .ptr(&a.r_raw, stream)
            .ptr(a.r_scale, stream)
            .ptr_mut(&mut k_out, stream)
            .ptr_mut(&mut v_out, stream)
            .ptr_mut(&mut q_out, stream)
            .ptr_mut(&mut r_out, stream)
            .opt(start, stream)
            .i32(log_scale)
            .f32(alpha)
            .f32(n_floor)
            .f32(a.qk_threshold)
            .f32(a.rel_threshold)
            .f32(a.eps)
            .opt(table, stream)
            .opt_mut(bias.as_mut(), stream)
            .i32(i32::try_from(extent)?);
        if let Some(batch) = a.batch {
            check("states", batch.states.len(), rows * BATCH_STATE_WORDS)?;
            check("positions", batch.positions.len(), rows)?;
            args.ptr(&batch.states, stream)
                .ptr(&batch.positions, stream);
        } else {
            args.u64(0).u64(0);
        }
        // SAFETY: the arguments match `attention_post_rows_bf16`'s signature, which is
        // `attention_post_bf16`'s with `start` for `position`, then the table, the bias, the
        // extent, and the batch's state table and positions (null outside a batch, and
        // then the kernel reads neither); in a batch, each row's words 2 and 3 name live
        // `[3, 1024]` F32 states. Every per-row buffer holds `rows` rows as checked above and the grid
        // has one y index per row, and bias blocks only with a table and a bias of `rows` x
        // 24 x extent; the kernel reads no null pointer (the static conv weights only
        // without dynamic taps, `start` only with log scaling), writes no state, and reads
        // nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.attention_post_rows_bf16, config) }
    }
}
