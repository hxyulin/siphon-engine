//! The expert MLP of 1 to 256 rows (`kernels/experts_rows.cu`).
//!
//! [`ExpertsRows::expert_mlp_rows`] runs, for every (row, slot) pair, the routed expert's
//! `w13`, the softcapped SwiGLU and `w2`, as [`crate::ops::experts::Experts::expert_mlp`]
//! does for one row. Up to 32 rows it groups the pairs in shared memory in one kernel;
//! above 32 rows a 1-block pre-group kernel writes the grouped pairs into the tail of
//! `sync` and the resident MLP kernel reads them from L2. Each distinct expert's weights
//! stream from DRAM once for all its rows. Nothing depends on the host knowing the
//! routing, so the call can be captured in a CUDA graph.
//!
//! Each pair's results are the same, bit for bit, for every row count and whatever the
//! other rows route to. They differ from `expert_mlp`'s by F32 summation order.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, LaunchConfig,
    sys::{CUdevice_attribute, CUfunction_attribute},
};
use half::bf16;

use crate::{
    DeviceIn, DeviceOut,
    gpu::{KernelArgs, cubin, load_module},
    ops::experts::{ExpertFormat, ExpertWeights},
};

/// Most rows one call takes.
pub const MAX_ROWS: usize = 256;
/// Most rows the single-kernel shared-memory path takes (`kMaxRows` in the kernel).
const INLINE_MAX_ROWS: usize = 32;
/// `u32` words of `ExpertRowsMeta`'s fixed header (`offsetof(ExpertRowsMeta, pair_list) / 4`).
const META_HEADER_U32: usize = 904 / 4;
/// Experts each row routes to.
const SLOTS: usize = 8;
/// Routed experts.
const EXPERTS: usize = 128;
/// `w13` is `[experts, 2 * K2, K13]` and `w2` `[experts, N2, K2]`.
const K13: usize = 2304;
const K2: usize = 1152;
const N2: usize = 2304;
/// Warps per block, `kWarps` in the kernel.
const WARPS: usize = 4;
/// Ring slots per warp of the two kernels of each format: the deeper ring for up to
/// [`DEEP_RING_ROWS`] rows. On an RTX 5090, one step's expert MLPs with real routing took
/// 1517 instead of 1663 us at 1 row and 2182 instead of 2228 at 2, but 3198 instead of
/// 3139 at 4, where the 2 blocks per SM of the deeper ring held too few warps.
const RINGS: [usize; 2] = [3, 4];
const DEEP_RING_ROWS: usize = 3;
/// Dynamic shared memory per block: `kWarps` warps of `ring` slots, each 16 rows of 128
/// weight bytes and 16 bytes of block scales.
const fn smem(ring: usize) -> usize {
    WARPS * ring * 16 * (128 + 16)
}

#[derive(Debug)]
pub struct ExpertsRows {
    /// By ring ([`RINGS`]), then NVFP4 and MXFP8 (for `m <= 32`).
    kernels: [[CudaFunction; 2]; 2],
    /// Blocks of each inline kernel that fit on the GPU at once.
    blocks: [[usize; 2]; 2],
    /// 1-block grouping kernel for `m > 32`.
    group: CudaFunction,
    /// Resident wide MLP kernels (`m > 32`), NVFP4 and MXFP8.
    wide_kernels: [CudaFunction; 2],
    /// Blocks of each wide kernel that fit on the GPU at once.
    wide_blocks: [usize; 2],
}

impl ExpertsRows {
    /// Minimum `sync` length in `u32` words for calls of up to `max_rows` rows.
    #[must_use]
    pub const fn sync_len(max_rows: usize) -> usize {
        let pairs = max_rows * SLOTS;
        let groups = if pairs < EXPERTS { pairs } else { EXPERTS } + max_rows;
        let counters = 1 + groups;
        if max_rows <= INLINE_MAX_ROWS {
            counters
        } else {
            counters + META_HEADER_U32 + pairs.div_ceil(2)
        }
    }

    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "experts_rows", cubin!("experts_rows"))?;
        let sms = usize::try_from(
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        )?;
        let load = |name: &str, smem: usize| -> anyhow::Result<(CudaFunction, usize)> {
            let function = module.load_function(name)?;
            function.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(smem)?,
            )?;
            let per_sm = function.occupancy_max_active_blocks_per_multiprocessor(
                u32::try_from(WARPS * 32)?,
                smem,
                None,
            )?;
            let per_sm = usize::try_from(per_sm)?;
            ensure!(per_sm > 0, "{name} does not fit on an SM");
            Ok((function, per_sm * sms))
        };
        let load_ring = |ring: usize| -> anyhow::Result<([CudaFunction; 2], [usize; 2])> {
            let (nvfp4, nvfp4_blocks) = load(&format!("expert_mlp_rows_nvfp4_{ring}"), smem(ring))?;
            let (mxfp8, mxfp8_blocks) = load(&format!("expert_mlp_rows_mxfp8_{ring}"), smem(ring))?;
            Ok(([nvfp4, mxfp8], [nvfp4_blocks, mxfp8_blocks]))
        };
        let (shallow, shallow_blocks) = load_ring(RINGS[0])?;
        let (deep, deep_blocks) = load_ring(RINGS[1])?;
        let group = module.load_function("expert_group_rows")?;
        let (wide_nvfp4, wide_nvfp4_blocks) =
            load("expert_mlp_rows_wide_nvfp4_3", smem(RINGS[0]))?;
        let (wide_mxfp8, wide_mxfp8_blocks) =
            load("expert_mlp_rows_wide_mxfp8_3", smem(RINGS[0]))?;
        Ok(Self {
            kernels: [shallow, deep],
            blocks: [shallow_blocks, deep_blocks],
            group,
            wide_kernels: [wide_nvfp4, wide_mxfp8],
            wide_blocks: [wide_nvfp4_blocks, wide_mxfp8_blocks],
        })
    }

    /// The expert MLP of `m` rows, 1 to [`MAX_ROWS`]: for row `r` and slot `s`, with
    /// `e = indices[r][s]`, `act[8 r + s] = softcapped_swiglu(x[r] @ w13[e]^T)` and
    /// `out[8 r + s] = act[8 r + s] @ w2[e]^T`.
    ///
    /// - `x`: BF16 `[m, 2304]`; `indices`: `[m, 8]` expert ids, in range and distinct within
    ///   each row, as the router writes them;
    /// - `w13`, `w2`: one layer's projections, both NVFP4 or both MXFP8, with `w13` of
    ///   `2304` interleaved gate and up rows of input width 2304 and `w2` of 2304 rows of
    ///   input width 1152;
    /// - `act`: BF16 `[m * 8, 1152]` and `out`: BF16 `[m * 8, 2304]`, written;
    /// - `sync`: at least `1 + 9 m` zeros, which the kernel leaves zero.
    ///
    /// Launches with programmatic dependent launch; the kernel reads nothing before the
    /// previous kernel ends.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "the MLP's inputs, validation, and inline/wide launch"
    )]
    pub fn expert_mlp_rows(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        indices: &impl DeviceIn<i32>,
        w13: &ExpertWeights<'_>,
        w2: &ExpertWeights<'_>,
        act: &mut impl DeviceOut<bf16>,
        out: &mut impl DeviceOut<bf16>,
        sync: &mut impl DeviceOut<u32>,
        gate_softcap: f32,
        up_softcap: f32,
    ) -> anyhow::Result<()> {
        let (x, indices, mut act, mut out, mut sync) = (
            x.view(),
            indices.view(),
            act.view_mut(),
            out.view_mut(),
            sync.view_mut(),
        );
        ensure!(
            indices.len().is_multiple_of(SLOTS),
            "indices has {} ids, not whole rows of {SLOTS}",
            indices.len()
        );
        let m = indices.len() / SLOTS;
        ensure!(
            (1..=MAX_ROWS).contains(&m),
            "{m} rows, expected 1 to {MAX_ROWS}"
        );
        ensure!(
            x.len() == m * K13,
            "x has {} values, expected {m} x {K13}",
            x.len()
        );
        ensure!(
            act.len() == m * SLOTS * K2 && out.len() == m * SLOTS * N2,
            "act has {} and out {} values, expected {m} x 8 x {K2} and x {N2}",
            act.len(),
            out.len()
        );
        // One counter per column group of up to 8 pairs, plus grouped routing metadata
        // above 32 rows (`Self::sync_len(m)`).
        let groups = (m * SLOTS).min(EXPERTS) + m;
        let sync_needed = Self::sync_len(m);
        ensure!(
            sync.len() >= sync_needed,
            "sync has {} words, expected at least {sync_needed}",
            sync.len(),
        );
        ensure!(
            w13.k == K13 && w13.n == 2 * K2 && w2.k == K2 && w2.n == N2,
            "w13 must be {} x {K13} and w2 {N2} x {K2}, not {} x {} and {} x {}",
            2 * K2,
            w13.n,
            w13.k,
            w2.n,
            w2.k
        );
        let format = match (w13.format, w2.format) {
            (ExpertFormat::Nvfp4 { .. }, ExpertFormat::Nvfp4 { .. }) => 0,
            (ExpertFormat::Mxfp8, ExpertFormat::Mxfp8) => 1,
            _ => anyhow::bail!("w13 and w2 must share a format"),
        };
        let experts = |p: &ExpertWeights<'_>, row_bytes: usize, group: usize| {
            let per_expert = p.n * row_bytes;
            ensure!(
                p.weight.len().is_multiple_of(per_expert) && !p.weight.is_empty(),
                "weight of {} bytes is not whole experts",
                p.weight.len()
            );
            let n = p.weight.len() / per_expert;
            ensure!(
                p.scales.len() == n * p.n * (p.k / group),
                "scales of {} bytes, expected {n} x {} x {}",
                p.scales.len(),
                p.n,
                p.k / group
            );
            if let ExpertFormat::Nvfp4 { global_scale } = p.format {
                ensure!(global_scale.len() == n, "one global scale per expert");
            }
            Ok(n)
        };
        let (bytes, group) = if format == 0 { (2, 16) } else { (1, 32) };
        let n13 = experts(w13, w13.k / bytes, group)?;
        let n2 = experts(w2, w2.k / bytes, group)?;
        ensure!(n13 == n2, "w13 has {n13} experts and w2 {n2}");

        if m > INLINE_MAX_ROWS {
            let sync_end = i32::try_from(1 + groups)?;
            let mut gargs = KernelArgs::default();
            gargs
                .ptr(&indices, stream)
                .i32(i32::try_from(m)?)
                .ptr_mut(&mut sync, stream)
                .i32(sync_end);
            let gcfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (u32::try_from(WARPS * 32)?, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `expert_group_rows(const int*, int, unsigned*, int)` writes `ExpertRowsMeta`
            // into `sync + sync_end`, which has at least `META_HEADER_U32 + ceil(8*m / 2)` words
            // as checked above, and reads `indices` only after `pdl_wait`.
            unsafe { gargs.launch_pdl(stream, &self.group, gcfg) }?;

            let mut args = KernelArgs::default();
            args.ptr(&x, stream)
                .i32(i32::try_from(m)?)
                .ptr(w13.weight, stream)
                .ptr(w13.scales, stream);
            global_scale(&mut args, w13.format, stream);
            args.ptr(w2.weight, stream).ptr(w2.scales, stream);
            global_scale(&mut args, w2.format, stream);
            args.ptr_mut(&mut act, stream)
                .ptr_mut(&mut out, stream)
                .f32(gate_softcap)
                .f32(up_softcap)
                .ptr_mut(&mut sync, stream)
                .i32(sync_end);
            let config = LaunchConfig {
                grid_dim: (
                    u32::try_from(self.wide_blocks[format]).context("grid")?,
                    1,
                    1,
                ),
                block_dim: (u32::try_from(WARPS * 32)?, 1, 1),
                shared_mem_bytes: u32::try_from(smem(RINGS[0]))?,
            };
            // SAFETY: the arguments match `expert_mlp_rows_wide_*(const bf16*, int, const u8*,
            // const u8*, const float*, const u8*, const u8*, const float*, bf16*, bf16*, float,
            // float, unsigned*, int)`. `expert_group_rows` wrote the metadata at `sync + sync_end`
            // before this kernel's `pdl_wait`, and the last block zeroes both counters and metadata.
            return unsafe { args.launch_pdl(stream, &self.wide_kernels[format], config) };
        }

        let ring = usize::from(m <= DEEP_RING_ROWS);
        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .ptr(&indices, stream)
            .i32(i32::try_from(m)?)
            .ptr(w13.weight, stream)
            .ptr(w13.scales, stream);
        global_scale(&mut args, w13.format, stream);
        args.ptr(w2.weight, stream).ptr(w2.scales, stream);
        global_scale(&mut args, w2.format, stream);
        args.ptr_mut(&mut act, stream)
            .ptr_mut(&mut out, stream)
            .f32(gate_softcap)
            .f32(up_softcap)
            .ptr_mut(&mut sync, stream);
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(self.blocks[ring][format]).context("grid")?,
                1,
                1,
            ),
            block_dim: (u32::try_from(WARPS * 32)?, 1, 1),
            shared_mem_bytes: u32::try_from(smem(RINGS[ring]))?,
        };
        // SAFETY: the arguments match `expert_mlp_rows_*(const bf16*, const int*, int,
        // const u8*, const u8*, const float*, const u8*, const u8*, const float*, bf16*,
        // bf16*, float, float, unsigned*)`. The checks above give `x` m rows of 2304, `act`
        // and `out` 8 m pairs of 1152 and 2304, the weights the kernel's shapes for every
        // expert id the router writes, and `sync` a counter per column group, zero as the kernel
        // leaves them. The grid holds only blocks that fit on the GPU at once with the
        // ring's shared memory, which the kernel's waits need. Only the NVFP4 kernels read
        // the global scales. The kernel reads nothing before `pdl_wait`.
        unsafe { args.launch_pdl(stream, &self.kernels[ring][format], config) }
    }
}

/// The global scale argument: NVFP4's scales, or a null pointer for MXFP8.
fn global_scale(args: &mut KernelArgs, format: ExpertFormat<'_>, stream: &CudaStream) {
    match format {
        ExpertFormat::Nvfp4 { global_scale } => args.ptr(global_scale, stream),
        ExpertFormat::Mxfp8 => args.u64(0),
    };
}
