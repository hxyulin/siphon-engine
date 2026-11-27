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
/// Most rows the 8-pair single-group inline kernel takes (`kSingleMaxRows` in the kernel).
const SINGLE_MAX_ROWS: usize = 8;
/// Most rows the single-kernel shared-memory path takes (`kMaxRows` in the kernel).
const INLINE_MAX_ROWS: usize = 8;
/// `u32` words of `ExpertRowsMeta`'s fixed header (`offsetof(ExpertRowsMeta, pair_list) / 4`).
const META_HEADER_U32: usize = 2056 / 4;
/// Pairs per chunk for the BF16 wide path.
const WIDE_BF16_CHUNK_CAP: i32 = 32;
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
/// Dynamic shared memory per block for MXFP6 (lo4 128 B + hi2 64 B + scales 16 B per row).
const fn smem_mxfp6(ring: usize) -> usize {
    WARPS * ring * 16 * (128 + 64 + 16)
}

/// CUDA entry points for multi-row mixture-of-experts blocks.
#[derive(Debug)]
pub struct ExpertsRows {
    /// By ring ([`RINGS`]), then format (`0 = nvfp4`, `1 = mxfp8`, `2 = nvfp4_s4`, for `m <= 8`).
    kernels: [[CudaFunction; 3]; 2],
    /// Blocks of each single-group kernel that fit on the GPU at once.
    blocks: [[usize; 3]; 2],
    /// Inline multi-group 8-pair MLP kernels (`9 <= m <= 32`).
    multi_kernels: [CudaFunction; 3],
    /// Blocks of each multi-group kernel that fit on the GPU at once.
    multi_blocks: [usize; 3],
    /// 1-block grouping kernel for `m > 32`.
    group: CudaFunction,
    /// Resident wide MLP kernels (`m > 32`).
    wide_kernels: [CudaFunction; 3],
    /// Blocks of each wide kernel that fit on the GPU at once.
    wide_blocks: [usize; 3],
    /// MXFP6 E2M3 single-group (`m <= 8`), multi-group (`9..=32`), and wide (`> 32`) kernels.
    mxfp6_kernel: CudaFunction,
    mxfp6_blocks: usize,
    mxfp6_multi: CudaFunction,
    mxfp6_multi_blocks: usize,
    mxfp6_wide: CudaFunction,
    mxfp6_wide_blocks: usize,
}

impl ExpertsRows {
    /// Minimum `sync` length in `u32` words for calls of up to `max_rows` rows.
    #[must_use]
    pub const fn sync_len(max_rows: usize) -> usize {
        let pairs = max_rows * SLOTS;
        let groups = if pairs < EXPERTS { pairs } else { EXPERTS } + max_rows;
        if max_rows <= INLINE_MAX_ROWS {
            1 + groups
        } else {
            let counters = (2 + groups).next_multiple_of(2);
            counters + META_HEADER_U32 + pairs.div_ceil(2)
        }
    }

    /// Loads the fused multi-row expert kernels for this CUDA context.
    ///
    /// # Errors
    ///
    /// Returns an error if a CUDA module or kernel cannot load or its launch attributes
    /// cannot be queried or set.
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
        let load_ring = |ring: usize| -> anyhow::Result<([CudaFunction; 3], [usize; 3])> {
            let (nvfp4, nvfp4_blocks) = load(&format!("expert_mlp_rows_nvfp4_{ring}"), smem(ring))?;
            let (mxfp8, mxfp8_blocks) = load(&format!("expert_mlp_rows_mxfp8_{ring}"), smem(ring))?;
            let (nvfp4_s4, nvfp4_s4_blocks) =
                load(&format!("expert_mlp_rows_nvfp4_s4_{ring}"), smem(ring))?;
            Ok((
                [nvfp4, mxfp8, nvfp4_s4],
                [nvfp4_blocks, mxfp8_blocks, nvfp4_s4_blocks],
            ))
        };
        let (shallow, shallow_blocks) = load_ring(RINGS[0])?;
        let (deep, deep_blocks) = load_ring(RINGS[1])?;
        let (multi_nvfp4, multi_nvfp4_blocks) =
            load("expert_mlp_rows_multi_nvfp4_3", smem(RINGS[0]))?;
        let (multi_mxfp8, multi_mxfp8_blocks) =
            load("expert_mlp_rows_multi_mxfp8_3", smem(RINGS[0]))?;
        let (multi_nvfp4_s4, multi_nvfp4_s4_blocks) =
            load("expert_mlp_rows_multi_nvfp4_s4_3", smem(RINGS[0]))?;
        let group = module.load_function("expert_group_rows")?;
        let (wide_nvfp4, wide_nvfp4_blocks) =
            load("expert_mlp_rows_wide_nvfp4_3", smem(RINGS[0]))?;
        let (wide_mxfp8, wide_mxfp8_blocks) =
            load("expert_mlp_rows_wide_mxfp8_3", smem(RINGS[0]))?;
        let (wide_nvfp4_s4, wide_nvfp4_s4_blocks) =
            load("expert_mlp_rows_wide_nvfp4_s4_3", smem(RINGS[0]))?;
        let (mxfp6_kernel, mxfp6_blocks) =
            load("expert_mlp_rows_mxfp6_3", smem_mxfp6(RINGS[0]))?;
        let (mxfp6_multi, mxfp6_multi_blocks) =
            load("expert_mlp_rows_multi_mxfp6_3", smem_mxfp6(RINGS[0]))?;
        let (mxfp6_wide, mxfp6_wide_blocks) =
            load("expert_mlp_rows_wide_mxfp6_3", smem_mxfp6(RINGS[0]))?;
        Ok(Self {
            kernels: [shallow, deep],
            blocks: [shallow_blocks, deep_blocks],
            multi_kernels: [multi_nvfp4, multi_mxfp8, multi_nvfp4_s4],
            multi_blocks: [multi_nvfp4_blocks, multi_mxfp8_blocks, multi_nvfp4_s4_blocks],
            group,
            wide_kernels: [wide_nvfp4, wide_mxfp8, wide_nvfp4_s4],
            wide_blocks: [wide_nvfp4_blocks, wide_mxfp8_blocks, wide_nvfp4_s4_blocks],
            mxfp6_kernel,
            mxfp6_blocks,
            mxfp6_multi,
            mxfp6_multi_blocks,
            mxfp6_wide,
            mxfp6_wide_blocks,
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
            (
                ExpertFormat::Nvfp4 {
                    scale_base: None, ..
                },
                ExpertFormat::Nvfp4 {
                    scale_base: None, ..
                },
            ) => 0,
            (ExpertFormat::Mxfp8, ExpertFormat::Mxfp8) => 1,
            (
                ExpertFormat::Nvfp4 {
                    scale_base: Some(_),
                    ..
                },
                ExpertFormat::Nvfp4 {
                    scale_base: Some(_),
                    ..
                },
            ) => 2,
            (ExpertFormat::Mxfp6 { .. }, ExpertFormat::Mxfp6 { .. }) => 3,
            _ => anyhow::bail!("w13 and w2 must share a format"),
        };
        let experts = |p: &ExpertWeights<'_>, row_bytes: usize, scale_div: usize| {
            let per_expert = p.n * row_bytes;
            ensure!(
                p.weight.len().is_multiple_of(per_expert) && !p.weight.is_empty(),
                "weight of {} bytes is not whole experts",
                p.weight.len()
            );
            let n = p.weight.len() / per_expert;
            ensure!(
                p.scales.len() == n * p.n * (p.k / scale_div),
                "scales of {} bytes, expected {n} x {} x {}",
                p.scales.len(),
                p.n,
                p.k / scale_div
            );
            if let ExpertFormat::Nvfp4 {
                global_scale,
                scale_base,
            } = p.format
            {
                ensure!(global_scale.len() == n, "one global scale per expert");
                if let Some(base) = scale_base {
                    ensure!(
                        base.len() == n * p.n,
                        "scale_base of {} bytes, expected {n} x {}",
                        base.len(),
                        p.n
                    );
                }
            } else if let ExpertFormat::Mxfp6 { hi2 } = p.format {
                ensure!(
                    hi2.len() == n * p.n * (p.k / 4),
                    "hi2 of {} bytes, expected {n} x {} x {}",
                    hi2.len(),
                    p.n,
                    p.k / 4
                );
            }
            Ok(n)
        };
        let (bytes, scale_div) = match format {
            0 => (2, 16),
            1 => (1, 32),
            _ => (2, 32),
        };
        let n13 = experts(w13, w13.k / bytes, scale_div)?;
        let n2 = experts(w2, w2.k / bytes, scale_div)?;
        ensure!(n13 == n2, "w13 has {n13} experts and w2 {n2}");

        if m > INLINE_MAX_ROWS {
            let sync_end = i32::try_from((2 + groups).next_multiple_of(2))?;
            let mut gargs = KernelArgs::default();
            gargs
                .ptr(&indices, stream)
                .i32(i32::try_from(m)?)
                .i32(WIDE_BF16_CHUNK_CAP)
                .ptr_mut(&mut sync, stream)
                .i32(sync_end);
            let gcfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (u32::try_from(WARPS * 32)?, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `expert_group_rows(const int*, int, int, unsigned*, int)` writes
            // `ExpertRowsMeta` into `sync + sync_end` (8-byte aligned), which has at least
            // `META_HEADER_U32 + ceil(8*m / 2)` words as checked above, and reads `indices`
            // only after `pdl_wait`.
            unsafe { gargs.launch_pdl(stream, &self.group, gcfg) }?;

            let mut args = KernelArgs::default();
            args.ptr(&x, stream).i32(i32::try_from(m)?);
            push_weight_args(&mut args, w13, stream);
            push_weight_args(&mut args, w2, stream);
            args.ptr_mut(&mut act, stream)
                .ptr_mut(&mut out, stream)
                .f32(gate_softcap)
                .f32(up_softcap)
                .ptr_mut(&mut sync, stream)
                .i32(sync_end);
            let (wide_kernel, wide_blocks, wide_smem) = if format == 3 {
                (&self.mxfp6_wide, self.mxfp6_wide_blocks, smem_mxfp6(RINGS[0]))
            } else {
                (&self.wide_kernels[format], self.wide_blocks[format], smem(RINGS[0]))
            };
            let config = LaunchConfig {
                grid_dim: (u32::try_from(wide_blocks).context("grid")?, 1, 1),
                block_dim: (u32::try_from(WARPS * 32)?, 1, 1),
                shared_mem_bytes: u32::try_from(wide_smem)?,
            };
            // SAFETY: the arguments match `expert_mlp_rows_wide_*`; `expert_group_rows` wrote the
            // metadata at `sync + sync_end` before this kernel's `pdl_wait`, and the last block
            // zeroes both counters and metadata.
            return unsafe { args.launch_pdl(stream, wide_kernel, config) };
        }

        let (kernel, grid_blocks, smem_bytes) = if format == 3 {
            if m <= SINGLE_MAX_ROWS {
                (&self.mxfp6_kernel, self.mxfp6_blocks, smem_mxfp6(RINGS[0]))
            } else {
                (&self.mxfp6_multi, self.mxfp6_multi_blocks, smem_mxfp6(RINGS[0]))
            }
        } else if m <= SINGLE_MAX_ROWS {
            let ring = usize::from(m <= DEEP_RING_ROWS);
            (
                &self.kernels[ring][format],
                self.blocks[ring][format],
                smem(RINGS[ring]),
            )
        } else {
            (
                &self.multi_kernels[format],
                self.multi_blocks[format],
                smem(RINGS[0]),
            )
        };
        let mut args = KernelArgs::default();
        args.ptr(&x, stream)
            .ptr(&indices, stream)
            .i32(i32::try_from(m)?);
        push_weight_args(&mut args, w13, stream);
        push_weight_args(&mut args, w2, stream);
        args.ptr_mut(&mut act, stream)
            .ptr_mut(&mut out, stream)
            .f32(gate_softcap)
            .f32(up_softcap)
            .ptr_mut(&mut sync, stream);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(grid_blocks).context("grid")?, 1, 1),
            block_dim: (u32::try_from(WARPS * 32)?, 1, 1),
            shared_mem_bytes: u32::try_from(smem_bytes)?,
        };
        // SAFETY: the arguments match `expert_mlp_rows[_multi]_*`. The checks above give `x` m
        // rows of 2304, `act` and `out` 8 m pairs of 1152 and 2304, the weights the kernel's
        // shapes for every expert id the router writes, and `sync` a counter per column group,
        // zero as the kernel leaves them. The grid holds only blocks that fit on the GPU at once
        // with the ring's shared memory, which the kernel's waits need. The kernel reads nothing
        // before `pdl_wait`.
        unsafe { args.launch_pdl(stream, kernel, config) }
    }
}

/// Pushes `(weight, scales, [scale_base,] global_scale)` for NVFP4/MXFP8 or `(lo4, hi2, scales)`
/// for MXFP6.
fn push_weight_args(args: &mut KernelArgs, p: &ExpertWeights<'_>, stream: &CudaStream) {
    match p.format {
        ExpertFormat::Nvfp4 {
            global_scale,
            scale_base,
        } => {
            args.ptr(p.weight, stream).ptr(p.scales, stream);
            if let Some(base) = scale_base {
                args.ptr(base, stream);
            }
            args.ptr(global_scale, stream);
        }
        ExpertFormat::Mxfp8 => {
            args.ptr(p.weight, stream).ptr(p.scales, stream).u64(0);
        }
        ExpertFormat::Mxfp6 { hi2 } => {
            args.ptr(p.weight, stream)
                .ptr(hi2, stream)
                .ptr(p.scales, stream);
        }
    }
}
