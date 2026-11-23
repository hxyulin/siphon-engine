//! NVFP4 and MXFP8 expert matrix-vector products (`kernels/experts.cu`).

use std::sync::Arc;

use anyhow::{Context as _, bail, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, cubin, load_module};

/// Experts each token routes to.
const SLOTS: usize = 8;
/// Output rows per block, one warp each.
const WARPS: u32 = 8;

/// Quantization format of one expert projection.
#[derive(Debug, Clone, Copy)]
pub enum ExpertFormat<'a> {
    /// Layers 0 to 19: E2M1 weights, one E4M3 scale per 16 values, and one F32 global scale per
    /// expert (`[experts]`), computed as `amax * 6 * 448 / outer_scale_denominator`.
    Nvfp4 { global_scale: &'a CudaSlice<f32> },
    /// Layers 20 to 23: E4M3 weights and one E8M0 scale per 32 values.
    Mxfp8,
}

impl ExpertFormat<'_> {
    fn group(self) -> usize {
        match self {
            Self::Nvfp4 { .. } => 16,
            Self::Mxfp8 => 32,
        }
    }

    fn row_bytes(self, k: usize) -> usize {
        match self {
            Self::Nvfp4 { .. } => k / 2,
            Self::Mxfp8 => k,
        }
    }
}

/// One expert projection (`w13` or `w2`) as raw checkpoint bytes on the GPU.
#[derive(Debug, Clone, Copy)]
pub struct ExpertWeights<'a> {
    pub format: ExpertFormat<'a>,
    /// `*_weight`: `[experts, n, k / 2]` for NVFP4, `[experts, n, k]` for MXFP8.
    pub weight: &'a CudaSlice<u8>,
    /// `*_weight_scale`: `[experts, n, k / 16]` for NVFP4, `[experts, n, k / 32]` for MXFP8.
    pub scales: &'a CudaSlice<u8>,
    /// Output rows per expert: 2304 for both projections. Tests pass a slice.
    pub n: usize,
    /// Input width: 2304 for `w13`, 1152 for `w2`.
    pub k: usize,
}

#[derive(Debug)]
pub struct Experts {
    nvfp4_8: CudaFunction,
    nvfp4_16: CudaFunction,
    mxfp8_4: CudaFunction,
    mxfp8_8: CudaFunction,
}

impl Experts {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "experts", cubin!("experts"))?;
        Ok(Self {
            nvfp4_8: module.load_function("expert_matvec_nvfp4_8")?,
            nvfp4_16: module.load_function("expert_matvec_nvfp4_16")?,
            mxfp8_4: module.load_function("expert_matvec_mxfp8_4")?,
            mxfp8_8: module.load_function("expert_matvec_mxfp8_8")?,
        })
    }

    /// Multiplies each row's input by the weights of its 8 selected experts, writing BF16
    /// `[rows, 8, n]` to `out`.
    ///
    /// `indices` is `[rows, 8]`. `x` is BF16 `[rows, 1, k]`, one input shared by all 8 slots
    /// (`w13`), or `[rows, 8, k]`, one input per slot (`w2`). Each id must be below the expert
    /// count of `weights`; an id outside it gives a NaN row.
    ///
    /// Equals MLX's `mx.gather_qmm` on one row per expert, bit for bit.
    pub fn expert_matvec(
        &self,
        stream: &CudaStream,
        x: &impl DeviceIn<bf16>,
        indices: &impl DeviceIn<i32>,
        weights: &ExpertWeights<'_>,
        out: &mut impl DeviceOut<bf16>,
    ) -> anyhow::Result<()> {
        let (x, indices, mut out) = (x.view(), indices.view(), out.view_mut());
        let &ExpertWeights {
            format,
            weight,
            scales,
            n,
            k,
        } = weights;
        let group = format.group();
        ensure!(n > 0, "n must be positive");
        ensure!(
            k > 0 && k.is_multiple_of(group),
            "k = {k} is not a positive multiple of {group}"
        );
        let row_bytes = n * format.row_bytes(k);
        ensure!(
            weight.len().is_multiple_of(row_bytes) && !weight.is_empty(),
            "weight has {} bytes, not a whole number of {n} x {} experts",
            weight.len(),
            format.row_bytes(k)
        );
        let experts = weight.len() / row_bytes;
        ensure!(
            scales.len() == experts * n * (k / group),
            "scales has {} bytes, expected {experts} x {n} x {}",
            scales.len(),
            k / group
        );
        if let ExpertFormat::Nvfp4 { global_scale } = format {
            ensure!(
                global_scale.len() == experts,
                "global_scale has {} elements, expected {experts}",
                global_scale.len()
            );
        }
        ensure!(
            indices.len().is_multiple_of(SLOTS),
            "indices has {} elements, not a multiple of {SLOTS}",
            indices.len()
        );
        let rows = indices.len() / SLOTS;
        if rows == 0 {
            return Ok(());
        }
        // The kernel's `inputs_per_row`: 1 for one input shared by the 8 slots, 8 for one each.
        let inputs_arg: i32 = if x.len() == rows * k {
            1
        } else if x.len() == rows * SLOTS * k {
            8
        } else {
            bail!(
                "x has {} elements, expected {rows} x 1 or {rows} x {SLOTS} inputs of {k}",
                x.len()
            );
        };
        ensure!(
            out.len() == rows * SLOTS * n,
            "out has {} elements, expected {rows} x {SLOTS} x {n}",
            out.len()
        );

        // MLX's rule for its fast kernel, which gives each lane twice as many values.
        let (fast_k, function_fast, function) = match format {
            ExpertFormat::Nvfp4 { .. } => (512, &self.nvfp4_16, &self.nvfp4_8),
            ExpertFormat::Mxfp8 => (256, &self.mxfp8_8, &self.mxfp8_4),
        };
        let fast = n.is_multiple_of(WARPS as usize) && k.is_multiple_of(fast_k);
        let function = if fast { function_fast } else { function };

        let slot_rows = u32::try_from(rows * SLOTS).context("too many rows")?;
        let tiles = u32::try_from(n.div_ceil(WARPS as usize)).context("n is too large")?;
        let n_arg = i32::try_from(n).context("n is too large")?;
        let k_arg = i32::try_from(k).context("k is too large")?;
        let experts_arg = i32::try_from(experts).context("too many experts")?;
        // A null pointer stands in for MXFP8's missing global scale.
        let null: u64 = 0;
        let config = LaunchConfig {
            grid_dim: (slot_rows, tiles, 1),
            block_dim: (WARPS * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&x).arg(&indices).arg(weight).arg(scales);
        match format {
            ExpertFormat::Nvfp4 { global_scale } => launch.arg(global_scale),
            ExpertFormat::Mxfp8 => launch.arg(&null),
        };
        launch
            .arg(&mut out)
            .arg(&n_arg)
            .arg(&k_arg)
            .arg(&inputs_arg)
            .arg(&experts_arg);
        // SAFETY: the arguments match `expert_matvec_*(const bf16*, const int*, const u8*,
        // const u8*, const float*, bf16*, int, int, int, int)`. The checks above keep every input
        // row inside `x`, every output inside `out`, and every weight and scale row of an expert
        // below `experts` inside `weight` and `scales`; the kernel skips other expert ids. Only
        // the NVFP4 kernels read `global_scale`, which then holds `experts` values.
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}
