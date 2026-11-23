//! RMS normalization and L2 clipping (`kernels/norm.cu`).

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{cubin, load_module};

/// Hidden width of the model.
const HIDDEN: usize = 2304;
/// Elements each RMSNorm thread loads with one 16-byte access.
const RMS_NORM_VEC: usize = 8;

#[derive(Debug)]
pub struct Norm {
    rms_norm_bf16_2304: CudaFunction,
}

impl Norm {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "norm", cubin!("norm"))?;
        Ok(Self {
            rms_norm_bf16_2304: module.load_function("rms_norm_bf16_2304")?,
        })
    }

    /// Weighted RMSNorm of each BF16 row of `x` (shape `[rows, 2304]`) into `out`.
    pub fn rms_norm(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<bf16>,
        weight: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        eps: f32,
    ) -> anyhow::Result<()> {
        ensure!(
            weight.len() == HIDDEN,
            "weight has {} elements, expected {HIDDEN}",
            weight.len()
        );
        ensure!(
            x.len().is_multiple_of(HIDDEN),
            "x has {} elements, not a multiple of {HIDDEN}",
            x.len()
        );
        ensure!(
            out.len() == x.len(),
            "out has {} elements, x has {}",
            out.len(),
            x.len()
        );
        let rows = u32::try_from(x.len() / HIDDEN).context("too many rows")?;
        let config = LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: ((HIDDEN / RMS_NORM_VEC) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.rms_norm_bf16_2304);
        launch.arg(x).arg(weight).arg(out).arg(&eps);
        // SAFETY: the arguments match `rms_norm_bf16_2304(const bf16*, const bf16*, bf16*, float)`,
        // and the checks above keep every row inside `x` and `out`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}
