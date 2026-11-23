use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _},
    nvrtc::Ptx,
};
use half::bf16;

/// Hidden width the RMSNorm kernel is compiled for.
const HIDDEN: usize = 2304;
/// Elements each RMSNorm thread loads with one 16-byte access.
const RMS_NORM_VEC: usize = 8;

/// One CUDA device with its stream and loaded kernels.
#[derive(Debug)]
pub struct Gpu {
    stream: Arc<CudaStream>,
    rms_norm: CudaFunction,
}

impl Gpu {
    /// Opens device `ordinal` and loads the embedded cubins.
    pub fn new(ordinal: usize) -> anyhow::Result<Self> {
        let ctx =
            CudaContext::new(ordinal).with_context(|| format!("opening CUDA device {ordinal}"))?;
        let stream = ctx.new_stream()?;
        let module = ctx
            .load_module(Ptx::from_binary(cubin::RMS_NORM.to_vec()))
            .context("loading rms_norm cubin")?;
        let rms_norm = module.load_function("rms_norm_bf16_2304")?;
        Ok(Self { stream, rms_norm })
    }

    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Normalizes each row of `x` (shape `[rows, 2304]`) into `out`, scaled by `weight`.
    pub fn rms_norm(
        &self,
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
        let mut launch = self.stream.launch_builder(&self.rms_norm);
        launch.arg(x).arg(weight).arg(out).arg(&eps);
        // SAFETY: the argument list matches `rms_norm_bf16_2304(const bf16*, const bf16*, bf16*, float)`,
        // and the checks above keep every row inside `x` and `out`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}

mod cubin {
    pub const RMS_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rms_norm.cubin"));
}
