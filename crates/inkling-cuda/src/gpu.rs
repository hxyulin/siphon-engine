use std::sync::Arc;

use anyhow::Context as _;
use cudarc::{
    driver::{CudaContext, CudaModule, CudaStream},
    nvrtc::Ptx,
};

/// One CUDA device and the stream every op launches on.
#[derive(Debug)]
pub struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl Gpu {
    pub fn new(ordinal: usize) -> anyhow::Result<Self> {
        let ctx =
            CudaContext::new(ordinal).with_context(|| format!("opening CUDA device {ordinal}"))?;
        let stream = ctx.new_stream()?;
        Ok(Self { ctx, stream })
    }

    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// Bytes of the cubin that `build.rs` compiled from `kernels/<name>.cu`.
macro_rules! cubin {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".cubin"))
    };
}
pub(crate) use cubin;

pub(crate) fn load_module(
    ctx: &Arc<CudaContext>,
    name: &str,
    cubin: &[u8],
) -> anyhow::Result<Arc<CudaModule>> {
    ctx.load_module(Ptx::from_binary(cubin.to_vec()))
        .with_context(|| format!("loading {name}.cubin"))
}
