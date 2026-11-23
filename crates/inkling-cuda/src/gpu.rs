use std::sync::Arc;

use anyhow::Context as _;
use cudarc::{
    driver::{CudaContext, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut},
    nvrtc::Ptx,
};

/// One CUDA device and the stream every op launches on.
///
/// All work runs on that one stream, so cudarc's per-buffer events, which order work
/// across streams, are turned off. They would also break CUDA graph capture: a captured
/// stream cannot wait on an event recorded outside the capture.
#[derive(Debug)]
pub struct Gpu {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl Gpu {
    pub fn new(ordinal: usize) -> anyhow::Result<Self> {
        let ctx =
            CudaContext::new(ordinal).with_context(|| format!("opening CUDA device {ordinal}"))?;
        // SAFETY: nothing has been allocated in this context yet, and every buffer and launch
        // afterwards goes through the one stream below, which orders all work itself.
        unsafe { ctx.disable_event_tracking() };
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

/// Device memory a launcher reads: a whole buffer or a view of rows of one.
pub trait DeviceIn<T> {
    fn view(&self) -> CudaView<'_, T>;
}

impl<T> DeviceIn<T> for CudaSlice<T> {
    fn view(&self) -> CudaView<'_, T> {
        self.as_view()
    }
}

impl<T> DeviceIn<T> for CudaView<'_, T> {
    fn view(&self) -> CudaView<'_, T> {
        self.slice(..)
    }
}

/// Device memory a launcher writes: a whole buffer or a view of rows of one.
pub trait DeviceOut<T> {
    fn view_mut(&mut self) -> CudaViewMut<'_, T>;
}

impl<T> DeviceOut<T> for CudaSlice<T> {
    fn view_mut(&mut self) -> CudaViewMut<'_, T> {
        self.as_view_mut()
    }
}

impl<T> DeviceOut<T> for CudaViewMut<'_, T> {
    fn view_mut(&mut self) -> CudaViewMut<'_, T> {
        self.slice_mut(..)
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
