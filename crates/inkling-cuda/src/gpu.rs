use std::{ffi::c_void, ptr, sync::Arc};

use anyhow::Context as _;
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
        DevicePtr, DevicePtrMut, LaunchConfig, sys,
    },
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

/// Arguments for [`KernelArgs::launch_pdl`], one 8-byte slot each. Every parameter of the
/// kernels launched this way is a pointer or a 4-byte scalar, and the driver reads a
/// parameter from the start of its slot, which on a little-endian host holds the value.
#[derive(Default)]
pub(crate) struct KernelArgs(Vec<u64>);

impl KernelArgs {
    /// A device buffer. Event tracking is off (see [`Gpu`]), so the guard `device_ptr`
    /// returns does nothing and is dropped at once.
    pub fn ptr<T>(&mut self, x: &impl DevicePtr<T>, stream: &CudaStream) -> &mut Self {
        self.0.push(x.device_ptr(stream).0);
        self
    }

    pub fn ptr_mut<T>(&mut self, x: &mut impl DevicePtrMut<T>, stream: &CudaStream) -> &mut Self {
        self.0.push(x.device_ptr_mut(stream).0);
        self
    }

    /// A device buffer, or a null pointer for `None`.
    pub fn opt<T>(&mut self, x: Option<&impl DevicePtr<T>>, stream: &CudaStream) -> &mut Self {
        self.0.push(x.map_or(0, |x| x.device_ptr(stream).0));
        self
    }

    pub fn opt_mut<T>(
        &mut self,
        x: Option<&mut impl DevicePtrMut<T>>,
        stream: &CudaStream,
    ) -> &mut Self {
        self.0.push(x.map_or(0, |x| x.device_ptr_mut(stream).0));
        self
    }

    pub fn i32(&mut self, v: i32) -> &mut Self {
        self.0.push(u64::from(v.cast_unsigned()));
        self
    }

    pub fn f32(&mut self, v: f32) -> &mut Self {
        self.0.push(u64::from(v.to_bits()));
        self
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.0.push(v);
        self
    }

    /// Launches `func` with programmatic stream serialization, so it may start before the
    /// previous kernel on `stream` finishes. The kernel must call `pdl_wait()` from
    /// `kernels/pdl.cuh` before it reads anything an earlier kernel writes.
    ///
    /// # Safety
    ///
    /// The arguments must match the kernel's parameters in number, order and type, every
    /// pointer must stay valid until the kernel finishes, and the kernel must keep to the
    /// rule above.
    pub unsafe fn launch_pdl(
        &self,
        stream: &CudaStream,
        func: &CudaFunction,
        cfg: LaunchConfig,
    ) -> anyhow::Result<()> {
        let mut params: Vec<*mut c_void> = self
            .0
            .iter()
            .map(|slot| ptr::from_ref(slot).cast_mut().cast())
            .collect();
        // SAFETY: zero is a valid bit pattern for the attribute value union.
        let mut value: sys::CUlaunchAttributeValue = unsafe { std::mem::zeroed() };
        value.programmaticStreamSerializationAllowed = 1;
        let mut attr = sys::CUlaunchAttribute {
            id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            value,
        };
        let config = sys::CUlaunchConfig {
            gridDimX: cfg.grid_dim.0,
            gridDimY: cfg.grid_dim.1,
            gridDimZ: cfg.grid_dim.2,
            blockDimX: cfg.block_dim.0,
            blockDimY: cfg.block_dim.1,
            blockDimZ: cfg.block_dim.2,
            sharedMemBytes: cfg.shared_mem_bytes,
            hStream: stream.cu_stream(),
            attrs: &raw mut attr,
            numAttrs: 1,
        };
        stream.context().bind_to_thread()?;
        // SAFETY: `params` points at one slot per kernel parameter, which outlive the call;
        // the caller vouches for their types and for the kernel.
        unsafe {
            sys::cuLaunchKernelEx(
                &raw const config,
                func.cu_function(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        }
        .result()?;
        Ok(())
    }
}
