use std::{ffi::c_void, mem::ManuallyDrop, ptr, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
        DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PinnedHostSlice, ValidAsZeroBits, sys,
    },
    nvrtc::Ptx,
};

/// The target the kernels are compiled for, `SIPHON_CUDA_ARCH` at build time.
const KERNEL_ARCH: &str = env!("SIPHON_CUDA_ARCH");

/// Whether the kernels have the block-scaled MMA (`mma.sync ... block_scale`), which only
/// `sm_120a` has. The `sm_89` build for the RTX 4090 lacks it, so it refuses the prefill
/// options that need it: FP8 global attention, quantized or FP8 expert activations and FP8
/// dense products.
pub const BLOCK_SCALED_MMA: bool = !cfg!(sm89);

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
        let (major, minor) = ctx.compute_capability()?;
        let device = format!("sm_{major}{minor}");
        let runs = if cfg!(sm89) {
            "the RTX 4090"
        } else {
            "the RTX 5090 and RTX PRO 6000 Blackwell"
        };
        ensure!(
            KERNEL_ARCH.trim_end_matches('a') == device,
            "the kernels are built for {KERNEL_ARCH}, which {runs} run, but CUDA device \
             {ordinal}, {}, is {device}",
            ctx.name()?
        );
        // SAFETY: nothing has been allocated in this context yet, and every buffer and launch
        // afterwards goes through the one stream below, which orders all work itself.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream()?;
        Ok(Self { ctx, stream })
    }

    /// The device's streaming multiprocessors.
    pub fn sms(&self) -> anyhow::Result<usize> {
        let sms = self
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        Ok(usize::try_from(sms)?)
    }

    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    #[must_use]
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The device's free memory and what this process's buffers take from the memory pool
    /// that every allocation comes from (cudarc allocates with `cuMemAllocAsync`). The pool
    /// leaves out the CUDA context, the module images and the CUDA graphs.
    pub fn memory(&self) -> anyhow::Result<MemoryUse> {
        use sys::CUmemPool_attribute as A;

        self.ctx.bind_to_thread()?;
        let (free, total) = self.ctx.mem_get_info()?;
        // SAFETY: the device comes from a live context, and each attribute below is a u64.
        let pool = unsafe { cudarc::driver::result::device::get_mem_pool(self.ctx.cu_device()) }?;
        let attribute = |attr| -> anyhow::Result<usize> {
            let mut value = 0u64;
            // SAFETY: `pool` is the device's pool and `value` is the u64 the attribute
            // writes.
            unsafe {
                sys::cuMemPoolGetAttribute(pool, attr, ptr::from_mut(&mut value).cast::<c_void>())
            }
            .result()
            .context("reading a memory pool attribute")?;
            Ok(usize::try_from(value)?)
        };
        Ok(MemoryUse {
            free,
            total,
            pool_used: attribute(A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
            pool_used_peak: attribute(A::CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
            pool_reserved: attribute(A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
            pool_reserved_peak: attribute(A::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)?,
        })
    }
}

/// A read-only buffer in page-locked host memory that kernels read through its mapped
/// device address, over PCIe, without taking device memory. For tensors that kernels only
/// gather a few rows of, such as the embedding table. The address stays valid inside CUDA
/// graphs.
#[derive(Debug)]
pub struct HostMapped<T> {
    /// A device slice of the mapped address, which this owns but never frees: it is leaked
    /// in `drop`, before the host memory goes.
    device: ManuallyDrop<CudaSlice<T>>,
    host: PinnedHostSlice<T>,
}

impl<T: DeviceRepr + ValidAsZeroBits + Copy> HostMapped<T> {
    /// Copies `values` into new page-locked, write-combined host memory mapped into the
    /// device's address space.
    pub fn new(stream: &Arc<CudaStream>, values: &[T]) -> anyhow::Result<Self> {
        let ctx = stream.context();
        // SAFETY: every element is written below before any kernel reads it.
        let mut host = unsafe {
            ctx.alloc_pinned_with_flags::<T>(
                values.len(),
                sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_WRITECOMBINED,
            )
        }
        .context("allocating mapped host memory")?;
        host.as_mut_slice()?.copy_from_slice(values);
        let mut address: sys::CUdeviceptr = 0;
        // SAFETY: `host` was allocated with CU_MEMHOSTALLOC_DEVICEMAP in this context, which
        // is bound to the thread, and `address` is where the call writes its device address.
        unsafe {
            sys::cuMemHostGetDevicePointer_v2(
                &raw mut address,
                host.as_mut_ptr()?.cast::<c_void>(),
                0,
            )
        }
        .result()
        .context("mapping host memory")?;
        // SAFETY: `address` maps the `values.len()` elements of `host`, which outlives the
        // slice: `drop` leaks the slice, so cudarc never frees the address as device memory.
        let device = unsafe { stream.upgrade_device_ptr::<T>(address, values.len()) };
        Ok(Self {
            device: ManuallyDrop::new(device),
            host,
        })
    }
}

impl<T> HostMapped<T> {
    /// The buffer as a device slice, for kernels to read. Kernels must not write it.
    #[must_use]
    pub fn slice(&self) -> &CudaSlice<T> {
        &self.device
    }

    /// Bytes of host memory the buffer takes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.host.num_bytes()
    }
}

impl<T> Drop for HostMapped<T> {
    fn drop(&mut self) {
        // SAFETY: `device` is not used again. Leaking it keeps cudarc from freeing a host
        // address as device memory; `host` frees the memory after this.
        let device = unsafe { ManuallyDrop::take(&mut self.device) };
        let stream = Arc::clone(device.stream());
        let _ = device.leak();
        // Kernels that read the buffer may still be queued.
        let _ = stream.synchronize();
    }
}

/// Device memory as [`Gpu::memory`] reads it, in bytes.
#[derive(Clone, Copy, Debug)]
pub struct MemoryUse {
    /// Free on the device, for every process.
    pub free: usize,
    pub total: usize,
    /// In this process's live buffers, now and at most so far.
    pub pool_used: usize,
    pub pool_used_peak: usize,
    /// Held by the pool from the driver, its rounding included, now and at most so far.
    pub pool_reserved: usize,
    pub pool_reserved_peak: usize,
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

    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.0.push(u64::from(v));
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

    /// Launches `func` without programmatic stream serialization.
    ///
    /// # Safety
    ///
    /// The arguments must match the kernel's parameters in number, order and type, and
    /// every pointer must stay valid until the kernel finishes.
    pub unsafe fn launch(
        &self,
        stream: &CudaStream,
        func: &CudaFunction,
        cfg: LaunchConfig,
    ) -> anyhow::Result<()> {
        // SAFETY: forwarded to `launch_with_pdl` with `pdl = false`.
        unsafe { self.launch_with_pdl(stream, func, cfg, false) }
    }

    /// Launches `func` with programmatic stream serialization, so it may start before the
    /// previous kernel on `stream` finishes. The kernel must call `pdl_wait()` from
    /// `kernels/pdl.cuh` before it reads anything an earlier kernel writes. The `sm_89`
    /// build launches without it, since Ada has no programmatic dependent launch.
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
        // SAFETY: forwarded to `launch_with_pdl` with `pdl = true`.
        unsafe { self.launch_with_pdl(stream, func, cfg, true) }
    }

    /// Shared launcher for [`Self::launch`] and [`Self::launch_pdl`].
    ///
    /// # Safety
    ///
    /// Same preconditions as [`Self::launch_pdl`] when `pdl` is true, or [`Self::launch`]
    /// otherwise.
    unsafe fn launch_with_pdl(
        &self,
        stream: &CudaStream,
        func: &CudaFunction,
        cfg: LaunchConfig,
        pdl: bool,
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
            numAttrs: u32::from(pdl && !cfg!(sm89)),
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
