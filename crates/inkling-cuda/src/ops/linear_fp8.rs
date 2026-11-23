//! Matrix-vector products with FP8 E4M3 weights for decode (`kernels/linear_fp8.cu`).
//!
//! [`LinearFp8::quantize`] turns a BF16 weight into an [`Fp8Matrix`]: E4M3 values with one
//! F32 scale per row, `amax / 448`. [`LinearFp8::gemv`] then runs a batch of products
//! `out = weight @ x` as [`crate::ops::linear::Linear::gemv_bf16`] does, with BF16 inputs
//! and outputs and F32 accumulation, reading half the weight bytes. A [`Start`] picks
//! whether a launch overlaps the previous kernel, as for the BF16 GEMV.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _,
    DevicePtrMut as _, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::{
    DeviceIn, DeviceOut,
    gpu::{cubin, load_module},
    ops::linear::{Prefetch, Start, launch_by_value},
};

/// Products one launch accepts, `kMaxSegments` in the kernel.
pub const MAX_PRODUCTS: usize = 8;
/// Threads per block, `kThreads` in the kernel.
const THREADS: u32 = 128;
/// Rows per block of every entry point: 4 warps of 2 rows.
const ROWS_PER_BLOCK: usize = 8;
/// Rows of an [`Fp8Matrix`] round up to a multiple of this, which `ROWS_PER_BLOCK`
/// divides, so no load leaves the matrix.
const ROW_PAD: usize = 32;

/// A weight `[rows, k]` as E4M3 bytes `[padded_rows, k]` followed by one F32 scale per
/// padded row, in one device buffer. Padding rows are zero.
#[derive(Debug)]
pub struct Fp8Matrix {
    data: CudaSlice<u8>,
    rows: usize,
    k: usize,
}

impl Fp8Matrix {
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }

    fn padded_rows(&self) -> usize {
        self.rows.next_multiple_of(ROW_PAD)
    }

    /// The whole buffer: values, padding and scales.
    #[must_use]
    pub fn buffer(&self) -> &CudaSlice<u8> {
        &self.data
    }

    /// Device bytes the matrix takes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.data.len()
    }

    /// The E4M3 values `[rows, k]`, as bytes.
    #[must_use]
    pub fn values(&self) -> CudaView<'_, u8> {
        self.data.slice(..self.rows * self.k)
    }

    /// The scales of the rows, as the F32 values' little-endian bytes.
    #[must_use]
    pub fn scales(&self) -> CudaView<'_, u8> {
        let start = self.padded_rows() * self.k;
        self.data.slice(start..start + 4 * self.rows)
    }
}

/// One product `out = weight @ x` of a batch.
pub struct Fp8Gemv<'a> {
    weight: &'a Fp8Matrix,
    x: CudaView<'a, bf16>,
    out: CudaViewMut<'a, bf16>,
}

impl<'a> Fp8Gemv<'a> {
    /// `x` has `weight.k()` elements and `out` `weight.rows()`.
    pub fn new(
        weight: &'a Fp8Matrix,
        x: &'a impl DeviceIn<bf16>,
        out: &'a mut impl DeviceOut<bf16>,
    ) -> Self {
        Self {
            weight,
            x: x.view(),
            out: out.view_mut(),
        }
    }
}

/// Kernel parameter: one segment per product, as `Segment` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Segment {
    w: u64,
    scale: u64,
    x: u64,
    out: u64,
    rows: i32,
    block_end: i32,
}

/// Kernel parameter, as `Batch` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Batch {
    seg: [Segment; MAX_PRODUCTS],
    count: i32,
    pdl: i32,
}

/// One entry point and the `k` it serves.
#[derive(Debug)]
struct Entry {
    k: usize,
    function: CudaFunction,
}

#[derive(Debug)]
pub struct LinearFp8 {
    entries: Vec<Entry>,
    quantize: CudaFunction,
}

impl LinearFp8 {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "linear_fp8", cubin!("linear_fp8"))?;
        let entry = |k, name: &str| -> anyhow::Result<Entry> {
            Ok(Entry {
                k,
                function: module.load_function(name)?,
            })
        };
        Ok(Self {
            entries: vec![
                entry(2304, "gemv_fp8_k2304")?,
                entry(3072, "gemv_fp8_k3072")?,
            ],
            quantize: module.load_function("quantize_e4m3_rows")?,
        })
    }

    /// Whether [`Self::gemv`] runs products with input length `k`.
    #[must_use]
    pub fn supports(&self, k: usize) -> bool {
        self.entries.iter().any(|e| e.k == k)
    }

    /// Quantizes `weight`, `[rows, k]` row-major, to E4M3 with one scale per row. `k` must
    /// be one [`Self::supports`].
    pub fn quantize(
        &self,
        stream: &Arc<CudaStream>,
        weight: &CudaSlice<bf16>,
        rows: usize,
        k: usize,
    ) -> anyhow::Result<Fp8Matrix> {
        ensure!(self.supports(k), "no FP8 product for k = {k}");
        self.quantize_any(stream, weight, rows, k)
    }

    /// [`Self::quantize`] for any `k` that is a multiple of 16, for kernels other than
    /// [`Self::gemv`] that read the same layout.
    pub fn quantize_any(
        &self,
        stream: &Arc<CudaStream>,
        weight: &CudaSlice<bf16>,
        rows: usize,
        k: usize,
    ) -> anyhow::Result<Fp8Matrix> {
        ensure!(
            k > 0 && k.is_multiple_of(16),
            "k = {k} is not a positive multiple of 16"
        );
        ensure!(rows > 0, "no rows to quantize");
        ensure!(
            weight.len() == rows * k,
            "weight has {} elements, expected {rows} x {k}",
            weight.len()
        );
        let padded = rows.next_multiple_of(ROW_PAD);
        let mut data = stream.alloc_zeros::<u8>(padded * k + 4 * padded)?;
        {
            let (mut values, mut scales) = data.split_at_mut(padded * k);
            let config = LaunchConfig {
                grid_dim: (u32::try_from(rows)?, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let k_arg = i32::try_from(k)?;
            let mut launch = stream.launch_builder(&self.quantize);
            launch
                .arg(weight)
                .arg(&k_arg)
                .arg(&mut values)
                .arg(&mut scales);
            // SAFETY: one block per row reads the row's k weights and writes its k bytes
            // and its 4-byte scale, all inside the buffers sized above; the scales start at
            // padded * k, a multiple of 16, so the F32 stores are aligned.
            unsafe { launch.launch(config) }?;
        }
        Ok(Fp8Matrix { data, rows, k })
    }

    /// Runs every product in one launch: BF16 inputs and outputs, F32 accumulation, and
    /// each row's scale applied once to its F32 sum before the output rounds to nearest
    /// even. All weights have the same `k`, one [`Self::supports`].
    pub fn gemv(
        &self,
        stream: &CudaStream,
        products: &mut [Fp8Gemv<'_>],
        start: Start,
    ) -> anyhow::Result<()> {
        ensure!(
            (1..=MAX_PRODUCTS).contains(&products.len()),
            "{} products, expected 1 to {MAX_PRODUCTS}",
            products.len()
        );
        let k = products[0].weight.k;
        let entry = self
            .entries
            .iter()
            .find(|e| e.k == k)
            .with_context(|| format!("no FP8 product for k = {k}"))?;
        let mut batch = Batch {
            count: i32::try_from(products.len())?,
            pdl: i32::from(start == Start::Early),
            ..Batch::default()
        };
        let mut blocks = 0usize;
        for (i, p) in products.iter_mut().enumerate() {
            let rows = p.weight.rows;
            ensure!(
                p.weight.k == k,
                "product {i} has k = {}, expected {k}",
                p.weight.k
            );
            ensure!(
                p.x.len() == k,
                "product {i}: input has {} elements",
                p.x.len()
            );
            ensure!(
                p.out.len() == rows,
                "product {i}: output has {} elements, expected {rows}",
                p.out.len()
            );
            blocks += rows.div_ceil(ROWS_PER_BLOCK);
            // Event tracking is off (see `Gpu`), so the guards do nothing.
            let (w, _) = p.weight.data.device_ptr(stream);
            let (x, _) = p.x.device_ptr(stream);
            let (out, _) = p.out.device_ptr_mut(stream);
            ensure!(
                (w | x).is_multiple_of(16),
                "product {i}: weight and input must be 16-byte aligned"
            );
            batch.seg[i] = Segment {
                w,
                scale: w + u64::try_from(p.weight.padded_rows() * k)?,
                x,
                out,
                rows: i32::try_from(rows).context("too many rows")?,
                block_end: i32::try_from(blocks).context("too many rows")?,
            };
        }
        let config = LaunchConfig {
            grid_dim: (u32::try_from(blocks)?, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: u32::try_from(k * size_of::<bf16>())?,
        };
        // SAFETY: `batch` has the kernel's `Batch` layout. Each weight buffer holds
        // `padded_rows` rows of k bytes and their scales, and a segment's blocks cover at
        // most its padded rows, so every load stays inside it; the kernel stores only rows
        // below `rows`, the output length checked above, and reads k inputs. The buffers
        // outlive the borrow of `products`, and the stream orders later work after the
        // kernel. The kernel reads x only after `pdl_wait`.
        unsafe {
            launch_by_value(
                stream,
                &entry.function,
                config,
                &batch,
                start == Start::Early,
            )
        }
    }
}

impl Prefetch for Fp8Matrix {
    fn device_bytes(&self, stream: &CudaStream) -> (u64, usize) {
        self.data.device_bytes(stream)
    }
}
