//! Matrix-vector products with FP8 E4M3 weights for decode (`kernels/linear_fp8.cu`).
//!
//! [`LinearFp8::quantize`] turns a BF16 weight into an [`Fp8Matrix`]: E4M3 values with one
//! F32 scale per row, `amax / 448`. [`LinearFp8::gemv`] then runs a batch of products
//! `out = weight @ x` as [`crate::ops::linear::Linear::gemv_bf16`] does, with BF16 inputs
//! and outputs and F32 accumulation, reading half the weight bytes. A [`Start`] picks
//! whether a launch overlaps the previous kernel, as for the BF16 GEMV.
//!
//! For prefill, [`LinearFp8::gemm`] runs matrix products `out = x @ weight^T` on FP8 tensor
//! cores, with `x` quantized to MXFP8 first (see `kernels/linear_fp8.cu`).

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _,
    DevicePtrMut as _, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::{
    DeviceIn, DeviceOut,
    gpu::{KernelArgs, cubin, load_module},
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
const _: () = assert!(ROW_PAD == siphon::memory::FP8_ROW_PAD);

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

/// A weight `[rows, k]` in MXFP8 format: E4M3 bytes `[padded_rows, k]` followed by one
/// UE8M0 scale byte per 32 values `[padded_rows, k / 32]`, in one device buffer. Padding
/// rows are zero. Used for `W_o` under `--output-projection mxfp8` and `--prefill-dense fp8`,
/// where per-32 block scales preserve outlier head precision that per-row F32 scales lose.
#[derive(Debug)]
pub struct Mxfp8Matrix {
    data: CudaSlice<u8>,
    rows: usize,
    k: usize,
}

impl Mxfp8Matrix {
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

    /// The UE8M0 scales `[rows, k / 32]`, as bytes.
    #[must_use]
    pub fn scales(&self) -> CudaView<'_, u8> {
        let start = self.padded_rows() * self.k;
        self.data.slice(start..start + self.rows * (self.k / 32))
    }
}

/// One product `out = weight @ x` of an MXFP8 GEMV batch.
pub struct Mxfp8Gemv<'a> {
    weight: &'a Mxfp8Matrix,
    x: CudaView<'a, bf16>,
    out: CudaViewMut<'a, bf16>,
}

impl<'a> Mxfp8Gemv<'a> {
    /// `x` has `weight.k()` elements and `out` `weight.rows()`.
    pub fn new(
        weight: &'a Mxfp8Matrix,
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

/// Three tile variants of one FP8 GEMM family (`128x128`, `64x64`, `32x64`).
#[derive(Debug)]
struct GemmFamily {
    large: CudaFunction,
    small: CudaFunction,
    thin: CudaFunction,
}

#[derive(Debug)]
pub struct LinearFp8 {
    entries: Vec<Entry>,
    mxfp8_entries: Vec<Entry>,
    quantize: CudaFunction,
    /// `gemm_fp8_{128x128,64x64,32x64}` (per-row F32 weight scales).
    gemm: GemmFamily,
    /// `gemm_mxfp8_{128x128,64x64,32x64}` (per-32 UE8M0 weight scales).
    gemm_mxfp8: GemmFamily,
    /// `gemm_fp8_128x128_argmax` (fused LM head + tile argmax).
    gemm_argmax: CudaFunction,
    quantize_rows: CudaFunction,
    /// Multiprocessor count, for picking between the `128x128` and `64x64` tiles.
    sms: usize,
}

/// Tile shapes, threads, and dynamic shared memory of the FP8/MXFP8 GEMM kernels
/// (3 stages of `bm` x rows and `bn` W rows of 128 bytes plus `bm` x row scales).
pub const GEMM_TILE: usize = 128;
const GEMM_THREADS: u32 = 256;
const GEMM_SHARED: u32 = 3 * (2 * 128 * 128 + 128 * 4);
const GEMM_SMALL_TILE: usize = 64;
const GEMM_SMALL_THREADS: u32 = 128;
const GEMM_SMALL_SHARED: u32 = 3 * (2 * 64 * 128 + 64 * 4);
const GEMM_THIN_M: usize = 32;
const GEMM_THIN_N: usize = 64;
const GEMM_THIN_SHARED: u32 = 3 * ((32 + 64) * 128 + 32 * 4);
/// k of [`LinearFp8::gemm`] must be a multiple of this: one pipeline stage.
const GEMM_K: usize = 128;

/// Device scratch for [`LinearFp8::gemm`]: the inputs quantized to MXFP8, E4M3 values and
/// one UE8M0 scale per 32.
#[derive(Debug)]
pub struct Fp8Rows {
    values: CudaSlice<u8>,
    scales: CudaSlice<u8>,
}

impl Fp8Rows {
    /// Room for `max_values` input values in all, over the distinct inputs of a call.
    pub fn new(stream: &Arc<CudaStream>, max_values: usize) -> anyhow::Result<Self> {
        Ok(Self {
            values: stream.alloc_zeros(max_values.max(1))?,
            scales: stream.alloc_zeros((max_values / 32).max(1))?,
        })
    }

    /// Device bytes the scratch takes for `max_values`.
    #[must_use]
    pub fn bytes(max_values: usize) -> usize {
        max_values.max(1) + (max_values / 32).max(1)
    }

    /// The E4M3 quantized values buffer.
    #[must_use]
    pub fn values(&self) -> &CudaSlice<u8> {
        &self.values
    }

    /// The UE8M0 block-scale buffer (one byte per 32 values).
    #[must_use]
    pub fn scales(&self) -> &CudaSlice<u8> {
        &self.scales
    }

    /// Splits the leading `count` slots of `rows * k` values into disjoint mutable
    /// `(values, scales)` views for fused norm-quant epilogues.
    pub fn split_slots_mut(
        &mut self,
        rows: usize,
        k: usize,
        count: usize,
    ) -> anyhow::Result<Vec<(CudaViewMut<'_, u8>, CudaViewMut<'_, u8>)>> {
        ensure!(rows > 0, "no rows");
        ensure!(count > 0, "need at least one slot");
        ensure!(
            k > 0 && k.is_multiple_of(32),
            "k = {k} is not a multiple of 32"
        );
        let slot_vals = rows * k;
        let slot_scales = slot_vals / 32;
        ensure!(
            count * slot_vals <= self.values.len() && count * slot_scales <= self.scales.len(),
            "the FP8 scratch holds {} values, need {}",
            self.values.len(),
            count * slot_vals
        );
        let vals = self
            .values
            .slice_mut(..count * slot_vals)
            .chunks_exact_mut(slot_vals);
        let scales = self
            .scales
            .slice_mut(..count * slot_scales)
            .chunks_exact_mut(slot_scales);
        Ok(vals.zip(scales).collect())
    }
}

/// One product `out [m, n] = x [m, k] @ weight [n, k]^T` of [`LinearFp8::gemm`].
pub struct Fp8Gemm<'a> {
    pub weight: &'a Fp8Matrix,
    pub x: CudaView<'a, bf16>,
    pub out: CudaViewMut<'a, bf16>,
}

/// One product `out [m, n] = x_q [m, k] @ weight [n, k]^T` of [`LinearFp8::gemm_quantized`]
/// whose MXFP8 input is already stored in [`Fp8Rows`] at value offset `x_offset`.
pub struct Fp8GemmQuantized<'a> {
    pub weight: &'a Fp8Matrix,
    pub x_offset: usize,
    pub out: CudaViewMut<'a, bf16>,
}

/// One product `out [m, n] = x [m, k] @ weight [n, k]^T` with separate E4M3 values and
/// per-row F32 scales (e.g. draft weights in `ops::draft::Fp8Rows`).
pub struct Fp8RawGemm<'a> {
    pub values: CudaView<'a, u8>,
    pub scales: CudaView<'a, f32>,
    pub rows: usize,
    pub k: usize,
    pub x: CudaView<'a, bf16>,
    pub out: CudaViewMut<'a, bf16>,
}

/// One product `out [m, n] = x [m, k] @ weight [n, k]^T` with MXFP8 block-32 weights.
pub struct Mxfp8Gemm<'a> {
    pub weight: &'a Mxfp8Matrix,
    pub x: CudaView<'a, bf16>,
    pub out: CudaViewMut<'a, bf16>,
}

/// Internal descriptor shared by [`LinearFp8::gemm`], [`LinearFp8::gemm_raw`], and
/// [`LinearFp8::gemm_mxfp8`].
struct RawProduct<'a> {
    w_ptr: u64,
    w_scale_ptr: u64,
    rows: usize,
    k: usize,
    x: CudaView<'a, bf16>,
    out: CudaViewMut<'a, bf16>,
}
/// Kernel parameter: one product, as `gemm8::Product` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GemmProduct {
    w: u64,
    w_scale: u64,
    x: u64,
    x_scale: u64,
    out: u64,
    n: i32,
    tile_end: i32,
}

/// Kernel parameter, as `gemm8::Batch` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GemmBatch {
    p: [GemmProduct; MAX_PRODUCTS],
    count: i32,
    m: i32,
    k: i32,
}

impl LinearFp8 {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "linear_fp8", cubin!("linear_fp8"))?;
        let sms = usize::try_from(ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )?)?;
        let entry = |k, name: &str| -> anyhow::Result<Entry> {
            Ok(Entry {
                k,
                function: module.load_function(name)?,
            })
        };
        let load_gemm = |name: &str, shared: u32| -> anyhow::Result<CudaFunction> {
            let function = module.load_function(name)?;
            function.set_attribute(
                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared.cast_signed(),
            )?;
            Ok(function)
        };
        Ok(Self {
            entries: vec![
                entry(2304, "gemv_fp8_k2304")?,
                entry(3072, "gemv_fp8_k3072")?,
            ],
            mxfp8_entries: vec![entry(3072, "gemv_mxfp8_k3072")?],
            quantize: module.load_function("quantize_e4m3_rows")?,
            gemm: GemmFamily {
                large: load_gemm("gemm_fp8_128x128", GEMM_SHARED)?,
                small: load_gemm("gemm_fp8_64x64", GEMM_SMALL_SHARED)?,
                thin: load_gemm("gemm_fp8_32x64", GEMM_THIN_SHARED)?,
            },
            gemm_mxfp8: GemmFamily {
                large: load_gemm("gemm_mxfp8_128x128", GEMM_SHARED)?,
                small: load_gemm("gemm_mxfp8_64x64", GEMM_SMALL_SHARED)?,
                thin: load_gemm("gemm_mxfp8_32x64", GEMM_THIN_SHARED)?,
            },
            gemm_argmax: load_gemm("gemm_fp8_128x128_argmax", GEMM_SHARED)?,
            quantize_rows: module.load_function("quantize_rows_mxfp8_dense")?,
            sms,
        })
    }

    /// Quantizes `weight` (`[rows, k]` row-major) to MXFP8: E4M3 values `[rows, k]` and
    /// one UE8M0 scale per 32 values `[rows, k / 32]`. `k` must be a multiple of 128.
    pub fn quantize_mxfp8(
        &self,
        stream: &Arc<CudaStream>,
        weight: &CudaSlice<bf16>,
        rows: usize,
        k: usize,
    ) -> anyhow::Result<Mxfp8Matrix> {
        ensure!(rows > 0, "no rows to quantize");
        ensure!(
            k > 0 && k.is_multiple_of(GEMM_K),
            "MXFP8 weight needs k a positive multiple of {GEMM_K}, not {k}"
        );
        ensure!(
            weight.len() == rows * k,
            "weight has {} elements, expected {rows} x {k}",
            weight.len()
        );
        let padded = rows.next_multiple_of(ROW_PAD);
        let mut data = stream.alloc_zeros::<u8>(padded * k + padded * (k / 32))?;
        {
            let (mut values, mut scales) = data.split_at_mut(padded * k);
            let rows_arg = i32::try_from(rows)?;
            let k_arg = i32::try_from(k)?;
            let x = weight.slice(..);
            let mut launch = stream.launch_builder(&self.quantize_rows);
            launch
                .arg(&x)
                .arg(&rows_arg)
                .arg(&k_arg)
                .arg(&mut values)
                .arg(&mut scales);
            let config = LaunchConfig {
                grid_dim: (u32::try_from(rows.div_ceil(8))?, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the arguments match `quantize_rows_mxfp8_dense(const bf16*, int, int, u8*,
            // u8*)`; `weight` holds `rows * k` elements, `values` holds `padded * k` bytes,
            // `scales` holds `padded * (k / 32)` bytes, and `k` is a multiple of 128 (hence 32).
            unsafe { launch.launch(config) }?;
        }
        Ok(Mxfp8Matrix { data, rows, k })
    }

    /// Runs every product `out = x @ weight^T`, all with `m` rows and the same k (a multiple
    /// of 128), on FP8 tensor cores: quantizes each distinct `x` once into `scratch` as
    /// MXFP8 (E4M3 values, a UE8M0 scale per 32, the block's amax / 448 rounded up to a
    /// power of two), multiplies the E4M3 values with the weights' in F32, and applies each
    /// weight row's scale to its sums before they round to BF16. Lowers precision on
    /// purpose: the inputs round to E4M3.
    pub fn gemm(
        &self,
        stream: &CudaStream,
        products: &mut [Fp8Gemm<'_>],
        m: usize,
        scratch: &mut Fp8Rows,
    ) -> anyhow::Result<()> {
        let mut raw: Vec<RawProduct<'_>> = products
            .iter_mut()
            .map(|p| {
                let (w_ptr, _) = p.weight.data.device_ptr(stream);
                let scale_off = u64::try_from(p.weight.padded_rows() * p.weight.k)?;
                Ok(RawProduct {
                    w_ptr,
                    w_scale_ptr: w_ptr + scale_off,
                    rows: p.weight.rows,
                    k: p.weight.k,
                    x: p.x.slice(..),
                    out: p.out.slice_mut(..),
                })
            })
            .collect::<anyhow::Result<_>>()?;
        self.launch_gemm(&self.gemm, stream, &mut raw, m, scratch)
    }

    /// [`Self::gemm`] for FP8 weights stored as separate E4M3 values `[rows, k]` and
    /// per-row F32 scales `[rows]` (such as draft backbone/head weights).
    pub fn gemm_raw(
        &self,
        stream: &CudaStream,
        products: &mut [Fp8RawGemm<'_>],
        m: usize,
        scratch: &mut Fp8Rows,
    ) -> anyhow::Result<()> {
        let mut raw: Vec<RawProduct<'_>> = products
            .iter_mut()
            .enumerate()
            .map(|(i, p)| {
                ensure!(
                    p.values.len() >= p.rows * p.k && p.scales.len() >= p.rows,
                    "product {i}: FP8 raw buffers too short for {} x {}",
                    p.rows,
                    p.k
                );
                let (w_ptr, _) = p.values.device_ptr(stream);
                let (w_scale_ptr, _) = p.scales.device_ptr(stream);
                Ok(RawProduct {
                    w_ptr,
                    w_scale_ptr,
                    rows: p.rows,
                    k: p.k,
                    x: p.x.slice(..),
                    out: p.out.slice_mut(..),
                })
            })
            .collect::<anyhow::Result<_>>()?;
        self.launch_gemm(&self.gemm, stream, &mut raw, m, scratch)
    }

    /// [`Self::gemm`] for MXFP8 block-scaled weights (`[n, k]` E4M3 + `[n, k / 32]` UE8M0
    /// scales): both operands use hardware block-32 scales inside `mma.sync.aligned.kind::mxf8f6f4`.
    pub fn gemm_mxfp8(
        &self,
        stream: &CudaStream,
        products: &mut [Mxfp8Gemm<'_>],
        m: usize,
        scratch: &mut Fp8Rows,
    ) -> anyhow::Result<()> {
        let mut raw: Vec<RawProduct<'_>> = products
            .iter_mut()
            .map(|p| {
                let (w_ptr, _) = p.weight.data.device_ptr(stream);
                let scale_off = u64::try_from(p.weight.padded_rows() * p.weight.k)?;
                Ok(RawProduct {
                    w_ptr,
                    w_scale_ptr: w_ptr + scale_off,
                    rows: p.weight.rows,
                    k: p.weight.k,
                    x: p.x.slice(..),
                    out: p.out.slice_mut(..),
                })
            })
            .collect::<anyhow::Result<_>>()?;
        self.launch_gemm(&self.gemm_mxfp8, stream, &mut raw, m, scratch)
    }

    fn launch_gemm(
        &self,
        family: &GemmFamily,
        stream: &CudaStream,
        products: &mut [RawProduct<'_>],
        m: usize,
        scratch: &mut Fp8Rows,
    ) -> anyhow::Result<()> {
        ensure!(
            (1..=MAX_PRODUCTS).contains(&products.len()),
            "{} products, expected 1 to {MAX_PRODUCTS}",
            products.len()
        );
        ensure!(m > 0, "no rows");
        let k = products[0].k;
        ensure!(
            k > 0 && k.is_multiple_of(GEMM_K),
            "the FP8 GEMM needs k a positive multiple of {GEMM_K}, not {k}"
        );
        let tiles_128: usize = m.div_ceil(GEMM_TILE)
            * products
                .iter()
                .map(|p| p.rows.div_ceil(GEMM_TILE))
                .sum::<usize>();
        let (function, tile_m, tile_n, threads, shared) = if m <= GEMM_THIN_M {
            (
                &family.thin,
                GEMM_THIN_M,
                GEMM_THIN_N,
                GEMM_SMALL_THREADS,
                GEMM_THIN_SHARED,
            )
        } else if m > GEMM_SMALL_TILE && tiles_128 >= self.sms {
            (&family.large, GEMM_TILE, GEMM_TILE, GEMM_THREADS, GEMM_SHARED)
        } else {
            (
                &family.small,
                GEMM_SMALL_TILE,
                GEMM_SMALL_TILE,
                GEMM_SMALL_THREADS,
                GEMM_SMALL_SHARED,
            )
        };
        // Each distinct input, by address, and its offset in the scratch.
        let mut inputs: Vec<(u64, usize)> = Vec::new();
        let mut batch = GemmBatch {
            count: i32::try_from(products.len())?,
            m: i32::try_from(m).context("too many rows")?,
            k: i32::try_from(k)?,
            ..GemmBatch::default()
        };
        let (values_base, _) = scratch.values.device_ptr(stream);
        let (scales_base, _) = scratch.scales.device_ptr(stream);
        let mut tiles = 0usize;
        for (i, p) in products.iter_mut().enumerate() {
            let n = p.rows;
            ensure!(p.k == k, "product {i} has k = {}, expected {k}", p.k);
            ensure!(
                p.x.len() == m * k && p.out.len() == m * n,
                "product {i}: x has {} and out {} elements, expected {m} x {k} and {m} x {n}",
                p.x.len(),
                p.out.len()
            );
            ensure!(
                p.w_ptr.is_multiple_of(16),
                "product {i}: weight must be 16-byte aligned"
            );
            let (x_ptr, _) = p.x.device_ptr(stream);
            let offset = if let Some(&(_, offset)) = inputs.iter().find(|&&(a, _)| a == x_ptr) {
                offset
            } else {
                let offset = inputs.len() * m * k;
                ensure!(
                    offset + m * k <= scratch.values.len(),
                    "the FP8 scratch holds {} values, the call needs {}",
                    scratch.values.len(),
                    offset + m * k
                );
                self.quantize_rows(stream, &p.x, m, k, scratch, offset)?;
                inputs.push((x_ptr, offset));
                offset
            };
            tiles += m.div_ceil(tile_m) * n.div_ceil(tile_n);
            let (out_ptr, _) = p.out.device_ptr_mut(stream);
            batch.p[i] = GemmProduct {
                w: p.w_ptr,
                w_scale: p.w_scale_ptr,
                x: values_base + u64::try_from(offset)?,
                x_scale: scales_base + u64::try_from(offset / 32)?,
                out: out_ptr,
                n: i32::try_from(n).context("too many columns")?,
                tile_end: i32::try_from(tiles).context("too many tiles")?,
            };
        }
        let config = LaunchConfig {
            grid_dim: (u32::try_from(tiles)?, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared,
        };
        // SAFETY: `batch` has the kernel's `gemm8::Batch` layout. Each product's x region
        // holds m rows of k quantized values and m * k / 32 scales, written above on this
        // stream; each weight holds at least n rows of k bytes and its per-row or per-32
        // scales, and the kernel reads only rows below n of it (zero-filling the rest) and
        // x rows below m. It stores only rows below m and columns below n of `out`, sized
        // m x n above. k is a multiple of 128, the stage width, so every 16-byte copy is
        // aligned and in bounds.
        unsafe { launch_by_value(stream, function, config, &batch, false) }
    }

    /// Single-product FP8 tensor-core LM head with a fused 128 x 128 tile-argmax epilogue:
    /// quantizes `x` (`[m, k]`) to MXFP8 in `scratch`, runs `gemm_fp8_128x128_argmax` over
    /// the first `vocab` rows of `weight`, writes one `(max_val, max_idx)` partial per
    /// `(row, n_tile)` into `partial_vals` / `partial_idxs`, and returns `tiles_n`
    /// (`vocab.div_ceil(GEMM_TILE)`). Triggers PDL for the follow-on `argmax_partials`.
    #[expect(clippy::too_many_arguments, reason = "weight, input, vocab, scratch and partials")]
    pub fn gemm_argmax(
        &self,
        stream: &CudaStream,
        weight: &Fp8Matrix,
        x: &impl DeviceIn<bf16>,
        vocab: usize,
        scratch: &mut Fp8Rows,
        partial_vals: &mut impl DeviceOut<f32>,
        partial_idxs: &mut impl DeviceOut<i32>,
        m: usize,
    ) -> anyhow::Result<usize> {
        ensure!(
            crate::BLOCK_SCALED_MMA,
            "FP8 tensor-core LM head needs block-scaled MMA, which sm_89 lacks"
        );
        let (x, mut partial_vals, mut partial_idxs) =
            (x.view(), partial_vals.view_mut(), partial_idxs.view_mut());
        ensure!(m > 0, "no rows");
        ensure!(
            vocab > 0 && vocab <= weight.rows,
            "vocab {vocab} out of range for weight rows {}",
            weight.rows
        );
        let k = weight.k;
        ensure!(
            k.is_multiple_of(GEMM_K),
            "the FP8 GEMM needs k a multiple of {GEMM_K}, not {k}"
        );
        ensure!(
            x.len() == m * k,
            "x has {} elements, expected {m} x {k}",
            x.len()
        );
        ensure!(
            scratch.values.len() >= m * k && scratch.scales.len() >= m * k / 32,
            "the FP8 scratch holds {} values, the call needs {}",
            scratch.values.len(),
            m * k
        );
        let tiles_m = m.div_ceil(GEMM_TILE);
        let tiles_n = vocab.div_ceil(GEMM_TILE);
        ensure!(
            partial_vals.len() >= m * tiles_n && partial_idxs.len() >= m * tiles_n,
            "partial buffers ({}, {}) need at least {} elements ({m} x {tiles_n})",
            partial_vals.len(),
            partial_idxs.len(),
            m * tiles_n
        );
        self.quantize_rows(stream, &x, m, k, scratch, 0)?;
        let (w_ptr, _) = weight.data.device_ptr(stream);
        let w_scale_ptr = w_ptr + u64::try_from(weight.padded_rows() * k)?;
        let (x_vals_ptr, _) = scratch.values.device_ptr(stream);
        let (x_scales_ptr, _) = scratch.scales.device_ptr(stream);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(tiles_m * tiles_n)?, 1, 1),
            block_dim: (GEMM_THREADS, 1, 1),
            shared_mem_bytes: GEMM_SHARED,
        };
        let mut args = KernelArgs::default();
        args.u64(w_ptr)
            .u64(w_scale_ptr)
            .u64(x_vals_ptr)
            .u64(x_scales_ptr)
            .ptr_mut(&mut partial_vals, stream)
            .ptr_mut(&mut partial_idxs, stream)
            .i32(i32::try_from(m).context("too many rows")?)
            .i32(i32::try_from(vocab).context("vocab too large")?)
            .i32(i32::try_from(k)?);
        // SAFETY: arguments match `gemm_fp8_128x128_argmax(const u8*, const float*, const u8*,
        // const u8*, float*, int*, int, int, int)`. `scratch` holds m * k quantized values and
        // m * k / 32 scales written above; `weight` holds at least `vocab` rows of k bytes and
        // scales; `partial_vals` and `partial_idxs` hold at least m * tiles_n elements.
        unsafe { args.launch(stream, &self.gemm_argmax, config) }?;
        Ok(tiles_n)
    }

    /// Runs every product `out = x_q @ weight^T` directly from MXFP8 inputs already written
    /// into `scratch` by fused norm-quant epilogues, skipping `quantize_rows_mxfp8_dense`.
    pub fn gemm_quantized(
        &self,
        stream: &CudaStream,
        products: &mut [Fp8GemmQuantized<'_>],
        m: usize,
        scratch: &Fp8Rows,
    ) -> anyhow::Result<()> {
        ensure!(
            (1..=MAX_PRODUCTS).contains(&products.len()),
            "{} products, expected 1 to {MAX_PRODUCTS}",
            products.len()
        );
        ensure!(m > 0, "no rows");
        let k = products[0].weight.k;
        ensure!(
            k.is_multiple_of(GEMM_K),
            "the FP8 GEMM needs k a multiple of {GEMM_K}, not {k}"
        );
        let tiles_128: usize = m.div_ceil(GEMM_TILE)
            * products
                .iter()
                .map(|p| p.weight.rows.div_ceil(GEMM_TILE))
                .sum::<usize>();
        let (function, tile_m, tile_n, threads, shared) = if m <= GEMM_THIN_M {
            (
                &self.gemm.thin,
                GEMM_THIN_M,
                GEMM_THIN_N,
                GEMM_SMALL_THREADS,
                GEMM_THIN_SHARED,
            )
        } else if m > GEMM_SMALL_TILE && tiles_128 >= self.sms {
            (
                &self.gemm.large,
                GEMM_TILE,
                GEMM_TILE,
                GEMM_THREADS,
                GEMM_SHARED,
            )
        } else {
            (
                &self.gemm.small,
                GEMM_SMALL_TILE,
                GEMM_SMALL_TILE,
                GEMM_SMALL_THREADS,
                GEMM_SMALL_SHARED,
            )
        };
        let mut batch = GemmBatch {
            count: i32::try_from(products.len())?,
            m: i32::try_from(m).context("too many rows")?,
            k: i32::try_from(k)?,
            ..GemmBatch::default()
        };
        let (values_base, _) = scratch.values.device_ptr(stream);
        let (scales_base, _) = scratch.scales.device_ptr(stream);
        let mut tiles = 0usize;
        for (i, p) in products.iter_mut().enumerate() {
            let n = p.weight.rows;
            ensure!(
                p.weight.k == k,
                "product {i} has k = {}, expected {k}",
                p.weight.k
            );
            ensure!(
                p.out.len() == m * n,
                "product {i}: out has {} elements, expected {m} x {n}",
                p.out.len()
            );
            ensure!(
                p.x_offset.is_multiple_of(32) && p.x_offset + m * k <= scratch.values.len(),
                "product {i}: x_offset {} + {} exceeds scratch capacity {}",
                p.x_offset,
                m * k,
                scratch.values.len()
            );
            tiles += m.div_ceil(tile_m) * n.div_ceil(tile_n);
            let (w_ptr, _) = p.weight.data.device_ptr(stream);
            let (out_ptr, _) = p.out.device_ptr_mut(stream);
            batch.p[i] = GemmProduct {
                w: w_ptr,
                w_scale: w_ptr + u64::try_from(p.weight.padded_rows() * k)?,
                x: values_base + u64::try_from(p.x_offset)?,
                x_scale: scales_base + u64::try_from(p.x_offset / 32)?,
                out: out_ptr,
                n: i32::try_from(n).context("too many columns")?,
                tile_end: i32::try_from(tiles).context("too many tiles")?,
            };
        }
        let config = LaunchConfig {
            grid_dim: (u32::try_from(tiles)?, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared,
        };
        // SAFETY: `batch` has the kernel's `gemm8::Batch` layout. Each product's `x_offset`
        // points to `m * k` MXFP8 values and `m * k / 32` scales inside `scratch` written on
        // this stream; each weight holds `padded_rows` rows of `k` bytes and their scales,
        // and `out` holds `m * n` BF16 values. `k` is a multiple of 128.
        unsafe { launch_by_value(stream, function, config, &batch, false) }
    }

    /// Quantizes `x` (`[rows, k]`) into `scratch` at value offset `offset` (a multiple of
    /// 32 k).
    fn quantize_rows(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, bf16>,
        rows: usize,
        k: usize,
        scratch: &mut Fp8Rows,
        offset: usize,
    ) -> anyhow::Result<()> {
        let mut values = scratch.values.slice_mut(offset..offset + rows * k);
        let mut scales = scratch
            .scales
            .slice_mut(offset / 32..(offset + rows * k) / 32);
        let rows_arg = i32::try_from(rows)?;
        let k_arg = i32::try_from(k)?;
        let mut launch = stream.launch_builder(&self.quantize_rows);
        launch
            .arg(x)
            .arg(&rows_arg)
            .arg(&k_arg)
            .arg(&mut values)
            .arg(&mut scales);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows.div_ceil(8))?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `quantize_rows_mxfp8_dense(const bf16*, int, int, u8*,
        // u8*)`; `x` holds rows * k values, `values` rows * k bytes and `scales` rows * k / 32,
        // and k is a multiple of 32.
        unsafe { launch.launch(config) }?;
        Ok(())
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

    /// Runs every MXFP8 block-32 product in one launch: BF16 inputs and outputs, F32
    /// accumulation, with each 32-element UE8M0 block scale applied as weights accumulate.
    pub fn gemv_mxfp8(
        &self,
        stream: &CudaStream,
        products: &mut [Mxfp8Gemv<'_>],
        start: Start,
    ) -> anyhow::Result<()> {
        ensure!(
            (1..=MAX_PRODUCTS).contains(&products.len()),
            "{} products, expected 1 to {MAX_PRODUCTS}",
            products.len()
        );
        let k = products[0].weight.k;
        let entry = self
            .mxfp8_entries
            .iter()
            .find(|e| e.k == k)
            .with_context(|| format!("no MXFP8 GEMV product for k = {k}"))?;
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
        // `padded_rows` rows of k bytes and `padded_rows * (k / 32)` scale bytes, and a
        // segment's blocks cover at most its padded rows, so every load stays inside it;
        // the kernel stores only rows below `rows` and reads x only after `pdl_wait`.
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

impl Prefetch for Mxfp8Matrix {
    fn device_bytes(&self, stream: &CudaStream) -> (u64, usize) {
        self.data.device_bytes(stream)
    }
}
