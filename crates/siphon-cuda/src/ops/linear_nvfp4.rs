//! Matrix-vector products with NVFP4 (E2M1) weights for decode (`kernels/linear_nvfp4.cu`).
//!
//! [`LinearNvfp4::quantize`] / [`LinearNvfp4::quantize_any`] turn a BF16 weight into an
//! [`Nvfp4Matrix`]: packed E2M1 pairs, one E4M3 block scale per 16 values (Four Over Six by
//! squared error), and one F32 outer scale per row, `amax / 1536`. [`LinearNvfp4::gemv`]
//! runs a batch of products `out = weight @ x` with BF16 inputs and outputs and F32
//! accumulation, and [`LinearNvfp4::gemv_argmax`] runs a fused greedy LM-head argmax.

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
/// Maximum partial blocks [`LinearNvfp4::gemv_argmax`] writes per launch.
pub const ARGMAX_MAX_BLOCKS: usize = 2048;
/// Threads per block, `kThreads` in the kernel.
const THREADS: u32 = 128;
/// Warps per block, `kWarps` in the kernel.
const WARPS: usize = 4;
/// Rows per tile of `gemv_nvfp4_k2304_argmax` (`kWarps * R = 4 * 4`).
const ARGMAX_TILE_ROWS: usize = 16;
/// Rows of an [`Nvfp4Matrix`] round up to a multiple of this, which every entry point's
/// block height divides, so no load leaves the matrix.
const ROW_PAD: usize = 32;

/// A weight `[rows, k]` as packed E2M1 bytes `[padded_rows, k / 2]`, E4M3 block scales
/// `[padded_rows, k / 16]` and one F32 outer scale per padded row, in one device buffer.
/// Padding rows are zero.
#[derive(Debug)]
pub struct Nvfp4Matrix {
    data: CudaSlice<u8>,
    rows: usize,
    k: usize,
}

impl Nvfp4Matrix {
    /// Allocates a zero-initialized [`Nvfp4Matrix`] of `[rows, k]` (for incremental
    /// population via [`LinearNvfp4::quantize_indexed`]).
    pub fn alloc_zeros(stream: &Arc<CudaStream>, rows: usize, k: usize) -> anyhow::Result<Self> {
        ensure!(rows > 0, "weight has no rows");
        ensure!(
            k > 0 && k.is_multiple_of(32),
            "NVFP4 quantization requires k multiple of 32, got {k}"
        );
        let padded = rows.next_multiple_of(ROW_PAD);
        let values_bytes = padded * (k / 2);
        let scales_bytes = padded * (k / 16);
        let outer_bytes = 4 * padded;
        let data = stream.alloc_zeros::<u8>(values_bytes + scales_bytes + outer_bytes)?;
        Ok(Self { data, rows, k })
    }

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

    fn values_bytes(&self) -> usize {
        self.padded_rows() * (self.k / 2)
    }

    fn scales_bytes(&self) -> usize {
        self.padded_rows() * (self.k / 16)
    }

    /// The whole buffer: values, block scales and outer scales.
    #[must_use]
    pub fn buffer(&self) -> &CudaSlice<u8> {
        &self.data
    }

    /// Device bytes the matrix takes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.data.len()
    }

    /// The packed E2M1 values `[rows, k / 2]`, as bytes.
    #[must_use]
    pub fn values(&self) -> CudaView<'_, u8> {
        self.data.slice(..self.rows * (self.k / 2))
    }

    /// The E4M3 block scales `[rows, k / 16]`, as bytes.
    #[must_use]
    pub fn scales(&self) -> CudaView<'_, u8> {
        let start = self.values_bytes();
        self.data.slice(start..start + self.rows * (self.k / 16))
    }

    /// The F32 outer scales `[rows]`, as little-endian bytes.
    #[must_use]
    pub fn outer(&self) -> CudaView<'_, u8> {
        let start = self.values_bytes() + self.scales_bytes();
        self.data.slice(start..start + 4 * self.rows)
    }
}

/// One product `out = weight @ x` of a batch.
pub struct Nvfp4Gemv<'a> {
    weight: &'a Nvfp4Matrix,
    x: CudaView<'a, bf16>,
    out: CudaViewMut<'a, bf16>,
}

impl<'a> Nvfp4Gemv<'a> {
    /// `x` has `weight.k()` elements and `out` `weight.rows()`.
    pub fn new(
        weight: &'a Nvfp4Matrix,
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
    outer: u64,
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

/// One entry point, the `k` it serves, and the output rows one block covers.
#[derive(Debug)]
struct Entry {
    k: usize,
    rows_per_block: usize,
    function: CudaFunction,
}

#[derive(Debug)]
pub struct LinearNvfp4 {
    entries: Vec<Entry>,
    argmax_k2304: CudaFunction,
    quantize: CudaFunction,
    quantize_indexed: CudaFunction,
}

impl LinearNvfp4 {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "linear_nvfp4", cubin!("linear_nvfp4"))?;
        let entry = |k, rows_per_block, name: &str| -> anyhow::Result<Entry> {
            Ok(Entry {
                k,
                rows_per_block,
                function: module.load_function(name)?,
            })
        };
        Ok(Self {
            entries: vec![
                entry(2304, 16, "gemv_nvfp4_k2304")?,
                entry(3072, 8, "gemv_nvfp4_k3072")?,
            ],
            argmax_k2304: module.load_function("gemv_nvfp4_k2304_argmax")?,
            quantize: module.load_function("quantize_nvfp4_rows")?,
            quantize_indexed: module.load_function("quantize_nvfp4_rows_indexed")?,
        })
    }

    /// Whether [`Self::gemv`] takes products with input length `k`.
    #[must_use]
    pub fn supports(&self, k: usize) -> bool {
        self.entries.iter().any(|e| e.k == k)
    }

    /// Quantizes a BF16 weight `[rows, k]` to an [`Nvfp4Matrix`], checking that `k` is
    /// supported by [`Self::gemv`].
    pub fn quantize(
        &self,
        stream: &Arc<CudaStream>,
        weight: &impl DeviceIn<bf16>,
        rows: usize,
        k: usize,
    ) -> anyhow::Result<Nvfp4Matrix> {
        ensure!(
            self.supports(k),
            "no NVFP4 product for k = {k}; use quantize_any for other multiples of 32"
        );
        self.quantize_any(stream, weight, rows, k)
    }

    /// Quantizes a BF16 weight `[rows, k]` to an [`Nvfp4Matrix`] for any `k` that is a
    /// positive multiple of 32 (e.g. draft projections served by `LinearRows`).
    pub fn quantize_any(
        &self,
        stream: &Arc<CudaStream>,
        weight: &impl DeviceIn<bf16>,
        rows: usize,
        k: usize,
    ) -> anyhow::Result<Nvfp4Matrix> {
        let weight = weight.view();
        ensure!(rows > 0, "weight has no rows");
        ensure!(
            k > 0 && k.is_multiple_of(32),
            "NVFP4 quantization requires k multiple of 32, got {k}"
        );
        ensure!(
            weight.len() == rows * k,
            "weight has {} elements, expected {rows} x {k}",
            weight.len()
        );
        let mut mat = Nvfp4Matrix::alloc_zeros(stream, rows, k)?;
        let values_bytes = mat.values_bytes();
        let scales_bytes = mat.scales_bytes();
        {
            let (mut values, mut rest) = mat.data.split_at_mut(values_bytes);
            let (mut scales, mut outer) = rest.split_at_mut(scales_bytes);
            let config = LaunchConfig {
                grid_dim: (u32::try_from(rows.div_ceil(WARPS))?, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let rows_arg = i32::try_from(rows)?;
            let k_arg = i32::try_from(k)?;
            let mut launch = stream.launch_builder(&self.quantize);
            launch
                .arg(&weight)
                .arg(&rows_arg)
                .arg(&k_arg)
                .arg(&mut values)
                .arg(&mut scales)
                .arg(&mut outer);
            // SAFETY: one warp per row reads the row's k BF16 weights and writes k / 2
            // value bytes, k / 16 scale bytes and one F32 outer scale; the sections start
            // at multiples of 64 bytes, so all stores are aligned and in bounds.
            unsafe { launch.launch(config) }?;
        }
        Ok(mat)
    }

    /// Quantizes indexed rows `ids[0..ids.len()]` of BF16 `src` (`[_, dst.k()]`) into
    /// rows `dst_offset..dst_offset + ids.len()` of `dst` (for dynamic draft shortlists).
    pub fn quantize_indexed(
        &self,
        stream: &CudaStream,
        src: &impl DeviceIn<bf16>,
        ids: &CudaView<'_, i32>,
        dst_offset: usize,
        dst: &mut Nvfp4Matrix,
    ) -> anyhow::Result<()> {
        let src = src.view();
        let rows = ids.len();
        if rows == 0 {
            return Ok(());
        }
        let k = dst.k();
        ensure!(
            src.len().is_multiple_of(k),
            "source weight has {} elements, not whole rows of {k}",
            src.len()
        );
        ensure!(
            dst_offset + rows <= dst.rows(),
            "indexed range {}..{} exceeds destination rows {}",
            dst_offset,
            dst_offset + rows,
            dst.rows()
        );
        let values_bytes = dst.values_bytes();
        let scales_bytes = dst.scales_bytes();
        let (mut values, mut rest) = dst.data.split_at_mut(values_bytes);
        let (mut scales, mut outer) = rest.split_at_mut(scales_bytes);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows.div_ceil(WARPS))?, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let rows_arg = i32::try_from(rows)?;
        let k_arg = i32::try_from(k)?;
        let offset_arg = i32::try_from(dst_offset)?;
        let mut launch = stream.launch_builder(&self.quantize_indexed);
        launch
            .arg(&src)
            .arg(ids)
            .arg(&rows_arg)
            .arg(&k_arg)
            .arg(&offset_arg)
            .arg(&mut values)
            .arg(&mut scales)
            .arg(&mut outer);
        // SAFETY: one warp per indexed row reads `src[ids[r]]` and writes `dst_offset + r`
        // into `values`, `scales` and `outer`, validated in bounds above.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Runs every product in one launch: BF16 inputs and outputs, F32 accumulation, and
    /// each row's outer scale applied once before rounding to nearest even. All weights
    /// must share `k`, one [`Self::supports`].
    pub fn gemv(
        &self,
        stream: &CudaStream,
        products: &mut [Nvfp4Gemv<'_>],
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
            .with_context(|| format!("no NVFP4 product for k = {k}"))?;
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
            blocks += rows.div_ceil(entry.rows_per_block);
            let (w, _) = p.weight.data.device_ptr(stream);
            let (x, _) = p.x.device_ptr(stream);
            let (out, _) = p.out.device_ptr_mut(stream);
            ensure!(
                (w | x).is_multiple_of(16),
                "product {i}: weight and input must be 16-byte aligned"
            );
            let values_bytes = u64::try_from(p.weight.values_bytes())?;
            let scales_bytes = u64::try_from(p.weight.scales_bytes())?;
            batch.seg[i] = Segment {
                w,
                scale: w + values_bytes,
                outer: w + values_bytes + scales_bytes,
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
        // SAFETY: `batch` matches the kernel's `Batch` layout. Each matrix buffer holds
        // `padded_rows` rows of values, scales and outer scales, and a segment's blocks
        // cover at most its padded rows. The kernel stores only rows below `rows` and
        // reads `x` only after `pdl_wait`.
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

    /// Fused greedy `B = 1` LM head for `k = 2304`: computes each row's BF16-rounded logit
    /// over `0..vocab`, reduces per block into `(partial_vals, partial_idxs)`, and returns
    /// the number of partials written (`<= ARGMAX_MAX_BLOCKS`) for `Misc::argmax_partials`.
    ///
    /// Matches `LinearNvfp4::gemv -> logit_softcap -> argmax_wide_f32` bit for bit (0-ULP)
    /// for all logits in the strictly monotonic range of softcap.
    #[expect(clippy::too_many_arguments, reason = "weight, input, vocab and partials")]
    pub fn gemv_argmax(
        &self,
        stream: &CudaStream,
        weight: &Nvfp4Matrix,
        x: &impl DeviceIn<bf16>,
        vocab: usize,
        partial_vals: &mut impl DeviceOut<f32>,
        partial_idxs: &mut impl DeviceOut<i32>,
        start: Start,
    ) -> anyhow::Result<usize> {
        let (x, mut partial_vals, mut partial_idxs) =
            (x.view(), partial_vals.view_mut(), partial_idxs.view_mut());
        ensure!(
            weight.k == 2304,
            "gemv_argmax supports k = 2304, got {}",
            weight.k
        );
        ensure!(x.len() == 2304, "input has {} elements, expected 2304", x.len());
        ensure!(
            vocab > 0 && vocab <= weight.rows,
            "vocab {vocab} out of range 1..={}",
            weight.rows
        );
        let total_tiles = vocab.div_ceil(ARGMAX_TILE_ROWS);
        let blocks = total_tiles.min(ARGMAX_MAX_BLOCKS);
        ensure!(
            partial_vals.len() >= blocks && partial_idxs.len() >= blocks,
            "partial buffers ({}, {}) need at least {blocks} elements",
            partial_vals.len(),
            partial_idxs.len()
        );
        let (w, _) = weight.data.device_ptr(stream);
        let (x_ptr, _) = x.device_ptr(stream);
        ensure!(
            (w | x_ptr).is_multiple_of(16),
            "weight and input must be 16-byte aligned"
        );
        let values_bytes = u64::try_from(weight.values_bytes())?;
        let scales_bytes = u64::try_from(weight.scales_bytes())?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(blocks)?, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: u32::try_from(2304 * size_of::<bf16>())?,
        };
        let mut args = KernelArgs::default();
        args.u64(w)
            .u64(w + values_bytes)
            .u64(w + values_bytes + scales_bytes)
            .u64(x_ptr)
            .ptr_mut(&mut partial_vals, stream)
            .ptr_mut(&mut partial_idxs, stream)
            .i32(i32::try_from(vocab).context("vocab too large")?)
            .i32(i32::from(start == Start::Early));
        // SAFETY: arguments match `gemv_nvfp4_k2304_argmax(const u8*, const u8*, const float*,
        // const bf16*, float*, int*, int, int)`. `weight` has `padded_rows` (multiple of 32 >=
        // total_tiles * 16), partial buffers hold at least `blocks` entries, and only weights
        // are read before `pdl_wait`.
        unsafe {
            if start == Start::Early {
                args.launch_pdl(stream, &self.argmax_k2304, config)?;
            } else {
                args.launch(stream, &self.argmax_k2304, config)?;
            }
        }
        Ok(blocks)
    }
}

impl Prefetch for Nvfp4Matrix {
    fn device_bytes(&self, stream: &CudaStream) -> (u64, usize) {
        self.data.device_bytes(stream)
    }
}
