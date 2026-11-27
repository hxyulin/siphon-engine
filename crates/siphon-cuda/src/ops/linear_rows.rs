//! Dense products of 1 to 256 rows with BF16, FP8 or NVFP4 weights (`kernels/linear_rows.cu`
//! and `kernels/linear_nvfp4_rows.cu`).
//!
//! [`LinearRows::run`] computes `out [m, n] = x [m, k] @ weight [n, k]^T` for a batch of
//! products that share `k` and the weight format, streaming each weight from DRAM once for
//! all `m` rows. It serves the projections of a step of a few rows (speculative verify,
//! batched decode, DFlash 2 draft) with BF16 weights, FP8 copies, or NVFP4 copies.
//!
//! Each output row depends on its input row only, bit for bit: it is the same for every
//! `m` from 1 to 256, whatever the other rows hold. The sums run on tensor cores in F32, in
//! an order that differs from the decode GEMVs' and the prefill GEMM's.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, CudaView, CudaViewMut, DevicePtr as _,
    DevicePtrMut as _, LaunchConfig, sys::CUfunction_attribute,
};
use half::bf16;

use crate::{
    DeviceIn, DeviceOut,
    gpu::{KernelArgs, cubin, load_module},
    ops::{
        linear::{Start, launch_by_value},
        linear_fp8::{Fp8Matrix, Mxfp8Matrix},
        linear_nvfp4::Nvfp4Matrix,
    },
};

/// Products one launch accepts, `kMaxSegments` in the kernel.
pub const MAX_PRODUCTS: usize = 8;
/// Rows of one column tile: the 8 columns of an `m16n8k16` product.
const TILE_COLUMNS: usize = 8;
/// Most column tiles one block runs, `NT` of the kernel's entry points.
const MAX_TILES: usize = 4;
/// Rows one block covers at most (`TILE_COLUMNS * MAX_TILES`).
const CHUNK_ROWS: usize = TILE_COLUMNS * MAX_TILES;
/// Most rows one launch takes (chunks of [`CHUNK_ROWS`] along `gridDim.x` above 32).
pub const MAX_ROWS: usize = 256;
/// Weight rows per block, `kTileRows` in the kernel.
const TILE_ROWS: usize = 16;
/// Most warps per block, `kMaxWarps` in the kernel.
const MAX_WARPS: usize = 8;
/// Shared memory per warp for BF16/FP8: `kSlots` slots of 16 rows x 256 bytes.
const WARP_SMEM: usize = 3 * TILE_ROWS * 256;
/// Shared memory per warp for NVFP4: `kSlots` slots of 16 rows x (128 value + 16 scale bytes).
const NVFP4_WARP_SMEM: usize = 3 * TILE_ROWS * (128 + 16);
/// Bytes of each BF16/FP8 weight row one stage reads, `kStageBytes` in `linear_rows.cu`.
const STAGE_BYTES: usize = 256;
/// Elements of `k` one NVFP4 stage reads, `kKPerStage` in `linear_nvfp4_rows.cu`.
const NVFP4_STAGE_K: usize = 256;
/// Warps a launch aims for, by weight format: tiles split their k range over warps until
/// the grid has this many. On one step's projections, 2048 beat 512 and 1024 with BF16
/// weights (1.15 ms per step for the input projections against 1.24 and 1.21), and 1024
/// beat 512 and 2048 with FP8 weights (0.68 ms against 0.74 and 0.71).
const TARGET_WARPS_BF16: usize = if cfg!(sm89) { 512 } else { 2048 };
const TARGET_WARPS_FP8: usize = if cfg!(sm89) { 512 } else { 1024 };
const TARGET_WARPS_NVFP4: usize = if cfg!(sm89) { 512 } else { 1024 };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WeightKind {
    Bf16,
    Fp8,
    Mxfp8,
    Nvfp4,
}

/// The weight of one product.
pub enum RowsWeight<'a> {
    /// BF16 `[n, k]`, row-major.
    Bf16(CudaView<'a, bf16>),
    /// E4M3 with one F32 scale per row.
    Fp8(&'a Fp8Matrix),
    /// E4M3 values `[rows, k]` and F32 scales `[rows]` in separate buffers (e.g. draft weights).
    Fp8Raw {
        values: CudaView<'a, u8>,
        scales: CudaView<'a, f32>,
        rows: usize,
        k: usize,
    },
    /// E4M3 with one UE8M0 scale per 32 weights.
    Mxfp8(&'a Mxfp8Matrix),
    /// NVFP4 (E2M1 + E4M3 block scales per 16 + F32 outer scale per row).
    Nvfp4(&'a Nvfp4Matrix),
}

impl RowsWeight<'_> {
    fn kind(&self) -> WeightKind {
        match self {
            Self::Bf16(_) => WeightKind::Bf16,
            Self::Fp8(_) | Self::Fp8Raw { .. } => WeightKind::Fp8,
            Self::Mxfp8(_) => WeightKind::Mxfp8,
            Self::Nvfp4(_) => WeightKind::Nvfp4,
        }
    }
}

/// One product `out = x @ weight^T` of a batch, over `m` rows.
pub struct RowsProduct<'a> {
    weight: RowsWeight<'a>,
    x: CudaView<'a, bf16>,
    out: CudaViewMut<'a, bf16>,
}

impl<'a> RowsProduct<'a> {
    /// `weight` is `[n, k]`, `x` `[m, k]` and `out` `[m, n]`, all row-major; the launch
    /// passes `m`.
    pub fn bf16(
        weight: &'a impl DeviceIn<bf16>,
        x: &'a impl DeviceIn<bf16>,
        out: &'a mut impl DeviceOut<bf16>,
    ) -> Self {
        Self {
            weight: RowsWeight::Bf16(weight.view()),
            x: x.view(),
            out: out.view_mut(),
        }
    }

    /// As [`Self::bf16`] with an FP8 weight of `weight.rows()` rows of `weight.k()`.
    pub fn fp8(
        weight: &'a Fp8Matrix,
        x: &'a impl DeviceIn<bf16>,
        out: &'a mut impl DeviceOut<bf16>,
    ) -> Self {
        Self {
            weight: RowsWeight::Fp8(weight),
            x: x.view(),
            out: out.view_mut(),
        }
    }

    /// As [`Self::fp8`] with separate E4M3 values `[rows, k]` and F32 scales `[rows]`.
    pub fn fp8_raw(
        values: CudaView<'a, u8>,
        scales: CudaView<'a, f32>,
        rows: usize,
        k: usize,
        x: &'a impl DeviceIn<bf16>,
        out: &'a mut impl DeviceOut<bf16>,
    ) -> Self {
        Self {
            weight: RowsWeight::Fp8Raw {
                values,
                scales,
                rows,
                k,
            },
            x: x.view(),
            out: out.view_mut(),
        }
    }

    /// As [`Self::bf16`] with an MXFP8 weight of `weight.rows()` rows of `weight.k()`.
    pub fn mxfp8(
        weight: &'a Mxfp8Matrix,
        x: &'a impl DeviceIn<bf16>,
        out: &'a mut impl DeviceOut<bf16>,
    ) -> Self {
        Self {
            weight: RowsWeight::Mxfp8(weight),
            x: x.view(),
            out: out.view_mut(),
        }
    }

    /// As [`Self::bf16`] with an NVFP4 weight of `weight.rows()` rows of `weight.k()`.
    pub fn nvfp4(
        weight: &'a Nvfp4Matrix,
        x: &'a impl DeviceIn<bf16>,
        out: &'a mut impl DeviceOut<bf16>,
    ) -> Self {
        Self {
            weight: RowsWeight::Nvfp4(weight),
            x: x.view(),
            out: out.view_mut(),
        }
    }
}

/// Kernel parameter: one segment per BF16/FP8 product, as `Segment` in `linear_rows.cu`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Segment {
    w: u64,
    scale: u64,
    x: u64,
    out: u64,
    n: i32,
    tile_end: i32,
    pair_end: i32,
}

/// Kernel parameter, as `Batch` in `linear_rows.cu`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Batch {
    seg: [Segment; MAX_PRODUCTS],
    count: i32,
    m: i32,
    pdl: i32,
}

/// Kernel parameter: one segment per NVFP4 product, as `Segment` in `linear_nvfp4_rows.cu`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Nvfp4Segment {
    w: u64,
    scale: u64,
    outer: u64,
    x: u64,
    out: u64,
    n: i32,
    tile_end: i32,
    pair_end: i32,
    _pad: i32,
}

/// Kernel parameter, as `Batch` in `linear_nvfp4_rows.cu`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Nvfp4Batch {
    seg: [Nvfp4Segment; MAX_PRODUCTS],
    count: i32,
    m: i32,
    pdl: i32,
}

/// One weight format and `k`: its entry point for each count of column tiles, 1 to
/// [`MAX_TILES`], the tile-pair entry points for 3 and 4 column tiles, and the chunked
/// 4-tile entry points for `m > 32`.
#[derive(Debug)]
struct Entry {
    kind: WeightKind,
    k: usize,
    warp_smem: usize,
    functions: Vec<CudaFunction>,
    pairs: Vec<CudaFunction>,
    chunk: CudaFunction,
    chunk_pair: CudaFunction,
}

/// Fewest column tiles that run blocks of two weight tiles. Each block reads its inputs
/// once for both tiles, which halves the input bytes per weight byte: at 3 and 4 column
/// tiles those are 3 and 4 times the FP8 weight bytes and come from L2.
const PAIR_FROM_TILES: usize = 3;

impl Entry {
    /// The entry point, grid `(x, y)` and shared memory per warp for `m` rows: single-chunk
    /// entry points up to [`CHUNK_ROWS`], and 32-row chunks along `gridDim.x` above so
    /// sibling chunks share each weight tile in L2.
    fn shape(
        &self,
        m: usize,
        splits: usize,
        tiles: usize,
        pairs: usize,
    ) -> (&CudaFunction, (usize, usize), usize) {
        if m > CHUNK_ROWS {
            let chunks = m.div_ceil(CHUNK_ROWS);
            if 2 * splits <= MAX_WARPS {
                (&self.chunk_pair, (chunks, pairs), 2 * self.warp_smem)
            } else {
                (&self.chunk, (chunks, tiles), self.warp_smem)
            }
        } else {
            let column_tiles = m.div_ceil(TILE_COLUMNS);
            if column_tiles >= PAIR_FROM_TILES && 2 * splits <= MAX_WARPS {
                (
                    &self.pairs[column_tiles - PAIR_FROM_TILES],
                    (pairs, 1),
                    2 * self.warp_smem,
                )
            } else {
                (
                    &self.functions[column_tiles - 1],
                    (tiles, 1),
                    self.warp_smem,
                )
            }
        }
    }
}

#[derive(Debug)]
pub struct LinearRows {
    entries: Vec<Entry>,
    argmax_entries: Vec<Entry>,
}

/// Validates `weight` against `width` and returns `(w_ptr, scale_ptr, outer_ptr, weight_rows)`.
fn resolve_weight(
    stream: &CudaStream,
    weight: &RowsWeight<'_>,
    width: usize,
) -> anyhow::Result<(u64, u64, u64, usize)> {
    match weight {
        RowsWeight::Bf16(w) => {
            ensure!(
                w.len().is_multiple_of(width),
                "weight has {} elements, not whole rows of {width}",
                w.len()
            );
            Ok((w.device_ptr(stream).0, 0u64, 0u64, w.len() / width))
        }
        RowsWeight::Fp8(w) => {
            ensure!(w.k() == width, "weight has k = {}, expected {width}", w.k());
            Ok((
                w.values().device_ptr(stream).0,
                w.scales().device_ptr(stream).0,
                0u64,
                w.rows(),
            ))
        }
        RowsWeight::Fp8Raw {
            values,
            scales,
            rows,
            k,
        } => {
            ensure!(*k == width, "weight has k = {k}, expected {width}");
            ensure!(
                values.len() >= *rows * width && scales.len() >= *rows,
                "FP8 raw buffers too short for {rows} x {width}"
            );
            Ok((
                values.device_ptr(stream).0,
                scales.device_ptr(stream).0,
                0u64,
                *rows,
            ))
        }
        RowsWeight::Mxfp8(w) => {
            ensure!(w.k() == width, "weight has k = {}, expected {width}", w.k());
            Ok((
                w.values().device_ptr(stream).0,
                w.scales().device_ptr(stream).0,
                0u64,
                w.rows(),
            ))
        }
        RowsWeight::Nvfp4(w) => {
            ensure!(w.k() == width, "weight has k = {}, expected {width}", w.k());
            Ok((
                w.values().device_ptr(stream).0,
                w.scales().device_ptr(stream).0,
                w.outer().device_ptr(stream).0,
                w.rows(),
            ))
        }
    }
}

impl LinearRows {
    #[expect(
        clippy::too_many_lines,
        reason = "loads BF16, FP8 and NVFP4 multi-row and argmax kernels"
    )]
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "linear_rows", cubin!("linear_rows"))?;
        let nvfp4_module = load_module(ctx, "linear_nvfp4_rows", cubin!("linear_nvfp4_rows"))?;
        let load_entry = |m: &Arc<cudarc::driver::CudaModule>,
                          kind: WeightKind,
                          k: usize,
                          warp_smem: usize,
                          name: &str|
         -> anyhow::Result<Entry> {
            let smem = i32::try_from(MAX_WARPS * warp_smem)?;
            let load = |name: String| -> anyhow::Result<CudaFunction> {
                let function = m.load_function(&name)?;
                function
                    .set_attribute(
                        CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        smem,
                    )
                    .with_context(|| format!("raising the shared memory of {name}"))?;
                Ok(function)
            };
            let functions = (1..=MAX_TILES)
                .map(|tiles| {
                    load(if tiles == 1 {
                        name.to_owned()
                    } else {
                        format!("{name}_{tiles}")
                    })
                })
                .collect::<anyhow::Result<_>>()?;
            let pairs = (PAIR_FROM_TILES..=MAX_TILES)
                .map(|tiles| load(format!("{name}_{tiles}_pair")))
                .collect::<anyhow::Result<_>>()?;
            let chunk = load(format!("{name}_{MAX_TILES}_chunk"))?;
            let chunk_pair = load(format!("{name}_{MAX_TILES}_pair_chunk"))?;
            Ok(Entry {
                kind,
                k,
                warp_smem,
                functions,
                pairs,
                chunk,
                chunk_pair,
            })
        };
        let mut entries = Vec::new();
        for (kind, k, name) in [
            (WeightKind::Bf16, 2304, "rows_bf16_k2304"),
            (WeightKind::Fp8, 2304, "rows_fp8_k2304"),
            (WeightKind::Bf16, 3072, "rows_bf16_k3072"),
            (WeightKind::Fp8, 3072, "rows_fp8_k3072"),
            (WeightKind::Mxfp8, 3072, "rows_mxfp8_k3072"),
            (WeightKind::Bf16, 4096, "rows_bf16_k4096"),
            (WeightKind::Fp8, 4096, "rows_fp8_k4096"),
            (WeightKind::Bf16, 4608, "rows_bf16_k4608"),
            (WeightKind::Fp8, 4608, "rows_fp8_k4608"),
            (WeightKind::Bf16, 6912, "rows_bf16_k6912"),
            (WeightKind::Fp8, 6912, "rows_fp8_k6912"),
            (WeightKind::Bf16, 9216, "rows_bf16_k9216"),
            (WeightKind::Fp8, 9216, "rows_fp8_k9216"),
        ] {
            entries.push(load_entry(&module, kind, k, WARP_SMEM, name)?);
        }
        for (k, name) in [
            (2304, "rows_nvfp4_k2304"),
            (3072, "rows_nvfp4_k3072"),
            (4096, "rows_nvfp4_k4096"),
            (4608, "rows_nvfp4_k4608"),
            (6912, "rows_nvfp4_k6912"),
            (9216, "rows_nvfp4_k9216"),
        ] {
            entries.push(load_entry(
                &nvfp4_module,
                WeightKind::Nvfp4,
                k,
                NVFP4_WARP_SMEM,
                name,
            )?);
        }
        let argmax_entries = vec![
            load_entry(
                &module,
                WeightKind::Bf16,
                2304,
                WARP_SMEM,
                "rows_bf16_k2304_argmax",
            )?,
            load_entry(
                &module,
                WeightKind::Fp8,
                2304,
                WARP_SMEM,
                "rows_fp8_k2304_argmax",
            )?,
            load_entry(
                &nvfp4_module,
                WeightKind::Nvfp4,
                2304,
                NVFP4_WARP_SMEM,
                "rows_nvfp4_k2304_argmax",
            )?,
        ];
        Ok(Self {
            entries,
            argmax_entries,
        })
    }

    /// Whether [`Self::run`] takes products with input length `k`.
    #[must_use]
    pub fn supports(&self, k: usize) -> bool {
        self.entries.iter().any(|e| e.k == k)
    }

    /// Runs every product over `m` rows, 1 to [`MAX_ROWS`], in one launch: BF16 inputs and
    /// outputs, F32 accumulation, and with FP8/NVFP4 weights each row's scale applied once
    /// to its sum. Every output rounds to nearest even once.
    ///
    /// All products share `k`, one [`Self::supports`], and the weight format. Inputs and
    /// weights start 16-byte aligned.
    pub fn run(
        &self,
        stream: &CudaStream,
        products: &mut [RowsProduct<'_>],
        m: usize,
        start: Start,
    ) -> anyhow::Result<()> {
        ensure!(
            (1..=MAX_PRODUCTS).contains(&products.len()),
            "{} products, expected 1 to {MAX_PRODUCTS}",
            products.len()
        );
        ensure!(
            (1..=MAX_ROWS).contains(&m),
            "m is {m}, expected 1 to {MAX_ROWS}"
        );
        let kind = products[0].weight.kind();
        ensure!(
            products[0].x.len().is_multiple_of(m),
            "input 0 has {} elements, not whole rows of {m}",
            products[0].x.len()
        );
        let width = products[0].x.len() / m;
        let entry = self
            .entries
            .iter()
            .find(|e| e.kind == kind && e.k == width)
            .with_context(|| format!("no multi-row {kind:?} product for k = {width}"))?;

        if kind == WeightKind::Nvfp4 {
            return Self::run_nvfp4(stream, entry, products, m, width, start);
        }

        let mut batch = Batch {
            count: i32::try_from(products.len())?,
            m: i32::try_from(m)?,
            pdl: i32::from(start == Start::Early),
            ..Batch::default()
        };
        let mut tiles = 0usize;
        let mut pairs = 0usize;
        for (i, p) in products.iter_mut().enumerate() {
            ensure!(
                p.weight.kind() == kind,
                "product {i} has another weight format"
            );
            ensure!(
                p.x.len() == m * width,
                "product {i}: input has {} elements, expected {m} x {width}",
                p.x.len()
            );
            ensure!(
                !p.out.is_empty() && p.out.len().is_multiple_of(m),
                "product {i}: output has {} elements, not whole rows of {m}",
                p.out.len()
            );
            let out_rows = p.out.len() / m;
            let (w, scale, _, weight_rows) = resolve_weight(stream, &p.weight, width)?;
            ensure!(
                weight_rows == out_rows,
                "product {i}: weight has {weight_rows} rows, expected {out_rows}"
            );
            let x = p.x.device_ptr(stream).0;
            let out = p.out.device_ptr_mut(stream).0;
            ensure!(
                (w | x).is_multiple_of(16),
                "product {i}: weight and input must be 16-byte aligned"
            );
            tiles += out_rows.div_ceil(TILE_ROWS);
            pairs += out_rows.div_ceil(TILE_ROWS).div_ceil(2);
            batch.seg[i] = Segment {
                w,
                scale,
                x,
                out,
                n: i32::try_from(out_rows).context("too many output rows")?,
                tile_end: i32::try_from(tiles).context("too many tiles")?,
                pair_end: i32::try_from(pairs).context("too many tiles")?,
            };
        }
        // The split depends on the shapes only, never on m, so each row's sum order is the
        // same for every m.
        let byte_weight = kind != WeightKind::Bf16;
        let stages = width * if byte_weight { 1 } else { 2 } / STAGE_BYTES;
        let target = if byte_weight {
            TARGET_WARPS_FP8
        } else {
            TARGET_WARPS_BF16
        };
        let splits = target.div_ceil(tiles).clamp(1, MAX_WARPS.min(stages));
        let (function, (grid_x, grid_y), per_warp) = entry.shape(m, splits, tiles, pairs);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(grid_x)?, u32::try_from(grid_y)?, 1),
            block_dim: (u32::try_from(32 * splits)?, 1, 1),
            shared_mem_bytes: u32::try_from(splits * per_warp)?,
        };
        // SAFETY: `batch` has the kernel's `Batch` layout. The checks above size each weight
        // as [n, k] (an FP8 weight's values and scales come from its matrix), each input as
        // [m, k] and each output as [m, n]. The kernel reads weight rows below n (it clamps
        // the last tile), input rows below m and columns below k, writes only rows below m
        // and columns below n, and before `pdl_wait` reads only weights. The stream orders
        // the kernel before any later use of the buffers.
        unsafe { launch_by_value(stream, function, config, &batch, start == Start::Early) }
    }

    fn run_nvfp4(
        stream: &CudaStream,
        entry: &Entry,
        products: &mut [RowsProduct<'_>],
        m: usize,
        width: usize,
        start: Start,
    ) -> anyhow::Result<()> {
        let mut batch = Nvfp4Batch {
            count: i32::try_from(products.len())?,
            m: i32::try_from(m)?,
            pdl: i32::from(start == Start::Early),
            ..Nvfp4Batch::default()
        };
        let mut tiles = 0usize;
        let mut pairs = 0usize;
        for (i, p) in products.iter_mut().enumerate() {
            ensure!(
                p.weight.kind() == WeightKind::Nvfp4,
                "product {i} has another weight format"
            );
            ensure!(
                p.x.len() == m * width,
                "product {i}: input has {} elements, expected {m} x {width}",
                p.x.len()
            );
            ensure!(
                !p.out.is_empty() && p.out.len().is_multiple_of(m),
                "product {i}: output has {} elements, not whole rows of {m}",
                p.out.len()
            );
            let out_rows = p.out.len() / m;
            let (w, scale, outer, weight_rows) = resolve_weight(stream, &p.weight, width)?;
            ensure!(
                weight_rows == out_rows,
                "product {i}: weight has {weight_rows} rows, expected {out_rows}"
            );
            let x = p.x.device_ptr(stream).0;
            let out = p.out.device_ptr_mut(stream).0;
            ensure!(
                (w | scale | x).is_multiple_of(16),
                "product {i}: weight, scales and input must be 16-byte aligned"
            );
            tiles += out_rows.div_ceil(TILE_ROWS);
            pairs += out_rows.div_ceil(TILE_ROWS).div_ceil(2);
            batch.seg[i] = Nvfp4Segment {
                w,
                scale,
                outer,
                x,
                out,
                n: i32::try_from(out_rows).context("too many output rows")?,
                tile_end: i32::try_from(tiles).context("too many tiles")?,
                pair_end: i32::try_from(pairs).context("too many tiles")?,
                _pad: 0,
            };
        }
        let stages = width / NVFP4_STAGE_K;
        let splits = TARGET_WARPS_NVFP4
            .div_ceil(tiles)
            .clamp(1, MAX_WARPS.min(stages));
        let (function, (grid_x, grid_y), per_warp) = entry.shape(m, splits, tiles, pairs);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(grid_x)?, u32::try_from(grid_y)?, 1),
            block_dim: (u32::try_from(32 * splits)?, 1, 1),
            shared_mem_bytes: u32::try_from(splits * per_warp)?,
        };
        // SAFETY: `batch` has `linear_nvfp4_rows.cu`'s `Batch` layout, buffers are validated
        // above, and before `pdl_wait` only read-only weight values and block scales are read.
        unsafe { launch_by_value(stream, function, config, &batch, start == Start::Early) }
    }

    /// Single-product LM head with a fused tile-argmax epilogue: computes `x @ weight^T`
    /// (BF16-rounded per element, masked to `0..vocab`) and writes one `(max_val, max_idx)`
    /// partial per `(row, tile)` into `partial_vals` and `partial_idxs`. Returns the number
    /// of partials written per row (the row stride for `Misc::argmax_partials`).
    ///
    /// Matches `LinearRows::run -> logit_softcap_bf16_f32 -> argmax_f32` bit for bit (0-ULP)
    /// for all logits in the strictly monotonic range of softcap, including tie-breaking and
    /// NaN handling.
    #[expect(
        clippy::too_many_arguments,
        reason = "weight, input, vocab and partials"
    )]
    pub fn run_argmax(
        &self,
        stream: &CudaStream,
        weight: &RowsWeight<'_>,
        x: &impl DeviceIn<bf16>,
        vocab: usize,
        partial_vals: &mut impl DeviceOut<f32>,
        partial_idxs: &mut impl DeviceOut<i32>,
        m: usize,
        start: Start,
    ) -> anyhow::Result<usize> {
        let (x, mut partial_vals, mut partial_idxs) =
            (x.view(), partial_vals.view_mut(), partial_idxs.view_mut());
        ensure!(
            (1..=MAX_ROWS).contains(&m),
            "m is {m}, expected 1 to {MAX_ROWS}"
        );
        ensure!(vocab > 0, "vocab must be positive");
        ensure!(
            x.len().is_multiple_of(m),
            "input has {} elements, not whole rows of {m}",
            x.len()
        );
        let width = x.len() / m;
        let kind = weight.kind();
        ensure!(
            kind != WeightKind::Mxfp8,
            "fused argmax does not take MXFP8 weights"
        );
        let entry = self
            .argmax_entries
            .iter()
            .find(|e| e.kind == kind && e.k == width)
            .with_context(|| {
                format!("no fused argmax multi-row {kind:?} product for k = {width}")
            })?;

        let (w_ptr, scale_ptr, outer_ptr, weight_rows) = resolve_weight(stream, weight, width)?;
        ensure!(
            vocab <= weight_rows,
            "vocab {vocab} exceeds weight rows {weight_rows}"
        );
        let x_ptr = x.device_ptr(stream).0;
        ensure!(
            (w_ptr | x_ptr).is_multiple_of(16),
            "weight and input must be 16-byte aligned"
        );
        if kind == WeightKind::Nvfp4 {
            ensure!(
                scale_ptr.is_multiple_of(16),
                "NVFP4 scales must be 16-byte aligned"
            );
        }

        // Compute `splits` from `weight_rows` (padded vocab) so warp reduction order matches
        // `LinearRows::run` bit for bit, while launching tiles only up to `vocab`.
        let padded_tiles = weight_rows.div_ceil(TILE_ROWS);
        let (stages, target) = match kind {
            WeightKind::Bf16 => (width * 2 / STAGE_BYTES, TARGET_WARPS_BF16),
            WeightKind::Fp8 => (width / STAGE_BYTES, TARGET_WARPS_FP8),
            WeightKind::Nvfp4 => (width / NVFP4_STAGE_K, TARGET_WARPS_NVFP4),
            WeightKind::Mxfp8 => unreachable!("rejected above"),
        };
        let splits = target
            .div_ceil(padded_tiles)
            .clamp(1, MAX_WARPS.min(stages));
        let vocab_tiles = vocab.div_ceil(TILE_ROWS);
        let vocab_pairs = vocab_tiles.div_ceil(2);
        let (function, (grid_x, grid_y), per_warp) =
            entry.shape(m, splits, vocab_tiles, vocab_pairs);
        let partial_stride = if m > CHUNK_ROWS { grid_y } else { grid_x };
        ensure!(
            partial_vals.len() >= m * partial_stride && partial_idxs.len() >= m * partial_stride,
            "partial buffers ({}, {}) need at least {} elements ({m} x {partial_stride})",
            partial_vals.len(),
            partial_idxs.len(),
            m * partial_stride
        );
        let config = LaunchConfig {
            grid_dim: (u32::try_from(grid_x)?, u32::try_from(grid_y)?, 1),
            block_dim: (u32::try_from(32 * splits)?, 1, 1),
            shared_mem_bytes: u32::try_from(splits * per_warp)?,
        };
        let mut args = KernelArgs::default();
        args.u64(w_ptr).u64(scale_ptr);
        if kind == WeightKind::Nvfp4 {
            args.u64(outer_ptr);
        }
        args.u64(x_ptr)
            .ptr_mut(&mut partial_vals, stream)
            .ptr_mut(&mut partial_idxs, stream)
            .i32(i32::try_from(m)?)
            .i32(i32::try_from(vocab).context("vocab too large")?)
            .i32(i32::try_from(partial_stride).context("partial_stride too large")?)
            .i32(i32::from(start == Start::Early));
        // SAFETY: arguments match `rows_{bf16,fp8}_k2304_argmax*` (9 args) or
        // `rows_nvfp4_k2304_argmax*` (10 args with `outer`). Buffers are validated above,
        // the kernel reads only read-only weights/scales before `pdl_wait`, and writes only
        // `[0..m, 0..partial_stride)`.
        unsafe {
            if start == Start::Early {
                args.launch_pdl(stream, function, config)?;
            } else {
                args.launch(stream, function, config)?;
            }
        }
        Ok(partial_stride)
    }
}
