//! Dense products of 1 to 256 rows with BF16 or FP8 weights (`kernels/linear_rows.cu`).
//!
//! [`LinearRows::run`] computes `out [m, n] = x [m, k] @ weight [n, k]^T` for a batch of
//! products that share `k` and the weight format, streaming each weight from DRAM once for
//! all `m` rows. It serves the projections of a step of a few rows, such as a speculative
//! verify, with the checkpoint's BF16 weights or the FP8 copies of
//! [`crate::ops::linear_fp8::LinearFp8::quantize`].
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
    gpu::{cubin, load_module},
    ops::{
        linear::{Start, launch_by_value},
        linear_fp8::Fp8Matrix,
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
/// Shared memory per warp: `kSlots` slots of 16 rows x 256 bytes.
const WARP_SMEM: usize = 3 * TILE_ROWS * 256;
/// Bytes of each weight row one stage reads, `kStageBytes` in the kernel.
const STAGE_BYTES: usize = 256;
/// Warps a launch aims for, by weight format: tiles split their k range over warps until
/// the grid has this many. On one step's projections, 2048 beat 512 and 1024 with BF16
/// weights (1.15 ms per step for the input projections against 1.24 and 1.21), and 1024
/// beat 512 and 2048 with FP8 weights (0.68 ms against 0.74 and 0.71).
const TARGET_WARPS_BF16: usize = 2048;
const TARGET_WARPS_FP8: usize = 1024;

/// The weight of one product.
pub enum RowsWeight<'a> {
    /// BF16 `[n, k]`, row-major.
    Bf16(CudaView<'a, bf16>),
    /// E4M3 with one F32 scale per row.
    Fp8(&'a Fp8Matrix),
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

    fn is_fp8(&self) -> bool {
        matches!(self.weight, RowsWeight::Fp8(_))
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
    n: i32,
    tile_end: i32,
    pair_end: i32,
}

/// Kernel parameter, as `Batch` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Batch {
    seg: [Segment; MAX_PRODUCTS],
    count: i32,
    m: i32,
    pdl: i32,
}

/// One weight format and `k`: its entry point for each count of column tiles, 1 to
/// [`MAX_TILES`], the tile-pair entry points for 3 and 4 column tiles, and the chunked
/// 4-tile entry points for `m > 32`.
#[derive(Debug)]
struct Entry {
    fp8: bool,
    k: usize,
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
                (&self.chunk_pair, (chunks, pairs), 2 * WARP_SMEM)
            } else {
                (&self.chunk, (chunks, tiles), WARP_SMEM)
            }
        } else {
            let column_tiles = m.div_ceil(TILE_COLUMNS);
            if column_tiles >= PAIR_FROM_TILES && 2 * splits <= MAX_WARPS {
                (
                    &self.pairs[column_tiles - PAIR_FROM_TILES],
                    (pairs, 1),
                    2 * WARP_SMEM,
                )
            } else {
                (&self.functions[column_tiles - 1], (tiles, 1), WARP_SMEM)
            }
        }
    }
}

#[derive(Debug)]
pub struct LinearRows {
    entries: Vec<Entry>,
}

impl LinearRows {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "linear_rows", cubin!("linear_rows"))?;
        let smem = i32::try_from(MAX_WARPS * WARP_SMEM)?;
        let mut entries = Vec::new();
        for (fp8, k, name) in [
            (false, 2304, "rows_bf16_k2304"),
            (true, 2304, "rows_fp8_k2304"),
            (false, 3072, "rows_bf16_k3072"),
            (true, 3072, "rows_fp8_k3072"),
            (false, 4608, "rows_bf16_k4608"),
            (true, 4608, "rows_fp8_k4608"),
        ] {
            let load = |name: String| -> anyhow::Result<CudaFunction> {
                let function = module.load_function(&name)?;
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
            entries.push(Entry {
                fp8,
                k,
                functions,
                pairs,
                chunk,
                chunk_pair,
            });
        }
        Ok(Self { entries })
    }

    /// Whether [`Self::run`] takes products with input length `k`.
    #[must_use]
    pub fn supports(&self, k: usize) -> bool {
        self.entries.iter().any(|e| e.k == k)
    }

    /// Runs every product over `m` rows, 1 to [`MAX_ROWS`], in one launch: BF16 inputs and
    /// outputs, F32 accumulation, and with FP8 weights each row's scale applied once to its
    /// sum. Every output rounds to nearest even once.
    ///
    /// All products share `k`, one [`Self::supports`], and the weight format. Inputs and
    /// BF16 weights start 16-byte aligned.
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
        let fp8 = products[0].is_fp8();
        ensure!(
            products[0].x.len().is_multiple_of(m),
            "input 0 has {} elements, not whole rows of {m}",
            products[0].x.len()
        );
        let width = products[0].x.len() / m;
        let entry = self
            .entries
            .iter()
            .find(|e| e.fp8 == fp8 && e.k == width)
            .with_context(|| format!("no multi-row product for k = {width}"))?;

        let mut batch = Batch {
            count: i32::try_from(products.len())?,
            m: i32::try_from(m)?,
            pdl: i32::from(start == Start::Early),
            ..Batch::default()
        };
        let mut tiles = 0usize;
        let mut pairs = 0usize;
        for (i, p) in products.iter_mut().enumerate() {
            ensure!(p.is_fp8() == fp8, "product {i} has another weight format");
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
            // Event tracking is off (see `Gpu`), so the guards do nothing.
            let (w, scale) = match &p.weight {
                RowsWeight::Bf16(w) => {
                    ensure!(
                        w.len() == out_rows * width,
                        "product {i}: weight has {} elements, expected {out_rows} x {width}",
                        w.len()
                    );
                    (w.device_ptr(stream).0, 0)
                }
                RowsWeight::Fp8(w) => {
                    ensure!(
                        w.rows() == out_rows && w.k() == width,
                        "product {i}: weight is {} x {}, expected {out_rows} x {width}",
                        w.rows(),
                        w.k()
                    );
                    (
                        w.values().device_ptr(stream).0,
                        w.scales().device_ptr(stream).0,
                    )
                }
            };
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
        let stages = width * if fp8 { 1 } else { 2 } / STAGE_BYTES;
        let target = if fp8 {
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
}
