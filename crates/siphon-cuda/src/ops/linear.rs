//! Matrix-vector and matrix-matrix products with BF16 weights (`kernels/linear.cu`).
//!
//! One launch runs a batch of products that share the input length `k`, each with its own
//! weight, input and output. The runner batches the projections of one layer input this
//! way, so a launch streams tens of megabytes instead of a few. Decode runs [`Gemv`]
//! products, one token at a time; prefill runs [`Gemm`] products over `m` rows at once on
//! tensor cores.
//!
//! [`Linear::prefetch`] pulls weights into L2 on a side stream while the main stream runs
//! latency-bound kernels, so a later product reads them from L2. Inside a graph capture the
//! side stream becomes a parallel branch of the same graph.

use std::{ffi::c_void, ptr, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, CudaViewMut,
    DevicePtr as _, DevicePtrMut as _, DeviceRepr, LaunchConfig, PushKernelArg as _,
    sys::{self, CUdevice_attribute, CUfunction_attribute},
};
use half::bf16;

use crate::{
    DeviceIn, DeviceOut,
    gpu::{cubin, load_module},
};

/// `k` must be a multiple of this: one 16-byte vector of BF16 weights.
const K_ALIGN: usize = 8;
/// Products one launch accepts, `kMaxSegments` in the kernel.
pub const MAX_PRODUCTS: usize = 8;
/// Weights one prefetch accepts, `kMaxRanges` in the kernel.
pub const MAX_PREFETCH: usize = 32;
/// Threads per block, `kThreads` in the kernel.
const THREADS: u32 = 256;
/// Largest `k * size_of::<T>()`: the input is staged in shared memory.
const MAX_X_BYTES: usize = 48 << 10;

/// One product `out = weight @ x` of a batch.
pub struct Gemv<'a, T> {
    weight: CudaView<'a, bf16>,
    x: CudaView<'a, T>,
    out: CudaViewMut<'a, T>,
}

impl<'a, T> Gemv<'a, T> {
    /// `weight` is `[out.len(), x.len()]` row-major.
    pub fn new(
        weight: &'a impl DeviceIn<bf16>,
        x: &'a impl DeviceIn<T>,
        out: &'a mut impl DeviceOut<T>,
    ) -> Self {
        Self {
            weight: weight.view(),
            x: x.view(),
            out: out.view_mut(),
        }
    }
}

/// One product `out = x @ weight^T` of a batch, over `m` rows.
pub struct Gemm<'a, T> {
    weight: CudaView<'a, bf16>,
    x: CudaView<'a, T>,
    out: CudaViewMut<'a, T>,
}

impl<'a, T> Gemm<'a, T> {
    /// `weight` is `[n, k]`, `x` is `[m, k]` and `out` is `[m, n]`, all row-major; the
    /// launch passes `m`.
    pub fn new(
        weight: &'a impl DeviceIn<bf16>,
        x: &'a impl DeviceIn<T>,
        out: &'a mut impl DeviceOut<T>,
    ) -> Self {
        Self {
            weight: weight.view(),
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
    k: i32,
    pdl: i32,
}

/// Kernel parameter: one product of a GEMM batch, as `gemm::Product` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GemmProduct {
    w: u64,
    x: u64,
    out: u64,
    n: i32,
    tile_end: i32,
}

/// Kernel parameter, as `gemm::Batch` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GemmBatch {
    p: [GemmProduct; MAX_PRODUCTS],
    count: i32,
    m: i32,
    k: i32,
    splits: i32,
}

/// Kernel parameter of `prefetch_l2`, as `Ranges` in the kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Ranges {
    p: [u64; MAX_PREFETCH],
    vecs: [i64; MAX_PREFETCH],
    count: i32,
}

// SAFETY: plain integers with the layout of the kernel's `Ranges`, passed by value.
unsafe impl DeviceRepr for Ranges {}

/// Below this many blocks of whole-warp rows, a batch splits each row over 8 warps.
/// The RTX 5090 has 170 SMs.
const SPLIT_BELOW_BLOCKS: usize = 128;

/// How a block divides its rows: one kernel entry point.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// `k <= 64`: 8 lanes per row.
    Narrow,
    /// One warp per row with this many loads in flight per lane.
    Wide(usize),
    /// 8 warps per row.
    Split,
}

impl Shape {
    /// The entry point for input length `k` and `rows` output rows in all. Exact fits keep
    /// a row's loads in one batch.
    fn new(k: usize, rows: usize) -> Self {
        let vecs = k / K_ALIGN;
        if vecs <= 8 {
            return Self::Narrow;
        }
        if rows.div_ceil(Self::Wide(0).rows_per_block()) < SPLIT_BELOW_BLOCKS {
            return Self::Split;
        }
        match vecs {
            288 => Self::Wide(9),
            384 => Self::Wide(12),
            _ => Self::Wide(8),
        }
    }

    fn rows_per_block(self) -> usize {
        match self {
            Self::Narrow => 32,
            Self::Wide(_) => 8,
            Self::Split => 1,
        }
    }
}

/// The entry points for one input and output type.
#[derive(Debug)]
struct Kernels {
    narrow: CudaFunction,
    /// `(loads per lane, kernel)` for the `k` values that fill one batch of loads.
    exact: Vec<(usize, CudaFunction)>,
    /// 8 loads per lane, for every other `k`.
    generic: CudaFunction,
    split: CudaFunction,
}

impl Kernels {
    fn get(&self, shape: Shape) -> &CudaFunction {
        match shape {
            Shape::Narrow => &self.narrow,
            Shape::Split => &self.split,
            Shape::Wide(v) => self
                .exact
                .iter()
                .find(|(loads, _)| *loads == v)
                .map_or(&self.generic, |(_, f)| f),
        }
    }
}

/// One GEMM entry point: the template arguments of its `GEMM_ENTRY` line in the kernel.
#[derive(Clone, Copy, Debug)]
struct GemmTile {
    name: &'static str,
    bm: usize,
    bn: usize,
    bk: usize,
    /// Warps along m, n and k.
    warps: [usize; 3],
    stages: usize,
}

impl GemmTile {
    fn threads(&self) -> usize {
        32 * self.warps.iter().product::<usize>()
    }

    /// `Tile::kSmemBytes` in the kernel, for input elements of `x_bytes` bytes: BF16 rows
    /// are swizzled, F32 rows padded by 8 elements.
    fn smem_bytes(&self, x_bytes: usize) -> usize {
        let x_row = if x_bytes == 2 {
            2 * self.bk
        } else {
            x_bytes * (self.bk + 8)
        };
        let stages = self.stages * (self.bm * x_row + self.bn * 2 * self.bk);
        let reduce = (self.warps[2] - 1).max(1) * self.bm * self.bn * 4;
        stages.max(reduce)
    }
}

const fn tile(
    name: &'static str,
    [bm, bn, bk]: [usize; 3],
    warps: [usize; 3],
    stages: usize,
) -> GemmTile {
    GemmTile {
        name,
        bm,
        bn,
        bk,
        warps,
        stages,
    }
}

/// The GEMM tiles of one input type.
struct GemmTiles {
    /// Tiles with the largest `m` each serves; the last serves every larger `m`.
    by_m: &'static [(usize, GemmTile)],
    /// Replaces the last tile of `by_m` when its tiles alone fill the GPU.
    wide: Option<GemmTile>,
}

/// Small tiles keep enough blocks streaming weights at small `m`. Measured on the RTX
/// 5090 over the runner's shapes; see `gemm_throughput` in `tests/linear.rs`.
const GEMM_BF16_TILES: GemmTiles = GemmTiles {
    by_m: &[
        (32, tile("gemm_bf16_16x64", [16, 64, 64], [1, 2, 2], 4)),
        (64, tile("gemm_bf16_32x64", [32, 64, 64], [1, 2, 2], 4)),
        (
            usize::MAX,
            tile("gemm_bf16_64x64", [64, 64, 64], [2, 2, 1], 3),
        ),
    ],
    wide: Some(tile("gemm_bf16_64x128", [64, 128, 64], [2, 2, 1], 2)),
};

const GEMM_F32_TILES: GemmTiles = GemmTiles {
    by_m: &[
        (64, tile("gemm_f32_16x64", [16, 64, 64], [1, 2, 2], 3)),
        (
            usize::MAX,
            tile("gemm_f32_64x64", [64, 64, 32], [2, 2, 1], 3),
        ),
    ],
    wide: None,
};

/// Most blocks of one tile's split, the portable cluster size limit.
const MAX_SPLITS: usize = 8;

/// One loaded GEMM entry point.
#[derive(Debug)]
struct GemmKernel {
    tile: GemmTile,
    function: CudaFunction,
    /// Blocks the whole GPU holds at once.
    slots: usize,
}

impl GemmKernel {
    fn load(
        module: &Arc<CudaModule>,
        tile: GemmTile,
        x_bytes: usize,
        sms: usize,
    ) -> anyhow::Result<Self> {
        let function = module.load_function(tile.name)?;
        let smem = tile.smem_bytes(x_bytes);
        function
            .set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(smem)?,
            )
            .with_context(|| format!("raising the shared memory of {}", tile.name))?;
        let per_sm = function.occupancy_max_active_blocks_per_multiprocessor(
            u32::try_from(tile.threads())?,
            smem,
            None,
        )?;
        ensure!(per_sm > 0, "{} does not fit on an SM", tile.name);
        Ok(Self {
            tile,
            function,
            slots: sms * usize::try_from(per_sm)?,
        })
    }

    /// Tiles of a batch over `m` rows with outputs of `ns` columns.
    fn tiles(&self, m: usize, ns: &[usize]) -> usize {
        m.div_ceil(self.tile.bm) * ns.iter().map(|n| n.div_ceil(self.tile.bn)).sum::<usize>()
    }
}

/// The loaded [`GemmTiles`] of one input type.
#[derive(Debug)]
struct GemmKernels {
    by_m: Vec<(usize, GemmKernel)>,
    wide: Option<GemmKernel>,
}

impl GemmKernels {
    fn load<T>(
        ctx: &Arc<CudaContext>,
        module: &Arc<CudaModule>,
        tiles: &GemmTiles,
    ) -> anyhow::Result<Self> {
        let sms = usize::try_from(
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        )?;
        let load = |tile| GemmKernel::load(module, tile, size_of::<T>(), sms);
        Ok(Self {
            by_m: tiles
                .by_m
                .iter()
                .map(|&(max_m, tile)| Ok((max_m, load(tile)?)))
                .collect::<anyhow::Result<_>>()?,
            wide: tiles.wide.map(load).transpose()?,
        })
    }

    fn get(&self, m: usize, ns: &[usize]) -> &GemmKernel {
        let last = self.by_m.len() - 1;
        let i = self
            .by_m
            .iter()
            .position(|(max_m, _)| m <= *max_m)
            .unwrap_or(last);
        match &self.wide {
            Some(wide) if i == last && wide.tiles(m, ns) >= wide.slots => wide,
            _ => &self.by_m[i].1,
        }
    }
}

/// A device buffer [`Linear::prefetch`] pulls into L2.
pub trait Prefetch {
    /// The buffer's device address and length in bytes. Event tracking is off (see
    /// [`crate::Gpu`]), so the address needs no guard.
    fn device_bytes(&self, stream: &CudaStream) -> (u64, usize);
}

impl<T: DeviceRepr> Prefetch for CudaSlice<T> {
    fn device_bytes(&self, stream: &CudaStream) -> (u64, usize) {
        (self.device_ptr(stream).0, self.len() * size_of::<T>())
    }
}

/// How [`Linear::prefetch`] pulls weights into L2.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pace {
    /// Real loads from [`PREFETCH_BLOCKS`] blocks, which slow down while other kernels
    /// load from DRAM.
    Loads,
    /// TMA bulk prefetches issued at a fixed rate in TB/s, which leave the SMs and the
    /// L2-to-SM bandwidth to the main stream but keep their rate whatever else reads DRAM.
    Rate(f32),
}

/// Blocks of the L2 prefetch kernel. A decode step took 3.98 ms with 80, 3.90 ms with 88
/// to 112 and 3.95 ms with 128: fewer fetch too slowly, more take SMs from the main stream.
/// Alone, 64 blocks stream 1.38 TB/s and 96 blocks 1.49 TB/s.
const PREFETCH_BLOCKS: u32 = 96;
/// One-warp blocks of the paced prefetch kernel: 16 issued too slowly at 1.6 TB/s (3.681
/// ms per BF16 step against 3.657); 32 and 64 were equal.
const PACED_BLOCKS: u32 = 32;

#[derive(Debug)]
pub struct Linear {
    bf16: Kernels,
    f32: Kernels,
    gemm_bf16: GemmKernels,
    gemm_f32: GemmKernels,
    prefetch_l2: CudaFunction,
    prefetch_l2_paced: CudaFunction,
    /// Runs prefetches beside the main stream.
    side: Arc<CudaStream>,
    fork: CudaEvent,
    join: CudaEvent,
}

impl Linear {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "linear", cubin!("linear"))?;
        // Thread-block clusters split k for the GEMM.
        ensure!(
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH)? != 0,
            "the device cannot launch thread-block clusters"
        );
        let f = |name: &str| module.load_function(name);
        Ok(Self {
            bf16: Kernels {
                narrow: f("gemv_bf16_l8_v1")?,
                exact: vec![(9, f("gemv_bf16_l32_v9")?), (12, f("gemv_bf16_l32_v12")?)],
                generic: f("gemv_bf16_l32_v8")?,
                split: f("gemv_bf16_l32_s8_v4")?,
            },
            f32: Kernels {
                narrow: f("gemv_f32_l8_v1")?,
                exact: Vec::new(),
                generic: f("gemv_f32_l32_v8")?,
                split: f("gemv_f32_l32_s8_v4")?,
            },
            gemm_bf16: GemmKernels::load::<bf16>(ctx, &module, &GEMM_BF16_TILES)?,
            gemm_f32: GemmKernels::load::<f32>(ctx, &module, &GEMM_F32_TILES)?,
            prefetch_l2: f("prefetch_l2")?,
            prefetch_l2_paced: f("prefetch_l2_paced")?,
            side: ctx.new_stream()?,
            fork: ctx.new_event(None)?,
            join: ctx.new_event(None)?,
        })
    }

    /// Starts pulling `weights` into L2, in order, on a side stream at `pace`, once the work
    /// enqueued on `stream` so far is done. Prefetches run one after another; a later
    /// [`Self::join_prefetch`] must rejoin them, also inside a graph capture. They only read,
    /// so they race with nothing. A paced prefetch ends once it has issued its requests,
    /// before the data arrives.
    pub fn prefetch(
        &self,
        stream: &CudaStream,
        weights: &[&dyn Prefetch],
        pace: Pace,
    ) -> anyhow::Result<()> {
        ensure!(
            (1..=MAX_PREFETCH).contains(&weights.len()),
            "{} weights to prefetch, expected 1 to {MAX_PREFETCH}",
            weights.len()
        );
        let mut ranges = Ranges {
            count: i32::try_from(weights.len())?,
            ..Ranges::default()
        };
        for (i, w) in weights.iter().enumerate() {
            let (ptr, bytes) = w.device_bytes(&self.side);
            ensure!(ptr.is_multiple_of(16), "weight {i} is not 16-byte aligned");
            ranges.p[i] = ptr;
            // A trailing partial vector stays out of L2, which costs one load later.
            ranges.vecs[i] = i64::try_from(bytes / 16)?;
        }
        self.fork.record(stream)?;
        self.side.wait(&self.fork)?;
        match pace {
            Pace::Loads => {
                let config = LaunchConfig {
                    grid_dim: (PREFETCH_BLOCKS, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut launch = self.side.launch_builder(&self.prefetch_l2);
                launch.arg(&ranges);
                // SAFETY: `ranges` has the kernel's `Ranges` layout, and the kernel reads
                // only the whole 16-byte vectors inside each weight.
                unsafe { launch.launch(config) }?;
            }
            Pace::Rate(tb_per_s) => {
                ensure!(tb_per_s > 0.0, "the prefetch rate must be positive");
                let config = LaunchConfig {
                    grid_dim: (PACED_BLOCKS, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                let bytes_per_ns = tb_per_s * 1000.0;
                let mut launch = self.side.launch_builder(&self.prefetch_l2_paced);
                launch.arg(&ranges).arg(&bytes_per_ns);
                // SAFETY: `ranges` has the kernel's `Ranges` layout, the kernel prefetches
                // only the whole 16-byte vectors inside each weight, and the rate is
                // positive.
                unsafe { launch.launch(config) }?;
            }
        }
        self.join.record(&self.side)?;
        Ok(())
    }

    /// Makes `stream` wait for the last [`Self::prefetch`].
    pub fn join_prefetch(&self, stream: &CudaStream) -> anyhow::Result<()> {
        Ok(stream.wait(&self.join)?)
    }

    /// Runs every product in one launch: BF16 inputs and outputs, F32 accumulation.
    ///
    /// All inputs have the same length `k`, a multiple of 8. Each output is rounded to
    /// nearest even from the F32 sum.
    pub fn gemv_bf16(
        &self,
        stream: &CudaStream,
        products: &mut [Gemv<'_, bf16>],
        start: Start,
    ) -> anyhow::Result<()> {
        launch(&self.bf16, stream, products, start)
    }

    /// Runs every product in one launch with F32 inputs and outputs. The BF16 weights widen
    /// exactly, so this matches an F32 product with the widened weights up to sum order.
    pub fn gemv_f32(
        &self,
        stream: &CudaStream,
        products: &mut [Gemv<'_, f32>],
        start: Start,
    ) -> anyhow::Result<()> {
        launch(&self.f32, stream, products, start)
    }

    /// Runs every product over `m` rows in one launch on tensor cores: BF16 inputs and
    /// outputs, F32 accumulation.
    ///
    /// All inputs have the same `k`, a multiple of 8, and every weight, input and output
    /// starts 16-byte aligned. Each output is rounded to nearest even from the F32 sum.
    /// A single row runs as [`Self::gemv_bf16`], which streams the weights faster.
    pub fn gemm_bf16(
        &self,
        stream: &CudaStream,
        products: &mut [Gemm<'_, bf16>],
        m: usize,
    ) -> anyhow::Result<()> {
        if m == 1 && fits_gemv::<bf16>(products) {
            return self.gemv_bf16(stream, &mut as_gemv(products), Start::Serial);
        }
        launch_gemm(&self.gemm_bf16, stream, products, m)
    }

    /// [`Self::gemm_bf16`] on tensor cores for every `m`, a single row included. Up to 16
    /// rows use the same tile and split, so each row sums in the same order for any `m`
    /// from 1 to 16.
    pub fn gemm_bf16_tiled(
        &self,
        stream: &CudaStream,
        products: &mut [Gemm<'_, bf16>],
        m: usize,
    ) -> anyhow::Result<()> {
        launch_gemm(&self.gemm_bf16, stream, products, m)
    }

    /// Runs every product over `m` rows in one launch with F32 inputs and outputs. Each
    /// input value runs as three BF16 parts holding about 24 of its bits, so this matches
    /// an F32 product with the widened weights up to sum order and about 2^-25 of each
    /// `|w x|`. It costs three times the tensor-core work of [`Self::gemm_bf16`]. A single
    /// row runs as [`Self::gemv_f32`].
    pub fn gemm_f32(
        &self,
        stream: &CudaStream,
        products: &mut [Gemm<'_, f32>],
        m: usize,
    ) -> anyhow::Result<()> {
        if m == 1 && fits_gemv::<f32>(products) {
            return self.gemv_f32(stream, &mut as_gemv(products), Start::Serial);
        }
        launch_gemm(&self.gemm_f32, stream, products, m)
    }
}

/// When a GEMV launch may start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// Once the previous kernel on the stream has finished.
    Serial,
    /// With programmatic dependent launch, once the previous kernel triggers. The GEMV
    /// issues its first weight loads, then waits for that kernel to finish before it
    /// reads its input. Either way the GEMV triggers once past its wait, so a next kernel
    /// launched with PDL may start early.
    ///
    /// In the BF16 decode step this saved 0.04 ms per step for the output projection,
    /// after the attention step. The projection batch at the start of a visit launched
    /// early cost 0.14 ms: its blocks take the SMs the visit's prefetch needs. With FP8
    /// weights the same early batch saved 0.15 ms. The LM head gained nothing either way.
    Early,
}

fn launch<T>(
    kernels: &Kernels,
    stream: &CudaStream,
    products: &mut [Gemv<'_, T>],
    start: Start,
) -> anyhow::Result<()> {
    ensure!(
        (1..=MAX_PRODUCTS).contains(&products.len()),
        "{} products, expected 1 to {MAX_PRODUCTS}",
        products.len()
    );
    let k = products[0].x.len();
    ensure!(
        k > 0 && k.is_multiple_of(K_ALIGN),
        "k is {k}, not a positive multiple of {K_ALIGN}"
    );
    let x_bytes = k * size_of::<T>();
    ensure!(x_bytes <= MAX_X_BYTES, "k is {k}, too long to stage");
    let rows = products.iter().map(|p| p.out.len()).sum();
    let shape = Shape::new(k, rows);
    let rows_per_block = shape.rows_per_block();

    let mut batch = Batch {
        count: i32::try_from(products.len())?,
        k: i32::try_from(k)?,
        pdl: i32::from(start == Start::Early),
        ..Batch::default()
    };
    let mut blocks = 0usize;
    // The guards order other streams after this launch; the runner uses one stream.
    let mut guards = Vec::with_capacity(3 * products.len());
    for (i, p) in products.iter_mut().enumerate() {
        let n = p.out.len();
        ensure!(n > 0, "product {i} has no output rows");
        ensure!(
            p.x.len() == k,
            "product {i} has k = {}, expected {k}",
            p.x.len()
        );
        ensure!(
            p.weight.len() == n * k,
            "product {i}: weight has {} elements, expected {n} x {k}",
            p.weight.len()
        );
        blocks += n.div_ceil(rows_per_block);
        let (w, w_guard) = p.weight.device_ptr(stream);
        let (x, x_guard) = p.x.device_ptr(stream);
        let (out, out_guard) = p.out.device_ptr_mut(stream);
        guards.extend([w_guard, x_guard, out_guard]);
        ensure!(
            (w | x).is_multiple_of(16),
            "product {i}: weight and input must be 16-byte aligned"
        );
        batch.seg[i] = Segment {
            w,
            x,
            out,
            rows: i32::try_from(n).context("too many rows")?,
            block_end: i32::try_from(blocks).context("too many rows")?,
        };
    }
    let config = LaunchConfig {
        grid_dim: (u32::try_from(blocks)?, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: u32::try_from(x_bytes)?,
    };
    // SAFETY: `batch` has the kernel's `Batch` layout. The checks above size each weight
    // as [rows, k], each input as [k] and each output as [rows]; the kernel reads only
    // rows below `rows` and columns below k, and before `pdl_wait` only weights. The views stay
    // borrowed until the launch is enqueued, and the stream orders the kernel before any
    // later use of the buffers.
    unsafe {
        launch_by_value(
            stream,
            kernels.get(shape),
            config,
            &batch,
            start == Start::Early,
        )
    }?;
    drop(guards);
    Ok(())
}

/// Launches `function`, whose only parameter is `batch` passed by value, with
/// programmatic dependent launch when `pdl` is set, so it may start before the previous
/// kernel on `stream` ends.
///
/// # Safety
///
/// `batch` must match the kernel's only parameter, every pointer in it must stay valid
/// until the kernel finishes, and with `pdl` the kernel must call `pdl_wait()` before it
/// reads anything an earlier kernel writes.
pub(crate) unsafe fn launch_by_value<B>(
    stream: &CudaStream,
    function: &CudaFunction,
    cfg: LaunchConfig,
    batch: &B,
    pdl: bool,
) -> anyhow::Result<()> {
    // SAFETY: zero is a valid bit pattern for the attribute value union.
    let mut value: sys::CUlaunchAttributeValue = unsafe { std::mem::zeroed() };
    value.programmaticStreamSerializationAllowed = i32::from(pdl);
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
    let mut params = [ptr::from_ref(batch).cast_mut().cast::<c_void>()];
    stream.context().bind_to_thread()?;
    // SAFETY: `params` points at `batch`, which outlives the call that copies it; the
    // caller vouches for its contents and for the kernel.
    unsafe {
        sys::cuLaunchKernelEx(
            &raw const config,
            function.cu_function(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        )
    }
    .result()?;
    Ok(())
}

/// Whether a one-row GEMM batch can run as GEMVs, which stage the input in shared memory.
fn fits_gemv<T>(products: &[Gemm<'_, T>]) -> bool {
    products
        .first()
        .is_some_and(|p| p.x.len() * size_of::<T>() <= MAX_X_BYTES)
}

/// The products of a one-row GEMM batch as GEMV products.
fn as_gemv<'b, T>(products: &'b mut [Gemm<'_, T>]) -> Vec<Gemv<'b, T>> {
    products
        .iter_mut()
        .map(|p| Gemv {
            weight: p.weight.slice(..),
            x: p.x.slice(..),
            out: p.out.slice_mut(..),
        })
        .collect()
}

fn launch_gemm<T: DeviceRepr>(
    kernels: &GemmKernels,
    stream: &CudaStream,
    products: &mut [Gemm<'_, T>],
    m: usize,
) -> anyhow::Result<()> {
    ensure!(
        (1..=MAX_PRODUCTS).contains(&products.len()),
        "{} products, expected 1 to {MAX_PRODUCTS}",
        products.len()
    );
    ensure!(m > 0, "m is 0");
    ensure!(
        products[0].x.len().is_multiple_of(m),
        "input 0 has {} elements, not a multiple of m = {m}",
        products[0].x.len()
    );
    let k = products[0].x.len() / m;
    ensure!(
        k > 0 && k.is_multiple_of(K_ALIGN),
        "k is {k}, not a positive multiple of {K_ALIGN}"
    );
    let ns: Vec<usize> = products.iter().map(|p| p.out.len() / m).collect();
    let kernel = kernels.get(m, &ns);
    let tile = &kernel.tile;
    let tiles_m = m.div_ceil(tile.bm);

    let mut batch = GemmBatch {
        count: i32::try_from(products.len())?,
        m: i32::try_from(m).context("m is too large")?,
        k: i32::try_from(k)?,
        ..GemmBatch::default()
    };
    let mut tiles = 0usize;
    // The guards order other streams after this launch; the runner uses one stream.
    let mut guards = Vec::with_capacity(3 * products.len());
    for (i, p) in products.iter_mut().enumerate() {
        ensure!(
            p.x.len() == m * k,
            "product {i}: input has {} elements, expected {m} x {k}",
            p.x.len()
        );
        ensure!(
            p.out.len().is_multiple_of(m) && !p.out.is_empty(),
            "product {i}: output has {} elements, not a positive multiple of m = {m}",
            p.out.len()
        );
        let n = p.out.len() / m;
        ensure!(
            p.weight.len() == n * k,
            "product {i}: weight has {} elements, expected {n} x {k}",
            p.weight.len()
        );
        tiles += tiles_m * n.div_ceil(tile.bn);
        let (w_ptr, w_guard) = p.weight.device_ptr(stream);
        let (x_ptr, x_guard) = p.x.device_ptr(stream);
        let (out, out_guard) = p.out.device_ptr_mut(stream);
        guards.extend([w_guard, x_guard, out_guard]);
        ensure!(
            (w_ptr | x_ptr | out).is_multiple_of(16),
            "product {i}: weight, input and output must be 16-byte aligned"
        );
        batch.p[i] = GemmProduct {
            w: w_ptr,
            x: x_ptr,
            out,
            n: i32::try_from(n).context("too many output columns")?,
            tile_end: i32::try_from(tiles).context("too many tiles")?,
        };
    }
    // Too few tiles to fill the GPU: split each tile's k steps over a cluster of blocks.
    let k_steps = k.div_ceil(tile.bk);
    let splits = (kernel.slots / tiles).clamp(1, MAX_SPLITS.min(k_steps));
    batch.splits = i32::try_from(splits)?;
    // SAFETY: the checks above size each weight as [n, k], each input as [m, k] and each
    // output as [m, n]; the kernel reads only rows below m and n and columns below k, and
    // writes only rows below m and columns below n. The views stay borrowed until the
    // launch is enqueued, and the stream orders the kernel before any later use of the
    // buffers.
    unsafe { launch_clusters(stream, kernel, &batch, size_of::<T>(), [tiles, splits])? };
    drop(guards);
    Ok(())
}

/// Launches `kernel` over `tiles` clusters of `splits` blocks, with inputs of `x_bytes`
/// bytes per element.
///
/// # Safety
///
/// Every pointer in `batch` must be valid for the rows and columns the kernel touches
/// until it finishes.
unsafe fn launch_clusters(
    stream: &CudaStream,
    kernel: &GemmKernel,
    batch: &GemmBatch,
    x_bytes: usize,
    [tiles, splits]: [usize; 2],
) -> anyhow::Result<()> {
    // SAFETY: zero is a valid bit pattern for the attribute value union.
    let mut value: sys::CUlaunchAttributeValue = unsafe { std::mem::zeroed() };
    value.clusterDim.x = u32::try_from(splits)?;
    value.clusterDim.y = 1;
    value.clusterDim.z = 1;
    let mut cluster = sys::CUlaunchAttribute {
        id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
        pad: [0; 4],
        value,
    };
    let config = sys::CUlaunchConfig {
        gridDimX: u32::try_from(tiles * splits)?,
        gridDimY: 1,
        gridDimZ: 1,
        blockDimX: u32::try_from(kernel.tile.threads())?,
        blockDimY: 1,
        blockDimZ: 1,
        sharedMemBytes: u32::try_from(kernel.tile.smem_bytes(x_bytes))?,
        hStream: stream.cu_stream(),
        attrs: &raw mut cluster,
        numAttrs: 1,
    };
    let mut params = [ptr::from_ref(batch).cast_mut().cast::<c_void>()];
    stream.context().bind_to_thread()?;
    // SAFETY: `batch` has the kernel's `gemm::Batch` layout and is its only parameter; it
    // outlives the call, which copies it. The grid is a whole number of clusters, and the
    // caller vouches for the pointers.
    unsafe {
        sys::cuLaunchKernelEx(
            &raw const config,
            kernel.function.cu_function(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        )
    }
    .result()?;
    Ok(())
}
