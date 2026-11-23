//! Matrix-vector products with BF16 weights (`kernels/linear.cu`).
//!
//! One launch runs a batch of products that share the input length `k`, each with its own
//! weight, input and output. The runner batches the projections of one layer input this
//! way, so a launch streams tens of megabytes instead of a few.
//!
//! [`Linear::prefetch`] pulls weights into L2 on a side stream while the main stream runs
//! latency-bound kernels, so a later product reads them from L2. Inside a graph capture the
//! side stream becomes a parallel branch of the same graph.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut,
    DevicePtr as _, DevicePtrMut as _, DeviceRepr, LaunchConfig, PushKernelArg as _,
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
pub const MAX_PREFETCH: usize = 24;
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
}

// SAFETY: plain integers with the layout of the kernel's `Batch`, passed by value.
unsafe impl DeviceRepr for Batch {}

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

/// Blocks of the L2 prefetch kernel. In the decode step 64 beat 32, 48, 80, 96, 128 and 170:
/// fewer fetch too slowly, more take SMs and bandwidth from the main stream.
const PREFETCH_BLOCKS: u32 = 64;

#[derive(Debug)]
pub struct Linear {
    bf16: Kernels,
    f32: Kernels,
    prefetch_l2: CudaFunction,
    /// Runs prefetches beside the main stream.
    side: Arc<CudaStream>,
    fork: CudaEvent,
    join: CudaEvent,
}

impl Linear {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "linear", cubin!("linear"))?;
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
            prefetch_l2: f("prefetch_l2")?,
            side: ctx.new_stream()?,
            fork: ctx.new_event(None)?,
            join: ctx.new_event(None)?,
        })
    }

    /// Starts pulling `weights` into L2, in order, on a side stream, once the work enqueued
    /// on `stream` so far is done. Prefetches run one after another; a later
    /// [`Self::join_prefetch`] must rejoin them, also inside a graph capture. They only read,
    /// so they race with nothing.
    pub fn prefetch(
        &self,
        stream: &CudaStream,
        weights: &[&CudaSlice<bf16>],
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
        let mut guards = Vec::with_capacity(weights.len());
        for (i, w) in weights.iter().enumerate() {
            let (ptr, guard) = w.device_ptr(&self.side);
            guards.push(guard);
            ensure!(ptr.is_multiple_of(16), "weight {i} is not 16-byte aligned");
            ranges.p[i] = ptr;
            // A trailing partial vector stays out of L2, which costs one load later.
            ranges.vecs[i] = i64::try_from(w.len() * size_of::<bf16>() / 16)?;
        }
        self.fork.record(stream)?;
        self.side.wait(&self.fork)?;
        let config = LaunchConfig {
            grid_dim: (PREFETCH_BLOCKS, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = self.side.launch_builder(&self.prefetch_l2);
        launch.arg(&ranges);
        // SAFETY: `ranges` has the kernel's `Ranges` layout, and the kernel reads only the
        // whole 16-byte vectors inside each weight.
        unsafe { launch.launch(config) }?;
        drop(guards);
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
    ) -> anyhow::Result<()> {
        launch(&self.bf16, stream, products)
    }

    /// Runs every product in one launch with F32 inputs and outputs. The BF16 weights widen
    /// exactly, so this matches an F32 product with the widened weights up to sum order.
    pub fn gemv_f32(
        &self,
        stream: &CudaStream,
        products: &mut [Gemv<'_, f32>],
    ) -> anyhow::Result<()> {
        launch(&self.f32, stream, products)
    }
}

fn launch<T: DeviceRepr>(
    kernels: &Kernels,
    stream: &CudaStream,
    products: &mut [Gemv<'_, T>],
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
    let mut launch = stream.launch_builder(kernels.get(shape));
    launch.arg(&batch);
    // SAFETY: `batch` has the kernel's `Batch` layout. The checks above size each weight
    // as [rows, k], each input as [k] and each output as [rows]; the kernel reads only
    // rows below `rows` and columns below k. The views stay borrowed until the launch is
    // enqueued, and the stream orders the kernel before any later use of the buffers.
    unsafe { launch.launch(config) }?;
    drop(guards);
    Ok(())
}
