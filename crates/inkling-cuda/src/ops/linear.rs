//! Dense BF16 and F32 matrix-vector products (`kernels/linear.cu`).

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{cubin, load_module};

/// `k` must be a multiple of this.
const K_ALIGN: usize = 64;
/// Most input rows one launch accepts: the grid's y limit.
const MAX_ROWS: usize = 65_535;

/// The three blockings MLX's matrix-vector kernel uses, as `(SN, BN)` in `kernels/linear.cu`.
#[derive(Clone, Copy)]
enum Blocking {
    /// `k <= 64`: 4 lanes per output row, 8 rows per warp.
    Narrow,
    /// `k >= 16 * n`: 8 warps split `k` for one output row.
    SplitK,
    /// Otherwise: one warp per output row.
    Wide,
}

impl Blocking {
    /// MLX's choice for an `[n, k]` matrix.
    fn for_shape(n: usize, k: usize) -> Self {
        if k <= 64 {
            Self::Narrow
        } else if k >= 16 * n {
            Self::SplitK
        } else {
            Self::Wide
        }
    }

    /// `(output rows per block, threads per block)`, matching the kernel's layout.
    fn block(self) -> (u32, u32) {
        match self {
            Self::Narrow => (32, 128),
            Self::SplitK => (1, 256),
            Self::Wide => (4, 128),
        }
    }
}

/// The kernels for one dtype, one per [`Blocking`].
#[derive(Debug)]
struct Kernels {
    narrow: CudaFunction,
    split_k: CudaFunction,
    wide: CudaFunction,
}

impl Kernels {
    fn get(&self, blocking: Blocking) -> &CudaFunction {
        match blocking {
            Blocking::Narrow => &self.narrow,
            Blocking::SplitK => &self.split_k,
            Blocking::Wide => &self.wide,
        }
    }
}

#[derive(Debug)]
pub struct Linear {
    bf16: Kernels,
    f32: Kernels,
}

impl Linear {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "linear", cubin!("linear"))?;
        let kernels = |dtype: &str| -> anyhow::Result<Kernels> {
            Ok(Kernels {
                narrow: module.load_function(&format!("gemv_{dtype}_sn4_bn1"))?,
                split_k: module.load_function(&format!("gemv_{dtype}_sn32_bn8"))?,
                wide: module.load_function(&format!("gemv_{dtype}_sn32_bn1"))?,
            })
        };
        Ok(Self {
            bf16: kernels("bf16")?,
            f32: kernels("f32")?,
        })
    }

    /// `out = x @ weight.T` in BF16 with F32 accumulation, as `nn.Linear` without bias.
    ///
    /// `weight` is `[n, k]` row-major, `x` is `[rows, k]` and `out` is `[rows, n]`, all
    /// contiguous. Each row is computed as its own M = 1 product, bit-exact with MLX's
    /// matrix-vector kernel. `k` must be a multiple of 64.
    pub fn gemv_bf16(
        &self,
        stream: &CudaStream,
        weight: &CudaSlice<bf16>,
        x: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        n: usize,
        k: usize,
    ) -> anyhow::Result<()> {
        launch(&self.bf16, stream, weight, x, out, n, k)
    }

    /// `out = x @ weight.T` in F32, with the same layouts and limits as [`Self::gemv_bf16`].
    ///
    /// Matches MLX's M = 1 F32 path, which keeps full F32 precision; MLX rounds inputs to
    /// TF32 only when it multiplies two or more rows at once.
    pub fn gemv_f32(
        &self,
        stream: &CudaStream,
        weight: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        k: usize,
    ) -> anyhow::Result<()> {
        launch(&self.f32, stream, weight, x, out, n, k)
    }
}

fn launch<T: DeviceRepr>(
    kernels: &Kernels,
    stream: &CudaStream,
    weight: &CudaSlice<T>,
    x: &CudaSlice<T>,
    out: &mut CudaSlice<T>,
    n: usize,
    k: usize,
) -> anyhow::Result<()> {
    ensure!(n > 0, "n must be positive");
    ensure!(
        k > 0 && k.is_multiple_of(K_ALIGN),
        "k is {k}, not a positive multiple of {K_ALIGN}"
    );
    ensure!(
        weight.len() == n * k,
        "weight has {} elements, expected {n} x {k}",
        weight.len()
    );
    ensure!(
        x.len().is_multiple_of(k),
        "x has {} elements, not a multiple of {k}",
        x.len()
    );
    let rows = x.len() / k;
    ensure!(
        (1..=MAX_ROWS).contains(&rows),
        "x has {rows} rows, expected 1 to {MAX_ROWS}"
    );
    ensure!(
        out.len() == rows * n,
        "out has {} elements, expected {rows} x {n}",
        out.len()
    );
    let n_arg = i32::try_from(n).context("n too large")?;
    let k_arg = i32::try_from(k).context("k too large")?;
    let blocking = Blocking::for_shape(n, k);
    let (rows_per_block, threads) = blocking.block();
    let config = LaunchConfig {
        grid_dim: (
            u32::try_from(n.div_ceil(rows_per_block as usize)).context("n too large")?,
            u32::try_from(rows).context("too many rows")?,
            1,
        ),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(kernels.get(blocking));
    launch.arg(weight).arg(x).arg(out).arg(&n_arg).arg(&k_arg);
    // SAFETY: the arguments match `gemv_*(const T*, const T*, T*, int, int)` with T the
    // element type of the slices. The checks above size `weight` as [n, k], `x` as [rows, k]
    // and `out` as [rows, n]; the kernel reads only rows below n and columns below k.
    unsafe { launch.launch(config) }?;
    Ok(())
}
