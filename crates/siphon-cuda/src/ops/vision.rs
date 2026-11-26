//! The vision tower's own kernels (`kernels/vision.cu`): embeddings, LayerNorm with the
//! residual update, the attention projections' bias and rotary embedding, bidirectional
//! attention over one image, and GELU. [`crate::vision::VisionTower`] runs them between
//! the BF16 or FP8 matrix products.
//!
//! Every row buffer is `[rows, 1024]`: the class token, then the image's patches.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr as _,
    LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{cubin, load_module};

/// Width of the tower, heads and head width, `kDim` and `kHeadDim` in the kernels.
pub const DIM: usize = 1024;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 64;
/// Side of the learned position-embedding grid, `kTable` in the kernels.
pub const POS_GRID: usize = 32;
/// Threads of the one-block-per-row kernels.
const ROW_THREADS: u32 = 256;
/// Query rows and threads of one attention block.
const ATTN_ROWS: usize = 64;
const ATTN_THREADS: u32 = 128;
/// Blocks of the grid-stride elementwise kernels.
const ELEMENTWISE_BLOCKS: u32 = 1024;
const ELEMENTWISE_THREADS: u32 = 256;

/// A null device pointer for an absent optional argument.
const NULL: u64 = 0;

#[derive(Debug)]
pub struct Vision {
    f32_to_bf16: CudaFunction,
    embed: CudaFunction,
    residual_norm: CudaFunction,
    qkv_rope: CudaFunction,
    attention: CudaFunction,
    gelu: CudaFunction,
}

/// The weights of [`Vision::embed`].
pub struct EmbedWeights<'a> {
    /// The resampled learned positions' table, `[32 * 32, 1024]`.
    pub table: &'a CudaSlice<bf16>,
    pub class: &'a CudaSlice<bf16>,
    pub class_position: &'a CudaSlice<bf16>,
    /// The pre-LayerNorm's weight and bias.
    pub norm_weight: &'a CudaSlice<bf16>,
    pub norm_bias: &'a CudaSlice<bf16>,
}

/// The residual update of [`Vision::residual_norm`]: `x += gamma * (delta + bias)`.
pub struct Residual<'a> {
    /// The product output, or `None` when it is in the rows of `out`.
    pub delta: Option<CudaView<'a, bf16>>,
    pub bias: &'a CudaSlice<bf16>,
    pub gamma: &'a CudaSlice<bf16>,
}

fn row_config(rows: usize) -> anyhow::Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
        block_dim: (ROW_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn elementwise_config(n: usize) -> LaunchConfig {
    let blocks = n
        .div_ceil(4 * ELEMENTWISE_THREADS as usize)
        .clamp(1, ELEMENTWISE_BLOCKS as usize);
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (ELEMENTWISE_THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Checks that a patch grid `[frames, height, width]` has even sides and `patches`
/// patches.
fn check_grid(grid: [usize; 3], patches: usize) -> anyhow::Result<(i32, i32)> {
    let [frames, h, w] = grid;
    ensure!(
        h > 0 && w > 0 && h.is_multiple_of(2) && w.is_multiple_of(2),
        "grid {h} x {w} must have positive even sides"
    );
    ensure!(
        frames * h * w == patches,
        "grid {grid:?} has {} patches, the input {patches}",
        frames * h * w
    );
    Ok((i32::try_from(h)?, i32::try_from(w)?))
}

impl Vision {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "vision", cubin!("vision"))?;
        let f = |name: &str| module.load_function(name);
        Ok(Self {
            f32_to_bf16: f("vision_f32_to_bf16")?,
            embed: f("vision_embed")?,
            residual_norm: f("vision_residual_norm")?,
            qkv_rope: f("vision_qkv_rope")?,
            attention: f("vision_attention")?,
            gelu: f("vision_gelu")?,
        })
    }

    /// Rounds `x` to BF16 into `out`, which has as many elements, a multiple of 4.
    pub fn f32_to_bf16(
        &self,
        stream: &CudaStream,
        x: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, bf16>,
    ) -> anyhow::Result<()> {
        let n = x.len();
        ensure!(
            out.len() == n && n.is_multiple_of(4),
            "x has {n} and out {} elements; they must match and be a multiple of 4",
            out.len()
        );
        let n_arg = i64::try_from(n)?;
        let mut launch = stream.launch_builder(&self.f32_to_bf16);
        launch.arg(x).arg(out).arg(&n_arg);
        // SAFETY: the arguments match `vision_f32_to_bf16(const float*, bf16*, long long)`;
        // both buffers hold n elements and every thread touches 4 at offsets below n.
        unsafe { launch.launch(elementwise_config(n)) }?;
        Ok(())
    }

    /// Writes `x` (F32 `[1 + patches, 1024]`) as the pre-LayerNorm of the embeddings: the
    /// class token in row 0, and in row r the patch embedding `patch_emb[r]` plus the
    /// position table resampled bilinearly to the patch's cell of `grid`. Row 0 of
    /// `patch_emb` is not read.
    pub fn embed(
        &self,
        stream: &CudaStream,
        patch_emb: &CudaView<'_, bf16>,
        weights: &EmbedWeights<'_>,
        grid: [usize; 3],
        eps: f32,
        x: &mut CudaViewMut<'_, f32>,
    ) -> anyhow::Result<()> {
        let rows = x.len() / DIM;
        ensure!(rows > 1 && x.len() == rows * DIM, "x is not [rows, {DIM}]");
        ensure!(patch_emb.len() == x.len(), "patch_emb is not shaped as x");
        ensure!(
            weights.table.len() == POS_GRID * POS_GRID * DIM,
            "the position table is not [{}, {DIM}]",
            POS_GRID * POS_GRID
        );
        for w in [
            weights.class,
            weights.class_position,
            weights.norm_weight,
            weights.norm_bias,
        ] {
            ensure!(w.len() == DIM, "an embedding vector is not [{DIM}]");
        }
        let (h, w) = check_grid(grid, rows - 1)?;
        let mut launch = stream.launch_builder(&self.embed);
        launch
            .arg(patch_emb)
            .arg(weights.table)
            .arg(weights.class)
            .arg(weights.class_position)
            .arg(weights.norm_weight)
            .arg(weights.norm_bias)
            .arg(&h)
            .arg(&w)
            .arg(&eps)
            .arg(x);
        // SAFETY: the arguments match `vision_embed`; one block per row reads its row of
        // patch_emb, at most four rows of the 32 x 32 table (taps clamped to the grid) and
        // the 1024-wide vectors, and writes its row of x.
        unsafe { launch.launch(row_config(rows)?) }?;
        Ok(())
    }

    /// For every row: applies `residual` to `x` if given, then writes
    /// `out = BF16(LayerNorm(x))` with `norm`'s weight and bias, or `out = BF16(x)` without
    /// it. `residual.delta` may hold the same rows as `out`.
    pub fn residual_norm(
        &self,
        stream: &CudaStream,
        x: &mut CudaViewMut<'_, f32>,
        residual: Option<&Residual<'_>>,
        norm: Option<(&CudaSlice<bf16>, &CudaSlice<bf16>)>,
        eps: f32,
        out: &mut CudaViewMut<'_, bf16>,
    ) -> anyhow::Result<()> {
        let rows = x.len() / DIM;
        ensure!(rows > 0 && x.len() == rows * DIM, "x is not [rows, {DIM}]");
        ensure!(out.len() == x.len(), "out is not shaped as x");
        // Event tracking is off (see `Gpu`), so the guard does nothing and drops at once.
        let out_ptr: u64 = out.device_ptr(stream).0;
        let mut launch = stream.launch_builder(&self.residual_norm);
        launch.arg(x);
        match residual {
            Some(r) => {
                ensure!(
                    r.bias.len() == DIM && r.gamma.len() == DIM,
                    "bias or gamma not [{DIM}]"
                );
                match &r.delta {
                    Some(delta) => {
                        ensure!(delta.len() == rows * DIM, "delta is not shaped as x");
                        launch.arg(delta);
                    }
                    None => {
                        launch.arg(&out_ptr);
                    }
                }
                launch.arg(r.bias).arg(r.gamma);
            }
            None => {
                launch.arg(&NULL).arg(&NULL).arg(&NULL);
            }
        }
        match norm {
            Some((w, b)) => {
                ensure!(w.len() == DIM && b.len() == DIM, "norm weights not [{DIM}]");
                launch.arg(w).arg(b);
            }
            None => {
                launch.arg(&NULL).arg(&NULL);
            }
        }
        launch.arg(&eps).arg(out);
        // SAFETY: the arguments match `vision_residual_norm`; one block per row touches its
        // row of x, delta and out and the 1024-wide vectors. Each thread reads its columns
        // of delta before writing the same columns of out, so the two may alias.
        unsafe { launch.launch(row_config(rows)?) }?;
        Ok(())
    }

    /// Adds the biases to the projections `q`, `k` and `v` (`[1 + patches, 1024]`) in
    /// place, rotates queries and keys by the 2D rotary embedding of each row's cell of
    /// `grid`, and scales the queries by 1/8.
    #[expect(clippy::many_single_char_names, reason = "q, k and v")]
    pub fn qkv_rope(
        &self,
        stream: &CudaStream,
        q: &mut CudaViewMut<'_, bf16>,
        k: &mut CudaViewMut<'_, bf16>,
        v: &mut CudaViewMut<'_, bf16>,
        biases: [&CudaSlice<bf16>; 3],
        grid: [usize; 3],
    ) -> anyhow::Result<()> {
        let rows = q.len() / DIM;
        ensure!(rows > 1 && q.len() == rows * DIM, "q is not [rows, {DIM}]");
        ensure!(
            k.len() == q.len() && v.len() == q.len(),
            "q, k and v differ in shape"
        );
        ensure!(
            biases.iter().all(|b| b.len() == DIM),
            "a bias is not [{DIM}]"
        );
        let (h, w) = check_grid(grid, rows - 1)?;
        let mut launch = stream.launch_builder(&self.qkv_rope);
        launch
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(biases[0])
            .arg(biases[1])
            .arg(biases[2])
            .arg(&h)
            .arg(&w);
        // SAFETY: the arguments match `vision_qkv_rope`; one block per row reads and writes
        // its row of q, k and v, and reads the biases.
        unsafe { launch.launch(row_config(rows)?) }?;
        Ok(())
    }

    /// Bidirectional attention of every row over every row, 16 heads of 64: `out` gets
    /// softmax(q k^T) v per head. `q` must already carry the 1/8 scale.
    pub fn attention(
        &self,
        stream: &CudaStream,
        q: &CudaView<'_, bf16>,
        k: &CudaView<'_, bf16>,
        v: &CudaView<'_, bf16>,
        out: &mut CudaViewMut<'_, bf16>,
    ) -> anyhow::Result<()> {
        let rows = q.len() / DIM;
        ensure!(rows > 0 && q.len() == rows * DIM, "q is not [rows, {DIM}]");
        ensure!(
            k.len() == q.len() && v.len() == q.len() && out.len() == q.len(),
            "q, k, v and out differ in shape"
        );
        let rows_arg = i32::try_from(rows)?;
        let mut launch = stream.launch_builder(&self.attention);
        launch.arg(q).arg(k).arg(v).arg(out).arg(&rows_arg);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows.div_ceil(ATTN_ROWS))?, HEADS as u32, 1),
            block_dim: (ATTN_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `vision_attention`; it reads rows below `rows` of q,
        // k and v (later key rows load as zeros) and writes rows below `rows` of out.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `h = BF16(gelu(h + bias))` with the erf GELU, over rows of `cols` columns, `bias`
    /// `[cols]` or none.
    pub fn gelu(
        &self,
        stream: &CudaStream,
        h: &mut CudaViewMut<'_, bf16>,
        bias: Option<&CudaSlice<bf16>>,
        cols: usize,
    ) -> anyhow::Result<()> {
        let n = h.len();
        ensure!(
            cols > 0 && cols.is_multiple_of(4) && n.is_multiple_of(cols),
            "h has {n} elements, not rows of {cols}, a multiple of 4"
        );
        let cols_arg = i32::try_from(cols)?;
        let n_arg = i64::try_from(n)?;
        let mut launch = stream.launch_builder(&self.gelu);
        launch.arg(h);
        match bias {
            Some(b) => {
                ensure!(b.len() == cols, "bias is not [{cols}]");
                launch.arg(b);
            }
            None => {
                launch.arg(&NULL);
            }
        }
        launch.arg(&cols_arg).arg(&n_arg);
        // SAFETY: the arguments match `vision_gelu`; every thread touches 4 elements below
        // n, which stay inside one row since cols is a multiple of 4, and their bias.
        unsafe { launch.launch(elementwise_config(n)) }?;
        Ok(())
    }
}
