//! MoE expert routing and depth attention (`kernels/routing.cu`).

use std::{mem::size_of, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{cubin, load_module};

/// Hidden width of the model.
const HIDDEN: usize = 2304;
/// Routed experts.
pub const EXPERTS: usize = 128;
/// Experts chosen per token.
pub const TOP_K: usize = 8;
/// Threads per block of `router_logits_2304`, which fixes its summation order.
const LOGITS_THREADS: u32 = 256;
/// Threads per block of `expert_routing_128_top8`: one warp per row.
const ROUTING_THREADS: u32 = 256;
/// Threads per row of `depth_attention_2304`, which fixes its summation order.
const DEPTH_THREADS: u32 = 288;

/// How `Routing::router_logits` treats the F32 router weight.
///
/// MLX on the M5 reference machine computes an F32 matmul with one row in a GEMV at full
/// precision, but with two or more rows in a GEMM that truncates both operands' mantissas
/// to TF32's 10 bits. Matching MLX means picking `Tf32` exactly when MLX had more than one
/// row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterMatmul {
    /// Full F32 products, summed in the order of MLX's GEMV.
    F32,
    /// Weight mantissas truncated to 10 bits. The hidden states are BF16, which TF32 holds
    /// exactly.
    Tf32,
}

#[derive(Debug)]
pub struct Routing {
    router_logits_2304: CudaFunction,
    expert_routing_128_top8: CudaFunction,
    depth_attention_2304: CudaFunction,
}

impl Routing {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "routing", cubin!("routing"))?;
        Ok(Self {
            router_logits_2304: module.load_function("router_logits_2304")?,
            expert_routing_128_top8: module.load_function("expert_routing_128_top8")?,
            depth_attention_2304: module.load_function("depth_attention_2304")?,
        })
    }

    /// Router logits `F32(hidden) @ weight.T` for BF16 `hidden` (`[rows, 2304]`) and the
    /// F32 router weight (`[128, 2304]`) into `logits` (`[rows, 128]`).
    pub fn router_logits(
        &self,
        stream: &CudaStream,
        hidden: &CudaSlice<bf16>,
        weight: &CudaSlice<f32>,
        logits: &mut CudaSlice<f32>,
        matmul: RouterMatmul,
    ) -> anyhow::Result<()> {
        ensure!(
            weight.len() == EXPERTS * HIDDEN,
            "weight has {} elements, expected {EXPERTS} x {HIDDEN}",
            weight.len()
        );
        ensure!(
            hidden.len().is_multiple_of(HIDDEN),
            "hidden has {} elements, not a multiple of {HIDDEN}",
            hidden.len()
        );
        let rows = hidden.len() / HIDDEN;
        ensure!(
            logits.len() == rows * EXPERTS,
            "logits has {} elements, expected {rows} x {EXPERTS}",
            logits.len()
        );
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(rows).context("too many rows")?,
                EXPERTS as u32,
                1,
            ),
            block_dim: (LOGITS_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let truncate_tf32 = i32::from(matmul == RouterMatmul::Tf32);
        let mut launch = stream.launch_builder(&self.router_logits_2304);
        launch
            .arg(hidden)
            .arg(weight)
            .arg(logits)
            .arg(&truncate_tf32);
        // SAFETY: the arguments match `router_logits_2304(const bf16*, const float*, float*,
        // int)`, and the checks above keep every row inside `hidden` and `logits`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Top-8 experts of each row of `logits` (`[rows, 128]`) by sigmoid score plus `bias`
    /// (`[128]`), into `indices` and `weights` (both `[rows, 8]`). The weights are the
    /// unbiased scores over their sum plus 1e-20, times `route_scale`. Experts come in
    /// MLX's order: highest biased score first, ties to the lower index.
    pub fn expert_routing(
        &self,
        stream: &CudaStream,
        logits: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        indices: &mut CudaSlice<i32>,
        weights: &mut CudaSlice<f32>,
        route_scale: f32,
    ) -> anyhow::Result<()> {
        ensure!(
            bias.len() == EXPERTS,
            "bias has {} elements, expected {EXPERTS}",
            bias.len()
        );
        ensure!(
            logits.len().is_multiple_of(EXPERTS),
            "logits has {} elements, not a multiple of {EXPERTS}",
            logits.len()
        );
        let rows = logits.len() / EXPERTS;
        ensure!(
            indices.len() == rows * TOP_K && weights.len() == rows * TOP_K,
            "indices has {} and weights {} elements, expected {rows} x {TOP_K}",
            indices.len(),
            weights.len()
        );
        let rows = i32::try_from(rows).context("too many rows")?;
        let rows_per_block = ROUTING_THREADS / 32;
        let config = LaunchConfig {
            grid_dim: (rows.unsigned_abs().div_ceil(rows_per_block), 1, 1),
            block_dim: (ROUTING_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.expert_routing_128_top8);
        launch
            .arg(logits)
            .arg(bias)
            .arg(indices)
            .arg(weights)
            .arg(&rows)
            .arg(&route_scale);
        // SAFETY: the arguments match `expert_routing_128_top8(const float*, const float*,
        // int32_t*, float*, int, float)`, the kernel skips warps past `rows`, and the checks
        // above keep every row inside the buffers.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Depth attention: for each row, the `nq` F32 `queries` (`[nq, 2304]`) attend over
    /// the `nb + 1` candidate states `blocks` (`[nb, rows, 2304]`) and `partial`
    /// (`[rows, 2304]`) with scores scaled by `2304^-0.5`, into `out` (`[rows, nq, 2304]`).
    pub fn depth_attention(
        &self,
        stream: &CudaStream,
        blocks: &CudaSlice<f32>,
        partial: &CudaSlice<f32>,
        queries: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        ensure!(
            !partial.is_empty() && partial.len().is_multiple_of(HIDDEN),
            "partial has {} elements, not a positive multiple of {HIDDEN}",
            partial.len()
        );
        ensure!(
            blocks.len().is_multiple_of(partial.len()),
            "blocks has {} elements, not a multiple of partial's {}",
            blocks.len(),
            partial.len()
        );
        ensure!(
            !queries.is_empty() && queries.len().is_multiple_of(HIDDEN),
            "queries has {} elements, not a positive multiple of {HIDDEN}",
            queries.len()
        );
        let rows = partial.len() / HIDDEN;
        let nb = blocks.len() / partial.len();
        let nq = queries.len() / HIDDEN;
        ensure!(
            out.len() == rows * nq * HIDDEN,
            "out has {} elements, expected {rows} x {nq} x {HIDDEN}",
            out.len()
        );
        // Metal computes the scale in F64 and passes it as F32; 1 / 48 rounds the same way.
        let scale = (HIDDEN as f32).sqrt().recip();
        let shared_floats = (nb + 1) * nq * (DEPTH_THREADS as usize / 32 + 1);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: (DEPTH_THREADS, 1, 1),
            shared_mem_bytes: u32::try_from(shared_floats * size_of::<f32>())
                .context("too many candidates")?,
        };
        let nb = i32::try_from(nb).context("too many blocks")?;
        let nq = i32::try_from(nq).context("too many queries")?;
        let mut launch = stream.launch_builder(&self.depth_attention_2304);
        launch
            .arg(blocks)
            .arg(partial)
            .arg(queries)
            .arg(out)
            .arg(&nb)
            .arg(&nq)
            .arg(&scale);
        // SAFETY: the arguments match `depth_attention_2304(const float*, const float*,
        // const float*, float*, int, int, float)`, the checks above keep every row inside the
        // buffers, and the dynamic shared memory holds the (nb + 1) * nq * 10 floats the
        // kernel uses.
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}
