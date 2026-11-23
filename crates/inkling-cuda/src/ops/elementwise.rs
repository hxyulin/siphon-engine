//! Embedding gather, activations, gates, residual adds and the logit softcap
//! (`kernels/elementwise.cu`).

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{cubin, load_module};

/// Hidden width of the model.
const HIDDEN: usize = 2304;
/// Elements each embedding thread copies with one 16-byte access.
const EMBEDDING_VEC: usize = 8;
/// Width of one expert's activation, half the interleaved gate/up width.
const EXPERT_WIDTH: usize = 1152;
/// Query heads times head dimension: the width of the attention output and its gate.
const ATTENTION_WIDTH: usize = 24 * 128;
/// Experts each token is routed to.
const TOP_K: usize = 8;
/// LM head width, padded past the tokenizer's vocabulary.
const PADDED_VOCAB: usize = 201_024;
/// Logits the model returns per row; the padding columns are dropped.
const VOCAB: usize = 200_058;
/// Threads per block for the elementwise kernels.
const BLOCK: u32 = 256;
/// Upper bound on blocks for the grid-stride kernels.
const MAX_BLOCKS: usize = 1 << 16;

#[derive(Debug)]
pub struct Elementwise {
    embedding_bf16_2304: CudaFunction,
    softcapped_swiglu_bf16: CudaFunction,
    attention_gate_bf16: CudaFunction,
    log_scale_bf16: CudaFunction,
    add_bf16_bf16_f32: CudaFunction,
    add_f32_bf16_bf16: CudaFunction,
    moe_combine_bf16: CudaFunction,
    logit_softcap_bf16_f32: CudaFunction,
}

/// Launch shape for a grid-stride loop over `n` elements.
fn grid_stride(n: usize) -> LaunchConfig {
    let blocks = n.div_ceil(BLOCK as usize).clamp(1, MAX_BLOCKS);
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Fails unless `len` is a whole number of `width`-element rows.
fn ensure_rows(name: &str, len: usize, width: usize) -> anyhow::Result<usize> {
    ensure!(
        len.is_multiple_of(width),
        "{name} has {len} elements, not a multiple of {width}"
    );
    Ok(len / width)
}

impl Elementwise {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "elementwise", cubin!("elementwise"))?;
        Ok(Self {
            embedding_bf16_2304: module.load_function("embedding_bf16_2304")?,
            softcapped_swiglu_bf16: module.load_function("softcapped_swiglu_bf16")?,
            attention_gate_bf16: module.load_function("attention_gate_bf16")?,
            log_scale_bf16: module.load_function("log_scale_bf16")?,
            add_bf16_bf16_f32: module.load_function("add_bf16_bf16_f32")?,
            add_f32_bf16_bf16: module.load_function("add_f32_bf16_bf16")?,
            moe_combine_bf16: module.load_function("moe_combine_bf16")?,
            logit_softcap_bf16_f32: module.load_function("logit_softcap_bf16_f32")?,
        })
    }

    /// Copies row `ids[t]` of `table` (shape `[vocab, 2304]`) to row `t` of `out`.
    /// An ID outside the table gives a zero row.
    pub fn embedding(
        &self,
        stream: &CudaStream,
        ids: &CudaSlice<i32>,
        table: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        let vocab = ensure_rows("table", table.len(), HIDDEN)?;
        let vocab = i32::try_from(vocab).context("table has too many rows")?;
        ensure!(
            out.len() == ids.len() * HIDDEN,
            "out has {} elements, expected {} ids x {HIDDEN}",
            out.len(),
            ids.len()
        );
        if ids.is_empty() {
            return Ok(());
        }
        let tokens = u32::try_from(ids.len()).context("too many ids")?;
        let config = LaunchConfig {
            grid_dim: (tokens, 1, 1),
            block_dim: ((HIDDEN / EMBEDDING_VEC) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.embedding_bf16_2304);
        launch.arg(ids).arg(table).arg(out).arg(&vocab);
        // SAFETY: the arguments match `embedding_bf16_2304(const int*, const bf16*, bf16*, int)`.
        // `out` holds one row per id, and the kernel bounds-checks each id against `vocab`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Softcapped SwiGLU of the interleaved expert projection `gate_up` (shape
    /// `[rows, 2 * 1152]`, gate at even columns, up at odd) into `out` (`[rows, 1152]`).
    pub fn softcapped_swiglu(
        &self,
        stream: &CudaStream,
        gate_up: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
        gate_softcap: f32,
        up_softcap: f32,
    ) -> anyhow::Result<()> {
        ensure_rows("gate_up", gate_up.len(), 2 * EXPERT_WIDTH)?;
        ensure!(
            gate_up.len() == 2 * out.len(),
            "gate_up has {} elements, out has {}; expected twice as many",
            gate_up.len(),
            out.len()
        );
        let n = out.len();
        let mut launch = stream.launch_builder(&self.softcapped_swiglu_bf16);
        launch
            .arg(gate_up)
            .arg(out)
            .arg(&n)
            .arg(&gate_softcap)
            .arg(&up_softcap);
        // SAFETY: the arguments match `softcapped_swiglu_bf16(const bf162*, bf16*, size_t,
        // float, float)`, and `gate_up` holds `n` BF16 pairs for the `n` outputs.
        unsafe { launch.launch(grid_stride(n)) }?;
        Ok(())
    }

    /// `out = attn * sigmoid(gate)` over `[rows, 24 * 128]` BF16 matrices.
    pub fn attention_gate(
        &self,
        stream: &CudaStream,
        attn: &CudaSlice<bf16>,
        gate: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        ensure_rows("attn", attn.len(), ATTENTION_WIDTH)?;
        ensure!(
            gate.len() == attn.len() && out.len() == attn.len(),
            "attn, gate and out have {}, {} and {} elements",
            attn.len(),
            gate.len(),
            out.len()
        );
        let n = attn.len();
        let mut launch = stream.launch_builder(&self.attention_gate_bf16);
        launch.arg(attn).arg(gate).arg(out).arg(&n);
        // SAFETY: the arguments match `attention_gate_bf16(const bf16*, const bf16*, bf16*,
        // size_t)`, and all three buffers hold `n` elements.
        unsafe { launch.launch(grid_stride(n)) }?;
        Ok(())
    }

    /// Multiplies each `row_width`-element row of `x` in place by the log-scaling factor
    /// of its position. Row `i` is at position `start_position + i`. Used on the query
    /// states (`row_width` 24 * 128) and relative states (24 * 16) of full attention.
    pub fn log_scale(
        &self,
        stream: &CudaStream,
        x: &mut CudaSlice<bf16>,
        row_width: usize,
        start_position: u32,
        alpha: f32,
        n_floor: f32,
    ) -> anyhow::Result<()> {
        ensure!(row_width > 0, "row_width is zero");
        let rows = ensure_rows("x", x.len(), row_width)?;
        if rows == 0 {
            return Ok(());
        }
        let rows = u32::try_from(rows).context("too many rows")?;
        start_position
            .checked_add(rows)
            .context("positions overflow u32")?;
        let row_width = u32::try_from(row_width).context("row_width too large")?;
        let config = LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.log_scale_bf16);
        launch
            .arg(x)
            .arg(&row_width)
            .arg(&start_position)
            .arg(&alpha)
            .arg(&n_floor);
        // SAFETY: the arguments match `log_scale_bf16(bf16*, uint32_t, uint32_t, float,
        // float)`, and `x` holds `rows` rows of `row_width` elements, one block per row.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `out = f32(a) + f32(b)` for `[rows, 2304]` matrices: the residual add after
    /// attention.
    pub fn add_bf16_to_f32(
        &self,
        stream: &CudaStream,
        a: &CudaSlice<bf16>,
        b: &CudaSlice<bf16>,
        out: &mut CudaSlice<f32>,
    ) -> anyhow::Result<()> {
        ensure_rows("a", a.len(), HIDDEN)?;
        ensure!(
            b.len() == a.len() && out.len() == a.len(),
            "a, b and out have {}, {} and {} elements",
            a.len(),
            b.len(),
            out.len()
        );
        let n = a.len();
        let mut launch = stream.launch_builder(&self.add_bf16_bf16_f32);
        launch.arg(a).arg(b).arg(out).arg(&n);
        // SAFETY: the arguments match `add_bf16_bf16_f32(const bf16*, const bf16*, float*,
        // size_t)`, and all three buffers hold `n` elements.
        unsafe { launch.launch(grid_stride(n)) }?;
        Ok(())
    }

    /// `out = bf16(residual + f32(x))` for `[rows, 2304]` matrices: the residual add that
    /// ends a decoder layer.
    pub fn add_to_bf16(
        &self,
        stream: &CudaStream,
        residual: &CudaSlice<f32>,
        x: &CudaSlice<bf16>,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        ensure_rows("residual", residual.len(), HIDDEN)?;
        ensure!(
            x.len() == residual.len() && out.len() == residual.len(),
            "residual, x and out have {}, {} and {} elements",
            residual.len(),
            x.len(),
            out.len()
        );
        let n = residual.len();
        let mut launch = stream.launch_builder(&self.add_f32_bf16_bf16);
        launch.arg(residual).arg(x).arg(out).arg(&n);
        // SAFETY: the arguments match `add_f32_bf16_bf16(const float*, const bf16*, bf16*,
        // size_t)`, and all three buffers hold `n` elements.
        unsafe { launch.launch(grid_stride(n)) }?;
        Ok(())
    }

    /// Weighted sum of each token's 8 expert outputs: `experts` is `[rows, 8, 2304]`,
    /// `weights` is `[rows, 8]` in the router's slot order, `out` is `[rows, 2304]`.
    pub fn moe_combine(
        &self,
        stream: &CudaStream,
        experts: &CudaSlice<bf16>,
        weights: &CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        let rows = ensure_rows("out", out.len(), HIDDEN)?;
        ensure!(
            experts.len() == rows * TOP_K * HIDDEN,
            "experts has {} elements, expected {rows} x {TOP_K} x {HIDDEN}",
            experts.len()
        );
        ensure!(
            weights.len() == rows * TOP_K,
            "weights has {} elements, expected {rows} x {TOP_K}",
            weights.len()
        );
        let n = out.len();
        let mut launch = stream.launch_builder(&self.moe_combine_bf16);
        launch.arg(experts).arg(weights).arg(out).arg(&n);
        // SAFETY: the arguments match `moe_combine_bf16(const bf16*, const float*, bf16*,
        // size_t)`, and the checks above size `experts` and `weights` for `n / 2304` rows.
        unsafe { launch.launch(grid_stride(n)) }?;
        Ok(())
    }

    /// `out = tanh(f32(logits) / softcap) * softcap` for BF16 `logits` of shape
    /// `[rows, 201024]`, keeping the first 200058 columns: `out` is `[rows, 200058]` F32.
    pub fn logit_softcap(
        &self,
        stream: &CudaStream,
        logits: &CudaSlice<bf16>,
        out: &mut CudaSlice<f32>,
        softcap: f32,
    ) -> anyhow::Result<()> {
        let rows = ensure_rows("logits", logits.len(), PADDED_VOCAB)?;
        ensure!(
            out.len() == rows * VOCAB,
            "out has {} elements, expected {rows} x {VOCAB}",
            out.len()
        );
        let n = out.len();
        let (in_width, out_width) = (PADDED_VOCAB as u32, VOCAB as u32);
        let mut launch = stream.launch_builder(&self.logit_softcap_bf16_f32);
        launch
            .arg(logits)
            .arg(out)
            .arg(&n)
            .arg(&in_width)
            .arg(&out_width)
            .arg(&softcap);
        // SAFETY: the arguments match `logit_softcap_bf16_f32(const bf16*, float*, size_t,
        // uint32_t, uint32_t, float)`; `out` holds `n` elements and `logits` the same rows
        // at the wider stride.
        unsafe { launch.launch(grid_stride(n)) }?;
        Ok(())
    }
}
