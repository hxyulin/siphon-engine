//! Decode attention with relative bias, sinks and KV caches (`kernels/attention.cu`).
//!
//! Every Inkling layer has 24 query heads, 8 KV heads of width 128 and 16-wide relative
//! states. A decode step for one layer is:
//!
//! 1. [`Attention::relative_bias`] turns the relative states into the bias table row of each
//!    head.
//! 2. [`Attention::append`] writes the token's key and value into the layer's [`KvCache`].
//! 3. [`Attention::decode_attention`] attends from the token's query over the cache.

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{cubin, load_module};

/// Query heads.
pub const HEADS: usize = 24;
/// Key and value heads.
pub const KV_HEADS: usize = 8;
/// Width of one query, key or value head.
pub const HEAD_DIM: usize = 128;
/// Width of one head of relative states.
pub const REL_DIM: usize = 16;
/// `HEAD_DIM^-0.5`, the softmax scale, rounded to F32 as MLX passes it.
const SCALE: f32 = 0.088_388_346;
/// Keys per cache slot row: one token's `[KV_HEADS, HEAD_DIM]`.
const KV_ROW: usize = KV_HEADS * HEAD_DIM;
/// MLX on an M5 Max switches from its one-pass to its two-pass decode kernel at this many keys.
const TWO_PASS_KEYS: usize = 1024;
/// Key blocks of MLX's one-pass kernel (its 32 simdgroups).
const ONE_PASS_BLOCKS: usize = 32;
/// Key blocks of MLX's two-pass kernel on an M5 Max with 3 query heads per KV head.
const TWO_PASS_BLOCKS: usize = 64;
/// Threads per block of the relative-bias kernel.
const BIAS_BLOCK: u32 = 256;

/// One layer's keys and values.
///
/// `keys` and `values` are BF16 buffers of shape `[slots, 8, 128]`. A sliding cache is a ring:
/// position `p` lives in slot `p % slots`. A global cache stores position `p` in slot `p` and
/// refuses to append past its capacity. The layout matches MLX's `RotatingKVCache` and
/// `KVCache` fed one token at a time.
#[derive(Debug)]
pub struct KvCache {
    keys: CudaSlice<bf16>,
    values: CudaSlice<bf16>,
    slots: usize,
    window: Option<usize>,
    len: usize,
}

impl KvCache {
    /// A ring of `window` slots for a sliding-window layer, which sees the last `window` keys.
    pub fn sliding(stream: &Arc<CudaStream>, window: usize) -> anyhow::Result<Self> {
        Self::new(stream, window, Some(window))
    }

    /// A cache of `capacity` slots for a global layer, which sees every earlier key.
    pub fn global(stream: &Arc<CudaStream>, capacity: usize) -> anyhow::Result<Self> {
        Self::new(stream, capacity, None)
    }

    fn new(stream: &Arc<CudaStream>, slots: usize, window: Option<usize>) -> anyhow::Result<Self> {
        ensure!(slots > 0, "a KV cache needs at least one slot");
        u32::try_from(slots).context("too many KV cache slots")?;
        Ok(Self {
            keys: stream.alloc_zeros(slots * KV_ROW)?,
            values: stream.alloc_zeros(slots * KV_ROW)?,
            slots,
            window,
            len: 0,
        })
    }

    /// Tokens appended so far; the next token's position.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// The sliding window, or `None` for a global layer.
    #[must_use]
    pub fn window(&self) -> Option<usize> {
        self.window
    }

    #[must_use]
    pub fn keys(&self) -> &CudaSlice<bf16> {
        &self.keys
    }

    #[must_use]
    pub fn values(&self) -> &CudaSlice<bf16> {
        &self.values
    }
}

/// Per-block partial results of [`Attention::decode_attention`], reused across calls.
#[derive(Debug)]
pub struct AttentionScratch {
    out: CudaSlice<f32>,
    max: CudaSlice<f32>,
    sum: CudaSlice<f32>,
}

impl AttentionScratch {
    pub fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let rows = HEADS * TWO_PASS_BLOCKS;
        Ok(Self {
            out: stream.alloc_zeros(rows * HEAD_DIM)?,
            max: stream.alloc_zeros(rows)?,
            sum: stream.alloc_zeros(rows)?,
        })
    }
}

#[derive(Debug)]
pub struct Attention {
    relative_bias: CudaFunction,
    kv_append: CudaFunction,
    decode_partial: CudaFunction,
    decode_combine: CudaFunction,
}

fn check_len(name: &str, len: usize, want: usize) -> anyhow::Result<()> {
    ensure!(len == want, "{name} has {len} elements, expected {want}");
    Ok(())
}

/// Width of the bias table whose `[16, extent]` or `[24, extent]` buffer has `len` elements.
fn extent(name: &str, len: usize, rows: usize) -> anyhow::Result<u32> {
    ensure!(
        len > 0 && len.is_multiple_of(rows),
        "{name} has {len} elements, not a positive multiple of {rows}"
    );
    u32::try_from(len / rows).context("bias table too wide")
}

impl Attention {
    pub fn load(ctx: &Arc<CudaContext>) -> anyhow::Result<Self> {
        let module = load_module(ctx, "attention", cubin!("attention"))?;
        Ok(Self {
            relative_bias: module.load_function("relative_bias_bf16")?,
            kv_append: module.load_function("kv_append_bf16")?,
            decode_partial: module.load_function("decode_attention_partial_bf16")?,
            decode_combine: module.load_function("decode_attention_combine_bf16")?,
        })
    }

    /// `bias = rel @ table` for relative states `rel` `[24, 16]` and the layer's
    /// `rel_logits_proj.proj` table `[16, extent]`, written to `bias` `[24, extent]`.
    ///
    /// Bit-exact with MLX's BF16 matmul for this shape.
    pub fn relative_bias(
        &self,
        stream: &CudaStream,
        rel: &CudaSlice<bf16>,
        table: &CudaSlice<bf16>,
        bias: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        check_len("rel", rel.len(), HEADS * REL_DIM)?;
        let extent = extent("table", table.len(), REL_DIM)?;
        check_len("bias", bias.len(), HEADS * extent as usize)?;
        let config = LaunchConfig {
            grid_dim: (extent.div_ceil(BIAS_BLOCK), HEADS as u32, 1),
            block_dim: (BIAS_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let extent = i32::try_from(extent).context("bias table too wide")?;
        let mut launch = stream.launch_builder(&self.relative_bias);
        launch.arg(rel).arg(table).arg(bias).arg(&extent);
        // SAFETY: the arguments match `relative_bias_bf16(const bf16*, const bf16*, bf16*, int)`.
        // The checks above size `rel`, `table` and `bias` for 24 heads and `extent` columns, and
        // the kernel skips threads past `extent`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Appends one token's key `k` and value `v` (`[8, 128]` each) to `cache` and returns the
    /// token's position.
    ///
    /// Fails when a global cache is full.
    pub fn append(
        &self,
        stream: &CudaStream,
        cache: &mut KvCache,
        k: &CudaSlice<bf16>,
        v: &CudaSlice<bf16>,
    ) -> anyhow::Result<usize> {
        check_len("k", k.len(), KV_ROW)?;
        check_len("v", v.len(), KV_ROW)?;
        let position = cache.len;
        ensure!(
            cache.window.is_some() || position < cache.slots,
            "global KV cache is full at {} tokens",
            cache.slots
        );
        u32::try_from(position).context("position does not fit the kernels")?;
        let slot = u32::try_from(position % cache.slots)?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: ((KV_ROW / 8) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.kv_append);
        launch
            .arg(k)
            .arg(v)
            .arg(&mut cache.keys)
            .arg(&mut cache.values)
            .arg(&slot);
        // SAFETY: the arguments match `kv_append_bf16(const bf16*, const bf16*, bf16*, bf16*,
        // unsigned)`. `k` and `v` hold one 1024-element row and `slot < slots`, so the write stays
        // inside the `[slots, 1024]` caches.
        unsafe { launch.launch(config) }?;
        cache.len += 1;
        Ok(position)
    }

    /// Attention output `out` `[24, 128]` of the most recently appended token, whose query is
    /// `q` `[24, 128]`, over `cache`.
    ///
    /// `bias` `[24, extent]` comes from [`Attention::relative_bias`]; a key at distance `d`
    /// gets `bias[h, d]` when `d < extent` and nothing otherwise. `sinks` `[24]` are the sink
    /// logits. Splits the keys as MLX does on an M5 Max: 32 blocks with F32 partials below 1024
    /// keys, 64 blocks with BF16 partials from 1024 keys on.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn decode_attention(
        &self,
        stream: &CudaStream,
        q: &CudaSlice<bf16>,
        cache: &KvCache,
        bias: &CudaSlice<bf16>,
        sinks: &CudaSlice<bf16>,
        scratch: &mut AttentionScratch,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        check_len("q", q.len(), HEADS * HEAD_DIM)?;
        check_len("sinks", sinks.len(), HEADS)?;
        check_len("out", out.len(), HEADS * HEAD_DIM)?;
        let extent = extent("bias", bias.len(), HEADS)?;
        ensure!(
            !cache.is_empty(),
            "decode attention needs an appended token"
        );
        let position = cache.len - 1;
        let keys = (position + 1).min(cache.slots);
        let (blocks, round_partials) = if keys < TWO_PASS_KEYS {
            (ONE_PASS_BLOCKS, 0i32)
        } else {
            (TWO_PASS_BLOCKS, 1i32)
        };
        let position = u32::try_from(position)?;
        let slots = u32::try_from(cache.slots)?;
        let window = u32::try_from(cache.window.unwrap_or(0)).context("window too large")?;

        let mut launch = stream.launch_builder(&self.decode_partial);
        launch
            .arg(q)
            .arg(&cache.keys)
            .arg(&cache.values)
            .arg(bias)
            .arg(sinks)
            .arg(&mut scratch.out)
            .arg(&mut scratch.max)
            .arg(&mut scratch.sum)
            .arg(&position)
            .arg(&slots)
            .arg(&window)
            .arg(&extent)
            .arg(&SCALE)
            .arg(&round_partials);
        let config = LaunchConfig {
            grid_dim: (HEADS as u32, blocks as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `decode_attention_partial_bf16(const bf16* q, const bf16*
        // k_cache, const bf16* v_cache, const bf16* bias, const bf16* sinks, float*, float*,
        // float*, unsigned position, unsigned slots, unsigned window, unsigned extent, float,
        // int)`. The kernel reads slots below min(position + 1, slots) of the caches, bias
        // columns below `extent`, and writes 24 x `blocks` partial rows, which the scratch holds.
        unsafe { launch.launch(config) }?;

        let blocks = i32::try_from(blocks)?;
        let mut launch = stream.launch_builder(&self.decode_combine);
        launch
            .arg(&scratch.out)
            .arg(&scratch.max)
            .arg(&scratch.sum)
            .arg(out)
            .arg(&blocks);
        let config = LaunchConfig {
            grid_dim: (HEADS as u32, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `decode_attention_combine_bf16(const float*, const float*,
        // const float*, bf16*, int)`. It reads the 24 x `blocks` partial rows written above and
        // writes `out`, checked to hold 24 x 128 values.
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}
