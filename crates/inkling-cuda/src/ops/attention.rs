//! Decode attention with relative bias, sinks and KV caches (`kernels/attention.cu`).
//!
//! Every Inkling layer has 24 query heads, 8 KV heads of width 128 and 16-wide relative
//! states. A decode step for one layer is:
//!
//! 1. [`Attention::relative_bias`] turns the relative states into the bias table row of each
//!    head.
//! 2. [`Attention::append`] writes the token's key and value into the layer's [`KvCache`].
//! 3. [`Attention::decode_attention`] attends from the token's query over the cache.
//!
//! Append and attention read the token's position from a one-element device buffer, so a
//! captured CUDA graph can replay them at any position. The caller tracks the position on
//! the host and checks [`KvCache::holds`] before each token.
//!
//! A global cache stores BF16 or, with [`KvFormat::Fp8`], E4M3 keys and values. Each cache
//! picks its kernels when they are launched, so a captured graph holds the variant of each
//! cache and nothing branches when it replays.

use std::{fmt, str::FromStr, sync::Arc};

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg as _,
};
use half::bf16;

use crate::gpu::{DeviceIn, cubin, load_module};

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
/// Split rows the partial buffers hold per query head, `kMaxSplits` in the kernel.
const MAX_SPLITS: usize = 64;
/// Splits of a large cache: 8 KV heads times 40 splits is 320 blocks, two per SM on an
/// RTX 5090's 170 SMs, so one wave covers the cache.
const SPLITS: usize = 40;
/// Fewest cache slots per split, two 32-key tiles, so that small caches use fewer blocks.
const MIN_SPLIT_SLOTS: usize = 64;
/// Threads per decode attention block.
const DECODE_BLOCK: u32 = 128;
/// Threads per block of the relative-bias kernel.
const BIAS_BLOCK: u32 = 256;

/// Element format of the global-layer KV caches. Sliding caches are always BF16.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvFormat {
    /// BF16, the projections' own values.
    #[default]
    Bf16,
    /// E4M3 without a scale: each BF16 value rounded to the nearest E4M3 value, which halves
    /// the bytes attention reads. Keys are L2-clipped to norm `sqrt(512)` and values are
    /// RMS-normed, so every element lies far inside E4M3's range of +-448.
    Fp8,
}

impl KvFormat {
    /// Bytes of one stored key or value element.
    #[must_use]
    pub fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::Fp8 => 1,
        }
    }
}

impl FromStr for KvFormat {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            _ => anyhow::bail!("unknown KV cache format {s:?}; expected bf16 or fp8"),
        }
    }
}

impl fmt::Display for KvFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        })
    }
}

/// The key and value buffers of a [`KvCache`], each of shape `[slots, 8, 128]`.
#[derive(Debug)]
pub enum KvData {
    Bf16 {
        keys: CudaSlice<bf16>,
        values: CudaSlice<bf16>,
    },
    /// E4M3 bytes.
    Fp8 {
        keys: CudaSlice<u8>,
        values: CudaSlice<u8>,
    },
}

/// One layer's keys and values.
///
/// The keys and values are buffers of shape `[slots, 8, 128]`. A sliding cache is a ring:
/// position `p` lives in slot `p % slots`. A global cache stores position `p` in slot `p` and
/// holds positions below its capacity. The layout matches MLX's `RotatingKVCache` and
/// `KVCache` fed one token at a time.
///
/// The cache does not count its tokens: append and attention take the position, and a
/// cache at position `p` holds the tokens at positions below `p`.
#[derive(Debug)]
pub struct KvCache {
    data: KvData,
    slots: usize,
    window: Option<usize>,
}

impl KvCache {
    /// A BF16 ring of `window` slots for a sliding-window layer, which sees the last `window`
    /// keys.
    pub fn sliding(stream: &Arc<CudaStream>, window: usize) -> anyhow::Result<Self> {
        Self::new(stream, window, Some(window), KvFormat::Bf16)
    }

    /// A BF16 cache of `capacity` slots for a global layer, which sees every earlier key.
    pub fn global(stream: &Arc<CudaStream>, capacity: usize) -> anyhow::Result<Self> {
        Self::global_with_format(stream, capacity, KvFormat::Bf16)
    }

    /// [`KvCache::global`] storing `format` elements.
    pub fn global_with_format(
        stream: &Arc<CudaStream>,
        capacity: usize,
        format: KvFormat,
    ) -> anyhow::Result<Self> {
        Self::new(stream, capacity, None, format)
    }

    fn new(
        stream: &Arc<CudaStream>,
        slots: usize,
        window: Option<usize>,
        format: KvFormat,
    ) -> anyhow::Result<Self> {
        ensure!(slots > 0, "a KV cache needs at least one slot");
        u32::try_from(slots).context("too many KV cache slots")?;
        let len = slots * KV_ROW;
        let data = match format {
            KvFormat::Bf16 => KvData::Bf16 {
                keys: stream.alloc_zeros(len)?,
                values: stream.alloc_zeros(len)?,
            },
            KvFormat::Fp8 => KvData::Fp8 {
                keys: stream.alloc_zeros(len)?,
                values: stream.alloc_zeros(len)?,
            },
        };
        Ok(Self {
            data,
            slots,
            window,
        })
    }

    /// Whether the token at `position` fits: always for a sliding cache, below the capacity
    /// for a global one.
    #[must_use]
    pub fn holds(&self, position: usize) -> bool {
        self.window.is_some() || position < self.slots
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
    pub fn format(&self) -> KvFormat {
        match self.data {
            KvData::Bf16 { .. } => KvFormat::Bf16,
            KvData::Fp8 { .. } => KvFormat::Fp8,
        }
    }

    #[must_use]
    pub fn data(&self) -> &KvData {
        &self.data
    }
}

/// Per-split partial results of [`Attention::decode_attention`] and the per-KV-head counts of
/// finished splits, reused across calls and layers.
///
/// The kernel returns every count to zero, so calls must not overlap.
#[derive(Debug)]
pub struct AttentionScratch {
    out: CudaSlice<f32>,
    max: CudaSlice<f32>,
    sum: CudaSlice<f32>,
    counters: CudaSlice<u32>,
}

impl AttentionScratch {
    pub fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let rows = HEADS * MAX_SPLITS;
        Ok(Self {
            out: stream.alloc_zeros(rows * HEAD_DIM)?,
            max: stream.alloc_zeros(rows)?,
            sum: stream.alloc_zeros(rows)?,
            counters: stream.alloc_zeros(KV_HEADS)?,
        })
    }
}

/// Splits of the cache slots that [`Attention::decode_attention`] launches for a cache of
/// `slots`: [`SPLITS`], or fewer so that each covers at least [`MIN_SPLIT_SLOTS`].
fn splits(slots: usize) -> u32 {
    let splits = slots.div_ceil(MIN_SPLIT_SLOTS).clamp(1, SPLITS);
    debug_assert!(splits <= MAX_SPLITS);
    splits as u32
}

#[derive(Debug)]
pub struct Attention {
    relative_bias: CudaFunction,
    kv_append: CudaFunction,
    kv_append_fp8: CudaFunction,
    decode: CudaFunction,
    decode_fp8: CudaFunction,
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
            kv_append_fp8: module.load_function("kv_append_fp8")?,
            decode: module.load_function("decode_attention_bf16")?,
            decode_fp8: module.load_function("decode_attention_fp8")?,
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

    /// Writes one token's key `k` and value `v` (`[8, 128]` each) into `cache` at the token's
    /// position, the one value in `position`. An FP8 cache stores each value rounded to the
    /// nearest E4M3 value, ties to even.
    ///
    /// The caller checks [`KvCache::holds`] for that position first.
    pub fn append(
        &self,
        stream: &CudaStream,
        cache: &mut KvCache,
        position: &CudaSlice<u32>,
        k: &impl DeviceIn<bf16>,
        v: &impl DeviceIn<bf16>,
    ) -> anyhow::Result<()> {
        let (k, v) = (k.view(), v.view());
        check_len("k", k.len(), KV_ROW)?;
        check_len("v", v.len(), KV_ROW)?;
        check_len("position", position.len(), 1)?;
        let slots = u32::try_from(cache.slots)?;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: ((KV_ROW / 8) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let function = match cache.data {
            KvData::Bf16 { .. } => &self.kv_append,
            KvData::Fp8 { .. } => &self.kv_append_fp8,
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&k).arg(&v);
        match &mut cache.data {
            KvData::Bf16 { keys, values } => launch.arg(keys).arg(values),
            KvData::Fp8 { keys, values } => launch.arg(keys).arg(values),
        };
        launch.arg(position).arg(&slots);
        // SAFETY: the arguments match `kv_append_bf16(const bf16*, const bf16*, bf16*, bf16*,
        // const unsigned*, unsigned)`, or `kv_append_fp8` with `uint8_t*` caches for an FP8
        // cache. `k` and `v` hold one 1024-element row, `position` the one value the kernel
        // reads, and the kernel writes slot `position % slots`, inside the `[slots, 1024]`
        // caches.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Attention output `out` `[24, 128]` of the token at `position` (one value), whose query
    /// is `q` `[24, 128]`, over `cache` after the token's append.
    ///
    /// `bias` `[24, extent]` comes from [`Attention::relative_bias`]; a key at distance `d`
    /// gets `bias[h, d]` when `d < extent` and nothing otherwise. `sinks` `[24]` are the sink
    /// logits. An FP8 cache's E4M3 values are decoded exactly, so only the cache rounding
    /// differs from BF16. One launch: each block reads one KV head over one split of the cache for its 3
    /// query heads, and the last block of each KV head combines the splits. The kernel cuts
    /// the live slots into splits from `position`, so the launch is the same at every
    /// position.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn decode_attention(
        &self,
        stream: &CudaStream,
        q: &CudaSlice<bf16>,
        cache: &KvCache,
        position: &CudaSlice<u32>,
        bias: &CudaSlice<bf16>,
        sinks: &CudaSlice<bf16>,
        scratch: &mut AttentionScratch,
        out: &mut CudaSlice<bf16>,
    ) -> anyhow::Result<()> {
        check_len("q", q.len(), HEADS * HEAD_DIM)?;
        check_len("sinks", sinks.len(), HEADS)?;
        check_len("out", out.len(), HEADS * HEAD_DIM)?;
        check_len("position", position.len(), 1)?;
        let extent = extent("bias", bias.len(), HEADS)?;
        let slots = u32::try_from(cache.slots)?;
        let window = u32::try_from(cache.window.unwrap_or(0)).context("window too large")?;

        let function = match cache.data {
            KvData::Bf16 { .. } => &self.decode,
            KvData::Fp8 { .. } => &self.decode_fp8,
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(q);
        match &cache.data {
            KvData::Bf16 { keys, values } => launch.arg(keys).arg(values),
            KvData::Fp8 { keys, values } => launch.arg(keys).arg(values),
        };
        launch
            .arg(bias)
            .arg(sinks)
            .arg(&mut scratch.out)
            .arg(&mut scratch.max)
            .arg(&mut scratch.sum)
            .arg(&mut scratch.counters)
            .arg(out)
            .arg(position)
            .arg(&slots)
            .arg(&window)
            .arg(&extent)
            .arg(&SCALE);
        let config = LaunchConfig {
            grid_dim: (KV_HEADS as u32, splits(cache.slots), 1),
            block_dim: (DECODE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `decode_attention_bf16(const bf16* q, const bf16* k_cache,
        // const bf16* v_cache, const bf16* bias, const bf16* sinks, float* partial_out, float*
        // partial_max, float* partial_sum, unsigned* counters, bf16* out, const unsigned*
        // position, unsigned slots, unsigned window, unsigned extent, float scale)`, or
        // `decode_attention_fp8` with `uint8_t*` caches for an FP8 cache. The kernel
        // reads slots below min(position + 1, slots) of the caches and bias columns below
        // `extent`. It writes partial rows for 24 heads x at most `MAX_SPLITS` splits and 8
        // counters, which the scratch holds, and `out`, checked to hold 24 x 128 values.
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}
