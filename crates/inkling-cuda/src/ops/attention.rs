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
//! [`Attention::prefill_attention`] does steps 2 and 3 for a chunk of up to
//! [`MAX_PREFILL_ROWS`] consecutive tokens with tensor cores, after
//! [`Attention::relative_bias_rows`] has done step 1 for all of them.
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
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg as _,
    sys::{CUdevice_attribute, CUfunction_attribute},
};
use half::bf16;

use crate::gpu::{DeviceIn, DeviceOut, cubin, load_module};

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
/// Most rows one [`Attention::prefill_attention`] call takes, `kPrefillMaxBlocks` blocks of
/// rows in the kernel.
pub const MAX_PREFILL_ROWS: usize = 2048;
/// Rows of (query row, head) pairs per prefill block: 8 warps of 16.
const PREFILL_BLOCK_ROWS: usize = 128;
/// Threads per prefill block.
const PREFILL_BLOCK: u32 = 256;
/// Dynamic shared memory of a prefill block, checked against the kernel's layout by
/// `static_assert`s in `attention.cu`. One block fits on an SM.
const PREFILL_SHARED_BYTES: u32 = 84_240;
/// F32 values of one prefill partial slot: 128 rows of 128 outputs, a max and a sum.
const PREFILL_SLOT: usize = PREFILL_BLOCK_ROWS * (HEAD_DIM + 2);

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

    /// Zeroes every slot.
    pub fn clear(&mut self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        match &mut self.data {
            KvData::Bf16 { keys, values } => {
                stream.memset_zeros(keys)?;
                stream.memset_zeros(values)?;
            }
            KvData::Fp8 { keys, values } => {
                stream.memset_zeros(keys)?;
                stream.memset_zeros(values)?;
            }
        }
        Ok(())
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

/// Prefill blocks per KV head for a chunk of `rows`.
fn prefill_blocks(rows: usize) -> usize {
    (rows * (HEADS / KV_HEADS)).div_ceil(PREFILL_BLOCK_ROWS)
}

/// Partial results of [`Attention::prefill_attention`], the counts of finished segments
/// per unit of rows, and the decode scratch of a single-row call, reused across calls and
/// layers.
///
/// The kernel runs one block per SM. Each block stores at most two partial segments, so the
/// partial buffer holds two slots of [`PREFILL_SLOT`] values per SM, 22 MB on an RTX 5090,
/// whatever the chunk size. The kernel returns every count to zero, so calls must not
/// overlap.
#[derive(Debug)]
pub struct PrefillScratch {
    partials: CudaSlice<f32>,
    counters: CudaSlice<u32>,
    /// Blocks per launch, the device's SM count.
    blocks: usize,
    /// Most rows per call.
    max_rows: usize,
    /// Scratch of the decode kernel, which serves single rows.
    decode: AttentionScratch,
}

impl PrefillScratch {
    /// Scratch for chunks of up to `max_rows` rows.
    pub fn new(stream: &Arc<CudaStream>, max_rows: usize) -> anyhow::Result<Self> {
        ensure!(
            (1..=MAX_PREFILL_ROWS).contains(&max_rows),
            "prefill takes 1 to {MAX_PREFILL_ROWS} rows, not {max_rows}"
        );
        let blocks = usize::try_from(
            stream
                .context()
                .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        )?;
        Ok(Self {
            partials: stream.alloc_zeros(2 * blocks * PREFILL_SLOT)?,
            counters: stream.alloc_zeros(prefill_blocks(max_rows) * KV_HEADS)?,
            blocks,
            max_rows,
            decode: AttentionScratch::new(stream)?,
        })
    }
}

#[derive(Debug)]
pub struct Attention {
    relative_bias: CudaFunction,
    relative_bias_rows: CudaFunction,
    kv_append: CudaFunction,
    kv_append_fp8: CudaFunction,
    kv_append_rows: CudaFunction,
    kv_append_rows_fp8: CudaFunction,
    decode: CudaFunction,
    decode_fp8: CudaFunction,
    prefill: CudaFunction,
    prefill_fp8: CudaFunction,
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
        let prefill = module.load_function("prefill_attention_bf16")?;
        let prefill_fp8 = module.load_function("prefill_attention_fp8")?;
        for function in [&prefill, &prefill_fp8] {
            function.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                PREFILL_SHARED_BYTES.cast_signed(),
            )?;
        }
        Ok(Self {
            relative_bias: module.load_function("relative_bias_bf16")?,
            relative_bias_rows: module.load_function("relative_bias_rows_bf16")?,
            kv_append: module.load_function("kv_append_bf16")?,
            kv_append_fp8: module.load_function("kv_append_fp8")?,
            kv_append_rows: module.load_function("kv_append_rows_bf16")?,
            kv_append_rows_fp8: module.load_function("kv_append_rows_fp8")?,
            decode: module.load_function("decode_attention_bf16")?,
            decode_fp8: module.load_function("decode_attention_fp8")?,
            prefill,
            prefill_fp8,
        })
    }

    /// `bias = rel @ table` for relative states `rel` `[rows, 24, 16]` and the layer's
    /// `rel_logits_proj.proj` table `[16, extent]`, written to `bias` `[rows, 24, extent]`.
    /// Decode passes one row.
    ///
    /// Bit-exact with MLX's BF16 matmul for this shape.
    pub fn relative_bias(
        &self,
        stream: &CudaStream,
        rel: &impl DeviceIn<bf16>,
        table: &CudaSlice<bf16>,
        bias: &mut impl DeviceOut<bf16>,
    ) -> anyhow::Result<()> {
        let (rel, mut bias) = (rel.view(), bias.view_mut());
        ensure!(
            !rel.is_empty() && rel.len().is_multiple_of(HEADS * REL_DIM),
            "rel has {} elements, not a positive multiple of {HEADS} x {REL_DIM}",
            rel.len()
        );
        let heads = rel.len() / REL_DIM;
        let extent = extent("table", table.len(), REL_DIM)?;
        check_len("bias", bias.len(), heads * extent as usize)?;
        let config = LaunchConfig {
            grid_dim: (
                extent.div_ceil(BIAS_BLOCK),
                u32::try_from(heads).context("too many rows")?,
                1,
            ),
            block_dim: (BIAS_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let extent = i32::try_from(extent).context("bias table too wide")?;
        let mut launch = stream.launch_builder(&self.relative_bias);
        launch.arg(&rel).arg(table).arg(&mut bias).arg(&extent);
        // SAFETY: the arguments match `relative_bias_bf16(const bf16*, const bf16*, bf16*, int)`.
        // The kernel takes one head per `blockIdx.y`; the checks above size `rel`, `table`
        // and `bias` for `heads` heads and `extent` columns, and the kernel skips threads past
        // `extent`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// [`Attention::relative_bias`] for a chunk: `rel` `[rows, 24, 16]` and `table`
    /// `[16, extent]` give `bias` `[rows, 24, extent]`, each value bit for bit as
    /// [`Attention::relative_bias`] computes it. One block per row keeps the table columns in
    /// registers across the 24 heads. `extent` must be a multiple of 4.
    pub fn relative_bias_rows(
        &self,
        stream: &CudaStream,
        rel: &impl DeviceIn<bf16>,
        table: &CudaSlice<bf16>,
        bias: &mut impl DeviceOut<bf16>,
    ) -> anyhow::Result<()> {
        let (rel, mut bias) = (rel.view(), bias.view_mut());
        ensure!(
            !rel.is_empty() && rel.len().is_multiple_of(HEADS * REL_DIM),
            "rel has {} elements, not a positive multiple of {HEADS} x {REL_DIM}",
            rel.len()
        );
        let rows = rel.len() / (HEADS * REL_DIM);
        let extent = extent("table", table.len(), REL_DIM)?;
        ensure!(
            extent.is_multiple_of(4),
            "the multi-row bias needs an extent that is a multiple of 4, not {extent}"
        );
        check_len("bias", bias.len(), rows * HEADS * extent as usize)?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows).context("too many rows")?, 1, 1),
            block_dim: (BIAS_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let extent = i32::try_from(extent).context("bias table too wide")?;
        let mut launch = stream.launch_builder(&self.relative_bias_rows);
        launch.arg(&rel).arg(table).arg(&mut bias).arg(&extent);
        // SAFETY: the arguments match `relative_bias_rows_bf16(const bf16*, const bf16*, bf16*,
        // int)`. Block `i` reads row `i` of `rel` and columns below `extent` of `table`, and
        // writes row `i` of `bias`; the checks above size all three for `rows` rows and
        // `extent` columns, and `extent` is a multiple of 4 for the kernel's 4-wide accesses.
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
        position: &impl DeviceIn<u32>,
        k: &impl DeviceIn<bf16>,
        v: &impl DeviceIn<bf16>,
    ) -> anyhow::Result<()> {
        let (k, v, position) = (k.view(), v.view(), position.view());
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
        launch.arg(&position).arg(&slots);
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
        q: &impl DeviceIn<bf16>,
        cache: &KvCache,
        position: &impl DeviceIn<u32>,
        bias: &impl DeviceIn<bf16>,
        sinks: &impl DeviceIn<bf16>,
        scratch: &mut AttentionScratch,
        out: &mut impl DeviceOut<bf16>,
    ) -> anyhow::Result<()> {
        let (q, position, bias, sinks, mut out) = (
            q.view(),
            position.view(),
            bias.view(),
            sinks.view(),
            out.view_mut(),
        );
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
        launch.arg(&q);
        match &cache.data {
            KvData::Bf16 { keys, values } => launch.arg(keys).arg(values),
            KvData::Fp8 { keys, values } => launch.arg(keys).arg(values),
        };
        launch
            .arg(&bias)
            .arg(&sinks)
            .arg(&mut scratch.out)
            .arg(&mut scratch.max)
            .arg(&mut scratch.sum)
            .arg(&mut scratch.counters)
            .arg(&mut out)
            .arg(&position)
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

    /// Writes `rows` tokens' keys `k` and values `v` (`[rows, 8, 128]` each) into `cache` at
    /// positions `start + i`, as `rows` calls of [`Attention::append`] would: with more rows
    /// than slots, only the last `slots` rows, which those calls leave behind.
    fn append_rows(
        &self,
        stream: &CudaStream,
        cache: &mut KvCache,
        start: &CudaSlice<u32>,
        rows: usize,
        k: &CudaView<'_, bf16>,
        v: &CudaView<'_, bf16>,
    ) -> anyhow::Result<()> {
        let slots = u32::try_from(cache.slots)?;
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows)?, 1, 1),
            block_dim: ((KV_ROW / 8) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let function = match cache.data {
            KvData::Bf16 { .. } => &self.kv_append_rows,
            KvData::Fp8 { .. } => &self.kv_append_rows_fp8,
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(k).arg(v);
        match &mut cache.data {
            KvData::Bf16 { keys, values } => launch.arg(keys).arg(values),
            KvData::Fp8 { keys, values } => launch.arg(keys).arg(values),
        };
        launch.arg(start).arg(&slots);
        // SAFETY: the arguments match `kv_append_rows_bf16(const bf16*, const bf16*, bf16*,
        // bf16*, const unsigned*, unsigned)`, or `kv_append_rows_fp8` with `uint8_t*` caches.
        // The caller checked that `k` and `v` hold `rows` rows of 1024 and `start` one value;
        // block `i` reads row `i` and writes slot `(start + i) % slots` of the caches, or
        // nothing when `i + slots < rows`, so no two blocks write the same slot.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Chunked prefill: appends `rows` consecutive tokens to `cache` and computes their
    /// attention, as `rows` pairs of [`Attention::append`] and
    /// [`Attention::decode_attention`] would.
    ///
    /// Row `i` is the token at position `start + i`, where `start` holds one value. Its query
    /// is row `i` of `q` `[rows, 24, 128]`, its key and value rows `i` of `k` and `v`
    /// `[rows, 8, 128]`, its bias row `i` of `bias` `[rows, 24, extent]` (as
    /// [`Attention::relative_bias_rows`] builds it) and its sink logits row `i` of `sinks`
    /// `[rows, 24]`. The output goes to row `i` of `out` `[rows, 24, 128]`. Row `i` sees the
    /// cache's earlier tokens and the chunk's rows up to and including `i`, within the
    /// sliding window for a sliding cache. A chunk may be longer than the window; the ring
    /// then keeps its last `window` rows.
    ///
    /// For a sliding cache the kernel reads the chunk's own keys from `k` and `v` and the
    /// older ones from the ring, then appends, because the chunk's append overwrites slots
    /// that its earlier rows still see. A global cache never wraps, so it appends first and
    /// reads every key from the cache; an FP8 cache thus attends over the rounded keys, as
    /// decode does.
    ///
    /// A single row runs [`Attention::append`] and [`Attention::decode_attention`].
    ///
    /// `rows` is 1 to `scratch`'s row count. The caller checks [`KvCache::holds`] for
    /// position `start + rows - 1`, and `extent` must be a multiple of 8.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn prefill_attention(
        &self,
        stream: &CudaStream,
        q: &impl DeviceIn<bf16>,
        k: &impl DeviceIn<bf16>,
        v: &impl DeviceIn<bf16>,
        cache: &mut KvCache,
        start: &CudaSlice<u32>,
        rows: usize,
        bias: &impl DeviceIn<bf16>,
        sinks: &impl DeviceIn<bf16>,
        scratch: &mut PrefillScratch,
        out: &mut impl DeviceOut<bf16>,
    ) -> anyhow::Result<()> {
        let (q, k, v, bias, sinks) = (q.view(), k.view(), v.view(), bias.view(), sinks.view());
        let mut out = out.view_mut();
        ensure!(
            (1..=scratch.max_rows).contains(&rows),
            "prefill scratch takes 1 to {} rows, not {rows}",
            scratch.max_rows
        );
        check_len("q", q.len(), rows * HEADS * HEAD_DIM)?;
        check_len("k", k.len(), rows * KV_ROW)?;
        check_len("v", v.len(), rows * KV_ROW)?;
        check_len("sinks", sinks.len(), rows * HEADS)?;
        check_len("out", out.len(), rows * HEADS * HEAD_DIM)?;
        check_len("start", start.len(), 1)?;
        let extent = extent("bias", bias.len(), rows * HEADS)?;
        ensure!(
            extent.is_multiple_of(8),
            "prefill needs a bias extent that is a multiple of 8, not {extent}"
        );
        if rows == 1 {
            // Decode attention streams the cache faster than a block of 128 rows with 3
            // used ones, and computes the same thing.
            self.append(stream, cache, start, &k, &v)?;
            return self.decode_attention(
                stream,
                &q,
                cache,
                start,
                &bias,
                &sinks,
                &mut scratch.decode,
                &mut out,
            );
        }
        let slots = u32::try_from(cache.slots)?;
        let window = u32::try_from(cache.window.unwrap_or(0)).context("window too large")?;
        let rows_u32 = u32::try_from(rows)?;
        // A sliding cache appends after attention, a global one before.
        let input_keys = u32::from(cache.window.is_some());
        ensure!(
            input_keys == 0 || cache.format() == KvFormat::Bf16,
            "sliding caches are BF16"
        );
        if input_keys == 0 {
            self.append_rows(stream, cache, start, rows, &k, &v)?;
        }

        let function = match cache.data {
            KvData::Bf16 { .. } => &self.prefill,
            KvData::Fp8 { .. } => &self.prefill_fp8,
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&q).arg(&k).arg(&v);
        match &cache.data {
            KvData::Bf16 { keys, values } => launch.arg(keys).arg(values),
            KvData::Fp8 { keys, values } => launch.arg(keys).arg(values),
        };
        launch
            .arg(&bias)
            .arg(&sinks)
            .arg(&mut scratch.partials)
            .arg(&mut scratch.counters)
            .arg(&mut out)
            .arg(start)
            .arg(&rows_u32)
            .arg(&slots)
            .arg(&window)
            .arg(&extent)
            .arg(&input_keys)
            .arg(&SCALE);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(scratch.blocks)?, 1, 1),
            block_dim: (PREFILL_BLOCK, 1, 1),
            shared_mem_bytes: PREFILL_SHARED_BYTES,
        };
        // SAFETY: the arguments match `prefill_attention_bf16(const bf16* q, const bf16* k_in,
        // const bf16* v_in, const bf16* k_cache, const bf16* v_cache, const bf16* bias, const
        // bf16* sinks, float* partials, unsigned* counters, bf16* out, const unsigned* start,
        // unsigned rows, unsigned slots, unsigned window, unsigned extent, unsigned
        // input_keys, float scale)`, or `prefill_attention_fp8` with `uint8_t*` caches. The
        // checks above size `q`, `k`, `v`, `bias`, `sinks` and `out` for `rows` rows. The
        // kernel reads input rows below `rows`, cache slots below `slots` and bias columns
        // below `extent`. Its `scratch.blocks`
        // blocks write at most two partial slots each, which `scratch.partials` holds, and
        // `prefill_blocks(rows) * 8` counters, which `scratch.counters` holds since `rows` is
        // at most `scratch.max_rows`; the shared memory size matches the kernel's
        // `static_assert`s.
        unsafe { launch.launch(config) }?;

        if input_keys == 1 {
            self.append_rows(stream, cache, start, rows, &k, &v)?;
        }
        Ok(())
    }
}
