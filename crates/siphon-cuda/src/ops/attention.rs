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
//! [`Attention::decode_step`] runs a layer's whole decode step from its projections in one
//! launch: the post-processing of `Norm::attention_post`, steps 1 to 3, and the output gate.
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

use std::sync::Arc;

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, DevicePtr as _, LaunchConfig,
    PushKernelArg as _,
    sys::{CUdevice_attribute, CUfunction_attribute},
};
use half::bf16;

use crate::{
    gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module},
    ops::norm::KvConvWeights,
};

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
/// Conv taps of the key and value convolutions.
const CONV_TAPS: usize = 4;
/// Keys per cache slot row: one token's `[KV_HEADS, HEAD_DIM]`.
const KV_ROW: usize = KV_HEADS * HEAD_DIM;
/// Split rows the partial buffers hold per query head, `kMaxSplits` in the kernel.
const MAX_SPLITS: usize = 64;
/// Splits of a large cache: 8 KV heads times 40 splits is 320 blocks, two per SM on an
/// RTX 5090's 170 SMs, so one wave covers the cache.
const SPLITS: usize = 40;
/// Fewest cache slots per split by default, two 32-key tiles, so that small caches use
/// fewer blocks. [`DecodeStep::min_split_slots`] can lower it to [`ONE_TILE_SPLIT_SLOTS`].
pub const MIN_SPLIT_SLOTS: usize = 64;
/// Fewest cache slots per split of one 32-key tile: a full 1024-slot window then takes 32
/// splits of one tile each.
pub const ONE_TILE_SPLIT_SLOTS: usize = 32;
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
const _: () = assert!(PREFILL_SLOT == siphon::memory::PREFILL_SLOT);
/// Most rows [`Attention::attention_rows`] takes, `kRowsMax` in the kernel.
pub const ROWS_MAX: usize = 8;
/// Partials per query row of `attention_rows_*`: one from each of 32 blocks that take every
/// 32nd split of 64 slots, and one for the row's 8 nearest keys.
const ROWS_PARTS: usize = 33;
const _: () = assert!(ROWS_MAX * HEADS * ROWS_PARTS == siphon::memory::ROWS_ATTENTION_ROWS);
/// Blocks per KV head of `attention_rows_*`: the 32 split blocks and two for the rows' 8
/// nearest keys, 4 rows each.
const ROWS_BLOCKS: u32 = 34;
/// Most 8-column tiles of query rows: 3 heads x 8 rows.
const ROWS_QUERY_TILES: usize = 3;
/// Threads per block of `attention_rows_*`.
const ROWS_THREADS: u32 = 128;
/// Dynamic shared memory of `attention_rows_bf16` and `attention_rows_fp8`, the sizes of
/// `RowsShared` in the kernel.
const ROWS_SHARED_BYTES: [u32; 2] = [32_768, 49_152];

pub use siphon::KvFormat;

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
        Self::sliding_ring(stream, window, window)
    }

    /// [`KvCache::sliding`] with a ring of `slots`, at least `window`. The kernels find each
    /// key by its distance modulo `slots` and still see only the last `window` keys.
    pub fn sliding_ring(
        stream: &Arc<CudaStream>,
        window: usize,
        slots: usize,
    ) -> anyhow::Result<Self> {
        ensure!(
            slots >= window,
            "a ring of {slots} slots cannot hold a window of {window}"
        );
        Self::new(stream, slots, Some(window), KvFormat::Bf16)
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

    /// For a sliding cache: the device address and length in bytes of its key and value
    /// buffers, and the bytes of one slot of each. Position `p < slots` lives in slot `p`,
    /// so the first `min(p, slots)` slots hold every key and value at position `p`.
    #[must_use]
    pub fn ring_buffers(&self, stream: &CudaStream) -> Option<([(u64, usize); 2], usize)> {
        self.window?;
        let (keys, values, elem) = match &self.data {
            KvData::Bf16 { keys, values } => (
                keys.device_ptr(stream).0,
                values.device_ptr(stream).0,
                size_of::<bf16>(),
            ),
            KvData::Fp8 { keys, values } => {
                (keys.device_ptr(stream).0, values.device_ptr(stream).0, 1)
            }
        };
        let bytes = self.slots * KV_ROW * elem;
        Some(([(keys, bytes), (values, bytes)], KV_ROW * elem))
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

    /// The key and value buffers, for copying a saved cache back.
    pub fn data_mut(&mut self) -> &mut KvData {
        &mut self.data
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

/// What [`Attention::decode_step`] reads and writes besides the cache, the position and the
/// scratch.
pub struct DecodeStep<'a> {
    /// Key and value projections, BF16 `[1024]`.
    pub k_raw: &'a CudaSlice<bf16>,
    pub v_raw: &'a CudaSlice<bf16>,
    pub conv: KvConvWeights<'a>,
    /// Conv states, F32 `[3, 1024]`.
    pub k_state: &'a mut CudaSlice<f32>,
    pub v_state: &'a mut CudaSlice<f32>,
    /// Query projection `[24 * 128]` and relative-position projection `[24 * 16]`, with
    /// their L2-clip scales of the same shapes.
    pub q_raw: &'a CudaSlice<bf16>,
    pub q_scale: &'a CudaSlice<bf16>,
    pub r_raw: &'a CudaSlice<bf16>,
    pub r_scale: &'a CudaSlice<bf16>,
    /// The relative bias table `[16, extent]`.
    pub table: &'a CudaSlice<bf16>,
    /// Sink logits `[24]` and output gate projection `[24 * 128]`.
    pub sinks: &'a CudaSlice<bf16>,
    pub gate: &'a CudaSlice<bf16>,
    /// For full attention, alpha and n_floor of the query log scaling.
    pub log_scale: Option<(f32, f32)>,
    pub qk_threshold: f32,
    pub rel_threshold: f32,
    pub eps: f32,
    /// Receives the gated attention output `[24 * 128]`.
    pub out: &'a mut CudaSlice<bf16>,
    /// Fewest cache slots per split, [`MIN_SPLIT_SLOTS`] or [`ONE_TILE_SPLIT_SLOTS`]. The
    /// summation order depends on it.
    pub min_split_slots: usize,
}

/// Splits of the cache slots that [`Attention::decode_attention`] launches for a cache of
/// `slots`: [`SPLITS`], or fewer so that each covers at least `min_slots`.
fn splits(slots: usize, min_slots: usize) -> u32 {
    let splits = slots.div_ceil(min_slots).clamp(1, SPLITS);
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
    decode_step: CudaFunction,
    decode_step_fp8: CudaFunction,
    prefill: CudaFunction,
    prefill_fp8: CudaFunction,
    /// `attention_rows_bf16_<NT>` and `attention_rows_fp8_<NT>` for NT = 1 to 3.
    rows: Vec<[CudaFunction; 2]>,
}

/// The partials of [`Attention::attention_rows`]: per query row of up to 8 x 24, one
/// output, max and sum per block of its KV head. The counters, two per KV head, count the
/// finished blocks and combiners, and return to zero at the end of every call.
#[derive(Debug)]
pub struct RowsAttentionScratch {
    out: CudaSlice<f32>,
    max: CudaSlice<f32>,
    sum: CudaSlice<f32>,
    counters: CudaSlice<u32>,
}

impl RowsAttentionScratch {
    pub fn new(stream: &Arc<CudaStream>) -> anyhow::Result<Self> {
        let rows = ROWS_MAX * HEADS * ROWS_PARTS;
        Ok(Self {
            out: stream.alloc_zeros(rows * HEAD_DIM)?,
            max: stream.alloc_zeros(rows)?,
            sum: stream.alloc_zeros(rows)?,
            counters: stream.alloc_zeros(2 * KV_HEADS)?,
        })
    }
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
        let rows_function = |name: &str, smem: u32| -> anyhow::Result<CudaFunction> {
            let function = module.load_function(name)?;
            function.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem.cast_signed(),
            )?;
            Ok(function)
        };
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
            decode_step: module.load_function("decode_step_bf16")?,
            decode_step_fp8: module.load_function("decode_step_fp8")?,
            prefill,
            prefill_fp8,
            rows: (1..=ROWS_QUERY_TILES)
                .map(|nt| {
                    Ok([
                        rows_function(&format!("attention_rows_bf16_{nt}"), ROWS_SHARED_BYTES[0])?,
                        rows_function(&format!("attention_rows_fp8_{nt}"), ROWS_SHARED_BYTES[1])?,
                    ])
                })
                .collect::<anyhow::Result<_>>()?,
        })
    }

    /// The attention of the `rows` rows of a step, 1 to [`ROWS_MAX`], such as a speculative
    /// verify, at positions `start + i`: attends from each row's query (`q`
    /// `[rows, 24, 128]`) into `out` (`[rows, 24, 128]`) as `rows` pairs of
    /// [`Attention::append`] and [`Attention::decode_attention`] would, with each row's
    /// relative bias (`bias` `[rows, 24, extent]`) and sinks (`sinks` `[rows, 24]`), then
    /// appends their keys `k` and values `v` (`[rows, 8, 128]`) to `cache`. With `gate`
    /// (`[rows, 24, 128]`), `out` receives the gated output, as
    /// `Elementwise::attention_gate` computes it from the attention and the gate.
    ///
    /// Each key and value chunk streams once for all rows, on tensor cores. Each row's result
    /// depends only on its position, its query and the cache: it is the same, bit for bit,
    /// for every row count and whatever the other rows hold, so a call of one row at position
    /// `start + i` gives row `i`. It differs from [`Attention::decode_attention`]'s by
    /// summation order and by rounding the softmax weights to F16 for the value product, as
    /// prefill attention does. The caller checks [`KvCache::holds`] for the last row's
    /// position.
    #[expect(
        clippy::too_many_arguments,
        reason = "the attention inputs and outputs"
    )]
    pub fn attention_rows(
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
        gate: Option<&CudaView<'_, bf16>>,
        scratch: &mut RowsAttentionScratch,
        out: &mut impl DeviceOut<bf16>,
    ) -> anyhow::Result<()> {
        let (q, k, v, bias, sinks) = (q.view(), k.view(), v.view(), bias.view(), sinks.view());
        let mut out = out.view_mut();
        if let Some(gate) = gate {
            check_len("gate", gate.len(), rows * HEADS * HEAD_DIM)?;
        }
        ensure!(
            (1..=ROWS_MAX).contains(&rows),
            "{rows} rows, expected 1 to {ROWS_MAX}"
        );
        check_len("q", q.len(), rows * HEADS * HEAD_DIM)?;
        check_len("k", k.len(), rows * KV_ROW)?;
        check_len("v", v.len(), rows * KV_ROW)?;
        check_len("sinks", sinks.len(), rows * HEADS)?;
        check_len("out", out.len(), rows * HEADS * HEAD_DIM)?;
        check_len("start", start.len(), 1)?;
        let extent = extent("bias", bias.len(), rows * HEADS)?;
        let slots = u32::try_from(cache.slots)?;
        let window = u32::try_from(cache.window.unwrap_or(0)).context("window too large")?;
        // The entry point for the 8-column tiles of query rows, ceil(3 rows / 8).
        let entry = &self.rows[(3 * rows).div_ceil(8) - 1];
        let (function, smem) = match cache.data {
            KvData::Bf16 { .. } => (&entry[0], ROWS_SHARED_BYTES[0]),
            KvData::Fp8 { .. } => (&entry[1], ROWS_SHARED_BYTES[1]),
        };
        let mut args = KernelArgs::default();
        args.ptr(&q, stream).ptr(&k, stream).ptr(&v, stream);
        match &mut cache.data {
            KvData::Bf16 { keys, values } => args.ptr_mut(keys, stream).ptr_mut(values, stream),
            KvData::Fp8 { keys, values } => args.ptr_mut(keys, stream).ptr_mut(values, stream),
        };
        args.ptr(&bias, stream)
            .ptr(&sinks, stream)
            .ptr_mut(&mut scratch.out, stream)
            .ptr_mut(&mut scratch.max, stream)
            .ptr_mut(&mut scratch.sum, stream)
            .ptr_mut(&mut scratch.counters, stream)
            .ptr_mut(&mut out, stream)
            .opt(gate, stream)
            .ptr(start, stream)
            .i32(i32::try_from(rows)?)
            .u32(slots)
            .u32(window)
            .u32(extent)
            .f32(SCALE);
        let config = LaunchConfig {
            grid_dim: (KV_HEADS as u32, ROWS_BLOCKS, 1),
            block_dim: (ROWS_THREADS, 1, 1),
            shared_mem_bytes: smem,
        };
        // SAFETY: the arguments match `attention_rows_bf16_<NT>(const bf16* q, const bf16*
        // k_new, const bf16* v_new, bf16* k_cache, bf16* v_cache, const bf16* bias, const
        // bf16* sinks, float* partial_out, float* partial_max, float* partial_sum, unsigned*
        // counters, bf16* out, const bf16* gate, const unsigned* start, int rows, unsigned
        // slots, unsigned window, unsigned extent, float scale)`, or `attention_rows_fp8_<NT>`
        // with `uint8_t*` caches. The kernel reads slots below min(start + rows, slots) of the
        // caches, which the caller checked the cache holds, and `rows` rows of q, k, v, bias,
        // sinks and gate (if any). It writes partials for `rows` x 24 query rows of
        // `ROWS_PARTS` partials, which the scratch holds, uses the scratch's 16 counters,
        // which every call leaves at zero, and writes `rows` rows of `out`. The combining
        // blocks write the rows' keys and values at slots (start + row) mod slots once every
        // block of their KV head has finished reading the cache; no two blocks write the same
        // elements, since rows <= 8 < slots. Before `pdl_wait` it reads only `start` and the
        // caches, which no kernel of the pass writes before this one.
        unsafe { args.launch_pdl(stream, function, config) }
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
            grid_dim: (KV_HEADS as u32, splits(cache.slots, MIN_SPLIT_SLOTS), 1),
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

    /// One layer's decode step from its projections to the input of its output projection,
    /// in one launch, with the results of these calls in sequence, bit for bit:
    /// `Norm::attention_post` (writing `k`, `v`, `q` and `r`), [`Attention::relative_bias`]
    /// of `r` with `s.table`, [`Attention::append`] of `k` and `v`,
    /// [`Attention::decode_attention`] with `q` and `s.sinks`, and
    /// `Elementwise::attention_gate` with `s.gate` into `s.out`. It updates the conv states
    /// and appends to `cache` as they do.
    ///
    /// The kernel launches with programmatic dependent launch and reads `position` before it
    /// waits for the previous kernel, so that kernel must not write it. The caller
    /// checks [`KvCache::holds`] for the position first.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the struct only bundles borrows for this call"
    )]
    pub fn decode_step(
        &self,
        stream: &CudaStream,
        cache: &mut KvCache,
        position: &CudaSlice<u32>,
        scratch: &mut AttentionScratch,
        s: DecodeStep<'_>,
    ) -> anyhow::Result<()> {
        check_len("k_raw", s.k_raw.len(), KV_ROW)?;
        check_len("v_raw", s.v_raw.len(), KV_ROW)?;
        check_len("k_state", s.k_state.len(), (CONV_TAPS - 1) * KV_ROW)?;
        check_len("v_state", s.v_state.len(), (CONV_TAPS - 1) * KV_ROW)?;
        for (name, x) in [("q_raw", s.q_raw), ("q_scale", s.q_scale), ("gate", s.gate)] {
            check_len(name, x.len(), HEADS * HEAD_DIM)?;
        }
        for (name, x) in [("r_raw", s.r_raw), ("r_scale", s.r_scale)] {
            check_len(name, x.len(), HEADS * REL_DIM)?;
        }
        check_len("sinks", s.sinks.len(), HEADS)?;
        check_len("out", s.out.len(), HEADS * HEAD_DIM)?;
        check_len("position", position.len(), 1)?;
        ensure!(
            [MIN_SPLIT_SLOTS, ONE_TILE_SPLIT_SLOTS].contains(&s.min_split_slots),
            "min_split_slots is {}, expected {MIN_SPLIT_SLOTS} or {ONE_TILE_SPLIT_SLOTS}",
            s.min_split_slots
        );
        let extent = extent("table", s.table.len(), REL_DIM)?;
        let slots = u32::try_from(cache.slots)?;
        let window = u32::try_from(cache.window.unwrap_or(0)).context("window too large")?;

        let mut args = KernelArgs::default();
        args.ptr(s.k_raw, stream).ptr(s.v_raw, stream);
        match &s.conv {
            KvConvWeights::Static { k, v } => {
                check_len("k conv weight", k.len(), KV_ROW * CONV_TAPS)?;
                check_len("v conv weight", v.len(), KV_ROW * CONV_TAPS)?;
                args.ptr(*k, stream)
                    .ptr(*v, stream)
                    .u64(0)
                    .u64(0)
                    .u64(0)
                    .u64(0);
            }
            KvConvWeights::Dynamic {
                k_dynamic,
                v_dynamic,
                k_static,
                v_static,
            } => {
                check_len("k_dynamic", k_dynamic.len(), KV_HEADS * CONV_TAPS)?;
                check_len("v_dynamic", v_dynamic.len(), KV_HEADS * CONV_TAPS)?;
                check_len("k_static", k_static.len(), CONV_TAPS * KV_ROW)?;
                check_len("v_static", v_static.len(), CONV_TAPS * KV_ROW)?;
                args.u64(0)
                    .u64(0)
                    .ptr(k_dynamic, stream)
                    .ptr(v_dynamic, stream)
                    .ptr(*k_static, stream)
                    .ptr(*v_static, stream);
            }
        }
        args.ptr_mut(s.k_state, stream)
            .ptr_mut(s.v_state, stream)
            .ptr(s.q_raw, stream)
            .ptr(s.q_scale, stream)
            .ptr(s.r_raw, stream)
            .ptr(s.r_scale, stream)
            .ptr(s.table, stream)
            .ptr(s.sinks, stream)
            .ptr(s.gate, stream);
        let function = match &mut cache.data {
            KvData::Bf16 { keys, values } => {
                args.ptr_mut(keys, stream).ptr_mut(values, stream);
                &self.decode_step
            }
            KvData::Fp8 { keys, values } => {
                args.ptr_mut(keys, stream).ptr_mut(values, stream);
                &self.decode_step_fp8
            }
        };
        let (log_scale, alpha, n_floor) = match s.log_scale {
            Some((alpha, n_floor)) => (1, alpha, n_floor),
            None => (0, 0.0, 1.0),
        };
        args.ptr_mut(&mut scratch.out, stream)
            .ptr_mut(&mut scratch.max, stream)
            .ptr_mut(&mut scratch.sum, stream)
            .ptr_mut(&mut scratch.counters, stream)
            .ptr_mut(s.out, stream)
            .ptr(position, stream)
            .i32(slots.cast_signed())
            .i32(window.cast_signed())
            .i32(extent.cast_signed())
            .i32(log_scale)
            .f32(alpha)
            .f32(n_floor)
            .f32(s.qk_threshold)
            .f32(s.rel_threshold)
            .f32(s.eps)
            .f32(SCALE);
        let config = LaunchConfig {
            grid_dim: (KV_HEADS as u32, splits(cache.slots, s.min_split_slots), 1),
            block_dim: (DECODE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `decode_step_bf16`'s signature, or `decode_step_fp8`'s
        // with `uint8_t*` caches for an FP8 cache; the unsigned scalars pass as their bits.
        // Every buffer has the length checked above, and the kernel reads no null pointer
        // (the static conv weights only without dynamic taps). It reads slots below
        // min(position + 1, slots) of the caches and writes slot position % slots, reads
        // table columns below `extent`, and writes partial rows for 24 heads x at most
        // `MAX_SPLITS` splits and 8 counters, which the scratch holds. Before `pdl_wait` it
        // reads only weights and the position, which the doc above keeps the previous kernel
        // from writing.
        unsafe { args.launch_pdl(stream, function, config) }
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
