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

use std::{
    mem::ManuallyDrop,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context as _, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, DevicePtr as _, LaunchConfig,
    PushKernelArg as _,
    sys::{self, CUdevice_attribute, CUfunction_attribute},
};
use half::bf16;

use crate::{
    gpu::{DeviceIn, DeviceOut, KernelArgs, cubin, load_module},
    ops::{
        Products,
        norm::KvConvWeights,
        spec::{
            BATCH_STATE_WORDS, BATCH_VERIFY_MAX_ROWS, BatchRows, PACKED_SLICE_WORDS, PackedRows,
        },
    },
    vmm::{Page, PagePool, Pooled},
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
pub const MAX_PREFILL_ROWS: usize = 4096;
/// Rows of (query row, head) pairs per prefill block: 8 warps of 16.
const PREFILL_BLOCK_ROWS: usize = 128;
/// Threads per prefill block.
const PREFILL_BLOCK: u32 = 256;
/// Dynamic shared memory of a prefill block, checked against the kernel's layout by
/// `static_assert`s in `attention.cu`. One block fits on an SM.
const PREFILL_SHARED_BYTES: u32 = 84_432;
/// Maximum global `block_m` blocks across up to `BATCH_ROWS` packed slices totaling at most
/// `MAX_PREFILL_ROWS` rows (`kPrefillPackedMaxBlocks` in `attention.cu`).
pub const PREFILL_PACKED_MAX_BLOCKS: usize = 192;
/// Dynamic shared memory of a packed prefill block, checked against `PrefillPackedShared`'s
/// `static_assert`s in `attention.cu`.
const PREFILL_PACKED_SHARED_BYTES: u32 = 85_248;
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
/// Split blocks per KV head of `attention_rows_*`, and the slots of each split.
const ROWS_SPLIT_BLOCKS: u32 = 32;
const ROWS_SPLIT_SLOTS: u32 = 64;
/// Most 8-column tiles of query rows: 3 heads x 8 rows.
const ROWS_QUERY_TILES: usize = 3;
/// Threads per block of `attention_rows_*`.
const ROWS_THREADS: u32 = 128;
/// Dynamic shared memory of `attention_rows_bf16`, `attention_rows_fp8`,
/// `attention_rows_fp4` and `attention_rows_fp4_g32`, the sizes of `RowsShared` in the kernel.
const ROWS_SHARED_BYTES: [u32; 4] = [32_768, 49_152, 27_648, 26_112];
/// Minimum batch sequence count at which [`BatchSplitMode::Auto`] switches from exact
/// full split-K (bit-identical to single-sequence [`Attention::attention_rows`]) to
/// adaptive SM-aware split-K. Keeps small batches (`seqs < 8`, including `B = 1..5`
/// bit-exact parity tests) bit-identical while avoiding over-splitting at `B >= 8`.
pub const BATCH_ADAPTIVE_MIN_SEQS: usize = 8;
/// Bytes of one slot of an NVFP4 key or value buffer: 8 heads of 64 bytes of E2M1 values,
/// then 8 heads of 8 E4M3 scales (`kFp4Row` in the kernel).
pub const FP4_ROW: usize = 576;
/// Bytes of one slot of a group-32 NVFP4 key or value buffer: 8 heads of 64 bytes of E2M1
/// values, then 8 heads of 4 E4M3 scales (`kFp4G32Row` in the kernel).
pub const FP4_G32_ROW: usize = 544;

/// Split-K policy for [`Attention::attention_batch`] and
/// [`Attention::attention_batch_with_mode`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchSplitMode {
    /// Exact full split-K for `seqs < BATCH_ADAPTIVE_MIN_SEQS` (preserving bit-exact
    /// parity with [`Attention::attention_rows`] for small batches) and SM-aware adaptive
    /// split-K (with the fused single-block no-split fast path when `split_blocks == 1`)
    /// for `seqs >= BATCH_ADAPTIVE_MIN_SEQS`.
    ///
    /// [`Attention::load`] defaults to [`Self::Exact`] instead when
    /// `SIPHON_EXACT_BATCH_ATTN=1` is set in the environment.
    #[default]
    Auto,
    /// Always uses `slots.div_ceil(64).clamp(1, 32)` split blocks, matching
    /// [`Attention::attention_rows`] bit for bit at any batch size.
    Exact,
    /// Always uses SM-aware adaptive split-K `(2 * sms / (seqs * 8)).clamp(1, max_splits)`,
    /// dispatching to the fused single-block no-split kernel when `split_blocks == 1`.
    Adaptive,
    /// Forces a specific split count in `1..=32` (clamped to `1..=max_splits`), using the
    /// fused single-block no-split kernel when the resulting split count is `1`.
    Fixed(u32),
}

/// Split blocks per KV head for a batched attention launch on a GPU with `sms` SMs.
///
/// Returns `1` when the fused single-block no-split kernel should run, or `2..=32` for
/// parameterized split-K.
#[must_use]
pub fn batch_split_blocks(sms: u32, seqs: usize, slots: usize, mode: BatchSplitMode) -> u32 {
    let slots = u32::try_from(slots).unwrap_or(u32::MAX);
    let max_splits = slots.div_ceil(ROWS_SPLIT_SLOTS).clamp(1, ROWS_SPLIT_BLOCKS);
    let adaptive = || {
        let kv_heads = u32::try_from(seqs.saturating_mul(KV_HEADS)).unwrap_or(u32::MAX);
        // Target 2 blocks/SM occupancy (matching `__launch_bounds__(128, 2)` and FP8's
        // 48 KiB SMEM per block). Floor division prevents tail-wave overflow past
        // `2 * sms` and transitions cleanly to `1` (fused single-block no-split) once
        // `seqs * 8 > sms`.
        ((2 * sms) / kv_heads.max(1)).clamp(1, max_splits)
    };
    match mode {
        BatchSplitMode::Auto => {
            if seqs < BATCH_ADAPTIVE_MIN_SEQS {
                max_splits
            } else {
                adaptive()
            }
        }
        BatchSplitMode::Exact => max_splits,
        BatchSplitMode::Adaptive => adaptive(),
        BatchSplitMode::Fixed(n) => n.clamp(1, max_splits),
    }
}

pub use siphon::{KvFormat, KvQuant};

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
    /// NVFP4, [`FP4_ROW`] bytes per slot (see `KvFormat::Nvfp4`).
    Nvfp4 {
        keys: CudaSlice<u8>,
        /// Value rows with their scales packed into each row.
        values: CudaSlice<u8>,
    },
    /// Group-32 NVFP4, [`FP4_G32_ROW`] bytes per slot (see `KvFormat::Nvfp4G32`).
    Nvfp4G32 {
        /// Key rows with their group-32 scales packed into each row.
        keys: CudaSlice<u8>,
        /// Value rows with their group-32 scales packed into each row.
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
/// One contiguous VMM reservation holding every sliding ring of a slot (or a prefix-cache
/// `LocalState`), so mapping, zeroing, copying and unmapping all sliding rings takes a single
/// `cuMemSetAccess` / `cuMemsetD8Async` / `cuMemcpyDtoDAsync_v2` / `cuMemUnmap` call instead
/// of one per sliding visit.
#[derive(Debug)]
pub struct RingSlab {
    pooled: Pooled,
    mapped: Arc<AtomicBool>,
    layout: siphon::memory::RingSlabLayout,
}

impl RingSlab {
    /// Reserves one contiguous VMM range for `ring_count` sliding rings of `slots` positions
    /// in `format` (`Bf16`, `Fp8`, or `Nvfp4`).
    pub fn reserve(
        pool: &PagePool,
        ring_count: usize,
        slots: usize,
        format: KvFormat,
    ) -> anyhow::Result<Self> {
        ensure!(ring_count > 0, "a ring slab needs at least one ring");
        ensure!(slots > 0, "a ring needs at least one slot");
        let buffer_bytes = match format {
            KvFormat::Bf16 => slots * KV_ROW * size_of::<bf16>(),
            KvFormat::Fp8 => slots * KV_ROW,
            KvFormat::Nvfp4 => slots * FP4_ROW,
            KvFormat::Nvfp4G32 => slots * FP4_G32_ROW,
            KvFormat::Mixed | KvFormat::Nvfp4All | KvFormat::Nvfp4G32All => {
                anyhow::bail!(
                    "sliding caches take a concrete format (BF16, FP8, NVFP4 or NVFP4-G32)"
                )
            }
        };
        let layout =
            siphon::memory::RingSlabLayout::new(ring_count, buffer_bytes, pool.page_bytes());
        let pooled = Pooled::reserve(pool, layout.total_bytes)?;
        Ok(Self {
            pooled,
            mapped: Arc::new(AtomicBool::new(false)),
            layout,
        })
    }

    /// Device pointers `(keys, values)` for sliding ring `index` inside this slab.
    #[must_use]
    pub fn ring_ptrs(&self, index: usize) -> (u64, u64) {
        let (k, v) = self.layout.ring_offsets(index);
        (self.pooled.ptr() + k as u64, self.pooled.ptr() + v as u64)
    }

    /// Shared flag reflecting whether the slab's pages are currently mapped.
    #[must_use]
    pub fn mapped_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.mapped)
    }

    /// Whether every ring in the slab is currently backed by pages.
    #[must_use]
    pub fn is_mapped(&self) -> bool {
        self.pooled.mapped_bytes() >= self.layout.total_bytes
    }

    /// Pages more that [`Self::map`] or [`Self::map_uninit`] would take from the pool.
    #[must_use]
    pub fn pages_needed(&self) -> usize {
        self.pooled.pages_to(self.layout.total_bytes)
    }

    /// Total pages the slab takes when mapped.
    #[must_use]
    pub fn pages_total(&self) -> usize {
        self.layout.pages
    }

    /// Pages currently held by this slab.
    #[must_use]
    pub fn held_pages(&self) -> usize {
        self.pooled.pages().len()
    }

    #[must_use]
    pub fn pool(&self) -> &PagePool {
        self.pooled.pool()
    }

    /// Live byte span of all packed rings (`ring_count * 2 * buffer_bytes`).
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.layout.ring_count * self.layout.ring_stride
    }

    /// Maps the whole slab in one `cuMemSetAccess` and zeroes it in one `cuMemsetD8Async`.
    pub fn map(&mut self, stream: &CudaStream) -> anyhow::Result<()> {
        self.pooled.map_to(self.layout.total_bytes, stream)?;
        self.mapped.store(true, Ordering::Release);
        Ok(())
    }

    /// Maps the whole slab in one `cuMemSetAccess` without zeroing (for callers that
    /// immediately overwrite every byte with `copy_to`).
    pub fn map_uninit(&mut self) -> anyhow::Result<()> {
        self.pooled.map_to_uninit(self.layout.total_bytes)?;
        self.mapped.store(true, Ordering::Release);
        Ok(())
    }

    /// Returns every page of the slab to the pool in one contiguous `cuMemUnmap`.
    pub fn unmap(&mut self) -> anyhow::Result<()> {
        self.pooled.unmap_from(0)?;
        self.mapped.store(false, Ordering::Release);
        Ok(())
    }

    /// Zeroes every mapped byte of the slab in one async memset.
    pub fn zero(&self, stream: &CudaStream) -> anyhow::Result<()> {
        self.pooled.zero_mapped(stream)
    }

    /// Copies all packed rings into `dst` in one async device-to-device transfer.
    pub fn copy_to(&self, dst: &mut Self, stream: &CudaStream) -> anyhow::Result<()> {
        ensure!(self.layout == dst.layout, "ring slab layouts differ");
        ensure!(
            self.is_mapped() && dst.is_mapped(),
            "ring slabs must be mapped before copying"
        );
        let bytes = self.bytes();
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: both slabs are mapped for at least `bytes` and live across the copy.
        unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                dst.pooled.ptr(),
                self.pooled.ptr(),
                bytes,
                stream.cu_stream(),
            )
        }
        .result()
        .context("copying sliding ring slab")?;
        Ok(())
    }

    /// Unmaps this slab and returns its physical pages so another slab can map them via
    /// [`Self::map_shared_pages`].
    pub fn take_pages(&mut self) -> anyhow::Result<Vec<Arc<Page>>> {
        let pages = self.pooled.pages().to_vec();
        self.unmap()?;
        Ok(pages)
    }

    /// Replaces any mapped pages in this slab with `pages` in one `cuMemSetAccess`.
    pub fn map_shared_pages(&mut self, pages: &[Arc<Page>]) -> anyhow::Result<()> {
        self.unmap()?;
        if !pages.is_empty() {
            self.pooled.map_shared(pages)?;
            self.mapped.store(self.is_mapped(), Ordering::Release);
        }
        Ok(())
    }
}

/// How a [`KvCache`]'s device memory is owned.
#[derive(Debug)]
enum KvBacking {
    /// Ordinary `CudaSlice` allocations owned by `data`.
    Owned,
    /// Per-cache VMM reservations owned directly by this `KvCache`.
    Pooled(Vec<Pooled>),
    /// Contiguous VMM slab owned externally by a [`RingSlab`] (dropped after the cache).
    Slab(Arc<AtomicBool>),
}

/// cache at position `p` holds the tokens at positions below `p`.
#[derive(Debug)]
pub struct KvCache {
    /// The buffers. With `Pooled` or `Slab`, their addresses come from VMM reservations,
    /// which own the memory, and dropping the cache leaks them.
    data: ManuallyDrop<KvData>,
    slots: usize,
    window: Option<usize>,
    backing: KvBacking,
}

impl Drop for KvCache {
    fn drop(&mut self) {
        // SAFETY: `data` is not used again.
        let data = unsafe { ManuallyDrop::take(&mut self.data) };
        if matches!(self.backing, KvBacking::Owned) {
            return;
        }
        // The VMM reservation owns the addresses; it unmaps and frees them after this.
        match data {
            KvData::Bf16 { keys, values } => {
                let _ = (keys.leak(), values.leak());
            }
            KvData::Fp8 { keys, values }
            | KvData::Nvfp4 { keys, values }
            | KvData::Nvfp4G32 { keys, values } => {
                let _ = (keys.leak(), values.leak());
            }
        }
    }
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
        Self::sliding_ring_with_format(stream, window, slots, KvFormat::Bf16)
    }

    /// [`KvCache::sliding_ring`] storing `format` (`Bf16`, `Fp8` or `Nvfp4`) elements.
    pub fn sliding_ring_with_format(
        stream: &Arc<CudaStream>,
        window: usize,
        slots: usize,
        format: KvFormat,
    ) -> anyhow::Result<Self> {
        ensure!(
            slots >= window,
            "a ring of {slots} slots cannot hold a window of {window}"
        );
        Self::new(stream, slots, Some(window), format)
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

    /// [`KvCache::global_with_format`] at addresses reserved for `capacity` positions,
    /// backed by pages of `pool` only as [`KvCache::map`] asks.
    pub fn global_pooled(
        stream: &Arc<CudaStream>,
        capacity: usize,
        format: KvFormat,
        pool: &PagePool,
    ) -> anyhow::Result<Self> {
        Self::new_pooled(stream, capacity, None, format, pool)
    }

    /// A pooled cache of `capacity` slots. A ring whose key buffer is not a whole number
    /// of pages keeps its keys and values in one reservation, so that an FP8 ring of 1 MiB
    /// buffers or an NVFP4 ring of 576 KiB buffers takes one 2 MiB page, not two.
    fn new_pooled(
        stream: &Arc<CudaStream>,
        capacity: usize,
        window: Option<usize>,
        format: KvFormat,
        pool: &PagePool,
    ) -> anyhow::Result<Self> {
        ensure!(capacity > 0, "a KV cache needs at least one slot");
        u32::try_from(capacity).context("too many KV cache slots")?;
        let (len, bytes) = match format {
            KvFormat::Bf16 => (capacity * KV_ROW, capacity * KV_ROW * size_of::<bf16>()),
            KvFormat::Fp8 => (capacity * KV_ROW, capacity * KV_ROW),
            KvFormat::Nvfp4 => (capacity * FP4_ROW, capacity * FP4_ROW),
            KvFormat::Nvfp4G32 => (capacity * FP4_G32_ROW, capacity * FP4_G32_ROW),
            KvFormat::Mixed | KvFormat::Nvfp4All | KvFormat::Nvfp4G32All => {
                anyhow::bail!("each cache takes a concrete format (BF16, FP8, NVFP4 or NVFP4-G32)")
            }
        };
        let packed = window.is_some() && !bytes.is_multiple_of(pool.page_bytes());
        let (pooled, keys, values) = if packed {
            let both = Pooled::reserve(pool, 2 * bytes)?;
            let (keys, values) = (both.ptr(), both.ptr() + bytes as u64);
            (vec![both], keys, values)
        } else {
            let keys = Pooled::reserve(pool, bytes)?;
            let values = Pooled::reserve(pool, bytes)?;
            let ptrs = (keys.ptr(), values.ptr());
            (vec![keys, values], ptrs.0, ptrs.1)
        };
        // SAFETY: each address starts `len` elements of a reservation that lives as long
        // as the cache; `Drop` leaks the slices, so cudarc never frees them. Kernels touch
        // only positions below `mapped`, which `map` backs first.
        let data = unsafe {
            match format {
                KvFormat::Bf16 => KvData::Bf16 {
                    keys: stream.upgrade_device_ptr(keys, len),
                    values: stream.upgrade_device_ptr(values, len),
                },
                KvFormat::Fp8 => KvData::Fp8 {
                    keys: stream.upgrade_device_ptr(keys, len),
                    values: stream.upgrade_device_ptr(values, len),
                },
                KvFormat::Nvfp4 => KvData::Nvfp4 {
                    keys: stream.upgrade_device_ptr(keys, len),
                    values: stream.upgrade_device_ptr(values, len),
                },
                KvFormat::Nvfp4G32 => KvData::Nvfp4G32 {
                    keys: stream.upgrade_device_ptr(keys, len),
                    values: stream.upgrade_device_ptr(values, len),
                },
                KvFormat::Mixed | KvFormat::Nvfp4All | KvFormat::Nvfp4G32All => {
                    unreachable!("refused above")
                }
            }
        };
        Ok(Self {
            data: ManuallyDrop::new(data),
            slots: capacity,
            window,
            backing: KvBacking::Pooled(pooled),
        })
    }

    /// A pooled sliding ring whose keys and values sit at slot `index` inside a shared
    /// [`RingSlab`]. The caller must ensure `slab` outlives this `KvCache`.
    pub fn sliding_in_slab(
        stream: &Arc<CudaStream>,
        window: usize,
        slots: usize,
        format: KvFormat,
        slab: &RingSlab,
        index: usize,
    ) -> anyhow::Result<Self> {
        ensure!(
            slots >= window,
            "a ring of {slots} slots cannot hold a window of {window}"
        );
        ensure!(
            matches!(
                format,
                KvFormat::Bf16 | KvFormat::Fp8 | KvFormat::Nvfp4 | KvFormat::Nvfp4G32
            ),
            "sliding caches take a concrete format (BF16, FP8, NVFP4 or NVFP4-G32)"
        );
        u32::try_from(slots).context("too many KV cache slots")?;
        let (keys, values) = slab.ring_ptrs(index);
        // SAFETY: `keys` and `values` point into `slab`'s reservation, which outlives this
        // cache; `Drop` leaks the slices so cudarc never frees them.
        let data = unsafe {
            match format {
                KvFormat::Bf16 => KvData::Bf16 {
                    keys: stream.upgrade_device_ptr(keys, slots * KV_ROW),
                    values: stream.upgrade_device_ptr(values, slots * KV_ROW),
                },
                KvFormat::Fp8 => KvData::Fp8 {
                    keys: stream.upgrade_device_ptr(keys, slots * KV_ROW),
                    values: stream.upgrade_device_ptr(values, slots * KV_ROW),
                },
                KvFormat::Nvfp4 => KvData::Nvfp4 {
                    keys: stream.upgrade_device_ptr(keys, slots * FP4_ROW),
                    values: stream.upgrade_device_ptr(values, slots * FP4_ROW),
                },
                KvFormat::Nvfp4G32 => KvData::Nvfp4G32 {
                    keys: stream.upgrade_device_ptr(keys, slots * FP4_G32_ROW),
                    values: stream.upgrade_device_ptr(values, slots * FP4_G32_ROW),
                },
                KvFormat::Mixed | KvFormat::Nvfp4All | KvFormat::Nvfp4G32All => {
                    unreachable!("refused above")
                }
            }
        };
        Ok(Self {
            data: ManuallyDrop::new(data),
            slots,
            window: Some(window),
            backing: KvBacking::Slab(slab.mapped_flag()),
        })
    }

    /// [`KvCache::sliding_ring`] at addresses reserved for the ring, backed by pages of
    /// `pool` only while [`KvCache::map`] keeps it mapped: a slot without a sequence holds
    /// none. A ring maps whole or not at all.
    pub fn sliding_pooled(
        stream: &Arc<CudaStream>,
        window: usize,
        slots: usize,
        pool: &PagePool,
    ) -> anyhow::Result<Self> {
        Self::sliding_pooled_with_format(stream, window, slots, KvFormat::Bf16, pool)
    }

    /// [`KvCache::sliding_pooled`] storing `format` (`Bf16`, `Fp8` or `Nvfp4`) elements. An
    /// FP8 or NVFP4 ring's keys and values share pages: one reservation holds the keys, then
    /// the values.
    pub fn sliding_pooled_with_format(
        stream: &Arc<CudaStream>,
        window: usize,
        slots: usize,
        format: KvFormat,
        pool: &PagePool,
    ) -> anyhow::Result<Self> {
        ensure!(
            slots >= window,
            "a ring of {slots} slots cannot hold a window of {window}"
        );
        ensure!(
            matches!(
                format,
                KvFormat::Bf16 | KvFormat::Fp8 | KvFormat::Nvfp4 | KvFormat::Nvfp4G32
            ),
            "sliding caches are BF16, FP8, NVFP4 or NVFP4-G32"
        );
        Self::new_pooled(stream, slots, Some(window), format, pool)
    }

    /// Bytes of each reservation that back the positions below `positions`: of a packed
    /// ring's one reservation, both buffers whole, since its values follow its keys.
    fn pooled_bytes(&self, positions: usize) -> usize {
        let bytes = positions.min(self.slots) * self.row_bytes();
        match &self.backing {
            KvBacking::Pooled(p) if p.len() == 1 && bytes > 0 => 2 * self.slots * self.row_bytes(),
            _ => bytes,
        }
    }

    /// Whether the cache's memory comes from a page pool (directly or via a [`RingSlab`]).
    #[must_use]
    pub fn is_pooled(&self) -> bool {
        !matches!(self.backing, KvBacking::Owned)
    }

    /// Positions whose memory is backed: every slot unless pooled.
    #[must_use]
    pub fn mapped(&self) -> usize {
        match &self.backing {
            KvBacking::Pooled(p) => match p.as_slice() {
                [both] if both.mapped_bytes() >= self.pooled_bytes(self.slots) => self.slots,
                [_] => 0,
                [keys, ..] => (keys.mapped_bytes() / self.row_bytes()).min(self.slots),
                [] => self.slots,
            },
            KvBacking::Slab(mapped) => {
                if mapped.load(Ordering::Acquire) {
                    self.slots
                } else {
                    0
                }
            }
            KvBacking::Owned => self.slots,
        }
    }

    /// Pages more that [`KvCache::map`] of `positions` would take from the pool.
    #[must_use]
    pub fn pages_to(&self, positions: usize) -> usize {
        let bytes = self.pooled_bytes(positions);
        match &self.backing {
            KvBacking::Pooled(p) => p.iter().map(|b| b.pages_to(bytes)).sum(),
            KvBacking::Owned | KvBacking::Slab(_) => 0,
        }
    }

    /// Backs the positions below `positions` (at most the capacity) with pages, zeroed on
    /// `stream`. Does nothing for a cache that is not directly pooled.
    pub fn map(&mut self, positions: usize, stream: &CudaStream) -> anyhow::Result<()> {
        let bytes = self.pooled_bytes(positions);
        if let KvBacking::Pooled(pooled) = &mut self.backing {
            let more: usize = pooled.iter().map(|b| b.pages_to(bytes)).sum();
            let free = pooled[0].pool().free_pages();
            ensure!(
                more <= free,
                "the KV pool is out of pages: {more} more needed for {positions} positions, \
                 {free} free"
            );
            for b in pooled {
                b.map_to(bytes, stream)?;
            }
        }
        Ok(())
    }

    /// Returns the pages past the first `positions` to the pool. Queued work must not
    /// touch those positions any more. Does nothing for a cache that is not directly pooled.
    pub fn unmap_from(&mut self, positions: usize) -> anyhow::Result<()> {
        let bytes = self.pooled_bytes(positions);
        if let KvBacking::Pooled(pooled) = &mut self.backing {
            for b in pooled {
                b.unmap_from(bytes)?;
            }
        }
        Ok(())
    }

    /// The directly pooled reservations: the keys' and the values', or a packed ring's one.
    #[must_use]
    pub fn pooled(&self) -> Option<&[Pooled]> {
        match &self.backing {
            KvBacking::Pooled(p) => Some(p.as_slice()),
            KvBacking::Owned | KvBacking::Slab(_) => None,
        }
    }

    /// The directly pooled reservations, to map shared pages into.
    pub fn pooled_mut(&mut self) -> Option<&mut [Pooled]> {
        match &mut self.backing {
            KvBacking::Pooled(p) => Some(p.as_mut_slice()),
            KvBacking::Owned | KvBacking::Slab(_) => None,
        }
    }

    /// Pages of the pool that the cache takes while it holds the positions below
    /// `positions`, 0 unless directly pooled.
    #[must_use]
    pub fn pages_for(&self, positions: usize) -> usize {
        let bytes = self.pooled_bytes(positions);
        match &self.backing {
            KvBacking::Pooled(p) => p
                .iter()
                .map(|b| bytes.min(b.reserved_bytes()).div_ceil(b.page_bytes()))
                .sum(),
            KvBacking::Owned | KvBacking::Slab(_) => 0,
        }
    }

    fn new(
        stream: &Arc<CudaStream>,
        slots: usize,
        window: Option<usize>,
        format: KvFormat,
    ) -> anyhow::Result<Self> {
        ensure!(slots > 0, "a KV cache needs at least one slot");
        u32::try_from(slots).context("too many KV cache slots")?;
        ensure!(
            window.is_none()
                || matches!(
                    format,
                    KvFormat::Bf16 | KvFormat::Fp8 | KvFormat::Nvfp4 | KvFormat::Nvfp4G32
                ),
            "sliding caches are BF16, FP8, NVFP4 or NVFP4-G32"
        );
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
            KvFormat::Nvfp4 => KvData::Nvfp4 {
                keys: stream.alloc_zeros(slots * FP4_ROW)?,
                values: stream.alloc_zeros(slots * FP4_ROW)?,
            },
            KvFormat::Nvfp4G32 => KvData::Nvfp4G32 {
                keys: stream.alloc_zeros(slots * FP4_G32_ROW)?,
                values: stream.alloc_zeros(slots * FP4_G32_ROW)?,
            },
            KvFormat::Mixed | KvFormat::Nvfp4All | KvFormat::Nvfp4G32All => {
                anyhow::bail!("each cache takes a concrete format (BF16, FP8, NVFP4 or NVFP4-G32)")
            }
        };
        Ok(Self {
            data: ManuallyDrop::new(data),
            slots,
            window,
            backing: KvBacking::Owned,
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
        let (keys, values) = match &*self.data {
            KvData::Bf16 { keys, values } => {
                (keys.device_ptr(stream).0, values.device_ptr(stream).0)
            }
            KvData::Fp8 { keys, values }
            | KvData::Nvfp4 { keys, values }
            | KvData::Nvfp4G32 { keys, values } => {
                (keys.device_ptr(stream).0, values.device_ptr(stream).0)
            }
        };
        let row_bytes = self.row_bytes();
        let bytes = self.slots * row_bytes;
        Some(([(keys, bytes), (values, bytes)], row_bytes))
    }

    /// The sliding window, or `None` for a global layer.
    #[must_use]
    pub fn window(&self) -> Option<usize> {
        self.window
    }

    #[must_use]
    pub fn format(&self) -> KvFormat {
        match *self.data {
            KvData::Bf16 { .. } => KvFormat::Bf16,
            KvData::Fp8 { .. } => KvFormat::Fp8,
            KvData::Nvfp4 { .. } => KvFormat::Nvfp4,
            KvData::Nvfp4G32 { .. } => KvFormat::Nvfp4G32,
        }
    }

    /// Bytes of one slot in each of the key and value buffers.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        match *self.data {
            KvData::Bf16 { .. } => KV_ROW * size_of::<bf16>(),
            KvData::Fp8 { .. } => KV_ROW,
            KvData::Nvfp4 { .. } => FP4_ROW,
            KvData::Nvfp4G32 { .. } => FP4_G32_ROW,
        }
    }

    /// Elements of one slot in each of the key and value buffers: 1,024 values, or for
    /// NVFP4 [`FP4_ROW`] / [`FP4_G32_ROW`] bytes.
    #[must_use]
    pub fn row_len(&self) -> usize {
        match *self.data {
            KvData::Nvfp4 { .. } => FP4_ROW,
            KvData::Nvfp4G32 { .. } => FP4_G32_ROW,
            _ => KV_ROW,
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

    /// Zeroes every slot, of a pooled cache every mapped one: it keeps its pages, which
    /// the next sequence then needs not map again ([`KvCache::unmap_from`] returns them).
    pub fn clear(&mut self, stream: &Arc<CudaStream>) -> anyhow::Result<()> {
        let len = self.mapped() * self.row_len();
        match &mut *self.data {
            KvData::Bf16 { keys, values } => {
                stream.memset_zeros(&mut keys.slice_mut(..len))?;
                stream.memset_zeros(&mut values.slice_mut(..len))?;
            }
            KvData::Fp8 { keys, values }
            | KvData::Nvfp4 { keys, values }
            | KvData::Nvfp4G32 { keys, values } => {
                stream.memset_zeros(&mut keys.slice_mut(..len))?;
                stream.memset_zeros(&mut values.slice_mut(..len))?;
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
/// `slots`: at most [`SPLITS`] (capped at `2 * sms / KV_HEADS` so one 2-block/SM wave covers
/// the device), or fewer so that each covers at least `min_slots`.
fn splits(sms: u32, slots: usize, min_slots: usize) -> u32 {
    let max_splits = ((2 * sms as usize) / KV_HEADS).clamp(1, SPLITS);
    let splits = slots.div_ceil(min_slots).clamp(1, max_splits);
    debug_assert!(splits <= MAX_SPLITS);
    splits as u32
}

/// Prefill blocks per KV head for a chunk of `rows`.
#[must_use]
pub fn prefill_blocks(rows: usize) -> usize {
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
    decode_step_fp4: CudaFunction,
    decode_step_fp4_g32: CudaFunction,
    kv_append_rows_fp4: CudaFunction,
    kv_append_rows_fp4_g32: CudaFunction,
    kv_append_rows_packed: CudaFunction,
    kv_append_rows_packed_fp8: CudaFunction,
    kv_append_rows_packed_fp4: CudaFunction,
    kv_append_rows_packed_fp4_g32: CudaFunction,
    prefill_fp4: CudaFunction,
    prefill_fp4_g32: CudaFunction,
    /// `prefill_attention_fp4_mma`, for [`Products::Fp4`] on an NVFP4 cache.
    prefill_fp4_mma: CudaFunction,
    prefill_fp4_g32_mma: CudaFunction,
    prefill: CudaFunction,
    prefill_fp8: CudaFunction,
    /// `prefill_attention_fp8_mma`, for [`Products::Fp8`].
    prefill_fp8_mma: CudaFunction,
    prefill_packed_fp4: CudaFunction,
    prefill_packed_fp4_g32: CudaFunction,
    prefill_packed: CudaFunction,
    prefill_packed_fp8: CudaFunction,
    prefill_packed_fp8_mma: CudaFunction,
    /// `attention_rows_{bf16,fp8,fp4,fp4_g32}_<NT>` for NT = 1 to 3.
    rows: Vec<[CudaFunction; 4]>,
    /// `attention_rows_tree_{bf16,fp8,fp4,fp4_g32}_<NT>` for NT = 1 to 3.
    rows_tree: Vec<[CudaFunction; 4]>,
    /// `attention_batch_{bf16,fp8,fp4,fp4_g32}`.
    batch: [CudaFunction; 4],
    /// `attention_batch_verify_{bf16,fp8,fp4,fp4_g32}_<NT>` for NT = 1 to 3.
    batch_verify: Vec<[CudaFunction; 4]>,
    /// `attention_batch_nosplit_{bf16,fp8,fp4,fp4_g32}_<NT>` for NT = 1 to 3.
    batch_nosplit: Vec<[CudaFunction; 4]>,
    /// Device streaming multiprocessor count, used to size adaptive split-K grids.
    sms: u32,
    /// Default split-K policy used by [`Attention::attention_batch`].
    batch_split_mode: BatchSplitMode,
    kv_round: CudaFunction,
    kv_round_channels: CudaFunction,
}

/// A 4-bit rounding of the global caches' keys and values, applied before they are stored,
/// to measure its cost in quality. The caches stay BF16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvRounding {
    pub quant: KvQuant,
    /// Rounds each head in the Hadamard basis (see `kv_round_bf16`).
    pub rotate: bool,
    /// The visits whose writes are rounded, one bit per visit index.
    pub visits: u64,
    /// Keeps the newest positions, at least this many, rounded to E4M3 only and rounds them
    /// to 4 bits as they age, before each prefill chunk (prefill only; 0 for none).
    pub recent: usize,
    /// Rounds aged keys with one scale per channel and group of positions, as KIVI does,
    /// instead of per token. Needs `recent`.
    pub keys_per_channel: bool,
}

impl Default for KvRounding {
    fn default() -> Self {
        Self {
            quant: KvQuant::None,
            rotate: false,
            visits: u64::MAX,
            recent: 0,
            keys_per_channel: false,
        }
    }
}

impl KvRounding {
    /// Whether visit `visit`'s keys and values are rounded.
    #[must_use]
    pub fn applies(&self, visit: usize) -> bool {
        self.quant != KvQuant::None && visit < 64 && (self.visits >> visit) & 1 == 1
    }

    /// The kernel's mode for aged rows: the 4-bit format.
    fn mode(self) -> i32 {
        match self.quant {
            KvQuant::None => 0,
            KvQuant::Nvfp4 => 1,
            KvQuant::Q4_0 => 2,
            KvQuant::Nvfp4G32 => 4,
        }
    }

    /// The kernel's mode for rows as they are written: E4M3 while they are recent, the
    /// 4-bit format otherwise.
    fn write_mode(self) -> i32 {
        if self.recent > 0 { 3 } else { self.mode() }
    }

    /// Rows per group of [`Self::keys_per_channel`].
    #[must_use]
    pub fn channel_rows(self) -> usize {
        if self.quant == KvQuant::Nvfp4 { 16 } else { 32 }
    }
}

/// The partials of [`Attention::attention_rows`] and [`Attention::attention_batch`]: per
/// query row of up to 8 x 24, one output, max and sum per block of its KV head. The
/// counters, two per KV head and batch row, count the finished blocks and combiners, and
/// return to zero at the end of every call.
#[derive(Debug)]
pub struct RowsAttentionScratch {
    /// Rows the partials and counters serve.
    rows: usize,
    out: CudaSlice<f32>,
    max: CudaSlice<f32>,
    sum: CudaSlice<f32>,
    counters: CudaSlice<u32>,
}

impl RowsAttentionScratch {
    /// Partials and counters for calls of up to `rows` rows: at least [`ROWS_MAX`] for
    /// [`Attention::attention_rows`], up to [`BATCH_VERIFY_MAX_ROWS`] for
    /// [`Attention::attention_batch`].
    pub fn new(stream: &Arc<CudaStream>, rows: usize) -> anyhow::Result<Self> {
        ensure!(
            (1..=BATCH_VERIFY_MAX_ROWS).contains(&rows),
            "{rows} rows, expected 1 to {BATCH_VERIFY_MAX_ROWS}"
        );
        let rows = rows.max(ROWS_MAX);
        let queries = rows * HEADS * ROWS_PARTS;
        Ok(Self {
            rows,
            out: stream.alloc_zeros(queries * HEAD_DIM)?,
            max: stream.alloc_zeros(queries)?,
            sum: stream.alloc_zeros(queries)?,
            counters: stream.alloc_zeros(2 * KV_HEADS * rows)?,
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
    #[expect(
        clippy::too_many_lines,
        reason = "loads all attention, packed-prefill, and adaptive split-K kernel entry points"
    )]
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
        let prefill_fp8_mma = module.load_function("prefill_attention_fp8_mma")?;
        let prefill_fp4 = module.load_function("prefill_attention_fp4")?;
        let prefill_fp4_g32 = module.load_function("prefill_attention_fp4_g32")?;
        let prefill_fp4_mma = module.load_function("prefill_attention_fp4_mma")?;
        let prefill_fp4_g32_mma = module.load_function("prefill_attention_fp4_g32_mma")?;
        for function in [
            &prefill,
            &prefill_fp8,
            &prefill_fp8_mma,
            &prefill_fp4,
            &prefill_fp4_g32,
            &prefill_fp4_mma,
            &prefill_fp4_g32_mma,
        ] {
            function.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                PREFILL_SHARED_BYTES.cast_signed(),
            )?;
        }
        let sms = u32::try_from(
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?,
        )?;
        let batch_split_mode = match std::env::var("SIPHON_EXACT_BATCH_ATTN").as_deref() {
            Ok("1" | "true" | "TRUE" | "yes" | "YES") => BatchSplitMode::Exact,
            _ => BatchSplitMode::Auto,
        };
        let prefill_packed = module.load_function("prefill_attention_packed_bf16")?;
        let prefill_packed_fp8 = module.load_function("prefill_attention_packed_fp8")?;
        let prefill_packed_fp8_mma = module.load_function("prefill_attention_packed_fp8_mma")?;
        let prefill_packed_fp4 = module.load_function("prefill_attention_packed_fp4")?;
        let prefill_packed_fp4_g32 = module.load_function("prefill_attention_packed_fp4_g32")?;
        for function in [
            &prefill_packed,
            &prefill_packed_fp8,
            &prefill_packed_fp8_mma,
            &prefill_packed_fp4,
            &prefill_packed_fp4_g32,
        ] {
            function.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                PREFILL_PACKED_SHARED_BYTES.cast_signed(),
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
            decode_step_fp4: module.load_function("decode_step_fp4")?,
            decode_step_fp4_g32: module.load_function("decode_step_fp4_g32")?,
            kv_append_rows_fp4: module.load_function("kv_append_rows_fp4")?,
            kv_append_rows_fp4_g32: module.load_function("kv_append_rows_fp4_g32")?,
            kv_append_rows_packed: module.load_function("kv_append_rows_packed_bf16")?,
            kv_append_rows_packed_fp8: module.load_function("kv_append_rows_packed_fp8")?,
            kv_append_rows_packed_fp4: module.load_function("kv_append_rows_packed_fp4")?,
            kv_append_rows_packed_fp4_g32: module.load_function("kv_append_rows_packed_fp4_g32")?,
            prefill_fp4,
            prefill_fp4_g32,
            prefill_fp4_mma,
            prefill_fp4_g32_mma,
            prefill,
            prefill_fp8,
            prefill_fp8_mma,
            prefill_packed_fp4,
            prefill_packed_fp4_g32,
            prefill_packed,
            prefill_packed_fp8,
            prefill_packed_fp8_mma,
            rows: (1..=ROWS_QUERY_TILES)
                .map(|nt| {
                    Ok([
                        rows_function(&format!("attention_rows_bf16_{nt}"), ROWS_SHARED_BYTES[0])?,
                        rows_function(&format!("attention_rows_fp8_{nt}"), ROWS_SHARED_BYTES[1])?,
                        rows_function(&format!("attention_rows_fp4_{nt}"), ROWS_SHARED_BYTES[2])?,
                        rows_function(
                            &format!("attention_rows_fp4_g32_{nt}"),
                            ROWS_SHARED_BYTES[3],
                        )?,
                    ])
                })
                .collect::<anyhow::Result<_>>()?,
            rows_tree: (1..=ROWS_QUERY_TILES)
                .map(|nt| {
                    Ok([
                        rows_function(
                            &format!("attention_rows_tree_bf16_{nt}"),
                            ROWS_SHARED_BYTES[0],
                        )?,
                        rows_function(
                            &format!("attention_rows_tree_fp8_{nt}"),
                            ROWS_SHARED_BYTES[1],
                        )?,
                        rows_function(
                            &format!("attention_rows_tree_fp4_{nt}"),
                            ROWS_SHARED_BYTES[2],
                        )?,
                        rows_function(
                            &format!("attention_rows_tree_fp4_g32_{nt}"),
                            ROWS_SHARED_BYTES[3],
                        )?,
                    ])
                })
                .collect::<anyhow::Result<_>>()?,
            batch: [
                rows_function("attention_batch_bf16", ROWS_SHARED_BYTES[0])?,
                rows_function("attention_batch_fp8", ROWS_SHARED_BYTES[1])?,
                rows_function("attention_batch_fp4", ROWS_SHARED_BYTES[2])?,
                rows_function("attention_batch_fp4_g32", ROWS_SHARED_BYTES[3])?,
            ],
            batch_verify: (1..=ROWS_QUERY_TILES)
                .map(|nt| {
                    Ok([
                        rows_function(
                            &format!("attention_batch_verify_bf16_{nt}"),
                            ROWS_SHARED_BYTES[0],
                        )?,
                        rows_function(
                            &format!("attention_batch_verify_fp8_{nt}"),
                            ROWS_SHARED_BYTES[1],
                        )?,
                        rows_function(
                            &format!("attention_batch_verify_fp4_{nt}"),
                            ROWS_SHARED_BYTES[2],
                        )?,
                        rows_function(
                            &format!("attention_batch_verify_fp4_g32_{nt}"),
                            ROWS_SHARED_BYTES[3],
                        )?,
                    ])
                })
                .collect::<anyhow::Result<_>>()?,
            batch_nosplit: (1..=ROWS_QUERY_TILES)
                .map(|nt| {
                    Ok([
                        rows_function(
                            &format!("attention_batch_nosplit_bf16_{nt}"),
                            ROWS_SHARED_BYTES[0],
                        )?,
                        rows_function(
                            &format!("attention_batch_nosplit_fp8_{nt}"),
                            ROWS_SHARED_BYTES[1],
                        )?,
                        rows_function(
                            &format!("attention_batch_nosplit_fp4_{nt}"),
                            ROWS_SHARED_BYTES[2],
                        )?,
                        rows_function(
                            &format!("attention_batch_nosplit_fp4_g32_{nt}"),
                            ROWS_SHARED_BYTES[3],
                        )?,
                    ])
                })
                .collect::<anyhow::Result<_>>()?,
            sms,
            batch_split_mode,
            kv_round: module.load_function("kv_round_bf16")?,
            kv_round_channels: module.load_function("kv_round_channels_bf16")?,
        })
    }

    /// Streaming multiprocessor count of the context this [`Attention`] loaded on.
    #[must_use]
    pub fn sms(&self) -> u32 {
        self.sms
    }

    /// Default split-K policy used by [`Attention::attention_batch`].
    #[must_use]
    pub fn batch_split_mode(&self) -> BatchSplitMode {
        self.batch_split_mode
    }

    /// Overrides the default split-K policy used by [`Attention::attention_batch`].
    pub fn set_batch_split_mode(&mut self, mode: BatchSplitMode) {
        self.batch_split_mode = mode;
    }

    /// Rounds `rows` rows of keys `k` and values `v` (`[rows, 8, 128]`) as `rounding` writes
    /// them, in place.
    pub fn round_rows(
        &self,
        stream: &CudaStream,
        k: &mut impl DeviceOut<bf16>,
        v: &mut impl DeviceOut<bf16>,
        rows: usize,
        rounding: KvRounding,
    ) -> anyhow::Result<()> {
        let (mut k, mut v) = (k.view_mut(), v.view_mut());
        check_len("k", k.len(), rows * KV_ROW)?;
        check_len("v", v.len(), rows * KV_ROW)?;
        let mut args = KernelArgs::default();
        args.ptr_mut(&mut k, stream)
            .ptr_mut(&mut v, stream)
            .u64(0)
            .u32(0)
            .i32(rounding.write_mode())
            .i32(i32::from(rounding.rotate))
            .i32(3);
        self.launch_round(stream, &args, rows)
    }

    /// Rounds the row of `cache` at the position in the device word `position` as
    /// `rounding` writes it, in place: what a decode step appended.
    pub fn round_cache_row(
        &self,
        stream: &CudaStream,
        cache: &mut KvCache,
        position: u64,
        rounding: KvRounding,
    ) -> anyhow::Result<()> {
        ensure!(cache.window.is_none(), "only global caches are rounded");
        let KvData::Bf16 { keys, values } = &mut *cache.data else {
            anyhow::bail!("4-bit rounding needs a BF16 cache");
        };
        let mut args = KernelArgs::default();
        args.ptr_mut(keys, stream)
            .ptr_mut(values, stream)
            .u64(position)
            .u32(0)
            .i32(rounding.write_mode())
            .i32(i32::from(rounding.rotate))
            .i32(3);
        self.launch_round(stream, &args, 1)
    }

    /// Rounds rows `first .. first + rows` of `cache` to `rounding`'s 4-bit format as they
    /// age out of its recent window: values per token, keys per token or, with
    /// [`KvRounding::keys_per_channel`], per channel over groups of
    /// [`KvRounding::channel_rows`] (`first` and `rows` multiples of it).
    pub fn age_rows(
        &self,
        stream: &CudaStream,
        cache: &mut KvCache,
        first: usize,
        rows: usize,
        rounding: KvRounding,
    ) -> anyhow::Result<()> {
        ensure!(first + rows <= cache.slots, "rows past the cache");
        let KvData::Bf16 { keys, values } = &mut *cache.data else {
            anyhow::bail!("4-bit rounding needs a BF16 cache");
        };
        let mut args = KernelArgs::default();
        args.ptr_mut(keys, stream)
            .ptr_mut(values, stream)
            .u64(0)
            .u32(u32::try_from(first)?)
            .i32(rounding.mode())
            .i32(i32::from(rounding.rotate))
            .i32(if rounding.keys_per_channel { 2 } else { 3 });
        self.launch_round(stream, &args, rows)?;
        if rounding.keys_per_channel {
            let group = rounding.channel_rows();
            ensure!(
                first.is_multiple_of(group) && rows.is_multiple_of(group),
                "per-channel rows come in groups of {group}"
            );
            let groups = rows / group;
            let mut args = KernelArgs::default();
            args.ptr_mut(keys, stream)
                .u32(u32::try_from(first)?)
                .u32(u32::try_from(groups)?)
                .i32(rounding.mode());
            let config = LaunchConfig {
                grid_dim: (u32::try_from((groups * KV_ROW).div_ceil(256))?, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: the arguments match `kv_round_channels_bf16(bf16*, unsigned, unsigned,
            // int)`; the groups lie inside the cache (checked above), and the kernel waits
            // for earlier kernels before reading.
            unsafe { args.launch_pdl(stream, &self.kv_round_channels, config) }?;
        }
        Ok(())
    }

    fn launch_round(
        &self,
        stream: &CudaStream,
        args: &KernelArgs,
        rows: usize,
    ) -> anyhow::Result<()> {
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows)?, 2, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments match `kv_round_bf16(bf16*, bf16*, const unsigned*, unsigned,
        // int, int, int)`; every row the grid names lies inside k and v (the caller checks the
        // cache holds the position), and the kernel calls `pdl_wait()` before its first read
        // and never triggers its successor early.
        unsafe { args.launch_pdl(stream, &self.kv_round, config) }
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
        let (function, smem) = match *cache.data {
            KvData::Bf16 { .. } => (&entry[0], ROWS_SHARED_BYTES[0]),
            KvData::Fp8 { .. } => (&entry[1], ROWS_SHARED_BYTES[1]),
            KvData::Nvfp4 { .. } => (&entry[2], ROWS_SHARED_BYTES[2]),
            KvData::Nvfp4G32 { .. } => (&entry[3], ROWS_SHARED_BYTES[3]),
        };
        let mut args = KernelArgs::default();
        args.ptr(&q, stream).ptr(&k, stream).ptr(&v, stream);
        match &mut *cache.data {
            KvData::Bf16 { keys, values } => args.ptr_mut(keys, stream).ptr_mut(values, stream),
            KvData::Fp8 { keys, values }
            | KvData::Nvfp4 { keys, values }
            | KvData::Nvfp4G32 { keys, values } => {
                args.ptr_mut(keys, stream).ptr_mut(values, stream)
            }
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
        let rows_blocks = if cfg!(sm89) {
            slots
                .div_ceil(ROWS_SPLIT_SLOTS)
                .clamp(12, ROWS_SPLIT_BLOCKS)
                + 2
        } else {
            ROWS_BLOCKS
        };
        let config = LaunchConfig {
            grid_dim: (KV_HEADS as u32, rows_blocks, 1),
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

    /// Tree-verify variant of [`Self::attention_rows`]: row `r` sits at sequence position
    /// `*start + tree_depth[r]`, attends over committed prefix slots below `*start` plus
    /// its own tree ancestors `tree_attn_anc[r * 8 .. r * 8 + 8]`, and stages its native
    /// K and V into `k_stage` and `v_stage` (`[8, cache.row_bytes()]`) instead of mutating
    /// `cache`. `Spec::kv_commit_tree` commits the winning branch after acceptance.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors attention_rows plus tree topology and staging buffers"
    )]
    pub fn attention_rows_tree(
        &self,
        stream: &CudaStream,
        q: &impl DeviceIn<bf16>,
        k: &impl DeviceIn<bf16>,
        v: &impl DeviceIn<bf16>,
        cache: &KvCache,
        start: &CudaSlice<u32>,
        rows: usize,
        bias: &impl DeviceIn<bf16>,
        sinks: &impl DeviceIn<bf16>,
        gate: Option<&CudaView<'_, bf16>>,
        scratch: &mut RowsAttentionScratch,
        out: &mut impl DeviceOut<bf16>,
        tree_depth: &CudaSlice<u32>,
        tree_attn_anc: &CudaSlice<i32>,
        k_stage: &mut CudaSlice<u8>,
        v_stage: &mut CudaSlice<u8>,
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
        check_len("tree_depth", tree_depth.len(), ROWS_MAX)?;
        check_len("tree_attn_anc", tree_attn_anc.len(), ROWS_MAX * ROWS_MAX)?;
        check_len("k_stage", k_stage.len(), ROWS_MAX * cache.row_bytes())?;
        check_len("v_stage", v_stage.len(), ROWS_MAX * cache.row_bytes())?;
        let extent = extent("bias", bias.len(), rows * HEADS)?;
        let slots = u32::try_from(cache.slots)?;
        let window = u32::try_from(cache.window.unwrap_or(0)).context("window too large")?;
        let entry = &self.rows_tree[(3 * rows).div_ceil(8) - 1];
        let (function, smem) = match *cache.data {
            KvData::Bf16 { .. } => (&entry[0], ROWS_SHARED_BYTES[0]),
            KvData::Fp8 { .. } => (&entry[1], ROWS_SHARED_BYTES[1]),
            KvData::Nvfp4 { .. } => (&entry[2], ROWS_SHARED_BYTES[2]),
            KvData::Nvfp4G32 { .. } => (&entry[3], ROWS_SHARED_BYTES[3]),
        };
        let mut args = KernelArgs::default();
        args.ptr(&q, stream).ptr(&k, stream).ptr(&v, stream);
        match &*cache.data {
            KvData::Bf16 { keys, values } => args.ptr(keys, stream).ptr(values, stream),
            KvData::Fp8 { keys, values }
            | KvData::Nvfp4 { keys, values }
            | KvData::Nvfp4G32 { keys, values } => args.ptr(keys, stream).ptr(values, stream),
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
            .f32(SCALE)
            .ptr(tree_depth, stream)
            .ptr(tree_attn_anc, stream)
            .ptr_mut(k_stage, stream)
            .ptr_mut(v_stage, stream);
        let rows_blocks = if cfg!(sm89) {
            slots
                .div_ceil(ROWS_SPLIT_SLOTS)
                .clamp(12, ROWS_SPLIT_BLOCKS)
                + 2
        } else {
            ROWS_BLOCKS
        };
        let config = LaunchConfig {
            grid_dim: (KV_HEADS as u32, rows_blocks, 1),
            block_dim: (ROWS_THREADS, 1, 1),
            shared_mem_bytes: smem,
        };
        // SAFETY: matches `attention_rows_tree_{bf16,fp8,fp4}_<NT>`; `k_cache` and
        // `v_cache` are read-only below `min(*start, slots)`, `k_stage` and `v_stage` hold
        // 8 rows of `cache.row_bytes()` bytes in the cache's native format, and only
        // `start`, `k_cache`, `v_cache` are read before `pdl_wait`.
        unsafe { args.launch_pdl(stream, function, config) }
    }

    /// [`Attention::attention_rows`] for the rows of a batched step or batched verify, each
    /// sequence attending from its queries into its own cache at its own position and
    /// appending its keys and values there. `batch` names each sequence's caches and
    /// position; `like` is any of the sequences' caches for this visit, whose format, slots
    /// and window every sequence's cache shares. Every per-row buffer is `[rows, ...]` as in
    /// [`Attention::attention_rows`].
    ///
    /// Uses [`Attention::batch_split_mode`] ([`BatchSplitMode::Auto`] by default, or
    /// [`BatchSplitMode::Exact`] when `SIPHON_EXACT_BATCH_ATTN=1` is set).
    #[expect(clippy::too_many_arguments, reason = "mirrors attention_rows")]
    pub fn attention_batch(
        &self,
        stream: &CudaStream,
        q: &impl DeviceIn<bf16>,
        k: &impl DeviceIn<bf16>,
        v: &impl DeviceIn<bf16>,
        like: &KvCache,
        batch: &BatchRows<'_>,
        rows: usize,
        bias: &impl DeviceIn<bf16>,
        sinks: &impl DeviceIn<bf16>,
        gate: Option<&CudaView<'_, bf16>>,
        scratch: &mut RowsAttentionScratch,
        out: &mut impl DeviceOut<bf16>,
    ) -> anyhow::Result<()> {
        self.attention_batch_with_mode(
            stream,
            q,
            k,
            v,
            like,
            batch,
            rows,
            bias,
            sinks,
            gate,
            scratch,
            out,
            self.batch_split_mode,
        )
    }

    /// [`Attention::attention_batch`] with an explicit [`BatchSplitMode`].
    ///
    /// When [`batch_split_blocks`] selects `1`, dispatches to the fused single-block
    /// no-split kernel (`attention_batch_nosplit_*`), avoiding all partial-buffer reads,
    /// writes, and atomic counters. Otherwise launches parameterized split-K with
    /// `split_blocks` split CTAs per `(seq, kv_head)` and single-block combine.
    #[expect(clippy::too_many_arguments, reason = "mirrors attention_rows")]
    pub fn attention_batch_with_mode(
        &self,
        stream: &CudaStream,
        q: &impl DeviceIn<bf16>,
        k: &impl DeviceIn<bf16>,
        v: &impl DeviceIn<bf16>,
        like: &KvCache,
        batch: &BatchRows<'_>,
        rows: usize,
        bias: &impl DeviceIn<bf16>,
        sinks: &impl DeviceIn<bf16>,
        gate: Option<&CudaView<'_, bf16>>,
        scratch: &mut RowsAttentionScratch,
        out: &mut impl DeviceOut<bf16>,
        mode: BatchSplitMode,
    ) -> anyhow::Result<()> {
        let (q, k, v, bias, sinks) = (q.view(), k.view(), v.view(), bias.view(), sinks.view());
        let mut out = out.view_mut();
        if let Some(gate) = gate {
            check_len("gate", gate.len(), rows * HEADS * HEAD_DIM)?;
        }
        ensure!(
            (1..=scratch.rows).contains(&rows),
            "{rows} rows, the scratch holds {}",
            scratch.rows
        );
        let rows_per_seq = batch.rows_per_seq;
        ensure!(
            (1..=ROWS_MAX).contains(&rows_per_seq) && rows.is_multiple_of(rows_per_seq),
            "{rows} rows is not a multiple of rows_per_seq {rows_per_seq} (1..={ROWS_MAX})"
        );
        let seqs = rows / rows_per_seq;
        check_len("q", q.len(), rows * HEADS * HEAD_DIM)?;
        check_len("k", k.len(), rows * KV_ROW)?;
        check_len("v", v.len(), rows * KV_ROW)?;
        check_len("sinks", sinks.len(), rows * HEADS)?;
        check_len("out", out.len(), rows * HEADS * HEAD_DIM)?;
        check_len("states", batch.states.len(), seqs * BATCH_STATE_WORDS)?;
        check_len("positions", batch.positions.len(), seqs)?;
        let extent = extent("bias", bias.len(), rows * HEADS)?;
        let slots = u32::try_from(like.slots)?;
        let window = u32::try_from(like.window.unwrap_or(0)).context("window too large")?;
        let split_blocks = batch_split_blocks(self.sms, seqs, like.slots, mode);
        let format_idx = match *like.data {
            KvData::Bf16 { .. } => 0,
            KvData::Fp8 { .. } => 1,
            KvData::Nvfp4 { .. } => 2,
            KvData::Nvfp4G32 { .. } => 3,
        };
        let smem = ROWS_SHARED_BYTES[format_idx];
        let nt_idx = (3 * rows_per_seq).div_ceil(8) - 1;
        if split_blocks == 1 {
            return self.launch_batch_nosplit(
                stream,
                nt_idx,
                format_idx,
                &q,
                &k,
                &v,
                batch,
                &bias,
                &sinks,
                &mut out,
                gate,
                seqs,
                rows_per_seq,
                slots,
                window,
                extent,
            );
        }
        let mut args = KernelArgs::default();
        args.ptr(&q, stream)
            .ptr(&k, stream)
            .ptr(&v, stream)
            .ptr(&batch.states, stream)
            .ptr(&bias, stream)
            .ptr(&sinks, stream)
            .ptr_mut(&mut scratch.out, stream)
            .ptr_mut(&mut scratch.max, stream)
            .ptr_mut(&mut scratch.sum, stream)
            .ptr_mut(&mut scratch.counters, stream)
            .ptr_mut(&mut out, stream)
            .opt(gate, stream)
            .ptr(&batch.positions, stream);
        let (function, grid_y) = if rows_per_seq == 1 {
            args.u32(slots)
                .u32(window)
                .u32(extent)
                .f32(SCALE)
                .i32(split_blocks.cast_signed());
            (&self.batch[format_idx], split_blocks + 1)
        } else {
            args.i32(i32::try_from(rows_per_seq)?)
                .u32(slots)
                .u32(window)
                .u32(extent)
                .f32(SCALE)
                .i32(split_blocks.cast_signed());
            let recent_blocks = u32::try_from(rows_per_seq.div_ceil(4))?;
            (
                &self.batch_verify[nt_idx][format_idx],
                split_blocks + recent_blocks,
            )
        };
        let config = LaunchConfig {
            grid_dim: (KV_HEADS as u32, grid_y, u32::try_from(seqs)?),
            block_dim: (ROWS_THREADS, 1, 1),
            shared_mem_bytes: smem,
        };
        // SAFETY: matches `attention_batch_{bf16,fp8,fp4}` (when `rows_per_seq == 1`) or
        // `attention_batch_verify_{bf16,fp8,fp4}_<NT>`; sequence `z` attends over
        // `rows_per_seq` rows at `positions[z]` in its own cache, uses partials within
        // `scratch.rows`, and single-block combine leaves every counter at zero without
        // spin-waiting.
        unsafe { args.launch_pdl(stream, function, config) }
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors attention_batch_nosplit parameters"
    )]
    fn launch_batch_nosplit(
        &self,
        stream: &CudaStream,
        nt_idx: usize,
        format_idx: usize,
        q: &CudaView<'_, bf16>,
        k: &CudaView<'_, bf16>,
        v: &CudaView<'_, bf16>,
        batch: &BatchRows<'_>,
        bias: &CudaView<'_, bf16>,
        sinks: &CudaView<'_, bf16>,
        out: &mut cudarc::driver::CudaViewMut<'_, bf16>,
        gate: Option<&CudaView<'_, bf16>>,
        seqs: usize,
        rows_per_seq: usize,
        slots: u32,
        window: u32,
        extent: u32,
    ) -> anyhow::Result<()> {
        let function = &self.batch_nosplit[nt_idx][format_idx];
        let smem = ROWS_SHARED_BYTES[format_idx];
        let mut args = KernelArgs::default();
        args.ptr(q, stream)
            .ptr(k, stream)
            .ptr(v, stream)
            .ptr(&batch.states, stream)
            .ptr(bias, stream)
            .ptr(sinks, stream)
            .ptr_mut(out, stream)
            .opt(gate, stream)
            .ptr(&batch.positions, stream)
            .i32(i32::try_from(rows_per_seq)?)
            .u32(slots)
            .u32(window)
            .u32(extent)
            .f32(SCALE);
        let config = LaunchConfig {
            grid_dim: (KV_HEADS as u32, 1, u32::try_from(seqs)?),
            block_dim: (ROWS_THREADS, 1, 1),
            shared_mem_bytes: smem,
        };
        // SAFETY: matches `attention_batch_nosplit_{bf16,fp8,fp4}_<NT>`; each CTA owns
        // one `(kv_head, seq)` pair exclusively, reads all cache slots and recent keys,
        // combines in shared memory/registers without touching `scratch`, writes `out`,
        // and appends `k`/`v` after synchronizing its warps.
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
        let function = match *cache.data {
            KvData::Bf16 { .. } => &self.kv_append,
            KvData::Fp8 { .. } => &self.kv_append_fp8,
            KvData::Nvfp4 { .. } | KvData::Nvfp4G32 { .. } => {
                anyhow::bail!("NVFP4 caches append through the fused decode step")
            }
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&k).arg(&v);
        match &mut *cache.data {
            KvData::Bf16 { keys, values } => launch.arg(keys).arg(values),
            KvData::Fp8 { keys, values }
            | KvData::Nvfp4 { keys, values }
            | KvData::Nvfp4G32 { keys, values } => launch.arg(keys).arg(values),
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

        let function = match *cache.data {
            KvData::Bf16 { .. } => &self.decode,
            KvData::Fp8 { .. } => &self.decode_fp8,
            KvData::Nvfp4 { .. } | KvData::Nvfp4G32 { .. } => {
                anyhow::bail!("NVFP4 caches attend through the fused decode step")
            }
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&q);
        match &*cache.data {
            KvData::Bf16 { keys, values } => launch.arg(keys).arg(values),
            KvData::Fp8 { keys, values }
            | KvData::Nvfp4 { keys, values }
            | KvData::Nvfp4G32 { keys, values } => launch.arg(keys).arg(values),
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
            grid_dim: (
                KV_HEADS as u32,
                splits(self.sms, cache.slots, MIN_SPLIT_SLOTS),
                1,
            ),
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
    #[expect(clippy::too_many_lines, reason = "one argument per kernel parameter")]
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
        let function = match &mut *cache.data {
            KvData::Bf16 { keys, values } => {
                args.ptr_mut(keys, stream).ptr_mut(values, stream);
                &self.decode_step
            }
            KvData::Fp8 { keys, values } => {
                args.ptr_mut(keys, stream).ptr_mut(values, stream);
                &self.decode_step_fp8
            }
            KvData::Nvfp4 { keys, values } => {
                args.ptr_mut(keys, stream).ptr_mut(values, stream);
                &self.decode_step_fp4
            }
            KvData::Nvfp4G32 { keys, values } => {
                args.ptr_mut(keys, stream).ptr_mut(values, stream);
                &self.decode_step_fp4_g32
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
            grid_dim: (
                KV_HEADS as u32,
                splits(self.sms, cache.slots, s.min_split_slots),
                1,
            ),
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
        start: &impl DeviceIn<u32>,
        rows: usize,
        k: &CudaView<'_, bf16>,
        v: &CudaView<'_, bf16>,
    ) -> anyhow::Result<()> {
        let start = start.view();
        let slots = u32::try_from(cache.slots)?;
        // One thread per 8 values, or per group of 16 / 32 for NVFP4 / NVFP4-G32.
        let threads = match cache.format() {
            KvFormat::Nvfp4 => KV_ROW / 16,
            KvFormat::Nvfp4G32 => KV_ROW / 32,
            _ => KV_ROW / 8,
        };
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows)?, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let function = match *cache.data {
            KvData::Bf16 { .. } => &self.kv_append_rows,
            KvData::Fp8 { .. } => &self.kv_append_rows_fp8,
            KvData::Nvfp4 { .. } => &self.kv_append_rows_fp4,
            KvData::Nvfp4G32 { .. } => &self.kv_append_rows_fp4_g32,
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(k).arg(v);
        match &mut *cache.data {
            KvData::Bf16 { keys, values } => launch.arg(keys).arg(values),
            KvData::Fp8 { keys, values }
            | KvData::Nvfp4 { keys, values }
            | KvData::Nvfp4G32 { keys, values } => launch.arg(keys).arg(values),
        };
        launch.arg(&start).arg(&slots);
        // SAFETY: the arguments match `kv_append_rows_bf16(const bf16*, const bf16*, bf16*,
        // bf16*, const unsigned*, unsigned)`, or `kv_append_rows_fp8` and `kv_append_rows_fp4`
        // with `uint8_t*` caches.
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
    /// With [`Products::Fp8`], an FP8 cache's products run on the FP8 tensor cores
    /// (`prefill_attention_fp8_mma`): the queries and the softmax weights round to E4M3.
    /// With [`Products::Fp4`], an NVFP4 cache runs `Q K^T` directly on Blackwell's native
    /// NVFP4 tensor cores (`prefill_attention_fp4_mma`, `m16n8k64`) and `P V` on FP8 tensor
    /// cores (`m16n8k32`), while an FP8 cache uses `prefill_attention_fp8_mma`. A BF16 cache
    /// ignores both.
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
        start: &impl DeviceIn<u32>,
        rows: usize,
        bias: &impl DeviceIn<bf16>,
        sinks: &impl DeviceIn<bf16>,
        scratch: &mut PrefillScratch,
        out: &mut impl DeviceOut<bf16>,
        products: Products,
    ) -> anyhow::Result<()> {
        let (q, k, v, bias, sinks) = (q.view(), k.view(), v.view(), bias.view(), sinks.view());
        let start = start.view();
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
        if rows == 1 && !matches!(cache.format(), KvFormat::Nvfp4 | KvFormat::Nvfp4G32) {
            // Decode attention streams the cache faster than a block of 128 rows with 3
            // used ones, and computes the same thing.
            self.append(stream, cache, &start, &k, &v)?;
            return self.decode_attention(
                stream,
                &q,
                cache,
                &start,
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
            input_keys == 0
                || matches!(
                    cache.format(),
                    KvFormat::Bf16 | KvFormat::Fp8 | KvFormat::Nvfp4 | KvFormat::Nvfp4G32
                ),
            "sliding caches are BF16, FP8, NVFP4 or NVFP4-G32"
        );
        ensure!(
            crate::BLOCK_SCALED_MMA
                || !matches!(
                    (&*cache.data, products),
                    (
                        KvData::Fp8 { .. } | KvData::Nvfp4 { .. } | KvData::Nvfp4G32 { .. },
                        Products::Fp4
                    )
                ),
            "{products} prefill attention needs the block-scaled MMA, which this sm_89 build lacks"
        );
        if input_keys == 0 {
            self.append_rows(stream, cache, &start, rows, &k, &v)?;
        }

        let function = match (&*cache.data, products) {
            (KvData::Bf16 { .. }, _) => &self.prefill,
            (KvData::Fp8 { .. }, Products::Bf16) => &self.prefill_fp8,
            (KvData::Fp8 { .. }, Products::Fp8 | Products::Fp4) => &self.prefill_fp8_mma,
            // On sm_89, `prefill_attention_fp4[_g32]_mma` decodes staged NVFP4 keys and values to
            // E4M3 and runs both Q K^T and P V on native Ada FP8 tensor cores (330 TF/s).
            (KvData::Nvfp4 { .. }, Products::Fp8) if !crate::BLOCK_SCALED_MMA => {
                &self.prefill_fp4_mma
            }
            (KvData::Nvfp4G32 { .. }, Products::Fp8) if !crate::BLOCK_SCALED_MMA => {
                &self.prefill_fp4_g32_mma
            }
            (KvData::Nvfp4 { .. }, Products::Bf16 | Products::Fp8) => &self.prefill_fp4,
            (KvData::Nvfp4 { .. }, Products::Fp4) => &self.prefill_fp4_mma,
            (KvData::Nvfp4G32 { .. }, Products::Bf16 | Products::Fp8) => &self.prefill_fp4_g32,
            (KvData::Nvfp4G32 { .. }, Products::Fp4) => &self.prefill_fp4_g32_mma,
        };
        let null_table: u64 = 0;
        let mut launch = stream.launch_builder(function);
        launch.arg(&q).arg(&k).arg(&v);
        match &*cache.data {
            KvData::Bf16 { keys, values } => launch.arg(keys).arg(values),
            KvData::Fp8 { keys, values }
            | KvData::Nvfp4 { keys, values }
            | KvData::Nvfp4G32 { keys, values } => launch.arg(keys).arg(values),
        };
        launch
            .arg(&bias)
            .arg(&sinks)
            .arg(&mut scratch.partials)
            .arg(&mut scratch.counters)
            .arg(&mut out)
            .arg(&start)
            .arg(&rows_u32)
            .arg(&slots)
            .arg(&window)
            .arg(&extent)
            .arg(&input_keys)
            .arg(&SCALE)
            .arg(&null_table);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(scratch.blocks)?, 1, 1),
            block_dim: (PREFILL_BLOCK, 1, 1),
            shared_mem_bytes: PREFILL_SHARED_BYTES,
        };
        // SAFETY: the arguments match `prefill_attention_bf16(const bf16* q, const bf16* k_in,
        // const bf16* v_in, const bf16* k_cache, const bf16* v_cache, const bf16* bias, const
        // bf16* sinks, float* partials, unsigned* counters, bf16* out, const unsigned* start,
        // unsigned rows, unsigned slots, unsigned window, unsigned extent, unsigned
        // input_keys, float scale, const bf16* rel_table_t)`, or `prefill_attention_fp8`,
        // `prefill_attention_fp8_mma`, `prefill_attention_fp4` and `prefill_attention_fp4_mma`
        // with `uint8_t*` caches. The checks above size `q`, `k`, `v`, `bias`, `sinks` and `out`
        // for `rows` rows. The kernel reads input rows below `rows`, cache slots below `slots`
        // and bias columns below `extent`. Its `scratch.blocks` blocks write at most two partial
        // slots each, which `scratch.partials` holds, and `prefill_blocks(rows) * 8` counters,
        // which `scratch.counters` holds since `rows` is at most `scratch.max_rows`; the shared
        // memory size matches the kernel's `static_assert`s.
        unsafe { launch.launch(config) }?;

        if input_keys == 1 {
            self.append_rows(stream, cache, &start, rows, &k, &v)?;
        }
        Ok(())
    }

    /// [`Self::prefill_attention`] with fused on-the-fly relative bias (`rel @ rel_table_t^T`
    /// via `mma_bf16`), eliminating `relative_bias_rows` and bias DRAM traffic.
    ///
    /// `rel` is `[rows, 24, 16]` BF16 and `rel_table_t` is `[extent, 16]` BF16 (the transpose
    /// of `rel_table`).
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn prefill_attention_fused_bias(
        &self,
        stream: &CudaStream,
        q: &impl DeviceIn<bf16>,
        k: &impl DeviceIn<bf16>,
        v: &impl DeviceIn<bf16>,
        cache: &mut KvCache,
        start: &impl DeviceIn<u32>,
        rows: usize,
        rel: &impl DeviceIn<bf16>,
        rel_table_t: &impl DeviceIn<bf16>,
        extent: usize,
        sinks: &impl DeviceIn<bf16>,
        scratch: &mut PrefillScratch,
        out: &mut impl DeviceOut<bf16>,
        products: Products,
    ) -> anyhow::Result<()> {
        let (q, k, v, rel, rel_table_t, sinks) = (
            q.view(),
            k.view(),
            v.view(),
            rel.view(),
            rel_table_t.view(),
            sinks.view(),
        );
        let start = start.view();
        let mut out = out.view_mut();
        ensure!(
            (1..=scratch.max_rows).contains(&rows),
            "prefill scratch takes 1 to {} rows, not {rows}",
            scratch.max_rows
        );
        check_len("q", q.len(), rows * HEADS * HEAD_DIM)?;
        check_len("k", k.len(), rows * KV_ROW)?;
        check_len("v", v.len(), rows * KV_ROW)?;
        check_len("rel", rel.len(), rows * HEADS * REL_DIM)?;
        check_len("rel_table_t", rel_table_t.len(), extent * REL_DIM)?;
        check_len("sinks", sinks.len(), rows * HEADS)?;
        check_len("out", out.len(), rows * HEADS * HEAD_DIM)?;
        check_len("start", start.len(), 1)?;
        ensure!(
            extent > 0 && extent.is_multiple_of(8),
            "prefill needs a bias extent that is a positive multiple of 8, not {extent}"
        );
        let extent_u32 = u32::try_from(extent).context("extent too large")?;
        let slots = u32::try_from(cache.slots)?;
        let window = u32::try_from(cache.window.unwrap_or(0)).context("window too large")?;
        let rows_u32 = u32::try_from(rows)?;
        let input_keys = u32::from(cache.window.is_some());
        ensure!(
            input_keys == 0
                || matches!(
                    cache.format(),
                    KvFormat::Bf16 | KvFormat::Fp8 | KvFormat::Nvfp4 | KvFormat::Nvfp4G32
                ),
            "sliding caches are BF16, FP8, NVFP4 or NVFP4-G32"
        );
        ensure!(
            crate::BLOCK_SCALED_MMA
                || !matches!(
                    (&*cache.data, products),
                    (
                        KvData::Fp8 { .. } | KvData::Nvfp4 { .. } | KvData::Nvfp4G32 { .. },
                        Products::Fp4
                    )
                ),
            "{products} prefill attention needs the block-scaled MMA, which this sm_89 build lacks"
        );
        if input_keys == 0 {
            self.append_rows(stream, cache, &start, rows, &k, &v)?;
        }

        let function = match (&*cache.data, products) {
            (KvData::Bf16 { .. }, _) => &self.prefill,
            (KvData::Fp8 { .. }, Products::Bf16) => &self.prefill_fp8,
            (KvData::Fp8 { .. }, Products::Fp8 | Products::Fp4) => &self.prefill_fp8_mma,
            (KvData::Nvfp4 { .. }, Products::Fp8) if !crate::BLOCK_SCALED_MMA => {
                &self.prefill_fp4_mma
            }
            (KvData::Nvfp4G32 { .. }, Products::Fp8) if !crate::BLOCK_SCALED_MMA => {
                &self.prefill_fp4_g32_mma
            }
            (KvData::Nvfp4 { .. }, Products::Bf16 | Products::Fp8) => &self.prefill_fp4,
            (KvData::Nvfp4 { .. }, Products::Fp4) => &self.prefill_fp4_mma,
            (KvData::Nvfp4G32 { .. }, Products::Bf16 | Products::Fp8) => &self.prefill_fp4_g32,
            (KvData::Nvfp4G32 { .. }, Products::Fp4) => &self.prefill_fp4_g32_mma,
        };
        let mut launch = stream.launch_builder(function);
        launch.arg(&q).arg(&k).arg(&v);
        match &*cache.data {
            KvData::Bf16 { keys, values } => launch.arg(keys).arg(values),
            KvData::Fp8 { keys, values }
            | KvData::Nvfp4 { keys, values }
            | KvData::Nvfp4G32 { keys, values } => launch.arg(keys).arg(values),
        };
        launch
            .arg(&rel)
            .arg(&sinks)
            .arg(&mut scratch.partials)
            .arg(&mut scratch.counters)
            .arg(&mut out)
            .arg(&start)
            .arg(&rows_u32)
            .arg(&slots)
            .arg(&window)
            .arg(&extent_u32)
            .arg(&input_keys)
            .arg(&SCALE)
            .arg(&rel_table_t);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(scratch.blocks)?, 1, 1),
            block_dim: (PREFILL_BLOCK, 1, 1),
            shared_mem_bytes: PREFILL_SHARED_BYTES,
        };
        // SAFETY: matches `prefill_attention_*` with non-null `rel_table_t`; `rel` holds
        // `[rows, HEADS, REL_DIM]` and `rel_table_t` holds `[extent, REL_DIM]`.
        unsafe { launch.launch(config) }?;
        if input_keys == 1 {
            self.append_rows(stream, cache, &start, rows, &k, &v)?;
        }
        Ok(())
    }

    /// Appends all `rows` packed tokens across `packed.num_slices` slices into each slice's
    /// own KV cache (words 0 and 1 of `packed.states`) in one launch. Every slice must share
    /// the cache `format` and `slots` of `cache`.
    pub fn append_rows_packed(
        &self,
        stream: &CudaStream,
        cache: &KvCache,
        packed: &PackedRows<'_>,
        rows: usize,
        k: &impl DeviceIn<bf16>,
        v: &impl DeviceIn<bf16>,
    ) -> anyhow::Result<()> {
        let (k, v) = (k.view(), v.view());
        ensure!(rows > 0, "no rows");
        ensure!(packed.num_slices > 0, "no packed slices");
        check_len("k", k.len(), rows * KV_ROW)?;
        check_len("v", v.len(), rows * KV_ROW)?;
        check_len(
            "slices",
            packed.slices.len(),
            packed.num_slices * PACKED_SLICE_WORDS,
        )?;
        ensure!(
            !packed.states.is_empty() && packed.states.len().is_multiple_of(BATCH_STATE_WORDS),
            "states has {} elements, not a positive multiple of {BATCH_STATE_WORDS}",
            packed.states.len()
        );
        let slots = u32::try_from(cache.slots)?;
        let num_slices = i32::try_from(packed.num_slices)?;
        let threads = match cache.format() {
            KvFormat::Nvfp4 => KV_ROW / 16,
            KvFormat::Nvfp4G32 => KV_ROW / 32,
            _ => KV_ROW / 8,
        };
        let config = LaunchConfig {
            grid_dim: (u32::try_from(rows)?, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let function = match *cache.data {
            KvData::Bf16 { .. } => &self.kv_append_rows_packed,
            KvData::Fp8 { .. } => &self.kv_append_rows_packed_fp8,
            KvData::Nvfp4 { .. } => &self.kv_append_rows_packed_fp4,
            KvData::Nvfp4G32 { .. } => &self.kv_append_rows_packed_fp4_g32,
        };
        let mut launch = stream.launch_builder(function);
        launch
            .arg(&k)
            .arg(&v)
            .arg(&packed.states)
            .arg(&packed.slices)
            .arg(&num_slices)
            .arg(&slots);
        // SAFETY: the arguments match `kv_append_rows_packed_{bf16,fp8,fp4}(const bf16*,
        // const bf16*, const unsigned long long*, const PackedSlice*, int, unsigned)`. Each
        // row finds its slice in `slices`, appends into that slice's live, distinct KV caches
        // (words 0 and 1 of `states`) of `slots` slots and `cache.format()`, and skips
        // overwritten rows when `row_len > slots`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Launches ONLY the ragged packed flash prefill attention kernel (without KV append)
    /// for the global query blocks listed in `block_descs`. Use
    /// [`Attention::prefill_attention_packed`] for the standard append + attention pair, or
    /// call [`Attention::append_rows_packed`] once and this method twice when a packed FP8
    /// `Auto` batch mixes slices below and above `FP8_ATTENTION_KEYS`.
    #[expect(clippy::too_many_arguments, reason = "mirrors the kernel inputs")]
    pub fn prefill_attention_packed_kernel(
        &self,
        stream: &CudaStream,
        q: &impl DeviceIn<bf16>,
        k: &impl DeviceIn<bf16>,
        v: &impl DeviceIn<bf16>,
        cache: &KvCache,
        packed: &PackedRows<'_>,
        block_descs: &CudaView<'_, u32>,
        rows: usize,
        bias: &impl DeviceIn<bf16>,
        sinks: &impl DeviceIn<bf16>,
        scratch: &mut PrefillScratch,
        counters: &mut CudaSlice<u32>,
        out: &mut impl DeviceOut<bf16>,
        products: Products,
    ) -> anyhow::Result<()> {
        let (q, k, v, bias, sinks) = (q.view(), k.view(), v.view(), bias.view(), sinks.view());
        let mut out = out.view_mut();
        ensure!(
            (1..=scratch.max_rows).contains(&rows),
            "prefill scratch takes 1 to {} rows, not {rows}",
            scratch.max_rows
        );
        ensure!(packed.num_slices > 0, "no packed slices");
        let total_blocks_m = block_descs.len();
        ensure!(
            (1..=PREFILL_PACKED_MAX_BLOCKS).contains(&total_blocks_m),
            "packed prefill takes 1 to {PREFILL_PACKED_MAX_BLOCKS} query blocks, not {total_blocks_m}"
        );
        check_len("q", q.len(), rows * HEADS * HEAD_DIM)?;
        check_len("k", k.len(), rows * KV_ROW)?;
        check_len("v", v.len(), rows * KV_ROW)?;
        check_len("sinks", sinks.len(), rows * HEADS)?;
        check_len("out", out.len(), rows * HEADS * HEAD_DIM)?;
        check_len(
            "slices",
            packed.slices.len(),
            packed.num_slices * PACKED_SLICE_WORDS,
        )?;
        ensure!(
            !packed.states.is_empty() && packed.states.len().is_multiple_of(BATCH_STATE_WORDS),
            "states has {} elements, not a positive multiple of {BATCH_STATE_WORDS}",
            packed.states.len()
        );
        ensure!(
            counters.len() >= total_blocks_m * KV_HEADS,
            "packed counters has {} elements, need at least {}",
            counters.len(),
            total_blocks_m * KV_HEADS
        );
        let extent = extent("bias", bias.len(), rows * HEADS)?;
        ensure!(
            extent.is_multiple_of(8),
            "prefill needs a bias extent that is a multiple of 8, not {extent}"
        );
        let slots = u32::try_from(cache.slots)?;
        let window = u32::try_from(cache.window.unwrap_or(0)).context("window too large")?;
        let total_blocks_u32 = u32::try_from(total_blocks_m)?;
        let input_keys = u32::from(cache.window.is_some());
        ensure!(
            input_keys == 0
                || matches!(
                    cache.format(),
                    KvFormat::Bf16 | KvFormat::Fp8 | KvFormat::Nvfp4 | KvFormat::Nvfp4G32
                ),
            "sliding caches are BF16, FP8, NVFP4 or NVFP4-G32"
        );
        let function = match (&*cache.data, products) {
            (KvData::Bf16 { .. }, _) => &self.prefill_packed,
            (KvData::Fp8 { .. }, Products::Bf16) => &self.prefill_packed_fp8,
            (KvData::Fp8 { .. }, Products::Fp8 | Products::Fp4) => &self.prefill_packed_fp8_mma,
            (KvData::Nvfp4 { .. }, _) => &self.prefill_packed_fp4,
            (KvData::Nvfp4G32 { .. }, _) => &self.prefill_packed_fp4_g32,
        };
        let mut launch = stream.launch_builder(function);
        launch
            .arg(&q)
            .arg(&k)
            .arg(&v)
            .arg(&packed.states)
            .arg(&packed.slices)
            .arg(block_descs)
            .arg(&bias)
            .arg(&sinks)
            .arg(&mut scratch.partials)
            .arg(counters)
            .arg(&mut out)
            .arg(&total_blocks_u32)
            .arg(&slots)
            .arg(&window)
            .arg(&extent)
            .arg(&input_keys)
            .arg(&SCALE);
        let config = LaunchConfig {
            grid_dim: (u32::try_from(scratch.blocks)?, 1, 1),
            block_dim: (PREFILL_BLOCK, 1, 1),
            shared_mem_bytes: PREFILL_PACKED_SHARED_BYTES,
        };
        // SAFETY: the arguments match `prefill_attention_packed_{bf16,fp8,fp8_mma,fp4}`.
        // Every per-row buffer holds `rows` rows as checked above, `block_descs` names
        // `total_blocks_m <= 128` valid `(local_block_m, slice_idx)` pairs into `slices`,
        // each slice's KV caches (words 0 and 1 of `states`) match `cache.format()` and
        // `slots` and hold all positions up to `kv_start + row_len - 1`, `scratch.partials`
        // holds two slots per SM, `counters` holds at least `total_blocks_m * 8` zeroed
        // counters (returned to zero), and `PREFILL_PACKED_SHARED_BYTES` matches the
        // `static_assert`s in `attention.cu`.
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Ragged packed multi-slice form of [`Attention::prefill_attention`]: appends all `rows`
    /// tokens across `packed.num_slices` slices into their own KV caches and computes their
    /// attention in one persistent grid launch across all S slices (2 launches total: 1 KV
    /// append + 1 flash attention, independent of `S`).
    #[expect(clippy::too_many_arguments, reason = "mirrors prefill_attention")]
    pub fn prefill_attention_packed(
        &self,
        stream: &CudaStream,
        q: &impl DeviceIn<bf16>,
        k: &impl DeviceIn<bf16>,
        v: &impl DeviceIn<bf16>,
        cache: &KvCache,
        packed: &PackedRows<'_>,
        block_descs: &CudaView<'_, u32>,
        rows: usize,
        bias: &impl DeviceIn<bf16>,
        sinks: &impl DeviceIn<bf16>,
        scratch: &mut PrefillScratch,
        counters: &mut CudaSlice<u32>,
        out: &mut impl DeviceOut<bf16>,
        products: Products,
    ) -> anyhow::Result<()> {
        let sliding = cache.window.is_some();
        if !sliding {
            self.append_rows_packed(stream, cache, packed, rows, k, v)?;
        }
        self.prefill_attention_packed_kernel(
            stream,
            q,
            k,
            v,
            cache,
            packed,
            block_descs,
            rows,
            bias,
            sinks,
            scratch,
            counters,
            out,
            products,
        )?;
        if sliding {
            self.append_rows_packed(stream, cache, packed, rows, k, v)?;
        }
        Ok(())
    }
}
