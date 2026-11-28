//! The multi-row attention of `Attention::attention_rows`.
//!
//! - Invariance, bit for bit: each row's output is the same for every row count from 1 to
//!   8, whatever the other rows hold, and equal to a one-row call at its own position after
//!   the rows before it, as the speculative loop's exactness test needs.
//! - Against `Attention::append` and `Attention::decode_attention` row by row, which sum
//!   in another order: the outputs' relative RMS difference must stay below 2^-8.
//!
//! Each case runs on sliding rings of 1024 and 1032 slots with a window of 1024, before,
//! across and after the wrap, and on global caches in BF16 and FP8.

#![cfg(target_os = "linux")]

mod common;

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr as _};
use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::{
        attention::{
            Attention, AttentionScratch, BATCH_ADAPTIVE_MIN_SEQS, BatchSplitMode, HEAD_DIM, HEADS,
            KV_HEADS, KvCache, KvData, KvFormat, KvQuant, KvRounding, ROWS_MAX,
            RowsAttentionScratch, batch_split_blocks,
        },
        spec::{BATCH_STATE_WORDS, BATCH_VERIFY_MAX_ROWS, BatchRows},
    },
};

const WINDOW: usize = 1024;
const RING: usize = 1032;
const CAPACITY: usize = 4096;
const EXTENT: usize = 1024;
const KV_ROW: usize = KV_HEADS * HEAD_DIM;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Sliding,
    SlidingFp8,
    SlidingNvfp4G32,
    Ring,
    Global,
    GlobalFp8,
    GlobalNvfp4,
    GlobalNvfp4G32,
}

fn cache(stream: &Arc<CudaStream>, kind: Kind) -> anyhow::Result<KvCache> {
    match kind {
        Kind::Sliding => KvCache::sliding(stream, WINDOW),
        Kind::SlidingFp8 => {
            KvCache::sliding_ring_with_format(stream, WINDOW, WINDOW, KvFormat::Fp8)
        }
        Kind::SlidingNvfp4G32 => {
            KvCache::sliding_ring_with_format(stream, WINDOW, WINDOW, KvFormat::Nvfp4G32)
        }
        Kind::Ring => KvCache::sliding_ring(stream, WINDOW, RING),
        Kind::Global => KvCache::global(stream, CAPACITY),
        Kind::GlobalFp8 => KvCache::global_with_format(stream, CAPACITY, KvFormat::Fp8),
        Kind::GlobalNvfp4 => KvCache::global_with_format(stream, CAPACITY, KvFormat::Nvfp4),
        Kind::GlobalNvfp4G32 => KvCache::global_with_format(stream, CAPACITY, KvFormat::Nvfp4G32),
    }
}

/// Fills every slot of `cache` with the same pseudo-random keys and values for a seed.
fn fill(stream: &Arc<CudaStream>, cache: &mut KvCache, seed: u32) -> anyhow::Result<()> {
    let slots = cache.slots();
    match cache.data_mut() {
        KvData::Bf16 { keys, values } => {
            let n = keys.len();
            stream.memcpy_htod(&common::pseudo_random_bf16(n, seed, 1.0, 0.0), keys)?;
            stream.memcpy_htod(&common::pseudo_random_bf16(n, seed + 1, 1.0, 0.0), values)?;
        }
        KvData::Nvfp4 { keys, values } | KvData::Nvfp4G32 { keys, values } => {
            let row_bytes = keys.len() / slots;
            // Each slot is 576 (g16) or 544 (g32) bytes: 512 bytes of E2M1 nibble pairs
            // followed by 64 or 32 bytes of E4M3 scales (positive, below 0.75, never NaN).
            let bytes = |seed: u32| -> Vec<u8> {
                let mut out = vec![0u8; slots * row_bytes];
                for s in 0..slots {
                    let row = &mut out[s * row_bytes..(s + 1) * row_bytes];
                    for (i, b) in row.iter_mut().enumerate() {
                        let mut h = u32::try_from(s * row_bytes + i).expect("fits")
                            ^ seed.wrapping_mul(0x85EB_CA77);
                        h = h.wrapping_mul(0x9E37_79B1);
                        h ^= h >> 15;
                        *b = if i < KV_HEADS * HEAD_DIM / 2 {
                            u8::try_from((h >> 8) & 0xff).expect("byte")
                        } else {
                            u8::try_from((h >> 8) % 49).expect("scale")
                        };
                    }
                }
                out
            };
            stream.memcpy_htod(&bytes(seed), keys)?;
            stream.memcpy_htod(&bytes(seed + 1), values)?;
        }
        KvData::Fp8 { keys, values } => {
            let n = keys.len();
            // E4M3 values up to about 1.5, never NaN.
            let bytes = |seed: u32| -> Vec<u8> {
                (0..n)
                    .map(|i| {
                        let mut h =
                            u32::try_from(i).expect("fits") ^ seed.wrapping_mul(0x85EB_CA77);
                        h = h.wrapping_mul(0x9E37_79B1);
                        h ^= h >> 15;
                        let magnitude = u8::try_from((h >> 8) % 61).expect("below 61");
                        magnitude | if h & 1 == 0 { 0 } else { 0x80 }
                    })
                    .collect()
            };
            stream.memcpy_htod(&bytes(seed), keys)?;
            stream.memcpy_htod(&bytes(seed + 1), values)?;
        }
    }
    Ok(())
}

/// One step's rows: queries, new keys and values, bias and sinks.
struct Rows {
    q: Vec<bf16>,
    k: Vec<bf16>,
    v: Vec<bf16>,
    bias: Vec<bf16>,
    sinks: Vec<bf16>,
}

impl Rows {
    fn new(seed: u32) -> Self {
        let m = ROWS_MAX;
        Self {
            q: common::pseudo_random_bf16(m * HEADS * HEAD_DIM, seed, 1.5, 0.0),
            k: common::pseudo_random_bf16(m * KV_ROW, seed + 1, 1.0, 0.0),
            v: common::pseudo_random_bf16(m * KV_ROW, seed + 2, 1.0, 0.0),
            bias: common::pseudo_random_bf16(m * HEADS * EXTENT, seed + 3, 0.5, 0.0),
            sinks: common::pseudo_random_bf16(m * HEADS, seed + 4, 1.0, 0.0),
        }
    }

    /// Rows `from .. from + m`.
    fn slice(&self, from: usize, m: usize) -> Self {
        let cut = |x: &[bf16], w: usize| x[from * w..(from + m) * w].to_vec();
        Self {
            q: cut(&self.q, HEADS * HEAD_DIM),
            k: cut(&self.k, KV_ROW),
            v: cut(&self.v, KV_ROW),
            bias: cut(&self.bias, HEADS * EXTENT),
            sinks: cut(&self.sinks, HEADS),
        }
    }
}

struct Setup {
    gpu: Gpu,
    attention: Attention,
}

/// The multi-row attention of `rows` (count of them) at `start`, on `cache` as it stands.
fn run_rows(
    setup: &Setup,
    cache: &mut KvCache,
    scratch: &mut RowsAttentionScratch,
    rows: &Rows,
    start: usize,
) -> anyhow::Result<Vec<bf16>> {
    let stream = setup.gpu.stream();
    let count = rows.sinks.len() / HEADS;
    let start_dev: CudaSlice<u32> = stream.clone_htod(&[u32::try_from(start)?])?;
    let q_dev = stream.clone_htod(&rows.q)?;
    let k_dev = stream.clone_htod(&rows.k)?;
    let v_dev = stream.clone_htod(&rows.v)?;
    let bias = stream.clone_htod(&rows.bias)?;
    let sinks = stream.clone_htod(&rows.sinks)?;
    let mut out = stream.alloc_zeros::<bf16>(count * HEADS * HEAD_DIM)?;
    setup.attention.attention_rows(
        stream, &q_dev, &k_dev, &v_dev, cache, &start_dev, count, &bias, &sinks, None, scratch,
        &mut out,
    )?;
    Ok(stream.clone_dtoh(&out)?)
}

fn same(a: &[bf16], b: &[bf16]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

const CASES: &[(Kind, usize)] = &[
    (Kind::Sliding, 500),
    (Kind::Sliding, 1020),
    (Kind::Sliding, 2500),
    (Kind::Ring, 1020),
    (Kind::Ring, 2500),
    (Kind::Global, 3),
    (Kind::Global, 2500),
    (Kind::GlobalFp8, 2500),
];

#[test]
fn rows_are_invariant() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let setup = Setup {
        attention: Attention::load(gpu.context())?,
        gpu,
    };
    let stream = setup.gpu.stream();
    let mut scratch = RowsAttentionScratch::new(stream, ROWS_MAX)?;
    for &(kind, start) in CASES {
        let label = format!("{kind:?} at {start}");
        let rows = Rows::new(start as u32);
        let mut kv_cache = cache(stream, kind)?;
        fill(stream, &mut kv_cache, 7)?;
        let full = run_rows(&setup, &mut kv_cache, &mut scratch, &rows, start)?;
        let width = HEADS * HEAD_DIM;
        for m in 1..ROWS_MAX {
            fill(stream, &mut kv_cache, 7)?;
            let part = run_rows(
                &setup,
                &mut kv_cache,
                &mut scratch,
                &rows.slice(0, m),
                start,
            )?;
            assert!(
                same(&part, &full[..m * width]),
                "{label}: m = {m} differs from m = 8"
            );
        }
        // One row at a time, each at its own position after the rows before it.
        fill(stream, &mut kv_cache, 7)?;
        for j in 0..ROWS_MAX {
            let one = run_rows(
                &setup,
                &mut kv_cache,
                &mut scratch,
                &rows.slice(j, 1),
                start + j,
            )?;
            assert!(
                same(&one, &full[j * width..(j + 1) * width]),
                "{label}: row {j} alone differs"
            );
        }
        // Rows 1 to 7 with other values.
        let mut other = Rows::new(start as u32 + 99);
        other.q[..width].copy_from_slice(&rows.q[..width]);
        other.k[..KV_ROW].copy_from_slice(&rows.k[..KV_ROW]);
        other.v[..KV_ROW].copy_from_slice(&rows.v[..KV_ROW]);
        other.bias[..HEADS * EXTENT].copy_from_slice(&rows.bias[..HEADS * EXTENT]);
        other.sinks[..HEADS].copy_from_slice(&rows.sinks[..HEADS]);
        fill(stream, &mut kv_cache, 7)?;
        let mixed = run_rows(&setup, &mut kv_cache, &mut scratch, &other, start)?;
        assert!(
            same(&mixed[..width], &full[..width]),
            "{label}: row 0 depends on the other rows"
        );
        eprintln!("{label}: rows bit-identical for m = 1 to 8, alone and with other rows");
    }
    Ok(())
}

#[test]
fn rows_against_decode_attention() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let setup = Setup {
        attention: Attention::load(gpu.context())?,
        gpu,
    };
    let stream = setup.gpu.stream();
    let mut scratch = RowsAttentionScratch::new(stream, ROWS_MAX)?;
    let mut decode = AttentionScratch::new(stream)?;
    let width = HEADS * HEAD_DIM;
    for &(kind, start) in CASES {
        let label = format!("{kind:?} at {start}");
        let rows = Rows::new(start as u32 + 7);
        let mut kv_cache = cache(stream, kind)?;
        fill(stream, &mut kv_cache, 11)?;
        let got = run_rows(&setup, &mut kv_cache, &mut scratch, &rows, start)?;
        fill(stream, &mut kv_cache, 11)?;
        let mut want = Vec::new();
        for j in 0..ROWS_MAX {
            let row = rows.slice(j, 1);
            let position = stream.clone_htod(&[u32::try_from(start + j)?])?;
            let k_dev = stream.clone_htod(&row.k)?;
            let v_dev = stream.clone_htod(&row.v)?;
            setup
                .attention
                .append(stream, &mut kv_cache, &position, &k_dev, &v_dev)?;
            let q_dev = stream.clone_htod(&row.q)?;
            let bias = stream.clone_htod(&row.bias)?;
            let sinks = stream.clone_htod(&row.sinks)?;
            let mut out = stream.alloc_zeros::<bf16>(width)?;
            setup.attention.decode_attention(
                stream,
                &q_dev,
                &kv_cache,
                &position,
                &bias,
                &sinks,
                &mut decode,
                &mut out,
            )?;
            want.extend(stream.clone_dtoh(&out)?);
        }
        let (mut diff, mut norm, mut differ, mut ulps) = (0f64, 0f64, 0, 0);
        for (g, w) in got.iter().zip(&want) {
            let (gf, wf) = (f64::from(g.to_f32()), f64::from(w.to_f32()));
            diff += (gf - wf).powi(2);
            norm += wf * wf;
            differ += usize::from(g.to_bits() != w.to_bits());
            ulps = ulps.max(common::bf16_ulps(*g, *w));
        }
        let rel = (diff / norm).sqrt();
        eprintln!(
            "{label}: {differ} of {} outputs differ from decode attention, at most {ulps} BF16 \
             ulps, relative RMS {rel:.2e}",
            got.len()
        );
        assert!(
            rel < 2f64.powi(-8),
            "{label}: too far from decode attention"
        );
    }
    Ok(())
}

fn copy_cache(stream: &Arc<CudaStream>, src: &KvCache, dst: &mut KvCache) -> anyhow::Result<()> {
    match (src.data(), dst.data_mut()) {
        (
            KvData::Bf16 {
                keys: sk,
                values: sv,
            },
            KvData::Bf16 {
                keys: dk,
                values: dv,
            },
        ) => {
            stream.memcpy_dtod(sk, dk)?;
            stream.memcpy_dtod(sv, dv)?;
        }
        (
            KvData::Fp8 {
                keys: sk,
                values: sv,
            },
            KvData::Fp8 {
                keys: dk,
                values: dv,
            },
        )
        | (
            KvData::Nvfp4 {
                keys: sk,
                values: sv,
            },
            KvData::Nvfp4 {
                keys: dk,
                values: dv,
            },
        )
        | (
            KvData::Nvfp4G32 {
                keys: sk,
                values: sv,
            },
            KvData::Nvfp4G32 {
                keys: dk,
                values: dv,
            },
        ) => {
            stream.memcpy_dtod(sk, dk)?;
            stream.memcpy_dtod(sv, dv)?;
        }
        _ => anyhow::bail!("cache format mismatch"),
    }
    Ok(())
}

fn cache_slot_bytes(
    stream: &Arc<CudaStream>,
    cache: &KvCache,
    slot: usize,
) -> anyhow::Result<Vec<u8>> {
    let len = cache.row_len();
    let at = (slot % cache.slots()) * len;
    let mut out = Vec::with_capacity(2 * cache.row_bytes());
    match cache.data() {
        KvData::Bf16 { keys, values } => {
            for buf in [keys, values] {
                for v in stream.clone_dtoh(&buf.slice(at..at + len))? {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        KvData::Fp8 { keys, values }
        | KvData::Nvfp4 { keys, values }
        | KvData::Nvfp4G32 { keys, values } => {
            for buf in [keys, values] {
                out.extend(stream.clone_dtoh(&buf.slice(at..at + len))?);
            }
        }
    }
    Ok(out)
}

fn cache_ptrs(stream: &CudaStream, cache: &KvCache) -> (u64, u64) {
    match cache.data() {
        KvData::Bf16 { keys, values } => (keys.device_ptr(stream).0, values.device_ptr(stream).0),
        KvData::Fp8 { keys, values }
        | KvData::Nvfp4 { keys, values }
        | KvData::Nvfp4G32 { keys, values } => {
            (keys.device_ptr(stream).0, values.device_ptr(stream).0)
        }
    }
}

struct BatchInputs {
    q: CudaSlice<bf16>,
    k: CudaSlice<bf16>,
    v: CudaSlice<bf16>,
    bias: CudaSlice<bf16>,
    sinks: CudaSlice<bf16>,
    gate: CudaSlice<bf16>,
}

impl BatchInputs {
    fn new(setup: &Setup, kind: Kind, total_rows: usize, seed: u32) -> anyhow::Result<Self> {
        let stream = setup.gpu.stream();
        let q = stream.clone_htod(&common::pseudo_random_bf16(
            total_rows * HEADS * HEAD_DIM,
            seed,
            1.5,
            0.0,
        ))?;
        let mut k = stream.clone_htod(&common::pseudo_random_bf16(
            total_rows * KV_ROW,
            seed + 1,
            1.0,
            0.0,
        ))?;
        let mut v = stream.clone_htod(&common::pseudo_random_bf16(
            total_rows * KV_ROW,
            seed + 2,
            1.0,
            0.0,
        ))?;
        if matches!(
            kind,
            Kind::GlobalNvfp4 | Kind::GlobalNvfp4G32 | Kind::SlidingNvfp4G32
        ) {
            let rounding = KvRounding {
                quant: if kind == Kind::GlobalNvfp4 {
                    KvQuant::Nvfp4
                } else {
                    KvQuant::Nvfp4G32
                },
                ..KvRounding::default()
            };
            setup
                .attention
                .round_rows(stream, &mut k, &mut v, total_rows, rounding)?;
        }
        let bias = stream.clone_htod(&common::pseudo_random_bf16(
            total_rows * HEADS * EXTENT,
            seed + 3,
            0.5,
            0.0,
        ))?;
        let sinks = stream.clone_htod(&common::pseudo_random_bf16(
            total_rows * HEADS,
            seed + 4,
            1.0,
            0.0,
        ))?;
        let gate = stream.clone_htod(&common::pseudo_random_bf16(
            total_rows * HEADS * HEAD_DIM,
            seed + 5,
            1.5,
            0.0,
        ))?;
        Ok(Self {
            q,
            k,
            v,
            bias,
            sinks,
            gate,
        })
    }
}

#[expect(clippy::too_many_arguments, reason = "batched attention test helper")]
fn run_batch(
    setup: &Setup,
    caches: &[KvCache],
    inputs: &BatchInputs,
    positions: &[usize],
    rows_per_seq: usize,
    use_gate: bool,
    mode: BatchSplitMode,
    scratch: &mut RowsAttentionScratch,
) -> anyhow::Result<Vec<bf16>> {
    let stream = setup.gpu.stream();
    let seqs = caches.len();
    let total_rows = seqs * rows_per_seq;
    let mut states = vec![0u64; seqs * BATCH_STATE_WORDS];
    for (z, c) in caches.iter().enumerate() {
        let (k_ptr, v_ptr) = cache_ptrs(stream, c);
        states[z * BATCH_STATE_WORDS] = k_ptr;
        states[z * BATCH_STATE_WORDS + 1] = v_ptr;
    }
    let pos_u32: Vec<u32> = positions
        .iter()
        .map(|&p| u32::try_from(p).expect("position fits u32"))
        .collect();
    let states_dev = stream.clone_htod(&states)?;
    let positions_dev = stream.clone_htod(&pos_u32)?;
    let batch = BatchRows {
        states: states_dev.as_view(),
        positions: positions_dev.as_view(),
        rows_per_seq,
    };
    let gate_view = use_gate.then(|| inputs.gate.as_view());
    let mut out = stream.alloc_zeros::<bf16>(total_rows * HEADS * HEAD_DIM)?;
    setup.attention.attention_batch_with_mode(
        stream,
        &inputs.q,
        &inputs.k,
        &inputs.v,
        &caches[0],
        &batch,
        total_rows,
        &inputs.bias,
        &inputs.sinks,
        gate_view.as_ref(),
        scratch,
        &mut out,
        mode,
    )?;
    Ok(stream.clone_dtoh(&out)?)
}

fn error_stats(got: &[bf16], want: &[bf16]) -> (f64, f64, f64, u32) {
    let (mut sq_diff, mut sq_norm, mut abs_sum, mut max_abs, mut max_ulps) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0u32);
    for (&g, &w) in got.iter().zip(want) {
        let (gf, wf) = (f64::from(g.to_f32()), f64::from(w.to_f32()));
        let d = (gf - wf).abs();
        sq_diff += d * d;
        sq_norm += wf * wf;
        abs_sum += d;
        max_abs = max_abs.max(d);
        max_ulps = max_ulps.max(common::bf16_ulps(g, w));
    }
    let rel_rms = (sq_diff / sq_norm.max(1e-30)).sqrt();
    let mean_abs = abs_sum / got.len() as f64;
    (rel_rms, max_abs, mean_abs, max_ulps)
}

#[test]
fn batch_split_blocks_schedule() {
    assert_eq!(BATCH_ADAPTIVE_MIN_SEQS, 8);
    // RTX PRO 6000 Blackwell (188 SMs): 2 * 188 = 376 target CTAs.
    // Auto stays exact for seqs < 8 (preserving B=1..5 bit-exact parity).
    for seqs in 1..BATCH_ADAPTIVE_MIN_SEQS {
        assert_eq!(
            batch_split_blocks(188, seqs, 4096, BatchSplitMode::Auto),
            32
        );
        assert_eq!(
            batch_split_blocks(188, seqs, 1024, BatchSplitMode::Auto),
            16
        );
    }
    // Auto uses adaptive split-K for seqs >= 8.
    assert_eq!(batch_split_blocks(188, 8, 4096, BatchSplitMode::Auto), 5);
    assert_eq!(batch_split_blocks(188, 16, 4096, BatchSplitMode::Auto), 2);
    assert_eq!(batch_split_blocks(188, 24, 4096, BatchSplitMode::Auto), 1);
    assert_eq!(batch_split_blocks(188, 32, 4096, BatchSplitMode::Auto), 1);
    assert_eq!(batch_split_blocks(188, 48, 4096, BatchSplitMode::Auto), 1);
    assert_eq!(batch_split_blocks(188, 64, 4096, BatchSplitMode::Auto), 1);
    assert_eq!(batch_split_blocks(188, 8, 1024, BatchSplitMode::Auto), 5);
    assert_eq!(batch_split_blocks(188, 16, 1024, BatchSplitMode::Auto), 2);
    assert_eq!(batch_split_blocks(188, 32, 1024, BatchSplitMode::Auto), 1);

    // Exact always uses full splits regardless of batch size.
    for seqs in [1, 5, 8, 16, 32, 64] {
        assert_eq!(
            batch_split_blocks(188, seqs, 65_536, BatchSplitMode::Exact),
            32
        );
        assert_eq!(
            batch_split_blocks(188, seqs, 1024, BatchSplitMode::Exact),
            16
        );
        assert_eq!(batch_split_blocks(188, seqs, 64, BatchSplitMode::Exact), 1);
    }

    // Adaptive works at all batch sizes across 188 / 170 / 128 SM GPUs.
    assert_eq!(
        batch_split_blocks(188, 1, 4096, BatchSplitMode::Adaptive),
        32
    );
    assert_eq!(
        batch_split_blocks(188, 5, 4096, BatchSplitMode::Adaptive),
        9
    );
    assert_eq!(
        batch_split_blocks(170, 16, 4096, BatchSplitMode::Adaptive),
        2
    );
    assert_eq!(
        batch_split_blocks(170, 32, 4096, BatchSplitMode::Adaptive),
        1
    );
    assert_eq!(
        batch_split_blocks(128, 8, 4096, BatchSplitMode::Adaptive),
        4
    );
    assert_eq!(
        batch_split_blocks(128, 16, 4096, BatchSplitMode::Adaptive),
        2
    );
    assert_eq!(
        batch_split_blocks(128, 32, 4096, BatchSplitMode::Adaptive),
        1
    );

    // Fixed clamps to 1..=max_splits.
    assert_eq!(
        batch_split_blocks(188, 8, 4096, BatchSplitMode::Fixed(0)),
        1
    );
    assert_eq!(
        batch_split_blocks(188, 8, 4096, BatchSplitMode::Fixed(1)),
        1
    );
    assert_eq!(
        batch_split_blocks(188, 8, 4096, BatchSplitMode::Fixed(4)),
        4
    );
    assert_eq!(
        batch_split_blocks(188, 8, 1024, BatchSplitMode::Fixed(32)),
        16
    );
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "verifies B=1 and B=5 parity across cache formats and widths"
)]
fn batch_exact_parity_b1_and_b5() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let setup = Setup {
        attention: Attention::load(gpu.context())?,
        gpu,
    };
    let stream = setup.gpu.stream();
    let mut scratch = RowsAttentionScratch::new(stream, BATCH_VERIFY_MAX_ROWS)?;
    let width = HEADS * HEAD_DIM;
    let kinds = [
        Kind::Sliding,
        Kind::SlidingFp8,
        Kind::SlidingNvfp4G32,
        Kind::Ring,
        Kind::Global,
        Kind::GlobalFp8,
        Kind::GlobalNvfp4,
        Kind::GlobalNvfp4G32,
    ];
    let positions_5 = [3usize, 500, 1020, 2500, 3900];

    for kind in kinds {
        let mut templates = Vec::with_capacity(5);
        for z in 0u32..5 {
            let mut c = cache(stream, kind)?;
            fill(stream, &mut c, 31 + z * 7)?;
            templates.push(c);
        }
        for seqs in [1usize, 5] {
            for rows_per_seq in [1usize, 4, 8] {
                for use_gate in [false, true] {
                    let positions = &positions_5[..seqs];
                    let total_rows = seqs * rows_per_seq;
                    let inputs = BatchInputs::new(
                        &setup,
                        kind,
                        total_rows,
                        1000 + seqs as u32 * 13 + rows_per_seq as u32,
                    )?;

                    // Reference: single-sequence `attention_rows` for each sequence.
                    let mut ref_caches = Vec::with_capacity(seqs);
                    let mut want = Vec::with_capacity(total_rows * width);
                    for (z, &pos) in positions.iter().enumerate() {
                        let mut c = cache(stream, kind)?;
                        copy_cache(stream, &templates[z], &mut c)?;
                        let start_dev = stream.clone_htod(&[u32::try_from(pos)?])?;
                        let row0 = z * rows_per_seq;
                        let q_v = inputs.q.slice(row0 * width..(row0 + rows_per_seq) * width);
                        let k_v = inputs
                            .k
                            .slice(row0 * KV_ROW..(row0 + rows_per_seq) * KV_ROW);
                        let v_v = inputs
                            .v
                            .slice(row0 * KV_ROW..(row0 + rows_per_seq) * KV_ROW);
                        let b_v = inputs
                            .bias
                            .slice(row0 * HEADS * EXTENT..(row0 + rows_per_seq) * HEADS * EXTENT);
                        let s_v = inputs
                            .sinks
                            .slice(row0 * HEADS..(row0 + rows_per_seq) * HEADS);
                        let g_v = use_gate.then(|| {
                            inputs
                                .gate
                                .slice(row0 * width..(row0 + rows_per_seq) * width)
                        });
                        let mut out = stream.alloc_zeros::<bf16>(rows_per_seq * width)?;
                        setup.attention.attention_rows(
                            stream,
                            &q_v,
                            &k_v,
                            &v_v,
                            &mut c,
                            &start_dev,
                            rows_per_seq,
                            &b_v,
                            &s_v,
                            g_v.as_ref(),
                            &mut scratch,
                            &mut out,
                        )?;
                        want.extend(stream.clone_dtoh(&out)?);
                        ref_caches.push(c);
                    }

                    // Default `BatchSplitMode::Auto` must be bit-identical for B=1 and B=5.
                    let mut batch_caches = Vec::with_capacity(seqs);
                    for tmpl in &templates[..seqs] {
                        let mut c = cache(stream, kind)?;
                        copy_cache(stream, tmpl, &mut c)?;
                        batch_caches.push(c);
                    }
                    let got = run_batch(
                        &setup,
                        &batch_caches,
                        &inputs,
                        positions,
                        rows_per_seq,
                        use_gate,
                        BatchSplitMode::Auto,
                        &mut scratch,
                    )?;
                    assert!(
                        same(&got, &want),
                        "{kind:?} B={seqs} W={rows_per_seq} gate={use_gate}: Auto differs from attention_rows"
                    );
                    for z in 0..seqs {
                        for j in 0..rows_per_seq {
                            let slot = positions[z] + j;
                            assert_eq!(
                                cache_slot_bytes(stream, &batch_caches[z], slot)?,
                                cache_slot_bytes(stream, &ref_caches[z], slot)?,
                                "{kind:?} B={seqs} W={rows_per_seq}: appended KV mismatch seq {z} row {j}"
                            );
                        }
                    }
                }
            }
        }
        eprintln!("{kind:?}: B=1 and B=5 (W=1,4,8, gated/ungated) bit-identical to attention_rows");
    }
    Ok(())
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "sweeps batch sizes, widths, modes and cache kinds"
)]
fn batch_adaptive_and_nosplit_accuracy() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let setup = Setup {
        attention: Attention::load(gpu.context())?,
        gpu,
    };
    let stream = setup.gpu.stream();
    let mut scratch = RowsAttentionScratch::new(stream, BATCH_VERIFY_MAX_ROWS)?;
    let kinds = [
        Kind::Sliding,
        Kind::SlidingFp8,
        Kind::SlidingNvfp4G32,
        Kind::Ring,
        Kind::Global,
        Kind::GlobalFp8,
        Kind::GlobalNvfp4,
        Kind::GlobalNvfp4G32,
    ];
    // Positions spanning empty/recent-only (`has_far == false`), boundary 7/8/63/64,
    // mid-window, across sliding wrap (1020), and deep global cache.
    let pos_pool = [0usize, 3, 7, 8, 63, 64, 500, 1020, 1024, 2500, 3900];
    // (seqs, rows_per_seq, modes) covering B = 1, 8, 16, 32, 64 and both decode & verify.
    let shapes: &[(usize, usize, &[BatchSplitMode])] = &[
        (1, 1, &[BatchSplitMode::Fixed(1), BatchSplitMode::Fixed(4)]),
        (1, 8, &[BatchSplitMode::Fixed(1), BatchSplitMode::Fixed(4)]),
        (
            8,
            1,
            &[
                BatchSplitMode::Auto,
                BatchSplitMode::Adaptive,
                BatchSplitMode::Fixed(1),
            ],
        ),
        (8, 8, &[BatchSplitMode::Auto, BatchSplitMode::Fixed(1)]),
        (16, 1, &[BatchSplitMode::Auto, BatchSplitMode::Fixed(1)]),
        (16, 8, &[BatchSplitMode::Auto, BatchSplitMode::Fixed(1)]),
        (32, 1, &[BatchSplitMode::Auto, BatchSplitMode::Fixed(2)]),
        (32, 8, &[BatchSplitMode::Auto, BatchSplitMode::Fixed(2)]),
        (64, 1, &[BatchSplitMode::Auto, BatchSplitMode::Fixed(2)]),
        (64, 4, &[BatchSplitMode::Auto]),
    ];

    for kind in kinds {
        let mut templates = Vec::with_capacity(4);
        for t in 0u32..4 {
            let mut c = cache(stream, kind)?;
            fill(stream, &mut c, 71 + t * 19)?;
            templates.push(c);
        }
        for &(seqs, rows_per_seq, modes) in shapes {
            let total_rows = seqs * rows_per_seq;
            let positions: Vec<usize> = (0..seqs).map(|z| pos_pool[z % pos_pool.len()]).collect();
            let inputs = BatchInputs::new(
                &setup,
                kind,
                total_rows,
                5000 + seqs as u32 * 31 + rows_per_seq as u32,
            )?;

            let mut caches = Vec::with_capacity(seqs);
            for z in 0..seqs {
                let mut c = cache(stream, kind)?;
                copy_cache(stream, &templates[z % templates.len()], &mut c)?;
                caches.push(c);
            }
            let use_gate = rows_per_seq > 1 || seqs >= 16;
            let exact = run_batch(
                &setup,
                &caches,
                &inputs,
                &positions,
                rows_per_seq,
                use_gate,
                BatchSplitMode::Exact,
                &mut scratch,
            )?;
            let check_seqs = [0, seqs / 2, seqs - 1];
            let mut exact_appended = Vec::new();
            for &z in &check_seqs {
                for j in 0..rows_per_seq {
                    exact_appended.push((
                        z,
                        j,
                        cache_slot_bytes(stream, &caches[z], positions[z] + j)?,
                    ));
                }
            }

            for &mode in modes {
                for (z, c) in caches.iter_mut().enumerate() {
                    copy_cache(stream, &templates[z % templates.len()], c)?;
                }
                let got = run_batch(
                    &setup,
                    &caches,
                    &inputs,
                    &positions,
                    rows_per_seq,
                    use_gate,
                    mode,
                    &mut scratch,
                )?;
                let (rel_rms, max_abs, mean_abs, max_ulps) = error_stats(&got, &exact);
                let split_n =
                    batch_split_blocks(setup.attention.sms(), seqs, caches[0].slots(), mode);
                eprintln!(
                    "{kind:?} B={seqs:2} W={rows_per_seq} {mode:?} (splits={split_n:2}): \
                     rel_rms={rel_rms:.2e} max_abs={max_abs:.2e} mean_abs={mean_abs:.2e} max_ulps={max_ulps}"
                );
                assert!(
                    rel_rms < 2f64.powi(-9),
                    "{kind:?} B={seqs} W={rows_per_seq} {mode:?}: rel_rms {rel_rms:.2e} too high"
                );
                assert!(
                    max_abs < 2e-2,
                    "{kind:?} B={seqs} W={rows_per_seq} {mode:?}: max_abs {max_abs:.2e} too high"
                );
                for (z, j, want_bytes) in &exact_appended {
                    let got_bytes = cache_slot_bytes(stream, &caches[*z], positions[*z] + *j)?;
                    assert_eq!(
                        got_bytes, *want_bytes,
                        "{kind:?} B={seqs} W={rows_per_seq} {mode:?}: appended KV mismatch seq {z} row {j}"
                    );
                }
                // Run a second identical call to verify counters were cleanly reset to zero.
                for (z, c) in caches.iter_mut().enumerate() {
                    copy_cache(stream, &templates[z % templates.len()], c)?;
                }
                let got2 = run_batch(
                    &setup,
                    &caches,
                    &inputs,
                    &positions,
                    rows_per_seq,
                    use_gate,
                    mode,
                    &mut scratch,
                )?;
                assert!(
                    same(&got, &got2),
                    "{kind:?} B={seqs} W={rows_per_seq} {mode:?}: non-deterministic / dirty counters on repeat"
                );
            }
        }
    }
    Ok(())
}
