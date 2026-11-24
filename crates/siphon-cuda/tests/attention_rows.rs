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

use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::attention::{
        Attention, AttentionScratch, HEAD_DIM, HEADS, KV_HEADS, KvCache, KvData, KvFormat,
        ROWS_MAX, RowsAttentionScratch,
    },
};

const WINDOW: usize = 1024;
const RING: usize = 1032;
const CAPACITY: usize = 4096;
const EXTENT: usize = 1024;
const KV_ROW: usize = KV_HEADS * HEAD_DIM;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Sliding,
    Ring,
    Global,
    GlobalFp8,
}

fn cache(stream: &Arc<CudaStream>, kind: Kind) -> anyhow::Result<KvCache> {
    match kind {
        Kind::Sliding => KvCache::sliding(stream, WINDOW),
        Kind::Ring => KvCache::sliding_ring(stream, WINDOW, RING),
        Kind::Global => KvCache::global(stream, CAPACITY),
        Kind::GlobalFp8 => KvCache::global_with_format(stream, CAPACITY, KvFormat::Fp8),
    }
}

/// Fills every slot of `cache` with the same pseudo-random keys and values for a seed.
fn fill(stream: &Arc<CudaStream>, cache: &mut KvCache, seed: u32) -> anyhow::Result<()> {
    match cache.data_mut() {
        KvData::Bf16 { keys, values } => {
            let n = keys.len();
            stream.memcpy_htod(&common::pseudo_random_bf16(n, seed, 1.0, 0.0), keys)?;
            stream.memcpy_htod(&common::pseudo_random_bf16(n, seed + 1, 1.0, 0.0), values)?;
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
        stream, &q_dev, &k_dev, &v_dev, cache, &start_dev, count, &bias, &sinks, scratch, &mut out,
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
    let mut scratch = RowsAttentionScratch::new(stream)?;
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
    let mut scratch = RowsAttentionScratch::new(stream)?;
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
