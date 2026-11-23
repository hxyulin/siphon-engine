//! Attention kernels against MLX outputs from `reference/fixtures/attention.py`.
//!
//! The fixture feeds 1100 tokens into a sliding (1024-slot ring) and a global cache and records
//! MLX's decode attention at a few positions across the window and the bias table width.

#![cfg(target_os = "linux")]

mod common;

use common::{Fixture, assert_bf16_close};
use half::bf16;
use inkling_cuda::{
    Gpu,
    ops::attention::{Attention, AttentionScratch, HEAD_DIM, HEADS, KV_HEADS, KvCache},
};

const WINDOW: usize = 1024;
/// Global cache capacity; the fixture's 1100 tokens fit.
const CAPACITY: usize = 1280;
/// Largest BF16 difference from MLX, which comes from exp2f standing in for Apple's exp.
const ATTENTION_ULPS: u32 = 1;

struct Setup {
    fixture: Fixture,
    gpu: Gpu,
    attention: Attention,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        let attention = Attention::load(gpu.context())?;
        Ok(Self {
            fixture: Fixture::load("attention")?,
            gpu,
            attention,
        })
    }

    fn checks(&self) -> anyhow::Result<Vec<usize>> {
        let (checks, _) = self.fixture.i32("checks")?;
        Ok(checks
            .into_iter()
            .map(|c| usize::try_from(c).expect("positions are non-negative"))
            .collect())
    }
}

#[test]
fn relative_bias_matches_mlx() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let Setup {
        fixture,
        gpu,
        attention,
    } = &setup;
    let stream = gpu.stream();
    for case in ["sliding", "global"] {
        let (table, shape) = fixture.bf16(&format!("{case}.table"))?;
        let table_dev = stream.clone_htod(&table)?;
        let mut bias_dev = stream.alloc_zeros::<bf16>(HEADS * shape[1])?;
        for position in setup.checks()? {
            let prefix = format!("{case}.{position}");
            let (rel, _) = fixture.bf16(&format!("{prefix}.rel"))?;
            let (want, _) = fixture.bf16(&format!("{prefix}.bias"))?;
            let rel_dev = stream.clone_htod(&rel)?;
            attention.relative_bias(stream, &rel_dev, &table_dev, &mut bias_dev)?;
            let got = stream.clone_dtoh(&bias_dev)?;
            assert_bf16_close(&format!("{prefix}.bias"), &got, &want, 0);
        }
    }
    Ok(())
}

/// Appends the fixture's keys and values one token at a time and compares the attention output
/// with MLX at each checked position.
fn check_decode(setup: &Setup, case: &str, mut cache: KvCache) -> anyhow::Result<()> {
    let Setup {
        fixture,
        gpu,
        attention,
    } = setup;
    let stream = gpu.stream();
    let (k, k_shape) = fixture.bf16("k")?;
    let (v, _) = fixture.bf16("v")?;
    let row = KV_HEADS * HEAD_DIM;
    let checks = setup.checks()?;
    let mut scratch = AttentionScratch::new(stream)?;
    let mut out_dev = stream.alloc_zeros::<bf16>(HEADS * HEAD_DIM)?;
    let mut position_dev = stream.alloc_zeros::<u32>(1)?;
    for position in 0..k_shape[0] {
        assert!(cache.holds(position));
        stream.memcpy_htod(&[u32::try_from(position)?], &mut position_dev)?;
        let k_dev = stream.clone_htod(&k[position * row..(position + 1) * row])?;
        let v_dev = stream.clone_htod(&v[position * row..(position + 1) * row])?;
        attention.append(stream, &mut cache, &position_dev, &k_dev, &v_dev)?;
        if !checks.contains(&position) {
            continue;
        }
        let prefix = format!("{case}.{position}");
        let (q, _) = fixture.bf16(&format!("{prefix}.q"))?;
        let (bias, _) = fixture.bf16(&format!("{prefix}.bias"))?;
        let (sinks, _) = fixture.bf16(&format!("{prefix}.sinks"))?;
        let (want, _) = fixture.bf16(&format!("{prefix}.out"))?;
        let q_dev = stream.clone_htod(&q)?;
        let bias_dev = stream.clone_htod(&bias)?;
        let sinks_dev = stream.clone_htod(&sinks)?;
        attention.decode_attention(
            stream,
            &q_dev,
            &cache,
            &position_dev,
            &bias_dev,
            &sinks_dev,
            &mut scratch,
            &mut out_dev,
        )?;
        let got = stream.clone_dtoh(&out_dev)?;
        assert_bf16_close(&prefix, &got, &want, ATTENTION_ULPS);
    }
    Ok(())
}

#[test]
fn sliding_decode_attention_matches_mlx() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let cache = KvCache::sliding(setup.gpu.stream(), WINDOW)?;
    check_decode(&setup, "sliding", cache)
}

#[test]
fn global_decode_attention_matches_mlx() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let cache = KvCache::global(setup.gpu.stream(), CAPACITY)?;
    check_decode(&setup, "global", cache)
}

#[test]
fn global_cache_holds_its_capacity() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let global = KvCache::global(gpu.stream(), 2)?;
    assert!(global.holds(0) && global.holds(1));
    assert!(!global.holds(2));
    let sliding = KvCache::sliding(gpu.stream(), 2)?;
    assert!(sliding.holds(2) && sliding.holds(1 << 20));
    Ok(())
}
