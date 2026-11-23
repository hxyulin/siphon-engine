//! Elementwise kernels against MLX outputs from `reference/fixtures/elementwise.py`.
//!
//! Kernels without transcendental functions must match MLX bit for bit. The others use
//! CUDA's exp, tanh and log, which round differently from Metal's `precise::exp`,
//! `precise::tanh` and `precise::log` (Metal's are up to 3 F32 ulps from the true value).
//! The F32 logit softcap shows those differences directly; a BF16 result can land one
//! ulp away, though none does in this fixture.

#![cfg(target_os = "linux")]

mod common;

use std::{collections::BTreeMap, sync::Arc};

use common::{Fixture, assert_bf16_close};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use inkling_cuda::{Gpu, ops::elementwise::Elementwise};

const PADDED_VOCAB: usize = 201_024;
const VOCAB: usize = 200_058;

/// Distance between two F32 values in units in the last place, ordering negatives correctly.
fn f32_ulps(a: f32, b: f32) -> u32 {
    let key = |v: f32| {
        let bits = i64::from(v.to_bits());
        if bits & 0x8000_0000 == 0 {
            bits
        } else {
            0x8000_0000 - bits
        }
    };
    u32::try_from(key(a).abs_diff(key(b))).unwrap_or(u32::MAX)
}

/// Fails if any element differs by more than `max_ulps`, and prints how many elements
/// differ by each ulp count.
fn assert_f32_ulps(label: &str, got: &[f32], want: &[f32], max_ulps: u32) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut counts = BTreeMap::<u32, usize>::new();
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let ulps = f32_ulps(g, w);
        assert!(
            ulps <= max_ulps,
            "{label}[{i}]: got {g:e}, want {w:e} ({ulps} ulps)"
        );
        *counts.entry(ulps).or_default() += 1;
    }
    eprintln!(
        "{label}: {} elements, count by ulp difference {counts:?}",
        want.len()
    );
}

struct Setup {
    gpu: Gpu,
    ops: Elementwise,
    fixture: Fixture,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        let ops = Elementwise::load(gpu.context())?;
        Ok(Self {
            gpu,
            ops,
            fixture: Fixture::load("elementwise")?,
        })
    }

    fn stream(&self) -> &Arc<CudaStream> {
        self.gpu.stream()
    }

    /// Fixture tensor `key` copied to the GPU.
    fn bf16(&self, key: &str) -> anyhow::Result<CudaSlice<bf16>> {
        Ok(self.stream().clone_htod(&self.fixture.bf16(key)?.0)?)
    }

    fn f32(&self, key: &str) -> anyhow::Result<CudaSlice<f32>> {
        Ok(self.stream().clone_htod(&self.fixture.f32(key)?.0)?)
    }
}

#[test]
fn embedding_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let (ids, _) = s.fixture.i32("embedding.ids")?;
    let (want, _) = s.fixture.bf16("embedding.out")?;
    let ids = s.stream().clone_htod(&ids)?;
    let table = s.bf16("embedding.table")?;
    let mut out = s.stream().alloc_zeros::<bf16>(want.len())?;
    s.ops.embedding(s.stream(), &ids, &table, &mut out)?;
    assert_bf16_close("embedding", &s.stream().clone_dtoh(&out)?, &want, 0);
    Ok(())
}

#[test]
fn softcapped_swiglu_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let (want, _) = s.fixture.bf16("swiglu.out")?;
    let gate_up = s.bf16("swiglu.gate_up")?;
    let mut out = s.stream().alloc_zeros::<bf16>(want.len())?;
    s.ops
        .softcapped_swiglu(s.stream(), &gate_up, &mut out, 4.0, 25.0)?;
    assert_bf16_close("swiglu", &s.stream().clone_dtoh(&out)?, &want, 1);
    Ok(())
}

#[test]
fn attention_gate_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let (want, _) = s.fixture.bf16("attention_gate.out")?;
    let attn = s.bf16("attention_gate.attn")?;
    let gate = s.bf16("attention_gate.gate")?;
    let mut out = s.stream().alloc_zeros::<bf16>(want.len())?;
    s.ops.attention_gate(s.stream(), &attn, &gate, &mut out)?;
    assert_bf16_close("attention_gate", &s.stream().clone_dtoh(&out)?, &want, 1);
    Ok(())
}

#[test]
fn log_scale_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    for (name, start) in [("log_scale.low", 8188), ("log_scale.high", 1_000_000)] {
        for (part, width) in [("query", 24 * 128), ("relative", 24 * 16)] {
            let (want, _) = s.fixture.bf16(&format!("{name}.{part}_out"))?;
            let mut x = s.bf16(&format!("{name}.{part}"))?;
            s.ops
                .log_scale(s.stream(), &mut x, width, start, 0.1, 8192.0)?;
            let got = s.stream().clone_dtoh(&x)?;
            assert_bf16_close(&format!("{name}.{part}"), &got, &want, 1);
        }
    }
    Ok(())
}

#[test]
fn residual_adds_match_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let (want, _) = s.fixture.f32("add_bf16_to_f32.out")?;
    let a = s.bf16("add_bf16_to_f32.a")?;
    let b = s.bf16("add_bf16_to_f32.b")?;
    let mut out = s.stream().alloc_zeros::<f32>(want.len())?;
    s.ops.add_bf16_to_f32(s.stream(), &a, &b, &mut out)?;
    assert_f32_ulps("add_bf16_to_f32", &s.stream().clone_dtoh(&out)?, &want, 0);

    let (want, _) = s.fixture.bf16("add_to_bf16.out")?;
    let residual = s.f32("add_to_bf16.residual")?;
    let x = s.bf16("add_to_bf16.x")?;
    let mut out = s.stream().alloc_zeros::<bf16>(want.len())?;
    s.ops.add_to_bf16(s.stream(), &residual, &x, &mut out)?;
    assert_bf16_close("add_to_bf16", &s.stream().clone_dtoh(&out)?, &want, 0);
    Ok(())
}

#[test]
fn moe_combine_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let (want, _) = s.fixture.bf16("moe_combine.out")?;
    let experts = s.bf16("moe_combine.experts")?;
    let weights = s.f32("moe_combine.weights")?;
    let mut out = s.stream().alloc_zeros::<bf16>(want.len())?;
    s.ops
        .moe_combine(s.stream(), &experts, &weights, &mut out)?;
    assert_bf16_close("moe_combine", &s.stream().clone_dtoh(&out)?, &want, 0);
    Ok(())
}

/// The `k`-th finite BF16 bit pattern: positives 0x0000..0x7F7F, then negatives.
/// Mirrors `finite_bf16_bits` in the fixture script.
fn finite_bf16(k: usize) -> bf16 {
    let bits = if k < 0x7F80 { k } else { k - 0x7F80 + 0x8000 };
    bf16::from_bits(bits as u16)
}

#[test]
fn logit_softcap_matches_mlx() -> anyhow::Result<()> {
    // The fixture stores MLX's output for each finite BF16 input. The two rows below
    // are the ones the fixture script ran through `Model.__call__`; between them they
    // hold every finite BF16 value.
    const N_FINITE: usize = 2 * 0x7F80;
    let s = Setup::new()?;
    let (table, _) = s.fixture.f32("logit_softcap.table")?;
    let index = |row: usize, col: usize| (col * 7 + row * 4093) % N_FINITE;
    let logits: Vec<bf16> = (0..2)
        .flat_map(|row| (0..PADDED_VOCAB).map(move |col| finite_bf16(index(row, col))))
        .collect();
    let want: Vec<f32> = (0..2)
        .flat_map(|row| (0..VOCAB).map(move |col| index(row, col)))
        .map(|k| table[k])
        .collect();

    let logits = s.stream().clone_htod(&logits)?;
    let mut out = s.stream().alloc_zeros::<f32>(want.len())?;
    s.ops.logit_softcap(s.stream(), &logits, &mut out, 30.0)?;
    // Every finite input is covered, so 2 ulps is the measured worst case of CUDA's
    // tanh against Metal's here, not a guess.
    assert_f32_ulps("logit_softcap", &s.stream().clone_dtoh(&out)?, &want, 2);
    Ok(())
}
