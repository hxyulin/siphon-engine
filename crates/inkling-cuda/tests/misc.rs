//! Message-gate combine, casts and argmax against MLX outputs from `reference/fixtures/misc.py`.
//!
//! Casts, argmax and every message-gate step before the combine are bit-exact. The combine
//! uses CUDA's tanh, which rounds differently from Metal's `precise::tanh`, so its output is
//! checked against a bound on the error that a few-ulp tanh difference can cause.

#![cfg(target_os = "linux")]

mod common;

use common::{Fixture, assert_bf16_close};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use inkling_cuda::{
    Gpu,
    ops::{linear::Linear, misc::Misc, norm::Norm},
};

const HIDDEN: usize = 2304;
const GATE_RANK: usize = 64;
const EPS: f32 = 1e-6;
/// Allowed combine error, as a multiple of `|h| + |m|`. Metal's tanh is up to 3 F32 ulps from
/// the true value, which is at most 3 * 2^-24 for a result below 1.
const COMBINE_TOLERANCE: f32 = 1.0 / (1 << 21) as f32;

struct Setup {
    fixture: Fixture,
    gpu: Gpu,
    misc: Misc,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        let misc = Misc::load(gpu.context())?;
        Ok(Self {
            fixture: Fixture::load("misc")?,
            gpu,
            misc,
        })
    }

    fn stream(&self) -> &std::sync::Arc<CudaStream> {
        self.gpu.stream()
    }

    fn f32(&self, key: &str) -> anyhow::Result<CudaSlice<f32>> {
        let (host, _) = self.fixture.f32(key)?;
        Ok(self.stream().clone_htod(&host)?)
    }
}

/// Fails unless `got` equals `want` bit for bit, NaNs included.
fn assert_f32_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{label}[{i}]: got {g} ({:#010x}), want {w} ({:#010x})",
            g.to_bits(),
            w.to_bits()
        );
    }
    eprintln!("{label}: all {} elements bit-exact", want.len());
}

/// Fails if any combine output is off by more than `COMBINE_TOLERANCE * (|h| + |m|)`.
fn assert_combine_close(label: &str, got: &[f32], want: &[f32], h: &[f32], m: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = 0.0f32;
    let mut differing = 0;
    for i in 0..want.len() {
        let scale = h[i].abs() + m[i].abs();
        let err = (got[i] - want[i]).abs();
        assert!(
            err <= COMBINE_TOLERANCE * scale,
            "{label}[{i}]: got {}, want {}, |h| + |m| = {scale}",
            got[i],
            want[i]
        );
        worst = worst.max(err / scale);
        differing += usize::from(got[i].to_bits() != want[i].to_bits());
    }
    eprintln!(
        "{label}: {differing} of {} elements differ; worst error {worst:e} x (|h| + |m|)",
        want.len()
    );
}

#[test]
fn message_gate_combine_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let (h, _) = s.fixture.f32("combine.h")?;
    let (m, _) = s.fixture.f32("combine.m")?;
    let (want, _) = s.fixture.f32("combine.out")?;
    let mut out = s.stream().alloc_zeros::<f32>(want.len())?;
    s.misc.message_gate_combine(
        s.stream(),
        &s.stream().clone_htod(&h)?,
        &s.stream().clone_htod(&m)?,
        &s.f32("combine.f_pre")?,
        &s.f32("combine.g_pre")?,
        &mut out,
    )?;
    assert_combine_close("combine", &s.stream().clone_dtoh(&out)?, &want, &h, &m);
    Ok(())
}

#[test]
fn cast_f32_to_bf16_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let (want, _) = s.fixture.bf16("cast_f32_to_bf16.out")?;
    let mut out = s.stream().alloc_zeros::<bf16>(want.len())?;
    s.misc
        .cast_f32_to_bf16(s.stream(), &s.f32("cast_f32_to_bf16.x")?, &mut out)?;
    let got = s.stream().clone_dtoh(&out)?;
    // Compare bits: the inputs include NaNs, and MLX writes one NaN pattern for all of them.
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "cast_f32_to_bf16[{i}]: got {:#06x}, want {:#06x}",
            g.to_bits(),
            w.to_bits()
        );
    }
    eprintln!("cast_f32_to_bf16: all {} elements bit-exact", want.len());
    Ok(())
}

#[test]
fn cast_bf16_to_f32_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let (x, _) = s.fixture.bf16("cast_bf16_to_f32.x")?;
    let (want, _) = s.fixture.f32("cast_bf16_to_f32.out")?;
    let mut out = s.stream().alloc_zeros::<f32>(want.len())?;
    s.misc
        .cast_bf16_to_f32(s.stream(), &s.stream().clone_htod(&x)?, &mut out)?;
    assert_f32_exact("cast_bf16_to_f32", &s.stream().clone_dtoh(&out)?, &want);
    Ok(())
}

#[test]
fn argmax_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    for name in ["argmax.vocab", "argmax.special"] {
        let (x, shape) = s.fixture.f32(&format!("{name}.x"))?;
        let (want, _) = s.fixture.i32(&format!("{name}.out"))?;
        let mut out = s.stream().alloc_zeros::<i32>(want.len())?;
        s.misc
            .argmax_f32(s.stream(), &s.stream().clone_htod(&x)?, &mut out, shape[1])?;
        assert_eq!(s.stream().clone_dtoh(&out)?, want, "{name}");
    }
    Ok(())
}

/// Runs `MessageGate.__call__` for one row with the norm, linear and misc kernels, checking
/// each intermediate against MLX.
#[test]
fn message_gate_chain_matches_mlx() -> anyhow::Result<()> {
    let s = Setup::new()?;
    let stream = s.stream();
    let norm = Norm::load(s.gpu.context())?;
    let linear = Linear::load(s.gpu.context())?;
    let fixture = &s.fixture;

    // 1. Unweighted RMS norm of h and m, side by side in one 4608-wide row.
    let (h, _) = fixture.f32("gate.h")?;
    let (m, _) = fixture.f32("gate.m")?;
    let h_dev = stream.clone_htod(&h)?;
    let m_dev = stream.clone_htod(&m)?;
    let mut h_norm = stream.alloc_zeros::<f32>(HIDDEN)?;
    let mut m_norm = stream.alloc_zeros::<f32>(HIDDEN)?;
    norm.unweighted_rms_norm(stream, &h_dev, &mut h_norm, EPS)?;
    norm.unweighted_rms_norm(stream, &m_dev, &mut m_norm, EPS)?;
    let mut states = stream.alloc_zeros::<f32>(2 * HIDDEN)?;
    stream.memcpy_dtod(&h_norm, &mut states.slice_mut(..HIDDEN))?;
    stream.memcpy_dtod(&m_norm, &mut states.slice_mut(HIDDEN..))?;
    let (want, _) = fixture.f32("gate.states")?;
    assert_f32_exact("gate.states", &stream.clone_dtoh(&states)?, &want);

    // 2-3. Cast to BF16, down projection, cast back to F32.
    let mut states_bf16 = stream.alloc_zeros::<bf16>(2 * HIDDEN)?;
    s.misc.cast_f32_to_bf16(stream, &states, &mut states_bf16)?;
    let (down_weight, _) = fixture.bf16("gate.down_weight")?;
    let mut down = stream.alloc_zeros::<bf16>(GATE_RANK)?;
    linear.gemv_bf16(
        stream,
        &stream.clone_htod(&down_weight)?,
        &states_bf16,
        &mut down,
        GATE_RANK,
        2 * HIDDEN,
    )?;
    let (want, _) = fixture.bf16("gate.down")?;
    assert_bf16_close("gate.down", &stream.clone_dtoh(&down)?, &want, 0);
    let mut down_f32 = stream.alloc_zeros::<f32>(GATE_RANK)?;
    s.misc.cast_bf16_to_f32(stream, &down, &mut down_f32)?;

    // 4. Forget and gain projections, with the BF16 weights widened to F32 as the loader does.
    let project = |key: &str| -> anyhow::Result<CudaSlice<f32>> {
        let (weight, _) = fixture.bf16(&format!("gate.{key}_weight"))?;
        let weight: Vec<f32> = weight.iter().map(|w| w.to_f32()).collect();
        let mut out = stream.alloc_zeros::<f32>(HIDDEN)?;
        linear.gemv_f32(
            stream,
            &stream.clone_htod(&weight)?,
            &down_f32,
            &mut out,
            HIDDEN,
            GATE_RANK,
        )?;
        Ok(out)
    };
    let f_pre = project("forget")?;
    let g_pre = project("gain")?;
    let (want, _) = fixture.f32("gate.f_pre")?;
    assert_f32_exact("gate.f_pre", &stream.clone_dtoh(&f_pre)?, &want);
    let (want, _) = fixture.f32("gate.g_pre")?;
    assert_f32_exact("gate.g_pre", &stream.clone_dtoh(&g_pre)?, &want);

    // 5. Combine.
    let mut out = stream.alloc_zeros::<f32>(HIDDEN)?;
    s.misc
        .message_gate_combine(stream, &h_dev, &m_dev, &f_pre, &g_pre, &mut out)?;
    let (want, _) = fixture.f32("gate.out")?;
    assert_combine_close("gate.out", &stream.clone_dtoh(&out)?, &want, &h, &m);
    Ok(())
}
