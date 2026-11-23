//! Norm kernels against MLX outputs from `reference/fixtures/norm.py`.
//!
//! The kernels copy MLX's summation order, so every comparison requires bit-exact output.

#![cfg(target_os = "linux")]

mod common;

use common::{Fixture, assert_bf16_close};
use half::bf16;
use inkling_cuda::{Gpu, ops::norm::Norm};

const EPS: f32 = 1e-6;
const QK_THRESHOLD: f32 = 22.627_417;
const REL_THRESHOLD: f32 = 8.0;

/// Distance between two F32 values in units in the last place, ordering negatives correctly.
fn f32_ulps(a: f32, b: f32) -> u64 {
    let key = |v: f32| {
        let bits = i64::from(v.to_bits());
        if bits & 0x8000_0000 == 0 {
            bits
        } else {
            0x8000_0000 - bits
        }
    };
    key(a).abs_diff(key(b))
}

/// Fails unless `got` equals `want` bit for bit.
fn assert_f32_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{label}[{i}]: got {g}, want {w} ({} ulps)",
            f32_ulps(g, w)
        );
    }
    eprintln!("{label}: all {} elements bit-exact", want.len());
}

struct Setup {
    fixture: Fixture,
    gpu: Gpu,
    norm: Norm,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        let norm = Norm::load(gpu.context())?;
        Ok(Self {
            fixture: Fixture::load("norm")?,
            gpu,
            norm,
        })
    }
}

#[test]
fn rms_norm_matches_mlx() -> anyhow::Result<()> {
    let Setup { fixture, gpu, norm } = Setup::new()?;
    let (x, _) = fixture.bf16("rms_norm.x")?;
    let (weight, _) = fixture.bf16("rms_norm.weight")?;
    let (want, _) = fixture.bf16("rms_norm.out")?;

    let stream = gpu.stream();
    let x_dev = stream.clone_htod(&x)?;
    let weight_dev = stream.clone_htod(&weight)?;
    let mut out_dev = stream.alloc_zeros::<bf16>(x.len())?;
    norm.rms_norm(stream, &x_dev, &weight_dev, &mut out_dev, EPS)?;
    let got = stream.clone_dtoh(&out_dev)?;

    assert_bf16_close("rms_norm", &got, &want, 0);
    Ok(())
}

#[test]
fn rms_norm_f32_matches_mlx() -> anyhow::Result<()> {
    let Setup { fixture, gpu, norm } = Setup::new()?;
    let (x, _) = fixture.f32("rms_norm_f32.x")?;
    let (weight, _) = fixture.bf16("rms_norm_f32.weight")?;
    let (want, _) = fixture.bf16("rms_norm_f32.out")?;

    let stream = gpu.stream();
    let x_dev = stream.clone_htod(&x)?;
    let weight_dev = stream.clone_htod(&weight)?;
    let mut out_dev = stream.alloc_zeros::<bf16>(x.len())?;
    norm.rms_norm_f32(stream, &x_dev, &weight_dev, &mut out_dev, EPS)?;
    let got = stream.clone_dtoh(&out_dev)?;

    assert_bf16_close("rms_norm_f32", &got, &want, 0);
    Ok(())
}

#[test]
fn unweighted_rms_norm_matches_mlx() -> anyhow::Result<()> {
    let Setup { fixture, gpu, norm } = Setup::new()?;
    let (x, _) = fixture.f32("unweighted_rms_norm.x")?;
    let (want, _) = fixture.f32("unweighted_rms_norm.out")?;

    let stream = gpu.stream();
    let x_dev = stream.clone_htod(&x)?;
    let mut out_dev = stream.alloc_zeros::<f32>(x.len())?;
    norm.unweighted_rms_norm(stream, &x_dev, &mut out_dev, EPS)?;
    let got = stream.clone_dtoh(&out_dev)?;

    assert_f32_exact("unweighted_rms_norm", &got, &want);
    Ok(())
}

#[test]
fn unweighted_rms_norm_heads_matches_mlx() -> anyhow::Result<()> {
    let Setup { fixture, gpu, norm } = Setup::new()?;
    let (x, _) = fixture.bf16("unweighted_rms_norm_heads.x")?;
    let (want, _) = fixture.bf16("unweighted_rms_norm_heads.out")?;

    let stream = gpu.stream();
    let x_dev = stream.clone_htod(&x)?;
    let mut out_dev = stream.alloc_zeros::<bf16>(x.len())?;
    norm.unweighted_rms_norm_heads(stream, &x_dev, &mut out_dev, EPS)?;
    let got = stream.clone_dtoh(&out_dev)?;

    assert_bf16_close("unweighted_rms_norm_heads", &got, &want, 0);
    Ok(())
}

/// Runs one L2 clip case from the fixture. `scaled` says whether the fixture has a scale tensor.
fn check_l2_clip(setup: &Setup, name: &str, scaled: bool, threshold: f32) -> anyhow::Result<()> {
    let Setup { fixture, gpu, norm } = setup;
    let (x, shape) = fixture.bf16(&format!("{name}.x"))?;
    let (want, _) = fixture.bf16(&format!("{name}.out"))?;
    let &[.., heads, head_dim] = shape.as_slice() else {
        anyhow::bail!("{name}.x has shape {shape:?}");
    };

    let stream = gpu.stream();
    let x_dev = stream.clone_htod(&x)?;
    let scale_dev = if scaled {
        let (scale, _) = fixture.bf16(&format!("{name}.scale"))?;
        Some(stream.clone_htod(&scale)?)
    } else {
        None
    };
    let mut out_dev = stream.alloc_zeros::<bf16>(x.len())?;
    norm.l2_clip(
        stream,
        &x_dev,
        scale_dev.as_ref(),
        &mut out_dev,
        heads,
        head_dim,
        threshold,
    )?;
    let got = stream.clone_dtoh(&out_dev)?;

    assert_bf16_close(name, &got, &want, 0);
    Ok(())
}

#[test]
fn l2_clip_matches_mlx() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    // Length 1 is decode; MLX sums fewer than 32 rows with a different kernel than prefill.
    for length in [1, 6] {
        check_l2_clip(&setup, &format!("q_l2_clip.{length}"), true, QK_THRESHOLD)?;
        check_l2_clip(&setup, &format!("k_l2_clip.{length}"), false, QK_THRESHOLD)?;
        check_l2_clip(
            &setup,
            &format!("rel_l2_clip.{length}"),
            true,
            REL_THRESHOLD,
        )?;
    }
    Ok(())
}
