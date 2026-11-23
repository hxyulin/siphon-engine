//! Norm kernels against MLX outputs from `reference/fixtures/norm.py`.

#![cfg(target_os = "linux")]

mod common;

use common::{Fixture, assert_bf16_close};
use half::bf16;
use inkling_cuda::{Gpu, ops::norm::Norm};

#[test]
fn rms_norm_matches_mlx() -> anyhow::Result<()> {
    let fixture = Fixture::load("norm")?;
    let (x, _) = fixture.bf16("rms_norm.x")?;
    let (weight, _) = fixture.bf16("rms_norm.weight")?;
    let (want, _) = fixture.bf16("rms_norm.out")?;

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let norm = Norm::load(gpu.context())?;
    let x_dev = stream.clone_htod(&x)?;
    let weight_dev = stream.clone_htod(&weight)?;
    let mut out_dev = stream.alloc_zeros::<bf16>(x.len())?;
    norm.rms_norm(stream, &x_dev, &weight_dev, &mut out_dev, 1e-6)?;
    let got = stream.clone_dtoh(&out_dev)?;

    assert_bf16_close("rms_norm", &got, &want, 1);
    Ok(())
}
