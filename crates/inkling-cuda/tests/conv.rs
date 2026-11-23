//! Conv kernels against MLX outputs from `reference/fixtures/conv.py`.
//!
//! Each case runs a prefill from a zero state, then one decode row from the state the prefill
//! left on the GPU, and compares both outputs and both states with MLX.

#![cfg(target_os = "linux")]

mod common;

use std::sync::Arc;

use common::{Fixture, assert_bf16_close};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use inkling_cuda::{Gpu, ops::conv::Conv};

/// Rows in the fixture's prefill; the decode step is the row after them.
const PREFILL: usize = 10;

/// Fails unless every element has the same bits as MLX's, and reports how many differ.
fn assert_f32_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    let differing = got
        .iter()
        .zip(want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    eprintln!("{label}: {differing} of {} elements differ", want.len());
    assert_eq!(differing, 0, "{label}: state differs from MLX");
}

/// Splits `x` into prefill and decode rows, runs `conv` on each with one shared state, and
/// checks outputs and states against `<prefix>.{prefill,decode}.{out,state}`.
fn check_prefill_then_decode(
    fixture: &Fixture,
    prefix: &str,
    stream: &Arc<CudaStream>,
    dims: usize,
    mut conv: impl FnMut(usize, usize, &mut CudaSlice<f32>, &mut CudaSlice<bf16>) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let mut state = stream.alloc_zeros::<f32>(3 * dims)?;
    for (step, rows) in [("prefill", 0..PREFILL), ("decode", PREFILL..PREFILL + 1)] {
        let mut out = stream.alloc_zeros::<bf16>(rows.len() * dims)?;
        conv(rows.start, rows.len(), &mut state, &mut out)?;
        let (want, _) = fixture.bf16(&format!("{prefix}.{step}.out"))?;
        assert_bf16_close(
            &format!("{prefix}.{step}.out"),
            &stream.clone_dtoh(&out)?,
            &want,
            0,
        );
        let (want, _) = fixture.f32(&format!("{prefix}.{step}.state"))?;
        assert_f32_exact(
            &format!("{prefix}.{step}.state"),
            &stream.clone_dtoh(&state)?,
            &want,
        );
    }
    Ok(())
}

fn short_conv_matches_mlx(dims: usize) -> anyhow::Result<()> {
    let fixture = Fixture::load("conv")?;
    let prefix = format!("short_conv_{dims}");
    let (x, _) = fixture.bf16(&format!("{prefix}.x"))?;
    let (weight, shape) = fixture.bf16(&format!("{prefix}.weight"))?;
    assert_eq!(shape, [dims, 4, 1]);

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let ops = Conv::load(gpu.context())?;
    let weight_dev = stream.clone_htod(&weight)?;
    check_prefill_then_decode(
        &fixture,
        &prefix,
        stream,
        dims,
        |start, rows, state, out| {
            let x_dev = stream.clone_htod(&x[start * dims..(start + rows) * dims])?;
            ops.short_conv(stream, &x_dev, &weight_dev, state, out)
        },
    )
}

#[test]
fn short_conv_2304_matches_mlx() -> anyhow::Result<()> {
    short_conv_matches_mlx(2304)
}

#[test]
fn short_conv_1024_matches_mlx() -> anyhow::Result<()> {
    short_conv_matches_mlx(1024)
}

#[test]
fn dynamic_short_conv_matches_mlx() -> anyhow::Result<()> {
    const DIMS: usize = 1024;
    const DYNAMIC: usize = 32;
    let fixture = Fixture::load("conv")?;
    let (x, _) = fixture.bf16("dynamic_short_conv.x")?;
    let (dynamic, shape) = fixture.bf16("dynamic_short_conv.dynamic")?;
    assert_eq!(shape, [PREFILL + 1, DYNAMIC]);
    let (static_weight, shape) = fixture.f32("dynamic_short_conv.static_weight")?;
    assert_eq!(shape, [4, DIMS]);

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let ops = Conv::load(gpu.context())?;
    let static_dev = stream.clone_htod(&static_weight)?;
    check_prefill_then_decode(
        &fixture,
        "dynamic_short_conv",
        stream,
        DIMS,
        |start, rows, state, out| {
            let x_dev = stream.clone_htod(&x[start * DIMS..(start + rows) * DIMS])?;
            let dynamic_dev =
                stream.clone_htod(&dynamic[start * DYNAMIC..(start + rows) * DYNAMIC])?;
            ops.dynamic_short_conv(stream, &x_dev, &dynamic_dev, &static_dev, state, out)
        },
    )
}
