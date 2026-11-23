//! Conv kernels against MLX outputs from `reference/fixtures/conv.py`.
//!
//! Each case runs a prefill from a zero state, then one decode row from the state the prefill
//! left on the GPU, and compares both outputs and both states with MLX.

#![cfg(target_os = "linux")]

mod common;

use std::sync::Arc;

use common::{
    Fixture, assert_bf16_close, assert_f32_within_ulps, pseudo_random, pseudo_random_bf16,
};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use inkling_cuda::{
    Gpu,
    ops::{conv::Conv, elementwise::Elementwise},
};

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

const HIDDEN: usize = 2304;
const TOP_K: usize = 8;

/// The fused post-attention conv and residual add against `short_conv` followed by the
/// elementwise add (or cast, at a block start), over two decode steps so the second reads
/// the state the first wrote. The fused kernel repeats the same rounded operations, so the
/// outputs and states must match bit for bit.
#[test]
fn short_conv_residual_matches_separate_kernels() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let conv = Conv::load(gpu.context())?;
    let elementwise = Elementwise::load(gpu.context())?;
    let weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN * 4, 1, 0.5, 0.0))?;
    let start = pseudo_random(3 * HIDDEN, 2, 2.0);
    let mut state_fused = stream.clone_htod(&start)?;
    let mut state_separate = stream.clone_htod(&start)?;
    for (step, with_hidden) in [(0u32, true), (1, false), (2, true)] {
        let x = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 10 + step, 3.0, 0.0))?;
        let hidden = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 20 + step, 3.0, 0.0))?;
        let mut conv_out = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut want = stream.alloc_zeros::<f32>(HIDDEN)?;
        conv.short_conv(stream, &x, &weight, &mut state_separate, &mut conv_out)?;
        if with_hidden {
            elementwise.add_bf16_to_f32(stream, &hidden, &conv_out, &mut want)?;
        } else {
            let widened: Vec<f32> = stream
                .clone_dtoh(&conv_out)?
                .iter()
                .map(|v| v.to_f32())
                .collect();
            want = stream.clone_htod(&widened)?;
        }
        let mut got = stream.alloc_zeros::<f32>(HIDDEN)?;
        conv.short_conv_residual(
            stream,
            &x,
            &weight,
            &mut state_fused,
            with_hidden.then_some(&hidden),
            &mut got,
        )?;
        let label = format!("step {step}");
        assert_f32_within_ulps(
            &label,
            &stream.clone_dtoh(&got)?,
            &stream.clone_dtoh(&want)?,
            0,
        );
        assert_f32_within_ulps(
            &format!("{label} state"),
            &stream.clone_dtoh(&state_fused)?,
            &stream.clone_dtoh(&state_separate)?,
            0,
        );
    }
    Ok(())
}

/// The fused MoE combine, post-MLP conv and residual add against `moe_combine`,
/// `short_conv` and `add_to_bf16`, over two decode steps. Same rounded operations, so bit
/// for bit.
#[test]
fn moe_conv_residual_matches_separate_kernels() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let conv = Conv::load(gpu.context())?;
    let elementwise = Elementwise::load(gpu.context())?;
    let weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN * 4, 3, 0.5, 0.0))?;
    let start = pseudo_random(3 * HIDDEN, 4, 2.0);
    let mut state_fused = stream.clone_htod(&start)?;
    let mut state_separate = stream.clone_htod(&start)?;
    for step in 0..2u32 {
        let experts =
            stream.clone_htod(&pseudo_random_bf16(TOP_K * HIDDEN, 30 + step, 2.0, 0.0))?;
        let routing: Vec<f32> = pseudo_random(TOP_K, 40 + step, 1.0)
            .iter()
            .map(|v| v.abs())
            .collect();
        let routing = stream.clone_htod(&routing)?;
        let residual = stream.clone_htod(&pseudo_random(HIDDEN, 50 + step, 4.0))?;
        let mut combined = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut conv_out = stream.alloc_zeros::<bf16>(HIDDEN)?;
        let mut want = stream.alloc_zeros::<bf16>(HIDDEN)?;
        elementwise.moe_combine(stream, &experts, &routing, &mut combined)?;
        conv.short_conv(
            stream,
            &combined,
            &weight,
            &mut state_separate,
            &mut conv_out,
        )?;
        elementwise.add_to_bf16(stream, &residual, &conv_out, &mut want)?;
        let mut got = stream.alloc_zeros::<bf16>(HIDDEN)?;
        conv.moe_conv_residual(
            stream,
            &experts,
            &routing,
            &weight,
            &mut state_fused,
            &residual,
            &mut got,
        )?;
        let label = format!("step {step}");
        assert_bf16_close(
            &label,
            &stream.clone_dtoh(&got)?,
            &stream.clone_dtoh(&want)?,
            0,
        );
        assert_f32_within_ulps(
            &format!("{label} state"),
            &stream.clone_dtoh(&state_fused)?,
            &stream.clone_dtoh(&state_separate)?,
            0,
        );
    }
    Ok(())
}
