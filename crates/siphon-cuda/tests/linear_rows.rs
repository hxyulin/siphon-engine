//! The multi-row dense products of `ops::linear_rows`.
//!
//! - Against an F64 reference over the (quantized) weights: any order of k F32 additions
//!   of exact products is within `gamma * sum |w_i x_i|` of the exact dot product, with
//!   `gamma = k u / (1 - k u)` and `u = 2^-24` (Higham, eq. 3.5); the FP8 scale adds half
//!   an F32 ulp and the BF16 output half a BF16 ulp.
//! - Row invariance: each row's output is the same, bit for bit, for every m from 1 to 8,
//!   and whatever the other rows hold.
//! - Against the decode GEMVs at one row and the prefill GEMM at m rows, which sum in
//!   other orders: printed, and bounded by the reference bound of both.

#![cfg(target_os = "linux")]

mod common;

use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::{
        linear::{Gemm, Gemv, Linear, Start},
        linear_fp8::{Fp8Gemv, Fp8Matrix, LinearFp8},
        linear_rows::{LinearRows, MAX_ROWS, RowsProduct},
    },
};

use common::pseudo_random_bf16;

/// Rows of the test weights and their k: partial tiles of 16 rows, and every entry point.
const SHAPES: &[(usize, usize)] = &[(1000, 2304), (24, 2304), (2304, 3072), (64, 4608)];

struct Setup {
    gpu: Gpu,
    rows: LinearRows,
    linear: Linear,
    fp8: LinearFp8,
}

fn setup() -> anyhow::Result<Setup> {
    let gpu = Gpu::new(0)?;
    let ctx = gpu.context();
    Ok(Setup {
        rows: LinearRows::load(ctx)?,
        linear: Linear::load(ctx)?,
        fp8: LinearFp8::load(ctx)?,
        gpu,
    })
}

/// Weights with the spread of real projections: mostly small, a few large per row, so
/// that FP8 rows have subnormal values.
fn weights(n: usize, k: usize, seed: u32) -> Vec<bf16> {
    let mut w = pseudo_random_bf16(n * k, seed, 0.05, 0.0);
    for r in 0..n {
        w[r * k + (r * 7) % k] = bf16::from_f32(0.8);
    }
    w
}

/// The weight as F64 values: BF16, or the FP8 matrix's E4M3 values times their scales.
fn dense(s: &Setup, w: &[bf16], fp8: Option<&Fp8Matrix>) -> anyhow::Result<Vec<f64>> {
    let Some(m) = fp8 else {
        return Ok(w.iter().map(|v| f64::from(v.to_f32())).collect());
    };
    let stream = s.gpu.stream();
    let bytes = stream.clone_dtoh(&m.values())?;
    let scales: Vec<f32> = stream
        .clone_dtoh(&m.scales())?
        .as_chunks()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let k = m.k();
    Ok(bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| f64::from(e4m3(b)) * f64::from(scales[i / k]))
        .collect())
}

fn e4m3(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 3) & 0xf);
    let mantissa = f32::from(bits & 7);
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    sign * magnitude
}

/// Runs the multi-row product of `rows` rows of `input`.
fn run(
    setup: &Setup,
    weight: &cudarc::driver::CudaSlice<bf16>,
    fp8: Option<&Fp8Matrix>,
    input: &[bf16],
    out_rows: usize,
    rows: usize,
) -> anyhow::Result<Vec<bf16>> {
    let stream = setup.gpu.stream();
    let input = stream.clone_htod(input)?;
    let mut out = stream.alloc_zeros::<bf16>(rows * out_rows)?;
    let mut p = [match fp8 {
        Some(q) => RowsProduct::fp8(q, &input, &mut out),
        None => RowsProduct::bf16(weight, &input, &mut out),
    }];
    setup.rows.run(stream, &mut p, rows, Start::Serial)?;
    Ok(stream.clone_dtoh(&out)?)
}

/// Largest error of `got` against the F64 product, in units of each output's bound.
fn check_bound(
    label: &str,
    got: &[bf16],
    weight: &[f64],
    input: &[bf16],
    out_rows: usize,
    width: usize,
    rows: usize,
) -> f64 {
    let gamma = {
        let ku = width as f64 * 2f64.powi(-24);
        ku / (1.0 - ku)
    };
    let mut worst = 0f64;
    for c in 0..rows {
        let xr = &input[c * width..(c + 1) * width];
        for r in 0..out_rows {
            let wr = &weight[r * width..(r + 1) * width];
            let (mut exact, mut abs) = (0f64, 0f64);
            for (a, b) in wr.iter().zip(xr) {
                let p = a * f64::from(b.to_f32());
                exact += p;
                abs += p.abs();
            }
            let bound = gamma * abs + exact.abs() * (2f64.powi(-24) + 2f64.powi(-8)) + 1e-30;
            let err = (f64::from(got[c * out_rows + r].to_f32()) - exact).abs();
            worst = worst.max(err / bound);
        }
    }
    assert!(worst <= 1.0, "{label}: error {worst} times the bound");
    worst
}

#[test]
fn rows_match_reference() -> anyhow::Result<()> {
    let s = setup()?;
    let stream = s.gpu.stream();
    for (i, &(n, k)) in SHAPES.iter().enumerate() {
        let w = weights(n, k, i as u32 + 1);
        let w_dev = stream.clone_htod(&w)?;
        let q = s.fp8.quantize_any(stream, &w_dev, n, k)?;
        let x = pseudo_random_bf16(MAX_ROWS * k, 100 + i as u32, 2.0, 0.0);
        for fp8 in [None, Some(&q)] {
            let wd = dense(&s, &w, fp8)?;
            let name = if fp8.is_some() { "fp8" } else { "bf16" };
            let mut worst = 0f64;
            for m in 1..=MAX_ROWS {
                let got = run(&s, &w_dev, fp8, &x[..m * k], n, m)?;
                worst = worst.max(check_bound(
                    &format!("{name} {n} x {k}, m = {m}"),
                    &got,
                    &wd,
                    &x,
                    n,
                    k,
                    m,
                ));
            }
            eprintln!("{name} {n} x {k}: worst error {worst:.3} of the bound");
        }
    }
    Ok(())
}

#[test]
fn rows_are_invariant() -> anyhow::Result<()> {
    let s = setup()?;
    let stream = s.gpu.stream();
    for (i, &(n, k)) in SHAPES.iter().enumerate() {
        let w = weights(n, k, i as u32 + 1);
        let w_dev = stream.clone_htod(&w)?;
        let q = s.fp8.quantize_any(stream, &w_dev, n, k)?;
        let x = pseudo_random_bf16(MAX_ROWS * k, 200 + i as u32, 2.0, 0.0);
        // The same first row, other values in the other rows, with a few huge ones.
        let mut other = pseudo_random_bf16(MAX_ROWS * k, 300 + i as u32, 50.0, 1.0);
        other[..k].copy_from_slice(&x[..k]);
        other[3 * k] = bf16::from_f32(3e38);
        for fp8 in [None, Some(&q)] {
            let name = if fp8.is_some() { "fp8" } else { "bf16" };
            let full = run(&s, &w_dev, fp8, &x, n, MAX_ROWS)?;
            for m in 1..MAX_ROWS {
                let part = run(&s, &w_dev, fp8, &x[..m * k], n, m)?;
                assert!(
                    part.iter()
                        .zip(&full)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{name} {n} x {k}: rows of m = {m} differ from m = 8"
                );
            }
            // Each row alone, as a one-row pass would run it.
            for r in 0..MAX_ROWS {
                let one = run(&s, &w_dev, fp8, &x[r * k..(r + 1) * k], n, 1)?;
                assert!(
                    one.iter()
                        .zip(&full[r * n..(r + 1) * n])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{name} {n} x {k}: row {r} alone differs from row {r} of 8"
                );
            }
            let mixed = run(&s, &w_dev, fp8, &other, n, MAX_ROWS)?;
            assert!(
                mixed[..n]
                    .iter()
                    .zip(&full[..n])
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name} {n} x {k}: row 0 depends on the other rows"
            );
            eprintln!("{name} {n} x {k}: rows bit-identical for m = 1 to 8 and other data");
        }
    }
    Ok(())
}

/// Largest difference in BF16 ulps and its relative size against the row's largest output.
fn compare(label: &str, got: &[bf16], want: &[bf16]) {
    let ulps = got
        .iter()
        .zip(want)
        .map(|(a, b)| common::bf16_ulps(*a, *b))
        .max()
        .unwrap_or(0);
    let differ = got
        .iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!(
        "{label}: {differ} of {} outputs differ, at most {ulps} BF16 ulps",
        got.len()
    );
}

#[test]
fn rows_against_decode_and_prefill() -> anyhow::Result<()> {
    let s = setup()?;
    let stream = s.gpu.stream();
    for (i, &(n, k)) in SHAPES.iter().enumerate() {
        let w = weights(n, k, i as u32 + 1);
        let w_dev = stream.clone_htod(&w)?;
        let x = pseudo_random_bf16(MAX_ROWS * k, 400 + i as u32, 2.0, 0.0);
        let x_dev = stream.clone_htod(&x)?;

        // One row against the BF16 and FP8 decode GEMVs.
        let row0 = run(&s, &w_dev, None, &x[..k], n, 1)?;
        let x0 = stream.clone_htod(&x[..k])?;
        let mut out = stream.alloc_zeros::<bf16>(n)?;
        s.linear.gemv_bf16(
            stream,
            &mut [Gemv::new(&w_dev, &x0, &mut out)],
            Start::Serial,
        )?;
        compare(
            &format!("bf16 {n} x {k}, m = 1 against the decode GEMV"),
            &row0,
            &stream.clone_dtoh(&out)?,
        );
        if s.fp8.supports(k) {
            let q = s.fp8.quantize(stream, &w_dev, n, k)?;
            let row0 = run(&s, &w_dev, Some(&q), &x[..k], n, 1)?;
            s.fp8.gemv(
                stream,
                &mut [Fp8Gemv::new(&q, &x0, &mut out)],
                Start::Serial,
            )?;
            compare(
                &format!("fp8 {n} x {k}, m = 1 against the decode GEMV"),
                &row0,
                &stream.clone_dtoh(&out)?,
            );
        }

        // m rows against the prefill GEMM.
        for m in [2, 4, 8] {
            let got = run(&s, &w_dev, None, &x[..m * k], n, m)?;
            let xm = x_dev.slice(..m * k);
            let mut out = stream.alloc_zeros::<bf16>(m * n)?;
            s.linear
                .gemm_bf16(stream, &mut [Gemm::new(&w_dev, &xm, &mut out)], m)?;
            compare(
                &format!("bf16 {n} x {k}, m = {m} against the prefill GEMM"),
                &got,
                &stream.clone_dtoh(&out)?,
            );
        }
    }
    Ok(())
}
