//! Matrix-vector kernels against MLX outputs from `reference/fixtures/linear.py`.
//!
//! The kernels copy the blocking and summation order of MLX's M = 1 kernel, so every
//! comparison requires bit-exact output. The weights come from an integer hash that the
//! fixture script and this file both implement, so full model shapes need no stored weights.

#![cfg(target_os = "linux")]

mod common;

use common::{Fixture, assert_bf16_close};
use half::bf16;
use inkling_cuda::{Gpu, ops::linear::Linear};

const BF16_CASES: &[&str] = &[
    "bf16.q_proj",
    "bf16.kv_proj",
    "bf16.r_proj",
    "bf16.sink_proj",
    "bf16.o_proj",
    "bf16.router",
    "bf16.gate_down",
    "bf16.lm_head",
    "bf16.multi_row",
    "bf16.leftover",
    "bf16.split_leftover",
    "bf16.ragged_n",
    "bf16.narrow",
];

const F32_CASES: &[&str] = &[
    "f32.gate_up",
    "f32.gate_up_rows",
    "f32.wide",
    "f32.split",
    "f32.split_leftover",
];

/// BF16 weight `[n, k]` from the hash in `reference/fixtures/linear.py`.
fn hash_weight(seed: u32, n: usize, k: usize) -> Vec<bf16> {
    let salt = seed.wrapping_mul(0x9E37_79B9);
    (0..n * k)
        .map(|i| {
            let mut x = (i as u32) ^ salt;
            x ^= x >> 16;
            x = x.wrapping_mul(0x7FEB_352D);
            x ^= x >> 15;
            x = x.wrapping_mul(0x846C_A68B);
            x ^= x >> 16;
            let sign = (x >> 31) << 15;
            let exponent = (117 + ((x >> 28) & 7)) << 7;
            let mantissa = (x >> 16) & 0x7F;
            bf16::from_bits((sign | exponent | mantissa) as u16)
        })
        .collect()
}

/// Shape and BF16 weight of one case, checked against the fixture's sample of the weight.
fn case_weight(
    fixture: &Fixture,
    name: &str,
    x_shape: &[usize],
    out_shape: &[usize],
) -> anyhow::Result<(usize, usize, Vec<bf16>)> {
    let (&[_, k], &[_, n]) = (x_shape, out_shape) else {
        anyhow::bail!("{name}: x is {x_shape:?}, out is {out_shape:?}");
    };
    let (seed, _) = fixture.i32(&format!("{name}.seed"))?;
    let weight = hash_weight(u32::try_from(seed[0])?, n, k);
    let (check, _) = fixture.bf16(&format!("{name}.w_check"))?;
    let sample: Vec<bf16> = weight[..32]
        .iter()
        .chain(&weight[weight.len() - 32..])
        .copied()
        .collect();
    anyhow::ensure!(
        sample == check,
        "{name}: weight hash differs from the fixture's"
    );
    Ok((n, k, weight))
}

#[test]
fn gemv_bf16_matches_mlx() -> anyhow::Result<()> {
    let fixture = Fixture::load("linear")?;
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    let stream = gpu.stream();
    for name in BF16_CASES {
        let (x, x_shape) = fixture.bf16(&format!("{name}.x"))?;
        let (want, out_shape) = fixture.bf16(&format!("{name}.out"))?;
        let (n, k, weight) = case_weight(&fixture, name, &x_shape, &out_shape)?;
        let weight_dev = stream.clone_htod(&weight)?;
        let x_dev = stream.clone_htod(&x)?;
        let mut out_dev = stream.alloc_zeros::<bf16>(want.len())?;
        linear.gemv_bf16(stream, &weight_dev, &x_dev, &mut out_dev, n, k)?;
        assert_bf16_close(
            &format!("{name} ({n} x {k})"),
            &stream.clone_dtoh(&out_dev)?,
            &want,
            0,
        );
    }
    Ok(())
}

#[test]
fn gemv_f32_matches_mlx() -> anyhow::Result<()> {
    let fixture = Fixture::load("linear")?;
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    let stream = gpu.stream();
    for name in F32_CASES {
        let (x, x_shape) = fixture.f32(&format!("{name}.x"))?;
        let (want, out_shape) = fixture.f32(&format!("{name}.out"))?;
        let (n, k, weight) = case_weight(&fixture, name, &x_shape, &out_shape)?;
        // The loader widens the BF16 checkpoint weights the same way.
        let weight: Vec<f32> = weight.iter().map(|w| w.to_f32()).collect();
        let weight_dev = stream.clone_htod(&weight)?;
        let x_dev = stream.clone_htod(&x)?;
        let mut out_dev = stream.alloc_zeros::<f32>(want.len())?;
        linear.gemv_f32(stream, &weight_dev, &x_dev, &mut out_dev, n, k)?;
        let got = stream.clone_dtoh(&out_dev)?;
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(g.to_bits() == w.to_bits(), "{name}[{i}]: got {g}, want {w}");
        }
        eprintln!("{name} ({n} x {k}): all {} elements bit-exact", want.len());
    }
    Ok(())
}
