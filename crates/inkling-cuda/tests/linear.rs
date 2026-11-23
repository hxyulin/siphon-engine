//! Matrix-vector kernels against an F64 reference and the MLX outputs from
//! `reference/fixtures/linear.py`.
//!
//! The kernels sum in a different order from MLX, so the tests bound the error instead of
//! requiring equal bits. Any order of k F32 multiply-adds computes a dot product within
//! `gamma * sum |w_i x_i|` of the exact value, with `gamma = k u / (1 - k u)` and
//! `u = 2^-24` (Higham, Accuracy and Stability of Numerical Algorithms, eq. 3.5). The
//! output then rounds to its format, adding at most half an ulp. Each output must lie
//! within that bound of the F64 reference. MLX's output carries the same kind of error, so
//! the distance to MLX may be up to twice the sum bound plus two half-ulps.
//!
//! The weights come from an integer hash that the fixture script and this file both
//! implement, so full model shapes need no stored weights.

#![cfg(target_os = "linux")]

mod common;

use common::{Fixture, bf16_ulps};
use half::bf16;
use inkling_cuda::{
    Gpu,
    ops::linear::{Gemv, Linear},
};

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

/// Output format of a product: how far rounding to it can move a value.
#[derive(Clone, Copy)]
enum Format {
    Bf16,
    F32,
}

impl Format {
    /// Half an ulp at magnitude `v`.
    fn half_ulp(self, v: f64) -> f64 {
        let mantissa_bits = match self {
            Self::Bf16 => 7,
            Self::F32 => 23,
        };
        let v = v.abs().max(f64::from(f32::MIN_POSITIVE));
        (v.log2().floor() - f64::from(mantissa_bits) - 1.0).exp2()
    }
}

/// The F64 product `weight @ x` and the bound `gamma * sum |w x|` for each row.
fn reference(weight: &[bf16], x: &[f64], k: usize) -> Vec<(f64, f64)> {
    let u = f64::from(f32::EPSILON) / 2.0;
    let ku = k as f64 * u;
    let gamma = ku / (1.0 - ku);
    weight
        .chunks_exact(k)
        .map(|row| {
            let (sum, abs) = row.iter().zip(x).fold((0.0, 0.0), |(s, a), (w, x)| {
                let p = w.to_f64() * x;
                (s + p, a + p.abs())
            });
            (sum, gamma * abs)
        })
        .collect()
}

/// Checks `got` against the reference, and `mlx` (if given) against `got`. Returns the
/// largest error as a fraction of its bound.
fn check(
    label: &str,
    got: &[f64],
    reference: &[(f64, f64)],
    mlx: Option<&[f64]>,
    format: Format,
) -> f64 {
    assert_eq!(got.len(), reference.len(), "{label}: length");
    let mut worst = 0.0f64;
    for (i, (&g, &(want, sum_bound))) in got.iter().zip(reference).enumerate() {
        let half_ulp = format.half_ulp(g.abs().max(want.abs()));
        let bound = sum_bound + half_ulp;
        let err = (g - want).abs();
        assert!(
            err <= bound,
            "{label}[{i}]: got {g}, reference {want}, error {err:e} > bound {bound:e}"
        );
        worst = worst.max(err / bound);
        if let Some(mlx) = mlx {
            let m = mlx[i];
            let bound = 2.0 * sum_bound + half_ulp + format.half_ulp(m.abs().max(want.abs()));
            let err = (g - m).abs();
            assert!(
                err <= bound,
                "{label}[{i}]: got {g}, MLX {m}, distance {err:e} > bound {bound:e}"
            );
        }
    }
    worst
}

fn widen(values: &[bf16]) -> Vec<f64> {
    values.iter().map(|v| v.to_f64()).collect()
}

#[test]
fn gemv_bf16_matches_reference_and_mlx() -> anyhow::Result<()> {
    let fixture = Fixture::load("linear")?;
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    let stream = gpu.stream();
    for name in BF16_CASES {
        let (x, x_shape) = fixture.bf16(&format!("{name}.x"))?;
        let (want, out_shape) = fixture.bf16(&format!("{name}.out"))?;
        let (n, k, weight) = case_weight(&fixture, name, &x_shape, &out_shape)?;
        let rows = x_shape[0];
        let weight_dev = stream.clone_htod(&weight)?;
        // Each input row is its own product in one batch.
        let xs = x
            .chunks_exact(k)
            .map(|row| stream.clone_htod(row))
            .collect::<Result<Vec<_>, _>>()?;
        let mut outs = (0..rows)
            .map(|_| stream.alloc_zeros::<bf16>(n))
            .collect::<Result<Vec<_>, _>>()?;
        let mut products: Vec<_> = xs
            .iter()
            .zip(&mut outs)
            .map(|(x, out)| Gemv::new(&weight_dev, x, out))
            .collect();
        linear.gemv_bf16(stream, &mut products)?;
        drop(products);
        let mut worst = 0.0f64;
        let mut differ = 0;
        for (r, out) in outs.iter().enumerate() {
            let got = stream.clone_dtoh(out)?;
            let mlx = &want[r * n..(r + 1) * n];
            differ += got
                .iter()
                .zip(mlx)
                .filter(|(g, m)| bf16_ulps(**g, **m) > 0)
                .count();
            let reference = reference(&weight, &widen(&x[r * k..(r + 1) * k]), k);
            worst = worst.max(check(
                &format!("{name} row {r}"),
                &widen(&got),
                &reference,
                Some(&widen(mlx)),
                Format::Bf16,
            ));
        }
        eprintln!(
            "{name} ({n} x {k}, {rows} rows): worst error {:.3} of the bound, {differ} of {} differ from MLX",
            worst,
            rows * n
        );
    }
    Ok(())
}

#[test]
fn gemv_f32_matches_reference_and_mlx() -> anyhow::Result<()> {
    let fixture = Fixture::load("linear")?;
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    let stream = gpu.stream();
    for name in F32_CASES {
        let (x, x_shape) = fixture.f32(&format!("{name}.x"))?;
        let (want, out_shape) = fixture.f32(&format!("{name}.out"))?;
        let (n, k, weight) = case_weight(&fixture, name, &x_shape, &out_shape)?;
        let rows = x_shape[0];
        let weight_dev = stream.clone_htod(&weight)?;
        let xs = x
            .chunks_exact(k)
            .map(|row| stream.clone_htod(row))
            .collect::<Result<Vec<_>, _>>()?;
        let mut outs = (0..rows)
            .map(|_| stream.alloc_zeros::<f32>(n))
            .collect::<Result<Vec<_>, _>>()?;
        let mut products: Vec<_> = xs
            .iter()
            .zip(&mut outs)
            .map(|(x, out)| Gemv::new(&weight_dev, x, out))
            .collect();
        linear.gemv_f32(stream, &mut products)?;
        drop(products);
        let mut worst = 0.0f64;
        for (r, out) in outs.iter().enumerate() {
            let got: Vec<f64> = stream.clone_dtoh(out)?.into_iter().map(f64::from).collect();
            let mlx: Vec<f64> = want[r * n..(r + 1) * n]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            let xr: Vec<f64> = x[r * k..(r + 1) * k]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            worst = worst.max(check(
                &format!("{name} row {r}"),
                &got,
                &reference(&weight, &xr, k),
                Some(&mlx),
                Format::F32,
            ));
        }
        eprintln!("{name} ({n} x {k}, {rows} rows): worst error {worst:.3} of the bound");
    }
    Ok(())
}

/// The runner's batch for one global-layer visit: eight products with three inputs and
/// row counts that leave partial blocks.
#[test]
fn gemv_batch_of_eight() -> anyhow::Result<()> {
    const K: usize = 2304;
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    let stream = gpu.stream();
    // (rows, input) for q, r, gate, sink, k, k_proj, v, v_proj.
    let shapes = [
        (3072, 0),
        (384, 0),
        (3072, 0),
        (24, 0),
        (1024, 1),
        (32, 1),
        (1024, 2),
        (32, 2),
    ];
    let inputs: Vec<Vec<bf16>> = (0..3)
        .map(|i| {
            // Hash values scaled into a typical normalized-activation range.
            hash_weight(100 + i, 1, K)
                .iter()
                .map(|v| bf16::from_f32(v.to_f32() * 64.0))
                .collect()
        })
        .collect();
    let weights: Vec<Vec<bf16>> = shapes
        .iter()
        .zip(20..)
        .map(|(&(n, _), seed)| hash_weight(seed, n, K))
        .collect();
    let inputs_dev = inputs
        .iter()
        .map(|x| stream.clone_htod(x))
        .collect::<Result<Vec<_>, _>>()?;
    let weights_dev = weights
        .iter()
        .map(|w| stream.clone_htod(w))
        .collect::<Result<Vec<_>, _>>()?;
    let mut outs = shapes
        .iter()
        .map(|&(n, _)| stream.alloc_zeros::<bf16>(n))
        .collect::<Result<Vec<_>, _>>()?;
    let mut products: Vec<_> = weights_dev
        .iter()
        .zip(&mut outs)
        .zip(&shapes)
        .map(|((w, out), &(_, input))| Gemv::new(w, &inputs_dev[input], out))
        .collect();
    linear.gemv_bf16(stream, &mut products)?;
    drop(products);
    for (i, (out, &(n, input))) in outs.iter().zip(&shapes).enumerate() {
        let got = widen(&stream.clone_dtoh(out)?);
        let worst = check(
            &format!("product {i}"),
            &got,
            &reference(&weights[i], &widen(&inputs[input]), K),
            None,
            Format::Bf16,
        );
        eprintln!("product {i} ({n} x {K}): worst error {worst:.3} of the bound");
    }
    Ok(())
}

/// The entry points for a `k` without an exact fit, in both input types.
#[test]
fn gemv_generic_k() -> anyhow::Result<()> {
    const N: usize = 2048;
    const K: usize = 320;
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    let stream = gpu.stream();
    let weight = hash_weight(40, N, K);
    let weight_dev = stream.clone_htod(&weight)?;
    let x: Vec<bf16> = hash_weight(41, 1, K)
        .iter()
        .map(|v| bf16::from_f32(v.to_f32() * 64.0))
        .collect();
    let reference = reference(&weight, &widen(&x), K);

    let x_dev = stream.clone_htod(&x)?;
    let mut out = stream.alloc_zeros::<bf16>(N)?;
    linear.gemv_bf16(stream, &mut [Gemv::new(&weight_dev, &x_dev, &mut out)])?;
    let worst = check(
        "bf16",
        &widen(&stream.clone_dtoh(&out)?),
        &reference,
        None,
        Format::Bf16,
    );
    eprintln!("bf16 ({N} x {K}): worst error {worst:.3} of the bound");

    let x_f32: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let x_dev = stream.clone_htod(&x_f32)?;
    let mut out = stream.alloc_zeros::<f32>(N)?;
    linear.gemv_f32(stream, &mut [Gemv::new(&weight_dev, &x_dev, &mut out)])?;
    let got: Vec<f64> = stream
        .clone_dtoh(&out)?
        .into_iter()
        .map(f64::from)
        .collect();
    let worst = check("f32", &got, &reference, None, Format::F32);
    eprintln!("f32 ({N} x {K}): worst error {worst:.3} of the bound");
    Ok(())
}

/// One timed batch: products as `(rows, k)`, with F32 or BF16 inputs.
struct TimedCase {
    label: &'static str,
    shapes: &'static [(usize, usize)],
    f32_input: bool,
}

const ATTENTION_INPUTS: &[(usize, usize)] = &[
    (3072, 2304),
    (384, 2304),
    (3072, 2304),
    (24, 2304),
    (1024, 2304),
    (1024, 2304),
];
const ATTENTION_INPUTS_GLOBAL: &[(usize, usize)] = &[
    (3072, 2304),
    (384, 2304),
    (3072, 2304),
    (24, 2304),
    (1024, 2304),
    (32, 2304),
    (1024, 2304),
    (32, 2304),
];

/// Times the runner's products from DRAM, as replayed in a CUDA graph. Run with
/// `cargo test --release --test linear -- --ignored --nocapture gemv_bandwidth`.
#[test]
#[ignore = "timing, not a check"]
fn gemv_bandwidth() -> anyhow::Result<()> {
    let case = |label, shapes, f32_input| TimedCase {
        label,
        shapes,
        f32_input,
    };
    let cases = [
        case("attention inputs, sliding", ATTENTION_INPUTS, false),
        case("attention inputs, global", ATTENTION_INPUTS_GLOBAL, false),
        case("o", &[(2304, 3072)], false),
        case("gate down", &[(64, 4608)], false),
        case("gate forget and gain", &[(2304, 64), (2304, 64)], true),
        case("q alone", &[(3072, 2304)], false),
        case("k alone", &[(1024, 2304)], false),
        case("lm head", &[(201_024, 2304)], false),
    ];
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    for case in &cases {
        let us = time_case(&gpu, &linear, case)?;
        let bytes: usize = case.shapes.iter().map(|(n, k)| n * k * 2).sum();
        eprintln!(
            "{:28} {:7.2} MB  {us:8.2} us  {:6.3} TB/s",
            case.label,
            bytes as f64 / 1e6,
            bytes as f64 / us / 1e6
        );
    }
    Ok(())
}

/// Microseconds per launch of `case`, averaged over graph replays.
fn time_case(gpu: &Gpu, linear: &Linear, case: &TimedCase) -> anyhow::Result<f64> {
    use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};

    /// Weight copies cycle through at least this many bytes, so no launch hits L2.
    const ROTATE_BYTES: usize = 512 << 20;
    const LAUNCHES: usize = 40;
    const REPLAYS: usize = 10;

    let stream = gpu.stream();
    let shapes = case.shapes;
    let k = shapes[0].1;
    let bytes: usize = shapes.iter().map(|(n, k)| n * k * 2).sum();
    let copies = ROTATE_BYTES.div_ceil(bytes).clamp(1, LAUNCHES);
    let weights = (0..copies)
        .map(|_| {
            shapes
                .iter()
                .map(|&(n, k)| stream.alloc_zeros::<bf16>(n * k))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let x_bf16 = stream.alloc_zeros::<bf16>(k)?;
    let x_f32 = stream.alloc_zeros::<f32>(k)?;
    let mut outs_bf16 = shapes
        .iter()
        .map(|&(n, _)| stream.alloc_zeros::<bf16>(n))
        .collect::<Result<Vec<_>, _>>()?;
    let mut outs_f32 = shapes
        .iter()
        .map(|&(n, _)| stream.alloc_zeros::<f32>(n))
        .collect::<Result<Vec<_>, _>>()?;

    stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    for i in 0..LAUNCHES {
        let w = &weights[i % copies];
        if case.f32_input {
            let mut products: Vec<_> = w
                .iter()
                .zip(&mut outs_f32)
                .map(|(w, out)| Gemv::new(w, &x_f32, out))
                .collect();
            linear.gemv_f32(stream, &mut products)?;
        } else {
            let mut products: Vec<_> = w
                .iter()
                .zip(&mut outs_bf16)
                .map(|(w, out)| Gemv::new(w, &x_bf16, out))
                .collect();
            linear.gemv_bf16(stream, &mut products)?;
        }
    }
    let graph = stream
        .end_capture(CUgraphInstantiate_flags(0))?
        .ok_or_else(|| anyhow::anyhow!("empty capture"))?;
    graph.launch()?;
    let start = gpu
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = gpu
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(stream)?;
    for _ in 0..REPLAYS {
        graph.launch()?;
    }
    end.record(stream)?;
    Ok(f64::from(start.elapsed_ms(&end)?) * 1e3 / (REPLAYS * LAUNCHES) as f64)
}
