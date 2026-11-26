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
use siphon_cuda::{
    Gpu,
    ops::linear::{Gemm, Gemv, Linear, Start},
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
    reference_terms(weight, x, k, k)
}

/// As [`reference`], with `gamma` for a sum of `terms` F32 additions.
fn reference_terms(weight: &[bf16], x: &[f64], k: usize, terms: usize) -> Vec<(f64, f64)> {
    let u = f64::from(f32::EPSILON) / 2.0;
    let ku = terms as f64 * u;
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
        linear.gemv_bf16(stream, &mut products, Start::Serial)?;
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
        linear.gemv_f32(stream, &mut products, Start::Serial)?;
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
    // An early start runs the same kernel with programmatic dependent launch.
    linear.gemv_bf16(stream, &mut products, Start::Early)?;
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
    linear.gemv_bf16(
        stream,
        &mut [Gemv::new(&weight_dev, &x_dev, &mut out)],
        Start::Serial,
    )?;
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
    linear.gemv_f32(
        stream,
        &mut [Gemv::new(&weight_dev, &x_dev, &mut out)],
        Start::Serial,
    )?;
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
struct TimedCase<'a> {
    label: &'a str,
    shapes: &'a [(usize, usize)],
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
            linear.gemv_f32(stream, &mut products, Start::Serial)?;
        } else {
            let mut products: Vec<_> = w
                .iter()
                .zip(&mut outs_bf16)
                .map(|(w, out)| Gemv::new(w, &x_bf16, out))
                .collect();
            linear.gemv_bf16(stream, &mut products, Start::Serial)?;
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

/// Inputs `[m, k]` in a typical normalized-activation range, from the weight hash.
fn hash_input(seed: u32, m: usize, k: usize) -> Vec<bf16> {
    hash_weight(seed, m, k)
        .iter()
        .map(|v| bf16::from_f32(v.to_f32() * 64.0))
        .collect()
}

/// F32 inputs `[m, k]` with all 24 significand bits in use, so the GEMM's three-way BF16
/// split of each value matters.
fn hash_input_f32(seed: u32, m: usize, k: usize) -> Vec<f32> {
    let coarse = hash_input(seed, m, k);
    let fine = hash_weight(seed + 1000, m, k);
    coarse
        .iter()
        .zip(&fine)
        .map(|(c, f)| c.to_f32() * (1.0 + f.to_f32()) + f.to_f32())
        .collect()
}

/// Checks `got`, `[m, n]`, against `x @ weight^T`, row by row. Returns the largest error
/// as a fraction of its bound.
fn check_gemm(
    label: &str,
    got: &[f64],
    weight: &[bf16],
    x: &[f64],
    (m, n, k): (usize, usize, usize),
    terms: usize,
    format: Format,
) -> f64 {
    assert_eq!(got.len(), m * n, "{label}: output length");
    (0..m)
        .map(|r| {
            let reference = reference_terms(weight, &x[r * k..(r + 1) * k], k, terms);
            check(
                &format!("{label} row {r}"),
                &got[r * n..(r + 1) * n],
                &reference,
                None,
                format,
            )
        })
        .fold(0.0, f64::max)
}

/// Batches of products as `(m, k, [n per product])`. Together they reach every tile size,
/// m = 1, odd m and n, and m, n and k off every tile's multiple.
const GEMM_CASES: &[(usize, usize, &[usize])] = &[
    (1, 2304, &[200, 24, 64]),
    (1, 136, &[3]),
    (5, 520, &[130, 7]),
    (16, 64, &[64]),
    (17, 264, &[129, 2, 300]),
    (31, 2304, &[100]),
    (33, 3072, &[257, 64]),
    (64, 72, &[128, 1, 9, 16, 33, 65, 127, 200]),
    (100, 4608, &[64]),
    (129, 2304, &[384, 24]),
    (300, 520, &[136, 255]),
    (1024, 264, &[136]),
    // Enough tiles to fill the GPU without splitting k, and to pick the 64 x 128 and 128 x 128 tiles.
    (3, 64, &[24_000]),
    (512, 72, &[3072]),
    (1024, 72, &[2817]),
    (2048, 64, &[2305]),
];

/// Every case of [`GEMM_CASES`] through `gemm_bf16`, against the F64 reference.
#[test]
fn gemm_bf16_matches_reference() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    let stream = gpu.stream();
    for (case, &(m, k, ns)) in GEMM_CASES.iter().enumerate() {
        let seed = 200 + 16 * case as u32;
        let x = hash_input(seed, m, k);
        let weights: Vec<Vec<bf16>> = ns
            .iter()
            .zip(seed + 1..)
            .map(|(&n, s)| hash_weight(s, n, k))
            .collect();
        let x_dev = stream.clone_htod(&x)?;
        let weights_dev = weights
            .iter()
            .map(|w| stream.clone_htod(w))
            .collect::<Result<Vec<_>, _>>()?;
        let mut outs = ns
            .iter()
            .map(|&n| stream.alloc_zeros::<bf16>(m * n))
            .collect::<Result<Vec<_>, _>>()?;
        let mut products: Vec<_> = weights_dev
            .iter()
            .zip(&mut outs)
            .map(|(w, out)| Gemm::new(w, &x_dev, out))
            .collect();
        linear.gemm_bf16(stream, &mut products, m)?;
        drop(products);
        let x = widen(&x);
        for ((out, w), &n) in outs.iter().zip(&weights).zip(ns) {
            let got = widen(&stream.clone_dtoh(out)?);
            let label = format!("m {m}, n {n}, k {k}");
            let worst = check_gemm(&label, &got, w, &x, (m, n, k), k, Format::Bf16);
            eprintln!("bf16 {label}: worst error {worst:.3} of the bound");
        }
    }
    Ok(())
}

/// Every case of [`GEMM_CASES`] through `gemm_f32`. Each input runs as three BF16 parts
/// that miss at most 2^-24 of its value, and the three products are summed separately, so
/// the bound is that of 3k + 1 additions.
#[test]
fn gemm_f32_matches_reference() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    let stream = gpu.stream();
    for (case, &(m, k, ns)) in GEMM_CASES.iter().enumerate() {
        let seed = 400 + 16 * case as u32;
        let x = hash_input_f32(seed, m, k);
        let weights: Vec<Vec<bf16>> = ns
            .iter()
            .zip(seed + 1..)
            .map(|(&n, s)| hash_weight(s, n, k))
            .collect();
        let x_dev = stream.clone_htod(&x)?;
        let weights_dev = weights
            .iter()
            .map(|w| stream.clone_htod(w))
            .collect::<Result<Vec<_>, _>>()?;
        let mut outs = ns
            .iter()
            .map(|&n| stream.alloc_zeros::<f32>(m * n))
            .collect::<Result<Vec<_>, _>>()?;
        let mut products: Vec<_> = weights_dev
            .iter()
            .zip(&mut outs)
            .map(|(w, out)| Gemm::new(w, &x_dev, out))
            .collect();
        linear.gemm_f32(stream, &mut products, m)?;
        drop(products);
        let x: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        for ((out, w), &n) in outs.iter().zip(&weights).zip(ns) {
            let got: Vec<f64> = stream.clone_dtoh(out)?.into_iter().map(f64::from).collect();
            let label = format!("m {m}, n {n}, k {k}");
            let worst = check_gemm(&label, &got, w, &x, (m, n, k), 3 * k + 1, Format::F32);
            eprintln!("f32 {label}: worst error {worst:.3} of the bound");
        }
    }
    Ok(())
}

/// A product can read rows of a larger buffer, as the runner's views do.
#[test]
fn gemm_on_views() -> anyhow::Result<()> {
    const M: usize = 40;
    const N: usize = 96;
    const K: usize = 2304;
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    let stream = gpu.stream();
    let weight = hash_weight(60, 2 * N, K);
    let x = hash_input(61, 2 * M, K);
    let weight_dev = stream.clone_htod(&weight)?;
    let x_dev = stream.clone_htod(&x)?;
    let mut out = stream.alloc_zeros::<bf16>(2 * M * N)?;
    {
        let w_view = weight_dev.slice(N * K..);
        let x_view = x_dev.slice(M * K..);
        let mut out_view = out.slice_mut(M * N..);
        linear.gemm_bf16(stream, &mut [Gemm::new(&w_view, &x_view, &mut out_view)], M)?;
    }
    let got = stream.clone_dtoh(&out)?;
    assert!(
        got[..M * N].iter().all(|v| v.to_f32() == 0.0),
        "wrote outside the view"
    );
    let worst = check_gemm(
        "views",
        &widen(&got[M * N..]),
        &weight[N * K..],
        &widen(&x[M * K..]),
        (M, N, K),
        K,
        Format::Bf16,
    );
    eprintln!("views: worst error {worst:.3} of the bound");
    Ok(())
}

/// Directory of the original checkpoint: `SIPHON_MODEL`, or the GPU host's usual location.
fn model_dir() -> anyhow::Result<std::path::PathBuf> {
    use anyhow::Context as _;
    if let Some(dir) = std::env::var_os("SIPHON_MODEL") {
        return Ok(dir.into());
    }
    let home = std::env::var_os("HOME").context("HOME is not set")?;
    Ok(std::path::PathBuf::from(home).join(".local/share/inkling-mini/model"))
}

/// A BF16 tensor of the checkpoint and its `[n, k]` shape.
fn checkpoint_matrix(
    checkpoint: &siphon::Checkpoint,
    name: &str,
) -> anyhow::Result<(Vec<bf16>, usize, usize)> {
    use anyhow::Context as _;
    let entry = checkpoint
        .get(name)
        .with_context(|| format!("{name} missing from checkpoint"))?;
    let &[n, k] = entry.shape.as_slice() else {
        anyhow::bail!("{name} has shape {:?}", entry.shape);
    };
    let values: Vec<bf16> = checkpoint
        .bytes(entry)
        .as_chunks()
        .0
        .iter()
        .map(|b| bf16::from_le_bytes(*b))
        .collect();
    anyhow::ensure!(values.len() == n * k, "{name} is not BF16");
    Ok((values, n, k))
}

/// Real projections of a global layer, its output projection and the LM head, each as one
/// batch, against the F64 reference.
#[test]
#[ignore = "reads the original checkpoint from SIPHON_MODEL or ~/.local/share/inkling-mini/model"]
fn gemm_checkpoint_weights() -> anyhow::Result<()> {
    let checkpoint = siphon::Checkpoint::open(model_dir()?)?;
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    let stream = gpu.stream();
    let a = "model.llm.layers.11.attn";
    let batches: [(usize, Vec<String>); 4] = [
        (
            37,
            [
                "wq_du",
                "wr_du",
                "gate_attn",
                "attn_sink_proj",
                "wk_dv",
                "wv_dv",
                "k_sconv.weight_proj",
                "v_sconv.weight_proj",
            ]
            .iter()
            .map(|w| {
                if w.ends_with("proj") && w.contains("sconv") {
                    format!("{a}.{w}")
                } else {
                    format!("{a}.{w}.weight")
                }
            })
            .collect(),
        ),
        (130, vec![format!("{a}.wo_ud.weight")]),
        (
            19,
            vec!["model.llm.layers.11.controllers.attn.down_weight".to_owned()],
        ),
        (3, vec!["model.llm.unembed.weight".to_owned()]),
    ];
    for (seed, (m, names)) in (500..).zip(&batches) {
        let m = *m;
        let mats = names
            .iter()
            .map(|name| checkpoint_matrix(&checkpoint, name))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let k = mats[0].2;
        let x = hash_input(seed, m, k);
        let x_dev = stream.clone_htod(&x)?;
        let weights_dev = mats
            .iter()
            .map(|(w, ..)| stream.clone_htod(w))
            .collect::<Result<Vec<_>, _>>()?;
        let mut outs = mats
            .iter()
            .map(|&(_, n, _)| stream.alloc_zeros::<bf16>(m * n))
            .collect::<Result<Vec<_>, _>>()?;
        let mut products: Vec<_> = weights_dev
            .iter()
            .zip(&mut outs)
            .map(|(w, out)| Gemm::new(w, &x_dev, out))
            .collect();
        linear.gemm_bf16(stream, &mut products, m)?;
        drop(products);
        let x = widen(&x);
        for ((out, (w, n, _)), name) in outs.iter().zip(&mats).zip(names) {
            let got = widen(&stream.clone_dtoh(out)?);
            let worst = check_gemm(name, &got, w, &x, (m, *n, k), k, Format::Bf16);
            eprintln!("{name} ({n} x {k}, m {m}): worst error {worst:.3} of the bound");
        }
    }
    Ok(())
}

/// Row counts of the timed GEMMs.
const GEMM_TIMED_M: [usize; 8] = [1, 16, 64, 128, 256, 512, 1024, 2048];

/// Timed GEMM batches: products as `n`, sharing `k`.
const GEMM_TIMED: &[(&str, usize, &[usize])] = &[
    ("attention inputs", 2304, &[3072, 384, 3072, 24, 1024, 1024]),
    ("o", 3072, &[2304]),
    ("gate down", 4608, &[64]),
    ("lm head", 2304, &[201_024]),
];

/// Times `gemm_bf16` on the runner's prefill shapes, from DRAM, as replayed in a CUDA
/// graph, and the batched GEMV at m = 1 for comparison. Run with
/// `cargo test --release --test linear -- --ignored --nocapture gemm_throughput`.
#[test]
#[ignore = "timing, not a check"]
fn gemm_throughput() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let linear = Linear::load(gpu.context())?;
    eprintln!(
        "{:18} {:>5} {:>10} {:>9} {:>8}",
        "batch", "m", "us", "TFLOP/s", "TB/s"
    );
    for &(label, k, ns) in GEMM_TIMED {
        let shapes: Vec<(usize, usize)> = ns.iter().map(|&n| (n, k)).collect();
        let n_sum: usize = ns.iter().sum();
        for m in GEMM_TIMED_M {
            if label == "lm head" && m > 512 {
                continue;
            }
            let us = time_gemm(&gpu, &linear, &shapes, m)?;
            let flop = 2.0 * (m * n_sum * k) as f64;
            let bytes = 2 * (n_sum * k + m * k + m * n_sum);
            eprintln!(
                "{label:18} {m:5} {us:10.2} {:9.1} {:8.3}",
                flop / us / 1e6,
                bytes as f64 / us / 1e6
            );
        }
        if label == "gate down" {
            // The F32 projections of the message gates, next to the gate's down projection.
            let f32_shapes = &[(2304, 64), (2304, 64)];
            for m in GEMM_TIMED_M {
                let us = time_gemm_as::<f32>(&gpu, &linear, f32_shapes, m)?;
                let flop = 2.0 * (m * 2 * 2304 * 64) as f64;
                let bytes = 2 * 2304 * 64 * 2 + 4 * (m * 64 + 2 * m * 2304);
                eprintln!(
                    "{:18} {m:5} {us:10.2} {:9.1} {:8.3}",
                    "f32 forget, gain",
                    flop / us / 1e6,
                    bytes as f64 / us / 1e6
                );
            }
        }
        let us = time_case(
            &gpu,
            &linear,
            &TimedCase {
                label,
                shapes: &shapes,
                f32_input: false,
            },
        )?;
        let bytes = 2 * n_sum * k;
        eprintln!(
            "{label:18} {:>5} {us:10.2} {:>9} {:8.3}",
            "gemv",
            "",
            bytes as f64 / us / 1e6
        );
    }
    Ok(())
}

/// Microseconds per `gemm_bf16` launch over `m` rows, averaged over graph replays.
fn time_gemm(
    gpu: &Gpu,
    linear: &Linear,
    shapes: &[(usize, usize)],
    m: usize,
) -> anyhow::Result<f64> {
    time_gemm_as::<bf16>(gpu, linear, shapes, m)
}

/// Input and output types of a timed GEMM.
trait GemmInput: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy {
    fn from_bf16(v: bf16) -> Self;
    fn run(
        linear: &Linear,
        stream: &cudarc::driver::CudaStream,
        products: &mut [Gemm<'_, Self>],
        m: usize,
    ) -> anyhow::Result<()>;
}

impl GemmInput for bf16 {
    fn from_bf16(v: bf16) -> Self {
        v
    }
    fn run(
        linear: &Linear,
        stream: &cudarc::driver::CudaStream,
        products: &mut [Gemm<'_, Self>],
        m: usize,
    ) -> anyhow::Result<()> {
        linear.gemm_bf16(stream, products, m)
    }
}

impl GemmInput for f32 {
    fn from_bf16(v: bf16) -> Self {
        v.to_f32()
    }
    fn run(
        linear: &Linear,
        stream: &cudarc::driver::CudaStream,
        products: &mut [Gemm<'_, Self>],
        m: usize,
    ) -> anyhow::Result<()> {
        linear.gemm_f32(stream, products, m)
    }
}

/// As [`time_gemm`], with inputs and outputs of type `T`.
fn time_gemm_as<T: GemmInput>(
    gpu: &Gpu,
    linear: &Linear,
    shapes: &[(usize, usize)],
    m: usize,
) -> anyhow::Result<f64> {
    use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};

    /// Weight copies cycle through at least this many bytes, so no launch hits L2.
    const ROTATE_BYTES: usize = 512 << 20;
    const REPLAYS: usize = 5;

    let stream = gpu.stream();
    let k = shapes[0].1;
    let bytes: usize = shapes.iter().map(|(n, k)| n * k * 2).sum();
    let flop: usize = shapes.iter().map(|(n, k)| 2 * m * n * k).sum();
    // About 20 ms of work per replay at 200 TFLOP/s or 1.7 TB/s.
    let estimate_us = (flop / 200_000_000).max(bytes / 1_700_000).max(1);
    let launches = (20_000 / estimate_us).clamp(2, 40);
    let copies = ROTATE_BYTES.div_ceil(bytes).clamp(1, launches);
    let weights = (0..copies)
        .map(|c| {
            shapes
                .iter()
                .zip(0u32..)
                .map(|(&(n, k), i)| stream.clone_htod(&hash_weight(c as u32 * 16 + i, n, k)))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let x: Vec<T> = hash_input(7, m, k).into_iter().map(T::from_bf16).collect();
    let x = stream.clone_htod(&x)?;
    let mut outs = shapes
        .iter()
        .map(|&(n, _)| stream.alloc_zeros::<T>(m * n))
        .collect::<Result<Vec<_>, _>>()?;

    stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    for i in 0..launches {
        let mut products: Vec<_> = weights[i % copies]
            .iter()
            .zip(&mut outs)
            .map(|(w, out)| Gemm::new(w, &x, out))
            .collect();
        T::run(linear, stream, &mut products, m)?;
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
    Ok(f64::from(start.elapsed_ms(&end)?) * 1e3 / (REPLAYS * launches) as f64)
}
