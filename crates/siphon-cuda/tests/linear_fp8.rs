//! The FP8 quantizer and matrix-vector kernels.
//!
//! The quantizer must give exactly the E4M3 bytes and scales a CPU model computes. The
//! products are checked against an F64 reference over the quantized weights: any order of
//! k F32 multiply-adds is within `gamma * sum |q_i x_i|` of the exact dot product, with
//! `gamma = k u / (1 - k u)` and `u = 2^-24` (Higham, eq. 3.5). The scale multiplies the
//! F32 sum once, adding half an F32 ulp, and the output rounds to BF16, adding half a
//! BF16 ulp. The tests also print how far the FP8 products are from the BF16 ones.

#![cfg(target_os = "linux")]

use std::sync::OnceLock;

use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::{
        linear::{Gemv, Linear, Start},
        linear_fp8::{Fp8Gemv, Fp8Matrix, LinearFp8},
    },
};

/// Pseudo-random BF16 values from an integer hash, spread over `2^-10` to `2^0` with
/// either sign, plus a few large and tiny values per row to exercise saturation-free
/// scaling and E4M3 subnormals.
fn hash_values(seed: u32, n: usize) -> Vec<bf16> {
    let salt = seed.wrapping_mul(0x9E37_79B9);
    (0..n)
        .map(|i| {
            let mut x = (i as u32) ^ salt;
            x ^= x >> 16;
            x = x.wrapping_mul(0x7FEB_352D);
            x ^= x >> 15;
            x = x.wrapping_mul(0x846C_A68B);
            x ^= x >> 16;
            let sign = (x >> 31) << 15;
            let exponent = (117 + ((x >> 27) & 15)) << 7;
            let mantissa = (x >> 16) & 0x7F;
            bf16::from_bits((sign | exponent | mantissa) as u16)
        })
        .collect()
}

fn e4m3_value(bits: u8) -> f32 {
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

/// `x` rounded to E4M3: nearest, ties to the even code, saturated to +-448. Codes 0..=126
/// are the non-negative finite values in increasing order.
fn e4m3_from_f32(x: f32) -> u8 {
    static VALUES: OnceLock<Vec<f32>> = OnceLock::new();
    let values = VALUES.get_or_init(|| (0..=126).map(e4m3_value).collect());
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs();
    let below = values.partition_point(|&v| v <= a) - 1;
    let code = if below == 126 {
        126
    } else {
        let (low, high) = (values[below], values[below + 1]);
        match (a - low).total_cmp(&(high - a)) {
            std::cmp::Ordering::Less => below,
            std::cmp::Ordering::Greater => below + 1,
            std::cmp::Ordering::Equal => below + (below & 1),
        }
    };
    sign | u8::try_from(code).expect("codes fit in a byte")
}

/// The CPU model of the quantizer: E4M3 bytes and the scale of each row.
fn quantize_cpu(w: &[bf16], k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut bytes = Vec::with_capacity(w.len());
    let mut scales = Vec::new();
    for row in w.chunks_exact(k) {
        let amax = row.iter().map(|v| v.to_f32().abs()).fold(0.0, f32::max);
        let s = amax / 448.0;
        scales.push(s);
        bytes.extend(row.iter().map(|v| {
            if s > 0.0 {
                e4m3_from_f32(v.to_f32() / s)
            } else {
                0
            }
        }));
    }
    (bytes, scales)
}

/// The quantized matrix's E4M3 bytes and scales, read back.
fn read_back(gpu: &Gpu, m: &Fp8Matrix) -> anyhow::Result<(Vec<u8>, Vec<f32>)> {
    let stream = gpu.stream();
    let bytes = stream.clone_dtoh(&m.values())?;
    let scales = stream
        .clone_dtoh(&m.scales())?
        .as_chunks()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok((bytes, scales))
}

/// Rows of the test weights: some leave a partial block of rows.
const SHAPES: &[(usize, usize)] = &[(37, 2304), (24, 2304), (100, 3072), (1, 2304)];

#[test]
fn quantize_matches_cpu_model() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let fp8 = LinearFp8::load(gpu.context())?;
    let stream = gpu.stream();
    for (seed, &(rows, k)) in SHAPES.iter().enumerate() {
        let mut w = hash_values(seed as u32 + 1, rows * k);
        // An all-zero row, and a row whose small values fall into E4M3 subnormals.
        if rows > 2 {
            w[k..2 * k].fill(bf16::ZERO);
            w[2 * k] = bf16::from_f32(300.0);
        }
        let dev = stream.clone_htod(&w)?;
        let m = fp8.quantize(stream, &dev, rows, k)?;
        let (bytes, scales) = read_back(&gpu, &m)?;
        let (want_bytes, want_scales) = quantize_cpu(&w, k);
        assert_eq!(scales, want_scales, "scales of {rows} x {k}");
        let differ = bytes
            .iter()
            .zip(&want_bytes)
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(differ, 0, "{differ} E4M3 bytes of {rows} x {k} differ");
        let subnormal = bytes
            .iter()
            .filter(|&&b| b & 0x78 == 0 && b & 7 != 0)
            .count();
        eprintln!("{rows} x {k}: bytes and scales match, {subnormal} subnormal values");
    }
    Ok(())
}

/// The F64 product of the dequantized weights with `x`, and its error bound for each row.
fn reference(bytes: &[u8], scales: &[f32], x: &[bf16], k: usize) -> Vec<(f64, f64)> {
    let u = f64::from(f32::EPSILON) / 2.0;
    let ku = k as f64 * u;
    let gamma = ku / (1.0 - ku);
    bytes
        .chunks_exact(k)
        .zip(scales)
        .map(|(row, &s)| {
            let (sum, abs) = row.iter().zip(x).fold((0.0, 0.0), |(sum, abs), (&q, x)| {
                let p = f64::from(e4m3_value(q)) * x.to_f64();
                (sum + p, abs + p.abs())
            });
            let s = f64::from(s);
            let value = s * sum;
            // The F32 sum's error, then half an F32 ulp for the scale multiply.
            (value, s * gamma * abs + value.abs() * u)
        })
        .collect()
}

/// Half a BF16 ulp at magnitude `v`.
fn bf16_half_ulp(v: f64) -> f64 {
    let v = v.abs().max(f64::from(f32::MIN_POSITIVE));
    (v.log2().floor() - 8.0).exp2()
}

/// Checks each product's outputs `got` against the F64 reference over its quantized
/// weights. Returns the largest error as a fraction of its bound, and the summed squares
/// of the distance from the BF16 products `want` and of `want`.
fn check_batch(
    gpu: &Gpu,
    mats: &[Fp8Matrix],
    inputs: &[Vec<bf16>],
    got: &[Vec<bf16>],
    want: &[Vec<bf16>],
) -> anyhow::Result<(f64, f64, f64)> {
    let mut worst = 0.0f64;
    let (mut err2, mut ref2) = (0.0f64, 0.0f64);
    for (i, (m, x)) in mats.iter().zip(inputs).enumerate() {
        let (bytes, scales) = read_back(gpu, m)?;
        for (row, ((g, (r, bound)), b)) in got[i]
            .iter()
            .zip(reference(&bytes, &scales, x, m.k()))
            .zip(&want[i])
            .enumerate()
        {
            let err = (g.to_f64() - r).abs();
            let allowed = bound + bf16_half_ulp(r + bound.copysign(r));
            assert!(
                err <= allowed,
                "product {i} row {row}: got {g}, reference {r}, error {err:e} > {allowed:e}"
            );
            worst = worst.max(err / allowed);
            let d = g.to_f64() - b.to_f64();
            err2 += d * d;
            ref2 += b.to_f64() * b.to_f64();
        }
    }
    Ok((worst, err2, ref2))
}

#[test]
fn gemv_matches_reference() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let fp8 = LinearFp8::load(gpu.context())?;
    let linear = Linear::load(gpu.context())?;
    // The real attention input batch of a global layer, then one batch per k with ragged
    // row counts, each product with its own input.
    let batches: &[&[(usize, usize)]] = &[
        &[
            (3072, 2304),
            (384, 2304),
            (3072, 2304),
            (24, 2304),
            (1024, 2304),
            (1024, 2304),
            (32, 2304),
            (32, 2304),
        ],
        &[(37, 2304), (1, 2304), (24, 2304)],
        &[(2304, 3072)],
        &[(100, 3072), (7, 3072)],
    ];
    let mut seed = 100;
    for batch in batches {
        let mut weights = Vec::new();
        let mut inputs = Vec::new();
        for &(rows, k) in *batch {
            seed += 2;
            let w = hash_values(seed, rows * k);
            // Normalized-activation-like inputs: magnitudes up to about 4.
            let x: Vec<bf16> = hash_values(seed + 1, k)
                .iter()
                .map(|v| bf16::from_f32(v.to_f32() * 4.0))
                .collect();
            weights.push((w, rows, k));
            inputs.push(x);
        }
        let dev_w = weights
            .iter()
            .map(|(w, _, _)| stream.clone_htod(w))
            .collect::<Result<Vec<_>, _>>()?;
        let mats = dev_w
            .iter()
            .zip(&weights)
            .map(|(d, &(_, rows, k))| fp8.quantize(stream, d, rows, k))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let dev_x = inputs
            .iter()
            .map(|x| stream.clone_htod(x))
            .collect::<Result<Vec<_>, _>>()?;
        let mut outs = weights
            .iter()
            .map(|&(_, rows, _)| stream.alloc_zeros::<bf16>(rows))
            .collect::<Result<Vec<_>, _>>()?;
        let mut bf16_outs = weights
            .iter()
            .map(|&(_, rows, _)| stream.alloc_zeros::<bf16>(rows))
            .collect::<Result<Vec<_>, _>>()?;
        {
            let mut products: Vec<_> = mats
                .iter()
                .zip(&dev_x)
                .zip(&mut outs)
                .map(|((m, x), out)| Fp8Gemv::new(m, x, out))
                .collect();
            fp8.gemv(stream, &mut products, Start::Early)?;
            let mut products: Vec<_> = dev_w
                .iter()
                .zip(&dev_x)
                .zip(&mut bf16_outs)
                .map(|((w, x), out)| Gemv::new(w, x, out))
                .collect();
            linear.gemv_bf16(stream, &mut products, Start::Serial)?;
        }
        let got = outs
            .iter()
            .map(|o| stream.clone_dtoh(o))
            .collect::<Result<Vec<_>, _>>()?;
        let want = bf16_outs
            .iter()
            .map(|o| stream.clone_dtoh(o))
            .collect::<Result<Vec<_>, _>>()?;
        let (worst, err2, ref2) = check_batch(&gpu, &mats, &inputs, &got, &want)?;
        eprintln!(
            "batch {batch:?}: worst error {worst:.3} of the bound; RMS distance from the BF16 \
             product {:.2e} of its RMS",
            (err2 / ref2).sqrt()
        );
    }
    Ok(())
}

/// Times the FP8 products from DRAM next to the BF16 ones, as replayed in a CUDA graph.
/// Run with `cargo test --release --test linear_fp8 -- --ignored --nocapture`.
#[test]
#[ignore = "timing, not a check"]
fn gemv_fp8_bandwidth() -> anyhow::Result<()> {
    let sliding: &[(usize, usize)] = &[
        (3072, 2304),
        (384, 2304),
        (3072, 2304),
        (24, 2304),
        (1024, 2304),
        (1024, 2304),
    ];
    let cases: &[(&str, &[(usize, usize)])] = &[
        ("attention inputs, sliding", sliding),
        ("o", &[(2304, 3072)]),
        ("lm head", &[(201_024, 2304)]),
    ];
    let gpu = Gpu::new(0)?;
    let fp8 = LinearFp8::load(gpu.context())?;
    let linear = Linear::load(gpu.context())?;
    for (&(label, shapes), hot) in cases
        .iter()
        .flat_map(|c| [(c, false), (c, true)])
        .filter(|&(c, hot)| !hot || c.0 != "lm head")
    {
        let (fp8_us, bf16_us) = time_case(&gpu, &fp8, &linear, shapes, hot)?;
        let label = if hot {
            format!("{label}, L2")
        } else {
            label.to_owned()
        };
        let fp8_bytes: usize = shapes.iter().map(|(n, k)| n * k + 4 * n).sum();
        let bf16_bytes: usize = shapes.iter().map(|(n, k)| 2 * n * k).sum();
        eprintln!(
            "{label:28} fp8 {:7.2} MB {fp8_us:8.2} us {:6.3} TB/s | bf16 {:7.2} MB \
             {bf16_us:8.2} us {:6.3} TB/s",
            fp8_bytes as f64 / 1e6,
            fp8_bytes as f64 / fp8_us / 1e6,
            bf16_bytes as f64 / 1e6,
            bf16_bytes as f64 / bf16_us / 1e6,
        );
    }
    Ok(())
}

/// Microseconds per FP8 and per BF16 launch of `shapes`, averaged over graph replays of
/// launches that cycle through enough weight copies that none hits L2, or with `hot`
/// over one copy that stays in L2.
fn time_case(
    gpu: &Gpu,
    fp8: &LinearFp8,
    linear: &Linear,
    shapes: &[(usize, usize)],
    hot: bool,
) -> anyhow::Result<(f64, f64)> {
    use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};

    const ROTATE_BYTES: usize = 512 << 20;
    const LAUNCHES: usize = 40;
    const REPLAYS: usize = 10;

    let stream = gpu.stream();
    let k = shapes[0].1;
    let bf16_bytes: usize = shapes.iter().map(|(n, k)| 2 * n * k).sum();
    let rotate = if hot { 0 } else { ROTATE_BYTES };
    let bf16_copies = rotate.div_ceil(bf16_bytes).clamp(1, LAUNCHES);
    let fp8_copies = rotate.div_ceil(bf16_bytes / 2).clamp(1, LAUNCHES);
    let bf16_weights = (0..bf16_copies)
        .map(|_| {
            shapes
                .iter()
                .map(|&(n, k)| stream.alloc_zeros::<bf16>(n * k))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fp8_weights = (0..fp8_copies)
        .map(|c| {
            shapes
                .iter()
                .enumerate()
                .map(|(i, &(n, k))| fp8.quantize(stream, &bf16_weights[c % bf16_copies][i], n, k))
                .collect::<anyhow::Result<Vec<_>>>()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let x = stream.alloc_zeros::<bf16>(k)?;
    let mut outs = shapes
        .iter()
        .map(|&(n, _)| stream.alloc_zeros::<bf16>(n))
        .collect::<Result<Vec<_>, _>>()?;

    let mut time = |fp8_launch: bool| -> anyhow::Result<f64> {
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        for i in 0..LAUNCHES {
            if fp8_launch {
                let mut products: Vec<_> = fp8_weights[i % fp8_copies]
                    .iter()
                    .zip(&mut outs)
                    .map(|(w, out)| Fp8Gemv::new(w, &x, out))
                    .collect();
                fp8.gemv(stream, &mut products, Start::Early)?;
            } else {
                let mut products: Vec<_> = bf16_weights[i % bf16_copies]
                    .iter()
                    .zip(&mut outs)
                    .map(|(w, out)| Gemv::new(w, &x, out))
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
    };
    let fp8_us = time(true)?;
    let bf16_us = time(false)?;
    Ok((fp8_us, bf16_us))
}
