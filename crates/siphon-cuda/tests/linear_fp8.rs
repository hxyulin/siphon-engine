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
        linear_fp8::{
            Fp8Gemm, Fp8Gemv, Fp8Matrix, Fp8RawGemm, Fp8Rows, LinearFp8, Mxfp8Gemm, Mxfp8Gemv,
        },
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

/// `x` rows of k values as the FP8 GEMM quantizes them, dequantized: MXFP8 blocks of 32
/// with a UE8M0 scale, the block's amax / 448 rounded up to a power of two.
fn mxfp8_rows(x: &[bf16]) -> Vec<f64> {
    x.chunks(32)
        .flat_map(|block| {
            let amax = block.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
            let raw = amax * (1.0 / 448.0);
            let mut exponent = ((raw.to_bits() + 0x007f_ffff) & 0x7f80_0000) >> 23;
            if raw <= f32::from_bits(1 << 23) / 2.0 {
                exponent = 0;
            }
            let exponent = exponent.min(254);
            let inverse = if exponent == 0 {
                2f32.powi(127)
            } else {
                1.0 / f32::from_bits(exponent << 23)
            };
            let scale = 2f64.powi(i32::try_from(exponent).expect("small") - 127);
            block.iter().map(move |v| {
                let q = (v.to_f32() * inverse).clamp(-448.0, 448.0);
                f64::from(e4m3_value(e4m3_from_f32(q))) * scale
            })
        })
        .collect()
}

/// `x` rows of k values as the sm_89 FP8 GEMM quantizes them, dequantized: per-row E4M3
/// with one F32 scale (`amax / 448`) per row.
fn per_row_fp8_rows(x: &[bf16], k: usize) -> Vec<f64> {
    x.chunks_exact(k)
        .flat_map(|row| {
            let amax = row.iter().fold(0f32, |m, v| m.max(v.to_f32().abs()));
            let scale = f64::from(amax / 448.0);
            let inv = if amax > 0.0 { 448.0 / amax } else { 0.0 };
            row.iter()
                .map(move |v| f64::from(e4m3_value(e4m3_from_f32(v.to_f32() * inv))) * scale)
        })
        .collect()
}

/// The FP8 GEMM against the F64 reference over the quantized weights and inputs: batches of
/// products that share an input and that do not, partial row and column tiles, and k of
/// the attention inputs, the output projection and the message gates.
#[test]
fn gemm_matches_reference() -> anyhow::Result<()> {
    // (m, k, [(n, input)]): products with the same input index share x.
    type Case = (usize, usize, &'static [(usize, usize)]);
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let fp8 = LinearFp8::load(gpu.context())?;
    let mut scratch = Fp8Rows::new(stream, 3 * 2048 * 4608)?;
    let u = f64::from(f32::EPSILON) / 2.0;
    let cases: &[Case] = &[
        (
            300,
            2304,
            &[(3072, 0), (24, 0), (1024, 1), (32, 1), (384, 2)],
        ),
        (2048, 3072, &[(2304, 0)]),
        (1921, 2304, &[(2305, 0)]),
        // m = 32 exercises gemm_fp8_32x64 (B=32 decode / B=4 W=8 verify).
        (
            32,
            2304,
            &[(3072, 0), (384, 0), (3072, 0), (24, 0), (1024, 1), (1024, 2)],
        ),
        // m = 64 exercises gemm_fp8_64x64.
        (64, 3072, &[(2304, 0)]),
        // m = 256 exercises DFlash 2 shapes (o: k=4096, down: k=6912, gate+up: k=2304, fc: k=9216).
        (256, 4096, &[(2304, 0)]),
        (256, 6912, &[(2304, 0)]),
        (256, 2304, &[(6912, 0), (6912, 0)]),
        (256, 9216, &[(2304, 0)]),
        (77, 4608, &[(64, 0), (64, 1), (64, 2)]),
        (2, 2304, &[(130, 0)]),
    ];
    for (case, &(m, k, products)) in cases.iter().enumerate() {
        let inputs: Vec<Vec<bf16>> = (0..3)
            .map(|i| hash_values(100 + 10 * case as u32 + i, m * k))
            .collect();
        let quantized: Vec<Vec<f64>> = inputs
            .iter()
            .map(|x| {
                if cfg!(sm89) {
                    per_row_fp8_rows(x, k)
                } else {
                    mxfp8_rows(x)
                }
            })
            .collect();
        let x_dev = inputs
            .iter()
            .map(|x| stream.clone_htod(x))
            .collect::<Result<Vec<_>, _>>()?;
        let mats = products
            .iter()
            .enumerate()
            .map(|(i, &(n, _))| {
                let w = hash_values(7 + 31 * case as u32 + i as u32, n * k);
                fp8.quantize_any(stream, &stream.clone_htod(&w)?, n, k)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut outs = products
            .iter()
            .map(|&(n, _)| stream.alloc_zeros::<bf16>(m * n))
            .collect::<Result<Vec<_>, _>>()?;
        {
            let mut gemms: Vec<Fp8Gemm<'_>> = mats
                .iter()
                .zip(products)
                .zip(outs.iter_mut())
                .map(|((weight, &(_, input)), out)| Fp8Gemm {
                    weight,
                    x: x_dev[input].slice(..),
                    out: out.slice_mut(..),
                })
                .collect();
            fp8.gemm(stream, &mut gemms, m, &mut scratch)?;
        }
        let mut worst = 0.0f64;
        for (i, ((mat, &(n, input)), out)) in mats.iter().zip(products).zip(&outs).enumerate() {
            let (bytes, scales) = read_back(&gpu, mat)?;
            let got = stream.clone_dtoh(out)?;
            let x = &quantized[input];
            // Every row of the small cases; both ends and every 61st row of the large one.
            let rows: Vec<usize> = (0..m)
                .filter(|&r| m <= 300 || r % 61 == 0 || r + 2 >= m || r < 2)
                .collect();
            for r in rows {
                for c in 0..n {
                    let (sum, abs) = bytes[c * k..(c + 1) * k]
                        .iter()
                        .zip(&x[r * k..(r + 1) * k])
                        .fold((0.0, 0.0), |(sum, abs), (&q, &xv)| {
                            let p = f64::from(e4m3_value(q)) * xv;
                            (sum + p, abs + p.abs())
                        });
                    let s = f64::from(scales[c]);
                    let want = s * sum;
                    // F32 sums in any order (Higham's gamma for k terms), the scale's
                    // rounding and the output's half BF16 ulp. On sm_89, Ada's native FP8
                    // MMA accumulates each 32-element dot internally with ~14-bit significand
                    // before adding to the F32 accumulator.
                    let ku = k as f64 * u;
                    let ada_mma_err = if cfg!(sm89) { 2f64.powi(-10) } else { 0.0 };
                    let bound = s * (ku / (1.0 - ku) + ada_mma_err) * abs + want.abs() * 2.0 * u;
                    let allowed = bound + bf16_half_ulp(want + bound.copysign(want));
                    let err = (got[r * n + c].to_f64() - want).abs();
                    assert!(
                        err <= allowed,
                        "case {case} product {i} row {r} column {c}: got {}, want {want}, \
                         error {err:e} > {allowed:e}",
                        got[r * n + c]
                    );
                    worst = worst.max(err / allowed);
                }
            }
        }
        eprintln!("gemm case {case} ({m} x {k}): worst error {worst:.3} of bound");
    }
    Ok(())
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

/// Checks `LinearFp8::gemm_raw` (separate unpadded E4M3 values + F32 scales, as used by
/// batched DFlash 2) matches `LinearFp8::gemm` bit for bit across 32x64, 64x64, and
/// 128x128 tiles and survives CUDA graph capture/replay.
#[test]
fn gemm_raw_matches_gemm_and_captures_graph() -> anyhow::Result<()> {
    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let fp8 = LinearFp8::load(gpu.context())?;
    let mut scratch = Fp8Rows::new(stream, 256 * 9216)?;

    let cases: &[(usize, usize, &[usize])] = &[
        (32, 2304, &[4096, 1024, 1024]),
        (64, 4096, &[2304]),
        (256, 6912, &[2304]),
        (256, 2304, &[6912, 6912]),
        (256, 9216, &[2304]),
    ];
    for (ci, &(m, k, ns)) in cases.iter().enumerate() {
        let x = stream.clone_htod(&hash_values(500 + ci as u32, m * k))?;
        let mats = ns
            .iter()
            .enumerate()
            .map(|(i, &n)| {
                let w = stream.clone_htod(&hash_values(600 + 13 * ci as u32 + i as u32, n * k))?;
                fp8.quantize_any(stream, &w, n, k)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        // Build separate unpadded values + F32 scales buffers matching `ops::draft::Fp8Rows`.
        let raw_bufs = mats
            .iter()
            .map(|mat| -> anyhow::Result<_> {
                let (bytes, scales) = read_back(&gpu, mat)?;
                Ok((stream.clone_htod(&bytes)?, stream.clone_htod(&scales)?))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut want_outs = ns
            .iter()
            .map(|&n| stream.alloc_zeros::<bf16>(m * n))
            .collect::<Result<Vec<_>, _>>()?;
        let mut got_outs = ns
            .iter()
            .map(|&n| stream.alloc_zeros::<bf16>(m * n))
            .collect::<Result<Vec<_>, _>>()?;
        {
            let mut prods: Vec<_> = mats
                .iter()
                .zip(&mut want_outs)
                .map(|(weight, out)| Fp8Gemm {
                    weight,
                    x: x.slice(..),
                    out: out.slice_mut(..),
                })
                .collect();
            fp8.gemm(stream, &mut prods, m, &mut scratch)?;
        }
        // Capture and replay `gemm_raw` in a CUDA graph.
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        {
            let mut prods: Vec<_> = raw_bufs
                .iter()
                .zip(ns)
                .zip(&mut got_outs)
                .map(|(((vals, sc), &n), out)| Fp8RawGemm {
                    values: vals.slice(..),
                    scales: sc.slice(..),
                    rows: n,
                    k,
                    x: x.slice(..),
                    out: out.slice_mut(..),
                })
                .collect();
            fp8.gemm_raw(stream, &mut prods, m, &mut scratch)?;
        }
        let graph = stream
            .end_capture(CUgraphInstantiate_flags(0))?
            .ok_or_else(|| anyhow::anyhow!("empty capture"))?;
        graph.launch()?;
        for (i, (want, got)) in want_outs.iter().zip(&got_outs).enumerate() {
            let w = stream.clone_dtoh(want)?;
            let g = stream.clone_dtoh(got)?;
            assert_eq!(
                w, g,
                "case {ci} ({m} x {k}) product {i}: gemm_raw differed from gemm"
            );
        }
    }
    Ok(())
}

/// Checks `LinearFp8::quantize_mxfp8` + `LinearFp8::gemm_mxfp8` (block-32 UE8M0 weight
/// scales, used for `W_o` under `--prefill-dense fp8`) against an F64 reference across
/// `32x64`, `64x64`, and `128x128` tiles, partial shapes, and CUDA graph capture/replay.
#[test]
#[cfg_attr(sm89, ignore = "needs the block-scaled MMA, which sm_89 lacks")]
fn gemm_mxfp8_matches_reference() -> anyhow::Result<()> {
    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let fp8 = LinearFp8::load(gpu.context())?;
    let mut scratch = Fp8Rows::new(stream, 2048 * 3072)?;
    let u = f64::from(f32::EPSILON) / 2.0;

    // Cover 32x64 (m=2, m=32), 64x64 (m=64, m=256), 128x128 (m=2048), and odd n.
    let cases: &[(usize, usize, usize)] = &[
        (2, 130, 3072),
        (32, 2304, 3072),
        (64, 2304, 3072),
        (256, 2304, 3072),
        (2048, 2304, 3072),
        (37, 2305, 2304),
    ];
    for (ci, &(m, n, k)) in cases.iter().enumerate() {
        let x_host = hash_values(800 + ci as u32, m * k);
        let mut w_host = hash_values(900 + ci as u32, n * k);
        // Inject large head outliers into W_o rows to exercise per-32 block scales.
        for r in (0..n).step_by(17) {
            w_host[r * k + 64] = bf16::from_f32(32.0);
        }
        let x_ref = mxfp8_rows(&x_host);
        let w_ref = mxfp8_rows(&w_host);
        let x_dev = stream.clone_htod(&x_host)?;
        let w_dev = stream.clone_htod(&w_host)?;
        let mat = fp8.quantize_mxfp8(stream, &w_dev, n, k)?;
        let mut out_dev = stream.alloc_zeros::<bf16>(m * n)?;

        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        fp8.gemm_mxfp8(
            stream,
            &mut [Mxfp8Gemm {
                weight: &mat,
                x: x_dev.slice(..),
                out: out_dev.slice_mut(..),
            }],
            m,
            &mut scratch,
        )?;
        let graph = stream
            .end_capture(CUgraphInstantiate_flags(0))?
            .ok_or_else(|| anyhow::anyhow!("empty capture"))?;
        graph.launch()?;

        let got = stream.clone_dtoh(&out_dev)?;
        let mut worst = 0.0f64;
        let ku = k as f64 * u;
        let gamma = ku / (1.0 - ku);
        let rows: Vec<usize> = (0..m)
            .filter(|&r| m <= 64 || r % 37 == 0 || r + 2 >= m || r < 2)
            .collect();
        for r in rows {
            for c in (0..n).filter(|&c| n <= 200 || c % 19 == 0 || c + 2 >= n || c < 2) {
                let (sum, abs) = w_ref[c * k..(c + 1) * k]
                    .iter()
                    .zip(&x_ref[r * k..(r + 1) * k])
                    .fold((0.0, 0.0), |(sum, abs), (&wv, &xv)| {
                        let p = wv * xv;
                        (sum + p, abs + p.abs())
                    });
                let bound = gamma * abs;
                let allowed = bound + bf16_half_ulp(sum + bound.copysign(sum));
                let err = (got[r * n + c].to_f64() - sum).abs();
                assert!(
                    err <= allowed,
                    "mxfp8 case {ci} ({m} x {n} x {k}) row {r} col {c}: got {}, want {sum}, \
                     err {err:e} > {allowed:e}",
                    got[r * n + c]
                );
                worst = worst.max(err / allowed);
            }
        }
        eprintln!("gemm_mxfp8 case {ci} ({m} x {n} x {k}): worst error {worst:.3} of bound");
    }
    Ok(())
}

/// Checks `LinearFp8::gemv_mxfp8` (`gemv_mxfp8_k3072`) against the F64 exact product of the
/// MXFP8 weights with BF16 inputs across partial and full `W_o` row counts, both `Serial`
/// and `Early` (PDL) starts, multi-segment batches, all-zero rows, and outlier rows.
#[test]
fn gemv_mxfp8_matches_reference() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let fp8 = LinearFp8::load(gpu.context())?;
    let linear = Linear::load(gpu.context())?;
    let u = f64::from(f32::EPSILON) / 2.0;
    let k = 3072usize;
    let ku = k as f64 * u;
    let gamma = ku / (1.0 - ku);

    for (ci, &n) in [1usize, 7, 37, 100, 2304].iter().enumerate() {
        let mut w_host = hash_values(1100 + ci as u32, n * k);
        // Inject head outliers and an all-zero row when present.
        for r in (0..n).step_by(13) {
            w_host[r * k + 32] = bf16::from_f32(32.0);
        }
        if n > 2 {
            w_host[k..2 * k].fill(bf16::from_f32(0.0));
        }
        let x_host = hash_values(1200 + ci as u32, k);
        let w_ref = mxfp8_rows(&w_host);

        let w_dev = stream.clone_htod(&w_host)?;
        let x_dev = stream.clone_htod(&x_host)?;
        let mat = fp8.quantize_mxfp8(stream, &w_dev, n, k)?;

        for start in [Start::Serial, Start::Early] {
            let mut out_dev = stream.alloc_zeros::<bf16>(n)?;
            fp8.gemv_mxfp8(stream, &mut [Mxfp8Gemv::new(&mat, &x_dev, &mut out_dev)], start)?;
            let got = stream.clone_dtoh(&out_dev)?;

            let mut worst = 0.0f64;
            for r in 0..n {
                let (sum, abs) = w_ref[r * k..(r + 1) * k]
                    .iter()
                    .zip(&x_host)
                    .fold((0.0, 0.0), |(sum, abs), (&wv, &xv)| {
                        let p = wv * f64::from(xv.to_f32());
                        (sum + p, abs + p.abs())
                    });
                let bound = gamma * abs;
                let allowed = bound + bf16_half_ulp(sum + bound.copysign(sum));
                let err = (got[r].to_f64() - sum).abs();
                assert!(
                    err <= allowed,
                    "gemv_mxfp8 n={n} start={start:?} row={r}: got {}, want {sum}, err {err:e} > {allowed:e}",
                    got[r]
                );
                worst = worst.max(if allowed > 0.0 { err / allowed } else { 0.0 });
            }
            if start == Start::Serial {
                // Compare against BF16 GEMV on the full W_o shape to check quantization error.
                let mut bf16_out = stream.alloc_zeros::<bf16>(n)?;
                linear.gemv_bf16(
                    stream,
                    &mut [Gemv::new(&w_dev, &x_dev, &mut bf16_out)],
                    Start::Serial,
                )?;
                let bf16_got = stream.clone_dtoh(&bf16_out)?;
                let max_bf16_diff = got
                    .iter()
                    .zip(&bf16_got)
                    .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "gemv_mxfp8 n={n} k={k}: worst error {worst:.3} of bound, max |mxfp8 - bf16| = {max_bf16_diff:.4}"
                );
            }
        }
    }

    // Multi-segment batch: two MXFP8 weights sharing k = 3072.
    let (n0, n1) = (37usize, 2304usize);
    let w0_dev = stream.clone_htod(&hash_values(1301, n0 * k))?;
    let w1_dev = stream.clone_htod(&hash_values(1302, n1 * k))?;
    let m0 = fp8.quantize_mxfp8(stream, &w0_dev, n0, k)?;
    let m1 = fp8.quantize_mxfp8(stream, &w1_dev, n1, k)?;
    let x0_dev = stream.clone_htod(&hash_values(1303, k))?;
    let x1_dev = stream.clone_htod(&hash_values(1304, k))?;
    let mut o0_batch = stream.alloc_zeros::<bf16>(n0)?;
    let mut o1_batch = stream.alloc_zeros::<bf16>(n1)?;
    let mut o0_solo = stream.alloc_zeros::<bf16>(n0)?;
    let mut o1_solo = stream.alloc_zeros::<bf16>(n1)?;
    fp8.gemv_mxfp8(
        stream,
        &mut [
            Mxfp8Gemv::new(&m0, &x0_dev, &mut o0_batch),
            Mxfp8Gemv::new(&m1, &x1_dev, &mut o1_batch),
        ],
        Start::Early,
    )?;
    fp8.gemv_mxfp8(
        stream,
        &mut [Mxfp8Gemv::new(&m0, &x0_dev, &mut o0_solo)],
        Start::Serial,
    )?;
    fp8.gemv_mxfp8(
        stream,
        &mut [Mxfp8Gemv::new(&m1, &x1_dev, &mut o1_solo)],
        Start::Serial,
    )?;
    assert_eq!(stream.clone_dtoh(&o0_batch)?, stream.clone_dtoh(&o0_solo)?);
    assert_eq!(stream.clone_dtoh(&o1_batch)?, stream.clone_dtoh(&o1_solo)?);
    Ok(())
}

/// Times `LinearFp8::gemm` / `gemm_mxfp8` next to `Linear::gemm_bf16` on batched decode,
/// batched verify, DFlash 2, and prefill dense shapes. Run with
/// `cargo test --release --test linear_fp8 -- --ignored --nocapture gemm_fp8_throughput`.
#[test]
#[ignore = "timing, not a check"]
#[expect(clippy::too_many_lines, reason = "benchmarks multiple shapes and formats")]
fn gemm_fp8_throughput() -> anyhow::Result<()> {
    use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};
    use siphon_cuda::ops::linear::Gemm;

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let fp8 = LinearFp8::load(gpu.context())?;
    let linear = Linear::load(gpu.context())?;
    let mut scratch = Fp8Rows::new(stream, 3 * 2048 * 4608)?;

    let batches: &[(&str, usize, &[usize], bool)] = &[
        (
            "attn inputs",
            2304,
            &[3072, 384, 3072, 24, 1024, 1024, 32, 32],
            false,
        ),
        ("W_o (mxfp8)", 3072, &[2304], true),
        ("dflash2 o", 4096, &[2304], false),
        ("dflash2 down", 6912, &[2304], false),
        ("dflash2 gate+up", 2304, &[6912, 6912], false),
    ];
    eprintln!(
        "{:18} {:>5} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "batch", "m", "fp8 (us)", "fp8 TF/s", "bf16 (us)", "bf16 TF/s", "speedup"
    );
    for &(label, k, ns, mxfp8) in batches {
        if mxfp8 && !siphon_cuda::BLOCK_SCALED_MMA {
            continue;
        }
        let n_sum: usize = ns.iter().sum();
        let bf16_w: Vec<_> = ns
            .iter()
            .enumerate()
            .map(|(i, &n)| stream.clone_htod(&hash_values(10 + i as u32, n * k)))
            .collect::<Result<_, _>>()?;
        let fp8_w: Vec<_> = bf16_w
            .iter()
            .zip(ns)
            .map(|(w, &n)| fp8.quantize_any(stream, w, n, k))
            .collect::<anyhow::Result<_>>()?;
        let mxfp8_w: Vec<_> = if mxfp8 {
            bf16_w
                .iter()
                .zip(ns)
                .map(|(w, &n)| fp8.quantize_mxfp8(stream, w, n, k))
                .collect::<anyhow::Result<_>>()?
        } else {
            Vec::new()
        };
        for m in [32, 64, 256, 2048] {
            if label.starts_with("dflash2") && m > 256 {
                continue;
            }
            let x = stream.clone_htod(&hash_values(99, m * k))?;
            let mut outs: Vec<_> = ns
                .iter()
                .map(|&n| stream.alloc_zeros::<bf16>(m * n))
                .collect::<Result<_, _>>()?;
            let mut bench = |use_fp8: bool| -> anyhow::Result<f64> {
                const LAUNCHES: usize = 20;
                const REPLAYS: usize = 5;
                stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                for _ in 0..LAUNCHES {
                    if use_fp8 && mxfp8 {
                        let mut products: Vec<_> = mxfp8_w
                            .iter()
                            .zip(&mut outs)
                            .map(|(w, out)| Mxfp8Gemm {
                                weight: w,
                                x: x.slice(..),
                                out: out.slice_mut(..),
                            })
                            .collect();
                        fp8.gemm_mxfp8(stream, &mut products, m, &mut scratch)?;
                    } else if use_fp8 {
                        let mut products: Vec<_> = fp8_w
                            .iter()
                            .zip(&mut outs)
                            .map(|(w, out)| Fp8Gemm {
                                weight: w,
                                x: x.slice(..),
                                out: out.slice_mut(..),
                            })
                            .collect();
                        fp8.gemm(stream, &mut products, m, &mut scratch)?;
                    } else {
                        let mut products: Vec<_> = bf16_w
                            .iter()
                            .zip(&mut outs)
                            .map(|(w, out)| Gemm::new(w, &x, out))
                            .collect();
                        linear.gemm_bf16(stream, &mut products, m)?;
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
            let fp8_us = bench(true)?;
            let bf16_us = bench(false)?;
            let flop = 2.0 * (m * n_sum * k) as f64;
            eprintln!(
                "{label:18} {m:5} {fp8_us:10.2} {:10.1} {bf16_us:10.2} {:10.1} {:7.2}x",
                flop / fp8_us / 1e6,
                flop / bf16_us / 1e6,
                bf16_us / fp8_us,
            );
        }
    }
    Ok(())
}
