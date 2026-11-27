//! The NVFP4 quantizer and matrix-vector kernels.
//!
//! The quantizer must match a CPU model of per-row Four Over Six NVFP4 (squared error)
//! bit for bit. The products are checked against an F64 reference over the dequantized
//! weights, and the tests also print RMS distance from the BF16 products.

#![cfg(target_os = "linux")]

use std::sync::OnceLock;

use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};
use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::{
        linear::{Gemv, Linear, Start},
        linear_fp8::LinearFp8,
        linear_nvfp4::{LinearNvfp4, Nvfp4Gemv, Nvfp4Matrix},
        linear_rows::{LinearRows, MAX_ROWS, RowsProduct},
    },
};

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

fn e2m1_value(nibble: u8) -> f64 {
    const VALUES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = VALUES[usize::from(nibble & 7)];
    if nibble & 8 == 0 { v } else { -v }
}

fn e4m3_value(bits: u8) -> f64 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 3) & 0xf);
    let mantissa = f64::from(bits & 7) / 8.0;
    if exponent == 0 {
        sign * mantissa * 2f64.powi(-6)
    } else {
        sign * (1.0 + mantissa) * 2f64.powi(exponent - 7)
    }
}

fn e4m3_from_f32(x: f32) -> u8 {
    static VALUES: OnceLock<Vec<f64>> = OnceLock::new();
    let values = VALUES.get_or_init(|| (0..=126).map(e4m3_value).collect());
    if x.is_nan() {
        return 0x7f;
    }
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = f64::from(x.abs());
    let above = values.partition_point(|&v| v < a).min(values.len() - 1);
    let below = above.saturating_sub(1);
    let (d_above, d_below) = (values[above] - a, a - values[below]);
    let code = match d_below.total_cmp(&d_above) {
        std::cmp::Ordering::Less => below,
        std::cmp::Ordering::Equal if below % 2 == 0 => below,
        _ => above,
    };
    sign | u8::try_from(code).expect("codes fit in a byte")
}

fn to_e2m1(x: f32) -> (u8, f32) {
    let a = x.abs();
    let code: u8 = if a <= 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a <= 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a <= 2.5 {
        4
    } else if a < 3.5 {
        5
    } else if a <= 5.0 {
        6
    } else {
        7
    };
    let value = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(code)];
    if x.is_sign_negative() {
        (code | 8, -value)
    } else {
        (code, value)
    }
}

fn fp4_candidate(x: &[f32], scale: f32, inverse: f32, outer: f32) -> ([u8; 8], f32) {
    let mut bytes = [0u8; 8];
    let mut error = [0f32; 16];
    for i in 0..16 {
        let (code, value) = to_e2m1(x[i] * inverse);
        bytes[i / 2] |= code << (4 * (i % 2));
        let d = (value * scale) * outer - x[i];
        error[i] = d * d;
    }
    let mut width = 8;
    while width > 0 {
        for i in 0..width {
            error[i] = error[2 * i] + error[2 * i + 1];
        }
        width /= 2;
    }
    (bytes, error[0])
}

fn quantize_cpu(w: &[bf16], k: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
    let rows = w.len() / k;
    let mut values = Vec::with_capacity(rows * (k / 2));
    let mut scales = Vec::with_capacity(rows * (k / 16));
    let mut outer = Vec::with_capacity(rows);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "matches the F32 constants in the kernel"
    )]
    let (sixth, inv_1536) = ((1.0f64 / 6.0) as f32, (1.0f64 / 1536.0) as f32);
    for row in w.chunks_exact(k) {
        let row_f: Vec<f32> = row.iter().map(|v| v.to_f32()).collect();
        let amax = row_f.iter().fold(0f32, |m, v| m.max(v.abs()));
        let encoder = (1536.0 / amax).min(f32::MAX);
        let decoder = 1.0 / encoder;
        let out_s = amax * inv_1536;
        outer.push(out_s);
        let (encoder6, encoder4) = (encoder * sixth, encoder * 0.25);
        for block in row_f.as_chunks::<16>().0 {
            let block_max = block.iter().fold(0f32, |m, v| m.max(v.abs()));
            #[expect(
                clippy::cast_possible_truncation,
                reason = "E4M3 values are exact in F32"
            )]
            let round = |v: f32| e4m3_value(e4m3_from_f32(v)) as f32;
            let cand_s6 = round((block_max * encoder6).min(448.0));
            let cand_s4 = round((block_max * encoder4).min(448.0));
            let inverse6 = (1.0 / (cand_s6 * decoder)).min(f32::MAX);
            let inverse4 = (1.0 / (cand_s4 * decoder)).min(f32::MAX);
            let (bytes6, err6) = fp4_candidate(block, cand_s6, inverse6, out_s);
            let (bytes4, err4) = fp4_candidate(block, cand_s4, inverse4, out_s);
            let (bytes, chosen_scale) = if err4 < err6 {
                (bytes4, cand_s4)
            } else {
                (bytes6, cand_s6)
            };
            scales.push(e4m3_from_f32(chosen_scale));
            values.extend_from_slice(&bytes);
        }
    }
    (values, scales, outer)
}

fn read_back(gpu: &Gpu, m: &Nvfp4Matrix) -> anyhow::Result<(Vec<u8>, Vec<u8>, Vec<f32>)> {
    let stream = gpu.stream();
    let values = stream.clone_dtoh(&m.values())?;
    let scales = stream.clone_dtoh(&m.scales())?;
    let outer = stream
        .clone_dtoh(&m.outer())?
        .as_chunks()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    Ok((values, scales, outer))
}

const SHAPES: &[(usize, usize)] = &[(37, 2304), (24, 2304), (100, 3072), (1, 2304)];

#[test]
fn quantize_matches_cpu_model() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let nvfp4 = LinearNvfp4::load(gpu.context())?;
    let stream = gpu.stream();
    for (seed, &(rows, k)) in SHAPES.iter().enumerate() {
        let mut w = hash_values(seed as u32 + 1, rows * k);
        if rows > 2 {
            w[k..2 * k].fill(bf16::ZERO);
            w[2 * k] = bf16::from_f32(300.0);
        }
        let dev = stream.clone_htod(&w)?;
        let m = nvfp4.quantize(stream, &dev, rows, k)?;
        let (values, scales, outer) = read_back(&gpu, &m)?;
        let (want_values, want_scales, want_outer) = quantize_cpu(&w, k);
        assert_eq!(outer, want_outer, "outer scales of {rows} x {k}");
        assert_eq!(scales, want_scales, "block scales of {rows} x {k}");
        assert_eq!(values, want_values, "E2M1 bytes of {rows} x {k}");
        eprintln!("{rows} x {k}: values, block scales and outer scales match");
    }
    Ok(())
}

fn reference(values: &[u8], scales: &[u8], outer: &[f32], x: &[bf16], k: usize) -> Vec<(f64, f64)> {
    let unit = f64::from(f32::EPSILON) / 2.0;
    let ku = k as f64 * unit;
    let gamma = ku / (1.0 - ku);
    let row_bytes = k / 2;
    let row_scales = k / 16;
    values
        .chunks_exact(row_bytes)
        .zip(scales.chunks_exact(row_scales))
        .zip(outer)
        .map(|((vrow, srow), &out_s)| {
            let (mut sum, mut abs) = (0.0f64, 0.0f64);
            for (g, &scale_byte) in srow.iter().enumerate() {
                let block_scale = e4m3_value(scale_byte);
                let mut gsum = 0.0f64;
                let mut gabs = 0.0f64;
                for i in 0..16 {
                    let col = g * 16 + i;
                    let byte = vrow[col / 2];
                    let nibble = if col % 2 == 0 { byte & 15 } else { byte >> 4 };
                    let prod = e2m1_value(nibble) * x[col].to_f64();
                    gsum += prod;
                    gabs += prod.abs();
                }
                sum += block_scale * gsum;
                abs += block_scale.abs() * gabs;
            }
            let os = f64::from(out_s);
            let value = os * sum;
            (value, os * gamma * abs + value.abs() * (4.0 * unit))
        })
        .collect()
}

fn bf16_half_ulp(v: f64) -> f64 {
    let v = v.abs().max(f64::from(f32::MIN_POSITIVE));
    (v.log2().floor() - 8.0).exp2()
}

fn check_batch(
    gpu: &Gpu,
    mats: &[Nvfp4Matrix],
    inputs: &[Vec<bf16>],
    got: &[Vec<bf16>],
    want: &[Vec<bf16>],
) -> anyhow::Result<(f64, f64, f64)> {
    let mut worst = 0.0f64;
    let (mut err2, mut ref2) = (0.0f64, 0.0f64);
    for (i, (m, x)) in mats.iter().zip(inputs).enumerate() {
        let (values, scales, outer) = read_back(gpu, m)?;
        for (row, ((g, (r, bound)), b)) in got[i]
            .iter()
            .zip(reference(&values, &scales, &outer, x, m.k()))
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
    let nvfp4 = LinearNvfp4::load(gpu.context())?;
    let linear = Linear::load(gpu.context())?;
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
    let mut seed = 200;
    for batch in batches {
        let mut weights = Vec::new();
        let mut inputs = Vec::new();
        for &(rows, k) in *batch {
            seed += 2;
            let w = hash_values(seed, rows * k);
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
            .map(|(d, &(_, rows, k))| nvfp4.quantize(stream, d, rows, k))
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
                .map(|((m, x), out)| Nvfp4Gemv::new(m, x, out))
                .collect();
            nvfp4.gemv(stream, &mut products, Start::Early)?;
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

/// Checks `LinearNvfp4::quantize_any` against the CPU Four Over Six model on all DFlash 2
/// input widths (`k = 2304, 4096, 6912, 9216`).
#[test]
fn quantize_any_dflash2_shapes() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let nvfp4 = LinearNvfp4::load(gpu.context())?;
    let stream = gpu.stream();
    for (seed, &(rows, k)) in [
        (37, 2304),
        (64, 4096),
        (33, 4608),
        (48, 6912),
        (19, 9216),
    ]
    .iter()
    .enumerate()
    {
        let mut w = hash_values(seed as u32 + 50, rows * k);
        w[k..2 * k].fill(bf16::ZERO);
        let dev = stream.clone_htod(&w)?;
        let m = nvfp4.quantize_any(stream, &dev, rows, k)?;
        let (values, scales, outer) = read_back(&gpu, &m)?;
        let (want_values, want_scales, want_outer) = quantize_cpu(&w, k);
        assert_eq!(outer, want_outer, "outer scales of {rows} x {k}");
        assert_eq!(scales, want_scales, "block scales of {rows} x {k}");
        assert_eq!(values, want_values, "E2M1 bytes of {rows} x {k}");
    }
    Ok(())
}

/// Checks `LinearRows::run` with NVFP4 weights across all 6 `k` widths, multi-product
/// batches (mirroring DFlash 2 Q/K/V and Gate/Up), single-stream `m = 1..16`, batched
/// `m > 16` up to `MAX_ROWS` (256), and CUDA graph capture/replay.
#[test]
#[expect(clippy::too_many_lines, reason = "covers all k widths, m sizes and graph replay")]
fn rows_nvfp4_matches_reference() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let nvfp4 = LinearNvfp4::load(gpu.context())?;
    let rows_op = LinearRows::load(gpu.context())?;

    let batches: &[&[(usize, usize)]] = &[
        // DFlash 2 Q + K + V (k = 2304, 3 segments).
        &[(256, 2304), (64, 2304), (64, 2304)],
        // Unaligned row count + bundled O (k = 3072).
        &[(37, 3072), (17, 3072)],
        // DFlash 2 O (k = 4096).
        &[(128, 4096)],
        // Message gate down width (k = 4608).
        &[(48, 4608)],
        // DFlash 2 MLP down (k = 6912).
        &[(64, 6912)],
        // DFlash 2 fc (k = 9216).
        &[(64, 9216)],
    ];
    let row_counts = [1, 7, 8, 16, 24, 32, 33, 64, 128, MAX_ROWS];
    let mut seed = 500u32;
    for &batch in batches {
        let k = batch[0].1;
        let mut dev_w = Vec::new();
        let mut mats = Vec::new();
        let mut decoded = Vec::new();
        for &(n, width) in batch {
            seed += 1;
            let w = hash_values(seed, n * width);
            let dw = stream.clone_htod(&w)?;
            let mat = nvfp4.quantize_any(stream, &dw, n, width)?;
            decoded.push(read_back(&gpu, &mat)?);
            dev_w.push(dw);
            mats.push(mat);
        }
        seed += 1;
        let x_all: Vec<bf16> = hash_values(seed, MAX_ROWS * k)
            .iter()
            .map(|v| bf16::from_f32(v.to_f32() * 4.0))
            .collect();
        let dev_x_all = stream.clone_htod(&x_all)?;

        for &m in &row_counts {
            let x_view = dev_x_all.slice(..m * k);
            let mut outs: Vec<_> = batch
                .iter()
                .map(|&(n, _)| stream.alloc_zeros::<bf16>(m * n))
                .collect::<Result<_, _>>()?;
            let mut bf16_outs: Vec<_> = batch
                .iter()
                .map(|&(n, _)| stream.alloc_zeros::<bf16>(m * n))
                .collect::<Result<_, _>>()?;
            {
                let mut prods: Vec<_> = mats
                    .iter()
                    .zip(&mut outs)
                    .map(|(mat, out)| RowsProduct::nvfp4(mat, &x_view, out))
                    .collect();
                rows_op.run(stream, &mut prods, m, Start::Early)?;
                let mut bf_prods: Vec<_> = dev_w
                    .iter()
                    .zip(&mut bf16_outs)
                    .map(|(w, out)| RowsProduct::bf16(w, &x_view, out))
                    .collect();
                rows_op.run(stream, &mut bf_prods, m, Start::Serial)?;
            }
            let mut worst = 0.0f64;
            let (mut err2, mut ref2) = (0.0f64, 0.0f64);
            for (i, &(n, _)) in batch.iter().enumerate() {
                let got = stream.clone_dtoh(&outs[i])?;
                let want = stream.clone_dtoh(&bf16_outs[i])?;
                let (values, scales, outer) = &decoded[i];
                for r in 0..m {
                    let xr = &x_all[r * k..(r + 1) * k];
                    let gr = &got[r * n..(r + 1) * n];
                    let wr = &want[r * n..(r + 1) * n];
                    for (col, ((g, (rf, bound)), b)) in gr
                        .iter()
                        .zip(reference(values, scales, outer, xr, k))
                        .zip(wr)
                        .enumerate()
                    {
                        let err = (g.to_f64() - rf).abs();
                        let allowed = bound + bf16_half_ulp(rf + bound.copysign(rf));
                        assert!(
                            err <= allowed,
                            "rows_nvfp4 batch {batch:?} m={m} seg={i} row={r} col={col}: \
                             got {g}, ref {rf}, err {err:e} > {allowed:e}"
                        );
                        worst = worst.max(err / allowed);
                        let d = g.to_f64() - b.to_f64();
                        err2 += d * d;
                        ref2 += b.to_f64() * b.to_f64();
                    }
                }
            }
            let rel_rms = (err2 / ref2).sqrt();
            assert!(
                rel_rms < 0.15,
                "rows_nvfp4 batch {batch:?} m={m}: rel RMS {rel_rms:.3} too large vs BF16"
            );
            if m == 8 || m == 64 {
                eprintln!(
                    "rows_nvfp4 {batch:?} m={m}: worst {worst:.3} of bound, rel RMS vs BF16 \
                     {rel_rms:.3}"
                );
            }
        }

        // Verify CUDA graph capture and replay matches eager output bit-for-bit at m = 8
        // (single-stream draft) and m = 64 (batched draft).
        for m in [8usize, 64] {
            let x_view = dev_x_all.slice(..m * k);
            let mut eager_outs: Vec<_> = batch
                .iter()
                .map(|&(n, _)| stream.alloc_zeros::<bf16>(m * n))
                .collect::<Result<_, _>>()?;
            let mut graph_outs: Vec<_> = batch
                .iter()
                .map(|&(n, _)| stream.alloc_zeros::<bf16>(m * n))
                .collect::<Result<_, _>>()?;
            {
                let mut prods: Vec<_> = mats
                    .iter()
                    .zip(&mut eager_outs)
                    .map(|(mat, out)| RowsProduct::nvfp4(mat, &x_view, out))
                    .collect();
                rows_op.run(stream, &mut prods, m, Start::Early)?;
            }
            stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let enqueued = {
                let mut prods: Vec<_> = mats
                    .iter()
                    .zip(&mut graph_outs)
                    .map(|(mat, out)| RowsProduct::nvfp4(mat, &x_view, out))
                    .collect();
                rows_op.run(stream, &mut prods, m, Start::Early)
            };
            let graph = stream.end_capture(CUgraphInstantiate_flags(0));
            enqueued?;
            let graph = graph?.ok_or_else(|| anyhow::anyhow!("empty rows_nvfp4 capture"))?;
            graph.upload()?;
            graph.launch()?;
            for (i, (eager, graph)) in eager_outs.iter().zip(&graph_outs).enumerate() {
                let a = stream.clone_dtoh(eager)?;
                let b = stream.clone_dtoh(graph)?;
                assert!(
                    a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "CUDA graph output differs from eager for {batch:?} m={m} seg={i}"
                );
            }
        }
    }
    Ok(())
}

/// Checks bit-exact row invariance of `LinearRows` with NVFP4 weights across `m = 1..=256`.
#[test]
fn rows_nvfp4_is_row_invariant() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let nvfp4 = LinearNvfp4::load(gpu.context())?;
    let rows_op = LinearRows::load(gpu.context())?;

    for (i, &(n, k)) in [(64, 2304), (48, 4096), (37, 6912), (32, 9216)]
        .iter()
        .enumerate()
    {
        let w = hash_values(700 + i as u32, n * k);
        let dw = stream.clone_htod(&w)?;
        let mat = nvfp4.quantize_any(stream, &dw, n, k)?;
        let x = hash_values(800 + i as u32, MAX_ROWS * k);
        let mut other = hash_values(900 + i as u32, MAX_ROWS * k);
        other[..k].copy_from_slice(&x[..k]);
        other[3 * k] = bf16::from_f32(3e38);

        let run = |input: &[bf16], m: usize| -> anyhow::Result<Vec<bf16>> {
            let dx = stream.clone_htod(input)?;
            let mut out = stream.alloc_zeros::<bf16>(m * n)?;
            let mut prods = [RowsProduct::nvfp4(&mat, &dx, &mut out)];
            rows_op.run(stream, &mut prods, m, Start::Serial)?;
            Ok(stream.clone_dtoh(&out)?)
        };

        let full = run(&x, MAX_ROWS)?;
        for &m in &[1, 2, 7, 8, 15, 16, 24, 32, 33, 64, 128, 255] {
            let part = run(&x[..m * k], m)?;
            assert!(
                part.iter()
                    .zip(&full)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "nvfp4 {n} x {k}: rows of m = {m} differ from m = {MAX_ROWS}"
            );
        }
        let mixed = run(&other, MAX_ROWS)?;
        assert!(
            mixed[..n]
                .iter()
                .zip(&full[..n])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "nvfp4 {n} x {k}: row 0 depends on the other rows"
        );
    }
    Ok(())
}

/// Times the NVFP4 products from DRAM and L2 next to FP8.
/// Run with `cargo test --release --test linear_nvfp4 -- --ignored --nocapture`.
#[test]
#[ignore = "timing, not a check"]
#[expect(clippy::too_many_lines, reason = "benchmarks multiple shapes and cache residencies")]
fn gemv_nvfp4_bandwidth() -> anyhow::Result<()> {
    use cudarc::driver::sys::CUevent_flags;
    use siphon_cuda::ops::linear_fp8::Fp8Gemv;

    const ROTATE_BYTES: usize = 512 << 20;
    const LAUNCHES: usize = 40;
    const REPLAYS: usize = 10;

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
    ];
    let gpu = Gpu::new(0)?;
    let nvfp4 = LinearNvfp4::load(gpu.context())?;
    let fp8 = LinearFp8::load(gpu.context())?;
    let stream = gpu.stream();

    for (&(label, shapes), hot) in cases.iter().flat_map(|c| [(c, false), (c, true)]) {
        let k = shapes[0].1;
        let nvfp4_bytes: usize = shapes.iter().map(|(n, k)| n * k * 9 / 16 + 4 * n).sum();
        let fp8_bytes: usize = shapes.iter().map(|(n, k)| n * k + 4 * n).sum();
        let rotate = if hot { 0 } else { ROTATE_BYTES };
        let copies = rotate.div_ceil(nvfp4_bytes).clamp(1, LAUNCHES);
        let src_weights: Vec<_> = shapes
            .iter()
            .map(|&(n, k)| stream.alloc_zeros::<bf16>(n * k))
            .collect::<Result<_, _>>()?;
        let nvfp4_weights = (0..copies)
            .map(|_| {
                shapes
                    .iter()
                    .enumerate()
                    .map(|(i, &(n, k))| nvfp4.quantize(stream, &src_weights[i], n, k))
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let fp8_copies = rotate.div_ceil(fp8_bytes).clamp(1, LAUNCHES);
        let fp8_weights = (0..fp8_copies)
            .map(|_| {
                shapes
                    .iter()
                    .enumerate()
                    .map(|(i, &(n, k))| fp8.quantize(stream, &src_weights[i], n, k))
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let x = stream.alloc_zeros::<bf16>(k)?;
        let mut outs: Vec<_> = shapes
            .iter()
            .map(|&(n, _)| stream.alloc_zeros::<bf16>(n))
            .collect::<Result<_, _>>()?;

        let mut time = |use_nvfp4: bool| -> anyhow::Result<f64> {
            stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            for i in 0..LAUNCHES {
                if use_nvfp4 {
                    let mut products: Vec<_> = nvfp4_weights[i % copies]
                        .iter()
                        .zip(&mut outs)
                        .map(|(w, out)| Nvfp4Gemv::new(w, &x, out))
                        .collect();
                    nvfp4.gemv(stream, &mut products, Start::Early)?;
                } else {
                    let mut products: Vec<_> = fp8_weights[i % fp8_copies]
                        .iter()
                        .zip(&mut outs)
                        .map(|(w, out)| Fp8Gemv::new(w, &x, out))
                        .collect();
                    fp8.gemv(stream, &mut products, Start::Early)?;
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
        let nv_us = time(true)?;
        let fp8_us = time(false)?;
        let tag = if hot {
            format!("{label}, L2")
        } else {
            label.to_owned()
        };
        eprintln!(
            "{tag:28} nvfp4 {:6.2} MB {nv_us:7.2} us {:6.3} TB/s | fp8 {:6.2} MB {fp8_us:7.2} us \
             {:6.3} TB/s",
            nvfp4_bytes as f64 / 1e6,
            nvfp4_bytes as f64 / nv_us / 1e6,
            fp8_bytes as f64 / 1e6,
            fp8_bytes as f64 / fp8_us / 1e6,
        );
    }
    Ok(())
}

#[test]
fn quantize_indexed_matches_quantize_any() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let nvfp4 = LinearNvfp4::load(gpu.context())?;
    let stream = gpu.stream();
    let total_rows = 128usize;
    let k = 2304usize;
    let src_host = hash_values(0x51a7, total_rows * k);
    let src_dev = stream.clone_htod(&src_host)?;

    let ids: Vec<i32> = vec![127, 0, 42, 7, 19, 100, 3, 88, 55, 64, 11, 29];
    let split = 5usize;
    let mut gathered_host = Vec::with_capacity(ids.len() * k);
    for &id in &ids {
        let r = usize::try_from(id)?;
        gathered_host.extend_from_slice(&src_host[r * k..(r + 1) * k]);
    }
    let gathered_dev = stream.clone_htod(&gathered_host)?;
    let want = nvfp4.quantize_any(stream, &gathered_dev, ids.len(), k)?;

    let ids_dev = stream.clone_htod(&ids)?;
    let mut got = Nvfp4Matrix::alloc_zeros(stream, ids.len(), k)?;
    nvfp4.quantize_indexed(stream, &src_dev, &ids_dev.slice(..split), 0, &mut got)?;
    nvfp4.quantize_indexed(stream, &src_dev, &ids_dev.slice(split..), split, &mut got)?;

    assert_eq!(stream.clone_dtoh(&got.values())?, stream.clone_dtoh(&want.values())?);
    assert_eq!(stream.clone_dtoh(&got.scales())?, stream.clone_dtoh(&want.scales())?);
    assert_eq!(stream.clone_dtoh(&got.outer())?, stream.clone_dtoh(&want.outer())?);
    Ok(())
}
