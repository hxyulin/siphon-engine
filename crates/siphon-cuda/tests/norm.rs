//! Norm kernels against MLX outputs from `reference/fixtures/norm.py`.
//!
//! The kernels copy MLX's summation order, so every comparison requires bit-exact output.

#![cfg(target_os = "linux")]

mod common;

use common::{
    Fixture, assert_bf16_close, assert_f32_within_ulps, pseudo_random, pseudo_random_bf16,
};
use cudarc::driver::CudaSlice;
use half::bf16;
use siphon_cuda::{
    BLOCK_SCALED_MMA, Gpu,
    ops::{
        QuantDest, QuantFormat,
        attention::Attention,
        conv::Conv,
        elementwise::Elementwise,
        experts::{ExpertFormat, Experts, QuantizedRows},
        linear_fp8::{Fp8Gemm, Fp8GemmQuantized, Fp8Rows, LinearFp8},
        misc::{GateNormQuant, Misc},
        norm::{AttentionPost, AttentionPostRows, KvConvWeights, Norm},
        routing::{DepthRows, DepthRowsQuant, NormDest, QuantNormDest, Routing, RowsPartial},
    },
};

const EPS: f32 = 1e-6;
const QK_THRESHOLD: f32 = 22.627_417;
const REL_THRESHOLD: f32 = 8.0;

/// Distance between two F32 values in units in the last place, ordering negatives correctly.
fn f32_ulps(a: f32, b: f32) -> u64 {
    let key = |v: f32| {
        let bits = i64::from(v.to_bits());
        if bits & 0x8000_0000 == 0 {
            bits
        } else {
            0x8000_0000 - bits
        }
    };
    key(a).abs_diff(key(b))
}

/// Fails unless `got` equals `want` bit for bit.
fn assert_f32_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits(),
            "{label}[{i}]: got {g}, want {w} ({} ulps)",
            f32_ulps(g, w)
        );
    }
    eprintln!("{label}: all {} elements bit-exact", want.len());
}

struct Setup {
    fixture: Fixture,
    gpu: Gpu,
    norm: Norm,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        let norm = Norm::load(gpu.context())?;
        Ok(Self {
            fixture: Fixture::load("norm")?,
            gpu,
            norm,
        })
    }
}

#[test]
fn rms_norm_matches_mlx() -> anyhow::Result<()> {
    let Setup { fixture, gpu, norm } = Setup::new()?;
    let (x, _) = fixture.bf16("rms_norm.x")?;
    let (weight, _) = fixture.bf16("rms_norm.weight")?;
    let (want, _) = fixture.bf16("rms_norm.out")?;

    let stream = gpu.stream();
    let x_dev = stream.clone_htod(&x)?;
    let weight_dev = stream.clone_htod(&weight)?;
    let mut out_dev = stream.alloc_zeros::<bf16>(x.len())?;
    norm.rms_norm(stream, &x_dev, &weight_dev, &mut out_dev, EPS)?;
    let got = stream.clone_dtoh(&out_dev)?;

    assert_bf16_close("rms_norm", &got, &want, 0);
    Ok(())
}

#[test]
fn rms_norm_f32_matches_mlx() -> anyhow::Result<()> {
    let Setup { fixture, gpu, norm } = Setup::new()?;
    let (x, _) = fixture.f32("rms_norm_f32.x")?;
    let (weight, _) = fixture.bf16("rms_norm_f32.weight")?;
    let (want, _) = fixture.bf16("rms_norm_f32.out")?;

    let stream = gpu.stream();
    let x_dev = stream.clone_htod(&x)?;
    let weight_dev = stream.clone_htod(&weight)?;
    let mut out_dev = stream.alloc_zeros::<bf16>(x.len())?;
    norm.rms_norm_f32(stream, &x_dev, &weight_dev, &mut out_dev, EPS)?;
    let got = stream.clone_dtoh(&out_dev)?;

    assert_bf16_close("rms_norm_f32", &got, &want, 0);
    Ok(())
}

#[test]
fn unweighted_rms_norm_matches_mlx() -> anyhow::Result<()> {
    let Setup { fixture, gpu, norm } = Setup::new()?;
    let (x, _) = fixture.f32("unweighted_rms_norm.x")?;
    let (want, _) = fixture.f32("unweighted_rms_norm.out")?;

    let stream = gpu.stream();
    let x_dev = stream.clone_htod(&x)?;
    let mut out_dev = stream.alloc_zeros::<f32>(x.len())?;
    norm.unweighted_rms_norm(stream, &x_dev, &mut out_dev, EPS)?;
    let got = stream.clone_dtoh(&out_dev)?;

    assert_f32_exact("unweighted_rms_norm", &got, &want);
    Ok(())
}

#[test]
fn unweighted_rms_norm_heads_matches_mlx() -> anyhow::Result<()> {
    let Setup { fixture, gpu, norm } = Setup::new()?;
    let (x, _) = fixture.bf16("unweighted_rms_norm_heads.x")?;
    let (want, _) = fixture.bf16("unweighted_rms_norm_heads.out")?;

    let stream = gpu.stream();
    let x_dev = stream.clone_htod(&x)?;
    let mut out_dev = stream.alloc_zeros::<bf16>(x.len())?;
    norm.unweighted_rms_norm_heads(stream, &x_dev, &mut out_dev, EPS)?;
    let got = stream.clone_dtoh(&out_dev)?;

    assert_bf16_close("unweighted_rms_norm_heads", &got, &want, 0);
    Ok(())
}

/// Runs one L2 clip case from the fixture. `scaled` says whether the fixture has a scale tensor.
fn check_l2_clip(setup: &Setup, name: &str, scaled: bool, threshold: f32) -> anyhow::Result<()> {
    let Setup { fixture, gpu, norm } = setup;
    let (x, shape) = fixture.bf16(&format!("{name}.x"))?;
    let (want, _) = fixture.bf16(&format!("{name}.out"))?;
    let &[.., heads, head_dim] = shape.as_slice() else {
        anyhow::bail!("{name}.x has shape {shape:?}");
    };

    let stream = gpu.stream();
    let x_dev = stream.clone_htod(&x)?;
    let scale_dev = if scaled {
        let (scale, _) = fixture.bf16(&format!("{name}.scale"))?;
        Some(stream.clone_htod(&scale)?)
    } else {
        None
    };
    let mut out_dev = stream.alloc_zeros::<bf16>(x.len())?;
    norm.l2_clip(
        stream,
        &x_dev,
        scale_dev.as_ref(),
        &mut out_dev,
        heads,
        head_dim,
        threshold,
    )?;
    let got = stream.clone_dtoh(&out_dev)?;

    assert_bf16_close(name, &got, &want, 0);
    Ok(())
}

#[test]
fn l2_clip_matches_mlx() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    // Length 1 is decode; MLX sums fewer than 32 rows with a different kernel than prefill.
    for length in [1, 6] {
        check_l2_clip(&setup, &format!("q_l2_clip.{length}"), true, QK_THRESHOLD)?;
        check_l2_clip(&setup, &format!("k_l2_clip.{length}"), false, QK_THRESHOLD)?;
        check_l2_clip(
            &setup,
            &format!("rel_l2_clip.{length}"),
            true,
            REL_THRESHOLD,
        )?;
    }
    Ok(())
}

const HIDDEN: usize = 2304;
const KV: usize = 8 * 128;
const Q_HEADS: usize = 24;

/// The BF16-input unweighted norm against a cast followed by the F32 norm. Widening is
/// exact and the sums are the same, so the outputs must match bit for bit.
#[test]
fn unweighted_rms_norm_bf16_matches_cast_then_norm() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let norm = Norm::load(gpu.context())?;
    let misc = Misc::load(gpu.context())?;
    let x = stream.clone_htod(&pseudo_random_bf16(3 * HIDDEN, 7, 4.0, 0.0))?;
    let mut wide = stream.alloc_zeros::<f32>(3 * HIDDEN)?;
    let mut want = stream.alloc_zeros::<f32>(3 * HIDDEN)?;
    misc.cast_bf16_to_f32(stream, &x, &mut wide)?;
    norm.unweighted_rms_norm(stream, &wide, &mut want, EPS)?;
    let mut got = stream.alloc_zeros::<f32>(3 * HIDDEN)?;
    norm.unweighted_rms_norm_bf16(stream, &x, &mut got, EPS)?;
    assert_f32_within_ulps(
        "rows",
        &stream.clone_dtoh(&got)?,
        &stream.clone_dtoh(&want)?,
        0,
    );
    Ok(())
}

/// The fused attention post-processing against the separate kernels, for static and
/// dynamic convs, with and without log scaling, over two steps so the conv states carry.
///
/// The fused kernel repeats every rounded operation in order, so keys, queries,
/// relative-position states and conv states match bit for bit. The value norm computes
/// 1 / sqrt in float where `unweighted_rms_norm_heads` rounds a double rsqrt once; the two
/// differ by at most one F32 ulp, which moves a BF16 value by at most one ulp.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the separate kernels and the fused one, per case"
)]
fn attention_post_matches_separate_kernels() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let norm = Norm::load(gpu.context())?;
    let conv = Conv::load(gpu.context())?;
    let elementwise = Elementwise::load(gpu.context())?;
    let position = stream.clone_htod(&[20_000u32])?;
    let (alpha, n_floor) = (0.1, 8192.0);
    let q_scale = stream.clone_htod(&pseudo_random_bf16(Q_HEADS * 128, 60, 0.5, 1.0))?;
    let r_scale = stream.clone_htod(&pseudo_random_bf16(Q_HEADS * 16, 61, 0.5, 1.0))?;
    let k_weight = stream.clone_htod(&pseudo_random_bf16(KV * 4, 62, 0.5, 0.0))?;
    let v_weight = stream.clone_htod(&pseudo_random_bf16(KV * 4, 63, 0.5, 0.0))?;
    let k_static = stream.clone_htod(&pseudo_random(4 * KV, 64, 0.5))?;
    let v_static = stream.clone_htod(&pseudo_random(4 * KV, 65, 0.5))?;
    for (dynamic, log_scale) in [(false, false), (false, true), (true, false), (true, true)] {
        let label = format!("dynamic {dynamic}, log scale {log_scale}");
        let start_k = pseudo_random(3 * KV, 66, 2.0);
        let start_v = pseudo_random(3 * KV, 67, 2.0);
        let (mut k_state, mut v_state) =
            (stream.clone_htod(&start_k)?, stream.clone_htod(&start_v)?);
        let (mut k_state_ref, mut v_state_ref) =
            (stream.clone_htod(&start_k)?, stream.clone_htod(&start_v)?);
        for step in 0..2u32 {
            let seed = 100 * step;
            // Scales chosen so some heads exceed the clip thresholds and some do not.
            let k_raw = stream.clone_htod(&pseudo_random_bf16(KV, seed + 1, 6.0, 0.0))?;
            let v_raw = stream.clone_htod(&pseudo_random_bf16(KV, seed + 2, 6.0, 0.0))?;
            let k_dyn = stream.clone_htod(&pseudo_random_bf16(32, seed + 3, 0.5, 0.0))?;
            let v_dyn = stream.clone_htod(&pseudo_random_bf16(32, seed + 4, 0.5, 0.0))?;
            let q_raw =
                stream.clone_htod(&pseudo_random_bf16(Q_HEADS * 128, seed + 5, 3.0, 0.0))?;
            let r_raw = stream.clone_htod(&pseudo_random_bf16(Q_HEADS * 16, seed + 6, 3.0, 0.0))?;

            let zeros =
                |n: usize| -> anyhow::Result<CudaSlice<bf16>> { Ok(stream.alloc_zeros(n)?) };
            let (mut k_conv, mut v_conv) = (zeros(KV)?, zeros(KV)?);
            if dynamic {
                conv.dynamic_short_conv(
                    stream,
                    &k_raw,
                    &k_dyn,
                    &k_static,
                    &mut k_state_ref,
                    &mut k_conv,
                )?;
                conv.dynamic_short_conv(
                    stream,
                    &v_raw,
                    &v_dyn,
                    &v_static,
                    &mut v_state_ref,
                    &mut v_conv,
                )?;
            } else {
                conv.short_conv(stream, &k_raw, &k_weight, &mut k_state_ref, &mut k_conv)?;
                conv.short_conv(stream, &v_raw, &v_weight, &mut v_state_ref, &mut v_conv)?;
            }
            let (mut k_want, mut v_want) = (zeros(KV)?, zeros(KV)?);
            let (mut q_want, mut r_want) = (zeros(Q_HEADS * 128)?, zeros(Q_HEADS * 16)?);
            norm.l2_clip(stream, &k_conv, None, &mut k_want, 8, 128, QK_THRESHOLD)?;
            norm.unweighted_rms_norm_heads(stream, &v_conv, &mut v_want, EPS)?;
            norm.l2_clip(
                stream,
                &q_raw,
                Some(&q_scale),
                &mut q_want,
                Q_HEADS,
                128,
                QK_THRESHOLD,
            )?;
            norm.l2_clip(
                stream,
                &r_raw,
                Some(&r_scale),
                &mut r_want,
                Q_HEADS,
                16,
                REL_THRESHOLD,
            )?;
            if log_scale {
                elementwise.log_scale(
                    stream,
                    &mut q_want,
                    Q_HEADS * 128,
                    &position,
                    alpha,
                    n_floor,
                )?;
                elementwise.log_scale(
                    stream,
                    &mut r_want,
                    Q_HEADS * 16,
                    &position,
                    alpha,
                    n_floor,
                )?;
            }

            let (mut k, mut v) = (zeros(KV)?, zeros(KV)?);
            let (mut q, mut r) = (zeros(Q_HEADS * 128)?, zeros(Q_HEADS * 16)?);
            let conv_weights = if dynamic {
                KvConvWeights::Dynamic {
                    k_dynamic: k_dyn.as_view(),
                    v_dynamic: v_dyn.as_view(),
                    k_static: &k_static,
                    v_static: &v_static,
                }
            } else {
                KvConvWeights::Static {
                    k: &k_weight,
                    v: &v_weight,
                }
            };
            norm.attention_post(
                stream,
                AttentionPost {
                    k_raw: &k_raw,
                    v_raw: &v_raw,
                    conv: conv_weights,
                    k_state: &mut k_state,
                    v_state: &mut v_state,
                    q_raw: &q_raw,
                    q_scale: &q_scale,
                    r_raw: &r_raw,
                    r_scale: &r_scale,
                    k: &mut k,
                    v: &mut v,
                    q: &mut q,
                    r: &mut r,
                    log_scale: log_scale.then_some((&position, alpha, n_floor)),
                    qk_threshold: QK_THRESHOLD,
                    rel_threshold: REL_THRESHOLD,
                    eps: EPS,
                },
            )?;
            let label = format!("{label}, step {step}");
            for (name, got, want, ulps) in [
                ("k", &k, &k_want, 0),
                ("v", &v, &v_want, 1),
                ("q", &q, &q_want, 0),
                ("r", &r, &r_want, 0),
            ] {
                let got = stream.clone_dtoh(got)?;
                assert!(
                    got.iter().any(|v| v.to_f32() != 0.0),
                    "{label}: {name} is all zero"
                );
                assert_bf16_close(
                    &format!("{label}: {name}"),
                    &got,
                    &stream.clone_dtoh(want)?,
                    ulps,
                );
            }
            for (name, got, want) in [
                ("k state", &k_state, &k_state_ref),
                ("v state", &v_state, &v_state_ref),
            ] {
                assert_f32_within_ulps(
                    &format!("{label}: {name}"),
                    &stream.clone_dtoh(got)?,
                    &stream.clone_dtoh(want)?,
                    0,
                );
            }
        }
    }
    Ok(())
}

/// The multi-row attention post-processing against one single-row call per row, over
/// chunks of 5, 1, 2 and 7 rows with carried conv states, for static and dynamic convs,
/// with and without log scaling, and its relative bias against
/// `Attention::relative_bias_rows` of its relative states. Same rounded operations, so bit
/// for bit.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the single-row and multi-row calls per chunk"
)]
fn attention_post_rows_match_single_rows() -> anyhow::Result<()> {
    const CHUNKS: [usize; 4] = [5, 1, 2, 7];
    // Columns of the relative bias table.
    const EXTENT: usize = 64;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let norm = Norm::load(gpu.context())?;
    let conv = Conv::load(gpu.context())?;
    let attention = Attention::load(gpu.context())?;
    let table = stream.clone_htod(&pseudo_random_bf16(16 * EXTENT, 78, 1.0, 0.0))?;
    let (alpha, n_floor) = (0.1, 8192.0);
    let q_scale = stream.clone_htod(&pseudo_random_bf16(Q_HEADS * 128, 70, 0.5, 1.0))?;
    let r_scale = stream.clone_htod(&pseudo_random_bf16(Q_HEADS * 16, 71, 0.5, 1.0))?;
    let k_weight = stream.clone_htod(&pseudo_random_bf16(KV * 4, 72, 0.5, 0.0))?;
    let v_weight = stream.clone_htod(&pseudo_random_bf16(KV * 4, 73, 0.5, 0.0))?;
    let k_static = stream.clone_htod(&pseudo_random(4 * KV, 74, 0.5))?;
    let v_static = stream.clone_htod(&pseudo_random(4 * KV, 75, 0.5))?;
    for (dynamic, log_scale) in [(false, false), (false, true), (true, false), (true, true)] {
        let start_k = pseudo_random(3 * KV, 76, 2.0);
        let start_v = pseudo_random(3 * KV, 77, 2.0);
        let (mut k_state, mut v_state) =
            (stream.clone_htod(&start_k)?, stream.clone_htod(&start_v)?);
        let (mut k_state_ref, mut v_state_ref) =
            (stream.clone_htod(&start_k)?, stream.clone_htod(&start_v)?);
        // Positions around the log-scaling floor, so tau changes within a chunk.
        let mut position = 8185u32;
        for (chunk, &rows) in CHUNKS.iter().enumerate() {
            let label = format!("dynamic {dynamic}, log scale {log_scale}, chunk {chunk}");
            let seed = 200 + 10 * chunk as u32;
            let k_raw = pseudo_random_bf16(rows * KV, seed + 1, 6.0, 0.0);
            let v_raw = pseudo_random_bf16(rows * KV, seed + 2, 6.0, 0.0);
            let k_dyn = pseudo_random_bf16(rows * 32, seed + 3, 0.5, 0.0);
            let v_dyn = pseudo_random_bf16(rows * 32, seed + 4, 0.5, 0.0);
            let q_raw = pseudo_random_bf16(rows * Q_HEADS * 128, seed + 5, 3.0, 0.0);
            let r_raw = pseudo_random_bf16(rows * Q_HEADS * 16, seed + 6, 3.0, 0.0);

            let mut want: [Vec<bf16>; 4] = Default::default();
            for r in 0..rows {
                let part =
                    |x: &[bf16], width: usize| stream.clone_htod(&x[r * width..(r + 1) * width]);
                let (k_raw, v_raw) = (part(&k_raw, KV)?, part(&v_raw, KV)?);
                let (k_dyn, v_dyn) = (part(&k_dyn, 32)?, part(&v_dyn, 32)?);
                let (q_raw, r_raw) = (part(&q_raw, Q_HEADS * 128)?, part(&r_raw, Q_HEADS * 16)?);
                let at = stream.clone_htod(&[position + r as u32])?;
                let (mut k, mut v) = (stream.alloc_zeros(KV)?, stream.alloc_zeros(KV)?);
                let mut q = stream.alloc_zeros(Q_HEADS * 128)?;
                let mut rel = stream.alloc_zeros(Q_HEADS * 16)?;
                let conv_weights = if dynamic {
                    KvConvWeights::Dynamic {
                        k_dynamic: k_dyn.as_view(),
                        v_dynamic: v_dyn.as_view(),
                        k_static: &k_static,
                        v_static: &v_static,
                    }
                } else {
                    KvConvWeights::Static {
                        k: &k_weight,
                        v: &v_weight,
                    }
                };
                norm.attention_post(
                    stream,
                    AttentionPost {
                        k_raw: &k_raw,
                        v_raw: &v_raw,
                        conv: conv_weights,
                        k_state: &mut k_state_ref,
                        v_state: &mut v_state_ref,
                        q_raw: &q_raw,
                        q_scale: &q_scale,
                        r_raw: &r_raw,
                        r_scale: &r_scale,
                        k: &mut k,
                        v: &mut v,
                        q: &mut q,
                        r: &mut rel,
                        log_scale: log_scale.then_some((&at, alpha, n_floor)),
                        qk_threshold: QK_THRESHOLD,
                        rel_threshold: REL_THRESHOLD,
                        eps: EPS,
                    },
                )?;
                for (w, x) in want.iter_mut().zip([&k, &v, &q, &rel]) {
                    w.extend(stream.clone_dtoh(x)?);
                }
            }

            let (k_raw, v_raw) = (stream.clone_htod(&k_raw)?, stream.clone_htod(&v_raw)?);
            let (k_dyn, v_dyn) = (stream.clone_htod(&k_dyn)?, stream.clone_htod(&v_dyn)?);
            let (q_raw, r_raw) = (stream.clone_htod(&q_raw)?, stream.clone_htod(&r_raw)?);
            let start = stream.clone_htod(&[position])?;
            let mut k: CudaSlice<bf16> = stream.alloc_zeros(rows * KV)?;
            let mut v: CudaSlice<bf16> = stream.alloc_zeros(rows * KV)?;
            let mut q: CudaSlice<bf16> = stream.alloc_zeros(rows * Q_HEADS * 128)?;
            let mut rel: CudaSlice<bf16> = stream.alloc_zeros(rows * Q_HEADS * 16)?;
            let mut bias: CudaSlice<bf16> = stream.alloc_zeros(rows * Q_HEADS * EXTENT)?;
            let conv_weights = if dynamic {
                KvConvWeights::Dynamic {
                    k_dynamic: k_dyn.as_view(),
                    v_dynamic: v_dyn.as_view(),
                    k_static: &k_static,
                    v_static: &v_static,
                }
            } else {
                KvConvWeights::Static {
                    k: &k_weight,
                    v: &v_weight,
                }
            };
            norm.attention_post_rows(
                stream,
                AttentionPostRows {
                    k_raw: k_raw.as_view(),
                    v_raw: v_raw.as_view(),
                    conv: conv_weights,
                    k_state: &k_state,
                    v_state: &v_state,
                    q_raw: q_raw.as_view(),
                    q_scale: &q_scale,
                    r_raw: r_raw.as_view(),
                    r_scale: &r_scale,
                    k: k.as_view_mut(),
                    v: v.as_view_mut(),
                    q: q.as_view_mut(),
                    r: rel.as_view_mut(),
                    log_scale: log_scale.then_some((start.as_view(), alpha, n_floor)),
                    qk_threshold: QK_THRESHOLD,
                    rel_threshold: REL_THRESHOLD,
                    eps: EPS,
                    bias: Some((&table, bias.as_view_mut())),
                    batch: None,
                },
                rows,
            )?;
            let mut bias_ref: CudaSlice<bf16> = stream.alloc_zeros(rows * Q_HEADS * EXTENT)?;
            attention.relative_bias_rows(stream, &rel, &table, &mut bias_ref)?;
            assert_bf16_close(
                &format!("{label}: bias"),
                &stream.clone_dtoh(&bias)?,
                &stream.clone_dtoh(&bias_ref)?,
                0,
            );
            conv.shift_state(stream, &k_raw, &mut k_state, KV, rows)?;
            conv.shift_state(stream, &v_raw, &mut v_state, KV, rows)?;
            for (name, got, want) in [
                ("k", &k, &want[0]),
                ("v", &v, &want[1]),
                ("q", &q, &want[2]),
                ("r", &rel, &want[3]),
            ] {
                assert_bf16_close(
                    &format!("{label}: {name}"),
                    &stream.clone_dtoh(got)?,
                    want,
                    0,
                );
            }
            for (name, got, want) in [
                ("k state", &k_state, &k_state_ref),
                ("v state", &v_state, &v_state_ref),
            ] {
                assert_f32_within_ulps(
                    &format!("{label}: {name}"),
                    &stream.clone_dtoh(got)?,
                    &stream.clone_dtoh(want)?,
                    0,
                );
            }
            position += rows as u32;
        }
    }
    Ok(())
}

/// Runs the standalone quantizer matching `format` on BF16 `x` (`[rows, HIDDEN]`) into `out`.
fn quantize_reference(
    stream: &cudarc::driver::CudaStream,
    experts: &Experts,
    x: &CudaSlice<bf16>,
    format: QuantFormat,
    dummy_global_scale: &CudaSlice<f32>,
    out: &mut QuantizedRows,
) -> anyhow::Result<()> {
    match format {
        QuantFormat::Mxfp8 => experts.quantize_rows(stream, x, HIDDEN, ExpertFormat::Mxfp8, out),
        QuantFormat::Mxfp8Paired => experts.quantize_rows_fp8(
            stream,
            x,
            HIDDEN,
            ExpertFormat::Nvfp4 {
                global_scale: dummy_global_scale,
                scale_base: None,
            },
            out,
        ),
        QuantFormat::Nvfp4 => experts.quantize_rows(
            stream,
            x,
            HIDDEN,
            ExpertFormat::Nvfp4 {
                global_scale: dummy_global_scale,
                scale_base: None,
            },
            out,
        ),
    }
}

/// Asserts that `got` matches `want` bit for bit for `rows` rows of `HIDDEN` in `format`.
fn assert_quant_exact(
    label: &str,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    got: &QuantizedRows,
    want: &QuantizedRows,
    rows: usize,
    format: QuantFormat,
) -> anyhow::Result<()> {
    let v_len = rows * format.value_bytes(HIDDEN);
    let s_len = rows * format.scale_bytes(HIDDEN);
    assert_eq!(
        stream.clone_dtoh(&got.values().slice(..v_len))?,
        stream.clone_dtoh(&want.values().slice(..v_len))?,
        "{label}: quantized values"
    );
    assert_eq!(
        stream.clone_dtoh(&got.scales().slice(..s_len))?,
        stream.clone_dtoh(&want.scales().slice(..s_len))?,
        "{label}: quantized scales"
    );
    if format == QuantFormat::Nvfp4 {
        let g = stream.clone_dtoh(&got.outer().slice(..rows))?;
        let w = stream.clone_dtoh(&want.outer().slice(..rows))?;
        assert_f32_exact(&format!("{label}: outer scale"), &g, &w);
    }
    Ok(())
}

/// Fused `rms_norm_quant` (BF16 input) and `rms_norm_f32_quant` (F32 input) match unfused
/// `rms_norm`/`rms_norm_f32` + standalone quantization bit for bit across MXFP8, paired
/// MXFP8, and (on Blackwell) NVFP4, both with and without BF16 output, and feed
/// `LinearFp8::gemm_quantized` identically to `LinearFp8::gemm`.
#[test]
#[expect(clippy::too_many_lines, reason = "tests BF16/F32 inputs, all formats, and FP8 GEMM")]
fn rms_norm_quant_matches_separate() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let norm = Norm::load(gpu.context())?;
    let experts = Experts::load(gpu.context())?;
    let linear_fp8 = LinearFp8::load(gpu.context())?;
    let weight = stream.clone_htod(&pseudo_random_bf16(HIDDEN, 801, 0.5, 1.0))?;
    let dummy_scale = stream.alloc_zeros::<f32>(1)?;
    let formats: Vec<QuantFormat> = if BLOCK_SCALED_MMA {
        vec![
            QuantFormat::Mxfp8,
            QuantFormat::Mxfp8Paired,
            QuantFormat::Nvfp4,
        ]
    } else {
        vec![QuantFormat::Mxfp8, QuantFormat::Mxfp8Paired]
    };

    for rows in [1usize, 17, 64] {
        let seed_off = u32::try_from(rows)?;
        let x_bf16 = stream.clone_htod(&pseudo_random_bf16(rows * HIDDEN, 810 + seed_off, 4.0, 0.0))?;
        let x_f32 = stream.clone_htod(&pseudo_random(rows * HIDDEN, 820 + seed_off, 4.0))?;
        let mut ref_bf16 = stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
        let mut ref_f32 = stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
        norm.rms_norm(stream, &x_bf16, &weight, &mut ref_bf16, EPS)?;
        norm.rms_norm_f32(stream, &x_f32, &weight, &mut ref_f32, EPS)?;
        let ref_bf16_host = stream.clone_dtoh(&ref_bf16)?;
        let ref_f32_host = stream.clone_dtoh(&ref_f32)?;

        let mut want_q = QuantizedRows::new(stream, rows * HIDDEN, rows)?;
        let mut got_q = QuantizedRows::new(stream, rows * HIDDEN, rows)?;
        for &format in &formats {
            quantize_reference(stream, &experts, &ref_bf16, format, &dummy_scale, &mut want_q)?;
            for write_bf16 in [true, false] {
                let mut got_bf16 = stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
                let dest = got_q.as_quant_dest(rows, HIDDEN, format)?;
                norm.rms_norm_quant(
                    stream,
                    &x_bf16,
                    &weight,
                    write_bf16.then(|| got_bf16.as_view_mut()),
                    dest,
                    EPS,
                )?;
                let label = format!("rms_norm_quant rows={rows} {format:?} bf16={write_bf16}");
                if write_bf16 {
                    assert_bf16_close(&label, &stream.clone_dtoh(&got_bf16)?, &ref_bf16_host, 0);
                }
                assert_quant_exact(&label, stream, &got_q, &want_q, rows, format)?;
            }

            quantize_reference(stream, &experts, &ref_f32, format, &dummy_scale, &mut want_q)?;
            for write_bf16 in [true, false] {
                let mut got_bf16 = stream.alloc_zeros::<bf16>(rows * HIDDEN)?;
                let dest = got_q.as_quant_dest(rows, HIDDEN, format)?;
                norm.rms_norm_f32_quant(
                    stream,
                    &x_f32,
                    &weight,
                    write_bf16.then(|| got_bf16.as_view_mut()),
                    dest,
                    EPS,
                )?;
                let label = format!("rms_norm_f32_quant rows={rows} {format:?} bf16={write_bf16}");
                if write_bf16 {
                    assert_bf16_close(&label, &stream.clone_dtoh(&got_bf16)?, &ref_f32_host, 0);
                }
                assert_quant_exact(&label, stream, &got_q, &want_q, rows, format)?;
            }
        }
    }

    // End-to-end check: `rms_norm_quant` -> `LinearFp8::gemm_quantized` matches
    // `rms_norm` -> `LinearFp8::gemm` bit for bit.
    if BLOCK_SCALED_MMA {
        const M: usize = 35;
        const N: usize = 256;
        let x = stream.clone_htod(&pseudo_random_bf16(M * HIDDEN, 850, 3.0, 0.0))?;
        let w_bf16 = stream.clone_htod(&pseudo_random_bf16(N * HIDDEN, 851, 0.5, 0.0))?;
        let w_fp8 = linear_fp8.quantize(stream, &w_bf16, N, HIDDEN)?;
        let mut normed = stream.alloc_zeros::<bf16>(M * HIDDEN)?;
        norm.rms_norm(stream, &x, &weight, &mut normed, EPS)?;
        let mut scratch_ref = Fp8Rows::new(stream, M * HIDDEN)?;
        let mut want_y = stream.alloc_zeros::<bf16>(M * N)?;
        linear_fp8.gemm(
            stream,
            &mut [Fp8Gemm {
                weight: &w_fp8,
                x: normed.as_view(),
                out: want_y.as_view_mut(),
            }],
            M,
            &mut scratch_ref,
        )?;

        let mut scratch_got = Fp8Rows::new(stream, M * HIDDEN)?;
        {
            let [(values, scales)]: [_; 1] = scratch_got
                .split_slots_mut(M, HIDDEN, 1)?
                .try_into()
                .map_err(|_| anyhow::anyhow!("expected 1 dense slot"))?;
            norm.rms_norm_quant(
                stream,
                &x,
                &weight,
                None,
                QuantDest {
                    format: QuantFormat::Mxfp8,
                    values,
                    scales,
                    outer: None,
                },
                EPS,
            )?;
        }
        let mut got_y = stream.alloc_zeros::<bf16>(M * N)?;
        linear_fp8.gemm_quantized(
            stream,
            &mut [Fp8GemmQuantized {
                weight: &w_fp8,
                x_offset: 0,
                out: got_y.as_view_mut(),
            }],
            M,
            &scratch_got,
        )?;
        assert_bf16_close(
            "rms_norm_quant -> gemm_quantized",
            &stream.clone_dtoh(&got_y)?,
            &stream.clone_dtoh(&want_y)?,
            0,
        );
    }
    Ok(())
}

/// Fused `Routing::depth_attention_rows_quant` (1 and 3 queries) and
/// `Misc::message_gate_out_rows_quant` match unfused norm + standalone quantization bit for
/// bit across MXFP8, paired MXFP8, and (on Blackwell) NVFP4.
#[test]
#[expect(clippy::too_many_lines, reason = "tests depth attention (nq=1,3) and message gate")]
fn fused_depth_and_gate_norm_quant_match_separate() -> anyhow::Result<()> {
    const ROWS: usize = 35;
    const NB: usize = 2;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let routing = Routing::load(gpu.context())?;
    let misc = Misc::load(gpu.context())?;
    let experts = Experts::load(gpu.context())?;
    let dummy_scale = stream.alloc_zeros::<f32>(1)?;
    let formats: Vec<QuantFormat> = if BLOCK_SCALED_MMA {
        vec![
            QuantFormat::Mxfp8,
            QuantFormat::Mxfp8Paired,
            QuantFormat::Nvfp4,
        ]
    } else {
        vec![QuantFormat::Mxfp8, QuantFormat::Mxfp8Paired]
    };

    let blocks = stream.clone_htod(&pseudo_random(NB * ROWS * HIDDEN, 900, 1.5))?;
    let partial_bf16 = stream.clone_htod(&pseudo_random_bf16(ROWS * HIDDEN, 901, 2.0, 0.0))?;
    let q1 = stream.clone_htod(&pseudo_random(HIDDEN, 902, 1.0))?;
    let q3 = stream.clone_htod(&pseudo_random(3 * HIDDEN, 903, 1.0))?;
    let norm_weights: Vec<CudaSlice<bf16>> = (0..3u32)
        .map(|k| {
            stream
                .clone_htod(&pseudo_random_bf16(HIDDEN, 910 + k, 0.5, 1.0))
                .map_err(anyhow::Error::from)
        })
        .collect::<anyhow::Result<_>>()?;

    // 1. depth_attention_rows_quant (nq = 1): all formats, with and without BF16 output.
    let mut ref_bf16 = stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?;
    routing.depth_attention_rows(
        stream,
        DepthRows {
            blocks: blocks.as_view(),
            partial: RowsPartial::Bf16(partial_bf16.as_view()),
            normed_partial: None,
            queries: &q1,
            out: None,
            probs: None,
            dests: [
                Some(NormDest {
                    out: ref_bf16.as_view_mut(),
                    weight: Some(&norm_weights[0]),
                }),
                None,
                None,
                None,
            ],
            dst_stride: HIDDEN,
            rows: ROWS,
        },
        EPS,
    )?;
    let ref_bf16_host = stream.clone_dtoh(&ref_bf16)?;
    let mut want_q = QuantizedRows::new(stream, ROWS * HIDDEN, ROWS)?;
    let mut got_q = QuantizedRows::new(stream, ROWS * HIDDEN, ROWS)?;
    for &format in &formats {
        quantize_reference(stream, &experts, &ref_bf16, format, &dummy_scale, &mut want_q)?;
        for write_bf16 in [true, false] {
            let mut got_bf16 = stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?;
            let quant = got_q.as_quant_dest(ROWS, HIDDEN, format)?;
            routing.depth_attention_rows_quant(
                stream,
                DepthRowsQuant {
                    blocks: blocks.as_view(),
                    partial: RowsPartial::Bf16(partial_bf16.as_view()),
                    normed_partial: None,
                    queries: &q1,
                    out: None,
                    probs: None,
                    dests: [
                        Some(QuantNormDest {
                            out: write_bf16.then(|| got_bf16.as_view_mut()),
                            weight: Some(&norm_weights[0]),
                            quant,
                        }),
                        None,
                        None,
                        None,
                    ],
                    dst_stride: HIDDEN,
                    quant_col_offset: 0,
                    rows: ROWS,
                },
                EPS,
            )?;
            let label = format!("depth_attention_rows_quant nq=1 {format:?} bf16={write_bf16}");
            if write_bf16 {
                assert_bf16_close(&label, &stream.clone_dtoh(&got_bf16)?, &ref_bf16_host, 0);
            }
            assert_quant_exact(&label, stream, &got_q, &want_q, ROWS, format)?;
        }
    }

    // 2. depth_attention_rows_quant (nq = 3): 3 MXFP8 destinations into Fp8Rows slots.
    let mut ref_d0 = stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?;
    let mut ref_d1 = stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?;
    let mut ref_d2 = stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?;
    routing.depth_attention_rows(
        stream,
        DepthRows {
            blocks: blocks.as_view(),
            partial: RowsPartial::Bf16(partial_bf16.as_view()),
            normed_partial: None,
            queries: &q3,
            out: None,
            probs: None,
            dests: [
                Some(NormDest {
                    out: ref_d0.as_view_mut(),
                    weight: Some(&norm_weights[0]),
                }),
                Some(NormDest {
                    out: ref_d1.as_view_mut(),
                    weight: Some(&norm_weights[1]),
                }),
                Some(NormDest {
                    out: ref_d2.as_view_mut(),
                    weight: Some(&norm_weights[2]),
                }),
                None,
            ],
            dst_stride: HIDDEN,
            rows: ROWS,
        },
        EPS,
    )?;
    let mut fp8_rows = Fp8Rows::new(stream, 3 * ROWS * HIDDEN)?;
    {
        let [(v0, s0), (v1, s1), (v2, s2)]: [_; 3] = fp8_rows
            .split_slots_mut(ROWS, HIDDEN, 3)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected 3 dense slots"))?;
        routing.depth_attention_rows_quant(
            stream,
            DepthRowsQuant {
                blocks: blocks.as_view(),
                partial: RowsPartial::Bf16(partial_bf16.as_view()),
                normed_partial: None,
                queries: &q3,
                out: None,
                probs: None,
                dests: [
                    Some(QuantNormDest {
                        out: None,
                        weight: Some(&norm_weights[0]),
                        quant: QuantDest {
                            format: QuantFormat::Mxfp8,
                            values: v0,
                            scales: s0,
                            outer: None,
                        },
                    }),
                    Some(QuantNormDest {
                        out: None,
                        weight: Some(&norm_weights[1]),
                        quant: QuantDest {
                            format: QuantFormat::Mxfp8,
                            values: v1,
                            scales: s1,
                            outer: None,
                        },
                    }),
                    Some(QuantNormDest {
                        out: None,
                        weight: Some(&norm_weights[2]),
                        quant: QuantDest {
                            format: QuantFormat::Mxfp8,
                            values: v2,
                            scales: s2,
                            outer: None,
                        },
                    }),
                    None,
                ],
                dst_stride: HIDDEN,
                quant_col_offset: 0,
                rows: ROWS,
            },
            EPS,
        )?;
    }
    let slot_vals = ROWS * HIDDEN;
    let slot_scales = slot_vals / 32;
    for (k, ref_d) in [&ref_d0, &ref_d1, &ref_d2].into_iter().enumerate() {
        quantize_reference(
            stream,
            &experts,
            ref_d,
            QuantFormat::Mxfp8,
            &dummy_scale,
            &mut want_q,
        )?;
        assert_eq!(
            stream.clone_dtoh(&fp8_rows.values().slice(k * slot_vals..(k + 1) * slot_vals))?,
            stream.clone_dtoh(&want_q.values().slice(..slot_vals))?,
            "depth_attention_rows_quant nq=3 slot {k}: values"
        );
        assert_eq!(
            stream.clone_dtoh(
                &fp8_rows
                    .scales()
                    .slice(k * slot_scales..(k + 1) * slot_scales)
            )?,
            stream.clone_dtoh(&want_q.scales().slice(..slot_scales))?,
            "depth_attention_rows_quant nq=3 slot {k}: scales"
        );
    }

    // 3. message_gate_out_rows_quant: all formats, with and without BF16 normed output.
    let down = stream.clone_htod(&pseudo_random_bf16(ROWS * 64, 920, 2.0, 0.0))?;
    let forget = stream.clone_htod(&pseudo_random_bf16(HIDDEN * 64, 921, 0.3, 0.0))?;
    let gain = stream.clone_htod(&pseudo_random_bf16(HIDDEN * 64, 922, 0.3, 0.0))?;
    let h = stream.clone_htod(&pseudo_random(ROWS * HIDDEN, 923, 3.0))?;
    let m_vec = stream.clone_htod(&pseudo_random(ROWS * HIDDEN, 924, 1.5))?;
    let mut gate_out_ref = stream.alloc_zeros::<f32>(ROWS * HIDDEN)?;
    let mut gate_norm_ref = stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?;
    misc.message_gate_out_rows(
        stream,
        &down,
        &forget,
        &gain,
        (&h, HIDDEN),
        (&m_vec, HIDDEN),
        None,
        &mut gate_out_ref,
        Some((&norm_weights[0], &mut gate_norm_ref)),
        EPS,
        ROWS,
    )?;
    let gate_out_ref_host = stream.clone_dtoh(&gate_out_ref)?;
    let gate_norm_ref_host = stream.clone_dtoh(&gate_norm_ref)?;
    for &format in &formats {
        quantize_reference(
            stream,
            &experts,
            &gate_norm_ref,
            format,
            &dummy_scale,
            &mut want_q,
        )?;
        for write_bf16 in [true, false] {
            let mut gate_out = stream.alloc_zeros::<f32>(ROWS * HIDDEN)?;
            let mut gate_norm = stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?;
            let quant = got_q.as_quant_dest(ROWS, HIDDEN, format)?;
            misc.message_gate_out_rows_quant(
                stream,
                &down,
                &forget,
                &gain,
                (&h, HIDDEN),
                (&m_vec, HIDDEN),
                None,
                &mut gate_out,
                GateNormQuant {
                    weight: &norm_weights[0],
                    normed: write_bf16.then(|| gate_norm.as_view_mut()),
                    quant,
                },
                EPS,
                ROWS,
            )?;
            let label = format!("message_gate_out_rows_quant {format:?} bf16={write_bf16}");
            assert_f32_exact(&format!("{label}: out"), &stream.clone_dtoh(&gate_out)?, &gate_out_ref_host);
            if write_bf16 {
                assert_bf16_close(
                    &format!("{label}: normed"),
                    &stream.clone_dtoh(&gate_norm)?,
                    &gate_norm_ref_host,
                    0,
                );
            }
            assert_quant_exact(&label, stream, &got_q, &want_q, ROWS, format)?;
        }
    }

    // 4. message_gate_out_rows_3 (3-slot batched attention-side message gates): matches
    //    3 sequential calls of message_gate_out_rows + standalone quant BIT FOR BIT for
    //    both Shared (sliding) and InPlace (global) buffer layouts, BF16 and MXFP8.
    {
        use siphon_cuda::ops::misc::{GateOut3Buffers, GateOut3Slot, MessageMix};
        const NB: usize = 3;
        let blocks = stream.clone_htod(&pseudo_random(NB * ROWS * HIDDEN, 930, 1.5))?;
        let probs_stride = 4 * (NB + 1);
        let probs_host: Vec<f32> = (0..ROWS * probs_stride)
            .map(|i| 0.1 + ((i * 37) % 17) as f32 * 0.05)
            .collect();
        let probs = stream.clone_htod(&probs_host)?;
        let downs3 = [
            stream.clone_htod(&pseudo_random_bf16(ROWS * 64, 940, 2.0, 0.0))?,
            stream.clone_htod(&pseudo_random_bf16(ROWS * 64, 941, 2.0, 0.0))?,
            stream.clone_htod(&pseudo_random_bf16(ROWS * 64, 942, 2.0, 0.0))?,
        ];
        let forgets3 = [
            stream.clone_htod(&pseudo_random_bf16(HIDDEN * 64, 943, 0.3, 0.0))?,
            stream.clone_htod(&pseudo_random_bf16(HIDDEN * 64, 944, 0.3, 0.0))?,
            stream.clone_htod(&pseudo_random_bf16(HIDDEN * 64, 945, 0.3, 0.0))?,
        ];
        let gains3 = [
            stream.clone_htod(&pseudo_random_bf16(HIDDEN * 64, 946, 0.3, 0.0))?,
            stream.clone_htod(&pseudo_random_bf16(HIDDEN * 64, 947, 0.3, 0.0))?,
            stream.clone_htod(&pseudo_random_bf16(HIDDEN * 64, 948, 0.3, 0.0))?,
        ];
        let h3_host = pseudo_random(ROWS * 3 * HIDDEN, 949, 2.5);
        for h_shared in [true, false] {
            let (h_stride, h_slot_stride) = if h_shared {
                (HIDDEN, 0)
            } else {
                (3 * HIDDEN, HIDDEN)
            };
            let h_ref = if h_shared {
                h.clone()
            } else {
                stream.clone_htod(&h3_host)?
            };
            let mut ref_out = stream.alloc_zeros::<f32>(ROWS * HIDDEN)?;
            let mut ref_norms = [
                stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?,
                stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?,
                stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?,
            ];
            for k in 0..3 {
                misc.message_gate_out_rows(
                    stream,
                    &downs3[k],
                    &forgets3[k],
                    &gains3[k],
                    (&h_ref.slice(k * h_slot_stride..), h_stride),
                    (&m_vec, HIDDEN),
                    Some(&MessageMix {
                        blocks: blocks.as_view(),
                        probs: probs.slice(k * (NB + 1)..),
                        probs_stride,
                    }),
                    &mut ref_out,
                    Some((&norm_weights[k], &mut ref_norms[k])),
                    EPS,
                    ROWS,
                )?;
            }

            let mut batched_norms = [
                stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?,
                stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?,
                stream.alloc_zeros::<bf16>(ROWS * HIDDEN)?,
            ];
            let mut out_scratch = stream.alloc_zeros::<f32>(ROWS * 3 * HIDDEN)?;
            let mut in_place_buf = stream.clone_htod(&h3_host)?;
            let (o0, mut rest) = out_scratch.split_at_mut(ROWS * HIDDEN);
            let (o1, o2) = rest.split_at_mut(ROWS * HIDDEN);
            let mut buffers = if h_shared {
                GateOut3Buffers::Shared {
                    h: h.as_view(),
                    out: [o0, o1, o2],
                }
            } else {
                GateOut3Buffers::InPlace {
                    h_and_out: in_place_buf.as_view_mut(),
                }
            };
            let q_slots = fp8_rows.split_slots_mut(ROWS, HIDDEN, 3)?;
            let mut q_iter = q_slots.into_iter();
            let [n0, n1, n2] = &mut batched_norms;
            let (v0, s0) = q_iter.next().unwrap();
            let (v1, s1) = q_iter.next().unwrap();
            let (v2, s2) = q_iter.next().unwrap();
            let mut slots = [
                GateOut3Slot {
                    down: downs3[0].as_view(),
                    forget: &forgets3[0],
                    gain: &gains3[0],
                    probs: Some(probs.slice(0..)),
                    norm_weight: &norm_weights[0],
                    normed: Some(n0.as_view_mut()),
                    q_values: Some(v0),
                    q_scales: Some(s0),
                },
                GateOut3Slot {
                    down: downs3[1].as_view(),
                    forget: &forgets3[1],
                    gain: &gains3[1],
                    probs: Some(probs.slice(NB + 1..)),
                    norm_weight: &norm_weights[1],
                    normed: Some(n1.as_view_mut()),
                    q_values: Some(v1),
                    q_scales: Some(s1),
                },
                GateOut3Slot {
                    down: downs3[2].as_view(),
                    forget: &forgets3[2],
                    gain: &gains3[2],
                    probs: Some(probs.slice(2 * (NB + 1)..)),
                    norm_weight: &norm_weights[2],
                    normed: Some(n2.as_view_mut()),
                    q_values: Some(v2),
                    q_scales: Some(s2),
                },
            ];
            misc.message_gate_out_rows_3(
                stream,
                &mut slots,
                &mut buffers,
                (&m_vec, HIDDEN),
                Some(&blocks.as_view()),
                probs_stride,
                Some(QuantFormat::Mxfp8),
                EPS,
                ROWS,
            )?;
            let slot_vals = ROWS * HIDDEN;
            let slot_scales = ROWS * (HIDDEN / 32);
            for k in 0..3 {
                assert_bf16_close(
                    &format!("message_gate_out_rows_3 h_shared={h_shared} slot {k}: normed"),
                    &stream.clone_dtoh(&batched_norms[k])?,
                    &stream.clone_dtoh(&ref_norms[k])?,
                    0,
                );
                quantize_reference(
                    stream,
                    &experts,
                    &ref_norms[k],
                    QuantFormat::Mxfp8,
                    &dummy_scale,
                    &mut want_q,
                )?;
                assert_eq!(
                    stream.clone_dtoh(&fp8_rows.values().slice(k * slot_vals..(k + 1) * slot_vals))?,
                    stream.clone_dtoh(&want_q.values().slice(..slot_vals))?,
                    "message_gate_out_rows_3 h_shared={h_shared} slot {k}: mxfp8 values"
                );
                assert_eq!(
                    stream.clone_dtoh(
                        &fp8_rows
                            .scales()
                            .slice(k * slot_scales..(k + 1) * slot_scales)
                    )?,
                    stream.clone_dtoh(&want_q.scales().slice(..slot_scales))?,
                    "message_gate_out_rows_3 h_shared={h_shared} slot {k}: mxfp8 scales"
                );
            }
        }
    }
    Ok(())
}
