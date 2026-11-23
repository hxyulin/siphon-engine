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
use inkling_cuda::{
    Gpu,
    ops::{
        conv::Conv,
        elementwise::Elementwise,
        misc::Misc,
        norm::{AttentionPost, AttentionPostRows, KvConvWeights, Norm},
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
/// with and without log scaling. Same rounded operations, so bit for bit.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the single-row and multi-row calls per chunk"
)]
fn attention_post_rows_match_single_rows() -> anyhow::Result<()> {
    const CHUNKS: [usize; 4] = [5, 1, 2, 7];
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let norm = Norm::load(gpu.context())?;
    let conv = Conv::load(gpu.context())?;
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
                    log_scale: log_scale.then_some((&start, alpha, n_floor)),
                    qk_threshold: QK_THRESHOLD,
                    rel_threshold: REL_THRESHOLD,
                    eps: EPS,
                },
                rows,
            )?;
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
