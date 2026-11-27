//! Attention kernels against MLX outputs from `reference/fixtures/attention.py` and against an
//! F64 reference on the host.
//!
//! The fixture feeds 1100 tokens into a sliding (1024-slot ring) and a global cache and records
//! MLX's decode attention at a few positions across the window and the bias table width. The
//! host reference covers what the fixture is too short for: a global cache of 40,000 keys cut
//! into the full number of splits, and a sliding ring that has wrapped several times.
//!
//! Chunked prefill is checked against the same F64 reference and against append plus decode
//! attention run row by row, including the cache contents both leave behind.
//!
//! FP8 global caches are checked against the same F64 reference applied to the E4M3 values
//! the cache holds, with the BF16 tolerance: decoding E4M3 is exact, so only the rounding of
//! the cache differs. The tests also print the error against the reference on the BF16
//! values, which is the quantization error.

#![cfg(target_os = "linux")]

mod common;

use std::sync::{Arc, OnceLock};

use common::{
    Fixture, assert_bf16_close, assert_f32_within_ulps, bf16_ulps, pseudo_random,
    pseudo_random_bf16,
};
use cudarc::driver::{CudaSlice, CudaStream};
use half::bf16;
use siphon_cuda::{
    Gpu,
    ops::{
        Products,
        attention::{
            Attention, AttentionScratch, DecodeStep, HEAD_DIM, HEADS, KV_HEADS, KvCache, KvData,
            KvFormat, MAX_PREFILL_ROWS, MIN_SPLIT_SLOTS, PrefillScratch, REL_DIM,
        },
        elementwise::Elementwise,
        norm::{AttentionPost, KvConvWeights, Norm},
    },
};

const WINDOW: usize = 1024;
/// Global cache capacity; the fixture's 1100 tokens fit.
const CAPACITY: usize = 1280;
/// Bias table width in the model and the fixture.
const EXTENT: usize = 1024;
/// `HEAD_DIM^-0.5` as the launcher passes it.
const SCALE: f32 = 0.088_388_346;
/// Query and key norm limit of the model's L2 clip, `sqrt(512)`.
const QK_THRESHOLD: f32 = 22.627_417;
const ROW: usize = KV_HEADS * HEAD_DIM;

struct Setup {
    fixture: Fixture,
    gpu: Gpu,
    attention: Attention,
}

impl Setup {
    fn new() -> anyhow::Result<Self> {
        let gpu = Gpu::new(0)?;
        let attention = Attention::load(gpu.context())?;
        Ok(Self {
            fixture: Fixture::load("attention")?,
            gpu,
            attention,
        })
    }

    fn checks(&self) -> anyhow::Result<Vec<usize>> {
        let (checks, _) = self.fixture.i32("checks")?;
        Ok(checks
            .into_iter()
            .map(|c| usize::try_from(c).expect("positions are non-negative"))
            .collect())
    }
}

/// One query's inputs: `q` `[24, 128]`, `bias` `[24, EXTENT]` and `sinks` `[24]`.
#[derive(Clone)]
struct Query {
    q: Vec<bf16>,
    bias: Vec<bf16>,
    sinks: Vec<bf16>,
}

/// Decode attention in F64 over the host copy of a cache, `keys` and `values` `[slots, 8,
/// 128]`, for the token at `position`. Returns each output and the same weighted sum of the
/// absolute values, the scale of the terms whose rounding errors add up in the output.
fn reference(
    query: &Query,
    keys: &[f32],
    values: &[f32],
    window: Option<usize>,
    position: usize,
) -> (Vec<f64>, Vec<f64>) {
    let slots = keys.len() / ROW;
    let live = (position + 1).min(slots);
    let newest = position % slots;
    let mut out = vec![0.0; HEADS * HEAD_DIM];
    let mut magnitude = vec![0.0; HEADS * HEAD_DIM];
    for h in 0..HEADS {
        let kv = (h / (HEADS / KV_HEADS)) * HEAD_DIM;
        let q: Vec<f64> = query.q[h * HEAD_DIM..(h + 1) * HEAD_DIM]
            .iter()
            .map(|&x| f64::from(SCALE * x.to_f32()))
            .collect();
        let mut scores = Vec::with_capacity(live);
        for s in 0..live {
            let distance = (newest + slots - s) % slots;
            if window.is_some_and(|w| distance >= w) {
                continue;
            }
            let k = &keys[s * ROW + kv..s * ROW + kv + HEAD_DIM];
            let mut score: f64 = q.iter().zip(k).map(|(&a, &b)| a * f64::from(b)).sum();
            if distance < EXTENT {
                score += query.bias[h * EXTENT + distance].to_f64();
            }
            scores.push((s, score));
        }
        let sink = query.sinks[h].to_f64();
        let max = scores.iter().map(|&(_, s)| s).fold(sink, f64::max);
        let mut denominator = (sink - max).exp();
        let head_out = &mut out[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        let head_magnitude = &mut magnitude[h * HEAD_DIM..(h + 1) * HEAD_DIM];
        for &(s, score) in &scores {
            let weight = (score - max).exp();
            denominator += weight;
            let value = &values[s * ROW + kv..s * ROW + kv + HEAD_DIM];
            for ((o, m), v) in head_out
                .iter_mut()
                .zip(head_magnitude.iter_mut())
                .zip(value)
            {
                *o += weight * f64::from(*v);
                *m += weight * f64::from(v.abs());
            }
        }
        for (o, m) in head_out.iter_mut().zip(head_magnitude.iter_mut()) {
            *o /= denominator;
            *m /= denominator;
        }
    }
    (out, magnitude)
}

/// F32 rounding error allowed per unit of [`reference`]'s magnitude. The kernel's errors are
/// those of F32 dot products of 128 terms and of exp2 on the scores, about 3e-5 of the
/// magnitude; this allows 8 times that.
const F32_ERROR: f64 = 1.0 / 4096.0;
/// Additional error allowed against MLX per unit of magnitude. From 1024 keys on MLX rounds
/// its 64 partial outputs to BF16, each with a relative error of at most 2^-8, and the
/// partials' absolute values add up to the magnitude.
const MLX_PARTIAL_ERROR: f64 = 1.0 / 256.0;

/// Additional error allowed for prefill attention per unit of magnitude: it rounds each
/// softmax weight to F16 for the tensor-core product with the values, a relative error of
/// at most 2^-11 per weight (down to weights 2^-29 of the largest, below which the error
/// is under 2^-40 of the largest), so at most 2^-11 of the weighted sum of absolute values.
const PREFILL_P_ERROR: f64 = 1.0 / 2048.0;
/// Absolute error allowed for prefill attention: BF16 values below 2^-14 lose bits when
/// they convert to F16, at most 2^-25 each, and the weights sum to at most 1.
const PREFILL_V_ERROR: f64 = 1.0 / 33_554_432.0;
/// Error of the FP8 product prefill attention per unit of magnitude, against a reference
/// with the same MXFP8 queries ([`mxfp8_queries`]): the softmax weights round to E4M3,
/// relative error up to 2^-4.
const PREFILL_FP8_ERROR: f64 = 1.0 / 14.0;
/// Absolute error of the FP8 product prefill attention: weights below 2^-17 of a row's
/// largest flush to zero in E4M3, and the values are at most about 8.
const PREFILL_FP8_V_ERROR: f64 = 8.0 / 131_072.0;
/// Error of the FP4 product prefill attention (`prefill_attention_fp4_mma`) per unit of
/// magnitude, against a reference with the same NVFP4 queries ([`nvfp4_queries`]) and
/// stored NVFP4 values: softmax weights round to E4M3 (`<= 2^-4`) and decoded NVFP4 values
/// also round to E4M3 (`<= 2^-4`) for `pv8`, so combined relative error is at most ~1/7.
const PREFILL_FP4_ERROR: f64 = 1.0 / 6.0;

/// How far one comparison came from its tolerance.
#[derive(Clone, Copy, Debug, Default)]
struct Closeness {
    /// Elements that differ from `want` rounded to BF16.
    differing: usize,
    /// The largest share of its tolerance an element uses.
    worst: f64,
    /// The largest error over the element's magnitude, after taking off half an ulp for the
    /// output rounding, among elements with magnitude at least 2^-10.
    relative: f64,
}

impl Closeness {
    fn merge(self, other: Self) -> Self {
        Self {
            differing: self.differing + other.differing,
            worst: self.worst.max(other.worst),
            relative: self.relative.max(other.relative),
        }
    }
}

/// How much error [`check_within`] allows.
#[derive(Clone, Copy, Debug)]
struct Tolerance {
    /// BF16 ulps of `want`: 1 for the output rounding, 2 when `want` is itself a rounded
    /// output.
    ulps: f64,
    /// Error per unit of magnitude from [`reference`].
    relative: f64,
    /// Error regardless of magnitude.
    absolute: f64,
}

impl Tolerance {
    /// Against the F64 reference: 1 ulp plus `relative` times the magnitude.
    fn reference(relative: f64) -> Self {
        Self {
            ulps: 1.0,
            relative,
            absolute: 0.0,
        }
    }
}

/// Fails unless every output is within the tolerance of `want`: some BF16 ulps (the output
/// rounding) plus a share of its magnitude from [`reference`] (the arithmetic before it).
fn check_within(
    label: &str,
    got: &[bf16],
    want: &[f64],
    magnitude: &[f64],
    tolerance: Tolerance,
) -> Closeness {
    let mut closeness = Closeness::default();
    for (i, ((&g, &w), &a)) in got.iter().zip(want).zip(magnitude).enumerate() {
        let rounded = bf16::from_f64(w);
        let ulp = (bf16::from_bits(rounded.to_bits() + 1).to_f64() - rounded.to_f64()).abs();
        let allowed = tolerance.ulps * ulp + tolerance.relative * a + tolerance.absolute;
        let err = (g.to_f64() - w).abs();
        assert!(
            err <= allowed,
            "{label}[{i}]: got {g}, want {w} (magnitude {a})"
        );
        closeness.worst = closeness.worst.max(err / allowed);
        if a >= 1.0 / 1024.0 {
            closeness.relative = closeness.relative.max((err - ulp / 2.0).max(0.0) / a);
        }
        closeness.differing += usize::from(bf16_ulps(g, rounded) > 0);
    }
    closeness
}

/// [`check_within`], printing the result.
fn assert_within(label: &str, got: &[bf16], want: &[f64], magnitude: &[f64], relative: f64) {
    let closeness = check_within(label, got, want, magnitude, Tolerance::reference(relative));
    eprintln!(
        "{label}: {} of {} elements differ from the rounded value; the worst uses {:.0}% of \
         its tolerance",
        closeness.differing,
        want.len(),
        100.0 * closeness.worst
    );
}

/// The value of E4M3 code `bits` (no NaN codes reach it).
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
    // The last code whose value is at most `a`; code 0 is zero.
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

/// The value `format`'s cache stores for `x`.
fn stored(format: KvFormat, x: bf16) -> f32 {
    match format {
        KvFormat::Bf16 => x.to_f32(),
        KvFormat::Fp8 => e4m3_value(e4m3_from_f32(x.to_f32())),
        KvFormat::Nvfp4 | KvFormat::Nvfp4All | KvFormat::Mixed => {
            unimplemented!("NVFP4 rounds groups, not single values")
        }
    }
}

/// The queries `[heads, 128]` as the FP8 product prefill attention quantizes them:
/// - on `sm_120a`: MXFP8 per head and 32 dims, a UE8M0 scale of the block's amax / 448 rounded up
///   (at least 2^-126) and E4M3 values rounded to nearest even;
/// - on `sm_89`: per-head-row E4M3 with F32 scale `amax / 448` folded into the score scale.
fn mxfp8_queries(q: &[bf16]) -> Vec<bf16> {
    if cfg!(sm89) {
        q.chunks(HEAD_DIM)
            .flat_map(|head| {
                let amax = head.iter().fold(0f32, |m, x| m.max(x.to_f32().abs()));
                let scale = amax * (1.0 / 448.0);
                let inv = if amax > 0.0 { 448.0 / amax } else { 0.0 };
                head.iter()
                    .map(move |x| bf16::from_f32(e4m3_value(e4m3_from_f32(x.to_f32() * inv)) * scale))
            })
            .collect()
    } else {
        q.chunks(32)
            .flat_map(|block| {
                let amax = block.iter().fold(0f32, |m, x| m.max(x.to_f32().abs()));
                let raw = amax / 448.0;
                let exponent = ((raw.to_bits() + 0x007f_ffff) & 0x7f80_0000) >> 23;
                let scale = f32::from_bits(exponent.max(1) << 23);
                block
                    .iter()
                    .map(move |x| bf16::from_f32(e4m3_value(e4m3_from_f32(x.to_f32() / scale)) * scale))
            })
            .collect()
    }
}

/// Rounds `y` to the nearest E2M1 value, ties to even, saturated to `+-6.0`, matching
/// `cvt.rn.satfinite.e2m1x2.f32`.
fn e2m1_round(y: f32) -> f32 {
    let a = y.abs();
    let mag: f32 = if a <= 0.25 {
        0.0
    } else if a < 0.75 {
        0.5
    } else if a <= 1.25 {
        1.0
    } else if a < 1.75 {
        1.5
    } else if a <= 2.5 {
        2.0
    } else if a < 3.5 {
        3.0
    } else if a <= 5.0 {
        4.0
    } else {
        6.0
    };
    mag.copysign(y)
}

/// Values as an NVFP4 KV cache (`kv_append_rows_fp4`) stores them: per 16 elements, an
/// E4M3 scale `E4M3(amax / 6)` and E2M1 values rounded to nearest even.
fn nvfp4_stored(values: &[bf16]) -> Vec<f32> {
    values
        .chunks(16)
        .flat_map(|group| {
            let amax = group.iter().fold(0f32, |m, x| m.max(x.to_f32().abs()));
            let scale = e4m3_value(e4m3_from_f32(amax / 6.0));
            group.iter().map(move |x| {
                if scale == 0.0 {
                    0.0
                } else {
                    e2m1_round(x.to_f32() / scale) * scale
                }
            })
        })
        .collect()
}

/// The queries `[heads, 128]` as `prefill_attention_fp4_mma` quantizes them: a per-head
/// F32 outer scale `amax / 1536`, per-16 UE4M3 scales `E4M3(group_amax * enc / 6)`, and
/// E2M1 values rounded to nearest even.
fn nvfp4_queries(q: &[bf16]) -> Vec<bf16> {
    q.chunks(HEAD_DIM)
        .flat_map(|head| {
            let row_amax = head.iter().fold(0f32, |m, x| m.max(x.to_f32().abs()));
            let enc = if row_amax > 0.0 {
                (1536.0 / row_amax).min(1e35)
            } else {
                0.0
            };
            let outer = if enc > 0.0 { 1.0 / enc } else { 0.0 };
            head.chunks(16).flat_map(move |group| {
                let gmax = group.iter().fold(0f32, |m, x| m.max(x.to_f32().abs()));
                let sv = e4m3_value(e4m3_from_f32(gmax * enc / 6.0));
                let inv = if sv == 0.0 { 0.0 } else { enc / sv };
                group
                    .iter()
                    .map(move |x| bf16::from_f32(e2m1_round(x.to_f32() * inv) * sv * outer))
            })
        })
        .collect()
}

/// Cosine similarity and max absolute difference between two BF16 slices.
fn cosine_and_max_abs(a: &[bf16], b: &[bf16]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (xf, yf) = (x.to_f64(), y.to_f64());
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
        max_abs = max_abs.max((xf - yf).abs());
    }
    let denom = (na * nb).sqrt();
    let cosine = if denom > 0.0 { dot / denom } else { 1.0 };
    (cosine, max_abs)
}

/// Prints how far `got` is from `want`, the reference on the unquantized cache: the RMS error
/// over the RMS output, and the largest error over the largest output.
fn report_quantization_error(label: &str, got: &[bf16], want: &[f64]) {
    let (mut err2, mut want2) = (0.0f64, 0.0f64);
    let (mut max_err, mut max_want) = (0.0f64, 0.0f64);
    for (&g, &w) in got.iter().zip(want) {
        let err = g.to_f64() - w;
        err2 += err * err;
        want2 += w * w;
        max_err = max_err.max(err.abs());
        max_want = max_want.max(w.abs());
    }
    eprintln!(
        "{label} vs unquantized F64: RMS error {:.2e} of the RMS output, largest error {:.2e} \
         of the largest output",
        (err2 / want2).sqrt(),
        max_err / max_want
    );
}

#[test]
fn relative_bias_matches_mlx() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let Setup {
        fixture,
        gpu,
        attention,
    } = &setup;
    let stream = gpu.stream();
    for case in ["sliding", "global"] {
        let (table, shape) = fixture.bf16(&format!("{case}.table"))?;
        let table_dev = stream.clone_htod(&table)?;
        let mut bias_dev = stream.alloc_zeros::<bf16>(HEADS * shape[1])?;
        for position in setup.checks()? {
            let prefix = format!("{case}.{position}");
            let (rel, _) = fixture.bf16(&format!("{prefix}.rel"))?;
            let (want, _) = fixture.bf16(&format!("{prefix}.bias"))?;
            let rel_dev = stream.clone_htod(&rel)?;
            attention.relative_bias(stream, &rel_dev, &table_dev, &mut bias_dev)?;
            let got = stream.clone_dtoh(&bias_dev)?;
            assert_bf16_close(&format!("{prefix}.bias"), &got, &want, 0);
        }
    }
    Ok(())
}

/// The multi-row bias kernel gives each row exactly what the single-row kernel gives, on the
/// fixture's tables and random relative states.
#[test]
fn relative_bias_rows_matches_relative_bias() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let stream = setup.gpu.stream();
    let mut rng = Rng(11);
    for case in ["sliding", "global"] {
        let (table, shape) = setup.fixture.bf16(&format!("{case}.table"))?;
        let extent = shape[1];
        let table_dev = stream.clone_htod(&table)?;
        let rows = 37;
        let rel = rng.normals(rows * HEADS * REL_DIM, 1.0);
        let rel_dev = stream.clone_htod(&rel)?;
        let mut bias_dev = stream.alloc_zeros::<bf16>(rows * HEADS * extent)?;
        setup
            .attention
            .relative_bias_rows(stream, &rel_dev, &table_dev, &mut bias_dev)?;
        let got = stream.clone_dtoh(&bias_dev)?;
        let mut want_dev = stream.alloc_zeros::<bf16>(HEADS * extent)?;
        for row in 0..rows {
            let rel_row =
                stream.clone_htod(&rel[row * HEADS * REL_DIM..(row + 1) * HEADS * REL_DIM])?;
            setup
                .attention
                .relative_bias(stream, &rel_row, &table_dev, &mut want_dev)?;
            let want = stream.clone_dtoh(&want_dev)?;
            let span = row * HEADS * extent..(row + 1) * HEADS * extent;
            assert_bf16_close(&format!("{case} row {row}"), &got[span], &want, 0);
        }
    }
    Ok(())
}

/// Host copy of a cache, `[slots, 8, 128]` each.
struct HostCache {
    /// The values the cache stores.
    keys: Vec<f32>,
    values: Vec<f32>,
    /// The BF16 values appended, before any rounding to the cache format.
    exact_keys: Vec<f32>,
    exact_values: Vec<f32>,
}

/// Appends `k` and `v` (`[tokens, 8, 128]`) to `cache` one token at a time and, at each
/// position that `query` returns inputs for, runs decode attention and passes the output and
/// the host copy of the cache to `check`.
fn run_decode(
    gpu: &Gpu,
    attention: &Attention,
    mut cache: KvCache,
    k: &[bf16],
    v: &[bf16],
    mut query: impl FnMut(usize) -> anyhow::Result<Option<Query>>,
    mut check: impl FnMut(usize, &Query, &HostCache, &[bf16]) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let stream = gpu.stream();
    let slots = cache.slots();
    let format = cache.format();
    let mut host = HostCache {
        keys: vec![0.0; slots * ROW],
        values: vec![0.0; slots * ROW],
        exact_keys: vec![0.0; slots * ROW],
        exact_values: vec![0.0; slots * ROW],
    };
    let k_dev = stream.clone_htod(k)?;
    let v_dev = stream.clone_htod(v)?;
    let mut scratch = AttentionScratch::new(stream)?;
    let mut out_dev = stream.alloc_zeros::<bf16>(HEADS * HEAD_DIM)?;
    let mut position_dev = stream.alloc_zeros::<u32>(1)?;
    for position in 0..k.len() / ROW {
        assert!(cache.holds(position));
        let token = position * ROW..(position + 1) * ROW;
        let slot = (position % slots) * ROW;
        for i in 0..ROW {
            let (key, value) = (k[token.start + i], v[token.start + i]);
            host.keys[slot + i] = stored(format, key);
            host.values[slot + i] = stored(format, value);
            host.exact_keys[slot + i] = key.to_f32();
            host.exact_values[slot + i] = value.to_f32();
        }
        stream.memcpy_htod(&[u32::try_from(position)?], &mut position_dev)?;
        attention.append(
            stream,
            &mut cache,
            &position_dev,
            &k_dev.slice(token.clone()),
            &v_dev.slice(token),
        )?;
        let Some(inputs) = query(position)? else {
            continue;
        };
        let q_dev = stream.clone_htod(&inputs.q)?;
        let bias_dev = stream.clone_htod(&inputs.bias)?;
        let sinks_dev = stream.clone_htod(&inputs.sinks)?;
        attention.decode_attention(
            stream,
            &q_dev,
            &cache,
            &position_dev,
            &bias_dev,
            &sinks_dev,
            &mut scratch,
            &mut out_dev,
        )?;
        let got = stream.clone_dtoh(&out_dev)?;
        check(position, &inputs, &host, &got)?;
    }
    Ok(())
}

/// Runs the fixture's case through the kernels and compares each checked position with MLX
/// and with the F64 reference.
fn check_fixture(setup: &Setup, case: &str, cache: KvCache) -> anyhow::Result<()> {
    let fixture = &setup.fixture;
    let (k, _) = fixture.bf16("k")?;
    let (v, _) = fixture.bf16("v")?;
    let checks = setup.checks()?;
    let window = cache.window();
    let format = cache.format();
    run_decode(
        &setup.gpu,
        &setup.attention,
        cache,
        &k,
        &v,
        |position| {
            if !checks.contains(&position) {
                return Ok(None);
            }
            let prefix = format!("{case}.{position}");
            Ok(Some(Query {
                q: fixture.bf16(&format!("{prefix}.q"))?.0,
                bias: fixture.bf16(&format!("{prefix}.bias"))?.0,
                sinks: fixture.bf16(&format!("{prefix}.sinks"))?.0,
            }))
        },
        |position, query, host, got| {
            let prefix = format!("{case}.{position}");
            let (want, magnitude) = reference(query, &host.keys, &host.values, window, position);
            assert_within(
                &format!("{prefix} vs F64"),
                got,
                &want,
                &magnitude,
                F32_ERROR,
            );
            if format == KvFormat::Fp8 {
                let (exact, _) = reference(
                    query,
                    &host.exact_keys,
                    &host.exact_values,
                    window,
                    position,
                );
                report_quantization_error(&prefix, got, &exact);
                return Ok(());
            }
            let (mlx, _) = fixture.bf16(&format!("{prefix}.out"))?;
            let mlx: Vec<f64> = mlx.iter().map(|x| x.to_f64()).collect();
            let relative = F32_ERROR + MLX_PARTIAL_ERROR;
            assert_within(&format!("{prefix} vs MLX"), got, &mlx, &magnitude, relative);
            Ok(())
        },
    )
}

#[test]
fn sliding_decode_attention_matches_mlx() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let cache = KvCache::sliding(setup.gpu.stream(), WINDOW)?;
    check_fixture(&setup, "sliding", cache)
}

#[test]
fn global_decode_attention_matches_mlx() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let cache = KvCache::global(setup.gpu.stream(), CAPACITY)?;
    check_fixture(&setup, "global", cache)
}

/// The fixture's global case on an FP8 cache, against F64 on the E4M3 values.
#[test]
fn fp8_global_decode_attention_matches_reference() -> anyhow::Result<()> {
    let setup = Setup::new()?;
    let cache = KvCache::global_with_format(setup.gpu.stream(), CAPACITY, KvFormat::Fp8)?;
    check_fixture(&setup, "global", cache)
}

/// Xorshift64* with approximately normal samples, enough to make test data without a
/// dependency.
struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
        bits as f32 / (1u64 << 24) as f32
    }

    /// The sum of 4 uniforms, rescaled to unit variance.
    fn normal(&mut self) -> f32 {
        let sum: f32 = (0..4).map(|_| self.uniform()).sum();
        (sum - 2.0) * 3.0f32.sqrt()
    }

    fn normals(&mut self, len: usize, scale: f32) -> Vec<bf16> {
        (0..len)
            .map(|_| bf16::from_f32(scale * self.normal()))
            .collect()
    }

    /// `rows` heads of 128 that mix in a few shared directions, so that some keys line up
    /// with a query, with norms from a third of the model's clip limit up to it.
    fn heads(&mut self, rows: usize, topics: &[Vec<f32>]) -> Vec<bf16> {
        let mut out = Vec::with_capacity(rows * HEAD_DIM);
        for _ in 0..rows {
            let mix: Vec<f32> = topics.iter().map(|_| self.normal()).collect();
            let x: Vec<f32> = (0..HEAD_DIM)
                .map(|i| {
                    self.normal() + topics.iter().zip(&mix).map(|(t, m)| t[i] * m).sum::<f32>()
                })
                .collect();
            let norm = x.iter().map(|a| a * a).sum::<f32>().sqrt();
            let target = QK_THRESHOLD * (1.0 / 3.0 + self.uniform() * (1.25 - 1.0 / 3.0)).min(1.0);
            out.extend(x.iter().map(|a| bf16::from_f32(a * target / norm)));
        }
        out
    }
}

/// Random keys, values and queries; checks decode attention at `checks` against the F64
/// reference.
fn check_random(cache: KvCache, tokens: usize, checks: &[usize], seed: u64) -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let attention = Attention::load(gpu.context())?;
    let mut rng = Rng(seed);
    let topics: Vec<Vec<f32>> = (0..4)
        .map(|_| (0..HEAD_DIM).map(|_| 1.5 * rng.normal()).collect())
        .collect();
    let k = rng.heads(tokens * KV_HEADS, &topics);
    let v = rng.normals(tokens * ROW, 1.0);
    let window = cache.window();
    let format = cache.format();
    let label = match (window, format) {
        (Some(_), _) => "sliding",
        (None, KvFormat::Bf16) => "global",
        (None, KvFormat::Fp8) => "fp8 global",
        (None, KvFormat::Nvfp4 | KvFormat::Nvfp4All | KvFormat::Mixed) => "nvfp4 global",
    };
    run_decode(
        &gpu,
        &attention,
        cache,
        &k,
        &v,
        |position| {
            Ok(checks.contains(&position).then(|| Query {
                q: rng.heads(HEADS, &topics),
                bias: rng.normals(HEADS * EXTENT, 1.5),
                sinks: rng.normals(HEADS, 2.0),
            }))
        },
        |position, query, host, got| {
            let label = format!("{label}.{position}");
            let (want, magnitude) = reference(query, &host.keys, &host.values, window, position);
            assert_within(&label, got, &want, &magnitude, F32_ERROR);
            if format == KvFormat::Fp8 {
                let (exact, _) = reference(
                    query,
                    &host.exact_keys,
                    &host.exact_values,
                    window,
                    position,
                );
                report_quantization_error(&label, got, &exact);
            }
            Ok(())
        },
    )
}

#[test]
fn long_global_decode_attention_matches_reference() -> anyhow::Result<()> {
    let tokens = 40_000;
    let gpu = Gpu::new(0)?;
    let cache = KvCache::global(gpu.stream(), 40_960)?;
    let checks = [0, 1, 31, 32, 33, 1023, 1024, 2561, 8191, 20_000, tokens - 1];
    check_random(cache, tokens, &checks, 1)
}

#[test]
fn long_fp8_global_decode_attention_matches_reference() -> anyhow::Result<()> {
    let tokens = 40_000;
    let gpu = Gpu::new(0)?;
    let cache = KvCache::global_with_format(gpu.stream(), 40_960, KvFormat::Fp8)?;
    let checks = [
        0,
        1,
        31,
        32,
        33,
        63,
        64,
        65,
        1023,
        1024,
        2561,
        8191,
        20_000,
        tokens - 1,
    ];
    check_random(cache, tokens, &checks, 1)
}

/// The FP8 append stores every BF16 value rounded to E4M3 as the host does it: nearest, ties
/// to even, saturated to +-448, including zeros, subnormals, ties and out-of-range values.
#[test]
fn fp8_append_rounds_to_nearest_even() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let attention = Attention::load(gpu.context())?;
    let tokens = 64;
    let mut cache = KvCache::global_with_format(stream, tokens, KvFormat::Fp8)?;
    let mut rng = Rng(3);
    let edges: Vec<bf16> = [
        0.0,
        -0.0,
        448.0,
        -448.0,
        464.0,
        480.0,
        1e4,
        -1e4,
        0.015_625,
        0.001_953_125,
        0.000_976_562_5,
        0.002_929_687_5,
        -0.002_929_687_5,
        1.0625,
        1.1875,
        22.627,
        -22.627,
        11.3,
        3e-5,
    ]
    .into_iter()
    .map(bf16::from_f32)
    .collect();
    let mut k = rng.normals(tokens * ROW, 4.0);
    let mut v = rng.normals(tokens * ROW, 1.0);
    k[..edges.len()].copy_from_slice(&edges);
    v[ROW..ROW + edges.len()].copy_from_slice(&edges);
    let k_dev = stream.clone_htod(&k)?;
    let v_dev = stream.clone_htod(&v)?;
    let mut position_dev = stream.alloc_zeros::<u32>(1)?;
    for position in 0..tokens {
        let token = position * ROW..(position + 1) * ROW;
        stream.memcpy_htod(&[u32::try_from(position)?], &mut position_dev)?;
        attention.append(
            stream,
            &mut cache,
            &position_dev,
            &k_dev.slice(token.clone()),
            &v_dev.slice(token),
        )?;
    }
    let KvData::Fp8 { keys, values } = cache.data() else {
        anyhow::bail!("the cache is not FP8");
    };
    for (name, got, want) in [
        ("keys", stream.clone_dtoh(keys)?, &k),
        ("values", stream.clone_dtoh(values)?, &v),
    ] {
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let expected = e4m3_from_f32(w.to_f32());
            assert_eq!(
                g, expected,
                "{name}[{i}]: {w} stored as {g:#04x}, want {expected:#04x}"
            );
        }
    }
    Ok(())
}

#[test]
fn wrapped_sliding_decode_attention_matches_reference() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let cache = KvCache::sliding(gpu.stream(), WINDOW)?;
    let checks = [0, 5, 1023, 1024, 1500, 2047, 2048, 4999];
    check_random(cache, 5000, &checks, 2)
}

#[test]
fn global_cache_holds_its_capacity() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let global = KvCache::global(gpu.stream(), 2)?;
    assert!(global.holds(0) && global.holds(1));
    assert!(!global.holds(2));
    let sliding = KvCache::sliding(gpu.stream(), 2)?;
    assert!(sliding.holds(2) && sliding.holds(1 << 20));
    Ok(())
}

/// Relative-state norm limit of the model's L2 clip.
const REL_THRESHOLD: f32 = 8.0;
/// The model's RMS norm epsilon.
const EPS: f32 = 1e-6;
/// The model's query log scaling: alpha and n_floor.
const LOG_SCALE: (f32, f32) = (0.1, 8192.0);

/// The conv weights of one decode step: the static weights, or the step's dynamic taps with
/// the F32 static weights.
fn conv_weights<'a>(
    dynamic: bool,
    weights: &'a [CudaSlice<bf16>; 2],
    taps: &'a [CudaSlice<bf16>; 2],
    statics: &'a [CudaSlice<f32>; 2],
) -> KvConvWeights<'a> {
    if dynamic {
        KvConvWeights::Dynamic {
            k_dynamic: taps[0].as_view(),
            v_dynamic: taps[1].as_view(),
            k_static: &statics[0],
            v_static: &statics[1],
        }
    } else {
        KvConvWeights::Static {
            k: &weights[0],
            v: &weights[1],
        }
    }
}

/// The key and value row in `slot` of `cache`, as bits.
fn cache_row(
    stream: &Arc<CudaStream>,
    cache: &KvCache,
    slot: usize,
) -> anyhow::Result<[Vec<u16>; 2]> {
    let row = slot * ROW..(slot + 1) * ROW;
    let [k, v] = match cache.data() {
        KvData::Bf16 { keys, values } => [keys, values].map(|x| {
            stream
                .clone_dtoh(&x.slice(row.clone()))
                .map(|v| v.iter().map(|b| b.to_bits()).collect())
        }),
        KvData::Fp8 { keys, values } => [keys, values].map(|x| {
            stream
                .clone_dtoh(&x.slice(row.clone()))
                .map(|v| v.iter().map(|&b| u16::from(b)).collect())
        }),
        KvData::Nvfp4 { .. } => anyhow::bail!("NVFP4 rows are not elements"),
    };
    Ok([k?, v?])
}

/// `Attention::decode_step` against the kernels it replaces (`Norm::attention_post`,
/// relative bias, append, decode attention and the output gate), on random inputs at each
/// position in `checks`, alternating static and dynamic convs. Positions between checks get
/// random keys and values through `append`. The output, the conv states and the appended
/// cache row must match bit for bit.
#[expect(
    clippy::too_many_lines,
    reason = "the separate kernels and the fused one, per position"
)]
fn check_decode_step(
    new_cache: fn(&Arc<CudaStream>) -> anyhow::Result<KvCache>,
    checks: &[usize],
) -> anyhow::Result<()> {
    /// Filler tokens, cycled.
    const FILL: usize = 2048;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let attention = Attention::load(gpu.context())?;
    let norm = Norm::load(gpu.context())?;
    let elementwise = Elementwise::load(gpu.context())?;
    let (mut cache_ref, mut cache) = (new_cache(stream)?, new_cache(stream)?);
    let log_scale = cache.window().is_none();
    let slots = cache.slots();
    let mut rng = Rng(11);
    let topics: Vec<Vec<f32>> = (0..4)
        .map(|_| (0..HEAD_DIM).map(|_| 1.5 * rng.normal()).collect())
        .collect();
    let fill_k = stream.clone_htod(&rng.heads(FILL * KV_HEADS, &topics))?;
    let fill_v = stream.clone_htod(&rng.normals(FILL * ROW, 1.0))?;
    let table = stream.clone_htod(&rng.normals(REL_DIM * EXTENT, 0.4))?;
    let q_scale = stream.clone_htod(&pseudo_random_bf16(HEADS * HEAD_DIM, 60, 0.5, 1.0))?;
    let r_scale = stream.clone_htod(&pseudo_random_bf16(HEADS * REL_DIM, 61, 0.5, 1.0))?;
    let weights = [
        stream.clone_htod(&pseudo_random_bf16(ROW * 4, 62, 0.5, 0.0))?,
        stream.clone_htod(&pseudo_random_bf16(ROW * 4, 63, 0.5, 0.0))?,
    ];
    let statics = [
        stream.clone_htod(&pseudo_random(4 * ROW, 64, 0.5))?,
        stream.clone_htod(&pseudo_random(4 * ROW, 65, 0.5))?,
    ];
    let start = [
        pseudo_random(3 * ROW, 66, 2.0),
        pseudo_random(3 * ROW, 67, 2.0),
    ];
    let mut states_ref = [stream.clone_htod(&start[0])?, stream.clone_htod(&start[1])?];
    let mut states = [stream.clone_htod(&start[0])?, stream.clone_htod(&start[1])?];
    let (mut scratch_ref, mut scratch) = (
        AttentionScratch::new(stream)?,
        AttentionScratch::new(stream)?,
    );
    let mut position_dev = stream.alloc_zeros::<u32>(1)?;
    let zeros = |n: usize| -> anyhow::Result<CudaSlice<bf16>> { Ok(stream.alloc_zeros(n)?) };
    let (mut k, mut v) = (zeros(ROW)?, zeros(ROW)?);
    let (mut q, mut r) = (zeros(HEADS * HEAD_DIM)?, zeros(HEADS * REL_DIM)?);
    let mut bias = zeros(HEADS * EXTENT)?;
    let mut out = zeros(HEADS * HEAD_DIM)?;
    let (mut gated_ref, mut gated) = (zeros(HEADS * HEAD_DIM)?, zeros(HEADS * HEAD_DIM)?);

    let mut filled = 0;
    for (index, &position) in checks.iter().enumerate() {
        assert!(cache.holds(position));
        for p in filled..position {
            stream.memcpy_htod(&[u32::try_from(p)?], &mut position_dev)?;
            let row = (p % FILL) * ROW..(p % FILL + 1) * ROW;
            for c in [&mut cache_ref, &mut cache] {
                attention.append(
                    stream,
                    c,
                    &position_dev,
                    &fill_k.slice(row.clone()),
                    &fill_v.slice(row.clone()),
                )?;
            }
        }
        filled = position + 1;
        stream.memcpy_htod(&[u32::try_from(position)?], &mut position_dev)?;
        let dynamic = index % 2 == 1;
        // Scales chosen so some heads exceed the clip thresholds and some do not.
        let k_raw = stream.clone_htod(&rng.normals(ROW, 3.0))?;
        let v_raw = stream.clone_htod(&rng.normals(ROW, 3.0))?;
        let taps = [
            stream.clone_htod(&rng.normals(KV_HEADS * 4, 0.3))?,
            stream.clone_htod(&rng.normals(KV_HEADS * 4, 0.3))?,
        ];
        let q_raw = stream.clone_htod(&rng.normals(HEADS * HEAD_DIM, 2.0))?;
        let r_raw = stream.clone_htod(&rng.normals(HEADS * REL_DIM, 2.0))?;
        let gate = stream.clone_htod(&rng.normals(HEADS * HEAD_DIM, 2.0))?;
        let sinks = stream.clone_htod(&rng.normals(HEADS, 2.0))?;

        let [k_state, v_state] = &mut states_ref;
        norm.attention_post(
            stream,
            AttentionPost {
                k_raw: &k_raw,
                v_raw: &v_raw,
                conv: conv_weights(dynamic, &weights, &taps, &statics),
                k_state,
                v_state,
                q_raw: &q_raw,
                q_scale: &q_scale,
                r_raw: &r_raw,
                r_scale: &r_scale,
                k: &mut k,
                v: &mut v,
                q: &mut q,
                r: &mut r,
                log_scale: log_scale.then_some((&position_dev, LOG_SCALE.0, LOG_SCALE.1)),
                qk_threshold: QK_THRESHOLD,
                rel_threshold: REL_THRESHOLD,
                eps: EPS,
            },
        )?;
        attention.relative_bias(stream, &r, &table, &mut bias)?;
        attention.append(stream, &mut cache_ref, &position_dev, &k, &v)?;
        attention.decode_attention(
            stream,
            &q,
            &cache_ref,
            &position_dev,
            &bias,
            &sinks,
            &mut scratch_ref,
            &mut out,
        )?;
        elementwise.attention_gate(stream, &out, &gate, &mut gated_ref)?;

        let [k_state, v_state] = &mut states;
        attention.decode_step(
            stream,
            &mut cache,
            &position_dev,
            &mut scratch,
            DecodeStep {
                k_raw: &k_raw,
                v_raw: &v_raw,
                conv: conv_weights(dynamic, &weights, &taps, &statics),
                k_state,
                v_state,
                q_raw: &q_raw,
                q_scale: &q_scale,
                r_raw: &r_raw,
                r_scale: &r_scale,
                table: &table,
                sinks: &sinks,
                gate: &gate,
                log_scale: log_scale.then_some(LOG_SCALE),
                qk_threshold: QK_THRESHOLD,
                rel_threshold: REL_THRESHOLD,
                eps: EPS,
                out: &mut gated,
                min_split_slots: MIN_SPLIT_SLOTS,
            },
        )?;

        let label = format!("position {position}, dynamic {dynamic}");
        let want = stream.clone_dtoh(&gated_ref)?;
        assert!(
            want.iter().any(|x| x.to_f32() != 0.0),
            "{label}: output is all zero"
        );
        assert_bf16_close(&label, &stream.clone_dtoh(&gated)?, &want, 0);
        for (i, name) in ["k state", "v state"].into_iter().enumerate() {
            assert_f32_within_ulps(
                &format!("{label}: {name}"),
                &stream.clone_dtoh(&states[i])?,
                &stream.clone_dtoh(&states_ref[i])?,
                0,
            );
        }
        let slot = position % slots;
        assert!(
            cache_row(stream, &cache, slot)? == cache_row(stream, &cache_ref, slot)?,
            "{label}: appended row differs"
        );
    }
    Ok(())
}

#[test]
fn sliding_decode_step_matches_separate_kernels() -> anyhow::Result<()> {
    let checks = [
        0, 1, 31, 32, 33, 63, 64, 65, 500, 1023, 1024, 1025, 1500, 2047, 2048, 3001,
    ];
    check_decode_step(|stream| KvCache::sliding(stream, WINDOW), &checks)
}

#[test]
fn global_decode_step_matches_separate_kernels() -> anyhow::Result<()> {
    let checks = [0, 1, 31, 32, 33, 1023, 1024, 1025, 2561, 8191, 8192, 20_000];
    check_decode_step(|stream| KvCache::global(stream, 20_480), &checks)
}

#[test]
fn fp8_global_decode_step_matches_separate_kernels() -> anyhow::Result<()> {
    let checks = [
        0, 1, 31, 32, 63, 64, 65, 1023, 1024, 1025, 2561, 8191, 8192, 20_000,
    ];
    check_decode_step(
        |stream| KvCache::global_with_format(stream, 20_480, KvFormat::Fp8),
        &checks,
    )
}

/// One chunk of a prefill test: `rows` tokens, whether to also run decode attention on each
/// row and compare, and the rows to check against the F64 reference (all when `None`).
struct Chunk {
    rows: usize,
    decode: bool,
    checks: Option<Vec<usize>>,
}

impl Chunk {
    /// A chunk checked in full, against decode and the F64 reference.
    fn full(rows: usize) -> Self {
        Self {
            rows,
            decode: true,
            checks: None,
        }
    }

    /// A chunk that only fills the cache.
    fn fill(rows: usize) -> Self {
        Self {
            rows,
            decode: false,
            checks: Some(Vec::new()),
        }
    }

    /// A chunk compared with decode in full and with the F64 reference at `checks`.
    fn sampled(rows: usize, checks: Vec<usize>) -> Self {
        Self {
            rows,
            decode: true,
            checks: Some(checks),
        }
    }
}

/// Fails unless two caches hold the same bytes.
fn assert_same_cache(stream: &Arc<CudaStream>, a: &KvCache, b: &KvCache) -> anyhow::Result<()> {
    let same = match (a.data(), b.data()) {
        (
            KvData::Bf16 { keys, values },
            KvData::Bf16 {
                keys: keys_b,
                values: values_b,
            },
        ) => {
            stream.clone_dtoh(keys)? == stream.clone_dtoh(keys_b)?
                && stream.clone_dtoh(values)? == stream.clone_dtoh(values_b)?
        }
        (
            KvData::Fp8 { keys, values },
            KvData::Fp8 {
                keys: keys_b,
                values: values_b,
            },
        )
        | (
            KvData::Nvfp4 { keys, values },
            KvData::Nvfp4 {
                keys: keys_b,
                values: values_b,
            },
        ) => {
            stream.clone_dtoh(keys)? == stream.clone_dtoh(keys_b)?
                && stream.clone_dtoh(values)? == stream.clone_dtoh(values_b)?
        }
        _ => false,
    };
    assert!(same, "prefill and decode left different caches");
    Ok(())
}

/// Runs `chunks` of random tokens through prefill attention on one cache and through
/// append plus decode attention, row by row, on another made the same way. Checks the
/// prefill rows against the F64 reference, against decode, and the two caches' contents
/// after every chunk compared with decode.
fn check_prefill(
    make_cache: impl Fn(&Arc<CudaStream>) -> anyhow::Result<KvCache>,
    chunks: &[Chunk],
    seed: u64,
) -> anyhow::Result<()> {
    check_prefill_with(make_cache, chunks, seed, Products::Bf16)
}

/// [`check_prefill`] with the attention products of `products`. FP8 products are checked
/// against the reference on their own MXFP8 queries within [`PREFILL_FP8_ERROR`], and
/// report their RMS error.
#[expect(
    clippy::too_many_lines,
    reason = "one pass over the chunks keeps the two caches in step"
)]
fn check_prefill_with(
    make_cache: impl Fn(&Arc<CudaStream>) -> anyhow::Result<KvCache>,
    chunks: &[Chunk],
    seed: u64,
    products: Products,
) -> anyhow::Result<()> {
    let fp8_products = products == Products::Fp8;
    let (product_error, value_error) = if fp8_products {
        (PREFILL_FP8_ERROR, PREFILL_FP8_V_ERROR)
    } else {
        (PREFILL_P_ERROR, PREFILL_V_ERROR)
    };
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let attention = Attention::load(gpu.context())?;
    let mut prefill_cache = make_cache(stream)?;
    let mut decode_cache = make_cache(stream)?;
    let window = prefill_cache.window();
    let format = prefill_cache.format();
    let label = match (window, format) {
        (Some(_), _) => "sliding",
        (None, KvFormat::Bf16) => "global",
        (None, KvFormat::Fp8) if fp8_products => "fp8 global, fp8 products",
        (None, KvFormat::Fp8) => "fp8 global",
        (None, KvFormat::Nvfp4 | KvFormat::Nvfp4All | KvFormat::Mixed) => "nvfp4 global",
    };

    let tokens: usize = chunks.iter().map(|c| c.rows).sum();
    let mut rng = Rng(seed);
    let topics: Vec<Vec<f32>> = (0..4)
        .map(|_| (0..HEAD_DIM).map(|_| 1.5 * rng.normal()).collect())
        .collect();
    let k = rng.heads(tokens * KV_HEADS, &topics);
    let v = rng.normals(tokens * ROW, 1.0);
    let stored_k: Vec<f32> = k.iter().map(|&x| stored(format, x)).collect();
    let stored_v: Vec<f32> = v.iter().map(|&x| stored(format, x)).collect();
    let exact_k: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();
    let exact_v: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();
    let k_dev = stream.clone_htod(&k)?;
    let v_dev = stream.clone_htod(&v)?;

    let mut scratch = PrefillScratch::new(stream, MAX_PREFILL_ROWS)?;
    let mut decode_scratch = AttentionScratch::new(stream)?;
    let mut start_dev = stream.alloc_zeros::<u32>(1)?;
    let mut position_dev = stream.alloc_zeros::<u32>(1)?;
    let mut decode_out = stream.alloc_zeros::<bf16>(HEADS * HEAD_DIM)?;
    let head_len = HEADS * HEAD_DIM;
    let bias_len = HEADS * EXTENT;

    let mut start = 0;
    for chunk in chunks {
        let m = chunk.rows;
        let checks = chunk.checks.clone().unwrap_or_else(|| (0..m).collect());
        let inputs_needed = chunk.decode || !checks.is_empty();
        let (q, bias, sinks) = if inputs_needed {
            (
                rng.heads(m * HEADS, &topics),
                rng.normals(m * bias_len, 1.5),
                rng.normals(m * HEADS, 2.0),
            )
        } else {
            (
                rng.heads(m * HEADS, &topics),
                vec![bf16::ZERO; m * bias_len],
                vec![bf16::ZERO; m * HEADS],
            )
        };
        let q_dev = stream.clone_htod(&q)?;
        let bias_dev = stream.clone_htod(&bias)?;
        let sinks_dev = stream.clone_htod(&sinks)?;
        let mut out_dev = stream.alloc_zeros::<bf16>(m * head_len)?;
        stream.memcpy_htod(&[u32::try_from(start)?], &mut start_dev)?;
        let rows = start * ROW..(start + m) * ROW;
        attention.prefill_attention(
            stream,
            &q_dev,
            &k_dev.slice(rows.clone()),
            &v_dev.slice(rows),
            &mut prefill_cache,
            &start_dev,
            m,
            &bias_dev,
            &sinks_dev,
            &mut scratch,
            &mut out_dev,
            products,
        )?;
        let got = stream.clone_dtoh(&out_dev)?;

        let mut decoded = Vec::new();
        for i in 0..m {
            let position = start + i;
            let token = position * ROW..(position + 1) * ROW;
            stream.memcpy_htod(&[u32::try_from(position)?], &mut position_dev)?;
            attention.append(
                stream,
                &mut decode_cache,
                &position_dev,
                &k_dev.slice(token.clone()),
                &v_dev.slice(token),
            )?;
            if chunk.decode {
                let q_row = stream.clone_htod(&q[i * head_len..(i + 1) * head_len])?;
                let bias_row = stream.clone_htod(&bias[i * bias_len..(i + 1) * bias_len])?;
                let sinks_row = stream.clone_htod(&sinks[i * HEADS..(i + 1) * HEADS])?;
                attention.decode_attention(
                    stream,
                    &q_row,
                    &decode_cache,
                    &position_dev,
                    &bias_row,
                    &sinks_row,
                    &mut decode_scratch,
                    &mut decode_out,
                )?;
                decoded.extend(stream.clone_dtoh(&decode_out)?);
            }
        }

        let mut vs_reference = Closeness::default();
        let mut vs_decode = Closeness::default();
        for &i in &checks {
            let position = start + i;
            let query = Query {
                q: q[i * head_len..(i + 1) * head_len].to_vec(),
                bias: bias[i * bias_len..(i + 1) * bias_len].to_vec(),
                sinks: sinks[i * HEADS..(i + 1) * HEADS].to_vec(),
            };
            // A single row runs decode attention, which keeps the queries unquantized.
            let quantized_query = (fp8_products && m > 1).then(|| Query {
                q: mxfp8_queries(&query.q),
                ..query.clone()
            });
            let history = ..(position + 1) * ROW;
            let (want, magnitude) = reference(
                quantized_query.as_ref().unwrap_or(&query),
                &stored_k[history],
                &stored_v[history],
                window,
                position,
            );
            let row = &got[i * head_len..(i + 1) * head_len];
            let row_label = format!("{label} prefill {position}");
            vs_reference = vs_reference.merge(check_within(
                &format!("{row_label} vs F64"),
                row,
                &want,
                &magnitude,
                Tolerance {
                    absolute: value_error,
                    ..Tolerance::reference(F32_ERROR + product_error)
                },
            ));
            if fp8_products && i == checks[0] {
                report_quantization_error(&format!("{row_label} (stored keys)"), row, &want);
            }
            // Decode keeps the queries unquantized, so FP8 products are only checked against
            // the reference on their own queries.
            if chunk.decode && !fp8_products {
                let decoded_row = &decoded[i * head_len..(i + 1) * head_len];
                check_within(
                    &format!("{label} decode {position} vs F64"),
                    decoded_row,
                    &want,
                    &magnitude,
                    Tolerance::reference(F32_ERROR),
                );
                let decoded_row: Vec<f64> = decoded_row.iter().map(|x| x.to_f64()).collect();
                vs_decode = vs_decode.merge(check_within(
                    &format!("{row_label} vs decode"),
                    row,
                    &decoded_row,
                    &magnitude,
                    Tolerance {
                        ulps: 2.0,
                        relative: 2.0 * F32_ERROR + product_error,
                        absolute: value_error,
                    },
                ));
            }
            if format == KvFormat::Fp8 && i == checks[0] {
                let (exact, _) = reference(
                    &query,
                    &exact_k[history],
                    &exact_v[history],
                    window,
                    position,
                );
                report_quantization_error(&row_label, row, &exact);
            }
        }
        if !checks.is_empty() {
            eprintln!(
                "{label} chunk {start}+{m}: {} rows vs F64: {} of {} elements differ from the \
                 rounded value, worst {:.0}% of tolerance, largest error {:.2e} of magnitude",
                checks.len(),
                vs_reference.differing,
                checks.len() * head_len,
                100.0 * vs_reference.worst,
                vs_reference.relative,
            );
        }
        if chunk.decode {
            eprintln!(
                "{label} chunk {start}+{m}: vs decode: {} of {} elements differ, worst {:.0}% of \
                 tolerance, largest difference {:.2e} of magnitude",
                vs_decode.differing,
                checks.len() * head_len,
                100.0 * vs_decode.worst,
                vs_decode.relative,
            );
            assert_same_cache(stream, &prefill_cache, &decode_cache)?;
        }
        start += m;
    }
    assert_same_cache(stream, &prefill_cache, &decode_cache)
}

/// Rows spread over a chunk of `rows`: both ends and every 61st row.
fn spread(rows: usize) -> Vec<usize> {
    let mut checks: Vec<usize> = (0..rows).step_by(61).collect();
    checks.extend([1, rows / 2, rows - 2, rows - 1]);
    checks.sort_unstable();
    checks.dedup();
    checks.retain(|&r| r < rows);
    checks
}

/// A chunk at 0 with an odd row count, a single row (which runs decode attention) and two
/// rows, a full 1024-row chunk that wraps the ring, a 512-row chunk on a full ring, more odd
/// chunks, and chunks longer than the window on a full ring.
#[test]
fn sliding_prefill_attention_matches_decode() -> anyhow::Result<()> {
    let chunks = [
        Chunk::full(77),
        Chunk::full(1),
        Chunk::full(2),
        Chunk::sampled(1024, spread(1024)),
        Chunk::sampled(512, spread(512)),
        Chunk::full(1),
        Chunk::full(301),
        Chunk::sampled(1023, spread(1023)),
        Chunk::sampled(2048, spread(2048)),
        Chunk::sampled(1500, spread(1500)),
    ];
    check_prefill(|stream| KvCache::sliding(stream, WINDOW), &chunks, 4)
}

/// A first chunk longer than the window, from an empty ring, then a short one.
#[test]
fn long_sliding_prefill_chunk_matches_decode() -> anyhow::Result<()> {
    let chunks = [Chunk::sampled(2048, spread(2048)), Chunk::full(5)];
    check_prefill(|stream| KvCache::sliding(stream, WINDOW), &chunks, 10)
}

/// A full 1024-row chunk at 0, then odd and single-row chunks, and a 2048-row chunk.
#[test]
fn global_prefill_attention_matches_decode() -> anyhow::Result<()> {
    let chunks = [
        Chunk::sampled(1024, spread(1024)),
        Chunk::full(1),
        Chunk::full(5),
        Chunk::full(77),
        Chunk::sampled(512, spread(512)),
        Chunk::full(3),
        Chunk::sampled(2048, spread(2048)),
    ];
    check_prefill(|stream| KvCache::global(stream, 4096), &chunks, 5)
}

#[test]
fn fp8_global_prefill_attention_matches_decode() -> anyhow::Result<()> {
    let chunks = [
        Chunk::full(77),
        Chunk::full(1),
        Chunk::full(2),
        Chunk::sampled(1024, spread(1024)),
        Chunk::sampled(512, spread(512)),
        Chunk::sampled(2048, spread(2048)),
    ];
    check_prefill(
        |stream| KvCache::global_with_format(stream, 4096, KvFormat::Fp8),
        &chunks,
        6,
    )
}

#[test]
fn fp8_product_prefill_attention_matches_decode() -> anyhow::Result<()> {
    let chunks = [
        Chunk::full(77),
        Chunk::full(1),
        Chunk::full(2),
        Chunk::sampled(1024, spread(1024)),
        Chunk::sampled(512, spread(512)),
        Chunk::sampled(2048, spread(2048)),
    ];
    check_prefill_with(
        |stream| KvCache::global_with_format(stream, 4096, KvFormat::Fp8),
        &chunks,
        6,
        Products::Fp8,
    )
}

/// Chunks past 39,000 tokens, beyond the bias table and the decode kernel's split sizes.
fn long_chunks() -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = (0..38).map(|_| Chunk::fill(1024)).collect();
    chunks.push(Chunk::fill(1000));
    chunks.push(Chunk::sampled(512, vec![0, 1, 255, 510, 511]));
    chunks.push(Chunk::sampled(1, vec![0]));
    chunks.push(Chunk::sampled(2, vec![0, 1]));
    chunks.push(Chunk::sampled(333, vec![0, 332]));
    chunks
}

#[test]
fn long_global_prefill_attention_matches_decode() -> anyhow::Result<()> {
    check_prefill(|stream| KvCache::global(stream, 40_960), &long_chunks(), 7)
}

#[test]
fn long_fp8_global_prefill_attention_matches_decode() -> anyhow::Result<()> {
    check_prefill(
        |stream| KvCache::global_with_format(stream, 40_960, KvFormat::Fp8),
        &long_chunks(),
        8,
    )
}

#[test]
fn long_fp8_product_prefill_attention_matches_decode() -> anyhow::Result<()> {
    check_prefill_with(
        |stream| KvCache::global_with_format(stream, 40_960, KvFormat::Fp8),
        &long_chunks(),
        8,
        Products::Fp8,
    )
}

/// Checks `prefill_attention_fp4_mma` (`Products::Fp4` on an NVFP4 global cache) against:
/// 1. The F64 host reference on its own NVFP4-quantized queries and stored NVFP4 KV cache,
/// 2. The exact F16 `prefill_attention_fp4` (`Products::Bf16`) on the same NVFP4 KV cache,
/// 3. The unquantized BF16 `prefill_attention_bf16` reference.
///
/// Verifies byte-identical NVFP4 cache contents, strict F64 error bounds, and cosine /
/// max-abs parity tolerances.
#[expect(
    clippy::too_many_lines,
    reason = "runs three caches in lockstep over the chunks"
)]
fn check_prefill_fp4_mma(capacity: usize, chunks: &[Chunk], seed: u64) -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let attention = Attention::load(gpu.context())?;
    let mut fp4_mma_cache = KvCache::global_with_format(stream, capacity, KvFormat::Nvfp4)?;
    let mut fp4_ref_cache = KvCache::global_with_format(stream, capacity, KvFormat::Nvfp4)?;
    let mut bf16_ref_cache = KvCache::global_with_format(stream, capacity, KvFormat::Bf16)?;

    let tokens: usize = chunks.iter().map(|c| c.rows).sum();
    let mut rng = Rng(seed);
    let topics: Vec<Vec<f32>> = (0..4)
        .map(|_| (0..HEAD_DIM).map(|_| 1.5 * rng.normal()).collect())
        .collect();
    let k = rng.heads(tokens * KV_HEADS, &topics);
    let v = rng.normals(tokens * ROW, 1.0);
    let stored_k = nvfp4_stored(&k);
    let stored_v = nvfp4_stored(&v);
    let k_dev = stream.clone_htod(&k)?;
    let v_dev = stream.clone_htod(&v)?;

    let mut scratch = PrefillScratch::new(stream, MAX_PREFILL_ROWS)?;
    let mut start_dev = stream.alloc_zeros::<u32>(1)?;
    let head_len = HEADS * HEAD_DIM;
    let bias_len = HEADS * EXTENT;

    let mut start = 0;
    for chunk in chunks {
        let m = chunk.rows;
        let checks = chunk.checks.clone().unwrap_or_else(|| (0..m).collect());
        let (q, bias, sinks) = if checks.is_empty() {
            (
                rng.heads(m * HEADS, &topics),
                vec![bf16::ZERO; m * bias_len],
                vec![bf16::ZERO; m * HEADS],
            )
        } else {
            (
                rng.heads(m * HEADS, &topics),
                rng.normals(m * bias_len, 1.5),
                rng.normals(m * HEADS, 2.0),
            )
        };
        let q_dev = stream.clone_htod(&q)?;
        let bias_dev = stream.clone_htod(&bias)?;
        let sinks_dev = stream.clone_htod(&sinks)?;
        let mut out_mma = stream.alloc_zeros::<bf16>(m * head_len)?;
        let mut out_fp4 = stream.alloc_zeros::<bf16>(m * head_len)?;
        let mut out_bf16 = stream.alloc_zeros::<bf16>(m * head_len)?;
        stream.memcpy_htod(&[u32::try_from(start)?], &mut start_dev)?;
        let rows = start * ROW..(start + m) * ROW;

        attention.prefill_attention(
            stream,
            &q_dev,
            &k_dev.slice(rows.clone()),
            &v_dev.slice(rows.clone()),
            &mut fp4_mma_cache,
            &start_dev,
            m,
            &bias_dev,
            &sinks_dev,
            &mut scratch,
            &mut out_mma,
            Products::Fp4,
        )?;
        attention.prefill_attention(
            stream,
            &q_dev,
            &k_dev.slice(rows.clone()),
            &v_dev.slice(rows.clone()),
            &mut fp4_ref_cache,
            &start_dev,
            m,
            &bias_dev,
            &sinks_dev,
            &mut scratch,
            &mut out_fp4,
            Products::Bf16,
        )?;
        attention.prefill_attention(
            stream,
            &q_dev,
            &k_dev.slice(rows.clone()),
            &v_dev.slice(rows),
            &mut bf16_ref_cache,
            &start_dev,
            m,
            &bias_dev,
            &sinks_dev,
            &mut scratch,
            &mut out_bf16,
            Products::Bf16,
        )?;

        if !checks.is_empty() {
            let got_mma = stream.clone_dtoh(&out_mma)?;
            let got_fp4 = stream.clone_dtoh(&out_fp4)?;
            let got_bf16 = stream.clone_dtoh(&out_bf16)?;
            let mut vs_reference = Closeness::default();
            let mut min_cos_fp4 = 1.0f64;
            let mut max_abs_fp4 = 0.0f64;
            let mut min_cos_bf16 = 1.0f64;
            let mut min_cos_ref_bf16 = 1.0f64;
            let mut max_abs_bf16 = 0.0f64;
            for &i in &checks {
                let position = start + i;
                let span = i * head_len..(i + 1) * head_len;
                let query = Query {
                    q: nvfp4_queries(&q[span.clone()]),
                    bias: bias[i * bias_len..(i + 1) * bias_len].to_vec(),
                    sinks: sinks[i * HEADS..(i + 1) * HEADS].to_vec(),
                };
                let history = ..(position + 1) * ROW;
                let (want, magnitude) = reference(
                    &query,
                    &stored_k[history],
                    &stored_v[history],
                    None,
                    position,
                );
                let row_mma = &got_mma[span.clone()];
                let row_fp4 = &got_fp4[span.clone()];
                let row_bf16 = &got_bf16[span];
                vs_reference = vs_reference.merge(check_within(
                    &format!("nvfp4_mma prefill {position} vs F64"),
                    row_mma,
                    &want,
                    &magnitude,
                    Tolerance {
                        absolute: PREFILL_FP8_V_ERROR,
                        ..Tolerance::reference(F32_ERROR + PREFILL_FP4_ERROR)
                    },
                ));
                let (cos_fp4, abs_fp4) = cosine_and_max_abs(row_mma, row_fp4);
                let (cos_bf16, abs_bf16) = cosine_and_max_abs(row_mma, row_bf16);
                let (cos_ref_bf16, _) = cosine_and_max_abs(row_fp4, row_bf16);
                min_cos_fp4 = min_cos_fp4.min(cos_fp4);
                max_abs_fp4 = max_abs_fp4.max(abs_fp4);
                min_cos_bf16 = min_cos_bf16.min(cos_bf16);
                min_cos_ref_bf16 = min_cos_ref_bf16.min(cos_ref_bf16);
                max_abs_bf16 = max_abs_bf16.max(abs_bf16);
                assert!(
                    cos_fp4 >= 0.985,
                    "nvfp4_mma pos {position} vs nvfp4_f16: cosine {cos_fp4:.5} < 0.985 \
                     (max_abs {abs_fp4:.4})"
                );
                assert!(
                    cos_bf16 >= cos_ref_bf16 - 0.025,
                    "nvfp4_mma pos {position} vs bf16: cosine {cos_bf16:.5} dropped more than \
                     0.025 below nvfp4_f16 vs bf16 ({cos_ref_bf16:.5}, max_abs {abs_bf16:.4})"
                );
            }
            eprintln!(
                "nvfp4_mma chunk {start}+{m}: {} rows vs F64 worst {:.0}% tol; \
                 vs nvfp4_f16 min_cos={min_cos_fp4:.5} max_abs={max_abs_fp4:.4}; \
                 vs bf16 min_cos={min_cos_bf16:.5} (nvfp4_f16={min_cos_ref_bf16:.5}) \
                 max_abs={max_abs_bf16:.4}",
                checks.len(),
                100.0 * vs_reference.worst,
            );
        }
        start += m;
    }
    assert_same_cache(stream, &fp4_mma_cache, &fp4_ref_cache)
}

#[test]
#[cfg_attr(sm89, ignore = "needs the block-scaled MMA, which sm_89 lacks")]
fn fp4_product_prefill_attention_matches_reference() -> anyhow::Result<()> {
    let chunks = [
        Chunk::full(77),
        Chunk::full(1),
        Chunk::full(2),
        Chunk::sampled(1024, spread(1024)),
        Chunk::sampled(512, spread(512)),
        Chunk::sampled(2048, spread(2048)),
    ];
    check_prefill_fp4_mma(4096, &chunks, 12)
}

#[test]
#[cfg_attr(sm89, ignore = "needs the block-scaled MMA, which sm_89 lacks")]
fn long_fp4_product_prefill_attention_matches_reference() -> anyhow::Result<()> {
    check_prefill_fp4_mma(40_960, &long_chunks(), 13)
}

/// Throughput benchmark comparing `prefill_attention` variants (BF16, FP8, FP8-MMA,
/// NVFP4-F16, NVFP4-MMA) across sequence lengths `L = 2048, 4096, 8192` (with 2048-row
/// chunks ending at `L`, plus sliding window). Run with `--ignored --nocapture` on an
/// idle Blackwell GPU.
#[test]
#[ignore = "timing, not a check"]
#[expect(clippy::too_many_lines, reason = "benchmark sweep table")]
fn prefill_attention_fp4_timing() -> anyhow::Result<()> {
    use cudarc::driver::sys::CUevent_flags;

    const CALLS: usize = 5;
    const BATCHES: usize = 8;
    const CHUNK: usize = 2048;

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let attention = Attention::load(gpu.context())?;
    let mut scratch = PrefillScratch::new(stream, MAX_PREFILL_ROWS)?;
    let mut start_dev = stream.alloc_zeros::<u32>(1)?;
    let mut rng = Rng(14);
    let table = stream.clone_htod(&rng.normals(REL_DIM * EXTENT, 0.4))?;
    let q = stream.clone_htod(&rng.normals(CHUNK * HEADS * HEAD_DIM, 1.0))?;
    let k = stream.clone_htod(&rng.normals(CHUNK * ROW, 1.0))?;
    let v = stream.clone_htod(&rng.normals(CHUNK * ROW, 1.0))?;
    let rel = stream.clone_htod(&rng.normals(CHUNK * HEADS * REL_DIM, 1.0))?;
    let mut bias = stream.alloc_zeros::<bf16>(CHUNK * HEADS * EXTENT)?;
    attention.relative_bias_rows(stream, &rel, &table, &mut bias)?;
    let sinks = stream.clone_htod(&rng.normals(CHUNK * HEADS, 1.0))?;
    let mut out = stream.alloc_zeros::<bf16>(CHUNK * HEADS * HEAD_DIM)?;

    let begin = gpu
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = gpu
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    eprintln!("| variant | L | rows | start | us/call | TFLOP/s | KV GB/s |");
    eprintln!("| --- | --- | --- | --- | --- | --- | --- |");

    let variants: &[(&str, Option<KvFormat>, Products)] = &[
        ("global bf16", Some(KvFormat::Bf16), Products::Bf16),
        ("global fp8", Some(KvFormat::Fp8), Products::Bf16),
        ("global fp8_mma", Some(KvFormat::Fp8), Products::Fp8),
        ("global nvfp4", Some(KvFormat::Nvfp4), Products::Bf16),
        ("global nvfp4_mma", Some(KvFormat::Nvfp4), Products::Fp4),
        ("sliding bf16", None, Products::Bf16),
        ("sliding fp8_mma", None, Products::Fp8),
    ];

    for seq_len in [2048usize, 4096, 8192] {
        let start = seq_len - CHUNK;
        stream.memcpy_htod(&[u32::try_from(start)?], &mut start_dev)?;
        for &(name, format, products) in variants {
            if !siphon_cuda::BLOCK_SCALED_MMA && products == Products::Fp4 {
                continue;
            }
            let mut cache = match (format, products) {
                (None, Products::Fp8) => {
                    KvCache::sliding_ring_with_format(stream, WINDOW, WINDOW, KvFormat::Fp8)?
                }
                (None, _) => KvCache::sliding(stream, WINDOW)?,
                (Some(fmt), _) => KvCache::global_with_format(stream, seq_len, fmt)?,
            };
            let mut run = || {
                attention.prefill_attention(
                    stream,
                    &q,
                    &k,
                    &v,
                    &mut cache,
                    &start_dev,
                    CHUNK,
                    &bias,
                    &sinks,
                    &mut scratch,
                    &mut out,
                    products,
                )
            };
            run()?;
            let mut best = f64::INFINITY;
            for _ in 0..BATCHES {
                begin.record(stream)?;
                for _ in 0..CALLS {
                    run()?;
                }
                end.record(stream)?;
                best = best.min(f64::from(begin.elapsed_ms(&end)?) * 1e-3 / CALLS as f64);
            }
            let seen = |i: usize| {
                if format.is_none() {
                    (start + i + 1).min(WINDOW)
                } else {
                    start + i + 1
                }
            };
            let pairs: usize = (0..CHUNK).map(seen).sum();
            let flops = (pairs * HEADS * HEAD_DIM * 4) as f64;
            let distinct = if format.is_none() {
                seq_len.min(WINDOW + CHUNK - 1)
            } else {
                seq_len
            };
            let row_bytes = match (format, products) {
                (None, Products::Fp8) => KvFormat::Fp8.row_bytes(ROW),
                (None, _) => 2 * ROW,
                (Some(fmt), _) => fmt.row_bytes(ROW),
            };
            let bytes = (distinct * 2 * row_bytes) as f64;
            eprintln!(
                "| {name} | {seq_len} | {CHUNK} | {start} | {:.1} | {:.1} | {:.0} |",
                best * 1e6,
                flops / best * 1e-12,
                bytes / best * 1e-9,
            );
        }
    }
    Ok(())
}

/// Time per prefill call (append included) for the cases in the report. Run with
/// `--ignored --nocapture` on an idle GPU. Each case reports the fastest of several short
/// batches, which drops batches that shared the GPU with another process.
#[test]
#[ignore = "timing, not a check"]
#[expect(clippy::too_many_lines, reason = "one table of cases")]
fn prefill_attention_timing() -> anyhow::Result<()> {
    use cudarc::driver::sys::CUevent_flags;

    /// BF16 dense tensor-core rate of the RTX 5090, FLOP/s.
    const PEAK_FLOPS: f64 = 237e12;
    /// DRAM bandwidth of the RTX 5090, bytes/s.
    const PEAK_BYTES: f64 = 1.79e12;
    const CALLS: usize = 5;
    const BATCHES: usize = 8;

    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let attention = Attention::load(gpu.context())?;
    let mut scratch = PrefillScratch::new(stream, MAX_PREFILL_ROWS)?;
    let mut decode_scratch = AttentionScratch::new(stream)?;
    let mut start_dev = stream.alloc_zeros::<u32>(1)?;
    let mut rng = Rng(9);
    let table = stream.clone_htod(&rng.normals(REL_DIM * EXTENT, 0.4))?;

    eprintln!("| case | rows | start | us/call | TFLOP/s | % of 237 | KV GB/s | % of 1.79 TB/s |");
    eprintln!("| --- | --- | --- | --- | --- | --- | --- | --- |");
    let cases: &[(&str, Option<KvFormat>, usize, usize)] = &[
        ("sliding", None, 512, 0),
        ("sliding", None, 512, 32_768),
        ("sliding", None, 1024, 32_768),
        ("sliding", None, 2048, 0),
        ("sliding", None, 2048, 32_768),
        ("sliding", None, 1, 32_768),
        ("global bf16", Some(KvFormat::Bf16), 512, 0),
        ("global bf16", Some(KvFormat::Bf16), 512, 8192),
        ("global bf16", Some(KvFormat::Bf16), 512, 32_768),
        ("global bf16", Some(KvFormat::Bf16), 512, 131_072),
        ("global bf16", Some(KvFormat::Bf16), 1024, 32_768),
        ("global bf16", Some(KvFormat::Bf16), 2048, 32_768),
        ("global bf16", Some(KvFormat::Bf16), 2048, 131_072),
        ("global bf16", Some(KvFormat::Bf16), 1, 32_768),
        ("global bf16", Some(KvFormat::Bf16), 1, 131_072),
        ("global fp8", Some(KvFormat::Fp8), 512, 0),
        ("global fp8", Some(KvFormat::Fp8), 512, 8192),
        ("global fp8", Some(KvFormat::Fp8), 512, 32_768),
        ("global fp8", Some(KvFormat::Fp8), 512, 131_072),
        ("global fp8", Some(KvFormat::Fp8), 1, 32_768),
        ("global fp8", Some(KvFormat::Fp8), 1, 131_072),
    ];
    // `SIPHON_TIMING_ONLY` keeps the cases whose name starts with it.
    let only = std::env::var("SIPHON_TIMING_ONLY").ok();
    for &(name, format, rows, start) in cases {
        if only.as_deref().is_some_and(|o| !name.starts_with(o)) {
            continue;
        }
        let mut cache = match format {
            None => KvCache::sliding(stream, WINDOW)?,
            Some(format) => KvCache::global_with_format(stream, start + rows, format)?,
        };
        let q = stream.clone_htod(&rng.normals(rows * HEADS * HEAD_DIM, 1.0))?;
        let k = stream.clone_htod(&rng.normals(rows * ROW, 1.0))?;
        let v = stream.clone_htod(&rng.normals(rows * ROW, 1.0))?;
        let rel = stream.clone_htod(&rng.normals(rows * HEADS * REL_DIM, 1.0))?;
        let mut bias = stream.alloc_zeros::<bf16>(rows * HEADS * EXTENT)?;
        attention.relative_bias_rows(stream, &rel, &table, &mut bias)?;
        let sinks = stream.clone_htod(&rng.normals(rows * HEADS, 1.0))?;
        let mut out = stream.alloc_zeros::<bf16>(rows * HEADS * HEAD_DIM)?;
        stream.memcpy_htod(&[u32::try_from(start)?], &mut start_dev)?;
        let mut call = |cache: &mut KvCache| {
            attention.prefill_attention(
                stream,
                &q,
                &k,
                &v,
                cache,
                &start_dev,
                rows,
                &bias,
                &sinks,
                &mut scratch,
                &mut out,
                Products::Bf16,
            )
        };
        let begin = gpu
            .context()
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = gpu
            .context()
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        // Seconds per call of `f`, the fastest of the batches.
        let time = |f: &mut dyn FnMut() -> anyhow::Result<()>| -> anyhow::Result<f64> {
            f()?;
            let mut best = f64::INFINITY;
            for _ in 0..BATCHES {
                begin.record(stream)?;
                for _ in 0..CALLS {
                    f()?;
                }
                end.record(stream)?;
                best = best.min(f64::from(begin.elapsed_ms(&end)?) * 1e-3 / CALLS as f64);
            }
            Ok(best)
        };
        let seconds = time(&mut || call(&mut cache))?;

        // Keys each row sees, and the distinct keys the call reads.
        let seen = |i: usize| match format {
            None => (start + i + 1).min(WINDOW),
            Some(_) => start + i + 1,
        };
        let pairs: usize = (0..rows).map(seen).sum();
        let flops = (pairs * HEADS * HEAD_DIM * 4) as f64;
        let distinct = match format {
            None => (start + rows).min(WINDOW + rows - 1),
            Some(_) => start + rows,
        };
        let row_bytes = format.map_or(2 * ROW, |f| f.row_bytes(ROW));
        let bytes = (distinct * 2 * row_bytes) as f64;
        eprintln!(
            "| {name} | {rows} | {start} | {:.1} | {:.1} | {:.0}% | {:.0} | {:.0}% |",
            seconds * 1e6,
            flops / seconds * 1e-12,
            100.0 * flops / seconds / PEAK_FLOPS,
            bytes / seconds * 1e-9,
            100.0 * bytes / seconds / PEAK_BYTES,
        );

        if format.is_none() && start > 0 && rows > 1 {
            // The chunk's relative bias, which runs before prefill attention: the multi-row
            // kernel, and the single-row kernel launched over all rows at once.
            let written = (rows * HEADS * EXTENT * 2) as f64;
            let rows_kernel =
                time(&mut || attention.relative_bias_rows(stream, &rel, &table, &mut bias))?;
            let single_kernel =
                time(&mut || attention.relative_bias(stream, &rel, &table, &mut bias))?;
            for (label, seconds) in [
                ("relative_bias_rows", rows_kernel),
                ("relative_bias", single_kernel),
            ] {
                eprintln!(
                    "| {label} | {rows} | | {:.1} | | | {:.0} written | {:.0}% |",
                    seconds * 1e6,
                    written / seconds * 1e-9,
                    100.0 * written / seconds / PEAK_BYTES,
                );
            }
        }

        if rows == 1 {
            let position = stream.clone_htod(&[u32::try_from(start)?])?;
            let mut decode_out = stream.alloc_zeros::<bf16>(HEADS * HEAD_DIM)?;
            let mut decode = |cache: &mut KvCache| -> anyhow::Result<()> {
                attention.append(stream, cache, &position, &k, &v)?;
                attention.decode_attention(
                    stream,
                    &q,
                    cache,
                    &position,
                    &bias,
                    &sinks,
                    &mut decode_scratch,
                    &mut decode_out,
                )
            };
            let seconds = time(&mut || decode(&mut cache))?;
            eprintln!(
                "| {name}, append + decode | 1 | {start} | {:.1} | | | {:.0} | {:.0}% |",
                seconds * 1e6,
                bytes / seconds * 1e-9,
                100.0 * bytes / seconds / PEAK_BYTES,
            );
        }
    }
    Ok(())
}

/// Time per layer of the decode step between the projections and the output projection:
/// the separate kernels and `Attention::decode_step`, each captured 20 times in a CUDA graph
/// as the model replays them. Run with `--ignored --nocapture` on an idle GPU; each case
/// reports the fastest of several graph launches.
#[test]
#[ignore = "timing, not a check"]
#[expect(clippy::too_many_lines, reason = "one table of cases")]
fn decode_step_timing() -> anyhow::Result<()> {
    use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};

    const CALLS: usize = 20;
    const LAUNCHES: usize = 20;
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let attention = Attention::load(gpu.context())?;
    let norm = Norm::load(gpu.context())?;
    let elementwise = Elementwise::load(gpu.context())?;
    let mut rng = Rng(5);
    let mut upload = |len: usize, scale: f32| stream.clone_htod(&rng.normals(len, scale));
    let table = upload(REL_DIM * EXTENT, 0.4)?;
    let q_scale = upload(HEADS * HEAD_DIM, 0.5)?;
    let r_scale = upload(HEADS * REL_DIM, 0.5)?;
    let weights = [upload(ROW * 4, 0.5)?, upload(ROW * 4, 0.5)?];
    let taps = [upload(KV_HEADS * 4, 0.3)?, upload(KV_HEADS * 4, 0.3)?];
    let k_raw = upload(ROW, 3.0)?;
    let v_raw = upload(ROW, 3.0)?;
    let q_raw = upload(HEADS * HEAD_DIM, 2.0)?;
    let r_raw = upload(HEADS * REL_DIM, 2.0)?;
    let gate = upload(HEADS * HEAD_DIM, 2.0)?;
    let sinks = upload(HEADS, 2.0)?;
    let statics = [
        stream.clone_htod(&pseudo_random(4 * ROW, 64, 0.5))?,
        stream.clone_htod(&pseudo_random(4 * ROW, 65, 0.5))?,
    ];
    let mut states = [
        stream.alloc_zeros::<f32>(3 * ROW)?,
        stream.alloc_zeros::<f32>(3 * ROW)?,
    ];
    let mut scratch = AttentionScratch::new(stream)?;
    let zeros = |n: usize| -> anyhow::Result<CudaSlice<bf16>> { Ok(stream.alloc_zeros(n)?) };
    let (mut k, mut v) = (zeros(ROW)?, zeros(ROW)?);
    let (mut q, mut r) = (zeros(HEADS * HEAD_DIM)?, zeros(HEADS * REL_DIM)?);
    let mut bias = zeros(HEADS * EXTENT)?;
    let mut out = zeros(HEADS * HEAD_DIM)?;
    let mut gated = zeros(HEADS * HEAD_DIM)?;
    let begin = gpu
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = gpu
        .context()
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;

    eprintln!("| case | position | separate us | fused us |");
    eprintln!("| --- | --- | --- | --- |");
    let cases: &[(&str, Option<KvFormat>, usize)] = &[
        ("sliding", None, 400),
        ("sliding", None, 40_000),
        ("global bf16", Some(KvFormat::Bf16), 400),
        ("global bf16", Some(KvFormat::Bf16), 32_768),
        ("global bf16", Some(KvFormat::Bf16), 131_072),
        ("global fp8", Some(KvFormat::Fp8), 400),
        ("global fp8", Some(KvFormat::Fp8), 32_768),
        ("global fp8", Some(KvFormat::Fp8), 131_072),
    ];
    for &(name, format, position) in cases {
        let mut cache = match format {
            None => KvCache::sliding(stream, WINDOW)?,
            Some(format) => KvCache::global_with_format(stream, position + 1, format)?,
        };
        let log_scale = format.is_some();
        let position_dev = stream.clone_htod(&[u32::try_from(position)?])?;
        let mut time = |fused: bool| -> anyhow::Result<f64> {
            stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
            let mut enqueue = || -> anyhow::Result<()> {
                for _ in 0..CALLS {
                    let [k_state, v_state] = &mut states;
                    if fused {
                        attention.decode_step(
                            stream,
                            &mut cache,
                            &position_dev,
                            &mut scratch,
                            DecodeStep {
                                k_raw: &k_raw,
                                v_raw: &v_raw,
                                conv: conv_weights(log_scale, &weights, &taps, &statics),
                                k_state,
                                v_state,
                                q_raw: &q_raw,
                                q_scale: &q_scale,
                                r_raw: &r_raw,
                                r_scale: &r_scale,
                                table: &table,
                                sinks: &sinks,
                                gate: &gate,
                                log_scale: log_scale.then_some(LOG_SCALE),
                                qk_threshold: QK_THRESHOLD,
                                rel_threshold: REL_THRESHOLD,
                                eps: EPS,
                                out: &mut gated,
                                min_split_slots: MIN_SPLIT_SLOTS,
                            },
                        )?;
                    } else {
                        norm.attention_post(
                            stream,
                            AttentionPost {
                                k_raw: &k_raw,
                                v_raw: &v_raw,
                                conv: conv_weights(log_scale, &weights, &taps, &statics),
                                k_state,
                                v_state,
                                q_raw: &q_raw,
                                q_scale: &q_scale,
                                r_raw: &r_raw,
                                r_scale: &r_scale,
                                k: &mut k,
                                v: &mut v,
                                q: &mut q,
                                r: &mut r,
                                log_scale: log_scale.then_some((
                                    &position_dev,
                                    LOG_SCALE.0,
                                    LOG_SCALE.1,
                                )),
                                qk_threshold: QK_THRESHOLD,
                                rel_threshold: REL_THRESHOLD,
                                eps: EPS,
                            },
                        )?;
                        attention.relative_bias(stream, &r, &table, &mut bias)?;
                        attention.append(stream, &mut cache, &position_dev, &k, &v)?;
                        attention.decode_attention(
                            stream,
                            &q,
                            &cache,
                            &position_dev,
                            &bias,
                            &sinks,
                            &mut scratch,
                            &mut out,
                        )?;
                        elementwise.attention_gate(stream, &out, &gate, &mut gated)?;
                    }
                }
                Ok(())
            };
            let enqueued = enqueue();
            let graph = stream.end_capture(CUgraphInstantiate_flags(0));
            enqueued?;
            let graph = graph?.expect("the capture recorded work");
            graph.upload()?;
            graph.launch()?;
            let mut best = f64::INFINITY;
            for _ in 0..LAUNCHES {
                begin.record(stream)?;
                graph.launch()?;
                end.record(stream)?;
                best = best.min(f64::from(begin.elapsed_ms(&end)?) * 1e3 / CALLS as f64);
            }
            Ok(best)
        };
        let separate = time(false)?;
        let fused = time(true)?;
        eprintln!("| {name} | {position} | {separate:.2} | {fused:.2} |");
    }
    Ok(())
}
