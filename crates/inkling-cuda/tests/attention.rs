//! Attention kernels against MLX outputs from `reference/fixtures/attention.py` and against an
//! F64 reference on the host.
//!
//! The fixture feeds 1100 tokens into a sliding (1024-slot ring) and a global cache and records
//! MLX's decode attention at a few positions across the window and the bias table width. The
//! host reference covers what the fixture is too short for: a global cache of 40,000 keys cut
//! into the full number of splits, and a sliding ring that has wrapped several times.
//!
//! FP8 global caches are checked against the same F64 reference applied to the E4M3 values
//! the cache holds, with the BF16 tolerance: decoding E4M3 is exact, so only the rounding of
//! the cache differs. The tests also print the error against the reference on the BF16
//! values, which is the quantization error.

#![cfg(target_os = "linux")]

mod common;

use std::sync::OnceLock;

use common::{Fixture, assert_bf16_close, bf16_ulps};
use half::bf16;
use inkling_cuda::{
    Gpu,
    ops::attention::{
        Attention, AttentionScratch, HEAD_DIM, HEADS, KV_HEADS, KvCache, KvData, KvFormat,
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

/// Fails unless every output is within 1 BF16 ulp of `want` (the output rounding) plus
/// `relative` times its magnitude from [`reference`] (the arithmetic before it).
fn assert_within(label: &str, got: &[bf16], want: &[f64], magnitude: &[f64], relative: f64) {
    let mut worst = 0.0f64;
    let mut differing = 0;
    for (i, ((&g, &w), &a)) in got.iter().zip(want).zip(magnitude).enumerate() {
        let rounded = bf16::from_f64(w);
        let ulp = (bf16::from_bits(rounded.to_bits() + 1).to_f64() - rounded.to_f64()).abs();
        let allowed = ulp + relative * a;
        let err = (g.to_f64() - w).abs();
        assert!(
            err <= allowed,
            "{label}[{i}]: got {g}, want {w} (magnitude {a})"
        );
        worst = worst.max(err / allowed);
        differing += usize::from(bf16_ulps(g, rounded) > 0);
    }
    eprintln!(
        "{label}: {differing} of {} elements differ from the rounded value; the worst uses \
         {:.0}% of its tolerance",
        want.len(),
        100.0 * worst
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
    }
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
