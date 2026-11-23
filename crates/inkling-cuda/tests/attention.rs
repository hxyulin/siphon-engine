//! Attention kernels against MLX outputs from `reference/fixtures/attention.py` and against an
//! F64 reference on the host.
//!
//! The fixture feeds 1100 tokens into a sliding (1024-slot ring) and a global cache and records
//! MLX's decode attention at a few positions across the window and the bias table width. The
//! host reference covers what the fixture is too short for: a global cache of 40,000 keys cut
//! into the full number of splits, and a sliding ring that has wrapped several times.

#![cfg(target_os = "linux")]

mod common;

use common::{Fixture, assert_bf16_close, bf16_ulps};
use half::bf16;
use inkling_cuda::{
    Gpu,
    ops::attention::{Attention, AttentionScratch, HEAD_DIM, HEADS, KV_HEADS, KvCache},
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
    keys: &[bf16],
    values: &[bf16],
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
            let mut score: f64 = q.iter().zip(k).map(|(&a, &b)| a * b.to_f64()).sum();
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
                *o += weight * v.to_f64();
                *m += weight * v.to_f64().abs();
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
    mut check: impl FnMut(usize, &Query, &[bf16], &[bf16], &[bf16]) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let stream = gpu.stream();
    let slots = cache.slots();
    let mut keys = vec![bf16::ZERO; slots * ROW];
    let mut values = vec![bf16::ZERO; slots * ROW];
    let k_dev = stream.clone_htod(k)?;
    let v_dev = stream.clone_htod(v)?;
    let mut scratch = AttentionScratch::new(stream)?;
    let mut out_dev = stream.alloc_zeros::<bf16>(HEADS * HEAD_DIM)?;
    let mut position_dev = stream.alloc_zeros::<u32>(1)?;
    for position in 0..k.len() / ROW {
        assert!(cache.holds(position));
        let token = position * ROW..(position + 1) * ROW;
        let slot = (position % slots) * ROW;
        keys[slot..slot + ROW].copy_from_slice(&k[token.clone()]);
        values[slot..slot + ROW].copy_from_slice(&v[token.clone()]);
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
        check(position, &inputs, &keys, &values, &got)?;
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
        |position, query, keys, values, got| {
            let prefix = format!("{case}.{position}");
            let (want, magnitude) = reference(query, keys, values, window, position);
            assert_within(
                &format!("{prefix} vs F64"),
                got,
                &want,
                &magnitude,
                F32_ERROR,
            );
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
    let label = if window.is_some() {
        "sliding"
    } else {
        "global"
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
        |position, query, keys, values, got| {
            let (want, magnitude) = reference(query, keys, values, window, position);
            assert_within(
                &format!("{label}.{position}"),
                got,
                &want,
                &magnitude,
                F32_ERROR,
            );
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
