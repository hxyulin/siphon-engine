//! The vision tower.
//!
//! `attention_matches_f64` checks the bidirectional attention kernel against an F64
//! softmax over the same BF16 inputs. The kernel rounds the probabilities to BF16 before
//! multiplying them with the values (relative error 2^-9 each) and rounds its output to
//! BF16, so it stays within about 2^-8 of the largest value magnitude.
//!
//! `tower_matches_transformers` runs the whole tower on the six images of
//! `reference/fixtures/vision/` and compares the projected embeddings with Transformers in
//! F32, next to mlx-vlm in BF16 (see docs/vision.md). It needs the checkpoint and the full
//! fixtures that `reference/vision_capture.py capture` writes, so it is ignored by default:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> SIPHON_VISION_FIXTURES=<capture-dir> \
//!     cargo test --release -p siphon-cuda --test vision -- --ignored --nocapture
//! ```

#![cfg(target_os = "linux")]

use std::{env, fs, path::PathBuf};

use anyhow::Context as _;
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use siphon::{Checkpoint, memory};
use siphon_cuda::{
    Gpu,
    ops::vision::{DIM, HEAD_DIM, HEADS, Vision},
    vision::{PreparedImage, VisionFormat, VisionTower},
};

/// Uniform pseudo-random values in [-scale, scale), rounded to BF16.
fn values(seed: u64, n: usize, scale: f32) -> Vec<bf16> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let unit = (s >> 40) as f32 / (1u64 << 24) as f32;
            bf16::from_f32((2.0 * unit - 1.0) * scale)
        })
        .collect()
}

/// softmax(q k^T) v per head in F64, over rows of `[rows, 1024]`.
fn attention_f64(q: &[bf16], k: &[bf16], v: &[bf16], rows: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * DIM];
    let mut scores = vec![0.0f64; rows];
    for h in 0..HEADS {
        for i in 0..rows {
            let qi = &q[i * DIM + h * HEAD_DIM..][..HEAD_DIM];
            for (j, s) in scores.iter_mut().enumerate() {
                let kj = &k[j * DIM + h * HEAD_DIM..][..HEAD_DIM];
                *s = qi
                    .iter()
                    .zip(kj)
                    .map(|(a, b)| f64::from(a.to_f32()) * f64::from(b.to_f32()))
                    .sum();
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for s in &mut scores {
                *s = (*s - max).exp();
                total += *s;
            }
            let o = &mut out[i * DIM + h * HEAD_DIM..][..HEAD_DIM];
            for (j, s) in scores.iter().enumerate() {
                let vj = &v[j * DIM + h * HEAD_DIM..][..HEAD_DIM];
                for (od, vd) in o.iter_mut().zip(vj) {
                    *od += s / total * f64::from(vd.to_f32());
                }
            }
        }
    }
    out
}

#[test]
fn attention_matches_f64() -> anyhow::Result<()> {
    let gpu = Gpu::new(0)?;
    let stream = gpu.stream();
    let vision = Vision::load(gpu.context())?;
    // Partial and whole key tiles of 64, and more than one query block.
    for (seed, rows) in [1usize, 2, 63, 64, 65, 200, 577].into_iter().enumerate() {
        let seed = seed as u64 * 3;
        // Queries carry the 1/8 scale; scores spread over a few units.
        let q = values(seed + 1, rows * DIM, 0.25);
        let k = values(seed + 2, rows * DIM, 2.0);
        let v = values(seed + 3, rows * DIM, 2.0);
        let qd = stream.clone_htod(&q)?;
        let kd = stream.clone_htod(&k)?;
        let vd = stream.clone_htod(&v)?;
        let mut out = stream.alloc_zeros::<bf16>(rows * DIM)?;
        vision.attention(
            stream,
            &qd.as_view(),
            &kd.as_view(),
            &vd.as_view(),
            &mut out.as_view_mut(),
        )?;
        let got = stream.clone_dtoh(&out)?;
        let want = attention_f64(&q, &k, &v, rows);
        let (mut max, mut sum) = (0.0f64, 0.0f64);
        for (g, w) in got.iter().zip(&want) {
            let e = (f64::from(g.to_f32()) - w).abs();
            max = max.max(e);
            sum += e;
        }
        let mean = sum / want.len() as f64;
        println!("rows {rows}: max {max:.3e}, mean {mean:.3e}");
        // Values reach 2, so 2^-8 of it is 7.8e-3. With few keys the probabilities'
        // rounding errors do not average out, and the output's own rounding (half a BF16
        // ulp, 2^-8 near 1) dominates the mean.
        assert!(
            max < 8e-3 && mean < 2e-3,
            "rows {rows}: max {max}, mean {mean}"
        );
    }
    Ok(())
}

/// The tensors `reference/vision_capture.py capture` saved for one image.
struct Fixture(Vec<u8>);

impl Fixture {
    fn f64(&self, key: &str) -> anyhow::Result<Vec<f64>> {
        let tensors = SafeTensors::deserialize(&self.0)?;
        let view = tensors
            .tensor(key)
            .with_context(|| format!("{key} missing"))?;
        anyhow::ensure!(view.dtype() == Dtype::F32, "{key} is {:?}", view.dtype());
        Ok(view
            .data()
            .as_chunks()
            .0
            .iter()
            .map(|b| f64::from(f32::from_le_bytes(*b)))
            .collect())
    }
}

/// The image `name` of `reference/fixtures/vision/` as the reference prepared it.
fn image(dir: &str, name: &str) -> anyhow::Result<(PreparedImage, Fixture)> {
    let json = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../reference/fixtures/vision/{name}.json"));
    let meta: serde_json::Value = serde_json::from_str(&fs::read_to_string(&json)?)?;
    let grid: Vec<usize> = meta["grid"]
        .as_array()
        .context("no grid")?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).context("grid entry"))
        .collect::<anyhow::Result<_>>()?;
    let fixture = Fixture(fs::read(format!("{dir}/{name}.safetensors"))?);
    let patches = fixture
        .f64("patches")?
        .into_iter()
        .map(|x| x as f32)
        .collect();
    let image = PreparedImage {
        grid: [grid[0], grid[1], grid[2]],
        patches,
        hash: [0; 32],
    };
    Ok((image, fixture))
}

/// How far `got` is from `want`, both `[rows, width]`.
struct Error {
    max: f64,
    mean: f64,
    /// sqrt(sum (got - want)^2 / sum want^2).
    rel_rms: f64,
    cosine: f64,
    /// The lowest cosine similarity of one row.
    row_cosine: f64,
}

fn compare(got: &[f64], want: &[f64], width: usize) -> Error {
    let cos = |a: &[f64], b: &[f64]| {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let aa: f64 = a.iter().map(|x| x * x).sum();
        let bb: f64 = b.iter().map(|x| x * x).sum();
        dot / (aa.sqrt() * bb.sqrt())
    };
    let diff2: f64 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
    let want2: f64 = want.iter().map(|w| w * w).sum();
    Error {
        max: got
            .iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0, f64::max),
        mean: got
            .iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs())
            .sum::<f64>()
            / want.len() as f64,
        rel_rms: (diff2 / want2).sqrt(),
        cosine: cos(got, want),
        row_cosine: got
            .chunks(width)
            .zip(want.chunks(width))
            .map(|(g, w)| cos(g, w))
            .fold(1.0, f64::min),
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "max {:.4}, mean {:.5}, rel RMS {:.2}%, cosine {:.6}, lowest row {:.4}",
            self.max,
            self.mean,
            100.0 * self.rel_rms,
            self.cosine,
            self.row_cosine
        )
    }
}

#[test]
#[ignore = "needs the checkpoint (SIPHON_MODEL) and the capture (SIPHON_VISION_FIXTURES)"]
fn tower_matches_transformers() -> anyhow::Result<()> {
    let model = env::var("SIPHON_MODEL").context("set SIPHON_MODEL")?;
    let dir = env::var("SIPHON_VISION_FIXTURES").context("set SIPHON_VISION_FIXTURES")?;
    let checkpoint = Checkpoint::open(&model)?;
    let gpu = Gpu::new(0)?;
    let names = ["tiny", "alpha", "tall", "square", "screenshot", "huge"];
    let format = VisionFormat::Bf16;
    let before = gpu.memory()?.pool_used;
    let mut tower = VisionTower::load(&gpu, &checkpoint, format)?;
    let weights = gpu.memory()?.pool_used - before;
    println!(
        "{format}: weights {weights} bytes, planned {}",
        memory::vision_weight_bytes(format)
    );
    assert!(weights.abs_diff(memory::vision_weight_bytes(format)) < 1 << 20);
    for name in names {
        let (image, fixture) = image(&dir, name)?;
        let out = tower.encode(&image)?;
        let got: Vec<f64> = gpu
            .stream()
            .clone_dtoh(&out)?
            .iter()
            .map(|x| f64::from(x.to_f32()))
            .collect();
        let want = fixture.f64("f32.projected")?;
        assert_eq!(got.len(), want.len());
        let ours = compare(&got, &want, memory::VISION_OUT);
        let mlx = compare(&fixture.f64("bf16.projected")?, &want, memory::VISION_OUT);
        println!("{format} {name} ({} tokens): {ours}", image.tokens());
        println!("  mlx-vlm BF16: {mlx}");
        // The tower must do at least as well as mlx-vlm in BF16.
        assert!(
            ours.rel_rms <= mlx.rel_rms,
            "{format} {name}: relative RMS {} against mlx-vlm's {}",
            ours.rel_rms,
            mlx.rel_rms
        );
    }
    Ok(())
}
