//! Runs the engine token by token over an MLX-LM capture and reports how far each layer
//! visit and the logits are from MLX.
//!
//! usage: parity <model-dir> <capture.safetensors> [--passes N] [--detail STEP]
//!        [--kv bf16|fp8] [--weights bf16|fp8]
//!
//! `--detail` prints every visit of one step, with the number of BF16 values that differ.
//!
//! The capture comes from `reference/capture.py --token-by-token`. Every step feeds the
//! captured token, so a wrong prediction does not change later inputs.

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("parity needs the CUDA backend, which builds only on Linux");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{env, fs, path::PathBuf, time::Instant};

    use anyhow::{Context as _, bail, ensure};
    use half::bf16;
    use safetensors::{Dtype, SafeTensors};
    use siphon::{Checkpoint, TextConfig};
    use siphon_cuda::{
        Gpu,
        model::{Model, Trace},
        ops::attention::KvFormat,
        weights::{WeightFormat, Weights},
    };

    /// Global-layer KV capacity; captures are short.
    const CAPACITY: usize = 4096;

    struct Args {
        model: PathBuf,
        capture: PathBuf,
        passes: Option<usize>,
        detail: Option<String>,
        kv: KvFormat,
        weights: WeightFormat,
    }

    fn args() -> anyhow::Result<Args> {
        let mut positional = Vec::new();
        let mut passes = None;
        let mut detail = None;
        let mut kv = KvFormat::Bf16;
        let mut weights = WeightFormat::Bf16;
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            if arg == "--passes" {
                passes = Some(it.next().context("--passes needs a value")?.parse()?);
            } else if arg == "--detail" {
                detail = Some(it.next().context("--detail needs a step name")?);
            } else if arg == "--kv" {
                kv = it.next().context("--kv needs bf16 or fp8")?.parse()?;
            } else if arg == "--weights" {
                weights = it.next().context("--weights needs bf16 or fp8")?.parse()?;
            } else {
                positional.push(PathBuf::from(arg));
            }
        }
        let [model, capture] = <[PathBuf; 2]>::try_from(positional).ok().context(
            "usage: parity <model-dir> <capture.safetensors> [--passes N] [--detail STEP] \
             [--kv bf16|fp8] [--weights bf16|fp8]",
        )?;
        Ok(Args {
            model,
            capture,
            passes,
            detail,
            kv,
            weights,
        })
    }

    fn values<const N: usize, T>(
        capture: &SafeTensors<'_>,
        key: &str,
        dtype: Dtype,
        decode: fn([u8; N]) -> T,
    ) -> anyhow::Result<Vec<T>> {
        let view = capture
            .tensor(key)
            .with_context(|| format!("{key} missing from the capture"))?;
        ensure!(view.dtype() == dtype, "{key} is {:?}", view.dtype());
        Ok(view
            .data()
            .as_chunks::<N>()
            .0
            .iter()
            .map(|b| decode(*b))
            .collect())
    }

    /// Largest absolute difference and cosine similarity.
    fn compare(got: impl Iterator<Item = (f32, f32)>) -> (f32, f64) {
        let (mut worst, mut dot, mut gg, mut ww) = (0.0f32, 0.0f64, 0.0f64, 0.0f64);
        for (g, w) in got {
            worst = worst.max((g - w).abs());
            dot += f64::from(g) * f64::from(w);
            gg += f64::from(g) * f64::from(g);
            ww += f64::from(w) * f64::from(w);
        }
        (worst, dot / (gg.sqrt() * ww.sqrt()).max(f64::MIN_POSITIVE))
    }

    fn argmax(x: &[f32]) -> usize {
        let mut best = 0;
        for (i, &v) in x.iter().enumerate() {
            if v > x[best] {
                best = i;
            }
        }
        best
    }

    /// KL(MLX || engine) over softmax of the logits, in nats.
    fn kl(want: &[f32], got: &[f32]) -> f64 {
        let log_softmax = |x: &[f32]| {
            let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f64 = x.iter().map(|&v| f64::from(v - max).exp()).sum();
            let log_sum = sum.ln();
            x.iter()
                .map(|&v| f64::from(v - max) - log_sum)
                .collect::<Vec<_>>()
        };
        let (p, q) = (log_softmax(want), log_softmax(got));
        p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
    }

    pub fn main() -> anyhow::Result<()> {
        let args = args()?;
        let config = TextConfig::load(&args.model)?;
        let passes = args.passes.unwrap_or(config.loop_config.num_passes);
        let bytes = fs::read(&args.capture)
            .with_context(|| format!("reading {}", args.capture.display()))?;
        let capture = SafeTensors::deserialize(&bytes)?;
        let mut steps: Vec<String> = capture
            .names()
            .into_iter()
            .filter_map(|k| k.strip_suffix(".tokens").map(str::to_owned))
            .collect();
        steps.sort();
        if steps.iter().any(|s| !s.starts_with("step")) {
            bail!("the capture has a batched prefill; record it with --token-by-token");
        }

        let gpu = Gpu::new(0)?;
        let start = Instant::now();
        let checkpoint = Checkpoint::open(&args.model)?;
        let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
        gpu.stream().synchronize()?;
        eprintln!("loaded weights in {:.1} s", start.elapsed().as_secs_f64());
        let mut model = Model::with_formats(
            &gpu,
            config,
            weights,
            passes,
            CAPACITY,
            args.kv,
            args.weights,
        )?;

        let mut trace = Trace::default();
        let mut matches = 0;
        for name in &steps {
            let token = values(
                &capture,
                &format!("{name}.tokens"),
                Dtype::I32,
                i32::from_le_bytes,
            )?;
            let token = u32::try_from(token[0])?;
            let started = Instant::now();
            let next = model.step(token, Some(&mut trace))?;
            let elapsed = started.elapsed();

            let mut worst_visit = (0, 0.0f32, 1.0f64);
            for (v, visit) in trace.visits.iter().enumerate() {
                let key = format!("{name}.visit{v:02}.out.hidden");
                if capture.tensor(&key).is_err() {
                    continue;
                }
                let want = values(&capture, &key, Dtype::BF16, bf16::from_le_bytes)?;
                let (diff, cos) = compare(
                    visit
                        .output
                        .iter()
                        .zip(&want)
                        .map(|(g, w)| (g.to_f32(), w.to_f32())),
                );
                if args.detail.as_deref() == Some(name.as_str()) {
                    let differing = visit
                        .output
                        .iter()
                        .zip(&want)
                        .filter(|(g, w)| g != w)
                        .count();
                    println!(
                        "  visit {v:02}: {differing:>4} of {} differ, max diff {diff:.4}, cos {cos:.7}",
                        want.len()
                    );
                }
                if cos < worst_visit.2 {
                    worst_visit = (v, diff, cos);
                }
            }
            let want = values(
                &capture,
                &format!("{name}.logits"),
                Dtype::F32,
                f32::from_le_bytes,
            )?;
            let (diff, _) = compare(trace.logits.iter().copied().zip(want.iter().copied()));
            let want_next = argmax(&want);
            matches += usize::from(next as usize == want_next);
            println!(
                "{name}: token {token:>6} next {next:>6} mlx {want_next:>6} | logits max diff {diff:.4} kl {:.2e} | worst visit {:02} cos {:.6} max diff {:.4} | {:.0} ms",
                kl(&want, &trace.logits),
                worst_visit.0,
                worst_visit.2,
                worst_visit.1,
                elapsed.as_secs_f64() * 1e3,
            );
        }
        println!("greedy match {matches}/{}", steps.len());
        Ok(())
    }
}
