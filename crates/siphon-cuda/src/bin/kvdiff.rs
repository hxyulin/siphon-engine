//! Long-context divergence between two engine configurations: teacher-forces a long token
//! file through `Model::prefill` and records, at every Nth position, the top tokens'
//! log-probabilities. Given another run's record, it reports KL divergence, top-1 agreement
//! and the next token's log-probability against it, by range of positions.
//!
//! usage: kvdiff <model-dir> <tokens.u32> --out FILE [--against FILE] [--tokens N]
//!        [--every 64] [--top 256] [--chunk 2048] [--kv bf16|fp8] [--weights bf16|fp8]
//!        [kv options]
//!
//! The token file holds little-endian u32 ids. The record keeps, per sampled position, the
//! argmax, the next token's log-probability and the `--top` largest log-probabilities with
//! their ids. KL(record || this run) sums over the record's top tokens and lumps the rest
//! of the vocabulary into one outcome on each side, which only lowers it; with 256 tokens
//! the remainder holds well under 1% of the probability at almost every position.

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("kvdiff needs the CUDA backend, which builds only on Linux");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{
        env, fs,
        io::{BufWriter, Write as _},
        path::PathBuf,
        time::Instant,
    };

    use anyhow::{Context as _, ensure};
    use siphon::{Checkpoint, TextConfig};
    use siphon_cuda::{
        Gpu,
        kv_options::{self, KvOptions},
        model::{Model, PrefillLogits},
        ops::attention::KvFormat,
        weights::{WeightFormat, Weights},
    };

    const MAGIC: &[u8; 8] = b"KVDIFF01";
    /// Upper ends of the position ranges the report groups by.
    const RANGES: [usize; 11] = [
        1 << 10,
        4 << 10,
        16 << 10,
        32 << 10,
        64 << 10,
        128 << 10,
        256 << 10,
        384 << 10,
        512 << 10,
        768 << 10,
        1 << 20,
    ];

    struct Args {
        model: PathBuf,
        tokens: PathBuf,
        out: PathBuf,
        against: Option<PathBuf>,
        limit: Option<usize>,
        every: usize,
        top: usize,
        chunk: usize,
        kv: KvFormat,
        weights: WeightFormat,
        kv_options: KvOptions,
    }

    fn usage() -> String {
        format!(
            "usage: kvdiff <model-dir> <tokens.u32> --out FILE [--against FILE] [--tokens N] \
             [--every 64] [--top 256] [--chunk 2048] [--kv bf16|fp8] [--weights bf16|fp8] {}",
            kv_options::USAGE
        )
    }

    fn args() -> anyhow::Result<Args> {
        let mut positional = Vec::new();
        let (mut out, mut against, mut limit) = (None, None, None);
        let (mut every, mut top, mut chunk) = (64, 256, 2048);
        let mut kv = KvFormat::Bf16;
        let mut weights = WeightFormat::Bf16;
        let mut kv_options = KvOptions::default();
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            if kv_options.parse(&arg, &mut it)? {
                continue;
            }
            let mut value = || it.next().with_context(|| format!("{arg} needs a value"));
            match arg.as_str() {
                "--out" => out = Some(PathBuf::from(value()?)),
                "--against" => against = Some(PathBuf::from(value()?)),
                "--tokens" => limit = Some(value()?.parse()?),
                "--every" => every = value()?.parse()?,
                "--top" => top = value()?.parse()?,
                "--chunk" => chunk = value()?.parse()?,
                "--kv" => kv = value()?.parse()?,
                "--weights" => weights = value()?.parse()?,
                _ if arg.starts_with("--") => anyhow::bail!("unknown flag {arg}\n{}", usage()),
                _ => positional.push(PathBuf::from(arg)),
            }
        }
        let [model, tokens] = <[PathBuf; 2]>::try_from(positional)
            .ok()
            .with_context(usage)?;
        ensure!(every > 0 && top > 0, "--every and --top must be positive");
        Ok(Args {
            model,
            tokens,
            out: out.with_context(usage)?,
            against,
            limit,
            every,
            top,
            chunk,
            kv,
            weights,
            kv_options,
        })
    }

    /// One sampled position of a run.
    struct Record {
        position: u32,
        argmax: u32,
        /// Log-probability of the next token, NaN at the last position.
        next_logprob: f32,
        /// The largest log-probabilities, largest first.
        top: Vec<(u32, f32)>,
    }

    fn log_softmax(x: &[f32]) -> Vec<f32> {
        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f64 = x.iter().map(|&v| f64::from(v - max).exp()).sum();
        let log_sum = sum.ln() as f32;
        x.iter().map(|&v| v - max - log_sum).collect()
    }

    fn record(position: usize, logits: &[f32], next: Option<u32>, top: usize) -> Record {
        let lp = log_softmax(logits);
        let mut ids: Vec<u32> = (0..lp.len() as u32).collect();
        let k = top.min(ids.len());
        ids.select_nth_unstable_by(k - 1, |&a, &b| lp[b as usize].total_cmp(&lp[a as usize]));
        ids.truncate(k);
        ids.sort_by(|&a, &b| lp[b as usize].total_cmp(&lp[a as usize]).then(a.cmp(&b)));
        // The first largest logit, as the argmax kernel picks it.
        let mut argmax = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[argmax] {
                argmax = i;
            }
        }
        Record {
            position: position as u32,
            argmax: argmax as u32,
            next_logprob: next.map_or(f32::NAN, |t| lp[t as usize]),
            top: ids.iter().map(|&i| (i, lp[i as usize])).collect(),
        }
    }

    fn write_records(path: &PathBuf, top: usize, records: &[Record]) -> anyhow::Result<()> {
        let mut w = BufWriter::new(fs::File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&(records.len() as u32).to_le_bytes())?;
        w.write_all(&(top as u32).to_le_bytes())?;
        for r in records {
            w.write_all(&r.position.to_le_bytes())?;
            w.write_all(&r.argmax.to_le_bytes())?;
            w.write_all(&r.next_logprob.to_le_bytes())?;
            for k in 0..top {
                let (id, lp) = r
                    .top
                    .get(k)
                    .copied()
                    .unwrap_or((u32::MAX, f32::NEG_INFINITY));
                w.write_all(&id.to_le_bytes())?;
                w.write_all(&lp.to_le_bytes())?;
            }
        }
        w.flush()?;
        Ok(())
    }

    fn read_records(path: &PathBuf) -> anyhow::Result<Vec<Record>> {
        let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        ensure!(
            bytes.starts_with(MAGIC),
            "{} is not a kvdiff record",
            path.display()
        );
        let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let (n, top) = (word(8) as usize, word(12) as usize);
        let size = 12 + 8 * top;
        ensure!(
            bytes.len() == 16 + n * size,
            "{} is truncated",
            path.display()
        );
        Ok((0..n)
            .map(|r| {
                let at = 16 + r * size;
                let f = |i: usize| f32::from_bits(word(i));
                Record {
                    position: word(at),
                    argmax: word(at + 4),
                    next_logprob: f(at + 8),
                    top: (0..top)
                        .map(|k| (word(at + 12 + 8 * k), f(at + 16 + 8 * k)))
                        .filter(|&(id, _)| id != u32::MAX)
                        .collect(),
                }
            })
            .collect())
    }

    /// KL(base || run) over the base's top tokens and the rest of the vocabulary as one
    /// outcome.
    fn kl(base: &Record, run_logprobs: &[f32]) -> f64 {
        let (mut sum, mut p_top, mut q_top) = (0.0f64, 0.0f64, 0.0f64);
        for &(id, lp) in &base.top {
            let (lp, lq) = (f64::from(lp), f64::from(run_logprobs[id as usize]));
            sum += lp.exp() * (lp - lq);
            p_top += lp.exp();
            q_top += lq.exp();
        }
        let (p_rest, q_rest) = ((1.0 - p_top).max(0.0), (1.0 - q_top).max(1e-30));
        if p_rest > 0.0 {
            sum += p_rest * (p_rest.ln() - q_rest.ln());
        }
        sum
    }

    /// A sampled position's record and, against a base record, its range, KL, top-1
    /// agreement and next-token terms.
    type Sample = (Record, Option<(usize, f64, bool, Option<(f64, f64, f64)>)>);

    #[derive(Default)]
    struct Range {
        positions: usize,
        kls: Vec<f64>,
        top1: usize,
        targets: usize,
        dlogprob: f64,
        base_nll: f64,
        run_nll: f64,
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one run, then the report, top to bottom"
    )]
    pub fn main() -> anyhow::Result<()> {
        let args = args()?;
        let bytes =
            fs::read(&args.tokens).with_context(|| format!("reading {}", args.tokens.display()))?;
        let mut tokens: Vec<u32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        if let Some(limit) = args.limit {
            ensure!(
                limit <= tokens.len(),
                "the file has {} tokens",
                tokens.len()
            );
            tokens.truncate(limit);
        }
        let n = tokens.len();
        let base = args.against.as_ref().map(read_records).transpose()?;

        let config = TextConfig::load(&args.model)?;
        let passes = config.loop_config.num_passes;
        let gpu = Gpu::new(0)?;
        let started = Instant::now();
        let checkpoint = Checkpoint::open(&args.model)?;
        let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
        let mut model =
            Model::with_formats(&gpu, config, weights, passes, n + 1, args.kv, args.weights)?;
        args.kv_options.apply(&mut model)?;
        gpu.stream().synchronize()?;
        eprintln!("loaded in {:.1} s", started.elapsed().as_secs_f64());

        // Workers turn each sampled row into a record and its comparison while prefill goes on.
        let base_by_position: std::collections::HashMap<u32, &Record> =
            base.iter().flatten().map(|r| (r.position, r)).collect();
        let (send, receive) = std::sync::mpsc::sync_channel::<(usize, Vec<f32>)>(64);
        let receive = std::sync::Mutex::new(receive);
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get().min(16));
        let prefill_started = Instant::now();
        let mut samples: Vec<Sample> = std::thread::scope(|scope| -> anyhow::Result<_> {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut out: Vec<Sample> = Vec::new();
                        loop {
                            let Ok((i, logits)) = receive.lock().expect("lock").recv() else {
                                return out;
                            };
                            let next = tokens.get(i + 1).copied();
                            let r = record(i, &logits, next, args.top);
                            let cmp = base_by_position.get(&(i as u32)).map(|b| {
                                let lp = log_softmax(&logits);
                                let range = RANGES
                                    .iter()
                                    .position(|&end| i < end)
                                    .unwrap_or(RANGES.len() - 1);
                                let target = (!b.next_logprob.is_nan()).then(|| {
                                    (
                                        f64::from((b.next_logprob - r.next_logprob).abs()),
                                        -f64::from(b.next_logprob),
                                        -f64::from(r.next_logprob),
                                    )
                                });
                                (range, kl(b, &lp), b.argmax == r.argmax, target)
                            });
                            out.push((r, cmp));
                        }
                    })
                })
                .collect();
            let mut sampled = 0usize;
            let mut callback = |i: usize, logits: &[f32]| -> anyhow::Result<()> {
                send.send((i, logits.to_vec()))
                    .context("the workers stopped")?;
                sampled += 1;
                if sampled.is_multiple_of(1024) {
                    eprintln!(
                        "{i}/{n} positions, {:.0} s",
                        prefill_started.elapsed().as_secs_f64()
                    );
                }
                Ok(())
            };
            model.prefill(
                &tokens,
                args.chunk,
                PrefillLogits::Every(args.every, &mut callback),
            )?;
            drop(send);
            Ok(handles
                .into_iter()
                .flat_map(|h| h.join().expect("a worker panicked"))
                .collect())
        })?;
        samples.sort_by_key(|(r, _)| r.position);
        let mut ranges: Vec<Range> = RANGES.iter().map(|_| Range::default()).collect();
        for (_, cmp) in &samples {
            if let Some((range, kl, top1, target)) = cmp {
                let g = &mut ranges[*range];
                g.positions += 1;
                g.kls.push(*kl);
                g.top1 += usize::from(*top1);
                if let Some((d, base_nll, run_nll)) = target {
                    g.targets += 1;
                    g.dlogprob += d;
                    g.base_nll += base_nll;
                    g.run_nll += run_nll;
                }
            }
        }
        let records: Vec<Record> = samples.into_iter().map(|(r, _)| r).collect();
        let seconds = prefill_started.elapsed().as_secs_f64();
        write_records(&args.out, args.top, &records)?;
        println!(
            "{n} tokens, {} sampled every {}, prefill {seconds:.1} s ({:.0} tok/s); kv {} weights {}; {}",
            records.len(),
            args.every,
            n as f64 / seconds,
            args.kv,
            args.weights,
            args.kv_options
        );
        let Some(_) = base else {
            return Ok(());
        };
        println!(
            "{:>17} {:>6} {:>10} {:>10} {:>10} {:>8} {:>11} {:>16}",
            "positions",
            "count",
            "mean KL",
            "p99 KL",
            "max KL",
            "top-1",
            "|dlogprob|",
            "NLL base / run"
        );
        let mut all = Range::default();
        let mut lower = 0;
        for (g, &end) in ranges.iter_mut().zip(&RANGES) {
            if g.positions > 0 {
                print_range(&format!("{lower}..{end}"), g);
                all.positions += g.positions;
                all.kls.extend_from_slice(&g.kls);
                all.top1 += g.top1;
                all.targets += g.targets;
                all.dlogprob += g.dlogprob;
                all.base_nll += g.base_nll;
                all.run_nll += g.run_nll;
            }
            lower = end;
        }
        print_range("all", &mut all);
        Ok(())
    }

    fn print_range(label: &str, g: &mut Range) {
        g.kls.sort_by(f64::total_cmp);
        let n = g.kls.len().max(1);
        let mean = g.kls.iter().sum::<f64>() / n as f64;
        let p99 = g.kls[(n * 99 / 100).min(g.kls.len().saturating_sub(1))];
        let max = g.kls.last().copied().unwrap_or(0.0);
        let t = g.targets.max(1) as f64;
        println!(
            "{label:>17} {:>6} {mean:>10.2e} {p99:>10.2e} {max:>10.2e} {:>7.2}% {:>11.4} {:>7.4} / {:.4}",
            g.positions,
            g.top1 as f64 * 100.0 / g.positions.max(1) as f64,
            g.dlogprob / t,
            g.base_nll / t,
            g.run_nll / t,
        );
    }
}
