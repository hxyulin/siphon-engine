//! Quality gate: runs the engine teacher-forced over a long text and checks that it
//! still computes the same model as MLX-LM, within a multiple of MLX's own noise.
//!
//! usage: eval <model-dir> <capture.safetensors> [--passes N] [--kv bf16|fp8]
//!        [--weights bf16|fp8] [--output-projection bf16|fp8] [--prefill CHUNK] [--verify WIDTH]
//!        [--expert-activations bf16|quantized|fp8] [--prefill-attention auto|bf16|fp8]
//!        [--prefill-dense bf16|fp8] [--kv-quant none|nvfp4|q4_0] [--kv-rotate]
//!        [--kv-quant-visits V,V,...]
//!
//! The capture comes from `reference/capture_eval.py capture`. Every step feeds the
//! text's next token, so a wrong prediction does not change later inputs. With
//! `--prefill`, the text runs through `Model::prefill` in chunks of CHUNK rows instead
//! of one step per token, and every row's logits are scored. The gate
//! reports, per range of positions, top-1 agreement with MLX, the mean difference in
//! the log-probability of the actual next token, and perplexity. It also reports
//! KL(MLX || engine) at the captured positions and how many greedy tokens after the
//! text match MLX. It exits with status 1 if a check fails. `--weights fp8` makes the
//! stepping read FP8 copies of the attention input projections, the message gates and the
//! LM head, and with `--output-projection fp8` of the output projection too; prefill keeps
//! the BF16 weights but for the LM head, whose BF16 copy then sits in host memory.
//! `--dump FILE` writes one line per position: the position, the engine's and MLX's
//! log-probability of the next token, and their top-1 tokens, to compare with other
//! captures of the text.
//! `--expert-activations quantized` runs prefill's expert products on quantized
//! activations (see `ExpertActivations`); stepping stays BF16. `--prefill-attention`
//! picks prefill's global attention products over an FP8 cache (`auto` by default: FP8
//! tensor cores past `FP8_ATTENTION_KEYS` keys), and `--prefill-dense fp8` runs its dense products on the FP8 weight copies
//! (with `--weights fp8`). With `--verify`, the text runs through `Model::verify` in passes
//! of WIDTH rows (1 to 8), each keeping every row, as speculative decoding verifies the
//! anchor and drafts.

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("eval needs the CUDA backend, which builds only on Linux");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{env, fmt::Write as _, fs, path::PathBuf, time::Instant};

    use anyhow::{Context as _, ensure};
    use safetensors::{Dtype, SafeTensors};
    use siphon::{Checkpoint, TextConfig};
    use siphon_cuda::{
        Gpu,
        kv_options::KvOptions,
        model::{
            Accept, DEFAULT_PREFILL_ATTENTION, MAX_PREFILL_CHUNK, Model, PrefillAttention,
            PrefillLogits, VerifyTokens,
        },
        ops::{Products, attention::KvFormat, experts::ExpertActivations, spec::VERIFY_ROWS},
        weights::{OutputProjection, WeightFormat, Weights},
    };

    // Thresholds, from MLX's disagreement with itself between its chunked (multi-row)
    // and token-by-token (single-row) paths on the three texts in reference/prompts:
    // mean KL 6e-4 to 2.0e-3 and max KL up to 5.9e-2 at the sampled positions, top-1
    // disagreement 0.9% to 2.0% per position range, mean |log-prob difference| 0.021
    // to 0.031 per range, and perplexity within 0.25%. Each threshold allows about
    // 1.5 to 2 times the worst of that. Looser limits miss a sliding window that is not
    // enforced: in MLX that bug alone gives top-1 disagreement 2.2%, mean |log-prob
    // difference| 0.050, mean KL 9.5e-3 and max KL 0.21 on the long text. README.md,
    // section "Quality gate", has the numbers.
    const MAX_MEAN_KL: f64 = 4e-3;
    const MAX_KL: f64 = 0.12;
    const MIN_TOP1: f64 = 0.97;
    const MAX_MEAN_LOGPROB_DIFF: f64 = 0.045;
    const MAX_PERPLEXITY_RATIO: f64 = 5e-3;

    /// Worst sampled positions to list.
    const WORST: usize = 8;

    struct Args {
        model: PathBuf,
        capture: PathBuf,
        passes: Option<usize>,
        kv: KvFormat,
        weights: WeightFormat,
        output: OutputProjection,
        dump: Option<PathBuf>,
        /// Chunk size for `Model::prefill`, or `None` to step token by token.
        prefill: Option<usize>,
        /// Rows per `Model::verify` pass, instead of stepping or prefill.
        verify: Option<usize>,
        expert_activations: ExpertActivations,
        prefill_attention: PrefillAttention,
        prefill_dense: Products,
        kv_options: KvOptions,
    }

    fn args() -> anyhow::Result<Args> {
        let mut positional = Vec::new();
        let mut passes = None;
        let mut kv = KvFormat::Bf16;
        let mut weights = WeightFormat::Bf16;
        let mut output = OutputProjection::default();
        let mut dump = None;
        let mut prefill = None;
        let mut verify = None;
        let mut expert_activations = ExpertActivations::Bf16;
        let mut prefill_attention = DEFAULT_PREFILL_ATTENTION;
        let mut prefill_dense = Products::Bf16;
        let mut kv_options = KvOptions::default();
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            if kv_options.parse(&arg, &mut it)? {
            } else if arg == "--passes" {
                passes = Some(it.next().context("--passes needs a value")?.parse()?);
            } else if arg == "--kv" {
                kv = it.next().context("--kv needs bf16 or fp8")?.parse()?;
            } else if arg == "--weights" {
                weights = it.next().context("--weights needs bf16 or fp8")?.parse()?;
            } else if arg == "--dump" {
                dump = Some(PathBuf::from(it.next().context("--dump needs a file")?));
            } else if arg == "--output-projection" {
                output = it
                    .next()
                    .context("--output-projection needs blocks, rows or bf16")?
                    .parse()?;
            } else if arg == "--expert-activations" {
                expert_activations = it
                    .next()
                    .context("--expert-activations needs bf16, quantized or fp8")?
                    .parse()?;
            } else if arg == "--prefill-attention" {
                prefill_attention = it
                    .next()
                    .context("--prefill-attention needs auto, bf16 or fp8")?
                    .parse()?;
            } else if arg == "--prefill-dense" {
                prefill_dense = it
                    .next()
                    .context("--prefill-dense needs bf16 or fp8")?
                    .parse()?;
            } else if arg == "--prefill" {
                let chunk: usize = it.next().context("--prefill needs a chunk size")?.parse()?;
                ensure!(
                    (1..=MAX_PREFILL_CHUNK).contains(&chunk),
                    "--prefill takes 1 to {MAX_PREFILL_CHUNK} rows"
                );
                prefill = Some(chunk);
            } else if arg == "--verify" {
                let width: usize = it.next().context("--verify needs a width")?.parse()?;
                ensure!(
                    (1..=VERIFY_ROWS).contains(&width),
                    "--verify takes 1 to {VERIFY_ROWS} rows"
                );
                verify = Some(width);
            } else {
                positional.push(PathBuf::from(arg));
            }
        }
        let [model, capture] = <[PathBuf; 2]>::try_from(positional).ok().context(
            "usage: eval <model-dir> <capture.safetensors> [--passes N] [--kv bf16|fp8] \
             [--weights bf16|fp8] [--output-projection bf16|fp8] [--prefill CHUNK] \
             [--verify WIDTH] [--dump FILE] \
             [--expert-activations bf16|quantized|fp8] [--prefill-attention auto|bf16|fp8] \
             [--prefill-dense bf16|fp8] [kv options]",
        )?;
        ensure!(
            prefill.is_none() || verify.is_none(),
            "--prefill and --verify exclude each other"
        );
        Ok(Args {
            model,
            capture,
            passes,
            kv,
            weights,
            output,
            dump,
            prefill,
            verify,
            expert_activations,
            prefill_attention,
            prefill_dense,
            kv_options,
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

    fn ids(capture: &SafeTensors<'_>, key: &str) -> anyhow::Result<Vec<u32>> {
        values(capture, key, Dtype::I32, i32::from_le_bytes)?
            .into_iter()
            .map(|v| Ok(u32::try_from(v)?))
            .collect()
    }

    /// The MLX run the engine is checked against.
    struct Capture {
        tokens: Vec<u32>,
        /// At position i, MLX's log-probability of token i + 1.
        target_logprobs: Vec<f32>,
        argmax: Vec<u32>,
        positions: Vec<usize>,
        /// `[positions, vocab]`.
        logits: Vec<f32>,
        continuation: Vec<u32>,
        mode: String,
    }

    impl Capture {
        fn read(bytes: &[u8]) -> anyhow::Result<Self> {
            let capture = SafeTensors::deserialize(bytes)?;
            let (_, metadata) = SafeTensors::read_metadata(bytes)?;
            let mode = metadata
                .metadata()
                .as_ref()
                .and_then(|m| m.get("mode"))
                .map_or_else(|| "unknown".to_owned(), |m| m.trim_matches('"').to_owned());
            let capture = Self {
                tokens: ids(&capture, "tokens")?,
                target_logprobs: values(
                    &capture,
                    "target_logprobs",
                    Dtype::F32,
                    f32::from_le_bytes,
                )?,
                argmax: ids(&capture, "argmax")?,
                positions: ids(&capture, "sampled_positions")?
                    .into_iter()
                    .map(|p| p as usize)
                    .collect(),
                logits: values(&capture, "sampled_logits", Dtype::F32, f32::from_le_bytes)?,
                continuation: ids(&capture, "continuation")?,
                mode,
            };
            let n = capture.tokens.len();
            ensure!(n >= 2, "the capture has fewer than 2 tokens");
            ensure!(capture.target_logprobs.len() == n - 1 && capture.argmax.len() == n);
            ensure!(capture.positions.iter().all(|&p| p < n));
            ensure!(capture.logits.len() % capture.positions.len().max(1) == 0);
            ensure!(!capture.continuation.is_empty());
            Ok(capture)
        }

        fn vocab(&self) -> usize {
            self.logits.len() / self.positions.len().max(1)
        }
    }

    fn log_softmax(x: &[f32]) -> Vec<f64> {
        let max = x.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f64 = x.iter().map(|&v| f64::from(v - max).exp()).sum();
        let log_sum = sum.ln();
        x.iter().map(|&v| f64::from(v - max) - log_sum).collect()
    }

    /// Index of the largest logit, the first on ties, with NaN largest, as `mx.argmax` and
    /// the engine's argmax kernel pick it.
    fn argmax(x: &[f32]) -> u32 {
        let mut best = 0;
        for (i, &v) in x.iter().enumerate() {
            let b = x[best];
            if v > b || (v.is_nan() && !b.is_nan()) {
                best = i;
            }
        }
        best as u32
    }

    /// Log-probability of `token` under softmax of the logits.
    fn logprob(logits: &[f32], token: u32) -> f64 {
        let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sum: f64 = logits.iter().map(|&v| f64::from(v - max).exp()).sum();
        f64::from(logits[token as usize] - max) - sum.ln()
    }

    /// KL(MLX || engine) over softmax of the logits, in nats.
    fn kl(want: &[f32], got: &[f32]) -> f64 {
        let (p, q) = (log_softmax(want), log_softmax(got));
        p.iter().zip(&q).map(|(lp, lq)| lp.exp() * (lp - lq)).sum()
    }

    /// Agreement with MLX over a range of positions.
    struct Range {
        label: String,
        start: usize,
        end: usize,
        tokens: usize,
        top1: usize,
        /// Over positions with a next token.
        targets: usize,
        logprob_diff: f64,
        engine_logprob: f64,
        mlx_logprob: f64,
    }

    impl Range {
        fn new(label: String, start: usize, end: usize) -> Self {
            Self {
                label,
                start,
                end,
                tokens: 0,
                top1: 0,
                targets: 0,
                logprob_diff: 0.0,
                engine_logprob: 0.0,
                mlx_logprob: 0.0,
            }
        }

        fn top1(&self) -> f64 {
            self.top1 as f64 / self.tokens as f64
        }

        fn mean_logprob_diff(&self) -> f64 {
            self.logprob_diff / self.targets as f64
        }

        fn perplexities(&self) -> (f64, f64) {
            let n = self.targets as f64;
            (
                (-self.engine_logprob / n).exp(),
                (-self.mlx_logprob / n).exp(),
            )
        }
    }

    struct Check {
        name: String,
        value: f64,
        limit: f64,
        pass: bool,
    }

    fn at_most(name: String, value: f64, limit: f64) -> Check {
        Check {
            name,
            value,
            limit,
            pass: value <= limit,
        }
    }

    fn at_least(name: String, value: f64, limit: f64) -> Check {
        Check {
            name,
            value,
            limit,
            pass: value >= limit,
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one pass over the text, then the report, read top to bottom"
    )]
    pub fn main() -> anyhow::Result<()> {
        let args = args()?;
        let config = TextConfig::load(&args.model)?;
        let passes = args.passes.unwrap_or(config.loop_config.num_passes);
        let bytes = fs::read(&args.capture)
            .with_context(|| format!("reading {}", args.capture.display()))?;
        let capture = Capture::read(&bytes)?;
        drop(bytes);
        let n = capture.tokens.len();
        let vocab = capture.vocab();
        ensure!(
            vocab == config.unpadded_vocab_size,
            "the capture has {vocab} logits per position, the model {}",
            config.unpadded_vocab_size
        );

        // Ranges where the sliding window has not wrapped, where it has, and where the
        // global layers scale queries by log(position / floor).
        let (window, floor) = (config.sliding_window_size, config.log_scaling_n_floor);
        let mut ranges = vec![
            Range::new("before window wrap".to_owned(), 0, window.min(n)),
            Range::new("window wrapped".to_owned(), window, floor.min(n)),
            Range::new("log-scaled queries".to_owned(), floor, n),
        ];
        ranges.retain(|r| r.start < r.end);
        let mut all = Range::new("all".to_owned(), 0, n);

        let gpu = Gpu::new(0)?;
        let start = Instant::now();
        let checkpoint = Checkpoint::open(&args.model)?;
        let mut weights = Weights::load(&checkpoint, &config, gpu.stream())?;
        weights.output = args.output;
        gpu.stream().synchronize()?;
        eprintln!("loaded weights in {:.1} s", start.elapsed().as_secs_f64());
        let capacity = n + capture.continuation.len();
        let mut model = Model::with_formats(
            &gpu,
            config,
            weights,
            passes,
            capacity,
            args.kv,
            args.weights,
        )?;
        args.kv_options.apply(&mut model)?;
        model.set_expert_activations(args.expert_activations)?;
        model.set_prefill_attention(args.prefill_attention)?;
        model.set_prefill_dense(args.prefill_dense)?;

        let mut sampled = capture.positions.iter().copied().enumerate().peekable();
        // (position, KL, engine top-1, MLX top-1)
        let mut kls: Vec<(usize, f64, u32, u32)> = Vec::new();
        // With --dump, per position: the engine's and MLX's log-probability of the next
        // token and their top-1 tokens.
        let mut dumped: Vec<(usize, f64, f64, u32, u32)> = Vec::new();
        let started = Instant::now();
        // Scores the engine's logits at position i, whose top-1 token is `next`.
        let mut score = |i: usize, logits: &[f32], next: u32| -> anyhow::Result<()> {
            ensure!(
                logits.len() == vocab,
                "the engine returned {} logits",
                logits.len()
            );
            let mlx_next = capture.argmax[i];
            let target = capture.tokens.get(i + 1).map(|&t| {
                let mlx = f64::from(capture.target_logprobs[i]);
                (logprob(logits, t), mlx)
            });
            if let Some((engine, mlx)) = target {
                dumped.push((i, engine, mlx, next, mlx_next));
            }
            for range in ranges
                .iter_mut()
                .filter(|r| (r.start..r.end).contains(&i))
                .chain([&mut all])
            {
                range.tokens += 1;
                range.top1 += usize::from(next == mlx_next);
                if let Some((engine, mlx)) = target {
                    range.targets += 1;
                    range.logprob_diff += (engine - mlx).abs();
                    range.engine_logprob += engine;
                    range.mlx_logprob += mlx;
                }
            }
            if let Some((k, _)) = sampled.next_if(|&(_, p)| p == i) {
                let want = &capture.logits[k * vocab..(k + 1) * vocab];
                kls.push((i, kl(want, logits), next, mlx_next));
            }
            if (i + 1).is_multiple_of(1024) {
                let seconds = started.elapsed().as_secs_f64();
                eprintln!(
                    "{}/{n} tokens, {:.0} ms per token",
                    i + 1,
                    seconds * 1e3 / (i + 1) as f64
                );
            }
            Ok(())
        };
        let mut next = 0;
        if let Some(chunk) = args.prefill {
            let mut callback = |i: usize, logits: &[f32]| score(i, logits, argmax(logits));
            next = model.prefill(&capture.tokens, chunk, PrefillLogits::All(&mut callback))?;
        } else if let Some(width) = args.verify {
            for (c, rows) in capture.tokens.chunks(width).enumerate() {
                let verdict = model.verify(VerifyTokens::Host(rows), Accept::Rows(rows.len()))?;
                let logits = model.verify_logits()?;
                for (i, row) in logits.chunks_exact(vocab).enumerate() {
                    score(c * width + i, row, verdict.argmax[i])?;
                }
                next = verdict.next();
            }
        } else {
            for (i, &token) in capture.tokens.iter().enumerate() {
                next = model.step(token, None)?;
                score(i, &model.logits()?, next)?;
            }
        }
        let forced_seconds = started.elapsed().as_secs_f64();
        if let Some(path) = &args.dump {
            let mut text = String::new();
            for (i, engine, mlx, top, mlx_top) in &dumped {
                writeln!(text, "{i} {engine} {mlx} {top} {mlx_top}")?;
            }
            fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
        }

        // Greedy continuation from the engine's own predictions.
        let mut continuation = vec![next];
        while continuation.len() < capture.continuation.len() {
            continuation.push(model.step(next, None)?);
            next = *continuation.last().unwrap();
        }
        let continued = continuation
            .iter()
            .zip(&capture.continuation)
            .take_while(|(a, b)| a == b)
            .count();
        let seconds = started.elapsed().as_secs_f64();

        let engine_mode = match (args.prefill, args.verify) {
            (Some(chunk), _) => format!(
                "prefill in chunks of {chunk}, {} expert activations, {} attention and {} dense \
                 products",
                args.expert_activations, args.prefill_attention, args.prefill_dense
            ),
            (None, Some(width)) => format!(
                "verify passes of {width} rows, {} expert activations",
                args.expert_activations
            ),
            (None, None) => "token by token".to_owned(),
        };
        println!(
            "{n} tokens; MLX capture mode: {}; engine: {engine_mode}; {}",
            capture.mode, args.kv_options
        );
        println!(
            "{:<20} {:>13} {:>7} {:>8} {:>12}  perplexity engine / MLX",
            "range", "positions", "tokens", "top-1", "|dlogprob|"
        );
        for r in ranges.iter().chain([&all]) {
            let (engine, mlx) = r.perplexities();
            println!(
                "{:<20} {:>13} {:>7} {:>7.2}% {:>12.4}  {engine:.4} / {mlx:.4}",
                r.label,
                format!("{}..{}", r.start, r.end),
                r.tokens,
                r.top1() * 100.0,
                r.mean_logprob_diff(),
            );
        }

        let mut sorted: Vec<f64> = kls.iter().map(|k| k.1).collect();
        sorted.sort_by(f64::total_cmp);
        let mean_kl = sorted.iter().sum::<f64>() / sorted.len().max(1) as f64;
        let max_kl = sorted.last().copied().unwrap_or(0.0);
        println!(
            "KL(MLX || engine) at {} sampled positions: mean {mean_kl:.2e} median {:.2e} max {max_kl:.2e}",
            kls.len(),
            sorted.get(sorted.len() / 2).copied().unwrap_or(0.0),
        );
        kls.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(p, kl, engine, mlx) in kls.iter().take(WORST) {
            println!("  position {p:>6}: KL {kl:.2e}, top-1 engine {engine:>6} MLX {mlx:>6}");
        }
        println!(
            "greedy continuation matches MLX for {continued}/{} tokens",
            capture.continuation.len()
        );
        println!(
            "runtime {seconds:.0} s: {:.1} ms per teacher-forced token, every position's logits read back",
            forced_seconds * 1e3 / n as f64
        );

        let (engine_ppl, mlx_ppl) = all.perplexities();
        let mut checks = vec![
            at_most("mean KL".to_owned(), mean_kl, MAX_MEAN_KL),
            at_most("max KL".to_owned(), max_kl, MAX_KL),
            at_most(
                "|perplexity ratio - 1|".to_owned(),
                (engine_ppl / mlx_ppl - 1.0).abs(),
                MAX_PERPLEXITY_RATIO,
            ),
        ];
        for r in &ranges {
            checks.push(at_least(format!("top-1, {}", r.label), r.top1(), MIN_TOP1));
            checks.push(at_most(
                format!("|dlogprob|, {}", r.label),
                r.mean_logprob_diff(),
                MAX_MEAN_LOGPROB_DIFF,
            ));
        }
        for c in &checks {
            let (verdict, op) = if c.pass {
                ("pass", "within")
            } else {
                ("FAIL", "outside")
            };
            println!(
                "{verdict}  {:<34} {:.3e} {op} {:.1e}",
                c.name, c.value, c.limit
            );
        }
        if checks.iter().all(|c| c.pass) {
            println!("verdict: PASS");
            Ok(())
        } else {
            println!("verdict: FAIL");
            std::process::exit(1);
        }
    }
}
