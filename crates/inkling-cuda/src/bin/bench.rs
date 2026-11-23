//! Times greedy decode at several context lengths, or prefill at several prompt lengths.
//!
//! usage: bench <model-dir> [--contexts 0,4096,32768] [--passes N] [--steps N]
//!        [--warmup N] [--kv bf16|fp8] [--json]
//!        bench <model-dir> --prefill 512,4096,32768 [--chunk N] [--passes N]
//!        [--kv bf16|fp8] [--json]
//!
//! With `--prefill`, each prompt length runs once through `Model::prefill` in chunks of
//! `--chunk` rows from an empty cache, after one warmup chunk, and the time to the first
//! token is the time `prefill` takes to return the greedy next token. The prompt tokens
//! are arbitrary ids, so expert routing differs from real text.
//!
//! For each context length the runner skips its caches to that length without computing
//! the skipped tokens, runs `--warmup` steps, then times `--steps` steps. The tokens it
//! produces are meaningless, but each step does the same work as a real one. Without
//! `--contexts`, it runs 0, 4096 and 32768, and 131072 if the cache fits in free memory.
//! `--kv fp8` stores the global-layer KV caches as E4M3 instead of BF16.
//!
//! Results are comparable only on an otherwise idle GPU, so the output records the GPU
//! clocks, the processes on the GPU and the time.

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("bench needs the CUDA backend, which builds only on Linux");
    std::process::exit(1);
}

#[cfg(target_os = "linux")]
mod linux {
    use std::{env, fmt::Write as _, path::PathBuf, process::Command, time::Instant};

    use anyhow::{Context as _, ensure};
    use inkling::{Checkpoint, TextConfig};
    use inkling_cuda::{
        Gpu,
        model::{DEFAULT_PREFILL_CHUNK, MAX_PREFILL_CHUNK, Model, PrefillLogits},
        ops::attention::KvFormat,
        weights::Weights,
    };

    /// Bytes of weights one decode step reads.
    const WEIGHT_BYTES_PER_TOKEN: f64 = 5.5e9;
    /// RTX 5090 DRAM bandwidth.
    const PEAK_BANDWIDTH: f64 = 1.79e12;
    const DEFAULT_CONTEXTS: [usize; 3] = [0, 4096, 32768];
    /// Added to the defaults when its cache fits.
    const LONG_CONTEXT: usize = 131_072;
    /// Free memory to leave for the runner's per-step buffers.
    const HEADROOM: usize = 2 << 30;
    const USAGE: &str = "usage: bench <model-dir> [--contexts 0,4096,32768] [--passes N] \
                         [--steps N] [--warmup N] [--kv bf16|fp8] [--json]\n       \
                         bench <model-dir> --prefill 512,4096 [--chunk N] [--passes N] \
                         [--kv bf16|fp8] [--json]";

    struct Args {
        model: PathBuf,
        contexts: Option<Vec<usize>>,
        passes: Option<usize>,
        steps: usize,
        warmup: usize,
        kv: KvFormat,
        json: bool,
        /// Prompt lengths to time prefill at, instead of timing decode.
        prefill: Option<Vec<usize>>,
        chunk: usize,
    }

    fn token_counts(arg: &str, value: &str) -> anyhow::Result<Vec<usize>> {
        value
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()
            .with_context(|| format!("{arg} takes comma-separated token counts"))
    }

    fn args() -> anyhow::Result<Args> {
        let mut model = None;
        let mut contexts = None;
        let mut passes = None;
        let mut steps = 100;
        let mut warmup = 10;
        let mut kv = KvFormat::Bf16;
        let mut json = false;
        let mut prefill = None;
        let mut chunk = DEFAULT_PREFILL_CHUNK;
        let mut it = env::args().skip(1);
        while let Some(arg) = it.next() {
            let mut value = || it.next().with_context(|| format!("{arg} needs a value"));
            match arg.as_str() {
                "--contexts" => contexts = Some(token_counts(&arg, &value()?)?),
                "--prefill" => prefill = Some(token_counts(&arg, &value()?)?),
                "--chunk" => chunk = value()?.parse()?,
                "--passes" => passes = Some(value()?.parse()?),
                "--steps" => steps = value()?.parse()?,
                "--warmup" => warmup = value()?.parse()?,
                "--kv" => kv = value()?.parse()?,
                "--json" => json = true,
                _ if arg.starts_with("--") => anyhow::bail!("unknown flag {arg}\n{USAGE}"),
                _ => {
                    ensure!(model.is_none(), "{USAGE}");
                    model = Some(PathBuf::from(arg));
                }
            }
        }
        ensure!(steps > 0, "--steps must be positive");
        ensure!(
            (1..=MAX_PREFILL_CHUNK).contains(&chunk),
            "--chunk takes 1 to {MAX_PREFILL_CHUNK} rows"
        );
        ensure!(
            prefill.as_ref().is_none_or(|p| p.iter().all(|&n| n > 0)),
            "--prefill lengths must be positive"
        );
        Ok(Args {
            model: model.context(USAGE)?,
            contexts,
            passes,
            steps,
            warmup,
            kv,
            json,
            prefill,
            chunk,
        })
    }

    /// A snapshot of the GPU from `nvidia-smi`.
    struct GpuState {
        timestamp: String,
        name: String,
        driver: String,
        sm_mhz: String,
        mem_mhz: String,
        max_sm_mhz: String,
        max_mem_mhz: String,
        temperature_c: String,
        power_w: String,
        /// `pid:MiB` for every compute process, this one included.
        processes: Vec<String>,
    }

    fn nvidia_smi(args: &[&str]) -> anyhow::Result<String> {
        let out = Command::new("nvidia-smi")
            .args(args)
            .output()
            .context("running nvidia-smi")?;
        ensure!(out.status.success(), "nvidia-smi {args:?} failed");
        Ok(String::from_utf8(out.stdout)?)
    }

    fn gpu_state() -> anyhow::Result<GpuState> {
        let fields = "timestamp,name,driver_version,clocks.sm,clocks.mem,clocks.max.sm,\
                      clocks.max.mem,temperature.gpu,power.draw";
        let line = nvidia_smi(&[
            "--id=0",
            &format!("--query-gpu={fields}"),
            "--format=csv,noheader,nounits",
        ])?;
        let v: Vec<String> = line.trim().split(", ").map(str::to_owned).collect();
        let [
            timestamp,
            name,
            driver,
            sm,
            mem,
            max_sm,
            max_mem,
            temp,
            power,
        ] = <[String; 9]>::try_from(v)
            .ok()
            .with_context(|| format!("unexpected nvidia-smi output: {line}"))?;
        let processes = nvidia_smi(&[
            "--id=0",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])?
        .lines()
        .map(|l| l.replace(", ", ":"))
        .collect();
        Ok(GpuState {
            timestamp,
            name,
            driver,
            sm_mhz: sm,
            mem_mhz: mem,
            max_sm_mhz: max_sm,
            max_mem_mhz: max_mem,
            temperature_c: temp,
            power_w: power,
            processes,
        })
    }

    struct Row {
        context: usize,
        /// Seconds per step, in run order.
        steps: Vec<f64>,
        /// Seconds between the synchronized ends of the timed run.
        total: f64,
        kv_bytes: f64,
        /// Clocks read right after the timed run, while the GPU is still at speed.
        sm_mhz: String,
        mem_mhz: String,
    }

    impl Row {
        fn mean(&self) -> f64 {
            self.total / self.steps.len() as f64
        }

        /// The `p`th percentile of the step times, nearest rank.
        fn percentile(&self, p: usize) -> f64 {
            let mut sorted = self.steps.clone();
            sorted.sort_by(f64::total_cmp);
            let rank = (p * sorted.len()).div_ceil(100);
            sorted[rank.clamp(1, sorted.len()) - 1]
        }

        fn weight_fraction(&self) -> f64 {
            WEIGHT_BYTES_PER_TOKEN / self.mean() / PEAK_BANDWIDTH
        }

        fn total_fraction(&self) -> f64 {
            (WEIGHT_BYTES_PER_TOKEN + self.kv_bytes) / self.mean() / PEAK_BANDWIDTH
        }
    }

    /// One timed prefill.
    struct PrefillRow {
        tokens: usize,
        seconds: f64,
        sm_mhz: String,
        mem_mhz: String,
    }

    /// Times `Model::prefill` at each length in `lengths`.
    fn bench_prefill(args: &Args, lengths: &[usize]) -> anyhow::Result<()> {
        let config = TextConfig::load(&args.model)?;
        let passes = args.passes.unwrap_or(config.loop_config.num_passes);
        let vocab = u32::try_from(config.unpadded_vocab_size)?;
        let gpu = Gpu::new(0)?;
        let before = gpu_state()?;
        eprintln!("loading weights");
        let checkpoint = Checkpoint::open(&args.model)?;
        let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
        let longest = lengths.iter().copied().max().unwrap_or(0).max(args.chunk);
        let mut model = Model::with_kv_format(&gpu, config, weights, passes, longest, args.kv)?;
        // Arbitrary ids spread over the vocabulary.
        let prompt: Vec<u32> = (0..longest as u32)
            .map(|i| (i.wrapping_mul(7919) + 13) % vocab)
            .collect();

        // One chunk first, which allocates the prefill buffers.
        model.prefill(&prompt[..args.chunk], args.chunk, PrefillLogits::Last)?;
        let mut rows = Vec::new();
        for &tokens in lengths {
            model.reset()?;
            gpu.stream().synchronize()?;
            let start = Instant::now();
            model.prefill(&prompt[..tokens], args.chunk, PrefillLogits::Last)?;
            let seconds = start.elapsed().as_secs_f64();
            let after = gpu_state()?;
            rows.push(PrefillRow {
                tokens,
                seconds,
                sm_mhz: after.sm_mhz,
                mem_mhz: after.mem_mhz,
            });
            if !args.json {
                eprintln!("prefill {tokens} done");
            }
        }

        if args.json {
            let processes: Vec<String> = before.processes.iter().map(|p| quote(p)).collect();
            let rows: Vec<String> = rows
                .iter()
                .map(|r| {
                    format!(
                        "{{\"tokens\":{},\"ttft_ms\":{:.2},\"tok_per_s\":{:.1},\"sm_mhz\":{},\
                         \"mem_mhz\":{}}}",
                        r.tokens,
                        r.seconds * 1e3,
                        r.tokens as f64 / r.seconds,
                        quote(&r.sm_mhz),
                        quote(&r.mem_mhz),
                    )
                })
                .collect();
            println!(
                "{{\"timestamp\":{},\"gpu\":{},\"driver\":{},\"processes_before_load\":[{}],\
                 \"passes\":{passes},\"kv\":{},\"chunk\":{},\"rows\":[{}]}}",
                quote(&before.timestamp),
                quote(&before.name),
                quote(&before.driver),
                processes.join(","),
                quote(&args.kv.to_string()),
                args.chunk,
                rows.join(","),
            );
        } else {
            println!(
                "{} | {} | driver {} | max clocks SM {} MHz, memory {} MHz | compute processes \
                 before load: {}",
                before.timestamp,
                before.name,
                before.driver,
                before.max_sm_mhz,
                before.max_mem_mhz,
                if before.processes.is_empty() {
                    "none".to_owned()
                } else {
                    before.processes.join(" ")
                },
            );
            println!(
                "passes {passes}, {} global KV cache, prefill chunks of {}\n",
                args.kv, args.chunk
            );
            println!(
                "| prompt tokens | time to first token ms | prefill tok/s | SM MHz | mem MHz |"
            );
            println!("|---:|---:|---:|---:|---:|");
            for r in &rows {
                println!(
                    "| {} | {:.1} | {:.1} | {} | {} |",
                    r.tokens,
                    r.seconds * 1e3,
                    r.tokens as f64 / r.seconds,
                    r.sm_mhz,
                    r.mem_mhz,
                );
            }
        }
        Ok(())
    }

    pub fn main() -> anyhow::Result<()> {
        let mut args = args()?;
        if let Some(lengths) = args.prefill.take() {
            return bench_prefill(&args, &lengths);
        }
        let config = TextConfig::load(&args.model)?;
        let passes = args.passes.unwrap_or(config.loop_config.num_passes);
        let global_visits = config
            .visit_layers(passes)
            .into_iter()
            .filter(|&layer| !config.is_sliding(layer))
            .count();
        // Keys and values in every global visit.
        let kv_bytes_per_token = global_visits
            * 2
            * args.kv.element_bytes()
            * config.num_key_value_heads
            * config.head_dim;
        let tail = args.warmup + args.steps;

        let gpu = Gpu::new(0)?;
        let before = gpu_state()?;
        eprintln!("loading weights");
        let checkpoint = Checkpoint::open(&args.model)?;
        let weights = Weights::load(&checkpoint, &config, gpu.stream())?;

        let contexts = if let Some(c) = args.contexts.take() {
            c
        } else {
            let mut c = DEFAULT_CONTEXTS.to_vec();
            let (free, _) = gpu.context().mem_get_info()?;
            let need = (LONG_CONTEXT + tail) * kv_bytes_per_token + HEADROOM;
            if need <= free {
                c.push(LONG_CONTEXT);
            } else {
                eprintln!(
                    "skipping context {LONG_CONTEXT}: needs {} MiB, {} MiB free",
                    need >> 20,
                    free >> 20
                );
            }
            c
        };
        let capacity = contexts.iter().max().copied().unwrap_or(0) + tail;
        let mut model = Model::with_kv_format(&gpu, config, weights, passes, capacity, args.kv)?;

        let mut rows = Vec::new();
        let mut token = 1;
        for &context in &contexts {
            model.skip_to_for_benchmark(context)?;
            for _ in 0..args.warmup {
                token = model.step(token, None)?;
            }
            let mut steps = Vec::with_capacity(args.steps);
            gpu.stream().synchronize()?;
            let start = Instant::now();
            for _ in 0..args.steps {
                let t = Instant::now();
                token = model.step(token, None)?;
                steps.push(t.elapsed().as_secs_f64());
            }
            gpu.stream().synchronize()?;
            let total = start.elapsed().as_secs_f64();
            let after = gpu_state()?;
            // Mean context over the timed steps.
            let mean_context = context + args.warmup + args.steps / 2;
            rows.push(Row {
                context,
                steps,
                total,
                kv_bytes: (mean_context * kv_bytes_per_token) as f64,
                sm_mhz: after.sm_mhz,
                mem_mhz: after.mem_mhz,
            });
            if !args.json {
                eprintln!("context {context} done");
            }
        }

        if args.json {
            println!("{}", json(&before, passes, &args, &rows));
        } else {
            print!("{}", markdown(&before, passes, &args, &rows));
        }
        Ok(())
    }

    fn markdown(gpu: &GpuState, passes: usize, args: &Args, rows: &[Row]) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "{} | {} | driver {} | max clocks SM {} MHz, memory {} MHz | {} C, {} W before load | \
             compute processes before load: {}",
            gpu.timestamp,
            gpu.name,
            gpu.driver,
            gpu.max_sm_mhz,
            gpu.max_mem_mhz,
            gpu.temperature_c,
            gpu.power_w,
            if gpu.processes.is_empty() {
                "none".to_owned()
            } else {
                gpu.processes.join(" ")
            },
        );
        let _ = writeln!(
            s,
            "passes {passes}, {} global KV cache, {} warmup steps, {} timed steps\n",
            args.kv, args.warmup, args.steps
        );
        let _ = writeln!(
            s,
            "| context | mean ms | median ms | p90 ms | tok/s | weight BW | weight+KV BW | \
             SM MHz | mem MHz |"
        );
        let _ = writeln!(s, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for r in rows {
            let _ = writeln!(
                s,
                "| {} | {:.3} | {:.3} | {:.3} | {:.1} | {:.1}% | {:.1}% | {} | {} |",
                r.context,
                r.mean() * 1e3,
                r.percentile(50) * 1e3,
                r.percentile(90) * 1e3,
                1.0 / r.mean(),
                r.weight_fraction() * 100.0,
                r.total_fraction() * 100.0,
                r.sm_mhz,
                r.mem_mhz,
            );
        }
        s
    }

    fn quote(s: &str) -> String {
        format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
    }

    fn json(gpu: &GpuState, passes: usize, args: &Args, rows: &[Row]) -> String {
        let processes: Vec<String> = gpu.processes.iter().map(|p| quote(p)).collect();
        let rows: Vec<String> = rows
            .iter()
            .map(|r| {
                format!(
                    "{{\"context\":{},\"mean_ms\":{:.4},\"median_ms\":{:.4},\"p90_ms\":{:.4},\
                     \"tok_per_s\":{:.2},\"weight_bw_fraction\":{:.4},\
                     \"weight_kv_bw_fraction\":{:.4},\"sm_mhz\":{},\"mem_mhz\":{}}}",
                    r.context,
                    r.mean() * 1e3,
                    r.percentile(50) * 1e3,
                    r.percentile(90) * 1e3,
                    1.0 / r.mean(),
                    r.weight_fraction(),
                    r.total_fraction(),
                    quote(&r.sm_mhz),
                    quote(&r.mem_mhz),
                )
            })
            .collect();
        format!(
            "{{\"timestamp\":{},\"gpu\":{},\"driver\":{},\"max_sm_mhz\":{},\"max_mem_mhz\":{},\
             \"temperature_c_before_load\":{},\"power_w_before_load\":{},\"processes_before_load\":[{}],\
             \"passes\":{passes},\"kv\":{},\"warmup\":{},\"steps\":{},\"rows\":[{}]}}",
            quote(&gpu.timestamp),
            quote(&gpu.name),
            quote(&gpu.driver),
            quote(&gpu.max_sm_mhz),
            quote(&gpu.max_mem_mhz),
            quote(&gpu.temperature_c),
            quote(&gpu.power_w),
            processes.join(","),
            quote(&args.kv.to_string()),
            args.warmup,
            args.steps,
            rows.join(","),
        )
    }
}
