//! Times greedy decode at several context lengths, or prefill at several prompt lengths.
//!
//! usage: bench <model-dir> [--contexts 0,4096,32768] [--passes N] [--steps N]
//!        [--warmup N] [--kv bf16|fp8] [--weights bf16|fp8] [--json]
//!        bench <model-dir> --prefill 512,4096,32768 [--chunk N] [--passes N]
//!        [--kv bf16|fp8] [--expert-activations bf16|quantized] [--json]
//!
//! With `--prefill`, each prompt length runs once through `Model::prefill` in chunks of
//! `--chunk` rows from an empty cache, after one warmup chunk, and the time to the first
//! token is the time `prefill` takes to return the greedy next token. The prompt tokens
//! are arbitrary ids, so expert routing differs from real text. `--expert-activations
//! quantized` runs prefill's expert products on quantized activations (see
//! `ExpertActivations`).
//!
//! For each context length the runner skips its caches to that length without computing
//! the skipped tokens, runs `--warmup` steps, then times `--steps` steps. The tokens it
//! produces are meaningless, but each step does the same work as a real one. Without
//! `--contexts`, it runs 0, 4096 and 32768, and 131072 if the cache fits in free memory.
//! `--kv fp8` stores the global-layer KV caches as E4M3 instead of BF16. `--weights fp8`
//! makes decode read FP8 copies of the attention projections and the LM head.
//!
//! Results are comparable only on an otherwise idle GPU, so the output records the GPU
//! clocks, the processes on the GPU and the time.
//!
//! Speculative decoding, from the tokens of a capture (`--prompt`, a safetensors file
//! with a `tokens` tensor such as `reference/capture_eval.py` writes):
//!
//!        bench <model-dir> --prompt <capture> --spec K|auto [--widths 2,4,8]
//!        [--draft SPEC] [--shortlist FILE] [--sweep FILE] [--contexts 0,32768]
//!        [--steps N] [--weights bf16|fp8] [--kv bf16|fp8] [--drafter DIR]
//!        bench <model-dir> --prompt <capture> --verify-widths [--contexts 0,32768]
//!
//! The prompt is the capture's tokens except the last 512, which `--verify-widths` uses
//! as verify rows. A context longer than the prompt puts arbitrary token ids before it.
//! `--spec K` prefills the prompt once, then from that same state generates `--steps`
//! tokens with plain decode and with greedy speculation verifying K drafts per round,
//! and reports both rates, the rounds, tokens per round, per-position acceptance and the
//! split of a round between verify, draft and host. `--spec auto` picks each round's
//! width from the drafts' confidences (`WidthPolicy`), among `--widths` (default 1 to
//! 8), priced by verify passes of each width timed at that context first. `--draft`
//! takes a variant of the draft as `dspark` does: `fast` (the default, every
//! approximation), `exact`, or a `+`-joined set of `fp8-backbone`, `fp8-head`, `fp8-fold`
//! and `shortlist`, which ranks tokens by id unless `--shortlist` names a ranking file.
//! `--drafter DIR` runs the drafter in DIR instead of the bundled draft, with the same
//! variants: Inco AI's DFlash 2 drafter (`Dflash2Weights`) or a DFlash draft with DSpark
//! heads exported by the 2DSD project (`DflashWeights`), told apart by its `config.json`.
//! `--sweep FILE` also decodes `--steps` tokens one verify row at a time with a draft
//! round of 7 at every anchor, and writes the tokens and every anchor's drafts and
//! confidences to FILE as JSON lines, one per context. The verify and the draft's
//! shared LM head read the weights `--weights` names. `--verify-widths` times one verify
//! pass of each width from 1 to 8, each keeping one row, so the next starts one position
//! later.

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
    use siphon::{Checkpoint, TextConfig};
    use siphon_cuda::{
        Gpu,
        model::{DEFAULT_PREFILL_CHUNK, MAX_PREFILL_CHUNK, Model, PrefillLogits},
        ops::{attention::KvFormat, experts::ExpertActivations},
        weights::{WeightFormat, Weights},
    };

    /// Bytes of weights one decode step reads.
    const WEIGHT_BYTES_PER_TOKEN: f64 = 5.5e9;
    /// Bytes `--weights fp8` saves per step: the 3.51 GB of BF16 attention projections
    /// and LM head read as 1.76 GB of E4M3 values and scales.
    const FP8_SAVED_BYTES_PER_TOKEN: f64 = 1.75e9;
    /// Device memory the FP8 copies take: one per layer, 1.11 GB.
    const FP8_COPY_BYTES: usize = 1_120_000_000;
    /// RTX 5090 DRAM bandwidth.
    const PEAK_BANDWIDTH: f64 = 1.79e12;
    const DEFAULT_CONTEXTS: [usize; 3] = [0, 4096, 32768];
    /// Added to the defaults when its cache fits.
    const LONG_CONTEXT: usize = 131_072;
    /// Free memory to leave for the runner's per-step buffers.
    const HEADROOM: usize = 2 << 30;
    const USAGE: &str = "usage: bench <model-dir> [--contexts 0,4096,32768] [--passes N] \
                         [--steps N] [--warmup N] [--kv bf16|fp8] [--weights bf16|fp8] \
                         [--json]\n       \
                         bench <model-dir> --prefill 512,4096 [--chunk N] [--passes N] \
                         [--kv bf16|fp8] [--json]";

    struct Args {
        model: PathBuf,
        contexts: Option<Vec<usize>>,
        passes: Option<usize>,
        steps: usize,
        warmup: usize,
        kv: KvFormat,
        weights: WeightFormat,
        expert_activations: ExpertActivations,
        json: bool,
        /// Prompt lengths to time prefill at, instead of timing decode.
        prefill: Option<Vec<usize>>,
        chunk: usize,
        /// The capture whose tokens `--spec` and `--verify-widths` use.
        prompt: Option<PathBuf>,
        /// Drafts per round for speculative decoding, or 0 for an adaptive width.
        spec: Option<usize>,
        /// The widths `--spec auto` picks from.
        widths: Vec<usize>,
        /// The draft variant and the shortlist ranking file.
        draft: String,
        shortlist: Option<PathBuf>,
        /// A DFlash draft's export directory, in place of the bundled draft.
        drafter: Option<PathBuf>,
        /// Where `--spec` writes the sweep, if anywhere.
        sweep: Option<PathBuf>,
        verify_widths: bool,
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
        let mut weights = WeightFormat::Bf16;
        let mut expert_activations = ExpertActivations::Bf16;
        let mut json = false;
        let mut prefill = None;
        let mut chunk = DEFAULT_PREFILL_CHUNK;
        let mut prompt = None;
        let mut spec = None;
        let mut widths = (1..=8).collect();
        let mut draft = "fast".to_owned();
        let mut shortlist = None;
        let mut drafter = None;
        let mut sweep = None;
        let mut verify_widths = false;
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
                "--weights" => weights = value()?.parse()?,
                "--expert-activations" => expert_activations = value()?.parse()?,
                "--json" => json = true,
                "--prompt" => prompt = Some(PathBuf::from(value()?)),
                "--spec" => {
                    let v = value()?;
                    spec = Some(if v == "auto" { 0 } else { v.parse()? });
                }
                "--widths" => widths = token_counts(&arg, &value()?)?,
                "--draft" => draft = value()?,
                "--shortlist" => shortlist = Some(PathBuf::from(value()?)),
                "--drafter" => drafter = Some(PathBuf::from(value()?)),
                "--sweep" => sweep = Some(PathBuf::from(value()?)),
                "--verify-widths" => verify_widths = true,
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
            weights,
            expert_activations,
            json,
            prefill,
            chunk,
            prompt,
            spec,
            widths,
            draft,
            shortlist,
            drafter,
            sweep,
            verify_widths,
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
        /// Weight bytes one step reads.
        weight_bytes: f64,
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
            self.weight_bytes / self.mean() / PEAK_BANDWIDTH
        }

        fn total_fraction(&self) -> f64 {
            (self.weight_bytes + self.kv_bytes) / self.mean() / PEAK_BANDWIDTH
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
        model.set_expert_activations(args.expert_activations);
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
                 \"passes\":{passes},\"kv\":{},\"expert_activations\":{},\"chunk\":{},\
                 \"rows\":[{}]}}",
                quote(&before.timestamp),
                quote(&before.name),
                quote(&before.driver),
                processes.join(","),
                quote(&args.kv.to_string()),
                quote(&args.expert_activations.to_string()),
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
        if args.spec.is_some() || args.verify_widths {
            return speculation::run(&args);
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
            let mut need = (LONG_CONTEXT + tail) * kv_bytes_per_token + HEADROOM;
            if args.weights == WeightFormat::Fp8 {
                need += FP8_COPY_BYTES;
            }
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
        let mut model = Model::with_formats(
            &gpu,
            config,
            weights,
            passes,
            capacity,
            args.kv,
            args.weights,
        )?;
        let weight_bytes = match args.weights {
            WeightFormat::Bf16 => WEIGHT_BYTES_PER_TOKEN,
            WeightFormat::Fp8 => WEIGHT_BYTES_PER_TOKEN - FP8_SAVED_BYTES_PER_TOKEN,
        };

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
                weight_bytes,
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
            "passes {passes}, {} global KV cache, {} dense weights, {} warmup steps, {} timed \
             steps\n",
            args.kv, args.weights, args.warmup, args.steps
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
             \"passes\":{passes},\"kv\":{},\"weights\":{},\"warmup\":{},\"steps\":{},\"rows\":[{}]}}",
            quote(&gpu.timestamp),
            quote(&gpu.name),
            quote(&gpu.driver),
            quote(&gpu.max_sm_mhz),
            quote(&gpu.max_mem_mhz),
            quote(&gpu.temperature_c),
            quote(&gpu.power_w),
            processes.join(","),
            quote(&args.kv.to_string()),
            quote(&args.weights.to_string()),
            args.warmup,
            args.steps,
            rows.join(","),
        )
    }

    /// `--spec` and `--verify-widths`.
    mod speculation {
        use std::{fmt::Write as _, fs, time::Instant};

        use anyhow::{Context as _, ensure};
        use safetensors::{Dtype, SafeTensors};
        use siphon::{Checkpoint, TextConfig};
        use siphon_cuda::{
            Gpu,
            draft::{DraftOptions, read_ranking},
            model::{DEFAULT_PREFILL_CHUNK, Model, SavedState},
            ops::spec::VERIFY_ROWS,
            spec::{
                DraftSource, Drafts, Speculator, SweepRow, Width, WidthPolicy, load_draft, price,
                verify_times,
            },
            weights::{WeightFormat, Weights},
        };

        use super::{Args, gpu_state};

        /// Tokens at the end of the capture kept out of the prompt, for verify rows.
        const RESERVED: usize = 512;
        /// Default tokens generated per context.
        const DEFAULT_STEPS: usize = 256;
        /// Timed verify passes per width.
        const VERIFY_REPEATS: usize = 40;
        /// Bytes the byte model of a verify uses: dense weights in BF16 or with
        /// `--weights fp8` (the attention projections and LM head as E4M3 values and
        /// scales), and one expert's `w13` and `w2` with scales in NVFP4 (44 visits) and
        /// MXFP8 (4 visits).
        const DENSE_BYTES: f64 = 3.62e9;
        const DENSE_FP8_BYTES: f64 = 1.87e9;
        const NVFP4_EXPERT: f64 = 4.48e6;
        const MXFP8_EXPERT: f64 = 8.21e6;
        /// DRAM bandwidth times plain decode's efficiency, as the design's cost model.
        const EFFECTIVE_BANDWIDTH: f64 = 1.70e12 * 0.84;

        fn tokens(path: &std::path::Path) -> anyhow::Result<Vec<u32>> {
            let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            let capture = SafeTensors::deserialize(&bytes)?;
            let view = capture.tensor("tokens")?;
            ensure!(view.dtype() == Dtype::I32, "tokens are {:?}", view.dtype());
            view.data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| Ok(u32::try_from(i32::from_le_bytes(*b))?))
                .collect()
        }

        /// The prompt for `context`: arbitrary ids, then `text`, `context` tokens in all, or
        /// `text` alone when it is longer.
        fn prompt(text: &[u32], context: usize, vocab: u32) -> Vec<u32> {
            let filler = context.saturating_sub(text.len());
            (0..filler as u32)
                .map(|i| (i.wrapping_mul(7919) + 13) % vocab)
                .chain(text.iter().copied())
                .collect()
        }

        /// The design's byte model of a verify of `m` rows: dense weights once and each
        /// distinct expert once per visit, with `8 m^0.6` distinct experts per visit.
        fn modeled_ms(m: usize, weights: WeightFormat) -> f64 {
            let distinct = 8.0 * (m as f64).powf(0.6);
            let dense = match weights {
                WeightFormat::Bf16 => DENSE_BYTES,
                WeightFormat::Fp8 => DENSE_FP8_BYTES,
            };
            let bytes = dense + distinct * (44.0 * NVFP4_EXPERT + 4.0 * MXFP8_EXPERT);
            bytes / EFFECTIVE_BANDWIDTH * 1e3
        }

        pub fn run(args: &Args) -> anyhow::Result<()> {
            let path = args
                .prompt
                .as_ref()
                .context("--spec and --verify-widths need --prompt")?;
            let all = tokens(path)?;
            ensure!(all.len() > RESERVED + 16, "the capture is too short");
            let (text, rows) = all.split_at(all.len() - RESERVED);
            let config = TextConfig::load(&args.model)?;
            let passes = args.passes.unwrap_or(config.loop_config.num_passes);
            let vocab = u32::try_from(config.unpadded_vocab_size)?;
            let contexts = args.contexts.clone().unwrap_or_else(|| vec![0, 32768]);
            let steps = if args.steps == 100 {
                DEFAULT_STEPS
            } else {
                args.steps
            };
            let longest = contexts.iter().copied().max().unwrap_or(0).max(text.len());
            let capacity = longest + steps + RESERVED + 64;

            let gpu = Gpu::new(0)?;
            let before = gpu_state()?;
            eprintln!("loading weights");
            let checkpoint = Checkpoint::open(&args.model)?;
            let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
            let mut model = Model::with_formats(
                &gpu,
                config,
                weights,
                passes,
                capacity,
                args.kv,
                args.weights,
            )?;
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
                "passes {passes}, {} global KV cache, {} weights for decode, verify and the \
                 draft's LM head, draft {} {}, prompt {} ({} tokens)\n",
                args.kv,
                args.weights,
                args.drafter
                    .as_ref()
                    .map_or_else(|| "bundled".to_owned(), |d| d.display().to_string()),
                args.draft,
                path.display(),
                text.len()
            );
            if args.verify_widths {
                verify_widths(&mut model, text, rows, &contexts, vocab, args.weights)?;
            }
            if let Some(k) = args.spec {
                let ranking = args.shortlist.as_deref().map(read_ranking).transpose()?;
                let options = DraftOptions::parse(&args.draft, ranking.as_deref())?;
                let width = if k == 0 {
                    // Priced at each context before its run.
                    Width::Adaptive(WidthPolicy::new(&args.widths, [1.0; VERIFY_ROWS])?)
                } else {
                    Width::Fixed(k)
                };
                let source = args
                    .drafter
                    .as_deref()
                    .map_or(DraftSource::Bundled, DraftSource::Dflash);
                let draft = load_draft(&mut model, &checkpoint, source, capacity, &options)?;
                let mut speculator = Speculator::with_draft(&model, draft, width)?;
                spec(
                    &mut model,
                    &mut speculator,
                    text,
                    rows,
                    &contexts,
                    steps,
                    vocab,
                    args,
                )?;
            }
            Ok(())
        }

        fn verify_widths(
            model: &mut Model,
            text: &[u32],
            rows: &[u32],
            contexts: &[usize],
            vocab: u32,
            weights: WeightFormat,
        ) -> anyhow::Result<()> {
            println!(
                "| context | rows | min ms | median ms | byte model ms | min ms per row | SM MHz \
                 | mem MHz |"
            );
            println!("|---:|---:|---:|---:|---:|---:|---:|---:|");
            for &context in contexts {
                let prompt = prompt(text, context, vocab);
                model.reset()?;
                model.prefill(
                    &prompt,
                    DEFAULT_PREFILL_CHUNK,
                    siphon_cuda::model::PrefillLogits::Last,
                )?;
                let saved = model.save_state()?;
                for m in 1..=8 {
                    let mut times = verify_times(model, &saved, rows, m, VERIFY_REPEATS)?;
                    times.sort_by(f64::total_cmp);
                    let after = gpu_state()?;
                    println!(
                        "| {} | {m} | {:.3} | {:.3} | {:.2} | {:.3} | {} | {} |",
                        prompt.len(),
                        times[0],
                        times[times.len() / 2],
                        modeled_ms(m, weights),
                        times[0] / m as f64,
                        after.sm_mhz,
                        after.mem_mhz
                    );
                }
            }
            println!();
            Ok(())
        }

        /// Prices an adaptive width at the state `saved` (see [`price`]) and returns the
        /// label of the speculator's width.
        fn set_price(
            model: &mut Model,
            speculator: &mut Speculator,
            saved: &SavedState,
            rows: &[u32],
            first: u32,
            context: usize,
        ) -> anyhow::Result<String> {
            Ok(match speculator.width().clone() {
                Width::Adaptive(policy) => {
                    let round_ms = price(model, speculator, saved, rows, first)?;
                    let priced = policy
                        .widths()
                        .iter()
                        .map(|&w| format!("{w}: {:.3}", round_ms[w - 1]))
                        .collect::<Vec<_>>()
                        .join(", ");
                    println!("round ms by width at context {context}: {priced}");
                    speculator.set_width(Width::Adaptive(WidthPolicy::new(
                        policy.widths(),
                        round_ms,
                    )?))?;
                    format!("auto {:?}", policy.widths())
                }
                Width::Fixed(k) => format!("k = {k}"),
            })
        }

        /// Appends a sweep to `path` as one JSON line.
        fn write_sweep(
            path: &std::path::Path,
            context: usize,
            first: u32,
            tokens: &[u32],
            rows: &[SweepRow],
        ) -> anyhow::Result<()> {
            let mut line = format!(
                "{{\"context\":{context},\"first\":{first},\"tokens\":{tokens:?},\"drafts\":["
            );
            for (i, row) in rows.iter().enumerate() {
                let sep = if i == 0 { "" } else { "," };
                write!(line, "{sep}{:?}", row.drafts)?;
            }
            line.push_str("],\"confidence\":[");
            for (i, row) in rows.iter().enumerate() {
                let sep = if i == 0 { "" } else { "," };
                write!(line, "{sep}{:?}", row.confidence)?;
            }
            line.push_str("]}\n");
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            std::io::Write::write_all(&mut file, line.as_bytes())?;
            println!(
                "  sweep of {} anchors appended to {}",
                rows.len(),
                path.display()
            );
            Ok(())
        }

        #[expect(clippy::too_many_arguments, reason = "the run's inputs")]
        fn spec(
            model: &mut Model,
            speculator: &mut Speculator,
            text: &[u32],
            rows: &[u32],
            contexts: &[usize],
            steps: usize,
            vocab: u32,
            args: &Args,
        ) -> anyhow::Result<()> {
            for &context in contexts {
                let prompt = prompt(text, context, vocab);
                model.reset()?;
                let first = speculator.prefill(model, &prompt, DEFAULT_PREFILL_CHUNK)?;
                let saved = model.save_state()?;
                let label = set_price(model, speculator, &saved, rows, first, prompt.len())?;

                // Plain decode from the same state.
                let mut plain = Vec::with_capacity(steps);
                let mut token = first;
                for _ in 0..args.warmup.min(steps) {
                    token = model.step(token, None)?;
                }
                model.restore_state(&saved)?;
                token = first;
                model.stream().synchronize()?;
                let started = Instant::now();
                for _ in 0..steps {
                    token = model.step(token, None)?;
                    plain.push(token);
                }
                let plain_seconds = started.elapsed().as_secs_f64();
                let plain_clocks = gpu_state()?;

                // Speculation from the same state, after one warmup run that captures the
                // graphs.
                model.restore_state(&saved)?;
                speculator.restart(model, first)?;
                speculator.generate(model, 16, Drafts::Dspark)?;
                model.restore_state(&saved)?;
                speculator.restart(model, first)?;
                let out = speculator.generate(model, steps, Drafts::Dspark)?;
                let spec_clocks = gpu_state()?;
                let stats = speculator.stats();
                let agree = out.iter().zip(&plain).take_while(|(a, b)| a == b).count();
                let spec_rate = stats.emitted as f64 / stats.wall.as_secs_f64();
                let plain_rate = steps as f64 / plain_seconds;
                let per_round =
                    |d: std::time::Duration| d.as_secs_f64() * 1e3 / stats.rounds as f64;
                let acceptance: Vec<String> = stats
                    .acceptance()
                    .iter()
                    .map(|a| format!("{:.1}%", a * 100.0))
                    .collect();
                println!("context {} tokens, {steps} generated:", prompt.len());
                println!(
                    "  plain decode: {plain_rate:.1} tok/s ({:.3} ms per token), SM {} MHz, mem {} MHz",
                    plain_seconds * 1e3 / steps as f64,
                    plain_clocks.sm_mhz,
                    plain_clocks.mem_mhz
                );
                println!(
                    "  speculation {label}: {spec_rate:.1} tok/s ({:.2}x), {} rounds, {:.3} tokens \
                     and {:.3} verify rows per round, acceptance by position [{}], SM {} MHz, \
                     mem {} MHz",
                    spec_rate / plain_rate,
                    stats.rounds,
                    stats.tokens_per_round(),
                    stats.rows_per_round(),
                    acceptance.join(", "),
                    spec_clocks.sm_mhz,
                    spec_clocks.mem_mhz
                );
                println!("  rounds by width 1 to 8: {:?}", stats.widths);
                println!(
                    "  per round: {:.3} ms = verify {:.3} + draft {:.3} + host {:.3}",
                    per_round(stats.wall),
                    per_round(stats.verify),
                    per_round(stats.draft),
                    per_round(stats.host())
                );
                println!(
                    "  agreement with plain decode: first {agree} of {steps} tokens{}",
                    if args.weights == WeightFormat::Fp8 {
                        " (decode reads FP8 weights)"
                    } else {
                        ""
                    }
                );
                if let Some(path) = &args.sweep {
                    model.restore_state(&saved)?;
                    speculator.restart(model, first)?;
                    let (tokens, rows) = speculator.sweep(model, steps)?;
                    write_sweep(path, prompt.len(), first, &tokens, &rows)?;
                }
            }
            Ok(())
        }
    }
}
