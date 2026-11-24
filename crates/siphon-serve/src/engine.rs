//! The GPU side: one thread owns the model and the draft and runs one request at a time,
//! in arrival order, from a bounded queue.

use std::{
    path::PathBuf,
    sync::mpsc::{SyncSender, TrySendError},
};

use tokio::sync::mpsc::UnboundedSender;

/// What a request asks the engine for.
#[cfg_attr(
    not(target_os = "linux"),
    expect(dead_code, reason = "only the Linux engine runs jobs")
)]
pub struct Job {
    pub prompt: Vec<u32>,
    /// Tokens to generate at most, the stop token excluded.
    pub max_tokens: usize,
    pub events: UnboundedSender<Event>,
}

/// Why generation ended.
#[cfg_attr(
    not(target_os = "linux"),
    expect(dead_code, reason = "only the Linux engine runs jobs")
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model ended its turn.
    Stop,
    /// The request's token limit.
    Length,
}

/// What the engine reports for a job, in order: `Tokens` any number of times, then
/// `Done` or `Failed`.
#[cfg_attr(
    not(target_os = "linux"),
    expect(dead_code, reason = "only the Linux engine runs jobs")
)]
#[derive(Debug)]
pub enum Event {
    Tokens(Vec<u32>),
    Done(Finish),
    Failed(String),
}

/// How to load the model.
#[cfg_attr(
    not(target_os = "linux"),
    expect(dead_code, reason = "only the Linux engine runs jobs")
)]
#[derive(Clone, Debug)]
pub struct Options {
    pub model_dir: PathBuf,
    /// Positions of the global KV cache: the prompt and the generated tokens. `None` takes
    /// the most that fit the memory budget.
    pub capacity: Option<usize>,
    /// Bytes of GPU memory the plan may use, the CUDA context included.
    pub memory: Option<usize>,
    pub fp8_weights: bool,
    pub fp8_kv: bool,
    /// Verify widths the adaptive policy chooses among.
    pub widths: Vec<usize>,
    /// Jobs that wait while one runs.
    pub queue: usize,
    /// A DFlash draft's export directory, in place of the bundled draft.
    pub drafter: Option<PathBuf>,
    /// The draft's approximations, as `DraftOptions::parse` takes them.
    pub draft: String,
    /// Whether the speculator falls back to plain steps (`WidthPolicy::with_plain`).
    pub fallback: bool,
}

/// Sends jobs to the engine thread.
#[derive(Clone)]
pub struct Engine {
    jobs: SyncSender<Job>,
    max_context: usize,
}

/// Why the engine did not take a job.
pub enum Refused {
    Busy,
    Stopped,
}

impl Engine {
    /// Prompt and generated tokens that one job may hold together.
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Queues `job`, or refuses it when the queue is full or the engine has stopped.
    pub fn submit(&self, job: Job) -> Result<(), Refused> {
        self.jobs.try_send(job).map_err(|e| match e {
            TrySendError::Full(_) => Refused::Busy,
            TrySendError::Disconnected(_) => Refused::Stopped,
        })
    }
}

#[cfg(not(target_os = "linux"))]
pub fn start(_options: &Options, _stop: Vec<u32>, _warmup: Vec<u32>) -> anyhow::Result<Engine> {
    anyhow::bail!("the engine needs Linux with an NVIDIA GPU")
}

#[cfg(target_os = "linux")]
pub use cuda::start;

#[cfg(target_os = "linux")]
mod cuda {
    use std::{
        ops::ControlFlow,
        sync::mpsc::{Receiver, sync_channel},
        thread,
        time::Instant,
    };

    use anyhow::Context as _;
    use siphon::{
        Checkpoint, TextConfig,
        memory::{self, CONTEXT_BYTES, DraftChoice, PlanInputs, Planner, SAFETY_MARGIN, gib},
    };
    use siphon_cuda::{
        Gpu,
        draft::DraftOptions,
        model::{DEFAULT_PREFILL_CHUNK, Model},
        ops::{attention::KvFormat, spec::VERIFY_ROWS},
        spec::{DraftSource, Drafts, Speculator, Width, WidthPolicy, load_draft, price},
        weights::{WeightFormat, Weights},
    };

    use super::{Engine, Event, Finish, Job, Options};

    /// Positions past the last generated token that a round may write: a full verify.
    const MARGIN: usize = VERIFY_ROWS + 1;

    /// Loads the model on a new thread, prices the verify widths on `warmup` (a prompt
    /// of at least 32 tokens), and returns once the thread takes jobs. `stop` holds the
    /// ids that end a turn.
    pub fn start(options: &Options, stop: Vec<u32>, warmup: Vec<u32>) -> anyhow::Result<Engine> {
        anyhow::ensure!(warmup.len() >= 32, "the warmup prompt is too short");
        let (jobs, queue) = sync_channel(options.queue);
        let (ready, loaded) = sync_channel(1);
        let options = options.clone();
        thread::Builder::new()
            .name("engine".to_owned())
            .spawn(move || match load(&options, &warmup) {
                Ok((model, speculator, capacity)) => {
                    let _ = ready.send(Ok(capacity));
                    run(model, speculator, &queue, &stop);
                }
                Err(e) => {
                    let _ = ready.send(Err(e));
                }
            })?;
        let capacity = loaded
            .recv()
            .context("the engine thread died while loading")??;
        Ok(Engine {
            jobs,
            max_context: capacity - MARGIN,
        })
    }

    /// Plans the GPU memory and picks the capacity, before loading anything: the one
    /// `options` asks for if it fits, else the most that fit.
    fn plan(
        options: &Options,
        planner: &Planner,
        free: usize,
        warmup: usize,
    ) -> anyhow::Result<usize> {
        let budget = memory::budget(free, options.memory);
        let limit = options.memory.map_or_else(String::new, |l| {
            format!(
                ", within --memory {} GiB less the context's {} GiB",
                gib(l),
                gib(CONTEXT_BYTES)
            )
        });
        let budget_line = format!(
            "budget {} GiB: {} GiB free less a {} GiB margin{limit}",
            gib(budget),
            gib(free),
            gib(SAFETY_MARGIN)
        );
        let capacity = match options.capacity {
            Some(capacity) => {
                let plan = planner.plan(capacity);
                anyhow::ensure!(
                    plan.total() <= budget,
                    "--context {capacity} does not fit: it needs {plan}; {budget_line}; \
                     the largest that fits is {}",
                    planner.max_capacity(budget).unwrap_or(0)
                );
                capacity
            }
            None => planner.max_capacity(budget).with_context(|| {
                format!(
                    "the model does not fit: it needs {}; {budget_line}",
                    planner.plan(0)
                )
            })?,
        };
        anyhow::ensure!(
            capacity > warmup + 128 + MARGIN,
            "a context of {capacity} is too small"
        );
        eprintln!("memory plan: {}", planner.plan(capacity));
        eprintln!("memory {budget_line}");
        Ok(capacity)
    }

    fn load(options: &Options, warmup: &[u32]) -> anyhow::Result<(Model, Speculator, usize)> {
        let started = Instant::now();
        let config = TextConfig::load(&options.model_dir)?;
        let passes = config.loop_config.num_passes;
        let gpu = Gpu::new(0)?;
        let (free, _) = gpu.context().mem_get_info()?;
        let checkpoint = Checkpoint::open(&options.model_dir)?;
        let kv = if options.fp8_kv {
            KvFormat::Fp8
        } else {
            KvFormat::Bf16
        };
        let weight_format = if options.fp8_weights {
            WeightFormat::Fp8
        } else {
            WeightFormat::Bf16
        };
        let draft_options = DraftOptions::parse(&options.draft, None)?;
        let planner = Planner::new(&PlanInputs {
            config: &config,
            checkpoint: &checkpoint,
            weights: weight_format,
            kv,
            draft: options
                .drafter
                .as_deref()
                .map_or(DraftChoice::Bundled(draft_options.approx()), |dir| {
                    DraftChoice::Dflash(dir, draft_options.approx())
                }),
            prefill_chunk: DEFAULT_PREFILL_CHUNK,
        })?;
        let capacity = plan(options, &planner, free, warmup.len())?;
        let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
        let mut model =
            Model::with_formats(&gpu, config, weights, passes, capacity, kv, weight_format)?;
        let width = Width::Adaptive(WidthPolicy::new(&options.widths, [1.0; VERIFY_ROWS])?);
        let source = options
            .drafter
            .as_deref()
            .map_or(DraftSource::Bundled, DraftSource::Dflash);
        let draft = load_draft(&mut model, &checkpoint, source, capacity, &draft_options)?;
        let mut speculator = Speculator::with_draft(&model, draft, width)?;
        eprintln!("loaded in {:.1} s", started.elapsed().as_secs_f64());

        // Price each width at the warmup prompt, which also captures the graphs.
        model.reset()?;
        let first = speculator.prefill(&mut model, warmup, DEFAULT_PREFILL_CHUNK)?;
        let saved = model.save_state()?;
        let policy = price(&mut model, &mut speculator, &saved, warmup, first)?;
        let priced: Vec<String> = options
            .widths
            .iter()
            .map(|&w| format!("{w}: {:.3}", policy.round_ms()[w - 1]))
            .collect();
        eprintln!(
            "round ms by width: {}; plain step {:.3}, probe {:.3}",
            priced.join(", "),
            policy.plain().map_or(f64::NAN, |p| p.step_ms),
            policy.plain().map_or(f64::NAN, |p| p.probe_ms)
        );
        let policy = if options.fallback {
            policy
        } else {
            policy.with_plain(None)?
        };
        speculator.set_width(Width::Adaptive(policy))?;
        Ok((model, speculator, capacity))
    }

    fn run(mut model: Model, mut speculator: Speculator, queue: &Receiver<Job>, stop: &[u32]) {
        for job in queue {
            if job.events.is_closed() {
                continue;
            }
            let started = Instant::now();
            let events = job.events.clone();
            match generate(&mut model, &mut speculator, &job, stop) {
                Ok((finish, emitted, prefill_s)) => {
                    let decode_s = started.elapsed().as_secs_f64() - prefill_s;
                    let stats = speculator.stats();
                    let acceptance: Vec<String> = stats
                        .acceptance()
                        .iter()
                        .map(|a| format!("{:.1}%", a * 100.0))
                        .collect();
                    eprintln!(
                        "job: {} prompt tokens, prefill {:.0} ms, {emitted} tokens in {:.2} s \
                         ({:.1} tok/s, {:.2} tokens per round, {} plain steps, {} probes, \
                         draft {:.3} ms per round, acceptance by position [{}]), {}",
                        job.prompt.len(),
                        prefill_s * 1e3,
                        decode_s,
                        emitted as f64 / decode_s.max(1e-9),
                        stats.tokens_per_round(),
                        stats.stepped.len(),
                        stats.probes,
                        stats.draft.as_secs_f64() * 1e3
                            / (stats.rounds + stats.probes).max(1) as f64,
                        acceptance.join(", "),
                        if events.is_closed() {
                            "client left".to_owned()
                        } else {
                            format!("{finish:?}")
                        },
                    );
                    let _ = events.send(Event::Done(finish));
                }
                Err(e) => {
                    eprintln!("job failed: {e:#}");
                    let _ = events.send(Event::Failed(format!("{e:#}")));
                }
            }
        }
    }

    /// Runs `job` and returns why it ended, the tokens it emitted, and the prefill's
    /// seconds.
    fn generate(
        model: &mut Model,
        speculator: &mut Speculator,
        job: &Job,
        stop: &[u32],
    ) -> anyhow::Result<(Finish, usize, f64)> {
        let started = Instant::now();
        model.reset()?;
        let first = speculator.prefill(model, &job.prompt, DEFAULT_PREFILL_CHUNK)?;
        let prefill_s = started.elapsed().as_secs_f64();
        let mut emitted = 0;
        let mut finish = Finish::Length;
        // Sends the tokens of `ids` that fit, up to a stop token; breaks when the job is
        // over or its client has left.
        let mut take = |ids: &[u32]| {
            let room = job.max_tokens - emitted;
            let end = ids.iter().position(|id| stop.contains(id));
            let keep = end.unwrap_or(ids.len()).min(room);
            if keep > 0 {
                emitted += keep;
                if job
                    .events
                    .send(Event::Tokens(ids[..keep].to_vec()))
                    .is_err()
                {
                    return ControlFlow::Break(());
                }
            }
            if end.is_some_and(|e| e <= room) {
                finish = Finish::Stop;
                return ControlFlow::Break(());
            }
            if emitted == job.max_tokens {
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        };
        if take(&[first]).is_continue() {
            speculator.generate_with(model, job.max_tokens, Drafts::Dspark, &mut take)?;
        }
        Ok((finish, emitted, prefill_s))
    }
}
