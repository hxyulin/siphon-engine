//! The GPU side: one thread owns the model and the draft and runs requests in arrival
//! order from a bounded queue.
//!
//! A request that arrives at an idle server runs alone with greedy speculation in slot 0.
//! With `streams` above 1, a request that arrives meanwhile stops the speculation after its
//! round, and from then on every request runs batched plain decode in its own slot, up to
//! `streams` at once, until the server is idle again. A new request's prompt prefills in
//! chunks between the batch's steps. Batched steps and speculation give each sequence the
//! same tokens, so a request's output does not depend on what else runs (see
//! `docs/batching.md`).
//!
//! A request that runs alone prefills through the speculator's prefix cache
//! (`Speculator::prefill_cached`): when its prompt starts with the tokens of the last such
//! prompt up to that prompt's last multiple of the prefill chunk, it prefills only the
//! rest. Batched requests prefill whole. Taking slot 0 for another sequence or rebuilding
//! the draft's state empties the cache.

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

/// What the engine reports for a job, in order: `Cached` at most once, `Tokens` any number
/// of times, then `Done` or `Failed`.
#[cfg_attr(
    not(target_os = "linux"),
    expect(dead_code, reason = "only the Linux engine runs jobs")
)]
#[derive(Debug)]
pub enum Event {
    /// Prompt tokens whose state came from the prefix cache.
    Cached(usize),
    Tokens(Vec<u32>),
    Done(Finish),
    Failed(String),
}

/// How to load the model.
#[cfg_attr(
    not(target_os = "linux"),
    expect(dead_code, reason = "only the Linux engine runs jobs")
)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "independent switches from the command line"
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
    /// Jobs that wait while `streams` run.
    pub queue: usize,
    /// Requests that run at once, each in its own sequence slot, 1 to 8.
    pub streams: usize,
    /// A DFlash draft's export directory, in place of the bundled draft.
    pub drafter: Option<PathBuf>,
    /// The draft's approximations, as `DraftOptions::parse` takes them.
    pub draft: String,
    /// Whether the speculator falls back to plain steps (`WidthPolicy::with_plain`).
    pub fallback: bool,
    /// Whether requests that run alone prefill through a prefix cache.
    pub prefix_cache: bool,
    /// Prefill's global attention products over an FP8 cache, as `PrefillAttention`
    /// parses them.
    pub prefill_attention: String,
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
        collections::VecDeque,
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
        Gpu, MemoryUse,
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
                Ok((model, speculator, capacity, check)) => {
                    let _ = ready.send(Ok(capacity));
                    run(model, speculator, check, &queue, &stop, options.streams);
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

    /// Compares the memory the process uses with its plan: once after warmup, when every
    /// buffer the plan counts exists, and again after each job.
    struct MemoryCheck {
        gpu: Gpu,
        /// Bytes the plan allows after the CUDA context.
        planned: usize,
        /// Bytes outside the memory pool after warmup: module images and CUDA graphs.
        outside_pool: usize,
        /// Whether a job has already gone past the plan, so that it warns once.
        warned: bool,
    }

    impl MemoryCheck {
        /// Measures the use after warmup against `planned`, with `free` the device's free
        /// memory right after creating the context.
        fn new(gpu: Gpu, planned: usize, free: usize) -> anyhow::Result<Self> {
            let now = gpu.memory()?;
            // Other processes can change the free memory meanwhile; the pool is exact.
            let used = free.saturating_sub(now.free);
            let outside_pool = used.saturating_sub(now.pool_reserved);
            let mut check = Self {
                gpu,
                planned,
                outside_pool,
                warned: false,
            };
            eprintln!(
                "memory check: {} GiB of a planned {} GiB in use after warmup ({} GiB in \
                 buffers, peak {} GiB; {} GiB in modules and graphs)",
                gib(check.peak(&now)),
                gib(planned),
                gib(now.pool_reserved),
                gib(now.pool_reserved_peak),
                gib(outside_pool),
            );
            check.warn(&now);
            Ok(check)
        }

        /// The process's peak use: the pool's peak and what lies outside it.
        fn peak(&self, now: &MemoryUse) -> usize {
            now.pool_reserved_peak + self.outside_pool
        }

        fn after_job(&mut self) {
            match self.gpu.memory() {
                Ok(now) => self.warn(&now),
                Err(e) => eprintln!("memory check failed: {e:#}"),
            }
        }

        fn warn(&mut self, now: &MemoryUse) {
            let peak = self.peak(now);
            if !self.warned && peak > self.planned {
                self.warned = true;
                eprintln!(
                    "warning: memory use peaked at {} GiB, {} MiB past the plan; the plan \
                     undercounts this configuration",
                    gib(peak),
                    (peak - self.planned) >> 20
                );
            }
        }
    }

    fn load(
        options: &Options,
        warmup: &[u32],
    ) -> anyhow::Result<(Model, Speculator, usize, MemoryCheck)> {
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
            streams: options.streams,
            prefix_cache: options.prefix_cache,
        })?;
        let capacity = plan(options, &planner, free, warmup.len())?;
        let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
        let mut model =
            Model::with_formats(&gpu, config, weights, passes, capacity, kv, weight_format)?;
        model.set_prefill_attention(options.prefill_attention.parse()?);
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
        if options.streams > 1 {
            // Capture the batched step of every size now, rather than in a request. Batched
            // steps record their rows' taps, so that a request that leaves speculation can
            // return to it (see `Speculator::follow`).
            model.add_slots(options.streams - 1)?;
            model.set_batch_taps(true);
            for size in 1..=options.streams {
                let rows: Vec<(usize, u32)> = (0..size).map(|s| (s, warmup[0])).collect();
                model.step_batch(&rows)?;
            }
            for slot in 0..options.streams {
                model.reset_slot(slot)?;
            }
            eprintln!("serving up to {} requests at once", options.streams);
        }
        drop(saved);
        if options.prefix_cache {
            let bytes = speculator.enable_prefix_cache(&model)?;
            eprintln!("prefix cache: {} MiB of copies", bytes >> 20);
        }
        // The full chunk's buffers now, so that the check below sees every planned buffer
        // and a plan that undercounts fails here rather than on the first long prompt.
        // After pricing, so that they reuse the memory pricing freed, as a long prompt's
        // prefill would.
        model.reserve_prefill(DEFAULT_PREFILL_CHUNK)?;
        let check = MemoryCheck::new(gpu, planner.plan(capacity).total(), free)?;
        Ok((model, speculator, capacity, check))
    }

    /// A request the engine has taken.
    #[expect(
        clippy::struct_excessive_bools,
        reason = "independent states of a request"
    )]
    struct Request {
        job: Job,
        slot: usize,
        started: Instant,
        /// Prompt tokens prefilled so far.
        prefilled: usize,
        /// Prompt tokens whose state came from the prefix cache.
        cached: usize,
        prefill_s: f64,
        /// The token to feed next, once the prompt is prefilled.
        next: Option<u32>,
        /// The tokens sent to the client. Every one but the last has been fed.
        output: Vec<u32>,
        finish: Finish,
        /// Whether the job is over: finished, failed, or its client left.
        done: bool,
        /// Whether the client got an error instead of the end of its tokens.
        failed: bool,
        /// Whether the draft follows this request through batched steps.
        followed: bool,
        /// Whether returning to speculation failed, so that the request stays batched.
        stay_batched: bool,
        /// Tokens emitted by speculation and by batched steps.
        speculated: usize,
        batched: usize,
    }

    impl Request {
        fn new(job: Job, slot: usize) -> Self {
            Self {
                job,
                slot,
                started: Instant::now(),
                prefilled: 0,
                cached: 0,
                prefill_s: 0.0,
                next: None,
                output: Vec::new(),
                finish: Finish::Length,
                done: false,
                failed: false,
                followed: false,
                stay_batched: false,
                speculated: 0,
                batched: 0,
            }
        }

        /// Sends the tokens of `ids` that fit, up to a stop token; breaks when the job is
        /// over or its client has left.
        fn take(&mut self, ids: &[u32], stop: &[u32]) -> ControlFlow<()> {
            let room = self.job.max_tokens - self.output.len();
            let end = ids.iter().position(|id| stop.contains(id));
            let keep = end.unwrap_or(ids.len()).min(room);
            if keep > 0 {
                self.output.extend_from_slice(&ids[..keep]);
                if self
                    .job
                    .events
                    .send(Event::Tokens(ids[..keep].to_vec()))
                    .is_err()
                {
                    return ControlFlow::Break(());
                }
            }
            if end.is_some_and(|e| e <= room) {
                self.finish = Finish::Stop;
                return ControlFlow::Break(());
            }
            if self.output.len() == self.job.max_tokens {
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        }

        /// Logs the job and tells its client it is done.
        fn finish(&self, detail: &str) {
            let decode_s = self.started.elapsed().as_secs_f64() - self.prefill_s;
            let events = &self.job.events;
            eprintln!(
                "job: {} prompt tokens ({} cached), prefill {:.0} ms, {} tokens in {:.2} s \
                 ({:.1} tok/s, {} speculated, {} batched steps{detail}), {}",
                self.job.prompt.len(),
                self.cached,
                self.prefill_s * 1e3,
                self.output.len(),
                decode_s,
                self.output.len() as f64 / decode_s.max(1e-9),
                self.speculated,
                self.batched,
                if events.is_closed() {
                    "client left".to_owned()
                } else {
                    format!("{:?}", self.finish)
                },
            );
            let _ = events.send(Event::Done(self.finish));
        }

        fn fail(&mut self, e: &anyhow::Error) {
            eprintln!("job failed: {e:#}");
            let _ = self.job.events.send(Event::Failed(format!("{e:#}")));
            self.done = true;
            self.failed = true;
        }
    }

    fn run(
        mut model: Model,
        mut speculator: Speculator,
        mut check: MemoryCheck,
        queue: &Receiver<Job>,
        stop: &[u32],
        streams: usize,
    ) {
        let mut waiting = VecDeque::new();
        // A request that a batch left alone, to continue with speculation.
        let mut resumed = None;
        loop {
            let mut request = if let Some(request) = resumed.take() {
                request
            } else {
                let job = match waiting.pop_front() {
                    Some(job) => job,
                    None => match queue.recv() {
                        Ok(job) => job,
                        Err(_) => return,
                    },
                };
                if job.events.is_closed() {
                    continue;
                }
                // Another job already waits: start them together.
                if streams > 1 {
                    if let Ok(next) = queue.try_recv() {
                        waiting.push_back(next);
                    }
                    if !waiting.is_empty() {
                        waiting.push_front(job);
                        resumed = batched(
                            &mut model,
                            &mut speculator,
                            Vec::new(),
                            &mut waiting,
                            queue,
                            stop,
                            streams,
                        );
                        check.after_job();
                        continue;
                    }
                }
                Request::new(job, 0)
            };
            let interrupt = (streams > 1).then_some((queue, &mut waiting));
            match alone(&mut model, &mut speculator, &mut request, stop, interrupt) {
                Ok(true) => {
                    speculator.follow(&model);
                    request.followed = true;
                    resumed = batched(
                        &mut model,
                        &mut speculator,
                        vec![request],
                        &mut waiting,
                        queue,
                        stop,
                        streams,
                    );
                }
                Ok(false) => {}
                Err(e) => request.fail(&e),
            }
            check.after_job();
        }
    }

    /// Runs `request` alone in slot 0 with speculation: from its prompt, or from its next
    /// token when it returns from a batch (`Speculator::resume`). With `interrupt`, stops
    /// after the round in which another job arrives, moves that job to the waiting list
    /// and returns true: the request then continues in a batch from slot 0. Otherwise
    /// finishes it.
    fn alone(
        model: &mut Model,
        speculator: &mut Speculator,
        request: &mut Request,
        stop: &[u32],
        mut interrupt: Option<(&Receiver<Job>, &mut VecDeque<Job>)>,
    ) -> anyhow::Result<bool> {
        let go_on = if request.next.is_some() {
            true
        } else {
            let (first, cached) =
                speculator.prefill_cached(model, &request.job.prompt, DEFAULT_PREFILL_CHUNK)?;
            request.cached = cached;
            let _ = request.job.events.send(Event::Cached(cached));
            request.prefilled = request.job.prompt.len();
            request.prefill_s = request.started.elapsed().as_secs_f64();
            request.next = Some(first);
            request.take(&[first], stop).is_continue()
        };
        let mut interrupted = false;
        let mut last = request.next.unwrap_or_default();
        let before = request.output.len();
        let started = Instant::now();
        if go_on {
            let remaining = request.job.max_tokens - request.output.len();
            speculator.generate_with(model, remaining, Drafts::Dspark, |ids| {
                request.speculated += ids.len();
                last = ids.last().copied().unwrap_or(last);
                if request.take(ids, stop).is_break() {
                    return ControlFlow::Break(());
                }
                if let Some((queue, waiting)) = interrupt.as_mut()
                    && let Ok(job) = queue.try_recv()
                {
                    waiting.push_back(job);
                    interrupted = true;
                    return ControlFlow::Break(());
                }
                ControlFlow::Continue(())
            })?;
        }
        let stats = speculator.stats();
        let speculated = request.output.len() - before;
        let seconds = started.elapsed().as_secs_f64();
        let acceptance: Vec<String> = stats
            .acceptance()
            .iter()
            .map(|a| format!("{:.1}%", a * 100.0))
            .collect();
        let episode = format!(
            "speculated {speculated} tokens in {seconds:.2} s ({:.1} tok/s, {:.2} tokens per \
             round, {} plain steps, {} probes, draft {:.3} ms per round, acceptance by \
             position [{}])",
            speculated as f64 / seconds.max(1e-9),
            stats.tokens_per_round(),
            stats.stepped.len(),
            stats.probes,
            stats.draft.as_secs_f64() * 1e3 / (stats.rounds + stats.probes).max(1) as f64,
            acceptance.join(", "),
        );
        if interrupted {
            eprintln!("{episode}, then batched from position {}", model.position());
            request.next = Some(last);
            return Ok(true);
        }
        request.finish(&format!(", last {episode}"));
        Ok(false)
    }

    /// Prompt and generated tokens per remaining token above which a lone request that
    /// ran without the draft gets the draft's state rebuilt, by prefilling its tokens
    /// again, to speculate. A one-row batched step takes about 3.7 ms and speculation about
    /// 2 ms per token, so each remaining token saves about 1.7 ms, while the rebuild
    /// prefills about 20,000 tokens per second: it pays off once the remaining tokens
    /// exceed about 1/34 of the position. Requiring 1/16 leaves room for requests that
    /// stop well before their limit.
    const REBUILD_RATIO: usize = 16;

    /// Moves a lone request from a batch back to speculation in slot 0: commits the taps
    /// the draft kept while following it, or rebuilds the draft's state from its tokens in
    /// a free slot, then moves it to slot 0 if it is elsewhere. Returns false when it
    /// stays batched: it never ran with the draft and too few tokens remain to rebuild.
    fn speculate_again(
        model: &mut Model,
        speculator: &mut Speculator,
        r: &mut Request,
        streams: usize,
    ) -> anyhow::Result<bool> {
        let started = Instant::now();
        let position = model.slot_position(r.slot)?;
        let next = r.next.context("the request has no next token")?;
        let how = if r.followed {
            "the draft kept its taps".to_owned()
        } else {
            let remaining = r.job.max_tokens - r.output.len();
            if remaining * REBUILD_RATIO < position {
                return Ok(false);
            }
            let fed = &r.output[..r.output.len() - 1];
            let tokens: Vec<u32> = r.job.prompt.iter().chain(fed).copied().collect();
            anyhow::ensure!(
                tokens.len() == position,
                "{} tokens fed, the slot is at {position}",
                tokens.len()
            );
            let free = (0..streams)
                .find(|&s| s != r.slot)
                .context("no free slot to rebuild the draft in")?;
            speculator.rebuild(model, free, &tokens)?;
            r.followed = true;
            format!("rebuilt the draft in slot {free}")
        };
        let from = r.slot;
        model.move_slot(from, 0)?;
        r.slot = 0;
        speculator.resume(model, next)?;
        model.stream().synchronize()?;
        eprintln!(
            "speculating again at position {position} from slot {from}: {how}, {:.1} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
        r.followed = false;
        Ok(true)
    }

    /// Runs `active` and the jobs that arrive while any runs, up to `streams` at once, each
    /// in its own slot, until none is left or one is left alone. Each turn prefills one
    /// chunk of the oldest request still prefilling, then runs one batched step of the
    /// others. A request left alone with nothing waiting returns, in slot 0, to continue
    /// with speculation (`speculate_again`).
    fn batched(
        model: &mut Model,
        speculator: &mut Speculator,
        mut active: Vec<Request>,
        waiting: &mut VecDeque<Job>,
        queue: &Receiver<Job>,
        stop: &[u32],
        streams: usize,
    ) -> Option<Request> {
        loop {
            while active.len() < streams {
                let Some(job) = waiting.pop_front().or_else(|| queue.try_recv().ok()) else {
                    break;
                };
                if job.events.is_closed() {
                    continue;
                }
                // Slot 0 last, since a new sequence there empties the prefix cache.
                let Some(slot) = (1..streams)
                    .chain([0])
                    .find(|s| active.iter().all(|r| r.slot != *s))
                else {
                    break;
                };
                let mut request = Request::new(job, slot);
                match model.reset_slot(slot) {
                    Ok(()) => active.push(request),
                    Err(e) => request.fail(&e),
                }
            }
            if active.is_empty() {
                return None;
            }
            if let [r] = active.as_mut_slice()
                && r.next.is_some()
                && !r.stay_batched
                && waiting.is_empty()
            {
                match speculate_again(model, speculator, r, streams) {
                    Ok(true) => return active.pop(),
                    Ok(false) => {}
                    Err(e) => {
                        eprintln!("returning to speculation failed, staying batched: {e:#}");
                        speculator.forget();
                        r.followed = false;
                        r.stay_batched = true;
                    }
                }
            }
            if let Some(r) = active.iter_mut().find(|r| r.next.is_none()) {
                if r.job.events.is_closed() {
                    r.done = true;
                } else {
                    let end = (r.prefilled + DEFAULT_PREFILL_CHUNK).min(r.job.prompt.len());
                    match model.prefill_slot(
                        r.slot,
                        &r.job.prompt[r.prefilled..end],
                        DEFAULT_PREFILL_CHUNK,
                    ) {
                        Ok(first) => {
                            r.prefilled = end;
                            if end == r.job.prompt.len() {
                                r.prefill_s = r.started.elapsed().as_secs_f64();
                                r.next = Some(first);
                                r.done = r.take(&[first], stop).is_break();
                            }
                        }
                        Err(e) => r.fail(&e),
                    }
                }
            }
            step_rows(model, speculator, &mut active, stop);
            active.retain(|r| {
                if r.done {
                    if r.followed {
                        speculator.forget();
                    }
                    if !r.failed && r.next.is_some() {
                        r.finish("");
                    }
                }
                !r.done
            });
        }
    }

    /// Runs one batched step of the requests of `active` that have a next token, and keeps
    /// the taps of the one the draft follows.
    fn step_rows(
        model: &mut Model,
        speculator: &mut Speculator,
        active: &mut [Request],
        stop: &[u32],
    ) {
        let rows: Vec<(usize, u32)> = active
            .iter()
            .filter(|r| !r.done)
            .filter_map(|r| r.next.map(|t| (r.slot, t)))
            .collect();
        if rows.is_empty() {
            return;
        }
        match model.step_batch(&rows) {
            Ok(next) => {
                let mut next = next.into_iter();
                let stepped = active.iter_mut().filter(|r| !r.done && r.next.is_some());
                for (row, r) in stepped.enumerate() {
                    let token = next.next().unwrap_or_default();
                    if r.followed
                        && let Err(e) = speculator.keep_batched(model, row, token)
                    {
                        eprintln!("the draft stops following a request: {e:#}");
                        speculator.forget();
                        r.followed = false;
                    }
                    r.batched += 1;
                    r.next = Some(token);
                    r.done = r.take(&[token], stop).is_break();
                }
            }
            Err(e) => {
                for r in active.iter_mut().filter(|r| !r.done && r.next.is_some()) {
                    r.fail(&e);
                }
            }
        }
    }
}
