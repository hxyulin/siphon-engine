//! The GPU side: one thread owns the model and the draft and runs requests in arrival
//! order from a bounded queue.
//!
//! A request that arrives at an idle server runs alone with speculation in slot 0.
//! With `streams` above 1, a request that arrives meanwhile stops the speculation after its
//! round, and from then on every request runs batched plain decode in its own slot, up to
//! `streams` at once, until the server is idle again. A new request's prompt prefills in
//! chunks between the batch's steps, with enough steps after each chunk to keep every
//! decoding request at `Options::decode_floor` tokens per second. Batched steps and speculation give each sequence the
//! same tokens, so a request's output does not depend on what else runs (see
//! `docs/batching.md`).
//!
//! The jobs of one submission share a prompt: an HTTP request with `n` choices submits `n`
//! jobs that differ only in their sampling seeds. When one of them finishes its prefill in
//! a batch, the engine copies that slot's state into the slots of the siblings that have
//! not started prefilling (`Model::move_slot`) and draws each sibling's first token from
//! the same logits with its own sampling (`Model::redraw_slot`). Each copy then decodes as
//! if it had prefilled the prompt itself.
//!
//! Every request prefills through the speculator's prefix cache, which keeps one entry per
//! slot: a prompt's state at its last multiple of the prefill chunk. A request whose prompt
//! starts with an entry's tokens takes that entry's slot and prefills only the rest. A
//! batched request otherwise takes a slot without an entry, else the least recently used
//! one (`Speculator::claim_slot`). A request that runs alone speculates in slot 0, and an
//! entry it hits moves there (`Speculator::prefill_cached`), when the entry holds the
//! draft's state too. An entry saved by a batched prefill does not, so a request that hits
//! one alone runs as a batch of one from the entry's slot, and returns to speculation as a
//! request left alone by a batch does: by rebuilding the draft's state when enough tokens
//! remain (`speculate_again`).
//!
//! Each request picks its tokens greedily or by sampling with its own [`Sampling`]. A
//! sampled token depends only on the logits, the request's seed and the token's position
//! (see `siphon_cuda::ops::sample`), and speculation keeps a draft only when it equals the
//! token the target samples at that position. So sampling keeps the batching property: a
//! request with a seed gets the same tokens alone or in a batch.

use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, SyncSender, TrySendError},
    },
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
    /// vLLM's `ignore_eos`: generate `max_tokens` tokens, passing the model's end tokens
    /// on as output.
    pub ignore_eos: bool,
    pub sampling: Sampling,
    pub events: UnboundedSender<Event>,
    /// The submission the job came in: jobs of one submission share their prompt.
    group: u64,
    /// The job's place in the engine's [`Load`], from [`Engine::submit`].
    ticket: Option<Ticket>,
    /// The engine's drain state, from [`Engine::submit`].
    control: Option<Arc<Control>>,
}

impl Job {
    pub fn new(
        prompt: Vec<u32>,
        max_tokens: usize,
        ignore_eos: bool,
        sampling: Sampling,
        events: UnboundedSender<Event>,
    ) -> Self {
        Self {
            prompt,
            max_tokens,
            ignore_eos,
            sampling,
            events,
            group: 0,
            ticket: None,
            control: None,
        }
    }

    /// Whether a drain gave up on the job ([`Engine::cancel`]).
    #[cfg_attr(
        not(target_os = "linux"),
        expect(dead_code, reason = "only the Linux engine runs jobs")
    )]
    fn cancelled(&self) -> bool {
        self.control
            .as_ref()
            .is_some_and(|c| c.cancel.load(Ordering::Relaxed))
    }

    /// Counts the job as running rather than waiting.
    #[cfg_attr(
        not(target_os = "linux"),
        expect(dead_code, reason = "only the Linux engine runs jobs")
    )]
    fn start(&mut self) {
        if let Some(ticket) = &mut self.ticket {
            ticket.start();
        }
    }
}

/// Jobs that wait for the engine and jobs it runs, counted across threads so that a
/// submission of several jobs fits whole or not at all.
#[derive(Debug, Default)]
struct Load(Mutex<Counts>);

#[derive(Debug, Default)]
struct Counts {
    waiting: usize,
    running: usize,
}

impl Load {
    fn counts(&self) -> MutexGuard<'_, Counts> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Counts `n` more waiting jobs if they fit: at most `queue` may wait beyond the
    /// `streams` that run.
    fn reserve(&self, n: usize, queue: usize, streams: usize) -> bool {
        let mut counts = self.counts();
        let idle = streams.saturating_sub(counts.running);
        let fits = counts.waiting + n <= queue + idle;
        if fits {
            counts.waiting += n;
        }
        fits
    }
}

/// What a drain tells the engine thread, set from the HTTP side.
#[derive(Debug, Default)]
struct Control {
    /// New jobs are refused ([`Engine::drain`]).
    draining: AtomicBool,
    /// The jobs left end now, with an error ([`Engine::cancel`]).
    cancel: AtomicBool,
    /// Where the engine thread reports once it has stopped ([`Engine::stop`]); it stops
    /// when this is set and no job is left.
    stopped: Mutex<Option<mpsc::Sender<String>>>,
}

/// One job's count in a [`Load`]: waiting until [`Ticket::start`], then running until
/// dropped.
#[derive(Debug)]
struct Ticket {
    load: Arc<Load>,
    running: bool,
}

impl Ticket {
    fn new(load: &Arc<Load>) -> Self {
        Self {
            load: Arc::clone(load),
            running: false,
        }
    }

    fn start(&mut self) {
        if !self.running {
            self.running = true;
            let mut counts = self.load.counts();
            counts.waiting -= 1;
            counts.running += 1;
        }
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        let mut counts = self.load.counts();
        if self.running {
            counts.running -= 1;
        } else {
            counts.waiting -= 1;
        }
    }
}

/// How a request picks its tokens, as vLLM's sampling parameters: temperature 0 is greedy,
/// and the filters apply in the order temperature, top-k, top-p, min-p.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    /// 0 keeps every token.
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
}

/// Why generation ended.
#[cfg_attr(
    all(not(target_os = "linux"), not(test)),
    expect(dead_code, reason = "only the Linux engine runs jobs")
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model ended its turn.
    Stop,
    /// The request's token limit.
    Length,
}

/// What the engine reports for a job, in order: `Started`, `Cached` at most once, `Tokens`
/// any number of times, then `Done` or `Failed`. A job whose client left while it waited
/// gets none.
#[cfg_attr(
    all(not(target_os = "linux"), not(test)),
    expect(dead_code, reason = "only the Linux engine runs jobs")
)]
#[derive(Debug)]
pub enum Event {
    /// The engine took the job from its queue.
    Started,
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
    /// Free bytes the plan leaves unclaimed; `None` leaves `memory::DEFAULT_MARGIN`.
    pub margin: Option<usize>,
    /// Rows of each prefill chunk; `None` runs `DEFAULT_PREFILL_CHUNK`.
    pub prefill_chunk: Option<usize>,
    pub fp8_weights: bool,
    pub fp8_kv: bool,
    /// Verify widths the adaptive policy chooses among.
    pub widths: Vec<usize>,
    /// Jobs that wait while `streams` run.
    pub queue: usize,
    /// Requests that run at once, each in its own sequence slot, 1 to 32.
    pub streams: usize,
    /// A DFlash draft's export directory, in place of the bundled draft.
    pub drafter: Option<PathBuf>,
    /// The draft's approximations, as `DraftOptions::parse` takes them.
    pub draft: String,
    /// Whether the speculator falls back to plain steps (`WidthPolicy::with_plain`).
    pub fallback: bool,
    /// Entries of the prefix cache, 1 to `streams`, or 0 for none.
    pub prefix_entries: usize,
    /// How long an unused prefix cache entry lives.
    pub prefix_ttl: std::time::Duration,
    /// Bytes of page-locked host memory for the prefix cache's host tier, 0 for none.
    pub host_cache: usize,
    /// Where the host tier's entries are saved after a drain and loaded at startup.
    pub cache_dir: Option<PathBuf>,
    /// Prefill's global attention products over an FP8 cache, as `PrefillAttention`
    /// parses them.
    pub prefill_attention: String,
    /// Whether every request decodes greedily, whatever it asks.
    pub greedy: bool,
    /// Tokens per second that each decoding request gets at least while another prefills,
    /// with `streams` above 1, or 0 for one batched step per prefill chunk.
    pub decode_floor: f64,
}

/// Sends jobs to the engine thread.
#[derive(Clone)]
pub struct Engine {
    /// Submissions: the jobs of one HTTP request, which share a prompt.
    jobs: SyncSender<Vec<Job>>,
    max_context: usize,
    load: Arc<Load>,
    queue: usize,
    streams: usize,
    control: Arc<Control>,
}

/// Why the engine did not take a job.
pub enum Refused {
    Busy,
    /// The server drains before it stops ([`Engine::drain`]).
    Draining,
    Stopped,
}

impl Engine {
    /// Prompt and generated tokens that one job may hold together.
    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Jobs that one submission may hold: as many as run and wait at once.
    pub fn max_jobs(&self) -> usize {
        self.queue + self.streams
    }

    /// Queues `jobs`, which share their prompt, as one submission, or refuses them all
    /// when they do not fit the queue beside the jobs already there or the engine has
    /// stopped.
    pub fn submit(&self, mut jobs: Vec<Job>) -> Result<(), Refused> {
        static GROUPS: AtomicU64 = AtomicU64::new(0);
        if self.draining() {
            return Err(Refused::Draining);
        }
        if !self.load.reserve(jobs.len(), self.queue, self.streams) {
            return Err(Refused::Busy);
        }
        let group = GROUPS.fetch_add(1, Ordering::Relaxed);
        for job in &mut jobs {
            job.group = group;
            // Each ticket gives back its reservation when its job is dropped, so a refused
            // send below gives them all back.
            job.ticket = Some(Ticket::new(&self.load));
            job.control = Some(Arc::clone(&self.control));
        }
        self.jobs.try_send(jobs).map_err(|e| match e {
            TrySendError::Full(_) => Refused::Busy,
            TrySendError::Disconnected(_) => Refused::Stopped,
        })
    }

    /// Refuses new jobs from now on; the running and waiting ones go on.
    pub fn drain(&self) {
        self.control.draining.store(true, Ordering::Relaxed);
    }

    pub fn draining(&self) -> bool {
        self.control.draining.load(Ordering::Relaxed)
    }

    /// Jobs running and jobs waiting.
    pub fn counts(&self) -> (usize, usize) {
        let counts = self.load.counts();
        (counts.running, counts.waiting)
    }

    /// Ends every job left, running or waiting, with an error at its next step.
    pub fn cancel(&self) {
        self.control.cancel.store(true, Ordering::Relaxed);
    }

    /// Stops the engine thread once no job is left, after it saves the prefix cache's host
    /// tier to `--cache-dir`, and returns its report. Blocks until then.
    pub fn stop(&self) -> String {
        let (sender, receiver) = mpsc::channel();
        *self
            .control
            .stopped
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(sender);
        // Wakes the thread if it waits for jobs.
        let _ = self.jobs.try_send(Vec::new());
        receiver
            .recv()
            .unwrap_or_else(|_| "the engine thread had already stopped".to_owned())
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
        memory::{self, CONTEXT_BYTES, DEFAULT_MARGIN, DraftChoice, PlanInputs, Planner, gib},
    };
    use siphon_cuda::{
        Gpu, MemoryUse,
        draft::DraftOptions,
        model::{DEFAULT_PREFILL_CHUNK, Model, Sampling as ModelSampling},
        ops::{attention::KvFormat, spec::VERIFY_ROWS},
        spec::{
            DiskLoaded, DraftSource, Drafts, HOST_CHUNK_POSITIONS, Speculator, Width, WidthPolicy,
            load_draft, price,
        },
        weights::{WeightFormat, Weights},
    };

    use super::{Control, Engine, Event, Finish, Job, Options};

    /// Why a job that a drain gave up on failed.
    const CANCELLED: &str = "the server stopped before the request finished";

    /// Positions past the last generated token that a round may write: a full verify.
    const MARGIN: usize = VERIFY_ROWS + 1;

    /// Loads the model on a new thread, prices the verify widths on `warmup` (a prompt
    /// of at least 32 tokens), and returns once the thread takes jobs. `stop` holds the
    /// ids that end a turn.
    pub fn start(options: &Options, stop: Vec<u32>, warmup: Vec<u32>) -> anyhow::Result<Engine> {
        anyhow::ensure!(warmup.len() >= 32, "the warmup prompt is too short");
        // Each submission holds at least one job, so the channel never holds more
        // submissions than `Load::reserve` lets wait.
        let (jobs, queue) = sync_channel(options.queue + options.streams);
        let (ready, loaded) = sync_channel(1);
        let (queue_len, streams) = (options.queue, options.streams);
        let options = options.clone();
        let control = std::sync::Arc::new(Control::default());
        let engine_control = std::sync::Arc::clone(&control);
        thread::Builder::new()
            .name("engine".to_owned())
            .spawn(move || match load(&options, &warmup) {
                Ok((model, speculator, capacity, check)) => {
                    let _ = ready.send(Ok(capacity));
                    run(
                        model,
                        speculator,
                        check,
                        &queue,
                        &stop,
                        &options,
                        &engine_control,
                    );
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
            load: std::sync::Arc::default(),
            queue: queue_len,
            streams,
            control,
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
        let margin = options.margin.unwrap_or(DEFAULT_MARGIN);
        let budget = memory::budget(free, options.memory, margin);
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
            gib(margin)
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
        /// The peak after a job last printed, 0 before the first job.
        reported: usize,
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
                reported: 0,
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
                Ok(now) => {
                    let peak = self.peak(&now);
                    if peak > self.reported {
                        self.reported = peak;
                        eprintln!(
                            "memory check: peak {} MiB of a planned {} MiB after a job",
                            peak >> 20,
                            self.planned >> 20
                        );
                    }
                    self.warn(&now);
                }
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
        let (weight_format, kv) = formats(options);
        let draft_options = DraftOptions::parse(&options.draft, None)?;
        let chunk = options.prefill_chunk.unwrap_or(DEFAULT_PREFILL_CHUNK);
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
            prefill_chunk: chunk,
            streams: options.streams,
            prefix_entries: options.prefix_entries,
            sms: gpu.sms()?,
        })?;
        let capacity = plan(options, &planner, free, warmup.len())?;
        let weights = Weights::load(&checkpoint, &config, gpu.stream())?;
        let mut model =
            Model::with_formats(&gpu, config, weights, passes, capacity, kv, weight_format)?;
        model.set_prefill_attention(options.prefill_attention.parse()?);
        if !options.greedy {
            model.enable_sampling()?;
        }
        let width = Width::Adaptive(WidthPolicy::new(&options.widths, [1.0; VERIFY_ROWS])?);
        let source = options
            .drafter
            .as_deref()
            .map_or(DraftSource::Bundled, DraftSource::Dflash);
        let draft = load_draft(&mut model, &checkpoint, source, capacity, &draft_options)?;
        let mut speculator = Speculator::with_draft(&model, draft, width)?;
        speculator.set_prefill_chunk(chunk)?;
        eprintln!("loaded in {:.1} s", started.elapsed().as_secs_f64());

        // Price each width at the warmup prompt, which also captures the graphs.
        model.reset()?;
        let first = speculator.prefill(&mut model, warmup, chunk)?;
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
        if options.prefix_entries > 0 {
            let bytes = speculator.enable_prefix_cache(
                &model,
                options.prefix_entries,
                options.prefix_ttl,
            )?;
            eprintln!(
                "prefix cache: {} entries, {} MiB of copies, {} s to live",
                options.prefix_entries,
                bytes >> 20,
                options.prefix_ttl.as_secs()
            );
        }
        enable_host_cache(options, &model, &mut speculator)?;
        // The full chunk's buffers now, so that the check below sees every planned buffer
        // and a plan that undercounts fails here rather than on the first long prompt.
        // After pricing, so that they reuse the memory pricing freed, as a long prompt's
        // prefill would.
        model.reserve_prefill(chunk)?;
        let check = MemoryCheck::new(gpu, planner.plan(capacity).total(), free)?;
        Ok((model, speculator, capacity, check))
    }

    /// The weight and KV cache formats `options` ask for.
    fn formats(options: &Options) -> (WeightFormat, KvFormat) {
        let weights = if options.fp8_weights {
            WeightFormat::Fp8
        } else {
            WeightFormat::Bf16
        };
        let kv = if options.fp8_kv {
            KvFormat::Fp8
        } else {
            KvFormat::Bf16
        };
        (weights, kv)
    }

    /// Adds the prefix cache's host tier of `--host-cache` bytes, if any, and logs how long
    /// page-locking its memory took.
    fn enable_host_cache(
        options: &Options,
        model: &Model,
        speculator: &mut Speculator,
    ) -> anyhow::Result<()> {
        if options.host_cache == 0 {
            anyhow::ensure!(
                options.cache_dir.is_none(),
                "--cache-dir needs --host-cache"
            );
            return Ok(());
        }
        anyhow::ensure!(
            options.prefix_entries > 0,
            "--host-cache needs the prefix cache"
        );
        anyhow::ensure!(
            speculator.prefill_chunk() == HOST_CHUNK_POSITIONS,
            "--host-cache needs --prefill-chunk {HOST_CHUNK_POSITIONS}, the host cache's chunk"
        );
        let started = Instant::now();
        let (chunks, chunk_bytes) = speculator.enable_host_cache(model, options.host_cache)?;
        eprintln!(
            "host cache: {:.2} GiB of page-locked host memory in {:.2} s, {chunks} chunks of \
             {} MiB ({HOST_CHUNK_POSITIONS} positions each)",
            (chunks * chunk_bytes) as f64 / f64::from(1u32 << 30),
            started.elapsed().as_secs_f64(),
            chunk_bytes >> 20
        );
        let Some(dir) = &options.cache_dir else {
            return Ok(());
        };
        let line = match speculator.load_host_cache(dir, &cache_identity(options)?)? {
            DiskLoaded::Missing => format!("no saved host cache in {}", dir.display()),
            DiskLoaded::Discarded(why) => {
                format!("discarded the saved host cache in {}: {why}", dir.display())
            }
            DiskLoaded::Loaded {
                entries,
                skipped,
                bytes,
                seconds,
            } => format!(
                "loaded {entries} entries ({:.2} GiB) from {} in {seconds:.1} s ({:.2} GB/s), \
                 skipped {skipped}",
                bytes as f64 / f64::from(1u32 << 30),
                dir.display(),
                bytes as f64 / seconds.max(1e-9) / 1e9
            ),
        };
        eprintln!("host cache: {line}");
        log_host(speculator);
        Ok(())
    }

    /// What changes the bytes of the host tier's saved entries or their meaning, for the
    /// manifest in `--cache-dir`: the checkpoint and the drafter (their configs and the
    /// names and sizes of their weight files), the formats, the draft's approximations,
    /// prefill's attention products and the kernels.
    fn cache_identity(options: &Options) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::json!({
            "checkpoint": weights_hash(&options.model_dir)?,
            "drafter": options.drafter.as_deref().map(weights_hash).transpose()?,
            "kv": if options.fp8_kv { "fp8" } else { "bf16" },
            "weights": if options.fp8_weights { "fp8" } else { "bf16" },
            "draft": options.draft,
            "prefill_attention": options.prefill_attention,
            "kernels": siphon_cuda::KERNELS_HASH,
        }))
    }

    /// FNV-1a over `dir`'s config.json and the names and sizes of its safetensors files.
    fn weights_hash(dir: &std::path::Path) -> anyhow::Result<String> {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut add = |bytes: &[u8]| {
            for &b in bytes {
                hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
        };
        let config = dir.join("config.json");
        add(&std::fs::read(&config).with_context(|| format!("reading {}", config.display()))?);
        let mut files: Vec<(String, u64)> = std::fs::read_dir(dir)?
            .filter_map(Result::ok)
            .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            .map(|e| {
                Ok((
                    e.file_name().to_string_lossy().into_owned(),
                    e.metadata()?.len(),
                ))
            })
            .collect::<std::io::Result<_>>()?;
        files.sort();
        for (name, size) in files {
            add(name.as_bytes());
            add(&size.to_le_bytes());
        }
        Ok(format!("{hash:016x}"))
    }

    /// Saves the host tier to `--cache-dir`, if both are on, before the engine thread
    /// stops, and returns a line for the log.
    fn shut_down(model: &Model, speculator: &mut Speculator, options: &Options) -> String {
        let Some(dir) = options.cache_dir.as_deref() else {
            return "engine stopped".to_owned();
        };
        let saved = cache_identity(options)
            .and_then(|identity| speculator.save_host_cache(model, dir, &identity));
        log_host(speculator);
        match saved {
            Ok(s) => format!(
                "host cache: saved {} entries ({:.2} GiB) to {} in {:.1} s ({:.2} GB/s)",
                s.entries,
                s.bytes as f64 / f64::from(1u32 << 30),
                dir.display(),
                s.seconds,
                s.bytes as f64 / s.seconds.max(1e-9) / 1e9
            ),
            Err(e) => format!("host cache: saving to {} failed: {e:#}", dir.display()),
        }
    }

    /// The model's form of a request's sampling, greedy under `--greedy`.
    fn model_sampling(job: &Job, greedy: bool) -> anyhow::Result<ModelSampling> {
        let s = job.sampling;
        if greedy {
            return Ok(ModelSampling::GREEDY);
        }
        ModelSampling::new(s.temperature, s.top_k, s.top_p, s.min_p, s.seed)
    }

    /// A request the engine has taken.
    #[expect(
        clippy::struct_excessive_bools,
        reason = "independent states of a request"
    )]
    struct Request {
        job: Job,
        /// How the request picks its tokens.
        sampling: ModelSampling,
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
        /// Whether the prompt's state came from a sibling's prefill (`share_prefill`).
        shared: bool,
        /// Whether the draft follows this request through batched steps.
        followed: bool,
        /// Whether returning to speculation failed, so that the request stays batched.
        stay_batched: bool,
        /// Tokens emitted by speculation and by batched steps.
        speculated: usize,
        batched: usize,
    }

    impl Request {
        fn new(mut job: Job, slot: usize, greedy: bool) -> Self {
            job.start();
            let _ = job.events.send(Event::Started);
            let (sampling, error) = match model_sampling(&job, greedy) {
                Ok(s) => (s, None),
                Err(e) => (ModelSampling::GREEDY, Some(e)),
            };
            let mut request = Self {
                job,
                sampling,
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
                shared: false,
                followed: false,
                stay_batched: false,
                speculated: 0,
                batched: 0,
            };
            if let Some(e) = error {
                request.fail(&e);
            } else if request.job.cancelled() {
                request.fail(&anyhow::anyhow!(CANCELLED));
            }
            request
        }

        /// Sends the tokens of `ids` that fit, up to a stop token; breaks when the job is
        /// over or its client has left.
        fn take(&mut self, ids: &[u32], stop: &[u32]) -> ControlFlow<()> {
            if self.job.cancelled() {
                self.fail(&anyhow::anyhow!(CANCELLED));
                return ControlFlow::Break(());
            }
            let room = self.job.max_tokens - self.output.len();
            let end = if self.job.ignore_eos {
                None
            } else {
                ids.iter().position(|id| stop.contains(id))
            };
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
                "job: {} prompt tokens ({} cached), prefill {:.0} ms{}, {} tokens in {:.2} s \
                 ({:.1} tok/s, {} speculated, {} batched steps{detail}), {}",
                self.job.prompt.len(),
                self.cached,
                self.prefill_s * 1e3,
                if self.shared { " (copied)" } else { "" },
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
        queue: &Receiver<Vec<Job>>,
        stop: &[u32],
        options: &Options,
        control: &Control,
    ) {
        let (streams, greedy) = (options.streams, options.greedy);
        let mut waiting = VecDeque::new();
        // A request that a batch left alone, to continue with speculation.
        let mut resumed = None;
        loop {
            let mut request = if let Some(request) = resumed.take() {
                request
            } else {
                if waiting.is_empty() {
                    let stopping = control
                        .stopped
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .take();
                    if let Some(report) = stopping {
                        let _ = report.send(shut_down(&model, &mut speculator, options));
                        return;
                    }
                    match queue.recv() {
                        Ok(jobs) => waiting.extend(jobs),
                        Err(_) => return,
                    }
                }
                let Some(job) = waiting.pop_front() else {
                    continue;
                };
                if job.events.is_closed() {
                    continue;
                }
                // Another job already waits: start them together.
                if streams > 1 {
                    if let Ok(jobs) = queue.try_recv() {
                        waiting.extend(jobs);
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
                            greedy,
                            options.decode_floor,
                        );
                        check.after_job();
                        continue;
                    }
                }
                // A prompt whose entry lacks the draft's state prefills from that entry as
                // a batch of one rather than whole in slot 0.
                if streams > 1
                    && speculator
                        .prefix_lookup(&model, &job.prompt, speculator.prefill_chunk(), &[])
                        .is_some_and(|hit| !hit.draft)
                {
                    waiting.push_front(job);
                    resumed = batched(
                        &mut model,
                        &mut speculator,
                        Vec::new(),
                        &mut waiting,
                        queue,
                        stop,
                        streams,
                        greedy,
                        options.decode_floor,
                    );
                    check.after_job();
                    continue;
                }
                let request = Request::new(job, 0, greedy);
                if request.done {
                    continue;
                }
                request
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
                        greedy,
                        options.decode_floor,
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
        mut interrupt: Option<(&Receiver<Vec<Job>>, &mut VecDeque<Job>)>,
    ) -> anyhow::Result<bool> {
        model.set_sampling(request.sampling);
        let go_on = if request.next.is_some() {
            true
        } else {
            let (first, cached) = speculator.prefill_cached(
                model,
                &request.job.prompt,
                speculator.prefill_chunk(),
            )?;
            log_host(speculator);
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
                    && let Ok(jobs) = queue.try_recv()
                {
                    waiting.extend(jobs);
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

    /// Prints what the prefix cache's host tier did since the last call.
    fn log_host(speculator: &mut Speculator) {
        for line in speculator.host_notes() {
            eprintln!("{line}");
        }
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
    /// the draft kept while following it, or moves it to slot 0 and rebuilds the draft's
    /// state from its tokens in an idle slot, which also gives the request's prefix cache
    /// entry the draft's state. Returns false when it stays batched: it never ran with the
    /// draft and too few tokens remain to rebuild.
    fn speculate_again(
        model: &mut Model,
        speculator: &mut Speculator,
        r: &mut Request,
    ) -> anyhow::Result<bool> {
        let started = Instant::now();
        let from = r.slot;
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
            // Into slot 0 first, so that the rebuild may take the slot it leaves.
            speculator.move_to_slot0(model, r.slot)?;
            r.slot = 0;
            let free = speculator
                .idle_slot(model, &[0])
                .context("no free slot to rebuild the draft in")?;
            speculator.rebuild(model, free, 0, &tokens)?;
            log_host(speculator);
            r.followed = true;
            format!("rebuilt the draft in slot {free}")
        };
        speculator.move_to_slot0(model, r.slot)?;
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
    /// chunk of the oldest request still prefilling, then runs batched steps of the others:
    /// one, or after a chunk as many as give each decoding request `floor` tokens per
    /// second of the time since the last step (`floor_steps`). A request left alone with
    /// nothing waiting returns, in slot 0, to continue with speculation (`speculate_again`). Each request takes its slot from the prefix
    /// cache (`Speculator::claim_slot`), and prefills from where its entry ends.
    #[expect(clippy::too_many_arguments, reason = "the scheduler's state")]
    fn batched(
        model: &mut Model,
        speculator: &mut Speculator,
        mut active: Vec<Request>,
        waiting: &mut VecDeque<Job>,
        queue: &Receiver<Vec<Job>>,
        stop: &[u32],
        streams: usize,
        greedy: bool,
        floor: f64,
    ) -> Option<Request> {
        // When the last batched step synced: the decoding requests have waited since then.
        let mut synced = Instant::now();
        loop {
            while active.len() < streams {
                if waiting.is_empty()
                    && let Ok(jobs) = queue.try_recv()
                {
                    waiting.extend(jobs);
                }
                let Some(job) = waiting.pop_front() else {
                    break;
                };
                if job.events.is_closed() {
                    continue;
                }
                let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
                let mut request = Request::new(job, 0, greedy);
                if request.done {
                    continue;
                }
                let prompt = &request.job.prompt;
                let claimed =
                    speculator.claim_slot(model, prompt, speculator.prefill_chunk(), &busy);
                log_host(speculator);
                match claimed {
                    Ok((slot, cached)) => {
                        request.slot = slot;
                        request.prefilled = cached;
                        request.cached = cached;
                        let _ = request.job.events.send(Event::Cached(cached));
                        active.push(request);
                    }
                    Err(e) => request.fail(&e),
                }
            }
            if active.is_empty() {
                return None;
            }
            let chunked = prefill_chunk(model, speculator, &mut active, stop);
            // Right after its prefill, a lone request speculates before any batched step,
            // as it would have from the start.
            if let [r] = active.as_mut_slice()
                && r.next.is_some()
                && !r.done
                && !r.stay_batched
                && waiting.is_empty()
            {
                match speculate_again(model, speculator, r) {
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
            let first_step = Instant::now();
            let mut steps = 0;
            while step_rows(model, speculator, &mut active, stop) {
                retire(speculator, &mut active);
                steps += 1;
                if !chunked || !floor_steps(floor, steps, synced, first_step) {
                    break;
                }
            }
            retire(speculator, &mut active);
            synced = Instant::now();
        }
    }

    /// Whether to run another batched step after a prefill chunk: while the `steps` since
    /// the step that ended at `synced` give a decoding request fewer than `floor` tokens
    /// per second of the time since then. The steps stop once they took as long as what
    /// ran before them, the chunk and the admissions, so the prefill keeps at least half
    /// of the time even when `floor` asks for more than a batched step can give.
    fn floor_steps(floor: f64, steps: usize, synced: Instant, first_step: Instant) -> bool {
        let waited = first_step.duration_since(synced).as_secs_f64();
        let stepped = first_step.elapsed().as_secs_f64();
        #[expect(clippy::cast_precision_loss, reason = "a few steps")]
        let behind = (steps as f64) < floor * (waited + stepped);
        behind && stepped < waited
    }

    /// Drops the requests of `active` that are done, and finishes those that ended without
    /// a failure.
    fn retire(speculator: &mut Speculator, active: &mut Vec<Request>) {
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

    /// Prefills one chunk of the oldest request of `active` still prefilling, in its slot,
    /// and keeps the prompt's state in the prefix cache when the chunk ends at the prompt's
    /// last multiple of the chunk. Returns whether it ran one.
    fn prefill_chunk(
        model: &mut Model,
        speculator: &mut Speculator,
        active: &mut [Request],
        stop: &[u32],
    ) -> bool {
        let Some(i) = active.iter().position(|r| r.next.is_none()) else {
            return false;
        };
        let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
        let r = &mut active[i];
        if r.job.events.is_closed() {
            r.done = true;
            return false;
        }
        if r.job.cancelled() {
            r.fail(&anyhow::anyhow!(CANCELLED));
            return false;
        }
        let len = r.job.prompt.len();
        let chunk = speculator.prefill_chunk();
        let end = (r.prefilled + chunk).min(len);
        let boundary = len.saturating_sub(1) / chunk * chunk;
        model.set_sampling(r.sampling);
        let first = match model.prefill_slot(r.slot, &r.job.prompt[r.prefilled..end], chunk) {
            Ok(first) => first,
            Err(e) => {
                r.fail(&e);
                return true;
            }
        };
        r.prefilled = end;
        if end == boundary
            && end > r.cached
            && let Err(e) = speculator.save_prefix(model, r.slot, &r.job.prompt, chunk, &busy)
        {
            eprintln!("keeping a prompt in the prefix cache failed: {e:#}");
        }
        log_host(speculator);
        if end == len {
            r.prefill_s = r.started.elapsed().as_secs_f64();
            r.next = Some(first);
            r.done = r.take(&[first], stop).is_break();
            share_prefill(model, active, i, stop);
        }
        true
    }

    /// Gives the prompt state that `active[from]` has just prefilled to its siblings, the
    /// requests of the same submission that have not started prefilling: copies its slot
    /// into theirs and draws each one's first token from the prefill's logits with its own
    /// sampling. Runs before anything else touches the model, since the logits live in
    /// the decode buffers and the next step moves `from`'s slot past the prompt.
    ///
    /// A copy is exact: a sibling's slot was reset on admission, and `move_slot` copies the
    /// global caches up to the prompt's end and the rings and conv states whole, so the
    /// slot ends up byte for byte as its own prefill would leave it. Its first token is the
    /// draw its own prefill would make, since a draw depends only on the logits, the
    /// sampling and the position.
    fn share_prefill(model: &mut Model, active: &mut [Request], from: usize, stop: &[u32]) {
        let (slot, group, len) = {
            let r = &active[from];
            (r.slot, r.job.group, r.job.prompt.len())
        };
        let siblings = active.iter_mut().filter(|r| {
            r.job.group == group
                && r.slot != slot
                && !r.done
                && r.next.is_none()
                && r.prefilled == 0
        });
        for r in siblings {
            let copied = model
                .move_slot(slot, r.slot)
                .and_then(|()| model.redraw_slot(r.slot, r.sampling));
            match copied {
                Ok(first) => {
                    r.prefilled = len;
                    r.shared = true;
                    r.prefill_s = r.started.elapsed().as_secs_f64();
                    r.next = Some(first);
                    r.done = r.take(&[first], stop).is_break();
                }
                Err(e) => r.fail(&e),
            }
        }
    }

    /// Runs one batched step of the requests of `active` that have a next token, and keeps
    /// the taps of the one the draft follows. Returns false when none has one.
    fn step_rows(
        model: &mut Model,
        speculator: &mut Speculator,
        active: &mut [Request],
        stop: &[u32],
    ) -> bool {
        let stepped: Vec<&Request> = active
            .iter()
            .filter(|r| !r.done && r.next.is_some())
            .collect();
        let rows: Vec<(usize, u32)> = stepped
            .iter()
            .map(|r| (r.slot, r.next.unwrap_or_default()))
            .collect();
        let sampling: Vec<ModelSampling> = stepped.iter().map(|r| r.sampling).collect();
        if rows.is_empty() {
            return false;
        }
        match model.step_batch_sampled(&rows, &sampling) {
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
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submissions_fit_whole_or_not_at_all() {
        let load = Arc::new(Load::default());
        // Two streams and a queue of 3: five jobs fit while none runs.
        assert!(!load.reserve(6, 3, 2));
        assert!(load.reserve(4, 3, 2));
        let mut tickets: Vec<Ticket> = (0..4).map(|_| Ticket::new(&load)).collect();
        assert!(!load.reserve(2, 3, 2));
        assert!(load.reserve(1, 3, 2));
        tickets.push(Ticket::new(&load));
        // Starting a job moves it from waiting to running: the idle streams shrink as the
        // waiting jobs do, so nothing more fits.
        tickets[0].start();
        tickets[1].start();
        assert!(!load.reserve(1, 3, 2));
        // A finished job and a cancelled waiting one each free a place.
        tickets.swap_remove(0);
        assert!(load.reserve(1, 3, 2));
        tickets.push(Ticket::new(&load));
        tickets.pop();
        tickets.pop();
        assert!(load.reserve(2, 3, 2));
        let counts = load.counts();
        assert_eq!((counts.waiting, counts.running), (4, 1));
    }
}
