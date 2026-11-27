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

use crate::serve::vision::{ImageStats, Images};

/// Why a job that a drain gave up on failed.
#[cfg(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64")))]
const CANCELLED: &str = "the server stopped before the request finished";

/// What a request asks the engine for.
#[cfg_attr(
    not(target_os = "linux"),
    expect(dead_code, reason = "only the Linux engine runs jobs")
)]
pub struct Job {
    pub prompt: Vec<u32>,
    /// The prompt's images, in prompt order.
    pub images: Images,
    /// The prompt's prefix cache key ([`siphon_cuda::spec::Prompt`]) when it has images;
    /// a prompt without them is its own key.
    key: Option<Vec<u32>>,
    /// Tokens to generate at most, the stop token excluded.
    pub max_tokens: usize,
    /// vLLM's `ignore_eos`: generate `max_tokens` tokens, passing the model's end tokens
    /// on as output.
    pub ignore_eos: bool,
    pub sampling: Sampling,
    pub events: UnboundedSender<Event>,
    /// The request's id and the response's id, for the job's log lines.
    pub label: Arc<str>,
    /// The request's log line, which takes the engine's summary of the job while the
    /// request runs.
    pub log: std::sync::Weak<crate::serve::log::RequestLog>,
    /// The submission the job came in: jobs of one submission share their prompt.
    group: u64,
    /// The job's place in the engine's [`Load`], from [`Engine::submit`].
    ticket: Option<Ticket>,
    /// The engine's drain state, from [`Engine::submit`].
    control: Option<Arc<Control>>,
    /// When the job was made, for the queue's order.
    #[cfg_attr(
        not(target_os = "linux"),
        expect(dead_code, reason = "only the Linux engine runs jobs")
    )]
    queued: std::time::Instant,
}

impl Job {
    pub fn new(
        prompt: Vec<u32>,
        max_tokens: usize,
        ignore_eos: bool,
        sampling: Sampling,
        events: UnboundedSender<Event>,
        label: Arc<str>,
    ) -> Self {
        Self {
            prompt,
            images: Images::from([]),
            key: None,
            max_tokens,
            ignore_eos,
            sampling,
            events,
            label,
            log: std::sync::Weak::new(),
            group: 0,
            ticket: None,
            control: None,
            queued: std::time::Instant::now(),
        }
    }

    /// The job with the prompt's `images` and its prefix cache `key`.
    #[must_use]
    pub fn with_images(self, images: Images, key: Option<Vec<u32>>) -> Self {
        Self {
            images,
            key,
            ..self
        }
    }

    /// The job with the request's log line `log`.
    #[must_use]
    pub fn with_log(self, log: std::sync::Weak<crate::serve::log::RequestLog>) -> Self {
        Self { log, ..self }
    }

    /// The key the prefix cache matches the prompt by.
    #[cfg_attr(
        not(target_os = "linux"),
        expect(dead_code, reason = "only the Linux engine runs jobs")
    )]
    fn key(&self) -> &[u32] {
        self.key.as_deref().unwrap_or(&self.prompt)
    }

    /// Whether a drain gave up on the job ([`Engine::cancel`]).
    #[cfg_attr(
        not(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64"))),
        expect(dead_code, reason = "only GPU backends run jobs")
    )]
    fn cancelled(&self) -> bool {
        self.control
            .as_ref()
            .is_some_and(|c| c.cancel.load(Ordering::Relaxed))
    }

    /// Counts the job as running rather than waiting.
    #[cfg_attr(
        not(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64"))),
        expect(dead_code, reason = "only GPU backends run jobs")
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
    not(any(
        target_os = "linux",
        all(target_os = "macos", target_arch = "aarch64"),
        test
    )),
    expect(dead_code, reason = "only GPU backends and tests emit finishes")
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model ended its turn.
    Stop,
    /// The request's token limit.
    Length,
}

/// Inline capacity for small token events: covers every single-token batched decode step
/// and every speculation verify round (`VERIFY_ROWS + 1 = 9`) without heap allocation.
pub const SMALL_TOKENS_CAP: usize = 9;

/// What the engine reports for a job, in order: `Started`, `Cached` at most once, `Tokens`
/// or `SmallTokens` any number of times, each after a `Round` when a speculation round
/// delivered them, then `Done` or `Failed`. A job whose client left while it waited gets
/// none.
#[cfg_attr(
    not(any(
        target_os = "linux",
        all(target_os = "macos", target_arch = "aarch64"),
        test
    )),
    expect(dead_code, reason = "only GPU backends and tests emit events")
)]
#[derive(Debug)]
pub enum Event {
    /// The engine took the job from its queue.
    Started,
    /// Prompt tokens whose state came from the prefix cache.
    #[cfg_attr(
        not(any(target_os = "linux", test)),
        expect(
            dead_code,
            reason = "only the CUDA backend and tests report cache hits"
        )
    )]
    Cached(usize),
    /// The next `Tokens` come from one speculation round, or from a plain step under
    /// `--fallback`.
    #[cfg_attr(
        not(any(target_os = "linux", test)),
        expect(
            dead_code,
            reason = "only the CUDA backend and tests report speculation"
        )
    )]
    Round,
    /// Inline small token batch (`1..=SMALL_TOKENS_CAP`), avoiding per-step heap
    /// allocation on the GPU engine thread.
    SmallTokens([u32; SMALL_TOKENS_CAP], u8),
    Tokens(Vec<u32>),
    /// The vision tower's work on the prompt's images, after the prefill of a prompt with
    /// any, before the first tokens.
    #[cfg_attr(
        not(any(target_os = "linux", test)),
        expect(dead_code, reason = "only the Linux engine runs jobs")
    )]
    Images(ImageStats),
    Done(Finish),
    Failed(String),
}

impl Event {
    /// Builds a token event from `ids`, keeping up to [`SMALL_TOKENS_CAP`] tokens inline
    /// with zero heap allocation.
    #[cfg_attr(
        not(any(
            target_os = "linux",
            all(target_os = "macos", target_arch = "aarch64"),
            test
        )),
        expect(dead_code, reason = "only GPU backends and tests emit token events")
    )]
    #[must_use]
    pub fn from_tokens(ids: &[u32]) -> Self {
        if ids.len() <= SMALL_TOKENS_CAP {
            let mut buf = [0u32; SMALL_TOKENS_CAP];
            buf[..ids.len()].copy_from_slice(ids);
            Self::SmallTokens(buf, ids.len() as u8)
        } else {
            Self::Tokens(ids.to_vec())
        }
    }
}

/// Whether the concatenation `first ++ second` starts with `prefix`, without allocating.
#[cfg_attr(
    all(not(target_os = "linux"), not(test)),
    expect(dead_code, reason = "used by the Linux engine and tests")
)]
pub(crate) fn starts_with_chained(first: &[u32], second: &[u32], prefix: &[u32]) -> bool {
    if prefix.len() > first.len() + second.len() {
        return false;
    }
    let split = prefix.len().min(first.len());
    first[..split] == prefix[..split] && second[..prefix.len() - split] == prefix[split..]
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
    /// Positions of the KV cache pool, which every running request shares and one request
    /// may fill: prompts and generated tokens. `None` takes
    /// the most that fit the memory budget.
    pub capacity: Option<usize>,
    /// Bytes of GPU memory the plan may use, the CUDA context included.
    pub memory: Option<usize>,
    /// Free bytes the plan leaves unclaimed; `None` leaves `memory::DEFAULT_MARGIN`.
    pub margin: Option<usize>,
    /// Rows of each prefill chunk; `None` runs `DEFAULT_PREFILL_CHUNK`.
    pub prefill_chunk: Option<usize>,
    pub fp8_weights: bool,
    /// Format of the attention output projection decode reads under `--weights fp8`:
    /// `bf16` (default), `fp8` (per-row E4M3), or `mxfp8` (per-32 UE8M0 block-scaled E4M3).
    pub output_projection: siphon::OutputProjection,
    /// Optional override for the LM head's weight format (`bf16`, `fp8`, or `nvfp4`).
    pub lm_head: Option<String>,
    /// The global KV cache format, as `KvFormat` parses it.
    pub kv: String,
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
    /// Draft weight override (`auto|bf16|fp8|nvfp4|nvfp4-backbone`).
    pub draft_weights: String,
    /// Optional shortlist ranked-token budget (`--draft-shortlist`).
    pub draft_shortlist: Option<usize>,
    /// Optional shortlist context-token budget (`--draft-context-rows`).
    pub draft_context_rows: Option<usize>,
    /// Whether the speculator falls back to plain steps (`WidthPolicy::with_plain`).
    pub fallback: bool,
    /// Whether opt-in 2-branch speculative tree verification is enabled (`--spec-tree`).
    pub spec_tree: bool,
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
    /// Dense product precision in prefill and wide batched passes (`m >= 48`): `bf16` or
    /// `fp8`, as `Products` parses it.
    pub prefill_dense: String,
    /// Expert activation precision (`bf16`, `quantized`, or `fp8`), as `ExpertActivations`
    /// parses it.
    pub expert_activations: String,
    /// Storage format of NVFP4 expert block scales (`8bit` E4M3 or `4bit` per-row offsets).
    pub nvfp4_scales: siphon::Nvfp4ScaleFormat,
    /// Storage format of MXFP8 expert layers (`mxfp8`, `mxfp6`, or load-time `nvfp4`).
    pub mxfp8_experts: siphon::Mxfp8ExpertFormat,
    /// Whether every request decodes greedily, whatever it asks.
    pub greedy: bool,
    /// Tokens per second that each decoding request gets at least while another prefills,
    /// with `streams` above 1, or 0 for one batched step per prefill chunk.
    pub decode_floor: f64,
    /// Max prefill tokens piggybacked onto one active decode/verify forward pass, or `None`
    /// when piggybacking is disabled.
    pub piggyback_budget: Option<usize>,
    /// The vision tower, with `--vision`.
    pub vision: Option<VisionOptions>,
    /// Opt-in logit margin for speculative acceptance (`0.0` for exact lossless acceptance).
    pub accept_margin: f32,
    /// Opt-in beta-relaxed / maximal-coupling rejection sampling parameter (`(0, 1]`).
    pub spec_beta: Option<f32>,
    /// Draft temperature scale `alpha = T_draft / T_target` for DFlash 2 (default `0.85`).
    pub draft_temp_scale: f32,
    /// Split mixed-confidence batched DFlash 2 rounds into high/low confidence micro-batches.
    pub group_by_confidence: bool,
    /// The KV options of `siphon_cuda::kv_options` as given on the command line.
    pub kv_args: Vec<String>,
    /// Maximum MoE experts per token (`1..=8`, `--moe-top-k`).
    pub moe_top_k: usize,
    /// Minimum MoE experts kept per token under threshold pruning (`1..=moe_top_k`,
    /// `--moe-min-k`).
    pub moe_min_k: usize,
    /// Probability share threshold for MoE expert pruning (`0.0..=1.0`, `--moe-prune`).
    pub moe_prune: f32,
}

/// `--vision`'s tower, which runs in BF16: its FP8 products drift 3 to 5 times further
/// from the reference (docs/vision.md).
#[derive(Clone, Copy, Debug)]
pub struct VisionOptions {
    /// `--max-image-pixels`.
    pub max_pixels: usize,
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

#[cfg(test)]
impl Engine {
    /// An engine without a GPU thread: its jobs arrive on the receiver, for tests.
    pub fn fake(max_context: usize) -> (Self, mpsc::Receiver<Vec<Job>>) {
        let (jobs, receiver) = mpsc::sync_channel(8);
        let engine = Self {
            jobs,
            max_context,
            load: Arc::default(),
            queue: 8,
            streams: 1,
            control: Arc::default(),
        };
        (engine, receiver)
    }
}

#[cfg(not(any(target_os = "linux", all(target_os = "macos", target_arch = "aarch64"))))]
pub fn start(_options: &Options, _stop: Vec<u32>, _warmup: Vec<u32>) -> anyhow::Result<Engine> {
    anyhow::bail!("the engine requires Linux with CUDA or Apple Silicon macOS with MLX")
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod mlx;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) use mlx::start;

#[cfg(target_os = "linux")]
pub use cuda::start;

#[cfg(target_os = "linux")]
mod cuda {
    use std::{
        collections::VecDeque,
        ops::ControlFlow,
        sync::mpsc::{Receiver, TryRecvError, sync_channel},
        thread,
        time::Instant,
    };

    use std::sync::Arc;

    use anyhow::Context as _;
    use cudarc::driver::{CudaEvent, CudaSlice, sys::CUevent_flags};
    use half::bf16;
    use siphon::{
        Checkpoint, TextConfig,
        memory::{
            self, CONTEXT_BYTES, DEFAULT_MARGIN, DraftChoice, PlanInputs, Planner, VisionInputs,
            gib,
        },
        vision::VisionConfig,
    };
    use siphon_cuda::{
        Gpu, MemoryUse,
        draft::DraftOptions,
        kv_options::KvOptions,
        model::{
            BATCH_ROWS, BATCH_VERIFY_MAX_ROWS, DEFAULT_PREFILL_CHUNK, ImageRows, MAP_POSITIONS,
            Model, MoePruning, PackedPrefill, Sampling as ModelSampling,
        },
        ops::{attention::KvFormat, spec::VERIFY_ROWS},
        spec::{
            DiskLoaded, DraftSource, Drafts, HOST_CHUNK_POSITIONS, PrefixTier, Prompt, Rebuild,
            Speculator, Width, WidthPolicy, load_draft, price,
        },
        vision::{VisionFormat, VisionTower},
        vmm::{PagePool, granularity},
        weights::{OutputProjection, WeightFormat, Weights},
    };

    use super::{CANCELLED, Control, Engine, Event, Finish, Job, Options, starts_with_chained};
    use crate::serve::{
        log::log,
        piggyback::{self, PiggybackSlice, PrefillCandidate, TurnPlan},
        vision::{ImageStats, PromptImage},
    };

    /// Prompts of at least this many tokens get a log line about their prefix cache lookup
    /// when they start, since their prefill takes seconds.
    const LONG_PROMPT: usize = 16_384;

    /// The vision tower, with `--vision`.
    type Tower = Option<VisionTower>;

    /// An image whose rows a prefill chunk encoded and the next chunk needs too: its index
    /// among the prompt's images, and its rows.
    type Kept = Option<(usize, Arc<CudaSlice<bf16>>)>;

    /// The GPU events around each image the tower encoded for one request's prefill.
    #[derive(Debug, Default)]
    struct Encodes(Vec<(CudaEvent, CudaEvent)>);

    impl Encodes {
        /// The tower's work for a prompt with `images`, of which `cached` tokens came from
        /// the prefix cache. Waits for the last encode.
        fn stats(&self, images: &[PromptImage], cached: usize) -> ImageStats {
            let ms = self
                .0
                .iter()
                .map(|(start, end)| start.elapsed_ms(end).map_or(0.0, f64::from))
                // From 0: `sum` of no floats is -0.
                .fold(0.0, |all, ms| all + ms);
            ImageStats {
                encoded: self.0.len(),
                ms,
                cached: images
                    .iter()
                    .filter(|image| image.positions().end <= cached)
                    .count(),
            }
        }
    }

    /// The rows of the `images` whose placeholders overlap prompt positions `start..end`,
    /// encoded by the tower as prefill reaches them. An image that runs past `end` stays in
    /// `kept` for the next chunk, so that each image is encoded once per prefill. Each
    /// encode's events go to `timers`.
    fn image_rows(
        tower: &mut Tower,
        hidden: usize,
        images: &[PromptImage],
        kept: &mut Kept,
        timers: &mut Encodes,
        (start, end): (usize, usize),
    ) -> anyhow::Result<Vec<ImageRows>> {
        let mut rows = Vec::new();
        for (i, image) in images.iter().enumerate() {
            let positions = image.positions();
            if positions.end <= start || positions.start >= end {
                continue;
            }
            let encoded = match kept.take() {
                Some((k, encoded)) if k == i => encoded,
                _ => {
                    let tower = tower.as_mut().context("this server has no vision tower")?;
                    let timed = Some(CUevent_flags::CU_EVENT_DEFAULT);
                    let begin = tower.stream().record_event(timed)?;
                    let encoded = tower.encode(&image.image)?;
                    timers.0.push((begin, tower.stream().record_event(timed)?));
                    anyhow::ensure!(
                        encoded.len() == positions.len() * hidden,
                        "the vision tower gave {} values for an image of {} tokens",
                        encoded.len(),
                        positions.len()
                    );
                    Arc::new(encoded)
                }
            };
            if positions.end > end {
                *kept = Some((i, Arc::clone(&encoded)));
            }
            rows.push(ImageRows {
                position: positions.start,
                rows: encoded,
            });
        }
        Ok(rows)
    }

    /// Positions past the last generated token that a round may write: a full verify.
    const MARGIN: usize = VERIFY_ROWS + 1;

    /// Positions past its prompt that a request needs room for in the KV pool to start:
    /// one chunk of the pool, so that it decodes that far before it needs more.
    const HEADROOM: usize = MAP_POSITIONS;

    /// Number of [`MAP_POSITIONS`] chunks to pre-map ahead when free KV pool pages allow,
    /// avoiding mid-decode `cuMemSetAccess` stalls during batched speculation.
    const PREMAP_GROW_CHUNKS: usize = 2;

    /// How long a request may wait before it goes ahead of shorter ones: until then the
    /// queue starts the shortest prompts first.
    const AGE_LIMIT: std::time::Duration = std::time::Duration::from_secs(20);

    /// Bytes per second that a parked sequence returns from host memory at, a low
    /// estimate of the 26.5 to 27.4 GB/s measured over PCIe 4.0 x16.
    const RESTORE_RATE: f64 = 20e9;

    /// The share of the time that swapping may take when sequences that do not fit the
    /// KV pool together take turns: a request runs at least this many times as long as it
    /// takes to swap it out and back before it may be swapped out again.
    const SWAP_SHARE: f64 = 0.1;

    /// The shortest turn a request gets before round robin may swap it out.
    const MIN_QUANTUM: f64 = 1.0;

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
                Ok((model, speculator, tower, capacity, check)) => {
                    let _ = ready.send(Ok(capacity));
                    run(
                        model,
                        speculator,
                        tower,
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
                let pool_limit = planner.max_pool_positions();
                anyhow::ensure!(
                    capacity <= pool_limit,
                    "--context {capacity} exceeds the pool limit of {pool_limit} positions \
                     ({} stream{} × {} per-request model limit)",
                    options.streams,
                    if options.streams == 1 { "" } else { "s" },
                    planner.max_positions()
                );
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
        let slot_capacity = capacity.min(planner.max_positions());
        anyhow::ensure!(
            slot_capacity > warmup + 128 + MARGIN,
            "a context of {capacity} is too small"
        );
        log!("memory plan: {}", planner.plan(capacity));
        log!("memory {budget_line}");
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
        /// `kv_pool_bytes` is the KV pool, which lies outside the memory pool too.
        fn new(
            gpu: Gpu,
            planned: usize,
            free: usize,
            kv_pool_bytes: usize,
        ) -> anyhow::Result<Self> {
            let now = gpu.memory()?;
            // Other processes can change the free memory meanwhile; the pool is exact.
            let used = free.saturating_sub(now.free);
            let outside_pool = used.saturating_sub(now.pool_reserved);
            let kv_pool = kv_pool_bytes.min(outside_pool);
            let mut check = Self {
                gpu,
                planned,
                outside_pool,
                warned: false,
                reported: 0,
            };
            log!(
                "memory check: {} GiB of a planned {} GiB in use after warmup ({} GiB in \
                 buffers, peak {} GiB; {} GiB in the KV pool; {} GiB in modules and graphs)",
                gib(check.peak(&now)),
                gib(planned),
                gib(now.pool_reserved),
                gib(now.pool_reserved_peak),
                gib(kv_pool),
                gib(outside_pool - kv_pool),
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
                        log!(
                            "memory check: peak {} MiB of a planned {} MiB after a job",
                            peak >> 20,
                            self.planned >> 20
                        );
                    }
                    self.warn(&now);
                }
                Err(e) => log!("memory check failed: {e:#}"),
            }
        }

        fn warn(&mut self, now: &MemoryUse) {
            let peak = self.peak(now);
            if !self.warned && peak > self.planned {
                self.warned = true;
                log!(
                    "warning: memory use peaked at {} GiB, {} MiB past the plan; the plan \
                     undercounts this configuration",
                    gib(peak),
                    (peak - self.planned) >> 20
                );
            }
        }
    }

    #[expect(clippy::too_many_lines, reason = "startup, step by step")]
    fn load(
        options: &Options,
        warmup: &[u32],
    ) -> anyhow::Result<(Model, Speculator, Tower, usize, MemoryCheck)> {
        let started = Instant::now();
        let config = TextConfig::load(&options.model_dir)?;
        let passes = config.loop_config.num_passes;
        let gpu = Gpu::new(0)?;
        let (free, _) = gpu.context().mem_get_info()?;
        let checkpoint = Checkpoint::open(&options.model_dir)?;
        let (weight_format, kv) = formats(options)?;
        let mut draft_options = DraftOptions::parse(&options.draft, None)?
            .with_shortlist_budget(options.draft_shortlist, options.draft_context_rows)?;
        draft_options.apply_draft_weights(&options.draft_weights)?;
        let chunk = options.prefill_chunk.unwrap_or(DEFAULT_PREFILL_CHUNK);
        let kv_options = kv_options(options)?;
        let output = output_projection(options);
        let lm_head: Option<siphon::LmHeadFormat> =
            options.lm_head.as_deref().map(str::parse).transpose()?;
        let prefill_dense_fp8 = options.prefill_dense == "fp8";
        let planner = Planner::new(&PlanInputs {
            config: &config,
            checkpoint: &checkpoint,
            weights: weight_format,
            output,
            lm_head,
            prefill_dense_fp8,
            block_scaled_mma: siphon_cuda::BLOCK_SCALED_MMA,
            nvfp4_scales: options.nvfp4_scales,
            mxfp8_experts: options.mxfp8_experts,
            kv,
            draft: options
                .drafter
                .as_deref()
                .map_or(DraftChoice::Bundled(draft_options.approx()), |dir| {
                    DraftChoice::Dflash(dir, draft_options.approx())
                }),
            prefill_chunk: chunk,
            streams: options.streams,
            pooled: true,
            prefix_entries: options.prefix_entries,
            sms: gpu.sms()?,
            vision: vision_inputs(options)?,
        })?;
        let capacity = plan(options, &planner, free, warmup.len())?;
        // Each individual sequence is capped at the model's `model_max_length` (1,048,576),
        // while the shared KV page pool can hold up to `streams * model_max_length` aggregate
        // positions across concurrent requests.
        let slot_capacity = capacity.min(config.model_max_length);
        let weights = Weights::load_with_options(
            &checkpoint,
            &config,
            &gpu,
            siphon_cuda::weights::LoadOptions {
                weights: weight_format,
                output,
                lm_head,
                prefill_dense_fp8,
                nvfp4_scales: options.nvfp4_scales,
                mxfp8_experts: options.mxfp8_experts,
            },
        )?;
        let (page, _) = granularity(gpu.context())?;
        if page != memory::KV_PAGE_BYTES {
            log!(
                "warning: the device maps pages of {} KiB, the plan assumes {} KiB",
                page >> 10,
                memory::KV_PAGE_BYTES >> 10
            );
        }
        let pool_started = Instant::now();
        let pool = PagePool::new(gpu.context(), planner.plan(capacity).kv.div_ceil(page))?;
        log!(
            "KV pool: {} pages of {} MiB ({} GiB) in {:.2} s, for {capacity} positions over \
             up to {} streams (up to {slot_capacity} per request)",
            pool.total_pages(),
            page >> 20,
            gib(pool.total_pages() * page),
            pool_started.elapsed().as_secs_f64(),
            options.streams
        );
        let mut model = Model::with_kv_pool(
            &gpu,
            config,
            weights,
            passes,
            slot_capacity,
            kv,
            weight_format,
            Some(pool),
        )?;
        kv_options.apply(&mut model)?;
        let moe_pruning = MoePruning {
            top_k: options.moe_top_k,
            threshold: options.moe_prune,
            min_k: options.moe_min_k,
        };
        model.set_moe_pruning(moe_pruning)?;
        if moe_pruning.is_active() {
            log!("moe pruning: {moe_pruning}");
        }
        model.set_prefill_attention(options.prefill_attention.parse()?)?;
        model.set_prefill_dense(options.prefill_dense.parse()?)?;
        model.set_expert_activations(options.expert_activations.parse()?)?;
        if !options.greedy {
            model.enable_sampling()?;
        }
        model.set_accept_margin(options.accept_margin)?;
        if options.accept_margin > 0.0 {
            log!(
                "speculative margin acceptance enabled (margin {:.3})",
                options.accept_margin
            );
        }
        model.set_spec_beta(options.spec_beta)?;
        if let Some(beta) = options.spec_beta {
            log!("speculative beta rejection enabled (beta {beta:.2})");
        }
        let width = Width::Adaptive(WidthPolicy::new(&options.widths, [1.0; VERIFY_ROWS])?);
        let source = options
            .drafter
            .as_deref()
            .map_or(DraftSource::Bundled, DraftSource::Dflash);
        let draft = load_draft(
            &mut model,
            &checkpoint,
            source,
            slot_capacity,
            &draft_options,
        )?;
        let mut speculator = Speculator::with_draft(&model, draft, width)?;
        speculator.set_draft_temp_scale(options.draft_temp_scale)?;
        if options.spec_tree {
            speculator.set_spec_tree(siphon_cuda::spec::SpecTreeConfig::two_branch())?;
        }
        speculator.set_prefill_chunk(chunk)?;
        let tower = load_tower(options, &gpu, &checkpoint)?;
        log!("loaded in {:.1} s", started.elapsed().as_secs_f64());

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
        log!(
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
            if speculator.supports_batch_spec() {
                let spec_seqs = options.streams.min(speculator.max_batch_spec_seqs());
                model.reserve_batch_verify(spec_seqs)?;
                speculator.reserve_batch_spec(&model, spec_seqs, options.streams - 1)?;
                let warmup_started = Instant::now();
                let captured = speculator.warmup_batch_spec(&mut model, spec_seqs, warmup[0])?;
                log!(
                    "batched DFlash 2 speculation enabled (up to {spec_seqs} sequences per verify, \
                     pre-captured {captured} bucketed verify+draft graph pairs in {:.2} s)",
                    warmup_started.elapsed().as_secs_f64()
                );
            }
            if let Some(budget) = options.piggyback_budget {
                log!(
                    "hybrid piggyback prefill enabled (up to {budget} prefill tokens per \
                     decode/verify step)"
                );
            }
            log!("serving up to {} requests at once", options.streams);
        }
        drop(saved);
        // Pricing left the draft's caches mapped; the pool starts empty.
        speculator.trim_draft(&model)?;
        model.reset()?;
        if options.prefix_entries > 0 {
            let bytes = speculator.enable_prefix_cache(
                &model,
                options.prefix_entries,
                options.prefix_ttl,
            )?;
            log!(
                "prefix cache: {} entries, {} MiB of copies besides their rings' in the KV pool, \
                 {} s to live",
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
        let kv_pool = model
            .kv_pool()
            .map_or(0, |p| p.total_pages() * p.page_bytes());
        let check = MemoryCheck::new(gpu, planner.plan(capacity).total(), free, kv_pool)?;
        Ok((model, speculator, tower, slot_capacity, check))
    }

    /// Loads the vision tower with `--vision`, on the model's stream, with its buffers for
    /// the largest image, so that the memory check after warmup sees them.
    fn load_tower(options: &Options, gpu: &Gpu, checkpoint: &Checkpoint) -> anyhow::Result<Tower> {
        let Some(v) = vision_inputs(options)? else {
            return Ok(None);
        };
        let mut tower = VisionTower::load(gpu, checkpoint, v.format)?;
        tower.reserve(v.max_image_tokens * v.patches_per_token)?;
        log!(
            "vision tower: {} weights, images shrunk to {} pixels, up to {} image tokens",
            v.format,
            options.vision.map_or(0, |o| o.max_pixels),
            v.max_image_tokens
        );
        Ok(Some(tower))
    }

    /// The plan's vision tower and largest image, with `--vision`.
    fn vision_inputs(options: &Options) -> anyhow::Result<Option<VisionInputs>> {
        options
            .vision
            .map(|v| {
                let config = VisionConfig::load(&options.model_dir)?;
                Ok(VisionInputs {
                    format: VisionFormat::Bf16,
                    max_image_tokens: memory::max_image_tokens(config.token_size(), v.max_pixels),
                    patches_per_token: config.merge_size * config.merge_size,
                })
            })
            .transpose()
    }

    /// The attention output projection's format `options` ask for.
    fn output_projection(options: &Options) -> OutputProjection {
        options.output_projection
    }

    /// The KV options of `siphon_cuda::kv_options` that `options` carries.
    fn kv_options(options: &Options) -> anyhow::Result<KvOptions> {
        let mut kv_options = KvOptions::default();
        let mut kv_args = options.kv_args.iter().cloned();
        while let Some(arg) = kv_args.next() {
            anyhow::ensure!(
                kv_options.parse(&arg, &mut kv_args)?,
                "unknown KV option {arg}"
            );
        }
        Ok(kv_options)
    }

    /// The weight and KV cache formats `options` ask for.
    fn formats(options: &Options) -> anyhow::Result<(WeightFormat, KvFormat)> {
        let weights = if options.fp8_weights {
            WeightFormat::Fp8
        } else {
            WeightFormat::Bf16
        };
        Ok((weights, options.kv.parse()?))
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
        log!(
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
        log!("host cache: {line}");
        log_cache(speculator);
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
            "kv": options.kv,
            "weights": if options.fp8_weights { "fp8" } else { "bf16" },
            "lm_head": options.lm_head,
            "draft": options.draft,
            "draft_weights": options.draft_weights,
            "prefill_attention": options.prefill_attention,
            "prefill_dense": options.prefill_dense,
            "expert_activations": options.expert_activations,
            "nvfp4_scales": options.nvfp4_scales.to_string(),
            "mxfp8_experts": options.mxfp8_experts.to_string(),
            "moe_top_k": options.moe_top_k,
            "moe_min_k": options.moe_min_k,
            "moe_prune": options.moe_prune,
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
        log_cache(speculator);
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
        /// The prefix cache tier its cached prompt tokens came from.
        tier: Option<&'static str>,
        /// Whether the draft follows this request through batched steps.
        followed: bool,
        /// Whether this slot's DFlash 2 sliding rings hold its positions for batched
        /// speculation.
        draft_ready: bool,
        /// Whether returning to speculation failed, so that the request stays batched.
        stay_batched: bool,
        /// Tokens emitted by speculation and by batched steps.
        speculated: usize,
        batched: usize,
        /// Batched steps it ran as the only row, which pay for a draft rebuild
        /// (`rebuild_turn`).
        rent: usize,
        /// The image a batched prefill chunk encoded for the next one.
        kept: Kept,
        /// The tower's encodes for its prefill.
        encodes: Encodes,
        /// When the request last started to run: admitted, or back from the host tier.
        since: Instant,
        /// Its id in the host tier while swapped out (`park`).
        parked: Option<usize>,
        /// Times it was swapped out, and the seconds it spent swapped out.
        parks: usize,
        parked_s: f64,
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
                tier: None,
                followed: false,
                draft_ready: false,
                stay_batched: false,
                speculated: 0,
                batched: 0,
                rent: 0,
                kept: None,
                encodes: Encodes::default(),
                since: Instant::now(),
                parked: None,
                parks: 0,
                parked_s: 0.0,
            };
            if let Some(e) = error {
                request.fail(&e);
            } else if request.job.cancelled() {
                request.fail(&anyhow::anyhow!(CANCELLED));
            }
            request
        }

        /// Positions its slot holds: the prompt prefilled so far, then every generated
        /// token but the last, which is fed next.
        fn position(&self) -> usize {
            if self.next.is_some() {
                self.job.prompt.len() + self.output.len() - 1
            } else {
                self.prefilled
            }
        }

        /// The ids at its positions: the prompt's key ([`Job::key`]), then the generated
        /// tokens, which are their own keys.
        fn fed(&self) -> Vec<u32> {
            let key = self.job.key();
            if self.next.is_some() {
                let generated = &self.output[..self.output.len() - 1];
                key.iter().chain(generated).copied().collect()
            } else {
                key[..self.prefilled].to_vec()
            }
        }

        /// The most positions it can reach: its prompt and every token it may generate.
        fn final_position(&self, capacity: usize) -> usize {
            (self.job.prompt.len() + self.job.max_tokens + MARGIN).min(capacity)
        }

        /// The positions that a lone request takes room for when it starts: its prompt
        /// and [`HEADROOM`] more, at most its final position.
        fn first_position(&self, capacity: usize) -> usize {
            (self.job.prompt.len() + HEADROOM).min(self.final_position(capacity))
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
                    .send(Event::from_tokens(&ids[..keep]))
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

        /// Gives the request's log line the engine's summary of the job, or logs it on its
        /// own line when the request has already ended, and tells its client it is done.
        fn finish(&self, detail: &str) {
            let decode_s = self.started.elapsed().as_secs_f64() - self.prefill_s;
            let events = &self.job.events;
            let images = match self.job.images.len() {
                0 => String::new(),
                n => {
                    let stats = self.encodes.stats(&self.job.images, self.cached);
                    format!(
                        ", {n} images: {} encoded in {:.0} ms, {} from the cache",
                        stats.encoded, stats.ms, stats.cached
                    )
                }
            };
            let parks = if self.parks > 0 {
                format!(
                    ", swapped out {} times for {:.2} s",
                    self.parks, self.parked_s
                )
            } else {
                String::new()
            };
            let summary = format!(
                "{} speculated, {} batched steps{}{images}{parks}{detail}",
                self.speculated,
                self.batched,
                if self.shared {
                    ", prompt copied from a sibling"
                } else {
                    ""
                },
            );
            let tier = if self.shared {
                Some("copied")
            } else {
                self.tier
            };
            match self.job.log.upgrade() {
                Some(log) if !events.is_closed() => log.engine(tier, summary),
                _ => log!(
                    "job: {}: client left after {} of {} tokens in {:.2} s ({} prompt tokens, \
                     {} cached): {summary}",
                    self.job.label,
                    self.output.len(),
                    self.job.max_tokens,
                    decode_s.max(0.0),
                    self.job.prompt.len(),
                    self.cached,
                ),
            }
            let _ = events.send(Event::Done(self.finish));
        }

        /// Records that `cached` prompt tokens came from the prefix cache, in the tier of
        /// `speculator`'s last lookup, tells the client, and logs the lookup of a long
        /// prompt.
        fn cached_from(&mut self, speculator: &Speculator, cached: usize) {
            self.cached = cached;
            let tier = speculator.last_prefix_tier().filter(|_| cached > 0);
            self.tier = tier.map(PrefixTier::name);
            let _ = self.job.events.send(Event::Cached(cached));
            let len = self.job.prompt.len();
            if len < LONG_PROMPT {
                return;
            }
            match tier {
                Some(tier) => log!(
                    "job: {}: prefix cache hit, {} of {len} prompt tokens from the {} tier",
                    self.job.label,
                    self.cached,
                    tier.name()
                ),
                None => log!(
                    "job: {}: prefix cache miss, prefilling all {len} prompt tokens",
                    self.job.label
                ),
            }
        }

        /// Tells its client what the tower did, once its prompt's prefill has ended.
        fn send_images(&self) {
            if !self.job.images.is_empty() {
                let stats = self.encodes.stats(&self.job.images, self.cached);
                let _ = self.job.events.send(Event::Images(stats));
            }
        }

        fn fail(&mut self, e: &anyhow::Error) {
            log!("job failed: {}: {e:#}", self.job.label);
            let _ = self.job.events.send(Event::Failed(format!("{e:#}")));
            self.done = true;
            self.failed = true;
        }
    }

    /// What the scheduler goes by.
    struct Sched {
        streams: usize,
        greedy: bool,
        /// Tokens per second each decoding request gets at least while another prefills.
        floor: f64,
        /// Max prefill tokens piggybacked onto one active decode/verify step, or `None` when off.
        piggyback_budget: Option<usize>,
        /// Positions of the KV pool: the most one request may hold.
        capacity: usize,
        /// Split mixed-confidence batched DFlash 2 rounds into high/low micro-batches.
        group_by_confidence: bool,
    }

    #[expect(clippy::too_many_arguments, reason = "the engine thread's state")]
    #[expect(
        clippy::too_many_lines,
        reason = "the engine thread's loop, top to bottom"
    )]
    fn run(
        mut model: Model,
        mut speculator: Speculator,
        mut tower: Tower,
        mut check: MemoryCheck,
        queue: &Receiver<Vec<Job>>,
        stop: &[u32],
        options: &Options,
        control: &Control,
    ) {
        let capacity = model.slot_capacity();
        let sched = Sched {
            streams: options.streams,
            greedy: options.greedy,
            floor: options.decode_floor,
            piggyback_budget: options.piggyback_budget,
            capacity,
            group_by_confidence: options.group_by_confidence,
        };
        let (streams, greedy) = (sched.streams, sched.greedy);
        let mut waiting = VecDeque::new();
        // A request that a batch left alone, to continue with speculation.
        let mut resumed = None;
        // A draft rebuild in progress, and when it started (`idle_rebuild`, `rebuild_turn`).
        let mut rebuild = None;
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
                    // Idle: once no job came for `IDLE_DELAY`, a step of a draft rebuild,
                    // then a look at the queue.
                    let jobs = if streams == 1 {
                        queue.recv().ok()
                    } else if rebuild.is_none()
                        && let Ok(jobs) = queue.recv_timeout(IDLE_DELAY)
                    {
                        Some(jobs)
                    } else if idle_rebuild(&mut model, &mut speculator, &mut rebuild) {
                        match queue.try_recv() {
                            Ok(jobs) => Some(jobs),
                            Err(TryRecvError::Empty) => continue,
                            Err(TryRecvError::Disconnected) => None,
                        }
                    } else {
                        queue.recv().ok()
                    };
                    let Some(jobs) = jobs else { return };
                    waiting.extend(jobs);
                }
                let Some(job) = waiting.pop_front() else {
                    continue;
                };
                if job.events.is_closed() {
                    continue;
                }
                // Another job already waits: start them together.
                if streams > 1 {
                    while let Ok(jobs) = queue.try_recv() {
                        waiting.extend(jobs);
                    }
                    if !waiting.is_empty() {
                        waiting.push_front(job);
                        resumed = batched(
                            &mut model,
                            &mut speculator,
                            &mut tower,
                            Vec::new(),
                            &mut waiting,
                            queue,
                            stop,
                            &sched,
                            &mut rebuild,
                        );
                        check.after_job();
                        continue;
                    }
                }
                // A prompt whose entry lacks the draft's state prefills from that entry as
                // a batch of one rather than whole in slot 0.
                let hit =
                    speculator.prefix_lookup(&model, job.key(), speculator.prefill_chunk(), &[]);
                if streams > 1 && hit.is_some_and(|hit| !hit.draft) {
                    waiting.push_front(job);
                    resumed = batched(
                        &mut model,
                        &mut speculator,
                        &mut tower,
                        Vec::new(),
                        &mut waiting,
                        queue,
                        stop,
                        &sched,
                        &mut rebuild,
                    );
                    check.after_job();
                    continue;
                }
                let request = Request::new(job, 0, greedy);
                if request.done {
                    continue;
                }
                // Room for the prompt and `HEADROOM` more, the draft's and its entry's
                // copies of the rings; `alone` maps the rest as the reply grows. A sequence
                // that moves into or out of slot 0 first is copied while the pool has room
                // for the copy (`Model::transfer_slot`), so the room counts the copy too:
                // the hit's, or slot 0's entry moving aside. Only a hit in slot 0 stays.
                // Dropping the draft's pages would drop the hit's draft state with them,
                // and the hit with it.
                let end = request.first_position(capacity);
                let hit_slot = hit.and_then(|h| h.slot);
                let mut need =
                    model.pages_for(end) + speculator.draft_pages_to(end) + model.ring_pages();
                if hit_slot == Some(0) {
                    need = need.saturating_sub(model.slot_pages(0).unwrap_or(0));
                } else {
                    need += model.pages_for(model.slot_position(0).unwrap_or(0));
                }
                let keep: Vec<usize> = hit_slot.into_iter().collect();
                let draft_hit = hit.is_some_and(|h| h.draft && h.slot.is_some());
                let reclaim = if draft_hit {
                    Reclaim::Entries
                } else {
                    Reclaim::Draft
                };
                make_room(&mut model, &mut speculator, need, &keep, reclaim);
                request
            };
            // Speculation takes the draft from any rebuild.
            rebuild = None;
            let interrupt = (streams > 1).then_some((queue, &mut waiting));
            let ran = alone(
                &mut model,
                &mut speculator,
                &mut tower,
                &mut request,
                stop,
                capacity,
                interrupt,
            );
            match ran {
                Ok(true) => {
                    if speculator.supports_batch_spec() {
                        if let Err(e) = speculator.sync_slot0_draft(&model) {
                            log!("syncing slot 0 DFlash 2 state failed: {e:#}");
                            speculator.follow(&model);
                            request.followed = true;
                        } else {
                            request.draft_ready = true;
                        }
                    } else {
                        speculator.follow(&model);
                        request.followed = true;
                    }
                    resumed = batched(
                        &mut model,
                        &mut speculator,
                        &mut tower,
                        vec![request],
                        &mut waiting,
                        queue,
                        stop,
                        &sched,
                        &mut rebuild,
                    );
                }
                Ok(false) => {}
                Err(e) => request.fail(&e),
            }
            check.after_job();
        }
    }

    /// Pre-maps KV pages for `slot` past `position` (up to `PREMAP_GROW_CHUNKS` chunks ahead,
    /// falling back to 1 chunk) when the pool has surplus free pages beyond `reserve_pages`,
    /// coalescing multiple pages into one `cuMemSetAccess` per buffer and preventing
    /// mid-decode VMM stalls.
    fn premap_slot_headroom(
        model: &mut Model,
        slot: usize,
        position: usize,
        final_position: usize,
        reserve_pages: usize,
    ) {
        let Some(pool) = model.kv_pool() else {
            return;
        };
        let available = pool.free_pages().saturating_sub(reserve_pages);
        for chunks in [PREMAP_GROW_CHUNKS, 1] {
            let target =
                memory::decode_premap_target(position, final_position, MAP_POSITIONS, chunks);
            let Ok(need) = model.slot_pages_to(slot, target) else {
                return;
            };
            if need == 0 {
                return;
            }
            if need <= available {
                let _ = model.map_slot(slot, target);
                return;
            }
        }
    }

    /// Returns the pages of idle slot `slot` that its prefix cache entry does not need to
    /// the KV pool. A failure goes to the log.
    fn release(model: &mut Model, speculator: &mut Speculator, slot: usize) {
        if let Err(e) = speculator.release_slot(model, slot) {
            log!("returning slot {slot}'s pages failed: {e:#}");
        }
    }

    /// What `make_room` may free, each level with what the levels before it free.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum Reclaim {
        /// The pages that idle slots keep past their prefix cache entries and their rings,
        /// and the copies of entries no longer live.
        Pages,
        /// Whole entries with their pages, least recently used first.
        Entries,
        /// The draft's pages, if it follows no request.
        Draft,
    }

    /// Frees pages of the KV pool until `need` are free, from slots outside `keep`, as far
    /// as `reclaim` goes. Returns whether `need` pages are free.
    fn make_room(
        model: &mut Model,
        speculator: &mut Speculator,
        need: usize,
        keep: &[usize],
        reclaim: Reclaim,
    ) -> bool {
        let free = |m: &Model| m.kv_pool().map_or(usize::MAX, PagePool::free_pages);
        if free(model) >= need {
            return true;
        }
        if let Err(e) = speculator.release_dead_entries(model) {
            log!("returning the pages of dropped prefix cache entries failed: {e:#}");
        }
        if free(model) >= need {
            return true;
        }
        for slot in (0..model.slots()).filter(|s| !keep.contains(s)) {
            release(model, speculator, slot);
            if free(model) >= need {
                return true;
            }
        }
        if reclaim < Reclaim::Entries {
            return false;
        }
        for (slot, _) in speculator.evictable_slots(model, keep) {
            // The prefix cache notes the entry it drops, with its pages.
            match speculator.evict_slot(model, slot, need) {
                Ok(()) => log_cache(speculator),
                Err(e) => log!("dropping slot {slot}'s cached prompt failed: {e:#}"),
            }
            if free(model) >= need {
                return true;
            }
        }
        if reclaim < Reclaim::Draft {
            return false;
        }
        match speculator.trim_draft(model) {
            Ok(0) => {}
            Ok(pages) => log!("KV pool: dropped the draft's {pages} pages"),
            Err(e) => log!("dropping the draft's pages failed: {e:#}"),
        }
        free(model) >= need
    }

    /// Seconds a request of `positions` must run before round robin may swap it out:
    /// swapping it out and back, at [`RESTORE_RATE`] each way, takes [`SWAP_SHARE`] of
    /// its turn.
    fn quantum(model: &Model, positions: usize) -> f64 {
        let page = model.kv_pool().map_or(0, PagePool::page_bytes);
        let bytes = (model.pages_for(positions) * page) as f64;
        let swap = 2.0 * bytes / RESTORE_RATE;
        (swap * (1.0 - SWAP_SHARE) / SWAP_SHARE).max(MIN_QUANTUM)
    }

    /// The request of `active` to swap out for room: one still prefilling, the latest to
    /// start first, since its client sees no tokens yet; else the decoding request that
    /// has run longest since it started. Only one that has had its turn ([`quantum`])
    /// unless `force`. Never the request in slot `spare`.
    fn victim(
        model: &Model,
        active: &[Request],
        spare: Option<usize>,
        force: bool,
    ) -> Option<usize> {
        let had_turn =
            |r: &Request| force || r.since.elapsed().as_secs_f64() >= quantum(model, r.position());
        let candidates = || {
            active.iter().enumerate().filter(|&(_, r)| {
                Some(r.slot) != spare && !r.done && r.position() > 0 && had_turn(r)
            })
        };
        candidates()
            .filter(|(_, r)| r.next.is_none())
            .max_by_key(|(_, r)| r.since)
            .or_else(|| candidates().min_by_key(|(_, r)| r.since))
            .map(|(i, _)| i)
    }

    /// Swaps `r` out to the host tier, freeing its slot and its pages. Returns false, and
    /// leaves it running, when there is no host tier or it does not fit there.
    fn park(model: &mut Model, speculator: &mut Speculator, r: &mut Request) -> bool {
        let fed = r.fed();
        let parked = speculator.park(model, r.slot, &fed, r.job.prompt.len());
        log_cache(speculator);
        match parked {
            Ok(Some(id)) => {
                if r.followed {
                    speculator.forget();
                    r.followed = false;
                }
                log!(
                    "round robin: swapped out a request at position {} after {:.2} s",
                    fed.len(),
                    r.since.elapsed().as_secs_f64()
                );
                r.parked = Some(id);
                // An image that a prefill chunk kept for the next one is encoded again
                // when the request returns, rather than held in GPU memory meanwhile.
                r.kept = None;
                r.parks += 1;
                r.since = Instant::now();
                true
            }
            Ok(None) => false,
            Err(e) => {
                // The slot may be half freed: the request ends.
                r.fail(&e);
                false
            }
        }
    }

    /// Swaps requests of `active` out (`victim`, `park`) until `need(active)` pages of
    /// the KV pool are free, after dropping what `make_room` drops. Only requests that
    /// have had their turn unless `force`; never the one in slot `spare`. Returns whether
    /// the room is there.
    fn room_for(
        model: &mut Model,
        speculator: &mut Speculator,
        active: &mut Vec<Request>,
        parked: &mut Vec<Request>,
        spare: Option<usize>,
        force: bool,
        need: impl Fn(&Model, &[Request]) -> usize,
    ) -> bool {
        loop {
            let pages = need(model, active);
            let free_pages = model.kv_pool().map_or(usize::MAX, PagePool::free_pages);
            if pages == 0 || free_pages >= pages {
                return true;
            }
            let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
            if make_room(model, speculator, pages, &busy, Reclaim::Draft) {
                return true;
            }
            let Some(v) = victim(model, active, spare, force) else {
                return false;
            };
            if !park(model, speculator, &mut active[v]) {
                return false;
            }
            parked.push(active.remove(v));
        }
    }

    /// Pages that the requests of `active` may still take beyond what they hold, when they
    /// cannot be swapped out: their every position (`Request::final_position`).
    fn growth(model: &Model, active: &[Request], capacity: usize) -> usize {
        active
            .iter()
            .filter(|r| !r.done)
            .map(|r| {
                model
                    .pages_for(r.final_position(capacity))
                    .saturating_sub(model.slot_pages(r.slot).unwrap_or(0))
            })
            .sum()
    }

    /// Pages that the requests of `active` still prefilling will map for the rest of their
    /// prompts, [`HEADROOM`] more, and their prefix cache entries' copies of the rings.
    fn prefill_growth(model: &Model, active: &[Request], capacity: usize) -> usize {
        active
            .iter()
            .filter(|r| !r.done && r.next.is_none())
            .map(|r| {
                let end = (r.job.prompt.len() + HEADROOM).min(capacity);
                (model.pages_for(end) + model.ring_pages())
                    .saturating_sub(model.slot_pages(r.slot).unwrap_or(0))
            })
            .sum()
    }

    /// What `admit` may start next.
    #[derive(Clone, Copy)]
    enum Pick {
        Waiting(usize),
        Parked(usize),
    }

    /// Starts waiting jobs and swapped-out requests while they fit: a free slot, and room in
    /// the KV pool for their positions and [`HEADROOM`] more, or without a host tier for
    /// every position they may reach. Jobs that waited past [`AGE_LIMIT`] go first, in
    /// arrival order, then swapped-out requests in the order they left, then the other
    /// jobs from the shortest prompt. When one does not fit, running requests that have
    /// had their turn are swapped out for it (round robin); a job past the age limit that
    /// still does not fit stops the others from passing it.
    fn admit(
        model: &mut Model,
        speculator: &mut Speculator,
        active: &mut Vec<Request>,
        waiting: &mut VecDeque<Job>,
        parked: &mut Vec<Request>,
        sched: &Sched,
    ) {
        if waiting.is_empty() && parked.is_empty() {
            return;
        }
        let can_park = speculator.can_park();
        loop {
            waiting.retain(|j| !j.events.is_closed());
            parked.retain_mut(|r| {
                let gone = r.job.events.is_closed() || r.job.cancelled();
                if gone {
                    if let Some(id) = r.parked.take() {
                        speculator.drop_parked(id);
                    }
                    if r.job.cancelled() {
                        r.fail(&anyhow::anyhow!(CANCELLED));
                    }
                }
                !gone
            });
            if (waiting.is_empty() && parked.is_empty())
                || (active.len() >= sched.streams && !can_park)
            {
                return;
            }
            let now = Instant::now();
            let aged = |j: &Job| now.duration_since(j.queued) >= AGE_LIMIT;
            let mut order: Vec<(Pick, bool)> = (0..waiting.len())
                .filter(|&i| aged(&waiting[i]))
                .map(|i| (Pick::Waiting(i), true))
                .collect();
            order.extend((0..parked.len()).map(|i| (Pick::Parked(i), false)));
            let mut young: Vec<usize> =
                (0..waiting.len()).filter(|&i| !aged(&waiting[i])).collect();
            young.sort_by_key(|&i| waiting[i].prompt.len());
            order.extend(young.into_iter().map(|i| (Pick::Waiting(i), false)));
            let mut started = false;
            for (pick, firm) in order {
                let fits = pick_room(model, speculator, active, waiting, parked, pick, sched);
                let fits = fits
                    || (can_park && {
                        let mut room = false;
                        while let Some(v) = victim(model, active, None, false) {
                            if !park(model, speculator, &mut active[v]) {
                                break;
                            }
                            parked.push(active.remove(v));
                            if pick_room(model, speculator, active, waiting, parked, pick, sched) {
                                room = true;
                                break;
                            }
                        }
                        room
                    });
                if fits {
                    start_pick(
                        model,
                        speculator,
                        active,
                        waiting,
                        parked,
                        pick,
                        sched.greedy,
                    );
                    started = true;
                    break;
                }
                if firm {
                    return;
                }
            }
            if !started {
                return;
            }
        }
    }

    /// Whether `pick` fits now, after `make_room`: a free slot, and pages for its positions
    /// and [`HEADROOM`] more and its entry's copies of the rings, beside what the running
    /// requests still prefill, or without a host tier every position they may reach. The
    /// slot it will take keeps its pages: a waiting job's prefix cache slot, or the idle
    /// slot a swapped-out request returns to.
    fn pick_room(
        model: &mut Model,
        speculator: &mut Speculator,
        active: &[Request],
        waiting: &VecDeque<Job>,
        parked: &[Request],
        pick: Pick,
        sched: &Sched,
    ) -> bool {
        if active.len() >= sched.streams {
            return false;
        }
        let can_park = speculator.can_park();
        let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
        let (positions, slot) = match pick {
            Pick::Waiting(i) => {
                let job = &waiting[i];
                let hit = speculator
                    .prefix_lookup(model, job.key(), speculator.prefill_chunk(), &busy)
                    .and_then(|h| h.slot);
                let end = if can_park {
                    job.prompt.len() + HEADROOM
                } else {
                    job.prompt.len() + job.max_tokens + MARGIN
                };
                (end, hit)
            }
            Pick::Parked(i) => (
                parked[i].position() + HEADROOM,
                speculator.return_slot(model, &busy),
            ),
        };
        let positions = positions.min(sched.capacity);
        let held = slot.map_or(0, |s| model.slot_pages(s).unwrap_or(0));
        // Its prefix cache entry copies its rings too.
        let mine = (model.pages_for(positions) + model.ring_pages()).saturating_sub(held);
        let others = if can_park {
            prefill_growth(model, active, sched.capacity)
        } else {
            growth(model, active, sched.capacity)
        };
        let mut keep = busy;
        keep.extend(slot);
        make_room(model, speculator, mine + others, &keep, Reclaim::Draft)
    }

    /// Starts `pick`: a waiting job in the slot the prefix cache gives it, or a swapped-out
    /// request back from the host tier.
    fn start_pick(
        model: &mut Model,
        speculator: &mut Speculator,
        active: &mut Vec<Request>,
        waiting: &mut VecDeque<Job>,
        parked: &mut Vec<Request>,
        pick: Pick,
        greedy: bool,
    ) {
        let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
        match pick {
            Pick::Waiting(i) => {
                let Some(job) = waiting.remove(i) else { return };
                let mut request = Request::new(job, 0, greedy);
                if request.done {
                    return;
                }
                let key = request.job.key();
                let claimed = speculator.claim_slot(model, key, speculator.prefill_chunk(), &busy);
                log_cache(speculator);
                match claimed {
                    Ok((slot, cached)) => {
                        request.slot = slot;
                        request.prefilled = cached;
                        request.draft_ready = speculator.supports_batch_spec()
                            && (cached == 0 || speculator.slot_draft_committed(slot) == cached);
                        request.cached_from(speculator, cached);
                        active.push(request);
                    }
                    Err(e) => request.fail(&e),
                }
            }
            Pick::Parked(i) => {
                let mut r = parked.remove(i);
                let Some(id) = r.parked.take() else { return };
                let back = speculator.unpark(model, id, &busy);
                log_cache(speculator);
                match back {
                    Ok((slot, position)) if position == r.position() => {
                        r.parked_s += r.since.elapsed().as_secs_f64();
                        log!(
                            "round robin: a request at position {position} returned to slot \
                             {slot} after {:.2} s",
                            r.since.elapsed().as_secs_f64()
                        );
                        r.slot = slot;
                        r.draft_ready = false;
                        r.since = Instant::now();
                        active.push(r);
                    }
                    Ok((_, position)) => r.fail(&anyhow::anyhow!(
                        "a swapped-out request came back at position {position}, not {}",
                        r.position()
                    )),
                    Err(e) => r.fail(&e),
                }
            }
        }
    }

    /// Runs `request` alone in slot 0 with speculation: from its prompt, or from its next
    /// token when it returns from a batch (`Speculator::resume`). With `interrupt`, stops
    /// after the round in which another job arrives, moves that job to the waiting list
    /// and returns true: the request then continues in a batch from slot 0. Otherwise
    /// finishes it.
    ///
    /// The reply takes pages of the KV pool as it grows, [`HEADROOM`] positions at a time,
    /// from other slots' prefix cache entries if need be (`make_room`); slot 0's entry and
    /// the draft keep theirs. A reply that the pool cannot hold further, after every other
    /// entry has gone, ends there with [`Finish::Length`], as at its token limit.
    fn alone(
        model: &mut Model,
        speculator: &mut Speculator,
        tower: &mut Tower,
        request: &mut Request,
        stop: &[u32],
        capacity: usize,
        mut interrupt: Option<(&Receiver<Vec<Job>>, &mut VecDeque<Job>)>,
    ) -> anyhow::Result<bool> {
        model.set_sampling(request.sampling);
        let go_on = if request.next.is_some() {
            true
        } else {
            let hidden = model.config().hidden_size;
            let chunk = speculator.prefill_chunk();
            let job = &request.job;
            let encodes = &mut request.encodes;
            let mut kept = None;
            let mut source = |start: usize, end: usize| {
                image_rows(tower, hidden, &job.images, &mut kept, encodes, (start, end))
            };
            let prompt = Prompt {
                tokens: &job.prompt,
                key: job.key(),
                images: if job.images.is_empty() {
                    None
                } else {
                    Some(&mut source)
                },
            };
            let (first, cached) = speculator.prefill_cached(model, prompt, chunk)?;
            log_cache(speculator);
            request.cached_from(speculator, cached);
            request.prefilled = request.job.prompt.len();
            request.prefill_s = request.started.elapsed().as_secs_f64();
            request.send_images();
            request.next = Some(first);
            request.take(&[first], stop).is_continue()
        };
        let mut interrupted = false;
        let mut last = request.next.unwrap_or_default();
        let before = request.output.len();
        let started = Instant::now();
        if go_on {
            let remaining = request.job.max_tokens - request.output.len();
            let final_position = request.final_position(capacity);
            // Positions below which slot 0 and the draft have their pages, mapped or free.
            let mut held = 0;
            let room = |model: &mut Model, speculator: &mut Speculator, end: usize| {
                if end <= held {
                    return true;
                }
                // The draft maps a few positions past the model's writes.
                let need = |m: &Model, s: &Speculator, to: usize| {
                    m.slot_pages_to(0, to).unwrap_or(0) + s.draft_pages_to(to + MARGIN)
                };
                for to in [(end + HEADROOM).min(final_position).max(end), end] {
                    let pages = need(model, speculator, to);
                    if make_room(model, speculator, pages, &[0], Reclaim::Entries) {
                        let _ = model.map_slot(0, to);
                        held = to;
                        return true;
                    }
                }
                // The reply ends here, still with `Finish::Length`.
                log!("KV pool: full at position {}", model.position());
                false
            };
            speculator.generate_with_room(model, remaining, Drafts::Dspark, room, |ids| {
                request.speculated += ids.len();
                let _ = request.job.events.send(Event::Round);
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
            log!("{episode}, then batched from position {}", model.position());
            request.next = Some(last);
            return Ok(true);
        }
        request.finish(&format!(", last {episode}"));
        Ok(false)
    }

    /// Logs what the prefix cache and its host tier did since the last call: saves,
    /// restores and dropped entries.
    fn log_cache(speculator: &mut Speculator) {
        for line in speculator.cache_notes() {
            log!("{line}");
        }
    }

    /// Positions that a draft rebuild prefills in the time of one batched step of a lone
    /// request, about: an RTX 5090 prefills about 20,000 positions per second and runs a
    /// one-row batched step in about 3.6 ms, 72 positions; an RTX 4090 prefilled 72,000
    /// positions in 9.4 s and stepped in 7.1 ms, 55 positions.
    const RENT_RATIO: usize = 64;

    /// A draft rebuild in progress, and when it started.
    type Rebuilding = Option<(Rebuild, Instant)>;

    /// How long the server waits idle before it starts a draft rebuild (`idle_rebuild`):
    /// agents often send their next request right after a reply, which would otherwise
    /// wait for the rebuild's step in flight.
    const IDLE_DELAY: std::time::Duration = std::time::Duration::from_millis(500);

    /// Moves a lone request from a batch back to speculation in slot 0, when the draft
    /// follows it: commits the taps the draft kept while following it. Returns false when
    /// the draft does not follow it.
    fn speculate_again(
        model: &mut Model,
        speculator: &mut Speculator,
        r: &mut Request,
    ) -> anyhow::Result<bool> {
        if r.draft_ready && speculator.supports_batch_spec() {
            let started = Instant::now();
            let from = r.slot;
            let position = model.slot_position(r.slot)?;
            let next = r.next.context("the request has no next token")?;
            speculator.resume_dflash2_slot(model, r.slot, next)?;
            r.slot = 0;
            model.stream().synchronize()?;
            log!(
                "speculating again (DFlash 2) at position {position} from slot {from}, {:.1} ms",
                started.elapsed().as_secs_f64() * 1e3
            );
            r.followed = false;
            return Ok(true);
        }
        if !r.followed {
            return Ok(false);
        }
        let started = Instant::now();
        let from = r.slot;
        let position = model.slot_position(r.slot)?;
        let next = r.next.context("the request has no next token")?;
        speculator.move_to_slot0(model, r.slot)?;
        r.slot = 0;
        speculator.resume(model, next)?;
        model.stream().synchronize()?;
        log!(
            "speculating again at position {position} from slot {from}, {:.1} ms",
            started.elapsed().as_secs_f64() * 1e3
        );
        r.followed = false;
        Ok(true)
    }

    /// Runs the next step of `rebuild` for a sequence whose positions took `prompt`, after
    /// making room for it beside slots `keep` as far as `reclaim` goes, short of the
    /// draft's pages. Drops
    /// the rebuild when there is no room or the step fails. Returns whether the draft now
    /// holds every position of `prompt`, `None` when no step ran.
    fn step_rebuild(
        model: &mut Model,
        speculator: &mut Speculator,
        rebuild: &mut Rebuilding,
        prompt: Prompt<'_>,
        keep: &[usize],
        reclaim: Reclaim,
    ) -> Option<bool> {
        let (r, _) = rebuild.as_mut()?;
        let mut keep = keep.to_vec();
        keep.push(r.slot());
        let need = speculator.rebuild_pages(model, r, prompt.key);
        if !make_room(
            model,
            speculator,
            need,
            &keep,
            reclaim.min(Reclaim::Entries),
        ) {
            log!("no room in the KV pool to rebuild the draft");
            *rebuild = None;
            return None;
        }
        let (r, _) = rebuild.as_mut()?;
        let stepped = speculator.rebuild_step(model, r, prompt);
        log_cache(speculator);
        match stepped {
            Ok(done) => Some(done),
            Err(e) => {
                log!("rebuilding the draft failed: {e:#}");
                *rebuild = None;
                None
            }
        }
    }

    /// Drops `rebuild` unless it can go on for a sequence whose positions took `key`.
    fn keep_rebuild(speculator: &Speculator, model: &Model, rebuild: &mut Rebuilding, key: &[u32]) {
        if rebuild
            .as_ref()
            .is_some_and(|(r, _)| !speculator.rebuild_live(model, r) || !key.starts_with(r.key()))
        {
            *rebuild = None;
        }
    }

    /// On an idle server, runs a step of a rebuild that gives the draft's state to the
    /// longest prefix cache entry, when it lacks it (`Speculator::entry_to_rebuild`), in a
    /// slot without an entry: its conversation's next turn then speculates from the cache
    /// rather than decoding batched. A rebuild that a request interrupts goes on while
    /// that request, or the next idle time, continues the same tokens. Returns whether a
    /// step ran.
    fn idle_rebuild(
        model: &mut Model,
        speculator: &mut Speculator,
        rebuild: &mut Rebuilding,
    ) -> bool {
        let vocab = model.config().vocab_size;
        let Some((slot, key)) = speculator.entry_to_rebuild(model, vocab) else {
            *rebuild = None;
            return false;
        };
        keep_rebuild(speculator, model, rebuild, &key);
        if rebuild.is_none() {
            let Some(scratch) = speculator.empty_slot(model, &[slot]) else {
                return false;
            };
            match speculator.start_rebuild(model, scratch) {
                Ok(r) => *rebuild = Some((r, Instant::now())),
                Err(e) => {
                    log!("starting a draft rebuild failed: {e:#}");
                    return false;
                }
            }
            log!(
                "idle: rebuilding the draft's state for the cached prompt of {} tokens in slot \
                 {slot}, in slot {scratch}",
                key.len()
            );
        }
        let prompt = Prompt::text(&key);
        match step_rebuild(model, speculator, rebuild, prompt, &[slot], Reclaim::Pages) {
            Some(true) => {
                if let Some((_, started)) = rebuild.take() {
                    log!(
                        "idle: the cached prompt of {} tokens has the draft's state, {:.2} s",
                        key.len(),
                        started.elapsed().as_secs_f64()
                    );
                }
                true
            }
            Some(false) => true,
            None => false,
        }
    }

    /// Runs a step of a draft rebuild for the lone request of `active`, which decodes
    /// batched without the draft's state, so that it returns to speculation once the
    /// draft holds its every position (`followed`, then `speculate_again`). Its reply may
    /// end at any point, whatever its token limit, so it rents before it buys: the rebuild
    /// starts, or goes on, once the request's batched steps as the only row could have
    /// prefilled the positions the rebuild still lacks ([`RENT_RATIO`]), which costs it at
    /// most about twice the time of the better choice. The steps then run between batched
    /// steps as prefill chunks do (`floor_steps`), and a request that arrives meanwhile
    /// waits for one step at most. The rebuild also gives the request's prefix cache
    /// entry the draft's state. Returns whether a step ran.
    fn rebuild_turn(
        model: &mut Model,
        speculator: &mut Speculator,
        tower: &mut Tower,
        active: &mut [Request],
        rebuild: &mut Rebuilding,
    ) -> bool {
        let [r] = active else {
            return false;
        };
        if r.next.is_none() || r.done || r.followed || r.stay_batched {
            return false;
        }
        let generated = &r.output[..r.output.len() - 1];
        let base_key = r.job.key();
        let key_len = base_key.len() + generated.len();
        if rebuild.as_ref().is_some_and(|(rb, _)| {
            !speculator.rebuild_live(model, rb)
                || !starts_with_chained(base_key, generated, rb.key())
        }) {
            *rebuild = None;
        }
        let done = rebuild.as_ref().map_or(0, |(rb, _)| rb.key().len());
        if r.rent * RENT_RATIO < key_len - done {
            return false;
        }
        if rebuild.is_none() {
            if speculator.following() {
                return false;
            }
            let Some(scratch) = speculator.idle_slot(model, &[r.slot]) else {
                return false;
            };
            match speculator.start_rebuild(model, scratch) {
                Ok(rb) => *rebuild = Some((rb, Instant::now())),
                Err(e) => {
                    log!("starting a draft rebuild failed: {e:#}");
                    return false;
                }
            }
            log!(
                "rebuilding the draft at position {key_len} of a request in slot {} after {} \
                 batched steps, in slot {scratch}",
                r.slot,
                r.rent
            );
        }
        let mut tokens = Vec::with_capacity(r.job.prompt.len() + generated.len());
        tokens.extend_from_slice(&r.job.prompt);
        tokens.extend_from_slice(generated);
        let owned_key = r.job.key.as_deref().map(|k| {
            let mut buf = Vec::with_capacity(k.len() + generated.len());
            buf.extend_from_slice(k);
            buf.extend_from_slice(generated);
            buf
        });
        let key_slice: &[u32] = owned_key.as_deref().unwrap_or(&tokens);
        let hidden = model.config().hidden_size;
        let images = &r.job.images;
        let mut kept = None;
        // The draft's rebuild is not the request's prefill, so its encodes are not
        // reported.
        let mut encodes = Encodes::default();
        let mut source = |start: usize, end: usize| {
            image_rows(tower, hidden, images, &mut kept, &mut encodes, (start, end))
        };
        let prompt = Prompt {
            tokens: &tokens,
            key: key_slice,
            images: if images.is_empty() {
                None
            } else {
                Some(&mut source)
            },
        };
        match step_rebuild(
            model,
            speculator,
            rebuild,
            prompt,
            &[r.slot],
            Reclaim::Entries,
        ) {
            Some(true) => {
                let Some((rb, started)) = rebuild.take() else {
                    return true;
                };
                match speculator.follow_rebuilt(model, rb, &tokens) {
                    Ok(()) => {
                        log!(
                            "rebuilt the draft at position {} in {:.2} s",
                            tokens.len(),
                            started.elapsed().as_secs_f64()
                        );
                        r.followed = true;
                    }
                    Err(e) => log!("following a rebuilt request failed: {e:#}"),
                }
                true
            }
            Some(false) => true,
            // Without room or after a failure, the request stays batched rather than try
            // again after every step.
            None => {
                r.stay_batched = true;
                false
            }
        }
    }

    /// Runs `active` and the jobs that arrive while any runs, up to `streams` at once, each
    /// in its own slot, until none is left or one is left alone. Starts every waiting job
    /// that fits the KV pool (`admit`), swapping running requests out to the host tier and
    /// back in turns when they do not fit together. Each turn prefills one chunk of a
    /// request still prefilling, the one with the fewest prompt tokens left, then runs
    /// batched steps of the others: one, or after a chunk as many as give each decoding
    /// request `floor` tokens per second of the time since the last step (`floor_steps`).
    /// A request left alone with nothing waiting returns, in slot 0, to continue with
    /// speculation (`speculate_again`). Each request takes its slot from the prefix cache
    /// (`Speculator::claim_slot`), and prefills from where its entry ends.
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "the scheduler's state and loop"
    )]
    fn batched(
        model: &mut Model,
        speculator: &mut Speculator,
        tower: &mut Tower,
        mut active: Vec<Request>,
        waiting: &mut VecDeque<Job>,
        queue: &Receiver<Vec<Job>>,
        stop: &[u32],
        sched: &Sched,
        rebuild: &mut Rebuilding,
    ) -> Option<Request> {
        // Requests swapped out to the host tier, to return when they fit.
        let mut parked: Vec<Request> = Vec::new();
        // Reusable buffers for `step_rows`, eliminating per-step Vec allocations.
        let mut scratch = StepScratch::with_capacity(sched.streams);
        // When the last batched step synced: the decoding requests have waited since then.
        let mut synced = Instant::now();
        loop {
            while let Ok(jobs) = queue.try_recv() {
                waiting.extend(jobs);
            }
            admit(model, speculator, &mut active, waiting, &mut parked, sched);
            if active.is_empty() {
                if waiting.is_empty() && parked.is_empty() {
                    return None;
                }
                // Nothing fits an empty pool: the first in line fails rather than wait.
                if let Some(mut r) = parked.pop() {
                    if let Some(id) = r.parked.take() {
                        speculator.drop_parked(id);
                    }
                    r.fail(&anyhow::anyhow!("the request no longer fits the KV pool"));
                } else if let Some(job) = waiting.pop_front() {
                    let mut r = Request::new(job, 0, sched.greedy);
                    r.fail(&anyhow::anyhow!("the request does not fit the KV pool"));
                }
                continue;
            }
            // When piggybacking is enabled and both decoders and prefillers are
            // active, carve a bounded prefill chunk to ride along with the active decode or
            // batched-verify forward pass instead of stalling active streams behind an
            // exclusive prefill step.
            if sched.piggyback_budget.is_some()
                && active.iter().any(|r| !r.done && r.next.is_some())
                && active.iter().any(|r| !r.done && r.next.is_none())
            {
                for r in &mut active {
                    if r.next.is_some() || r.done {
                        continue;
                    }
                    if r.job.events.is_closed() {
                        r.done = true;
                    } else if r.job.cancelled() {
                        r.fail(&anyhow::anyhow!(CANCELLED));
                    }
                }
                retire(model, speculator, &mut active);
                let batch_spec = speculator.supports_batch_spec();
                let next_pages = |m: &Model, r: &Request| {
                    let advance = if batch_spec && r.draft_ready {
                        VERIFY_ROWS
                    } else {
                        1
                    };
                    m.slot_pages_to(r.slot, r.position() + advance).unwrap_or(0)
                };
                while !room_for(
                    model,
                    speculator,
                    &mut active,
                    &mut parked,
                    None,
                    true,
                    |m, a| {
                        a.iter()
                            .filter(|r| !r.done && r.next.is_some())
                            .map(|r| next_pages(m, r))
                            .sum()
                    },
                ) {
                    let Some(r) = active
                        .iter_mut()
                        .filter(|r| !r.done && r.next.is_some() && next_pages(model, r) > 0)
                        .max_by_key(|r| r.since)
                    else {
                        break;
                    };
                    r.fail(&anyhow::anyhow!("the KV pool is full"));
                }
                retire(model, speculator, &mut active);
                let spec_count = active
                    .iter()
                    .filter(|r| !r.done && r.next.is_some() && batch_spec && r.draft_ready)
                    .count();
                let plain_count = active
                    .iter()
                    .filter(|r| !r.done && r.next.is_some() && !(batch_spec && r.draft_ready))
                    .count();
                let carrier_rows = if spec_count > 0 {
                    let chunk_len = spec_count.min(speculator.max_batch_spec_seqs());
                    let max_w = (BATCH_VERIFY_MAX_ROWS / chunk_len).min(VERIFY_ROWS);
                    chunk_len * max_w
                } else {
                    plain_count
                };
                let now = Instant::now();
                let mut queued_order: Vec<usize> = (0..active.len()).collect();
                queued_order.sort_by_key(|&i| active[i].job.queued);
                let mut arrival_ranks = vec![0usize; active.len()];
                for (rank, &idx) in queued_order.iter().enumerate() {
                    arrival_ranks[idx] = rank;
                }
                let candidates: Vec<PrefillCandidate> = active
                    .iter()
                    .enumerate()
                    .map(|(i, r)| PrefillCandidate {
                        index: i,
                        slot: r.slot,
                        group: r.job.group,
                        prefilled: r.prefilled,
                        prompt_len: r.job.prompt.len(),
                        aged: now.duration_since(r.job.queued) >= AGE_LIMIT,
                        arrival_rank: arrival_ranks[i],
                        is_prefilling: r.next.is_none(),
                        alive: !r.done,
                    })
                    .collect();
                let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
                let mut pages_needed: usize = active
                    .iter()
                    .filter(|r| !r.done && r.next.is_some())
                    .map(|r| next_pages(model, r))
                    .sum();
                let plan = piggyback::plan_piggyback(
                    &candidates,
                    carrier_rows,
                    speculator.prefill_chunk(),
                    sched.piggyback_budget,
                    BATCH_ROWS,
                    |c, end| {
                        let extra = model.slot_pages_to(c.slot, end).unwrap_or(0);
                        if make_room(
                            model,
                            speculator,
                            pages_needed + extra,
                            &busy,
                            Reclaim::Draft,
                        ) {
                            pages_needed += extra;
                            true
                        } else {
                            false
                        }
                    },
                );
                if let TurnPlan::Piggyback { slices, .. } = plan {
                    step_rows_hybrid(
                        model,
                        speculator,
                        tower,
                        &mut active,
                        &slices,
                        stop,
                        sched.group_by_confidence,
                        &mut scratch,
                    );
                    retire(model, speculator, &mut active);
                    synced = Instant::now();
                    continue;
                }
            }
            // Drain short prefills up to one full prefill-chunk token budget per turn so a
            // burst of short prompts doesn't stall behind per-request decode rounds and graph
            // size churn.
            let prefill_budget = speculator.prefill_chunk();
            let mut prefilled_tokens = 0usize;
            loop {
                let n = prefill_chunk(model, speculator, tower, &mut active, &mut parked, stop);
                if n == 0 {
                    break;
                }
                prefilled_tokens += n;
                retire(model, speculator, &mut active);
                if prefilled_tokens >= prefill_budget {
                    break;
                }
                while let Ok(jobs) = queue.try_recv() {
                    waiting.extend(jobs);
                }
                admit(model, speculator, &mut active, waiting, &mut parked, sched);
            }
            let chunked = prefilled_tokens > 0;
            // Drain newly arrived jobs before deciding we are alone, so a request that
            // arrived during prefill avoids a wasted `speculate_again` (`move_to_slot0` +
            // GPU sync) round-trip.
            if active.len() == 1 && waiting.is_empty() && parked.is_empty() {
                let mut arrived = false;
                while let Ok(jobs) = queue.try_recv() {
                    waiting.extend(jobs);
                    arrived = true;
                }
                if arrived {
                    admit(model, speculator, &mut active, waiting, &mut parked, sched);
                }
            }
            let alone = waiting.is_empty() && parked.is_empty();
            let chunked =
                chunked || (alone && rebuild_turn(model, speculator, tower, &mut active, rebuild));
            // A lone request that the draft follows speculates before any batched step: right
            // after its prefill, as it would have from the start, or once its rebuild ends.
            if let [r] = active.as_mut_slice()
                && r.next.is_some()
                && !r.done
                && !r.stay_batched
                && alone
            {
                match speculate_again(model, speculator, r) {
                    Ok(true) => return active.pop(),
                    Ok(false) => {}
                    Err(e) => {
                        log!("returning to speculation failed, staying batched: {e:#}");
                        speculator.forget();
                        r.followed = false;
                        r.draft_ready = false;
                        r.stay_batched = true;
                    }
                }
            }
            let first_step = Instant::now();
            let mut steps = 0;
            loop {
                // Room for every row's next position, swapping requests out if need be. When
                // none can go to the host tier, the latest to start of those that need pages
                // fail until the rest fit.
                let batch_spec = speculator.supports_batch_spec();
                let next_pages = |m: &Model, r: &Request| {
                    let advance = if batch_spec && r.draft_ready {
                        VERIFY_ROWS
                    } else {
                        1
                    };
                    m.slot_pages_to(r.slot, r.position() + advance).unwrap_or(0)
                };
                while !room_for(
                    model,
                    speculator,
                    &mut active,
                    &mut parked,
                    None,
                    true,
                    |m, a| {
                        a.iter()
                            .filter(|r| !r.done && r.next.is_some())
                            .map(|r| next_pages(m, r))
                            .sum()
                    },
                ) {
                    let Some(r) = active
                        .iter_mut()
                        .filter(|r| !r.done && r.next.is_some() && next_pages(model, r) > 0)
                        .max_by_key(|r| r.since)
                    else {
                        break;
                    };
                    r.fail(&anyhow::anyhow!("the KV pool is full"));
                }
                // Pre-map ahead in bulk for any decoding request about to cross a KV chunk
                // boundary, avoiding repeated single-page `cuMemSetAccess` stalls mid-decode.
                let total_immediate: usize = active
                    .iter()
                    .filter(|r| !r.done && r.next.is_some())
                    .map(|r| next_pages(model, r))
                    .sum();
                if total_immediate > 0 {
                    let mut remaining_immediate = total_immediate;
                    for r in active.iter().filter(|r| !r.done && r.next.is_some()) {
                        let need = next_pages(model, r);
                        if need > 0 {
                            remaining_immediate = remaining_immediate.saturating_sub(need);
                            let advance = if batch_spec && r.draft_ready {
                                VERIFY_ROWS
                            } else {
                                1
                            };
                            premap_slot_headroom(
                                model,
                                r.slot,
                                r.position() + advance,
                                r.final_position(sched.capacity),
                                remaining_immediate,
                            );
                        }
                    }
                }
                let emitted = step_rows(
                    model,
                    speculator,
                    &mut active,
                    stop,
                    sched.group_by_confidence,
                    &mut scratch,
                );
                if emitted == 0 {
                    break;
                }
                retire(model, speculator, &mut active);
                steps += emitted;
                if !chunked || !floor_steps(sched.floor, steps, synced, first_step) {
                    break;
                }
            }
            retire(model, speculator, &mut active);
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
    /// a failure. Their slots keep their pages until the KV pool runs short (`make_room`),
    /// so that the next request in a slot maps none.
    fn retire(model: &mut Model, speculator: &mut Speculator, active: &mut Vec<Request>) {
        active.retain(|r| {
            if r.done {
                if r.followed {
                    speculator.forget();
                }
                if !r.failed && r.next.is_some() {
                    r.finish("");
                }
                if r.failed {
                    release(model, speculator, r.slot);
                }
            }
            !r.done
        });
    }

    /// Prefills one chunk — packing multiple short prompts up to `prefill_chunk` total rows
    /// when possible, or one chunk of the highest-priority request otherwise — and keeps
    /// prompt states in the prefix cache at chunk boundaries. Returns the total prompt tokens
    /// processed (0 if none).
    #[expect(
        clippy::too_many_lines,
        reason = "packed and single prefill chunk with taps, images, and prefix cache"
    )]
    fn prefill_chunk(
        model: &mut Model,
        speculator: &mut Speculator,
        tower: &mut Tower,
        active: &mut Vec<Request>,
        parked: &mut Vec<Request>,
        stop: &[u32],
    ) -> usize {
        if !active.iter().any(|r| r.next.is_none() && !r.done) {
            return 0;
        }
        // Priority order: aged requests first (by arrival), then shortest remaining prompt.
        let mut order: Vec<usize> = active
            .iter()
            .enumerate()
            .filter(|(_, r)| r.next.is_none() && !r.done)
            .map(|(i, _)| i)
            .collect();
        let now = Instant::now();
        order.sort_by_key(|&i| {
            let r = &active[i];
            let aged = now.duration_since(r.job.queued) >= AGE_LIMIT;
            if aged {
                (0u8, Some(r.job.queued), 0usize)
            } else {
                (1u8, None, r.job.prompt.len() - r.prefilled)
            }
        });
        let first_slot = active[order[0]].slot;
        let chunk = speculator.prefill_chunk();
        let want = |m: &Model, r: &Request| {
            let end = (r.prefilled + chunk).min(r.job.prompt.len());
            m.slot_pages_to(r.slot, end).unwrap_or(0)
        };
        // Ensure KV pool room for the highest-priority request (parking others if needed).
        let room = room_for(
            model,
            speculator,
            active,
            parked,
            Some(first_slot),
            true,
            |m, a| {
                a.iter()
                    .find(|r| r.slot == first_slot)
                    .map_or(0, |r| want(m, r))
            },
        );
        let Some(i0) = active.iter().position(|r| r.slot == first_slot) else {
            return 0;
        };
        if !room {
            active[i0].fail(&anyhow::anyhow!("the KV pool is full"));
            return 1;
        }
        // Drop closed/cancelled requests before deciding whether to pack.
        for r in active.iter_mut() {
            if r.next.is_some() || r.done {
                continue;
            }
            if r.job.events.is_closed() {
                r.done = true;
            } else if r.job.cancelled() {
                r.fail(&anyhow::anyhow!(CANCELLED));
            }
        }
        let Some(i0) = active
            .iter()
            .position(|r| r.slot == first_slot && r.next.is_none() && !r.done)
        else {
            return 0;
        };

        // Siblings of the same `job.group` share one prefill via `share_prefill` + `redraw_slot`,
        // so exclude requests with unprefilled siblings from multi-prompt packing.
        let has_unprefilled_sibling = |active: &[Request], idx: usize| {
            let (slot, group) = (active[idx].slot, active[idx].job.group);
            active.iter().any(|other| {
                other.job.group == group
                    && other.slot != slot
                    && !other.done
                    && other.next.is_none()
                    && other.prefilled == 0
            })
        };

        // Greedily pack additional short prefilling requests into the same `chunk`-row pass.
        let first_rem = active[i0].job.prompt.len() - active[i0].prefilled;
        let mut packed_slots = vec![first_slot];
        let mut packed_rows = first_rem.min(chunk);
        if first_rem < chunk && !has_unprefilled_sibling(active, i0) {
            let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
            let mut candidates: Vec<usize> = active
                .iter()
                .enumerate()
                .filter(|&(idx, r)| {
                    idx != i0
                        && r.next.is_none()
                        && !r.done
                        && !has_unprefilled_sibling(active, idx)
                        && (r.job.prompt.len() - r.prefilled) <= chunk - packed_rows
                })
                .map(|(idx, _)| idx)
                .collect();
            candidates.sort_by_key(|&idx| {
                let r = &active[idx];
                let aged = now.duration_since(r.job.queued) >= AGE_LIMIT;
                if aged {
                    (0u8, Some(r.job.queued), 0usize)
                } else {
                    (1u8, None, r.job.prompt.len() - r.prefilled)
                }
            });
            let mut pages_needed = want(model, &active[i0]);
            for idx in candidates {
                let rem = active[idx].job.prompt.len() - active[idx].prefilled;
                if packed_rows + rem > chunk || packed_slots.len() >= siphon_cuda::model::BATCH_ROWS
                {
                    continue;
                }
                let extra_pages = want(model, &active[idx]);
                if extra_pages == 0
                    || make_room(
                        model,
                        speculator,
                        pages_needed + extra_pages,
                        &busy,
                        Reclaim::Draft,
                    )
                {
                    pages_needed += extra_pages;
                    packed_rows += rem;
                    packed_slots.push(active[idx].slot);
                }
            }
        }

        if packed_slots.len() >= 2 {
            let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
            let hidden = model.config().hidden_size;
            let batch_spec = speculator.supports_batch_spec();
            let indices: Vec<usize> = packed_slots
                .iter()
                .filter_map(|&slot| active.iter().position(|r| r.slot == slot))
                .collect();
            // Encode any overlapping images before borrowing `active` immutably for slices.
            let mut encoded_images: Vec<Vec<ImageRows>> = Vec::with_capacity(indices.len());
            for &idx in &indices {
                let r = &mut active[idx];
                let start = r.prefilled;
                let end = r.job.prompt.len();
                let imgs = if r.job.images.is_empty() {
                    Vec::new()
                } else {
                    match image_rows(
                        tower,
                        hidden,
                        &r.job.images,
                        &mut r.kept,
                        &mut r.encodes,
                        (start, end),
                    ) {
                        Ok(imgs) => imgs,
                        Err(e) => {
                            r.fail(&e);
                            return 1;
                        }
                    }
                };
                encoded_images.push(imgs);
            }
            let capacity = model.slot_capacity();
            let mut commit_draft = Vec::with_capacity(indices.len());
            let mut reset_draft = Vec::with_capacity(indices.len());
            let mut slices = Vec::with_capacity(indices.len());
            let mut ends = Vec::with_capacity(indices.len());
            let mut total_processed = 0usize;
            for (&idx, imgs) in indices.iter().zip(encoded_images) {
                let r = &active[idx];
                let len = r.job.prompt.len();
                let end = (r.prefilled + chunk).min(len);
                if end == len {
                    // Pre-map prompt + decode headroom in one coalesced pass before prefill
                    // so the subsequent decode rounds hit zero VMM mapping stalls, while
                    // reserving pages for all other requests still prefilling.
                    let other_reserve: usize = active
                        .iter()
                        .filter(|o| o.slot != r.slot && !o.done && o.next.is_none())
                        .map(|o| {
                            let o_end = (o.job.prompt.len() + HEADROOM).min(capacity);
                            (model.pages_for(o_end) + model.ring_pages())
                                .saturating_sub(model.slot_pages(o.slot).unwrap_or(0))
                        })
                        .sum();
                    premap_slot_headroom(
                        model,
                        r.slot,
                        end,
                        r.final_position(capacity),
                        other_reserve,
                    );
                }
                total_processed += end - r.prefilled;
                ends.push((idx, end, len));
                commit_draft.push(batch_spec && r.draft_ready);
                reset_draft.push(r.prefilled == 0);
                slices.push(PackedPrefill {
                    slot: r.slot,
                    tokens: &r.job.prompt[r.prefilled..end],
                    images: imgs,
                    need_first_token: end == len,
                    sampling: r.sampling,
                });
            }
            let results =
                speculator.prefill_packed_batch(model, &slices, &commit_draft, &reset_draft, chunk);
            drop(slices);
            match results {
                Ok(tokens) => {
                    for ((idx, end, len), first) in ends.into_iter().zip(tokens) {
                        let boundary = len.saturating_sub(1) / chunk * chunk;
                        let slot = active[idx].slot;
                        active[idx].prefilled = end;
                        if end == boundary && end > active[idx].cached {
                            if make_room(
                                model,
                                speculator,
                                model.ring_pages(),
                                &busy,
                                Reclaim::Draft,
                            ) {
                                let key = active[idx].job.key();
                                if let Err(e) =
                                    speculator.save_prefix(model, slot, key, chunk, &busy)
                                {
                                    log!("keeping a prompt in the prefix cache failed: {e:#}");
                                }
                            } else {
                                log!("no room in the KV pool to keep a prompt in the prefix cache");
                            }
                        }
                        if end == len
                            && let Some(first) = first
                        {
                            let r = &mut active[idx];
                            r.kept = None;
                            r.prefill_s = r.started.elapsed().as_secs_f64();
                            r.send_images();
                            r.next = Some(first);
                            r.done = r.take(&[first], stop).is_break();
                        }
                    }
                    log_cache(speculator);
                    return total_processed.max(1);
                }
                Err(e) => {
                    for &idx in &indices {
                        active[idx].fail(&e);
                    }
                    return total_processed.max(1);
                }
            }
        }

        let i = i0;
        let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
        let capacity = model.slot_capacity();
        let slot_i = active[i].slot;
        let other_reserve: usize = active
            .iter()
            .filter(|o| o.slot != slot_i && !o.done && o.next.is_none())
            .map(|o| {
                let o_end = (o.job.prompt.len() + HEADROOM).min(capacity);
                (model.pages_for(o_end) + model.ring_pages())
                    .saturating_sub(model.slot_pages(o.slot).unwrap_or(0))
            })
            .sum();
        let r = &mut active[i];
        let len = r.job.prompt.len();
        let end = (r.prefilled + chunk).min(len);
        if end == len {
            premap_slot_headroom(
                model,
                r.slot,
                end,
                r.final_position(capacity),
                other_reserve,
            );
        }
        let processed = end - r.prefilled;
        let boundary = len.saturating_sub(1) / chunk * chunk;
        model.set_sampling(r.sampling);
        let tokens = &r.job.prompt[r.prefilled..end];
        let use_draft = r.draft_ready && speculator.supports_batch_spec();
        let reset_draft = r.prefilled == 0;
        let prefilled = if use_draft {
            if r.job.images.is_empty() {
                speculator.prefill_slot_batch(model, r.slot, tokens, None, chunk, reset_draft)
            } else {
                let hidden = model.config().hidden_size;
                let (images, kept, encodes) = (&r.job.images, &mut r.kept, &mut r.encodes);
                let mut source = |start: usize, end: usize| {
                    image_rows(tower, hidden, images, kept, encodes, (start, end))
                };
                speculator.prefill_slot_batch(
                    model,
                    r.slot,
                    tokens,
                    Some(&mut source),
                    chunk,
                    reset_draft,
                )
            }
        } else if r.job.images.is_empty() {
            model.prefill_slot(r.slot, tokens, chunk)
        } else {
            // The tower runs here, inside the chunk, so the decode floor counts it as
            // prefill.
            let hidden = model.config().hidden_size;
            let (images, kept, encodes) = (&r.job.images, &mut r.kept, &mut r.encodes);
            let mut source = |start: usize, end: usize| {
                image_rows(tower, hidden, images, kept, encodes, (start, end))
            };
            model.prefill_slot_with_images(r.slot, tokens, chunk, &mut source)
        };
        let first = match prefilled {
            Ok(first) => first,
            Err(e) => {
                r.fail(&e);
                return processed.max(1);
            }
        };
        r.prefilled = end;
        let slot = r.slot;
        // The entry copies the slot's rings into pages of the pool.
        if end == boundary && end > r.cached {
            if make_room(model, speculator, model.ring_pages(), &busy, Reclaim::Draft) {
                let key = active[i].job.key();
                if let Err(e) = speculator.save_prefix(model, slot, key, chunk, &busy) {
                    log!("keeping a prompt in the prefix cache failed: {e:#}");
                }
            } else {
                log!("no room in the KV pool to keep a prompt in the prefix cache");
            }
        }
        let r = &mut active[i];
        log_cache(speculator);
        if end == len {
            r.kept = None;
            r.prefill_s = r.started.elapsed().as_secs_f64();
            r.send_images();
            r.next = Some(first);
            r.done = r.take(&[first], stop).is_break();
            share_prefill(model, speculator, active, i, stop);
        }
        processed.max(1)
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
    fn share_prefill(
        model: &mut Model,
        speculator: &mut Speculator,
        active: &mut [Request],
        from: usize,
        stop: &[u32],
    ) {
        let (slot, group, len, draft_ready) = {
            let r = &active[from];
            (r.slot, r.job.group, r.job.prompt.len(), r.draft_ready)
        };
        let copy_draft = draft_ready && speculator.supports_batch_spec();
        let siblings = active.iter_mut().filter(|r| {
            r.job.group == group
                && r.slot != slot
                && !r.done
                && r.next.is_none()
                && r.prefilled == 0
        });
        for r in siblings {
            let copied = model.move_slot(slot, r.slot).and_then(|()| {
                if copy_draft {
                    speculator.copy_slot_draft(model, slot, r.slot)?;
                }
                model.redraw_slot(r.slot, r.sampling)
            });
            match copied {
                Ok(first) => {
                    r.prefilled = len;
                    r.shared = true;
                    r.draft_ready = copy_draft;
                    r.prefill_s = r.started.elapsed().as_secs_f64();
                    r.next = Some(first);
                    r.done = r.take(&[first], stop).is_break();
                }
                Err(e) => r.fail(&e),
            }
        }
    }

    /// Reusable per-step vectors for [`step_rows`], avoiding heap allocations on every
    /// decode/verify step.
    struct StepScratch {
        spec_indices: Vec<usize>,
        plain_indices: Vec<usize>,
        slots: Vec<usize>,
        anchors: Vec<u32>,
        spec_samplings: Vec<ModelSampling>,
        rows: Vec<(usize, u32)>,
        plain_samplings: Vec<ModelSampling>,
    }

    impl StepScratch {
        fn with_capacity(streams: usize) -> Self {
            Self {
                spec_indices: Vec::with_capacity(streams),
                plain_indices: Vec::with_capacity(streams),
                slots: Vec::with_capacity(streams),
                anchors: Vec::with_capacity(streams),
                spec_samplings: Vec::with_capacity(streams),
                rows: Vec::with_capacity(streams),
                plain_samplings: Vec::with_capacity(streams),
            }
        }
    }

    /// Runs one batched decode step or batched DFlash 2 speculation round over the requests
    /// of `active` that have a next token. Returns the minimum number of tokens emitted per
    /// active decoding request (0 when none decoded).
    #[expect(
        clippy::too_many_lines,
        reason = "batched speculation and plain decode step"
    )]
    fn step_rows(
        model: &mut Model,
        speculator: &mut Speculator,
        active: &mut [Request],
        stop: &[u32],
        group_by_confidence: bool,
        scratch: &mut StepScratch,
    ) -> usize {
        let batch_spec = speculator.supports_batch_spec();
        scratch.spec_indices.clear();
        scratch.plain_indices.clear();
        for (idx, r) in active.iter().enumerate() {
            if r.done || r.next.is_none() {
                continue;
            }
            if batch_spec && r.draft_ready {
                scratch.spec_indices.push(idx);
            } else {
                scratch.plain_indices.push(idx);
            }
        }
        if scratch.spec_indices.is_empty() && scratch.plain_indices.is_empty() {
            return 0;
        }
        let mut min_emitted = usize::MAX;
        if !scratch.spec_indices.is_empty() {
            let max_seqs = speculator.max_batch_spec_seqs();
            let grouped = if group_by_confidence || scratch.spec_indices.len() > max_seqs {
                let candidates: Vec<(usize, usize, Option<f32>)> = scratch
                    .spec_indices
                    .iter()
                    .map(|&i| {
                        let slot = active[i].slot;
                        (i, slot, speculator.batch_slot_confidence(slot))
                    })
                    .collect();
                Some(super::group_spec_chunks(
                    &candidates,
                    max_seqs,
                    group_by_confidence,
                ))
            } else {
                None
            };
            let chunks: &[&[usize]] = &[];
            let single = [scratch.spec_indices.as_slice()];
            let grouped_slices: Vec<&[usize]> = grouped
                .as_ref()
                .map(|g| g.iter().map(Vec::as_slice).collect())
                .unwrap_or_default();
            let iter_chunks = if grouped.is_some() {
                grouped_slices.as_slice()
            } else if scratch.spec_indices.is_empty() {
                chunks
            } else {
                &single
            };
            for &chunk in iter_chunks {
                let width = (BATCH_VERIFY_MAX_ROWS / chunk.len()).min(VERIFY_ROWS);
                scratch.slots.clear();
                scratch.anchors.clear();
                scratch.spec_samplings.clear();
                for &i in chunk {
                    scratch.slots.push(active[i].slot);
                    scratch.anchors.push(active[i].next.unwrap_or_default());
                    scratch.spec_samplings.push(active[i].sampling);
                }
                match speculator.speculate_batch_round(
                    model,
                    &scratch.slots,
                    &scratch.anchors,
                    &scratch.spec_samplings,
                    width,
                ) {
                    Ok(verdicts) => {
                        for (&idx, verdict) in chunk.iter().zip(verdicts) {
                            let r = &mut active[idx];
                            let emitted = verdict.emitted();
                            min_emitted = min_emitted.min(emitted.len());
                            r.speculated += emitted.len();
                            let _ = r.job.events.send(Event::Round);
                            r.next = Some(verdict.next());
                            r.done = r.take(&emitted, stop).is_break();
                        }
                    }
                    Err(e) => {
                        speculator.invalidate_batch_draft();
                        for &idx in chunk {
                            active[idx].fail(&e);
                        }
                        min_emitted = min_emitted.min(1);
                    }
                }
            }
        }
        if !scratch.plain_indices.is_empty() {
            scratch.rows.clear();
            scratch.plain_samplings.clear();
            for &i in &scratch.plain_indices {
                scratch
                    .rows
                    .push((active[i].slot, active[i].next.unwrap_or_default()));
                scratch.plain_samplings.push(active[i].sampling);
            }
            match model.step_batch_sampled(&scratch.rows, &scratch.plain_samplings) {
                Ok(next) => {
                    let alone_plain = scratch.rows.len() == 1 && scratch.spec_indices.is_empty();
                    for (row, (&idx, token)) in scratch.plain_indices.iter().zip(next).enumerate() {
                        let r = &mut active[idx];
                        if r.followed
                            && let Err(e) = speculator.keep_batched(model, row, token)
                        {
                            log!("the draft stops following a request: {e:#}");
                            speculator.forget();
                            r.followed = false;
                        }
                        r.batched += 1;
                        if alone_plain {
                            r.rent += 1;
                        }
                        r.next = Some(token);
                        r.done = r.take(&[token], stop).is_break();
                    }
                    min_emitted = min_emitted.min(1);
                }
                Err(e) => {
                    for &idx in &scratch.plain_indices {
                        active[idx].fail(&e);
                    }
                    min_emitted = min_emitted.min(1);
                }
            }
        }
        if min_emitted == usize::MAX {
            0
        } else {
            min_emitted.max(1)
        }
    }

    /// Commits progress for piggybacked prefill slices after a hybrid decode/verify pass:
    /// advances `prefilled`, saves prefix cache entries at chunk boundaries, and hands off
    /// finishing prompts into decode/verify state.
    fn finish_piggybacked_prefills(
        model: &mut Model,
        speculator: &mut Speculator,
        active: &mut [Request],
        slices: &[PiggybackSlice],
        tokens: &[Option<u32>],
        chunk: usize,
        stop: &[u32],
    ) {
        let busy: Vec<usize> = active.iter().map(|r| r.slot).collect();
        for (s, &first) in slices.iter().zip(tokens) {
            active[s.index].prefilled = s.end;
            if s.hits_prefix_boundary && s.end > active[s.index].cached {
                if make_room(model, speculator, model.ring_pages(), &busy, Reclaim::Draft) {
                    let key = active[s.index].job.key();
                    if let Err(e) = speculator.save_prefix(model, s.slot, key, chunk, &busy) {
                        log!("keeping a prompt in the prefix cache failed: {e:#}");
                    }
                } else {
                    log!("no room in the KV pool to keep a prompt in the prefix cache");
                }
            }
            if s.finishes_prompt
                && let Some(first) = first
            {
                let r = &mut active[s.index];
                r.kept = None;
                r.prefill_s = r.started.elapsed().as_secs_f64();
                r.send_images();
                r.next = Some(first);
                r.done = r.take(&[first], stop).is_break();
            }
        }
    }

    /// Runs one hybrid decode/verify step with `slices` piggybacked onto the primary active
    /// decode or batched-verify pass (followed by any overflow verify micro-batches or
    /// secondary plain-decode rows).
    #[expect(
        clippy::too_many_arguments,
        clippy::too_many_lines,
        reason = "encodes images, dispatches hybrid verify or hybrid decode, handles overflow"
    )]
    fn step_rows_hybrid(
        model: &mut Model,
        speculator: &mut Speculator,
        tower: &mut Tower,
        active: &mut [Request],
        slices: &[PiggybackSlice],
        stop: &[u32],
        group_by_confidence: bool,
        scratch: &mut StepScratch,
    ) {
        let batch_spec = speculator.supports_batch_spec();
        let mut spec_indices = Vec::new();
        let mut plain_indices = Vec::new();
        for (idx, r) in active.iter().enumerate() {
            if r.done || r.next.is_none() {
                continue;
            }
            if batch_spec && r.draft_ready {
                spec_indices.push(idx);
            } else {
                plain_indices.push(idx);
            }
        }
        if spec_indices.is_empty() && plain_indices.is_empty() {
            return;
        }
        let hidden = model.config().hidden_size;
        let mut live_slices = Vec::with_capacity(slices.len());
        let mut encoded_images = Vec::with_capacity(slices.len());
        for &s in slices {
            let r = &mut active[s.index];
            if r.done || r.next.is_some() {
                continue;
            }
            let imgs = if r.job.images.is_empty() {
                Vec::new()
            } else {
                match image_rows(
                    tower,
                    hidden,
                    &r.job.images,
                    &mut r.kept,
                    &mut r.encodes,
                    (s.start, s.end),
                ) {
                    Ok(imgs) => imgs,
                    Err(e) => {
                        r.fail(&e);
                        continue;
                    }
                }
            };
            live_slices.push(s);
            encoded_images.push(imgs);
        }
        if live_slices.is_empty() {
            let _ = step_rows(
                model,
                speculator,
                active,
                stop,
                group_by_confidence,
                scratch,
            );
            return;
        }
        let chunk = speculator.prefill_chunk();
        let mut commit_draft = Vec::with_capacity(live_slices.len());
        let mut reset_draft = Vec::with_capacity(live_slices.len());
        let mut packed_slices = Vec::with_capacity(live_slices.len());
        for (&s, imgs) in live_slices.iter().zip(encoded_images) {
            let r = &active[s.index];
            commit_draft.push(batch_spec && r.draft_ready);
            reset_draft.push(s.start == 0);
            packed_slices.push(PackedPrefill {
                slot: s.slot,
                tokens: &r.job.prompt[s.start..s.end],
                images: imgs,
                need_first_token: s.finishes_prompt,
                sampling: r.sampling,
            });
        }
        if spec_indices.is_empty() {
            let rows: Vec<(usize, u32)> = plain_indices
                .iter()
                .map(|&i| (active[i].slot, active[i].next.unwrap_or_default()))
                .collect();
            let sampling: Vec<ModelSampling> =
                plain_indices.iter().map(|&i| active[i].sampling).collect();
            let result = speculator.step_hybrid_batch(
                model,
                &rows,
                &sampling,
                &packed_slices,
                &commit_draft,
                &reset_draft,
                chunk,
            );
            drop(packed_slices);
            match result {
                Ok(res) => {
                    for (row, (&idx, token)) in
                        plain_indices.iter().zip(res.decode_tokens).enumerate()
                    {
                        let r = &mut active[idx];
                        if r.followed
                            && let Err(e) = speculator.keep_batched(model, row, token)
                        {
                            log!("the draft stops following a request: {e:#}");
                            speculator.forget();
                            r.followed = false;
                        }
                        r.batched += 1;
                        r.next = Some(token);
                        r.done = r.take(&[token], stop).is_break();
                    }
                    finish_piggybacked_prefills(
                        model,
                        speculator,
                        active,
                        &live_slices,
                        &res.prefill_tokens,
                        chunk,
                        stop,
                    );
                }
                Err(e) => {
                    for &idx in &plain_indices {
                        active[idx].fail(&e);
                    }
                    for &s in &live_slices {
                        active[s.index].fail(&e);
                    }
                }
            }
        } else {
            let max_seqs = speculator.max_batch_spec_seqs();
            let (first_chunk, rest_spec) = spec_indices.split_at(spec_indices.len().min(max_seqs));
            let width = (BATCH_VERIFY_MAX_ROWS / first_chunk.len()).min(VERIFY_ROWS);
            let slots: Vec<usize> = first_chunk.iter().map(|&i| active[i].slot).collect();
            let anchors: Vec<u32> = first_chunk
                .iter()
                .map(|&i| active[i].next.unwrap_or_default())
                .collect();
            let samplings: Vec<ModelSampling> =
                first_chunk.iter().map(|&i| active[i].sampling).collect();
            let result = speculator.speculate_hybrid_round(
                model,
                &slots,
                &anchors,
                &samplings,
                width,
                &packed_slices,
                &commit_draft,
                &reset_draft,
                chunk,
            );
            drop(packed_slices);
            match result {
                Ok(res) => {
                    for (&idx, verdict) in first_chunk.iter().zip(res.verdicts) {
                        let r = &mut active[idx];
                        let emitted = verdict.emitted();
                        r.speculated += emitted.len();
                        let _ = r.job.events.send(Event::Round);
                        r.next = Some(verdict.next());
                        r.done = r.take(&emitted, stop).is_break();
                    }
                    finish_piggybacked_prefills(
                        model,
                        speculator,
                        active,
                        &live_slices,
                        &res.prefill_tokens,
                        chunk,
                        stop,
                    );
                }
                Err(e) => {
                    speculator.invalidate_batch_draft();
                    for &idx in first_chunk {
                        active[idx].fail(&e);
                    }
                    for &s in &live_slices {
                        active[s.index].fail(&e);
                    }
                }
            }
            for extra in rest_spec.chunks(max_seqs) {
                let extra_w = (BATCH_VERIFY_MAX_ROWS / extra.len()).min(VERIFY_ROWS);
                let extra_slots: Vec<usize> = extra.iter().map(|&i| active[i].slot).collect();
                let extra_anchors: Vec<u32> = extra
                    .iter()
                    .map(|&i| active[i].next.unwrap_or_default())
                    .collect();
                let extra_samplings: Vec<ModelSampling> =
                    extra.iter().map(|&i| active[i].sampling).collect();
                match speculator.speculate_batch_round(
                    model,
                    &extra_slots,
                    &extra_anchors,
                    &extra_samplings,
                    extra_w,
                ) {
                    Ok(verdicts) => {
                        for (&idx, verdict) in extra.iter().zip(verdicts) {
                            let r = &mut active[idx];
                            let emitted = verdict.emitted();
                            r.speculated += emitted.len();
                            let _ = r.job.events.send(Event::Round);
                            r.next = Some(verdict.next());
                            r.done = r.take(&emitted, stop).is_break();
                        }
                    }
                    Err(e) => {
                        speculator.invalidate_batch_draft();
                        for &idx in extra {
                            active[idx].fail(&e);
                        }
                    }
                }
            }
            if !plain_indices.is_empty() {
                let rows: Vec<(usize, u32)> = plain_indices
                    .iter()
                    .map(|&i| (active[i].slot, active[i].next.unwrap_or_default()))
                    .collect();
                let sampling: Vec<ModelSampling> =
                    plain_indices.iter().map(|&i| active[i].sampling).collect();
                match model.step_batch_sampled(&rows, &sampling) {
                    Ok(next) => {
                        for (row, (&idx, token)) in plain_indices.iter().zip(next).enumerate() {
                            let r = &mut active[idx];
                            if r.followed
                                && let Err(e) = speculator.keep_batched(model, row, token)
                            {
                                log!("the draft stops following a request: {e:#}");
                                speculator.forget();
                                r.followed = false;
                            }
                            r.batched += 1;
                            r.next = Some(token);
                            r.done = r.take(&[token], stop).is_break();
                        }
                    }
                    Err(e) => {
                        for &idx in &plain_indices {
                            active[idx].fail(&e);
                        }
                    }
                }
            }
        }
        log_cache(speculator);
    }
}

/// Partitions speculative decoding candidates `(idx, slot, confidence)` into micro-batches
/// of at most `max_seqs` entries.
///
/// When `items.len() > max_seqs`, sorts descending by calibrated confidence before chunking
/// so high-confidence requests batch together (picking wider verify widths) and low-confidence
/// requests batch together. When `group_by_confidence` is set and `items.len() <= max_seqs`,
/// splits a mixed-confidence batch (`max - min >= 0.30` across known confidences) into high
/// (`>= 0.50`) and low (`< 0.50`) groups. Each chunk is canonicalized in ascending `slot`
/// order so pre-drafted slot caches remain hits when group membership is unchanged.
#[cfg_attr(
    all(not(target_os = "linux"), not(test)),
    expect(dead_code, reason = "only the Linux engine schedules micro-batches")
)]
fn group_spec_chunks(
    items: &[(usize, usize, Option<f32>)],
    max_seqs: usize,
    group_by_confidence: bool,
) -> Vec<Vec<usize>> {
    const SPLIT_SPREAD: f32 = 0.30;
    const SPLIT_THRESHOLD: f32 = 0.50;

    if items.is_empty() {
        return Vec::new();
    }
    if items.len() <= max_seqs && !group_by_confidence {
        return vec![items.iter().map(|&(idx, _, _)| idx).collect()];
    }
    let mut ranked = items.to_vec();
    ranked.sort_by(|a, b| {
        let ca = a.2.unwrap_or(SPLIT_THRESHOLD);
        let cb = b.2.unwrap_or(SPLIT_THRESHOLD);
        cb.total_cmp(&ca).then_with(|| a.1.cmp(&b.1))
    });
    let mut chunks: Vec<Vec<(usize, usize)>> = if ranked.len() > max_seqs {
        ranked
            .chunks(max_seqs)
            .map(|c| c.iter().map(|&(idx, slot, _)| (idx, slot)).collect())
            .collect()
    } else {
        let mut min_c = f32::INFINITY;
        let mut max_c = f32::NEG_INFINITY;
        for &(_, _, conf) in &ranked {
            if let Some(c) = conf {
                min_c = min_c.min(c);
                max_c = max_c.max(c);
            }
        }
        if min_c.is_finite() && max_c - min_c >= SPLIT_SPREAD {
            let mut high = Vec::new();
            let mut low = Vec::new();
            for &(idx, slot, conf) in &ranked {
                if conf.unwrap_or(SPLIT_THRESHOLD) >= SPLIT_THRESHOLD {
                    high.push((idx, slot));
                } else {
                    low.push((idx, slot));
                }
            }
            if !high.is_empty() && !low.is_empty() {
                vec![high, low]
            } else {
                vec![ranked.iter().map(|&(idx, slot, _)| (idx, slot)).collect()]
            }
        } else {
            vec![ranked.iter().map(|&(idx, slot, _)| (idx, slot)).collect()]
        }
    };
    for chunk in &mut chunks {
        chunk.sort_by_key(|&(_, slot)| slot);
    }
    chunks
        .into_iter()
        .map(|chunk| chunk.into_iter().map(|(idx, _)| idx).collect())
        .collect()
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

    #[test]
    fn confidence_grouping_sorts_multi_chunks_and_splits_mixed() {
        // Multi-chunk (> max_seqs): sorts descending by confidence, canonicalizes each chunk by slot.
        let items = [
            (0, 3, Some(0.20)),
            (1, 1, Some(0.90)),
            (2, 4, Some(0.15)),
            (3, 0, Some(0.85)),
        ];
        assert_eq!(
            group_spec_chunks(&items, 2, false),
            vec![vec![3, 1], vec![0, 2]]
        );

        // Single chunk with grouping off preserves input order unchanged.
        assert_eq!(group_spec_chunks(&items, 4, false), vec![vec![0, 1, 2, 3]]);

        // Single chunk with grouping on splits wide spread into high/low, each in slot order.
        assert_eq!(
            group_spec_chunks(&items, 4, true),
            vec![vec![3, 1], vec![0, 2]]
        );

        // Homogeneous batch with grouping on stays one chunk (no extra verify launch).
        let homogeneous = [(0, 2, Some(0.80)), (1, 0, Some(0.72)), (2, 1, Some(0.75))];
        assert_eq!(
            group_spec_chunks(&homogeneous, 4, true),
            vec![vec![1, 2, 0]]
        );
    }

    #[test]
    fn small_token_events_stay_inline_up_to_cap() {
        for len in 1..=SMALL_TOKENS_CAP {
            let ids: Vec<u32> = (0..len as u32).map(|i| i + 10).collect();
            match Event::from_tokens(&ids) {
                Event::SmallTokens(buf, n) => {
                    assert_eq!(usize::from(n), len);
                    assert_eq!(&buf[..len], ids.as_slice());
                }
                other => panic!("expected SmallTokens for len {len}, got {other:?}"),
            }
        }
        let big: Vec<u32> = (0..SMALL_TOKENS_CAP as u32 + 2).collect();
        match Event::from_tokens(&big) {
            Event::Tokens(got) => assert_eq!(got, big),
            other => panic!("expected Tokens fallback, got {other:?}"),
        }
    }

    #[test]
    fn starts_with_chained_matches_concatenation() {
        let a = [1u32, 2, 3];
        let b = [4u32, 5, 6];
        assert!(starts_with_chained(&a, &b, &[]));
        assert!(starts_with_chained(&a, &b, &[1, 2]));
        assert!(starts_with_chained(&a, &b, &[1, 2, 3]));
        assert!(starts_with_chained(&a, &b, &[1, 2, 3, 4]));
        assert!(starts_with_chained(&a, &b, &[1, 2, 3, 4, 5, 6]));
        assert!(!starts_with_chained(&a, &b, &[1, 2, 3, 4, 5, 6, 7]));
        assert!(!starts_with_chained(&a, &b, &[1, 2, 9]));
        assert!(!starts_with_chained(&a, &b, &[1, 2, 3, 9]));
    }
}
