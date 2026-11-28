//! Baseline token generation for Inkling 2 Mini on Apple Silicon.
//!
//! [`Model`] runs a persistent Python worker using MLX-LM with Inkling 2 support.
//! Callers supply token IDs and handle tokenization, end-of-turn tokens, and decoding.

use std::{
    ffi::OsStr,
    io::{self, BufRead, BufReader, Write as _},
    ops::ControlFlow,
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};

const WORKER: &str = include_str!("worker.py");

/// Settings for one sequence. Each request starts with a fresh KV cache.
#[derive(Debug, Clone, Copy, Serialize)]
#[non_exhaustive]
pub struct GenerationOptions {
    /// Maximum number of new tokens to emit.
    pub max_tokens: usize,
    /// Number of prompt tokens processed in one prefill step.
    pub prefill_chunk: usize,
    /// Sampling temperature. Zero selects greedy decoding.
    pub temperature: f32,
    /// Keep at most this many tokens by score. Zero means no limit.
    pub top_k: u32,
    /// Keep the smallest set of tokens reaching this share of probability mass.
    pub top_p: f32,
    /// Drop tokens below this fraction of the most likely token's probability.
    pub min_p: f32,
    /// Random seed for this sequence.
    pub seed: u64,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            prefill_chunk: 2048,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
        }
    }
}

/// An error loading or running an MLX model.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct Error(#[from] Failure);

#[derive(Debug, thiserror::Error)]
enum Failure {
    #[error("MLX inference requires Apple Silicon macOS")]
    Platform,
    #[error("resolving the local model directory: {0}")]
    Path(#[source] io::Error),
    #[error("starting MLX worker with {python:?}: {source}")]
    Spawn {
        python: std::ffi::OsString,
        #[source]
        source: io::Error,
    },
    #[error("communicating with the MLX worker: {0}")]
    Io(#[from] io::Error),
    #[error("invalid MLX worker message: {0}")]
    Json(#[from] serde_json::Error),
    #[error("the MLX worker exited or is unavailable")]
    WorkerLost,
    #[error("MLX-LM: {0}")]
    Model(String),
    #[error("invalid generation settings: {0}")]
    Invalid(&'static str),
}

/// A loaded Inkling 2 Mini model that generates one sequence at a time on Metal.
#[derive(Debug)]
pub struct Model {
    worker: Option<Worker>,
    info: ModelInfo,
}

impl Model {
    /// Loads a local checkpoint using a Python executable with MLX-LM installed.
    ///
    /// # Errors
    ///
    /// Returns an error on an unsupported platform, if Python cannot start, or if
    /// MLX-LM cannot load the checkpoint on Metal.
    pub fn load(model_dir: impl AsRef<Path>, python: impl AsRef<OsStr>) -> Result<Self, Error> {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return Err(Failure::Platform.into());
        }

        let model_dir = model_dir.as_ref().canonicalize().map_err(Failure::Path)?;
        let python = python.as_ref();
        let mut command = Command::new(python);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt as _;
            // Let the server drain requests after a terminal Ctrl-C.
            command.process_group(0);
        }
        let mut child = command
            .args(["-u", "-c", WORKER])
            .arg(model_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|source| Failure::Spawn {
                python: python.to_owned(),
                source,
            })?;
        let mut worker = Worker {
            input: child.stdin.take().expect("MLX worker stdin is piped"),
            output: BufReader::new(child.stdout.take().expect("MLX worker stdout is piped")),
            child,
            reply: String::new(),
        };
        let info = worker.read(|_| {})?;
        Ok(Self {
            worker: Some(worker),
            info,
        })
    }

    /// Returns the checkpoint's maximum combined prompt and generation length.
    #[must_use]
    pub fn max_context(&self) -> usize {
        self.info.max_context
    }

    /// Generates token IDs until the limit or until `on_token` returns a break.
    ///
    /// The callback receives every token, including special tokens. A break releases
    /// the request's KV cache and leaves the model ready for another prompt.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid token IDs, sampling settings, context length, or
    /// a failed MLX-LM call. Reload the model after a worker communication failure.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        options: GenerationOptions,
        mut on_token: impl FnMut(u32) -> ControlFlow<()>,
    ) -> Result<(), Error> {
        self.generate_with_progress(prompt, options, |_| {}, |token, _| on_token(token))
    }

    /// Generates tokens while reporting completed prefill chunks and time to receive
    /// each token from the worker. The first token's time includes prefill.
    ///
    /// `on_prefill` receives the number of new prompt tokens in each chunk. As with
    /// [`Self::generate`], returning a break from `on_token` ends generation.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::generate`].
    pub fn generate_with_progress(
        &mut self,
        prompt: &[u32],
        options: GenerationOptions,
        mut on_prefill: impl FnMut(usize),
        mut on_token: impl FnMut(u32, Duration) -> ControlFlow<()>,
    ) -> Result<(), Error> {
        if prompt.is_empty() || prompt.iter().any(|&id| id >= self.info.vocab_size) {
            return Err(Failure::Invalid("expected a nonempty prompt with valid token IDs").into());
        }
        if prompt.len().saturating_add(options.max_tokens) > self.max_context() {
            return Err(Failure::Invalid("prompt and output exceed the checkpoint context").into());
        }
        if options.prefill_chunk == 0 {
            return Err(Failure::Invalid("prefill_chunk must be positive").into());
        }
        if !(options.temperature.is_finite()
            && options.temperature >= 0.0
            && options.top_p > 0.0
            && options.top_p <= 1.0
            && (0.0..=1.0).contains(&options.min_p))
        {
            return Err(Failure::Invalid("invalid temperature, top_p, or min_p").into());
        }
        if options.max_tokens == 0 {
            return Ok(());
        }

        self.request::<()>(&Request::Generate { prompt, options })?;
        let mut prefilled = 0;
        for i in 0..options.max_tokens {
            let started = Instant::now();
            let token = self.request_with_progress(&Request::Next, |processed| {
                let processed = processed.min(prompt.len());
                if processed > prefilled {
                    on_prefill(processed - prefilled);
                    prefilled = processed;
                }
            })?;
            // Older MLX-LM builds can report prefill only when the first token is ready.
            if i == 0 && prefilled < prompt.len() {
                on_prefill(prompt.len() - prefilled);
                prefilled = prompt.len();
            }
            if on_token(token, started.elapsed()).is_break() {
                break;
            }
        }
        self.request(&Request::Reset)
    }

    fn request<T: DeserializeOwned>(&mut self, request: &Request<'_>) -> Result<T, Error> {
        self.request_with_progress(request, |_| {})
    }

    fn request_with_progress<T: DeserializeOwned>(
        &mut self,
        request: &Request<'_>,
        on_prefill: impl FnMut(usize),
    ) -> Result<T, Error> {
        let worker = self.worker.as_mut().ok_or(Failure::WorkerLost)?;
        let result = worker.exchange(request, on_prefill);
        if result
            .as_ref()
            .is_err_and(|e| !matches!(e, Failure::Model(_)))
        {
            self.worker = None;
        }
        Ok(result?)
    }
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    max_context: usize,
    vocab_size: u32,
}

#[derive(Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request<'a> {
    Generate {
        prompt: &'a [u32],
        options: GenerationOptions,
    },
    Next,
    Reset,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Reply<T> {
    Ok(T),
    Error(String),
    Prefill(usize),
}

#[derive(Debug)]
struct Worker {
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    child: Child,
    reply: String,
}

impl Worker {
    fn exchange<T: DeserializeOwned>(
        &mut self,
        request: &Request<'_>,
        on_prefill: impl FnMut(usize),
    ) -> Result<T, Failure> {
        let mut message = serde_json::to_vec(request)?;
        message.push(b'\n');
        self.input.write_all(&message)?;
        self.read(on_prefill)
    }

    fn read<T: DeserializeOwned>(
        &mut self,
        mut on_prefill: impl FnMut(usize),
    ) -> Result<T, Failure> {
        loop {
            self.reply.clear();
            if self.output.read_line(&mut self.reply)? == 0 {
                return Err(Failure::WorkerLost);
            }
            match serde_json::from_str(&self.reply)? {
                Reply::Ok(value) => return Ok(value),
                Reply::Error(message) => return Err(Failure::Model(message)),
                Reply::Prefill(tokens) => on_prefill(tokens),
            }
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
