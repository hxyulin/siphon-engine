//! `siphon-serve`: an OpenAI-compatible chat server for Inkling 2 Mini on one GPU.

mod anthropic;
mod chat;
mod engine;
mod metrics;
mod openai;
mod parse;
mod request;
mod stop;
mod tokenize;

use std::{env, net::SocketAddr, path::PathBuf, process::ExitCode, sync::Arc};

use anyhow::{Context as _, bail, ensure};
use serde_json::json;

/// Requests that may run at once: up to the rows of a batched step.
const STREAMS: std::ops::RangeInclusive<usize> = 1..=32;
#[cfg(target_os = "linux")]
const _: () = assert!(*STREAMS.end() == siphon::memory::BATCH_ROWS);
/// Rows of a prefill chunk.
const PREFILL_CHUNK: std::ops::RangeInclusive<usize> = 1..=2048;
#[cfg(target_os = "linux")]
const _: () = assert!(*PREFILL_CHUNK.end() == siphon::memory::MAX_PREFILL_CHUNK);

const USAGE: &str = "usage: siphon-serve <model-dir> [--host 127.0.0.1] [--port 8000] \
                     [--context N] [--memory GiB] [--margin GiB] [--prefill-chunk 2048] [--weights fp8|bf16] [--kv fp8|bf16] [--streams 1] \
                     [--queue 8] [--widths 1,2,...,8] [--name inkling-2-mini] [--drafter DIR] [--draft fast] \
                     [--fallback] [--no-prefix-cache] [--prefix-cache-entries N] [--prefix-cache-ttl 600] \
                     [--prefill-attention auto|bf16|fp8] [--greedy]
--context sets the positions of the KV cache: the prompt and the generated tokens. Without it
the server takes the most that fit in GPU memory, up to the model's limit of 1,048,576. Either
way it plans the memory before loading the weights and stops if the plan does not fit.
--streams runs up to N requests at once (1 to 32), each with its own KV cache of --context
positions. A request alone speculates; once another arrives, all run batched plain decode.
Either way a request gets the same tokens. Up to --queue more requests wait.
--memory caps the GPU memory the plan may use, the CUDA context included. Without it the
plan may use the free memory at startup less --margin, 0.5 GiB by default.
--prefill-chunk runs prompts in chunks of N rows, 1 to 2048. Its buffers grow with N: 0.85 GiB
at 2048 rows and 0.47 GiB at 1024, and the memory they free goes to the context.
--drafter runs the drafter in DIR instead of the bundled draft: Inco AI's DFlash 2 drafter
or a DFlash draft exported by the 2DSD project.
--draft picks the draft's approximations as bench does: fast (the default, FP8 plus the
counted shortlist), fp8, exact, or a +-joined set of fp8-backbone, fp8-head, fp8-fold, fp8-kv
and shortlist.
--fallback runs plain steps where no verify width is expected to be faster. It speeds up
free prose, but where it steps depends on timings measured at startup, so the output can
differ between runs.
--no-prefix-cache turns off the prefix cache. The cache keeps the state of recent prompts at
their last multiple of --prefill-chunk tokens, one per stream, and a request whose prompt starts with
one of them prefills only the rest. Each entry costs about 0.17 GiB, 0.2 GiB with DFlash 2,
whatever the context. --prefix-cache-entries keeps fewer entries than --streams, to leave
more memory for the context. --prefix-cache-ttl drops an entry unused for that many seconds.
--prefill-attention picks prefill's global attention products over an FP8 cache. auto (the
default) runs BF16 up to 16,384 keys and FP8 tensor cores past them, where they cut time to
first token by up to a fifth; bf16 keeps every prompt as before, and fp8 runs FP8 throughout.
--greedy decodes every request greedily, whatever its sampling fields ask. Without it, a
request samples with its temperature, top_k, top_p, min_p and seed, and the fields it leaves
out come from the checkpoint's generation_config.json (temperature 1 for Inkling 2 Mini).
Speculation keeps the model's distribution exactly, and a request with a seed gets the same
tokens on the same configuration, alone or in a batch (without --fallback).
Set SIPHON_API_KEY to require `Authorization: Bearer <key>` or `x-api-key: <key>` on every route
but /health, /version and /metrics.";

/// The prompt that prices the verify widths at startup.
const WARMUP: &str = "Explain, step by step, how a hash map handles collisions, and compare \
                      separate chaining with open addressing in terms of memory use, cache \
                      behaviour and worst-case lookup time.";

struct Args {
    model_dir: PathBuf,
    address: SocketAddr,
    name: String,
    engine: engine::Options,
}

fn args() -> anyhow::Result<Args> {
    let mut model_dir = None;
    let mut host = "127.0.0.1".to_owned();
    let mut port = 8000u16;
    let mut name = "inkling-2-mini".to_owned();
    let mut capacity = None;
    let mut memory = None;
    let mut margin = None;
    let mut prefill_chunk = None;
    let mut fp8_weights = true;
    let mut fp8_kv = true;
    let mut queue = 8;
    let mut streams = 1;
    let mut widths: Vec<usize> = (1..=8).collect();
    let mut drafter = None;
    let mut draft = "fast".to_owned();
    let mut fallback = false;
    let mut prefix_cache = true;
    let mut prefix_entries = None;
    let mut prefix_ttl = 600;
    let mut prefill_attention = "auto".to_owned();
    let mut greedy = false;
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || {
            it.next()
                .with_context(|| format!("{arg} needs a value\n{USAGE}"))
        };
        match arg.as_str() {
            "--host" => host = value()?,
            "--port" => port = value()?.parse()?,
            "--name" => name = value()?,
            "--context" => capacity = Some(value()?.parse()?),
            "--memory" => memory = Some(gib_bytes(&value()?, "--memory", false)?),
            "--margin" => margin = Some(gib_bytes(&value()?, "--margin", true)?),
            "--prefill-chunk" => prefill_chunk = Some(chunk_rows(&value()?)?),
            "--weights" => fp8_weights = fp8(&value()?)?,
            "--kv" => fp8_kv = fp8(&value()?)?,
            "--queue" => queue = value()?.parse()?,
            "--streams" => {
                streams = value()?.parse()?;
                ensure!(STREAMS.contains(&streams), "--streams takes 1 to 32");
            }
            "--drafter" => drafter = Some(PathBuf::from(value()?)),
            "--draft" => draft = value()?,
            "--fallback" => fallback = true,
            "--no-prefix-cache" => prefix_cache = false,
            "--prefix-cache-entries" => prefix_entries = Some(value()?.parse()?),
            "--prefix-cache-ttl" => prefix_ttl = value()?.parse()?,
            "--prefill-attention" => prefill_attention = value()?,
            "--greedy" => greedy = true,
            "--widths" => {
                widths = value()?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()
                    .context("--widths takes comma-separated widths")?;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ if arg.starts_with("--") => bail!("unknown flag {arg}\n{USAGE}"),
            _ => {
                ensure!(model_dir.is_none(), "{USAGE}");
                model_dir = Some(PathBuf::from(arg));
            }
        }
    }
    let model_dir = model_dir.context(USAGE)?;
    let address = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("{host}:{port} is not an address"))?;
    Ok(Args {
        engine: engine::Options {
            model_dir: model_dir.clone(),
            capacity,
            memory,
            margin,
            prefill_chunk,
            fp8_weights,
            fp8_kv,
            widths,
            queue,
            streams,
            drafter,
            draft,
            fallback,
            prefix_entries: if prefix_cache {
                entries(prefix_entries, streams)?
            } else {
                0
            },
            prefix_ttl: std::time::Duration::from_secs(prefix_ttl),
            prefill_attention,
            greedy,
        },
        model_dir,
        address,
        name,
    })
}

/// The bytes of `flag`'s value in GiB: positive, or with `zero` 0 or more.
fn gib_bytes(value: &str, flag: &str, zero: bool) -> anyhow::Result<usize> {
    let gib: f64 = value.parse().with_context(|| format!("{flag} takes GiB"))?;
    ensure!(
        gib > 0.0 || (zero && gib == 0.0),
        "{flag} takes a {} number of GiB",
        if zero { "non-negative" } else { "positive" }
    );
    #[expect(clippy::cast_sign_loss, reason = "gib is not negative")]
    Ok((gib * f64::from(1u32 << 30)) as usize)
}

/// The rows of `--prefill-chunk`.
fn chunk_rows(value: &str) -> anyhow::Result<usize> {
    let rows = value.parse().context("--prefill-chunk takes rows")?;
    ensure!(
        PREFILL_CHUNK.contains(&rows),
        "--prefill-chunk takes 1 to 2048"
    );
    Ok(rows)
}

/// The prefix cache's entries: `--prefix-cache-entries`, 1 to `streams`, or `streams`.
fn entries(entries: Option<usize>, streams: usize) -> anyhow::Result<usize> {
    let entries = entries.unwrap_or(streams);
    ensure!(
        (1..=streams).contains(&entries),
        "--prefix-cache-entries takes 1 to --streams, {streams}"
    );
    Ok(entries)
}

fn fp8(format: &str) -> anyhow::Result<bool> {
    match format {
        "fp8" => Ok(true),
        "bf16" => Ok(false),
        _ => bail!("formats are fp8 or bf16, not {format}"),
    }
}

fn main() -> ExitCode {
    // Load every kernel's code when its module loads, at startup, rather than at its first
    // launch, which may come in a request after the memory check. Unless the user chose.
    if env::var_os("CUDA_MODULE_LOADING").is_none() {
        // SAFETY: no other thread exists yet to read the environment.
        unsafe { env::set_var("CUDA_MODULE_LOADING", "EAGER") };
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let args = args()?;
    let api_key = env::var("SIPHON_API_KEY").ok().filter(|k| !k.is_empty());
    let vocab = chat::Vocab::load(&args.model_dir)?;
    let warmup = vocab
        .render(
            &[json!({"role": "user", "content": WARMUP})],
            &[],
            chat::DEFAULT_EFFORT,
        )
        .map_err(|e| anyhow::anyhow!("rendering the warmup prompt: {e}"))?;
    let sampling = request::SamplingDefaults::load(&args.model_dir)?;
    let engine = engine::start(&args.engine, vocab.stop_ids(), warmup)?;
    let app = Arc::new(openai::App {
        vocab,
        engine,
        metrics: Arc::new(metrics::Metrics::new(&args.name)),
        name: args.name,
        api_key,
        started: openai::started(),
        sampling,
        greedy: args.engine.greedy,
    });
    if app.api_key.is_none() && !args.address.ip().is_loopback() {
        eprintln!(
            "warning: listening on {} without SIPHON_API_KEY",
            args.address
        );
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(args.address)
            .await
            .with_context(|| format!("binding {}", args.address))?;
        eprintln!("serving {} on http://{}", app.name, args.address);
        axum::serve(listener, openai::router(app))
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
            .context("serving")
    })
}
