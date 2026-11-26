//! `siphon-serve`: an OpenAI-compatible chat server for Inkling 2 Mini on one GPU.

mod anthropic;
mod chat;
mod engine;
mod metrics;
mod openai;
mod parse;
mod request;
#[cfg(test)]
mod server_tests;
mod stop;
mod tokenize;
mod vision;

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

/// Tokens per second each decoding request gets at least while another prefills.
const DECODE_FLOOR: f64 = 20.0;

const USAGE: &str = "usage: siphon-serve <model-dir> [--host 127.0.0.1] [--port 8000] \
                     [--context N] [--memory GiB] [--margin GiB] [--prefill-chunk 2048] [--weights fp8|bf16] \
                     [--output-projection bf16|fp8] [--kv fp8|bf16|nvfp4|mixed] [--streams 1] \
                     [--queue 8] [--widths 1,2,...,8] [--name inkling-2-mini] [--drafter DIR] [--draft fast] \
                     [--fallback] [--no-prefix-cache] [--prefix-cache-entries N] [--prefix-cache-ttl 600] \
                     [--host-cache GiB] [--cache-dir DIR] [--drain-timeout 600] [--prefill-attention auto|bf16|fp8] \
                     [--decode-floor 20] [--greedy] [--vision] \
                     [--max-image-pixels 2073600] [--max-images 16]
--context sets the positions of the KV cache pool: the prompts and generated tokens of every
running request together. One request may take all of them, up to the model's limit of
1,048,576. Without it the server takes the most that fit in GPU memory. Either way it plans
the memory before loading the weights and stops if the plan does not fit.
--streams runs up to N requests at once (1 to 32), as many as the pool holds; each stream
costs its sliding rings and prefix cache entry, about 0.33 GiB. A request alone speculates;
once another arrives, all run batched plain decode. Either way a request gets the same
tokens. With --host-cache, requests that do not fit the pool together take turns, swapped
out to host memory and back. Up to --queue more requests wait.
--decode-floor gives each decoding request at least that many tokens per second while
another request's prompt prefills, by running more batched steps between its chunks of
--prefill-chunk tokens. The prefill keeps at least half of the time. 0 runs one step per chunk.
--memory caps the GPU memory the plan may use, the CUDA context included. Without it the
plan may use the free memory at startup less --margin, 0.25 GiB by default.
--prefill-chunk runs prompts in chunks of N rows, 1 to 2048. Its buffers grow with N: 0.85 GiB
at 2048 rows and 0.47 GiB at 1024, and the memory they free goes to the context.
--output-projection fp8 makes decode read an FP8 copy of the attention output projection with
--weights fp8, which saves 4 to 6% of a plain step but moves the logits further from the
model; bf16 (the default) reads its BF16 weights.
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
--host-cache keeps entries in that many GiB of page-locked host memory too (0, the default,
for none), global KV cache included, so a conversation whose stream another request took
restores over PCIe instead of prefilling again. The memory is allocated at startup. An
entry takes 16 KiB per token with the FP8 KV cache (32 KiB with BF16), plus the bundled
draft's 2 KiB, plus 0.2 GiB; conversations that share a prefix share its memory, and the
least recently used entries go when it is full.
--cache-dir saves the host cache's entries to DIR when the server stops, with the
conversations the streams hold, and loads them back at startup, so that the first turn after
a restart restores instead of prefilling. It needs --host-cache. A directory saved with
another checkpoint, drafter, format, draft or build of the kernels is discarded.
On SIGTERM or Ctrl-C the server drains: it answers new requests with 503 and Retry-After,
/health turns 503, and it finishes the running and queued requests. After --drain-timeout
seconds it ends what is left with an error. Then it saves --cache-dir and exits.
--prefill-attention picks prefill's global attention products over an FP8 cache. auto (the
default) runs BF16 up to 16,384 keys and FP8 tensor cores past them, where they cut time to
first token by up to a fifth; bf16 keeps every prompt as before, and fp8 runs FP8 throughout.
--vision loads the vision tower and takes images in requests: base64 data: URLs in OpenAI
image_url parts and base64 image blocks in Anthropic messages. Image URLs are not fetched.
--max-image-pixels shrinks a larger image to this many pixels, keeping its aspect ratio
(default 1920 x 1080 = 2,073,600, up to 1,369 image tokens). --max-images refuses a request
with more images than this (default 16).
--kv nvfp4 stores the global KV caches as NVFP4 (E2M1 values, one E4M3 scale per 16): 9 KiB
per position instead of 16 with fp8, at about twice FP8's divergence from a BF16 cache.
--kv mixed keeps the first two global visits in FP8 and the other six in NVFP4: 10.75 KiB per
position, at about FP8's divergence.
--kv-quant nvfp4|q4_0 [--kv-rotate] [--kv-quant-visits V,...]
rounds the global keys and values through a 4-bit format with a BF16 cache (--kv bf16), to
measure the quality of a 4-bit cache; it saves no memory.
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
    /// How long a drain waits for the running and queued requests.
    drain_timeout: std::time::Duration,
    /// `--max-images`, with `--vision`.
    max_images: usize,
}

#[expect(clippy::too_many_lines, reason = "one match arm per flag")]
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
    let mut fp8_output = false;
    let mut kv = "fp8".to_owned();
    let mut queue = 8;
    let mut streams = 1;
    let mut widths: Vec<usize> = (1..=8).collect();
    let mut drafter = None;
    let mut draft = "fast".to_owned();
    let mut fallback = false;
    let mut prefix_cache = true;
    let mut prefix_entries = None;
    let mut prefix_ttl = 600;
    let mut host_cache = 0;
    let mut cache_dir = None;
    let mut drain_timeout = 600;
    let mut prefill_attention = "auto".to_owned();
    let mut greedy = false;
    let mut decode_floor = DECODE_FLOOR;
    let mut vision = false;
    let mut max_image_pixels = siphon::vision::DEFAULT_MAX_PIXELS;
    let mut max_images = vision::DEFAULT_MAX_IMAGES;
    let mut kv_args = Vec::new();
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
            "--output-projection" => fp8_output = fp8(&value()?)?,
            "--kv" => {
                kv = value()?;
                ensure!(
                    ["fp8", "bf16", "nvfp4", "mixed"].contains(&kv.as_str()),
                    "--kv takes fp8, bf16, nvfp4 or mixed"
                );
            }
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
            "--host-cache" => host_cache = gib_bytes(&value()?, "--host-cache", true)?,
            "--cache-dir" => cache_dir = Some(PathBuf::from(value()?)),
            "--drain-timeout" => drain_timeout = value()?.parse()?,
            "--prefill-attention" => prefill_attention = value()?,
            "--greedy" => greedy = true,
            "--vision" => vision = true,
            "--max-image-pixels" => {
                max_image_pixels = value()?
                    .parse()
                    .context("--max-image-pixels takes a pixel count")?;
                ensure!(
                    max_image_pixels >= 1600,
                    "--max-image-pixels takes at least 1600"
                );
            }
            "--max-images" => {
                max_images = value()?.parse().context("--max-images takes a count")?;
                ensure!(max_images >= 1, "--max-images takes at least 1");
            }
            "--decode-floor" => {
                decode_floor = value()?
                    .parse()
                    .context("--decode-floor takes tokens per second")?;
                ensure!(
                    decode_floor >= 0.0,
                    "--decode-floor takes 0 or more tokens per second"
                );
            }
            "--kv-quant" | "--kv-quant-visits" => {
                let v = value()?;
                kv_args.extend([arg, v]);
            }
            "--kv-rotate" => kv_args.push(arg),
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
            fp8_output,
            kv,
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
            host_cache,
            cache_dir,
            prefill_attention,
            greedy,
            decode_floor,
            vision: vision.then_some(engine::VisionOptions {
                max_pixels: max_image_pixels,
            }),
            kv_args,
        },
        max_images,
        model_dir,
        address,
        name,
        drain_timeout: std::time::Duration::from_secs(drain_timeout),
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

/// Waits for SIGTERM or Ctrl-C.
async fn stop_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = ctrl_c => {}
                    _ = term.recv() => {}
                }
            }
            Err(_) => {
                let _ = ctrl_c.await;
            }
        }
    }
    #[cfg(not(unix))]
    let _ = ctrl_c.await;
}

/// Drains the engine before the server stops: refuses new requests, waits up to `timeout`
/// for the running and queued ones, cancels what is left, then stops the engine, which
/// saves the host cache to `--cache-dir`. Logs each step.
async fn drain(engine: engine::Engine, timeout: std::time::Duration) {
    engine.drain();
    let started = std::time::Instant::now();
    let (running, waiting) = engine.counts();
    eprintln!(
        "draining: refusing new requests, finishing {running} running and {waiting} queued \
         within {} s",
        timeout.as_secs()
    );
    let busy = |e: &engine::Engine| {
        let (r, w) = e.counts();
        r + w > 0
    };
    while busy(&engine) && started.elapsed() < timeout {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let (left_running, left_waiting) = engine.counts();
    if left_running + left_waiting > 0 {
        engine.cancel();
        eprintln!(
            "drain: cut {left_running} running and {left_waiting} queued requests after {:.1} s",
            started.elapsed().as_secs_f64()
        );
        while busy(&engine) {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }
    eprintln!(
        "drain: finished {} requests in {:.1} s",
        (running + waiting).saturating_sub(left_running + left_waiting),
        started.elapsed().as_secs_f64()
    );
    match tokio::task::spawn_blocking(move || engine.stop()).await {
        Ok(report) => eprintln!("{report}"),
        Err(e) => eprintln!("stopping the engine failed: {e}"),
    }
}

fn fp8(format: &str) -> anyhow::Result<bool> {
    match format {
        "fp8" => Ok(true),
        "bf16" => Ok(false),
        _ => bail!("formats are fp8 or bf16, not {format}"),
    }
}

/// Image preprocessing for `--vision`, shrinking images to `--max-image-pixels`.
fn load_vision(
    dir: &std::path::Path,
    max_pixels: usize,
    max_images: usize,
) -> anyhow::Result<vision::Vision> {
    let mut config = siphon::vision::VisionConfig::load(dir)?;
    config.max_pixels = max_pixels;
    let (side, placeholder) = (config.token_size(), config.image_token_id);
    Ok(vision::Vision::new(
        Box::new(config),
        side,
        placeholder,
        max_pixels,
        max_images,
    ))
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
    let mut vocab = chat::Vocab::load(&args.model_dir)?;
    if let Some(options) = &args.engine.vision {
        vocab.set_vision(Some(load_vision(
            &args.model_dir,
            options.max_pixels,
            args.max_images,
        )?));
    }
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
        let engine = app.engine.clone();
        axum::serve(listener, openai::router(app))
            .with_graceful_shutdown(async move {
                stop_signal().await;
                drain(engine, args.drain_timeout).await;
            })
            .await
            .context("serving")
    })
}
