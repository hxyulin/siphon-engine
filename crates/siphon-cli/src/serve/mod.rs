//! `siphon serve`: an OpenAI-compatible chat server for Inkling 2 Mini on one GPU.

mod anthropic;
mod chat;
mod cli;
mod engine;
mod invalid;
mod keepalive;
mod log;
mod metrics;
mod openai;
mod parse;
mod piggyback;
mod request;
mod responses;
#[cfg(test)]
mod server_tests;
mod sse;
mod stop;
mod timings;
mod tokenize;
mod vision;

use std::{env, net::SocketAddr, process::ExitCode, sync::Arc, time::SystemTime};

use anyhow::Context as _;

use crate::serve::{
    cli::{Args, Backend},
    log::log,
};
use axum::serve::ListenerExt as _;
use clap::{ArgMatches, Command, CommandFactory};
use serde_json::json;

/// The commit the server is built from (`build.rs`).
pub const COMMIT: &str = env!("SIPHON_COMMIT");

/// The prompt that prices the verify widths at startup.
const WARMUP: &str = "Explain, step by step, how a hash map handles collisions, and compare \
                      separate chaining with open addressing in terms of memory use, cache \
                      behaviour and worst-case lookup time.";

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
    log!(
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
        log!(
            "drain: cut {left_running} running and {left_waiting} queued requests after {:.1} s",
            started.elapsed().as_secs_f64()
        );
        while busy(&engine) {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }
    log!(
        "drain: finished {} requests in {:.1} s",
        (running + waiting).saturating_sub(left_running + left_waiting),
        started.elapsed().as_secs_f64()
    );
    match tokio::task::spawn_blocking(move || engine.stop()).await {
        Ok(report) => log!("{report}"),
        Err(e) => log!("stopping the engine failed: {e}"),
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

/// Defines the HTTP server subcommand.
#[must_use]
pub fn command() -> Command {
    cli::Cli::command()
}

/// Starts the HTTP server with arguments parsed by [`command`].
///
/// `command` is the server subcommand from the parsed CLI so validation errors
/// can display its usage.
///
/// # Errors
///
/// Returns a Clap error when the parsed options cannot be used together.
pub fn run(
    command: &mut Command,
    matches: &ArgMatches,
    process_started: SystemTime,
) -> Result<ExitCode, clap::Error> {
    let args = Args::from_matches(command, matches)?;
    #[cfg(target_os = "linux")]
    if env::var_os("CUDA_MODULE_LOADING").is_none() {
        // Include the loaded kernels in the CUDA memory check.
        // SAFETY: the CLI has not started any threads.
        unsafe { env::set_var("CUDA_MODULE_LOADING", "EAGER") };
    }
    Ok(match serve(args, process_started) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log!("error: {e:#}");
            ExitCode::FAILURE
        }
    })
}

/// Wraps `listener` so every accepted TCP connection enables `TCP_NODELAY`, avoiding up to
/// 40 ms of Nagle + delayed-ACK jitter on small SSE chunks.
fn nodelay_listener(
    listener: tokio::net::TcpListener,
) -> impl axum::serve::Listener<Io = tokio::net::TcpStream, Addr = SocketAddr> {
    listener.tap_io(|stream| {
        let _ = stream.set_nodelay(true);
    })
}

fn serve(args: Args, process_started: SystemTime) -> anyhow::Result<()> {
    log!(
        "siphon serve {} (commit {COMMIT}) starting with {} on {}",
        env!("CARGO_PKG_VERSION"),
        args.model_dir.display(),
        args.backend.name(),
    );
    let api_key = env::var("SIPHON_API_KEY").ok().filter(|k| !k.is_empty());
    let mut vocab = chat::Vocab::load(&args.model_dir)?;
    if let Some(options) = &args.engine.vision {
        vocab.set_vision(Some(load_vision(
            &args.model_dir,
            options.max_pixels,
            args.max_images,
        )?));
    }
    let warmup = if args.backend == Backend::Cuda {
        vocab
            .render(
                &[json!({"role": "user", "content": WARMUP})],
                &[],
                chat::DEFAULT_EFFORT,
            )
            .map_err(|e| anyhow::anyhow!("rendering the warmup prompt: {e}"))?
    } else {
        Vec::new()
    };
    let sampling = request::SamplingDefaults::load(&args.model_dir)?;
    let engine = engine::start(&args.engine, vocab.stop_ids(), warmup)?;
    let app = Arc::new(openai::App {
        vocab,
        engine,
        metrics: Arc::new(metrics::Metrics::new(&args.name)),
        name: args.name,
        api_key,
        started: openai::started(),
        process_started,
        sampling,
        greedy: args.engine.greedy,
        keepalive: (!args.keepalive_after.is_zero()).then_some(keepalive::KeepAlive {
            after: args.keepalive_after,
            every: keepalive::EVERY,
        }),
    });
    if app.api_key.is_none() && !args.address.ip().is_loopback() {
        log!(
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
        log!("serving {} on http://{}", app.name, args.address);
        let engine = app.engine.clone();
        axum::serve(nodelay_listener(listener), openai::router(app))
            .with_graceful_shutdown(async move {
                stop_signal().await;
                drain(engine, args.drain_timeout).await;
            })
            .await
            .context("serving")
    })
}

#[cfg(test)]
mod tests {
    use axum::serve::Listener as _;

    use super::*;

    #[tokio::test]
    async fn accepted_connections_set_tcp_nodelay() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mut wrapped = nodelay_listener(listener);
        let client = tokio::spawn(async move { tokio::net::TcpStream::connect(addr).await });
        let (server_stream, _) = wrapped.accept().await;
        assert!(server_stream.nodelay().unwrap());
        let _ = client.await.unwrap().unwrap();
    }
}
