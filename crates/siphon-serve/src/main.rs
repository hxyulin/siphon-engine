//! `siphon-serve`: an OpenAI-compatible chat server for Inkling 2 Mini on one GPU.

mod chat;
mod engine;
mod openai;
mod parse;

use std::{env, net::SocketAddr, path::PathBuf, process::ExitCode, sync::Arc};

use anyhow::{Context as _, bail, ensure};
use serde_json::json;

const USAGE: &str = "usage: siphon-serve <model-dir> [--host 127.0.0.1] [--port 8000] \
                     [--context 65536] [--weights fp8|bf16] [--kv fp8|bf16] [--queue 8] \
                     [--widths 1,2,...,8] [--name inkling-2-mini]
Set SIPHON_API_KEY to require `Authorization: Bearer <key>` on /v1 routes.";

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
    let mut capacity = 65536;
    let mut fp8_weights = true;
    let mut fp8_kv = true;
    let mut queue = 8;
    let mut widths: Vec<usize> = (1..=8).collect();
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
            "--context" => capacity = value()?.parse()?,
            "--weights" => fp8_weights = fp8(&value()?)?,
            "--kv" => fp8_kv = fp8(&value()?)?,
            "--queue" => queue = value()?.parse()?,
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
            fp8_weights,
            fp8_kv,
            widths,
            queue,
        },
        model_dir,
        address,
        name,
    })
}

fn fp8(format: &str) -> anyhow::Result<bool> {
    match format {
        "fp8" => Ok(true),
        "bf16" => Ok(false),
        _ => bail!("formats are fp8 or bf16, not {format}"),
    }
}

fn main() -> ExitCode {
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
    let engine = engine::start(&args.engine, vocab.stop_ids(), warmup)?;
    let app = Arc::new(openai::App {
        vocab,
        engine,
        name: args.name,
        api_key,
        started: openai::started(),
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
