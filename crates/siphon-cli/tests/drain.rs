//! A deploy through the server's drain and `--cache-dir`: on SIGTERM the server finishes
//! the running request, answers a new one and `/health` with 503 and `Retry-After`, saves
//! its host cache and exits; the next server loads the cache, and the same request then
//! reads its prompt from it and gets the same reply.
//!
//! Needs the checkpoint and a GPU, so it is ignored by default:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cli --test drain -- --ignored --nocapture
//!
//! `SIPHON_KV` passes its value as the server's `--kv`, to run the test with another cache format.
//! ```

#![cfg(target_os = "linux")]

mod common;

use std::{
    env,
    ffi::OsString,
    path::{Path, PathBuf},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context as _, ensure};
use common::{Server, http};
use serde_json::{Value, json};

/// A server with a host cache that it saves to and loads from `cache`.
fn start(cache: &Path) -> anyhow::Result<Server> {
    let mut args: Vec<OsString> = ["--context", "32768", "--greedy", "--host-cache", "2"]
        .map(OsString::from)
        .into();
    args.extend([OsString::from("--cache-dir"), cache.into()]);
    Server::start(&args)
}

/// A chat request of about 3,000 tokens of the repository's docs, for `max_tokens` tokens.
fn request(max_tokens: usize) -> anyhow::Result<String> {
    let docs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/engine.md");
    let text: String = std::fs::read_to_string(docs)?
        .chars()
        .take(12_000)
        .collect();
    Ok(json!({
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": format!("{text}\n\nSummarize this.")}],
        "max_tokens": max_tokens,
        "ignore_eos": true,
    })
    .to_string())
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn drain_then_restart_from_disk() -> anyhow::Result<()> {
    let cache = env::temp_dir().join(format!("siphon-drain-test-{}", std::process::id()));
    let mut first = start(&cache)?;
    let port = first.port;
    let body = request(1500)?;
    let running = {
        let body = body.clone();
        thread::spawn(move || http(port, "POST", "/v1/chat/completions", &body))
    };
    // Until the request runs, then a second into its generation.
    let started = Instant::now();
    let running_gauge = "vllm:num_requests_running{model_name=\"inkling-2-mini\"} 1";
    while !matches!(http(port, "GET", "/metrics", ""), Ok((200, _, m)) if m.contains(running_gauge))
    {
        ensure!(
            started.elapsed() < Duration::from_secs(60),
            "the request did not start"
        );
        thread::sleep(Duration::from_millis(20));
    }
    thread::sleep(Duration::from_secs(1));
    first.terminate()?;
    let started = Instant::now();
    while !first.logged("draining:") {
        ensure!(
            started.elapsed() < Duration::from_secs(10),
            "no drain started"
        );
        thread::sleep(Duration::from_millis(10));
    }
    let (status, ..) = http(port, "GET", "/health", "")?;
    ensure!(status == 503, "/health answered {status} while draining");
    let (status, head, reply) = http(port, "POST", "/v1/chat/completions", &request(8)?)?;
    println!("a new request while draining: {status}, {reply}");
    ensure!(status == 503, "a new request got {status} while draining");
    ensure!(head.contains("retry-after: 1"), "no Retry-After: {head}");

    let (status, _, reply) = running.join().expect("the request thread")?;
    ensure!(status == 200, "the running request got {status}: {reply}");
    let reply: Value = serde_json::from_str(&reply)?;
    ensure!(
        reply["usage"]["completion_tokens"] == 1500,
        "the running request was cut: {}",
        reply["usage"]
    );
    let exit = first.wait()?;
    ensure!(exit.success(), "the server exited with {exit}");
    for line in ["drain: finished 1 requests", "host cache: saved"] {
        ensure!(first.logged(line), "the log lacks {line:?}");
    }

    let second = start(&cache)?;
    ensure!(
        second.logged("host cache: loaded 1 entries"),
        "the cache did not load"
    );
    let (status, _, again) = http(second.port, "POST", "/v1/chat/completions", &body)?;
    ensure!(status == 200, "the request after the restart got {status}");
    let again: Value = serde_json::from_str(&again)?;
    println!("after the restart: usage {}", again["usage"]);
    let prompt = reply["usage"]["prompt_tokens"]
        .as_u64()
        .context("no prompt tokens")?;
    ensure!(
        again["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64()
            == Some((prompt - 1) / 2048 * 2048),
        "the prompt did not come from the saved cache"
    );
    ensure!(
        again["choices"][0]["message"] == reply["choices"][0]["message"],
        "the reply after the restart differs"
    );
    drop(second);
    std::fs::remove_dir_all(&cache)?;
    Ok(())
}
