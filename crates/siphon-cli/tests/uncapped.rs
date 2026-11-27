//! Requests without a token cap, as the Codex CLI sends them, through the prefix cache: a
//! repeated prompt reads its state from the cache and gets the same reply, and a reply that
//! runs to the end of the context finishes with `length` and leaves the entry in place.
//!
//! Needs the checkpoint and a GPU, so it is ignored by default:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-cli --test uncapped -- --ignored --nocapture
//! ```
//!
//! `SIPHON_KV` passes its value as the server's `--kv`, to run the test with another cache
//! format.

#![cfg(target_os = "linux")]

mod common;

use std::path::PathBuf;

use anyhow::{Context as _, ensure};
use common::{Server, http};
use serde_json::{Value, json};

/// Positions of the server's KV pool: small, so that a reply reaches the end in seconds.
const CONTEXT: &str = "8192";

/// A chat completion's reply, which must succeed.
fn chat(port: u16, request: &Value) -> anyhow::Result<Value> {
    let (status, _, body) = http(port, "POST", "/v1/chat/completions", &request.to_string())?;
    ensure!(status == 200, "the request got {status}: {body}");
    Ok(serde_json::from_str(&body)?)
}

fn usage(reply: &Value, field: &str) -> anyhow::Result<u64> {
    reply["usage"][field]
        .as_u64()
        .with_context(|| format!("no usage.{field}: {}", reply["usage"]))
}

fn cached(reply: &Value) -> Option<u64> {
    reply["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64()
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn uncapped_requests_read_the_prefix_cache() -> anyhow::Result<()> {
    let server = Server::start(&["--context", CONTEXT, "--greedy"])?;
    let port = server.port;
    let (_, _, models) = http(port, "GET", "/v1/models", "")?;
    let models: Value = serde_json::from_str(&models)?;
    let max_context = models["data"][0]["max_model_len"]
        .as_u64()
        .context("no max_model_len")?;

    // About 3,000 tokens of the repository's docs, without `max_tokens`.
    let docs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/engine.md");
    let text: String = std::fs::read_to_string(docs)?
        .chars()
        .take(12_000)
        .collect();
    let ask = json!({
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": format!("{text}\n\nSummarize this in one paragraph.")}],
    });
    let first = chat(port, &ask)?;
    let prompt = usage(&first, "prompt_tokens")?;
    let boundary = (prompt - 1) / 2048 * 2048;
    println!("first: usage {}", first["usage"]);

    let again = chat(port, &ask)?;
    println!("again: usage {}", again["usage"]);
    ensure!(
        cached(&again) == Some(boundary),
        "the repeated prompt read {:?} tokens from the cache, not {boundary}",
        cached(&again)
    );
    ensure!(
        again["choices"][0]["message"] == first["choices"][0]["message"],
        "the reply from the cache differs"
    );

    // A reply that runs to the end of the context.
    let mut long = ask.clone();
    long["ignore_eos"] = json!(true);
    let full = chat(port, &long)?;
    println!(
        "to the end: usage {}, finish {}",
        full["usage"], full["choices"][0]["finish_reason"]
    );
    ensure!(
        cached(&full) == Some(boundary),
        "the long reply missed the cache"
    );
    ensure!(
        full["choices"][0]["finish_reason"] == "length",
        "the reply at the end of the context finished with {}",
        full["choices"][0]["finish_reason"]
    );
    ensure!(
        prompt + usage(&full, "completion_tokens")? == max_context,
        "the reply stopped before the end of the context: {}",
        full["usage"]
    );

    // Its entry stays, and still gives the same reply.
    let after = chat(port, &ask)?;
    println!("after: usage {}", after["usage"]);
    ensure!(
        cached(&after) == Some(boundary),
        "after the long reply the prompt read {:?} tokens from the cache",
        cached(&after)
    );
    ensure!(
        after["choices"][0]["message"] == first["choices"][0]["message"],
        "the reply after the long reply differs"
    );
    Ok(())
}
