//! A deploy through `siphon-serve`'s drain and `--cache-dir`: on SIGTERM the server finishes
//! the running request, answers a new one and `/health` with 503 and `Retry-After`, saves
//! its host cache and exits; the next server loads the cache, and the same request then
//! reads its prompt from it and gets the same reply.
//!
//! Needs the checkpoint and a GPU, so it is ignored by default:
//!
//! ```sh
//! SIPHON_MODEL=<model-dir> cargo test --release -p siphon-serve --test drain -- --ignored --nocapture
//! ```

#![cfg(target_os = "linux")]

use std::{
    env,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context as _, ensure};
use serde_json::{Value, json};

/// One HTTP/1.1 exchange: the status, the headers in lower case, and the body.
fn http(port: u16, method: &str, path: &str, body: &str) -> anyhow::Result<(u16, String, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )?;
    let mut reply = String::new();
    stream.read_to_string(&mut reply)?;
    let (head, body) = reply.split_once("\r\n\r\n").context("no header end")?;
    let status = head
        .split(' ')
        .nth(1)
        .context("no status")?
        .parse()
        .context("a bad status")?;
    Ok((status, head.to_lowercase(), body.to_owned()))
}

/// A free local port.
fn free_port() -> anyhow::Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

/// A running server and the lines of its log so far.
struct Server {
    child: Child,
    port: u16,
    log: Arc<Mutex<Vec<String>>>,
}

impl Server {
    fn start(cache: &Path) -> anyhow::Result<Self> {
        let model = env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?;
        let port = free_port()?;
        let mut child = Command::new(env!("CARGO_BIN_EXE_siphon-serve"))
            .arg(&model)
            .args([
                "--port",
                &port.to_string(),
                "--context",
                "32768",
                "--greedy",
            ])
            .args(["--host-cache", "2", "--cache-dir"])
            .arg(cache)
            .stderr(Stdio::piped())
            .spawn()?;
        let log = Arc::new(Mutex::new(Vec::new()));
        let stderr = child.stderr.take().context("no stderr")?;
        let lines = Arc::clone(&log);
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                eprintln!("[server {port}] {line}");
                lines.lock().expect("log lock").push(line);
            }
        });
        let started = Instant::now();
        while !matches!(http(port, "GET", "/health", ""), Ok((200, ..))) {
            ensure!(
                child.try_wait()?.is_none(),
                "the server exited while loading"
            );
            ensure!(
                started.elapsed() < Duration::from_secs(600),
                "the server did not start"
            );
            thread::sleep(Duration::from_millis(500));
        }
        Ok(Self { child, port, log })
    }

    fn logged(&self, text: &str) -> bool {
        self.log
            .lock()
            .expect("log lock")
            .iter()
            .any(|l| l.contains(text))
    }

    fn wait(&mut self) -> anyhow::Result<std::process::ExitStatus> {
        Ok(self.child.wait()?)
    }

    fn terminate(&self) -> anyhow::Result<()> {
        let status = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status()?;
        ensure!(status.success(), "kill failed");
        Ok(())
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
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
    let mut first = Server::start(&cache)?;
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

    let second = Server::start(&cache)?;
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
