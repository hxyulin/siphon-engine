//! Shared helpers for the server tests: a `siphon-serve` process and HTTP requests to it.

#![allow(dead_code, reason = "each test binary uses a different subset")]

use std::{
    env,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context as _, ensure};

/// One HTTP/1.1 exchange: the status, the headers in lower case, and the body.
pub fn http(
    port: u16,
    method: &str,
    path: &str,
    body: &str,
) -> anyhow::Result<(u16, String, String)> {
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
pub struct Server {
    child: Child,
    pub port: u16,
    log: Arc<Mutex<Vec<String>>>,
}

impl Server {
    /// Starts `siphon-serve` on `SIPHON_MODEL` with `args`, and `--kv` from `SIPHON_KV` if
    /// set, and waits until it answers `/health`.
    pub fn start<S: AsRef<std::ffi::OsStr>>(args: &[S]) -> anyhow::Result<Self> {
        let model = env::var_os("SIPHON_MODEL").context("set SIPHON_MODEL")?;
        let port = free_port()?;
        let mut child = Command::new(env!("CARGO_BIN_EXE_siphon-serve"))
            .arg(&model)
            .args(["--port", &port.to_string()])
            .args(args)
            .args(
                env::var("SIPHON_KV")
                    .map(|kv| ["--kv".to_owned(), kv])
                    .into_iter()
                    .flatten(),
            )
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

    pub fn logged(&self, text: &str) -> bool {
        self.log
            .lock()
            .expect("log lock")
            .iter()
            .any(|l| l.contains(text))
    }

    pub fn wait(&mut self) -> anyhow::Result<std::process::ExitStatus> {
        Ok(self.child.wait()?)
    }

    pub fn terminate(&self) -> anyhow::Result<()> {
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
