//! The server's log on stderr: one line per event, each after a UTC timestamp to the
//! millisecond, and one `request` line per HTTP request when it ends ([`RequestLog`]).

use std::{
    fmt::{self, Write as _},
    sync::{Arc, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use axum::http::{Method, StatusCode};

use crate::{
    chat::Thinking,
    metrics::{Reason, Times},
    timings::Timings,
};

/// Writes one log line: the timestamp, then the `format!` arguments.
macro_rules! log {
    ($($arg:tt)*) => {
        $crate::log::line(format_args!($($arg)*))
    };
}
pub(crate) use log;

/// Writes `args` to stderr as one line after the current [`timestamp`].
pub fn line(args: fmt::Arguments<'_>) {
    let line = format!("{} {args}", timestamp(SystemTime::now()));
    #[cfg(test)]
    CAPTURED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(line.clone());
    eprintln!("{line}");
}

/// Every line logged so far, for tests.
#[cfg(test)]
pub static CAPTURED: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// `time` in UTC as `2026-11-27T10:15:02.123Z`.
pub fn timestamp(time: SystemTime) -> String {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = since.as_secs();
    let (days, of_day) = (secs / 86_400, secs % 86_400);
    let (year, month, day) = civil(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        of_day / 3600,
        of_day / 60 % 60,
        of_day % 60,
        since.subsec_millis()
    )
}

/// The year, month and day of `days` since 1970-01-01 (Howard Hinnant's
/// `civil_from_days`).
fn civil(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

/// What one HTTP request did, written as one `request` line when the last part of the
/// server that holds it lets go: the response body once sent, the stream task, and each
/// job's events ([`crate::openai::Results`]). So the line comes after the reply's last
/// byte and after its jobs have ended, and a client that leaves early still gets one.
#[derive(Debug)]
pub struct RequestLog {
    id: Arc<str>,
    method: Method,
    path: String,
    started: Instant,
    fields: Mutex<Fields>,
}

#[derive(Debug, Default)]
struct Fields {
    status: Option<StatusCode>,
    /// The error message of a response that is not a success, or of a failed job.
    error: Option<String>,
    /// Whether the client left before the response's end.
    disconnected: bool,
    /// Whether a keep-alive sent status 200 before the reply was ready.
    committed: bool,
    reply: Option<String>,
    stream: Option<bool>,
    thinking: Option<Thinking>,
    prompt: Option<usize>,
    /// Each ended job's events, generated tokens, cached prompt tokens and end.
    jobs: Vec<(Times, usize, usize, Reason)>,
    /// The engine's prefix cache tier and its summary of each job.
    tier: Option<&'static str>,
    engine: Vec<String>,
}

impl RequestLog {
    pub fn new(id: Arc<str>, method: Method, path: &str) -> Self {
        Self {
            id,
            method,
            path: path.to_owned(),
            started: Instant::now(),
            fields: Mutex::new(Fields::default()),
        }
    }

    fn fields(&self) -> std::sync::MutexGuard<'_, Fields> {
        self.fields
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The handler answered with `status`, and `error` for a response that is not a
    /// success.
    pub fn respond(&self, status: StatusCode, error: Option<&str>) {
        let mut f = self.fields();
        f.status = Some(status);
        if let Some(error) = error {
            f.error = Some(error.to_owned());
        }
    }

    /// The client left before the response's end.
    pub fn disconnected(&self) {
        self.fields().disconnected = true;
    }

    /// A keep-alive sent status 200 before the reply was ready, so the client got 200
    /// whatever `status` the reply turns out to have.
    pub fn committed(&self) {
        self.fields().committed = true;
    }

    /// The reply the request runs: its id, whether it streams, its thinking and its
    /// prompt's tokens.
    pub fn reply(&self, id: &str, stream: bool, thinking: Option<Thinking>, prompt: usize) {
        let mut f = self.fields();
        f.reply = Some(id.to_owned());
        f.stream = Some(stream);
        f.thinking = thinking;
        f.prompt = Some(prompt);
    }

    /// One job ended at `reason` after `times`, having generated `tokens` from a prompt of
    /// which `cached` tokens came from the prefix cache.
    pub fn job(&self, times: Times, tokens: usize, cached: usize, reason: Reason) {
        self.fields().jobs.push((times, tokens, cached, reason));
    }

    /// A job failed with `message`.
    pub fn failed(&self, message: &str) {
        self.fields()
            .error
            .get_or_insert_with(|| message.to_owned());
    }

    /// The engine's summary of a job: the prefix cache tier its prompt came from, if any,
    /// and what it did.
    #[cfg_attr(
        not(target_os = "linux"),
        expect(dead_code, reason = "only the Linux engine runs jobs")
    )]
    pub fn engine(&self, tier: Option<&'static str>, summary: String) {
        let mut f = self.fields();
        f.tier = f.tier.or(tier);
        f.engine.push(summary);
    }

    /// The `request` line, or `None` for a successful health check or metrics scrape,
    /// which would fill the log.
    fn line(&self) -> Option<String> {
        let f = self.fields();
        let quiet =
            self.method == Method::GET && matches!(self.path.as_str(), "/health" | "/metrics");
        let status = match f.status {
            _ if f.disconnected && f.status.is_none() => "499".to_owned(),
            Some(s) => s.as_u16().to_string(),
            None => "-".to_owned(),
        };
        if quiet && f.status.is_some_and(|s| s.is_success()) {
            return None;
        }
        let mut out = format!("request id={}", value(&self.id));
        let mut field = |key: &str, v: &dyn fmt::Display| {
            let _ = write!(out, " {key}={}", value(&v.to_string()));
        };
        field("route", &format_args!("{} {}", self.method, self.path));
        field("status", &status);
        field("ms", &self.started.elapsed().as_millis());
        if let Some(reply) = &f.reply {
            field("reply", reply);
        }
        if let Some(stream) = f.stream {
            field("stream", &stream);
        }
        match f.thinking {
            Some(Thinking::Effort(e)) => field("effort", &e),
            Some(Thinking::Off) => field("effort", &"off"),
            None => {}
        }
        if !f.jobs.is_empty() {
            let reasons: Vec<&str> = f.jobs.iter().map(|j| j.3.label()).collect();
            field("finish", &reasons.join(","));
        }
        if let Some(prompt) = f.prompt {
            let cached = f.jobs.iter().map(|j| j.2).max().unwrap_or(0);
            let output: usize = f.jobs.iter().map(|j| j.1).sum();
            let times: Vec<Times> = f.jobs.iter().map(|j| j.0).collect();
            field("prompt", &prompt);
            if !times.is_empty() {
                field("cached", &cached);
                if let Some(tier) = f.tier {
                    field("cache", &tier);
                }
                let t = Timings::new(&times, prompt, cached, output);
                field("queue_ms", &format_args!("{:.0}", t.queue_ms));
                if let Some(images) = t.images {
                    field("images", &images.encoded);
                    field("image_ms", &format_args!("{:.0}", images.ms));
                }
                field("prefill_ms", &format_args!("{:.0}", t.prompt_ms));
                field("output", &output);
                if t.predicted_ms > 0.0 {
                    #[expect(clippy::cast_precision_loss, reason = "token counts")]
                    let rate = output as f64 * 1e3 / t.predicted_ms;
                    field("tok_s", &format_args!("{rate:.1}"));
                }
                if t.rounds > 0 {
                    #[expect(clippy::cast_precision_loss, reason = "token counts")]
                    let per_round = t.round_tokens as f64 / t.rounds as f64;
                    field("per_round", &format_args!("{per_round:.2}"));
                }
            }
        }
        if !f.engine.is_empty() {
            field("engine", &f.engine.join("; "));
        }
        if f.committed {
            field("keepalive", &"committed");
        }
        if f.disconnected {
            field("client", &"disconnected");
        }
        if let Some(error) = &f.error {
            field("error", error);
        }
        Some(out)
    }
}

impl Drop for RequestLog {
    fn drop(&mut self) {
        if let Some(line) = self.line() {
            log!("{line}");
        }
    }
}

/// `s` as a logfmt value: bare when it holds no space, quote or `=`, else in JSON quotes.
fn value(s: &str) -> String {
    if !s.is_empty() && !s.contains([' ', '"', '=']) && !s.chars().any(char::is_control) {
        s.to_owned()
    } else {
        serde_json::Value::from(s).to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn timestamps_are_utc_to_the_millisecond() {
        let at = |secs: u64, ms: u64| {
            timestamp(UNIX_EPOCH + Duration::from_secs(secs) + Duration::from_millis(ms))
        };
        assert_eq!(at(0, 0), "1970-01-01T00:00:00.000Z");
        assert_eq!(at(951_782_400, 5), "2000-02-29T00:00:00.005Z");
        assert_eq!(at(1_795_775_702, 123), "2026-11-27T10:35:02.123Z");
        assert_eq!(at(4_107_542_399, 999), "2100-02-28T23:59:59.999Z");
    }

    #[test]
    fn values_are_quoted_only_when_needed() {
        assert_eq!(value("chatcmpl-1"), "chatcmpl-1");
        assert_eq!(value("POST /v1/x"), "\"POST /v1/x\"");
        assert_eq!(value("a=\"b\""), "\"a=\\\"b\\\"\"");
        assert_eq!(value(""), "\"\"");
    }

    #[test]
    fn rejected_requests_and_quiet_routes() {
        let log = RequestLog::new("req-1".into(), Method::POST, "/v1/chat/completions");
        log.respond(StatusCode::BAD_REQUEST, Some("reasoning_effort: bad"));
        let line = log.line().unwrap();
        assert!(
            line.starts_with("request id=req-1 route=\"POST /v1/chat/completions\" status=400 ms="),
            "{line}"
        );
        assert!(line.ends_with(" error=\"reasoning_effort: bad\""), "{line}");

        let health = RequestLog::new("req-2".into(), Method::GET, "/health");
        health.respond(StatusCode::OK, None);
        assert_eq!(health.line(), None);
        health.respond(StatusCode::SERVICE_UNAVAILABLE, None);
        assert!(health.line().unwrap().contains("status=503"));

        let gone = RequestLog::new("req-3".into(), Method::POST, "/v1/completions");
        gone.disconnected();
        let line = gone.line().unwrap();
        assert!(
            line.contains("status=499") && line.ends_with("client=disconnected"),
            "{line}"
        );
    }
}
