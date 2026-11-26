//! Prometheus metrics for `GET /metrics`, under vLLM's names, so that dashboards and
//! benchmarks written for vLLM read them.
//!
//! The HTTP layer records them from each job's events ([`Tracker`]), so they count what
//! the engine sent: generated tokens include the ones a stop string later cut off.
//!
//! Each choice of a request with `n` choices is its own job, and counts as a request in
//! the request counts, gauges and latency histograms. Its prompt counts once, with the
//! first choice, in the prompt tokens and the prefix cache lookups.

use std::{
    fmt::Write as _,
    sync::{Arc, Mutex},
    time::Instant,
};

use crate::vision::ImageStats;

/// vLLM's bucket edges for time to first token, in seconds.
const TTFT_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
    20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
];

/// vLLM's bucket edges for the gap between tokens, below which this server adds edges
/// from 0.5 ms: a token takes 1 to 3 ms here, under vLLM's smallest edge of 10 ms.
const ITL_BUCKETS: &[f64] = &[
    0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15,
    0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
];

/// vLLM's bucket edges for whole-request and queue times, in seconds.
const REQUEST_BUCKETS: &[f64] = &[
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0,
    480.0, 960.0, 1920.0, 7680.0,
];

/// Why a request ended, as vLLM's `finished_reason` label names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The model ended its turn, or a stop string or stop token matched.
    Stop,
    Length,
    /// The client left before the end.
    Abort,
    Error,
}

impl Reason {
    const ALL: [Self; 4] = [Self::Stop, Self::Length, Self::Abort, Self::Error];

    fn label(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Abort => "abort",
            Self::Error => "error",
        }
    }
}

/// A cumulative histogram with fixed upper edges.
#[derive(Clone, Debug)]
struct Histogram {
    edges: &'static [f64],
    /// Observations per bucket, the last past every edge.
    counts: Vec<u64>,
    sum: f64,
}

impl Histogram {
    fn new(edges: &'static [f64]) -> Self {
        Self {
            edges,
            counts: vec![0; edges.len() + 1],
            sum: 0.0,
        }
    }

    /// Records `value` `times` times.
    fn observe(&mut self, value: f64, times: u64) {
        let bucket = self.edges.partition_point(|&edge| edge < value);
        self.counts[bucket] += times;
        #[expect(clippy::cast_precision_loss, reason = "token counts per event")]
        let times = times as f64;
        self.sum += value * times;
    }

    fn write(&self, out: &mut String, name: &str, help: &str, labels: &str) {
        header(out, name, help, "histogram");
        let mut total = 0;
        for (i, count) in self.counts.iter().enumerate() {
            total += count;
            let edge = self
                .edges
                .get(i)
                .map_or_else(|| "+Inf".to_owned(), f64::to_string);
            let _ = writeln!(out, "{name}_bucket{{{labels},le=\"{edge}\"}} {total}");
        }
        let _ = writeln!(out, "{name}_count{{{labels}}} {total}");
        let _ = writeln!(out, "{name}_sum{{{labels}}} {}", self.sum);
    }
}

#[derive(Clone, Debug)]
struct State {
    prompt_tokens: u64,
    generation_tokens: u64,
    /// Prompt tokens looked up in the prefix cache, and the ones found.
    cache_queries: u64,
    cache_hits: u64,
    /// Finished requests by [`Reason::ALL`] index.
    finished: [u64; 4],
    waiting: u64,
    running: u64,
    queue: Histogram,
    ttft: Histogram,
    itl: Histogram,
    e2e: Histogram,
}

/// The server's counters, gauges and histograms.
#[derive(Debug)]
pub struct Metrics {
    /// The `model_name` label, escaped.
    labels: String,
    state: Mutex<State>,
}

impl Metrics {
    pub fn new(model: &str) -> Self {
        let escaped = model
            .replace('\\', r"\\")
            .replace('"', "\\\"")
            .replace('\n', r"\n");
        Self {
            labels: format!("model_name=\"{escaped}\""),
            state: Mutex::new(State {
                prompt_tokens: 0,
                generation_tokens: 0,
                cache_queries: 0,
                cache_hits: 0,
                finished: [0; 4],
                waiting: 0,
                running: 0,
                queue: Histogram::new(REQUEST_BUCKETS),
                ttft: Histogram::new(TTFT_BUCKETS),
                itl: Histogram::new(ITL_BUCKETS),
                e2e: Histogram::new(REQUEST_BUCKETS),
            }),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("metrics lock poisoned")
    }

    /// Starts tracking a request of `prompt` tokens that the engine queued: 0 for every
    /// choice of an n-choice request but the first.
    pub fn track(self: &Arc<Self>, prompt: usize) -> Tracker {
        self.state().waiting += 1;
        Tracker {
            metrics: Arc::clone(self),
            prompt: prompt as u64,
            phase: Phase::Waiting,
            cached: 0,
            last: None,
            times: Times::new(Instant::now()),
        }
    }

    /// The Prometheus text exposition of every metric.
    pub fn render(&self) -> String {
        // Snapshot under lock so Prometheus formatting never stalls concurrent token tracking.
        let s = self.state().clone();
        let labels = &self.labels;
        let mut out = String::with_capacity(4096);
        let counters = [
            (
                "vllm:prompt_tokens_total",
                "Prompt tokens of requests that reached their first token.",
                s.prompt_tokens,
            ),
            (
                "vllm:generation_tokens_total",
                "Tokens generated.",
                s.generation_tokens,
            ),
            (
                "vllm:prefix_cache_queries_total",
                "Prompt tokens looked up in the prefix cache.",
                s.cache_queries,
            ),
            (
                "vllm:prefix_cache_hits_total",
                "Prompt tokens whose state came from the prefix cache.",
                s.cache_hits,
            ),
        ];
        for (name, help, value) in counters {
            header(&mut out, name, help, "counter");
            let _ = writeln!(out, "{name}{{{labels}}} {value}");
        }
        let name = "vllm:request_success_total";
        header(&mut out, name, "Requests finished, by reason.", "counter");
        for (reason, count) in Reason::ALL.iter().zip(s.finished) {
            let reason = reason.label();
            let _ = writeln!(
                out,
                "{name}{{{labels},finished_reason=\"{reason}\"}} {count}"
            );
        }
        let gauges = [
            (
                "vllm:num_requests_running",
                "Requests the engine has started.",
                s.running,
            ),
            (
                "vllm:num_requests_waiting",
                "Requests queued for the engine.",
                s.waiting,
            ),
        ];
        for (name, help, value) in gauges {
            header(&mut out, name, help, "gauge");
            let _ = writeln!(out, "{name}{{{labels}}} {value}");
        }
        s.queue.write(
            &mut out,
            "vllm:request_queue_time_seconds",
            "Seconds from queueing to the engine starting the request.",
            labels,
        );
        s.ttft.write(
            &mut out,
            "vllm:time_to_first_token_seconds",
            "Seconds from queueing to the first token.",
            labels,
        );
        s.itl.write(
            &mut out,
            "vllm:inter_token_latency_seconds",
            "Seconds per token after the first: a gap that delivered n tokens counts n \
             times as gap / n.",
            labels,
        );
        s.e2e.write(
            &mut out,
            "vllm:e2e_request_latency_seconds",
            "Seconds from queueing to the end of the request.",
            labels,
        );
        out
    }
}

fn header(out: &mut String, name: &str, help: &str, kind: &str) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Waiting,
    Running,
    Finished,
}

/// When one job's events arrived, and how it speculated, for the request's `timings`
/// ([`crate::timings`]).
#[derive(Clone, Copy, Debug)]
pub struct Times {
    /// When the HTTP layer queued the job, after it parsed and tokenized the request.
    pub submitted: Instant,
    /// When the engine took the job from its queue.
    pub started: Option<Instant>,
    /// When the first tokens arrived.
    pub first: Option<Instant>,
    /// When the job ended.
    pub ended: Option<Instant>,
    /// Speculation rounds, and the tokens they delivered.
    pub rounds: u64,
    pub round_tokens: u64,
    /// What the vision tower did for a prompt with images.
    pub images: Option<ImageStats>,
    /// Whether the next tokens come from a speculation round.
    pub(crate) round: bool,
}

impl Times {
    /// A job submitted at `submitted` that has had no events yet.
    pub fn new(submitted: Instant) -> Self {
        Self {
            submitted,
            started: None,
            first: None,
            ended: None,
            rounds: 0,
            round_tokens: 0,
            images: None,
            round: false,
        }
    }
}

/// One request's part in the metrics. Dropping it before [`Tracker::finish`] counts the
/// request as aborted.
#[derive(Debug)]
pub struct Tracker {
    metrics: Arc<Metrics>,
    prompt: u64,
    phase: Phase,
    cached: u64,
    /// When the last tokens arrived.
    last: Option<Instant>,
    times: Times,
}

impl Tracker {
    /// The engine took the request from its queue.
    pub fn started(&mut self) {
        self.started_at(Instant::now());
    }

    fn started_at(&mut self, now: Instant) {
        if self.phase != Phase::Waiting {
            return;
        }
        self.phase = Phase::Running;
        self.times.started = Some(now);
        let mut s = self.metrics.state();
        s.waiting -= 1;
        s.running += 1;
        s.queue
            .observe((now - self.times.submitted).as_secs_f64(), 1);
    }

    /// The next tokens come from a speculation round.
    pub fn round(&mut self) {
        self.times.round = true;
    }

    /// When the job's events arrived so far.
    pub fn times(&self) -> Times {
        self.times
    }

    /// Prompt tokens whose state came from the prefix cache.
    pub fn cached(&mut self, tokens: usize) {
        self.cached = tokens as u64;
    }

    /// What the vision tower did for the prompt.
    pub fn images(&mut self, stats: ImageStats) {
        self.times.images = Some(stats);
    }

    /// `n` tokens arrived.
    pub fn tokens(&mut self, n: usize) {
        self.tokens_at(n, Instant::now());
    }

    fn tokens_at(&mut self, n: usize, now: Instant) {
        if n == 0 || self.phase == Phase::Finished {
            return;
        }
        self.started_at(now);
        let n = n as u64;
        if std::mem::take(&mut self.times.round) {
            self.times.rounds += 1;
            self.times.round_tokens += n;
        }
        self.times.first.get_or_insert(now);
        let mut s = self.metrics.state();
        s.generation_tokens += n;
        match self.last {
            None => {
                s.ttft
                    .observe((now - self.times.submitted).as_secs_f64(), 1);
                s.prompt_tokens += self.prompt;
                s.cache_queries += self.prompt;
                // A later choice of an n-choice request counts no prompt, so none of its
                // cached tokens either.
                s.cache_hits += self.cached.min(self.prompt);
            }
            Some(last) => {
                #[expect(clippy::cast_precision_loss, reason = "token counts per event")]
                let per_token = (now - last).as_secs_f64() / n as f64;
                s.itl.observe(per_token, n);
            }
        }
        self.last = Some(now);
    }

    /// The request ended; later calls do nothing.
    pub fn finish(&mut self, reason: Reason) {
        self.finish_at(reason, Instant::now());
    }

    fn finish_at(&mut self, reason: Reason, now: Instant) {
        let mut s = self.metrics.state();
        match self.phase {
            Phase::Finished => return,
            Phase::Waiting => s.waiting -= 1,
            Phase::Running => s.running -= 1,
        }
        self.phase = Phase::Finished;
        self.times.ended = Some(now);
        let index = Reason::ALL
            .iter()
            .position(|&r| r == reason)
            .expect("ALL lists every reason");
        s.finished[index] += 1;
        s.e2e.observe((now - self.times.submitted).as_secs_f64(), 1);
    }
}

impl Drop for Tracker {
    fn drop(&mut self) {
        self.finish(Reason::Abort);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    /// The value of the line that starts with `series` followed by a space.
    fn value(text: &str, series: &str) -> f64 {
        text.lines()
            .find_map(|line| line.strip_prefix(series)?.strip_prefix(' '))
            .unwrap_or_else(|| panic!("no {series} in\n{text}"))
            .parse()
            .unwrap()
    }

    #[test]
    fn a_request_moves_through_the_gauges_and_counters() {
        let metrics = Arc::new(Metrics::new("m"));
        let t0 = Instant::now();
        let mut tracker = metrics.track(100);
        let text = metrics.render();
        assert_eq!(
            value(&text, "vllm:num_requests_waiting{model_name=\"m\"}"),
            1.0
        );
        assert_eq!(
            value(&text, "vllm:num_requests_running{model_name=\"m\"}"),
            0.0
        );

        tracker.started_at(t0 + Duration::from_millis(5));
        tracker.cached(64);
        tracker.tokens_at(1, t0 + Duration::from_millis(50));
        // Four tokens 8 ms later: four gaps of 2 ms.
        tracker.tokens_at(4, t0 + Duration::from_millis(58));
        let text = metrics.render();
        assert_eq!(
            value(&text, "vllm:num_requests_waiting{model_name=\"m\"}"),
            0.0
        );
        assert_eq!(
            value(&text, "vllm:num_requests_running{model_name=\"m\"}"),
            1.0
        );
        assert_eq!(
            value(&text, "vllm:prompt_tokens_total{model_name=\"m\"}"),
            100.0
        );
        assert_eq!(
            value(&text, "vllm:generation_tokens_total{model_name=\"m\"}"),
            5.0
        );
        assert_eq!(
            value(&text, "vllm:prefix_cache_queries_total{model_name=\"m\"}"),
            100.0
        );
        assert_eq!(
            value(&text, "vllm:prefix_cache_hits_total{model_name=\"m\"}"),
            64.0
        );
        let itl = "vllm:inter_token_latency_seconds";
        assert_eq!(
            value(&text, &format!("{itl}_count{{model_name=\"m\"}}")),
            4.0
        );
        assert_eq!(
            value(
                &text,
                &format!("{itl}_bucket{{model_name=\"m\",le=\"0.0015\"}}")
            ),
            0.0
        );
        assert_eq!(
            value(
                &text,
                &format!("{itl}_bucket{{model_name=\"m\",le=\"0.002\"}}")
            ),
            4.0
        );
        assert_eq!(
            value(
                &text,
                &format!("{itl}_bucket{{model_name=\"m\",le=\"+Inf\"}}")
            ),
            4.0
        );
        let ttft = "vllm:time_to_first_token_seconds";
        assert_eq!(
            value(
                &text,
                &format!("{ttft}_bucket{{model_name=\"m\",le=\"0.04\"}}")
            ),
            0.0
        );
        assert_eq!(
            value(
                &text,
                &format!("{ttft}_bucket{{model_name=\"m\",le=\"0.06\"}}")
            ),
            1.0
        );

        tracker.finish(Reason::Length);
        drop(tracker);
        let text = metrics.render();
        assert_eq!(
            value(&text, "vllm:num_requests_running{model_name=\"m\"}"),
            0.0
        );
        let success = "vllm:request_success_total{model_name=\"m\",finished_reason=";
        assert_eq!(value(&text, &format!("{success}\"length\"}}")), 1.0);
        assert_eq!(value(&text, &format!("{success}\"abort\"}}")), 0.0);
    }

    #[test]
    fn dropping_a_tracker_counts_an_abort() {
        let metrics = Arc::new(Metrics::new("a \"b\""));
        drop(metrics.track(10));
        let text = metrics.render();
        let labels = r#"model_name="a \"b\"""#;
        assert_eq!(
            value(&text, &format!("vllm:num_requests_waiting{{{labels}}}")),
            0.0
        );
        assert_eq!(
            value(
                &text,
                &format!("vllm:request_success_total{{{labels},finished_reason=\"abort\"}}")
            ),
            1.0
        );
        // Never reached a first token: no prompt tokens counted.
        assert_eq!(
            value(&text, &format!("vllm:prompt_tokens_total{{{labels}}}")),
            0.0
        );
    }
}
