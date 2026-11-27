//! A request's `timings`, under the names of llama.cpp's server so that tools written for
//! it read them, with Siphon's queue time, time to first token and speculation added.
//!
//! The times come from when the job's events reached the HTTP layer ([`Times`]): the
//! queue runs from the job's submission, after the request was parsed and tokenized, to
//! the engine's `Started`; the prompt from `Started` to the first tokens, less the vision
//! tower's encodes, which the engine times on the GPU; generation from the first tokens to
//! the end. A request with several choices counts from its earliest start and first tokens
//! to its last end, and sums the choices' tokens, rounds and encodes.

use std::time::Instant;

use axum::http::HeaderValue;
use serde_json::{Value, json};

use crate::{metrics::Times, vision::ImageStats};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timings {
    /// Prompt tokens read from the prefix cache, and the prompt tokens prefilled.
    pub cache_n: usize,
    pub prompt_n: usize,
    /// Generated tokens the reply holds.
    pub predicted_n: usize,
    pub queue_ms: f64,
    pub prompt_ms: f64,
    pub predicted_ms: f64,
    /// Speculation rounds, and the tokens they delivered.
    pub rounds: u64,
    pub round_tokens: u64,
    /// What the vision tower did, for a prompt with images. `prompt_ms` leaves out its
    /// encodes.
    pub images: Option<ImageStats>,
}

impl Timings {
    /// The timings of a request whose jobs' events arrived at `times`, of a prompt of
    /// `prompt_tokens` with `cached` read from the prefix cache, that generated
    /// `predicted_n` tokens.
    pub fn new(times: &[Times], prompt_tokens: usize, cached: usize, predicted_n: usize) -> Self {
        let now = Instant::now();
        let submitted = times.iter().map(|t| t.submitted).min().unwrap_or(now);
        let ended = times
            .iter()
            .map(|t| t.ended.unwrap_or(now))
            .max()
            .unwrap_or(now);
        let first = times.iter().filter_map(|t| t.first).min().unwrap_or(ended);
        let started = times
            .iter()
            .filter_map(|t| t.started)
            .min()
            .unwrap_or(first)
            .min(first);
        let ms =
            |from: Instant, to: Instant| to.saturating_duration_since(from).as_secs_f64() * 1e3;
        // Choices share their prompt: each reports the same images from the cache.
        let images = times
            .iter()
            .filter_map(|t| t.images)
            .reduce(|a, b| ImageStats {
                encoded: a.encoded + b.encoded,
                ms: a.ms + b.ms,
                cached: a.cached.max(b.cached),
            });
        let image_ms = images.map_or(0.0, |i| i.ms);
        Self {
            cache_n: cached,
            prompt_n: prompt_tokens.saturating_sub(cached),
            predicted_n,
            queue_ms: ms(submitted, started),
            prompt_ms: (ms(started, first) - image_ms).max(0.0),
            predicted_ms: ms(first, ended),
            rounds: times.iter().map(|t| t.rounds).sum(),
            round_tokens: times.iter().map(|t| t.round_tokens).sum(),
            images,
        }
    }

    /// Milliseconds the vision tower encoded for.
    fn image_ms(&self) -> f64 {
        self.images.map_or(0.0, |i| i.ms)
    }

    /// The `timings` object.
    pub fn json(&self) -> Value {
        let mut timings = json!({
            "cache_n": self.cache_n,
            "prompt_n": self.prompt_n,
            "prompt_ms": round(self.prompt_ms),
            "prompt_per_token_ms": round(per(self.prompt_ms, self.prompt_n as f64)),
            "prompt_per_second": round(per(self.prompt_n as f64 * 1e3, self.prompt_ms)),
            "predicted_n": self.predicted_n,
            "predicted_ms": round(self.predicted_ms),
            "predicted_per_token_ms": round(per(self.predicted_ms, self.predicted_n as f64)),
            "predicted_per_second": round(per(self.predicted_n as f64 * 1e3, self.predicted_ms)),
            "queue_ms": round(self.queue_ms),
            "ttft_ms": round(self.queue_ms + self.image_ms() + self.prompt_ms),
        });
        if let Some(images) = self.images {
            timings["image_n"] = json!(images.encoded);
            timings["image_cache_n"] = json!(images.cached);
            timings["image_ms"] = json!(round(images.ms));
        }
        if self.rounds > 0 {
            timings["draft_rounds"] = json!(self.rounds);
            timings["tokens_per_round"] =
                json!(round(self.round_tokens as f64 / self.rounds as f64));
        }
        timings
    }

    /// A `Server-Timing` header with the queue, image encode, prefill and decode times in
    /// milliseconds; the image encode only for a prompt with images.
    pub fn server_timing(&self) -> HeaderValue {
        let image = match self.images {
            Some(images) => format!(", image;dur={:.3}", images.ms),
            None => String::new(),
        };
        let value = format!(
            "queue;dur={:.3}{image}, prefill;dur={:.3}, decode;dur={:.3}",
            self.queue_ms, self.prompt_ms, self.predicted_ms
        );
        HeaderValue::from_str(&value).expect("the header is ASCII")
    }
}

/// `a / b`, or 0 when `b` is 0.
fn per(a: f64, b: f64) -> f64 {
    if b > 0.0 { a / b } else { 0.0 }
}

/// `x` to three decimals: microseconds for times in milliseconds.
fn round(x: f64) -> f64 {
    (x * 1e3).round() / 1e3
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn at(start: Instant, ms: u64) -> Instant {
        start + Duration::from_millis(ms)
    }

    #[test]
    fn fields_follow_llama_cpp() {
        let t0 = Instant::now();
        let job = Times {
            started: Some(at(t0, 20)),
            first: Some(at(t0, 520)),
            ended: Some(at(t0, 1520)),
            rounds: 40,
            round_tokens: 150,
            ..Times::new(t0)
        };
        let timings = Timings::new(&[job], 30_000, 20_000, 200);
        assert_eq!(
            timings.json(),
            json!({
                "cache_n": 20_000, "prompt_n": 10_000, "prompt_ms": 500.0,
                "prompt_per_token_ms": 0.05, "prompt_per_second": 20_000.0,
                "predicted_n": 200, "predicted_ms": 1000.0, "predicted_per_token_ms": 5.0,
                "predicted_per_second": 200.0, "queue_ms": 20.0, "ttft_ms": 520.0,
                "draft_rounds": 40, "tokens_per_round": 3.75,
            })
        );
        assert_eq!(
            timings.server_timing(),
            "queue;dur=20.000, prefill;dur=500.000, decode;dur=1000.000"
        );

        // Choices: the earliest start and first token, the last end, the sums.
        let other = Times {
            started: Some(at(t0, 30)),
            first: Some(at(t0, 510)),
            ended: Some(at(t0, 2000)),
            rounds: 0,
            round_tokens: 0,
            ..Times::new(t0)
        };
        let both = Timings::new(&[job, other], 30_000, 20_000, 400);
        assert_eq!(
            (both.queue_ms, both.prompt_ms, both.predicted_ms),
            (20.0, 490.0, 1490.0)
        );
        assert_eq!(both.json()["draft_rounds"], 40);

        // A job without tokens or rounds: no division by zero, no speculation fields.
        let empty = Times {
            started: Some(at(t0, 5)),
            ended: Some(at(t0, 9)),
            ..Times::new(t0)
        };
        let json = Timings::new(&[empty], 10, 10, 0).json();
        assert_eq!(json["prompt_per_second"], 0.0);
        assert_eq!(json["predicted_per_token_ms"], 0.0);
        assert_eq!(json["prompt_ms"], 4.0);
        assert!(json.get("draft_rounds").is_none());
        assert!(json.get("image_ms").is_none());

        // Images: the encodes leave the prompt time, and their own fields report them.
        let encoded = |encoded: usize, ms: f64| Times {
            images: Some(ImageStats {
                encoded,
                ms,
                cached: 1,
            }),
            ..job
        };
        let timings = Timings::new(&[encoded(2, 120.0), encoded(0, 0.0)], 30_000, 20_000, 200);
        let json = timings.json();
        assert_eq!(
            (&json["image_n"], &json["image_cache_n"], &json["image_ms"]),
            (&json!(2), &json!(1), &json!(120.0))
        );
        assert_eq!(json["prompt_ms"], 380.0);
        assert_eq!(json["ttft_ms"], 520.0);
        assert_eq!(
            timings.server_timing(),
            "queue;dur=20.000, image;dur=120.000, prefill;dur=380.000, decode;dur=1000.000"
        );
    }
}
