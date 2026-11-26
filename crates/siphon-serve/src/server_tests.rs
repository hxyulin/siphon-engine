//! The router end to end, with an engine whose jobs a test thread answers.

use std::{
    sync::{Arc, Mutex},
    thread,
};

use axum::{
    Router,
    body::{self, Body},
    http::{Request, Response},
};
use futures_util::StreamExt as _;
use serde_json::{Value, json};
use tower::ServiceExt as _;

use crate::{
    chat::{Vocab, tests::vocab},
    engine::{Engine, Event, Finish, Job},
    metrics::Metrics,
    openai::{self, App},
    request::SamplingDefaults,
};

/// A server whose engine answers every job with `script`'s events, and the labels of the
/// jobs it took.
pub(crate) struct Server {
    pub(crate) router: Router,
    pub(crate) labels: Arc<Mutex<Vec<String>>>,
}

impl Server {
    pub(crate) fn new(
        max_context: usize,
        script: impl Fn(&Vocab, &Job) -> Vec<Event> + Send + 'static,
    ) -> Self {
        Self::with_engine(max_context, move |vocab, job| {
            for event in script(vocab, job) {
                let _ = job.events.send(event);
            }
        })
    }

    /// A server whose engine runs `engine` on each job, on one thread.
    pub(crate) fn with_engine(
        max_context: usize,
        engine: impl Fn(&Vocab, &Job) + Send + 'static,
    ) -> Self {
        let (fake, jobs) = Engine::fake(max_context);
        let labels = Arc::new(Mutex::new(Vec::new()));
        let taken = Arc::clone(&labels);
        let engine_vocab = vocab();
        thread::spawn(move || {
            for submission in jobs {
                for job in submission {
                    taken.lock().unwrap().push(job.label.to_string());
                    engine(&engine_vocab, &job);
                }
            }
        });
        let engine = fake;
        let app = Arc::new(App {
            vocab: vocab(),
            engine,
            metrics: Arc::new(Metrics::new("m")),
            name: "m".to_owned(),
            api_key: None,
            started: 0,
            sampling: SamplingDefaults::default(),
            greedy: false,
        });
        Self {
            router: openai::router(app),
            labels,
        }
    }

    pub(crate) async fn call(&self, request: Request<Body>) -> Response<Body> {
        self.router.clone().oneshot(request).await.unwrap()
    }
}

/// A reply of `text` that ends the turn, after a prefill that read `cached` tokens from the
/// prefix cache.
pub(crate) fn text_reply(vocab: &Vocab, text: &str, cached: usize) -> Vec<Event> {
    let mut ids = vec![vocab.special.content_text];
    ids.extend(vocab.encode(text));
    ids.push(vocab.special.end_message);
    vec![
        Event::Started,
        Event::Cached(cached),
        Event::Tokens(ids),
        Event::Done(Finish::Stop),
    ]
}

/// [`text_reply`] in two speculation rounds, the first of three tokens.
pub(crate) fn speculated_reply(vocab: &Vocab, text: &str, cached: usize) -> Vec<Event> {
    let mut events = text_reply(vocab, text, cached);
    let Event::Tokens(ids) = events.remove(2) else {
        unreachable!("text_reply's third event holds the tokens")
    };
    let (a, b) = ids.split_at(3);
    let rounds = [
        Event::Round,
        Event::Tokens(a.to_vec()),
        Event::Round,
        Event::Tokens(b.to_vec()),
    ];
    events.splice(2..2, rounds);
    events
}

/// The data of each `data:` line of a server-sent stream.
pub(crate) async fn sse_data(response: Response<Body>) -> Vec<Value> {
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

/// A POST of `body` as JSON to `path`, with `headers`.
pub(crate) fn post(path: &str, body: &Value, headers: &[(&str, &str)]) -> Request<Body> {
    let mut request = Request::post(path).header("content-type", "application/json");
    for (name, value) in headers {
        request = request.header(*name, *value);
    }
    request.body(Body::from(body.to_string())).unwrap()
}

pub(crate) async fn json_body(response: Response<Body>) -> Value {
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

pub(crate) fn chat_body(stream: bool) -> Value {
    json!({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16, "stream": stream})
}

pub(crate) fn messages_body(stream: bool) -> Value {
    json!({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16, "stream": stream})
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn request_ids_reach_the_response_and_the_job() {
    let server = Server::new(4096, |vocab, _| text_reply(vocab, "Hello.", 0));
    let header = |response: &Response<Body>| {
        response.headers()["x-request-id"]
            .to_str()
            .unwrap()
            .to_owned()
    };

    // The client's id comes back and labels the job, next to the response's own id.
    let response = server
        .call(post(
            "/v1/chat/completions",
            &chat_body(false),
            &[("x-request-id", "trace-7f3a")],
        ))
        .await;
    assert_eq!(header(&response), "trace-7f3a");
    let id = json_body(response).await["id"].as_str().unwrap().to_owned();
    assert_eq!(server.labels.lock().unwrap()[0], format!("trace-7f3a {id}"));

    // A stream, and x-client-request-id when x-request-id is missing.
    let response = server
        .call(post(
            "/v1/messages",
            &messages_body(true),
            &[("x-client-request-id", "cc-42")],
        ))
        .await;
    assert_eq!(header(&response), "cc-42");
    let _ = body::to_bytes(response.into_body(), usize::MAX).await;
    let label = server.labels.lock().unwrap()[1].clone();
    assert!(label.starts_with("cc-42 msg_"), "{label}");

    // An invalid id is replaced by one the server makes.
    let response = server
        .call(post(
            "/v1/completions",
            &json!({"prompt": "Hi", "max_tokens": 4}),
            &[("x-request-id", "has a space")],
        ))
        .await;
    let made = header(&response);
    assert!(made.starts_with("req-"), "{made}");
    let label = server.labels.lock().unwrap()[2].clone();
    assert!(label.starts_with(&format!("{made} cmpl-")), "{label}");

    // Errors carry the id too.
    let response = server
        .call(post("/v1/chat/completions", &json!({"messages": 3}), &[]))
        .await;
    assert_eq!(response.status(), 400);
    assert!(header(&response).starts_with("req-"));
}

/// Checks `timings` against a reply of `predicted_n` tokens in two rounds, after a prompt
/// that read 2 tokens from the cache.
fn check_timings(timings: &Value, prompt_tokens: u64, predicted_n: u64) {
    assert_eq!(timings["cache_n"], 2, "{timings}");
    assert_eq!(timings["prompt_n"], prompt_tokens - 2, "{timings}");
    assert_eq!(timings["predicted_n"], predicted_n, "{timings}");
    assert_eq!(timings["draft_rounds"], 2, "{timings}");
    let per_round = timings["tokens_per_round"].as_f64().unwrap();
    assert!(
        (per_round - predicted_n as f64 / 2.0).abs() < 1e-3,
        "{timings}"
    );
    for field in [
        "prompt_ms",
        "prompt_per_token_ms",
        "prompt_per_second",
        "predicted_ms",
        "predicted_per_token_ms",
        "predicted_per_second",
        "queue_ms",
        "ttft_ms",
    ] {
        assert!(
            timings[field].as_f64().unwrap() >= 0.0,
            "{field} in {timings}"
        );
    }
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn timings_reach_every_route() {
    let server = Server::new(4096, |vocab, _| speculated_reply(vocab, "Hello there.", 2));

    // Chat, whole: a top-level field and a Server-Timing header.
    let response = server
        .call(post("/v1/chat/completions", &chat_body(false), &[]))
        .await;
    let header = response.headers()["server-timing"]
        .to_str()
        .unwrap()
        .to_owned();
    assert!(
        header.starts_with("queue;dur=")
            && header.contains(", prefill;dur=")
            && header.contains(", decode;dur="),
        "{header}"
    );
    let body = json_body(response).await;
    let usage = &body["usage"];
    check_timings(
        &body["timings"],
        usage["prompt_tokens"].as_u64().unwrap(),
        usage["completion_tokens"].as_u64().unwrap(),
    );

    // Chat, streamed: the finish chunk holds them, or the usage chunk when asked for.
    let response = server
        .call(post("/v1/chat/completions", &chat_body(true), &[]))
        .await;
    assert!(response.headers().get("server-timing").is_none());
    let chunks = sse_data(response).await;
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        chunks.iter().filter(|c| c.get("timings").is_some()).count(),
        1
    );
    let predicted = body["usage"]["completion_tokens"].as_u64().unwrap();
    let prompt = body["usage"]["prompt_tokens"].as_u64().unwrap();
    check_timings(&last["timings"], prompt, predicted);

    let mut with_usage = chat_body(true);
    with_usage["stream_options"] = json!({"include_usage": true});
    let response = server
        .call(post("/v1/chat/completions", &with_usage, &[]))
        .await;
    let chunks = sse_data(response).await;
    let last = chunks.last().unwrap();
    assert!(last.get("usage").is_some());
    assert_eq!(
        chunks.iter().filter(|c| c.get("timings").is_some()).count(),
        1
    );
    check_timings(&last["timings"], prompt, predicted);

    // Completions, whole.
    let response = server
        .call(post(
            "/v1/completions",
            &json!({"prompt": "Hello there, how are you?", "max_tokens": 16}),
            &[],
        ))
        .await;
    assert!(response.headers().contains_key("server-timing"));
    let body = json_body(response).await;
    check_timings(
        &body["timings"],
        body["usage"]["prompt_tokens"].as_u64().unwrap(),
        body["usage"]["completion_tokens"].as_u64().unwrap(),
    );

    // Messages, whole and streamed: the body, and the message_delta event.
    let response = server
        .call(post("/v1/messages", &messages_body(false), &[]))
        .await;
    assert!(response.headers().contains_key("server-timing"));
    let body = json_body(response).await;
    let usage = &body["usage"];
    let prompt = usage["input_tokens"].as_u64().unwrap() + 2;
    let predicted = usage["output_tokens"].as_u64().unwrap();
    check_timings(&body["timings"], prompt, predicted);
    let response = server
        .call(post("/v1/messages", &messages_body(true), &[]))
        .await;
    let events = sse_data(response).await;
    let delta = events
        .iter()
        .find(|e| e["type"] == "message_delta")
        .unwrap();
    check_timings(&delta["timings"], prompt, predicted);
    assert_eq!(
        events.iter().filter(|e| e.get("timings").is_some()).count(),
        1
    );
}

fn gzip(bytes: &[u8]) -> Vec<u8> {
    use std::io::Write as _;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(bytes).unwrap();
    encoder.finish().unwrap()
}

/// A POST of `bytes` to `path` with `Content-Encoding: encoding`.
fn encoded_post(path: &str, bytes: Vec<u8>, encoding: &str) -> Request<Body> {
    Request::post(path)
        .header("content-type", "application/json")
        .header("content-encoding", encoding)
        .body(Body::from(bytes))
        .unwrap()
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn compressed_bodies_decompress_within_the_limit() {
    let server = Server::new(4096, |vocab, _| text_reply(vocab, "Hello.", 0));
    let request = json!({"messages": [{"role": "user", "content": "Count to three."}],
                         "max_tokens": 16, "seed": 1});
    let plain = request.to_string().into_bytes();
    let reply = |response: Response<Body>| async move {
        assert_eq!(response.status(), 200);
        let mut body = json_body(response).await;
        // Fields that differ between any two requests.
        for field in ["id", "created", "timings"] {
            body.as_object_mut().unwrap().remove(field);
        }
        body
    };
    let expected = reply(
        server
            .call(encoded_post(
                "/v1/chat/completions",
                plain.clone(),
                "identity",
            ))
            .await,
    )
    .await;
    for (encoding, bytes) in [
        ("gzip", gzip(&plain)),
        ("zstd", zstd::encode_all(&plain[..], 3).unwrap()),
    ] {
        let response = server
            .call(encoded_post("/v1/chat/completions", bytes, encoding))
            .await;
        assert_eq!(reply(response).await, expected, "{encoding}");
    }

    // An encoding the server does not decode.
    let response = server
        .call(encoded_post("/v1/chat/completions", plain.clone(), "br"))
        .await;
    assert_eq!(response.status(), 415);

    // A body that inflates past the 2 MiB limit gets a 413, Anthropic's too.
    let mut big = request.clone();
    big["messages"][0]["content"] = json!(" ".repeat(3 << 20));
    let bomb = gzip(big.to_string().as_bytes());
    assert!(bomb.len() < 64 << 10, "{} compressed bytes", bomb.len());
    let response = server
        .call(encoded_post("/v1/chat/completions", bomb.clone(), "gzip"))
        .await;
    assert_eq!(response.status(), 413);
    let error = json_body(response).await;
    assert_eq!(error["error"]["type"], "invalid_request_error", "{error}");
    let response = server
        .call(encoded_post("/v1/messages", bomb, "gzip"))
        .await;
    assert_eq!(response.status(), 413);
    assert_eq!(
        json_body(response).await["error"]["type"],
        "request_too_large"
    );
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn responses_compress_and_streams_do_not() {
    // The engine sends the first token, then waits until the test has read it.
    let (go, wait) = std::sync::mpsc::channel::<()>();
    let wait = Mutex::new(wait);
    let server = Server::with_engine(4096, move |vocab, job| {
        let mut events = text_reply(vocab, "Hello there.", 0).into_iter();
        for event in events.by_ref().take(2) {
            let _ = job.events.send(event);
        }
        let Some(Event::Tokens(ids)) = events.next() else {
            unreachable!("text_reply's third event holds the tokens")
        };
        let _ = job.events.send(Event::Tokens(ids[..2].to_vec()));
        if job.label.contains("chatcmpl") {
            let _ = wait.lock().unwrap().recv();
        }
        let _ = job.events.send(Event::Tokens(ids[2..].to_vec()));
        for event in events {
            let _ = job.events.send(event);
        }
    });
    let gzip_accepted = [("accept-encoding", "gzip")];

    // A whole reply comes back compressed, as asked.
    let response = server
        .call(post("/v1/messages", &messages_body(false), &gzip_accepted))
        .await;
    assert_eq!(response.headers()["content-encoding"], "gzip");
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let mut text = String::new();
    std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&bytes[..]), &mut text)
        .unwrap();
    let body: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(body["content"][0]["text"], "Hello there.");

    // A stream is not compressed, and its first chunk arrives while the job still runs.
    let response = server
        .call(post(
            "/v1/chat/completions",
            &chat_body(true),
            &gzip_accepted,
        ))
        .await;
    assert!(response.headers().get("content-encoding").is_none());
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let mut stream = response.into_body().into_data_stream();
    let first = tokio::time::timeout(std::time::Duration::from_secs(10), stream.next())
        .await
        .expect("the first chunk came before the job ended")
        .unwrap()
        .unwrap();
    assert!(
        std::str::from_utf8(&first)
            .unwrap()
            .contains("\"role\":\"assistant\""),
        "{first:?}"
    );
    go.send(()).unwrap();
    let mut rest = Vec::new();
    while let Some(chunk) = stream.next().await {
        rest.extend_from_slice(&chunk.unwrap());
    }
    assert!(
        std::str::from_utf8(&rest)
            .unwrap()
            .ends_with("data: [DONE]\n\n")
    );
}
