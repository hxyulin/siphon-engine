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
    keepalive::KeepAlive,
    metrics::Metrics,
    openai::{self, App},
    request::SamplingDefaults,
    vision::ImageStats,
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
        Self::with_keepalive(max_context, None, engine)
    }

    /// [`Server::with_engine`] with `keepalive` for replies that are not streamed.
    pub(crate) fn with_keepalive(
        max_context: usize,
        keepalive: Option<KeepAlive>,
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
            keepalive,
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
    let response = server
        .call(encoded_post("/v1/chat/completions", gzip(&plain), "gzip"))
        .await;
    assert_eq!(reply(response).await, expected);
    #[cfg(feature = "http-zstd")]
    let zstd_body = zstd::encode_all(&plain[..], 3).unwrap();
    #[cfg(not(feature = "http-zstd"))]
    let zstd_body = plain.clone();
    let response = server
        .call(encoded_post("/v1/chat/completions", zstd_body, "zstd"))
        .await;
    #[cfg(feature = "http-zstd")]
    assert_eq!(reply(response).await, expected);
    #[cfg(not(feature = "http-zstd"))]
    assert_eq!(response.status(), 415);

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

    let response = server
        .call(post(
            "/v1/messages",
            &messages_body(false),
            &[("accept-encoding", "zstd")],
        ))
        .await;
    #[cfg(feature = "http-zstd")]
    assert_eq!(response.headers()["content-encoding"], "zstd");
    #[cfg(not(feature = "http-zstd"))]
    assert!(response.headers().get("content-encoding").is_none());
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    #[cfg(feature = "http-zstd")]
    let bytes = zstd::decode_all(&bytes[..]).unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
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

/// The calls of [`tool_reply`]: names and argument objects as the model writes them.
const CALLS: [(&str, &str); 2] = [
    (
        "get_weather",
        r#"{"city":"Paris \"Rive {Gauche}\"","units":["C"]}"#,
    ),
    ("get_time", r#"{"zone":"Europe/Paris"}"#),
];

/// A reply that thinks and makes both [`CALLS`], one token per event. After a prompt that
/// opens a call (a forced `tool_choice`), the first call's block is already open.
fn tool_reply(vocab: &Vocab, job: &Job) -> Vec<Event> {
    let s = vocab.special;
    let forced = job.prompt.last() == Some(&s.content_invoke_tool_json);
    let mut ids = Vec::new();
    if !forced {
        ids.push(s.content_thinking);
        ids.extend(vocab.encode("Two calls."));
        ids.push(s.end_message);
    }
    for (i, (name, args)) in CALLS.iter().enumerate() {
        if !(forced && i == 0) {
            ids.push(s.message_model);
            ids.extend(vocab.encode(name));
            ids.push(s.content_invoke_tool_json);
        }
        ids.extend(vocab.encode(&format!(r#"{{"name":"{name}","args":{args}}}"#)));
        ids.push(s.end_message);
    }
    let mut events = vec![Event::Started, Event::Cached(0)];
    events.extend(ids.into_iter().map(|id| Event::Tokens(vec![id])));
    events.push(Event::Done(Finish::Stop));
    events
}

fn weather_tools() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "parameters": {}}},
           {"type": "function", "function": {"name": "get_time", "parameters": {}}}])
}

/// Each streamed call of `chunks`, a chat stream: its name, its id, and its argument
/// pieces. Checks that a call's first chunk names it and the later ones only add
/// arguments.
fn chat_calls(chunks: &[Value]) -> Vec<(String, String, Vec<String>)> {
    let mut calls: Vec<(String, String, Vec<String>)> = Vec::new();
    for call in chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["tool_calls"].as_array())
        .flatten()
    {
        let index = call["index"].as_u64().unwrap() as usize;
        let piece = call["function"]["arguments"].as_str().unwrap().to_owned();
        if index == calls.len() {
            assert_eq!(call["type"], "function", "{call}");
            assert_eq!(piece, "", "{call}");
            let name = call["function"]["name"].as_str().unwrap().to_owned();
            calls.push((name, call["id"].as_str().unwrap().to_owned(), Vec::new()));
        } else {
            assert_eq!(index + 1, calls.len(), "{call}");
            assert!(call.get("id").is_none() && call["function"].get("name").is_none());
            calls[index].2.push(piece);
        }
    }
    calls
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn chat_streams_tool_calls_piece_by_piece() {
    let server = Server::new(4096, tool_reply);
    let request = |stream: bool, choice: Value| {
        let body = json!({"messages": [{"role": "user", "content": "Weather and time?"}],
                          "tools": weather_tools(), "tool_choice": choice,
                          "max_tokens": 200, "stream": stream});
        post("/v1/chat/completions", &body, &[])
    };
    let named = json!({"type": "function", "function": {"name": "get_weather"}});
    // Parallel calls, a call forced by name, and calls required.
    for (choice, n) in [(json!("auto"), 2), (named, 1), (json!("required"), 2)] {
        let whole = json_body(server.call(request(false, choice.clone())).await).await;
        let message = &whole["choices"][0]["message"];
        let want = message["tool_calls"].as_array().unwrap();
        assert_eq!(want.len(), n, "{choice}: {message}");
        let chunks = sse_data(server.call(request(true, choice.clone())).await).await;
        let calls = chat_calls(&chunks);
        assert_eq!(calls.len(), n, "{choice}");
        for (i, (name, id, pieces)) in calls.iter().enumerate() {
            assert_eq!(name, CALLS[i].0);
            assert!(id.starts_with("call_"));
            assert!(
                pieces.len() > 1,
                "{choice}: arguments in one piece: {pieces:?}"
            );
            assert_eq!(pieces.concat(), CALLS[i].1);
            assert_eq!(want[i]["function"]["arguments"], CALLS[i].1);
            assert_eq!(want[i]["function"]["name"], CALLS[i].0);
        }
        let last = &chunks.last().unwrap()["choices"][0];
        assert_eq!(last["finish_reason"], "tool_calls", "{choice}");
    }
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn messages_stream_tool_input_piece_by_piece() {
    let server = Server::new(4096, tool_reply);
    let request = |stream: bool, choice: Value| {
        let body = json!({"messages": [{"role": "user", "content": "Weather and time?"}],
                          "tools": [{"name": "get_weather", "input_schema": {}},
                                    {"name": "get_time", "input_schema": {}}],
                          "tool_choice": choice, "max_tokens": 200, "stream": stream});
        post("/v1/messages", &body, &[])
    };
    let named = json!({"type": "tool", "name": "get_weather"});
    for (choice, n) in [(json!({"type": "auto"}), 2), (named, 1)] {
        let whole = json_body(server.call(request(false, choice.clone())).await).await;
        let uses: Vec<&Value> = whole["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|b| b["type"] == "tool_use")
            .collect();
        assert_eq!(uses.len(), n, "{choice}: {whole}");
        let events = sse_data(server.call(request(true, choice.clone())).await).await;
        let starts: Vec<&Value> = events
            .iter()
            .filter(|e| {
                e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use"
            })
            .collect();
        assert_eq!(starts.len(), n, "{choice}");
        for (i, start) in starts.iter().enumerate() {
            let index = &start["index"];
            assert_eq!(start["content_block"]["name"], CALLS[i].0);
            assert_eq!(start["content_block"]["input"], json!({}));
            let at = events.iter().position(|e| e == *start).unwrap();
            let pieces: Vec<&str> = events[at + 1..]
                .iter()
                .take_while(|e| e["type"] != "content_block_stop")
                .map(|e| {
                    assert_eq!(e["index"], *index);
                    assert_eq!(e["delta"]["type"], "input_json_delta");
                    e["delta"]["partial_json"].as_str().unwrap()
                })
                .collect();
            assert!(pieces.len() > 1, "{choice}: input in one piece: {pieces:?}");
            assert_eq!(pieces.concat(), CALLS[i].1);
            let input: Value = serde_json::from_str(CALLS[i].1).unwrap();
            assert_eq!(uses[i]["input"], input);
            assert_eq!(
                uses[i]["id"].as_str().unwrap().split('_').next(),
                Some("toolu")
            );
        }
        let delta = events
            .iter()
            .find(|e| e["type"] == "message_delta")
            .unwrap();
        assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    }
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn responses_stream_function_arguments_piece_by_piece() {
    let server = Server::new(4096, tool_reply);
    let request = |stream: bool, choice: Value| {
        let body = json!({"input": "Weather and time?",
                          "tools": [{"type": "function", "name": "get_weather", "parameters": {}},
                                    {"type": "function", "name": "get_time", "parameters": {}}],
                          "tool_choice": choice, "max_output_tokens": 200, "stream": stream});
        post("/v1/responses", &body, &[])
    };
    let named = json!({"type": "function", "name": "get_weather"});
    for (choice, n) in [(json!("auto"), 2), (named, 1)] {
        let whole = json_body(server.call(request(false, choice.clone())).await).await;
        let items: Vec<&Value> = whole["output"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|i| i["type"] == "function_call")
            .collect();
        assert_eq!(items.len(), n, "{choice}: {whole}");
        let events = sse_data(server.call(request(true, choice.clone())).await).await;
        let added: Vec<&Value> = events
            .iter()
            .filter(|e| {
                e["type"] == "response.output_item.added" && e["item"]["type"] == "function_call"
            })
            .collect();
        assert_eq!(added.len(), n, "{choice}");
        for (i, start) in added.iter().enumerate() {
            let item = &start["item"];
            assert_eq!(item["name"], CALLS[i].0);
            assert_eq!(item["arguments"], "");
            assert_eq!(item["status"], "in_progress");
            let at = events.iter().position(|e| e == *start).unwrap();
            let pieces: Vec<&str> = events[at + 1..]
                .iter()
                .take_while(|e| e["type"] == "response.function_call_arguments.delta")
                .map(|e| {
                    assert_eq!(e["item_id"], item["id"]);
                    e["delta"].as_str().unwrap()
                })
                .collect();
            assert!(
                pieces.len() > 1,
                "{choice}: arguments in one piece: {pieces:?}"
            );
            assert_eq!(pieces.concat(), CALLS[i].1);
            let done = &events[at + 1 + pieces.len()];
            assert_eq!(done["type"], "response.function_call_arguments.done");
            assert_eq!(done["arguments"], CALLS[i].1);
            assert_eq!(done["name"], CALLS[i].0);
            assert_eq!(items[i]["arguments"], CALLS[i].1);
            assert_eq!(items[i]["name"], CALLS[i].0);
        }
    }
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn image_encodes_leave_the_prompt_time() {
    let stats = ImageStats {
        encoded: 1,
        ms: 50.0,
        cached: 2,
    };
    let server = Server::with_engine(4096, move |vocab, job| {
        let _ = job.events.send(Event::Started);
        let _ = job.events.send(Event::Cached(2));
        thread::sleep(std::time::Duration::from_millis(80));
        let _ = job.events.send(Event::Images(stats));
        for event in text_reply(vocab, "A cat.", 2).into_iter().skip(2) {
            let _ = job.events.send(event);
        }
    });
    let response = server
        .call(post("/v1/chat/completions", &chat_body(false), &[]))
        .await;
    let header = response.headers()["server-timing"]
        .to_str()
        .unwrap()
        .to_owned();
    assert!(
        header.contains(", image;dur=50.000, prefill;dur="),
        "{header}"
    );
    let timings = json_body(response).await["timings"].clone();
    assert_eq!(timings["image_n"], 1, "{timings}");
    assert_eq!(timings["image_cache_n"], 2, "{timings}");
    assert_eq!(timings["image_ms"], 50.0, "{timings}");
    // The prompt took at least the 80 ms sleep, of which 50 ms were encodes.
    let prompt_ms = timings["prompt_ms"].as_f64().unwrap();
    assert!((30.0..80.0).contains(&prompt_ms), "{timings}");
    let ttft = timings["ttft_ms"].as_f64().unwrap();
    let queue = timings["queue_ms"].as_f64().unwrap();
    assert!((ttft - queue - prompt_ms - 50.0).abs() < 1e-2, "{timings}");

    // A prompt without images has none of the fields.
    let server = Server::new(4096, |vocab, _| text_reply(vocab, "Hi.", 0));
    let body = json_body(
        server
            .call(post("/v1/messages", &messages_body(false), &[]))
            .await,
    )
    .await;
    assert!(
        body["timings"].get("image_ms").is_none(),
        "{}",
        body["timings"]
    );
}

/// The `request` log line of the request `id`, once its last part has let go of it.
async fn request_line(id: &str) -> String {
    let needle = format!(" request id={id} ");
    for _ in 0..400 {
        let found = crate::log::CAPTURED
            .lock()
            .unwrap()
            .iter()
            .find(|line| line.contains(&needle))
            .cloned();
        if let Some(line) = found {
            return line;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    panic!("no request line for {id}");
}

/// A request's error body, after checking its status.
async fn error_of(server: &Server, request: Request<Body>, status: u16) -> Value {
    let response = server.call(request).await;
    assert_eq!(response.status(), status);
    json_body(response).await
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
#[expect(clippy::too_many_lines, reason = "one case per kind of error")]
async fn rejected_requests_name_the_field_and_reach_the_log() {
    let server = Server::new(4096, |vocab, _| text_reply(vocab, "Hello.", 0));
    let chat = |extra: Value, id: &str| {
        let mut body = chat_body(false);
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        post("/v1/chat/completions", &body, &[("x-request-id", id)])
    };

    // An effort sent as a string: OpenAI's param, and advice to send a number.
    let body = error_of(
        &server,
        chat(json!({"reasoning_effort": "0.9"}), "e-str"),
        400,
    )
    .await;
    let error = &body["error"];
    assert_eq!(error["param"], "reasoning_effort");
    assert_eq!(error["type"], "invalid_request_error");
    assert_eq!(
        error["message"],
        "reasoning_effort: got \"0.9\" (a string); send 0.9 as a JSON number, not a string. \
         Accepted: a number from 0 to 0.99, or one of \"none\", \"minimal\", \"low\", \
         \"medium\", \"high\", \"xhigh\" or \"max\""
    );
    let line = request_line("e-str").await;
    assert!(
        line.contains(" route=\"POST /v1/chat/completions\" status=400 ")
            && line.contains(" error=\"reasoning_effort: got \\\"0.9\\\" (a string); send 0.9"),
        "{line}"
    );

    // An unknown name, and a number out of range.
    let body = error_of(
        &server,
        chat(json!({"reasoning_effort": "huge"}), "e-name"),
        400,
    )
    .await;
    let message = body["error"]["message"].as_str().unwrap();
    assert!(
        message.starts_with("reasoning_effort: expected a number from 0 to 0.99, or one of")
            && message.ends_with("got \"huge\" (a string)"),
        "{message}"
    );
    let body = error_of(
        &server,
        chat(json!({"reasoning_effort": 1.5}), "e-range"),
        400,
    )
    .await;
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .ends_with("got 1.5 (a number)")
    );

    // A string where serde wants a number, in a flattened sampling field and a plain one.
    let body = error_of(&server, chat(json!({"temperature": "0.7"}), "t-str"), 400).await;
    assert_eq!(body["error"]["param"], "temperature");
    assert_eq!(
        body["error"]["message"],
        "temperature: expected a number, got \"0.7\" (a string); send 0.7 as a JSON number, \
         not a string"
    );
    let body = error_of(&server, chat(json!({"max_tokens": "5"}), "m-str"), 400).await;
    assert_eq!(body["error"]["param"], "max_tokens");

    // A message's part, by its path; an image names the flag that enables it.
    let image = json!({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]});
    let body = error_of(&server, chat(image, "img"), 400).await;
    assert_eq!(body["error"]["param"], "messages[0].content[1]");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("start it with --vision"),
        "{body}"
    );
    let role = json!({"messages": [{"role": "user", "content": "a"}, {"role": "robot"}]});
    let body = error_of(&server, chat(role, "role"), 400).await;
    assert_eq!(body["error"]["param"], "messages[1].role");

    // The context's numbers.
    let long = json!({"messages": [{"role": "user", "content": "word ".repeat(5000)}]});
    let body = error_of(&server, chat(long, "long"), 400).await;
    assert_eq!(body["error"]["code"], "context_length_exceeded");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("context holds 4096") && message.contains("at most 4095 tokens"),
        "{message}"
    );

    // Anthropic's shape, with the field's path in the message.
    let mut messages = messages_body(false);
    messages["tool_choice"] = json!({"type": "sometimes"});
    let body = error_of(&server, post("/v1/messages", &messages, &[]), 400).await;
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(
        body["error"]["message"],
        "tool_choice.type: expected \"auto\", \"any\", \"tool\" or \"none\", got \"sometimes\" \
         (a string)"
    );

    // A route that does not exist still gets its line.
    let response = server
        .call(
            Request::get("/v1/nothing")
                .header("x-request-id", "no-route")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
    assert_eq!(response.status(), 404);
    let _ = body::to_bytes(response.into_body(), usize::MAX).await;
    let line = request_line("no-route").await;
    assert!(
        line.contains(" status=404 ") && line.ends_with(" error=\"no such route\""),
        "{line}"
    );
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn finished_requests_log_one_line() {
    let server = Server::new(4096, |vocab, _| speculated_reply(vocab, "Hello there.", 2));
    let field = |line: &str, key: &str| {
        line.split(' ')
            .find_map(|f| f.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("no {key} in {line}"))
            .to_owned()
    };

    let response = server
        .call(post(
            "/v1/chat/completions",
            &chat_body(false),
            &[("x-request-id", "ok-whole")],
        ))
        .await;
    let body = json_body(response).await;
    let line = request_line("ok-whole").await;
    assert!(
        line.split(' ').next().unwrap().ends_with('Z'),
        "a timestamp starts the line: {line}"
    );
    assert_eq!(field(&line, "route"), "\"POST");
    assert_eq!(field(&line, "status"), "200");
    assert_eq!(field(&line, "reply"), body["id"].as_str().unwrap());
    assert_eq!(field(&line, "stream"), "false");
    assert_eq!(field(&line, "effort"), "0.9");
    assert_eq!(field(&line, "finish"), "stop");
    assert_eq!(
        field(&line, "prompt"),
        body["usage"]["prompt_tokens"].to_string()
    );
    assert_eq!(field(&line, "cached"), "2");
    assert_eq!(
        field(&line, "output"),
        body["usage"]["completion_tokens"].to_string()
    );
    for key in ["ms", "queue_ms", "prefill_ms", "per_round"] {
        field(&line, key);
    }
    assert!(!line.contains("error="), "{line}");

    // A stream, with thinking off.
    let mut body = messages_body(true);
    body["thinking"] = json!({"type": "disabled"});
    let response = server
        .call(post(
            "/v1/messages",
            &body,
            &[("x-request-id", "ok-stream")],
        ))
        .await;
    let _ = body::to_bytes(response.into_body(), usize::MAX).await;
    let line = request_line("ok-stream").await;
    assert_eq!(field(&line, "stream"), "true");
    assert_eq!(field(&line, "effort"), "off");
    assert_eq!(field(&line, "finish"), "stop");
    assert!(!line.contains("client="), "{line}");
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn a_client_that_leaves_mid_stream_is_logged() {
    // A reply that goes on, a token at a time, until its client leaves.
    let server = Server::with_engine(4096, |vocab, job| {
        let _ = job.events.send(Event::Started);
        let _ = job.events.send(Event::Cached(0));
        let word = vocab.encode("word");
        let _ = job
            .events
            .send(Event::Tokens(vec![vocab.special.content_text]));
        while job.events.send(Event::Tokens(word.clone())).is_ok() {
            thread::sleep(std::time::Duration::from_millis(5));
        }
    });
    let response = server
        .call(post(
            "/v1/chat/completions",
            &chat_body(true),
            &[("x-request-id", "gone")],
        ))
        .await;
    let mut data = response.into_body().into_data_stream();
    let _ = data.next().await;
    let _ = data.next().await;
    drop(data);
    let line = request_line("gone").await;
    assert!(
        line.contains(" status=200 ")
            && line.contains(" finish=abort ")
            && line.ends_with(" client=disconnected"),
        "{line}"
    );
}

/// Keep-alive after 100 ms of silence, then every 50 ms.
const KEEPALIVE: KeepAlive = KeepAlive {
    after: std::time::Duration::from_millis(100),
    every: std::time::Duration::from_millis(50),
};

/// A server whose engine waits `delay` before it answers each job with `script`'s events.
fn slow_server(
    delay: std::time::Duration,
    script: impl Fn(&Vocab) -> Vec<Event> + Send + 'static,
) -> Server {
    Server::with_keepalive(4096, Some(KEEPALIVE), move |vocab, job| {
        thread::sleep(delay);
        for event in script(vocab) {
            let _ = job.events.send(event);
        }
    })
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn a_fast_reply_keeps_its_status_and_body() {
    let server = slow_server(std::time::Duration::ZERO, |vocab| {
        text_reply(vocab, "Hello.", 0)
    });
    let response = server
        .call(post("/v1/chat/completions", &chat_body(false), &[]))
        .await;
    assert_eq!(response.status(), 200);
    assert!(response.headers().contains_key("server-timing"));
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(bytes[0], b'{', "no keep-alive before a fast reply");
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Hello.");

    // A rejected request keeps its 400.
    let body = json!({"messages": [{"role": "robot"}]});
    let response = server.call(post("/v1/messages", &body, &[])).await;
    assert_eq!(response.status(), 400);
}

/// The status line, headers and each body chunk of a response read off a socket, with the
/// time each chunk arrived after the request was sent.
async fn over_tcp(
    server: &Server,
    path: &str,
    body: &Value,
) -> (String, Vec<(std::time::Duration, Vec<u8>)>) {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(axum::serve(listener, server.router.clone()).into_future());
    let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
    let body = body.to_string();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n\
         Accept-Encoding: gzip, zstd\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let sent = tokio::time::Instant::now();
    socket.write_all(request.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    let mut reads = Vec::new();
    let mut buffer = vec![0u8; 8192];
    loop {
        let n = socket.read(&mut buffer).await.unwrap();
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buffer[..n]);
        reads.push((sent.elapsed(), raw.len()));
    }
    let end = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    let head = String::from_utf8(raw[..end].to_vec()).unwrap();
    // Each chunk of the chunked body, with the time the read that completed it arrived.
    let mut chunks = Vec::new();
    let mut at = end;
    loop {
        let line = raw[at..].windows(2).position(|w| w == b"\r\n").unwrap();
        let size =
            usize::from_str_radix(std::str::from_utf8(&raw[at..at + line]).unwrap(), 16).unwrap();
        at += line + 2;
        if size == 0 {
            break;
        }
        let arrived = reads.iter().find(|(_, len)| *len >= at + size).unwrap().0;
        chunks.push((arrived, raw[at..at + size].to_vec()));
        at += size + 2;
    }
    (head, chunks)
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn a_slow_reply_sends_spaces_then_its_json() {
    let delay = std::time::Duration::from_millis(600);
    let server = slow_server(delay, |vocab| text_reply(vocab, "Hello.", 0));
    let (head, chunks) = over_tcp(&server, "/v1/chat/completions", &chat_body(false)).await;
    let head = head.to_ascii_lowercase();
    assert!(head.starts_with("http/1.1 200"), "{head}");
    assert!(head.contains("content-type: application/json"), "{head}");
    assert!(head.contains("transfer-encoding: chunked"), "{head}");
    // The spaces are not compressed, which would hold them back.
    assert!(!head.contains("content-encoding"), "{head}");
    // The first space reaches the socket right after the grace period, long before the
    // reply, and spaces follow every interval.
    let (first, bytes) = &chunks[0];
    assert_eq!(bytes, b" ");
    assert!(*first < delay / 2, "first space after {first:?}");
    let spaces = chunks.iter().filter(|(_, b)| b == b" ").count();
    assert!((5..=12).contains(&spaces), "{spaces} spaces");
    let body: Vec<u8> = chunks.iter().flat_map(|(_, b)| b.clone()).collect();
    let text = std::str::from_utf8(&body).unwrap();
    assert!(
        text.starts_with("      ") && text.trim_start().starts_with('{'),
        "{text}"
    );
    let reply: Value = serde_json::from_str(text).unwrap();
    assert_eq!(reply["choices"][0]["message"]["content"], "Hello.");
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn a_slow_failure_sends_the_error_after_status_200() {
    let server = slow_server(std::time::Duration::from_millis(300), |_| {
        vec![
            Event::Started,
            Event::Failed("the GPU fell over".to_owned()),
        ]
    });
    for (path, body, id) in [
        ("/v1/chat/completions", chat_body(false), "slow-fail-chat"),
        ("/v1/messages", messages_body(false), "slow-fail-messages"),
    ] {
        let response = server
            .call(post(path, &body, &[("x-request-id", id)]))
            .await;
        assert_eq!(response.status(), 200);
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(bytes[0], b' ');
        let error: Value = serde_json::from_slice(&bytes).unwrap();
        let line = request_line(id).await;
        assert!(line.contains(" keepalive=committed"), "{line}");
        if path == "/v1/messages" {
            assert_eq!(error["type"], "error");
            assert_eq!(error["error"]["type"], "api_error");
            assert_eq!(error["error"]["message"], "the GPU fell over");
        } else {
            assert_eq!(error["error"]["type"], "server_error");
            assert_eq!(error["error"]["message"], "the GPU fell over");
            // The log records the real outcome, not the 200 the client got.
            assert!(
                line.contains(" status=500 ") && line.ends_with(" error=\"the GPU fell over\""),
                "{line}"
            );
        }
    }
}

#[tokio::test]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn a_client_that_leaves_during_keepalive_cancels_the_job() {
    let (left, gone) = std::sync::mpsc::channel();
    let left = Mutex::new(left);
    let server = Server::with_keepalive(4096, Some(KEEPALIVE), move |_, job| {
        let _ = job.events.send(Event::Started);
        // A long prefill, until the client leaves.
        while !job.events.is_closed() {
            thread::sleep(std::time::Duration::from_millis(5));
        }
        left.lock().unwrap().send(()).unwrap();
    });
    let response = server
        .call(post(
            "/v1/completions",
            &json!({"prompt": "Hi", "max_tokens": 4}),
            &[("x-request-id", "left-waiting")],
        ))
        .await;
    assert_eq!(response.status(), 200);
    let mut data = response.into_body().into_data_stream();
    assert_eq!(data.next().await.unwrap().unwrap(), " ");
    drop(data);
    gone.recv_timeout(std::time::Duration::from_secs(5))
        .expect("the job ends once its client leaves");
    let line = request_line("left-waiting").await;
    assert!(
        line.contains(" keepalive=committed") && line.contains(" client=disconnected"),
        "{line}"
    );
}

#[tokio::test(start_paused = true)]
#[ignore = "needs the tokenizer (SIPHON_MODEL)"]
async fn chat_streams_send_comments_before_the_first_token() {
    let (go, wait) = std::sync::mpsc::channel::<()>();
    let wait = Mutex::new(wait);
    let server = Server::with_engine(4096, move |vocab, job| {
        let _ = job.events.send(Event::Started);
        // A prefill that lasts until the test has seen the keep-alives.
        wait.lock().unwrap().recv().unwrap();
        for event in text_reply(vocab, "Hello.", 0) {
            let _ = job.events.send(event);
        }
    });
    for path in ["/v1/chat/completions", "/v1/completions"] {
        let body = if path == "/v1/completions" {
            json!({"prompt": "Hi", "max_tokens": 16, "stream": true})
        } else {
            chat_body(true)
        };
        let started = tokio::time::Instant::now();
        let response = server.call(post(path, &body, &[])).await;
        let mut data = response.into_body().into_data_stream();
        for _ in 0..2 {
            let frame = data.next().await.unwrap().unwrap();
            assert_eq!(frame, ":\n\n", "{path}");
        }
        assert!(
            started.elapsed() <= std::time::Duration::from_secs(30),
            "two comments in {:?}",
            started.elapsed()
        );
        go.send(()).unwrap();
        let rest = body::to_bytes(Body::from_stream(data), usize::MAX)
            .await
            .unwrap();
        assert!(
            std::str::from_utf8(&rest).unwrap().contains("data: "),
            "{path}"
        );
    }
}
