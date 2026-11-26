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
        let (engine, jobs) = Engine::fake(max_context);
        let labels = Arc::new(Mutex::new(Vec::new()));
        let taken = Arc::clone(&labels);
        let script_vocab = vocab();
        thread::spawn(move || {
            for submission in jobs {
                for job in submission {
                    taken.lock().unwrap().push(job.label.to_string());
                    for event in script(&script_vocab, &job) {
                        let _ = job.events.send(event);
                    }
                }
            }
        });
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
