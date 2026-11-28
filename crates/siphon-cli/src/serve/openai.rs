//! The OpenAI-compatible HTTP API.
//!
//! - `POST /v1/chat/completions`, streaming or not: messages rendered with the checkpoint's
//!   chat template, `tools` and `tool_choice` (`auto`, `none`, `required` or a named
//!   function), `parallel_tool_calls`, `reasoning_effort` or
//!   `chat_template_kwargs.enable_thinking`, `stop`, vLLM's `continue_final_message`, and
//!   `return_token_ids`.
//! - `POST /v1/completions`, streaming or not: a text prompt or token ids, with `stop` and
//!   `return_token_ids`.
//! - Anthropic's `POST /v1/messages` and `POST /v1/messages/count_tokens`
//!   ([`crate::serve::anthropic`]), which share the chat route's prompt rendering and output.
//! - OpenAI's `POST /v1/responses` ([`crate::serve::responses`]), which shares them too.
//! - `GET /metrics`, which needs no API key: Prometheus metrics ([`crate::serve::metrics`]).
//! - `GET /v1/models` and `GET /v1/models/{id}`, with `max_model_len`.
//! - The tokenizer routes of [`crate::serve::tokenize`].
//! - `GET /health` and `GET /version`, which need no API key.
//!
//! Both completion routes sample with `temperature`, `top_k`, `top_p`, `min_p` and `seed`,
//! with the checkpoint's `generation_config.json` for the fields a request leaves out
//! ([`crate::serve::request::sampling`]); `temperature` 0 decodes greedily. The penalties are
//! accepted and ignored. [`crate::serve::request`] lists the fields the server refuses, such as
//! `logprobs`. Stop strings match the reply's content,
//! not its reasoning or tool calls. Both routes take vLLM's `stop_token_ids`: the output
//! ends at the first of them, whose id the output keeps and whose text it leaves out, as
//! vLLM does. A choice reports the stop string or stop token that matched as vLLM's
//! `stop_reason`.
//!
//! Both routes take `n`: the request's `n` jobs share its prompt (see [`crate::serve::engine`]),
//! and the reply merges them into choices 0 to n-1 ([`Fanout`]). Choice `i` of a request
//! with a seed samples with seed + i, as vLLM does. A stream sends each choice's text in
//! chunks of one choice each, interleaved as the engine delivers it, and each choice ends
//! with its own chunk that holds its `finish_reason`. The usage counts the prompt once
//! and sums the choices' tokens.
//!
//! With an API key set, a request carries it as `Authorization: Bearer <key>` or as
//! Anthropic's `x-api-key: <key>`.

use std::{
    convert::Infallible,
    future::poll_fn,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll, ready},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Extension, Json, Router,
    body::{Body, Bytes, HttpBody},
    extract::{DefaultBodyLimit, FromRequest, Path, Request, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode, header},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures_util::StreamExt as _;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::{
    compression::{CompressionLayer, DefaultPredicate, Predicate as _},
    decompression::RequestDecompressionLayer,
};

use crate::serve::{
    anthropic,
    chat::{self, Ending, Thinking, Vocab},
    engine::{Engine, Event, Finish, Job, Refused, Sampling},
    invalid::{InvalidRequest, serde_detail},
    keepalive::{self, Uncompressed},
    log::RequestLog,
    metrics::{Metrics, Reason, Times, Tracker},
    parse::{Delta, Detokenizer, Parser, ToolCall, ToolPart},
    request::{self, SamplingDefaults, SamplingFields, ToolChoice},
    responses,
    stop::Stops,
    timings::Timings,
    tokenize::{self, DecodeRequest, EncodeRequest},
    vision::PromptImage,
};

pub struct App {
    pub vocab: Vocab,
    pub engine: Engine,
    pub metrics: Arc<Metrics>,
    /// The model id that `/v1/models` lists and responses report.
    pub name: String,
    /// The bearer token every route but `/health` and `/version` must carry, if set.
    pub api_key: Option<String>,
    pub started: u64,
    pub process_started: SystemTime,
    /// The sampling of fields a request leaves out.
    pub sampling: SamplingDefaults,
    /// `--greedy`: every request decodes greedily, whatever its sampling fields ask.
    pub greedy: bool,
    /// `--keepalive-after`: keep-alive bytes for replies that are not streamed.
    pub keepalive: Option<keepalive::KeepAlive>,
}

/// The sampling of each of a request's `n` choices. Under `--greedy` every choice would be
/// the same, so `n` above 1 is refused, as it is for temperature 0.
fn choice_samplings(
    app: &App,
    fields: &SamplingFields,
    n: usize,
) -> Result<Vec<Sampling>, ApiError> {
    if app.greedy && n > 1 {
        return Err(InvalidRequest::field(
            "n",
            format!("must be 1 when the server decodes greedily (--greedy), got {n}"),
        )
        .into());
    }
    Ok(request::samplings(fields, &app.sampling, n)?)
}

/// Request body bytes allowed per position of the context. A prompt of English text
/// takes about 5 bytes per token; JSON escapes and token-id arrays take more.
const BODY_BYTES_PER_POSITION: usize = 32;

/// The routes, behind the API key where one is set.
///
/// Request bodies may use `Content-Encoding: gzip`, or `zstd` with the `http-zstd` feature.
/// The body limit applies to the decompressed bytes. Responses negotiate compression via
/// `Accept-Encoding`, and tower-http leaves server-sent events uncompressed.
pub fn router(app: Arc<App>) -> Router {
    // axum's default limit of 2 MB holds a prompt of only about 400K tokens.
    let body_limit = (app.engine.max_context() * BODY_BYTES_PER_POSITION).max(2 << 20)
        + app
            .vocab
            .vision()
            .map_or(0, crate::serve::vision::Vision::body_bytes);
    let decompress = RequestDecompressionLayer::new();
    let compress = CompressionLayer::new();
    #[cfg(not(feature = "http-zstd"))]
    let (decompress, compress) = (decompress.no_zstd(), compress.no_zstd());
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/models/{*id}", get(model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/tokenize", post(tokenize_route))
        .route("/v1/tokenize", post(tokenize_route))
        .route("/detokenize", post(detokenize_route))
        .route("/v1/detokenize", post(detokenize_route))
        .route("/v1/token/encode", post(token_encode))
        .route("/v1/token/decode", post(token_decode))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/v1/responses", post(responses::responses))
        .layer(DefaultBodyLimit::max(body_limit))
        .route_layer(middleware::from_fn_with_state(Arc::clone(&app), authorize))
        .route("/health", get(health))
        .route("/metrics", get(metrics_route))
        .route(
            "/version",
            get(|| async {
                Json(json!({"version": env!("CARGO_PKG_VERSION"), "commit": crate::serve::COMMIT}))
            }),
        )
        .with_state(app)
        .layer(middleware::from_fn(request_id))
        .layer(decompress)
        .layer(compress.compress_when(DefaultPredicate::new().and(
            |_, _, _: &HeaderMap, extensions: &axum::http::Extensions| {
                extensions.get::<Uncompressed>().is_none()
            },
        )))
}

/// A request's id: the client's `x-request-id` or `x-client-request-id` when valid, else
/// one the server makes. Log lines carry it, and the response echoes it. It holds the
/// request's [`RequestLog`].
#[derive(Clone, Debug)]
pub(crate) struct RequestId {
    pub(crate) id: Arc<str>,
    pub(crate) log: Arc<RequestLog>,
}

/// The longest client request id the server takes.
const REQUEST_ID_BYTES: usize = 128;

impl RequestId {
    /// The client's id from `headers`, if one is valid: 1 to 128 printable ASCII bytes
    /// without spaces.
    fn from_client(headers: &HeaderMap) -> Option<Arc<str>> {
        ["x-request-id", "x-client-request-id"]
            .into_iter()
            .filter_map(|name| headers.get(name))
            .map(header::HeaderValue::as_bytes)
            .find(|id| {
                (1..=REQUEST_ID_BYTES).contains(&id.len()) && id.iter().all(u8::is_ascii_graphic)
            })
            .and_then(|id| std::str::from_utf8(id).ok())
            .map(Arc::from)
    }

    /// A new id: 64 random bits.
    fn generate() -> Arc<str> {
        use std::hash::{BuildHasher as _, Hasher as _};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u64(NEXT.fetch_add(1, Ordering::Relaxed));
        format!("req-{:016x}", hasher.finish()).into()
    }

    /// The log label of the job of the response `response_id`: choice `choice` of `n`.
    pub(crate) fn label(&self, response_id: &str, choice: usize, n: usize) -> Arc<str> {
        if n > 1 {
            format!("{} {response_id} choice {choice}", self.id).into()
        } else {
            format!("{} {response_id}", self.id).into()
        }
    }
}

/// An error response's message, which [`request_id`] puts in the request's log line.
#[derive(Clone, Debug)]
pub(crate) struct ErrorNote(pub(crate) Arc<str>);

/// Gives the request its [`RequestId`], puts it in the response's `x-request-id`, and
/// ends its [`RequestLog`]: with the response's status, or as a client that left before
/// the response or before the end of its body.
async fn request_id(mut request: Request, next: Next) -> Response {
    let id = RequestId::from_client(request.headers()).unwrap_or_else(RequestId::generate);
    let log = Arc::new(RequestLog::new(
        Arc::clone(&id),
        request.method().clone(),
        request.uri().path(),
    ));
    request.extensions_mut().insert(RequestId {
        id: Arc::clone(&id),
        log: Arc::clone(&log),
    });
    // Dropped with this future when the client leaves before the response.
    let pending = Pending(Some(log));
    let mut response = next.run(request).await;
    let log = pending.done();
    let error = response
        .extensions()
        .get::<ErrorNote>()
        .map(|e| e.0.clone());
    let error = error.or_else(|| {
        (response.status() == StatusCode::NOT_FOUND).then(|| Arc::from("no such route"))
    });
    log.respond(response.status(), error.as_deref());
    if let Ok(value) = header::HeaderValue::from_str(&id) {
        response.headers_mut().insert("x-request-id", value);
    }
    response.map(|body| {
        Body::new(LoggedBody {
            inner: body,
            log: Some(log),
        })
    })
}

/// A request's log while its handler runs, which records a client that left if dropped.
struct Pending(Option<Arc<RequestLog>>);

impl Pending {
    fn done(mut self) -> Arc<RequestLog> {
        self.0.take().expect("taken once")
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(log) = &self.0 {
            log.disconnected();
        }
    }
}

/// A response body that holds the request's log until its end, and records a client that
/// left before it.
struct LoggedBody {
    inner: Body,
    log: Option<Arc<RequestLog>>,
}

impl HttpBody for LoggedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<http_body::Frame<Bytes>, axum::Error>>> {
        let frame = ready!(std::pin::Pin::new(&mut self.inner).poll_frame(cx));
        if frame.is_none() {
            self.log = None;
        }
        Poll::Ready(frame)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for LoggedBody {
    fn drop(&mut self) {
        if let Some(log) = &self.log
            && !self.inner.is_end_stream()
        {
            log.disconnected();
        }
    }
}

/// `ok`, or a 503 once the server drains, so that the proxy in front stops sending
/// requests here.
async fn health(State(app): State<Arc<App>>) -> Response {
    if app.engine.draining() {
        return (StatusCode::SERVICE_UNAVAILABLE, "draining").into_response();
    }
    "ok".into_response()
}

/// An OpenAI-style error response.
#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
    /// The request field the error is about, OpenAI's error `param`.
    pub(crate) param: Option<String>,
    /// Seconds for a `Retry-After` header.
    pub(crate) retry_after: Option<u32>,
    /// OpenAI's error `code`, when it is one clients act on.
    pub(crate) code: Option<&'static str>,
}

/// Seconds a client waits before it retries a request that a draining server refused. The
/// proxy in front sends the retry to the server that replaces this one.
const DRAIN_RETRY_AFTER: u32 = 1;

impl ApiError {
    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            param: None,
            retry_after: None,
            code: None,
        }
    }

    /// A 400 for a prompt that fills the context, with OpenAI's code for it, which Codex
    /// reads to compact the conversation.
    pub(crate) fn context_exceeded(message: impl Into<String>) -> Self {
        Self {
            code: Some("context_length_exceeded"),
            param: Some("messages".to_owned()),
            ..Self::bad_request(message)
        }
    }

    /// `response` with this error's message for the request's log line.
    pub(crate) fn noted(&self, mut response: Response) -> Response {
        response
            .extensions_mut()
            .insert(ErrorNote(self.message.as_str().into()));
        response
    }

    /// A 503 with `Retry-After` for a request that arrives while the server drains.
    pub(crate) fn draining() -> Self {
        Self {
            retry_after: Some(DRAIN_RETRY_AFTER),
            ..Self::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "the server is shutting down, retry",
            )
        }
    }

    /// `response` with this error's `Retry-After` header, if it has one.
    pub(crate) fn with_retry_after(&self, mut response: Response) -> Response {
        if let Some(seconds) = self.retry_after {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, header::HeaderValue::from(seconds));
        }
        response
    }

    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }
}

impl From<InvalidRequest> for ApiError {
    fn from(e: InvalidRequest) -> Self {
        Self {
            param: e.param().map(str::to_owned),
            ..Self::bad_request(e.to_string())
        }
    }
}

impl From<JsonRejection> for ApiError {
    /// A 413 for a body past the limit, else a 400; a body that does not fit the route's
    /// fields names the field, as [`InvalidRequest`] does.
    fn from(e: JsonRejection) -> Self {
        let status = match e.status() {
            StatusCode::PAYLOAD_TOO_LARGE => StatusCode::PAYLOAD_TOO_LARGE,
            _ => StatusCode::BAD_REQUEST,
        };
        let text = e.body_text();
        let Some(rest) =
            text.strip_prefix("Failed to deserialize the JSON body into the target type: ")
        else {
            return Self::new(status, text);
        };
        // serde_path_to_error puts the field's path first, unless it is the body itself.
        match rest.split_once(": ") {
            Some((path, detail)) if !path.contains(char::is_whitespace) => {
                InvalidRequest::field(path, serde_detail(detail)).into()
            }
            _ => InvalidRequest::new(serde_detail(rest)).into(),
        }
    }
}

/// A route's JSON body, as `Json` extracts it, with an error that names the field when the
/// body does not fit.
///
/// Serde loses the path of a field in a `#[serde(flatten)]` struct, such as a request's
/// [`SamplingFields`], so for an error without one this reads the body's sampling fields
/// alone, whose errors keep it.
pub(crate) struct JsonBody<T>(pub(crate) T);

impl<T: DeserializeOwned, S: Send + Sync> FromRequest<S> for JsonBody<T> {
    type Rejection = ApiError;

    async fn from_request(request: Request, state: &S) -> Result<Self, ApiError> {
        let json = request
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| {
                let mime = v.split(';').next().unwrap_or("").trim();
                mime.starts_with("application/") && mime.ends_with("json")
            });
        let bytes = Bytes::from_request(request, state).await.map_err(|e| {
            let status = match e.status() {
                StatusCode::PAYLOAD_TOO_LARGE => StatusCode::PAYLOAD_TOO_LARGE,
                _ => StatusCode::BAD_REQUEST,
            };
            ApiError::new(status, e.body_text())
        })?;
        if !json {
            return Err(ApiError::bad_request(
                "expected the header Content-Type: application/json",
            ));
        }
        match Json::<T>::from_bytes(&bytes) {
            Ok(Json(value)) => Ok(Self(value)),
            Err(e) => {
                let error = ApiError::from(e);
                if error.param.is_some() {
                    return Err(error);
                }
                let fields = &mut serde_json::Deserializer::from_slice(&bytes);
                match serde_path_to_error::deserialize::<_, SamplingFields>(fields) {
                    Err(e) if e.path().iter().next().is_some() => Err(InvalidRequest::field(
                        e.path().to_string(),
                        serde_detail(&e.inner().to_string()),
                    )
                    .into()),
                    _ => Err(error),
                }
            }
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (kind, code) = match self.status {
            StatusCode::BAD_REQUEST | StatusCode::PAYLOAD_TOO_LARGE => {
                ("invalid_request_error", Value::Null)
            }
            StatusCode::NOT_FOUND => ("invalid_request_error", json!("model_not_found")),
            StatusCode::UNAUTHORIZED => ("authentication_error", Value::Null),
            _ => ("server_error", Value::Null),
        };
        let code = self.code.map_or(code, Value::from);
        let body = json!({"error": {
            "message": self.message, "type": kind, "param": self.param, "code": code,
        }});
        self.noted(self.with_retry_after((self.status, Json(body)).into_response()))
    }
}

async fn authorize(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    if let Some(key) = &app.api_key
        && !authorized(&headers, key)
    {
        let error = ApiError::new(StatusCode::UNAUTHORIZED, "invalid API key");
        if anthropic::serves(request.uri().path()) {
            return anthropic::Error(error).into_response();
        }
        return error.into_response();
    }
    next.run(request).await
}

/// Whether `headers` carry `key` as `Authorization: Bearer <key>` or as `x-api-key: <key>`,
/// the header Anthropic's clients send.
pub(crate) fn authorized(headers: &HeaderMap, key: &str) -> bool {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let api_key = headers.get("x-api-key").and_then(|v| v.to_str().ok());
    [bearer, api_key]
        .into_iter()
        .flatten()
        .any(|given| same_bytes(given.as_bytes(), key.as_bytes()))
}

/// Compares in time that depends only on the lengths.
fn same_bytes(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn model_card(app: &App) -> Value {
    json!({
        "id": app.name,
        "object": "model",
        "created": app.started,
        "owned_by": "siphon",
        "root": app.name,
        "parent": null,
        "max_model_len": app.engine.max_context(),
    })
}

async fn metrics_route(State(app): State<Arc<App>>) -> Response {
    let mut text = app.metrics.render();
    app.engine.write_metrics(&mut text);
    super::metrics::write_process_start(&mut text, app.process_started);
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        text,
    )
        .into_response()
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [model_card(&app)]}))
}

async fn model(
    State(app): State<Arc<App>>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    if id != app.name {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("The model `{id}` does not exist."),
        ));
    }
    Ok(Json(model_card(&app)))
}

/// Runs `f` off the async workers: tokenizing a long prompt takes milliseconds.
pub(crate) async fn blocking<T: Send + 'static>(
    app: &Arc<App>,
    f: impl FnOnce(&App) -> Result<T, ApiError> + Send + 'static,
) -> Result<T, ApiError> {
    let app = Arc::clone(app);
    tokio::task::spawn_blocking(move || f(&app))
        .await
        .map_err(|e| ApiError::internal(e.to_string()))?
}

async fn tokenize_route(
    State(app): State<Arc<App>>,
    JsonBody(request): JsonBody<EncodeRequest>,
) -> Result<Json<Value>, ApiError> {
    let ids = blocking(&app, |app| Ok(tokenize::encode(&app.vocab, request)?)).await?;
    Ok(Json(tokenize::tokenized(&ids, app.engine.max_context())))
}

async fn token_encode(
    State(app): State<Arc<App>>,
    JsonBody(request): JsonBody<EncodeRequest>,
) -> Result<Json<Value>, ApiError> {
    let ids = blocking(&app, |app| Ok(tokenize::encode(&app.vocab, request)?)).await?;
    Ok(Json(tokenize::encoded(&ids)))
}

async fn detokenize_route(
    State(app): State<Arc<App>>,
    JsonBody(request): JsonBody<DecodeRequest>,
) -> Result<Json<Value>, ApiError> {
    let text = blocking(&app, move |app| Ok(tokenize::decode(&app.vocab, &request)?)).await?;
    Ok(Json(json!({"prompt": text})))
}

async fn token_decode(
    State(app): State<Arc<App>>,
    JsonBody(request): JsonBody<DecodeRequest>,
) -> Result<Json<Value>, ApiError> {
    let text = blocking(&app, move |app| Ok(tokenize::decode(&app.vocab, &request)?)).await?;
    Ok(Json(json!({"text": text})))
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Value>,
    stream: Option<bool>,
    stream_options: Option<StreamOptions>,
    max_tokens: Option<u64>,
    max_completion_tokens: Option<u64>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
    reasoning_effort: Option<Value>,
    chat_template_kwargs: Option<Value>,
    n: Option<u64>,
    stop: Option<Value>,
    stop_token_ids: Option<Vec<Value>>,
    response_format: Option<Value>,
    /// vLLM's extensions.
    continue_final_message: Option<bool>,
    add_generation_prompt: Option<bool>,
    return_token_ids: Option<bool>,
    ignore_eos: Option<bool>,
    #[serde(flatten)]
    sampling: SamplingFields,
    /// Every other field: refused ones are checked, the rest ignored.
    #[serde(flatten)]
    other: Map<String, Value>,
}

#[derive(Deserialize)]
struct StreamOptions {
    include_usage: Option<bool>,
}

/// How a reply goes back to the client.
pub(crate) enum Delivery {
    Whole,
    /// Server-sent events, ending with a usage chunk if `include_usage`.
    Stream {
        include_usage: bool,
    },
}

impl Delivery {
    fn new(stream: Option<bool>, options: Option<StreamOptions>) -> Self {
        if stream.unwrap_or(false) {
            Self::Stream {
                include_usage: options.and_then(|o| o.include_usage).unwrap_or(false),
            }
        } else {
            Self::Whole
        }
    }
}

/// A validated request, ready for the engine.
pub(crate) struct Prepared {
    pub(crate) prompt: Vec<u32>,
    /// The prompt's images, with `--vision`.
    pub(crate) images: Vec<PromptImage>,
    pub(crate) max_tokens: usize,
    pub(crate) delivery: Delivery,
    /// vLLM's `return_token_ids`: the prompt's and the completion's ids.
    pub(crate) return_token_ids: bool,
    /// vLLM's `ignore_eos`: run to `max_tokens` through the model's end tokens.
    pub(crate) ignore_eos: bool,
    /// Each choice's sampling, one per choice.
    pub(crate) samplings: Vec<Sampling>,
    /// The output of a choice before its first token, for every choice.
    pub(crate) output: Output,
    /// The request's thinking, for the log; `None` for a raw completion.
    pub(crate) thinking: Option<Thinking>,
}

fn prepare_chat(app: &App, request: ChatRequest) -> Result<Prepared, ApiError> {
    request::refuse(&request.other)?;
    let n = request::choices(request.n)?;
    request::response_format(request.response_format.as_ref())?;
    let stops = request::stops("stop", request.stop.as_ref())?;
    let stop_ids = tokenize::token_ids(
        &app.vocab,
        "stop_token_ids",
        &request.stop_token_ids.unwrap_or_default(),
    )?;
    let ending = request::ending(
        request.continue_final_message,
        request.add_generation_prompt,
    )?;
    if matches!(ending, Ending::Bare) {
        return Err(InvalidRequest::field(
            "add_generation_prompt",
            "false needs continue_final_message true; a chat prompt ends either with the \
             model's turn or inside the last assistant message",
        )
        .into());
    }
    let mut tools = request.tools.unwrap_or_default();
    let choice = ToolChoice::parse(request.tool_choice.as_ref())?;
    if choice == ToolChoice::None {
        tools.clear();
    }
    let parallel = request.parallel_tool_calls != Some(false);
    let thinking = chat::request_thinking(
        "reasoning_effort",
        request.reasoning_effort.as_ref(),
        request.chat_template_kwargs.as_ref(),
    )?;
    let (ending, calls) = tool_ending(&choice, &tools, parallel, thinking.ending(ending, &tools))?;
    let rendered =
        app.vocab
            .render_prompt(&request.messages, &tools, thinking.effort(), &ending)?;
    let max_tokens = max_tokens(
        app.engine.max_context(),
        rendered.ids.len(),
        if request.max_completion_tokens.is_some() {
            "max_completion_tokens"
        } else {
            "max_tokens"
        },
        request.max_completion_tokens.or(request.max_tokens),
    )?;
    Ok(Prepared {
        prompt: rendered.ids,
        images: rendered.images,
        max_tokens,
        delivery: Delivery::new(request.stream, request.stream_options),
        return_token_ids: request.return_token_ids.unwrap_or(false),
        ignore_eos: request.ignore_eos.unwrap_or(false),
        samplings: choice_samplings(app, &request.sampling, n)?,
        output: Output::new(
            Decoder::Chat(Parser::new(&app.vocab, &ending)),
            Stops::new(stops),
            stop_ids,
            calls,
        ),
        thinking: Some(thinking),
    })
}

/// How a chat prompt with `tools` ends under `choice`, given the `ending` its other fields
/// ask for, and which tool calls end the output.
pub(crate) fn tool_ending<'a>(
    choice: &'a ToolChoice,
    tools: &'a [Value],
    parallel: bool,
    ending: Ending<'a>,
) -> Result<(Ending<'a>, Calls), ApiError> {
    // A forced call skips the model's thinking, and after it the model tends to write text
    // about a call it did not decide on, so the output ends with the forced calls.
    let (call, calls) = match choice {
        ToolChoice::Auto | ToolChoice::None => {
            (None, if parallel { Calls::Free } else { Calls::One })
        }
        ToolChoice::Function(name) => (Some(Ending::Call(name)), Calls::One),
        ToolChoice::Required => {
            let ending = match tools {
                [tool] => chat::tool_function_name(tool).map_or(Ending::AnyCall, Ending::Call),
                _ => Ending::AnyCall,
            };
            (
                Some(ending),
                if parallel { Calls::Only } else { Calls::One },
            )
        }
    };
    let ending = match (call, ending) {
        (None, ending) => ending,
        (Some(call), Ending::Generate | Ending::Answer) => call,
        (Some(_), _) => {
            return Err(InvalidRequest::field(
                "tool_choice",
                "forces a tool call, which cannot continue the last assistant message; send \
                 \"auto\" or end the messages with a user or tool message",
            )
            .into());
        }
    };
    Ok((ending, calls))
}

/// The tokens a request may generate: `requested`, capped at the positions left after a
/// prompt of `prompt_len` tokens in a context of `max_context`, or all of them. A cap past
/// the context is cut to fit, as a client's token count can differ from the server's, so
/// the reply finishes with `length` there. `param` names the cap's field in errors.
pub(crate) fn max_tokens(
    max_context: usize,
    prompt_len: usize,
    param: &str,
    requested: Option<u64>,
) -> Result<usize, ApiError> {
    let room = max_context.saturating_sub(prompt_len);
    if room == 0 {
        return Err(ApiError::context_exceeded(format!(
            "the prompt has {prompt_len} tokens, and this server's context holds \
             {max_context} for the prompt and the reply together; send a prompt of at most \
             {} tokens",
            max_context - 1
        )));
    }
    match requested {
        None => Ok(room),
        Some(0) => Err(InvalidRequest::field(
            param,
            format!("expected an integer of at least 1, got 0; up to {room} fit after the {prompt_len}-token prompt"),
        )
        .into()),
        Some(n) => Ok(usize::try_from(n).unwrap_or(usize::MAX).min(room)),
    }
}

/// Queues a job of request `id`'s reply `reply_id` and returns its events.
#[expect(clippy::too_many_arguments, reason = "a job's parts")]
pub(crate) fn submit(
    app: &App,
    prompt: Vec<u32>,
    images: Vec<PromptImage>,
    max_tokens: usize,
    ignore_eos: bool,
    sampling: Sampling,
    id: &RequestId,
    reply_id: &str,
) -> Result<Results, ApiError> {
    let mut results = submit_choices(
        app,
        prompt,
        images,
        max_tokens,
        ignore_eos,
        &[sampling],
        id,
        reply_id,
    )?;
    results
        .pop()
        .ok_or_else(|| ApiError::internal("the engine took no job"))
}

/// Queues one job per entry of `samplings`, all of the same prompt and `images`, and
/// returns their events in that order. The jobs fit the queue together or not at all.
/// They are request `id`'s reply `reply_id`, which names them in the log.
#[expect(clippy::too_many_arguments, reason = "a job's parts")]
pub(crate) fn submit_choices(
    app: &App,
    prompt: Vec<u32>,
    images: Vec<PromptImage>,
    max_tokens: usize,
    ignore_eos: bool,
    samplings: &[Sampling],
    id: &RequestId,
    reply_id: &str,
) -> Result<Vec<Results>, ApiError> {
    let most = app.engine.max_jobs();
    let n = samplings.len();
    if n > most {
        return Err(InvalidRequest::field(
            "n",
            format!("is {n}, more than the {most} requests this server runs and queues at once"),
        )
        .into());
    }
    let prompt_len = prompt.len();
    let mut key = app
        .vocab
        .vision()
        .and_then(|v| v.cache_key(&prompt, &images));
    let images: crate::serve::vision::Images = images.into();
    let mut prompt = Some(prompt);
    let last = samplings.len().saturating_sub(1);
    let (jobs, receivers): (Vec<Job>, Vec<_>) = samplings
        .iter()
        .enumerate()
        .map(|(i, &sampling)| {
            let (events, receiver) = mpsc::unbounded_channel();
            // Move the prompt and cache key into the last (or only) job to avoid cloning
            // multi-megabyte prompts on the common `n == 1` path.
            let job_prompt = if i == last {
                prompt.take().unwrap_or_default()
            } else {
                prompt.as_deref().unwrap_or_default().to_vec()
            };
            let job_key = if i == last { key.take() } else { key.clone() };
            let job = Job::new(
                job_prompt,
                max_tokens,
                ignore_eos,
                sampling,
                events,
                id.label(reply_id, i, n),
            )
            .with_images(std::sync::Arc::clone(&images), job_key)
            .with_log(Arc::downgrade(&id.log));
            (job, receiver)
        })
        .unzip();
    app.engine.submit(jobs).map_err(|e| match e {
        Refused::Busy => ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "the request queue is full ({most} requests run and wait at most), retry later"
            ),
        ),
        Refused::Draining => ApiError::draining(),
        Refused::Stopped => ApiError::internal("the engine has stopped"),
    })?;
    // The prompt counts once in the metrics, with the first choice.
    Ok(receivers
        .into_iter()
        .enumerate()
        .map(|(i, receiver)| Results {
            receiver,
            tracker: app.metrics.track(if i == 0 { prompt_len } else { 0 }),
            log: Some(Arc::clone(&id.log)),
        })
        .collect())
}

/// One job's events, recorded in the metrics as they arrive. Dropping it cancels the job,
/// counts it as aborted unless it finished, and gives the request's log the job's end.
pub(crate) struct Results {
    pub(crate) receiver: mpsc::UnboundedReceiver<Event>,
    pub(crate) tracker: Tracker,
    pub(crate) log: Option<Arc<RequestLog>>,
}

impl Drop for Results {
    fn drop(&mut self) {
        self.tracker.finish(Reason::Abort);
        if let Some(log) = &self.log {
            let t = &self.tracker;
            let reason = t.reason().unwrap_or(Reason::Abort);
            log.job(t.times(), t.generated(), t.cached_tokens(), reason);
        }
    }
}

impl Results {
    /// The next event, with a closed channel as a failure.
    pub(crate) async fn next(&mut self) -> Event {
        poll_fn(|cx| self.poll_next(cx)).await
    }

    fn poll_next(&mut self, cx: &mut Context<'_>) -> Poll<Event> {
        let event = ready!(self.receiver.poll_recv(cx))
            .unwrap_or_else(|| Event::Failed("the engine dropped the request".to_owned()));
        match &event {
            Event::Started => self.tracker.started(),
            Event::Cached(n) => self.tracker.cached(*n),
            Event::Images(stats) => self.tracker.images(*stats),
            Event::Round => self.tracker.round(),
            Event::SmallTokens(_, len) => self.tracker.tokens(usize::from(*len)),
            Event::Tokens(ids) => self.tracker.tokens(ids.len()),
            Event::Done(finish) => self.tracker.finish(reason(*finish)),
            Event::Failed(message) => {
                self.tracker.finish(Reason::Error);
                if let Some(log) = &self.log {
                    log.failed(message);
                }
            }
        }
        Poll::Ready(event)
    }

    /// Ends the job with `finish`, which a stop the output matched may have set first, and
    /// returns when its events arrived.
    pub(crate) fn close(mut self, finish: Finish) -> Times {
        self.tracker.finish(reason(finish));
        self.tracker.times()
    }
}

/// The jobs of one request's choices, merged. Dropping it cancels the jobs still open.
pub(crate) struct Fanout {
    /// Each choice's events, `None` once the choice has ended.
    choices: Vec<Option<Results>>,
    /// When each choice's events arrived so far.
    times: Vec<Times>,
    /// The choice to look at first, which turns, so that no choice waits behind another.
    turn: usize,
}

impl Fanout {
    pub(crate) fn new(results: Vec<Results>) -> Self {
        Self {
            times: results.iter().map(|r| r.tracker.times()).collect(),
            choices: results.into_iter().map(Some).collect(),
            turn: 0,
        }
    }

    /// The request's timings so far, for a prompt of `prompt_tokens` with `cached` read
    /// from the prefix cache and choices that generated `predicted_n` tokens in all.
    pub(crate) fn timings(
        &self,
        prompt_tokens: usize,
        cached: usize,
        predicted_n: usize,
    ) -> Timings {
        Timings::new(&self.times, prompt_tokens, cached, predicted_n)
    }

    /// The next event of an open choice, with the choice's index. Waits forever when no
    /// choice is open.
    pub(crate) async fn next(&mut self) -> (usize, Event) {
        poll_fn(|cx| {
            let n = self.choices.len();
            for i in (0..n).map(|k| (self.turn + k) % n) {
                if let Some(results) = &mut self.choices[i]
                    && let Poll::Ready(event) = results.poll_next(cx)
                {
                    self.times[i] = results.tracker.times();
                    self.turn = (i + 1) % n;
                    return Poll::Ready((i, event));
                }
            }
            Poll::Pending
        })
        .await
    }

    /// Ends choice `i` with `finish`, which a stop the output matched may have set first,
    /// and cancels its job if it still runs.
    pub(crate) fn close(&mut self, i: usize, finish: Finish) {
        if let Some(results) = self.choices[i].take() {
            self.times[i] = results.close(finish);
        }
    }
}

fn reason(finish: Finish) -> Reason {
    match finish {
        Finish::Stop => Reason::Stop,
        Finish::Length => Reason::Length,
    }
}

async fn chat_completions(
    State(app): State<Arc<App>>,
    Extension(id): Extension<RequestId>,
    JsonBody(request): JsonBody<ChatRequest>,
) -> Response {
    let keepalive = app.keepalive.filter(|_| request.stream != Some(true));
    let log = Arc::clone(&id.log);
    keepalive::respond(keepalive, &log, async move {
        let prepared = blocking(&app, |app| prepare_chat(app, request)).await?;
        respond(app, prepared, Kind::Chat, &id).await
    })
    .await
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: Value,
    max_tokens: Option<u64>,
    stream: Option<bool>,
    stream_options: Option<StreamOptions>,
    n: Option<u64>,
    best_of: Option<u64>,
    stop: Option<Value>,
    stop_token_ids: Option<Vec<Value>>,
    return_token_ids: Option<bool>,
    ignore_eos: Option<bool>,
    #[serde(flatten)]
    sampling: SamplingFields,
    #[serde(flatten)]
    other: Map<String, Value>,
}

fn prepare_completion(app: &App, request: CompletionRequest) -> Result<Prepared, ApiError> {
    request::refuse(&request.other)?;
    let n = request::choices(request.n)?;
    request::best_of(request.best_of, n)?;
    let stops = request::stops("stop", request.stop.as_ref())?;
    let stop_ids = tokenize::token_ids(
        &app.vocab,
        "stop_token_ids",
        &request.stop_token_ids.unwrap_or_default(),
    )?;
    let accepted = "a string, an array of token ids, or an array of one string or one \
                    array of token ids";
    let prompt = match request.prompt {
        Value::String(text) => app.vocab.encode(&text),
        Value::Array(items) if items.iter().all(Value::is_u64) => {
            tokenize::token_ids(&app.vocab, "prompt", &items)?
        }
        Value::Array(items) => match items.as_slice() {
            [Value::String(text)] => app.vocab.encode(text),
            [Value::Array(ids)] => tokenize::token_ids(&app.vocab, "prompt[0]", ids)?,
            [_, _, ..] => {
                return Err(InvalidRequest::field(
                    "prompt",
                    format!(
                        "holds {} prompts; this server takes one prompt per request",
                        items.len()
                    ),
                )
                .into());
            }
            _ => {
                return Err(
                    InvalidRequest::expected("prompt", accepted, &Value::Array(items)).into(),
                );
            }
        },
        other => return Err(InvalidRequest::expected("prompt", accepted, &other).into()),
    };
    if prompt.is_empty() {
        return Err(InvalidRequest::field("prompt", "is empty; send at least one token").into());
    }
    let max_tokens = max_tokens(
        app.engine.max_context(),
        prompt.len(),
        "max_tokens",
        request.max_tokens,
    )?;
    Ok(Prepared {
        prompt,
        images: Vec::new(),
        max_tokens,
        delivery: Delivery::new(request.stream, request.stream_options),
        return_token_ids: request.return_token_ids.unwrap_or(false),
        ignore_eos: request.ignore_eos.unwrap_or(false),
        samplings: choice_samplings(app, &request.sampling, n)?,
        output: Output::new(
            Decoder::Text(Detokenizer::default()),
            Stops::new(stops),
            stop_ids,
            Calls::Free,
        ),
        thinking: None,
    })
}

/// Raw completion of a text prompt, encoded with special tokens as plain text, or of a
/// token-id prompt. The text returned includes any special tokens generated.
async fn completions(
    State(app): State<Arc<App>>,
    Extension(id): Extension<RequestId>,
    JsonBody(request): JsonBody<CompletionRequest>,
) -> Response {
    let keepalive = app.keepalive.filter(|_| request.stream != Some(true));
    let log = Arc::clone(&id.log);
    keepalive::respond(keepalive, &log, async move {
        let prepared = blocking(&app, |app| prepare_completion(app, request)).await?;
        respond(app, prepared, Kind::Text, &id).await
    })
    .await
}

/// Runs a prepared request and answers it, streaming or not.
async fn respond(
    app: Arc<App>,
    prepared: Prepared,
    kind: Kind,
    id: &RequestId,
) -> Result<Response, ApiError> {
    let Prepared {
        prompt,
        images,
        max_tokens,
        delivery,
        return_token_ids,
        ignore_eos,
        samplings,
        output,
        thinking,
    } = prepared;
    let reply = Reply::new(&app, prompt.len(), kind);
    let streaming = matches!(delivery, Delivery::Stream { .. });
    id.log.reply(&reply.id, streaming, thinking, prompt.len());
    let outputs = vec![output; samplings.len()];
    let prompt_ids = return_token_ids.then(|| prompt.clone());
    let results = submit_choices(
        &app, prompt, images, max_tokens, ignore_eos, &samplings, id, &reply.id,
    )?;
    let fanout = Fanout::new(results);
    if let Delivery::Stream { include_usage } = delivery {
        let (out, rx) = mpsc::channel(64);
        let chunks = Chunks {
            out,
            reply,
            prompt_ids,
        };
        tokio::spawn(
            async move { stream(&app.vocab, fanout, outputs, chunks, include_usage).await },
        );
        let events = ReceiverStream::new(rx)
            .map(|data: String| Ok::<_, Infallible>(SseEvent::default().data(data)));
        let body = Sse::new(events).keep_alive(KeepAlive::default());
        return Ok(body.into_response());
    }
    // Runs in the handler, so that a client that leaves drops the jobs.
    let (body, timings) = whole(&app.vocab, fanout, outputs, reply, prompt_ids).await?;
    Ok(with_server_timing(Json(body).into_response(), &timings))
}

/// `response` with a `Server-Timing` header of `timings`.
pub(crate) fn with_server_timing(mut response: Response, timings: &Timings) -> Response {
    response
        .headers_mut()
        .insert("server-timing", timings.server_timing());
    response
}

/// The body of a reply that is not streamed, with one choice per entry of `outputs`, and
/// its timings, which the body holds too.
async fn whole(
    vocab: &Vocab,
    mut fanout: Fanout,
    mut outputs: Vec<Output>,
    mut reply: Reply,
    prompt_ids: Option<Vec<u32>>,
) -> Result<(Value, Timings), ApiError> {
    let mut messages: Vec<Delta> = outputs.iter().map(|_| Delta::default()).collect();
    let mut finishes: Vec<Option<Finish>> = vec![None; outputs.len()];
    while finishes.iter().any(Option::is_none) {
        let (i, event) = fanout.next().await;
        let output = &mut outputs[i];
        let finish = match event {
            Event::Started | Event::Round | Event::Images(_) => continue,
            Event::Cached(n) => {
                if i == 0 {
                    reply.cached_tokens = n;
                }
                continue;
            }
            Event::SmallTokens(buf, len) => {
                output.push_into(vocab, &buf[..usize::from(len)], &mut messages[i]);
                if !output.done {
                    continue;
                }
                Finish::Stop
            }
            Event::Tokens(ids) => {
                output.push_into(vocab, &ids, &mut messages[i]);
                if !output.done {
                    continue;
                }
                Finish::Stop
            }
            Event::Done(finish) => finish,
            Event::Failed(e) => return Err(ApiError::internal(e)),
        };
        // Ends the job if a stop ended the output first.
        fanout.close(i, finish);
        append(&mut messages[i], output.finish(vocab, finish));
        finishes[i] = Some(finish);
    }
    let choices = outputs
        .iter()
        .zip(messages)
        .zip(finishes)
        .enumerate()
        .map(|(i, ((output, message), finish))| {
            let finish = finish.unwrap_or(Finish::Stop);
            let mut choice = reply.choice(i, message);
            choice["finish_reason"] = json!(output.finish_reason(finish));
            choice["stop_reason"] = output.stop_reason();
            if prompt_ids.is_some() {
                choice["token_ids"] = json!(output.ids);
            }
            choice
        })
        .collect();
    let mut body = reply.body(false, choices);
    body["usage"] = reply.usage(&outputs);
    if let Some(ids) = prompt_ids {
        body["prompt_token_ids"] = json!(ids);
    }
    let timings = reply.timings(&fanout, &outputs);
    body["timings"] = timings.json();
    Ok((body, timings))
}

pub(crate) fn append(message: &mut Delta, delta: Delta) {
    message.content += &delta.content;
    message.reasoning += &delta.reasoning;
    message.tool_calls.extend(delta.tool_calls);
    for part in delta.tool_parts {
        message.tool_part(part);
    }
}

/// Turns generated ids into reply text.
#[derive(Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per choice, made once per request"
)]
pub(crate) enum Decoder {
    /// Chat: reasoning, content and tool calls.
    Chat(Parser),
    /// Raw completion: all the text, special tokens included.
    Text(Detokenizer),
}

/// Which tool calls end an output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Calls {
    /// None do.
    Free,
    /// The first call does (`parallel_tool_calls: false`, or a named `tool_choice`).
    One,
    /// The output holds only calls: text or thinking after one ends it
    /// (`tool_choice: required`).
    Only,
}

/// One job's output on its way to the client, cut at the first stop string or stop token.
#[derive(Clone)]
pub(crate) struct Output {
    decoder: Decoder,
    stops: Stops,
    /// vLLM's `stop_token_ids`.
    stop_ids: Vec<u32>,
    /// The stop token that ended the output.
    stopped_by: Option<u32>,
    /// The generated ids the output took, then the turn's end token if the model ended it,
    /// as vLLM returns them. The engine does not say which end token it saw, so this is
    /// always `<|content_model_end_sampling|>`.
    ids: Vec<u32>,
    /// Generated ids taken, the end token excluded.
    pub(crate) tokens: usize,
    limit: Calls,
    calls: usize,
    /// Whether a stop string, a stop token or `limit` ended the output before the job did.
    pub(crate) done: bool,
}

impl Output {
    pub(crate) fn new(decoder: Decoder, stops: Stops, stop_ids: Vec<u32>, limit: Calls) -> Self {
        Self {
            decoder,
            stops,
            stop_ids,
            stopped_by: None,
            ids: Vec::new(),
            tokens: 0,
            limit,
            calls: 0,
            done: false,
        }
    }

    /// Takes generated ids, one at a time so that a stop ends the output at its token.
    #[cfg_attr(not(test), expect(dead_code, reason = "kept for tests"))]
    pub(crate) fn push(&mut self, vocab: &Vocab, ids: &[u32]) -> Delta {
        let mut out = Delta::default();
        self.push_into(vocab, ids, &mut out);
        out
    }

    /// Takes generated ids, appending directly into `out` to avoid per-token allocations.
    pub(crate) fn push_into(&mut self, vocab: &Vocab, ids: &[u32], out: &mut Delta) {
        if self.done || ids.is_empty() {
            return;
        }
        // Fast path: no stop strings, no stop token ids, and no tool-call limit.
        if self.stop_ids.is_empty() && self.stops.is_empty() && self.limit == Calls::Free {
            self.ids.extend_from_slice(ids);
            self.tokens += ids.len();
            let before_calls = out.tool_calls.len();
            match &mut self.decoder {
                Decoder::Chat(parser) => parser.push_into(vocab, ids, out),
                Decoder::Text(text) => {
                    for &id in ids {
                        text.push_into(vocab, id, &mut out.content);
                    }
                }
            }
            self.calls += out.tool_calls.len() - before_calls;
            return;
        }
        let mut scratch = Delta::default();
        for &id in ids {
            if self.done {
                break;
            }
            self.ids.push(id);
            self.tokens += 1;
            if self.stop_ids.contains(&id) {
                // The token ends the output without its text, as an end token would.
                self.end(vocab, out);
                if !self.done {
                    self.stopped_by = Some(id);
                    self.done = true;
                }
                break;
            }
            scratch.clear();
            match &mut self.decoder {
                Decoder::Chat(parser) => parser.push_into(vocab, &[id], &mut scratch),
                Decoder::Text(text) => text.push_into(vocab, id, &mut scratch.content),
            }
            self.take(&mut scratch, out);
        }
    }

    /// Ends the output after the job ended with `finish`, or after `done`.
    pub(crate) fn finish(&mut self, vocab: &Vocab, finish: Finish) -> Delta {
        let mut out = Delta::default();
        if self.done {
            return out;
        }
        self.end(vocab, &mut out);
        if !self.done && finish == Finish::Stop {
            self.ids.push(vocab.special.content_model_end_sampling);
        }
        out
    }

    /// Adds to `out` what the decoder and the stop strings held back, which can still
    /// match a stop string.
    fn end(&mut self, vocab: &Vocab, out: &mut Delta) {
        let mut delta = Delta::default();
        match &mut self.decoder {
            Decoder::Chat(parser) => parser.finish_into(vocab, &mut delta),
            Decoder::Text(text) => text.flush_into(vocab, &mut delta.content),
        }
        self.take(&mut delta, out);
        if !self.done {
            out.content += &self.stops.flush();
        }
    }

    /// Cuts `delta` at a stop string or where `limit` ends the output, and adds it to `out`.
    fn take(&mut self, delta: &mut Delta, out: &mut Delta) {
        self.stops.filter(&mut delta.content);
        if self.stops.matched().is_some() {
            self.done = true;
        }
        match self.limit {
            Calls::Free => {}
            Calls::One => {
                if !delta.tool_calls.is_empty() {
                    delta.tool_calls.truncate(1);
                    if let Some(end) = delta.tool_parts.iter().position(|p| *p == ToolPart::End) {
                        delta.tool_parts.truncate(end + 1);
                    }
                    self.done = true;
                }
            }
            Calls::Only => {
                if !delta.content.is_empty() || !delta.reasoning.is_empty() {
                    delta.content.clear();
                    delta.reasoning.clear();
                    self.done = true;
                }
            }
        }
        self.calls += delta.tool_calls.len();
        out.content.push_str(&delta.content);
        out.reasoning.push_str(&delta.reasoning);
        out.tool_calls.append(&mut delta.tool_calls);
        for part in delta.tool_parts.drain(..) {
            out.tool_part(part);
        }
    }

    pub(crate) fn finish_reason(&self, finish: Finish) -> &'static str {
        let finish = if self.done { Finish::Stop } else { finish };
        match finish {
            Finish::Stop if self.calls > 0 => "tool_calls",
            Finish::Stop => "stop",
            Finish::Length => "length",
        }
    }

    /// The stop string that ended the output, if one did.
    pub(crate) fn stop_string(&self) -> Option<&str> {
        self.stops.matched()
    }

    /// vLLM's `stop_reason`: the stop string or the stop token's id that matched, or null.
    fn stop_reason(&self) -> Value {
        match (self.stops.matched(), self.stopped_by) {
            (Some(stop), _) => json!(stop),
            (None, Some(id)) => json!(id),
            (None, None) => Value::Null,
        }
    }

    /// Generated ids inside thinking blocks.
    pub(crate) fn reasoning_tokens(&self) -> usize {
        match &self.decoder {
            Decoder::Chat(parser) => parser.reasoning_tokens(),
            Decoder::Text(_) => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Chat,
    Text,
}

/// The fields every response or chunk of one request shares.
struct Reply {
    id: String,
    created: u64,
    model: String,
    prompt_tokens: usize,
    /// Prompt tokens whose state came from the prefix cache.
    cached_tokens: usize,
    kind: Kind,
}

impl Reply {
    fn new(app: &App, prompt_tokens: usize, kind: Kind) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let created = now();
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let prefix = match kind {
            Kind::Chat => "chatcmpl",
            Kind::Text => "cmpl",
        };
        Self {
            id: format!("{prefix}-{created:x}{n:06x}"),
            created,
            model: app.name.clone(),
            prompt_tokens,
            cached_tokens: 0,
            kind,
        }
    }

    /// A response or chunk with `choices`.
    fn body(&self, chunk: bool, choices: Vec<Value>) -> Value {
        let object = match (self.kind, chunk) {
            (Kind::Chat, false) => "chat.completion",
            (Kind::Chat, true) => "chat.completion.chunk",
            (Kind::Text, _) => "text_completion",
        };
        json!({
            "id": self.id,
            "object": object,
            "created": self.created,
            "model": self.model,
            "choices": Value::Array(choices),
        })
    }

    /// Choice `index` of a whole response, without its finish reason.
    fn choice(&self, index: usize, message: Delta) -> Value {
        if self.kind == Kind::Text {
            return json!({"index": index, "text": message.content, "logprobs": null});
        }
        let mut body = json!({
            "role": "assistant",
            "content": if message.content.is_empty() && !message.tool_calls.is_empty() {
                Value::Null
            } else {
                Value::String(message.content)
            },
        });
        if !message.reasoning.is_empty() {
            body["reasoning_content"] = Value::String(message.reasoning);
        }
        if !message.tool_calls.is_empty() {
            body["tool_calls"] = message
                .tool_calls
                .iter()
                .map(|call| self.tool_call(call))
                .collect();
        }
        json!({"index": index, "message": body, "logprobs": null})
    }

    /// Streamed choice `index` with the fields of `part` and no finish reason.
    fn streamed(index: usize, part: Value) -> Value {
        let mut choice = json!({"index": index, "logprobs": null, "finish_reason": null});
        if let (Value::Object(choice), Value::Object(part)) = (&mut choice, part) {
            choice.extend(part);
        }
        choice
    }

    /// Streamed choice `index` that adds no text.
    fn empty(&self, index: usize) -> Value {
        Self::streamed(
            index,
            match self.kind {
                Kind::Chat => json!({"delta": {}}),
                Kind::Text => json!({"text": ""}),
            },
        )
    }

    /// Choice `index`'s streamed parts of `delta`, one chunk each; `calls` counts the
    /// choice's tool calls opened. A call's first chunk holds its index, id, type and name,
    /// and the chunks after it pieces of its arguments, as OpenAI streams them.
    fn deltas(&self, index: usize, delta: &Delta, calls: &mut usize) -> Vec<Value> {
        let mut parts = Vec::new();
        match self.kind {
            Kind::Text => {
                if !delta.content.is_empty() {
                    parts.push(json!({"text": delta.content}));
                }
            }
            Kind::Chat => {
                if !delta.reasoning.is_empty() {
                    parts.push(json!({"delta": {"reasoning_content": delta.reasoning}}));
                }
                if !delta.content.is_empty() {
                    parts.push(json!({"delta": {"content": delta.content}}));
                }
                for part in &delta.tool_parts {
                    let value = match part {
                        ToolPart::Start(name) => {
                            *calls += 1;
                            json!({"index": *calls - 1, "id": self.call_id(), "type": "function",
                                   "function": {"name": name, "arguments": ""}})
                        }
                        ToolPart::Arguments(piece) => {
                            json!({"index": calls.saturating_sub(1),
                                   "function": {"arguments": piece}})
                        }
                        ToolPart::End => continue,
                    };
                    parts.push(json!({"delta": {"tool_calls": [value]}}));
                }
            }
        }
        parts
            .into_iter()
            .map(|part| Self::streamed(index, part))
            .collect()
    }

    /// Pre-formatted JSON prefix shared by every streamed chunk of this reply, up to the
    /// choice's `"index":` value.
    fn chunk_prefix(&self) -> String {
        let object = match self.kind {
            Kind::Chat => "chat.completion.chunk",
            Kind::Text => "text_completion",
        };
        let mut buf = Vec::with_capacity(96 + self.id.len() + self.model.len());
        buf.extend_from_slice(br#"{"id":"#);
        let _ = serde_json::to_writer(&mut buf, &self.id);
        buf.extend_from_slice(br#","object":""#);
        buf.extend_from_slice(object.as_bytes());
        buf.extend_from_slice(br#"","created":"#);
        let _ = std::io::Write::write_fmt(&mut buf, format_args!("{}", self.created));
        buf.extend_from_slice(br#","model":"#);
        let _ = serde_json::to_writer(&mut buf, &self.model);
        buf.extend_from_slice(br#","choices":[{"index":"#);
        String::from_utf8(buf).expect("serde_json emits valid UTF-8")
    }

    /// Formats `delta`'s streamed SSE JSON chunks directly into `out`, avoiding
    /// `serde_json::Value` tree construction on every streamed token.
    fn format_delta_chunks(
        &self,
        prefix: &str,
        index: usize,
        delta: &Delta,
        calls: &mut usize,
        out: &mut Vec<String>,
    ) {
        let start_chunk = |extra: usize| {
            let mut buf = Vec::with_capacity(prefix.len() + 64 + extra);
            buf.extend_from_slice(prefix.as_bytes());
            let _ = std::io::Write::write_fmt(
                &mut buf,
                format_args!(r#"{index},"logprobs":null,"finish_reason":null,"#),
            );
            buf
        };
        match self.kind {
            Kind::Text => {
                if !delta.content.is_empty() {
                    let mut buf = start_chunk(delta.content.len() + 16);
                    buf.extend_from_slice(br#""text":"#);
                    let _ = serde_json::to_writer(&mut buf, &delta.content);
                    buf.extend_from_slice(b"}]}");
                    out.push(String::from_utf8(buf).expect("serde_json emits valid UTF-8"));
                }
            }
            Kind::Chat => {
                if !delta.reasoning.is_empty() {
                    let mut buf = start_chunk(delta.reasoning.len() + 36);
                    buf.extend_from_slice(br#""delta":{"reasoning_content":"#);
                    let _ = serde_json::to_writer(&mut buf, &delta.reasoning);
                    buf.extend_from_slice(b"}}]}");
                    out.push(String::from_utf8(buf).expect("serde_json emits valid UTF-8"));
                }
                if !delta.content.is_empty() {
                    let mut buf = start_chunk(delta.content.len() + 28);
                    buf.extend_from_slice(br#""delta":{"content":"#);
                    let _ = serde_json::to_writer(&mut buf, &delta.content);
                    buf.extend_from_slice(b"}}]}");
                    out.push(String::from_utf8(buf).expect("serde_json emits valid UTF-8"));
                }
                for part in &delta.tool_parts {
                    match part {
                        ToolPart::Start(name) => {
                            *calls += 1;
                            let call_index = *calls - 1;
                            let call_id = self.call_id();
                            let mut buf = start_chunk(name.len() + call_id.len() + 96);
                            let _ = std::io::Write::write_fmt(
                                &mut buf,
                                format_args!(
                                    r#""delta":{{"tool_calls":[{{"index":{call_index},"id":"{call_id}","type":"function","function":{{"name":"#
                                ),
                            );
                            let _ = serde_json::to_writer(&mut buf, name);
                            buf.extend_from_slice(br#","arguments":""}}]}}]}"#);
                            out.push(String::from_utf8(buf).expect("serde_json emits valid UTF-8"));
                        }
                        ToolPart::Arguments(piece) => {
                            let call_index = calls.saturating_sub(1);
                            let mut buf = start_chunk(piece.len() + 64);
                            let _ = std::io::Write::write_fmt(
                                &mut buf,
                                format_args!(
                                    r#""delta":{{"tool_calls":[{{"index":{call_index},"function":{{"arguments":"#
                                ),
                            );
                            let _ = serde_json::to_writer(&mut buf, piece);
                            buf.extend_from_slice(b"}}]}}]}");
                            out.push(String::from_utf8(buf).expect("serde_json emits valid UTF-8"));
                        }
                        ToolPart::End => {}
                    }
                }
            }
        }
    }

    fn tool_call(&self, call: &ToolCall) -> Value {
        json!({
            "id": self.call_id(),
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        })
    }

    /// A new tool call id.
    fn call_id(&self) -> String {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        format!("call_{:x}{n:06x}", self.created)
    }

    /// The timings of a reply whose choices are `outputs`, generated by `fanout`.
    fn timings(&self, fanout: &Fanout, outputs: &[Output]) -> Timings {
        let predicted = outputs.iter().map(|o| o.tokens).sum();
        fanout.timings(self.prompt_tokens, self.cached_tokens, predicted)
    }

    /// The usage of a reply whose choices are `outputs`: the prompt once, and the tokens of
    /// every choice.
    fn usage(&self, outputs: &[Output]) -> Value {
        let completion_tokens: usize = outputs.iter().map(|o| o.tokens).sum();
        let reasoning_tokens: usize = outputs.iter().map(Output::reasoning_tokens).sum();
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": self.prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        })
    }
}

/// Sends one request's server-sent chunks, each event's data.
struct Chunks {
    out: mpsc::Sender<String>,
    reply: Reply,
    /// The prompt's ids, sent with the first chunk (`return_token_ids`).
    prompt_ids: Option<Vec<u32>>,
}

impl Chunks {
    /// Sends one chunk per choice, the last with `timings` if given; false once the client
    /// has left.
    async fn send(&mut self, choices: Vec<Value>, mut timings: Option<&Timings>) -> bool {
        let last = choices.len().saturating_sub(1);
        for (k, choice) in choices.into_iter().enumerate() {
            let mut chunk = self.reply.body(true, vec![choice]);
            if let Some(ids) = self.prompt_ids.take() {
                chunk["prompt_token_ids"] = json!(ids);
            }
            if k == last
                && let Some(timings) = timings.take()
            {
                chunk["timings"] = timings.json();
            }
            if !self.raw(chunk.to_string()).await {
                return false;
            }
        }
        true
    }

    async fn raw(&self, data: String) -> bool {
        self.out.send(data).await.is_ok()
    }
}

/// Puts `ids` on the first of `choices`, adding an empty choice `index` if there is none.
fn attach_ids(reply: &Reply, index: usize, choices: &mut Vec<Value>, ids: &[u32]) {
    if ids.is_empty() {
        return;
    }
    if choices.is_empty() {
        choices.push(reply.empty(index));
    }
    choices[0]["token_ids"] = json!(ids);
}

/// Turns the engine's events for one request's choices into server-sent chunks of one
/// choice each, in the order the events arrive, as vLLM and OpenAI interleave them. A chat
/// stream starts with each choice's role chunk. Each choice ends with a chunk that holds
/// its finish reason, and the usage chunk, if asked for, follows the last. Returning drops
/// `fanout`, which cancels the jobs still open: when the client has left or a job failed.
async fn stream(
    vocab: &Vocab,
    mut fanout: Fanout,
    mut outputs: Vec<Output>,
    mut chunks: Chunks,
    include_usage: bool,
) {
    let n = outputs.len();
    let token_ids = chunks.prompt_ids.is_some();
    let prefix = chunks.reply.chunk_prefix();
    // The role chunks wait for the first token, as vLLM's do, so that a client or proxy
    // that times the first chunk measures the prefill. Keep-alive comments hold the
    // connection open meanwhile.
    let mut roles = chunks.reply.kind == Kind::Chat;
    // Each choice's tool calls sent, and its ids already sent as `token_ids`.
    let mut calls = vec![0; n];
    let mut sent = vec![0; n];
    let mut open = n;
    let mut delta = Delta::default();
    let mut formatted = Vec::new();
    while open > 0 {
        let (i, event) = fanout.next().await;
        let output = &mut outputs[i];
        delta.clear();
        let finish = match event {
            Event::Started | Event::Round | Event::Images(_) => continue,
            Event::Cached(cached) => {
                if i == 0 {
                    chunks.reply.cached_tokens = cached;
                }
                continue;
            }
            Event::SmallTokens(buf, len) => {
                output.push_into(vocab, &buf[..usize::from(len)], &mut delta);
                output.done.then_some(Finish::Stop)
            }
            Event::Tokens(ids) => {
                output.push_into(vocab, &ids, &mut delta);
                output.done.then_some(Finish::Stop)
            }
            Event::Done(finish) => Some(finish),
            Event::Failed(message) => {
                let error = json!({"error": {"message": message, "type": "server_error", "param": null, "code": null}});
                let _ = chunks.raw(error.to_string()).await;
                return;
            }
        };
        if std::mem::take(&mut roles) {
            let roles = (0..n)
                .map(|i| Reply::streamed(i, json!({"delta": {"role": "assistant", "content": ""}})))
                .collect();
            if !chunks.send(roles, None).await {
                return;
            }
        }
        // Fast path: direct JSON serialization for mid-stream deltas without `token_ids`
        // or pending `prompt_ids`.
        if finish.is_none() && !token_ids && chunks.prompt_ids.is_none() {
            formatted.clear();
            chunks
                .reply
                .format_delta_chunks(&prefix, i, &delta, &mut calls[i], &mut formatted);
            for frame in formatted.drain(..) {
                if !chunks.raw(frame).await {
                    return;
                }
            }
            continue;
        }
        let mut choices = chunks.reply.deltas(i, &delta, &mut calls[i]);
        if let Some(finish) = finish {
            // Ends the job if a stop ended the output first.
            fanout.close(i, finish);
            let rest = output.finish(vocab, finish);
            choices.extend(chunks.reply.deltas(i, &rest, &mut calls[i]));
            let mut last = chunks.reply.empty(i);
            last["finish_reason"] = json!(output.finish_reason(finish));
            last["stop_reason"] = output.stop_reason();
            choices.push(last);
            open -= 1;
        }
        if token_ids {
            attach_ids(&chunks.reply, i, &mut choices, &output.ids[sent[i]..]);
            sent[i] = output.ids.len();
        }
        // The last choice's finish chunk carries the timings, unless a usage chunk follows.
        let timings = (open == 0).then(|| chunks.reply.timings(&fanout, &outputs));
        if !chunks
            .send(choices, timings.as_ref().filter(|_| !include_usage))
            .await
        {
            return;
        }
    }
    let timings = chunks.reply.timings(&fanout, &outputs);
    if include_usage {
        let mut chunk = chunks.reply.body(true, Vec::new());
        chunk["usage"] = chunks.reply.usage(&outputs);
        chunk["timings"] = timings.json();
        if !chunks.raw(chunk.to_string()).await {
            return;
        }
    }
    let _ = chunks.raw("[DONE]".to_owned()).await;
}

pub(crate) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Seconds since the Unix epoch, for [`App::started`].
pub fn started() -> u64 {
    now()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::chat::tests::vocab;

    #[test]
    fn client_request_ids_are_checked() {
        let id = |pairs: &[(&'static str, &[u8])]| {
            let mut headers = HeaderMap::new();
            for &(name, value) in pairs {
                headers.insert(name, header::HeaderValue::from_bytes(value).unwrap());
            }
            RequestId::from_client(&headers).map(|id| id.to_string())
        };
        assert_eq!(
            id(&[("x-request-id", b"a-1_B.c")]).as_deref(),
            Some("a-1_B.c")
        );
        assert_eq!(
            id(&[("x-client-request-id", b"c"), ("x-request-id", b"r")]).as_deref(),
            Some("r")
        );
        // An invalid x-request-id falls back to x-client-request-id.
        assert_eq!(
            id(&[("x-request-id", b"a b"), ("x-client-request-id", b"c")]).as_deref(),
            Some("c")
        );
        let long = [b'x'; 129];
        for bad in [&b"a b"[..], b"a\tb", "é".as_bytes(), b"", &long[..]] {
            assert_eq!(id(&[("x-request-id", bad)]), None, "{bad:?}");
        }
        assert!(id(&[("x-request-id", &long[..128])]).is_some());
        let (a, b) = (RequestId::generate(), RequestId::generate());
        assert!(a.starts_with("req-") && a.len() == 20 && a != b, "{a} {b}");
    }

    #[test]
    fn max_tokens_fits_the_context() {
        let max_tokens =
            |context, prompt, requested| max_tokens(context, prompt, "max_tokens", requested);
        assert_eq!(max_tokens(100, 40, None).ok(), Some(60));
        assert_eq!(max_tokens(100, 40, Some(60)).ok(), Some(60));
        assert_eq!(max_tokens(100, 40, Some(61)).ok(), Some(60));
        assert_eq!(max_tokens(100, 40, Some(u64::MAX)).ok(), Some(60));
        let zero = max_tokens(100, 40, Some(0)).unwrap_err();
        assert_eq!(zero.param.as_deref(), Some("max_tokens"));
        assert_eq!(
            zero.message,
            "max_tokens: expected an integer of at least 1, got 0; up to 60 fit after the \
             40-token prompt"
        );
        let full = max_tokens(100, 100, None).unwrap_err();
        assert_eq!(full.code, Some("context_length_exceeded"));
        assert!(
            full.message.contains("has 100 tokens") && full.message.contains("at most 99 tokens"),
            "{}",
            full.message
        );
    }

    /// Feeds `ids` one at a time and returns the content, the tool calls and the output.
    fn run(vocab: &Vocab, mut output: Output, ids: &[u32]) -> (Delta, Output) {
        let mut all = Delta::default();
        for &id in ids {
            append(&mut all, output.push(vocab, &[id]));
            if output.done {
                break;
            }
        }
        append(&mut all, output.finish(vocab, Finish::Stop));
        (all, output)
    }

    fn chat_output(vocab: &Vocab, stops: &[&str], limit: Calls) -> Output {
        Output::new(
            Decoder::Chat(Parser::new(vocab, &Ending::Generate)),
            Stops::new(stops.iter().map(|&s| s.to_owned()).collect()),
            Vec::new(),
            limit,
        )
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn stops_cut_content_across_tokens() {
        let vocab = vocab();
        let s = vocab.special;
        let mut ids = vec![s.content_thinking];
        ids.extend(vocab.encode("Think. STOP here is reasoning."));
        ids.extend([s.end_message, s.message_model, s.content_text]);
        let text = vocab.encode("Café 🦀🦀 crab. More text.");
        ids.extend(&text);
        // The engine sends the ids before the turn's end token.
        ids.push(s.end_message);

        // The emoji spans several byte-level tokens; the stop starts inside the text.
        let (all, output) = run(
            &vocab,
            chat_output(&vocab, &["🦀 crab", "STOP"], Calls::Free),
            &ids,
        );
        assert_eq!(all.reasoning, "Think. STOP here is reasoning.");
        assert_eq!(all.content, "Café 🦀");
        assert_eq!(output.stop_reason(), json!("🦀 crab"));
        assert_eq!(output.finish_reason(Finish::Stop), "stop");
        // The ids end at the token that completed the stop, without an end token.
        assert!(output.ids.len() < ids.len());
        assert_eq!(output.tokens, output.ids.len());
        assert!(vocab.decode(&output.ids).contains("🦀 crab"));

        // Without a match, the held text comes back at the end with the end token.
        let (all, output) = run(&vocab, chat_output(&vocab, &["text!"], Calls::Free), &ids);
        assert_eq!(all.content, "Café 🦀🦀 crab. More text.");
        assert_eq!(output.stop_reason(), Value::Null);
        assert_eq!(output.ids.len(), ids.len() + 1);
        assert_eq!(output.ids.last(), Some(&s.content_model_end_sampling));

        // Raw text: special tokens are text, and stops can match them.
        let raw = Output::new(
            Decoder::Text(Detokenizer::default()),
            Stops::new(vec!["<|end_message|>".to_owned()]),
            Vec::new(),
            Calls::Free,
        );
        let (all, output) = run(&vocab, raw, &ids);
        assert_eq!(
            all.content,
            "<|content_thinking|>Think. STOP here is reasoning."
        );
        assert_eq!(output.stop_reason(), json!("<|end_message|>"));
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn stop_tokens_end_the_output_without_their_text() {
        let vocab = vocab();
        let s = vocab.special;
        let mut ids = vec![s.content_thinking];
        ids.extend(vocab.encode("Plan."));
        ids.extend([s.end_message, s.message_model, s.content_text]);
        let text = vocab.encode("One, two, three.");
        let comma = vocab.encode(",")[0];
        assert!(text.contains(&comma), "a comma is its own token here");
        ids.extend(&text);
        ids.push(s.end_message);
        let with_stops = |strings: &[&str], stop_ids: Vec<u32>| {
            let mut output = chat_output(&vocab, strings, Calls::Free);
            output.stop_ids = stop_ids;
            output
        };

        // The first stop token ends the output; its id stays, its text goes.
        let (all, output) = run(&vocab, with_stops(&[], vec![comma]), &ids);
        assert_eq!(all.reasoning, "Plan.");
        assert_eq!(all.content, "One");
        assert_eq!(output.stop_reason(), json!(comma));
        assert_eq!(output.finish_reason(Finish::Stop), "stop");
        assert_eq!(output.ids.last(), Some(&comma));
        assert_eq!(output.tokens, output.ids.len());

        // Text held back for a stop string comes out when a stop token ends the output.
        let (all, output) = run(&vocab, with_stops(&["One, t"], vec![comma]), &ids);
        assert_eq!(all.content, "One");
        assert_eq!(output.stop_reason(), json!(comma));

        // A stop string that ends first wins.
        let (all, output) = run(&vocab, with_stops(&["ne"], vec![comma]), &ids);
        assert_eq!(all.content, "O");
        assert_eq!(output.stop_reason(), json!("ne"));

        // A special token can be a stop token, in chat and raw text.
        let (all, output) = run(&vocab, with_stops(&[], vec![s.end_message]), &ids);
        assert_eq!(
            (all.reasoning.as_str(), all.content.as_str()),
            ("Plan.", "")
        );
        assert_eq!(output.stop_reason(), json!(s.end_message));
        let raw = Output::new(
            Decoder::Text(Detokenizer::default()),
            Stops::default(),
            vec![s.end_message],
            Calls::Free,
        );
        let (all, output) = run(&vocab, raw, &ids);
        assert_eq!(all.content, "<|content_thinking|>Plan.");
        assert_eq!(output.ids.len(), vocab.encode("Plan.").len() + 2);
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn one_tool_call_and_forced_calls() {
        let vocab = vocab();
        let s = vocab.special;
        let call = |name: &str| {
            let mut ids = vec![s.message_model];
            ids.extend(vocab.encode(name));
            ids.push(s.content_invoke_tool_json);
            ids.extend(vocab.encode(&format!(r#"{{"name":"{name}","args":{{"x":1}}}}"#)));
            ids.push(s.end_message);
            ids
        };
        let mut ids = call("a");
        ids.extend(call("b"));
        let (all, output) = run(&vocab, chat_output(&vocab, &[], Calls::Free), &ids);
        assert_eq!(all.tool_calls.len(), 2);
        assert_eq!(output.finish_reason(Finish::Stop), "tool_calls");
        let (all, output) = run(&vocab, chat_output(&vocab, &[], Calls::One), &ids);
        assert_eq!(all.tool_calls.len(), 1);
        assert_eq!(all.tool_calls[0].name, "a");
        assert!(output.done);

        // A prompt that opened the call: generation starts at the payload. Text or
        // thinking after the forced calls ends the output.
        let mut ids = vocab.encode(r#"{"name":"get_weather","args":{"city":"Paris"}}"#);
        ids.push(s.end_message);
        ids.extend(call("b"));
        ids.extend([s.message_model, s.content_thinking]);
        ids.extend(vocab.encode("Why did I call that?"));
        for ending in [Ending::Call("get_weather"), Ending::AnyCall] {
            let parser = Parser::new(&vocab, &ending);
            let forced = Output::new(
                Decoder::Chat(parser),
                Stops::default(),
                Vec::new(),
                Calls::Only,
            );
            let (all, output) = run(&vocab, forced, &ids);
            assert_eq!(all.tool_calls.len(), 2);
            assert_eq!(all.tool_calls[0].name, "get_weather");
            assert_eq!(all.tool_calls[0].arguments, r#"{"city":"Paris"}"#);
            assert_eq!((all.content.as_str(), all.reasoning.as_str()), ("", ""));
            assert_eq!(output.finish_reason(Finish::Stop), "tool_calls");
        }

        // A continued message: generation starts inside its text block.
        let continued = Output::new(
            Decoder::Chat(Parser::new(&vocab, &Ending::Continue)),
            Stops::default(),
            Vec::new(),
            Calls::Free,
        );
        let mut ids = vocab.encode("    return 1");
        ids.extend([s.end_message, s.content_model_end_sampling]);
        let (all, _) = run(&vocab, continued, &ids);
        assert_eq!(all.content, "    return 1");
    }

    /// A fanout over one channel per entry of `events`, each holding its events, for a
    /// prompt of `prompt` tokens. The senders stay open, as the engine's do while a job
    /// runs.
    fn fanout(
        metrics: &Arc<Metrics>,
        prompt: usize,
        events: Vec<Vec<Event>>,
    ) -> (Fanout, Vec<mpsc::UnboundedSender<Event>>) {
        let mut senders = Vec::new();
        let mut results = Vec::new();
        for (i, list) in events.into_iter().enumerate() {
            let (sender, receiver) = mpsc::unbounded_channel();
            for event in list {
                sender.send(event).unwrap();
            }
            senders.push(sender);
            results.push(Results {
                receiver,
                tracker: metrics.track(if i == 0 { prompt } else { 0 }),
                log: None,
            });
        }
        (Fanout::new(results), senders)
    }

    fn reply(kind: Kind) -> Reply {
        Reply {
            id: "cmpl-1".to_owned(),
            created: 1,
            model: "m".to_owned(),
            prompt_tokens: 3,
            cached_tokens: 0,
            kind,
        }
    }

    #[tokio::test]
    async fn fanout_takes_turns_and_cancels_one_choice() {
        let metrics = Arc::new(Metrics::new("m"));
        let events = (0..3)
            .map(|_| {
                vec![
                    Event::Started,
                    Event::Tokens(vec![1]),
                    Event::Tokens(vec![2, 3]),
                ]
            })
            .collect();
        let (mut fanout, senders) = fanout(&metrics, 10, events);
        let mut order = Vec::new();
        for _ in 0..9 {
            let (i, event) = fanout.next().await;
            order.push((i, matches!(event, Event::Tokens(_))));
        }
        // Each choice gets a turn in index order, whatever the others hold.
        let want: Vec<(usize, bool)> = [false, true, true]
            .iter()
            .flat_map(|&tokens| (0..3).map(move |i| (i, tokens)))
            .collect();
        assert_eq!(order, want);
        // Closing a choice cancels its job alone.
        fanout.close(1, Finish::Stop);
        assert!(senders[1].is_closed());
        assert!(!senders[0].is_closed() && !senders[2].is_closed());
        senders[2].send(Event::Done(Finish::Length)).unwrap();
        assert!(matches!(
            fanout.next().await,
            (2, Event::Done(Finish::Length))
        ));
        // The prompt counts once; every choice's tokens count.
        let text = metrics.render();
        assert!(
            text.contains("vllm:prompt_tokens_total{model_name=\"m\"} 10\n"),
            "{text}"
        );
        assert!(
            text.contains("vllm:generation_tokens_total{model_name=\"m\"} 9\n"),
            "{text}"
        );
        assert!(
            text.contains("vllm:prefix_cache_queries_total{model_name=\"m\"} 10\n"),
            "{text}"
        );
        // Dropping the fanout cancels the open choices.
        drop(fanout);
        assert!(senders.iter().all(mpsc::UnboundedSender::is_closed));
    }

    #[test]
    fn usage_counts_the_prompt_once() {
        let mut outputs = vec![
            Output::new(
                Decoder::Text(Detokenizer::default()),
                Stops::default(),
                Vec::new(),
                Calls::Free,
            );
            2
        ];
        outputs[0].tokens = 5;
        outputs[1].tokens = 7;
        let usage = reply(Kind::Text).usage(&outputs);
        assert_eq!(usage["prompt_tokens"], 3);
        assert_eq!(usage["completion_tokens"], 12);
        assert_eq!(usage["total_tokens"], 15);
    }

    /// Each choice's engine events for the text of `texts`, the first two ended by the
    /// model and the third by its token limit.
    fn choice_events(vocab: &Vocab, texts: &[&str]) -> Vec<Vec<Event>> {
        texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let mut events = vec![Event::Started];
                // A token at a time, as batched steps deliver them.
                events.extend(
                    vocab
                        .encode(text)
                        .into_iter()
                        .map(|id| Event::Tokens(vec![id])),
                );
                events.push(Event::Done(if i == 2 {
                    Finish::Length
                } else {
                    Finish::Stop
                }));
                events
            })
            .collect()
    }

    const TEXTS: [&str; 3] = ["Hello world. Bye.", "Stop here now.", "abc"];

    #[tokio::test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn whole_reply_merges_choices() {
        let vocab = vocab();
        let metrics = Arc::new(Metrics::new("m"));
        let (fanout, senders) = fanout(&metrics, 3, choice_events(&vocab, &TEXTS));
        let output = Output::new(
            Decoder::Text(Detokenizer::default()),
            Stops::new(vec!["here".to_owned()]),
            Vec::new(),
            Calls::Free,
        );
        let outputs = vec![output; 3];
        let body = whole(
            &vocab,
            fanout,
            outputs,
            reply(Kind::Text),
            Some(vec![1, 2, 3]),
        )
        .await
        .unwrap()
        .0;
        let choices = body["choices"].as_array().unwrap();
        let texts: Vec<&str> = choices
            .iter()
            .map(|c| c["text"].as_str().unwrap())
            .collect();
        assert_eq!(texts, ["Hello world. Bye.", "Stop ", "abc"]);
        for (i, choice) in choices.iter().enumerate() {
            assert_eq!(choice["index"], i);
        }
        let reasons: Vec<&Value> = choices.iter().map(|c| &c["finish_reason"]).collect();
        assert_eq!(reasons, [&json!("stop"), &json!("stop"), &json!("length")]);
        let stops: Vec<&Value> = choices.iter().map(|c| &c["stop_reason"]).collect();
        assert_eq!(stops, [&Value::Null, &json!("here"), &Value::Null]);
        // The stop string cancelled its choice's job.
        assert!(senders[1].is_closed());
        // The first choice's ids end with the end token, which counts as no token.
        let ids: Vec<usize> = choices
            .iter()
            .map(|c| c["token_ids"].as_array().unwrap().len())
            .collect();
        assert_eq!(body["usage"]["prompt_tokens"], 3);
        assert_eq!(
            body["usage"]["completion_tokens"],
            ids[0] - 1 + ids[1] + ids[2]
        );
        assert_eq!(body["prompt_token_ids"], json!([1, 2, 3]));
    }

    #[tokio::test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn stream_interleaves_choices() {
        let vocab = vocab();
        let metrics = Arc::new(Metrics::new("m"));
        let (fanout, _senders) = fanout(&metrics, 3, choice_events(&vocab, &TEXTS));
        let output = Output::new(
            Decoder::Chat(Parser::new(&vocab, &Ending::Continue)),
            Stops::new(vec!["here".to_owned()]),
            Vec::new(),
            Calls::Free,
        );
        let (out, mut rx) = mpsc::channel(1024);
        let chunks = Chunks {
            out,
            reply: reply(Kind::Chat),
            prompt_ids: None,
        };
        stream(&vocab, fanout, vec![output; 3], chunks, true).await;
        let mut frames = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            frames.push(frame);
        }
        assert_eq!(frames.pop().as_deref(), Some("[DONE]"));
        let chunks: Vec<Value> = frames
            .iter()
            .map(|f| serde_json::from_str(f).unwrap())
            .collect();
        let (usage, chunks) = chunks.split_last().unwrap();
        assert_eq!(usage["choices"], json!([]));
        assert_eq!(usage["usage"]["prompt_tokens"], 3);
        // One role chunk per choice first, then one choice per chunk.
        for (i, chunk) in chunks[..3].iter().enumerate() {
            assert_eq!(chunk["choices"][0]["index"], i);
            assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant");
        }
        let mut content = vec![String::new(); 3];
        let mut finished = vec![None; 3];
        let mut first_index = Vec::new();
        for chunk in &chunks[3..] {
            assert_eq!(chunk["object"], "chat.completion.chunk");
            let [choice] = chunk["choices"].as_array().unwrap().as_slice() else {
                panic!("a chunk holds one choice: {chunk}");
            };
            let i = usize::try_from(choice["index"].as_u64().unwrap()).unwrap();
            assert!(
                finished[i].is_none(),
                "choice {i} sent a chunk after its end"
            );
            first_index.push(i);
            if let Some(text) = choice["delta"]["content"].as_str() {
                content[i] += text;
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                finished[i] = Some((reason.to_owned(), choice["stop_reason"].clone()));
            }
        }
        assert_eq!(content, ["Hello world. Bye.", "Stop ", "abc"]);
        assert_eq!(
            finished,
            [
                Some(("stop".to_owned(), Value::Null)),
                Some(("stop".to_owned(), json!("here"))),
                Some(("length".to_owned(), Value::Null)),
            ]
        );
        // The choices interleave rather than run one after another.
        assert_eq!(first_index[..3], [0, 1, 2]);
        // Every token of the first and last choice, and the second's up to the one that
        // completed the stop string.
        let ids: Vec<Vec<u32>> = TEXTS.iter().map(|t| vocab.encode(t)).collect();
        let cut = (1..=ids[1].len())
            .find(|&k| vocab.decode(&ids[1][..k]).contains("here"))
            .unwrap();
        assert_eq!(
            usage["usage"]["completion_tokens"],
            ids[0].len() + cut + ids[2].len()
        );
    }

    #[test]
    fn streamed_choice_shapes() {
        let reply = Reply {
            id: "chatcmpl-1".to_owned(),
            created: 1,
            model: "m".to_owned(),
            prompt_tokens: 3,
            cached_tokens: 0,
            kind: Kind::Chat,
        };
        let delta = Delta {
            content: "hi".to_owned(),
            reasoning: "hm".to_owned(),
            ..Delta::default()
        };
        let mut calls = 0;
        let choices = reply.deltas(0, &delta, &mut calls);
        assert_eq!(
            choices,
            [
                json!({"index": 0, "logprobs": null, "finish_reason": null,
                       "delta": {"reasoning_content": "hm"}}),
                json!({"index": 0, "logprobs": null, "finish_reason": null,
                       "delta": {"content": "hi"}}),
            ]
        );
        // A second call: its first chunk names it, and its arguments follow.
        let delta = Delta {
            tool_parts: vec![
                ToolPart::Start("f".to_owned()),
                ToolPart::Arguments("{\"a\":".to_owned()),
                ToolPart::End,
            ],
            ..Delta::default()
        };
        calls = 1;
        let mut choices = reply.deltas(0, &delta, &mut calls);
        assert_eq!(calls, 2);
        let id = choices[0]["delta"]["tool_calls"][0]
            .as_object_mut()
            .unwrap()
            .remove("id")
            .unwrap();
        assert!(id.as_str().unwrap().starts_with("call_"), "{id}");
        assert_eq!(
            choices,
            [
                json!({"index": 0, "logprobs": null, "finish_reason": null,
                       "delta": {"tool_calls": [{"index": 1, "type": "function",
                                 "function": {"name": "f", "arguments": ""}}]}}),
                json!({"index": 0, "logprobs": null, "finish_reason": null,
                       "delta": {"tool_calls": [{"index": 1,
                                 "function": {"arguments": "{\"a\":"}}]}}),
            ]
        );
        let text = Reply {
            kind: Kind::Text,
            ..reply
        };
        let mut choices = Vec::new();
        attach_ids(&text, 0, &mut choices, &[7, 8]);
        assert_eq!(
            choices,
            [
                json!({"index": 0, "logprobs": null, "finish_reason": null, "text": "",
                    "token_ids": [7, 8]})
            ]
        );
        let body = text.body(true, vec![choices[0].clone()]);
        assert_eq!(body["object"], "text_completion");
    }

    #[test]
    fn fast_sse_chunk_formatting_matches_value_serializer() {
        let chat = Reply {
            id: "chatcmpl-1".to_owned(),
            created: 1_700_000_000,
            model: "inkling-\"2\"-mini".to_owned(),
            prompt_tokens: 10,
            cached_tokens: 0,
            kind: Kind::Chat,
        };
        let prefix = chat.chunk_prefix();
        let deltas = [
            Delta {
                reasoning: "Line 1\n\"quoted\" \\ 🦀".to_owned(),
                content: "Hello 世界!".to_owned(),
                ..Delta::default()
            },
            Delta {
                tool_parts: vec![
                    ToolPart::Start("get_\"weather\"".to_owned()),
                    ToolPart::Arguments("{\"city\":\"Paris\n\"}".to_owned()),
                    ToolPart::End,
                ],
                ..Delta::default()
            },
        ];
        let mut fast_calls = 0usize;
        let mut slow_calls = 0usize;
        let mut fast_frames = Vec::new();
        for d in &deltas {
            fast_frames.clear();
            chat.format_delta_chunks(&prefix, 2, d, &mut fast_calls, &mut fast_frames);
            let slow_choices = chat.deltas(2, d, &mut slow_calls);
            assert_eq!(fast_calls, slow_calls);
            assert_eq!(fast_frames.len(), slow_choices.len());
            for (raw, choice) in fast_frames.iter().zip(slow_choices) {
                let mut got: Value = serde_json::from_str(raw).unwrap();
                let mut want = chat.body(true, vec![choice]);
                // Normalize generated unique tool call ids before equality check.
                if let Some(id) = got.pointer_mut("/choices/0/delta/tool_calls/0/id") {
                    assert!(id.as_str().unwrap().starts_with("call_"));
                    *id = json!("call_norm");
                }
                if let Some(id) = want.pointer_mut("/choices/0/delta/tool_calls/0/id") {
                    *id = json!("call_norm");
                }
                assert_eq!(got, want, "raw={raw}");
            }
        }

        // Also check Kind::Text parity.
        let text_reply = Reply {
            kind: Kind::Text,
            ..chat
        };
        let text_prefix = text_reply.chunk_prefix();
        let mut out = Vec::new();
        let d = Delta {
            content: "raw\ttext\"ok".to_owned(),
            ..Delta::default()
        };
        text_reply.format_delta_chunks(&text_prefix, 0, &d, &mut 0, &mut out);
        let want = text_reply.body(true, text_reply.deltas(0, &d, &mut 0));
        assert_eq!(serde_json::from_str::<Value>(&out[0]).unwrap(), want);

        // Microbenchmark: 10,000 streamed chat content chunks.
        let chat = Reply {
            kind: Kind::Chat,
            ..text_reply
        };
        let prefix = chat.chunk_prefix();
        let delta = Delta {
            content: " quick".to_owned(),
            ..Delta::default()
        };
        let n = 10_000usize;
        let t0 = std::time::Instant::now();
        let mut slow_bytes = 0usize;
        let mut calls = 0usize;
        for _ in 0..n {
            for choice in chat.deltas(0, &delta, &mut calls) {
                slow_bytes += chat.body(true, vec![choice]).to_string().len();
            }
        }
        let slow_ns = t0.elapsed().as_secs_f64() * 1e9 / n as f64;

        let t1 = std::time::Instant::now();
        let mut fast_bytes = 0usize;
        let mut buf = Vec::with_capacity(2);
        for _ in 0..n {
            buf.clear();
            chat.format_delta_chunks(&prefix, 0, &delta, &mut calls, &mut buf);
            for s in &buf {
                fast_bytes += s.len();
            }
        }
        let fast_ns = t1.elapsed().as_secs_f64() * 1e9 / n as f64;

        assert_eq!(slow_bytes, fast_bytes);
        eprintln!(
            "openai SSE chunk ({n} toks): value={slow_ns:.1} ns/tok, direct={fast_ns:.1} ns/tok ({:.1}x)",
            slow_ns / fast_ns.max(1e-9)
        );
    }

    #[tokio::test]
    async fn small_token_events_stream_identically() {
        let vocab = crate::serve::chat::tests::synthetic_vocab();
        let metrics = Arc::new(Metrics::new("m"));
        let encoded = vocab.encode("Hello world!");
        let mut events = vec![Event::Started];
        for chunk in encoded.chunks(3) {
            events.push(Event::from_tokens(chunk));
        }
        events.push(Event::Done(Finish::Stop));
        let (fanout, _senders) = fanout(&metrics, 5, vec![events]);
        let output = Output::new(
            Decoder::Text(Detokenizer::default()),
            Stops::default(),
            Vec::new(),
            Calls::Free,
        );
        let (out, mut rx) = mpsc::channel(128);
        let chunks = Chunks {
            out,
            reply: reply(Kind::Text),
            prompt_ids: None,
        };
        stream(&vocab, fanout, vec![output], chunks, false).await;
        let mut text = String::new();
        while let Ok(frame) = rx.try_recv() {
            if frame == "[DONE]" {
                continue;
            }
            let val: Value = serde_json::from_str(&frame).unwrap();
            if let Some(piece) = val["choices"][0]["text"].as_str() {
                text.push_str(piece);
            }
        }
        assert_eq!(text, "Hello world!");
    }
}
