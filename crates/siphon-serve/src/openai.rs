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
//!   ([`crate::anthropic`]), which share the chat route's prompt rendering and output.
//! - `GET /metrics`, which needs no API key: Prometheus metrics ([`crate::metrics`]).
//! - `GET /v1/models` and `GET /v1/models/{id}`, with `max_model_len`.
//! - The tokenizer routes of [`crate::tokenize`].
//! - `GET /health` and `GET /version`, which need no API key.
//!
//! Both completion routes sample with `temperature`, `top_k`, `top_p`, `min_p` and `seed`,
//! with the checkpoint's `generation_config.json` for the fields a request leaves out
//! ([`crate::request::sampling`]); `temperature` 0 decodes greedily. The penalties are
//! accepted and ignored. [`crate::request`] lists the fields the server refuses, such as
//! `logprobs`. Stop strings match the reply's content,
//! not its reasoning or tool calls. Both routes take vLLM's `stop_token_ids`: the output
//! ends at the first of them, whose id the output keeps and whose text it leaves out, as
//! vLLM does. A choice reports the stop string or stop token that matched as vLLM's
//! `stop_reason`.
//!
//! Both routes take `n`: the request's `n` jobs share its prompt (see [`crate::engine`]),
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
    Json, Router,
    extract::{DefaultBodyLimit, Path, Request, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode, header},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures_util::StreamExt as _;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    anthropic,
    chat::{self, Ending, InvalidRequest, Vocab},
    engine::{Engine, Event, Finish, Job, Refused, Sampling},
    metrics::{Metrics, Reason, Tracker},
    parse::{Delta, Detokenizer, Parser, ToolCall},
    request::{self, SamplingDefaults, SamplingFields, ToolChoice},
    stop::Stops,
    tokenize::{self, DecodeRequest, EncodeRequest},
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
    /// The sampling of fields a request leaves out.
    pub sampling: SamplingDefaults,
    /// `--greedy`: every request decodes greedily, whatever its sampling fields ask.
    pub greedy: bool,
}

/// The sampling of each of a request's `n` choices. Under `--greedy` every choice would be
/// the same, so `n` above 1 is refused, as it is for temperature 0.
fn choice_samplings(
    app: &App,
    fields: &SamplingFields,
    n: usize,
) -> Result<Vec<Sampling>, ApiError> {
    if app.greedy && n > 1 {
        return Err(ApiError::bad_request(format!(
            "n must be 1 when the server decodes greedily (--greedy), got {n}"
        )));
    }
    Ok(request::samplings(fields, &app.sampling, n)?)
}

/// Request body bytes allowed per position of the context. A prompt of English text
/// takes about 5 bytes per token; JSON escapes and token-id arrays take more.
const BODY_BYTES_PER_POSITION: usize = 32;

pub fn router(app: Arc<App>) -> Router {
    // axum's default limit of 2 MB holds a prompt of only about 400K tokens.
    let body_limit = (app.engine.max_context() * BODY_BYTES_PER_POSITION).max(2 << 20);
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
        .layer(DefaultBodyLimit::max(body_limit))
        .route_layer(middleware::from_fn_with_state(Arc::clone(&app), authorize))
        .route("/health", get(|| async { "ok" }))
        .route("/metrics", get(metrics_route))
        .route(
            "/version",
            get(|| async { Json(json!({"version": env!("CARGO_PKG_VERSION")})) }),
        )
        .with_state(app)
}

/// An OpenAI-style error response.
#[derive(Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
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
        Self::bad_request(e.0)
    }
}

impl From<JsonRejection> for ApiError {
    fn from(e: JsonRejection) -> Self {
        Self::bad_request(e.body_text())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (kind, code) = match self.status {
            StatusCode::BAD_REQUEST => ("invalid_request_error", Value::Null),
            StatusCode::NOT_FOUND => ("invalid_request_error", json!("model_not_found")),
            StatusCode::UNAUTHORIZED => ("authentication_error", Value::Null),
            _ => ("server_error", Value::Null),
        };
        let body =
            json!({"error": {"message": self.message, "type": kind, "param": null, "code": code}});
        (self.status, Json(body)).into_response()
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
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        app.metrics.render(),
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
    request: Result<Json<EncodeRequest>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(request) = request?;
    let ids = blocking(&app, |app| Ok(tokenize::encode(&app.vocab, request)?)).await?;
    Ok(Json(tokenize::tokenized(&ids, app.engine.max_context())))
}

async fn token_encode(
    State(app): State<Arc<App>>,
    request: Result<Json<EncodeRequest>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(request) = request?;
    let ids = blocking(&app, |app| Ok(tokenize::encode(&app.vocab, request)?)).await?;
    Ok(Json(tokenize::encoded(&ids)))
}

async fn detokenize_route(
    State(app): State<Arc<App>>,
    request: Result<Json<DecodeRequest>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(request) = request?;
    let text = blocking(&app, move |app| Ok(tokenize::decode(&app.vocab, &request)?)).await?;
    Ok(Json(json!({"prompt": text})))
}

async fn token_decode(
    State(app): State<Arc<App>>,
    request: Result<Json<DecodeRequest>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(request) = request?;
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
}

fn prepare_chat(app: &App, request: ChatRequest) -> Result<Prepared, ApiError> {
    request::refuse(&request.other)?;
    let n = request::choices(request.n)?;
    request::response_format(request.response_format.as_ref())?;
    let stops = request::stops(request.stop.as_ref())?;
    let stop_ids = tokenize::token_ids(&app.vocab, &request.stop_token_ids.unwrap_or_default())?;
    let ending = request::ending(
        request.continue_final_message,
        request.add_generation_prompt,
    )?;
    if matches!(ending, Ending::Bare) {
        return Err(ApiError::bad_request(
            "add_generation_prompt false needs continue_final_message",
        ));
    }
    let mut tools = request.tools.unwrap_or_default();
    let choice = ToolChoice::parse(request.tool_choice.as_ref())?;
    if choice == ToolChoice::None {
        tools.clear();
    }
    let parallel = request.parallel_tool_calls != Some(false);
    let (ending, calls) = tool_ending(&choice, &tools, parallel, ending)?;
    let effort = chat::request_effort(
        request.reasoning_effort.as_ref(),
        request.chat_template_kwargs.as_ref(),
    )?;
    let prompt = app
        .vocab
        .render_to(&request.messages, &tools, effort, &ending)?;
    let max_tokens = max_tokens(
        app.engine.max_context(),
        prompt.len(),
        request.max_completion_tokens.or(request.max_tokens),
    )?;
    Ok(Prepared {
        prompt,
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
        (Some(call), Ending::Generate) => call,
        (Some(_), _) => {
            return Err(ApiError::bad_request(
                "continue_final_message cannot be used with a forced tool call",
            ));
        }
    };
    Ok((ending, calls))
}

/// The tokens a request may generate: `requested`, or all that fit after a prompt of
/// `prompt_len` tokens in a context of `max_context`.
pub(crate) fn max_tokens(
    max_context: usize,
    prompt_len: usize,
    requested: Option<u64>,
) -> Result<usize, ApiError> {
    let room = max_context.saturating_sub(prompt_len);
    if room == 0 {
        return Err(ApiError::bad_request(format!(
            "the prompt has {prompt_len} tokens, and this server holds {max_context} in all"
        )));
    }
    match requested {
        None => Ok(room),
        Some(0) => Err(ApiError::bad_request("max_tokens must be at least 1")),
        Some(n) => {
            let n = usize::try_from(n).unwrap_or(usize::MAX);
            if n > room {
                return Err(ApiError::bad_request(format!(
                    "the prompt has {prompt_len} tokens and max_tokens is {n}, more than the \
                     {max_context} this server holds"
                )));
            }
            Ok(n)
        }
    }
}

/// Queues a job and returns its events.
pub(crate) fn submit(
    app: &App,
    prompt: &[u32],
    max_tokens: usize,
    ignore_eos: bool,
    sampling: Sampling,
) -> Result<Results, ApiError> {
    let mut results = submit_choices(app, prompt, max_tokens, ignore_eos, &[sampling])?;
    results
        .pop()
        .ok_or_else(|| ApiError::internal("the engine took no job"))
}

/// Queues one job per entry of `samplings`, all of the same prompt, and returns their
/// events in that order. The jobs fit the queue together or not at all.
pub(crate) fn submit_choices(
    app: &App,
    prompt: &[u32],
    max_tokens: usize,
    ignore_eos: bool,
    samplings: &[Sampling],
) -> Result<Vec<Results>, ApiError> {
    let most = app.engine.max_jobs();
    if samplings.len() > most {
        return Err(ApiError::bad_request(format!(
            "n is {}, more than the {most} requests this server runs and queues at once",
            samplings.len()
        )));
    }
    let prompt_len = prompt.len();
    let (jobs, receivers): (Vec<Job>, Vec<_>) = samplings
        .iter()
        .map(|&sampling| {
            let (events, receiver) = mpsc::unbounded_channel();
            let job = Job::new(prompt.to_vec(), max_tokens, ignore_eos, sampling, events);
            (job, receiver)
        })
        .unzip();
    app.engine.submit(jobs).map_err(|e| match e {
        Refused::Busy => ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "the request queue is full, retry later",
        ),
        Refused::Stopped => ApiError::internal("the engine has stopped"),
    })?;
    // The prompt counts once in the metrics, with the first choice.
    Ok(receivers
        .into_iter()
        .enumerate()
        .map(|(i, receiver)| Results {
            receiver,
            tracker: app.metrics.track(if i == 0 { prompt_len } else { 0 }),
        })
        .collect())
}

/// One job's events, recorded in the metrics as they arrive. Dropping it cancels the job,
/// and counts it as aborted unless it finished.
pub(crate) struct Results {
    pub(crate) receiver: mpsc::UnboundedReceiver<Event>,
    pub(crate) tracker: Tracker,
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
            Event::Tokens(ids) => self.tracker.tokens(ids.len()),
            Event::Done(finish) => self.tracker.finish(reason(*finish)),
            Event::Failed(_) => self.tracker.finish(Reason::Error),
        }
        Poll::Ready(event)
    }

    /// Ends the job with `finish`, which a stop the output matched may have set first.
    pub(crate) fn close(mut self, finish: Finish) {
        self.tracker.finish(reason(finish));
    }
}

/// The jobs of one request's choices, merged. Dropping it cancels the jobs still open.
pub(crate) struct Fanout {
    /// Each choice's events, `None` once the choice has ended.
    choices: Vec<Option<Results>>,
    /// The choice to look at first, which turns, so that no choice waits behind another.
    turn: usize,
}

impl Fanout {
    pub(crate) fn new(results: Vec<Results>) -> Self {
        Self {
            choices: results.into_iter().map(Some).collect(),
            turn: 0,
        }
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
            results.close(finish);
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
    request: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = request?;
    let prepared = blocking(&app, |app| prepare_chat(app, request)).await?;
    respond(app, prepared, Kind::Chat).await
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
    let stops = request::stops(request.stop.as_ref())?;
    let stop_ids = tokenize::token_ids(&app.vocab, &request.stop_token_ids.unwrap_or_default())?;
    let prompt = match request.prompt {
        Value::String(text) => app.vocab.encode(&text),
        Value::Array(items) if items.iter().all(Value::is_u64) => {
            tokenize::token_ids(&app.vocab, &items)?
        }
        Value::Array(items) => match items.as_slice() {
            [Value::String(text)] => app.vocab.encode(text),
            [Value::Array(ids)] => tokenize::token_ids(&app.vocab, ids)?,
            _ => return Err(ApiError::bad_request("only one prompt per request")),
        },
        _ => {
            return Err(ApiError::bad_request(
                "prompt must be a string or token ids",
            ));
        }
    };
    if prompt.is_empty() {
        return Err(ApiError::bad_request("the prompt is empty"));
    }
    let max_tokens = max_tokens(app.engine.max_context(), prompt.len(), request.max_tokens)?;
    Ok(Prepared {
        prompt,
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
    })
}

/// Raw completion of a text prompt, encoded with special tokens as plain text, or of a
/// token-id prompt. The text returned includes any special tokens generated.
async fn completions(
    State(app): State<Arc<App>>,
    request: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = request?;
    let prepared = blocking(&app, |app| prepare_completion(app, request)).await?;
    respond(app, prepared, Kind::Text).await
}

/// Runs a prepared request and answers it, streaming or not.
async fn respond(app: Arc<App>, prepared: Prepared, kind: Kind) -> Result<Response, ApiError> {
    let Prepared {
        prompt,
        max_tokens,
        delivery,
        return_token_ids,
        ignore_eos,
        samplings,
        output,
    } = prepared;
    let reply = Reply::new(&app, prompt.len(), kind);
    let outputs = vec![output; samplings.len()];
    let results = submit_choices(&app, &prompt, max_tokens, ignore_eos, &samplings)?;
    let prompt_ids = return_token_ids.then_some(prompt);
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
    let body = whole(&app.vocab, fanout, outputs, reply, prompt_ids).await?;
    Ok(Json(body).into_response())
}

/// The body of a reply that is not streamed, with one choice per entry of `outputs`.
async fn whole(
    vocab: &Vocab,
    mut fanout: Fanout,
    mut outputs: Vec<Output>,
    mut reply: Reply,
    prompt_ids: Option<Vec<u32>>,
) -> Result<Value, ApiError> {
    let mut messages: Vec<Delta> = outputs.iter().map(|_| Delta::default()).collect();
    let mut finishes: Vec<Option<Finish>> = vec![None; outputs.len()];
    while finishes.iter().any(Option::is_none) {
        let (i, event) = fanout.next().await;
        let output = &mut outputs[i];
        let finish = match event {
            Event::Started => continue,
            Event::Cached(n) => {
                if i == 0 {
                    reply.cached_tokens = n;
                }
                continue;
            }
            Event::Tokens(ids) => {
                append(&mut messages[i], output.push(vocab, &ids));
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
    Ok(body)
}

pub(crate) fn append(message: &mut Delta, delta: Delta) {
    message.content += &delta.content;
    message.reasoning += &delta.reasoning;
    message.tool_calls.extend(delta.tool_calls);
}

/// Turns generated ids into reply text.
#[derive(Clone)]
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
    pub(crate) fn push(&mut self, vocab: &Vocab, ids: &[u32]) -> Delta {
        let mut out = Delta::default();
        for &id in ids {
            if self.done {
                break;
            }
            self.ids.push(id);
            self.tokens += 1;
            if self.stop_ids.contains(&id) {
                // The token ends the output without its text, as an end token would.
                self.end(vocab, &mut out);
                if !self.done {
                    self.stopped_by = Some(id);
                    self.done = true;
                }
                break;
            }
            let delta = match &mut self.decoder {
                Decoder::Chat(parser) => parser.push(vocab, &[id]),
                Decoder::Text(text) => Delta {
                    content: text.push(vocab, id),
                    ..Delta::default()
                },
            };
            self.take(delta, &mut out);
        }
        out
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
        let delta = match &mut self.decoder {
            Decoder::Chat(parser) => parser.finish(vocab),
            Decoder::Text(text) => Delta {
                content: text.flush(vocab),
                ..Delta::default()
            },
        };
        self.take(delta, out);
        if !self.done {
            out.content += &self.stops.flush();
        }
    }

    /// Cuts `delta` at a stop string or where `limit` ends the output, and adds it to `out`.
    fn take(&mut self, mut delta: Delta, out: &mut Delta) {
        delta.content = self.stops.push(&delta.content);
        if self.stops.matched().is_some() {
            self.done = true;
        }
        match self.limit {
            Calls::Free => {}
            Calls::One => {
                if !delta.tool_calls.is_empty() {
                    delta.tool_calls.truncate(1);
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
        append(out, delta);
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

    fn reasoning_tokens(&self) -> usize {
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
                .map(|call| self.tool_call(call, None))
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
    /// choice's tool calls sent.
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
                for call in &delta.tool_calls {
                    let value = self.tool_call(call, Some(*calls));
                    *calls += 1;
                    parts.push(json!({"delta": {"tool_calls": [value]}}));
                }
            }
        }
        parts
            .into_iter()
            .map(|part| Self::streamed(index, part))
            .collect()
    }

    fn tool_call(&self, call: &ToolCall, index: Option<usize>) -> Value {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let mut value = json!({
            "id": format!("call_{:x}{n:06x}", self.created),
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        });
        if let Some(index) = index {
            value["index"] = json!(index);
        }
        value
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
    /// Sends one chunk per choice; false once the client has left.
    async fn send(&mut self, choices: Vec<Value>) -> bool {
        for choice in choices {
            let mut chunk = self.reply.body(true, vec![choice]);
            if let Some(ids) = self.prompt_ids.take() {
                chunk["prompt_token_ids"] = json!(ids);
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
    if chunks.reply.kind == Kind::Chat {
        let roles = (0..n)
            .map(|i| Reply::streamed(i, json!({"delta": {"role": "assistant", "content": ""}})))
            .collect();
        if !chunks.send(roles).await {
            return;
        }
    }
    // Each choice's tool calls sent, and its ids already sent as `token_ids`.
    let mut calls = vec![0; n];
    let mut sent = vec![0; n];
    let mut open = n;
    while open > 0 {
        let (i, event) = fanout.next().await;
        let output = &mut outputs[i];
        let (delta, finish) = match event {
            Event::Started => continue,
            Event::Cached(cached) => {
                if i == 0 {
                    chunks.reply.cached_tokens = cached;
                }
                continue;
            }
            Event::Tokens(ids) => {
                let delta = output.push(vocab, &ids);
                (delta, output.done.then_some(Finish::Stop))
            }
            Event::Done(finish) => (Delta::default(), Some(finish)),
            Event::Failed(message) => {
                let error = json!({"error": {"message": message, "type": "server_error", "param": null, "code": null}});
                let _ = chunks.raw(error.to_string()).await;
                return;
            }
        };
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
        if !chunks.send(choices).await {
            return;
        }
    }
    if include_usage {
        let mut chunk = chunks.reply.body(true, Vec::new());
        chunk["usage"] = chunks.reply.usage(&outputs);
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
    use crate::chat::tests::vocab;

    #[test]
    fn max_tokens_fits_the_context() {
        assert_eq!(max_tokens(100, 40, None).ok(), Some(60));
        assert_eq!(max_tokens(100, 40, Some(60)).ok(), Some(60));
        assert!(max_tokens(100, 40, Some(61)).is_err());
        assert!(max_tokens(100, 40, Some(0)).is_err());
        assert!(max_tokens(100, 100, None).is_err());
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
        .unwrap();
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
            tool_calls: Vec::new(),
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
}
