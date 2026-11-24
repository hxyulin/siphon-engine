//! The OpenAI-compatible HTTP API: `GET /v1/models`, `POST /v1/chat/completions` (with
//! streaming), `POST /v1/completions` (raw text or token ids, without streaming), and
//! `GET /health`.
//!
//! Decoding is greedy, so sampling fields such as `temperature` and `top_p` are accepted
//! and ignored.

use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Request, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode, header},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, KeepAlive, Sse},
    },
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    chat::{self, InvalidRequest, Vocab},
    engine::{Engine, Event, Finish, Job, Refused},
    parse::{Delta, Parser},
};

pub struct App {
    pub vocab: Vocab,
    pub engine: Engine,
    /// The model id that `/v1/models` lists and responses report.
    pub name: String,
    /// The bearer token every `/v1` request must carry, if set.
    pub api_key: Option<String>,
    pub started: u64,
}

/// Request body bytes allowed per position of the context. A prompt of English text
/// takes about 5 bytes per token; JSON escapes and token-id arrays take more.
const BODY_BYTES_PER_POSITION: usize = 32;

pub fn router(app: Arc<App>) -> Router {
    // axum's default limit of 2 MB holds a prompt of only about 400K tokens.
    let body_limit = (app.engine.max_context() * BODY_BYTES_PER_POSITION).max(2 << 20);
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(DefaultBodyLimit::max(body_limit))
        .route_layer(middleware::from_fn_with_state(Arc::clone(&app), authorize))
        .route("/health", get(|| async { "ok" }))
        .with_state(app)
}

/// An OpenAI-style error response.
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }
}

impl From<InvalidRequest> for ApiError {
    fn from(e: InvalidRequest) -> Self {
        Self::bad_request(e.0)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = match self.status {
            StatusCode::BAD_REQUEST => "invalid_request_error",
            StatusCode::UNAUTHORIZED => "authentication_error",
            _ => "server_error",
        };
        let body =
            json!({"error": {"message": self.message, "type": kind, "param": null, "code": null}});
        (self.status, Json(body)).into_response()
    }
}

async fn authorize(
    State(app): State<Arc<App>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    if let Some(key) = &app.api_key {
        let given = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        if !same_bytes(given.as_bytes(), key.as_bytes()) {
            return ApiError::new(StatusCode::UNAUTHORIZED, "invalid API key").into_response();
        }
    }
    next.run(request).await
}

/// Compares in time that depends only on the lengths.
fn same_bytes(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": app.name, "object": "model", "created": app.started, "owned_by": "siphon"}],
    }))
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
    reasoning_effort: Option<Value>,
    chat_template_kwargs: Option<Value>,
    n: Option<u64>,
    stop: Option<Value>,
    /// vLLM's extension: also return the prompt's and the completion's token ids.
    return_token_ids: Option<bool>,
}

#[derive(Deserialize)]
struct StreamOptions {
    include_usage: Option<bool>,
}

/// A validated request, ready for the engine.
struct Prepared {
    prompt: Vec<u32>,
    max_tokens: usize,
    stream: bool,
    include_usage: bool,
    return_token_ids: bool,
}

fn prepare(app: &App, request: ChatRequest) -> Result<Prepared, ApiError> {
    if request.n.is_some_and(|n| n != 1) {
        return Err(ApiError::bad_request("only n = 1 is supported"));
    }
    match &request.stop {
        None | Some(Value::Null) => {}
        Some(Value::Array(a)) if a.is_empty() => {}
        Some(_) => {
            return Err(ApiError::bad_request(
                "stop sequences are not supported yet",
            ));
        }
    }
    match &request.tool_choice {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) if s == "auto" || s == "none" => {}
        Some(_) => {
            return Err(ApiError::bad_request(
                "only the tool_choice values auto and none are supported",
            ));
        }
    }
    let tools = if matches!(&request.tool_choice, Some(Value::String(s)) if s == "none") {
        Vec::new()
    } else {
        request.tools.unwrap_or_default()
    };
    let effort = chat::effort(request.reasoning_effort.as_ref().or_else(|| {
        request
            .chat_template_kwargs
            .as_ref()
            .and_then(|k| k.get("reasoning_effort"))
    }))?;
    let stream = request.stream.unwrap_or(false);
    let return_token_ids = request.return_token_ids.unwrap_or(false);
    if stream && return_token_ids {
        return Err(ApiError::bad_request(
            "return_token_ids is supported only without streaming",
        ));
    }
    let prompt = app.vocab.render(&request.messages, &tools, effort)?;
    let max_tokens = max_tokens(
        app,
        prompt.len(),
        request.max_completion_tokens.or(request.max_tokens),
    )?;
    Ok(Prepared {
        prompt,
        max_tokens,
        stream,
        include_usage: request
            .stream_options
            .and_then(|o| o.include_usage)
            .unwrap_or(false),
        return_token_ids,
    })
}

/// The tokens a request may generate: `requested`, or all that fit after a prompt of
/// `prompt_len` tokens.
fn max_tokens(app: &App, prompt_len: usize, requested: Option<u64>) -> Result<usize, ApiError> {
    let max_context = app.engine.max_context();
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

/// Queues a job and returns the channel its events arrive on.
fn submit(
    app: &App,
    prompt: Vec<u32>,
    max_tokens: usize,
) -> Result<mpsc::UnboundedReceiver<Event>, ApiError> {
    let (events, results) = mpsc::unbounded_channel();
    app.engine
        .submit(Job {
            prompt,
            max_tokens,
            events,
        })
        .map_err(|e| match e {
            Refused::Busy => ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "the request queue is full, retry later",
            ),
            Refused::Stopped => {
                ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "the engine has stopped")
            }
        })?;
    Ok(results)
}

async fn chat_completions(
    State(app): State<Arc<App>>,
    request: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = request.map_err(|e| ApiError::bad_request(e.body_text()))?;
    // Tokenizing a long prompt takes milliseconds, so it leaves the async workers.
    let prepared = {
        let app = Arc::clone(&app);
        tokio::task::spawn_blocking(move || prepare(&app, request))
            .await
            .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))??
    };
    let prompt_tokens = prepared.prompt.len();
    let prompt_ids = prepared.return_token_ids.then(|| prepared.prompt.clone());
    let mut results = submit(&app, prepared.prompt, prepared.max_tokens)?;
    let reply = Reply::new(&app, prompt_tokens);
    if prepared.stream {
        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(stream(app, reply, results, tx, prepared.include_usage));
        let body = Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default());
        return Ok(body.into_response());
    }
    let mut parser = Parser::new(&app.vocab);
    let mut message = Delta::default();
    let mut token_ids = Vec::new();
    let finish = loop {
        match results.recv().await {
            Some(Event::Tokens(ids)) => {
                append(&mut message, parser.push(&app.vocab, &ids));
                token_ids.extend(ids);
            }
            Some(Event::Done(finish)) => break finish,
            Some(Event::Failed(e)) => {
                return Err(ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e));
            }
            None => {
                return Err(ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "the engine dropped the request",
                ));
            }
        }
    };
    append(&mut message, parser.finish(&app.vocab));
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
            .map(|call| reply.tool_call(call, None))
            .collect();
    }
    let usage = reply.usage(token_ids.len(), parser.reasoning_tokens());
    let mut response = json!({
        "id": reply.id,
        "object": "chat.completion",
        "created": reply.created,
        "model": reply.model,
        "choices": [{
            "index": 0,
            "message": body,
            "logprobs": null,
            "finish_reason": finish_reason(finish, !message.tool_calls.is_empty()),
        }],
        "usage": usage,
    });
    if let Some(prompt_ids) = prompt_ids {
        response["prompt_token_ids"] = json!(prompt_ids);
        response["choices"][0]["token_ids"] = json!(token_ids);
    }
    Ok(Json(response).into_response())
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: Value,
    max_tokens: Option<u64>,
    stream: Option<bool>,
    echo: Option<bool>,
    n: Option<u64>,
    stop: Option<Value>,
}

/// Raw completion of a text prompt, encoded with special tokens as plain text, or of a
/// token-id prompt. The text returned includes any special tokens generated.
async fn completions(
    State(app): State<Arc<App>>,
    request: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = request.map_err(|e| ApiError::bad_request(e.body_text()))?;
    if request.n.is_some_and(|n| n != 1) {
        return Err(ApiError::bad_request("only n = 1 is supported"));
    }
    if request.stream == Some(true) || request.echo == Some(true) {
        return Err(ApiError::bad_request(
            "stream and echo are not supported on /v1/completions",
        ));
    }
    if !matches!(&request.stop, None | Some(Value::Null)) {
        return Err(ApiError::bad_request(
            "stop sequences are not supported yet",
        ));
    }
    let prompt = match request.prompt {
        Value::String(text) => app.vocab.encode(&text),
        Value::Array(items) if items.iter().all(Value::is_u64) => items
            .iter()
            .map(|id| {
                id.as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .filter(|&id| (id as usize) < app.vocab.len())
                    .ok_or_else(|| ApiError::bad_request(format!("{id} is not a token id")))
            })
            .collect::<Result<_, _>>()?,
        Value::Array(items) => match items.as_slice() {
            [Value::String(text)] => app.vocab.encode(text),
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
    let max_tokens = max_tokens(&app, prompt.len(), request.max_tokens)?;
    let prompt_tokens = prompt.len();
    let mut results = submit(&app, prompt, max_tokens)?;
    let mut ids = Vec::new();
    let finish = loop {
        match results.recv().await {
            Some(Event::Tokens(new)) => ids.extend(new),
            Some(Event::Done(finish)) => break finish,
            Some(Event::Failed(e)) => {
                return Err(ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, e));
            }
            None => {
                return Err(ApiError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "the engine dropped the request",
                ));
            }
        }
    };
    let reply = Reply::new(&app, prompt_tokens);
    Ok(Json(json!({
        "id": reply.id.replacen("chatcmpl-", "cmpl-", 1),
        "object": "text_completion",
        "created": reply.created,
        "model": reply.model,
        "choices": [{
            "index": 0,
            "text": app.vocab.decode(&ids),
            "logprobs": null,
            "finish_reason": finish_reason(finish, false),
        }],
        "usage": reply.usage(ids.len(), 0),
    }))
    .into_response())
}

fn append(message: &mut Delta, delta: Delta) {
    message.content += &delta.content;
    message.reasoning += &delta.reasoning;
    message.tool_calls.extend(delta.tool_calls);
}

fn finish_reason(finish: Finish, called_tools: bool) -> &'static str {
    match finish {
        Finish::Stop if called_tools => "tool_calls",
        Finish::Stop => "stop",
        Finish::Length => "length",
    }
}

/// The fields every response or chunk of one request shares.
struct Reply {
    id: String,
    created: u64,
    model: String,
    prompt_tokens: usize,
}

impl Reply {
    fn new(app: &App, prompt_tokens: usize) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let created = now();
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        Self {
            id: format!("chatcmpl-{created:x}{n:06x}"),
            created,
            model: app.name.clone(),
            prompt_tokens,
        }
    }

    fn tool_call(&self, call: &crate::parse::ToolCall, index: Option<usize>) -> Value {
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

    fn usage(&self, completion_tokens: usize, reasoning_tokens: usize) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": self.prompt_tokens + completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        })
    }

    fn chunk(&self, delta: &Value, finish_reason: Option<&str>) -> SseEvent {
        SseEvent::default().data(
            json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "logprobs": null, "finish_reason": finish_reason}],
            })
            .to_string(),
        )
    }
}

/// Turns the engine's events for one request into server-sent chunks on `out`, which
/// closes when the client leaves; dropping `results` then cancels the job.
async fn stream(
    app: Arc<App>,
    reply: Reply,
    mut results: mpsc::UnboundedReceiver<Event>,
    out: mpsc::Sender<Result<SseEvent, Infallible>>,
    include_usage: bool,
) {
    let send = |event: SseEvent| out.send(Ok(event));
    if send(reply.chunk(&json!({"role": "assistant", "content": ""}), None))
        .await
        .is_err()
    {
        return;
    }
    let mut parser = Parser::new(&app.vocab);
    let mut tool_calls = 0;
    let mut completion_tokens = 0;
    let finish = loop {
        let delta = match results.recv().await {
            Some(Event::Tokens(ids)) => {
                completion_tokens += ids.len();
                parser.push(&app.vocab, &ids)
            }
            Some(Event::Done(finish)) => break Ok(finish),
            Some(Event::Failed(e)) => break Err(e),
            None => break Err("the engine dropped the request".to_owned()),
        };
        if !send_delta(&send, &reply, delta, &mut tool_calls).await {
            return;
        }
    };
    let finish = match finish {
        Ok(finish) => finish,
        Err(message) => {
            let error = json!({"error": {"message": message, "type": "server_error", "param": null, "code": null}});
            let _ = send(SseEvent::default().data(error.to_string())).await;
            return;
        }
    };
    if !send_delta(&send, &reply, parser.finish(&app.vocab), &mut tool_calls).await {
        return;
    }
    let reason = finish_reason(finish, tool_calls > 0);
    if send(reply.chunk(&json!({}), Some(reason))).await.is_err() {
        return;
    }
    if include_usage {
        let usage = reply.usage(completion_tokens, parser.reasoning_tokens());
        let chunk = json!({
            "id": reply.id,
            "object": "chat.completion.chunk",
            "created": reply.created,
            "model": reply.model,
            "choices": [],
            "usage": usage,
        });
        if send(SseEvent::default().data(chunk.to_string()))
            .await
            .is_err()
        {
            return;
        }
    }
    let _ = send(SseEvent::default().data("[DONE]")).await;
}

/// Sends the chunks of `delta`; false when the client has left.
async fn send_delta<F, Fut, E>(
    send: &F,
    reply: &Reply,
    delta: Delta,
    tool_calls: &mut usize,
) -> bool
where
    F: Fn(SseEvent) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    if !delta.reasoning.is_empty()
        && send(reply.chunk(&json!({"reasoning_content": delta.reasoning}), None))
            .await
            .is_err()
    {
        return false;
    }
    if !delta.content.is_empty()
        && send(reply.chunk(&json!({"content": delta.content}), None))
            .await
            .is_err()
    {
        return false;
    }
    for call in &delta.tool_calls {
        let value = reply.tool_call(call, Some(*tool_calls));
        *tool_calls += 1;
        if send(reply.chunk(&json!({"tool_calls": [value]}), None))
            .await
            .is_err()
        {
            return false;
        }
    }
    true
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Seconds since the Unix epoch, for [`App::started`].
pub fn started() -> u64 {
    now()
}
