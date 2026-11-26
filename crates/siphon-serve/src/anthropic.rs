//! Anthropic's Messages API, so that Claude Code and Anthropic's SDKs can use the server.
//!
//! - `POST /v1/messages`, streaming or not. The request becomes OpenAI-shaped messages and
//!   tools for [`Vocab::render_to`] and runs through the same job, output and stop handling
//!   as `/v1/chat/completions`.
//! - `POST /v1/messages/count_tokens`, which renders the same prompt and counts it.
//!
//! Without `--vision` the model sees no images, so image blocks become a short text that
//! says so; with it, base64 image blocks reach the prompt as images. Documents always
//! become such a text. Anthropic's server tools are left out of the declared tools. Fields the server has no
//! use for, such as `metadata`, `cache_control` and `context_management`, are ignored.
//! Errors on these routes take Anthropic's shape, and a full queue answers 529, which
//! Anthropic's clients retry.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use axum::{
    Extension, Json,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    chat::{self, Ending, InvalidRequest, Rendered, Thinking, Vocab},
    engine::Finish,
    openai::{
        self, ApiError, App, Calls, Decoder, Delivery, Output, Prepared, RequestId, blocking,
        tool_ending,
    },
    parse::{Delta, Parser, ToolPart},
    request::{self, SamplingFields, ToolChoice},
    sse::{self, Frame, Prompt},
    stop::Stops,
    timings::Timings,
};

/// Whether `path` is one of this module's routes, whose errors take Anthropic's shape.
pub fn serves(path: &str) -> bool {
    path == "/v1/messages" || path.starts_with("/v1/messages/")
}

/// The time without bytes after which a stream sends a `ping` event, or a comment before
/// `message_start`, as during the prefill of a long prompt. Claude Code drops a stream
/// that stays silent for too long.
const PING: Duration = Duration::from_secs(5);

/// The text that stands in for an image block.
pub(crate) const NO_IMAGE: &str = "[image omitted: the model cannot see images]";
/// The text that stands in for a document block.
pub(crate) const NO_DOCUMENT: &str = "[document omitted: the model cannot read documents]";
/// Claude Code's first system block starts with this. It is metadata for Anthropic's API,
/// and its value changes between requests, which would defeat the prefix cache.
const BILLING_HEADER: &str = "x-anthropic-billing-header:";

/// An error in Anthropic's shape.
pub struct Error(pub ApiError);

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, kind) = if self.0.retry_after.is_some() {
            // A draining server: a 503 that the proxy retries elsewhere, not 529.
            (self.0.status, "overloaded_error")
        } else {
            error_kind(self.0.status)
        };
        self.0
            .with_retry_after((status, Json(error_body(kind, &self.0.message))).into_response())
    }
}

/// Anthropic's status and error type for one of ours. A full queue is 529, overloaded.
fn error_kind(status: StatusCode) -> (StatusCode, &'static str) {
    match status {
        StatusCode::BAD_REQUEST => (status, "invalid_request_error"),
        StatusCode::UNAUTHORIZED => (status, "authentication_error"),
        StatusCode::NOT_FOUND => (status, "not_found_error"),
        StatusCode::PAYLOAD_TOO_LARGE => (status, "request_too_large"),
        StatusCode::SERVICE_UNAVAILABLE => (
            StatusCode::from_u16(529).expect("529 is a valid status"),
            "overloaded_error",
        ),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, "api_error"),
    }
}

fn error_body(kind: &str, message: &str) -> Value {
    json!({"type": "error", "error": {"type": kind, "message": message}})
}

#[derive(Deserialize)]
pub struct MessagesRequest {
    messages: Vec<Value>,
    max_tokens: Option<u64>,
    system: Option<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    thinking: Option<Value>,
    output_config: Option<Value>,
    stop_sequences: Option<Value>,
    stream: Option<bool>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<i64>,
}

pub async fn messages(
    State(app): State<Arc<App>>,
    Extension(id): Extension<RequestId>,
    request: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<Response, Error> {
    let Json(request) = request.map_err(|e| Error(e.into()))?;
    let prepared = blocking(&app, move |app| prepare(app, &request))
        .await
        .map_err(Error)?;
    let message = Message::new(&app);
    sse::respond(app, prepared, message, &id)
        .await
        .map_err(Error)
}

pub async fn count_tokens(
    State(app): State<Arc<App>>,
    request: Result<Json<MessagesRequest>, JsonRejection>,
) -> Result<Json<Value>, Error> {
    let Json(request) = request.map_err(|e| Error(e.into()))?;
    let body = blocking(&app, move |app| count(&app.vocab, &request))
        .await
        .map_err(Error)?;
    Ok(Json(body))
}

/// The body of a `count_tokens` reply: the tokens of the prompt the request would run.
fn count(vocab: &Vocab, request: &MessagesRequest) -> Result<Value, ApiError> {
    Ok(json!({"input_tokens": render(vocab, request)?.0.ids.len()}))
}

fn prepare(app: &App, request: &MessagesRequest) -> Result<Prepared, ApiError> {
    let Some(requested) = request.max_tokens else {
        return Err(ApiError::bad_request("max_tokens: Field required"));
    };
    let stops = request::stops(request.stop_sequences.as_ref())?;
    let sampling = request::sampling(
        &SamplingFields {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            ..SamplingFields::default()
        },
        &app.sampling,
    )?;
    let (rendered, parser, calls) = render(&app.vocab, request)?;
    let max_tokens = limit(app.engine.max_context(), rendered.ids.len(), requested)?;
    Ok(Prepared {
        prompt: rendered.ids,
        images: rendered.images,
        max_tokens,
        delivery: if request.stream.unwrap_or(false) {
            Delivery::Stream {
                include_usage: false,
            }
        } else {
            Delivery::Whole
        },
        return_token_ids: false,
        ignore_eos: false,
        samplings: vec![sampling],
        output: Output::new(Decoder::Chat(parser), Stops::new(stops), Vec::new(), calls),
    })
}

/// `max_tokens` checked against the context, with Anthropic's messages: Claude Code reads
/// them to shrink `max_tokens` or to compact the conversation.
fn limit(max_context: usize, prompt_len: usize, requested: u64) -> Result<usize, ApiError> {
    if prompt_len >= max_context {
        return Err(ApiError::bad_request(format!(
            "prompt is too long: {prompt_len} tokens > {max_context} maximum"
        )));
    }
    let n = usize::try_from(requested).unwrap_or(usize::MAX);
    if n > max_context - prompt_len {
        return Err(ApiError::bad_request(format!(
            "input length and `max_tokens` exceed context limit: {prompt_len} + {n} > \
             {max_context}, decrease input length or `max_tokens` and try again"
        )));
    }
    openai::max_tokens(max_context, prompt_len, Some(requested))
}

/// The prompt of a request, the parser for its output, and the tool calls that end it.
fn render(vocab: &Vocab, request: &MessagesRequest) -> Result<(Rendered, Parser, Calls), ApiError> {
    let images = vocab.vision().is_some();
    let messages = translate(request.system.as_ref(), &request.messages, images)?;
    let mut tools = tools(request.tools.as_deref().unwrap_or_default());
    let (choice, parallel) = tool_choice(request.tool_choice.as_ref())?;
    if choice == ToolChoice::None {
        tools.clear();
    }
    // A last assistant message is a prefill, which the reply continues.
    let ending = if messages.last().and_then(|m| m.get("role")) == Some(&json!("assistant")) {
        Ending::Continue
    } else {
        Ending::Generate
    };
    let thinking = thinking(request.thinking.as_ref(), request.output_config.as_ref());
    let (ending, calls) = tool_ending(&choice, &tools, parallel, thinking.ending(ending, &tools))?;
    let prompt = vocab.render_prompt(&messages, &tools, thinking.effort(), &ending)?;
    Ok((prompt, Parser::new(vocab, &ending), calls))
}

/// The request's thinking: off when `thinking` is disabled, else `output_config.effort`
/// when it names one of the template's efforts, else the template's default.
fn thinking(thinking: Option<&Value>, output_config: Option<&Value>) -> Thinking {
    if thinking.and_then(|t| t.get("type")).and_then(Value::as_str) == Some("disabled") {
        return Thinking::Off;
    }
    let effort = output_config
        .and_then(|c| c.get("effort"))
        .filter(|e| e.is_string())
        .and_then(|e| chat::effort(Some(e)).ok())
        .unwrap_or(chat::DEFAULT_EFFORT);
    Thinking::Effort(effort)
}

/// Anthropic's `tool_choice` as ours, and whether the reply may make several calls.
fn tool_choice(value: Option<&Value>) -> Result<(ToolChoice, bool), InvalidRequest> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok((ToolChoice::Auto, true));
    };
    let parallel = value.get("disable_parallel_tool_use") != Some(&Value::Bool(true));
    let choice = match value.get("type").and_then(Value::as_str) {
        Some("auto") => ToolChoice::Auto,
        Some("any") => ToolChoice::Required,
        Some("none") => ToolChoice::None,
        Some("tool") => match value.get("name").and_then(Value::as_str) {
            Some(name) => ToolChoice::Function(name.to_owned()),
            None => return Err(invalid("tool_choice of type tool needs a name")),
        },
        _ => return Err(invalid("tool_choice type must be auto, any, tool or none")),
    };
    Ok((choice, parallel))
}

fn invalid(message: &str) -> InvalidRequest {
    InvalidRequest(message.to_owned())
}

/// Anthropic's tools as OpenAI function tools. Server tools, whose `type` is set and not
/// `custom`, run on Anthropic's side and are left out.
fn tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter(|tool| {
            matches!(tool.get("type"), None | Some(Value::Null)) || tool["type"] == "custom"
        })
        .map(|tool| {
            let mut function = Map::new();
            for (from, to) in [
                ("name", "name"),
                ("description", "description"),
                ("input_schema", "parameters"),
            ] {
                if let Some(value) = tool.get(from) {
                    function.insert(to.to_owned(), value.clone());
                }
            }
            json!({"type": "function", "function": function})
        })
        .collect()
}

/// Anthropic's `system` and `messages` as OpenAI-shaped messages, with image blocks kept
/// as images if `images` (`--vision`).
fn translate(
    system: Option<&Value>,
    messages: &[Value],
    images: bool,
) -> Result<Vec<Value>, InvalidRequest> {
    let mut out = Vec::new();
    if let Some(system) = system {
        let text = system_text(system)?;
        if !text.is_empty() {
            out.push(json!({"role": "system", "content": text}));
        }
    }
    for message in messages {
        let content = message.get("content").unwrap_or(&Value::Null);
        match message.get("role").and_then(Value::as_str) {
            Some("user") => user(content, &mut out, images)?,
            Some("assistant") => out.push(assistant(content)?),
            // Claude Code sends these mid-conversation. The template renders a system
            // message where it stands.
            Some("system") => out.push(json!({"role": "system", "content": system_text(content)?})),
            Some(role) => return Err(InvalidRequest(format!("unknown message role: {role}"))),
            None => return Err(invalid("every message needs a string role")),
        }
    }
    Ok(out)
}

/// The text blocks of a system prompt joined into one, less Claude Code's billing header.
fn system_text(system: &Value) -> Result<String, InvalidRequest> {
    let blocks = match system {
        Value::String(text) => return Ok(text.clone()),
        Value::Array(blocks) => blocks,
        Value::Null => return Ok(String::new()),
        _ => {
            return Err(invalid(
                "system must be a string or an array of text blocks",
            ));
        }
    };
    let texts: Vec<&str> = blocks
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .filter(|text| !text.starts_with(BILLING_HEADER))
        .collect();
    Ok(texts.join("\n\n"))
}

/// The blocks of a message's content, borrowed without cloning the block array.
enum ContentBlocks<'a> {
    Empty,
    Text(&'a str),
    Array(&'a [Value]),
}

fn content_blocks(content: &Value) -> Result<ContentBlocks<'_>, InvalidRequest> {
    match content {
        Value::String(text) => Ok(ContentBlocks::Text(text.as_str())),
        Value::Array(blocks) => Ok(ContentBlocks::Array(blocks.as_slice())),
        Value::Null => Ok(ContentBlocks::Empty),
        _ => Err(invalid(
            "message content must be a string or an array of blocks",
        )),
    }
}

/// The content part a user block stands for: its text, the image itself if `images`, or a
/// note in place of an image or document. Other block types are left out.
fn block_part(block: &Value, images: bool) -> Option<Value> {
    match block.get("type").and_then(Value::as_str) {
        Some("text") => block.get("text").and_then(Value::as_str).map(text_part),
        Some("image") if images => Some(block.clone()),
        Some("image") => Some(text_part(NO_IMAGE)),
        Some("document") => Some(text_part(NO_DOCUMENT)),
        _ => None,
    }
}

fn text_part(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

/// Adds a user message: its text and image blocks as user messages and each `tool_result`
/// as a tool message, in their order.
fn user(content: &Value, out: &mut Vec<Value>, images: bool) -> Result<(), InvalidRequest> {
    let blocks = match content_blocks(content)? {
        ContentBlocks::Empty => return Ok(()),
        ContentBlocks::Text(text) => {
            out.push(json!({"role": "user", "content": [text_part(text)]}));
            return Ok(());
        }
        ContentBlocks::Array(blocks) => blocks,
    };
    let mut parts = Vec::new();
    let flush = |parts: &mut Vec<Value>, out: &mut Vec<Value>| {
        if !parts.is_empty() {
            out.push(json!({"role": "user", "content": std::mem::take(parts)}));
        }
    };
    for block in blocks {
        if block.get("type").and_then(Value::as_str) == Some("tool_result") {
            flush(&mut parts, out);
            let id = block
                .get("tool_use_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            let content = match content_blocks(block.get("content").unwrap_or(&Value::Null))? {
                ContentBlocks::Empty => Value::String(String::new()),
                ContentBlocks::Text(text) => Value::Array(vec![text_part(text)]),
                ContentBlocks::Array(items) => {
                    let texts: Vec<Value> = items
                        .iter()
                        .filter_map(|b| block_part(b, images))
                        .collect();
                    if texts.is_empty() {
                        Value::String(String::new())
                    } else {
                        Value::Array(texts)
                    }
                }
            };
            out.push(json!({"role": "tool", "tool_call_id": id, "content": content}));
        } else if let Some(part) = block_part(block, images) {
            parts.push(part);
        }
    }
    flush(&mut parts, out);
    Ok(())
}

/// An assistant message: thinking as `reasoning_content`, text blocks as content, and
/// `tool_use` blocks as tool calls. The template renders a message's thinking first, then its
/// text, then its calls.
fn assistant(content: &Value) -> Result<Value, InvalidRequest> {
    let blocks = match content_blocks(content)? {
        ContentBlocks::Empty => return Ok(json!({"role": "assistant", "content": []})),
        ContentBlocks::Text(text) => {
            return Ok(json!({"role": "assistant", "content": [text_part(text)]}));
        }
        ContentBlocks::Array(blocks) => blocks,
    };
    let mut thinking = Vec::new();
    let mut texts = Vec::new();
    let mut calls = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("thinking") => {
                if let Some(text) = block.get("thinking").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    thinking.push(text);
                }
            }
            Some("text") => {
                texts.push(text_part(
                    block.get("text").and_then(Value::as_str).unwrap_or(""),
                ));
            }
            Some("tool_use") => {
                let arguments = match block.get("input") {
                    Some(input) => input.to_string(),
                    None => "{}".to_owned(),
                };
                calls.push(json!({
                    "id": block.get("id").cloned().unwrap_or(Value::Null),
                    "type": "function",
                    "function": {
                        "name": block.get("name").cloned().unwrap_or(Value::Null),
                        "arguments": arguments,
                    },
                }));
            }
            // `redacted_thinking` and server tool blocks carry nothing the model can read.
            _ => {}
        }
    }
    let mut message = json!({"role": "assistant", "content": texts});
    if !thinking.is_empty() {
        message["reasoning_content"] = Value::String(thinking.join("\n\n"));
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    Ok(message)
}

/// One reply: the fields its body and events share, and its content blocks.
struct Message {
    id: String,
    model: String,
    blocks: Blocks,
}

impl Message {
    fn new(app: &App) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let id = format!("msg_{:x}{n:06x}", openai::now());
        Self {
            blocks: Blocks::new(&id),
            id,
            model: app.name.clone(),
        }
    }

    /// Anthropic's usage: `input_tokens` leaves out the tokens read from the cache.
    fn usage(prompt: &Prompt, output_tokens: usize) -> Value {
        json!({
            "input_tokens": prompt.tokens.saturating_sub(prompt.cached),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": prompt.cached,
            "output_tokens": output_tokens,
        })
    }

    /// The message with `content`, with no stop reason while it runs.
    fn message(
        &self,
        content: &[Value],
        stop: Option<(&str, Value)>,
        prompt: &Prompt,
        output_tokens: usize,
    ) -> Value {
        let (reason, sequence) = stop.unzip();
        json!({
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": content,
            "stop_reason": reason,
            "stop_sequence": sequence,
            "usage": Self::usage(prompt, output_tokens),
        })
    }
}

/// `message_start` goes out with the first content block, and its usage then holds the
/// cached tokens. Before it, a quiet stream sends a comment, since Anthropic's protocol
/// puts `message_start` first; after it, a `ping` event.
impl sse::Reply for Message {
    const QUIET: Duration = PING;

    fn id(&self) -> &str {
        &self.id
    }

    fn start(&mut self, prompt: &Prompt) -> Vec<Frame> {
        vec![(
            "message_start",
            json!({"type": "message_start", "message": self.message(&[], None, prompt, 0)}),
        )]
    }

    fn keep_alive(&mut self) -> Frame {
        ("ping", json!({"type": "ping"}))
    }

    fn push(&mut self, delta: &Delta, events: &mut Vec<Frame>) {
        self.blocks.push(delta, events);
    }

    fn end(
        &mut self,
        output: &Output,
        finish: Finish,
        prompt: &Prompt,
        timings: &Timings,
    ) -> Vec<Frame> {
        let mut events = Vec::new();
        self.blocks.close(&mut events);
        let stop = stop_reason(output, finish);
        events.push((
            "message_delta",
            json!({"type": "message_delta",
                   "delta": {"stop_reason": stop.0, "stop_sequence": stop.1},
                   "usage": Self::usage(prompt, output.tokens), "timings": timings.json()}),
        ));
        events.push(("message_stop", json!({"type": "message_stop"})));
        events
    }

    fn body(
        &mut self,
        output: &Output,
        finish: Finish,
        prompt: &Prompt,
        timings: &Timings,
    ) -> Value {
        let stop = stop_reason(output, finish);
        let mut body = self.message(&self.blocks.content, Some(stop), prompt, output.tokens);
        body["timings"] = timings.json();
        body
    }

    /// An error event, before `message_start` too.
    fn failure(&mut self, message: &str, _: &Prompt, _: bool) -> Vec<Frame> {
        vec![("error", error_body("api_error", message))]
    }

    fn failed_response(message: &str) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_body("api_error", message)),
        )
            .into_response()
    }
}

/// Anthropic's `stop_reason` and `stop_sequence` for an output that ended with `finish`.
fn stop_reason(output: &Output, finish: Finish) -> (&'static str, Value) {
    if let Some(stop) = output.stop_string() {
        return ("stop_sequence", json!(stop));
    }
    let reason = match output.finish_reason(finish) {
        "length" => "max_tokens",
        "tool_calls" => "tool_use",
        _ => "end_turn",
    };
    (reason, Value::Null)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Open {
    Thinking,
    Text,
    Tool,
}

impl Open {
    /// The field of a delta that adds to the block, and the delta's type.
    fn field(self) -> (&'static str, &'static str) {
        match self {
            Self::Thinking => ("thinking", "thinking_delta"),
            Self::Text => ("text", "text_delta"),
            Self::Tool => ("partial_json", "input_json_delta"),
        }
    }
}

/// Builds a reply's content blocks from parsed deltas, and the events that stream them.
struct Blocks {
    content: Vec<Value>,
    /// The kind of the last block while it is open.
    open: Option<Open>,
    /// The open tool block's arguments so far.
    arguments: String,
    /// The message id without its prefix, for tool ids.
    base: String,
}

impl Blocks {
    fn new(message_id: &str) -> Self {
        Self {
            content: Vec::new(),
            open: None,
            arguments: String::new(),
            base: message_id.trim_start_matches("msg_").to_owned(),
        }
    }

    /// Adds `delta`'s thinking, then its text, then its tool calls' parts.
    fn push(&mut self, delta: &Delta, events: &mut Vec<Frame>) {
        if !delta.reasoning.is_empty() {
            self.text(Open::Thinking, &delta.reasoning, events);
        }
        if !delta.content.is_empty() {
            self.text(Open::Text, &delta.content, events);
        }
        for part in &delta.tool_parts {
            match part {
                ToolPart::Start(name) => self.tool(name, events),
                ToolPart::Arguments(piece) if self.open == Some(Open::Tool) => {
                    self.arguments.push_str(piece);
                    self.grow(piece, events);
                }
                ToolPart::Arguments(_) => {}
                ToolPart::End => self.close(events),
            }
        }
    }

    /// Adds `text` to the open thinking or text block of `kind`, opening one if needed.
    fn text(&mut self, kind: Open, text: &str, events: &mut Vec<Frame>) {
        let (field, _) = kind.field();
        if self.open != Some(kind) {
            self.close(events);
            let mut block = json!({"type": field, field: ""});
            if kind == Open::Thinking {
                block["signature"] = json!("");
            }
            events.push((
                "content_block_start",
                json!({"type": "content_block_start", "index": self.content.len(),
                       "content_block": block}),
            ));
            self.content.push(block);
            self.open = Some(kind);
        }
        let index = self.content.len() - 1;
        if let Some(Value::String(all)) = self.content[index].get_mut(field) {
            all.push_str(text);
        }
        self.grow(text, events);
    }

    /// Streams `text` as a delta of the open block. A delta event for the same block at the
    /// end of `events` grows instead of a new one being added.
    fn grow(&mut self, text: &str, events: &mut Vec<Frame>) {
        let Some(kind) = self.open else {
            return;
        };
        let (field, delta_type) = kind.field();
        let index = self.content.len() - 1;
        if let Some(("content_block_delta", last)) = events.last_mut()
            && last.get("index").and_then(Value::as_u64) == Some(index as u64)
            && let Some(Value::String(sent)) = last.get_mut("delta").and_then(|d| d.get_mut(field))
        {
            sent.push_str(text);
            return;
        }
        events.push((
            "content_block_delta",
            json!({"type": "content_block_delta", "index": index,
                   "delta": {"type": delta_type, field: text}}),
        ));
    }

    /// Closes the open block. A thinking block gets an empty signature first, and a tool
    /// block takes its arguments as its input.
    fn close(&mut self, events: &mut Vec<Frame>) {
        let Some(kind) = self.open.take() else {
            return;
        };
        let index = self.content.len() - 1;
        match kind {
            Open::Thinking => events.push((
                "content_block_delta",
                json!({"type": "content_block_delta", "index": index,
                       "delta": {"type": "signature_delta", "signature": ""}}),
            )),
            Open::Tool => {
                // The parser streams a call's argument object as the model wrote it; an
                // empty object stands in if it does not parse, as when the reply was cut.
                let input = match serde_json::from_str(&std::mem::take(&mut self.arguments)) {
                    Ok(Value::Object(input)) => input,
                    _ => Map::new(),
                };
                self.content[index]["input"] = Value::Object(input);
            }
            Open::Text => {}
        }
        events.push((
            "content_block_stop",
            json!({"type": "content_block_stop", "index": index}),
        ));
    }

    /// Opens a tool block for a call of `name`.
    fn tool(&mut self, name: &str, events: &mut Vec<Frame>) {
        self.close(events);
        let index = self.content.len();
        let block = json!({"type": "tool_use", "id": format!("toolu_{}_{index}", self.base),
                           "name": name, "input": {}});
        events.push((
            "content_block_start",
            json!({"type": "content_block_start", "index": index, "content_block": block}),
        ));
        self.content.push(block);
        self.open = Some(Open::Tool);
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use axum::{body, http::HeaderMap, response::sse::Sse};
    use tokio::sync::mpsc;

    use super::*;
    use crate::{
        chat::tests::vocab,
        engine::Event,
        metrics::Metrics,
        openai::Results,
        sse::{KEEP_ALIVE, run},
    };

    /// A reply with the id `msg_1` from the model `m`.
    fn message() -> Message {
        Message {
            id: "msg_1".to_owned(),
            model: "m".to_owned(),
            blocks: Blocks::new("msg_1"),
        }
    }

    fn parse(request: Value) -> MessagesRequest {
        serde_json::from_value(request).unwrap()
    }

    /// The decoded prompt of `request`.
    fn rendered(vocab: &Vocab, request: Value) -> String {
        vocab.decode(&render(vocab, &parse(request)).unwrap().0.ids)
    }

    /// A request shaped like Claude Code's, with fields the server ignores.
    fn claude_code_request() -> Value {
        json!({
            "model": "inkling-2-mini",
            "max_tokens": 32000,
            "stream": true,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.0; cch=1a2b;"},
                {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Be brief."},
            ],
            "tools": [
                {"name": "Read", "description": "Reads a file.", "input_schema": {
                    "type": "object", "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"]}, "cache_control": {"type": "ephemeral"}},
                {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "What is in logo.png?"}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "I should read it.", "signature": "c2ln"},
                    {"type": "redacted_thinking", "data": "eHl6"},
                    {"type": "text", "text": "Reading it."},
                    {"type": "tool_use", "id": "toolu_01", "name": "Read",
                     "input": {"file_path": "logo.png"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": [
                        {"type": "text", "text": "PNG, 64x64"},
                        {"type": "image", "source": {"type": "base64",
                         "media_type": "image/png", "data": "iVBORw0KGgo="}},
                    ]},
                    {"type": "text", "text": "Describe it.", "cache_control": {"type": "ephemeral"}},
                ]},
                {"role": "system", "content": "The model inkling-2-mini is not a Claude model."},
                {"role": "assistant", "content": "A small logo."},
                {"role": "user", "content": "Thanks."},
            ],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
            "metadata": {"user_id": "u1"},
            "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
            "service_tier": "auto",
            "container": null,
            "mcp_servers": [],
        })
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn image_blocks_reach_the_prompt_with_vision() {
        let mut vocab = vocab();
        vocab.set_vision(Some(crate::vision::tests::vision(4)));
        let mut request = claude_code_request();
        let png = crate::vision::tests::png_header(80, 40);
        request["messages"][2]["content"][0]["content"][1]["source"]["data"] = json!(png);
        request["messages"][5]["content"] = json!([
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
             "data": png}},
            {"type": "text", "text": "And this one?"},
        ]);
        let (rendered, _, _) = render(&vocab, &parse(request)).unwrap();
        let text = vocab.decode(&rendered.ids);
        let image = "<|content_image|><|unused_200054|><|unused_200054|><|end_message|>";
        assert!(
            text.contains(&format!(
                "<|message_tool|>Read<|content_text|>PNG, 64x64<|end_message|>\
                 <|message_tool|>Read{image}<|message_user|><|content_text|>Describe it."
            )),
            "{text}"
        );
        assert!(
            text.contains(&format!(
                "<|message_user|>{image}<|message_user|><|content_text|>And this one?"
            )),
            "{text}"
        );
        assert!(!text.contains("image omitted"));
        assert_eq!(rendered.images.len(), 2);
        for image in &rendered.images {
            assert_eq!(
                rendered.ids[image.positions()],
                [crate::vision::tests::PLACEHOLDER; 2]
            );
        }
        // An image source by URL is refused rather than fetched.
        let mut request = claude_code_request();
        request["messages"][5]["content"] =
            json!([{"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}]);
        assert!(render(&vocab, &parse(request)).is_err());
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn translates_a_claude_code_request() {
        let vocab = vocab();
        let mut request = claude_code_request();
        let head = "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Reads a \
                    file.\",\"name\":\"Read\",\"parameters\":{\"properties\":{\"file_path\":\
                    {\"type\":\"string\"}},\"required\":[\"file_path\"],\"type\":\"object\"},\
                    \"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>\
                    You are Claude Code.\n\nBe brief.<|end_message|><|message_system|>\
                    <|content_text|>Thinking effort level: 0.7<|end_message|>";
        let body = "<|message_user|><|content_text|>What is in logo.png?<|end_message|>\
                    <|message_model|><|content_thinking|>I should read it.<|end_message|>\
                    <|message_model|><|content_text|>Reading it.<|end_message|>\
                    <|message_model|>Read<|content_invoke_tool_json|>{\"name\":\"Read\",\
                    \"args\":{\"file_path\":\"logo.png\"}}<|end_message|>\
                    <|content_model_end_sampling|>\
                    <|message_tool|>Read<|content_text|>PNG, 64x64<|end_message|>\
                    <|message_tool|>Read<|content_text|>[image omitted: the model cannot see \
                    images]<|end_message|>\
                    <|message_user|><|content_text|>Describe it.<|end_message|>\
                    <|message_system|><|content_text|>The model inkling-2-mini is not a Claude \
                    model.<|end_message|>\
                    <|message_model|><|content_text|>A small logo.<|end_message|>\
                    <|content_model_end_sampling|>\
                    <|message_user|><|content_text|>Thanks.<|end_message|>";
        assert_eq!(
            rendered(&vocab, request.clone()),
            format!("{head}{body}<|message_model|>")
        );

        // Disabled thinking is effort 0. With tools, the model thinks briefly anyway, as it
        // decides on tool calls there.
        request["thinking"] = json!({"type": "disabled"});
        let head = head.replace("level: 0.7", "level: 0");
        assert_eq!(
            rendered(&vocab, request.clone()),
            format!("{head}{body}<|message_model|>")
        );
        // A named tool opens its call.
        request["tool_choice"] = json!({"type": "tool", "name": "Read"});
        let text = rendered(&vocab, request.clone());
        assert!(text.ends_with("<|message_model|>Read<|content_invoke_tool_json|>"));
        // `none` drops the tools, so the prompt ends with the thought that skips thinking.
        request["tool_choice"] = json!({"type": "none"});
        let text = rendered(&vocab, request.clone());
        assert!(!text.contains("tool_declare"));
        assert!(text.ends_with(
            "<|message_user|><|content_text|>Thanks.<|end_message|><|message_model|>\
             <|content_thinking|>Respond directly.<|end_message|><|message_model|>"
        ));
        // A last assistant message is a prefill that the reply continues.
        let prefill = json!({"max_tokens": 8, "messages": [
            {"role": "user", "content": "Name a colour."},
            {"role": "assistant", "content": [{"type": "text", "text": "The colour is"}]}]});
        assert!(rendered(&vocab, prefill.clone()).ends_with(
            "<|message_user|><|content_text|>Name a colour.<|end_message|>\
             <|message_model|><|content_text|>The colour is"
        ));
        // The prefill's open text block is the reply, so disabled thinking adds no block.
        let mut disabled = prefill;
        disabled["thinking"] = json!({"type": "disabled"});
        assert!(
            rendered(&vocab, disabled).ends_with("<|message_model|><|content_text|>The colour is")
        );
        // A tool_choice that names a server tool, which is not declared, is refused.
        request["tool_choice"] = json!({"type": "tool", "name": "web_search"});
        assert!(render(&vocab, &parse(request)).is_err());
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn counts_the_rendered_prompt() {
        let vocab = vocab();
        let body = count(
            &vocab,
            &parse(json!({"model": "m", "messages": [{"role": "user", "content": "Hello"}]})),
        )
        .unwrap();
        let messages = [json!({"role": "user", "content": "Hello"})];
        let expected = vocab
            .render(&messages, &[], chat::DEFAULT_EFFORT)
            .unwrap()
            .len();
        assert_eq!(body, json!({"input_tokens": expected}));
        let full = count(&vocab, &parse(claude_code_request())).unwrap();
        let ids = render(&vocab, &parse(claude_code_request())).unwrap().0.ids;
        assert_eq!(full["input_tokens"], ids.len());
    }

    #[test]
    fn thinking_tool_choice_and_tools() {
        let effort = |level| Thinking::Effort(level);
        assert_eq!(thinking(None, None), effort(chat::DEFAULT_EFFORT));
        let disabled = json!({"type": "disabled"});
        let adaptive = json!({"type": "adaptive"});
        let low = json!({"effort": "low"});
        assert_eq!(thinking(Some(&disabled), Some(&low)), Thinking::Off);
        assert_eq!(thinking(Some(&adaptive), Some(&low)), effort(0.2));
        assert_eq!(
            thinking(None, Some(&json!({"effort": "max"}))),
            effort(0.99)
        );
        assert_eq!(
            thinking(None, Some(&json!({"effort": "ultra"}))),
            effort(chat::DEFAULT_EFFORT)
        );
        assert_eq!(
            thinking(None, Some(&json!({"effort": 0.5}))),
            effort(chat::DEFAULT_EFFORT)
        );

        let choice = |v: Value| tool_choice(Some(&v)).unwrap();
        assert_eq!(tool_choice(None).unwrap(), (ToolChoice::Auto, true));
        assert_eq!(choice(json!({"type": "any"})), (ToolChoice::Required, true));
        assert_eq!(choice(json!({"type": "none"})), (ToolChoice::None, true));
        assert_eq!(
            choice(json!({"type": "tool", "name": "f", "disable_parallel_tool_use": true})),
            (ToolChoice::Function("f".to_owned()), false)
        );
        assert_eq!(
            choice(json!({"type": "auto", "disable_parallel_tool_use": true})),
            (ToolChoice::Auto, false)
        );
        assert!(tool_choice(Some(&json!({"type": "tool"}))).is_err());
        assert!(tool_choice(Some(&json!({"type": "required"}))).is_err());

        let declared = tools(&[
            json!({"name": "f", "input_schema": {"type": "object"}}),
            json!({"type": "custom", "name": "g", "description": "G."}),
            json!({"type": "bash_20250124", "name": "bash"}),
            json!({"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}),
        ]);
        assert_eq!(
            declared,
            [
                json!({"type": "function", "function": {"name": "f",
                       "parameters": {"type": "object"}}}),
                json!({"type": "function", "function": {"name": "g", "description": "G."}}),
            ]
        );
    }

    #[test]
    fn max_tokens_against_the_context() {
        assert_eq!(limit(100, 40, 60).ok(), Some(60));
        let message = |r: Result<usize, ApiError>| r.err().unwrap().message;
        assert_eq!(
            message(limit(100, 40, 61)),
            "input length and `max_tokens` exceed context limit: 40 + 61 > 100, decrease \
             input length or `max_tokens` and try again"
        );
        assert_eq!(
            message(limit(100, 100, 1)),
            "prompt is too long: 100 tokens > 100 maximum"
        );
        assert!(limit(100, 40, 0).is_err());
    }

    async fn error_response(status: StatusCode) -> (StatusCode, Value) {
        let response = Error(ApiError::new(status, "why")).into_response();
        let status = response.status();
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn errors_take_anthropics_shape() {
        let body = |kind: &str| json!({"type": "error", "error": {"type": kind, "message": "why"}});
        assert_eq!(
            error_response(StatusCode::SERVICE_UNAVAILABLE).await,
            (StatusCode::from_u16(529).unwrap(), body("overloaded_error"))
        );
        assert_eq!(
            error_response(StatusCode::BAD_REQUEST).await,
            (StatusCode::BAD_REQUEST, body("invalid_request_error"))
        );
        assert_eq!(
            error_response(StatusCode::UNAUTHORIZED).await,
            (StatusCode::UNAUTHORIZED, body("authentication_error"))
        );
        assert_eq!(
            error_response(StatusCode::INTERNAL_SERVER_ERROR).await,
            (StatusCode::INTERNAL_SERVER_ERROR, body("api_error"))
        );
        assert!(serves("/v1/messages"));
        assert!(serves("/v1/messages/count_tokens"));
        assert!(!serves("/v1/chat/completions"));
        assert!(!serves("/v1/messagesx"));
    }

    #[test]
    fn x_api_key_is_accepted() {
        let mut headers = HeaderMap::new();
        assert!(!openai::authorized(&headers, "k3y"));
        headers.insert("x-api-key", "k3y".parse().unwrap());
        assert!(openai::authorized(&headers, "k3y"));
        assert!(!openai::authorized(&headers, "other"));
        let mut headers = HeaderMap::new();
        headers.insert("authorization", "Bearer k3y".parse().unwrap());
        assert!(openai::authorized(&headers, "k3y"));
        headers.insert("x-api-key", "wrong".parse().unwrap());
        assert!(openai::authorized(&headers, "k3y"));
    }

    /// Runs `events` through [`run`] for a prompt of 10 tokens and returns the frames sent.
    async fn collect(
        vocab: &Vocab,
        stops: &[&str],
        calls: Calls,
        events: Vec<Event>,
        stream: bool,
    ) -> Vec<Frame> {
        let parser = Parser::new(vocab, &Ending::Generate);
        collect_parsed(vocab, parser, stops, calls, events, stream).await
    }

    /// [`collect`] with the output parsed by `parser`.
    async fn collect_parsed(
        vocab: &Vocab,
        parser: Parser,
        stops: &[&str],
        calls: Calls,
        events: Vec<Event>,
        stream: bool,
    ) -> Vec<Frame> {
        let (sender, receiver) = mpsc::unbounded_channel();
        for event in events {
            sender.send(event).unwrap();
        }
        let results = Results {
            receiver,
            tracker: Arc::new(Metrics::new("m")).track(10),
        };
        let output = Output::new(
            Decoder::Chat(parser),
            Stops::new(stops.iter().map(|&s| s.to_owned()).collect()),
            Vec::new(),
            calls,
        );
        let (out, mut rx) = mpsc::channel(1024);
        run(vocab, results, output, message(), 10, out, stream).await;
        let mut frames = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            frames.push(frame);
        }
        frames
    }

    /// Removes the timings from the last frame, the `message_delta` or the message, and
    /// returns them.
    fn take_timings(frames: &mut [Frame]) -> Value {
        let (name, data) = frames
            .iter_mut()
            .rev()
            .find(|(name, _)| *name != "message_stop")
            .expect("a frame");
        data.as_object_mut()
            .and_then(|data| data.remove("timings"))
            .unwrap_or_else(|| panic!("no timings in {name}"))
    }

    /// Ids of a reply that thinks, writes text and calls a tool, as the engine sends them:
    /// without the turn's end token.
    fn reply(vocab: &Vocab) -> Vec<u32> {
        let s = vocab.special;
        let mut ids = vec![s.content_thinking];
        ids.extend(vocab.encode("Let me look."));
        ids.extend([s.end_message, s.message_model, s.content_text]);
        ids.extend(vocab.encode("Reading the file."));
        ids.extend([s.end_message, s.message_model]);
        ids.extend(vocab.encode("Read"));
        ids.push(s.content_invoke_tool_json);
        ids.extend(vocab.encode(r#"{"name":"Read","args":{"file_path":"a.rs"}}"#));
        ids.push(s.end_message);
        ids
    }

    fn text(vocab: &Vocab, text: &str) -> Vec<u32> {
        let mut ids = vec![vocab.special.content_text];
        ids.extend(vocab.encode(text));
        ids
    }

    #[tokio::test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn streams_thinking_text_and_a_tool_call() {
        let vocab = vocab();
        let ids = reply(&vocab);
        let n = ids.len();
        let events = vec![
            Event::Started,
            Event::Cached(4),
            Event::Tokens(ids),
            Event::Done(Finish::Stop),
        ];
        let mut frames = collect(&vocab, &[], Calls::Free, events, true).await;
        let timings = take_timings(&mut frames);
        assert_eq!(timings["predicted_n"], n);
        assert_eq!(timings["cache_n"], 4);
        let delta = |index: usize, delta: Value| {
            (
                "content_block_delta",
                json!({"type": "content_block_delta", "index": index,
                                           "delta": delta}),
            )
        };
        let start = |index: usize, block: Value| {
            (
                "content_block_start",
                json!({"type": "content_block_start", "index": index,
                                           "content_block": block}),
            )
        };
        let stop = |index: usize| {
            (
                "content_block_stop",
                json!({"type": "content_block_stop", "index": index}),
            )
        };
        let expected: Vec<Frame> = vec![
            (
                "message_start",
                json!({"type": "message_start", "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 6, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 4, "output_tokens": 0}}}),
            ),
            start(
                0,
                json!({"type": "thinking", "thinking": "", "signature": ""}),
            ),
            delta(
                0,
                json!({"type": "thinking_delta", "thinking": "Let me look."}),
            ),
            delta(0, json!({"type": "signature_delta", "signature": ""})),
            stop(0),
            start(1, json!({"type": "text", "text": ""})),
            delta(
                1,
                json!({"type": "text_delta", "text": "Reading the file."}),
            ),
            stop(1),
            start(
                2,
                json!({"type": "tool_use", "id": "toolu_1_2", "name": "Read",
                            "input": {}}),
            ),
            delta(
                2,
                json!({"type": "input_json_delta",
                            "partial_json": r#"{"file_path":"a.rs"}"#}),
            ),
            stop(2),
            (
                "message_delta",
                json!({"type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": null},
                "usage": {"input_tokens": 6, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 4, "output_tokens": n}}),
            ),
            ("message_stop", json!({"type": "message_stop"})),
        ];
        assert_eq!(frames, expected);

        // Ids one at a time: the deltas split, the blocks stay the same.
        let one_by_one: Vec<Event> = reply(&vocab)
            .into_iter()
            .map(|id| Event::Tokens(vec![id]))
            .chain([Event::Done(Finish::Stop)])
            .collect();
        let frames = collect(&vocab, &[], Calls::Free, one_by_one, true).await;
        let kinds =
            |frames: &[Frame], name: &str| frames.iter().filter(|(n, _)| *n == name).count();
        assert_eq!(kinds(&frames, "content_block_start"), 3);
        assert_eq!(kinds(&frames, "content_block_stop"), 3);
        let thinking: String = frames
            .iter()
            .filter_map(|(_, data)| data.pointer("/delta/thinking").and_then(Value::as_str))
            .collect();
        assert_eq!(thinking, "Let me look.");
    }

    #[tokio::test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn disabled_thinking_streams_only_text() {
        let vocab = vocab();
        let request = parse(json!({"max_tokens": 32, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Capital of Australia?"}]}));
        let (_, parser, calls) = render(&vocab, &request).unwrap();
        // After the prompt's thinking block, the model opens a text block.
        let mut ids = text(&vocab, "Canberra.");
        ids.push(vocab.special.end_message);
        let n = ids.len();
        let events = vec![Event::Tokens(ids), Event::Done(Finish::Stop)];
        let frames = collect_parsed(&vocab, parser.clone(), &[], calls, events, true).await;
        let names: Vec<&str> = frames.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(
            frames[1].1,
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}})
        );
        assert_eq!(
            frames[2].1,
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "Canberra."}})
        );

        let mut ids = text(&vocab, "Canberra.");
        ids.push(vocab.special.end_message);
        let events = vec![Event::Tokens(ids), Event::Done(Finish::Stop)];
        let (_, _, calls) = render(&vocab, &request).unwrap();
        let frames = collect_parsed(&vocab, parser, &[], calls, events, false).await;
        assert_eq!(
            frames[0].1["content"],
            json!([{"type": "text", "text": "Canberra."}])
        );
        assert_eq!(frames[0].1["usage"]["output_tokens"], n);
    }

    #[tokio::test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn whole_body_and_usage() {
        let vocab = vocab();
        let ids = reply(&vocab);
        let n = ids.len();
        let mut events = vec![Event::Started, Event::Cached(4)];
        events.extend(ids.into_iter().map(|id| Event::Tokens(vec![id])));
        events.push(Event::Done(Finish::Stop));
        let mut frames = collect(&vocab, &[], Calls::Free, events, false).await;
        assert_eq!(take_timings(&mut frames)["prompt_n"], 6);
        assert_eq!(
            frames,
            [(
                "body",
                json!({
                    "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                    "content": [
                        {"type": "thinking", "thinking": "Let me look.", "signature": ""},
                        {"type": "text", "text": "Reading the file."},
                        {"type": "tool_use", "id": "toolu_1_2", "name": "Read",
                         "input": {"file_path": "a.rs"}},
                    ],
                    "stop_reason": "tool_use",
                    "stop_sequence": null,
                    "usage": {"input_tokens": 6, "cache_creation_input_tokens": 0,
                              "cache_read_input_tokens": 4, "output_tokens": n},
                })
            )]
        );
    }

    #[tokio::test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn stop_reasons() {
        let vocab = vocab();
        let body = async |stops: &[&str], calls: Calls, ids: Vec<u32>, finish: Finish| {
            let events = vec![Event::Tokens(ids), Event::Done(finish)];
            let mut frames = collect(&vocab, stops, calls, events, false).await;
            frames.pop().unwrap().1
        };
        let reason = |body: &Value| (body["stop_reason"].clone(), body["stop_sequence"].clone());

        let ended = body(&[], Calls::Free, text(&vocab, "Done."), Finish::Stop).await;
        assert_eq!(reason(&ended), (json!("end_turn"), Value::Null));
        assert_eq!(ended["content"], json!([{"type": "text", "text": "Done."}]));

        let cut = body(&[], Calls::Free, text(&vocab, "Lon"), Finish::Length).await;
        assert_eq!(reason(&cut), (json!("max_tokens"), Value::Null));

        let stopped = body(
            &["STOP"],
            Calls::Free,
            text(&vocab, "one STOP two"),
            Finish::Stop,
        )
        .await;
        assert_eq!(reason(&stopped), (json!("stop_sequence"), json!("STOP")));
        assert_eq!(
            stopped["content"],
            json!([{"type": "text", "text": "one "}])
        );

        let called = body(&[], Calls::Free, reply(&vocab), Finish::Stop).await;
        assert_eq!(reason(&called), (json!("tool_use"), Value::Null));

        // disable_parallel_tool_use: the output ends after the first call.
        let mut two = reply(&vocab);
        two.extend(reply(&vocab));
        let one = body(&[], Calls::One, two, Finish::Length).await;
        assert_eq!(reason(&one), (json!("tool_use"), Value::Null));
        let calls = one["content"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|b| b["type"] == "tool_use")
            .count();
        assert_eq!(calls, 1);
    }

    /// The names of `frames`, with `:` for a comment.
    fn names(frames: &[Frame]) -> Vec<&'static str> {
        frames
            .iter()
            .map(|(name, _)| if *name == KEEP_ALIVE { ":" } else { *name })
            .collect()
    }

    /// Runs a stream whose events arrive after `delays` seconds each, on a paused clock.
    async fn timed(vocab: &Vocab, events: Vec<(u64, Event)>) -> Vec<Frame> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let results = Results {
            receiver,
            tracker: Arc::new(Metrics::new("m")).track(10),
        };
        let output = Output::new(
            Decoder::Chat(Parser::new(vocab, &Ending::Generate)),
            Stops::new(Vec::new()),
            Vec::new(),
            Calls::Free,
        );
        let (out, mut rx) = mpsc::channel(1024);
        let engine = async move {
            for (delay, event) in events {
                tokio::time::sleep(Duration::from_secs(delay)).await;
                sender.send(event).unwrap();
            }
        };
        tokio::join!(
            run(vocab, results, output, message(), 10, out, true),
            engine
        );
        let mut frames = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            frames.push(frame);
        }
        frames
    }

    #[tokio::test(start_paused = true)]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn message_start_waits_for_the_first_token() {
        let vocab = vocab();
        let s = vocab.special;
        // A 12 s prefill, then a thinking marker that adds no block, then text, then 7 s
        // without bytes in the reply.
        let events = vec![
            (0, Event::Started),
            (0, Event::Cached(4)),
            (12, Event::Tokens(vec![s.content_thinking])),
            (0, Event::Tokens(vocab.encode("Plan."))),
            (7, Event::Tokens(vec![s.end_message])),
            (0, Event::Done(Finish::Stop)),
        ];
        let frames = timed(&vocab, events).await;
        assert_eq!(
            names(&frames)[..5],
            [
                ":",
                ":",
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );
        // The prefix cache's tokens reached message_start's usage.
        assert_eq!(
            frames[2].1["message"]["usage"]["cache_read_input_tokens"],
            4
        );
        assert_eq!(frames[2].1["message"]["usage"]["input_tokens"], 6);
        assert_eq!(frames[4].1["delta"]["thinking"], "Plan.");
        // After message_start, a quiet stream pings.
        assert_eq!(names(&frames)[5], "ping");
        assert_eq!(*names(&frames).last().unwrap(), "message_stop");

        // An error before the first token is an error event alone.
        let events = vec![
            (0, Event::Started),
            (6, Event::Failed("the KV pool is full".to_owned())),
        ];
        let frames = timed(&vocab, events).await;
        assert_eq!(names(&frames), [":", "error"]);
        assert_eq!(
            frames[1].1,
            json!({"type": "error", "error": {"type": "api_error",
                                              "message": "the KV pool is full"}})
        );

        // A reply without content still starts with message_start.
        let events = vec![(0, Event::Started), (1, Event::Done(Finish::Length))];
        let frames = timed(&vocab, events).await;
        assert_eq!(
            names(&frames),
            ["message_start", "message_delta", "message_stop"]
        );
    }

    #[tokio::test]
    async fn keep_alives_are_comments() {
        let frames = [(KEEP_ALIVE, Value::Null), ("ping", json!({"type": "ping"}))];
        let events = futures_util::stream::iter(frames.map(|f| Ok::<_, Infallible>(sse::sse(f))));
        let response = Sse::new(events).into_response();
        let bytes = body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            std::str::from_utf8(&bytes).unwrap(),
            ": \n\nevent: ping\ndata: {\"type\":\"ping\"}\n\n"
        );
    }

    #[tokio::test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn a_failure_mid_stream_sends_an_error_event() {
        let vocab = vocab();
        let events = vec![
            Event::Started,
            Event::Tokens(text(&vocab, "Hi")),
            Event::Failed("the GPU fell over".to_owned()),
        ];
        let frames = collect(&vocab, &[], Calls::Free, events, true).await;
        let names: Vec<&str> = frames.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "error"
            ]
        );
        assert_eq!(
            frames[3].1,
            json!({"type": "error", "error": {"type": "api_error",
                                              "message": "the GPU fell over"}})
        );
    }

    #[test]
    fn borrowed_translation_and_grow_delta_coalescing() {
        let translated = translate(
            Some(&json!([
                {"type": "text", "text": "x-anthropic-billing-header: ignored"},
                {"type": "text", "text": "System."},
            ])),
            &[
                json!({"role": "user", "content": "Hello"}),
                json!({"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "Think."},
                    {"type": "text", "text": "Calling."},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "a.rs"}},
                ]}),
                json!({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                    {"type": "text", "text": "Done?"},
                ]}),
            ],
            false,
        )
        .unwrap();
        assert_eq!(translated.len(), 5);
        assert_eq!(translated[0]["role"], "system");
        assert_eq!(translated[1]["role"], "user");
        assert_eq!(translated[2]["reasoning_content"], "Think.");
        assert_eq!(translated[3]["role"], "tool");
        assert_eq!(translated[3]["tool_call_id"], "t1");
        assert_eq!(translated[4]["role"], "user");

        // Verify `Blocks::grow` coalesces consecutive deltas on the same block.
        let mut blocks = Blocks::new("msg_1");
        let mut events = Vec::new();
        blocks.push(
            &Delta {
                content: "Hel".to_owned(),
                ..Delta::default()
            },
            &mut events,
        );
        blocks.push(
            &Delta {
                content: "lo".to_owned(),
                ..Delta::default()
            },
            &mut events,
        );
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].0, "content_block_start");
        assert_eq!(events[1].0, "content_block_delta");
        assert_eq!(events[1].1["delta"]["text"], "Hello");
    }
}
