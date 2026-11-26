//! OpenAI's Responses API, `POST /v1/responses`, so that the Codex CLI and OpenAI's SDKs can
//! use the server.
//!
//! The request becomes OpenAI-shaped chat messages and tools for [`Vocab::render_prompt`]
//! and runs through the same job, output and stop handling as `/v1/chat/completions`. The
//! server stores nothing: a request carries the whole conversation, as Codex sends it with
//! `store: false`, and `previous_response_id` is refused.
//!
//! The model's thinking is public. A reasoning item holds its whole text as a
//! `reasoning_text` content part, as open-weight servers return it, and also as its summary
//! when the request asks for one, since Codex shows summaries by default. With `include:
//! ["reasoning.encrypted_content"]`, `encrypted_content` holds the same text in base64. A
//! reasoning item sent back renders as the assistant's past thinking, as on the chat route.
//!
//! Tools of a type other than `function`, such as web search or Codex's namespaces, are
//! left out, as are input items the model cannot read, such as hosted tool calls. Errors
//! take OpenAI's shape.

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
    response::{IntoResponse, Response},
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    anthropic::{NO_DOCUMENT, NO_IMAGE},
    chat::{self, Ending, InvalidRequest, Rendered, Vocab},
    engine::Finish,
    openai::{
        self, ApiError, App, Calls, Decoder, Delivery, Output, Prepared, RequestId, blocking,
        tool_ending,
    },
    parse::{Delta, Parser, ToolPart},
    request::{self, SamplingFields, ToolChoice},
    sse::{self, Frame, KEEP_ALIVE, Prompt},
    stop::Stops,
    timings::Timings,
};

/// The time without bytes after which a stream sends an SSE comment, as the chat route's
/// keep-alive does.
const QUIET: Duration = Duration::from_secs(15);

/// The `include` value that asks for the reasoning text in `encrypted_content`.
const ENCRYPTED_REASONING: &str = "reasoning.encrypted_content";

#[derive(Deserialize)]
pub struct ResponsesRequest {
    instructions: Option<Value>,
    input: Option<Value>,
    tools: Option<Vec<Value>>,
    tool_choice: Option<Value>,
    parallel_tool_calls: Option<bool>,
    reasoning: Option<Value>,
    max_output_tokens: Option<u64>,
    stop: Option<Value>,
    stream: Option<bool>,
    include: Option<Vec<Value>>,
    text: Option<Value>,
    previous_response_id: Option<Value>,
    conversation: Option<Value>,
    background: Option<bool>,
    #[serde(flatten)]
    sampling: SamplingFields,
    /// Every other field: refused ones are checked, the rest, such as `store`, ignored.
    #[serde(flatten)]
    other: Map<String, Value>,
}

pub async fn responses(
    State(app): State<Arc<App>>,
    Extension(id): Extension<RequestId>,
    request: Result<Json<ResponsesRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = request?;
    let prepared = blocking(&app, move |app| prepare(app, &request)).await?;
    let (prepared, reply) = prepared;
    sse::respond(app, prepared, reply, &id).await
}

fn prepare(app: &App, request: &ResponsesRequest) -> Result<(Prepared, Reply), ApiError> {
    let set = |value: Option<&Value>| value.is_some_and(|v| !v.is_null());
    if set(request.previous_response_id.as_ref()) || set(request.conversation.as_ref()) {
        return Err(ApiError::bad_request(
            "previous_response_id and conversation are not supported: this server stores no \
             responses, so send the whole conversation in input",
        ));
    }
    if request.background == Some(true) {
        return Err(ApiError::bad_request(
            "background is not supported: this server stores no responses",
        ));
    }
    request::refuse(&request.other)?;
    request::response_format(request.text.as_ref().and_then(|t| t.get("format")))?;
    let stops = request::stops(request.stop.as_ref())?;
    let sampling = request::sampling(&request.sampling, &app.sampling)?;
    let (rendered, parser, calls) = render(&app.vocab, request)?;
    let max_tokens = openai::max_tokens(
        app.engine.max_context(),
        rendered.ids.len(),
        request.max_output_tokens,
    )?;
    let encrypted = request
        .include
        .iter()
        .flatten()
        .any(|v| v == ENCRYPTED_REASONING);
    let prepared = Prepared {
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
    };
    // Codex shows a reasoning item's summary unless its user sets show_raw_agent_reasoning,
    // so a request that asks for a summary gets the whole text there too.
    let summary = request
        .reasoning
        .as_ref()
        .and_then(|r| r.get("summary"))
        .is_some_and(Value::is_string);
    Ok((prepared, Reply::new(app, encrypted, summary)))
}

/// The prompt of a request, the parser for its output, and the tool calls that end it.
fn render(
    vocab: &Vocab,
    request: &ResponsesRequest,
) -> Result<(Rendered, Parser, Calls), ApiError> {
    let images = vocab.vision().is_some();
    let messages = translate(
        request.instructions.as_ref(),
        request.input.as_ref(),
        images,
    )?;
    let mut tools = tools(request.tools.as_deref().unwrap_or_default());
    let choice = ToolChoice::parse(request.tool_choice.as_ref())?;
    if choice == ToolChoice::None {
        tools.clear();
    }
    let parallel = request.parallel_tool_calls != Some(false);
    let effort = request.reasoning.as_ref().and_then(|r| r.get("effort"));
    let thinking = chat::request_thinking(effort, None)?;
    let ending = thinking.ending(Ending::Generate, &tools);
    let (ending, calls) = tool_ending(&choice, &tools, parallel, ending)?;
    let prompt = vocab.render_prompt(&messages, &tools, thinking.effort(), &ending)?;
    Ok((prompt, Parser::new(vocab, &ending), calls))
}

/// The function tools among `tools`, in the Responses API's flat shape, which the chat
/// template reads as it reads chat tools. Other types are left out.
fn tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
        .cloned()
        .collect()
}

fn invalid(message: &str) -> InvalidRequest {
    InvalidRequest(message.to_owned())
}

/// `instructions` and `input` as chat messages, with images kept if `images` (`--vision`).
fn translate(
    instructions: Option<&Value>,
    input: Option<&Value>,
    images: bool,
) -> Result<Vec<Value>, InvalidRequest> {
    let mut out = Vec::new();
    match instructions {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) if text.is_empty() => {}
        Some(Value::String(text)) => out.push(json!({"role": "system", "content": text})),
        Some(Value::Array(list)) => items(list, &mut out, images)?,
        Some(_) => return Err(invalid("instructions must be a string")),
    }
    match input {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) => out.push(json!({"role": "user", "content": text})),
        Some(Value::Array(list)) => items(list, &mut out, images)?,
        Some(_) => return Err(invalid("input must be a string or an array of items")),
    }
    Ok(out)
}

/// One model turn: the reasoning, assistant messages and function calls that follow each
/// other, which become one assistant message, as the model wrote them in one turn.
#[derive(Default)]
struct Turn {
    reasoning: Vec<String>,
    texts: Vec<Value>,
    calls: Vec<Value>,
}

impl Turn {
    /// The assistant message: the template renders its thinking, then its text, then its
    /// calls.
    fn message(self) -> Value {
        let mut message = json!({"role": "assistant", "content": self.texts});
        if !self.reasoning.is_empty() {
            message["reasoning_content"] = Value::String(self.reasoning.join("\n\n"));
        }
        if !self.calls.is_empty() {
            message["tool_calls"] = Value::Array(self.calls);
        }
        message
    }
}

/// Adds the chat messages of input `list`.
fn items(list: &[Value], out: &mut Vec<Value>, images: bool) -> Result<(), InvalidRequest> {
    // Ends the open turn, then adds `message`.
    fn push(message: Value, turn: &mut Option<Turn>, out: &mut Vec<Value>) {
        if let Some(turn) = turn.take() {
            out.push(turn.message());
        }
        out.push(message);
    }
    let mut turn: Option<Turn> = None;
    for item in list {
        let kind = match item.get("type").and_then(Value::as_str) {
            // An easy input message: a role and content, without a type.
            None if item.get("role").is_some() => "message",
            None => return Err(invalid("every input item needs a type")),
            Some(kind) => kind,
        };
        match kind {
            "message" => {
                let content = item.get("content").unwrap_or(&Value::Null);
                match item.get("role").and_then(Value::as_str) {
                    Some("assistant") => turn
                        .get_or_insert_default()
                        .texts
                        .extend(parts(content, images)?),
                    Some(role @ ("user" | "system" | "developer")) => {
                        let content = match content {
                            Value::String(text) => Value::String(text.clone()),
                            _ => Value::Array(parts(content, images)?),
                        };
                        push(json!({"role": role, "content": content}), &mut turn, out);
                    }
                    Some(role) => {
                        return Err(InvalidRequest(format!("unknown message role: {role}")));
                    }
                    None => return Err(invalid("every message needs a string role")),
                }
            }
            "reasoning" => {
                let text = reasoning_text(item);
                if !text.is_empty() {
                    turn.get_or_insert_default().reasoning.push(text);
                }
            }
            "function_call" => {
                let field = |key| item.get(key).cloned().unwrap_or(Value::Null);
                turn.get_or_insert_default().calls.push(json!({
                    "id": field("call_id"),
                    "type": "function",
                    "function": {"name": field("name"), "arguments": field("arguments")},
                }));
            }
            "function_call_output" => {
                let content = match item.get("output") {
                    Some(Value::String(text)) => Value::String(text.clone()),
                    Some(output @ Value::Array(_)) => Value::Array(parts(output, images)?),
                    _ => Value::String(String::new()),
                };
                let id = item.get("call_id").cloned().unwrap_or(Value::Null);
                let message = json!({"role": "tool", "tool_call_id": id, "content": content});
                push(message, &mut turn, out);
            }
            "item_reference" => {
                return Err(invalid(
                    "item_reference is not supported: this server stores no responses",
                ));
            }
            // Hosted and custom tool calls and their outputs, and compaction items: the
            // server declares no such tools, so the model made none of them.
            _ => {}
        }
    }
    if let Some(turn) = turn {
        out.push(turn.message());
    }
    Ok(())
}

/// The chat content parts of a message's content: text parts as text, an image as itself
/// if `images` or else a note, a file as a note. Other parts are left out.
fn parts(content: &Value, images: bool) -> Result<Vec<Value>, InvalidRequest> {
    let text = |text: &str| json!({"type": "text", "text": text});
    let list = match content {
        Value::String(s) => return Ok(vec![text(s)]),
        Value::Array(list) => list,
        Value::Null => return Ok(Vec::new()),
        _ => return Err(invalid("content must be a string or an array of parts")),
    };
    Ok(list
        .iter()
        .filter_map(|part| match part.get("type").and_then(Value::as_str) {
            Some("input_text" | "output_text" | "text") => {
                Some(text(part.get("text").and_then(Value::as_str).unwrap_or("")))
            }
            Some("refusal") => Some(text(
                part.get("refusal").and_then(Value::as_str).unwrap_or(""),
            )),
            Some("input_image") if images => Some(part.clone()),
            Some("input_image") => Some(text(NO_IMAGE)),
            Some("input_file") => Some(text(NO_DOCUMENT)),
            _ => None,
        })
        .collect())
}

/// The thinking text of a reasoning item: its content's text, else its `encrypted_content`
/// as this server encodes it, else its summary's text.
fn reasoning_text(item: &Value) -> String {
    let texts = |key: &str| {
        item.get(key)
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            })
            .unwrap_or_default()
    };
    let content = texts("content");
    if !content.is_empty() {
        return content;
    }
    item.get("encrypted_content")
        .and_then(Value::as_str)
        .and_then(|data| BASE64.decode(data).ok())
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .unwrap_or_else(|| texts("summary"))
}

/// The event `kind` with `data`. [`sse::Reply::number`] numbers it when it is sent.
fn event(kind: &'static str, mut data: Value) -> Frame {
    data["type"] = json!(kind);
    (kind, data)
}

/// The object `data` with the fields of `fields` added.
fn with(mut data: Value, fields: Value) -> Value {
    if let (Value::Object(data), Value::Object(fields)) = (&mut data, fields) {
        data.extend(fields);
    }
    data
}

/// The kind of the open output item, the last one, which grows with each delta.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Open {
    Reasoning,
    Text,
    /// A function call, whose arguments grow.
    Call,
}

/// One response: its output items, built from parsed deltas, and the events that stream
/// them, numbered in order.
pub(crate) struct Reply {
    id: String,
    /// The id without its prefix, for item ids.
    base: String,
    model: String,
    created_at: u64,
    /// Whether the reasoning items carry their text in `encrypted_content` too.
    encrypted: bool,
    /// Whether the reasoning items carry their text as a summary too.
    summary: bool,
    output: Vec<Value>,
    open: Option<Open>,
    /// The open item's text so far.
    text: String,
    sequence: u64,
}

impl Reply {
    fn new(app: &App, encrypted: bool, summary: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let created_at = openai::now();
        let base = format!("{created_at:x}{n:06x}");
        Self::with_id(base, app.name.clone(), created_at, encrypted, summary)
    }

    fn with_id(
        base: String,
        model: String,
        created_at: u64,
        encrypted: bool,
        summary: bool,
    ) -> Self {
        Self {
            id: format!("resp_{base}"),
            base,
            model,
            created_at,
            encrypted,
            summary,
            output: Vec::new(),
            open: None,
            text: String::new(),
            sequence: 0,
        }
    }

    /// The response object with `status` and the output so far.
    fn response(&self, status: &str) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "error": null,
            "incomplete_details": null,
            "model": self.model,
            "output": self.output,
        })
    }

    /// The finished response, with its usage and timings.
    fn finished(
        &self,
        output: &Output,
        finish: Finish,
        prompt: &Prompt,
        timings: &Timings,
    ) -> Value {
        let incomplete = output.finish_reason(finish) == "length";
        let mut response = self.response(if incomplete {
            "incomplete"
        } else {
            "completed"
        });
        if incomplete {
            response["incomplete_details"] = json!({"reason": "max_output_tokens"});
        }
        response["usage"] = json!({
            "input_tokens": prompt.tokens,
            "input_tokens_details": {"cached_tokens": prompt.cached},
            "output_tokens": output.tokens,
            "output_tokens_details": {"reasoning_tokens": output.reasoning_tokens()},
            "total_tokens": prompt.tokens + output.tokens,
        });
        response["timings"] = timings.json();
        response
    }

    /// The id of output item `index` of `kind`.
    fn item_id(&self, prefix: &str, index: usize) -> String {
        format!("{prefix}_{}_{index}", self.base)
    }

    /// Adds `text` to the open item of `kind`, opening one if needed. Delta events for the
    /// same item at the end of `events` grow instead of new ones being added.
    fn grow(&mut self, kind: Open, text: &str, events: &mut Vec<Frame>) {
        let index = if self.open == Some(kind) {
            self.output.len() - 1
        } else {
            self.open_item(kind, events)
        };
        self.text.push_str(text);
        let deltas: &[&str] = match kind {
            Open::Reasoning if self.summary => &[
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            ],
            Open::Reasoning => &["response.reasoning_text.delta"],
            Open::Text => &["response.output_text.delta"],
            Open::Call => &["response.function_call_arguments.delta"],
        };
        let trailing = events
            .iter()
            .rev()
            .take_while(|(name, _)| deltas.contains(name))
            .count();
        if trailing == deltas.len() {
            let at = events.len() - trailing;
            for (_, event) in &mut events[at..] {
                if let Some(Value::String(sent)) = event.get_mut("delta") {
                    sent.push_str(text);
                }
            }
            return;
        }
        let item_id = &self.output[index]["id"];
        for &name in deltas {
            let payload = match name {
                "response.reasoning_summary_text.delta" => json!({
                    "type": name,
                    "item_id": item_id,
                    "output_index": index,
                    "delta": text,
                    "summary_index": 0,
                }),
                "response.output_text.delta" => json!({
                    "type": name,
                    "item_id": item_id,
                    "output_index": index,
                    "delta": text,
                    "content_index": 0,
                    "logprobs": [],
                }),
                "response.function_call_arguments.delta" => json!({
                    "type": name,
                    "item_id": item_id,
                    "output_index": index,
                    "delta": text,
                }),
                _ => json!({
                    "type": name,
                    "item_id": item_id,
                    "output_index": index,
                    "delta": text,
                    "content_index": 0,
                }),
            };
            events.push((name, payload));
        }
    }

    /// Opens a reasoning or message item of `kind` and returns its index.
    fn open_item(&mut self, kind: Open, events: &mut Vec<Frame>) -> usize {
        self.close(events);
        let index = self.output.len();
        let (item, part) = match kind {
            Open::Call => return self.open_call("", events),
            Open::Reasoning => (
                json!({"type": "reasoning", "id": self.item_id("rs", index), "summary": [],
                       "content": []}),
                json!({"type": "reasoning_text", "text": ""}),
            ),
            Open::Text => (
                json!({"type": "message", "id": self.item_id("msg", index),
                       "role": "assistant", "status": "in_progress", "content": []}),
                json!({"type": "output_text", "text": "", "annotations": []}),
            ),
        };
        let at = json!({"item_id": item["id"], "output_index": index});
        let added = json!({"output_index": index, "item": item});
        events.push(event("response.output_item.added", added));
        let fields = json!({"content_index": 0, "part": part});
        events.push(event(
            "response.content_part.added",
            with(at.clone(), fields),
        ));
        if kind == Open::Reasoning && self.summary {
            let fields = json!({"summary_index": 0, "part": {"type": "summary_text", "text": ""}});
            events.push(event(
                "response.reasoning_summary_part.added",
                with(at, fields),
            ));
        }
        self.output.push(item);
        self.open = Some(kind);
        self.text.clear();
        index
    }

    /// Closes the open item: its text's done events, then the whole item.
    fn close(&mut self, events: &mut Vec<Frame>) {
        let Some(kind) = self.open.take() else {
            return;
        };
        let index = self.output.len() - 1;
        let text = std::mem::take(&mut self.text);
        let at = json!({"item_id": self.output[index]["id"], "output_index": index});
        let reasoning = match kind {
            Open::Call => {
                let item = &mut self.output[index];
                let fields = json!({"name": item["name"], "arguments": text});
                events.push(event(
                    "response.function_call_arguments.done",
                    with(at, fields),
                ));
                item["arguments"] = json!(text);
                item["status"] = json!("completed");
                let done = json!({"output_index": index, "item": item});
                events.push(event("response.output_item.done", done));
                return;
            }
            Open::Reasoning => true,
            Open::Text => false,
        };
        let (done, part) = if reasoning {
            (
                "response.reasoning_text.done",
                json!({"type": "reasoning_text", "text": text}),
            )
        } else {
            (
                "response.output_text.done",
                json!({"type": "output_text", "text": text, "annotations": []}),
            )
        };
        let mut fields = json!({"content_index": 0, "text": text});
        if !reasoning {
            fields["logprobs"] = json!([]);
        }
        events.push(event(done, with(at.clone(), fields)));
        let fields = json!({"content_index": 0, "part": part});
        events.push(event(
            "response.content_part.done",
            with(at.clone(), fields),
        ));
        let item = &mut self.output[index];
        item["content"] = json!([part]);
        if reasoning {
            if self.encrypted {
                item["encrypted_content"] = json!(BASE64.encode(&text));
            }
            if self.summary {
                let part = json!({"type": "summary_text", "text": text});
                item["summary"] = json!([part]);
                let fields = json!({"summary_index": 0, "text": text});
                events.push(event(
                    "response.reasoning_summary_text.done",
                    with(at.clone(), fields),
                ));
                let fields = json!({"summary_index": 0, "part": part});
                events.push(event(
                    "response.reasoning_summary_part.done",
                    with(at, fields),
                ));
            }
        } else {
            item["status"] = json!("completed");
        }
        let done = json!({"output_index": index, "item": self.output[index]});
        events.push(event("response.output_item.done", done));
    }

    /// Opens a function call item for a call of `name` and returns its index.
    fn open_call(&mut self, name: &str, events: &mut Vec<Frame>) -> usize {
        self.close(events);
        let index = self.output.len();
        let item = json!({"type": "function_call", "id": self.item_id("fc", index),
                          "call_id": self.item_id("call", index), "name": name,
                          "arguments": "", "status": "in_progress"});
        let added = json!({"output_index": index, "item": item});
        events.push(event("response.output_item.added", added));
        self.output.push(item);
        self.open = Some(Open::Call);
        self.text.clear();
        index
    }
}

/// `response.created` and `response.in_progress` go out with the first output item, so
/// that a client or proxy that times the first event measures the prefill. Keep-alives
/// are SSE comments throughout.
impl sse::Reply for Reply {
    const QUIET: Duration = QUIET;

    fn id(&self) -> &str {
        &self.id
    }

    /// The response as it stood before the first token's items, which these events precede.
    fn start(&mut self, _: &Prompt) -> Vec<Frame> {
        let mut response = self.response("in_progress");
        response["output"] = json!([]);
        vec![
            event("response.created", json!({"response": response})),
            event("response.in_progress", json!({"response": response})),
        ]
    }

    fn keep_alive(&mut self) -> Frame {
        (KEEP_ALIVE, Value::Null)
    }

    fn push(&mut self, delta: &Delta, events: &mut Vec<Frame>) {
        if !delta.reasoning.is_empty() {
            self.grow(Open::Reasoning, &delta.reasoning, events);
        }
        if !delta.content.is_empty() {
            self.grow(Open::Text, &delta.content, events);
        }
        for part in &delta.tool_parts {
            match part {
                ToolPart::Start(name) => {
                    self.open_call(name, events);
                }
                ToolPart::Arguments(piece) if self.open == Some(Open::Call) => {
                    self.grow(Open::Call, piece, events);
                }
                ToolPart::Arguments(_) => {}
                ToolPart::End => self.close(events),
            }
        }
    }

    fn end(
        &mut self,
        output: &Output,
        finish: Finish,
        prompt: &Prompt,
        timings: &Timings,
    ) -> Vec<Frame> {
        let mut events = Vec::new();
        self.close(&mut events);
        let response = self.finished(output, finish, prompt, timings);
        let kind = if response["status"] == "incomplete" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        events.push(event(kind, json!({"response": response})));
        events
    }

    fn body(
        &mut self,
        output: &Output,
        finish: Finish,
        prompt: &Prompt,
        timings: &Timings,
    ) -> Value {
        self.close(&mut Vec::new());
        self.finished(output, finish, prompt, timings)
    }

    /// `response.failed`, after `response.created` if the stream had not started, since
    /// OpenAI's SDKs expect that first.
    fn failure(&mut self, message: &str, prompt: &Prompt, started: bool) -> Vec<Frame> {
        let mut events = if started {
            Vec::new()
        } else {
            sse::Reply::start(self, prompt)
        };
        let mut response = self.response("failed");
        response["error"] = json!({"code": "server_error", "message": message});
        events.push(event("response.failed", json!({"response": response})));
        events
    }

    fn failed_response(message: &str) -> Response {
        ApiError::internal(message).into_response()
    }

    /// Gives each event its `sequence_number`, in the order they are sent.
    fn number(&mut self, events: &mut [Frame]) {
        for (name, data) in events {
            if *name != KEEP_ALIVE {
                data["sequence_number"] = json!(self.sequence);
                self.sequence += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use tokio::sync::mpsc;

    use super::*;
    use crate::{
        chat::tests::vocab,
        engine::Event,
        metrics::Metrics,
        openai::Results,
        server_tests::{Server, json_body, post, text_reply},
        sse::run,
    };

    fn parse(request: Value) -> ResponsesRequest {
        serde_json::from_value(request).unwrap()
    }

    /// The prompt ids of `request`.
    fn ids(vocab: &Vocab, request: Value) -> Vec<u32> {
        render(vocab, &parse(request)).unwrap().0.ids
    }

    /// The prompt the chat route renders for `messages` and `tools` at the default effort.
    fn chat_ids(vocab: &Vocab, messages: &[Value], tools: &[Value]) -> Vec<u32> {
        vocab
            .render_prompt(messages, tools, chat::DEFAULT_EFFORT, &Ending::Generate)
            .unwrap()
            .ids
    }

    fn exec_tool() -> Value {
        json!({"type": "function", "name": "exec_command", "description": "Runs a command.",
               "strict": false, "parameters": {"type": "object",
               "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}})
    }

    /// A request shaped like Codex's second turn, with fields the server ignores.
    fn codex_request() -> Value {
        json!({
            "model": "inkling-2-mini",
            "instructions": "",
            "input": [
                {"type": "message", "id": "msg_a", "role": "developer",
                 "content": [{"type": "input_text", "text": "You are a coding agent."}]},
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "<environment_context/>"}]},
                {"role": "user", "content": "List the files."},
                {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text",
                 "text": "I will list them."}], "content": null,
                 "encrypted_content": BASE64.encode("I will list them.")},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}", "status": "completed"},
                {"type": "function_call_output", "id": "fco_1", "call_id": "call_1",
                 "output": "a.rs\nb.rs\n"},
                {"type": "reasoning", "id": "rs_2", "summary": [],
                 "encrypted_content": BASE64.encode("Two files.")},
                {"type": "message", "role": "assistant", "id": "msg_2",
                 "content": [{"type": "output_text", "text": "a.rs and b.rs.",
                              "annotations": []}]},
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "Thanks."}]},
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            ],
            "tools": [
                exec_tool(),
                {"type": "namespace", "name": "multi_agent_v1", "tools": [
                    {"type": "function", "name": "close_agent", "parameters": {}}]},
                {"type": "web_search", "external_web_access": false},
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "reasoning": {"summary": "auto"},
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "01a2185c",
            "client_metadata": {"turn_id": "t1"},
        })
    }

    /// The chat messages that [`codex_request`] stands for.
    fn codex_messages() -> Vec<Value> {
        vec![
            json!({"role": "developer", "content": [{"type": "text",
                   "text": "You are a coding agent."}]}),
            json!({"role": "user", "content": [{"type": "text",
                   "text": "<environment_context/>"}]}),
            json!({"role": "user", "content": "List the files."}),
            json!({"role": "assistant", "content": [], "reasoning_content": "I will list them.",
                   "tool_calls": [{"id": "call_1", "type": "function", "function":
                   {"name": "exec_command", "arguments": "{\"cmd\":\"ls\"}"}}]}),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "a.rs\nb.rs\n"}),
            json!({"role": "assistant", "content": [{"type": "text", "text": "a.rs and b.rs."}],
                   "reasoning_content": "Two files."}),
            json!({"role": "user", "content": [{"type": "text", "text": "Thanks."}]}),
        ]
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn translates_a_codex_request_as_the_chat_route_renders_it() {
        let vocab = vocab();
        let tools = [
            json!({"type": "function", "function": {"name": "exec_command",
                     "description": "Runs a command.", "parameters": {"type": "object",
                     "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}),
        ];
        let expected = chat_ids(&vocab, &codex_messages(), &tools);
        assert_eq!(ids(&vocab, codex_request()), expected);
        let text = vocab.decode(&expected);
        assert!(
            text.contains(
                "<|message_model|><|content_thinking|>I will list them.<|end_message|>\
                 <|message_model|>exec_command<|content_invoke_tool_json|>"
            ),
            "{text}"
        );
        assert!(text.contains("<|message_tool|>exec_command<|content_text|>a.rs"));
        assert!(!text.contains("close_agent") && !text.contains("web_search"));

        // Without encrypted_content, the summary carries the thinking back.
        let mut request = codex_request();
        for item in request["input"].as_array_mut().unwrap() {
            if item["type"] == "reasoning" {
                let text = String::from_utf8(
                    BASE64
                        .decode(item["encrypted_content"].as_str().unwrap())
                        .unwrap(),
                )
                .unwrap();
                item["summary"] = json!([{"type": "summary_text", "text": text}]);
                item.as_object_mut().unwrap().remove("encrypted_content");
            }
        }
        assert_eq!(ids(&vocab, request.clone()), expected);

        // Content's reasoning_text comes first, as Codex returns it.
        for item in request["input"].as_array_mut().unwrap() {
            if item["type"] == "reasoning" {
                item["content"] = json!([{"type": "reasoning_text",
                                          "text": item["summary"][0]["text"]}]);
                item["summary"] = json!([{"type": "summary_text", "text": "Short."}]);
                item["encrypted_content"] = json!(BASE64.encode("Stale."));
            }
        }
        assert_eq!(ids(&vocab, request), expected);
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn instructions_effort_tool_choice_and_images() {
        let vocab = vocab();
        let decoded = |request: Value| vocab.decode(&ids(&vocab, request));
        let base = json!({"instructions": "Be brief.", "input": "Capital of France?"});
        let text = decoded(base.clone());
        assert!(text.starts_with(
            "<|message_system|><|content_text|>Be brief.<|end_message|><|message_system|>\
             <|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|>\
             <|content_text|>Capital of France?<|end_message|><|message_model|>"
        ));
        let mut low = base.clone();
        low["reasoning"] = json!({"effort": "low"});
        assert!(decoded(low).contains("Thinking effort level: 0.2"));
        // `none` turns thinking off: without tools the prompt ends with the server's thought.
        let mut off = base.clone();
        off["reasoning"] = json!({"effort": "none"});
        assert!(decoded(off).ends_with(
            "<|message_model|><|content_thinking|>Respond directly.<|end_message|>\
             <|message_model|>"
        ));
        let mut unknown = base.clone();
        unknown["reasoning"] = json!({"effort": "ultra"});
        assert!(render(&vocab, &parse(unknown)).is_err());

        // A named function opens its call; `none` drops the tools.
        let mut named = base.clone();
        named["tools"] = json!([exec_tool()]);
        named["tool_choice"] = json!({"type": "function", "name": "exec_command"});
        let (rendered, _, calls) = render(&vocab, &parse(named.clone())).unwrap();
        assert!(
            vocab
                .decode(&rendered.ids)
                .ends_with("<|message_model|>exec_command<|content_invoke_tool_json|>")
        );
        assert_eq!(calls, Calls::One);
        named["tool_choice"] = json!("none");
        assert!(!decoded(named.clone()).contains("tool_declare"));
        named["tool_choice"] = json!("auto");
        named["parallel_tool_calls"] = json!(false);
        assert_eq!(render(&vocab, &parse(named)).unwrap().2, Calls::One);

        // Without --vision an image becomes a note, in a user message and a tool output.
        let image = json!({"type": "input_image", "image_url": "data:image/png;base64,AAAA"});
        let request = json!({"input": [
            {"role": "user", "content": [image.clone(), {"type": "input_text", "text": "What?"}]},
            {"type": "function_call", "call_id": "c", "name": "view_image", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c", "output": [image]},
        ]});
        let text = decoded(request);
        assert_eq!(text.matches(NO_IMAGE).count(), 2, "{text}");
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn image_parts_reach_the_prompt_with_vision() {
        let mut vocab = vocab();
        vocab.set_vision(Some(crate::vision::tests::vision(4)));
        let png = crate::vision::tests::png_header(80, 40);
        let image =
            json!({"type": "input_image", "image_url": format!("data:image/png;base64,{png}")});
        let request = json!({"input": [
            {"role": "user", "content": [image.clone(), {"type": "input_text", "text": "What?"}]},
        ]});
        let (rendered, _, _) = render(&vocab, &parse(request)).unwrap();
        assert_eq!(rendered.images.len(), 1);
        assert!(!vocab.decode(&rendered.ids).contains("image omitted"));
    }

    #[test]
    fn refused_and_ignored_fields() {
        let reasons = [
            json!({"previous_response_id": "resp_1"}),
            json!({"conversation": "conv_1"}),
            json!({"background": true}),
            json!({"top_logprobs": 3}),
            json!({"text": {"format": {"type": "json_schema", "schema": {}}}}),
            json!({"input": [{"type": "item_reference", "id": "msg_1"}]}),
        ];
        for request in reasons {
            let request = parse(request);
            // These are refused before the prompt is rendered, so no tokenizer is needed,
            // except item_reference, which translate refuses.
            let refused = request.previous_response_id.is_some()
                || request.conversation.is_some()
                || request.background.is_some()
                || request::refuse(&request.other).is_err()
                || request::response_format(request.text.as_ref().and_then(|t| t.get("format")))
                    .is_err()
                || translate(None, request.input.as_ref(), false).is_err();
            assert!(refused);
        }
        let ignored = parse(
            json!({"store": false, "prompt_cache_key": "k", "truncation": "auto",
                                   "metadata": {"a": "b"}, "service_tier": "auto"}),
        );
        assert!(request::refuse(&ignored.other).is_ok());
    }

    /// Ids of a reply that thinks, writes text and calls a tool, as the engine sends them.
    fn reply_ids(vocab: &Vocab) -> Vec<u32> {
        let s = vocab.special;
        let mut ids = vec![s.content_thinking];
        ids.extend(vocab.encode("Let me look."));
        ids.extend([s.end_message, s.message_model, s.content_text]);
        ids.extend(vocab.encode("Listing."));
        ids.extend([s.end_message, s.message_model]);
        ids.extend(vocab.encode("exec_command"));
        ids.push(s.content_invoke_tool_json);
        ids.extend(vocab.encode(r#"{"name":"exec_command","args":{"cmd":"ls"}}"#));
        ids.push(s.end_message);
        ids
    }

    /// A reply that sends the reasoning as a summary too, unless `raw`.
    fn test_reply(raw: bool) -> Reply {
        Reply::with_id("1".to_owned(), "m".to_owned(), 7, true, !raw)
    }

    /// Runs `events` through [`run`] for a prompt of 10 tokens and returns the frames sent.
    async fn collect(vocab: &Vocab, events: Vec<Event>, stream: bool) -> Vec<Frame> {
        collect_with(vocab, events, stream, test_reply(false)).await
    }

    /// [`collect`] with `reply`.
    async fn collect_with(
        vocab: &Vocab,
        events: Vec<Event>,
        stream: bool,
        reply: Reply,
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
            Decoder::Chat(Parser::new(vocab, &Ending::Generate)),
            Stops::new(Vec::new()),
            Vec::new(),
            Calls::Free,
        );
        let (out, mut rx) = mpsc::channel(1024);
        run(vocab, results, output, reply, 10, out, stream).await;
        let mut frames = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            frames.push(frame);
        }
        frames
    }

    fn names(frames: &[Frame]) -> Vec<&'static str> {
        frames
            .iter()
            .map(|(name, _)| if *name == KEEP_ALIVE { ":" } else { *name })
            .collect()
    }

    #[tokio::test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn streams_typed_events_in_order() {
        let vocab = vocab();
        let ids = reply_ids(&vocab);
        let n = ids.len();
        let events = vec![
            Event::Started,
            Event::Cached(4),
            Event::Tokens(ids),
            Event::Done(Finish::Stop),
        ];
        let frames = collect(&vocab, events, true).await;
        assert_eq!(
            names(&frames),
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_summary_part.added",
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (i, (name, data)) in frames.iter().enumerate() {
            assert_eq!(data["type"], *name);
            assert_eq!(data["sequence_number"], i);
        }
        assert_eq!(frames[0].1["response"]["status"], "in_progress");
        assert_eq!(frames[0].1["response"]["output"], json!([]));
        for i in [5, 6] {
            assert_eq!(frames[i].1["delta"], "Let me look.");
            assert_eq!(frames[i].1["item_id"], "rs_1_0");
        }
        assert_eq!(frames[7].1["text"], "Let me look.");
        assert_eq!(frames[14].1["delta"], "Listing.");
        assert_eq!(frames[19].1["delta"], r#"{"cmd":"ls"}"#);
        let response = &frames[22].1["response"];
        assert_eq!(response["status"], "completed");
        assert_eq!(
            response["usage"],
            json!({"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4},
                   "output_tokens": n, "output_tokens_details": {"reasoning_tokens": 4},
                   "total_tokens": 10 + n})
        );
        assert_eq!(response["timings"]["cache_n"], 4);
        assert_eq!(response["output"].as_array().unwrap().len(), 3);

        // Ids one at a time: the deltas split, the items stay the same.
        let one_by_one: Vec<Event> = reply_ids(&vocab)
            .into_iter()
            .map(|id| Event::Tokens(vec![id]))
            .chain([Event::Done(Finish::Stop)])
            .collect();
        let split = collect(&vocab, one_by_one, true).await;
        assert_eq!(
            split.last().unwrap().1["response"]["output"],
            response["output"]
        );
        let added = names(&split)
            .iter()
            .filter(|n| **n == "response.output_item.added")
            .count();
        assert_eq!(added, 3);
        // The reasoning deltas of one id at a time come in pairs, one of each kind.
        let raw: String = split
            .iter()
            .filter(|(name, _)| *name == "response.reasoning_text.delta")
            .map(|(_, data)| data["delta"].as_str().unwrap())
            .collect();
        assert_eq!(raw, "Let me look.");

        // Without a summary asked for, the text comes as reasoning_text alone.
        let events = vec![Event::Tokens(reply_ids(&vocab)), Event::Done(Finish::Stop)];
        let frames = collect_with(&vocab, events, true, test_reply(true)).await;
        assert!(!names(&frames).iter().any(|n| n.contains("summary")));
        let item = &frames.last().unwrap().1["response"]["output"][0];
        assert_eq!(item["summary"], json!([]));
        assert_eq!(
            item["content"],
            json!([{"type": "reasoning_text", "text": "Let me look."}])
        );
    }

    #[tokio::test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn whole_response_round_trips_its_reasoning() {
        let vocab = vocab();
        let events = vec![
            Event::Started,
            Event::Cached(4),
            Event::Tokens(reply_ids(&vocab)),
            Event::Done(Finish::Stop),
        ];
        let frames = collect(&vocab, events, false).await;
        assert_eq!(frames.len(), 1);
        let mut body = frames[0].1.clone();
        assert!(body.as_object_mut().unwrap().remove("timings").is_some());
        let output = body["output"].clone();
        assert_eq!(
            output,
            json!([
                {"type": "reasoning", "id": "rs_1_0",
                 "summary": [{"type": "summary_text", "text": "Let me look."}],
                 "content": [{"type": "reasoning_text", "text": "Let me look."}],
                 "encrypted_content": BASE64.encode("Let me look.")},
                {"type": "message", "id": "msg_1_1", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": "Listing.", "annotations": []}]},
                {"type": "function_call", "id": "fc_1_2", "call_id": "call_1_2",
                 "name": "exec_command", "arguments": r#"{"cmd":"ls"}"#, "status": "completed"},
            ])
        );
        assert_eq!(body["id"], "resp_1");
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["incomplete_details"], Value::Null);
        assert_eq!(body["usage"]["input_tokens_details"]["cached_tokens"], 4);

        // The next turn sends the output back with the call's result: the prompt renders
        // the thinking, text and call as the chat route renders the same turn.
        let mut input = vec![json!({"role": "user", "content": "List the files."})];
        input.extend(output.as_array().unwrap().iter().cloned());
        input.push(
            json!({"type": "function_call_output", "call_id": "call_1_2",
                          "output": "a.rs"}),
        );
        let request = json!({"input": input, "tools": [exec_tool()]});
        let messages = [
            json!({"role": "user", "content": "List the files."}),
            json!({"role": "assistant", "content": [{"type": "text", "text": "Listing."}],
                   "reasoning_content": "Let me look.",
                   "tool_calls": [{"id": "call_1_2", "type": "function", "function":
                   {"name": "exec_command", "arguments": r#"{"cmd":"ls"}"#}}]}),
            json!({"role": "tool", "tool_call_id": "call_1_2", "content": "a.rs"}),
        ];
        assert_eq!(
            ids(&vocab, request),
            chat_ids(&vocab, &messages, &[exec_tool()])
        );
    }

    #[tokio::test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn a_cut_reply_is_incomplete() {
        let vocab = vocab();
        let mut ids = vec![vocab.special.content_text];
        ids.extend(vocab.encode("Lon"));
        let events = || vec![Event::Tokens(ids.clone()), Event::Done(Finish::Length)];
        let frames = collect(&vocab, events(), false).await;
        assert_eq!(frames[0].1["status"], "incomplete");
        assert_eq!(
            frames[0].1["incomplete_details"],
            json!({"reason": "max_output_tokens"})
        );
        let frames = collect(&vocab, events(), true).await;
        let (name, last) = frames.last().unwrap();
        assert_eq!(*name, "response.incomplete");
        assert_eq!(last["response"]["status"], "incomplete");
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
            run(vocab, results, output, test_reply(false), 10, out, true),
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
    async fn nothing_but_comments_before_the_first_token() {
        let vocab = vocab();
        let s = vocab.special;
        // A 40 s prefill, a thinking marker that adds no item, then text, then 20 s without
        // bytes in the reply.
        let events = vec![
            (0, Event::Started),
            (0, Event::Cached(4)),
            (40, Event::Tokens(vec![s.content_thinking])),
            (0, Event::Tokens(vocab.encode("Plan."))),
            (20, Event::Tokens(vec![s.end_message])),
            (0, Event::Done(Finish::Stop)),
        ];
        let frames = timed(&vocab, events).await;
        let seen = names(&frames);
        assert_eq!(
            seen[..5],
            [
                ":",
                ":",
                "response.created",
                "response.in_progress",
                "response.output_item.added"
            ]
        );
        // After the start, a quiet stream sends comments too.
        assert!(seen[5..].contains(&":"), "{seen:?}");
        assert_eq!(*seen.last().unwrap(), "response.completed");

        // A failure before the first token: the response opens and fails at once.
        let events = vec![
            (0, Event::Started),
            (20, Event::Failed("the KV pool is full".to_owned())),
        ];
        let frames = timed(&vocab, events).await;
        assert_eq!(
            names(&frames),
            [
                ":",
                "response.created",
                "response.in_progress",
                "response.failed"
            ]
        );
        let response = &frames[3].1["response"];
        assert_eq!(response["status"], "failed");
        assert_eq!(
            response["error"],
            json!({"code": "server_error", "message": "the KV pool is full"})
        );

        // A failure mid-stream ends with response.failed.
        let mut ids = vec![s.content_text];
        ids.extend(vocab.encode("Hi"));
        let events = vec![
            (0, Event::Tokens(ids)),
            (0, Event::Failed("the GPU fell over".to_owned())),
        ];
        let names_mid = names(&timed(&vocab, events).await);
        assert_eq!(names_mid.first(), Some(&"response.created"));
        assert_eq!(names_mid.last(), Some(&"response.failed"));
        assert_eq!(
            names_mid
                .iter()
                .filter(|n| **n == "response.created")
                .count(),
            1
        );
    }

    async fn sse_text(response: Response<Body>) -> String {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    async fn the_route_answers_streams_and_refuses() {
        let server = Server::new(4096, |vocab, _| text_reply(vocab, "Hello.", 2));
        let body = json!({"input": "Hi", "max_output_tokens": 16, "store": false});

        let response = server.call(post("/v1/responses", &body, &[])).await;
        assert_eq!(response.status(), 200);
        assert!(response.headers().contains_key("server-timing"));
        let reply = json_body(response).await;
        assert_eq!(reply["status"], "completed");
        assert_eq!(reply["output"][0]["content"][0]["text"], "Hello.");
        assert_eq!(reply["usage"]["input_tokens_details"]["cached_tokens"], 2);
        assert_eq!(reply["timings"]["cache_n"], 2);
        let label = server.labels.lock().unwrap()[0].clone();
        assert!(label.contains(" resp_"), "{label}");

        let mut streamed = body.clone();
        streamed["stream"] = json!(true);
        let response = server
            .call(post(
                "/v1/responses",
                &streamed,
                &[("x-client-request-id", "codex-1")],
            ))
            .await;
        assert_eq!(response.headers()["x-request-id"], "codex-1");
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let text = sse_text(response).await;
        assert!(
            text.starts_with("event: response.created\ndata: {"),
            "{text}"
        );
        assert!(text.contains("event: response.completed\n"), "{text}");
        assert!(!text.contains("[DONE]"));

        // Stored state is refused with OpenAI's error shape.
        let mut chained = body.clone();
        chained["previous_response_id"] = json!("resp_1");
        let response = server.call(post("/v1/responses", &chained, &[])).await;
        assert_eq!(response.status(), 400);
        let error = json_body(response).await;
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .starts_with("previous_response_id"),
            "{error}"
        );

        // A prompt that fills the context gets OpenAI's code for it, which Codex reads to
        // compact the conversation.
        let small = Server::new(8, |vocab, _| text_reply(vocab, "Hello.", 0));
        let response = small
            .call(post(
                "/v1/responses",
                &json!({"input": "A prompt longer than eight tokens, surely."}),
                &[],
            ))
            .await;
        assert_eq!(response.status(), 400);
        assert_eq!(
            json_body(response).await["error"]["code"],
            "context_length_exceeded"
        );
    }
}
