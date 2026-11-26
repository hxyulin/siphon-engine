//! Inkling chat prompts: OpenAI-style messages to token ids.
//!
//! The rendering follows the checkpoint's `chat_template.jinja`, with two changes from
//! vLLM's Inkling renderer: the `developer` role is a system message, and a tool call's
//! `arguments` may be a JSON string, as OpenAI clients send them. Text is encoded with
//! special tokens disabled and every marker is pushed as its id, so a message that
//! contains `<|end_message|>` cannot end its block.

use std::path::Path;

use anyhow::Context as _;
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::vision::{Collector, PromptImage, Vision};

/// A request the server cannot serve, reported to the client as a 400.
#[derive(Debug)]
pub struct InvalidRequest(pub String);

impl std::fmt::Display for InvalidRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidRequest {}

fn invalid(message: impl Into<String>) -> InvalidRequest {
    InvalidRequest(message.into())
}

/// Ids of the special tokens that frame Inkling messages.
#[derive(Clone, Copy, Debug)]
pub struct Special {
    pub endoftext: u32,
    pub message_user: u32,
    pub message_model: u32,
    pub message_system: u32,
    pub message_tool: u32,
    pub content_text: u32,
    pub content_thinking: u32,
    pub content_xml: u32,
    pub content_invoke_tool_json: u32,
    pub content_invoke_tool_text: u32,
    pub content_tool_error: u32,
    pub content_model_end_sampling: u32,
    pub end_message: u32,
}

/// How a rendered prompt ends, and so where generation starts.
#[derive(Clone, Copy, Debug)]
pub enum Ending<'a> {
    /// A new model message, `<|message_model|>`, as `add_generation_prompt` adds.
    Generate,
    /// No generation prompt, for counting a conversation's tokens.
    Bare,
    /// The final assistant message's last text block left open, as vLLM's
    /// `continue_final_message` does.
    Continue,
    /// A model message that calls the named tool, opened up to its JSON payload.
    Call(&'a str),
    /// A model message that calls a tool, opened up to its payload without the tool's name
    /// in its header, so that the model names the tool in the payload.
    AnyCall,
}

/// The checkpoint's tokenizer and its special tokens.
pub struct Vocab {
    tokenizer: Tokenizer,
    pub special: Special,
    /// Every special token's id, sorted.
    specials: Vec<u32>,
    /// Image input, with `--vision`.
    vision: Option<Vision>,
}

/// A rendered prompt and its images, in prompt order.
#[derive(Debug)]
pub struct Rendered {
    pub ids: Vec<u32>,
    pub images: Vec<PromptImage>,
}

/// `chat_template.jinja`'s named thinking efforts.
const EFFORTS: [(&str, f64); 7] = [
    ("none", 0.0),
    ("minimal", 0.1),
    ("low", 0.2),
    ("medium", 0.7),
    ("high", 0.9),
    ("xhigh", 0.99),
    ("max", 0.99),
];

/// The thinking effort when a request names none, as in `chat_template.jinja`.
pub const DEFAULT_EFFORT: f64 = 0.9;

impl Vocab {
    /// Loads `tokenizer.json` from the checkpoint folder `dir`.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading {}", path.display()))?;
        tokenizer.set_encode_special_tokens(true);
        let id = |names: &[&str]| {
            names
                .iter()
                .find_map(|name| tokenizer.token_to_id(name))
                .with_context(|| format!("the tokenizer has no {}", names[0]))
        };
        let special = Special {
            endoftext: id(&["<|endoftext|>"])?,
            message_user: id(&["<|message_user|>"])?,
            message_model: id(&["<|message_model|>"])?,
            message_system: id(&["<|message_system|>"])?,
            message_tool: id(&["<|message_tool|>"])?,
            content_text: id(&["<|content_text|>"])?,
            content_thinking: id(&["<|content_thinking|>"])?,
            content_xml: id(&["<|content_xml|>", "<|unused_200024|>"])?,
            content_invoke_tool_json: id(&["<|content_invoke_tool_json|>"])?,
            content_invoke_tool_text: id(&["<|content_invoke_tool_text|>"])?,
            content_tool_error: id(&["<|content_tool_error|>"])?,
            content_model_end_sampling: id(&["<|content_model_end_sampling|>"])?,
            end_message: id(&["<|end_message|>"])?,
        };
        let mut specials: Vec<u32> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, token)| token.special)
            .map(|(id, _)| id)
            .collect();
        specials.sort_unstable();
        Ok(Self {
            tokenizer,
            special,
            specials,
            vision: None,
        })
    }

    /// Takes image parts with `vision`; without it they are refused.
    pub fn set_vision(&mut self, vision: Option<Vision>) {
        self.vision = vision;
    }

    pub fn vision(&self) -> Option<&Vision> {
        self.vision.as_ref()
    }

    /// The ids of `text`, with any special-token spelling in it encoded as plain text.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer
            .encode_fast(text, false)
            .expect("byte-level encoding accepts any text")
            .get_ids()
            .to_vec()
    }

    /// The text of `ids`, special tokens included.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer
            .decode(ids, false)
            .expect("the ids come from this vocabulary")
    }

    /// Ids in the vocabulary, special tokens included.
    pub fn len(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Whether `id` is a special token, which carries no text.
    pub fn is_special(&self, id: u32) -> bool {
        self.specials.binary_search(&id).is_ok()
    }

    /// The ids that end a turn.
    pub fn stop_ids(&self) -> Vec<u32> {
        vec![
            self.special.content_model_end_sampling,
            self.special.endoftext,
        ]
    }

    /// Renders `messages` and `tools` (OpenAI chat format) with the thinking effort
    /// `effort`, ending with the model's message token so generation starts a block.
    pub fn render(
        &self,
        messages: &[Value],
        tools: &[Value],
        effort: f64,
    ) -> Result<Vec<u32>, InvalidRequest> {
        self.render_to(messages, tools, effort, &Ending::Generate)
    }

    /// [`Vocab::render`] with the prompt ending as `ending` says.
    pub fn render_to(
        &self,
        messages: &[Value],
        tools: &[Value],
        effort: f64,
        ending: &Ending<'_>,
    ) -> Result<Vec<u32>, InvalidRequest> {
        Ok(self.render_prompt(messages, tools, effort, ending)?.ids)
    }

    /// [`Vocab::render_to`] that also returns the prompt's images. An image part takes
    /// its message's header, then the ids image preprocessing gives it, where the chat
    /// template puts `<|content_image|><|unused_200054|><|end_message|>`. Without
    /// `--vision`, an image part is refused.
    #[expect(
        clippy::too_many_lines,
        reason = "one pass over the messages, as the template's loop"
    )]
    pub fn render_prompt(
        &self,
        messages: &[Value],
        tools: &[Value],
        effort: f64,
        ending: &Ending<'_>,
    ) -> Result<Rendered, InvalidRequest> {
        let s = self.special;
        let mut images = Collector::new(self.vision.as_ref());
        let continuing = matches!(ending, Ending::Continue);
        if continuing
            && messages
                .last()
                .and_then(|m| m.get("role"))
                .and_then(Value::as_str)
                != Some("assistant")
        {
            return Err(invalid(
                "continue_final_message needs an assistant message last",
            ));
        }
        let mut ids = Vec::new();
        let block = |ids: &mut Vec<u32>, header: &[u32], kind: u32, text: &str| {
            self.block(ids, header, kind, text);
        };
        if !tools.is_empty() {
            self.declare_tools(&mut ids, tools)?;
        }
        let effort_block = format!("Thinking effort level: {}", effort_text(effort));
        let mut effort_done = false;
        for (i, message) in messages.iter().enumerate() {
            // The final assistant message, left open for the model to continue.
            let open = continuing && i + 1 == messages.len();
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("every message needs a string role"))?;
            let role_token = match role {
                "user" => s.message_user,
                "assistant" => s.message_model,
                "system" | "developer" => s.message_system,
                "tool" => s.message_tool,
                _ => return Err(invalid(format!("unknown message role: {role}"))),
            };
            let assistant = role == "assistant";
            if !effort_done && role_token != s.message_system {
                block(&mut ids, &[s.message_system], s.content_text, &effort_block);
                effort_done = true;
            }
            let mut header = vec![role_token];
            let name = if role == "tool" {
                non_empty(message, "name").or_else(|| tool_name(messages, message))
            } else if assistant {
                non_empty(message, "name")
            } else {
                None
            };
            if let Some(name) = name {
                header.extend(self.encode(name));
            }
            let reasoning = if assistant {
                non_empty(message, "reasoning_content").or_else(|| non_empty(message, "reasoning"))
            } else {
                None
            };
            let tool_calls = match message.get("tool_calls") {
                Some(Value::Array(calls)) if assistant => calls.as_slice(),
                None | Some(Value::Null) => &[],
                Some(_) => return Err(invalid("tool_calls must be an array on an assistant")),
            };
            if let Some(reasoning) = reasoning {
                block(&mut ids, &header, s.content_thinking, reasoning);
            }
            // An assistant's empty text is left out next to its reasoning or tool calls.
            let skip_empty = assistant && (!tool_calls.is_empty() || reasoning.is_some());
            let mut parts = parts(message, skip_empty)?;
            if open && !tool_calls.is_empty() {
                return Err(invalid(
                    "continue_final_message cannot continue a message with tool calls",
                ));
            }
            let has_image = parts.iter().any(|p| matches!(p, Part::Image(_)));
            if self.vision.is_some() && has_image && role != "user" && role != "tool" {
                return Err(invalid(
                    "Image content is only supported for user and tool messages",
                ));
            }
            // An open message is an assistant's, so its parts are text.
            let last = open.then(|| match parts.pop() {
                Some(Part::Text(text)) => text,
                _ => "",
            });
            for part in parts {
                match part {
                    Part::Text(text) => block(&mut ids, &header, s.content_text, text),
                    Part::Image(image) => {
                        ids.extend_from_slice(&header);
                        let at = ids.len();
                        ids.extend(images.image(image, at)?);
                    }
                }
            }
            if let Some(last) = last {
                // The last text block stays open: no end marker.
                ids.extend_from_slice(&header);
                ids.push(s.content_text);
                ids.extend(self.encode(last));
                continue;
            }
            for call in tool_calls {
                self.tool_call_block(&mut ids, message, call)?;
            }
            if assistant {
                ids.push(s.content_model_end_sampling);
            }
        }
        if !effort_done {
            block(&mut ids, &[s.message_system], s.content_text, &effort_block);
        }
        self.end(&mut ids, tools, ending)?;
        Ok(Rendered {
            ids,
            images: images.images,
        })
    }
}

impl Vocab {
    /// Appends one block: `header`, the content kind `kind`, `text` and the end marker.
    fn block(&self, ids: &mut Vec<u32>, header: &[u32], kind: u32, text: &str) {
        ids.extend_from_slice(header);
        ids.push(kind);
        ids.extend(self.encode(text));
        ids.push(self.special.end_message);
    }

    /// Appends the system block that declares `tools`.
    fn declare_tools(&self, ids: &mut Vec<u32>, tools: &[Value]) -> Result<(), InvalidRequest> {
        let specs: Vec<Value> = tools.iter().map(tool_spec).collect::<Result<_, _>>()?;
        let mut header = vec![self.special.message_system];
        header.extend(self.encode("tool_declare"));
        let specs = canonical_json(&Value::Array(specs));
        self.block(ids, &header, self.special.content_xml, &specs);
        Ok(())
    }

    /// Appends what `ending` adds after the messages.
    fn end(
        &self,
        ids: &mut Vec<u32>,
        tools: &[Value],
        ending: &Ending<'_>,
    ) -> Result<(), InvalidRequest> {
        let s = self.special;
        match *ending {
            Ending::Generate => ids.push(s.message_model),
            Ending::Bare | Ending::Continue => {}
            Ending::Call(name) => {
                if !tools
                    .iter()
                    .any(|tool| tool_function_name(tool) == Some(name))
                {
                    return Err(invalid(format!(
                        "tool_choice names {name}, which is not a tool"
                    )));
                }
                ids.push(s.message_model);
                ids.extend(self.encode(name));
                ids.push(s.content_invoke_tool_json);
            }
            Ending::AnyCall => {
                if tools.is_empty() {
                    return Err(invalid("tool_choice required needs tools"));
                }
                ids.extend([s.message_model, s.content_invoke_tool_json]);
            }
        }
        Ok(())
    }

    /// Appends the block of one tool call that the assistant `message` made.
    fn tool_call_block(
        &self,
        ids: &mut Vec<u32>,
        message: &Value,
        call: &Value,
    ) -> Result<(), InvalidRequest> {
        let (name, args) = tool_call(call)?;
        if let Some(own) = non_empty(message, "name")
            && own != name
        {
            return Err(invalid(format!(
                "assistant name {own} does not match its tool call {name}"
            )));
        }
        let mut header = vec![self.special.message_model];
        header.extend(self.encode(name));
        let payload = format!(
            "{{\"name\":{},\"args\":{}}}",
            canonical_json(&Value::String(name.to_owned())),
            canonical_json(&Value::Object(args))
        );
        self.block(
            ids,
            &header,
            self.special.content_invoke_tool_json,
            &payload,
        );
        Ok(())
    }
}

/// The thinking effort of a chat request: its `reasoning_effort`, else
/// `chat_template_kwargs.reasoning_effort`, else effort 0 (no thinking) when
/// `chat_template_kwargs.enable_thinking` is false, else [`DEFAULT_EFFORT`]. The checkpoint's
/// template has no `enable_thinking` switch; effort 0 is its no-thinking level.
pub fn request_effort(
    reasoning_effort: Option<&Value>,
    chat_template_kwargs: Option<&Value>,
) -> Result<f64, InvalidRequest> {
    let kwarg = |key| {
        chat_template_kwargs
            .and_then(|k| k.get(key))
            .filter(|v| !v.is_null())
    };
    match reasoning_effort
        .filter(|v| !v.is_null())
        .or_else(|| kwarg("reasoning_effort"))
    {
        Some(named) => effort(Some(named)),
        None => match kwarg("enable_thinking") {
            None | Some(Value::Bool(true)) => Ok(DEFAULT_EFFORT),
            Some(Value::Bool(false)) => Ok(0.0),
            Some(_) => Err(invalid("enable_thinking must be a boolean")),
        },
    }
}

/// The thinking effort of a request: a name from `chat_template.jinja` or a number in
/// [0, 0.99], or [`DEFAULT_EFFORT`].
pub fn effort(value: Option<&Value>) -> Result<f64, InvalidRequest> {
    let value = match value {
        None | Some(Value::Null) => return Ok(DEFAULT_EFFORT),
        Some(Value::String(name)) => EFFORTS
            .iter()
            .find(|(n, _)| *n == name.trim())
            .map(|&(_, v)| v)
            .ok_or_else(|| invalid(format!("unknown reasoning_effort: {name}")))?,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(f64::NAN),
        Some(_) => return Err(invalid("reasoning_effort must be a string or a number")),
    };
    if (0.0..=0.99).contains(&value) {
        Ok(value)
    } else {
        Err(invalid("reasoning_effort must be in [0.0, 0.99]"))
    }
}

/// The effort as the template prints it: `0` for zero, else the shortest decimal.
fn effort_text(effort: f64) -> String {
    if effort == 0.0 {
        "0".to_owned()
    } else {
        effort.to_string()
    }
}

/// A string field of `message` that is present and not empty.
fn non_empty<'a>(message: &'a Value, key: &str) -> Option<&'a str> {
    message
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// The function name of the assistant tool call that a tool message answers.
fn tool_name<'a>(messages: &'a [Value], message: &Value) -> Option<&'a str> {
    let id = non_empty(message, "tool_call_id")?;
    messages
        .iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .filter_map(|m| m.get("tool_calls").and_then(Value::as_array))
        .flatten()
        .find(|call| call.get("id").and_then(Value::as_str) == Some(id))
        .and_then(|call| call.pointer("/function/name"))
        .and_then(Value::as_str)
}

/// One block of a message's content.
enum Part<'a> {
    Text(&'a str),
    Image(&'a Value),
}

/// The parts of a message's content, a block each; an empty string is left out when
/// `skip_empty`.
fn parts(message: &Value, skip_empty: bool) -> Result<Vec<Part<'_>>, InvalidRequest> {
    match message.get("content") {
        Some(Value::String(text)) if text.is_empty() && skip_empty => Ok(Vec::new()),
        Some(Value::String(text)) => Ok(vec![Part::Text(text.as_str())]),
        Some(Value::Array(parts)) => parts.iter().map(content_part).collect(),
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(_) => Err(invalid("message content must be a string or an array")),
    }
}

/// One content part: text or an image; audio is refused.
fn content_part(part: &Value) -> Result<Part<'_>, InvalidRequest> {
    if let Value::String(text) = part {
        return Ok(Part::Text(text));
    }
    match part.get("type").and_then(Value::as_str) {
        None | Some("text" | "input_text") => Ok(Part::Text(
            part.get("text").and_then(Value::as_str).unwrap_or(""),
        )),
        Some("image" | "input_image" | "image_url") => Ok(Part::Image(part)),
        Some("audio" | "input_audio" | "audio_url") => {
            Err(invalid("Inkling 2 does not accept audio input"))
        }
        Some(other) => Err(invalid(format!("unsupported content part type: {other}"))),
    }
}

/// The function name of a declared tool.
pub fn tool_function_name(tool: &Value) -> Option<&str> {
    tool.get("function")
        .unwrap_or(tool)
        .get("name")
        .and_then(Value::as_str)
}

/// The declaration of one tool, with the template's keys.
fn tool_spec(tool: &Value) -> Result<Value, InvalidRequest> {
    let function = tool.get("function").unwrap_or(tool);
    let name =
        tool_function_name(tool).ok_or_else(|| invalid("every tool needs a function name"))?;
    let description = function
        .get("description")
        .filter(|d| !is_empty(d))
        .cloned()
        .unwrap_or_else(|| Value::String(String::new()));
    let parameters = function
        .get("parameters")
        .filter(|p| !is_empty(p))
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    let kind = tool
        .get("type")
        .filter(|t| !is_empty(t))
        .cloned()
        .unwrap_or_else(|| Value::String("function".to_owned()));
    let mut spec = Map::new();
    spec.insert("description".to_owned(), description);
    spec.insert("name".to_owned(), Value::String(name.to_owned()));
    spec.insert("parameters".to_owned(), parameters);
    spec.insert("type".to_owned(), kind);
    Ok(Value::Object(spec))
}

/// Jinja's falsy values among JSON ones.
fn is_empty(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        Value::Number(_) => false,
    }
}

/// The name and argument object of an assistant tool call.
fn tool_call(call: &Value) -> Result<(&str, Map<String, Value>), InvalidRequest> {
    let function = call
        .get("function")
        .ok_or_else(|| invalid("a tool call needs a function"))?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("tool call function name must be a string"))?;
    let args = match function.get("arguments") {
        None | Some(Value::Null) => Value::Object(Map::new()),
        Some(Value::String(s)) if s.trim().is_empty() => Value::Object(Map::new()),
        Some(Value::String(s)) => serde_json::from_str(s)
            .map_err(|e| invalid(format!("tool call arguments are not JSON: {e}")))?,
        Some(other) => other.clone(),
    };
    match args {
        Value::Object(args) => Ok((name, args)),
        _ => Err(invalid("tool call arguments must be an object")),
    }
}

/// Compact JSON with every object's keys sorted, as the template's
/// `tojson(sort_keys=true, separators=(",", ":"))` writes it.
pub fn canonical_json(value: &Value) -> String {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                Value::Object(
                    keys.into_iter()
                        .map(|k| (k.clone(), sorted(&map[k])))
                        .collect(),
                )
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    sorted(value).to_string()
}

#[cfg(test)]
pub(crate) mod tests {
    use serde_json::json;

    use super::*;

    /// The checkpoint's tokenizer, from `SIPHON_MODEL`.
    pub(crate) fn vocab() -> Vocab {
        let dir = std::env::var_os("SIPHON_MODEL").expect("set SIPHON_MODEL to the checkpoint");
        Vocab::load(Path::new(&dir)).unwrap()
    }

    #[test]
    fn effort_names_numbers_and_default() {
        assert_eq!(effort(None).unwrap(), 0.9);
        assert_eq!(effort(Some(&json!("medium"))).unwrap(), 0.7);
        assert_eq!(effort(Some(&json!(0))).unwrap(), 0.0);
        assert!(effort(Some(&json!(1.0))).is_err());
        assert!(effort(Some(&json!("huge"))).is_err());
        assert_eq!(effort_text(0.0), "0");
        assert_eq!(effort_text(0.9), "0.9");
        assert_eq!(effort_text(0.99), "0.99");
    }

    #[test]
    fn request_effort_and_enable_thinking() {
        let off = json!({"enable_thinking": false});
        let on = json!({"enable_thinking": true});
        assert_eq!(request_effort(None, None).unwrap(), DEFAULT_EFFORT);
        assert_eq!(request_effort(None, Some(&off)).unwrap(), 0.0);
        assert_eq!(request_effort(None, Some(&on)).unwrap(), DEFAULT_EFFORT);
        assert_eq!(
            request_effort(Some(&json!("none")), Some(&off)).unwrap(),
            0.0
        );
        assert_eq!(
            request_effort(Some(&json!("low")), Some(&off)).unwrap(),
            0.2
        );
        assert_eq!(request_effort(Some(&Value::Null), Some(&off)).unwrap(), 0.0);
        let kwarg = json!({"reasoning_effort": "medium", "enable_thinking": false});
        assert_eq!(request_effort(None, Some(&kwarg)).unwrap(), 0.7);
        assert!(request_effort(None, Some(&json!({"enable_thinking": "no"}))).is_err());
    }

    #[test]
    fn canonical_json_sorts_nested_keys() {
        let value = json!({"b": 1, "a": {"z": [1, {"y": 2, "x": 3}], "c": "é"}});
        assert_eq!(
            canonical_json(&value),
            r#"{"a":{"c":"é","z":[1,{"x":3,"y":2}]},"b":1}"#
        );
    }

    /// The expected strings are the checkpoint's `chat_template.jinja` rendered by
    /// Python's jinja2, plus the `<|message_model|>` that starts generation.
    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn render_matches_the_chat_template() {
        let vocab = vocab();
        let render = |messages: Value, tools: Value, effort: f64| {
            let ids = vocab
                .render(
                    messages.as_array().unwrap(),
                    tools.as_array().unwrap(),
                    effort,
                )
                .unwrap();
            (vocab.decode(&ids), ids)
        };

        let (text, ids) = render(
            json!([{"role": "system", "content": "Be brief."},
                   {"role": "user", "content": "Hi <|end_message|> there"}]),
            json!([]),
            DEFAULT_EFFORT,
        );
        assert_eq!(
            text,
            "<|message_system|><|content_text|>Be brief.<|end_message|><|message_system|>\
             <|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|>\
             <|content_text|>Hi <|end_message|> there<|end_message|><|message_model|>"
        );
        // The marker inside the user's text stays text.
        let ends = ids
            .iter()
            .filter(|&&id| id == vocab.special.end_message)
            .count();
        assert_eq!(ends, 3);

        let (text, _) = render(
            json!([{"role": "user", "content": "Hello"}]),
            json!([]),
            0.2,
        );
        assert_eq!(
            text,
            "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>\
             <|message_user|><|content_text|>Hello<|end_message|><|message_model|>"
        );

        let tools = json!([{"type": "function", "function": {
            "name": "get_weather", "description": "Current weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                "unit": {"type": "string"}}, "required": ["city"]}}}]);
        let call = json!({"id": "c1", "type": "function", "function": {
            "name": "get_weather", "arguments": "{\"unit\": \"c\", \"city\": \"Paris\"}"}});
        let (text, _) = render(
            json!([{"role": "user", "content": "Weather in Paris?"},
                   {"role": "assistant", "content": "", "reasoning_content": "Need the tool.",
                    "tool_calls": [call]},
                   {"role": "tool", "tool_call_id": "c1", "content": "18 C, sunny"},
                   {"role": "assistant", "content": "It is 18 C and sunny."},
                   {"role": "user", "content": [{"type": "text", "text": "Thanks"}, "And tomorrow?"]}]),
            tools,
            0.0,
        );
        assert_eq!(
            text,
            "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Current weather\",\
             \"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"type\":\"string\"},\
             \"unit\":{\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"},\
             \"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking \
             effort level: 0<|end_message|><|message_user|><|content_text|>Weather in Paris?\
             <|end_message|><|message_model|><|content_thinking|>Need the tool.<|end_message|>\
             <|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\
             \"args\":{\"city\":\"Paris\",\"unit\":\"c\"}}<|end_message|>\
             <|content_model_end_sampling|><|message_tool|>get_weather<|content_text|>18 C, sunny\
             <|end_message|><|message_model|><|content_text|>It is 18 C and sunny.<|end_message|>\
             <|content_model_end_sampling|><|message_user|><|content_text|>Thanks<|end_message|>\
             <|message_user|><|content_text|>And tomorrow?<|end_message|><|message_model|>"
        );
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn render_endings() {
        let vocab = vocab();
        let tools = [json!({"type": "function", "function": {"name": "get_weather"}})];
        let user = json!({"role": "user", "content": "Hi"});
        let text = |messages: &[Value], ending| {
            vocab.decode(&vocab.render_to(messages, &tools, 0.0, &ending).unwrap())
        };
        let head = "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"\",\
                    \"name\":\"get_weather\",\"parameters\":{},\"type\":\"function\"}]\
                    <|end_message|><|message_system|><|content_text|>Thinking effort level: 0\
                    <|end_message|><|message_user|><|content_text|>Hi<|end_message|>";
        let alone = std::slice::from_ref(&user);
        assert_eq!(text(alone, Ending::Bare), head);
        assert_eq!(
            text(alone, Ending::AnyCall),
            format!("{head}<|message_model|><|content_invoke_tool_json|>")
        );
        assert_eq!(
            text(alone, Ending::Call("get_weather")),
            format!("{head}<|message_model|>get_weather<|content_invoke_tool_json|>")
        );
        let partial = json!({"role": "assistant", "reasoning_content": "Hm.",
                             "content": "def f():"});
        assert_eq!(
            text(&[user.clone(), partial], Ending::Continue),
            format!(
                "{head}<|message_model|><|content_thinking|>Hm.<|end_message|>\
                 <|message_model|><|content_text|>def f():"
            )
        );
        let empty = json!({"role": "assistant", "content": null});
        assert_eq!(
            text(&[user.clone(), empty], Ending::Continue),
            format!("{head}<|message_model|><|content_text|>")
        );
        assert!(
            vocab
                .render_to(alone, &tools, 0.0, &Ending::Continue)
                .is_err()
        );
        assert!(
            vocab
                .render_to(alone, &tools, 0.0, &Ending::Call("other"))
                .is_err()
        );
    }
}
