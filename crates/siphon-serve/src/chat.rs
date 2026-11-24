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

/// The checkpoint's tokenizer and its special tokens.
pub struct Vocab {
    tokenizer: Tokenizer,
    pub special: Special,
    /// Every special token's id, sorted.
    specials: Vec<u32>,
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
        })
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
        let s = self.special;
        let mut ids = Vec::new();
        let block = |ids: &mut Vec<u32>, header: &[u32], kind: u32, text: &str| {
            self.block(ids, header, kind, text);
        };
        if !tools.is_empty() {
            let specs: Vec<Value> = tools.iter().map(tool_spec).collect::<Result<_, _>>()?;
            let mut header = vec![s.message_system];
            header.extend(self.encode("tool_declare"));
            block(
                &mut ids,
                &header,
                s.content_xml,
                &canonical_json(&Value::Array(specs)),
            );
        }
        let effort_block = format!("Thinking effort level: {}", effort_text(effort));
        let mut effort_done = false;
        for message in messages {
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
            match message.get("content") {
                Some(Value::String(text)) => {
                    if !text.is_empty()
                        || !(assistant && (!tool_calls.is_empty() || reasoning.is_some()))
                    {
                        block(&mut ids, &header, s.content_text, text);
                    }
                }
                Some(Value::Array(parts)) => {
                    for part in parts {
                        let text = content_part(part)?;
                        block(&mut ids, &header, s.content_text, text);
                    }
                }
                None | Some(Value::Null) => {}
                Some(_) => return Err(invalid("message content must be a string or an array")),
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
        ids.push(s.message_model);
        Ok(ids)
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

/// The text of one content part; images and audio are refused.
fn content_part(part: &Value) -> Result<&str, InvalidRequest> {
    if let Value::String(text) = part {
        return Ok(text);
    }
    match part.get("type").and_then(Value::as_str) {
        None | Some("text" | "input_text") => {
            Ok(part.get("text").and_then(Value::as_str).unwrap_or(""))
        }
        Some("image" | "input_image" | "image_url") => {
            Err(invalid("image input is not supported by this server"))
        }
        Some("audio" | "input_audio" | "audio_url") => {
            Err(invalid("Inkling 2 does not accept audio input"))
        }
        Some(other) => Err(invalid(format!("unsupported content part type: {other}"))),
    }
}

/// The declaration of one tool, with the template's keys.
fn tool_spec(tool: &Value) -> Result<Value, InvalidRequest> {
    let function = tool.get("function").unwrap_or(tool);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("every tool needs a function name"))?;
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
}
