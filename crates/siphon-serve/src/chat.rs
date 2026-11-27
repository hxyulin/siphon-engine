//! Inkling chat prompts: OpenAI-style messages to token ids.
//!
//! The rendering follows the checkpoint's `chat_template.jinja`, with two changes from
//! vLLM's Inkling renderer: the `developer` role is a system message, and a tool call's
//! `arguments` may be a JSON string, as OpenAI clients send them. Text is encoded with
//! special tokens disabled and every marker is pushed as its id, so a message that
//! contains `<|end_message|>` cannot end its block.

use std::{collections::HashMap, path::Path};

use anyhow::Context as _;
use serde_json::{Map, Value};
use tokenizers::{DecoderWrapper, Tokenizer};

use crate::{
    invalid::{InvalidRequest, one_of},
    vision::{Collector, PromptImage, Vision},
};

/// The roles a chat message may have.
const ROLES: [&str; 5] = ["system", "developer", "user", "assistant", "tool"];

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
    /// A thinking block of [`OFF_THOUGHT`], then a new model message, so the model answers
    /// without thinking. See [`Thinking::Off`].
    Answer,
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

/// Flat decoded-byte table for `ByteLevel` decoders: `bytes[offsets[id]..offsets[id + 1]]`
/// holds the exact raw bytes that `tokenizers`'s `ByteLevel::decode_chain` produces for `id`.
struct ByteTable {
    bytes: Vec<u8>,
    offsets: Vec<u32>,
}

impl ByteTable {
    fn from_tokenizer(tokenizer: &Tokenizer) -> Option<Self> {
        if !matches!(tokenizer.get_decoder(), Some(DecoderWrapper::ByteLevel(_))) {
            return None;
        }
        let inv = byte_level_inverse();
        let vocab_len = tokenizer.get_vocab_size(true);
        let max_added = tokenizer
            .get_added_tokens_decoder()
            .keys()
            .copied()
            .max()
            .map_or(0, |id| id as usize + 1);
        let len = vocab_len.max(max_added);
        let mut bytes = Vec::with_capacity(len * 6);
        let mut offsets = Vec::with_capacity(len + 1);
        let mut scratch = Vec::with_capacity(64);
        for id in 0..len {
            offsets.push(u32::try_from(bytes.len()).ok()?);
            let Ok(id_u32) = u32::try_from(id) else {
                break;
            };
            if let Some(tok) = tokenizer.id_to_token(id_u32) {
                scratch.clear();
                let mut all_mapped = true;
                for c in tok.chars() {
                    let code = c as usize;
                    if code < inv.len() && inv[code] != u16::MAX {
                        scratch.push(inv[code] as u8);
                    } else {
                        all_mapped = false;
                        break;
                    }
                }
                if all_mapped {
                    bytes.extend_from_slice(&scratch);
                } else {
                    bytes.extend_from_slice(tok.as_bytes());
                }
            }
        }
        offsets.push(u32::try_from(bytes.len()).ok()?);
        Some(Self { bytes, offsets })
    }

    #[inline]
    fn get(&self, id: u32) -> &[u8] {
        let idx = id as usize;
        let Some(&end) = self.offsets.get(idx + 1) else {
            return &[];
        };
        let start = self.offsets[idx] as usize;
        &self.bytes[start..end as usize]
    }
}

/// Inverse of GPT-2's `bytes_to_unicode` map, matching `tokenizers`'s `ByteLevel` decoder.
fn byte_level_inverse() -> [u16; 512] {
    let mut map = [u16::MAX; 512];
    for b in (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
        map[usize::from(b)] = u16::from(b);
    }
    let mut n = 0usize;
    for b in 0..=255u8 {
        if map[usize::from(b)] == u16::MAX {
            map[256 + n] = u16::from(b);
            n += 1;
        }
    }
    map
}

/// The checkpoint's tokenizer and its special tokens.
pub struct Vocab {
    tokenizer: Tokenizer,
    pub special: Special,
    /// Every special token's id, sorted.
    specials: Vec<u32>,
    /// Inclusive bounds of `specials`, for O(1) rejection of non-special ids.
    special_min: u32,
    special_max: u32,
    /// Precomputed byte table for O(1) decoding with `ByteLevel` tokenizers.
    byte_table: Option<ByteTable>,
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

/// The thought a prompt gives the model when thinking is off. The model ends its own short
/// thinking blocks at effort 0 with these words. An empty thinking block does not work: the
/// model opens a second one and thinks there.
pub const OFF_THOUGHT: &str = "Respond directly.";

/// How much a request lets the model think.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Thinking {
    /// The template's thinking effort in [0, 0.99]. Even effort 0 writes a short thinking
    /// block.
    Effort(f64),
    /// No thinking: effort 0, and a request without tools ends its prompt with a thinking
    /// block of [`OFF_THOUGHT`], so the model answers at once.
    Off,
}

impl Thinking {
    /// The effort the system block states.
    pub fn effort(self) -> f64 {
        match self {
            Self::Effort(effort) => effort,
            Self::Off => 0.0,
        }
    }

    /// `ending` for a request with `tools`: with thinking off and no tools,
    /// [`Ending::Generate`] becomes [`Ending::Answer`]. A request with tools keeps effort
    /// 0's own thinking, because the model decides on a tool call there: after the
    /// prompt's thought it answers in text instead of calling the tool.
    pub fn ending<'a>(self, ending: Ending<'a>, tools: &[Value]) -> Ending<'a> {
        match (self, ending) {
            (Self::Off, Ending::Generate) if tools.is_empty() => Ending::Answer,
            (_, ending) => ending,
        }
    }
}

impl Vocab {
    /// Loads `tokenizer.json` from the checkpoint folder `dir`.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("loading {}", path.display()))?;
        Self::from_tokenizer(tokenizer)
    }

    /// Builds a [`Vocab`] from an already-loaded HuggingFace [`Tokenizer`].
    pub fn from_tokenizer(mut tokenizer: Tokenizer) -> anyhow::Result<Self> {
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
        let special_min = specials.first().copied().unwrap_or(u32::MAX);
        let special_max = specials.last().copied().unwrap_or(0);
        let byte_table = ByteTable::from_tokenizer(&tokenizer);
        Ok(Self {
            tokenizer,
            special,
            specials,
            special_min,
            special_max,
            byte_table,
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
        let mut out = Vec::new();
        self.encode_into(text, &mut out);
        out
    }

    /// Appends the ids of `text` to `out`, avoiding a temporary `Vec` allocation.
    pub(crate) fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let encoding = self
            .tokenizer
            .encode_fast(text, false)
            .expect("byte-level encoding accepts any text");
        out.extend_from_slice(encoding.get_ids());
    }

    /// Raw bytes for one token id on byte-level tokenizers, or `None` for other decoders.
    #[inline]
    pub(crate) fn byte_token(&self, id: u32) -> Option<&[u8]> {
        self.byte_table.as_ref().map(|table| table.get(id))
    }

    /// The text of `ids`, special tokens included.
    pub fn decode(&self, ids: &[u32]) -> String {
        if let Some(table) = &self.byte_table {
            let cap: usize = ids.iter().map(|&id| table.get(id).len()).sum();
            let mut raw = Vec::with_capacity(cap);
            for &id in ids {
                raw.extend_from_slice(table.get(id));
            }
            return match String::from_utf8(raw) {
                Ok(s) => s,
                Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
            };
        }
        self.tokenizer
            .decode(ids, false)
            .expect("the ids come from this vocabulary")
    }

    /// Direct HuggingFace tokenizer decode, for tests comparing against the fast byte table.
    #[cfg(test)]
    pub(crate) fn decode_hf(&self, ids: &[u32]) -> String {
        self.tokenizer
            .decode(ids, false)
            .expect("the ids come from this vocabulary")
    }

    /// Ids in the vocabulary, special tokens included.
    pub fn len(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Whether `id` is a special token, which carries no text.
    #[inline]
    pub fn is_special(&self, id: u32) -> bool {
        id >= self.special_min
            && id <= self.special_max
            && self.specials.binary_search(&id).is_ok()
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
            let last = messages
                .last()
                .and_then(|m| m.get("role"))
                .map_or_else(|| "no message".to_owned(), crate::invalid::describe);
            return Err(InvalidRequest::field(
                "continue_final_message",
                format!("needs an assistant message last, but the last role is {last}"),
            ));
        }
        let mut ids = Vec::with_capacity(messages.len().saturating_mul(32).max(64));
        let mut header = Vec::with_capacity(8);
        // Lazily indexed tool_call_id -> function name map, avoiding O(messages^2) scans.
        let mut tool_names: HashMap<&str, &str> = HashMap::new();
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
            let at = |e: InvalidRequest| e.within(format_args!("messages[{i}]"));
            let role_value = message.get("role").unwrap_or(&Value::Null);
            let bad_role = || {
                at(InvalidRequest::expected(
                    "role",
                    &one_of(&ROLES),
                    role_value,
                ))
            };
            let role = role_value.as_str().ok_or_else(bad_role)?;
            let role_token = match role {
                "user" => s.message_user,
                "assistant" => s.message_model,
                "system" | "developer" => s.message_system,
                "tool" => s.message_tool,
                _ => return Err(bad_role()),
            };
            let assistant = role == "assistant";
            if !effort_done && role_token != s.message_system {
                block(&mut ids, &[s.message_system], s.content_text, &effort_block);
                effort_done = true;
            }
            header.clear();
            header.push(role_token);
            let name = if role == "tool" {
                non_empty(message, "name").or_else(|| {
                    let id = non_empty(message, "tool_call_id")?;
                    tool_names
                        .get(id)
                        .copied()
                        .or_else(|| tool_name(&messages[i + 1..], id))
                })
            } else if assistant {
                non_empty(message, "name")
            } else {
                None
            };
            if let Some(name) = name {
                self.encode_into(name, &mut header);
            }
            let reasoning = if assistant {
                non_empty(message, "reasoning_content").or_else(|| non_empty(message, "reasoning"))
            } else {
                None
            };
            let tool_calls = match message.get("tool_calls") {
                Some(Value::Array(calls)) if assistant => calls.as_slice(),
                None | Some(Value::Null) => &[],
                Some(Value::Array(_)) => {
                    return Err(at(InvalidRequest::field(
                        "tool_calls",
                        format!("only assistant messages make tool calls, not a {role} message"),
                    )));
                }
                Some(other) => {
                    return Err(at(InvalidRequest::expected(
                        "tool_calls",
                        "an array of tool calls",
                        other,
                    )));
                }
            };
            for call in tool_calls {
                if let Some(call_id) = non_empty(call, "id")
                    && let Some(func_name) = call
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                {
                    tool_names.entry(call_id).or_insert(func_name);
                }
            }
            if let Some(reasoning) = reasoning {
                block(&mut ids, &header, s.content_thinking, reasoning);
            }
            // An assistant's empty text is left out next to its reasoning or tool calls.
            let skip_empty = assistant && (!tool_calls.is_empty() || reasoning.is_some());
            let mut parts = parts(message, skip_empty).map_err(at)?;
            if open && !tool_calls.is_empty() {
                return Err(at(InvalidRequest::field(
                    "tool_calls",
                    "the last assistant message is left open for the reply to continue, and \
                     a message with tool calls cannot be continued",
                )));
            }
            let image = parts.iter().position(|p| matches!(p, Part::Image(_)));
            if let Some(j) = image
                && self.vision.is_some()
                && role != "user"
                && role != "tool"
            {
                return Err(at(InvalidRequest::field(
                    format!("content[{j}]"),
                    format!(
                        "images are accepted in user and tool messages, not in a {role} message"
                    ),
                )));
            }
            // An open message is an assistant's, so its parts are text.
            let last = open.then(|| match parts.pop() {
                Some(Part::Text(text)) => text,
                _ => "",
            });
            for (j, part) in parts.into_iter().enumerate() {
                match part {
                    Part::Text(text) => block(&mut ids, &header, s.content_text, text),
                    Part::Image(image) => {
                        ids.extend_from_slice(&header);
                        let position = ids.len();
                        let image_ids = images
                            .image(image, position)
                            .map_err(|e| at(e.within(format_args!("content[{j}]"))))?;
                        ids.extend(image_ids);
                    }
                }
            }
            if let Some(last) = last {
                // The last text block stays open: no end marker.
                ids.extend_from_slice(&header);
                ids.push(s.content_text);
                self.encode_into(last, &mut ids);
                continue;
            }
            for (k, call) in tool_calls.iter().enumerate() {
                self.tool_call_block(&mut ids, message, k, call)
                    .map_err(at)?;
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
        self.encode_into(text, ids);
        ids.push(self.special.end_message);
    }

    /// Appends the system block that declares `tools`.
    fn declare_tools(&self, ids: &mut Vec<u32>, tools: &[Value]) -> Result<(), InvalidRequest> {
        let specs: Vec<Value> = tools
            .iter()
            .enumerate()
            .map(|(k, tool)| tool_spec(tool).map_err(|e| e.within(format_args!("tools[{k}]"))))
            .collect::<Result<_, _>>()?;
        let mut header = vec![self.special.message_system];
        self.encode_into("tool_declare", &mut header);
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
            Ending::Answer => {
                self.block(ids, &[s.message_model], s.content_thinking, OFF_THOUGHT);
                ids.push(s.message_model);
            }
            Ending::Bare | Ending::Continue => {}
            Ending::Call(name) => {
                if !tools
                    .iter()
                    .any(|tool| tool_function_name(tool) == Some(name))
                {
                    let names: Vec<&str> = tools.iter().filter_map(tool_function_name).collect();
                    let declared = if names.is_empty() {
                        "the request declares no tools".to_owned()
                    } else {
                        format!("the request's tools are {}", one_of(&names))
                    };
                    return Err(InvalidRequest::field(
                        "tool_choice",
                        format!(
                            "names the tool {}, which is not declared; {declared}",
                            crate::invalid::quote(name)
                        ),
                    ));
                }
                ids.push(s.message_model);
                self.encode_into(name, ids);
                ids.push(s.content_invoke_tool_json);
            }
            Ending::AnyCall => {
                if tools.is_empty() {
                    return Err(InvalidRequest::field(
                        "tool_choice",
                        "requires a tool call, but the request declares no tools",
                    ));
                }
                ids.extend([s.message_model, s.content_invoke_tool_json]);
            }
        }
        Ok(())
    }

    /// Appends the block of tool call `k` that the assistant `message` made.
    fn tool_call_block(
        &self,
        ids: &mut Vec<u32>,
        message: &Value,
        k: usize,
        call: &Value,
    ) -> Result<(), InvalidRequest> {
        let (name, args) =
            tool_call(call).map_err(|e| e.within(format_args!("tool_calls[{k}]")))?;
        if let Some(own) = non_empty(message, "name")
            && own != name
        {
            return Err(InvalidRequest::field(
                "name",
                format!(
                    "{} does not match the function name {} of tool_calls[{k}]",
                    crate::invalid::quote(own),
                    crate::invalid::quote(name)
                ),
            ));
        }
        let mut header = vec![self.special.message_model];
        self.encode_into(name, &mut header);
        let mut payload = Vec::with_capacity(64);
        payload.extend_from_slice(br#"{"name":"#);
        let _ = serde_json::to_writer(&mut payload, name);
        payload.extend_from_slice(br#","args":"#);
        write_canonical_map(&args, &mut payload);
        payload.push(b'}');
        let payload = String::from_utf8(payload).expect("serde_json emits valid UTF-8");
        self.block(
            ids,
            &header,
            self.special.content_invoke_tool_json,
            &payload,
        );
        Ok(())
    }
}

/// The thinking of a chat request: its `reasoning_effort`, else
/// `chat_template_kwargs.reasoning_effort`, else off when
/// `chat_template_kwargs.enable_thinking` is false, else [`DEFAULT_EFFORT`]. The effort
/// named `none` turns thinking off; a number is the template's effort, so 0 still thinks
/// briefly. `effort_param` names the `reasoning_effort` field in errors.
pub fn request_thinking(
    effort_param: &str,
    reasoning_effort: Option<&Value>,
    chat_template_kwargs: Option<&Value>,
) -> Result<Thinking, InvalidRequest> {
    let kwarg = |key| {
        chat_template_kwargs
            .and_then(|k| k.get(key))
            .filter(|v| !v.is_null())
    };
    let (param, value) = match reasoning_effort.filter(|v| !v.is_null()) {
        Some(value) => (effort_param, Some(value)),
        None => (
            "chat_template_kwargs.reasoning_effort",
            kwarg("reasoning_effort"),
        ),
    };
    match value {
        Some(Value::String(name)) if name.trim() == "none" => Ok(Thinking::Off),
        Some(named) => effort(param, Some(named)).map(Thinking::Effort),
        None => match kwarg("enable_thinking") {
            None | Some(Value::Bool(true)) => Ok(Thinking::Effort(DEFAULT_EFFORT)),
            Some(Value::Bool(false)) => Ok(Thinking::Off),
            Some(other) => Err(InvalidRequest::expected(
                "chat_template_kwargs.enable_thinking",
                "a boolean",
                other,
            )),
        },
    }
}

/// The highest effort a number may ask for, as `chat_template.jinja` takes it.
const MAX_EFFORT: f64 = 0.99;

/// The thinking effort of a request's field `param`: a name from `chat_template.jinja` or
/// a number from 0 to [`MAX_EFFORT`], or [`DEFAULT_EFFORT`].
pub fn effort(param: &str, value: Option<&Value>) -> Result<f64, InvalidRequest> {
    let names: Vec<&str> = EFFORTS.iter().map(|&(name, _)| name).collect();
    let accepted = format!(
        "a number from 0 to {MAX_EFFORT}, or one of {}",
        one_of(&names)
    );
    let in_range = |v: f64| (0.0..=MAX_EFFORT).contains(&v);
    match value {
        None | Some(Value::Null) => Ok(DEFAULT_EFFORT),
        Some(Value::String(name)) => {
            if let Some(&(_, v)) = EFFORTS.iter().find(|(n, _)| *n == name.trim()) {
                return Ok(v);
            }
            // A number in a string: say so, unless it is out of range anyway.
            let quoted = crate::invalid::quote(name);
            Err(match name.trim().parse::<f64>() {
                Ok(v) if in_range(v) => InvalidRequest::field(
                    param,
                    format!(
                        "got {quoted} (a string); send {} as a JSON number, not a string. \
                         Accepted: {accepted}",
                        name.trim()
                    ),
                ),
                _ => InvalidRequest::field(
                    param,
                    format!("expected {accepted}, got {quoted} (a string)"),
                ),
            })
        }
        Some(number @ Value::Number(n)) => match n.as_f64() {
            Some(v) if in_range(v) => Ok(v),
            _ => Err(InvalidRequest::expected(param, &accepted, number)),
        },
        Some(other) => Err(InvalidRequest::expected(param, &accepted, other)),
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

/// Fallback scan for a tool call id that appeared after its tool message.
fn tool_name<'a>(messages: &'a [Value], id: &str) -> Option<&'a str> {
    messages
        .iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
        .filter_map(|m| m.get("tool_calls").and_then(Value::as_array))
        .flatten()
        .find(|call| call.get("id").and_then(Value::as_str) == Some(id))
        .and_then(|call| call.get("function"))
        .and_then(|func| func.get("name"))
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
        Some(Value::Array(parts)) => parts
            .iter()
            .enumerate()
            .map(|(j, part)| content_part(part).map_err(|e| e.within(format_args!("content[{j}]"))))
            .collect(),
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(other) => Err(InvalidRequest::expected(
            "content",
            "a string or an array of content parts",
            other,
        )),
    }
}

/// The content part types a chat message takes.
const PART_TYPES: [&str; 5] = ["text", "input_text", "image_url", "input_image", "image"];

/// One content part: text or an image; audio is refused.
fn content_part(part: &Value) -> Result<Part<'_>, InvalidRequest> {
    if let Value::String(text) = part {
        return Ok(Part::Text(text));
    }
    let kind = part.get("type").unwrap_or(&Value::Null);
    match kind.as_str() {
        _ if kind.is_null() => Ok(Part::Text(
            part.get("text").and_then(Value::as_str).unwrap_or(""),
        )),
        Some("text" | "input_text") => Ok(Part::Text(
            part.get("text").and_then(Value::as_str).unwrap_or(""),
        )),
        Some("image" | "input_image" | "image_url") => Ok(Part::Image(part)),
        Some(audio @ ("audio" | "input_audio" | "audio_url")) => Err(InvalidRequest::field(
            "type",
            format!("{audio:?} is not supported: Inkling 2 reads text and images, not audio"),
        )),
        _ => Err(InvalidRequest::expected("type", &one_of(&PART_TYPES), kind)),
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
    let name = tool_function_name(tool).ok_or_else(|| {
        let param = if tool.get("function").is_some() {
            "function.name"
        } else {
            "name"
        };
        match function.get("name") {
            None => InvalidRequest::missing(param, "the function's name as a string"),
            Some(other) => InvalidRequest::expected(param, "a string", other),
        }
    })?;
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
    let function = call.get("function").ok_or_else(|| {
        InvalidRequest::missing(
            "function",
            "an object with the function's name and arguments",
        )
    })?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| match function.get("name") {
            None => InvalidRequest::missing("function.name", "a string"),
            Some(other) => InvalidRequest::expected("function.name", "a string", other),
        })?;
    let args = match function.get("arguments") {
        None | Some(Value::Null) => Value::Object(Map::new()),
        Some(Value::String(s)) if s.trim().is_empty() => Value::Object(Map::new()),
        Some(Value::String(s)) => serde_json::from_str(s).map_err(|e| {
            InvalidRequest::field(
                "function.arguments",
                format!("expected a JSON object in a string, but it is not JSON: {e}"),
            )
        })?,
        Some(other) => other.clone(),
    };
    match args {
        Value::Object(args) => Ok((name, args)),
        other => Err(InvalidRequest::expected(
            "function.arguments",
            "a JSON object, or a string that holds one",
            &other,
        )),
    }
}

/// Compact JSON with every object's keys sorted, as the template's
/// `tojson(sort_keys=true, separators=(",", ":"))` writes it, streaming directly into a
/// byte buffer without cloning the JSON tree.
pub fn canonical_json(value: &Value) -> String {
    let mut out = Vec::with_capacity(128);
    write_canonical(value, &mut out);
    String::from_utf8(out).expect("serde_json emits valid UTF-8")
}

fn write_canonical(value: &Value, out: &mut Vec<u8>) {
    match value {
        Value::Object(map) => write_canonical_map(map, out),
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical(item, out);
            }
            out.push(b']');
        }
        other => {
            let _ = serde_json::to_writer(out, other);
        }
    }
}

fn write_canonical_map(map: &Map<String, Value>, out: &mut Vec<u8>) {
    out.push(b'{');
    // Check whether keys are already in sorted order (always true when `serde_json` uses
    // `BTreeMap`, i.e., without `preserve_order`), avoiding even a temporary ref slice.
    let mut sorted = true;
    let mut prev: Option<&str> = None;
    for k in map.keys() {
        if let Some(p) = prev
            && k.as_str() < p
        {
            sorted = false;
            break;
        }
        prev = Some(k.as_str());
    }
    if sorted {
        for (i, (k, v)) in map.iter().enumerate() {
            if i > 0 {
                out.push(b',');
            }
            let _ = serde_json::to_writer(&mut *out, k);
            out.push(b':');
            write_canonical(v, out);
        }
    } else {
        let mut entries: Vec<(&String, &Value)> = map.iter().collect();
        entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
        for (i, (k, v)) in entries.into_iter().enumerate() {
            if i > 0 {
                out.push(b',');
            }
            let _ = serde_json::to_writer(&mut *out, k);
            out.push(b':');
            write_canonical(v, out);
        }
    }
    out.push(b'}');
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

    /// An in-memory byte-level BPE [`Vocab`] with Inkling's special tokens, usable in unit
    /// tests and microbenchmarks without needing `SIPHON_MODEL`.
    pub(crate) fn synthetic_vocab() -> Vocab {
        use tokenizers::{
            AddedToken, Tokenizer,
            models::bpe::{BPE, Vocab as BpeVocab},
            pre_tokenizers::byte_level::ByteLevel,
        };

        let inv = byte_level_inverse();
        let mut byte_to_char = ['\0'; 256];
        for (code, &byte) in inv.iter().enumerate() {
            if byte != u16::MAX {
                byte_to_char[usize::from(byte)] = char::from_u32(code as u32).unwrap();
            }
        }
        let mut map = BpeVocab::default();
        for b in 0..=255u8 {
            map.insert(byte_to_char[usize::from(b)].to_string(), u32::from(b));
        }
        let extra = [
            "Hello", "Ġworld", "Ġthe", "Ġquick", "Ġbrown", "Ġfox", "Ġjumps", "Ġover", "Ġlazy",
            "Ġdog", "get_weather", "Paris", "name", "args", "city", "Plan", "Think",
        ];
        for (i, tok) in extra.iter().enumerate() {
            map.insert((*tok).to_owned(), 256 + i as u32);
        }
        let bpe = BPE::builder()
            .vocab_and_merges(map, Vec::new())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(bpe);
        let bl = ByteLevel::default().add_prefix_space(false);
        tokenizer.with_pre_tokenizer(Some(bl));
        tokenizer.with_decoder(Some(bl));
        let specials = [
            "<|endoftext|>",
            "<|message_user|>",
            "<|message_model|>",
            "<|message_system|>",
            "<|message_tool|>",
            "<|content_text|>",
            "<|content_thinking|>",
            "<|content_xml|>",
            "<|content_invoke_tool_json|>",
            "<|content_invoke_tool_text|>",
            "<|content_tool_error|>",
            "<|content_model_end_sampling|>",
            "<|end_message|>",
        ];
        let added: Vec<AddedToken> = specials
            .iter()
            .map(|&s| AddedToken::from(s, true))
            .collect();
        let _ = tokenizer.add_special_tokens(added);
        Vocab::from_tokenizer(tokenizer).unwrap()
    }

    #[test]
    fn byte_table_matches_hf_tokenizer_decode() {
        let vocab = synthetic_vocab();
        assert!(vocab.byte_table.is_some());
        let samples = [
            "Hello, 世界 🦀! Line 1\nLine 2\t\"quoted\" \\slash\\",
            "café naïve résumé 日本語 한국어 🚀🔥",
            "",
            "   \n\r\t  ",
        ];
        for sample in samples {
            let ids = vocab.encode(sample);
            let fast = vocab.decode(&ids);
            let hf = vocab.tokenizer.decode(&ids, false).unwrap();
            assert_eq!(fast, hf);
            assert_eq!(fast, sample);
        }
        // Special tokens and invalid raw UTF-8 byte sequences also match HF decode.
        let s = vocab.special;
        let mixed = [
            s.message_model,
            s.content_text,
            u32::from(b'A'),
            0xFF,
            u32::from(b'B'),
            0xF0,
            0x9F,
            0xA6,
            0x80,
            s.end_message,
            999_999, // unknown id
        ];
        assert_eq!(
            vocab.decode(&mixed),
            vocab.tokenizer.decode(&mixed, false).unwrap()
        );
        assert!(vocab.is_special(s.message_model));
        assert!(vocab.is_special(s.end_message));
        assert!(!vocab.is_special(u32::from(b'A')));
        assert!(!vocab.is_special(999_999));
    }

    #[test]
    fn render_prompt_indexes_many_tool_calls() {
        let vocab = synthetic_vocab();
        let mut messages = vec![json!({"role": "user", "content": "Run 100 tools."})];
        for i in 0..100 {
            let id = format!("call_{i}");
            messages.push(json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                }]
            }));
            messages.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": "sunny"
            }));
        }
        let ids = vocab.render(&messages, &[], 0.0).unwrap();
        let decoded = vocab.decode(&ids);
        assert_eq!(
            decoded.matches("<|message_tool|>get_weather<|content_text|>sunny").count(),
            100
        );
    }

    #[test]
    fn benchmark_canonical_json_vs_deep_clone() {
        fn old_canonical(value: &Value) -> String {
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

        let tools: Value = Value::Array(
            (0..20)
                .map(|i| {
                    json!({
                        "type": "function",
                        "name": format!("tool_{i}"),
                        "description": "Reads or writes files in the workspace with validation.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Absolute path"},
                                "limit": {"type": "integer", "default": 200},
                                "flags": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["path"]
                        }
                    })
                })
                .collect(),
        );
        assert_eq!(canonical_json(&tools), old_canonical(&tools));

        let iters = 500u32;
        let t0 = std::time::Instant::now();
        let mut old_bytes = 0usize;
        for _ in 0..iters {
            old_bytes += old_canonical(&tools).len();
        }
        let old_us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

        let t1 = std::time::Instant::now();
        let mut new_bytes = 0usize;
        for _ in 0..iters {
            new_bytes += canonical_json(&tools).len();
        }
        let new_us = t1.elapsed().as_secs_f64() * 1e6 / f64::from(iters);

        assert_eq!(old_bytes, new_bytes);
        eprintln!(
            "canonical_json (20 tools): old={old_us:.1} us, new={new_us:.1} us ({:.2}x)",
            old_us / new_us.max(1e-9)
        );
    }

    #[test]
    fn effort_names_numbers_and_default() {
        let effort = |value: Option<&Value>| effort("reasoning_effort", value);
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
    fn request_thinking_and_enable_thinking() {
        let off = json!({"enable_thinking": false});
        let on = json!({"enable_thinking": true});
        let thinking = |effort: Option<Value>, kwargs: &Value| {
            request_thinking("reasoning_effort", effort.as_ref(), Some(kwargs)).unwrap()
        };
        let default = Thinking::Effort(DEFAULT_EFFORT);
        assert_eq!(
            request_thinking("reasoning_effort", None, None).unwrap(),
            default
        );
        assert_eq!(thinking(None, &off), Thinking::Off);
        assert_eq!(thinking(None, &on), default);
        assert_eq!(thinking(Some(json!("none")), &on), Thinking::Off);
        assert_eq!(thinking(Some(json!(" none ")), &on), Thinking::Off);
        assert_eq!(thinking(Some(json!(0)), &off), Thinking::Effort(0.0));
        assert_eq!(thinking(Some(json!("low")), &off), Thinking::Effort(0.2));
        assert_eq!(thinking(Some(Value::Null), &off), Thinking::Off);
        let kwarg = json!({"reasoning_effort": "medium", "enable_thinking": false});
        assert_eq!(thinking(None, &kwarg), Thinking::Effort(0.7));
        let kwarg = json!({"reasoning_effort": "none"});
        assert_eq!(thinking(None, &kwarg), Thinking::Off);
        let e = request_thinking(
            "reasoning_effort",
            None,
            Some(&json!({"enable_thinking": "no"})),
        )
        .unwrap_err();
        assert_eq!(
            e.to_string(),
            "chat_template_kwargs.enable_thinking: expected a boolean, got \"no\" (a string)"
        );
        let e = request_thinking(
            "reasoning_effort",
            None,
            Some(&json!({"reasoning_effort": 2})),
        )
        .unwrap_err();
        assert_eq!(e.param(), Some("chat_template_kwargs.reasoning_effort"));
        assert!(request_thinking("reasoning_effort", Some(&json!("nothing")), None).is_err());
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
            text(alone, Ending::Answer),
            format!(
                "{head}<|message_model|><|content_thinking|>Respond directly.<|end_message|>\
                 <|message_model|>"
            )
        );
        let generate = |thinking: Thinking, tools: &[Value]| {
            matches!(thinking.ending(Ending::Generate, tools), Ending::Generate)
        };
        assert!(matches!(
            Thinking::Off.ending(Ending::Generate, &[]),
            Ending::Answer
        ));
        assert!(generate(Thinking::Off, &tools));
        assert!(generate(Thinking::Effort(0.0), &[]));
        assert!(matches!(
            Thinking::Off.ending(Ending::Continue, &[]),
            Ending::Continue
        ));
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
