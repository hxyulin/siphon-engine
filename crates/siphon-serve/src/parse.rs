//! Generated Inkling ids to OpenAI message parts.
//!
//! The model writes blocks, each opened by `<|message_model|>`, an optional name, and a
//! content-kind token, and closed by `<|end_message|>`:
//!
//! ```text
//! <|message_model|><|content_thinking|>reasoning<|end_message|>
//! <|message_model|><|content_text|>visible text<|end_message|>
//! <|message_model|>name<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>
//! ```
//!
//! Every marker is one special token, so the parser switches state on ids and decodes
//! only the text between them. This follows vLLM's Inkling parser.

use serde_json::Value;

use crate::chat::{Ending, Special, Vocab};

/// One tool call the model made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    /// The call's argument object as JSON text.
    pub arguments: String,
}

/// The text and tool calls that some ids added.
#[derive(Debug, Default)]
pub struct Delta {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// After `<|message_model|>`: the ids before the content kind are the block's name.
    Header,
    /// Inside a text block, or between blocks.
    Text,
    Thinking,
    Tool,
}

/// Parses one generation, fed a few ids at a time.
#[derive(Clone)]
pub struct Parser {
    special: Special,
    state: State,
    header: Vec<u32>,
    tool: Vec<u32>,
    tool_name: Vec<u32>,
    content: Detokenizer,
    reasoning: Detokenizer,
    reasoning_tokens: usize,
}

impl Parser {
    /// A parser for ids generated after a prompt that ends as `ending` says.
    pub fn new(vocab: &Vocab, ending: &Ending<'_>) -> Self {
        let (state, tool_name) = match ending {
            Ending::Generate | Ending::Bare => (State::Header, Vec::new()),
            Ending::AnyCall => (State::Tool, Vec::new()),
            Ending::Continue => (State::Text, Vec::new()),
            Ending::Call(name) => (State::Tool, vocab.encode(name)),
        };
        Self {
            special: vocab.special,
            state,
            header: Vec::new(),
            tool: Vec::new(),
            tool_name,
            content: Detokenizer::default(),
            reasoning: Detokenizer::default(),
            reasoning_tokens: 0,
        }
    }

    /// Ids inside thinking blocks so far.
    pub fn reasoning_tokens(&self) -> usize {
        self.reasoning_tokens
    }

    /// Parses `ids` and returns what they added.
    pub fn push(&mut self, vocab: &Vocab, ids: &[u32]) -> Delta {
        let s = self.special;
        let mut delta = Delta::default();
        for &id in ids {
            if id == s.message_model {
                self.end_tool(vocab, &mut delta);
                self.state = State::Header;
                self.header.clear();
            } else if id == s.content_text
                || id == s.content_invoke_tool_text
                || id == s.content_tool_error
            {
                self.end_tool(vocab, &mut delta);
                self.state = State::Text;
            } else if id == s.content_thinking {
                self.end_tool(vocab, &mut delta);
                self.state = State::Thinking;
            } else if id == s.content_invoke_tool_json {
                self.end_tool(vocab, &mut delta);
                self.state = State::Tool;
                self.tool.clear();
                self.tool_name = std::mem::take(&mut self.header);
            } else if id == s.end_message || id == s.content_model_end_sampling {
                self.end_tool(vocab, &mut delta);
                self.state = State::Text;
            } else if vocab.is_special(id) {
                // Other markers carry no text.
            } else {
                match self.state {
                    State::Header => self.header.push(id),
                    State::Text => delta.content += &self.content.push(vocab, id),
                    State::Thinking => {
                        self.reasoning_tokens += 1;
                        delta.reasoning += &self.reasoning.push(vocab, id);
                    }
                    State::Tool => self.tool.push(id),
                }
            }
        }
        delta
    }

    /// Ends the generation: flushes held-back bytes and an unclosed tool call.
    pub fn finish(&mut self, vocab: &Vocab) -> Delta {
        let mut delta = Delta::default();
        self.end_tool(vocab, &mut delta);
        delta.content += &self.content.flush(vocab);
        delta.reasoning += &self.reasoning.flush(vocab);
        delta
    }

    /// Closes a tool block, if one is open: a call when its payload is a JSON object with
    /// an `args` object, else its text as content.
    fn end_tool(&mut self, vocab: &Vocab, delta: &mut Delta) {
        if self.state != State::Tool {
            return;
        }
        self.state = State::Text;
        let raw = vocab.decode(&self.tool);
        let header = vocab.decode(&self.tool_name);
        match serde_json::from_str::<Value>(&raw) {
            Ok(Value::Object(mut payload)) => {
                let name = match payload.remove("name") {
                    Some(Value::String(name)) => name,
                    _ => header,
                };
                let arguments = match payload.remove("args") {
                    Some(args @ Value::Object(_)) => args.to_string(),
                    _ => "{}".to_owned(),
                };
                delta.tool_calls.push(ToolCall { name, arguments });
            }
            _ => delta.content += &raw,
        }
    }
}

/// Turns ids into text a few at a time, special tokens included, holding back ids that end
/// inside a UTF-8 character. Keeps one decoded step of context, as
/// text-generation-inference does.
#[derive(Clone, Default)]
pub struct Detokenizer {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl Detokenizer {
    pub fn push(&mut self, vocab: &Vocab, id: u32) -> String {
        self.ids.push(id);
        let before = vocab.decode(&self.ids[self.prefix..self.read]);
        let after = vocab.decode(&self.ids[self.prefix..]);
        if after.len() > before.len()
            && !after.ends_with('\u{fffd}')
            && let Some(new) = after.get(before.len()..)
        {
            self.prefix = self.read;
            self.read = self.ids.len();
            return new.to_owned();
        }
        String::new()
    }

    pub fn flush(&mut self, vocab: &Vocab) -> String {
        let before = vocab.decode(&self.ids[self.prefix..self.read]);
        let after = vocab.decode(&self.ids[self.prefix..]);
        self.prefix = self.ids.len();
        self.read = self.ids.len();
        after.get(before.len()..).unwrap_or_default().to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::tests::vocab;

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn parses_blocks_one_id_at_a_time() {
        let vocab = vocab();
        let s = vocab.special;
        let mut ids = vec![s.content_thinking];
        ids.extend(vocab.encode("Let me think."));
        ids.extend([s.end_message, s.message_model, s.content_text]);
        ids.extend(vocab.encode("Hello, 世界 🦀!"));
        ids.extend([s.end_message, s.message_model]);
        ids.extend(vocab.encode("get_weather"));
        ids.push(s.content_invoke_tool_json);
        ids.extend(vocab.encode(r#"{"name":"get_weather","args":{"city":"Paris"}}"#));
        ids.extend([s.end_message, s.content_model_end_sampling]);

        let mut parser = Parser::new(&vocab, &Ending::Generate);
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls = Vec::new();
        for &id in &ids {
            let delta = parser.push(&vocab, &[id]);
            assert!(
                !delta.content.contains('\u{fffd}'),
                "a split character leaked"
            );
            content += &delta.content;
            reasoning += &delta.reasoning;
            calls.extend(delta.tool_calls);
        }
        let rest = parser.finish(&vocab);
        assert!(rest.content.is_empty() && rest.reasoning.is_empty());
        assert_eq!(reasoning, "Let me think.");
        assert_eq!(content, "Hello, 世界 🦀!");
        assert_eq!(
            calls,
            [ToolCall {
                name: "get_weather".to_owned(),
                arguments: r#"{"city":"Paris"}"#.to_owned(),
            }]
        );
        assert_eq!(
            parser.reasoning_tokens(),
            vocab.encode("Let me think.").len()
        );
    }
}
