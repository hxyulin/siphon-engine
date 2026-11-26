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

/// One step of a streamed tool call, in the order the model wrote them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolPart {
    /// A call opens, with its name.
    Start(String),
    /// A piece of the open call's arguments. A call's pieces join to its
    /// [`ToolCall::arguments`].
    Arguments(String),
    /// The open call ends.
    End,
}

/// The text and tool calls that some ids added.
#[derive(Debug, Default)]
pub struct Delta {
    pub content: String,
    pub reasoning: String,
    /// The calls that ended, whole.
    pub tool_calls: Vec<ToolCall>,
    /// The steps of the calls that opened, grew or ended.
    pub tool_parts: Vec<ToolPart>,
}

impl Delta {
    /// Adds `part`, joining an arguments piece to one just before it.
    pub fn tool_part(&mut self, part: ToolPart) {
        if let (ToolPart::Arguments(new), Some(ToolPart::Arguments(last))) =
            (&part, self.tool_parts.last_mut())
        {
            last.push_str(new);
            return;
        }
        self.tool_parts.push(part);
    }
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

/// How far the parser has read an open tool block's payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scan {
    /// The text so far can still open `{"name":"..","args":{`.
    Head,
    /// The text opens no call that can stream; the block is parsed whole at its end.
    Whole,
    /// Inside the argument object, which streams: its nesting depth, and whether a string
    /// or an escape in one is open.
    Args {
        depth: usize,
        string: bool,
        escape: bool,
    },
    /// After the argument object.
    Tail,
}

/// Parses one generation, fed a few ids at a time.
#[derive(Clone)]
pub struct Parser {
    special: Special,
    state: State,
    header: Vec<u32>,
    tool: Vec<u32>,
    tool_name: Vec<u32>,
    /// The open tool block's text, and the part of it before its arguments.
    tool_text: Detokenizer,
    tool_head: String,
    scan: Scan,
    /// The streaming call's name and its arguments so far.
    call_name: String,
    call_arguments: String,
    content: Detokenizer,
    reasoning: Detokenizer,
    reasoning_tokens: usize,
}

impl Parser {
    /// A parser for ids generated after a prompt that ends as `ending` says.
    pub fn new(vocab: &Vocab, ending: &Ending<'_>) -> Self {
        let (state, tool_name) = match ending {
            Ending::Generate | Ending::Answer | Ending::Bare => (State::Header, Vec::new()),
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
            tool_text: Detokenizer::default(),
            tool_head: String::new(),
            scan: Scan::Head,
            call_name: String::new(),
            call_arguments: String::new(),
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
                    State::Tool => {
                        self.tool.push(id);
                        let text = self.tool_text.push(vocab, id);
                        self.scan(text, &mut delta);
                    }
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

    /// Reads `text`, the next of an open tool block's payload: opens the call once its
    /// payload reaches its argument object, then streams the object.
    fn scan(&mut self, mut text: String, delta: &mut Delta) {
        if self.scan == Scan::Head {
            self.tool_head.push_str(&text);
            match call_head(&self.tool_head) {
                Head::More => return,
                Head::Not => {
                    self.scan = Scan::Whole;
                    return;
                }
                Head::Call(name, at) => {
                    delta.tool_part(ToolPart::Start(name.clone()));
                    self.call_name = name;
                    text = self.tool_head.split_off(at);
                    self.scan = Scan::Args {
                        depth: 0,
                        string: false,
                        escape: false,
                    };
                }
            }
        }
        let Scan::Args {
            mut depth,
            mut string,
            mut escape,
        } = self.scan
        else {
            return;
        };
        let mut end = text.len();
        let mut closed = false;
        for (i, c) in text.char_indices() {
            if string {
                if escape {
                    escape = false;
                } else if c == '\\' {
                    escape = true;
                } else if c == '"' {
                    string = false;
                }
                continue;
            }
            match c {
                '"' => string = true,
                '{' | '[' => depth += 1,
                '}' | ']' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        end = i + c.len_utf8();
                        closed = true;
                        break;
                    }
                }
                _ => {}
            }
        }
        self.scan = if closed {
            Scan::Tail
        } else {
            Scan::Args {
                depth,
                string,
                escape,
            }
        };
        let piece = &text[..end];
        if !piece.is_empty() {
            self.call_arguments.push_str(piece);
            delta.tool_part(ToolPart::Arguments(piece.to_owned()));
        }
    }

    /// Closes a tool block, if one is open. A call that streamed ends with the arguments it
    /// streamed. Otherwise the payload is parsed whole: a call when it is a JSON object with
    /// an `args` object, else its text as content.
    fn end_tool(&mut self, vocab: &Vocab, delta: &mut Delta) {
        if self.state != State::Tool {
            return;
        }
        self.state = State::Text;
        let rest = self.tool_text.flush(vocab);
        self.scan(rest, delta);
        let ids = std::mem::take(&mut self.tool);
        let scan = std::mem::replace(&mut self.scan, Scan::Head);
        self.tool_text = Detokenizer::default();
        self.tool_head.clear();
        if matches!(scan, Scan::Args { .. } | Scan::Tail) {
            delta.tool_part(ToolPart::End);
            delta.tool_calls.push(ToolCall {
                name: std::mem::take(&mut self.call_name),
                arguments: std::mem::take(&mut self.call_arguments),
            });
            return;
        }
        let raw = vocab.decode(&ids);
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
                delta.tool_part(ToolPart::Start(name.clone()));
                delta.tool_part(ToolPart::Arguments(arguments.clone()));
                delta.tool_part(ToolPart::End);
                delta.tool_calls.push(ToolCall { name, arguments });
            }
            _ => delta.content += &raw,
        }
    }
}

/// What the start of a tool payload says.
#[derive(Debug, PartialEq, Eq)]
enum Head {
    /// It can still open a call.
    More,
    /// It cannot.
    Not,
    /// It opens a call of this name, whose argument object starts at this byte.
    Call(String, usize),
}

/// Reads the start of a tool payload as `{"name":"<name>","args":{`, with any JSON
/// whitespace between the tokens.
fn call_head(text: &str) -> Head {
    match read_head(text) {
        Ok(at) => at,
        Err(head) => head,
    }
}

/// [`call_head`], with [`Head::More`] and [`Head::Not`] as errors.
fn read_head(text: &str) -> Result<Head, Head> {
    fn literal(rest: &mut &str, literal: &str) -> Result<(), Head> {
        *rest = rest.trim_start_matches([' ', '\t', '\n', '\r']);
        if let Some(after) = rest.strip_prefix(literal) {
            *rest = after;
            Ok(())
        } else if literal.starts_with(*rest) {
            Err(Head::More)
        } else {
            Err(Head::Not)
        }
    }
    let mut rest = text;
    for token in ["{", "\"name\"", ":", "\""] {
        literal(&mut rest, token)?;
    }
    let mut escape = false;
    let close = rest
        .char_indices()
        .find_map(|(i, c)| {
            let close = !escape && c == '"';
            escape = !escape && c == '\\';
            close.then_some(i)
        })
        .ok_or(Head::More)?;
    let name = serde_json::from_str::<String>(&text[text.len() - rest.len() - 1..][..close + 2])
        .map_err(|_| Head::Not)?;
    rest = &rest[close + 1..];
    for token in [",", "\"args\"", ":", "{"] {
        literal(&mut rest, token)?;
    }
    Ok(Head::Call(name, text.len() - rest.len() - 1))
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

    #[test]
    fn call_heads_open_at_the_argument_object() {
        let head = r#"{"name": "get_weather", "args": {"#;
        for end in 0..head.len() - 1 {
            assert_eq!(call_head(&head[..end]), Head::More, "{}", &head[..end]);
        }
        assert_eq!(
            call_head(head),
            Head::Call("get_weather".to_owned(), head.len() - 1)
        );
        assert_eq!(
            call_head(r#"{"name":"a\"b","args":{}}"#),
            Head::Call("a\"b".to_owned(), 22)
        );
        for not in [
            r#"{"args":{}"#,
            r#"{"name":"a","args":["#,
            "[",
            r#"{"name":1"#,
        ] {
            assert_eq!(call_head(not), Head::Not, "{not}");
        }
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn other_payloads_end_whole() {
        let vocab = vocab();
        let s = vocab.special;
        let run = |payload: &str| {
            let mut ids = vec![s.message_model];
            ids.extend(vocab.encode("f"));
            ids.push(s.content_invoke_tool_json);
            ids.extend(vocab.encode(payload));
            ids.push(s.end_message);
            let mut parser = Parser::new(&vocab, &Ending::Generate);
            let mut all = Delta::default();
            for &id in &ids {
                let delta = parser.push(&vocab, &[id]);
                all.content += &delta.content;
                all.tool_calls.extend(delta.tool_calls);
                for part in delta.tool_parts {
                    all.tool_part(part);
                }
            }
            all
        };
        // Arguments before the name: one call, its parts at its end.
        let all = run(r#"{"args":{"b":1,"a":2},"name":"g"}"#);
        let call = ToolCall {
            name: "g".to_owned(),
            arguments: r#"{"a":2,"b":1}"#.to_owned(),
        };
        assert_eq!(all.tool_calls, std::slice::from_ref(&call));
        assert_eq!(
            all.tool_parts,
            [
                ToolPart::Start(call.name),
                ToolPart::Arguments(call.arguments),
                ToolPart::End
            ]
        );
        // Not JSON: text.
        let all = run("no call");
        assert_eq!(all.content, "no call");
        assert!(all.tool_calls.is_empty() && all.tool_parts.is_empty());
        // Streamed arguments are kept as written, and text after them is dropped.
        let all = run(r#"{"name":"g","args":{ "a" : [1, "}"] } }"#);
        assert_eq!(all.tool_calls[0].arguments, r#"{ "a" : [1, "}"] }"#);
    }
}
