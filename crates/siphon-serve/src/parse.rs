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
    /// Empties all fields while keeping their allocated capacities for reuse.
    pub fn clear(&mut self) {
        self.content.clear();
        self.reasoning.clear();
        self.tool_calls.clear();
        self.tool_parts.clear();
    }

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
    tool_scratch: String,
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
            tool_scratch: String::new(),
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
    #[cfg_attr(not(test), expect(dead_code, reason = "kept for tests"))]
    pub fn push(&mut self, vocab: &Vocab, ids: &[u32]) -> Delta {
        let mut delta = Delta::default();
        self.push_into(vocab, ids, &mut delta);
        delta
    }

    /// Parses `ids`, appending directly into `delta` to avoid per-token allocations.
    pub fn push_into(&mut self, vocab: &Vocab, ids: &[u32], delta: &mut Delta) {
        let s = self.special;
        for &id in ids {
            if id == s.message_model {
                self.end_tool(vocab, delta);
                self.state = State::Header;
                self.header.clear();
            } else if id == s.content_text
                || id == s.content_invoke_tool_text
                || id == s.content_tool_error
            {
                self.end_tool(vocab, delta);
                self.state = State::Text;
            } else if id == s.content_thinking {
                self.end_tool(vocab, delta);
                self.state = State::Thinking;
            } else if id == s.content_invoke_tool_json {
                self.end_tool(vocab, delta);
                self.state = State::Tool;
                self.tool_name = std::mem::take(&mut self.header);
            } else if id == s.end_message || id == s.content_model_end_sampling {
                self.end_tool(vocab, delta);
                self.state = State::Text;
            } else if vocab.is_special(id) {
                // Other markers carry no text.
            } else {
                match self.state {
                    State::Header => self.header.push(id),
                    State::Text => self.content.push_into(vocab, id, &mut delta.content),
                    State::Thinking => {
                        self.reasoning_tokens += 1;
                        self.reasoning.push_into(vocab, id, &mut delta.reasoning);
                    }
                    State::Tool => {
                        self.tool.push(id);
                        let mut scratch = std::mem::take(&mut self.tool_scratch);
                        scratch.clear();
                        self.tool_text.push_into(vocab, id, &mut scratch);
                        if !scratch.is_empty() {
                            self.scan(&scratch, delta);
                        }
                        self.tool_scratch = scratch;
                    }
                }
            }
        }
    }

    /// Ends the generation: flushes held-back bytes and an unclosed tool call.
    #[cfg_attr(not(test), expect(dead_code, reason = "kept for tests"))]
    pub fn finish(&mut self, vocab: &Vocab) -> Delta {
        let mut delta = Delta::default();
        self.finish_into(vocab, &mut delta);
        delta
    }

    /// Ends the generation, appending held-back bytes and an unclosed tool call to `delta`.
    pub fn finish_into(&mut self, vocab: &Vocab, delta: &mut Delta) {
        self.end_tool(vocab, delta);
        self.content.flush_into(vocab, &mut delta.content);
        self.reasoning.flush_into(vocab, &mut delta.reasoning);
    }

    /// Reads `text`, the next of an open tool block's payload: opens the call once its
    /// payload reaches its argument object, then streams the object.
    fn scan(&mut self, text: &str, delta: &mut Delta) {
        let owned_tail: String;
        let text = if self.scan == Scan::Head {
            self.tool_head.push_str(text);
            match call_head(&self.tool_head) {
                Head::More => return,
                Head::Not => {
                    self.scan = Scan::Whole;
                    return;
                }
                Head::Call(name, at) => {
                    delta.tool_part(ToolPart::Start(name.clone()));
                    self.call_name = name;
                    owned_tail = self.tool_head.split_off(at);
                    self.scan = Scan::Args {
                        depth: 0,
                        string: false,
                        escape: false,
                    };
                    owned_tail.as_str()
                }
            }
        } else {
            text
        };
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
        if !rest.is_empty() {
            self.scan(&rest, delta);
        }
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
/// inside a UTF-8 character.
///
/// For byte-level tokenizers, uses [`Vocab::byte_token`] for O(1) table lookup and
/// incremental UTF-8 streaming with zero per-token tokenizer calls. For other decoders,
/// keeps a bounded one-step context window as text-generation-inference does.
#[derive(Clone, Default)]
pub struct Detokenizer {
    /// Held incomplete/trailing UTF-8 bytes on the O(1) byte-level path.
    pending: Vec<u8>,
    /// Bounded token window for non-byte-level fallback decoders.
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
    /// Cached byte length of `vocab.decode(&ids[prefix..read])` on the fallback path.
    before_len: Option<usize>,
}

impl Detokenizer {
    #[cfg_attr(not(test), expect(dead_code, reason = "kept for tests"))]
    pub fn push(&mut self, vocab: &Vocab, id: u32) -> String {
        let mut out = String::new();
        self.push_into(vocab, id, &mut out);
        out
    }

    /// Decodes `id` and appends any newly completed UTF-8 text directly into `out`.
    pub fn push_into(&mut self, vocab: &Vocab, id: u32, out: &mut String) {
        if let Some(bytes) = vocab.byte_token(id) {
            if bytes.is_empty() {
                return;
            }
            if self.pending.is_empty()
                && let Ok(valid) = std::str::from_utf8(bytes)
            {
                out.push_str(valid);
                return;
            }
            self.pending.extend_from_slice(bytes);
            flush_valid_utf8(&mut self.pending, out);
            return;
        }
        self.ids.push(id);
        let before_len = if self.prefix == self.read {
            0
        } else if let Some(len) = self.before_len {
            len
        } else {
            let len = vocab.decode(&self.ids[self.prefix..self.read]).len();
            self.before_len = Some(len);
            len
        };
        let after = vocab.decode(&self.ids[self.prefix..]);
        if after.len() > before_len
            && !after.ends_with('\u{fffd}')
            && let Some(new) = after.get(before_len..)
        {
            out.push_str(new);
            let old_read = self.read;
            let len = self.ids.len();
            if old_read > 0 {
                let keep = len - old_read;
                self.ids.copy_within(old_read.., 0);
                self.ids.truncate(keep);
                self.prefix = 0;
                self.read = keep;
            } else {
                self.prefix = old_read;
                self.read = len;
            }
            self.before_len = None;
        }
    }

    pub fn flush(&mut self, vocab: &Vocab) -> String {
        let mut out = String::new();
        self.flush_into(vocab, &mut out);
        out
    }

    /// Flushes any held-back bytes directly into `out`.
    pub fn flush_into(&mut self, vocab: &Vocab, out: &mut String) {
        if !self.pending.is_empty() {
            for chunk in self.pending.utf8_chunks() {
                out.push_str(chunk.valid());
                if !chunk.invalid().is_empty() {
                    out.push('\u{fffd}');
                }
            }
            self.pending.clear();
            return;
        }
        if self.read == self.ids.len() {
            self.ids.clear();
            self.prefix = 0;
            self.read = 0;
            self.before_len = None;
            return;
        }
        let before_len = if self.prefix == self.read {
            0
        } else {
            self.before_len
                .unwrap_or_else(|| vocab.decode(&self.ids[self.prefix..self.read]).len())
        };
        let after = vocab.decode(&self.ids[self.prefix..]);
        if let Some(tail) = after.get(before_len..) {
            out.push_str(tail);
        }
        self.ids.clear();
        self.prefix = 0;
        self.read = 0;
        self.before_len = None;
    }
}

/// Emits every completed UTF-8 chunk from `pending` into `out`, holding back only a
/// trailing incomplete UTF-8 sequence (or trailing invalid byte) so no `\u{fffd}` leaks
/// across token boundaries.
fn flush_valid_utf8(pending: &mut Vec<u8>, out: &mut String) {
    let total = pending.len();
    let mut consumed = 0;
    for chunk in pending.utf8_chunks() {
        let valid = chunk.valid();
        if !valid.is_empty() {
            out.push_str(valid);
            consumed += valid.len();
        }
        let invalid = chunk.invalid();
        if invalid.is_empty() {
            break;
        }
        if consumed + invalid.len() < total {
            // Followed by more bytes: definitely invalid rather than an incomplete suffix.
            out.push('\u{fffd}');
            consumed += invalid.len();
        } else {
            break;
        }
    }
    if consumed == total {
        pending.clear();
    } else if consumed > 0 {
        let rem = total - consumed;
        pending.copy_within(consumed.., 0);
        pending.truncate(rem);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::tests::{synthetic_vocab, vocab};

    #[test]
    fn detokenizer_handles_split_utf8_and_invalid_bytes() {
        let vocab = synthetic_vocab();
        let mut detok = Detokenizer::default();
        let mut out = String::new();
        // Feed "a🦀世界b" one raw byte token at a time.
        let bytes = "a🦀世界b".as_bytes();
        for &b in bytes {
            let piece = detok.push(&vocab, u32::from(b));
            assert!(!piece.contains('\u{fffd}'), "split char leaked: {piece:?}");
            out.push_str(&piece);
        }
        out.push_str(&detok.flush(&vocab));
        assert_eq!(out, "a🦀世界b");

        // Invalid byte followed by ASCII emits replacement once followed by valid bytes;
        // trailing invalid byte is held until flush.
        let mut detok = Detokenizer::default();
        assert_eq!(detok.push(&vocab, 0xFF), "");
        assert_eq!(detok.push(&vocab, u32::from(b'x')), "\u{fffd}x");
        assert_eq!(detok.push(&vocab, 0xFE), "");
        assert_eq!(detok.flush(&vocab), "\u{fffd}");
    }

    #[test]
    fn parser_streams_synthetic_thinking_text_and_tool_call() {
        let vocab = synthetic_vocab();
        let s = vocab.special;
        let mut ids = vec![s.content_thinking];
        ids.extend(vocab.encode("Plan."));
        ids.extend([s.end_message, s.message_model, s.content_text]);
        ids.extend(vocab.encode("Hi 🦀!"));
        ids.extend([s.end_message, s.message_model]);
        ids.extend(vocab.encode("get_weather"));
        ids.push(s.content_invoke_tool_json);
        ids.extend(vocab.encode(r#"{"name":"get_weather","args":{"city":"Paris"}}"#));
        ids.extend([s.end_message, s.content_model_end_sampling]);

        let mut parser = Parser::new(&vocab, &Ending::Generate);
        let mut all = Delta::default();
        for &id in &ids {
            parser.push_into(&vocab, &[id], &mut all);
            assert!(!all.content.contains('\u{fffd}'));
        }
        parser.finish_into(&vocab, &mut all);
        assert_eq!(all.reasoning, "Plan.");
        assert_eq!(all.content, "Hi 🦀!");
        assert_eq!(
            all.tool_calls,
            [ToolCall {
                name: "get_weather".to_owned(),
                arguments: r#"{"city":"Paris"}"#.to_owned(),
            }]
        );
    }

    #[test]
    fn benchmark_detokenizer_fast_vs_hf_window() {
        let vocab = synthetic_vocab();
        let sample = vocab.encode("Hello world the quick brown fox jumps over lazy dog 🦀 世界!\n");
        let n_tokens = 4_000usize;
        let stream: Vec<u32> = (0..n_tokens).map(|i| sample[i % sample.len()]).collect();

        // Simulate old double-HF-decode per token using `vocab.tokenizer.decode` via a
        // non-fast loop over a 2-token window.
        let t0 = std::time::Instant::now();
        let mut old_ids = Vec::new();
        let mut prefix = 0usize;
        let mut read = 0usize;
        let mut old_text = String::new();
        for &id in &stream {
            old_ids.push(id);
            let before = vocab.decode_hf(&old_ids[prefix..read]);
            let after = vocab.decode_hf(&old_ids[prefix..]);
            if after.len() > before.len()
                && !after.ends_with('\u{fffd}')
                && let Some(new) = after.get(before.len()..)
            {
                prefix = read;
                read = old_ids.len();
                old_text.push_str(new);
            }
        }
        let old_ns = t0.elapsed().as_secs_f64() * 1e9 / n_tokens as f64;

        let t1 = std::time::Instant::now();
        let mut detok = Detokenizer::default();
        let mut new_text = String::new();
        for &id in &stream {
            detok.push_into(&vocab, id, &mut new_text);
        }
        detok.flush_into(&vocab, &mut new_text);
        let new_ns = t1.elapsed().as_secs_f64() * 1e9 / n_tokens as f64;

        assert_eq!(new_text, old_text);
        eprintln!(
            "detokenizer ({n_tokens} toks): old_window={old_ns:.1} ns/tok, fast_byte_table={new_ns:.1} ns/tok ({:.1}x)",
            old_ns / new_ns.max(1e-9)
        );
    }

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
