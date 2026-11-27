//! The tokenizer routes, with the tokenizer and chat template the prompts go through.
//!
//! - `POST /tokenize` and `POST /v1/tokenize`, in vLLM's shape: `{"prompt"}` or
//!   `{"messages", "tools"?, ...}` to `{"tokens", "count", "max_model_len"}`.
//! - `POST /detokenize` and `POST /v1/detokenize`: `{"tokens"}` to `{"prompt"}`.
//! - `POST /v1/token/encode`, in TabbyAPI's shape: `{"text"}`, a string or messages, to
//!   `{"tokens", "length"}`.
//! - `POST /v1/token/decode`: `{"tokens"}` to `{"text"}`.
//!
//! Text is encoded as `/v1/completions` encodes a text prompt: a special token's spelling in
//! it is plain text, and no token is added. Messages render as a chat request renders them,
//! with `tools`, `reasoning_effort`, `chat_template_kwargs`, `add_generation_prompt` and
//! `continue_final_message`. Decoding keeps special tokens.

use serde::Deserialize;
use serde_json::{Value, json};

use crate::serve::{
    chat::{self, Vocab},
    invalid::InvalidRequest,
    request,
};

/// The fields of every encode request.
#[derive(Deserialize)]
pub struct EncodeRequest {
    /// vLLM's and SGLang's text.
    prompt: Option<String>,
    /// TabbyAPI's text, or its messages.
    text: Option<Value>,
    messages: Option<Vec<Value>>,
    tools: Option<Vec<Value>>,
    reasoning_effort: Option<Value>,
    chat_template_kwargs: Option<Value>,
    add_generation_prompt: Option<bool>,
    continue_final_message: Option<bool>,
}

#[derive(Deserialize)]
pub struct DecodeRequest {
    tokens: Vec<Value>,
}

/// The ids of a request's text or messages.
pub fn encode(vocab: &Vocab, request: EncodeRequest) -> Result<Vec<u32>, InvalidRequest> {
    let messages = match (request.prompt, request.text, request.messages) {
        (Some(text), None, None) | (None, Some(Value::String(text)), None) => {
            return Ok(vocab.encode(&text));
        }
        (None, Some(Value::Array(messages)), None) | (None, None, Some(messages)) => messages,
        _ => {
            return Err(InvalidRequest::new(
                "send exactly one of prompt (a string), text (a string or an array of \
                 messages) or messages (an array of messages)",
            ));
        }
    };
    let thinking = chat::request_thinking(
        "reasoning_effort",
        request.reasoning_effort.as_ref(),
        request.chat_template_kwargs.as_ref(),
    )?;
    let ending = request::ending(
        request.continue_final_message,
        request.add_generation_prompt,
    )?;
    let tools = request.tools.unwrap_or_default();
    vocab.render_to(
        &messages,
        &tools,
        thinking.effort(),
        &thinking.ending(ending, &tools),
    )
}

/// vLLM's `/tokenize` answer.
pub fn tokenized(ids: &[u32], max_model_len: usize) -> Value {
    json!({"tokens": ids, "count": ids.len(), "max_model_len": max_model_len})
}

/// TabbyAPI's `/v1/token/encode` answer.
pub fn encoded(ids: &[u32]) -> Value {
    json!({"tokens": ids, "length": ids.len()})
}

/// The text of a request's ids, special tokens included.
pub fn decode(vocab: &Vocab, request: &DecodeRequest) -> Result<String, InvalidRequest> {
    let ids = token_ids(vocab, "tokens", &request.tokens)?;
    Ok(vocab.decode(&ids))
}

/// Ids in the vocabulary, refusing anything else. `param` names the field in errors.
pub fn token_ids(vocab: &Vocab, param: &str, items: &[Value]) -> Result<Vec<u32>, InvalidRequest> {
    items
        .iter()
        .enumerate()
        .map(|(i, id)| {
            id.as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .filter(|&id| (id as usize) < vocab.len())
                .ok_or_else(|| {
                    InvalidRequest::expected(
                        format!("{param}[{i}]"),
                        &format!("an integer token id from 0 to {}", vocab.len() - 1),
                        id,
                    )
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::serve::chat::{Ending, tests::vocab};

    fn request(value: Value) -> EncodeRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn answer_shapes() {
        assert_eq!(
            tokenized(&[5, 6], 1024),
            json!({"tokens": [5, 6], "count": 2, "max_model_len": 1024})
        );
        assert_eq!(encoded(&[5, 6]), json!({"tokens": [5, 6], "length": 2}));
    }

    #[test]
    #[ignore = "needs the tokenizer (SIPHON_MODEL)"]
    fn encodes_text_and_messages_as_the_prompt_path_does() {
        let vocab = vocab();
        let text = "Hello <|end_message|> world";
        let ids = vocab.encode(text);
        assert_eq!(
            encode(&vocab, request(json!({"prompt": text}))).unwrap(),
            ids
        );
        assert_eq!(encode(&vocab, request(json!({"text": text}))).unwrap(), ids);
        // Special-token spellings stay text.
        assert!(!ids.contains(&vocab.special.end_message));

        let messages = json!([{"role": "user", "content": "Hi"}]);
        let tools = json!([{"type": "function", "function": {"name": "f"}}]);
        let (msgs, tool_list) = (messages.as_array().unwrap(), tools.as_array().unwrap());
        let chat = vocab.render(msgs, tool_list, 0.0).unwrap();
        // Thinking off is effort 0 with tools, as on the chat route.
        let body = json!({"messages": messages, "tools": tools,
                          "chat_template_kwargs": {"enable_thinking": false}});
        assert_eq!(encode(&vocab, request(body)).unwrap(), chat);
        let body = json!({"messages": messages, "tools": tools, "reasoning_effort": 0});
        assert_eq!(encode(&vocab, request(body)).unwrap(), chat);
        // Without tools, it ends the prompt with the thought that skips thinking.
        let off = vocab.render_to(msgs, &[], 0.0, &Ending::Answer).unwrap();
        let body = json!({"text": messages, "reasoning_effort": "none"});
        assert_eq!(encode(&vocab, request(body)).unwrap(), off);
        let body = json!({"messages": messages, "tools": tools, "reasoning_effort": "none",
                          "add_generation_prompt": false});
        assert_eq!(
            encode(&vocab, request(body)).unwrap(),
            chat[..chat.len() - 1]
        );

        assert!(encode(&vocab, request(json!({}))).is_err());
        assert!(encode(&vocab, request(json!({"prompt": "a", "messages": []}))).is_err());

        let decoded = decode(
            &vocab,
            &DecodeRequest {
                tokens: vec![json!(chat[0])],
            },
        )
        .unwrap();
        assert_eq!(decoded, "<|message_system|>");
        let bad = DecodeRequest {
            tokens: vec![json!(vocab.len())],
        };
        assert!(decode(&vocab, &bad).is_err());
    }
}
