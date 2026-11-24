//! Request fields shared by the routes: the ones the server refuses, stop strings,
//! `tool_choice`, and how a chat prompt ends.
//!
//! Fields not named here or in a route's request type are ignored, as vLLM ignores them.

use serde_json::{Map, Value};

use crate::chat::{Ending, InvalidRequest};

/// Fields whose meaning the server cannot honour without engine work, refused when set to
/// anything but their default (null, false, 0 or empty).
const REFUSED: &[(&str, &str)] = &[
    ("logprobs", "logprobs are not supported"),
    ("top_logprobs", "logprobs are not supported"),
    ("prompt_logprobs", "logprobs are not supported"),
    ("echo", "echo is not supported"),
    ("suffix", "suffix is not supported"),
    ("logit_bias", "logit_bias is not supported"),
    ("min_tokens", "min_tokens is not supported"),
    (
        "stop_token_ids",
        "stop_token_ids is not supported; send stop strings",
    ),
    ("bad_words", "bad_words is not supported"),
    ("allowed_token_ids", "allowed_token_ids is not supported"),
    (
        "structured_outputs",
        "constrained decoding is not supported",
    ),
    ("guided_json", "constrained decoding is not supported"),
    ("guided_regex", "constrained decoding is not supported"),
    ("guided_choice", "constrained decoding is not supported"),
    ("guided_grammar", "constrained decoding is not supported"),
    ("functions", "functions is not supported; send tools"),
    (
        "function_call",
        "function_call is not supported; send tool_choice",
    ),
    ("audio", "audio output is not supported"),
    ("web_search_options", "web search is not supported"),
];

/// Refuses a request that sets one of the [`REFUSED`] fields among `fields`.
pub fn refuse(fields: &Map<String, Value>) -> Result<(), InvalidRequest> {
    for (name, why) in REFUSED {
        if fields.get(*name).is_some_and(is_set) {
            return Err(InvalidRequest(format!("{name}: {why}")));
        }
    }
    Ok(())
}

/// Whether a field holds anything but a default: null, false, 0 or empty.
fn is_set(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Refuses `n` or `best_of` above 1.
pub fn one_choice(name: &str, value: Option<u64>) -> Result<(), InvalidRequest> {
    if value.is_some_and(|n| n > 1) {
        return Err(InvalidRequest(format!("{name}: only 1 is supported")));
    }
    Ok(())
}

/// Accepts `response_format` only as plain text: JSON modes need constrained decoding.
pub fn response_format(value: Option<&Value>) -> Result<(), InvalidRequest> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(format) => match format.get("type").and_then(Value::as_str) {
            Some("text") => Ok(()),
            kind => Err(InvalidRequest(format!(
                "response_format {} is not supported: it needs constrained decoding",
                kind.unwrap_or("without a type")
            ))),
        },
    }
}

/// The stop strings of `stop`: a string or an array of strings, none empty.
pub fn stops(value: Option<&Value>) -> Result<Vec<String>, InvalidRequest> {
    let stops = match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| InvalidRequest("stop must hold strings".to_owned()))
            })
            .collect::<Result<_, _>>()?,
        Some(_) => {
            return Err(InvalidRequest(
                "stop must be a string or an array of strings".to_owned(),
            ));
        }
    };
    if stops.iter().any(String::is_empty) {
        return Err(InvalidRequest("stop strings must not be empty".to_owned()));
    }
    Ok(stops)
}

/// A request's `tool_choice`.
#[derive(Debug, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function(String),
}

impl ToolChoice {
    pub fn parse(value: Option<&Value>) -> Result<Self, InvalidRequest> {
        let invalid = || {
            InvalidRequest(
                "tool_choice must be auto, none, required or {\"type\": \"function\", \
                 \"function\": {\"name\": ...}}"
                    .to_owned(),
            )
        };
        match value {
            None | Some(Value::Null) => Ok(Self::Auto),
            Some(Value::String(s)) => match s.as_str() {
                "auto" => Ok(Self::Auto),
                "none" => Ok(Self::None),
                "required" => Ok(Self::Required),
                _ => Err(invalid()),
            },
            Some(choice @ Value::Object(_)) => {
                if choice.get("type").and_then(Value::as_str) != Some("function") {
                    return Err(invalid());
                }
                choice
                    .pointer("/function/name")
                    .or_else(|| choice.get("name"))
                    .and_then(Value::as_str)
                    .map(|name| Self::Function(name.to_owned()))
                    .ok_or_else(invalid)
            }
            Some(_) => Err(invalid()),
        }
    }
}

/// How a chat prompt ends, from vLLM's `continue_final_message` and
/// `add_generation_prompt` (default true).
pub fn ending(
    continue_final_message: Option<bool>,
    add_generation_prompt: Option<bool>,
) -> Result<Ending<'static>, InvalidRequest> {
    match (
        continue_final_message.unwrap_or(false),
        add_generation_prompt,
    ) {
        (true, Some(true)) => Err(InvalidRequest(
            "continue_final_message needs add_generation_prompt false".to_owned(),
        )),
        (true, _) => Ok(Ending::Continue),
        (false, Some(false)) => Ok(Ending::Bare),
        (false, _) => Ok(Ending::Generate),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn fields(value: &Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn refuses_only_set_fields() {
        let ok = fields(
            &json!({"logprobs": false, "top_logprobs": 0, "logit_bias": {},
            "echo": null, "ignore_eos": false, "temperature": 0.7, "top_k": 20, "seed": 3}),
        );
        assert!(refuse(&ok).is_ok());
        for bad in [
            json!({"logprobs": true}),
            json!({"logprobs": 100}),
            json!({"top_logprobs": 5}),
            json!({"echo": true}),
            json!({"logit_bias": {"42": -100}}),
            json!({"guided_json": {"type": "object"}}),
            json!({"functions": [{"name": "f"}]}),
        ] {
            let e = refuse(&fields(&bad)).unwrap_err();
            let name = bad.as_object().unwrap().keys().next().unwrap().clone();
            assert!(e.0.starts_with(&name), "{e} for {bad}");
        }
    }

    #[test]
    fn choices_and_formats() {
        assert!(one_choice("n", None).is_ok());
        assert!(one_choice("n", Some(1)).is_ok());
        assert!(one_choice("n", Some(2)).is_err());
        assert!(response_format(None).is_ok());
        assert!(response_format(Some(&json!({"type": "text"}))).is_ok());
        assert!(response_format(Some(&json!({"type": "json_object"}))).is_err());
        assert!(response_format(Some(&json!({"type": "json_schema", "json_schema": {}}))).is_err());
    }

    #[test]
    fn stop_strings() {
        assert_eq!(stops(None).unwrap(), Vec::<String>::new());
        assert_eq!(stops(Some(&json!("\n"))).unwrap(), ["\n"]);
        assert_eq!(stops(Some(&json!(["a", "b"]))).unwrap(), ["a", "b"]);
        assert_eq!(stops(Some(&json!([]))).unwrap(), Vec::<String>::new());
        assert!(stops(Some(&json!([""]))).is_err());
        assert!(stops(Some(&json!([1]))).is_err());
        assert!(stops(Some(&json!(1))).is_err());
    }

    #[test]
    fn tool_choices() {
        let parse = |v: Value| ToolChoice::parse(Some(&v));
        assert_eq!(ToolChoice::parse(None).unwrap(), ToolChoice::Auto);
        assert_eq!(parse(json!("none")).unwrap(), ToolChoice::None);
        assert_eq!(parse(json!("required")).unwrap(), ToolChoice::Required);
        assert_eq!(
            parse(json!({"type": "function", "function": {"name": "f"}})).unwrap(),
            ToolChoice::Function("f".to_owned())
        );
        assert_eq!(
            parse(json!({"type": "function", "name": "g"})).unwrap(),
            ToolChoice::Function("g".to_owned())
        );
        assert!(parse(json!("any")).is_err());
        assert!(parse(json!({"type": "function"})).is_err());
        assert!(parse(json!({"type": "custom", "name": "f"})).is_err());
    }

    #[test]
    fn endings() {
        assert!(matches!(ending(None, None), Ok(Ending::Generate)));
        assert!(matches!(
            ending(Some(false), Some(true)),
            Ok(Ending::Generate)
        ));
        assert!(matches!(ending(None, Some(false)), Ok(Ending::Bare)));
        assert!(matches!(
            ending(Some(true), Some(false)),
            Ok(Ending::Continue)
        ));
        assert!(matches!(ending(Some(true), None), Ok(Ending::Continue)));
        assert!(ending(Some(true), Some(true)).is_err());
    }
}
