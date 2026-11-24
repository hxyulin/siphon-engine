//! Request fields shared by the routes: the ones the server refuses, stop strings,
//! sampling, `tool_choice`, and how a chat prompt ends.
//!
//! Fields not named here or in a route's request type are ignored, as vLLM ignores them.

use std::{
    collections::hash_map::RandomState,
    hash::BuildHasher as _,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::Context as _;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{
    chat::{Ending, InvalidRequest},
    engine::Sampling,
};

/// The sampling fields of a request, as vLLM names them.
#[derive(Debug, Default, Deserialize)]
pub struct SamplingFields {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub seed: Option<i64>,
}

/// The sampling a request gets for the fields it leaves out: the checkpoint's
/// `generation_config.json`, as vLLM uses it, else vLLM's own defaults (temperature 1, no
/// filters).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: f64,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
        }
    }
}

impl SamplingDefaults {
    /// Reads `generation_config.json` in `dir`, if there is one.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let path = dir.join("generation_config.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Ok(Self::default());
        };
        let config: Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let mut defaults = Self::default();
        let number = |key: &str| config.get(key).and_then(Value::as_f64);
        if let Some(t) = number("temperature") {
            defaults.temperature = t;
        }
        if let Some(p) = number("top_p") {
            defaults.top_p = p;
        }
        if let Some(k) = config.get("top_k").and_then(Value::as_i64) {
            defaults.top_k = k;
        }
        if let Some(m) = number("min_p") {
            defaults.min_p = m;
        }
        Ok(defaults)
    }
}

/// A request's sampling: its fields, the defaults for the ones it leaves out, and a random
/// seed when it sends none. Rejects values vLLM rejects.
pub fn sampling(
    fields: &SamplingFields,
    defaults: &SamplingDefaults,
) -> Result<Sampling, InvalidRequest> {
    let invalid = |message: &str| Err(InvalidRequest(message.to_owned()));
    let temperature = fields.temperature.unwrap_or(defaults.temperature);
    let top_p = fields.top_p.unwrap_or(defaults.top_p);
    let top_k = fields.top_k.unwrap_or(defaults.top_k);
    let min_p = fields.min_p.unwrap_or(defaults.min_p);
    if !(temperature.is_finite() && temperature >= 0.0) {
        return invalid("temperature must be at least 0");
    }
    if !(top_p > 0.0 && top_p <= 1.0) {
        return invalid("top_p must be in (0, 1]");
    }
    if top_k < -1 {
        return invalid("top_k must be -1 or 0 (no limit) or at least 1");
    }
    if !(0.0..=1.0).contains(&min_p) {
        return invalid("min_p must be in [0, 1]");
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "F32 settings for the kernels"
    )]
    Ok(Sampling {
        temperature: temperature as f32,
        top_k: u32::try_from(top_k.max(0)).unwrap_or(u32::MAX),
        top_p: top_p as f32,
        min_p: min_p as f32,
        seed: fields.seed.map_or_else(random_seed, i64::cast_unsigned),
    })
}

/// A seed for a request that sends none.
fn random_seed() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    RandomState::new().hash_one(COUNTER.fetch_add(1, Ordering::Relaxed))
}

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
    fn sampling_fields() {
        let defaults = SamplingDefaults::default();
        let parse = |v: Value| {
            let fields: SamplingFields = serde_json::from_value(v).unwrap();
            sampling(&fields, &defaults)
        };
        let s = parse(json!({})).unwrap();
        assert_eq!(
            (s.temperature, s.top_k, s.top_p, s.min_p),
            (1.0, 0, 1.0, 0.0)
        );
        let s = parse(json!({"temperature": 0.8, "top_p": 0.95, "top_k": -1, "seed": -1})).unwrap();
        assert_eq!((s.temperature, s.top_k, s.top_p), (0.8, 0, 0.95));
        assert_eq!(s.seed, u64::MAX);
        assert_eq!(parse(json!({"top_k": 20})).unwrap().top_k, 20);
        assert_eq!(parse(json!({"temperature": 0})).unwrap().temperature, 0.0);
        assert_ne!(
            parse(json!({})).unwrap().seed,
            parse(json!({})).unwrap().seed
        );
        for bad in [
            json!({"temperature": -0.1}),
            json!({"top_p": 0}),
            json!({"top_p": 1.5}),
            json!({"top_k": -2}),
            json!({"min_p": 2}),
        ] {
            assert!(parse(bad.clone()).is_err(), "{bad}");
        }
        let defaults = SamplingDefaults {
            temperature: 0.6,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.0,
        };
        let fields: SamplingFields = serde_json::from_value(json!({"top_p": 1.0})).unwrap();
        let s = sampling(&fields, &defaults).unwrap();
        assert_eq!((s.temperature, s.top_k, s.top_p), (0.6, 40, 1.0));
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
