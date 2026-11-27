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

use crate::serve::{
    chat::Ending,
    engine::Sampling,
    invalid::{InvalidRequest, describe, one_of},
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
    let invalid = |param: &str, accepted: &str, got: f64| {
        Err(InvalidRequest::field(
            param,
            format!("expected {accepted}, got {got}"),
        ))
    };
    let temperature = fields.temperature.unwrap_or(defaults.temperature);
    let top_p = fields.top_p.unwrap_or(defaults.top_p);
    let top_k = fields.top_k.unwrap_or(defaults.top_k);
    let min_p = fields.min_p.unwrap_or(defaults.min_p);
    if !(temperature.is_finite() && temperature >= 0.0) {
        return invalid(
            "temperature",
            "a number of at least 0 (0 is greedy)",
            temperature,
        );
    }
    if !(top_p > 0.0 && top_p <= 1.0) {
        return invalid("top_p", "a number above 0 and at most 1", top_p);
    }
    if top_k < -1 {
        return Err(InvalidRequest::field(
            "top_k",
            format!("expected -1 or 0 (no limit) or an integer of at least 1, got {top_k}"),
        ));
    }
    if !(0.0..=1.0).contains(&min_p) {
        return invalid("min_p", "a number from 0 to 1", min_p);
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

/// The sampling of each of a request's `n` choices: [`sampling`] with a seed per choice.
/// With a request seed `s`, choice `i` gets seed `s + i`, as vLLM's parallel sampling
/// derives its child requests' seeds (`ParentRequest._get_child_sampling_params` in
/// `vllm/v1/engine/parallel_sampling.py`). Without one, each choice gets its own random
/// seed. Greedy choices would all be the same, so `n` above 1 with temperature 0 is
/// refused, as vLLM's `SamplingParams` refuses it.
pub fn samplings(
    fields: &SamplingFields,
    defaults: &SamplingDefaults,
    n: usize,
) -> Result<Vec<Sampling>, InvalidRequest> {
    let first = sampling(fields, defaults)?;
    if n > 1 && first.temperature == 0.0 {
        return Err(InvalidRequest::field(
            "n",
            format!(
                "must be 1 with greedy sampling (temperature 0), got {n}: every choice would \
                 be the same"
            ),
        ));
    }
    Ok((0..n)
        .zip(0_u64..)
        .map(|(_, i)| Sampling {
            seed: match fields.seed {
                Some(_) => first.seed.wrapping_add(i),
                None if i == 0 => first.seed,
                None => random_seed(),
            },
            ..first
        })
        .collect())
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
            return Err(InvalidRequest::field(
                *name,
                format!("{why}; leave it out or send null"),
            ));
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

/// A request's `n`: the choices to generate, 1 when it sends none.
pub fn choices(n: Option<u64>) -> Result<usize, InvalidRequest> {
    match n {
        None => Ok(1),
        Some(0) => Err(InvalidRequest::field(
            "n",
            "expected an integer of at least 1, got 0",
        )),
        Some(n) => Ok(usize::try_from(n).unwrap_or(usize::MAX)),
    }
}

/// Accepts `best_of` only when it equals `n`: generating more choices than it returns
/// needs log probabilities to rank them.
pub fn best_of(best_of: Option<u64>, n: usize) -> Result<(), InvalidRequest> {
    let Some(best_of) = best_of else {
        return Ok(());
    };
    let best_of = usize::try_from(best_of).unwrap_or(usize::MAX);
    if best_of < n {
        return Err(InvalidRequest::field(
            "best_of",
            format!("must be at least n ({n}), got {best_of}"),
        ));
    }
    if best_of > n {
        return Err(InvalidRequest::field(
            "best_of",
            format!(
                "above n is not supported, got {best_of} with n {n}: ranking choices needs \
                 log probabilities; send best_of equal to n or leave it out"
            ),
        ));
    }
    Ok(())
}

/// Accepts `response_format` only as plain text: JSON modes need constrained decoding.
pub fn response_format(value: Option<&Value>) -> Result<(), InvalidRequest> {
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(format) => match format.get("type").and_then(Value::as_str) {
            Some("text") => Ok(()),
            _ => Err(InvalidRequest::field(
                "response_format.type",
                format!(
                    "only \"text\" is supported, got {}: JSON modes need constrained decoding",
                    describe(format.get("type").unwrap_or(&Value::Null))
                ),
            )),
        },
    }
}

/// The stop strings of `stop`: a string or an array of strings, none empty.
/// `param` names the field in errors: `stop`, or Anthropic's `stop_sequences`.
pub fn stops(param: &str, value: Option<&Value>) -> Result<Vec<String>, InvalidRequest> {
    let stops = match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                item.as_str().map(str::to_owned).ok_or_else(|| {
                    InvalidRequest::expected(format!("{param}[{i}]"), "a string", item)
                })
            })
            .collect::<Result<_, _>>()?,
        Some(other) => {
            return Err(InvalidRequest::expected(
                param,
                "a string or an array of strings",
                other,
            ));
        }
    };
    if let Some(i) = stops.iter().position(String::is_empty) {
        let param = match value {
            Some(Value::Array(_)) => format!("{param}[{i}]"),
            _ => param.to_owned(),
        };
        return Err(InvalidRequest::field(
            param,
            "stop strings must not be empty",
        ));
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
            InvalidRequest::expected(
                "tool_choice",
                &format!(
                    "{} or {{\"type\": \"function\", \"function\": {{\"name\": ...}}}}",
                    one_of(&["auto", "none", "required"])
                ),
                value.unwrap_or(&Value::Null),
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
        (true, Some(true)) => Err(InvalidRequest::field(
            "add_generation_prompt",
            "must be false (or left out) with continue_final_message true, got true",
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
            assert_eq!(e.param(), Some(name.as_str()), "{e} for {bad}");
            assert!(
                e.to_string().starts_with(&format!("{name}: ")),
                "{e} for {bad}"
            );
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
    fn choice_seeds() {
        let defaults = SamplingDefaults::default();
        let parse = |v: Value, n| {
            let fields: SamplingFields = serde_json::from_value(v).unwrap();
            samplings(&fields, &defaults, n).unwrap()
        };
        // Choice i of a seeded request gets the seed plus i, as in vLLM.
        let seeds: Vec<u64> = parse(json!({"seed": 7, "temperature": 0.8}), 4)
            .iter()
            .map(|s| s.seed)
            .collect();
        assert_eq!(seeds, [7, 8, 9, 10]);
        let seeded = parse(json!({"seed": 7, "temperature": 0.8}), 2);
        let temperatures: Vec<f32> = seeded.iter().map(|s| s.temperature).collect();
        assert_eq!(temperatures, [0.8, 0.8]);
        // The first choice keeps the request's seed, so n = 1 is unchanged.
        assert_eq!(parse(json!({"seed": 7}), 1)[0].seed, 7);
        // -1 is u64::MAX, and the next seed wraps.
        let wrapped: Vec<u64> = parse(json!({"seed": -1}), 2)
            .iter()
            .map(|s| s.seed)
            .collect();
        assert_eq!(wrapped, [u64::MAX, 0]);
        // Greedy choices would be identical: n above 1 is refused, n = 1 is not.
        let greedy: SamplingFields = serde_json::from_value(json!({"temperature": 0})).unwrap();
        assert!(samplings(&greedy, &defaults, 2).is_err());
        assert!(samplings(&greedy, &defaults, 1).is_ok());
        // Without a seed, each choice draws its own.
        let random = parse(json!({}), 3);
        assert_ne!(random[0].seed, random[1].seed);
        assert_ne!(random[1].seed, random[2].seed);
    }

    #[test]
    fn choices_and_formats() {
        assert_eq!(choices(None).unwrap(), 1);
        assert_eq!(choices(Some(1)).unwrap(), 1);
        assert_eq!(choices(Some(4)).unwrap(), 4);
        assert!(choices(Some(0)).is_err());
        assert!(best_of(None, 4).is_ok());
        assert!(best_of(Some(1), 1).is_ok());
        assert!(best_of(Some(4), 4).is_ok());
        assert!(best_of(Some(2), 4).is_err());
        assert!(best_of(Some(2), 1).is_err());
        assert!(response_format(None).is_ok());
        assert!(response_format(Some(&json!({"type": "text"}))).is_ok());
        assert!(response_format(Some(&json!({"type": "json_object"}))).is_err());
        assert!(response_format(Some(&json!({"type": "json_schema", "json_schema": {}}))).is_err());
    }

    #[test]
    fn stop_strings() {
        let stops = |value: Option<&Value>| stops("stop", value);
        assert_eq!(stops(None).unwrap(), Vec::<String>::new());
        assert_eq!(stops(Some(&json!("\n"))).unwrap(), ["\n"]);
        assert_eq!(stops(Some(&json!(["a", "b"]))).unwrap(), ["a", "b"]);
        assert_eq!(stops(Some(&json!([]))).unwrap(), Vec::<String>::new());
        assert_eq!(
            stops(Some(&json!(["a", ""]))).unwrap_err().to_string(),
            "stop[1]: stop strings must not be empty"
        );
        assert_eq!(
            stops(Some(&json!(["a", 1]))).unwrap_err().to_string(),
            "stop[1]: expected a string, got 1 (a number)"
        );
        assert_eq!(
            stops(Some(&json!(1))).unwrap_err().to_string(),
            "stop: expected a string or an array of strings, got 1 (a number)"
        );
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
