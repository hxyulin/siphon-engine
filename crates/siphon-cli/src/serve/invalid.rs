//! Request validation errors that name the field, quote what arrived and say what the
//! server accepts.

use std::fmt;

use serde_json::Value;

/// A request the server cannot serve, reported to the client as a 400.
///
/// `param` is the field's path in the request, such as `messages[3].content[1].type`; the
/// OpenAI error body reports it as `param`, and the message starts with it.
#[derive(Debug)]
pub struct InvalidRequest {
    param: Option<String>,
    detail: String,
}

impl InvalidRequest {
    /// An error about the request as a whole, not one field.
    pub fn new(detail: impl Into<String>) -> Self {
        Self {
            param: None,
            detail: detail.into(),
        }
    }

    /// An error about the field `param`.
    pub fn field(param: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            param: Some(param.into()),
            detail: detail.into(),
        }
    }

    /// The field `param` holds `got` where the server takes `accepted`. When `accepted`
    /// is a number or a boolean and `got` is a string that reads as one, the message says
    /// to send it unquoted.
    pub fn expected(param: impl Into<String>, accepted: &str, got: &Value) -> Self {
        Self::field(
            param,
            format!(
                "expected {accepted}, got {}{}",
                describe(got),
                hint(accepted, got)
            ),
        )
    }

    /// The field `param` is missing; the server takes `accepted` there.
    pub fn missing(param: impl Into<String>, accepted: &str) -> Self {
        Self::field(param, format!("missing; expected {accepted}"))
    }

    /// This error, with its field inside `prefix`: `content[1]` within `messages[3]` is
    /// `messages[3].content[1]`.
    #[must_use]
    pub fn within(self, prefix: impl fmt::Display) -> Self {
        let param = match self.param {
            None => prefix.to_string(),
            Some(p) if p.is_empty() => prefix.to_string(),
            Some(p) if p.starts_with('[') => format!("{prefix}{p}"),
            Some(p) => format!("{prefix}.{p}"),
        };
        Self {
            param: Some(param),
            detail: self.detail,
        }
    }

    /// The field's path, if the error is about one field.
    pub fn param(&self) -> Option<&str> {
        self.param.as_deref()
    }

    /// This error about `messages[i]` of chat messages translated from another API's
    /// request, moved to the field that message came from, `origins[i]`.
    #[must_use]
    pub fn relocate(self, origins: &[Origin]) -> Self {
        let Some(param) = &self.param else {
            return self;
        };
        let Some((i, rest)) = index(param, "messages") else {
            return self;
        };
        let Some(origin) = origins.get(i) else {
            return self;
        };
        let mapped = |rest: &str, field: &str, paths: &[String]| {
            let (j, rest) = index(rest, field)?;
            Some(format!("{}{rest}", paths.get(j)?))
        };
        let param = mapped(rest, ".content", &origin.parts)
            .or_else(|| {
                // A call's own fields have other names in the source API.
                mapped(rest, ".tool_calls", &origin.calls).map(|p| {
                    p.split_once(".function")
                        .map_or(p.clone(), |(call, _)| call.to_owned())
                })
            })
            .unwrap_or_else(|| origin.path.clone());
        Self {
            param: Some(param),
            detail: self.detail,
        }
    }
}

/// `field[i]` at the start of `param`: `i` and the rest of `param`.
fn index<'a>(param: &'a str, field: &str) -> Option<(usize, &'a str)> {
    let rest = param.strip_prefix(field)?.strip_prefix('[')?;
    let (i, rest) = rest.split_once(']')?;
    Some((i.parse().ok()?, rest))
}

/// Where one translated chat message came from in the request: the item's path, and the
/// paths of its content parts and tool calls.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Origin {
    pub path: String,
    pub parts: Vec<String>,
    pub calls: Vec<String>,
}

impl Origin {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            ..Self::default()
        }
    }
}

impl fmt::Display for InvalidRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.param {
            Some(param) => write!(f, "{param}: {}", self.detail),
            None => f.write_str(&self.detail),
        }
    }
}

impl std::error::Error for InvalidRequest {}

/// The longest string an error quotes before it cuts the rest.
const QUOTE_CHARS: usize = 60;

/// A value and its JSON type, for an error message: `"0.9" (a string)`, `5 (a number)`,
/// `an object`.
pub fn describe(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(b) => format!("{b} (a boolean)"),
        Value::Number(n) => format!("{n} (a number)"),
        Value::String(s) => format!("{} (a string)", quote(s)),
        Value::Array(a) => format!("an array of {} items", a.len()),
        Value::Object(_) => "an object".to_owned(),
    }
}

/// `s` in JSON quotes, cut after [`QUOTE_CHARS`] characters.
pub fn quote(s: &str) -> String {
    match s.char_indices().nth(QUOTE_CHARS) {
        Some((end, _)) => format!("{}...", Value::from(&s[..end])),
        None => Value::from(s).to_string(),
    }
}

/// Advice for a number or boolean sent as a string, when `accepted` asks for one.
fn hint(accepted: &str, got: &Value) -> String {
    let Value::String(s) = got else {
        return String::new();
    };
    let s = s.trim();
    let wants_number = accepted.contains("number") || accepted.contains("integer");
    if wants_number && s.parse::<f64>().is_ok_and(f64::is_finite) {
        return format!("; send {s} as a JSON number, not a string");
    }
    if accepted.contains("boolean") && matches!(s, "true" | "false") {
        return format!("; send {s} as a JSON boolean, not a string");
    }
    String::new()
}

/// One of `names`, quoted and joined for a message: `"a", "b" or "c"`.
pub fn one_of(names: &[&str]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| format!("\"{n}\"")).collect();
    match quoted.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} or {last}", rest.join(", ")),
        None => String::new(),
    }
}

/// A serde message from deserializing a request, such as `invalid type: string "0.7",
/// expected f64 at line 1 column 35`, in the words of [`InvalidRequest::expected`]:
/// `expected a number, got "0.7" (a string); send 0.7 as a JSON number, not a string`.
pub fn serde_detail(message: &str) -> String {
    let message = match message.rfind(" at line ") {
        Some(at) if message[at..].contains(" column ") => &message[..at],
        _ => message,
    };
    let Some(rest) = message.strip_prefix("invalid type: ") else {
        return message.replace("invalid value: ", "invalid value ");
    };
    let Some((unexpected, expected)) = rest.rsplit_once(", expected ") else {
        return message.to_owned();
    };
    let accepted = match expected {
        "f64" | "f32" => "a number",
        "u64" | "u32" | "usize" => "a non-negative integer",
        "i64" | "i32" => "an integer",
        "a sequence" => "an array",
        "a map" => "an object",
        e if e.starts_with("struct ") => "an object",
        e => e,
    };
    let got = match unexpected.split_once(' ') {
        Some(("string", s)) => serde_json::from_str::<String>(s).map_or_else(
            |_| format!("{s} (a string)"),
            |s| describe(&Value::String(s)),
        ),
        Some(("integer" | "floating", n)) => {
            let n = n.trim_start_matches("point ").trim_matches('`');
            format!("{n} (a number)")
        }
        Some(("boolean", b)) => format!("{} (a boolean)", b.trim_matches('`')),
        _ => match unexpected {
            "sequence" => "an array".to_owned(),
            "map" => "an object".to_owned(),
            "null" | "unit value" => "null".to_owned(),
            other => other.to_owned(),
        },
    };
    let hint = match unexpected.split_once(' ') {
        Some(("string", s)) => serde_json::from_str::<String>(s)
            .map(|s| hint(accepted, &Value::String(s)))
            .unwrap_or_default(),
        _ => String::new(),
    };
    format!("expected {accepted}, got {got}{hint}")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn messages_name_the_field_value_and_type() {
        let e = InvalidRequest::expected("reasoning_effort", "a number", &json!("0.9"));
        assert_eq!(e.param(), Some("reasoning_effort"));
        assert_eq!(
            e.to_string(),
            "reasoning_effort: expected a number, got \"0.9\" (a string); send 0.9 as a JSON \
             number, not a string"
        );
        let e = InvalidRequest::expected("stream", "a boolean", &json!("true"));
        assert!(
            e.to_string()
                .ends_with("send true as a JSON boolean, not a string")
        );
        // A string that is no number gets no advice, and a string field none either.
        let e = InvalidRequest::expected("role", "a string", &json!(7));
        assert_eq!(e.to_string(), "role: expected a string, got 7 (a number)");
        let e = InvalidRequest::expected("n", "a number", &json!("many"));
        assert_eq!(
            e.to_string(),
            "n: expected a number, got \"many\" (a string)"
        );
    }

    #[test]
    fn paths_nest() {
        let e = InvalidRequest::field("type", "unknown")
            .within("content[1]")
            .within("messages[3]");
        assert_eq!(e.param(), Some("messages[3].content[1].type"));
        let e = InvalidRequest::new("bad").within("tools[0]");
        assert_eq!(e.to_string(), "tools[0]: bad");
        let e = InvalidRequest::field("[2]", "bad").within("stop");
        assert_eq!(e.param(), Some("stop[2]"));
    }

    #[test]
    fn translated_messages_point_at_their_source() {
        let origins = [
            Origin::new("system"),
            Origin {
                path: "messages[0]".to_owned(),
                parts: vec!["messages[0].content[2]".to_owned()],
                calls: vec!["messages[0].content[4]".to_owned()],
            },
        ];
        let at = |param: &str| {
            InvalidRequest::field(param, "bad")
                .relocate(&origins)
                .param()
                .map(str::to_owned)
        };
        assert_eq!(
            at("messages[1].content[0]").as_deref(),
            Some("messages[0].content[2]")
        );
        assert_eq!(
            at("messages[1].tool_calls[0].function.arguments").as_deref(),
            Some("messages[0].content[4]")
        );
        assert_eq!(at("messages[1].role").as_deref(), Some("messages[0]"));
        assert_eq!(at("messages[0].content[3]").as_deref(), Some("system"));
        assert_eq!(at("tools[0]").as_deref(), Some("tools[0]"));
    }

    #[test]
    fn long_strings_are_cut() {
        let long = "x".repeat(200);
        assert_eq!(
            describe(&json!(long)),
            format!("\"{}\"... (a string)", "x".repeat(60))
        );
        assert_eq!(describe(&json!([1, 2])), "an array of 2 items");
        assert_eq!(one_of(&["a", "b", "c"]), "\"a\", \"b\" or \"c\"");
    }

    #[test]
    fn serde_messages_are_reworded() {
        assert_eq!(
            serde_detail("invalid type: string \"0.7\", expected f64 at line 1 column 35"),
            "expected a number, got \"0.7\" (a string); send 0.7 as a JSON number, not a string"
        );
        assert_eq!(
            serde_detail("invalid type: string \"5\", expected u64 at line 1 column 17"),
            "expected a non-negative integer, got \"5\" (a string); send 5 as a JSON number, \
             not a string"
        );
        assert_eq!(
            serde_detail("invalid type: string \"hi\", expected a sequence at line 1 column 16"),
            "expected an array, got \"hi\" (a string)"
        );
        assert_eq!(
            serde_detail("invalid type: floating point `0.5`, expected u64"),
            "expected a non-negative integer, got 0.5 (a number)"
        );
        assert_eq!(
            serde_detail("invalid type: string \"yes\", expected a boolean"),
            "expected a boolean, got \"yes\" (a string)"
        );
        assert_eq!(
            serde_detail("missing field `messages` at line 1 column 2"),
            "missing field `messages`"
        );
    }
}
