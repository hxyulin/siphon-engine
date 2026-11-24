//! Stop strings: generated text is cut where the first of them starts.

/// One request's stop strings and the text held back because it could still begin one.
#[derive(Debug, Default)]
pub struct Stops {
    strings: Vec<String>,
    held: String,
    /// The index in `strings` of the string that matched.
    matched: Option<usize>,
}

impl Stops {
    /// Stop strings, none of them empty.
    pub fn new(stops: Vec<String>) -> Self {
        debug_assert!(stops.iter().all(|s| !s.is_empty()));
        Self {
            strings: stops,
            ..Self::default()
        }
    }

    /// Takes the next piece of text and returns what may be sent: the text before a stop
    /// string, less a tail that could be the start of one. Returns nothing after a match.
    pub fn push(&mut self, text: &str) -> String {
        if self.matched.is_some() {
            return String::new();
        }
        if self.strings.is_empty() {
            return text.to_owned();
        }
        self.held.push_str(text);
        let first = self
            .strings
            .iter()
            .enumerate()
            .filter_map(|(i, stop)| self.held.find(stop.as_str()).map(|at| (at, i)))
            .min();
        if let Some((at, i)) = first {
            self.matched = Some(i);
            self.held.truncate(at);
            return std::mem::take(&mut self.held);
        }
        let keep = self.partial();
        let rest = self.held.split_off(keep);
        std::mem::replace(&mut self.held, rest)
    }

    /// Ends the text: returns what was held back, which no stop string can complete now.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }

    /// The stop string that matched, if one did.
    pub fn matched(&self) -> Option<&str> {
        self.matched.map(|i| self.strings[i].as_str())
    }

    /// Where the longest tail of the held text that begins a stop string starts.
    fn partial(&self) -> usize {
        let longest = self.strings.iter().map(String::len).max().unwrap_or(0);
        let from = self.held.len().saturating_sub(longest);
        self.held
            .char_indices()
            .map(|(i, _)| i)
            .filter(|&i| i >= from)
            .find(|&i| {
                let tail = &self.held[i..];
                self.strings.iter().any(|stop| stop.starts_with(tail))
            })
            .unwrap_or(self.held.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(list: &[&str]) -> Stops {
        Stops::new(list.iter().map(|&s| s.to_owned()).collect())
    }

    /// Feeds `pieces` and returns the text sent and the matched stop.
    fn run(list: &[&str], pieces: &[&str]) -> (String, Option<String>) {
        let mut stops = stops(list);
        let mut sent = String::new();
        for piece in pieces {
            sent += &stops.push(piece);
        }
        if stops.matched().is_none() {
            sent += &stops.flush();
        }
        (sent, stops.matched().map(str::to_owned))
    }

    #[test]
    fn without_stops_text_passes() {
        assert_eq!(run(&[], &["a", "b"]), ("ab".to_owned(), None));
    }

    #[test]
    fn cuts_before_a_match_split_across_pieces() {
        let (sent, matched) = run(
            &["</answer>"],
            &["The answer", " is 4</", "ans", "wer> and more"],
        );
        assert_eq!(sent, "The answer is 4");
        assert_eq!(matched.as_deref(), Some("</answer>"));
    }

    #[test]
    fn holds_back_only_what_could_start_a_stop() {
        let mut stops = stops(&["END"]);
        assert_eq!(stops.push("abc E"), "abc ");
        assert_eq!(stops.push("N"), "");
        assert_eq!(stops.push("x"), "ENx");
        assert_eq!(stops.flush(), "");
        assert_eq!(stops.matched(), None);
    }

    #[test]
    fn releases_a_held_prefix_at_the_end() {
        assert_eq!(run(&["\n\n"], &["one\n"]), ("one\n".to_owned(), None));
    }

    #[test]
    fn earliest_match_wins() {
        let (sent, matched) = run(&["world", "lo"], &["hello world"]);
        assert_eq!(sent, "hel");
        assert_eq!(matched.as_deref(), Some("lo"));
    }

    #[test]
    fn nothing_after_a_match() {
        let mut stops = stops(&["x"]);
        assert_eq!(stops.push("axb"), "a");
        assert_eq!(stops.push("more"), "");
        assert_eq!(stops.matched(), Some("x"));
    }

    #[test]
    fn multibyte_characters() {
        // The stop's first character matches the text's last one, so it is held.
        let mut stops = stops(&["世界!"]);
        assert_eq!(stops.push("你好世"), "你好");
        assert_eq!(stops.push("界"), "");
        assert_eq!(stops.push("?"), "世界?");
        // A stop that starts mid-way through an emoji's bytes never matches.
        let (sent, matched) = run(&["🦀x"], &["a🦀", "🦀", "x b"]);
        assert_eq!(sent, "a🦀");
        assert_eq!(matched.as_deref(), Some("🦀x"));
        let (sent, matched) = run(&["é"], &["café", " au lait"]);
        assert_eq!(sent, "caf");
        assert_eq!(matched.as_deref(), Some("é"));
    }
}
