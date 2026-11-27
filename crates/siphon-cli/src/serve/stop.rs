//! Stop strings: generated text is cut where the first of them starts.

/// One request's stop strings and the text held back because it could still begin one.
#[derive(Clone, Debug, Default)]
pub struct Stops {
    strings: Vec<String>,
    /// Length in bytes of the longest stop string (0 when `strings` is empty).
    max_len: usize,
    /// 256-bit set of every stop string's first UTF-8 byte, for O(1) rejection.
    first_bytes: [u64; 4],
    held: String,
    /// The index in `strings` of the string that matched.
    matched: Option<usize>,
}

impl Stops {
    /// Stop strings, none of them empty.
    pub fn new(stops: Vec<String>) -> Self {
        debug_assert!(stops.iter().all(|s| !s.is_empty()));
        let mut max_len = 0;
        let mut first_bytes = [0u64; 4];
        for stop in &stops {
            max_len = max_len.max(stop.len());
            if let Some(&b) = stop.as_bytes().first() {
                first_bytes[(b >> 6) as usize] |= 1u64 << (b & 63);
            }
        }
        Self {
            strings: stops,
            max_len,
            first_bytes,
            held: String::new(),
            matched: None,
        }
    }

    /// Whether no stop strings are configured.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    #[inline]
    fn has_first_byte(&self, b: u8) -> bool {
        (self.first_bytes[(b >> 6) as usize] >> (b & 63)) & 1 != 0
    }

    /// Takes the next piece of text and returns what may be sent: the text before a stop
    /// string, less a tail that could be the start of one. Returns nothing after a match.
    #[cfg_attr(not(test), expect(dead_code, reason = "kept for tests"))]
    pub fn push(&mut self, text: &str) -> String {
        let mut buf = text.to_owned();
        self.filter(&mut buf);
        buf
    }

    /// Filters `text` in place so it holds what may be sent: zero allocations when no stop
    /// strings are set, after a match, or when nothing is held and `text` contains no stop
    /// prefix byte.
    pub fn filter(&mut self, text: &mut String) {
        if self.matched.is_some() {
            text.clear();
            return;
        }
        if self.strings.is_empty() || text.is_empty() {
            return;
        }
        // Fast path: nothing held and no byte of `text` can start any stop string.
        if self.held.is_empty() && !text.bytes().any(|b| self.has_first_byte(b)) {
            return;
        }
        let prev_len = self.held.len();
        self.held.push_str(text);
        let mut first: Option<(usize, usize)> = None;
        for (i, stop) in self.strings.iter().enumerate() {
            // `self.held[..prev_len]` had no match, so any new match must start within
            // `stop.len() - 1` bytes before `prev_len`.
            let raw_from = prev_len.saturating_sub(stop.len().saturating_sub(1));
            let scan_from = ceil_char_boundary(&self.held, raw_from);
            if let Some(rel) = self.held[scan_from..].find(stop.as_str()) {
                let at = scan_from + rel;
                if first.is_none_or(|best| (at, i) < best) {
                    first = Some((at, i));
                }
            }
        }
        if let Some((at, i)) = first {
            self.matched = Some(i);
            text.clear();
            text.push_str(&self.held[..at]);
            self.held.clear();
            return;
        }
        let keep = self.partial();
        text.clear();
        if keep > 0 {
            text.push_str(&self.held[..keep]);
            if keep == self.held.len() {
                self.held.clear();
            } else {
                self.held.drain(..keep);
            }
        }
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
        // Any complete match was already caught, so a partial prefix is strictly shorter
        // than `self.max_len`.
        let max_partial = self.max_len.saturating_sub(1);
        if max_partial == 0 {
            return self.held.len();
        }
        let from = ceil_char_boundary(&self.held, self.held.len().saturating_sub(max_partial));
        let tail_window = &self.held[from..];
        let bytes = tail_window.as_bytes();
        tail_window
            .char_indices()
            .find_map(|(rel, _)| {
                if !self.has_first_byte(bytes[rel]) {
                    return None;
                }
                let tail = &tail_window[rel..];
                self.strings
                    .iter()
                    .any(|stop| stop.starts_with(tail))
                    .then_some(from + rel)
            })
            .unwrap_or(self.held.len())
    }
}

#[inline]
fn ceil_char_boundary(s: &str, mut index: usize) -> usize {
    if index >= s.len() {
        return s.len();
    }
    while index < s.len() && !s.is_char_boundary(index) {
        index += 1;
    }
    index
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

    #[test]
    fn filter_in_place_matches_push_and_benchmarks_speedup() {
        type Case<'a> = (&'a [&'a str], &'a [&'a str], &'a str, Option<&'a str>);
        // Parity across tricky partial overlaps and single-byte stops.
        let cases: &[Case<'_>] = &[
            (&[], &["alpha", " beta", " gamma"], "alpha beta gamma", None),
            (
                &["STOP", "END"],
                &["a S", "TO", "X EN", "D!"],
                "a STOX ",
                Some("END"),
            ),
            (
                &["\n\n\n"],
                &["a\n", "\n", "b\n", "\n\n", "c"],
                "a\n\nb",
                Some("\n\n\n"),
            ),
            (
                &["aaa"],
                &["a", "a", "b", "a", "a", "a", "c"],
                "aab",
                Some("aaa"),
            ),
            (&["a"], &["xyz", "a", "more"], "xyz", Some("a")),
        ];
        for &(list, pieces, want_sent, want_match) in cases {
            let mut stops = stops(list);
            let mut sent = String::new();
            for &piece in pieces {
                let mut buf = piece.to_owned();
                stops.filter(&mut buf);
                sent.push_str(&buf);
            }
            if stops.matched().is_none() {
                sent.push_str(&stops.flush());
            }
            assert_eq!(sent, want_sent, "{list:?}");
            assert_eq!(stops.matched(), want_match, "{list:?}");
        }

        // Microbenchmark: 20,000 streamed tokens with 4 stop sequences vs naive O(held) scan.
        let stop_list = vec![
            "\nUser:".to_owned(),
            "\nAssistant:".to_owned(),
            "<|eot_id|>".to_owned(),
            "```end```".to_owned(),
        ];
        let pieces = [
            "The ", "quick ", "brown ", "fox ", "jumps ", "over ", "lazy ", "dog. ",
        ];
        let n = 20_000usize;

        let t0 = std::time::Instant::now();
        let mut held = String::new();
        let mut naive_len = 0usize;
        for i in 0..n {
            held.push_str(pieces[i % pieces.len()]);
            let _ = stop_list
                .iter()
                .enumerate()
                .filter_map(|(idx, s)| held.find(s.as_str()).map(|at| (at, idx)))
                .min();
            let longest = stop_list.iter().map(String::len).max().unwrap_or(0);
            let from = held.len().saturating_sub(longest);
            let keep = held
                .char_indices()
                .map(|(idx, _)| idx)
                .filter(|&idx| idx >= from)
                .find(|&idx| {
                    let tail = &held[idx..];
                    stop_list.iter().any(|s| s.starts_with(tail))
                })
                .unwrap_or(held.len());
            let rest = held.split_off(keep);
            let emitted = std::mem::replace(&mut held, rest);
            naive_len += emitted.len();
        }
        let naive_ns = t0.elapsed().as_secs_f64() * 1e9 / n as f64;

        let t1 = std::time::Instant::now();
        let mut fast = Stops::new(stop_list);
        let mut fast_len = 0usize;
        let mut buf = String::with_capacity(16);
        for i in 0..n {
            buf.clear();
            buf.push_str(pieces[i % pieces.len()]);
            fast.filter(&mut buf);
            fast_len += buf.len();
        }
        let fast_ns = t1.elapsed().as_secs_f64() * 1e9 / n as f64;

        assert_eq!(naive_len, fast_len);
        eprintln!(
            "stops ({n} toks, 4 stops): naive={naive_ns:.1} ns/tok, fast={fast_ns:.1} ns/tok ({:.1}x)",
            naive_ns / fast_ns.max(1e-9)
        );
    }
}
