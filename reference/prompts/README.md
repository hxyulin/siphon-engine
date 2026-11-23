# Eval texts

Texts for the quality gate (`reference/capture_eval.py` and the `eval` binary). Each one
joins source files from a crate published on crates.io. A `==> path <==` line comes
before each file, and the text stops at the last whole line that fits a token budget.
Token counts are from the Inkling 2 Mini tokenizer, without special tokens.

| File | Tokens | Source | Licence |
| --- | --- | --- | --- |
| `anyhow.txt` | 11,186 | anyhow 1.0.104: `README.md`, `src/lib.rs`, and the first 175 lines of `src/error.rs` | MIT OR Apache-2.0 |
| `regex-readme.txt` | 2,798 | regex 1.13.2: the first 246 lines of `README.md` | MIT OR Apache-2.0 |
| `half.txt` | 2,785 | half 2.7.1: `README.md` and the first 65 lines of `src/lib.rs` (CRLF converted to LF) | MIT OR Apache-2.0 |

The crates' repositories are <https://github.com/dtolnay/anyhow>,
<https://github.com/rust-lang/regex> and <https://github.com/VoidStarKat/half-rs>. The texts
are used under the MIT licence:

```text
Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

The regex crate's copyright line is "Copyright (c) 2014 The Rust Project Developers". The
anyhow and half licence files have no copyright line.
