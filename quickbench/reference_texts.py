"""Builds the fidelity track's texts from public-domain Project Gutenberg books,
fetched from the pglaf.org mirror.

Each text is a book's body (between Gutenberg's START and END markers, CRLF turned
into LF), cut at the last whole line that fits a token budget. Run it in the
reference environment, which has the tokenizer, then capture MLX over each text:

    uv run --project reference python -I quickbench/reference_texts.py <model-dir> <out-dir>
    uv run --project reference python -I reference/capture_eval.py capture \
        <model-dir> <out-dir>/fidelity-32k.txt <out-dir>/fidelity-32k.safetensors
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

# name -> (Gutenberg ebook number, token budget)
TEXTS = {
    "fidelity-4k": (1661, 4_000),  # The Adventures of Sherlock Holmes
    "fidelity-32k": (1342, 32_000),  # Pride and Prejudice
    "fidelity-128k": (2600, 128_000),  # War and Peace
}


def body(number: int) -> str:
    url = f"https://gutenberg.pglaf.org/cache/epub/{number}/pg{number}.txt"
    text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8-sig")
    text = text.replace("\r\n", "\n")
    start = text.index("\n", text.index("*** START OF")) + 1
    return text[start : text.index("*** END OF")].strip() + "\n"


def cut(text: str, budget: int, tokenizer) -> str:
    lines = text.splitlines(keepends=True)
    lo, hi = 0, len(lines)
    while lo < hi:  # the most lines whose tokens fit the budget
        mid = (lo + hi + 1) // 2
        if len(tokenizer.encode("".join(lines[:mid]), add_special_tokens=False)) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return "".join(lines[:lo])


def main() -> None:
    model, out = Path(sys.argv[1]), Path(sys.argv[2])
    tokenizer = AutoTokenizer.from_pretrained(model)
    out.mkdir(parents=True, exist_ok=True)
    for name, (number, budget) in TEXTS.items():
        text = cut(body(number), budget, tokenizer)
        (out / f"{name}.txt").write_text(text)
        n = len(tokenizer.encode(text, add_special_tokens=False))
        print(f"{name}: {n} tokens, sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
