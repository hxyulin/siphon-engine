"""Teacher-forced MLX-LM capture over a long text, for the engine's quality gate.

`capture` runs MLX over every token of a text and saves:

- `tokens`, I32 `[n]`: the text's token ids, without special tokens.
- `target_logprobs`, F32 `[n - 1]`: at position i, MLX's log-probability of token i + 1.
- `argmax`, I32 `[n]`: MLX's top-1 token at every position.
- `sampled_positions`, I32 `[k]`, and `sampled_logits`, F32 `[k, vocab]`: full logits
  at every 256th position, around the sliding-window wrap at 1024, around the start of
  query log-scaling at 8192, and at the last 16 positions.
- `continuation`, I32 `[c]`: greedy tokens after the text. The first is `argmax[-1]`.

By default the text goes through MLX in chunks, which takes MLX's multi-row path. With
`--token-by-token`, every token is its own step, the single-row decode path the
engine follows.

`compare` reports how far one capture is from another of the same text with the
metrics the engine's `eval` binary uses. Comparing the two MLX modes gives the noise
floor the engine's thresholds are based on.
"""

import argparse
import hashlib
import json
import math
import platform
import time
from importlib import metadata
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from safetensors.numpy import load_file

# The sliding window wraps at position 1024, and global layers scale queries from
# position 8192 on (`sliding_window_size` and `log_scaling_n_floor` in the config).
WINDOW = 1024
LOG_FLOOR = 8192
SAMPLE_EVERY = 256
SAMPLE_RANGES = [(WINDOW - 4, WINDOW + 7), (LOG_FLOOR - 4, LOG_FLOOR + 9)]
SAMPLE_TAIL = 16


def sampled_positions(n: int) -> list[int]:
    positions = set(range(0, n, SAMPLE_EVERY))
    for start, stop in SAMPLE_RANGES:
        positions.update(range(start, min(stop, n)))
    positions.update(range(max(0, n - SAMPLE_TAIL), n))
    return sorted(positions)


def capture(args: argparse.Namespace) -> None:
    text = args.text.read_text()
    model, tokenizer = load(str(args.model))
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n = len(tokens)
    sampled = sampled_positions(n)
    sampled_set = set(sampled)

    cache = model.make_cache()
    target_logprobs = np.zeros(n - 1, np.float32)
    argmax = np.zeros(n, np.int32)
    sampled_logits: dict[int, np.ndarray] = {}

    def keep(start: int, logits: mx.array) -> None:
        """Takes what the gate needs from the logits of positions `start..`."""
        logits = logits[0]
        rows = logits.shape[0]
        stop = start + rows
        top = mx.argmax(logits, axis=-1)
        # The last position of the text has no target.
        targets = mx.array(tokens[start + 1 : stop + 1], dtype=mx.int32)
        logprobs = None
        if targets.size:
            log_norm = mx.logsumexp(logits[: targets.size], axis=-1)
            picked = mx.take_along_axis(
                logits[: targets.size], targets[:, None], axis=-1
            )
            logprobs = picked[:, 0] - log_norm
        rows_kept = [p - start for p in range(start, stop) if p in sampled_set]
        kept = logits[mx.array(rows_kept, dtype=mx.int32)] if rows_kept else None
        mx.eval(top, logprobs, kept)
        argmax[start:stop] = np.array(top)
        if logprobs is not None:
            target_logprobs[start : start + targets.size] = np.array(logprobs)
        for i, row in enumerate(rows_kept):
            sampled_logits[start + row] = np.array(kept[i])

    started = time.perf_counter()
    step = 1 if args.token_by_token else args.chunk
    for start in range(0, n, step):
        chunk = mx.array([tokens[start : start + step]], dtype=mx.int32)
        keep(start, model(chunk, cache=cache))
        if (start + step) % 2048 < step:
            print(f"{start + chunk.shape[1]}/{n} tokens", flush=True)
    seconds = time.perf_counter() - started

    continuation = [int(argmax[-1])]
    for _ in range(args.continue_tokens - 1):
        logits = model(mx.array([[continuation[-1]]], dtype=mx.int32), cache=cache)
        continuation.append(int(mx.argmax(logits[0, -1]).item()))

    tensors = {
        "tokens": np.array(tokens, np.int32),
        "target_logprobs": target_logprobs,
        "argmax": argmax,
        "sampled_positions": np.array(sampled, np.int32),
        "sampled_logits": np.stack([sampled_logits[p] for p in sampled]),
        "continuation": np.array(continuation, np.int32),
    }
    info = {
        "model": str(args.model),
        "text": args.text.name,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "tokens": n,
        "mode": "token-by-token" if args.token_by_token else f"chunks of {args.chunk}",
        "perplexity": math.exp(-float(target_logprobs.mean())),
        "seconds": round(seconds, 1),
        "continuation_text": tokenizer.decode(continuation),
        "mlx": metadata.version("mlx"),
        "mlx_lm": metadata.version("mlx-lm"),
        "machine": platform.platform(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.out),
        {k: mx.array(v) for k, v in tensors.items()},
        metadata={k: json.dumps(v) for k, v in info.items()},
    )
    print(json.dumps(info, indent=2))


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def compare(args: argparse.Namespace) -> None:
    """Reports how far capture `b` is from the reference capture `a`."""
    a, b = load_file(str(args.reference)), load_file(str(args.other))
    if not np.array_equal(a["tokens"], b["tokens"]):
        raise SystemExit("the captures are of different texts")
    n = a["tokens"].size

    ppl_a = math.exp(-float(a["target_logprobs"].astype(np.float64).mean()))
    ppl_b = math.exp(-float(b["target_logprobs"].astype(np.float64).mean()))
    dlp = np.abs(a["target_logprobs"] - b["target_logprobs"])
    print(f"tokens {n}")
    print(
        f"perplexity {ppl_a:.4f} vs {ppl_b:.4f}, ratio - 1 {ppl_b / ppl_a - 1:+.2e}; "
        f"target log-prob |diff| mean {dlp.mean():.2e} max {dlp.max():.2e} "
        f"at position {int(dlp.argmax())}"
    )
    agree = a["argmax"] == b["argmax"]
    print(f"top-1 agreement {agree.sum()}/{n} ({agree.mean():.4%})")
    for label, start, stop in [
        ("before window wrap", 0, min(WINDOW, n)),
        ("window wrapped", WINDOW, min(LOG_FLOOR, n)),
        ("log-scaled queries", LOG_FLOOR, n),
    ]:
        if start < stop:
            print(
                f"  {label:<20} {start:>5}..{stop:<5} top-1 {agree[start:stop].mean():.2%}"
                f" |dlogprob| {dlp[start : stop - 1].mean():.4f}"
            )

    positions = a["sampled_positions"]
    if not np.array_equal(positions, b["sampled_positions"]):
        raise SystemExit("the captures sampled different positions")
    pa, pb = log_softmax(a["sampled_logits"]), log_softmax(b["sampled_logits"])
    kl = (np.exp(pa) * (pa - pb)).sum(axis=-1)
    print(
        f"KL at {positions.size} sampled positions: mean {kl.mean():.2e} "
        f"median {np.median(kl):.2e} max {kl.max():.2e}"
    )
    for i in np.argsort(kl)[::-1][:5]:
        print(f"  position {positions[i]:>6}: KL {kl[i]:.2e}")

    ca, cb = a["continuation"], b["continuation"]
    prefix = next((i for i, (x, y) in enumerate(zip(ca, cb)) if x != y), len(ca))
    print(f"continuation matches for {prefix}/{len(ca)} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture", help="capture MLX over a text")
    cap.add_argument("model", type=Path)
    cap.add_argument("text", type=Path)
    cap.add_argument("out", type=Path)
    cap.add_argument("--chunk", type=int, default=512)
    cap.add_argument("--continue-tokens", type=int, default=32)
    cap.add_argument("--token-by-token", action="store_true")
    cap.set_defaults(run=capture)
    cmp = sub.add_parser("compare", help="compare two captures of the same text")
    cmp.add_argument("reference", type=Path)
    cmp.add_argument("other", type=Path)
    cmp.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
