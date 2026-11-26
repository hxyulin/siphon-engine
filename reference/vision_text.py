# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "mlx==0.32.4",
#     "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm@111ac668078e017c9db1eb3936f4d7a6e0a88f89",
#     "numpy",
#     "safetensors",
# ]
# ///
"""What different image embeddings do to the text model's answers.

For each fixture image, mlx-vlm's text model runs the capture's chat prompt with
the image rows taken from Transformers' F32 projection (the baseline) and from
each other encoder: mlx-vlm's vision tower in BF16, and each tensor of the
Siphon dump (Siphon's GPU tower).

- The baseline generates `--tokens` greedy tokens.
- Each variant runs teacher-forced over the prompt and the baseline's tokens.
  KL(baseline || variant) is taken over every row after the image's last
  placeholder: the rest of the prompt and the generated rows.
- Each variant also generates its own greedy tokens, compared position by
  position with the baseline's.

usage: uv run --script reference/vision_text.py <model-dir> <capture-dir> <siphon-dir> <out.json>
           [--tokens N] [--only NAME ...]

`<capture-dir>/<name>.safetensors` holds `f32.projected`, `bf16.projected` and
`prompt_ids` (`vision_capture.py capture`); `<siphon-dir>/<name>.safetensors`
holds `bf16` (`cargo run --release -p siphon-cuda --bin vision --
<model-dir> --dump`).
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

IMAGE_TOKEN = 200054
EOS = 200006
NAMES = ("tiny", "alpha", "tall", "square", "screenshot", "huge")
GREEDY_COMPARE = 24


def embeddings(model, ids: list[int], features: np.ndarray) -> mx.array:
    """The prompt's input rows with the image placeholders replaced by `features`."""
    h = model.language_model.model.embed(mx.array([ids]))
    slots = np.flatnonzero(np.array(ids) == IMAGE_TOKEN)
    start, end = int(slots[0]), int(slots[-1]) + 1
    assert end - start == len(slots) == features.shape[0], "placeholders do not match features"
    image = mx.array(features)[None].astype(h.dtype)
    return mx.concatenate([h[:, :start], image, h[:, end:]], axis=1)


def greedy(model, ids: list[int], features: np.ndarray, count: int) -> list[int]:
    cache = model.make_cache()
    out = model.language_model(inputs_embeds=embeddings(model, ids, features), cache=cache)
    tokens = [int(mx.argmax(out.logits[0, -1]).item())]
    while len(tokens) < count:
        step = model.language_model(inputs=mx.array([[tokens[-1]]]), cache=cache)
        tokens.append(int(mx.argmax(step.logits[0, -1]).item()))
    return tokens


def log_probs(model, ids: list[int], features: np.ndarray, first: int) -> mx.array:
    """Log-softmax of every row from `first` on, teacher-forced over `ids`."""
    out = model.language_model(inputs_embeds=embeddings(model, ids, features), cache=model.make_cache())
    logits = out.logits[0, first:].astype(mx.float32)
    result = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(result)
    return result


def text(tokenizer, tokens: list[int]) -> str:
    if EOS in tokens:
        tokens = tokens[: tokens.index(EOS)]
    return tokenizer.decode(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("siphon", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    from mlx_vlm import load

    model, processor = load(str(args.model))
    tokenizer = processor.tokenizer
    results = {}
    for name in args.only or NAMES:
        capture = load_file(str(args.capture / f"{name}.safetensors"))
        siphon = load_file(str(args.siphon / f"{name}.safetensors"))
        ids = [int(i) for i in capture["prompt_ids"]]
        first = int(np.flatnonzero(np.array(ids) == IMAGE_TOKEN)[-1]) + 1
        variants = {"mlx-vlm bf16": capture["bf16.projected"]}
        variants.update({f"siphon {key}": value for key, value in siphon.items()})
        base_tokens = greedy(model, ids, capture["f32.projected"], args.tokens)
        forced = ids + base_tokens[:-1]
        base = log_probs(model, forced, capture["f32.projected"], first)
        p = mx.exp(base)
        image = {"rows": int(base.shape[0]), "f32": {"text": text(tokenizer, base_tokens)}}
        print(f"{name}: {base.shape[0]} rows after the image", flush=True)
        print(f"  f32: {image['f32']['text']!r}", flush=True)
        for key, features in variants.items():
            q = log_probs(model, forced, features, first)
            kl = (p * (base - q)).sum(axis=-1)
            mx.eval(kl)
            kl = np.asarray(kl)
            tokens = greedy(model, ids, features, args.tokens)
            same = [a == b for a, b in zip(tokens, base_tokens)]
            prefix = same.index(False) if False in same else len(same)
            image[key] = {
                "kl_mean": float(kl.mean()),
                "kl_max": float(kl.max()),
                "first_token_same": bool(same[0]),
                "greedy_same_24": int(sum(same[:GREEDY_COMPARE])),
                "same_prefix": prefix,
                "text": text(tokenizer, tokens),
            }
            r = image[key]
            print(
                f"  {key}: KL mean {r['kl_mean']:.2e} max {r['kl_max']:.3f}, first token "
                f"{'same' if r['first_token_same'] else 'differs'}, {r['greedy_same_24']}/24 greedy, "
                f"same for {prefix} tokens",
                flush=True,
            )
            if prefix < args.tokens:
                print(f"    {r['text']!r}", flush=True)
        results[name] = image
    args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
