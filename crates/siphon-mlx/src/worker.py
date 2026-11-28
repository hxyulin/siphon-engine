"""Serve one local MLX-LM model over a private pipe owned by the Rust process."""

import json
import os
import sys
from inspect import signature
from pathlib import Path


def sampler(options):
    temperature = options["temperature"]
    if temperature == 0:
        return None

    top_k = options["top_k"]
    top_p = options["top_p"]
    min_p = options["min_p"]

    def sample(logprobs):
        logits = logprobs.astype(mx.float32)
        logits = (logits - logits.max(axis=-1, keepdims=True)) / temperature
        if 0 < top_k < logits.shape[-1]:
            logits = apply_top_k(logits, top_k)
        if top_p < 1:
            logits = apply_top_p(logits, top_p)
        if min_p > 0:
            logits = apply_min_p(logits, min_p)
        return mx.random.categorical(logits)

    return sample


def reply(**message):
    print(json.dumps(message), file=protocol, flush=True)


def reply_error(error):
    reply(error=f"{type(error).__name__}: {error}")


# Keep the response pipe separate from Python and native-library diagnostics.
protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

try:
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError("Metal is unavailable")
    mx.set_default_device(mx.gpu)
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p
    from mlx_lm.utils import load_model

    progress = {}
    if "prompt_progress_callback" in signature(generate_step).parameters:
        progress["prompt_progress_callback"] = lambda done, _total: reply(prefill=done)

    model_path = Path(sys.argv[1])
    config = json.loads((model_path / "config.json").read_bytes())
    if config["model_type"] != "inkling2_loop":
        raise ValueError("expected an Inkling 2 Mini checkpoint (inkling2_loop)")
    text = config["text_config"]
    model, _ = load_model(model_path, trust_remote_code=False)
    reply(
        ok={"max_context": text["model_max_length"], "vocab_size": text["vocab_size"]}
    )
# Report library exceptions through the startup protocol.
except Exception as error:  # noqa: BLE001
    reply_error(error)
    sys.exit(1)

tokens = None
for line in sys.stdin:
    try:
        request = json.loads(line)
        match request["op"]:
            case "generate":
                options = request["options"]
                mx.random.seed(options["seed"])
                tokens = generate_step(
                    mx.array(request["prompt"], dtype=mx.uint32),
                    model,
                    max_tokens=options["max_tokens"],
                    prefill_step_size=options["prefill_chunk"],
                    sampler=sampler(options),
                    **progress,
                )
                reply(ok=None)
            case "next":
                token = next(tokens)[0]
                reply(ok=token)
            case "reset":
                tokens = None
                reply(ok=None)
            case _:
                raise ValueError("unknown worker operation")
    # Discard the request cache and keep serving after library errors.
    except Exception as error:  # noqa: BLE001
        tokens = None
        reply_error(error)
