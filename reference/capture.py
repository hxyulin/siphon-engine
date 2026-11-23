"""Record each decoder-layer visit's inputs and outputs from MLX-LM.

The engine's parity tests read the resulting safetensors file. Keys look like
`prefill.visit07.in.hidden` and `decode2.logits`. One prefill over the prompt
is followed by greedy decode steps, each with its own key prefix.

With `--token-by-token`, the prompt is fed one token per step as well, so every
step takes MLX's single-row decode path. Steps are then named `step000`,
`step001` and so on, prompt tokens first.

`--taps` also records the DSpark draft's four target features of every step as
`<step>.taps`, BF16 `[1, rows, 4, 2304]`: the partial block entering visits 5,
23 and 41, and the final normed hidden state (`target_layer_ids` in the
checkpoint's `mtp_config`). `--routing` records every visit's router top-8 as
`<step>.routing`, I32 `[48, 1, rows, 8]`.

With `--text FILE`, the script runs a text's tokens (without special tokens)
teacher-forced in chunks of `--chunk` rows instead of the prompt and decode
steps. It then skips the per-visit tensors and writes flat arrays: `tokens`,
I32 `[n]`, MLX's top-1 token at every position as `argmax`, I32 `[n]`, and,
if asked, `taps` `[n, 4, 2304]` and `routing` `[48, n, 8]`.
"""

import argparse
import json
import platform
from importlib import metadata
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2.layers import DecoderLayer, TopKRouter
from mlx_lm.models.inkling2.model import TextModel

DEFAULT_PROMPT = "Write a Rust function that returns the n-th Fibonacci number."
# DSpark `target_layer_ids` below the last one: visits whose input is a tap.
TAP_VISITS = (5, 23, 41)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--token-by-token", action="store_true")
    parser.add_argument("--taps", action="store_true", help="record the DSpark taps")
    parser.add_argument(
        "--routing", action="store_true", help="record each visit's router top-8"
    )
    parser.add_argument("--text", type=Path, help="run this text teacher-forced")
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = load(str(args.model))
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )

    tensors: dict[str, mx.array] = {}
    phase = {"name": "", "visit": 0}
    original_call = DecoderLayer.__call__
    # Per step: the tap tensors and router indices, in visit order.
    taps: list[mx.array] = []
    routing: list[mx.array] = []
    record_visits = args.text is None

    def recorded_call(self, hidden_states, block_states, **kwargs):
        if args.taps and phase["visit"] in TAP_VISITS:
            taps.append(hidden_states)
        if not record_visits:
            phase["visit"] += 1
            return original_call(self, hidden_states, block_states, **kwargs)
        prefix = f"{phase['name']}.visit{phase['visit']:02d}"
        phase["visit"] += 1
        tensors[f"{prefix}.in.hidden"] = hidden_states
        if block_states is not None:
            tensors[f"{prefix}.in.blocks"] = block_states
        if kwargs.get("messages") is not None:
            tensors[f"{prefix}.in.messages"] = kwargs["messages"]
        hidden_states, block_states = original_call(
            self, hidden_states, block_states, **kwargs
        )
        tensors[f"{prefix}.out.hidden"] = hidden_states
        tensors[f"{prefix}.out.blocks"] = block_states
        return hidden_states, block_states

    DecoderLayer.__call__ = recorded_call
    original_text_call = TextModel.__call__
    original_router_call = TopKRouter.__call__

    def text_call(self, *call_args, **kwargs):
        hidden_states = original_text_call(self, *call_args, **kwargs)
        if args.taps:
            taps.append(hidden_states)
        return hidden_states

    def router_call(self, hidden_states):
        indices, weights = original_router_call(self, hidden_states)
        if args.routing:
            routing.append(indices.astype(mx.int32))
        return indices, weights

    TextModel.__call__ = text_call
    TopKRouter.__call__ = router_call
    language_model = model.model.language_model
    cache = model.make_cache()

    def take_step_records() -> tuple[mx.array | None, mx.array | None]:
        """The step's taps `[1, rows, 4, 2304]` and routing `[48, 1, rows, 8]`."""
        step_taps = mx.stack(taps, axis=2) if args.taps else None
        step_routing = mx.stack(routing) if args.routing else None
        taps.clear()
        routing.clear()
        phase["visit"] = 0
        return step_taps, step_routing

    if args.text is not None:
        capture_text(args, model, tokenizer, cache, take_step_records)
        return

    def step(name: str, ids: list[int]) -> int:
        phase.update(name=name, visit=0)
        inputs = mx.array([ids], dtype=mx.int32)
        tensors[f"{name}.tokens"] = inputs
        tensors[f"{name}.embed"] = language_model.embed(inputs)
        logits = model(inputs, cache=cache)
        tensors[f"{name}.logits"] = logits
        step_taps, step_routing = take_step_records()
        if step_taps is not None:
            tensors[f"{name}.taps"] = step_taps
        if step_routing is not None:
            tensors[f"{name}.routing"] = step_routing
        mx.eval(tensors)
        return int(mx.argmax(logits[0, -1]).item())

    if args.token_by_token:
        for i, prompt_id in enumerate(prompt_ids):
            token = step(f"step{i:03d}", [prompt_id])
        names = (f"step{len(prompt_ids) + i:03d}" for i in range(args.decode_steps))
    else:
        token = step("prefill", prompt_ids)
        names = (f"decode{i}" for i in range(args.decode_steps))
    generated = [token]
    for name in names:
        token = step(name, [token])
        generated.append(token)

    info = {
        "model": str(args.model),
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_ids),
        "token_by_token": args.token_by_token,
        "generated": generated,
        "generated_text": tokenizer.decode(generated),
        "mlx": metadata.version("mlx"),
        "mlx_lm": metadata.version("mlx-lm"),
        "machine": platform.platform(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.out), tensors, metadata={k: json.dumps(v) for k, v in info.items()}
    )
    print(json.dumps(info, indent=2))


def capture_text(args, model, tokenizer, cache, take_step_records) -> None:
    """Runs `args.text` teacher-forced and saves the flat arrays."""
    text = args.text.read_text()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n = len(tokens)
    argmax, taps, routing = [], [], []
    for start in range(0, n, args.chunk):
        chunk = mx.array([tokens[start : start + args.chunk]], dtype=mx.int32)
        logits = model(chunk, cache=cache)
        step_taps, step_routing = take_step_records()
        argmax.append(mx.argmax(logits[0], axis=-1).astype(mx.int32))
        if step_taps is not None:
            taps.append(step_taps[0])
        if step_routing is not None:
            routing.append(step_routing[:, 0])
        mx.eval(argmax, taps, routing)
        print(f"{start + chunk.shape[1]}/{n} tokens", flush=True)
    tensors = {
        "tokens": mx.array(tokens, dtype=mx.int32),
        "argmax": mx.concatenate(argmax),
    }
    if taps:
        tensors["taps"] = mx.concatenate(taps)
    if routing:
        tensors["routing"] = mx.concatenate(routing, axis=1)
    info = {
        "model": str(args.model),
        "text": args.text.name,
        "tokens": n,
        "chunk": args.chunk,
        "tap_visits": [*TAP_VISITS, "final"],
        "mlx": metadata.version("mlx"),
        "mlx_lm": metadata.version("mlx-lm"),
        "machine": platform.platform(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.out), tensors, metadata={k: json.dumps(v) for k, v in info.items()}
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
