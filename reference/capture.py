"""Record each decoder-layer visit's inputs and outputs from MLX-LM.

The engine's parity tests read the resulting safetensors file. Keys look like
`prefill.visit07.in.hidden` and `decode2.logits`. One prefill over the prompt
is followed by greedy decode steps, each with its own key prefix.
"""

import argparse
import json
import platform
from importlib import metadata
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.inkling2.layers import DecoderLayer

DEFAULT_PROMPT = "Write a Rust function that returns the n-th Fibonacci number."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--decode-steps", type=int, default=8)
    args = parser.parse_args()

    model, tokenizer = load(str(args.model))
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], add_generation_prompt=True
    )

    tensors: dict[str, mx.array] = {}
    phase = {"name": "", "visit": 0}
    original_call = DecoderLayer.__call__

    def recorded_call(self, hidden_states, block_states, **kwargs):
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
    language_model = model.model.language_model
    cache = model.make_cache()

    def step(name: str, ids: list[int]) -> int:
        phase.update(name=name, visit=0)
        inputs = mx.array([ids], dtype=mx.int32)
        tensors[f"{name}.tokens"] = inputs
        tensors[f"{name}.embed"] = language_model.embed(inputs)
        logits = model(inputs, cache=cache)
        tensors[f"{name}.logits"] = logits
        mx.eval(tensors)
        return int(mx.argmax(logits[0, -1]).item())

    token = step("prefill", prompt_ids)
    generated = [token]
    for i in range(args.decode_steps):
        token = step(f"decode{i}", [token])
        generated.append(token)

    info = {
        "model": str(args.model),
        "prompt": args.prompt,
        "prompt_tokens": len(prompt_ids),
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


if __name__ == "__main__":
    main()
