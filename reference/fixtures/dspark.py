"""MLX outputs of the DSpark draft's small kernels for tests/draft.rs.

The outputs come from `reference/dspark.py` in its BF16 mode, so the kernels are
checked against the same rounding points as the whole draft.
"""

import sys
from pathlib import Path

import mlx.core as mx

from common import save

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dspark import Draft, fold  # noqa: E402

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, RANK = 2304, 24, 8, 128, 256
mx.random.seed(0)


def bf16(x: mx.array) -> mx.array:
    return x.astype(mx.bfloat16)


def norm_weight(n: int) -> mx.array:
    return bf16(mx.random.normal((n,)) * 0.3 + 1)


def hidden_rows(rows: int, scale: float) -> mx.array:
    """Rows with a spread of magnitudes and a few large channels, as hidden states."""
    x = mx.random.normal((rows, HIDDEN)) * scale * mx.exp(mx.random.normal((rows, 1)) * 0.5)
    return x + (mx.random.uniform(shape=(rows, HIDDEN)) < 0.002) * 20 * scale


weights = {
    "fuse_weight": bf16(mx.random.normal((4, 4))),
    "mix.weight": norm_weight(HIDDEN),
    "norm.weight": norm_weight(HIDDEN),
    "head.weight": norm_weight(HEAD_DIM),
}
draft = Draft(weights, vocab=0, float32=False)
f32 = lambda x: x.astype(mx.float32)  # noqa: E731
tensors: dict[str, mx.array] = {}

# The tap mix: three partial blocks of growing size, then a normed hidden state.
taps = mx.stack([hidden_rows(5, s) for s in (2.0, 6.0, 12.0, 1.0)], axis=1)
taps = bf16(taps)
tensors["mix.taps"] = taps
tensors["mix.alpha"] = draft.alpha[1]
tensors["mix.weight"] = weights["mix.weight"]
tensors["mix.out"] = bf16(draft.mix(f32(taps), 1, "mix.weight"))

# The fused add-norm, and the plain norm of the residual.
residual = bf16(hidden_rows(8, 3.0))
update = bf16(hidden_rows(8, 1.0))
total = f32(residual) + f32(update)
tensors["add_norm.residual"] = residual
tensors["add_norm.x"] = update
tensors["add_norm.weight"] = weights["norm.weight"]
tensors["add_norm.out"] = bf16(draft.rms_norm(total, "norm.weight"))
tensors["add_norm.residual_out"] = bf16(total)
tensors["norm.out"] = bf16(draft.rms_norm(f32(residual), "norm.weight"))

# The block input: the anchor's normed embedding, then mask rows.
embed = bf16(mx.random.normal((10, HIDDEN)) * 0.02)
mask = bf16(mx.random.normal((HIDDEN,)))
weights["embed_norm"] = norm_weight(HIDDEN)
draft.w["embed_norm"] = weights["embed_norm"]
tensors["block_input.embed"] = embed
tensors["block_input.embed_norm"] = weights["embed_norm"]
tensors["block_input.mask_embed"] = mask
tensors["block_input.anchor"] = mx.array([3], dtype=mx.int32)
anchor_row = draft.rms_norm(f32(embed[3]), "embed_norm")
tensors["block_input.out"] = bf16(
    mx.concatenate([anchor_row[None], mx.broadcast_to(f32(mask), (7, HIDDEN))])
)

# Per-head norm and RoPE: the block's queries just past the window, and context keys
# at a large position, where the F32 angle loses bits.
for name, rows, heads, start in (("q", 8, HEADS, 1030), ("k", 3, KV_HEADS, 70001)):
    x = bf16(mx.random.normal((rows, heads, HEAD_DIM)) * mx.exp(mx.random.normal((rows, heads, 1))))
    positions = mx.arange(start, start + rows)
    out = draft.rope(draft.rms_norm(f32(x), "head.weight"), positions)
    tensors[f"rope.{name}.x"] = x
    tensors[f"rope.{name}.start"] = mx.array([start], dtype=mx.int32)
    tensors[f"rope.{name}.out"] = bf16(out)
tensors["rope.weight"] = weights["head.weight"]

# Block attention at a position past the local window (the ring of 1032 slots wraps)
# and at a short context. Keys and values are given per position, 0..P+8.
for start in (1030, 5):
    keys = bf16(mx.random.normal((start + 8, KV_HEADS, HEAD_DIM)))
    values = bf16(mx.random.normal((start + 8, KV_HEADS, HEAD_DIM)))
    q = bf16(mx.random.normal((1, 8, HEADS, HEAD_DIM)) * 1.5)
    ctx = (f32(keys[:start]), f32(values[:start]))
    block_k = f32(keys[start:])[None]
    block_v = f32(values[start:])[None]
    tensors[f"attention.{start}.q"] = q[0]
    tensors[f"attention.{start}.keys"] = keys
    tensors[f"attention.{start}.values"] = values
    for local in (True, False):
        out = draft.attention(f32(q), block_k, block_v, ctx, [start], local)
        tensors[f"attention.{start}.{'local' if local else 'global'}"] = bf16(out[0])

# silu(gate) * up.
gate = bf16(mx.random.normal((8, 9216)) * 3)
up = bf16(mx.random.normal((8, 9216)) * 2)
tensors["silu.gate"] = gate
tensors["silu.up"] = up
tensors["silu.out"] = bf16(draft.silu_mul(f32(gate), f32(up)))

# The Markov gate: the [y; f] product, the bias and the [state; w1] product, in F32.
context = mx.random.normal((3 * RANK,)) * 2
bias = f32(bf16(mx.random.normal((3 * RANK,)) * 0.5))
raw = mx.random.normal((3 * RANK,)) * 2
state = bf16(mx.random.uniform(-1, 1, (RANK,)))
new_state, latent = draft.gate((bias + context) + raw, f32(state))
tensors["gate.context"] = context
tensors["gate.bias"] = bias
tensors["gate.raw"] = raw
tensors["gate.state"] = state
tensors["gate.state_out"] = bf16(new_state)
tensors["gate.latent"] = bf16(latent)

# Markov logits over a small vocabulary from the F32 fold product, with one clear winner.
vocab = 5000
fold_weight = bf16(mx.random.normal((vocab, RANK)) * 0.05)
fold_bias = bf16(mx.random.normal((vocab,)) * 2)
base = bf16(mx.random.normal((vocab,)) * 8)
base[1234] = 40.0
draft.vocab = vocab
draft.fold_weight, draft.fold_bias = f32(fold_weight), f32(fold_bias)
logits = draft.step_logits(f32(base), f32(latent))
tensors["logits.base"] = base
tensors["logits.fold_bias"] = fold_bias
tensors["logits.acc"] = f32(latent) @ f32(fold_weight).T
tensors["logits.out"] = bf16(logits)
tensors["logits.argmax"] = mx.argmax(logits).astype(mx.int32)[None]

# The load-time fold over 100 LM-head rows.
lm_head = bf16(mx.random.normal((100, HIDDEN)) * 0.02)
hidden_out = bf16(mx.random.normal((HIDDEN, RANK)) * 0.05)
hidden_out_bias = bf16(mx.random.normal((HIDDEN,)) * 0.1)
weight_out, bias_out = fold(lm_head, hidden_out, hidden_out_bias, mx.bfloat16)
tensors["fold.lm_head"] = lm_head
tensors["fold.hidden_out"] = hidden_out
tensors["fold.hidden_out_bias"] = hidden_out_bias
tensors["fold.weight"] = bf16(weight_out)
tensors["fold.bias"] = bf16(bias_out)

save("dspark", tensors)
