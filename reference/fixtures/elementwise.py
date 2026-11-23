"""MLX outputs for tests/elementwise.rs.

Runs the MLX-LM code on the GPU. Where the operation is a method body
(`SparseMoeBlock.__call__`, `Model.__call__`), the method runs on a stub `self` whose
submodules return the test inputs. Where it is a few inline lines of a larger method,
the lines are copied here with the method named.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm.models.inkling2.layers import SparseMoeBlock, _positions, _softcapped_swiglu
from mlx_lm.models.inkling2.model import Model

from common import save

HIDDEN = 2304
EXPERT_WIDTH = 1152
HEADS, HEAD_DIM, D_REL = 24, 128, 16
TOP_K = 8
PADDED_VOCAB, VOCAB = 201024, 200058
GATE_SOFTCAP, UP_SOFTCAP, LOGIT_SOFTCAP = 4.0, 25.0, 30.0
ALPHA, N_FLOOR = 0.1, 8192

mx.random.seed(0)
tensors: dict[str, mx.array] = {}


def normal(shape, scale=1.0) -> mx.array:
    """BF16 normal samples whose scale varies per row, so some rows saturate the softcaps."""
    row_scale = mx.exp(mx.random.uniform(-1, 1, (*shape[:-1], 1)) * 2) * scale
    return (mx.random.normal(shape) * row_scale).astype(mx.bfloat16)


# Embedding gather (`TextModel.embed` before `embed_norm`).
embed = nn.Embedding(16, HIDDEN)
embed.weight = normal((16, HIDDEN))
ids = mx.array([3, 0, 15, 7, 3, 9, 0, 12], dtype=mx.int32)
tensors |= {
    "embedding.table": embed.weight,
    "embedding.ids": ids,
    "embedding.out": embed(ids),
}

# Softcapped SwiGLU on the interleaved projection, sliced as in `Experts.__call__`.
gate_up = normal((12, 2 * EXPERT_WIDTH), 8.0)
tensors |= {
    "swiglu.gate_up": gate_up,
    "swiglu.out": _softcapped_swiglu(
        gate_up[..., 0::2], gate_up[..., 1::2], GATE_SOFTCAP, UP_SOFTCAP
    ),
}

# Attention output gate, the last lines of `Attention._attend`.
attn = normal((4, HEADS * HEAD_DIM))
gate = normal((4, HEADS * HEAD_DIM), 3.0)
gated = attn.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))
tensors |= {
    "attention_gate.attn": attn,
    "attention_gate.gate": gate,
    "attention_gate.out": gated.astype(gate.dtype),
}


# Log-scaling: tau from `FullAttention.__call__`, applied as in `Attention._attend`.
def log_scale(name: str, offset: int, length: int) -> None:
    position_ids = _positions(offset, length).astype(mx.float32)
    tau = 1.0 + ALPHA * mx.log(mx.maximum((position_ids + 1) / N_FLOOR, 1.0))
    tau = tau[..., None, None]
    query = normal((length, HEADS, HEAD_DIM))
    relative = normal((length, HEADS, D_REL))
    tensors.update(
        {
            f"{name}.query": query,
            f"{name}.relative": relative,
            f"{name}.query_out": (query.astype(mx.float32) * tau).astype(query.dtype),
            f"{name}.relative_out": (relative.astype(mx.float32) * tau).astype(
                relative.dtype
            ),
        }
    )


# Positions 8188 to 8195 cross the floor, where tau leaves 1.
log_scale("log_scale.low", 8188, 8)
log_scale("log_scale.high", 1_000_000, 2)

# Residual adds in `DecoderLayer.__call__`: block_residual + attention output in F32,
# then residual + MLP output rounded to BF16.
a, b = normal((2, HIDDEN), 4.0), normal((2, HIDDEN))
residual = mx.random.normal((2, HIDDEN)) * 8
x = normal((2, HIDDEN))
tensors |= {
    "add_bf16_to_f32.a": a,
    "add_bf16_to_f32.b": b,
    "add_bf16_to_f32.out": a.astype(mx.float32) + b,
    "add_to_bf16.residual": residual,
    "add_to_bf16.x": x,
    "add_to_bf16.out": (residual + x.astype(mx.float32)).astype(mx.bfloat16),
}

# MoE combine: `SparseMoeBlock.__call__` with the router and experts stubbed out.
rows = 3
hidden = mx.zeros((1, rows, HIDDEN), mx.bfloat16)
expert_states = normal((1, rows, TOP_K, HIDDEN))
scores = mx.random.uniform(0.01, 1.0, (1, rows, TOP_K))
routing_weights = scores / scores.sum(axis=-1, keepdims=True) * 2.5
block = SimpleNamespace(
    sharding_group=None,
    gate=lambda h: (mx.zeros((1, rows, TOP_K), mx.int32), routing_weights),
    experts=lambda h, indices: expert_states,
)
tensors |= {
    "moe_combine.experts": expert_states[0],
    "moe_combine.weights": routing_weights[0],
    "moe_combine.out": SparseMoeBlock.__call__(block, hidden)[0],
}


# Final logit softcap: `Model.__call__` with the language model and LM head stubbed out.
# Two full-width rows would take 1.6 MB of F32 output, so the rows use every finite
# BF16 value and the fixture stores the output per value: tests/elementwise.rs rebuilds
# the same rows from `finite_bf16_bits` and looks each output up by input value.
def finite_bf16_bits(k: np.ndarray) -> np.ndarray:
    """The k-th finite BF16 bit pattern: positives 0x0000..0x7F7F, then negatives."""
    return np.where(k < 0x7F80, k, k - 0x7F80 + 0x8000).astype(np.uint16)


N_FINITE = 2 * 0x7F80
columns = np.arange(PADDED_VOCAB)
k = np.stack([(columns * 7 + row * 4093) % N_FINITE for row in range(2)])
logits = mx.array(finite_bf16_bits(k)).view(mx.bfloat16)[None]
model = SimpleNamespace(
    model=SimpleNamespace(language_model=lambda inputs, cache, input_embeddings: None),
    lm_head=lambda h: logits,
    sharding_group=None,
    args=SimpleNamespace(
        text=SimpleNamespace(
            final_logit_softcapping=LOGIT_SOFTCAP, unpadded_vocab_size=VOCAB
        )
    ),
)
softcapped = np.array(Model.__call__(model, mx.zeros((1, 2), mx.int32))[0])
assert softcapped.shape == (2, VOCAB)
table = np.empty(N_FINITE, np.float32)
table[k[:, :VOCAB]] = softcapped
assert np.array_equal(table[k[:, :VOCAB]], softcapped), "softcap is not elementwise"
assert len(np.unique(k[:, :VOCAB])) == N_FINITE, "rows miss some BF16 values"
tensors["logit_softcap.table"] = mx.array(table)

save("elementwise", tensors)
