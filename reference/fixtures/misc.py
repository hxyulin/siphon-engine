"""MLX outputs for tests/misc.rs."""

import mlx.core as mx
import numpy as np
from mlx_lm.models.inkling2.layers import MessageGate, unweighted_rms_norm

from common import save

HIDDEN = 2304
GATE_RANK = 64
VOCAB = 200_058

mx.random.seed(0)
rng = np.random.default_rng(0)
tensors = {}


def row_scales(shape: tuple) -> mx.array:
    """Per-row magnitudes spread over about two orders of magnitude."""
    return mx.exp(mx.random.normal(shape))


# The combine step of `MessageGate.__call__`, on its own.
h = mx.random.normal((3, HIDDEN)) * row_scales((3, 1))
m = mx.random.normal((3, HIDDEN)) * row_scales((3, 1))
f_pre = mx.random.normal((3, HIDDEN)) * 1.5
g_pre = mx.random.normal((3, HIDDEN)) * 1.5
tensors |= {
    "combine.h": h,
    "combine.m": m,
    "combine.f_pre": f_pre,
    "combine.g_pre": g_pre,
    "combine.out": (1 - mx.tanh(f_pre)) * h + mx.tanh(g_pre) * m,
}

# F32 to BF16: specials, subnormals, exact ties and values across the exponent range.
special = np.array(
    [
        0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF,
        0x00800000, 0x0080FFFF, 0x00808000, 0x3F808000, 0x3F818000, 0x3F808001,
        0x7F7FFFFF, 0xFF7FFFFF, 0x7F7F8000, 0x7F800000, 0xFF800000, 0x7FC00000,
        0x7FC00001, 0x7F800001, 0xFFC12345, 0xFFFFFFFF,
    ],
    np.uint32,
)
random_bits = rng.integers(0, 2**32, 8192, dtype=np.uint32)
ties = (rng.integers(0, 2**16, 2048, dtype=np.uint32) << 16) | 0x8000
spread = rng.standard_normal(8192) * np.exp2(rng.uniform(-140, 124, 8192))
spread = spread.astype(np.float32).view(np.uint32)
cast_bits = np.concatenate([special, random_bits, ties, spread])
cast_x = mx.array(cast_bits.view(np.float32))
tensors["cast_f32_to_bf16.x"] = cast_x
tensors["cast_f32_to_bf16.out"] = cast_x.astype(mx.bfloat16)

# BF16 to F32: every bit pattern.
all_bf16 = mx.array(np.arange(2**16, dtype=np.uint16)).view(mx.bfloat16)
tensors["cast_bf16_to_f32.x"] = all_bf16
tensors["cast_bf16_to_f32.out"] = all_bf16.astype(mx.float32)

# Argmax over the vocabulary. Row 0 is plain logits. Row 1 holds BF16-rounded values
# with the maximum placed at three positions, so the first one must win.
logits = mx.random.normal((2, VOCAB)) * 4
row1 = np.array(logits[1].astype(mx.bfloat16).astype(mx.float32))
top = row1.max() + 1
row1[[123_456, 7_890, 199_999]] = top
logits = mx.concatenate([logits[:1], mx.array(row1)[None]])
tensors["argmax.vocab.x"] = logits
tensors["argmax.vocab.out"] = mx.argmax(logits, axis=-1).astype(mx.int32)

nan, inf = float("nan"), float("inf")
special_rows = mx.array(
    [
        [1, nan, 3, 3, 2, 0, 0, 0],
        [nan, 1, 2, 3, 4, 5, 6, 7],
        [1, 2, nan, nan, 9, 9, 9, 9],
        [-inf] * 8,
        [nan] * 8,
        [-5, -3, -3, -inf, -inf, -9, -4, -3],
        [-1, -0.0, 0.0, -2, -0.0, -3, -4, -5],
        [1, inf, inf, 2, 3, 4, 5, 6],
        [1, inf, inf, nan, 3, 4, 5, 6],
    ]
)
tensors["argmax.special.x"] = special_rows
tensors["argmax.special.out"] = mx.argmax(special_rows, axis=-1).astype(mx.int32)

# The whole message gate, with random weights in the checkpoint's dtypes: BF16
# throughout, with the forget and gain projections kept BF16 by `cast_predicate`.
# MLX promotes those two to F32 for the product with F32 states.
gate = MessageGate(HIDDEN, GATE_RANK)
gate.down_proj.weight = (
    mx.random.normal((GATE_RANK, 2 * HIDDEN)) * (2 * HIDDEN) ** -0.5
).astype(mx.bfloat16)
gate.forget_gate_proj.weight = (mx.random.normal((HIDDEN, GATE_RANK)) * 0.25).astype(
    mx.bfloat16
)
gate.message_gate_proj.weight = (mx.random.normal((HIDDEN, GATE_RANK)) * 0.25).astype(
    mx.bfloat16
)
h = mx.random.normal((1, HIDDEN)) * 3
h = h + (mx.random.uniform(shape=(1, HIDDEN)) < 0.002) * 60.0
m = mx.random.normal((1, HIDDEN)) * 0.7
out = gate(h, m)

# The same chain one step at a time, for the intermediates.
states = mx.concatenate([unweighted_rms_norm(h), unweighted_rms_norm(m)], axis=-1)
down = states.astype(mx.bfloat16) @ gate.down_proj.weight.T
down_f32 = down.astype(mx.float32)
f_pre = down_f32 @ gate.forget_gate_proj.weight.astype(mx.float32).T
g_pre = down_f32 @ gate.message_gate_proj.weight.astype(mx.float32).T
assert mx.array_equal((1 - mx.tanh(f_pre)) * h + mx.tanh(g_pre) * m, out)

tensors |= {
    "gate.h": h,
    "gate.m": m,
    "gate.down_weight": gate.down_proj.weight,
    "gate.forget_weight": gate.forget_gate_proj.weight,
    "gate.gain_weight": gate.message_gate_proj.weight,
    "gate.states": states,
    "gate.down": down,
    "gate.f_pre": f_pre,
    "gate.g_pre": g_pre,
    "gate.out": out,
}

save("misc", tensors)
