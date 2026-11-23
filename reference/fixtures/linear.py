"""MLX outputs for tests/linear.rs.

The weights come from an integer hash that tests/linear.rs repeats, so the fixture
stores only inputs and outputs and can cover full model shapes. Every output row is
its own M = 1 product: MLX multiplies two or more F32 rows with a TF32 kernel, and
decode multiplies one.
"""

import mlx.core as mx
import numpy as np
from mlx import nn

from common import save


def hash_weight_bits(seed: int, n: int, k: int) -> np.ndarray:
    """BF16 bit patterns of an `[n, k]` weight, with magnitudes from 2^-10 to 2^-2."""
    x = np.arange(n * k, dtype=np.uint32) ^ np.uint32((seed * 0x9E3779B9) & 0xFFFFFFFF)
    x ^= x >> 16
    x *= np.uint32(0x7FEB352D)
    x ^= x >> 15
    x *= np.uint32(0x846CA68B)
    x ^= x >> 16
    sign = (x >> 31) << 15
    exponent = (np.uint32(117) + ((x >> 28) & 7)) << 7
    mantissa = (x >> 16) & 0x7F
    return (sign | exponent | mantissa).astype(np.uint16).reshape(n, k)


def gemv_rows(x: mx.array, w: mx.array) -> mx.array:
    return mx.concatenate([x[r : r + 1] @ w.T for r in range(x.shape[0])])


# (name, n, k, rows, dtype). The comments give MLX's blocking for each shape.
CASES = [
    ("bf16.q_proj", 3072, 2304, 1, mx.bfloat16),  # one simdgroup per row
    ("bf16.kv_proj", 1024, 2304, 1, mx.bfloat16),
    ("bf16.r_proj", 384, 2304, 1, mx.bfloat16),
    ("bf16.sink_proj", 24, 2304, 1, mx.bfloat16),  # split k
    ("bf16.o_proj", 2304, 3072, 1, mx.bfloat16),
    ("bf16.router", 32, 2304, 1, mx.bfloat16),
    ("bf16.gate_down", 64, 4608, 1, mx.bfloat16),
    ("bf16.lm_head", 4096, 2304, 1, mx.bfloat16),
    ("bf16.multi_row", 3072, 2304, 3, mx.bfloat16),
    ("bf16.leftover", 100, 192, 2, mx.bfloat16),  # k not a multiple of 128
    ("bf16.split_leftover", 8, 320, 1, mx.bfloat16),  # split k, all leftover
    ("bf16.ragged_n", 1001, 2304, 1, mx.bfloat16),
    ("bf16.narrow", 512, 64, 2, mx.bfloat16),  # 4 lanes per row
    ("f32.gate_up", 2304, 64, 1, mx.float32),
    ("f32.gate_up_rows", 2304, 64, 3, mx.float32),
    ("f32.wide", 300, 2304, 1, mx.float32),
    ("f32.split", 24, 2304, 1, mx.float32),
    ("f32.split_leftover", 8, 320, 1, mx.float32),
]

mx.random.seed(0)
tensors = {}
for seed, (name, n, k, rows, dtype) in enumerate(CASES):
    bits = hash_weight_bits(seed, n, k)
    w = mx.array(bits).view(mx.bfloat16).astype(dtype)
    x = mx.random.normal((rows, k)).astype(dtype)
    out = gemv_rows(x, w)
    tensors[f"{name}.seed"] = mx.array([seed], mx.int32)
    tensors[f"{name}.x"] = x
    tensors[f"{name}.out"] = out
    # A sample of the weight, to catch a mismatch between the two hash implementations.
    flat = w.reshape(-1).astype(mx.bfloat16)
    tensors[f"{name}.w_check"] = mx.concatenate([flat[:32], flat[-32:]])

# nn.Linear on a [B, L, k] input takes the same path as the 2-D product above.
name, n, k, _, _ = CASES[0]
linear = nn.Linear(k, n, bias=False)
linear.weight = mx.array(hash_weight_bits(0, n, k)).view(mx.bfloat16)
assert mx.array_equal(linear(tensors[f"{name}.x"][None]), tensors[f"{name}.out"][None])

save("linear", tensors)
