"""MLX outputs for tests/norm.rs."""

import mlx.core as mx
from mlx import nn
from mlx_lm.models.inkling2.layers import L2Clip, ScaledL2Clip, unweighted_rms_norm

from common import save

EPS = 1e-6
QK_THRESHOLD = 22.627416997969522
REL_THRESHOLD = 8.0

mx.random.seed(0)
x = (mx.random.normal((37, 2304)) * 2).astype(mx.bfloat16)
weight = (mx.random.normal((2304,)) * 0.5 + 1).astype(mx.bfloat16)
tensors = {
    "rms_norm.x": x,
    "rms_norm.weight": weight,
    "rms_norm.out": mx.fast.rms_norm(x, weight, EPS),
}


def row_scales(shape: tuple) -> mx.array:
    """Per-row magnitudes spread over about two orders of magnitude."""
    return mx.exp(mx.random.normal(shape))


# F32 residual-stream rows, with a few large outlier channels as in real hidden states.
rows = mx.random.normal((7, 2304)) * row_scales((7, 1))
rows = rows + (mx.random.uniform(shape=(7, 2304)) < 0.002) * 60.0
tensors["unweighted_rms_norm.x"] = rows
tensors["unweighted_rms_norm.out"] = unweighted_rms_norm(rows)

# `input_layernorm` and friends: F32 input, BF16 weight, output cast to the model dtype.
norm = nn.RMSNorm(2304, eps=EPS)
norm.weight = (mx.random.normal((2304,)) * 0.3 + 1).astype(mx.bfloat16)
rows = mx.random.normal((7, 2304)) * row_scales((7, 1))
tensors["rms_norm_f32.x"] = rows
tensors["rms_norm_f32.weight"] = norm.weight
tensors["rms_norm_f32.out"] = norm(rows).astype(mx.bfloat16)

# Value states `[B, L, kv_heads, head_dim]`, normalized as in `Attention._attend`.
values = (mx.random.normal((1, 6, 8, 128)) * row_scales((1, 6, 8, 1))).astype(
    mx.bfloat16
)
tensors["unweighted_rms_norm_heads.x"] = values
tensors["unweighted_rms_norm_heads.out"] = unweighted_rms_norm(
    values.astype(mx.float32), EPS
).astype(values.dtype)


def clip_input(length: int, heads: int, dim: int, threshold: float) -> mx.array:
    """Heads whose norms spread from a quarter to four times the threshold."""
    per_element = threshold / dim**0.5
    spread = mx.exp(mx.random.uniform(-1.4, 1.4, (1, length, heads, 1)))
    states = mx.random.normal((1, length, heads, dim)) * per_element * spread
    return states.astype(mx.bfloat16)


# Decode (L = 1) and prefill lengths: MLX reduces fewer than 32 rows with a different
# kernel, and the CUDA kernel must match both.
for length in (1, 6):
    q_clip = ScaledL2Clip(24, 128, QK_THRESHOLD)
    q_clip.scale = (mx.random.normal((24, 128)) * 0.2 + 1).astype(mx.bfloat16)
    q = clip_input(length, 24, 128, QK_THRESHOLD)
    tensors[f"q_l2_clip.{length}.x"] = q
    tensors[f"q_l2_clip.{length}.out"] = q_clip(q)

    k = clip_input(length, 8, 128, QK_THRESHOLD)
    tensors[f"k_l2_clip.{length}.x"] = k
    tensors[f"k_l2_clip.{length}.out"] = L2Clip(QK_THRESHOLD)(k)

    rel_clip = ScaledL2Clip(24, 16, REL_THRESHOLD)
    rel_clip.scale = (mx.random.normal((24, 16)) * 0.2 + 1).astype(mx.bfloat16)
    rel = clip_input(length, 24, 16, REL_THRESHOLD)
    tensors[f"rel_l2_clip.{length}.x"] = rel
    tensors[f"rel_l2_clip.{length}.out"] = rel_clip(rel)

    tensors[f"q_l2_clip.{length}.scale"] = q_clip.scale
    tensors[f"rel_l2_clip.{length}.scale"] = rel_clip.scale

save("norm", tensors)
