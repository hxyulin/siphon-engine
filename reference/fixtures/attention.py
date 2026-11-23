"""MLX outputs for tests/attention.rs.

One shared stream of keys and values is fed, one token at a time, into a
`RotatingKVCache(1024)` (sliding layers) and a `KVCache` (global layers). At a
few positions each case runs `Attention._attention`, which for one query row
calls `mx.fast.scaled_dot_product_attention` with the relative-bias mask and the
sinks. The positions cross the 1024-token window and the 1024-wide bias table,
and MLX switches from its one-pass to its two-pass kernel at 1024 keys.
"""

from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.inkling2.layers import Attention, L2Clip, RelativeLogits, unweighted_rms_norm

from common import save

HEADS = 24
KV_HEADS = 8
HEAD_DIM = 128
D_REL = 16
R = 1024
WINDOW = 1024
STEPS = 1100
CHECKS = (0, 1, 2, 5, 40, 1023, 1024, 1025, STEPS - 1)
QK_THRESHOLD = 22.627416997969522
REL_THRESHOLD = 8.0
EPS = 1e-6

mx.random.seed(0)
TOPICS = mx.random.normal((4, HEAD_DIM)) * 1.5


def heads(shape: tuple, threshold: float) -> mx.array:
    """BF16 heads L2-clipped to `threshold`, with norms from a third of it up to it.

    Each head mixes in a few shared directions so that some keys line up with a query
    and the softmax has peaks, as in a trained model.
    """
    mix = mx.random.normal(shape[:-1] + (len(TOPICS),)) @ TOPICS[:, : shape[-1]]
    x = mx.random.normal(shape) + mix
    norm = mx.sqrt((x * x).sum(axis=-1, keepdims=True))
    target = threshold * mx.random.uniform(1 / 3, 1.25, shape[:-1] + (1,))
    x = (x * target / norm).astype(mx.bfloat16)
    return L2Clip(threshold)(x)


def values(shape: tuple) -> mx.array:
    v = (mx.random.normal(shape) * 3).astype(mx.bfloat16)
    return unweighted_rms_norm(v.astype(mx.float32), EPS).astype(mx.bfloat16)


cases = {
    "sliding": (RotatingKVCache(max_size=WINDOW), WINDOW),
    "global": (KVCache(), 0),
}
tables = {name: RelativeLogits(D_REL, R) for name in cases}
for table in tables.values():
    # The checkpoint's tables have an RMS near 0.3.
    table.proj = (mx.random.normal((D_REL, R)) * 0.3).astype(mx.bfloat16)

tensors = {f"{name}.table": tables[name].proj for name in cases}
keys, vals = [], []
for step in range(STEPS):
    k = heads((1, KV_HEADS, 1, HEAD_DIM), QK_THRESHOLD)
    v = values((1, KV_HEADS, 1, HEAD_DIM))
    keys.append(k[0, :, 0])
    vals.append(v[0, :, 0])
    for name, (cache, window) in cases.items():
        cached_k, cached_v = cache.update_and_fetch(k, v)
        mx.eval(cached_k, cached_v)
        if step not in CHECKS:
            continue
        q = heads((1, HEADS, 1, HEAD_DIM), QK_THRESHOLD)
        rel = heads((1, 1, HEADS, D_REL), REL_THRESHOLD)
        sinks = (mx.random.normal((1, 1, HEADS)) * 2).astype(mx.bfloat16)
        bias = tables[name](rel)
        layer = SimpleNamespace(scaling=HEAD_DIM**-0.5, sliding_window=window)
        out = Attention._attention(layer, q, cached_k, cached_v, bias, sinks, cache)
        prefix = f"{name}.{step}"
        tensors[f"{prefix}.q"] = q.reshape(HEADS, HEAD_DIM)
        tensors[f"{prefix}.rel"] = rel.reshape(HEADS, D_REL)
        tensors[f"{prefix}.sinks"] = sinks.reshape(HEADS)
        tensors[f"{prefix}.bias"] = bias.reshape(HEADS, R)
        tensors[f"{prefix}.out"] = out.reshape(HEADS, HEAD_DIM)

tensors["k"] = mx.stack(keys)
tensors["v"] = mx.stack(vals)
tensors["checks"] = mx.array(CHECKS, dtype=mx.int32)
save("attention", tensors)
