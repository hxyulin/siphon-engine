"""MLX outputs for tests/norm.rs."""

import mlx.core as mx

from common import save

mx.random.seed(0)
x = (mx.random.normal((37, 2304)) * 2).astype(mx.bfloat16)
weight = (mx.random.normal((2304,)) * 0.5 + 1).astype(mx.bfloat16)
save(
    "norm",
    {
        "rms_norm.x": x,
        "rms_norm.weight": weight,
        "rms_norm.out": mx.fast.rms_norm(x, weight, 1e-6),
    },
)
