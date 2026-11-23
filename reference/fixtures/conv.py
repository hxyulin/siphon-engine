"""MLX outputs for tests/conv.rs.

Each case runs a prefill of PREFILL rows from a zero state, then one decode row
that starts from the state the prefill returned. The decode output is also
checked against MLX run on all PREFILL + 1 rows at once, so the carried state
reproduces the unsplit sequence.
"""

import mlx.core as mx

from common import save
from mlx_lm.models.inkling2.layers import (
    DynamicShortConvolution,
    ShortConvolution,
    _conv_state,
)

PREFILL = 10
KERNEL = 4
# Hidden width, which feeds weight_proj in the dynamic convolution.
HIDDEN = 2304
KV_DIMS = 1024
KV_HEADS = 8


def run(call, x, *args):
    """Prefill, then decode the last row from the prefill state; returns the tensors to save."""
    prefill_out, window = call(x[:, :PREFILL], *(a[:, :PREFILL] for a in args), None, None)
    prefill_state = _conv_state(window, None, KERNEL - 1)
    decode_out, window = call(x[:, PREFILL:], *(a[:, PREFILL:] for a in args), prefill_state, None)
    decode_state = _conv_state(window, None, KERNEL - 1)
    full_out, _ = call(x, *args, None, None)
    assert mx.array_equal(full_out[:, PREFILL:], decode_out).item()
    return {
        "prefill.out": prefill_out[0],
        "prefill.state": prefill_state[0],
        "decode.out": decode_out[0],
        "decode.state": decode_state[0],
    }


def short_conv(dims: int) -> dict[str, mx.array]:
    conv = ShortConvolution(dims, KERNEL)
    conv.conv1d.weight = (mx.random.normal((dims, KERNEL, 1)) * 0.4).astype(mx.bfloat16)
    x = (mx.random.normal((1, PREFILL + 1, dims)) * 2).astype(mx.bfloat16)
    tensors = {"x": x[0], "weight": conv.conv1d.weight, **run(conv, x)}
    return {f"short_conv_{dims}.{k}": v for k, v in tensors.items()}


def dynamic_short_conv() -> dict[str, mx.array]:
    conv = DynamicShortConvolution(KV_DIMS, HIDDEN, KV_HEADS, KERNEL)
    # F32 and not BF16-rounded, so the test covers a full-precision static weight.
    conv.static_weight = mx.random.normal((KERNEL, KV_DIMS)) * 0.4
    conv.weight_proj.weight = (
        mx.random.normal((KV_HEADS * KERNEL, HIDDEN)) * HIDDEN**-0.5 * 0.3
    ).astype(mx.bfloat16)
    x = (mx.random.normal((1, PREFILL + 1, KV_DIMS)) * 2).astype(mx.bfloat16)
    hidden = mx.random.normal((1, PREFILL + 1, HIDDEN)).astype(mx.bfloat16)
    tensors = {
        "x": x[0],
        # The engine runs weight_proj as a separate GEMM, so the kernel reads its output.
        "dynamic": conv.weight_proj(hidden)[0],
        "static_weight": conv.static_weight,
        **run(conv, x, hidden),
    }
    return {f"dynamic_short_conv.{k}": v for k, v in tensors.items()}


mx.random.seed(0)
save("conv", {**short_conv(HIDDEN), **short_conv(KV_DIMS), **dynamic_short_conv()})
