"""MLX outputs for tests/experts.rs, from real checkpoint weights.

Writes three files:

- `experts_nvfp4` (layer 0) and `experts_mxfp8` (layer 20): the gate_up and down
  projections for a few rows, as `mx.gather_qmm` computes them in decode. Each output row
  depends only on its own weight row, so the fixture keeps a slice of output rows of a few
  experts. MLX runs on the full projections; the slice is taken afterwards.
- `experts_mlp`: one row through the whole expert MLP (`Experts.__call__`) for layers 0
  and 20. The test reads the weights from the checkpoint, so only inputs and outputs are
  saved.

Weights and scales are the raw checkpoint bytes, saved as U8.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn
from mlx_lm.layers.moe import QuantizedSwitchLinear
from mlx_lm.models.inkling2.layers import Experts, SoftcappedSwiGLU, _softcapped_swiglu

from common import save

MODEL = Path(
    os.environ.get(
        "INKLING_MODEL", "~/.cache/huggingface/local-models/Inkling-2-Mini"
    )
).expanduser()
HIDDEN = 2304
INTERMEDIATE = 1152
GATE_SOFTCAP = 4.0
UP_SOFTCAP = 25.0
SLOTS = 8
# NVFP4 and MXFP8 store 4 and 8 bits per value, with groups of 16 and 32.
FORMATS = {"nvfp4": (4, 16), "mxfp8": (8, 32)}

# Fixture shapes: experts kept, rows of inputs, and the slice of output rows kept.
EXPERTS = 10
ROWS = 3
SLICE = {"nvfp4": (1024, 192), "mxfp8": (1408, 128)}

checkpoint = mx.load(str(MODEL / "model-00001-of-00001.safetensors"))


def projection(layer: int, which: str, experts: mx.array) -> dict[str, mx.array]:
    """Raw checkpoint tensors of `w13` or `w2` for `experts`, plus the NVFP4 global scale."""
    prefix = f"model.llm.layers.{layer}.mlp.experts.{which}_weight"
    tensors = {
        "weight": checkpoint[prefix][experts],
        "scales": checkpoint[prefix + "_scale"][experts],
    }
    if prefix + "_amax" in checkpoint:
        # As `sanitize` computes it.
        scale = 6.0 * 448.0 / checkpoint[prefix + "_outer_scale_denominator"].astype(
            mx.float32
        )
        amax = checkpoint[prefix + "_amax"][experts]
        tensors["global_scale"] = amax.astype(mx.float32) * scale
    return tensors


def gather_qmm(x: mx.array, p: dict[str, mx.array], mode: str, idx: mx.array):
    bits, group_size = FORMATS[mode]
    return mx.gather_qmm(
        x,
        p["weight"].view(mx.uint32),
        p["scales"],
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        global_scale=p.get("global_scale"),
    )


def hidden_rows(rows: int) -> mx.array:
    """BF16 normalized hidden states, with per-row scale and a few outlier channels."""
    x = mx.random.normal((rows, HIDDEN)) * mx.exp(0.3 * mx.random.normal((rows, 1)))
    x = x + (mx.random.uniform(shape=(rows, HIDDEN)) < 0.003) * 12.0
    return x.astype(mx.bfloat16)


def matvec_fixture(layer: int, mode: str) -> None:
    experts = mx.random.permutation(128)[:EXPERTS]
    # Each row picks 8 of the kept experts, in its own order.
    idx = mx.stack([mx.random.permutation(EXPERTS)[:SLOTS] for _ in range(ROWS)])
    idx = idx.astype(mx.uint32)[:, None]
    w13 = projection(layer, "w13", experts)
    w2 = projection(layer, "w2", experts)

    # Shapes as in `Experts.__call__` with B = ROWS and L = 1.
    x = hidden_rows(ROWS)
    gate_up = gather_qmm(x[:, None, None, None], w13, mode, idx)
    # The down projection's input is the real SwiGLU of the gate_up output.
    act = _softcapped_swiglu(
        gate_up[..., 0::2], gate_up[..., 1::2], GATE_SOFTCAP, UP_SOFTCAP
    )
    down = gather_qmm(act, w2, mode, idx)

    start, n = SLICE[mode]
    rows = slice(start, start + n)
    tensors = {
        "indices": idx.reshape(ROWS, SLOTS).astype(mx.int32),
        "gate_up.x": x,
        "gate_up.out": gate_up[..., rows].reshape(ROWS, SLOTS, n),
        "down.x": act.reshape(ROWS, SLOTS, INTERMEDIATE),
        "down.out": down[..., rows].reshape(ROWS, SLOTS, n),
    }
    for name, p in (("gate_up", w13), ("down", w2)):
        tensors[f"{name}.weight"] = p["weight"][:, rows]
        tensors[f"{name}.scales"] = p["scales"][:, rows]
        if "global_scale" in p:
            tensors[f"{name}.global_scale"] = p["global_scale"]
    save(f"experts_{mode}", tensors)


def quantized_switch_linear(p: dict[str, mx.array], mode: str, dims: int, out: int):
    """A `QuantizedSwitchLinear` as `sanitize` builds it, holding the given experts."""
    bits, group_size = FORMATS[mode]
    has_global_scale = "global_scale" in p
    ql = QuantizedSwitchLinear(
        dims,
        out,
        p["weight"].shape[0],
        bias=False,
        group_size=group_size,
        bits=bits,
        mode=mode,
        global_scale=has_global_scale,
    )
    ql.weight = p["weight"].view(mx.uint32)
    ql.scales = p["scales"]
    if has_global_scale:
        ql.global_scale = p["global_scale"]
    return ql


def mlp_fixture() -> None:
    tensors = {}
    for layer, mode in ((0, "nvfp4"), (20, "mxfp8")):
        experts = mx.random.permutation(128)[:SLOTS]
        # `Experts` with only the 8 selected experts loaded; MLX indexes them 0 to 7.
        module = Experts.__new__(Experts)
        nn.Module.__init__(module)
        module.gate_up_proj = quantized_switch_linear(
            projection(layer, "w13", experts), mode, HIDDEN, 2 * INTERMEDIATE
        )
        module.down_proj = quantized_switch_linear(
            projection(layer, "w2", experts), mode, INTERMEDIATE, HIDDEN
        )
        module.act_fn = SoftcappedSwiGLU(
            SimpleNamespace(mlp_gate_softcap=GATE_SOFTCAP, mlp_up_softcap=UP_SOFTCAP)
        )
        x = hidden_rows(1)[None]
        local = mx.arange(SLOTS, dtype=mx.uint32).reshape(1, 1, SLOTS)
        out = module(x, local)
        gate_up = module.gate_up_proj(mx.expand_dims(x, (-2, -3)), local)
        tensors[f"layer{layer}.x"] = x.reshape(1, HIDDEN)
        tensors[f"layer{layer}.indices"] = experts.reshape(1, SLOTS).astype(mx.int32)
        tensors[f"layer{layer}.gate_up"] = gate_up.reshape(1, SLOTS, 2 * INTERMEDIATE)
        tensors[f"layer{layer}.out"] = out.reshape(1, SLOTS, HIDDEN)
    save("experts_mlp", tensors)


if not MODEL.is_dir():
    sys.exit(f"{MODEL} not found; set INKLING_MODEL to the checkpoint directory")
mx.random.seed(0)
matvec_fixture(0, "nvfp4")
matvec_fixture(20, "mxfp8")
mlp_fixture()
