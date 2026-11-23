"""MLX outputs for tests/routing.rs."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx_lm.kernels.expert_routing import expert_routing
from mlx_lm.models.inkling2.layers import (
    BlockAttentionResidual,
    MessageInput,
    TopKRouter,
    _depth_attention,
)

from common import save

HIDDEN = 2304
EXPERTS = 128
TOP_K = 8
ROUTE_SCALE = 8.0
# Expert e and e ^ PARTNER share a bias, so equal logits give an exact tie.
PARTNER = 64
# Experts whose router weight row is copied to their partner's row.
TIED_ROWS = [3, 10, 20, 40]

assert mx.default_device() == mx.gpu, "MLX must run the Metal kernels"


def router_weight() -> np.ndarray:
    """Pseudo-random F32 router weight with full mantissas, std about 0.017.

    A stored [128, 2304] F32 weight would exceed the 1 MB fixture budget, so
    tests/routing.rs regenerates the same values with the same integer hash.
    Every step is exact or a single correctly rounded F32 operation.
    """
    h = np.arange(EXPERTS * HIDDEN, dtype=np.uint32)
    h = h * np.uint32(0x9E3779B1)
    h ^= h >> np.uint32(15)
    h = h * np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(12)
    unit = (h >> np.uint32(8)).astype(np.float32) * np.float32(2.0**-24)
    w = (unit - np.float32(0.5)) * np.float32(0.06)
    w = w.astype(np.float32).reshape(EXPERTS, HIDDEN)
    for e in TIED_ROWS:
        w[e ^ PARTNER] = w[e]
    return w


def router_bias() -> mx.array:
    half = mx.random.normal((PARTNER,)) * 0.05
    return mx.concatenate([half, half]).astype(mx.float32)


mx.random.seed(0)
tensors: dict[str, mx.array] = {}

# TopKRouter end to end, for a prefill-sized batch and a single decode row.
# On the M5, MLX's F32 GEMM for two or more rows truncates its inputs to TF32,
# while its GEMV for one row keeps full F32, so the two cases round
# differently.
args = SimpleNamespace(
    num_experts_per_tok=TOP_K,
    route_scale=ROUTE_SCALE,
    n_routed_experts=EXPERTS,
    hidden_size=HIDDEN,
)
weight = router_weight()
router = TopKRouter(args)
router.weight = mx.array(weight)
router.e_score_correction_bias = router_bias()
tensors["router.bias"] = router.e_score_correction_bias
tensors["router.weight_rows"] = router.weight[::64]
for name, rows in [("router16", 16), ("router1", 1)]:
    hidden = mx.random.normal((1, rows, HIDDEN)).astype(mx.bfloat16)
    indices, weights = router(hidden)
    tensors[f"{name}.hidden"] = hidden
    tensors[f"{name}.logits"] = hidden.astype(mx.float32) @ router.weight.T
    tensors[f"{name}.indices"] = indices.astype(mx.int32)
    tensors[f"{name}.weights"] = weights


def place_pair(row: np.ndarray, bias: np.ndarray, e: int, above: int) -> int:
    """Gives the first pair from e on whose bias allows it (experts e and
    e ^ PARTNER) one logit that ranks it just below `above` other experts, so
    the pair takes places above + 1 and above + 2. Equal logit and bias make
    the biased scores tie exactly. Returns the pair's lower expert."""
    for e in range(e, e + PARTNER):
        e %= PARTNER
        others = np.delete(np.arange(EXPERTS), [e, e ^ PARTNER])
        scores = 1 / (1 + np.exp(-row[others].astype(np.float64))) + bias[others]
        ranked = np.sort(scores)[::-1]
        s = (ranked[above - 1] + ranked[above]) / 2 - bias[e]
        if 0.01 < s < 0.99:
            row[e] = row[e ^ PARTNER] = np.float32(np.log(s / (1 - s)))
            return e
    raise ValueError("no pair fits")


# The expert_routing kernel alone, on logits with exact ties. Row 0 has zero
# logits, so the bias alone ranks the experts and every bias pair ties. Every
# other row ties a pair across the 8th and 9th place, odd rows also tie a
# pair in 4th and 5th place, and every third row breaks the boundary tie by
# one ulp of the logit.
bias = np.array(tensors["router.bias"])
logits = np.array(mx.random.normal((12, EXPERTS)), dtype=np.float32)
logits[0] = 0
for r in range(1, logits.shape[0]):
    # The upper pair goes first so that it cannot push the boundary pair down.
    if r % 2:
        place_pair(logits[r], bias, 5 * r + 17, 3)
    e = place_pair(logits[r], bias, 5 * r, TOP_K - 1)
    if r % 3 == 0:
        logits[r, e ^ PARTNER] = np.nextafter(logits[r, e], np.float32(np.inf))
logits = mx.array(logits)
bias = tensors["router.bias"]
indices, weights = expert_routing(
    logits,
    bias,
    top_k=TOP_K,
    multiplier=1.0,
    score_eps=1e-20,
    route_scale=ROUTE_SCALE,
)
tensors["ties.logits"] = logits
tensors["ties.indices"] = indices.astype(mx.int32)
tensors["ties.weights"] = weights


def states(blocks: int, rows: int) -> tuple[mx.array, mx.array]:
    """Unit-RMS F32 states like the normed residuals the model stacks."""
    s = mx.fast.rms_norm(mx.random.normal((blocks + 1, 1, rows, HIDDEN)), None, 1e-6)
    return s[:-1], s[-1]


# BlockAttentionResidual with one query (the MLP input) and with three (the
# attention input), and MessageInput with its four queries per looped layer.
for name, blocks, rows, queries in [("depth_a", 2, 3, 1), ("depth_b", 4, 1, 3)]:
    block_states, partial = states(blocks, rows)
    module = BlockAttentionResidual(HIDDEN, queries)
    module.query = mx.random.normal(module.query.shape) * 1.5
    tensors[f"{name}.blocks"] = block_states
    tensors[f"{name}.partial"] = partial
    tensors[f"{name}.queries"] = module.query.reshape(-1, HIDDEN)
    tensors[f"{name}.out"] = module(block_states, partial)

block_states, partial = states(9, 1)
message = MessageInput(HIDDEN, 2)
message.query = mx.random.normal(message.query.shape) * 1.5
tensors["depth_c.blocks"] = block_states
tensors["depth_c.partial"] = partial
tensors["depth_c.queries"] = message.query[1]
tensors["depth_c.out"] = message((block_states, partial), 1)
# One query so strong that one candidate takes nearly all the weight.
tensors["depth_d.queries"] = message.query[1] * 20
tensors["depth_d.out"] = _depth_attention(
    block_states, partial, tensors["depth_d.queries"]
)

mx.eval(tensors)
save("routing", tensors)
