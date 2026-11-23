"""Full-precision F32 products for MLX-LM's Inkling 2 model, for `capture_eval.py --float32`.

MLX's F32 matmul on Apple GPUs (measured on an M5 Max) keeps only about 11 bits of
inputs that are not BF16 values, so an F32 capture would run its products at roughly
TF32 precision. Single-row products (decode steps) keep full precision, multi-row ones
(chunks) do not. `exact_products(model)` patches the operations that lose precision so
that every operand of a matmul is BF16-exact: an operand that may not be a BF16 value
splits into three BF16 parts, as `dspark.matmul` does. The experts keep their packed
weights: `gather_qmm` without the NVFP4 global scale multiplies BF16-valued inputs
exactly, and the global scale multiplies the sum once.

Run as a script, it records the inputs of every matmul-like operation of a chunk and of
a few decode steps in F32, reports the relative error of the stock and the patched
operation against an F64 NumPy computation, and exits with status 1 if any operation
the patched model runs is further than 1e-6 from F64:

    uv run --project reference python -I reference/exact_products.py <model-dir> reference/prompts/anyhow.txt
"""

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.layers.moe import QuantizedSwitchLinear
from mlx_lm.models.inkling2 import layers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dspark import matmul, split_bf16  # noqa: E402

# MLX's NVFP4 dequantization divides the global scale by the largest E2M1 value times
# the largest E4M3 value.
NVFP4_GLOBAL_DIVISOR = 6.0 * 448.0


def bf16_exact(x: mx.array) -> bool:
    return bool(mx.all(x == x.astype(mx.bfloat16).astype(x.dtype)).item())


def linear(self: nn.Linear, x: mx.array) -> mx.array:
    y = matmul(x, self["weight"].T, b_exact=True)
    return y + self["bias"] if "bias" in self else y


def relative_logits(self: layers.RelativeLogits, relative_states: mx.array) -> mx.array:
    return matmul(
        relative_states, self.proj.astype(relative_states.dtype), b_exact=True
    )


def router_logits(hidden_states: mx.array, weight: mx.array) -> mx.array:
    return matmul(hidden_states.astype(mx.float32), weight.T, b_exact=True)


def router(self: layers.TopKRouter, hidden_states: mx.array):
    """`TopKRouter.__call__` with exact logits."""
    logits = router_logits(hidden_states, self.weight)
    if layers._use_kernels():
        return layers.expert_routing(
            logits,
            self.e_score_correction_bias,
            top_k=self.top_k,
            multiplier=1.0,
            score_eps=layers._ROUTER_SCORE_EPS,
            route_scale=self.routed_scaling_factor,
        )
    scores = mx.sigmoid(logits)
    biased = scores + self.e_score_correction_bias
    k = self.top_k
    indices = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + layers._ROUTER_SCORE_EPS)
    return indices, weights * self.routed_scaling_factor


def experts(self: QuantizedSwitchLinear, x, indices, sorted_indices=False):
    """`QuantizedSwitchLinear.__call__` with BF16-exact parts of `x`. NVFP4 and MXFP8
    weights times their block scales are BF16 values; the NVFP4 global scale multiplies
    the sum instead of the weights."""
    if self.mode not in ("nvfp4", "mxfp8") or "bias" in self:
        raise ValueError(f"no exact product for {self.mode} experts")
    out = None
    for part in split_bf16(x.astype(mx.float32)):
        y = mx.gather_qmm(
            part,
            self["weight"],
            self["scales"],
            None,
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        out = y if out is None else out + y
    if self.global_scale is not None:
        scale = self.global_scale / NVFP4_GLOBAL_DIVISOR
        out = out * scale[indices][..., None, None]
    return out


def attention_reference(
    query,
    key,
    value,
    position_bias,
    sinks,
    *,
    scale,
    window,
    query_positions,
    key_positions,
):
    """`layers._attention_reference` with exact products."""
    B, Hq, L, D = query.shape
    Hkv, S = key.shape[1], key.shape[2]
    R = position_bias.shape[-1]
    distance = query_positions[..., :, None] - key_positions[..., None, :]
    visible = (distance >= 0) & (key_positions[..., None, :] >= 0)
    if window:
        visible = visible & (distance < window)
    if distance.ndim == 3:
        distance, visible = distance[:, None], visible[:, None]
    bias = mx.take_along_axis(
        position_bias.transpose(0, 2, 1, 3),
        mx.broadcast_to(mx.clip(distance, 0, R - 1), (B, Hq, L, S)),
        axis=-1,
    )
    bias = mx.where(distance < R, bias.astype(mx.float32), 0.0)
    attention_mask = mx.where(visible, bias, -mx.inf).reshape(B, Hkv, -1, L, S)

    query = query.astype(mx.float32).reshape(B, Hkv, -1, L, D)
    key = key.astype(mx.float32)[:, :, None]
    attn_weights = matmul(query, key.swapaxes(-1, -2)) * scale + attention_mask
    sinks = sinks.astype(mx.float32).transpose(0, 2, 1).reshape(B, Hkv, -1, L, 1)
    combined_logits = mx.concatenate([attn_weights, sinks], axis=-1)
    attn_weights = mx.softmax(combined_logits, axis=-1)[..., :S]
    attn_output = matmul(attn_weights, value.astype(mx.float32)[:, :, None])
    return attn_output.reshape(B, Hq, L, D).transpose(0, 2, 1, 3)


PATCHES = [
    (nn.Linear, "__call__", linear),
    (layers.RelativeLogits, "__call__", relative_logits),
    (layers.TopKRouter, "__call__", router),
    (QuantizedSwitchLinear, "__call__", experts),
    # The multi-row path; the fused single-row attention is already exact.
    (layers, "_attention_reference", attention_reference),
]


@contextmanager
def exact_products(model: nn.Module):
    """Runs the model's products at full F32 precision inside the block."""
    for path, module in model.named_modules():
        weight = getattr(module, "weight", None)
        patched = isinstance(module, (nn.Linear, layers.TopKRouter))
        if patched and not bf16_exact(weight):
            raise ValueError(f"{path}.weight is not BF16-exact")
        if isinstance(module, layers.RelativeLogits) and not bf16_exact(module.proj):
            raise ValueError(f"{path}.proj is not BF16-exact")
    saved = [(owner, name, getattr(owner, name)) for owner, name, _ in PATCHES]
    try:
        for owner, name, fn in PATCHES:
            setattr(owner, name, fn)
        yield
    finally:
        for owner, name, fn in saved:
            setattr(owner, name, fn)


# The measurement below: every recorded operation is run stock, patched, and in F64.

TOLERANCE = 1e-6


def f64(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32), dtype=np.float64)


def rel_err(y: mx.array, ref: np.ndarray) -> float:
    return float(np.linalg.norm(f64(y).ravel() - ref.ravel()) / np.linalg.norm(ref))


def bf16_rounding(ref: np.ndarray) -> float:
    return rel_err(mx.array(ref.astype(np.float32)).astype(mx.bfloat16), ref)


def softmax64(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention64(q, k, v, bias, sinks, scale, window, qpos, kpos) -> np.ndarray:
    """F64 `_attention_reference`, for batch 1."""
    q, k, v, bias, sinks = map(f64, (q, k, v, bias, sinks))
    qpos, kpos = np.array(qpos).reshape(-1), np.array(kpos).reshape(-1)
    _, Hq, L, D = q.shape
    Hkv, S, R = k.shape[1], k.shape[2], bias.shape[-1]
    distance = qpos[:, None] - kpos[None, :]
    visible = (distance >= 0) & (kpos[None, :] >= 0)
    if window:
        visible &= distance < window
    # bias is [1, L, Hq, R]
    gathered = np.take_along_axis(
        bias[0].transpose(1, 0, 2),
        np.broadcast_to(np.clip(distance, 0, R - 1), (Hq, L, S)),
        -1,
    )
    gathered = np.where(distance < R, gathered, 0.0)
    mask = np.where(visible, gathered, -np.inf)
    g = Hq // Hkv
    kk = np.repeat(k[0], g, axis=0)
    vv = np.repeat(v[0], g, axis=0)
    logits = q[0] @ kk.transpose(0, 2, 1) * scale + mask
    sink = sinks[0].transpose(1, 0)[:, :, None]
    weights = softmax64(np.concatenate([logits, sink], -1))[..., :S]
    return (weights @ vv).transpose(1, 0, 2)[None]


def experts64(
    self: QuantizedSwitchLinear, x: mx.array, indices: mx.array
) -> np.ndarray:
    """F64 `gather_qmm` over every expert `indices` selects, from exactly dequantized
    weights."""
    x64 = f64(x)
    idx = np.array(indices)
    out_shape = np.broadcast_shapes(x64.shape[:-2], idx.shape) + (
        1,
        self["weight"].shape[1],
    )
    xb = np.broadcast_to(
        x64, np.broadcast_shapes(x64.shape[:-2], idx.shape) + x64.shape[-2:]
    )
    ib = np.broadcast_to(idx, out_shape[:-2])
    out = np.zeros(out_shape)
    for e in np.unique(ib):
        w = mx.dequantize(
            self["weight"][int(e)],
            self["scales"][int(e)],
            None,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            dtype=mx.float32,
        )
        w64 = f64(w)
        if self.global_scale is not None:
            w64 = w64 * (float(self.global_scale[int(e)]) / NVFP4_GLOBAL_DIVISOR)
        sel = ib == e
        out[sel] = xb[sel] @ w64.T
    return out


def measure(args: argparse.Namespace) -> None:
    from mlx_lm import load

    model, tokenizer = load(str(args.model))
    model.set_dtype(mx.float32)
    tokens = tokenizer.encode(args.text.read_text(), add_special_tokens=False)

    # Records, per operation and shape class, the call at visit `args.visit` (or the
    # last one, if there are fewer).
    calls: dict[str, int] = {}
    rec: dict[str, tuple] = {}

    def take(name: str) -> bool:
        calls[name] = calls.get(name, 0) + 1
        return calls[name] <= args.visit

    stock = {(o, n): getattr(o, n) for o, n, _ in PATCHES}
    stock_fast_rms = mx.fast.rms_norm
    stock_conv = nn.Conv1d.__call__
    stock_depth = layers.depth_attention
    stock_attention = layers.Attention._attention

    def rec_linear(self, x):
        out_dims, in_dims = self["weight"].shape
        proj = {3072: "o_proj", 64: "message gate"}.get(in_dims)
        proj = proj or {3072: "q_proj", 1024: "k_proj"}.get(out_dims)
        if out_dims > 100000:
            proj = "lm_head"
        name = (
            f"linear {proj} [{x.size // in_dims}, {in_dims}] x [{in_dims}, {out_dims}]"
        )
        if proj and take(name):
            rec[name] = ("linear", self, x)
        return stock[(nn.Linear, "__call__")](self, x)

    def rec_rel(self, rs):
        name = f"relative logits [{rs.size // 16}, 16] x [16, {self.proj.shape[1]}]"
        if take(name):
            rec[name] = ("rel", self, rs)
        return stock[(layers.RelativeLogits, "__call__")](self, rs)

    def rec_router(self, h):
        name = f"router logits [{h.size // 2304}, 2304] x [2304, 128]"
        if take(name):
            rec[name] = ("router", self, h)
        return stock[(layers.TopKRouter, "__call__")](self, h)

    def rec_experts(self, x, indices, sorted_indices=False):
        kind = "down" if self.input_dims == 1152 else "gate_up"
        name = f"experts {self.mode} {kind} ({indices.size} rows)"
        if take(name):
            rec[name] = ("experts", self, x, indices, sorted_indices)
        return stock[(QuantizedSwitchLinear, "__call__")](
            self, x, indices, sorted_indices
        )

    def rec_attention(self, q, k, v, pb, sinks, cache):
        L, S = q.shape[2], k.shape[2]
        end = cache.offset if cache is not None else L
        qpos = layers._positions(end - L, L)
        kpos = layers._key_positions(cache, S, L, end)
        out = stock_attention(self, q, k, v, pb, sinks, cache)
        local = "sliding" if self.sliding_window else "global"
        path = "fused" if L == 1 else "matmuls"
        name = f"attention {local}, {path} (L {L})"
        if take(name):
            mx.eval(out)
            rec[name] = ("attention", self, q, k, v, pb, sinks, qpos, kpos, out)
        return out

    def rec_rms(x, weight, eps):
        name = "rms_norm" + (" weighted" if weight is not None else "")
        if take(name):
            rec[name] = ("rms", x, weight, eps)
        return stock_fast_rms(x, weight, eps)

    def rec_conv(self, x):
        name = f"depthwise conv1d ({self.weight.shape[0]} channels)"
        if take(name):
            rec[name] = ("conv", self, x)
        return stock_conv(self, x)

    def rec_depth(blocks, partial, queries, *, scale):
        name = f"depth attention kernel ({blocks.shape[0] + 1} sources, {queries.shape[0]} queries)"
        if take(name):
            rec[name] = ("depth", blocks, partial, queries, scale)
        return stock_depth(blocks, partial, queries, scale=scale)

    recorders = [
        (nn.Linear, "__call__", rec_linear),
        (layers.RelativeLogits, "__call__", rec_rel),
        (layers.TopKRouter, "__call__", rec_router),
        (QuantizedSwitchLinear, "__call__", rec_experts),
        (layers.Attention, "_attention", rec_attention),
        (mx.fast, "rms_norm", rec_rms),
        (nn.Conv1d, "__call__", rec_conv),
        (layers, "depth_attention", rec_depth),
    ]
    saved = [(o, n, getattr(o, n)) for o, n, _ in recorders]
    for o, n, fn in recorders:
        setattr(o, n, fn)
    try:
        cache = model.make_cache()
        mx.eval(model(mx.array([tokens[: args.rows]]), cache=cache))
        for t in tokens[args.rows : args.rows + args.visit]:
            mx.eval(model(mx.array([[t]]), cache=cache))
    finally:
        for o, n, fn in saved:
            setattr(o, n, fn)
    print(f"recorded {len(rec)} operations", flush=True)

    rows = []
    for name in sorted(rec):
        kind, *a = rec[name]
        if kind == "linear":
            self, x = a
            ref = f64(x) @ f64(self["weight"]).T
            got = stock[(nn.Linear, "__call__")](self, x), linear(self, x)
        elif kind == "rel":
            self, rs = a
            ref = f64(rs) @ f64(self.proj)
            got = (
                stock[(layers.RelativeLogits, "__call__")](self, rs),
                relative_logits(self, rs),
            )
        elif kind == "router":
            self, h = a
            ref = f64(h) @ f64(self.weight).T
            got = h.astype(mx.float32) @ self.weight.T, router_logits(h, self.weight)
        elif kind == "experts":
            self, x, idx, srt = a
            got = (
                stock[(QuantizedSwitchLinear, "__call__")](self, x, idx, srt),
                experts(self, x, idx, srt),
            )
            ref = experts64(self, x, idx)
        elif kind == "attention":
            self, q, k, v, pb, sinks, qpos, kpos, out = a
            ref = attention64(
                q, k, v, pb, sinks, self.scaling, self.sliding_window, qpos, kpos
            )
            patched = None
            if q.shape[2] > 1:
                patched = attention_reference(
                    q,
                    k,
                    v,
                    pb,
                    sinks,
                    scale=self.scaling,
                    window=self.sliding_window,
                    query_positions=qpos,
                    key_positions=kpos,
                )
            got = out, patched
        elif kind == "rms":
            x, w, eps = a
            x64 = f64(x)
            ref = x64 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + eps)
            if w is not None:
                ref = ref * f64(w)
            got = stock_fast_rms(x, w, eps), None
        elif kind == "conv":
            self, x = a
            x64, w64 = f64(x), f64(self.weight)  # w: [C, K, 1]
            K = w64.shape[1]
            T = x64.shape[1] - K + 1
            ref = sum(x64[:, k : k + T] * w64[:, k, 0] for k in range(K))
            got = stock_conv(self, x), None
        elif kind == "depth":
            blocks, partial, queries, scale = a
            src = np.concatenate([f64(blocks), f64(partial)[None]])  # [N, ..., D]
            src = np.moveaxis(src, 0, -2)  # [..., N, D]
            w = softmax64(src @ f64(queries).T * scale, axis=-2)  # [..., N, Q]
            ref = np.swapaxes(w, -1, -2) @ src
            got = stock_depth(blocks, partial, queries, scale=scale), None
        before = rel_err(got[0], ref)
        after = rel_err(got[1], ref) if got[1] is not None else None
        rows.append((name, before, after, bf16_rounding(ref)))

    print("| Operation | Stock F32 | Patched | One BF16 rounding |")
    print("| --- | --- | --- | --- |")
    for name, before, after, bf in rows:
        a = f"{after:.1e}" if after is not None else "not patched"
        print(f"| {name} | {before:.1e} | {a} | {bf:.1e} |")
    worst = max(before if after is None else after for _, before, after, _ in rows)
    print(f"worst error of the patched model's operations: {worst:.1e}")
    if worst > TOLERANCE:
        raise SystemExit(f"above the tolerance {TOLERANCE:.0e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("text", type=Path)
    parser.add_argument("--rows", type=int, default=512, help="rows of the first chunk")
    parser.add_argument(
        "--visit", type=int, default=8, help="which call of each operation to record"
    )
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
