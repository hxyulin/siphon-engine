"""MLX reference for the DSpark draft bundled with Inkling 2 Mini (`mtp.safetensors`).

`run` treats every position of a text as an anchor with the true context. It reads a
taps capture from `capture.py --text FILE --taps`, runs one draft round per anchor and
saves:

- `draft_ids`, I32 `[n, 7]`: the greedy draft ids of the round anchored at each
  position. Row 0 and rows whose block would run past the text are -1.
- `markov_margin`, F32 `[n, 7]`: the gap between the best and second-best Markov
  logit of each draft step, to explain disagreements as near-ties.
- At the sampled anchors `sampled_positions`, I32 `[s]`: `y`, F32 `[s, 8, 2304]`,
  the draft decoder's output for the whole block; `base_logits`, F32 `[s, 7, vocab]`,
  rows 1 to 7 through the LM head; and `markov_logits`, F32 `[s, 7, vocab]`, the
  softcapped logits each draft step took its argmax over.

By default the draft runs with BF16 weights and activations and rounds where vLLM's
draft rounds (section 1.9 of the design). `--float32` keeps every activation and
weight in F32, for the noise floor of the BF16 path. Both read the same BF16 taps.

`accept` reports the teacher-forced greedy acceptance of a `run` output: draft i of
the round anchored at P is accepted if it and every earlier draft equal MLX's top-1
token at the position before it. `compare` reports how far two `run` outputs are
apart. `routing` reads `capture.py --routing` captures and reports the number of
distinct experts that m consecutive positions route to at each visit, D_v(m), and
the expert bytes a verify of m rows would read.
"""

import argparse
import json
import platform
import struct
from importlib import metadata
from pathlib import Path

import mlx.core as mx
import numpy as np

BLOCK = 8
STEPS = BLOCK - 1
WINDOW = 1024
SOFTCAP = 30.0
EPS = 1e-6
HEADS, KV_HEADS, HEAD_DIM = 24, 8, 128
RANK = 256
THETA = 1e6
LOCAL_LAYERS = (0, 2)
SAMPLE_EVERY = 128
# Around the 1024-position wrap of the local layers' window.
SAMPLE_RANGES = [(WINDOW - 4, WINDOW + 12)]


def sampled_positions(n: int) -> list[int]:
    positions = set(range(SAMPLE_EVERY, n, SAMPLE_EVERY))
    for start, stop in SAMPLE_RANGES:
        positions.update(range(start, min(stop, n)))
    return sorted(p for p in positions if 1 <= p <= n - BLOCK)


def rope_inv_freq() -> np.ndarray:
    """F32 `1 / theta^(2i/128)` as vLLM's table builds it in F32: a correctly rounded
    power, then a correctly rounded reciprocal."""
    exponent = np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM
    power = (np.float64(THETA) ** exponent).astype(np.float32)
    return np.float32(1.0) / power


def split_bf16(x: mx.array) -> list[mx.array]:
    """Three BF16-exact F32 parts that sum to `x` within about 2^-24 of it."""
    parts = []
    for _ in range(3):
        part = x.astype(mx.bfloat16).astype(mx.float32)
        parts.append(part)
        x = x - part
    return parts


def matmul(a: mx.array, b: mx.array, a_exact: bool = False, b_exact: bool = False):
    """`a @ b` in F32 with full precision.

    MLX's F32 matmul on Apple GPUs (measured on an M5 Max) keeps only about 11 bits of
    inputs that are not BF16 values: its relative error is 1e-3 on random F32 inputs but
    6e-7 on BF16-valued ones. So each operand that may not be BF16-exact is split into
    three BF16-exact parts, and the products of the parts sum in F32.
    """
    a_parts = [a] if a_exact else split_bf16(a)
    b_parts = [b] if b_exact else split_bf16(b)
    out = None
    for i, x in enumerate(a_parts):
        for j, y in enumerate(b_parts):
            if i + j > 2:
                continue
            term = x @ y
            out = term if out is None else out + term
    return out


class Draft:
    """The draft's weights and its forward pass, in BF16 or F32."""

    def __init__(self, weights: dict[str, mx.array], vocab: int, float32: bool):
        """`weights` holds the draft's tensors under their `mtp.safetensors` names and
        the target's as `embed`, `embed_norm` and `lm_head`. Tensors a test does not
        need may be missing."""
        self.dtype = mx.float32 if float32 else mx.bfloat16
        self.vocab = vocab
        if "fuse_weight" in weights:
            self.alpha = mx.softmax(weights["fuse_weight"].astype(mx.float32), axis=-1)
        if "lm_head" in weights and "markov_head.hidden_out.weight" in weights:
            self.fold_weight, self.fold_bias = fold(
                weights["lm_head"],
                weights["markov_head.hidden_out.weight"],
                weights["markov_head.hidden_out.bias"],
                self.dtype,
            )
        self.w = {k: v.astype(self.dtype) for k, v in weights.items()}
        if "markov_head.rnn_proj.weight" in weights:
            self.rnn_bias = weights["markov_head.rnn_proj.bias"].astype(mx.float32)
            # The rnn_proj products take BF16 inputs and write F32: run them in F32
            # over the (exactly widened) weights.
            self.rnn_weight = weights["markov_head.rnn_proj.weight"].astype(mx.float32)
            self.markov_w1 = weights["markov_head.markov_w1.weight"].astype(mx.float32)
        self.inv_freq = rope_inv_freq()
        mx.eval(self.__dict__)

    @classmethod
    def load(cls, model_dir: Path, float32: bool) -> "Draft":
        config = json.loads((model_dir / "config.json").read_text())
        weights = mx.load(str(model_dir / "mtp.safetensors"))
        target = model_dir / "model-00001-of-00001.safetensors"
        weights["embed"] = read_bf16(target, "model.llm.embed.weight")
        weights["embed_norm"] = read_bf16(target, "model.llm.embed_norm.weight")
        weights["lm_head"] = read_bf16(target, "model.llm.unembed.weight")
        return cls(weights, config["text_config"]["unpadded_vocab_size"], float32)

    def round(self, x: mx.array) -> mx.array:
        """Rounds F32 values to the activation type and back."""
        return x.astype(self.dtype).astype(mx.float32)

    @property
    def exact(self) -> bool:
        """Whether activations hold BF16 values, which MLX's F32 matmul keeps exactly."""
        return self.dtype == mx.bfloat16

    def linear(self, x: mx.array, name: str) -> mx.array:
        """`x · Wᵀ` with F32 accumulation, rounded to the activation type. The weights
        are BF16 values in both modes."""
        if self.exact:
            return (x.astype(self.dtype) @ self.w[name].T).astype(mx.float32)
        return matmul(x, self.w[name].T, b_exact=True)

    def rms_norm(self, x: mx.array, name: str) -> mx.array:
        """F32 RMSNorm, rounded once."""
        weight = self.w[name].astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + EPS)
        return self.round(x * weight)

    def mix(self, taps: mx.array, row: int, norm: str) -> mx.array:
        """RMSNorm of the softmax-weighted sum of the 4 taps `[..., 4, 2304]`."""
        mixed = (taps.astype(mx.float32) * self.alpha[row][:, None]).sum(axis=-2)
        return self.rms_norm(mixed, norm)

    def rope(self, x: mx.array, positions: mx.array) -> mx.array:
        """NeoX RoPE in F32 of `x` `[..., heads, 128]` at `positions` (shape `x[..., 0, 0]`).

        The angle is the F32 product of the position and the frequency, as in vLLM's
        F32 table; its cosine and sine are correctly rounded (through F64), since
        Metal's are off by several F32 ulps at angles of tens of thousands."""
        angle = np.array(positions, np.float32)[..., None, None] * self.inv_freq
        cos = mx.array(np.cos(angle.astype(np.float64)).astype(np.float32))
        sin = mx.array(np.sin(angle.astype(np.float64)).astype(np.float32))
        x1, x2 = x[..., : HEAD_DIM // 2], x[..., HEAD_DIM // 2 :]
        return self.round(mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1))

    def context_kv(self, layer: int, taps: mx.array, positions: mx.array):
        """Context keys and values `[rows, 8, 128]` from target taps `[rows, 4, 2304]`."""
        p = f"layers.{layer}."
        c = self.mix(taps, layer, p + "dflare_norm.weight")
        k = self.linear(c, p + "self_attn.k_proj_ctx.weight")
        v = self.linear(c, p + "self_attn.v_proj_ctx.weight")
        k = k.reshape(-1, KV_HEADS, HEAD_DIM)
        k = self.rope(self.rms_norm(k, p + "self_attn.k_norm_ctx.weight"), positions)
        return k, v.reshape(-1, KV_HEADS, HEAD_DIM)

    def block(self, anchors: mx.array, starts: list[int], ctx: list) -> mx.array:
        """The draft decoder over the blocks of `anchors` (token ids `[b]`) at the
        positions `starts`, with context K/V `ctx[layer] = (k, v)` for positions
        0..: returns Y `[b, 8, 2304]`."""
        b = len(starts)
        start = mx.array(starts, dtype=mx.int32)
        positions = start[:, None] + mx.arange(BLOCK, dtype=mx.int32)
        anchor = self.rms_norm(
            self.w["embed"][anchors].astype(mx.float32), "embed_norm"
        )
        mask = mx.broadcast_to(
            self.w["mask_embed"].astype(mx.float32), (b, STEPS, anchor.shape[-1])
        )
        residual = mx.concatenate([anchor[:, None], mask], axis=1)
        h = self.rms_norm(residual, "layers.0.input_layernorm.weight")
        update = None
        for layer in range(3):
            p = f"layers.{layer}."
            if update is not None:
                total = residual + update
                residual = self.round(total)
                h = self.rms_norm(total, p + "input_layernorm.weight")
            q = self.linear(h, p + "self_attn.q_proj.weight")
            k = self.linear(h, p + "self_attn.k_proj.weight")
            v = self.linear(h, p + "self_attn.v_proj.weight")
            q = q.reshape(b, BLOCK, HEADS, HEAD_DIM)
            k = k.reshape(b, BLOCK, KV_HEADS, HEAD_DIM)
            v = v.reshape(b, BLOCK, KV_HEADS, HEAD_DIM)
            q = self.rope(self.rms_norm(q, p + "self_attn.q_norm.weight"), positions)
            k = self.rope(self.rms_norm(k, p + "self_attn.k_norm.weight"), positions)
            a = self.attention(q, k, v, ctx[layer], starts, layer in LOCAL_LAYERS)
            total = residual + self.linear(a, p + "self_attn.o_proj.weight")
            residual = self.round(total)
            h = self.rms_norm(total, p + "post_attention_layernorm.weight")
            gate = self.linear(h, p + "mlp.gate_proj.weight")
            up = self.linear(h, p + "mlp.up_proj.weight")
            act = self.silu_mul(gate, up)
            update = self.linear(act, p + "mlp.down_proj.weight")
        return self.rms_norm(residual + update, "norm.weight")

    def silu_mul(self, gate: mx.array, up: mx.array) -> mx.array:
        return self.round(gate / (1.0 + mx.exp(-gate)) * up)

    def attention(self, q, k, v, ctx, starts: list[int], local: bool) -> mx.array:
        """Bidirectional block attention: each block sees its own 8 keys and the
        context positions before its start, the last 1024 of them if `local`."""
        ctx_k, ctx_v = ctx
        lo = max(0, starts[0] - WINDOW) if local else 0
        hi = starts[-1]
        group = HEADS // KV_HEADS
        b = q.shape[0]
        # [b, kv, group, 8, 128]
        qh = q.reshape(b, BLOCK, KV_HEADS, group, HEAD_DIM).transpose(0, 2, 3, 1, 4)
        scale = HEAD_DIM**-0.5
        kb = k.transpose(0, 2, 1, 3)  # [b, kv, 8, 128]
        vb = v.transpose(0, 2, 1, 3)
        ex = self.exact
        own = matmul(qh, kb[:, :, None].swapaxes(-1, -2), ex, ex) * scale  # [b, kv, g, 8, 8]
        if hi > lo:
            kc = ctx_k[lo:hi].transpose(1, 0, 2)  # [kv, n, 128]
            vc = ctx_v[lo:hi].transpose(1, 0, 2)
            past = matmul(qh, kc[None, :, None].swapaxes(-1, -2), ex, ex) * scale
            key_pos = mx.arange(lo, hi)[None, :]
            start = mx.array(starts)[:, None]
            visible = key_pos < start
            if local:
                visible = visible & (key_pos >= start - WINDOW)
            past = mx.where(visible[:, None, None, None, :], past, -mx.inf)
            scores = mx.concatenate([past, own], axis=-1)
        else:
            scores = own
        weights = mx.softmax(scores, axis=-1, precise=True)
        out = matmul(weights[..., -BLOCK:], vb[:, :, None], b_exact=ex)
        if hi > lo:
            out = out + matmul(weights[..., :-BLOCK], vc[None, :, None], b_exact=ex)
        out = out.transpose(0, 3, 1, 2, 4).reshape(b, BLOCK, HEADS * HEAD_DIM)
        return self.round(out)

    def markov(self, y: mx.array, f: mx.array, anchors: mx.array):
        """The Markov head over rows 1..7 of Y `[b, 8, 2304]` with head context f
        `[b, 2304]`: returns base logits, Markov logits `[b, 7, vocab]` and ids `[b, 7]`."""
        y = y[:, 1:]
        base = self.linear(y, "lm_head")[..., : self.vocab]
        inputs = mx.concatenate([y, mx.broadcast_to(f[:, None], y.shape)], axis=-1)
        ex = self.exact
        context = self.rnn_bias + matmul(inputs, self.rnn_weight[:, 2 * RANK :].T, ex, True)
        state = mx.zeros((y.shape[0], RANK), mx.float32)
        prev = anchors
        logits, ids = [], []
        for i in range(STEPS):
            step_in = mx.concatenate([state, self.markov_w1[prev]], axis=-1)
            raw = context[:, i] + matmul(step_in, self.rnn_weight[:, : 2 * RANK].T, ex, True)
            state, latent = self.gate(raw, state)
            step_logits = self.step_logits(base[:, i], latent)
            prev = mx.argmax(step_logits, axis=-1).astype(mx.int32)
            logits.append(step_logits)
            ids.append(prev)
        return base, mx.stack(logits, axis=1), mx.stack(ids, axis=1)


    def gate(self, raw: mx.array, state: mx.array) -> tuple[mx.array, mx.array]:
        """The Markov gate of the F32 `rnn_proj` output `raw` `[..., 768]`: returns
        the new state and the latent, every op rounded to the activation type."""
        gate = self.round(raw[..., :RANK])
        cand = self.round(raw[..., RANK : 2 * RANK])
        out = self.round(raw[..., 2 * RANK :])
        sigma = self.round(1.0 / (1.0 + mx.exp(-gate)))
        keep = self.round(sigma * state)
        new = self.round(self.round(1.0 - sigma) * self.round(mx.tanh(cand)))
        return self.round(keep + new), self.round(mx.tanh(out))

    def step_logits(self, base: mx.array, latent: mx.array) -> mx.array:
        """Softcapped Markov logits from base logits `[..., vocab]` and a latent."""
        product = matmul(latent, self.fold_weight.T, self.exact, self.exact)
        bias = self.round(self.fold_bias + product)
        total = self.round(base[..., : self.vocab] + bias[..., : self.vocab])
        return self.round(SOFTCAP * mx.tanh(total / SOFTCAP))


def read_bf16(path: Path, name: str) -> mx.array:
    """One BF16 tensor of a large safetensors file, without loading the rest."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        info = json.loads(f.read(size))[name]
        assert info["dtype"] == "BF16"
        start, stop = info["data_offsets"]
        f.seek(8 + size + start)
        raw = np.frombuffer(f.read(stop - start), np.uint16).reshape(info["shape"])
    return mx.array(raw).view(mx.bfloat16)


def fold(lm_head, hidden_out_weight, hidden_out_bias, dtype):
    """`lm_head · hidden_out` in F64, rounded once to `dtype`, as vLLM folds."""
    head = np.array(lm_head.astype(mx.float32), dtype=np.float64)
    weight = np.array(hidden_out_weight.astype(mx.float32), dtype=np.float64)
    bias = np.array(hidden_out_bias.astype(mx.float32), dtype=np.float64)
    fold_weight = (head @ weight).astype(np.float32)
    fold_bias = (head @ bias).astype(np.float32)
    # Rounded to the weight type, then widened: products sum in F32.
    return (
        mx.array(fold_weight).astype(dtype).astype(mx.float32),
        mx.array(fold_bias).astype(dtype).astype(mx.float32),
    )


def run(args: argparse.Namespace) -> None:
    capture = mx.load(str(args.taps))
    tokens = np.array(capture["tokens"])
    taps = capture["taps"]
    n = tokens.size
    if args.limit:
        n = min(n, args.limit)
    draft = Draft.load(args.model, args.float32)
    positions = mx.arange(n, dtype=mx.int32)
    ctx = [draft.context_kv(layer, taps[:n], positions) for layer in range(3)]
    f_all = draft.mix(taps[:n], 3, "markov_head.dflare_norm.weight")
    mx.eval(ctx, f_all)

    sampled = sampled_positions(n)
    sampled_set = set(sampled)
    draft_ids = np.full((n, STEPS), -1, np.int32)
    margins = np.zeros((n, STEPS), np.float32)
    kept: dict[str, dict[int, np.ndarray]] = {"y": {}, "base": {}, "markov": {}}
    anchors_all = list(range(1, n - BLOCK + 1))
    for i in range(0, len(anchors_all), args.batch):
        starts = anchors_all[i : i + args.batch]
        anchor_ids = mx.array(tokens[starts], dtype=mx.int32)
        y = draft.block(anchor_ids, starts, ctx)
        f = f_all[mx.array(starts) - 1]
        base, logits, ids = draft.markov(y, f, anchor_ids)
        top2 = mx.topk(logits, 2, axis=-1)
        margin = mx.max(top2, axis=-1) - mx.min(top2, axis=-1)
        mx.eval(ids, margin)
        draft_ids[starts] = np.array(ids)
        margins[starts] = np.array(margin)
        rows = [j for j, p in enumerate(starts) if p in sampled_set]
        if rows:
            sel = mx.array(rows)
            parts = (y[sel], base[sel], logits[sel])
            mx.eval(parts)
            for j, row in enumerate(rows):
                for key, part in zip(("y", "base", "markov"), parts):
                    kept[key][starts[row]] = np.array(part[j])
        if (i // args.batch) % 16 == 0:
            print(f"anchor {starts[-1]}/{n - BLOCK}", flush=True)

    tensors = {
        "tokens": tokens[:n].astype(np.int32),
        "draft_ids": draft_ids,
        "markov_margin": margins,
        "sampled_positions": np.array(sampled, np.int32),
        "y": np.stack([kept["y"][p] for p in sampled]),
        "base_logits": np.stack([kept["base"][p] for p in sampled]),
        "markov_logits": np.stack([kept["markov"][p] for p in sampled]),
    }
    if "argmax" in capture:
        tensors["argmax"] = np.array(capture["argmax"])[:n].astype(np.int32)
    info = {
        "model": str(args.model),
        "taps": str(args.taps),
        "dtype": "float32" if args.float32 else "bfloat16",
        "anchors": len(anchors_all),
        "mlx": metadata.version("mlx"),
        "machine": platform.platform(),
    }
    saved = {k: mx.array(v) for k, v in tensors.items()}
    if not args.float32:
        # These hold BF16 values: store them as such.
        for key in ("y", "base_logits", "markov_logits"):
            saved[key] = saved[key].astype(mx.bfloat16)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.out),
        saved,
        metadata={k: json.dumps(v) for k, v in info.items()},
    )
    print(json.dumps(info, indent=2))
    report_acceptance(tensors)


def report_acceptance(t: dict) -> None:
    if "argmax" not in t:
        return
    ids, argmax, tokens = t["draft_ids"], t["argmax"], t["tokens"]
    n = tokens.size
    # Anchors whose 7 targets all lie inside the text.
    anchors = np.arange(1, n - STEPS)
    anchors = anchors[ids[anchors, 0] >= 0]
    for label, target in (("MLX top-1", argmax), ("text", tokens)):
        if label == "MLX top-1":
            # Draft i (1-based) of anchor P must equal the target's top-1 at P+i-1.
            want = np.stack([target[anchors + i] for i in range(STEPS)], axis=1)
        else:
            want = np.stack([target[anchors + i + 1] for i in range(STEPS)], axis=1)
        hit = ids[anchors] == want
        prefix = np.cumprod(hit, axis=1)
        rates = prefix.mean(axis=0)
        print(
            f"acceptance against {label} over {anchors.size} anchors: "
            + ", ".join(f"{r:.1%}" for r in rates)
            + f"; tokens per step at k=3,5,7: "
            + ", ".join(f"{1 + rates[:k].sum():.2f}" for k in (3, 5, 7))
        )


def load(path: Path) -> dict[str, np.ndarray]:
    """A safetensors file as NumPy arrays, BF16 widened to F32."""
    out = {}
    for k, v in mx.load(str(path)).items():
        out[k] = np.array(v.astype(mx.float32) if v.dtype == mx.bfloat16 else v)
    return out


def accept(args: argparse.Namespace) -> None:
    for path in args.runs:
        print(path)
        report_acceptance(load(path))


def compare(args: argparse.Namespace) -> None:
    a, b = load(args.reference), load(args.other)
    ids_a, ids_b = a["draft_ids"], b["draft_ids"]
    valid = ids_a[:, 0] >= 0
    same = ids_a[valid] == ids_b[valid]
    # The first differing step of each anchor; later steps follow other tokens.
    first = np.where(same.all(axis=1), STEPS, np.argmin(same, axis=1))
    print(
        f"draft ids equal at {same.mean():.4%} of anchor-step pairs; "
        f"whole rounds equal {np.mean(first == STEPS):.2%} of {valid.sum()} anchors"
    )
    # Steps whose earlier draft ids agree, so both sides fed the head the same tokens.
    same_prefix = np.arange(STEPS)[None, :] <= first[:, None]
    print(
        f"first step equal at {same[:, 0].mean():.4%}; steps with an equal prefix "
        f"equal at {same[same_prefix].mean():.4%} of {same_prefix.sum()}"
    )
    split = np.nonzero(first < STEPS)[0]
    anchors = np.nonzero(valid)[0][split]
    margins = a["markov_margin"][anchors, first[split]]
    if margins.size:
        print(
            f"Markov top-2 margin of the first file at the first differing step: "
            f"median {np.median(margins):.3g}, max {margins.max():.3g}; "
            f"over all steps: median {np.median(a['markov_margin'][valid]):.3g}"
        )
    sampled = a["sampled_positions"]
    sampled_prefix = same_prefix[np.searchsorted(np.nonzero(valid)[0], sampled)]
    for key in ("y", "base_logits", "markov_logits"):
        x, y = a[key].astype(np.float64), b[key].astype(np.float64)
        if key == "markov_logits":
            x, y = x[sampled_prefix], y[sampled_prefix]
        d = np.abs(x - y)
        rel = np.linalg.norm(x - y) / np.linalg.norm(x)
        print(
            f"{key}: max |diff| {d.max():.4g}, mean |diff| {d.mean():.4g}, "
            f"relative L2 {rel:.3e}"
        )


def checkpoint_expert_bytes(model_dir: Path, layers: int) -> list[float]:
    """Bytes of one routed expert per layer: its weights and scales."""
    path = model_dir / "model-00001-of-00001.safetensors"
    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    sizes = [0.0] * layers
    for name, info in header.items():
        if ".mlp.experts." not in name or name.startswith("model.visual"):
            continue
        layer = int(name.split(".layers.")[1].split(".")[0])
        start, stop = info["data_offsets"]
        if info["shape"] and info["shape"][0] == 128:
            sizes[layer] += (stop - start) / 128
    return sizes


def routing(args: argparse.Namespace) -> None:
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    loop = config["loop_config"]
    layers = config["num_hidden_layers"]
    visits = (
        list(range(loop["core_start"]))
        + list(range(loop["core_start"], loop["core_end"])) * loop["num_passes"]
        + list(range(loop["core_end"], layers))
    )
    expert_bytes = checkpoint_expert_bytes(args.model, layers)
    cost = np.array([expert_bytes[layer] for layer in visits])
    print(f"expert bytes per visit (MB): {sorted(set(np.round(cost / 1e6, 3)))}")
    widths = range(1, 9)
    per_text = []
    for path in args.captures:
        r = np.array(mx.load(str(path))["routing"])  # [48, n, 8]
        n = r.shape[1]
        onehot = np.zeros((r.shape[0], n, 128), bool)
        np.put_along_axis(onehot, r.astype(np.int64), True, axis=2)
        d = np.zeros((r.shape[0], len(widths)))
        for j, m in enumerate(widths):
            # Distinct experts over every window of m consecutive positions.
            c = np.cumsum(np.concatenate([np.zeros((r.shape[0], 1, 128)), onehot], 1), 1)
            window = c[:, m:] - c[:, :-m]
            d[:, j] = (window > 0).sum(axis=2).mean(axis=1)
        per_text.append((path.name, n, d))
    print("| m | " + " | ".join(name for name, _, _ in per_text) + " | mean | uniform |")
    print("| ---: |" + " ---: |" * (len(per_text) + 2))
    mean = np.mean([d for _, _, d in per_text], axis=0)  # [48, widths]
    for j, m in enumerate(widths):
        cells = [f"{d[:, j].mean():.2f}" for _, _, d in per_text]
        uniform = 128 * (1 - (120 / 128) ** m)
        print(f"| {m} | " + " | ".join(cells) + f" | {mean[:, j].mean():.2f} | {uniform:.1f} |")
    print()
    # T_verify as in section 3.3 of the design: dense weights plus experts, at
    # decode's effective bandwidth.
    rate = args.bandwidth * args.efficiency
    print(
        f"Modeled verify time: ({args.dense_gb} GB dense + expert GB) / "
        f"({args.bandwidth} TB/s x {args.efficiency})"
    )
    print(
        "| m | Expert GB | T_verify ms | Expert GB, uniform | T_verify, uniform "
        "| min D_v | max D_v | visits 0-5 | 6-41 | 42-47 |"
    )
    print("| ---: |" + " ---: |" * 9)
    for j, m in enumerate(widths):
        dv = mean[:, j]
        uniform = 128 * (1 - (120 / 128) ** m)
        gb = (dv * cost).sum() / 1e9
        gb_uniform = uniform * cost.sum() / 1e9
        print(
            f"| {m} | {gb:.3f} | {(args.dense_gb + gb) / rate:.2f} | {gb_uniform:.3f} "
            f"| {(args.dense_gb + gb_uniform) / rate:.2f} "
            f"| {dv.min():.2f} (visit {dv.argmin()}) | {dv.max():.2f} (visit {dv.argmax()}) "
            f"| {dv[:6].mean():.2f} | {dv[6:42].mean():.2f} | {dv[42:].mean():.2f} |"
        )
    if args.per_visit:
        print()
        print("visit layer " + " ".join(f"m={m:<5}" for m in widths))
        for v in range(mean.shape[0]):
            print(f"{v:5} {visits[v]:5} " + " ".join(f"{x:7.2f}" for x in mean[v]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the draft at every anchor of a taps capture")
    r.add_argument("model", type=Path)
    r.add_argument("taps", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--float32", action="store_true")
    r.add_argument("--batch", type=int, default=32, help="anchors per draft pass")
    r.add_argument("--limit", type=int, help="use only the first LIMIT positions")
    r.set_defaults(run=run)
    a = sub.add_parser("accept", help="teacher-forced acceptance of run outputs")
    a.add_argument("runs", type=Path, nargs="+")
    a.set_defaults(run=accept)
    c = sub.add_parser("compare", help="compare two run outputs of the same taps")
    c.add_argument("reference", type=Path)
    c.add_argument("other", type=Path)
    c.set_defaults(run=compare)
    g = sub.add_parser("routing", help="distinct experts per verify width")
    g.add_argument("model", type=Path)
    g.add_argument("captures", type=Path, nargs="+")
    g.add_argument("--per-visit", action="store_true")
    g.add_argument("--dense-gb", type=float, default=3.62, help="BF16 dense weights")
    g.add_argument("--bandwidth", type=float, default=1.70, help="TB/s")
    g.add_argument("--efficiency", type=float, default=0.84)
    g.set_defaults(run=routing)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
