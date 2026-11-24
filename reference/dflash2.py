# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "torch==2.11.0",
#     "transformers==5.17.0",
#     "safetensors==0.8.0",
#     "numpy==2.3.5",
#     "dflash @ git+https://github.com/z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
#
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Inco AI's DFlash 2 drafter for Inkling 2 Mini, run in PyTorch at every anchor of a text.

usage: uv run --script reference/dflash2.py <taps.safetensors> <drafter-dir> <model-dir> <out>
       [--float32] [--raw-embed] [--first N]

Reads `tokens` (I32 [T]) and `taps` (BF16 [T, 4, 2304]: the partial block entering visits
5, 23 and 41 and the final normed hidden state, as `capture.py --taps` and Siphon's
`tests/dflash.rs` write them). At every anchor p from `--first` (16) to T - 8 it runs one
round: the context is the taps of positions max(0, p - 2047) to p - 1, the block is the
anchor token tokens[p] and 7 mask tokens at positions p to p + 7. Writes `ids` (I32
[A, 7]), `confidence` and `gap` (F32 [A, 7]) and `y` (BF16 [A, 7, 2304]) to <out>.

The backbone is z-lab's `DFlash2DraftModel` (github.com/z-lab/dflash). Around it the
script follows SGLang's DFlash worker with Inco AI's compatibility patch
(`sglang_inkling_dflash2.py`), which served the model card's numbers:

- The block's rows are the target's input embeddings, `embed_norm(embed(id))`; the mask
  rows embed token 200056. z-lab's generator reads the raw embedding table instead
  (`--raw-embed`).
- The candidates are the top 16 of `y @ lm_head.T` over the 200,058 real tokens.
- The selector scores candidate c of draft e after predecessor t as
  `unary[e, c] + bf16(<bf16(A[t] * h[e]), B[c]>)` with `h = hidden_projection(y)`, then
  applies the softcap 30 tanh(s / 30) after the correction. z-lab's selector caps the
  unary logits before adding. The greedy walk starts from the anchor.
- A draft's confidence is the softmax probability of its candidate over the 16 capped
  scores, the q a temperature-1 walk samples from. `gap` is the top-1 minus top-2 capped
  score of the step, `kgap` the 16th minus the 17th logit of the row and `rank` the
  logit rank of the picked candidate.

With `--float32` every weight and activation is F32, for the BF16 noise floor.
If the taps file holds `argmax` (the target's top-1 at each position, from
`capture.py`), the script prints the teacher-forced acceptance by draft position.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dflash import DFlash2DraftModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file

STEPS, BLOCK = 7, 8

p = argparse.ArgumentParser()
p.add_argument("taps")
p.add_argument("drafter")
p.add_argument("model")
p.add_argument("out")
p.add_argument("--float32", action="store_true")
p.add_argument("--raw-embed", action="store_true")
p.add_argument("--first", type=int, default=16)
args = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dtype = torch.float32 if args.float32 else torch.bfloat16
dev = torch.device("cuda")

t = load_file(args.taps, device="cuda")
tokens, taps = t["tokens"].long(), t["taps"]
argmax = t.get("argmax")
T = tokens.numel()
feats = taps.reshape(T, -1).to(dtype)

model_dir = Path(args.model)
text = json.loads((model_dir / "config.json").read_text())["text_config"]
vocab = text["unpadded_vocab_size"]
eps = text["rms_norm_eps"]
index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]


def get(name):
    with safe_open(model_dir / index[name], "pt", device="cuda") as h:
        return h.get_tensor(name)


embed = get("model.llm.embed.weight")
embed_norm = get("model.llm.embed_norm.weight")
lm_head = get("model.llm.unembed.weight")[:vocab].to(dtype).contiguous()

draft = DFlash2DraftModel.from_pretrained(args.drafter, dtype=dtype).to(dev).eval()
cfg = draft.config.dflash_config
assert cfg["selector_softcap_mode"] == "after_correction"
mask_id, softcap, window = cfg["mask_token_id"], cfg["final_logit_softcapping"], draft.config.sliding_window
sel = draft.candidate_selector
top_k = sel.top_k


def embed_rows(ids):
    x = embed[ids]
    if args.raw_embed:
        return x.to(dtype)
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * embed_norm.float()
    return x.to(torch.bfloat16).to(dtype)


ids_out, conf_out, gap_out, y_out, kgap_out, rank_out = [], [], [], [], [], []
with torch.inference_mode():
    for pos in range(args.first, T - BLOCK):
        lo = max(0, pos - (window - 1))
        block_ids = torch.full((BLOCK,), mask_id, device=dev, dtype=torch.long)
        block_ids[0] = tokens[pos]
        hidden = draft(
            target_hidden=feats[None, lo:pos],
            noise_embedding=embed_rows(block_ids)[None],
            position_ids=torch.arange(lo, pos + BLOCK, device=dev)[None],
        )[0, 1:]
        logits = hidden @ lm_head.T
        unary, cand = torch.topk(logits, top_k + 1, dim=-1)
        unary = unary.float()
        # How far the 16th candidate's logit lies above the 17th, which a draft computed
        # in other rounding can swap in.
        kgap = unary[:, top_k - 1] - unary[:, top_k]
        unary, cand = unary[:, :top_k], cand[:, :top_k]
        h = sel.hidden_projection(hidden)
        pred = tokens[pos]
        path, conf, gap, rank = [], [], [], []
        for e in range(STEPS):
            a = sel.predecessor_codebook.weight[pred] * h[e]
            corr = sel.successor_codebook.weight[cand[e]] @ a
            s = unary[e] + corr.float()
            s = softcap * torch.tanh(s / softcap)
            i = int(torch.argmax(s))
            pred = cand[e, i]
            top2 = torch.topk(s, 2).values
            path.append(int(pred))
            rank.append(i)
            conf.append(float(torch.softmax(s, -1)[i]))
            gap.append(float(top2[0] - top2[1]))
        ids_out.append(path)
        kgap_out.append(kgap.tolist())
        rank_out.append(rank)
        conf_out.append(conf)
        gap_out.append(gap)
        y_out.append(hidden.to(torch.bfloat16).cpu())
        if pos % 500 == 0:
            print(f"anchor {pos}/{T - BLOCK}", flush=True)

ids = torch.tensor(ids_out, dtype=torch.int32)
save_file(
    {
        "ids": ids,
        "confidence": torch.tensor(conf_out, dtype=torch.float32),
        "gap": torch.tensor(gap_out, dtype=torch.float32),
        "kgap": torch.tensor(kgap_out, dtype=torch.float32),
        "rank": torch.tensor(rank_out, dtype=torch.int32),
        "y": torch.stack(y_out).contiguous(),
    },
    args.out,
)
print(f"wrote {ids.shape[0]} anchors to {args.out}")

if argmax is not None:
    a = np.arange(args.first, T - BLOCK)
    target = argmax.cpu().numpy()
    # Draft i (1-based) at anchor p must equal the target's top-1 at p + i - 1.
    want = np.stack([target[a + i] for i in range(STEPS)], axis=1)
    hit = np.cumprod(ids.numpy() == want, axis=1)
    rates = hit.mean(axis=0)
    print(
        "teacher-forced acceptance by draft: "
        + ", ".join(f"{r:.1%}" for r in rates)
        + f"; tokens per round at k = 7: {1 + rates.sum():.3f}"
    )
