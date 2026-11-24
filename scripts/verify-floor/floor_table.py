"""DRAM floor of a verify pass per width and context, next to measured times.

usage: floor_table.py <union.json>:<context>:<bench.log>[,...] [--bw 1.78e12]

Each argument names an `expert_union.py --bench --json` file, the context the pass ran at
and the `bench --verify-widths` log with the measured times. The floor is the bytes a pass
must read from DRAM over `--bw`:

- the dense FP8 copies decode reads (every visit's attention projections, the message
  gates of 24 looped visits and the LM head), from the checkpoint's shapes;
- the routers, stored as F32;
- the union of experts the rows route to at each visit (from the union file);
- the KV read once per pass: FP8 global caches (8 visits, 2 KiB per token) and BF16
  sliding rings (40 visits, 4 KiB per token, at most 1024 tokens).
"""

import argparse
import json
import re

HIDDEN, VOCAB, Q, KV, REL, HEADS = 2304, 201024, 3072, 1024, 384, 24
IN_ROWS = Q + REL + Q + HEADS + KV + KV  # q, r, gate, sink, k, v
GLOBAL_EXTRA = 2 * 32  # dynamic k and v conv projections
SLIDING, GLOBAL, GATED = 40, 8, 24


def fp8(rows, k):
    return rows * k + rows * 4  # E4M3 values and one F32 scale per row


ATTN_PROJ = SLIDING * (fp8(IN_ROWS, HIDDEN) + fp8(HIDDEN, Q)) + GLOBAL * (
    fp8(IN_ROWS + GLOBAL_EXTRA, HIDDEN) + fp8(HIDDEN, Q)
)
GATES = GATED * 4 * (fp8(64, 2 * HIDDEN) + 2 * fp8(HIDDEN, 64))
LM_HEAD = fp8(VOCAB, HIDDEN)
ROUTER = 48 * 128 * HIDDEN * 4


def kv_bytes(context, rows):
    return GLOBAL * (context + rows) * 2048 + SLIDING * min(context + rows, 1024) * 4096


def floor_bytes(context, rows, expert_gb):
    return ATTN_PROJ + GATES + LM_HEAD + ROUTER + expert_gb * 1e9 + kv_bytes(context, rows)


def measured(log):
    out = {}
    for line in open(log):
        m = re.match(r"\| (\d+) \| (\d) \| ([\d.]+) \| ([\d.]+) \|", line)
        if m:
            out[int(m.group(2))] = (int(m.group(1)), float(m.group(3)), float(m.group(4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--bw", type=float, default=1.78e12)
    a = ap.parse_args()
    print(
        f"dense GB: attention projections {ATTN_PROJ / 1e9:.3f}, gates {GATES / 1e9:.3f}, "
        f"LM head {LM_HEAD / 1e9:.3f}, routers {ROUTER / 1e9:.3f}"
    )
    print("| context | w | distinct | expert GB | KV GB | floor GB | floor ms | min ms | median ms | % of floor (median) |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run in a.runs:
        union_path, context, log = run.split(":")
        union = json.load(open(union_path))
        times = measured(log)
        for w in range(1, 9):
            u = union[str(w)]
            ctx = times[w][0] if int(context) == 0 else int(context)
            b = floor_bytes(ctx, w, u["expert_gb"])
            fl = b / a.bw * 1e3
            _, tmin, tmed = times[w]
            print(
                f"| {ctx} | {w} | {u['distinct']:.1f} | {u['expert_gb']:.2f} | "
                f"{kv_bytes(ctx, w) / 1e9:.3f} | {b / 1e9:.2f} | {fl:.2f} | {tmin:.2f} | "
                f"{tmed:.2f} | {100 * fl / tmed:.0f}% |"
            )


main()
