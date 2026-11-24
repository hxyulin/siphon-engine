"""How the L2 prefetch stream's busy time overlaps the main chain's kernel classes in a
verify trace.

usage: prefetch_overlap.py <report.sqlite> [--graph I] [--skip 3]

For each pass, intersects every prefetch kernel's interval with every chain kernel's
charged interval (max(start, previous end) .. end) and sums the overlap by class, per pass.
"""

import argparse
import collections
import sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("sqlite")
ap.add_argument("--graph", type=int)
ap.add_argument("--skip", type=int, default=3)
a = ap.parse_args()

CLASSES = [
    ("attention", ("attention_rows_", "attention_post_rows")),
    ("dense", ("rows_fp8_k2304", "rows_fp8_k3072")),
    ("experts", ("expert_mlp_rows",)),
    ("router", ("moe_router_step_rows",)),
    ("message gates", ("message_gate_", "rows_fp8_k4608")),
    ("depth attention", ("depth_attention_rows",)),
    ("lm head", ("logit_softcap", "argmax", "accept_greedy")),
]


def klass(name, grid):
    if name.startswith("rows_fp8_k2304") and grid > 4096:
        return "lm head"
    for c, prefixes in CLASSES:
        if name.startswith(prefixes):
            return c
    return "rest"


db = sqlite3.connect(a.sqlite)
rows = db.execute(
    """select k.start, k.end, s.value, k.streamId, k.gridX * k.gridY * k.gridZ, k.graphId
       from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName
       where k.graphId > 0 order by k.start"""
).fetchall()
if a.graph is not None:
    graphs = list(dict.fromkeys(r[5] for r in rows))
    rows = [r for r in rows if r[5] == graphs[a.graph]]
main_stream = next(r[3] for r in rows if r[2].startswith("embedding"))
chain = [r for r in rows if r[3] == main_stream]
side = [(r[0], r[1]) for r in rows if r[3] != main_stream]
starts = [i for i, r in enumerate(chain) if r[2].startswith("embedding")]
passes = [chain[i : (starts[j + 1] if j + 1 < len(starts) else len(chain))] for j, i in enumerate(starts)]
passes = passes[a.skip :]
over = collections.defaultdict(float)
total = collections.defaultdict(float)
for p in passes:
    prev = p[0][0]
    ivs = []
    for s, e, name, _, grid, _ in p:
        lo = max(s, prev)
        prev = max(prev, e)
        if e > lo:
            ivs.append((lo, e, klass(name, grid)))
            total[klass(name, grid)] += e - lo
    t0, t1 = p[0][0], p[-1][1]
    for ps, pe in side:
        if pe < t0 or ps > t1:
            continue
        for lo, hi, c in ivs:
            o = min(pe, hi) - max(ps, lo)
            if o > 0:
                over[c] += o
n = len(passes)
print(f"{n} passes; prefetch busy overlapping each class, us per pass (class chain time)")
for c in sorted(total, key=lambda c: -over[c]):
    print(f"  {c:16s} {over[c] / n / 1e3:8.1f}  ({total[c] / n / 1e3:7.1f}, {100 * over[c] / total[c]:3.0f}% covered)")
print(f"  total            {sum(over.values()) / n / 1e3:8.1f}")
