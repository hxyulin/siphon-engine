"""Cold-cache DRAM bytes per kernel class for one verify pass in an ncu CSV from
`ncu_bytes.sh`.

usage: ncu_classes.py <ncu.csv> [--kernels]

Takes the last pass that runs from an `embedding_bf16_2304` kernel to `accept_greedy`,
classes kernels as `verify_trace.py` does, and prints each class's DRAM bytes (read plus
write, each kernel from cold caches) and the time those take at 1.78 TB/s. `prefetch` is
the side stream's L2 prefetch.
"""

import collections
import csv
import re
import sys

BW = 1.78e12
UNITS = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "ns": 1, "us": 1e3, "ms": 1e6}
CLASSES = [
    ("attention", ("attention_rows_", "attention_post_rows")),
    ("dense", ("rows_fp8_k2304", "rows_fp8_k3072")),
    ("experts", ("expert_mlp_rows",)),
    ("router", ("moe_router_step_rows",)),
    ("message gates", ("message_gate_", "rows_fp8_k4608")),
    ("depth attention", ("depth_attention_rows",)),
    ("lm head", ("logit_softcap", "argmax", "accept_greedy")),
    ("prefetch", ("prefetch_l2",)),
]


def klass(name, grid):
    if name.startswith("rows_fp8_k2304") and grid > 4096:
        return "lm head"
    for c, prefixes in CLASSES:
        if name.startswith(prefixes):
            return c
    return "rest"


rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) > 10]
h = rows[0]
ii, ik, ig, im, iu, iv = (
    h.index(c) for c in ("ID", "Kernel Name", "Grid Size", "Metric Name", "Metric Unit", "Metric Value")
)
k = {}
for r in rows[1:]:
    d = k.setdefault(int(r[ii]), {"name": r[ik], "grid": r[ig]})
    d[r[im]] = float(r[iv].replace(",", "")) * UNITS.get(r[iu], 1)
ids = sorted(k)
names = [k[i]["name"] for i in ids]
start = max(j for j, n in enumerate(names) if n.startswith("embedding"))
end = next(j for j in range(start, len(names)) if names[j].startswith("accept_greedy"))
# The prefetch kernels of this pass may launch before its embedding kernel; take the
# side-stream kernels between the previous pass's end and this pass's end.
prev_end = max((j for j in range(start) if names[j].startswith("accept_greedy")), default=-1)
agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
per_kernel = collections.defaultdict(lambda: [0, 0.0])
for j in range(prev_end + 1, end + 1):
    d = k[ids[j]]
    grid = 1
    for x in re.findall(r"\d+", d["grid"]):
        grid *= int(x)
    c = klass(d["name"], grid)
    if j < start and c != "prefetch":
        continue
    b = d.get("dram__bytes_op_read.sum", 0) + d.get("dram__bytes_op_write.sum", 0)
    g = agg[c]
    g[0] += 1
    g[1] += b
    g[2] += d.get("gpu__time_duration.sum", 0)
    p = per_kernel[(c, d["name"][:40] + (" (lm head)" if c == "lm head" and grid > 4096 else ""))]
    p[0] += 1
    p[1] += b
total = sum(g[1] for c, g in agg.items() if c != "prefetch")
print(f"chain kernels' cold DRAM bytes {total / 1e9:.3f} GB -> {total / BW * 1e6:.0f} us at 1.78 TB/s")
print(f"{'class':16s} {'kernels':>7} {'MB':>9} {'floor us':>9} {'cold us':>9}")
for c, (n, b, t) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print(f"{c:16s} {n:7d} {b / 1e6:9.1f} {b / BW * 1e6:9.1f} {t / 1e3:9.1f}")
if "--kernels" in sys.argv:
    for (c, name), (n, b) in sorted(per_kernel.items(), key=lambda x: -x[1][1]):
        print(f"  {c:16s} {name:52s} {n:4d} {b / 1e6:9.2f} MB")
