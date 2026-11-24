"""Split the verify passes of an nsys trace (`smallm verify` under `--cuda-graph-trace=node`)
into kernel classes, with each kernel's chain time.

usage: verify_trace.py <report.sqlite> [--skip 3] [--kernels] [--graph I] [--json out.json]

Graph kernels on the stream that runs `embedding_bf16_2304` form the main chain; the other
graph stream is the L2 prefetch. A pass starts at each `embedding_bf16_2304`. A kernel's
chain time is end - max(start, previous chain kernel's end), so a PDL kernel that starts
early is charged only from the previous kernel's end; time between kernels is idle.
`--skip N` drops the first N passes. `--graph I` keeps only the I-th CUDA graph to run (from
0), for a `bench --verify-widths` trace where each width has its own graph.
"""

import argparse
import collections
import json
import sqlite3

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--skip", type=int, default=3)
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--graph", type=int)
    ap.add_argument("--json")
    a = ap.parse_args()
    db = sqlite3.connect(a.sqlite)
    rows = db.execute(
        """select k.start, k.end, s.value, k.streamId, k.gridX * k.gridY * k.gridZ, k.graphId
           from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName
           where k.graphId > 0 order by k.start"""
    ).fetchall()
    if a.graph is not None:
        graphs = list(dict.fromkeys(r[5] for r in rows))
        rows = [r for r in rows if r[5] == graphs[a.graph]]
    rows = [r[:5] for r in rows]
    main_stream = next(r[3] for r in rows if r[2].startswith("embedding"))
    chain = [r for r in rows if r[3] == main_stream]
    side = [r for r in rows if r[3] != main_stream]
    starts = [i for i, r in enumerate(chain) if r[2].startswith("embedding")]
    passes = []
    for j, i in enumerate(starts):
        end = starts[j + 1] if j + 1 < len(starts) else len(chain)
        passes.append(chain[i:end])
    passes = passes[a.skip :]
    n = len(passes)
    by_class = collections.defaultdict(float)
    by_kernel = collections.defaultdict(lambda: [0, 0.0, 0.0])
    span = busy = prefetch = 0.0
    for p in passes:
        t0, t1 = p[0][0], p[-1][1]
        span += t1 - t0
        prev = t0
        for s, e, name, _, grid in p:
            c = max(0, e - max(s, prev))
            prev = max(prev, e)
            busy += c
            by_class[klass(name, grid)] += c
            k = by_kernel[(klass(name, grid), name if grid <= 4096 else name + " (lm head)")]
            k[0] += 1
            k[1] += c
            k[2] += e - s
        prefetch += sum(e - s for s, e, *_ in side if t0 <= s < t1)
    us = lambda x: x / n / 1e3
    out = {
        "passes": n,
        "span_us": us(span),
        "busy_us": us(busy),
        "idle_us": us(span - busy),
        "prefetch_us": us(prefetch),
        "kernels_per_pass": sum(len(p) for p in passes) / n,
        "classes": {c: us(t) for c, t in by_class.items()},
        "kernels": [
            {"class": c, "kernel": k, "calls": v[0] / n, "chain_us": us(v[1]), "raw_us": us(v[2])}
            for (c, k), v in sorted(by_kernel.items(), key=lambda x: -x[1][1])
        ],
    }
    print(
        f"{n} passes: span {out['span_us']:.0f} us, chain busy {out['busy_us']:.0f}, "
        f"idle {out['idle_us']:.0f}, prefetch busy {out['prefetch_us']:.0f}, "
        f"{out['kernels_per_pass']:.0f} chain kernels per pass"
    )
    for c, t in sorted(out["classes"].items(), key=lambda x: -x[1]):
        print(f"  {c:16s} {t:8.1f} us")
    if a.kernels:
        print(f"{'class':16s} {'kernel':44s} {'calls':>6} {'chain us':>9} {'us/call':>8} {'raw us':>8}")
        for k in out["kernels"]:
            print(
                f"{k['class']:16s} {k['kernel'][:44]:44s} {k['calls']:6.1f} {k['chain_us']:9.1f} "
                f"{k['chain_us'] / k['calls']:8.2f} {k['raw_us']:8.1f}"
            )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


main()
