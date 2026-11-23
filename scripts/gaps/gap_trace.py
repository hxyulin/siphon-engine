"""Split a decode or prefill nsys trace into kernel time on the main chain, idle gaps
between kernels, and side-stream prefetch time, and list per-kernel chain time.

usage: gap_trace.py <report.sqlite> <steps> [--skip N] [--top N] [--json out.json]

Kernels other than `prefetch_l2` form the main chain. A kernel's chain time is
end - max(start, previous chain kernel's end), so a PDL kernel that starts early is
charged only from the previous kernel's end. The idle time is the span minus the union of
chain kernels. `--skip N` drops the first N steps (warmup), where a step ends at each
`argmax` kernel; without it, every kernel counts and the totals divide by `steps`.
"""

import argparse
import collections
import json
import sqlite3


def load(path):
    db = sqlite3.connect(path)
    return db.execute(
        """select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k
           join StringIds s on s.id = coalesce(k.demangledName, k.shortName)
           order by k.start"""
    ).fetchall()


def short(name):
    # Drop template arguments and parameter lists so instantiations group by kernel.
    return name.split("(")[0].split("<")[0][:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("steps", type=int)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--json")
    a = ap.parse_args()

    rows = load(a.sqlite)
    if a.skip:
        ends = [i for i, r in enumerate(rows) if r[2].startswith("argmax")]
        cut = ends[a.skip - 1] + 1
        rows = rows[cut:]
        last = [i for i, r in enumerate(rows) if r[2].startswith("argmax")][a.steps - 1]
        rows = rows[: last + 1]
    steps = a.steps
    side = [r for r in rows if r[2].startswith("prefetch_l2")]
    chain = [r for r in rows if not r[2].startswith("prefetch_l2")]

    eff = collections.Counter()
    raw = collections.Counter()
    n = collections.Counter()
    lead = collections.Counter()  # time a kernel ran before the previous kernel ended
    gap_hist = collections.Counter()
    prev = chain[0][0]
    busy = 0
    idle = 0
    for st, en, name in chain:
        k = short(name)
        if st > prev:
            idle += st - prev
            gap_hist[min((st - prev) // 500, 10)] += 1
        else:
            lead[k] += prev - st
        e = en - max(st, prev)
        if e > 0:
            eff[k] += e
            busy += e
        raw[k] += en - st
        n[k] += 1
        prev = max(prev, en)
    span = prev - chain[0][0]

    # Prefetch time that overlaps chain kernels, and prefetch time while the chain is idle.
    intervals = []
    prev = 0
    for st, en, _ in chain:
        st = max(st, prev)
        if en > st:
            intervals.append((st, en))
            prev = en
    side_busy = sum(en - st for st, en, _ in side)
    overlap = 0
    j = 0
    for st, en, _ in side:
        while j < len(intervals) and intervals[j][1] <= st:
            j += 1
        k = j
        while k < len(intervals) and intervals[k][0] < en:
            overlap += min(en, intervals[k][1]) - max(st, intervals[k][0])
            k += 1

    us = lambda x: x / steps / 1e3  # noqa: E731
    print(
        f"span {us(span):.1f} us/step, chain busy {us(busy):.1f}, idle {us(idle):.1f}, "
        f"chain kernels {len(chain) / steps:.0f}/step, prefetch busy {us(side_busy):.1f} "
        f"(overlapping chain kernels {us(overlap):.1f})"
    )
    print(
        "idle gaps per step by length (0.5 us bins, last is >= 5 us):",
        [round(gap_hist[b] / steps, 1) for b in range(11)],
    )
    print(f"{'share':>6} {'chain us':>9} {'calls':>6} {'each':>7} {'raw each':>8} {'lead':>7}  kernel")
    out = []
    for k, e in eff.most_common(a.top):
        print(
            f"{e / busy * 100:5.1f}% {us(e):9.1f} {n[k] / steps:6.1f} {e / n[k] / 1e3:7.2f} "
            f"{raw[k] / n[k] / 1e3:8.2f} {lead[k] / n[k] / 1e3:7.2f}  {k}"
        )
    for k in eff:
        out.append(
            {
                "kernel": k,
                "chain_us": us(eff[k]),
                "calls": n[k] / steps,
                "raw_us": us(raw[k]),
                "lead_us": us(lead[k]),
            }
        )
    if a.json:
        with open(a.json, "w") as f:
            json.dump(
                {
                    "span_us": us(span),
                    "busy_us": us(busy),
                    "idle_us": us(idle),
                    "prefetch_us": us(side_busy),
                    "prefetch_overlap_us": us(overlap),
                    "kernels": out,
                },
                f,
                indent=1,
            )


main()
