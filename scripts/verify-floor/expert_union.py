"""Distinct experts per visit and expert bytes for verify passes of 1 to 8 rows, from a
`reference/capture.py --text ... --routing` file.

usage: expert_union.py <routing.safetensors> [--last 512] [--bench] [--json out.json]

Windows of w consecutive positions, stepping by w over the last `--last` positions (the
rows `smallm verify` runs), give each pass's rows. With `--bench`, the windows are the 40
timed passes of `bench --verify-widths` instead: offsets 3 to 42 of those positions,
stepping by one. A visit's
expert bytes are its distinct experts times one expert's w13 and w2 with scales: NVFP4
for layers 0 to 19 and MXFP8 for layers 20 to 23.
"""

import argparse
import json

import numpy as np
from safetensors.numpy import load_file

NVFP4_EXPERT = 2 * 1152 * 2304 * (0.5 + 1 / 16) + 2304 * 1152 * (0.5 + 1 / 16)
MXFP8_EXPERT = 3 * 1152 * 2304 * (1 + 1 / 32)
# Layer of each of the 48 visits: 0-5, the core 6-17 three times, 18-23.
VISIT_LAYERS = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))

ap = argparse.ArgumentParser()
ap.add_argument("routing")
ap.add_argument("--last", type=int, default=512)
ap.add_argument("--bench", action="store_true")
ap.add_argument("--json")
a = ap.parse_args()
r = load_file(a.routing)["routing"]  # [48, n, 8]
visits, n, _ = r.shape
assert visits == len(VISIT_LAYERS)
size = np.array([MXFP8_EXPERT if l >= 20 else NVFP4_EXPERT for l in VISIT_LAYERS])
base = n - a.last
out = {}
print(f"{n} positions; windows over the last {a.last}")
print("| w | distinct per visit | min | max | expert GB | w*8 GB | GB per row |")
for w in range(1, 9):
    distinct, gb = [], []
    starts = range(base + 3, base + 43) if a.bench else range(base, n - w + 1, w)
    for s in starts:
        d = np.array([len(np.unique(r[v, s : s + w])) for v in range(visits)])
        distinct.append(d.mean())
        gb.append((d * size).sum() / 1e9)
    distinct, gb = np.array(distinct), np.array(gb)
    full = 8 * w * size.sum() / 1e9
    out[w] = {"distinct": distinct.mean(), "expert_gb": gb.mean(), "windows": len(gb)}
    print(
        f"| {w} | {distinct.mean():.1f} | {distinct.min():.1f} | {distinct.max():.1f} "
        f"| {gb.mean():.3f} | {full:.3f} | {gb.mean() / w:.3f} |"
    )
if a.json:
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
