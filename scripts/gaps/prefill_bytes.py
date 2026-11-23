"""Per-kernel DRAM bytes per prefill chunk from ncu_prefill_bytes.sh output: the second
half of the launches (the timed chunk after the warmup chunk), grouped by kernel.

usage: prefill_bytes.py <np.csv> [--json out.json]
"""
import collections
import csv
import json
import sys

UNITS = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "ns": 1, "nsecond": 1,
         "us": 1e3, "usecond": 1e3, "ms": 1e6, "msecond": 1e6}
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) > 10]
h = rows[0]
ii, ik, im, iu, iv = (h.index(c) for c in ("ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"))
k = {}
for r in rows[1:]:
    d = k.setdefault(int(r[ii]), {"name": r[ik]})
    try:
        d[r[im]] = float(r[iv].replace(",", "")) * UNITS.get(r[iu], 1)
    except ValueError:
        d[r[im]] = 0.0
ids = sorted(k)
# The model load and warmup chunk come first; the timed chunk repeats the warmup chunk's
# kernel sequence, so take the last launches that match the count of the last
# embedding-to-end run.
starts = [i for i in ids if k[i]["name"].startswith("embedding")]
sel = [i for i in ids if i >= starts[-1]]
agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
for i in sel:
    d = k[i]
    g = agg[d["name"].split("(")[0][:48]]
    g[0] += 1
    g[1] += d.get("gpu__time_duration.sum", 0)
    g[2] += d.get("dram__bytes_op_read.sum", 0) + d.get("dram__bytes_op_write.sum", 0)
out = {}
print(f"{len(sel)} launches in the last chunk")
for n, g in sorted(agg.items(), key=lambda x: -x[1][2]):
    print(f"{n:48s} {g[0]:5d} {g[1] / 1e3:9.1f} us cold {g[2] / 1e6:9.1f} MB {g[2] / 1.7e12 * 1e6:8.1f} us floor")
    out[n] = {"calls": g[0], "cold_us": g[1] / 1e3, "dram_mb": g[2] / 1e6, "floor_us": g[2] / 1.7e12 * 1e6}
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
