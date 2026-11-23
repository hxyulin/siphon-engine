"""Per-kernel DRAM bytes and isolated time for the last full decode step in an ncu CSV.

usage: ncu_bytes.py <ncu.csv> [--json out.json]

Expects the metrics gpu__time_duration.sum, dram__bytes_op_read.sum,
dram__bytes_op_write.sum and lts__t_bytes.sum, collected with --cache-control all so each
kernel starts from cold caches: its DRAM bytes are then the bytes it must move once. A
step runs from an `embedding` kernel to the next `argmax` kernel. Prints each kernel's
calls, isolated time, DRAM bytes and the time those bytes take at 1.70 TB/s.
"""

import collections
import csv
import json
import sys

BW = 1.70e12
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
names = [k[i]["name"] for i in ids]
starts = [j for j, n in enumerate(names) if n.startswith("embedding")]
start = starts[-1]
end = next(j for j in range(start, len(names)) if names[j].startswith("argmax"))
agg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])
for j in range(start, end + 1):
    d = k[ids[j]]
    g = agg[d["name"].split("(")[0][:48]]
    g[0] += 1
    g[1] += d.get("gpu__time_duration.sum", 0)
    g[2] += d.get("dram__bytes_op_read.sum", 0) + d.get("dram__bytes_op_write.sum", 0)
    g[3] += d.get("lts__t_bytes.sum", 0)
tot_b = sum(g[2] for n, g in agg.items() if not n.startswith("prefetch"))
tot_t = sum(g[1] for n, g in agg.items() if not n.startswith("prefetch"))
print(f"step kernels {end - start + 1}; chain kernels' cold DRAM bytes {tot_b / 1e9:.3f} GB "
      f"-> floor {tot_b / BW * 1e6:.0f} us; isolated cold time {tot_t / 1e3:.0f} us")
print(f"{'kernel':48s} {'calls':>5} {'cold us':>9} {'MB':>9} {'floor us':>9} {'TB/s':>6} {'L2 MB':>8}")
out = {}
for n, g in sorted(agg.items(), key=lambda x: -x[1][2]):
    print(f"{n:48s} {g[0]:5d} {g[1] / 1e3:9.1f} {g[2] / 1e6:9.1f} {g[2] / BW * 1e6:9.1f} "
          f"{g[2] / g[1] / 1e3 if g[1] else 0:6.2f} {g[3] / 1e6:8.1f}")
    out[n] = {"calls": g[0], "cold_us": g[1] / 1e3, "dram_mb": g[2] / 1e6,
              "floor_us": g[2] / BW * 1e6, "l2_mb": g[3] / 1e6}
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
