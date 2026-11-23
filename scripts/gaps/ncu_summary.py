"""Summarize an ncu report's detail sections per kernel: duration, DRAM and L2 throughput,
occupancy, waves, registers and the largest warp-stall reasons (mean over launches).

usage: ncu -i <report> --page details --csv > d.csv; ncu_summary.py d.csv
"""
import collections
import csv
import sys

WANT = {
    "Duration": "dur",
    "DRAM Throughput": "dram%",
    "L2 Cache Throughput": "l2%",
    "Memory Throughput": "mem%",
    "Compute (SM) Throughput": "sm%",
    "Achieved Occupancy": "occ%",
    "Theoretical Occupancy": "theo%",
    "Waves Per SM": "waves",
    "Registers Per Thread": "regs",
    "Grid Size": "grid",
    "Block Size": "block",
    "Warp Cycles Per Issued Instruction": "cpi",
    "Issued Warp Per Scheduler": "issue",
}

rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) > 5]
h = rows[0]
ii, ik, isec, im, iu, iv = (h.index(c) for c in ("ID", "Kernel Name", "Section Name", "Metric Name", "Metric Unit", "Metric Value"))
per = collections.defaultdict(lambda: collections.defaultdict(list))
units = {}
for r in rows[1:]:
    name = r[ik].split("(")[0][:40]
    try:
        v = float(r[iv].replace(",", ""))
    except ValueError:
        continue
    m = r[im]
    if m in WANT:
        per[name][WANT[m]].append(v)
        units[WANT[m]] = r[iu]
    elif r[isec] == "WarpStateStats" and m.startswith("Stall "):
        per[name]["stall:" + m[6:]].append(v)
for name, d in per.items():
    mean = {k: sum(v) / len(v) for k, v in d.items()}
    n = len(d.get("dur", [0]))
    base = " ".join(
        f"{k} {mean[k]:.3g}{units.get(k, '') if k == 'dur' else ''}"
        for k in WANT.values() if k in mean
    )
    stalls = sorted(((v, k[6:]) for k, v in mean.items() if k.startswith("stall:")), reverse=True)[:5]
    print(f"{name} ({n} launches): {base}")
    print("   stalls (cycles per issued instruction): " + ", ".join(f"{k} {v:.2f}" for v, k in stalls))
