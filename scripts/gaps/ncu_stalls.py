"""Largest warp-stall reasons per kernel from `ncu -i <report> --page raw --csv`, as warp
cycles per issued instruction (mean over launches), plus L2 and DRAM bytes per launch.

usage: ncu -i <report> --page raw --csv > raw.csv; ncu_stalls.py raw.csv
"""
import collections
import csv
import sys

PREFIX = "smsp__average_warps_issue_stalled_"
SUFFIX = "_per_issue_active.ratio"
rows = list(csv.reader(open(sys.argv[1])))
h = rows[0]
data = [r for r in rows[2:] if len(r) == len(h)]
ik = h.index("Kernel Name")
cols = {c: i for i, c in enumerate(h)}
per = collections.defaultdict(lambda: collections.defaultdict(list))
for r in data:
    name = r[ik].split("(")[0][:40]
    for c, i in cols.items():
        if (c.startswith(PREFIX) and c.endswith(SUFFIX)) or c in (
            "gpu__time_duration.sum", "lts__t_bytes.sum", "dram__bytes_op_read.sum",
            "smsp__warps_active.avg.pct_of_peak_sustained_active",
            "launch__registers_per_thread", "launch__occupancy_limit_registers",
            "launch__occupancy_limit_shared_mem", "launch__waves_per_multiprocessor"):
            try:
                per[name][c].append(float(r[i].replace(",", "")))
            except ValueError:
                pass
for name, d in per.items():
    mean = {c: sum(v) / len(v) for c, v in d.items() if v}
    stalls = sorted(((v, c[len(PREFIX):-len(SUFFIX)]) for c, v in mean.items()
                     if c.startswith(PREFIX) and "not_issued" not in c), reverse=True)[:6]
    total = sum(v for c, v in mean.items() if c.startswith(PREFIX) and "not_issued" not in c)
    print(f"{name}: {mean.get('gpu__time_duration.sum', 0) / 1e3:.2f} us, "
          f"L2 {mean.get('lts__t_bytes.sum', 0) / 1e6:.1f} MB, DRAM read "
          f"{mean.get('dram__bytes_op_read.sum', 0) / 1e6:.1f} MB, warps active "
          f"{mean.get('smsp__warps_active.avg.pct_of_peak_sustained_active', 0):.0f}%")
    print(f"   stall cycles per issued instruction (sum {total:.1f}): "
          + ", ".join(f"{c} {v:.2f}" for v, c in stalls))
