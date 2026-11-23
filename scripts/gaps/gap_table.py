"""Join per-kernel chain time (gap_trace.py --json) with per-kernel DRAM floors
(ncu_bytes.py --json) and rank kernels by time left above their floor.

usage: gap_table.py <trace.json> <ncu.json> [more ncu.json for kernels the first lacks]
"""
import json
import sys

trace = json.load(open(sys.argv[1]))
floors = {}
for path in reversed(sys.argv[2:]):
    for name, d in json.load(open(path)).items():
        floors[name[:48]] = d
rows = []
for k in trace["kernels"]:
    f = floors.get(k["kernel"][:48])
    floor = f["floor_us"] if f else 0.0
    rows.append((k["kernel"], k["calls"], k["chain_us"], floor, k["chain_us"] - floor))
pf = sum(d["floor_us"] for n, d in floors.items() if n.startswith("prefetch"))
chain_floor = sum(r[3] for r in rows)
print(f"span {trace['span_us']:.0f} us, chain busy {trace['busy_us']:.0f}, idle {trace['idle_us']:.0f}; "
      f"prefetch busy {trace['prefetch_us']:.0f} moving bytes worth {pf:.0f} us at 1.70 TB/s")
print(f"sum of chain kernels' own floors {chain_floor:.0f} us (bytes the prefetch already "
      f"brought into L2 count again here)")
print(f"{'kernel':34s} {'calls':>5} {'chain us':>9} {'floor us':>9} {'share':>6} {'left us':>8}")
for name, calls, chain, floor, left in sorted(rows, key=lambda r: -r[4]):
    if chain < 3:
        continue
    share = f"{floor / chain * 100:5.0f}%" if floor else "     -"
    print(f"{name[:34]:34s} {calls:5.0f} {chain:9.1f} {floor:9.1f} {share} {left:8.1f}")
