"""Grid size, block size, registers and shared memory per kernel in an nsys sqlite export,
with calls per step, for estimating each kernel boundary's fixed cost.

usage: grid_sizes.py <report.sqlite> <steps>
"""
import collections
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
steps = int(sys.argv[2])
rows = db.execute(
    """select s.value, k.gridX * k.gridY * k.gridZ, k.blockX * k.blockY * k.blockZ,
              k.registersPerThread, k.staticSharedMemory + k.dynamicSharedMemory
       from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s
       on s.id = coalesce(k.demangledName, k.shortName)"""
).fetchall()
agg = collections.defaultdict(collections.Counter)
for name, grid, block, regs, smem in rows:
    agg[name.split("(")[0].split("<")[0][:48]][(grid, block, regs, smem)] += 1
for name, c in sorted(agg.items(), key=lambda x: -sum(x[1].values())):
    for (grid, block, regs, smem), n in c.most_common(3):
        print(f"{name:48s} {n / steps:7.1f}/step grid {grid:6d} block {block:5d} regs {regs:4d} smem {smem:6d}")
