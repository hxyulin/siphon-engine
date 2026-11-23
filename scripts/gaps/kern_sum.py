"""Print `nsys stats -f csv -r cuda_gpu_kern_sum` rows from stdin: calls, median and
minimum time per call, and the kernel name."""
import csv
import sys

rows = list(csv.reader(sys.stdin))
start = next(i for i, r in enumerate(rows) if "Instances" in r)
h = rows[start]
for x in rows[start + 1:]:
    if len(x) != len(h):
        continue
    d = dict(zip(h, x))
    print(f"  {int(d['Instances']):5d}x  med {float(d['Med (ns)']) / 1e3:9.2f} us  "
          f"min {float(d['Min (ns)']) / 1e3:9.2f} us  {d['Name'][:100]}")
