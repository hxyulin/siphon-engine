#!/usr/bin/env bash
# Time the CUTLASS sm_120 block-scaled GEMMs on the expert shapes under nsys, holding the
# GPU lock: one expert's w13 ([2304, 2304] weight) and w2 ([2304, 1152]) at 128 rows and
# at 16384 rows, and grouped runs over 128 experts whose row counts split 16384 rows (a
# 2048-row chunk times 8 experts per row) at random, rounded up to multiples of 32.
# usage: cutlass_bench.sh <bindir> <outdir> <tile>
set -euo pipefail
bin=$1 out=$2 tile=$3
mkdir -p "$out"
python3 - "$out" <<'PY'
import random, sys
random.seed(7)
counts = [0] * 128
for _ in range(16384):
    counts[random.randrange(128)] += 1
for name, n, k in (("w13", 2304, 2304), ("w2", 2304, 1152)):
    with open(f"{sys.argv[1]}/groups_{name}.txt", "w") as f:
        for e, m in enumerate(counts):
            f.write(f"{e} {max(32, (m + 31) // 32 * 32)}x{n}x{k}\n")
print("rows per expert: min", min(counts), "max", max(counts),
      "padded total", sum(max(32, (m + 31) // 32 * 32) for m in counts))
PY
lock="$HOME/.cache/inkling-gpu.lock"
prof() {
    name=$1; shift
    flock "$lock" nsys profile -f true -o "$out/$name" -t cuda "$@" > "$out/$name.log" 2>&1
    nsys stats -f csv --force-export=true -r cuda_gpu_kern_sum "$out/$name.nsys-rep" 2>/dev/null \
        | python3 "$(dirname "$0")/kern_sum.py"
    grep -E "TFLOPS|Disposition|runtime" "$out/$name.log" | tr -s " " | head -4
}
for m in 128 16384; do
    for shape in "2304 2304" "2304 1152"; do
        set -- $shape
        echo "== NVFP4 single m=$m n=$1 k=$2"
        prof a_${m}_$2 "$bin/a_nvfp4_$tile" --m=$m --n=$1 --k=$2 --iterations=50
        echo "== MXFP8 single m=$m n=$1 k=$2"
        prof c_${m}_$2 "$bin/c_mxfp8_$tile" --m=$m --n=$1 --k=$2 --iterations=50
    done
done
for w in w13 w2; do
    echo "== NVFP4 grouped $w"
    prof d_nvfp4_$w "$bin/d_nvfp4_$tile" --benchmark="$out/groups_$w.txt" --alpha=1 --beta=0 --no_verif --iterations=20
    echo "== MXFP8 grouped $w"
    prof d_mxfp8_$w "$bin/d_mxfp8_$tile" --benchmark="$out/groups_$w.txt" --alpha=1 --beta=0 --no_verif --iterations=20
done
