#!/usr/bin/env bash
# Per-kernel DRAM bytes from cold caches (--cache-control all) for verify passes with FP8
# dense weights and an FP8 global cache, under the GPU lock. Skips the prefill's kernels
# and the first pass, then profiles the next two passes' kernels (main and prefetch).
# usage: ncu_bytes.sh <smallm> <model> <capture-dir> <outdir> <capture> <prefill-kernels> <widths...>
set -euo pipefail
smallm=$1 model=$2 caps=$3 out=$4 capture=$5 prefill=$6
shift 6
mkdir -p "$out"
per_pass=763
for w in "$@"; do
    name="$capture-w$w"
    flock "$HOME/.cache/inkling-gpu.lock" ncu --cache-control all --clock-control none \
        --metrics gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum \
        --launch-skip $((prefill + per_pass)) --launch-count $((2 * per_pass)) --csv \
        "$smallm" verify "$model" "$caps/$capture-chunked.safetensors" --width "$w" \
        --passes 3 --weights fp8 --kv fp8 > "$out/$name.csv" 2> "$out/$name.err"
done
echo done
