#!/usr/bin/env bash
# Cold-cache DRAM bytes and isolated time of every kernel in a 2048-token prefill (one
# warmup chunk and one timed chunk), under the shared GPU lock.
# usage: ncu_prefill_bytes.sh <bench> <model> <outdir> [bench flags...]
set -euo pipefail
bench=$1 model=$2 out=$3
shift 3
flock "$HOME/.cache/inkling-gpu.lock" ncu --cache-control all --clock-control none \
    --metrics gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum \
    --csv "$bench" "$model" --prefill 2048 "$@" > "$out/np_2048.csv" 2> "$out/np_2048.err"
