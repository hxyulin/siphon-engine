#!/usr/bin/env bash
# nsys kernel traces (graph nodes) of repeated verify passes with FP8 dense weights and an
# FP8 global cache, one report per capture and width, under the GPU lock.
# usage: traces.sh <smallm> <model> <capture-dir> <outdir> <capture> <widths...>
set -euo pipefail
smallm=$1 model=$2 caps=$3 out=$4 capture=$5
shift 5
mkdir -p "$out"
for w in "$@"; do
    name="$capture-w$w"
    flock "$HOME/.cache/inkling-gpu.lock" nsys profile -f true -o "$out/$name" -t cuda \
        --cuda-graph-trace=node --cuda-memory-usage=false \
        "$smallm" verify "$model" "$caps/$capture-chunked.safetensors" --width "$w" \
        --passes 40 --weights fp8 --kv fp8 > "$out/$name.log" 2>&1
    nsys export -f true -t sqlite -o "$out/$name.sqlite" "$out/$name.nsys-rep" > /dev/null
done
echo done
