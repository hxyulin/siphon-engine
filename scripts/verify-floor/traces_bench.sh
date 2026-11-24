#!/usr/bin/env bash
# nsys kernel trace (graph nodes) of `bench --verify-widths` with FP8 dense weights and an FP8
# global cache at one capture and context, under the GPU lock. Each width has its own graph.
# usage: traces_bench.sh <bench> <model> <capture-dir> <outdir> <capture> <context>
set -euo pipefail
bench=$1 model=$2 caps=$3 out=$4 capture=$5 context=$6
mkdir -p "$out"
name="bench-$capture-$context"
flock "$HOME/.cache/inkling-gpu.lock" nsys profile -f true -o "$out/$name" -t cuda \
    --cuda-graph-trace=node --cuda-memory-usage=false \
    "$bench" "$model" --prompt "$caps/$capture-chunked.safetensors" --verify-widths \
    --contexts "$context" --weights fp8 --kv fp8 > "$out/$name.log" 2>&1
nsys export -f true -t sqlite -o "$out/$name.sqlite" "$out/$name.nsys-rep" > /dev/null
echo done
