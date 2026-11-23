#!/usr/bin/env bash
# Trace decode at contexts 0 and 131072 with BF16 and FP8 dense weights, one nsys report
# per configuration, under the shared GPU lock.
# usage: decode_traces.sh <bench> <model> <outdir>
set -euo pipefail
bench=$1 model=$2 out=$3
mkdir -p "$out"
run() {
    name=$1; shift
    flock "$HOME/.cache/inkling-gpu.lock" nsys profile -f true -o "$out/$name" -t cuda \
        --cuda-graph-trace=node --cuda-memory-usage=false \
        "$bench" "$model" --steps 100 --warmup 20 "$@" > "$out/$name.log" 2>&1
    nsys export -f true -t sqlite -o "$out/$name.sqlite" "$out/$name.nsys-rep" > /dev/null
}
run d_bf16_0 --contexts 0
run d_fp8_0 --contexts 0 --weights fp8
run d_bf16_128k --contexts 131072 --kv fp8
run d_fp8_128k --contexts 131072 --kv fp8 --weights fp8
echo done
