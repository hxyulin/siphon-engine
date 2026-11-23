#!/usr/bin/env bash
# Per-kernel DRAM bytes for one decode step with cold caches (ncu), then nsys traces of
# prefill at 32K tokens in both expert-activation modes, each under the shared GPU lock.
# usage: ncu_and_prefill.sh <bench> <model> <outdir>
set -euo pipefail
bench=$1 model=$2 out=$3
mkdir -p "$out"
lock="$HOME/.cache/inkling-gpu.lock"
metrics=gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum
nc() {
    name=$1; shift
    flock "$lock" ncu --graph-profiling node --cache-control all --clock-control none \
        --metrics "$metrics" --csv "$bench" "$model" --steps 2 --warmup 1 "$@" \
        > "$out/$name.csv" 2> "$out/$name.err"
}
nc n_bf16_0 --contexts 0
nc n_fp8_0 --contexts 0 --weights fp8
nc n_fp8_128k --contexts 131072 --kv fp8 --weights fp8
pf() {
    name=$1; shift
    flock "$lock" nsys profile -f true -o "$out/$name" -t cuda --cuda-memory-usage=false \
        "$bench" "$model" --prefill 32768 "$@" > "$out/$name.log" 2>&1
    nsys export -f true -t sqlite -o "$out/$name.sqlite" "$out/$name.nsys-rep" > /dev/null
}
pf p_bf16_32k
pf p_q_32k --expert-activations quantized
echo done
