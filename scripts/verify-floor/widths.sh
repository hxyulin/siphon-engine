#!/usr/bin/env bash
# Time one verify pass per width (1 to 8) with FP8 dense weights and an FP8 global cache at
# about 2.3K (regex-readme), 10.7K (anyhow) and 32K (anyhow after filler), under the GPU lock.
# usage: widths.sh <bench> <model> <capture-dir> <outdir> [repeat]
set -euo pipefail
bench=$1 model=$2 caps=$3 out=$4 repeat=${5:-1}
mkdir -p "$out"
for r in $(seq 1 "$repeat"); do
    for spec in "regex-readme 0" "anyhow 0" "anyhow 32768"; do
        set -- $spec
        flock "$HOME/.cache/inkling-gpu.lock" "$bench" "$model" \
            --prompt "$caps/$1-chunked.safetensors" --verify-widths --contexts "$2" \
            --weights fp8 --kv fp8 > "$out/widths-$1-$2-r$r.log" 2>&1
    done
done
echo done
