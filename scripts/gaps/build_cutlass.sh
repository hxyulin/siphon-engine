#!/usr/bin/env bash
# Build one CUTLASS example source for sm_120a with the pixi environment's nvcc.
# usage: build_cutlass.sh <cutlass-dir> <source.cu> <output> [extra nvcc flags...]
set -euo pipefail
cutlass=$1 src=$2 out=$3
shift 3
nvcc -std=c++17 -O3 -DNDEBUG --expt-relaxed-constexpr \
    -gencode arch=compute_120a,code=sm_120a \
    -I "$cutlass/include" -I "$cutlass/tools/util/include" -I "$cutlass/examples/common" \
    "$@" -o "$out" "$src" -lcuda
