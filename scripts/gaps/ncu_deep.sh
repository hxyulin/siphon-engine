#!/usr/bin/env bash
# ncu detail sections (throughput, occupancy, warp stalls, launch statistics) for the
# kernels with the most time left: decode with FP8 weights at context 0 and prefill with
# both expert-activation modes, warm caches, under the shared GPU lock.
# usage: ncu_deep.sh <bench> <model> <outdir>
set -euo pipefail
bench=$1 model=$2 out=$3
mkdir -p "$out"
lock="$HOME/.cache/inkling-gpu.lock"
sections="--section SpeedOfLight --section Occupancy --section WarpStateStats --section LaunchStats --section SchedulerStats --section MemoryWorkloadAnalysis --section SourceCounters"
flock "$lock" ncu -f -o "$out/deep_decode" --graph-profiling node --cache-control none --clock-control none \
    $sections -k 'regex:depth_attention_step|decode_step_bf16|message_gate_step|expert_mlp_nvfp4|moe_router_step|moe_conv_residual|gemv_fp8_k2304' \
    --launch-skip 300 --launch-count 60 "$bench" "$model" --contexts 0 --weights fp8 --steps 3 --warmup 2 > "$out/deep_decode.log" 2>&1
flock "$lock" ncu -f -o "$out/deep_decode128k" --graph-profiling node --cache-control none --clock-control none \
    $sections -k 'regex:decode_step_fp8' --launch-count 8 "$bench" "$model" --contexts 131072 --kv fp8 --weights fp8 --steps 2 --warmup 1 > "$out/deep_decode128k.log" 2>&1
flock "$lock" ncu -f -o "$out/deep_prefill" --cache-control none --clock-control none \
    $sections -k 'regex:expert_gemm_nvfp4|gemm_bf16_64x128|prefill_attention_bf16' \
    --launch-skip 100 --launch-count 12 "$bench" "$model" --prefill 8192 > "$out/deep_prefill.log" 2>&1
flock "$lock" ncu -f -o "$out/deep_prefill_q" --cache-control none --clock-control none \
    $sections -k 'regex:expert_gemm_q' --launch-skip 40 --launch-count 8 "$bench" "$model" --prefill 8192 --expert-activations quantized > "$out/deep_prefill_q.log" 2>&1
echo done
