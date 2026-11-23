#!/usr/bin/env bash
# Load the original, a byte-for-byte copy and control-bit patched cubins of
# softcapped_swiglu_bf16, compare outputs with the original and time them.
# usage: sass_tests.sh (in the folder holding sh, ew.cubin and the patched cubins)
set -u
# Full grid on a memory-bound size, then one warp, where each instruction's stall count
# sets the time.
for cfg in "33554432 680 256" "65536 1 32"; do
    for v in rt s4 s8 s11 s12 s15 y0 y1 st0; do
        echo "== n blocks threads = $cfg: ew.cubin vs $v.cubin"
        ./sh ew.cubin "$v.cubin" $cfg 2>&1 | tail -3
    done
done
