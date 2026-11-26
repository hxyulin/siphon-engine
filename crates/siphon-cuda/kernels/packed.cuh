// Shared device metadata for ragged packed multi-prompt prefill (`Model::prefill_packed`).
//
// A packed prefill concatenates S slices (1 <= S <= 64) of uneven lengths into contiguous
// `[m, ...]` activation buffers (m <= 2048). Each slice has its own row range, starting KV
// position, and index into the per-visit `[S, 6]` state address table (words 0..1 are K/V
// cache pointers, words 2..5 are K/V/post-attn/post-MLP conv state pointers).

#pragma once

#include <cstdint>

// One sequence slice in a ragged packed prefill pass (16 bytes, aligned for 128-bit load).
struct alignas(16) PackedSlice {
    uint32_t row_start;  // Starting row offset in the packed `[m, ...]` tensors.
    uint32_t row_len;    // Number of rows in this slice (>= 1).
    uint32_t kv_start;   // Starting KV position of row 0 of this slice.
    uint32_t state_idx;  // Index into the per-visit `states[S, 6]` pointer table.
};

// Binary search over `slices[0..num_slices)` (sorted by `row_start`) for the slice
// containing `row`.
__device__ __forceinline__ int find_packed_slice(const PackedSlice* slices, int num_slices,
                                                 uint32_t row) {
    int lo = 0;
    int hi = num_slices - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (slices[mid].row_start <= row) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}
