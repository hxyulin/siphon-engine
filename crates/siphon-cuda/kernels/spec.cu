// Speculative decoding: the verify pass's feature taps, its greedy acceptance, and the
// rollback of the state that rejected rows changed.
//
// A verify runs m <= 8 rows (the anchor and m - 1 drafts) at positions start..start+m-1
// through chunked prefill. Each row appends to every KV cache and shifts every conv
// state, as a step would. After the acceptance keeps rows 0..a, two kinds of state hold
// rejected rows:
// - the sliding rings, where row j overwrote slot (start + j) mod slots, which held the
//   key of position start + j - slots. ring_backup saves those slots before the pass
//   and ring_restore puts back the rejected rows' slots after it. Global caches need
//   nothing: attention reads only slots below each row's position, and later rows
//   overwrite the stale ones.
// - the conv states, F32 [3, dims] per visit and stream (keys, values, post-attention,
//   post-MLP). conv_backup saves them before the pass, each visit's last kernel
//   (moe_conv_residual_step_rows_2304 in conv.cu) records its m input rows during it, and
//   conv_rollback rebuilds each state as the last 3
//   rows of [old state; inputs of rows 0..a].
//
// None of these kernels is launched with programmatic dependent launch.

#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int kHidden = 2304;
constexpr int kKvDims = 1024;
constexpr int kVerifyRows = 8;
constexpr int kStateRows = 3;
constexpr int kStreams = 4;
// One row of conv inputs of a visit: keys, values, post-attention and post-MLP inputs.
constexpr int kVisitChannels = 2 * kKvDims + 2 * kHidden;

__device__ __forceinline__ int stream_dims(int s) {
    return s < 2 ? kKvDims : kHidden;
}

// Offset of stream s in a row of kVisitChannels.
__device__ __forceinline__ int stream_offset(int s) {
    return s < 2 ? s * kKvDims : 2 * kKvDims + (s - 2) * kHidden;
}

}  // namespace

// Copies rows of src [rows, 2304] into tap `tap` of taps [rows, count, 2304]. One block of
// 288 threads per row, 8 values per thread.
extern "C" __global__ void __launch_bounds__(288)
copy_tap(const __nv_bfloat16* src, __nv_bfloat16* taps, int tap, int count) {
    const size_t r = blockIdx.x;
    const int c = threadIdx.x * 8;
    *reinterpret_cast<uint4*>(taps + (r * count + tap) * kHidden + c) =
        *reinterpret_cast<const uint4*>(src + r * kHidden + c);
}

// Copies every conv state into backup [visits, 3 * kVisitChannels]: stream s of a visit at
// 3 * stream_offset(s). states holds the visits' F32 [3, dims] buffers, 4 per visit.
// Grid: (blocks, 4, visits).
extern "C" __global__ void conv_backup(const unsigned long long* states, float* backup) {
    const int s = blockIdx.y;
    const int v = blockIdx.z;
    const int n = kStateRows * stream_dims(s);
    const float* src = reinterpret_cast<const float*>(states[v * kStreams + s]);
    float* dst = backup + size_t(v) * kStateRows * kVisitChannels + kStateRows * stream_offset(s);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

// Sets every conv state to rows keep..keep+2 of [backup (3 rows); inputs (m rows)], where
// keep = result[0] + 1 rows of the verify were accepted. inputs is [visits, 8,
// kVisitChannels]. Grid: (blocks, 4, visits).
extern "C" __global__ void conv_rollback(const unsigned long long* states, const float* backup,
                                         const __nv_bfloat16* inputs, const int* result) {
    const int s = blockIdx.y;
    const int v = blockIdx.z;
    const int dims = stream_dims(s);
    const int offset = stream_offset(s);
    const int keep = result[0] + 1;
    float* state = reinterpret_cast<float*>(states[v * kStreams + s]);
    const float* old = backup + size_t(v) * kStateRows * kVisitChannels + kStateRows * offset;
    const __nv_bfloat16* rows = inputs + size_t(v) * kVerifyRows * kVisitChannels + offset;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < dims; c += gridDim.x * blockDim.x) {
#pragma unroll
        for (int r = 0; r < kStateRows; ++r) {
            const int i = keep + r;
            state[r * dims + c] = i < kStateRows ? old[i * dims + c]
                                                 : __bfloat162float(rows[(i - kStateRows) * kVisitChannels + c]);
        }
    }
}

// Saves the ring slots that rows 0..gridDim.x-1 of a verify at *start will overwrite:
// slot (start + j) mod slots of each sliding visit's BF16 keys and values. rings holds a
// keys and a values pointer per visit; backup is [visits, 8, 2, 1024]. Grid: (rows,
// visits), 128 threads.
// Saves the slots that rows 0..gridDim.x of a verify at *start will overwrite in each
// sliding ring: slot (*start + j) % slots of each ring's keys and values (`row_bytes`
// bytes per slot: 2048 for BF16, 1024 for FP8).
extern "C" __global__ void __launch_bounds__(128)
ring_backup(const unsigned long long* rings, unsigned slots, const unsigned* start, uint8_t* backup,
            unsigned row_bytes) {
    const int j = blockIdx.x;
    const int v = blockIdx.y;
    const size_t slot = (*start + j) % slots;
    const unsigned byte_off = threadIdx.x * 16u;
    if (byte_off >= row_bytes) {
        return;
    }
    const auto* keys = reinterpret_cast<const uint8_t*>(rings[2 * v]);
    const auto* values = reinterpret_cast<const uint8_t*>(rings[2 * v + 1]);
    uint8_t* dst = backup + (size_t(v) * kVerifyRows + j) * 2 * size_t(row_bytes);
    *reinterpret_cast<uint4*>(dst + byte_off) =
        *reinterpret_cast<const uint4*>(keys + slot * size_t(row_bytes) + byte_off);
    *reinterpret_cast<uint4*>(dst + row_bytes + byte_off) =
        *reinterpret_cast<const uint4*>(values + slot * size_t(row_bytes) + byte_off);
}

// Puts back the slots of the rejected rows j > result[0], from ring_backup's copy.
extern "C" __global__ void __launch_bounds__(128)
ring_restore(const unsigned long long* rings, unsigned slots, const unsigned* start, const uint8_t* backup,
             const int* result, unsigned row_bytes) {
    const int j = blockIdx.x;
    if (j <= result[0]) {
        return;
    }
    const int v = blockIdx.y;
    const size_t slot = (*start + j) % slots;
    const unsigned byte_off = threadIdx.x * 16u;
    if (byte_off >= row_bytes) {
        return;
    }
    auto* keys = reinterpret_cast<uint8_t*>(rings[2 * v]);
    auto* values = reinterpret_cast<uint8_t*>(rings[2 * v + 1]);
    const uint8_t* src = backup + (size_t(v) * kVerifyRows + j) * 2 * size_t(row_bytes);
    *reinterpret_cast<uint4*>(keys + slot * size_t(row_bytes) + byte_off) =
        *reinterpret_cast<const uint4*>(src + byte_off);
    *reinterpret_cast<uint4*>(values + slot * size_t(row_bytes) + byte_off) =
        *reinterpret_cast<const uint4*>(src + row_bytes + byte_off);
}

// Greedy acceptance of a verify of m rows: tokens [m] holds the anchor and m - 1 drafts,
// argmax [m] each row's greedy token. Accepts the longest prefix of drafts where draft i
// equals argmax[i - 1], or exactly *keep - 1 drafts when *keep is nonzero. Writes
// result = [accepted, tokens[0..8], argmax[0..8]] (rows past m left alone) and the next
// anchor, argmax[accepted], to anchor. One thread.
extern "C" __global__ void accept_greedy(const int* tokens, const int* argmax, const unsigned* keep, int m,
                                         int* result, int* anchor) {
    int accepted = 0;
    if (*keep != 0) {
        accepted = min(int(*keep), m) - 1;
    } else {
        while (accepted + 1 < m && tokens[accepted + 1] == argmax[accepted]) {
            ++accepted;
        }
    }
    result[0] = accepted;
    for (int i = 0; i < m; ++i) {
        result[1 + i] = tokens[i];
        result[1 + kVerifyRows + i] = argmax[i];
    }
    *anchor = argmax[accepted];
}
