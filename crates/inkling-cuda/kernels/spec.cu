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
//   post-MLP). conv_backup saves them before the pass, conv_inputs_rows records each
//   visit's m input rows during it, and conv_rollback rebuilds each state as the last 3
//   rows of [old state; inputs of rows 0..a].
//
// None of these kernels is launched with programmatic dependent launch.

#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int kHidden = 2304;
constexpr int kKvDims = 1024;
constexpr int kTopK = 8;
constexpr int kTaps = 4;
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

// Copies rows of src [rows, 2304] into tap `tap` of taps [rows, 4, 2304]. One block of 288
// threads per row, 8 values per thread.
extern "C" __global__ void __launch_bounds__(288)
copy_tap(const __nv_bfloat16* src, __nv_bfloat16* taps, int tap) {
    const size_t r = blockIdx.x;
    const int c = threadIdx.x * 8;
    *reinterpret_cast<uint4*>(taps + (r * kTaps + tap) * kHidden + c) =
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

// Records one visit's conv inputs of `rows` rows into inputs [8, kVisitChannels]: the raw
// keys and values [rows, 1024], the attention output projection [rows, 2304] and the MoE
// combine of experts [rows, 8, 2304] with weights [rows, 8], rounded as
// moe_conv_residual_rows_2304 rounds it. Grid: (9, rows), 256 threads.
extern "C" __global__ void conv_inputs_rows(const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                                            const __nv_bfloat16* proj, const __nv_bfloat16* experts,
                                            const float* weights, __nv_bfloat16* inputs) {
    const size_t r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kHidden) {
        return;
    }
    __nv_bfloat16* row = inputs + r * kVisitChannels;
    if (c < kKvDims) {
        row[stream_offset(0) + c] = k_raw[r * kKvDims + c];
        row[stream_offset(1) + c] = v_raw[r * kKvDims + c];
    }
    row[stream_offset(2) + c] = proj[r * kHidden + c];
    float total = 0.f;
#pragma unroll
    for (int x = 0; x < kTopK; ++x) {
        const float e = __bfloat162float(experts[(r * kTopK + x) * kHidden + c]);
        total = __fadd_rn(total, __fmul_rn(e, weights[r * kTopK + x]));
    }
    row[stream_offset(3) + c] = __float2bfloat16(total);
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
extern "C" __global__ void __launch_bounds__(128)
ring_backup(const unsigned long long* rings, unsigned slots, const unsigned* start, __nv_bfloat16* backup) {
    const int j = blockIdx.x;
    const int v = blockIdx.y;
    const size_t slot = (*start + j) % slots;
    const int c = threadIdx.x * 8;
    const auto* keys = reinterpret_cast<const __nv_bfloat16*>(rings[2 * v]);
    const auto* values = reinterpret_cast<const __nv_bfloat16*>(rings[2 * v + 1]);
    __nv_bfloat16* dst = backup + (size_t(v) * kVerifyRows + j) * 2 * kKvDims;
    *reinterpret_cast<uint4*>(dst + c) = *reinterpret_cast<const uint4*>(keys + slot * kKvDims + c);
    *reinterpret_cast<uint4*>(dst + kKvDims + c) = *reinterpret_cast<const uint4*>(values + slot * kKvDims + c);
}

// Puts back the slots of the rejected rows j > result[0], from ring_backup's copy.
extern "C" __global__ void __launch_bounds__(128)
ring_restore(const unsigned long long* rings, unsigned slots, const unsigned* start, const __nv_bfloat16* backup,
             const int* result) {
    const int j = blockIdx.x;
    if (j <= result[0]) {
        return;
    }
    const int v = blockIdx.y;
    const size_t slot = (*start + j) % slots;
    const int c = threadIdx.x * 8;
    auto* keys = reinterpret_cast<__nv_bfloat16*>(rings[2 * v]);
    auto* values = reinterpret_cast<__nv_bfloat16*>(rings[2 * v + 1]);
    const __nv_bfloat16* src = backup + (size_t(v) * kVerifyRows + j) * 2 * kKvDims;
    *reinterpret_cast<uint4*>(keys + slot * kKvDims + c) = *reinterpret_cast<const uint4*>(src + c);
    *reinterpret_cast<uint4*>(values + slot * kKvDims + c) = *reinterpret_cast<const uint4*>(src + kKvDims + c);
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

// positions[i] = *start + i for i < rows (at most 32), for attention one row at a time.
extern "C" __global__ void row_positions(const unsigned* start, unsigned* positions, int rows) {
    if (int(threadIdx.x) < rows) {
        positions[threadIdx.x] = *start + threadIdx.x;
    }
}
