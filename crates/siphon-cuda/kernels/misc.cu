// Message-gate combine, F32/BF16 casts and argmax, following MLX 0.32.4 on Apple GPUs.
//
// - Casts: MLX's F32-to-BF16 cast rounds to nearest even, flushes subnormal inputs
//   to a signed zero and turns every NaN into 0x7fc0. CUDA's own conversion gives
//   0x7fff for NaN, so the cast here is written out. BF16-to-F32 is a 16-bit
//   shift in both, keeping subnormals and NaN payloads.
// - The combine copies the separate MLX ops of `MessageGate.__call__`, each
//   rounded on its own with subnormals flushed. Metal's tanh is not correctly
//   rounded and nothing here reproduces it. On the random F32 inputs of the misc
//   fixture, 29% of combine outputs differ from MLX with a correctly rounded tanh
//   (through double) and 36% with tanhf, each by at most 3.1e-7 * (|h| + |m|).
// - Argmax: `mx.argmax` returns the first maximum, and treats NaN as larger than
//   every number, so the first NaN wins when there is one.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>
#include <cfloat>
#include <climits>
#include <cstdint>

#include "pdl.cuh"

namespace {

constexpr int kArgmaxThreads = 1024;
constexpr int kArgmaxWideThreads = 256;

__device__ __forceinline__ size_t grid_index() {
    return size_t(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ size_t grid_stride() {
    return size_t(gridDim.x) * blockDim.x;
}

__device__ __forceinline__ float fadd(float a, float b) {
    float r;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fsub(float a, float b) {
    float r;
    asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fmul(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float mlx_tanh(float x) {
    return float(tanh(double(x)));
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float x) {
    if (isnan(x)) {
        return 0x7fc0;
    }
    uint32_t bits = __float_as_uint(x);
    if (fabsf(x) < FLT_MIN) {
        return uint16_t((bits >> 16) & 0x8000);
    }
    bits += 0x7fff + ((bits >> 16) & 1);
    return uint16_t(bits >> 16);
}

// True if (a, ia) should be the argmax over (b, ib).
__device__ __forceinline__ bool argmax_better(float a, int ia, float b, int ib) {
    const bool a_nan = isnan(a);
    const bool b_nan = isnan(b);
    if (a_nan || b_nan) {
        return a_nan && (!b_nan || ia < ib);
    }
    return a > b || (a == b && ia < ib);
}

}  // namespace

// out = (1 - tanh(f_pre)) * h + tanh(g_pre) * m over n F32 elements.
extern "C" __global__ void message_gate_combine_f32(const float* h, const float* m,
                                                    const float* f_pre, const float* g_pre,
                                                    float* out, size_t n) {
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        const float forget = mlx_tanh(f_pre[i]);
        const float gain = mlx_tanh(g_pre[i]);
        out[i] = fadd(fmul(fsub(1.f, forget), h[i]), fmul(gain, m[i]));
    }
}

extern "C" __global__ void cast_f32_to_bf16(const float* x, uint16_t* out, size_t n) {
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        out[i] = f32_to_bf16_bits(x[i]);
    }
}

extern "C" __global__ void cast_bf16_to_f32(const uint16_t* x, float* out, size_t n) {
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        out[i] = __uint_as_float(uint32_t(x[i]) << 16);
    }
}

// out[r] = index of the largest of the n values in row r of x. One block per row.
// The comparison orders every (value, index) pair, so the reduction order does
// not change the result.
extern "C" __global__ void __launch_bounds__(kArgmaxThreads)
argmax_f32(const float* x, int* out, int n) {
    __shared__ float warp_vals[kArgmaxThreads / 32];
    __shared__ int warp_idxs[kArgmaxThreads / 32];
    const float* row = x + size_t(blockIdx.x) * n;

    // Threads without an element hold (-inf, INT_MAX), which loses to every element.
    float best = -INFINITY;
    int best_idx = INT_MAX;
    for (int i = threadIdx.x; i < n; i += kArgmaxThreads) {
        const float v = row[i];
        if (argmax_better(v, i, best, best_idx)) {
            best = v;
            best_idx = i;
        }
    }

    const auto warp_reduce = [&] {
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            const float v = __shfl_xor_sync(0xffffffffu, best, offset);
            const int idx = __shfl_xor_sync(0xffffffffu, best_idx, offset);
            if (argmax_better(v, idx, best, best_idx)) {
                best = v;
                best_idx = idx;
            }
        }
    };

    warp_reduce();
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) {
        warp_vals[warp] = best;
        warp_idxs[warp] = best_idx;
    }
    __syncthreads();
    if (warp == 0) {
        best = warp_vals[lane];
        best_idx = warp_idxs[lane];
        warp_reduce();
        if (lane == 0) {
            out[blockIdx.x] = best_idx;
        }
    }
}

// The index of the largest of the n values of one long row x, over many blocks: each
// block reduces a range of x to partial_vals and partial_idxs at blockIdx.x, and the
// last block to finish, found with `counter`, reduces those into out[0]. `counter` must
// start at 0; the last block resets it. x must be 16-byte aligned.
extern "C" __global__ void __launch_bounds__(kArgmaxWideThreads)
argmax_wide_f32(const float* x, int n, float* partial_vals, int* partial_idxs, unsigned* counter, int* out) {
    __shared__ float warp_vals[kArgmaxWideThreads / 32];
    __shared__ int warp_idxs[kArgmaxWideThreads / 32];
    __shared__ bool last;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    float best = -INFINITY;
    int best_idx = INT_MAX;
    const auto take = [&](float v, int i) {
        if (argmax_better(v, i, best, best_idx)) {
            best = v;
            best_idx = i;
        }
    };
    const auto block_reduce = [&] {
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            take(__shfl_xor_sync(0xffffffffu, best, offset), __shfl_xor_sync(0xffffffffu, best_idx, offset));
        }
        if (lane == 0) {
            warp_vals[warp] = best;
            warp_idxs[warp] = best_idx;
        }
        __syncthreads();
        if (warp == 0) {
            best = lane < kArgmaxWideThreads / 32 ? warp_vals[lane] : -INFINITY;
            best_idx = lane < kArgmaxWideThreads / 32 ? warp_idxs[lane] : INT_MAX;
#pragma unroll
            for (int offset = 16; offset >= 1; offset /= 2) {
                take(__shfl_xor_sync(0xffffffffu, best, offset), __shfl_xor_sync(0xffffffffu, best_idx, offset));
            }
        }
    };
    pdl_trigger();
    pdl_wait();
    const int vectors = n / 4;
    const int stride = gridDim.x * kArgmaxWideThreads;
    for (int v = blockIdx.x * kArgmaxWideThreads + threadIdx.x; v < vectors; v += stride) {
        const float4 q = __ldcg(reinterpret_cast<const float4*>(x) + v);
        take(q.x, 4 * v);
        take(q.y, 4 * v + 1);
        take(q.z, 4 * v + 2);
        take(q.w, 4 * v + 3);
    }
    const int tail = 4 * vectors + blockIdx.x * kArgmaxWideThreads + threadIdx.x;
    if (blockIdx.x == 0 && tail < n) {
        take(x[tail], tail);
    }
    block_reduce();
    if (threadIdx.x == 0) {
        partial_vals[blockIdx.x] = best;
        partial_idxs[blockIdx.x] = best_idx;
        __threadfence();
        last = atomicAdd(counter, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    best = -INFINITY;
    best_idx = INT_MAX;
    for (int i = threadIdx.x; i < int(gridDim.x); i += kArgmaxWideThreads) {
        take(__ldcg(partial_vals + i), __ldcg(partial_idxs + i));
    }
    __syncthreads();
    block_reduce();
    if (threadIdx.x == 0) {
        out[0] = best_idx;
        *counter = 0;
    }
}

namespace {

constexpr int kHidden = 2304;
constexpr int kGateRank = 64;
constexpr int kGateThreads = 288;
constexpr int kGateWarps = kGateThreads / 32;
// gemv_f32_sn4_bn1's layout: 4 lanes per output row, 8 rows per warp. Lane l of a row
// sums columns 16 i + 4 l + 0..3 for i = 0..3.
constexpr int kLanesPerRow = 4;
constexpr int kRowsPerWarp = 32 / kLanesPerRow;
constexpr int kGateRowsPerBlock = kGateWarps * kRowsPerWarp;
constexpr int kGateBlocks = kHidden / kGateRowsPerBlock;
constexpr int kGateSteps = kGateRank / (4 * kLanesPerRow);
static_assert(kHidden % kGateRowsPerBlock == 0, "blocks cover the row");
// The last block repeats the 576-thread sums of rms_norm_f32_2304 as virtual threads
// t and t + 288.
constexpr int kNormVirtualWarps = 2 * kGateWarps;
static_assert(kGateThreads * 8 == kHidden, "two runs of 4 per thread cover the row");

__device__ __forceinline__ float ffma(float a, float b, float c) {
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

}  // namespace

// The end of MessageGate for one row, from the BF16 down projection `down` ([64]):
// f_pre = forget @ down, g_pre = gain @ down (BF16 weights [2304, 64], widened to F32
// as they are read), then out = (1 - tanh(f_pre)) * h + tanh(g_pre) * m. Each output
// row sums in the order of MLX's GEMV for k = 64; tanh is CUDA's tanhf.
//
// With `normed` not null, also writes normed = bf16(rms_norm(out) * norm_weight), the
// RMSNorm after every query, key and value gate, summed as rms_norm_f32_2304 does.
// The last block to finish, found with `counter`, applies it. `counter` must start
// at 0 and is reset by that block.
namespace {

__device__ __forceinline__ void
message_gate_out_row(const __nv_bfloat16* down, const __nv_bfloat16* __restrict__ forget,
                     const __nv_bfloat16* __restrict__ gain, const float* h, const float* m, float* out,
                     const __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* normed,
                     unsigned* counter, float eps) {
    __shared__ float vw[kNormVirtualWarps];
    __shared__ bool last;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int sub = lane % kLanesPerRow;
    const int row = blockIdx.x * kGateRowsPerBlock + warp * kRowsPerWarp + lane / kLanesPerRow;

    uint2 fw[kGateSteps];
    uint2 gw[kGateSteps];
    float x[kGateSteps][4];
#pragma unroll
    for (int i = 0; i < kGateSteps; ++i) {
        const int col = i * 4 * kLanesPerRow + sub * 4;
        fw[i] = *reinterpret_cast<const uint2*>(forget + size_t(row) * kGateRank + col);
        gw[i] = *reinterpret_cast<const uint2*>(gain + size_t(row) * kGateRank + col);
    }
    // The norm weights, for the block that turns out to be last.
    const int ca = 4 * threadIdx.x;
    const int cb = 4 * (threadIdx.x + kGateThreads);
    float wa[4];
    float wb[4];
    if (normed != nullptr) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            wa[i] = __bfloat162float(norm_weight[ca + i]);
            wb[i] = __bfloat162float(norm_weight[cb + i]);
        }
    }
    // The gate's inputs, which earlier kernels write and which may have left L2 since:
    // the messages were written at the start of the pass.
    if (sub == 0) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(h + row));
        asm volatile("prefetch.global.L2 [%0];" ::"l"(m + row));
    }
    pdl_trigger();
    pdl_wait();
#pragma unroll
    for (int i = 0; i < kGateSteps; ++i) {
        const int col = i * 4 * kLanesPerRow + sub * 4;
        const uint2 packed = *reinterpret_cast<const uint2*>(down + col);
        const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            x[i][k] = __bfloat162float(vals[k]);
        }
    }
    float f = 0.f;
    float g = 0.f;
#pragma unroll
    for (int i = 0; i < kGateSteps; ++i) {
        const auto* fv = reinterpret_cast<const __nv_bfloat16*>(&fw[i]);
        const auto* gv = reinterpret_cast<const __nv_bfloat16*>(&gw[i]);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            f = ffma(__bfloat162float(fv[k]), x[i][k], f);
            g = ffma(__bfloat162float(gv[k]), x[i][k], g);
        }
    }
    // MLX's simd_shuffle_down tree over the row's 4 lanes; lane 0 of the row ends with it.
#pragma unroll
    for (int offset = kLanesPerRow / 2; offset > 0; offset /= 2) {
        f = fadd(f, __shfl_down_sync(0xffffffffu, f, offset));
        g = fadd(g, __shfl_down_sync(0xffffffffu, g, offset));
    }
    if (sub == 0) {
        const float forget_t = tanhf(f);
        const float gain_t = tanhf(g);
        out[row] = fadd(fmul(fsub(1.f, forget_t), h[row]), fmul(gain_t, m[row]));
    }
    if (normed == nullptr) {
        return;
    }
    // Make this block's outputs visible to the last block before counting it as done.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(counter, 1u) == kGateBlocks - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    float a[4];
    float b[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a[i] = __ldcg(out + ca + i);
        b[i] = __ldcg(out + cb + i);
    }
    float pa = 0.f;
    float pb = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        pa = fmaf(a[i], a[i], pa);
        pb = fmaf(b[i], b[i], pb);
    }
    pa = warp_sum(pa);
    pb = warp_sum(pb);
    if (lane == 0) {
        vw[warp] = pa;
        vw[warp + kGateWarps] = pb;
    }
    __syncthreads();
    const float ss = warp_sum(lane < kNormVirtualWarps ? vw[lane] : 0.f);
    // Two correctly rounded float operations; rms_norm_f32_2304 rounds a double rsqrt once.
    const float inv = 1.f / sqrtf(ss / static_cast<float>(kHidden) + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        normed[ca + i] = __float2bfloat16((a[i] * inv) * wa[i]);
        normed[cb + i] = __float2bfloat16((b[i] * inv) * wb[i]);
    }
    if (threadIdx.x == 0) {
        *counter = 0;
    }
}

}  // namespace

// Up to three gates in one launch, one per blockIdx.y (the slot s): down + 64 s,
// forget_s, gain_s, h + s * h_stride, m + 2304 s, out + 2304 s, norm_weight_s,
// normed_s and counter + s. A null normed_s skips that slot's norm.
extern "C" __global__ void __launch_bounds__(kGateThreads)
message_gate_out_2304(const __nv_bfloat16* down, const __nv_bfloat16* __restrict__ forget0,
                      const __nv_bfloat16* __restrict__ forget1, const __nv_bfloat16* __restrict__ forget2,
                      const __nv_bfloat16* __restrict__ gain0, const __nv_bfloat16* __restrict__ gain1,
                      const __nv_bfloat16* __restrict__ gain2, const float* h, int h_stride, const float* m,
                      float* out, const __nv_bfloat16* __restrict__ norm_weight0,
                      const __nv_bfloat16* __restrict__ norm_weight1,
                      const __nv_bfloat16* __restrict__ norm_weight2, __nv_bfloat16* normed0,
                      __nv_bfloat16* normed1, __nv_bfloat16* normed2, unsigned* counter, float eps) {
    // An explicit branch: nvcc 13.0 miscompiled the same selection written as a generic
    // lambda in message_gate_step_2304, giving the third gate a garbage pointer.
    const int s = blockIdx.y;
    const __nv_bfloat16* forget = s == 0 ? forget0 : forget1;
    const __nv_bfloat16* gain = s == 0 ? gain0 : gain1;
    const __nv_bfloat16* norm_weight = s == 0 ? norm_weight0 : norm_weight1;
    __nv_bfloat16* normed = s == 0 ? normed0 : normed1;
    if (s == 2) {
        forget = forget2;
        gain = gain2;
        norm_weight = norm_weight2;
        normed = normed2;
    }
    message_gate_out_row(down + s * kGateRank, forget, gain, h + size_t(s) * h_stride, m + size_t(s) * kHidden,
                         out + size_t(s) * kHidden, norm_weight, normed, counter + s, eps);
}

// ---------------------------------------------------------------------------
// The whole MessageGate for decode in one launch: the down projection and the end of
// message_gate_out_2304, for up to three gates (blockIdx.y), one cluster of 16 blocks per
// gate. Block r sums the down projection's partial products over two slices of its inner
// dimension, then adds rows 4r .. 4r + 3 of all 32 partials, which every block sends it,
// and sends the 4 BF16 results to every block. Each block then computes 144 output rows
// from the BF16 down projection. With a norm, every block sends its two partial sums of
// squares to every block, and each normalizes its own rows.
//
// The exchanges are st.async stores into the receivers' shared memory that complete on
// their mbarriers, a few per block: remote shared-memory stores are slow in volume on
// this GPU, and an all-to-all of the partial projections took 10 us per launch. The
// launch had 32 blocks per gate and two grid barriers on global counters before, and took
// 4.5 us in the decode step instead of 3.5; the sums are the same, in the same order.
// ---------------------------------------------------------------------------

namespace {

constexpr int kDownCols = 2 * kHidden / kGateBlocks;          // 144 inner columns per block
constexpr int kDownChunks = kDownCols / 8;                    // of 8 BF16 each
constexpr int kDownItems = kGateRank * kDownChunks;           // 1152 (row, chunk) pairs
constexpr int kDownPerThread = kDownItems / kGateThreads;     // 4
static_assert(kDownItems % kGateThreads == 0, "threads cover the down items");

__device__ __forceinline__ float dot8_bf16(uint4 w, uint4 x) {
    const auto* wv = reinterpret_cast<const __nv_bfloat16*>(&w);
    const auto* xv = reinterpret_cast<const __nv_bfloat16*>(&x);
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc = fmaf(__bfloat162float(wv[i]), __bfloat162float(xv[i]), acc);
    }
    return acc;
}

// The weights of message_gate_step for one gate: BF16, or E4M3 bytes with one F32 scale
// per row (the layout of linear_fp8.cu).
template <bool kFp8>
struct GateWeights {
    using Elem = std::conditional_t<kFp8, uint8_t, __nv_bfloat16>;
    const Elem* down;
    const float* down_scale;
    const Elem* forget;
    const float* forget_scale;
    const Elem* gain;
    const float* gain_scale;
};

// Two E4M3 values, widened exactly through F16.
__device__ __forceinline__ float2 e4m3x2_to_float2(uint32_t pair) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3);
    return __half22float2(__half2(h));
}

// 8 down weights of one (row, chunk) item: a uint4 of BF16 or a uint2 of E4M3.
template <bool kFp8>
using DownChunk = std::conditional_t<kFp8, uint2, uint4>;
// 4 forget or gain weights: a uint2 of BF16 or a uint32 of E4M3.
template <bool kFp8>
using RowChunk = std::conditional_t<kFp8, uint32_t, uint2>;

__device__ __forceinline__ float dot8(uint4 w, uint4 x) { return dot8_bf16(w, x); }

__device__ __forceinline__ float dot8(uint2 w, uint4 x) {
    const auto* xv = reinterpret_cast<const __nv_bfloat16*>(&x);
    const uint32_t words[2] = {w.x, w.y};
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float2 a = e4m3x2_to_float2(words[i] & 0xffffu);
        const float2 b = e4m3x2_to_float2(words[i] >> 16);
        acc = fmaf(a.x, __bfloat162float(xv[4 * i]), acc);
        acc = fmaf(a.y, __bfloat162float(xv[4 * i + 1]), acc);
        acc = fmaf(b.x, __bfloat162float(xv[4 * i + 2]), acc);
        acc = fmaf(b.y, __bfloat162float(xv[4 * i + 3]), acc);
    }
    return acc;
}

__device__ __forceinline__ void unpack4(uint2 w, float (&v)[4]) {
    const auto* wv = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = __bfloat162float(wv[k]);
    }
}

__device__ __forceinline__ void unpack4(uint32_t w, float (&v)[4]) {
    const float2 a = e4m3x2_to_float2(w & 0xffffu);
    const float2 b = e4m3x2_to_float2(w >> 16);
    v[0] = a.x;
    v[1] = a.y;
    v[2] = b.x;
    v[3] = b.y;
}

// Gate g = blockIdx.y of message_gate_step_2304 or message_gate_step_fp8_2304, as the
// header above describes. Block r does the work that blocks 2r and 2r + 1 of 32 did, each
// half with the same sums.
constexpr int kGateCluster = 16;
constexpr int kGateHalves = kGateBlocks / kGateCluster;
constexpr int kGateRowsPerRank = kGateRank / kGateCluster;
static_assert(kGateHalves * kGateCluster == kGateBlocks, "each block takes whole halves");
static_assert(kGateRowsPerRank == 4, "each block adds one float4 of rows per partial");

__device__ __forceinline__ uint32_t gate_smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t gate_cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

// Address `a` of this block's shared memory in block `rank` of the cluster.
__device__ __forceinline__ uint32_t gate_cluster_addr(uint32_t a, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
    return r;
}

__device__ __forceinline__ void gate_expect(uint32_t barrier, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(barrier), "r"(bytes) : "memory");
}

__device__ __forceinline__ void gate_wait(uint32_t barrier) {
    asm volatile(
        "{\n .reg .pred p;\n wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
        " @!p bra wait;\n}" ::"r"(barrier)
        : "memory");
}

template <bool kFp8>
__device__ __forceinline__ void
message_gate_step(const GateWeights<kFp8>& wt, const __nv_bfloat16* input, const float* h, const float* m,
                     float* out, const __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* normed,
                     __nv_bfloat16* down, float eps) {
    __shared__ float item_sums[kGateHalves][kDownItems];
    __shared__ __align__(16) float mine[kGateHalves * kGateRank];
    // Rows 4 rank .. of each of the 32 partial down projections.
    __shared__ __align__(16) float gathered[kGateBlocks][kGateRowsPerRank];
    __shared__ __align__(16) __nv_bfloat16 down_s[kGateRank];
    __shared__ __align__(16) float norm_parts[kGateBlocks];
    __shared__ float warp_ss[kGateHalves][kGateWarps];
    __shared__ __align__(8) uint64_t arrived[3];
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const uint32_t rank = gate_cluster_rank();
    const uint32_t gather_barrier = gate_smem_addr(&arrived[0]);
    const uint32_t down_barrier = gate_smem_addr(&arrived[1]);
    const uint32_t norm_barrier = gate_smem_addr(&arrived[2]);
    if (t == 0) {
        for (int i = 0; i < 3; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(gate_smem_addr(&arrived[i])) : "memory");
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        gate_expect(gather_barrier, sizeof(gathered));
        gate_expect(down_barrier, sizeof(down_s));
        if (normed != nullptr) {
            gate_expect(norm_barrier, sizeof(norm_parts));
        }
    }
    // Every block arrives now and waits before its first remote store, so the barriers of
    // the whole cluster are ready by then.
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Weights first, as message_gate_step loads them, for both halves.
    DownChunk<kFp8> dw[kGateHalves][kDownPerThread];
    RowChunk<kFp8> fw[kGateHalves][kGateSteps];
    RowChunk<kFp8> gw[kGateHalves][kGateSteps];
    float forget_scale[kGateHalves];
    float gain_scale[kGateHalves];
    float nw[kGateHalves];
    int rows[kGateHalves];
    const int sub = lane % kLanesPerRow;
#pragma unroll
    for (int half = 0; half < kGateHalves; ++half) {
        const int blk = int(rank) * kGateHalves + half;
        const int col0 = blk * kDownCols;
#pragma unroll
        for (int i = 0; i < kDownPerThread; ++i) {
            const int item = t + i * kGateThreads;
            const int k = item / kDownChunks;
            const int c = item % kDownChunks;
            dw[half][i] = *reinterpret_cast<const DownChunk<kFp8>*>(wt.down + size_t(k) * 2 * kHidden + col0 + 8 * c);
        }
        rows[half] = blk * kGateRowsPerBlock + warp * kRowsPerWarp + lane / kLanesPerRow;
#pragma unroll
        for (int i = 0; i < kGateSteps; ++i) {
            const int col = i * 4 * kLanesPerRow + sub * 4;
            fw[half][i] = *reinterpret_cast<const RowChunk<kFp8>*>(wt.forget + size_t(rows[half]) * kGateRank + col);
            gw[half][i] = *reinterpret_cast<const RowChunk<kFp8>*>(wt.gain + size_t(rows[half]) * kGateRank + col);
        }
        forget_scale[half] = 1.f;
        gain_scale[half] = 1.f;
        if constexpr (kFp8) {
            forget_scale[half] = wt.forget_scale[rows[half]];
            gain_scale[half] = wt.gain_scale[rows[half]];
        }
        nw[half] = norm_weight != nullptr ? __bfloat162float(norm_weight[rows[half]]) : 0.f;
    }
    // Thread t < 4 finishes down projection row 4 rank + t.
    float down_scale = 1.f;
    if constexpr (kFp8) {
        if (t < kGateRowsPerRank) {
            down_scale = wt.down_scale[int(rank) * kGateRowsPerRank + t];
        }
    }
    pdl_trigger();
    pdl_wait();

    float hv[kGateHalves] = {};
    float mv[kGateHalves] = {};
    if (sub == 0) {
#pragma unroll
        for (int half = 0; half < kGateHalves; ++half) {
            hv[half] = h[rows[half]];
            mv[half] = m[rows[half]];
        }
    }
#pragma unroll
    for (int half = 0; half < kGateHalves; ++half) {
        const int col0 = (int(rank) * kGateHalves + half) * kDownCols;
#pragma unroll
        for (int i = 0; i < kDownPerThread; ++i) {
            const int item = t + i * kGateThreads;
            const int c = item % kDownChunks;
            const uint4 x = *reinterpret_cast<const uint4*>(input + col0 + 8 * c);
            item_sums[half][item] = dot8(dw[half][i], x);
        }
    }
    __syncthreads();
    if (t < kGateHalves * kGateRank) {
        const int half = t / kGateRank;
        const int r = t % kGateRank;
        float acc = 0.f;
#pragma unroll
        for (int c = 0; c < kDownChunks; ++c) {
            acc += item_sums[half][r * kDownChunks + c];
        }
        mine[t] = acc;
    }
    __syncthreads();
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    // Thread q < 32 sends rows 4 d .. of this block's half h, q = 2 d + h, to block d, as
    // partial 2 rank + h.
    if (t < kGateCluster * kGateHalves) {
        const uint32_t dst_rank = uint32_t(t / kGateHalves);
        const int half = t % kGateHalves;
        const float4 v = reinterpret_cast<const float4*>(mine + half * kGateRank)[dst_rank];
        const uint32_t dst =
            gate_cluster_addr(gate_smem_addr(&gathered[int(rank) * kGateHalves + half][0]), dst_rank);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
                     "r"(__float_as_uint(v.x)), "r"(__float_as_uint(v.y)), "r"(__float_as_uint(v.z)),
                     "r"(__float_as_uint(v.w)), "r"(gate_cluster_addr(gather_barrier, dst_rank))
                     : "memory");
    }
    if (warp == 0) {
        gate_wait(gather_barrier);
        // Lane t < 4: row 4 rank + t, added over the 32 partials in order.
        float acc = 0.f;
        if (lane < kGateRowsPerRank) {
#pragma unroll 8
            for (int b = 0; b < kGateBlocks; ++b) {
                acc += gathered[b][lane];
            }
        }
        const __nv_bfloat16 d = __float2bfloat16(kFp8 ? down_scale * acc : acc);
        const uint32_t bits = __bfloat16_as_ushort(d);
        // Lane 0 packs the 4 values and lanes 0 .. 15 send them to every block.
        const uint32_t b1 = __shfl_sync(0xffffffffu, bits, 1);
        const uint32_t b2 = __shfl_sync(0xffffffffu, bits, 2);
        const uint32_t b3 = __shfl_sync(0xffffffffu, bits, 3);
        const uint32_t b0 = __shfl_sync(0xffffffffu, bits, 0);
        if (down != nullptr && lane < kGateRowsPerRank) {
            down[blockIdx.y * kGateRank + int(rank) * kGateRowsPerRank + lane] = d;
        }
        if (lane < kGateCluster) {
            const uint32_t dst =
                gate_cluster_addr(gate_smem_addr(down_s + int(rank) * kGateRowsPerRank), uint32_t(lane));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(dst),
                         "r"(b0 | (b1 << 16)), "r"(b2 | (b3 << 16)), "r"(gate_cluster_addr(down_barrier, uint32_t(lane)))
                         : "memory");
        }
    }
    gate_wait(down_barrier);

    float o[kGateHalves];
#pragma unroll
    for (int half = 0; half < kGateHalves; ++half) {
        float f = 0.f;
        float gg = 0.f;
#pragma unroll
        for (int i = 0; i < kGateSteps; ++i) {
            const int col = i * 4 * kLanesPerRow + sub * 4;
            const uint2 packed = *reinterpret_cast<const uint2*>(down_s + col);
            const auto* xv = reinterpret_cast<const __nv_bfloat16*>(&packed);
            float fv[4];
            float gv[4];
            unpack4(fw[half][i], fv);
            unpack4(gw[half][i], gv);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float x = __bfloat162float(xv[k]);
                f = ffma(fv[k], x, f);
                gg = ffma(gv[k], x, gg);
            }
        }
#pragma unroll
        for (int offset = kLanesPerRow / 2; offset > 0; offset /= 2) {
            f = fadd(f, __shfl_down_sync(0xffffffffu, f, offset));
            gg = fadd(gg, __shfl_down_sync(0xffffffffu, gg, offset));
        }
        if constexpr (kFp8) {
            f = fmul(forget_scale[half], f);
            gg = fmul(gain_scale[half], gg);
        }
        o[half] = 0.f;
        if (sub == 0) {
            o[half] = fadd(fmul(fsub(1.f, tanhf(f)), hv[half]), fmul(tanhf(gg), mv[half]));
            out[rows[half]] = o[half];
        }
    }
    if (normed == nullptr) {
        return;
    }
#pragma unroll
    for (int half = 0; half < kGateHalves; ++half) {
        const float ss = warp_sum(o[half] * o[half]);
        if (lane == 0) {
            warp_ss[half][warp] = ss;
        }
    }
    __syncthreads();
    if (warp == 0) {
        // Lanes 0 and 1 add the two halves' warp sums in message_gate_step's order, and
        // lanes 0 .. 15 send both to every block.
        float acc = 0.f;
        if (lane < kGateHalves) {
#pragma unroll
            for (int w = 0; w < kGateWarps; ++w) {
                acc += warp_ss[lane][w];
            }
        }
        const float s0 = __shfl_sync(0xffffffffu, acc, 0);
        const float s1 = __shfl_sync(0xffffffffu, acc, 1);
        if (lane < kGateCluster) {
            const uint32_t dst =
                gate_cluster_addr(gate_smem_addr(norm_parts + int(rank) * kGateHalves), uint32_t(lane));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(dst),
                         "r"(__float_as_uint(s0)), "r"(__float_as_uint(s1)),
                         "r"(gate_cluster_addr(norm_barrier, uint32_t(lane)))
                         : "memory");
        }
    }
    gate_wait(norm_barrier);
    const float total = warp_sum(norm_parts[lane]);
    const float inv = 1.f / sqrtf(total / static_cast<float>(kHidden) + eps);
    if (sub == 0) {
#pragma unroll
        for (int half = 0; half < kGateHalves; ++half) {
            normed[rows[half]] = __float2bfloat16((o[half] * inv) * nw[half]);
        }
    }
}

}  // namespace

// Gate s reads down_weight_s ([64, 4608] BF16), input_s ([4608] BF16: the normalized
// state and message), forget_s and gain_s, h + s * h_stride and m + 2304 s, and writes
// out + 2304 s and, with a norm weight, normed_s = bf16(rms_norm(out) * norm_weight_s).
// `down`, if not null, receives each gate's BF16 down projection ([slots, 64]). The grid
// is (16, gates) blocks, one cluster per gate.
extern "C" __global__ void __cluster_dims__(kGateCluster, 1, 1) __launch_bounds__(kGateThreads)
message_gate_step_2304(const __nv_bfloat16* __restrict__ down_weight0, const __nv_bfloat16* __restrict__ down_weight1,
                       const __nv_bfloat16* __restrict__ down_weight2, const __nv_bfloat16* input0,
                       const __nv_bfloat16* input1, const __nv_bfloat16* input2,
                       const __nv_bfloat16* __restrict__ forget0, const __nv_bfloat16* __restrict__ forget1,
                       const __nv_bfloat16* __restrict__ forget2, const __nv_bfloat16* __restrict__ gain0,
                       const __nv_bfloat16* __restrict__ gain1, const __nv_bfloat16* __restrict__ gain2,
                       const float* h, int h_stride, const float* m, float* out,
                       const __nv_bfloat16* __restrict__ norm_weight0, const __nv_bfloat16* __restrict__ norm_weight1,
                       const __nv_bfloat16* __restrict__ norm_weight2, __nv_bfloat16* normed0,
                       __nv_bfloat16* normed1, __nv_bfloat16* normed2, __nv_bfloat16* down, float eps) {
    const int g = blockIdx.y;
    // An explicit branch rather than a generic lambda, which nvcc 13.0 miscompiled here.
    GateWeights<false> wt{down_weight0, nullptr, forget0, nullptr, gain0, nullptr};
    const __nv_bfloat16* input = input0;
    const __nv_bfloat16* norm_weight = norm_weight0;
    __nv_bfloat16* normed = normed0;
    if (g == 1) {
        wt = {down_weight1, nullptr, forget1, nullptr, gain1, nullptr};
        input = input1;
        norm_weight = norm_weight1;
        normed = normed1;
    } else if (g == 2) {
        wt = {down_weight2, nullptr, forget2, nullptr, gain2, nullptr};
        input = input2;
        norm_weight = norm_weight2;
        normed = normed2;
    }
    message_gate_step<false>(wt, input, h + size_t(g) * h_stride, m + size_t(g) * kHidden,
                             out + size_t(g) * kHidden, norm_weight, normed, down, eps);
}

// message_gate_step_2304 with E4M3 weights: down_s, forget_s and gain_s are the bytes of
// each matrix in the layout of linear_fp8.cu, and *_scale_s their F32 row scales. Each
// product sums the exactly widened weights in F32 and multiplies the row's scale into the
// sum once.
extern "C" __global__ void __cluster_dims__(kGateCluster, 1, 1) __launch_bounds__(kGateThreads)
message_gate_step_fp8_2304(const uint8_t* __restrict__ down0, const uint8_t* __restrict__ down1,
                           const uint8_t* __restrict__ down2, const float* __restrict__ down_scale0,
                           const float* __restrict__ down_scale1, const float* __restrict__ down_scale2,
                           const __nv_bfloat16* input0, const __nv_bfloat16* input1, const __nv_bfloat16* input2,
                           const uint8_t* __restrict__ forget0, const uint8_t* __restrict__ forget1,
                           const uint8_t* __restrict__ forget2, const float* __restrict__ forget_scale0,
                           const float* __restrict__ forget_scale1, const float* __restrict__ forget_scale2,
                           const uint8_t* __restrict__ gain0, const uint8_t* __restrict__ gain1,
                           const uint8_t* __restrict__ gain2, const float* __restrict__ gain_scale0,
                           const float* __restrict__ gain_scale1, const float* __restrict__ gain_scale2,
                           const float* h, int h_stride, const float* m, float* out,
                           const __nv_bfloat16* __restrict__ norm_weight0,
                           const __nv_bfloat16* __restrict__ norm_weight1,
                           const __nv_bfloat16* __restrict__ norm_weight2, __nv_bfloat16* normed0,
                           __nv_bfloat16* normed1, __nv_bfloat16* normed2, __nv_bfloat16* down, float eps) {
    const int g = blockIdx.y;
    // An explicit branch, as in message_gate_step_2304.
    GateWeights<true> wt{down0, down_scale0, forget0, forget_scale0, gain0, gain_scale0};
    const __nv_bfloat16* input = input0;
    const __nv_bfloat16* norm_weight = norm_weight0;
    __nv_bfloat16* normed = normed0;
    if (g == 1) {
        wt = {down1, down_scale1, forget1, forget_scale1, gain1, gain_scale1};
        input = input1;
        norm_weight = norm_weight1;
        normed = normed1;
    } else if (g == 2) {
        wt = {down2, down_scale2, forget2, forget_scale2, gain2, gain_scale2};
        input = input2;
        norm_weight = norm_weight2;
        normed = normed2;
    }
    message_gate_step<true>(wt, input, h + size_t(g) * h_stride, m + size_t(g) * kHidden,
                            out + size_t(g) * kHidden, norm_weight, normed, down, eps);
}

// ---------------------------------------------------------------------------
// message_gate_out_2304 for the rows of a chunk, in two kernels: the gate outputs, then
// the optional norm. The two products with the BF16 down projection run on tensor cores
// (mma.sync with F32 accumulators), so they sum in another order than
// message_gate_out_2304 and results differ by F32 rounding. The combine after them is
// message_gate_out_2304's.
// ---------------------------------------------------------------------------

namespace {

constexpr int kGtRows = 64;
constexpr int kGtCols = 64;
constexpr int kGtThreads = 256;
// Most block states a rebuilt message mixes, as depth_attention_rows_2304 allows.
constexpr int kGtMaxBlocks = 5;
// 16-byte chunks per row of a [*, 64] BF16 tile.
constexpr int kGtChunks = kGateRank / 8;
static_assert(kHidden % kGtCols == 0, "tiles cover the row");

// Element offset of 16-byte chunk `chunk` of row `row` in a [rows, 64] BF16 tile. The XOR
// spreads the 8 rows an ldmatrix reads over distinct banks.
__device__ __forceinline__ int gate_swizzle(int row, int chunk) {
    return row * kGateRank + ((chunk ^ (row & 7)) << 3);
}

__device__ __forceinline__ void gate_ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    const auto a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

// c += a b for a 16x16 BF16 A fragment, a 16x8 BF16 B fragment and F32 c.
__device__ __forceinline__ void gate_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

}  // namespace

// For each row and column d: out[row, d] = (1 - tanh(f)) * h[row, d] + tanh(g) * m[row, d],
// with f and g the dots of down[row] ([rows, 64] BF16) with forget[d] and gain[d]
// ([2304, 64] BF16). The h and m rows are h_stride and m_stride floats apart (even); out
// is [rows, 2304]. Grid: (2304 / 64, ceil(rows / 64)), 256 threads.
//
// With probs not null, the message is rebuilt from its depth-attention candidates as
// depth_attention_rows_2304 computed it: m[row] = fmaf(w_nb, m_given[row], sum over
// i < nb of w_i * blocks[i, row], chained by fmaf from 0), where blocks[i] starts at
// blocks + i * block_stride with rows 2304 apart, and w_i = probs[row * probs_stride + i].
// nb is at most 5.
extern "C" __global__ void __launch_bounds__(kGtThreads)
message_gate_out_rows_2304(const __nv_bfloat16* down, const __nv_bfloat16* __restrict__ forget,
                           const __nv_bfloat16* __restrict__ gain, const float* h, int h_stride, const float* m,
                           int m_stride, const float* blocks, int block_stride, int nb, const float* probs,
                           int probs_stride, float* out, int rows) {
    __shared__ __align__(16) __nv_bfloat16 as[kGtRows * kGateRank];
    __shared__ __align__(16) __nv_bfloat16 fs[kGtCols * kGateRank];
    __shared__ __align__(16) __nv_bfloat16 gs[kGtCols * kGateRank];
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int d0 = blockIdx.x * kGtCols;
    const int r0 = blockIdx.y * kGtRows;
    static_assert(kGtCols * kGtChunks == 2 * kGtThreads, "two weight chunks per thread");
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = tid + i * kGtThreads;
        const int r = idx / kGtChunks;
        const int c = idx % kGtChunks;
        const size_t at = size_t(d0 + r) * kGateRank + 8 * c;
        *reinterpret_cast<uint4*>(&fs[gate_swizzle(r, c)]) = *reinterpret_cast<const uint4*>(forget + at);
        *reinterpret_cast<uint4*>(&gs[gate_swizzle(r, c)]) = *reinterpret_cast<const uint4*>(gain + at);
    }
    pdl_trigger();
    pdl_wait();
    static_assert(kGtRows * kGtChunks == 2 * kGtThreads, "two down chunks per thread");
    uint4 dv[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = tid + i * kGtThreads;
        const int r = idx / kGtChunks;
        const int c = idx % kGtChunks;
        dv[i] = make_uint4(0, 0, 0, 0);
        if (r0 + r < rows) {
            dv[i] = *reinterpret_cast<const uint4*>(down + size_t(r0 + r) * kGateRank + 8 * c);
        }
    }
    // Warp w takes rows 16 (w / 2) .. and columns 32 (w % 2) .. of the tile: 4 column
    // groups of 8, each with an f and a g accumulator. Accumulator element (half, j) of
    // column group n is row wr + lane / 4 + 8 half and column wc + 8 n + 2 (lane % 4) + j.
    // h and m load during the products.
    const int wr = warp / 2 * 16;
    const int wc = warp % 2 * 32;
    float2 hv[2][4];
    float2 mv[2][4];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int row = min(r0 + wr + lane / 4 + 8 * half, rows - 1);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int d = d0 + wc + 8 * n + 2 * (lane % 4);
            hv[half][n] = *reinterpret_cast<const float2*>(h + size_t(row) * h_stride + d);
            mv[half][n] = *reinterpret_cast<const float2*>(m + size_t(row) * m_stride + d);
        }
        if (probs != nullptr) {
            // Every load first; the bank holds at most kGtMaxBlocks states.
            const float* w = probs + size_t(row) * probs_stride;
            float2 b[kGtMaxBlocks][4];
            float wi[kGtMaxBlocks];
#pragma unroll
            for (int i = 0; i < kGtMaxBlocks; ++i) {
                if (i < nb) {
                    wi[i] = w[i];
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        const int d = d0 + wc + 8 * n + 2 * (lane % 4);
                        b[i][n] = *reinterpret_cast<const float2*>(blocks + size_t(i) * block_stride +
                                                                   size_t(row) * kHidden + d);
                    }
                }
            }
            const float wp = w[nb];
            float2 acc[4] = {};
#pragma unroll
            for (int i = 0; i < kGtMaxBlocks; ++i) {
                if (i < nb) {
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        acc[n].x = fmaf(wi[i], b[i][n].x, acc[n].x);
                        acc[n].y = fmaf(wi[i], b[i][n].y, acc[n].y);
                    }
                }
            }
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                mv[half][n].x = fmaf(wp, mv[half][n].x, acc[n].x);
                mv[half][n].y = fmaf(wp, mv[half][n].y, acc[n].y);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = tid + i * kGtThreads;
        *reinterpret_cast<uint4*>(&as[gate_swizzle(idx / kGtChunks, idx % kGtChunks)]) = dv[i];
    }
    __syncthreads();
    float acc_f[4][4] = {};
    float acc_g[4][4] = {};
#pragma unroll
    for (int k = 0; k < kGateRank / 16; ++k) {
        uint32_t a[4];
        gate_ldmatrix_x4(a, &as[gate_swizzle(wr + lane % 16, 2 * k + lane / 16)]);
#pragma unroll
        for (int np = 0; np < 2; ++np) {
            // Column groups 2 np and 2 np + 1: lanes 0-15 address the first, 16-31 the second.
            const int n = wc + 16 * np + lane % 8 + 8 * (lane / 16);
            const int chunk = 2 * k + (lane / 8) % 2;
            uint32_t bf[4];
            uint32_t bg[4];
            gate_ldmatrix_x4(bf, &fs[gate_swizzle(n, chunk)]);
            gate_ldmatrix_x4(bg, &gs[gate_swizzle(n, chunk)]);
            gate_mma(acc_f[2 * np], a, bf[0], bf[1]);
            gate_mma(acc_f[2 * np + 1], a, bf[2], bf[3]);
            gate_mma(acc_g[2 * np], a, bg[0], bg[1]);
            gate_mma(acc_g[2 * np + 1], a, bg[2], bg[3]);
        }
    }

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const int row = r0 + wr + lane / 4 + 8 * half;
        if (row >= rows) {
            continue;
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int d = d0 + wc + 8 * n + 2 * (lane % 4);
            const float o0 = fadd(fmul(fsub(1.f, tanhf(acc_f[n][2 * half])), hv[half][n].x),
                                  fmul(tanhf(acc_g[n][2 * half]), mv[half][n].x));
            const float o1 = fadd(fmul(fsub(1.f, tanhf(acc_f[n][2 * half + 1])), hv[half][n].y),
                                  fmul(tanhf(acc_g[n][2 * half + 1]), mv[half][n].y));
            *reinterpret_cast<float2*>(out + size_t(row) * kHidden + d) = make_float2(o0, o1);
        }
    }
}

// normed = bf16(rms_norm(out) * norm_weight) for each row (blockIdx.x) of out and normed
// ([rows, 2304]): the norm at the end of message_gate_out_row, with the same sums, read
// with vector loads. 288 threads.
extern "C" __global__ void __launch_bounds__(kGateThreads)
message_gate_norm_rows_2304(const float* out, const __nv_bfloat16* __restrict__ norm_weight,
                            __nv_bfloat16* normed, float eps) {
    __shared__ float vw[kNormVirtualWarps];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const size_t at = size_t(blockIdx.x) * kHidden;
    const int ca = 4 * threadIdx.x;
    const int cb = 4 * (threadIdx.x + kGateThreads);
    const uint2 wpa = *reinterpret_cast<const uint2*>(norm_weight + ca);
    const uint2 wpb = *reinterpret_cast<const uint2*>(norm_weight + cb);
    pdl_trigger();
    pdl_wait();
    const float4 av = *reinterpret_cast<const float4*>(out + at + ca);
    const float4 bv = *reinterpret_cast<const float4*>(out + at + cb);
    const float a[4] = {av.x, av.y, av.z, av.w};
    const float b[4] = {bv.x, bv.y, bv.z, bv.w};
    const auto* wa = reinterpret_cast<const __nv_bfloat16*>(&wpa);
    const auto* wb = reinterpret_cast<const __nv_bfloat16*>(&wpb);
    float pa = 0.f;
    float pb = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        pa = fmaf(a[i], a[i], pa);
        pb = fmaf(b[i], b[i], pb);
    }
    pa = warp_sum(pa);
    pb = warp_sum(pb);
    if (lane == 0) {
        vw[warp] = pa;
        vw[warp + kGateWarps] = pb;
    }
    __syncthreads();
    const float ss = warp_sum(lane < kNormVirtualWarps ? vw[lane] : 0.f);
    // Two correctly rounded float operations; rms_norm_f32_2304 rounds a double rsqrt once.
    const float inv = 1.f / sqrtf(ss / static_cast<float>(kHidden) + eps);
    uint2 na;
    uint2 nb;
    auto* va = reinterpret_cast<__nv_bfloat16*>(&na);
    auto* vb = reinterpret_cast<__nv_bfloat16*>(&nb);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        va[i] = __float2bfloat16((a[i] * inv) * __bfloat162float(wa[i]));
        vb[i] = __float2bfloat16((b[i] * inv) * __bfloat162float(wb[i]));
    }
    *reinterpret_cast<uint2*>(normed + at + ca) = na;
    *reinterpret_cast<uint2*>(normed + at + cb) = nb;
}

namespace prologue {

constexpr int kVec = 4;
constexpr int kThreads = kHidden / kVec;
constexpr int kWarps = kThreads / 32;

// The reductions and rounding of rms_norm_bf16_2304 and unweighted_rms_norm_bf16_f32_2304
// in norm.cu, which this kernel must equal bit for bit.
__device__ __forceinline__ float simd_sum(float v) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float row_sum_squares(const float (&v)[kVec], float* warp_sums) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        acc = fmaf(v[i], v[i], acc);
    }
    acc = simd_sum(acc);
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        warp_sums[threadIdx.x / 32] = acc;
    }
    __syncthreads();
    return simd_sum(lane < kWarps ? warp_sums[lane] : 0.f);
}

__device__ __forceinline__ float inv_rms(float sum_squares, float eps) {
    const float mean_eps = sum_squares / static_cast<float>(kHidden) + eps;
    return static_cast<float>(1.0 / sqrt(static_cast<double>(mean_eps)));
}

__device__ __forceinline__ void unpack(uint2 packed, float (&v)[kVec]) {
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        v[i] = __bfloat162float(vals[i]);
    }
}

__device__ __forceinline__ uint2 pack(const float (&v)[kVec]) {
    uint2 packed;
    auto* vals = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        vals[i] = __float2bfloat16(v[i]);
    }
    return packed;
}

}  // namespace prologue

// The start of a decode step, in one block: the embedding row of token ids[0] (a zero row
// for an id outside [0, vocab)), its RMS norm with embed_norm into `hidden`, and the inputs
// of the first layer's attention from `hidden`: its unweighted RMS norm into `normed` (F32)
// and its RMS norm with input_norm into `input`. Equals embedding_bf16_2304 followed by
// rms_norm_bf16_2304, unweighted_rms_norm_bf16_f32_2304 and rms_norm_bf16_2304, bit for bit.
extern "C" __global__ void __launch_bounds__(prologue::kThreads)
decode_prologue_2304(const int32_t* ids, const __nv_bfloat16* __restrict__ table, int32_t vocab,
                     const __nv_bfloat16* __restrict__ embed_norm, const __nv_bfloat16* __restrict__ input_norm,
                     __nv_bfloat16* hidden, float* normed, __nv_bfloat16* input, float eps) {
    using namespace prologue;
    __shared__ float warp_sums[2][kWarps];
    const int col = threadIdx.x * kVec;
    float we[kVec];
    float wi[kVec];
    unpack(*reinterpret_cast<const uint2*>(embed_norm + col), we);
    unpack(*reinterpret_cast<const uint2*>(input_norm + col), wi);
    pdl_trigger();
    pdl_wait();
    const int32_t id = ids[0];
    float v[kVec] = {};
    if (id >= 0 && id < vocab) {
        unpack(*reinterpret_cast<const uint2*>(table + size_t(id) * kHidden + col), v);
    }
    const float scale = inv_rms(row_sum_squares(v, warp_sums[0]), eps);
    float h[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        h[i] = __bfloat162float(__float2bfloat16(__bfloat162float(__float2bfloat16(v[i] * scale)) * we[i]));
    }
    *reinterpret_cast<uint2*>(hidden + col) = pack(h);
    const float scale_h = inv_rms(row_sum_squares(h, warp_sums[1]), eps);
    float y[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        y[i] = __bfloat162float(__float2bfloat16(h[i] * scale_h)) * wi[i];
    }
    *reinterpret_cast<float4*>(normed + col) =
        make_float4(h[0] * scale_h, h[1] * scale_h, h[2] * scale_h, h[3] * scale_h);
    *reinterpret_cast<uint2*>(input + col) = pack(y);
}
