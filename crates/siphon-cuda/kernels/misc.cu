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

#include "cluster_sm89.cuh"
#include "pdl.cuh"
#include "quant_epilogue.cuh"

namespace {

constexpr int kArgmaxThreads = 1024;
constexpr int kArgmaxWideThreads = 256;
constexpr int kArgmaxPartialsThreads = 256;

__device__ __forceinline__ size_t grid_index() {
    return size_t(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ size_t grid_stride() {
    return size_t(gridDim.x) * blockDim.x;
}

#if SIPHON_FAST_MATH
__device__ __forceinline__ float fadd(float a, float b) { return a + b; }
__device__ __forceinline__ float fsub(float a, float b) { return a - b; }
__device__ __forceinline__ float fmul(float a, float b) { return a * b; }
__device__ __forceinline__ float mlx_tanh(float x) { return tanhf(x); }
#else
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
#endif

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
// Vectorized with 16-byte float4 loads (peeling any unaligned head/tail) and PDL-enabled.
// The comparison orders every (value, index) pair, so the reduction order does not change
// the result.
extern "C" __global__ void __launch_bounds__(kArgmaxThreads)
argmax_f32(const float* x, int* out, int n) {
    __shared__ float warp_vals[kArgmaxThreads / 32];
    __shared__ int warp_idxs[kArgmaxThreads / 32];
    pdl_trigger();
    pdl_wait();
    const float* row = x + size_t(blockIdx.x) * n;

    // Threads without an element hold (-inf, INT_MAX), which loses to every element.
    float best = -INFINITY;
    int best_idx = INT_MAX;
    const auto take = [&](float v, int idx) {
        if (argmax_better(v, idx, best, best_idx)) {
            best = v;
            best_idx = idx;
        }
    };
    const uintptr_t addr = reinterpret_cast<uintptr_t>(row);
    const int head = min(n, static_cast<int>(((16u - (addr & 15u)) & 15u) / 4u));
    if (threadIdx.x < head) {
        take(row[threadIdx.x], threadIdx.x);
    }
    const float4* vecs = reinterpret_cast<const float4*>(row + head);
    const int nvecs = (n - head) / 4;
    for (int v = threadIdx.x; v < nvecs; v += kArgmaxThreads) {
        const float4 q = vecs[v];
        const int base = head + 4 * v;
        take(q.x, base);
        take(q.y, base + 1);
        take(q.z, base + 2);
        take(q.w, base + 3);
    }
    const int tail = head + 4 * nvecs + threadIdx.x;
    if (tail < n) {
        take(row[tail], tail);
    }

    const auto warp_reduce = [&] {
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            take(__shfl_xor_sync(0xffffffffu, best, offset),
                 __shfl_xor_sync(0xffffffffu, best_idx, offset));
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

// Second stage of the fused LM head greedy argmax: reduces `n` (val, idx) partials per row
// (with row stride `stride`) into out[row]. One block per row, vectorized float4 + int4
// loads, PDL-enabled.
extern "C" __global__ void __launch_bounds__(kArgmaxPartialsThreads)
argmax_partials(const float* vals, const int* idxs, int* out, int n, int stride) {
    __shared__ float warp_vals[kArgmaxPartialsThreads / 32];
    __shared__ int warp_idxs[kArgmaxPartialsThreads / 32];
    pdl_trigger();
    pdl_wait();
    const float* vrow = vals + size_t(blockIdx.x) * stride;
    const int* irow = idxs + size_t(blockIdx.x) * stride;

    float best = -INFINITY;
    int best_idx = INT_MAX;
    const auto take = [&](float v, int idx) {
        if (argmax_better(v, idx, best, best_idx)) {
            best = v;
            best_idx = idx;
        }
    };
    const uintptr_t va = reinterpret_cast<uintptr_t>(vrow);
    const uintptr_t ia = reinterpret_cast<uintptr_t>(irow);
    if ((va & 3u) == 0 && (va & 15u) == (ia & 15u)) {
        const int head = min(n, static_cast<int>(((16u - (va & 15u)) & 15u) / 4u));
        if (threadIdx.x < head) {
            take(vrow[threadIdx.x], irow[threadIdx.x]);
        }
        const float4* vv = reinterpret_cast<const float4*>(vrow + head);
        const int4* iv = reinterpret_cast<const int4*>(irow + head);
        const int nvecs = (n - head) / 4;
        for (int v = threadIdx.x; v < nvecs; v += kArgmaxPartialsThreads) {
            const float4 fq = vv[v];
            const int4 iq = iv[v];
            take(fq.x, iq.x);
            take(fq.y, iq.y);
            take(fq.z, iq.z);
            take(fq.w, iq.w);
        }
        const int tail = head + 4 * nvecs + threadIdx.x;
        if (tail < n) {
            take(vrow[tail], irow[tail]);
        }
    } else {
        for (int i = threadIdx.x; i < n; i += kArgmaxPartialsThreads) {
            take(vrow[i], irow[i]);
        }
    }

    const auto warp_reduce = [&] {
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            take(__shfl_xor_sync(0xffffffffu, best, offset),
                 __shfl_xor_sync(0xffffffffu, best_idx, offset));
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
        best = lane < kArgmaxPartialsThreads / 32 ? warp_vals[lane] : -INFINITY;
        best_idx = lane < kArgmaxPartialsThreads / 32 ? warp_idxs[lane] : INT_MAX;
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
#if SIPHON_SM89
    return blockIdx.x % kGateCluster;
#else
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
#endif
}

#if SIPHON_SM89
// Every block's static shared memory: message_gate_step's 3 clusters, and
// gate_rows::run's 4 clusters for each of 3 gates.
__device__ sm89::Mailboxes<3 * kGateCluster, 16384> gate_step_mail;
__device__ sm89::BarrierState gate_step_barriers[3];
__device__ sm89::Mailboxes<12 * kGateCluster, 4096> gate_rows_mail;
__device__ sm89::BarrierState gate_rows_barriers[12];
#else
// Address `a` of this block's shared memory in block `rank` of the cluster.
__device__ __forceinline__ uint32_t gate_cluster_addr(uint32_t a, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
    return r;
}

__device__ __forceinline__ void gate_expect(uint32_t barrier, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(barrier), "r"(bytes) : "memory");
}
#endif

// Waits until `bytes` have arrived at `target` on `barrier`. Every thread of the calling
// warp calls it, or of the block with kBlock. Only sm_89 uses the template arguments and
// `target`, and kRows picks gate_rows_mail there.
template <bool kBlock, bool kRows>
__device__ __forceinline__ void gate_wait(uint32_t barrier, void* target, uint32_t bytes, uint32_t parity = 0) {
#if SIPHON_SM89
    (void)parity;
    if constexpr (kRows) {
        gate_rows_mail.wait<kBlock>(kGateCluster, barrier, target, bytes);
    } else {
        gate_step_mail.wait<kBlock>(kGateCluster, barrier, target, bytes);
    }
#else
    asm volatile(
        "{\n .reg .pred p;\n wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
        " @!p bra wait;\n}" ::"r"(barrier),
        "r"(parity)
        : "memory");
#endif
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
#if !SIPHON_SM89
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
#endif

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
#if !SIPHON_SM89
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
    // Thread q < 32 sends rows 4 d .. of this block's half h, q = 2 d + h, to block d, as
    // partial 2 rank + h.
    if (t < kGateCluster * kGateHalves) {
        const uint32_t dst_rank = uint32_t(t / kGateHalves);
        const int half = t % kGateHalves;
        const float4 v = reinterpret_cast<const float4*>(mine + half * kGateRank)[dst_rank];
#if SIPHON_SM89
        gate_step_mail.send<4>(kGateCluster, int(dst_rank), gate_smem_addr(&gathered[int(rank) * kGateHalves + half][0]),
                               gather_barrier,
                               {__float_as_uint(v.x), __float_as_uint(v.y), __float_as_uint(v.z), __float_as_uint(v.w)});
#else
        const uint32_t dst =
            gate_cluster_addr(gate_smem_addr(&gathered[int(rank) * kGateHalves + half][0]), dst_rank);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
                     "r"(__float_as_uint(v.x)), "r"(__float_as_uint(v.y)), "r"(__float_as_uint(v.z)),
                     "r"(__float_as_uint(v.w)), "r"(gate_cluster_addr(gather_barrier, dst_rank))
                     : "memory");
#endif
    }
    if (warp == 0) {
        gate_wait<false, false>(gather_barrier, gathered, sizeof(gathered));
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
#if SIPHON_SM89
            gate_step_mail.send<2>(kGateCluster, lane, gate_smem_addr(down_s + int(rank) * kGateRowsPerRank),
                                   down_barrier, {b0 | (b1 << 16), b2 | (b3 << 16)});
#else
            const uint32_t dst =
                gate_cluster_addr(gate_smem_addr(down_s + int(rank) * kGateRowsPerRank), uint32_t(lane));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(dst),
                         "r"(b0 | (b1 << 16)), "r"(b2 | (b3 << 16)), "r"(gate_cluster_addr(down_barrier, uint32_t(lane)))
                         : "memory");
#endif
        }
    }
    gate_wait<true, false>(down_barrier, down_s, sizeof(down_s));

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
#if SIPHON_SM89
            gate_step_mail.send<2>(kGateCluster, lane, gate_smem_addr(norm_parts + int(rank) * kGateHalves),
                                   norm_barrier, {__float_as_uint(s0), __float_as_uint(s1)});
#else
            const uint32_t dst =
                gate_cluster_addr(gate_smem_addr(norm_parts + int(rank) * kGateHalves), uint32_t(lane));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(dst),
                         "r"(__float_as_uint(s0)), "r"(__float_as_uint(s1)),
                         "r"(gate_cluster_addr(norm_barrier, uint32_t(lane)))
                         : "memory");
#endif
        }
    }
    gate_wait<true, false>(norm_barrier, norm_parts, sizeof(norm_parts));
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
extern "C" __global__ void SIPHON_CLUSTER_DIMS(kGateCluster) __launch_bounds__(kGateThreads)
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
extern "C" __global__ void SIPHON_CLUSTER_DIMS(kGateCluster) __launch_bounds__(kGateThreads)
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
// Fused B=1 decode micro-chains:
//   - depth_gate_step[_fp8]_2304 (Pre-Attention WITH 3 message gates, grid (16, 3, 1)):
//     1-query depth attention per gate cluster + unweighted RMSNorm + MessageGate +
//     weighted RMSNorm into qkv_in[0..3] (and normed_partial from cluster 0).
//   - depth_gate_mlp_step[_fp8]_2304 (Pre-MLP WITH 1 message gate, grid (16, 1, 1)):
//     post-attention short_conv_residual + 1-query depth attention + unweighted RMSNorm +
//     MessageGate + weighted RMSNorm into mlp_x (leaving moe_router_step_2304 separate
//     so its router weight loads overlap via PDL).
// ---------------------------------------------------------------------------

namespace depth_gate {

constexpr int kDaBlocks = 5;
constexpr int kDaPartial = kDaBlocks;
constexpr int kChPerBlock = kHidden / kGateCluster;  // 144
constexpr float kShortConvDivisor = static_cast<float>(2.23606797749979);

static_assert(kChPerBlock == kDownCols, "each block covers 144 channels in both phases");
static_assert(kChPerBlock % 8 == 0, "144 channels is 18 uint4s");

__device__ __forceinline__ float round_to_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

__device__ __forceinline__ float div_short_conv(float x) {
    const float r = 1.f / kShortConvDivisor;
    const float q = __fmul_rn(x, r);
    return copysignf(fmaf(fmaf(-q, kShortConvDivisor, x), r, q), x);
}

__device__ __forceinline__ float inv_rms_f32(float sum_squares, float eps) {
#if SIPHON_FAST_MATH
    return rsqrtf(sum_squares / static_cast<float>(kHidden) + eps);
#else
    return 1.f / sqrtf(sum_squares / static_cast<float>(kHidden) + eps);
#endif
}

__device__ __forceinline__ float fast_exp(float x) {
#if SIPHON_FAST_MATH
    return __expf(x);
#else
    return exp2f(__fmul_rn(x, 1.4426950408889634f));
#endif
}

// Bit-exact with depth_attention_step_2304's warp_transpose_sum(float[32]) for 8 values.
__device__ __forceinline__ float warp_transpose_sum8_exact(float (&v)[8]) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[k] += __shfl_xor_sync(0xffffffffu, v[k], 16);
        v[k] += __shfl_xor_sync(0xffffffffu, v[k], 8);
    }
#pragma unroll
    for (int half = 4; half >= 1; half /= 2) {
        const bool upper = (lane & half) != 0;
#pragma unroll
        for (int k = 0; k < half; ++k) {
            const float send = upper ? v[k] : v[k + half];
            const float keep = upper ? v[k + half] : v[k];
            v[k] = keep + __shfl_xor_sync(0xffffffffu, send, half);
        }
    }
    return v[0];
}

__device__ __forceinline__ void load4_f32(const float* p, float (&v)[4]) {
    const float4 t = *reinterpret_cast<const float4*>(p);
    v[0] = t.x;
    v[1] = t.y;
    v[2] = t.z;
    v[3] = t.w;
}

__device__ __forceinline__ void load8_f32(const float* p, int ca, int cb, float (&v)[8]) {
    const float4 a = *reinterpret_cast<const float4*>(p + ca);
    const float4 b = *reinterpret_cast<const float4*>(p + cb);
    v[0] = a.x;
    v[1] = a.y;
    v[2] = a.z;
    v[3] = a.w;
    v[4] = b.x;
    v[5] = b.y;
    v[6] = b.z;
    v[7] = b.w;
}

__device__ __forceinline__ void unpack_bf16x4(uint2 packed, float* v) {
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = __bfloat162float(vals[i]);
    }
}

__device__ __forceinline__ uint2 pack_bf16x4(const float (&v)[4]) {
    uint2 packed;
    auto* vals = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vals[i] = __float2bfloat16(v[i]);
    }
    return packed;
}

// Replicated 1-query depth attention (0-ULP bit-exact with depth_attention_step_2304, ZERO
// Phase-1 DSMEM hops) + MessageGate (exact message_gate_step[_fp8]_2304).
template <bool kFp8, bool kConvResidual>
__device__ __forceinline__ void run(
    const float* blocks, int nb, const __nv_bfloat16* partial_bf16,
    const __nv_bfloat16* __restrict__ conv_weight, float* conv_state, const __nv_bfloat16* conv_hidden,
    float* residual, float* normed_partial, const float* __restrict__ query,
    const GateWeights<kFp8>& wt, const __nv_bfloat16* input_msg, const float* m, float* out,
    const __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* normed, float eps, float scale) {
    // Only Phase 2 (3 barriers) uses st.async; ZERO Phase-1 DSMEM hops!
    struct MailSmem {
        __align__(16) float gathered[kGateBlocks][kGateRowsPerRank];  // 512 B
        __align__(16) __nv_bfloat16 down_s[kGateRank];                // 128 B
        __align__(16) float norm_parts[kGateBlocks];                  // 128 B
        __align__(8) uint64_t arrived[3];                             // 24 B
    };
    __shared__ MailSmem mail;
    __shared__ float red[kGateWarps][8];
    __shared__ float red_norm[kGateWarps];
    __shared__ __align__(16) float h_smem[kChPerBlock];
    __shared__ __align__(16) float conv_upd[kConvResidual ? 3 : 1][kConvResidual ? kChPerBlock : 1];
    __shared__ __align__(16) __nv_bfloat16 x_down_smem[2 * kDownCols];
    __shared__ __align__(16) float mine[kGateHalves * kGateRank];
    __shared__ float warp_ss[kGateHalves][kGateWarps];

    union Phase12 {
        struct {
            uint4 conv_taps[kConvResidual ? 4 : 1][kConvResidual ? kGateThreads : 1];
            float q_s[8][kGateThreads];
        } pre;
        float item_sums[kGateHalves][kDownItems];
    };
    __shared__ __align__(16) Phase12 p23;

    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int sub = lane % kLanesPerRow;
    const int ca = 4 * t;
    const int cb = 4 * (t + kGateThreads);
    const uint32_t rank = gate_cluster_rank();

    const uint32_t gather_barrier = gate_smem_addr(&mail.arrived[0]);
    const uint32_t down_barrier = gate_smem_addr(&mail.arrived[1]);
    const uint32_t norm_barrier = gate_smem_addr(&mail.arrived[2]);

#if !SIPHON_SM89
    if (t == 0) {
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(gate_smem_addr(&mail.arrived[i])) : "memory");
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        gate_expect(gather_barrier, sizeof(mail.gathered));
        gate_expect(down_barrier, sizeof(mail.down_s));
        gate_expect(norm_barrier, sizeof(mail.norm_parts));
    }
    if constexpr (!kConvResidual) {
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    }
#endif

    // Preload all read-only weights before pdl_wait; stage q and conv_taps in SMEM
    // (reusing Phase 2 item_sums) so register count stays <= 112 regs/thread for
    // cluster co-residency with gemv_bf16_l32_v12 and prefetch_l2_paced on Blackwell.
    {
        float q_pre[8];
        load8_f32(query, ca, cb, q_pre);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            p23.pre.q_s[e][t] = q_pre[e];
        }
    }
    if constexpr (kConvResidual) {
        const uint4* cwa = reinterpret_cast<const uint4*>(conv_weight + 4 * ca);
        const uint4* cwb = reinterpret_cast<const uint4*>(conv_weight + 4 * cb);
        p23.pre.conv_taps[0][t] = cwa[0];
        p23.pre.conv_taps[1][t] = cwa[1];
        p23.pre.conv_taps[2][t] = cwb[0];
        p23.pre.conv_taps[3][t] = cwb[1];
    }

    DownChunk<kFp8> dw[kGateHalves][kDownPerThread];
    RowChunk<kFp8> fw[kGateHalves][kGateSteps];
    RowChunk<kFp8> gw[kGateHalves][kGateSteps];
    float forget_scale[kGateHalves];
    float gain_scale[kGateHalves];
    float nw[kGateHalves];
    int rows[kGateHalves];
    const int ch0 = int(rank) * kChPerBlock;
#pragma unroll
    for (int half = 0; half < kGateHalves; ++half) {
        const int blk = int(rank) * kGateHalves + half;
        const int col0 = blk * kDownCols;
#pragma unroll
        for (int i = 0; i < kDownPerThread; ++i) {
            const int item = t + i * kGateThreads;
            const int k = item / kDownChunks;
            const int ch = item % kDownChunks;
            dw[half][i] = *reinterpret_cast<const DownChunk<kFp8>*>(wt.down + size_t(k) * 2 * kHidden + col0 + 8 * ch);
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
        nw[half] = __bfloat162float(norm_weight[rows[half]]);
    }
    float down_scale = 1.f;
    if constexpr (kFp8) {
        if (t < kGateRowsPerRank) {
            down_scale = wt.down_scale[int(rank) * kGateRowsPerRank + t];
        }
    }

    pdl_trigger();
    pdl_wait();

    // Phase 1: exact 1-query depth_attention_step_2304 (ZERO DSMEM hops!).
    // Compute p[8] first, then stream blocks[i] once into s[5][8].
    const bool own_a = int(rank) < 8 && (t / 36) == int(rank);
    const bool own_b = int(rank) >= 8 && (t / 36) == int(rank) - 8;

    float p[8];
    if constexpr (kConvResidual) {
        const uint4 conv_taps[4] = {
            p23.pre.conv_taps[0][t], p23.pre.conv_taps[1][t], p23.pre.conv_taps[2][t], p23.pre.conv_taps[3][t]};
        const auto* cw = reinterpret_cast<const __nv_bfloat16*>(conv_taps);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int col = half == 0 ? ca : cb;
            const bool own = half == 0 ? own_a : own_b;
            float xh[4];
            unpack_bf16x4(*reinterpret_cast<const uint2*>(partial_bf16 + col), xh);
            float w0[4], w1[4], w2[4];
            load4_f32(conv_state + col, w0);
            load4_f32(conv_state + kHidden + col, w1);
            load4_f32(conv_state + 2 * kHidden + col, w2);
            float hid[4];
            if (conv_hidden != nullptr) {
                unpack_bf16x4(*reinterpret_cast<const uint2*>(conv_hidden + col), hid);
            }
            const int base = 4 * half;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float acc = 0.f;
                acc = __fadd_rn(acc, __fmul_rn(round_to_bf16(w0[e]), __bfloat162float(cw[4 * (base + e) + 0])));
                acc = __fadd_rn(acc, __fmul_rn(round_to_bf16(w1[e]), __bfloat162float(cw[4 * (base + e) + 1])));
                acc = __fadd_rn(acc, __fmul_rn(round_to_bf16(w2[e]), __bfloat162float(cw[4 * (base + e) + 2])));
                acc = __fadd_rn(acc, __fmul_rn(round_to_bf16(xh[e]), __bfloat162float(cw[4 * (base + e) + 3])));
                const float y = round_to_bf16(div_short_conv(__fadd_rn(round_to_bf16(acc), xh[e])));
                p[base + e] = conv_hidden != nullptr ? __fadd_rn(hid[e], y) : y;
            }
            if (own) {
                *reinterpret_cast<float4*>(residual + col) =
                    make_float4(p[base], p[base + 1], p[base + 2], p[base + 3]);
                const int off = col - ch0;
                *reinterpret_cast<float4*>(conv_upd[0] + off) = make_float4(w1[0], w1[1], w1[2], w1[3]);
                *reinterpret_cast<float4*>(conv_upd[1] + off) = make_float4(w2[0], w2[1], w2[2], w2[3]);
                *reinterpret_cast<float4*>(conv_upd[2] + off) = make_float4(xh[0], xh[1], xh[2], xh[3]);
            }
        }
        // Order all blocks' conv_state reads (and mbarrier.init) before any block writes conv_state
        // after barrier.cluster.wait below — zero extra cluster barriers!
        __syncthreads();
#if !SIPHON_SM89
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
#endif
    } else {
        unpack_bf16x4(*reinterpret_cast<const uint2*>(partial_bf16 + ca), p);
        unpack_bf16x4(*reinterpret_cast<const uint2*>(partial_bf16 + cb), p + 4);
    }

    float q[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        q[e] = p23.pre.q_s[e][t];
    }
    float s[kDaBlocks][8];
    float v[8] = {};
#pragma unroll
    for (int i = 0; i < kDaBlocks; ++i) {
        if (i < nb) {
            load8_f32(blocks + size_t(i) * kHidden, ca, cb, s[i]);
            float acc = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc = fmaf(s[i][e], q[e], acc);
            }
            v[i] = acc;
        }
    }
    {
        float acc_q = 0.f;
        float acc_s = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            acc_q = fmaf(p[e], q[e], acc_q);
            acc_s = fmaf(p[e], p[e], acc_s);
        }
        v[kDaPartial] = acc_q;
        v[kDaPartial + 1] = acc_s;
    }
    const float wdot = warp_transpose_sum8_exact(v);
    if (lane < 8) {
        red[warp][lane] = wdot;
    }
    __syncthreads();
    float tot = 0.f;
    if (lane < 8) {
#pragma unroll
        for (int w = 0; w < kGateWarps; ++w) {
            tot += red[w][lane];
        }
    }
    const float squares = __shfl_sync(0xffffffffu, tot, kDaPartial + 1);
    const float inv_p = inv_rms_f32(squares, eps);

    if constexpr (!kConvResidual) {
        if (normed_partial != nullptr && blockIdx.y == 0) {
            if (own_a) {
                *reinterpret_cast<float4*>(normed_partial + ca) =
                    make_float4(p[0] * inv_p, p[1] * inv_p, p[2] * inv_p, p[3] * inv_p);
            }
            if (own_b) {
                *reinterpret_cast<float4*>(normed_partial + cb) =
                    make_float4(p[4] * inv_p, p[5] * inv_p, p[6] * inv_p, p[7] * inv_p);
            }
        }
    }

    const bool valid = lane < nb || lane == kDaPartial;
    const float logit_da = valid ? (lane == kDaPartial ? tot * inv_p : tot) * scale : -INFINITY;
    float mx = logit_da;
#pragma unroll
    for (int offset = 1; offset < 8; offset *= 2) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, offset));
    }
    const float ex = valid ? fast_exp(logit_da - mx) : 0.f;
    float z = ex;
#pragma unroll
    for (int offset = 1; offset < 8; offset *= 2) {
        z += __shfl_xor_sync(0xffffffffu, z, offset);
    }
    const float prob = valid ? ex / z : 0.f;
    float pr[kDaBlocks + 1];
#pragma unroll
    for (int i = 0; i <= kDaBlocks; ++i) {
        pr[i] = __shfl_sync(0xffffffffu, prob, i);
    }

    float h_vec[8];
    const float pp = pr[kDaPartial] * inv_p;
    float h_acc = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < kDaBlocks; ++i) {
            if (i < nb) {
                acc = fmaf(pr[i], s[i][e], acc);
            }
        }
        h_vec[e] = fmaf(pp, p[e], acc);
        h_acc = fmaf(h_vec[e], h_vec[e], h_acc);
    }
    // Stage this block's 144 channels of `h` directly into local SMEM.
    if (own_a) {
        *reinterpret_cast<float4*>(h_smem + (ca - ch0)) =
            make_float4(h_vec[0], h_vec[1], h_vec[2], h_vec[3]);
    }
    if (own_b) {
        *reinterpret_cast<float4*>(h_smem + (cb - ch0)) =
            make_float4(h_vec[4], h_vec[5], h_vec[6], h_vec[7]);
    }
    h_acc = warp_sum(h_acc);
    if (lane == 0) {
        red_norm[warp] = h_acc;
    }
    __syncthreads();

    // Ranks 0..7 stage their 288 BF16 `h_norm` channels directly into local SMEM `x_down_smem`.
    if (rank < kGateCluster / 2) {
        float h_sum = 0.f;
#pragma unroll
        for (int w = 0; w < kGateWarps; ++w) {
            h_sum += red_norm[w];
        }
        const float inv_h = inv_rms_f32(h_sum, eps);
        const int d0 = int(rank) * 2 * kDownCols;
        if (rank < 4 && (t / 72) == int(rank)) {
            const float hn[4] = {h_vec[0] * inv_h, h_vec[1] * inv_h, h_vec[2] * inv_h, h_vec[3] * inv_h};
            *reinterpret_cast<uint2*>(x_down_smem + (ca - d0)) = pack_bf16x4(hn);
        } else if (rank >= 4 && (t / 72) == int(rank) - 4) {
            const float hn[4] = {h_vec[4] * inv_h, h_vec[5] * inv_h, h_vec[6] * inv_h, h_vec[7] * inv_h};
            *reinterpret_cast<uint2*>(x_down_smem + (cb - d0)) = pack_bf16x4(hn);
        }
        __syncthreads();
    }

    // Phase 2: MessageGate (exact message_gate_step_2304 / message_gate_step_fp8_2304).
    float hv[kGateHalves] = {};
    float mv[kGateHalves] = {};
    if (sub == 0) {
#pragma unroll
        for (int half = 0; half < kGateHalves; ++half) {
            hv[half] = h_smem[rows[half] - ch0];
            mv[half] = m[rows[half]];
        }
    }
    if (rank < kGateCluster / 2) {
#pragma unroll
        for (int half = 0; half < kGateHalves; ++half) {
#pragma unroll
            for (int i = 0; i < kDownPerThread; ++i) {
                const int item = t + i * kGateThreads;
                const int ch = item % kDownChunks;
                const uint4 x = *reinterpret_cast<const uint4*>(x_down_smem + half * kDownCols + 8 * ch);
                p23.item_sums[half][item] = dot8(dw[half][i], x);
            }
        }
    } else {
        const int msg_base = (int(rank) - kGateCluster / 2) * kGateHalves * kDownCols;
#pragma unroll
        for (int half = 0; half < kGateHalves; ++half) {
            const int col0 = msg_base + half * kDownCols;
#pragma unroll
            for (int i = 0; i < kDownPerThread; ++i) {
                const int item = t + i * kGateThreads;
                const int ch = item % kDownChunks;
                const uint4 x = *reinterpret_cast<const uint4*>(input_msg + col0 + 8 * ch);
                p23.item_sums[half][item] = dot8(dw[half][i], x);
            }
        }
    }
    __syncthreads();
    if (t < kGateHalves * kGateRank) {
        const int half = t / kGateRank;
        const int r = t % kGateRank;
        float acc = 0.f;
#pragma unroll
        for (int ch = 0; ch < kDownChunks; ++ch) {
            acc += p23.item_sums[half][r * kDownChunks + ch];
        }
        mine[t] = acc;
    }
    __syncthreads();
#if SIPHON_SM89
    if constexpr (kConvResidual) {
        sm89::barrier(gate_step_barriers, 3, kGateCluster);
    }
#else
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
    if (t < kGateCluster * kGateHalves) {
        const uint32_t dst_rank = uint32_t(t / kGateHalves);
        const int half = t % kGateHalves;
        const float4 v4 = reinterpret_cast<const float4*>(mine + half * kGateRank)[dst_rank];
#if SIPHON_SM89
        gate_step_mail.send<4>(kGateCluster, int(dst_rank),
                               gate_smem_addr(&mail.gathered[int(rank) * kGateHalves + half][0]), gather_barrier,
                               {__float_as_uint(v4.x), __float_as_uint(v4.y), __float_as_uint(v4.z),
                                __float_as_uint(v4.w)});
#else
        const uint32_t dst =
            gate_cluster_addr(gate_smem_addr(&mail.gathered[int(rank) * kGateHalves + half][0]), dst_rank);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                     ::"r"(dst), "r"(__float_as_uint(v4.x)), "r"(__float_as_uint(v4.y)), "r"(__float_as_uint(v4.z)),
                     "r"(__float_as_uint(v4.w)), "r"(gate_cluster_addr(gather_barrier, dst_rank))
                     : "memory");
#endif
    }
    if constexpr (kConvResidual) {
        // Warps 1..2 write this rank's 144 conv_state channels hidden behind warp 0's gather_barrier wait.
        if (t >= 32 && t < 32 + kChPerBlock / 4) {
            const int off = 4 * (t - 32);
            const int col = ch0 + off;
            *reinterpret_cast<float4*>(conv_state + col) =
                *reinterpret_cast<const float4*>(conv_upd[0] + off);
            *reinterpret_cast<float4*>(conv_state + kHidden + col) =
                *reinterpret_cast<const float4*>(conv_upd[1] + off);
            *reinterpret_cast<float4*>(conv_state + 2 * kHidden + col) =
                *reinterpret_cast<const float4*>(conv_upd[2] + off);
        }
    }
    if (warp == 0) {
        gate_wait<false, false>(gather_barrier, mail.gathered, sizeof(mail.gathered));
        float acc = 0.f;
        if (lane < kGateRowsPerRank) {
#pragma unroll 8
            for (int b = 0; b < kGateBlocks; ++b) {
                acc += mail.gathered[b][lane];
            }
        }
        const __nv_bfloat16 d = __float2bfloat16(kFp8 ? down_scale * acc : acc);
        const uint32_t bits = __bfloat16_as_ushort(d);
        const uint32_t b1 = __shfl_sync(0xffffffffu, bits, 1);
        const uint32_t b2 = __shfl_sync(0xffffffffu, bits, 2);
        const uint32_t b3 = __shfl_sync(0xffffffffu, bits, 3);
        const uint32_t b0 = __shfl_sync(0xffffffffu, bits, 0);
        if (lane < kGateCluster) {
#if SIPHON_SM89
            gate_step_mail.send<2>(kGateCluster, lane, gate_smem_addr(mail.down_s + int(rank) * kGateRowsPerRank),
                                   down_barrier, {b0 | (b1 << 16), b2 | (b3 << 16)});
#else
            const uint32_t dst =
                gate_cluster_addr(gate_smem_addr(mail.down_s + int(rank) * kGateRowsPerRank), uint32_t(lane));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                         ::"r"(dst), "r"(b0 | (b1 << 16)), "r"(b2 | (b3 << 16)),
                         "r"(gate_cluster_addr(down_barrier, uint32_t(lane)))
                         : "memory");
#endif
        }
    }
    gate_wait<true, false>(down_barrier, mail.down_s, sizeof(mail.down_s));

    float o[kGateHalves];
#pragma unroll
    for (int half = 0; half < kGateHalves; ++half) {
        float f = 0.f;
        float gg = 0.f;
#pragma unroll
        for (int i = 0; i < kGateSteps; ++i) {
            const int col = i * 4 * kLanesPerRow + sub * 4;
            const uint2 packed = *reinterpret_cast<const uint2*>(mail.down_s + col);
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
            if (out != nullptr) {
                out[rows[half]] = o[half];
            }
        }
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
#if SIPHON_SM89
            gate_step_mail.send<2>(kGateCluster, lane, gate_smem_addr(mail.norm_parts + int(rank) * kGateHalves),
                                   norm_barrier, {__float_as_uint(s0), __float_as_uint(s1)});
#else
            const uint32_t dst =
                gate_cluster_addr(gate_smem_addr(mail.norm_parts + int(rank) * kGateHalves), uint32_t(lane));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                         ::"r"(dst), "r"(__float_as_uint(s0)), "r"(__float_as_uint(s1)),
                         "r"(gate_cluster_addr(norm_barrier, uint32_t(lane)))
                         : "memory");
#endif
        }
    }
    gate_wait<true, false>(norm_barrier, mail.norm_parts, sizeof(mail.norm_parts));
    const float total = warp_sum(mail.norm_parts[lane]);
    const float inv = inv_rms_f32(total, eps);
    if (sub == 0) {
#pragma unroll
        for (int half = 0; half < kGateHalves; ++half) {
            normed[rows[half]] = __float2bfloat16((o[half] * inv) * nw[half]);
        }
    }
}

}  // namespace depth_gate

// Pre-attention depth attention + 3 message gates (BF16 weights), grid (16, 3, 1), 288 threads.
extern "C" __global__ void SIPHON_CLUSTER_DIMS(kGateCluster) __launch_bounds__(kGateThreads)
depth_gate_step_2304(const float* blocks, int nb, const __nv_bfloat16* partial_bf16, float* normed_partial,
                     const float* __restrict__ queries, int q_stride,
                     const __nv_bfloat16* __restrict__ down_weight0,
                     const __nv_bfloat16* __restrict__ down_weight1,
                     const __nv_bfloat16* __restrict__ down_weight2, const __nv_bfloat16* input0,
                     const __nv_bfloat16* input1, const __nv_bfloat16* input2,
                     const __nv_bfloat16* __restrict__ forget0, const __nv_bfloat16* __restrict__ forget1,
                     const __nv_bfloat16* __restrict__ forget2, const __nv_bfloat16* __restrict__ gain0,
                     const __nv_bfloat16* __restrict__ gain1, const __nv_bfloat16* __restrict__ gain2,
                     const float* m, float* out, const __nv_bfloat16* __restrict__ norm_weight0,
                     const __nv_bfloat16* __restrict__ norm_weight1,
                     const __nv_bfloat16* __restrict__ norm_weight2, __nv_bfloat16* normed0,
                     __nv_bfloat16* normed1, __nv_bfloat16* normed2, float eps, float scale) {
    const int g = blockIdx.y;
    GateWeights<false> wt{down_weight0, nullptr, forget0, nullptr, gain0, nullptr};
    const __nv_bfloat16* input_msg = input0;
    const __nv_bfloat16* norm_weight = norm_weight0;
    __nv_bfloat16* normed = normed0;
    if (g == 1) {
        wt = {down_weight1, nullptr, forget1, nullptr, gain1, nullptr};
        input_msg = input1;
        norm_weight = norm_weight1;
        normed = normed1;
    } else if (g == 2) {
        wt = {down_weight2, nullptr, forget2, nullptr, gain2, nullptr};
        input_msg = input2;
        norm_weight = norm_weight2;
        normed = normed2;
    }
    float* out_g = out != nullptr ? out + size_t(g) * kHidden : nullptr;
    depth_gate::run<false, false>(blocks, nb, partial_bf16, nullptr, nullptr, nullptr, nullptr, normed_partial,
                                  queries + size_t(g) * q_stride, wt, input_msg, m + size_t(g) * kHidden,
                                  out_g, norm_weight, normed, eps, scale);
}

// Pre-attention depth attention + 3 message gates (FP8 weights), grid (16, 3, 1), 288 threads.
extern "C" __global__ void SIPHON_CLUSTER_DIMS(kGateCluster) __launch_bounds__(kGateThreads)
depth_gate_step_fp8_2304(const float* blocks, int nb, const __nv_bfloat16* partial_bf16,
                         float* normed_partial, const float* __restrict__ queries, int q_stride,
                         const uint8_t* __restrict__ down_weight0, const uint8_t* __restrict__ down_weight1,
                         const uint8_t* __restrict__ down_weight2, const float* __restrict__ down_scale0,
                         const float* __restrict__ down_scale1, const float* __restrict__ down_scale2,
                         const __nv_bfloat16* input0, const __nv_bfloat16* input1,
                         const __nv_bfloat16* input2, const uint8_t* __restrict__ forget0,
                         const uint8_t* __restrict__ forget1, const uint8_t* __restrict__ forget2,
                         const float* __restrict__ forget_scale0, const float* __restrict__ forget_scale1,
                         const float* __restrict__ forget_scale2, const uint8_t* __restrict__ gain0,
                         const uint8_t* __restrict__ gain1, const uint8_t* __restrict__ gain2,
                         const float* __restrict__ gain_scale0, const float* __restrict__ gain_scale1,
                         const float* __restrict__ gain_scale2, const float* m, float* out,
                         const __nv_bfloat16* __restrict__ norm_weight0,
                         const __nv_bfloat16* __restrict__ norm_weight1,
                         const __nv_bfloat16* __restrict__ norm_weight2, __nv_bfloat16* normed0,
                         __nv_bfloat16* normed1, __nv_bfloat16* normed2, float eps, float scale) {
    const int g = blockIdx.y;
    GateWeights<true> wt{down_weight0, down_scale0, forget0, forget_scale0, gain0, gain_scale0};
    const __nv_bfloat16* input_msg = input0;
    const __nv_bfloat16* norm_weight = norm_weight0;
    __nv_bfloat16* normed = normed0;
    if (g == 1) {
        wt = {down_weight1, down_scale1, forget1, forget_scale1, gain1, gain_scale1};
        input_msg = input1;
        norm_weight = norm_weight1;
        normed = normed1;
    } else if (g == 2) {
        wt = {down_weight2, down_scale2, forget2, forget_scale2, gain2, gain_scale2};
        input_msg = input2;
        norm_weight = norm_weight2;
        normed = normed2;
    }
    float* out_g = out != nullptr ? out + size_t(g) * kHidden : nullptr;
    depth_gate::run<true, false>(blocks, nb, partial_bf16, nullptr, nullptr, nullptr, nullptr, normed_partial,
                                 queries + size_t(g) * q_stride, wt, input_msg, m + size_t(g) * kHidden,
                                 out_g, norm_weight, normed, eps, scale);
}

// Pre-MLP short_conv_residual + depth attention + message gate (BF16), grid (16, 1, 1),
// 288 threads. Leaves moe_router_step_2304 separate so its router weight loads overlap via PDL.
extern "C" __global__ void SIPHON_CLUSTER_DIMS(kGateCluster) __launch_bounds__(kGateThreads)
depth_gate_mlp_step_2304(const float* blocks, int nb, const __nv_bfloat16* partial_bf16,
                         const __nv_bfloat16* __restrict__ conv_weight, float* conv_state,
                         const __nv_bfloat16* conv_hidden, float* residual,
                         const float* __restrict__ queries,
                         const __nv_bfloat16* __restrict__ down_weight0, const __nv_bfloat16* input0,
                         const __nv_bfloat16* __restrict__ forget0,
                         const __nv_bfloat16* __restrict__ gain0, const float* m, float* out,
                         const __nv_bfloat16* __restrict__ norm_weight0, __nv_bfloat16* normed0,
                         float eps, float scale) {
    GateWeights<false> wt{down_weight0, nullptr, forget0, nullptr, gain0, nullptr};
    depth_gate::run<false, true>(blocks, nb, partial_bf16, conv_weight, conv_state, conv_hidden, residual,
                                 nullptr, queries, wt, input0, m, out, norm_weight0, normed0, eps, scale);
}

// Pre-MLP short_conv_residual + depth attention + message gate (FP8), grid (16, 1, 1),
// 288 threads. Leaves moe_router_step_2304 separate so its router weight loads overlap via PDL.
extern "C" __global__ void SIPHON_CLUSTER_DIMS(kGateCluster) __launch_bounds__(kGateThreads)
depth_gate_mlp_step_fp8_2304(const float* blocks, int nb, const __nv_bfloat16* partial_bf16,
                             const __nv_bfloat16* __restrict__ conv_weight, float* conv_state,
                             const __nv_bfloat16* conv_hidden, float* residual,
                             const float* __restrict__ queries,
                             const uint8_t* __restrict__ down_weight0,
                             const float* __restrict__ down_scale0, const __nv_bfloat16* input0,
                             const uint8_t* __restrict__ forget0,
                             const float* __restrict__ forget_scale0, const uint8_t* __restrict__ gain0,
                             const float* __restrict__ gain_scale0, const float* m, float* out,
                             const __nv_bfloat16* __restrict__ norm_weight0, __nv_bfloat16* normed0,
                             float eps, float scale) {
    GateWeights<true> wt{down_weight0, down_scale0, forget0, forget_scale0, gain0, gain_scale0};
    depth_gate::run<true, true>(blocks, nb, partial_bf16, conv_weight, conv_state, conv_hidden, residual,
                                nullptr, queries, wt, input0, m, out, norm_weight0, normed0, eps, scale);
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

// message_gate_norm_rows_2304 with fused MXFP8 / NVFP4 quantization directly from
// registers. `normed` may be null when downstream only reads the quantized rows.
extern "C" __global__ void __launch_bounds__(kGateThreads)
message_gate_norm_quant_rows_2304(const float* out, const __nv_bfloat16* __restrict__ norm_weight,
                                  __nv_bfloat16* normed, uint8_t* q_values, uint8_t* q_scales,
                                  float* q_outer, int q_mode, float eps) {
    __shared__ float vw[kNormVirtualWarps];
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const size_t row = blockIdx.x;
    const size_t at = row * kHidden;
    const int ca = 4 * t;
    const int cb = 4 * (t + kGateThreads);
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
    float qa[4];
    float qb[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        va[i] = __float2bfloat16((a[i] * inv) * __bfloat162float(wa[i]));
        vb[i] = __float2bfloat16((b[i] * inv) * __bfloat162float(wb[i]));
        qa[i] = __bfloat162float(va[i]);
        qb[i] = __bfloat162float(vb[i]);
    }
    if (normed != nullptr) {
        *reinterpret_cast<uint2*>(normed + at + ca) = na;
        *reinterpret_cast<uint2*>(normed + at + cb) = nb;
    }
    if (q_mode == quant_epilogue::kQuantMxfp8 || q_mode == quant_epilogue::kQuantMxfp8Paired) {
        const bool paired = q_mode == quant_epilogue::kQuantMxfp8Paired;
        uint8_t* qv = q_values + row * kHidden;
        uint8_t* qs = q_scales + row * (kHidden / 32);
        quant_epilogue::quantize_vec4_mxfp8(qa, t / 8, lane % 8, paired, qv, qs);
        quant_epilogue::quantize_vec4_mxfp8(qb, 36 + t / 8, lane % 8, paired, qv, qs);
    } else if (q_mode == quant_epilogue::kQuantNvfp4) {
        __shared__ float warp_maxs[kGateWarps];
        float local_max = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            local_max = fmaxf(local_max, fabsf(qa[i]));
            local_max = fmaxf(local_max, fabsf(qb[i]));
        }
        local_max = quant_epilogue::warp_max(local_max);
        if (lane == 0) {
            warp_maxs[warp] = local_max;
        }
        __syncthreads();
        float row_amax = 0.f;
#pragma unroll
        for (int w = 0; w < kGateWarps; ++w) {
            row_amax = fmaxf(row_amax, warp_maxs[w]);
        }
        if (t == 0) {
            q_outer[row] = __fmul_rn(row_amax, 1.f / 1536.f);
        }
        uint8_t* qv = q_values + row * (kHidden / 2);
        uint8_t* qs = q_scales + row * (kHidden / 16);
        quant_epilogue::quantize_vec4_nvfp4(qa, row_amax, t / 4, lane % 4, qv, qs);
        quant_epilogue::quantize_vec4_nvfp4(qb, row_amax, 72 + t / 4, lane % 4, qv, qs);
    }
}

// ---------------------------------------------------------------------------
// The whole MessageGate for the rows of a verify chunk or a batched step (1 to 32 rows),
// up to three gates per launch, one cluster of 16 blocks per gate (blockIdx.y) and group
// of 8 rows (blockIdx.z). It sums as the three
// kernels it replaces do: the down projection of rows_*_k4608 (linear_rows.cu), then
// message_gate_out_rows_2304 with a rebuilt message, then message_gate_norm_rows_2304.
// So its results equal theirs bit for bit, and each row's result depends on that row
// only.
//
// Down projection: rows_*_k4608 splits each tile of 16 down rows over 8 warps. Here
// warps 0 and 1 of block r run splits 2 (r % 4) and 2 (r % 4) + 1 of tile r / 4, with the
// same stages and mma order, and send each partial row to the block that owns it. Block
// r owns down rows 4r .. 4r + 3. Its warp 2 adds the 8 partials in split order, applies
// the FP8 row scale, rounds to BF16 and sends the 4 rows to every block. Both exchanges
// are st.async stores that complete on the receiver's mbarrier.
//
// Gate outputs: block r computes columns 144 r .. 144 r + 143, 16 per warp, with the
// tensor-core products and the combine of message_gate_out_rows_2304, into `out`. After a
// cluster barrier, block j < rows normalizes row j as message_gate_norm_rows_2304 does:
// a separate block per row, since one block normalizing every row was bound by one SM's
// L2 bandwidth.
// ---------------------------------------------------------------------------

namespace gate_rows {

constexpr int kGateSlots = 3;
constexpr int kMaxRows = 8;
// Warp splits of each down tile, as rows_*_k4608 launches with 64 output rows.
constexpr int kSplits = 8;
constexpr int kTileRows = 16;
constexpr int kTiles = kGateRank / kTileRows;
constexpr int kUnitsPerBlock = kTiles * kSplits / kGateCluster;
constexpr int kBlocksPerTile = kSplits / kUnitsPerBlock;
constexpr int kOwnedRows = kGateRank / kGateCluster;
constexpr int kCols = kHidden / kGateCluster;
constexpr int kStageBytes = 256;
constexpr int kChunks = kStageBytes / 16;
constexpr int kSlotVecs = kTileRows * kChunks;
constexpr int kK = 2 * kHidden;
// The warp that adds the owned rows' partials.
constexpr int kOwnerWarp = kUnitsPerBlock;
static_assert(kCols == 16 * kGateWarps, "two column groups of 8 per warp");
static_assert(kOwnedRows * kMaxRows == 32, "the owned partial rows fill a warp");
static_assert(kOwnerWarp < kGateWarps, "a warp left to add the partials");

template <bool kFp8>
constexpr int kStagesPerRow = kK * (kFp8 ? 1 : 2) / kStageBytes;
template <bool kFp8>
constexpr int kMaxStages = (kStagesPerRow<kFp8> + kSplits - 1) / kSplits;

// One gate's buffers.
struct Gate {
    // The down weight: BF16, or E4M3 bytes with one F32 scale per row.
    const uint8_t* down;
    const float* down_scale;
    const __nv_bfloat16* forget;
    const __nv_bfloat16* gain;
    // The down projection's input: the normalized state ([rows, 2304], rows x_state_stride
    // apart), then the normalized message (rows x_message_stride apart).
    const __nv_bfloat16* x_state;
    const __nv_bfloat16* x_message;
    // Rows h_stride apart.
    const float* h;
    // Rows probs_stride apart, nb + 1 weights each.
    const float* probs;
    const __nv_bfloat16* norm_weight;
    // [rows, 2304], or null to skip the norm.
    __nv_bfloat16* normed;
    // [rows, 2304] scratch.
    float* out;
};

struct GateRowsArgs {
    Gate g[kGateSlots];
    // The partial message, rows m_stride apart, and the nb block states, block_stride
    // apart with rows 2304 apart.
    const float* m;
    const float* blocks;
    int x_state_stride;
    int x_message_stride;
    int h_stride;
    int m_stride;
    int block_stride;
    int nb;
    int probs_stride;
    int rows;
    float eps;
};
static_assert(sizeof(Gate) == 88, "scripts/pdl_audit.py names the fields by offset");

__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t pair) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3);
    const uint32_t u = uint32_t(h.x) | (uint32_t(h.y) << 16);
    uint32_t r = ((u & 0x7fff7fffu) >> 3) | (u & 0x80008000u);
    __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&r);
    v = __hmul2(v, __floats2bfloat162_rn(0x1p112f, 0x1p112f));
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void copy16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(gate_smem_addr(dst)), "l"(src) : "memory");
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait_copies() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// linear_rows.cu's slot layout.
__device__ __forceinline__ int slot_index(int r, int c) { return r * kChunks + (c ^ ((r & 1) << 2)); }

__device__ __forceinline__ uint32_t pack_bf16(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return uint32_t(__bfloat16_as_ushort(lo)) | (uint32_t(__bfloat16_as_ushort(hi)) << 16);
}

__device__ __forceinline__ void send_v2(uint32_t local, uint32_t barrier, uint32_t rank, float x, float y) {
#if SIPHON_SM89
    gate_rows_mail.send<2>(kGateCluster, int(rank), local, barrier, {__float_as_uint(x), __float_as_uint(y)});
#else
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(
                     gate_cluster_addr(local, rank)),
                 "r"(__float_as_uint(x)), "r"(__float_as_uint(y)), "r"(gate_cluster_addr(barrier, rank))
                 : "memory");
#endif
}

__device__ __forceinline__ void send_v4(uint32_t local, uint32_t barrier, uint32_t rank, uint4 v) {
#if SIPHON_SM89
    gate_rows_mail.send<4>(kGateCluster, int(rank), local, barrier, {v.x, v.y, v.z, v.w});
#else
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(
                     gate_cluster_addr(local, rank)),
                 "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(gate_cluster_addr(barrier, rank))
                 : "memory");
#endif
}

template <bool kFp8>
constexpr size_t smem_bytes() {
    return size_t(kUnitsPerBlock) * kMaxStages<kFp8> * kSlotVecs * 16 + 2 * size_t(kCols) * kGateRank * 2;
}

template <bool kFp8>
__device__ __forceinline__ void run(const GateRowsArgs& a, uint4* ring) {
    constexpr int kBytes = kFp8 ? 1 : 2;
    constexpr int kKPerStage = kStageBytes / kBytes;
    constexpr int kS = kStagesPerRow<kFp8>;
    constexpr int kMS = kMaxStages<kFp8>;
    // Input vectors per lane per stage: one per BF16 chunk, two per FP8 chunk.
    constexpr int kXVecs = kChunks / 4 * (kFp8 ? 2 : 1);
    __shared__ __align__(16) float parts[kSplits][kOwnedRows][kMaxRows];
    // The BF16 down projection, [rank][row].
    __shared__ __align__(16) __nv_bfloat16 down_t[kGateRank][kMaxRows];
    __shared__ __align__(16) __nv_bfloat16 owned[kOwnedRows][kMaxRows];
    __shared__ float vw[kNormVirtualWarps];
    __shared__ __align__(8) uint64_t arrived[2];

    const int gate = blockIdx.y;
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int gq = lane / 4;
    const int tq = lane % 4;
    const uint32_t rank = gate_cluster_rank();
    const int total_groups = (a.rows + kMaxRows - 1) / kMaxRows;
    const uint32_t parts_barrier = gate_smem_addr(&arrived[0]);
    const uint32_t down_barrier = gate_smem_addr(&arrived[1]);
#if !SIPHON_SM89
    if (t == 0) {
        for (int i = 0; i < 2; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(gate_smem_addr(&arrived[i])) : "memory");
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        gate_expect(parts_barrier, sizeof(parts));
        gate_expect(down_barrier, sizeof(down_t));
    }
    // Every block arrives now and waits before its first remote store.
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
#endif

    // Weights first, all at once: this block's down stages, then its forget and gain rows.
    auto* fs = reinterpret_cast<__nv_bfloat16*>(ring + kUnitsPerBlock * kMS * kSlotVecs);
    __nv_bfloat16* gs = fs + kCols * kGateRank;
    const int tile = int(rank) / kBlocksPerTile;
    const int split = int(rank) % kBlocksPerTile * kUnitsPerBlock + warp;
    const int stages = warp < kUnitsPerBlock ? (kS - split + kSplits - 1) / kSplits : 0;
    uint4* buf = ring + warp * kMS * kSlotVecs;
    {
        const uint8_t* w = a.g[gate].down;
#pragma unroll
        for (int i = 0; i < kMS; ++i) {
            if (i < stages) {
                const int stage = split + i * kSplits;
#pragma unroll
                for (int q = 0; q < kSlotVecs / 32; ++q) {
                    const int idx = q * 32 + lane;
                    const int r = idx / kChunks;
                    const int c = idx % kChunks;
                    copy16(buf + i * kSlotVecs + slot_index(r, c),
                           w + size_t(tile * kTileRows + r) * kK * kBytes + stage * kStageBytes + c * 16);
                }
            }
        }
        commit();
    }
    const int col0 = int(rank) * kCols;
    {
        constexpr int kRowChunks = kGateRank / 8;
        constexpr int kPerThread = kCols * kRowChunks / kGateThreads;
        static_assert(kPerThread * kGateThreads == kCols * kRowChunks, "threads cover the rows");
        const __nv_bfloat16* forget = a.g[gate].forget;
        const __nv_bfloat16* gain = a.g[gate].gain;
#pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
            const int idx = t + i * kGateThreads;
            const int r = idx / kRowChunks;
            const int c = idx % kRowChunks;
            const size_t at = size_t(col0 + r) * kGateRank + 8 * c;
            copy16(&fs[gate_swizzle(r, c)], forget + at);
            copy16(&gs[gate_swizzle(r, c)], gain + at);
        }
        commit();
    }
    float own_scale = 1.f;
    if constexpr (kFp8) {
        if (warp == kOwnerWarp) {
            own_scale = a.g[gate].down_scale[int(rank) * kOwnedRows + lane / kMaxRows];
        }
    }
    pdl_trigger();
    // A branch keeps ptxas from placing the wait before the copies (see pdl.cuh).
    if (a.rows > 0) {
        pdl_wait();
    }
#if !SIPHON_SM89
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
    // The norm weights of row `rank`, for blocks below `kMaxRows` (reused across all groups;
    // loaded after pdl_wait so the `a.g[gate].normed != nullptr` guard stays PDL-clean).
    const __nv_bfloat16* norm_weight = a.g[gate].norm_weight;
    const bool norm = a.g[gate].normed != nullptr;
    const int ca = 4 * t;
    const int cb = 4 * (t + kGateThreads);
    uint2 wpa = make_uint2(0, 0);
    uint2 wpb = make_uint2(0, 0);
    if (norm && int(rank) < kMaxRows) {
        wpa = *reinterpret_cast<const uint2*>(norm_weight + ca);
        wpb = *reinterpret_cast<const uint2*>(norm_weight + cb);
    }

    // Persistent-cluster loop over 8-row groups so down, forget, gain and norm weights
    // stay resident in shared memory and registers across groups (4x-10.7x less weight traffic).
    for (int iter = 0, group = int(blockIdx.z); group < total_groups; ++iter, group += int(gridDim.z)) {
        const int row0 = group * kMaxRows;
        const int rows = min(a.rows - row0, kMaxRows);
        const uint32_t parity = uint32_t(iter & 1);
        if (iter > 0) {
            __syncthreads();
#if SIPHON_SM89
            sm89::barrier(gate_rows_barriers, 12, kGateCluster);
#else
            if (t == 0) {
                gate_expect(parts_barrier, sizeof(parts));
                gate_expect(down_barrier, sizeof(down_t));
            }
            asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
        }

        if (warp < kUnitsPerBlock) {
            // Every stage's inputs, then the products in rows_product's order. Stages below
            // kS / 2 cover the state, the rest the message.
            static_assert(kS % 2 == 0, "no stage spans both halves");
            const uint4* xs =
                reinterpret_cast<const uint4*>(a.g[gate].x_state + size_t(row0 + gq) * a.x_state_stride);
            const uint4* xm =
                reinterpret_cast<const uint4*>(a.g[gate].x_message + size_t(row0 + gq) * a.x_message_stride);
            const bool live = gq < rows;
            uint4 xv[kMS][kXVecs];
#pragma unroll
            for (int i = 0; i < kMS; ++i) {
                if (i < stages) {
                    const int stage = split + i * kSplits;
                    const bool message = stage >= kS / 2;
                    const uint4* x = message ? xm : xs;
                    const int k0 = (message ? stage - kS / 2 : stage) * kKPerStage;
#pragma unroll
                    for (int j = 0; j < kChunks / 4; ++j) {
                        const int c = 4 * j + tq;
#pragma unroll
                        for (int hh = 0; hh < kXVecs / (kChunks / 4); ++hh) {
                            const int xi = (k0 + c * (kFp8 ? 16 : 8)) / 8 + hh;
                            xv[i][j * (kXVecs / (kChunks / 4)) + hh] = live ? x[xi] : make_uint4(0, 0, 0, 0);
                        }
                    }
                }
            }
            if (iter == 0) {
                wait_copies<1>();
            }
            __syncwarp();
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < kMS; ++i) {
                if (i < stages) {
                    const uint4* slot = buf + i * kSlotVecs;
#pragma unroll
                    for (int j = 0; j < kChunks / 4; ++j) {
                        const int c = 4 * j + tq;
                        const uint4 w0 = slot[slot_index(gq, c)];
                        const uint4 w8 = slot[slot_index(gq + 8, c)];
                        if constexpr (kFp8) {
                            const uint4 x0 = xv[i][2 * j];
                            const uint4 x1 = xv[i][2 * j + 1];
                            const uint32_t aw[4] = {w0.x, w0.y, w0.z, w0.w};
                            const uint32_t bw[4] = {w8.x, w8.y, w8.z, w8.w};
                            const uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                const uint32_t af[4] = {e4m3x2_to_bf16x2(aw[q] & 0xffffu),
                                                        e4m3x2_to_bf16x2(bw[q] & 0xffffu), e4m3x2_to_bf16x2(aw[q] >> 16),
                                                        e4m3x2_to_bf16x2(bw[q] >> 16)};
                                gate_mma(acc, af, xw[2 * q], xw[2 * q + 1]);
                            }
                        } else {
                            const uint4 xx = xv[i][j];
                            const uint32_t a0[4] = {w0.x, w8.x, w0.y, w8.y};
                            const uint32_t a1[4] = {w0.z, w8.z, w0.w, w8.w};
                            gate_mma(acc, a0, xx.x, xx.y);
                            gate_mma(acc, a1, xx.z, xx.w);
                        }
                    }
                }
            }
            // acc[0], acc[1]: tile row gq, rows 2 tq and 2 tq + 1; acc[2], acc[3]: tile row gq + 8.
            const int r_lo = tile * kTileRows + gq;
            const int r_hi = r_lo + 8;
            send_v2(gate_smem_addr(&parts[split][r_lo % kOwnedRows][2 * tq]), parts_barrier,
                    uint32_t(r_lo / kOwnedRows), acc[0], acc[1]);
            send_v2(gate_smem_addr(&parts[split][r_hi % kOwnedRows][2 * tq]), parts_barrier,
                    uint32_t(r_hi / kOwnedRows), acc[2], acc[3]);
        } else if (warp == kOwnerWarp) {
            gate_wait<false, true>(parts_barrier, parts, sizeof(parts), parity);
            // Lane l: down row 4 rank + l / 8 of row l % 8, added over the splits in order.
            const int i = lane / kMaxRows;
            const int j = lane % kMaxRows;
            float acc = parts[0][i][j];
#pragma unroll
            for (int u = 1; u < kSplits; ++u) {
                acc += parts[u][i][j];
            }
            owned[i][j] = __float2bfloat16_rn(kFp8 ? own_scale * acc : acc);
            __syncwarp();
            // Lane l sends owned rows 2 (l % 2) and 2 (l % 2) + 1 to block l / 2.
            const uint32_t dst = uint32_t(lane / 2);
#pragma unroll
            for (int k = 0; k < 2; ++k) {
                const int r = 2 * (lane % 2) + k;
                send_v4(gate_smem_addr(&down_t[int(rank) * kOwnedRows + r][0]), down_barrier, dst,
                        *reinterpret_cast<const uint4*>(&owned[r][0]));
            }
        }
        if (iter == 0) {
            wait_copies<0>();
        }
        __syncthreads();
        gate_wait<true, true>(down_barrier, down_t, sizeof(down_t), parity);

        // The gate products: warp w takes columns 16 w .. 16 w + 15 of the block as two groups
        // of 8, A holding the down projection's rows (rows 8 .. 15 zero).
        float acc_f[2][4] = {};
        float acc_g[2][4] = {};
#pragma unroll
        for (int k = 0; k < kGateRank / 16; ++k) {
            const uint32_t af[4] = {pack_bf16(down_t[16 * k + 2 * tq][gq], down_t[16 * k + 2 * tq + 1][gq]), 0u,
                                    pack_bf16(down_t[16 * k + 8 + 2 * tq][gq], down_t[16 * k + 9 + 2 * tq][gq]), 0u};
            const int n = 16 * warp + lane % 8 + 8 * (lane / 16);
            const int chunk = 2 * k + (lane / 8) % 2;
            uint32_t bf[4];
            uint32_t bg[4];
            gate_ldmatrix_x4(bf, &fs[gate_swizzle(n, chunk)]);
            gate_ldmatrix_x4(bg, &gs[gate_swizzle(n, chunk)]);
            gate_mma(acc_f[0], af, bf[0], bf[1]);
            gate_mma(acc_f[1], af, bf[2], bf[3]);
            gate_mma(acc_g[0], af, bg[0], bg[1]);
            gate_mma(acc_g[1], af, bg[2], bg[3]);
        }
        if (gq < rows) {
            const int row = row0 + gq;
            const float* pw = a.g[gate].probs + size_t(row) * a.probs_stride;
            const int nb = a.nb;
            float wi[kGtMaxBlocks];
#pragma unroll
            for (int i = 0; i < kGtMaxBlocks; ++i) {
                wi[i] = i < nb ? pw[i] : 0.f;
            }
            const float wp = pw[nb];
            const float* h = a.g[gate].h + size_t(row) * a.h_stride;
            const float* m = a.m + size_t(row) * a.m_stride;
            float* out = a.g[gate].out + size_t(row) * kHidden;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const int d = col0 + 16 * warp + 8 * n + 2 * tq;
                const float2 hv = *reinterpret_cast<const float2*>(h + d);
                float2 mv = *reinterpret_cast<const float2*>(m + d);
                float2 b[kGtMaxBlocks];
#pragma unroll
                for (int i = 0; i < kGtMaxBlocks; ++i) {
                    if (i < nb) {
                        b[i] = *reinterpret_cast<const float2*>(a.blocks + size_t(i) * a.block_stride +
                                                                size_t(row) * kHidden + d);
                    }
                }
                float2 acc = {};
#pragma unroll
                for (int i = 0; i < kGtMaxBlocks; ++i) {
                    if (i < nb) {
                        acc.x = fmaf(wi[i], b[i].x, acc.x);
                        acc.y = fmaf(wi[i], b[i].y, acc.y);
                    }
                }
                mv.x = fmaf(wp, mv.x, acc.x);
                mv.y = fmaf(wp, mv.y, acc.y);
                const float o0 =
                    fadd(fmul(fsub(1.f, tanhf(acc_f[n][0])), hv.x), fmul(tanhf(acc_g[n][0]), mv.x));
                const float o1 =
                    fadd(fmul(fsub(1.f, tanhf(acc_f[n][1])), hv.y), fmul(tanhf(acc_g[n][1]), mv.y));
                *reinterpret_cast<float2*>(out + d) = make_float2(o0, o1);
            }
        }
        if (norm) {
            // Every block's outputs are visible to the cluster after the barrier.
#if SIPHON_SM89
            sm89::barrier(gate_rows_barriers, 12, kGateCluster);
#else
            asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
            if (int(rank) < rows) {
                // message_gate_norm_rows_2304 for row `rank` of the cluster's rows.
                const size_t at = size_t(row0 + int(rank)) * kHidden;
                const float4 av = __ldcg(reinterpret_cast<const float4*>(a.g[gate].out + at + ca));
                const float4 bv = __ldcg(reinterpret_cast<const float4*>(a.g[gate].out + at + cb));
                const float va[4] = {av.x, av.y, av.z, av.w};
                const float vb[4] = {bv.x, bv.y, bv.z, bv.w};
                const auto* wa = reinterpret_cast<const __nv_bfloat16*>(&wpa);
                const auto* wb = reinterpret_cast<const __nv_bfloat16*>(&wpb);
                float pa = 0.f;
                float pb = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    pa = fmaf(va[i], va[i], pa);
                    pb = fmaf(vb[i], vb[i], pb);
                }
                pa = warp_sum(pa);
                pb = warp_sum(pb);
                if (lane == 0) {
                    vw[warp] = pa;
                    vw[warp + kGateWarps] = pb;
                }
                __syncthreads();
                const float ss = warp_sum(lane < kNormVirtualWarps ? vw[lane] : 0.f);
                const float inv = 1.f / sqrtf(ss / static_cast<float>(kHidden) + a.eps);
                uint2 na;
                uint2 nbv;
                auto* oa = reinterpret_cast<__nv_bfloat16*>(&na);
                auto* ob = reinterpret_cast<__nv_bfloat16*>(&nbv);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    oa[i] = __float2bfloat16((va[i] * inv) * __bfloat162float(wa[i]));
                    ob[i] = __float2bfloat16((vb[i] * inv) * __bfloat162float(wb[i]));
                }
                __nv_bfloat16* normed = a.g[gate].normed;
                *reinterpret_cast<uint2*>(normed + at + ca) = na;
                *reinterpret_cast<uint2*>(normed + at + cb) = nbv;
            }
        }
    }
}

}  // namespace gate_rows

// gate_rows::run with BF16 down weights, and with E4M3 down weights. The grid is
// (16, gates, ceil(rows / 8)) blocks of 288 threads, one cluster per gate and 8 rows, with
// gate_rows::smem_bytes<fp8>() bytes of dynamic shared memory.
extern "C" __global__ void SIPHON_CLUSTER_DIMS(kGateCluster) __launch_bounds__(kGateThreads)
message_gate_rows_2304(const __grid_constant__ gate_rows::GateRowsArgs args) {
    extern __shared__ uint4 gate_ring[];
    gate_rows::run<false>(args, gate_ring);
}

extern "C" __global__ void SIPHON_CLUSTER_DIMS(kGateCluster) __launch_bounds__(kGateThreads)
message_gate_rows_fp8_2304(const __grid_constant__ gate_rows::GateRowsArgs args) {
    extern __shared__ uint4 gate_ring[];
    gate_rows::run<true>(args, gate_ring);
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
