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
#include <cfloat>
#include <climits>
#include <cstdint>

#include "pdl.cuh"

namespace {

constexpr int kArgmaxThreads = 1024;

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
message_gate_out_row(const __nv_bfloat16* __restrict__ down, const __nv_bfloat16* __restrict__ forget,
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
    const int ca = 4 * threadIdx.x;
    const int cb = 4 * (threadIdx.x + kGateThreads);
    float a[4];
    float b[4];
    float wa[4];
    float wb[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a[i] = __ldcg(out + ca + i);
        b[i] = __ldcg(out + cb + i);
        wa[i] = __bfloat162float(norm_weight[ca + i]);
        wb[i] = __bfloat162float(norm_weight[cb + i]);
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

extern "C" __global__ void __launch_bounds__(kGateThreads)
message_gate_out_2304(const __nv_bfloat16* __restrict__ down, const __nv_bfloat16* __restrict__ forget,
                      const __nv_bfloat16* __restrict__ gain, const float* h, const float* m, float* out,
                      const __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* normed,
                      unsigned* counter, float eps) {
    message_gate_out_row(down, forget, gain, h, m, out, norm_weight, normed, counter, eps);
}

// message_gate_out_2304 for each row (blockIdx.y) of a chunk: down [rows, 64], out and
// normed [rows, 2304], one counter per row holding 0, and the h and m rows h_stride and
// m_stride floats apart.
extern "C" __global__ void __launch_bounds__(kGateThreads)
message_gate_out_rows_2304(const __nv_bfloat16* __restrict__ down, const __nv_bfloat16* __restrict__ forget,
                           const __nv_bfloat16* __restrict__ gain, const float* h, int h_stride, const float* m,
                           int m_stride, float* out, const __nv_bfloat16* __restrict__ norm_weight,
                           __nv_bfloat16* normed, unsigned* counters, float eps) {
    const size_t row = blockIdx.y;
    message_gate_out_row(down + row * kGateRank, forget, gain, h + row * size_t(h_stride), m + row * size_t(m_stride),
                         out + row * kHidden, norm_weight, normed != nullptr ? normed + row * kHidden : nullptr,
                         counters + row, eps);
}
