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
