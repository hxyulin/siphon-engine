// Causal short convolutions over one sequence, with a conv state that carries the
// last K - 1 = 3 window rows from one call to the next.
//
// Rounding matches MLX-LM's ShortConvolution and DynamicShortConvolution as
// measured on MLX 0.32.4:
// - MLX's depthwise conv1d on BF16 accumulates in F32, adding taps in index
//   order (oldest row first), and rounds to BF16 once. A product of two BF16
//   values is exact in F32, so whether the add is fused does not matter.
// - mx.sum over the 4 lags adds them in index order (lag 0 first), starting
//   from 0. The F32 products are rounded before the sum, so this file uses
//   __fmul_rn and __fadd_rn to stop nvcc from fusing them into FMAs.
// - MLX's F32 divide is correctly rounded, so the residual scaling is a true
//   divide, not a multiply by the reciprocal.
//
// One thread owns one channel and walks the rows in order, keeping the window
// in registers. That lets the state be updated in place: each thread reads its
// own state column before overwriting it.

#include <cuda_bf16.h>

namespace {

constexpr int kKernel = 4;

// The residual divisors are Python doubles that MLX turns into F32 scalars.
constexpr float kShortDivisor = static_cast<float>(2.23606797749979);  // sqrt(K + 1)
constexpr float kDynamicDivisor = 3.0f;                                // sqrt(2K + 1)

constexpr int kDynamicDims = 1024;
constexpr int kDynamicHeads = 8;
constexpr int kHeadDim = kDynamicDims / kDynamicHeads;

__device__ float round_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

}  // namespace

// ShortConvolution: x [rows, dims] BF16, weight [dims, 4] BF16 (the checkpoint's
// [dims, 4, 1] conv1d weight, tap 3 on the newest row), state [3, dims] F32.
extern "C" __global__ void short_conv_bf16(const __nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* __restrict__ weight,
                                           float* __restrict__ state,
                                           __nv_bfloat16* __restrict__ out, int dims, int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dims) {
        return;
    }
    float w[kKernel];
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        w[k] = __bfloat162float(weight[size_t(c) * kKernel + k]);
    }
    // win[0] is the oldest row. The window stays in F32, as MLX builds it.
    float win[kKernel];
#pragma unroll
    for (int k = 0; k < kKernel - 1; ++k) {
        win[k] = state[size_t(k) * dims + c];
    }
    for (int t = 0; t < rows; ++t) {
        const size_t i = size_t(t) * dims + c;
        win[kKernel - 1] = __bfloat162float(x[i]);
        // MLX casts the window to the BF16 weight dtype before the conv.
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < kKernel; ++k) {
            acc = __fadd_rn(acc, __fmul_rn(round_bf16(win[k]), w[k]));
        }
        const float conv = round_bf16(acc);
        out[i] = __float2bfloat16(__fdiv_rn(__fadd_rn(conv, win[kKernel - 1]), kShortDivisor));
#pragma unroll
        for (int k = 0; k < kKernel - 1; ++k) {
            win[k] = win[k + 1];
        }
    }
#pragma unroll
    for (int k = 0; k < kKernel - 1; ++k) {
        state[size_t(k) * dims + c] = win[k];
    }
}

// DynamicShortConvolution: x [rows, 1024] BF16, dynamic [rows, 8 heads * 4 lags]
// BF16 (the weight_proj output), static_weight [4 lags, 1024] F32,
// state [3, 1024] F32.
extern "C" __global__ void dynamic_short_conv_bf16(const __nv_bfloat16* __restrict__ x,
                                                   const __nv_bfloat16* __restrict__ dynamic,
                                                   const float* __restrict__ static_weight,
                                                   float* __restrict__ state,
                                                   __nv_bfloat16* __restrict__ out, int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kDynamicDims) {
        return;
    }
    const int head = c / kHeadDim;
    float static_w[kKernel];
#pragma unroll
    for (int lag = 0; lag < kKernel; ++lag) {
        static_w[lag] = static_weight[lag * kDynamicDims + c];
    }
    // win[lag] is the row `lag` steps back from the current one.
    float win[kKernel];
#pragma unroll
    for (int lag = 1; lag < kKernel; ++lag) {
        win[lag] = state[size_t(kKernel - 1 - lag) * kDynamicDims + c];
    }
    for (int t = 0; t < rows; ++t) {
        const size_t i = size_t(t) * kDynamicDims + c;
        win[0] = __bfloat162float(x[i]);
        const __nv_bfloat16* dyn = dynamic + size_t(t) * kDynamicHeads * kKernel + head * kKernel;
        float acc = 0.f;
#pragma unroll
        for (int lag = 0; lag < kKernel; ++lag) {
            const float w = __fadd_rn(static_w[lag], __bfloat162float(dyn[lag]));
            acc = __fadd_rn(acc, __fmul_rn(win[lag], w));
        }
        out[i] = __float2bfloat16(__fdiv_rn(__fadd_rn(acc, win[0]), kDynamicDivisor));
#pragma unroll
        for (int lag = kKernel - 1; lag > 0; --lag) {
            win[lag] = win[lag - 1];
        }
    }
#pragma unroll
    for (int lag = 1; lag < kKernel; ++lag) {
        state[size_t(kKernel - 1 - lag) * kDynamicDims + c] = win[lag];
    }
}
