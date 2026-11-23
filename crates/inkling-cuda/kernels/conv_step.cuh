// One decode row of MLX-LM's ShortConvolution and DynamicShortConvolution for one
// channel, shared by the fused kernels in conv.cu and norm.cu. Rounding follows the
// multi-row kernels in conv.cu, which document it.

#pragma once

#include <cuda_bf16.h>

namespace conv_step {

constexpr int kKernel = 4;
constexpr float kShortDivisor = static_cast<float>(2.23606797749979);  // sqrt(K + 1)
constexpr float kDynamicDivisor = 3.0f;                                // sqrt(2K + 1)

__device__ __forceinline__ float round_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// ShortConvolution: weight [dims, 4] BF16 (tap 3 on the newest row), state [3, dims] F32
// with row 0 the oldest. Shifts x into the state and returns the BF16 output.
__device__ __forceinline__ __nv_bfloat16 short_conv(float x, const __nv_bfloat16* __restrict__ weight,
                                                    float* __restrict__ state, int c, int dims) {
    float win[kKernel];
#pragma unroll
    for (int k = 0; k < kKernel - 1; ++k) {
        win[k] = state[size_t(k) * dims + c];
    }
    win[kKernel - 1] = x;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        acc = __fadd_rn(acc, __fmul_rn(round_bf16(win[k]), __bfloat162float(weight[size_t(c) * kKernel + k])));
    }
    const float conv = round_bf16(acc);
#pragma unroll
    for (int k = 0; k < kKernel - 1; ++k) {
        state[size_t(k) * dims + c] = win[k + 1];
    }
    return __float2bfloat16(__fdiv_rn(__fadd_rn(conv, x), kShortDivisor));
}

// DynamicShortConvolution: dynamic [4] BF16 (this head's taps for the row), static_weight
// [4 lags, dims] F32, state [3, dims] F32 with row 2 one step back.
__device__ __forceinline__ __nv_bfloat16 dynamic_conv(float x, const __nv_bfloat16* __restrict__ dynamic,
                                                      const float* __restrict__ static_weight,
                                                      float* __restrict__ state, int c, int dims) {
    float win[kKernel];
    win[0] = x;
#pragma unroll
    for (int lag = 1; lag < kKernel; ++lag) {
        win[lag] = state[size_t(kKernel - 1 - lag) * dims + c];
    }
    float acc = 0.f;
#pragma unroll
    for (int lag = 0; lag < kKernel; ++lag) {
        const float w = __fadd_rn(static_weight[lag * dims + c], __bfloat162float(dynamic[lag]));
        acc = __fadd_rn(acc, __fmul_rn(win[lag], w));
    }
#pragma unroll
    for (int lag = 1; lag < kKernel; ++lag) {
        state[size_t(kKernel - 1 - lag) * dims + c] = win[lag - 1];
    }
    return __float2bfloat16(__fdiv_rn(__fadd_rn(acc, x), kDynamicDivisor));
}

}  // namespace conv_step
