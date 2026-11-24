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
                                                    float* state, int c, int dims) {
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
__device__ __forceinline__ __nv_bfloat16 dynamic_conv(float x, const __nv_bfloat16* dynamic,
                                                      const float* __restrict__ static_weight,
                                                      float* state, int c, int dims) {
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

// The multi-row forms. Row t of a chunk convolves rows t - 3 .. t of the chunk's
// inputs, reaching into the carried state for rows before the chunk, and leaves the
// state alone; shift_state_rows in conv.cu then shifts the chunk into the state. Each
// row rounds exactly as the single-row form above.

// The window of row t for channel c, oldest first: x [rows, dims] BF16 rows t - 3 .. t,
// with the rows before 0 taken from state [3, dims].
__device__ __forceinline__ void load_window(const __nv_bfloat16* x, const float* state, int c, int dims, int t,
                                            float (&win)[kKernel]) {
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        const int row = t - (kKernel - 1) + k;
        win[k] = row >= 0 ? __bfloat162float(x[size_t(row) * dims + c])
                          : state[size_t(kKernel - 1 + row) * dims + c];
    }
}

// short_conv over a given window, oldest row first; win[3] is the current row.
__device__ __forceinline__ __nv_bfloat16 short_conv_window(const float (&win)[kKernel],
                                                           const __nv_bfloat16* __restrict__ weight,
                                                           int c) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        acc = __fadd_rn(acc, __fmul_rn(round_bf16(win[k]), __bfloat162float(weight[size_t(c) * kKernel + k])));
    }
    const float conv = round_bf16(acc);
    return __float2bfloat16(__fdiv_rn(__fadd_rn(conv, win[kKernel - 1]), kShortDivisor));
}

// dynamic_conv over a given window, oldest row first; win[3] is the current row.
__device__ __forceinline__ __nv_bfloat16 dynamic_conv_window(const float (&win)[kKernel],
                                                             const __nv_bfloat16* dynamic,
                                                             const float* __restrict__ static_weight,
                                                             int c, int dims) {
    float acc = 0.f;
#pragma unroll
    for (int lag = 0; lag < kKernel; ++lag) {
        const float w = __fadd_rn(static_weight[lag * dims + c], __bfloat162float(dynamic[lag]));
        acc = __fadd_rn(acc, __fmul_rn(win[kKernel - 1 - lag], w));
    }
    return __float2bfloat16(__fdiv_rn(__fadd_rn(acc, win[kKernel - 1]), kDynamicDivisor));
}

}  // namespace conv_step
