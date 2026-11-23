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

#include "conv_step.cuh"
#include "pdl.cuh"

namespace {

constexpr int kHidden = 2304;
constexpr int kTopK = 8;

}  // namespace

// The post-attention conv and residual add of DecoderLayer for one row:
// residual = f32(hidden) + f32(short_conv(x)), or f32(short_conv(x)) when hidden is
// null (a block start). Equals short_conv_bf16 followed by add_bf16_bf16_f32 or
// cast_bf16_to_f32 bit for bit, except that subnormal sums are not flushed to zero.
extern "C" __global__ void short_conv_residual_2304(const __nv_bfloat16* __restrict__ x,
                                                    const __nv_bfloat16* __restrict__ weight,
                                                    float* __restrict__ state,
                                                    const __nv_bfloat16* __restrict__ hidden,
                                                    float* __restrict__ residual) {
    pdl_trigger();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kHidden) {
        return;
    }
    const float y = __bfloat162float(conv_step::short_conv(__bfloat162float(x[c]), weight, state, c, kHidden));
    residual[c] = hidden != nullptr ? __fadd_rn(__bfloat162float(hidden[c]), y) : y;
}

// The end of SparseMoeBlock and DecoderLayer for one row: the MoE combine of the 8
// expert outputs (`experts` [8, 2304] BF16, `weights` [8] F32), the post-MLP conv,
// and out = bf16(residual + conv). Equals moe_combine_bf16, short_conv_bf16 and
// add_f32_bf16_bf16 in sequence bit for bit, except that subnormals are not flushed
// to zero.
extern "C" __global__ void moe_conv_residual_2304(const __nv_bfloat16* __restrict__ experts,
                                                  const float* __restrict__ weights,
                                                  const __nv_bfloat16* __restrict__ weight,
                                                  float* __restrict__ state,
                                                  const float* __restrict__ residual,
                                                  __nv_bfloat16* __restrict__ out) {
    pdl_trigger();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kHidden) {
        return;
    }
    float total = 0.f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
        total = __fadd_rn(total, __fmul_rn(__bfloat162float(experts[size_t(k) * kHidden + c]), weights[k]));
    }
    const float x = round_bf16(total);
    const float y = __bfloat162float(conv_step::short_conv(x, weight, state, c, kHidden));
    out[c] = __float2bfloat16(__fadd_rn(residual[c], y));
}

// ---------------------------------------------------------------------------
// Multi-row forms of the two kernels above, for prefill. Each block takes one row
// (blockIdx.y) of a chunk and convolves it from the chunk's own earlier inputs and the
// carried state, which it only reads, so the rows run in parallel. shift_state_rows
// then shifts the chunk's inputs into the state. Every row rounds as the single-row
// kernels do, so a chunk gives their results bit for bit.
// ---------------------------------------------------------------------------

// short_conv_residual_2304 for rows [rows, 2304] of x, hidden and residual.
extern "C" __global__ void short_conv_residual_rows_2304(const __nv_bfloat16* __restrict__ x,
                                                         const __nv_bfloat16* __restrict__ weight,
                                                         const float* __restrict__ state,
                                                         const __nv_bfloat16* __restrict__ hidden,
                                                         float* __restrict__ residual) {
    pdl_trigger();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (c >= kHidden) {
        return;
    }
    float win[conv_step::kKernel];
    conv_step::load_window(x, state, c, kHidden, t, win);
    const float y = __bfloat162float(conv_step::short_conv_window(win, weight, c));
    const size_t i = size_t(t) * kHidden + c;
    residual[i] = hidden != nullptr ? __fadd_rn(__bfloat162float(hidden[i]), y) : y;
}

namespace {

// Rows per block of moe_conv_residual_rows_2304, and columns per thread and block.
constexpr int kMoeConvRows = 8;
constexpr int kMoeConvCols = 4;
constexpr int kMoeConvThreads = 192;
constexpr int kMoeConvBlockCols = kMoeConvThreads * kMoeConvCols;
static_assert(kHidden % kMoeConvBlockCols == 0, "blocks cover the row");

__device__ __forceinline__ void load_bf16x4(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 packed = *reinterpret_cast<const uint2*>(p);
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = __bfloat162float(vals[i]);
    }
}

}  // namespace

// moe_conv_residual_2304 for the rows of a chunk: the MoE combine of each row's 8 expert
// outputs (experts [rows, 8, 2304], weights [rows, 8]), the post-MLP conv and
// out = bf16(residual + conv) ([rows, 2304]). A block takes 16 rows and 768 columns
// (4 per thread) and walks its rows in order, recombining the 3 rows before its first
// for the conv window; rows before the chunk come from the state, which it only reads.
// The combines of the chunk's last 3 rows go to `combined` for shift_state_rows. Every
// row rounds as the single-row kernel does. Grid: (3, ceil(rows / 16)), 192 threads.
extern "C" __global__ void __launch_bounds__(kMoeConvThreads)
moe_conv_residual_rows_2304(const __nv_bfloat16* experts, const float* weights,
                            const __nv_bfloat16* __restrict__ weight, const float* state,
                            const float* residual, __nv_bfloat16* __restrict__ combined,
                            __nv_bfloat16* __restrict__ out, int rows) {
    const int c0 = blockIdx.x * kMoeConvBlockCols + threadIdx.x * kMoeConvCols;
    const int r0 = blockIdx.y * kMoeConvRows;
    const int r_end = min(r0 + kMoeConvRows, rows);
    // The conv taps of this thread's 4 channels.
    float taps[kMoeConvCols][kKernel];
#pragma unroll
    for (int j = 0; j < kMoeConvCols; ++j) {
        load_bf16x4(weight + size_t(c0 + j) * kKernel, taps[j]);
    }
    pdl_trigger();
    pdl_wait();
    // comb[i] holds the combine of row r0 - 3 + i: every load of the block's rows is
    // issued before any conv.
    constexpr int kWindowRows = kMoeConvRows + kKernel - 1;
    float comb[kWindowRows][kMoeConvCols];
#pragma unroll
    for (int i = 0; i < kWindowRows; ++i) {
        const int r = r0 - (kKernel - 1) + i;
        if (r < 0) {
            const float4 v = *reinterpret_cast<const float4*>(state + size_t(kKernel - 1 + r) * kHidden + c0);
            comb[i][0] = v.x;
            comb[i][1] = v.y;
            comb[i][2] = v.z;
            comb[i][3] = v.w;
        } else if (r < r_end) {
            const __nv_bfloat16* e = experts + size_t(r) * kTopK * kHidden + c0;
            const float* w = weights + size_t(r) * kTopK;
            float total[kMoeConvCols] = {};
#pragma unroll
            for (int x = 0; x < kTopK; ++x) {
                float v[kMoeConvCols];
                load_bf16x4(e + size_t(x) * kHidden, v);
                const float wx = w[x];
#pragma unroll
                for (int j = 0; j < kMoeConvCols; ++j) {
                    total[j] = __fadd_rn(total[j], __fmul_rn(v[j], wx));
                }
            }
#pragma unroll
            for (int j = 0; j < kMoeConvCols; ++j) {
                comb[i][j] = round_bf16(total[j]);
            }
        }
    }
#pragma unroll
    for (int i = kKernel - 1; i < kWindowRows; ++i) {
        const int r = r0 - (kKernel - 1) + i;
        if (r >= r_end) {
            break;
        }
        const float4 res = *reinterpret_cast<const float4*>(residual + size_t(r) * kHidden + c0);
        const float rv[kMoeConvCols] = {res.x, res.y, res.z, res.w};
        __nv_bfloat16 o[kMoeConvCols];
        __nv_bfloat16 cb[kMoeConvCols];
#pragma unroll
        for (int j = 0; j < kMoeConvCols; ++j) {
            cb[j] = __float2bfloat16(comb[i][j]);
            // short_conv_window in conv_step.cuh, over rows r - 3 .. r.
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < kKernel; ++k) {
                acc = __fadd_rn(acc, __fmul_rn(round_bf16(comb[i - (kKernel - 1) + k][j]), taps[j][k]));
            }
            const float y = round_bf16(__fdiv_rn(__fadd_rn(round_bf16(acc), comb[i][j]), kShortDivisor));
            o[j] = __float2bfloat16(__fadd_rn(rv[j], y));
        }
        *reinterpret_cast<uint2*>(out + size_t(r) * kHidden + c0) = *reinterpret_cast<const uint2*>(o);
        if (r >= rows - (kKernel - 1)) {
            *reinterpret_cast<uint2*>(combined + size_t(r) * kHidden + c0) = *reinterpret_cast<const uint2*>(cb);
        }
    }
}

// Shifts the conv inputs x [rows, dims] of a chunk into state [3, dims]: afterwards the
// state holds the last three rows of the old state followed by x. One thread per channel.
extern "C" __global__ void shift_state_rows(const __nv_bfloat16* x, float* state,
                                           int dims, int rows) {
    pdl_trigger();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dims) {
        return;
    }
    constexpr int kRows = conv_step::kKernel - 1;
    float old[kRows];
#pragma unroll
    for (int k = 0; k < kRows; ++k) {
        old[k] = state[size_t(k) * dims + c];
    }
#pragma unroll
    for (int k = 0; k < kRows; ++k) {
        // Index k + rows of the old state followed by x.
        const int i = k + rows;
        state[size_t(k) * dims + c] = i < kRows ? old[i] : __bfloat162float(x[size_t(i - kRows) * dims + c]);
    }
}
