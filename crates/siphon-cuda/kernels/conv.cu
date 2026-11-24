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
extern "C" __global__ void short_conv_residual_2304(const __nv_bfloat16* x,
                                                    const __nv_bfloat16* __restrict__ weight,
                                                    float* state, const __nv_bfloat16* hidden,
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
extern "C" __global__ void moe_conv_residual_2304(const __nv_bfloat16* experts, const float* weights,
                                                  const __nv_bfloat16* __restrict__ weight,
                                                  float* state, const float* residual,
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
extern "C" __global__ void short_conv_residual_rows_2304(const __nv_bfloat16* x,
                                                         const __nv_bfloat16* __restrict__ weight,
                                                         const float* state, const __nv_bfloat16* hidden,
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

namespace {

// Channels per thread and threads per block of moe_conv_residual_step_rows_2304. The
// kernel is bound by the latency of its loads, so it runs one channel per thread over more
// warps: 4 channels per thread in 3 blocks took 4.4 us at one row and 7.5 us at four.
constexpr int kStepCols = 1;
constexpr int kStepThreads = 128;
constexpr int kStepBlockCols = kStepCols * kStepThreads;
static_assert(kHidden % kStepBlockCols == 0, "blocks cover the row");

__device__ __forceinline__ void load_bf16s(const __nv_bfloat16* p, float (&v)[kStepCols]) {
#pragma unroll
    for (int j = 0; j < kStepCols; ++j) {
        v[j] = __bfloat162float(p[j]);
    }
}

// The last 3 rows of `old` followed by the `rows` values of `x` (stride `dims`) into
// `state`, for the thread's channels from c: new row k is window row k + rows.
__device__ __forceinline__ void shift_step(float* state, const float (&old)[kKernel - 1][kStepCols],
                                           const __nv_bfloat16* x, int dims, int c, int rows) {
#pragma unroll
    for (int k = 0; k < kKernel - 1; ++k) {
        const int i = k + rows;
        float v[kStepCols];
        if (i < kKernel - 1) {
#pragma unroll
            for (int j = 0; j < kStepCols; ++j) {
                v[j] = old[i][j];
            }
        } else {
            load_bf16s(x + size_t(i - (kKernel - 1)) * dims + c, v);
        }
#pragma unroll
        for (int j = 0; j < kStepCols; ++j) {
            state[size_t(k) * dims + c + j] = v[j];
        }
    }
}

__device__ __forceinline__ void load_state_step(const float* state, int dims, int c,
                                                float (&old)[kKernel - 1][kStepCols]) {
#pragma unroll
    for (int k = 0; k < kKernel - 1; ++k) {
#pragma unroll
        for (int j = 0; j < kStepCols; ++j) {
            old[k][j] = state[size_t(k) * dims + c + j];
        }
    }
}

__device__ __forceinline__ void copy_step(__nv_bfloat16* dst, const __nv_bfloat16* src) {
#pragma unroll
    for (int j = 0; j < kStepCols; ++j) {
        dst[j] = src[j];
    }
}

constexpr int kKvDims = 1024;
// One row of a verify's conv inputs (see spec.cu): keys, values, post-attention and
// post-MLP inputs.
constexpr int kVisitChannels = 2 * kKvDims + 2 * kHidden;

}  // namespace

// moe_conv_residual_rows_2304 for the 1 to 8 rows of a step, which also does the work of
// the kernels after it in a visit of a step: it shifts the rows' inputs into all four conv
// states of the visit, as shift_state_rows does, from `k_raw`, `v_raw` ([rows, 1024]),
// `proj` and the combines, after every kernel that reads those states has run. With a
// non-null `inputs`, it also records the rows' conv inputs into inputs [8, 6656] as
// conv_rollback in spec.cu reads them. Each thread owns kStepCols channels of every stream,
// so it alone reads and writes them. Every row rounds as moe_conv_residual_rows_2304 does.
// Grid: 2304 / kStepBlockCols blocks of kStepThreads threads.
extern "C" __global__ void __launch_bounds__(kStepThreads)
moe_conv_residual_step_rows_2304(const __nv_bfloat16* experts, const float* weights,
                                 const __nv_bfloat16* __restrict__ weight, float* state,
                                 const float* residual, __nv_bfloat16* __restrict__ out, int rows,
                                 const __nv_bfloat16* proj, float* proj_state,
                                 const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                                 float* k_state, float* v_state, __nv_bfloat16* inputs) {
    const int c0 = blockIdx.x * kStepBlockCols + threadIdx.x * kStepCols;
    const bool kv = c0 < kKvDims;
    float taps[kStepCols][kKernel];
#pragma unroll
    for (int j = 0; j < kStepCols; ++j) {
        load_bf16x4(weight + size_t(c0 + j) * kKernel, taps[j]);
    }
    pdl_trigger();
    pdl_wait();
    // comb[i] holds the combine of row i - 3, or state row i for i < 3.
    constexpr int kWindowRows = kMoeConvRows + kKernel - 1;
    float comb[kWindowRows][kStepCols];
#pragma unroll
    for (int i = 0; i < kWindowRows; ++i) {
        const int r = i - (kKernel - 1);
        if (r < 0) {
#pragma unroll
            for (int j = 0; j < kStepCols; ++j) {
                comb[i][j] = state[size_t(i) * kHidden + c0 + j];
            }
        } else if (r < rows) {
            const __nv_bfloat16* e = experts + size_t(r) * kTopK * kHidden + c0;
            const float* w = weights + size_t(r) * kTopK;
            float total[kStepCols] = {};
#pragma unroll
            for (int x = 0; x < kTopK; ++x) {
                float v[kStepCols];
                load_bf16s(e + size_t(x) * kHidden, v);
                const float wx = w[x];
#pragma unroll
                for (int j = 0; j < kStepCols; ++j) {
                    total[j] = __fadd_rn(total[j], __fmul_rn(v[j], wx));
                }
            }
#pragma unroll
            for (int j = 0; j < kStepCols; ++j) {
                comb[i][j] = round_bf16(total[j]);
            }
        }
    }
    float proj_old[kKernel - 1][kStepCols];
    load_state_step(proj_state, kHidden, c0, proj_old);
    float k_old[kKernel - 1][kStepCols];
    float v_old[kKernel - 1][kStepCols];
    if (kv) {
        load_state_step(k_state, kKvDims, c0, k_old);
        load_state_step(v_state, kKvDims, c0, v_old);
    }
#pragma unroll
    for (int i = kKernel - 1; i < kWindowRows; ++i) {
        const int r = i - (kKernel - 1);
        if (r >= rows) {
            break;
        }
        __nv_bfloat16 o[kStepCols];
        __nv_bfloat16 cb[kStepCols];
#pragma unroll
        for (int j = 0; j < kStepCols; ++j) {
            cb[j] = __float2bfloat16(comb[i][j]);
            // short_conv_window in conv_step.cuh, over rows r - 3 .. r.
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < kKernel; ++k) {
                acc = __fadd_rn(acc, __fmul_rn(round_bf16(comb[i - (kKernel - 1) + k][j]), taps[j][k]));
            }
            const float y = round_bf16(__fdiv_rn(__fadd_rn(round_bf16(acc), comb[i][j]), kShortDivisor));
            o[j] = __float2bfloat16(__fadd_rn(residual[size_t(r) * kHidden + c0 + j], y));
        }
        copy_step(out + size_t(r) * kHidden + c0, o);
        // The last 3 rows' combines become the post-MLP state, row k = r + 3 - rows.
        if (r + kKernel - 1 >= rows) {
#pragma unroll
            for (int j = 0; j < kStepCols; ++j) {
                state[size_t(r + kKernel - 1 - rows) * kHidden + c0 + j] = comb[i][j];
            }
        }
        if (inputs != nullptr) {
            __nv_bfloat16* row = inputs + size_t(r) * kVisitChannels;
            copy_step(row + 2 * kKvDims + kHidden + c0, cb);
            copy_step(row + 2 * kKvDims + c0, proj + size_t(r) * kHidden + c0);
            if (kv) {
                copy_step(row + c0, k_raw + size_t(r) * kKvDims + c0);
                copy_step(row + kKvDims + c0, v_raw + size_t(r) * kKvDims + c0);
            }
        }
    }
    // With fewer than 3 rows, old state row k + rows moves to row k. Selects, not an
    // indexed register array, which nvcc turns into a jump table.
#pragma unroll
    for (int k = 0; k < kKernel - 2; ++k) {
        if (k + rows < kKernel - 1) {
            const bool second = k + rows == 2;
#pragma unroll
            for (int j = 0; j < kStepCols; ++j) {
                state[size_t(k) * kHidden + c0 + j] = second ? comb[2][j] : comb[1][j];
            }
        }
    }
    shift_step(proj_state, proj_old, proj, kHidden, c0, rows);
    if (kv) {
        shift_step(k_state, k_old, k_raw, kKvDims, c0, rows);
        shift_step(v_state, v_old, v_raw, kKvDims, c0, rows);
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
