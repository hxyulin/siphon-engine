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

// Channels per block of moe_conv_residual_step_rows_2304: warp r of a block takes row r of
// 32 channels. The kernel is bound by the latency of its loads, so every row's loads issue
// at once: one thread walking a channel's rows in order took 3.0 us at two rows and 8.5 us
// at eight.
constexpr int kStepBlockCols = 32;
static_assert(kHidden % kStepBlockCols == 0, "blocks cover the row");

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
// conv_rollback in spec.cu reads them. Every row rounds as moe_conv_residual_rows_2304
// does.
//
// Thread (lane, warp r) of block b takes channel c = 32 b + lane of row r: it combines the
// row's experts, convolves the row from the combines of rows r - 3 .. r, which the block
// shares in shared memory (the old state for rows before the step), and writes the row's
// outputs and recorded inputs. The block reads every old state row of its channels before
// its barrier and writes the new ones after it, so it alone touches them. Grid: 2304 / 32
// blocks of 32 rows threads.
extern "C" __global__ void __launch_bounds__(kStepBlockCols * kMoeConvRows)
moe_conv_residual_step_rows_2304(const __nv_bfloat16* experts, const float* weights,
                                 const __nv_bfloat16* __restrict__ weight, float* state,
                                 const float* residual, __nv_bfloat16* __restrict__ out, int rows,
                                 const __nv_bfloat16* proj, float* proj_state,
                                 const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                                 float* k_state, float* v_state, __nv_bfloat16* inputs) {
    constexpr int kOld = kKernel - 1;
    // comb[i] holds the combine of row i - 3, or old state row i for i < 3.
    __shared__ float comb[kMoeConvRows + kOld][kStepBlockCols];
    __shared__ float proj_old[kOld][kStepBlockCols];
    __shared__ float k_old[kOld][kStepBlockCols];
    __shared__ float v_old[kOld][kStepBlockCols];
    const int lane = threadIdx.x % 32;
    const int r = threadIdx.x / 32;
    const int c = blockIdx.x * kStepBlockCols + lane;
    const bool kv = c < kKvDims;
    float taps[kKernel];
    load_bf16x4(weight + size_t(c) * kKernel, taps);
    pdl_trigger();
    pdl_wait();
    float e[kTopK];
    float w[kTopK];
#pragma unroll
    for (int x = 0; x < kTopK; ++x) {
        e[x] = __bfloat162float(experts[(size_t(r) * kTopK + x) * kHidden + c]);
        w[x] = weights[size_t(r) * kTopK + x];
    }
    const float res = residual[size_t(r) * kHidden + c];
    // Rows 0 .. 2 load the old states' rows of their number.
    if (r < kOld) {
        comb[r][lane] = state[size_t(r) * kHidden + c];
        proj_old[r][lane] = proj_state[size_t(r) * kHidden + c];
        if (kv) {
            k_old[r][lane] = k_state[size_t(r) * kKvDims + c];
            v_old[r][lane] = v_state[size_t(r) * kKvDims + c];
        }
    }
    float total = 0.f;
#pragma unroll
    for (int x = 0; x < kTopK; ++x) {
        total = __fadd_rn(total, __fmul_rn(e[x], w[x]));
    }
    const float mine = round_bf16(total);
    comb[r + kOld][lane] = mine;
    // With fewer than 3 rows, rows 1 and 2 load the old state rows past the warps.
    if (rows < kOld && r == 0) {
#pragma unroll
        for (int k = 1; k < kOld; ++k) {
            if (k >= rows) {
                comb[k][lane] = state[size_t(k) * kHidden + c];
                proj_old[k][lane] = proj_state[size_t(k) * kHidden + c];
                if (kv) {
                    k_old[k][lane] = k_state[size_t(k) * kKvDims + c];
                    v_old[k][lane] = v_state[size_t(k) * kKvDims + c];
                }
            }
        }
    }
    __syncthreads();

    // short_conv_window in conv_step.cuh, over rows r - 3 .. r.
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        acc = __fadd_rn(acc, __fmul_rn(round_bf16(comb[r + k][lane]), taps[k]));
    }
    const float y = round_bf16(__fdiv_rn(__fadd_rn(round_bf16(acc), mine), kShortDivisor));
    out[size_t(r) * kHidden + c] = __float2bfloat16(__fadd_rn(res, y));
    if (inputs != nullptr) {
        __nv_bfloat16* row = inputs + size_t(r) * kVisitChannels;
        row[2 * kKvDims + kHidden + c] = __float2bfloat16(mine);
        row[2 * kKvDims + c] = proj[size_t(r) * kHidden + c];
        if (kv) {
            row[c] = k_raw[size_t(r) * kKvDims + c];
            row[kKvDims + c] = v_raw[size_t(r) * kKvDims + c];
        }
    }
    // New state row k is window row k + rows: an old row, or the input of row k + rows - 3.
    // Warp r writes rows k = r, r + rows, ... below 3.
    for (int k = r; k < kOld; k += rows) {
        const int i = k + rows;
        const bool fresh = i >= kOld;
        const size_t src = size_t(i - kOld);
        state[size_t(k) * kHidden + c] = comb[i][lane];
        proj_state[size_t(k) * kHidden + c] =
            fresh ? __bfloat162float(proj[src * kHidden + c]) : proj_old[min(i, kOld - 1)][lane];
        if (kv) {
            k_state[size_t(k) * kKvDims + c] =
                fresh ? __bfloat162float(k_raw[src * kKvDims + c]) : k_old[min(i, kOld - 1)][lane];
            v_state[size_t(k) * kKvDims + c] =
                fresh ? __bfloat162float(v_raw[src * kKvDims + c]) : v_old[min(i, kOld - 1)][lane];
        }
    }
}

namespace {

// Words per row of a batched step's state table: the device addresses of the row's key
// cache, value cache, and key, value, post-attention and post-MLP conv states.
constexpr int kBatchStateWords = 6;

__device__ __forceinline__ float* batch_state(const unsigned long long* states, int row, int word) {
    return reinterpret_cast<float*>(states[size_t(row) * kBatchStateWords + word]);
}

}  // namespace

// The post-attention short conv and residual add of a batched step, which
// depth_attention_rows_2304 in routing.cu fuses for a chunk (kPartialConvResidual): row r
// (blockIdx.y) convolves its own post-attention conv state (word 4 of its state table,
// [3, 2304] F32, only read) and its attention output proj [rows, 2304], adds `hidden`
// [rows, 2304] if not null, and writes the F32 sum to residual [rows, 2304]. Every value
// rounds as in the fused kernel for a chunk of one row. Grid (2304 / 256, rows), 256
// threads.
extern "C" __global__ void __launch_bounds__(256)
conv_residual_batch_2304(const __nv_bfloat16* proj, const __nv_bfloat16* __restrict__ weight,
                         const unsigned long long* states, const __nv_bfloat16* hidden, float* residual) {
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    float taps[kKernel];
    load_bf16x4(weight + size_t(c) * kKernel, taps);
    pdl_trigger();
    pdl_wait();
    const float* state = batch_state(states, r, 4);
    const size_t at = size_t(r) * kHidden + c;
    float win[kKernel];
#pragma unroll
    for (int k = 0; k < kKernel - 1; ++k) {
        win[k] = state[size_t(k) * kHidden + c];
    }
    win[kKernel - 1] = __bfloat162float(proj[at]);
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        acc = __fadd_rn(acc, __fmul_rn(round_bf16(win[k]), taps[k]));
    }
    const float y = round_bf16(__fdiv_rn(__fadd_rn(round_bf16(acc), win[kKernel - 1]), kShortDivisor));
    residual[at] = hidden != nullptr ? __fadd_rn(__bfloat162float(hidden[at]), y) : y;
}

// moe_conv_residual_step_rows_2304 for the rows of a batched step, each from its own
// sequence: row r (blockIdx.y) combines its experts, convolves its own post-MLP conv state
// (word 5 of its state table) and the combine, adds the residual, and shifts its four conv
// inputs (k_raw, v_raw [rows, 1024], proj [rows, 2304] and the combine) into its own four
// states (words 2 to 5). Every value rounds as in moe_conv_residual_step_rows_2304 for a
// step of one row. Grid (2304 / 256, rows), 256 threads.
extern "C" __global__ void __launch_bounds__(256)
moe_conv_residual_batch_2304(const __nv_bfloat16* experts, const float* weights,
                             const __nv_bfloat16* __restrict__ weight, const float* residual,
                             __nv_bfloat16* __restrict__ out, const __nv_bfloat16* proj, const __nv_bfloat16* k_raw,
                             const __nv_bfloat16* v_raw, const unsigned long long* states) {
    constexpr int kOld = kKernel - 1;
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const bool kv = c < kKvDims;
    float taps[kKernel];
    load_bf16x4(weight + size_t(c) * kKernel, taps);
    pdl_trigger();
    pdl_wait();
    float e[kTopK];
    float w[kTopK];
#pragma unroll
    for (int x = 0; x < kTopK; ++x) {
        e[x] = __bfloat162float(experts[(size_t(r) * kTopK + x) * kHidden + c]);
        w[x] = weights[size_t(r) * kTopK + x];
    }
    const size_t at = size_t(r) * kHidden + c;
    const float res = residual[at];
    float* const state = batch_state(states, r, 5);
    float* const proj_state = batch_state(states, r, 4);
    float* const k_state = batch_state(states, r, 2);
    float* const v_state = batch_state(states, r, 3);
    float comb[kKernel];
    float proj_old[kOld];
    float k_old[kOld];
    float v_old[kOld];
#pragma unroll
    for (int k = 0; k < kOld; ++k) {
        comb[k] = state[size_t(k) * kHidden + c];
        proj_old[k] = proj_state[size_t(k) * kHidden + c];
        if (kv) {
            k_old[k] = k_state[size_t(k) * kKvDims + c];
            v_old[k] = v_state[size_t(k) * kKvDims + c];
        }
    }
    float total = 0.f;
#pragma unroll
    for (int x = 0; x < kTopK; ++x) {
        total = __fadd_rn(total, __fmul_rn(e[x], w[x]));
    }
    const float mine = round_bf16(total);
    comb[kOld] = mine;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        acc = __fadd_rn(acc, __fmul_rn(round_bf16(comb[k]), taps[k]));
    }
    const float y = round_bf16(__fdiv_rn(__fadd_rn(round_bf16(acc), mine), kShortDivisor));
    out[at] = __float2bfloat16(__fadd_rn(res, y));
    // Each state drops its oldest row and takes the row's input.
#pragma unroll
    for (int k = 0; k < kOld; ++k) {
        state[size_t(k) * kHidden + c] = comb[k + 1];
        proj_state[size_t(k) * kHidden + c] =
            k + 1 < kOld ? proj_old[k + 1] : __bfloat162float(proj[at]);
        if (kv) {
            const size_t kv_at = size_t(r) * kKvDims + c;
            k_state[size_t(k) * kKvDims + c] = k + 1 < kOld ? k_old[k + 1] : __bfloat162float(k_raw[kv_at]);
            v_state[size_t(k) * kKvDims + c] = k + 1 < kOld ? v_old[k + 1] : __bfloat162float(v_raw[kv_at]);
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
