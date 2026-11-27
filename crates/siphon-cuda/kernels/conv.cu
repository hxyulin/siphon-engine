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
#include "packed.cuh"
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
        const float wk = weights[k];
        if (wk != 0.f) {
            total = __fadd_rn(total, __fmul_rn(__bfloat162float(experts[size_t(k) * kHidden + c]), wk));
        }
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

// Rows per block of moe_conv_residual_rows_packed_2304 (and step max rows).
constexpr int kMoeConvRows = 8;
constexpr int kMoeConvCols = 4;
constexpr int kMoeConvThreads = 192;
constexpr int kMoeConvBlockCols = kMoeConvThreads * kMoeConvCols;
// Tile of moe_conv_residual_rows_2304: 29 output rows + 3 halo rows = 32 rows (4 full
// steps of 8 half-warps). Each half-warp (16 threads x 8 BF16s = 128 cols) handles one
// row with 16-byte uint4 loads/stores, reducing halo expert reads from 37.5% to 10.3%.
constexpr int kMoeConvWideRows = 29;
constexpr int kMoeConvWideCols = 8;
constexpr int kMoeConvWideColThreads = 16;
constexpr int kMoeConvWideRowGroups = 8;
constexpr int kMoeConvWideThreads = kMoeConvWideColThreads * kMoeConvWideRowGroups;
constexpr int kMoeConvWideBlockCols = kMoeConvWideColThreads * kMoeConvWideCols;
constexpr int kMoeConvWideWindow = kMoeConvWideRows + kKernel - 1;
// Max rows of moe_conv_residual_step_rows_2304 and moe_conv_residual_batch_verify_2304.
constexpr int kStepMaxRows = 8;
static_assert(kHidden % kMoeConvBlockCols == 0, "blocks cover the row");
static_assert(kHidden % kMoeConvWideBlockCols == 0, "wide blocks cover the row");
static_assert(kMoeConvWideWindow % kMoeConvWideRowGroups == 0, "window fills whole steps");

__device__ __forceinline__ void load_bf16x4(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 packed = *reinterpret_cast<const uint2*>(p);
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = __bfloat162float(vals[i]);
    }
}

__device__ __forceinline__ void unpack_bf16x8(uint4 packed, float (&v)[8]) {
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = __bfloat162float(vals[i]);
    }
}

}  // namespace

// moe_conv_residual_2304 for the rows of a chunk: the MoE combine of each row's 8 expert
// outputs (experts [rows, 8, 2304], weights [rows, 8]), the post-MLP conv and
// out = bf16(residual + conv) ([rows, 2304]). A block takes 29 output rows and 128 columns
// (16 threads x 8 BF16s per row, 8 rows in parallel across 4 warps), walking the 32-row
// window (3 halo rows + 29 output rows) in 4 full 8-row steps through shared memory.
// Rows before the chunk come from `state`, which it only reads. The combines of the
// chunk's last 3 rows go to `combined` for shift_state_rows. Every row rounds as the
// single-row kernel does, bit for bit. Grid: (18, ceil(rows / 29)), 128 threads.
extern "C" __global__ void __launch_bounds__(kMoeConvWideThreads)
moe_conv_residual_rows_2304(const __nv_bfloat16* experts, const float* weights,
                            const __nv_bfloat16* __restrict__ weight, const float* state,
                            const float* residual, __nv_bfloat16* __restrict__ combined,
                            __nv_bfloat16* __restrict__ out, int rows) {
    __shared__ __align__(16) __nv_bfloat16 comb_s[kMoeConvWideWindow][kMoeConvWideBlockCols];
    __shared__ float taps_s[kKernel][kMoeConvWideCols][kMoeConvWideColThreads];
    const int tid = threadIdx.x;
    const int c_block = int(blockIdx.x) * kMoeConvWideBlockCols;
    const int g = tid / kMoeConvWideColThreads;
    const int lane16 = tid % kMoeConvWideColThreads;
    const int c_local = lane16 * kMoeConvWideCols;
    const int c0 = c_block + c_local;
    const int r0 = int(blockIdx.y) * kMoeConvWideRows;
    const int r_end = min(r0 + kMoeConvWideRows, rows);

    // Load the 128 channels' conv taps before pdl_wait() (weight is read-only).
    {
        float w_taps[kKernel];
        load_bf16x4(weight + size_t(c_block + tid) * kKernel, w_taps);
        const int tap_lane = tid / kMoeConvWideCols;
        const int tap_j = tid % kMoeConvWideCols;
#pragma unroll
        for (int k = 0; k < kKernel; ++k) {
            taps_s[k][tap_j][tap_lane] = w_taps[k];
        }
    }
    pdl_trigger();
    if (rows > 0) {
        pdl_wait();
    }

    constexpr int kSteps = kMoeConvWideWindow / kMoeConvWideRowGroups;
#pragma unroll 1
    for (int step = 0; step < kSteps; ++step) {
        const int base_r = r0 - (kKernel - 1) + step * kMoeConvWideRowGroups;
        if (base_r >= r_end) {
            break;
        }
        const int idx = step * kMoeConvWideRowGroups + g;
        const int r = base_r + g;
        if (r < 0) {
            const float* sp = state + size_t(kKernel - 1 + r) * kHidden + c0;
            const float4 s0 = *reinterpret_cast<const float4*>(sp);
            const float4 s1 = *reinterpret_cast<const float4*>(sp + 4);
            __nv_bfloat16 cb[kMoeConvWideCols] = {
                __float2bfloat16(s0.x), __float2bfloat16(s0.y), __float2bfloat16(s0.z), __float2bfloat16(s0.w),
                __float2bfloat16(s1.x), __float2bfloat16(s1.y), __float2bfloat16(s1.z), __float2bfloat16(s1.w),
            };
            *reinterpret_cast<uint4*>(&comb_s[idx][c_local]) = *reinterpret_cast<const uint4*>(cb);
        } else if (r < r_end) {
            const float* w_ptr = weights + size_t(r) * kTopK;
            const float4 w0 = *reinterpret_cast<const float4*>(w_ptr);
            const float4 w1 = *reinterpret_cast<const float4*>(w_ptr + 4);
            const float wv[kTopK] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            const __nv_bfloat16* e = experts + size_t(r) * kTopK * kHidden + c0;
            float total[kMoeConvWideCols] = {};
#pragma unroll
            for (int x = 0; x < kTopK; x += 2) {
                const float wx0 = wv[x];
                const float wx1 = wv[x + 1];
                const uint4 raw0 = wx0 != 0.f ? *reinterpret_cast<const uint4*>(e + size_t(x) * kHidden)
                                              : make_uint4(0u, 0u, 0u, 0u);
                const uint4 raw1 = wx1 != 0.f ? *reinterpret_cast<const uint4*>(e + size_t(x + 1) * kHidden)
                                              : make_uint4(0u, 0u, 0u, 0u);
                if (wx0 != 0.f) {
                    float v0[kMoeConvWideCols];
                    unpack_bf16x8(raw0, v0);
#pragma unroll
                    for (int j = 0; j < kMoeConvWideCols; ++j) {
                        total[j] = __fadd_rn(total[j], __fmul_rn(v0[j], wx0));
                    }
                }
                if (wx1 != 0.f) {
                    float v1[kMoeConvWideCols];
                    unpack_bf16x8(raw1, v1);
#pragma unroll
                    for (int j = 0; j < kMoeConvWideCols; ++j) {
                        total[j] = __fadd_rn(total[j], __fmul_rn(v1[j], wx1));
                    }
                }
            }
            __nv_bfloat16 cb[kMoeConvWideCols];
#pragma unroll
            for (int j = 0; j < kMoeConvWideCols; ++j) {
                cb[j] = __float2bfloat16(total[j]);
            }
            *reinterpret_cast<uint4*>(&comb_s[idx][c_local]) = *reinterpret_cast<const uint4*>(cb);
        }
        __syncthreads();
        if (idx >= kKernel - 1 && r < r_end) {
            const float* rp = residual + size_t(r) * kHidden + c0;
            const float4 r_lo = *reinterpret_cast<const float4*>(rp);
            const float4 r_hi = *reinterpret_cast<const float4*>(rp + 4);
            const float rv[kMoeConvWideCols] = {r_lo.x, r_lo.y, r_lo.z, r_lo.w, r_hi.x, r_hi.y, r_hi.z, r_hi.w};
            float win[kKernel][kMoeConvWideCols];
#pragma unroll
            for (int k = 0; k < kKernel; ++k) {
                unpack_bf16x8(*reinterpret_cast<const uint4*>(&comb_s[idx - (kKernel - 1) + k][c_local]), win[k]);
            }
            __nv_bfloat16 o[kMoeConvWideCols];
#pragma unroll
            for (int j = 0; j < kMoeConvWideCols; ++j) {
                // short_conv_window in conv_step.cuh, over rows r - 3 .. r.
                float acc = 0.f;
#pragma unroll
                for (int k = 0; k < kKernel; ++k) {
                    acc = __fadd_rn(acc, __fmul_rn(win[k][j], taps_s[k][j][lane16]));
                }
                const float y = round_bf16(__fdiv_rn(__fadd_rn(round_bf16(acc), win[kKernel - 1][j]), kShortDivisor));
                o[j] = __float2bfloat16(__fadd_rn(rv[j], y));
            }
            *reinterpret_cast<uint4*>(out + size_t(r) * kHidden + c0) = *reinterpret_cast<const uint4*>(o);
            if (r >= rows - (kKernel - 1)) {
                *reinterpret_cast<uint4*>(combined + size_t(r) * kHidden + c0) =
                    *reinterpret_cast<const uint4*>(&comb_s[idx][c_local]);
            }
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

// moe_conv_residual_rows_2304 for the 1 to 8 rows of a step. With a null `inputs` (a
// small prefill chunk), it also shifts the rows' inputs into all four conv states of the
// visit, as shift_state_rows does, from `k_raw`, `v_raw` ([rows, 1024]), `proj` and the
// combines, after every kernel that reads those states has run. With a non-null `inputs`
// (verify mode), it records the rows' conv inputs into inputs [8, 6656] and leaves all
// four conv states untouched so conv_rollback in spec.cu can shift them in place from
// their pre-verify values without a conv_backup copy. Every row rounds as
// moe_conv_residual_rows_2304 does.
//
// Thread (lane, warp r) of block b takes channel c = 32 b + lane of row r: it combines the
// row's experts, convolves the row from the combines of rows r - 3 .. r, which the block
// shares in shared memory (the old state for rows before the step), and writes the row's
// outputs and recorded inputs (or shifted states). Grid: 2304 / 32 blocks of 32 * rows
// threads.
extern "C" __global__ void __launch_bounds__(kStepBlockCols * kStepMaxRows)
moe_conv_residual_step_rows_2304(const __nv_bfloat16* experts, const float* weights,
                                 const __nv_bfloat16* __restrict__ weight, float* state,
                                 const float* residual, __nv_bfloat16* __restrict__ out, int rows,
                                 const __nv_bfloat16* proj, float* proj_state,
                                 const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                                 float* k_state, float* v_state, __nv_bfloat16* inputs) {
    constexpr int kOld = kKernel - 1;
    // comb[i] holds the combine of row i - 3, or old state row i for i < 3.
    __shared__ float comb[kStepMaxRows + kOld][kStepBlockCols];
    __shared__ float proj_old[kOld][kStepBlockCols];
    __shared__ float k_old[kOld][kStepBlockCols];
    __shared__ float v_old[kOld][kStepBlockCols];
    const int lane = threadIdx.x % 32;
    const int r = threadIdx.x / 32;
    const int c = blockIdx.x * kStepBlockCols + lane;
    const bool kv = c < kKvDims;
    const bool record_only = inputs != nullptr;
    float taps[kKernel];
    load_bf16x4(weight + size_t(c) * kKernel, taps);
    pdl_trigger();
    pdl_wait();
    float e[kTopK];
    float w[kTopK];
#pragma unroll
    for (int x = 0; x < kTopK; ++x) {
        w[x] = weights[size_t(r) * kTopK + x];
        e[x] = w[x] != 0.f ? __bfloat162float(experts[(size_t(r) * kTopK + x) * kHidden + c]) : 0.f;
    }
    const float res = residual[size_t(r) * kHidden + c];
    // Rows 0 .. 2 load the post-MLP conv state rows needed for row 0's output.
    // The other three states only ever contribute rows k >= rows (below, when !record_only).
    if (r < kOld) {
        comb[r][lane] = state[size_t(r) * kHidden + c];
    }
    float total = 0.f;
#pragma unroll
    for (int x = 0; x < kTopK; ++x) {
        if (w[x] != 0.f) {
            total = __fadd_rn(total, __fmul_rn(e[x], w[x]));
        }
    }
    const float mine = round_bf16(total);
    comb[r + kOld][lane] = mine;
    // With fewer than 3 rows, warp 0 loads the old state rows past the active warps.
    // In verify mode (`record_only`), conv states are not shifted here so proj/k/v old
    // state rows are never read.
    if (rows < kOld && r == 0) {
#pragma unroll
        for (int k = 1; k < kOld; ++k) {
            if (k >= rows) {
                comb[k][lane] = state[size_t(k) * kHidden + c];
                if (!record_only) {
                    proj_old[k][lane] = proj_state[size_t(k) * kHidden + c];
                    if (kv) {
                        k_old[k][lane] = k_state[size_t(k) * kKvDims + c];
                        v_old[k][lane] = v_state[size_t(k) * kKvDims + c];
                    }
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
    if (record_only) {
        __nv_bfloat16* row = inputs + size_t(r) * kVisitChannels;
        row[2 * kKvDims + kHidden + c] = __float2bfloat16(mine);
        row[2 * kKvDims + c] = proj[size_t(r) * kHidden + c];
        if (kv) {
            row[c] = k_raw[size_t(r) * kKvDims + c];
            row[kKvDims + c] = v_raw[size_t(r) * kKvDims + c];
        }
        return;
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

// The post-attention short conv and residual add of a batched step or batched verify,
// which depth_attention_rows_2304 in routing.cu fuses for a chunk (kPartialConvResidual):
// row r (blockIdx.y) belongs to sequence seq = r / rows_per_seq at local offset
// r % rows_per_seq, convolves its sequence's post-attention conv state (word 4 of its
// state table, [3, 2304] F32, only read) and its sequence's rows of proj [rows, 2304],
// adds `hidden` [rows, 2304] if not null, and writes the F32 sum to residual
// [rows, 2304]. Grid (2304 / 256, rows), 256 threads.
extern "C" __global__ void __launch_bounds__(256)
conv_residual_batch_2304(const __nv_bfloat16* proj, const __nv_bfloat16* __restrict__ weight,
                         const unsigned long long* states, const __nv_bfloat16* hidden, float* residual,
                         int rows_per_seq) {
    const int r = blockIdx.y;
    const int seq = r / rows_per_seq;
    const int local = r % rows_per_seq;
    const int first = seq * rows_per_seq;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    float taps[kKernel];
    load_bf16x4(weight + size_t(c) * kKernel, taps);
    pdl_trigger();
    pdl_wait();
    const float* state = batch_state(states, seq, 4);
    const size_t at = size_t(r) * kHidden + c;
    float win[kKernel];
    conv_step::load_window(proj + size_t(first) * kHidden, state, c, kHidden, local, win);
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
        w[x] = weights[size_t(r) * kTopK + x];
        e[x] = w[x] != 0.f ? __bfloat162float(experts[(size_t(r) * kTopK + x) * kHidden + c]) : 0.f;
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
        if (w[x] != 0.f) {
            total = __fadd_rn(total, __fmul_rn(e[x], w[x]));
        }
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

// Batched multi-row form of moe_conv_residual_step_rows_2304: sequence seq = blockIdx.y
// runs `rows` (1..=8) consecutive verify rows starting at row0 = seq * rows, using its own
// four conv states (words 2..5 of `states[seq]`). With a non-null `inputs`, it records its
// conv inputs into `inputs[seq, 0..rows, 6656]` and leaves all four conv states untouched
// for in-place `conv_rollback_batch`; with a null `inputs`, it shifts all four states.
// Grid: (2304 / 32, seqs), 32 * rows threads.
extern "C" __global__ void __launch_bounds__(kStepBlockCols * kStepMaxRows)
moe_conv_residual_batch_verify_2304(const __nv_bfloat16* experts, const float* weights,
                                    const __nv_bfloat16* __restrict__ weight, const float* residual,
                                    __nv_bfloat16* __restrict__ out, int rows, const __nv_bfloat16* proj,
                                    const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                                    const unsigned long long* states, __nv_bfloat16* inputs) {
    constexpr int kOld = kKernel - 1;
    __shared__ float comb[kStepMaxRows + kOld][kStepBlockCols];
    __shared__ float proj_old[kOld][kStepBlockCols];
    __shared__ float k_old[kOld][kStepBlockCols];
    __shared__ float v_old[kOld][kStepBlockCols];
    const int seq = blockIdx.y;
    const size_t row0 = size_t(seq) * size_t(rows);
    const int lane = threadIdx.x % 32;
    const int r = threadIdx.x / 32;
    const int c = blockIdx.x * kStepBlockCols + lane;
    const bool kv = c < kKvDims;
    const bool record_only = inputs != nullptr;
    float taps[kKernel];
    load_bf16x4(weight + size_t(c) * kKernel, taps);
    pdl_trigger();
    pdl_wait();
    float* const state = batch_state(states, seq, 5);
    const __nv_bfloat16* seq_experts = experts + row0 * kTopK * kHidden;
    const float* seq_weights = weights + row0 * kTopK;
    const float* seq_residual = residual + row0 * kHidden;
    __nv_bfloat16* seq_out = out + row0 * kHidden;
    const __nv_bfloat16* seq_proj = proj + row0 * kHidden;
    const __nv_bfloat16* seq_k = k_raw + row0 * kKvDims;
    const __nv_bfloat16* seq_v = v_raw + row0 * kKvDims;
    float e[kTopK];
    float w[kTopK];
#pragma unroll
    for (int x = 0; x < kTopK; ++x) {
        w[x] = seq_weights[size_t(r) * kTopK + x];
        e[x] = w[x] != 0.f ? __bfloat162float(seq_experts[(size_t(r) * kTopK + x) * kHidden + c]) : 0.f;
    }
    const float res = seq_residual[size_t(r) * kHidden + c];
    if (r < kOld) {
        comb[r][lane] = state[size_t(r) * kHidden + c];
    }
    float total = 0.f;
#pragma unroll
    for (int x = 0; x < kTopK; ++x) {
        if (w[x] != 0.f) {
            total = __fadd_rn(total, __fmul_rn(e[x], w[x]));
        }
    }
    const float mine = round_bf16(total);
    comb[r + kOld][lane] = mine;
    if (rows < kOld && r == 0) {
        const float* const proj_state = !record_only ? batch_state(states, seq, 4) : nullptr;
        const float* const k_state = (!record_only && kv) ? batch_state(states, seq, 2) : nullptr;
        const float* const v_state = (!record_only && kv) ? batch_state(states, seq, 3) : nullptr;
#pragma unroll
        for (int k = 1; k < kOld; ++k) {
            if (k >= rows) {
                comb[k][lane] = state[size_t(k) * kHidden + c];
                if (!record_only) {
                    proj_old[k][lane] = proj_state[size_t(k) * kHidden + c];
                    if (kv) {
                        k_old[k][lane] = k_state[size_t(k) * kKvDims + c];
                        v_old[k][lane] = v_state[size_t(k) * kKvDims + c];
                    }
                }
            }
        }
    }
    __syncthreads();

    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        acc = __fadd_rn(acc, __fmul_rn(round_bf16(comb[r + k][lane]), taps[k]));
    }
    const float y = round_bf16(__fdiv_rn(__fadd_rn(round_bf16(acc), mine), kShortDivisor));
    seq_out[size_t(r) * kHidden + c] = __float2bfloat16(__fadd_rn(res, y));
    if (record_only) {
        __nv_bfloat16* row = inputs + (size_t(seq) * kStepMaxRows + r) * kVisitChannels;
        row[2 * kKvDims + kHidden + c] = __float2bfloat16(mine);
        row[2 * kKvDims + c] = seq_proj[size_t(r) * kHidden + c];
        if (kv) {
            row[c] = seq_k[size_t(r) * kKvDims + c];
            row[kKvDims + c] = seq_v[size_t(r) * kKvDims + c];
        }
        return;
    }
    float* const proj_state = batch_state(states, seq, 4);
    float* const k_state = kv ? batch_state(states, seq, 2) : nullptr;
    float* const v_state = kv ? batch_state(states, seq, 3) : nullptr;
    for (int k = r; k < kOld; k += rows) {
        const int i = k + rows;
        const bool fresh = i >= kOld;
        const size_t src = size_t(i - kOld);
        state[size_t(k) * kHidden + c] = comb[i][lane];
        proj_state[size_t(k) * kHidden + c] =
            fresh ? __bfloat162float(seq_proj[src * kHidden + c]) : proj_old[min(i, kOld - 1)][lane];
        if (kv) {
            k_state[size_t(k) * kKvDims + c] =
                fresh ? __bfloat162float(seq_k[src * kKvDims + c]) : k_old[min(i, kOld - 1)][lane];
            v_state[size_t(k) * kKvDims + c] =
                fresh ? __bfloat162float(seq_v[src * kKvDims + c]) : v_old[min(i, kOld - 1)][lane];
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

// ---------------------------------------------------------------------------
// Ragged packed multi-slice forms for Model::prefill_packed. Each kernel runs
// once across all S variable-length slices using `slices[S]` and the per-visit
// state pointer table `states[S, 6]` (matching `attention_batch` / `conv_residual_batch`).
// Every row and final conv state rounds bit-for-bit identically to the per-slice kernels.
// ---------------------------------------------------------------------------

// Ragged packed form of `short_conv_residual_rows_2304`: row r = blockIdx.y finds its
// slice in `slices[0..num_slices)`, convolves strictly within that slice's rows and its
// post-attention conv state (word 4 of `states`), adds `hidden[r]` if non-null, and writes
// `residual[r]`. The conv state is only read; `shift_visit_states_packed_2304` updates it
// at the end of the visit. Grid: (2304 / 256, m), 256 threads.
extern "C" __global__ void __launch_bounds__(256)
short_conv_residual_rows_packed_2304(const __nv_bfloat16* x,
                                     const __nv_bfloat16* __restrict__ weight,
                                     const unsigned long long* states,
                                     const PackedSlice* slices,
                                     int num_slices,
                                     const __nv_bfloat16* hidden,
                                     float* __restrict__ residual) {
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    float taps[kKernel];
    load_bf16x4(weight + size_t(c) * kKernel, taps);
    pdl_trigger();
    pdl_wait();
    const int si = find_packed_slice(slices, num_slices, uint32_t(r));
    const PackedSlice sl = slices[si];
    const int first = int(sl.row_start);
    const int local = r - first;
    const float* state = batch_state(states, int(sl.state_idx), 4);
    float win[kKernel];
    conv_step::load_window(x + size_t(first) * kHidden, state, c, kHidden, local, win);
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        acc = __fadd_rn(acc, __fmul_rn(round_bf16(win[k]), taps[k]));
    }
    const float y = round_bf16(__fdiv_rn(__fadd_rn(round_bf16(acc), win[kKernel - 1]), kShortDivisor));
    const size_t at = size_t(r) * kHidden + c;
    residual[at] = hidden != nullptr ? __fadd_rn(__bfloat162float(hidden[at]), y) : y;
}

// Ragged packed form of `moe_conv_residual_rows_2304`: `tiles[blockIdx.y]` packs
// `(local_tile << 8) | slice_idx`, so each CTA processes up to `kMoeConvRows` (8) rows of
// one slice and 768 columns (4 per thread) without crossing slice boundaries. Reads that
// slice's post-MLP conv state (word 5 of `states`) and writes the combines of each slice's
// last `min(row_len, 3)` rows to `combined` for `shift_visit_states_packed_2304`.
// Grid: (3, total_tiles), 192 threads.
extern "C" __global__ void __launch_bounds__(kMoeConvThreads)
moe_conv_residual_rows_packed_2304(const __nv_bfloat16* experts, const float* weights,
                                   const __nv_bfloat16* __restrict__ weight,
                                   const unsigned long long* states,
                                   const PackedSlice* slices, const uint32_t* tiles,
                                   const float* residual, __nv_bfloat16* __restrict__ combined,
                                   __nv_bfloat16* __restrict__ out) {
    const int c0 = blockIdx.x * kMoeConvBlockCols + threadIdx.x * kMoeConvCols;
    float taps[kMoeConvCols][kKernel];
#pragma unroll
    for (int j = 0; j < kMoeConvCols; ++j) {
        load_bf16x4(weight + size_t(c0 + j) * kKernel, taps[j]);
    }
    pdl_trigger();
    pdl_wait();
    const uint32_t desc = tiles[blockIdx.y];
    const int si = int(desc & 0xffu);
    const int tile = int(desc >> 8);
    const PackedSlice sl = slices[si];
    const size_t first = size_t(sl.row_start);
    const int rows = int(sl.row_len);
    const float* state = batch_state(states, int(sl.state_idx), 5);
    const __nv_bfloat16* seq_experts = experts + first * kTopK * kHidden;
    const float* seq_weights = weights + first * kTopK;
    const float* seq_residual = residual + first * kHidden;
    __nv_bfloat16* seq_combined = combined + first * kHidden;
    __nv_bfloat16* seq_out = out + first * kHidden;

    const int r0 = tile * kMoeConvRows;
    const int r_end = min(r0 + kMoeConvRows, rows);
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
            const __nv_bfloat16* e = seq_experts + size_t(r) * kTopK * kHidden + c0;
            const float* w = seq_weights + size_t(r) * kTopK;
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
        const float4 res = *reinterpret_cast<const float4*>(seq_residual + size_t(r) * kHidden + c0);
        const float rv[kMoeConvCols] = {res.x, res.y, res.z, res.w};
        __nv_bfloat16 o[kMoeConvCols];
        __nv_bfloat16 cb[kMoeConvCols];
#pragma unroll
        for (int j = 0; j < kMoeConvCols; ++j) {
            cb[j] = __float2bfloat16(comb[i][j]);
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < kKernel; ++k) {
                acc = __fadd_rn(acc, __fmul_rn(round_bf16(comb[i - (kKernel - 1) + k][j]), taps[j][k]));
            }
            const float y = round_bf16(__fdiv_rn(__fadd_rn(round_bf16(acc), comb[i][j]), kShortDivisor));
            o[j] = __float2bfloat16(__fadd_rn(rv[j], y));
        }
        *reinterpret_cast<uint2*>(seq_out + size_t(r) * kHidden + c0) = *reinterpret_cast<const uint2*>(o);
        if (r >= rows - (kKernel - 1)) {
            *reinterpret_cast<uint2*>(seq_combined + size_t(r) * kHidden + c0) = *reinterpret_cast<const uint2*>(cb);
        }
    }
}

// Shifts up to all four conv streams (k_raw, v_raw, proj, combined) of a visit into each
// slice's conv states (words 2..5 of `states`) in one launch across all S slices. Any null
// stream pointer is skipped. Grid: (2304 / 256, num_slices), 256 threads.
extern "C" __global__ void __launch_bounds__(256)
shift_visit_states_packed_2304(const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                               const __nv_bfloat16* proj, const __nv_bfloat16* combined,
                               const unsigned long long* states, const PackedSlice* slices) {
    pdl_trigger();
    pdl_wait();
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kHidden) {
        return;
    }
    const PackedSlice sl = slices[blockIdx.y];
    const size_t first = size_t(sl.row_start);
    const int rows = int(sl.row_len);
    const int state_idx = int(sl.state_idx);
    constexpr int kOld = conv_step::kKernel - 1;
    auto shift_one = [&](const __nv_bfloat16* x, float* state, int dims) {
        float old[kOld];
#pragma unroll
        for (int k = 0; k < kOld; ++k) {
            old[k] = state[size_t(k) * dims + c];
        }
#pragma unroll
        for (int k = 0; k < kOld; ++k) {
            const int i = k + rows;
            state[size_t(k) * dims + c] =
                i < kOld ? old[i] : __bfloat162float(x[(first + size_t(i - kOld)) * dims + c]);
        }
    };
    if (c < kKvDims) {
        if (k_raw != nullptr) {
            shift_one(k_raw, batch_state(states, state_idx, 2), kKvDims);
        }
        if (v_raw != nullptr) {
            shift_one(v_raw, batch_state(states, state_idx, 3), kKvDims);
        }
    }
    if (proj != nullptr) {
        shift_one(proj, batch_state(states, state_idx, 4), kHidden);
    }
    if (combined != nullptr) {
        shift_one(combined, batch_state(states, state_idx, 5), kHidden);
    }
}

// Tree-verify post-attention short conv and residual add: row r (blockIdx.y) convolves
// its 4 tree ancestors `conv_anc[r * 4 + 0..4]` (oldest first; negative offsets index
// `state[3 + anc]`), adds `hidden[r]` if not null, and writes the F32 sum to `residual[r]`.
// Matches `depth_attention_rows`'s kPartialConvResidual rounding bit-for-bit.
// Grid: (2304 / 256, rows), 256 threads.
extern "C" __global__ void __launch_bounds__(256)
short_conv_residual_tree_2304(const __nv_bfloat16* proj, const __nv_bfloat16* __restrict__ weight,
                              const float* state, const __nv_bfloat16* hidden, float* residual,
                              const int* conv_anc) {
    const int r = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    float taps[kKernel];
    load_bf16x4(weight + size_t(c) * kKernel, taps);
    pdl_trigger();
    pdl_wait();
    float win[kKernel];
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        const int anc = conv_anc[r * kKernel + k];
        win[k] = anc >= 0 ? __bfloat162float(proj[size_t(anc) * kHidden + c])
                          : state[size_t(kKernel - 1 + anc) * kHidden + c];
    }
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        acc = __fadd_rn(acc, __fmul_rn(round_bf16(win[k]), taps[k]));
    }
    const float y = round_bf16(__fdiv_rn(__fadd_rn(round_bf16(acc), win[kKernel - 1]), kShortDivisor));
    const size_t at = size_t(r) * kHidden + c;
    residual[at] = hidden != nullptr ? __fadd_rn(__bfloat162float(hidden[at]), y) : y;
}

// Tree-verify variant of moe_conv_residual_step_rows_2304: combines each tree node's 8
// experts, convolves along `conv_anc[r * 4 + 0..4]` in shared memory (where comb[0..3]
// holds `state[0..3]` and comb[3..3+rows] holds the combines of rows 0..rows), adds
// `residual`, writes `out`, and records all 4 streams' raw conv inputs into `inputs[8, 6656]`.
// Leaves all 4 conv states untouched (`conv_commit_tree` updates them along the winning path).
// Grid: (2304 / 32, 1, 1), 32 * rows threads.
extern "C" __global__ void __launch_bounds__(kStepBlockCols * kStepMaxRows)
moe_conv_residual_tree_rows_2304(const __nv_bfloat16* experts, const float* weights,
                                 const __nv_bfloat16* __restrict__ weight, const float* state,
                                 const float* residual, __nv_bfloat16* out, int rows,
                                 const __nv_bfloat16* proj, const __nv_bfloat16* k_raw,
                                 const __nv_bfloat16* v_raw, __nv_bfloat16* inputs,
                                 const int* conv_anc) {
    constexpr int kOld = kKernel - 1;
    __shared__ float comb[kStepMaxRows + kOld][kStepBlockCols];
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
    if (r < kOld) {
        comb[r][lane] = state[size_t(r) * kHidden + c];
    }
    float total = 0.f;
#pragma unroll
    for (int x = 0; x < kTopK; ++x) {
        if (w[x] != 0.f) {
            total = __fadd_rn(total, __fmul_rn(e[x], w[x]));
        }
    }
    const float mine = round_bf16(total);
    comb[r + kOld][lane] = mine;
    if (rows < kOld && r == 0) {
#pragma unroll
        for (int k = 1; k < kOld; ++k) {
            if (k >= rows) {
                comb[k][lane] = state[size_t(k) * kHidden + c];
            }
        }
    }
    __syncthreads();

    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kKernel; ++k) {
        const int idx = conv_anc[r * kKernel + k] + kOld;
        acc = __fadd_rn(acc, __fmul_rn(round_bf16(comb[idx][lane]), taps[k]));
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
}
