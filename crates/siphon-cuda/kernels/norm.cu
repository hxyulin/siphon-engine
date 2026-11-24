// RMS normalization and L2 clipping, bit-exact with MLX 0.32.4 on Apple GPUs.
//
// Float sums depend on the order of additions, so each kernel copies the
// order of the MLX Metal kernel it replaces:
//
// - `mx.fast.rms_norm` (kernels/rms_norm.metal, `rms_single_row`): each thread
//   owns 4 consecutive elements and accumulates their squares with FMA. Each
//   simdgroup reduces with `simd_sum`, which on Apple GPUs is a butterfly over
//   lane offsets 1, 2, 4, 8, 16. The per-simdgroup sums are then reduced by
//   another `simd_sum` with zeros in the unused lanes. `metal::precise::rsqrt`
//   is correctly rounded, which `rsqrtf` and `1 / sqrtf` are not.
// - The L2 clip's `sum(axis=-1)` (kernels/reduction/reduce_row.h): for rows of
//   128 each lane adds its 4 rounded squares in order starting from zero, then
//   `simd_sum`. Rows of 16 are summed sequentially by one thread. The squares
//   are separate rounded products, not FMAs.
//
// nvcc contracts `a * b + c` into an FMA by default, so the L2 clip uses
// `__fmul_rn` and `__fadd_rn`, which it never contracts.

#include <cuda_bf16.h>
#include <cstdint>

#include "pdl.cuh"

namespace {

constexpr int kHidden = 2304;
constexpr int kHeadDim = 128;
// MLX's RMS_N_READS and REDUCE_N_READS.
constexpr int kVec = 4;
constexpr int kRowThreads = kHidden / kVec;
constexpr int kRowWarps = kRowThreads / 32;
static_assert(kRowThreads % 32 == 0, "row must split into whole warps");
static_assert(kHeadDim == 32 * kVec, "one warp covers one head");

// Matches Metal's `simd_sum`. Every lane ends with the same value because each
// pair adds the same two operands.
__device__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ float fma_sum_squares(const float (&v)[kVec]) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        acc = fmaf(v[i], v[i], acc);
    }
    return acc;
}

// Sum of squares of one `kHidden` row, returned to every thread of the block.
__device__ float row_sum_squares(const float (&v)[kVec]) {
    __shared__ float warp_sums[kRowWarps];
    const float acc = warp_sum(fma_sum_squares(v));
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        warp_sums[threadIdx.x / 32] = acc;
    }
    __syncthreads();
    // Every warp repeats warp 0's final reduction, so no second barrier is needed
    // to broadcast it.
    return warp_sum(lane < kRowWarps ? warp_sums[lane] : 0.f);
}

__device__ float inv_rms(float sum_squares, int n, float eps) {
    const float mean_eps = sum_squares / static_cast<float>(n) + eps;
    // IEEE double sqrt and divide, then one rounding to float, equal
    // `metal::precise::rsqrt` for every normal float (checked exhaustively on
    // the Mac against both exponent parities).
    return static_cast<float>(1.0 / sqrt(static_cast<double>(mean_eps)));
}

__device__ void load_bf16x4(const __nv_bfloat16* p, float (&v)[kVec]) {
    const uint2 packed = *reinterpret_cast<const uint2*>(p);
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        v[i] = __bfloat162float(vals[i]);
    }
}

__device__ void load_f32x4(const float* p, float (&v)[kVec]) {
    const float4 packed = *reinterpret_cast<const float4*>(p);
    v[0] = packed.x;
    v[1] = packed.y;
    v[2] = packed.z;
    v[3] = packed.w;
}

__device__ void store_bf16x4(__nv_bfloat16* p, const float (&v)[kVec]) {
    uint2 packed;
    auto* vals = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        vals[i] = __float2bfloat16(v[i]);
    }
    *reinterpret_cast<uint2*>(p) = packed;
}

}  // namespace

// Weighted RMSNorm of BF16 rows. MLX rounds the normalized value to BF16
// before the weight multiply because the Metal kernel's T is bfloat.
extern "C" __global__ void __launch_bounds__(kRowThreads)
rms_norm_bf16_2304(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out, float eps) {
    const size_t offset = size_t(blockIdx.x) * kHidden + threadIdx.x * kVec;
    float v[kVec];
    load_bf16x4(x + offset, v);
    const float scale = inv_rms(row_sum_squares(v), kHidden, eps);
    float w[kVec];
    load_bf16x4(weight + threadIdx.x * kVec, w);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        v[i] = __bfloat162float(__float2bfloat16(v[i] * scale)) * w[i];
    }
    store_bf16x4(out + offset, v);
}

// Weighted RMSNorm of F32 rows with a BF16 weight, written as BF16. MLX
// promotes the weight to F32 and returns F32 (`x * inv_rms`, then `* weight`,
// each rounded to F32); every caller then casts to BF16, which happens here.
extern "C" __global__ void __launch_bounds__(kRowThreads)
rms_norm_f32_2304(const float* x, const __nv_bfloat16* weight, __nv_bfloat16* out, float eps) {
    const size_t offset = size_t(blockIdx.x) * kHidden + threadIdx.x * kVec;
    float v[kVec];
    load_f32x4(x + offset, v);
    const float scale = inv_rms(row_sum_squares(v), kHidden, eps);
    float w[kVec];
    load_bf16x4(weight + threadIdx.x * kVec, w);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        v[i] = (v[i] * scale) * w[i];
    }
    store_bf16x4(out + offset, v);
}

// Unweighted RMSNorm of F32 rows.
extern "C" __global__ void __launch_bounds__(kRowThreads)
unweighted_rms_norm_f32_2304(const float* x, float* out, float eps) {
    const size_t offset = size_t(blockIdx.x) * kHidden + threadIdx.x * kVec;
    float v[kVec];
    load_f32x4(x + offset, v);
    const float scale = inv_rms(row_sum_squares(v), kHidden, eps);
    *reinterpret_cast<float4*>(out + offset) =
        make_float4(v[0] * scale, v[1] * scale, v[2] * scale, v[3] * scale);
}

// Unweighted RMSNorm of BF16 rows, written as F32: `unweighted_rms_norm(x.astype(float32))`.
// Widening is exact, so this equals cast_bf16_to_f32 followed by
// unweighted_rms_norm_f32_2304 bit for bit.
extern "C" __global__ void __launch_bounds__(kRowThreads)
unweighted_rms_norm_bf16_f32_2304(const __nv_bfloat16* x, float* out, float eps) {
    pdl_trigger();
    pdl_wait();
    const size_t offset = size_t(blockIdx.x) * kHidden + threadIdx.x * kVec;
    float v[kVec];
    load_bf16x4(x + offset, v);
    const float scale = inv_rms(row_sum_squares(v), kHidden, eps);
    *reinterpret_cast<float4*>(out + offset) =
        make_float4(v[0] * scale, v[1] * scale, v[2] * scale, v[3] * scale);
}

// Unweighted RMSNorm of each 128-wide BF16 head, computed in F32. One warp per
// head. MLX uses a single simdgroup for rows of 128, so its second `simd_sum`
// adds only zeros and is skipped here.
extern "C" __global__ void __launch_bounds__(32)
unweighted_rms_norm_bf16_128(const __nv_bfloat16* x, __nv_bfloat16* out, float eps) {
    const size_t offset = size_t(blockIdx.x) * kHeadDim + threadIdx.x * kVec;
    float v[kVec];
    load_bf16x4(x + offset, v);
    const float scale = inv_rms(warp_sum(fma_sum_squares(v)), kHeadDim, eps);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        v[i] *= scale;
    }
    store_bf16x4(out + offset, v);
}

namespace {

// `states * min(threshold / (norm + 1e-6), 1)` with MLX's F32 constants.
__device__ float l2_clip_factor(float sum_squares, float threshold) {
    const float norm = __fsqrt_rn(sum_squares);
    return fminf(__fdiv_rn(threshold, __fadd_rn(norm, 1e-6f)), 1.f);
}

}  // namespace

// L2 clip of each 128-wide BF16 head. One warp per head. `scale` is
// `[heads, 128]`, or null for an unscaled clip; multiplying by MLX's scalar
// 1.0 is exact, so null skips the multiply.
extern "C" __global__ void __launch_bounds__(32)
l2_clip_bf16_128(const __nv_bfloat16* x, const __nv_bfloat16* scale, __nv_bfloat16* out, int heads,
                 float threshold) {
    const size_t offset = size_t(blockIdx.x) * kHeadDim + threadIdx.x * kVec;
    float s[kVec];
    load_bf16x4(x + offset, s);
    if (scale != nullptr) {
        float sc[kVec];
        load_bf16x4(scale + (blockIdx.x % unsigned(heads)) * kHeadDim + threadIdx.x * kVec, sc);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            s[i] = __fmul_rn(s[i], sc[i]);
        }
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        acc = __fadd_rn(acc, __fmul_rn(s[i], s[i]));
    }
    const float factor = l2_clip_factor(warp_sum(acc), threshold);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        s[i] = __fmul_rn(s[i], factor);
    }
    store_bf16x4(out + offset, s);
}

// L2 clip of each 16-wide BF16 head (the relative-position states). One thread
// per head, summing in order as MLX's `row_reduce_small` does. `scale` is
// `[heads, 16]`, or null for an unscaled clip.
extern "C" __global__ void __launch_bounds__(128)
l2_clip_bf16_16(const __nv_bfloat16* x, const __nv_bfloat16* scale, __nv_bfloat16* out, int heads,
                float threshold, unsigned total_heads) {
    constexpr int kDim = 16;
    const unsigned head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= total_heads) {
        return;
    }
    float s[kDim];
#pragma unroll
    for (int i = 0; i < kDim; i += kVec) {
        float part[kVec];
        load_bf16x4(x + size_t(head) * kDim + i, part);
        float sc[kVec] = {1.f, 1.f, 1.f, 1.f};
        if (scale != nullptr) {
            load_bf16x4(scale + (head % unsigned(heads)) * kDim + i, sc);
        }
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
            s[i + j] = __fmul_rn(part[j], sc[j]);
        }
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kDim; ++i) {
        acc = __fadd_rn(acc, __fmul_rn(s[i], s[i]));
    }
    const float factor = l2_clip_factor(acc, threshold);
#pragma unroll
    for (int i = 0; i < kDim; i += kVec) {
        float part[kVec];
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
            part[j] = __fmul_rn(s[i + j], factor);
        }
        store_bf16x4(out + size_t(head) * kDim + i, part);
    }
}

// ---------------------------------------------------------------------------
// Fused decode kernel. It keeps the summation order of the separate kernels, so
// its results differ from theirs only where rsqrt rounds differently.
// ---------------------------------------------------------------------------

#include "conv_step.cuh"

namespace {

constexpr int kKvHeads = 8;
constexpr int kKvDims = kKvHeads * kHeadDim;
constexpr int kQHeads = 24;
constexpr int kRelDim = 16;
// Blocks of 4 warps, one head per warp and 4 consecutive elements per lane as in the
// separate kernels: keys, values and queries, then one block whose first 24 threads
// each clip one relative-position head.
constexpr int kPostThreads = 128;
constexpr int kHeadsPerBlock = kPostThreads / 32;
constexpr int kKBlock0 = 0;
constexpr int kVBlock0 = kKBlock0 + kKvHeads / kHeadsPerBlock;
constexpr int kQBlock0 = kVBlock0 + kKvHeads / kHeadsPerBlock;
constexpr int kRBlock = kQBlock0 + kQHeads / kHeadsPerBlock;

// 1 / sqrt(mean + eps) with two correctly rounded float operations; inv_rms rounds a
// double rsqrt once. They differ by at most one ulp.
__device__ float inv_rms_f32(float sum_squares, int n, float eps) {
    return 1.f / sqrtf(sum_squares / static_cast<float>(n) + eps);
}

// The `FullAttention` query scale for the row at `position`, as log_scale_bf16 computes it.
__device__ float log_scale_tau(uint32_t position, float alpha, float n_floor) {
    const float pos = float(position);
    const float ratio = fmaxf(__fdiv_rn(__fadd_rn(pos, 1.f), n_floor), 1.f);
    return __fadd_rn(1.f, __fmul_rn(alpha, __logf(ratio)));
}

// l2_clip_bf16_128's sum for one head: each lane adds its 4 rounded squares, then a
// warp_sum.
__device__ float l2_sum_128(const float (&v)[kVec]) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        acc = __fadd_rn(acc, __fmul_rn(v[i], v[i]));
    }
    return warp_sum(acc);
}

}  // namespace

// Everything between attention's projections and its KV append for one row:
// - keys: short conv (static, or dynamic when k_dynamic is not null), then the L2 clip;
// - values: the same conv, then the unweighted RMS norm of each head;
// - queries and relative-position states: the scaled L2 clip, then, with log_scale set,
//   the `FullAttention` position scale.
// Each step rounds to BF16 where the separate kernels do.
//
// With kRows, the row is row `row` of a chunk: every per-row pointer is the chunk's,
// the convs read the chunk's earlier rows and leave the state alone, and the row sits
// at position `position`. Without it, the row is the only one and the convs update the
// state.
namespace {

template <bool kRows>
__device__ __forceinline__ void
attention_post_row(const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                   const __nv_bfloat16* __restrict__ k_weight, const __nv_bfloat16* __restrict__ v_weight,
                   const __nv_bfloat16* k_dynamic, const __nv_bfloat16* v_dynamic,
                   const float* __restrict__ k_static, const float* __restrict__ v_static,
                   float* k_state, float* v_state,
                   const __nv_bfloat16* q_raw, const __nv_bfloat16* __restrict__ q_scale,
                   const __nv_bfloat16* r_raw, const __nv_bfloat16* __restrict__ r_scale,
                   __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v,
                   __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ r, int row,
                   uint32_t position, int log_scale, float alpha, float n_floor, float qk_threshold,
                   float rel_threshold, float eps) {
    if constexpr (kRows) {
        k += size_t(row) * kKvDims;
        v += size_t(row) * kKvDims;
        q_raw += size_t(row) * kQHeads * kHeadDim;
        q += size_t(row) * kQHeads * kHeadDim;
        r_raw += size_t(row) * kQHeads * kRelDim;
        r += size_t(row) * kQHeads * kRelDim;
        if (k_dynamic != nullptr) {
            k_dynamic += size_t(row) * kKvHeads * conv_step::kKernel;
            v_dynamic += size_t(row) * kKvHeads * conv_step::kKernel;
        }
    }
    const int b = blockIdx.x;
    const int head_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (b < kQBlock0) {
        const bool is_k = b < kVBlock0;
        const int head = (is_k ? b - kKBlock0 : b - kVBlock0) * kHeadsPerBlock + head_in_block;
        const int c0 = head * kHeadDim + lane * kVec;
        float* state = is_k ? k_state : v_state;
        const __nv_bfloat16* dynamic = is_k ? k_dynamic : v_dynamic;
        float y[kVec];
        if constexpr (kRows) {
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const int c = c0 + i;
                float win[conv_step::kKernel];
                conv_step::load_window(is_k ? k_raw : v_raw, state, c, kKvDims, row, win);
                const __nv_bfloat16 conv =
                    dynamic != nullptr
                        ? conv_step::dynamic_conv_window(win, dynamic + head * conv_step::kKernel,
                                                         is_k ? k_static : v_static, c, kKvDims)
                        : conv_step::short_conv_window(win, is_k ? k_weight : v_weight, c);
                y[i] = __bfloat162float(conv);
            }
        } else {
            float x[kVec];
            load_bf16x4((is_k ? k_raw : v_raw) + c0, x);
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                const int c = c0 + i;
                const __nv_bfloat16 conv =
                    dynamic != nullptr
                        ? conv_step::dynamic_conv(x[i], dynamic + head * conv_step::kKernel,
                                                  is_k ? k_static : v_static, state, c, kKvDims)
                        : conv_step::short_conv(x[i], is_k ? k_weight : v_weight, state, c, kKvDims);
                y[i] = __bfloat162float(conv);
            }
        }
        if (is_k) {
            const float factor = l2_clip_factor(l2_sum_128(y), qk_threshold);
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                y[i] = __fmul_rn(y[i], factor);
            }
            store_bf16x4(k + c0, y);
        } else {
            const float scale = inv_rms_f32(warp_sum(fma_sum_squares(y)), kHeadDim, eps);
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                y[i] *= scale;
            }
            store_bf16x4(v + c0, y);
        }
        return;
    }
    const float tau = log_scale ? log_scale_tau(position, alpha, n_floor) : 1.f;
    // bf16(x * tau) as log_scale_bf16 does it, or x unchanged.
    const auto scaled = [&](float x) {
        return log_scale ? __bfloat162float(__float2bfloat16(__fmul_rn(x, tau))) : x;
    };
    if (b < kRBlock) {
        const int head = (b - kQBlock0) * kHeadsPerBlock + head_in_block;
        const int c0 = head * kHeadDim + lane * kVec;
        float s[kVec];
        float sc[kVec];
        load_bf16x4(q_raw + c0, s);
        load_bf16x4(q_scale + c0, sc);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            s[i] = __fmul_rn(s[i], sc[i]);
        }
        const float factor = l2_clip_factor(l2_sum_128(s), qk_threshold);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            s[i] = scaled(__bfloat162float(__float2bfloat16(__fmul_rn(s[i], factor))));
        }
        store_bf16x4(q + c0, s);
        return;
    }
    // Relative-position heads: one thread each, summing in order as l2_clip_bf16_16 does.
    const int head = threadIdx.x;
    if (head >= kQHeads) {
        return;
    }
    float s[kRelDim];
#pragma unroll
    for (int i = 0; i < kRelDim; i += kVec) {
        float part[kVec];
        float sc[kVec];
        load_bf16x4(r_raw + head * kRelDim + i, part);
        load_bf16x4(r_scale + head * kRelDim + i, sc);
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
            s[i + j] = __fmul_rn(part[j], sc[j]);
        }
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kRelDim; ++i) {
        acc = __fadd_rn(acc, __fmul_rn(s[i], s[i]));
    }
    const float factor = l2_clip_factor(acc, rel_threshold);
#pragma unroll
    for (int i = 0; i < kRelDim; i += kVec) {
        float part[kVec];
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
            part[j] = scaled(__bfloat162float(__float2bfloat16(__fmul_rn(s[i + j], factor))));
        }
        store_bf16x4(r + head * kRelDim + i, part);
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kPostThreads)
attention_post_bf16(const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                    const __nv_bfloat16* __restrict__ k_weight, const __nv_bfloat16* __restrict__ v_weight,
                    const __nv_bfloat16* k_dynamic, const __nv_bfloat16* v_dynamic,
                    const float* __restrict__ k_static, const float* __restrict__ v_static,
                    float* k_state, float* v_state,
                    const __nv_bfloat16* q_raw, const __nv_bfloat16* __restrict__ q_scale,
                    const __nv_bfloat16* r_raw, const __nv_bfloat16* __restrict__ r_scale,
                    __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v,
                    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ r,
                    const uint32_t* position, int log_scale, float alpha, float n_floor,
                    float qk_threshold, float rel_threshold, float eps) {
    pdl_trigger();
    pdl_wait();
    attention_post_row<false>(k_raw, v_raw, k_weight, v_weight, k_dynamic, v_dynamic, k_static, v_static,
                              k_state, v_state, q_raw, q_scale, r_raw, r_scale, k, v, q, r, 0,
                              log_scale ? *position : 0u, log_scale, alpha, n_floor, qk_threshold,
                              rel_threshold, eps);
}

namespace {

// Blocks per row of attention_post_rows_bf16 that compute the relative bias, and the heads
// each one takes.
constexpr int kBiasBlocks = 24;
constexpr int kBiasHeads = kQHeads / kBiasBlocks;
static_assert(kBiasHeads * kBiasBlocks == kQHeads && kBiasHeads <= kPostThreads, "whole heads");

// The relative state of `head` in row `row` before rounding to BF16 storage: the scaled L2
// clip, summed in order as l2_clip_bf16_16 does. `scaled` applies the log scale.
template <class Scaled>
__device__ __forceinline__ void rel_head(const __nv_bfloat16* r_raw, const __nv_bfloat16* r_scale, int row,
                                         int head, float rel_threshold, const Scaled& scaled,
                                         float (&out)[kRelDim]) {
    const size_t at = (size_t(row) * kQHeads + head) * kRelDim;
    float x[kRelDim];
#pragma unroll
    for (int i = 0; i < kRelDim; i += kVec) {
        float part[kVec];
        float sc[kVec];
        load_bf16x4(r_raw + at + i, part);
        load_bf16x4(r_scale + head * kRelDim + i, sc);
#pragma unroll
        for (int j = 0; j < kVec; ++j) {
            x[i + j] = __fmul_rn(part[j], sc[j]);
        }
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kRelDim; ++i) {
        acc = __fadd_rn(acc, __fmul_rn(x[i], x[i]));
    }
    const float factor = l2_clip_factor(acc, rel_threshold);
#pragma unroll
    for (int i = 0; i < kRelDim; ++i) {
        out[i] = scaled(__bfloat162float(__float2bfloat16(__fmul_rn(x[i], factor))));
    }
}

__device__ __forceinline__ float fmul_ftz(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fadd_ftz(float a, float b) {
    float r;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// relative_bias_dot in attention.cu: sum_k r[k] * t[k] in MLX's BF16 matmul order.
__device__ __forceinline__ float relative_bias_dot(const float (&r)[kRelDim], const float (&t)[kRelDim]) {
    float x[kRelDim];
#pragma unroll
    for (int k = 0; k < kRelDim; ++k) {
        x[k] = fmul_ftz(r[k], t[k]);
    }
    float pair[kRelDim / 2];
#pragma unroll
    for (int k = 0; k < kRelDim / 2; ++k) {
        pair[k] = fadd_ftz(x[2 * k], x[2 * k + 1]);
    }
    float quad[kRelDim / 4];
#pragma unroll
    for (int k = 0; k < kRelDim / 4; ++k) {
        quad[k] = fadd_ftz(pair[k], pair[k + kRelDim / 4]);
    }
    return fadd_ftz(fadd_ftz(fadd_ftz(quad[0], quad[1]), quad[2]), quad[3]);
}

}  // namespace

// attention_post_bf16 for each row (blockIdx.y) of a chunk whose first row sits at
// position *start. The per-row buffers are [rows, ...]. The k and v conv states are
// only read; shift_state_rows in conv.cu shifts k_raw and v_raw into them afterwards.
// With a non-null `table` [16, extent], kBiasBlocks more blocks per row also compute the
// relative bias [rows, 24, extent] of the rows' relative states, as relative_bias_rows_bf16
// in attention.cu does, bit for bit; extent is a multiple of 4.
// Every value rounds as attention_post_row computes it. Inputs that earlier kernels write
// are not __restrict__, so that no load of them becomes a non-coherent load, which ptxas
// may move above pdl_wait.
// For a batched step, `states` (rows x 6 device addresses, see attention_batch in
// attention.cu) and `positions` ([rows]) are not null: each row is the first row of its
// own chunk, with its own k and v conv states (words 2 and 3) at its own position.
extern "C" __global__ void __launch_bounds__(kPostThreads)
attention_post_rows_bf16(const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw,
                         const __nv_bfloat16* __restrict__ k_weight, const __nv_bfloat16* __restrict__ v_weight,
                         const __nv_bfloat16* k_dynamic, const __nv_bfloat16* v_dynamic,
                         const float* __restrict__ k_static, const float* __restrict__ v_static,
                         const float* k_state, const float* v_state, const __nv_bfloat16* q_raw,
                         const __nv_bfloat16* __restrict__ q_scale, const __nv_bfloat16* r_raw,
                         const __nv_bfloat16* __restrict__ r_scale, __nv_bfloat16* k, __nv_bfloat16* v,
                         __nv_bfloat16* q, __nv_bfloat16* r, const uint32_t* start, int log_scale, float alpha,
                         float n_floor, float qk_threshold, float rel_threshold, float eps,
                         const __nv_bfloat16* __restrict__ table, __nv_bfloat16* bias, int extent,
                         const unsigned long long* states, const uint32_t* positions) {
    const int row = blockIdx.y;
    const int b = blockIdx.x;
    if (b >= kQBlock0) {
        pdl_trigger();
        pdl_wait();
        const float tau =
            log_scale ? log_scale_tau(positions != nullptr ? positions[row] : *start + uint32_t(row), alpha, n_floor)
                      : 1.f;
        // bf16(x * tau) as log_scale_bf16 does it, or x unchanged.
        const auto scaled = [&](float x) {
            return log_scale ? __bfloat162float(__float2bfloat16(__fmul_rn(x, tau))) : x;
        };
        if (b > kRBlock) {
            // The relative bias of heads h0 .. h0 + kBiasHeads from their relative states,
            // which these blocks compute as the r block does. Thread t takes table columns
            // 4t.. and every 4 * kPostThreads further on.
            __shared__ __align__(16) float rel_s[kBiasHeads][kRelDim];
            const int h0 = (b - kRBlock - 1) * kBiasHeads;
            if (threadIdx.x < kBiasHeads) {
                float rel[kRelDim];
                rel_head(r_raw, r_scale, row, h0 + threadIdx.x, rel_threshold, scaled, rel);
#pragma unroll
                for (int i = 0; i < kRelDim; ++i) {
                    rel_s[threadIdx.x][i] = rel[i];
                }
            }
            __syncthreads();
            for (int c = kVec * threadIdx.x; c < extent; c += kVec * kPostThreads) {
                float t[kVec][kRelDim];
#pragma unroll
                for (int k = 0; k < kRelDim; ++k) {
                    float v[kVec];
                    load_bf16x4(table + size_t(k) * extent + c, v);
#pragma unroll
                    for (int j = 0; j < kVec; ++j) {
                        t[j][k] = v[j];
                    }
                }
#pragma unroll
                for (int h = 0; h < kBiasHeads; ++h) {
                    float rv[kRelDim];
#pragma unroll
                    for (int k = 0; k < kRelDim; ++k) {
                        rv[k] = rel_s[h][k];
                    }
                    float out[kVec];
#pragma unroll
                    for (int j = 0; j < kVec; ++j) {
                        out[j] = relative_bias_dot(rv, t[j]);
                    }
                    store_bf16x4(bias + (size_t(row) * kQHeads + h0 + h) * extent + c, out);
                }
            }
            return;
        }
        if (b < kRBlock) {
            // Queries: the scaled L2 clip of each head, one warp per head.
            const int head = (b - kQBlock0) * kHeadsPerBlock + threadIdx.x / 32;
            const size_t c0 = size_t(row) * kQHeads * kHeadDim + head * kHeadDim + threadIdx.x % 32 * kVec;
            float x[kVec];
            float sc[kVec];
            load_bf16x4(q_raw + c0, x);
            load_bf16x4(q_scale + head * kHeadDim + threadIdx.x % 32 * kVec, sc);
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                x[i] = __fmul_rn(x[i], sc[i]);
            }
            const float factor = l2_clip_factor(l2_sum_128(x), qk_threshold);
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                x[i] = scaled(__bfloat162float(__float2bfloat16(__fmul_rn(x[i], factor))));
            }
            store_bf16x4(q + c0, x);
            return;
        }
        // Relative-position heads: one thread each, summing in order as l2_clip_bf16_16 does.
        const int head = threadIdx.x;
        if (head >= kQHeads) {
            return;
        }
        float rel[kRelDim];
        rel_head(r_raw, r_scale, row, head, rel_threshold, scaled, rel);
#pragma unroll
        for (int i = 0; i < kRelDim; i += kVec) {
            const float part[kVec] = {rel[i], rel[i + 1], rel[i + 2], rel[i + 3]};
            store_bf16x4(r + (size_t(row) * kQHeads + head) * kRelDim + i, part);
        }
        return;
    }
    // Keys and values: short conv (static, or dynamic when k_dynamic is not null) of
    // rows row - 3 .. row, reaching into the conv state before the chunk, then the L2 clip
    // (keys) or the unweighted RMS norm of the head (values).
    const bool is_k = b < kVBlock0;
    const int head = (is_k ? b - kKBlock0 : b - kVBlock0) * kHeadsPerBlock + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int c0 = head * kHeadDim + lane * kVec;
    const __nv_bfloat16* x = is_k ? k_raw : v_raw;
    const float* state = states != nullptr ? reinterpret_cast<const float*>(states[size_t(row) * 6 + (is_k ? 2 : 3)])
                         : is_k            ? k_state
                                           : v_state;
    // The row's place in its chunk: a batched row starts its own.
    const int first = states != nullptr ? row : 0;
    const __nv_bfloat16* dynamic = is_k ? k_dynamic : v_dynamic;
    const bool dyn = dynamic != nullptr;
    // Weights first. Static conv: 4 taps per channel. Dynamic conv: the static weights by
    // lag, [4, 1024].
    float taps[kVec][conv_step::kKernel];
    float stat[conv_step::kKernel][kVec];
    if (dyn) {
        const float* sw = is_k ? k_static : v_static;
#pragma unroll
        for (int lag = 0; lag < conv_step::kKernel; ++lag) {
            load_f32x4(sw + lag * kKvDims + c0, stat[lag]);
        }
    } else {
        const __nv_bfloat16* w = is_k ? k_weight : v_weight;
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            load_bf16x4(w + size_t(c0 + i) * conv_step::kKernel, taps[i]);
        }
    }
    pdl_trigger();
    pdl_wait();
    float win[conv_step::kKernel][kVec];
#pragma unroll
    for (int kk = 0; kk < conv_step::kKernel; ++kk) {
        const int t = row - first - (conv_step::kKernel - 1) + kk;
        if (t >= 0) {
            load_bf16x4(x + size_t(first + t) * kKvDims + c0, win[kk]);
        } else {
            load_f32x4(state + size_t(conv_step::kKernel - 1 + t) * kKvDims + c0, win[kk]);
        }
    }
    float dw[conv_step::kKernel];
    if (dyn) {
        load_bf16x4(dynamic + size_t(row) * kKvHeads * conv_step::kKernel + head * conv_step::kKernel, dw);
    }
    float y[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        float acc = 0.f;
        float out;
        if (dyn) {
            // dynamic_conv_window in conv_step.cuh.
#pragma unroll
            for (int lag = 0; lag < conv_step::kKernel; ++lag) {
                const float w = __fadd_rn(stat[lag][i], dw[lag]);
                acc = __fadd_rn(acc, __fmul_rn(win[conv_step::kKernel - 1 - lag][i], w));
            }
            out = __fdiv_rn(__fadd_rn(acc, win[conv_step::kKernel - 1][i]), conv_step::kDynamicDivisor);
        } else {
            // short_conv_window in conv_step.cuh.
#pragma unroll
            for (int kk = 0; kk < conv_step::kKernel; ++kk) {
                acc = __fadd_rn(acc, __fmul_rn(conv_step::round_bf16(win[kk][i]), taps[i][kk]));
            }
            out = __fdiv_rn(__fadd_rn(conv_step::round_bf16(acc), win[conv_step::kKernel - 1][i]),
                            conv_step::kShortDivisor);
        }
        y[i] = __bfloat162float(__float2bfloat16(out));
    }
    const size_t at = size_t(row) * kKvDims + c0;
    if (is_k) {
        const float factor = l2_clip_factor(l2_sum_128(y), qk_threshold);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            y[i] = __fmul_rn(y[i], factor);
        }
        store_bf16x4(k + at, y);
    } else {
        const float scale = inv_rms_f32(warp_sum(fma_sum_squares(y)), kHeadDim, eps);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            y[i] *= scale;
        }
        store_bf16x4(v + at, y);
    }
}
