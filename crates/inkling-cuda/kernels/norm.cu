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

// The `FullAttention` query scale for the row at *position, as log_scale_bf16 computes it.
__device__ float log_scale_tau(const uint32_t* position, float alpha, float n_floor) {
    const float pos = float(*position);
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
extern "C" __global__ void __launch_bounds__(kPostThreads)
attention_post_bf16(const __nv_bfloat16* __restrict__ k_raw, const __nv_bfloat16* __restrict__ v_raw,
                    const __nv_bfloat16* __restrict__ k_weight, const __nv_bfloat16* __restrict__ v_weight,
                    const __nv_bfloat16* __restrict__ k_dynamic, const __nv_bfloat16* __restrict__ v_dynamic,
                    const float* __restrict__ k_static, const float* __restrict__ v_static,
                    float* __restrict__ k_state, float* __restrict__ v_state,
                    const __nv_bfloat16* __restrict__ q_raw, const __nv_bfloat16* __restrict__ q_scale,
                    const __nv_bfloat16* __restrict__ r_raw, const __nv_bfloat16* __restrict__ r_scale,
                    __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v,
                    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ r,
                    const uint32_t* __restrict__ position, int log_scale, float alpha, float n_floor,
                    float qk_threshold, float rel_threshold, float eps) {
    const int b = blockIdx.x;
    const int head_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    pdl_trigger();
    pdl_wait();
    if (b < kQBlock0) {
        const bool is_k = b < kVBlock0;
        const int head = (is_k ? b - kKBlock0 : b - kVBlock0) * kHeadsPerBlock + head_in_block;
        const int c0 = head * kHeadDim + lane * kVec;
        float x[kVec];
        load_bf16x4((is_k ? k_raw : v_raw) + c0, x);
        float* state = is_k ? k_state : v_state;
        const __nv_bfloat16* dynamic = is_k ? k_dynamic : v_dynamic;
        float y[kVec];
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
