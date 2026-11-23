// MoE expert routing and depth attention.
//
// Each kernel repeats the floating-point operations of the MLX Metal kernel it
// replaces, in the same order, so that results round the same way: MLX's GEMV
// (mlx/backend/metal/kernels/gemv.h), `expert_routing` in
// mlx_lm/kernels/expert_routing.py and `depth_attention` in
// mlx_lm/models/inkling2/kernels.py. Probes of those kernels on the M5 Max
// that made the reference fixtures showed:
//   - Metal contracts `a += b * c` into a fused multiply-add.
//   - simd_sum is an xor butterfly over lane offsets 1, 2, 4, 8, 16.
//   - Float division is correctly rounded.
//   - exp and precise::exp are built on the hardware exp2, which is not
//     correctly rounded; see apple_exp2 below.

#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int kHidden = 2304;
constexpr int kExperts = 128;
constexpr int kTopK = 8;
// MLX's _ROUTER_SCORE_EPS, rounded to F32 as MLX passes it.
constexpr float kScoreEps = 1e-20f;

// simd_sum on Apple GPUs: every lane ends with the same xor-butterfly sum.
__device__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// The Apple GPU exp2, as measured on 17.8 million inputs with normal results.
// The hardware floors x to 23 fraction bits, x = n + j / 2^23, and returns
// 2^n * T[j]; inputs in (-2^-23, 0) return 1. T[j] equals the correctly
// rounded exp2(j / 2^23) for 64% of the 2^23 entries and is one ulp above it
// for the rest. That table is not available here, so the correctly rounded
// value stands in for it and results can be one ulp low. Results outside the
// normal range were not measured and fall back to exp2f.
__device__ float apple_exp2(float x) {
    if (!(fabsf(x) < 126.f)) {
        return exp2f(x);
    }
    if (x < 0.f && x > -0x1p-23f) {
        return 1.f;
    }
    const auto fixed = static_cast<int64_t>(floor(static_cast<double>(x) * 0x1p23));
    const int n = static_cast<int>(fixed >> 23);
    const int j = static_cast<int>(fixed & ((1 << 23) - 1));
    const auto t = static_cast<float>(exp2(static_cast<double>(j) * 0x1p-23));
    return ldexpf(t, n);
}

// metal::exp in MLX's fast-math build: exp2 of the F32 product x * log2(e).
__device__ float apple_fast_exp(float x) {
    return apple_exp2(__fmul_rn(x, 1.4426950408889634f));
}

// metal::precise::exp: exp2 of the rounded product, corrected by the
// product's rounding error. Matched Metal on the same 17.8 million inputs,
// given exact exp2 results.
__device__ float apple_precise_exp(float x) {
    const float log2e = 1.44269502162933349609375f;
    const float log2e_lo = 1.925963033500011079013347625732421875e-8f;
    const float ln2 = 0.693147180559945309f;
    const float hi = __fmul_rn(x, log2e);
    const float lo = fmaf(x, log2e_lo, fmaf(x, log2e, -hi));
    const float e = apple_exp2(hi);
    return fmaf(e, __fmul_rn(lo, ln2), e);
}

}  // namespace

// Threads per router logit: MLX's GEMV for a 128 x 2304 matrix runs 8
// simdgroups across the inner dimension, 4 elements per lane.
constexpr int kLogitsThreads = 256;

// logits[row, e] = dot(F32(hidden[row]), weight[e]) for a [rows, 2304] BF16
// `hidden` and a [128, 2304] F32 `weight`, one block per (row, expert). Sums in
// the order of MLX's GEMV (BN = 8, SN = 32, TN = 4), which MLX uses for a
// single row. With `truncate_tf32` set, weight mantissas are cut to 10 bits
// first, as MLX's GEMM on the M5 does for two or more rows; that GEMM's
// summation order is unknown, so those logits can differ by an ulp.
extern "C" __global__ void __launch_bounds__(kLogitsThreads)
router_logits_2304(const __nv_bfloat16* hidden, const float* weight, float* logits, int truncate_tf32) {
    constexpr int kWarps = kLogitsThreads / 32;
    constexpr int kPerLane = 4;
    constexpr int kStride = kLogitsThreads * kPerLane;
    __shared__ float warp_dots[kWarps];
    const int row = blockIdx.x;
    const int e = blockIdx.y;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const __nv_bfloat16* x = hidden + size_t(row) * kHidden;
    const float* w = weight + size_t(e) * kHidden;

    float acc = 0.f;
    for (int k0 = threadIdx.x * kPerLane; k0 < kHidden; k0 += kStride) {
#pragma unroll
        for (int t = 0; t < kPerLane; ++t) {
            float wk = w[k0 + t];
            if (truncate_tf32) {
                wk = __uint_as_float(__float_as_uint(wk) & 0xffffe000u);
            }
            acc = fmaf(wk, __bfloat162float(x[k0 + t]), acc);
        }
    }
    // simd_shuffle_down reduction: only lane 0's sum is used.
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        warp_dots[warp] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = warp_dots[0];
        for (int s = 1; s < kWarps; ++s) {
            total += warp_dots[s];
        }
        logits[size_t(row) * kExperts + e] = total;
    }
}

// Top-8 of sigmoid(logits) + bias over 128 experts per row, with weights
// score / (sum of the 8 scores + eps) * route_scale. One warp per row; lane l
// holds experts l, l + 32, l + 64 and l + 96. Experts come out in selection
// order: highest biased score first, ties to the lower index.
extern "C" __global__ void expert_routing_128_top8(const float* logits, const float* bias, int32_t* indices,
                                                   float* weights, int rows, float route_scale) {
    constexpr int kPerLane = kExperts / 32;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const float* r = logits + size_t(row) * kExperts;

    float score[kPerLane];
    float biased[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
        const int e = lane + 32 * i;
        const float x = r[e];
        // Metal's form of the sigmoid: exp of |x| cannot overflow to inf/inf.
        const float y = 1.f / (1.f + apple_precise_exp(fabsf(x)));
        score[i] = x < 0 ? y : 1.f - y;
        biased[i] = score[i] + bias[e];
    }

    float w[kTopK];
    for (int k = 0; k < kTopK; ++k) {
        float best = -INFINITY;
        int best_e = kExperts;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int e = lane + 32 * i;
            if (biased[i] > best || (biased[i] == best && e < best_e)) {
                best = biased[i];
                best_e = e;
            }
        }
        float top = best;
        for (int offset = 16; offset > 0; offset /= 2) {
            top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, offset));
        }
        int winner = best == top ? best_e : kExperts;
        for (int offset = 16; offset > 0; offset /= 2) {
            winner = min(winner, __shfl_xor_sync(0xffffffffu, winner, offset));
        }
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            if (lane + 32 * i == winner) {
                s = score[i];
                biased[i] = -INFINITY;
            }
        }
        // One lane holds the score and the rest hold zero, so the sum is exact.
        w[k] = warp_sum(s);
        if (lane == 0) {
            indices[size_t(row) * kTopK + k] = winner;
        }
    }

    if (lane == 0) {
        float total = 0.f;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            total = w[k] + total;
        }
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            weights[size_t(row) * kTopK + k] = w[k] / (total + kScoreEps) * route_scale;
        }
    }
}

// Threads per row in MLX's depth_attention launch, row_threads(2304) in
// mlx_lm/kernels/common.py. The dot products' summation order depends on it.
constexpr int kDepthThreads = 288;

// For each row, nq queries attend over nb + 1 candidate states: `blocks`
// ([nb, rows, 2304]) then `partial` ([rows, 2304]). out is [rows, nq, 2304].
// Each (candidate, query) dot product is summed the way the Metal kernel sums
// it: per thread over d = tid, tid + 288, ..., then across each warp, then
// across the 9 warps in order. Dynamic shared memory holds
// (nb + 1) * nq * 10 floats.
extern "C" __global__ void __launch_bounds__(kDepthThreads)
depth_attention_2304(const float* blocks, const float* partial, const float* queries, float* out, int nb, int nq,
                     float scale) {
    constexpr int kWarps = kDepthThreads / 32;
    extern __shared__ float smem[];
    const int n = nb + 1;
    float* warp_dots = smem;                // [n * nq][kWarps]
    float* probs = smem + n * nq * kWarps;  // [n * nq]

    const int row = blockIdx.x;
    const int rows = gridDim.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    auto state = [&](int i, int d) {
        return i < nb ? blocks[(size_t(i) * rows + row) * kHidden + d] : partial[size_t(row) * kHidden + d];
    };

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nq; ++j) {
            float acc = 0.f;
            for (int d = threadIdx.x; d < kHidden; d += kDepthThreads) {
                acc = fmaf(state(i, d), queries[size_t(j) * kHidden + d], acc);
            }
            acc = warp_sum(acc);
            if (lane == 0) {
                warp_dots[(i * nq + j) * kWarps + warp] = acc;
            }
        }
    }
    __syncthreads();

    // Softmax over the n candidates of each query.
    if (threadIdx.x < nq) {
        const int j = threadIdx.x;
        float m = -INFINITY;
        for (int i = 0; i < n; ++i) {
            float t = 0.f;
            for (int s = 0; s < kWarps; ++s) {
                t += warp_dots[(i * nq + j) * kWarps + s];
            }
            probs[i * nq + j] = t * scale;
            m = fmaxf(m, probs[i * nq + j]);
        }
        float z = 0.f;
        for (int i = 0; i < n; ++i) {
            probs[i * nq + j] = apple_fast_exp(probs[i * nq + j] - m);
            z += probs[i * nq + j];
        }
        for (int i = 0; i < n; ++i) {
            probs[i * nq + j] /= z;
        }
    }
    __syncthreads();

    for (int d = threadIdx.x; d < kHidden; d += kDepthThreads) {
        for (int j = 0; j < nq; ++j) {
            float o = 0.f;
            for (int i = 0; i < n; ++i) {
                o = fmaf(probs[i * nq + j], state(i, d), o);
            }
            out[(size_t(row) * nq + j) * kHidden + d] = o;
        }
    }
}
