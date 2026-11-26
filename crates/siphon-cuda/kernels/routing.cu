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

#include "cluster_sm89.cuh"
#include "pdl.cuh"

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

// ---------------------------------------------------------------------------
// Fused single-row kernels for decode. Each keeps the summation order of the
// separate kernels it replaces, so results differ from theirs only where exp and
// rsqrt round differently (by an ulp, for a small share of inputs).
// ---------------------------------------------------------------------------

namespace {

constexpr int kDepthWarps = kDepthThreads / 32;
constexpr int kDepthPer = kHidden / kDepthThreads;
static_assert(kHidden % kDepthThreads == 0, "threads cover the row");
// The unweighted and weighted RMS norm kernels in norm.cu run 576 threads of 4
// elements. A block of 288 threads repeats their sums as virtual threads t and
// t + 288; virtual warps w and w + 9 then fall on real warp w.
constexpr int kNormVirtualWarps = 2 * kDepthWarps;
constexpr int kFusedMaxCandidates = 6;
constexpr int kFusedMaxQueries = 4;
constexpr int kFusedDots = kFusedMaxCandidates * kFusedMaxQueries;
constexpr int kDests = 4;

// The norm kernels' sum of squares of a 2304 row, given this thread's two runs of 4:
// elements 4t.. (a) and 4(t + 288).. (b). Every thread gets the total. `vw` is a
// [kNormVirtualWarps] shared array that no other thread reads concurrently.
__device__ float norm_sum_squares(const float (&a)[4], const float (&b)[4], float* vw) {
    float pa = 0.f;
    float pb = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        pa = fmaf(a[i], a[i], pa);
        pb = fmaf(b[i], b[i], pb);
    }
    pa = warp_sum(pa);
    pb = warp_sum(pb);
    const int lane = threadIdx.x % 32;
    if (lane == 0) {
        vw[threadIdx.x / 32] = pa;
        vw[threadIdx.x / 32 + kDepthWarps] = pb;
    }
    __syncthreads();
    return warp_sum(lane < kNormVirtualWarps ? vw[lane] : 0.f);
}

// 1 / sqrt(mean + eps) with two correctly rounded float operations. The norm kernels
// round a double rsqrt once instead; the two differ by at most one ulp.
__device__ float inv_rms_f32(float sum_squares, float eps) {
    return 1.f / sqrtf(sum_squares / static_cast<float>(kHidden) + eps);
}

// MLX's fast-math exp, exp2(x * log2(e)) with the product rounded, through CUDA's
// exp2f rather than apple_exp2's FP64 emulation of the Apple table.
__device__ float fast_exp(float x) {
    return exp2f(__fmul_rn(x, 1.4426950408889634f));
}

__device__ void load4(const float* p, float (&v)[4]) {
    const float4 t = *reinterpret_cast<const float4*>(p);
    v[0] = t.x;
    v[1] = t.y;
    v[2] = t.z;
    v[3] = t.w;
}

__device__ void load4(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 packed = *reinterpret_cast<const uint2*>(p);
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = __bfloat162float(vals[i]);
    }
}

__device__ void store4(__nv_bfloat16* p, const float (&v)[4]) {
    uint2 packed;
    auto* vals = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vals[i] = __float2bfloat16(v[i]);
    }
    *reinterpret_cast<uint2*>(p) = packed;
}

}  // namespace

// Depth attention for one row with the RMS norms around it, in one block.
//
// The candidates are the nb rows of `blocks` and the partial state. The partial
// is `partial_f32` or, if null, `partial_bf16` widened; with normalize_partial
// it first gets an unweighted RMS norm. `normed_partial`, if not null, receives
// the partial as used. The nq queries give out[j] = sum_i softmax_i(scale *
// candidate_i . query_j) * candidate_i, written to `out` if not null.
//
// Each non-null dst[k] receives row min(k, nq - 1) of the output after an RMS
// norm, multiplied by weight[k] if that is not null, rounded to BF16: MLX's
// `norm(x).astype(bfloat16)` for an F32 x.
//
// Sums follow depth_attention_2304 and the norm kernels. All loads are issued
// up front, which the separate kernel, looping over candidates and queries,
// did not do.
//
// `block_stride` is the distance between block states.
namespace {

__device__ __forceinline__ void
depth_attention_fused_row(const float* blocks, int block_stride, int nb, const float* partial_f32,
                          const __nv_bfloat16* partial_bf16, int normalize_partial, float* normed_partial,
                          const float* __restrict__ queries, int nq, float* out, __nv_bfloat16* dst0,
                          __nv_bfloat16* dst1, __nv_bfloat16* dst2, __nv_bfloat16* dst3,
                          const __nv_bfloat16* weight0, const __nv_bfloat16* weight1,
                          const __nv_bfloat16* weight2, const __nv_bfloat16* weight3, float eps,
                          float scale) {
    __shared__ float warp_dots[kFusedDots][kDepthWarps];
    __shared__ float probs[kFusedDots];
    __shared__ float vw[kFusedMaxQueries + 1][kNormVirtualWarps];
    __shared__ float rows[kFusedMaxQueries][kHidden];
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int n = nb + 1;
    __nv_bfloat16* const dst[kDests] = {dst0, dst1, dst2, dst3};
    const __nv_bfloat16* const weight[kDests] = {weight0, weight1, weight2, weight3};
    // Element m of this thread in the dot layout is d = t + 288 m, as in depth_attention_2304.
    float q[kFusedMaxQueries][kDepthPer];
    float s[kFusedMaxCandidates][kDepthPer];
#pragma unroll
    for (int m = 0; m < kDepthPer; ++m) {
#pragma unroll
        for (int j = 0; j < kFusedMaxQueries; ++j) {
            q[j][m] = j < nq ? queries[size_t(j) * kHidden + t + m * kDepthThreads] : 0.f;
        }
    }
    pdl_trigger();
    pdl_wait();
#pragma unroll
    for (int m = 0; m < kDepthPer; ++m) {
        const int d = t + m * kDepthThreads;
        const float pd = partial_f32 != nullptr ? partial_f32[d] : __bfloat162float(partial_bf16[d]);
#pragma unroll
        for (int i = 0; i < kFusedMaxCandidates; ++i) {
            s[i][m] = i < nb ? blocks[size_t(i) * block_stride + d] : (i == nb ? pd : 0.f);
        }
    }
    // The partial's scale; a select per candidate keeps s in registers.
    float inv_p = 1.f;
    if (normalize_partial) {
        // The norm kernels' layout: elements 4t.. and 4(t + 288)...
        float a[4];
        float b[4];
        const int ca = 4 * t;
        const int cb = 4 * (t + kDepthThreads);
        if (partial_f32 != nullptr) {
            load4(partial_f32 + ca, a);
            load4(partial_f32 + cb, b);
        } else {
            load4(partial_bf16 + ca, a);
            load4(partial_bf16 + cb, b);
        }
        const float inv = inv_rms_f32(norm_sum_squares(a, b, vw[kFusedMaxQueries]), eps);
        inv_p = inv;
        if (normed_partial != nullptr) {
            *reinterpret_cast<float4*>(normed_partial + ca) = make_float4(a[0] * inv, a[1] * inv, a[2] * inv, a[3] * inv);
            *reinterpret_cast<float4*>(normed_partial + cb) = make_float4(b[0] * inv, b[1] * inv, b[2] * inv, b[3] * inv);
        }
    }
#pragma unroll
    for (int i = 0; i < kFusedMaxCandidates; ++i) {
        const float f = i == nb ? inv_p : 1.f;
#pragma unroll
        for (int m = 0; m < kDepthPer; ++m) {
            s[i][m] *= f;
        }
    }
    if (!normalize_partial && normed_partial != nullptr) {
#pragma unroll
        for (int m = 0; m < kDepthPer; ++m) {
            const int d = t + m * kDepthThreads;
            normed_partial[d] = partial_f32 != nullptr ? partial_f32[d] : __bfloat162float(partial_bf16[d]);
        }
    }

#pragma unroll
    for (int i = 0; i < kFusedMaxCandidates; ++i) {
#pragma unroll
        for (int j = 0; j < kFusedMaxQueries; ++j) {
            if (i < n && j < nq) {
                float acc = 0.f;
#pragma unroll
                for (int m = 0; m < kDepthPer; ++m) {
                    acc = fmaf(s[i][m], q[j][m], acc);
                }
                acc = warp_sum(acc);
                if (lane == 0) {
                    warp_dots[i * kFusedMaxQueries + j][warp] = acc;
                }
            }
        }
    }
    __syncthreads();

    // Softmax over the candidates of each query, with depth_attention_2304's sums: warp j
    // takes query j and lane i candidate i. The maximum does not depend on order; lane 0
    // adds the exponentials in candidate order.
    if (warp < nq) {
        const int j = warp;
        const int i = lane;
        float logit = -INFINITY;
        if (i < n) {
            float tot = 0.f;
            for (int w = 0; w < kDepthWarps; ++w) {
                tot += warp_dots[i * kFusedMaxQueries + j][w];
            }
            logit = tot * scale;
        }
        float mx = logit;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, offset));
        }
        const float e = i < n ? fast_exp(logit - mx) : 0.f;
        float z = 0.f;
#pragma unroll
        for (int c = 0; c < kFusedMaxCandidates; ++c) {
            const float ec = __shfl_sync(0xffffffffu, e, c);
            if (c < n) {
                z += ec;
            }
        }
        if (i < n) {
            probs[i * kFusedMaxQueries + j] = e / z;
        }
    }
    __syncthreads();

    // Rows that some destination normalizes.
    unsigned normed_rows = 0;
#pragma unroll
    for (int k = 0; k < kDests; ++k) {
        if (dst[k] != nullptr) {
            normed_rows |= 1u << min(k, nq - 1);
        }
    }
#pragma unroll
    for (int j = 0; j < kFusedMaxQueries; ++j) {
        if (j >= nq) {
            continue;
        }
#pragma unroll
        for (int m = 0; m < kDepthPer; ++m) {
            const int d = t + m * kDepthThreads;
            float o = 0.f;
#pragma unroll
            for (int i = 0; i < kFusedMaxCandidates; ++i) {
                if (i < n) {
                    o = fmaf(probs[i * kFusedMaxQueries + j], s[i][m], o);
                }
            }
            if (out != nullptr) {
                out[size_t(j) * kHidden + d] = o;
            }
            if (normed_rows & (1u << j)) {
                rows[j][d] = o;
            }
        }
    }
    if (normed_rows == 0) {
        return;
    }
    __syncthreads();
    const int ca = 4 * t;
    const int cb = 4 * (t + kDepthThreads);
    // The norm kernels' sums for every normalized row, sharing one barrier.
    float ra[kFusedMaxQueries][4];
    float rb[kFusedMaxQueries][4];
#pragma unroll
    for (int j = 0; j < kFusedMaxQueries; ++j) {
        if (!(normed_rows & (1u << j))) {
            continue;
        }
        float pa = 0.f;
        float pb = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ra[j][i] = rows[j][ca + i];
            rb[j][i] = rows[j][cb + i];
            pa = fmaf(ra[j][i], ra[j][i], pa);
            pb = fmaf(rb[j][i], rb[j][i], pb);
        }
        pa = warp_sum(pa);
        pb = warp_sum(pb);
        if (lane == 0) {
            vw[j][warp] = pa;
            vw[j][warp + kDepthWarps] = pb;
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kFusedMaxQueries; ++j) {
        if (!(normed_rows & (1u << j))) {
            continue;
        }
        const float (&a)[4] = ra[j];
        const float (&b)[4] = rb[j];
        const float inv = inv_rms_f32(warp_sum(lane < kNormVirtualWarps ? vw[j][lane] : 0.f), eps);
#pragma unroll
        for (int k = 0; k < kDests; ++k) {
            if (dst[k] == nullptr || min(k, nq - 1) != j) {
                continue;
            }
            float ya[4];
            float yb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ya[i] = a[i] * inv;
                yb[i] = b[i] * inv;
            }
            if (weight[k] != nullptr) {
                float wa[4];
                float wb[4];
                load4(weight[k] + ca, wa);
                load4(weight[k] + cb, wb);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ya[i] *= wa[i];
                    yb[i] *= wb[i];
                }
            }
            store4(dst[k] + ca, ya);
            store4(dst[k] + cb, yb);
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kDepthThreads)
depth_attention_fused_2304(const float* blocks, int nb, const float* partial_f32,
                           const __nv_bfloat16* partial_bf16, int normalize_partial, float* normed_partial,
                           const float* __restrict__ queries, int nq, float* out, __nv_bfloat16* dst0,
                           __nv_bfloat16* dst1, __nv_bfloat16* dst2, __nv_bfloat16* dst3,
                           const __nv_bfloat16* weight0, const __nv_bfloat16* weight1,
                           const __nv_bfloat16* weight2, const __nv_bfloat16* weight3, float eps,
                           float scale) {
    depth_attention_fused_row(blocks, kHidden, nb, partial_f32, partial_bf16, normalize_partial, normed_partial,
                              queries, nq, out, dst0, dst1, dst2, dst3, weight0, weight1, weight2, weight3, eps,
                              scale);
}

namespace {

constexpr int kRouterThreads = kDepthThreads;
constexpr int kExpertsPerBlock = 2;
constexpr int kRouterBlocks = kExperts / kExpertsPerBlock;
// router_logits_2304's layout: thread t < 256 sums columns 4t + 1024 c + 0..3.
constexpr int kGemvChunks = (kHidden + kLogitsThreads * 4 - 1) / (kLogitsThreads * 4);

}  // namespace

namespace {

// The top-8 routing of expert_routing_128_top8 for one row of 128 logits, with the
// sigmoid through CUDA's expf. Every thread of the block must call it.
__device__ __forceinline__ void route_top8(const float* logits, const float* __restrict__ bias, int32_t* indices,
                                           float* weights, float route_scale) {
    // Each expert's rank among the biased scores, ties to the lower index, picks the top 8
    // without sequential rounds.
    __shared__ float biased[kExperts];
    __shared__ float score[kExperts];
    __shared__ float top[kTopK];
    const int e = threadIdx.x;
    if (e < kExperts) {
        const float l = __ldcg(logits + e);
        // Metal's form of the sigmoid: exp of |x| cannot overflow to inf/inf.
        const float y = 1.f / (1.f + expf(fabsf(l)));
        score[e] = l < 0 ? y : 1.f - y;
        biased[e] = score[e] + bias[e];
    }
    __syncthreads();
    if (e < kExperts) {
        const float mine = biased[e];
        int rank = 0;
#pragma unroll 8
        for (int other = 0; other < kExperts; ++other) {
            const float b = biased[other];
            rank += b > mine || (b == mine && other < e);
        }
        if (rank < kTopK) {
            indices[rank] = e;
            top[rank] = score[e];
        }
    }
    __syncthreads();
    if (threadIdx.x < kTopK) {
        float total = 0.f;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            total = top[k] + total;
        }
        weights[threadIdx.x] = top[threadIdx.x] / (total + kScoreEps) * route_scale;
    }
}

}  // namespace

// The MoE input norm and router for one row: x_out = bf16(rms_norm(x) * norm_weight),
// logits = F32(x_out) @ router.T, then the top-8 routing of expert_routing_128_top8.
// The norm and the logits sum as rms_norm_f32_2304 and router_logits_2304 do; the
// sigmoid uses CUDA's expf.
//
// Each block computes the norm itself and the logits of two experts. The last block
// to finish, found with `counter`, routes. `counter` must start at 0; the last block
// resets it, so the kernel can run again from a graph.
namespace {

__device__ __forceinline__ void
moe_router_row(const float* x, const __nv_bfloat16* __restrict__ norm_weight, float eps,
               const float* __restrict__ router, const float* __restrict__ bias,
               __nv_bfloat16* x_out, float* logits, int32_t* indices, float* weights,
               unsigned* counter, float route_scale) {
    __shared__ float vw[kNormVirtualWarps];
    __shared__ float xs[kHidden];
    __shared__ float warp_dots[kExpertsPerBlock][kLogitsThreads / 32];
    __shared__ bool last;
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int e0 = blockIdx.x * kExpertsPerBlock;

    float4 w[kExpertsPerBlock][kGemvChunks];
#pragma unroll
    for (int e = 0; e < kExpertsPerBlock; ++e) {
#pragma unroll
        for (int c = 0; c < kGemvChunks; ++c) {
            const int k0 = t * 4 + c * kLogitsThreads * 4;
            w[e][c] = t < kLogitsThreads && k0 < kHidden
                          ? *reinterpret_cast<const float4*>(router + size_t(e0 + e) * kHidden + k0)
                          : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    const int ca = 4 * t;
    const int cb = 4 * (t + kRouterThreads);
    float wa[4];
    float wb[4];
    load4(norm_weight + ca, wa);
    load4(norm_weight + cb, wb);
    pdl_trigger();
    pdl_wait();
    float a[4];
    float b[4];
    load4(x + ca, a);
    load4(x + cb, b);
    const float inv = inv_rms_f32(norm_sum_squares(a, b, vw), eps);
    float ya[4];
    float yb[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        ya[i] = __bfloat162float(__float2bfloat16((a[i] * inv) * wa[i]));
        yb[i] = __bfloat162float(__float2bfloat16((b[i] * inv) * wb[i]));
        xs[ca + i] = ya[i];
        xs[cb + i] = yb[i];
    }
    if (blockIdx.x == 0) {
        store4(x_out + ca, ya);
        store4(x_out + cb, yb);
    }
    __syncthreads();

    if (t < kLogitsThreads) {
        float acc[kExpertsPerBlock] = {};
#pragma unroll
        for (int c = 0; c < kGemvChunks; ++c) {
            const int k0 = t * 4 + c * kLogitsThreads * 4;
            if (k0 < kHidden) {
#pragma unroll
                for (int e = 0; e < kExpertsPerBlock; ++e) {
                    const float wv[4] = {w[e][c].x, w[e][c].y, w[e][c].z, w[e][c].w};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        acc[e] = fmaf(wv[i], xs[k0 + i], acc[e]);
                    }
                }
            }
        }
#pragma unroll
        for (int e = 0; e < kExpertsPerBlock; ++e) {
            for (int offset = 16; offset > 0; offset /= 2) {
                acc[e] += __shfl_down_sync(0xffffffffu, acc[e], offset);
            }
            if (lane == 0) {
                warp_dots[e][warp] = acc[e];
            }
        }
    }
    __syncthreads();
    if (t == 0) {
#pragma unroll
        for (int e = 0; e < kExpertsPerBlock; ++e) {
            float total = warp_dots[e][0];
            for (int s = 1; s < kLogitsThreads / 32; ++s) {
                total += warp_dots[e][s];
            }
            logits[e0 + e] = total;
        }
        __threadfence();
        last = atomicAdd(counter, 1u) == kRouterBlocks - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();

    route_top8(logits, bias, indices, weights, route_scale);
    if (threadIdx.x == 0) {
        *counter = 0;
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kRouterThreads)
moe_router_2304(const float* x, const __nv_bfloat16* __restrict__ norm_weight, float eps,
                const float* __restrict__ router, const float* __restrict__ bias,
                __nv_bfloat16* x_out, float* logits, int32_t* indices, float* weights,
                unsigned* counter, float route_scale) {
    moe_router_row(x, norm_weight, eps, router, bias, x_out, logits, indices, weights, counter, route_scale);
}

// ---------------------------------------------------------------------------
// The router for the rows of a chunk, from rows that are already normalized (the kernel
// before writes them as a weighted RMS norm destination): logits = F32(x) @ router.T as a
// tiled F32 matrix product split along the inner dimension, then the top-8 routing of
// moe_router_2304. The logits sum in their own order, so they differ from
// moe_router_2304's by F32 rounding.
// ---------------------------------------------------------------------------

namespace {

constexpr int kRouterTileRows = 64;
constexpr int kRouterTileK = 16;
constexpr int kRouterTileThreads = 256;
// Each thread holds 4 rows (4 tr + 0..3) by 8 experts (4 te + 0..3 and 64 + 4 te + 0..3).
constexpr int kRouterRowGroups = kRouterTileRows / 4;
constexpr int kRouterExpertGroups = kRouterTileThreads / kRouterRowGroups;
static_assert(kRouterExpertGroups * 8 == kExperts, "threads cover the experts");
// The x tile holds [k][row], padded so that the transposing stores spread over banks.
constexpr int kRouterXPitch = kRouterTileRows + 4;

}  // namespace

// partials[s, row, e] = sum over k in slice s of x[row, k] * router[e, k], for the
// [rows, 2304] BF16 x and the [128, 2304] F32 router. Slice s is columns
// s * slice .. (s + 1) * slice, a multiple of 16. Grid: (ceil(rows / 64), 2304 / slice),
// 256 threads.
extern "C" __global__ void __launch_bounds__(kRouterTileThreads)
moe_router_logits_rows_2304(const __nv_bfloat16* x, const float* __restrict__ router,
                            float* partials, int rows, int slice) {
    __shared__ __align__(16) float xs[2][kRouterTileK][kRouterXPitch];
    __shared__ __align__(16) float ws[2][kRouterTileK][kExperts];
    const int tid = threadIdx.x;
    const int r0 = blockIdx.x * kRouterTileRows;
    const int k0 = blockIdx.y * slice;
    const int tiles = slice / kRouterTileK;
    const int tr = tid / kRouterExpertGroups;
    const int te = tid % kRouterExpertGroups;

    // Loader roles. x: thread i takes row i / 4 and columns 4 (i % 4) .. of the tile.
    // router: thread i takes experts i / 4 and 64 + i / 4, columns 4 (i % 4) ...
    const int xr = tid / 4;
    const int xk = tid % 4 * 4;
    const bool x_valid = r0 + xr < rows;
    const __nv_bfloat16* xp = x + size_t(x_valid ? r0 + xr : 0) * kHidden + k0 + xk;
    const int we = tid / 4;
    const int wk = tid % 4 * 4;
    const float* wp0 = router + size_t(we) * kHidden + k0 + wk;
    const float* wp1 = router + size_t(we + 64) * kHidden + k0 + wk;

    float4 w0 = *reinterpret_cast<const float4*>(wp0);
    float4 w1 = *reinterpret_cast<const float4*>(wp1);
    pdl_trigger();
    pdl_wait();
    uint2 xv = x_valid ? *reinterpret_cast<const uint2*>(xp) : make_uint2(0, 0);

    const auto stage = [&](int buf) {
        const auto* xb = reinterpret_cast<const __nv_bfloat16*>(&xv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            xs[buf][xk + j][xr] = __bfloat162float(xb[j]);
        }
        const float a[4] = {w0.x, w0.y, w0.z, w0.w};
        const float b[4] = {w1.x, w1.y, w1.z, w1.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            ws[buf][wk + j][we] = a[j];
            ws[buf][wk + j][we + 64] = b[j];
        }
    };
    stage(0);
    __syncthreads();

    float acc[4][8] = {};
    for (int tile = 0; tile < tiles; ++tile) {
        const int buf = tile & 1;
        if (tile + 1 < tiles) {
            const int off = (tile + 1) * kRouterTileK;
            w0 = *reinterpret_cast<const float4*>(wp0 + off);
            w1 = *reinterpret_cast<const float4*>(wp1 + off);
            xv = x_valid ? *reinterpret_cast<const uint2*>(xp + off) : make_uint2(0, 0);
        }
#pragma unroll
        for (int k = 0; k < kRouterTileK; ++k) {
            const float4 xa = *reinterpret_cast<const float4*>(&xs[buf][k][4 * tr]);
            const float4 wa = *reinterpret_cast<const float4*>(&ws[buf][k][4 * te]);
            const float4 wb = *reinterpret_cast<const float4*>(&ws[buf][k][64 + 4 * te]);
            const float xr4[4] = {xa.x, xa.y, xa.z, xa.w};
            const float we8[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[i][j] = fmaf(xr4[i], we8[j], acc[i][j]);
                }
            }
        }
        if (tile + 1 < tiles) {
            stage(buf ^ 1);
        }
        __syncthreads();
    }

    float* out = partials + size_t(blockIdx.y) * rows * kExperts;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = r0 + 4 * tr + i;
        if (row < rows) {
            float* o = out + size_t(row) * kExperts;
            *reinterpret_cast<float4*>(o + 4 * te) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *reinterpret_cast<float4*>(o + 64 + 4 * te) =
                make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        }
    }
}

// For each row (blockIdx.x), logits[row, e] = the sum of the `splits` partials in order,
// then moe_router_2304's top-8 routing into indices and weights ([rows, 8]). 128 threads.
extern "C" __global__ void __launch_bounds__(kExperts)
moe_router_route_rows_2304(const float* partials, int splits, const float* __restrict__ bias,
                           float* logits, int32_t* indices, float* weights, int rows, float route_scale) {
    const size_t row = blockIdx.x;
    const int e = threadIdx.x;
    pdl_trigger();
    pdl_wait();
    float total = 0.f;
    for (int s = 0; s < splits; ++s) {
        total += partials[(size_t(s) * rows + row) * kExperts + e];
    }
    logits[row * kExperts + e] = total;
    // route_top8 reads each logit back in the thread that wrote it.
    route_top8(logits + row * kExperts, bias, indices + row * kTopK, weights + row * kTopK, route_scale);
}

// ---------------------------------------------------------------------------
// Decode kernels with their own summation order. They keep F32 accumulation and round
// to BF16 where the kernels above do, but sum in an order chosen for latency, so their
// results differ from the kernels above by F32 rounding.
// ---------------------------------------------------------------------------

namespace {

constexpr int kStepThreads = 288;
constexpr int kStepWarps = kStepThreads / 32;
constexpr int kStepBlocks = 5;
constexpr int kStepQueries = 4;
// Slot of the partial among the candidates, after the block states.
constexpr int kStepPartial = kStepBlocks;
// Index of the partial's sum of squares among the values of the first reduction; the
// dot of candidate i and query j is value 4 i + j.
constexpr int kStepSquares = (kStepPartial + 1) * kStepQueries;
static_assert(kStepSquares < 32, "the first reduction fits a warp");
static_assert(kStepThreads * 8 == kHidden, "two runs of 4 per thread cover the row");

// How depth_attention_step_2304 gets its partial candidate.
enum StepPartial : int {
    // An F32 row, already normalized.
    kPartialNormalized = 0,
    // An F32 row that gets an unweighted RMS norm.
    kPartialF32 = 1,
    // A BF16 row, widened, that gets an unweighted RMS norm.
    kPartialBf16 = 2,
    // The post-attention conv and residual add of short_conv_residual_2304, then an
    // unweighted RMS norm.
    kPartialConvResidual = 3,
};

constexpr float kShortConvDivisor = static_cast<float>(2.23606797749979);  // sqrt(K + 1)

// x / kShortConvDivisor, correctly rounded, without the slow-path call of a general
// division: the product with the rounded reciprocal, corrected once by its exact
// remainder. It equals __fdiv_rn(x, kShortConvDivisor) for every x of every binade
// checked exhaustively, 2^-100 to 2^99; the division scales exactly by powers of two
// between them. copysignf keeps the sign of a zero x.
__device__ __forceinline__ float div_short_conv(float x) {
    const float r = 1.f / kShortConvDivisor;
    const float q = __fmul_rn(x, r);
    return copysignf(fmaf(fmaf(-q, kShortConvDivisor, x), r, q), x);
}

// Each lane passes 32 values; lane l gets the warp's sum of value l, in 31 shuffles.
__device__ __forceinline__ float warp_transpose_sum(float (&v)[32]) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int half = 16; half >= 1; half /= 2) {
        const bool upper = (lane & half) != 0;
#pragma unroll
        for (int k = 0; k < half; ++k) {
            const float send = upper ? v[k] : v[k + half];
            const float keep = upper ? v[k + half] : v[k];
            v[k] = keep + __shfl_xor_sync(0xffffffffu, send, half);
        }
    }
    return v[0];
}

__device__ __forceinline__ void load8(const float* p, int ca, int cb, float (&v)[8]) {
    const float4 a = *reinterpret_cast<const float4*>(p + ca);
    const float4 b = *reinterpret_cast<const float4*>(p + cb);
    v[0] = a.x;
    v[1] = a.y;
    v[2] = a.z;
    v[3] = a.w;
    v[4] = b.x;
    v[5] = b.y;
    v[6] = b.z;
    v[7] = b.w;
}

__device__ __forceinline__ void store8(float* p, int ca, int cb, const float (&v)[8]) {
    *reinterpret_cast<float4*>(p + ca) = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(p + cb) = make_float4(v[4], v[5], v[6], v[7]);
}

__device__ __forceinline__ void unpack_bf16x4(uint2 packed, float* v) {
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = __bfloat162float(vals[i]);
    }
}

__device__ __forceinline__ float round_to_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// Pulls the line holding p into L2. L2 is coherent, so this is safe before pdl_wait even
// for data an earlier kernel is still writing: it only turns a later load's DRAM access
// into an L2 hit.
__device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

}  // namespace

// Depth attention for one row with the norms around it, as depth_attention_fused_2304,
// for decode. Thread t holds elements 4t.. and 4(t + 288).. of every row, loaded as
// vectors. The dot products, the partial's sum of squares and the output norms are
// summed per thread, then across each warp, then across the 9 warps in order; the
// partial's scale multiplies its dot products instead of its elements. Every warp
// computes the softmax itself, so the kernel has two block barriers.
//
// `mode` picks the partial (see StepPartial). With kPartialConvResidual, `partial_bf16`
// is the attention output x, and the kernel computes residual = f32(conv_hidden) +
// f32(short_conv(x)) (or f32(short_conv(x)) for a null conv_hidden) with conv_weight
// ([2304, 4] BF16) and conv_state ([3, 2304] F32, updated in place), as
// short_conv_residual_2304 does, and writes it to `residual`.
//
// Block b of the grid takes queries + b * query_stride and writes out + b * out_stride
// and dst_k + b * dst_stride, over the same candidates; only block 0 writes
// normed_partial, residual and conv_state. Destination k receives output row
// min(k, nq - 1) after an RMS norm, times weight_k if that is not null, as BF16.
extern "C" __global__ void __launch_bounds__(kStepThreads)
depth_attention_step_2304(const float* blocks, int nb, const float* partial_f32,
                          const __nv_bfloat16* partial_bf16, int mode, const __nv_bfloat16* __restrict__ conv_weight,
                          float* conv_state, const __nv_bfloat16* conv_hidden, float* residual,
                          float* normed_partial, const float* __restrict__ queries, int nq, int query_stride,
                          float* out, int out_stride, __nv_bfloat16* dst0, __nv_bfloat16* dst1,
                          __nv_bfloat16* dst2, __nv_bfloat16* dst3, int dst_stride,
                          const __nv_bfloat16* __restrict__ weight0, const __nv_bfloat16* __restrict__ weight1,
                          const __nv_bfloat16* __restrict__ weight2, const __nv_bfloat16* __restrict__ weight3,
                          float eps, float scale) {
    __shared__ float red[kStepWarps][32];
    __shared__ float red_norm[kStepQueries][kStepWarps];
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int ca = 4 * t;
    const int cb = 4 * (t + kStepThreads);
    const size_t b = blockIdx.x;
    __nv_bfloat16* const dst[kDests] = {dst0, dst1, dst2, dst3};
    const __nv_bfloat16* const weight[kDests] = {weight0, weight1, weight2, weight3};

    // Weights first: they do not depend on earlier kernels.
    float q[kStepQueries][8];
#pragma unroll
    for (int j = 0; j < kStepQueries; ++j) {
        if (j < nq) {
            load8(queries + b * query_stride + size_t(j) * kHidden, ca, cb, q[j]);
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                q[j][e] = 0.f;
            }
        }
    }
    uint2 wpack[kDests][2];
#pragma unroll
    for (int k = 0; k < kDests; ++k) {
        if (dst[k] != nullptr && weight[k] != nullptr) {
            wpack[k][0] = *reinterpret_cast<const uint2*>(weight[k] + ca);
            wpack[k][1] = *reinterpret_cast<const uint2*>(weight[k] + cb);
        }
    }
    // The conv taps of this thread's 8 channels, 4 per channel.
    uint4 taps[4];
    if (mode == kPartialConvResidual) {
        const uint4* wa = reinterpret_cast<const uint4*>(conv_weight + 4 * ca);
        const uint4* wb = reinterpret_cast<const uint4*>(conv_weight + 4 * cb);
        taps[0] = wa[0];
        taps[1] = wa[1];
        taps[2] = wb[0];
        taps[3] = wb[1];
    }
    // The inputs that earlier kernels write may have left L2 since; one thread per line.
    if (t % 8 == 0) {
        for (int i = 0; i < nb; ++i) {
            prefetch_l2(blocks + size_t(i) * kHidden + ca);
            prefetch_l2(blocks + size_t(i) * kHidden + cb);
        }
        if (mode == kPartialConvResidual) {
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                prefetch_l2(conv_state + size_t(k) * kHidden + ca);
                prefetch_l2(conv_state + size_t(k) * kHidden + cb);
            }
        }
        if (partial_f32 != nullptr) {
            prefetch_l2(partial_f32 + ca);
            prefetch_l2(partial_f32 + cb);
        }
    }
    if (t % 16 == 0) {
        if (partial_bf16 != nullptr) {
            prefetch_l2(partial_bf16 + ca);
            prefetch_l2(partial_bf16 + cb);
        }
        if (conv_hidden != nullptr) {
            prefetch_l2(conv_hidden + ca);
            prefetch_l2(conv_hidden + cb);
        }
    }
    pdl_trigger();
    pdl_wait();

    // Every load before any store, so that none waits for another.
    float s[kStepBlocks][8];
#pragma unroll
    for (int i = 0; i < kStepBlocks; ++i) {
        if (i < nb) {
            load8(blocks + size_t(i) * kHidden, ca, cb, s[i]);
        }
    }
    float p[8];
    // The conv's new state rows, stored at the end.
    float x[8];
    float win[3][8];
    if (mode == kPartialConvResidual) {
        unpack_bf16x4(*reinterpret_cast<const uint2*>(partial_bf16 + ca), x);
        unpack_bf16x4(*reinterpret_cast<const uint2*>(partial_bf16 + cb), x + 4);
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            load8(conv_state + size_t(k) * kHidden, ca, cb, win[k]);
        }
        float hid[8];
        if (conv_hidden != nullptr) {
            unpack_bf16x4(*reinterpret_cast<const uint2*>(conv_hidden + ca), hid);
            unpack_bf16x4(*reinterpret_cast<const uint2*>(conv_hidden + cb), hid + 4);
        }
        const auto* w = reinterpret_cast<const __nv_bfloat16*>(taps);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            // short_conv in conv_step.cuh: taps in order, oldest row first, each product
            // and sum rounded on its own.
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                acc = __fadd_rn(acc, __fmul_rn(round_to_bf16(win[k][e]), __bfloat162float(w[4 * e + k])));
            }
            acc = __fadd_rn(acc, __fmul_rn(round_to_bf16(x[e]), __bfloat162float(w[4 * e + 3])));
            const float y = round_to_bf16(div_short_conv(__fadd_rn(round_to_bf16(acc), x[e])));
            p[e] = conv_hidden != nullptr ? __fadd_rn(hid[e], y) : y;
        }
    } else if (mode == kPartialBf16) {
        unpack_bf16x4(*reinterpret_cast<const uint2*>(partial_bf16 + ca), p);
        unpack_bf16x4(*reinterpret_cast<const uint2*>(partial_bf16 + cb), p + 4);
    } else {
        load8(partial_f32, ca, cb, p);
    }

    // First reduction: every dot product, and the partial's sum of squares.
    float v[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        v[k] = 0.f;
    }
#pragma unroll
    for (int j = 0; j < kStepQueries; ++j) {
        if (j >= nq) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < kStepBlocks; ++i) {
            if (i < nb) {
                float acc = 0.f;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc = fmaf(s[i][e], q[j][e], acc);
                }
                v[4 * i + j] = acc;
            }
        }
        float acc = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            acc = fmaf(p[e], q[j][e], acc);
        }
        v[4 * kStepPartial + j] = acc;
    }
    if (mode != kPartialNormalized) {
        float acc = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            acc = fmaf(p[e], p[e], acc);
        }
        v[kStepSquares] = acc;
    }
    red[warp][lane] = warp_transpose_sum(v);
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kStepWarps; ++w) {
        total += red[w][lane];
    }
    const float squares = __shfl_sync(0xffffffffu, total, kStepSquares);
    const float inv_p = mode == kPartialNormalized ? 1.f : inv_rms_f32(squares, eps);

    // Softmax over the candidates of each query: lane 4 i + j holds candidate i of query j.
    const int ci = lane / 4;
    const int qj = lane % 4;
    const bool valid = qj < nq && (ci < nb || ci == kStepPartial);
    const float logit = valid ? (ci == kStepPartial ? total * inv_p : total) * scale : -INFINITY;
    float mx = logit;
#pragma unroll
    for (int offset = 4; offset < 32; offset *= 2) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, offset));
    }
    const float ex = valid ? fast_exp(logit - mx) : 0.f;
    float z = ex;
#pragma unroll
    for (int offset = 4; offset < 32; offset *= 2) {
        z += __shfl_xor_sync(0xffffffffu, z, offset);
    }
    const float prob = valid ? ex / z : 0.f;

    // Output rows. The partial's scale folds into its weights.
    float o[kStepQueries][8];
#pragma unroll
    for (int j = 0; j < kStepQueries; ++j) {
        float pr[kStepBlocks + 1];
#pragma unroll
        for (int i = 0; i <= kStepBlocks; ++i) {
            pr[i] = __shfl_sync(0xffffffffu, prob, 4 * i + j);
        }
        const float pp = pr[kStepPartial] * inv_p;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float acc = 0.f;
#pragma unroll
            for (int i = 0; i < kStepBlocks; ++i) {
                if (i < nb) {
                    acc = fmaf(pr[i], s[i][e], acc);
                }
            }
            o[j][e] = fmaf(pp, p[e], acc);
        }
        if (j < nq && out != nullptr) {
            store8(out + b * out_stride + size_t(j) * kHidden, ca, cb, o[j]);
        }
    }
    if (mode == kPartialConvResidual && b == 0) {
        store8(residual, ca, cb, p);
        store8(conv_state, ca, cb, win[1]);
        store8(conv_state + kHidden, ca, cb, win[2]);
        store8(conv_state + 2 * kHidden, ca, cb, x);
    }
    if (normed_partial != nullptr && b == 0) {
        float np[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            np[e] = p[e] * inv_p;
        }
        store8(normed_partial, ca, cb, np);
    }

    unsigned normed_rows = 0;
#pragma unroll
    for (int k = 0; k < kDests; ++k) {
        if (dst[k] != nullptr) {
            normed_rows |= 1u << min(k, nq - 1);
        }
    }
    if (normed_rows == 0) {
        return;
    }
#pragma unroll
    for (int j = 0; j < kStepQueries; ++j) {
        if (normed_rows & (1u << j)) {
            float acc = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc = fmaf(o[j][e], o[j][e], acc);
            }
            acc = warp_sum(acc);
            if (lane == 0) {
                red_norm[j][warp] = acc;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kStepQueries; ++j) {
        if (!(normed_rows & (1u << j))) {
            continue;
        }
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < kStepWarps; ++w) {
            sum += red_norm[j][w];
        }
        const float inv = inv_rms_f32(sum, eps);
#pragma unroll
        for (int k = 0; k < kDests; ++k) {
            if (dst[k] == nullptr || min(k, nq - 1) != j) {
                continue;
            }
            float y[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                y[e] = o[j][e] * inv;
            }
            if (weight[k] != nullptr) {
                float w[8];
                unpack_bf16x4(wpack[k][0], w);
                unpack_bf16x4(wpack[k][1], w + 4);
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    y[e] *= w[e];
                }
            }
            __nv_bfloat16* d = dst[k] + b * dst_stride;
            const float ya[4] = {y[0], y[1], y[2], y[3]};
            const float yb[4] = {y[4], y[5], y[6], y[7]};
            store4(d + ca, ya);
            store4(d + cb, yb);
        }
    }
}

namespace {

// Cluster helpers for moe_router_step_2304: one-way stores into another block's shared
// memory that complete on that block's mbarrier, so the receiver waits on its own barrier
// instead of a cluster barrier round trip.
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t cluster_rank() {
#if SIPHON_SM89
    // Both cluster kernels run clusters of kClusterRouterBlocks.
    return blockIdx.x % 16;
#else
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
#endif
}

// Address `a` of this block's shared memory in block `rank` of the cluster.
#if !SIPHON_SM89
__device__ __forceinline__ uint32_t cluster_addr(uint32_t a, uint32_t rank) {
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
    return r;
}
#endif

// The sum of the router dot products of one row, as moe_router_step_rows_2304 adds them:
// virtual thread t = 32 w + lane of a 288-thread block sums its 8 products in order
// (elements 4t.. then 4(t + 288)..), each virtual warp w adds with warp_sum, and the 9
// warp sums add in order. Lane `lane` holds the router row's elements at chunks lane + 32 i
// (4 floats each), the first 9 for the 4t.. runs and the last 9 for the 4(t + 288).. runs.
__device__ __forceinline__ float router_logit(const float4 (&w)[2 * kStepWarps], const __nv_bfloat16* x,
                                              int lane) {
    float total = 0.f;
#pragma unroll
    for (int v = 0; v < kStepWarps; ++v) {
        float xa[4], xb[4];
        unpack_bf16x4(*reinterpret_cast<const uint2*>(x + 4 * (lane + 32 * v)), xa);
        unpack_bf16x4(*reinterpret_cast<const uint2*>(x + 4 * (lane + 32 * v + kStepThreads)), xb);
        const float4 a = w[v];
        const float4 b = w[v + kStepWarps];
        float acc = 0.f;
        acc = fmaf(a.x, xa[0], acc);
        acc = fmaf(a.y, xa[1], acc);
        acc = fmaf(a.z, xa[2], acc);
        acc = fmaf(a.w, xa[3], acc);
        acc = fmaf(b.x, xb[0], acc);
        acc = fmaf(b.y, xb[1], acc);
        acc = fmaf(b.z, xb[2], acc);
        acc = fmaf(b.w, xb[3], acc);
        total += warp_sum(acc);
    }
    return total;
}

// warp_sum's xor butterfly (offsets 1, 2, 4, 8, 16) of 32 values at once, in 31
// shuffles: lane l gets the sum of value bitreverse5(l). Each step adds the same pairs of
// partial sums as warp_sum's step of that offset, so every sum equals warp_sum's bit for
// bit. Step s keeps the lower half of the values in lanes whose bit s is clear and the
// upper half in the others.
__device__ __forceinline__ float warp_butterfly_transpose(float (&v)[32]) {
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int s = 0; s < 5; ++s) {
        const int offset = 1 << s;
        const int half = 16 >> s;
        const bool upper = (lane & offset) != 0;
#pragma unroll
        for (int k = 0; k < half; ++k) {
            const float send = upper ? v[k] : v[k + half];
            const float keep = upper ? v[k + half] : v[k];
            v[k] = keep + __shfl_xor_sync(0xffffffffu, send, offset);
        }
    }
    return v[0];
}

__device__ __forceinline__ int bitreverse5(int x) {
    return int(__brev(unsigned(x)) >> 27);
}

// The router sums of virtual warp `vw` (of router_logit's 9) for R rows of x [R, 2304]
// and the 8 experts whose router rows the block holds: lane `lane` holds expert j's
// elements 4 (lane + 32 vw).. in w[j] and 4 (lane + 32 vw + 288).. in w[8 + j], and loads
// the same elements of each row of x once for all 8 experts. Each (row, expert) pair's
// 8 products add in router_logit's order, and the warp sums of all R * 8 pairs come from
// warp_butterfly_transpose, 32 at a time. Pair i = 8 r + j's sum goes to sums[i].
template <int R>
__device__ __forceinline__ void router_sums_rows(const float4 (&w)[16], const __nv_bfloat16* x, int lane,
                                                 int vw, float* sums) {
    constexpr int kSums = R * 8;
    constexpr int kGroups = (kSums + 31) / 32;
    float xa[R][4], xb[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const __nv_bfloat16* xr = x + size_t(r) * kHidden;
        unpack_bf16x4(*reinterpret_cast<const uint2*>(xr + 4 * (lane + 32 * vw)), xa[r]);
        unpack_bf16x4(*reinterpret_cast<const uint2*>(xr + 4 * (lane + 32 * vw + kStepThreads)), xb[r]);
    }
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
        float v[32];
#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const int i = 32 * g + k;
            v[k] = 0.f;
            if (i < kSums) {
                const int r = i / 8;
                const float4 a = w[i % 8];
                const float4 b = w[8 + i % 8];
                float acc = 0.f;
                acc = fmaf(a.x, xa[r][0], acc);
                acc = fmaf(a.y, xa[r][1], acc);
                acc = fmaf(a.z, xa[r][2], acc);
                acc = fmaf(a.w, xa[r][3], acc);
                acc = fmaf(b.x, xb[r][0], acc);
                acc = fmaf(b.y, xb[r][1], acc);
                acc = fmaf(b.z, xb[r][2], acc);
                acc = fmaf(b.w, xb[r][3], acc);
                v[k] = acc;
            }
        }
        const float sum = warp_butterfly_transpose(v);
        const int i = 32 * g + bitreverse5(lane);
        if (i < kSums) {
            sums[i] = sum;
        }
    }
}

constexpr int kClusterRouterBlocks = 16;
constexpr int kClusterRouterWarps = kExperts / kClusterRouterBlocks;
constexpr int kClusterRouterThreads = 32 * kClusterRouterWarps;
static_assert(kClusterRouterThreads == 2 * kExperts, "two threads per expert rank it");
static_assert(kClusterRouterBlocks == 16, "cluster_rank on sm_89");

#if SIPHON_SM89
// Every block's static shared memory, for up to 4 clusters (32 rows).
__device__ sm89::Mailboxes<4 * kClusterRouterBlocks, 8192> router_mail;
#endif

}  // namespace

// The router for one BF16 row x that is already normalized: logits = F32(x) @ router.T,
// then the top-8 routing of moe_router_2304, as one cluster of 16 blocks of 8 warps.
// Warp w of block r takes expert 8 r + w: its lanes hold the whole router row, loaded
// before pdl_wait, and sum the logit exactly as moe_router_step_rows_2304 does. Each
// block sends its 8 logits to block 0 with two st.async stores, and block 0 routes: each
// expert's rank among the biased scores, ties to the lower index, counted by two threads
// over half the experts each, as route_top8 does.
extern "C" __global__ void SIPHON_CLUSTER_DIMS(kClusterRouterBlocks) __launch_bounds__(kClusterRouterThreads)
moe_router_step_2304(const __nv_bfloat16* x, const float* __restrict__ router, const float* __restrict__ bias,
                     float* logits, int32_t* indices, float* weights, float route_scale) {
    __shared__ __align__(16) float all[kExperts];
    __shared__ __align__(16) float mine[kClusterRouterWarps];
    __shared__ __align__(8) uint64_t arrived;
    __shared__ float biased[kExperts];
    __shared__ float score[kExperts];
    __shared__ int ranks[kClusterRouterThreads];
    __shared__ float top[kTopK];
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const uint32_t rank = cluster_rank();
    const uint32_t barrier = smem_addr(&arrived);
#if !SIPHON_SM89
    if (t == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(barrier) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        if (rank == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(barrier),
                         "r"(uint32_t(kExperts * sizeof(float)))
                         : "memory");
        }
    }
    // Every block arrives now and waits before its first remote store, so block 0's
    // barrier is ready by then.
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
#endif

    const int e = int(rank) * kClusterRouterWarps + warp;
    const auto* row = reinterpret_cast<const float4*>(router + size_t(e) * kHidden);
    float4 w[2 * kStepWarps];
#pragma unroll
    for (int i = 0; i < 2 * kStepWarps; ++i) {
        w[i] = row[lane + 32 * i];
    }
    const float b = rank == 0 && t < kExperts ? bias[t] : 0.f;
    pdl_trigger();
    pdl_wait();

    const float logit = router_logit(w, x, lane);
    if (lane == 0) {
        mine[warp] = logit;
    }
    __syncthreads();
#if SIPHON_SM89
    if (t < kClusterRouterWarps / 4) {
        const float4 v = reinterpret_cast<const float4*>(mine)[t];
        router_mail.send<4>(kClusterRouterBlocks, 0, smem_addr(all + rank * kClusterRouterWarps + 4 * t), barrier,
                            {__float_as_uint(v.x), __float_as_uint(v.y), __float_as_uint(v.z), __float_as_uint(v.w)});
    }
    if (rank != 0) {
        return;
    }
    router_mail.wait<true>(kClusterRouterBlocks, barrier, all, sizeof(all));
#else
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (t < kClusterRouterWarps / 4) {
        const float4 v = reinterpret_cast<const float4*>(mine)[t];
        const uint32_t dst = cluster_addr(smem_addr(all + rank * kClusterRouterWarps + 4 * t), 0);
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
                     "r"(__float_as_uint(v.x)), "r"(__float_as_uint(v.y)), "r"(__float_as_uint(v.z)),
                     "r"(__float_as_uint(v.w)), "r"(cluster_addr(barrier, 0))
                     : "memory");
    }
    if (rank != 0) {
        return;
    }
    asm volatile(
        "{\n .reg .pred p;\n wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
        " @!p bra wait;\n}" ::"r"(barrier)
        : "memory");
#endif

    // route_top8 on the gathered logits.
    if (t < kExperts) {
        const float l = all[t];
        logits[t] = l;
        // Metal's form of the sigmoid: exp of |x| cannot overflow to inf/inf.
        const float y = 1.f / (1.f + expf(fabsf(l)));
        score[t] = l < 0 ? y : 1.f - y;
        biased[t] = score[t] + b;
    }
    __syncthreads();
    {
        const int me = t % kExperts;
        const int half = t / kExperts;
        const float my = biased[me];
        int r = 0;
#pragma unroll 8
        for (int other = half * kExperts / 2; other < (half + 1) * kExperts / 2; ++other) {
            const float o = biased[other];
            r += o > my || (o == my && other < me);
        }
        ranks[t] = r;
    }
    __syncthreads();
    if (t < kExperts) {
        const int r = ranks[t] + ranks[t + kExperts];
        if (r < kTopK) {
            indices[r] = t;
            top[r] = score[t];
        }
    }
    __syncthreads();
    if (t < kTopK) {
        float total = 0.f;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            total = top[k] + total;
        }
        weights[t] = top[t] / (total + kScoreEps) * route_scale;
    }
}

// moe_router_step_2304 for 1 to kStepRouterRows rows of a step, such as a speculative
// verify: each row's logits sum exactly as moe_router_step_2304 sums one row's, so each
// row's logits, indices and weights equal a one-row call's bit for bit, for every row
// count. The same cluster of 16 blocks, block b holding the router rows of experts
// 8 b .. 8 b + 7, but with 9 warps: warp v takes virtual warp v of router_logit for all 8
// experts and every row (router_sums_rows), so each element of x loads once per block.
// The 9 warp sums of each (row, expert) pair add in router_logit's order, each block sends
// its 8 logits of row r to block r, and block r routes row r as moe_router_step_2304's
// block 0 routes its row. Above 8 rows, cluster y (blockIdx.y) takes rows 8 y .. 8 y + 7
// the same way.
constexpr int kStepRouterRows = 8;
static_assert(kStepRouterRows <= kClusterRouterBlocks, "a block per row routes");
constexpr int kRowsRouterThreads = kStepThreads;
static_assert(kRowsRouterThreads >= 2 * kExperts, "two threads per expert rank it");

extern "C" __global__ void SIPHON_CLUSTER_DIMS(kClusterRouterBlocks) __launch_bounds__(kRowsRouterThreads)
moe_router_step_rows_2304(const __nv_bfloat16* x, int rows, const float* __restrict__ router,
                          const float* __restrict__ bias, float* logits, int32_t* indices,
                          float* weights, float route_scale) {
    constexpr int kPairs = kStepRouterRows * kClusterRouterWarps;
    __shared__ __align__(16) float all[kExperts];
    __shared__ __align__(16) float mine[kStepRouterRows][kClusterRouterWarps];
    __shared__ float sums[kStepWarps][kPairs];
    __shared__ __align__(8) uint64_t arrived;
    __shared__ float biased[kExperts];
    __shared__ float score[kExperts];
    __shared__ int ranks[2 * kExperts];
    __shared__ float top[kTopK];
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const uint32_t rank = cluster_rank();
    const uint32_t barrier = smem_addr(&arrived);
    // This cluster's rows.
    {
        const int row0 = int(blockIdx.y) * kStepRouterRows;
        x += size_t(row0) * kHidden;
        logits += size_t(row0) * kExperts;
        indices += size_t(row0) * kTopK;
        weights += size_t(row0) * kTopK;
        rows = min(rows - row0, kStepRouterRows);
    }
    // Block `rank` routes row `rank`, if there is one.
    const bool routes = int(rank) < rows;
#if !SIPHON_SM89
    if (t == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(barrier) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        if (routes) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(barrier),
                         "r"(uint32_t(kExperts * sizeof(float)))
                         : "memory");
        }
    }
    // Every block arrives now and waits before its first remote store, so the routing
    // blocks' barriers are ready by then.
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
#endif

    // w[j], w[8 + j]: expert 8 rank + j's elements of virtual warp `warp`.
    float4 w[2 * kClusterRouterWarps];
#pragma unroll
    for (int j = 0; j < kClusterRouterWarps; ++j) {
        const auto* row =
            reinterpret_cast<const float4*>(router + size_t(int(rank) * kClusterRouterWarps + j) * kHidden);
        w[j] = row[lane + 32 * warp];
        w[kClusterRouterWarps + j] = row[lane + 32 * warp + kStepThreads];
    }
    const float b = routes && t < kExperts ? bias[t] : 0.f;
    pdl_trigger();
    pdl_wait();

    // A ladder of comparisons rather than a switch, whose jump table scripts/pdl_audit.py
    // cannot follow.
    float* my_sums = sums[warp];
    if (rows <= 4) {
        if (rows <= 2) {
            if (rows <= 1) {
                router_sums_rows<1>(w, x, lane, warp, my_sums);
            } else {
                router_sums_rows<2>(w, x, lane, warp, my_sums);
            }
        } else if (rows <= 3) {
            router_sums_rows<3>(w, x, lane, warp, my_sums);
        } else {
            router_sums_rows<4>(w, x, lane, warp, my_sums);
        }
    } else if (rows <= 6) {
        if (rows <= 5) {
            router_sums_rows<5>(w, x, lane, warp, my_sums);
        } else {
            router_sums_rows<6>(w, x, lane, warp, my_sums);
        }
    } else if (rows <= 7) {
        router_sums_rows<7>(w, x, lane, warp, my_sums);
    } else {
        router_sums_rows<8>(w, x, lane, warp, my_sums);
    }
    __syncthreads();
    if (t < kClusterRouterWarps * rows) {
        float total = 0.f;
#pragma unroll
        for (int v = 0; v < kStepWarps; ++v) {
            total += sums[v][t];
        }
        mine[t / kClusterRouterWarps][t % kClusterRouterWarps] = total;
    }
    __syncthreads();
#if SIPHON_SM89
    if (t < 2 * rows) {
        const int r = t / 2;
        const int q = t % 2;
        const float4 v = reinterpret_cast<const float4*>(mine[r])[q];
        router_mail.send<4>(kClusterRouterBlocks, r, smem_addr(all + rank * kClusterRouterWarps + 4 * q), barrier,
                            {__float_as_uint(v.x), __float_as_uint(v.y), __float_as_uint(v.z), __float_as_uint(v.w)});
    }
    if (!routes) {
        return;
    }
    router_mail.wait<true>(kClusterRouterBlocks, barrier, all, sizeof(all));
#else
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    // Thread 2 r + q sends this block's logits 4 q .. 4 q + 3 of row r to block r.
    if (t < 2 * rows) {
        const int r = t / 2;
        const int q = t % 2;
        const float4 v = reinterpret_cast<const float4*>(mine[r])[q];
        const uint32_t dst = cluster_addr(smem_addr(all + rank * kClusterRouterWarps + 4 * q), uint32_t(r));
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(dst),
                     "r"(__float_as_uint(v.x)), "r"(__float_as_uint(v.y)), "r"(__float_as_uint(v.z)),
                     "r"(__float_as_uint(v.w)), "r"(cluster_addr(barrier, uint32_t(r)))
                     : "memory");
    }
    if (!routes) {
        return;
    }
    asm volatile(
        "{\n .reg .pred p;\n wait: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], 0;\n"
        " @!p bra wait;\n}" ::"r"(barrier)
        : "memory");
#endif

    // route_top8 on row `rank`'s gathered logits.
    const int r = int(rank);
    if (t < kExperts) {
        const float l = all[t];
        logits[r * kExperts + t] = l;
        // Metal's form of the sigmoid: exp of |x| cannot overflow to inf/inf.
        const float y = 1.f / (1.f + expf(fabsf(l)));
        score[t] = l < 0 ? y : 1.f - y;
        biased[t] = score[t] + b;
    }
    __syncthreads();
    if (t < 2 * kExperts) {
        const int me = t % kExperts;
        const int half = t / kExperts;
        const float my = biased[me];
        int rk = 0;
#pragma unroll 8
        for (int other = half * kExperts / 2; other < (half + 1) * kExperts / 2; ++other) {
            const float o = biased[other];
            rk += o > my || (o == my && other < me);
        }
        ranks[t] = rk;
    }
    __syncthreads();
    if (t < kExperts) {
        const int rk = ranks[t] + ranks[t + kExperts];
        if (rk < kTopK) {
            indices[r * kTopK + rk] = t;
            top[rk] = score[t];
        }
    }
    __syncthreads();
    if (t < kTopK) {
        float total = 0.f;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            total = top[k] + total;
        }
        weights[r * kTopK + t] = top[t] / (total + kScoreEps) * route_scale;
    }
}

// ---------------------------------------------------------------------------
// Depth attention for the rows of a chunk, in the order of depth_attention_step_2304
// rather than MLX's: thread t holds elements 4t.. and 4(t + 288).. of every row, and each
// sum runs per thread, then across each warp, then across the 9 warps in order. Results
// differ from depth_attention_fused_2304 by F32 rounding.
//
// One kernel per block count NB and query count NQ, so that each holds only the registers
// it needs. A block loads the queries once, then takes rows blockIdx.x,
// blockIdx.x + gridDim.x, and so on.
//
// Pointers to data that earlier kernels write are not __restrict__ here and in the router
// below: a const __restrict__ pointer lets nvcc load through the non-coherent path, and
// ptxas may move such loads above pdl_wait, before the data is written.
// ---------------------------------------------------------------------------

namespace {

// Sums v[0 .. V) across the warp, V a power of two: lane l gets the sum of v[l % V], in
// log2(V) + (5 - log2(V)) steps of xor shuffles.
template <int V>
__device__ __forceinline__ float warp_transpose_sum_n(float (&v)[V]) {
    static_assert(V >= 1 && V <= 32 && (V & (V - 1)) == 0, "V is a power of two up to 32");
    const int lane = threadIdx.x % 32;
#pragma unroll
    for (int half = V / 2; half >= 1; half /= 2) {
        const bool upper = (lane & half) != 0;
#pragma unroll
        for (int k = 0; k < half; ++k) {
            const float send = upper ? v[k] : v[k + half];
            const float keep = upper ? v[k + half] : v[k];
            v[k] = keep + __shfl_xor_sync(0xffffffffu, send, half);
        }
    }
    float s = v[0];
#pragma unroll
    for (int offset = V; offset < 32; offset *= 2) {
        s += __shfl_xor_sync(0xffffffffu, s, offset);
    }
    return s;
}

__host__ __device__ constexpr int pow2_at_least(int n) {
    int p = 1;
    while (p < n) {
        p *= 2;
    }
    return p;
}

__device__ __forceinline__ void load8_bf16(const __nv_bfloat16* p, int ca, int cb, float (&v)[8]) {
    unpack_bf16x4(*reinterpret_cast<const uint2*>(p + ca), v);
    unpack_bf16x4(*reinterpret_cast<const uint2*>(p + cb), v + 4);
}

template <int NB, int NQ>
__device__ __forceinline__ void
depth_attention_rows(const float* blocks, int block_stride, const float* partial_f32,
                     const __nv_bfloat16* partial_bf16, int mode, const __nv_bfloat16* __restrict__ conv_weight,
                     const float* conv_state, const __nv_bfloat16* conv_hidden, float* residual,
                     float* normed_partial, const float* __restrict__ queries, float* out, float* probs,
                     __nv_bfloat16* dst0, __nv_bfloat16* dst1, __nv_bfloat16* dst2, __nv_bfloat16* dst3,
                     int dst_stride,
                     const __nv_bfloat16* __restrict__ weight0, const __nv_bfloat16* __restrict__ weight1,
                     const __nv_bfloat16* __restrict__ weight2, const __nv_bfloat16* __restrict__ weight3,
                     float eps, float scale, int rows, int probs_batch, int dst_batch) {
    constexpr int kCandidates = NB + 1;
    constexpr int kDots = kCandidates * NQ;
    // The dots, then the partial's sum of squares.
    constexpr int kValues = pow2_at_least(kDots + 1);
    static_assert(kValues <= 32, "the first reduction fits a warp");
    __shared__ float red[2][kStepWarps][32];
    __shared__ float red_norm[2][NQ][kStepWarps];
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int ca = 4 * t;
    const int cb = 4 * (t + kStepThreads);
    // Batch y (blockIdx.y) has its own queries, weights and destinations.
    const size_t batch = blockIdx.y;
    queries += batch * NQ * kHidden;
    if (probs != nullptr) {
        probs += batch * probs_batch;
    }
    __nv_bfloat16* const dst[kDests] = {
        dst0 != nullptr ? dst0 + batch * dst_batch : nullptr, dst1 != nullptr ? dst1 + batch * dst_batch : nullptr,
        dst2 != nullptr ? dst2 + batch * dst_batch : nullptr, dst3 != nullptr ? dst3 + batch * dst_batch : nullptr};
    const __nv_bfloat16* const weight[kDests] = {weight0, weight1, weight2, weight3};
    unsigned normed_rows = 0;
#pragma unroll
    for (int k = 0; k < kDests; ++k) {
        if (dst[k] != nullptr) {
            normed_rows |= 1u << min(k, NQ - 1);
        }
    }

    // Weights first: they do not depend on earlier kernels.
    float q[NQ][8];
#pragma unroll
    for (int j = 0; j < NQ; ++j) {
        load8(queries + size_t(j) * kHidden, ca, cb, q[j]);
    }
    pdl_trigger();
    // The conv taps of this thread's 8 channels, 4 per channel, and the conv state in L2:
    // both were last read a visit of this layer ago.
    uint4 taps[4] = {};
    if (mode == kPartialConvResidual) {
        const uint4* wa = reinterpret_cast<const uint4*>(conv_weight + 4 * ca);
        const uint4* wb = reinterpret_cast<const uint4*>(conv_weight + 4 * cb);
        taps[0] = wa[0];
        taps[1] = wa[1];
        taps[2] = wb[0];
        taps[3] = wb[1];
        if (t % 8 == 0) {
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                prefetch_l2(conv_state + size_t(k) * kHidden + ca);
                prefetch_l2(conv_state + size_t(k) * kHidden + cb);
            }
        }
    }
    // A branch keeps ptxas from placing the wait before the tap loads (see pdl.cuh).
    if (rows > 0) {
        pdl_wait();
    }

    int parity = 0;
    for (int row = blockIdx.x; row < rows; row += gridDim.x, parity ^= 1) {
        const size_t at = size_t(row) * kHidden;
        float s[NB][8];
#pragma unroll
        for (int i = 0; i < NB; ++i) {
            load8(blocks + size_t(i) * block_stride + at, ca, cb, s[i]);
        }
        float p[8];
        if (mode == kPartialConvResidual) {
            // short_conv_residual_rows_2304 for this row: the window is rows row - 3 .. row
            // of the attention output, reaching into the conv state before the chunk.
            float win[4][8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int r = row - 3 + k;
                if (r >= 0) {
                    load8_bf16(partial_bf16 + size_t(r) * kHidden, ca, cb, win[k]);
                } else {
                    load8(conv_state + size_t(3 + r) * kHidden, ca, cb, win[k]);
                }
            }
            float hid[8];
            if (conv_hidden != nullptr) {
                load8_bf16(conv_hidden + at, ca, cb, hid);
            }
            const auto* w = reinterpret_cast<const __nv_bfloat16*>(taps);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float acc = 0.f;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    acc = __fadd_rn(acc, __fmul_rn(round_to_bf16(win[k][e]), __bfloat162float(w[4 * e + k])));
                }
                const float y = round_to_bf16(div_short_conv(__fadd_rn(round_to_bf16(acc), win[3][e])));
                p[e] = conv_hidden != nullptr ? __fadd_rn(hid[e], y) : y;
            }
            store8(residual + at, ca, cb, p);
        } else if (mode == kPartialBf16) {
            load8_bf16(partial_bf16 + at, ca, cb, p);
        } else {
            load8(partial_f32 + at, ca, cb, p);
        }

        // First reduction: every dot product (candidate i, query j at i * NQ + j, the
        // partial last), and the partial's sum of squares.
        float v[kValues];
#pragma unroll
        for (int k = 0; k < kValues; ++k) {
            v[k] = 0.f;
        }
#pragma unroll
        for (int j = 0; j < NQ; ++j) {
#pragma unroll
            for (int i = 0; i < NB; ++i) {
                float acc = 0.f;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc = fmaf(s[i][e], q[j][e], acc);
                }
                v[i * NQ + j] = acc;
            }
            float acc = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc = fmaf(p[e], q[j][e], acc);
            }
            v[NB * NQ + j] = acc;
        }
        if (mode != kPartialNormalized) {
            float acc = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc = fmaf(p[e], p[e], acc);
            }
            v[kDots] = acc;
        }
        red[parity][warp][lane] = warp_transpose_sum_n(v);
        __syncthreads();
        float total = 0.f;
#pragma unroll
        for (int w = 0; w < kStepWarps; ++w) {
            total += red[parity][w][lane];
        }
        const float squares = __shfl_sync(0xffffffffu, total, kDots);
        const float inv_p = mode == kPartialNormalized ? 1.f : inv_rms_f32(squares, eps);

        // Softmax over the candidates of each query: lane k < kDots holds candidate k / NQ
        // of query k % NQ.
        const bool valid = lane < kDots;
        const int qj = lane % NQ;
        const float logit = valid ? (lane >= NB * NQ ? total * inv_p : total) * scale : -INFINITY;
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < kCandidates; ++i) {
            mx = fmaxf(mx, __shfl_sync(0xffffffffu, logit, i * NQ + qj));
        }
        const float ex = valid ? fast_exp(logit - mx) : 0.f;
        float z = 0.f;
#pragma unroll
        for (int i = 0; i < kCandidates; ++i) {
            z += __shfl_sync(0xffffffffu, ex, i * NQ + qj);
        }
        const float prob = valid ? ex / z : 0.f;

        // Output rows. The partial's scale folds into its weight.
        float o[NQ][8];
#pragma unroll
        for (int j = 0; j < NQ; ++j) {
            float pr[kCandidates];
#pragma unroll
            for (int i = 0; i < kCandidates; ++i) {
                pr[i] = __shfl_sync(0xffffffffu, prob, i * NQ + j);
            }
            const float pp = pr[NB] * inv_p;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                float acc = 0.f;
#pragma unroll
                for (int i = 0; i < NB; ++i) {
                    acc = fmaf(pr[i], s[i][e], acc);
                }
                o[j][e] = fmaf(pp, p[e], acc);
            }
            if (out != nullptr) {
                store8(out + (size_t(row) * NQ + j) * kHidden, ca, cb, o[j]);
            }
            if (probs != nullptr && t == 0) {
                float* pj = probs + (size_t(row) * NQ + j) * kCandidates;
#pragma unroll
                for (int i = 0; i < NB; ++i) {
                    pj[i] = pr[i];
                }
                pj[NB] = pp;
            }
        }
        if (normed_partial != nullptr) {
            float np[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                np[e] = p[e] * inv_p;
            }
            store8(normed_partial + at, ca, cb, np);
        }
        if (normed_rows == 0) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < NQ; ++j) {
            if (normed_rows & (1u << j)) {
                float acc = 0.f;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc = fmaf(o[j][e], o[j][e], acc);
                }
                acc = warp_sum(acc);
                if (lane == 0) {
                    red_norm[parity][j][warp] = acc;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NQ; ++j) {
            if (!(normed_rows & (1u << j))) {
                continue;
            }
            float sum = 0.f;
#pragma unroll
            for (int w = 0; w < kStepWarps; ++w) {
                sum += red_norm[parity][j][w];
            }
            const float inv = inv_rms_f32(sum, eps);
#pragma unroll
            for (int k = 0; k < kDests; ++k) {
                if (dst[k] == nullptr || min(k, NQ - 1) != j) {
                    continue;
                }
                float y[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    y[e] = o[j][e] * inv;
                }
                if (weight[k] != nullptr) {
                    float w[8];
                    load8_bf16(weight[k], ca, cb, w);
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        y[e] *= w[e];
                    }
                }
                __nv_bfloat16* d = dst[k] + size_t(row) * dst_stride;
                const float ya[4] = {y[0], y[1], y[2], y[3]};
                const float yb[4] = {y[4], y[5], y[6], y[7]};
                store4(d + ca, ya);
                store4(d + cb, yb);
            }
        }
    }
}

}  // namespace

// depth_attention_rows_2304_<NB>_<NQ>: NB block states ([NB] runs of [rows, 2304],
// block_stride apart) and the partial attend for NQ queries, for each of `rows` rows.
// `mode` picks the partial as in depth_attention_step_2304; with kPartialConvResidual,
// partial_bf16 is the attention output [rows, 2304], conv_state ([3, 2304] F32, only
// read) holds the rows before the chunk, and each row's residual goes to `residual`.
// out is [rows, NQ, 2304]. probs, if not null, receives each output's weights
// ([rows, NQ, NB + 1]): out = fmaf(w_NB, partial, sum over i < NB of w_i * block_i, the
// products chained by fmaf in order from 0), with the partial as given, for kernels that
// rebuild an output from its candidates. Destination k receives output row
// min(k, NQ - 1) after an RMS norm, times weight_k if that is not null, as BF16 rows
// dst_stride apart.
//
// The grid's y dimension runs a batch of such attentions over the same candidates: batch y
// reads queries + y * NQ * 2304 and writes probs + y * probs_batch and dst_k + y * dst_batch.
// out, normed_partial and residual take one batch only.
#define DEPTH_ATTENTION_ROWS(NB, NQ)                                                                            \
    extern "C" __global__ void __launch_bounds__(kStepThreads) depth_attention_rows_2304_##NB##_##NQ(        \
        const float* blocks, int block_stride, const float* partial_f32,                                       \
        const __nv_bfloat16* partial_bf16, int mode, const __nv_bfloat16* __restrict__ conv_weight,            \
        const float* conv_state, const __nv_bfloat16* conv_hidden, float* residual,                             \
        float* normed_partial, const float* __restrict__ queries, float* out, float* probs,                     \
        __nv_bfloat16* dst0, __nv_bfloat16* dst1, __nv_bfloat16* dst2, __nv_bfloat16* dst3, int dst_stride,    \
        const __nv_bfloat16* __restrict__ weight0, const __nv_bfloat16* __restrict__ weight1,                  \
        const __nv_bfloat16* __restrict__ weight2, const __nv_bfloat16* __restrict__ weight3, float eps,       \
        float scale, int rows, int probs_batch, int dst_batch) {                                               \
        depth_attention_rows<NB, NQ>(blocks, block_stride, partial_f32, partial_bf16, mode, conv_weight,        \
                                     conv_state, conv_hidden, residual, normed_partial, queries, out, probs,    \
                                     dst0, dst1, dst2, dst3, dst_stride, weight0, weight1, weight2, weight3,    \
                                     eps, scale, rows, probs_batch, dst_batch);                                \
    }

#define DEPTH_ATTENTION_ROWS_Q(NQ) \
    DEPTH_ATTENTION_ROWS(1, NQ)    \
    DEPTH_ATTENTION_ROWS(2, NQ)    \
    DEPTH_ATTENTION_ROWS(3, NQ)    \
    DEPTH_ATTENTION_ROWS(4, NQ)    \
    DEPTH_ATTENTION_ROWS(5, NQ)

DEPTH_ATTENTION_ROWS_Q(1)
DEPTH_ATTENTION_ROWS_Q(3)
DEPTH_ATTENTION_ROWS_Q(4)
