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
extern "C" __global__ void __launch_bounds__(kDepthThreads)
depth_attention_fused_2304(const float* __restrict__ blocks, int nb, const float* partial_f32,
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
            s[i][m] = i < nb ? blocks[size_t(i) * kHidden + d] : (i == nb ? pd : 0.f);
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

namespace {

constexpr int kRouterThreads = kDepthThreads;
constexpr int kExpertsPerBlock = 2;
constexpr int kRouterBlocks = kExperts / kExpertsPerBlock;
// router_logits_2304's layout: thread t < 256 sums columns 4t + 1024 c + 0..3.
constexpr int kGemvChunks = (kHidden + kLogitsThreads * 4 - 1) / (kLogitsThreads * 4);

}  // namespace

// The MoE input norm and router for one row: x_out = bf16(rms_norm(x) * norm_weight),
// logits = F32(x_out) @ router.T, then the top-8 routing of expert_routing_128_top8.
// The norm and the logits sum as rms_norm_f32_2304 and router_logits_2304 do; the
// sigmoid uses CUDA's expf.
//
// Each block computes the norm itself and the logits of two experts. The last block
// to finish, found with `counter`, routes. `counter` must start at 0; the last block
// resets it, so the kernel can run again from a graph.
extern "C" __global__ void __launch_bounds__(kRouterThreads)
moe_router_2304(const float* __restrict__ x, const __nv_bfloat16* __restrict__ norm_weight, float eps,
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
    if (threadIdx.x == 0) {
        *counter = 0;
    }
}
