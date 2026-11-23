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
//
// `block_stride` is the distance between block states: 2304 for one row, or a whole
// chunk's rows for depth_attention_fused_rows_2304.
namespace {

__device__ __forceinline__ void
depth_attention_fused_row(const float* __restrict__ blocks, int block_stride, int nb, const float* partial_f32,
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
depth_attention_fused_2304(const float* __restrict__ blocks, int nb, const float* partial_f32,
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

// depth_attention_fused_2304 for each row (blockIdx.x) of a chunk. Block state i of row
// r is at blocks + i * block_stride + r * 2304. The partial, normed_partial and out rows
// are consecutive ([rows, 2304] and [rows, nq, 2304]); the destination rows are
// dst_stride apart, so a destination can be one half of a [rows, 2 * 2304] buffer.
extern "C" __global__ void __launch_bounds__(kDepthThreads)
depth_attention_fused_rows_2304(const float* __restrict__ blocks, int block_stride, int nb,
                                const float* partial_f32, const __nv_bfloat16* partial_bf16,
                                int normalize_partial, float* normed_partial, const float* __restrict__ queries,
                                int nq, float* out, __nv_bfloat16* dst0, __nv_bfloat16* dst1,
                                __nv_bfloat16* dst2, __nv_bfloat16* dst3, int dst_stride,
                                const __nv_bfloat16* weight0, const __nv_bfloat16* weight1,
                                const __nv_bfloat16* weight2, const __nv_bfloat16* weight3, float eps,
                                float scale) {
    const size_t row = blockIdx.x;
    const size_t at = row * kHidden;
    const size_t dst_at = row * size_t(dst_stride);
    const auto offset = [&](__nv_bfloat16* p) { return p != nullptr ? p + dst_at : nullptr; };
    depth_attention_fused_row(blocks + at, block_stride, nb, partial_f32 != nullptr ? partial_f32 + at : nullptr,
                              partial_bf16 != nullptr ? partial_bf16 + at : nullptr, normalize_partial,
                              normed_partial != nullptr ? normed_partial + at : nullptr, queries, nq,
                              out != nullptr ? out + row * size_t(nq) * kHidden : nullptr, offset(dst0),
                              offset(dst1), offset(dst2), offset(dst3), weight0, weight1, weight2, weight3, eps,
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
moe_router_row(const float* __restrict__ x, const __nv_bfloat16* __restrict__ norm_weight, float eps,
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
moe_router_2304(const float* __restrict__ x, const __nv_bfloat16* __restrict__ norm_weight, float eps,
                const float* __restrict__ router, const float* __restrict__ bias,
                __nv_bfloat16* x_out, float* logits, int32_t* indices, float* weights,
                unsigned* counter, float route_scale) {
    moe_router_row(x, norm_weight, eps, router, bias, x_out, logits, indices, weights, counter, route_scale);
}

// ---------------------------------------------------------------------------
// moe_router_2304 for the rows of a chunk, in three kernels: the norm, the logits as a
// tiled F32 matrix product split along the inner dimension, and the routing. Every sum
// keeps moe_router_2304's order, so each row's results equal a single-row call's bit
// for bit.
//
// router_logits_2304's order for one logit: thread t < 256 of the block (warp w = t / 32,
// lane l = t % 32) chains fmaf over its columns 4t + 1024c + 0..3 for c = 0, 1 and, if
// t < 64, 2. Each warp then adds its 32 lane sums with a shuffle-down tree, pairing lanes
// 16 apart, then 8, 4, 2 and 1. The 8 warp sums are added in order.
//
// Visiting the lanes in bit-reversed order, lane = rev5(j) for j = 0..31, makes the
// tree's pairs adjacent: the tree adds steps 2i and 2i + 1, then adjacent pairs, and so
// on. Steps j = 8g .. 8g + 7 (g = 0..3) are the lanes r + 4k (k = 0..7) for r = rev2(g),
// and form one subtree T(w, g); the warp's sum is (T0 + T1) + (T2 + T3).
//
// moe_router_logits_rows_2304 computes one subtree for a tile of rows and experts, and
// moe_router_route_rows_2304 adds the 32 subtrees of each logit.
// ---------------------------------------------------------------------------

namespace {

constexpr int kSplits = kLogitsThreads / 32;
constexpr int kLanes = 32;
constexpr int kQuarters = 4;
constexpr int kSubtrees = kSplits * kQuarters;
constexpr int kSubtreeLanes = kLanes / kQuarters;
constexpr int kMaxChunks = 3;
constexpr int kTileRows = 64;
constexpr int kTileExperts = 32;
constexpr int kTileThreads = 128;
// Each thread holds 4 rows by 4 experts: the row pairs (2 p, 2 p + 1) for p = tr + 16 i
// and the experts te + 8 i. One 16-byte shared-memory read gives a row pair's x. A warp
// spans 8 values of tr and 4 of te, so its reads stay within 128 and 64 bytes.
constexpr int kThreadPairs = 2;
constexpr int kThreadRows = 2 * kThreadPairs;
constexpr int kThreadExperts = 4;
constexpr int kPairGroups = kTileRows / kThreadRows;
constexpr int kExpertGroups = kTileExperts / kThreadExperts;
static_assert(kPairGroups * kExpertGroups == kTileThreads, "threads cover the tile");
// Rows per block of moe_router_norm_rows_2304: one per warp.
constexpr int kNormRows = 8;
constexpr int kRuns = kHidden / 4;

// Columns of chunk c exist for warps w < 2 only (t < 64).
__host__ __device__ constexpr int chunks_of(int w) { return w < 2 ? 3 : 2; }

// The permuted copy of the normed rows that moe_router_norm_rows_2304 writes for the
// logits kernel holds runs of 4 BF16 values. Run (w, g, k, c) of every row comes
// before the next run, so that a subtree's runs for consecutive rows are contiguous:
// run index (subtree_base(w, g) + k * chunks_of(w) + c) * rows + row.
__device__ __forceinline__ int subtree_base(int w, int g) {
    const int before = w < 2 ? w * kQuarters * kSubtreeLanes * 3
                             : 2 * kQuarters * kSubtreeLanes * 3 + (w - 2) * kQuarters * kSubtreeLanes * 2;
    return before + g * kSubtreeLanes * chunks_of(w);
}

__device__ __forceinline__ int rev2(int g) { return ((g & 1) << 1) | (g >> 1); }

__device__ __forceinline__ int rev3(int m) { return ((m & 1) << 2) | (m & 2) | (m >> 2); }

// The run of the permuted copy that holds columns col .. col + 3 (col a multiple of 4).
__device__ __forceinline__ int permuted_run(int col) {
    const int t = col % 1024 / 4;
    const int c = col / 1024;
    const int w = t / kLanes;
    const int lane = t % kLanes;
    const int g = rev2(lane % kQuarters);
    const int k = lane / kQuarters;
    return subtree_base(w, g) + k * chunks_of(w) + c;
}

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <int Bytes>
__device__ __forceinline__ void cp_async(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(dst), "l"(src), "n"(Bytes),
                 "r"(valid ? Bytes : 0));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }

__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }

__device__ __forceinline__ void bf16x4_to_f32(uint2 p, float (&v)[4]) {
    v[0] = __uint_as_float(p.x << 16);
    v[1] = __uint_as_float(p.x & 0xffff0000u);
    v[2] = __uint_as_float(p.y << 16);
    v[3] = __uint_as_float(p.y & 0xffff0000u);
}

template <int Chunks>
__device__ __forceinline__ void router_subtree_tile(const uint2* __restrict__ xp, const float* __restrict__ router,
                                                    float* partials, int rows, int w, int g, uint2* xs,
                                                    float4* ws) {
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int tr = warp / 2 * 8 + lane / 4;
    const int te = warp % 2 * 4 + lane % 4;
    const int r0 = blockIdx.x * kTileRows;
    const int e0 = blockIdx.y * kTileExperts;
    const int r = rev2(g);

    for (int idx = tid; idx < kSubtreeLanes * Chunks * kTileExperts; idx += kTileThreads) {
        const int e = idx % kTileExperts;
        const int kc = idx / kTileExperts;
        const int t = w * kLanes + r + kQuarters * (kc / Chunks);
        const float* src = router + size_t(e0 + e) * kHidden + 4 * t + 1024 * (kc % Chunks);
        cp_async<16>(smem_addr(&ws[kc * kTileExperts + e]), src, true);
    }
    pdl_trigger();
    pdl_wait();
    const uint2* xsub = xp + size_t(subtree_base(w, g)) * rows;
    for (int idx = tid; idx < kSubtreeLanes * Chunks * kTileRows; idx += kTileThreads) {
        const int row = idx % kTileRows;
        const int kc = idx / kTileRows;
        const bool valid = r0 + row < rows;
        const uint2* src = xsub + size_t(kc) * rows + (valid ? r0 + row : 0);
        cp_async<8>(smem_addr(&xs[kc * kTileRows + row]), src, valid);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // level[b] holds the sum of the last 2^b steps that closed a subtree of that size.
    float level[3][kThreadRows][kThreadExperts];
#pragma unroll 1
    for (int m = 0; m < kSubtreeLanes; ++m) {
        const int k = rev3(m);
        float acc[kThreadRows][kThreadExperts] = {};
#pragma unroll
        for (int c = 0; c < Chunks; ++c) {
            float xv[kThreadRows][4];
            float wv[kThreadExperts][4];
#pragma unroll
            for (int i = 0; i < kThreadPairs; ++i) {
                const uint4 pair =
                    *reinterpret_cast<const uint4*>(&xs[(k * Chunks + c) * kTileRows + 2 * (tr + kPairGroups * i)]);
                bf16x4_to_f32(make_uint2(pair.x, pair.y), xv[2 * i]);
                bf16x4_to_f32(make_uint2(pair.z, pair.w), xv[2 * i + 1]);
            }
#pragma unroll
            for (int i = 0; i < kThreadExperts; ++i) {
                const float4 v = ws[(k * Chunks + c) * kTileExperts + te + kExpertGroups * i];
                wv[i][0] = v.x;
                wv[i][1] = v.y;
                wv[i][2] = v.z;
                wv[i][3] = v.w;
            }
#pragma unroll
            for (int n = 0; n < 4; ++n) {
#pragma unroll
                for (int a = 0; a < kThreadRows; ++a) {
#pragma unroll
                    for (int b = 0; b < kThreadExperts; ++b) {
                        acc[a][b] = fmaf(wv[b][n], xv[a][n], acc[a][b]);
                    }
                }
            }
        }
        // Fold step m into the subtree: a binary counter over m. The branches are uniform
        // and every array index is a constant.
        const auto store = [&](float (&dst)[kThreadRows][kThreadExperts]) {
#pragma unroll
            for (int a = 0; a < kThreadRows; ++a) {
#pragma unroll
                for (int b = 0; b < kThreadExperts; ++b) {
                    dst[a][b] = acc[a][b];
                }
            }
        };
        const auto add = [&](const float (&src)[kThreadRows][kThreadExperts]) {
#pragma unroll
            for (int a = 0; a < kThreadRows; ++a) {
#pragma unroll
                for (int b = 0; b < kThreadExperts; ++b) {
                    acc[a][b] = src[a][b] + acc[a][b];
                }
            }
        };
        if (!(m & 1)) {
            store(level[0]);
        } else {
            add(level[0]);
            if (!(m & 2)) {
                store(level[1]);
            } else {
                add(level[1]);
                if (!(m & 4)) {
                    store(level[2]);
                } else {
                    add(level[2]);
                    store(level[0]);
                }
            }
        }
    }

    float* out = partials + size_t(w * kQuarters + g) * rows * kExperts;
#pragma unroll
    for (int a = 0; a < kThreadRows; ++a) {
        const int row = r0 + 2 * (tr + kPairGroups * (a / 2)) + a % 2;
        if (row < rows) {
#pragma unroll
            for (int b = 0; b < kThreadExperts; ++b) {
                out[size_t(row) * kExperts + e0 + te + kExpertGroups * b] = level[0][a][b];
            }
        }
    }
}

}  // namespace

// Step 1 of 3: x_out = bf16(rms_norm(x) * norm_weight) for each row of x and x_out
// ([rows, 2304]), as moe_router_2304 computes it, and the same values permuted for the
// logits kernel into x_perm ([rows * 2304], see subtree_base). One warp per row, 8 rows
// per block (256 threads). The warp repeats the norm's sums: virtual warp v of the 576
// virtual threads (lanes l, columns 4 (32 v + l) ..) is the warp's step v, and lane v
// keeps that step's sum for the final butterfly.
extern "C" __global__ void __launch_bounds__(kNormRows * 32)
moe_router_norm_rows_2304(const float* __restrict__ x, const __nv_bfloat16* __restrict__ norm_weight, float eps,
                          __nv_bfloat16* x_out, uint2* x_perm, int rows) {
    constexpr int kSteps = kRuns / 32;
    // The block's rows by permuted run, padded so that consecutive rows fall on
    // different banks.
    __shared__ uint2 ys[kNormRows][kRuns + 1];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int r0 = blockIdx.x * kNormRows;
    const int row = r0 + warp;
    pdl_trigger();
    pdl_wait();
    if (row < rows) {
        const size_t at = size_t(row) * kHidden;
        float v[kSteps][4];
#pragma unroll
        for (int s = 0; s < kSteps; ++s) {
            load4(x + at + 4 * (32 * s + lane), v[s]);
        }
        float mine = 0.f;
#pragma unroll
        for (int s = 0; s < kSteps; ++s) {
            float p = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                p = fmaf(v[s][i], v[s][i], p);
            }
            p = warp_sum(p);
            if (lane == s) {
                mine = p;
            }
        }
        static_assert(kSteps == kNormVirtualWarps, "one step per virtual warp");
        const float inv = inv_rms_f32(warp_sum(mine), eps);
#pragma unroll
        for (int s = 0; s < kSteps; ++s) {
            const int col = 4 * (32 * s + lane);
            float w[4];
            float y[4];
            load4(norm_weight + col, w);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                y[i] = (v[s][i] * inv) * w[i];
            }
            store4(x_out + at + col, y);
            store4(reinterpret_cast<__nv_bfloat16*>(&ys[warp][permuted_run(col)]), y);
        }
    }
    __syncthreads();
    // Each run's 8 rows are 64 contiguous bytes of x_perm.
    for (int idx = threadIdx.x; idx < kRuns * kNormRows; idx += blockDim.x) {
        const int j = idx % kNormRows;
        const int run = idx / kNormRows;
        if (r0 + j < rows) {
            x_perm[size_t(run) * rows + r0 + j] = ys[j][run];
        }
    }
}

// Step 2 of 3: partials[w * 4 + g, row, e] = subtree T(w, g) of logit (row, e) of
// F32(x) @ router.T, from the permuted rows x_perm and the [128, 2304] F32 router.
// Grid: (ceil(rows / 64), 4, 32), 128 threads.
extern "C" __global__ void __launch_bounds__(kTileThreads, 4)
moe_router_logits_rows_2304(const uint2* __restrict__ x_perm, const float* __restrict__ router, float* partials,
                            int rows) {
    // Runs of 4: x at [k][c][row] (BF16), the router at [k][c][expert] (F32).
    __shared__ __align__(16) uint2 xs[kSubtreeLanes * kMaxChunks * kTileRows];
    __shared__ float4 ws[kSubtreeLanes * kMaxChunks * kTileExperts];
    const int w = blockIdx.z / kQuarters;
    const int g = blockIdx.z % kQuarters;
    if (w < 2) {
        router_subtree_tile<3>(x_perm, router, partials, rows, w, g, xs, ws);
    } else {
        router_subtree_tile<2>(x_perm, router, partials, rows, w, g, xs, ws);
    }
}

// Step 3 of 3: for each row (blockIdx.x), logits[row, e] = the sum of warps
// (T0 + T1) + (T2 + T3), added in warp order, then moe_router_2304's top-8 routing into
// indices and weights ([rows, 8]). 128 threads.
extern "C" __global__ void __launch_bounds__(kExperts)
moe_router_route_rows_2304(const float* __restrict__ partials, const float* __restrict__ bias, float* logits,
                           int32_t* indices, float* weights, int rows, float route_scale) {
    const size_t row = blockIdx.x;
    const int e = threadIdx.x;
    pdl_trigger();
    pdl_wait();
    float p[kSubtrees];
#pragma unroll
    for (int s = 0; s < kSubtrees; ++s) {
        p[s] = partials[(size_t(s) * rows + row) * kExperts + e];
    }
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kSplits; ++w) {
        const float* q = p + w * kQuarters;
        const float warp_sum = (q[0] + q[1]) + (q[2] + q[3]);
        total = w == 0 ? warp_sum : total + warp_sum;
    }
    logits[row * kExperts + e] = total;
    // route_top8 reads each logit back in the thread that wrote it.
    route_top8(logits + row * kExperts, bias, indices + row * kTopK, weights + row * kTopK, route_scale);
}
