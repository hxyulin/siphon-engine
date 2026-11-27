// Picking each row's next token from its F32 logits: greedy, or one draw from vLLM's
// filtered distribution.
//
// A row with temperature 0 takes the argmax, with the rule of argmax_f32 in misc.cu (the
// first maximum, NaN above every number), so greedy rows give the same tokens.
//
// Any other row samples from softmax(x / T) after vLLM's filters, in this order:
// - top-k keeps every token whose logit is at least the k-th largest (ties kept);
// - top-p keeps, among those, every token whose strictly larger tokens hold less than a
//   share p of their mass (tokens with equal logits are kept or dropped together);
// - min-p keeps tokens whose probability is at least min_p times the largest.
// Each filter is exact up to F32 rounding of the probabilities. sample_prepare finds the
// filters' thresholds with a cluster of 8 blocks per row, each histogramming an eighth of
// the row and merging the histograms through distributed shared memory: the row's
// maximum, then a histogram of the tokens by distance below it, then a radix over the
// ordered bits of the logits in the bin that holds the threshold, counting tokens for
// top-k and summing their probability for top-p. The probability sums use fixed point
// (2^40 for the largest token) in integer atomics, so the thresholds do not depend on the
// order in which threads add.
//
// sample_race then draws with the exponential race: token i scores x_i / T - ln E_i, with
// E_i an Exp(1) variable, and the kept token with the highest score is a draw from the
// filtered distribution. E_i comes from a hash of the row's seed, its position and i, so a
// row's token depends only on its logits, seed and position, never on the other rows or
// on the reduction order. Speculation relies on this: a verify row and a batched row with
// the same logits, seed and position pick the same token.

#include <cuda_bf16.h>
#include <cfloat>
#include <climits>
#include <cstdint>

#include <cooperative_groups.h>

#include "arch.cuh"
#include "mlx_ops.cuh"
#include "pdl.cuh"
#include "race.cuh"

namespace cg = cooperative_groups;

namespace {

// sample_prepare runs each row in a cluster of kCluster blocks of kPrepareThreads, kCluster
// and PREPARE_THREADS in ops/sample.rs. Ada has no clusters, so one block takes each row
// there. The histograms count in integers and the maxima compare exactly, so the results
// do not depend on how many blocks share a row.
#if SIPHON_SM89
constexpr int kCluster = 1;

// The cluster_group calls sample_prepare makes, for a cluster of one block.
struct ClusterGroup {
    __device__ void sync() { __syncthreads(); }
    __device__ unsigned block_rank() const { return 0; }
    template <typename T>
    __device__ T* map_shared_rank(T* p, int) const {
        return p;
    }
};
#else
constexpr int kCluster = 8;
using ClusterGroup = cg::cluster_group;
#endif
constexpr int kPrepareThreads = 512;
constexpr int kWarps = kPrepareThreads / 32;
constexpr int kRaceThreads = 256;
// Blocks per row of sample_race, kRaceBlocks in ops/sample.rs.
constexpr int kRaceBlocks = 32;
constexpr int kBins = 2048;
// The scaled distance below the maximum that the first level of a threshold search bins,
// the same as kMassFloor, and the tokens of the chosen bin it copies to shared memory.
constexpr float kRange = 28.5f;
constexpr int kList = 1024;
// The largest token's probability in fixed point, and the scaled logit below which a
// token's fixed-point probability rounds to 0: exp(-28.5) * 2^40 < 0.5. Rows hold at most
// kMaxTokens tokens, for the histogram's limbs.
constexpr float kMassScale = 1099511627776.0f;
constexpr float kMassFloor = -28.5f;
constexpr int kMaxTokens = 1 << 18;
static_assert(static_cast<unsigned long long>(kMaxTokens) * 0x3fff < (1ull << 32), "a limb's sum fits 32 bits");

// One row's settings, struct Params in ops/sample.rs. temperature 0 is greedy; top_k 0,
// top_p 1 and min_p 0 turn their filters off.
struct Params {
    float temperature;
    int top_k;
    float top_p;
    float min_p;
    unsigned seed_lo;
    unsigned seed_hi;
    unsigned position;
    unsigned pad;
};

// What sample_prepare leaves for sample_race: whether the row samples, the lowest kept
// key, and the row's maximum.
struct State {
    unsigned sample;
    unsigned lowest;
    float max;
    unsigned pad;
};

// True if (a, ia) should be the argmax over (b, ib): argmax_better in misc.cu.
__device__ __forceinline__ bool argmax_better(float a, int ia, float b, int ib) {
    const bool a_nan = isnan(a);
    const bool b_nan = isnan(b);
    if (a_nan || b_nan) {
        return a_nan && (!b_nan || ia < ib);
    }
    return a > b || (a == b && ia < ib);
}

// The bits of x, ordered as the values: a larger x has a larger key. NaN takes the key
// of -inf, below every kept token.
__device__ __forceinline__ unsigned order_key(float x) {
    if (isnan(x)) {
        x = -INFINITY;
    }
    const unsigned b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// The key of a row's draw: its seed and position (race.cuh).
__device__ __forceinline__ unsigned long long draw_key(const Params& p) {
    return race::draw_key(p.seed_lo, p.seed_hi, p.position);
}

using race::kMaxBoost;
using race::race_boost;

// The fixed-point probability of a token, relative to the row's largest. Converts
// without 64-bit conversion instructions, which run at the FP64 rate on this GPU: below
// 2^24 the value rounds to a 32-bit integer, and above it the float is an integer whose
// mantissa shifts into place.
__device__ __forceinline__ unsigned long long mass(float x, float max, float inv_t) {
    const float z = (x - max) * inv_t;
    if (!(z >= kMassFloor)) {
        return 0;
    }
    const float f = __expf(z) * kMassScale;
    if (f < 16777216.0f) {
        return __float2uint_rn(f);
    }
    const unsigned bits = __float_as_uint(f);
    const int exponent = static_cast<int>(bits >> 23) - 127;
    const unsigned long long mantissa = (bits & 0x7fffffu) | 0x800000u;
    return mantissa << (exponent - 23);
}

// Calls f(x[i], i) for the elements i of x that thread `t` of `threads` takes, in some
// order: 16-byte loads, four in flight per thread, so that one block streams a row near
// one SM's L2 bandwidth instead of waiting on one load at a time.
template <int kThreads, typename F>
__device__ __forceinline__ void for_each(const float* x, int n, int t, F&& f) {
    constexpr int kUnroll = 4;
    const int head = min(n, static_cast<int>(((16 - (reinterpret_cast<uintptr_t>(x) & 15)) & 15) / 4));
    if (t < head) {
        f(x[t], t);
    }
    const float4* v = reinterpret_cast<const float4*>(x + head);
    const int vectors = (n - head) / 4;
    int i = t;
    for (; i + (kUnroll - 1) * kThreads < vectors; i += kUnroll * kThreads) {
        float4 q[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            q[u] = v[i + u * kThreads];
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int b = head + 4 * (i + u * kThreads);
            f(q[u].x, b);
            f(q[u].y, b + 1);
            f(q[u].z, b + 2);
            f(q[u].w, b + 3);
        }
    }
    for (; i < vectors; i += kThreads) {
        const float4 q = v[i];
        const int b = head + 4 * i;
        f(q.x, b);
        f(q.y, b + 1);
        f(q.z, b + 2);
        f(q.w, b + 3);
    }
    const int tail = head + 4 * vectors + t;
    if (tail < n) {
        f(x[tail], tail);
    }
}

// Calls f(v, i) over BF16 logits with 16-byte uint4 (8-wide) loads, four in flight per
// thread. With kSoftcap and cap > 0, applies softcap_f32 inline (matching
// logit_softcap_bf16_f32 bit for bit).
template <int kThreads, bool kSoftcap, typename F>
__device__ __forceinline__ void for_each_bf16(const __nv_bfloat16* x, int n, int t, float cap, F&& f) {
    constexpr int kUnroll = 4;
    const auto eval = [&](__nv_bfloat16 b) -> float {
        const float v = __bfloat162float(b);
        if constexpr (kSoftcap) {
            return cap > 0.f ? softcap_f32(v, cap) : v;
        } else {
            return v;
        }
    };
    const int head =
        min(n, static_cast<int>(((16u - (reinterpret_cast<uintptr_t>(x) & 15u)) & 15u) / 2u));
    if (t < head) {
        f(eval(x[t]), t);
    }
    const uint4* v = reinterpret_cast<const uint4*>(x + head);
    const int vectors = (n - head) / 8;
    int i = t;
    for (; i + (kUnroll - 1) * kThreads < vectors; i += kUnroll * kThreads) {
        uint4 q[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            q[u] = v[i + u * kThreads];
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int b = head + 8 * (i + u * kThreads);
            const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&q[u]);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                f(eval(vals[k]), b + k);
            }
        }
    }
    for (; i < vectors; i += kThreads) {
        const uint4 q = v[i];
        const int b = head + 8 * i;
        const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&q);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            f(eval(vals[k]), b + k);
        }
    }
    const int tail = head + 8 * vectors + t;
    if (tail < n) {
        f(eval(x[tail]), tail);
    }
}

// The sum of v over the block, in every thread.
__device__ __forceinline__ unsigned long long block_sum(unsigned long long v, unsigned long long* scratch) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = v;
    }
    __syncthreads();
    v = threadIdx.x % 32 < kWarps ? scratch[threadIdx.x % 32] : 0;
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// The largest of v over the block, in every thread.
__device__ __forceinline__ float block_max(float v, float* scratch) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = v;
    }
    __syncthreads();
    v = threadIdx.x % 32 < kWarps ? scratch[threadIdx.x % 32] : -INFINITY;
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

// The F32 sum of v over the block, in every thread.
__device__ __forceinline__ float block_sum_f32(float v, float* scratch) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = v;
    }
    __syncthreads();
    v = threadIdx.x % 32 < kWarps ? scratch[threadIdx.x % 32] : 0.f;
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// The smallest (or, with `largest`, the largest) of v over the block, in every thread.
__device__ __forceinline__ unsigned block_extreme(unsigned v, bool largest, unsigned* scratch) {
    const auto pick_one = [largest](unsigned a, unsigned b) { return largest ? max(a, b) : min(a, b); };
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        v = pick_one(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    __syncthreads();
    if (threadIdx.x % 32 == 0) {
        scratch[threadIdx.x / 32] = v;
    }
    __syncthreads();
    v = threadIdx.x % 32 < kWarps ? scratch[threadIdx.x % 32] : (largest ? 0u : UINT_MAX);
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        v = pick_one(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    __syncthreads();
    return v;
}

// The exclusive prefix sum of v over the block's threads in thread order.
__device__ __forceinline__ unsigned long long block_exclusive_scan(unsigned long long v,
                                                                   unsigned long long* scratch) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    unsigned long long inclusive = v;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        const unsigned long long up = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) {
            inclusive += up;
        }
    }
    __syncthreads();
    if (lane == 31) {
        scratch[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        unsigned long long total = lane < kWarps ? scratch[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const unsigned long long up = __shfl_up_sync(0xffffffffu, total, offset);
            if (lane >= offset) {
                total += up;
            }
        }
        scratch[lane] = total;
    }
    __syncthreads();
    const unsigned long long before = warp > 0 ? scratch[warp - 1] : 0;
    return before + inclusive - v;
}

// A histogram of weights below 2^42 in shared memory, as three 14-bit limbs per bin, each
// summed in its own 32-bit word. A limb's sum stays below 2^32 for up to 2^18 tokens, so
// the words never wrap, and the adds need no result: sm_120 has no native 64-bit
// shared-memory add, and an add that returns its old value (for a carry) stalls the
// thread on every token.
struct Hist {
    unsigned limb[3][kBins];

    __device__ void clear(int bins) {
        for (int b = threadIdx.x; b < bins; b += kPrepareThreads) {
            limb[0][b] = 0;
            limb[1][b] = 0;
            limb[2][b] = 0;
        }
    }

    __device__ void add(int bin, unsigned long long w) {
        const unsigned l0 = static_cast<unsigned>(w) & 0x3fffu;
        const unsigned l1 = static_cast<unsigned>(w >> 14) & 0x3fffu;
        const unsigned l2 = static_cast<unsigned>(w >> 28);
        if (l0 != 0) {
            atomicAdd(&limb[0][bin], l0);
        }
        if (l1 != 0) {
            atomicAdd(&limb[1][bin], l1);
        }
        if (l2 != 0) {
            atomicAdd(&limb[2][bin], l2);
        }
    }

    __device__ unsigned long long get(int bin) const {
        return static_cast<unsigned long long>(limb[0][bin]) +
               (static_cast<unsigned long long>(limb[1][bin]) << 14) +
               (static_cast<unsigned long long>(limb[2][bin]) << 28);
    }
};

// The shared memory of one block of a row's cluster.
struct Search {
    // This block's histogram of its own tokens.
    Hist hist;
    // After merge(), this block's share of the cluster's histogram: bins
    // rank * per .. rank * per + per - 1, per = bins / kCluster.
    // Unused on sm_89, where a block's histogram is the cluster's.
    unsigned long long merged[SIPHON_SM89 ? 1 : kBins / kCluster];
    unsigned long long scan[33];
    unsigned found[3];
    unsigned uscratch[32];
    float fscratch[32];
    int iscratch[32];
    // Values each block posts for the others: its maximum, its range of keys, its argmax,
    // and its filtered partition sum for beta-relaxed rejection sampling.
    float max;
    float z_sum;
    unsigned kmin;
    unsigned kmax;
    float best;
    int best_idx;
    // This block's tokens in the chosen bin.
    unsigned count;
    unsigned keys[kList];
    unsigned long long weights[kList];
};

// One block's share of a row: tokens offset .. offset + n - 1.
template <typename Elem>
struct Part {
    const Elem* x;
    int n;
    int offset;
    float cap;
};

template <int kThreads, bool kSoftcap, typename F>
__device__ __forceinline__ void for_each_part(const Part<float>& part, int t, F&& f) {
    for_each<kThreads>(part.x, part.n, t, f);
}

template <int kThreads, bool kSoftcap, typename F>
__device__ __forceinline__ void for_each_part(const Part<__nv_bfloat16>& part, int t, F&& f) {
    for_each_bf16<kThreads, kSoftcap>(part.x, part.n, t, part.cap, f);
}

// Sums every block's histogram of `bins` bins into each block's share of it, `merged`.
// Starts with a cluster barrier, so that every block's histogram is complete and every
// block has finished reading the shares of the last merge.
__device__ void merge(Search& sh, int bins, ClusterGroup& cluster) {
    cluster.sync();
#if SIPHON_SM89
    return;
#endif
    const int per = bins / kCluster;
    const int first = static_cast<int>(cluster.block_rank()) * per;
    for (int j = threadIdx.x; j < per; j += kPrepareThreads) {
        unsigned long long total = 0;
#pragma unroll
        for (int r = 0; r < kCluster; ++r) {
            total += cluster.map_shared_rank(&sh.hist, r)->get(first + j);
        }
        sh.merged[j] = total;
    }
    cluster.sync();
}

// Bin `bin` of the cluster's histogram of `bins` bins, after merge().
__device__ __forceinline__ unsigned long long merged(Search& sh, int bins, int bin, ClusterGroup& cluster) {
#if SIPHON_SM89
    return sh.hist.get(bin);
#endif
    const int per = bins / kCluster;
    return cluster.map_shared_rank(sh.merged, bin / per)[bin % per];
}

// The sum of every bin of the cluster's histogram, after merge().
__device__ unsigned long long merged_total(Search& sh, int bins, ClusterGroup& cluster) {
    unsigned long long own = 0;
    for (int b = threadIdx.x; b < bins; b += kPrepareThreads) {
        own += merged(sh, bins, b, cluster);
    }
    return block_sum(own, sh.scan);
}

// The bin of the cluster's histogram where the weight before it first reaches `target`,
// given the weight `above` of everything before the histogram, and the weight before that
// bin. Bins count in rank order: bin r, or bin bins - 1 - r when descending. Every block
// finds the same bin. Returns UINT_MAX when no bin reaches the target, which the callers
// rule out.
__device__ unsigned pick(Search& sh, int bins, bool descending, unsigned long long above,
                         unsigned long long target, unsigned long long* before_bin,
                         ClusterGroup& cluster) {
    const int per = bins > kPrepareThreads ? bins / kPrepareThreads : 1;
    const int first = per * threadIdx.x;
    const auto at = [&](int rank) { return descending ? bins - 1 - rank : rank; };
    unsigned long long h[kBins / kPrepareThreads];
    unsigned long long own = 0;
    for (int r = 0; r < per; ++r) {
        h[r] = first + r < bins ? merged(sh, bins, at(first + r), cluster) : 0;
        own += h[r];
    }
    if (threadIdx.x == 0) {
        sh.found[0] = UINT_MAX;
    }
    unsigned long long before = above + block_exclusive_scan(own, sh.scan);
    for (int r = 0; r < per && first + r < bins; ++r) {
        if (before < target && before + h[r] >= target) {
            sh.found[0] = static_cast<unsigned>(at(first + r));
            sh.found[1] = static_cast<unsigned>(before);
            sh.found[2] = static_cast<unsigned>(before >> 32);
        }
        before += h[r];
    }
    __syncthreads();
    const unsigned bin = sh.found[0];
    *before_bin = (static_cast<unsigned long long>(sh.found[2]) << 32) | sh.found[1];
    __syncthreads();
    return bin;
}

// The k-th largest key of the row among those at or above `lowest`: a radix over all keys,
// 11, 11 and 10 bits at a time. Only for a k past every token within kRange of the maximum.
template <typename Elem>
__device__ unsigned count_threshold(const Part<Elem>& part, unsigned lowest, unsigned long long k,
                                    Search& sh, ClusterGroup& cluster) {
    unsigned prefix = 0;
    unsigned long long above = 0;
    for (int hi = 32; hi > 0;) {
        const int width = hi == 10 ? 10 : 11;
        const int shift = hi - width;
        const int bins = 1 << width;
        sh.hist.clear(bins);
        __syncthreads();
        for_each_part<kPrepareThreads, true>(part, threadIdx.x, [&](float v, int) {
            const unsigned key = order_key(v);
            if (isnan(v) || key < lowest || (hi < 32 && (key >> hi) != (prefix >> hi))) {
                return;
            }
            sh.hist.add((key >> shift) & (bins - 1), 1ull);
        });
        merge(sh, bins, cluster);
        const unsigned digit = pick(sh, bins, true, above, k, &above, cluster);
        if (digit == UINT_MAX) {
            return lowest;
        }
        prefix |= digit << shift;
        hi = shift;
    }
    return prefix;
}

// The lowest key a filter keeps among the row's keys at or above `lowest`: the largest key
// K whose strictly larger keys weigh less than the target. Each token weighs 1 (by_mass
// false, target k) or its fixed-point probability (by_mass true, target the share `share`
// of the total, rounded up). Returns `lowest` when the filter keeps every token. Each
// block of the cluster histograms its own part of the row, and merge() sums them.
//
// The first level bins the tokens by their scaled distance below the maximum,
// d = (max - x) / T, in 2048 bins over [0, kRange): a bin holds one contiguous run of
// keys, and the bins spread the bulk of a vocabulary so that the shared-memory atomics
// rarely collide. Tokens past kRange weigh nothing by mass; by count, they only add to the
// total, and a k past the tokens in range falls back to a radix over all keys. Each block
// copies its tokens of the bin where the weight reaches the target to shared memory (when
// they fit), and a radix over their keys, 11 bits at a time, finds the exact key.
template <typename Elem>
__device__ unsigned threshold(const Part<Elem>& part, unsigned lowest, float top, float inv_t,
                              bool by_mass, unsigned long long k, float share, Search& sh,
                              ClusterGroup& cluster) {
    sh.hist.clear(kBins);
    if (threadIdx.x == 0) {
        sh.count = 0;
    }
    __syncthreads();
    // The bin of a token, or -1 for a token the search leaves out, and its weight. A
    // token whose fixed-point probability rounds to 0 weighs nothing, so it is left out.
    const auto bin_of = [&](float v) {
        if (isnan(v) || order_key(v) < lowest) {
            return -1;
        }
        const float d = (top - v) * inv_t;
        if (!(d < kRange)) {
            return -1;
        }
        return min(static_cast<int>(d * (kBins / kRange)), kBins - 1);
    };
    const auto weight = [&](float v) { return by_mass ? mass(v, top, inv_t) : 1ull; };
    for_each_part<kPrepareThreads, true>(part, threadIdx.x, [&](float v, int) {
        const int bin = bin_of(v);
        if (bin >= 0) {
            const unsigned long long w = weight(v);
            if (w != 0) {
                sh.hist.add(bin, w);
            }
        }
    });
    merge(sh, kBins, cluster);
    const unsigned long long in_range = merged_total(sh, kBins, cluster);
    unsigned long long target = k;
    if (by_mass) {
        const double t = ceil(static_cast<double>(share) * static_cast<double>(in_range));
        target = t < 1.0 ? 1ull : min(static_cast<unsigned long long>(t), in_range);
    } else if (in_range < k) {
        return count_threshold(part, lowest, k, sh, cluster);
    }
    unsigned long long above;
    const unsigned chosen = pick(sh, kBins, false, 0, target, &above, cluster);
    if (chosen == UINT_MAX) {
        return lowest;
    }
    // This block's tokens of the chosen bin, and the cluster's range of their keys.
    unsigned kmin = UINT_MAX;
    unsigned kmax = 0;
    for_each_part<kPrepareThreads, true>(part, threadIdx.x, [&](float v, int) {
        if (bin_of(v) != static_cast<int>(chosen)) {
            return;
        }
        const unsigned long long w = weight(v);
        if (w == 0) {
            return;
        }
        const unsigned key = order_key(v);
        kmin = min(kmin, key);
        kmax = max(kmax, key);
        const unsigned slot = atomicAdd(&sh.count, 1u);
        if (slot < kList) {
            sh.keys[slot] = key;
            sh.weights[slot] = w;
        }
    });
    kmin = block_extreme(kmin, false, sh.uscratch);
    kmax = block_extreme(kmax, true, sh.uscratch);
    if (threadIdx.x == 0) {
        sh.kmin = kmin;
        sh.kmax = kmax;
    }
    cluster.sync();
    for (int r = 0; r < kCluster; ++r) {
        const Search* other = cluster.map_shared_rank(&sh, r);
        kmin = min(kmin, other->kmin);
        kmax = max(kmax, other->kmax);
    }
    const unsigned count = sh.count;
    const unsigned range = kmax - kmin;
    const int bits = range == 0 ? 0 : 32 - __clz(range);
    unsigned prefix = 0;
    for (int hi = bits; hi > 0;) {
        const int width = hi < 11 ? hi : 11;
        const int shift = hi - width;
        // At least kCluster bins, so that each block holds a share.
        const int bins = max(1 << width, kCluster);
        sh.hist.clear(bins);
        __syncthreads();
        const auto add = [&](unsigned key, unsigned long long w) {
            const unsigned rel = key - kmin;
            if (hi < 32 && (rel >> hi) != (prefix >> hi)) {
                return;
            }
            sh.hist.add((rel >> shift) & ((1u << width) - 1), w);
        };
        if (count <= kList) {
            for (unsigned j = threadIdx.x; j < count; j += kPrepareThreads) {
                add(sh.keys[j], sh.weights[j]);
            }
        } else {
            for_each_part<kPrepareThreads, true>(part, threadIdx.x, [&](float v, int) {
                if (bin_of(v) == static_cast<int>(chosen)) {
                    const unsigned long long w = weight(v);
                    if (w != 0) {
                        add(order_key(v), w);
                    }
                }
            });
        }
        merge(sh, bins, cluster);
        const unsigned digit = pick(sh, bins, true, above, target, &above, cluster);
        if (digit == UINT_MAX) {
            return lowest;
        }
        prefix |= digit << shift;
        hi = shift;
    }
    return kmin + prefix;
}

template <typename Elem>
__device__ __forceinline__ void sample_prepare_impl(const Elem* logits, int n, int stride,
                                                    float cap, const Params* params, State* state,
                                                    int* out, Search& sh) {
    pdl_wait();
#if SIPHON_SM89
    ClusterGroup cluster;
#else
    cg::cluster_group cluster = cg::this_cluster();
#endif
    const int rank = static_cast<int>(cluster.block_rank());
    const int row = blockIdx.x / kCluster;
    const Elem* x = logits + static_cast<size_t>(row) * stride;
    constexpr int kAlign = sizeof(Elem) == 2 ? 8 : 4;
    const int share = ((n + kCluster - 1) / kCluster + (kAlign - 1)) & ~(kAlign - 1);
    const int offset = min(n, rank * share);
    const Part<Elem> part{x + offset, min(share, n - offset), offset, cap};
    const Params p = params[row];

    if (!(p.temperature > 0.f)) {
        float best = -INFINITY;
        int best_idx = INT_MAX;
        for_each_part<kPrepareThreads, false>(part, threadIdx.x, [&](float v, int i) {
            if (argmax_better(v, part.offset + i, best, best_idx)) {
                best = v;
                best_idx = part.offset + i;
            }
        });
        const auto warp_reduce = [&] {
#pragma unroll
            for (int o = 16; o >= 1; o /= 2) {
                const float v = __shfl_xor_sync(0xffffffffu, best, o);
                const int idx = __shfl_xor_sync(0xffffffffu, best_idx, o);
                if (argmax_better(v, idx, best, best_idx)) {
                    best = v;
                    best_idx = idx;
                }
            }
        };
        warp_reduce();
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        if (lane == 0) {
            sh.fscratch[warp] = best;
            sh.iscratch[warp] = best_idx;
        }
        __syncthreads();
        if (warp == 0) {
            best = lane < kWarps ? sh.fscratch[lane] : -INFINITY;
            best_idx = lane < kWarps ? sh.iscratch[lane] : INT_MAX;
            warp_reduce();
            if (lane == 0) {
                sh.best = best;
                sh.best_idx = best_idx;
            }
        }
        cluster.sync();
        if (rank == 0 && warp == 0) {
            best = lane < kCluster ? cluster.map_shared_rank(&sh, lane)->best : -INFINITY;
            best_idx = lane < kCluster ? cluster.map_shared_rank(&sh, lane)->best_idx : INT_MAX;
            warp_reduce();
            if (lane == 0) {
                out[row] = best_idx;
                state[row] = State{0u, 0u, 0.f, 0u};
            }
        }
        // No block leaves while another may read its shared memory.
        cluster.sync();
        pdl_trigger();
        return;
    }

    const bool top_k = p.top_k > 0 && p.top_k < n;
    const bool top_p = p.top_p < 1.f;
    const bool min_p = p.min_p > 0.f;
    float max = 0.f;
    if (top_k || top_p || min_p) {
        float m = -INFINITY;
        for_each_part<kPrepareThreads, true>(part, threadIdx.x, [&](float v, int) { m = fmaxf(m, v); });
        m = block_max(m, sh.fscratch);
        if (threadIdx.x == 0) {
            sh.max = m;
        }
        cluster.sync();
        for (int r = 0; r < kCluster; ++r) {
            m = fmaxf(m, cluster.map_shared_rank(&sh, r)->max);
        }
        max = m;
    }
    const float inv_t = 1.f / p.temperature;
    unsigned lowest = 0;
    if (top_k) {
        lowest = threshold(part, lowest, max, inv_t, false, static_cast<unsigned long long>(p.top_k),
                           1.f, sh, cluster);
    }
    if (top_p) {
        lowest = threshold(part, lowest, max, inv_t, true, 0, p.top_p, sh, cluster);
    }
    if (rank == 0 && threadIdx.x == 0) {
        state[row] = State{1u, lowest, max, 0u};
    }
    // No block leaves while another may read its shared memory.
    cluster.sync();
    pdl_trigger();
}

template <typename Elem>
__device__ __forceinline__ void sample_race_impl(const Elem* logits, int n, int stride, float cap,
                                                 const Params* params, const State* state,
                                                 float* partial_scores, int* partial_idxs,
                                                 unsigned* counters, int* out) {
    __shared__ float warp_scores[kRaceThreads / 32];
    __shared__ int warp_idxs[kRaceThreads / 32];
    __shared__ bool last;
    pdl_trigger();
    pdl_wait();
    const int row = blockIdx.y;
    const State s = state[row];
    if (s.sample == 0) {
        return;
    }
    const Params p = params[row];
    const Elem* x = logits + static_cast<size_t>(row) * stride;
    const float inv_t = 1.f / p.temperature;
    const bool min_p = p.min_p > 0.f;
    const unsigned long long key = draw_key(p);
    float best = -INFINITY;
    int best_idx = INT_MAX;
    const auto take = [&](float score, int i) {
        if (score > best || (score == best && i < best_idx)) {
            best = score;
            best_idx = i;
        }
    };
    const auto consider = [&](float v, int i) {
        if (order_key(v) < s.lowest || isnan(v)) {
            return;
        }
        const float scaled = v * inv_t;
        // A token that cannot beat the best so far even with the largest boost.
        if (scaled + kMaxBoost < best) {
            return;
        }
        if (min_p && expf((v - s.max) * inv_t) < p.min_p) {
            return;
        }
        take(scaled + race_boost(key, i), i);
    };
    const Part<Elem> part{x, n, 0, cap};
    for_each_part<kRaceBlocks * kRaceThreads, true>(part, blockIdx.x * kRaceThreads + threadIdx.x,
                                                    consider);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const auto warp_reduce = [&] {
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            take(__shfl_xor_sync(0xffffffffu, best, offset),
                 __shfl_xor_sync(0xffffffffu, best_idx, offset));
        }
    };
    warp_reduce();
    if (lane == 0) {
        warp_scores[warp] = best;
        warp_idxs[warp] = best_idx;
    }
    __syncthreads();
    if (warp == 0) {
        best = lane < kRaceThreads / 32 ? warp_scores[lane] : -INFINITY;
        best_idx = lane < kRaceThreads / 32 ? warp_idxs[lane] : INT_MAX;
        warp_reduce();
        if (lane == 0) {
            partial_scores[row * kRaceBlocks + blockIdx.x] = best;
            partial_idxs[row * kRaceBlocks + blockIdx.x] = best_idx;
            __threadfence();
            last = atomicAdd(&counters[row], 1u) == kRaceBlocks - 1;
        }
    }
    __syncthreads();
    if (!last) {
        return;
    }
    if (warp == 0) {
        __threadfence();
        best = lane < kRaceBlocks ? __ldcg(&partial_scores[row * kRaceBlocks + lane]) : -INFINITY;
        best_idx = lane < kRaceBlocks ? __ldcg(&partial_idxs[row * kRaceBlocks + lane]) : INT_MAX;
        warp_reduce();
        if (lane == 0) {
            // Every token filtered out cannot happen (the largest is always kept), but a
            // row of NaN would leave INT_MAX; take token 0 then.
            out[row] = best_idx == INT_MAX ? 0 : best_idx;
            counters[row] = 0;
        }
    }
}

}  // namespace

// The first stage, one cluster of kCluster blocks per row of logits [rows, n], each block
// taking an eighth of the row: a greedy row writes its argmax to out; a sampling row writes
// its thresholds to state for sample_race. Grid: rows * kCluster blocks.
extern "C" __global__ void SIPHON_CLUSTER_DIMS(kCluster) __launch_bounds__(kPrepareThreads)
sample_prepare(const float* logits, int n, const Params* params, State* state, int* out) {
    __shared__ Search sh;
    sample_prepare_impl<float>(logits, n, n, 0.f, params, state, out, sh);
}

// BF16 logits [rows, stride] (reading the first n columns with 8-wide uint4 loads) and
// applying softcap(x, cap) inline on non-greedy rows.
extern "C" __global__ void SIPHON_CLUSTER_DIMS(kCluster) __launch_bounds__(kPrepareThreads)
sample_prepare_bf16(const __nv_bfloat16* logits, int n, int stride, float cap, const Params* params,
                    State* state, int* out) {
    __shared__ Search sh;
    sample_prepare_impl<__nv_bfloat16>(logits, n, stride, cap, params, state, out, sh);
}

// The second stage: the exponential race over each sampling row's kept tokens. Grid
// (kRaceBlocks, rows) of 256 threads. Each block writes its best (score, index) to the
// row's partials, and the row's last block to finish, found with counters[row], writes
// the winner to out. counters must start at 0; the last block resets its row's.
extern "C" __global__ void __launch_bounds__(kRaceThreads)
sample_race(const float* logits, int n, const Params* params, const State* state, float* partial_scores,
            int* partial_idxs, unsigned* counters, int* out) {
    sample_race_impl<float>(logits, n, n, 0.f, params, state, partial_scores, partial_idxs, counters,
                            out);
}

// BF16 logits [rows, stride] (reading the first n columns with 8-wide uint4 loads) with
// inline softcap(x, cap).
extern "C" __global__ void __launch_bounds__(kRaceThreads)
sample_race_bf16(const __nv_bfloat16* logits, int n, int stride, float cap, const Params* params,
                 const State* state, float* partial_scores, int* partial_idxs, unsigned* counters,
                 int* out) {
    sample_race_impl<__nv_bfloat16>(logits, n, stride, cap, params, state, partial_scores,
                                    partial_idxs, counters, out);
}

// ---------------------------------------------------------------------------------------
// Beta-relaxed / maximal-coupling rejection sampling for 16-sparse DFlash 2 drafts.
//
// For beta = 1.0, this implements exact lossless Leviathan/Chen maximal-coupling rejection
// sampling with residual resampling r(y) \propto max(0, p(y) - q(y)). For 0 < beta < 1,
// it relaxes acceptance to min(1, p(x) / (beta * q(x))) with residual max(0, p - beta * q),
// guaranteeing 0% out-of-support mass (supp(p_tilde) <= supp(p_filt)).
//
// Grid matches sample_prepare_bf16 (rows * kCluster blocks, 512 threads) and
// sample_race_bf16 ((kRaceBlocks, rows), 256 threads). Accepted sampling rows early-return
// in sample_race_beta_bf16 without scanning the vocabulary.

extern "C" __global__ void SIPHON_CLUSTER_DIMS(kCluster) __launch_bounds__(kPrepareThreads)
sample_prepare_beta_bf16(const __nv_bfloat16* logits, int n, int stride, float cap,
                         const Params* params, const int* tokens, int width, const int* cand,
                         const float* q_probs, int cand_steps_stride, float beta, State* state,
                         int* accept_flag, float* r_log, int* out) {
    constexpr int kCandK = 16;
    __shared__ Search sh;
    pdl_wait();
#if SIPHON_SM89
    ClusterGroup cluster;
#else
    cg::cluster_group cluster = cg::this_cluster();
#endif
    const int rank = static_cast<int>(cluster.block_rank());
    const int row = blockIdx.x / kCluster;
    const __nv_bfloat16* x = logits + static_cast<size_t>(row) * stride;
    constexpr int kAlign = 8;
    const int share = ((n + kCluster - 1) / kCluster + (kAlign - 1)) & ~(kAlign - 1);
    const int offset = min(n, rank * share);
    const Part<__nv_bfloat16> part{x + offset, min(share, n - offset), offset, cap};
    const Params p = params[row];

    if (!(p.temperature > 0.f)) {
        float best = -INFINITY;
        int best_idx = INT_MAX;
        for_each_part<kPrepareThreads, false>(part, threadIdx.x, [&](float v, int i) {
            if (argmax_better(v, part.offset + i, best, best_idx)) {
                best = v;
                best_idx = part.offset + i;
            }
        });
        const auto warp_reduce = [&] {
#pragma unroll
            for (int o = 16; o >= 1; o /= 2) {
                const float v = __shfl_xor_sync(0xffffffffu, best, o);
                const int idx = __shfl_xor_sync(0xffffffffu, best_idx, o);
                if (argmax_better(v, idx, best, best_idx)) {
                    best = v;
                    best_idx = idx;
                }
            }
        };
        warp_reduce();
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        if (lane == 0) {
            sh.fscratch[warp] = best;
            sh.iscratch[warp] = best_idx;
        }
        __syncthreads();
        if (warp == 0) {
            best = lane < kWarps ? sh.fscratch[lane] : -INFINITY;
            best_idx = lane < kWarps ? sh.iscratch[lane] : INT_MAX;
            warp_reduce();
            if (lane == 0) {
                sh.best = best;
                sh.best_idx = best_idx;
            }
        }
        cluster.sync();
        if (rank == 0 && warp == 0) {
            best = lane < kCluster ? cluster.map_shared_rank(&sh, lane)->best : -INFINITY;
            best_idx = lane < kCluster ? cluster.map_shared_rank(&sh, lane)->best_idx : INT_MAX;
            warp_reduce();
            if (lane == 0) {
                out[row] = best_idx;
                state[row] = State{0u, 0u, 0.f, 0u};
                accept_flag[row] = -1;
            }
        }
        cluster.sync();
        pdl_trigger();
        return;
    }

    const bool top_k = p.top_k > 0 && p.top_k < n;
    const bool top_p = p.top_p < 1.f;
    const bool min_p = p.min_p > 0.f;
    float m = -INFINITY;
    for_each_part<kPrepareThreads, true>(part, threadIdx.x, [&](float v, int) { m = fmaxf(m, v); });
    m = block_max(m, sh.fscratch);
    if (threadIdx.x == 0) {
        sh.max = m;
    }
    cluster.sync();
    for (int r = 0; r < kCluster; ++r) {
        m = fmaxf(m, cluster.map_shared_rank(&sh, r)->max);
    }
    const float max = m;
    const float inv_t = 1.f / p.temperature;
    unsigned lowest = 0;
    if (top_k) {
        lowest = threshold(part, lowest, max, inv_t, false, static_cast<unsigned long long>(p.top_k),
                           1.f, sh, cluster);
    }
    if (top_p) {
        lowest = threshold(part, lowest, max, inv_t, true, 0, p.top_p, sh, cluster);
    }

    // Filtered partition function Z_kept = sum_{kept} exp((v - max) / T).
    float own_z = 0.f;
    for_each_part<kPrepareThreads, true>(part, threadIdx.x, [&](float v, int) {
        if (isnan(v) || order_key(v) < lowest) {
            return;
        }
        const float z = (v - max) * inv_t;
        if (!(z >= kMassFloor)) {
            return;
        }
        const float w = expf(z);
        if (min_p && w < p.min_p) {
            return;
        }
        own_z += w;
    });
    own_z = block_sum_f32(own_z, sh.fscratch);
    if (threadIdx.x == 0) {
        sh.z_sum = own_z;
    }
    cluster.sync();

    if (rank == 0 && (threadIdx.x / 32) == 0) {
        const int lane = threadIdx.x & 31;
        float z_kept = 0.f;
        for (int r = 0; r < kCluster; ++r) {
            z_kept += cluster.map_shared_rank(&sh, r)->z_sum;
        }
        z_kept = fmaxf(z_kept, 1e-30f);

        const int seq = row / width;
        const int local_row = row % width;
        const bool has_draft = (local_row < width - 1) && (cand != nullptr) && (q_probs != nullptr);
        if (!has_draft) {
            if (lane == 0) {
                state[row] = State{1u, lowest, max, 0u};
                accept_flag[row] = 0;
            }
        } else {
            const int cand_row = seq * cand_steps_stride + local_row;
            const int draft_tok = tokens[seq * width + local_row + 1];
            int hit = 0;
            if (lane < kCandK) {
                const int cid = cand[static_cast<size_t>(cand_row) * kCandK + lane];
                const float qp = q_probs[static_cast<size_t>(cand_row) * kCandK + lane];
                float w_p = 0.f;
                bool passes = false;
                if (cid >= 0 && cid < n) {
                    const float raw = __bfloat162float(x[cid]);
                    const float v = cap > 0.f ? softcap_f32(raw, cap) : raw;
                    if (!isnan(v) && order_key(v) >= lowest) {
                        const float w = expf((v - max) * inv_t);
                        if (!min_p || w >= p.min_p) {
                            w_p = w;
                            passes = true;
                        }
                    }
                }
                const float w_q = beta * qp * z_kept;
                r_log[static_cast<size_t>(row) * kCandK + lane] =
                    (w_p > w_q) ? logf(w_p - w_q) : -INFINITY;
                if (cid == draft_tok && qp > 0.f && passes) {
                    const float p_d = w_p / z_kept;
                    const float u = race::unit(race::mix64(draw_key(p) ^ race::kAcceptSalt));
                    if (u * beta * qp <= p_d) {
                        hit = 1;
                    }
                }
            }
            const int acc = __any_sync(0xffffffffu, hit != 0) ? 1 : 0;
            if (lane == 0) {
                state[row] = State{1u, lowest, max, 0u};
                accept_flag[row] = acc;
                if (acc == 1) {
                    out[row] = draft_tok;
                }
            }
        }
    }
    cluster.sync();
    pdl_trigger();
}

extern "C" __global__ void __launch_bounds__(kRaceThreads)
sample_race_beta_bf16(const __nv_bfloat16* logits, int n, int stride, float cap,
                      const Params* params, const State* state, int width, const int* cand,
                      int cand_steps_stride, const int* accept_flag, const float* r_log,
                      float* partial_scores, int* partial_idxs, float* fb_scores, int* fb_idxs,
                      unsigned* counters, int* out) {
    constexpr int kCandK = 16;
    __shared__ float warp_scores[kRaceThreads / 32];
    __shared__ int warp_idxs[kRaceThreads / 32];
    __shared__ float fb_warp_scores[kRaceThreads / 32];
    __shared__ int fb_warp_idxs[kRaceThreads / 32];
    __shared__ int sh_cand[kCandK];
    __shared__ float sh_rlog[kCandK];
    __shared__ bool last;
    pdl_trigger();
    pdl_wait();
    const int row = blockIdx.y;
    const State s = state[row];
    if (s.sample == 0 || accept_flag[row] != 0) {
        return;
    }
    const int seq = row / width;
    const int local_row = row % width;
    const bool has_draft = (local_row < width - 1) && (cand != nullptr) && (r_log != nullptr);
    if (has_draft && threadIdx.x < kCandK) {
        const int cand_row = seq * cand_steps_stride + local_row;
        sh_cand[threadIdx.x] = cand[static_cast<size_t>(cand_row) * kCandK + threadIdx.x];
        sh_rlog[threadIdx.x] = r_log[static_cast<size_t>(row) * kCandK + threadIdx.x];
    }
    __syncthreads();

    const Params p = params[row];
    const __nv_bfloat16* x = logits + static_cast<size_t>(row) * stride;
    const float inv_t = 1.f / p.temperature;
    const bool min_p = p.min_p > 0.f;
    const unsigned long long key = race::mix64(draw_key(p) ^ race::kResidSalt);
    float best = -INFINITY;
    int best_idx = INT_MAX;
    float fb_best = -INFINITY;
    int fb_idx = INT_MAX;
    const auto take = [&](float score, int i) {
        if (score > best || (score == best && i < best_idx)) {
            best = score;
            best_idx = i;
        }
    };
    const auto fb_take = [&](float score, int i) {
        if (score > fb_best || (score == fb_best && i < fb_idx)) {
            fb_best = score;
            fb_idx = i;
        }
    };
    const auto consider = [&](float v, int i) {
        if (order_key(v) < s.lowest || isnan(v)) {
            return;
        }
        const float z = (v - s.max) * inv_t;
        if (z + kMaxBoost + 0.01f < fminf(best, fb_best)) {
            return;
        }
        if (min_p && expf(z) < p.min_p) {
            return;
        }
        float z_res = z;
        bool res_ok = true;
        if (has_draft) {
#pragma unroll
            for (int k = 0; k < kCandK; ++k) {
                if (i == sh_cand[k]) {
                    z_res = sh_rlog[k];
                    res_ok = isfinite(z_res);
                    break;
                }
            }
        }
        const bool need_res = res_ok && (z_res + kMaxBoost >= best);
        const bool need_fb = (z + kMaxBoost >= fb_best);
        if (!need_res && !need_fb) {
            return;
        }
        const float boost = race_boost(key, i);
        if (need_res) {
            take(z_res + boost, i);
        }
        if (need_fb) {
            fb_take(z + boost, i);
        }
    };
    const Part<__nv_bfloat16> part{x, n, 0, cap};
    for_each_part<kRaceBlocks * kRaceThreads, true>(part, blockIdx.x * kRaceThreads + threadIdx.x,
                                                    consider);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const auto warp_reduce = [&] {
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            take(__shfl_xor_sync(0xffffffffu, best, offset),
                 __shfl_xor_sync(0xffffffffu, best_idx, offset));
            fb_take(__shfl_xor_sync(0xffffffffu, fb_best, offset),
                    __shfl_xor_sync(0xffffffffu, fb_idx, offset));
        }
    };
    warp_reduce();
    if (lane == 0) {
        warp_scores[warp] = best;
        warp_idxs[warp] = best_idx;
        fb_warp_scores[warp] = fb_best;
        fb_warp_idxs[warp] = fb_idx;
    }
    __syncthreads();
    if (warp == 0) {
        best = lane < kRaceThreads / 32 ? warp_scores[lane] : -INFINITY;
        best_idx = lane < kRaceThreads / 32 ? warp_idxs[lane] : INT_MAX;
        fb_best = lane < kRaceThreads / 32 ? fb_warp_scores[lane] : -INFINITY;
        fb_idx = lane < kRaceThreads / 32 ? fb_warp_idxs[lane] : INT_MAX;
        warp_reduce();
        if (lane == 0) {
            partial_scores[row * kRaceBlocks + blockIdx.x] = best;
            partial_idxs[row * kRaceBlocks + blockIdx.x] = best_idx;
            fb_scores[row * kRaceBlocks + blockIdx.x] = fb_best;
            fb_idxs[row * kRaceBlocks + blockIdx.x] = fb_idx;
            __threadfence();
            last = atomicAdd(&counters[row], 1u) == kRaceBlocks - 1;
        }
    }
    __syncthreads();
    if (!last) {
        return;
    }
    if (warp == 0) {
        __threadfence();
        best = lane < kRaceBlocks ? __ldcg(&partial_scores[row * kRaceBlocks + lane]) : -INFINITY;
        best_idx = lane < kRaceBlocks ? __ldcg(&partial_idxs[row * kRaceBlocks + lane]) : INT_MAX;
        fb_best = lane < kRaceBlocks ? __ldcg(&fb_scores[row * kRaceBlocks + lane]) : -INFINITY;
        fb_idx = lane < kRaceBlocks ? __ldcg(&fb_idxs[row * kRaceBlocks + lane]) : INT_MAX;
        warp_reduce();
        if (lane == 0) {
            const int chosen = best_idx != INT_MAX ? best_idx : (fb_idx != INT_MAX ? fb_idx : 0);
            out[row] = chosen;
            counters[row] = 0;
        }
    }
}

// -ln E_i of tokens 0..n of the draw with this seed and position, for tests of the host
// port (crates/siphon/src/race.rs). One thread per token.
extern "C" __global__ void sample_noise(unsigned seed_lo, unsigned seed_hi, unsigned position,
                                        int n, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = race::race_boost(race::draw_key(seed_lo, seed_hi, position), i);
    }
}
