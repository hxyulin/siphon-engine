// Decode attention with relative-position bias, attention sinks and KV caches,
// following MLX 0.32.4 on the M5 Max that made the reference fixtures.
//
// Shapes are those of every Inkling layer: 24 query heads, 8 KV heads (GQA
// groups of 3), head dim 128, relative width 16.
//
// KV cache layout: keys and values are separate BF16 buffers of shape
// [slots, 8, 128], so appending one token writes 2 KiB contiguously into each.
// A sliding layer's cache is a ring of `slots` = window entries and position p
// lives in slot p mod slots. A global layer's cache stores position p in slot p
// and never wraps. Both cases share one rule: at query position p the cache
// holds n = min(p + 1, slots) keys in slots 0..n-1, and slot s holds the key at
// distance (p - s) mod slots. This matches MLX's RotatingKVCache and KVCache
// when they are fed one token at a time.
//
// Arithmetic follows the Metal kernels it replaces:
//
// - `relative_bias`: MLX's BF16 matmul for M = 24, K = 16 sums the 16 exact
//   products as a fixed tree, found by probing it with cancelling inputs:
//   pairs (k, k + 1), then pair k with pair k + 8, then the four results in
//   order. With that tree the output is bit-exact.
// - `decode_attention_*`: `sdpa_vector` (one pass, fewer than 1024 keys) and
//   `sdpa_vector_2pass_1/2` (two passes, 1024 keys or more) in
//   kernels/sdpa_vector.h. Both split the keys into `blocks` interleaved
//   strides, key s going to block s mod blocks, keep an online softmax per
//   block and then combine the blocks. The one-pass kernel is the two-pass
//   structure with 32 blocks whose partial outputs stay in F32; the two-pass
//   kernel on an M5 Max uses 64 blocks for this GQA factor and rounds the
//   unnormalized partial outputs to BF16. The kernels pick the same split from
//   the position they read on the device; the launcher always starts 64 blocks
//   and blocks past the split exit, so a captured graph serves every position.
//   Within a block:
//     q' = scale * q                        (scale applied to q, in F32)
//     score = sum over the lane's 4 dims    (FMA chain), then simd_sum
//     score += mask                         (BF16 bias, or 0 past the table)
//     m' = max(m, score)
//     factor = exp(m - m'), e = exp(score - m')
//     sum = sum * factor + e                (FMA)
//     o = o * factor + e * v                (FMA onto the rounded product)
//   Block 0 starts from the sink: m = sink, sum = 1. Other blocks start from
//   m = -FLT_MAX, sum = 0. exp is Metal's fast::exp, exp2(x * log2(e)), and is
//   not correctly rounded on Apple GPUs; exp2f stands in for it, so results
//   differ from MLX by rounding.
//
// The Metal compiler contracts `a += b * c` into an FMA even with
// -fno-fast-math (see norm.cu's RMSNorm). Against the fixtures, the FMA forms
// above leave 1 differing BF16 output in 55296 and unfused forms leave 8. The
// kernels spell every operation as a PTX .rn.ftz instruction so that nvcc
// cannot fuse or split them; .ftz because MLX flushes subnormals.

#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

namespace {

constexpr int kHeads = 24;
constexpr int kKvHeads = 8;
constexpr int kGroup = kHeads / kKvHeads;
constexpr int kHeadDim = 128;
constexpr int kRelDim = 16;
// Elements of a 128-wide head each lane owns, MLX's qk_per_thread.
constexpr int kVec = kHeadDim / 32;
constexpr int kKvRow = kKvHeads * kHeadDim;
// MLX on an M5 Max switches from its one-pass to its two-pass kernel at this
// many keys, and their block counts.
constexpr unsigned kTwoPassKeys = 1024;
constexpr int kOnePassBlocks = 32;
constexpr int kTwoPassBlocks = 64;

__device__ __forceinline__ float fadd(float a, float b) {
    float r;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fsub(float a, float b) {
    float r;
    asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fmul(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float ffma(float a, float b, float c) {
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

__device__ __forceinline__ float fdiv(float a, float b) {
    float r;
    asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// Metal's fast::exp: exp2 of the F32 product x * log2(e).
__device__ __forceinline__ float fast_exp(float x) {
    return exp2f(fmul(x, 1.4426950408889634f));
}

// simd_sum on Apple GPUs: an xor butterfly over lane offsets 1, 2, 4, 8, 16.
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        v = fadd(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ __forceinline__ void load_bf16x4(const __nv_bfloat16* p, float (&v)[kVec]) {
    const uint2 packed = *reinterpret_cast<const uint2*>(p);
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        v[i] = __bfloat162float(vals[i]);
    }
}

__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

// Key blocks of the split MLX uses for a query at `position` over a cache of
// `slots`: one pass below kTwoPassKeys keys, two passes from there on.
__device__ __forceinline__ int split_blocks(unsigned position, unsigned slots) {
    return min(position + 1, slots) < kTwoPassKeys ? kOnePassBlocks : kTwoPassBlocks;
}

}  // namespace

// bias[h, r] = sum_k rel[h, k] * table[k, r] for rel [24, 16] and table
// [16, extent], written as BF16 [24, extent]. One thread per output.
extern "C" __global__ void __launch_bounds__(256)
relative_bias_bf16(const __nv_bfloat16* rel, const __nv_bfloat16* table, __nv_bfloat16* bias, int extent) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y;
    if (r >= extent) {
        return;
    }
    // Products of two BF16 values are exact in F32.
    float x[kRelDim];
#pragma unroll
    for (int k = 0; k < kRelDim; ++k) {
        x[k] = fmul(__bfloat162float(rel[h * kRelDim + k]), __bfloat162float(table[size_t(k) * extent + r]));
    }
    float pair[kRelDim / 2];
#pragma unroll
    for (int k = 0; k < kRelDim / 2; ++k) {
        pair[k] = fadd(x[2 * k], x[2 * k + 1]);
    }
    float quad[kRelDim / 4];
#pragma unroll
    for (int k = 0; k < kRelDim / 4; ++k) {
        quad[k] = fadd(pair[k], pair[k + kRelDim / 4]);
    }
    const float acc = fadd(fadd(fadd(quad[0], quad[1]), quad[2]), quad[3]);
    bias[h * extent + r] = __float2bfloat16(acc);
}

// Writes one token's k and v ([8, 128] each) into the caches ([slots, 8, 128])
// at slot *position mod slots. One block of 128 threads, 8 elements per thread.
extern "C" __global__ void __launch_bounds__(128)
kv_append_bf16(const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
               const unsigned* position, unsigned slots) {
    const size_t i = threadIdx.x * 8;
    const size_t dst = size_t(*position % slots) * kKvRow + i;
    *reinterpret_cast<uint4*>(k_cache + dst) = *reinterpret_cast<const uint4*>(k + i);
    *reinterpret_cast<uint4*>(v_cache + dst) = *reinterpret_cast<const uint4*>(v + i);
}

// First pass: for query head h = blockIdx.x and block b = blockIdx.y, an online
// softmax over cache slots b, b + blocks, ... below n, where `blocks` is the
// split for this position and blocks b >= blocks exit. One warp; lane l owns
// dims 4l..4l+3. Writes the unnormalized output to partial_out[h, b, 128]
// (rounded to BF16 values in the two-pass split), and the running max and sum
// to partial_max[h, b] and partial_sum[h, b].
//
// `*position_ptr` is the query's absolute position p, `slots` the cache size,
// `window` the sliding window (0 for none) and `extent` the bias table width.
extern "C" __global__ void __launch_bounds__(32)
decode_attention_partial_bf16(const __nv_bfloat16* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                              const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out,
                              float* partial_max, float* partial_sum, const unsigned* position_ptr,
                              unsigned slots, unsigned window, unsigned extent, float scale) {
    const unsigned position = *position_ptr;
    const int h = blockIdx.x;
    const int block = blockIdx.y;
    const int blocks = split_blocks(position, slots);
    if (block >= blocks) {
        return;
    }
    const bool round_partials = blocks == kTwoPassBlocks;
    const int lane = threadIdx.x;
    const int kv_head = h / kGroup;
    const unsigned n = min(position + 1, slots);
    const unsigned newest = position % slots;

    float qs[kVec];
    load_bf16x4(q + h * kHeadDim + lane * kVec, qs);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        qs[i] = fmul(scale, qs[i]);
    }

    float m = -FLT_MAX;
    float sum = 0.f;
    float o[kVec] = {0.f, 0.f, 0.f, 0.f};
    if (block == 0) {
        m = __bfloat162float(sinks[h]);
        sum = 1.f;
    }

    for (unsigned s = block; s < n; s += blocks) {
        const unsigned distance = (newest + slots - s) % slots;
        if (window != 0 && distance >= window) {
            continue;
        }
        const size_t offset = size_t(s) * kKvRow + kv_head * kHeadDim + lane * kVec;
        float kv[kVec];
        load_bf16x4(k_cache + offset, kv);
        float score = 0.f;
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            score = ffma(qs[i], kv[i], score);
        }
        score = warp_sum(score);
        if (distance < extent) {
            score = fadd(score, __bfloat162float(bias[h * extent + distance]));
        }

        const float new_m = fmaxf(m, score);
        const float factor = fast_exp(fsub(m, new_m));
        const float e = fast_exp(fsub(score, new_m));
        m = new_m;
        sum = ffma(sum, factor, e);
        load_bf16x4(v_cache + offset, kv);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            o[i] = ffma(o[i], factor, fmul(e, kv[i]));
        }
    }

    const size_t row = size_t(h) * blocks + block;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        partial_out[row * kHeadDim + lane * kVec + i] = round_partials ? round_bf16(o[i]) : o[i];
    }
    if (lane == 0) {
        partial_max[row] = m;
        partial_sum[row] = sum;
    }
}

// Second pass, MLX's sdpa_vector_2pass_2: combines the `blocks` partials of
// query head h = blockIdx.x into out[h, 128], with `blocks` the split for
// *position over `slots`, as in the first pass. 32 warps; warp w sums blocks
// w, w + 32, ... for dims 4l..4l+3 of lane l, then a transpose through shared
// memory lets each warp add up one group of 4 dims across the 32 warps. With 32
// blocks this is also exactly the end of MLX's one-pass kernel.
extern "C" __global__ void __launch_bounds__(1024)
decode_attention_combine_bf16(const float* partial_out, const float* partial_max, const float* partial_sum,
                              __nv_bfloat16* out, const unsigned* position, unsigned slots) {
    __shared__ float outputs[32 * 32];
    const int blocks = split_blocks(*position, slots);
    const int h = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const float* maxs = partial_max + size_t(h) * blocks;
    const float* sums = partial_sum + size_t(h) * blocks;

    float m = -FLT_MAX;
    for (int b = lane; b < blocks; b += 32) {
        m = fmaxf(m, maxs[b]);
    }
    m = warp_max(m);

    float sum = 0.f;
    for (int b = lane; b < blocks; b += 32) {
        sum = ffma(fast_exp(fsub(maxs[b], m)), sums[b], sum);
    }
    sum = warp_sum(sum);

    float o[kVec] = {0.f, 0.f, 0.f, 0.f};
    for (int b = warp; b < blocks; b += 32) {
        const float factor = fast_exp(fsub(maxs[b], m));
        const float* p = partial_out + (size_t(h) * blocks + b) * kHeadDim + lane * kVec;
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            o[i] = ffma(factor, p[i], o[i]);
        }
    }

#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        outputs[lane * 32 + warp] = o[i];
        __syncthreads();
        o[i] = warp_sum(outputs[warp * 32 + lane]);
        o[i] = sum == 0.f ? o[i] : fdiv(o[i], sum);
        __syncthreads();
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            out[h * kHeadDim + warp * kVec + i] = __float2bfloat16(o[i]);
        }
    }
}
