// Decode attention with relative-position bias, attention sinks and KV caches.
//
// Shapes are those of every Inkling layer: 24 query heads, 8 KV heads (GQA
// groups of 3), head dim 128, relative width 16.
//
// KV cache layout: keys and values are separate buffers of shape [slots, 8, 128],
// so appending one token writes one contiguous row into each. The elements are
// BF16, or E4M3 for an FP8 global cache (see `kv_append_fp8`).
// A sliding layer's cache is a ring of `slots` = window entries and position p
// lives in slot p mod slots. A global layer's cache stores position p in slot p
// and never wraps. Both cases share one rule: at query position p the cache
// holds n = min(p + 1, slots) keys in slots 0..n-1, and slot s holds the key at
// distance (p - s) mod slots. This matches MLX's RotatingKVCache and KVCache
// when they are fed one token at a time.
//
// `relative_bias` is bit-exact with MLX: its BF16 matmul for M = 24, K = 16
// sums the 16 exact products as a fixed tree, found by probing it with
// cancelling inputs: pairs (k, k + 1), then pair k with pair k + 8, then the
// four results in order. The PTX .rn.ftz helpers keep nvcc from fusing or
// reordering that tree; .ftz because MLX flushes subnormals.
//
// `decode_attention_bf16` and `decode_attention_fp8` compute, for each query
// head h with KV head g:
//   q' = scale * q                          (F32)
//   score_s = q' . k_s + bias[h, d_s]       (bias only for distance d_s < extent)
//   out = sum_s e^(score_s) v_s / (e^sink + sum_s e^(score_s))
// over the keys with distance below the window (all of them for a global
// layer). It is memory bound, so it reads each KV head once for its 3 query
// heads and keeps enough loads in flight to stream the cache at DRAM speed.
// Only the summation order differs from MLX's sdpa_vector; every sum is F32.
// The FP8 variant runs the same arithmetic on the cache values decoded exactly
// to F32, over tiles of 64 keys so that a tile holds the same 16 KiB.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <cstdint>

namespace {

constexpr int kHeads = 24;
constexpr int kKvHeads = 8;
constexpr int kGroup = kHeads / kKvHeads;
constexpr int kHeadDim = 128;
constexpr int kRelDim = 16;
constexpr int kKvRow = kKvHeads * kHeadDim;

// Decode attention tiling. A block of 4 warps handles one KV head over one
// contiguous range of cache slots (a split), in tiles of 32 BF16 or 64 E4M3
// keys.
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
// Dims of a key each warp multiplies in the score phase.
constexpr int kWarpDims = kHeadDim / kWarps;
// Split rows the partial buffers hold per query head. The launcher keeps the
// grid's split count at or below this.
constexpr int kMaxSplits = 64;

__device__ __forceinline__ float fadd(float a, float b) {
    float r;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fmul(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fast_exp(float x) {
    return exp2f(x * 1.4426950408889634f);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

// The 8 BF16 values packed in a 16-byte word, as F32.
__device__ __forceinline__ void unpack_bf16x8(uint4 raw, float (&v)[8]) {
    const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = __uint_as_float(words[i] << 16);
        v[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
    }
}

__device__ __forceinline__ void unpack_bf16x4(uint2 raw, float (&v)[4]) {
    v[0] = __uint_as_float(raw.x << 16);
    v[1] = __uint_as_float(raw.x & 0xffff0000u);
    v[2] = __uint_as_float(raw.y << 16);
    v[3] = __uint_as_float(raw.y & 0xffff0000u);
}

// Two E4M3 values (the low byte is the first) as F32, exact: the hardware
// converts E4M3 pairs to F16 pairs.
__device__ __forceinline__ float2 e4m3x2_to_float2(uint32_t pair) {
    return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3)));
}

__device__ __forceinline__ void unpack_e4m3x4(uint32_t raw, float (&v)[4]) {
    const float2 a = e4m3x2_to_float2(raw & 0xffffu);
    const float2 b = e4m3x2_to_float2(raw >> 16);
    v[0] = a.x;
    v[1] = a.y;
    v[2] = b.x;
    v[3] = b.y;
}

// The 16 E4M3 values packed in a 16-byte word, as F32.
__device__ __forceinline__ void unpack_e4m3x16(uint4 raw, float (&v)[16]) {
    const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float w[4];
        unpack_e4m3x4(words[i], w);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[4 * i + j] = w[j];
        }
    }
}

// Four floats rounded to E4M3 (nearest even, saturated to +-448), first value
// in the low byte.
__device__ __forceinline__ uint32_t pack_e4m3x4(float a, float b, float c, float d) {
    const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return lo | hi << 16;
}

// Cache element formats of decode attention. `Elem` is the stored type, `kTile`
// the keys per tile and `kChunkElems` the elements in 16 bytes. `unpack_chunk`
// decodes 16 bytes and `unpack4` the 4 elements at `p`.
struct Bf16Format {
    using Elem = __nv_bfloat16;
    static constexpr int kTile = 32;
    static constexpr int kChunkElems = 8;
    __device__ static void unpack_chunk(uint4 raw, float (&v)[kChunkElems]) { unpack_bf16x8(raw, v); }
    __device__ static void unpack4(const Elem* p, float (&v)[4]) {
        unpack_bf16x4(*reinterpret_cast<const uint2*>(p), v);
    }
};

struct Fp8Format {
    using Elem = uint8_t;
    static constexpr int kTile = 64;
    static constexpr int kChunkElems = 16;
    __device__ static void unpack_chunk(uint4 raw, float (&v)[kChunkElems]) { unpack_e4m3x16(raw, v); }
    __device__ static void unpack4(const Elem* p, float (&v)[4]) {
        unpack_e4m3x4(*reinterpret_cast<const uint32_t*>(p), v);
    }
};

// Copies 16 bytes from global to shared memory without blocking, or writes 16
// zero bytes when `valid` is false.
__device__ __forceinline__ void copy_async16(void* dst, const void* src, bool valid) {
    const auto shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(shared), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void copy_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void copy_async_wait_all() {
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

template <class F>
struct alignas(16) DecodeShared {
    static constexpr int kTile = F::kTile;
    // 16-byte chunks in one key or value head.
    static constexpr int kChunks = kHeadDim / F::kChunkElems;
    // Two stages of key and value tiles, [kTile, 128] each. Key rows are
    // swizzled: logical chunk c of row r sits at chunk c ^ (r % kChunks), so
    // that the lanes of a quarter warp, reading the same logical chunk of
    // consecutive rows, hit distinct banks.
    typename F::Elem k[2][kTile * kHeadDim];
    typename F::Elem v[2][kTile * kHeadDim];
    // Per-warp partial scores over each warp's quarter of the dims.
    float scores[kWarps][kGroup][kTile];
    // Softmax weights of the tile's keys for the 3 heads, padded to 4.
    float4 weights[kTile];
    // Rescale factor of each head's accumulators for the current tile.
    float factor[kGroup];
    // Combined softmax denominator of each head, in the combining block.
    float total[kGroup];
    bool last;
};

// Starts copying the key and value tile that begins at slot `first`. Slots at
// or past `end` read as zeros.
template <class F>
__device__ __forceinline__ void load_tile(DecodeShared<F>& sm, int stage, const typename F::Elem* k_cache,
                                          const typename F::Elem* v_cache, int kv_head, unsigned first,
                                          unsigned end) {
    constexpr int kTile = F::kTile;
    constexpr int kChunks = DecodeShared<F>::kChunks;
    constexpr int kElems = F::kChunkElems;
#pragma unroll
    for (int k = 0; k < kTile * kChunks / kThreads; ++k) {
        const int i = threadIdx.x + k * kThreads;
        const int row = i / kChunks;
        const int chunk = i % kChunks;
        const unsigned slot = first + row;
        const bool valid = slot < end;
        const size_t offset = size_t(valid ? slot : 0) * kKvRow + kv_head * kHeadDim + chunk * kElems;
        copy_async16(&sm.k[stage][row * kHeadDim + (chunk ^ (row % kChunks)) * kElems], k_cache + offset, valid);
        copy_async16(&sm.v[stage][row * kHeadDim + chunk * kElems], v_cache + offset, valid);
    }
    copy_async_commit();
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

// The FP8 variant of `kv_append_bf16`: rounds each BF16 value to E4M3, nearest
// even and saturated to +-448, without a scale. Keys are L2-clipped to norm
// sqrt(512) and values RMS-normed per head, so no element exceeds 22.7 and
// saturation never happens. E4M3 keeps 3 mantissa bits (relative error at most
// 2^-4) down to 2^-6; smaller elements round to multiples of 2^-9.
extern "C" __global__ void __launch_bounds__(128)
kv_append_fp8(const __nv_bfloat16* k, const __nv_bfloat16* v, uint8_t* k_cache, uint8_t* v_cache,
              const unsigned* position, unsigned slots) {
    const size_t i = threadIdx.x * 8;
    const size_t dst = size_t(*position % slots) * kKvRow + i;
    float x[8];
    unpack_bf16x8(*reinterpret_cast<const uint4*>(k + i), x);
    *reinterpret_cast<uint2*>(k_cache + dst) =
        make_uint2(pack_e4m3x4(x[0], x[1], x[2], x[3]), pack_e4m3x4(x[4], x[5], x[6], x[7]));
    unpack_bf16x8(*reinterpret_cast<const uint4*>(v + i), x);
    *reinterpret_cast<uint2*>(v_cache + dst) =
        make_uint2(pack_e4m3x4(x[0], x[1], x[2], x[3]), pack_e4m3x4(x[4], x[5], x[6], x[7]));
}

namespace {

// Decode attention of one token over a KV cache, for all 24 query heads.
//
// Block (g, j) handles KV head g = blockIdx.x and its query heads 3g..3g+2
// over split j = blockIdx.y of the cache slots. The n = min(p + 1, slots) live
// slots are cut into gridDim.y contiguous splits of whole tiles; splits past
// the last live slot hold no keys. The launch therefore stays the same at every
// position p = *position_ptr, and a captured graph serves them all.
//
// Within a block, for each tile of T = 32 (BF16) or 64 (E4M3) keys:
// 1. Warp w multiplies dims 32w..32w+31 of the 3 queries with the T keys,
//    lane l taking keys l, l + 32, ...
// 2. Warp h < 3 adds the 4 partial scores and the bias of head h, and updates
//    head h's running max over the split. Lane l keeps a running sum over its
//    own keys.
// 3. Every warp rescales its accumulators [3 heads, 4 dims per lane] by the
//    change in each head's max, then adds the weighted values of T / 4 of the
//    tile's keys.
// The next tile's copy runs during the current tile's arithmetic.
//
// Each block writes its unnormalized output, max and sum per head to
// partial_out[h, j, 128], partial_max[h, j] and partial_sum[h, j] (F32). The
// last block of KV head g to finish, counted in counters[g], combines the
// splits of heads 3g..3g+2 into out[h, 128] and resets counters[g] to zero for
// the next call. The sink enters split 0 as a key with score sinks[h] and a
// zero value.
//
// `slots` is the cache size, `window` the sliding window (0 for none) and
// `extent` the bias table width.
template <class F>
__device__ __forceinline__ void decode_attention(const __nv_bfloat16* q, const typename F::Elem* k_cache,
                                                 const typename F::Elem* v_cache, const __nv_bfloat16* bias,
                                                 const __nv_bfloat16* sinks, float* partial_out, float* partial_max,
                                                 float* partial_sum, unsigned* counters, __nv_bfloat16* out,
                                                 const unsigned* position_ptr, unsigned slots, unsigned window,
                                                 unsigned extent, float scale) {
    constexpr int kTile = F::kTile;
    // Keys of a tile each lane scores, and each warp accumulates in the value
    // phase.
    constexpr int kLaneKeys = kTile / 32;
    constexpr int kWarpKeys = kTile / kWarps;
    constexpr int kElems = F::kChunkElems;
    constexpr int kChunks = DecodeShared<F>::kChunks;
    __shared__ DecodeShared<F> sm;
    const unsigned position = *position_ptr;
    const int kv_head = blockIdx.x;
    const int split = blockIdx.y;
    const int splits = gridDim.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const unsigned n = min(position + 1, slots);
    const unsigned newest = position % slots;
    // Slots per split: an even share of n, rounded up to whole tiles.
    const unsigned share = (n + splits - 1) / splits;
    const unsigned chunk = (share + kTile - 1) / kTile * kTile;
    const int used = (n + chunk - 1) / chunk;
    // The 3 query heads of this block are rows head0..head0+2.
    const int head0 = kv_head * kGroup;
    // Score and softmax state of head h = warp, for warps 0..2.
    const int head = head0 + min(warp, kGroup - 1);

    if (split < used) {
        const unsigned begin = split * chunk;
        const unsigned end = min(begin + chunk, n);
        const int tiles = (end - begin + kTile - 1) / kTile;
        load_tile(sm, 0, k_cache, v_cache, kv_head, begin, end);

        // This warp's quarter of the 3 scaled queries.
        float qs[kGroup][kWarpDims];
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
#pragma unroll
            for (int c = 0; c < kWarpDims / 8; ++c) {
                const uint4 raw =
                    *reinterpret_cast<const uint4*>(q + (head0 + h) * kHeadDim + warp * kWarpDims + c * 8);
                float vals[8];
                unpack_bf16x8(raw, vals);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    qs[h][c * 8 + i] = scale * vals[i];
                }
            }
        }

        float m = -FLT_MAX;
        float sum = 0.f;
        if (split == 0) {
            m = __bfloat162float(sinks[head]);
            sum = lane == 0 ? 1.f : 0.f;
        }
        float o[kGroup][4] = {};

        for (int t = 0; t < tiles; ++t) {
            const int stage = t % 2;
            copy_async_wait_all();
            __syncthreads();
            if (t + 1 < tiles) {
                load_tile(sm, 1 - stage, k_cache, v_cache, kv_head, begin + (t + 1) * kTile, end);
            }

            // 1. Partial scores of keys `lane + 32 j` over this warp's dims.
#pragma unroll
            for (int j = 0; j < kLaneKeys; ++j) {
                const int row = lane + 32 * j;
                const typename F::Elem* keys = sm.k[stage] + row * kHeadDim;
                float acc[kGroup] = {};
#pragma unroll
                for (int c = 0; c < kWarpDims / kElems; ++c) {
                    const int chunk_index = warp * (kWarpDims / kElems) + c;
                    const uint4 raw = *reinterpret_cast<const uint4*>(keys + (chunk_index ^ (row % kChunks)) * kElems);
                    float kv[kElems];
                    F::unpack_chunk(raw, kv);
#pragma unroll
                    for (int i = 0; i < kElems; ++i) {
#pragma unroll
                        for (int h = 0; h < kGroup; ++h) {
                            acc[h] = fmaf(qs[h][c * kElems + i], kv[i], acc[h]);
                        }
                    }
                }
#pragma unroll
                for (int h = 0; h < kGroup; ++h) {
                    sm.scores[warp][h][row] = acc[h];
                }
            }
            __syncthreads();

            // 2. Softmax weights of head `warp`.
            if (warp < kGroup) {
                float score[kLaneKeys];
#pragma unroll
                for (int j = 0; j < kLaneKeys; ++j) {
                    const int row = lane + 32 * j;
                    const unsigned slot = begin + t * kTile + row;
                    const unsigned distance = (newest + slots - slot) % slots;
                    score[j] = -INFINITY;
                    if (slot < end && (window == 0 || distance < window)) {
                        score[j] = (sm.scores[0][warp][row] + sm.scores[1][warp][row]) +
                                   (sm.scores[2][warp][row] + sm.scores[3][warp][row]);
                        if (distance < extent) {
                            score[j] += __bfloat162float(bias[head * extent + distance]);
                        }
                    }
                }
                float tile_max = score[0];
#pragma unroll
                for (int j = 1; j < kLaneKeys; ++j) {
                    tile_max = fmaxf(tile_max, score[j]);
                }
                const float new_m = fmaxf(m, warp_max(tile_max));
                const float factor = fast_exp(m - new_m);
#pragma unroll
                for (int j = 0; j < kLaneKeys; ++j) {
                    const float weight = fast_exp(score[j] - new_m);
                    sum = j == 0 ? fmaf(sum, factor, weight) : sum + weight;
                    reinterpret_cast<float*>(&sm.weights[lane + 32 * j])[warp] = weight;
                }
                m = new_m;
                if (lane == 0) {
                    sm.factor[warp] = factor;
                }
            }
            __syncthreads();

            // 3. Weighted values of keys 8w..8w+7, dims 4l..4l+3.
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                const float factor = sm.factor[h];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o[h][i] *= factor;
                }
            }
            const typename F::Elem* values = sm.v[stage];
#pragma unroll
            for (int r = warp * kWarpKeys; r < (warp + 1) * kWarpKeys; ++r) {
                float v[4];
                F::unpack4(values + r * kHeadDim + lane * 4, v);
                const float4 w = sm.weights[r];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    o[0][i] = fmaf(w.x, v[i], o[0][i]);
                    o[1][i] = fmaf(w.y, v[i], o[1][i]);
                    o[2][i] = fmaf(w.z, v[i], o[2][i]);
                }
            }
        }

        // Sum the 4 warps' accumulators and write this split's partials.
        __syncthreads();
        static_assert(sizeof(sm.k[0]) >= kWarps * kGroup * kHeadDim * sizeof(float));
        auto* reduce = reinterpret_cast<float(*)[kGroup][kHeadDim]>(sm.k[0]);
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                reduce[warp][h][lane * 4 + i] = o[h][i];
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const float* r = &reduce[0][h][threadIdx.x];
            const float value = (r[0] + r[kGroup * kHeadDim]) + (r[2 * kGroup * kHeadDim] + r[3 * kGroup * kHeadDim]);
            partial_out[(size_t(head0 + h) * kMaxSplits + split) * kHeadDim + threadIdx.x] = value;
        }
        if (warp < kGroup) {
            sum = warp_sum(sum);
            if (lane == 0) {
                partial_max[head * kMaxSplits + split] = m;
                partial_sum[head * kMaxSplits + split] = sum;
            }
        }
    }

    // Count finished blocks; the last one of this KV head combines.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        sm.last = atomicAdd(&counters[kv_head], 1u) == unsigned(splits - 1);
    }
    __syncthreads();
    if (!sm.last) {
        return;
    }
    __threadfence();

    // Rescale factors of the used splits, per head, and the combined sums.
    auto* factors = reinterpret_cast<float(*)[kMaxSplits]>(sm.v[0]);
    if (warp < kGroup) {
        float maxs[2];
        float sums[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int j = lane + 32 * i;
            maxs[i] = j < used ? __ldcg(&partial_max[head * kMaxSplits + j]) : -FLT_MAX;
            sums[i] = j < used ? __ldcg(&partial_sum[head * kMaxSplits + j]) : 0.f;
        }
        const float m = warp_max(fmaxf(maxs[0], maxs[1]));
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const float factor = lane + 32 * i < used ? fast_exp(maxs[i] - m) : 0.f;
            factors[warp][lane + 32 * i] = factor;
            sum = fmaf(factor, sums[i], sum);
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            sm.total[warp] = sum;
        }
    }
    __syncthreads();

    // Warp w sums splits w, w + 4, ... for dims 4l..4l+3 of each head.
    float o[kGroup][4] = {};
    for (int j = warp; j < used; j += kWarps) {
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const float4 p =
                __ldcg(reinterpret_cast<const float4*>(partial_out + (size_t(head0 + h) * kMaxSplits + j) * kHeadDim) +
                       lane);
            const float factor = factors[h][j];
            o[h][0] = fmaf(factor, p.x, o[h][0]);
            o[h][1] = fmaf(factor, p.y, o[h][1]);
            o[h][2] = fmaf(factor, p.z, o[h][2]);
            o[h][3] = fmaf(factor, p.w, o[h][3]);
        }
    }
    auto* reduce = reinterpret_cast<float(*)[kGroup][kHeadDim]>(sm.k[0]);
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            reduce[warp][h][lane * 4 + i] = o[h][i];
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        const float* r = &reduce[0][h][threadIdx.x];
        const float value = (r[0] + r[kGroup * kHeadDim]) + (r[2 * kGroup * kHeadDim] + r[3 * kGroup * kHeadDim]);
        const float total = sm.total[h];
        out[(head0 + h) * kHeadDim + threadIdx.x] = __float2bfloat16(total == 0.f ? value : value / total);
    }
    if (threadIdx.x == 0) {
        counters[kv_head] = 0;
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads)
decode_attention_bf16(const __nv_bfloat16* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                      const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out, float* partial_max,
                      float* partial_sum, unsigned* counters, __nv_bfloat16* out, const unsigned* position_ptr,
                      unsigned slots, unsigned window, unsigned extent, float scale) {
    decode_attention<Bf16Format>(q, k_cache, v_cache, bias, sinks, partial_out, partial_max, partial_sum, counters,
                                 out, position_ptr, slots, window, extent, scale);
}

// `decode_attention_bf16` over E4M3 caches written by `kv_append_fp8`.
extern "C" __global__ void __launch_bounds__(kThreads)
decode_attention_fp8(const __nv_bfloat16* q, const uint8_t* k_cache, const uint8_t* v_cache,
                     const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out, float* partial_max,
                     float* partial_sum, unsigned* counters, __nv_bfloat16* out, const unsigned* position_ptr,
                     unsigned slots, unsigned window, unsigned extent, float scale) {
    decode_attention<Fp8Format>(q, k_cache, v_cache, bias, sinks, partial_out, partial_max, partial_sum, counters,
                                out, position_ptr, slots, window, extent, scale);
}
