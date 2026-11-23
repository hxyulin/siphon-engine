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
#include <climits>
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

// sum_k r[k] * t[k] for the 16 relative dims, summed as MLX's BF16 matmul does
// (see the top). Products of two BF16 values are exact in F32.
__device__ __forceinline__ float relative_bias_dot(const float (&r)[kRelDim], const float (&t)[kRelDim]) {
    float x[kRelDim];
#pragma unroll
    for (int k = 0; k < kRelDim; ++k) {
        x[k] = fmul(r[k], t[k]);
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
    return fadd(fadd(fadd(quad[0], quad[1]), quad[2]), quad[3]);
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
    float a[kRelDim];
    float b[kRelDim];
#pragma unroll
    for (int k = 0; k < kRelDim; ++k) {
        a[k] = __bfloat162float(rel[h * kRelDim + k]);
        b[k] = __bfloat162float(table[size_t(k) * extent + r]);
    }
    bias[h * extent + r] = __float2bfloat16(relative_bias_dot(a, b));
}

// The bias of `rows` = gridDim.x rows at once: rel [rows, 24, 16] and table
// [16, extent] give bias [rows, 24, extent], each value as relative_bias_bf16
// computes it, bit for bit. Block i takes row i. Thread t keeps table columns
// 4t..4t+3 (and those 4 * blockDim.x further on) in registers and computes
// them for all 24 heads, writing 4 values at a time. `extent` is a multiple
// of 4.
extern "C" __global__ void __launch_bounds__(256)
relative_bias_rows_bf16(const __nv_bfloat16* rel, const __nv_bfloat16* table, __nv_bfloat16* bias, int extent) {
    __shared__ __align__(16) float rel_s[kHeads * kRelDim];
    const size_t row = blockIdx.x;
    for (int i = threadIdx.x; i < kHeads * kRelDim; i += blockDim.x) {
        rel_s[i] = __bfloat162float(rel[row * kHeads * kRelDim + i]);
    }
    __syncthreads();
    for (int c = 4 * threadIdx.x; c < extent; c += 4 * blockDim.x) {
        float t[4][kRelDim];
#pragma unroll
        for (int k = 0; k < kRelDim; ++k) {
            float v[4];
            unpack_bf16x4(*reinterpret_cast<const uint2*>(table + size_t(k) * extent + c), v);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                t[j][k] = v[j];
            }
        }
#pragma unroll 2
        for (int h = 0; h < kHeads; ++h) {
            float r[kRelDim];
#pragma unroll
            for (int k = 0; k < kRelDim; k += 4) {
                const float4 x = *reinterpret_cast<const float4*>(&rel_s[h * kRelDim + k]);
                r[k] = x.x;
                r[k + 1] = x.y;
                r[k + 2] = x.z;
                r[k + 3] = x.w;
            }
            uint32_t packed[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const __nv_bfloat162 pair =
                    __floats2bfloat162_rn(relative_bias_dot(r, t[2 * j]), relative_bias_dot(r, t[2 * j + 1]));
                packed[j] = *reinterpret_cast<const uint32_t*>(&pair);
            }
            *reinterpret_cast<uint2*>(bias + (row * kHeads + h) * extent + c) = make_uint2(packed[0], packed[1]);
        }
    }
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

// Chunked prefill: attention of m consecutive query rows, at positions
// start..start+m-1, over a KV cache plus the chunk's own keys.
//
// Row i sees exactly what a decode step at position start + i sees after its
// append: keys at positions max(0, start + i - window + 1)..start + i (from 0
// for a global layer), with the same bias, sink and scale. Positions below
// `cache_end` come from the cache, at slot p mod slots. Positions from
// `cache_end` on come from the chunk's input rows k_in and v_in, row p - start.
// The launcher passes cache_end = start for a sliding cache, whose ring the
// chunk's append would overwrite, and appends after this kernel; it appends
// first and passes cache_end = start + m for a global cache, which never wraps.
// A chunk may be longer than the window: its late rows then read only input
// rows.
//
// Tensor cores do both products with mma.sync.m16n8k16 and F32 accumulators:
// S = Q K^T in BF16 and O += P V in F16, and the softmax runs online in F32
// (exp2 domain). The 3 query heads of KV head g share its keys, so they form
// the M dimension together: M row 3i + h is query row i, head 3g + h. A block
// of 8 warps takes 128 M rows, 16 per warp, and walks their key range in tiles
// of 64 keys, copying tile t + 1 while it computes tile t. Each warp copies the
// bias of its own rows for the next tile once it has read the current one.
// Tiles that every row sees in full skip the mask, and tiles beyond the bias
// extent skip the bias. E4M3 tiles are copied to a staging buffer and decoded
// with the hardware converter in shared memory, keys to BF16 and values to
// F16; decoding is exact. The block needs 84 KB of shared memory and up to 255
// registers per thread, so one block runs per SM.
//
// Differences from decode attention, all inside F32 rounding except the
// second: the scale multiplies the F32 dot product instead of q, and P is
// rounded to F16 for the P V product (relative error at most 2^-11 per weight
// down to 2^-29 of the largest weight, an eighth of BF16's 2^-8, at the same
// tensor-core rate). V converts to F16 exactly from magnitude 2^-14 up, and
// within 2^-25 below. The softmax denominator sums the F32 weights.
//
// Schedule: one block per SM, persistent. A unit is (block of 128 M rows, KV
// head), and its work items are its key tiles. The items of all units, in
// order, are cut into min(gridDim.x, items) contiguous ranges of equal length
// (within one tile), one per block, so every SM computes the same number of
// tiles. A block walks its range one segment (the part of one unit inside the
// range) at a time. A segment that covers its whole unit writes `out`
// directly. Otherwise it writes its unnormalized output, max and sum (log2
// domain) to a partial slot, and the last segment of the unit to finish,
// counted in counters, adds the others to its own and writes `out`. Nothing
// waits for another block, so the schedule cannot deadlock when fewer blocks
// fit. Block b holds items begin(b)..begin(b + 1) - 1, and item x lies in block
// max{b : begin(b) <= x}. Only the first and last segment of a block can be
// partial; they use slots 2b and 2b + 1.

namespace {

constexpr int kPrefillKeys = 64;
// Most blocks of M rows per KV head: 2048 rows of 3 heads in blocks of 128.
constexpr int kPrefillMaxBlocks = 48;
// 64 consecutive distances span at most 9 aligned chunks of 8 bias values.
constexpr int kBiasChunks = kPrefillKeys / 8 + 1;
constexpr int kBiasStride = kBiasChunks * 8;
constexpr float kLog2e = 1.4426950408889634f;
// The softmax weights and sums carry a factor of 2^15, which cancels in O / l,
// so that F16 weights keep full precision down to 2^-29 of the largest one
// instead of 2^-14. The largest weight, 2^15, stays below F16's 65504.
constexpr float kPScaleLog2 = 15.f;
constexpr float kPScale = 32768.f;

// Element offset of 16-byte chunk `chunk` of row `row` in a [rows, 128] BF16
// tile. The XOR spreads the 8 rows an ldmatrix reads over distinct banks.
__device__ __forceinline__ int swizzle(int row, int chunk) {
    return row * kHeadDim + ((chunk ^ (row & 7)) << 3);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    const auto a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* p) {
    const auto a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

// c += a b for a 16x16 BF16 A fragment, a 16x8 BF16 B fragment and F32 c.
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 2^x, flushing subnormal results to zero; -inf gives 0.
__device__ __forceinline__ float exp2_ftz(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// c += a b for a 16x16 F16 A fragment, a 16x8 F16 B fragment and F32 c.
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// A BF16 pair as an F16 pair: exact for magnitudes from 2^-14 to 65504, and
// within 2^-25 below.
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t x) {
    return pack_f16x2(__uint_as_float(x << 16), __uint_as_float(x & 0xffff0000u));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// Two E4M3 values (first in the low byte) as a BF16 pair, exact.
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t pair) {
    const float2 f = e4m3x2_to_float2(pair);
    return pack_bf16x2(f.x, f.y);
}

// Two E4M3 values (first in the low byte) as an F16 pair, exact.
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t pair) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3);
    return uint32_t(h.x) | uint32_t(h.y) << 16;
}

// The 16 E4M3 values in `raw` as two 16-byte chunks of F16.
__device__ __forceinline__ void e4m3x16_to_f16(uint4 raw, uint4& lo, uint4& hi) {
    lo = make_uint4(e4m3x2_to_f16x2(raw.x & 0xffffu), e4m3x2_to_f16x2(raw.x >> 16), e4m3x2_to_f16x2(raw.y & 0xffffu),
                    e4m3x2_to_f16x2(raw.y >> 16));
    hi = make_uint4(e4m3x2_to_f16x2(raw.z & 0xffffu), e4m3x2_to_f16x2(raw.z >> 16), e4m3x2_to_f16x2(raw.w & 0xffffu),
                    e4m3x2_to_f16x2(raw.w >> 16));
}

// The 16 E4M3 values in `raw` as two 16-byte chunks of BF16.
__device__ __forceinline__ void e4m3x16_to_bf16(uint4 raw, uint4& lo, uint4& hi) {
    lo = make_uint4(e4m3x2_to_bf16x2(raw.x & 0xffffu), e4m3x2_to_bf16x2(raw.x >> 16),
                    e4m3x2_to_bf16x2(raw.y & 0xffffu), e4m3x2_to_bf16x2(raw.y >> 16));
    hi = make_uint4(e4m3x2_to_bf16x2(raw.z & 0xffffu), e4m3x2_to_bf16x2(raw.z >> 16),
                    e4m3x2_to_bf16x2(raw.w & 0xffffu), e4m3x2_to_bf16x2(raw.w >> 16));
}

template <class F, int kWarps>
struct alignas(16) PrefillShared {
    static constexpr bool kFp8 = sizeof(typename F::Elem) == 1;
    static constexpr int kRows = 16 * kWarps;
    // BF16 key and value tiles, [64, 128] swizzled: two stages, or for E4M3 one
    // tile that each staged tile is decoded into.
    static constexpr int kStages = kFp8 ? 1 : 2;
    __nv_bfloat16 k[kStages][kPrefillKeys * kHeadDim];
    __nv_bfloat16 v[kStages][kPrefillKeys * kHeadDim];
    // E4M3 staging tiles, [64, 128] unswizzled (a stub for BF16).
    static constexpr int kStaged = kFp8 ? kPrefillKeys * kHeadDim : 16;
    uint8_t k8[2][kStaged];
    uint8_t v8[2][kStaged];
    // Bias values of each M row for the tile's distances: 9 aligned chunks
    // starting at the chunk that holds the smallest distance. Each warp loads
    // and reads only its own 16 rows.
    __nv_bfloat16 bias[kRows * kBiasStride];
    // tiles[b]: work items of all KV heads of the blocks of M rows before b.
    unsigned tiles[kPrefillMaxBlocks + 1];
    bool last;
};

// The keys of one block of M rows: positions lo..hi-1, those below a_end in
// a_tiles tiles from the cache and those from b_lo on from the input rows.
struct PrefillRange {
    unsigned r_lo;
    unsigned r_hi;
    unsigned lo;
    unsigned hi;
    unsigned a_end;
    unsigned a_tiles;
    unsigned b_lo;
    unsigned tiles;
};

// One tile of keys: positions p0..min(p0 + 64, end) - 1, from the input rows
// or the cache. `full` means every M row of the block sees every key, `biased`
// that some row has a key closer than the bias extent, and `near` that every
// row's keys all are.
struct PrefillTile {
    unsigned p0;
    unsigned end;
    bool input;
    bool full;
    bool biased;
    bool near;
};

// A block's part of one unit: items begin.., tiles t_begin..t_end - 1 of the
// unit's range.tiles, which start at item unit_begin.
struct PrefillSegment {
    PrefillRange range;
    unsigned begin;
    unsigned unit_begin;
    unsigned t_begin;
    unsigned t_end;
    int block_m;
    int kv_head;
};

template <class F, int kWarps>
__device__ __forceinline__ void prefill_attention(const __nv_bfloat16* q, const __nv_bfloat16* k_in,
                                                  const __nv_bfloat16* v_in, const typename F::Elem* k_cache,
                                                  const typename F::Elem* v_cache, const __nv_bfloat16* bias,
                                                  const __nv_bfloat16* sinks, float* partials, unsigned* counters, __nv_bfloat16* out,
                                                  const unsigned* start_ptr, unsigned rows, unsigned slots,
                                                  unsigned window, unsigned extent, unsigned input_keys, float scale) {
    using Shared = PrefillShared<F, kWarps>;
    constexpr int kThreads = kWarps * 32;
    constexpr int kRows = Shared::kRows;
    constexpr bool kFp8 = Shared::kFp8;
    // Floats of one partial slot: [kWarps, 16 n-tiles, 32 lanes] float4 of
    // output fragments, then the max and the sum of each M row.
    constexpr int kSlot = kRows * (kHeadDim + 2);
    extern __shared__ __align__(16) unsigned char prefill_smem[];
    Shared& sm = *reinterpret_cast<Shared*>(prefill_smem);

    const unsigned start = *start_ptr;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const unsigned total_m = rows * kGroup;
    const int blocks_m = int((total_m + kRows - 1) / kRows);
    const unsigned cache_end = input_keys ? start : start + rows;

    auto range_of = [&](int block_m) {
        PrefillRange r;
        const unsigned m0 = block_m * kRows;
        r.r_lo = m0 / kGroup;
        r.r_hi = (min(m0 + kRows, total_m) - 1) / kGroup;
        r.hi = start + r.r_hi + 1;
        r.lo = window != 0 && start + r.r_lo + 1 > window ? start + r.r_lo + 1 - window : 0;
        r.a_end = min(r.hi, cache_end);
        r.a_tiles = r.a_end > r.lo ? (r.a_end - r.lo + kPrefillKeys - 1) / kPrefillKeys : 0;
        r.b_lo = max(r.lo, cache_end);
        const unsigned b_tiles = r.hi > r.b_lo ? (r.hi - r.b_lo + kPrefillKeys - 1) / kPrefillKeys : 0;
        r.tiles = r.a_tiles + b_tiles;
        return r;
    };

    // Work item (block of M rows b, KV head g, tile t) has index
    // tiles[b] + g * count(b) + t. Lane l of warp 0 counts blocks 2l and
    // 2l + 1 and the warp scans the counts.
    if (warp == 0) {
        const unsigned c0 = 2 * lane < blocks_m ? range_of(2 * lane).tiles * kKvHeads : 0;
        const unsigned c1 = 2 * lane + 1 < blocks_m ? range_of(2 * lane + 1).tiles * kKvHeads : 0;
        unsigned inclusive = c0 + c1;
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const unsigned v = __shfl_up_sync(0xffffffffu, inclusive, offset);
            inclusive += lane >= offset ? v : 0;
        }
        const unsigned exclusive = inclusive - c0 - c1;
        if (2 * lane <= blocks_m) {
            sm.tiles[2 * lane] = exclusive;
        }
        if (2 * lane + 1 <= blocks_m) {
            sm.tiles[2 * lane + 1] = exclusive + c0;
        }
    }
    __syncthreads();

    // Every block that takes part gets at least one item, so each segment's
    // block holds a piece of its unit.
    const unsigned total = sm.tiles[blocks_m];
    const unsigned grid = min(gridDim.x, total);
    if (blockIdx.x >= grid) {
        return;
    }
    auto begin_of = [&](unsigned b) { return unsigned((unsigned long long)b * total / grid); };
    auto block_of = [&](unsigned x) { return unsigned(((unsigned long long)(x + 1) * grid - 1) / total); };
    const unsigned my_begin = begin_of(blockIdx.x);
    const unsigned my_end = begin_of(blockIdx.x + 1);
    const float score_scale = scale * kLog2e;

    // The segment that starts at item x: its block of M rows, KV head and
    // tiles. The blocks of M rows only increase along a block's range.
    int block_m = 0;
    auto segment_at = [&](unsigned x) {
        PrefillSegment g;
        while (sm.tiles[block_m + 1] <= x) {
            ++block_m;
        }
        g.block_m = block_m;
        g.range = range_of(block_m);
        g.kv_head = int((x - sm.tiles[block_m]) / g.range.tiles);
        g.unit_begin = sm.tiles[block_m] + g.kv_head * g.range.tiles;
        g.begin = x;
        g.t_begin = x - g.unit_begin;
        g.t_end = min(my_end, g.unit_begin + g.range.tiles) - g.unit_begin;
        return g;
    };

    auto tile_at = [&](const PrefillSegment& g, unsigned t) {
        const PrefillRange& range = g.range;
        PrefillTile tile;
        tile.input = t >= range.a_tiles;
        tile.p0 = tile.input ? range.b_lo + (t - range.a_tiles) * kPrefillKeys : range.lo + t * kPrefillKeys;
        tile.end = tile.input ? range.hi : range.a_end;
        const unsigned last = tile.p0 + kPrefillKeys - 1;
        tile.full = last < tile.end && last <= start + range.r_lo &&
                    (window == 0 || start + range.r_hi - tile.p0 < window);
        tile.biased = int(start + range.r_lo) - int(last) < int(extent);
        tile.near = start + range.r_hi - tile.p0 < extent;
        return tile;
    };

    // Copies one BF16 [64, 128] tile of keys or values from `src` into `dst`,
    // swizzled, as thread `thread` of `threads`.
    auto copy_bf16 = [&](const PrefillSegment& g, const PrefillTile& tile, __nv_bfloat16* dst,
                         const __nv_bfloat16* src, int thread, int threads) {
        const unsigned first = tile.input ? tile.p0 - start : tile.p0 % slots;
        const unsigned wrap = tile.input ? 0xffffffffu : slots;
        constexpr int kChunks = kHeadDim / 8;
        const int chunk = thread % kChunks;
#pragma unroll 4
        for (int row = thread / kChunks; row < kPrefillKeys; row += threads / kChunks) {
            const bool valid = tile.p0 + row < tile.end;
            unsigned source = first + row;
            source = source >= wrap ? source - wrap : source;
            const size_t offset = size_t(valid ? source : 0) * kKvRow + g.kv_head * kHeadDim + chunk * 8;
            copy_async16(&dst[swizzle(row, chunk)], src + offset, valid);
        }
    };

    // Starts copying a tile's keys and values into stage `stage`.
    auto load = [&](const PrefillSegment& g, const PrefillTile& tile, int stage) {
        // Source row of the tile's first key, and the ring size it wraps at.
        // A tile has fewer keys than the ring has slots, so it wraps at most
        // once.
        const unsigned first = tile.input ? tile.p0 - start : tile.p0 % slots;
        const unsigned wrap = tile.input ? 0xffffffffu : slots;
        if constexpr (kFp8) {
            // 8 chunks of 16 E4M3 values per row.
            constexpr int kChunks = kHeadDim / 16;
            const int chunk = threadIdx.x % kChunks;
#pragma unroll
            for (int row = threadIdx.x / kChunks; row < kPrefillKeys; row += kThreads / kChunks) {
                const bool valid = tile.p0 + row < tile.end;
                unsigned source = first + row;
                source = source >= wrap ? source - wrap : source;
                const size_t offset = size_t(valid ? source : 0) * kKvRow + g.kv_head * kHeadDim + chunk * 16;
                copy_async16(&sm.k8[stage][row * kHeadDim + chunk * 16], k_cache + offset, valid);
                copy_async16(&sm.v8[stage][row * kHeadDim + chunk * 16], v_cache + offset, valid);
            }
        } else {
            copy_bf16(g, tile, sm.k[stage], tile.input ? k_in : reinterpret_cast<const __nv_bfloat16*>(k_cache),
                      threadIdx.x, kThreads);
            copy_bf16(g, tile, sm.v[stage], tile.input ? v_in : reinterpret_cast<const __nv_bfloat16*>(v_cache),
                      threadIdx.x, kThreads);
        }
        copy_async_commit();
    };

    // Starts copying the bias of this warp's 16 rows for a tile: lane l copies
    // chunks l, l + 32, ... of the rows' 9 chunks each.
    auto load_bias = [&](const PrefillSegment& g, const PrefillTile& tile) {
        const unsigned m0 = g.block_m * kRows;
        const int top = int(tile.p0) + kPrefillKeys - 1;
#pragma unroll
        for (int i = lane; i < 16 * kBiasChunks; i += 32) {
            const int row = warp * 16 + i / kBiasChunks;
            const int c = i % kBiasChunks;
            const unsigned mg = m0 + row;
            const unsigned r = mg / kGroup;
            const int chunk = ((int(start + r) - top) >> 3) + c;
            const bool valid = mg < total_m && chunk >= 0 && (chunk + 1) * 8 <= int(extent);
            const __nv_bfloat16* src =
                bias + (size_t(r) * kHeads + g.kv_head * kGroup + mg % kGroup) * extent + chunk * 8;
            copy_async16(&sm.bias[row * kBiasStride + c * 8], valid ? src : bias, valid);
        }
        copy_async_commit();
    };

    // Query A fragments of a segment: register j of k-step s holds row (j & 1),
    // dims 16s + 2(lane % 4) + 8(j >> 1) and the next.
    uint32_t qa[kHeadDim / 16][4];
    auto load_q = [&](const PrefillSegment& g) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned mg = g.block_m * kRows + warp * 16 + lane / 4 + 8 * i;
            const bool valid = mg < total_m;
            const size_t head = size_t(valid ? mg / kGroup : 0) * kHeads + g.kv_head * kGroup + mg % kGroup;
            const __nv_bfloat16* row = q + head * kHeadDim + (lane % 4) * 2;
#pragma unroll
            for (int s = 0; s < kHeadDim / 16; ++s) {
                qa[s][i] = valid ? *reinterpret_cast<const uint32_t*>(row + 16 * s) : 0;
                qa[s][2 + i] = valid ? *reinterpret_cast<const uint32_t*>(row + 16 * s + 8) : 0;
            }
        }
    };

    // O += P V in F16 for one tile, P as A fragments: k-step kk covers keys
    // 16kk..16kk+15. BF16 values convert to F16 in registers; E4M3 values were
    // decoded to F16.
    auto pv = [&](float (&o)[kHeadDim / 8][4], const uint32_t (&pa)[kPrefillKeys / 16][4],
                  const __nv_bfloat16* values) {
#pragma unroll
        for (int kk = 0; kk < kPrefillKeys / 16; ++kk) {
#pragma unroll
            for (int dp = 0; dp < kHeadDim / 16; ++dp) {
                uint32_t b[4];
                ldmatrix_x4_trans(b, values + swizzle(kk * 16 + ((lane >> 3) & 1) * 8 + (lane & 7), 2 * dp + (lane >> 4)));
                if constexpr (!kFp8) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        b[i] = bf16x2_to_f16x2(b[i]);
                    }
                }
                mma_f16(o[2 * dp], pa[kk], b[0], b[1]);
                mma_f16(o[2 * dp + 1], pa[kk], b[2], b[3]);
            }
        }
    };

    // BF16 blocks run the two halves of their warps half a tile apart: warps
    // 4-7 ("late" warps) compute each tile's P V at the start of the next tile.
    // Each scheduler holds warp w and w + 4, so one warp's scores and softmax
    // run while the other's products keep the tensor cores busy. The late
    // warps still read the previous tile's values then, so only the keys of
    // tile t + 1 are copied at the start of tile t; the late warps copy its
    // values once all of them are done with tile t - 1's.
    constexpr bool kSkew = !kFp8;
    const bool late = kSkew && warp >= kWarps / 2;
    uint32_t pending[kPrefillKeys / 16][4];
    bool has_pending = false;

    // The first segment's first tile, bias and queries. Each later segment's
    // are copied during the last tile of the segment before it.
    PrefillSegment seg = segment_at(my_begin);
    int stage = 0;
    {
        const PrefillTile first = tile_at(seg, seg.t_begin);
        load(seg, first, 0);
        if (first.biased) {
            load_bias(seg, first);
        }
        load_q(seg);
    }

    for (;;) {
        const bool has_next = seg.unit_begin + seg.t_end < my_end;
        const PrefillSegment next_seg = segment_at(has_next ? seg.unit_begin + seg.t_end : seg.begin);
        const unsigned m0 = seg.block_m * kRows;
        const int kv_head = seg.kv_head;

        // This thread's two M rows: local rows 16w + lane/4 and 8 below it.
        int local[2];
        bool valid_row[2];
        unsigned position[2];
        size_t row_head[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            local[i] = warp * 16 + lane / 4 + 8 * i;
            const unsigned mg = m0 + local[i];
            valid_row[i] = mg < total_m;
            const unsigned r = valid_row[i] ? mg / kGroup : 0;
            position[i] = start + r;
            row_head[i] = size_t(r) * kHeads + kv_head * kGroup + mg % kGroup;
        }

        float m[2];
        float l[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            m[i] = -FLT_MAX;
            l[i] = 0.f;
            if (seg.t_begin == 0) {
                m[i] = valid_row[i] ? __bfloat162float(sinks[row_head[i]]) * kLog2e : 0.f;
                l[i] = lane % 4 == 0 ? kPScale : 0.f;
            }
        }
        float o[kHeadDim / 8][4] = {};

        for (unsigned t = seg.t_begin; t < seg.t_end; ++t) {
            const PrefillTile tile = tile_at(seg, t);
            copy_async_wait_all();
            __syncthreads();
            // The next tile, of this segment or the next one.
            const bool last_tile = t + 1 == seg.t_end;
            const bool more = !last_tile || has_next;
            const PrefillSegment& next_owner = last_tile ? next_seg : seg;
            const PrefillTile next = tile_at(next_owner, last_tile ? next_seg.t_begin : t + 1);
            if constexpr (kSkew) {
                if (more) {
                    copy_bf16(next_owner, next, sm.k[1 - stage],
                              next.input ? k_in : reinterpret_cast<const __nv_bfloat16*>(k_cache), threadIdx.x,
                              kThreads);
                    copy_async_commit();
                }
                if (late) {
                    if (has_pending) {
                        pv(o, pending, sm.v[1 - stage]);
                        has_pending = false;
                    }
                    // Every late warp is done with the values in stage 1 - stage.
                    asm volatile("bar.sync 1, %0;" ::"r"(kThreads / 2) : "memory");
                    if (more) {
                        copy_bf16(next_owner, next, sm.v[1 - stage],
                                  next.input ? v_in : reinterpret_cast<const __nv_bfloat16*>(v_cache),
                                  threadIdx.x - kThreads / 2, kThreads / 2);
                        copy_async_commit();
                    }
                }
            } else if (more) {
                load(next_owner, next, 1 - stage);
            }
            if constexpr (kFp8) {
                for (int i = threadIdx.x; i < kPrefillKeys * 8; i += kThreads) {
                    const int row = i / 8;
                    const int chunk = i % 8;
                    uint4 lo4;
                    uint4 hi4;
                    e4m3x16_to_bf16(*reinterpret_cast<const uint4*>(&sm.k8[stage][row * kHeadDim + chunk * 16]), lo4,
                                    hi4);
                    *reinterpret_cast<uint4*>(&sm.k[0][swizzle(row, 2 * chunk)]) = lo4;
                    *reinterpret_cast<uint4*>(&sm.k[0][swizzle(row, 2 * chunk + 1)]) = hi4;
                    e4m3x16_to_f16(*reinterpret_cast<const uint4*>(&sm.v8[stage][row * kHeadDim + chunk * 16]), lo4,
                                   hi4);
                    *reinterpret_cast<uint4*>(&sm.v[0][swizzle(row, 2 * chunk)]) = lo4;
                    *reinterpret_cast<uint4*>(&sm.v[0][swizzle(row, 2 * chunk + 1)]) = hi4;
                }
                __syncthreads();
            }
            const __nv_bfloat16* keys = sm.k[kFp8 ? 0 : stage];
            const __nv_bfloat16* values = sm.v[kFp8 ? 0 : stage];

            // S = Q K^T: n-tile n holds keys 8n..8n+7.
            float s[kPrefillKeys / 8][4] = {};
#pragma unroll
            for (int ks = 0; ks < kHeadDim / 16; ++ks) {
#pragma unroll
                for (int np = 0; np < kPrefillKeys / 16; ++np) {
                    uint32_t b[4];
                    ldmatrix_x4(b, keys + swizzle(np * 16 + (lane >> 4) * 8 + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
                    mma_bf16(s[2 * np], qa[ks], b[0], b[1]);
                    mma_bf16(s[2 * np + 1], qa[ks], b[2], b[3]);
                }
            }
            // The queries are spent after the segment's last Q K^T; start
            // loading the next segment's.
            if (last_tile && has_next) {
                load_q(next_seg);
            }

            // Scores in the log2 domain, with bias and mask. Element e of n-tile n
            // is row e / 2, key 8n + 2(lane % 4) + e % 2.
            // Element (n, e) of row i has its bias at bias_base[i] - 8n - e % 2.
            const __nv_bfloat16* bias_base[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int first = int(position[i]) - int(tile.p0 + kPrefillKeys - 1);
                bias_base[i] = sm.bias + local[i] * kBiasStride + (first & 7) + kPrefillKeys - 1 - (lane % 4) * 2;
            }
            if (tile.full && !tile.biased) {
#pragma unroll
                for (int n = 0; n < kPrefillKeys / 8; ++n) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        s[n][e] *= score_scale;
                    }
                }
            } else if (tile.full && tile.near) {
#pragma unroll
                for (int n = 0; n < kPrefillKeys / 8; ++n) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float b = __bfloat162float(bias_base[e / 2][-(n * 8 + e % 2)]);
                        s[n][e] = fmaf(b, kLog2e, s[n][e] * score_scale);
                    }
                }
            } else {
#pragma unroll
                for (int n = 0; n < kPrefillKeys / 8; ++n) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int i = e / 2;
                        const int j = n * 8 + (lane % 4) * 2 + e % 2;
                        const unsigned p = tile.p0 + j;
                        const unsigned distance = position[i] - p;
                        const bool visible = p < tile.end && p <= position[i] && (window == 0 || distance < window);
                        float x = s[n][e] * score_scale;
                        if (tile.biased && distance < extent) {
                            x = fmaf(__bfloat162float(bias_base[i][-(n * 8 + e % 2)]), kLog2e, x);
                        }
                        s[n][e] = visible ? x : -INFINITY;
                    }
                }
            }

            // This warp has read its bias rows; copy the next tile's while the
            // softmax and P V run.
            if (more && next.biased) {
                __syncwarp();
                load_bias(next_owner, next);
            }

            // Online softmax over the row's 4 lanes.
            float factor[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                float mx = m[i];
#pragma unroll
                for (int n = 0; n < kPrefillKeys / 8; ++n) {
                    mx = fmaxf(mx, fmaxf(s[n][2 * i], s[n][2 * i + 1]));
                }
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                factor[i] = exp2_ftz(m[i] - mx);
                m[i] = mx;
            }
            // Weights times kPScale, so that F16 keeps weights down to 2^-29 of the
            // row's largest at full precision.
            const float offset[2] = {m[0] - kPScaleLog2, m[1] - kPScaleLog2};
            float sum[2] = {0.f, 0.f};
#pragma unroll
            for (int n = 0; n < kPrefillKeys / 8; ++n) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    s[n][e] = exp2_ftz(s[n][e] - offset[e / 2]);
                    sum[e / 2] += s[n][e];
                }
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                l[i] = fmaf(l[i], factor[i], sum[i]);
            }
#pragma unroll
            for (int n = 0; n < kHeadDim / 8; ++n) {
                o[n][0] *= factor[0];
                o[n][1] *= factor[0];
                o[n][2] *= factor[1];
                o[n][3] *= factor[1];
            }

            uint32_t pa[kPrefillKeys / 16][4];
#pragma unroll
            for (int kk = 0; kk < kPrefillKeys / 16; ++kk) {
                pa[kk][0] = pack_f16x2(s[2 * kk][0], s[2 * kk][1]);
                pa[kk][1] = pack_f16x2(s[2 * kk][2], s[2 * kk][3]);
                pa[kk][2] = pack_f16x2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
                pa[kk][3] = pack_f16x2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
            }
            if (late) {
#pragma unroll
                for (int kk = 0; kk < kPrefillKeys / 16; ++kk) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        pending[kk][i] = pa[kk][i];
                    }
                }
                has_pending = true;
            } else {
                pv(o, pa, values);
            }
            stage ^= 1;
        }
        // The late warps' last tile, whose values sit in the stage before the
        // next segment's.
        if (has_pending) {
            pv(o, pending, sm.v[1 - stage]);
            has_pending = false;
        }

#pragma unroll
        for (int i = 0; i < 2; ++i) {
            l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
            l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        }

        if (seg.t_begin != 0 || seg.t_end != seg.range.tiles) {
            // A partial segment: store it, and continue with the next segment
            // unless this is the unit's last to finish.
            float* slot = partials + size_t(2 * blockIdx.x + (seg.begin != my_begin)) * kSlot;
            float4* slot_o = reinterpret_cast<float4*>(slot) + warp * (kHeadDim / 8) * 32 + lane;
#pragma unroll
            for (int n = 0; n < kHeadDim / 8; ++n) {
                __stcg(slot_o + n * 32, make_float4(o[n][0], o[n][1], o[n][2], o[n][3]));
            }
            if (lane % 4 == 0) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    __stcg(slot + kRows * kHeadDim + local[i], m[i]);
                    __stcg(slot + kRows * kHeadDim + kRows + local[i], l[i]);
                }
            }
            // The barrier orders every thread's stores before thread 0's
            // release, and thread 0's acquire before every thread's loads.
            __syncthreads();
            unsigned* counter = &counters[seg.block_m * kKvHeads + kv_head];
            const unsigned first_block = block_of(seg.unit_begin);
            const unsigned last_block = block_of(seg.unit_begin + seg.range.tiles - 1);
            if (threadIdx.x == 0) {
                __threadfence();
                sm.last = atomicAdd(counter, 1u) == last_block - first_block;
                __threadfence();
            }
            __syncthreads();
            const bool combine = sm.last;
            if (combine) {
                for (unsigned b = first_block; b <= last_block; ++b) {
                    if (b == blockIdx.x) {
                        continue;
                    }
                    const float* other = partials + size_t(2 * b + (seg.unit_begin > begin_of(b))) * kSlot;
                    float own[2];
                    float theirs[2];
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        const float mo = __ldcg(other + kRows * kHeadDim + local[i]);
                        const float lo = __ldcg(other + kRows * kHeadDim + kRows + local[i]);
                        const float mx = fmaxf(m[i], mo);
                        own[i] = exp2_ftz(m[i] - mx);
                        theirs[i] = exp2_ftz(mo - mx);
                        m[i] = mx;
                        l[i] = fmaf(l[i], own[i], lo * theirs[i]);
                    }
                    const float4* other_o =
                        reinterpret_cast<const float4*>(other) + warp * (kHeadDim / 8) * 32 + lane;
#pragma unroll
                    for (int n = 0; n < kHeadDim / 8; ++n) {
                        const float4 p = __ldcg(other_o + n * 32);
                        o[n][0] = fmaf(o[n][0], own[0], p.x * theirs[0]);
                        o[n][1] = fmaf(o[n][1], own[0], p.y * theirs[0]);
                        o[n][2] = fmaf(o[n][2], own[1], p.z * theirs[1]);
                        o[n][3] = fmaf(o[n][3], own[1], p.w * theirs[1]);
                    }
                }
                if (threadIdx.x == 0) {
                    *counter = 0;
                }
            }
            // sm.last is read; the next segment may set it again.
            __syncthreads();
            if (!combine) {
                if (!has_next) {
                    break;
                }
                seg = next_seg;
                continue;
            }
        }

#pragma unroll
        for (int i = 0; i < 2; ++i) {
            if (!valid_row[i]) {
                continue;
            }
            const float inv = 1.f / l[i];
            __nv_bfloat16* dst = out + row_head[i] * kHeadDim + (lane % 4) * 2;
#pragma unroll
            for (int n = 0; n < kHeadDim / 8; ++n) {
                *reinterpret_cast<__nv_bfloat162*>(dst + 8 * n) =
                    __floats2bfloat162_rn(o[n][2 * i] * inv, o[n][2 * i + 1] * inv);
            }
        }
        if (!has_next) {
            break;
        }
        seg = next_seg;
    }
}

constexpr int kPrefillWarps = 8;

}  // namespace

// Writes `rows` = gridDim.x tokens' k and v ([rows, 8, 128] each) into the
// caches at slots (*start + i) mod slots, as `rows` calls of kv_append_bf16
// would: when rows exceeds slots, only the last `slots` rows, which those
// calls leave behind. One block of 128 threads per token.
extern "C" __global__ void __launch_bounds__(128)
kv_append_rows_bf16(const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* k_cache,
                    __nv_bfloat16* v_cache, const unsigned* start, unsigned slots) {
    if (blockIdx.x + slots < gridDim.x) {
        return;
    }
    const size_t i = threadIdx.x * 8;
    const size_t src = size_t(blockIdx.x) * kKvRow + i;
    const size_t dst = size_t((*start + blockIdx.x) % slots) * kKvRow + i;
    *reinterpret_cast<uint4*>(k_cache + dst) = *reinterpret_cast<const uint4*>(k + src);
    *reinterpret_cast<uint4*>(v_cache + dst) = *reinterpret_cast<const uint4*>(v + src);
}

// The FP8 variant of `kv_append_rows_bf16`, rounding as `kv_append_fp8`.
extern "C" __global__ void __launch_bounds__(128)
kv_append_rows_fp8(const __nv_bfloat16* k, const __nv_bfloat16* v, uint8_t* k_cache, uint8_t* v_cache,
                   const unsigned* start, unsigned slots) {
    if (blockIdx.x + slots < gridDim.x) {
        return;
    }
    const size_t i = threadIdx.x * 8;
    const size_t src = size_t(blockIdx.x) * kKvRow + i;
    const size_t dst = size_t((*start + blockIdx.x) % slots) * kKvRow + i;
    float x[8];
    unpack_bf16x8(*reinterpret_cast<const uint4*>(k + src), x);
    *reinterpret_cast<uint2*>(k_cache + dst) =
        make_uint2(pack_e4m3x4(x[0], x[1], x[2], x[3]), pack_e4m3x4(x[4], x[5], x[6], x[7]));
    unpack_bf16x8(*reinterpret_cast<const uint4*>(v + src), x);
    *reinterpret_cast<uint2*>(v_cache + dst) =
        make_uint2(pack_e4m3x4(x[0], x[1], x[2], x[3]), pack_e4m3x4(x[4], x[5], x[6], x[7]));
}

extern "C" __global__ void __launch_bounds__(kPrefillWarps * 32)
prefill_attention_bf16(const __nv_bfloat16* q, const __nv_bfloat16* k_in, const __nv_bfloat16* v_in,
                       const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, const __nv_bfloat16* bias,
                       const __nv_bfloat16* sinks, float* partials, unsigned* counters, __nv_bfloat16* out,
                       const unsigned* start, unsigned rows, unsigned slots, unsigned window, unsigned extent,
                       unsigned input_keys, float scale) {
    prefill_attention<Bf16Format, kPrefillWarps>(q, k_in, v_in, k_cache, v_cache, bias, sinks, partials,
                                                 counters, out, start, rows, slots, window, extent, input_keys,
                                                 scale);
}

// `prefill_attention_bf16` over E4M3 caches. Every key comes from the cache
// (input_keys must be 0).
extern "C" __global__ void __launch_bounds__(kPrefillWarps * 32)
prefill_attention_fp8(const __nv_bfloat16* q, const __nv_bfloat16* k_in, const __nv_bfloat16* v_in,
                      const uint8_t* k_cache, const uint8_t* v_cache, const __nv_bfloat16* bias,
                      const __nv_bfloat16* sinks, float* partials, unsigned* counters, __nv_bfloat16* out,
                      const unsigned* start, unsigned rows, unsigned slots, unsigned window, unsigned extent,
                      unsigned input_keys, float scale) {
    prefill_attention<Fp8Format, kPrefillWarps>(q, k_in, v_in, k_cache, v_cache, bias, sinks, partials,
                                                counters, out, start, rows, slots, window, extent, input_keys,
                                                scale);
}

// The launcher passes PREFILL_SHARED_BYTES of dynamic shared memory.
static_assert(sizeof(PrefillShared<Bf16Format, kPrefillWarps>) <= 84240);
static_assert(sizeof(PrefillShared<Fp8Format, kPrefillWarps>) <= 84240);
