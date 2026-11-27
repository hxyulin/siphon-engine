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
//
// `decode_step_bf16` and `decode_step_fp8` run one layer's whole decode step
// between its projections and its output projection in one launch, with the
// results of the separate kernels: attention_post_bf16 (norm.cu),
// relative_bias_bf16, kv_append_*, decode_attention_* and attention_gate_bf16
// (elementwise.cu).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <climits>
#include <cstdint>

#include "conv_step.cuh"
#include "packed.cuh"
#include "pdl.cuh"

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
    static constexpr bool kFp4 = false;
    static constexpr int kTile = 32;
    static constexpr int kChunkElems = 8;
    // Elems of one head's row, and of the 16-byte chunk at which chunk c starts.
    static constexpr int kRowElems = kHeadDim;
    static constexpr int kChunkStride = 8;
    __device__ static void unpack_chunk(uint4 raw, float (&v)[kChunkElems]) { unpack_bf16x8(raw, v); }
    __device__ static void unpack4(const Elem* p, float (&v)[4]) {
        unpack_bf16x4(*reinterpret_cast<const uint2*>(p), v);
    }
    // Stores one BF16 value as it is.
    __device__ static void store1(Elem* p, __nv_bfloat16 x) { *p = x; }
    // Stores 4 values that are already BF16 values.
    __device__ static void store4(Elem* p, const float (&v)[4]) {
        const __nv_bfloat162 lo = __floats2bfloat162_rn(v[0], v[1]);
        const __nv_bfloat162 hi = __floats2bfloat162_rn(v[2], v[3]);
        *reinterpret_cast<uint2*>(p) =
            make_uint2(*reinterpret_cast<const uint32_t*>(&lo), *reinterpret_cast<const uint32_t*>(&hi));
    }
};

struct Fp8Format {
    using Elem = uint8_t;
    static constexpr bool kFp4 = false;
    static constexpr int kTile = 64;
    static constexpr int kChunkElems = 16;
    static constexpr int kRowElems = kHeadDim;
    static constexpr int kChunkStride = 16;
    __device__ static void unpack_chunk(uint4 raw, float (&v)[kChunkElems]) { unpack_e4m3x16(raw, v); }
    __device__ static void unpack4(const Elem* p, float (&v)[4]) {
        unpack_e4m3x4(*reinterpret_cast<const uint32_t*>(p), v);
    }
    // Stores 4 BF16 values rounded to E4M3 as kv_append_fp8 does.
    __device__ static void store4(Elem* p, const float (&v)[4]) {
        *reinterpret_cast<uint32_t*>(p) = pack_e4m3x4(v[0], v[1], v[2], v[3]);
    }
    // Stores one BF16 value rounded to E4M3 as kv_append_fp8 does.
    __device__ static void store1(Elem* p, __nv_bfloat16 x) {
        *p = uint8_t(pack_e4m3x4(__bfloat162float(x), 0.f, 0.f, 0.f) & 0xffu);
    }
};

// `Fp8Format` for the prefill attention that multiplies on FP8 tensor cores
// (`prefill_attention_fp8_mma`).
struct Fp8MmaFormat : Fp8Format {};

template <class F>
struct IsFp8Mma {
    static constexpr bool value = false;
};
template <>
struct IsFp8Mma<Fp8MmaFormat> {
    static constexpr bool value = true;
};

// NVFP4 global caches (`KvFormat::Nvfp4`). Each slot holds kFp4Row = 576 bytes per
// buffer: the 8 heads' E2M1 values, 64 bytes each with element 2i in the low nibble of
// byte i, then the 8 heads' E4M3 scales, 8 each, one per 16 elements. An element is its
// E2M1 value times its group's scale. The appends round each group as kv_round_bf16's
// NVFP4 mode does: scale E4M3(amax / 6) to nearest, values E2M1(x / scale) to nearest
// even, saturated at 6. Rounding an already rounded group again gives the same bytes
// (its largest element is 6 times the scale), so the verify and prefill paths round
// their rows to NVFP4 values in BF16 first (kv_round_bf16) and every reader sees one
// value per element. Attention decodes exactly: an E2M1 value times an E4M3 scale has
// at most 6 significant bits and lies in F16's normal range.
constexpr int kFp4HeadBytes = kHeadDim / 2;
constexpr int kFp4HeadScales = kHeadDim / 16;
constexpr int kFp4Data = kKvHeads * kFp4HeadBytes;
constexpr int kFp4Row = kKvHeads * (kFp4HeadBytes + kFp4HeadScales);

__device__ __forceinline__ size_t fp4_data(unsigned slot, int head) {
    return size_t(slot) * kFp4Row + head * kFp4HeadBytes;
}

__device__ __forceinline__ size_t fp4_scales(unsigned slot, int head) {
    return size_t(slot) * kFp4Row + kFp4Data + head * kFp4HeadScales;
}

// Four bytes of E2M1 pairs as four F16 pairs, byte i's low nibble in the low half of
// word i.
__device__ __forceinline__ void e2m1x8_to_f16x8(uint32_t raw, uint32_t (&h)[4]) {
#if SIPHON_SM89
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        h[i] = e2m1x2_to_f16x2_sm89(raw >> (8 * i));
    }
#else
    asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\ncvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\ncvt.rn.f16x2.e2m1x2 %3, b3;\n}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(raw));
#endif
}

#if SIPHON_SM89
// The E2M1 code of `y`, as cvt.rn.satfinite.e2m1x2.f32 rounds it: to nearest, ties to the
// even code, saturated at 6, sign kept, NaN to 6. Ada has no E2M1 converter. The
// magnitude's code is the number of rounding thresholds it reaches.
__device__ __forceinline__ uint32_t e2m1_code_sm89(float y) {
    const float a = fabsf(y);
    const uint32_t code = uint32_t(a > 0.25f) + uint32_t(a >= 0.75f) + uint32_t(a > 1.25f) +
                          uint32_t(a >= 1.75f) + uint32_t(a > 2.5f) + uint32_t(a >= 3.5f) + uint32_t(a > 5.f);
    return isnan(y) ? 7u : code | (__float_as_uint(y) >> 31 << 3);
}
#endif

// Eight values, already divided by their scale, as four bytes of E2M1 pairs.
__device__ __forceinline__ uint32_t pack_e2m1x8(const float (&x)[8]) {
#if SIPHON_SM89
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        r |= e2m1_code_sm89(x[i]) << (4 * i);
    }
    return r;
#else
    uint32_t r;
    asm("{\n.reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\ncvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\ncvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n}"
        : "=r"(r)
        : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
    return r;
#endif
}

__device__ __forceinline__ uint8_t fp4_scale_of(float amax) {
    return __nv_cvt_float_to_fp8(amax / 6.f, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float e4m3_value(uint8_t s) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3)));
}

__device__ __forceinline__ __half2 e4m3_half2(uint8_t s) {
    const __half h = __half(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3));
    return __halves2half2(h, h);
}

__device__ __forceinline__ uint32_t hmul2_bits(uint32_t x, __half2 s) {
    const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&x), s);
    return *reinterpret_cast<const uint32_t*>(&r);
}

// A group of 16 values, x[i] the value of element i, as 8 bytes of E2M1 pairs and its
// scale byte.
__device__ __forceinline__ uint2 fp4_group(const float (&x)[16], uint8_t& scale) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        amax = fmaxf(amax, fabsf(x[i]));
    }
    scale = fp4_scale_of(amax);
    const float s = e4m3_value(scale);
    float q[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        q[i] = s == 0.f ? 0.f : __fdiv_rn(x[i], s);
    }
    const float lo[8] = {q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]};
    const float hi[8] = {q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15]};
    return make_uint2(pack_e2m1x8(lo), pack_e2m1x8(hi));
}

// The 32 elements of a 16-byte chunk times their two groups' scales, as F32.
__device__ __forceinline__ void unpack_fp4x32(uint4 raw, uint8_t s0, uint8_t s1, float (&v)[32]) {
    const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t h[4];
        e2m1x8_to_f16x8(words[w], h);
        const float s = e4m3_value(w < 2 ? s0 : s1);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h[i]));
            v[8 * w + 2 * i] = f.x * s;
            v[8 * w + 2 * i + 1] = f.y * s;
        }
    }
}

// The NVFP4 cache format of decode attention, prefill and the rows kernels. Tiles hold
// the raw bytes, kRowElems per head row, and the scales beside them.
struct Fp4Format {
    using Elem = uint8_t;
    static constexpr bool kFp4 = true;
    static constexpr int kTile = 64;
    static constexpr int kChunkElems = 32;
    static constexpr int kRowElems = kFp4HeadBytes;
    static constexpr int kChunkStride = 16;
};

// `Fp4Format` for the prefill attention that multiplies Q K^T on Blackwell's
// native NVFP4 tensor cores (`kind::mxf4nvf4`, m16n8k64) and P V on block-scaled
// FP8 tensor cores (`kind::mxf8f6f4`, m16n8k32) (`prefill_attention_fp4_mma`).
struct Fp4MmaFormat : Fp4Format {};

template <class F>
struct IsFp4Mma {
    static constexpr bool value = false;
};
template <>
struct IsFp4Mma<Fp4MmaFormat> {
    static constexpr bool value = true;
};

// Streaming L2 eviction policy: KV cache slabs read once per step should evict first so
// they do not displace reusable weights and activations from L2.
__device__ __forceinline__ uint64_t l2_evict_first() {
    uint64_t policy;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

// Copies 8 bytes from global to shared memory without blocking, or writes 8 zero bytes
// when `valid` is false.
__device__ __forceinline__ void copy_async8(void* dst, const void* src, bool valid) {
    const auto shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(shared), "l"(src), "r"(valid ? 8 : 0));
}

__device__ __forceinline__ void copy_async8_evict(void* dst, const void* src, bool valid, uint64_t policy) {
    const auto shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 8, %2, %3;" ::"r"(shared), "l"(src),
                 "r"(valid ? 8 : 0), "l"(policy));
}

// atomicAdd with acquire and release semantics at GPU scope.
__device__ __forceinline__ unsigned atomic_add_acq_rel(unsigned* p, unsigned v) {
    unsigned old;
    asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}

// Copies 16 bytes from global to shared memory without blocking, or writes 16
// zero bytes when `valid` is false.
__device__ __forceinline__ void copy_async16(void* dst, const void* src, bool valid) {
    const auto shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(shared), "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void copy_async16_evict(void* dst, const void* src, bool valid, uint64_t policy) {
    const auto shared = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;" ::"r"(shared), "l"(src),
                 "r"(valid ? 16 : 0), "l"(policy));
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
    typename F::Elem k[2][kTile * F::kRowElems];
    typename F::Elem v[2][kTile * F::kRowElems];
    // NVFP4: each tile row's 8 scale bytes (stubs otherwise).
    static constexpr int kScaleTile = F::kFp4 ? kTile * kFp4HeadScales : 16;
    uint8_t ks[2][kScaleTile];
    uint8_t vs[2][kScaleTile];
    // Per-warp partial scores over each warp's quarter of the dims.
    float scores[kWarps][kGroup][kTile];
    // Softmax weights of the tile's keys for the 3 heads, padded to 4.
    float4 weights[kTile];
    // Rescale factor of each head's accumulators for the current tile.
    float factor[kGroup];
    // Combined softmax denominator of each head, in the combining block.
    float total[kGroup];
    // Fused step only: the 3 scaled queries, the 3 relative states, and the
    // new key and value of the block's KV head in the cache format.
    float q[kGroup][kHeadDim];
    float rel[kGroup][kRelDim];
    alignas(16) typename F::Elem new_k[F::kRowElems];
    alignas(16) typename F::Elem new_v[F::kRowElems];
    alignas(8) uint8_t new_ks[kFp4HeadScales];
    alignas(8) uint8_t new_vs[kFp4HeadScales];
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
    constexpr int kElems = F::kChunkStride;
    constexpr int kRow = F::kRowElems;
#pragma unroll
    for (int k = 0; k < kTile * kChunks / kThreads; ++k) {
        const int i = threadIdx.x + k * kThreads;
        const int row = i / kChunks;
        const int chunk = i % kChunks;
        const unsigned slot = first + row;
        const bool valid = slot < end;
        const size_t offset = F::kFp4 ? fp4_data(valid ? slot : 0, kv_head) + chunk * kElems
                                      : size_t(valid ? slot : 0) * kKvRow + kv_head * kHeadDim + chunk * kElems;
        copy_async16(&sm.k[stage][row * kRow + (chunk ^ (row % kChunks)) * kElems], k_cache + offset, valid);
        copy_async16(&sm.v[stage][row * kRow + chunk * kElems], v_cache + offset, valid);
    }
    if constexpr (F::kFp4) {
        // Thread t < kTile copies row t's key scales, thread kTile + t its value scales.
        static_assert(2 * kTile <= kThreads);
        const int row = threadIdx.x % kTile;
        const unsigned slot = first + row;
        const bool valid = slot < end;
        const size_t offset = fp4_scales(valid ? slot : 0, kv_head);
        if (threadIdx.x < kTile) {
            copy_async8(&sm.ks[stage][row * kFp4HeadScales], k_cache + offset, valid);
        } else if (threadIdx.x < 2 * kTile) {
            copy_async8(&sm.vs[stage][row * kFp4HeadScales], v_cache + offset, valid);
        }
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

// The helpers below repeat those of the same names in norm.cu and
// elementwise.cu, so that the fused decode step rounds exactly as
// attention_post_bf16 and attention_gate_bf16 do.

// Metal's `simd_sum`: a butterfly over lane offsets 1, 2, 4, 8, 16.
__device__ __forceinline__ float warp_sum_metal(float v) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// `states * min(threshold / (norm + 1e-6), 1)` with MLX's F32 constants.
__device__ __forceinline__ float l2_clip_factor(float sum_squares, float threshold) {
    const float norm = __fsqrt_rn(sum_squares);
    return fminf(__fdiv_rn(threshold, __fadd_rn(norm, 1e-6f)), 1.f);
}

// The L2 clip's sum for one 128-wide head: each lane adds its 4 rounded
// squares, then a warp sum.
__device__ __forceinline__ float l2_sum_128(const float (&v)[4]) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc = __fadd_rn(acc, __fmul_rn(v[i], v[i]));
    }
    return warp_sum_metal(acc);
}

__device__ __forceinline__ float fma_sum_squares(const float (&v)[4]) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        acc = fmaf(v[i], v[i], acc);
    }
    return acc;
}

__device__ __forceinline__ float inv_rms_f32(float sum_squares, int n, float eps) {
    return 1.f / sqrtf(sum_squares / static_cast<float>(n) + eps);
}

// The `FullAttention` query scale at `position`.
__device__ __forceinline__ float log_scale_tau(uint32_t position, float alpha, float n_floor) {
    const float pos = float(position);
    const float ratio = fmaxf(__fdiv_rn(__fadd_rn(pos, 1.f), n_floor), 1.f);
    return __fadd_rn(1.f, __fmul_rn(alpha, __logf(ratio)));
}

__device__ __forceinline__ float fsub(float a, float b) {
    float r;
    asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fdiv(float a, float b) {
    float r;
    asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// MLX's Metal sigmoid: y = 1 / (1 + exp(|x|)), and 1 - y for positive x.
__device__ __forceinline__ float mlx_sigmoid(float x) {
    const float y = fdiv(1.f, fadd(1.f, expf(fabsf(x))));
    return x < 0.f ? y : fsub(1.f, y);
}

// Round to nearest even, flushing subnormal inputs as MLX's cast does.
__device__ __forceinline__ __nv_bfloat16 to_bf16_ftz(float x) {
    return __float2bfloat16(fabsf(x) < FLT_MIN ? copysignf(0.f, x) : x);
}

// conv_step::short_conv_window with the channel's 4 taps in registers: the
// BF16 output as F32.
__device__ __forceinline__ float short_conv_taps(const float (&win)[conv_step::kKernel],
                                                 const float (&taps)[conv_step::kKernel]) {
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < conv_step::kKernel; ++k) {
        acc = __fadd_rn(acc, __fmul_rn(conv_step::round_bf16(win[k]), taps[k]));
    }
    const float conv = conv_step::round_bf16(acc);
    return conv_step::round_bf16(
        __fdiv_rn(__fadd_rn(conv, win[conv_step::kKernel - 1]), conv_step::kShortDivisor));
}

// conv_step::dynamic_conv_window with the channel's F32 static weights and the
// row's BF16 dynamic taps, per lag, in registers: the BF16 output as F32.
__device__ __forceinline__ float dynamic_conv_taps(const float (&win)[conv_step::kKernel],
                                                   const float (&statics)[conv_step::kKernel],
                                                   const float (&dynamic)[conv_step::kKernel]) {
    float acc = 0.f;
#pragma unroll
    for (int lag = 0; lag < conv_step::kKernel; ++lag) {
        const float w = __fadd_rn(statics[lag], dynamic[lag]);
        acc = __fadd_rn(acc, __fmul_rn(win[conv_step::kKernel - 1 - lag], w));
    }
    return conv_step::round_bf16(
        __fdiv_rn(__fadd_rn(acc, win[conv_step::kKernel - 1]), conv_step::kDynamicDivisor));
}

// Inputs of the fused decode step (`decode_step_bf16`) beyond those of decode
// attention: those of attention_post_bf16, the bias table, the gate and the
// caches to append to.
template <class F>
struct StepInputs {
    const __nv_bfloat16* k_raw;
    const __nv_bfloat16* v_raw;
    const __nv_bfloat16* k_weight;
    const __nv_bfloat16* v_weight;
    const __nv_bfloat16* k_dynamic;
    const __nv_bfloat16* v_dynamic;
    const float* k_static;
    const float* v_static;
    float* k_state;
    float* v_state;
    const __nv_bfloat16* q_raw;
    const __nv_bfloat16* q_scale;
    const __nv_bfloat16* r_raw;
    const __nv_bfloat16* r_scale;
    const __nv_bfloat16* table;
    const __nv_bfloat16* gate;
    typename F::Elem* k_cache;
    typename F::Elem* v_cache;
    int log_scale;
    float alpha;
    float n_floor;
    float qk_threshold;
    float rel_threshold;
    float eps;
};

// Table columns [16] of this lane's keys in the tile that starts at slot
// `first`, as BF16 bits: key lane + 32 j at distance d < extent gets
// table[:, d]. Other entries are left alone; the score phase skips their
// bias. The bits stay unconverted so that nothing waits for the loads until
// the score phase.
template <int kLaneKeys>
__device__ __forceinline__ void load_columns(unsigned short (&columns)[kLaneKeys][kRelDim], const __nv_bfloat16* table,
                                             unsigned extent, unsigned first, unsigned end, unsigned newest,
                                             unsigned slots, int lane) {
#pragma unroll
    for (int j = 0; j < kLaneKeys; ++j) {
        const unsigned slot = first + lane + 32 * j;
        const unsigned distance = (newest + slots - slot) % slots;
        if (slot < end && distance < extent) {
#pragma unroll
            for (int k = 0; k < kRelDim; ++k) {
                columns[j][k] = __bfloat16_as_ushort(table[size_t(k) * extent + distance]);
            }
        }
    }
}

// Writes the step's new key and value row of this block's KV head over row
// `row` of the tile in `stage`. Each thread rewrites the chunks it copied in
// load_tile, after its own copies have landed, so the barrier that follows
// publishes them with the rest of the tile.
template <class F>
__device__ __forceinline__ void patch_tile(DecodeShared<F>& sm, int stage, int row) {
    constexpr int kChunks = DecodeShared<F>::kChunks;
    constexpr int kElems = F::kChunkStride;
    constexpr int kRow = F::kRowElems;
#pragma unroll
    for (int k = 0; k < F::kTile * kChunks / kThreads; ++k) {
        const int i = threadIdx.x + k * kThreads;
        if (i / kChunks == row) {
            const int chunk = i % kChunks;
            *reinterpret_cast<uint4*>(&sm.k[stage][row * kRow + (chunk ^ (row % kChunks)) * kElems]) =
                *reinterpret_cast<const uint4*>(&sm.new_k[chunk * kElems]);
            *reinterpret_cast<uint4*>(&sm.v[stage][row * kRow + chunk * kElems]) =
                *reinterpret_cast<const uint4*>(&sm.new_v[chunk * kElems]);
        }
    }
    if constexpr (F::kFp4) {
        // The threads that copied row `row`'s scales in load_tile.
        if (threadIdx.x == row) {
            *reinterpret_cast<uint2*>(&sm.ks[stage][row * kFp4HeadScales]) =
                *reinterpret_cast<const uint2*>(sm.new_ks);
        } else if (threadIdx.x == F::kTile + row) {
            *reinterpret_cast<uint2*>(&sm.vs[stage][row * kFp4HeadScales]) =
                *reinterpret_cast<const uint2*>(sm.new_vs);
        }
    }
}

// Decode attention of one token over a KV cache, for all 24 query heads.
//
// Block (g, j) handles KV head g = blockIdx.x and its query heads 3g..3g+2
// over split j = blockIdx.y of the cache slots. The n = min(p + 1, slots) live
// slots are cut into gridDim.y contiguous splits of whole tiles; splits past
// the last live slot hold no keys, and their blocks exit at once. The launch
// therefore stays the same at every position p = *position_ptr, and a captured
// graph serves them all.
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
//
// With kStep, the block also does the work of the kernels around attention in
// a decode step (see decode_step_bf16), from `step`: q and bias are unused.
template <class F, bool kStep>
__device__ __forceinline__ void decode_attention(const __nv_bfloat16* q, const typename F::Elem* k_cache,
                                                 const typename F::Elem* v_cache, const __nv_bfloat16* bias,
                                                 const __nv_bfloat16* sinks, float* partial_out, float* partial_max,
                                                 float* partial_sum, unsigned* counters, __nv_bfloat16* out,
                                                 const unsigned* position_ptr, unsigned slots, unsigned window,
                                                 unsigned extent, float scale, const StepInputs<F>& step) {
    constexpr int kTile = F::kTile;
    // Keys of a tile each lane scores, and each warp accumulates in the value
    // phase.
    constexpr int kLaneKeys = kTile / 32;
    constexpr int kWarpKeys = kTile / kWarps;
    constexpr int kElems = F::kChunkElems;
    constexpr int kChunks = DecodeShared<F>::kChunks;
    __shared__ DecodeShared<F> sm;
    const int kv_head = blockIdx.x;
    const int split = blockIdx.y;
    const int splits = gridDim.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    // The 3 query heads of this block are rows head0..head0+2.
    const int head0 = kv_head * kGroup;
    // Score and softmax state of head h = warp, for warps 0..2.
    const int head = head0 + min(warp, kGroup - 1);
    // With kStep: the relative head of lane l < 3 of warp 2, and the key and
    // value channels c0..c0+3 that each lane of warp 3 computes in the
    // appending block. Index kv is 0 for the key, 1 for the value.
    const int rel_head = head0 + min(lane, kGroup - 1);
    const int c0 = kv_head * kHeadDim + lane * 4;
    const bool is_kv_warp = warp == kWarps - 1;
    const bool is_rel_warp = warp == kGroup - 1;

    // With kStep, the weights load first, since they do not depend on the
    // position. Warp 3 loads the conv weights in case this block appends: 16
    // BF16 taps [channel, tap] for a static conv, the F32 static weights
    // [lag, channel] for a dynamic one.
    const bool dynamic = step.k_dynamic != nullptr;
    uint2 q_scale_raw;
    uint2 r_scale_raw[kRelDim / 4];
    uint4 taps_raw[2][2];
    float4 statics[2][conv_step::kKernel];
    if constexpr (kStep) {
        q_scale_raw = *reinterpret_cast<const uint2*>(step.q_scale + head * kHeadDim + lane * 4);
#pragma unroll
        for (int i = 0; i < kRelDim / 4; ++i) {
            r_scale_raw[i] = *reinterpret_cast<const uint2*>(step.r_scale + rel_head * kRelDim + 4 * i);
        }
        if (is_kv_warp) {
#pragma unroll
            for (int kv = 0; kv < 2; ++kv) {
                if (dynamic) {
                    const float* weights = kv == 0 ? step.k_static : step.v_static;
#pragma unroll
                    for (int lag = 0; lag < conv_step::kKernel; ++lag) {
                        statics[kv][lag] = *reinterpret_cast<const float4*>(weights + lag * kKvRow + c0);
                    }
                } else {
                    const __nv_bfloat16* weights = kv == 0 ? step.k_weight : step.v_weight;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        taps_raw[kv][i] = *reinterpret_cast<const uint4*>(weights + c0 * conv_step::kKernel + 8 * i);
                    }
                }
            }
        }
    }

    const unsigned position = *position_ptr;

    // With kStep, the loads from the projections issue before the position
    // arrives, so that the two latencies overlap. Warp 3 loads the raw
    // inputs, conv states [3 rows] and dynamic taps of its key (index 0) and
    // value (1) channels in every block; only the appending block uses them.
    uint16_t gate_raw[kGroup];
    uint2 q_raw;
    uint2 r_raw[kRelDim / 4];
    uint2 x_raw[2];
    float4 state[2][conv_step::kKernel - 1];
    uint2 dynamic_raw[2];
    if constexpr (kStep) {
        // The output projection after this kernel launches with PDL; triggering here lets
        // it launch once every block has started, 0.04 ms per decode step.
        pdl_trigger();
        pdl_wait();
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            gate_raw[h] = __bfloat16_as_ushort(step.gate[(head0 + h) * kHeadDim + threadIdx.x]);
        }
        q_raw = *reinterpret_cast<const uint2*>(step.q_raw + head * kHeadDim + lane * 4);
        if (is_rel_warp) {
#pragma unroll
            for (int i = 0; i < kRelDim / 4; ++i) {
                r_raw[i] = *reinterpret_cast<const uint2*>(step.r_raw + rel_head * kRelDim + 4 * i);
            }
        }
        if (is_kv_warp) {
#pragma unroll
            for (int kv = 0; kv < 2; ++kv) {
                x_raw[kv] = *reinterpret_cast<const uint2*>((kv == 0 ? step.k_raw : step.v_raw) + c0);
#pragma unroll
                for (int k = 0; k < conv_step::kKernel - 1; ++k) {
                    state[kv][k] =
                        *reinterpret_cast<const float4*>((kv == 0 ? step.k_state : step.v_state) + k * kKvRow + c0);
                }
                if (dynamic) {
                    dynamic_raw[kv] = *reinterpret_cast<const uint2*>((kv == 0 ? step.k_dynamic : step.v_dynamic) +
                                                                      kv_head * conv_step::kKernel);
                }
            }
        }
    }

    const unsigned n = min(position + 1, slots);
    const unsigned newest = position % slots;
    // Slots per split: an even share of n, rounded up to whole tiles.
    const unsigned share = (n + splits - 1) / splits;
    const unsigned chunk = (share + kTile - 1) / kTile * kTile;
    const int used = (n + chunk - 1) / chunk;
    if (split >= used) {
        return;
    }
    const unsigned begin = split * chunk;
    const unsigned end = min(begin + chunk, n);
    const int tiles = (end - begin + kTile - 1) / kTile;
    // With kStep, the block whose split holds the newest slot appends.
    const bool appends = kStep && newest - begin < end - begin;
    load_tile(sm, 0, k_cache, v_cache, kv_head, begin, end);

    // This warp's quarter of the 3 scaled queries.
    float qs[kGroup][kWarpDims];
    // With kStep: head `head`'s relative state, the table columns of this
    // lane's keys in the current tile, and the output gates (the sigmoids)
    // of this thread's dims.
    float rel[kRelDim];
    unsigned short columns[kLaneKeys][kRelDim];
    float gate[kGroup];
    if constexpr (kStep) {
        if (warp < kGroup) {
            load_columns(columns, step.table, extent, begin, end, newest, slots, lane);
        }
        const float tau = step.log_scale ? log_scale_tau(position, step.alpha, step.n_floor) : 1.f;
        // bf16(x * tau) as log_scale_bf16 does it, or x unchanged.
        const auto scaled = [&](float x) {
            return step.log_scale ? conv_step::round_bf16(__fmul_rn(x, tau)) : x;
        };
        if (warp < kGroup) {
            // The query of head `head`: scaled L2 clip and log scale.
            float s[4];
            float sc[4];
            unpack_bf16x4(q_raw, s);
            unpack_bf16x4(q_scale_raw, sc);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[i] = __fmul_rn(s[i], sc[i]);
            }
            const float factor = l2_clip_factor(l2_sum_128(s), step.qk_threshold);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sm.q[warp][lane * 4 + i] = scale * scaled(conv_step::round_bf16(__fmul_rn(s[i], factor)));
            }
        }
        if (is_rel_warp && lane < kGroup) {
            // The relative state of head head0 + lane, summed in order as
            // l2_clip_bf16_16 does.
            float s[kRelDim];
#pragma unroll
            for (int i = 0; i < kRelDim / 4; ++i) {
                float part[4];
                float sc[4];
                unpack_bf16x4(r_raw[i], part);
                unpack_bf16x4(r_scale_raw[i], sc);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    s[4 * i + j] = __fmul_rn(part[j], sc[j]);
                }
            }
            float acc = 0.f;
#pragma unroll
            for (int i = 0; i < kRelDim; ++i) {
                acc = __fadd_rn(acc, __fmul_rn(s[i], s[i]));
            }
            const float factor = l2_clip_factor(acc, step.rel_threshold);
#pragma unroll
            for (int i = 0; i < kRelDim; ++i) {
                sm.rel[lane][i] = scaled(conv_step::round_bf16(__fmul_rn(s[i], factor)));
            }
        }
        if (appends && is_kv_warp) {
            // The convs, then the L2 clip of the key and the RMS norm of the
            // value, rounded to BF16, then stored in the cache format.
            // win[kv][i]: channel c0 + i's window, oldest row first.
            float win[2][4][conv_step::kKernel];
            float y[2][4];
#pragma unroll
            for (int kv = 0; kv < 2; ++kv) {
                float x[4];
                unpack_bf16x4(x_raw[kv], x);
                float taps[16];
                float lag_taps[conv_step::kKernel];
                if (dynamic) {
                    unpack_bf16x4(dynamic_raw[kv], lag_taps);
                } else {
                    float half[8];
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        unpack_bf16x8(taps_raw[kv][i], half);
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            taps[8 * i + j] = half[j];
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const auto element = [i](float4 v) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; };
#pragma unroll
                    for (int k = 0; k < conv_step::kKernel - 1; ++k) {
                        win[kv][i][k] = element(state[kv][k]);
                    }
                    win[kv][i][conv_step::kKernel - 1] = x[i];
                    if (dynamic) {
                        float lag_statics[conv_step::kKernel];
#pragma unroll
                        for (int lag = 0; lag < conv_step::kKernel; ++lag) {
                            lag_statics[lag] = element(statics[kv][lag]);
                        }
                        y[kv][i] = dynamic_conv_taps(win[kv][i], lag_statics, lag_taps);
                    } else {
                        const float channel_taps[conv_step::kKernel] = {taps[4 * i], taps[4 * i + 1],
                                                                        taps[4 * i + 2], taps[4 * i + 3]};
                        y[kv][i] = short_conv_taps(win[kv][i], channel_taps);
                    }
                }
            }
            const float factor = l2_clip_factor(l2_sum_128(y[0]), step.qk_threshold);
            const float norm = inv_rms_f32(warp_sum_metal(fma_sum_squares(y[1])), kHeadDim, step.eps);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                y[0][i] = conv_step::round_bf16(__fmul_rn(y[0][i], factor));
                y[1][i] = conv_step::round_bf16(y[1][i] * norm);
            }
            if constexpr (F::kFp4) {
                // Lanes 4i .. 4i + 3 hold group i of the head's 8.
#pragma unroll
                for (int kv = 0; kv < 2; ++kv) {
                    float amax = fmaxf(fmaxf(fabsf(y[kv][0]), fabsf(y[kv][1])), fmaxf(fabsf(y[kv][2]), fabsf(y[kv][3])));
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                    const uint8_t s8 = fp4_scale_of(amax);
                    const float sc = e4m3_value(s8);
                    float x[8] = {};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        x[i] = sc == 0.f ? 0.f : __fdiv_rn(y[kv][i], sc);
                    }
                    const auto code = uint16_t(pack_e2m1x8(x) & 0xffffu);
                    uint8_t* cache = kv == 0 ? step.k_cache : step.v_cache;
                    *reinterpret_cast<uint16_t*>(cache + fp4_data(newest, kv_head) + 2 * lane) = code;
                    *reinterpret_cast<uint16_t*>((kv == 0 ? sm.new_k : sm.new_v) + 2 * lane) = code;
                    if (lane % 4 == 0) {
                        cache[fp4_scales(newest, kv_head) + lane / 4] = s8;
                        (kv == 0 ? sm.new_ks : sm.new_vs)[lane / 4] = s8;
                    }
                }
            } else {
                const size_t row = size_t(newest) * kKvRow + c0;
                F::store4(step.k_cache + row, y[0]);
                F::store4(step.v_cache + row, y[1]);
                F::store4(sm.new_k + lane * 4, y[0]);
                F::store4(sm.new_v + lane * 4, y[1]);
            }
            // Shift the raw inputs into the conv states.
#pragma unroll
            for (int kv = 0; kv < 2; ++kv) {
#pragma unroll
                for (int k = 0; k < conv_step::kKernel - 1; ++k) {
                    *reinterpret_cast<float4*>((kv == 0 ? step.k_state : step.v_state) + k * kKvRow + c0) =
                        make_float4(win[kv][0][k + 1], win[kv][1][k + 1], win[kv][2][k + 1], win[kv][3][k + 1]);
                }
            }
        }
        // Warp 3 takes its gates in the softmax phase of the first tile, where
        // it waits anyway.
        if (!is_kv_warp) {
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                gate[h] = mlx_sigmoid(__uint_as_float(unsigned(gate_raw[h]) << 16));
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
#pragma unroll
            for (int c = 0; c < kWarpDims; ++c) {
                qs[h][c] = sm.q[h][warp * kWarpDims + c];
            }
        }
#pragma unroll
        for (int k = 0; k < kRelDim; ++k) {
            rel[k] = sm.rel[min(warp, kGroup - 1)][k];
        }
    } else {
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
        const unsigned first = begin + t * kTile;
        copy_async_wait_all();
        if (appends && newest - first < unsigned(kTile)) {
            patch_tile(sm, stage, int(newest - first));
        }
        __syncthreads();
        if (t + 1 < tiles) {
            load_tile(sm, 1 - stage, k_cache, v_cache, kv_head, first + kTile, end);
        }

        // 1. Partial scores of keys `lane + 32 j` over this warp's dims.
#pragma unroll
        for (int j = 0; j < kLaneKeys; ++j) {
            const int row = lane + 32 * j;
            const typename F::Elem* keys = sm.k[stage] + row * F::kRowElems;
            float acc[kGroup] = {};
            if constexpr (F::kFp4) {
                // Chunk `warp` holds this warp's 32 dims, with scales 2 warp and 2 warp + 1.
                const uint4 raw = *reinterpret_cast<const uint4*>(keys + (warp ^ (row % kChunks)) * 16);
                const uint8_t* sc = &sm.ks[stage][row * kFp4HeadScales + 2 * warp];
                float kv[32];
                unpack_fp4x32(raw, sc[0], sc[1], kv);
#pragma unroll
                for (int i = 0; i < 32; ++i) {
#pragma unroll
                    for (int h = 0; h < kGroup; ++h) {
                        acc[h] = fmaf(qs[h][i], kv[i], acc[h]);
                    }
                }
            }
#pragma unroll
            for (int c = 0; c < (F::kFp4 ? 0 : kWarpDims / kElems); ++c) {
                const int chunk_index = warp * (kWarpDims / kElems) + c;
                const uint4 raw =
                    *reinterpret_cast<const uint4*>(keys + (chunk_index ^ (row % kChunks)) * F::kChunkStride);
                float kv[kElems];
                if constexpr (!F::kFp4) {
                    F::unpack_chunk(raw, kv);
                }
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
                const unsigned slot = first + row;
                const unsigned distance = (newest + slots - slot) % slots;
                score[j] = -INFINITY;
                if (slot < end && (window == 0 || distance < window)) {
                    score[j] = (sm.scores[0][warp][row] + sm.scores[1][warp][row]) +
                               (sm.scores[2][warp][row] + sm.scores[3][warp][row]);
                    if (distance < extent) {
                        if constexpr (kStep) {
                            float column[kRelDim];
#pragma unroll
                            for (int k = 0; k < kRelDim; ++k) {
                                column[k] = __uint_as_float(unsigned(columns[j][k]) << 16);
                            }
                            score[j] += conv_step::round_bf16(relative_bias_dot(rel, column));
                        } else {
                            score[j] += __bfloat162float(bias[head * extent + distance]);
                        }
                    }
                }
            }
            if constexpr (kStep) {
                if (t + 1 < tiles) {
                    load_columns(columns, step.table, extent, first + kTile, end, newest, slots, lane);
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
        } else if (kStep && t == 0) {
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                gate[h] = mlx_sigmoid(__uint_as_float(unsigned(gate_raw[h]) << 16));
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
            if constexpr (F::kFp4) {
                uint32_t h[4];
                e2m1x8_to_f16x8(*reinterpret_cast<const uint16_t*>(values + r * kFp4HeadBytes + 2 * lane), h);
                const float sc = e4m3_value(sm.vs[stage][r * kFp4HeadScales + lane / 4]);
                const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&h[0]));
                const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&h[1]));
                v[0] = a.x * sc;
                v[1] = a.y * sc;
                v[2] = b.x * sc;
                v[3] = b.y * sc;
            } else {
                F::unpack4(values + r * kHeadDim + lane * 4, v);
            }
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
    static_assert(sizeof(sm.k) >= kWarps * kGroup * kHeadDim * sizeof(float));
    auto* reduce = reinterpret_cast<float(*)[kGroup][kHeadDim]>(&sm.k[0][0]);
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

    // Count finished blocks; the last one of this KV head combines. The
    // barrier orders the block's partial stores before thread 0's release,
    // and its acquire before the other threads' loads after the second
    // barrier.
    __syncthreads();
    if (threadIdx.x == 0) {
        sm.last = atomic_add_acq_rel(&counters[kv_head], 1u) == unsigned(used - 1);
    }
    __syncthreads();
    if (!sm.last) {
        return;
    }

    // Up to kFastSplits splits, the partial outputs load together with the
    // maxes and sums: warp w takes splits w, w + 4, ... for dims 4l..4l+3.
    constexpr int kFastSplits = 16;
    float4 parts[kFastSplits / kWarps][kGroup];
    if (used <= kFastSplits) {
#pragma unroll
        for (int i = 0; i < kFastSplits / kWarps; ++i) {
            const int j = warp + kWarps * i;
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                parts[i][h] = j < used ? __ldcg(reinterpret_cast<const float4*>(
                                             partial_out + (size_t(head0 + h) * kMaxSplits + j) * kHeadDim) +
                                         lane)
                                       : make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    }

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
    float c[kGroup][4] = {};
    const auto accumulate = [&](int h, int j, float4 p) {
        const float factor = factors[h][j];
        c[h][0] = fmaf(factor, p.x, c[h][0]);
        c[h][1] = fmaf(factor, p.y, c[h][1]);
        c[h][2] = fmaf(factor, p.z, c[h][2]);
        c[h][3] = fmaf(factor, p.w, c[h][3]);
    };
    if (used <= kFastSplits) {
#pragma unroll
        for (int i = 0; i < kFastSplits / kWarps; ++i) {
            const int j = warp + kWarps * i;
            if (j < used) {
#pragma unroll
                for (int h = 0; h < kGroup; ++h) {
                    accumulate(h, j, parts[i][h]);
                }
            }
        }
    } else {
        for (int j = warp; j < used; j += kWarps) {
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                accumulate(h, j, __ldcg(reinterpret_cast<const float4*>(
                                            partial_out + (size_t(head0 + h) * kMaxSplits + j) * kHeadDim) +
                                        lane));
            }
        }
    }
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            reduce[warp][h][lane * 4 + i] = c[h][i];
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        const float* r = &reduce[0][h][threadIdx.x];
        const float value = (r[0] + r[kGroup * kHeadDim]) + (r[2 * kGroup * kHeadDim] + r[3 * kGroup * kHeadDim]);
        const float total = sm.total[h];
        const __nv_bfloat16 attn = __float2bfloat16(total == 0.f ? value : value / total);
        if constexpr (kStep) {
            // attention_gate_bf16.
            out[(head0 + h) * kHeadDim + threadIdx.x] = to_bf16_ftz(fmul(__bfloat162float(attn), gate[h]));
        } else {
            out[(head0 + h) * kHeadDim + threadIdx.x] = attn;
        }
    }
    if (threadIdx.x == 0) {
        counters[kv_head] = 0;
    }
}

// The fused decode step of one layer; see decode_step_bf16.
template <class F>
__device__ __forceinline__ void decode_step(
    const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw, const __nv_bfloat16* k_weight,
    const __nv_bfloat16* v_weight, const __nv_bfloat16* k_dynamic, const __nv_bfloat16* v_dynamic,
    const float* k_static, const float* v_static, float* k_state, float* v_state, const __nv_bfloat16* q_raw,
    const __nv_bfloat16* q_scale, const __nv_bfloat16* r_raw, const __nv_bfloat16* r_scale,
    const __nv_bfloat16* table, const __nv_bfloat16* sinks, const __nv_bfloat16* gate, typename F::Elem* k_cache,
    typename F::Elem* v_cache, float* partial_out, float* partial_max, float* partial_sum, unsigned* counters,
    __nv_bfloat16* out, const unsigned* position, unsigned slots, unsigned window, unsigned extent, int log_scale,
    float alpha, float n_floor, float qk_threshold, float rel_threshold, float eps, float scale) {
    const StepInputs<F> step{k_raw,   v_raw,   k_weight, v_weight, k_dynamic, v_dynamic, k_static,
                             v_static, k_state, v_state,  q_raw,    q_scale,   r_raw,     r_scale,
                             table,   gate,    k_cache,  v_cache,  log_scale, alpha,     n_floor,
                             qk_threshold, rel_threshold, eps};
    decode_attention<F, true>(nullptr, k_cache, v_cache, nullptr, sinks, partial_out, partial_max, partial_sum,
                              counters, out, position, slots, window, extent, scale, step);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads)
decode_attention_bf16(const __nv_bfloat16* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                      const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out, float* partial_max,
                      float* partial_sum, unsigned* counters, __nv_bfloat16* out, const unsigned* position_ptr,
                      unsigned slots, unsigned window, unsigned extent, float scale) {
    decode_attention<Bf16Format, false>(q, k_cache, v_cache, bias, sinks, partial_out, partial_max, partial_sum,
                                        counters, out, position_ptr, slots, window, extent, scale, {});
}

// `decode_attention_bf16` over E4M3 caches written by `kv_append_fp8`.
extern "C" __global__ void __launch_bounds__(kThreads)
decode_attention_fp8(const __nv_bfloat16* q, const uint8_t* k_cache, const uint8_t* v_cache,
                     const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out, float* partial_max,
                     float* partial_sum, unsigned* counters, __nv_bfloat16* out, const unsigned* position_ptr,
                     unsigned slots, unsigned window, unsigned extent, float scale) {
    decode_attention<Fp8Format, false>(q, k_cache, v_cache, bias, sinks, partial_out, partial_max, partial_sum,
                                       counters, out, position_ptr, slots, window, extent, scale, {});
}

// One layer's decode step from its projections to the input of the output
// projection, in one launch: attention_post_bf16, relative_bias_bf16,
// kv_append_bf16, decode_attention_bf16 and attention_gate_bf16 in sequence,
// with the same results bit for bit.
//
// None of them needs a grid-wide barrier: each attention block recomputes the
// post-processed query and relative state of its 3 heads (a few hundred
// values), computes the bias of its keys from the table, and the block whose
// split holds the newest slot computes KV head g's new key and value, updates
// the conv states of their channels, writes them to the cache and patches its
// shared copy of that tile. No other block reads the newest slot. The
// combining block applies the gate.
//
// Launched with PDL after the projections. Before pdl_wait a block reads only
// weights and the position, so the previous kernel must not write the
// position.
extern "C" __global__ void __launch_bounds__(kThreads)
decode_step_bf16(const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw, const __nv_bfloat16* k_weight,
                 const __nv_bfloat16* v_weight, const __nv_bfloat16* k_dynamic, const __nv_bfloat16* v_dynamic,
                 const float* k_static, const float* v_static, float* k_state, float* v_state,
                 const __nv_bfloat16* q_raw, const __nv_bfloat16* q_scale, const __nv_bfloat16* r_raw,
                 const __nv_bfloat16* r_scale, const __nv_bfloat16* table, const __nv_bfloat16* sinks,
                 const __nv_bfloat16* gate, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, float* partial_out,
                 float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out,
                 const unsigned* position, unsigned slots, unsigned window, unsigned extent, int log_scale,
                 float alpha, float n_floor, float qk_threshold, float rel_threshold, float eps, float scale) {
    decode_step<Bf16Format>(k_raw, v_raw, k_weight, v_weight, k_dynamic, v_dynamic, k_static, v_static, k_state,
                            v_state, q_raw, q_scale, r_raw, r_scale, table, sinks, gate, k_cache, v_cache,
                            partial_out, partial_max, partial_sum, counters, out, position, slots, window, extent,
                            log_scale, alpha, n_floor, qk_threshold, rel_threshold, eps, scale);
}

// `decode_step_bf16` with an E4M3 cache, appending as `kv_append_fp8`.
extern "C" __global__ void __launch_bounds__(kThreads)
decode_step_fp8(const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw, const __nv_bfloat16* k_weight,
                const __nv_bfloat16* v_weight, const __nv_bfloat16* k_dynamic, const __nv_bfloat16* v_dynamic,
                const float* k_static, const float* v_static, float* k_state, float* v_state,
                const __nv_bfloat16* q_raw, const __nv_bfloat16* q_scale, const __nv_bfloat16* r_raw,
                const __nv_bfloat16* r_scale, const __nv_bfloat16* table, const __nv_bfloat16* sinks,
                const __nv_bfloat16* gate, uint8_t* k_cache, uint8_t* v_cache, float* partial_out,
                float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out,
                const unsigned* position, unsigned slots, unsigned window, unsigned extent, int log_scale,
                float alpha, float n_floor, float qk_threshold, float rel_threshold, float eps, float scale) {
    decode_step<Fp8Format>(k_raw, v_raw, k_weight, v_weight, k_dynamic, v_dynamic, k_static, v_static, k_state,
                           v_state, q_raw, q_scale, r_raw, r_scale, table, sinks, gate, k_cache, v_cache,
                           partial_out, partial_max, partial_sum, counters, out, position, slots, window, extent,
                           log_scale, alpha, n_floor, qk_threshold, rel_threshold, eps, scale);
}

// `decode_step_bf16` with an NVFP4 cache, appending each group as kv_append_rows_fp4.
extern "C" __global__ void __launch_bounds__(kThreads)
decode_step_fp4(const __nv_bfloat16* k_raw, const __nv_bfloat16* v_raw, const __nv_bfloat16* k_weight,
                const __nv_bfloat16* v_weight, const __nv_bfloat16* k_dynamic, const __nv_bfloat16* v_dynamic,
                const float* k_static, const float* v_static, float* k_state, float* v_state,
                const __nv_bfloat16* q_raw, const __nv_bfloat16* q_scale, const __nv_bfloat16* r_raw,
                const __nv_bfloat16* r_scale, const __nv_bfloat16* table, const __nv_bfloat16* sinks,
                const __nv_bfloat16* gate, uint8_t* k_cache, uint8_t* v_cache, float* partial_out,
                float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out,
                const unsigned* position, unsigned slots, unsigned window, unsigned extent, int log_scale,
                float alpha, float n_floor, float qk_threshold, float rel_threshold, float eps, float scale) {
    decode_step<Fp4Format>(k_raw, v_raw, k_weight, v_weight, k_dynamic, v_dynamic, k_static, v_static, k_state,
                           v_state, q_raw, q_scale, r_raw, r_scale, table, sinks, gate, k_cache, v_cache,
                           partial_out, partial_max, partial_sum, counters, out, position, slots, window, extent,
                           log_scale, alpha, n_floor, qk_threshold, rel_threshold, eps, scale);
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
// counted in counters, reads back all of the unit's partials, its own
// included, combines them in block order and writes `out`. Nothing
// waits for another block, so the schedule cannot deadlock when fewer blocks
// fit. Block b holds items begin(b)..begin(b + 1) - 1, and item x lies in block
// max{b : begin(b) <= x}. Only the first and last segment of a block can be
// partial; they use slots 2b and 2b + 1.
//
// `prefill_attention_fp8_mma` (opt-in) runs both products of an E4M3 cache on
// the block-scaled FP8 MMA (`kind::mxf8f6f4`, m16n8k32, F32 accumulators):
// - S = Q K^T: the queries are quantized per row and 32 dims to MXFP8 (E4M3
//   values, a UE8M0 scale, the block's amax / 448 rounded up to a power of
//   two) when a segment loads them; the scales go into the MMA. The keys are
//   the cache's E4M3 bytes, read with ldmatrix from the staged tile.
// - O += P V: the softmax weights carry a factor of 2^8 instead of 2^15, so
//   that the largest is 256, and round to E4M3. The values are the cache's
//   bytes, transposed in shared memory to [dim][key] for the B operand into two
//   ping-pong stages so warp skewing overlaps tensor-core products with SFU
//   softmax. Within each block of 32 keys, MMA k position 4t + j (and 16 + 4t +
//   j) holds key 8 (j / 2) + 2t + j % 2 (and 16 more), the keys a lane holds in
//   the S fragments, so the weights pack into A registers without shuffles; the
//   transpose stores the values in the same order.
// The softmax and its denominator stay F32. Only the rounding of the queries
// and the weights to E4M3 differs from `prefill_attention_fp8`.
//
// `prefill_attention_fp4_mma` (opt-in) accelerates NVFP4 global caches on
// Blackwell (`sm_120a`):
// - S = Q K^T: runs directly on native NVFP4 tensor cores (`kind::mxf4nvf4`,
//   scale_vec::4X, m16n8k64, F32 accumulators) with ZERO key dequantization.
//   The keys stay in their cache layout (64 bytes of E2M1 + 8 UE4M3 scales per
//   row, swizzled for conflict-free ldmatrix). Queries are quantized once per
//   segment in registers to per-16 E2M1 + UE4M3 scales under a per-row F32
//   outer scale folded into the softmax score scale.
// - O += P V: decodes and transposes the staged NVFP4 values directly into
//   two ping-pong [128, 64] E4M3 tiles and multiplies with `mma_fp8` (m16n8k32)
//   using warp skewing, preserving full E4M3 precision for softmax weights P
//   while balancing tensor-core and SFU throughput.

namespace {

constexpr int kPrefillKeys = 64;
// Most blocks of M rows per KV head: 4096 rows of 3 heads in blocks of 128.
constexpr int kPrefillMaxBlocks = 96;
// 64 consecutive distances span at most 9 aligned chunks of 8 bias values.
constexpr int kBiasChunks = kPrefillKeys / 8 + 1;
constexpr int kBiasStride = kBiasChunks * 8;
constexpr float kLog2e = 1.4426950408889634f;
// The softmax weights and sums carry a factor of 2^15, which cancels in O / l,
// so that F16 weights keep full precision down to 2^-29 of the largest one
// instead of 2^-14. The largest weight, 2^15, stays below F16's 65504.
constexpr float kPScaleLog2 = 15.f;
constexpr float kPScale = 32768.f;
// The FP8 MMA variant's factor: the largest weight, 256, stays below E4M3's
// 448, and weights down to 2^-14 of it keep E4M3's full precision.
constexpr float kPScaleLog2Fp8 = 8.f;
constexpr float kPScaleFp8 = 256.f;

// Byte offset of 16-byte unit `unit` of key `row` in a staged [64, 128] E4M3
// tile of the FP8 MMA variant: keys XOR their row, so that the 8 rows one
// ldmatrix reads fall in distinct banks; values XOR bits 1-2 and 4 of the row,
// for the transpose's reads.
__device__ __forceinline__ int k8_offset(int row, int unit) {
    return row * kHeadDim + ((unit ^ (row & 7)) << 4);
}

__device__ __forceinline__ int v8_offset(int row, int unit) {
    return row * kHeadDim + ((unit ^ ((row >> 1 & 3) | (row >> 2 & 4))) << 4);
}

// Byte offset of 16-byte unit `unit` (0..3) of key `row` (0..63) in a staged
// [64, 64] NVFP4 tile: XOR with (row >> 1) & 3 so that the 8 rows an ldmatrix
// reads fall in distinct banks (matches `q_offset` in expert_matmul.cu).
__device__ __forceinline__ int k4_offset(int row, int unit) {
    return row * kFp4HeadBytes + ((unit ^ ((row >> 1) & 3)) << 4);
}

// Byte offset of 16-byte unit `unit` (4 per row) of dim `row` in the
// transposed [128, 64] value tile.
__device__ __forceinline__ int vt_offset(int row, int unit) {
    return row * kPrefillKeys + ((unit ^ (row >> 1 & 3)) << 4);
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t selector) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(selector));
    return d;
}

// Two F16 values packed in a uint32_t (low half first) rounded to two E4M3 bytes
// (first in the low byte), nearest even, saturated to +-448.
__device__ __forceinline__ uint32_t f16x2_to_e4m3x2(uint32_t h) {
    const __half2_raw raw = {static_cast<unsigned short>(h & 0xffffu), static_cast<unsigned short>(h >> 16)};
    return uint32_t(__nv_cvt_halfraw2_to_fp8x2(raw, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint32_t f16x4_to_e4m3x4(uint32_t h0, uint32_t h1) {
    return f16x2_to_e4m3x2(h0) | (f16x2_to_e4m3x2(h1) << 16);
}

// c += a b, block-scaled NVFP4 with F32 accumulators: lanes t = 0 and 1 of each
// quad pass the A scales (4 UE4M3 bytes, one per 16 values) of rows g and g + 8;
// lane `kThreadB` of each quad passes the B scales of column g.
template <int kThreadB>
__device__ __forceinline__ void mma_fp4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                        uint32_t scale_a, uint32_t scale_b) {
#if SIPHON_SM89
    __trap();
#else
    asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1."
        "e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, "
        "{%11, %12}, {%13}, {%14, %15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(scale_a), "h"(uint16_t(0)),
          "h"(uint16_t(0)), "r"(scale_b), "h"(uint16_t(0)), "h"(uint16_t(kThreadB)));
#endif
}

// c += a b, FP8 E4M3 with F32 accumulators: on sm_120a, block-scaled MMA with A scales of
// rows g and g + 8 from byte 0 of lanes 4g and 4g + 1's `scale_a`, B scales 1; on sm_89,
// native Ada FP8 MMA (per-head-row Q scale folded into `row_score_scale`, unscaled K/V/P).
__device__ __forceinline__ void mma_fp8(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                        uint32_t scale_a) {
#if SIPHON_SM89
    (void)scale_a;
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %11}, {%12}, {%11, %11};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(scale_a), "h"(uint16_t(0)),
          "r"(0x7f7f7f7fu));
#endif
}

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
    static constexpr bool kMma8 = IsFp8Mma<F>::value;
    static constexpr bool kMma4 = IsFp4Mma<F>::value;
    static constexpr bool kMmaLow = kMma8 || kMma4;
    static constexpr int kRows = 16 * kWarps;
    // BF16 key and value tiles, [64, 128] swizzled: two stages; for Fp8Format /
    // Fp4Format one tile that each staged tile is decoded into; for Fp8MmaFormat
    // and Fp4MmaFormat, `k` holds two ping-pong stages of the transposed [128, 64]
    // E4M3 value tile so warp skewing overlaps tensor-core MMA with SFU softmax.
    static constexpr int kStages = (kFp8 && !kMmaLow) ? 1 : 2;
    static constexpr int kKTile = kMmaLow ? kPrefillKeys * kHeadDim / 2 : kPrefillKeys * kHeadDim;
    static constexpr int kVTile = kMmaLow ? 16 : kPrefillKeys * kHeadDim;
    __nv_bfloat16 k[kStages][kKTile];
    __nv_bfloat16 v[kStages][kVTile];
    // Raw cache staging tiles: [64, 128] for E4M3, [64, 64+8] for NVFP4 (stub
    // for BF16).
    static constexpr int kStaged =
        F::kFp4 ? kPrefillKeys * (kFp4HeadBytes + kFp4HeadScales) : (kFp8 ? kPrefillKeys * kHeadDim : 16);
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
    constexpr bool kMma8 = Shared::kMma8;
    constexpr bool kMma4 = Shared::kMma4;
    constexpr bool kMmaLow = Shared::kMmaLow;
    constexpr float kPLog2 = kMmaLow ? kPScaleLog2Fp8 : kPScaleLog2;
    constexpr float kP = kMmaLow ? kPScaleFp8 : kPScale;
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
    // tiles[b] + g * count(b) + t. Lane l of warp 0 counts 4 blocks
    // (4l..4l+3, covering up to 128 >= kPrefillMaxBlocks) and the warp scans.
    if (warp == 0) {
        unsigned c[4] = {0, 0, 0, 0};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int b = 4 * lane + k;
            if (b < blocks_m) {
                c[k] = range_of(b).tiles * kKvHeads;
            }
        }
        const unsigned sum4 = c[0] + c[1] + c[2] + c[3];
        unsigned inclusive = sum4;
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const unsigned v = __shfl_up_sync(0xffffffffu, inclusive, offset);
            inclusive += lane >= offset ? v : 0;
        }
        const unsigned exclusive = inclusive - sum4;
        const unsigned p[4] = {
            exclusive,
            exclusive + c[0],
            exclusive + c[0] + c[1],
            exclusive + c[0] + c[1] + c[2],
        };
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int b = 4 * lane + k;
            if (b <= blocks_m) {
                sm.tiles[b] = p[k];
            }
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
        if constexpr (F::kFp4) {
            constexpr int kScalesAt = kPrefillKeys * kFp4HeadBytes;
            if (tile.input) {
                // For sliding rings (`tile.input`), pack the chunk's new BF16 keys/values
                // into NVFP4 in shared memory so the ring is only updated after prefill
                // attention finishes reading wrapped slots.
                auto pack_group = [](const __nv_bfloat16* src, uint8_t& scale) {
                    float lo[8], hi[8], vals[16];
                    unpack_bf16x8(*reinterpret_cast<const uint4*>(src), lo);
                    unpack_bf16x8(*reinterpret_cast<const uint4*>(src + 8), hi);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        vals[i] = lo[i];
                        vals[8 + i] = hi[i];
                    }
                    return fp4_group(vals, scale);
                };
#pragma unroll
                for (int i = threadIdx.x; i < kPrefillKeys * kFp4HeadScales; i += kThreads) {
                    const int row = i / kFp4HeadScales;
                    const int group = i % kFp4HeadScales;
                    const bool valid = tile.p0 + row < tile.end;
                    uint2 kd = make_uint2(0, 0);
                    uint2 vd = make_uint2(0, 0);
                    uint8_t ks = 0;
                    uint8_t vs = 0;
                    if (valid) {
                        const unsigned source = first + row;
                        const size_t offset = size_t(source) * kKvRow + g.kv_head * kHeadDim + group * 16;
                        kd = pack_group(k_in + offset, ks);
                        vd = pack_group(v_in + offset, vs);
                    }
                    const int dst_off =
                        kMma4 ? k4_offset(row, group / 2) + (group % 2) * 8
                              : row * kFp4HeadBytes + group * 8;
                    *reinterpret_cast<uint2*>(&sm.k8[stage][dst_off]) = kd;
                    *reinterpret_cast<uint2*>(&sm.v8[stage][dst_off]) = vd;
                    sm.k8[stage][kScalesAt + row * kFp4HeadScales + group] = ks;
                    sm.v8[stage][kScalesAt + row * kFp4HeadScales + group] = vs;
                }
            } else {
                // 4 chunks of 32 E2M1 values per row (swizzled for `kMma4`'s ldmatrix,
                // unswizzled for F16 decode), then the row's 8 scales after the tile's data.
                constexpr int kChunks = kFp4HeadBytes / 16;
                const int chunk = threadIdx.x % kChunks;
#pragma unroll
                for (int row = threadIdx.x / kChunks; row < kPrefillKeys; row += kThreads / kChunks) {
                    const bool valid = tile.p0 + row < tile.end;
                    unsigned source = first + row;
                    source = source >= wrap ? source - wrap : source;
                    const unsigned slot = valid ? source : 0;
                    const size_t offset = fp4_data(slot, g.kv_head) + chunk * 16;
                    const int dst_off = kMma4 ? k4_offset(row, chunk) : row * kFp4HeadBytes + chunk * 16;
                    copy_async16(&sm.k8[stage][dst_off], k_cache + offset, valid);
                    copy_async16(&sm.v8[stage][dst_off], v_cache + offset, valid);
                    if (chunk == 0) {
                        const size_t scales = fp4_scales(slot, g.kv_head);
                        copy_async8(&sm.k8[stage][kScalesAt + row * kFp4HeadScales], k_cache + scales, valid);
                        copy_async8(&sm.v8[stage][kScalesAt + row * kFp4HeadScales], v_cache + scales, valid);
                    }
                }
            }
        } else if constexpr (kFp8) {
            // 8 chunks of 16 E4M3 values per row. For sliding rings (`tile.input`), convert
            // the chunk's new BF16 keys/values to E4M3 on the fly into shared memory so the
            // ring is only updated after prefill attention finishes reading wrapped slots.
            constexpr int kChunks = kHeadDim / 16;
            const int chunk = threadIdx.x % kChunks;
#pragma unroll
            for (int row = threadIdx.x / kChunks; row < kPrefillKeys; row += kThreads / kChunks) {
                const bool valid = tile.p0 + row < tile.end;
                unsigned source = first + row;
                source = source >= wrap ? source - wrap : source;
                if (tile.input) {
                    uint4 kk = make_uint4(0, 0, 0, 0);
                    uint4 vv = make_uint4(0, 0, 0, 0);
                    if (valid) {
                        const size_t offset = size_t(source) * kKvRow + g.kv_head * kHeadDim + chunk * 16;
                        auto pack16 = [](const __nv_bfloat16* src) {
                            float a[8], b[8];
                            unpack_bf16x8(*reinterpret_cast<const uint4*>(src), a);
                            unpack_bf16x8(*reinterpret_cast<const uint4*>(src + 8), b);
                            return make_uint4(pack_e4m3x4(a[0], a[1], a[2], a[3]),
                                              pack_e4m3x4(a[4], a[5], a[6], a[7]),
                                              pack_e4m3x4(b[0], b[1], b[2], b[3]),
                                              pack_e4m3x4(b[4], b[5], b[6], b[7]));
                        };
                        kk = pack16(k_in + offset);
                        vv = pack16(v_in + offset);
                    }
                    if constexpr (kMma8) {
                        *reinterpret_cast<uint4*>(&sm.k8[stage][k8_offset(row, chunk)]) = kk;
                        *reinterpret_cast<uint4*>(&sm.v8[stage][v8_offset(row, chunk)]) = vv;
                    } else {
                        *reinterpret_cast<uint4*>(&sm.k8[stage][row * kHeadDim + chunk * 16]) = kk;
                        *reinterpret_cast<uint4*>(&sm.v8[stage][row * kHeadDim + chunk * 16]) = vv;
                    }
                } else {
                    const size_t offset = size_t(valid ? source : 0) * kKvRow + g.kv_head * kHeadDim + chunk * 16;
                    if constexpr (kMma8) {
                        copy_async16(&sm.k8[stage][k8_offset(row, chunk)], k_cache + offset, valid);
                        copy_async16(&sm.v8[stage][v8_offset(row, chunk)], v_cache + offset, valid);
                    } else {
                        copy_async16(&sm.k8[stage][row * kHeadDim + chunk * 16], k_cache + offset, valid);
                        copy_async16(&sm.v8[stage][row * kHeadDim + chunk * 16], v_cache + offset, valid);
                    }
                }
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
    // dims 16s + 2(lane % 4) + 8(j >> 1) and the next. The FP8 MMA variant
    // holds E4M3 fragments instead: register j of k-step s holds row (j & 1),
    // dims 32s + 4(lane % 4) + 16(j >> 1) and the next 3, and qs[s] the UE8M0
    // scale of row g (lane t = 0) or g + 8 (lane t = 1) for those 32 dims.
    // The FP4 MMA variant holds E2M1 fragments: register j of k-step ks holds
    // row (j & 1), dims 64*ks + 8*(lane % 4) + 32*(j >> 1) and the next 7,
    // qs4[ks] the 4 packed UE4M3 scales of row g (t = 0) or g + 8 (t = 1), and
    // q_outer[i] the per-row outer scale folded into `row_score_scale`.
    uint32_t qa[kMmaLow ? 1 : kHeadDim / 16][4];
    uint32_t qa8[kMma8 ? kHeadDim / 32 : 1][4];
    uint32_t qs[kMma8 ? kHeadDim / 32 : 1];
    uint32_t qa4[kMma4 ? kHeadDim / 64 : 1][4];
    uint32_t qs4[kMma4 ? kHeadDim / 64 : 1];
    float q_outer[2] = {1.f, 1.f};
    auto load_q4 = [&](const PrefillSegment& g) {
        uint32_t scale_packed[2][kHeadDim / 64];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned mg = g.block_m * kRows + warp * 16 + lane / 4 + 8 * i;
            const bool valid = mg < total_m;
            const size_t head = size_t(valid ? mg / kGroup : 0) * kHeads + g.kv_head * kGroup + mg % kGroup;
            const __nv_bfloat16* row = q + head * kHeadDim + (lane % 4) * 8;
            float x[2][2][8];
            float gmax[2][2];
            float row_amax = 0.f;
#pragma unroll
            for (int ks = 0; ks < kHeadDim / 64; ++ks) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint4 raw =
                        valid ? *reinterpret_cast<const uint4*>(row + 64 * ks + 32 * h) : make_uint4(0, 0, 0, 0);
                    unpack_bf16x8(raw, x[ks][h]);
                    float am = 0.f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        am = fmaxf(am, fabsf(x[ks][h][j]));
                    }
                    // Lanes (t=0,1) hold group 0/2 (16 dims); lanes (t=2,3) hold group 1/3.
                    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
                    gmax[ks][h] = am;
                    row_amax = fmaxf(row_amax, am);
                }
            }
            row_amax = fmaxf(row_amax, __shfl_xor_sync(0xffffffffu, row_amax, 2));
            // Per-row outer scale maps the row's largest group scale (amax / 6) to 256,
            // below E4M3's 448, keeping small groups in E4M3's normal range.
            const float enc = row_amax > 0.f ? fminf(__fdiv_rn(1536.f, row_amax), 1e35f) : 0.f;
            q_outer[i] = enc > 0.f ? __fdiv_rn(1.f, enc) : 0.f;
#pragma unroll
            for (int ks = 0; ks < kHeadDim / 64; ++ks) {
                uint8_t sc[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    sc[h] = fp4_scale_of(gmax[ks][h] * enc);
                    const float sv = e4m3_value(sc[h]);
                    const float inv = sv == 0.f ? 0.f : __fdiv_rn(enc, sv);
                    float scaled[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        scaled[j] = x[ks][h][j] * inv;
                    }
                    qa4[ks][2 * h + i] = pack_e2m1x8(scaled);
                }
                const uint32_t s0 = sc[0];
                const uint32_t s1 = __shfl_xor_sync(0xffffffffu, s0, 2);
                const uint32_t s2 = sc[1];
                const uint32_t s3 = __shfl_xor_sync(0xffffffffu, s2, 2);
                scale_packed[i][ks] = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
            }
        }
#pragma unroll
        for (int ks = 0; ks < kHeadDim / 64; ++ks) {
            qs4[ks] = (lane & 1) ? scale_packed[1][ks] : scale_packed[0][ks];
        }
    };
    auto load_q8 = [&](const PrefillSegment& g) {
#if SIPHON_SM89
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned mg = g.block_m * kRows + warp * 16 + lane / 4 + 8 * i;
            const bool valid = mg < total_m;
            const size_t head = size_t(valid ? mg / kGroup : 0) * kHeads + g.kv_head * kGroup + mg % kGroup;
            const __nv_bfloat16* row = q + head * kHeadDim + (lane % 4) * 4;
            float x[kHeadDim / 32][8];
            float amax = 0.f;
#pragma unroll
            for (int s = 0; s < kHeadDim / 32; ++s) {
                float lo4[4];
                float hi4[4];
                unpack_bf16x4(valid ? *reinterpret_cast<const uint2*>(row + 32 * s) : make_uint2(0, 0), lo4);
                unpack_bf16x4(valid ? *reinterpret_cast<const uint2*>(row + 32 * s + 16) : make_uint2(0, 0), hi4);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    x[s][j] = lo4[j];
                    x[s][4 + j] = hi4[j];
                    amax = fmaxf(amax, fmaxf(fabsf(lo4[j]), fabsf(hi4[j])));
                }
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            const float sc = __fmul_rn(amax, 1.f / 448.f);
            const float inv = amax > 0.f ? __fdiv_rn(448.f, amax) : 0.f;
            q_outer[i] = sc;
#pragma unroll
            for (int s = 0; s < kHeadDim / 32; ++s) {
                qa8[s][i] = pack_e4m3x4(x[s][0] * inv, x[s][1] * inv, x[s][2] * inv, x[s][3] * inv);
                qa8[s][2 + i] = pack_e4m3x4(x[s][4] * inv, x[s][5] * inv, x[s][6] * inv, x[s][7] * inv);
            }
        }
#pragma unroll
        for (int s = 0; s < kHeadDim / 32; ++s) {
            qs[s] = 0x7f7f7f7fu;
        }
#else
        uint32_t scale[2][kHeadDim / 32];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned mg = g.block_m * kRows + warp * 16 + lane / 4 + 8 * i;
            const bool valid = mg < total_m;
            const size_t head = size_t(valid ? mg / kGroup : 0) * kHeads + g.kv_head * kGroup + mg % kGroup;
            const __nv_bfloat16* row = q + head * kHeadDim + (lane % 4) * 4;
#pragma unroll
            for (int s = 0; s < kHeadDim / 32; ++s) {
                float x[8];
                float lo4[4];
                float hi4[4];
                unpack_bf16x4(valid ? *reinterpret_cast<const uint2*>(row + 32 * s) : make_uint2(0, 0), lo4);
                unpack_bf16x4(valid ? *reinterpret_cast<const uint2*>(row + 32 * s + 16) : make_uint2(0, 0), hi4);
                float amax = 0.f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    x[j] = lo4[j];
                    x[4 + j] = hi4[j];
                    amax = fmaxf(amax, fmaxf(fabsf(lo4[j]), fabsf(hi4[j])));
                }
                // The quad holds the row's 32 dims of this block.
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                // UE8M0 of amax / 448 rounded up, as quantize_rows_mxfp8; at
                // least 2^-126, so that its inverse is finite.
                const float raw = __fmul_rn(amax, 1.f / 448.f);
                const int exponent = max(((__float_as_int(raw) + 0x007fffff) & 0x7f800000) >> 23, 1);
                const float inverse = __int_as_float((254 - exponent) << 23);
                scale[i][s] = unsigned(exponent);
                qa8[s][i] = pack_e4m3x4(x[0] * inverse, x[1] * inverse, x[2] * inverse, x[3] * inverse);
                qa8[s][2 + i] = pack_e4m3x4(x[4] * inverse, x[5] * inverse, x[6] * inverse, x[7] * inverse);
            }
        }
#pragma unroll
        for (int s = 0; s < kHeadDim / 32; ++s) {
            qs[s] = lane & 1 ? scale[1][s] : scale[0][s];
        }
#endif
    };
    auto load_q = [&](const PrefillSegment& g) {
        if constexpr (kMma4) {
            load_q4(g);
        } else if constexpr (kMma8) {
            load_q8(g);
        } else {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const unsigned mg = g.block_m * kRows + warp * 16 + lane / 4 + 8 * i;
                const bool valid = mg < total_m;
                const size_t head = size_t(valid ? mg / kGroup : 0) * kHeads + g.kv_head * kGroup + mg % kGroup;
                const __nv_bfloat16* row = q + head * kHeadDim + (lane % 4) * 2;
#pragma unroll
                for (int s = 0; s < kHeadDim / 16; ++s) {
                    uint32_t lo = valid ? *reinterpret_cast<const uint32_t*>(row + 16 * s) : 0;
                    uint32_t hi = valid ? *reinterpret_cast<const uint32_t*>(row + 16 * s + 8) : 0;
                    if constexpr (F::kFp4) {
                        lo = bf16x2_to_f16x2(lo);
                        hi = bf16x2_to_f16x2(hi);
                    }
                    qa[s][i] = lo;
                    qa[s][2 + i] = hi;
                }
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

    // O += P V on FP8 for one tile, P as E4M3 A fragments: k-step kk covers
    // keys 32kk..32kk+31 in the order of the transposed values in `vt`.
    auto pv8 = [&](float (&o)[kHeadDim / 8][4], const uint32_t (&pa)[kPrefillKeys / 32][4], const uint8_t* vt) {
#pragma unroll
        for (int kk = 0; kk < kPrefillKeys / 32; ++kk) {
#pragma unroll
            for (int dp = 0; dp < kHeadDim / 16; ++dp) {
                uint32_t b[4];
                const int dim = dp * 16 + (lane & 7) + 8 * (lane >> 4);
                ldmatrix_x4(b, vt + vt_offset(dim, 2 * kk + ((lane >> 3) & 1)));
                mma_fp8(o[2 * dp], pa[kk], b[0], b[1], 0x7f7f7f7fu);
                mma_fp8(o[2 * dp + 1], pa[kk], b[2], b[3], 0x7f7f7f7fu);
            }
        }
    };

    // BF16, FP8-MMA and FP4-MMA blocks run the two halves of their warps half a
    // tile apart: warps 4-7 ("late" warps) compute each tile's P V at the start
    // of the next tile. Each scheduler holds warp w and w + 4, so one warp's
    // scores and softmax run while the other's products keep the tensor cores
    // busy. BF16 late warps read `sm.v[1 - stage]` directly, so they sync on
    // barrier 1 before refilling it; FP8-MMA and FP4-MMA keep two transposed
    // value stages in `sm.k[0..1]`, so `load(next, 1 - stage)` never conflicts.
    constexpr bool kSkewBf16 = !kFp8;
    constexpr bool kSkewLow = kMmaLow;
    constexpr bool kSkew = kSkewBf16 || kSkewLow;
    const bool late = kSkew && warp >= kWarps / 2;
    uint32_t pending[kMmaLow ? 1 : kPrefillKeys / 16][4];
    uint32_t pending8[kMmaLow ? kPrefillKeys / 32 : 1][4];
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
                l[i] = lane % 4 == 0 ? kP : 0.f;
            }
        }
        float o[kHeadDim / 8][4] = {};
        // Capture this segment's score scale before `load_q(next_seg)` updates `q_outer`
        // on the segment's last tile.
        constexpr bool kHasQOuter = kMma4 || (SIPHON_SM89 && kMma8);
        const float row_score_scale[2] = {
            kHasQOuter ? score_scale * q_outer[0] : score_scale,
            kHasQOuter ? score_scale * q_outer[1] : score_scale,
        };

        for (unsigned t = seg.t_begin; t < seg.t_end; ++t) {
            const PrefillTile tile = tile_at(seg, t);
            copy_async_wait_all();
            __syncthreads();
            // The next tile, of this segment or the next one.
            const bool last_tile = t + 1 == seg.t_end;
            const bool more = !last_tile || has_next;
            const PrefillSegment& next_owner = last_tile ? next_seg : seg;
            const PrefillTile next = tile_at(next_owner, last_tile ? next_seg.t_begin : t + 1);
            if constexpr (kSkewBf16) {
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
            if constexpr (kMma8) {
                // Transposes the staged values to [dim][key] in stage `stage`, keys in the
                // MMA order: lane (t, kk, h, dim quad parity) = (lane % 4, lane / 8 % 2,
                // lane / 4 % 2, lane / 16) gathers stored positions 32kk + 16h + 4t + j
                // (keys 32kk + 16h + 8 (j / 2) + 2t + j % 2) of 4 dims and writes one word
                // per dim.
                const int t = lane % 4;
                const int base_key = 32 * (lane / 8 % 2) + 16 * (lane / 4 % 2) + 2 * t;
                const int position = 32 * (lane / 8 % 2) + 16 * (lane / 4 % 2) + 4 * t;
                uint8_t* vt_dst = reinterpret_cast<uint8_t*>(sm.k[stage]);
#pragma unroll
                for (int it = 0; it < 2; ++it) {
                    const int quad = 2 * (warp + kWarps * it) + lane / 16;
                    uint32_t w[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int key = base_key + 8 * (j / 2) + j % 2;
                        w[j] = *reinterpret_cast<const uint32_t*>(&sm.v8[stage][v8_offset(key, quad / 4) + 4 * (quad % 4)]);
                    }
                    const uint32_t t0 = prmt(w[0], w[1], 0x5140u);
                    const uint32_t t1 = prmt(w[2], w[3], 0x5140u);
                    const uint32_t t2 = prmt(w[0], w[1], 0x7362u);
                    const uint32_t t3 = prmt(w[2], w[3], 0x7362u);
                    const uint32_t out4[4] = {prmt(t0, t1, 0x5410u), prmt(t0, t1, 0x7632u), prmt(t2, t3, 0x5410u),
                                              prmt(t2, t3, 0x7632u)};
#pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        const int dim = 4 * quad + d;
                        *reinterpret_cast<uint32_t*>(vt_dst + vt_offset(dim, position / 16) + position % 16) = out4[d];
                    }
                }
                __syncthreads();
            } else if constexpr (kMma4) {
                // Decodes and transposes the staged NVFP4 values directly into [dim][key]
                // E4M3 in stage `stage`, matching `pv8`'s key order. Each quad of 4 dims
                // lies inside one 16-dim NVFP4 scale group (`quad / 4`) and occupies 2 bytes.
                constexpr int kScalesAt = kPrefillKeys * kFp4HeadBytes;
                const int t = lane % 4;
                const int base_key = 32 * (lane / 8 % 2) + 16 * (lane / 4 % 2) + 2 * t;
                const int position = 32 * (lane / 8 % 2) + 16 * (lane / 4 % 2) + 4 * t;
                const uint8_t* v4 = sm.v8[stage];
                uint8_t* vt_dst = reinterpret_cast<uint8_t*>(sm.k[stage]);
#pragma unroll
                for (int it = 0; it < 2; ++it) {
                    const int quad = 2 * (warp + kWarps * it) + lane / 16;
                    const int unit = quad / 8;
                    const int sub = 2 * (quad % 8);
                    const int group = quad / 4;
                    uint32_t w[4];
#pragma unroll
                    for (int p = 0; p < 2; ++p) {
                        const int key0 = base_key + 8 * p;
                        const int key1 = key0 + 1;
                        const uint16_t r0 = *reinterpret_cast<const uint16_t*>(&v4[k4_offset(key0, unit) + sub]);
                        const uint16_t r1 = *reinterpret_cast<const uint16_t*>(&v4[k4_offset(key1, unit) + sub]);
                        uint32_t h[4];
                        e2m1x8_to_f16x8(uint32_t(r0) | (uint32_t(r1) << 16), h);
                        const __half2 sc0 = e4m3_half2(v4[kScalesAt + key0 * kFp4HeadScales + group]);
                        const __half2 sc1 = e4m3_half2(v4[kScalesAt + key1 * kFp4HeadScales + group]);
                        w[2 * p] = f16x4_to_e4m3x4(hmul2_bits(h[0], sc0), hmul2_bits(h[1], sc0));
                        w[2 * p + 1] = f16x4_to_e4m3x4(hmul2_bits(h[2], sc1), hmul2_bits(h[3], sc1));
                    }
                    const uint32_t t0 = prmt(w[0], w[1], 0x5140u);
                    const uint32_t t1 = prmt(w[2], w[3], 0x5140u);
                    const uint32_t t2 = prmt(w[0], w[1], 0x7362u);
                    const uint32_t t3 = prmt(w[2], w[3], 0x7362u);
                    const uint32_t out4[4] = {prmt(t0, t1, 0x5410u), prmt(t0, t1, 0x7632u), prmt(t2, t3, 0x5410u),
                                              prmt(t2, t3, 0x7632u)};
#pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        const int dim = 4 * quad + d;
                        *reinterpret_cast<uint32_t*>(vt_dst + vt_offset(dim, position / 16) + position % 16) = out4[d];
                    }
                }
                __syncthreads();
            } else if constexpr (F::kFp4) {
                // Each group of 16 elements decodes directly to two F16 chunks for both K and V.
                constexpr int kScalesAt = kPrefillKeys * kFp4HeadBytes;
                for (int i = threadIdx.x; i < kPrefillKeys * kFp4HeadScales; i += kThreads) {
                    const int row = i / kFp4HeadScales;
                    const int group = i % kFp4HeadScales;
#pragma unroll
                    for (int kv = 0; kv < 2; ++kv) {
                        const uint8_t* staged = kv == 0 ? sm.k8[stage] : sm.v8[stage];
                        const uint2 raw = *reinterpret_cast<const uint2*>(&staged[row * kFp4HeadBytes + group * 8]);
                        const __half2 sc = e4m3_half2(staged[kScalesAt + row * kFp4HeadScales + group]);
                        uint32_t h[8];
                        uint32_t lo[4];
                        uint32_t hi[4];
                        e2m1x8_to_f16x8(raw.x, lo);
                        e2m1x8_to_f16x8(raw.y, hi);
#pragma unroll
                        for (int w = 0; w < 4; ++w) {
                            h[w] = hmul2_bits(lo[w], sc);
                            h[4 + w] = hmul2_bits(hi[w], sc);
                        }
                        __nv_bfloat16* dst = kv == 0 ? sm.k[0] : sm.v[0];
                        *reinterpret_cast<uint4*>(&dst[swizzle(row, 2 * group)]) = make_uint4(h[0], h[1], h[2], h[3]);
                        *reinterpret_cast<uint4*>(&dst[swizzle(row, 2 * group + 1)]) =
                            make_uint4(h[4], h[5], h[6], h[7]);
                    }
                }
                __syncthreads();
            } else if constexpr (kFp8) {
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
            if constexpr (kSkewLow) {
                // Late warps finish the previous tile's P V from `sm.k[1 - stage]` while
                // early warps start this tile's Q K^T and softmax.
                if (late && has_pending) {
                    pv8(o, pending8, reinterpret_cast<const uint8_t*>(sm.k[1 - stage]));
                    has_pending = false;
                }
            }
            const __nv_bfloat16* keys = sm.k[kFp8 ? 0 : stage];
            const __nv_bfloat16* values = sm.v[kFp8 ? 0 : stage];

            // S = Q K^T: n-tile n holds keys 8n..8n+7.
            float s[kPrefillKeys / 8][4] = {};
            if constexpr (kMma4) {
                const uint8_t* keys4 = sm.k8[stage];
                constexpr int kScalesAt = kPrefillKeys * kFp4HeadBytes;
                const int g = lane / 4;
                const int t = lane % 4;
                // Lane (g, t=0) loads ks=0 scales (bytes 0..3) of key 8n + g;
                // lane (g, t=1) loads ks=1 scales (bytes 4..7). Conflict-free.
                uint32_t sb[kPrefillKeys / 8];
#pragma unroll
                for (int n = 0; n < kPrefillKeys / 8; ++n) {
                    sb[n] = *reinterpret_cast<const uint32_t*>(keys4 + kScalesAt + (8 * n + g) * kFp4HeadScales +
                                                               4 * (t & 1));
                }
#pragma unroll
                for (int np = 0; np < kPrefillKeys / 16; ++np) {
                    uint32_t b[4];
                    const int key = np * 16 + (lane & 7) + 8 * (lane >> 4);
                    ldmatrix_x4(b, keys4 + k4_offset(key, (lane >> 3) & 1));
                    mma_fp4<0>(s[2 * np], qa4[0], b[0], b[1], qs4[0], sb[2 * np]);
                    mma_fp4<0>(s[2 * np + 1], qa4[0], b[2], b[3], qs4[0], sb[2 * np + 1]);
                }
#pragma unroll
                for (int np = 0; np < kPrefillKeys / 16; ++np) {
                    uint32_t b[4];
                    const int key = np * 16 + (lane & 7) + 8 * (lane >> 4);
                    ldmatrix_x4(b, keys4 + k4_offset(key, 2 + ((lane >> 3) & 1)));
                    mma_fp4<1>(s[2 * np], qa4[1], b[0], b[1], qs4[1], sb[2 * np]);
                    mma_fp4<1>(s[2 * np + 1], qa4[1], b[2], b[3], qs4[1], sb[2 * np + 1]);
                }
            } else if constexpr (kMma8) {
                const uint8_t* keys8 = sm.k8[stage];
#pragma unroll
                for (int ks = 0; ks < kHeadDim / 32; ++ks) {
#pragma unroll
                    for (int np = 0; np < kPrefillKeys / 16; ++np) {
                        uint32_t b[4];
                        const int key = np * 16 + (lane & 7) + 8 * (lane >> 4);
                        ldmatrix_x4(b, keys8 + k8_offset(key, 2 * ks + ((lane >> 3) & 1)));
                        mma_fp8(s[2 * np], qa8[ks], b[0], b[1], qs[ks]);
                        mma_fp8(s[2 * np + 1], qa8[ks], b[2], b[3], qs[ks]);
                    }
                }
            } else {
#pragma unroll
                for (int ks = 0; ks < kHeadDim / 16; ++ks) {
#pragma unroll
                    for (int np = 0; np < kPrefillKeys / 16; ++np) {
                        uint32_t b[4];
                        ldmatrix_x4(b, keys + swizzle(np * 16 + (lane >> 4) * 8 + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
                        if constexpr (F::kFp4) {
                            mma_f16(s[2 * np], qa[ks], b[0], b[1]);
                            mma_f16(s[2 * np + 1], qa[ks], b[2], b[3]);
                        } else {
                            mma_bf16(s[2 * np], qa[ks], b[0], b[1]);
                            mma_bf16(s[2 * np + 1], qa[ks], b[2], b[3]);
                        }
                    }
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
                        s[n][e] *= row_score_scale[e / 2];
                    }
                }
            } else if (tile.full && tile.near) {
#pragma unroll
                for (int n = 0; n < kPrefillKeys / 8; ++n) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float b = __bfloat162float(bias_base[e / 2][-(n * 8 + e % 2)]);
                        s[n][e] = fmaf(b, kLog2e, s[n][e] * row_score_scale[e / 2]);
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
                        float x = s[n][e] * row_score_scale[i];
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
            const float offset[2] = {m[0] - kPLog2, m[1] - kPLog2};
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

            if constexpr (kMmaLow) {
                // A register j of k-step kk: row j % 2, n-tiles 4kk + 2 (j / 2)
                // and 4kk + 2 (j / 2) + 1.
                uint32_t pa8[kPrefillKeys / 32][4];
#pragma unroll
                for (int kk = 0; kk < kPrefillKeys / 32; ++kk) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int n = 4 * kk + 2 * (j / 2);
                        const int e = 2 * (j % 2);
                        pa8[kk][j] = pack_e4m3x4(s[n][e], s[n][e + 1], s[n + 1][e], s[n + 1][e + 1]);
                    }
                }
                if (late) {
#pragma unroll
                    for (int kk = 0; kk < kPrefillKeys / 32; ++kk) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            pending8[kk][j] = pa8[kk][j];
                        }
                    }
                    has_pending = true;
                } else {
                    pv8(o, pa8, reinterpret_cast<const uint8_t*>(sm.k[stage]));
                }
            } else {
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
            }
            stage ^= 1;
        }
        // The late warps' last tile, whose values sit in the stage before the
        // next segment's.
        if (has_pending) {
            if constexpr (kSkewLow) {
                pv8(o, pending8, reinterpret_cast<const uint8_t*>(sm.k[1 - stage]));
            } else {
                pv(o, pending, sm.v[1 - stage]);
            }
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
                // Fold the unit's partials in block order, this block's own read back
                // like the others, so that the sum order does not depend on which
                // block finishes last.
                {
                    const float* first = partials + size_t(2 * first_block + (seg.unit_begin > begin_of(first_block))) * kSlot;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        m[i] = __ldcg(first + kRows * kHeadDim + local[i]);
                        l[i] = __ldcg(first + kRows * kHeadDim + kRows + local[i]);
                    }
                    const float4* first_o = reinterpret_cast<const float4*>(first) + warp * (kHeadDim / 8) * 32 + lane;
#pragma unroll
                    for (int n = 0; n < kHeadDim / 8; ++n) {
                        const float4 p = __ldcg(first_o + n * 32);
                        o[n][0] = p.x;
                        o[n][1] = p.y;
                        o[n][2] = p.z;
                        o[n][3] = p.w;
                    }
                }
                for (unsigned b = first_block + 1; b <= last_block; ++b) {
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

// The NVFP4 variant of `kv_append_rows_bf16`: thread i of a row's 64 rounds group i
// (16 elements of head i / 8) of its key and value as kv_round_bf16's NVFP4 mode does.
extern "C" __global__ void __launch_bounds__(64)
kv_append_rows_fp4(const __nv_bfloat16* k, const __nv_bfloat16* v, uint8_t* k_cache, uint8_t* v_cache,
                   const unsigned* start, unsigned slots) {
    if (blockIdx.x + slots < gridDim.x) {
        return;
    }
    const int head = threadIdx.x / kFp4HeadScales;
    const int group = threadIdx.x % kFp4HeadScales;
    const size_t src = size_t(blockIdx.x) * kKvRow + threadIdx.x * 16;
    const unsigned slot = (*start + blockIdx.x) % slots;
#pragma unroll
    for (int kv = 0; kv < 2; ++kv) {
        const __nv_bfloat16* x = kv == 0 ? k : v;
        float lo[8];
        float hi[8];
        unpack_bf16x8(*reinterpret_cast<const uint4*>(x + src), lo);
        unpack_bf16x8(*reinterpret_cast<const uint4*>(x + src + 8), hi);
        float vals[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            vals[i] = lo[i];
            vals[8 + i] = hi[i];
        }
        uint8_t scale;
        const uint2 data = fp4_group(vals, scale);
        uint8_t* cache = kv == 0 ? k_cache : v_cache;
        *reinterpret_cast<uint2*>(cache + fp4_data(slot, head) + group * 8) = data;
        cache[fp4_scales(slot, head) + group] = scale;
    }
}

// `prefill_attention_bf16` over NVFP4 caches (global and sliding rings with `input_keys == 1`).
extern "C" __global__ void __launch_bounds__(kPrefillWarps * 32)
prefill_attention_fp4(const __nv_bfloat16* q, const __nv_bfloat16* k_in, const __nv_bfloat16* v_in,
                      const uint8_t* k_cache, const uint8_t* v_cache, const __nv_bfloat16* bias,
                      const __nv_bfloat16* sinks, float* partials, unsigned* counters, __nv_bfloat16* out,
                      const unsigned* start, unsigned rows, unsigned slots, unsigned window, unsigned extent,
                      unsigned input_keys, float scale) {
    prefill_attention<Fp4Format, kPrefillWarps>(q, k_in, v_in, k_cache, v_cache, bias, sinks, partials,
                                                counters, out, start, rows, slots, window, extent, input_keys,
                                                scale);
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

// `prefill_attention_fp8` with both products on FP8 tensor cores (see above).
extern "C" __global__ void __launch_bounds__(kPrefillWarps * 32)
prefill_attention_fp8_mma(const __nv_bfloat16* q, const __nv_bfloat16* k_in, const __nv_bfloat16* v_in,
                          const uint8_t* k_cache, const uint8_t* v_cache, const __nv_bfloat16* bias,
                          const __nv_bfloat16* sinks, float* partials, unsigned* counters, __nv_bfloat16* out,
                          const unsigned* start, unsigned rows, unsigned slots, unsigned window, unsigned extent,
                          unsigned input_keys, float scale) {
    prefill_attention<Fp8MmaFormat, kPrefillWarps>(q, k_in, v_in, k_cache, v_cache, bias, sinks, partials,
                                                   counters, out, start, rows, slots, window, extent, input_keys,
                                                   scale);
}

// `prefill_attention_fp4` with Q K^T on native Blackwell NVFP4 tensor cores
// (`kind::mxf4nvf4`, m16n8k64) and P V on block-scaled FP8 tensor cores
// (`kind::mxf8f6f4`, m16n8k32). Every key comes from the NVFP4 cache.
extern "C" __global__ void __launch_bounds__(kPrefillWarps * 32)
prefill_attention_fp4_mma(const __nv_bfloat16* q, const __nv_bfloat16* k_in, const __nv_bfloat16* v_in,
                          const uint8_t* k_cache, const uint8_t* v_cache, const __nv_bfloat16* bias,
                          const __nv_bfloat16* sinks, float* partials, unsigned* counters, __nv_bfloat16* out,
                          const unsigned* start, unsigned rows, unsigned slots, unsigned window, unsigned extent,
                          unsigned input_keys, float scale) {
    prefill_attention<Fp4MmaFormat, kPrefillWarps>(q, k_in, v_in, k_cache, v_cache, bias, sinks, partials,
                                                   counters, out, start, rows, slots, window, extent, input_keys,
                                                   scale);
}

// The launcher passes PREFILL_SHARED_BYTES of dynamic shared memory.
static_assert(sizeof(PrefillShared<Bf16Format, kPrefillWarps>) <= 84432);
static_assert(sizeof(PrefillShared<Fp8Format, kPrefillWarps>) <= 84432);
static_assert(sizeof(PrefillShared<Fp8MmaFormat, kPrefillWarps>) <= 84432);
static_assert(sizeof(PrefillShared<Fp4Format, kPrefillWarps>) <= 84432);
static_assert(sizeof(PrefillShared<Fp4MmaFormat, kPrefillWarps>) <= 84432);
static_assert(kPrefillKeys * (kFp4HeadBytes + kFp4HeadScales) <= PrefillShared<Fp4Format, kPrefillWarps>::kStaged);
static_assert(kPrefillKeys * (kFp4HeadBytes + kFp4HeadScales) <= PrefillShared<Fp4MmaFormat, kPrefillWarps>::kStaged);

// ---------------------------------------------------------------------------
// Ragged packed multi-slice prefill KV append & flash attention: runs across
// all S variable-length slices (1 <= S <= 64, sum(row_len) <= 4096) in one
// persistent grid launch so short slices saturate every SM concurrently.
// ---------------------------------------------------------------------------

namespace {

// Upper bound on sum_s ceil(row_len[s] * 3 / 128) across S <= 64 slices totaling <= 4096
// rows: at most 64 + 95 = 159 blocks, rounded up to 192 (6 blocks per warp-0 lane).
constexpr int kPrefillPackedMaxBlocks = 192;

template <class F, int kWarps>
struct alignas(16) PrefillPackedShared {
    static constexpr bool kFp8 = sizeof(typename F::Elem) == 1;
    static constexpr bool kMma8 = IsFp8Mma<F>::value;
    static constexpr int kRows = 16 * kWarps;
    static constexpr int kStages = kFp8 ? 1 : 2;
    static constexpr int kKTile = kMma8 ? kPrefillKeys * kHeadDim / 2 : kPrefillKeys * kHeadDim;
    static constexpr int kVTile = kMma8 ? 16 : kPrefillKeys * kHeadDim;
    __nv_bfloat16 k[kStages][kKTile];
    __nv_bfloat16 v[kStages][kVTile];
    static constexpr int kStaged = kFp8 ? kPrefillKeys * kHeadDim : 16;
    uint8_t k8[2][kStaged];
    uint8_t v8[2][kStaged];
    __nv_bfloat16 bias[kRows * kBiasStride];
    unsigned tiles[kPrefillPackedMaxBlocks + 1];
    uint16_t block_desc[kPrefillPackedMaxBlocks];
    bool last;
};

struct PrefillPackedSegment {
    PrefillRange range;
    unsigned begin;
    unsigned unit_begin;
    unsigned t_begin;
    unsigned t_end;
    unsigned row_start;
    unsigned rows;
    unsigned start;
    unsigned state_idx;
    int gblock;
    int block_m;
    int kv_head;
};

template <class F, int kWarps>
__device__ __forceinline__ void prefill_attention_packed(
    const __nv_bfloat16* q, const __nv_bfloat16* k_in, const __nv_bfloat16* v_in,
    const unsigned long long* states, const PackedSlice* slices, const uint32_t* block_descs,
    const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partials, unsigned* counters,
    __nv_bfloat16* out, unsigned total_blocks_m, unsigned slots, unsigned window, unsigned extent,
    unsigned input_keys, float scale) {
    using Shared = PrefillPackedShared<F, kWarps>;
    constexpr int kThreads = kWarps * 32;
    constexpr int kRows = Shared::kRows;
    constexpr bool kFp8 = Shared::kFp8;
    constexpr bool kMma8 = Shared::kMma8;
    constexpr float kPLog2 = kMma8 ? kPScaleLog2Fp8 : kPScaleLog2;
    constexpr float kP = kMma8 ? kPScaleFp8 : kPScale;
    constexpr int kSlot = kRows * (kHeadDim + 2);
    extern __shared__ __align__(16) unsigned char prefill_packed_smem[];
    Shared& sm = *reinterpret_cast<Shared*>(prefill_packed_smem);

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;

    auto range_of = [&](const PackedSlice& sl, int block_m) {
        PrefillRange r;
        const unsigned total_m = sl.row_len * kGroup;
        const unsigned cache_end = input_keys ? sl.kv_start : sl.kv_start + sl.row_len;
        const unsigned m0 = unsigned(block_m) * kRows;
        r.r_lo = m0 / kGroup;
        r.r_hi = (min(m0 + kRows, total_m) - 1) / kGroup;
        r.hi = sl.kv_start + r.r_hi + 1;
        r.lo = window != 0 && sl.kv_start + r.r_lo + 1 > window
                   ? sl.kv_start + r.r_lo + 1 - window
                   : 0;
        r.a_end = min(r.hi, cache_end);
        r.a_tiles = r.a_end > r.lo ? (r.a_end - r.lo + kPrefillKeys - 1) / kPrefillKeys : 0;
        r.b_lo = max(r.lo, cache_end);
        const unsigned b_tiles = r.hi > r.b_lo ? (r.hi - r.b_lo + kPrefillKeys - 1) / kPrefillKeys : 0;
        r.tiles = r.a_tiles + b_tiles;
        return r;
    };

    // Warp 0 scans the tile counts across all `total_blocks_m <= 192` global blocks
    // (6 blocks per lane), storing prefix sums in `sm.tiles` and descriptors in `sm.block_desc`.
    if (warp == 0) {
        unsigned c[6] = {0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            const int gb = 6 * lane + k;
            if (gb < int(total_blocks_m)) {
                const uint32_t desc = block_descs[gb];
                sm.block_desc[gb] = uint16_t(desc);
                const int si = int(desc & 0xffu);
                const int bm = int(desc >> 8);
                c[k] = range_of(slices[si], bm).tiles * kKvHeads;
            }
        }
        const unsigned sum6 = c[0] + c[1] + c[2] + c[3] + c[4] + c[5];
        unsigned inclusive = sum6;
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const unsigned v = __shfl_up_sync(0xffffffffu, inclusive, offset);
            inclusive += lane >= offset ? v : 0;
        }
        const unsigned exclusive = inclusive - sum6;
        unsigned running = exclusive;
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            const int gb = 6 * lane + k;
            if (gb <= int(total_blocks_m)) {
                sm.tiles[gb] = running;
            }
            running += c[k];
        }
        if (lane == 31 && total_blocks_m == kPrefillPackedMaxBlocks) {
            sm.tiles[kPrefillPackedMaxBlocks] = inclusive;
        }
    }
    __syncthreads();

    const unsigned total = sm.tiles[total_blocks_m];
    const unsigned grid = min(gridDim.x, total);
    if (blockIdx.x >= grid) {
        return;
    }
    auto begin_of = [&](unsigned b) { return unsigned((unsigned long long)b * total / grid); };
    auto block_of = [&](unsigned x) { return unsigned(((unsigned long long)(x + 1) * grid - 1) / total); };
    const unsigned my_begin = begin_of(blockIdx.x);
    const unsigned my_end = begin_of(blockIdx.x + 1);
    const float score_scale = scale * kLog2e;

    int gblock = 0;
    auto segment_at = [&](unsigned x) {
        PrefillPackedSegment g;
        while (sm.tiles[gblock + 1] <= x) {
            ++gblock;
        }
        const uint16_t desc = sm.block_desc[gblock];
        const int si = int(desc & 0xffu);
        const int bm = int(desc >> 8);
        const PackedSlice sl = slices[si];
        g.gblock = gblock;
        g.block_m = bm;
        g.row_start = sl.row_start;
        g.rows = sl.row_len;
        g.start = sl.kv_start;
        g.state_idx = sl.state_idx;
        g.range = range_of(sl, bm);
        g.kv_head = int((x - sm.tiles[gblock]) / g.range.tiles);
        g.unit_begin = sm.tiles[gblock] + g.kv_head * g.range.tiles;
        g.begin = x;
        g.t_begin = x - g.unit_begin;
        g.t_end = min(my_end, g.unit_begin + g.range.tiles) - g.unit_begin;
        return g;
    };

    auto tile_at = [&](const PrefillPackedSegment& g, unsigned t) {
        const PrefillRange& range = g.range;
        PrefillTile tile;
        tile.input = t >= range.a_tiles;
        tile.p0 = tile.input ? range.b_lo + (t - range.a_tiles) * kPrefillKeys : range.lo + t * kPrefillKeys;
        tile.end = tile.input ? range.hi : range.a_end;
        const unsigned last = tile.p0 + kPrefillKeys - 1;
        tile.full = last < tile.end && last <= g.start + range.r_lo &&
                    (window == 0 || g.start + range.r_hi - tile.p0 < window);
        tile.biased = int(g.start + range.r_lo) - int(last) < int(extent);
        tile.near = g.start + range.r_hi - tile.p0 < extent;
        return tile;
    };

    auto k_cache_of = [&](const PrefillPackedSegment& g) {
        return reinterpret_cast<const typename F::Elem*>(states[size_t(g.state_idx) * 6 + 0]);
    };
    auto v_cache_of = [&](const PrefillPackedSegment& g) {
        return reinterpret_cast<const typename F::Elem*>(states[size_t(g.state_idx) * 6 + 1]);
    };
    auto k_in_of = [&](const PrefillPackedSegment& g) { return k_in + size_t(g.row_start) * kKvRow; };
    auto v_in_of = [&](const PrefillPackedSegment& g) { return v_in + size_t(g.row_start) * kKvRow; };

    auto copy_bf16 = [&](const PrefillPackedSegment& g, const PrefillTile& tile, __nv_bfloat16* dst,
                         const __nv_bfloat16* src, int thread, int threads) {
        const unsigned first = tile.input ? tile.p0 - g.start : tile.p0 % slots;
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

    auto load = [&](const PrefillPackedSegment& g, const PrefillTile& tile, int stage) {
        const unsigned first = tile.input ? tile.p0 - g.start : tile.p0 % slots;
        const unsigned wrap = tile.input ? 0xffffffffu : slots;
        const typename F::Elem* kc = k_cache_of(g);
        const typename F::Elem* vc = v_cache_of(g);
        if constexpr (F::kFp4) {
            constexpr int kChunks = kFp4HeadBytes / 16;
            const int chunk = threadIdx.x % kChunks;
#pragma unroll
            for (int row = threadIdx.x / kChunks; row < kPrefillKeys; row += kThreads / kChunks) {
                const bool valid = tile.p0 + row < tile.end;
                unsigned source = first + row;
                source = source >= wrap ? source - wrap : source;
                const unsigned slot = valid ? source : 0;
                const size_t offset = fp4_data(slot, g.kv_head) + chunk * 16;
                copy_async16(&sm.k8[stage][row * kFp4HeadBytes + chunk * 16], kc + offset, valid);
                copy_async16(&sm.v8[stage][row * kFp4HeadBytes + chunk * 16], vc + offset, valid);
                if (chunk == 0) {
                    const size_t scales = fp4_scales(slot, g.kv_head);
                    constexpr int kScalesAt = kPrefillKeys * kFp4HeadBytes;
                    copy_async8(&sm.k8[stage][kScalesAt + row * kFp4HeadScales], kc + scales, valid);
                    copy_async8(&sm.v8[stage][kScalesAt + row * kFp4HeadScales], vc + scales, valid);
                }
            }
        } else if constexpr (kFp8) {
            constexpr int kChunks = kHeadDim / 16;
            const int chunk = threadIdx.x % kChunks;
            const __nv_bfloat16* kin = k_in_of(g);
            const __nv_bfloat16* vin = v_in_of(g);
#pragma unroll
            for (int row = threadIdx.x / kChunks; row < kPrefillKeys; row += kThreads / kChunks) {
                const bool valid = tile.p0 + row < tile.end;
                unsigned source = first + row;
                source = source >= wrap ? source - wrap : source;
                if (tile.input) {
                    uint4 kk = make_uint4(0, 0, 0, 0);
                    uint4 vv = make_uint4(0, 0, 0, 0);
                    if (valid) {
                        const size_t offset = size_t(source) * kKvRow + g.kv_head * kHeadDim + chunk * 16;
                        auto pack16 = [](const __nv_bfloat16* src) {
                            float a[8], b[8];
                            unpack_bf16x8(*reinterpret_cast<const uint4*>(src), a);
                            unpack_bf16x8(*reinterpret_cast<const uint4*>(src + 8), b);
                            return make_uint4(pack_e4m3x4(a[0], a[1], a[2], a[3]),
                                              pack_e4m3x4(a[4], a[5], a[6], a[7]),
                                              pack_e4m3x4(b[0], b[1], b[2], b[3]),
                                              pack_e4m3x4(b[4], b[5], b[6], b[7]));
                        };
                        kk = pack16(kin + offset);
                        vv = pack16(vin + offset);
                    }
                    if constexpr (kMma8) {
                        *reinterpret_cast<uint4*>(&sm.k8[stage][k8_offset(row, chunk)]) = kk;
                        *reinterpret_cast<uint4*>(&sm.v8[stage][v8_offset(row, chunk)]) = vv;
                    } else {
                        *reinterpret_cast<uint4*>(&sm.k8[stage][row * kHeadDim + chunk * 16]) = kk;
                        *reinterpret_cast<uint4*>(&sm.v8[stage][row * kHeadDim + chunk * 16]) = vv;
                    }
                } else {
                    const size_t offset = size_t(valid ? source : 0) * kKvRow + g.kv_head * kHeadDim + chunk * 16;
                    if constexpr (kMma8) {
                        copy_async16(&sm.k8[stage][k8_offset(row, chunk)], kc + offset, valid);
                        copy_async16(&sm.v8[stage][v8_offset(row, chunk)], vc + offset, valid);
                    } else {
                        copy_async16(&sm.k8[stage][row * kHeadDim + chunk * 16], kc + offset, valid);
                        copy_async16(&sm.v8[stage][row * kHeadDim + chunk * 16], vc + offset, valid);
                    }
                }
            }
        } else {
            copy_bf16(g, tile, sm.k[stage], tile.input ? k_in_of(g) : reinterpret_cast<const __nv_bfloat16*>(kc),
                      threadIdx.x, kThreads);
            copy_bf16(g, tile, sm.v[stage], tile.input ? v_in_of(g) : reinterpret_cast<const __nv_bfloat16*>(vc),
                      threadIdx.x, kThreads);
        }
        copy_async_commit();
    };

    auto load_bias = [&](const PrefillPackedSegment& g, const PrefillTile& tile) {
        const unsigned total_m = g.rows * kGroup;
        const unsigned m0 = g.block_m * kRows;
        const int top = int(tile.p0) + kPrefillKeys - 1;
#pragma unroll
        for (int i = lane; i < 16 * kBiasChunks; i += 32) {
            const int row = warp * 16 + i / kBiasChunks;
            const int c = i % kBiasChunks;
            const unsigned mg = m0 + row;
            const unsigned r = mg / kGroup;
            const int chunk = ((int(g.start + r) - top) >> 3) + c;
            const bool valid = mg < total_m && chunk >= 0 && (chunk + 1) * 8 <= int(extent);
            const __nv_bfloat16* src =
                bias + (size_t(g.row_start + r) * kHeads + g.kv_head * kGroup + mg % kGroup) * extent + chunk * 8;
            copy_async16(&sm.bias[row * kBiasStride + c * 8], valid ? src : bias, valid);
        }
        copy_async_commit();
    };

    uint32_t qa[kMma8 ? 1 : kHeadDim / 16][4];
    uint32_t qa8[kMma8 ? kHeadDim / 32 : 1][4];
    uint32_t qs[kMma8 ? kHeadDim / 32 : 1];
    float q_outer[2] = {1.f, 1.f};
    auto load_q8 = [&](const PrefillPackedSegment& g) {
        const unsigned total_m = g.rows * kGroup;
#if SIPHON_SM89
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned mg = g.block_m * kRows + warp * 16 + lane / 4 + 8 * i;
            const bool valid = mg < total_m;
            const size_t head =
                size_t(g.row_start + (valid ? mg / kGroup : 0)) * kHeads + g.kv_head * kGroup + mg % kGroup;
            const __nv_bfloat16* row = q + head * kHeadDim + (lane % 4) * 4;
            float x[kHeadDim / 32][8];
            float amax = 0.f;
#pragma unroll
            for (int s = 0; s < kHeadDim / 32; ++s) {
                float lo4[4];
                float hi4[4];
                unpack_bf16x4(valid ? *reinterpret_cast<const uint2*>(row + 32 * s) : make_uint2(0, 0), lo4);
                unpack_bf16x4(valid ? *reinterpret_cast<const uint2*>(row + 32 * s + 16) : make_uint2(0, 0), hi4);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    x[s][j] = lo4[j];
                    x[s][4 + j] = hi4[j];
                    amax = fmaxf(amax, fmaxf(fabsf(lo4[j]), fabsf(hi4[j])));
                }
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            const float sc = __fmul_rn(amax, 1.f / 448.f);
            const float inv = amax > 0.f ? __fdiv_rn(448.f, amax) : 0.f;
            q_outer[i] = sc;
#pragma unroll
            for (int s = 0; s < kHeadDim / 32; ++s) {
                qa8[s][i] = pack_e4m3x4(x[s][0] * inv, x[s][1] * inv, x[s][2] * inv, x[s][3] * inv);
                qa8[s][2 + i] = pack_e4m3x4(x[s][4] * inv, x[s][5] * inv, x[s][6] * inv, x[s][7] * inv);
            }
        }
#pragma unroll
        for (int s = 0; s < kHeadDim / 32; ++s) {
            qs[s] = 0x7f7f7f7fu;
        }
#else
        uint32_t scale[2][kHeadDim / 32];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned mg = g.block_m * kRows + warp * 16 + lane / 4 + 8 * i;
            const bool valid = mg < total_m;
            const size_t head =
                size_t(g.row_start + (valid ? mg / kGroup : 0)) * kHeads + g.kv_head * kGroup + mg % kGroup;
            const __nv_bfloat16* row = q + head * kHeadDim + (lane % 4) * 4;
#pragma unroll
            for (int s = 0; s < kHeadDim / 32; ++s) {
                float x[8];
                float lo4[4];
                float hi4[4];
                unpack_bf16x4(valid ? *reinterpret_cast<const uint2*>(row + 32 * s) : make_uint2(0, 0), lo4);
                unpack_bf16x4(valid ? *reinterpret_cast<const uint2*>(row + 32 * s + 16) : make_uint2(0, 0), hi4);
                float amax = 0.f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    x[j] = lo4[j];
                    x[4 + j] = hi4[j];
                    amax = fmaxf(amax, fmaxf(fabsf(lo4[j]), fabsf(hi4[j])));
                }
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                const float raw = __fmul_rn(amax, 1.f / 448.f);
                const int exponent = max(((__float_as_int(raw) + 0x007fffff) & 0x7f800000) >> 23, 1);
                const float inverse = __int_as_float((254 - exponent) << 23);
                scale[i][s] = unsigned(exponent);
                qa8[s][i] = pack_e4m3x4(x[0] * inverse, x[1] * inverse, x[2] * inverse, x[3] * inverse);
                qa8[s][2 + i] = pack_e4m3x4(x[4] * inverse, x[5] * inverse, x[6] * inverse, x[7] * inverse);
            }
        }
#pragma unroll
        for (int s = 0; s < kHeadDim / 32; ++s) {
            qs[s] = lane & 1 ? scale[1][s] : scale[0][s];
        }
#endif
    };
    auto load_q = [&](const PrefillPackedSegment& g) {
        if constexpr (kMma8) {
            load_q8(g);
        } else {
            const unsigned total_m = g.rows * kGroup;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const unsigned mg = g.block_m * kRows + warp * 16 + lane / 4 + 8 * i;
                const bool valid = mg < total_m;
                const size_t head =
                    size_t(g.row_start + (valid ? mg / kGroup : 0)) * kHeads + g.kv_head * kGroup + mg % kGroup;
                const __nv_bfloat16* row = q + head * kHeadDim + (lane % 4) * 2;
#pragma unroll
                for (int s = 0; s < kHeadDim / 16; ++s) {
                    uint32_t lo = valid ? *reinterpret_cast<const uint32_t*>(row + 16 * s) : 0;
                    uint32_t hi = valid ? *reinterpret_cast<const uint32_t*>(row + 16 * s + 8) : 0;
                    if constexpr (F::kFp4) {
                        lo = bf16x2_to_f16x2(lo);
                        hi = bf16x2_to_f16x2(hi);
                    }
                    qa[s][i] = lo;
                    qa[s][2 + i] = hi;
                }
            }
        }
    };

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

    auto pv8 = [&](float (&o)[kHeadDim / 8][4], const uint32_t (&pa)[kPrefillKeys / 32][4]) {
#pragma unroll
        for (int kk = 0; kk < kPrefillKeys / 32; ++kk) {
#pragma unroll
            for (int dp = 0; dp < kHeadDim / 16; ++dp) {
                uint32_t b[4];
                const int dim = dp * 16 + (lane & 7) + 8 * (lane >> 4);
                ldmatrix_x4(b, reinterpret_cast<const uint8_t*>(sm.k[0]) + vt_offset(dim, 2 * kk + ((lane >> 3) & 1)));
                mma_fp8(o[2 * dp], pa[kk], b[0], b[1], 0x7f7f7f7fu);
                mma_fp8(o[2 * dp + 1], pa[kk], b[2], b[3], 0x7f7f7f7fu);
            }
        }
    };

    constexpr bool kSkew = !kFp8;
    const bool late = kSkew && warp >= kWarps / 2;
    uint32_t pending[kPrefillKeys / 16][4];
    bool has_pending = false;

    PrefillPackedSegment seg = segment_at(my_begin);
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
        const PrefillPackedSegment next_seg = segment_at(has_next ? seg.unit_begin + seg.t_end : seg.begin);
        const unsigned total_m = seg.rows * kGroup;
        const unsigned m0 = seg.block_m * kRows;
        const int kv_head = seg.kv_head;

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
            position[i] = seg.start + r;
            row_head[i] = size_t(seg.row_start + r) * kHeads + kv_head * kGroup + mg % kGroup;
        }

        float m[2];
        float l[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            m[i] = -FLT_MAX;
            l[i] = 0.f;
            if (seg.t_begin == 0) {
                m[i] = valid_row[i] ? __bfloat162float(sinks[row_head[i]]) * kLog2e : 0.f;
                l[i] = lane % 4 == 0 ? kP : 0.f;
            }
        }
        float o[kHeadDim / 8][4] = {};
        constexpr bool kHasQOuter = SIPHON_SM89 && kMma8;
        const float row_score_scale[2] = {
            kHasQOuter ? score_scale * q_outer[0] : score_scale,
            kHasQOuter ? score_scale * q_outer[1] : score_scale,
        };

        for (unsigned t = seg.t_begin; t < seg.t_end; ++t) {
            const PrefillTile tile = tile_at(seg, t);
            copy_async_wait_all();
            __syncthreads();
            const bool last_tile = t + 1 == seg.t_end;
            const bool more = !last_tile || has_next;
            const PrefillPackedSegment& next_owner = last_tile ? next_seg : seg;
            const PrefillTile next = tile_at(next_owner, last_tile ? next_seg.t_begin : t + 1);
            if constexpr (kSkew) {
                if (more) {
                    copy_bf16(next_owner, next, sm.k[1 - stage],
                              next.input ? k_in_of(next_owner)
                                         : reinterpret_cast<const __nv_bfloat16*>(k_cache_of(next_owner)),
                              threadIdx.x, kThreads);
                    copy_async_commit();
                }
                if (late) {
                    if (has_pending) {
                        pv(o, pending, sm.v[1 - stage]);
                        has_pending = false;
                    }
                    asm volatile("bar.sync 1, %0;" ::"r"(kThreads / 2) : "memory");
                    if (more) {
                        copy_bf16(next_owner, next, sm.v[1 - stage],
                                  next.input ? v_in_of(next_owner)
                                             : reinterpret_cast<const __nv_bfloat16*>(v_cache_of(next_owner)),
                                  threadIdx.x - kThreads / 2, kThreads / 2);
                        copy_async_commit();
                    }
                }
            } else if (more) {
                load(next_owner, next, 1 - stage);
            }
            if constexpr (kMma8) {
                const int tt = lane % 4;
                const int base_key = 32 * (lane / 8 % 2) + 16 * (lane / 4 % 2) + 2 * tt;
                const int pos = 32 * (lane / 8 % 2) + 16 * (lane / 4 % 2) + 4 * tt;
#pragma unroll
                for (int it = 0; it < 2; ++it) {
                    const int quad = 2 * (warp + kWarps * it) + lane / 16;
                    uint32_t w[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int key = base_key + 8 * (j / 2) + j % 2;
                        w[j] = *reinterpret_cast<const uint32_t*>(&sm.v8[stage][v8_offset(key, quad / 4) + 4 * (quad % 4)]);
                    }
                    const uint32_t t0 = prmt(w[0], w[1], 0x5140u);
                    const uint32_t t1 = prmt(w[2], w[3], 0x5140u);
                    const uint32_t t2 = prmt(w[0], w[1], 0x7362u);
                    const uint32_t t3 = prmt(w[2], w[3], 0x7362u);
                    const uint32_t out4[4] = {prmt(t0, t1, 0x5410u), prmt(t0, t1, 0x7632u), prmt(t2, t3, 0x5410u),
                                              prmt(t2, t3, 0x7632u)};
#pragma unroll
                    for (int d = 0; d < 4; ++d) {
                        const int dim = 4 * quad + d;
                        *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(sm.k[0]) + vt_offset(dim, pos / 16) +
                                                     pos % 16) = out4[d];
                    }
                }
                __syncthreads();
            } else if constexpr (F::kFp4) {
                constexpr int kScalesAt = kPrefillKeys * kFp4HeadBytes;
                for (int i = threadIdx.x; i < kPrefillKeys * kFp4HeadScales; i += kThreads) {
                    const int row = i / kFp4HeadScales;
                    const int group = i % kFp4HeadScales;
#pragma unroll
                    for (int kv = 0; kv < 2; ++kv) {
                        const uint8_t* staged = kv == 0 ? sm.k8[stage] : sm.v8[stage];
                        const uint2 raw = *reinterpret_cast<const uint2*>(&staged[row * kFp4HeadBytes + group * 8]);
                        const __half2 sc = e4m3_half2(staged[kScalesAt + row * kFp4HeadScales + group]);
                        uint32_t h[8];
                        uint32_t lo[4];
                        uint32_t hi[4];
                        e2m1x8_to_f16x8(raw.x, lo);
                        e2m1x8_to_f16x8(raw.y, hi);
#pragma unroll
                        for (int w = 0; w < 4; ++w) {
                            h[w] = hmul2_bits(lo[w], sc);
                            h[4 + w] = hmul2_bits(hi[w], sc);
                        }
                        __nv_bfloat16* dst = kv == 0 ? sm.k[0] : sm.v[0];
                        *reinterpret_cast<uint4*>(&dst[swizzle(row, 2 * group)]) = make_uint4(h[0], h[1], h[2], h[3]);
                        *reinterpret_cast<uint4*>(&dst[swizzle(row, 2 * group + 1)]) =
                            make_uint4(h[4], h[5], h[6], h[7]);
                    }
                }
                __syncthreads();
            } else if constexpr (kFp8) {
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

            float s[kPrefillKeys / 8][4] = {};
            if constexpr (kMma8) {
                const uint8_t* keys8 = sm.k8[stage];
#pragma unroll
                for (int ks = 0; ks < kHeadDim / 32; ++ks) {
#pragma unroll
                    for (int np = 0; np < kPrefillKeys / 16; ++np) {
                        uint32_t b[4];
                        const int key = np * 16 + (lane & 7) + 8 * (lane >> 4);
                        ldmatrix_x4(b, keys8 + k8_offset(key, 2 * ks + ((lane >> 3) & 1)));
                        mma_fp8(s[2 * np], qa8[ks], b[0], b[1], qs[ks]);
                        mma_fp8(s[2 * np + 1], qa8[ks], b[2], b[3], qs[ks]);
                    }
                }
            }
#pragma unroll
            for (int ks = 0; ks < (kMma8 ? 0 : kHeadDim / 16); ++ks) {
#pragma unroll
                for (int np = 0; np < kPrefillKeys / 16; ++np) {
                    uint32_t b[4];
                    ldmatrix_x4(b, keys + swizzle(np * 16 + (lane >> 4) * 8 + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
                    if constexpr (F::kFp4) {
                        mma_f16(s[2 * np], qa[ks], b[0], b[1]);
                        mma_f16(s[2 * np + 1], qa[ks], b[2], b[3]);
                    } else {
                        mma_bf16(s[2 * np], qa[ks], b[0], b[1]);
                        mma_bf16(s[2 * np + 1], qa[ks], b[2], b[3]);
                    }
                }
            }
            if (last_tile && has_next) {
                load_q(next_seg);
            }

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
                        s[n][e] *= row_score_scale[e / 2];
                    }
                }
            } else if (tile.full && tile.near) {
#pragma unroll
                for (int n = 0; n < kPrefillKeys / 8; ++n) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const float b = __bfloat162float(bias_base[e / 2][-(n * 8 + e % 2)]);
                        s[n][e] = fmaf(b, kLog2e, s[n][e] * row_score_scale[e / 2]);
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
                        float x = s[n][e] * row_score_scale[i];
                        if (tile.biased && distance < extent) {
                            x = fmaf(__bfloat162float(bias_base[i][-(n * 8 + e % 2)]), kLog2e, x);
                        }
                        s[n][e] = visible ? x : -INFINITY;
                    }
                }
            }

            if (more && next.biased) {
                __syncwarp();
                load_bias(next_owner, next);
            }

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
            const float offset[2] = {m[0] - kPLog2, m[1] - kPLog2};
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

            if constexpr (kMma8) {
                uint32_t pa8[kPrefillKeys / 32][4];
#pragma unroll
                for (int kk = 0; kk < kPrefillKeys / 32; ++kk) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int n = 4 * kk + 2 * (j / 2);
                        const int e = 2 * (j % 2);
                        pa8[kk][j] = pack_e4m3x4(s[n][e], s[n][e + 1], s[n + 1][e], s[n + 1][e + 1]);
                    }
                }
                pv8(o, pa8);
            } else {
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
            }
            stage ^= 1;
        }
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
            __syncthreads();
            unsigned* counter = &counters[seg.gblock * kKvHeads + kv_head];
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
                {
                    const float* first =
                        partials + size_t(2 * first_block + (seg.unit_begin > begin_of(first_block))) * kSlot;
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        m[i] = __ldcg(first + kRows * kHeadDim + local[i]);
                        l[i] = __ldcg(first + kRows * kHeadDim + kRows + local[i]);
                    }
                    const float4* first_o = reinterpret_cast<const float4*>(first) + warp * (kHeadDim / 8) * 32 + lane;
#pragma unroll
                    for (int n = 0; n < kHeadDim / 8; ++n) {
                        const float4 p = __ldcg(first_o + n * 32);
                        o[n][0] = p.x;
                        o[n][1] = p.y;
                        o[n][2] = p.z;
                        o[n][3] = p.w;
                    }
                }
                for (unsigned b = first_block + 1; b <= last_block; ++b) {
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

}  // namespace

// Ragged packed multi-slice KV append across all `m = gridDim.x` rows: each row appends
// into its own slice's KV cache (words 0 and 1 of `states`) at `(kv_start + local) % slots`,
// skipping overwritten rows when a slice's length exceeds `slots`.
extern "C" __global__ void __launch_bounds__(128)
kv_append_rows_packed_bf16(const __nv_bfloat16* k, const __nv_bfloat16* v,
                           const unsigned long long* states, const PackedSlice* slices,
                           int num_slices, unsigned slots) {
    const uint32_t row = blockIdx.x;
    const int si = find_packed_slice(slices, num_slices, row);
    const PackedSlice sl = slices[si];
    const uint32_t local = row - sl.row_start;
    if (local + slots < sl.row_len) {
        return;
    }
    __nv_bfloat16* k_cache = reinterpret_cast<__nv_bfloat16*>(states[size_t(sl.state_idx) * 6 + 0]);
    __nv_bfloat16* v_cache = reinterpret_cast<__nv_bfloat16*>(states[size_t(sl.state_idx) * 6 + 1]);
    const size_t i = threadIdx.x * 8;
    const size_t src = size_t(row) * kKvRow + i;
    const size_t dst = size_t((sl.kv_start + local) % slots) * kKvRow + i;
    *reinterpret_cast<uint4*>(k_cache + dst) = *reinterpret_cast<const uint4*>(k + src);
    *reinterpret_cast<uint4*>(v_cache + dst) = *reinterpret_cast<const uint4*>(v + src);
}

extern "C" __global__ void __launch_bounds__(128)
kv_append_rows_packed_fp8(const __nv_bfloat16* k, const __nv_bfloat16* v,
                          const unsigned long long* states, const PackedSlice* slices,
                          int num_slices, unsigned slots) {
    const uint32_t row = blockIdx.x;
    const int si = find_packed_slice(slices, num_slices, row);
    const PackedSlice sl = slices[si];
    const uint32_t local = row - sl.row_start;
    if (local + slots < sl.row_len) {
        return;
    }
    uint8_t* k_cache = reinterpret_cast<uint8_t*>(states[size_t(sl.state_idx) * 6 + 0]);
    uint8_t* v_cache = reinterpret_cast<uint8_t*>(states[size_t(sl.state_idx) * 6 + 1]);
    const size_t i = threadIdx.x * 8;
    const size_t src = size_t(row) * kKvRow + i;
    const size_t dst = size_t((sl.kv_start + local) % slots) * kKvRow + i;
    float x[8];
    unpack_bf16x8(*reinterpret_cast<const uint4*>(k + src), x);
    *reinterpret_cast<uint2*>(k_cache + dst) =
        make_uint2(pack_e4m3x4(x[0], x[1], x[2], x[3]), pack_e4m3x4(x[4], x[5], x[6], x[7]));
    unpack_bf16x8(*reinterpret_cast<const uint4*>(v + src), x);
    *reinterpret_cast<uint2*>(v_cache + dst) =
        make_uint2(pack_e4m3x4(x[0], x[1], x[2], x[3]), pack_e4m3x4(x[4], x[5], x[6], x[7]));
}

extern "C" __global__ void __launch_bounds__(64)
kv_append_rows_packed_fp4(const __nv_bfloat16* k, const __nv_bfloat16* v,
                          const unsigned long long* states, const PackedSlice* slices,
                          int num_slices, unsigned slots) {
    const uint32_t row = blockIdx.x;
    const int si = find_packed_slice(slices, num_slices, row);
    const PackedSlice sl = slices[si];
    const uint32_t local = row - sl.row_start;
    if (local + slots < sl.row_len) {
        return;
    }
    uint8_t* k_cache = reinterpret_cast<uint8_t*>(states[size_t(sl.state_idx) * 6 + 0]);
    uint8_t* v_cache = reinterpret_cast<uint8_t*>(states[size_t(sl.state_idx) * 6 + 1]);
    const int head = threadIdx.x / kFp4HeadScales;
    const int group = threadIdx.x % kFp4HeadScales;
    const size_t src = size_t(row) * kKvRow + threadIdx.x * 16;
    const unsigned slot = (sl.kv_start + local) % slots;
#pragma unroll
    for (int kv = 0; kv < 2; ++kv) {
        const __nv_bfloat16* x = kv == 0 ? k : v;
        float lo[8];
        float hi[8];
        unpack_bf16x8(*reinterpret_cast<const uint4*>(x + src), lo);
        unpack_bf16x8(*reinterpret_cast<const uint4*>(x + src + 8), hi);
        float vals[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            vals[i] = lo[i];
            vals[8 + i] = hi[i];
        }
        uint8_t scale;
        const uint2 data = fp4_group(vals, scale);
        uint8_t* cache = kv == 0 ? k_cache : v_cache;
        *reinterpret_cast<uint2*>(cache + fp4_data(slot, head) + group * 8) = data;
        cache[fp4_scales(slot, head) + group] = scale;
    }
}

extern "C" __global__ void __launch_bounds__(kPrefillWarps * 32)
prefill_attention_packed_fp4(const __nv_bfloat16* q, const __nv_bfloat16* k_in,
                             const __nv_bfloat16* v_in, const unsigned long long* states,
                             const PackedSlice* slices, const uint32_t* block_descs,
                             const __nv_bfloat16* bias, const __nv_bfloat16* sinks,
                             float* partials, unsigned* counters, __nv_bfloat16* out,
                             unsigned total_blocks_m, unsigned slots, unsigned window,
                             unsigned extent, unsigned input_keys, float scale) {
    prefill_attention_packed<Fp4Format, kPrefillWarps>(
        q, k_in, v_in, states, slices, block_descs, bias, sinks, partials, counters, out,
        total_blocks_m, slots, window, extent, input_keys, scale);
}

extern "C" __global__ void __launch_bounds__(kPrefillWarps * 32)
prefill_attention_packed_bf16(const __nv_bfloat16* q, const __nv_bfloat16* k_in,
                              const __nv_bfloat16* v_in, const unsigned long long* states,
                              const PackedSlice* slices, const uint32_t* block_descs,
                              const __nv_bfloat16* bias, const __nv_bfloat16* sinks,
                              float* partials, unsigned* counters, __nv_bfloat16* out,
                              unsigned total_blocks_m, unsigned slots, unsigned window,
                              unsigned extent, unsigned input_keys, float scale) {
    prefill_attention_packed<Bf16Format, kPrefillWarps>(
        q, k_in, v_in, states, slices, block_descs, bias, sinks, partials, counters, out,
        total_blocks_m, slots, window, extent, input_keys, scale);
}

extern "C" __global__ void __launch_bounds__(kPrefillWarps * 32)
prefill_attention_packed_fp8(const __nv_bfloat16* q, const __nv_bfloat16* k_in,
                             const __nv_bfloat16* v_in, const unsigned long long* states,
                             const PackedSlice* slices, const uint32_t* block_descs,
                             const __nv_bfloat16* bias, const __nv_bfloat16* sinks,
                             float* partials, unsigned* counters, __nv_bfloat16* out,
                             unsigned total_blocks_m, unsigned slots, unsigned window,
                             unsigned extent, unsigned input_keys, float scale) {
    prefill_attention_packed<Fp8Format, kPrefillWarps>(
        q, k_in, v_in, states, slices, block_descs, bias, sinks, partials, counters, out,
        total_blocks_m, slots, window, extent, input_keys, scale);
}

extern "C" __global__ void __launch_bounds__(kPrefillWarps * 32)
prefill_attention_packed_fp8_mma(const __nv_bfloat16* q, const __nv_bfloat16* k_in,
                                 const __nv_bfloat16* v_in, const unsigned long long* states,
                                 const PackedSlice* slices, const uint32_t* block_descs,
                                 const __nv_bfloat16* bias, const __nv_bfloat16* sinks,
                                 float* partials, unsigned* counters, __nv_bfloat16* out,
                                 unsigned total_blocks_m, unsigned slots, unsigned window,
                                 unsigned extent, unsigned input_keys, float scale) {
    prefill_attention_packed<Fp8MmaFormat, kPrefillWarps>(
        q, k_in, v_in, states, slices, block_descs, bias, sinks, partials, counters, out,
        total_blocks_m, slots, window, extent, input_keys, scale);
}

// The launcher passes PREFILL_PACKED_SHARED_BYTES of dynamic shared memory.
static_assert(sizeof(PrefillPackedShared<Bf16Format, kPrefillWarps>) <= 85248);
static_assert(sizeof(PrefillPackedShared<Fp8Format, kPrefillWarps>) <= 85248);
static_assert(sizeof(PrefillPackedShared<Fp8MmaFormat, kPrefillWarps>) <= 85248);
static_assert(sizeof(PrefillPackedShared<Fp4Format, kPrefillWarps>) <= 85248);
static_assert(kPrefillKeys * (kFp4HeadBytes + kFp4HeadScales) <=
              PrefillPackedShared<Fp4Format, kPrefillWarps>::kStaged);


// ---------------------------------------------------------------------------
// Attention of the 1 to 8 rows of a step, such as a speculative verify, at positions
// p0 .. p0 + m - 1 (p0 = *start), before their keys and values are appended. Row j
// attends causally over the keys at positions up to p0 + j within the window, as a
// decode step at that position would after appending, with the relative bias and sink of
// its own row.
//
// Each row's keys split by their distance d from the row. The keys at d >= 8 come from
// the cache as it is: the rows' new keys sit at d <= 7 from every row that sees them, so
// these keys are the same whichever rows the call holds. Two extra blocks per KV head (the
// recent blocks, rows 0 .. 3 and 4 .. 7) take d = 0 .. 7 and the sink, reading each key
// from the cache or, for the rows' own keys, from k_new and v_new rounded as the append
// rounds them. An earlier row still sees the old entry of the slot a later row's key goes
// to, at d >= slots - 7, when the ring has fewer than window + m - 1 slots.
//
// The other kRowsBlocks blocks of a KV head cut the cache slots into splits of 64 by
// absolute slot: block b takes splits b, b + kRowsBlocks, and so on, up to the last live
// slot. Warp w of a block takes keys 16 w .. 16 w + 15 of each of its splits, in a chunk,
// with its own copies in flight and its own running max, sums and outputs. Per chunk it
// computes S^T = K Q^T and O^T += V^T P^T with mma.sync.m16n8k16 and F32 accumulators.
// The query rows of the KV head (3 heads x m rows, qi = 3 j + h) are the N dimension, in
// ceil(3 m / 8) tiles of 8, so that the keys fill the M dimension of S and the head dims
// that of O. Numerics follow prefill attention: S in BF16 for a BF16 cache and in F16 for
// an E4M3 cache (E4M3 converts to F16 exactly, and a BF16 query is exact in F16 from
// 2^-14 up), the scale multiplies the F32 dot product, and P V runs in F16 with weights
// scaled by 2^15 and values converted to F16. The softmax sums add the F32 weights. The
// block merges its 4 warps in a fixed order and writes one partial output, max and sum
// per query row. The last blocks of the KV head to finish add the partials of the blocks
// in order, then the row's recent partial (rows_combine).
//
// Keys a row must not see score -inf and weigh exactly zero, so their products add
// exact zeros, and blocks without any of its keys leave max -FLT_MAX, sum 0 and output 0,
// which add nothing. Every element of S and of O^T depends only on its query row's column
// of Q and P and the chunk's keys, and each key's place in a chunk depends only on its
// slot. So a row's result depends only on its position, its query and the cache, never
// on m or the other rows: a call of one row at position p0 + j gives it bit for bit.
// ---------------------------------------------------------------------------

namespace {

constexpr int kRowsMax = 8;
// Query rows per KV head: 3 heads of up to 8 rows.
constexpr int kRowsQueries = kGroup * kRowsMax;
constexpr int kRowsWarps = 4;
constexpr int kRowsThreads = kRowsWarps * 32;
// Keys per warp chunk, the M dimension of one mma tile, and slots per split.
constexpr int kRowsChunk = 16;
constexpr int kRowsSplitSlots = kRowsWarps * kRowsChunk;
// Split blocks per KV head, and the partials per query row: theirs and the recent one.
constexpr int kRowsBlocks = 32;
constexpr int kRowsParts = kRowsBlocks + 1;
// Recent blocks per KV head, one row per warp.
constexpr int kRowsRecentBlocks = 2;
// Distances the recent blocks take. A row sees the new keys of rows before it at
// distances up to kRowsMax - 1.
constexpr int kRowsRecent = 8;
static_assert(kRowsRecent >= kRowsMax && kRowsRecent == 8);

// Chunks each warp keeps in flight: one for BF16 (each warp refills its key and value
// tiles as soon as it has read them), three for E4M3, within two blocks per SM.
template <class F>
struct RowsStages;
template <>
struct RowsStages<Bf16Format> {
    static constexpr int kValue = 1;
};
template <>
struct RowsStages<Fp8Format> {
    static constexpr int kValue = 3;
};
template <>
struct RowsStages<Fp4Format> {
    static constexpr int kValue = 3;
};

template <class F>
struct alignas(16) RowsShared {
    static constexpr int kStages = RowsStages<F>::kValue;
    // Each warp's key and value chunks, [16, 128], 16-byte chunk c of row r at c ^ (r & 7);
    // NVFP4: [16, 64] bytes, chunk c of row r at c ^ (r & 3), then [16, 8] scales.
    static constexpr int kTileElems =
        F::kFp4 ? kRowsChunk * (kFp4HeadBytes + kFp4HeadScales) : kRowsChunk * kHeadDim;
    typename F::Elem k[kRowsWarps][kStages][kTileElems];
    typename F::Elem v[kRowsWarps][kStages][kTileElems];
};
// Where a warp's NVFP4 tile keeps its scales.
constexpr int kRowsFp4Scales = kRowsChunk * kFp4HeadBytes;

// What a split block keeps in the chunks' place once its warps are done: two sums of
// scaled warp outputs, and each warp's max and sum per query row.
// Element (qi, dim) of a sum sits at column rows_merge_column(dim) of row qi: the padding
// and the XOR keep the lanes of a warp, which hold 8 dims 16 apart, on distinct banks.
constexpr int kRowsMergeStride = kHeadDim + 4;
__device__ __forceinline__ int rows_merge_column(int dim) {
    return dim ^ ((dim >> 4) & 7);
}
struct RowsMerge {
    float o[2][kRowsQueries][kRowsMergeStride];
    float m[kRowsWarps][kRowsQueries];
    float l[kRowsWarps][kRowsQueries];
};
static_assert(sizeof(RowsMerge) <= sizeof(RowsShared<Bf16Format>));

// What a combining block keeps in shared memory: its arrival order among the KV head's
// blocks, and the rescale factor of every partial and the combined sum of up to kRowsWarps
// query rows at once.
constexpr int kRowsCombineQueries = 1;
struct RowsCombine {
    float factor[kRowsWarps][kRowsParts];
    float total[kRowsWarps];
    unsigned arrival;
};

// Most blocks of a KV head that share its combine, one per query row: a single block reads
// the partials of 24 query rows at one SM's L2 bandwidth, and 8 blocks of 3 rows each took
// 0.7 us longer than 24 at 8 rows.
constexpr int kRowsCombiners = 24;
static_assert(kRowsCombiners * kRowsCombineQueries == kRowsQueries);

// A BF16 value as the cache stores it and attention reads it back.
template <class F>
__device__ __forceinline__ float rows_cache_value(float x);

template <>
__device__ __forceinline__ float rows_cache_value<Bf16Format>(float x) {
    return x;
}

// The verify pass rounds the rows to NVFP4 values before this kernel (kv_round_bf16),
// which the append keeps exactly.
template <>
__device__ __forceinline__ float rows_cache_value<Fp4Format>(float x) {
    return x;
}

template <>
__device__ __forceinline__ float rows_cache_value<Fp8Format>(float x) {
    float v[4];
    unpack_e4m3x4(pack_e4m3x4(x, 0.f, 0.f, 0.f), v);
    return v[0];
}

template <int N>
__device__ __forceinline__ void copy_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// The 8x8 B16 matrix of a fragment, transposed.
__device__ __forceinline__ uint32_t movmatrix_trans(uint32_t x) {
    uint32_t r;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(r) : "r"(x));
    return r;
}

// Byte offset of 16-byte chunk `chunk` of row `row` in a warp's [16, 128] chunk tile.
template <class F>
__device__ __forceinline__ int rows_offset(int row, int chunk) {
    if constexpr (F::kFp4) {
        return row * kFp4HeadBytes + ((chunk ^ (row & 3)) << 4);
    }
    constexpr int kRowBytes = kHeadDim * int(sizeof(typename F::Elem));
    return row * kRowBytes + ((chunk ^ (row & 7)) << 4);
}

// Starts copying the 16 keys or values of KV head `kv_head` from slot `first` into a
// warp's tile; slots at or past `end` read as zeros. Every lane of the warp calls it.
template <class F>
__device__ __forceinline__ void rows_copy_chunk(typename F::Elem* dst, const typename F::Elem* cache, int kv_head,
                                                unsigned first, unsigned end, int lane) {
    const uint64_t policy = l2_evict_first();
    if constexpr (F::kFp4) {
        // 4 data chunks per row, two per lane, and lanes 0 .. 15 copy row lane's scales.
        constexpr int kChunks = kFp4HeadBytes / 16;
#pragma unroll
        for (int k = 0; k < kRowsChunk * kChunks / 32; ++k) {
            const int i = lane + 32 * k;
            const int row = i / kChunks;
            const int chunk = i % kChunks;
            const unsigned slot = first + row;
            const bool valid = slot < end;
            copy_async16_evict(reinterpret_cast<char*>(dst) + rows_offset<F>(row, chunk),
                               cache + fp4_data(valid ? slot : 0, kv_head) + chunk * 16, valid, policy);
        }
        if (lane < kRowsChunk) {
            const unsigned slot = first + lane;
            const bool valid = slot < end;
            copy_async8_evict(reinterpret_cast<char*>(dst) + kRowsFp4Scales + lane * kFp4HeadScales,
                              cache + fp4_scales(valid ? slot : 0, kv_head), valid, policy);
        }
        return;
    }
    constexpr int kChunks = kHeadDim * int(sizeof(typename F::Elem)) / 16;
    constexpr int kElems = 16 / int(sizeof(typename F::Elem));
#pragma unroll
    for (int k = 0; k < kRowsChunk * kChunks / 32; ++k) {
        const int i = lane + 32 * k;
        const int row = i / kChunks;
        const int chunk = i % kChunks;
        const unsigned slot = first + row;
        const bool valid = slot < end;
        const size_t offset = size_t(valid ? slot : 0) * kKvRow + kv_head * kHeadDim + chunk * kElems;
        copy_async16_evict(reinterpret_cast<char*>(dst) + rows_offset<F>(row, chunk), cache + offset, valid, policy);
    }
}

// Distance from slot `slot` to a row's newest slot in a cache of `slots` slots.
__device__ __forceinline__ unsigned rows_distance(unsigned newest, unsigned slot, unsigned slots) {
    return newest >= slot ? newest - slot : newest + slots - slot;
}

// Four cache elements as loaded, and as F32.
template <class F>
struct RowsRaw4;
template <>
struct RowsRaw4<Bf16Format> {
    using Type = uint2;
    __device__ static void unpack(uint2 raw, float (&v)[4]) { unpack_bf16x4(raw, v); }
};
template <>
struct RowsRaw4<Fp8Format> {
    using Type = uint32_t;
    __device__ static void unpack(uint32_t raw, float (&v)[4]) { unpack_e4m3x4(raw, v); }
};
template <>
struct RowsRaw4<Fp4Format> {
    using Type = uint32_t;
    // Two bytes of E2M1 pairs, then the scale.
    __device__ static void unpack(uint32_t raw, float (&v)[4]) {
        uint32_t h[4];
        e2m1x8_to_f16x8(raw & 0xffffu, h);
        const float sc = e4m3_value(uint8_t(raw >> 16));
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&h[0]));
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&h[1]));
        v[0] = a.x * sc;
        v[1] = a.y * sc;
        v[2] = b.x * sc;
        v[3] = b.y * sc;
    }
};

// A recent block: each row's keys at distances 0 .. 7 and its sink. Warp w of recent block
// r takes row 4 r + w. Lane l scores key d = l % 8 over dims 32 (l / 8) .. 32 (l / 8) + 31
// for the 3 heads, adds the 4 parts by a butterfly, and adds the weighted values of dims
// 4 l .. 4 l + 3 over d = 0 .. 7 in order. Writes partial kRowsBlocks of each query row.
// Every load goes out first, from both places a key or value may come from (row 0 or slot 0
// where one does not apply), so that the block waits for memory once.
template <class F, bool kDirectCombine = false, bool kTree = false>
__device__ __forceinline__ void rows_recent(const __nv_bfloat16* q, const __nv_bfloat16* k_new,
                                            const __nv_bfloat16* v_new, const typename F::Elem* k_cache,
                                            const typename F::Elem* v_cache, const __nv_bfloat16* bias,
                                            const __nv_bfloat16* sinks, float* partial_out, float* partial_max,
                                            float* partial_sum, unsigned start, int rows, unsigned slots,
                                            unsigned window, unsigned extent, float scale, int recent_block,
                                            const RowsMerge* mg = nullptr, __nv_bfloat16* out = nullptr,
                                            const __nv_bfloat16* gate = nullptr,
                                            const uint32_t* tree_depth = nullptr,
                                            const int* tree_attn_anc = nullptr) {
    using Raw4 = typename RowsRaw4<F>::Type;
    constexpr int kElems = F::kChunkElems;
    const int kv_head = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int head0 = kv_head * kGroup;
    const int d = lane % kRowsRecent;
    const int part = lane / kRowsRecent;
    static_assert(kRowsRecentBlocks * kRowsWarps == kRowsMax);
    const int j = recent_block * kRowsWarps + warp;
    if (j >= rows) {
        return;
    }
    const int row_depth = kTree ? int(tree_depth[j]) : j;
    const unsigned position = start + unsigned(row_depth);
    const unsigned newest = position % slots;
    // Whether key `dist` exists, whether it is a row's new key, and its slot otherwise.
    const auto exists = [&](int dist) { return unsigned(dist) <= position && (window == 0 || unsigned(dist) < window); };
    const auto old_slot = [&](int dist) {
        return exists(dist) && dist > row_depth ? (newest >= unsigned(dist) ? newest - dist : newest + slots - dist) : 0u;
    };
    const auto anc_row = [&](int dist) {
        if constexpr (kTree) {
            return tree_attn_anc[j * kRowsRecent + dist];
        } else {
            return dist <= j ? j - dist : -1;
        }
    };

    uint4 key_new[4];
    uint4 key_old[32 / kElems];
    uint16_t key_old_scales = 0;
    uint4 q_raw[kGroup][4];
    unsigned short bias_raw[kGroup];
    unsigned short sink_raw[kGroup];
    uint2 value_new[kRowsRecent];
    Raw4 value_old[kRowsRecent];
    {
        const int anc_d = anc_row(d);
        const __nv_bfloat16* src = k_new + (size_t(anc_d >= 0 ? anc_d : 0) * kKvHeads + kv_head) * kHeadDim + 32 * part;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            key_new[c] = *reinterpret_cast<const uint4*>(src + 8 * c);
        }
        if constexpr (F::kFp4) {
            key_old[0] = *reinterpret_cast<const uint4*>(k_cache + fp4_data(old_slot(d), kv_head) + 16 * part);
            key_old_scales = *reinterpret_cast<const uint16_t*>(k_cache + fp4_scales(old_slot(d), kv_head) + 2 * part);
        } else {
            const typename F::Elem* old = k_cache + size_t(old_slot(d)) * kKvRow + kv_head * kHeadDim + 32 * part;
#pragma unroll
            for (int c = 0; c < 32 / kElems; ++c) {
                key_old[c] = *reinterpret_cast<const uint4*>(old + kElems * c);
            }
        }
    }
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        const size_t query = size_t(j) * kHeads + head0 + h;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            q_raw[h][c] = *reinterpret_cast<const uint4*>(q + query * kHeadDim + 32 * part + 8 * c);
        }
        bias_raw[h] = __bfloat16_as_ushort(bias[query * extent + min(unsigned(d), extent - 1)]);
        sink_raw[h] = __bfloat16_as_ushort(sinks[query]);
    }
#pragma unroll
    for (int key_d = 0; key_d < kRowsRecent; ++key_d) {
        const int anc_kd = anc_row(key_d);
        value_new[key_d] = *reinterpret_cast<const uint2*>(
            v_new + (size_t(anc_kd >= 0 ? anc_kd : 0) * kKvHeads + kv_head) * kHeadDim + 4 * lane);
        if constexpr (F::kFp4) {
            // Two data bytes in the low half, the group's scale above them.
            value_old[key_d] =
                Raw4(*reinterpret_cast<const uint16_t*>(v_cache + fp4_data(old_slot(key_d), kv_head) + 2 * lane)) |
                Raw4(v_cache[fp4_scales(old_slot(key_d), kv_head) + lane / 4]) << 16;
        } else {
            value_old[key_d] = *reinterpret_cast<const Raw4*>(v_cache + size_t(old_slot(key_d)) * kKvRow +
                                                              kv_head * kHeadDim + 4 * lane);
        }
    }

    // Key d of this row, dims 32 part .. 32 part + 31, as attention reads it.
    float key[32];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float v[8];
        unpack_bf16x8(key_new[c], v);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            key[8 * c + e] = rows_cache_value<F>(v[e]);
        }
    }
    if (d > row_depth) {
        if constexpr (F::kFp4) {
            unpack_fp4x32(key_old[0], uint8_t(key_old_scales & 0xffu), uint8_t(key_old_scales >> 8), key);
        } else {
#pragma unroll
            for (int c = 0; c < 32 / kElems; ++c) {
                float v[kElems];
                F::unpack_chunk(key_old[c], v);
#pragma unroll
                for (int e = 0; e < kElems; ++e) {
                    key[kElems * c + e] = v[e];
                }
            }
        }
    }
    float weight[kGroup];
    float row_max[kGroup];
    float row_sum[kGroup];
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        float dot = 0.f;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            float v[8];
            unpack_bf16x8(q_raw[h][c], v);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                dot = fmaf(v[e], key[8 * c + e], dot);
            }
        }
        dot += __shfl_xor_sync(0xffffffffu, dot, 8);
        dot += __shfl_xor_sync(0xffffffffu, dot, 16);
        float score = -INFINITY;
        if (exists(d)) {
            score = dot * scale;
            if (unsigned(d) < extent) {
                score += __uint_as_float(unsigned(bias_raw[h]) << 16);
            }
        }
        float mx = score;
#pragma unroll
        for (int offset = 1; offset < kRowsRecent; offset *= 2) {
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, offset));
        }
        const float sink = __uint_as_float(unsigned(sink_raw[h]) << 16);
        row_max[h] = fmaxf(mx, sink);
        weight[h] = exp2_ftz(fmaf(score - row_max[h], kLog2e, kPScaleLog2));
        float total = weight[h];
#pragma unroll
        for (int offset = 1; offset < kRowsRecent; offset *= 2) {
            total += __shfl_xor_sync(0xffffffffu, total, offset);
        }
        row_sum[h] = total + exp2_ftz(fmaf(sink - row_max[h], kLog2e, kPScaleLog2));
    }
    // Weighted values of dims 4 lane .. 4 lane + 3. A key that does not exist weighs 0.
    float o[kGroup][4] = {};
#pragma unroll
    for (int key_d = 0; key_d < kRowsRecent; ++key_d) {
        float v[4];
        if (key_d <= row_depth) {
            unpack_bf16x4(value_new[key_d], v);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                v[e] = rows_cache_value<F>(v[e]);
            }
        } else {
            RowsRaw4<F>::unpack(value_old[key_d], v);
        }
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const float w = __shfl_sync(0xffffffffu, weight[h], key_d);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                o[h][e] = fmaf(w, v[e], o[h][e]);
            }
        }
    }
    if constexpr (kDirectCombine) {
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const int qi = j * kGroup + h;
            const float recent_m = __shfl_sync(0xffffffffu, row_max[h], 0);
            const float recent_l = __shfl_sync(0xffffffffu, row_sum[h], 0);
            const size_t query = size_t(j) * kHeads + head0 + h;
            const size_t at_base = query * kHeadDim + 4 * lane;
            if (mg != nullptr) {
                const float split_m = mg->m[0][qi];
                const float split_l = mg->l[0][qi];
                const float largest = fmaxf(split_m, recent_m);
                const float split_factor = exp2_ftz((split_m - largest) * kLog2e);
                const float recent_factor = exp2_ftz((recent_m - largest) * kLog2e);
                const float total = fmaf(recent_factor, recent_l, fmaf(split_factor, split_l, 0.f));
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float split_v = mg->o[0][qi][rows_merge_column(4 * lane + e)];
                    const float value = fmaf(recent_factor, o[h][e], fmaf(split_factor, split_v, 0.f));
                    const __nv_bfloat16 attn = __float2bfloat16(value / total);
                    const size_t at = at_base + e;
                    out[at] = gate != nullptr
                                  ? to_bf16_ftz(fmul(__bfloat162float(attn), mlx_sigmoid(__bfloat162float(gate[at]))))
                                  : attn;
                }
            } else {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const __nv_bfloat16 attn = __float2bfloat16(o[h][e] / recent_l);
                    const size_t at = at_base + e;
                    out[at] = gate != nullptr
                                  ? to_bf16_ftz(fmul(__bfloat162float(attn), mlx_sigmoid(__bfloat162float(gate[at]))))
                                  : attn;
                }
            }
        }
    } else {
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const size_t query = size_t(j) * kHeads + head0 + h;
            *reinterpret_cast<float4*>(partial_out + (query * kRowsParts + kRowsBlocks) * kHeadDim + 4 * lane) =
                make_float4(o[h][0], o[h][1], o[h][2], o[h][3]);
            if (lane == 0) {
                partial_max[query * kRowsParts + kRowsBlocks] = row_max[h];
                partial_sum[query * kRowsParts + kRowsBlocks] = row_sum[h];
            }
        }
    }
}

// An acquire load at GPU scope.
__device__ __forceinline__ unsigned load_acquire(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// The combine of the partials of the KV head blockIdx.x, for its query row `combiner`. Each query row's partials (split blocks 0 ..
// used - 1, then the recent partial) are scaled by e^(their max - the largest max) and
// summed in that order, then the output is divided by the sum and rounded to BF16. With a
// non-null `gate`, the output is then gated as attention_gate_bf16 in elementwise.cu does.
// Then the key and value of the KV head's rows go to slot (*start + row) mod slots, as
// kv_append_rows does: every block of the KV head has finished reading the caches, and the
// other KV heads read other channels.
//
// Split blocks `used` and later hold no keys: their partials are max -FLT_MAX, sum 0 and
// output 0, which add exactly nothing, so they are not read.
template <class F, bool kSingleCombiner = false, bool kTree = false>
__device__ __forceinline__ void rows_combine(RowsCombine& sm, const float* partial_out, const float* partial_max,
                                             const float* partial_sum, __nv_bfloat16* out,
                                             const __nv_bfloat16* gate, const __nv_bfloat16* k_new,
                                             const __nv_bfloat16* v_new, typename F::Elem* k_cache,
                                             typename F::Elem* v_cache, unsigned start, int rows,
                                             unsigned slots, int used, int combiner, int combiners,
                                             typename F::Elem* k_stage = nullptr,
                                             typename F::Elem* v_stage = nullptr) {
    static_assert(kRowsBlocks == 32 && kRowsThreads == kHeadDim && kRowsWarps >= kRowsCombineQueries);
    const int kv_head = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int queries = kGroup * rows;
    const int d = threadIdx.x;
    const auto compute_factors = [&](int slot, size_t query) {
        const float m = lane < used ? __ldcg(&partial_max[query * kRowsParts + lane]) : -FLT_MAX;
        const float s = lane < used ? __ldcg(&partial_sum[query * kRowsParts + lane]) : 0.f;
        const float recent_max = __ldcg(&partial_max[query * kRowsParts + kRowsBlocks]);
        const float recent_sum = __ldcg(&partial_sum[query * kRowsParts + kRowsBlocks]);
        const float largest = fmaxf(warp_max(m), recent_max);
        const float factor = exp2_ftz((m - largest) * kLog2e);
        const float recent_factor = exp2_ftz((recent_max - largest) * kLog2e);
        sm.factor[slot][lane] = factor;
        float total = 0.f;
#pragma unroll
        for (int b = 0; b < kRowsBlocks; ++b) {
            total = fmaf(__shfl_sync(0xffffffffu, factor, b), __shfl_sync(0xffffffffu, s, b), total);
        }
        total = fmaf(recent_factor, recent_sum, total);
        if (lane == 0) {
            sm.factor[slot][kRowsBlocks] = recent_factor;
            sm.total[slot] = total;
        }
    };
    const auto emit_query = [&](int slot, int qi, size_t query) {
        float value = 0.f;
        for (int b = 0; b < used; ++b) {
            value = fmaf(sm.factor[slot][b], __ldcg(&partial_out[(query * kRowsParts + b) * kHeadDim + d]), value);
        }
        value = fmaf(sm.factor[slot][kRowsBlocks],
                     __ldcg(&partial_out[(query * kRowsParts + kRowsBlocks) * kHeadDim + d]), value);
        const float total = sm.total[slot];
        const __nv_bfloat16 attn = __float2bfloat16(value / total);
        const size_t at = query * kHeadDim + d;
        out[at] = gate != nullptr
                      ? to_bf16_ftz(fmul(__bfloat162float(attn), mlx_sigmoid(__bfloat162float(gate[at]))))
                      : attn;
        typename F::Elem* dst_k = kTree ? k_stage : k_cache;
        typename F::Elem* dst_v = kTree ? v_stage : v_cache;
        if (dst_k != nullptr && qi % kGroup == 0) {
            const size_t row = qi / kGroup;
            const size_t channel = size_t(kv_head) * kHeadDim + d;
            const unsigned slot_idx = kTree ? unsigned(row) : (start + unsigned(row)) % slots;
            if constexpr (F::kFp4) {
                // Thread 16 i rounds group i of the head, as kv_append_rows_fp4 does.
                if (d % 16 == 0) {
#pragma unroll
                    for (int kv = 0; kv < 2; ++kv) {
                        const __nv_bfloat16* x = (kv == 0 ? k_new : v_new) + row * kKvRow + channel;
                        float vals[16];
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            vals[i] = __bfloat162float(x[i]);
                        }
                        uint8_t sc;
                        const uint2 data = fp4_group(vals, sc);
                        typename F::Elem* cache = kv == 0 ? dst_k : dst_v;
                        *reinterpret_cast<uint2*>(cache + fp4_data(slot_idx, kv_head) + d / 2) = data;
                        cache[fp4_scales(slot_idx, kv_head) + d / 16] = sc;
                    }
                }
            } else {
                const size_t dst = size_t(slot_idx) * kKvRow + channel;
                F::store1(dst_k + dst, k_new[row * kKvRow + channel]);
                F::store1(dst_v + dst, v_new[row * kKvRow + channel]);
            }
        }
    };
    if constexpr (kSingleCombiner) {
        // Single-block combine for batched multi-row verify: 4 query rows per iteration
        // (one warp per query row for factor reduction), cutting barriers 4x with bit-exact
        // summation order.
        for (int qi0 = 0; qi0 < queries; qi0 += kRowsWarps) {
            const int qi_w = qi0 + warp;
            if (qi_w < queries) {
                const size_t query = size_t(qi_w / kGroup) * kHeads + kv_head * kGroup + qi_w % kGroup;
                compute_factors(warp, query);
            }
            __syncthreads();
            const int batch_q = min(kRowsWarps, queries - qi0);
            for (int j = 0; j < batch_q; ++j) {
                const int qi = qi0 + j;
                const size_t query = size_t(qi / kGroup) * kHeads + kv_head * kGroup + qi % kGroup;
                emit_query(j, qi, query);
            }
            if (qi0 + kRowsWarps < queries) {
                __syncthreads();
            }
        }
        return;
    }
    // This block's query rows j are qi = combiner + j * combiners.
    const auto query_of = [&](int j) {
        const int qi = combiner + j * combiners;
        return size_t(qi / kGroup) * kHeads + kv_head * kGroup + qi % kGroup;
    };
    const auto owns = [&](int j) { return combiner + j * combiners < queries; };
    if (warp < kRowsCombineQueries && owns(warp)) {
        compute_factors(warp, query_of(warp));
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < kRowsCombineQueries; ++j) {
        if (!owns(j)) {
            continue;
        }
        emit_query(j, combiner + j * combiners, query_of(j));
    }
}

// A split block: see the section comment. NT is the number of 8-column tiles of query
// rows, ceil(3 m / 8). Strides splits by `split_stride` (1..=kRowsBlocks). With
// kWritePartials = true, writes the block's partials of every query row; with
// kWritePartials = false (single-block no-split fast path), leaves the merged output, max
// and sum in `RowsMerge::o[0]`, `m[0]` and `l[0]` in shared memory.
template <class F, int NT, bool kWritePartials = true, bool kTree = false>
__device__ __forceinline__ void rows_split(const __nv_bfloat16* q, const typename F::Elem* k_cache,
                                           const typename F::Elem* v_cache, const __nv_bfloat16* bias,
                                           float* partial_out, float* partial_max, float* partial_sum,
                                           unsigned start, int rows, unsigned slots, unsigned window,
                                           unsigned extent, float scale, unsigned live, int block, int split_stride,
                                           char* smem, const uint32_t* tree_depth = nullptr) {
    constexpr bool kFp8 = sizeof(typename F::Elem) == 1;
    constexpr int kStages = RowsStages<F>::kValue;
    auto& sm = *reinterpret_cast<RowsShared<F>*>(smem);
    const int kv_head = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int head0 = kv_head * kGroup;
    const int queries = kGroup * rows;

    // Chunk k of this warp starts at slot chunk_first(k); it has chunks below `live`.
    const auto chunk_first = [&](int k) {
        return unsigned(block + k * split_stride) * kRowsSplitSlots + unsigned(warp * kRowsChunk);
    };
    int chunks = 0;
    while (chunk_first(chunks) < live) {
        ++chunks;
    }
    // The host uploads the start before the pass, and no kernel of the pass writes this
    // visit's cache before its attention: the rows append after it. So the first chunk
    // copies before pdl_wait, while the previous kernel still runs. Copying all three E4M3
    // chunks early slowed the projections before the attention by as much as it saved.
    constexpr int kEarly = 1;
#pragma unroll
    for (int k = 0; k < kStages; ++k) {
        if (k == kEarly) {
            pdl_wait();
        }
        if (k < chunks) {
            rows_copy_chunk<F>(sm.k[warp][k], k_cache, kv_head, chunk_first(k), live, lane);
        }
        copy_async_commit();
        if (k < chunks) {
            rows_copy_chunk<F>(sm.v[warp][k], v_cache, kv_head, chunk_first(k), live, lane);
        }
        copy_async_commit();
    }
    if constexpr (kEarly == kStages) {
        pdl_wait();
    }

    // B fragments of the queries, for n-tile j's column g (query row 8 j + g) and k-step
    // ks. E4M3 tiles: lane (g, t) holds dims 32 t .. 32 t + 31 in F16, k-step ks taking dims
    // 32 t + 4 ks .. + 3, so that each lane reads 32 contiguous bytes of a key. BF16 tiles:
    // the standard fragment, dims 16 ks + 2 t, + 1, + 8, + 9.
    uint32_t qf[NT][8][2];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int qi = 8 * j + g;
        const __nv_bfloat16* qp = q + (size_t(qi / kGroup) * kHeads + head0 + qi % kGroup) * kHeadDim;
        if constexpr (kFp8) {
            uint4 raw[4] = {};
            if (qi < queries) {
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    raw[c] = *reinterpret_cast<const uint4*>(qp + 32 * t + 8 * c);
                }
            }
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint32_t w[4] = {raw[c].x, raw[c].y, raw[c].z, raw[c].w};
                qf[j][2 * c][0] = bf16x2_to_f16x2(w[0]);
                qf[j][2 * c][1] = bf16x2_to_f16x2(w[1]);
                qf[j][2 * c + 1][0] = bf16x2_to_f16x2(w[2]);
                qf[j][2 * c + 1][1] = bf16x2_to_f16x2(w[3]);
            }
        } else {
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                qf[j][ks][0] = qi < queries ? *reinterpret_cast<const uint32_t*>(qp + 16 * ks + 2 * t) : 0u;
                qf[j][ks][1] = qi < queries ? *reinterpret_cast<const uint32_t*>(qp + 16 * ks + 2 * t + 8) : 0u;
            }
        }
    }

    // Each lane's two columns per n-tile, 8 j + 2 t + e: query row, newest slot, live
    // slots and bias row.
    unsigned newest[NT][2];
    unsigned live_slots[NT][2];
    unsigned bias_row[NT][2];
    bool valid[NT][2];
    const unsigned first_newest = start % slots;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int qi = 8 * j + 2 * t + e;
            const unsigned row = unsigned(qi / kGroup);
            const unsigned row_depth = (kTree && qi < queries) ? tree_depth[row] : row;
            valid[j][e] = qi < queries;
            newest[j][e] =
                first_newest + row_depth < slots ? first_newest + row_depth : first_newest + row_depth - slots;
            live_slots[j][e] = min(start + row_depth + 1, slots);
            bias_row[j][e] = unsigned((qi / kGroup) * kHeads + head0 + qi % kGroup) * extent;
        }
    }

    // O^T fragments: o[j][i] holds dims dim_of(i, g) and dim_of(i, g + 8) of columns 2 t and
    // 2 t + 1 of n-tile j.
    float o[NT][8][4];
    float m[NT][2];
    float l[NT][2];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            m[j][e] = -FLT_MAX;
            l[j][e] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                o[j][i][r] = 0.f;
            }
        }
    }

    for (int k = 0; k < chunks; ++k) {
        const int stage = k % kStages;
        const unsigned first = chunk_first(k);
        // Chunks that every row sees in full, beyond the bias extent, skip the mask and bias.
        const bool plain = window == 0 && first + unsigned(kRowsChunk - 1) + max(extent, unsigned(kRowsRecent)) <= start;
        // Bias of element (key g + 8 (r / 2), column 2 t + r % 2), loaded before the scores.
        unsigned short bias_raw[NT][4];
        unsigned distance[NT][4];
        if (!plain) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int e = r % 2;
                    const unsigned slot = first + unsigned(g + 8 * (r / 2));
                    distance[j][r] = rows_distance(newest[j][e], slot, slots);
                    bias_raw[j][r] = 0;
                    if (valid[j][e] && distance[j][r] < extent) {
                        bias_raw[j][r] = __bfloat16_as_ushort(bias[bias_row[j][e] + distance[j][r]]);
                    }
                }
            }
        }

        copy_async_wait<2 * kStages - 1>();
        __syncwarp();
        float s[NT][4];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                s[j][r] = 0.f;
            }
        }
        if constexpr (F::kFp4) {
            // Rows g and g + 8, dims 32 t .. 32 t + 31: chunk t and scales 2 t, 2 t + 1. K-step
            // ks takes dims 32 t + 4 ks .. + 3, as in the E4M3 layout: bytes 2 ks and 2 ks + 1.
            const char* tile = reinterpret_cast<const char*>(sm.k[warp][stage]);
            const uint4 ra = *reinterpret_cast<const uint4*>(tile + rows_offset<F>(g, t));
            const uint4 rb = *reinterpret_cast<const uint4*>(tile + rows_offset<F>(g + 8, t));
            const uint8_t* sc = reinterpret_cast<const uint8_t*>(tile) + kRowsFp4Scales;
            const __half2 sa[2] = {e4m3_half2(sc[g * kFp4HeadScales + 2 * t]),
                                   e4m3_half2(sc[g * kFp4HeadScales + 2 * t + 1])};
            const __half2 sb[2] = {e4m3_half2(sc[(g + 8) * kFp4HeadScales + 2 * t]),
                                   e4m3_half2(sc[(g + 8) * kFp4HeadScales + 2 * t + 1])};
            const uint32_t wa[4] = {ra.x, ra.y, ra.z, ra.w};
            const uint32_t wb[4] = {rb.x, rb.y, rb.z, rb.w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                uint32_t ha[4];
                uint32_t hb[4];
                e2m1x8_to_f16x8(wa[w], ha);
                e2m1x8_to_f16x8(wb[w], hb);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int ks = 2 * w + half;
                    const uint32_t a[4] = {hmul2_bits(ha[2 * half], sa[w / 2]), hmul2_bits(hb[2 * half], sb[w / 2]),
                                           hmul2_bits(ha[2 * half + 1], sa[w / 2]),
                                           hmul2_bits(hb[2 * half + 1], sb[w / 2])};
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        mma_f16(s[j], a, qf[j][ks][0], qf[j][ks][1]);
                    }
                }
            }
        } else if constexpr (kFp8) {
            const char* tile = reinterpret_cast<const char*>(sm.k[warp][stage]);
            const uint4 a0 = *reinterpret_cast<const uint4*>(tile + rows_offset<F>(g, 2 * t));
            const uint4 a1 = *reinterpret_cast<const uint4*>(tile + rows_offset<F>(g, 2 * t + 1));
            const uint4 b0 = *reinterpret_cast<const uint4*>(tile + rows_offset<F>(g + 8, 2 * t));
            const uint4 b1 = *reinterpret_cast<const uint4*>(tile + rows_offset<F>(g + 8, 2 * t + 1));
            const uint32_t lo[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const uint32_t hi[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                const uint32_t a[4] = {e4m3x2_to_f16x2(lo[ks] & 0xffffu), e4m3x2_to_f16x2(hi[ks] & 0xffffu),
                                       e4m3x2_to_f16x2(lo[ks] >> 16), e4m3x2_to_f16x2(hi[ks] >> 16)};
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    mma_f16(s[j], a, qf[j][ks][0], qf[j][ks][1]);
                }
            }
        } else {
            const char* tile = reinterpret_cast<const char*>(sm.k[warp][stage]);
            const int row = lane % 8 + 8 * ((lane / 8) % 2);
#pragma unroll
            for (int ks = 0; ks < 8; ++ks) {
                uint32_t a[4];
                ldmatrix_x4(a, tile + rows_offset<F>(row, 2 * ks + lane / 16));
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    mma_bf16(s[j], a, qf[j][ks][0], qf[j][ks][1]);
                }
            }
        }
        __syncwarp();
        if (k + kStages < chunks) {
            rows_copy_chunk<F>(sm.k[warp][stage], k_cache, kv_head, chunk_first(k + kStages), live, lane);
        }
        copy_async_commit();

        // Online softmax per column over the chunk's 16 keys: lanes g = 0 .. 7 hold them.
        float p[NT][4];
        float factor[NT][2];
        bool rescale = false;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float score[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int r = 2 * h + e;
                    score[h] = s[j][r] * scale;
                    if (!plain) {
                        const unsigned slot = first + unsigned(g + 8 * h);
                        const unsigned dist = distance[j][r];
                        const bool sees = valid[j][e] && slot < live_slots[j][e] && dist >= unsigned(kRowsRecent) &&
                                          (window == 0 || dist < window);
                        if (!sees) {
                            score[h] = -INFINITY;
                        } else if (dist < extent) {
                            score[h] += __uint_as_float(unsigned(bias_raw[j][r]) << 16);
                        }
                    }
                }
                float most = fmaxf(score[0], score[1]);
                most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, 4));
                most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, 8));
                most = fmaxf(most, __shfl_xor_sync(0xffffffffu, most, 16));
                const float new_m = fmaxf(m[j][e], most);
                factor[j][e] = exp2_ftz((m[j][e] - new_m) * kLog2e);
                rescale |= factor[j][e] != 1.f;
                p[j][e] = exp2_ftz(fmaf(score[0] - new_m, kLog2e, kPScaleLog2));
                p[j][2 + e] = exp2_ftz(fmaf(score[1] - new_m, kLog2e, kPScaleLog2));
                l[j][e] = fmaf(l[j][e], factor[j][e], p[j][e] + p[j][2 + e]);
                m[j][e] = new_m;
            }
        }
        // Rescaling by exactly 1 changes nothing, so only chunks that raise a max rescale.
        if (__any_sync(0xffffffffu, rescale)) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        o[j][i][r] *= factor[j][r % 2];
                    }
                }
            }
        }
        // P^T as B fragments: the C fragment's 8x8 blocks of keys g and g + 8, transposed.
        uint32_t pb[NT][2];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            pb[j][0] = movmatrix_trans(pack_f16x2(p[j][0], p[j][1]));
            pb[j][1] = movmatrix_trans(pack_f16x2(p[j][2], p[j][3]));
        }

        copy_async_wait<2 * kStages - 1>();
        __syncwarp();
        if constexpr (F::kFp4) {
            // As the E4M3 layout: m-tile i holds dims 16 g + i and 16 g + 8 + i, here group g of
            // keys 2 t, 2 t + 1, 2 t + 8 and 2 t + 9, 8 bytes in half g % 2 of chunk g / 2.
            const char* tile = reinterpret_cast<const char*>(sm.v[warp][stage]);
            const uint8_t* sc = reinterpret_cast<const uint8_t*>(tile) + kRowsFp4Scales;
            const int keys[4] = {2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9};
            // vals[k][w]: F16 pairs of dims 16 g + 2 w, + 1 of key k, scaled.
            uint32_t vals[4][8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint2 raw =
                    *reinterpret_cast<const uint2*>(tile + rows_offset<F>(keys[k], g / 2) + 8 * (g % 2));
                const __half2 scale2 = e4m3_half2(sc[keys[k] * kFp4HeadScales + g]);
                uint32_t lo[4];
                uint32_t hi[4];
                e2m1x8_to_f16x8(raw.x, lo);
                e2m1x8_to_f16x8(raw.y, hi);
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    vals[k][w] = hmul2_bits(lo[w], scale2);
                    vals[k][4 + w] = hmul2_bits(hi[w], scale2);
                }
            }
            // F16 pair of dim 16 g + d of keys k and k + 1, the first key in the low half.
            const auto pair = [&](int k, int dim) {
                return __byte_perm(vals[k][dim / 2], vals[k + 1][dim / 2], dim % 2 ? 0x7632u : 0x5410u);
            };
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t a[4] = {pair(0, i), pair(0, 8 + i), pair(2, i), pair(2, 8 + i)};
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    mma_f16(o[j][i], a, pb[j][0], pb[j][1]);
                }
            }
        } else if constexpr (kFp8) {
            // A fragments of V^T: m-tile i holds dims 16 g + i and 16 g + 8 + i of row g, from
            // chunk g of keys 2 t, 2 t + 1, 2 t + 8 and 2 t + 9.
            const char* tile = reinterpret_cast<const char*>(sm.v[warp][stage]);
            const uint4 va = *reinterpret_cast<const uint4*>(tile + rows_offset<F>(2 * t, g));
            const uint4 vb = *reinterpret_cast<const uint4*>(tile + rows_offset<F>(2 * t + 1, g));
            const uint4 vc = *reinterpret_cast<const uint4*>(tile + rows_offset<F>(2 * t + 8, g));
            const uint4 vd = *reinterpret_cast<const uint4*>(tile + rows_offset<F>(2 * t + 9, g));
            const uint32_t wa[4] = {va.x, va.y, va.z, va.w};
            const uint32_t wb[4] = {vb.x, vb.y, vb.z, vb.w};
            const uint32_t wc[4] = {vc.x, vc.y, vc.z, vc.w};
            const uint32_t wd[4] = {vd.x, vd.y, vd.z, vd.w};
            // Byte i of two keys' chunks as an E4M3 pair, first key in the low byte.
            uint32_t ab[8];
            uint32_t cd[8];
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                ab[2 * w] = __byte_perm(wa[w], wb[w], 0x5140);
                ab[2 * w + 1] = __byte_perm(wa[w], wb[w], 0x7362);
                cd[2 * w] = __byte_perm(wc[w], wd[w], 0x5140);
                cd[2 * w + 1] = __byte_perm(wc[w], wd[w], 0x7362);
            }
            const auto pair = [](const uint32_t(&x)[8], int i) { return (x[i / 2] >> (16 * (i % 2))) & 0xffffu; };
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t a[4] = {e4m3x2_to_f16x2(pair(ab, i)), e4m3x2_to_f16x2(pair(ab, 8 + i)),
                                       e4m3x2_to_f16x2(pair(cd, i)), e4m3x2_to_f16x2(pair(cd, 8 + i))};
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    mma_f16(o[j][i], a, pb[j][0], pb[j][1]);
                }
            }
        } else {
            // A fragments of V^T: m-tile i holds dims 16 i + g and 16 i + 8 + g.
            const char* tile = reinterpret_cast<const char*>(sm.v[warp][stage]);
            const int row = lane % 8 + 8 * (lane / 16);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t a[4];
                ldmatrix_x4_trans(a, tile + rows_offset<F>(row, 2 * i + (lane / 8) % 2));
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    a[r] = bf16x2_to_f16x2(a[r]);
                }
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    mma_f16(o[j][i], a, pb[j][0], pb[j][1]);
                }
            }
        }
        __syncwarp();
        if (k + kStages < chunks) {
            rows_copy_chunk<F>(sm.v[warp][stage], v_cache, kv_head, chunk_first(k + kStages), live, lane);
        }
        copy_async_commit();
    }

    // Each column's sum over lanes g = 0 .. 7.
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            l[j][e] += __shfl_xor_sync(0xffffffffu, l[j][e], 4);
            l[j][e] += __shfl_xor_sync(0xffffffffu, l[j][e], 8);
            l[j][e] += __shfl_xor_sync(0xffffffffu, l[j][e], 16);
        }
    }
    copy_async_wait_all();
    __syncthreads();

    // Merge the warps: each scales its outputs and sum by e^(its max - the block's max);
    // the outputs add as (o0 + o2) + (o1 + o3), the sums in warp order.
    auto& mg = *reinterpret_cast<RowsMerge*>(smem);
    if (g == 0) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int qi = 8 * j + 2 * t + e;
                mg.m[warp][qi] = m[j][e];
                mg.l[warp][qi] = l[j][e];
            }
        }
    }
    __syncthreads();
    float scale_w[NT][2];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int qi = 8 * j + 2 * t + e;
            float most = mg.m[0][qi];
#pragma unroll
            for (int w = 1; w < kRowsWarps; ++w) {
                most = fmaxf(most, mg.m[w][qi]);
            }
            scale_w[j][e] = exp2_ftz((m[j][e] - most) * kLog2e);
        }
    }
    const auto dim_of = [&](int i, int row) { return kFp8 ? 16 * (row % 8) + 8 * (row / 8) + i : 16 * i + row; };
    const auto store = [&](bool add) {
        float(&dst)[kRowsQueries][kRowsMergeStride] = mg.o[warp % 2];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    const int qi = 8 * j + 2 * t + r % 2;
                    const int dim = rows_merge_column(dim_of(i, g + 8 * (r / 2)));
                    const float x = o[j][i][r] * scale_w[j][r % 2];
                    dst[qi][dim] = add ? dst[qi][dim] + x : x;
                }
            }
        }
    };
    if (warp < 2) {
        store(false);
    }
    __syncthreads();
    if (warp >= 2) {
        store(true);
    }
    __syncthreads();
    const int d = threadIdx.x;
    const int column = rows_merge_column(d);
    if constexpr (kWritePartials) {
        for (int qi = 0; qi < queries; ++qi) {
            const size_t query = size_t(qi / kGroup) * kHeads + head0 + qi % kGroup;
            partial_out[(query * kRowsParts + block) * kHeadDim + d] = mg.o[0][qi][column] + mg.o[1][qi][column];
        }
        if (threadIdx.x < queries) {
            const int qi = threadIdx.x;
            float most = mg.m[0][qi];
#pragma unroll
            for (int w = 1; w < kRowsWarps; ++w) {
                most = fmaxf(most, mg.m[w][qi]);
            }
            float total = 0.f;
#pragma unroll
            for (int w = 0; w < kRowsWarps; ++w) {
                total = fmaf(exp2_ftz((mg.m[w][qi] - most) * kLog2e), mg.l[w][qi], total);
            }
            const size_t query = size_t(qi / kGroup) * kHeads + head0 + qi % kGroup;
            partial_max[query * kRowsParts + block] = most;
            partial_sum[query * kRowsParts + block] = total;
        }
    } else {
        for (int qi = 0; qi < queries; ++qi) {
            mg.o[0][qi][column] += mg.o[1][qi][column];
        }
        if (threadIdx.x < queries) {
            const int qi = threadIdx.x;
            float most = mg.m[0][qi];
#pragma unroll
            for (int w = 1; w < kRowsWarps; ++w) {
                most = fmaxf(most, mg.m[w][qi]);
            }
            float total = 0.f;
#pragma unroll
            for (int w = 0; w < kRowsWarps; ++w) {
                total = fmaf(exp2_ftz((mg.m[w][qi] - most) * kLog2e), mg.l[w][qi], total);
            }
            mg.m[0][qi] = most;
            mg.l[0][qi] = total;
        }
    }
}

template <class F, int NT, bool kSingleCombiner = false, bool kTree = false>
__device__ __forceinline__ void attention_rows(const __nv_bfloat16* q, const __nv_bfloat16* k_new,
                                               const __nv_bfloat16* v_new, typename F::Elem* k_cache,
                                               typename F::Elem* v_cache, const __nv_bfloat16* bias,
                                               const __nv_bfloat16* sinks, float* partial_out, float* partial_max,
                                               float* partial_sum, unsigned* counters, __nv_bfloat16* out,
                                               const __nv_bfloat16* gate, const unsigned* start_ptr, int rows,
                                               unsigned slots, unsigned window, unsigned extent, float scale,
                                               int split_blocks, char* smem, const uint32_t* tree_depth = nullptr,
                                               const int* tree_attn_anc = nullptr,
                                               typename F::Elem* k_stage = nullptr,
                                               typename F::Elem* v_stage = nullptr) {
    const int kv_head = blockIdx.x;
    // The grid launches split blocks 0 .. split_blocks - 1, then recent blocks from 0: block
    // index kRowsBlocks + r is recent block r. Each split block strides cache splits by
    // `split_blocks` (1..=kRowsBlocks), so every live slot is covered exactly once.
    const int y = int(blockIdx.y);
    const int block = y < split_blocks ? y : kRowsBlocks + (y - split_blocks);
    const int queries = kGroup * rows;
    pdl_trigger();
    const unsigned start = *start_ptr;
    // Live slots of the cache: in tree mode only prefix slots below `start` hold keys
    // (new tree keys are staged separately and never written to the cache during forward).
    const unsigned live = kTree ? min(start, slots) : min(start + unsigned(rows), slots);
    // Split blocks with any key: those whose first split starts below the live slots.
    const int used = min(split_blocks, int((live + kRowsSplitSlots - 1) / kRowsSplitSlots));
    if (block >= kRowsBlocks) {
        pdl_wait();
        rows_recent<F, false, kTree>(q, k_new, v_new, k_cache, v_cache, bias, sinks, partial_out, partial_max,
                                     partial_sum, start, rows, slots, window, extent, scale, block - kRowsBlocks,
                                     nullptr, nullptr, nullptr, tree_depth, tree_attn_anc);
    } else if (block < used) {
        rows_split<F, NT, true, kTree>(q, k_cache, v_cache, bias, partial_out, partial_max, partial_sum, start, rows,
                                       slots, window, extent, scale, live, block, split_blocks, smem, tree_depth);
    } else {
        pdl_wait();
    }

    const unsigned blocks = gridDim.y;
    __syncthreads();
    auto& combine = *reinterpret_cast<RowsCombine*>(smem);
    if constexpr (kSingleCombiner) {
        if (threadIdx.x == 0) {
            combine.arrival = atomic_add_acq_rel(&counters[kv_head], 1u);
        }
        __syncthreads();
        if (combine.arrival != blocks - 1) {
            return;
        }
        rows_combine<F, true, kTree>(combine, partial_out, partial_max, partial_sum, out, gate, k_new, v_new, k_cache,
                                     v_cache, start, rows, slots, used, 0, 1, k_stage, v_stage);
        __syncthreads();
        if (threadIdx.x == 0) {
            counters[kv_head] = 0;
        }
        return;
    }

    // Count finished blocks. The last `combiners` to finish share the KV head's combine;
    // all but the very last wait until every block has counted. At most 23 blocks per KV
    // head wait, and every other block leaves once counted, so a block that has not started yet gets
    // an SM; the next kernel, which waits for this one, launches only once every block of
    // this one has started, so it takes none of their SMs first. The barrier
    // orders the block's partial stores before thread 0's release, and its acquire before
    // the other threads' loads after the second barrier. counters[kv_head] counts the
    // arrivals and counters[8 + kv_head] the finished combiners; the last combiner resets
    // both for the next call.
    const int combiners = min(queries, kRowsCombiners);
    if (threadIdx.x == 0) {
        const unsigned arrival = atomic_add_acq_rel(&counters[kv_head], 1u);
        if (arrival >= blocks - unsigned(combiners) && arrival != blocks - 1) {
            while (load_acquire(&counters[kv_head]) != blocks) {
                __nanosleep(64);
            }
        }
        combine.arrival = arrival;
    }
    __syncthreads();
    const int combiner = int(combine.arrival) - int(blocks - unsigned(combiners));
    if (combiner < 0) {
        return;
    }
    rows_combine<F, false, kTree>(combine, partial_out, partial_max, partial_sum, out, gate, k_new, v_new, k_cache,
                                  v_cache, start, rows, slots, used, combiner, combiners, k_stage, v_stage);
    __syncthreads();
    if (threadIdx.x == 0 && atomic_add_acq_rel(&counters[kKvHeads + kv_head], 1u) == unsigned(combiners - 1)) {
        counters[kv_head] = 0;
        counters[kKvHeads + kv_head] = 0;
    }
}

// Fused single-block no-split attention for large batches (when `seqs * kKvHeads` already
// saturates the GPU's SMs): one block per (kv_head, seq) streams all cache slots (stride 1),
// keeps the merged split result in shared memory (`RowsMerge`), runs `rows_recent` in
// registers with direct combine into `out`, and appends `k_new` and `v_new` to the caches.
// Zero partial-buffer traffic, zero atomic counters, zero spin-waiting.
template <class F, int NT>
__device__ __forceinline__ void attention_rows_nosplit(
    const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, typename F::Elem* k_cache,
    typename F::Elem* v_cache, const __nv_bfloat16* bias, const __nv_bfloat16* sinks, __nv_bfloat16* out,
    const __nv_bfloat16* gate, const unsigned* start_ptr, int rows, unsigned slots, unsigned window,
    unsigned extent, float scale, char* smem) {
    const int kv_head = blockIdx.x;
    pdl_trigger();
    const unsigned start = *start_ptr;
    const unsigned live = min(start + unsigned(rows), slots);
    // Any row has a key at distance >= kRowsRecent iff the last row's position >= kRowsRecent.
    const bool has_far = start + unsigned(rows - 1) >= unsigned(kRowsRecent);
    auto& mg = *reinterpret_cast<RowsMerge*>(smem);
    if (has_far) {
        rows_split<F, NT, false>(q, k_cache, v_cache, bias, nullptr, nullptr, nullptr, start, rows, slots, window,
                                 extent, scale, live, 0, 1, smem);
        __syncthreads();
    } else {
        pdl_wait();
    }
    const int recent_blocks = (rows + kRowsWarps - 1) / kRowsWarps;
    for (int rb = 0; rb < recent_blocks; ++rb) {
        rows_recent<F, true>(q, k_new, v_new, k_cache, v_cache, bias, sinks, nullptr, nullptr, nullptr, start, rows,
                             slots, window, extent, scale, rb, has_far ? &mg : nullptr, out, gate);
    }
    // Wait until every warp finishes reading the caches before appending the rows' new keys
    // and values (a sliding ring's new slot may be read by another row of the same call).
    __syncthreads();
    if (k_cache != nullptr) {
        const int d = threadIdx.x;
        const size_t channel = size_t(kv_head) * kHeadDim + d;
        for (int row = 0; row < rows; ++row) {
            if constexpr (F::kFp4) {
                // Thread 16 i rounds group i of the head, as kv_append_rows_fp4 does.
                if (d % 16 == 0) {
                    const unsigned slot_idx = (start + unsigned(row)) % slots;
#pragma unroll
                    for (int kv = 0; kv < 2; ++kv) {
                        const __nv_bfloat16* x = (kv == 0 ? k_new : v_new) + size_t(row) * kKvRow + channel;
                        float vals[16];
#pragma unroll
                        for (int i = 0; i < 16; ++i) {
                            vals[i] = __bfloat162float(x[i]);
                        }
                        uint8_t sc;
                        const uint2 data = fp4_group(vals, sc);
                        typename F::Elem* cache = kv == 0 ? k_cache : v_cache;
                        *reinterpret_cast<uint2*>(cache + fp4_data(slot_idx, kv_head) + d / 2) = data;
                        cache[fp4_scales(slot_idx, kv_head) + d / 16] = sc;
                    }
                }
            } else {
                const size_t dst = size_t((start + unsigned(row)) % slots) * kKvRow + channel;
                F::store1(k_cache + dst, k_new[size_t(row) * kKvRow + channel]);
                F::store1(v_cache + dst, v_new[size_t(row) * kKvRow + channel]);
            }
        }
    }
}

}  // namespace

// The attention of `rows` rows (see above), combined and appended by the last blocks of
// each KV head to finish (rows_combine). q: [rows, 24, 128]; k_new, v_new: the rows' keys and
// values, [rows, 8, 128], not yet in the cache; bias: [rows, 24, extent]; sinks: [rows,
// 24]; partials: [rows * 24, kRowsParts] of 128, 1 and 1 floats; counters: 16 zeros,
// left zero; out and gate (or null): [rows, 24, 128]. Grid (8, kRowsBlocks +
// kRowsRecentBlocks), 128 threads, sizeof(RowsShared<F>) bytes of dynamic shared memory.
// `attention_rows_<format>_<NT>` takes rows with ceil(3 rows / 8) = NT.
#define ATTENTION_ROWS(NT)                                                                                   \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_rows_bf16_##NT(                  \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, __nv_bfloat16* k_cache, \
        __nv_bfloat16* v_cache, const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out,       \
        float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out,                         \
        const __nv_bfloat16* gate, const unsigned* start, int rows, unsigned slots, unsigned window,            \
        unsigned extent, float scale) {                                                                          \
        extern __shared__ __align__(16) char rows_smem[];                                                        \
        attention_rows<Bf16Format, NT>(q, k_new, v_new, k_cache, v_cache, bias, sinks, partial_out,              \
                                       partial_max, partial_sum, counters, out, gate, start, rows, slots,        \
                                       window, extent, scale, kRowsBlocks, rows_smem);                           \
    }                                                                                                            \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_rows_fp8_##NT(                   \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, uint8_t* k_cache,      \
        uint8_t* v_cache, const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out,             \
        float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out,                         \
        const __nv_bfloat16* gate, const unsigned* start, int rows, unsigned slots, unsigned window,            \
        unsigned extent, float scale) {                                                                          \
        extern __shared__ __align__(16) char rows_smem[];                                                        \
        attention_rows<Fp8Format, NT>(q, k_new, v_new, k_cache, v_cache, bias, sinks, partial_out,               \
                                      partial_max, partial_sum, counters, out, gate, start, rows, slots,         \
                                      window, extent, scale, kRowsBlocks, rows_smem);                            \
    }

ATTENTION_ROWS(1)
ATTENTION_ROWS(2)
ATTENTION_ROWS(3)

// The NVFP4 variant of attention_rows, `attention_rows_fp4_<NT>`.
#define ATTENTION_ROWS_FP4(NT)                                                                                \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_rows_fp4_##NT(                   \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, uint8_t* k_cache,      \
        uint8_t* v_cache, const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out,             \
        float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out,                         \
        const __nv_bfloat16* gate, const unsigned* start, int rows, unsigned slots, unsigned window,            \
        unsigned extent, float scale) {                                                                          \
        extern __shared__ __align__(16) char rows_smem[];                                                        \
        attention_rows<Fp4Format, NT>(q, k_new, v_new, k_cache, v_cache, bias, sinks, partial_out,               \
                                      partial_max, partial_sum, counters, out, gate, start, rows, slots,         \
                                      window, extent, scale, kRowsBlocks, rows_smem);                            \
    }

ATTENTION_ROWS_FP4(1)
ATTENTION_ROWS_FP4(2)
ATTENTION_ROWS_FP4(3)

// Tree-verify attention (`attention_rows_tree_{bf16,fp8,fp4}_<NT>`): each row attends over
// the committed prefix (`k_cache`, `v_cache`) plus its own tree ancestors (`tree_attn_anc`)
// at true sequence position `*start + tree_depth[row]`, and stages its native-format K/V
// into `k_stage`, `v_stage` (`[8, row_bytes]`) instead of mutating the caches.
#define ATTENTION_ROWS_TREE(NT)                                                                                     \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_rows_tree_bf16_##NT(                    \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, __nv_bfloat16* k_cache,     \
        __nv_bfloat16* v_cache, const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out,          \
        float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out, const __nv_bfloat16* gate,  \
        const unsigned* start, int rows, unsigned slots, unsigned window, unsigned extent, float scale,             \
        const uint32_t* tree_depth, const int* tree_attn_anc, __nv_bfloat16* k_stage, __nv_bfloat16* v_stage) {     \
        extern __shared__ __align__(16) char rows_smem[];                                                           \
        attention_rows<Bf16Format, NT, false, true>(q, k_new, v_new, k_cache, v_cache, bias, sinks, partial_out,    \
                                                    partial_max, partial_sum, counters, out, gate, start, rows,     \
                                                    slots, window, extent, scale, kRowsBlocks, rows_smem,           \
                                                    tree_depth, tree_attn_anc, k_stage, v_stage);                   \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_rows_tree_fp8_##NT(                     \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, uint8_t* k_cache,           \
        uint8_t* v_cache, const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out,                \
        float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out, const __nv_bfloat16* gate,  \
        const unsigned* start, int rows, unsigned slots, unsigned window, unsigned extent, float scale,             \
        const uint32_t* tree_depth, const int* tree_attn_anc, uint8_t* k_stage, uint8_t* v_stage) {                 \
        extern __shared__ __align__(16) char rows_smem[];                                                           \
        attention_rows<Fp8Format, NT, false, true>(q, k_new, v_new, k_cache, v_cache, bias, sinks, partial_out,     \
                                                   partial_max, partial_sum, counters, out, gate, start, rows,      \
                                                   slots, window, extent, scale, kRowsBlocks, rows_smem,            \
                                                   tree_depth, tree_attn_anc, k_stage, v_stage);                    \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_rows_tree_fp4_##NT(                     \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new, uint8_t* k_cache,           \
        uint8_t* v_cache, const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out,                \
        float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out, const __nv_bfloat16* gate,  \
        const unsigned* start, int rows, unsigned slots, unsigned window, unsigned extent, float scale,             \
        const uint32_t* tree_depth, const int* tree_attn_anc, uint8_t* k_stage, uint8_t* v_stage) {                 \
        extern __shared__ __align__(16) char rows_smem[];                                                           \
        attention_rows<Fp4Format, NT, false, true>(q, k_new, v_new, k_cache, v_cache, bias, sinks, partial_out,     \
                                                   partial_max, partial_sum, counters, out, gate, start, rows,      \
                                                   slots, window, extent, scale, kRowsBlocks, rows_smem,            \
                                                   tree_depth, tree_attn_anc, k_stage, v_stage);                    \
    }

ATTENTION_ROWS_TREE(1)
ATTENTION_ROWS_TREE(2)
ATTENTION_ROWS_TREE(3)

// Words per row of a batched step's state table (see `BATCH_STATE_WORDS` in ops/spec.rs):
// the device addresses of the row's key cache, value cache, and key, value,
// post-attention and post-MLP conv states.
constexpr int kBatchStateWords = 6;

// The attention of a batched step: row z of the batch (blockIdx.z), each from its own
// sequence, attends from its query into its own cache at its own position, as
// attention_rows does for one row, and appends its key and value there. states: the
// rows' state tables, [rows, kBatchStateWords] addresses; positions: [rows]. q, k_new,
// v_new, bias, sinks, gate and out are [rows, ...] as in attention_rows; each row takes
// its own 24 query rows of the partials and its own 16 counters ([rows, 16]).
// Each instance launches `split_blocks` split blocks (1..=kRowsBlocks, striding cache splits
// by `split_blocks`) and the first recent block (row 0), using single-block combine
// (`kSingleCombiner = true`) so zero blocks ever spin-wait and any `split_blocks >= 1` is
// valid. With `split_blocks == slots.div_ceil(64).clamp(1, 32)`, results are bit-identical
// to `attention_rows`.
// Grid (8, split_blocks + 1, rows), 128 threads, sizeof(RowsShared<F>) bytes of dynamic
// shared memory.
template <class F>
__device__ __forceinline__ void attention_batch(const __nv_bfloat16* q, const __nv_bfloat16* k_new,
                                                const __nv_bfloat16* v_new, const unsigned long long* states,
                                                const __nv_bfloat16* bias, const __nv_bfloat16* sinks,
                                                float* partial_out, float* partial_max, float* partial_sum,
                                                unsigned* counters, __nv_bfloat16* out, const __nv_bfloat16* gate,
                                                const unsigned* positions, unsigned slots, unsigned window,
                                                unsigned extent, float scale, int split_blocks, char* smem) {
    const size_t z = blockIdx.z;
    const size_t q_at = z * kHeads * kHeadDim;
    const size_t kv_at = z * kKvRow;
    const size_t query0 = z * kHeads * kRowsParts;
    auto* const k_cache = reinterpret_cast<typename F::Elem*>(states[z * kBatchStateWords]);
    auto* const v_cache = reinterpret_cast<typename F::Elem*>(states[z * kBatchStateWords + 1]);
    attention_rows<F, 1, true>(q + q_at, k_new + kv_at, v_new + kv_at, k_cache, v_cache,
                               bias + z * kHeads * extent, sinks + z * kHeads, partial_out + query0 * kHeadDim,
                               partial_max + query0, partial_sum + query0, counters + z * 2 * kKvHeads, out + q_at,
                               gate != nullptr ? gate + q_at : nullptr, positions + z, 1, slots, window, extent,
                               scale, split_blocks, smem);
}

extern "C" __global__ void __launch_bounds__(kRowsThreads, 2)
attention_batch_bf16(const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
                     const unsigned long long* states, const __nv_bfloat16* bias, const __nv_bfloat16* sinks,
                     float* partial_out, float* partial_max, float* partial_sum, unsigned* counters,
                     __nv_bfloat16* out, const __nv_bfloat16* gate, const unsigned* positions, unsigned slots,
                     unsigned window, unsigned extent, float scale, int split_blocks) {
    extern __shared__ __align__(16) char rows_smem[];
    attention_batch<Bf16Format>(q, k_new, v_new, states, bias, sinks, partial_out, partial_max, partial_sum,
                                counters, out, gate, positions, slots, window, extent, scale, split_blocks,
                                rows_smem);
}

extern "C" __global__ void __launch_bounds__(kRowsThreads, 2)
attention_batch_fp8(const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
                    const unsigned long long* states, const __nv_bfloat16* bias, const __nv_bfloat16* sinks,
                    float* partial_out, float* partial_max, float* partial_sum, unsigned* counters,
                    __nv_bfloat16* out, const __nv_bfloat16* gate, const unsigned* positions, unsigned slots,
                    unsigned window, unsigned extent, float scale, int split_blocks) {
    extern __shared__ __align__(16) char rows_smem[];
    attention_batch<Fp8Format>(q, k_new, v_new, states, bias, sinks, partial_out, partial_max, partial_sum,
                               counters, out, gate, positions, slots, window, extent, scale, split_blocks,
                               rows_smem);
}

extern "C" __global__ void __launch_bounds__(kRowsThreads, 2)
attention_batch_fp4(const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
                    const unsigned long long* states, const __nv_bfloat16* bias, const __nv_bfloat16* sinks,
                    float* partial_out, float* partial_max, float* partial_sum, unsigned* counters,
                    __nv_bfloat16* out, const __nv_bfloat16* gate, const unsigned* positions, unsigned slots,
                    unsigned window, unsigned extent, float scale, int split_blocks) {
    extern __shared__ __align__(16) char rows_smem[];
    attention_batch<Fp4Format>(q, k_new, v_new, states, bias, sinks, partial_out, partial_max, partial_sum,
                               counters, out, gate, positions, slots, window, extent, scale, split_blocks,
                               rows_smem);
}

// Batched multi-row verify attention: sequence z (blockIdx.z) verifies `rows` (1..=8)
// consecutive tokens starting at `positions[z]` in its own cache (`states[z * 6 + {0,1}]`).
// Uses single-block combine (kSingleCombiner = true) so zero blocks ever spin-wait and
// `split_blocks` may be below `3 * rows` (e.g. 16 for 1,024-slot sliding rings).
// Grid (8, split_blocks + ceil(rows / 4), seqs), 128 threads, sizeof(RowsShared<F>) smem.
template <class F, int NT>
__device__ __forceinline__ void attention_batch_verify(
    const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
    const unsigned long long* states, const __nv_bfloat16* bias, const __nv_bfloat16* sinks, float* partial_out,
    float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out, const __nv_bfloat16* gate,
    const unsigned* positions, int rows, unsigned slots, unsigned window, unsigned extent, float scale,
    int split_blocks, char* smem) {
    const size_t z = blockIdx.z;
    const size_t row0 = z * size_t(rows);
    const size_t q_at = row0 * kHeads * kHeadDim;
    const size_t kv_at = row0 * kKvRow;
    const size_t query0 = row0 * kHeads * kRowsParts;
    auto* const k_cache = reinterpret_cast<typename F::Elem*>(states[z * kBatchStateWords]);
    auto* const v_cache = reinterpret_cast<typename F::Elem*>(states[z * kBatchStateWords + 1]);
    attention_rows<F, NT, true>(q + q_at, k_new + kv_at, v_new + kv_at, k_cache, v_cache,
                                bias + row0 * kHeads * extent, sinks + row0 * kHeads,
                                partial_out + query0 * kHeadDim, partial_max + query0, partial_sum + query0,
                                counters + z * 2 * kKvHeads, out + q_at,
                                gate != nullptr ? gate + q_at : nullptr, positions + z, rows, slots, window,
                                extent, scale, split_blocks, smem);
}

#define ATTENTION_BATCH_VERIFY(NT)                                                                             \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_batch_verify_bf16_##NT(            \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,                        \
        const unsigned long long* states, const __nv_bfloat16* bias, const __nv_bfloat16* sinks,               \
        float* partial_out, float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out,    \
        const __nv_bfloat16* gate, const unsigned* positions, int rows, unsigned slots, unsigned window,       \
        unsigned extent, float scale, int split_blocks) {                                                      \
        extern __shared__ __align__(16) char rows_smem[];                                                      \
        attention_batch_verify<Bf16Format, NT>(q, k_new, v_new, states, bias, sinks, partial_out, partial_max, \
                                               partial_sum, counters, out, gate, positions, rows, slots,       \
                                               window, extent, scale, split_blocks, rows_smem);                \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_batch_verify_fp8_##NT(             \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,                        \
        const unsigned long long* states, const __nv_bfloat16* bias, const __nv_bfloat16* sinks,               \
        float* partial_out, float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out,    \
        const __nv_bfloat16* gate, const unsigned* positions, int rows, unsigned slots, unsigned window,       \
        unsigned extent, float scale, int split_blocks) {                                                      \
        extern __shared__ __align__(16) char rows_smem[];                                                      \
        attention_batch_verify<Fp8Format, NT>(q, k_new, v_new, states, bias, sinks, partial_out, partial_max,  \
                                              partial_sum, counters, out, gate, positions, rows, slots,        \
                                              window, extent, scale, split_blocks, rows_smem);                 \
    }                                                                                                          \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_batch_verify_fp4_##NT(             \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,                        \
        const unsigned long long* states, const __nv_bfloat16* bias, const __nv_bfloat16* sinks,               \
        float* partial_out, float* partial_max, float* partial_sum, unsigned* counters, __nv_bfloat16* out,    \
        const __nv_bfloat16* gate, const unsigned* positions, int rows, unsigned slots, unsigned window,       \
        unsigned extent, float scale, int split_blocks) {                                                      \
        extern __shared__ __align__(16) char rows_smem[];                                                      \
        attention_batch_verify<Fp4Format, NT>(q, k_new, v_new, states, bias, sinks, partial_out, partial_max,  \
                                              partial_sum, counters, out, gate, positions, rows, slots,        \
                                              window, extent, scale, split_blocks, rows_smem);                 \
    }

ATTENTION_BATCH_VERIFY(1)
ATTENTION_BATCH_VERIFY(2)
ATTENTION_BATCH_VERIFY(3)

// Fused single-block no-split batched attention (both decode `rows == 1` and verify
// `rows == 1..=8`): one CTA per `(kv_head, seq)` reads all cache slots + recent keys,
// combines in shared memory/registers, writes `out` (with optional `gate`), and appends
// `k_new, v_new`. Zero partial-buffer traffic and zero atomic counters.
// Grid (8, 1, seqs), 128 threads, sizeof(RowsShared<F>) smem.
template <class F, int NT>
__device__ __forceinline__ void attention_batch_nosplit(
    const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,
    const unsigned long long* states, const __nv_bfloat16* bias, const __nv_bfloat16* sinks,
    __nv_bfloat16* out, const __nv_bfloat16* gate, const unsigned* positions, int rows, unsigned slots,
    unsigned window, unsigned extent, float scale, char* smem) {
    const size_t z = blockIdx.z;
    const size_t row0 = z * size_t(rows);
    const size_t q_at = row0 * kHeads * kHeadDim;
    const size_t kv_at = row0 * kKvRow;
    auto* const k_cache = reinterpret_cast<typename F::Elem*>(states[z * kBatchStateWords]);
    auto* const v_cache = reinterpret_cast<typename F::Elem*>(states[z * kBatchStateWords + 1]);
    attention_rows_nosplit<F, NT>(q + q_at, k_new + kv_at, v_new + kv_at, k_cache, v_cache,
                                  bias + row0 * kHeads * extent, sinks + row0 * kHeads, out + q_at,
                                  gate != nullptr ? gate + q_at : nullptr, positions + z, rows, slots, window,
                                  extent, scale, smem);
}

#define ATTENTION_BATCH_NOSPLIT(NT)                                                                             \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_batch_nosplit_bf16_##NT(            \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,                         \
        const unsigned long long* states, const __nv_bfloat16* bias, const __nv_bfloat16* sinks,                \
        __nv_bfloat16* out, const __nv_bfloat16* gate, const unsigned* positions, int rows, unsigned slots,     \
        unsigned window, unsigned extent, float scale) {                                                        \
        extern __shared__ __align__(16) char rows_smem[];                                                       \
        attention_batch_nosplit<Bf16Format, NT>(q, k_new, v_new, states, bias, sinks, out, gate, positions,     \
                                                rows, slots, window, extent, scale, rows_smem);                 \
    }                                                                                                           \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_batch_nosplit_fp8_##NT(             \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,                         \
        const unsigned long long* states, const __nv_bfloat16* bias, const __nv_bfloat16* sinks,                \
        __nv_bfloat16* out, const __nv_bfloat16* gate, const unsigned* positions, int rows, unsigned slots,     \
        unsigned window, unsigned extent, float scale) {                                                        \
        extern __shared__ __align__(16) char rows_smem[];                                                       \
        attention_batch_nosplit<Fp8Format, NT>(q, k_new, v_new, states, bias, sinks, out, gate, positions,      \
                                               rows, slots, window, extent, scale, rows_smem);                  \
    }                                                                                                           \
    extern "C" __global__ void __launch_bounds__(kRowsThreads, 2) attention_batch_nosplit_fp4_##NT(             \
        const __nv_bfloat16* q, const __nv_bfloat16* k_new, const __nv_bfloat16* v_new,                         \
        const unsigned long long* states, const __nv_bfloat16* bias, const __nv_bfloat16* sinks,                \
        __nv_bfloat16* out, const __nv_bfloat16* gate, const unsigned* positions, int rows, unsigned slots,     \
        unsigned window, unsigned extent, float scale) {                                                        \
        extern __shared__ __align__(16) char rows_smem[];                                                       \
        attention_batch_nosplit<Fp4Format, NT>(q, k_new, v_new, states, bias, sinks, out, gate, positions,      \
                                               rows, slots, window, extent, scale, rows_smem);                  \
    }

ATTENTION_BATCH_NOSPLIT(1)
ATTENTION_BATCH_NOSPLIT(2)
ATTENTION_BATCH_NOSPLIT(3)

// The launcher passes these as the dynamic shared memory of the three kernels.
static_assert(sizeof(RowsShared<Bf16Format>) == 32768);
static_assert(sizeof(RowsShared<Fp8Format>) == 49152);
static_assert(sizeof(RowsShared<Fp4Format>) == 27648);
static_assert(sizeof(RowsMerge) <= sizeof(RowsShared<Fp4Format>));
static_assert(sizeof(RowsCombine) <= sizeof(RowsShared<Fp4Format>));

// Rounds a global cache's keys and values through a 4-bit format and back, in place,
// to measure what a 4-bit cache would cost in quality (`KvQuant` on the host). The
// storage stays BF16, which holds every rounded value exactly for NVFP4 and to 8
// significant bits for q4_0. Row r is row *start + r of k and v, or row r without
// start; each row is [8 heads, 128]. One warp per head of one row of k (blockIdx.y
// 0) or v (1); lane l holds elements 4l .. 4l + 3.
//
// mode 1, NVFP4: per 16 elements, scale s = E4M3(amax / 6) rounded to nearest, each
// element E2M1(x / s) rounded to nearest, ties to even, saturated to +-6.
// mode 2, q4_0 as llama.cpp's quantize_row_q4_0: per 32 elements, d = m / -8 for the
// signed element m of largest magnitude, q = min(15, floor(x / d + 8.5)), value
// (q - 8) * F16(d).
// With rotate, each head goes through the orthonormal 128-point Hadamard transform
// before rounding and back after, as if queries and outputs were rotated the same way
// and the cache stored rotated keys and values.
namespace {

__device__ __forceinline__ float e2m1_round(float y) {
    const float a = fabsf(y);
    float r;
    if (a <= 0.25f) {
        r = 0.f;
    } else if (a < 0.75f) {
        r = 0.5f;
    } else if (a <= 1.25f) {
        r = 1.f;
    } else if (a < 1.75f) {
        r = 1.5f;
    } else if (a <= 2.5f) {
        r = 2.f;
    } else if (a < 3.5f) {
        r = 3.f;
    } else if (a <= 5.f) {
        r = 4.f;
    } else {
        r = 6.f;
    }
    return copysignf(r, y);
}

// The orthonormal Hadamard transform (Sylvester order) of the warp's 128 elements;
// its own inverse.
__device__ __forceinline__ void hadamard128(float (&x)[4]) {
    const float a0 = x[0] + x[1], a1 = x[0] - x[1], a2 = x[2] + x[3], a3 = x[2] - x[3];
    x[0] = a0 + a2;
    x[2] = a0 - a2;
    x[1] = a1 + a3;
    x[3] = a1 - a3;
    const unsigned lane = threadIdx.x & 31u;
#pragma unroll
    for (unsigned m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, x[j], m);
            x[j] = (lane & m) ? o - x[j] : x[j] + o;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        x[j] *= 0.08838834764831845f;
    }
}

__device__ __forceinline__ void round_nvfp4(float (&x)[4]) {
    float amax = fmaxf(fmaxf(fabsf(x[0]), fabsf(x[1])), fmaxf(fabsf(x[2]), fabsf(x[3])));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(amax / 6.f, __NV_SATFINITE, __NV_E4M3);
    const float s = __half2float(__half(__nv_cvt_fp8_to_halfraw(s8, __NV_E4M3)));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        x[j] = s == 0.f ? 0.f : e2m1_round(__fdiv_rn(x[j], s)) * s;
    }
}

__device__ __forceinline__ void round_q4_0(float (&x)[4]) {
    float m = x[0];
#pragma unroll
    for (int j = 1; j < 4; ++j) {
        if (fabsf(x[j]) > fabsf(m)) {
            m = x[j];
        }
    }
#pragma unroll
    for (unsigned o = 1; o < 8; o <<= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, m, o);
        // Ties keep the earlier element, as llama.cpp's scan does.
        const bool lower = (threadIdx.x & o) != 0;
        if (fabsf(other) > fabsf(m) || (fabsf(other) == fabsf(m) && lower)) {
            m = other;
        }
    }
    const float d = m / -8.f;
    const float id = d != 0.f ? 1.f / d : 0.f;
    const float dh = __half2float(__float2half_rn(d));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = min(15, int(x[j] * id + 8.5f));
        x[j] = float(q - 8) * dh;
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(256)
kv_round_bf16(__nv_bfloat16* k, __nv_bfloat16* v, const unsigned* start, unsigned first, int mode, int rotate,
              int which) {
    // Everything this kernel reads comes from earlier kernels. It never triggers early, so
    // the next kernel starts after it ends.
    pdl_wait();
    if (!(which & (1 << blockIdx.y))) {
        return;
    }
    const unsigned row = blockIdx.x + first + (start ? *start : 0u);
    __nv_bfloat16* base = (blockIdx.y == 0 ? k : v) + size_t(row) * kKvRow + (threadIdx.x / 32) * kHeadDim +
                          (threadIdx.x & 31u) * 4;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        x[j] = __bfloat162float(base[j]);
    }
    if (rotate) {
        hadamard128(x);
    }
    if (mode == 1) {
        round_nvfp4(x);
    } else if (mode == 2) {
        round_q4_0(x);
    } else {
        unpack_e4m3x4(pack_e4m3x4(x[0], x[1], x[2], x[3]), x);
    }
    if (rotate) {
        hadamard128(x);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        base[j] = __float2bfloat16_rn(x[j]);
    }
}

// Rounds keys per channel instead of per token, as KIVI quantizes them: each group of
// 16 (mode 1, NVFP4) or 32 (mode 2, q4_0) consecutive rows from row `first` shares one
// scale per channel. One thread per group and channel of k ([rows, 1024]).
extern "C" __global__ void __launch_bounds__(256)
kv_round_channels_bf16(__nv_bfloat16* k, unsigned first, unsigned groups, int mode) {
    pdl_wait();
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= groups * kKvRow) {
        return;
    }
    const int rows = mode == 1 ? 16 : 32;
    __nv_bfloat16* base = k + (size_t(first) + size_t(i / kKvRow) * rows) * kKvRow + i % kKvRow;
    float x[32];
    float amax = 0.f, m = 0.f;
    for (int r = 0; r < rows; ++r) {
        x[r] = __bfloat162float(base[size_t(r) * kKvRow]);
        if (fabsf(x[r]) > fabsf(m)) {
            m = x[r];
        }
        amax = fmaxf(amax, fabsf(x[r]));
    }
    if (mode == 1) {
        const __nv_fp8_storage_t s8 = __nv_cvt_float_to_fp8(amax / 6.f, __NV_SATFINITE, __NV_E4M3);
        const float sc = __half2float(__half(__nv_cvt_fp8_to_halfraw(s8, __NV_E4M3)));
        for (int r = 0; r < rows; ++r) {
            x[r] = sc == 0.f ? 0.f : e2m1_round(__fdiv_rn(x[r], sc)) * sc;
        }
    } else {
        const float d = m / -8.f;
        const float id = d != 0.f ? 1.f / d : 0.f;
        const float dh = __half2float(__float2half_rn(d));
        for (int r = 0; r < rows; ++r) {
            x[r] = float(min(15, int(x[r] * id + 8.5f)) - 8) * dh;
        }
    }
    for (int r = 0; r < rows; ++r) {
        base[size_t(r) * kKvRow] = __float2bfloat16_rn(x[r]);
    }
}
