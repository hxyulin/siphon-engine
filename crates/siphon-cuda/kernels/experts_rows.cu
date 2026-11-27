// The expert MLP of 1 to 256 rows: for every (row, slot) pair, the routed expert's w13,
// the softcapped SwiGLU and w2, as experts.cu's decode `expert_mlp` does for one row. Each
// distinct expert's weights stream from DRAM once per call, for all the rows that route to
// it, so the weight bytes follow the distinct experts, not 8 m.
//
// Three resident tiers share the same exact BF16 tensor-core arithmetic and L2::evict_first
// streaming policy (preserving L2 for KV cache, dense weights, and activations across looped
// core passes):
// - m = 1..8 (`expert_mlp_rows`): every expert has at most 8 pairs (one column group), so
//   each block groups in shared memory (1 barrier at m == 1, 3 barriers at m = 2..8, zero
//   warp_count/pstart/vstart/binary search), uses static strided task scheduling, streams
//   16-row weight tiles through a per-warp ring, and writes coalesced 16-byte `act` / 32-byte
//   `out` stores via warp shuffles with zero smem epilogue traffic.
// - m = 9..32 (`expert_mlp_rows_multi`): each block groups pairs deterministically into
//   8-pair column groups in shared memory so neighboring warps process sibling groups of
//   each tile concurrently, with static strided scheduling and the same zero-smem
//   warp-shuffle coalesced stores.
// - m = 33..256 (`expert_group_rows` + `expert_mlp_rows_wide`): a 1-block pre-group kernel
//   reads `indices` once into registers and LPT bucket-sorts active experts into 32-pair
//   `ExpertChunk` descriptors in `sync + sync_end`; the persistent dynamic-queue wide kernel
//   processes up to 32 pairs per weight load, overlaps w2 prefetch with w13->w2 sync waits,
//   and writes coalesced uint4 `act`/`out` stores through the warp's ring smem.
//
// Numerics. Each weight times its block scale is exactly a BF16 value: an E2M1 value has
// at most 2 significant bits and an E4M3 scale 4, an E4M3 weight 4 bits and an E8M0 scale
// none, and their products stay in BF16's normal range for the checkpoint's scales. The
// kernel widens the weights to F16 in hardware, moves the bits to BF16 (2^-112 times the
// value), and multiplies by the scale times 2^112, which is exact. The tensor cores sum
// the exact products with the BF16 inputs in F32. The F32 sum is multiplied by NVFP4's
// `global_scale / (448 * 6)` and rounded to BF16 once, and gate and up go through the
// same softcapped SwiGLU as `expert_mlp`. So the results differ from `expert_mlp` only by
// summation order: it multiplies each block's partial sum by the scale, this kernel each
// product.
//
// Each pair's results depend only on its own input row and expert: the tensor cores
// compute each column separately, every column executes the same instructions in the same
// k order, and the tiles and stages depend on the expert alone. So they are the same, bit
// for bit, for every m and whatever the other rows route to.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstddef>
#include <cstdint>

#include "arch.cuh"
#include "mlx_ops.cuh"
#include "mxfp6.cuh"
#include "pdl.cuh"

namespace {

constexpr int kSingleMaxRows = 8;
constexpr int kMaxRows = 32;
constexpr int kMaxPairs = kMaxRows * 8;
constexpr int kWideMaxRows = 256;
constexpr int kWideMaxPairs = kWideMaxRows * 8;
constexpr int kExperts = 128;
// Warps per block; each runs its own tasks.
constexpr int kWarps = 4;
// Pairs each thread groups, and the warps of 32 pairs, for `m = 9..32`.
constexpr int kPairChunks = kMaxPairs / (kWarps * 32);
constexpr int kPairWarps = kMaxPairs / 32;
// Bytes of each weight row per stage: 128, or 64 for NVFP4's w2, whose rows are 576 bytes.
constexpr int kStageBytes = 128;
constexpr int kK13 = 2304;
constexpr int kN13 = 2304;
constexpr int kK2 = 1152;
constexpr int kN2 = 2304;
// 16-row tiles of either projection.
constexpr int kTiles = 144;
static_assert(kTiles * 16 == kN13 && kTiles * 16 == kN2, "whole tiles");
static_assert(kPairChunks * kWarps * 32 == kMaxPairs, "threads cover the pairs");
static_assert(kWarps * 32 == kExperts, "a thread per expert ordinal");
static_assert(kSingleMaxRows * 8 <= 2 * 32, "single-group pairs fit in two warps");
// `F8E4M3_MAX * F4E2M1_MAX` in MLX's `inv_scale_enc`.
constexpr float kNvfp4ScaleEnc = 448.f * 6.f;

// Stage geometry of a projection with input width K.
template <bool kFp4, int K, bool kScale4 = false>
struct Geometry {
    static constexpr int kRowBytes = kFp4 ? K / 2 : K;
    static constexpr int kStageBytes = kRowBytes % ::kStageBytes == 0 ? ::kStageBytes : 64;
    // NVFP4: 9 stages per row for both projections; MXFP8: 18 for w13 and 9 for w2.
    static constexpr int kStages = kRowBytes / kStageBytes;
    // 16-byte chunks per row and stage.
    static constexpr int kChunks = kStageBytes / 16;
    static constexpr int kKPerStage = K / kStages;
    // Values per chunk.
    static constexpr int kValues = kFp4 ? 32 : 16;
    static constexpr int kGroup = kFp4 ? 16 : 32;
    static constexpr int kRowScales = K / kGroup;
    // Block-scale bytes per row and stage: 16 for NVFP4's w13, 8 for its w2, 4 for MXFP8
    // (halved to 8 and 4 with 4-bit offset scales).
    static constexpr int kStageScales = kKPerStage / kGroup;
    static constexpr int kRowScaleBytes = kScale4 ? kRowScales / 2 : kRowScales;
    static constexpr int kStageScaleBytes = kScale4 ? kStageScales / 2 : kStageScales;
    static_assert(kStageBytes * kStages == kRowBytes && kChunks % 4 == 0, "whole stages");
    static_assert(!kScale4 || (kFp4 && kStageScaleBytes >= 4 && kStageScaleBytes % 4 == 0),
                  "4-bit offset scales align to cp.async");
};

// 16-byte vectors per ring slot: 16 rows of the larger stage, w13's, then 16 bytes of block
// scales per row.
constexpr int kSlotWeightVecs = 16 * kStageBytes / 16;
constexpr int kSlotVecs = kSlotWeightVecs + 16;

__device__ __forceinline__ uint64_t evict_first_policy() {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

// 16 weight bytes to shared memory, past L1, first out of L2: expert weights are read once.
__device__ __forceinline__ void copy16(void* dst, const void* src, uint64_t policy) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                 "l"(src), "l"(policy)
                 : "memory");
}

// N = 4, 8 or 16 bytes of block scales to shared memory, first out of L2.
template <int N>
__device__ __forceinline__ void copy_scales(void* dst, const void* src, uint64_t policy) {
    asm volatile("cp.async.ca.shared.global.L2::cache_hint [%0], [%1], %2, %3;" ::"r"(
                     static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                 "l"(src), "n"(N), "l"(policy)
                 : "memory");
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait_copies() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ uint32_t load_relaxed(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint32_t load_acquire(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void fence_acq_rel_gpu() {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ void atomic_inc_release(uint32_t* p) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}

// Wait for all `kTiles` w13 tiles of a group, then order subsequent activation loads
// after the stores those tiles cover (`__threadfence()` emits `MEMBAR.SC.GPU` without
// `ld.acquire.gpu`'s L1-cache `CCTL.IVALL` wipe on sm_120a).
__device__ __forceinline__ void wait_sync_tiles(const uint32_t* counter, int lane) {
    if (lane == 0) {
        while (load_relaxed(counter) < kTiles) {
            __nanosleep(64);
        }
        __threadfence();
    }
    __syncwarp();
}

// Non-returning GPU-scope atomic increment (`REDG.E.ADD.STRONG.GPU`), called by lane 0
// after all lanes' `__threadfence()` + `__syncwarp()`.
__device__ __forceinline__ void atomic_inc_relaxed(uint32_t* p) {
    asm volatile("red.relaxed.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                    uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Two F16 values whose low 7 mantissa bits are zero, as E2M1 and E4M3 values are, as two
// BF16 values 2^-112 times as large: the exponent and mantissa move right by 3 bits.
__device__ __forceinline__ uint32_t f16x2_to_scaled_bf16x2(uint32_t u) {
    return ((u & 0x7fff7fffu) >> 3) | (u & 0x80008000u);
}

// a * b on two BF16 pairs; exact for the weight and scale products here.
__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a, uint32_t b) {
    __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a),
                               *reinterpret_cast<__nv_bfloat162*>(&b));
    return *reinterpret_cast<uint32_t*>(&r);
}

// Two E2M1 values (one byte, low nibble first) as F16, exactly.
__device__ __forceinline__ uint32_t e2m1x2_to_f16x2(uint32_t byte) {
#if SIPHON_SM89
    return e2m1x2_to_f16x2_sm89(byte);
#else
    const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(uint8_t(byte), __NV_E2M1);
    return uint32_t(h.x) | (uint32_t(h.y) << 16);
#endif
}

// Two E4M3 values (low byte first) as F16, exactly.
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t bits) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(bits), __NV_E4M3);
    return uint32_t(h.x) | (uint32_t(h.y) << 16);
}

// Two E4M3 scales (low byte, high byte) each multiplied by 2^112 and duplicated into a
// BF16x2 pair: bit-identical to `scale_pair(e4m3(...))` for all 254 finite E4M3 values
// (including +/-0 and all subnormals). `0x7780` is BF16 2^112, so two exact BF16 muls lift
// `scale * 2^-112` to `scale * 2^+112` while keeping zeros exact.
__device__ __forceinline__ void e4m3x2_scale_pairs(uint32_t bits, uint32_t& lo_pair,
                                                   uint32_t& hi_pair) {
    const uint32_t scaled = f16x2_to_scaled_bf16x2(e4m3x2_to_f16x2(bits));
    const uint32_t lifted = bf16x2_mul(bf16x2_mul(scaled, 0x77807780u), 0x77807780u);
    const uint32_t lo = lifted & 0xffffu;
    const uint32_t hi = lifted >> 16;
    lo_pair = lo | (lo << 16);
    hi_pair = hi | (hi << 16);
}

// MLX's `fp8_e8m0` (`2^(bits - 127)`, with 0 mapped to `2^-127`) times `2^112` duplicated
// into a BF16x2 pair: exact for all 256 E8M0 byte values with zero FP conversions.
__device__ __forceinline__ uint32_t e8m0_scale_pair(uint32_t bits) {
    const uint32_t exp = bits == 0 ? 112u : min(bits + 112u, 255u);
    const uint32_t b = exp << 7;
    return b | (b << 16);
}

// Reconstructs two E4M3 block scales (packed in uint16_t) from one packed byte of 4-bit
// offsets subtracted from the row's E4M3 base scale (`base >= off_i` by construction).
__device__ __forceinline__ uint32_t unpack_scale4_pair(uint32_t packed_byte, uint32_t base) {
    const uint32_t s0 = base - (packed_byte & 0xfu);
    const uint32_t s1 = base - ((packed_byte >> 4) & 0xfu);
    return s0 | (s1 << 8);
}

// Chunk c of tile row r within a slot of kChunks chunks per row. With 8 or more chunks,
// rows of odd index swap the 64-byte halves of each 128 bytes, so the 8 lanes of a
// shared-memory phase, which read rows g and g + 1, hit distinct banks. With 4 chunks the
// two rows already fill one 128-byte line.
template <int kChunks>
__device__ __forceinline__ int slot_index(int r, int c) {
    if constexpr (kChunks >= 8) {
        return r * kChunks + (c ^ ((r & 1) << 2));
    } else {
        return r * kChunks + c;
    }
}

// The product of weight rows `row0 .. row0 + 16` of one expert (`w`, `s` at the expert's
// first row) with the input of this lane's column, `x` (K values, or null for no pair),
// into the accumulators of the m16n8 fragment: acc[0], acc[1] are row g and columns 2t,
// 2t + 1; acc[2], acc[3] are row g + 8. `ready()` runs after the first weight copies are
// issued and before anything reads `x`.
template <bool kFp4, int K, int kRing, bool kScale4 = false, typename Ready>
__device__ __forceinline__ void tile_product(const uint8_t* w, const uint8_t* s,
                                             const uint8_t* s_row_base, int row0,
                                             const __nv_bfloat16* x, uint4* ring, int lane,
                                             uint64_t policy, float* acc, Ready ready) {
    using G = Geometry<kFp4, K, kScale4>;
    constexpr int kJ = G::kChunks / 4;
    constexpr int kXVecs = G::kValues / 8;
    const int g = lane / 4;
    const int t = lane % 4;
    const uint8_t* const w_base = w + size_t(row0) * G::kRowBytes;
    const uint8_t* const s_base = s + size_t(row0) * G::kRowScaleBytes;
    uint32_t base_a = 0u;
    uint32_t base_b = 0u;
    if constexpr (kScale4) {
        base_a = s_row_base[row0 + g];
        base_b = s_row_base[row0 + g + 8];
    }

    auto issue = [&](int i) {
        if (i < G::kStages) {
            uint4* slot = ring + i % kRing * kSlotVecs;
            const uint8_t* const ws = w_base + i * G::kStageBytes;
#pragma unroll
            for (int q = 0; q < 16 * G::kChunks / 32; ++q) {
                const int idx = q * 32 + lane;
                const int r = idx / G::kChunks;
                const int c = idx % G::kChunks;
                copy16(slot + slot_index<G::kChunks>(r, c), ws + r * G::kRowBytes + c * 16, policy);
            }
            // The stage's block scales ride in the same group, 16 bytes apart per row.
            if (lane < 16) {
                copy_scales<G::kStageScaleBytes>(
                    slot + kSlotWeightVecs + lane,
                    s_base + lane * G::kRowScaleBytes + i * G::kStageScaleBytes, policy);
            }
        }
        commit();
    };
#pragma unroll
    for (int q = 0; q < kRing - 1; ++q) {
        issue(q);
    }
    ready();
    for (int i = 0; i < G::kStages; ++i) {
        issue(i + kRing - 1);
        // The stage's inputs load from L2 while its weights arrive. Keep this predicated
        // rather than branching so lanes with `g >= count` do not diverge from active lanes.
        uint4 xv[kJ][kXVecs];
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int k = i * G::kKPerStage + (4 * j + t) * G::kValues;
#pragma unroll
            for (int h = 0; h < kXVecs; ++h) {
                xv[j][h] = x != nullptr ? reinterpret_cast<const uint4*>(x + k)[h]
                                        : make_uint4(0, 0, 0, 0);
            }
        }
        wait_copies<kRing - 1>();
        __syncwarp();
        const uint4* slot = ring + i % kRing * kSlotVecs;
        const auto* scales = reinterpret_cast<const uint8_t*>(slot + kSlotWeightVecs);
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int c = 4 * j + t;
            const int sc = c * G::kValues / G::kGroup;
            const uint32_t sa = kScale4
                                    ? unpack_scale4_pair(scales[g * 16 + c], base_a)
                                    : (kFp4 ? *reinterpret_cast<const uint16_t*>(scales + g * 16 + sc)
                                            : scales[g * 16 + sc]);
            const uint32_t sb =
                kScale4 ? unpack_scale4_pair(scales[(g + 8) * 16 + c], base_b)
                        : (kFp4 ? *reinterpret_cast<const uint16_t*>(scales + (g + 8) * 16 + sc)
                                : scales[(g + 8) * 16 + sc]);
            const uint4 av = slot[slot_index<G::kChunks>(g, c)];
            const uint4 bv = slot[slot_index<G::kChunks>(g + 8, c)];
            const uint32_t aw[4] = {av.x, av.y, av.z, av.w};
            const uint32_t bw[4] = {bv.x, bv.y, bv.z, bv.w};
            uint32_t xw[G::kValues / 2];
#pragma unroll
            for (int h = 0; h < kXVecs; ++h) {
                xw[4 * h] = xv[j][h].x;
                xw[4 * h + 1] = xv[j][h].y;
                xw[4 * h + 2] = xv[j][h].z;
                xw[4 * h + 3] = xv[j][h].w;
            }
            // The chunk's weights times their scales as BF16 pairs, in k order.
            uint32_t a[G::kValues / 2];
            uint32_t b[G::kValues / 2];
            if constexpr (kFp4) {
                uint32_t scale_a[2];
                uint32_t scale_b[2];
                e4m3x2_scale_pairs(sa, scale_a[0], scale_a[1]);
                e4m3x2_scale_pairs(sb, scale_b[0], scale_b[1]);
#pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        a[4 * wi + byte] = bf16x2_mul(
                            f16x2_to_scaled_bf16x2(e2m1x2_to_f16x2((aw[wi] >> (8 * byte)) & 0xff)),
                            scale_a[wi / 2]);
                        b[4 * wi + byte] = bf16x2_mul(
                            f16x2_to_scaled_bf16x2(e2m1x2_to_f16x2((bw[wi] >> (8 * byte)) & 0xff)),
                            scale_b[wi / 2]);
                    }
                }
            } else {
                const uint32_t scale_a = e8m0_scale_pair(sa);
                const uint32_t scale_b = e8m0_scale_pair(sb);
#pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        a[2 * wi + h] = bf16x2_mul(
                            f16x2_to_scaled_bf16x2(e4m3x2_to_f16x2((aw[wi] >> (16 * h)) & 0xffff)),
                            scale_a);
                        b[2 * wi + h] = bf16x2_mul(
                            f16x2_to_scaled_bf16x2(e4m3x2_to_f16x2((bw[wi] >> (16 * h)) & 0xffff)),
                            scale_b);
                    }
                }
            }
#pragma unroll
            for (int q = 0; q < G::kValues / 4; ++q) {
                mma(acc, a[2 * q], b[2 * q], a[2 * q + 1], b[2 * q + 1], xw[2 * q], xw[2 * q + 1]);
            }
        }
        // Every lane has read the slot before a later stage overwrites it.
        __syncwarp();
    }
}

// Multi-group form of `tile_product` for `expert_mlp_rows_wide`: processes `ng` (1..=4)
// column groups of 8 pairs (up to 32 pairs) against a single DRAM load and dequantization
// of the 16-row weight tile. Each group runs the exact same mma.sync instructions in the
// same k order as `tile_product`, keeping every output bit-identical. `ready()` runs after
// the first weight copies are issued and before anything reads `x_ptrs`.
template <bool kFp4, int K, int kRing, bool kScale4 = false, typename Ready>
__device__ __forceinline__ void tile_product_wide(const uint8_t* w, const uint8_t* s,
                                                  const uint8_t* s_row_base, int row0,
                                                  const __nv_bfloat16* const x_ptrs[4], int ng,
                                                  uint4* ring, int lane, uint64_t policy,
                                                  float acc[4][4], Ready ready) {
    using G = Geometry<kFp4, K, kScale4>;
    constexpr int kJ = G::kChunks / 4;
    constexpr int kXVecs = G::kValues / 8;
    const int g = lane / 4;
    const int t = lane % 4;
    uint32_t base_a = 0u;
    uint32_t base_b = 0u;
    if constexpr (kScale4) {
        base_a = s_row_base[row0 + g];
        base_b = s_row_base[row0 + g + 8];
    }

    auto issue = [&](int i) {
        if (i < G::kStages) {
            uint4* slot = ring + i % kRing * kSlotVecs;
#pragma unroll
            for (int q = 0; q < 16 * G::kChunks / 32; ++q) {
                const int idx = q * 32 + lane;
                const int r = idx / G::kChunks;
                const int c = idx % G::kChunks;
                copy16(slot + slot_index<G::kChunks>(r, c),
                       w + size_t(row0 + r) * G::kRowBytes + i * G::kStageBytes + c * 16, policy);
            }
            if (lane < 16) {
                copy_scales<G::kStageScaleBytes>(
                    slot + kSlotWeightVecs + lane,
                    s + size_t(row0 + lane) * G::kRowScaleBytes + i * G::kStageScaleBytes, policy);
            }
        }
        commit();
    };
#pragma unroll
    for (int q = 0; q < kRing - 1; ++q) {
        issue(q);
    }
    ready();
    for (int i = 0; i < G::kStages; ++i) {
        issue(i + kRing - 1);
        uint4 xv0[kJ][kXVecs];
        const __nv_bfloat16* const x0 = x_ptrs[0];
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int k = i * G::kKPerStage + (4 * j + t) * G::kValues;
#pragma unroll
            for (int h = 0; h < kXVecs; ++h) {
                xv0[j][h] = x0 != nullptr ? reinterpret_cast<const uint4*>(x0 + k)[h]
                                          : make_uint4(0, 0, 0, 0);
            }
        }
        wait_copies<kRing - 1>();
        __syncwarp();
        const uint4* slot = ring + i % kRing * kSlotVecs;
        const auto* scales = reinterpret_cast<const uint8_t*>(slot + kSlotWeightVecs);
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int c = 4 * j + t;
            const int k = i * G::kKPerStage + c * G::kValues;
            const int sc = c * G::kValues / G::kGroup;
            const uint32_t sa = kScale4
                                    ? unpack_scale4_pair(scales[g * 16 + c], base_a)
                                    : (kFp4 ? *reinterpret_cast<const uint16_t*>(scales + g * 16 + sc)
                                            : scales[g * 16 + sc]);
            const uint32_t sb =
                kScale4 ? unpack_scale4_pair(scales[(g + 8) * 16 + c], base_b)
                        : (kFp4 ? *reinterpret_cast<const uint16_t*>(scales + (g + 8) * 16 + sc)
                                : scales[(g + 8) * 16 + sc]);
            const uint4 av = slot[slot_index<G::kChunks>(g, c)];
            const uint4 bv = slot[slot_index<G::kChunks>(g + 8, c)];
            const uint32_t aw[4] = {av.x, av.y, av.z, av.w};
            const uint32_t bw[4] = {bv.x, bv.y, bv.z, bv.w};
            uint32_t a[G::kValues / 2];
            uint32_t b[G::kValues / 2];
            if constexpr (kFp4) {
                uint32_t scale_a[2];
                uint32_t scale_b[2];
                e4m3x2_scale_pairs(sa, scale_a[0], scale_a[1]);
                e4m3x2_scale_pairs(sb, scale_b[0], scale_b[1]);
#pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
                    for (int byte = 0; byte < 4; ++byte) {
                        a[4 * wi + byte] = bf16x2_mul(
                            f16x2_to_scaled_bf16x2(e2m1x2_to_f16x2((aw[wi] >> (8 * byte)) & 0xff)),
                            scale_a[wi / 2]);
                        b[4 * wi + byte] = bf16x2_mul(
                            f16x2_to_scaled_bf16x2(e2m1x2_to_f16x2((bw[wi] >> (8 * byte)) & 0xff)),
                            scale_b[wi / 2]);
                    }
                }
            } else {
                const uint32_t scale_a = e8m0_scale_pair(sa);
                const uint32_t scale_b = e8m0_scale_pair(sb);
#pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        a[2 * wi + h] = bf16x2_mul(
                            f16x2_to_scaled_bf16x2(e4m3x2_to_f16x2((aw[wi] >> (16 * h)) & 0xffff)),
                            scale_a);
                        b[2 * wi + h] = bf16x2_mul(
                            f16x2_to_scaled_bf16x2(e4m3x2_to_f16x2((bw[wi] >> (16 * h)) & 0xffff)),
                            scale_b);
                    }
                }
            }
            {
                uint32_t xw[G::kValues / 2];
#pragma unroll
                for (int h = 0; h < kXVecs; ++h) {
                    xw[4 * h] = xv0[j][h].x;
                    xw[4 * h + 1] = xv0[j][h].y;
                    xw[4 * h + 2] = xv0[j][h].z;
                    xw[4 * h + 3] = xv0[j][h].w;
                }
#pragma unroll
                for (int q = 0; q < G::kValues / 4; ++q) {
                    mma(acc[0], a[2 * q], b[2 * q], a[2 * q + 1], b[2 * q + 1], xw[2 * q],
                        xw[2 * q + 1]);
                }
            }
            if (ng > 1) {
#pragma unroll
                for (int u = 1; u < 4; ++u) {
                    if (u < ng) {
                        const __nv_bfloat16* const xu = x_ptrs[u];
                        uint32_t xw[G::kValues / 2];
#pragma unroll
                        for (int h = 0; h < kXVecs; ++h) {
                            const uint4 v = xu != nullptr ? reinterpret_cast<const uint4*>(xu + k)[h]
                                                          : make_uint4(0, 0, 0, 0);
                            xw[4 * h] = v.x;
                            xw[4 * h + 1] = v.y;
                            xw[4 * h + 2] = v.z;
                            xw[4 * h + 3] = v.w;
                        }
#pragma unroll
                        for (int q = 0; q < G::kValues / 4; ++q) {
                            mma(acc[u], a[2 * q], b[2 * q], a[2 * q + 1], b[2 * q + 1], xw[2 * q],
                                xw[2 * q + 1]);
                        }
                    }
                }
            }
        }
        __syncwarp();
    }
    wait_copies<0>();
    __syncwarp();
}

// Zero-smem warp-shuffle coalesced w13 SwiGLU epilogue for 1..=8 pairs: gathers each
// pair's 8 BF16 activations into one 16-byte `uint4` global store via 4 warp shuffles.
__device__ __forceinline__ void store_act_single(const float acc[4], float scale, int count,
                                                 const uint8_t* pairs_of, int row0,
                                                 float gate_softcap, float up_softcap,
                                                 __nv_bfloat16* act, int lane, int g, int t,
                                                 uint32_t* done_counter) {
    uint16_t act_val[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float gate = acc[i] * scale;
        const float up = __shfl_down_sync(0xffffffffu, gate, 4);
        const int column = 2 * t + (i & 1);
        if ((g & 1) == 0 && column < count) {
            act_val[i] = __bfloat16_as_ushort(softcapped_swiglu(
                __bfloat162float(__float2bfloat16(gate)),
                __bfloat162float(__float2bfloat16(up)), gate_softcap, up_softcap));
        }
    }
    const uint32_t e04 = uint32_t(act_val[0]) | (uint32_t(act_val[2]) << 16);
    const uint32_t o04 = uint32_t(act_val[1]) | (uint32_t(act_val[3]) << 16);
    const uint32_t e_hi = __shfl_down_sync(0xffffffffu, e04, 8);
    const uint32_t o_hi = __shfl_down_sync(0xffffffffu, o04, 8);
    const uint32_t e_lo16 = (e04 & 0xffffu) | ((e_hi & 0xffffu) << 16);
    const uint32_t e_hi16 = (e04 >> 16) | (e_hi & 0xffff0000u);
    const uint32_t o_lo16 = (o04 & 0xffffu) | ((o_hi & 0xffffu) << 16);
    const uint32_t o_hi16 = (o04 >> 16) | (o_hi & 0xffff0000u);
    const uint32_t peer_lo = __shfl_xor_sync(0xffffffffu, (g == 0) ? o_lo16 : e_lo16, 16);
    const uint32_t peer_hi = __shfl_xor_sync(0xffffffffu, (g == 0) ? o_hi16 : e_hi16, 16);
    if (g == 0 || g == 4) {
        const int col = 2 * t + (g >> 2);
        if (col < count) {
            const uint4 packed = (g == 0) ? make_uint4(e_lo16, peer_lo, e_hi16, peer_hi)
                                          : make_uint4(peer_lo, o_lo16, peer_hi, o_hi16);
            const int pair = int(pairs_of[col]);
            *reinterpret_cast<uint4*>(act + size_t(pair) * kK2 + row0 / 2) = packed;
        }
    }
    __threadfence();
    __syncwarp();
    if (lane == 0) {
        atomic_inc_relaxed(done_counter);
    }
}

// Zero-smem warp-shuffle coalesced w2 output epilogue for 1..=8 pairs: gathers each
// pair's 16 BF16 outputs into two contiguous 16-byte `uint4` global stores (one 32-byte
// sector per pair) via 6 warp shuffles.
__device__ __forceinline__ void store_out_single(const float acc[4], float scale, int count,
                                                 const uint8_t* pairs_of, int row0,
                                                 __nv_bfloat16* out, int g, int t) {
    const uint32_t e_lo = uint32_t(__bfloat16_as_ushort(__float2bfloat16(acc[0] * scale)));
    const uint32_t o_lo = uint32_t(__bfloat16_as_ushort(__float2bfloat16(acc[1] * scale)));
    const uint32_t e_hi = uint32_t(__bfloat16_as_ushort(__float2bfloat16(acc[2] * scale)));
    const uint32_t o_hi = uint32_t(__bfloat16_as_ushort(__float2bfloat16(acc[3] * scale)));
    const uint32_t e08 = e_lo | (e_hi << 16);
    const uint32_t o08 = o_lo | (o_hi << 16);
    const uint32_t e_n = __shfl_down_sync(0xffffffffu, e08, 4);
    const uint32_t o_n = __shfl_down_sync(0xffffffffu, o08, 4);
    const uint32_t e_plo = (e08 & 0xffffu) | ((e_n & 0xffffu) << 16);
    const uint32_t e_phi = (e08 >> 16) | (e_n & 0xffff0000u);
    const uint32_t o_plo = (o08 & 0xffffu) | ((o_n & 0xffffu) << 16);
    const uint32_t o_phi = (o08 >> 16) | (o_n & 0xffff0000u);
    const bool keep_e = (g & 2) == 0;
    const uint32_t p_lo = __shfl_xor_sync(0xffffffffu, keep_e ? o_plo : e_plo, 8);
    const uint32_t p_hi = __shfl_xor_sync(0xffffffffu, keep_e ? o_phi : e_phi, 8);
    const uint32_t a0 = keep_e ? e_plo : p_lo;
    const uint32_t a1 = keep_e ? p_lo : o_plo;
    const uint32_t b0 = keep_e ? e_phi : p_hi;
    const uint32_t b1 = keep_e ? p_hi : o_phi;
    const bool lo_half = (g & 4) == 0;
    const uint32_t q0 = __shfl_xor_sync(0xffffffffu, lo_half ? b0 : a0, 16);
    const uint32_t q1 = __shfl_xor_sync(0xffffffffu, lo_half ? b1 : a1, 16);
    if ((g & 1) == 0) {
        const int col = 2 * t + ((g >> 1) & 1);
        if (col < count) {
            const uint4 packed =
                lo_half ? make_uint4(a0, a1, q0, q1) : make_uint4(q0, q1, b0, b1);
            const int pair = int(pairs_of[col]);
            *reinterpret_cast<uint4*>(out + size_t(pair) * kN2 + row0 + (lo_half ? 0 : 8)) =
                packed;
        }
    }
}

// Tier 1 (`m = 1..8`): every expert has at most 8 pairs (one column group).
template <bool kFp4, int kRing, bool kScale4 = false>
__device__ __forceinline__ void expert_mlp_rows(const __nv_bfloat16* x, const int32_t* indices,
                                                int m, const uint8_t* w13, const uint8_t* s13,
                                                const uint8_t* s13_base, const float* g13,
                                                const uint8_t* w2, const uint8_t* s2,
                                                const uint8_t* s2_base, const float* g2,
                                                __nv_bfloat16* act, __nv_bfloat16* out,
                                                float gate_softcap, float up_softcap,
                                                uint32_t* sync, uint4* ring_all) {
    using G13 = Geometry<kFp4, kK13, kScale4>;
    using G2 = Geometry<kFp4, kK2, kScale4>;
    __shared__ uint32_t counts_u32[kExperts];
    __shared__ unsigned present[4];
    __shared__ uint8_t expert_id[kExperts];
    __shared__ uint8_t expert_count[kExperts];
    // Indexed directly by expert id `e`: `pair_list[8 * e + 0..count-1]`.
    __shared__ uint8_t pair_list[kExperts * 8];
    __shared__ int distinct_s;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int pairs = m * 8;
    const unsigned below = (1u << lane) - 1;

    // Zero shared metadata before waiting on the router so it overlaps PDL startup.
    if (m > 1) {
        counts_u32[tid] = 0;
        if (tid < 4) {
            present[tid] = 0;
        }
    }

    // The expert ids come from the router just before; nothing loads before the wait.
    pdl_wait();
    pdl_trigger();
    if (m == 1) {
        // One row routes to up to 8 distinct experts (pruned slots carry -1): 1 barrier,
        // zero smem atomics or bitmasks.
        if (warp == 0) {
            const int e = lane < 8 ? indices[lane] : -1;
            const bool valid_e = e >= 0 && e < kExperts;
            const unsigned valid = __ballot_sync(0xffffffffu, valid_e);
            if (valid_e) {
                const int ord = __popc(valid & below);
                expert_id[ord] = uint8_t(e);
                expert_count[ord] = 1;
                pair_list[8 * e] = uint8_t(lane);
            }
            if (lane == 0) {
                distinct_s = __popc(valid);
            }
        }
        __syncthreads();
    } else {
        const int pe = tid < pairs ? indices[tid] : -1;
        __syncthreads();

        // Only warps with `warp * 32 < pairs` hold pairs (`m <= 8` => at most 2 warps).
        if (warp * 32 < pairs) {
            const unsigned same = __match_any_sync(0xffffffffu, pe);
            const int leader = __ffs(same) - 1;
            uint32_t col_base = 0;
            if (pe >= 0 && lane == leader) {
                col_base = atomicAdd(&counts_u32[pe], uint32_t(__popc(same)));
                atomicOr(&present[pe >> 5], 1u << (pe & 31));
            }
            col_base = __shfl_sync(0xffffffffu, col_base, leader);
            if (pe >= 0) {
                pair_list[8 * pe + col_base + __popc(same & below)] = uint8_t(tid);
            }
        }
        __syncthreads();

        const uint32_t cnt = counts_u32[tid];
        if (cnt > 0) {
            int ord = __popc(present[warp] & below);
            for (int w = 0; w < warp; ++w) {
                ord += __popc(present[w]);
            }
            expert_id[ord] = uint8_t(tid);
            expert_count[ord] = uint8_t(cnt);
        }
        if (tid == 0) {
            distinct_s = __popc(present[0]) + __popc(present[1]) + __popc(present[2]) +
                         __popc(present[3]);
        }
        __syncthreads();
    }

    const int distinct = distinct_s;
    const uint64_t policy = evict_first_policy();
    uint4* const ring = ring_all + warp * kRing * kSlotVecs;
    const int tasks13 = distinct * kTiles;
    const int tasks = 2 * tasks13;
    const int warps = gridDim.x * kWarps;

    for (int task = blockIdx.x * kWarps + warp; task < tasks; task += warps) {
        const bool up_proj = task < tasks13;
        const int flat = up_proj ? task : task - tasks13;
        const int ordinal = flat / kTiles;
        const int row0 = (flat % kTiles) * 16;
        const int e = int(expert_id[ordinal]);
        const uint8_t* const pairs_of = &pair_list[8 * e];
        const int count = int(expert_count[ordinal]);
        const int mine = g < count ? int(pairs_of[g]) : -1;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        if (up_proj) {
            tile_product<kFp4, kK13, kRing, kScale4>(
                w13 + size_t(e) * kN13 * G13::kRowBytes,
                s13 + size_t(e) * kN13 * G13::kRowScaleBytes,
                kScale4 ? s13_base + size_t(e) * kN13 : nullptr, row0,
                mine >= 0 ? x + size_t(mine / 8) * kK13 : nullptr, ring, lane, policy, acc, [] {});
            const float scale = kFp4 ? __fdiv_rn(g13[e], kNvfp4ScaleEnc) : 1.f;
            store_act_single(acc, scale, count, pairs_of, row0, gate_softcap, up_softcap, act,
                             lane, g, t, &sync[1 + ordinal]);
        } else {
            const uint8_t* const w2e = w2 + size_t(e) * kN2 * G2::kRowBytes;
            // Overlap the first `kRing - 1` weight stages with any wait for w13 activations.
            tile_product<kFp4, kK2, kRing, kScale4>(
                w2e, s2 + size_t(e) * kN2 * G2::kRowScaleBytes,
                kScale4 ? s2_base + size_t(e) * kN2 : nullptr, row0,
                mine >= 0 ? act + size_t(mine) * kK2 : nullptr, ring, lane, policy, acc,
                [&] { wait_sync_tiles(&sync[1 + ordinal], lane); });
            const float scale = kFp4 ? __fdiv_rn(g2[e], kNvfp4ScaleEnc) : 1.f;
            store_out_single(acc, scale, count, pairs_of, row0, out, g, t);
        }
    }

    __syncthreads();
    if (warp == 0) {
        int last = 0;
        if (lane == 0) {
            last = atomicAdd(&sync[0], 1u) == gridDim.x - 1 ? 1 : 0;
        }
        if (__shfl_sync(0xffffffffu, last, 0)) {
            for (int i = lane; i < distinct; i += 32) {
                sync[1 + i] = 0;
            }
            __syncwarp();
            if (lane == 0) {
                sync[0] = 0;
            }
        }
    }
}

// Tier 2 (`m = 9..32`): deterministic inline shared-memory grouping into 8-pair column
// groups so sibling groups of one tile run on neighboring warps, using the same zero-smem
// warp-shuffle coalesced stores and L2::evict_first streaming policy.
template <bool kFp4, int kRing, bool kScale4 = false>
__device__ __forceinline__ void expert_mlp_rows_multi(
    const __nv_bfloat16* x, const int32_t* indices, int m, const uint8_t* w13, const uint8_t* s13,
    const uint8_t* s13_base, const float* g13, const uint8_t* w2, const uint8_t* s2,
    const uint8_t* s2_base, const float* g2, __nv_bfloat16* act, __nv_bfloat16* out,
    float gate_softcap, float up_softcap, uint32_t* sync, uint4* ring_all) {
    using G13 = Geometry<kFp4, kK13, kScale4>;
    using G2 = Geometry<kFp4, kK2, kScale4>;
    __shared__ uint8_t warp_count[kPairWarps][kExperts];
    __shared__ uint8_t expert_id[kExperts];
    __shared__ uint8_t expert_count[kExperts];
    __shared__ uint16_t pstart[kExperts + 1];
    __shared__ uint16_t vstart[kExperts + 1];
    __shared__ uint16_t pstart_of_e[kExperts];
    __shared__ uint8_t pair_list[kMaxPairs];
    __shared__ int warp_sums[3][kWarps];
    __shared__ int distinct_s;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int pairs = m * 8;
    const int pair_warps = (pairs + 31) / 32;
    const unsigned below = (1u << lane) - 1;

    // Zero warp_count before waiting on the router so it overlaps PDL startup.
#pragma unroll
    for (int i = 0; i < kPairWarps; ++i) {
        warp_count[i][tid] = 0;
    }

    pdl_wait();
    pdl_trigger();
    const int pe0 = tid < pairs ? indices[tid] : -1;
    const int pe1 = (tid + 128) < pairs ? indices[tid + 128] : -1;
    __syncthreads();

    const unsigned same0 = __match_any_sync(0xffffffffu, pe0);
    if (pe0 >= 0 && (same0 & below) == 0) {
        warp_count[warp][pe0] = uint8_t(__popc(same0));
    }
    unsigned same1 = 0;
    if (pairs > 128) {
        same1 = __match_any_sync(0xffffffffu, pe1);
        if (pe1 >= 0 && (same1 & below) == 0) {
            warp_count[kWarps + warp][pe1] = uint8_t(__popc(same1));
        }
    }
    __syncthreads();

    // Thread `tid` owns expert `e = tid`: sum its pair count across active pair warps and
    // run one combined prefix scan for ordinal, group offset, and pair offset.
    int cnt_e = 0;
    for (int i = 0; i < pair_warps; ++i) {
        cnt_e += int(warp_count[i][tid]);
    }
    const int act_e = cnt_e > 0 ? 1 : 0;
    const int grp_e = (cnt_e + 7) >> 3;
    int act_inc = act_e;
    int grp_inc = grp_e;
    int cnt_inc = cnt_e;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int va = __shfl_up_sync(0xffffffffu, act_inc, o);
        const int vg = __shfl_up_sync(0xffffffffu, grp_inc, o);
        const int vc = __shfl_up_sync(0xffffffffu, cnt_inc, o);
        act_inc += lane >= o ? va : 0;
        grp_inc += lane >= o ? vg : 0;
        cnt_inc += lane >= o ? vc : 0;
    }
    if (lane == 31) {
        warp_sums[0][warp] = act_inc;
        warp_sums[1][warp] = grp_inc;
        warp_sums[2][warp] = cnt_inc;
    }
    __syncthreads();

    for (int w = 0; w < warp; ++w) {
        act_inc += warp_sums[0][w];
        grp_inc += warp_sums[1][w];
        cnt_inc += warp_sums[2][w];
    }
    if (act_e) {
        const int ord = act_inc - 1;
        const int p0 = cnt_inc - cnt_e;
        expert_id[ord] = uint8_t(tid);
        expert_count[ord] = uint8_t(cnt_e);
        vstart[ord] = uint16_t(grp_inc - grp_e);
        pstart[ord] = uint16_t(p0);
        pstart_of_e[tid] = uint16_t(p0);
    }
    if (tid == kExperts - 1) {
        distinct_s = act_inc;
        vstart[act_inc] = uint16_t(grp_inc);
        pstart[act_inc] = uint16_t(cnt_inc);
    }
    __syncthreads();

    // Deterministic pair scatter in ascending pair order (identical across all CTAs so every
    // column group maps to the same pairs in w13 and w2).
    if (pe0 >= 0) {
        int col = __popc(same0 & below);
        for (int i = 0; i < warp; ++i) {
            col += int(warp_count[i][pe0]);
        }
        pair_list[int(pstart_of_e[pe0]) + col] = uint8_t(tid);
    }
    if (pairs > 128 && pe1 >= 0) {
        int col = __popc(same1 & below);
        for (int i = 0; i < kWarps + warp; ++i) {
            col += int(warp_count[i][pe1]);
        }
        pair_list[int(pstart_of_e[pe1]) + col] = uint8_t(128 + tid);
    }
    __syncthreads();

    const int distinct = distinct_s;
    const int groups_total = int(vstart[distinct]);
    const uint64_t policy = evict_first_policy();
    uint4* const ring = ring_all + warp * kRing * kSlotVecs;
    const int tasks13 = groups_total * kTiles;
    const int tasks = 2 * tasks13;
    const int warps = gridDim.x * kWarps;

    for (int task = blockIdx.x * kWarps + warp; task < tasks; task += warps) {
        const bool up_proj = task < tasks13;
        const int flat = up_proj ? task : task - tasks13;
        int lo = 0;
        int hi = distinct - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (int(vstart[mid]) * kTiles <= flat) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        const int ordinal = lo;
        const int groups = int(vstart[lo + 1]) - int(vstart[lo]);
        const int local = flat - int(vstart[lo]) * kTiles;
        const int row0 = (local / groups) * 16;
        const int group = int(vstart[lo]) + local % groups;
        const int e = int(expert_id[ordinal]);
        const int col0 = 8 * (group - int(vstart[ordinal]));
        const uint8_t* const pairs_of = &pair_list[int(pstart[ordinal]) + col0];
        const int count = min(int(expert_count[ordinal]) - col0, 8);
        const int mine = g < count ? int(pairs_of[g]) : -1;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        if (up_proj) {
            tile_product<kFp4, kK13, kRing, kScale4>(
                w13 + size_t(e) * kN13 * G13::kRowBytes,
                s13 + size_t(e) * kN13 * G13::kRowScaleBytes,
                kScale4 ? s13_base + size_t(e) * kN13 : nullptr, row0,
                mine >= 0 ? x + size_t(mine / 8) * kK13 : nullptr, ring, lane, policy, acc, [] {});
            const float scale = kFp4 ? __fdiv_rn(g13[e], kNvfp4ScaleEnc) : 1.f;
            store_act_single(acc, scale, count, pairs_of, row0, gate_softcap, up_softcap, act,
                             lane, g, t, &sync[1 + group]);
        } else {
            const uint8_t* const w2e = w2 + size_t(e) * kN2 * G2::kRowBytes;
            tile_product<kFp4, kK2, kRing, kScale4>(
                w2e, s2 + size_t(e) * kN2 * G2::kRowScaleBytes,
                kScale4 ? s2_base + size_t(e) * kN2 : nullptr, row0,
                mine >= 0 ? act + size_t(mine) * kK2 : nullptr, ring, lane, policy, acc,
                [&] { wait_sync_tiles(&sync[1 + group], lane); });
            const float scale = kFp4 ? __fdiv_rn(g2[e], kNvfp4ScaleEnc) : 1.f;
            store_out_single(acc, scale, count, pairs_of, row0, out, g, t);
        }
    }

    __syncthreads();
    if (warp == 0) {
        int last = 0;
        if (lane == 0) {
            last = atomicAdd(&sync[0], 1u) == gridDim.x - 1 ? 1 : 0;
        }
        if (__shfl_sync(0xffffffffu, last, 0)) {
            for (int i = lane; i < groups_total; i += 32) {
                sync[1 + i] = 0;
            }
            __syncwarp();
            if (lane == 0) {
                sync[0] = 0;
            }
        }
    }
}

// Grouped routing metadata written once by `expert_group_rows` for calls with m > 32
// (up to kWideMaxRows = 256 rows, 2048 pairs), stored in global memory right after the
// sync counters and read from L2 by every block of `expert_mlp_rows_wide`.
//
// Experts are bucket-sorted in descending pair-count order (Longest-Processing-Time-first)
// while keeping all chunks of one expert adjacent, so heavy experts start first with full
// L2 weight reuse across sibling chunks and cold 1..8-pair experts drain the tail smoothly.
constexpr int kMaxWideChunks = 256;

struct ExpertChunk {
    uint16_t expert;
    uint16_t pstart;
    uint16_t count;
    uint16_t pad;
};
static_assert(sizeof(ExpertChunk) == 8, "ExpertChunk is 8 bytes");

struct ExpertRowsMeta {
    int32_t num_chunks;
    int32_t pad;
    ExpertChunk chunks[kMaxWideChunks];
    uint16_t pair_list[kWideMaxPairs];
};
static_assert(offsetof(ExpertRowsMeta, chunks) == 8, "chunks offset");
static_assert(offsetof(ExpertRowsMeta, pair_list) == 8 + kMaxWideChunks * 8, "meta header layout");

// Persistent dynamic-queue BF16-dequant wide MLP (bit-exact with `expert_mlp_rows`).
// Uses sync[0] as the dynamic warp work-queue counter, sync[1] as the finished-block counter,
// and sync[2 + chunk] as each 32-pair chunk's w13 tile completion counter.
template <bool kFp4, int kRing, bool kScale4 = false>
__device__ __forceinline__ void expert_mlp_rows_wide(
    const __nv_bfloat16* x, int m, const uint8_t* w13, const uint8_t* s13, const uint8_t* s13_base,
    const float* g13, const uint8_t* w2, const uint8_t* s2, const uint8_t* s2_base,
    const float* g2, __nv_bfloat16* act, __nv_bfloat16* out, float gate_softcap, float up_softcap,
    uint32_t* sync, int sync_end, uint4* ring_all) {
    using G13 = Geometry<kFp4, kK13, kScale4>;
    using G2 = Geometry<kFp4, kK2, kScale4>;
    __shared__ int num_chunks_s;
    __shared__ bool last_block;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int pairs = m * 8;

    // Wait for `expert_group_rows` to finish writing `meta`.
    pdl_wait();
    pdl_trigger();
    const auto* meta = reinterpret_cast<const ExpertRowsMeta*>(sync + sync_end);
    if (tid == 0) {
        num_chunks_s = meta->num_chunks;
    }
    __syncthreads();

    const int num_chunks = num_chunks_s;
    const uint64_t policy = evict_first_policy();
    uint4* ring = ring_all + warp * kRing * kSlotVecs;
    const int tasks13 = num_chunks * kTiles;
    const int tasks = 2 * tasks13;

    for (;;) {
        int task;
        if (lane == 0) {
            task = int(atomicAdd(&sync[0], 1u));
        }
        task = __shfl_sync(0xffffffffu, task, 0);
        if (task >= tasks) {
            break;
        }
        const bool up_proj = task < tasks13;
        const int flat = up_proj ? task : task - tasks13;
        const int chunk = flat / kTiles;
        const int row0 = (flat % kTiles) * 16;
        const ExpertChunk ch = meta->chunks[chunk];
        const int e = int(ch.expert);
        const uint16_t* const pairs_of = &meta->pair_list[ch.pstart];
        const int count = int(ch.count);
        const int ng = (count + 7) / 8;
        const __nv_bfloat16* x_ptrs[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int idx = u * 8 + g;
            if (idx < count) {
                const int pair = int(pairs_of[idx]);
                x_ptrs[u] = up_proj ? (x + size_t(pair / 8) * kK13) : (act + size_t(pair) * kK2);
            } else {
                x_ptrs[u] = nullptr;
            }
        }
        float acc[4][4] = {{0.f, 0.f, 0.f, 0.f},
                           {0.f, 0.f, 0.f, 0.f},
                           {0.f, 0.f, 0.f, 0.f},
                           {0.f, 0.f, 0.f, 0.f}};
        if (up_proj) {
            tile_product_wide<kFp4, kK13, kRing, kScale4>(
                w13 + size_t(e) * kN13 * G13::kRowBytes,
                s13 + size_t(e) * kN13 * G13::kRowScaleBytes,
                kScale4 ? s13_base + size_t(e) * kN13 : nullptr, row0, x_ptrs, ng, ring, lane,
                policy, acc, [] {});
            const float scale = kFp4 ? __fdiv_rn(g13[e], kNvfp4ScaleEnc) : 1.f;
            __nv_bfloat16* act_smem = reinterpret_cast<__nv_bfloat16*>(ring);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (u < ng) {
                    const int col_off = u * 8;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gate = acc[u][i] * scale;
                        const float up = __shfl_down_sync(0xffffffffu, gate, 4);
                        const int column = col_off + 2 * t + (i & 1);
                        if (g % 2 == 0 && column < count) {
                            act_smem[column * 8 + (g >> 1) + (i >> 1) * 4] = softcapped_swiglu(
                                __bfloat162float(__float2bfloat16(gate)),
                                __bfloat162float(__float2bfloat16(up)), gate_softcap, up_softcap);
                        }
                    }
                }
            }
            __syncwarp();
            if (lane < count) {
                const int pair = int(pairs_of[lane]);
                *reinterpret_cast<uint4*>(act + size_t(pair) * kK2 + row0 / 2) =
                    *reinterpret_cast<const uint4*>(act_smem + lane * 8);
            }
            fence_acq_rel_gpu();
            __syncwarp();
            if (lane == 0) {
                atomic_inc_release(&sync[2 + chunk]);
            }
        } else {
            const uint8_t* w2e = w2 + size_t(e) * kN2 * G2::kRowBytes;
            tile_product_wide<kFp4, kK2, kRing, kScale4>(
                w2e, s2 + size_t(e) * kN2 * G2::kRowScaleBytes,
                kScale4 ? s2_base + size_t(e) * kN2 : nullptr, row0, x_ptrs, ng, ring, lane, policy,
                acc, [&] {
                    if (lane == 0) {
                        while (load_acquire(&sync[2 + chunk]) < kTiles) {
                            __nanosleep(64);
                        }
                    }
                    __syncwarp();
                });
            const float scale = kFp4 ? __fdiv_rn(g2[e], kNvfp4ScaleEnc) : 1.f;
            __nv_bfloat16* out_smem = reinterpret_cast<__nv_bfloat16*>(ring);
            constexpr int kWideOutStride = 24;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (u < ng) {
                    const int col_off = u * 8;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int column = col_off + 2 * t + (i & 1);
                        if (column < count) {
                            out_smem[column * kWideOutStride + g + (i >> 1) * 8] =
                                __float2bfloat16(acc[u][i] * scale);
                        }
                    }
                }
            }
            __syncwarp();
            if (lane < count) {
                const int pair = int(pairs_of[lane]);
                __nv_bfloat16* dst = out + size_t(pair) * kN2 + row0;
                const __nv_bfloat16* src = out_smem + lane * kWideOutStride;
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
                *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(src + 8);
            }
            __syncwarp();
        }
    }
    __syncthreads();
    if (tid == 0) {
        last_block = atomicAdd(&sync[1], 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (last_block) {
        for (int i = tid; i < num_chunks; i += kWarps * 32) {
            sync[2 + i] = 0;
        }
        const int meta_words = int(offsetof(ExpertRowsMeta, pair_list) / 4) + (pairs + 1) / 2;
        uint32_t* meta_u32 = sync + sync_end;
        for (int i = tid; i < meta_words; i += kWarps * 32) {
            meta_u32[i] = 0;
        }
        __syncthreads();
        if (tid == 0) {
            sync[0] = 0;
            sync[1] = 0;
        }
    }
}

}  // namespace

// Groups `m * 8` (row, slot) pairs by expert in ascending pair order into `meta`
// (`sync + sync_end`), bucket-sorting active experts in descending pair-count order (LPT)
// and splitting each expert's pairs into chunks of at most `chunk_cap` (32) pairs.
// One block of kWarps * 32 = 128 threads. Reads `indices` once into registers and uses
// double-buffered per-warp counts to cut block barriers by >3x.
extern "C" __global__ void __launch_bounds__(kWarps * 32)
    expert_group_rows(const int32_t* indices, int m, int chunk_cap, uint32_t* sync, int sync_end) {
    static_assert(kWideMaxPairs <= 16 * 128, "register cache holds every pair");
    __shared__ uint32_t counts_u32[kExperts];
    __shared__ uint8_t warp_count[2][kWarps][kExperts];
    __shared__ uint8_t warp_rb_cnt[kWarps][16];
    __shared__ uint16_t bucket_base[16];
    __shared__ uint8_t expert_id[kExperts];
    __shared__ uint16_t expert_count[kExperts];
    __shared__ uint16_t pstart[kExperts + 1];
    __shared__ uint16_t cstart[kExperts + 1];
    __shared__ uint16_t write_pos[kExperts];
    __shared__ int warp_sums[2][kWarps];
    __shared__ int distinct;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int pairs = m * 8;
    const int loops = (pairs + 127) / 128;
    const unsigned below = (1u << lane) - 1;

    pdl_wait();
    pdl_trigger();
    counts_u32[tid] = 0;
    write_pos[tid] = 0;
#pragma unroll
    for (int buf = 0; buf < 2; ++buf) {
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            warp_count[buf][w][tid] = 0;
        }
    }
    __syncthreads();

    uint8_t my_e[16];
    uint8_t my_off[16];
    for (int c = 0; c < loops; ++c) {
        const int p = c * 128 + tid;
        const int e = p < pairs ? indices[p] : -1;
        my_e[c] = e >= 0 ? uint8_t(e) : uint8_t(255);
        const unsigned same = __match_any_sync(0xffffffffu, e);
        my_off[c] = uint8_t(__popc(same & below));
        if (e >= 0 && (same & below) == 0) {
            atomicAdd(&counts_u32[e], uint32_t(__popc(same)));
        }
    }
    __syncthreads();

    // 16-bucket descending LPT sort of active experts: rb = 0 is heaviest (>= 121 pairs),
    // rb = 15 is lightest (1..8 pairs). Ties preserve ascending expert id order.
    const int cnt_e = int(counts_u32[tid]);
    const int my_rb = cnt_e > 0 ? (15 - min((cnt_e - 1) / 8, 15)) : -1;
    int in_warp_rank = 0;
#pragma unroll
    for (int rb = 0; rb < 16; ++rb) {
        const bool match = (cnt_e > 0) && (my_rb == rb);
        const unsigned mask = __ballot_sync(0xffffffffu, match);
        if (lane == 0) {
            warp_rb_cnt[warp][rb] = uint8_t(__popc(mask));
        }
        if (match) {
            in_warp_rank = __popc(mask & below);
        }
    }
    __syncthreads();
    if (warp == 0) {
        const int val = lane < 16 ? (int(warp_rb_cnt[0][lane]) + int(warp_rb_cnt[1][lane]) +
                                     int(warp_rb_cnt[2][lane]) + int(warp_rb_cnt[3][lane]))
                                  : 0;
        int inc = val;
#pragma unroll
        for (int o = 1; o < 16; o <<= 1) {
            const int v = __shfl_up_sync(0xffffffffu, inc, o);
            inc += lane >= o ? v : 0;
        }
        if (lane < 16) {
            bucket_base[lane] = uint16_t(inc - val);
        }
        if (lane == 15) {
            distinct = inc;
        }
    }
    __syncthreads();
    if (cnt_e > 0) {
        int ord = int(bucket_base[my_rb]) + in_warp_rank;
        for (int w = 0; w < warp; ++w) {
            ord += int(warp_rb_cnt[w][my_rb]);
        }
        expert_id[ord] = uint8_t(tid);
        expert_count[ord] = uint16_t(cnt_e);
    }
    __syncthreads();

    const int d = distinct;
    int cnt = tid < d ? int(expert_count[tid]) : 0;
    int nch = cnt > 0 ? (cnt + chunk_cap - 1) / chunk_cap : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int vg = __shfl_up_sync(0xffffffffu, nch, o);
        const int vc = __shfl_up_sync(0xffffffffu, cnt, o);
        nch += lane >= o ? vg : 0;
        cnt += lane >= o ? vc : 0;
    }
    if (lane == 31) {
        warp_sums[0][warp] = nch;
        warp_sums[1][warp] = cnt;
    }
    __syncthreads();
    for (int w = 0; w < warp; ++w) {
        nch += warp_sums[0][w];
        cnt += warp_sums[1][w];
    }
    cstart[tid + 1] = uint16_t(nch);
    pstart[tid + 1] = uint16_t(cnt);
    if (tid == 0) {
        cstart[0] = 0;
        pstart[0] = 0;
    }
    __syncthreads();

    auto* meta = reinterpret_cast<ExpertRowsMeta*>(sync + sync_end);
    if (tid == 0) {
        meta->num_chunks = int32_t(cstart[d]);
        meta->pad = 0;
    }
    if (tid < d) {
        const int e = int(expert_id[tid]);
        const int cnt_o = int(expert_count[tid]);
        const int ps = int(pstart[tid]);
        const int cs = int(cstart[tid]);
        const int ce = int(cstart[tid + 1]);
        write_pos[e] = uint16_t(ps);
        for (int k = 0; k < ce - cs; ++k) {
            ExpertChunk ch;
            ch.expert = uint16_t(e);
            ch.pstart = uint16_t(ps + k * chunk_cap);
            ch.count = uint16_t(min(chunk_cap, cnt_o - k * chunk_cap));
            ch.pad = 0;
            meta->chunks[cs + k] = ch;
        }
    }
    __syncthreads();

    for (int c = 0; c < loops; ++c) {
        const int cur = c & 1;
        const int prev = cur ^ 1;
        const int e = my_e[c] < kExperts ? int(my_e[c]) : -1;
        const unsigned same = __match_any_sync(0xffffffffu, e);
        if (e >= 0 && (same & below) == 0) {
            warp_count[cur][warp][e] = uint8_t(__popc(same));
        }
        __syncthreads();
        if (e >= 0) {
            int off = int(my_off[c]);
            for (int w = 0; w < warp; ++w) {
                off += warp_count[cur][w][e];
            }
            meta->pair_list[write_pos[e] + off] = uint16_t(c * 128 + tid);
        }
        const uint16_t delta =
            uint16_t(warp_count[cur][0][tid] + warp_count[cur][1][tid] + warp_count[cur][2][tid] +
                     warp_count[cur][3][tid]);
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            warp_count[prev][w][tid] = 0;
        }
        __syncthreads();
        write_pos[tid] = uint16_t(write_pos[tid] + delta);
    }
}

// `expert_mlp_rows_<format>_<ring>` (`m = 1..8`), `expert_mlp_rows_multi_<format>_3`
// (`m = 9..32`), and `expert_mlp_rows_wide_<format>_3` (`m = 33..256`).
// Blocks of kWarps warps with kWarps x ring slots of dynamic shared memory; every block of
// the grid must be resident at once.
#define EXPERT_MLP_ROWS(name, fp4, RING, MIN_BLOCKS)                                               \
    extern "C" __global__ void __launch_bounds__(kWarps * 32, MIN_BLOCKS)                          \
        name(const __nv_bfloat16* x, const int32_t* indices, int m, const uint8_t* w13,            \
             const uint8_t* s13, const float* g13, const uint8_t* w2, const uint8_t* s2,           \
             const float* g2, __nv_bfloat16* act, __nv_bfloat16* out, float gate_softcap,          \
             float up_softcap, uint32_t* sync) {                                                   \
        extern __shared__ uint4 ring[];                                                            \
        expert_mlp_rows<fp4, RING, false>(x, indices, m, w13, s13, nullptr, g13, w2, s2, nullptr,  \
                                          g2, act, out, gate_softcap, up_softcap, sync, ring);     \
    }

#define EXPERT_MLP_ROWS_S4(name, RING, MIN_BLOCKS)                                                 \
    extern "C" __global__ void __launch_bounds__(kWarps * 32, MIN_BLOCKS)                          \
        name(const __nv_bfloat16* x, const int32_t* indices, int m, const uint8_t* w13,            \
             const uint8_t* s13, const uint8_t* s13_base, const float* g13, const uint8_t* w2,     \
             const uint8_t* s2, const uint8_t* s2_base, const float* g2, __nv_bfloat16* act,       \
             __nv_bfloat16* out, float gate_softcap, float up_softcap, uint32_t* sync) {           \
        extern __shared__ uint4 ring[];                                                            \
        expert_mlp_rows<true, RING, true>(x, indices, m, w13, s13, s13_base, g13, w2, s2, s2_base, \
                                          g2, act, out, gate_softcap, up_softcap, sync, ring);     \
    }

#define EXPERT_MLP_ROWS_MULTI(name, fp4, RING, MIN_BLOCKS)                                         \
    extern "C" __global__ void __launch_bounds__(kWarps * 32, MIN_BLOCKS)                          \
        name(const __nv_bfloat16* x, const int32_t* indices, int m, const uint8_t* w13,            \
             const uint8_t* s13, const float* g13, const uint8_t* w2, const uint8_t* s2,           \
             const float* g2, __nv_bfloat16* act, __nv_bfloat16* out, float gate_softcap,          \
             float up_softcap, uint32_t* sync) {                                                   \
        extern __shared__ uint4 ring[];                                                            \
        expert_mlp_rows_multi<fp4, RING, false>(x, indices, m, w13, s13, nullptr, g13, w2, s2,     \
                                                nullptr, g2, act, out, gate_softcap, up_softcap,   \
                                                sync, ring);                                       \
    }

#define EXPERT_MLP_ROWS_MULTI_S4(name, RING, MIN_BLOCKS)                                           \
    extern "C" __global__ void __launch_bounds__(kWarps * 32, MIN_BLOCKS)                          \
        name(const __nv_bfloat16* x, const int32_t* indices, int m, const uint8_t* w13,            \
             const uint8_t* s13, const uint8_t* s13_base, const float* g13, const uint8_t* w2,     \
             const uint8_t* s2, const uint8_t* s2_base, const float* g2, __nv_bfloat16* act,       \
             __nv_bfloat16* out, float gate_softcap, float up_softcap, uint32_t* sync) {           \
        extern __shared__ uint4 ring[];                                                            \
        expert_mlp_rows_multi<true, RING, true>(x, indices, m, w13, s13, s13_base, g13, w2, s2,    \
                                                s2_base, g2, act, out, gate_softcap, up_softcap,   \
                                                sync, ring);                                       \
    }

#define EXPERT_MLP_ROWS_WIDE(name, fp4, RING)                                                      \
    extern "C" __global__ void __launch_bounds__(kWarps * 32)                                       \
        name(const __nv_bfloat16* x, int m, const uint8_t* w13, const uint8_t* s13,                \
             const float* g13, const uint8_t* w2, const uint8_t* s2, const float* g2,              \
             __nv_bfloat16* act, __nv_bfloat16* out, float gate_softcap, float up_softcap,         \
             uint32_t* sync, int sync_end) {                                                       \
        extern __shared__ uint4 ring[];                                                            \
        expert_mlp_rows_wide<fp4, RING, false>(x, m, w13, s13, nullptr, g13, w2, s2, nullptr, g2,  \
                                               act, out, gate_softcap, up_softcap, sync, sync_end, \
                                               ring);                                              \
    }

#define EXPERT_MLP_ROWS_WIDE_S4(name, RING)                                                        \
    extern "C" __global__ void __launch_bounds__(kWarps * 32)                                       \
        name(const __nv_bfloat16* x, int m, const uint8_t* w13, const uint8_t* s13,                \
             const uint8_t* s13_base, const float* g13, const uint8_t* w2, const uint8_t* s2,      \
             const uint8_t* s2_base, const float* g2, __nv_bfloat16* act, __nv_bfloat16* out,      \
             float gate_softcap, float up_softcap, uint32_t* sync, int sync_end) {                 \
        extern __shared__ uint4 ring[];                                                            \
        expert_mlp_rows_wide<true, RING, true>(x, m, w13, s13, s13_base, g13, w2, s2, s2_base, g2, \
                                               act, out, gate_softcap, up_softcap, sync, sync_end, \
                                               ring);                                              \
    }

EXPERT_MLP_ROWS(expert_mlp_rows_nvfp4_3, true, 3, 3)
EXPERT_MLP_ROWS(expert_mlp_rows_mxfp8_3, false, 3, 3)
EXPERT_MLP_ROWS_S4(expert_mlp_rows_nvfp4_s4_3, 3, 3)
EXPERT_MLP_ROWS(expert_mlp_rows_nvfp4_4, true, 4, 2)
EXPERT_MLP_ROWS(expert_mlp_rows_mxfp8_4, false, 4, 2)
EXPERT_MLP_ROWS_S4(expert_mlp_rows_nvfp4_s4_4, 4, 2)
EXPERT_MLP_ROWS_MULTI(expert_mlp_rows_multi_nvfp4_3, true, 3, 3)
EXPERT_MLP_ROWS_MULTI(expert_mlp_rows_multi_mxfp8_3, false, 3, 3)
EXPERT_MLP_ROWS_MULTI_S4(expert_mlp_rows_multi_nvfp4_s4_3, 3, 3)
EXPERT_MLP_ROWS_WIDE(expert_mlp_rows_wide_nvfp4_3, true, 3)
EXPERT_MLP_ROWS_WIDE(expert_mlp_rows_wide_mxfp8_3, false, 3)
EXPERT_MLP_ROWS_WIDE_S4(expert_mlp_rows_wide_nvfp4_s4_3, 3)

namespace {

// MXFP6 E2M3 (2-plane lo4 + hi2) ring slot: 16 rows of 128 B lo4 (128 uint4), 16 rows of
// 64 B hi2 (64 uint4), and 16 rows of 16 B UE8M0 scales (16 uint4) = 208 uint4 (3,328 B).
constexpr int kMxfp6SlotLo4Vecs = 128;
constexpr int kMxfp6SlotHi2Vecs = 64;
constexpr int kMxfp6SlotVecs = kMxfp6SlotLo4Vecs + kMxfp6SlotHi2Vecs + 16;

template <int K>
struct GeometryMxfp6 {
    static constexpr int kRowLo4Bytes = K / 2;
    static constexpr int kRowHi2Bytes = K / 4;
    static constexpr int kStageLo4Bytes = kRowLo4Bytes % ::kStageBytes == 0 ? ::kStageBytes : 64;
    static constexpr int kStageHi2Bytes = kStageLo4Bytes / 2;
    static constexpr int kStages = kRowLo4Bytes / kStageLo4Bytes;
    static constexpr int kChunks = kStageLo4Bytes / 16;
    static constexpr int kHi2Chunks = kStageHi2Bytes / 16;
    static constexpr int kKPerStage = K / kStages;
    static constexpr int kValues = 32;
    static constexpr int kRowScaleBytes = K / 32;
    static constexpr int kStageScaleBytes = kKPerStage / 32;
    static_assert(kStageLo4Bytes * kStages == kRowLo4Bytes && kChunks % 4 == 0, "whole stages");
    static_assert(kHi2Chunks * 2 == kChunks && (16 * kHi2Chunks) % 32 == 0, "aligned hi2 copies");
};

// Decodes 32 MXFP6 values (`lo4` uint4 + `hi2` uint2) times UE8M0 scale into 16 BF16x2 pairs.
__device__ __forceinline__ void decode_mxfp6_chunk(const uint4& lv, const uint2& hv,
                                                   uint32_t scale_pair, uint32_t (&out)[16]) {
    const uint32_t lw[4] = {lv.x, lv.y, lv.z, lv.w};
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
        const uint32_t hi16 = ((wi < 2 ? hv.x : hv.y) >> (16 * (wi & 1))) & 0xffffu;
        const uint32_t e0 = mxfp6::e2m3x4_to_e4m3x4(lw[wi] & 0xffffu, hi16 & 0xffu);
        const uint32_t e1 = mxfp6::e2m3x4_to_e4m3x4(lw[wi] >> 16, hi16 >> 8);
        out[4 * wi + 0] =
            bf16x2_mul(f16x2_to_scaled_bf16x2(e4m3x2_to_f16x2(e0 & 0xffffu)), scale_pair);
        out[4 * wi + 1] =
            bf16x2_mul(f16x2_to_scaled_bf16x2(e4m3x2_to_f16x2(e0 >> 16)), scale_pair);
        out[4 * wi + 2] =
            bf16x2_mul(f16x2_to_scaled_bf16x2(e4m3x2_to_f16x2(e1 & 0xffffu)), scale_pair);
        out[4 * wi + 3] =
            bf16x2_mul(f16x2_to_scaled_bf16x2(e4m3x2_to_f16x2(e1 >> 16)), scale_pair);
    }
}

template <int K, int kRing, typename Ready>
__device__ __forceinline__ void tile_product_mxfp6(const uint8_t* w_lo4, const uint8_t* w_hi2,
                                                   const uint8_t* s, int row0,
                                                   const __nv_bfloat16* x, uint4* ring, int lane,
                                                   uint64_t policy, float* acc, Ready ready) {
    using G = GeometryMxfp6<K>;
    constexpr int kJ = G::kChunks / 4;
    constexpr int kXVecs = 4;
    const int g = lane / 4;
    const int t = lane % 4;
    const uint8_t* const lo_base = w_lo4 + size_t(row0) * G::kRowLo4Bytes;
    const uint8_t* const hi_base = w_hi2 + size_t(row0) * G::kRowHi2Bytes;
    const uint8_t* const s_base = s + size_t(row0) * G::kRowScaleBytes;

    auto issue = [&](int i) {
        if (i < G::kStages) {
            uint4* slot = ring + i % kRing * kMxfp6SlotVecs;
            const uint8_t* const ls = lo_base + i * G::kStageLo4Bytes;
            const uint8_t* const hs = hi_base + i * G::kStageHi2Bytes;
#pragma unroll
            for (int q = 0; q < 16 * G::kChunks / 32; ++q) {
                const int idx = q * 32 + lane;
                const int r = idx / G::kChunks;
                const int c = idx % G::kChunks;
                copy16(slot + slot_index<G::kChunks>(r, c), ls + r * G::kRowLo4Bytes + c * 16,
                       policy);
            }
#pragma unroll
            for (int q = 0; q < 16 * G::kHi2Chunks / 32; ++q) {
                const int idx = q * 32 + lane;
                const int r = idx / G::kHi2Chunks;
                const int c = idx % G::kHi2Chunks;
                copy16(slot + kMxfp6SlotLo4Vecs + r * G::kHi2Chunks + c,
                       hs + r * G::kRowHi2Bytes + c * 16, policy);
            }
            if (lane < 16) {
                copy_scales<G::kStageScaleBytes>(
                    slot + kMxfp6SlotLo4Vecs + kMxfp6SlotHi2Vecs + lane,
                    s_base + lane * G::kRowScaleBytes + i * G::kStageScaleBytes, policy);
            }
        }
        commit();
    };
#pragma unroll
    for (int q = 0; q < kRing - 1; ++q) {
        issue(q);
    }
    ready();
    for (int i = 0; i < G::kStages; ++i) {
        issue(i + kRing - 1);
        uint4 xv[kJ][kXVecs];
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int k = i * G::kKPerStage + (4 * j + t) * G::kValues;
#pragma unroll
            for (int h = 0; h < kXVecs; ++h) {
                xv[j][h] = x != nullptr ? reinterpret_cast<const uint4*>(x + k)[h]
                                        : make_uint4(0, 0, 0, 0);
            }
        }
        wait_copies<kRing - 1>();
        __syncwarp();
        const uint4* slot = ring + i % kRing * kMxfp6SlotVecs;
        const uint4* hi_slot = slot + kMxfp6SlotLo4Vecs;
        const auto* scales =
            reinterpret_cast<const uint8_t*>(slot + kMxfp6SlotLo4Vecs + kMxfp6SlotHi2Vecs);
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int c = 4 * j + t;
            const uint32_t scale_a = e8m0_scale_pair(scales[g * 16 + c]);
            const uint32_t scale_b = e8m0_scale_pair(scales[(g + 8) * 16 + c]);
            const uint4 av = slot[slot_index<G::kChunks>(g, c)];
            const uint4 bv = slot[slot_index<G::kChunks>(g + 8, c)];
            const uint2 ah = reinterpret_cast<const uint2*>(hi_slot + g * G::kHi2Chunks)[c];
            const uint2 bh = reinterpret_cast<const uint2*>(hi_slot + (g + 8) * G::kHi2Chunks)[c];
            uint32_t xw[16];
#pragma unroll
            for (int h = 0; h < kXVecs; ++h) {
                xw[4 * h + 0] = xv[j][h].x;
                xw[4 * h + 1] = xv[j][h].y;
                xw[4 * h + 2] = xv[j][h].z;
                xw[4 * h + 3] = xv[j][h].w;
            }
            uint32_t a[16];
            uint32_t b[16];
            decode_mxfp6_chunk(av, ah, scale_a, a);
            decode_mxfp6_chunk(bv, bh, scale_b, b);
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                mma(acc, a[2 * q], b[2 * q], a[2 * q + 1], b[2 * q + 1], xw[2 * q], xw[2 * q + 1]);
            }
        }
        __syncwarp();
    }
}

template <int K, int kRing, typename Ready>
__device__ __forceinline__ void tile_product_wide_mxfp6(
    const uint8_t* w_lo4, const uint8_t* w_hi2, const uint8_t* s, int row0,
    const __nv_bfloat16* const x_ptrs[4], int ng, uint4* ring, int lane, uint64_t policy,
    float acc[4][4], Ready ready) {
    using G = GeometryMxfp6<K>;
    constexpr int kJ = G::kChunks / 4;
    constexpr int kXVecs = 4;
    const int g = lane / 4;
    const int t = lane % 4;
    const uint8_t* const lo_base = w_lo4 + size_t(row0) * G::kRowLo4Bytes;
    const uint8_t* const hi_base = w_hi2 + size_t(row0) * G::kRowHi2Bytes;
    const uint8_t* const s_base = s + size_t(row0) * G::kRowScaleBytes;

    auto issue = [&](int i) {
        if (i < G::kStages) {
            uint4* slot = ring + i % kRing * kMxfp6SlotVecs;
            const uint8_t* const ls = lo_base + i * G::kStageLo4Bytes;
            const uint8_t* const hs = hi_base + i * G::kStageHi2Bytes;
#pragma unroll
            for (int q = 0; q < 16 * G::kChunks / 32; ++q) {
                const int idx = q * 32 + lane;
                const int r = idx / G::kChunks;
                const int c = idx % G::kChunks;
                copy16(slot + slot_index<G::kChunks>(r, c), ls + r * G::kRowLo4Bytes + c * 16,
                       policy);
            }
#pragma unroll
            for (int q = 0; q < 16 * G::kHi2Chunks / 32; ++q) {
                const int idx = q * 32 + lane;
                const int r = idx / G::kHi2Chunks;
                const int c = idx % G::kHi2Chunks;
                copy16(slot + kMxfp6SlotLo4Vecs + r * G::kHi2Chunks + c,
                       hs + r * G::kRowHi2Bytes + c * 16, policy);
            }
            if (lane < 16) {
                copy_scales<G::kStageScaleBytes>(
                    slot + kMxfp6SlotLo4Vecs + kMxfp6SlotHi2Vecs + lane,
                    s_base + lane * G::kRowScaleBytes + i * G::kStageScaleBytes, policy);
            }
        }
        commit();
    };
#pragma unroll
    for (int q = 0; q < kRing - 1; ++q) {
        issue(q);
    }
    ready();
    for (int i = 0; i < G::kStages; ++i) {
        issue(i + kRing - 1);
        uint4 xv0[kJ][kXVecs];
        const __nv_bfloat16* const x0 = x_ptrs[0];
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int k = i * G::kKPerStage + (4 * j + t) * G::kValues;
#pragma unroll
            for (int h = 0; h < kXVecs; ++h) {
                xv0[j][h] = x0 != nullptr ? reinterpret_cast<const uint4*>(x0 + k)[h]
                                          : make_uint4(0, 0, 0, 0);
            }
        }
        wait_copies<kRing - 1>();
        __syncwarp();
        const uint4* slot = ring + i % kRing * kMxfp6SlotVecs;
        const uint4* hi_slot = slot + kMxfp6SlotLo4Vecs;
        const auto* scales =
            reinterpret_cast<const uint8_t*>(slot + kMxfp6SlotLo4Vecs + kMxfp6SlotHi2Vecs);
#pragma unroll
        for (int j = 0; j < kJ; ++j) {
            const int c = 4 * j + t;
            const int k = i * G::kKPerStage + c * G::kValues;
            const uint32_t scale_a = e8m0_scale_pair(scales[g * 16 + c]);
            const uint32_t scale_b = e8m0_scale_pair(scales[(g + 8) * 16 + c]);
            const uint4 av = slot[slot_index<G::kChunks>(g, c)];
            const uint4 bv = slot[slot_index<G::kChunks>(g + 8, c)];
            const uint2 ah = reinterpret_cast<const uint2*>(hi_slot + g * G::kHi2Chunks)[c];
            const uint2 bh = reinterpret_cast<const uint2*>(hi_slot + (g + 8) * G::kHi2Chunks)[c];
            uint32_t a[16];
            uint32_t b[16];
            decode_mxfp6_chunk(av, ah, scale_a, a);
            decode_mxfp6_chunk(bv, bh, scale_b, b);
            {
                uint32_t xw[16];
#pragma unroll
                for (int h = 0; h < kXVecs; ++h) {
                    xw[4 * h + 0] = xv0[j][h].x;
                    xw[4 * h + 1] = xv0[j][h].y;
                    xw[4 * h + 2] = xv0[j][h].z;
                    xw[4 * h + 3] = xv0[j][h].w;
                }
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    mma(acc[0], a[2 * q], b[2 * q], a[2 * q + 1], b[2 * q + 1], xw[2 * q],
                        xw[2 * q + 1]);
                }
            }
            if (ng > 1) {
#pragma unroll
                for (int u = 1; u < 4; ++u) {
                    if (u < ng) {
                        const __nv_bfloat16* const xu = x_ptrs[u];
                        uint32_t xw[16];
#pragma unroll
                        for (int h = 0; h < kXVecs; ++h) {
                            const uint4 v = xu != nullptr
                                                ? reinterpret_cast<const uint4*>(xu + k)[h]
                                                : make_uint4(0, 0, 0, 0);
                            xw[4 * h + 0] = v.x;
                            xw[4 * h + 1] = v.y;
                            xw[4 * h + 2] = v.z;
                            xw[4 * h + 3] = v.w;
                        }
#pragma unroll
                        for (int q = 0; q < 8; ++q) {
                            mma(acc[u], a[2 * q], b[2 * q], a[2 * q + 1], b[2 * q + 1], xw[2 * q],
                                xw[2 * q + 1]);
                        }
                    }
                }
            }
        }
        __syncwarp();
    }
    wait_copies<0>();
    __syncwarp();
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kWarps * 32, 2)
    expert_mlp_rows_mxfp6_3(const __nv_bfloat16* x, const int32_t* indices, int m,
                            const uint8_t* w13_lo4, const uint8_t* w13_hi2, const uint8_t* s13,
                            const uint8_t* w2_lo4, const uint8_t* w2_hi2, const uint8_t* s2,
                            __nv_bfloat16* act, __nv_bfloat16* out, float gate_softcap,
                            float up_softcap, uint32_t* sync) {
    using G13 = GeometryMxfp6<kK13>;
    using G2 = GeometryMxfp6<kK2>;
    constexpr int kRing = 3;
    extern __shared__ uint4 ring_all[];
    __shared__ uint32_t counts_u32[kExperts];
    __shared__ unsigned present[4];
    __shared__ uint8_t expert_id[kExperts];
    __shared__ uint8_t expert_count[kExperts];
    __shared__ uint8_t pair_list[kExperts * 8];
    __shared__ int distinct_s;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int pairs = m * 8;
    const unsigned below = (1u << lane) - 1;

    if (m > 1) {
        counts_u32[tid] = 0;
        if (tid < 4) {
            present[tid] = 0;
        }
    }
    pdl_wait();
    pdl_trigger();
    if (m == 1) {
        if (warp == 0) {
            const int e = lane < 8 ? indices[lane] : -1;
            const bool valid_e = e >= 0 && e < kExperts;
            const unsigned valid = __ballot_sync(0xffffffffu, valid_e);
            if (valid_e) {
                const int ord = __popc(valid & below);
                expert_id[ord] = uint8_t(e);
                expert_count[ord] = 1;
                pair_list[8 * e] = uint8_t(lane);
            }
            if (lane == 0) {
                distinct_s = __popc(valid);
            }
        }
        __syncthreads();
    } else {
        const int pe = tid < pairs ? indices[tid] : -1;
        __syncthreads();
        if (warp * 32 < pairs) {
            const unsigned same = __match_any_sync(0xffffffffu, pe);
            const int leader = __ffs(same) - 1;
            uint32_t col_base = 0;
            if (pe >= 0 && lane == leader) {
                col_base = atomicAdd(&counts_u32[pe], uint32_t(__popc(same)));
                atomicOr(&present[pe >> 5], 1u << (pe & 31));
            }
            col_base = __shfl_sync(0xffffffffu, col_base, leader);
            if (pe >= 0) {
                pair_list[8 * pe + col_base + __popc(same & below)] = uint8_t(tid);
            }
        }
        __syncthreads();
        const uint32_t cnt = counts_u32[tid];
        if (cnt > 0) {
            int ord = __popc(present[warp] & below);
            for (int w = 0; w < warp; ++w) {
                ord += __popc(present[w]);
            }
            expert_id[ord] = uint8_t(tid);
            expert_count[ord] = uint8_t(cnt);
        }
        if (tid == 0) {
            distinct_s = __popc(present[0]) + __popc(present[1]) + __popc(present[2]) +
                         __popc(present[3]);
        }
        __syncthreads();
    }

    const int distinct = distinct_s;
    const uint64_t policy = evict_first_policy();
    uint4* const ring = ring_all + warp * kRing * kMxfp6SlotVecs;
    const int tasks13 = distinct * kTiles;
    const int tasks = 2 * tasks13;
    const int warps = gridDim.x * kWarps;

    for (int task = blockIdx.x * kWarps + warp; task < tasks; task += warps) {
        const bool up_proj = task < tasks13;
        const int flat = up_proj ? task : task - tasks13;
        const int ordinal = flat / kTiles;
        const int row0 = (flat % kTiles) * 16;
        const int e = int(expert_id[ordinal]);
        const uint8_t* const pairs_of = &pair_list[8 * e];
        const int count = int(expert_count[ordinal]);
        const int mine = g < count ? int(pairs_of[g]) : -1;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        if (up_proj) {
            tile_product_mxfp6<kK13, kRing>(
                w13_lo4 + size_t(e) * kN13 * G13::kRowLo4Bytes,
                w13_hi2 + size_t(e) * kN13 * G13::kRowHi2Bytes,
                s13 + size_t(e) * kN13 * G13::kRowScaleBytes, row0,
                mine >= 0 ? x + size_t(mine / 8) * kK13 : nullptr, ring, lane, policy, acc, [] {});
            store_act_single(acc, 1.f, count, pairs_of, row0, gate_softcap, up_softcap, act, lane,
                             g, t, &sync[1 + ordinal]);
        } else {
            tile_product_mxfp6<kK2, kRing>(
                w2_lo4 + size_t(e) * kN2 * G2::kRowLo4Bytes,
                w2_hi2 + size_t(e) * kN2 * G2::kRowHi2Bytes,
                s2 + size_t(e) * kN2 * G2::kRowScaleBytes, row0,
                mine >= 0 ? act + size_t(mine) * kK2 : nullptr, ring, lane, policy, acc,
                [&] { wait_sync_tiles(&sync[1 + ordinal], lane); });
            store_out_single(acc, 1.f, count, pairs_of, row0, out, g, t);
        }
    }

    __syncthreads();
    if (warp == 0) {
        int last = 0;
        if (lane == 0) {
            last = atomicAdd(&sync[0], 1u) == gridDim.x - 1 ? 1 : 0;
        }
        if (__shfl_sync(0xffffffffu, last, 0)) {
            for (int i = lane; i < distinct; i += 32) {
                sync[1 + i] = 0;
            }
            __syncwarp();
            if (lane == 0) {
                sync[0] = 0;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(kWarps * 32, 2)
    expert_mlp_rows_multi_mxfp6_3(const __nv_bfloat16* x, const int32_t* indices, int m,
                                  const uint8_t* w13_lo4, const uint8_t* w13_hi2,
                                  const uint8_t* s13, const uint8_t* w2_lo4, const uint8_t* w2_hi2,
                                  const uint8_t* s2, __nv_bfloat16* act, __nv_bfloat16* out,
                                  float gate_softcap, float up_softcap, uint32_t* sync) {
    using G13 = GeometryMxfp6<kK13>;
    using G2 = GeometryMxfp6<kK2>;
    constexpr int kRing = 3;
    extern __shared__ uint4 ring_all[];
    __shared__ uint8_t warp_count[kPairWarps][kExperts];
    __shared__ uint8_t expert_id[kExperts];
    __shared__ uint8_t expert_count[kExperts];
    __shared__ uint16_t pstart[kExperts + 1];
    __shared__ uint16_t vstart[kExperts + 1];
    __shared__ uint16_t pstart_of_e[kExperts];
    __shared__ uint8_t pair_list[kMaxPairs];
    __shared__ int warp_sums[3][kWarps];
    __shared__ int distinct_s;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int pairs = m * 8;
    const int pair_warps = (pairs + 31) / 32;
    const unsigned below = (1u << lane) - 1;

#pragma unroll
    for (int i = 0; i < kPairWarps; ++i) {
        warp_count[i][tid] = 0;
    }
    pdl_wait();
    pdl_trigger();
    const int pe0 = tid < pairs ? indices[tid] : -1;
    const int pe1 = (tid + 128) < pairs ? indices[tid + 128] : -1;
    __syncthreads();

    const unsigned same0 = __match_any_sync(0xffffffffu, pe0);
    if (pe0 >= 0 && (same0 & below) == 0) {
        warp_count[warp][pe0] = uint8_t(__popc(same0));
    }
    unsigned same1 = 0;
    if (pairs > 128) {
        same1 = __match_any_sync(0xffffffffu, pe1);
        if (pe1 >= 0 && (same1 & below) == 0) {
            warp_count[kWarps + warp][pe1] = uint8_t(__popc(same1));
        }
    }
    __syncthreads();

    int cnt_e = 0;
    for (int i = 0; i < pair_warps; ++i) {
        cnt_e += int(warp_count[i][tid]);
    }
    const int act_e = cnt_e > 0 ? 1 : 0;
    const int grp_e = (cnt_e + 7) >> 3;
    int act_inc = act_e;
    int grp_inc = grp_e;
    int cnt_inc = cnt_e;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int va = __shfl_up_sync(0xffffffffu, act_inc, o);
        const int vg = __shfl_up_sync(0xffffffffu, grp_inc, o);
        const int vc = __shfl_up_sync(0xffffffffu, cnt_inc, o);
        act_inc += lane >= o ? va : 0;
        grp_inc += lane >= o ? vg : 0;
        cnt_inc += lane >= o ? vc : 0;
    }
    if (lane == 31) {
        warp_sums[0][warp] = act_inc;
        warp_sums[1][warp] = grp_inc;
        warp_sums[2][warp] = cnt_inc;
    }
    __syncthreads();
    for (int w = 0; w < warp; ++w) {
        act_inc += warp_sums[0][w];
        grp_inc += warp_sums[1][w];
        cnt_inc += warp_sums[2][w];
    }
    if (act_e) {
        const int ord = act_inc - 1;
        const int p0 = cnt_inc - cnt_e;
        expert_id[ord] = uint8_t(tid);
        expert_count[ord] = uint8_t(cnt_e);
        vstart[ord] = uint16_t(grp_inc - grp_e);
        pstart[ord] = uint16_t(p0);
        pstart_of_e[tid] = uint16_t(p0);
    }
    if (tid == kExperts - 1) {
        distinct_s = act_inc;
        vstart[act_inc] = uint16_t(grp_inc);
        pstart[act_inc] = uint16_t(cnt_inc);
    }
    __syncthreads();

    if (pe0 >= 0) {
        int col = __popc(same0 & below);
        for (int i = 0; i < warp; ++i) {
            col += int(warp_count[i][pe0]);
        }
        pair_list[int(pstart_of_e[pe0]) + col] = uint8_t(tid);
    }
    if (pairs > 128 && pe1 >= 0) {
        int col = __popc(same1 & below);
        for (int i = 0; i < kWarps + warp; ++i) {
            col += int(warp_count[i][pe1]);
        }
        pair_list[int(pstart_of_e[pe1]) + col] = uint8_t(128 + tid);
    }
    __syncthreads();

    const int distinct = distinct_s;
    const int groups_total = int(vstart[distinct]);
    const uint64_t policy = evict_first_policy();
    uint4* const ring = ring_all + warp * kRing * kMxfp6SlotVecs;
    const int tasks13 = groups_total * kTiles;
    const int tasks = 2 * tasks13;
    const int warps = gridDim.x * kWarps;

    for (int task = blockIdx.x * kWarps + warp; task < tasks; task += warps) {
        const bool up_proj = task < tasks13;
        const int flat = up_proj ? task : task - tasks13;
        int lo = 0;
        int hi = distinct - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (int(vstart[mid]) * kTiles <= flat) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        const int ordinal = lo;
        const int groups = int(vstart[lo + 1]) - int(vstart[lo]);
        const int local = flat - int(vstart[lo]) * kTiles;
        const int row0 = (local / groups) * 16;
        const int group = int(vstart[lo]) + local % groups;
        const int e = int(expert_id[ordinal]);
        const int col0 = 8 * (group - int(vstart[ordinal]));
        const uint8_t* const pairs_of = &pair_list[int(pstart[ordinal]) + col0];
        const int count = min(int(expert_count[ordinal]) - col0, 8);
        const int mine = g < count ? int(pairs_of[g]) : -1;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        if (up_proj) {
            tile_product_mxfp6<kK13, kRing>(
                w13_lo4 + size_t(e) * kN13 * G13::kRowLo4Bytes,
                w13_hi2 + size_t(e) * kN13 * G13::kRowHi2Bytes,
                s13 + size_t(e) * kN13 * G13::kRowScaleBytes, row0,
                mine >= 0 ? x + size_t(mine / 8) * kK13 : nullptr, ring, lane, policy, acc, [] {});
            store_act_single(acc, 1.f, count, pairs_of, row0, gate_softcap, up_softcap, act, lane,
                             g, t, &sync[1 + group]);
        } else {
            tile_product_mxfp6<kK2, kRing>(
                w2_lo4 + size_t(e) * kN2 * G2::kRowLo4Bytes,
                w2_hi2 + size_t(e) * kN2 * G2::kRowHi2Bytes,
                s2 + size_t(e) * kN2 * G2::kRowScaleBytes, row0,
                mine >= 0 ? act + size_t(mine) * kK2 : nullptr, ring, lane, policy, acc,
                [&] { wait_sync_tiles(&sync[1 + group], lane); });
            store_out_single(acc, 1.f, count, pairs_of, row0, out, g, t);
        }
    }

    __syncthreads();
    if (warp == 0) {
        int last = 0;
        if (lane == 0) {
            last = atomicAdd(&sync[0], 1u) == gridDim.x - 1 ? 1 : 0;
        }
        if (__shfl_sync(0xffffffffu, last, 0)) {
            for (int i = lane; i < groups_total; i += 32) {
                sync[1 + i] = 0;
            }
            __syncwarp();
            if (lane == 0) {
                sync[0] = 0;
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(kWarps * 32)
    expert_mlp_rows_wide_mxfp6_3(const __nv_bfloat16* x, int m, const uint8_t* w13_lo4,
                                 const uint8_t* w13_hi2, const uint8_t* s13,
                                 const uint8_t* w2_lo4, const uint8_t* w2_hi2, const uint8_t* s2,
                                 __nv_bfloat16* act, __nv_bfloat16* out, float gate_softcap,
                                 float up_softcap, uint32_t* sync, int sync_end) {
    using G13 = GeometryMxfp6<kK13>;
    using G2 = GeometryMxfp6<kK2>;
    constexpr int kRing = 3;
    extern __shared__ uint4 ring_all[];
    __shared__ int num_chunks_s;
    __shared__ bool last_block;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int pairs = m * 8;

    pdl_wait();
    pdl_trigger();
    const auto* meta = reinterpret_cast<const ExpertRowsMeta*>(sync + sync_end);
    if (tid == 0) {
        num_chunks_s = meta->num_chunks;
    }
    __syncthreads();

    const int num_chunks = num_chunks_s;
    const uint64_t policy = evict_first_policy();
    uint4* ring = ring_all + warp * kRing * kMxfp6SlotVecs;
    const int tasks13 = num_chunks * kTiles;
    const int tasks = 2 * tasks13;

    for (;;) {
        int task;
        if (lane == 0) {
            task = int(atomicAdd(&sync[0], 1u));
        }
        task = __shfl_sync(0xffffffffu, task, 0);
        if (task >= tasks) {
            break;
        }
        const bool up_proj = task < tasks13;
        const int flat = up_proj ? task : task - tasks13;
        const int chunk = flat / kTiles;
        const int row0 = (flat % kTiles) * 16;
        const ExpertChunk ch = meta->chunks[chunk];
        const int e = int(ch.expert);
        const uint16_t* const pairs_of = &meta->pair_list[ch.pstart];
        const int count = int(ch.count);
        const int ng = (count + 7) / 8;
        const __nv_bfloat16* x_ptrs[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int idx = u * 8 + g;
            if (idx < count) {
                const int pair = int(pairs_of[idx]);
                x_ptrs[u] = up_proj ? (x + size_t(pair / 8) * kK13) : (act + size_t(pair) * kK2);
            } else {
                x_ptrs[u] = nullptr;
            }
        }
        float acc[4][4] = {{0.f, 0.f, 0.f, 0.f},
                           {0.f, 0.f, 0.f, 0.f},
                           {0.f, 0.f, 0.f, 0.f},
                           {0.f, 0.f, 0.f, 0.f}};
        if (up_proj) {
            tile_product_wide_mxfp6<kK13, kRing>(
                w13_lo4 + size_t(e) * kN13 * G13::kRowLo4Bytes,
                w13_hi2 + size_t(e) * kN13 * G13::kRowHi2Bytes,
                s13 + size_t(e) * kN13 * G13::kRowScaleBytes, row0, x_ptrs, ng, ring, lane, policy,
                acc, [] {});
            __nv_bfloat16* act_smem = reinterpret_cast<__nv_bfloat16*>(ring);
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (u < ng) {
                    const int col_off = u * 8;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gate = acc[u][i];
                        const float up = __shfl_down_sync(0xffffffffu, gate, 4);
                        const int column = col_off + 2 * t + (i & 1);
                        if (g % 2 == 0 && column < count) {
                            act_smem[column * 8 + (g >> 1) + (i >> 1) * 4] = softcapped_swiglu(
                                __bfloat162float(__float2bfloat16(gate)),
                                __bfloat162float(__float2bfloat16(up)), gate_softcap, up_softcap);
                        }
                    }
                }
            }
            __syncwarp();
            if (lane < count) {
                const int pair = int(pairs_of[lane]);
                *reinterpret_cast<uint4*>(act + size_t(pair) * kK2 + row0 / 2) =
                    *reinterpret_cast<const uint4*>(act_smem + lane * 8);
            }
            fence_acq_rel_gpu();
            __syncwarp();
            if (lane == 0) {
                atomic_inc_release(&sync[2 + chunk]);
            }
        } else {
            tile_product_wide_mxfp6<kK2, kRing>(
                w2_lo4 + size_t(e) * kN2 * G2::kRowLo4Bytes,
                w2_hi2 + size_t(e) * kN2 * G2::kRowHi2Bytes,
                s2 + size_t(e) * kN2 * G2::kRowScaleBytes, row0, x_ptrs, ng, ring, lane, policy,
                acc, [&] {
                    if (lane == 0) {
                        while (load_acquire(&sync[2 + chunk]) < kTiles) {
                            __nanosleep(64);
                        }
                    }
                    __syncwarp();
                });
            __nv_bfloat16* out_smem = reinterpret_cast<__nv_bfloat16*>(ring);
            constexpr int kWideOutStride = 24;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (u < ng) {
                    const int col_off = u * 8;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int column = col_off + 2 * t + (i & 1);
                        if (column < count) {
                            out_smem[column * kWideOutStride + g + (i >> 1) * 8] =
                                __float2bfloat16(acc[u][i]);
                        }
                    }
                }
            }
            __syncwarp();
            if (lane < count) {
                const int pair = int(pairs_of[lane]);
                __nv_bfloat16* dst = out + size_t(pair) * kN2 + row0;
                const __nv_bfloat16* src = out_smem + lane * kWideOutStride;
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
                *reinterpret_cast<uint4*>(dst + 8) = *reinterpret_cast<const uint4*>(src + 8);
            }
            __syncwarp();
        }
    }
    __syncthreads();
    if (tid == 0) {
        last_block = atomicAdd(&sync[1], 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (last_block) {
        for (int i = tid; i < num_chunks; i += kWarps * 32) {
            sync[2 + i] = 0;
        }
        const int meta_words = int(offsetof(ExpertRowsMeta, pair_list) / 4) + (pairs + 1) / 2;
        uint32_t* meta_u32 = sync + sync_end;
        for (int i = tid; i < meta_words; i += kWarps * 32) {
            meta_u32[i] = 0;
        }
        __syncthreads();
        if (tid == 0) {
            sync[0] = 0;
            sync[1] = 0;
        }
    }
}
