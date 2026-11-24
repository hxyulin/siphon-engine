// The expert MLP of 1 to 8 rows in one kernel: for every (row, slot) pair, the routed
// expert's w13, the softcapped SwiGLU and w2, as experts.cu's decode `expert_mlp` does for
// one row. Each distinct expert's weights stream from DRAM once per call, for all the rows
// that route to it, so the weight bytes follow the distinct experts, not 8 m.
//
// Every block first groups the m x 8 pairs by expert in shared memory: the distinct
// experts in increasing id order, and each expert's pairs in row order (a row routes to an
// expert at most once, so an expert has at most m pairs). The tasks are then, in order, the
// 16-row tiles of every distinct expert's w13 and the 16-row tiles of every w2. Each warp
// takes tasks warp, warp + warps, and so on over the grid. A w13 task adds 1 to its expert's
// count `sync[1 + expert ordinal]` after storing its activations; a w2 task waits until the
// count reaches 144, the tiles of w13. As in `expert_mlp`, a warp reaches each of its w13
// tasks without waiting, so every count completes as long as all blocks are resident, which
// the launcher ensures. The last block to finish zeroes `sync`.
//
// A task streams its 16 weight rows in 9 stages through a per-warp ring of shared memory
// with cp.async (linear_rows.cu describes the layout), each stage's block scales in the
// same slot and copy group, and multiplies them on tensor
// cores: mma.sync m16n8k16 with the 16 weight rows as A and the expert's pairs as the
// columns of B, so one to eight pairs cost the same. Lane (g, t) holds 16 bytes of weight
// rows g and g + 8 and the inputs of the expert's pair g at the same k positions.
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
// compute each column separately, and the tiles and stages depend on the expert alone. So
// they are the same, bit for bit, for every m and whatever the other rows route to.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>

#include "mlx_ops.cuh"
#include "pdl.cuh"

namespace {

constexpr int kMaxRows = 8;
constexpr int kMaxPairs = kMaxRows * 8;
// Warps per block; each runs its own tasks.
constexpr int kWarps = 4;
// Ring slots per warp: copies run kRing - 1 stages ahead of the products. The launcher picks
// 4 slots for 1 to 3 rows, where fewer warps run deeper copies faster, and 3 above.
// The ring changes when loads issue, not the arithmetic, so results do not depend on it.
// Bytes of each weight row per stage: 128, or 64 for NVFP4's w2, whose rows are 576 bytes.
constexpr int kStageBytes = 128;
constexpr int kK13 = 2304;
constexpr int kN13 = 2304;
constexpr int kK2 = 1152;
constexpr int kN2 = 2304;
// 16-row tiles of either projection.
constexpr int kTiles = 144;
static_assert(kTiles * 16 == kN13 && kTiles * 16 == kN2, "whole tiles");
static_assert(kMaxPairs <= 64 && kWarps >= 2, "pairs fill at most two warps");
// `F8E4M3_MAX * F4E2M1_MAX` in MLX's `inv_scale_enc`.
constexpr float kNvfp4ScaleEnc = 448.f * 6.f;

// Stage geometry of a projection with input width K.
template <bool kFp4, int K>
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
    // Block-scale bytes per row and stage: 16 for NVFP4's w13, 8 for its w2, 4 for MXFP8.
    static constexpr int kStageScales = kKPerStage / kGroup;
    static_assert(kStageBytes * kStages == kRowBytes && kChunks % 4 == 0, "whole stages");
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
    const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(uint8_t(byte), __NV_E2M1);
    return uint32_t(h.x) | (uint32_t(h.y) << 16);
}

// Two E4M3 values (low byte first) as F16, exactly.
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t bits) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(bits), __NV_E4M3);
    return uint32_t(h.x) | (uint32_t(h.y) << 16);
}

// A scale times 2^112 as a BF16 pair: exact, since the scale is an E4M3 value or a power
// of two.
__device__ __forceinline__ uint32_t scale_pair(float scale) {
    const uint32_t b = __bfloat16_as_ushort(__float2bfloat16_rn(scale * 0x1p112f));
    return b | (b << 16);
}

__device__ __forceinline__ float e4m3(uint32_t bits) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(uint8_t(bits), __NV_E4M3)));
}

// MLX's `fp8_e8m0` to float: 2^(bits - 127), with 0 mapped to 2^-127.
__device__ __forceinline__ float e8m0(uint32_t bits) {
    return __uint_as_float(bits == 0 ? 0x400000u : bits << 23);
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
template <bool kFp4, int K, int kRing, typename Ready>
__device__ __forceinline__ void tile_product(const uint8_t* w, const uint8_t* s, int row0,
                                             const __nv_bfloat16* x, uint4* ring, int lane,
                                             uint64_t policy, float* acc, Ready ready) {
    using G = Geometry<kFp4, K>;
    constexpr int kJ = G::kChunks / 4;
    constexpr int kXVecs = G::kValues / 8;
    const int g = lane / 4;
    const int t = lane % 4;

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
            // The stage's block scales ride in the same group, 16 bytes apart per row.
            if (lane < 16) {
                copy_scales<G::kStageScales>(slot + kSlotWeightVecs + lane,
                                             s + size_t(row0 + lane) * G::kRowScales + i * G::kStageScales,
                                             policy);
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
        // The stage's inputs load from L2 while its weights arrive.
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
            // The scales of the chunk's values in rows g and g + 8: two for NVFP4's 32
            // values, one for MXFP8's 16.
            const int sc = c * G::kValues / G::kGroup;
            const uint32_t sa = kFp4 ? *reinterpret_cast<const uint16_t*>(scales + g * 16 + sc)
                                     : scales[g * 16 + sc];
            const uint32_t sb = kFp4 ? *reinterpret_cast<const uint16_t*>(scales + (g + 8) * 16 + sc)
                                     : scales[(g + 8) * 16 + sc];
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
                // 32 values in two scale blocks of 16: words 0, 1 and words 2, 3.
                const uint32_t scale_a[2] = {scale_pair(e4m3(sa & 0xff)), scale_pair(e4m3(sa >> 8))};
                const uint32_t scale_b[2] = {scale_pair(e4m3(sb & 0xff)), scale_pair(e4m3(sb >> 8))};
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
                // 16 values, half of one scale block.
                const uint32_t scale_a = scale_pair(e8m0(sa));
                const uint32_t scale_b = scale_pair(e8m0(sb));
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

template <bool kFp4, int kRing>
__device__ __forceinline__ void expert_mlp_rows(const __nv_bfloat16* x, const int32_t* indices,
                                                int m, const uint8_t* w13, const uint8_t* s13,
                                                const float* g13, const uint8_t* w2,
                                                const uint8_t* s2, const float* g2,
                                                __nv_bfloat16* act, __nv_bfloat16* out,
                                                float gate_softcap, float up_softcap,
                                                uint32_t* sync, uint4* ring_all) {
    using G13 = Geometry<kFp4, kK13>;
    using G2 = Geometry<kFp4, kK2>;
    __shared__ unsigned present[4];
    __shared__ uint8_t warp_count[2][128];
    // By expert ordinal: the expert's id, its pair count and its pairs in row order.
    __shared__ uint8_t expert_id[kMaxPairs];
    __shared__ uint8_t expert_count[kMaxPairs];
    __shared__ uint8_t expert_pairs[kMaxPairs][kMaxRows];
    __shared__ int distinct;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int pairs = m * 8;

    // The expert ids come from the router just before; nothing loads before the wait.
    pdl_wait();
    pdl_trigger();
    // Pair tid's expert.
    const int pe = tid < pairs ? indices[tid] : -1;
    if (tid < 4) {
        present[tid] = 0;
    }
    for (int i = tid; i < 2 * 128; i += kWarps * 32) {
        warp_count[i / 128][i % 128] = 0;
    }
    __syncthreads();
    // The pairs of this warp that route to the same expert, and the warp's count of each
    // expert; pairs fill at most the first two warps.
    unsigned same = 0;
    const unsigned below = (1u << lane) - 1;
    if (warp < 2) {
        same = __match_any_sync(0xffffffffu, pe);
    }
    if (tid < pairs) {
        atomicOr(&present[pe >> 5], 1u << (pe & 31));
        if ((same & below) == 0) {
            warp_count[warp][pe] = uint8_t(__popc(same));
        }
    }
    __syncthreads();
    if (tid < pairs) {
        int ordinal = __popc(present[pe >> 5] & ((1u << (pe & 31)) - 1));
        for (int i = 0; i < (pe >> 5); ++i) {
            ordinal += __popc(present[i]);
        }
        // The expert's pairs before this one, in pair order, and all of them.
        const int column = (warp == 1 ? warp_count[0][pe] : 0) + __popc(same & below);
        const int count = warp_count[0][pe] + warp_count[1][pe];
        expert_pairs[ordinal][column] = uint8_t(tid);
        if (column == 0) {
            expert_id[ordinal] = uint8_t(pe);
            expert_count[ordinal] = uint8_t(count);
        }
    }
    if (tid == 0) {
        distinct = __popc(present[0]) + __popc(present[1]) + __popc(present[2]) +
                   __popc(present[3]);
    }
    __syncthreads();

    const uint64_t policy = evict_first_policy();
    uint4* ring = ring_all + warp * kRing * kSlotVecs;
    const int tasks13 = distinct * kTiles;
    const int tasks = 2 * tasks13;
    const int warps = gridDim.x * kWarps;
    for (int task = blockIdx.x * kWarps + warp; task < tasks; task += warps) {
        const bool up_proj = task < tasks13;
        const int ordinal = (up_proj ? task : task - tasks13) / kTiles;
        const int row0 = (up_proj ? task : task - tasks13) % kTiles * 16;
        const int e = expert_id[ordinal];
        const int count = expert_count[ordinal];
        const int mine = g < count ? expert_pairs[ordinal][g] : -1;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        if (up_proj) {
            tile_product<kFp4, kK13, kRing>(w13 + size_t(e) * kN13 * G13::kRowBytes,
                                     s13 + size_t(e) * kN13 * G13::kRowScales, row0,
                                     mine >= 0 ? x + size_t(mine / 8) * kK13 : nullptr, ring,
                                     lane, policy, acc, [] {});
            const float scale = kFp4 ? __fdiv_rn(g13[e], kNvfp4ScaleEnc) : 1.f;
            // Rows alternate gate (even) and up (odd): lane g + 1 holds the up values of
            // lane g's gate rows.
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float gate = acc[i] * scale;
                const float up = __shfl_down_sync(0xffffffffu, gate, 4);
                const int column = 2 * t + (i & 1);
                if (g % 2 == 0 && column < count) {
                    const int row = row0 + g + (i >> 1) * 8;
                    const int pair = expert_pairs[ordinal][column];
                    act[size_t(pair) * kK2 + row / 2] = softcapped_swiglu(
                        __bfloat162float(__float2bfloat16(gate)),
                        __bfloat162float(__float2bfloat16(up)), gate_softcap, up_softcap);
                }
            }
            __threadfence();
            __syncwarp();
            if (lane == 0) {
                atomicAdd(&sync[1 + ordinal], 1u);
            }
        } else {
            const uint8_t* w2e = w2 + size_t(e) * kN2 * G2::kRowBytes;
            // The first stages of the tile's weights copy while the warp waits for the
            // activations. Prefetching the whole tile into L2 as well was slower: 4.93 ms per
            // step at 8 rows against 4.58, since it competes with the w13 tiles still running.
            tile_product<kFp4, kK2, kRing>(w2e, s2 + size_t(e) * kN2 * G2::kRowScales, row0,
                                    mine >= 0 ? act + size_t(mine) * kK2 : nullptr, ring, lane,
                                    policy, acc, [&] {
                                        if (lane == 0) {
                                            while (load_relaxed(&sync[1 + ordinal]) < kTiles) {
                                                __nanosleep(64);
                                            }
                                            // With the warp barrier below, orders the warp's
                                            // activation loads after the count, and so after
                                            // the stores it covers.
                                            __threadfence();
                                        }
                                        __syncwarp();
                                    });
            const float scale = kFp4 ? __fdiv_rn(g2[e], kNvfp4ScaleEnc) : 1.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int column = 2 * t + (i & 1);
                if (column < count) {
                    const int row = row0 + g + (i >> 1) * 8;
                    const int pair = expert_pairs[ordinal][column];
                    out[size_t(pair) * kN2 + row] = __float2bfloat16(acc[i] * scale);
                }
            }
        }
    }
    __syncthreads();
    if (tid == 0 && atomicAdd(&sync[0], 1u) == gridDim.x - 1) {
        // Every block has finished with the counts.
        for (int i = 0; i < distinct; ++i) {
            sync[1 + i] = 0;
        }
        sync[0] = 0;
    }
}

}  // namespace

// `expert_mlp_rows_<format>`:
// - `x`: BF16 `[m, 2304]`; `indices`: `[m, 8]` expert ids, each in [0, experts) and distinct
//   within a row; m from 1 to 8;
// - `w13`, `s13`, `w2`, `s2`: raw checkpoint bytes as for `expert_mlp` in experts.cu, w13
//   `[experts, 2304, 2304]` with gate (even) and up (odd) rows interleaved, w2
//   `[experts, 2304, 1152]`; `g13`, `g2`: NVFP4's F32 global scales, unused by MXFP8;
// - `act`: BF16 `[m * 8, 1152]`, written; `out`: BF16 `[m * 8, 2304]`, pair `8 r + s` for
//   row r and slot s;
// - `sync`: 1 + 8 m zeros, left zero.
// Blocks of kWarps warps with kWarps x ring slots of dynamic shared memory; every block of
// the grid must be resident at once. `expert_mlp_rows_<format>_<ring>` runs `ring` slots.
#define EXPERT_MLP_ROWS(name, fp4, RING)                                                           \
    extern "C" __global__ void __launch_bounds__(kWarps * 32)                                       \
        name(const __nv_bfloat16* x, const int32_t* indices, int m, const uint8_t* w13,            \
             const uint8_t* s13, const float* g13, const uint8_t* w2, const uint8_t* s2,           \
             const float* g2, __nv_bfloat16* act, __nv_bfloat16* out, float gate_softcap,          \
             float up_softcap, uint32_t* sync) {                                                   \
        extern __shared__ uint4 ring[];                                                            \
        expert_mlp_rows<fp4, RING>(x, indices, m, w13, s13, g13, w2, s2, g2, act, out,             \
                                   gate_softcap, up_softcap, sync, ring);                          \
    }

EXPERT_MLP_ROWS(expert_mlp_rows_nvfp4_3, true, 3)
EXPERT_MLP_ROWS(expert_mlp_rows_mxfp8_3, false, 3)
EXPERT_MLP_ROWS(expert_mlp_rows_nvfp4_4, true, 4)
EXPERT_MLP_ROWS(expert_mlp_rows_mxfp8_4, false, 4)
