// Dense products `out = x @ W^T` for 1 to 32 rows, with BF16 or FP8 E4M3 weights, BF16
// inputs and outputs and F32 accumulation: the attention projections, the message gates'
// down projections and the LM head of a step of a few rows, such as a speculative verify.
//
// x is [m, k] and out [m, n], row-major, m from 1 to 32. W is [n, k] row-major: BF16 as
// stored in the checkpoint, or the E4M3 bytes of an Fp8Matrix (linear_fp8.cu), whose rows
// are padded to a multiple of 32 and followed by one F32 scale per row.
//
// The weights stream from DRAM once for all m rows. A block owns a tile of 16 weight rows
// and its `splits` warps share the tile's k range: warp w takes stages w, w + splits, and
// so on, where a stage is 16 rows x 256 bytes. Each warp copies its stages into its own
// ring of kSlots shared-memory slots with cp.async, kSlots - 1 stages ahead; every copy
// instruction moves 512 contiguous bytes of a row pair. Reading straight into mma
// fragments instead touched 8 rows per load instruction, 64 bytes each, and streamed at
// 1.42 TB/s against 1.53 TB/s through shared memory.
//
// The products run on tensor cores: mma.sync m16n8k16 with the 16 weight rows as A and
// 8 input rows as the columns of B, so 1 and 8 rows cost the same instructions. Rows
// 8 j .. 8 j + 7 form column tile j, and an entry point runs NT tiles, the same A
// fragments with each tile's B. Lane (g, t) of a warp, g = lane / 4 and t = lane % 4,
// reads 16 bytes of weight rows g and g + 8 at the same k positions, and the 16 bytes of
// input rows g + 8 j at those positions; rows at or past m read as zeros. Since A and B see the same order of k, the products are the
// exact dot products, summed in the tensor core's order. FP8 weights widen exactly to
// BF16 first (E4M3 to F16 in hardware, then F16 to BF16 by moving the bits and
// multiplying by 2^112), and the row's scale multiplies the F32 sum once.
//
// The warps of a block add their partial sums in warp order through shared memory, and
// the result rounds to BF16 once. Each output depends on its own input row only: the
// tensor core computes each column of the product separately, and `splits` comes from
// the weight shapes, not from m. A row's column tile runs the same instructions on its
// accumulators whatever NT is. So a row's result is the same, bit for bit, for every m
// and whatever the other rows hold.
//
// With programmatic dependent launch (the launcher picks it per call), each warp issues
// its first kSlots - 1 stages, which hold only weights, before pdl_wait, and reads x
// after it; it triggers after the wait, as the GEMVs in linear.cu do.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#include "pdl.cuh"

namespace {

constexpr int kMaxSegments = 8;
// Most warps per block, and so most splits of a tile's k range.
constexpr int kMaxWarps = 8;
// Ring slots per warp: copies run kSlots - 1 stages ahead of the products.
constexpr int kSlots = 3;
// Bytes of each tile row per stage, and its 16-byte chunks.
constexpr int kStageBytes = 256;
constexpr int kChunks = kStageBytes / 16;
// Weight rows per tile. Tiles of 32 rows, which halve the input loads, streamed at most
// 1.31 TB/s against 1.60 for BF16 weights: fewer warps, and shared memory caps the splits.
constexpr int kTileRows = 16;
// 16-byte vectors per slot.
constexpr int kSlotVecs = kTileRows * kChunks;

struct Segment {
    const uint8_t* w;
    // FP8 only: one F32 scale per row.
    const float* scale;
    const __nv_bfloat16* x;
    __nv_bfloat16* out;
    int n;
    // Tiles of this segment and every segment before it.
    int tile_end;
    // Tile pairs of this segment and every segment before it: a segment's tiles in pairs,
    // the last alone when the segment has an odd count.
    int pair_end;
};

struct Batch {
    Segment seg[kMaxSegments];
    int count;
    int m;
    // Nonzero when launched with programmatic dependent launch.
    int pdl;
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16 weight bytes from global to shared memory, past L1.
__device__ __forceinline__ void copy16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr(dst)), "l"(src)
                 : "memory");
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait_copies() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                    uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Two E4M3 values (low byte first) as two BF16 values, exactly. Every E4M3 value is a
// normal F16 value with the low 7 mantissa bits zero, so shifting its exponent and
// mantissa right by 3 gives a BF16 value 2^-112 times as large, which the multiply
// scales back exactly.
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t pair) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3);
    const uint32_t u = uint32_t(h.x) | (uint32_t(h.y) << 16);
    uint32_t r = ((u & 0x7fff7fffu) >> 3) | (u & 0x80008000u);
    __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&r);
    v = __hmul2(v, __floats2bfloat162_rn(0x1p112f, 0x1p112f));
    return *reinterpret_cast<uint32_t*>(&v);
}

// Chunk c of tile row r within a slot. Rows of odd index swap their two 64-byte halves of
// each 128 bytes, so the 8 lanes of a shared-memory phase, which read rows g and g + 1,
// hit distinct banks.
__device__ __forceinline__ int slot_index(int r, int c) { return r * kChunks + (c ^ ((r & 1) << 2)); }

// kT tiles per block: 1, or 2 consecutive tiles of one segment, whose warps share their
// input loads. Each tile runs the same instructions on its own accumulators either way, so
// kT changes no result; it halves the input bytes each weight byte needs, which at 3 and 4
// column tiles are 3 and 4 times the FP8 weight bytes and come from L2.
template <bool kFp8, int K, int NT, int kT>
__device__ __forceinline__ void rows_product(const Batch& b, uint4* ring) {
    constexpr int kBytes = kFp8 ? 1 : 2;
    constexpr int kKPerStage = kStageBytes / kBytes;
    constexpr int kStagesPerRow = K / kKPerStage;
    static_assert(kStagesPerRow * kKPerStage == K, "whole stages per row");
    static_assert(kT == 1 || kT == 2, "one tile or a pair");

    int s = 0;
    int begin = 0;
    while (s + 1 < b.count && int(blockIdx.x) >= (kT == 1 ? b.seg[s].tile_end : b.seg[s].pair_end)) {
        begin = kT == 1 ? b.seg[s].tile_end : b.seg[s].pair_end;
        ++s;
    }
    const Segment& seg = b.seg[s];
    const int splits = blockDim.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    // Weight row of the block's first tile; tile u starts kTileRows u rows later, and a
    // second tile past the segment's rows computes clamped rows it never stores.
    const int r0 = (int(blockIdx.x) - begin) * kT * kTileRows;
    uint4* buf = ring + warp * kSlots * kT * kSlotVecs;
    // The launcher keeps splits at most kStagesPerRow, so every warp has a stage.
    const int stages = (kStagesPerRow - warp + splits - 1) / splits;

    // Stage i of this warp into its slot. Rows past n repeat row n - 1 (BF16 weights have
    // no padding rows); their results are not stored.
    auto issue = [&](int i) {
        if (i < stages) {
            const int stage = warp + i * splits;
#pragma unroll
            for (int u = 0; u < kT; ++u) {
                uint4* slot = buf + (i % kSlots * kT + u) * kSlotVecs;
#pragma unroll
                for (int q = 0; q < kSlotVecs / 32; ++q) {
                    const int idx = q * 32 + lane;
                    const int r = idx / kChunks;
                    const int c = idx % kChunks;
                    const int row = min(r0 + u * kTileRows + r, seg.n - 1);
                    copy16(slot + slot_index(r, c),
                           seg.w + size_t(row) * K * kBytes + stage * kStageBytes + c * 16);
                }
            }
        }
        commit();
    };
#pragma unroll
    for (int i = 0; i < kSlots - 1; ++i) {
        issue(i);
    }
    // A branch keeps ptxas from placing the wait before the copies (see pdl.cuh).
    if (b.pdl) {
        pdl_wait();
    }
    pdl_trigger();

    // Plain loads: x may come from a kernel that ran just before this one.
    const uint4* x[NT];
    bool live[NT];
    float acc[kT][NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        x[n] = reinterpret_cast<const uint4*>(seg.x + size_t(g + 8 * n) * K);
        live[n] = g + 8 * n < b.m;
#pragma unroll
        for (int u = 0; u < kT; ++u) {
            acc[u][n][0] = acc[u][n][1] = acc[u][n][2] = acc[u][n][3] = 0.f;
        }
    }
    // Input vectors per lane per stage and tile: one per BF16 chunk, two per FP8 chunk.
    constexpr int kXVecs = kChunks / 4 * (kFp8 ? 2 : 1);
    for (int i = 0; i < stages; ++i) {
        issue(i + kSlots - 1);
        const int stage = warp + i * splits;
        // The stage's inputs load while its weights arrive; they mostly come from L2.
        uint4 xv[NT][kXVecs];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int j = 0; j < kChunks / 4; ++j) {
                const int c = 4 * j + t;
#pragma unroll
                for (int h = 0; h < kXVecs / (kChunks / 4); ++h) {
                    const int xi = (stage * kKPerStage + c * (kFp8 ? 16 : 8)) / 8 + h;
                    xv[n][j * (kXVecs / (kChunks / 4)) + h] = live[n] ? x[n][xi] : make_uint4(0, 0, 0, 0);
                }
            }
        }
        wait_copies<kSlots - 1>();
        __syncwarp();
#pragma unroll
        for (int u = 0; u < kT; ++u) {
            const uint4* slot = buf + (i % kSlots * kT + u) * kSlotVecs;
#pragma unroll
            for (int j = 0; j < kChunks / 4; ++j) {
                const int c = 4 * j + t;
                const uint4 a = slot[slot_index(g, c)];
                const uint4 a8 = slot[slot_index(g + 8, c)];
                if constexpr (kFp8) {
                    // 16 E4M3 weights per row and 16 inputs: four products of k = 16.
                    const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
                    const uint32_t bw[4] = {a8.x, a8.y, a8.z, a8.w};
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const uint32_t a0 = e4m3x2_to_bf16x2(aw[q] & 0xffffu);
                        const uint32_t a1 = e4m3x2_to_bf16x2(bw[q] & 0xffffu);
                        const uint32_t a2 = e4m3x2_to_bf16x2(aw[q] >> 16);
                        const uint32_t a3 = e4m3x2_to_bf16x2(bw[q] >> 16);
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            const uint4 x0 = xv[n][2 * j];
                            const uint4 x1 = xv[n][2 * j + 1];
                            const uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                            mma(acc[u][n], a0, a1, a2, a3, xw[2 * q], xw[2 * q + 1]);
                        }
                    }
                } else {
                    // 8 BF16 weights per row and 8 inputs: two products of k = 16.
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        mma(acc[u][n], a.x, a8.x, a.y, a8.y, xv[n][j].x, xv[n][j].y);
                        mma(acc[u][n], a.z, a8.z, a.w, a8.w, xv[n][j].z, xv[n][j].w);
                    }
                }
            }
        }
        // Every lane has read the slot before a later stage overwrites it.
        __syncwarp();
    }

    // The partial sums meet in the warps' rings, whose copies have all completed.
    static_assert(NT * 32 * 16 <= kSlots * kSlotVecs * 16, "the partials fit a warp's ring");
    if (splits > 1) {
#pragma unroll
        for (int u = 0; u < kT; ++u) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                reinterpret_cast<float4*>(buf + u * kSlots * kSlotVecs)[n * 32 + lane] =
                    make_float4(acc[u][n][0], acc[u][n][1], acc[u][n][2], acc[u][n][3]);
            }
        }
        __syncthreads();
        if (warp != 0) {
            return;
        }
        for (int w = 1; w < splits; ++w) {
#pragma unroll
            for (int u = 0; u < kT; ++u) {
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const float4 p = reinterpret_cast<const float4*>(
                        ring + (w * kT + u) * kSlots * kSlotVecs)[n * 32 + lane];
                    acc[u][n][0] += p.x;
                    acc[u][n][1] += p.y;
                    acc[u][n][2] += p.z;
                    acc[u][n][3] += p.w;
                }
            }
        }
    }
    // acc[u][n][0], acc[u][n][1]: weight row g of tile u, input rows 8 n + 2t and
    // 8 n + 2t + 1; acc[u][n][2], acc[u][n][3]: weight row g + 8.
#pragma unroll
    for (int u = 0; u < kT; ++u) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int col = 8 * n + 2 * t + (e & 1);
                const int row = r0 + u * kTileRows + g + (e >> 1) * 8;
                if (col < b.m && row < seg.n) {
                    const float v = kFp8 ? seg.scale[row] * acc[u][n][e] : acc[u][n][e];
                    seg.out[size_t(col) * seg.n + row] = __float2bfloat16_rn(v);
                }
            }
        }
    }
}

}  // namespace

// Entry points by weight format, k and column tiles: `<name>` for 1 to 8 rows, and
// `<name>_<NT>` for NT tiles of 8 rows, up to 8 NT rows. Blocks of 1 to kMaxWarps warps,
// one tile of 16 weight rows each, with splits x kSlots x 4 KiB of dynamic shared memory.
// `<name>_<NT>_pair`, for 3 and 4 column tiles, gives each block a pair of tiles (see
// rows_product) with twice the shared memory.
#define ROWS_ENTRY_NT(name, fp8, K, NT, kT)                                                \
    extern "C" __global__ void __launch_bounds__(kMaxWarps * 32)                           \
        name(const __grid_constant__ Batch batch) {                                        \
        extern __shared__ uint4 ring[];                                                    \
        rows_product<fp8, K, NT, kT>(batch, ring);                                         \
    }
#define ROWS_ENTRY(name, fp8, K)                                                           \
    ROWS_ENTRY_NT(name, fp8, K, 1, 1)                                                      \
    ROWS_ENTRY_NT(name##_2, fp8, K, 2, 1)                                                  \
    ROWS_ENTRY_NT(name##_3, fp8, K, 3, 1)                                                  \
    ROWS_ENTRY_NT(name##_4, fp8, K, 4, 1)                                                  \
    ROWS_ENTRY_NT(name##_3_pair, fp8, K, 3, 2)                                             \
    ROWS_ENTRY_NT(name##_4_pair, fp8, K, 4, 2)

// The attention input projections and the LM head.
ROWS_ENTRY(rows_bf16_k2304, false, 2304)
ROWS_ENTRY(rows_fp8_k2304, true, 2304)
// The output projection.
ROWS_ENTRY(rows_bf16_k3072, false, 3072)
ROWS_ENTRY(rows_fp8_k3072, true, 3072)
// The message gates' down projections.
ROWS_ENTRY(rows_bf16_k4608, false, 4608)
ROWS_ENTRY(rows_fp8_k4608, true, 4608)
