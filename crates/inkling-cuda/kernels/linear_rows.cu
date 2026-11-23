// Dense products `out = x @ W^T` for 1 to 8 rows, with BF16 or FP8 E4M3 weights, BF16
// inputs and outputs and F32 accumulation: the attention projections, the message gates'
// down projections and the LM head of a step of a few rows, such as a speculative verify.
//
// x is [m, k] and out [m, n], row-major, m from 1 to 8. W is [n, k] row-major: BF16 as
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
// the 8 input rows as the columns of B, so 1 and 8 rows cost the same instructions. Lane
// (g, t) of a warp, g = lane / 4 and t = lane % 4, reads 16 bytes of weight rows g and
// g + 8 at the same k positions, and the 16 bytes of input row g at those positions;
// rows g >= m read as zeros. Since A and B see the same order of k, the products are the
// exact dot products, summed in the tensor core's order. FP8 weights widen exactly to
// BF16 first (E4M3 to F16 in hardware, then F16 to BF16 by moving the bits and
// multiplying by 2^112), and the row's scale multiplies the F32 sum once.
//
// The warps of a block add their partial sums in warp order through shared memory, and
// the result rounds to BF16 once. Each output depends on its own input row only: the
// tensor core computes each column of the product separately, and `splits` comes from
// the weight shapes, not from m. So a row's result is the same, bit for bit, for every m
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

template <bool kFp8, int K>
__device__ __forceinline__ void rows_product(const Batch& b, uint4* ring) {
    constexpr int kBytes = kFp8 ? 1 : 2;
    constexpr int kKPerStage = kStageBytes / kBytes;
    constexpr int kStagesPerRow = K / kKPerStage;
    static_assert(kStagesPerRow * kKPerStage == K, "whole stages per row");

    int s = 0;
    int tile_begin = 0;
    while (s + 1 < b.count && int(blockIdx.x) >= b.seg[s].tile_end) {
        tile_begin = b.seg[s].tile_end;
        ++s;
    }
    const Segment& seg = b.seg[s];
    const int splits = blockDim.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int r0 = (int(blockIdx.x) - tile_begin) * kTileRows;
    uint4* buf = ring + warp * kSlots * kSlotVecs;
    // The launcher keeps splits at most kStagesPerRow, so every warp has a stage.
    const int stages = (kStagesPerRow - warp + splits - 1) / splits;

    // Stage i of this warp into its slot. Rows past n repeat row n - 1 (BF16 weights have
    // no padding rows); their results are not stored.
    auto issue = [&](int i) {
        if (i < stages) {
            const int stage = warp + i * splits;
            uint4* slot = buf + i % kSlots * kSlotVecs;
#pragma unroll
            for (int q = 0; q < kSlotVecs / 32; ++q) {
                const int idx = q * 32 + lane;
                const int r = idx / kChunks;
                const int c = idx % kChunks;
                const int row = min(r0 + r, seg.n - 1);
                copy16(slot + slot_index(r, c),
                       seg.w + size_t(row) * K * kBytes + stage * kStageBytes + c * 16);
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
    const uint4* x = reinterpret_cast<const uint4*>(seg.x + size_t(g) * K);
    const bool live = g < b.m;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    // Input vectors per lane per stage: one per BF16 chunk, two per FP8 chunk.
    constexpr int kXVecs = kChunks / 4 * (kFp8 ? 2 : 1);
    for (int i = 0; i < stages; ++i) {
        issue(i + kSlots - 1);
        const int stage = warp + i * splits;
        // The stage's inputs load while its weights arrive; they mostly come from L2.
        uint4 xv[kXVecs];
#pragma unroll
        for (int j = 0; j < kChunks / 4; ++j) {
            const int c = 4 * j + t;
#pragma unroll
            for (int h = 0; h < kXVecs / (kChunks / 4); ++h) {
                const int xi = (stage * kKPerStage + c * (kFp8 ? 16 : 8)) / 8 + h;
                xv[j * (kXVecs / (kChunks / 4)) + h] = live ? x[xi] : make_uint4(0, 0, 0, 0);
            }
        }
        wait_copies<kSlots - 1>();
        __syncwarp();
        const uint4* slot = buf + i % kSlots * kSlotVecs;
#pragma unroll
        for (int j = 0; j < kChunks / 4; ++j) {
            const int c = 4 * j + t;
            const uint4 a = slot[slot_index(g, c)];
            const uint4 a8 = slot[slot_index(g + 8, c)];
            if constexpr (kFp8) {
                // 16 E4M3 weights per row and 16 inputs: four products of k = 16.
                const uint4 x0 = xv[2 * j];
                const uint4 x1 = xv[2 * j + 1];
                const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
                const uint32_t bw[4] = {a8.x, a8.y, a8.z, a8.w};
                const uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    mma(acc, e4m3x2_to_bf16x2(aw[q] & 0xffffu), e4m3x2_to_bf16x2(bw[q] & 0xffffu),
                        e4m3x2_to_bf16x2(aw[q] >> 16), e4m3x2_to_bf16x2(bw[q] >> 16), xw[2 * q],
                        xw[2 * q + 1]);
                }
            } else {
                // 8 BF16 weights per row and 8 inputs: two products of k = 16.
                mma(acc, a.x, a8.x, a.y, a8.y, xv[j].x, xv[j].y);
                mma(acc, a.z, a8.z, a.w, a8.w, xv[j].z, xv[j].w);
            }
        }
        // Every lane has read the slot before a later stage overwrites it.
        __syncwarp();
    }

    // The partial sums meet in the warps' rings, whose copies have all completed.
    if (splits > 1) {
        reinterpret_cast<float4*>(buf)[lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        __syncthreads();
        if (warp != 0) {
            return;
        }
        for (int w = 1; w < splits; ++w) {
            const float4 p = reinterpret_cast<const float4*>(ring + w * kSlots * kSlotVecs)[lane];
            acc[0] += p.x;
            acc[1] += p.y;
            acc[2] += p.z;
            acc[3] += p.w;
        }
    }
    // acc[0], acc[1]: weight row g, input rows 2t and 2t + 1; acc[2], acc[3]: row g + 8.
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int col = 2 * t + (e & 1);
        const int row = r0 + g + (e >> 1) * 8;
        if (col < b.m && row < seg.n) {
            const float v = kFp8 ? seg.scale[row] * acc[e] : acc[e];
            seg.out[size_t(col) * seg.n + row] = __float2bfloat16_rn(v);
        }
    }
}

}  // namespace

// Entry points by weight format and k. Blocks of 1 to kMaxWarps warps, one tile of 16
// rows each, with splits x kSlots x 4 KiB of dynamic shared memory.
#define ROWS_ENTRY(name, fp8, K)                                                           \
    extern "C" __global__ void __launch_bounds__(kMaxWarps * 32)                           \
        name(const __grid_constant__ Batch batch) {                                        \
        extern __shared__ uint4 ring[];                                                    \
        rows_product<fp8, K>(batch, ring);                                                 \
    }

// The attention input projections and the LM head.
ROWS_ENTRY(rows_bf16_k2304, false, 2304)
ROWS_ENTRY(rows_fp8_k2304, true, 2304)
// The output projection.
ROWS_ENTRY(rows_bf16_k3072, false, 3072)
ROWS_ENTRY(rows_fp8_k3072, true, 3072)
// The message gates' down projections.
ROWS_ENTRY(rows_bf16_k4608, false, 4608)
ROWS_ENTRY(rows_fp8_k4608, true, 4608)
