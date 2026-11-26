// Expert-grouped NVFP4 and MXFP8 matrix products for prefill.
//
// `expert_matvec` (experts.cu) reads each (row, slot) pair's expert weights
// separately. With many rows, most of those reads repeat: 512 rows make 4,096
// reads of at most 128 distinct experts. These kernels group the pairs by
// expert first, so each CTA reads one expert's weight tile once and applies it
// to up to 64 of that expert's inputs.
//
// 1. `expert_group` counts the pairs per expert, takes a prefix sum, writes
//    the pairs sorted by expert (`perm`) and a list of tiles of at most 64
//    pairs of one expert (128 for the quantized products below). The grouping
//    depends only on the ids, so one serves both projections of a layer.
// 2. `expert_gemm_*` runs one CTA per (tile, 128 output columns). It streams
//    its weight and scale tile into L2 ahead of use. Then, 64 input values
//    at a time, it copies the raw weight bytes, their scales and the tile's
//    BF16 inputs to shared memory with `cp.async`, decodes the weights to BF16,
//    and multiplies with BF16 `mma.sync.m16n8k16` into F32 accumulators. One
//    more row of CTAs fills the output rows of ids outside `[0, experts)` with
//    NaN.
//
// The weights are the MMA's A operand (16 output columns per fragment) and the
// inputs the B operand (8 pairs per fragment), so a tile with few pairs skips
// whole 8-pair fragments instead of padding to 16 or 64.
//
// Decoding is exact. An E2M1 value has 2 significant bits and an E4M3 value 4,
// so their product has at most 6, and E4M3 times a power of two keeps 4. BF16
// holds 8. The products with the BF16 inputs are then exact in the tensor
// cores, and only the F32 accumulation order differs from `expert_matvec`.
// NVFP4's per-expert global scale multiplies each result once, as in
// `expert_matvec`.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>

#include "mlx_ops.cuh"
#include "arch.cuh"
#include "mxfp8.cuh"

namespace {

constexpr int kSlots = 8;
// Output columns (weight rows) per CTA, 16 per warp.
constexpr int kTileN = 128;
// Pairs per tile.
constexpr int kTileT = 64;
// Input values per pipeline stage.
constexpr int kTileK = 64;
constexpr int kWarps = kTileN / 16;
constexpr int kThreads = kWarps * 32;
// Largest expert count `expert_group` supports.
constexpr int kMaxExperts = 1024;
constexpr int kGroupThreads = 1024;
// Largest pair count `expert_group` supports: 2048 rows of 8 slots.
constexpr int kMaxPairs = 16384;
constexpr int kGroupRounds = kMaxPairs / kGroupThreads;
// `F8E4M3_MAX * F4E2M1_MAX`, as in experts.cu.
constexpr float kNvfp4ScaleEnc = 448.f * 6.f;

__device__ uint32_t smem_addr(const void* p) {
    return uint32_t(__cvta_generic_to_shared(p));
}

// Copies 4 or 8 bytes.
template <int kBytes>
__device__ void cp_async_small(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(smem_addr(dst)), "l"(src),
                 "n"(kBytes));
}

// Copies `src_bytes` (0 or 16) and zero-fills the rest of 16 bytes.
__device__ void cp_async16(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_addr(dst)), "l"(src),
                 "r"(src_bytes));
}

// Starts loading `bytes` (a multiple of 16, 16-byte aligned) into L2. Ada has no bulk
// prefetch, so there it prefetches each 128-byte line the range touches.
__device__ void prefetch_l2_bulk(const void* src, uint32_t bytes) {
#if SIPHON_SM89
    const auto* p = static_cast<const char*>(src);
    for (uint32_t line = 0; line < bytes; line += 128) {
        asm volatile("prefetch.global.L2 [%0];" ::"l"(p + line) : "memory");
    }
#else
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
#endif
}

__device__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int kPending>
__device__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(kPending));
}

__device__ uint32_t pack_bf16x2(float2 v) {
    const __nv_bfloat162 b = __float22bfloat162_rn(v);
    return *reinterpret_cast<const uint32_t*>(&b);
}

// Two E2M1 values (one byte, low nibble first) times an E4M3 scale, as BF16x2.
// The product is exact in half precision: at most 6 significant bits, and
// magnitudes between 2^-10 and 2688.
__device__ uint32_t decode_e2m1x2(uint32_t byte, __half2 scale) {
    const __half2 w = __half2(__nv_cvt_fp4x2_to_halfraw2(uint8_t(byte), __NV_E2M1));
    return pack_bf16x2(__half22float2(__hmul2(w, scale)));
}

// Two E4M3 values (low byte first) times an E8M0 scale, as BF16x2.
__device__ uint32_t decode_e4m3x2(uint32_t bytes, float scale) {
    const float2 w =
        __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(uint16_t(bytes), __NV_E4M3)));
    return pack_bf16x2(make_float2(w.x * scale, w.y * scale));
}

// E8M0 to float. 0 stands for 2^-127, which `expert_matvec`'s flushing
// multiply turns into zero, so it decodes to zero here.
__device__ float e8m0(uint32_t bits) {
    return __uint_as_float(bits << 23);
}

__device__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Inclusive sum over the block's threads, for `kGroupThreads` threads.
__device__ int block_scan(int v, int* warp_sums) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        const int other = __shfl_up_sync(0xffffffffu, v, offset);
        v += lane >= offset ? other : 0;
    }
    if (lane == 31) {
        warp_sums[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        int s = warp_sums[lane];
#pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            const int other = __shfl_up_sync(0xffffffffu, s, offset);
            s += lane >= offset ? other : 0;
        }
        warp_sums[lane] = s;
    }
    __syncthreads();
    return v + (warp > 0 ? warp_sums[warp - 1] : 0);
}

// Shared memory of one pipeline stage: weight bytes, then inputs.
template <bool kFp4>
struct Stage {
    static constexpr int kWeightRow = kFp4 ? kTileK / 2 : kTileK;
    static constexpr int kWeight = kTileN * kWeightRow;
    // 64 BF16 inputs (128 bytes) per pair, with 16-byte units swapped in pairs
    // on odd pairs so that the B-fragment loads are free of bank conflicts.
    static constexpr int kInput = kTileT * kTileK * 2;
    static constexpr int kBytes = kWeight + kInput;
    // Scale bytes of one weight row for an aligned pair of chunks: 8 NVFP4
    // groups or 4 MXFP8 groups. They load with every even chunk into a ring
    // of their own after the stages.
    static constexpr int kScaleRow = kFp4 ? 8 : 4;
    static constexpr int kScaleSlot = kTileN * kScaleRow;
};

// Slots in the scale ring: a pair of chunks holds its slot until its second
// chunk is used, while up to `kStages - 1` later chunks load.
template <int kStages>
__host__ __device__ constexpr int scale_slots() {
    return (kStages + 2) / 2;
}

template <bool kFp4, int kStages>
__host__ __device__ constexpr int gemm_shared_bytes() {
    using S = Stage<kFp4>;
    return kStages * S::kBytes + scale_slots<kStages>() * S::kScaleSlot;
}

// The output values per pair of an `expert_gemm_*` kernel with n product
// columns: n, or n / 2 with `kSwiglu`.
template <bool kSwiglu>
__device__ int out_width(int n) {
    return kSwiglu ? n / 2 : n;
}

// The work of the extra row of CTAs: NaN output rows for the pairs with
// unknown ids, which `expert_group` put last in `perm`.
template <bool kSwiglu, int kBlockThreads = kThreads>
__device__ void nan_rows(__nv_bfloat16* __restrict__ out, const int* __restrict__ perm,
                         const int* __restrict__ group_counts, int n, int n0) {
    const __nv_bfloat16 nan = __float2bfloat16(__int_as_float(0x7fc00000));
    const int out_n = out_width<kSwiglu>(n);
    const int out0 = out_width<kSwiglu>(n0);
    const int out_columns = out_width<kSwiglu>(min(kTileN, n - n0));
    const int valid = group_counts[1];
    for (int i = valid * kTileN + int(threadIdx.x); i < group_counts[2] * kTileN;
         i += kBlockThreads) {
        if (i % kTileN < out_columns) {
            out[size_t(perm[i / kTileN]) * out_n + out0 + i % kTileN] = nan;
        }
    }
}

// Row stride, in BF16 values, of the epilogue's [pair][column] staging.
constexpr int kOutStride = kTileN + 8;

// The end of a CTA's epilogue, after it staged its BF16 results as
// [pair][column] in shared memory (`staged`, `kOutStride` apart): writes each
// pair's output row with wide stores, the products or with `kSwiglu` the
// softcapped SwiGLU of each (gate, up) column pair. `pairs` holds the tile's
// flat pair indices.
template <bool kSwiglu, int kBlockThreads = kThreads>
__device__ void write_tile(const __nv_bfloat16* staged, __nv_bfloat16* __restrict__ out,
                           const int* __restrict__ pairs, int count, int n, int n0,
                           float gate_softcap, float up_softcap) {
    const int out_n = out_width<kSwiglu>(n);
    const int out0 = out_width<kSwiglu>(n0);
    const int columns = min(kTileN, n - n0);
    for (int unit = threadIdx.x; unit < count * (kTileN / 8); unit += kBlockThreads) {
        const int pair = unit / (kTileN / 8);
        const int part = unit % (kTileN / 8);
        if (part * 8 < columns) {
            const uint4 v = *reinterpret_cast<const uint4*>(staged + pair * kOutStride + part * 8);
            __nv_bfloat16* dst = out + size_t(pairs[pair]) * out_n + out0;
            if constexpr (kSwiglu) {
                // Four (gate, up) pairs, as `softcapped_swiglu_bf16` reads them.
                const uint32_t words[4] = {v.x, v.y, v.z, v.w};
                __nv_bfloat16 act[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 gu =
                        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[i]));
                    act[i] = softcapped_swiglu(gu.x, gu.y, gate_softcap, up_softcap);
                }
                *reinterpret_cast<uint2*>(dst + part * 4) = *reinterpret_cast<const uint2*>(act);
            } else {
                *reinterpret_cast<uint4*>(dst + part * 8) = v;
            }
        }
    }
}

// One CTA: 128 output columns of one tile of pairs. Each warp owns 16 output
// columns and all 64 pairs. With `kSwiglu`, the columns interleave gate (even)
// and up (odd) values, and the CTA writes the softcapped SwiGLU of each pair of
// BF16 results instead: n / 2 outputs per pair.
//
// The K dimension is permuted inside each 64-value stage so that each lane
// loads its fragments with wide shared loads. Lane (g, t) = (lane / 4,
// lane % 4) holds MMA k pairs t and t + 4 of each 16-value step. In step s
// those pairs stand for input values 16t + 4s + {0, 1} and 16t + 4s + {2, 3}.
// Over the four steps, a lane reads values [16t, 16t + 16): 8 bytes of a
// NVFP4 weight row, 16 of an MXFP8 one, and 32 of an input row. Weights and
// inputs use the same permutation, so the dot products are unchanged. Those
// 16 values also share one scale.
//
// Shared memory is `gemm_shared_bytes` dynamic bytes and no static ones; two
// CTAs fit in an SM's 100 KiB when that is at most 50,176 bytes, since the
// driver reserves 1 KiB per CTA.
template <bool kFp4, int kStages, bool kSwiglu>
__device__ void expert_gemm(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ weight,
                            const uint8_t* __restrict__ scales, const float* __restrict__ global_scale,
                            __nv_bfloat16* __restrict__ out, const int* __restrict__ perm,
                            const int4* __restrict__ tiles, const int* __restrict__ group_counts,
                            int n, int k, int inputs_per_row, float gate_softcap,
                            float up_softcap) {
    using S = Stage<kFp4>;
    const int tile_count = group_counts[0];
    const int n0 = blockIdx.x * kTileN;
    const int tid = threadIdx.x;
    if (int(blockIdx.y) == tile_count) {
        nan_rows<kSwiglu>(out, perm, group_counts, n, n0);
        return;
    }
    if (int(blockIdx.y) > tile_count) {
        return;
    }
    const int4 tile = tiles[blockIdx.y];
    const int expert = tile.x;
    const int start = tile.y;
    const int count = tile.z;

    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* const scale_ring = smem + kStages * S::kBytes;

    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g = lane / 4;
    const int t = lane % 4;

    // Source rows for this thread's cp.async units, fixed across stages.
    constexpr int kWeightUnits = S::kWeight / 16 / kThreads;
    const size_t weight_row_bytes = kFp4 ? k / 2 : k;
    const uint8_t* weight_src[kWeightUnits];
    int weight_valid[kWeightUnits];
    int weight_dst[kWeightUnits];
#pragma unroll
    for (int i = 0; i < kWeightUnits; ++i) {
        const int unit = tid + i * kThreads;
        const int row = unit / (S::kWeightRow / 16);
        const int part = unit % (S::kWeightRow / 16);
        const bool valid = n0 + row < n;
        weight_valid[i] = valid ? 16 : 0;
        weight_src[i] = valid ? weight + (size_t(expert) * n + n0 + row) * weight_row_bytes + part * 16
                              : weight;
        weight_dst[i] = row * S::kWeightRow + part * 16;
    }
    // Pair fragments holding at least one real pair. The MMAs skip the rest,
    // so their inputs are not loaded; pairs past `count` in the last fragment
    // load zeros.
    const int fragments = (count + 7) / 8;
    constexpr int kInputUnits = S::kInput / 16 / kThreads;
    const __nv_bfloat16* input_src[kInputUnits];
    int input_dst[kInputUnits];
    int input_bytes[kInputUnits];
    bool input_load[kInputUnits];
#pragma unroll
    for (int i = 0; i < kInputUnits; ++i) {
        const int unit = tid + i * kThreads;
        const int pair = unit / 8;
        const int part = unit % 8;
        // Pairs past `count` reuse the first pair's row; their inputs load as
        // zeros and their results are dropped.
        const int entry = perm[start + (pair < count ? pair : 0)];
        const int input = inputs_per_row == 1 ? entry / kSlots : entry;
        input_src[i] = x + size_t(input) * k + part * 8;
        input_dst[i] = S::kWeight + pair * 128 + (part ^ (pair & 1)) * 16;
        input_bytes[i] = pair < count ? 16 : 0;
        input_load[i] = pair < fragments * 8;
    }
    const size_t scale_row_bytes = k / (kFp4 ? 16 : 32);
    // Threads below kTileN copy the scales of weight row `tid`.
    const bool scale_load = tid < kTileN && n0 + tid < n;
    const uint8_t* scale_src =
        scales + (size_t(expert) * n + n0 + min(tid, kTileN - 1)) * scale_row_bytes;

    auto load_stage = [&](int chunk, int stage) {
        uint8_t* base = smem + stage * S::kBytes;
        const int k0 = chunk * kTileK;
        if (chunk % 2 == 0 && scale_load) {
            const int slot = chunk / 2 % scale_slots<kStages>();
            cp_async_small<S::kScaleRow>(scale_ring + slot * S::kScaleSlot + tid * S::kScaleRow,
                                         scale_src + chunk / 2 * S::kScaleRow);
        }
#pragma unroll
        for (int i = 0; i < kWeightUnits; ++i) {
            cp_async16(base + weight_dst[i], weight_src[i] + (kFp4 ? k0 / 2 : k0),
                         weight_valid[i]);
        }
#pragma unroll
        for (int i = 0; i < kInputUnits; ++i) {
            if (input_load[i]) {
                cp_async16(base + input_dst[i], input_src[i] + k0, input_bytes[i]);
            }
        }
    };

    // Weights stream into L2 ahead of the pipeline, 256 bytes of each row at
    // a time, one segment ahead of the one in use, so the copies to shared
    // memory wait on L2 rather than DRAM. The whole tile at once would not
    // fit: 340 resident MXFP8 blocks would ask for 100 MB of the 96 MB L2.
    // Lane 0 of each warp prefetches its 16 rows. (The bulk prefetch takes
    // uniform operands, so per-lane addresses would serialize.) The scales,
    // an eighth of the weight bytes or less, are prefetched whole.
    constexpr int kSegmentBytes = 256;
    constexpr int kSegmentChunks = kSegmentBytes / S::kWeightRow;
    const bool warp_active = n0 + warp * 16 < n;
    const size_t warp_first_row = size_t(expert) * n + n0 + warp * 16;
    auto prefetch_segment = [&](int segment) {
        const int offset = segment * kSegmentBytes;
        if (warp_active && lane == 0 && offset < int(weight_row_bytes)) {
            const uint32_t bytes = min(kSegmentBytes, int(weight_row_bytes) - offset);
            for (int r = 0; r < 16; ++r) {
                prefetch_l2_bulk(weight + (warp_first_row + r) * weight_row_bytes + offset, bytes);
            }
        }
    };
    if (warp_active && lane == 0) {
        prefetch_l2_bulk(scales + warp_first_row * scale_row_bytes,
                         uint32_t(16 * scale_row_bytes));
    }
    prefetch_segment(0);
    prefetch_segment(1);

    const int chunks = k / kTileK;
    float acc[kTileT / 8][4];
#pragma unroll
    for (int j = 0; j < kTileT / 8; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[j][i] = 0.f;
        }
    }

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < chunks) {
            load_stage(s, s);
        }
        cp_async_commit();
    }

    const int row0 = warp * 16 + g;

    // One chunk's MMAs over the first `kFragments` pair fragments.
    auto compute = [&]<int kFragments>(int chunk, int stage) {
        const uint8_t* base = smem + stage * S::kBytes;
        // The lane's 16 values of rows `row0` and `row0 + 8` share one scale:
        // NVFP4 group 4c + t or MXFP8 group 2c + t / 2 of chunk c.
        const uint8_t* scale_slot =
            scale_ring + chunk / 2 % scale_slots<kStages>() * S::kScaleSlot +
            (kFp4 ? 4 * (chunk % 2) + t : 2 * (chunk % 2) + t / 2);
        const uint32_t scale_lo = scale_slot[row0 * S::kScaleRow];
        const uint32_t scale_hi = scale_slot[(row0 + 8) * S::kScaleRow];

        // A fragments for the four 16-value steps: [step][register].
        uint32_t a[4][4];
        if constexpr (kFp4) {
            const __half2 s_lo =
                __half2half2(__half(__nv_cvt_fp8_to_halfraw(scale_lo, __NV_E4M3)));
            const __half2 s_hi =
                __half2half2(__half(__nv_cvt_fp8_to_halfraw(scale_hi, __NV_E4M3)));
            const uint2 w_lo = *reinterpret_cast<const uint2*>(base + row0 * S::kWeightRow + 8 * t);
            const uint2 w_hi =
                *reinterpret_cast<const uint2*>(base + (row0 + 8) * S::kWeightRow + 8 * t);
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const uint32_t lo = s < 2 ? w_lo.x : w_lo.y;
                const uint32_t hi = s < 2 ? w_hi.x : w_hi.y;
                const int shift = (s % 2) * 16;
                a[s][0] = decode_e2m1x2(lo >> shift, s_lo);
                a[s][1] = decode_e2m1x2(hi >> shift, s_hi);
                a[s][2] = decode_e2m1x2(lo >> (shift + 8), s_lo);
                a[s][3] = decode_e2m1x2(hi >> (shift + 8), s_hi);
            }
        } else {
            const float s_lo = e8m0(scale_lo);
            const float s_hi = e8m0(scale_hi);
            const uint4 w_lo = *reinterpret_cast<const uint4*>(base + row0 * S::kWeightRow + 16 * t);
            const uint4 w_hi =
                *reinterpret_cast<const uint4*>(base + (row0 + 8) * S::kWeightRow + 16 * t);
            const uint32_t lo[4] = {w_lo.x, w_lo.y, w_lo.z, w_lo.w};
            const uint32_t hi[4] = {w_hi.x, w_hi.y, w_hi.z, w_hi.w};
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                a[s][0] = decode_e4m3x2(lo[s], s_lo);
                a[s][1] = decode_e4m3x2(hi[s], s_hi);
                a[s][2] = decode_e4m3x2(lo[s] >> 16, s_lo);
                a[s][3] = decode_e4m3x2(hi[s] >> 16, s_hi);
            }
        }

        const uint4* inputs = reinterpret_cast<const uint4*>(base + S::kWeight);
#pragma unroll
        for (int j = 0; j < kFragments; ++j) {
            const int pair = j * 8 + g;
            const int swap = pair & 1;
            const uint4 v0 = inputs[pair * 8 + ((2 * t) ^ swap)];
            const uint4 v1 = inputs[pair * 8 + ((2 * t + 1) ^ swap)];
            mma_bf16(acc[j], a[0], v0.x, v0.y);
            mma_bf16(acc[j], a[1], v0.z, v0.w);
            mma_bf16(acc[j], a[2], v1.x, v1.y);
            mma_bf16(acc[j], a[3], v1.z, v1.w);
        }
    };

    // The main loop, unrolled over the stages so that buffer offsets are
    // constants, and compiled once per fragment count so that the MMAs of
    // different fragments interleave without branches between them.
    auto main_loop = [&]<int kFragments>() {
        for (int chunk0 = 0; chunk0 < chunks; chunk0 += kStages) {
#pragma unroll
            for (int stage = 0; stage < kStages; ++stage) {
                const int chunk = chunk0 + stage;
                if (chunk < chunks) {
                    cp_async_wait<kStages - 2>();
                    __syncthreads();
                    const int next = chunk + kStages - 1;
                    if (next < chunks) {
                        load_stage(next, (stage + kStages - 1) % kStages);
                    }
                    cp_async_commit();
                    if (chunk % kSegmentChunks == 0 && chunk > 0) {
                        prefetch_segment(chunk / kSegmentChunks + 1);
                    }
                    if (warp_active) {
                        compute.template operator()<kFragments>(chunk, stage);
                    }
                }
            }
        }
    };
    switch (fragments) {
        case 1: main_loop.template operator()<1>(); break;
        case 2: main_loop.template operator()<2>(); break;
        case 3: main_loop.template operator()<3>(); break;
        case 4: main_loop.template operator()<4>(); break;
        case 5: main_loop.template operator()<5>(); break;
        case 6: main_loop.template operator()<6>(); break;
        case 7: main_loop.template operator()<7>(); break;
        default: main_loop.template operator()<8>(); break;
    }
    cp_async_wait<0>();
    __syncthreads();

    // Stage the BF16 results as [pair][column] so each pair's output row is
    // written with 16-byte stores.
    __nv_bfloat16* staged = reinterpret_cast<__nv_bfloat16*>(smem);
    if (warp_active) {
        float scale = 1.f;
        if constexpr (kFp4) {
            scale = __fdiv_rn(global_scale[expert], kNvfp4ScaleEnc);
        }
#pragma unroll
        for (int j = 0; j < kTileT / 8; ++j) {
            const int pair = j * 8 + 2 * t;
            staged[pair * kOutStride + row0] = __float2bfloat16(acc[j][0] * scale);
            staged[(pair + 1) * kOutStride + row0] = __float2bfloat16(acc[j][1] * scale);
            staged[pair * kOutStride + row0 + 8] = __float2bfloat16(acc[j][2] * scale);
            staged[(pair + 1) * kOutStride + row0 + 8] = __float2bfloat16(acc[j][3] * scale);
        }
    }
    __syncthreads();
    write_tile<kSwiglu>(staged, out, perm + start, count, n, n0, gate_softcap, up_softcap);
}

// ---------------------------------------------------------------------------
// Quantized activations (opt-in): the checkpoint's `required_activation_
// quantization` for the expert inputs, multiplied with block-scaled FP4 and
// FP8 tensor-core MMAs on the raw weights.
//
// `quantize_rows_*` quantizes each BF16 input row once, following vLLM's
// Inkling kernels (`vllm/models/inkling2/nvidia/ops/nvfp4.py` and `mxfp8.py`):
// - NVFP4: E2M1 values in blocks of 16 with E4M3 block scales, under a per-row
//   ("per_token") outer scale amax / 1536. Each block tries the scale that maps
//   its max to 6 and the one that maps it to 4, and keeps the one with the
//   smaller sum of absolute reconstruction errors ("fouroversix_mae"), 6 on
//   ties.
// - MXFP8: E4M3 values in blocks of 32 with a UE8M0 scale, the block's amax /
//   448 rounded up to a power of two.
// Both round to nearest even with saturation.
//
// Values and scales are stored per row in natural order: k / 2 (NVFP4) or k
// (MXFP8) value bytes, and k / 16 E4M3 or k / 32 UE8M0 scale bytes.

constexpr float kFp32Max = 3.4028234663852886e38f;

__device__ float e4m3_round(float x) {
    const __nv_fp8_e4m3 q(x);  // cvt.rn.satfinite
    return float(q);
}

// Two values rounded to E2M1 (cvt.rn.satfinite): the byte (`lo` in the low
// nibble) and both rounded values.
__device__ uint32_t e2m1x2(float lo, float hi, float& lo_value, float& hi_value) {
    uint32_t byte;
    uint32_t halves;
#if SIPHON_SM89
    __trap();
    byte = halves = 0;
#else
    asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %3, %2; cvt.u32.u8 %0, b; "
        "cvt.rn.f16x2.e2m1x2 %1, b; }"
        : "=r"(byte), "=r"(halves)
        : "f"(lo), "f"(hi));
#endif
    const __half2 h = *reinterpret_cast<const __half2*>(&halves);
    lo_value = __low2float(h);
    hi_value = __high2float(h);
    return byte;
}

// One NVFP4 block candidate with block scale `scale` (an E4M3 value) and
// `inverse`: its 8 bytes, and its error summed pairwise as vLLM's
// `_block_error` does.
__device__ float fp4_candidate(const float (&x)[16], float scale, float inverse, float outer,
                               uint2& packed) {
    float error[16];
    uint32_t bytes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float lo, hi;
        bytes[i] = e2m1x2(__fmul_rn(x[2 * i], inverse), __fmul_rn(x[2 * i + 1], inverse), lo, hi);
        error[2 * i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(lo, scale), outer), x[2 * i]));
        error[2 * i + 1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(hi, scale), outer), x[2 * i + 1]));
    }
    packed.x = bytes[0] | bytes[1] << 8 | bytes[2] << 16 | bytes[3] << 24;
    packed.y = bytes[4] | bytes[5] << 8 | bytes[6] << 16 | bytes[7] << 24;
#pragma unroll
    for (int width = 8; width >= 1; width /= 2) {
#pragma unroll
        for (int i = 0; i < width; ++i) {
            error[i] = __fadd_rn(error[2 * i], error[2 * i + 1]);
        }
    }
    return error[0];
}

// Max |x| over a warp's row, every lane taking blocks of `kBlock` values.
template <int kBlock>
__device__ float row_amax(const __nv_bfloat16* row, int k) {
    float amax = 0.f;
    for (int b = threadIdx.x % 32; b < k / kBlock; b += 32) {
#pragma unroll
        for (int i = 0; i < kBlock; i += 8) {
            float x[8];
            load8(row + b * kBlock + i, x);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                amax = fmaxf(amax, fabsf(x[j]));
            }
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    return amax;
}

// NVFP4: one warp per row.
__device__ void quantize_row_nvfp4(const __nv_bfloat16* __restrict__ row, int k,
                                   uint8_t* __restrict__ values, uint8_t* __restrict__ scales,
                                   float* __restrict__ outer) {
    const float amax = row_amax<16>(row, k);
    const float encoder = fminf(__fdiv_rn(1536.f, amax), kFp32Max);
    const float decoder = __fdiv_rn(1.f, encoder);
    const float outer_scale = __fmul_rn(amax, 1.f / 1536.f);
    if (threadIdx.x % 32 == 0) {
        *outer = outer_scale;
    }
    const float encoder6 = __fmul_rn(encoder, 1.f / 6.f);
    const float encoder4 = __fmul_rn(encoder, 0.25f);
    for (int b = threadIdx.x % 32; b < k / 16; b += 32) {
        float x[16];
        load8(row + b * 16, x);
        load8(row + b * 16 + 8, x + 8);
        float block_max = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            block_max = fmaxf(block_max, fabsf(x[i]));
        }
        const float scale6 = e4m3_round(fminf(__fmul_rn(block_max, encoder6), 448.f));
        const float scale4 = e4m3_round(fminf(__fmul_rn(block_max, encoder4), 448.f));
        const float inverse6 = fminf(__fdiv_rn(1.f, __fmul_rn(scale6, decoder)), kFp32Max);
        const float inverse4 = fminf(__fdiv_rn(1.f, __fmul_rn(scale4, decoder)), kFp32Max);
        uint2 packed6, packed4;
        const float error6 = fp4_candidate(x, scale6, inverse6, outer_scale, packed6);
        const float error4 = fp4_candidate(x, scale4, inverse4, outer_scale, packed4);
        const bool choose4 = error4 < error6;
        const uint2 packed = choose4 ? packed4 : packed6;
        const __nv_fp8_e4m3 scale(choose4 ? scale4 : scale6);
        scales[b] = scale.__x;
        *reinterpret_cast<uint2*>(values + b * 8) = packed;
    }
}

// The quantized GEMM, structured like CUTLASS's sm_120 grouped block-scaled
// GEMM: tiles of 128 pairs by 128 output columns, so that an expert's weights
// stream from L2 about once per call instead of once per 64 pairs. 8 warps
// split the tile 4 ways along the columns and 2 ways along the pairs; each
// warp owns 32 columns and 64 pairs, and loads its fragments with `ldmatrix`.
//
// A pipeline stage holds two MMA k steps, 64 bytes, of each of the 128 weight
// rows and 128 pairs: each L2 request then covers two sectors, and the L2's
// request rate binds less. The four 16-byte units of each row are XOR-swizzled
// so that the 8 rows one `ldmatrix` reads fall in distinct banks. Scales load
// into a ring of 2 slots of 16 bytes per row, each slot covering 4 NVFP4 k
// steps (4 E4M3 bytes per step) or 16 MXFP8 k steps (one E8M0 byte), in
// pieces of 8 (NVFP4) or 4 (MXFP8) bytes as their steps come up. Two stages
// and two CTAs per SM.
constexpr int kQTileT = 128;
constexpr int kQStages = 2;
constexpr int kQRowBytes = 64;
constexpr int kQStageSteps = kQRowBytes / 32;
constexpr int kQRows = kTileN + kQTileT;
constexpr int kQStageBytes = kQRows * kQRowBytes;
constexpr int kQSlotRow = 16;
constexpr int kQSlots = 2;
constexpr int kQShared = kQStages * kQStageBytes + kQSlots * kQRows * kQSlotRow;
static_assert(kQTileT * kOutStride * 2 <= kQShared, "the epilogue staging fits");

template <bool kFp4>
struct QFormat {
    // k steps per scale slot; bytes and k steps of one scale piece per row;
    // values per scale.
    static constexpr int kSlotSteps = kFp4 ? 4 : 16;
    static constexpr int kPiece = kFp4 ? 8 : 4;
    static constexpr int kPieceSteps = kFp4 ? 2 : 4;
    static constexpr int kGroup = kFp4 ? 16 : 32;
};

// Byte offset of 16-byte unit `u` (0 to 3) of row `row` in a stage.
__device__ __forceinline__ int q_offset(int row, int u) {
    return row * kQRowBytes + ((u ^ (row >> 1 & 3)) << 4);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// Block-scaled FP4 and FP8 MMAs with F32 accumulation. Lanes t = 0 and 1 of
// each quad pass the A scales of rows g and g + 8. The B scales of column g
// come from lane `kThreadB` of each quad (NVFP4: four E4M3 bytes, one per 16
// values), or from byte `kByte` of every lane's register (MXFP8: one E8M0
// byte per 32 values, A scales too).
template <int kThreadB>
__device__ __forceinline__ void mma_fp4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                            uint32_t b1, uint32_t scale_a, uint32_t scale_b) {
#if SIPHON_SM89
    __trap();
#else
    asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1."
        "e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
        "{%13}, {%14,%15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(scale_a),
          "h"(uint16_t(0)), "h"(uint16_t(0)), "r"(scale_b), "h"(uint16_t(0)),
          "h"(uint16_t(kThreadB)));
#endif
}

template <int kByte>
__device__ __forceinline__ void mma_fp8(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                            uint32_t b1, uint32_t scale_a, uint32_t scale_b) {
#if SIPHON_SM89
    __trap();
#else
    asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3."
        "e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
        "{%13}, {%14,%15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(scale_a),
          "h"(uint16_t(kByte)), "h"(uint16_t(0)), "r"(scale_b), "h"(uint16_t(kByte)),
          "h"(uint16_t(0)));
#endif
}

// One CTA of the quantized GEMM: 128 output columns of one tile of up to 128
// pairs of one expert. `xq`, `xq_scales` and `xq_outer` hold the quantized
// inputs; `xq_outer` (NVFP4 only) is each input row's outer scale.
template <bool kFp4, bool kSwiglu>
__device__ void expert_gemm_q(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xq_scales,
                              const float* __restrict__ xq_outer, const uint8_t* __restrict__ weight,
                              const uint8_t* __restrict__ scales,
                              const float* __restrict__ global_scale,
                              __nv_bfloat16* __restrict__ out, const int* __restrict__ perm,
                              const int4* __restrict__ tiles, const int* __restrict__ group_counts,
                              int n, int k, int inputs_per_row, float gate_softcap,
                              float up_softcap) {
    using F = QFormat<kFp4>;
    const int tile_count = group_counts[0];
    const int n0 = blockIdx.x * kTileN;
    const int tid = threadIdx.x;
    if (int(blockIdx.y) == tile_count) {
        nan_rows<kSwiglu>(out, perm, group_counts, n, n0);
        return;
    }
    if (int(blockIdx.y) > tile_count) {
        return;
    }
    const int4 tile = tiles[blockIdx.y];
    const int expert = tile.x;
    const int start = tile.y;
    const int count = tile.z;

    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* const slots = smem + kQStages * kQStageBytes;
    __shared__ float pair_outer[kQTileT];

    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    // This warp's 32 columns and 64 pairs.
    const int col0 = warp % 4 * 32;
    const int pair0 = warp / 4 * 64;

    const size_t row_bytes = kFp4 ? k / 2 : k;
    const size_t scale_row_bytes = k / F::kGroup;
    auto input_of = [&](int pair) {
        const int entry = perm[start + (pair < count ? pair : 0)];
        return inputs_per_row == 1 ? entry / kSlots : entry;
    };

    // Each thread copies unit tid % 4 of weight rows tid / 4 and tid / 4 + 64,
    // and of pairs tid / 4 and tid / 4 + 64, per stage, and the scale pieces of
    // row tid (weight rows, then pairs).
    const int copy_unit = tid % 4;
    const uint8_t* weight_src[2];
    const uint8_t* input_src[2];
    uint32_t weight_dst[2];
    uint32_t input_dst[2];
    int weight_bytes[2];
    int input_bytes[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int row = tid / 4 + 64 * i;
        const bool valid = n0 + row < n;
        weight_bytes[i] = valid ? 16 : 0;
        weight_src[i] = weight + (size_t(expert) * n + n0 + (valid ? row : 0)) * row_bytes + copy_unit * 16;
        weight_dst[i] = q_offset(row, copy_unit);
        input_src[i] = xq + size_t(input_of(row)) * row_bytes + copy_unit * 16;
        input_dst[i] = q_offset(kTileN + row, copy_unit);
        input_bytes[i] = row < count ? 16 : 0;
    }
    const uint8_t* scale_src =
        tid < kTileN ? scales + (size_t(expert) * n + n0 + (n0 + tid < n ? tid : 0)) * scale_row_bytes
                     : xq_scales + size_t(input_of(tid - kTileN)) * scale_row_bytes;
    if (kFp4 && tid < kQTileT) {
        pair_outer[tid] = xq_outer[input_of(tid)];
    }

    auto load_stage = [&](int index, int stage) {
        uint8_t* base = smem + stage * kQStageBytes;
        const int step = index * kQStageSteps;
        if (step % F::kPieceSteps == 0) {
            const int piece = step / F::kPieceSteps;
            uint8_t* dst = slots + step / F::kSlotSteps % kQSlots * (kQRows * kQSlotRow) +
                           tid * kQSlotRow + step % F::kSlotSteps / F::kPieceSteps * F::kPiece;
            cp_async_small<F::kPiece>(dst, scale_src + piece * F::kPiece);
        }
        const int offset = index * kQRowBytes;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            cp_async16(base + weight_dst[i], weight_src[i] + offset, weight_bytes[i]);
            cp_async16(base + input_dst[i], input_src[i] + offset, input_bytes[i]);
        }
    };

    // MXFP8 only: L2 prefetch of the weights by the CTAs of an expert's first
    // tile, as in `expert_gemm`; lane 0 of warp w prefetches rows 16w to
    // 16w + 15. It took 591 us against 657 us without for MXFP8 `w13`, but
    // 342 us against 334 us for NVFP4, whose weights take half the bytes.
    constexpr int kSegmentBytes = 256;
    constexpr int kSegmentStages = kSegmentBytes / kQRowBytes;
    const bool prefetch = !kFp4 && (blockIdx.y == 0 || tiles[blockIdx.y - 1].x != expert) &&
                          n0 + warp * 16 < n && lane == 0;
    const size_t warp_first_row = size_t(expert) * n + n0 + warp * 16;
    auto prefetch_segment = [&](int segment) {
        const int offset = segment * kSegmentBytes;
        if (prefetch && offset < int(row_bytes)) {
            const uint32_t bytes = min(kSegmentBytes, int(row_bytes) - offset);
            for (int r = 0; r < 16; ++r) {
                prefetch_l2_bulk(weight + (warp_first_row + r) * row_bytes + offset, bytes);
            }
        }
    };
    if (prefetch) {
        prefetch_l2_bulk(scales + warp_first_row * scale_row_bytes,
                         uint32_t(16 * scale_row_bytes));
    }
    prefetch_segment(0);
    prefetch_segment(1);

    const int stages = int(row_bytes) / kQRowBytes;
    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                acc[i][j][v] = 0.f;
            }
#pragma unroll
    for (int s = 0; s < kQStages - 1; ++s) {
        if (s < stages) {
            load_stage(s, s);
        }
        cp_async_commit();
    }

    const uint32_t smem0 = smem_addr(smem);
    // ldmatrix rows: A rows col0 + 16i + lane % 8 + 8 (lane / 8 % 2), unit
    // 2 sub + lane / 16; B pairs pair0 + 16q + lane % 8 + 8 (lane / 16), unit
    // 2 sub + lane / 8 % 2, for k step `sub` of a stage.
    const int a_row = col0 + lane % 8 + 8 * (lane / 8 % 2);
    const int b_pair = kTileN + pair0 + lane % 8 + 8 * (lane / 16);
    // The weight rows whose scales this lane passes: lanes t = 0 and 1 of each
    // quad give rows g and g + 8 of each 16-row fragment.
    const int scale_row = col0 + g + 8 * (t & 1);
    // Pair whose scale row this lane addresses for ldmatrix: lane l gives row
    // l % 8 of matrix l / 8, pairs 8m to 8m + 7 of each 32.
    const int scale_pair = kTileN + pair0 + lane;

    // One k step, `step`, from stage buffer `stage`, unit pair `sub` of it.
    auto compute = [&]<int kFragments, int kSub>(int step, int stage, int sub) {
        const uint32_t base = smem0 + stage * kQStageBytes;
        const uint8_t* slot = slots + step / F::kSlotSteps % kQSlots * (kQRows * kQSlotRow);
        uint32_t a[2][4];
        ldmatrix_x4(a[0], base + q_offset(a_row, 2 * sub + lane / 16));
        ldmatrix_x4(a[1], base + q_offset(a_row + 16, 2 * sub + lane / 16));
        uint32_t sa[2];
        // The scale word of this step: NVFP4 steps take one each, MXFP8 steps
        // one byte of word step % 16 / 4.
        const int word = kFp4 ? step % 4 : step % 16 / 4;
        sa[0] = *reinterpret_cast<const uint32_t*>(slot + scale_row * kQSlotRow + 4 * word);
        sa[1] = *reinterpret_cast<const uint32_t*>(slot + (scale_row + 16) * kQSlotRow + 4 * word);
        // B scales: NVFP4 lane (g, t) loads step t of the slot for pairs 8m + g
        // (ldmatrix, 4 matrices per 32 pairs) and the MMA takes them from lane
        // kSub; MXFP8 loads word `word` of pair 8j + g and takes byte kSub.
        uint32_t sb[8];
        if constexpr (kFp4) {
            uint32_t r[4];
            ldmatrix_x4(r, smem_addr(slot) + scale_pair * kQSlotRow);
            sb[0] = r[0]; sb[1] = r[1]; sb[2] = r[2]; sb[3] = r[3];
            if constexpr (kFragments > 4) {
                ldmatrix_x4(r, smem_addr(slot) + (scale_pair + 32) * kQSlotRow);
                sb[4] = r[0]; sb[5] = r[1]; sb[6] = r[2]; sb[7] = r[3];
            }
        } else {
#pragma unroll
            for (int j = 0; j < kFragments; ++j) {
                sb[j] = *reinterpret_cast<const uint32_t*>(
                    slot + (kTileN + pair0 + 8 * j + g) * kQSlotRow + 4 * word);
            }
        }
#pragma unroll
        for (int q = 0; q < (kFragments + 1) / 2; ++q) {
            uint32_t b[4];
            ldmatrix_x4(b, base + q_offset(b_pair + 16 * q, 2 * sub + lane / 8 % 2));
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int j = 2 * q + h;
                if (j < kFragments) {
#pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        if constexpr (kFp4) {
                            mma_fp4<kSub>(acc[i][j], a[i], b[2 * h], b[2 * h + 1], sa[i], sb[j]);
                        } else {
                            mma_fp8<kSub>(acc[i][j], a[i], b[2 * h], b[2 * h + 1], sa[i], sb[j]);
                        }
                    }
                }
            }
        }
    };

    // Pair fragments of this warp holding at least one real pair.
    const int fragments = min(max((count - pair0 + 7) / 8, 0), 8);
    auto main_loop = [&]<int kFragments>() {
        for (int index0 = 0; index0 < stages; index0 += kQStages) {
#pragma unroll
            for (int stage = 0; stage < kQStages; ++stage) {
                const int index = index0 + stage;
                if (index < stages) {
                    cp_async_wait<kQStages - 2>();
                    __syncthreads();
                    const int next = index + kQStages - 1;
                    if (next < stages) {
                        load_stage(next, (stage + kQStages - 1) % kQStages);
                    }
                    cp_async_commit();
                    if (index % kSegmentStages == 0 && index > 0) {
                        prefetch_segment(index / kSegmentStages + 1);
                    }
                    if constexpr (kFragments > 0) {
                        // A step's selector is step % 4 = (2 stage + sub) % 4,
                        // since index0 is even; the switch folds away once the
                        // loop unrolls.
                        const int step = index * kQStageSteps;
                        if (stage % 2 == 0) {
                            compute.template operator()<kFragments, 0>(step, stage, 0);
                            compute.template operator()<kFragments, 1>(step + 1, stage, 1);
                        } else {
                            compute.template operator()<kFragments, 2>(step, stage, 0);
                            compute.template operator()<kFragments, 3>(step + 1, stage, 1);
                        }
                    }
                }
            }
        }
    };
    static_assert(kQStages % 2 == 0 && kQStageSteps == 2,
                  "the selectors take step % 4 from the stage");
    switch (fragments) {
        case 0: main_loop.template operator()<0>(); break;
        case 1: main_loop.template operator()<1>(); break;
        case 2: main_loop.template operator()<2>(); break;
        case 3: main_loop.template operator()<3>(); break;
        case 4: main_loop.template operator()<4>(); break;
        case 5: main_loop.template operator()<5>(); break;
        case 6: main_loop.template operator()<6>(); break;
        case 7: main_loop.template operator()<7>(); break;
        default: main_loop.template operator()<8>(); break;
    }
    cp_async_wait<0>();
    __syncthreads();

    // Stage the BF16 results as [pair][column], then write them out. The
    // output scale is the weights' global scale times each input row's outer
    // scale (NVFP4), or 1 (MXFP8).
    __nv_bfloat16* staged = reinterpret_cast<__nv_bfloat16*>(smem);
    float w_scale = 1.f;
    if constexpr (kFp4) {
        w_scale = __fdiv_rn(global_scale[expert], kNvfp4ScaleEnc);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int pair = pair0 + j * 8 + 2 * t;
        float s0 = 1.f;
        float s1 = 1.f;
        if constexpr (kFp4) {
            s0 = __fmul_rn(pair_outer[pair], w_scale);
            s1 = __fmul_rn(pair_outer[pair + 1], w_scale);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int col = col0 + 16 * i + g;
            staged[pair * kOutStride + col] = __float2bfloat16(acc[i][j][0] * s0);
            staged[(pair + 1) * kOutStride + col] = __float2bfloat16(acc[i][j][1] * s1);
            staged[pair * kOutStride + col + 8] = __float2bfloat16(acc[i][j][2] * s0);
            staged[(pair + 1) * kOutStride + col + 8] = __float2bfloat16(acc[i][j][3] * s1);
        }
    }
    __syncthreads();
    write_tile<kSwiglu>(staged, out, perm + start, count, n, n0, gate_softcap, up_softcap);
}

// ---------------------------------------------------------------------------
// FP8 activations for the NVFP4 experts (opt-in): E2M1 weights with their E4M3
// scale per 16 values, against MXFP8 inputs (E4M3 values, a UE8M0 scale per 32;
// `quantize_rows_fp8`), on the block-scaled FP8 MMA (`kind::mxf8f6f4`,
// m16n8k32), whose scales are UE8M0 per 32 values.
//
// An E2M1 value times its E4M3 scale has up to 6 significant bits, more than
// E4M3's 4. So each weight w is split into two E4M3 values, w = 2^e (hi +
// lo / 16), where 2^e, one per weight row and 32 values, is the larger of the
// two group scales' powers of two divided by 32, hi is w / 2^e truncated to 4
// significant bits, and lo the rest times 16. Each k step of 32 runs one MMA
// on hi with A scale 2^e and one on lo with A scale 2^(e - 4), both
// accumulating in F32, with the input's UE8M0 scale as B scale. The split is
// exact while the two group scales of a block are within a factor of 2^10 of
// each other (as E4M3 exponents, a subnormal counting as 2^-6), so that every
// part stays inside E4M3's normal range; `nvfp4_scale_spread` checks it. The
// checkpoint's largest ratio is 2^5. Every product is then exact, and the sums
// are F32, as for the BF16 inputs.
//
// Write a scale as M 2^(f - 10), with M its 4-bit significand (0 to 15) and f
// its exponent field (a subnormal taking field 1), and let F be the larger
// field of the block's two scales. Then w / 2^e = v M 2^(2 - (F - f)) for the
// E2M1 magnitude v, with e = F - 12. hi and lo bytes thus depend only on v, M
// and the gap F - f, so each CTA first builds a table of them in shared memory:
// for each M and gap, 8 hi and 8 lo bytes indexed by the E2M1 magnitude. A
// lane decodes its 8 values with `prmt` lookups into its row's entry and adds
// each nibble's sign.
//
// Tiles of 128 weight rows by 128 pairs as in `expert_gemm_q`, two CTAs per
// SM, but each of the 8 warps takes 16 columns and all 128 pairs, so that each
// weight is decoded once. A stage holds 64 k values: 32 bytes of each weight
// row and 64 of each input row, in 3 stages. Lane (g, t) loads
// bytes 4t..4t+3 of a weight row's 16-byte unit, values 8t..8t+7 of the k step,
// all of one group (t / 2), and decodes them into A registers a0 (values
// 8t..8t+3) and a2 (8t+4..8t+7) of rows g and g + 8. `quantize_rows_fp8`
// stores each input block so that the same lane's B registers hold the same
// values.
//
// Scales load into the ring of `expert_gemm_q`: a slot of 16 bytes per row
// covers 4 stages (256 values): 16 E4M3 weight scales, or 8 UE8M0 input scales
// in its first 8 bytes. Pieces of 2 stages load with every even stage.
constexpr int kQ8Stages = 3;
constexpr int kQ8WeightRow = 32;
constexpr int kQ8InputRow = 64;
constexpr int kQ8Weights = kTileN * kQ8WeightRow;
constexpr int kQ8StageBytes = kQ8Weights + kQTileT * kQ8InputRow;
// Largest exponent difference of the two E4M3 scales of a 32-value block for
// which the split is exact, and the decode table's entries: 16 significands
// by gaps 0 to kQ8MaxSpread, 16 bytes each.
constexpr int kQ8MaxSpread = 10;
constexpr int kQ8TableEntries = 16 * (kQ8MaxSpread + 1);
constexpr int kQ8Table = kQ8Stages * kQ8StageBytes + kQSlots * kQRows * kQSlotRow;
constexpr int kQ8Shared = kQ8Table + kQ8TableEntries * 16;
static_assert(kQTileT * kOutStride * 2 <= kQ8Table, "the epilogue staging fits");

// Byte offsets in a stage of 16-byte unit `u` of weight row `row` (u < 2) and of
// input row `pair` (u < 4), swizzled so that the 8 rows one `ldmatrix` reads
// fall in distinct banks.
__device__ __forceinline__ int q8_weight_offset(int row, int u) {
    return row * kQ8WeightRow + ((u ^ (row >> 2 & 1)) << 4);
}

__device__ __forceinline__ int q8_input_offset(int pair, int u) {
    return kQ8Weights + pair * kQ8InputRow + ((u ^ (pair >> 1 & 3)) << 4);
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t selector) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(selector));
    return d;
}

// The decode table entry of significand M and gap d, as thread `M (kQ8MaxSpread
// + 1) + d` builds it: for each E2M1 magnitude v (0, 0.5, 1, 1.5, 2, 3, 4, 6),
// the E4M3 bytes of hi = v M 2^(2 - d) truncated to 4 significant bits (bytes
// 0-7) and of lo = 16 (v M 2^(2 - d) - hi) (bytes 8-15). Both are exact: the
// product has at most 6 significant bits and lies in E4M3's normal range for
// d up to kQ8MaxSpread when M is 8 or more (a normal scale).
__device__ void q8_table_entry(uint8_t* entry, int index) {
    const int m = index / (kQ8MaxSpread + 1);
    const int d = index % (kQ8MaxSpread + 1);
    const float magnitudes[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const float w = magnitudes[v] * float(m) * exp2f(float(2 - d));
        // Truncated to 3 mantissa bits.
        const float hi = __uint_as_float(__float_as_uint(w) & 0xfff00000u);
        entry[v] = __nv_fp8_e4m3(hi).__x;
        entry[8 + v] = __nv_fp8_e4m3((w - hi) * 16.f).__x;
    }
}

// E4M3 exponent field of a scale byte, a subnormal taking field 1.
__device__ __forceinline__ uint32_t scale_field(uint32_t byte) {
    return max((byte & 0x7fu) >> 3, 1u);
}

// The 8 E2M1 weights of one lane's row (a word, low nibble first) through the
// decode table entry {hi0, hi1, lo0, lo1}: hi of values 0-3 in `hi[0]`, 4-7 in
// `hi[1]`, and lo.
__device__ __forceinline__ void split_e2m1x8(uint32_t x, uint4 entry, uint32_t (&hi)[2],
                                             uint32_t (&lo)[2]) {
    const uint32_t magnitude = x & 0x77777777u;
    // Byte i of `up` has the sign of nibble 2i in its top bit, byte i of x that
    // of nibble 2i + 1; prmt's sign mode spreads them to bytes.
    const uint32_t up = x << 4;
    const uint32_t sign0 = prmt(up, x, 0xd9c8u) & 0x80808080u;
    const uint32_t sign1 = prmt(up, x, 0xfbeau) & 0x80808080u;
    hi[0] = prmt(entry.x, entry.y, magnitude) | sign0;
    hi[1] = prmt(entry.x, entry.y, magnitude >> 16) | sign1;
    lo[0] = prmt(entry.z, entry.w, magnitude) | sign0;
    lo[1] = prmt(entry.z, entry.w, magnitude >> 16) | sign1;
}

// c += a b, block-scaled FP8: the A scales of rows g and g + 8 from byte 0 of
// lanes 4g and 4g + 1's `scale_a`, the B scales of column g from byte 0 of lane
// 4g's `scale_b`.
__device__ __forceinline__ void mma_fp8_scaled(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                               uint32_t b1, uint32_t scale_a, uint32_t scale_b) {
#if SIPHON_SM89
    __trap();
#else
    asm("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3."
        "e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, "
        "{%11,%11}, {%12}, {%11,%11};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(scale_a),
          "h"(uint16_t(0)), "r"(scale_b));
#endif
}

// One CTA of the FP8-activation NVFP4 GEMM: 128 output columns of one tile of
// up to 128 pairs of one expert. `xq` and `xq_scales` come from
// `quantize_rows_fp8`.
template <bool kSwiglu>
__device__ void expert_gemm_q8(const uint8_t* __restrict__ xq, const uint8_t* __restrict__ xq_scales,
                               const uint8_t* __restrict__ weight, const uint8_t* __restrict__ scales,
                               const float* __restrict__ global_scale,
                               __nv_bfloat16* __restrict__ out, const int* __restrict__ perm,
                               const int4* __restrict__ tiles, const int* __restrict__ group_counts,
                               int n, int k, int inputs_per_row, float gate_softcap,
                               float up_softcap) {
    const int tile_count = group_counts[0];
    const int n0 = blockIdx.x * kTileN;
    const int tid = threadIdx.x;
    if (int(blockIdx.y) == tile_count) {
        nan_rows<kSwiglu>(out, perm, group_counts, n, n0);
        return;
    }
    if (int(blockIdx.y) > tile_count) {
        return;
    }
    const int4 tile = tiles[blockIdx.y];
    const int expert = tile.x;
    const int start = tile.y;
    const int count = tile.z;

    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* const slots = smem + kQ8Stages * kQ8StageBytes;
    const uint4* const table = reinterpret_cast<const uint4*>(smem + kQ8Table);
    // The main loop's first barrier publishes the table.
    if (tid < kQ8TableEntries) {
        q8_table_entry(smem + kQ8Table + 16 * tid, tid);
    }

    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    // This warp's 16 columns; it takes all 128 pairs.
    const int col0 = warp * 16;

    const size_t weight_row_bytes = k / 2;
    auto input_of = [&](int pair) {
        const int entry = perm[start + (pair < count ? pair : 0)];
        return inputs_per_row == 1 ? entry / kSlots : entry;
    };

    // Each thread copies unit tid % 2 of weight row tid / 2, unit tid % 4 of
    // pairs tid / 4 and tid / 4 + 64, and the scale pieces of row tid (weight
    // rows, then pairs).
    const int weight_row = tid / 2;
    const bool weight_valid = n0 + weight_row < n;
    const uint8_t* weight_src =
        weight + (size_t(expert) * n + n0 + (weight_valid ? weight_row : 0)) * weight_row_bytes +
        tid % 2 * 16;
    const uint32_t weight_dst = q8_weight_offset(weight_row, tid % 2);
    const int weight_bytes = weight_valid ? 16 : 0;
    const uint8_t* input_src[2];
    uint32_t input_dst[2];
    int input_bytes[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int pair = tid / 4 + 64 * i;
        input_src[i] = xq + size_t(input_of(pair)) * k + tid % 4 * 16;
        input_dst[i] = q8_input_offset(pair, tid % 4);
        input_bytes[i] = pair < count ? 16 : 0;
    }
    const bool weight_scales = tid < kTileN;
    const uint8_t* scale_src =
        weight_scales
            ? scales + (size_t(expert) * n + n0 + (n0 + tid < n ? tid : 0)) * (k / 16)
            : xq_scales + size_t(input_of(tid - kTileN)) * (k / 32);

    auto load_stage = [&](int index, int stage) {
        uint8_t* base = smem + stage * kQ8StageBytes;
        if (index % 2 == 0) {
            uint8_t* dst = slots + index / 4 % kQSlots * (kQRows * kQSlotRow) + tid * kQSlotRow;
            const int half = index % 4 / 2;
            if (weight_scales) {
                cp_async_small<8>(dst + 8 * half, scale_src + index / 2 * 8);
            } else {
                cp_async_small<4>(dst + 4 * half, scale_src + index / 2 * 4);
            }
        }
        cp_async16(base + weight_dst, weight_src + index * kQ8WeightRow, weight_bytes);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            cp_async16(base + input_dst[i], input_src[i] + index * kQ8InputRow, input_bytes[i]);
        }
    };

    const int stages = k / 64;
    float acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            acc[j][v] = 0.f;
        }
#pragma unroll
    for (int s = 0; s < kQ8Stages - 1; ++s) {
        if (s < stages) {
            load_stage(s, s);
        }
        cp_async_commit();
    }

    const uint32_t smem0 = smem_addr(smem);
    // ldmatrix rows: weight rows col0 + lane % 8 + 8 (lane / 8 % 2), unit lane /
    // 16, both k steps of the stage; input pairs 16q + lane % 8 + 8 (lane / 16),
    // unit 2s + lane / 8 % 2.
    const int a_row = col0 + lane % 8 + 8 * (lane / 8 % 2);
    const int b_pair = lane % 8 + 8 * (lane / 16);
    // The scale byte of this lane's group within a pair of group scales.
    const int group_shift = t < 2 ? 0 : 8;

    // One stage: 2 k steps.
    auto compute = [&]<int kFragments>(int index, int stage) {
        const uint32_t base = smem0 + stage * kQ8StageBytes;
        const uint8_t* slot = slots + index / 4 % kQSlots * (kQRows * kQSlotRow);
        uint32_t r[4];
        ldmatrix_x4(r, base + q8_weight_offset(a_row, lane / 16));
        // The stage's 4 weight scales of rows g and g + 8.
        uint32_t scale_words[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            scale_words[h] = *reinterpret_cast<const uint32_t*>(
                slot + (col0 + g + 8 * h) * kQSlotRow + index % 4 * 4);
        }
        const uint8_t* input_scales = slot + (kTileN + g) * kQSlotRow + index % 4 * 2;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            // hi and lo A fragments and A scales for k step s.
            uint32_t a_hi[4];
            uint32_t a_lo[4];
            uint32_t field[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                // The two group scales of row g + 8h for this k step, the larger
                // field, and this lane's scale's significand and gap.
                const uint32_t pair = scale_words[h] >> (16 * s);
                field[h] = max(scale_field(pair & 0xffu), scale_field(pair >> 8 & 0xffu));
                const uint32_t scale = pair >> group_shift & 0x7fu;
                const uint32_t m = (scale & 7u) | (scale >= 8u ? 8u : 0u);
                const uint32_t gap = min(field[h] - scale_field(scale), uint32_t(kQ8MaxSpread));
                uint32_t hi[2];
                uint32_t lo[2];
                split_e2m1x8(r[2 * s + h], table[m * (kQ8MaxSpread + 1) + gap], hi, lo);
                a_hi[h] = hi[0];
                a_hi[2 + h] = hi[1];
                a_lo[h] = lo[0];
                a_lo[2 + h] = lo[1];
            }
            // 2^e = 2^(field - 7) / 32 as UE8M0, for row g (lane t = 0) or g + 8
            // (lane t = 1); lo carries a factor of 16.
            const uint32_t scale_hi = (t & 1 ? field[1] : field[0]) + 127 - 12;
            const uint32_t scale_lo = scale_hi - 4;
#pragma unroll
            for (int q = 0; q < kFragments / 2; ++q) {
                uint32_t b[4];
                ldmatrix_x4(b, base + q8_input_offset(b_pair + 16 * q, 2 * s + lane / 8 % 2));
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int j = 2 * q + h;
                    // The input scale of pair 8j + g.
                    const uint32_t scale_b = input_scales[8 * j * kQSlotRow + s];
                    mma_fp8_scaled(acc[j], a_hi, b[2 * h], b[2 * h + 1], scale_hi, scale_b);
                    mma_fp8_scaled(acc[j], a_lo, b[2 * h], b[2 * h + 1], scale_lo, scale_b);
                }
            }
        }
    };

    // Pair fragments holding at least one real pair, rounded up to a multiple
    // of 4; the extra fragments multiply zero inputs. (Multiples of 2 ran
    // slower.)
    const int fragments = (count + 7) / 8;
    auto main_loop = [&]<int kFragments>() {
        for (int index0 = 0; index0 < stages; index0 += kQ8Stages) {
#pragma unroll
            for (int stage = 0; stage < kQ8Stages; ++stage) {
                const int index = index0 + stage;
                if (index < stages) {
                    cp_async_wait<kQ8Stages - 2>();
                    __syncthreads();
                    const int next = index + kQ8Stages - 1;
                    if (next < stages) {
                        load_stage(next, (stage + kQ8Stages - 1) % kQ8Stages);
                    }
                    cp_async_commit();
                    compute.template operator()<kFragments>(index, stage);
                }
            }
        }
    };
    if (fragments <= 4) {
        main_loop.template operator()<4>();
    } else if (fragments <= 8) {
        main_loop.template operator()<8>();
    } else if (fragments <= 12) {
        main_loop.template operator()<12>();
    } else {
        main_loop.template operator()<16>();
    }
    cp_async_wait<0>();
    __syncthreads();

    __nv_bfloat16* staged = reinterpret_cast<__nv_bfloat16*>(smem);
    const float w_scale = __fdiv_rn(global_scale[expert], kNvfp4ScaleEnc);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int pair = j * 8 + 2 * t;
        const int col = col0 + g;
        staged[pair * kOutStride + col] = __float2bfloat16(acc[j][0] * w_scale);
        staged[(pair + 1) * kOutStride + col] = __float2bfloat16(acc[j][1] * w_scale);
        staged[pair * kOutStride + col + 8] = __float2bfloat16(acc[j][2] * w_scale);
        staged[(pair + 1) * kOutStride + col + 8] = __float2bfloat16(acc[j][3] * w_scale);
    }
    __syncthreads();
    write_tile<kSwiglu>(staged, out, perm + start, count, n, n0, gate_softcap, up_softcap);
}

}  // namespace

// Groups the `entries` (row, slot) pairs of `indices` by expert.
//
// Writes `perm[i]`, the flat index `row * 8 + slot` of the i-th pair in expert
// order, and `tiles[i] = {expert, first position in perm, pairs, 0}` for each
// run of at most `tile_pairs` (64 or 128) pairs of one expert, ordered by
// expert. Pairs with an id outside `[0, experts)` come last in `perm`, in no
// particular order, and get no tile. `group_counts` receives the number of
// tiles, the number of pairs with known ids, and `entries`. One block of 1024
// threads; at most 16,384 pairs and 1024 experts.
extern "C" __global__ void __launch_bounds__(kGroupThreads)
    expert_group(const int32_t* __restrict__ indices, int entries, int experts, int tile_pairs,
                 int32_t* __restrict__ perm, int4* __restrict__ tiles,
                 int32_t* __restrict__ group_counts) {
    __shared__ int count[kMaxExperts];
    __shared__ int first[kMaxExperts];
    __shared__ int warp_sums[32];
    __shared__ int invalid_count;
    __shared__ int valid_count;
    const int tid = threadIdx.x;

    // Each thread keeps its pairs' ids in registers; -1 marks no pair.
    int ids[kGroupRounds];
#pragma unroll
    for (int r = 0; r < kGroupRounds; ++r) {
        const int i = r * kGroupThreads + tid;
        ids[r] = i < entries ? indices[i] : -1;
    }
    for (int e = tid; e < kMaxExperts; e += kGroupThreads) {
        count[e] = 0;
    }
    if (tid == 0) {
        invalid_count = 0;
    }
    __syncthreads();

    // Count. Each pair's rank among its expert's pairs, or among the pairs
    // with unknown ids, is the count before its own atomic; the order of ranks
    // varies between runs, but each output depends only on its own pair.
    int rank[kGroupRounds];
#pragma unroll
    for (int r = 0; r < kGroupRounds; ++r) {
        const int i = r * kGroupThreads + tid;
        const bool valid = ids[r] >= 0 && ids[r] < experts;
        if (valid) {
            rank[r] = atomicAdd(&count[ids[r]], 1);
        } else if (i < entries) {
            rank[r] = atomicAdd(&invalid_count, 1);
        }
    }
    __syncthreads();

    // Exclusive prefix sums of pairs and tiles, one expert per thread.
    int pairs = 0;
    int expert_tiles = 0;
    if (tid < experts) {
        pairs = count[tid];
        expert_tiles = (pairs + tile_pairs - 1) / tile_pairs;
    }
    const int pair_end = block_scan(pairs, warp_sums);
    if (tid == kGroupThreads - 1) {
        valid_count = pair_end;
    }
    __syncthreads();
    const int tile_end = block_scan(expert_tiles, warp_sums);
    const int pair_start = pair_end - pairs;
    const int tile_start = tile_end - expert_tiles;
    if (tid < experts) {
        first[tid] = pair_start;
        for (int j = 0; j < expert_tiles; ++j) {
            tiles[tile_start + j] =
                make_int4(tid, pair_start + j * tile_pairs, min(tile_pairs, pairs - j * tile_pairs), 0);
        }
    }
    if (tid == kGroupThreads - 1) {
        group_counts[0] = tile_end;
        group_counts[1] = pair_end;
        group_counts[2] = entries;
    }
    __syncthreads();

    // Scatter.
#pragma unroll
    for (int r = 0; r < kGroupRounds; ++r) {
        const int i = r * kGroupThreads + tid;
        if (ids[r] >= 0 && ids[r] < experts) {
            perm[first[ids[r]] + rank[r]] = i;
        } else if (i < entries) {
            perm[valid_count + rank[r]] = i;
        }
    }
}

// Arguments, shared by the GEMM kernels:
// - `x`, `weight`, `scales`, `global_scale`, `out`, `inputs_per_row`: as for
//   `expert_matvec_*` in experts.cu;
// - `perm`, `tiles`, `group_counts`: written by `expert_group`;
// - `gate_softcap`, `up_softcap`: the SwiGLU caps of the `_swiglu` kernels,
//   which write `[pairs, n / 2]`; the others ignore them.
// Grid: (ceil(n / 128), an upper bound on the tile count plus one); the row
// of blocks at the tile count writes the NaN rows, and later rows exit. Needs
// `gemm_shared_bytes` of dynamic shared memory: 38,912 bytes for NVFP4 and
// 50,176 for MXFP8. n must be a multiple of 16 and k of 128.
#define EXPERT_GEMM(name, fp4, stages, swiglu)                                                    \
    static_assert(gemm_shared_bytes<fp4, stages>() <= 50176);                                     \
    extern "C" __global__ void __launch_bounds__(kThreads, 2)                                     \
        name(const __nv_bfloat16* x, const uint8_t* weight, const uint8_t* scales,                \
             const float* global_scale, __nv_bfloat16* out, const int* perm, const int4* tiles,   \
             const int* group_counts, int n, int k, int inputs_per_row, float gate_softcap,       \
             float up_softcap) {                                                                  \
        expert_gemm<fp4, stages, swiglu>(x, weight, scales, global_scale, out, perm, tiles,       \
                                         group_counts, n, k, inputs_per_row, gate_softcap,        \
                                         up_softcap);                                             \
    }

EXPERT_GEMM(expert_gemm_nvfp4, true, 3, false)
EXPERT_GEMM(expert_gemm_mxfp8, false, 3, false)
EXPERT_GEMM(expert_gemm_nvfp4_swiglu, true, 3, true)
EXPERT_GEMM(expert_gemm_mxfp8_swiglu, false, 3, true)

// Quantizes `rows` BF16 rows of k values (`x`, `[rows, k]`) for the quantized
// GEMMs: `values` `[rows, k / 2]` (NVFP4) or `[rows, k]` (MXFP8) in the stored
// word order, `scales` `[rows, k / 16]` or `[rows, k / 32]`, and for NVFP4
// `outer` `[rows]`. One warp per row, 8 rows per block of 256 threads. k must be
// a multiple of 32.
extern "C" __global__ void __launch_bounds__(256)
    quantize_rows_nvfp4(const __nv_bfloat16* __restrict__ x, int rows, int k,
                        uint8_t* __restrict__ values, uint8_t* __restrict__ scales,
                        float* __restrict__ outer) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row < rows) {
        quantize_row_nvfp4(x + size_t(row) * k, k, values + size_t(row) * (k / 2),
                           scales + size_t(row) * (k / 16), outer + row);
    }
}

extern "C" __global__ void __launch_bounds__(256)
    quantize_rows_mxfp8(const __nv_bfloat16* __restrict__ x, int rows, int k,
                        uint8_t* __restrict__ values, uint8_t* __restrict__ scales,
                        float* __restrict__ outer) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row < rows) {
        quantize_row_mxfp8<false>(x + size_t(row) * k, k, values + size_t(row) * k,
                                  scales + size_t(row) * (k / 32));
    }
}

// `quantize_rows_mxfp8` with each block of 32 values in the order that
// `expert_gemm_q8_*` loads, for the NVFP4 experts with FP8 activations.
extern "C" __global__ void __launch_bounds__(256)
    quantize_rows_fp8(const __nv_bfloat16* __restrict__ x, int rows, int k,
                      uint8_t* __restrict__ values, uint8_t* __restrict__ scales,
                      float* __restrict__ outer) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row < rows) {
        quantize_row_mxfp8<true>(x + size_t(row) * k, k, values + size_t(row) * k,
                                 scales + size_t(row) * (k / 32));
    }
}

// The quantized GEMMs: `expert_gemm_*` with `quantize_rows_*` inputs (`xq`,
// `xq_scales`, `xq_outer`) in place of `x`, over a grouping with tiles of 128
// pairs. Needs `kQShared` (40,960) bytes of dynamic shared memory and k a
// multiple of 128.
#define EXPERT_GEMM_Q(name, fp4, swiglu)                                                          \
    static_assert(kQShared + kQTileT * 4 <= 50176);                                               \
    extern "C" __global__ void __launch_bounds__(kThreads, 2)                                     \
        name(const uint8_t* xq, const uint8_t* xq_scales, const float* xq_outer,                  \
             const uint8_t* weight, const uint8_t* scales, const float* global_scale,             \
             __nv_bfloat16* out, const int* perm, const int4* tiles, const int* group_counts,     \
             int n, int k, int inputs_per_row, float gate_softcap, float up_softcap) {            \
        expert_gemm_q<fp4, swiglu>(xq, xq_scales, xq_outer, weight, scales, global_scale, out,   \
                                   perm, tiles, group_counts, n, k, inputs_per_row,              \
                                   gate_softcap, up_softcap);                                    \
    }

EXPERT_GEMM_Q(expert_gemm_q_nvfp4, true, false)
EXPERT_GEMM_Q(expert_gemm_q_mxfp8, false, false)
EXPERT_GEMM_Q(expert_gemm_q_nvfp4_swiglu, true, true)
EXPERT_GEMM_Q(expert_gemm_q_mxfp8_swiglu, false, true)

// The NVFP4 GEMMs with FP8 activations: `expert_gemm_q_nvfp4` with
// `quantize_rows_fp8` inputs. Needs `kQ8Shared` (47,872) bytes of dynamic shared
// memory, k a multiple of 128, and weight scales that `nvfp4_scale_spread`
// passes.
#define EXPERT_GEMM_Q8(name, swiglu)                                                              \
    static_assert(kQ8Shared <= 50176);                                                            \
    extern "C" __global__ void __launch_bounds__(kThreads, 2)                                     \
        name(const uint8_t* xq, const uint8_t* xq_scales, const float* xq_outer,                  \
             const uint8_t* weight, const uint8_t* scales, const float* global_scale,             \
             __nv_bfloat16* out, const int* perm, const int4* tiles, const int* group_counts,     \
             int n, int k, int inputs_per_row, float gate_softcap, float up_softcap) {            \
        expert_gemm_q8<swiglu>(xq, xq_scales, weight, scales, global_scale, out, perm, tiles,     \
                               group_counts, n, k, inputs_per_row, gate_softcap, up_softcap);     \
    }

EXPERT_GEMM_Q8(expert_gemm_q8_nvfp4, false)
EXPERT_GEMM_Q8(expert_gemm_q8_nvfp4_swiglu, true)

// Counts the NVFP4 group scales (`scales`, `count` pairs of E4M3 bytes, one
// pair per 32-value block) for which `expert_gemm_q8_*` would not be exact
// into `*bad`: a nonzero scale whose gap to the block's larger exponent field
// passes kQ8MaxSpread, or whose smallest product, 0.5 M 2^(2 - gap), falls
// below E4M3's normal range (only possible for a subnormal scale).
extern "C" __global__ void __launch_bounds__(256)
    nvfp4_scale_spread(const uint16_t* __restrict__ scales, size_t count, unsigned* bad) {
    unsigned local = 0;
    for (size_t i = blockIdx.x * size_t(256) + threadIdx.x; i < count; i += size_t(gridDim.x) * 256) {
        const uint32_t pair = scales[i];
        const uint32_t field = max(scale_field(pair & 0xffu), scale_field(pair >> 8 & 0xffu));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t scale = pair >> (8 * h) & 0x7fu;
            const uint32_t m = (scale & 7u) | (scale >= 8u ? 8u : 0u);
            const int gap = int(field - scale_field(scale));
            // 0.5 M 2^(2 - gap) >= 2^-6, that is M 2^(7 - gap) >= 1.
            local += m != 0 && (gap > kQ8MaxSpread || (m << 7) < (1u << gap));
        }
    }
    if (local) {
        atomicAdd(bad, local);
    }
}
