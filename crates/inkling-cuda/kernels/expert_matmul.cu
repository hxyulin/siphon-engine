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
//    pairs of one expert. It also fills the output rows of ids outside
//    `[0, experts)` with NaN.
// 2. `expert_gemm_*` runs one CTA per (tile, 128 output columns). It streams
//    its weight and scale tile into L2 ahead of use. Then, 64 input values
//    at a time, it copies the raw weight bytes and the tile's BF16 inputs to
//    shared memory with `cp.async`, decodes the weights to BF16 with the scales
//    held in registers, and multiplies with BF16 `mma.sync.m16n8k16` into F32
//    accumulators.
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
// Largest pair count `expert_group` supports: 1024 rows of 8 slots.
constexpr int kMaxPairs = 8192;
constexpr int kGroupRounds = kMaxPairs / kGroupThreads;
// `F8E4M3_MAX * F4E2M1_MAX`, as in experts.cu.
constexpr float kNvfp4ScaleEnc = 448.f * 6.f;

__device__ uint32_t smem_addr(const void* p) {
    return uint32_t(__cvta_generic_to_shared(p));
}

// Copies `src_bytes` (0 or 16) and zero-fills the rest of 16 bytes.
__device__ void cp_async16(void* dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_addr(dst)), "l"(src),
                 "r"(src_bytes));
}

// Starts loading `bytes` (a multiple of 16, 16-byte aligned) into L2.
__device__ void prefetch_l2_bulk(const void* src, uint32_t bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src), "r"(bytes) : "memory");
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
};

// One CTA: 128 output columns of one tile of pairs. Each warp owns 16 output
// columns and all 64 pairs.
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
// Shared memory is `kStages * Stage::kBytes` dynamic bytes plus 256 static;
// two CTAs fit in an SM's 100 KiB when that is at most 49,920 dynamic bytes.
template <bool kFp4, int kStages>
__device__ void expert_gemm(const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ weight,
                            const uint8_t* __restrict__ scales, const float* __restrict__ global_scale,
                            __nv_bfloat16* __restrict__ out, const int* __restrict__ perm,
                            const int4* __restrict__ tiles, const int* __restrict__ tile_count,
                            int n, int k, int inputs_per_row) {
    using S = Stage<kFp4>;
    if (int(blockIdx.y) >= *tile_count) {
        return;
    }
    const int4 tile = tiles[blockIdx.y];
    const int expert = tile.x;
    const int start = tile.y;
    const int count = tile.z;
    const int n0 = blockIdx.x * kTileN;

    extern __shared__ __align__(16) uint8_t smem[];
    // Flat pair index of each pair. Pairs past `count` reuse the first pair's;
    // their inputs load as zeros and their results are dropped.
    __shared__ int pair_entry[kTileT];

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int g = lane / 4;
    const int t = lane % 4;

    if (tid < kTileT) {
        pair_entry[tid] = perm[start + (tid < count ? tid : 0)];
    }
    __syncthreads();

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
        const int entry = pair_entry[pair];
        const int input = inputs_per_row == 1 ? entry / kSlots : entry;
        input_src[i] = x + size_t(input) * k + part * 8;
        input_dst[i] = S::kWeight + pair * 128 + (part ^ (pair & 1)) * 16;
        input_bytes[i] = pair < count ? 16 : 0;
        input_load[i] = pair < fragments * 8;
    }
    const size_t scale_row_bytes = k / (kFp4 ? 16 : 32);

    auto load_stage = [&](int chunk, int stage) {
        uint8_t* base = smem + stage * S::kBytes;
        const int k0 = chunk * kTileK;
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

    // Weights stream into L2 ahead of the pipeline, 512 bytes of each row at
    // a time, one segment ahead of the one in use, so the copies to shared
    // memory wait on L2 rather than DRAM. The whole tile at once would not
    // fit: 340 resident MXFP8 blocks would ask for 100 MB of the 96 MB L2.
    // Lane 0 of each warp prefetches its 16 rows. (The bulk prefetch takes
    // uniform operands, so per-lane addresses would serialize.) The scales,
    // an eighth of the weight bytes or less, are prefetched whole.
    constexpr int kSegmentBytes = 512;
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
    // The lane's scale bytes for rows `row0` and `row0 + 8` in one chunk,
    // packed low and high. Its 16 values are NVFP4 group 4c + t or MXFP8 group
    // 2c + t / 2.
    const uint8_t* scale_lo = scales + (size_t(expert) * n + n0 + row0) * scale_row_bytes;
    const uint8_t* scale_hi = scale_lo + 8 * scale_row_bytes;
    auto load_scales = [&](int chunk) -> uint32_t {
        const int group = kFp4 ? 4 * chunk + t : 2 * chunk + t / 2;
        return uint32_t(__ldg(scale_lo + group)) | uint32_t(__ldg(scale_hi + group)) << 8;
    };

    // One chunk's MMAs over the first `kFragments` pair fragments.
    auto compute = [&]<int kFragments>(int stage, uint32_t scale_bytes) {
        const uint8_t* base = smem + stage * S::kBytes;

        // A fragments for the four 16-value steps: [step][register].
        uint32_t a[4][4];
        if constexpr (kFp4) {
            const __half2 s_lo =
                __half2half2(__half(__nv_cvt_fp8_to_halfraw(scale_bytes & 0xff, __NV_E4M3)));
            const __half2 s_hi =
                __half2half2(__half(__nv_cvt_fp8_to_halfraw(scale_bytes >> 8, __NV_E4M3)));
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
            const float s_lo = e8m0(scale_bytes & 0xff);
            const float s_hi = e8m0(scale_bytes >> 8);
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
        // Scale bytes for the chunks in the pipeline: `ring[stage]` holds those
        // of the chunk in that stage, loaded when its copies start.
        uint32_t ring[kStages];
#pragma unroll
        for (int s = 0; s < kStages - 1; ++s) {
            ring[s] = warp_active && s < chunks ? load_scales(s) : 0;
        }
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
                        if (warp_active) {
                            ring[(stage + kStages - 1) % kStages] = load_scales(next);
                        }
                    }
                    cp_async_commit();
                    if (chunk % kSegmentChunks == 0 && chunk > 0) {
                        prefetch_segment(chunk / kSegmentChunks + 1);
                    }
                    if (warp_active) {
                        compute.template operator()<kFragments>(stage, ring[stage]);
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
    constexpr int kOutStride = kTileN + 8;
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
    const int columns = min(kTileN, n - n0);
    for (int unit = tid; unit < count * (kTileN / 8); unit += kThreads) {
        const int pair = unit / (kTileN / 8);
        const int part = unit % (kTileN / 8);
        if (part * 8 < columns) {
            const uint4 v = *reinterpret_cast<const uint4*>(staged + pair * kOutStride + part * 8);
            *reinterpret_cast<uint4*>(out + size_t(pair_entry[pair]) * n + n0 + part * 8) = v;
        }
    }
}

}  // namespace

// Groups the `entries` (row, slot) pairs of `indices` by expert.
//
// Writes `perm[i]`, the flat index `row * 8 + slot` of the i-th pair in expert
// order, and `tiles[i] = {expert, first position in perm, pairs, 0}` for each
// run of at most 64 pairs of one expert, ordered by expert, with their number
// in `tile_count[0]`. Pairs with an id outside `[0, experts)` are left out and
// their `n` outputs in `out` are set to NaN. One block of 1024 threads; at
// most 8192 pairs and 1024 experts.
extern "C" __global__ void __launch_bounds__(kGroupThreads)
    expert_group(const int32_t* __restrict__ indices, int entries, int experts, int n,
                 __nv_bfloat16* __restrict__ out, int32_t* __restrict__ perm,
                 int4* __restrict__ tiles, int32_t* __restrict__ tile_count) {
    __shared__ int count[kMaxExperts];
    __shared__ int first[kMaxExperts];
    __shared__ int warp_sums[32];
    __shared__ int invalid_count;
    __shared__ int invalid[kMaxPairs];
    const int tid = threadIdx.x;
    const int lane = tid % 32;

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

    // Count. Each pair's rank among its expert's pairs is the count before
    // its own atomic; the order of ranks varies between runs, but each output
    // depends only on its own pair.
    int rank[kGroupRounds];
#pragma unroll
    for (int r = 0; r < kGroupRounds; ++r) {
        const int i = r * kGroupThreads + tid;
        const bool valid = ids[r] >= 0 && ids[r] < experts;
        rank[r] = valid ? atomicAdd(&count[ids[r]], 1) : 0;
        if (i < entries && !valid) {
            invalid[atomicAdd(&invalid_count, 1)] = i;
        }
    }
    __syncthreads();

    // Exclusive prefix sums of pairs and tiles, one expert per thread.
    int pairs = 0;
    int expert_tiles = 0;
    if (tid < experts) {
        pairs = count[tid];
        expert_tiles = (pairs + kTileT - 1) / kTileT;
    }
    const int pair_end = block_scan(pairs, warp_sums);
    __syncthreads();
    const int tile_end = block_scan(expert_tiles, warp_sums);
    const int pair_start = pair_end - pairs;
    const int tile_start = tile_end - expert_tiles;
    if (tid < experts) {
        first[tid] = pair_start;
        for (int j = 0; j < expert_tiles; ++j) {
            tiles[tile_start + j] =
                make_int4(tid, pair_start + j * kTileT, min(kTileT, pairs - j * kTileT), 0);
        }
    }
    if (tid == kGroupThreads - 1) {
        tile_count[0] = tile_end;
    }
    __syncthreads();

    // Scatter.
#pragma unroll
    for (int r = 0; r < kGroupRounds; ++r) {
        if (ids[r] >= 0 && ids[r] < experts) {
            perm[first[ids[r]] + rank[r]] = r * kGroupThreads + tid;
        }
    }

    // NaN rows for unknown ids, one warp per pair.
    const __nv_bfloat16 nan = __float2bfloat16(__int_as_float(0x7fc00000));
    for (int b = tid / 32; b < invalid_count; b += kGroupThreads / 32) {
        for (int c = lane; c < n; c += 32) {
            out[size_t(invalid[b]) * n + c] = nan;
        }
    }
}

// Arguments, shared by the GEMM kernels:
// - `x`, `weight`, `scales`, `global_scale`, `out`, `inputs_per_row`: as for
//   `expert_matvec_*` in experts.cu;
// - `perm`, `tiles`, `tile_count`: written by `expert_group`.
// Grid: (ceil(n / 128), an upper bound on the tile count); blocks past
// `tile_count[0]` exit. Needs `Stage::kBytes * kStages` bytes of dynamic shared
// memory: 4 * 12,288 for NVFP4 and 3 * 16,384 for MXFP8. n must be a multiple
// of 16 and k of 64.
#define EXPERT_GEMM(name, fp4, stages)                                                            \
    extern "C" __global__ void __launch_bounds__(kThreads, 2)                                     \
        name(const __nv_bfloat16* x, const uint8_t* weight, const uint8_t* scales,                \
             const float* global_scale, __nv_bfloat16* out, const int* perm, const int4* tiles,   \
             const int* tile_count, int n, int k, int inputs_per_row) {                           \
        expert_gemm<fp4, stages>(x, weight, scales, global_scale, out, perm, tiles, tile_count,   \
                                 n, k, inputs_per_row);                                           \
    }

EXPERT_GEMM(expert_gemm_nvfp4, true, 4)
EXPERT_GEMM(expert_gemm_mxfp8, false, 3)
