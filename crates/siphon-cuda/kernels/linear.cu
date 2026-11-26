// Matrix-vector products `out = W @ x` with BF16 weights and F32 accumulation.
//
// One launch runs a batch of up to kMaxSegments products that share k. Each segment has
// its own weight [rows, k] (row-major), input x [k] and output [rows], so the projections
// that read one layer input run as one launch, and the launch streams enough bytes to
// approach DRAM bandwidth.
//
// Each block of kWarps warps owns consecutive rows of one segment. A row is split over
// L lanes (32, or fewer when k is short, so a warp holds 32 / L rows), and over S warps
// when the batch has too few rows to fill the GPU. Each lane first issues V 16-byte
// weight loads, then stages x in shared memory, then accumulates `acc += w * x` over its
// loads in order. A row's lanes reduce with a butterfly shuffle, and its S warps through
// shared memory. The sum order therefore differs from MLX's; tests/linear.rs bounds the
// error.
//
// Programmatic dependent launch (pdl.cuh): a block issues its first batch of weight
// loads, calls pdl_wait, then pdl_trigger, and only then reads x. The launcher picks per
// call whether this kernel itself starts early; without PDL the wait returns at once. The
// trigger lets a next kernel launched with PDL start as soon as the last block of this
// grid has passed its wait, which saved 0.06 ms per decode step. Triggering before the
// wait cost 0.08 ms: the kernels after a PDL-launched GEMV, up to the persistent expert
// grid, then launch and take SMs while it still waits.

#include <cuda_bf16.h>
#include <cstdint>

#include "cluster_sm89.cuh"
#include "pdl.cuh"

namespace {

constexpr int kMaxSegments = 8;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;

struct Segment {
    const __nv_bfloat16* w;
    const void* x;
    void* out;
    int rows;
    // Blocks of this segment and every segment before it.
    int block_end;
};

struct Batch {
    Segment seg[kMaxSegments];
    int count;
    int k;
    // Nonzero when launched with programmatic dependent launch.
    int pdl;
};

// Streams weights past L1: each byte is read once per step.
__device__ __forceinline__ uint4 load_weight(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ float lo(uint32_t pair) { return __uint_as_float(pair << 16); }
__device__ __forceinline__ float hi(uint32_t pair) { return __uint_as_float(pair & 0xffff0000u); }

// acc + dot(8 BF16 weights, 8 inputs at x[8 * i]).
__device__ __forceinline__ float dot8(uint4 w, const __nv_bfloat16* x, int i, float acc) {
    const uint4 v = reinterpret_cast<const uint4*>(x)[i];
    acc = fmaf(lo(w.x), lo(v.x), acc);
    acc = fmaf(hi(w.x), hi(v.x), acc);
    acc = fmaf(lo(w.y), lo(v.y), acc);
    acc = fmaf(hi(w.y), hi(v.y), acc);
    acc = fmaf(lo(w.z), lo(v.z), acc);
    acc = fmaf(hi(w.z), hi(v.z), acc);
    acc = fmaf(lo(w.w), lo(v.w), acc);
    acc = fmaf(hi(w.w), hi(v.w), acc);
    return acc;
}

__device__ __forceinline__ float dot8(uint4 w, const float* x, int i, float acc) {
    const float4 a = reinterpret_cast<const float4*>(x)[2 * i];
    const float4 b = reinterpret_cast<const float4*>(x)[2 * i + 1];
    acc = fmaf(lo(w.x), a.x, acc);
    acc = fmaf(hi(w.x), a.y, acc);
    acc = fmaf(lo(w.y), a.z, acc);
    acc = fmaf(hi(w.y), a.w, acc);
    acc = fmaf(lo(w.z), b.x, acc);
    acc = fmaf(hi(w.z), b.y, acc);
    acc = fmaf(lo(w.w), b.z, acc);
    acc = fmaf(hi(w.w), b.w, acc);
    return acc;
}

__device__ __forceinline__ void store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void store(float* p, float v) { *p = v; }

// Tx and To are the input and output element types; L lanes of S warps share a row and
// each lane keeps V weight loads in flight. `x_smem` holds k elements of Tx.
template <typename Tx, typename To, int L, int S, int V>
__device__ void gemv(const Batch& b, Tx* x_smem) {
    static_assert(kWarps % S == 0, "a block holds whole rows");
    constexpr int kRowsPerWarp = 32 / L;
    constexpr int kRowsPerBlock = kWarps / S * kRowsPerWarp;
    // Consecutive 16-byte vectors of a row go to one warp's L lanes, then the next warp's.
    constexpr int kStride = S * L;

    int s = 0;
    int block_begin = 0;
    while (s + 1 < b.count && int(blockIdx.x) >= b.seg[s].block_end) {
        block_begin = b.seg[s].block_end;
        ++s;
    }
    const Segment& seg = b.seg[s];
    const int k = b.k;
    const int vecs = k / 8;  // 16-byte weight vectors per row
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int sub = (warp % S) * L + lane % L;
    const int row = (int(blockIdx.x) - block_begin) * kRowsPerBlock +
                    warp / S * kRowsPerWarp + lane / L;
    const bool live = row < seg.rows;
    const auto* w_row = reinterpret_cast<const uint4*>(seg.w + size_t(live ? row : 0) * k);

    // Weight loads go out first: no kernel writes them, so they may run ahead of
    // pdl_wait. ptxas moves read-only loads freely across griddepcontrol.wait, and with an
    // unconditional wait it placed the wait first. A branch keeps the wait after the
    // loads. The trigger comes after the wait.
    uint4 wv[V];
#pragma unroll
    for (int v = 0; v < V; ++v) {
        const int i = v * kStride + sub;
        wv[v] = live && i < vecs ? load_weight(w_row + i) : make_uint4(0, 0, 0, 0);
    }
    if (b.pdl) {
        pdl_wait();
    }
    pdl_trigger();

    // k is a multiple of 8, so x is whole 16-byte vectors. Plain loads, not the read-only
    // path: with PDL the kernel that writes x may still have run when this one started.
    const auto* x = static_cast<const uint4*>(seg.x);
    constexpr int kXPerVec = 16 / sizeof(Tx);
    for (int i = threadIdx.x; i < k / kXPerVec; i += kThreads) {
        reinterpret_cast<uint4*>(x_smem)[i] = x[i];
    }
    __syncthreads();

    float acc = 0.f;
#pragma unroll
    for (int v = 0; v < V; ++v) {
        const int i = v * kStride + sub;
        if (i < vecs) {
            acc = dot8(wv[v], x_smem, i, acc);
        }
    }
    // Rows longer than one batch of loads.
    for (int base = V * kStride; base < vecs; base += V * kStride) {
#pragma unroll
        for (int v = 0; v < V; ++v) {
            const int i = base + v * kStride + sub;
            wv[v] = live && i < vecs ? load_weight(w_row + i) : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int v = 0; v < V; ++v) {
            const int i = base + v * kStride + sub;
            if (i < vecs) {
                acc = dot8(wv[v], x_smem, i, acc);
            }
        }
    }

#pragma unroll
    for (int offset = L / 2; offset >= 1; offset /= 2) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if constexpr (S > 1) {
        __shared__ float partial[kWarps][kRowsPerWarp];
        if (lane % L == 0) {
            partial[warp][lane / L] = acc;
        }
        __syncthreads();
        if (warp % S == 0) {
#pragma unroll
            for (int w = 1; w < S; ++w) {
                acc += partial[warp + w][lane / L];
            }
        }
    }
    if (live && sub == 0) {
        store(static_cast<To*>(seg.out) + row, acc);
    }
}

}  // namespace

// Entry points by input and output type, lanes per row L, warps per row S and loads per
// lane V. The launcher picks them from k and the row count. Dynamic shared memory:
// k * sizeof(Tx) bytes.
#define GEMV_ENTRY(name, Tx, To, L, S, V)                                                  \
    extern "C" __global__ void __launch_bounds__(kThreads)                                 \
    name(const __grid_constant__ Batch batch) {                                            \
        extern __shared__ uint4 x_smem[];                                                  \
        gemv<Tx, To, L, S, V>(batch, reinterpret_cast<Tx*>(x_smem));                       \
    }

// k = 2304 and 3072 fill one batch of loads exactly; v8 serves every other k.
GEMV_ENTRY(gemv_bf16_l32_v9, __nv_bfloat16, __nv_bfloat16, 32, 1, 9)
GEMV_ENTRY(gemv_bf16_l32_v12, __nv_bfloat16, __nv_bfloat16, 32, 1, 12)
GEMV_ENTRY(gemv_bf16_l32_v8, __nv_bfloat16, __nv_bfloat16, 32, 1, 8)
// Few rows: 8 warps per row, 4 loads per lane.
GEMV_ENTRY(gemv_bf16_l32_s8_v4, __nv_bfloat16, __nv_bfloat16, 32, 8, 4)
// k <= 64: 8 lanes per row, 4 rows per warp.
GEMV_ENTRY(gemv_bf16_l8_v1, __nv_bfloat16, __nv_bfloat16, 8, 1, 1)
// F32 input and output with BF16 weights: the message gates' forget and gain projections.
GEMV_ENTRY(gemv_f32_l8_v1, float, float, 8, 1, 1)
GEMV_ENTRY(gemv_f32_l32_v8, float, float, 32, 1, 8)
GEMV_ENTRY(gemv_f32_l32_s8_v4, float, float, 32, 8, 4)

constexpr int kMaxRanges = 32;

// Up to kMaxRanges weights for prefetch_l2, each as 16-byte vectors. The last `rings`
// ranges are the key and value slots of sliding KV caches, ring_slot_vecs vectors per
// slot, which fill one slot per position: they are prefetched only once *position fills
// every slot.
struct Ranges {
    const uint4* p[kMaxRanges];
    long long vecs[kMaxRanges];
    int count;
    int rings;
    long long ring_slot_vecs;
    const unsigned* position;
};

// The vectors of range s to prefetch.
__device__ __forceinline__ long long range_vecs(const Ranges& r, int s) {
    if (s < r.count - r.rings) {
        return r.vecs[s];
    }
    return (long long)(*r.position) * r.ring_slot_vecs < r.vecs[s] ? 0 : r.vecs[s];
}

// Pulls the ranges into L2, in order, for later kernels. Runs on a side stream beside
// latency-bound kernels. Prefetch instructions are hints the memory system drops under
// load, so this issues real 16-byte loads, 8 per thread at a time. The trap that uses
// their values never fires, since count is never negative; it keeps the loads alive.
extern "C" __global__ void __launch_bounds__(kThreads)
prefetch_l2(const __grid_constant__ Ranges r) {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    const long long stride = (long long)gridDim.x * kThreads;
    uint32_t seen = 0;
    for (int s = 0; s < r.count; ++s) {
        const uint4* p = r.p[s];
        const long long vecs = range_vecs(r, s);
        for (long long base = blockIdx.x * kThreads + threadIdx.x; base < vecs;
             base += 8 * stride) {
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const long long i = base + u * stride;
                if (i < vecs) {
                    uint32_t a, b, c, d;
                    asm volatile(
                        "ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, "
                        "[%4], %5;"
                        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
                        : "l"(p + i), "l"(policy));
                    seen ^= a ^ b ^ c ^ d;
                }
            }
        }
    }
    if (r.count < 0 && seen == 0) {
        __trap();
    }
}

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// The ranges as TMA bulk prefetches into L2 of up to kChunk bytes, in order, chunk i due
// i * kChunk / bytes_per_ns after the block starts, spread round robin over the lanes of
// the grid. Bulk prefetches return nothing to the SM, so the main stream keeps the L2
// bandwidth to the SMs, but they do not slow down when other kernels read DRAM:
// unpaced, they flooded the memory system and took a BF16 step from 3.83 to 4.21 ms.
extern "C" __global__ void __launch_bounds__(32)
prefetch_l2_paced(const __grid_constant__ Ranges r, float bytes_per_ns) {
    constexpr long long kChunk = 4096;
#if !SIPHON_SM89
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
#endif
    const unsigned long long t0 = global_ns();
    const long long lanes = (long long)gridDim.x * 32;
    const long long lane = blockIdx.x * 32ll + threadIdx.x;
    long long chunk_base = 0;
    for (int s = 0; s < r.count; ++s) {
        const char* p = reinterpret_cast<const char*>(r.p[s]);
        const long long bytes = range_vecs(r, s) * 16;
        const long long chunks = (bytes + kChunk - 1) / kChunk;
        for (long long c = (lane - chunk_base % lanes + lanes) % lanes; c < chunks; c += lanes) {
            const unsigned long long due = t0 + (unsigned long long)(float(chunk_base + c) * kChunk / bytes_per_ns);
            while (global_ns() < due) {
                __nanosleep(100);
            }
            const unsigned n = unsigned(min(kChunk, bytes - c * kChunk));
#if SIPHON_SM89
            // No bulk prefetch: one L2 prefetch per 128-byte line.
            for (unsigned line = 0; line < n; line += 128) {
                asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p + c * kChunk + line) : "memory");
            }
#else
            asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
                         ::"l"(p + c * kChunk), "r"(n), "l"(policy) : "memory");
#endif
        }
        chunk_base += chunks;
    }
}

// Matrix products `out = x @ W^T` on tensor cores, for prefill: x is [m, k], W is [n, k]
// as stored in the checkpoint, out is [m, n], all row-major. BF16 weights, F32
// accumulation.
//
// One launch runs up to kMaxProducts products that share m and k, each with its own
// weight, input and output. A block computes one BM x BN tile of one product. Blocks walk
// a product's tiles with m fastest, so the blocks that share a weight tile run together
// and the weight streams from DRAM about once.
//
// A block stages BM x BK of x and BN x BK of W per k step in shared memory with cp.async,
// STAGES steps deep, and its warps multiply them with mma.sync m16n8k16 fed by ldmatrix.
// The warps form a WM x WN x WK grid: WM x WN split the tile, and WK > 1 splits each k
// step's 16-wide slices over warps, whose partial sums meet in shared memory at the end.
// BF16 rows of shared memory are XOR-swizzled in 16-byte chunks, so ldmatrix reads are
// free of bank conflicts. Rows and columns past m, n and k load as zeros through
// cp.async's source size, and stores past m and n are skipped.
//
// When the tiles cannot fill the GPU, as for small m or narrow outputs, the launcher runs
// each tile as a cluster of `splits` blocks. Each block sums a contiguous share of the k
// steps, and the blocks add their partial sums through distributed shared memory, so the
// split needs no scratch buffer and no second launch.
//
// With F32 x, each value splits into three BF16 parts, hi + mid + lo, which carry 24 of
// its bits; each part runs its own mma. BF16 x BF16 products are exact in F32, so the
// result matches an F32 product with the widened weights up to sum order and about 2^-25
// of each |w x|.

namespace gemm {

constexpr int kMaxProducts = 8;

struct Product {
    const __nv_bfloat16* w;
    const void* x;
    void* out;
    int n;
    // Tiles of this product and every product before it.
    int tile_end;
};

struct Batch {
    Product p[kMaxProducts];
    int count;
    int m;
    int k;
    // Blocks per tile, one cluster: each sums a share of the k steps.
    int splits;
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16 bytes from global to shared memory, or 16 zero bytes when `valid` is false.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src),
                 "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// d += a @ b for one 16 x 8 x 16 slice.
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#if SIPHON_SM89
// Each block's parked sums for the split-k reduction, BM x BN floats at block x times
// BM x BN, and a barrier per cluster. A split launch holds at most a GPU's worth of
// blocks, 256 on the 4090, so 2^21 floats cover tiles up to 64 x 128.
constexpr int kRedFloats = 1 << 21;
constexpr int kMaxClusters = 1024;
__device__ float red_sm89[kRedFloats];
__device__ sm89::BarrierState barriers_sm89[kMaxClusters];
#else
// Float at shared address `addr` of block `rank` of this cluster.
__device__ __forceinline__ float ld_cluster(uint32_t addr, int rank) {
    uint32_t remote;
    float v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(addr), "r"(rank));
    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
    return v;
}

__device__ __forceinline__ void cluster_sync() {
    asm volatile(
        "barrier.cluster.arrive.release.aligned;\n"
        "barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
#endif

__device__ __forceinline__ uint32_t pack(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return uint32_t(__bfloat16_as_ushort(lo)) | (uint32_t(__bfloat16_as_ushort(hi)) << 16);
}

// Splits two F32 values into three BF16 pairs whose sums give back about 24 bits of each.
// Each difference is exact in F32.
__device__ __forceinline__ void split3(float2 v, uint32_t& hi, uint32_t& mid, uint32_t& lo) {
    const __nv_bfloat16 hx = __float2bfloat16_rn(v.x), hy = __float2bfloat16_rn(v.y);
    const float rx = v.x - __bfloat162float(hx), ry = v.y - __bfloat162float(hy);
    const __nv_bfloat16 mx = __float2bfloat16_rn(rx), my = __float2bfloat16_rn(ry);
    const __nv_bfloat16 lx = __float2bfloat16_rn(rx - __bfloat162float(mx));
    const __nv_bfloat16 ly = __float2bfloat16_rn(ry - __bfloat162float(my));
    hi = pack(hx, hy);
    mid = pack(mx, my);
    lo = pack(lx, ly);
}

__device__ __forceinline__ void store2(__nv_bfloat16* out, size_t i, bool pair, bool second,
                                       float a, float b) {
    if (pair) {
        *reinterpret_cast<__nv_bfloat162*>(out + i) = __floats2bfloat162_rn(a, b);
    } else {
        out[i] = __float2bfloat16_rn(a);
        if (second) {
            out[i + 1] = __float2bfloat16_rn(b);
        }
    }
}

__device__ __forceinline__ void store2(float* out, size_t i, bool pair, bool second, float a,
                                       float b) {
    if (pair) {
        *reinterpret_cast<float2*>(out + i) = make_float2(a, b);
    } else {
        out[i] = a;
        if (second) {
            out[i + 1] = b;
        }
    }
}

// Shared-memory layout of one stage of a [rows, BK] operand tile of element type T, in
// 16-byte chunks. BF16 rows of 4 or more chunks swap chunks by an XOR of the row, so the
// 8 rows one ldmatrix reads fall in distinct banks. F32 rows (x of gemm_f32, read as
// float2 pairs) carry 8 elements of padding instead.
template <typename T, int BK>
struct Layout {
    static constexpr int kChunks = BK * int(sizeof(T)) / 16;  // per row
    static constexpr bool kSwizzle = sizeof(T) == 2;
    static constexpr int kRowBytes = kSwizzle ? BK * 2 : (BK + 8) * 4;
    static_assert(!kSwizzle || kChunks == 4 || kChunks % 8 == 0, "swizzle of 4 or 8k chunks");

    // XOR applied to the chunk index of row `row`.
    __device__ static constexpr int mask(int row) {
        if constexpr (!kSwizzle) {
            return 0;
        } else if constexpr (kChunks == 4) {
            return row / 2 % 4;
        } else {
            return row % 8;
        }
    }

    __device__ static constexpr int offset(int row, int chunk) {
        return row * kRowBytes + (chunk ^ mask(row)) * 16;
    }
};

template <typename Tx, int BM, int BN, int BK, int WM, int WN, int WK, int STAGES>
struct Tile {
    using A = Layout<Tx, BK>;
    using B = Layout<__nv_bfloat16, BK>;
    static constexpr int kWarps = WM * WN * WK;
    static constexpr int kThreads = 32 * kWarps;
    static constexpr int kWarpM = BM / WM;
    static constexpr int kWarpN = BN / WN;
    static constexpr int kMi = kWarpM / 16;
    static constexpr int kNi = kWarpN / 8;
    static constexpr int kStageA = BM * A::kRowBytes;
    static constexpr int kStageB = BN * B::kRowBytes;
    static constexpr int kStage = kStageA + kStageB;
    static constexpr int kLoadsA = BM * A::kChunks / kThreads;  // chunks per thread
    static constexpr int kLoadsB = BN * B::kChunks / kThreads;
    // Partial sums of WK - 1 warps per slice, or one tile for the cluster reduction.
    static constexpr int kReduceBytes = (WK > 2 ? WK - 1 : 1) * BM * BN * 4;
    static constexpr int kSmemBytes =
        STAGES * kStage > kReduceBytes ? STAGES * kStage : kReduceBytes;
    static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0, "whole 16 x 16 warp slices");
    static_assert(BK % (16 * WK) == 0, "whole k slices per warp");
    static_assert(kLoadsA * kThreads == BM * A::kChunks, "threads share x chunks evenly");
    static_assert(kLoadsB * kThreads == BN * B::kChunks, "threads share W chunks evenly");
    static_assert(STAGES >= 2, "a pipeline needs two stages");
};

// One thread's share of the copies of an operand tile: a source row and chunk per copy,
// fixed for the whole k loop.
template <typename T, int BK, int LOADS, int THREADS>
struct Loader {
    using L = Layout<T, BK>;
    static constexpr int kPerChunk = 16 / int(sizeof(T));
    const T* src[LOADS];
    int col[LOADS];      // first element of the chunk within a k step
    uint32_t dst[LOADS];  // byte offset within a stage
    uint32_t live;        // bit i: copy i reads a row below the row count

    __device__ Loader(const T* base, int row0, int rows, int k, int tid) : live(0) {
#pragma unroll
        for (int i = 0; i < LOADS; ++i) {
            const int c = tid + i * THREADS;
            const int r = c / L::kChunks;
            const int ch = c % L::kChunks;
            col[i] = ch * kPerChunk;
            dst[i] = L::offset(r, ch);
            const bool ok = row0 + r < rows;
            live |= uint32_t(ok) << i;
            src[i] = base + size_t(ok ? row0 + r : 0) * k + col[i];
        }
    }

    __device__ void load(uint32_t stage, int k0, int k) const {
#pragma unroll
        for (int i = 0; i < LOADS; ++i) {
            const bool ok = (live >> i & 1) && col[i] < k - k0;
            cp_async16(stage + dst[i], ok ? src[i] + k0 : src[i], ok);
        }
    }
};

template <typename Tx, int BM, int BN, int BK, int WM, int WN, int WK, int STAGES>
__device__ void gemm(const Batch& b, char* smem) {
    using T = Tile<Tx, BM, BN, BK, WM, WN, WK, STAGES>;
    using LA = typename T::A;
    using LB = typename T::B;

    // Clusters run along x, so a cluster is `splits` consecutive blocks.
    const int splits = b.splits;
    const int rank = int(blockIdx.x) % splits;
    const int block = int(blockIdx.x) / splits;
    int s = 0;
    int tile_begin = 0;
    while (s + 1 < b.count && block >= b.p[s].tile_end) {
        tile_begin = b.p[s].tile_end;
        ++s;
    }
    const Product& p = b.p[s];
    const int m = b.m, n = p.n, k = b.k;
    const int tiles_m = (m + BM - 1) / BM;
    const int tile = block - tile_begin;
    const int m0 = (tile % tiles_m) * BM;
    const int n0 = (tile / tiles_m) * BN;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int wk = warp % WK;
    const int wn = warp / WK % WN;
    const int wm = warp / (WK * WN);

    const uint32_t smem0 = smem_addr(smem);
    const Loader<Tx, BK, T::kLoadsA, T::kThreads> load_a(static_cast<const Tx*>(p.x), m0, m, k,
                                                          tid);
    const Loader<__nv_bfloat16, BK, T::kLoadsB, T::kThreads> load_b(p.w, n0, n, k, tid);
    auto load = [&](int stage, int step) {
        const uint32_t base = smem0 + stage * T::kStage;
        load_a.load(base, step * BK, k);
        load_b.load(base + T::kStageA, step * BK, k);
    };

    // ldmatrix rows of this lane: A rows lane % 16 and chunk lane / 16 of each 16 x 16
    // slice; B rows lane % 8 + 8 (lane / 16) and chunk lane / 8 % 2. Slice origins are
    // multiples of 8 rows and 2 chunks, so the swizzle depends only on the lane.
    const int a_row = wm * T::kWarpM + lane % 16;
    const int b_row = wn * T::kWarpN + lane % 8 + lane / 16 * 8;
    const uint32_t a_base = a_row * LA::kRowBytes;
    const uint32_t b_base = T::kStageA + b_row * LB::kRowBytes;
    const int a_xor = (lane / 16) ^ LA::mask(a_row);
    const int b_xor = (lane / 8 % 2) ^ LB::mask(b_row);

    float acc[T::kMi][T::kNi][4] = {};

    // This block's share of the k steps.
    const int all_steps = (k + BK - 1) / BK;
    const int share = (all_steps + splits - 1) / splits;
    const int first_step = min(all_steps, rank * share);
    const int k_steps = min(all_steps, first_step + share);
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (first_step + st < k_steps) {
            load(st, first_step + st);
        }
        cp_async_commit();
    }

    int stage = 0;
    for (int step = first_step; step < k_steps; ++step) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        const int next = step + STAGES - 1;
        if (next < k_steps) {
            load((stage + STAGES - 1) % STAGES, next);
        }
        cp_async_commit();

        const uint32_t st_addr = smem0 + stage * T::kStage;
        const char* st_ptr = smem + stage * T::kStage;
#pragma unroll
        for (int kk = wk * 16; kk < BK; kk += 16 * WK) {
            uint32_t bf[T::kNi][2];
#pragma unroll
            for (int nj = 0; nj < T::kNi; nj += 2) {
                uint32_t r[4];
                ldmatrix_x4(r, st_addr + b_base + nj * 8 * LB::kRowBytes +
                                   ((kk / 8) ^ b_xor) * 16);
                bf[nj][0] = r[0];
                bf[nj][1] = r[1];
                bf[nj + 1][0] = r[2];
                bf[nj + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < T::kMi; ++mi) {
                if constexpr (sizeof(Tx) == 2) {
                    uint32_t af[4];
                    ldmatrix_x4(af, st_addr + a_base + mi * 16 * LA::kRowBytes +
                                        ((kk / 8) ^ a_xor) * 16);
#pragma unroll
                    for (int ni = 0; ni < T::kNi; ++ni) {
                        mma(acc[mi][ni], af, bf[ni][0], bf[ni][1]);
                    }
                } else {
                    // Fragment layout of mma's A operand: rows g and g + 8, columns 2t,
                    // 2t + 1 and the same plus 8.
                    const int g = lane / 4, t = lane % 4;
                    const float* base = reinterpret_cast<const float*>(
                        st_ptr + (wm * T::kWarpM + mi * 16 + g) * LA::kRowBytes) + kk + 2 * t;
                    constexpr int kRow = LA::kRowBytes / 4;
                    uint32_t hi[4], mid[4], lo[4];
                    const float2 v[4] = {
                        *reinterpret_cast<const float2*>(base),
                        *reinterpret_cast<const float2*>(base + 8 * kRow),
                        *reinterpret_cast<const float2*>(base + 8),
                        *reinterpret_cast<const float2*>(base + 8 * kRow + 8),
                    };
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        split3(v[i], hi[i], mid[i], lo[i]);
                    }
#pragma unroll
                    for (int ni = 0; ni < T::kNi; ++ni) {
                        mma(acc[mi][ni], lo, bf[ni][0], bf[ni][1]);
                        mma(acc[mi][ni], mid, bf[ni][0], bf[ni][1]);
                        mma(acc[mi][ni], hi, bf[ni][0], bf[ni][1]);
                    }
                }
            }
        }
        stage = stage + 1 == STAGES ? 0 : stage + 1;
    }

    constexpr int kValues = T::kMi * T::kNi * 4;  // accumulators per thread
    float* const red = reinterpret_cast<float*>(smem);
    const int slice = wm * WN + wn;
    if constexpr (WK > 1) {
        // Warps with wk > 0 hand their partial sums to the wk = 0 warp of their slice.
        cp_async_wait<0>();
        __syncthreads();
        if (wk > 0) {
            float* dst = red + ((wk - 1) * WM * WN + slice) * kValues * 32 + lane;
#pragma unroll
            for (int v = 0; v < kValues; ++v) {
                dst[v * 32] = (&acc[0][0][0])[v];
            }
        }
        __syncthreads();
        if (wk == 0) {
#pragma unroll
            for (int j = 1; j < WK; ++j) {
                const float* src = red + ((j - 1) * WM * WN + slice) * kValues * 32 + lane;
#pragma unroll
                for (int v = 0; v < kValues; ++v) {
                    (&acc[0][0][0])[v] += src[v * 32];
                }
            }
        }
    }

    Tx* const out = static_cast<Tx*>(p.out);
    const int g = lane / 4, t = lane % 4;
    auto store = [&](int mi, int ni, const float (&v)[4]) {
        const int col = n0 + wn * T::kWarpN + ni * 8 + 2 * t;
        if (col >= n) {
            return;
        }
        const bool second = col + 1 < n;
        const bool pair = second && n % 2 == 0;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = m0 + wm * T::kWarpM + mi * 16 + g + 8 * h;
            if (row < m) {
                store2(out, size_t(row) * n + col, pair, second, v[2 * h], v[2 * h + 1]);
            }
        }
    };

    if (splits == 1) {
        if (wk == 0) {
#pragma unroll
            for (int mi = 0; mi < T::kMi; ++mi)
#pragma unroll
                for (int ni = 0; ni < T::kNi; ++ni) {
                    store(mi, ni, acc[mi][ni]);
                }
        }
        return;
    }

#if SIPHON_SM89
    // The same reduction with the parked sums in global memory.
    cp_async_wait<0>();
    if (size_t(blockIdx.x + 1) * BM * BN > size_t(kRedFloats)) {
        __trap();
    }
    if (wk == 0) {
        float* dst = red_sm89 + size_t(blockIdx.x) * BM * BN + slice * kValues * 32 + lane;
#pragma unroll
        for (int v = 0; v < kValues; ++v) {
            dst[v * 32] = (&acc[0][0][0])[v];
        }
    }
    sm89::barrier(barriers_sm89, kMaxClusters, splits);
    if (wk == 0) {
        const int cluster0 = int(blockIdx.x) - rank;
#pragma unroll
        for (int mi = 0; mi < T::kMi; ++mi)
#pragma unroll
            for (int ni = 0; ni < T::kNi; ++ni) {
                if ((mi * T::kNi + ni) % splits != rank) {
                    continue;
                }
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    v[e] = acc[mi][ni][e];
                }
                for (int j = 0; j < splits; ++j) {
                    if (j == rank) {
                        continue;
                    }
                    const float* src = red_sm89 + size_t(cluster0 + j) * BM * BN + slice * kValues * 32 + lane;
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        v[e] += __ldcg(src + ((mi * T::kNi + ni) * 4 + e) * 32);
                    }
                }
                store(mi, ni, v);
            }
    }
#else
    // Split k: every block of the cluster parks its sums in shared memory, then block r
    // adds up and stores the 16 x 8 slices (mi, ni) with (mi kNi + ni) % splits == r.
    // The second barrier keeps each block's shared memory alive until the reads finish.
    cp_async_wait<0>();
    __syncthreads();
    if (wk == 0) {
        float* dst = red + slice * kValues * 32 + lane;
#pragma unroll
        for (int v = 0; v < kValues; ++v) {
            dst[v * 32] = (&acc[0][0][0])[v];
        }
    }
    cluster_sync();
    if (wk == 0) {
        const uint32_t base = smem_addr(red + slice * kValues * 32 + lane);
#pragma unroll
        for (int mi = 0; mi < T::kMi; ++mi)
#pragma unroll
            for (int ni = 0; ni < T::kNi; ++ni) {
                if ((mi * T::kNi + ni) % splits != rank) {
                    continue;
                }
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    v[e] = acc[mi][ni][e];
                }
                for (int j = 0; j < splits; ++j) {
                    if (j == rank) {
                        continue;
                    }
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        v[e] += ld_cluster(base + ((mi * T::kNi + ni) * 4 + e) * 32 * 4, j);
                    }
                }
                store(mi, ni, v);
            }
    }
    cluster_sync();
#endif
}

}  // namespace gemm

// Entry points by input type and tile; the name gives BM x BN. The launcher picks one
// from m and sets the dynamic shared memory to Tile::kSmemBytes.
#define GEMM_ENTRY(name, Tx, BM, BN, BK, WM, WN, WK, STAGES)                               \
    extern "C" __global__ void __launch_bounds__(                                          \
        gemm::Tile<Tx, BM, BN, BK, WM, WN, WK, STAGES>::kThreads)                          \
        name(const __grid_constant__ gemm::Batch batch) {                                  \
        extern __shared__ __align__(16) char gemm_smem[];                                  \
        gemm::gemm<Tx, BM, BN, BK, WM, WN, WK, STAGES>(batch, gemm_smem);                  \
    }

// Small m: 16 or 32 rows, BK = 64 for long contiguous weight reads, two blocks per SM.
GEMM_ENTRY(gemm_bf16_16x64, __nv_bfloat16, 16, 64, 64, 1, 2, 2, 4)
GEMM_ENTRY(gemm_bf16_32x64, __nv_bfloat16, 32, 64, 64, 1, 2, 2, 4)
// Large m: 32 x 32 warp tiles, two blocks per SM; 64 x 128 when its tiles fill the GPU.
GEMM_ENTRY(gemm_bf16_64x64, __nv_bfloat16, 64, 64, 64, 2, 2, 1, 3)
GEMM_ENTRY(gemm_bf16_64x128, __nv_bfloat16, 64, 128, 64, 2, 2, 1, 2)
// F32 inputs: shared rows hold F32 x, so BK = 32 keeps the large tile at two blocks per SM.
GEMM_ENTRY(gemm_f32_16x64, float, 16, 64, 64, 1, 2, 2, 3)
GEMM_ENTRY(gemm_f32_64x64, float, 64, 64, 32, 2, 2, 1, 3)
