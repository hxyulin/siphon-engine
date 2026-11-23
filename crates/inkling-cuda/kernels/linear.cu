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

#include <cuda_bf16.h>
#include <cstdint>

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

    // Weight loads go out first, so x staging hides under their latency.
    uint4 wv[V];
#pragma unroll
    for (int v = 0; v < V; ++v) {
        const int i = v * kStride + sub;
        wv[v] = live && i < vecs ? load_weight(w_row + i) : make_uint4(0, 0, 0, 0);
    }

    // k is a multiple of 8, so x is whole 16-byte vectors.
    const auto* x = static_cast<const uint4*>(seg.x);
    constexpr int kXPerVec = 16 / sizeof(Tx);
    for (int i = threadIdx.x; i < k / kXPerVec; i += kThreads) {
        reinterpret_cast<uint4*>(x_smem)[i] = __ldg(x + i);
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

constexpr int kMaxRanges = 24;

// Up to kMaxRanges weights for prefetch_l2, each as 16-byte vectors.
struct Ranges {
    const uint4* p[kMaxRanges];
    long long vecs[kMaxRanges];
    int count;
};

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
        const long long vecs = r.vecs[s];
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
