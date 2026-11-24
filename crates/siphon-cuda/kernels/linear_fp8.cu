// Matrix-vector products `out = W @ x` with FP8 E4M3 weights, BF16 inputs and outputs,
// and F32 accumulation, for decode. Also the kernel that quantizes BF16 weights to that
// format.
//
// A weight matrix [rows, k] is stored as E4M3 bytes [padded_rows, k], row-major, followed
// by one F32 scale per padded row, padded_rows being rows rounded up to a multiple of 32.
// Padding rows are zero. Row r holds round_e4m3(w / scale[r]) with scale[r] =
// amax(|w[r]|) / 448, so the largest weight of each row maps to E4M3's largest value.
//
// One launch runs a batch of up to kMaxSegments products that share k, as the BF16 GEMV in
// linear.cu does. A block of kWarps warps owns 2 kWarps consecutive rows, and each warp two
// rows, which are contiguous in memory: its 32 lanes read them as k / 8 sixteen-byte
// vectors, V = k / 256 per lane, so every load instruction covers 512 contiguous bytes. The
// padding keeps every load in bounds without a check.
//
// Each lane widens its weights exactly (E4M3 to F16 to F32), multiplies them with x and
// sums in F32 in order; a row's lanes then reduce with a butterfly shuffle, and the row's
// scale multiplies the F32 sum once before it rounds to BF16.
//
// With programmatic dependent launch (the launcher picks it per call), the kernel issues
// its weight and scale loads, which no kernel writes, before pdl_wait, and reads x only
// after it. It triggers after the wait, as linear.cu's GEMV does.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#include "pdl.cuh"

namespace {

constexpr int kMaxSegments = 8;
// Four warps per block beat eight and two in the decode step: 3.05 ms against 3.10 ms.
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kRowsPerWarp = 2;
constexpr float kE4m3Max = 448.f;

struct Segment {
    const uint8_t* w;
    const float* scale;
    const __nv_bfloat16* x;
    __nv_bfloat16* out;
    int rows;
    // Blocks of this segment and every segment before it.
    int block_end;
};

struct Batch {
    Segment seg[kMaxSegments];
    int count;
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

__device__ __forceinline__ float load_scale(const float* p) {
    float v;
    asm volatile("ld.global.nc.L1::no_allocate.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float lo(uint32_t pair) { return __uint_as_float(pair << 16); }
__device__ __forceinline__ float hi(uint32_t pair) { return __uint_as_float(pair & 0xffff0000u); }

// Two E4M3 values, widened exactly through F16.
__device__ __forceinline__ float2 e4m3x2_to_float2(uint32_t pair) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3);
    return __half22float2(__half2(h));
}

// acc + dot(4 E4M3 weights in `w`, 4 BF16 inputs in `x01`, `x23`).
__device__ __forceinline__ float dot4(uint32_t w, uint32_t x01, uint32_t x23, float acc) {
    const float2 a = e4m3x2_to_float2(w & 0xffffu);
    const float2 b = e4m3x2_to_float2(w >> 16);
    acc = fmaf(a.x, lo(x01), acc);
    acc = fmaf(a.y, hi(x01), acc);
    acc = fmaf(b.x, lo(x23), acc);
    acc = fmaf(b.y, hi(x23), acc);
    return acc;
}

// Shared-memory slot of the input's 16-byte unit u. A lane reads the two units of its
// segment, 32 bytes past the previous lane's, so plain slots put two lanes of each
// quarter warp in one bank; swapping the units of every other group of 8 spreads them.
// With the input's loads issued together (see gemv_fp8), a step took 2.737 instead of
// 2.756 ms at context 0 and 3.261 instead of 3.286 ms at 32K.
__device__ __forceinline__ int x_slot(int u) { return u ^ ((u >> 3) & 1); }

// acc + dot(16 E4M3 weights, the 16 BF16 inputs of segment `seg` of `xs`, units 2 seg and
// 2 seg + 1).
__device__ __forceinline__ float dot16(uint4 w, const uint4* xs, int seg, float acc) {
    const uint4 x0 = xs[x_slot(2 * seg)];
    const uint4 x1 = xs[x_slot(2 * seg + 1)];
    acc = dot4(w.x, x0.x, x0.y, acc);
    acc = dot4(w.y, x0.z, x0.w, acc);
    acc = dot4(w.z, x1.x, x1.y, acc);
    acc = dot4(w.w, x1.z, x1.w, acc);
    return acc;
}

// K inputs per row, V 16-byte loads per lane.
template <int K, int V>
__device__ void gemv_fp8(const Batch& b, __nv_bfloat16* x_smem) {
    constexpr int R = kRowsPerWarp;
    static_assert(R * K == 512 * V, "a warp's rows fill its loads exactly");
    constexpr int kRowsPerBlock = kWarps * R;
    constexpr int kRowVecs = K / 16;

    int s = 0;
    int block_begin = 0;
    while (s + 1 < b.count && int(blockIdx.x) >= b.seg[s].block_end) {
        block_begin = b.seg[s].block_end;
        ++s;
    }
    const Segment& seg = b.seg[s];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int row0 = (int(blockIdx.x) - block_begin) * kRowsPerBlock + warp * R;
    const auto* w = reinterpret_cast<const uint4*>(seg.w + size_t(row0) * K);

    uint4 wv[V];
#pragma unroll
    for (int v = 0; v < V; ++v) {
        wv[v] = load_weight(w + v * 32 + lane);
    }
    // Lane r < R loads the scale of row r; padding rows have scales too.
    const float scale = lane < R ? load_scale(seg.scale + row0 + lane) : 0.f;

    // Triggering before the wait cost 50 us per step: the next kernel's blocks then take
    // SM slots while this one still needs them. ptxas moves read-only loads freely across
    // griddepcontrol.wait and placed an unconditional wait before them; a branch keeps it
    // after the loads.
    if (b.pdl) {
        pdl_wait();
    }
    pdl_trigger();

    // Every load of x before any store, so the loads overlap.
    const auto* x = reinterpret_cast<const uint4*>(seg.x);
    auto* xs = reinterpret_cast<uint4*>(x_smem);
    constexpr int kXUnits = K / 8;
    constexpr int kXPerThread = (kXUnits + kThreads - 1) / kThreads;
    uint4 xv[kXPerThread];
#pragma unroll
    for (int i = 0; i < kXPerThread; ++i) {
        const int u = threadIdx.x + i * kThreads;
        if (u < kXUnits) {
            xv[i] = x[u];
        }
    }
#pragma unroll
    for (int i = 0; i < kXPerThread; ++i) {
        const int u = threadIdx.x + i * kThreads;
        if (u < kXUnits) {
            xs[x_slot(u)] = xv[i];
        }
    }
    __syncthreads();

    float acc[R] = {};
#pragma unroll
    for (int v = 0; v < V; ++v) {
        const int j = v * 32 + lane;
        const int r = j / kRowVecs;
        const float d = dot16(wv[v], xs, j - r * kRowVecs, 0.f);
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            // Rows this load touches: the select folds away where r is known.
            if (rr * kRowVecs <= v * 32 + 31 && (rr + 1) * kRowVecs > v * 32) {
                acc[rr] += r == rr ? d : 0.f;
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            acc[rr] += __shfl_xor_sync(0xffffffffu, acc[rr], offset);
        }
    }
    // Lane r stores row r.
    float mine = acc[0];
#pragma unroll
    for (int rr = 1; rr < R; ++rr) {
        mine = lane == rr ? acc[rr] : mine;
    }
    const int row = row0 + lane;
    if (lane < R && row < seg.rows) {
        seg.out[row] = __float2bfloat16_rn(scale * mine);
    }
}

}  // namespace

// Entry points by k: the launcher picks one. Dynamic shared memory: k * 2 bytes.
#define GEMV_FP8_ENTRY(name, K, V)                                                         \
    extern "C" __global__ void __launch_bounds__(kThreads)                                 \
    name(const __grid_constant__ Batch batch) {                                            \
        extern __shared__ uint4 x_smem[];                                                  \
        gemv_fp8<K, V>(batch, reinterpret_cast<__nv_bfloat16*>(x_smem));                  \
    }

// The attention input projections and the LM head.
GEMV_FP8_ENTRY(gemv_fp8_k2304, 2304, 9)
// The output projection.
GEMV_FP8_ENTRY(gemv_fp8_k3072, 3072, 12)

// Quantizes a BF16 matrix [rows, k] to the layout above: one block per row writes the
// row's E4M3 bytes to q [rows, k] and its scale to scale[row]. The caller zeroes the
// padding rows. An all-zero row gets scale 0 and zero bytes. Each byte is w / scale
// rounded to the nearest E4M3 value, ties to even, saturated to +-448.
extern "C" __global__ void __launch_bounds__(kThreads)
quantize_e4m3_rows(const __nv_bfloat16* __restrict__ w, int k, uint8_t* __restrict__ q,
                   float* __restrict__ scale) {
    __shared__ float warp_max[kWarps];
    const int row = blockIdx.x;
    const __nv_bfloat16* src = w + size_t(row) * k;
    float amax = 0.f;
    for (int i = threadIdx.x; i < k; i += kThreads) {
        amax = fmaxf(amax, fabsf(__bfloat162float(src[i])));
    }
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    if (threadIdx.x % 32 == 0) {
        warp_max[threadIdx.x / 32] = amax;
    }
    __syncthreads();
    amax = warp_max[0];
#pragma unroll
    for (int i = 1; i < kWarps; ++i) {
        amax = fmaxf(amax, warp_max[i]);
    }
    const float s = amax / kE4m3Max;
    if (threadIdx.x == 0) {
        scale[row] = s;
    }
    uint8_t* dst = q + size_t(row) * k;
    for (int i = threadIdx.x; i < k; i += kThreads) {
        const float v = s > 0.f ? __bfloat162float(src[i]) / s : 0.f;
        dst[i] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}
