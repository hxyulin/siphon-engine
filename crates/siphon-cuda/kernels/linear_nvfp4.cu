// Matrix-vector products `out = W @ x` with NVFP4 (E2M1) weights, BF16 inputs and
// outputs, and F32 accumulation, for decode, plus a fused greedy LM-head argmax variant
// (`gemv_nvfp4_k2304_argmax`) and BF16 -> NVFP4 quantization kernels.
//
// A weight matrix [rows, k] is stored in one buffer, with `padded_rows` being `rows`
// rounded up to a multiple of 32 (padding rows are zero):
// - E2M1 pairs [padded_rows, k / 2], row-major, low nibble even column;
// - E4M3 block scales [padded_rows, k / 16], one per 16 values;
// - F32 outer scales [padded_rows], one per row, `amax(|w[r]|) / 1536`.
//
// Each 16-value block picks between the E4M3 scale that maps its maximum to 6 and the one
// that maps it to 4, keeping the one with the smaller sum of squared errors (Four Over
// Six, matching `quantize_row_nvfp4` in expert_matmul.cu).
//
// Shared memory stores `x` transposed across the 4 `uint4` halves of each 32-element chunk
// (`xs[h * (K / 32) + p]`), so consecutive lanes accessing consecutive chunks `p` hit
// distinct 4-bank groups with zero bank conflicts and zero extra shared memory.
//
// With programmatic dependent launch, the kernel issues its read-only weight, block-scale
// and outer-scale loads before pdl_wait, and reads x only after it.

#include "arch.cuh"
#include "pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if !SIPHON_SM89
#include <cuda_fp4.h>
#endif
#include <cuda_fp8.h>
#include <cfloat>
#include <climits>
#include <cstdint>

namespace {

constexpr int kMaxSegments = 8;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr float kFp32Max = 3.4028234663852886e38f;

struct Segment {
    const uint8_t* w;
    const uint8_t* scale;
    const float* outer;
    const __nv_bfloat16* x;
    __nv_bfloat16* out;
    int rows;
    // Blocks of this segment and every segment before it.
    int block_end;
};
static_assert(sizeof(Segment) == 48, "Segment matches Rust layout and pdl_audit.py");

struct Batch {
    Segment seg[kMaxSegments];
    int count;
    // Nonzero when launched with programmatic dependent launch.
    int pdl;
};

// Streams weights and scales past L1: each byte is read once per step.
__device__ __forceinline__ uint4 load_weight(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t load_scale_u16(const uint16_t* p) {
    uint16_t v;
    asm volatile("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float load_outer(const float* p) {
    float v;
    asm volatile("ld.global.nc.L1::no_allocate.f32 %0, [%1];" : "=f"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ float lo(uint32_t pair) { return __uint_as_float(pair << 16); }
__device__ __forceinline__ float hi(uint32_t pair) { return __uint_as_float(pair & 0xffff0000u); }

// Two E2M1 values, low nibble first, widened exactly through F16.
__device__ __forceinline__ float2 e2m1x2_to_float2(uint32_t byte) {
#if SIPHON_SM89
    __half2_raw h;
    const uint32_t v = e2m1x2_to_f16x2_sm89(byte);
    h.x = uint16_t(v);
    h.y = uint16_t(v >> 16);
    return __half22float2(__half2(h));
#else
    const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(uint8_t(byte), __NV_E2M1);
    return __half22float2(__half2(h));
#endif
}

__device__ __forceinline__ float e4m3_to_float(uint32_t byte) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(uint8_t(byte), __NV_E4M3)));
}

// Sums 8 E2M1 weights in `w` times 8 BF16 inputs in `xv` into two F32 accumulators to
// hide FMA latency.
__device__ __forceinline__ void dot8_split(uint32_t w, uint4 xv, float& acc0, float& acc1) {
    const float2 a = e2m1x2_to_float2(w);
    const float2 b = e2m1x2_to_float2(w >> 8);
    const float2 c = e2m1x2_to_float2(w >> 16);
    const float2 d = e2m1x2_to_float2(w >> 24);
    acc0 = fmaf(lo(xv.x), a.x, acc0);
    acc1 = fmaf(hi(xv.x), a.y, acc1);
    acc0 = fmaf(lo(xv.y), b.x, acc0);
    acc1 = fmaf(hi(xv.y), b.y, acc1);
    acc0 = fmaf(lo(xv.z), c.x, acc0);
    acc1 = fmaf(hi(xv.z), c.y, acc1);
    acc0 = fmaf(lo(xv.w), d.x, acc0);
    acc1 = fmaf(hi(xv.w), d.y, acc1);
}

// Dot product of 32 E2M1 weights (two 16-value scale groups in `w` and `s`) with 32 BF16
// inputs stored transposed in `xs` at chunk `p` (stride `kRowChunks` between the 4 uint4s).
template <int kRowChunks>
__device__ __forceinline__ float dot32(uint4 w, uint32_t s, const uint4* xs, int p) {
    const float s0 = e4m3_to_float(s & 0xffu);
    const float s1 = e4m3_to_float(s >> 8);
    float g0a = 0.f, g0b = 0.f;
    dot8_split(w.x, xs[p], g0a, g0b);
    dot8_split(w.y, xs[kRowChunks + p], g0a, g0b);
    float g1a = 0.f, g1b = 0.f;
    dot8_split(w.z, xs[2 * kRowChunks + p], g1a, g1b);
    dot8_split(w.w, xs[3 * kRowChunks + p], g1a, g1b);
    return fmaf(s1, g1a + g1b, s0 * (g0a + g0b));
}

// Loads `K` BF16 inputs from `x_ptr` into `xs` transposed by `h = i & 3` so `dot32`
// accesses across consecutive chunks `p` are bank-conflict-free.
template <int K>
__device__ __forceinline__ void load_x_smem(const __nv_bfloat16* x_ptr, uint4* xs) {
    constexpr int kRowChunks = K / 32;
    constexpr int kXUnits = K / 8;
    constexpr int kXPerThread = (kXUnits + kThreads - 1) / kThreads;
    const auto* x = reinterpret_cast<const uint4*>(x_ptr);
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
            xs[(u & 3) * kRowChunks + (u >> 2)] = xv[i];
        }
    }
}

// True if (a, ia) beats (b, ib) under argmax_f32's rule: larger value wins, ties go to
// the smaller index, and NaN beats every number (first NaN wins).
__device__ __forceinline__ bool argmax_better(float a, int ia, float b, int ib) {
    const bool a_nan = isnan(a);
    const bool b_nan = isnan(b);
    if (a_nan || b_nan) {
        return a_nan && (!b_nan || ia < ib);
    }
    return a > b || (a == b && ia < ib);
}

// K inputs per row, R rows per warp, V 16-byte weight loads per lane.
template <int K, int R, int V>
__device__ void gemv_nvfp4(const Batch& b, uint4* xs) {
    constexpr int kRowChunks = K / 32;
    constexpr int kWarpChunks = R * kRowChunks;
    static_assert(kWarpChunks == 32 * V, "a warp's rows fill its loads exactly");
    constexpr int kRowsPerBlock = kWarps * R;

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
    const auto* w = reinterpret_cast<const uint4*>(seg.w + size_t(row0) * (K / 2));
    const auto* sc = reinterpret_cast<const uint16_t*>(seg.scale + size_t(row0) * (K / 16));

    uint4 wv[V];
    uint32_t sv[V];
#pragma unroll
    for (int v = 0; v < V; ++v) {
        const int j = v * 32 + lane;
        wv[v] = load_weight(w + j);
        sv[v] = load_scale_u16(sc + j);
    }
    const float outer = lane < R ? load_outer(seg.outer + row0 + lane) : 0.f;

    if (b.pdl) {
        pdl_wait();
    }
    pdl_trigger();

    load_x_smem<K>(seg.x, xs);
    __syncthreads();

    float acc[R] = {};
#pragma unroll
    for (int v = 0; v < V; ++v) {
        const int j = v * 32 + lane;
        const int r = j / kRowChunks;
        const int p = j - r * kRowChunks;
        const float d = dot32<kRowChunks>(wv[v], sv[v], xs, p);
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            if (rr * kRowChunks <= v * 32 + 31 && (rr + 1) * kRowChunks > v * 32) {
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
    float mine = acc[0];
#pragma unroll
    for (int rr = 1; rr < R; ++rr) {
        mine = lane == rr ? acc[rr] : mine;
    }
    const int row = row0 + lane;
    if (lane < R && row < seg.rows) {
        seg.out[row] = __float2bfloat16_rn(outer * mine);
    }
}

// Single-product greedy LM head with fused block-strided argmax: each block loads `x`
// into shared memory once, strides across tiles of `kWarps * R` rows (16 rows for K=2304),
// rounds each logit to BF16 (matching `gemv_nvfp4` bit for bit), masks `row < vocab`, and
// writes one `(max_val, max_idx)` partial per block for `argmax_partials`.
template <int K, int R, int V>
__device__ void gemv_nvfp4_argmax(const uint8_t* weight, const uint8_t* scales,
                                  const float* outer_ptr, const __nv_bfloat16* x_in,
                                  float* partial_vals, int* partial_idxs, int vocab, int pdl,
                                  uint4* xs) {
    constexpr int kRowChunks = K / 32;
    constexpr int kWarpChunks = R * kRowChunks;
    static_assert(kWarpChunks == 32 * V, "a warp's rows fill its loads exactly");
    constexpr int kRowsPerBlock = kWarps * R;

    __shared__ float warp_vals[kWarps];
    __shared__ int warp_idxs[kWarps];

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int total_tiles = (vocab + kRowsPerBlock - 1) / kRowsPerBlock;

    // Preload the block's first tile of read-only weights and scales before pdl_wait.
    int tile = int(blockIdx.x);
    int row0 = tile * kRowsPerBlock + warp * R;
    uint4 wv[V];
    uint32_t sv[V];
    float outer = 0.f;
    if (tile < total_tiles) {
        const auto* w = reinterpret_cast<const uint4*>(weight + size_t(row0) * (K / 2));
        const auto* sc = reinterpret_cast<const uint16_t*>(scales + size_t(row0) * (K / 16));
#pragma unroll
        for (int v = 0; v < V; ++v) {
            const int j = v * 32 + lane;
            wv[v] = load_weight(w + j);
            sv[v] = load_scale_u16(sc + j);
        }
        outer = lane < R ? load_outer(outer_ptr + row0 + lane) : 0.f;
    }

    if (pdl) {
        pdl_wait();
    }
    pdl_trigger();

    load_x_smem<K>(x_in, xs);
    __syncthreads();

    float best_v = -INFINITY;
    int best_i = INT_MAX;

    while (tile < total_tiles) {
        float acc[R] = {};
#pragma unroll
        for (int v = 0; v < V; ++v) {
            const int j = v * 32 + lane;
            const int r = j / kRowChunks;
            const int p = j - r * kRowChunks;
            const float d = dot32<kRowChunks>(wv[v], sv[v], xs, p);
#pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                if (rr * kRowChunks <= v * 32 + 31 && (rr + 1) * kRowChunks > v * 32) {
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
        float mine = acc[0];
#pragma unroll
        for (int rr = 1; rr < R; ++rr) {
            mine = lane == rr ? acc[rr] : mine;
        }
        const int row = row0 + lane;
        if (lane < R && row < vocab) {
            const float v = __bfloat162float(__float2bfloat16_rn(outer * mine));
            if (argmax_better(v, row, best_v, best_i)) {
                best_v = v;
                best_i = row;
            }
        }

        tile += int(gridDim.x);
        if (tile < total_tiles) {
            row0 = tile * kRowsPerBlock + warp * R;
            const auto* w = reinterpret_cast<const uint4*>(weight + size_t(row0) * (K / 2));
            const auto* sc = reinterpret_cast<const uint16_t*>(scales + size_t(row0) * (K / 16));
#pragma unroll
            for (int v = 0; v < V; ++v) {
                const int j = v * 32 + lane;
                wv[v] = load_weight(w + j);
                sv[v] = load_scale_u16(sc + j);
            }
            outer = lane < R ? load_outer(outer_ptr + row0 + lane) : 0.f;
        }
    }

    // Reduce across lanes 0..R-1 (all lanes >= R hold (-INFINITY, INT_MAX)).
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
        if (argmax_better(ov, oi, best_v, best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    if (lane == 0) {
        warp_vals[warp] = best_v;
        warp_idxs[warp] = best_i;
    }
    __syncthreads();
    if (warp == 0) {
        best_v = lane < kWarps ? warp_vals[lane] : -INFINITY;
        best_i = lane < kWarps ? warp_idxs[lane] : INT_MAX;
#pragma unroll
        for (int offset = kWarps / 2; offset >= 1; offset /= 2) {
            const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
            const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
            if (argmax_better(ov, oi, best_v, best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            partial_vals[blockIdx.x] = best_v;
            partial_idxs[blockIdx.x] = best_i;
        }
    }
}

// Quantization helpers (Four Over Six, matching expert_matmul.cu).

__device__ __forceinline__ float e4m3_round(float x) {
    const __nv_fp8_e4m3 q(x);
    return float(q);
}

__device__ __forceinline__ uint32_t cvt_e2m1x2(float lo_in, float hi_in, float& lo_val,
                                               float& hi_val) {
#if SIPHON_SM89
    auto code = [](float y) -> uint32_t {
        const float a = fabsf(y);
        const uint32_t c = uint32_t(a > 0.25f) + uint32_t(a >= 0.75f) + uint32_t(a > 1.25f) +
                           uint32_t(a >= 1.75f) + uint32_t(a > 2.5f) + uint32_t(a >= 3.5f) +
                           uint32_t(a > 5.f);
        return isnan(y) ? 7u : c | (__float_as_uint(y) >> 31 << 3);
    };
    const uint32_t byte = code(lo_in) | (code(hi_in) << 4);
    const float2 vals = e2m1x2_to_float2(byte);
    lo_val = vals.x;
    hi_val = vals.y;
    return byte;
#else
    uint32_t byte;
    uint32_t halves;
    asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %3, %2; cvt.u32.u8 %0, b; "
        "cvt.rn.f16x2.e2m1x2 %1, b; }"
        : "=r"(byte), "=r"(halves)
        : "f"(lo_in), "f"(hi_in));
    const __half2 h = *reinterpret_cast<const __half2*>(&halves);
    lo_val = __low2float(h);
    hi_val = __high2float(h);
    return byte;
#endif
}

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float* x) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
        x[2 * i] = f.x;
        x[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ float fp4_candidate(const float (&x)[16], float scale, float inverse,
                                               float outer, uint2& packed) {
    float error[16];
    uint32_t bytes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float lv, hv;
        bytes[i] = cvt_e2m1x2(__fmul_rn(x[2 * i], inverse), __fmul_rn(x[2 * i + 1], inverse), lv, hv);
        const float d0 = __fsub_rn(__fmul_rn(__fmul_rn(lv, scale), outer), x[2 * i]);
        const float d1 = __fsub_rn(__fmul_rn(__fmul_rn(hv, scale), outer), x[2 * i + 1]);
        error[2 * i] = __fmul_rn(d0, d0);
        error[2 * i + 1] = __fmul_rn(d1, d1);
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

__device__ void quantize_row_nvfp4(const __nv_bfloat16* __restrict__ row, int k,
                                   uint8_t* __restrict__ values, uint8_t* __restrict__ scales,
                                   float* __restrict__ outer) {
    float amax = 0.f;
    for (int b = threadIdx.x % 32; b < k / 16; b += 32) {
        float x[16];
        load8(row + b * 16, x);
        load8(row + b * 16 + 8, x + 8);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            amax = fmaxf(amax, fabsf(x[i]));
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
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

}  // namespace

// Entry points by k: the launcher picks one. Dynamic shared memory: k * 2 bytes.
#define GEMV_NVFP4_ENTRY(name, K, R, V)                                                    \
    extern "C" __global__ void __launch_bounds__(kThreads)                                 \
    name(const __grid_constant__ Batch batch) {                                            \
        extern __shared__ uint4 x_smem[];                                                  \
        gemv_nvfp4<K, R, V>(batch, x_smem);                                                \
    }

// LM head & attention input projections (16 rows/block, 4 rows/warp, 9 exact loads/lane).
GEMV_NVFP4_ENTRY(gemv_nvfp4_k2304, 2304, 4, 9)
// Attention output projection (8 rows/block, 2 rows/warp, 6 exact loads/lane).
GEMV_NVFP4_ENTRY(gemv_nvfp4_k3072, 3072, 2, 6)

// Fused greedy B=1 LM head argmax (16 rows/tile, 4 rows/warp, 9 exact loads/lane).
extern "C" __global__ void __launch_bounds__(kThreads)
gemv_nvfp4_k2304_argmax(const uint8_t* weight, const uint8_t* scales, const float* outer,
                        const __nv_bfloat16* x_in, float* partial_vals, int* partial_idxs,
                        int vocab, int pdl) {
    extern __shared__ uint4 x_smem[];
    gemv_nvfp4_argmax<2304, 4, 9>(weight, scales, outer, x_in, partial_vals, partial_idxs,
                                  vocab, pdl, x_smem);
}

// Quantizes a BF16 matrix [rows, k] to the NVFP4 layout above: one warp per row, 4 rows
// per block. `k` must be a multiple of 32. The caller zeroes the padding rows.
extern "C" __global__ void __launch_bounds__(kThreads)
quantize_nvfp4_rows(const __nv_bfloat16* __restrict__ w, int rows, int k,
                    uint8_t* __restrict__ values, uint8_t* __restrict__ scales,
                    float* __restrict__ outer) {
    const int row = blockIdx.x * kWarps + threadIdx.x / 32;
    if (row < rows) {
        quantize_row_nvfp4(w + size_t(row) * k, k, values + size_t(row) * (k / 2),
                           scales + size_t(row) * (k / 16), outer + row);
    }
}

// Quantizes `rows` indexed rows (`ids[0..rows]`) of BF16 matrix `w` into rows
// `dst_offset..dst_offset + rows` of an Nvfp4Matrix buffer (for dynamic draft shortlists).
extern "C" __global__ void __launch_bounds__(kThreads)
quantize_nvfp4_rows_indexed(const __nv_bfloat16* __restrict__ w, const int* __restrict__ ids,
                            int rows, int k, int dst_offset, uint8_t* __restrict__ values,
                            uint8_t* __restrict__ scales, float* __restrict__ outer) {
    const int r = blockIdx.x * kWarps + threadIdx.x / 32;
    if (r < rows) {
        const int src_row = ids[r];
        const int dst_row = dst_offset + r;
        quantize_row_nvfp4(w + size_t(src_row) * k, k, values + size_t(dst_row) * (k / 2),
                           scales + size_t(dst_row) * (k / 16), outer + dst_row);
    }
}
