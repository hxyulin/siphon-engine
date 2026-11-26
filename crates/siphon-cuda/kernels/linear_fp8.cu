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

#include "mxfp8.cuh"
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

// ---------------------------------------------------------------------------
// Matrix products `out = x @ W^T` on FP8 tensor cores, for prefill (opt-in): W is an FP8
// matrix in the layout above, x is [m, k] BF16 quantized by `quantize_rows_mxfp8_dense`
// to MXFP8 (E4M3 values, a UE8M0 scale per 32, see mxfp8.cuh), out is [m, n] BF16.
//
// Each product of 32 values runs on the block-scaled FP8 MMA (`kind::mxf8f6f4`, m16n8k32,
// F32 accumulators) with the input's scales as A scales and B scales 1, so the sums are
// exact products of the E4M3 values in F32; the weight row's F32 scale multiplies each
// sum once before it rounds to BF16, as the FP8 GEMV does.
//
// One launch runs up to kMaxProducts products that share m and k. A block of 8 warps
// computes one 128 x 128 tile of one product, m fastest so that blocks sharing a weight
// tile run together; each warp 64 rows by 32 columns. A stage holds 128 values of each
// of the tile's x and W rows and the x rows' 4 scales, in 3 stages, with rows XOR-swizzled
// in 16-byte chunks so that ldmatrix reads are free of bank conflicts. Rows past m and n
// load as zeros, and stores past them are skipped.

namespace gemm8 {

constexpr int kMaxProducts = 8;
constexpr int kBm = 128;
constexpr int kBn = 128;
constexpr int kBk = 128;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kThreads = 32 * kWarpsM * kWarpsN;
constexpr int kStages = 3;
constexpr int kWarpRows = kBm / kWarpsM;
constexpr int kWarpCols = kBn / kWarpsN;
constexpr int kMi = kWarpRows / 16;
constexpr int kNi = kWarpCols / 8;
constexpr int kStageA = kBm * kBk;
constexpr int kStageB = kBn * kBk;
constexpr int kStageScales = kBm * 4;
constexpr int kStage = kStageA + kStageB + kStageScales;
constexpr int kSmemBytes = kStages * kStage;
static_assert(kSmemBytes <= 99 * 1024, "one block fits in an SM's shared memory");

struct Product {
    const uint8_t* w;
    const float* w_scale;
    const uint8_t* x;
    const uint8_t* x_scale;
    __nv_bfloat16* out;
    int n;
    // Tiles of this product and every product before it.
    int tile_end;
};

struct Batch {
    Product p[kMaxProducts];
    int count;
    int m;
    int k;
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src),
                 "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src),
                 "r"(valid ? 4 : 0));
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

// c += a b: the A scales of rows g and g + 8 from byte kByte of lanes 4g and 4g + 1's
// `scale_a`, B scales 1.
template <int kByte>
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                    uint32_t b1, uint32_t scale_a) {
#if SIPHON_SM89
    __trap();
#else
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3."
        "e4m3.f32.ue8m0 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
        "{%13}, {%12,%12};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(scale_a),
          "h"(uint16_t(kByte)), "h"(uint16_t(0)), "r"(0x7f7f7f7fu));
#endif
}

// Byte offset of 16-byte chunk `chunk` of row `row` in a [rows, 128] byte tile.
__device__ __forceinline__ int offset(int row, int chunk) {
    return row * kBk + ((chunk ^ (row & 7)) << 4);
}

__device__ void gemm(const Batch& b, uint8_t* smem) {
    const int block = blockIdx.x;
    int s = 0;
    int tile_begin = 0;
    while (s + 1 < b.count && block >= b.p[s].tile_end) {
        tile_begin = b.p[s].tile_end;
        ++s;
    }
    const Product& p = b.p[s];
    const int m = b.m, n = p.n, k = b.k;
    const int tiles_m = (m + kBm - 1) / kBm;
    const int tile = block - tile_begin;
    const int m0 = (tile % tiles_m) * kBm;
    const int n0 = (tile / tiles_m) * kBn;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int wn = warp % kWarpsN;
    const int wm = warp / kWarpsN;

    // Copies: each thread 4 chunks of x rows and 4 of W rows per stage (rows tid / 8 +
    // 32i, chunk tid % 8), and threads below 128 the 4 scales of x row tid.
    const int chunk = tid % 8;
    const uint8_t* a_src[4];
    const uint8_t* b_src[4];
    uint32_t dst[4];
    uint32_t a_live = 0;
    uint32_t b_live = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = tid / 8 + 32 * i;
        dst[i] = offset(row, chunk);
        const bool a_ok = m0 + row < m;
        const bool b_ok = n0 + row < n;
        a_live |= uint32_t(a_ok) << i;
        b_live |= uint32_t(b_ok) << i;
        a_src[i] = p.x + size_t(a_ok ? m0 + row : 0) * k + chunk * 16;
        b_src[i] = p.w + size_t(b_ok ? n0 + row : 0) * k + chunk * 16;
    }
    const bool scale_ok = tid < kBm && m0 + tid < m;
    const uint8_t* scale_src = p.x_scale + size_t(scale_ok ? m0 + tid : 0) * (k / 32);

    const uint32_t smem0 = smem_addr(smem);
    auto load = [&](int stage, int step) {
        const uint32_t base = smem0 + stage * kStage;
        const int k0 = step * kBk;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            cp_async16(base + dst[i], a_src[i] + k0, a_live >> i & 1);
            cp_async16(base + kStageA + dst[i], b_src[i] + k0, b_live >> i & 1);
        }
        if (tid < kBm) {
            cp_async4(base + kStageA + kStageB + 4 * tid, scale_src + step * 4, scale_ok);
        }
    };

    // ldmatrix rows of this lane: x rows lane % 16, chunk lane / 16 of each 16-row
    // fragment; W rows lane % 8 + 8 (lane / 16), chunk lane / 8 % 2 of each pair of
    // 8-column fragments. Fragment origins are multiples of 8 rows, so the swizzle
    // depends only on the lane.
    const int a_row = wm * kWarpRows + lane % 16;
    const int b_row = wn * kWarpCols + lane % 8 + lane / 16 * 8;
    // The x row whose scales this lane passes: g (t = 0) or g + 8 (t = 1).
    const int scale_row = wm * kWarpRows + lane / 4 + 8 * (lane % 2);

    float acc[kMi][kNi][4] = {};
    const int steps = k / kBk;
#pragma unroll
    for (int st = 0; st < kStages - 1; ++st) {
        if (st < steps) {
            load(st, st);
        }
        cp_async_commit();
    }

    int stage = 0;
    for (int step = 0; step < steps; ++step) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        const int next = step + kStages - 1;
        if (next < steps) {
            load((stage + kStages - 1) % kStages, next);
        }
        cp_async_commit();

        const uint32_t base = smem0 + stage * kStage;
        const uint32_t* scales =
            reinterpret_cast<const uint32_t*>(smem + stage * kStage + kStageA + kStageB);
        uint32_t sa[kMi];
#pragma unroll
        for (int mi = 0; mi < kMi; ++mi) {
            sa[mi] = scales[scale_row + 16 * mi];
        }
        auto k_step = [&]<int kK>() {
            uint32_t bf[kNi][2];
#pragma unroll
            for (int nj = 0; nj < kNi; nj += 2) {
                uint32_t r[4];
                ldmatrix_x4(r, base + kStageA + offset(b_row + nj * 8, 2 * kK + lane / 8 % 2));
                bf[nj][0] = r[0];
                bf[nj][1] = r[1];
                bf[nj + 1][0] = r[2];
                bf[nj + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < kMi; ++mi) {
                uint32_t af[4];
                ldmatrix_x4(af, base + offset(a_row + mi * 16, 2 * kK + lane / 16));
#pragma unroll
                for (int ni = 0; ni < kNi; ++ni) {
                    mma<kK>(acc[mi][ni], af, bf[ni][0], bf[ni][1], sa[mi]);
                }
            }
        };
        k_step.template operator()<0>();
        k_step.template operator()<1>();
        k_step.template operator()<2>();
        k_step.template operator()<3>();
        stage = stage + 1 == kStages ? 0 : stage + 1;
    }

    const int g = lane / 4, t = lane % 4;
#pragma unroll
    for (int ni = 0; ni < kNi; ++ni) {
        const int col = n0 + wn * kWarpCols + ni * 8 + 2 * t;
        if (col >= n) {
            continue;
        }
        const bool second = col + 1 < n;
        const float s0 = p.w_scale[col];
        const float s1 = second ? p.w_scale[col + 1] : 0.f;
#pragma unroll
        for (int mi = 0; mi < kMi; ++mi)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * kWarpRows + mi * 16 + g + 8 * h;
                if (row >= m) {
                    continue;
                }
                __nv_bfloat16* dst_row = p.out + size_t(row) * n + col;
                const float v0 = acc[mi][ni][2 * h] * s0;
                const float v1 = acc[mi][ni][2 * h + 1] * s1;
                if (second && n % 2 == 0) {
                    *reinterpret_cast<__nv_bfloat162*>(dst_row) = __floats2bfloat162_rn(v0, v1);
                } else {
                    dst_row[0] = __float2bfloat16_rn(v0);
                    if (second) {
                        dst_row[1] = __float2bfloat16_rn(v1);
                    }
                }
            }
    }
}

}  // namespace gemm8

// The FP8 GEMM entry point. Grid: one block per 128 x 128 tile of every product; dynamic
// shared memory gemm8::kSmemBytes (99,840 bytes). k must be a multiple of 128.
extern "C" __global__ void __launch_bounds__(gemm8::kThreads, 1)
    gemm_fp8_128x128(const __grid_constant__ gemm8::Batch batch) {
    extern __shared__ __align__(16) uint8_t gemm8_smem[];
    gemm8::gemm(batch, gemm8_smem);
}

// Quantizes `rows` BF16 rows of k values (`x`) to MXFP8 for gemm_fp8: E4M3 `values`
// [rows, k] and UE8M0 `scales` [rows, k / 32]. One warp per row, 8 rows per block of 256
// threads; k must be a multiple of 32.
extern "C" __global__ void __launch_bounds__(256)
    quantize_rows_mxfp8_dense(const __nv_bfloat16* __restrict__ x, int rows, int k,
                              uint8_t* __restrict__ values, uint8_t* __restrict__ scales) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32;
    if (row < rows) {
        quantize_row_mxfp8<false>(x + size_t(row) * k, k, values + size_t(row) * k,
                                  scales + size_t(row) * (k / 32));
    }
}
