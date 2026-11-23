// Message-gate combine, F32/BF16 casts and argmax, following MLX 0.32.4 on Apple GPUs.
//
// - Casts: MLX's F32-to-BF16 cast rounds to nearest even, flushes subnormal inputs
//   to a signed zero and turns every NaN into 0x7fc0. CUDA's own conversion gives
//   0x7fff for NaN, so the cast here is written out. BF16-to-F32 is a 16-bit
//   shift in both, keeping subnormals and NaN payloads.
// - The combine copies the separate MLX ops of `MessageGate.__call__`, each
//   rounded on its own with subnormals flushed. Metal's tanh is not correctly
//   rounded and nothing here reproduces it. On the random F32 inputs of the misc
//   fixture, 29% of combine outputs differ from MLX with a correctly rounded tanh
//   (through double) and 36% with tanhf, each by at most 3.1e-7 * (|h| + |m|).
// - Argmax: `mx.argmax` returns the first maximum, and treats NaN as larger than
//   every number, so the first NaN wins when there is one.

#include <cuda_bf16.h>
#include <cfloat>
#include <climits>
#include <cstdint>

#include "pdl.cuh"

namespace {

constexpr int kArgmaxThreads = 1024;

__device__ __forceinline__ size_t grid_index() {
    return size_t(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ size_t grid_stride() {
    return size_t(gridDim.x) * blockDim.x;
}

__device__ __forceinline__ float fadd(float a, float b) {
    float r;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fsub(float a, float b) {
    float r;
    asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fmul(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float mlx_tanh(float x) {
    return float(tanh(double(x)));
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float x) {
    if (isnan(x)) {
        return 0x7fc0;
    }
    uint32_t bits = __float_as_uint(x);
    if (fabsf(x) < FLT_MIN) {
        return uint16_t((bits >> 16) & 0x8000);
    }
    bits += 0x7fff + ((bits >> 16) & 1);
    return uint16_t(bits >> 16);
}

// True if (a, ia) should be the argmax over (b, ib).
__device__ __forceinline__ bool argmax_better(float a, int ia, float b, int ib) {
    const bool a_nan = isnan(a);
    const bool b_nan = isnan(b);
    if (a_nan || b_nan) {
        return a_nan && (!b_nan || ia < ib);
    }
    return a > b || (a == b && ia < ib);
}

}  // namespace

// out = (1 - tanh(f_pre)) * h + tanh(g_pre) * m over n F32 elements.
extern "C" __global__ void message_gate_combine_f32(const float* h, const float* m,
                                                    const float* f_pre, const float* g_pre,
                                                    float* out, size_t n) {
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        const float forget = mlx_tanh(f_pre[i]);
        const float gain = mlx_tanh(g_pre[i]);
        out[i] = fadd(fmul(fsub(1.f, forget), h[i]), fmul(gain, m[i]));
    }
}

extern "C" __global__ void cast_f32_to_bf16(const float* x, uint16_t* out, size_t n) {
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        out[i] = f32_to_bf16_bits(x[i]);
    }
}

extern "C" __global__ void cast_bf16_to_f32(const uint16_t* x, float* out, size_t n) {
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        out[i] = __uint_as_float(uint32_t(x[i]) << 16);
    }
}

// out[r] = index of the largest of the n values in row r of x. One block per row.
// The comparison orders every (value, index) pair, so the reduction order does
// not change the result.
extern "C" __global__ void __launch_bounds__(kArgmaxThreads)
argmax_f32(const float* x, int* out, int n) {
    __shared__ float warp_vals[kArgmaxThreads / 32];
    __shared__ int warp_idxs[kArgmaxThreads / 32];
    const float* row = x + size_t(blockIdx.x) * n;

    // Threads without an element hold (-inf, INT_MAX), which loses to every element.
    float best = -INFINITY;
    int best_idx = INT_MAX;
    for (int i = threadIdx.x; i < n; i += kArgmaxThreads) {
        const float v = row[i];
        if (argmax_better(v, i, best, best_idx)) {
            best = v;
            best_idx = i;
        }
    }

    const auto warp_reduce = [&] {
#pragma unroll
        for (int offset = 16; offset >= 1; offset /= 2) {
            const float v = __shfl_xor_sync(0xffffffffu, best, offset);
            const int idx = __shfl_xor_sync(0xffffffffu, best_idx, offset);
            if (argmax_better(v, idx, best, best_idx)) {
                best = v;
                best_idx = idx;
            }
        }
    };

    warp_reduce();
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) {
        warp_vals[warp] = best;
        warp_idxs[warp] = best_idx;
    }
    __syncthreads();
    if (warp == 0) {
        best = warp_vals[lane];
        best_idx = warp_idxs[lane];
        warp_reduce();
        if (lane == 0) {
            out[blockIdx.x] = best_idx;
        }
    }
}

namespace {

constexpr int kHidden = 2304;
constexpr int kGateRank = 64;
constexpr int kGateThreads = 288;
constexpr int kGateWarps = kGateThreads / 32;
// gemv_f32_sn4_bn1's layout: 4 lanes per output row, 8 rows per warp. Lane l of a row
// sums columns 16 i + 4 l + 0..3 for i = 0..3.
constexpr int kLanesPerRow = 4;
constexpr int kRowsPerWarp = 32 / kLanesPerRow;
constexpr int kGateRowsPerBlock = kGateWarps * kRowsPerWarp;
constexpr int kGateBlocks = kHidden / kGateRowsPerBlock;
constexpr int kGateSteps = kGateRank / (4 * kLanesPerRow);
static_assert(kHidden % kGateRowsPerBlock == 0, "blocks cover the row");
// The last block repeats the 576-thread sums of rms_norm_f32_2304 as virtual threads
// t and t + 288.
constexpr int kNormVirtualWarps = 2 * kGateWarps;
static_assert(kGateThreads * 8 == kHidden, "two runs of 4 per thread cover the row");

__device__ __forceinline__ float ffma(float a, float b, float c) {
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

}  // namespace

// The end of MessageGate for one row, from the BF16 down projection `down` ([64]):
// f_pre = forget @ down, g_pre = gain @ down (BF16 weights [2304, 64], widened to F32
// as they are read), then out = (1 - tanh(f_pre)) * h + tanh(g_pre) * m. Each output
// row sums in the order of MLX's GEMV for k = 64; tanh is CUDA's tanhf.
//
// With `normed` not null, also writes normed = bf16(rms_norm(out) * norm_weight), the
// RMSNorm after every query, key and value gate, summed as rms_norm_f32_2304 does.
// The last block to finish, found with `counter`, applies it. `counter` must start
// at 0 and is reset by that block.
namespace {

__device__ __forceinline__ void
message_gate_out_row(const __nv_bfloat16* __restrict__ down, const __nv_bfloat16* __restrict__ forget,
                     const __nv_bfloat16* __restrict__ gain, const float* h, const float* m, float* out,
                     const __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* normed,
                     unsigned* counter, float eps) {
    __shared__ float vw[kNormVirtualWarps];
    __shared__ bool last;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int sub = lane % kLanesPerRow;
    const int row = blockIdx.x * kGateRowsPerBlock + warp * kRowsPerWarp + lane / kLanesPerRow;

    uint2 fw[kGateSteps];
    uint2 gw[kGateSteps];
    float x[kGateSteps][4];
#pragma unroll
    for (int i = 0; i < kGateSteps; ++i) {
        const int col = i * 4 * kLanesPerRow + sub * 4;
        fw[i] = *reinterpret_cast<const uint2*>(forget + size_t(row) * kGateRank + col);
        gw[i] = *reinterpret_cast<const uint2*>(gain + size_t(row) * kGateRank + col);
    }
    pdl_trigger();
    pdl_wait();
#pragma unroll
    for (int i = 0; i < kGateSteps; ++i) {
        const int col = i * 4 * kLanesPerRow + sub * 4;
        const uint2 packed = *reinterpret_cast<const uint2*>(down + col);
        const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            x[i][k] = __bfloat162float(vals[k]);
        }
    }
    float f = 0.f;
    float g = 0.f;
#pragma unroll
    for (int i = 0; i < kGateSteps; ++i) {
        const auto* fv = reinterpret_cast<const __nv_bfloat16*>(&fw[i]);
        const auto* gv = reinterpret_cast<const __nv_bfloat16*>(&gw[i]);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            f = ffma(__bfloat162float(fv[k]), x[i][k], f);
            g = ffma(__bfloat162float(gv[k]), x[i][k], g);
        }
    }
    // MLX's simd_shuffle_down tree over the row's 4 lanes; lane 0 of the row ends with it.
#pragma unroll
    for (int offset = kLanesPerRow / 2; offset > 0; offset /= 2) {
        f = fadd(f, __shfl_down_sync(0xffffffffu, f, offset));
        g = fadd(g, __shfl_down_sync(0xffffffffu, g, offset));
    }
    if (sub == 0) {
        const float forget_t = tanhf(f);
        const float gain_t = tanhf(g);
        out[row] = fadd(fmul(fsub(1.f, forget_t), h[row]), fmul(gain_t, m[row]));
    }
    if (normed == nullptr) {
        return;
    }
    // Make this block's outputs visible to the last block before counting it as done.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(counter, 1u) == kGateBlocks - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    __threadfence();
    const int ca = 4 * threadIdx.x;
    const int cb = 4 * (threadIdx.x + kGateThreads);
    float a[4];
    float b[4];
    float wa[4];
    float wb[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a[i] = __ldcg(out + ca + i);
        b[i] = __ldcg(out + cb + i);
        wa[i] = __bfloat162float(norm_weight[ca + i]);
        wb[i] = __bfloat162float(norm_weight[cb + i]);
    }
    float pa = 0.f;
    float pb = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        pa = fmaf(a[i], a[i], pa);
        pb = fmaf(b[i], b[i], pb);
    }
    pa = warp_sum(pa);
    pb = warp_sum(pb);
    if (lane == 0) {
        vw[warp] = pa;
        vw[warp + kGateWarps] = pb;
    }
    __syncthreads();
    const float ss = warp_sum(lane < kNormVirtualWarps ? vw[lane] : 0.f);
    // Two correctly rounded float operations; rms_norm_f32_2304 rounds a double rsqrt once.
    const float inv = 1.f / sqrtf(ss / static_cast<float>(kHidden) + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        normed[ca + i] = __float2bfloat16((a[i] * inv) * wa[i]);
        normed[cb + i] = __float2bfloat16((b[i] * inv) * wb[i]);
    }
    if (threadIdx.x == 0) {
        *counter = 0;
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kGateThreads)
message_gate_out_2304(const __nv_bfloat16* __restrict__ down, const __nv_bfloat16* __restrict__ forget,
                      const __nv_bfloat16* __restrict__ gain, const float* h, const float* m, float* out,
                      const __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* normed,
                      unsigned* counter, float eps) {
    message_gate_out_row(down, forget, gain, h, m, out, norm_weight, normed, counter, eps);
}

// ---------------------------------------------------------------------------
// message_gate_out_2304 for the rows of a chunk, in two kernels: the gate outputs for a
// tile of rows and output columns that shares the weight loads, then the optional norm.
// Both keep message_gate_out_2304's order, so each row's results equal a single-row
// call's bit for bit: output (row, d) sums four chains, one per lane of its 4-lane
// group (columns 16 i + 4 l + 0..3), and adds them as (s0 + s2) + (s1 + s3).
// ---------------------------------------------------------------------------

namespace {

constexpr int kTileRows = 32;
constexpr int kTileCols = 64;
constexpr int kTileThreads = 256;
// Each thread holds 4 rows (rt + 8 i) by 2 adjacent columns (2 cp, 2 cp + 1). A warp
// spans 8 column pairs and 4 row groups.
constexpr int kThreadRows = 4;
constexpr int kColPairs = kTileCols / 2;
constexpr int kRowGroups = kTileRows / kThreadRows;
static_assert(kColPairs * kRowGroups == kTileThreads, "threads cover the tile");
// Rows of the down tile, padded so that 4 consecutive rows fall on different banks.
constexpr int kDownPitch = kGateRank + 4;
static_assert(kHidden % kTileCols == 0, "tiles cover the row");

}  // namespace

// out[row] = message_gate_out_2304's output for each row of a chunk: down [rows, 64],
// out [rows, 2304], and the h and m rows h_stride and m_stride floats apart (even, so
// pairs of floats stay aligned). Grid: (2304 / 64, ceil(rows / 32)), 256 threads.
extern "C" __global__ void __launch_bounds__(kTileThreads, 2)
message_gate_out_rows_2304(const __nv_bfloat16* __restrict__ down, const __nv_bfloat16* __restrict__ forget,
                           const __nv_bfloat16* __restrict__ gain, const float* h, int h_stride, const float* m,
                           int m_stride, float* out, int rows) {
    // ws[k][cp] = (forget[2 cp][k], gain[2 cp][k], forget[2 cp + 1][k], gain[2 cp + 1][k]).
    __shared__ float4 ws[kGateRank][kColPairs];
    __shared__ __align__(16) float ds[kTileRows][kDownPitch];
    const int tid = threadIdx.x;
    const int d0 = blockIdx.x * kTileCols;
    const int r0 = blockIdx.y * kTileRows;

    // 8 BF16 weights per load: thread idx takes column idx % 64 and k from 8 (idx / 64).
    constexpr int kWeightLoads = kTileCols * kGateRank / 8 / kTileThreads;
#pragma unroll
    for (int q = 0; q < kWeightLoads; ++q) {
        const int idx = tid + q * kTileThreads;
        const int col = idx % kTileCols;
        const int k0 = idx / kTileCols * 8;
        const size_t at = size_t(d0 + col) * kGateRank + k0;
        const uint4 fp = *reinterpret_cast<const uint4*>(forget + at);
        const uint4 gp = *reinterpret_cast<const uint4*>(gain + at);
        const auto* fv = reinterpret_cast<const __nv_bfloat16*>(&fp);
        const auto* gv = reinterpret_cast<const __nv_bfloat16*>(&gp);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            auto* dst = reinterpret_cast<float2*>(&ws[k0 + i][col / 2]) + col % 2;
            *dst = make_float2(__bfloat162float(fv[i]), __bfloat162float(gv[i]));
        }
    }
    pdl_trigger();
    pdl_wait();
    static_assert(kTileRows * kGateRank / 8 == kTileThreads, "one down load per thread");
    {
        const int row = tid / (kGateRank / 8);
        const int k0 = tid % (kGateRank / 8) * 8;
        uint4 p = make_uint4(0, 0, 0, 0);
        if (r0 + row < rows) {
            p = *reinterpret_cast<const uint4*>(down + size_t(r0 + row) * kGateRank + k0);
        }
        const auto* v = reinterpret_cast<const __nv_bfloat16*>(&p);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            ds[row][k0 + i] = __bfloat162float(v[i]);
        }
    }
    __syncthreads();

    const int lane = tid % 32;
    const int warp = tid / 32;
    const int cp = warp % 4 * 8 + lane % 8;
    const int rt = warp / 4 * 4 + lane / 8;
    const int d = d0 + 2 * cp;
    // h and m load during the products.
    float2 hv[kThreadRows];
    float2 mv[kThreadRows];
#pragma unroll
    for (int i = 0; i < kThreadRows; ++i) {
        const int row = min(r0 + rt + kRowGroups * i, rows - 1);
        hv[i] = *reinterpret_cast<const float2*>(h + size_t(row) * h_stride + d);
        mv[i] = *reinterpret_cast<const float2*>(m + size_t(row) * m_stride + d);
    }
    // Lane l of the single-row kernel's 4-lane group chains the k groups k4 = l, l + 4,
    // l + 8, l + 12. The lanes run in the order 0, 2, 1, 3, so that (s0 + s2) closes before
    // lanes 1 and 3 start. Index [i][j] is row rt + 8 i and column 2 cp + j.
    float f02[kThreadRows][2];
    float g02[kThreadRows][2];
    float f1[kThreadRows][2];
    float g1[kThreadRows][2];
    float fp[kThreadRows][2];
    float gp[kThreadRows][2];
#pragma unroll
    for (int step = 0; step < kLanesPerRow; ++step) {
        const int l = step == 1 ? 2 : step == 2 ? 1 : step;
        float f[kThreadRows][2] = {};
        float g[kThreadRows][2] = {};
#pragma unroll
        for (int i4 = 0; i4 < kGateRank / 16; ++i4) {
            const int k4 = 4 * i4 + l;
            float4 dv[kThreadRows];
#pragma unroll
            for (int i = 0; i < kThreadRows; ++i) {
                dv[i] = *reinterpret_cast<const float4*>(&ds[rt + kRowGroups * i][4 * k4]);
            }
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const float4 wv = ws[4 * k4 + kk][cp];
#pragma unroll
                for (int i = 0; i < kThreadRows; ++i) {
                    const float x = kk == 0 ? dv[i].x : kk == 1 ? dv[i].y : kk == 2 ? dv[i].z : dv[i].w;
                    f[i][0] = ffma(wv.x, x, f[i][0]);
                    g[i][0] = ffma(wv.y, x, g[i][0]);
                    f[i][1] = ffma(wv.z, x, f[i][1]);
                    g[i][1] = ffma(wv.w, x, g[i][1]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < kThreadRows; ++i) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                if (step == 0) {
                    f02[i][j] = f[i][j];
                    g02[i][j] = g[i][j];
                } else if (step == 1) {
                    f02[i][j] = fadd(f02[i][j], f[i][j]);
                    g02[i][j] = fadd(g02[i][j], g[i][j]);
                } else if (step == 2) {
                    f1[i][j] = f[i][j];
                    g1[i][j] = g[i][j];
                } else {
                    fp[i][j] = fadd(f02[i][j], fadd(f1[i][j], f[i][j]));
                    gp[i][j] = fadd(g02[i][j], fadd(g1[i][j], g[i][j]));
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kThreadRows; ++i) {
        const int row = r0 + rt + kRowGroups * i;
        if (row >= rows) {
            continue;
        }
        float o[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            o[j] = fadd(fmul(fsub(1.f, tanhf(fp[i][j])), j == 0 ? hv[i].x : hv[i].y),
                        fmul(tanhf(gp[i][j]), j == 0 ? mv[i].x : mv[i].y));
        }
        *reinterpret_cast<float2*>(out + size_t(row) * kHidden + d) = make_float2(o[0], o[1]);
    }
}

// normed = bf16(rms_norm(out) * norm_weight) for each row (blockIdx.x) of out and normed
// ([rows, 2304]): the norm at the end of message_gate_out_row, with the same sums, read
// with vector loads. 288 threads.
extern "C" __global__ void __launch_bounds__(kGateThreads)
message_gate_norm_rows_2304(const float* out, const __nv_bfloat16* __restrict__ norm_weight,
                            __nv_bfloat16* normed, float eps) {
    __shared__ float vw[kNormVirtualWarps];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const size_t at = size_t(blockIdx.x) * kHidden;
    const int ca = 4 * threadIdx.x;
    const int cb = 4 * (threadIdx.x + kGateThreads);
    const uint2 wpa = *reinterpret_cast<const uint2*>(norm_weight + ca);
    const uint2 wpb = *reinterpret_cast<const uint2*>(norm_weight + cb);
    pdl_trigger();
    pdl_wait();
    const float4 av = *reinterpret_cast<const float4*>(out + at + ca);
    const float4 bv = *reinterpret_cast<const float4*>(out + at + cb);
    const float a[4] = {av.x, av.y, av.z, av.w};
    const float b[4] = {bv.x, bv.y, bv.z, bv.w};
    const auto* wa = reinterpret_cast<const __nv_bfloat16*>(&wpa);
    const auto* wb = reinterpret_cast<const __nv_bfloat16*>(&wpb);
    float pa = 0.f;
    float pb = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        pa = fmaf(a[i], a[i], pa);
        pb = fmaf(b[i], b[i], pb);
    }
    pa = warp_sum(pa);
    pb = warp_sum(pb);
    if (lane == 0) {
        vw[warp] = pa;
        vw[warp + kGateWarps] = pb;
    }
    __syncthreads();
    const float ss = warp_sum(lane < kNormVirtualWarps ? vw[lane] : 0.f);
    // Two correctly rounded float operations; rms_norm_f32_2304 rounds a double rsqrt once.
    const float inv = 1.f / sqrtf(ss / static_cast<float>(kHidden) + eps);
    uint2 na;
    uint2 nb;
    auto* va = reinterpret_cast<__nv_bfloat16*>(&na);
    auto* vb = reinterpret_cast<__nv_bfloat16*>(&nb);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        va[i] = __float2bfloat16((a[i] * inv) * __bfloat162float(wa[i]));
        vb[i] = __float2bfloat16((b[i] * inv) * __bfloat162float(wb[i]));
    }
    *reinterpret_cast<uint2*>(normed + at + ca) = na;
    *reinterpret_cast<uint2*>(normed + at + cb) = nb;
}
