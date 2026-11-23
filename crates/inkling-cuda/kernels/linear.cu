// Dense matrix-vector products, bit-exact with MLX 0.32.4's M = 1 path.
//
// For one input row, `x @ W.T` with W stored [n, k] row-major runs MLX's
// GEMVKernel (kernels/gemv.h) with W as the matrix. Each kernel here computes
// every input row as its own M = 1 product with the same sums. MLX picks the
// blocking in matmul.cpp; TN is always 4, and SN and BN depend on the shape:
//
//   k <= 64       SN = 4,  BN = 1   8 rows per simdgroup, 4 lanes per row
//   k >= 16 * n   SN = 32, BN = 8   8 simdgroups split k for one row
//   otherwise     SN = 32, BN = 1   one simdgroup per row
//
// BM and TM only decide which rows a threadgroup owns, so they change no sum.
// For each output row:
// 1. Lane `l` of k-slice `s` owns columns i * BN*SN*4 + (s*SN + l)*4 + t, for
//    i ascending and t = 0..3, and accumulates `acc += w * x` from zero. Metal
//    contracts that into an FMA: the F32 path matches MLX only with FMA. For
//    BF16 the product is exact, so FMA and multiply-add agree. Columns past k
//    add fma(0, 0, acc), as MLX's zero-filled `load_safe` does.
// 2. The SN lanes of a row reduce with `simd_shuffle_down` by SN/2, ..., 1. That
//    is a tree with descending offsets, not `simd_sum`'s butterfly.
// 3. With BN = 8, slice 0 adds slices 1 to 7 in order.
// A NumPy emulation of these steps matched MLX bit for bit on every shape the
// model uses, for both BF16 and F32.

#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

namespace {

constexpr int kTN = 4;

// Metal flushes F32 subnormals to zero, so these use the PTX .ftz forms.
__device__ __forceinline__ float ffma(float a, float b, float c) {
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

__device__ __forceinline__ float fadd(float a, float b) {
    float r;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ void load4(const __nv_bfloat16* p, float (&v)[kTN]) {
    const uint2 packed = *reinterpret_cast<const uint2*>(p);
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
    for (int t = 0; t < kTN; ++t) {
        v[t] = __bfloat162float(vals[t]);
    }
}

__device__ __forceinline__ void load4(const float* p, float (&v)[kTN]) {
    const float4 packed = *reinterpret_cast<const float4*>(p);
    v[0] = packed.x;
    v[1] = packed.y;
    v[2] = packed.z;
    v[3] = packed.w;
}

// Round to nearest even, flushing subnormals as MLX's cast does.
__device__ __forceinline__ void store(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16(fabsf(v) < FLT_MIN ? copysignf(0.f, v) : v);
}

__device__ __forceinline__ void store(float* p, float v) {
    *p = v;
}

// out[r, row] = sum_c w[row, c] * x[r, c] for input row r = blockIdx.y. A block
// holds `BN == 1 ? 4 : 1` warps along the output rows times BN warps along k.
// Every thread stays to the end because the shuffles and barrier need the
// whole warp and block; threads past n compute on row 0 and store nothing.
template <typename T, int SN, int BN>
__device__ void gemv(const T* w, const T* x, T* out, int n, int k) {
    constexpr int kRowsPerWarp = 32 / SN;
    constexpr int kWarpsM = BN == 1 ? 4 : 1;
    constexpr int kRowsPerBlock = kWarpsM * kRowsPerWarp;
    constexpr int kBlockK = BN * SN * kTN;

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int thr_n = lane % SN;
    const int slice = warp % BN;
    const int row = blockIdx.x * kRowsPerBlock + (warp / BN) * kRowsPerWarp + lane / SN;
    const bool live = row < n;

    const T* w_row = w + size_t(live ? row : 0) * k;
    x += size_t(blockIdx.y) * k;
    out += size_t(blockIdx.y) * n;

    float acc = 0.f;
    for (int base = 0; base < k; base += kBlockK) {
        const int col = base + (slice * SN + thr_n) * kTN;
        float wv[kTN] = {0.f, 0.f, 0.f, 0.f};
        float xv[kTN] = {0.f, 0.f, 0.f, 0.f};
        // k is a multiple of 4, so a group of 4 columns is wholly in or out.
        if (col < k) {
            load4(w_row + col, wv);
            load4(x + col, xv);
        }
#pragma unroll
        for (int t = 0; t < kTN; ++t) {
            acc = ffma(wv[t], xv[t], acc);
        }
    }

#pragma unroll
    for (int offset = SN / 2; offset >= 1; offset /= 2) {
        acc = fadd(acc, __shfl_down_sync(0xffffffffu, acc, offset));
    }

    if constexpr (BN > 1) {
        static_assert(kWarpsM == 1, "a split-k block owns one warp of rows");
        __shared__ float partial[BN][kRowsPerWarp];
        if (thr_n == 0) {
            partial[slice][lane / SN] = acc;
        }
        __syncthreads();
        if (slice == 0 && thr_n == 0) {
#pragma unroll
            for (int s = 1; s < BN; ++s) {
                acc = fadd(acc, partial[s][lane / SN]);
            }
        }
    }

    if (live && slice == 0 && thr_n == 0) {
        store(out + row, acc);
    }
}

}  // namespace

// One entry point per dtype and MLX blocking. Names give SN and BN; the launcher
// picks one with MLX's rule above. Grid: (ceil(n / rows per block), input rows).
#define GEMV_ENTRY(name, T, SN, BN)                                                         \
    extern "C" __global__ void __launch_bounds__(32 * BN * (BN == 1 ? 4 : 1))               \
    name(const T* w, const T* x, T* out, int n, int k) {                                    \
        gemv<T, SN, BN>(w, x, out, n, k);                                                   \
    }

GEMV_ENTRY(gemv_bf16_sn4_bn1, __nv_bfloat16, 4, 1)
GEMV_ENTRY(gemv_bf16_sn32_bn1, __nv_bfloat16, 32, 1)
GEMV_ENTRY(gemv_bf16_sn32_bn8, __nv_bfloat16, 32, 8)
GEMV_ENTRY(gemv_f32_sn4_bn1, float, 4, 1)
GEMV_ENTRY(gemv_f32_sn32_bn1, float, 32, 1)
GEMV_ENTRY(gemv_f32_sn32_bn8, float, 32, 8)
