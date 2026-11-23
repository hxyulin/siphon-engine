// NVFP4 and MXFP8 expert matrix-vector products, bit-exact with MLX 0.32.4's
// decode path (`mx.gather_qmm` with one row per expert, which runs
// `fp_gather_qmv` or `fp_gather_qmv_fast` from kernels/fp_quantized.h).
//
// Each output value is one weight row dotted with one input row. MLX gives
// each output row one simdgroup. Lane `l` reads `kValues` consecutive inputs
// at `base + l * kValues` for each block of `32 * kValues`, and:
//
// - forms each product `x * w` in F32. A BF16 input times an E2M1 or E4M3
//   weight is exact in F32;
// - sums the products of its chunk starting from zero (`qdot`). NVFP4 adds
//   them in fours, `acc += ((p0 + p1) + p2) + p3`; MXFP8 adds them one at a
//   time;
// - adds `scale * acc` to its running result, where `scale` is the chunk's
//   group scale;
// - after the last block, reduces the 32 results with `simd_sum`, a butterfly
//   over lane offsets 1, 2, 4, 8 and 16.
//
// NVFP4 then multiplies by `global_scale / (448 * 6)` before rounding to BF16.
//
// `kValues` is MLX's `values_per_thread`. It doubles in the "fast" kernel,
// which MLX picks when the row count is a multiple of 8 and K is a multiple
// of 512 (NVFP4) or 256 (MXFP8). The launcher applies the same rule.
//
// MLX flushes subnormals and does not contract into FMA, so every add and
// multiply here is PTX `.rn.ftz`.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

// Experts each token routes to.
constexpr int kSlots = 8;
// Output rows per block, one warp each.
constexpr int kWarps = 8;
// `F8E4M3_MAX * F4E2M1_MAX` in MLX's `inv_scale_enc`.
constexpr float kNvfp4ScaleEnc = 448.f * 6.f;

__device__ float add_ftz(float a, float b) {
    float r;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ float mul_ftz(float a, float b) {
    float r;
    asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// Metal's `simd_sum`.
__device__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        v = add_ftz(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

__constant__ float kE2M1[16] = {
    0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

__device__ float e4m3(uint8_t bits) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(bits, __NV_E4M3)));
}

// MLX's `fp8_e8m0` to float: 2^(bits - 127), with 0 mapped to 2^-127, a
// subnormal that the flushing multiply turns into zero.
__device__ float e8m0(uint8_t bits) {
    return __uint_as_float(bits == 0 ? 0x400000u : uint32_t(bits) << 23);
}

// `qdot` for E2M1 weights: two values per byte, low nibble first.
template <int kValues>
__device__ float dot_e2m1(const __nv_bfloat16* x, const uint8_t* w) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kValues / 4; ++i) {
        const uint8_t b0 = w[2 * i];
        const uint8_t b1 = w[2 * i + 1];
        float t = add_ftz(mul_ftz(__bfloat162float(x[4 * i]), kE2M1[b0 & 15]),
                          mul_ftz(__bfloat162float(x[4 * i + 1]), kE2M1[b0 >> 4]));
        t = add_ftz(t, mul_ftz(__bfloat162float(x[4 * i + 2]), kE2M1[b1 & 15]));
        t = add_ftz(t, mul_ftz(__bfloat162float(x[4 * i + 3]), kE2M1[b1 >> 4]));
        acc = add_ftz(acc, t);
    }
    return acc;
}

// `qdot` for E4M3 weights.
template <int kValues>
__device__ float dot_e4m3(const __nv_bfloat16* x, const uint8_t* w) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kValues; ++i) {
        acc = add_ftz(acc, mul_ftz(__bfloat162float(x[i]), e4m3(w[i])));
    }
    return acc;
}

// One output value per warp. `blockIdx.x` is `row * 8 + slot`; `blockIdx.y`
// picks a tile of `kWarps` output rows.
template <bool kFp4, int kValues>
__device__ void expert_matvec(const __nv_bfloat16* x, const int32_t* indices, const uint8_t* weight,
                              const uint8_t* scales, const float* global_scale, __nv_bfloat16* out,
                              int n, int k, int inputs_per_row, int experts) {
    constexpr int kGroup = kFp4 ? 16 : 32;
    static_assert(kGroup % kValues == 0, "a chunk must not straddle two scale groups");
    const int lane = threadIdx.x % 32;
    const int col = blockIdx.y * kWarps + threadIdx.x / 32;
    if (col >= n) {
        return;
    }
    const unsigned slot_row = blockIdx.x;
    __nv_bfloat16* y = out + size_t(slot_row) * n + col;
    const int expert = indices[slot_row];
    if (expert < 0 || expert >= experts) {
        if (lane == 0) {
            *y = __float2bfloat16(__int_as_float(0x7fc00000));
        }
        return;
    }
    const unsigned row = slot_row / kSlots;
    const unsigned input = row * inputs_per_row + (inputs_per_row == 1 ? 0 : slot_row % kSlots);
    const __nv_bfloat16* xr = x + size_t(input) * k;
    const size_t weight_row = size_t(expert) * n + col;
    const uint8_t* wr = weight + weight_row * (kFp4 ? k / 2 : k);
    const uint8_t* sr = scales + weight_row * (k / kGroup);

    float result = 0.f;
    for (int base = lane * kValues; base < k; base += 32 * kValues) {
        float acc;
        float scale;
        if constexpr (kFp4) {
            acc = dot_e2m1<kValues>(xr + base, wr + base / 2);
            scale = e4m3(sr[base / kGroup]);
        } else {
            acc = dot_e4m3<kValues>(xr + base, wr + base);
            scale = e8m0(sr[base / kGroup]);
        }
        result = add_ftz(result, mul_ftz(scale, acc));
    }
    result = warp_sum(result);
    if constexpr (kFp4) {
        result = mul_ftz(result, __fdiv_rn(global_scale[expert], kNvfp4ScaleEnc));
    }
    if (lane == 0) {
        *y = __float2bfloat16(result);
    }
}

}  // namespace

// Arguments, shared by all four kernels:
// - `x`: BF16 `[rows, inputs_per_row, k]`, where `inputs_per_row` is 1 (one
//   input for all 8 slots) or 8 (one per slot);
// - `indices`: `[rows, 8]` expert ids;
// - `weight`, `scales`: raw checkpoint bytes, `[experts, n, k / 2]` E2M1 and
//   `[experts, n, k / 16]` E4M3 for NVFP4, `[experts, n, k]` E4M3 and
//   `[experts, n, k / 32]` E8M0 for MXFP8;
// - `global_scale`: F32 `[experts]` for NVFP4, unused by MXFP8;
// - `out`: BF16 `[rows, 8, n]`. An expert id outside `[0, experts)` gives NaN.
#define EXPERT_MATVEC(name, fp4, values)                                                           \
    extern "C" __global__ void __launch_bounds__(kWarps * 32)                                      \
        name(const __nv_bfloat16* x, const int32_t* indices, const uint8_t* weight,                \
             const uint8_t* scales, const float* global_scale, __nv_bfloat16* out, int n, int k,   \
             int inputs_per_row, int experts) {                                                    \
        expert_matvec<fp4, values>(x, indices, weight, scales, global_scale, out, n, k,            \
                                   inputs_per_row, experts);                                       \
    }

EXPERT_MATVEC(expert_matvec_nvfp4_8, true, 8)
EXPERT_MATVEC(expert_matvec_nvfp4_16, true, 16)
EXPERT_MATVEC(expert_matvec_mxfp8_4, false, 4)
EXPERT_MATVEC(expert_matvec_mxfp8_8, false, 8)
