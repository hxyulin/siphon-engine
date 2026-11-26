// Fused MXFP8 and NVFP4 activation quantization epilogues for RMSNorm kernels whose
// threads hold 4 consecutive BF16-rounded values per run (`norm.cu`, `misc.cu`,
// `routing.cu`).
//
// Each helper reproduces `mxfp8.cuh::quantize_row_mxfp8` (E4M3 values in blocks of 32
// with UE8M0 block scale, natural or paired word order for `expert_gemm_q8_nvfp4`) and
// `expert_matmul.cu::quantize_row_nvfp4` (E2M1 pairs in blocks of 16 with E4M3 block
// scales under per-row outer scale amax / 1536 and fouroversix_mae selection) bit for
// bit, directly from registers without a BF16 memory round-trip.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#include "arch.cuh"

namespace quant_epilogue {

// Quantization mode shared with `siphon_cuda::ops::QuantFormat`.
enum QuantMode : int {
    kQuantNone = 0,
    kQuantMxfp8 = 1,
    kQuantMxfp8Paired = 2,
    kQuantNvfp4 = 3,
};

constexpr float kFp32Max = 3.4028234663852886e38f;

// Warp-wide max over 32 lanes (all lanes receive the result).
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

// Quantizes 4 consecutive BF16-rounded values `x` held by lane `sub8 = lane % 8` of an
// 8-lane sub-warp (32 elements = one MXFP8 block `block32`) into `q_values_row` and
// `q_scales_row`. Matches `quantize_row_mxfp8<paired>` in `mxfp8.cuh` bit for bit.
__device__ __forceinline__ void quantize_vec4_mxfp8(const float (&x)[4], int block32, int sub8,
                                                    bool paired, uint8_t* q_values_row,
                                                    uint8_t* q_scales_row) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        amax = fmaxf(amax, fabsf(x[i]));
    }
#pragma unroll
    for (int offset = 1; offset < 8; offset *= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    const float raw = __fmul_rn(amax, 1.f / 448.f);
    const float min_scale = 0x1p-127f;
    int exponent = ((__float_as_int(raw) + 0x007fffff) & 0x7f800000) >> 23;
    exponent = raw <= min_scale ? 0 : min(exponent, 254);
    exponent = raw != raw ? 255 : exponent;
    const float scale = fmaxf(__int_as_float(exponent << 23), min_scale);
    float inverse = exponent == 0 ? 0x1p127f : __frcp_rn(scale);
    inverse = exponent == 255 ? 0.f : inverse;
    if (sub8 == 0) {
        q_scales_row[block32] = uint8_t(exponent);
    }
    uint32_t word = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float v = __fmul_rn(x[i], inverse);
        v = v != v ? v : fminf(fmaxf(v, -448.f), 448.f);
        const __nv_fp8_e4m3 q(v);
        word |= uint32_t(q.__x) << (8 * i);
    }
    // Paired word order of `expert_gemm_q8_nvfp4`: even sub8 -> words 0..3, odd -> 4..7.
    const int dst_w = paired ? ((sub8 & 1) * 4 + (sub8 >> 1)) : sub8;
    reinterpret_cast<uint32_t*>(q_values_row + size_t(block32) * 32)[dst_w] = word;
}

__device__ __forceinline__ float e4m3_round(float x) {
    const __nv_fp8_e4m3 q(x);
    return float(q);
}

// Two values rounded to E2M1 (`cvt.rn.satfinite`): byte (`lo` in low nibble) and both
// reconstructed values. Matches `e2m1x2` in `expert_matmul.cu`.
__device__ __forceinline__ uint32_t e2m1x2(float lo, float hi, float& lo_value, float& hi_value) {
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

// One NVFP4 candidate across a 4-lane sub-warp (16 elements): each lane packs its 4
// elements into `packed16` (2 bytes) and returns the exact pairwise binary-tree MAE sum
// of `fp4_candidate` in `expert_matmul.cu`.
__device__ __forceinline__ float fp4_candidate4(const float (&x)[4], float scale, float inverse,
                                                float outer, uint16_t& packed16) {
    float err[4];
    uint32_t b0;
    uint32_t b1;
    {
        float lo;
        float hi;
        b0 = e2m1x2(__fmul_rn(x[0], inverse), __fmul_rn(x[1], inverse), lo, hi);
        err[0] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(lo, scale), outer), x[0]));
        err[1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(hi, scale), outer), x[1]));
    }
    {
        float lo;
        float hi;
        b1 = e2m1x2(__fmul_rn(x[2], inverse), __fmul_rn(x[3], inverse), lo, hi);
        err[2] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(lo, scale), outer), x[2]));
        err[3] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(hi, scale), outer), x[3]));
    }
    packed16 = uint16_t(b0 | (b1 << 8));
    // Exact pairwise binary-tree sum over 16 elements:
    // (0+1)+(2+3) per lane, then lane(0)+lane(1) / lane(2)+lane(3), then total.
    float e = __fadd_rn(__fadd_rn(err[0], err[1]), __fadd_rn(err[2], err[3]));
    e = __fadd_rn(e, __shfl_xor_sync(0xffffffffu, e, 1));
    e = __fadd_rn(e, __shfl_xor_sync(0xffffffffu, e, 2));
    return e;
}

// Quantizes 4 consecutive BF16-rounded values `x` held by lane `sub4 = lane % 4` of a
// 4-lane sub-warp (16 elements = one NVFP4 block `block16`) under per-row `row_amax` into
// `q_values_row` and `q_scales_row`. Matches `quantize_row_nvfp4` in `expert_matmul.cu`
// bit for bit. Caller writes `*q_outer_row = __fmul_rn(row_amax, 1.f / 1536.f)`.
__device__ __forceinline__ void quantize_vec4_nvfp4(const float (&x)[4], float row_amax,
                                                    int block16, int sub4, uint8_t* q_values_row,
                                                    uint8_t* q_scales_row) {
    const float encoder = fminf(__fdiv_rn(1536.f, row_amax), kFp32Max);
    const float decoder = __fdiv_rn(1.f, encoder);
    const float outer_scale = __fmul_rn(row_amax, 1.f / 1536.f);
    const float encoder6 = __fmul_rn(encoder, 1.f / 6.f);
    const float encoder4 = __fmul_rn(encoder, 0.25f);

    float block_max = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        block_max = fmaxf(block_max, fabsf(x[i]));
    }
    block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffffu, block_max, 1));
    block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffffu, block_max, 2));

    const float scale6 = e4m3_round(fminf(__fmul_rn(block_max, encoder6), 448.f));
    const float scale4 = e4m3_round(fminf(__fmul_rn(block_max, encoder4), 448.f));
    const float inverse6 = fminf(__fdiv_rn(1.f, __fmul_rn(scale6, decoder)), kFp32Max);
    const float inverse4 = fminf(__fdiv_rn(1.f, __fmul_rn(scale4, decoder)), kFp32Max);
    uint16_t packed6;
    uint16_t packed4;
    const float error6 = fp4_candidate4(x, scale6, inverse6, outer_scale, packed6);
    const float error4 = fp4_candidate4(x, scale4, inverse4, outer_scale, packed4);
    const bool choose4 = error4 < error6;
    const uint16_t packed = choose4 ? packed4 : packed6;
    if (sub4 == 0) {
        const __nv_fp8_e4m3 scale(choose4 ? scale4 : scale6);
        q_scales_row[block16] = scale.__x;
    }
    reinterpret_cast<uint16_t*>(q_values_row + size_t(block16) * 8)[sub4] = packed;
}

}  // namespace quant_epilogue
