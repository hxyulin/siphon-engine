// MXFP8 quantization of BF16 rows, shared by expert_matmul.cu (the expert inputs) and
// linear_fp8.cu (the dense prefill products): E4M3 values in blocks of 32 with a UE8M0
// scale, the block's amax / 448 rounded up to a power of two, as vLLM's Inkling
// `mxfp8.py` does. Values round to nearest even with saturation.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

// Loads 8 BF16 values as F32.
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

// MXFP8: one warp per row. With `kPaired`, each block of 32 values is stored
// in the order `expert_gemm_q8` loads it: 4-byte word w of the block holds
// values 8t + 4h .. 8t + 4h + 3 for (h, t) = (w / 4, w % 4), so that the two
// words a lane loads for one MMA k step hold 8 consecutive values.
template <bool kPaired>
__device__ void quantize_row_mxfp8(const __nv_bfloat16* __restrict__ row, int k,
                                   uint8_t* __restrict__ values, uint8_t* __restrict__ scales) {
    for (int b = threadIdx.x % 32; b < k / 32; b += 32) {
        float x[32];
#pragma unroll
        for (int i = 0; i < 32; i += 8) {
            load8(row + b * 32 + i, x + i);
        }
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            amax = fmaxf(amax, fabsf(x[i]));
        }
        const float raw = __fmul_rn(amax, 1.f / 448.f);
        const float min_scale = 0x1p-127f;
        int exponent = ((__float_as_int(raw) + 0x007fffff) & 0x7f800000) >> 23;
        exponent = raw <= min_scale ? 0 : min(exponent, 254);
        exponent = raw != raw ? 255 : exponent;
        const float scale = fmaxf(__int_as_float(exponent << 23), min_scale);
        float inverse = exponent == 0 ? 0x1p127f : __frcp_rn(scale);
        inverse = exponent == 255 ? 0.f : inverse;
        scales[b] = uint8_t(exponent);
        uint32_t words[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            uint32_t word = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float v = __fmul_rn(x[4 * w + i], inverse);
                v = v != v ? v : fminf(fmaxf(v, -448.f), 448.f);
                const __nv_fp8_e4m3 q(v);
                word |= uint32_t(q.__x) << (8 * i);
            }
            words[w] = word;
        }
        uint4* dst = reinterpret_cast<uint4*>(values + b * 32);
        if constexpr (kPaired) {
            dst[0] = make_uint4(words[0], words[2], words[4], words[6]);
            dst[1] = make_uint4(words[1], words[3], words[5], words[7]);
        } else {
            dst[0] = make_uint4(words[0], words[1], words[2], words[3]);
            dst[1] = make_uint4(words[4], words[5], words[6], words[7]);
        }
    }
}

}  // namespace
