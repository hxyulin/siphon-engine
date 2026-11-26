// F32 helpers that repeat MLX's dtype steps on Apple GPUs, shared by elementwise.cu
// and experts.cu. elementwise.cu explains the flushing and rounding rules they follow.

#pragma once

#include <cuda_bf16.h>
#include <cfloat>

namespace {

#if SIPHON_FAST_MATH
__device__ __forceinline__ float fadd(float a, float b) { return a + b; }
__device__ __forceinline__ float fsub(float a, float b) { return a - b; }
__device__ __forceinline__ float fmul(float a, float b) { return a * b; }
__device__ __forceinline__ float fdiv(float a, float b) { return a / b; }

__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float mlx_sigmoid(float x) {
    return 1.f / (1.f + __expf(-x));
}
#else
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

__device__ __forceinline__ float fdiv(float a, float b) {
    float r;
    asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// Round to nearest even, flushing subnormal inputs as MLX's cast does.
__device__ __forceinline__ __nv_bfloat16 to_bf16(float x) {
    return __float2bfloat16(fabsf(x) < FLT_MIN ? copysignf(0.f, x) : x);
}

// MLX's Metal sigmoid (`Sigmoid` in mlx/backend/metal/kernels/unary_ops.h):
// y = 1 / (1 + exp(|x|)), and 1 - y for positive x. expf agrees with Metal's exp on
// 98.0% of BF16 inputs, a correctly rounded exp on 97.4%.
__device__ __forceinline__ float mlx_sigmoid(float x) {
    const float y = fdiv(1.f, fadd(1.f, expf(fabsf(x))));
    return x < 0.f ? y : fsub(1.f, y);
}
#endif

// The SwiGLU softcap with CUDA's tanhf (at most 2 ulps from the true value), which
// avoids the FP64 tanh of elementwise.cu's softcap. SwiGLU runs on 9216 values per
// layer visit; the FP64 version took 2.7 us per call on an RTX 5090.
__device__ __forceinline__ float softcap_f32(float x, float cap) {
    return fmul(tanhf(fdiv(x, cap)), cap);
}

// MLX's `_softcapped_swiglu` on one gate and up value, already rounded to BF16. It
// multiplies left to right: (gate_capped * sigmoid) * up_capped.
__device__ __forceinline__ __nv_bfloat16 softcapped_swiglu(float gate, float up,
                                                           float gate_softcap, float up_softcap) {
    const float g = fmul(softcap_f32(gate, gate_softcap), mlx_sigmoid(gate));
    return to_bf16(fmul(g, softcap_f32(up, up_softcap)));
}

}  // namespace
