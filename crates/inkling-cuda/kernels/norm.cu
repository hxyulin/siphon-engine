// RMSNorm over rows of a [rows, HIDDEN] BF16 matrix, with a BF16 weight.
//
// Rounding matches the MLX and Hugging Face reference: the normalized value is
// rounded to BF16 before the weight multiply. One block handles one row and
// each thread owns 8 contiguous elements, so HIDDEN / 8 threads per block.

#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int kVec = 8;

template <int Hidden>
__device__ void rms_norm_row(const __nv_bfloat16* __restrict__ x,
                             const __nv_bfloat16* __restrict__ weight,
                             __nv_bfloat16* __restrict__ out, float eps) {
    static_assert(Hidden % (kVec * 32) == 0, "row must split into whole warps");
    constexpr int kThreads = Hidden / kVec;
    constexpr int kWarps = kThreads / 32;
    __shared__ float warp_sums[kWarps];

    const int row = blockIdx.x;
    const int col = threadIdx.x * kVec;
    const uint4 packed = *reinterpret_cast<const uint4*>(x + size_t(row) * Hidden + col);
    const auto* vals = reinterpret_cast<const __nv_bfloat16*>(&packed);

    float v[kVec];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        v[i] = __bfloat162float(vals[i]);
        sum += v[i] * v[i];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    if (threadIdx.x % 32 == 0) {
        warp_sums[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        total += warp_sums[w];
    }
    // Correctly rounded sqrt and divide, not the approximate rsqrtf.
    const float inv_rms = 1.f / sqrtf(total / Hidden + eps);

    const uint4 wpacked = *reinterpret_cast<const uint4*>(weight + col);
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(&wpacked);
    uint4 result;
    auto* r = reinterpret_cast<__nv_bfloat16*>(&result);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        const float normed = __bfloat162float(__float2bfloat16(v[i] * inv_rms));
        r[i] = __float2bfloat16(normed * __bfloat162float(w[i]));
    }
    *reinterpret_cast<uint4*>(out + size_t(row) * Hidden + col) = result;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(2304 / kVec)
rms_norm_bf16_2304(const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out, float eps) {
    rms_norm_row<2304>(x, weight, out, eps);
}
