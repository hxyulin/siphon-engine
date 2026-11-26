// Elementwise kernels: embedding gather, softcapped SwiGLU, attention output gate,
// log-scaling of queries, residual adds, MoE combine and the final logit softcap.
//
// Each kernel repeats MLX's dtype steps: it widens BF16 to F32, does the F32 math
// one rounded operation at a time, and rounds to BF16 where MLX casts. Two details of
// MLX on Apple GPUs decide the last bit:
// - Metal flushes F32 subnormals to zero, both as inputs and as results of add,
//   multiply and divide, and MLX's F32-to-BF16 cast flushes them too. A sigmoid of a
//   gate below -87.5 is 0 in MLX, not a subnormal.
// - Each MLX op rounds its result, so a multiply followed by an add must not become
//   one FMA, which nvcc does by default.
// The fadd, fsub, fmul and fdiv helpers below use the PTX .rn.ftz forms for both reasons.
//
// Metal's precise::exp, tanh and log are not correctly rounded and no CUDA function
// reproduces them. Each one below is the candidate that agreed with MLX on the most
// inputs, measured over every finite BF16 value (and every position below 2^21 for log).

#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

#include "mlx_ops.cuh"
#include "pdl.cuh"

namespace {

constexpr int kHidden = 2304;
constexpr int kVec = 8;
constexpr int kTopK = 8;

__device__ __forceinline__ size_t grid_index() {
    return size_t(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ size_t grid_stride() {
    return size_t(gridDim.x) * blockDim.x;
}

// tanh(x / cap) * cap. Using CUDA's tanhf (softcap_f32) avoids the 1:64 FP64 path on
// sm_120 while staying within 2 ULPs of Metal across every finite BF16 value (98.0%
// exact match against 98.4% for double tanh).
__device__ __forceinline__ float softcap(float x, float cap) {
    return softcap_f32(x, cap);
}

}  // namespace

// out[t] = table[ids[t]] for [vocab, 2304] BF16 rows. One block per token, each
// thread copies 8 elements. IDs outside [0, vocab) give a zero row rather than
// reading past the table.
extern "C" __global__ void __launch_bounds__(kHidden / kVec)
embedding_bf16_2304(const int32_t* ids, const __nv_bfloat16* table, __nv_bfloat16* out,
                    int32_t vocab) {
    const int32_t id = ids[blockIdx.x];
    const int col = threadIdx.x * kVec;
    uint4 row = make_uint4(0, 0, 0, 0);
    if (id >= 0 && id < vocab) {
        row = *reinterpret_cast<const uint4*>(table + size_t(id) * kHidden + col);
    }
    *reinterpret_cast<uint4*>(out + size_t(blockIdx.x) * kHidden + col) = row;
}

// Softcapped SwiGLU on the interleaved gate/up projection: gate_up[2i] is the gate
// and gate_up[2i + 1] the up value of output i, so rows need no special handling.
extern "C" __global__ void softcapped_swiglu_bf16(const __nv_bfloat162* gate_up,
                                                  __nv_bfloat16* out, size_t n,
                                                  float gate_softcap, float up_softcap) {
    pdl_trigger();
    pdl_wait();
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        const float2 gu = __bfloat1622float2(gate_up[i]);
        out[i] = softcapped_swiglu(gu.x, gu.y, gate_softcap, up_softcap);
    }
}

// out = bf16(f32(attn) * sigmoid(f32(gate))), the end of `Attention._attend`, for n
// elements, a multiple of 8. Each thread takes 8 consecutive elements per step.
extern "C" __global__ void attention_gate_bf16(const __nv_bfloat16* attn,
                                               const __nv_bfloat16* gate,
                                               __nv_bfloat16* out, size_t n) {
    pdl_trigger();
    pdl_wait();
    for (size_t v = grid_index(); v < n / kVec; v += grid_stride()) {
        const uint4 a = reinterpret_cast<const uint4*>(attn)[v];
        const uint4 g = reinterpret_cast<const uint4*>(gate)[v];
        const auto* av = reinterpret_cast<const __nv_bfloat16*>(&a);
        const auto* gv = reinterpret_cast<const __nv_bfloat16*>(&g);
        uint4 o;
        auto* ov = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            ov[i] = to_bf16(fmul(__bfloat162float(av[i]), mlx_sigmoid(__bfloat162float(gv[i]))));
        }
        reinterpret_cast<uint4*>(out)[v] = o;
    }
}

// In-place x = bf16(f32(x) * tau) for rows at positions *start, *start + 1, ..., with
// tau = 1 + alpha * log(max((position + 1) / n_floor, 1)) as `FullAttention` computes
// it in F32. One block per row. __logf (hardware log2 times ln 2) agrees with Metal's
// log for 96.5% of positions below 2^21, logf for 86.7%.
extern "C" __global__ void log_scale_bf16(__nv_bfloat16* x, uint32_t row_width,
                                          const uint32_t* start, float alpha, float n_floor) {
    const float position = float(*start + blockIdx.x);
    const float ratio = fmaxf(fdiv(fadd(position, 1.f), n_floor), 1.f);
    const float tau = fadd(1.f, fmul(alpha, __logf(ratio)));
    __nv_bfloat16* row = x + size_t(blockIdx.x) * row_width;
    for (uint32_t c = threadIdx.x; c < row_width; c += blockDim.x) {
        row[c] = to_bf16(fmul(__bfloat162float(row[c]), tau));
    }
}

// out = f32(a) + f32(b): the attention residual add in `DecoderLayer`.
extern "C" __global__ void add_bf16_bf16_f32(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                             float* out, size_t n) {
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        out[i] = fadd(__bfloat162float(a[i]), __bfloat162float(b[i]));
    }
}

// out = bf16(residual + f32(x)): the MLP residual add that ends `DecoderLayer`.
extern "C" __global__ void add_f32_bf16_bf16(const float* residual, const __nv_bfloat16* x,
                                             __nv_bfloat16* out, size_t n) {
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        out[i] = to_bf16(fadd(residual[i], __bfloat162float(x[i])));
    }
}

// out[l, c] = bf16(sum_k f32(experts[l, k, c]) * weights[l, k]). MLX materializes the
// F32 products, then its column reduction adds them to a zero total one at a time in
// slot order k = 0, 1, ..., 7. Matching that order matters: F32 addition is not
// associative.
extern "C" __global__ void moe_combine_bf16(const __nv_bfloat16* experts, const float* weights,
                                            __nv_bfloat16* out, size_t n) {
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        const size_t row = i / kHidden;
        const size_t col = i % kHidden;
        const __nv_bfloat16* e = experts + row * kTopK * kHidden + col;
        const float* w = weights + row * kTopK;
        float total = 0.f;
#pragma unroll
        for (int k = 0; k < kTopK; ++k) {
            const float wk = w[k];
            if (wk != 0.f) {
                total = fadd(total, fmul(__bfloat162float(e[k * kHidden]), wk));
            }
        }
        out[i] = to_bf16(total);
    }
}

// out[l, c] = tanh(f32(logits[l, c]) / cap) * cap for the first out_width of in_width
// columns, dropping the padded vocabulary entries as `Model.__call__` does.
// Vectorized with 16-byte uint4 (8 BF16) loads and 2x float4 (or 4x float2 when out_width
// leaves odd rows 8-byte aligned, as for out_width = 200058) stores, with PDL.
extern "C" __global__ void logit_softcap_bf16_f32(const __nv_bfloat16* logits, float* out,
                                                  size_t n, uint32_t in_width,
                                                  uint32_t out_width, float cap) {
    pdl_trigger();
    pdl_wait();
    const size_t rows = n / out_width;
    const size_t chunks_per_row = (size_t(out_width) + kVec - 1) / kVec;
    const size_t total = rows * chunks_per_row;
    for (size_t idx = grid_index(); idx < total; idx += grid_stride()) {
        const size_t row = idx / chunks_per_row;
        const uint32_t col = static_cast<uint32_t>((idx % chunks_per_row) * kVec);
        const __nv_bfloat16* src = logits + row * in_width + col;
        float* dst = out + row * out_width + col;
        if (col + kVec <= out_width && (reinterpret_cast<uintptr_t>(src) & 15u) == 0) {
            const uint4 packed = *reinterpret_cast<const uint4*>(src);
            const auto* sv = reinterpret_cast<const __nv_bfloat16*>(&packed);
            float y[kVec];
#pragma unroll
            for (int i = 0; i < kVec; ++i) {
                y[i] = softcap(__bfloat162float(sv[i]), cap);
            }
            const uintptr_t daddr = reinterpret_cast<uintptr_t>(dst);
            if ((daddr & 15u) == 0) {
                reinterpret_cast<float4*>(dst)[0] = make_float4(y[0], y[1], y[2], y[3]);
                reinterpret_cast<float4*>(dst)[1] = make_float4(y[4], y[5], y[6], y[7]);
            } else if ((daddr & 7u) == 0) {
                reinterpret_cast<float2*>(dst)[0] = make_float2(y[0], y[1]);
                reinterpret_cast<float2*>(dst)[1] = make_float2(y[2], y[3]);
                reinterpret_cast<float2*>(dst)[2] = make_float2(y[4], y[5]);
                reinterpret_cast<float2*>(dst)[3] = make_float2(y[6], y[7]);
            } else {
#pragma unroll
                for (int i = 0; i < kVec; ++i) {
                    dst[i] = y[i];
                }
            }
        } else {
            for (uint32_t i = 0; i < kVec && col + i < out_width; ++i) {
                dst[i] = softcap(__bfloat162float(src[i]), cap);
            }
        }
    }
}
