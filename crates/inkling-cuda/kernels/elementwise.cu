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

// tanh(x / cap) * cap. A correctly rounded tanh (through double) agrees with Metal's
// on 98.4% of BF16 inputs to the logit softcap, tanhf on 98.0%. In SwiGLU outputs that
// is 11 against 18 differing BF16 values per million, so SwiGLU uses tanhf (softcap_f32).
__device__ __forceinline__ float softcap(float x, float cap) {
    return fmul(float(tanh(double(fdiv(x, cap)))), cap);
}

// The SwiGLU softcap with CUDA's tanhf (at most 2 ulps from the true value), which avoids
// the FP64 tanh above. SwiGLU runs on 9216 values per layer visit; the FP64 version took
// 2.7 us per call on an RTX 5090.
__device__ __forceinline__ float softcap_f32(float x, float cap) {
    return fmul(tanhf(fdiv(x, cap)), cap);
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
// MLX (`_softcapped_swiglu`) multiplies left to right: (gate_capped * sigmoid) * up_capped.
extern "C" __global__ void softcapped_swiglu_bf16(const __nv_bfloat162* gate_up,
                                                  __nv_bfloat16* out, size_t n,
                                                  float gate_softcap, float up_softcap) {
    pdl_trigger();
    pdl_wait();
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        const float2 gu = __bfloat1622float2(gate_up[i]);
        const float gate = fmul(softcap_f32(gu.x, gate_softcap), mlx_sigmoid(gu.x));
        out[i] = to_bf16(fmul(gate, softcap_f32(gu.y, up_softcap)));
    }
}

// out = bf16(f32(attn) * sigmoid(f32(gate))), the end of `Attention._attend`.
extern "C" __global__ void attention_gate_bf16(const __nv_bfloat16* attn,
                                               const __nv_bfloat16* gate,
                                               __nv_bfloat16* out, size_t n) {
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        const float g = mlx_sigmoid(__bfloat162float(gate[i]));
        out[i] = to_bf16(fmul(__bfloat162float(attn[i]), g));
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
            total = fadd(total, fmul(__bfloat162float(e[k * kHidden]), w[k]));
        }
        out[i] = to_bf16(total);
    }
}

// out[l, c] = tanh(f32(logits[l, c]) / cap) * cap for the first out_width of in_width
// columns, dropping the padded vocabulary entries as `Model.__call__` does.
extern "C" __global__ void logit_softcap_bf16_f32(const __nv_bfloat16* logits, float* out,
                                                  size_t n, uint32_t in_width,
                                                  uint32_t out_width, float cap) {
    for (size_t i = grid_index(); i < n; i += grid_stride()) {
        const size_t row = i / out_width;
        const size_t col = i % out_width;
        out[i] = softcap(__bfloat162float(logits[row * in_width + col]), cap);
    }
}
