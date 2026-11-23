// Kernels of the DSpark draft (`mtp.safetensors`), following the numerics of vLLM's
// draft (`models/inkling2/nvidia/dspark.py` and `ops/dspark.py`) and of
// `reference/dspark.py`:
// - RMSNorms sum squares in F32 and round once to BF16. The fused add-norm stores
//   the residual sum rounded to BF16 but normalizes the unrounded F32 sum.
// - The per-head Q and K norms round to BF16 before RoPE. RoPE is NeoX style (first
//   half against second half) in F32 with theta 1e6, and rounds once.
// - `silu(gate) * up` runs in F32 and rounds once.
// - The Markov gate rounds every operation to BF16, and the logits are
//   bf16(30 tanh(bf16(base + bias) / 30)) with bias = bf16(fold_bias + fold_weight z).
// The products themselves run through the GEMM and GEMV kernels of linear.cu.
//
// Unlike the target's kernels these do not copy MLX's summation order: the draft is
// checked against the reference within its BF16 noise, not bit for bit.

#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

namespace {

constexpr int kHidden = 2304;
constexpr int kHeadDim = 128;
constexpr int kHalf = kHeadDim / 2;
constexpr int kKvHeads = 8;
constexpr int kGroup = 3;  // query heads per KV head
constexpr int kBlock = 8;  // rows of a draft block
constexpr int kTaps = 4;
constexpr int kRank = 256;
constexpr int kRowThreads = 256;
constexpr int kRowVec = kHidden / kRowThreads;  // 9 elements per thread, strided
static_assert(kHidden % kRowThreads == 0, "rows split evenly over the threads");

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// Sum over a block of `kRowThreads` threads, returned to every thread.
__device__ float block_sum(float v) {
    __shared__ float partial[kRowThreads / 32];
    v = warp_sum(v);
    if (threadIdx.x % 32 == 0) {
        partial[threadIdx.x / 32] = v;
    }
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kRowThreads / 32; ++w) {
        total += partial[w];
    }
    __syncthreads();
    return total;
}

__device__ __forceinline__ float inv_rms(float sum_squares, int n, float eps) {
    return 1.f / sqrtf(sum_squares / static_cast<float>(n) + eps);
}

// Writes bf16(v * inv_rms(v) * weight) for one 2304-wide row held as 9 values per thread.
__device__ void store_normed_row(const float (&v)[kRowVec], const __nv_bfloat16* weight,
                                 __nv_bfloat16* out, float eps) {
    float squares = 0.f;
#pragma unroll
    for (int i = 0; i < kRowVec; ++i) {
        squares = fmaf(v[i], v[i], squares);
    }
    const float inv = inv_rms(block_sum(squares), kHidden, eps);
#pragma unroll
    for (int i = 0; i < kRowVec; ++i) {
        const int c = threadIdx.x + i * kRowThreads;
        out[c] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(v[i], inv), bf(weight[c])));
    }
}

// The argmax rule of vLLM's `greedy_argmax`: NaN counts as -inf, ties go to the first index.
__device__ __forceinline__ bool better(float a, int ia, float b, int ib) {
    a = isnan(a) ? -INFINITY : a;
    b = isnan(b) ? -INFINITY : b;
    return a > b || (a == b && ia < ib);
}

}  // namespace

// out[r] = rms_norm(sum_j alpha[j] * taps[r, j]; weight) for taps BF16 [rows, 4, 2304]:
// the draft's softmax mix of the four target features. One block per row.
extern "C" __global__ void __launch_bounds__(kRowThreads)
draft_mix_rms_norm(const __nv_bfloat16* taps, const float* alpha, const __nv_bfloat16* weight,
                   __nv_bfloat16* out, float eps) {
    const size_t row = blockIdx.x;
    const __nv_bfloat16* x = taps + row * kTaps * kHidden;
    float a[kTaps];
#pragma unroll
    for (int j = 0; j < kTaps; ++j) {
        a[j] = alpha[j];
    }
    float v[kRowVec];
#pragma unroll
    for (int i = 0; i < kRowVec; ++i) {
        const int c = threadIdx.x + i * kRowThreads;
        float m = 0.f;
#pragma unroll
        for (int j = 0; j < kTaps; ++j) {
            m = fmaf(bf(x[j * kHidden + c]), a[j], m);
        }
        v[i] = m;
    }
    store_normed_row(v, weight, out + row * kHidden, eps);
}

// For each row of [rows, 2304] BF16 matrices: sum = residual + x in F32,
// residual = bf16(sum), out = rms_norm(sum; weight). With x null, the residual is only
// read and normalized.
extern "C" __global__ void __launch_bounds__(kRowThreads)
draft_add_rms_norm(const __nv_bfloat16* x, __nv_bfloat16* residual, const __nv_bfloat16* weight,
                   __nv_bfloat16* out, float eps) {
    const size_t base = size_t(blockIdx.x) * kHidden;
    float v[kRowVec];
#pragma unroll
    for (int i = 0; i < kRowVec; ++i) {
        const size_t c = base + threadIdx.x + i * kRowThreads;
        v[i] = bf(residual[c]);
        if (x != nullptr) {
            v[i] = __fadd_rn(v[i], bf(x[c]));
            residual[c] = __float2bfloat16_rn(v[i]);
        }
    }
    store_normed_row(v, weight, out + base, eps);
}

// The block's input rows, BF16 [8, 2304]: row 0 is rms_norm(embed[anchor]; embed_norm),
// the anchor's normalized target embedding, and rows 1 to 7 are mask_embed. One block
// per row. An anchor outside [0, vocab) gives a zero embedding.
extern "C" __global__ void __launch_bounds__(kRowThreads)
draft_block_input(const int32_t* anchor, const __nv_bfloat16* embed, int32_t vocab,
                  const __nv_bfloat16* embed_norm, const __nv_bfloat16* mask_embed,
                  __nv_bfloat16* out, float eps) {
    __nv_bfloat16* row = out + size_t(blockIdx.x) * kHidden;
    if (blockIdx.x > 0) {
        for (int c = threadIdx.x; c < kHidden; c += kRowThreads) {
            row[c] = mask_embed[c];
        }
        return;
    }
    const int32_t id = *anchor;
    const bool valid = id >= 0 && id < vocab;
    float v[kRowVec];
#pragma unroll
    for (int i = 0; i < kRowVec; ++i) {
        const int c = threadIdx.x + i * kRowThreads;
        v[i] = valid ? bf(embed[size_t(id) * kHidden + c]) : 0.f;
    }
    store_normed_row(v, embed_norm, row, eps);
}

// In place on the leading `heads` 128-wide heads of each row of x (row stride
// `row_stride` elements): per-head RMSNorm with `weight`, rounded to BF16, then NeoX
// RoPE at position `start + row`. inv_freq is F32 [64], theta^(-2i/128). Grid (rows,
// heads), 64 threads: thread i rotates elements i and i + 64.
extern "C" __global__ void __launch_bounds__(kHalf)
draft_head_norm_rope(__nv_bfloat16* x, int32_t row_stride, const __nv_bfloat16* weight,
                     const float* inv_freq, int32_t start, float eps) {
    __shared__ float partial[kHalf / 32];
    __nv_bfloat16* head = x + size_t(blockIdx.x) * row_stride + blockIdx.y * kHeadDim;
    const int i = threadIdx.x;
    const float x1 = bf(head[i]);
    const float x2 = bf(head[i + kHalf]);
    const float s = warp_sum(fmaf(x1, x1, x2 * x2));
    if (i % 32 == 0) {
        partial[i / 32] = s;
    }
    __syncthreads();
    const float inv = inv_rms(partial[0] + partial[1], kHeadDim, eps);
    const float n1 = round_bf16(__fmul_rn(__fmul_rn(x1, inv), bf(weight[i])));
    const float n2 = round_bf16(__fmul_rn(__fmul_rn(x2, inv), bf(weight[i + kHalf])));
    const float position = static_cast<float>(start + static_cast<int32_t>(blockIdx.x));
    float sin_f, cos_f;
    sincosf(__fmul_rn(position, inv_freq[i]), &sin_f, &cos_f);
    head[i] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(n1, cos_f), __fmul_rn(n2, sin_f)));
    head[i + kHalf] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(n2, cos_f), __fmul_rn(n1, sin_f)));
}

// Copies rows of keys and values (BF16 [rows, 1024] with row stride `row_stride`) into
// the caches (BF16 [slots, 1024]): row r goes to slot (start + r) mod slots. One block
// per row, 128 threads of 8 elements.
extern "C" __global__ void __launch_bounds__(128)
draft_store_kv(const __nv_bfloat16* k, const __nv_bfloat16* v, int32_t row_stride,
               __nv_bfloat16* cache_k, __nv_bfloat16* cache_v, int32_t start, int32_t slots) {
    constexpr int kWidth = kKvHeads * kHeadDim;
    const size_t slot = size_t((start + static_cast<int32_t>(blockIdx.x)) % slots);
    const size_t src = size_t(blockIdx.x) * row_stride + threadIdx.x * 8;
    const size_t dst = slot * kWidth + threadIdx.x * 8;
    *reinterpret_cast<uint4*>(cache_k + dst) = *reinterpret_cast<const uint4*>(k + src);
    *reinterpret_cast<uint4*>(cache_v + dst) = *reinterpret_cast<const uint4*>(v + src);
}

namespace {

constexpr int kAttnThreads = 256;
constexpr int kQueryRows = kBlock * kGroup;  // 24 query rows per KV head
constexpr int kKeyChunk = 32;

}  // namespace

// One split of the block attention: the 8 rows' queries (BF16 [8, 24 * 128], after
// norm and RoPE) of KV head blockIdx.x against keys at positions
// first + split * keys_per_split, up to first + keys. Position p sits in cache slot
// p mod slots (caches BF16 [slots, 8, 128]). Writes the unnormalized F32 output
// [24, 128] and the row max and sum [24, 2] of the split. Query row r is token r / 3,
// head 3 * blockIdx.x + r % 3. Scale 128^-0.5.
extern "C" __global__ void __launch_bounds__(kAttnThreads)
draft_attention(const __nv_bfloat16* q, const __nv_bfloat16* cache_k,
                const __nv_bfloat16* cache_v, int32_t slots, int32_t first, int32_t keys,
                int32_t keys_per_split, float* partial_out, float* partial_stats) {
    __shared__ float qs[kQueryRows][kHeadDim];
    __shared__ float ks[kKeyChunk][kHeadDim + 1];
    __shared__ float vs[kKeyChunk][kHeadDim];
    __shared__ float ps[kQueryRows][kKeyChunk];
    __shared__ float row_max[kQueryRows];
    __shared__ float row_sum[kQueryRows];
    __shared__ float row_scale[kQueryRows];

    const int kv = blockIdx.x;
    const int split = blockIdx.y;
    const int t = threadIdx.x;
    const float scale = rsqrtf(static_cast<float>(kHeadDim));
    for (int idx = t; idx < kQueryRows * kHeadDim; idx += kAttnThreads) {
        const int r = idx / kHeadDim, d = idx % kHeadDim;
        const int token = r / kGroup, head = kv * kGroup + r % kGroup;
        qs[r][d] = bf(q[size_t(token) * kGroup * kKvHeads * kHeadDim + head * kHeadDim + d]);
    }
    if (t < kQueryRows) {
        row_max[t] = -INFINITY;
        row_sum[t] = 0.f;
    }
    // Thread t accumulates dimension t % 128 of rows t / 128, t / 128 + 2, ...
    constexpr int kRowsPerThread = kQueryRows * kHeadDim / kAttnThreads;
    const int d = t % kHeadDim;
    const int r0 = t / kHeadDim;
    float acc[kRowsPerThread];
#pragma unroll
    for (int j = 0; j < kRowsPerThread; ++j) {
        acc[j] = 0.f;
    }

    const int begin = split * keys_per_split;
    const int end = min(keys, begin + keys_per_split);
    for (int c0 = begin; c0 < end; c0 += kKeyChunk) {
        const int n = min(kKeyChunk, end - c0);
        __syncthreads();
        for (int idx = t; idx < kKeyChunk * kHeadDim; idx += kAttnThreads) {
            const int key = idx / kHeadDim, dd = idx % kHeadDim;
            float kval = 0.f, vval = 0.f;
            if (key < n) {
                const size_t slot = size_t((first + c0 + key) % slots);
                const size_t off = (slot * kKvHeads + kv) * kHeadDim + dd;
                kval = bf(cache_k[off]);
                vval = bf(cache_v[off]);
            }
            ks[key][dd] = kval;
            vs[key][dd] = vval;
        }
        __syncthreads();
        for (int idx = t; idx < kQueryRows * kKeyChunk; idx += kAttnThreads) {
            const int r = idx / kKeyChunk, key = idx % kKeyChunk;
            float s = -INFINITY;
            if (key < n) {
                float dot = 0.f;
#pragma unroll 16
                for (int dd = 0; dd < kHeadDim; ++dd) {
                    dot = fmaf(qs[r][dd], ks[key][dd], dot);
                }
                s = dot * scale;
            }
            ps[r][key] = s;
        }
        __syncthreads();
        if (t < kQueryRows) {
            float m = row_max[t];
            for (int key = 0; key < n; ++key) {
                m = fmaxf(m, ps[t][key]);
            }
            const float correction = expf(row_max[t] - m);
            float sum = row_sum[t] * correction;
            for (int key = 0; key < kKeyChunk; ++key) {
                const float p = key < n ? expf(ps[t][key] - m) : 0.f;
                ps[t][key] = p;
                sum += p;
            }
            row_max[t] = m;
            row_sum[t] = sum;
            row_scale[t] = correction;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < kRowsPerThread; ++j) {
            const int r = r0 + j * (kAttnThreads / kHeadDim);
            float a = acc[j] * row_scale[r];
            for (int key = 0; key < n; ++key) {
                a = fmaf(ps[r][key], vs[key][d], a);
            }
            acc[j] = a;
        }
    }
    __syncthreads();
    const size_t base = (size_t(split) * kKvHeads + kv) * kQueryRows;
#pragma unroll
    for (int j = 0; j < kRowsPerThread; ++j) {
        const int r = r0 + j * (kAttnThreads / kHeadDim);
        partial_out[(base + r) * kHeadDim + d] = acc[j];
    }
    if (t < kQueryRows) {
        partial_stats[(base + t) * 2] = row_max[t];
        partial_stats[(base + t) * 2 + 1] = row_sum[t];
    }
}

// Combines the splits of draft_attention into out, BF16 [8, 24 * 128]. Grid (8 KV
// heads, 24 query rows), 128 threads.
extern "C" __global__ void __launch_bounds__(kHeadDim)
draft_attention_combine(const float* partial_out, const float* partial_stats, int32_t splits,
                        __nv_bfloat16* out) {
    const int kv = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) {
        m = fmaxf(m, partial_stats[((size_t(s) * kKvHeads + kv) * kQueryRows + r) * 2]);
    }
    float sum = 0.f, value = 0.f;
    for (int s = 0; s < splits; ++s) {
        const size_t row = (size_t(s) * kKvHeads + kv) * kQueryRows + r;
        const float w = expf(partial_stats[row * 2] - m);
        sum = fmaf(partial_stats[row * 2 + 1], w, sum);
        value = fmaf(partial_out[row * kHeadDim + d], w, value);
    }
    const int token = r / kGroup, head = kv * kGroup + r % kGroup;
    out[size_t(token) * kGroup * kKvHeads * kHeadDim + head * kHeadDim + d] =
        __float2bfloat16_rn(value / sum);
}

// out = bf16(silu(gate) * up) over n elements, in F32 with IEEE exp and division.
extern "C" __global__ void draft_silu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                          __nv_bfloat16* out, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x) {
        const float g = bf(gate[i]);
        const float silu = __fdiv_rn(g, __fadd_rn(1.f, expf(-g)));
        out[i] = __float2bfloat16_rn(__fmul_rn(silu, bf(up[i])));
    }
}

// The inputs of the Markov head's context product, F32 [7, 4608]: row i is
// [y[i + 1]; f], the draft output of mask row i + 1 and the head's mixed taps, widened.
extern "C" __global__ void draft_markov_context_input(const __nv_bfloat16* y,
                                                      const __nv_bfloat16* f, float* out) {
    const int row = blockIdx.x;
    for (int c = threadIdx.x; c < 2 * kHidden; c += blockDim.x) {
        const float v = c < kHidden ? bf(y[size_t(row + 1) * kHidden + c]) : bf(f[c - kHidden]);
        out[size_t(row) * 2 * kHidden + c] = v;
    }
}

// Starts the Markov loop: step_in, F32 [512], becomes [0; markov_w1[anchor]], the zero
// state and the anchor's token embedding. One block of 256 threads.
extern "C" __global__ void __launch_bounds__(kRank)
draft_markov_init(const int32_t* anchor, const __nv_bfloat16* w1, int32_t vocab,
                  float* step_in) {
    const int32_t id = *anchor;
    const bool valid = id >= 0 && id < vocab;
    step_in[threadIdx.x] = 0.f;
    step_in[kRank + threadIdx.x] = valid ? bf(w1[size_t(id) * kRank + threadIdx.x]) : 0.f;
}

// One Markov gate: r = (context + bias) + raw (all F32 [768]: the step's [y; f] product,
// the rnn_proj bias and the [state; w1] product). Reads the state from step_in[0:256],
// replaces it with the new state, and writes the latent tanh(out) to latent (F32
// [256]). Every operation rounds to BF16, as the reference's BF16 torch ops do.
extern "C" __global__ void __launch_bounds__(kRank)
draft_markov_gate(const float* raw, const float* context, const float* bias, float* step_in,
                  float* latent) {
    const int j = threadIdx.x;
    float r[3];
#pragma unroll
    for (int part = 0; part < 3; ++part) {
        const int c = part * kRank + j;
        r[part] = round_bf16(__fadd_rn(__fadd_rn(context[c], bias[c]), raw[c]));
    }
    const float state = step_in[j];
    const float sigma = round_bf16(__fdiv_rn(1.f, __fadd_rn(1.f, expf(-r[0]))));
    const float keep = round_bf16(__fmul_rn(sigma, state));
    const float update = round_bf16(__fmul_rn(round_bf16(__fsub_rn(1.f, sigma)),
                                              round_bf16(tanhf(r[1]))));
    step_in[j] = round_bf16(__fadd_rn(keep, update));
    latent[j] = round_bf16(tanhf(r[2]));
}

// The Markov logits of one step and the first pass of their argmax. For column c below
// vocab: bias = bf16(fold_bias[c] + acc[c]), where acc is the F32 product
// fold_weight · latent, then logit = bf16(30 tanh(bf16(base[c] + bias) / 30)). Writes
// the logits to `logits` if it is not null, and each block's best (value, index) to
// partial_value and partial_index.
extern "C" __global__ void __launch_bounds__(256)
draft_markov_logits(const __nv_bfloat16* base, const __nv_bfloat16* fold_bias, const float* acc,
                    int32_t vocab, float softcap, __nv_bfloat16* logits, float* partial_value,
                    int32_t* partial_index) {
    __shared__ float best_value[256];
    __shared__ int32_t best_index[256];
    float value = -INFINITY;
    int32_t index = INT32_MAX;
    for (int32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < vocab;
         c += gridDim.x * blockDim.x) {
        const float bias = round_bf16(__fadd_rn(bf(fold_bias[c]), acc[c]));
        const float total = round_bf16(__fadd_rn(bf(base[c]), bias));
        const __nv_bfloat16 logit =
            __float2bfloat16_rn(__fmul_rn(tanhf(__fdiv_rn(total, softcap)), softcap));
        if (logits != nullptr) {
            logits[c] = logit;
        }
        if (better(bf(logit), c, value, index)) {
            value = bf(logit);
            index = c;
        }
    }
    best_value[threadIdx.x] = value;
    best_index[threadIdx.x] = index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const int o = threadIdx.x + stride;
            if (better(best_value[o], best_index[o], best_value[threadIdx.x],
                       best_index[threadIdx.x])) {
                best_value[threadIdx.x] = best_value[o];
                best_index[threadIdx.x] = best_index[o];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_value[blockIdx.x] = best_value[0];
        partial_index[blockIdx.x] = best_index[0];
    }
}

// The second argmax pass: reduces `count` partials to the step's draft id, written to
// *id, and puts that token's markov_w1 row into step_in[256:512] for the next step. One
// block of 256 threads.
extern "C" __global__ void __launch_bounds__(kRank)
draft_markov_pick(const float* partial_value, const int32_t* partial_index, int32_t count,
                  int32_t* id, const __nv_bfloat16* w1, float* step_in) {
    __shared__ float best_value[kRank];
    __shared__ int32_t best_index[kRank];
    float value = -INFINITY;
    int32_t index = INT32_MAX;
    for (int i = threadIdx.x; i < count; i += kRank) {
        if (better(partial_value[i], partial_index[i], value, index)) {
            value = partial_value[i];
            index = partial_index[i];
        }
    }
    best_value[threadIdx.x] = value;
    best_index[threadIdx.x] = index;
    __syncthreads();
    for (int stride = kRank / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const int o = threadIdx.x + stride;
            if (better(best_value[o], best_index[o], best_value[threadIdx.x],
                       best_index[threadIdx.x])) {
                best_value[threadIdx.x] = best_value[o];
                best_index[threadIdx.x] = best_index[o];
            }
        }
        __syncthreads();
    }
    const int32_t token = best_index[0] == INT32_MAX ? 0 : best_index[0];
    if (threadIdx.x == 0) {
        *id = token;
    }
    step_in[kRank + threadIdx.x] = bf(w1[size_t(token) * kRank + threadIdx.x]);
}

namespace {

constexpr int kFoldRows = 16;
constexpr int kFoldK = 16;

}  // namespace

// fold_weight = bf16(lm_head · hidden_out) for lm_head BF16 [rows, 2304] and hidden_out
// BF16 [2304, 256], summed in F64 and rounded to F32, then to BF16, as vLLM's fold
// (torch rounds a double to BF16 through float). Grid ceil(rows / 16), 256 threads, one
// output column each.
extern "C" __global__ void __launch_bounds__(kRank)
draft_fold_weight(const __nv_bfloat16* lm_head, int32_t rows, const __nv_bfloat16* hidden_out,
                  __nv_bfloat16* fold_weight) {
    __shared__ double us[kFoldRows][kFoldK];
    __shared__ double hs[kFoldK][kRank];
    const int row0 = blockIdx.x * kFoldRows;
    const int j = threadIdx.x;
    double acc[kFoldRows];
#pragma unroll
    for (int r = 0; r < kFoldRows; ++r) {
        acc[r] = 0.0;
    }
    for (int k0 = 0; k0 < kHidden; k0 += kFoldK) {
        __syncthreads();
        {
            const int r = j / kFoldK, k = j % kFoldK;
            us[r][k] = row0 + r < rows ? double(bf(lm_head[size_t(row0 + r) * kHidden + k0 + k]))
                                       : 0.0;
        }
#pragma unroll
        for (int k = 0; k < kFoldK; ++k) {
            hs[k][j] = double(bf(hidden_out[size_t(k0 + k) * kRank + j]));
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kFoldK; ++k) {
            const double h = hs[k][j];
#pragma unroll
            for (int r = 0; r < kFoldRows; ++r) {
                acc[r] = fma(us[r][k], h, acc[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kFoldRows; ++r) {
        if (row0 + r < rows) {
            fold_weight[size_t(row0 + r) * kRank + j] =
                __float2bfloat16_rn(static_cast<float>(acc[r]));
        }
    }
}

// fold_bias = bf16(lm_head · hidden_out_bias) in F64, one warp per row of lm_head.
extern "C" __global__ void __launch_bounds__(256)
draft_fold_bias(const __nv_bfloat16* lm_head, int32_t rows, const __nv_bfloat16* bias,
                __nv_bfloat16* fold_bias) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    double acc = 0.0;
    for (int c = lane; c < kHidden; c += 32) {
        acc = fma(double(bf(lm_head[size_t(row) * kHidden + c])), double(bf(bias[c])), acc);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        fold_bias[row] = __float2bfloat16_rn(static_cast<float>(acc));
    }
}
