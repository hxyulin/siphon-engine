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
#include <cuda_fp8.h>
#include <cfloat>
#include <cstdint>

#include "pdl.cuh"

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

namespace {

// One row of draft_mix_rms_norm: x is BF16 [4, 2304], out BF16 [2304].
__device__ void mix_row(const __nv_bfloat16* x, const float* alpha, const __nv_bfloat16* weight,
                        __nv_bfloat16* out, float eps) {
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
    store_normed_row(v, weight, out, eps);
}

}  // namespace

// out[r] = rms_norm(sum_j alpha[j] * taps[r, j]; weight) for taps BF16 [rows, 4, 2304]:
// the draft's softmax mix of the four target features. One block per row.
extern "C" __global__ void __launch_bounds__(kRowThreads)
draft_mix_rms_norm(const __nv_bfloat16* taps, const float* alpha, const __nv_bfloat16* weight,
                   __nv_bfloat16* out, float eps) {
    const size_t row = blockIdx.x;
    mix_row(taps + row * kTaps * kHidden, alpha, weight, out + row * kHidden, eps);
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

namespace {

// Per-head RMSNorm of the 128 values at `in` with `weight`, rounded to BF16, then NeoX
// RoPE at `position`, written to `out`, which may equal `in`. A block of 64 threads:
// thread i rotates elements i and i + 64.
__device__ void norm_rope_head(const __nv_bfloat16* in, __nv_bfloat16* out,
                               const __nv_bfloat16* weight, const float* inv_freq,
                               int32_t position, float eps) {
    __shared__ float partial[kHalf / 32];
    const int i = threadIdx.x;
    const float x1 = bf(in[i]);
    const float x2 = bf(in[i + kHalf]);
    const float s = warp_sum(fmaf(x1, x1, x2 * x2));
    if (i % 32 == 0) {
        partial[i / 32] = s;
    }
    __syncthreads();
    const float inv = inv_rms(partial[0] + partial[1], kHeadDim, eps);
    const float n1 = round_bf16(__fmul_rn(__fmul_rn(x1, inv), bf(weight[i])));
    const float n2 = round_bf16(__fmul_rn(__fmul_rn(x2, inv), bf(weight[i + kHalf])));
    float sin_f, cos_f;
    sincosf(__fmul_rn(static_cast<float>(position), inv_freq[i]), &sin_f, &cos_f);
    out[i] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(n1, cos_f), __fmul_rn(n2, sin_f)));
    out[i + kHalf] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(n2, cos_f), __fmul_rn(n1, sin_f)));
}

}  // namespace

// In place on the leading `heads` 128-wide heads of each row of x (row stride
// `row_stride` elements): per-head RMSNorm with `weight`, rounded to BF16, then NeoX
// RoPE at position `start + row`. inv_freq is F32 [64], theta^(-2i/128). Grid (rows,
// heads), 64 threads: thread i rotates elements i and i + 64.
extern "C" __global__ void __launch_bounds__(kHalf)
draft_head_norm_rope(__nv_bfloat16* x, int32_t row_stride, const __nv_bfloat16* weight,
                     const float* inv_freq, int32_t start, float eps) {
    __nv_bfloat16* head = x + size_t(blockIdx.x) * row_stride + blockIdx.y * kHeadDim;
    norm_rope_head(head, head, weight, inv_freq, start + static_cast<int32_t>(blockIdx.x), eps);
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

constexpr int kAttnThreads = kBlock * 32;     // one warp per block row
constexpr int kQueryRows = kBlock * kGroup;  // 24 query rows per KV head
constexpr int kKeyChunk = 32;                // one key per lane
constexpr int kMaxSplits = 128;              // MAX_ATTENTION_SPLITS in src/ops/draft.rs
constexpr int kMinKeysPerSplit = 128;        // MIN_KEYS_PER_SPLIT in src/ops/draft.rs
constexpr int kKeyStride = kHeadDim + 8;     // BF16 elements per key row in shared memory
// Dynamic shared memory of draft_attention, ATTENTION_SMEM in src/ops/draft.rs.
constexpr int kAttentionSmem = 4 * kQueryRows * kHeadDim + 2 * 2 * kKeyChunk * kKeyStride +
                               2 * 2 * kKeyChunk * kHeadDim + 16 * kBlock * kKeyChunk;
static_assert(kAttentionSmem == 50176, "ATTENTION_SMEM in src/ops/draft.rs");

// The keys of each split of a block attention over `keys` keys, as attention_splits in
// src/ops/draft.rs: a multiple of 32, at least 128.
__device__ __forceinline__ int keys_per_split(int keys) {
    const int per = max((keys + kMaxSplits - 1) / kMaxSplits, kMinKeysPerSplit);
    return (per + 31) / 32 * 32;
}

// The first key position and the key count of a block attention. With `control`
// ([start, rows, anchor], the round's inputs) the block sits at start + rows and a
// window of `window` positions (0: all) precedes it; otherwise `first` and `keys` hold.
__device__ __forceinline__ void attention_range(const int32_t* control, int32_t window,
                                                int32_t& first, int32_t& keys) {
    if (control != nullptr) {
        const int32_t position = control[0] + control[1];
        first = window > 0 ? max(position - window, 0) : 0;
        keys = position + kBlock - first;
    }
}

__device__ __forceinline__ float bf_lo(uint32_t pair) { return __uint_as_float(pair << 16); }
__device__ __forceinline__ float bf_hi(uint32_t pair) { return __uint_as_float(pair & 0xffff0000u); }

__device__ __forceinline__ void cp_async16(void* smem, const void* global) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(global) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

}  // namespace

// One split of the block attention: the 8 rows' queries (BF16 [8, 24 * 128], after
// norm and RoPE) of KV head blockIdx.x against keys at positions
// first + split * keys_per_split, up to first + keys. Position p sits in cache slot
// p mod slots (caches BF16 [slots, 8, 128]). Writes the unnormalized F32 output
// [24, 128] and the row max and sum [24, 2] of the split. Query row r is token r / 3,
// head 3 * blockIdx.x + r % 3. Scale 128^-0.5.
//
// With `control` not null, `first` and `keys` come from the round's position (see
// attention_range) and blocks past the split count exit. Warp w holds the three query
// rows of token w and lane j key j of each chunk of 32. The arithmetic per query row is
// a fixed sequence, with every product and sum rounded on its own: per chunk of 32 keys,
// sequential F32 dot products over the 128 dimensions, the running max, the sum of the
// exponentials in key order, and the output rescaled then accumulated in key order.
// Splits depend only on the key count.
extern "C" __global__ void __launch_bounds__(kAttnThreads)
draft_attention(const __nv_bfloat16* q, const __nv_bfloat16* cache_k,
                const __nv_bfloat16* cache_v, int32_t slots, const int32_t* control,
                int32_t window, int32_t first, int32_t keys, float* partial_out,
                float* partial_stats) {
    // Dynamic shared memory, kAttentionSmem bytes: the queries, two chunks of keys and
    // values, and each warp's exponentials (rows in x, y and z).
    extern __shared__ __align__(16) unsigned char smem[];
    auto& qs = *reinterpret_cast<float (*)[kQueryRows][kHeadDim]>(smem);
    auto& ks = *reinterpret_cast<__nv_bfloat16 (*)[2][kKeyChunk][kKeyStride]>(
        smem + sizeof(float) * kQueryRows * kHeadDim);
    auto& vs = *reinterpret_cast<__nv_bfloat16 (*)[2][kKeyChunk][kHeadDim]>(
        smem + sizeof(float) * kQueryRows * kHeadDim + 2 * 2 * kKeyChunk * kKeyStride);
    auto& ps = *reinterpret_cast<float4 (*)[kBlock][kKeyChunk]>(
        smem + sizeof(float) * kQueryRows * kHeadDim + 2 * 2 * kKeyChunk * kKeyStride +
        2 * 2 * kKeyChunk * kHeadDim);

    pdl_wait();
    pdl_trigger();
    attention_range(control, window, first, keys);
    const int per_split = keys_per_split(keys);
    const int splits = (keys + per_split - 1) / per_split;
    const int kv = blockIdx.x;
    const int split = blockIdx.y;
    if (split >= splits) {
        return;
    }
    const int t = threadIdx.x;
    const int warp = t / 32, lane = t % 32;
    const float scale = rsqrtf(static_cast<float>(kHeadDim));

    const int begin = split * per_split;
    const int end = min(keys, begin + per_split);
    // Each thread copies 16-byte pieces: key t / 16 (and 16 more), piece t % 16.
    auto load_chunk = [&](int c0, int buf) {
        const int n = min(kKeyChunk, end - c0);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int key = t / 16 + half * 16;
            const int piece = t % 16;
            if (key < n) {
                const size_t slot = size_t((first + c0 + key) % slots);
                const size_t off = (slot * kKvHeads + kv) * kHeadDim + piece * 8;
                cp_async16(&ks[buf][key][piece * 8], cache_k + off);
                cp_async16(&vs[buf][key][piece * 8], cache_v + off);
            }
        }
        cp_async_commit();
    };
    load_chunk(begin, 0);

    for (int idx = t; idx < kQueryRows * kHeadDim; idx += kAttnThreads) {
        const int r = idx / kHeadDim, d = idx % kHeadDim;
        const int token = r / kGroup, head = kv * kGroup + r % kGroup;
        qs[r][d] = bf(q[size_t(token) * kGroup * kKvHeads * kHeadDim + head * kHeadDim + d]);
    }

    float row_max[kGroup], row_sum[kGroup], acc[kGroup][4];
#pragma unroll
    for (int i = 0; i < kGroup; ++i) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.f;
        }
    }

    int buf = 0;
    for (int c0 = begin; c0 < end; c0 += kKeyChunk, buf ^= 1) {
        const int n = min(kKeyChunk, end - c0);
        if (c0 + kKeyChunk < end) {
            load_chunk(c0 + kKeyChunk, buf ^ 1);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // Scores of key `lane` against the warp's three query rows.
        float s[kGroup];
        {
            float dot[kGroup] = {0.f, 0.f, 0.f};
            if (lane < n) {
                const __nv_bfloat16* kr = ks[buf][lane];
#pragma unroll 2
                for (int dd = 0; dd < kHeadDim; dd += 8) {
                    const uint4 kv8 = *reinterpret_cast<const uint4*>(kr + dd);
                    const float kval[8] = {bf_lo(kv8.x), bf_hi(kv8.x), bf_lo(kv8.y), bf_hi(kv8.y),
                                           bf_lo(kv8.z), bf_hi(kv8.z), bf_lo(kv8.w), bf_hi(kv8.w)};
#pragma unroll
                    for (int i = 0; i < kGroup; ++i) {
                        const float* qr = &qs[warp * kGroup + i][dd];
                        const float4 qa = *reinterpret_cast<const float4*>(qr);
                        const float4 qb = *reinterpret_cast<const float4*>(qr + 4);
                        const float qv[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            dot[i] = fmaf(qv[e], kval[e], dot[i]);
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < kGroup; ++i) {
                s[i] = lane < n ? __fmul_rn(dot[i], scale) : -INFINITY;
            }
        }

        float correction[kGroup];
        float4 p4 = make_float4(0.f, 0.f, 0.f, 0.f);
        float* p = &p4.x;
#pragma unroll
        for (int i = 0; i < kGroup; ++i) {
            float chunk_max = s[i];
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xffffffffu, chunk_max, offset));
            }
            const float m = fmaxf(row_max[i], chunk_max);
            correction[i] = expf(__fsub_rn(row_max[i], m));
            p[i] = lane < n ? expf(__fsub_rn(s[i], m)) : 0.f;
            row_max[i] = m;
        }
        ps[warp][lane] = p4;
        __syncwarp();
        {
            float sum[kGroup];
#pragma unroll
            for (int i = 0; i < kGroup; ++i) {
                sum[i] = __fmul_rn(row_sum[i], correction[i]);
            }
#pragma unroll 8
            for (int key = 0; key < kKeyChunk; ++key) {
                const float4 pk = ps[warp][key];
                sum[0] = __fadd_rn(sum[0], pk.x);
                sum[1] = __fadd_rn(sum[1], pk.y);
                sum[2] = __fadd_rn(sum[2], pk.z);
            }
#pragma unroll
            for (int i = 0; i < kGroup; ++i) {
                row_sum[i] = sum[i];
            }
        }

        // Lane j accumulates dimensions 4j to 4j + 3 of the three rows.
#pragma unroll
        for (int i = 0; i < kGroup; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = __fmul_rn(acc[i][j], correction[i]);
            }
        }
        for (int key = 0; key < n; ++key) {
            const float4 pk = ps[warp][key];
            const uint2 v4 = *reinterpret_cast<const uint2*>(&vs[buf][key][4 * lane]);
            const float v[4] = {bf_lo(v4.x), bf_hi(v4.x), bf_lo(v4.y), bf_hi(v4.y)};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[0][j] = fmaf(pk.x, v[j], acc[0][j]);
                acc[1][j] = fmaf(pk.y, v[j], acc[1][j]);
                acc[2][j] = fmaf(pk.z, v[j], acc[2][j]);
            }
        }
        __syncthreads();
    }

    const size_t base = (size_t(split) * kKvHeads + kv) * kQueryRows;
#pragma unroll
    for (int i = 0; i < kGroup; ++i) {
        const size_t r = base + warp * kGroup + i;
        *reinterpret_cast<float4*>(&partial_out[r * kHeadDim + 4 * lane]) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        if (lane == 0) {
            partial_stats[r * 2] = row_max[i];
            partial_stats[r * 2 + 1] = row_sum[i];
        }
    }
}

// Combines the splits of draft_attention into out, BF16 [8, 24 * 128]. Grid (8 KV
// heads, 24 query rows), 128 threads. With `control` not null the split count comes
// from the round's position, as in draft_attention; otherwise from first and keys.
extern "C" __global__ void __launch_bounds__(kHeadDim)
draft_attention_combine(const float* partial_out, const float* partial_stats,
                        const int32_t* control, int32_t window, int32_t first, int32_t keys,
                        __nv_bfloat16* out) {
    pdl_wait();
    pdl_trigger();
    attention_range(control, window, first, keys);
    const int per_split = keys_per_split(keys);
    const int splits = (keys + per_split - 1) / per_split;
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

// ---------------------------------------------------------------------------------------
// The round from device inputs. `control` is I32 [3]: the first position the round
// commits, the number of positions it commits (0 to 8) and the anchor token. The block
// sits right after the committed positions. These kernels run inside a CUDA graph, so
// nothing about a round's position reaches them from the host. Each is launched with
// programmatic dependent launch and reads the outputs of earlier kernels only after
// pdl_wait(), never through __restrict__ pointers.

namespace {

__device__ __forceinline__ int32_t control_rows(const int32_t* control) {
    return min(max(control[1], 0), kBlock);
}

// Streams weights past L1, as linear.cu does.
__device__ __forceinline__ uint4 load_weight(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ float lo(uint32_t pair) { return __uint_as_float(pair << 16); }
__device__ __forceinline__ float hi(uint32_t pair) { return __uint_as_float(pair & 0xffff0000u); }

// acc + dot(8 BF16 weights, x[8 i .. 8 i + 8]) in order, as linear.cu's F32-input GEMV.
__device__ __forceinline__ float dot8(uint4 w, const float* x, int i, float acc) {
    const float4 a = reinterpret_cast<const float4*>(x)[2 * i];
    const float4 b = reinterpret_cast<const float4*>(x)[2 * i + 1];
    acc = fmaf(lo(w.x), a.x, acc);
    acc = fmaf(hi(w.x), a.y, acc);
    acc = fmaf(lo(w.y), a.z, acc);
    acc = fmaf(hi(w.y), a.w, acc);
    acc = fmaf(lo(w.z), b.x, acc);
    acc = fmaf(hi(w.z), b.y, acc);
    acc = fmaf(lo(w.w), b.z, acc);
    acc = fmaf(hi(w.w), b.w, acc);
    return acc;
}

// The butterfly sum over a warp of linear.cu's GEMV, the same value in every lane.
__device__ __forceinline__ float butterfly(float acc) {
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

}  // namespace

// *dst = value. One thread.
extern "C" __global__ void draft_set_i32(int32_t* dst, int32_t value) { *dst = value; }

// Writes the round's inputs after a verify that kept *kept drafts: control = [start,
// min(*kept + 1, max_rows), *anchor]. One thread.
extern "C" __global__ void draft_set_control_kept(int32_t* control, int32_t start,
                                                  const int32_t* kept, int32_t max_rows,
                                                  const int32_t* anchor) {
    pdl_wait();
    control[0] = start;
    control[1] = min(max(*kept, 0) + 1, max_rows);
    control[2] = *anchor;
}

// Writes the round's inputs: control = [start, rows, *anchor]. One thread.
extern "C" __global__ void draft_set_control(int32_t* control, int32_t start, int32_t rows,
                                             const int32_t* anchor) {
    control[0] = start;
    control[1] = rows;
    control[2] = *anchor;
}

// The tap mixes of the round's committed rows: for row r below control[1] and layer l
// below 3, mixed[l, r] = mix(taps[r], alpha[l], norm_l), as draft_mix_rms_norm. The last
// committed row's Markov mix goes to f. Grid (8 rows, 4 mixes), 256 threads.
extern "C" __global__ void __launch_bounds__(kRowThreads)
draft_commit_mix(const __nv_bfloat16* taps, const int32_t* control, const float* alpha,
                 const __nv_bfloat16* norm0, const __nv_bfloat16* norm1,
                 const __nv_bfloat16* norm2, const __nv_bfloat16* norm3,
                 __nv_bfloat16* mixed, __nv_bfloat16* f, float eps) {
    pdl_wait();
    pdl_trigger();
    const int rows = control_rows(control);
    const int row = blockIdx.x, mix = blockIdx.y;
    if (row >= rows || (mix == kTaps - 1 && row != rows - 1)) {
        return;
    }
    const __nv_bfloat16* norms[kTaps] = {norm0, norm1, norm2, norm3};
    __nv_bfloat16* out =
        mix == kTaps - 1 ? f : mixed + (size_t(mix) * kBlock + row) * kHidden;
    mix_row(taps + size_t(row) * kTaps * kHidden, alpha + mix * kTaps, norms[mix], out, eps);
}

// The round's context keys and values: for row r below control[1], KV head h and layer
// l, the key head of ctx_k[l, r] gets draft_head_norm_rope's norm and RoPE at position
// start + r and lands in cache slot (start + r) mod slots, with the value head of
// ctx_v[l, r] beside it. Layer 1 skips positions at or past its slots. Grid (8 rows, 8
// heads, 3 layers), 64 threads.
extern "C" __global__ void __launch_bounds__(kHalf)
draft_commit_rope_store(const __nv_bfloat16* ctx_k, const __nv_bfloat16* ctx_v,
                        const int32_t* control, const float* inv_freq, float eps,
                        const __nv_bfloat16* k_norm0, const __nv_bfloat16* k_norm1,
                        const __nv_bfloat16* k_norm2, __nv_bfloat16* cache_k0,
                        __nv_bfloat16* cache_v0, int32_t slots0, __nv_bfloat16* cache_k1,
                        __nv_bfloat16* cache_v1, int32_t slots1, __nv_bfloat16* cache_k2,
                        __nv_bfloat16* cache_v2, int32_t slots2) {
    constexpr int kWidth = kKvHeads * kHeadDim;
    pdl_wait();
    pdl_trigger();
    const int rows = control_rows(control);
    const int row = blockIdx.x, head = blockIdx.y, layer = blockIdx.z;
    if (row >= rows) {
        return;
    }
    const int32_t position = control[0] + row;
    const __nv_bfloat16* norms[3] = {k_norm0, k_norm1, k_norm2};
    __nv_bfloat16* caches_k[3] = {cache_k0, cache_k1, cache_k2};
    __nv_bfloat16* caches_v[3] = {cache_v0, cache_v1, cache_v2};
    const int32_t slots[3] = {slots0, slots1, slots2};
    if (position < 0 || (layer == 1 && position >= slots[1])) {
        return;
    }
    const size_t src = (size_t(layer) * kBlock + row) * kWidth + head * kHeadDim;
    const size_t dst = size_t(position % slots[layer]) * kWidth + head * kHeadDim;
    norm_rope_head(ctx_k + src, caches_k[layer] + dst, norms[layer], inv_freq, position, eps);
    for (int i = threadIdx.x; i < kHeadDim; i += kHalf) {
        caches_v[layer][dst + i] = ctx_v[src + i];
    }
}

// The block's queries, keys and values at the round's position P = start + rows: the 24
// query heads of each row of q get draft_head_norm_rope's norm and RoPE at P + row in
// place; the 8 key heads of each row of k get them on their way to cache slot
// (P + row) mod slots, with the value heads of v beside them. With `limit` set,
// positions at or past `slots` are skipped. Grid (8 rows, 32 heads), 64 threads.
extern "C" __global__ void __launch_bounds__(kHalf)
draft_block_rope_store(__nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
                       const int32_t* control, const float* inv_freq, float eps,
                       const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                       __nv_bfloat16* cache_k, __nv_bfloat16* cache_v, int32_t slots,
                       int32_t limit) {
    constexpr int kWidth = kKvHeads * kHeadDim;
    constexpr int kQWidth = kKvHeads * kGroup * kHeadDim;
    pdl_wait();
    pdl_trigger();
    const int row = blockIdx.x, head = blockIdx.y;
    const int32_t position = control[0] + control_rows(control) + row;
    if (head < kKvHeads * kGroup) {
        __nv_bfloat16* x = q + size_t(row) * kQWidth + head * kHeadDim;
        norm_rope_head(x, x, q_norm, inv_freq, position, eps);
        return;
    }
    const int g = head - kKvHeads * kGroup;
    if (position < 0 || (limit != 0 && position >= slots)) {
        return;
    }
    const size_t src = size_t(row) * kWidth + g * kHeadDim;
    const size_t dst = size_t(position % slots) * kWidth + g * kHeadDim;
    norm_rope_head(k + src, cache_k + dst, k_norm, inv_freq, position, eps);
    for (int i = threadIdx.x; i < kHeadDim; i += kHalf) {
        cache_v[dst + i] = v[src + i];
    }
}

namespace {

constexpr int kStepWarps = 8;
constexpr int kStepBlocks = kRank / kStepWarps;  // one warp per state element
static_assert(kStepBlocks == 32, "STEP_BLOCKS in src/ops/draft.rs");

// The best (value, index) pair of a warp under `better`, in every lane.
__device__ __forceinline__ void warp_best(float& value, int32_t& index) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        const float v = __shfl_xor_sync(0xffffffffu, value, offset);
        const int32_t i = __shfl_xor_sync(0xffffffffu, index, offset);
        if (better(v, i, value, index)) {
            value = v;
            index = i;
        }
    }
}

// Sums v over the block into out[blockIdx.x]; `shared` holds a value per warp.
__device__ __forceinline__ void block_partial_sum(float v, float* shared, float* out) {
    v = warp_sum(v);
    if (threadIdx.x % 32 == 0) {
        shared[threadIdx.x / 32] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int w = 0; w < int(blockDim.x / 32); ++w) {
            total += shared[w];
        }
        out[blockIdx.x] = total;
    }
}

}  // namespace

// One Markov step, or with step == steps only the last pick. Picks the previous step's
// token: the anchor (control[2]) at step 0, otherwise the argmax of `count` partials
// from draft_markov_fold, written to ids[step - 1], with its confidence, the softmax
// probability exp(logit - cap) / sum of the `count` partial sums, at most 1, written to
// confidence[step - 1]. Then, for state element j (one warp
// each), the three rnn_proj rows j, 256 + j and 512 + j times [state; w1[token]] with
// linear.cu's split GEMV order, and the gate of draft_markov_gate: writes the new state
// to state_out and the latent to latent. state_in is unused at step 0, where the state
// is zero. Grid 32, 256 threads.
extern "C" __global__ void __launch_bounds__(kStepWarps * 32)
draft_markov_step(int32_t pdl, int32_t step, int32_t steps, const float* partial_value,
                  const int32_t* partial_index, const float* partial_sum, int32_t count,
                  float softcap, float* confidence, const int32_t* control,
                  int32_t* ids, const __nv_bfloat16* w1, int32_t vocab,
                  const __nv_bfloat16* rnn_state, const float* context, const float* bias,
                  const float* state_in, float* state_out, float* latent) {
    constexpr int kK = 2 * kRank;  // rnn_state is [768, 512]
    __shared__ __align__(16) float x[kK];
    __shared__ int32_t picked;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int j = blockIdx.x * kStepWarps + warp;
    const bool pick_only = step >= steps;

    // Weight loads first: no kernel writes them. A branch keeps pdl_wait after them.
    uint4 w[3][2];
#pragma unroll
    for (int part = 0; part < 3; ++part) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            w[part][h] = pick_only ? make_uint4(0, 0, 0, 0)
                                   : load_weight(rnn_state + size_t(part * kRank + j) * kK +
                                                 (h * 32 + lane) * 8);
        }
    }
    if (pdl) {
        pdl_wait();
    }
    pdl_trigger();

    if (warp == 0) {
        int32_t token;
        bool valid = true;
        if (step == 0) {
            token = control[2];
            valid = token >= 0 && token < vocab;
        } else {
            float value = -INFINITY;
            int32_t index = INT32_MAX;
            float total = 0.f;
            for (int i = lane; i < count; i += 32) {
                if (better(partial_value[i], partial_index[i], value, index)) {
                    value = partial_value[i];
                    index = partial_index[i];
                }
                total += partial_sum[i];
            }
            warp_best(value, index);
            total = warp_sum(total);
            token = index == INT32_MAX ? 0 : index;
            if (lane == 0 && blockIdx.x == 0) {
                const float p = __expf(value - softcap) / total;
                confidence[step - 1] = total > 0.f ? fminf(p, 1.f) : 0.f;
            }
        }
        if (lane == 0) {
            picked = valid ? token : -1;
            if (step > 0 && blockIdx.x == 0) {
                ids[step - 1] = token;
            }
        }
    }
    __syncthreads();
    if (pick_only) {
        return;
    }
    const int32_t token = picked;
    for (int i = threadIdx.x; i < kRank; i += kStepWarps * 32) {
        x[i] = step == 0 ? 0.f : state_in[i];
        x[kRank + i] = token >= 0 ? bf(w1[size_t(token) * kRank + i]) : 0.f;
    }
    __syncthreads();

    // linear.cu's split GEMV for k = 512: warps 0 and 1 of a row's block each sum one
    // 16-byte vector per lane, the first warp adds the second's sum, then six zero sums.
    float raw[3];
#pragma unroll
    for (int part = 0; part < 3; ++part) {
        const float a = butterfly(dot8(w[part][0], x, lane, 0.f));
        const float b = butterfly(dot8(w[part][1], x, 32 + lane, 0.f));
        float sum = a + b;
#pragma unroll
        for (int zero = 0; zero < kStepWarps - 2; ++zero) {
            sum += 0.f;
        }
        raw[part] = sum;
    }
    if (lane != 0) {
        return;
    }
    float r[3];
#pragma unroll
    for (int part = 0; part < 3; ++part) {
        const int c = part * kRank + j;
        r[part] = round_bf16(__fadd_rn(__fadd_rn(context[c], bias[c]), raw[part]));
    }
    const float state = x[j];
    const float sigma = round_bf16(__fdiv_rn(1.f, __fadd_rn(1.f, expf(-r[0]))));
    const float keep = round_bf16(__fmul_rn(sigma, state));
    const float update = round_bf16(__fmul_rn(round_bf16(__fsub_rn(1.f, sigma)),
                                              round_bf16(tanhf(r[1]))));
    state_out[j] = round_bf16(__fadd_rn(keep, update));
    latent[j] = round_bf16(tanhf(r[2]));
}

namespace {

constexpr int kFoldWarps = 8;
constexpr int kFoldGroup = 16;  // rows a warp sums at once, one 16-byte load per lane each

}  // namespace

// One Markov step's logits and the first pass of their argmax, fused with the product
// fold_weight · latent (fold_weight BF16 [rows, 256], latent F32 [256]): each row sums
// as linear.cu's F32-input GEMV for k = 256, then logit = bf16(cap tanh(bf16(base +
// bf16(fold_bias + acc)) / cap)) as draft_markov_logits. Writes the logits below `vocab`
// to `logits` if it is not null, each block's best (value, index) to partial_value
// and partial_index, and each block's sum of exp(logit - cap) to partial_sum. With
// `reverse` set, the warps walk the rows from the end, so consecutive steps meet the rows
// the previous one read last while they are still in L2.
extern "C" __global__ void __launch_bounds__(kFoldWarps * 32)
draft_markov_fold(int32_t pdl, int32_t reverse, const __nv_bfloat16* fold_weight,
                  const __nv_bfloat16* fold_bias, const __nv_bfloat16* base, const float* latent,
                  int32_t vocab, float softcap, __nv_bfloat16* logits, float* partial_value,
                  int32_t* partial_index, float* partial_sum) {
    __shared__ __align__(16) float z[kRank];
    __shared__ float warp_sums[kFoldWarps];
    __shared__ float best_value[kFoldWarps];
    __shared__ int32_t best_index[kFoldWarps];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int groups = (vocab + kFoldGroup - 1) / kFoldGroup;
    const int stride = gridDim.x * kFoldWarps;
    const int first = blockIdx.x * kFoldWarps + warp;
    auto group_of = [&](int g) { return reverse ? groups - 1 - g : g; };
    // Rows past the end load the last row, so no load sits behind a branch; their sums
    // go unused.
    auto load = [&](int g, uint4 (&wv)[kFoldGroup]) {
        const int row0 = group_of(g) * kFoldGroup;
#pragma unroll
        for (int i = 0; i < kFoldGroup; ++i) {
            wv[i] = load_weight(fold_weight + size_t(min(row0 + i, vocab - 1)) * kRank + lane * 8);
        }
    };

    uint4 wv[kFoldGroup];
    if (first < groups) {
        load(first, wv);
    }
    if (pdl) {
        pdl_wait();
    }
    pdl_trigger();
    z[threadIdx.x] = latent[threadIdx.x];
    __syncthreads();

    float value = -INFINITY;
    int32_t index = INT32_MAX;
    float sum = 0.f;
    for (int g = first; g < groups; g += stride) {
        float mine = 0.f;
#pragma unroll
        for (int i = 0; i < kFoldGroup; ++i) {
            const float acc = butterfly(dot8(wv[i], z, lane, 0.f));
            if (lane == i) {
                mine = acc;
            }
        }
        const int c = group_of(g) * kFoldGroup + lane;
        if (g + stride < groups) {
            load(g + stride, wv);
        }
        if (lane < kFoldGroup && c < vocab) {
            const float bias = round_bf16(__fadd_rn(bf(fold_bias[c]), mine));
            const float total = round_bf16(__fadd_rn(bf(base[c]), bias));
            const __nv_bfloat16 logit =
                __float2bfloat16_rn(__fmul_rn(tanhf(__fdiv_rn(total, softcap)), softcap));
            if (logits != nullptr) {
                logits[c] = logit;
            }
            sum += __expf(bf(logit) - softcap);
            if (better(bf(logit), c, value, index)) {
                value = bf(logit);
                index = c;
            }
        }
    }
    block_partial_sum(sum, warp_sums, partial_sum);
    warp_best(value, index);
    if (lane == 0) {
        best_value[warp] = value;
        best_index[warp] = index;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < kFoldWarps; ++w) {
            if (better(best_value[w], best_index[w], value, index)) {
                value = best_value[w];
                index = best_index[w];
            }
        }
        partial_value[blockIdx.x] = value;
        partial_index[blockIdx.x] = index;
    }
}

// ---------------------------------------------------------------------------------------
// Approximations. A draft token changes only how often the target accepts, so the draft
// may trade its numerics for time: FP8 weights, and a head over a shortlist of tokens.

namespace {

constexpr float kE4m3Max = 448.f;

__device__ __forceinline__ float2 e4m3x2(uint32_t pair) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3);
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// The 16 E4M3 values of a 16-byte vector, widened.
__device__ __forceinline__ void e4m3x16(uint4 v, float (&w)[16]) {
    const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 a = e4m3x2(words[i] & 0xffffu);
        const float2 b = e4m3x2(words[i] >> 16);
        w[4 * i] = a.x;
        w[4 * i + 1] = a.y;
        w[4 * i + 2] = b.x;
        w[4 * i + 3] = b.y;
    }
}

}  // namespace

// Quantizes rows of a BF16 matrix [*, k] to E4M3 with one F32 scale per row, as
// linear_fp8.cu's quantize_e4m3_rows: row r of the output (at first + r) is source row
// ids[r], or r without ids. One block of 256 threads per row.
extern "C" __global__ void __launch_bounds__(256)
draft_quantize_rows(const __nv_bfloat16* w, int32_t k, const int32_t* ids, int32_t first,
                    uint8_t* q, float* scale) {
    __shared__ float warp_max[8];
    const int row = blockIdx.x;
    const __nv_bfloat16* src = w + size_t(ids != nullptr ? ids[row] : row) * k;
    float amax = 0.f;
    for (int i = threadIdx.x; i < k; i += 256) {
        amax = fmaxf(amax, fabsf(bf(src[i])));
    }
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    if (threadIdx.x % 32 == 0) {
        warp_max[threadIdx.x / 32] = amax;
    }
    __syncthreads();
    amax = warp_max[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        amax = fmaxf(amax, warp_max[i]);
    }
    const float s = amax / kE4m3Max;
    if (threadIdx.x == 0) {
        scale[first + row] = s;
    }
    uint8_t* dst = q + size_t(first + row) * k;
    for (int i = threadIdx.x; i < k; i += 256) {
        const float v = s > 0.f ? bf(src[i]) / s : 0.f;
        dst[i] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}

// dst[first + r] = src[ids[r]] for rows of k BF16 elements. One block per row.
extern "C" __global__ void draft_gather_rows(const __nv_bfloat16* src, int32_t k,
                                             const int32_t* ids, int32_t first,
                                             __nv_bfloat16* dst) {
    const size_t from = size_t(ids[blockIdx.x]) * k;
    const size_t to = size_t(first + blockIdx.x) * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        dst[to + i] = src[from + i];
    }
}

namespace {

constexpr int kF8Tile = 3072;  // input columns staged in shared memory at once
constexpr int kF8MaxM = 8;     // input rows
constexpr int kF8Rows = 2;     // weight rows per warp
constexpr int kF8Loads = kF8Tile / 16 / 32;  // 16-byte loads per lane and row per tile
constexpr int kF8MaxSegments = 3;
// Dynamic shared memory of draft_gemm_fp8, F8_SMEM in src/ops/draft.rs.
constexpr int kF8Smem = kF8MaxM * kF8Tile * 2;
static_assert(kF8Smem == 49152, "F8_SMEM in src/ops/draft.rs");

struct F8Segment {
    const uint8_t* w;
    const float* scale;
    __nv_bfloat16* out;
    int rows;
    int ld;  // elements between output rows
    // Row groups of this segment and every segment before it.
    int block_end;
};

struct F8Batch {
    F8Segment seg[kF8MaxSegments];
    const __nv_bfloat16* x;
    int count;
    int m;
    int k;
};

// Stages columns t0 .. t0 + len of the first m rows of x (row stride k) in xs, zeroing
// rows m to 7.
__device__ __forceinline__ void stage_x(__nv_bfloat16* xs, const __nv_bfloat16* x, int m, int k,
                                        int t0, int len) {
    const int vecs = len / 8;
    for (int i = threadIdx.x; i < kF8MaxM * vecs; i += blockDim.x) {
        const int r = i / vecs, c = i % vecs;
        const uint4 v = r < m ? *reinterpret_cast<const uint4*>(x + size_t(r) * k + t0 + c * 8)
                              : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(xs + r * kF8Tile + c * 8) = v;
    }
}

}  // namespace

// out[i, n] = bf16(scale[n] * sum_j x[i, j] w[n, j]) for up to 8 input rows x (BF16
// [m, k]) and E4M3 weights w [rows, k] with one F32 scale per row, for up to 3 products
// that share x. A group of 16 weight rows is one warp's 2 rows for each of 8 warps; each
// warp sums its rows against all input rows, 16 columns per lane load, and lanes reduce
// with a butterfly. The blocks loop over the groups. x sits in shared memory: staged
// once when k is at most 3072, otherwise in tiles of 3072 columns per group. k is a
// multiple of 16.
extern "C" __global__ void __launch_bounds__(256, 2)
draft_gemm_fp8(const __grid_constant__ F8Batch b) {
    extern __shared__ __align__(16) unsigned char f8_smem[];
    __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(f8_smem);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int k = b.k;
    const bool once = k <= kF8Tile;
    const int groups = b.seg[b.count - 1].block_end;
    pdl_wait();
    pdl_trigger();
    if (once) {
        stage_x(xs, b.x, b.m, k, 0, k);
        __syncthreads();
    }
    for (int group = blockIdx.x; group < groups; group += gridDim.x) {
        int s = 0;
        int group_begin = 0;
        while (s + 1 < b.count && group >= b.seg[s].block_end) {
            group_begin = b.seg[s].block_end;
            ++s;
        }
        const F8Segment& seg = b.seg[s];
        const int row0 = ((group - group_begin) * 8 + warp) * kF8Rows;
        float acc[kF8Rows][kF8MaxM];
#pragma unroll
        for (int r = 0; r < kF8Rows; ++r) {
#pragma unroll
            for (int i = 0; i < kF8MaxM; ++i) {
                acc[r][i] = 0.f;
            }
        }
        for (int t0 = 0; t0 < k; t0 += kF8Tile) {
            const int len = min(kF8Tile, k - t0);
            const int chunks = len / 16;
            uint4 wv[kF8Rows][kF8Loads];
            // Out-of-range rows and chunks load the last ones, so no load sits behind a
            // branch; their products go unused.
#pragma unroll
            for (int r = 0; r < kF8Rows; ++r) {
                const size_t row = min(row0 + r, seg.rows - 1);
#pragma unroll
                for (int j = 0; j < kF8Loads; ++j) {
                    const int c = min(lane + 32 * j, chunks - 1);
                    wv[r][j] = load_weight(seg.w + row * k + t0 + c * 16);
                }
            }
            if (!once) {
                __syncthreads();
                stage_x(xs, b.x, b.m, k, t0, len);
                __syncthreads();
            }
#pragma unroll
            for (int j = 0; j < kF8Loads; ++j) {
                const int c = lane + 32 * j;
                if (c >= chunks) {
                    break;
                }
                float w[kF8Rows][16];
#pragma unroll
                for (int r = 0; r < kF8Rows; ++r) {
                    e4m3x16(wv[r][j], w[r]);
                }
#pragma unroll
                for (int i = 0; i < kF8MaxM; ++i) {
                    const __nv_bfloat16* xr = xs + i * kF8Tile + c * 16;
                    const uint4 xa = *reinterpret_cast<const uint4*>(xr);
                    const uint4 xb = *reinterpret_cast<const uint4*>(xr + 8);
                    const float xv[16] = {lo(xa.x), hi(xa.x), lo(xa.y), hi(xa.y),
                                          lo(xa.z), hi(xa.z), lo(xa.w), hi(xa.w),
                                          lo(xb.x), hi(xb.x), lo(xb.y), hi(xb.y),
                                          lo(xb.z), hi(xb.z), lo(xb.w), hi(xb.w)};
#pragma unroll
                    for (int r = 0; r < kF8Rows; ++r) {
#pragma unroll
                        for (int e = 0; e < 16; ++e) {
                            acc[r][i] = fmaf(w[r][e], xv[e], acc[r][i]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kF8Rows; ++r) {
#pragma unroll
            for (int i = 0; i < kF8MaxM; ++i) {
                acc[r][i] = butterfly(acc[r][i]);
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < kF8Rows; ++r) {
                const int row = row0 + r;
                if (row < seg.rows) {
                    const float sc = seg.scale[row];
#pragma unroll
                    for (int i = 0; i < kF8MaxM; ++i) {
                        if (i < b.m) {
                            seg.out[size_t(i) * seg.ld + row] =
                                __float2bfloat16_rn(acc[r][i] * sc);
                        }
                    }
                }
            }
        }
    }
}

namespace {

// A Markov logit from its pre-cap total: bf16(cap tanh(total / cap)).
__device__ __forceinline__ float step_logit(float total, float softcap) {
    return bf(__float2bfloat16_rn(__fmul_rn(tanhf(__fdiv_rn(total, softcap)), softcap)));
}

// Reduces each thread's best (value, index) over the block into partial[blockIdx.x].
__device__ void fold_partial(float value, int32_t index, float* best_value, int32_t* best_index,
                             float* partial_value, int32_t* partial_index) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    warp_best(value, index);
    if (lane == 0) {
        best_value[warp] = value;
        best_index[warp] = index;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < int(blockDim.x / 32); ++w) {
            if (better(best_value[w], best_index[w], value, index)) {
                value = best_value[w];
                index = best_index[w];
            }
        }
        partial_value[blockIdx.x] = value;
        partial_index[blockIdx.x] = index;
    }
}

}  // namespace

// One Markov step's logits and the first pass of their argmax over head rows that are
// FP8, a shortlist, or both: row n's fold product is sum_j w[n, j] latent[j], times
// scale[n] when the weights are E4M3 (`scale` not null), else BF16. Then logit =
// bf16(cap tanh(bf16(base[n] + bf16(fold_bias[n] + acc)) / cap)). The rows are
// `rows` or, with `count` not null, *count; row n stands for token map[n], or n without
// a map. Each block writes its best (value, token) pair and its sum of exp(logit - cap)
// to the partials.
extern "C" __global__ void __launch_bounds__(kFoldWarps * 32)
draft_markov_fold_any(const void* weight, const float* scale, const __nv_bfloat16* fold_bias,
                      const __nv_bfloat16* base, const float* latent, int32_t rows,
                      const int32_t* count, const int32_t* map, float softcap,
                      float* partial_value, int32_t* partial_index, float* partial_sum) {
    __shared__ __align__(16) float z[kRank];
    __shared__ float warp_sums[kFoldWarps];
    __shared__ float best_value[kFoldWarps];
    __shared__ int32_t best_index[kFoldWarps];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    pdl_wait();
    pdl_trigger();
    if (count != nullptr) {
        rows = *count;
    }
    z[threadIdx.x] = latent[threadIdx.x];
    __syncthreads();
    const bool fp8 = scale != nullptr;
    const int groups = (rows + kFoldGroup - 1) / kFoldGroup;
    const int stride = gridDim.x * kFoldWarps;
    float value = -INFINITY;
    int32_t index = INT32_MAX;
    float sum = 0.f;
    for (int g = blockIdx.x * kFoldWarps + warp; g < groups; g += stride) {
        const int row0 = g * kFoldGroup;
        float mine = 0.f;
        if (fp8) {
            // A row is 256 bytes: lanes 0-15 take 16 bytes each, lanes 16-31 the next row.
            uint4 wv[kFoldGroup / 2];
#pragma unroll
            for (int i = 0; i < kFoldGroup / 2; ++i) {
                const int row = min(row0 + 2 * i + lane / 16, rows - 1);
                wv[i] = load_weight(static_cast<const uint8_t*>(weight) + size_t(row) * kRank +
                                    (lane % 16) * 16);
            }
#pragma unroll
            for (int i = 0; i < kFoldGroup / 2; ++i) {
                float w[16];
                e4m3x16(wv[i], w);
                float acc = 0.f;
#pragma unroll
                for (int e = 0; e < 16; ++e) {
                    acc = fmaf(w[e], z[(lane % 16) * 16 + e], acc);
                }
#pragma unroll
                for (int offset = 8; offset >= 1; offset /= 2) {
                    acc += __shfl_xor_sync(0xffffffffu, acc, offset);
                }
                // Lane 0 holds row 2i, lane 16 row 2i + 1.
                const float other = __shfl_sync(0xffffffffu, acc, 16);
                if (lane == 2 * i) {
                    mine = acc;
                }
                if (lane == 2 * i + 1) {
                    mine = other;
                }
            }
        } else {
            uint4 wv[kFoldGroup];
#pragma unroll
            for (int i = 0; i < kFoldGroup; ++i) {
                wv[i] = load_weight(static_cast<const __nv_bfloat16*>(weight) +
                                    size_t(min(row0 + i, rows - 1)) * kRank + lane * 8);
            }
#pragma unroll
            for (int i = 0; i < kFoldGroup; ++i) {
                const float acc = butterfly(dot8(wv[i], z, lane, 0.f));
                if (lane == i) {
                    mine = acc;
                }
            }
        }
        const int n = row0 + lane;
        if (lane < kFoldGroup && n < rows) {
            if (fp8) {
                mine *= scale[n];
            }
            const float bias = round_bf16(__fadd_rn(bf(fold_bias[n]), mine));
            const float total = round_bf16(__fadd_rn(bf(base[n]), bias));
            const float logit = step_logit(total, softcap);
            sum += __expf(logit - softcap);
            const int32_t token = map != nullptr ? map[n] : n;
            if (better(logit, token, value, index)) {
                value = logit;
                index = token;
            }
        }
    }
    block_partial_sum(sum, warp_sums, partial_sum);
    fold_partial(value, index, best_value, best_index, partial_value, partial_index);
}

// ---------------------------------------------------------------------------------------
// The exact Markov argmax, pruned. A step's logit of row c is
// L(acc_c) = logit(bf16(base_c + bf16(fold_bias_c + acc_c))), acc_c = fold_weight[c] · z,
// and each operation is monotone in acc_c. An E4M3 copy of the fold, which stays in L2
// across the steps, gives an estimate ã_c with |acc_c - ã_c| <= err_c: the L1 norm of the
// row's quantization residual (the latent is a BF16 tanh, |z_k| <= 1), plus room for the
// F32 rounding of both sums. So L(ã_c - err_c) <= L_c <= L(ã_c + err_c). With L* the
// largest lower bound, a row whose upper bound is below L* has a smaller logit than some
// other row and cannot be the argmax, ties included. Each bound widens by one BF16 step,
// so a tanhf that is not monotone by an ulp cannot drop the argmax. Only the remaining
// rows get the exact BF16 product, in draft_markov_fold's arithmetic, so the argmax is the
// one draft_markov_fold finds.

namespace {

// The next BF16 value above or below v, a BF16 value.
__device__ __forceinline__ float bf16_up(float v) {
    uint32_t b = __float_as_uint(v) >> 16;
    b = b == 0x8000u ? 0x0001u : (b & 0x8000u) ? b - 1 : b + 1;
    return __uint_as_float(b << 16);
}

__device__ __forceinline__ float bf16_down(float v) {
    uint32_t b = __float_as_uint(v) >> 16;
    b = b == 0x0000u ? 0x8001u : (b & 0x8000u) ? b + 1 : b - 1;
    return __uint_as_float(b << 16);
}

// The logit of row c when its fold product is a.
__device__ __forceinline__ float logit_at(float base, float fold_bias, float a, float softcap) {
    const float bias = round_bf16(__fadd_rn(fold_bias, a));
    return step_logit(round_bf16(__fadd_rn(base, bias)), softcap);
}

}  // namespace

// err[c] for the pruned argmax: sum_k |w[c, k] - scale[c] q[c, k]| + 2^-18 sum_k |w[c, k]|,
// in F64 and rounded up, for the BF16 fold w and its E4M3 copy (q, scale). One warp per
// row.
extern "C" __global__ void __launch_bounds__(256)
draft_fold_error(const __nv_bfloat16* w, const uint8_t* q, const float* scale, int32_t rows,
                 float* err) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    double residual = 0.0, norm = 0.0;
    const double s = scale[row];
    for (int k = lane; k < kRank; k += 32) {
        const double wv = bf(w[size_t(row) * kRank + k]);
        const __half_raw h = __nv_cvt_fp8_to_halfraw(q[size_t(row) * kRank + k], __NV_E4M3);
        const double qv = __half2float(*reinterpret_cast<const __half*>(&h));
        residual += fabs(wv - s * qv);
        norm += fabs(wv);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        residual += __shfl_xor_sync(0xffffffffu, residual, offset);
        norm += __shfl_xor_sync(0xffffffffu, norm, offset);
    }
    if (lane == 0) {
        err[row] = __double2float_ru(residual + norm * 0x1p-18);
    }
}

// The bounds of one step's logits from the E4M3 fold (q, scale) and err: writes each
// row's upper bound to upper[c] and each block's largest lower bound to
// lower[blockIdx.x], and each block's sum of exp(logit - cap) over the estimates' logits
// to partial_sum[blockIdx.x], for the confidence. A warp takes 32 rows at a time: a row is 256 bytes, so lanes 0-15
// load 16 bytes each of one row and lanes 16-31 of the next, and after the sums lane j
// finishes row j with coalesced loads of its scale, error, base logit and fold bias.
// With `reverse` set, the warps walk the rows from the end, so consecutive steps meet the
// rows the previous one read last while they are still in L2. Grid FOLD_BLOCKS, 256
// threads.
extern "C" __global__ void __launch_bounds__(kFoldWarps * 32)
draft_markov_bound(int32_t reverse, const uint8_t* q, const float* scale, const float* err,
                   const __nv_bfloat16* fold_bias, const __nv_bfloat16* base,
                   const float* latent, int32_t vocab, float softcap, float* upper,
                   float* lower, float* partial_sum) {
    constexpr int kRowsPerGroup = 32;
    constexpr int kPairs = kRowsPerGroup / 2;
    __shared__ __align__(16) float z[kRank];
    __shared__ float block_lower[kFoldWarps];
    __shared__ float warp_sums[kFoldWarps];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int groups = (vocab + kRowsPerGroup - 1) / kRowsPerGroup;
    const int stride = gridDim.x * kFoldWarps;
    pdl_wait();
    pdl_trigger();
    z[threadIdx.x] = latent[threadIdx.x];
    __syncthreads();
    float zl[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) {
        zl[e] = z[(lane % 16) * 16 + e];
    }
    float best = -INFINITY;
    float sum = 0.f;
    for (int g = blockIdx.x * kFoldWarps + warp; g < groups; g += stride) {
        const int row0 = (reverse ? groups - 1 - g : g) * kRowsPerGroup;
        // Rows past the end read the last row, so no load sits behind a branch.
        const int c = min(row0 + lane, vocab - 1);
        const float sc = scale[c], e = err[c];
        const float b = bf(base[c]), fb = bf(fold_bias[c]);
        float mine = 0.f;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            uint4 wv[kPairs / 2];
#pragma unroll
            for (int i = 0; i < kPairs / 2; ++i) {
                const int row = min(row0 + 2 * (half * kPairs / 2 + i) + lane / 16, vocab - 1);
                wv[i] = load_weight(q + size_t(row) * kRank + (lane % 16) * 16);
            }
#pragma unroll
            for (int i = 0; i < kPairs / 2; ++i) {
                float w[16];
                e4m3x16(wv[i], w);
                float a = 0.f;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    a = fmaf(w[k], zl[k], a);
                }
#pragma unroll
                for (int offset = 8; offset >= 1; offset /= 2) {
                    a += __shfl_xor_sync(0xffffffffu, a, offset);
                }
                // Lanes 0-15 hold row 2p, lanes 16-31 row 2p + 1, for pair p.
                const int p = half * kPairs / 2 + i;
                const float v = __shfl_sync(0xffffffffu, a, (lane % 2) * 16);
                if (lane / 2 == p) {
                    mine = v;
                }
            }
        }
        if (row0 + lane < vocab) {
            const float estimate = mine * sc;
            best = fmaxf(best, logit_at(b, fb, estimate - e, softcap));
            upper[row0 + lane] = logit_at(b, fb, estimate + e, softcap);
            sum += __expf(logit_at(b, fb, estimate, softcap) - softcap);
        }
    }
    block_partial_sum(sum, warp_sums, partial_sum);
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, offset));
    }
    if (lane == 0) {
        block_lower[warp] = best;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < kFoldWarps; ++w) {
            best = fmaxf(best, block_lower[w]);
        }
        lower[blockIdx.x] = best;
    }
}

// The exact argmax of one step over the rows draft_markov_bound leaves in the running:
// those whose upper bound, one BF16 step up, reaches the largest of the `count` lower
// bounds, one BF16 step down. Each such row gets draft_markov_fold's arithmetic. Writes
// each block's best (value, index) to the partials. Grid FOLD_BLOCKS, 256 threads.
extern "C" __global__ void __launch_bounds__(kFoldWarps * 32)
draft_markov_exact(const __nv_bfloat16* fold_weight, const __nv_bfloat16* fold_bias,
                   const __nv_bfloat16* base, const float* latent, const float* upper,
                   const float* lower, int32_t count, int32_t vocab, float softcap,
                   float* partial_value, int32_t* partial_index) {
    __shared__ __align__(16) float z[kRank];
    __shared__ float floor_s;
    __shared__ float best_value[kFoldWarps];
    __shared__ int32_t best_index[kFoldWarps];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    pdl_wait();
    pdl_trigger();
    z[threadIdx.x] = latent[threadIdx.x];
    if (warp == 0) {
        float m = -INFINITY;
        for (int i = lane; i < count; i += 32) {
            m = fmaxf(m, lower[i]);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
        }
        if (lane == 0) {
            floor_s = bf16_down(m);
        }
    }
    __syncthreads();
    const float floor = floor_s;
    float value = -INFINITY;
    int32_t index = INT32_MAX;
    const int stride = gridDim.x * kFoldWarps * 32;
    for (int c0 = (blockIdx.x * kFoldWarps + warp) * 32; c0 < vocab; c0 += stride) {
        const int c = c0 + lane;
        const bool keep = c < vocab && bf16_up(upper[c]) >= floor;
        unsigned mask = __ballot_sync(0xffffffffu, keep);
        while (mask != 0) {
            const int bit = __ffs(mask) - 1;
            mask &= mask - 1;
            const int row = c0 + bit;
            const uint4 w = load_weight(fold_weight + size_t(row) * kRank + lane * 8);
            const float acc = butterfly(dot8(w, z, lane, 0.f));
            if (lane == bit) {
                const float bias = round_bf16(__fadd_rn(bf(fold_bias[row]), acc));
                const float total = round_bf16(__fadd_rn(bf(base[row]), bias));
                const float logit = bf(
                    __float2bfloat16_rn(__fmul_rn(tanhf(__fdiv_rn(total, softcap)), softcap)));
                if (better(logit, row, value, index)) {
                    value = logit;
                    index = row;
                }
            }
        }
    }
    fold_partial(value, index, best_value, best_index, partial_value, partial_index);
}
