// Kernels of the vision tower: patch and position embeddings, LayerNorm with the residual
// update, the attention projections' bias and 2D rotary embedding, bidirectional attention
// over one image, and GELU. The matrix products run through `linear.cu` (or `linear_fp8.cu`).
//
// The forward pass is Transformers' `Inkling2VisionModel` (transformers 5.24.0). The
// residual stream stays in F32 between layers, where the BF16 reference rounds it to BF16;
// every product input and output is BF16.
//
// Row layout: row 0 is the class token, rows 1..S-1 the patches in input order (the 2x2
// patches of each image token together, the tokens in raster order). All row buffers are
// [S, 1024].
//
// Plain sm_80 instructions only (mma.sync, ldmatrix, cp.async), so the file also builds
// for sm_89.

#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kDim = 1024;
constexpr int kHeadDim = 64;
// Side of the learned position-embedding grid.
constexpr int kTable = 32;
// Threads of the one-block-per-row kernels, each owning 4 consecutive columns.
constexpr int kRowThreads = 256;
constexpr int kPerThread = kDim / kRowThreads;
static_assert(kPerThread == 4);

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 rb(float x) { return __float2bfloat16_rn(x); }

// Four consecutive BF16 values as F32.
__device__ __forceinline__ void load4(const __nv_bfloat16* p, float (&v)[4]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    v[0] = __low2float(a);
    v[1] = __high2float(a);
    v[2] = __low2float(b);
    v[3] = __high2float(b);
}

__device__ __forceinline__ void store4(__nv_bfloat16* p, const float (&v)[4]) {
    __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]);
    __nv_bfloat162 b = __floats2bfloat162_rn(v[2], v[3]);
    uint2 raw;
    raw.x = *reinterpret_cast<uint32_t*>(&a);
    raw.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = raw;
}

// Row and column of patch `p` (0-based, the class token not counted) in a grid of
// `grid_h` x `grid_w` patches per frame, patches in 2x2-merge-block order. Frames repeat
// the same positions, as Transformers' `get_vision_position_ids` does.
__device__ __forceinline__ void patch_cell(int p, int grid_h, int grid_w, int& row, int& col) {
    const int within = p % (grid_h * grid_w);
    const int blocks_w = grid_w / 2;
    const int block = within / 4;
    row = (block / blocks_w) * 2 + (within / 2) % 2;
    col = (block % blocks_w) * 2 + within % 2;
}

// Sum of `v` over the block's 256 threads, returned to every thread. `shared` holds 8
// floats; the leading barrier lets a caller reuse it right after a previous call.
__device__ __forceinline__ float block_sum(float v, float* shared) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (threadIdx.x % 32 == 0) shared[threadIdx.x / 32] = v;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kRowThreads / 32; ++w) total += shared[w];
    return total;
}

// LayerNorm of a row held 4 values per thread, with weight and bias, in F32.
__device__ __forceinline__ void layer_norm(float (&v)[4], const __nv_bfloat16* w,
                                           const __nv_bfloat16* b, int col, float eps,
                                           float* shared) {
    const float mean = block_sum(v[0] + v[1] + v[2] + v[3], shared) / kDim;
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float d = v[i] - mean;
        sq += d * d;
    }
    const float rstd = 1.f / sqrtf(block_sum(sq, shared) / kDim + eps);
    float wv[4], bv[4];
    load4(w + col, wv);
    load4(b + col, bv);
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = (v[i] - mean) * rstd * wv[i] + bv[i];
}

// Taps and weights of bilinear interpolation (half-pixel centres, edge taps clamped) from
// the kTable-long side of the grid to `index` of `size`, as Transformers'
// `_interpolation_axis_taps_weights(mode="bilinear", align_corners=False)` in F32.
__device__ __forceinline__ void axis_taps(int index, int size, int (&tap)[2], float (&w)[2]) {
    const float src = __fsub_rn(__fdiv_rn(__fmul_rn(float(index) + 0.5f, float(kTable)),
                                          float(size)),
                                0.5f);
    const float f = floorf(src);
    const int t0 = int(f);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        tap[i] = min(max(t0 + i, 0), kTable - 1);
        w[i] = fmaxf(1.f - fabsf(__fsub_rn(__fsub_rn(src, f), float(i))), 0.f);
    }
}

}  // namespace

// Converts n F32 values to BF16 (n a multiple of 4).
extern "C" __global__ void vision_f32_to_bf16(const float* __restrict__ in,
                                              __nv_bfloat16* __restrict__ out, long long n) {
    const long long stride = (long long)gridDim.x * blockDim.x * 4;
    for (long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4; i < n;
         i += stride) {
        const float4 v = *reinterpret_cast<const float4*>(in + i);
        const float a[4] = {v.x, v.y, v.z, v.w};
        store4(out + i, a);
    }
}

// x[r] = LayerNorm_pre(embedding of row r), one block per row:
// - row 0: class_embedding + cls_position_embedding, rounded to BF16;
// - row r > 0: the patch embedding `patch_emb[r]` plus the position table resampled to the
//   patch's cell, rounded to BF16 before and after the add, as the BF16 reference does.
extern "C" __global__ void __launch_bounds__(kRowThreads)
vision_embed(const __nv_bfloat16* __restrict__ patch_emb, const __nv_bfloat16* __restrict__ table,
             const __nv_bfloat16* __restrict__ cls, const __nv_bfloat16* __restrict__ cls_pos,
             const __nv_bfloat16* __restrict__ ln_w, const __nv_bfloat16* __restrict__ ln_b,
             int grid_h, int grid_w, float eps, float* __restrict__ x) {
    __shared__ float shared[kRowThreads / 32];
    const int r = blockIdx.x;
    const int col = threadIdx.x * kPerThread;
    float v[4];
    if (r == 0) {
        float a[4], b[4];
        load4(cls + col, a);
        load4(cls_pos + col, b);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = bf(rb(a[i] + b[i]));
    } else {
        int row, column;
        patch_cell(r - 1, grid_h, grid_w, row, column);
        int ht[2], wt[2];
        float hw[2], ww[2];
        axis_taps(row, grid_h, ht, hw);
        axis_taps(column, grid_w, wt, ww);
        float pos[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float weight = __fmul_rn(hw[i], ww[j]);
                float t[4];
                load4(table + size_t(ht[i] * kTable + wt[j]) * kDim + col, t);
#pragma unroll
                for (int c = 0; c < 4; ++c) pos[c] = __fadd_rn(pos[c], __fmul_rn(t[c], weight));
            }
        }
        float e[4];
        load4(patch_emb + size_t(r) * kDim + col, e);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = bf(rb(e[i] + bf(rb(pos[i]))));
    }
    layer_norm(v, ln_w, ln_b, col, eps, shared);
    *reinterpret_cast<float4*>(x + size_t(r) * kDim + col) = make_float4(v[0], v[1], v[2], v[3]);
}

// One block per row. With `delta`, first updates the residual
// x += gamma * (delta + bias), with delta a BF16 product output. Then writes
// out = BF16(LayerNorm(x) * ln_w + ln_b), or BF16(x) when `ln_w` is null.
// `delta` and `out` may be the same buffer: each thread reads its columns before it
// writes them.
extern "C" __global__ void __launch_bounds__(kRowThreads)
vision_residual_norm(float* __restrict__ x, const __nv_bfloat16* delta,
                     const __nv_bfloat16* __restrict__ bias, const __nv_bfloat16* __restrict__ gamma,
                     const __nv_bfloat16* __restrict__ ln_w, const __nv_bfloat16* __restrict__ ln_b,
                     float eps, __nv_bfloat16* out) {
    __shared__ float shared[kRowThreads / 32];
    const size_t base = size_t(blockIdx.x) * kDim + threadIdx.x * kPerThread;
    const int col = threadIdx.x * kPerThread;
    const float4 x4 = *reinterpret_cast<const float4*>(x + base);
    float v[4] = {x4.x, x4.y, x4.z, x4.w};
    if (delta != nullptr) {
        float d[4], b[4], g[4];
        load4(delta + base, d);
        load4(bias + col, b);
        load4(gamma + col, g);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] += g[i] * (d[i] + b[i]);
        *reinterpret_cast<float4*>(x + base) = make_float4(v[0], v[1], v[2], v[3]);
    }
    if (ln_w != nullptr) layer_norm(v, ln_w, ln_b, col, eps, shared);
    store4(out + base, v);
}

// Adds the projections' biases to q, k and v in place and applies the 2D rotary embedding
// to q and k, one block per row. Row 0 (the class token) sits at position (0, 0) and patch
// cells at (row + 1, col + 1). Channel pair i of a head rotates by the column position
// times inv_freq[i] for i < 16 and by the row position times inv_freq[i - 16] after, with
// inv_freq[j] = 10000^(-j / 16) and pair (x0, x1) going to (x0 cos - x1 sin,
// x1 cos + x0 sin), in F32. Queries are then scaled by 1/8, the attention scale, which is
// exact in BF16.
extern "C" __global__ void __launch_bounds__(kRowThreads)
vision_qkv_rope(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
                __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ q_bias,
                const __nv_bfloat16* __restrict__ k_bias, const __nv_bfloat16* __restrict__ v_bias,
                int grid_h, int grid_w) {
    const int r = blockIdx.x;
    const int col = threadIdx.x * kPerThread;
    const size_t base = size_t(r) * kDim + col;
    int pos_h = 0, pos_w = 0;
    if (r > 0) {
        patch_cell(r - 1, grid_h, grid_w, pos_h, pos_w);
        ++pos_h;
        ++pos_w;
    }
    // The two pairs of this thread's columns.
    float cs[2], sn[2];
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int pair = (col % kHeadDim) / 2 + p;
        const int j = pair % 16;
        const float inv_freq = 1.f / powf(10000.f, float(2 * j) / 32.f);
        const float angle = __fmul_rn(float(pair < 16 ? pos_w : pos_h), inv_freq);
        sincosf(angle, &sn[p], &cs[p]);
    }
    float a[4], b[4];
    // Queries, then keys.
#pragma unroll
    for (int which = 0; which < 2; ++which) {
        __nv_bfloat16* t = which == 0 ? q : k;
        load4(t + base, a);
        load4((which == 0 ? q_bias : k_bias) + col, b);
        float out[4];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const float x0 = bf(rb(a[2 * p] + b[2 * p]));
            const float x1 = bf(rb(a[2 * p + 1] + b[2 * p + 1]));
            out[2 * p] = __fadd_rn(__fmul_rn(x0, cs[p]), __fmul_rn(-x1, sn[p]));
            out[2 * p + 1] = __fadd_rn(__fmul_rn(x1, cs[p]), __fmul_rn(x0, sn[p]));
        }
        if (which == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) out[i] = bf(rb(out[i])) * 0.125f;
        }
        store4(t + base, out);
    }
    load4(v + base, a);
    load4(v_bias + col, b);
#pragma unroll
    for (int i = 0; i < 4; ++i) a[i] += b[i];
    store4(v + base, a);
}

// h = BF16(gelu(h + bias)) with the exact (erf) GELU, `cols` columns per row, `bias` per
// column or null. n is a multiple of 4, and so is cols.
extern "C" __global__ void vision_gelu(__nv_bfloat16* __restrict__ h,
                                       const __nv_bfloat16* __restrict__ bias, int cols,
                                       long long n) {
    const long long stride = (long long)gridDim.x * blockDim.x * 4;
    for (long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 4; i < n;
         i += stride) {
        float v[4];
        load4(h + i, v);
        if (bias != nullptr) {
            float b[4];
            load4(bias + i % cols, b);
#pragma unroll
            for (int c = 0; c < 4; ++c) v[c] += b[c];
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) v[c] = 0.5f * v[c] * (1.f + erff(v[c] * 0.70710678118654752f));
        store4(h + i, v);
    }
}

// ---------------------------------------------------------------------------------------
// Bidirectional attention over the S rows of one image, 16 heads of 64, flash style.
//
// q, k and v are [S, 1024] BF16 with head h in columns 64h..64h+63; q is already scaled.
// out is [S, 1024]. A block takes 64 query rows of one head, 16 per warp, and walks the
// keys in tiles of 64: K and V tiles go to shared memory with cp.async, two stages deep;
// each warp computes its 16 x 64 scores with mma.sync m16n8k16, keeps an F32 running
// maximum and sum per row, and multiplies the BF16-rounded probabilities with V. Keys past
// S load as zeros and score minus infinity. Memory is the q, k, v and out buffers only.

namespace attn {

constexpr int kQRows = 64;
constexpr int kKeys = 64;
constexpr int kThreads = 128;
constexpr float kLog2e = 1.4426950408889634f;

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src),
                 "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                    uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// Element offset of 16-byte chunk `chunk` of row `row` in a 64 x 64 tile, XOR-swizzled so
// the 8 rows one ldmatrix reads fall in distinct banks.
__device__ __forceinline__ int tile_offset(int row, int chunk) {
    return row * kHeadDim + ((chunk ^ (row & 7)) * 8);
}

// Starts copying key tile `tile` of K and V for `head` into stage buffers `sk` and `sv`.
__device__ __forceinline__ void load_tile(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                          __nv_bfloat16* sk, __nv_bfloat16* sv, int tile,
                                          int head, int S) {
#pragma unroll
    for (int i = 0; i < kKeys * 8 / kThreads; ++i) {
        const int idx = threadIdx.x + i * kThreads;
        const int row = idx / 8;
        const int chunk = idx % 8;
        const int key = tile * kKeys + row;
        const bool valid = key < S;
        const size_t src = size_t(valid ? key : 0) * kDim + head * kHeadDim + chunk * 8;
        cp_async16(smem_addr(sk + tile_offset(row, chunk)), k + src, valid);
        cp_async16(smem_addr(sv + tile_offset(row, chunk)), v + src, valid);
    }
    cp_async_commit();
}

}  // namespace attn

extern "C" __global__ void __launch_bounds__(attn::kThreads)
vision_attention(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                 const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ out, int S) {
    using namespace attn;
    __shared__ alignas(128) __nv_bfloat16 sk[2][kKeys * kHeadDim];
    __shared__ alignas(128) __nv_bfloat16 sv[2][kKeys * kHeadDim];

    const int head = blockIdx.y;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int r0 = blockIdx.x * kQRows + warp * 16 + g;
    const int r1 = r0 + 8;
    const int tiles = (S + kKeys - 1) / kKeys;

    load_tile(k, v, sk[0], sv[0], 0, head, S);

    // The warp's 16 query rows as A fragments of the 4 k steps of 16 dims.
    uint32_t qa[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        const int d = head * kHeadDim + kk * 16 + 2 * t;
        const uint32_t* q0 = reinterpret_cast<const uint32_t*>(q + size_t(r0) * kDim + d);
        const uint32_t* q1 = reinterpret_cast<const uint32_t*>(q + size_t(r1) * kDim + d);
        qa[kk][0] = r0 < S ? q0[0] : 0u;
        qa[kk][1] = r1 < S ? q1[0] : 0u;
        qa[kk][2] = r0 < S ? q0[4] : 0u;
        qa[kk][3] = r1 < S ? q1[4] : 0u;
    }

    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    float o[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;

    for (int tile = 0; tile < tiles; ++tile) {
        const int stage = tile & 1;
        if (tile + 1 < tiles) {
            load_tile(k, v, sk[stage ^ 1], sv[stage ^ 1], tile + 1, head, S);
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();
        const uint32_t kbase = smem_addr(sk[stage]);
        const uint32_t vbase = smem_addr(sv[stage]);

        float s[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
            for (int kp = 0; kp < 2; ++kp) {
                uint32_t b[4];
                ldmatrix_x4(b, kbase + 2 * tile_offset(n * 8 + (lane & 7), kp * 4 + (lane >> 3)));
                mma(s[n], qa[2 * kp], b[0], b[1]);
                mma(s[n], qa[2 * kp + 1], b[2], b[3]);
            }
        }
        if ((tile + 1) * kKeys > S) {
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int key = tile * kKeys + n * 8 + 2 * t;
                if (key >= S) s[n][0] = s[n][2] = -INFINITY;
                if (key + 1 >= S) s[n][1] = s[n][3] = -INFINITY;
            }
        }

        float mx0 = m0, mx1 = m1;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
            mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
        }
#pragma unroll
        for (int o2 = 1; o2 < 4; o2 <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o2));
        }
        // Every tile holds at least one key below S, so the maxima are finite.
        const float c0 = exp2f((m0 - mx0) * kLog2e);
        const float c1 = exp2f((m1 - mx1) * kLog2e);
        m0 = mx0;
        m1 = mx1;
        l0 *= c0;
        l1 *= c1;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            o[n][0] *= c0;
            o[n][1] *= c0;
            o[n][2] *= c1;
            o[n][3] *= c1;
        }
        const float ms0 = m0 * kLog2e, ms1 = m1 * kLog2e;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            s[n][0] = exp2f(s[n][0] * kLog2e - ms0);
            s[n][1] = exp2f(s[n][1] * kLog2e - ms0);
            s[n][2] = exp2f(s[n][2] * kLog2e - ms1);
            s[n][3] = exp2f(s[n][3] * kLog2e - ms1);
            l0 += s[n][0] + s[n][1];
            l1 += s[n][2] + s[n][3];
        }

#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const uint32_t pa[4] = {pack(s[2 * kk][0], s[2 * kk][1]),
                                    pack(s[2 * kk][2], s[2 * kk][3]),
                                    pack(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                    pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int dp = 0; dp < 4; ++dp) {
                uint32_t b[4];
                const int row = kk * 16 + (lane & 7) + 8 * ((lane >> 3) & 1);
                ldmatrix_x4_trans(b, vbase + 2 * tile_offset(row, dp * 2 + (lane >> 4)));
                mma(o[2 * dp], pa, b[0], b[1]);
                mma(o[2 * dp + 1], pa, b[2], b[3]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int o2 = 1; o2 < 4; o2 <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, o2);
        l1 += __shfl_xor_sync(0xffffffffu, l1, o2);
    }
    const float inv0 = 1.f / l0, inv1 = 1.f / l1;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        const int d = head * kHeadDim + n * 8 + 2 * t;
        if (r0 < S) {
            *reinterpret_cast<uint32_t*>(out + size_t(r0) * kDim + d) =
                pack(o[n][0] * inv0, o[n][1] * inv0);
        }
        if (r1 < S) {
            *reinterpret_cast<uint32_t*>(out + size_t(r1) * kDim + d) =
                pack(o[n][2] * inv1, o[n][3] * inv1);
        }
    }
}
