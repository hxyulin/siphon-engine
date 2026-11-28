// NVFP4 and MXFP8 expert matrix-vector products for decode, and the whole decode expert
// MLP (`w13`, softcapped SwiGLU, `w2`) as one kernel.
//
// Each output value is one weight row dotted with one BF16 input row. The products run at
// DRAM speed, so the layout serves the loads: a warp covers `kRows` whole weight rows,
// `kChunks` 16-byte chunks per lane, and its i-th load reads chunk `lane + 32 * i` of
// them, 512 contiguous bytes.
//
// Numerics: each product of a BF16 input and an E2M1 or E4M3 weight is exact in F32. A
// lane sums the products of each scale group in F32 (16 values for NVFP4, 16 of a group
// of 32 for MXFP8) and multiplies the sum by the group's scale; the lanes of a row then
// add these in F32. NVFP4 multiplies by `global_scale / (448 * 6)`. The value is rounded
// to BF16 once, as MLX's `mx.gather_qmm` does; only the summation order differs from MLX.
//
// The MLP kernel takes `w13`, whose rows interleave gate (even) and up (odd) values. It
// rounds both to BF16, as `softcapped_swiglu_bf16` reads them, and stores
// `softcapped_swiglu(gate, up)`, so its activations equal `expert_matvec` followed by that
// kernel bit for bit.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>

#include "conv_step.cuh"
#include "mlx_ops.cuh"
#include "mxfp6.cuh"
#include "arch.cuh"
#include "pdl.cuh"

namespace {

// 16-byte weight chunks per lane.
constexpr int kChunks = 9;
// Warps per block.
constexpr int kWarps = 4;
// `F8E4M3_MAX * F4E2M1_MAX` in MLX's `inv_scale_enc`.
constexpr float kNvfp4ScaleEnc = 448.f * 6.f;

// Expert weights are read once per token, so they stream past L1 and leave L2 first.
__device__ __forceinline__ uint64_t evict_first_policy() {
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
    return policy;
}

__device__ __forceinline__ uint4 load_stream(const void* p, uint64_t policy) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p), "l"(policy));
    return v;
}

__device__ __forceinline__ uint2 load_stream_u64(const void* p, uint64_t policy) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(p), "l"(policy));
    return v;
}

__device__ __forceinline__ uint32_t load_stream_u16(const void* p, uint64_t policy) {
    uint16_t v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.u16 %0, [%1], %2;"
                 : "=h"(v)
                 : "l"(p), "l"(policy));
    return v;
}

__device__ __forceinline__ uint32_t load_stream_u8(const void* p, uint64_t policy) {
    uint16_t v;
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.u8 %0, [%1], %2;"
                 : "=h"(v)
                 : "l"(p), "l"(policy));
    return v;
}

// The two BF16 values of a 32-bit word, low half first.
__device__ __forceinline__ float bf16_lo(uint32_t pair) { return __uint_as_float(pair << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t pair) { return __uint_as_float(pair & 0xffff0000u); }

// Two E2M1 values, low nibble first, decoded by the hardware converter. Every E2M1 value is
// exact in half precision.
__device__ __forceinline__ float2 e2m1x2(uint32_t bits) {
#if SIPHON_SM89
    __half2_raw h;
    const uint32_t v = e2m1x2_to_f16x2_sm89(bits);
    h.x = uint16_t(v);
    h.y = uint16_t(v >> 16);
    return __half22float2(__half2(h));
#else
    return __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(uint8_t(bits), __NV_E2M1)));
#endif
}

// Two E4M3 values, low byte first.
__device__ __forceinline__ float2 e4m3x2(uint32_t bits) {
    return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(uint16_t(bits), __NV_E4M3)));
}

__device__ __forceinline__ float e4m3(uint32_t bits) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(uint8_t(bits), __NV_E4M3)));
}

// MLX's `fp8_e8m0` to float: 2^(bits - 127), with 0 mapped to 2^-127.
__device__ __forceinline__ float e8m0(uint32_t bits) {
    return __uint_as_float(bits == 0 ? 0x400000u : bits << 23);
}

// Two E4M3 block scales reconstructed from one packed byte of 4-bit offsets (`off_lo | off_hi<<4`)
// subtracted from the row's E4M3 base scale (`base >= off_i` by construction).
__device__ __forceinline__ uint32_t unpack_scale4_pair(uint32_t packed_byte, uint32_t base) {
    const uint32_t s0 = base - (packed_byte & 0xfu);
    const uint32_t s1 = base - ((packed_byte >> 4) & 0xfu);
    return s0 | (s1 << 8);
}

// Sum of 8 BF16 inputs (`x`, 4 words) times 8 E2M1 weights (`w`, 4 bytes).
__device__ __forceinline__ float dot8_e2m1(const uint32_t* x, uint32_t w, float acc) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const float2 wv = e2m1x2(w >> (8 * b));
        acc = fmaf(bf16_lo(x[b]), wv.x, acc);
        acc = fmaf(bf16_hi(x[b]), wv.y, acc);
    }
    return acc;
}

// Sum of 4 BF16 inputs (`x`, 2 words) times 4 E4M3 weights (`w`, 4 bytes).
__device__ __forceinline__ float dot4_e4m3(const uint32_t* x, uint32_t w, float acc) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const float2 wv = e4m3x2(w >> (16 * h));
        acc = fmaf(bf16_lo(x[h]), wv.x, acc);
        acc = fmaf(bf16_hi(x[h]), wv.y, acc);
    }
    return acc;
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Rows, chunks and scale groups of a projection with input width kK.
template <bool kFp4, int kK, bool kScale4 = false>
struct Geometry {
    static constexpr int kValues = kFp4 ? 32 : 16;  // per chunk
    static constexpr int kRowChunks = kK / kValues;
    static constexpr int kRows = 32 * kChunks / kRowChunks;  // per warp
    static constexpr int kBlockRows = kWarps * kRows;
    static constexpr int kGroup = kFp4 ? 16 : 32;
    static constexpr int kRowBytes = kFp4 ? kK / 2 : kK;
    static constexpr int kRowScales = kK / kGroup;
    static constexpr int kRowScaleBytes = kScale4 ? kRowScales / 2 : kRowScales;
    static_assert(kRows * kRowChunks == 32 * kChunks, "a warp covers whole rows");
    static_assert(!kScale4 || (kFp4 && (kRows == 4 || kRows == 8)),
                  "4-bit offset scales are NVFP4 with 4 or 8 rows per warp");
    static_assert(kScale4 ? kRowScaleBytes == kRowChunks
                          : (kFp4 ? kRowScales == 2 * kRowChunks : 2 * kRowScales == kRowChunks),
                  "chunk scales are contiguous across rows");
};

// One warp's `kRows` weight rows of one expert: chunk `lane + 32 * i` is row
// `chunk / kRowChunks`, position `chunk % kRowChunks` in it.
//
// The loads are volatile asm, which nvcc keeps in order with `pdl_wait` and the flag wait
// but interleaves with the arithmetic around them. So the kernels load only after their
// waits: loads placed before a wait all issue at once, and issuing a warp's nine chunks
// together took 21 us instead of 16.5 for an NVFP4 `w13` on an RTX 5090. More loads in
// flight lower DRAM throughput here.
template <bool kFp4, int kK, bool kScale4 = false>
struct WarpRows {
    using G = Geometry<kFp4, kK, kScale4>;
    uint4 w[kChunks];
    uint32_t s[kChunks];
    uint32_t base_lo;
    uint32_t base_hi;
    const uint8_t* wr;
    const uint8_t* sr;
    uint64_t policy;
    bool known;

    // Rows `row0 ..` of `expert`, whose rows number n. An unknown expert loads zeros.
    __device__ __forceinline__ WarpRows(const uint8_t* weight, const uint8_t* scales, int expert,
                                        bool known_expert, int n, int row0,
                                        const uint8_t* scale_base = nullptr)
        : base_lo(0u), base_hi(0u), known(known_expert) {
        policy = evict_first_policy();
        const size_t first_row = size_t(known ? expert : 0) * n + row0;
        wr = weight + first_row * G::kRowBytes;
        sr = scales + first_row * G::kRowScaleBytes;
        if constexpr (kScale4) {
            if (known) {
                base_lo = *reinterpret_cast<const uint32_t*>(scale_base + first_row);
                if constexpr (G::kRows == 8) {
                    base_hi = *reinterpret_cast<const uint32_t*>(scale_base + first_row + 4);
                }
            }
        }
    }

    __device__ __forceinline__ void load(int i, int lane) {
        const int chunk = lane + 32 * i;
        const uint8_t* sp = sr + (kScale4 ? chunk : (kFp4 ? 2 * chunk : chunk / 2));
        w[i] = known ? load_stream(wr + 16 * chunk, policy) : make_uint4(0, 0, 0, 0);
        s[i] = !known ? 0
                      : (kFp4 && !kScale4) ? load_stream_u16(sp, policy) : load_stream_u8(sp, policy);
    }

    // The rows' dot products with the BF16 input `x` (kK values), before the global scale:
    // lane r < kRows returns row r.
    __device__ __forceinline__ float dot(const __nv_bfloat16* x, int lane) const {
        const uint4* xr = reinterpret_cast<const uint4*>(x);
        float acc[G::kRows];
#pragma unroll
        for (int r = 0; r < G::kRows; ++r) {
            acc[r] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < kChunks; ++i) {
            const int chunk = lane + 32 * i;
            const int p = chunk % G::kRowChunks;
            uint32_t sc = s[i];
            if constexpr (kScale4) {
                const int r = chunk / G::kRowChunks;
                const uint32_t rb =
                    (r < 4 ? (base_lo >> (8 * r)) : (base_hi >> (8 * (r - 4)))) & 0xffu;
                sc = unpack_scale4_pair(sc, rb);
            }
            float sum = 0.f;
#pragma unroll
            for (int g = 0; g < (kFp4 ? 2 : 1); ++g) {
                // 16 inputs: one NVFP4 scale group (8 weight bytes), half an MXFP8 group
                // (16 weight bytes).
                const uint4 xa = xr[p * (kFp4 ? 4 : 2) + 2 * g];
                const uint4 xb = xr[p * (kFp4 ? 4 : 2) + 2 * g + 1];
                const uint32_t xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                float group = 0.f;
                if constexpr (kFp4) {
                    group = dot8_e2m1(xv, word(w[i], 2 * g), group);
                    group = dot8_e2m1(xv + 4, word(w[i], 2 * g + 1), group);
                    sum = fmaf(e4m3(sc >> (8 * g)), group, sum);
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        group = dot4_e4m3(xv + 2 * j, word(w[i], j), group);
                    }
                    sum = e8m0(sc & 0xff) * group;
                }
            }
            // Only the rows some lane reaches at step i.
#pragma unroll
            for (int r = 32 * i / G::kRowChunks; r <= (32 * i + 31) / G::kRowChunks; ++r) {
                acc[r] += chunk / G::kRowChunks == r ? sum : 0.f;
            }
        }
        float result = 0.f;
#pragma unroll
        for (int r = 0; r < G::kRows; ++r) {
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
            }
            result = lane == r ? acc[r] : result;
        }
        return result;
    }
};

__device__ __forceinline__ float output_scale(bool fp4, const float* global_scale, int expert) {
    return fp4 ? __fdiv_rn(global_scale[expert], kNvfp4ScaleEnc) : 1.f;
}

__device__ __forceinline__ uint32_t load_relaxed(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

#if SIPHON_SM89
__device__ __forceinline__ uint32_t load_acquire(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void red_inc_release(uint32_t* p) {
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory");
}
#endif

__device__ __forceinline__ uint32_t atom_inc_acq_rel(uint32_t* p) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(p) : "memory");
    return old;
}

constexpr float kNan = __builtin_nanf("");

// The end of the MoE block for one row, as moe_conv_residual_2304 in conv.cu computes it,
// bit for bit: the combine of the 8 expert outputs with `weights`, the post-MLP conv, and
// hidden = bf16(residual + conv).
struct MoeConv {
    const float* weights;
    const __nv_bfloat16* __restrict__ conv_weight;
    float* conv_state;
    const float* residual;
    __nv_bfloat16* hidden;
};

// One channel's inputs to the end of the MoE block that do not depend on this launch,
// loaded before the expert outputs are ready.
struct MoeConvInputs {
    float weights[8];
    float window[conv_step::kKernel - 1];
    uint2 taps;
    float residual;

    MoeConvInputs() = default;
    __device__ __forceinline__ MoeConvInputs(const MoeConv& m, int c, int n) {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            weights[k] = m.weights[k];
        }
#pragma unroll
        for (int k = 0; k < conv_step::kKernel - 1; ++k) {
            window[k] = m.conv_state[size_t(k) * n + c];
        }
        taps = *reinterpret_cast<const uint2*>(m.conv_weight + size_t(c) * conv_step::kKernel);
        residual = m.residual[c];
    }
};

// Channel c from the expert outputs `down` ([8, n] BF16), which other blocks of this launch
// wrote, so they are read from L2. The conv follows conv_step::short_conv.
__device__ __forceinline__ void moe_conv(const MoeConv& m, const MoeConvInputs& in,
                                         const __nv_bfloat16* down, int n, int c) {
    float vals[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        vals[k] = __bfloat162float(__ldcg(down + size_t(k) * n + c));
    }
    float total = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float wk = in.weights[k];
        total = wk != 0.f ? __fadd_rn(total, __fmul_rn(vals[k], wk)) : total;
    }
    const float x = conv_step::round_bf16(total);
    const auto* taps = reinterpret_cast<const __nv_bfloat16*>(&in.taps);
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < conv_step::kKernel; ++k) {
        const float w = k < conv_step::kKernel - 1 ? in.window[k] : x;
        acc = __fadd_rn(acc, __fmul_rn(conv_step::round_bf16(w), __bfloat162float(taps[k])));
    }
    const float conv = conv_step::round_bf16(acc);
#pragma unroll
    for (int k = 0; k < conv_step::kKernel - 1; ++k) {
        m.conv_state[size_t(k) * n + c] = k + 1 < conv_step::kKernel - 1 ? in.window[k + 1] : x;
    }
    const float y = __bfloat162float(
        __float2bfloat16(__fdiv_rn(__fadd_rn(conv, x), conv_step::kShortDivisor)));
    m.hidden[c] = __float2bfloat16(__fadd_rn(in.residual, y));
}

// One block covers `kBlockRows` output rows of one (row, slot) pair, `blockIdx.y`.
template <bool kFp4, int kK, bool kScale4 = false>
__device__ __forceinline__ void expert_matvec(const __nv_bfloat16* x, const int32_t* indices,
                                              const uint8_t* weight, const uint8_t* scales,
                                              const uint8_t* scale_base, const float* global_scale,
                                              __nv_bfloat16* out, int n, int inputs_per_row,
                                              int experts) {
    using G = Geometry<kFp4, kK, kScale4>;
    const int lane = threadIdx.x % 32;
    const int row0 = blockIdx.x * G::kBlockRows + threadIdx.x / 32 * G::kRows;
    const unsigned pair = blockIdx.y;
    pdl_wait();
    pdl_trigger();
    const int expert = indices[pair];
    const bool known = expert >= 0 && expert < experts;
    if (!known) {
        if (lane < G::kRows) {
            out[size_t(pair) * n + row0 + lane] = __float2bfloat16(kNan);
        }
        return;
    }
    WarpRows<kFp4, kK, kScale4> rows(weight, scales, expert, true, n, row0, scale_base);
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
        rows.load(i, lane);
    }
    const unsigned input = inputs_per_row == 1 ? pair / 8 : pair;
    float result = rows.dot(x + size_t(input) * kK, lane);
    if (lane < G::kRows) {
        result *= output_scale(kFp4, global_scale, expert);
        out[size_t(pair) * n + row0 + lane] = __float2bfloat16(result);
    }
}

// The decode expert MLP in one kernel: act = softcapped_swiglu(x @ w13) per pair, then
// out = act @ w2. Tasks `0 .. pairs * kTiles13` are `w13` row tiles and the rest `w2` row
// tiles, both ordered by pair, and block b runs tasks b, b + gridDim.x, and so on. A
// `w13` task adds 1 to its pair's count `sync[1 + pair]` after storing its activations;
// a `w2` task waits until the count reaches kTiles13, then streams its weights and reads
// the activations. With fewer blocks than tasks, the first pairs' `w13` tiles end
// early and their `w2` tiles stream while later `w13` tiles still run.
//
// A block reaches each of its `w13` tasks without waiting, since all its earlier tasks
// are `w13` tasks too. So as long as every block of the grid can be resident at once,
// which the launcher ensures, every count completes. The last block to finish zeroes
// `sync` for the next launch.
template <bool kFp4, int kK13, int kK2, bool kScale4 = false>
__device__ __forceinline__ void expert_mlp(const __nv_bfloat16* x, const int32_t* indices,
                                           const uint8_t* w13, const uint8_t* s13,
                                           const uint8_t* s13_base, const float* g13,
                                           const uint8_t* w2, const uint8_t* s2,
                                           const uint8_t* s2_base, const float* g2,
                                           __nv_bfloat16* act, __nv_bfloat16* out, int n2,
                                           int pairs, int experts, float gate_softcap,
                                           float up_softcap, uint32_t* sync, const MoeConv& conv) {
    using G13 = Geometry<kFp4, kK13, kScale4>;
    using G2 = Geometry<kFp4, kK2, kScale4>;
    constexpr int kN13 = 2 * kK2;
    constexpr int kTiles13 = kN13 / G13::kBlockRows;
    static_assert(kN13 % G13::kBlockRows == 0 && G13::kRows % 2 == 0,
                  "gate and up rows share a warp");
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int tiles2 = n2 / G2::kBlockRows;
    const unsigned tasks13 = unsigned(pairs) * kTiles13;
    const unsigned tasks = tasks13 + unsigned(pairs) * tiles2;

    pdl_wait();
    for (unsigned task = blockIdx.x; task < tasks; task += gridDim.x) {
        if (task < tasks13) {
            const unsigned pair = task / kTiles13;
            const int row0 = task % kTiles13 * G13::kBlockRows + warp * G13::kRows;
            const int expert = indices[pair];
            const bool known = expert >= 0 && expert < experts;
            WarpRows<kFp4, kK13, kScale4> rows(w13, s13, expert, known, kN13, row0, s13_base);
#pragma unroll
            for (int i = 0; i < kChunks; ++i) {
                rows.load(i, lane);
            }
            float result = rows.dot(x + size_t(pair / 8) * kK13, lane);
            if (known) {
                result *= output_scale(kFp4, g13, expert);
            }
            const float up = __shfl_down_sync(0xffffffffu, result, 1);
            if (lane < G13::kRows && lane % 2 == 0) {
                const float gate_bf16 = __bfloat162float(__float2bfloat16(result));
                const float up_bf16 = __bfloat162float(__float2bfloat16(up));
                act[size_t(pair) * kK2 + (row0 + lane) / 2] =
                    known ? softcapped_swiglu(gate_bf16, up_bf16, gate_softcap, up_softcap)
                          : __float2bfloat16(kNan);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
#if SIPHON_SM89
                red_inc_release(&sync[1 + pair]);
#else
                __threadfence();
                atomicAdd(&sync[1 + pair], 1u);
#endif
            }
        } else {
            const unsigned pair = (task - tasks13) / tiles2;
            const unsigned tile = (task - tasks13) % tiles2;
            const int row0 = int(tile) * G2::kBlockRows + warp * G2::kRows;
            const int expert = indices[pair];
            const bool known = expert >= 0 && expert < experts;
            if (threadIdx.x == 0) {
                while (load_relaxed(&sync[1 + pair]) < kTiles13) {
                    __nanosleep(32);
                }
#if SIPHON_SM89
                // One acquire load after the relaxed spin orders the block's activation loads
                // after the producers' release reductions without draining L1 during polling.
                (void)load_acquire(&sync[1 + pair]);
#else
                // With the barrier below, orders the block's activation loads after the
                // count, and so after the stores the count covers.
                __threadfence();
#endif
            }
            __syncthreads();
            // Loading weights before the wait was slower: 27.0 us instead of 24.6 per NVFP4
            // layer visit with all nine chunks.
            WarpRows<kFp4, kK2, kScale4> rows(w2, s2, expert, known, n2, row0, s2_base);
#pragma unroll
            for (int i = 0; i < kChunks; ++i) {
                rows.load(i, lane);
            }
            float result = rows.dot(act + size_t(pair) * kK2, lane);
            if (lane < G2::kRows) {
                result = known ? result * output_scale(kFp4, g2, expert) : kNan;
                out[size_t(pair) * n2 + row0 + lane] = __float2bfloat16(result);
            }
            if (conv.hidden != nullptr) {
                // The last pair's task of each row tile finishes the tile's channels once
                // the other pairs' tasks, all earlier in task order, have counted in. Its
                // other inputs load first, so only the expert outputs wait for the count.
                uint32_t* count = sync + 1 + pairs + tile;
                const bool finisher = pair + 1 == unsigned(pairs);
                const int channel = int(tile) * G2::kBlockRows + threadIdx.x;
                const MoeConvInputs inputs = finisher && threadIdx.x < G2::kBlockRows
                                                 ? MoeConvInputs(conv, channel, n2)
                                                 : MoeConvInputs();
                __syncthreads();
                if (threadIdx.x == 0) {
                    if (!finisher) {
#if SIPHON_SM89
                        red_inc_release(count);
#else
                        __threadfence();
                        atomicAdd(count, 1u);
#endif
                    } else {
                        while (load_relaxed(count) < unsigned(pairs) - 1) {
                            __nanosleep(32);
                        }
#if SIPHON_SM89
                        (void)load_acquire(count);
#else
                        __threadfence();
#endif
                        *count = 0;
                    }
                }
                if (finisher) {
                    __syncthreads();
                    if (threadIdx.x < G2::kBlockRows) {
                        moe_conv(conv, inputs, out, n2, channel);
                    }
                }
            }
        }
    }
    // Release the next PDL kernel only after the task loop so it does not compete for SM
    // slots or DRAM bandwidth while expert weights stream, while still overlapping with
    // counter cleanup and CTA teardown.
    pdl_trigger();
    if (warp == 0) {
        uint32_t last = 0;
        if (lane == 0) {
            last = (atom_inc_acq_rel(&sync[0]) == gridDim.x - 1) ? 1u : 0u;
        }
        if (__shfl_sync(0xffffffffu, last, 0)) {
            for (int p = lane; p <= pairs; p += 32) {
                sync[p] = 0;
            }
        }
    }
}

}  // namespace

// `expert_matvec_<format>_<k>`:
// - `x`: BF16 `[rows, inputs_per_row, k]`, where `inputs_per_row` is 1 (one input for all
//   8 slots) or 8 (one per slot);
// - `indices`: `[rows, 8]` expert ids;
// - `weight`, `scales`: raw checkpoint bytes, `[experts, n, k / 2]` E2M1 and
//   `[experts, n, k / 16]` E4M3 for NVFP4, `[experts, n, k]` E4M3 and `[experts, n, k / 32]`
//   E8M0 for MXFP8;
// - `global_scale`: F32 `[experts]` for NVFP4, unused by MXFP8;
// - `out`: BF16 `[rows, 8, n]`. An expert id outside `[0, experts)` gives NaN.
// The grid is (n / kBlockRows, rows * 8) blocks of `kWarps` warps; n must be a multiple
// of kBlockRows.
#define EXPERT_MATVEC(name, fp4, k)                                                                \
    extern "C" __global__ void __launch_bounds__(kWarps * 32, 7)                                    \
        name(const __nv_bfloat16* x, const int32_t* indices, const uint8_t* weight,                \
             const uint8_t* scales, const float* global_scale, __nv_bfloat16* out, int n,          \
             int inputs_per_row, int experts) {                                                    \
        expert_matvec<fp4, k, false>(x, indices, weight, scales, nullptr, global_scale, out, n,    \
                                     inputs_per_row, experts);                                     \
    }

#define EXPERT_MATVEC_S4(name, k)                                                                  \
    extern "C" __global__ void __launch_bounds__(kWarps * 32, 7)                                    \
        name(const __nv_bfloat16* x, const int32_t* indices, const uint8_t* weight,                \
             const uint8_t* scales, const uint8_t* scale_base, const float* global_scale,          \
             __nv_bfloat16* out, int n, int inputs_per_row, int experts) {                         \
        expert_matvec<true, k, true>(x, indices, weight, scales, scale_base, global_scale, out, n, \
                                     inputs_per_row, experts);                                     \
    }

EXPERT_MATVEC(expert_matvec_nvfp4_2304, true, 2304)
EXPERT_MATVEC(expert_matvec_nvfp4_1152, true, 1152)
EXPERT_MATVEC(expert_matvec_mxfp8_2304, false, 2304)
EXPERT_MATVEC(expert_matvec_mxfp8_1152, false, 1152)
EXPERT_MATVEC_S4(expert_matvec_nvfp4_s4_2304, 2304)
EXPERT_MATVEC_S4(expert_matvec_nvfp4_s4_1152, 1152)

// `expert_mlp_<format>`, for `w13` of 2304 interleaved rows and input width 2304, and `w2`
// of n2 rows and input width 1152, with the weights as for `expert_matvec`:
// - `x`: BF16 `[pairs / 8, 2304]`; `indices`: `[pairs / 8, 8]`;
// - `act`: BF16 `[pairs, 1152]`, written; `out`: BF16 `[pairs, n2]`;
// - `sync`: `1 + pairs` zeros, left zero; with `hidden`, `1 + pairs + n2 / kBlockRows of w2`.
// With a non-null `hidden`, which needs pairs = 8 (one row) and n2 = 2304, the kernel also
// ends the MoE block as moe_conv_residual_2304 does, from `weights` ([8] F32),
// `conv_weight`, `conv_state`, `residual` ([2304] F32) into `hidden` ([2304] BF16).
// Blocks have `kWarps` warps. The grid may have any size up to the number of tasks,
// `pairs * (2304 / kBlockRows of w13 + n2 / kBlockRows of w2)`, but all its blocks must
// fit on the GPU at once.
#define EXPERT_MLP(name, fp4, min_blocks)                                                          \
    extern "C" __global__ void __launch_bounds__(kWarps * 32, min_blocks)                          \
        name(const __nv_bfloat16* x, const int32_t* indices, const uint8_t* w13,                   \
             const uint8_t* s13, const float* g13, const uint8_t* w2, const uint8_t* s2,           \
             const float* g2, __nv_bfloat16* act, __nv_bfloat16* out, int n2, int pairs,           \
             int experts, float gate_softcap, float up_softcap, uint32_t* sync,                    \
             const float* weights, const __nv_bfloat16* __restrict__ conv_weight,                  \
             float* conv_state, const float* residual, __nv_bfloat16* hidden) {                    \
        const MoeConv conv{weights, conv_weight, conv_state, residual, hidden};                    \
        expert_mlp<fp4, 2304, 1152, false>(x, indices, w13, s13, nullptr, g13, w2, s2, nullptr,    \
                                           g2, act, out, n2, pairs, experts, gate_softcap,         \
                                           up_softcap, sync, conv);                                \
    }

EXPERT_MLP(expert_mlp_nvfp4, true, 5)
EXPERT_MLP(expert_mlp_mxfp8, false, 5)

extern "C" __global__ void __launch_bounds__(kWarps * 32, 5)
    expert_mlp_nvfp4_s4(const __nv_bfloat16* x, const int32_t* indices, const uint8_t* w13,
                        const uint8_t* s13, const uint8_t* s13_base, const float* g13,
                        const uint8_t* w2, const uint8_t* s2, const uint8_t* s2_base,
                        const float* g2, __nv_bfloat16* act, __nv_bfloat16* out, int n2, int pairs,
                        int experts, float gate_softcap, float up_softcap, uint32_t* sync,
                        const float* weights, const __nv_bfloat16* __restrict__ conv_weight,
                        float* conv_state, const float* residual, __nv_bfloat16* hidden) {
    const MoeConv conv{weights, conv_weight, conv_state, residual, hidden};
    expert_mlp<true, 2304, 1152, true>(x, indices, w13, s13, s13_base, g13, w2, s2, s2_base, g2,
                                       act, out, n2, pairs, experts, gate_softcap, up_softcap, sync,
                                       conv);
}

namespace {

__device__ __forceinline__ uint32_t nearest_e2m1_mag(float mag) {
    return uint32_t(mag >= 0.25f) + uint32_t(mag >= 0.75f) + uint32_t(mag >= 1.25f) +
           uint32_t(mag >= 1.75f) + uint32_t(mag >= 2.50f) + uint32_t(mag >= 3.50f) +
           uint32_t(mag >= 5.00f);
}

__device__ __forceinline__ void reround_nvfp4_block(uint8_t* w_blk, uint32_t s_orig,
                                                    uint32_t s_new) {
    constexpr float kE2m1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float so = e4m3(s_orig);
    const float sn = e4m3(s_new);
    const float ratio = sn > 0.f ? __fdiv_rn(so, sn) : 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t byte = w_blk[i];
        const uint32_t n0 = byte & 0xfu;
        const uint32_t n1 = byte >> 4;
        const uint32_t q0 = nearest_e2m1_mag(kE2m1[n0 & 7u] * ratio);
        const uint32_t q1 = nearest_e2m1_mag(kE2m1[n1 & 7u] * ratio);
        const uint32_t r0 = q0 == 0u ? 0u : ((n0 & 8u) | q0);
        const uint32_t r1 = q1 == 0u ? 0u : ((n1 & 8u) | q1);
        w_blk[i] = uint8_t(r0 | (r1 << 4));
    }
}

}  // namespace

// Packs NVFP4 8-bit E4M3 scales (`[rows, blocks_per_row]`) into per-row 8-bit E4M3 bases
// (`[rows]`) plus packed 4-bit offsets (`[rows, blocks_per_row / 2]`, `off = base - s`),
// clamping tiny blocks below `base - 15` up to `base - 15` and re-rounding their 16 E2M1
// weight nibbles in-place inside `weight` (`[rows, blocks_per_row * 8]`).
// Grid: ceil(rows / 8) blocks of 256 threads (8 warps, one warp per row).
extern "C" __global__ void __launch_bounds__(256)
    pack_nvfp4_scales_4bit(uint8_t* __restrict__ weight, const uint8_t* __restrict__ scales_in,
                           uint8_t* __restrict__ scales_out, uint8_t* __restrict__ scale_base,
                           int rows, int blocks_per_row) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = int(blockIdx.x) * 8 + warp;
    if (row >= rows) {
        return;
    }
    const uint8_t* s_row = scales_in + size_t(row) * blocks_per_row;
    uint32_t rmax = 0u;
    for (int b = lane; b < blocks_per_row; b += 32) {
        const uint32_t s = min(uint32_t(s_row[b]) & 0x7fu, 0x7eu);
        rmax = max(rmax, s);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        rmax = max(rmax, __shfl_xor_sync(0xffffffffu, rmax, off));
    }
    if (lane == 0) {
        scale_base[row] = uint8_t(rmax);
    }
    const uint32_t lo_s = rmax > 15u ? (rmax - 15u) : 0u;
    const int pairs = blocks_per_row >> 1;
    uint8_t* out_row = scales_out + size_t(row) * pairs;
    uint8_t* w_row = weight + size_t(row) * blocks_per_row * 8;
    for (int p = lane; p < pairs; p += 32) {
        uint32_t s0 = min(uint32_t(s_row[2 * p]) & 0x7fu, 0x7eu);
        uint32_t s1 = min(uint32_t(s_row[2 * p + 1]) & 0x7fu, 0x7eu);
        if (s0 < lo_s) {
            reround_nvfp4_block(w_row + size_t(2 * p) * 8, s0, lo_s);
            s0 = lo_s;
        }
        if (s1 < lo_s) {
            reround_nvfp4_block(w_row + size_t(2 * p + 1) * 8, s1, lo_s);
            s1 = lo_s;
        }
        out_row[p] = uint8_t((rmax - s0) | ((rmax - s1) << 4));
    }
}

namespace {

// MXFP6 E2M3 (2-plane lo4 + hi2) geometry: identical 32-value chunks, 4 rows/warp for
// w13 (K = 2304) and 8 rows/warp for w2 (K = 1152) as NVFP4, with 1 UE8M0 scale per chunk.
template <int kK>
struct WarpRowsMxfp6 {
    static constexpr int kValues = 32;
    static constexpr int kRowChunks = kK / kValues;
    static constexpr int kRows = 32 * kChunks / kRowChunks;
    static constexpr int kBlockRows = kWarps * kRows;
    static constexpr int kRowLo4Bytes = kK / 2;
    static constexpr int kRowHi2Bytes = kK / 4;
    static constexpr int kRowScaleBytes = kK / 32;
    static_assert(kRows * kRowChunks == 32 * kChunks, "a warp covers whole rows");
    static_assert(kRowScaleBytes == kRowChunks, "one UE8M0 scale per 32-value chunk");

    uint4 w_lo[kChunks];
    uint2 w_hi[kChunks];
    uint32_t s[kChunks];
    const uint8_t* lr;
    const uint8_t* hr;
    const uint8_t* sr;
    uint64_t policy;
    bool known;

    __device__ __forceinline__ WarpRowsMxfp6(const uint8_t* w_lo4, const uint8_t* w_hi2,
                                             const uint8_t* scales, int expert, bool known_expert,
                                             int n, int row0)
        : known(known_expert) {
        policy = evict_first_policy();
        const size_t first_row = size_t(known ? expert : 0) * n + row0;
        lr = w_lo4 + first_row * kRowLo4Bytes;
        hr = w_hi2 + first_row * kRowHi2Bytes;
        sr = scales + first_row * kRowScaleBytes;
    }

    __device__ __forceinline__ void load(int i, int lane) {
        const int chunk = lane + 32 * i;
        w_lo[i] = known ? load_stream(lr + 16 * chunk, policy) : make_uint4(0, 0, 0, 0);
        w_hi[i] = known ? load_stream_u64(hr + 8 * chunk, policy) : make_uint2(0, 0);
        s[i] = known ? load_stream_u8(sr + chunk, policy) : 0u;
    }

    __device__ __forceinline__ float dot(const __nv_bfloat16* x, int lane) const {
        const uint4* xr = reinterpret_cast<const uint4*>(x);
        float acc[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            acc[r] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < kChunks; ++i) {
            const int chunk = lane + 32 * i;
            const int p = chunk % kRowChunks;
            float group = 0.f;
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const uint4 xa = xr[p * 4 + 2 * g];
                const uint4 xb = xr[p * 4 + 2 * g + 1];
                const uint32_t xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                const uint32_t hi_word = g == 0 ? w_hi[i].x : w_hi[i].y;
                const uint32_t lo_w0 = word(w_lo[i], 2 * g);
                const uint32_t lo_w1 = word(w_lo[i], 2 * g + 1);
                const uint32_t e0 = mxfp6::e2m3x4_to_e4m3x4(lo_w0 & 0xffffu, hi_word & 0xffu);
                const uint32_t e1 = mxfp6::e2m3x4_to_e4m3x4(lo_w0 >> 16, (hi_word >> 8) & 0xffu);
                const uint32_t e2 = mxfp6::e2m3x4_to_e4m3x4(lo_w1 & 0xffffu, (hi_word >> 16) & 0xffu);
                const uint32_t e3 = mxfp6::e2m3x4_to_e4m3x4(lo_w1 >> 16, hi_word >> 24);
                group = dot4_e4m3(xv + 0, e0, group);
                group = dot4_e4m3(xv + 2, e1, group);
                group = dot4_e4m3(xv + 4, e2, group);
                group = dot4_e4m3(xv + 6, e3, group);
            }
            const float sum = e8m0(s[i] & 0xffu) * group;
#pragma unroll
            for (int r = 32 * i / kRowChunks; r <= (32 * i + 31) / kRowChunks; ++r) {
                acc[r] += chunk / kRowChunks == r ? sum : 0.f;
            }
        }
        float result = 0.f;
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
            }
            result = lane == r ? acc[r] : result;
        }
        return result;
    }
};

template <int kK>
__device__ __forceinline__ void expert_matvec_mxfp6(const __nv_bfloat16* x, const int32_t* indices,
                                                    const uint8_t* w_lo4, const uint8_t* w_hi2,
                                                    const uint8_t* scales, __nv_bfloat16* out,
                                                    int n, int inputs_per_row, int experts) {
    using W = WarpRowsMxfp6<kK>;
    const int lane = threadIdx.x % 32;
    const int row0 = blockIdx.x * W::kBlockRows + threadIdx.x / 32 * W::kRows;
    const unsigned pair = blockIdx.y;
    pdl_wait();
    pdl_trigger();
    const int expert = indices[pair];
    const bool known = expert >= 0 && expert < experts;
    if (!known) {
        if (lane < W::kRows) {
            out[size_t(pair) * n + row0 + lane] = __float2bfloat16(kNan);
        }
        return;
    }
    W rows(w_lo4, w_hi2, scales, expert, true, n, row0);
#pragma unroll
    for (int i = 0; i < kChunks; ++i) {
        rows.load(i, lane);
    }
    const unsigned input = inputs_per_row == 1 ? pair / 8 : pair;
    const float result = rows.dot(x + size_t(input) * kK, lane);
    if (lane < W::kRows) {
        out[size_t(pair) * n + row0 + lane] = __float2bfloat16(result);
    }
}

template <int kK13, int kK2>
__device__ __forceinline__ void expert_mlp_mxfp6_impl(
    const __nv_bfloat16* x, const int32_t* indices, const uint8_t* w13_lo4, const uint8_t* w13_hi2,
    const uint8_t* s13, const uint8_t* w2_lo4, const uint8_t* w2_hi2, const uint8_t* s2,
    __nv_bfloat16* act, __nv_bfloat16* out, int n2, int pairs, int experts, float gate_softcap,
    float up_softcap, uint32_t* sync, const MoeConv& conv) {
    using W13 = WarpRowsMxfp6<kK13>;
    using W2 = WarpRowsMxfp6<kK2>;
    constexpr int kN13 = 2 * kK2;
    constexpr int kTiles13 = kN13 / W13::kBlockRows;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int tiles2 = n2 / W2::kBlockRows;
    const unsigned tasks13 = unsigned(pairs) * kTiles13;
    const unsigned tasks = tasks13 + unsigned(pairs) * tiles2;

    pdl_wait();
    for (unsigned task = blockIdx.x; task < tasks; task += gridDim.x) {
        if (task < tasks13) {
            const unsigned pair = task / kTiles13;
            const int row0 = task % kTiles13 * W13::kBlockRows + warp * W13::kRows;
            const int expert = indices[pair];
            const bool known = expert >= 0 && expert < experts;
            W13 rows(w13_lo4, w13_hi2, s13, expert, known, kN13, row0);
#pragma unroll
            for (int i = 0; i < kChunks; ++i) {
                rows.load(i, lane);
            }
            const float result = rows.dot(x + size_t(pair / 8) * kK13, lane);
            const float up = __shfl_down_sync(0xffffffffu, result, 1);
            if (lane < W13::kRows && lane % 2 == 0) {
                const float gate_bf16 = __bfloat162float(__float2bfloat16(result));
                const float up_bf16 = __bfloat162float(__float2bfloat16(up));
                act[size_t(pair) * kK2 + (row0 + lane) / 2] =
                    known ? softcapped_swiglu(gate_bf16, up_bf16, gate_softcap, up_softcap)
                          : __float2bfloat16(kNan);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence();
                atomicAdd(&sync[1 + pair], 1u);
            }
        } else {
            const unsigned pair = (task - tasks13) / tiles2;
            const unsigned tile = (task - tasks13) % tiles2;
            const int row0 = int(tile) * W2::kBlockRows + warp * W2::kRows;
            const int expert = indices[pair];
            const bool known = expert >= 0 && expert < experts;
            if (threadIdx.x == 0) {
                while (load_relaxed(&sync[1 + pair]) < kTiles13) {
                    __nanosleep(32);
                }
                __threadfence();
            }
            __syncthreads();
            W2 rows(w2_lo4, w2_hi2, s2, expert, known, n2, row0);
#pragma unroll
            for (int i = 0; i < kChunks; ++i) {
                rows.load(i, lane);
            }
            float result = rows.dot(act + size_t(pair) * kK2, lane);
            if (lane < W2::kRows) {
                result = known ? result : kNan;
                out[size_t(pair) * n2 + row0 + lane] = __float2bfloat16(result);
            }
            if (conv.hidden != nullptr) {
                uint32_t* count = sync + 1 + pairs + tile;
                const bool finisher = pair + 1 == unsigned(pairs);
                const int channel = int(tile) * W2::kBlockRows + threadIdx.x;
                const MoeConvInputs inputs = finisher && threadIdx.x < W2::kBlockRows
                                                 ? MoeConvInputs(conv, channel, n2)
                                                 : MoeConvInputs();
                __syncthreads();
                if (threadIdx.x == 0) {
                    if (!finisher) {
#if SIPHON_SM89
                        red_inc_release(count);
#else
                        __threadfence();
                        atomicAdd(count, 1u);
#endif
                    } else {
                        while (load_relaxed(count) < unsigned(pairs) - 1) {
                            __nanosleep(32);
                        }
#if SIPHON_SM89
                        (void)load_acquire(count);
#else
                        __threadfence();
#endif
                        *count = 0;
                    }
                }
                if (finisher) {
                    __syncthreads();
                    if (threadIdx.x < W2::kBlockRows) {
                        moe_conv(conv, inputs, out, n2, channel);
                    }
                }
            }
        }
    }
    pdl_trigger();
    if (warp == 0) {
        uint32_t last = 0;
        if (lane == 0) {
            last = (atom_inc_acq_rel(&sync[0]) == gridDim.x - 1) ? 1u : 0u;
        }
        if (__shfl_sync(0xffffffffu, last, 0)) {
            for (int p = lane; p <= pairs; p += 32) {
                sync[p] = 0;
            }
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kWarps * 32, 6)
    expert_matvec_mxfp6_2304(const __nv_bfloat16* x, const int32_t* indices, const uint8_t* w_lo4,
                             const uint8_t* w_hi2, const uint8_t* scales, __nv_bfloat16* out, int n,
                             int inputs_per_row, int experts) {
    expert_matvec_mxfp6<2304>(x, indices, w_lo4, w_hi2, scales, out, n, inputs_per_row, experts);
}

extern "C" __global__ void __launch_bounds__(kWarps * 32, 6)
    expert_matvec_mxfp6_1152(const __nv_bfloat16* x, const int32_t* indices, const uint8_t* w_lo4,
                             const uint8_t* w_hi2, const uint8_t* scales, __nv_bfloat16* out, int n,
                             int inputs_per_row, int experts) {
    expert_matvec_mxfp6<1152>(x, indices, w_lo4, w_hi2, scales, out, n, inputs_per_row, experts);
}

extern "C" __global__ void __launch_bounds__(kWarps * 32, 4)
    expert_mlp_mxfp6(const __nv_bfloat16* x, const int32_t* indices, const uint8_t* w13_lo4,
                     const uint8_t* w13_hi2, const uint8_t* s13, const uint8_t* w2_lo4,
                     const uint8_t* w2_hi2, const uint8_t* s2, __nv_bfloat16* act,
                     __nv_bfloat16* out, int n2, int pairs, int experts, float gate_softcap,
                     float up_softcap, uint32_t* sync, const float* weights,
                     const __nv_bfloat16* __restrict__ conv_weight, float* conv_state,
                     const float* residual, __nv_bfloat16* hidden) {
    const MoeConv conv{weights, conv_weight, conv_state, residual, hidden};
    expert_mlp_mxfp6_impl<2304, 1152>(x, indices, w13_lo4, w13_hi2, s13, w2_lo4, w2_hi2, s2, act,
                                      out, n2, pairs, experts, gate_softcap, up_softcap, sync, conv);
}

// Re-quantizes MXFP8 blocks (32 E4M3 values + 1 UE8M0 scale each) into 2-plane MXFP6 E2M3
// (`w_lo4`: 16 bytes/block, `w_hi2`: 8 bytes/block, `s_out`: 1 UE8M0 scale/block) using
// MSE-optimal selection between floor(log2(amax)) - 2 and ceil(log2(amax / 7.5)).
// One thread per 32-value block; grid = ceil(total_blocks / 256) blocks of 256 threads.
extern "C" __global__ void __launch_bounds__(256)
    pack_mxfp8_to_mxfp6(const uint8_t* __restrict__ w_in, const uint8_t* __restrict__ s_in,
                        uint8_t* __restrict__ w_lo4, uint8_t* __restrict__ w_hi2,
                        uint8_t* __restrict__ s_out, size_t total_blocks) {
    const size_t blk = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (blk >= total_blocks) {
        return;
    }
    const uint4* w_ptr = reinterpret_cast<const uint4*>(w_in + blk * 32);
    const uint4 v0 = w_ptr[0];
    const uint4 v1 = w_ptr[1];
    const uint32_t words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float mags[32];
    uint8_t signs[32];
    float amax = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t wd = words[w];
        const float2 p0 = e4m3x2(wd & 0xffffu);
        const float2 p1 = e4m3x2(wd >> 16);
        const float vals[4] = {fabsf(p0.x), fabsf(p0.y), fabsf(p1.x), fabsf(p1.y)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            mags[4 * w + j] = vals[j];
            signs[4 * w + j] = uint8_t((wd >> (8 * j + 7)) & 1u);
            amax = fmaxf(amax, vals[j]);
        }
    }
    const int orig_e = int(s_in[blk]);
    if (amax == 0.f) {
        s_out[blk] = uint8_t(orig_e);
        reinterpret_cast<uint4*>(w_lo4 + blk * 16)[0] = make_uint4(0u, 0u, 0u, 0u);
        reinterpret_cast<uint2*>(w_hi2 + blk * 8)[0] = make_uint2(0u, 0u);
        return;
    }
    const int exp_amax = int(floorf(log2f(amax)));
    const int d0 = max(-orig_e + 1, min(254 - orig_e, exp_amax - 2));
    const int d1 = (amax > 7.5f * ldexpf(1.f, d0)) ? max(-orig_e + 1, min(254 - orig_e, d0 + 1)) : d0;
    int best_d = d0;
    if (d1 != d0) {
        float err0 = 0.f;
        float err1 = 0.f;
        const float s0 = ldexpf(1.f, d0), inv0 = ldexpf(1.f, -d0);
        const float s1 = ldexpf(1.f, d1), inv1 = ldexpf(1.f, -d1);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            float q0, q1;
            mxfp6::quant_e2m3_mag(mags[i] * inv0, q0);
            mxfp6::quant_e2m3_mag(mags[i] * inv1, q1);
            const float diff0 = q0 * s0 - mags[i];
            const float diff1 = q1 * s1 - mags[i];
            err0 = fmaf(diff0, diff0, err0);
            err1 = fmaf(diff1, diff1, err1);
        }
        if (err1 < err0) {
            best_d = d1;
        }
    }
    s_out[blk] = uint8_t(orig_e + best_d);
    const float inv = ldexpf(1.f, -best_d);
    uint32_t lo_out[4] = {0u, 0u, 0u, 0u};
    uint32_t hi_out[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        float qv;
        const uint32_t idx = mxfp6::quant_e2m3_mag(mags[i] * inv, qv);
        const uint32_t e = idx >> 3;
        const uint32_t m = idx & 7u;
        const uint32_t lo4 = (uint32_t(signs[i]) << 3) | m;
        lo_out[i >> 3] |= lo4 << ((i & 7) * 4);
        hi_out[i >> 4] |= e << ((i & 15) * 2);
    }
    reinterpret_cast<uint4*>(w_lo4 + blk * 16)[0] =
        make_uint4(lo_out[0], lo_out[1], lo_out[2], lo_out[3]);
    reinterpret_cast<uint2*>(w_hi2 + blk * 8)[0] = make_uint2(hi_out[0], hi_out[1]);
}

namespace {

__device__ __forceinline__ float e4m3_round(float x) {
    const __nv_fp8_e4m3 q(x);
    return float(q);
}

__device__ __forceinline__ uint32_t cvt_e2m1x2_val(float lo_in, float hi_in, float& lo_val,
                                                   float& hi_val) {
#if SIPHON_SM89
    auto code = [](float y) -> uint32_t {
        const float a = fabsf(y);
        const uint32_t c = uint32_t(a > 0.25f) + uint32_t(a >= 0.75f) + uint32_t(a > 1.25f) +
                           uint32_t(a >= 1.75f) + uint32_t(a > 2.5f) + uint32_t(a >= 3.5f) +
                           uint32_t(a > 5.f);
        return isnan(y) ? 7u : c | (__float_as_uint(y) >> 31 << 3);
    };
    const uint32_t byte = code(lo_in) | (code(hi_in) << 4);
    const float2 vals = e2m1x2(byte);
    lo_val = vals.x;
    hi_val = vals.y;
    return byte;
#else
    uint32_t byte;
    uint32_t halves;
    asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %3, %2; cvt.u32.u8 %0, b; "
        "cvt.rn.f16x2.e2m1x2 %1, b; }"
        : "=r"(byte), "=r"(halves)
        : "f"(lo_in), "f"(hi_in));
    const __half2 h = *reinterpret_cast<const __half2*>(&halves);
    lo_val = __low2float(h);
    hi_val = __high2float(h);
    return byte;
#endif
}

__device__ __forceinline__ float fp4_candidate(const float (&x)[16], float scale, float inverse,
                                               float outer, uint2& packed) {
    float err = 0.f;
    uint32_t bytes[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float lv, hv;
        bytes[i] = cvt_e2m1x2_val(x[2 * i] * inverse, x[2 * i + 1] * inverse, lv, hv);
        const float d0 = lv * scale * outer - x[2 * i];
        const float d1 = hv * scale * outer - x[2 * i + 1];
        err = fmaf(d0, d0, err);
        err = fmaf(d1, d1, err);
    }
    packed.x = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    packed.y = bytes[4] | (bytes[5] << 8) | (bytes[6] << 16) | (bytes[7] << 24);
    return err;
}

}  // namespace

// Computes per-expert `global_scale = amax * (2688 / 1536)` for Four Over Six NVFP4 so
// `outer_scale = global_scale / 2688 = amax / 1536` fits both 4- and 6-max block candidates
// inside E4M3's finite range. Grid: (n_experts, 1, 1), block: (256, 1, 1).
extern "C" __global__ void __launch_bounds__(256)
    mxfp8_expert_amax(const uint8_t* __restrict__ w_in, const uint8_t* __restrict__ s_in,
                      float* __restrict__ global_scale, size_t blocks_per_expert) {
    const size_t expert = size_t(blockIdx.x);
    const size_t base_blk = expert * blocks_per_expert;
    float local_max = 0.f;
    for (size_t blk = threadIdx.x; blk < blocks_per_expert; blk += 256) {
        const size_t b = base_blk + blk;
        const float sc = e8m0(s_in[b]);
        const uint4* w_ptr = reinterpret_cast<const uint4*>(w_in + b * 32);
        const uint4 v0 = w_ptr[0];
        const uint4 v1 = w_ptr[1];
        const uint32_t words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float bmax = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t wd = words[w];
            const float2 p0 = e4m3x2(wd & 0xffffu);
            const float2 p1 = e4m3x2(wd >> 16);
            bmax = fmaxf(bmax, fmaxf(fmaxf(fabsf(p0.x), fabsf(p0.y)),
                                     fmaxf(fabsf(p1.x), fabsf(p1.y))));
        }
        local_max = fmaxf(local_max, bmax * sc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, off));
    }
    __shared__ float smem[8];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) {
        smem[warp] = local_max;
    }
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8 ? smem[lane] : 0.f;
#pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        }
        if (lane == 0) {
            global_scale[expert] = v * (kNvfp4ScaleEnc / 1536.f);
        }
    }
}

// Re-quantizes MXFP8 blocks (32 E4M3 values + 1 UE8M0 scale each) into two 16-element
// Four Over Six NVFP4 blocks matching layers 0–19 (`w_out`: 16 bytes/MXFP8 block,
// `s_out`: 2 E4M3 scales/MXFP8 block).
extern "C" __global__ void __launch_bounds__(256)
    pack_mxfp8_to_nvfp4(const uint8_t* __restrict__ w_in, const uint8_t* __restrict__ s_in,
                        const float* __restrict__ global_scale, uint8_t* __restrict__ w_out,
                        uint8_t* __restrict__ s_out, size_t blocks_per_expert,
                        size_t total_blocks) {
    const size_t blk = size_t(blockIdx.x) * 256 + threadIdx.x;
    if (blk >= total_blocks) {
        return;
    }
    const size_t expert = blk / blocks_per_expert;
    const float outer_scale = __fdiv_rn(global_scale[expert], kNvfp4ScaleEnc);
    const float encoder = (outer_scale > 0.f) ? __fdiv_rn(1.f, outer_scale) : 0.f;
    const float encoder6 = encoder * (1.f / 6.f);
    const float encoder4 = encoder * 0.25f;

    const float sc = e8m0(s_in[blk]);
    const uint4* w_ptr = reinterpret_cast<const uint4*>(w_in + blk * 32);
    const uint4 v0 = w_ptr[0];
    const uint4 v1 = w_ptr[1];
    const uint32_t words[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float x[32];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t wd = words[w];
        const float2 p0 = e4m3x2(wd & 0xffffu);
        const float2 p1 = e4m3x2(wd >> 16);
        x[4 * w + 0] = p0.x * sc;
        x[4 * w + 1] = p0.y * sc;
        x[4 * w + 2] = p1.x * sc;
        x[4 * w + 3] = p1.y * sc;
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        float h[16];
        float bmax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            h[i] = x[half * 16 + i];
            bmax = fmaxf(bmax, fabsf(h[i]));
        }
        const float scale6 = e4m3_round(fminf(bmax * encoder6, 448.f));
        const float scale4 = e4m3_round(fminf(bmax * encoder4, 448.f));
        const float denom6 = scale6 * outer_scale;
        const float denom4 = scale4 * outer_scale;
        const float inv6 = (denom6 > 0.f) ? __fdiv_rn(1.f, denom6) : 0.f;
        const float inv4 = (denom4 > 0.f) ? __fdiv_rn(1.f, denom4) : 0.f;
        uint2 packed6, packed4;
        const float err6 = fp4_candidate(h, scale6, inv6, outer_scale, packed6);
        const float err4 = fp4_candidate(h, scale4, inv4, outer_scale, packed4);
        const bool choose4 = err4 < err6;
        const uint2 packed = choose4 ? packed4 : packed6;
        const __nv_fp8_e4m3 sc_fp8(choose4 ? scale4 : scale6);
        const size_t nv_blk = blk * 2 + half;
        s_out[nv_blk] = sc_fp8.__x;
        reinterpret_cast<uint2*>(w_out + nv_blk * 8)[0] = packed;
    }
}
