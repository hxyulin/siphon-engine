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
template <bool kFp4, int kK>
struct Geometry {
    static constexpr int kValues = kFp4 ? 32 : 16;  // per chunk
    static constexpr int kRowChunks = kK / kValues;
    static constexpr int kRows = 32 * kChunks / kRowChunks;  // per warp
    static constexpr int kBlockRows = kWarps * kRows;
    static constexpr int kGroup = kFp4 ? 16 : 32;
    static constexpr int kRowBytes = kFp4 ? kK / 2 : kK;
    static constexpr int kRowScales = kK / kGroup;
    static_assert(kRows * kRowChunks == 32 * kChunks, "a warp covers whole rows");
};

// One warp's `kRows` weight rows of one expert: chunk `lane + 32 * i` is row
// `chunk / kRowChunks`, position `chunk % kRowChunks` in it.
//
// The loads are volatile asm, which nvcc keeps in order with `pdl_wait` and the flag wait
// but interleaves with the arithmetic around them. So the kernels load only after their
// waits: loads placed before a wait all issue at once, and issuing a warp's nine chunks
// together took 21 us instead of 16.5 for an NVFP4 `w13` on an RTX 5090. More loads in
// flight lower DRAM throughput here.
template <bool kFp4, int kK>
struct WarpRows {
    using G = Geometry<kFp4, kK>;
    uint4 w[kChunks];
    uint32_t s[kChunks];
    const uint8_t* wr;
    const uint8_t* sr;
    uint64_t policy;
    bool known;

    // Rows `row0 ..` of `expert`, whose rows number n. An unknown expert loads zeros.
    __device__ __forceinline__ WarpRows(const uint8_t* weight, const uint8_t* scales, int expert,
                                        bool known_expert, int n, int row0)
        : known(known_expert) {
        policy = evict_first_policy();
        const size_t first_row = size_t(known ? expert : 0) * n + row0;
        wr = weight + first_row * G::kRowBytes;
        sr = scales + first_row * G::kRowScales;
    }

    __device__ __forceinline__ void load(int i, int lane) {
        const int chunk = lane + 32 * i;
        const int p = chunk % G::kRowChunks;
        const uint8_t* sp = sr + chunk / G::kRowChunks * G::kRowScales + (kFp4 ? 2 * p : p / 2);
        w[i] = known ? load_stream(wr + 16 * chunk, policy) : make_uint4(0, 0, 0, 0);
        s[i] = !known ? 0 : kFp4 ? load_stream_u16(sp, policy) : load_stream_u8(sp, policy);
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
                    sum = fmaf(e4m3(s[i] >> (8 * g)), group, sum);
                } else {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        group = dot4_e4m3(xv + 2 * j, word(w[i], j), group);
                    }
                    sum = e8m0(s[i] & 0xff) * group;
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
    float total = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float wk = in.weights[k];
        if (wk != 0.f) {
            total = __fadd_rn(total, __fmul_rn(__bfloat162float(__ldcg(down + size_t(k) * n + c)), wk));
        }
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
template <bool kFp4, int kK>
__device__ __forceinline__ void expert_matvec(const __nv_bfloat16* x, const int32_t* indices,
                                              const uint8_t* weight, const uint8_t* scales,
                                              const float* global_scale, __nv_bfloat16* out, int n,
                                              int inputs_per_row, int experts) {
    using G = Geometry<kFp4, kK>;
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
    WarpRows<kFp4, kK> rows(weight, scales, expert, true, n, row0);
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
template <bool kFp4, int kK13, int kK2>
__device__ __forceinline__ void expert_mlp(const __nv_bfloat16* x, const int32_t* indices,
                                           const uint8_t* w13, const uint8_t* s13,
                                           const float* g13, const uint8_t* w2, const uint8_t* s2,
                                           const float* g2, __nv_bfloat16* act,
                                           __nv_bfloat16* out, int n2, int pairs, int experts,
                                           float gate_softcap, float up_softcap, uint32_t* sync,
                                           const MoeConv& conv) {
    using G13 = Geometry<kFp4, kK13>;
    using G2 = Geometry<kFp4, kK2>;
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
    pdl_trigger();
    for (unsigned task = blockIdx.x; task < tasks; task += gridDim.x) {
        if (task < tasks13) {
            const unsigned pair = task / kTiles13;
            const int row0 = task % kTiles13 * G13::kBlockRows + warp * G13::kRows;
            const int expert = indices[pair];
            const bool known = expert >= 0 && expert < experts;
            if (!known) {
                if (lane < G13::kRows && lane % 2 == 0) {
                    act[size_t(pair) * kK2 + (row0 + lane) / 2] = __float2bfloat16(kNan);
                }
                continue;
            }
            WarpRows<kFp4, kK13> rows(w13, s13, expert, true, kN13, row0);
#pragma unroll
            for (int i = 0; i < kChunks; ++i) {
                rows.load(i, lane);
            }
            float result = rows.dot(x + size_t(pair / 8) * kK13, lane);
            result *= output_scale(kFp4, g13, expert);
            const float up = __shfl_down_sync(0xffffffffu, result, 1);
            if (lane < G13::kRows && lane % 2 == 0) {
                const float gate_bf16 = __bfloat162float(__float2bfloat16(result));
                const float up_bf16 = __bfloat162float(__float2bfloat16(up));
                act[size_t(pair) * kK2 + (row0 + lane) / 2] =
                    softcapped_swiglu(gate_bf16, up_bf16, gate_softcap, up_softcap);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                __threadfence();
                atomicAdd(&sync[1 + pair], 1u);
            }
        } else {
            const unsigned pair = (task - tasks13) / tiles2;
            const int row0 = (task - tasks13) % tiles2 * G2::kBlockRows + warp * G2::kRows;
            const int expert = indices[pair];
            const bool known = expert >= 0 && expert < experts;
            if (known) {
                if (threadIdx.x == 0) {
                    while (load_relaxed(&sync[1 + pair]) < kTiles13) {
                        __nanosleep(64);
                    }
                    // With the barrier below, orders the block's activation loads after the
                    // count, and so after the stores the count covers.
                    __threadfence();
                }
                __syncthreads();
                // Loading weights before the wait was slower: 27.0 us instead of 24.6 per NVFP4
                // layer visit with all nine chunks.
                WarpRows<kFp4, kK2> rows(w2, s2, expert, true, n2, row0);
#pragma unroll
                for (int i = 0; i < kChunks; ++i) {
                    rows.load(i, lane);
                }
                float result = rows.dot(act + size_t(pair) * kK2, lane);
                if (lane < G2::kRows) {
                    result *= output_scale(kFp4, g2, expert);
                    out[size_t(pair) * n2 + row0 + lane] = __float2bfloat16(result);
                }
            } else if (lane < G2::kRows) {
                out[size_t(pair) * n2 + row0 + lane] = __float2bfloat16(kNan);
            }
            if (conv.hidden != nullptr) {
                // The last pair's task of each row tile finishes the tile's channels once
                // the other pairs' tasks, all earlier in task order, have counted in. Its
                // other inputs load first, so only the expert outputs wait for the count.
                const unsigned tile = (task - tasks13) % tiles2;
                uint32_t* count = sync + 1 + pairs + tile;
                const bool finisher = pair + 1 == unsigned(pairs);
                const int channel = int(tile) * G2::kBlockRows + threadIdx.x;
                const MoeConvInputs inputs = finisher && threadIdx.x < G2::kBlockRows
                                                 ? MoeConvInputs(conv, channel, n2)
                                                 : MoeConvInputs();
                __syncthreads();
                if (threadIdx.x == 0) {
                    if (!finisher) {
                        __threadfence();
                        atomicAdd(count, 1u);
                    } else {
                        while (load_relaxed(count) < unsigned(pairs) - 1) {
                            __nanosleep(32);
                        }
                        __threadfence();
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
    if (threadIdx.x == 0 && atomicAdd(&sync[0], 1u) == gridDim.x - 1) {
        // Every block has finished with the counts.
        for (int p = 0; p < pairs; ++p) {
            sync[1 + p] = 0;
        }
        sync[0] = 0;
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
        expert_matvec<fp4, k>(x, indices, weight, scales, global_scale, out, n, inputs_per_row,    \
                              experts);                                                            \
    }

EXPERT_MATVEC(expert_matvec_nvfp4_2304, true, 2304)
EXPERT_MATVEC(expert_matvec_nvfp4_1152, true, 1152)
EXPERT_MATVEC(expert_matvec_mxfp8_2304, false, 2304)
EXPERT_MATVEC(expert_matvec_mxfp8_1152, false, 1152)

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
#define EXPERT_MLP(name, fp4)                                                                      \
    extern "C" __global__ void __launch_bounds__(kWarps * 32, 7)                                    \
        name(const __nv_bfloat16* x, const int32_t* indices, const uint8_t* w13,                   \
             const uint8_t* s13, const float* g13, const uint8_t* w2, const uint8_t* s2,           \
             const float* g2, __nv_bfloat16* act, __nv_bfloat16* out, int n2, int pairs,           \
             int experts, float gate_softcap, float up_softcap, uint32_t* sync,                    \
             const float* weights, const __nv_bfloat16* __restrict__ conv_weight,                  \
             float* conv_state, const float* residual, __nv_bfloat16* hidden) {                    \
        const MoeConv conv{weights, conv_weight, conv_state, residual, hidden};                    \
        expert_mlp<fp4, 2304, 1152>(x, indices, w13, s13, g13, w2, s2, g2, act, out, n2, pairs,    \
                                    experts, gate_softcap, up_softcap, sync, conv);                \
    }

EXPERT_MLP(expert_mlp_nvfp4, true)
EXPERT_MLP(expert_mlp_mxfp8, false)
