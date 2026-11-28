// Dense products `out = x @ W^T` for 1 to 256 rows, with NVFP4 (E2M1) weights, BF16 inputs
// and outputs, and F32 accumulation, plus fused greedy LM-head tile-argmax (`rows_nvfp4_k2304_argmax`):
// serves both multi-row NVFP4 target LM head (`m = 1..256`, verify/batch/prefill) and
// multi-row NVFP4 draft projections (`m = 1..256`, single-stream and batched DFlash 2)
// with the `Nvfp4Matrix` layout of `linear_nvfp4.cu`.
//
// x is [m, k] and out [m, n], row-major, m from 1 to 256. W is [n, k] row-major as an
// Nvfp4Matrix:
// - E2M1 pairs [padded_n, k / 2], low nibble even column;
// - E4M3 block scales [padded_n, k / 16], one per 16 values;
// - F32 outer scales [padded_n], one per row.
//
// The weights and block scales stream from DRAM once for all m rows, mirroring
// `linear_rows.cu`. A block owns a tile of 16 weight rows and its `splits` warps share the
// tile's k range in stages of 256 elements (128 bytes of packed E2M1 values plus 16 bytes
// of E4M3 block scales per row, copied via `cp.async.cg` into a 3-slot shared-memory ring).
//
// Each E2M1 weight (at most 2 significant bits) multiplied by its E4M3 block scale (at most
// 4 significant bits) has at most 6 significant bits and magnitude in [2^-10, 2688], which is
// exact in BF16 (8 significant bits). Widening E2M1 through F16 and shifting right by 3
// gives `e2m1 * 2^-112` in BF16; multiplying by `e4m3 * 2^112` produces the exact scaled
// BF16 weight in one HMUL2 before `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
// Each row's F32 outer scale multiplies the final F32 accumulator once before rounding to
// BF16, so every output row is bit-invariant across m = 1..256 and independent of other
// rows.
//
// With programmatic dependent launch, each warp issues its first `kSlots - 1` weight and
// block-scale stages before `pdl_wait`, and reads `x` only after it.

#include "arch.cuh"
#include "pdl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if !SIPHON_SM89
#include <cuda_fp4.h>
#endif
#include <cuda_fp8.h>
#include <cfloat>
#include <climits>
#include <cstdint>

namespace {

constexpr int kMaxSegments = 8;
// Most warps per block, and so most splits of a tile's k range.
constexpr int kMaxWarps = 8;
// Ring slots per warp: copies run kSlots - 1 stages ahead of the products.
constexpr int kSlots = 3;
// Elements of k per stage: 128 bytes of E2M1 values and 16 bytes of E4M3 scales per row.
constexpr int kKPerStage = 256;
// 16-byte value chunks per tile row per stage (128 / 16).
constexpr int kValChunks = kKPerStage / 32;
// Weight rows per tile.
constexpr int kTileRows = 16;
// 16-byte vectors per slot: 16 x 8 values + 16 x 1 block scales.
constexpr int kValSlotVecs = kTileRows * kValChunks;
constexpr int kScaleSlotVecs = kTileRows;
constexpr int kSlotVecs = kValSlotVecs + kScaleSlotVecs;

struct Segment {
    const uint8_t* w;
    const uint8_t* scale;
    const float* outer;
    const __nv_bfloat16* x;
    __nv_bfloat16* out;
    int n;
    // Tiles of this segment and every segment before it.
    int tile_end;
    // Tile pairs of this segment and every segment before it.
    int pair_end;
    int _pad;
};
static_assert(sizeof(Segment) == 56, "Segment matches Rust layout and pdl_audit.py");

struct Batch {
    Segment seg[kMaxSegments];
    int count;
    int m;
    // Nonzero when launched with programmatic dependent launch.
    int pdl;
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16 bytes from global to shared memory, past L1.
__device__ __forceinline__ void copy16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr(dst)), "l"(src)
                 : "memory");
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void wait_copies() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2,
                                    uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Converts two E4M3 block scales (low byte s0, high byte s1) into broadcast BF16x2 factors
// equal to `(s0 * 2^112, s0 * 2^112)` and `(s1 * 2^112, s1 * 2^112)`, bit-exact for all
// finite E4M3 scales with zero F16->F32 conversions (`0x7780` is BF16 2^112).
__device__ __forceinline__ void e4m3x2_to_bf16x2_p112(uint32_t pair, __nv_bfloat162& s0_p112,
                                                      __nv_bfloat162& s1_p112) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3);
    const uint32_t u = uint32_t(h.x) | (uint32_t(h.y) << 16);
    const uint32_t scaled = ((u & 0x7fff7fffu) >> 3) | (u & 0x80008000u);
    const uint32_t k_lift = 0x77807780u;
    const __nv_bfloat162 lift = *reinterpret_cast<const __nv_bfloat162*>(&k_lift);
    const __nv_bfloat162 lifted =
        __hmul2(__hmul2(*reinterpret_cast<const __nv_bfloat162*>(&scaled), lift), lift);
    const uint32_t bits = *reinterpret_cast<const uint32_t*>(&lifted);
    const uint32_t lo = (bits & 0xffffu) | ((bits & 0xffffu) << 16);
    const uint32_t hi = (bits >> 16) | (bits & 0xffff0000u);
    s0_p112 = *reinterpret_cast<const __nv_bfloat162*>(&lo);
    s1_p112 = *reinterpret_cast<const __nv_bfloat162*>(&hi);
}

// Two E2M1 values (low nibble first) widened to BF16x2 and multiplied by `scale_p112`
// (`e4m3 * 2^112`), exact with zero rounding error. On sm_89 fuses `prmt >> 3` + sign OR.
__device__ __forceinline__ uint32_t e2m1x2_to_bf16x2_scaled(uint32_t byte,
                                                            __nv_bfloat162 scale_p112) {
#if SIPHON_SM89
    const uint32_t select = ((byte & 0x7u) << 4) | ((byte & 0x70u) << 8);
    uint32_t h;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(h) : "r"(0x3e3c3800u), "r"(0x46444240u), "r"(select));
    uint32_t r = (h >> 3) | ((byte & 0x8u) << 12) | ((byte & 0x80u) << 24);
#else
    const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2(uint8_t(byte), __NV_E2M1);
    const uint32_t u = uint32_t(h.x) | (uint32_t(h.y) << 16);
    uint32_t r = ((u & 0x7fff7fffu) >> 3) | (u & 0x80008000u);
#endif
    __nv_bfloat162 v = *reinterpret_cast<__nv_bfloat162*>(&r);
    v = __hmul2(v, scale_p112);
    return *reinterpret_cast<uint32_t*>(&v);
}

// Swizzled index of value chunk c (0..7) in tile row r (0..15): odd rows swap their two
// 64-byte halves so the 8 lanes of a shared-memory phase hit distinct banks.
__device__ __forceinline__ int val_index(int r, int c) {
    return r * kValChunks + (c ^ ((r & 1) << 2));
}

// True if (a, ia) beats (b, ib) under argmax_f32's rule: larger value wins, ties go to the
// smaller index, and NaN beats every number (first NaN wins).
__device__ __forceinline__ bool argmax_better(float a, int ia, float b, int ib) {
    const bool a_nan = isnan(a);
    const bool b_nan = isnan(b);
    if (a_nan || b_nan) {
        return a_nan && (!b_nan || ia < ib);
    }
    return a > b || (a == b && ia < ib);
}

template <int K, int NT, int kT, bool kChunked = false>
__device__ __forceinline__ void rows_product_nvfp4(const Batch& b, uint4* ring) {
    constexpr int kStagesPerRow = K / kKPerStage;
    static_assert(kStagesPerRow * kKPerStage == K, "whole stages per row");
    static_assert(kT == 1 || kT == 2, "one tile or a pair");

    const int tile_block = kChunked ? int(blockIdx.y) : int(blockIdx.x);
    const int col_base = kChunked ? int(blockIdx.x) * (8 * NT) : 0;
    int s = 0;
    int begin = 0;
    while (s + 1 < b.count && tile_block >= (kT == 1 ? b.seg[s].tile_end : b.seg[s].pair_end)) {
        begin = kT == 1 ? b.seg[s].tile_end : b.seg[s].pair_end;
        ++s;
    }
    const Segment& seg = b.seg[s];
    const int splits = blockDim.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int r0 = (tile_block - begin) * kT * kTileRows;
    uint4* buf = ring + warp * kSlots * kT * kSlotVecs;
    const int stages = (kStagesPerRow - warp + splits - 1) / splits;

    auto issue = [&](int i) {
        if (i < stages) {
            const int stage = warp + i * splits;
#pragma unroll
            for (int u = 0; u < kT; ++u) {
                uint4* slot = buf + (i % kSlots * kT + u) * kSlotVecs;
#pragma unroll
                for (int q = 0; q < kValSlotVecs / 32; ++q) {
                    const int idx = q * 32 + lane;
                    const int r = idx / kValChunks;
                    const int c = idx % kValChunks;
                    const int row = min(r0 + u * kTileRows + r, seg.n - 1);
                    copy16(slot + val_index(r, c),
                           seg.w + size_t(row) * (K / 2) + stage * (kKPerStage / 2) + c * 16);
                }
                if (lane < kTileRows) {
                    const int row = min(r0 + u * kTileRows + lane, seg.n - 1);
                    copy16(slot + kValSlotVecs + lane,
                           seg.scale + size_t(row) * (K / 16) + stage * (kKPerStage / 16));
                }
            }
        }
        commit();
    };
#pragma unroll
    for (int i = 0; i < kSlots - 1; ++i) {
        issue(i);
    }
    if (b.pdl) {
        pdl_wait();
    }
    pdl_trigger();

    const uint4* x[NT];
    bool live[NT];
    float acc[kT][NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int col_row = col_base + g + 8 * n;
        x[n] = reinterpret_cast<const uint4*>(seg.x + size_t(col_row) * K);
        live[n] = col_row < b.m;
#pragma unroll
        for (int u = 0; u < kT; ++u) {
            acc[u][n][0] = acc[u][n][1] = acc[u][n][2] = acc[u][n][3] = 0.f;
        }
    }
    // 4 BF16 uint4s (32 inputs) per value chunk, 2 chunks per lane per stage = 8 uint4s.
    constexpr int kXVecs = (kValChunks / 4) * 4;
    for (int i = 0; i < stages; ++i) {
        issue(i + kSlots - 1);
        const int stage = warp + i * splits;
        uint4 xv[NT][kXVecs];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int j = 0; j < kValChunks / 4; ++j) {
                const int c = 4 * j + t;
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const int xi = (stage * kKPerStage + c * 32) / 8 + h;
                    xv[n][4 * j + h] = live[n] ? x[n][xi] : make_uint4(0, 0, 0, 0);
                }
            }
        }
        wait_copies<kSlots - 1>();
        __syncwarp();
#pragma unroll
        for (int u = 0; u < kT; ++u) {
            const uint4* slot = buf + (i % kSlots * kT + u) * kSlotVecs;
            const uint16_t* sc_g = reinterpret_cast<const uint16_t*>(slot + kValSlotVecs + g);
            const uint16_t* sc_g8 = reinterpret_cast<const uint16_t*>(slot + kValSlotVecs + g + 8);
#pragma unroll
            for (int j = 0; j < kValChunks / 4; ++j) {
                const int c = 4 * j + t;
                const uint4 a = slot[val_index(g, c)];
                const uint4 a8 = slot[val_index(g + 8, c)];
                __nv_bfloat162 sg0, sg1, sg8_0, sg8_1;
                e4m3x2_to_bf16x2_p112(sc_g[c], sg0, sg1);
                e4m3x2_to_bf16x2_p112(sc_g8[c], sg8_0, sg8_1);
                const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
                const uint32_t bw[4] = {a8.x, a8.y, a8.z, a8.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const __nv_bfloat162 sg = q < 2 ? sg0 : sg1;
                    const __nv_bfloat162 sg8 = q < 2 ? sg8_0 : sg8_1;
                    {
                        const uint32_t a0 = e2m1x2_to_bf16x2_scaled(aw[q], sg);
                        const uint32_t a1 = e2m1x2_to_bf16x2_scaled(bw[q], sg8);
                        const uint32_t a2 = e2m1x2_to_bf16x2_scaled(aw[q] >> 8, sg);
                        const uint32_t a3 = e2m1x2_to_bf16x2_scaled(bw[q] >> 8, sg8);
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            const uint4 xq = xv[n][4 * j + q];
                            mma(acc[u][n], a0, a1, a2, a3, xq.x, xq.y);
                        }
                    }
                    {
                        const uint32_t a0 = e2m1x2_to_bf16x2_scaled(aw[q] >> 16, sg);
                        const uint32_t a1 = e2m1x2_to_bf16x2_scaled(bw[q] >> 16, sg8);
                        const uint32_t a2 = e2m1x2_to_bf16x2_scaled(aw[q] >> 24, sg);
                        const uint32_t a3 = e2m1x2_to_bf16x2_scaled(bw[q] >> 24, sg8);
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            const uint4 xq = xv[n][4 * j + q];
                            mma(acc[u][n], a0, a1, a2, a3, xq.z, xq.w);
                        }
                    }
                }
            }
        }
        __syncwarp();
    }

    static_assert(NT * 32 * 16 <= kSlots * kSlotVecs * 16, "the partials fit a warp's ring");
    if (splits > 1) {
#pragma unroll
        for (int u = 0; u < kT; ++u) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                reinterpret_cast<float4*>(buf + u * kSlots * kSlotVecs)[n * 32 + lane] =
                    make_float4(acc[u][n][0], acc[u][n][1], acc[u][n][2], acc[u][n][3]);
            }
        }
        __syncthreads();
        if (warp != 0) {
            return;
        }
        for (int w = 1; w < splits; ++w) {
#pragma unroll
            for (int u = 0; u < kT; ++u) {
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const float4 p = reinterpret_cast<const float4*>(
                        ring + (w * kT + u) * kSlots * kSlotVecs)[n * 32 + lane];
                    acc[u][n][0] += p.x;
                    acc[u][n][1] += p.y;
                    acc[u][n][2] += p.z;
                    acc[u][n][3] += p.w;
                }
            }
        }
    }

#pragma unroll
    for (int u = 0; u < kT; ++u) {
        const int row0 = r0 + u * kTileRows + g;
        const int row8 = row0 + 8;
        const float outer0 = row0 < seg.n ? seg.outer[row0] : 0.f;
        const float outer8 = row8 < seg.n ? seg.outer[row8] : 0.f;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int col = col_base + 8 * n + 2 * t + (e & 1);
                const int row = (e < 2) ? row0 : row8;
                const float outer = (e < 2) ? outer0 : outer8;
                if (col < b.m && row < seg.n) {
                    seg.out[size_t(col) * seg.n + row] = __float2bfloat16_rn(outer * acc[u][n][e]);
                }
            }
        }
    }
}

// Single-product NVFP4 LM head with a fused tile-argmax epilogue: runs the exact same MMA
// and warp reduction as `rows_product_nvfp4`, applies `outer[row]`, rounds each logit to
// BF16 (matching `rows_product_nvfp4` bit for bit), masks `row < vocab`, reduces across
// the 8 g-lanes sharing each input row, and writes one `(max_val, max_idx)` partial per
// `(row, tile_block)`.
template <int K, int NT, int kT, bool kChunked = false>
__device__ __forceinline__ void rows_product_nvfp4_argmax(
    const uint8_t* weight, const uint8_t* scales, const float* outer, const __nv_bfloat16* x_in,
    float* partial_vals, int* partial_idxs, int m, int vocab, int partial_stride, int pdl,
    uint4* ring) {
    constexpr int kStagesPerRow = K / kKPerStage;
    static_assert(kStagesPerRow * kKPerStage == K, "whole stages per row");
    static_assert(kT == 1 || kT == 2, "one tile or a pair");

    const int tile_block = kChunked ? int(blockIdx.y) : int(blockIdx.x);
    const int col_base = kChunked ? int(blockIdx.x) * (8 * NT) : 0;
    const int splits = blockDim.x / 32;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int g = lane / 4;
    const int t = lane % 4;
    const int r0 = tile_block * kT * kTileRows;
    uint4* buf = ring + warp * kSlots * kT * kSlotVecs;
    const int stages = (kStagesPerRow - warp + splits - 1) / splits;

    auto issue = [&](int i) {
        if (i < stages) {
            const int stage = warp + i * splits;
#pragma unroll
            for (int u = 0; u < kT; ++u) {
                uint4* slot = buf + (i % kSlots * kT + u) * kSlotVecs;
#pragma unroll
                for (int q = 0; q < kValSlotVecs / 32; ++q) {
                    const int idx = q * 32 + lane;
                    const int r = idx / kValChunks;
                    const int c = idx % kValChunks;
                    const int row = min(r0 + u * kTileRows + r, vocab - 1);
                    copy16(slot + val_index(r, c),
                           weight + size_t(row) * (K / 2) + stage * (kKPerStage / 2) + c * 16);
                }
                if (lane < kTileRows) {
                    const int row = min(r0 + u * kTileRows + lane, vocab - 1);
                    copy16(slot + kValSlotVecs + lane,
                           scales + size_t(row) * (K / 16) + stage * (kKPerStage / 16));
                }
            }
        }
        commit();
    };
#pragma unroll
    for (int i = 0; i < kSlots - 1; ++i) {
        issue(i);
    }
    if (pdl) {
        pdl_wait();
    }
    pdl_trigger();

    const uint4* x[NT];
    bool live[NT];
    float acc[kT][NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int col_row = col_base + g + 8 * n;
        x[n] = reinterpret_cast<const uint4*>(x_in + size_t(col_row) * K);
        live[n] = col_row < m;
#pragma unroll
        for (int u = 0; u < kT; ++u) {
            acc[u][n][0] = acc[u][n][1] = acc[u][n][2] = acc[u][n][3] = 0.f;
        }
    }
    constexpr int kXVecs = (kValChunks / 4) * 4;
    for (int i = 0; i < stages; ++i) {
        issue(i + kSlots - 1);
        const int stage = warp + i * splits;
        uint4 xv[NT][kXVecs];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int j = 0; j < kValChunks / 4; ++j) {
                const int c = 4 * j + t;
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const int xi = (stage * kKPerStage + c * 32) / 8 + h;
                    xv[n][4 * j + h] = live[n] ? x[n][xi] : make_uint4(0, 0, 0, 0);
                }
            }
        }
        wait_copies<kSlots - 1>();
        __syncwarp();
#pragma unroll
        for (int u = 0; u < kT; ++u) {
            const uint4* slot = buf + (i % kSlots * kT + u) * kSlotVecs;
            const uint16_t* sc_g = reinterpret_cast<const uint16_t*>(slot + kValSlotVecs + g);
            const uint16_t* sc_g8 = reinterpret_cast<const uint16_t*>(slot + kValSlotVecs + g + 8);
#pragma unroll
            for (int j = 0; j < kValChunks / 4; ++j) {
                const int c = 4 * j + t;
                const uint4 a = slot[val_index(g, c)];
                const uint4 a8 = slot[val_index(g + 8, c)];
                __nv_bfloat162 sg0, sg1, sg8_0, sg8_1;
                e4m3x2_to_bf16x2_p112(sc_g[c], sg0, sg1);
                e4m3x2_to_bf16x2_p112(sc_g8[c], sg8_0, sg8_1);
                const uint32_t aw[4] = {a.x, a.y, a.z, a.w};
                const uint32_t bw[4] = {a8.x, a8.y, a8.z, a8.w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const __nv_bfloat162 sg = q < 2 ? sg0 : sg1;
                    const __nv_bfloat162 sg8 = q < 2 ? sg8_0 : sg8_1;
                    {
                        const uint32_t a0 = e2m1x2_to_bf16x2_scaled(aw[q], sg);
                        const uint32_t a1 = e2m1x2_to_bf16x2_scaled(bw[q], sg8);
                        const uint32_t a2 = e2m1x2_to_bf16x2_scaled(aw[q] >> 8, sg);
                        const uint32_t a3 = e2m1x2_to_bf16x2_scaled(bw[q] >> 8, sg8);
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            const uint4 xq = xv[n][4 * j + q];
                            mma(acc[u][n], a0, a1, a2, a3, xq.x, xq.y);
                        }
                    }
                    {
                        const uint32_t a0 = e2m1x2_to_bf16x2_scaled(aw[q] >> 16, sg);
                        const uint32_t a1 = e2m1x2_to_bf16x2_scaled(bw[q] >> 16, sg8);
                        const uint32_t a2 = e2m1x2_to_bf16x2_scaled(aw[q] >> 24, sg);
                        const uint32_t a3 = e2m1x2_to_bf16x2_scaled(bw[q] >> 24, sg8);
#pragma unroll
                        for (int n = 0; n < NT; ++n) {
                            const uint4 xq = xv[n][4 * j + q];
                            mma(acc[u][n], a0, a1, a2, a3, xq.z, xq.w);
                        }
                    }
                }
            }
        }
        __syncwarp();
    }

    static_assert(NT * 32 * 16 <= kSlots * kSlotVecs * 16, "the partials fit a warp's ring");
    if (splits > 1) {
#pragma unroll
        for (int u = 0; u < kT; ++u) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                reinterpret_cast<float4*>(buf + u * kSlots * kSlotVecs)[n * 32 + lane] =
                    make_float4(acc[u][n][0], acc[u][n][1], acc[u][n][2], acc[u][n][3]);
            }
        }
        __syncthreads();
        if (warp != 0) {
            return;
        }
        for (int w = 1; w < splits; ++w) {
#pragma unroll
            for (int u = 0; u < kT; ++u) {
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const float4 p = reinterpret_cast<const float4*>(
                        ring + (w * kT + u) * kSlots * kSlotVecs)[n * 32 + lane];
                    acc[u][n][0] += p.x;
                    acc[u][n][1] += p.y;
                    acc[u][n][2] += p.z;
                    acc[u][n][3] += p.w;
                }
            }
        }
    }

    float sc[kT][2];
#pragma unroll
    for (int u = 0; u < kT; ++u) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = r0 + u * kTileRows + g + h * 8;
            sc[u][h] = row < vocab ? outer[row] : 0.f;
        }
    }

#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int e_col = 0; e_col < 2; ++e_col) {
            const int col = col_base + 8 * n + 2 * t + e_col;
            float best_v = -INFINITY;
            int best_i = INT_MAX;
            if (col < m) {
#pragma unroll
                for (int u = 0; u < kT; ++u) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int row = r0 + u * kTileRows + g + h * 8;
                        if (row < vocab) {
                            const float raw = sc[u][h] * acc[u][n][2 * h + e_col];
                            const float v = __bfloat162float(__float2bfloat16_rn(raw));
                            if (argmax_better(v, row, best_v, best_i)) {
                                best_v = v;
                                best_i = row;
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int offset = 4; offset <= 16; offset *= 2) {
                const float ov = __shfl_xor_sync(0xffffffffu, best_v, offset);
                const int oi = __shfl_xor_sync(0xffffffffu, best_i, offset);
                if (argmax_better(ov, oi, best_v, best_i)) {
                    best_v = ov;
                    best_i = oi;
                }
            }
            if (g == 0 && col < m) {
                const size_t dst = size_t(col) * partial_stride + tile_block;
                partial_vals[dst] = best_v;
                partial_idxs[dst] = best_i;
            }
        }
    }
}

}  // namespace

#define NVFP4_ROWS_ENTRY_NT(name, K, NT, kT, chunked)                                      \
    extern "C" __global__ void __launch_bounds__(kMaxWarps * 32)                           \
        name(const __grid_constant__ Batch batch) {                                        \
        extern __shared__ uint4 ring[];                                                    \
        rows_product_nvfp4<K, NT, kT, chunked>(batch, ring);                               \
    }
#define NVFP4_ROWS_ENTRY(name, K)                                                          \
    NVFP4_ROWS_ENTRY_NT(name, K, 1, 1, false)                                              \
    NVFP4_ROWS_ENTRY_NT(name##_2, K, 2, 1, false)                                          \
    NVFP4_ROWS_ENTRY_NT(name##_3, K, 3, 1, false)                                          \
    NVFP4_ROWS_ENTRY_NT(name##_4, K, 4, 1, false)                                          \
    NVFP4_ROWS_ENTRY_NT(name##_3_pair, K, 3, 2, false)                                     \
    NVFP4_ROWS_ENTRY_NT(name##_4_pair, K, 4, 2, false)                                     \
    NVFP4_ROWS_ENTRY_NT(name##_4_chunk, K, 4, 1, true)                                     \
    NVFP4_ROWS_ENTRY_NT(name##_4_pair_chunk, K, 4, 2, true)

#define NVFP4_ROWS_ARGMAX_ENTRY_NT(name, K, NT, kT, chunked)                                   \
    extern "C" __global__ void __launch_bounds__(kMaxWarps * 32) name(                         \
        const uint8_t* weight, const uint8_t* scales, const float* outer,                      \
        const __nv_bfloat16* x_in, float* partial_vals, int* partial_idxs, int m, int vocab,   \
        int partial_stride, int pdl) {                                                         \
        extern __shared__ uint4 ring[];                                                        \
        rows_product_nvfp4_argmax<K, NT, kT, chunked>(weight, scales, outer, x_in,             \
                                                      partial_vals, partial_idxs, m, vocab,    \
                                                      partial_stride, pdl, ring);              \
    }
#define NVFP4_ROWS_ARGMAX_ENTRY(name, K)                                                       \
    NVFP4_ROWS_ARGMAX_ENTRY_NT(name##_argmax, K, 1, 1, false)                                  \
    NVFP4_ROWS_ARGMAX_ENTRY_NT(name##_argmax_2, K, 2, 1, false)                                \
    NVFP4_ROWS_ARGMAX_ENTRY_NT(name##_argmax_3, K, 3, 1, false)                                \
    NVFP4_ROWS_ARGMAX_ENTRY_NT(name##_argmax_4, K, 4, 1, false)                                \
    NVFP4_ROWS_ARGMAX_ENTRY_NT(name##_argmax_3_pair, K, 3, 2, false)                           \
    NVFP4_ROWS_ARGMAX_ENTRY_NT(name##_argmax_4_pair, K, 4, 2, false)                           \
    NVFP4_ROWS_ARGMAX_ENTRY_NT(name##_argmax_4_chunk, K, 4, 1, true)                           \
    NVFP4_ROWS_ARGMAX_ENTRY_NT(name##_argmax_4_pair_chunk, K, 4, 2, true)

// Attention input projections, MLP gate/up, context K/V, and LM head.
NVFP4_ROWS_ENTRY(rows_nvfp4_k2304, 2304)
NVFP4_ROWS_ARGMAX_ENTRY(rows_nvfp4_k2304, 2304)
// Bundled / 2DSD DFlash attention output projection.
NVFP4_ROWS_ENTRY(rows_nvfp4_k3072, 3072)
// DFlash 2 attention output projection (32 heads x 128).
NVFP4_ROWS_ENTRY(rows_nvfp4_k4096, 4096)
// Message gates' down projections.
NVFP4_ROWS_ENTRY(rows_nvfp4_k4608, 4608)
// DFlash 2 MLP down projection.
NVFP4_ROWS_ENTRY(rows_nvfp4_k6912, 6912)
// DFlash 2 context fusion projection (4 taps x 2304) and bundled MLP down.
NVFP4_ROWS_ENTRY(rows_nvfp4_k9216, 9216)
