// MXFP6 E2M3 bit-packing and exact E2M3 -> E4M3 SIMD decoding.
//
// Each 32-element MXFP6 E2M3 block shares one UE8M0 power-of-two scale (identical to MXFP8).
// Each 6-bit E2M3 value (1 sign, 2 exponent bits bias 1, 3 mantissa bits, max 7.5, no NaN/Inf)
// is stored in two 16-byte-aligned planes:
// - `lo4` (`[experts, n, k / 2]`): `[sign (bit 3), m2, m1, m0]` packed 2 values per byte
//   (even index in low nibble, odd index in high nibble).
// - `hi2` (`[experts, n, k / 4]`): `[e1, e0]` packed 4 values per byte (LSB-first, 2 bits each).
//
// Every E2M3 value has a bit-exact E4M3 representation:
// - normal (`e in 1..3`): `sign | ((e + 6) << 3) | m`
// - subnormal (`e == 0`): `sign | kSubLut[m]` where
//   `kSubLut = {0x00, 0x20, 0x28, 0x2c, 0x30, 0x32, 0x34, 0x36}`.

#pragma once

#include <cstdint>

namespace mxfp6 {

__device__ __forceinline__ uint32_t prmt_b32(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}

// Decodes 4 packed E2M3 values from 16 bits of `lo4` (4 nibbles `[s, m2, m1, m0]`) and
// 8 bits of `hi2` (4 two-bit exponents `[e1, e0]`) into 4 bit-exact E4M3 bytes in a `uint32_t`.
__device__ __forceinline__ uint32_t e2m3x4_to_e4m3x4(uint32_t lo16, uint32_t hi8) {
    // Subnormal E4M3 encodings for e == 0, m in 0..7:
    //   0->0x00, 1->0x20, 2->0x28, 3->0x2c, 4->0x30, 5->0x32, 6->0x34, 7->0x36
    constexpr uint32_t kSubLo = 0x2c282000u;
    constexpr uint32_t kSubHi = 0x36343230u;
    const uint32_t sub = prmt_b32(kSubLo, kSubHi, lo16 & 0x7777u);
    const uint32_t m4 = (lo16 & 0x0007u) | ((lo16 & 0x0070u) << 4) | ((lo16 & 0x0700u) << 8) |
                        ((lo16 & 0x7000u) << 12);
    const uint32_t e4 = ((hi8 & 0x03u) << 3) | ((hi8 & 0x0cu) << 9) | ((hi8 & 0x30u) << 15) |
                        ((hi8 & 0xc0u) << 21);
    const uint32_t norm = (e4 + 0x30303030u) | m4;
    // Per byte: `e4` is in {0x00, 0x08, 0x10, 0x18}; `nz` is 0x08 iff e != 0, else 0x00.
    // `(nz << 4) - (nz >> 3)` yields 0x7f where e != 0 and 0x00 where e == 0, with zero borrow.
    const uint32_t nz = (e4 | (e4 >> 1)) & 0x08080808u;
    const uint32_t norm_mask = (nz << 4) - (nz >> 3);
    const uint32_t mag = (norm & norm_mask) | (sub & ~norm_mask);
    const uint32_t s4 = ((lo16 & 0x0008u) << 4) | ((lo16 & 0x0080u) << 8) |
                        ((lo16 & 0x0800u) << 12) | ((lo16 & 0x8000u) << 16);
    return mag | s4;
}

// Quantizes a non-negative float `x` to nearest E2M3 magnitude index `0..31`
// (`(e << 3) | m`) and returns its exact quantized value in `*qval`.
__device__ __forceinline__ uint32_t quant_e2m3_mag(float x, float& qval) {
    uint32_t idx;
    if (x >= 7.25f) {
        idx = 31u;
    } else if (x < 1.0f) {
        idx = min(uint32_t(x * 8.0f + 0.5f), 8u);
    } else if (x < 2.0f) {
        idx = min(8u + uint32_t((x - 1.0f) * 8.0f + 0.5f), 16u);
    } else if (x < 4.0f) {
        idx = min(16u + uint32_t((x - 2.0f) * 4.0f + 0.5f), 24u);
    } else {
        idx = min(24u + uint32_t((x - 4.0f) * 2.0f + 0.5f), 31u);
    }
    const uint32_t e = idx >> 3;
    const uint32_t m = idx & 7u;
    qval = e == 0u ? (float(m) * 0.125f) : ((1.0f + float(m) * 0.125f) * float(1u << (e - 1u)));
    return idx;
}

}  // namespace mxfp6
