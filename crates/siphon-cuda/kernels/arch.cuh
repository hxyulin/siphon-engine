// The target the kernels are built for. The default target, sm_120a (RTX 5090, RTX PRO 6000
// Blackwell), has everything below. Ada (sm_89, RTX 4090) lacks programmatic dependent
// launch, thread-block clusters, bulk prefetches into L2 and the block-scaled MMA, so
// SIPHON_SM89 selects replacements: pdl.cuh's calls do nothing, cluster_sm89.cuh carries
// the cluster exchanges through global memory, and the kernels that need the block-scaled
// MMA trap if launched (the launchers refuse them first).

#pragma once

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define SIPHON_SM89 1
#else
#define SIPHON_SM89 0
#endif

#if SIPHON_SM89
#include <cstdint>

// Two E2M1 values (the low byte of `byte`, low nibble first) as F16x2 bits, exactly. Ada has
// no E2M1 converter, and CUDA's fallback goes through E2M3 in software. The F16 forms of
// the magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6 differ only in their high byte, so one
// byte permute looks up both high bytes, and the sign bits move to bits 15 and 31.
__device__ __forceinline__ uint32_t e2m1x2_to_f16x2_sm89(uint32_t byte) {
    const uint32_t select = ((byte & 0x7u) << 4) | ((byte & 0x70u) << 8);
    uint32_t h;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(h) : "r"(0x3e3c3800u), "r"(0x46444240u), "r"(select));
    return h | ((byte & 0x8u) << 12) | ((byte & 0x80u) << 24);
}
#endif

// __cluster_dims__(x, 1, 1) where clusters exist.
#if SIPHON_SM89
#define SIPHON_CLUSTER_DIMS(x)
#else
#define SIPHON_CLUSTER_DIMS(x) __cluster_dims__(x, 1, 1)
#endif
