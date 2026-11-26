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

// __cluster_dims__(x, 1, 1) where clusters exist.
#if SIPHON_SM89
#define SIPHON_CLUSTER_DIMS(x)
#else
#define SIPHON_CLUSTER_DIMS(x) __cluster_dims__(x, 1, 1)
#endif
