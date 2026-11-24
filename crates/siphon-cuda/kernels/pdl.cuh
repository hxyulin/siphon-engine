// Programmatic dependent launch. A kernel launched with programmatic stream
// serialization may start while earlier kernels still run. Before pdl_wait() it may
// read only data no kernel writes, such as weights; pdl_wait() returns once every
// earlier kernel has finished and its writes are visible. pdl_trigger() lets the next
// such kernel start early. Without the launch attribute on this kernel, pdl_wait() is a
// no-op, but pdl_trigger() still releases a successor that was launched with it.
//
// ptxas may move ld.global.nc loads across an unconditional pdl_wait(), and has placed
// the wait before every weight load. A kernel that must issue its loads first calls
// pdl_wait() inside a runtime branch, as the GEMVs in linear.cu do.
//
// The same freedom applies in the other direction: ptxas may move a non-coherent load
// of an earlier kernel's output above the wait, where it reads stale data. nvcc emits
// such loads for any `const T* __restrict__` pointer the kernel only reads, including
// parameters of inlined helpers. In a kernel launched with PDL, mark only weights and
// other data no kernel writes `__restrict__`; never activations, scratch, caches, conv
// states or counters. scripts/pdl_audit.py checks the compiled SASS for both rules.

#pragma once

__device__ __forceinline__ void pdl_wait() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void pdl_trigger() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
