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

#pragma once

__device__ __forceinline__ void pdl_wait() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void pdl_trigger() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
