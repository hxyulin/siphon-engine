// Programmatic dependent launch. A kernel launched with programmatic stream
// serialization may start while earlier kernels still run. Before pdl_wait() it may
// read only data no kernel writes, such as weights; pdl_wait() returns once every
// earlier kernel has finished and its writes are visible. pdl_trigger() lets the next
// such kernel start early. Without the launch attribute both are no-ops.

#pragma once

__device__ __forceinline__ void pdl_wait() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void pdl_trigger() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
