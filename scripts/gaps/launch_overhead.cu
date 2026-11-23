// Per-kernel boundary cost on this GPU: a chain of small kernels in a CUDA graph, with and
// without programmatic dependent launch, against one persistent kernel that separates the
// same steps with grid barriers. Also the fixed cost of a DRAM-streaming kernel.
//
// build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a launch_overhead.cu -o lo
// usage: ./lo
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// One step: every block reads the previous step's value and block 0 writes the next one,
// so each step depends on the one before.
__global__ void step_kernel(int* x, int i) {
    pdl_trigger();
    pdl_wait();
    int v = *(volatile int*)&x[i];
    if (blockIdx.x == 0 && threadIdx.x == 0) x[i + 1] = v + 1;
}

// The same steps in one kernel, with a sense-reversing grid barrier between them.
__global__ void persistent_kernel(int* x, int steps, unsigned* bar) {
    unsigned nb = gridDim.x;
    for (int i = 0; i < steps; i++) {
        int v = *(volatile int*)&x[i];
        if (blockIdx.x == 0 && threadIdx.x == 0) x[i + 1] = v + 1;
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned gen = *(volatile unsigned*)&bar[1];
            __threadfence();
            if (atomicAdd(&bar[0], 1) == nb - 1) {
                bar[0] = 0;
                __threadfence();
                atomicAdd(&bar[1], 1);
            } else {
                while (*(volatile unsigned*)&bar[1] == gen) {}
            }
            __threadfence();
        }
        __syncthreads();
    }
}

// The same with a monotonic arrival counter: one atomic per block per barrier, then a spin
// until every block of this step has arrived.
__global__ void persistent_counter_kernel(int* x, int steps, unsigned* bar) {
    unsigned nb = gridDim.x;
    for (int i = 0; i < steps; i++) {
        int v = *(volatile int*)&x[i];
        if (blockIdx.x == 0 && threadIdx.x == 0) x[i + 1] = v + 1;
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned target = nb * (i + 1);
            asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(bar + 2) : "memory");
            unsigned seen;
            do {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(bar + 2) : "memory");
            } while (seen < target);
        }
        __syncthreads();
    }
}

// Reads `n` 16-byte vectors from `src` and writes one value per block, so the loads stay.
__global__ void stream_kernel(const int4* __restrict__ src, size_t n, int* out, int pdl) {
    if (pdl) { pdl_trigger(); pdl_wait(); }
    int acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int4 v = __ldcs(&src[i]);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) out[blockIdx.x] = acc;
}

template <class F>
float time_graph(cudaStream_t s, F enqueue, int reps = 30) {
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    enqueue();
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        CK(cudaEventRecord(a, s));
        CK(cudaGraphLaunch(ge, s));
        CK(cudaEventRecord(b, s));
        CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, ms);
    }
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    return best * 1e3f;  // us
}

void launch(cudaStream_t s, bool pdl, dim3 grid, dim3 block, void (*k)(int*, int), int* x, int i) {
    cudaLaunchConfig_t c = {};
    c.gridDim = grid; c.blockDim = block; c.stream = s;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    c.attrs = at; c.numAttrs = pdl ? 1 : 0;
    CK(cudaLaunchKernelEx(&c, k, x, i));
}

int main() {
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const int N = 400;
    int* x; CK(cudaMalloc(&x, (N + 2) * sizeof(int)));
    unsigned* bar; CK(cudaMalloc(&bar, 16)); CK(cudaMemset(bar, 0, 16));
    printf("chain of %d dependent steps, us per step (min of 30 graph launches)\n", N);
    printf("%8s %10s %10s %12s %12s\n", "blocks", "plain", "pdl", "persistent", "counter");
    for (int blocks : {1, 170, 340, 680}) {
        float plain = time_graph(s, [&] { for (int i = 0; i < N; i++) launch(s, false, blocks, 256, step_kernel, x, i); });
        float pdl = time_graph(s, [&] { for (int i = 0; i < N; i++) launch(s, true, blocks, 256, step_kernel, x, i); });
        float pers = time_graph(s, [&] { persistent_kernel<<<blocks, 256, 0, s>>>(x, N, bar); });
        float cnt = time_graph(s, [&] {
            CK(cudaMemsetAsync(bar + 2, 0, 4, s));
            persistent_counter_kernel<<<blocks, 256, 0, s>>>(x, N, bar);
        });
        printf("%8d %10.3f %10.3f %12.3f %12.3f\n", blocks, plain / N, pdl / N, pers / N, cnt / N);
    }

    // Streaming kernels over distinct slices of a buffer larger than L2.
    const size_t arena = 1536ull << 20;
    int4* buf; CK(cudaMalloc(&buf, arena)); CK(cudaMemset(buf, 1, arena));
    int* out; CK(cudaMalloc(&out, 4096 * sizeof(int)));
    printf("\nDRAM streaming kernels, 680 blocks x 256 threads, chain of kernels each reading a fresh slice\n");
    printf("%10s %10s %10s %10s %10s\n", "MB each", "us plain", "us pdl", "TB/s plain", "TB/s pdl");
    for (size_t mb : {1, 2, 4, 8, 16, 32, 64}) {
        size_t bytes = mb << 20, n = bytes / 16;
        int count = (int)std::min<size_t>(64, arena / bytes);
        float t[2];
        for (int pdl = 0; pdl < 2; pdl++) {
            t[pdl] = time_graph(s, [&] {
                for (int i = 0; i < count; i++) {
                    cudaLaunchConfig_t c = {};
                    c.gridDim = 680; c.blockDim = 256; c.stream = s;
                    cudaLaunchAttribute at[1];
                    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                    at[0].val.programmaticStreamSerializationAllowed = 1;
                    c.attrs = at; c.numAttrs = pdl;
                    CK(cudaLaunchKernelEx(&c, stream_kernel, (const int4*)((char*)buf + i * bytes), n, out, pdl));
                }
            }, 10) / count;
        }
        printf("%10zu %10.2f %10.2f %10.3f %10.3f\n", mb, t[0], t[1], bytes / t[0] / 1e6, bytes / t[1] / 1e6);
    }
    return 0;
}
