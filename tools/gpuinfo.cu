// Device properties and microbenchmarks that set the decode ceiling: streaming read
// bandwidth, bandwidth at kernel-sized reads, L2 bandwidth and launch cost.
//
// nvcc -O3 -arch=sm_120a -o gpuinfo tools/gpuinfo.cu && ./gpuinfo

#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void read_kernel(const uint4* __restrict__ p, size_t n, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) *out = acc;
}
__global__ void read_cached(const uint4* __restrict__ p, size_t n, unsigned* out) {
    unsigned acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldcg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678) *out = acc;
}
__global__ void empty() {}

int attr(cudaDeviceAttr a) { int v = 0; cudaDeviceGetAttribute(&v, a, 0); return v; }

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int sms = p.multiProcessorCount;
    printf("device: %s, compute %d.%d, %d SMs\n", p.name, p.major, p.minor, sms);
    printf("global memory: %.1f GiB, bus %d bit, memory clock %d MHz (attr)\n", p.totalGlobalMem / 1073741824.0, p.memoryBusWidth, attr(cudaDevAttrMemoryClockRate) / 1000);
    printf("SM clock max: %d MHz\n", attr(cudaDevAttrClockRate) / 1000);
    printf("L2: %.1f MiB, persisting L2 max %.1f MiB\n", p.l2CacheSize / 1048576.0, p.persistingL2CacheMaxSize / 1048576.0);
    printf("per SM: %d regs, %zu KiB shared (opt-in per block %zu KiB), max %d threads, max %d blocks\n",
        p.regsPerMultiprocessor, p.sharedMemPerMultiprocessor / 1024, p.sharedMemPerBlockOptin / 1024, p.maxThreadsPerMultiProcessor, p.maxBlocksPerMultiProcessor);
    printf("per block: %d regs max, %d threads max; warp %d\n", p.regsPerBlock, p.maxThreadsPerBlock, p.warpSize);

    unsigned* out; CK(cudaMalloc(&out, 4));
    size_t big = 4ull << 30;
    uint4* buf; CK(cudaMalloc(&buf, big));
    CK(cudaMemset(buf, 1, big));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);

    // Peak streaming read over 4 GiB, several launch shapes.
    printf("\nstreaming read of 4 GiB:\n");
    for (int per_sm : {1, 2, 4, 8, 16}) for (int threads : {256, 512, 1024}) {
        int blocks = sms * per_sm;
        read_kernel<<<blocks, threads>>>(buf, big / 16, out);
        cudaEventRecord(a);
        for (int r = 0; r < 5; r++) read_kernel<<<blocks, threads>>>(buf, big / 16, out);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (threads == 512 || per_sm == 4) printf("  %3d blocks/SM x %4d threads: %7.1f GB/s\n", per_sm, threads, 5.0 * big / (ms * 1e6));
    }

    // Bandwidth at kernel-sized reads, fresh data each launch (rotating through 4 GiB).
    printf("\nsingle-kernel reads of fresh data (one launch per size, back to back):\n");
    for (size_t sz : {256ull << 10, 1ull << 20, 4718592ull, 16ull << 20, 64ull << 20, 256ull << 20, 926ull << 20}) {
        int reps = 200;
        size_t slots = big / sz;
        int blocks = (int)std::min<size_t>(sms * 8, (sz / 16 + 511) / 512);
        cudaEventRecord(a);
        for (int r = 0; r < reps; r++) read_kernel<<<blocks, 512>>>(buf + (r % slots) * (sz / 16), sz / 16, out);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        double us = ms * 1000 / reps;
        printf("  %8.2f MiB: %7.2f us per kernel, %7.1f GB/s\n", sz / 1048576.0, us, sz / (us * 1e3));
    }

    // L2-resident reads.
    printf("\nrepeated reads of one buffer (L2 when it fits):\n");
    for (size_t sz : {8ull << 20, 32ull << 20, 64ull << 20, 128ull << 20}) {
        int blocks = sms * 8;
        for (int r = 0; r < 3; r++) read_cached<<<blocks, 512>>>(buf, sz / 16, out);
        cudaEventRecord(a);
        for (int r = 0; r < 100; r++) read_cached<<<blocks, 512>>>(buf, sz / 16, out);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        printf("  %4zu MiB: %7.1f GB/s\n", sz >> 20, 100.0 * sz / (ms * 1e6));
    }

    // Launch overhead.
    printf("\nlaunch cost:\n");
    int n = 20000;
    for (int r = 0; r < 100; r++) empty<<<1, 32>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < n; r++) empty<<<1, 32>>>();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    printf("  stream launches, empty kernel: %.2f us each\n", ms * 1000 / n);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaGraph_t g; cudaGraphExec_t ge;
    int nodes = 2000;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int r = 0; r < nodes; r++) empty<<<1, 32, 0, s>>>();
    cudaStreamEndCapture(s, &g);
    CK(cudaGraphInstantiate(&ge, g, 0));
    cudaGraphLaunch(ge, s); cudaStreamSynchronize(s);
    cudaEventRecord(a, s);
    for (int r = 0; r < 10; r++) cudaGraphLaunch(ge, s);
    cudaEventRecord(b, s); cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms, a, b);
    printf("  graph of %d empty kernels: %.2f us per node\n", nodes, ms * 1000 / (10.0 * nodes));
    return 0;
}
