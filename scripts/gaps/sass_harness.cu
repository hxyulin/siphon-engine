// Loads two cubins of the engine's elementwise.cu through the driver API, runs
// softcapped_swiglu_bf16 from each on the same input, compares the outputs byte for byte
// and times each kernel (minimum of 20 launches).
//
// build: nvcc -O2 -std=c++17 sass_harness.cu -o sh -lcuda
// usage: ./sh <a.cubin> <b.cubin> [n] [blocks] [threads]
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <fstream>
#include <iterator>
#include <algorithm>

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char* s; cuGetErrorString(r, &s); \
    printf("%s:%d %s\n", __FILE__, __LINE__, s); exit(1); } } while (0)

static std::vector<char> slurp(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(f), {});
}

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s a.cubin b.cubin [n]\n", argv[0]); return 2; }
    size_t n = argc > 3 ? strtoull(argv[3], 0, 10) : (16u << 20);
    unsigned blocks = argc > 4 ? atoi(argv[4]) : 680, threads = argc > 5 ? atoi(argv[5]) : 256;
    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CK(cuCtxCreate(&ctx, nullptr, 0, dev));
    CUdeviceptr in, out[2];
    CK(cuMemAlloc(&in, n * 4));
    CK(cuMemAlloc(&out[0], n * 2)); CK(cuMemAlloc(&out[1], n * 2));
    std::vector<uint16_t> h(2 * n);
    uint32_t s = 12345;
    for (auto& v : h) {  // finite BF16 values spread over about [-64, 64]
        s = s * 1664525u + 1013904223u;
        v = uint16_t(((s >> 16) & 0x80ff) | 0x3c00 | ((s >> 8) & 0x0300));
    }
    CK(cuMemcpyHtoD(in, h.data(), n * 4));
    std::vector<uint16_t> res[2];
    for (int k = 0; k < 2; k++) {
        auto bin = slurp(argv[1 + k]);
        CUmodule mod;
        CUresult r = cuModuleLoadData(&mod, bin.data());
        if (r != CUDA_SUCCESS) { const char* e; cuGetErrorString(r, &e); printf("load %s: %s\n", argv[1 + k], e); return 1; }
        CUfunction fn; CK(cuModuleGetFunction(&fn, mod, "softcapped_swiglu_bf16"));
        float gs = 10.f, us = 30.f;
        void* args[] = {&in, &out[k], &n, &gs, &us};
        CUevent a, b; CK(cuEventCreate(&a, 0)); CK(cuEventCreate(&b, 0));
        float best = 1e30f;
        for (int i = 0; i < 21; i++) {
            CK(cuEventRecord(a, 0));
            CK(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, 0, args, nullptr));
            CK(cuEventRecord(b, 0)); CK(cuEventSynchronize(b));
            float ms; CK(cuEventElapsedTime(&ms, a, b));
            if (i) best = std::min(best, ms);
        }
        res[k].resize(n);
        CK(cuMemcpyDtoH(res[k].data(), out[k], n * 2));
        printf("%s: %.1f us (%.2f TB/s)\n", argv[1 + k], best * 1e3, n * 6 / (best * 1e-3) / 1e12);
    }
    size_t diff = 0;
    for (size_t i = 0; i < n; i++) diff += res[0][i] != res[1][i];
    printf("outputs %s: %zu of %zu BF16 values differ\n", diff ? "DIFFER" : "identical", diff, n);
    return 0;
}
