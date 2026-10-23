// launch_bench.hip: the per-launch floor on the MI325X for the runner's small kernels (RUNNER.md item 2's confirming
// measurement): back-to-back eager launches on one stream of (a) an empty kernel, (b) a 40-block kernel with one
// dependent global load per thread, (c) the same with 4 dependent loads, at 1 and 256 threads per block.
// Build: hipcc --offload-arch=gfx942 -O3 launch_bench.hip -o launch_bench
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k_empty() {}
__global__ void k_load(const float* __restrict__ x, float* y, int dep) {
    float v = x[threadIdx.x];
    for (int i = 0; i < dep; i++) v = x[((int)v + threadIdx.x + i) & 1023];
    if (v == 12345.f) y[0] = v;
}
int main() {
    float *x, *y;
    hipMalloc(&x, 4096 * 4);
    hipMalloc(&y, 4);
    hipMemset(x, 0, 4096 * 4);
    hipStream_t s;
    hipStreamCreate(&s);
    hipEvent_t a, b;
    hipEventCreate(&a);
    hipEventCreate(&b);
    auto run = [&](const char* name, auto f) {
        for (int i = 0; i < 100; i++) f();
        hipEventRecord(a, s);
        const int n = 2000;
        for (int i = 0; i < n; i++) f();
        hipEventRecord(b, s);
        hipEventSynchronize(b);
        float ms;
        hipEventElapsedTime(&ms, a, b);
        printf("%-34s %6.2f us per launch\n", name, ms * 1e3 / n);
    };
    run("empty 1x64", [&] { hipLaunchKernelGGL(k_empty, dim3(1), dim3(64), 0, s); });
    run("empty 40x256", [&] { hipLaunchKernelGGL(k_empty, dim3(40), dim3(256), 0, s); });
    run("empty 1024x256", [&] { hipLaunchKernelGGL(k_empty, dim3(1024), dim3(256), 0, s); });
    run("1 load 40x256", [&] { hipLaunchKernelGGL(k_load, dim3(40), dim3(256), 0, s, x, y, 0); });
    run("4 dependent loads 40x256", [&] { hipLaunchKernelGGL(k_load, dim3(40), dim3(256), 0, s, x, y, 3); });
    run("8 dependent loads 40x256", [&] { hipLaunchKernelGGL(k_load, dim3(40), dim3(256), 0, s, x, y, 7); });
    return 0;
}
