// math_lab: latency of a dependent chain of 16 calls per function, one wave (x10 ns)
#include <hip/hip_runtime.h>
#include <cstdio>
#define CHAIN(name, expr) { float v = x; long long t0 = wall_clock64(); for (int i = 0; i < 16; i++) { v = expr; } long long t1 = wall_clock64(); if (threadIdx.x == 0) { t[k] = t1 - t0; } k++; acc += v; }
__global__ void lab(float x0, long long* t, float* out) {
    float x = x0 + threadIdx.x * 1e-3f, acc = 0; int k = 0;
    CHAIN(expf, expf(v * 0.1f))
    CHAIN(__expf, __expf(v * 0.1f))
    CHAIN(log1pf, log1pf(v))
    CHAIN(__logf, __logf(v + 1.f))
    CHAIN(sqrtf, sqrtf(v + 1.f))
    CHAIN(sinf, sinf(v * 1000.f))
    CHAIN(__sinf, __sinf(v))
    CHAIN(div, 1.f / (v + 1.f))
    CHAIN(rcp, __frcp_rn(v + 1.f))
    if (threadIdx.x == 0) out[0] = acc;
}
int main() {
    long long* t; float* o; hipMalloc(&t, 256); hipMalloc(&o, 64);
    const char* names[] = {"expf", "__expf", "log1pf", "__logf", "sqrtf", "sinf(x*1000)", "__sinf", "1/x", "__frcp_rn"};
    for (int it = 0; it < 2; it++) {
        lab<<<1, 64>>>(0.5f, t, o); hipDeviceSynchronize();
        long long h[9]; hipMemcpy(h, t, 72, hipMemcpyDeviceToHost);
        for (int i = 0; i < 9; i++) printf("%-14s %6.1f ns per call\n", names[i], h[i] * 10.0 / 16);
        printf("\n");
    }
}
