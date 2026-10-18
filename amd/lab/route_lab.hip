// route_lab: where does a one-wave top-6 spend 10 us? wall_clock64 (100 MHz) between phases.
#include "../kernels/dsxa.hip"
#include <cstdio>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)
__global__ void lab(const float* logits, const float* bias, long long* t, float* out) {
    const int lane = threadIdx.x;
    long long t0 = wall_clock64();
    float l[6], b[6];
    for (int j = 0; j < 6; j++) { l[j] = logits[lane + 64 * j]; b[j] = bias[lane + 64 * j]; }
    float acc = 0;
    for (int j = 0; j < 6; j++) acc += l[j] + b[j];
    long long t1 = wall_clock64();
    float sc[6];
    for (int j = 0; j < 6; j++) { const float sp = l[j] > 20.f ? l[j] : log1pf(expf(l[j])); sc[j] = sqrtf(sp) + acc * 0; }
    long long t2 = wall_clock64();
    float s = 0;
    for (int j = 0; j < 6; j++) s += __shfl_xor(sc[j], 1 + j);
    long long t3 = wall_clock64();
    float d = 0;
    for (int j = 0; j < 6; j++) d += dpp<DPP_XOR1>(sc[j] + d);
    long long t4 = wall_clock64();
    if (lane == 0) { t[0] = t1 - t0; t[1] = t2 - t1; t[2] = t3 - t2; t[3] = t4 - t3; out[0] = s + d; }
}
int main() {
    float *l, *b, *o; long long* t;
    CK(hipMalloc(&l, 4096)); CK(hipMalloc(&b, 4096)); CK(hipMalloc(&o, 64)); CK(hipMalloc(&t, 64));
    CK(hipMemset(l, 0, 4096)); CK(hipMemset(b, 0, 4096));
    for (int it = 0; it < 3; it++) {
        lab<<<1, 64>>>(l, b, t, o);
        CK(hipDeviceSynchronize());
        long long h[4]; CK(hipMemcpy(h, t, 32, hipMemcpyDeviceToHost));
        printf("loads %lld  math %lld  6 shfl %lld  6 dpp %lld  (x10 ns)\n", h[0], h[1], h[2], h[3]);
    }
}
