#include "../kernels/dsxa.hip"
#include <cstdio>
__global__ void lab(const float* logits, const float* bias, long long* t, Route* r) {
    __shared__ Route rs;
    __shared__ float L[384], B[384];
    for (int i = threadIdx.x; i < 384; i += 64) { L[i] = logits[i]; B[i] = bias[i]; }
    __syncthreads();
    long long t0 = wall_clock64();
    route_top6(L, B, 1.5f, &rs, threadIdx.x);
    long long t1 = wall_clock64();
    if (threadIdx.x == 0) { t[0] = t1 - t0; r->ids[0] = rs.ids[0]; }
}
int main() {
    float *l, *b; long long* t; Route* r;
    hipMalloc(&l, 4096); hipMalloc(&b, 4096); hipMalloc(&t, 64); hipMalloc(&r, sizeof(Route));
    float h[384]; for (int i = 0; i < 384; i++) h[i] = (i * 37 % 101) * 0.01f;
    hipMemcpy(l, h, 1536, hipMemcpyHostToDevice); hipMemcpy(b, h, 1536, hipMemcpyHostToDevice);
    for (int it = 0; it < 3; it++) {
        lab<<<1, 64>>>(l, b, t, r); hipDeviceSynchronize();
        long long x; hipMemcpy(&x, t, 8, hipMemcpyDeviceToHost); printf("route_top6: %lld0 ns\n", x);
    }
}
