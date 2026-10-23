// router_publish (router_topk + publish_route in one launch) against the two launches: bit equality of the picks,
// weights and the published exchange (mapped host memory, as in production) over many random tokens, and launch time.
// nvcc -O3 -arch=sm_86 -o router_fuse bench/router_fuse.cu
#include "../nvidia/src/kernels/dsx.cu"
#include <cstdio>
#include <cstring>
#include <vector>

static float frand(unsigned &s) { s = s * 1664525u + 1013904223u; return ((s >> 8) / 16777216.f) * 2.f - 1.f; }

int main() {
  const int E = 384, k = 6, dim = 5120, XB = 1 << 20, reps = 2000;
  unsigned seed = 11;
  float *lg, *bias, *x, *w1, *w2;
  int *i1, *i2;
  cudaMalloc(&lg, E * 4); cudaMalloc(&bias, E * 4); cudaMalloc(&x, dim * 4);
  cudaMalloc(&i1, 64); cudaMalloc(&i2, 64); cudaMalloc(&w1, 64); cudaMalloc(&w2, 64);
  uint8_t *h1, *h2, *d1, *d2;
  cudaHostAlloc(&h1, XB, cudaHostAllocMapped); cudaHostAlloc(&h2, XB, cudaHostAllocMapped);
  memset(h1, 0, XB); memset(h2, 0, XB);
  cudaHostGetDevicePointer(&d1, h1, 0); cudaHostGetDevicePointer(&d2, h2, 0);
  std::vector<float> L(E), B(E), X(dim);
  int bad = 0, trials = 300;
  for (int tr = 0; tr < trials; tr++) {
    for (auto &v : L) v = frand(seed) * (tr % 3 == 0 ? 3 : 12);
    for (auto &v : B) v = frand(seed) * 0.5f;
    if (tr % 7 == 0) for (int e = 0; e < E; e += 5) L[e] = L[0];  // ties
    for (auto &v : X) v = frand(seed);
    cudaMemcpy(lg, L.data(), E * 4, cudaMemcpyHostToDevice); cudaMemcpy(bias, B.data(), E * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(x, X.data(), dim * 4, cudaMemcpyHostToDevice);
    router_topk<<<1, 32>>>(lg, bias, i1, w1, E, k, 1.5f, nullptr);
    publish_route<<<1, 1024>>>(i1, w1, k, x, dim, d1, tr + 1, 1);
    router_publish<<<1, 1024>>>(lg, bias, i2, w2, E, k, 1.5f, nullptr, x, dim, d2, tr + 1);
    cudaDeviceSynchronize();
    int a[8], b[8]; float wa[8], wb[8];
    cudaMemcpy(a, i1, k * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b, i2, k * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(wa, w1, k * 4, cudaMemcpyDeviceToHost); cudaMemcpy(wb, w2, k * 4, cudaMemcpyDeviceToHost);
    int n = memcmp(a, b, k * 4) != 0 || memcmp(wa, wb, k * 4) != 0;
    // the exchange, skipping the globaltimer stamp at +8
    n += memcmp(h1, h2, 8) != 0 || memcmp(h1 + 16, h2 + 16, XCH_X + dim * 4 - 16) != 0;
    bad += n;
  }
  printf("%d of %d tokens differ\n", bad, trials);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int v = 0; v < 2; v++)
    for (int it = 0; it < 2; it++) {
      cudaEventRecord(e0);
      for (int r = 0; r < reps; r++) {
        if (v == 0) {
          router_topk<<<1, 32>>>(lg, bias, i1, w1, E, k, 1.5f, nullptr);
          publish_route<<<1, 1024>>>(i1, w1, k, x, dim, d1, r, 1);
        } else {
          router_publish<<<1, 1024>>>(lg, bias, i2, w2, E, k, 1.5f, nullptr, x, dim, d2, r);
        }
      }
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      if (it) printf("  %s: %.2f us per token\n", v ? "router_publish (1 launch)" : "router_topk + publish_route (2)", ms * 1e3 / reps);
    }
  printf(bad ? "MISMATCH\n" : "all bit-identical\n");
  return bad != 0;
}
