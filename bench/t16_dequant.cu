// t16_dequant.cu - prefill's T16 -> f16 expansion (t16.cuh dequant_t16_f16, one thread per 16-weight tile row) next to
// the previous one-thread-per-weight kernel (old.cuh, from git), stored and GPU layouts, K 1.5 / 2 / 2.5, on a
// 4096 x 2048 expert matrix. Checks identical output; reports f16 written per second.
// build: nvcc -O3 -arch=sm_86 -I DIR (t16pre.cuh, old.cuh) -o t16_dequant bench/t16_dequant.cu
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_fp16.h>
#include "t16pre.cuh"  // t16.cuh up to its decode section (sed -n '1,/decode/p')
#include "old.cuh"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)
int main() {
  const int k = 4096, n = 2048, reps = 200;
  unsigned char *w; __half *a, *b;
  CK(cudaMalloc(&w, (size_t)k * n)); CK(cudaMalloc(&a, (size_t)k * n * 2)); CK(cudaMalloc(&b, (size_t)k * n * 2));
  std::vector<unsigned char> h((size_t)k * n); for (auto &x : h) x = rand();
  CK(cudaMemcpy(w, h.data(), h.size(), cudaMemcpyHostToDevice));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int k2 = 3; k2 <= 5; k2++) for (int rp = 0; rp < 2; rp++) {
    float ms[2];
    for (int v = 0; v < 2; v++) {
      auto run = [&] { if (v) dequant_t16_f16<<<dim3((k + 511) / 512, n / 16), 512>>>(k2, w, b, n, k, rp); else dequant_t16_f16_old<<<dim3((k + 255) / 256, n), 256>>>(k2, w, a, n, k, rp); };
      run(); cudaEventRecord(e0); for (int i = 0; i < reps; i++) run(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
      cudaEventElapsedTime(&ms[v], e0, e1); ms[v] /= reps;
    }
    std::vector<uint16_t> ha((size_t)k * n), hb((size_t)k * n);
    CK(cudaMemcpy(ha.data(), a, ha.size() * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb.data(), b, hb.size() * 2, cudaMemcpyDeviceToHost));
    size_t bad = 0; for (size_t i = 0; i < ha.size(); i++) bad += ha[i] != hb[i];
    printf("K %.1f %s: old %6.1f us (%4.0f GB/s)  new %6.1f us (%4.0f GB/s)  %.1fx  mismatches %zu\n", k2 / 2.0, rp ? "gpu   " : "stored",
           ms[0] * 1e3, k * (double)n * 2 / ms[0] / 1e6, ms[1] * 1e3, k * (double)n * 2 / ms[1] / 1e6, ms[0] / ms[1], bad);
  }
}
