#include <cstring>
// q8_gemv.cu - decode's Q8_0 GEMV (gemv_q8r: one warp per row) next to gemv_q8r4 (4 rows per warp, one load of x
// for all 4) on DeepSeek-V4.1-Flash's dense shapes, T = 1: GB/s of weights read, and whether outputs are identical.
// Weights rotate over a pool larger than L2. build: nvcc -O3 -arch=sm_86 -I DIR (q8k.cuh: both kernels from dsx.cu)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "q8k.cuh"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
int main() {
  struct S { const char *name; int rows, k; } shapes[] = {
      {"q_a 5120->1280", 1280, 5120}, {"q_b 1280->32768", 32768, 1280}, {"kv 5120->512", 512, 5120},
      {"o_b 8192->5120", 5120, 8192}, {"sh gate/up 5120->2304", 2304, 5120}, {"sh down 2304->5120", 5120, 2304}};
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (auto &s : shapes) {
    const size_t wb = (size_t)s.rows * s.k, db = (size_t)s.rows * (s.k / 32) * 2;
    const int pool = (int)((256ull << 20) / (wb + db)) + 2;
    int8_t *q; __half *d; float *x, *y0, *y1;
    CK(cudaMalloc(&q, wb * pool)); CK(cudaMalloc(&d, db * pool)); CK(cudaMalloc(&x, s.k * 4)); CK(cudaMalloc(&y0, s.rows * 4)); CK(cudaMalloc(&y1, s.rows * 4));
    std::vector<int8_t> hq(wb); for (auto &v : hq) v = (int8_t)(rand() % 255 - 127);
    std::vector<__half> hd(db / 2); for (auto &v : hd) v = __float2half(0.01f + (rand() % 100) * 1e-4f);
    std::vector<float> hx(s.k); for (auto &v : hx) v = (rand() % 2001) / 1000.f - 1.f;
    for (int p = 0; p < pool; p++) { CK(cudaMemcpy(q + wb * p, hq.data(), wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy((char *)d + db * p, hd.data(), db, cudaMemcpyHostToDevice)); }
    CK(cudaMemcpy(x, hx.data(), s.k * 4, cudaMemcpyHostToDevice));
    float ms[2];
    for (int v = 0; v < 2; v++) {
      auto run = [&](int i) {
        const int8_t *qq = q + wb * (i % pool); const __half *dd = (const __half *)((char *)d + db * (i % pool));
        if (v == 0) gemv_q8r<<<(s.rows + 3) / 4, 128>>>(qq, x, y0, s.rows, s.k, 1, s.k, s.rows, dd);
        else gemv_q8r4<<<(s.rows + 15) / 16, 128>>>(qq, x, y1, s.rows, s.k, 1, s.k, s.rows, dd);
      };
      for (int i = 0; i < 20; i++) run(i);
      cudaEventRecord(e0); for (int i = 0; i < 400; i++) run(i); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
      cudaEventElapsedTime(&ms[v], e0, e1); ms[v] /= 400;
    }
    std::vector<float> a(s.rows), b(s.rows);
    CK(cudaMemcpy(a.data(), y0, s.rows * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), y1, s.rows * 4, cudaMemcpyDeviceToHost));
    int diff = 0; for (int i = 0; i < s.rows; i++) diff += memcmp(&a[i], &b[i], 4) != 0;
    printf("%-24s q8r %6.1f us %4.0f GB/s | q8r4 %6.1f us %4.0f GB/s | %.2fx, %d of %d outputs differ\n", s.name, ms[0] * 1e3, (wb + db) / ms[0] / 1e6,
           ms[1] * 1e3, (wb + db) / ms[1] / 1e6, ms[0] / ms[1], diff, s.rows);
    cudaFree(q); cudaFree(d); cudaFree(x); cudaFree(y0); cudaFree(y1);
  }
}
