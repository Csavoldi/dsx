// q8h_gemv.cu - decode's Q8_0 GEMV gemv_q8r against the T = 1 variants with hoisted loads (gemv_q8h1: 4 loads per lane
// in flight, gemv_q8h1u2: 2, gemv_q8r1: 1, the production one) on DeepSeek-V4.1-Flash's dense shapes: bit equality, and GB/s of weights
// read (median of 5 interleaved rounds of 300 launches; weights rotate over a pool larger than L2).
// nvcc -O3 -arch=sm_86 -o q8h bench/q8h_gemv.cu   (includes the production kernels from nvidia/src/kernels/dsx.cu)
#include "../nvidia/src/kernels/dsx.cu"
#include <algorithm>
extern "C" __global__ void gemv_q8h1(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y,
                                     int rows, int k, int T, int ldx, int ldy, const __half *__restrict__ d) {
  gemv_q8h_body<1, 4>(qs, x, y, rows, k, 1, ldx, ldy, d);
}
extern "C" __global__ void gemv_q8h1u2(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y,
                                       int rows, int k, int T, int ldx, int ldy, const __half *__restrict__ d) {
  gemv_q8h_body<1, 2>(qs, x, y, rows, k, 1, ldx, ldy, d);
}
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
typedef void (*K)(const int8_t *, const float *, float *, int, int, int, int, int, const __half *);
int main() {
  struct S { const char *name; int rows, k; } shapes[] = {
      {"q_a 5120->1280", 1280, 5120}, {"q_b 1280->32768", 32768, 1280}, {"kv 5120->512", 512, 5120},
      {"o_b 8192->5120", 5120, 8192}, {"sh gate/up 5120->2304", 2304, 5120}, {"sh down 2304->5120", 5120, 2304},
      {"head 5120->129280", 129280, 5120}};
  const int NS = getenv("Q8H_NOHEAD") ? 6 : 7;
  const char *kn[] = {"q8r", "h1", "h1u2", "q8r1"};
  K ks[] = {gemv_q8r, gemv_q8h1, gemv_q8h1u2, gemv_q8r1};
  const int NK = 4, ROUNDS = 5, REPS = 300;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int bad = 0;
  for (int si = 0; si < NS; si++) {
    auto &s = shapes[si];
    const size_t wb = (size_t)s.rows * s.k, db = (size_t)s.rows * (s.k / 32) * 2;
    const int pool = std::max(2, (int)((256ull << 20) / (wb + db)) + 2);
    int8_t *q; __half *d; float *x, *y[NK];
    CK(cudaMalloc(&q, wb * pool)); CK(cudaMalloc(&d, db * pool)); CK(cudaMalloc(&x, s.k * 4));
    for (int v = 0; v < NK; v++) CK(cudaMalloc(&y[v], s.rows * 4));
    std::vector<int8_t> hq(wb); for (auto &v : hq) v = (int8_t)(rand() % 255 - 127);
    std::vector<__half> hd(db / 2); for (auto &v : hd) v = __float2half(0.01f + (rand() % 100) * 1e-4f);
    std::vector<float> hx(s.k); for (auto &v : hx) v = (rand() % 2001) / 1000.f - 1.f;
    for (int p = 0; p < pool; p++) { CK(cudaMemcpy(q + wb * p, hq.data(), wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy((char *)d + db * p, hd.data(), db, cudaMemcpyHostToDevice)); }
    CK(cudaMemcpy(x, hx.data(), s.k * 4, cudaMemcpyHostToDevice));
    auto run = [&](int v, int i) {
      const int8_t *qq = q + wb * (i % pool); const __half *dd = (const __half *)((char *)d + db * (i % pool));
      ks[v]<<<(s.rows + 3) / 4, 128>>>(qq, x, y[v], s.rows, s.k, 1, s.k, s.rows, dd);
    };
    std::vector<float> t[NK];
    for (int i = 0; i < 4000; i++) run(i % NK, i);  // ~1 s: clocks up
    for (int r = 0; r < ROUNDS; r++)
      for (int v = 0; v < NK; v++) {
        cudaEventRecord(e0); for (int i = 0; i < REPS; i++) run(v, i); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1); t[v].push_back(ms / REPS);
      }
    // equality on the same weights (pool slot 0)
    for (int v = 0; v < NK; v++) run(v, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> a(s.rows), b(s.rows);
    CK(cudaMemcpy(a.data(), y[0], s.rows * 4, cudaMemcpyDeviceToHost));
    printf("%-22s", s.name);
    for (int v = 0; v < NK; v++) {
      std::sort(t[v].begin(), t[v].end());
      float ms = t[v][ROUNDS / 2];
      CK(cudaMemcpy(b.data(), y[v], s.rows * 4, cudaMemcpyDeviceToHost));
      int diff = 0; for (int i = 0; i < s.rows; i++) diff += memcmp(&a[i], &b[i], 4) != 0;
      bad += diff;
      printf(" | %s %6.1f us %4.0f GB/s%s", kn[v], ms * 1e3, (wb + db) / ms / 1e6, diff ? " DIFF" : "");
    }
    printf("\n");
    cudaFree(q); cudaFree(d); cudaFree(x); for (int v = 0; v < NK; v++) cudaFree(y[v]);
  }
  printf(bad ? "MISMATCH\n" : "all variants bit-identical to gemv_q8r\n");
  return bad != 0;
}
