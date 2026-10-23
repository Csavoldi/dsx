// fuse2.cu - gemv_q8r_gu (shared gate + up + swiglu) against gemv_q8r1 x 2 + swiglu, and gemv_q8g_bf against gemv_q8g +
// round_bf16, on DeepSeek-V4.1-Flash's shapes (inter 2304 x 5120; wo_a 8 groups x 1024 rows x 4096): bit equality and
// median time of 5 interleaved rounds. nvcc -O3 -arch=sm_86 -o fuse2 bench/fuse2.cu
#include "../nvidia/src/kernels/dsx.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
static float frand(unsigned &s) { s = s * 1664525u + 1013904223u; return ((s >> 8) / 16777216.f) * 2.f - 1.f; }
template <class F> float timeit(F f) {
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0); for (int i = 0; i < 400; i++) f(i); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1); return ms * 1e3f / 400;
}
int main() {
  unsigned seed = 3;
  int bad = 0;
  {  // shared expert
    const int inter = 2304, k = 5120, P = 6;
    size_t wb = (size_t)inter * k, db = (size_t)inter * (k / 32) * 2;
    int8_t *gq, *uq; __half *gd, *ud; float *x, *g, *u, *h0, *h1;
    CK(cudaMalloc(&gq, wb * P)); CK(cudaMalloc(&uq, wb * P)); CK(cudaMalloc(&gd, db * P)); CK(cudaMalloc(&ud, db * P));
    CK(cudaMalloc(&x, k * 4)); CK(cudaMalloc(&g, inter * 4)); CK(cudaMalloc(&u, inter * 4)); CK(cudaMalloc(&h0, inter * 4)); CK(cudaMalloc(&h1, inter * 4));
    std::vector<int8_t> hq(wb); std::vector<__half> hd(db / 2); std::vector<float> hx(k);
    for (int p = 0; p < P; p++) {
      for (auto &v : hq) v = (int8_t)(rand() % 255 - 127); CK(cudaMemcpy(gq + wb * p, hq.data(), wb, cudaMemcpyHostToDevice));
      for (auto &v : hq) v = (int8_t)(rand() % 255 - 127); CK(cudaMemcpy(uq + wb * p, hq.data(), wb, cudaMemcpyHostToDevice));
      for (auto &v : hd) v = __float2half(0.002f + (rand() % 100) * 2e-5f); CK(cudaMemcpy((char *)gd + db * p, hd.data(), db, cudaMemcpyHostToDevice));
      for (auto &v : hd) v = __float2half(0.002f + (rand() % 100) * 2e-5f); CK(cudaMemcpy((char *)ud + db * p, hd.data(), db, cudaMemcpyHostToDevice));
    }
    for (auto &v : hx) v = frand(seed);
    CK(cudaMemcpy(x, hx.data(), k * 4, cudaMemcpyHostToDevice));
    auto sep = [&](int i) {
      int p = i % P;
      gemv_q8r1<<<inter / 4, 128>>>(gq + wb * p, x, g, inter, k, 1, k, inter, (const __half *)((char *)gd + db * p));
      gemv_q8r1<<<inter / 4, 128>>>(uq + wb * p, x, u, inter, k, 1, k, inter, (const __half *)((char *)ud + db * p));
      swiglu<<<(inter + 255) / 256, 256>>>(g, u, inter, nullptr, h0, inter, 10.f);
    };
    auto fus = [&](int i) {
      int p = i % P;
      gemv_q8r_gu<<<inter / 4, 128>>>(gq + wb * p, (const __half *)((char *)gd + db * p), uq + wb * p, (const __half *)((char *)ud + db * p), x, h1, inter, k, 10.f);
    };
    sep(0); fus(0); CK(cudaDeviceSynchronize());
    std::vector<float> a(inter), b(inter);
    CK(cudaMemcpy(a.data(), h0, inter * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), h1, inter * 4, cudaMemcpyDeviceToHost));
    int n = 0; for (int i = 0; i < inter; i++) n += memcmp(&a[i], &b[i], 4) != 0;
    bad += n;
    for (int i = 0; i < 3000; i++) { sep(i); fus(i); }
    std::vector<float> ts, tf; for (int r = 0; r < 5; r++) { ts.push_back(timeit(sep)); tf.push_back(timeit(fus)); }
    std::sort(ts.begin(), ts.end()); std::sort(tf.begin(), tf.end());
    printf("shared gate+up+swiglu: %d of %d differ | 3 launches %.2f us, gemv_q8r_gu %.2f us\n", n, inter, ts[2], tf[2]);
  }
  {  // wo_a
    const int groups = 8, olora = 1024, gsz = 4096, rows = groups * olora, P = 4;
    size_t wb = (size_t)rows * gsz, db = (size_t)rows * (gsz / 32) * 2;
    int8_t *q; __half *d; float *x, *y0, *y1;
    CK(cudaMalloc(&q, wb * P)); CK(cudaMalloc(&d, db * P)); CK(cudaMalloc(&x, groups * gsz * 4)); CK(cudaMalloc(&y0, rows * 4)); CK(cudaMalloc(&y1, rows * 4));
    std::vector<int8_t> hq(wb); std::vector<__half> hd(db / 2); std::vector<float> hx(groups * gsz);
    for (int p = 0; p < P; p++) {
      for (auto &v : hq) v = (int8_t)(rand() % 255 - 127); CK(cudaMemcpy(q + wb * p, hq.data(), wb, cudaMemcpyHostToDevice));
      for (auto &v : hd) v = __float2half(0.002f + (rand() % 100) * 2e-5f); CK(cudaMemcpy((char *)d + db * p, hd.data(), db, cudaMemcpyHostToDevice));
    }
    for (auto &v : hx) v = frand(seed);
    CK(cudaMemcpy(x, hx.data(), groups * gsz * 4, cudaMemcpyHostToDevice));
    auto sep = [&](int i) {
      int p = i % P;
      gemv_q8g<<<rows / 4, 128>>>(q + wb * p, x, y0, rows, gsz, olora, gsz, (const __half *)((char *)d + db * p));
      round_bf16<<<(rows + 255) / 256, 256>>>(y0, rows);
    };
    auto fus = [&](int i) { int p = i % P; gemv_q8g_bf<<<rows / 4, 128>>>(q + wb * p, x, y1, rows, gsz, olora, gsz, (const __half *)((char *)d + db * p)); };
    sep(0); fus(0); CK(cudaDeviceSynchronize());
    std::vector<float> a(rows), b(rows);
    CK(cudaMemcpy(a.data(), y0, rows * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), y1, rows * 4, cudaMemcpyDeviceToHost));
    int n = 0; for (int i = 0; i < rows; i++) n += memcmp(&a[i], &b[i], 4) != 0;
    bad += n;
    for (int i = 0; i < 2000; i++) { sep(i); fus(i); }
    std::vector<float> ts, tf; for (int r = 0; r < 5; r++) { ts.push_back(timeit(sep)); tf.push_back(timeit(fus)); }
    std::sort(ts.begin(), ts.end()); std::sort(tf.begin(), tf.end());
    printf("wo_a + round_bf16:     %d of %d differ | 2 launches %.2f us, gemv_q8g_bf %.2f us\n", n, rows, ts[2], tf[2]);
  }
  printf(bad ? "MISMATCH\n" : "all bit-identical\n");
  return bad != 0;
}
