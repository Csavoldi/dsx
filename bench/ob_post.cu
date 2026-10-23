// ob_post.cu - gemv_q8r1_post (o_b with the attention half's hc_post in the epilogue) against gemv_q8r1 + hc_post_mb on
// o_b's shape (5120 x 8192, 4 residual streams): bit equality of y and h, and median time of 5 interleaved rounds.
// nvcc -O3 -arch=sm_86 -o ob_post bench/ob_post.cu
#include "../nvidia/src/kernels/dsx.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static float frand(unsigned &s) { s = s * 1664525u + 1013904223u; return ((s >> 8) / 16777216.f) * 2.f - 1.f; }
static float bf(float v) { return __bfloat162float(__float2bfloat16(v)); }
int main() {
  const int rows = 5120, k = 8192, P = 4;
  unsigned seed = 4;
  size_t wb = (size_t)rows * k, db = (size_t)rows * (k / 32) * 2;
  int8_t *q; __half *d; float *x, *y0, *y1, *h0, *h1, *m;
  cudaMalloc(&q, wb * P); cudaMalloc(&d, db * P); cudaMalloc(&x, k * 4); cudaMalloc(&y0, rows * 4); cudaMalloc(&y1, rows * 4);
  cudaMalloc(&h0, 4 * rows * 4); cudaMalloc(&h1, 4 * rows * 4); cudaMalloc(&m, 24 * 4);
  std::vector<int8_t> hq(wb); std::vector<__half> hd(db / 2); std::vector<float> hx(k), hh(4 * rows), hm(24);
  for (int p = 0; p < P; p++) {
    for (auto &v : hq) v = (int8_t)(rand() % 255 - 127); cudaMemcpy(q + wb * p, hq.data(), wb, cudaMemcpyHostToDevice);
    for (auto &v : hd) v = __float2half(0.001f + (rand() % 100) * 1e-5f); cudaMemcpy((char *)d + db * p, hd.data(), db, cudaMemcpyHostToDevice);
  }
  for (auto &v : hx) v = frand(seed);
  for (auto &v : hh) v = bf(frand(seed) * 4);
  for (auto &v : hm) v = frand(seed);
  cudaMemcpy(x, hx.data(), k * 4, cudaMemcpyHostToDevice); cudaMemcpy(m, hm.data(), 96, cudaMemcpyHostToDevice);
  cudaMemcpy(h0, hh.data(), 16 * rows, cudaMemcpyHostToDevice); cudaMemcpy(h1, hh.data(), 16 * rows, cudaMemcpyHostToDevice);
  gemv_q8r1<<<rows / 4, 128>>>(q, x, y0, rows, k, 1, k, rows, d);
  hc_post_mb<<<(rows + 255) / 256, 256>>>(h0, y0, m, rows);
  gemv_q8r1_post<<<rows / 4, 128>>>(q, x, y1, rows, k, d, h1, m);
  std::vector<float> a(4 * rows), b(4 * rows);
  cudaMemcpy(a.data(), y0, rows * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), y1, rows * 4, cudaMemcpyDeviceToHost);
  int ny = 0; for (int i = 0; i < rows; i++) ny += memcmp(&a[i], &b[i], 4) != 0;
  cudaMemcpy(a.data(), h0, 16 * rows, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), h1, 16 * rows, cudaMemcpyDeviceToHost);
  int nh = 0; for (int i = 0; i < 4 * rows; i++) nh += memcmp(&a[i], &b[i], 4) != 0;
  printf("y differs in %d of %d, h in %d of %d\n", ny, rows, nh, 4 * rows);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto sep = [&](int i) { int p = i % P; gemv_q8r1<<<rows / 4, 128>>>(q + wb * p, x, y0, rows, k, 1, k, rows, (const __half *)((char *)d + db * p)); hc_post_mb<<<(rows + 255) / 256, 256>>>(h0, y0, m, rows); };
  auto fus = [&](int i) { int p = i % P; gemv_q8r1_post<<<rows / 4, 128>>>(q + wb * p, x, y1, rows, k, (const __half *)((char *)d + db * p), h1, m); };
  for (int i = 0; i < 2000; i++) { sep(i); fus(i); }
  std::vector<float> ts, tf;
  for (int r = 0; r < 5; r++) for (int v = 0; v < 2; v++) {
    cudaEventRecord(e0); for (int i = 0; i < 400; i++) (v ? (void)fus(i) : (void)sep(i)); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); (v ? tf : ts).push_back(ms * 1e3f / 400);
  }
  std::sort(ts.begin(), ts.end()); std::sort(tf.begin(), tf.end());
  printf("gemv_q8r1 + hc_post_mb %.2f us, gemv_q8r1_post %.2f us\n", ts[2], tf[2]);
  bool bad = ny || nh;
  printf(bad ? "MISMATCH\n" : "all bit-identical\n");
  return bad;
}
