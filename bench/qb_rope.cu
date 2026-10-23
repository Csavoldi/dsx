// qb_rope.cu - gemv_q8r1_rope (q_b with RoPE in the epilogue) against gemv_q8r1 + rope on q_b's shape (32768 x 1280,
// 64 heads x 512, rope dim 64): bit equality at several positions and median time of 5 interleaved rounds.
// nvcc -O3 -arch=sm_86 -o qb_rope bench/qb_rope.cu
#include "../nvidia/src/kernels/dsx.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static float frand(unsigned &s) { s = s * 1664525u + 1013904223u; return ((s >> 8) / 16777216.f) * 2.f - 1.f; }
int main() {
  const int rows = 32768, k = 1280, H = 64, hd = 512, rd = 64, P = 8;
  unsigned seed = 2;
  size_t wb = (size_t)rows * k, db = (size_t)rows * (k / 32) * 2;
  int8_t *q; __half *d; float *x, *fr, *y0, *y1;
  cudaMalloc(&q, wb * P); cudaMalloc(&d, db * P); cudaMalloc(&x, k * 4); cudaMalloc(&fr, rd * 2); cudaMalloc(&y0, rows * 4); cudaMalloc(&y1, rows * 4);
  std::vector<int8_t> hq(wb); std::vector<__half> hd_(db / 2); std::vector<float> hx(k), hf(rd / 2);
  for (int p = 0; p < P; p++) {
    for (auto &v : hq) v = (int8_t)(rand() % 255 - 127); cudaMemcpy(q + wb * p, hq.data(), wb, cudaMemcpyHostToDevice);
    for (auto &v : hd_) v = __float2half(0.002f + (rand() % 100) * 2e-5f); cudaMemcpy((char *)d + db * p, hd_.data(), db, cudaMemcpyHostToDevice);
  }
  for (auto &v : hx) v = frand(seed);
  for (int i = 0; i < rd / 2; i++) hf[i] = powf(10000.f, -2.f * i / rd);
  cudaMemcpy(x, hx.data(), k * 4, cudaMemcpyHostToDevice); cudaMemcpy(fr, hf.data(), rd * 2, cudaMemcpyHostToDevice);
  int bad = 0;
  std::vector<float> a(rows), b(rows);
  for (int pos : {0, 3, 999, 31000, 65000}) {
    gemv_q8r1<<<rows / 4, 128>>>(q, x, y0, rows, k, 1, k, rows, d);
    rope<<<dim3(1, (H * rd / 2 + 255) / 256), 256>>>(y0, fr, H, hd, rd, H * hd, pos, 0);
    gemv_q8r1_rope<<<rows / 4, 128>>>(q, x, y1, rows, k, d, fr, hd, rd, pos);
    cudaMemcpy(a.data(), y0, rows * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), y1, rows * 4, cudaMemcpyDeviceToHost);
    int n = 0; for (int i = 0; i < rows; i++) n += memcmp(&a[i], &b[i], 4) != 0;
    printf("pos %5d: %d of %d differ\n", pos, n, rows);
    if (n && getenv("QB_DEBUG")) {
      std::vector<float> pre(rows);
      gemv_q8r1<<<rows / 4, 128>>>(q, x, y0, rows, k, 1, k, rows, d);
      cudaMemcpy(pre.data(), y0, rows * 4, cudaMemcpyDeviceToHost);
      for (int i = 0, shown = 0; i < rows && shown < 3; i++) if (memcmp(&a[i], &b[i], 4)) {
        int c0 = i % hd; printf("  row %d (col %d, %s) pre %a / %a  rope %a  fused %a\n", i, c0, (c0 & 1) ? "odd" : "even", pre[i & ~1], pre[i | 1], a[i], b[i]);
        shown++;
      }
    }
    bad += n;
  }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto sep = [&](int i) { int p = i % P; gemv_q8r1<<<rows / 4, 128>>>(q + wb * p, x, y0, rows, k, 1, k, rows, (const __half *)((char *)d + db * p)); rope<<<dim3(1, (H * rd / 2 + 255) / 256), 256>>>(y0, fr, H, hd, rd, H * hd, i, 0); };
  auto fus = [&](int i) { int p = i % P; gemv_q8r1_rope<<<rows / 4, 128>>>(q + wb * p, x, y1, rows, k, (const __half *)((char *)d + db * p), fr, hd, rd, i); };
  for (int i = 0; i < 2000; i++) { sep(i); fus(i); }
  std::vector<float> ts, tf;
  for (int r = 0; r < 5; r++) for (int v = 0; v < 2; v++) {
    cudaEventRecord(e0); for (int i = 0; i < 400; i++) (v ? (void)fus(i) : (void)sep(i)); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); (v ? tf : ts).push_back(ms * 1e3f / 400);
  }
  std::sort(ts.begin(), ts.end()); std::sort(tf.begin(), tf.end());
  printf("gemv_q8r1 + rope %.2f us, gemv_q8r1_rope %.2f us\n", ts[2], tf[2]);
  printf(bad ? "MISMATCH\n" : "all bit-identical\n");
  return bad != 0;
}
