// hc_pre_norm (hc_pre + rmsnorm in one launch) against the two launches it replaces, and rope split
// across heads against one block: bit equality on random inputs and back-to-back launch time per token step.
// nvcc -O3 -arch=sm_86 -o hc_fuse bench/hc_fuse.cu   (includes the production kernels from nvidia/src/kernels/dsx.cu)
#include "../nvidia/src/kernels/dsx.cu"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static float frand(unsigned &s) { s = s * 1664525u + 1013904223u; return ((s >> 8) / 16777216.f) * 2.f - 1.f; }
static float bf(float v) { return __bfloat162float(__float2bfloat16(v)); }

int main() {
  const int NB = (5120 + 1023) / 1024;
  const int d = 5120, reps = 2000;
  unsigned seed = 7;
  std::vector<float> h(4 * d), x(d), mix(24), w(d);
  for (auto &v : h) v = bf(frand(seed) * 3);
  for (auto &v : x) v = frand(seed) * 2;
  for (auto &v : mix) v = frand(seed);
  for (auto &v : w) v = 1 + 0.1f * frand(seed);
  float *dh, *dh2, *dx, *dm, *dw, *dy, *dy2;
  cudaMalloc(&dh, 16 * d * 4); cudaMalloc(&dh2, 16 * d * 4); cudaMalloc(&dx, d * 4); cudaMalloc(&dm, 96);
  cudaMalloc(&dw, d * 4); cudaMalloc(&dy, d * 4); cudaMalloc(&dy2, d * 4);
  cudaMemcpy(dx, x.data(), d * 4, cudaMemcpyHostToDevice); cudaMemcpy(dm, mix.data(), 96, cudaMemcpyHostToDevice);
  cudaMemcpy(dw, w.data(), d * 4, cudaMemcpyHostToDevice);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int bad = 0;
  for (int post = 0; post < 1; post++)
    for (int round = 0; round < 2; round++) {
      cudaMemcpy(dh, h.data(), 4 * d * 4, cudaMemcpyHostToDevice); cudaMemcpy(dh2, h.data(), 4 * d * 4, cudaMemcpyHostToDevice);
      if (post) hc_post_mb<<<(d + 255) / 256, 256>>>(dh, dx, dm, d);
      hc_pre_mb<<<(d + 255) / 256, 256>>>(dh, dm, 0, dy, d);
      rmsnorm<<<1, 1024>>>(dy, dw, dy, d, d, d, 1e-20f, round);
      hc_pre_norm<<<NB, 1024>>>(dh2, dm, dy2, dw, d, 1e-20f, round);
      std::vector<float> a(d), b(d), ha(4 * d), hb(4 * d);
      cudaMemcpy(a.data(), dy, d * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), dy2, d * 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(ha.data(), dh, 4 * d * 4, cudaMemcpyDeviceToHost); cudaMemcpy(hb.data(), dh2, 4 * d * 4, cudaMemcpyDeviceToHost);
      int nx = 0, nh = 0;
      for (int i = 0; i < d; i++) nx += memcmp(&a[i], &b[i], 4) != 0;
      for (int i = 0; i < 4 * d; i++) nh += memcmp(&ha[i], &hb[i], 4) != 0;
      printf("post %d round %d: x differs in %d of %d, h in %d of %d\n", post, round, nx, d, nh, 4 * d);
      bad += nx + nh;
      // timing: each variant back to back, h rotating over 4 copies to keep it out of L1 like the real chain
      for (int v = 0; v < 3; v++) {
        for (int it = 0; it < 2; it++) {
          cudaEventRecord(e0);
          for (int r = 0; r < reps; r++) {
            float *hh = (v ? dh2 : dh) + (r & 3) * 0;
            if (v == 0) {
              if (post) hc_post_mb<<<(d + 255) / 256, 256>>>(hh, dx, dm, d);
              hc_pre_mb<<<(d + 255) / 256, 256>>>(hh, dm, 0, dy, d);
              rmsnorm<<<1, 1024>>>(dy, dw, dy, d, d, d, 1e-20f, round);
            } else {
              hc_pre_norm<<<v == 1 ? 1 : 5, 1024>>>(hh, dm, dy2, dw, d, 1e-20f, round);
            }
          }
          cudaEventRecord(e1); cudaEventSynchronize(e1);
          float ms; cudaEventElapsedTime(&ms, e0, e1);
          if (it) printf("  %s: %.2f us per step\n", v == 2 ? "hc_pre_norm, 5 blocks" : v ? "1-block variant (writes all of y)" : "hc_pre_mb + rmsnorm (2 launches)", ms * 1e3 / reps);
        }
      }
    }
  // rope: 64 heads x 512, rope dim 64, position 20000
  const int heads = 64, hd = 512, rd = 64;
  std::vector<float> q(heads * hd), fr(rd / 2);
  for (auto &v : q) v = frand(seed);
  for (int i = 0; i < rd / 2; i++) fr[i] = powf(10000.f, -2.f * i / rd);
  float *dq, *dq2, *df;
  cudaMalloc(&dq, heads * hd * 4); cudaMalloc(&dq2, heads * hd * 4); cudaMalloc(&df, rd * 2);
  cudaMemcpy(dq, q.data(), heads * hd * 4, cudaMemcpyHostToDevice); cudaMemcpy(dq2, q.data(), heads * hd * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(df, fr.data(), rd * 2, cudaMemcpyHostToDevice);
  int gy = (heads * rd / 2 + 255) / 256;
  rope<<<dim3(1, 1), 256>>>(dq, df, heads, hd, rd, heads * hd, 20000, 0);
  rope<<<dim3(1, gy), 256>>>(dq2, df, heads, hd, rd, heads * hd, 20000, 0);
  std::vector<float> a(heads * hd), b(heads * hd);
  cudaMemcpy(a.data(), dq, heads * hd * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), dq2, heads * hd * 4, cudaMemcpyDeviceToHost);
  int nr = 0;
  for (int i = 0; i < heads * hd; i++) nr += memcmp(&a[i], &b[i], 4) != 0;
  printf("rope: %d of %d differ\n", nr, heads * hd);
  bad += nr;
  for (int g : {1, gy})
    for (int it = 0; it < 2; it++) {
      cudaEventRecord(e0);
      for (int r = 0; r < reps; r++) rope<<<dim3(1, g), 256>>>(dq, df, heads, hd, rd, heads * hd, 20000 + r, r & 1);
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      if (it) printf("  rope grid y %d: %.2f us per launch\n", g, ms * 1e3 / reps);
    }
  printf(bad ? "MISMATCH\n" : "all bit-identical\n");
  return bad != 0;
}
