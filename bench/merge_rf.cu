// merge_rf.cu - sparse_attn_merge_rf (merge + inverse RoPE + fake FP8, one launch) against sparse_attn_merge + rope
// (inverse) + fake_fp8 on random split-K partials (64 heads x 512, S = 20 tiles, positions 0..60000): bit equality and
// median time of 5 interleaved rounds. nvcc -O3 -arch=sm_86 -o merge_rf bench/merge_rf.cu
#include "../nvidia/src/kernels/dsx.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
static float frand(unsigned &s) { s = s * 1664525u + 1013904223u; return ((s >> 8) / 16777216.f) * 2.f - 1.f; }
int main() {
  const int H = 64, hd = 512, rd = 64, S = 20;
  unsigned seed = 9;
  size_t pn = (size_t)S * H * (hd + 2);
  std::vector<float> hp(pn), hs(H), hf(rd / 2);
  for (size_t i = 0; i < pn; i++) hp[i] = frand(seed);
  for (int z = 0; z < S; z++) for (int h = 0; h < H; h++) {
    float *p = &hp[((size_t)z * H + h) * (hd + 2)];
    p[hd] = frand(seed) * 4; p[hd + 1] = 1.f + fabsf(frand(seed)) * 30;  // max logit, sum of exps
  }
  for (auto &v : hs) v = frand(seed);
  for (int i = 0; i < rd / 2; i++) hf[i] = powf(160000.f, -2.f * i / rd);
  float *part, *sinks, *fr, *o0, *o1;
  cudaMalloc(&part, pn * 4); cudaMalloc(&sinks, H * 4); cudaMalloc(&fr, rd * 2); cudaMalloc(&o0, H * hd * 4); cudaMalloc(&o1, H * hd * 4);
  cudaMemcpy(part, hp.data(), pn * 4, cudaMemcpyHostToDevice); cudaMemcpy(sinks, hs.data(), H * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(fr, hf.data(), rd * 2, cudaMemcpyHostToDevice);
  int bad = 0;
  std::vector<float> a(H * hd), b(H * hd);
  for (int pos : {0, 1, 127, 4096, 33333, 60000}) {
    sparse_attn_merge<<<dim3(1, H), 256>>>(part, S, sinks, o0, H, hd);
    rope<<<dim3(1, (H * rd / 2 + 255) / 256), 256>>>(o0, fr, H, hd, rd, H * hd, pos, 1);
    fake_fp8<<<(H * hd / 32 + 7) / 8, 256>>>(o0, H * hd);
    sparse_attn_merge_rf<<<dim3(1, H), hd / 2>>>(part, S, sinks, o1, H, hd, fr, rd, pos);
    cudaMemcpy(a.data(), o0, H * hd * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), o1, H * hd * 4, cudaMemcpyDeviceToHost);
    int n = 0; for (int i = 0; i < H * hd; i++) n += memcmp(&a[i], &b[i], 4) != 0;
    printf("pos %5d: %d of %d differ\n", pos, n, H * hd);
    bad += n;
  }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto sep = [&](int i) {
    sparse_attn_merge<<<dim3(1, H), 256>>>(part, S, sinks, o0, H, hd);
    rope<<<dim3(1, (H * rd / 2 + 255) / 256), 256>>>(o0, fr, H, hd, rd, H * hd, 20000 + i, 1);
    fake_fp8<<<(H * hd / 32 + 7) / 8, 256>>>(o0, H * hd);
  };
  auto fus = [&](int i) { sparse_attn_merge_rf<<<dim3(1, H), hd / 2>>>(part, S, sinks, o1, H, hd, fr, rd, 20000 + i); };
  for (int i = 0; i < 3000; i++) { sep(i); fus(i); }
  std::vector<float> ts, tf;
  for (int r = 0; r < 5; r++) for (int v = 0; v < 2; v++) {
    cudaEventRecord(e0); for (int i = 0; i < 1000; i++) (v ? (void)fus(i) : (void)sep(i)); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); (v ? tf : ts).push_back(ms);
  }
  std::sort(ts.begin(), ts.end()); std::sort(tf.begin(), tf.end());
  printf("merge + rope + fake_fp8 (3 launches) %.2f us, sparse_attn_merge_rf %.2f us\n", ts[2], tf[2]);
  printf(bad ? "MISMATCH\n" : "all bit-identical\n");
  return bad != 0;
}
