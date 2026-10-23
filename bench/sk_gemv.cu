// sk_gemv.cu - the split-k GEMVs (one row per 128-thread block; decode's router, q_a, kv, small f16 matrices): the
// previous kernels (scalar 2-byte weight loads, MAXT accumulators; copied here) against the production ones (one vector
// load per 4 weights; *_sk1: one-token variants), T = 1: bit equality and median time of 5 interleaved rounds.
// nvcc -O3 -arch=sm_86 -o sk bench/sk_gemv.cu
#include "../nvidia/src/kernels/dsx.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

template <int TY>
__device__ void old_sk_plain(const void *w, const float *x, float *y, int rows, int k, int T, int ldx, int ldy) {
  __shared__ float red[4][MAXT];
  int r = blockIdx.x, wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float acc[MAXT] = {0};
  size_t base = (size_t)r * k;
  for (int i = threadIdx.x * 4; i < k; i += blockDim.x * 4) {
    float w0 = load_w<TY>(w, base + i), w1 = load_w<TY>(w, base + i + 1), w2 = load_w<TY>(w, base + i + 2), w3 = load_w<TY>(w, base + i + 3);
    for (int t = 0; t < T; t++) {
      float4 a = *(const float4 *)(x + (size_t)t * ldx + i);
      acc[t] += w0 * a.x + w1 * a.y + w2 * a.z + w3 * a.w;
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) red[wid][t] = s;
  }
  __syncthreads();
  if (threadIdx.x < T) y[(size_t)threadIdx.x * ldy + r] = red[0][threadIdx.x] + red[1][threadIdx.x] + red[2][threadIdx.x] + red[3][threadIdx.x];
}
__global__ void old_bf16_sk(const void *w, const float *x, float *y, int rows, int k, int T, int ldx, int ldy) { old_sk_plain<30>(w, x, y, rows, k, T, ldx, ldy); }
__global__ void old_f16_sk(const void *w, const float *x, float *y, int rows, int k, int T, int ldx, int ldy) { old_sk_plain<1>(w, x, y, rows, k, T, ldx, ldy); }
__global__ void old_q8r_sk(const int8_t *__restrict__ qs, const float *__restrict__ x, float *__restrict__ y, int rows, int k, int T,
                           int ldx, int ldy, const __half *__restrict__ d) {
  __shared__ float red[4][MAXT];
  int r = blockIdx.x, wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int8_t *qr = qs + (size_t)r * k;
  const __half *dr = d + (size_t)r * (k / 32);
  float acc[MAXT] = {0};
  for (int i = threadIdx.x * 16; i < k; i += blockDim.x * 16) {
    int4 raw = __ldg((const int4 *)(qr + i));
    const int8_t *q = (const int8_t *)&raw;
    float sc = __half2float(dr[i / 32]);
    for (int t = 0; t < T; t++) {
      const float4 *xp = (const float4 *)(x + (size_t)t * ldx + i);
      float4 a = xp[0], b = xp[1], c = xp[2], e = xp[3];
      acc[t] += sc * (q[0] * a.x + q[1] * a.y + q[2] * a.z + q[3] * a.w + q[4] * b.x + q[5] * b.y + q[6] * b.z + q[7] * b.w +
                      q[8] * c.x + q[9] * c.y + q[10] * c.z + q[11] * c.w + q[12] * e.x + q[13] * e.y + q[14] * e.z + q[15] * e.w);
    }
  }
  for (int t = 0; t < T; t++) {
    float s = warp_sum(acc[t]);
    if (lane == 0) red[wid][t] = s;
  }
  __syncthreads();
  if (threadIdx.x < T) y[(size_t)threadIdx.x * ldy + r] = red[0][threadIdx.x] + red[1][threadIdx.x] + red[2][threadIdx.x] + red[3][threadIdx.x];
}

static float frand(unsigned &s) { s = s * 1664525u + 1013904223u; return ((s >> 8) / 16777216.f) * 2.f - 1.f; }

int main() {
  // kind 0: bf16, 1: f16, 2: q8
  struct S { const char *name; int kind, rows, k; } shapes[] = {
      {"router bf16 5120->384", 0, 384, 5120}, {"f16 5120->1024", 1, 1024, 5120}, {"f16 4096->512", 1, 512, 4096},
      {"q_a q8 5120->1280", 2, 1280, 5120}, {"kv q8 5120->512", 2, 512, 5120}};
  const int ROUNDS = 5, REPS = 400;
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  unsigned seed = 5;
  int bad = 0;
  for (auto &s : shapes) {
    size_t wbytes = s.kind == 2 ? (size_t)s.rows * s.k : (size_t)s.rows * s.k * 2, db = (size_t)s.rows * (s.k / 32) * 2;
    const int pool = (int)((128ull << 20) / (wbytes + db)) + 2;
    uint8_t *w; __half *d; float *x, *y[3];
    CK(cudaMalloc(&w, wbytes * pool)); CK(cudaMalloc(&d, db * pool)); CK(cudaMalloc(&x, s.k * 4));
    for (int v = 0; v < 3; v++) CK(cudaMalloc(&y[v], s.rows * 4));
    std::vector<uint8_t> hw(wbytes);
    if (s.kind == 2) for (auto &v : hw) v = (uint8_t)(rand() % 255 - 127);
    else for (size_t i = 0; i < wbytes / 2; i++) {
      float f = frand(seed) * 0.05f;
      uint16_t b = s.kind == 0 ? __bfloat16_as_ushort(__float2bfloat16(f)) : __half_as_ushort(__float2half(f));
      memcpy(&hw[i * 2], &b, 2);
    }
    std::vector<__half> hd(db / 2); for (auto &v : hd) v = __float2half(0.01f + (rand() % 100) * 1e-4f);
    std::vector<float> hx(s.k); for (auto &v : hx) v = frand(seed);
    for (int p = 0; p < pool; p++) { CK(cudaMemcpy(w + wbytes * p, hw.data(), wbytes, cudaMemcpyHostToDevice)); CK(cudaMemcpy((char *)d + db * p, hd.data(), db, cudaMemcpyHostToDevice)); }
    CK(cudaMemcpy(x, hx.data(), s.k * 4, cudaMemcpyHostToDevice));
    // v 0: old, 1: production generic (T = 1), 2: production one-token (*_sk1)
    auto run = [&](int v, int i) {
      const uint8_t *ww = w + wbytes * (i % pool); const __half *dd = (const __half *)((char *)d + db * (i % pool));
      if (s.kind == 0) { if (v == 0) old_bf16_sk<<<s.rows, 128>>>(ww, x, y[v], s.rows, s.k, 1, s.k, s.rows); else if (v == 1) gemv_bf16_sk<<<s.rows, 128>>>(ww, x, y[v], s.rows, s.k, 1, s.k, s.rows); else gemv_bf16_sk1<<<s.rows, 128>>>(ww, x, y[v], s.rows, s.k, 1, s.k, s.rows); }
      if (s.kind == 1) { if (v == 0) old_f16_sk<<<s.rows, 128>>>(ww, x, y[v], s.rows, s.k, 1, s.k, s.rows); else if (v == 1) gemv_f16_sk<<<s.rows, 128>>>(ww, x, y[v], s.rows, s.k, 1, s.k, s.rows); else gemv_f16_sk1<<<s.rows, 128>>>(ww, x, y[v], s.rows, s.k, 1, s.k, s.rows); }
      if (s.kind == 2) { const int8_t *q = (const int8_t *)ww; if (v == 0) old_q8r_sk<<<s.rows, 128>>>(q, x, y[v], s.rows, s.k, 1, s.k, s.rows, dd); else if (v == 1) gemv_q8r_sk<<<s.rows, 128>>>(q, x, y[v], s.rows, s.k, 1, s.k, s.rows, dd); else gemv_q8r_sk1<<<s.rows, 128>>>(q, x, y[v], s.rows, s.k, 1, s.k, s.rows, dd); }
    };
    for (int i = 0; i < 4000; i++) run(i % 3, i);
    std::vector<float> t[3];
    for (int r = 0; r < ROUNDS; r++)
      for (int v = 0; v < 3; v++) {
        cudaEventRecord(e0); for (int i = 0; i < REPS; i++) run(v, i); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1); t[v].push_back(ms / REPS);
      }
    for (int v = 0; v < 3; v++) run(v, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> a(s.rows), b(s.rows);
    CK(cudaMemcpy(a.data(), y[0], s.rows * 4, cudaMemcpyDeviceToHost));
    printf("%-22s", s.name);
    const char *nm[] = {"old", "new", "new1"};
    for (int v = 0; v < 3; v++) {
      std::sort(t[v].begin(), t[v].end());
      float ms = t[v][ROUNDS / 2];
      CK(cudaMemcpy(b.data(), y[v], s.rows * 4, cudaMemcpyDeviceToHost));
      int diff = 0; for (int i = 0; i < s.rows; i++) diff += memcmp(&a[i], &b[i], 4) != 0;
      bad += diff;
      printf(" | %s %5.1f us %4.0f GB/s%s", nm[v], ms * 1e3, (wbytes + (s.kind == 2 ? db : 0)) / ms / 1e6, diff ? " DIFF" : "");
    }
    printf("\n");
    cudaFree(w); cudaFree(d); cudaFree(x); for (int v = 0; v < 3; v++) cudaFree(y[v]);
  }
  printf(bad ? "MISMATCH\n" : "all bit-identical to the previous kernels\n");
  return bad != 0;
}
