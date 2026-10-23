// t16_mul1_gemv.cu - decode GEMV (T = 1) with T16's codebook (dsx's t16d_gemv inner loop, copied) against EXL3's
// stock mul1 codebook on the same streams, in two kernels: M1A = mask + IMAD by 0x83DCD12D + one dp4a per weight with
// the activation splat in all 4 bytes (exllamav3's int8 GEMV pattern); M1B = T16's pair-word activations, the two
// products of a pair split by PRMT into bytes 0-1 and 2-3, two dp4a per pair. GPU layout (repacked tiles), one warp
// per (pair of tile columns, split), the production splits (gate/up S = 4, down S = 3). A layer call = 6 experts out
// of a pool of 64 in VRAM, one launch per role over the 6 picks: gate, up (k 5120, n 2304), down (k 2304, n 5120). The integer accumulators of every
// kernel are checked against a host decode first.
// build: nvcc -O3 -arch=sm_86 -o t16_mul1_gemv bench/t16_mul1_gemv.cu
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__host__ __device__ constexpr int hend(int K2, int t) { return K2 % 2 ? (t + 1) * (K2 / 2) + (t + 1) / 2 : (t + 1) * (K2 / 2); }
__device__ __forceinline__ uint32_t fsr(uint32_t lo, uint32_t hi, int s) { return (uint32_t)((((uint64_t)hi << 32) | lo) >> s); }
__device__ __forceinline__ int dp4a(uint32_t a, int b, int c) { int d; asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d; }

// the 16 windows of tile column c from three consecutive words (as t16_prod16), before the multiply
template <int K2> __device__ __forceinline__ void windows(const uint32_t *ptr, int i0, int i1, int i2, int sh, uint32_t (&w)[16]) {
  const uint32_t a = ptr[i0], b = ptr[i1], c = ptr[i2];
  const uint32_t lo = fsr(c, b, sh), hi = fsr(b, a, sh);
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int o = hend(K2, 15) - hend(K2, j);
    w[j] = o < 32 ? __funnelshift_r(lo, hi, o) : hi >> (o - 32);
  }
}

// V 0 = T16, 1 = M1A, 2 = M1B. xp: pair words [k/2] (T16, M1B) or splat words [k] (M1A). RAW: write the integer
// accumulator (summed over groups) to raw instead of the scaled float.
template <int K2, int V, bool RAW>
__global__ void __launch_bounds__(128) gemv(const uint32_t *__restrict__ xp, const float *__restrict__ xd, const int *__restrict__ xs,
                                           const unsigned char *const *__restrict__ ws, int P, int k, int n, int S, float *__restrict__ part, long long *__restrict__ raw) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nbs = n / 16, groups = k / 256, pairs = nbs / 2;
  if (warp >= P * pairs * S) return;
  const int s = warp / (P * pairs), r = warp % (P * pairs), pk = r / pairs, nb = 2 * (r % pairs) + lane / 16, c = lane % 16;
  const unsigned char *w = ws[pk];
  const int g0 = groups * s / S, g1 = groups * (s + 1) / S;
  constexpr int WT = 4 * K2, NB = 32 * WT;
  const int e15 = hend(K2, 16 * c + 15) + NB, hw = (e15 - 1) / 32;
  const int i2 = hw % WT, i1 = (hw - 1) % WT, i0 = (hw - 2) % WT, sh = (hw + 1) * 32 - e15;
  const uint32_t *tp = (const uint32_t *)w + ((size_t)16 * g0 * nbs + nb) * WT;
  const uint32_t step = (uint32_t)nbs * WT;
  constexpr int XW = V == 1 ? 16 : 8;  // activation words per tile row of 16 inputs
  const uint4 *xe = (const uint4 *)xp + (size_t)(XW / 4) * 16 * g0;
  float f = 0.f;
  long long tot = 0;
  const float cb = V == 0 ? 0.009567260742f : 0.006767272949f, c0 = V == 0 ? 2.4375f : 3.453125f;
  for (int gr = g0; gr < g1; gr++) {
    int acc = 0;
#pragma unroll 4
    for (int i = 0; i < 16; i++, tp += step, xe += XW / 4) {
      uint32_t wv[16];
      windows<K2>(tp, i0, i1, i2, sh, wv);
      uint32_t xw[XW];
#pragma unroll
      for (int q = 0; q < XW / 4; q++) { const uint4 v = __ldg(xe + q); xw[4 * q] = v.x, xw[4 * q + 1] = v.y, xw[4 * q + 2] = v.z, xw[4 * q + 3] = v.w; }
      if (V == 0) {
#pragma unroll
        for (int q = 0; q < 8; q++) acc = dp4a(__byte_perm(wv[2 * q] * 0x3A2Fu, wv[2 * q + 1] * 0x3A2Fu, 0x5410), (int)xw[q], acc);
      } else if (V == 1) {
#pragma unroll
        for (int j = 0; j < 16; j++) acc = dp4a((wv[j] & 0xFFFFu) * 0x83DCD12Du, (int)xw[j], acc);
      } else {
#pragma unroll
        for (int q = 0; q < 8; q++) {
          const uint32_t p0 = (wv[2 * q] & 0xFFFFu) * 0x83DCD12Du, p1 = (wv[2 * q + 1] & 0xFFFFu) * 0x83DCD12Du;
          acc = dp4a(__byte_perm(p0, p1, 0x5410), (int)xw[q], acc);
          acc = dp4a(__byte_perm(p0, p1, 0x7632), (int)xw[q], acc);
        }
      }
    }
    if (RAW) tot += acc;
    else f += __ldg(xd + gr) * (cb * (float)acc - c0 * (float)__ldg(xs + gr));
  }
  if (RAW) raw[((size_t)s * P + pk) * n + 16 * nb + c] = tot;
  else part[((size_t)s * P + pk) * n + 16 * nb + c] = f;
}

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint64_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static size_t tbytes(int k2, int k, int n) { return (size_t)(k / 16) * (n / 16) * 16 * k2; }

// host: state of position t of a GPU-layout tile (u32 words, MSB first)
static uint32_t hstate(const uint32_t *wd, int k2, int t) {
  const int nb = 128 * k2, nw = 4 * k2, b0 = (hend(k2, t) - 16 + nb) % nb, i0 = b0 >> 5, i1 = (i0 + 1) % nw, s = b0 & 31;
  const uint64_t v = ((uint64_t)wd[i0] << 32) | wd[i1];
  return (uint32_t)(v >> (48 - s)) & 0xFFFFu;
}

template <int K2, int V, bool RAW> void launch(const uint32_t *xp, const float *xd, const int *xs, const unsigned char *const *w, int P, int k, int n, int S, float *part, long long *raw, cudaStream_t st) {
  const int warps = P * (n / 32) * S;
  gemv<K2, V, RAW><<<(warps * 32 + 127) / 128, 128, 0, st>>>(xp, xd, xs, w, P, k, n, S, part, raw);
}
template <int V, bool RAW> void launchk(int k2, const uint32_t *xp, const float *xd, const int *xs, const unsigned char *const *w, int P, int k, int n, int S, float *part, long long *raw, cudaStream_t st) {
  if (k2 == 3) launch<3, V, RAW>(xp, xd, xs, w, P, k, n, S, part, raw, st);
  else if (k2 == 4) launch<4, V, RAW>(xp, xd, xs, w, P, k, n, S, part, raw, st);
  else launch<5, V, RAW>(xp, xd, xs, w, P, k, n, S, part, raw, st);
}
template <bool RAW> void launchv(int v, int k2, const uint32_t *xp, const float *xd, const int *xs, const unsigned char *const *w, int P, int k, int n, int S, float *part, long long *raw, cudaStream_t st) {
  if (v == 0) launchk<0, RAW>(k2, xp, xd, xs, w, P, k, n, S, part, raw, st);
  else if (v == 1) launchk<1, RAW>(k2, xp, xd, xs, w, P, k, n, S, part, raw, st);
  else launchk<2, RAW>(k2, xp, xd, xs, w, P, k, n, S, part, raw, st);
}

int main(int argc, char **argv) {
  const int DIM = 5120, INTER = 2304, POOL = 64, CALLS = 400;
  const char *names[3] = {"T16", "M1A", "M1B"};
  // activations: q per input for both shapes (max k = DIM), pair words and splat words
  std::vector<int8_t> q(DIM);
  for (auto &v : q) v = (int8_t)((int)(rnd() % 255) - 127);
  std::vector<uint32_t> pw(DIM / 2), sw(DIM);
  for (int i = 0; i < DIM; i++) { const uint32_t b = (uint8_t)q[i]; sw[i] = b * 0x01010101u; pw[i / 2] |= (b | b << 8) << (16 * (i % 2)); }
  std::vector<float> xd(DIM / 256, 0.01f); std::vector<int> xs(DIM / 256, 0);
  uint32_t *dpw, *dsw; float *dxd, *part; int *dxs; long long *raw;
  CK(cudaMalloc(&dpw, pw.size() * 4)); CK(cudaMalloc(&dsw, sw.size() * 4)); CK(cudaMalloc(&dxd, xd.size() * 4)); CK(cudaMalloc(&dxs, xs.size() * 4));
  CK(cudaMemcpy(dpw, pw.data(), pw.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsw, sw.data(), sw.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dxd, xd.data(), xd.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dxs, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&part, 4 * 6 * (size_t)DIM * 4)); CK(cudaMalloc(&raw, 4 * 6 * (size_t)DIM * 8));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  int sm; cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
  int gpl, mpl; cudaDeviceGetAttribute(&gpl, cudaDevAttrClockRate, 0); cudaDeviceGetAttribute(&mpl, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceProp pr; cudaGetDeviceProperties(&pr, 0);
  printf("%s, %d SMs\n", pr.name, sm);
  for (int k2 = 3; k2 <= 5; k2++) {
    // the pool: gate, up [5120 -> 2304] and down [2304 -> 5120] per expert, random streams
    const size_t gb = tbytes(k2, DIM, INTER), db = tbytes(k2, INTER, DIM), eb = 2 * gb + db;
    std::vector<unsigned char> h(eb * POOL);
    for (auto &b : h) b = (unsigned char)rnd();
    unsigned char *dw; CK(cudaMalloc(&dw, h.size())); CK(cudaMemcpy(dw, h.data(), h.size(), cudaMemcpyHostToDevice));
    // pointer tables: [0] expert 0's gate (check); then per call and role the 6 picked experts' matrices
    std::vector<const unsigned char *> ptr(1 + (size_t)CALLS * 3 * 6);
    ptr[0] = dw;
    for (int i = 0; i < CALLS; i++)
      for (int role = 0; role < 3; role++)
        for (int j = 0; j < 6; j++) ptr[1 + ((size_t)i * 3 + role) * 6 + j] = dw + (size_t)((i * 7 + j * 11) % POOL) * eb + role * gb;
    const unsigned char **dptr; CK(cudaMalloc(&dptr, ptr.size() * sizeof(void *))); CK(cudaMemcpy(dptr, ptr.data(), ptr.size() * sizeof(void *), cudaMemcpyHostToDevice));
    const unsigned char *const *pp = dptr;
    // check: expert 0's gate matrix, every variant's integer accumulators against the host decode
    for (int v = 0; v < 3; v++) {
      const int k = DIM, n = INTER;
      launchv<true>(v, k2, v == 1 ? dsw : dpw, dxd, dxs, pp, 1, k, n, 1, part, raw, st);
      std::vector<long long> got(n); CK(cudaMemcpy(got.data(), raw, n * 8, cudaMemcpyDeviceToHost));
      size_t bad = 0;
      for (int o = 0; o < n; o++) {
        long long want = 0;
        const int nb = o / 16, c = o % 16;
        for (int kb = 0; kb < k / 16; kb++) {
          const uint32_t *tile = (const uint32_t *)(h.data()) + ((size_t)kb * (n / 16) + nb) * 4 * k2;
          for (int r = 0; r < 16; r++) {
            const uint32_t s = hstate(tile, k2, 16 * c + r);
            uint32_t b;
            if (v == 0) { const uint32_t p = (s * 0x3A2Fu) & 0xFFFF; b = (p & 0xFF) + (p >> 8); }
            else { const uint32_t x = s * 0x83DCD12Du; b = (x & 0xFF) + (x >> 8 & 0xFF) + (x >> 16 & 0xFF) + (x >> 24); }
            want += (long long)b * q[16 * kb + r];
          }
        }
        bad += want != got[o];
      }
      printf("K %.1f %s: %zu of %d accumulators differ from the host decode\n", k2 / 2.0, names[v], bad, n);
    }
    // timing: A B C C B A order over variants, one pause before each, CALLS layer calls each
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    double sum[3] = {0, 0, 0}; int cnt[3] = {0, 0, 0};
    const int order[6] = {0, 1, 2, 2, 1, 0};
    for (int oi = 0; oi < 6; oi++) {
      const int v = order[oi];
      std::this_thread::sleep_for(std::chrono::seconds(2));
      const uint32_t *x = v == 1 ? dsw : dpw;
      // one launch per role over the 6 picks, as dsx's t16d_gemv (gate, up, then down)
      auto call = [&](int i) {
        const unsigned char *const *pc = dptr + 1 + (size_t)i * 18;
        launchv<false>(v, k2, x, dxd, dxs, pc, 6, DIM, INTER, 4, part, raw, st);
        launchv<false>(v, k2, x, dxd, dxs, pc + 6, 6, DIM, INTER, 4, part, raw, st);
        launchv<false>(v, k2, x, dxd, dxs, pc + 12, 6, INTER, DIM, 3, part, raw, st);
      };
      call(0); CK(cudaStreamSynchronize(st));
      cudaEventRecord(e0, st);
      for (int i = 0; i < CALLS; i++) call(i);
      cudaEventRecord(e1, st); CK(cudaEventSynchronize(e1));
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      sum[v] += ms * 1e3 / CALLS; cnt[v]++;
    }
    const double wts = 6.0 * 3 * (double)DIM * INTER, bytes = 6.0 * eb;
    for (int v = 0; v < 3; v++) {
      const double us = sum[v] / cnt[v];
      printf("K %.1f %s: %7.1f us per layer call (6 experts x gate, up, down; 3 launches), %.0f GB/s, %.2f G weights/s\n", k2 / 2.0, names[v], us, bytes / us / 1e3, wts / us / 1e3);
    }
    CK(cudaFree(dw)); CK(cudaFree(dptr));
  }
}
