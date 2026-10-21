// t16_gemv.cu - single-token GEMV over T16 bitshift-trellis weights (K = 2 bits per weight), the GPU decode cost of
// the T16 codebook next to dsx's IQ tiles (nvidia/tests/gpu_expbench.rs). Shapes: six experts' gate rows
// [6 x 2304, 5120] and six experts' down rows [6 x 5120, 2304], a pool of copies >= 256 MB so L2 never holds them.
//
// T16: weight j of a row has the 16-bit state s = bits [2j, 2j + 16) of the row's bit stream (little-endian u32 words);
// p = (s * 0x3A2F) mod 2^16; w = ((p & 0xFF) + (p >> 8) - 255) / 104.51, times a per-row scale.
// Layout: per row k / 16 u32 words (16 weights each) plus one pad word for the last window; f32 scale per row.
// Decode per weight: SHF (window from two words) + IMAD (product) + IDP4A (byte sum added to the bits of 2^23, so the
// result is the float 2^23 + lo + hi) + FADD (- 2^23 - 255, exact) + FFMA. A warp owns R rows, so each activation
// load serves R rows; lane i decodes word i of each 32-word stripe and takes the next word from lane i + 1.
//
// RTX 3090 on .102 (2026-10-21, fastest of 5 trials, interleaved with gpu_expbench): R = 2 rows per warp is best,
// gate x6 41.1-41.3 us, down x6 42.0-42.4 us; dsx's repacked tiles at T = 1: IQ2_XXS gate 40-43 / down 36-39 us,
// IQ1_S 38 / 36 us (each + 2 us of activation quantization). Memory floor 19 us at 936 GB/s; a plain read of the same
// bytes takes 22.5 us. ncu (R = 2): stalls are mostly long scoreboard (waiting on global loads), DRAM ~48% busy, the
// FMA-heavy pipe (IMAD + IDP4A) ~45%: latency-bound, not decode-bound. Two attempts to prefetch the next stripe (8-byte
// pair loads per lane; next-stripe words in registers) were slower in this harness and were dropped.
//
// build: nvcc -O3 -arch=sm_86 -o t16_gemv t16_gemv.cu
// run:   ./t16_gemv [reps]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
#define MUL 0x3A2Fu

template <int R>
__global__ void __launch_bounds__(128) t16_gemv(const uint32_t *__restrict__ w, const float *__restrict__ sc, const float *__restrict__ x,
                                                float *__restrict__ y, int rows, int k) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int r0 = warp * R;
  if (r0 >= rows) return;
  const int wpr = k / 16 + 1;  // words per row, with the pad word
  const int nw = k / 16;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.f;
  for (int base = 0; base < nw; base += 32) {
    const int j = base + lane;
    const bool live = j < nw;
    float xs[16];
    if (live) {
      const float4 *xp = (const float4 *)(x + 16 * j);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float4 v = __ldg(xp + i);
        xs[4 * i] = v.x, xs[4 * i + 1] = v.y, xs[4 * i + 2] = v.z, xs[4 * i + 3] = v.w;
      }
    }
    uint32_t cur[R], nxt[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
      const uint32_t *wr = w + (size_t)(r0 + r) * wpr;
      cur[r] = live ? __ldg(wr + j) : 0u;
      nxt[r] = __shfl_down_sync(0xffffffffu, cur[r], 1);
      if (live && (lane == 31 || j + 1 >= nw)) nxt[r] = __ldg(wr + j + 1);
    }
    if (!live) continue;
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
      for (int i = 0; i < 16; i++) {
        const uint32_t s = __funnelshift_r(cur[r], nxt[r], 2 * i);
        const uint32_t p = s * MUL;
        const float f = __int_as_float(__dp4a(p, 0x0101u, 0x4B000000u)) - 8388863.f;  // lo + hi - 255
        acc[r] = fmaf(f, xs[i], acc[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    float a = acc[r];
#pragma unroll
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) y[r0 + r] = a * (sc[r0 + r] * (1.f / 104.51f));
  }
}

// memory floor reference: read the same bytes, almost no math
__global__ void __launch_bounds__(128) read_only(const uint4 *__restrict__ w, size_t n, float *y) {
  uint32_t a = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = __ldg(w + i);
    a ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (a == 0x12345678u) y[0] = 1.f;
}

static double ref_row(const uint32_t *wr, float sc, const float *x, int k) {
  double a = 0;
  for (int j = 0; j < k; j++) {
    const int m = j / 16, o = 2 * (j % 16);
    const uint64_t two = (uint64_t)wr[m] | ((uint64_t)wr[m + 1] << 32);
    const uint32_t s = (uint32_t)(two >> o) & 0xFFFF;
    const uint32_t p = (s * MUL) & 0xFFFF;
    a += (double)((int)(p & 0xFF) + (int)(p >> 8) - 255) * x[j];
  }
  return a * sc / 104.51;
}

int main(int argc, char **argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 50;
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("%s (%d SMs), T16 K=2 single-token GEMV, f32 activations\n", prop.name, prop.multiProcessorCount);
  struct Shape { int rows, k; const char *what; } shapes[] = {{2304 * 6, 5120, "gate x6"}, {5120 * 6, 2304, "down x6"}};
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (auto sh : shapes) {
    const int wpr = sh.k / 16 + 1;
    const size_t words = (size_t)sh.rows * wpr, bytes = words * 4 + (size_t)sh.rows * 4;
    const int nc = (int)ceil(256e6 / bytes);
    std::vector<uint32_t> hw(words);
    uint64_t z = 88172645463325252ull;
    for (auto &v : hw) { z ^= z << 13; z ^= z >> 7; z ^= z << 17; v = (uint32_t)z; }
    std::vector<float> hs(sh.rows), hx(sh.k);
    for (int r = 0; r < sh.rows; r++) hs[r] = 0.5f + (r % 7) * 0.1f;
    for (int i = 0; i < sh.k; i++) hx[i] = (i % 100) * 0.01f - 0.5f;
    std::vector<uint32_t *> dw(nc);
    std::vector<float *> ds(nc);
    for (int c = 0; c < nc; c++) {
      CK(cudaMalloc(&dw[c], words * 4));
      CK(cudaMalloc(&ds[c], sh.rows * 4));
      CK(cudaMemcpy(dw[c], hw.data(), words * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(ds[c], hs.data(), sh.rows * 4, cudaMemcpyHostToDevice));
    }
    float *dx, *dy;
    CK(cudaMalloc(&dx, sh.k * 4));
    CK(cudaMalloc(&dy, sh.rows * 4));
    CK(cudaMemcpy(dx, hx.data(), sh.k * 4, cudaMemcpyHostToDevice));
    auto run = [&](int R, int c) {
      const int warps = (sh.rows + R - 1) / R, blocks = (warps * 32 + 127) / 128;
      if (R == 1) t16_gemv<1><<<blocks, 128>>>(dw[c], ds[c], dx, dy, sh.rows, sh.k);
      if (R == 2) t16_gemv<2><<<blocks, 128>>>(dw[c], ds[c], dx, dy, sh.rows, sh.k);
      if (R == 4) t16_gemv<4><<<blocks, 128>>>(dw[c], ds[c], dx, dy, sh.rows, sh.k);
      if (R == 8) t16_gemv<8><<<blocks, 128>>>(dw[c], ds[c], dx, dy, sh.rows, sh.k);
    };
    // R = rows per warp
    for (int R : {1, 2, 4, 8}) {
      run(R, 0);
      CK(cudaDeviceSynchronize());
      std::vector<float> hy(sh.rows);
      CK(cudaMemcpy(hy.data(), dy, sh.rows * 4, cudaMemcpyDeviceToHost));
      double maxrel = 0;
      for (int r = 0; r < sh.rows; r += 97) {
        const double want = ref_row(hw.data() + (size_t)r * wpr, hs[r], hx.data(), sh.k);
        maxrel = fmax(maxrel, fabs(hy[r] - want) / (fabs(want) + 1e-3));
      }
      for (int i = 0; i < 20; i++) run(R, i % nc);
      // the fastest of 5 trials: other GPU users (a Plex transcode on .102's GPU0) only add time
      double us = 1e9;
      for (int t = 0; t < 5; t++) {
        CK(cudaEventRecord(e0));
        for (int i = 0; i < reps; i++) run(R, i % nc);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        us = fmin(us, ms * 1e3 / reps);
      }
      printf("%-8s %5.1f MB  R=%d rows/warp: %6.1f us  %5.0f GB/s  (floor at 936 GB/s: %.1f us)  max rel err %.1e\n", sh.what, bytes / 1e6, R, us,
             bytes / us / 1e3, bytes / 936e3, maxrel);
    }
    {
      for (int i = 0; i < 5; i++) read_only<<<prop.multiProcessorCount * 16, 128>>>((const uint4 *)dw[i % nc], words / 4, dy);
      CK(cudaEventRecord(e0));
      for (int i = 0; i < reps; i++) read_only<<<prop.multiProcessorCount * 16, 128>>>((const uint4 *)dw[i % nc], words / 4, dy);
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      printf("%-8s read-only pass over the same bytes: %.1f us (%.0f GB/s)\n", sh.what, ms * 1e3 / reps, words * 4 / (ms * 1e3 / reps) / 1e3);
    }
    for (int c = 0; c < nc; c++) { cudaFree(dw[c]); cudaFree(ds[c]); }
    cudaFree(dx);
    cudaFree(dy);
  }
  return 0;
}
