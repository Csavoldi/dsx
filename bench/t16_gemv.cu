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
// RTX 3090 on .102 (2026-10-21; each config in its own process after a 4 s pause: the 230 W power cap set by
// dsx-serve.sh lowers the clocks within a second of back-to-back GEMVs; repeats agree to 0.1 us):
//   v5 (int8 activations, dp4a, includes the xquant launch): gate x6 33.8 us (cfg 222), down x6 35.9 us (cfg 212)
//   v4 (f32 activations, permuted): gate 40.6, down 43.8 (cfg 124); v1 (f32): 44.2 / 44.1 (cfg 2)
//   dsx tiles at T = 1, same protocol: IQ2_XXS 43 / 39 us, IQ1_S 41-42 / 40 us, each + 2 us quant_act
//   memory floor 19 us at 936 GB/s; a plain read of the same bytes 22.5 us
// ncu (v5): ALU pipe (SHF, PRMT) 70-75%, FMA-heavy (IMAD, IDP4A) 53-57%, DRAM 63-66%; stalls split between long
// scoreboard and math-pipe throttle. Fewer instructions per weight (e.g. mma tiles) is the next step.
//
// build: nvcc -O3 -arch=sm_86 -o t16_gemv t16_gemv.cu
// run:   ./t16_gemv [reps] [cfg,cfg,...] [gate|down]   cfg: R (v1, R rows per warp), 100 + 10 U + R (v4),
//        200 + 10 U + R (v5); U = stripes of 32 words loaded before decoding
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
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

// v4: activations pre-permuted once per call (xperm) so lane l's four 16-byte loads for word j are contiguous across
// the warp: xp[g][j] = x[16 j + 4 g .. + 3] (float4), g = 0..3. U stripes of weights are loaded before any is decoded.
__global__ void xperm(const float4 *__restrict__ x, float4 *__restrict__ xp, int nw) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < nw) {
#pragma unroll
    for (int g = 0; g < 4; g++) xp[(size_t)g * nw + j] = x[4 * j + g];
  }
}

template <int R, int U>
__global__ void __launch_bounds__(128) t16_gemv4(const uint32_t *__restrict__ w, const float *__restrict__ sc, const float4 *__restrict__ xp,
                                                 float *__restrict__ y, int rows, int k) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int r0 = warp * R;
  if (r0 >= rows) return;
  const int wpr = k / 16 + 1, nw = k / 16;
  const uint32_t *wr0 = w + (size_t)r0 * wpr;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.f;
  for (int base = 0; base < nw; base += 32 * U) {
    uint32_t cur[U][R];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int j = base + 32 * u + lane;
#pragma unroll
      for (int r = 0; r < R; r++) cur[u][r] = j < nw ? __ldg(wr0 + (size_t)r * wpr + j) : 0u;
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int j = base + 32 * u + lane;
      const bool live = j < nw;
      float4 xv[4];
#pragma unroll
      for (int g = 0; g < 4; g++) xv[g] = live ? __ldg(xp + (size_t)g * nw + j) : make_float4(0, 0, 0, 0);
      const float *xs = (const float *)xv;
#pragma unroll
      for (int r = 0; r < R; r++) {
        uint32_t nxt = __shfl_down_sync(0xffffffffu, cur[u][r], 1);
        if (u + 1 < U) {
          const uint32_t n0 = __shfl_sync(0xffffffffu, cur[u + 1 < U ? u + 1 : u][r], 0);
          if (lane == 31) nxt = n0;
        } else if (lane == 31 && live) {
          nxt = __ldg(wr0 + (size_t)r * wpr + j + 1);
        }
        if (live && j + 1 >= nw) nxt = __ldg(wr0 + (size_t)r * wpr + nw);
        if (live) {
#pragma unroll
          for (int i = 0; i < 16; i++) {
            const uint32_t st = __funnelshift_r(cur[u][r], nxt, 2 * i);
            const float f = __int_as_float(__dp4a(st * MUL, 0x0101u, 0x4B000000u)) - 8388863.f;
            acc[r] = fmaf(f, xs[i], acc[r]);
          }
        }
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

// v5: int8 activations (per-256 scale, quantized once per call by xquant) and integer dot products. Four weights'
// products are gathered into one word of low bytes and one of high bytes (4 PRMT), and two dp4a (u8 x s8) add
// lo * x + hi * x; the -255 * sum(x) term uses each word's precomputed activation sum. Per 16 weights and row:
// ~16 SHF, 16 IMAD, 16 PRMT, 8 IDP4A, then one IMAD, I2F, FFMA for the word.
__device__ __forceinline__ int dp4a_us(uint32_t a, int b, int c) {
  int d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// x [k] f32 -> xq [k] int8, xd [k / 256] scale, xs [k / 16] sum of each word's 16 int8 values; one warp per 256 block
__global__ void xquant(const float *__restrict__ x, int8_t *__restrict__ xq, float *__restrict__ xd, int *__restrict__ xs, int k) {
  const int lane = threadIdx.x & 31, blk = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (blk * 256 >= k) return;
  float v[8], m = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = x[blk * 256 + lane * 8 + i], m = fmaxf(m, fabsf(v[i]));
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const float d = m / 127.f, id = m > 0 ? 127.f / m : 0.f;
  int sum = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int q = __float2int_rn(v[i] * id);
    xq[blk * 256 + lane * 8 + i] = (int8_t)q;
    sum += q;
  }
  sum += __shfl_xor_sync(0xffffffffu, sum, 1);  // lanes 2w, 2w + 1 hold word w's 16 values
  if (!(lane & 1)) xs[blk * 16 + lane / 2] = sum;
  if (lane == 0) xd[blk] = d;
}

template <int R, int U>
__global__ void __launch_bounds__(128) t16_gemv5(const uint32_t *__restrict__ w, const float *__restrict__ sc, const int4 *__restrict__ xq,
                                                 const float *__restrict__ xd, const int *__restrict__ xs, float *__restrict__ y, int rows, int k) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int r0 = warp * R;
  if (r0 >= rows) return;
  const int wpr = k / 16 + 1, nw = k / 16;
  const uint32_t *wr0 = w + (size_t)r0 * wpr;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.f;
  for (int base = 0; base < nw; base += 32 * U) {
    uint32_t cur[U][R];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int j = base + 32 * u + lane;
#pragma unroll
      for (int r = 0; r < R; r++) cur[u][r] = j < nw ? __ldg(wr0 + (size_t)r * wpr + j) : 0u;
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int j = base + 32 * u + lane;
      const bool live = j < nw;
      const int4 xv = live ? __ldg(xq + j) : make_int4(0, 0, 0, 0);
      const int sx = live ? __ldg(xs + j) : 0;
      const float d = live ? __ldg(xd + j / 16) : 0.f;
      const int xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
      for (int r = 0; r < R; r++) {
        uint32_t nxt = __shfl_down_sync(0xffffffffu, cur[u][r], 1);
        if (u + 1 < U) {
          const uint32_t n0 = __shfl_sync(0xffffffffu, cur[u + 1 < U ? u + 1 : u][r], 0);
          if (lane == 31) nxt = n0;
        } else if (lane == 31 && live) {
          nxt = __ldg(wr0 + (size_t)r * wpr + j + 1);
        }
        if (live && j + 1 >= nw) nxt = __ldg(wr0 + (size_t)r * wpr + nw);
        if (live) {
          int ai = -255 * sx;
#pragma unroll
          for (int q = 0; q < 4; q++) {
            uint32_t p[4];
#pragma unroll
            for (int i = 0; i < 4; i++) p[i] = __funnelshift_r(cur[u][r], nxt, 2 * (4 * q + i)) * MUL;
            const uint32_t a = __byte_perm(p[0], p[1], 0x5410), b = __byte_perm(p[2], p[3], 0x5410);
            ai = dp4a_us(__byte_perm(a, b, 0x6420), xw[q], ai);
            ai = dp4a_us(__byte_perm(a, b, 0x7531), xw[q], ai);
          }
          acc[r] = fmaf((float)ai, d, acc[r]);
        }
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
    // argv[3]: "gate" or "down", one shape only
    if (argc > 3 && strncmp(argv[3], sh.what, 4)) continue;
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
    float *dx, *dy, *dxp, *xqd;
    int8_t *xq8;
    int *xqs;
    CK(cudaMalloc(&xq8, sh.k));
    CK(cudaMalloc(&xqd, sh.k / 256 * 4));
    CK(cudaMalloc(&xqs, sh.k / 16 * 4));
    CK(cudaMalloc(&dx, sh.k * 4));
    CK(cudaMalloc(&dxp, sh.k * 4));
    CK(cudaMalloc(&dy, sh.rows * 4));
    CK(cudaMemcpy(dx, hx.data(), sh.k * 4, cudaMemcpyHostToDevice));
    auto run = [&](int R, int c) {
      const int warps = (sh.rows + (R % 10) - 1) / (R % 10), blocks = (warps * 32 + 127) / 128;
      if (R == 1) t16_gemv<1><<<blocks, 128>>>(dw[c], ds[c], dx, dy, sh.rows, sh.k);
      if (R == 2) t16_gemv<2><<<blocks, 128>>>(dw[c], ds[c], dx, dy, sh.rows, sh.k);
      if (R == 4) t16_gemv<4><<<blocks, 128>>>(dw[c], ds[c], dx, dy, sh.rows, sh.k);
      if (R == 8) t16_gemv<8><<<blocks, 128>>>(dw[c], ds[c], dx, dy, sh.rows, sh.k);
      if (R > 200) {
        const int Rr = R % 10, U = (R / 10) % 10, b5 = ((sh.rows + Rr - 1) / Rr * 32 + 127) / 128;
        xquant<<<(sh.k / 256 * 32 + 127) / 128, 128>>>(dx, xq8, xqd, xqs, sh.k);
        if (U == 1 && Rr == 2) t16_gemv5<2, 1><<<b5, 128>>>(dw[c], ds[c], (const int4 *)xq8, xqd, xqs, dy, sh.rows, sh.k);
        if (U == 2 && Rr == 2) t16_gemv5<2, 2><<<b5, 128>>>(dw[c], ds[c], (const int4 *)xq8, xqd, xqs, dy, sh.rows, sh.k);
        if (U == 2 && Rr == 4) t16_gemv5<4, 2><<<b5, 128>>>(dw[c], ds[c], (const int4 *)xq8, xqd, xqs, dy, sh.rows, sh.k);
        if (U == 4 && Rr == 2) t16_gemv5<2, 4><<<b5, 128>>>(dw[c], ds[c], (const int4 *)xq8, xqd, xqs, dy, sh.rows, sh.k);
        if (U == 1 && Rr == 4) t16_gemv5<4, 1><<<b5, 128>>>(dw[c], ds[c], (const int4 *)xq8, xqd, xqs, dy, sh.rows, sh.k);
      } else if (R > 100) {
        const int Rr = R % 10, U = (R / 10) % 10, b4 = ((sh.rows + Rr - 1) / Rr * 32 + 127) / 128;
        xperm<<<(sh.k / 16 + 127) / 128, 128>>>((const float4 *)dx, (float4 *)dxp, sh.k / 16);
        if (U == 1 && Rr == 1) t16_gemv4<1, 1><<<b4, 128>>>(dw[c], ds[c], (const float4 *)dxp, dy, sh.rows, sh.k);
        if (U == 1 && Rr == 2) t16_gemv4<2, 1><<<b4, 128>>>(dw[c], ds[c], (const float4 *)dxp, dy, sh.rows, sh.k);
        if (U == 2 && Rr == 1) t16_gemv4<1, 2><<<b4, 128>>>(dw[c], ds[c], (const float4 *)dxp, dy, sh.rows, sh.k);
        if (U == 2 && Rr == 2) t16_gemv4<2, 2><<<b4, 128>>>(dw[c], ds[c], (const float4 *)dxp, dy, sh.rows, sh.k);
        if (U == 2 && Rr == 4) t16_gemv4<4, 2><<<b4, 128>>>(dw[c], ds[c], (const float4 *)dxp, dy, sh.rows, sh.k);
        if (U == 2 && Rr == 8) t16_gemv4<8, 2><<<b4, 128>>>(dw[c], ds[c], (const float4 *)dxp, dy, sh.rows, sh.k);
        if (U == 4 && Rr == 2) t16_gemv4<2, 4><<<b4, 128>>>(dw[c], ds[c], (const float4 *)dxp, dy, sh.rows, sh.k);
        if (U == 4 && Rr == 4) t16_gemv4<4, 4><<<b4, 128>>>(dw[c], ds[c], (const float4 *)dxp, dy, sh.rows, sh.k);
        if (U == 3 && Rr == 4) t16_gemv4<4, 3><<<b4, 128>>>(dw[c], ds[c], (const float4 *)dxp, dy, sh.rows, sh.k);
      }
    };
    // R = rows per warp
    // R = rows per warp (v1); 100 + 10 U + R = v4 with U stripes loaded ahead (includes the xperm launch)
    std::vector<int> cfgs = {2, 124, 212, 222};
    if (argc > 2) { cfgs.clear(); for (char *t = argv[2]; *t;) { cfgs.push_back(strtol(t, &t, 10)); if (*t == ',') t++; } }
    for (int R : cfgs) {
      run(R, 0);
      CK(cudaDeviceSynchronize());
      std::vector<float> hy(sh.rows);
      CK(cudaMemcpy(hy.data(), dy, sh.rows * 4, cudaMemcpyDeviceToHost));
      // v5 is checked against the int8-quantized activations it computes with (same rounding as xquant)
      std::vector<float> xr = hx;
      if (R > 200)
        for (int b = 0; b < sh.k / 256; b++) {
          float m = 0;
          for (int i = 0; i < 256; i++) m = fmaxf(m, fabsf(hx[b * 256 + i]));
          const float d = m / 127.f, id = m > 0 ? 127.f / m : 0.f;
          for (int i = 0; i < 256; i++) xr[b * 256 + i] = nearbyintf(hx[b * 256 + i] * id) * d;
        }
      double maxrel = 0;
      for (int r = 0; r < sh.rows; r += 97) {
        const double want = ref_row(hw.data() + (size_t)r * wpr, hs[r], xr.data(), sh.k);
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
      printf("%-8s %5.1f MB  cfg %3d: %6.1f us  %5.0f GB/s  (floor at 936 GB/s: %.1f us)  max rel err %.1e\n", sh.what, bytes / 1e6, R, us,
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
    cudaFree(dxp);
    cudaFree(xq8);
    cudaFree(xqd);
    cudaFree(xqs);
    cudaFree(dy);
  }
  return 0;
}
