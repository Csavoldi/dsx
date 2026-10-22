// t16_tile.cu - GPU single-token GEMV over T16 experts in exllamav3's real layout (go/no-go next to dsx's IQ tiles).
// Tiles (exllamav3 v1.6.4): W [in][out] as 16x16 tiles, row-major tile order [in/16][out/16], each a circular bit
// stream of 256 K bits (K = 2: 16 u32). Lane L of a warp decodes positions 8L .. 8L + 7 exactly as exl3_dq.cuh
// dq8_aligned_2bits does; those 8 values are the lane's two f16 B fragments of mma.m16n8k16 for the tile's columns 0-7
// and 8-15 (exl3's fragment order: lane L <-> column L / 4, rows 2q, 2q + 1, 2q + 8, 2q + 9). T16 value of a 16-bit
// state s: fp16 fma(1024 + bytesum(lo16(s * 0x3A2F)), 0x20e6, 0xca1e), as the patched codebook.cuh.
// The activation row is row 0 of the A fragment. A warp walks one 16-column strip over a k slice (S slices per strip,
// partials summed by the post kernel: no atomics). Per call: prep (x * suh, had128 per expert) + GEMV + post (sum of
// slices, had128, * svh). Shapes: 6 experts' gate [5120 -> 2304] and down [2304 -> 5120].
//
// build: nvcc -O3 -arch=sm_86 -o t16_tile t16_tile.cu
// run:   ./t16_tile [reps] [gate|down|both] [S slices per strip] [U k tiles per step; 9 = v2 int8 mma]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
#define MULC 0x3A2Fu
#define WORDS 16  // u32 per tile at K = 2

__device__ __forceinline__ uint32_t fshift(uint32_t b, uint32_t a, int s) { return (uint32_t)((((uint64_t)a << 32) | b) >> s); }

// T16 value of state w (low 16 bits used): fp16(fma(1024 + bytesum, k_inv, k_bias)); two at a time as half2
__device__ __forceinline__ half2 t16_2(uint32_t w0, uint32_t w1) {
  const uint32_t s0 = __dp4a(w0 * MULC, 0x0101u, 0x6400u), s1 = __dp4a(w1 * MULC, 0x0101u, 0x6400u);
  const half2 h = __halves2half2(__ushort_as_half((unsigned short)s0), __ushort_as_half((unsigned short)s1));
  return __hfma2(h, __half2half2(__ushort_as_half(0x20e6)), __half2half2(__ushort_as_half(0xca1e)));
}

// exl3_dq.cuh dq8_aligned_2bits: positions t, t + 1, ..., t + 7 (t = 8 lane) -> b0 = {v0 v1}, b1 = {v2 v3} (cols 0-7),
// b2 = {v4 v5}, b3 = {v6 v7} (cols 8-15)
__device__ __forceinline__ void dq8_k2(const uint32_t *ptr, int t, uint32_t (&b)[4]) {
  const uint32_t i1 = t >> 4, i0 = (i1 + 15) & 15;
  uint32_t a = ptr[i0], c = ptr[i1];
  c = fshift(c, a, ((~t) & 8) << 1);
  const uint32_t w7 = c, w6 = c >> 2, w5 = c >> 4, w4 = c >> 6, w3 = c >> 8, w2 = c >> 10, w1 = c >> 12, w0 = c >> 14;
  half2 h[4] = {t16_2(w0, w1), t16_2(w2, w3), t16_2(w4, w5), t16_2(w6, w7)};
  memcpy(b, h, 16);
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// GEMV: E experts, each W_e [in][out] in tiles (row-major tile order), x_e [in] f16 (already x * suh, had128).
// Warp w: strip nb (16 outputs of one expert) and k slice s; part[s][e * out + col] f32.
template <int U>
__global__ void __launch_bounds__(128) t16_gemv(const uint32_t *__restrict__ w, const half *__restrict__ x, float *__restrict__ part,
                                                int in, int out, int E, int S) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nbs = out / 16, kbs = in / 16, strips = E * nbs;
  if (warp >= strips * S) return;
  const int s = warp / strips, st = warp % strips, e = st / nbs, nb = st % nbs;
  const int kb0 = (int)((long)kbs * s / S), kb1 = (int)((long)kbs * (s + 1) / S);
  const uint32_t *we = w + (size_t)e * kbs * nbs * WORDS;
  const half *xe = x + (size_t)e * in;
  const int q = lane & 3, g = lane >> 2;
  float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
  for (int kb = kb0; kb < kb1; kb += U) {
    uint32_t bw[U][4];
    uint32_t a[U][4];
#pragma unroll
    for (int u = 0; u < U; u++) {
      if (kb + u < kb1) {
        dq8_k2(we + ((size_t)(kb + u) * nbs + nb) * WORDS, 8 * lane, bw[u]);
        // A row 0 = x[16 kb .. +15]: lanes 0-3 (g == 0) hold cols 2q, 2q+1 (a0) and 2q+8, 2q+9 (a2)
        const uint32_t *xr = (const uint32_t *)(xe + 16 * (kb + u));
        a[u][0] = g == 0 ? xr[q] : 0u;
        a[u][1] = 0u;
        a[u][2] = g == 0 ? xr[q + 4] : 0u;
        a[u][3] = 0u;
      } else {
        bw[u][0] = bw[u][1] = bw[u][2] = bw[u][3] = 0u;
        a[u][0] = a[u][1] = a[u][2] = a[u][3] = 0u;
      }
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      mma16816(c0, a[u], bw[u][0], bw[u][1]);
      mma16816(c1, a[u], bw[u][2], bw[u][3]);
    }
  }
  // C row 0: lanes 0-3, cols 2q, 2q + 1 (c[0], c[1])
  if (g == 0) {
    float *p = part + (size_t)s * E * out + (size_t)e * out + 16 * nb;
    p[2 * q] = c0[0], p[2 * q + 1] = c0[1], p[8 + 2 * q] = c1[0], p[8 + 2 * q + 1] = c1[1];
  }
}

// 128-wide Walsh-Hadamard of one 128-block per warp (4 values per lane), scale 1/sqrt(128)
__device__ __forceinline__ void had128_warp(float (&v)[4], int lane) {
  // lane holds elements 4 lane + j; first the in-lane butterflies (h = 1, 2), then across lanes (h = 4 .. 64)
  float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2, v[2] = a0 - a2, v[1] = a1 + a3, v[3] = a1 - a3;
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
      v[j] = (lane & m) ? o - v[j] : v[j] + o;
    }
  }
#pragma unroll
  for (int j = 0; j < 4; j++) v[j] *= 0.08838834764831845f;
}

// prep: xh[e][i] = had128(x * suh_e) as f16; one warp per 128-block
__global__ void prep(const float *__restrict__ x, const half *__restrict__ suh, half *__restrict__ xh, int in, int E) {
  const int lane = threadIdx.x & 31, wb = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (wb >= E * in / 128) return;
  const int e = wb / (in / 128), b = wb % (in / 128), i = 128 * b + 4 * lane;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; j++) v[j] = x[i + j] * __half2float(suh[(size_t)e * in + i + j]);
  had128_warp(v, lane);
#pragma unroll
  for (int j = 0; j < 4; j++) xh[(size_t)e * in + i + j] = __float2half(v[j]);
}

// post: y[e][o] = had128(sum_s part[s][e][o]) * svh_e; one warp per 128-block
__global__ void post(const float *__restrict__ part, const half *__restrict__ svh, float *__restrict__ y, int out, int E, int S) {
  const int lane = threadIdx.x & 31, wb = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (wb >= E * out / 128) return;
  const int e = wb / (out / 128), b = wb % (out / 128), i = (size_t)e * out + 128 * b + 4 * lane;
  float v[4] = {0, 0, 0, 0};
  for (int s = 0; s < S; s++)
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] += part[(size_t)s * E * out + i + j];
  had128_warp(v, lane);
#pragma unroll
  for (int j = 0; j < 4; j++) y[i + j] = v[j] * __half2float(svh[i + j]);
}


// ---- v2: int8 mma. Lane L's 4 weights per column (tile rows 2q, 2q+1, 2q+8, 2q+9; q = L & 3) are one u32 of the
// mma.m16n8k32 B fragment, two tiles (kb, kb + 1) fill b0 / b1. The low bytes and the high bytes of the 4 products are
// two u8 B words (4 PRMT), so mma(A, B_lo) + mma(A, B_hi) = sum x (lo + hi); the -255 sum(x) term is one constant per
// 256-row group. A = int8 activations (per-256 scale) stored in that row order: 16 bytes per k block, [0 1 8 9 2 3 10 11
// 4 5 12 13 6 7 14 15]; lanes 0-3 hold row 0. Value = exact (lo + hi - 255) / 104.51 (not exllamav3's fp16 rounding).
__device__ __forceinline__ void mma_s8u8(int (&c)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

// products of lane's 8 positions of one tile (K = 2), as dq8_aligned_2bits orders them, low 16 bits valid
__device__ __forceinline__ void prod8_k2(const uint32_t *ptr, int t, uint32_t (&p)[8]) {
  const uint32_t i1 = t >> 4, i0 = (i1 + 15) & 15;
  uint32_t a = ptr[i0], c = ptr[i1];
  c = fshift(c, a, ((~t) & 8) << 1);
#pragma unroll
  for (int j = 0; j < 8; j++) p[j] = (c >> (2 * (7 - j))) * MULC;
}
// lo / hi bytes of p[o..o+3]
__device__ __forceinline__ void lohi(const uint32_t (&p)[8], int o, uint32_t &lo, uint32_t &hi) {
  const uint32_t x = __byte_perm(p[o], p[o + 1], 0x5410), y = __byte_perm(p[o + 2], p[o + 3], 0x5410);
  lo = __byte_perm(x, y, 0x6420), hi = __byte_perm(x, y, 0x7531);
}

// xq [E][in] int8 permuted per k block, xd [E][in/256] scale, xs [E][in/256] sum of the group's int8
__global__ void __launch_bounds__(128) t16_gemv2(const uint32_t *__restrict__ w, const int8_t *__restrict__ xq, const float *__restrict__ xd,
                                                 const int *__restrict__ xs, float *__restrict__ part, int in, int out, int E, int S) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nbs = out / 16, kbs = in / 16, groups = in / 256, strips = E * nbs;
  if (warp >= strips * S) return;
  const int s = warp / strips, st = warp % strips, e = st / nbs, nb = st % nbs;
  const int g0 = groups * s / S, g1 = groups * (s + 1) / S;
  const uint32_t *we = w + (size_t)e * kbs * nbs * WORDS;
  const uint32_t *xe = (const uint32_t *)(xq + (size_t)e * in);
  const int q = lane & 3, g = lane >> 2;
  float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f;  // row 0 of the two n8 tiles: cols 2q, 2q+1 / 8+2q, 8+2q+1
  for (int gr = g0; gr < g1; gr++) {
    int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
#pragma unroll 2
    for (int kb = 16 * gr; kb < 16 * gr + 16; kb += 2) {
      uint32_t pa[8], pb[8];
      prod8_k2(we + ((size_t)kb * nbs + nb) * WORDS, 8 * lane, pa);
      prod8_k2(we + ((size_t)(kb + 1) * nbs + nb) * WORDS, 8 * lane, pb);
      const uint32_t a0 = g == 0 ? __ldg(xe + 4 * kb + q) : 0u, a2 = g == 0 ? __ldg(xe + 4 * (kb + 1) + q) : 0u;
      uint32_t la, ha, lb, hb;
      lohi(pa, 0, la, ha);
      lohi(pb, 0, lb, hb);
      mma_s8u8(c0, a0, a2, la, lb);
      mma_s8u8(c0, a0, a2, ha, hb);
      lohi(pa, 4, la, ha);
      lohi(pb, 4, lb, hb);
      mma_s8u8(c1, a0, a2, la, lb);
      mma_s8u8(c1, a0, a2, ha, hb);
    }
    const float d = __ldg(xd + (size_t)e * groups + gr);
    const int corr = 255 * __ldg(xs + (size_t)e * groups + gr);
    f0 += d * (float)(c0[0] - corr), f1 += d * (float)(c0[1] - corr), f2 += d * (float)(c1[0] - corr), f3 += d * (float)(c1[1] - corr);
  }
  if (g == 0) {
    float *p = part + (size_t)s * E * out + (size_t)e * out + 16 * nb;
    const float k = 1.f / 104.51f;
    p[2 * q] = f0 * k, p[2 * q + 1] = f1 * k, p[8 + 2 * q] = f2 * k, p[8 + 2 * q + 1] = f3 * k;
  }
}

// prep2: per expert and 256-group (one warp): x * suh, had128 on both halves, int8 with absmax scale, written in the
// permuted row order; group sum of the int8 values
__constant__ int c_perm[16] = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};  // tile row -> byte slot
__global__ void prep2(const float *__restrict__ x, const half *__restrict__ suh, int8_t *__restrict__ xq, float *__restrict__ xd, int *__restrict__ xs,
                      int in, int E) {
  const int lane = threadIdx.x & 31, wg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, groups = in / 256;
  if (wg >= E * groups) return;
  const int e = wg / groups, gr = wg % groups;
  float v[2][4];
  for (int h = 0; h < 2; h++) {
    const int i = 256 * gr + 128 * h + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; j++) v[h][j] = x[i + j] * __half2float(suh[(size_t)e * in + i + j]);
    had128_warp(v[h], lane);
  }
  float m = 0.f;
  for (int h = 0; h < 2; h++)
    for (int j = 0; j < 4; j++) m = fmaxf(m, fabsf(v[h][j]));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const float id = m > 0.f ? 127.f / m : 0.f;
  int sum = 0;
  for (int h = 0; h < 2; h++)
    for (int j = 0; j < 4; j++) {
      const int i = 128 * h + 4 * lane + j, qv = __float2int_rn(v[h][j] * id);  // index within the group
      sum += qv;
      xq[(size_t)e * in + 256 * gr + 16 * (i / 16) + c_perm[i % 16]] = (int8_t)qv;
    }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) xd[(size_t)e * groups + gr] = m / 127.f, xs[(size_t)e * groups + gr] = sum;
}

// ---- host reference (exl3_dq.cuh dq() indexing, reconstruct.cu fragment map)
static int pos_r[256], pos_c[256];
static void build_map() {
  for (int L = 0; L < 32; L++) {
    if (L & 4) continue;
    const int q = L % 4, c0 = L / 8, c1 = c0 + 4, r[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
    for (int j = 0; j < 8; j++) {
      const int row = r[j % 4], cc = j < 4 ? c0 : c1;
      pos_r[8 * L + j] = row, pos_c[8 * L + j] = 2 * cc;
      pos_r[8 * (L + 4) + j] = row, pos_c[8 * (L + 4) + j] = 2 * cc + 1;
    }
  }
}
static float h2f(uint16_t h) { __half_raw r; r.x = h; return __half2float(__half(r)); }
static uint16_t f2h(float f) { __half_raw r = __half_raw(__float2half(f)); return r.x; }
// fp16 value exactly as the device: fma in fp16 of (1024 + sum) * k_inv + k_bias, one rounding (computed in double)
static float t16_ref(uint32_t st) {
  const uint32_t p = (st * MULC) & 0xFFFF;
  const double v = (1024.0 + (p & 0xFF) + (p >> 8)) * (double)h2f(0x20e6) + (double)h2f(0xca1e);
  return h2f(f2h((float)v));
}
static uint32_t state_ref(const uint32_t *w, int t) {
  const int b0 = t * 2 + 2 - 16 + 512, b1 = b0 + 16, i0 = b0 / 32, i1 = (b1 - 1) / 32, s0 = (i1 + 1) * 32 - b1;
  const uint64_t m = ((uint64_t)w[i0 % WORDS] << 32) | w[i1 % WORDS];
  return (uint32_t)(m >> s0) & 0xFFFF;
}
static void had128_host(std::vector<double> &v) {
  for (size_t b = 0; b < v.size(); b += 128)
    for (int h = 1; h < 128; h <<= 1)
      for (int i = 0; i < 128; i += 2 * h)
        for (int j = i; j < i + h; j++) { double a = v[b + j], c = v[b + j + h]; v[b + j] = a + c, v[b + j + h] = a - c; }
  for (auto &x : v) x *= 0.08838834764831845;
}

int main(int argc, char **argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 100;
  const char *which = argc > 2 ? argv[2] : "both";
  const int S = argc > 3 ? atoi(argv[3]) : 4, U1 = argc > 4 ? atoi(argv[4]) : 0;  // U1: one unroll only
  build_map();
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  struct Shape { int in, out; const char *what; } shapes[] = {{5120, 2304, "gate x6"}, {2304, 5120, "down x6"}};
  const int E = 6;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (auto sh : shapes) {
    if (strcmp(which, "both") && strncmp(which, sh.what, 4)) continue;
    const size_t words = (size_t)E * (sh.in / 16) * (sh.out / 16) * WORDS, bytes = words * 4;
    const int nc = (int)ceil(256e6 / bytes);
    std::vector<uint32_t> hw(words);
    uint64_t z = 88172645463325252ull;
    for (auto &v : hw) { z ^= z << 13; z ^= z >> 7; z ^= z << 17; v = (uint32_t)z; }
    std::vector<uint16_t> hsu((size_t)E * sh.in), hsv((size_t)E * sh.out);
    for (size_t i = 0; i < hsu.size(); i++) hsu[i] = f2h((i * 7 % 5 ? 1.f : -1.f) * (0.8f + 0.001f * (i % 97)));
    for (size_t i = 0; i < hsv.size(); i++) hsv[i] = f2h((i * 11 % 3 ? 1.f : -1.f) * (0.01f + 0.0001f * (i % 89)));
    std::vector<float> hx(sh.in);
    for (int i = 0; i < sh.in; i++) hx[i] = sinf(i * 0.37f);
    std::vector<uint32_t *> dw(nc);
    for (int c = 0; c < nc; c++) { CK(cudaMalloc(&dw[c], bytes)); CK(cudaMemcpy(dw[c], hw.data(), bytes, cudaMemcpyHostToDevice)); }
    float *dx, *dpart, *dy;
    half *dsu, *dsv, *dxh;
    CK(cudaMalloc(&dx, sh.in * 4));
    CK(cudaMalloc(&dsu, hsu.size() * 2));
    CK(cudaMalloc(&dsv, hsv.size() * 2));
    CK(cudaMalloc(&dxh, (size_t)E * sh.in * 2));
    CK(cudaMalloc(&dpart, (size_t)16 * E * sh.out * 4));
    CK(cudaMalloc(&dy, (size_t)E * sh.out * 4));
    CK(cudaMemcpy(dx, hx.data(), sh.in * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsu, hsu.data(), hsu.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsv, hsv.data(), hsv.size() * 2, cudaMemcpyHostToDevice));
    int8_t *dxq;
    float *dxd;
    int *dxs;
    CK(cudaMalloc(&dxq, (size_t)E * sh.in));
    CK(cudaMalloc(&dxd, (size_t)E * sh.in / 256 * 4));
    CK(cudaMalloc(&dxs, (size_t)E * sh.in / 256 * 4));
    auto run2 = [&](int c) {
      prep2<<<(E * sh.in / 256 * 32 + 127) / 128, 128>>>(dx, dsu, dxq, dxd, dxs, sh.in, E);
      const int warps = E * (sh.out / 16) * S, blocks = (warps * 32 + 127) / 128;
      t16_gemv2<<<blocks, 128>>>(dw[c], dxq, dxd, dxs, dpart, sh.in, sh.out, E, S);
      post<<<(E * sh.out / 128 * 32 + 127) / 128, 128>>>(dpart, dsv, dy, sh.out, E, S);
    };
    if (U1 == 9) {  // v2
      run2(0);
      CK(cudaDeviceSynchronize());
      std::vector<float> hy((size_t)E * sh.out), hd((size_t)E * sh.in / 256);
      std::vector<int8_t> hq((size_t)E * sh.in);
      CK(cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hq.data(), dxq, hq.size(), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hd.data(), dxd, hd.size() * 4, cudaMemcpyDeviceToHost));
      const int e = 1, kbs = sh.in / 16, nbs = sh.out / 16, perm[16] = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};
      std::vector<double> xr(sh.in), y(sh.out, 0.0);
      for (int i = 0; i < sh.in; i++) xr[i] = hq[(size_t)e * sh.in + 16 * (i / 16) + perm[i % 16]] * (double)hd[(size_t)e * sh.in / 256 + i / 256];
      const uint32_t *we = hw.data() + (size_t)e * kbs * nbs * WORDS;
      for (int kb = 0; kb < kbs; kb++)
        for (int nb = 0; nb < nbs; nb++)
          for (int t = 0; t < 256; t++) {
            const uint32_t p = (state_ref(we + ((size_t)kb * nbs + nb) * WORDS, t) * MULC) & 0xFFFF;
            y[16 * nb + pos_c[t]] += xr[16 * kb + pos_r[t]] * (((p & 0xFF) + (p >> 8) - 255.0) / 104.51);
          }
      had128_host(y);
      double ymax = 0, err = 0;
      for (int o = 0; o < sh.out; o++) ymax = fmax(ymax, fabs(y[o] * h2f(hsv[(size_t)e * sh.out + o])));
      for (int o = 0; o < sh.out; o++) err = fmax(err, fabs(hy[(size_t)e * sh.out + o] - y[o] * h2f(hsv[(size_t)e * sh.out + o])) / ymax);
      for (int i = 0; i < 20; i++) run2(i % nc);
      double us = 1e9;
      for (int t = 0; t < 5; t++) {
        CK(cudaEventRecord(e0));
        for (int i = 0; i < reps; i++) run2(i % nc);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        us = fmin(us, ms * 1e3 / reps);
      }
      printf("%-8s %5.1f MB  v2 int8 mma S=%d: %6.1f us per call (prep + GEMV + post)  %5.0f GB/s  max err %.1e of max |y| (vs the same int8 x)\n", sh.what,
             bytes / 1e6, S, us, bytes / us / 1e3, err);
    }
    auto run = [&](int c, int U) {
      prep<<<(E * sh.in / 128 * 32 + 127) / 128, 128>>>(dx, dsu, dxh, sh.in, E);
      const int warps = E * (sh.out / 16) * S, blocks = (warps * 32 + 127) / 128;
      if (U == 1) t16_gemv<1><<<blocks, 128>>>(dw[c], dxh, dpart, sh.in, sh.out, E, S);
      if (U == 2) t16_gemv<2><<<blocks, 128>>>(dw[c], dxh, dpart, sh.in, sh.out, E, S);
      if (U == 4) t16_gemv<4><<<blocks, 128>>>(dw[c], dxh, dpart, sh.in, sh.out, E, S);
      post<<<(E * sh.out / 128 * 32 + 127) / 128, 128>>>(dpart, dsv, dy, sh.out, E, S);
    };
    for (int U : {1, 2, 4}) {
      if ((U1 && U != U1) || U1 == 9) continue;
      run(0, U);
      CK(cudaDeviceSynchronize());
      // check one expert's outputs against the host reference (fp16 inputs as the device rounds them)
      std::vector<float> hy((size_t)E * sh.out);
      CK(cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost));
      double maxrel = 0;
      {
        const int e = 1, kbs = sh.in / 16, nbs = sh.out / 16;
        std::vector<double> xh(sh.in), y(sh.out, 0.0);
        for (int i = 0; i < sh.in; i++) xh[i] = hx[i] * h2f(hsu[(size_t)e * sh.in + i]);
        had128_host(xh);
        for (auto &v : xh) v = h2f(f2h((float)v));
        const uint32_t *we = hw.data() + (size_t)e * kbs * nbs * WORDS;
        for (int kb = 0; kb < kbs; kb++)
          for (int nb = 0; nb < nbs; nb++)
            for (int t = 0; t < 256; t++) y[16 * nb + pos_c[t]] += xh[16 * kb + pos_r[t]] * t16_ref(state_ref(we + ((size_t)kb * nbs + nb) * WORDS, t));
        had128_host(y);
        double ymax = 0;
        for (int o = 0; o < sh.out; o++) ymax = fmax(ymax, fabs(y[o] * h2f(hsv[(size_t)e * sh.out + o])));
        for (int o = 0; o < sh.out; o++) maxrel = fmax(maxrel, fabs(hy[(size_t)e * sh.out + o] - y[o] * h2f(hsv[(size_t)e * sh.out + o])) / ymax);
      }
      for (int i = 0; i < 20; i++) run(i % nc, U);
      double us = 1e9;
      for (int t = 0; t < 5; t++) {
        CK(cudaEventRecord(e0));
        for (int i = 0; i < reps; i++) run(i % nc, U);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        us = fmin(us, ms * 1e3 / reps);
      }
      printf("%-8s %5.1f MB  S=%d U=%d: %6.1f us per call (prep + GEMV + post)  %5.0f GB/s  max err %.1e of max |y|\n", sh.what, bytes / 1e6, S, U, us,
             bytes / us / 1e3, maxrel);
    }
    for (auto p : dw) cudaFree(p);
    cudaFree(dx); cudaFree(dsu); cudaFree(dsv); cudaFree(dxh); cudaFree(dpart); cudaFree(dy);
  }
  return 0;
}
