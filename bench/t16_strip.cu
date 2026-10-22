// t16_strip.cu - GPU single-token GEMV over T16 experts in a strip layout, K = 1.5 / 2 / 2.5, both Hadamards included.
// Strip layout (our own quantizer's; LDLQ-compatible: each strip is inside one input row): W_inner [in][out], every
// input row cut into strips of 256 consecutive outputs; strip (k, sn) at index k * (out / 256) + sn, a circular bit
// stream of 256 K bits in u32 words, LSB first. Position t's bit width is K for integer K, and for K = KA + 0.5
// alternates KA (even t) and KA + 1 (odd t), as exllamav3's half-rate path. Its state = the 16 bits ending at its end
// bit (wrapping to the stream's end). Value = (lo + hi - 255) / 104.51 of the low 16 bits of state * 0x3A2F.
// Kernel: a warp walks one strip column (256 outputs) over a k slice; lane L owns outputs 8L .. 8L + 7; per 4 input
// rows it decodes its 8 windows per row, gathers each output's 4 low bytes and 4 high bytes (PRMT) and adds two dp4a
// against the 4 int8 activations. Activations: had128(x * suh) as int8 with a scale and a sum per 256 rows.
// Per call: prep + GEMV + post (sum of k slices, had128, * svh).
//
// build: nvcc -O3 -arch=sm_86 -o t16_strip t16_strip.cu
// run:   ./t16_strip [reps] [gate|down] [S] [K2 = 3 | 4 | 5 for K = 1.5 / 2 / 2.5]
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

// end bit (exclusive) of position t, K = K2 / 2
__host__ __device__ constexpr int pend(int K2, int t) { return K2 % 2 == 0 ? (t + 1) * (K2 / 2) : (t + 1) * (K2 / 2) + (t + 1) / 2; }

// the 8 products of positions 8 L .. 8 L + 7 of one strip (W = 4 K2 = 8 K words), low 16 bits valid
template <int K2>
__device__ __forceinline__ void prod8(const uint32_t *s, int L, uint32_t (&p)[8]) {
  constexpr int W = 4 * K2, NB = 256 * K2 / 2;
  // halves of 4 positions keep each window set inside 32 bits (16 + 3 * 3 bits at most for K = 2.5)
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int t0 = 8 * L + 4 * h;
    const int b0 = (pend(K2, t0) - 16 + NB) % NB;  // start bit of the first window
    const int i0 = b0 >> 5;
    const uint32_t v = __funnelshift_r(s[i0], s[(i0 + 1) % W], b0 & 31);
#pragma unroll
    for (int j = 0; j < 4; j++) p[4 * h + j] = (v >> (pend(K2, 4 * h + j) - pend(K2, 4 * h))) * MULC;
  }
}

__device__ __forceinline__ int dp4a_us(uint32_t a, int b, int c) {  // u8 x s8
  int d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

template <int K2>
__global__ void __launch_bounds__(128) strip_gemv(const uint32_t *__restrict__ w, const int8_t *__restrict__ xq, const float *__restrict__ xd,
                                                  const int *__restrict__ xs, float *__restrict__ part, int in, int out, int E, int S) {
  constexpr int W = 4 * K2;
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int sns = out / 256, groups = in / 256, cols = E * sns;
  if (warp >= cols * S) return;
  const int s = warp / cols, cl = warp % cols, e = cl / sns, sn = cl % sns;
  const int g0 = groups * s / S, g1 = groups * (s + 1) / S;
  const uint32_t *we = w + (size_t)e * in * sns * W + (size_t)sn * W;
  const uint32_t *xe = (const uint32_t *)(xq + (size_t)e * in);
  float f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int gr = g0; gr < g1; gr++) {
    int acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 2
    for (int k = 256 * gr; k < 256 * gr + 256; k += 4) {
      uint32_t p[4][8];
#pragma unroll
      for (int r = 0; r < 4; r++) prod8<K2>(we + (size_t)(k + r) * sns * W, lane, p[r]);
      const int x4 = (int)__ldg(xe + k / 4);
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const uint32_t a = __byte_perm(p[0][j], p[1][j], 0x5410), b = __byte_perm(p[2][j], p[3][j], 0x5410);
        acc[j] = dp4a_us(__byte_perm(a, b, 0x6420), x4, acc[j]);
        acc[j] = dp4a_us(__byte_perm(a, b, 0x7531), x4, acc[j]);
      }
    }
    const float d = __ldg(xd + (size_t)e * groups + gr);
    const int corr = 255 * __ldg(xs + (size_t)e * groups + gr);
#pragma unroll
    for (int j = 0; j < 8; j++) f[j] += d * (float)(acc[j] - corr);
  }
  float *pp = part + (size_t)s * E * out + (size_t)e * out + 256 * sn + 8 * lane;
#pragma unroll
  for (int j = 0; j < 8; j++) pp[j] = f[j] * (1.f / 104.51f);
}

__device__ __forceinline__ void had128_warp(float (&v)[4], int lane) {
  float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2, v[2] = a0 - a2, v[1] = a1 + a3, v[3] = a1 - a3;
#pragma unroll
  for (int m = 1; m < 32; m <<= 1)
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
      v[j] = (lane & m) ? o - v[j] : v[j] + o;
    }
#pragma unroll
  for (int j = 0; j < 4; j++) v[j] *= 0.08838834764831845f;
}

// x * suh, had128, int8 per 256 rows (natural order), group sums; one warp per 256-group
__global__ void prep(const float *__restrict__ x, const half *__restrict__ suh, int8_t *__restrict__ xq, float *__restrict__ xd, int *__restrict__ xs, int in, int E) {
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
  for (int h = 0; h < 2; h++) {
    char4 c;
    int q[4];
    for (int j = 0; j < 4; j++) q[j] = __float2int_rn(v[h][j] * id), sum += q[j];
    c.x = q[0], c.y = q[1], c.z = q[2], c.w = q[3];
    *(char4 *)(xq + (size_t)e * in + 256 * gr + 128 * h + 4 * lane) = c;
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) xd[(size_t)e * groups + gr] = m / 127.f, xs[(size_t)e * groups + gr] = sum;
}

__global__ void post(const float *__restrict__ part, const half *__restrict__ svh, float *__restrict__ y, int out, int E, int S) {
  const int lane = threadIdx.x & 31, wb = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (wb >= E * out / 128) return;
  const int e = wb / (out / 128), b = wb % (out / 128);
  const size_t i = (size_t)e * out + 128 * b + 4 * lane;
  float v[4] = {0, 0, 0, 0};
  for (int s = 0; s < S; s++)
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] += part[(size_t)s * E * out + i + j];
  had128_warp(v, lane);
#pragma unroll
  for (int j = 0; j < 4; j++) y[i + j] = v[j] * __half2float(svh[i + j]);
}

// ---- host reference
static float h2f(uint16_t h) { __half_raw r; r.x = h; return __half2float(__half(r)); }
static uint16_t f2h(float f) { __half_raw r = __half_raw(__float2half(f)); return r.x; }
static int tval(const uint32_t *s, int K2, int t) {
  const int W = 4 * K2, NB = 256 * K2 / 2, b0 = (pend(K2, t) - 16 + NB) % NB;
  uint32_t st = 0;
  for (int i = 0; i < 16; i++) {
    const int b = (b0 + i) % NB;
    st |= ((s[b >> 5] >> (b & 31)) & 1u) << i;
  }
  (void)W;
  const uint32_t p = (st * MULC) & 0xFFFF;
  return (int)(p & 0xFF) + (int)(p >> 8) - 255;
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
  const char *which = argc > 2 ? argv[2] : "gate";
  const int S = argc > 3 ? atoi(argv[3]) : 4, K2 = argc > 4 ? atoi(argv[4]) : 4;
  struct Shape { int in, out; const char *what; } shapes[] = {{5120, 2304, "gate x6"}, {2304, 5120, "down x6"}};
  const int E = 6, W = 4 * K2;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (auto sh : shapes) {
    if (strncmp(which, sh.what, 4)) continue;
    const int sns = sh.out / 256;
    const size_t words = (size_t)E * sh.in * sns * W, bytes = words * 4;
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
    float *dx, *dpart, *dy, *dxd;
    half *dsu, *dsv;
    int8_t *dxq;
    int *dxs;
    CK(cudaMalloc(&dx, sh.in * 4));
    CK(cudaMalloc(&dsu, hsu.size() * 2));
    CK(cudaMalloc(&dsv, hsv.size() * 2));
    CK(cudaMalloc(&dxq, (size_t)E * sh.in));
    CK(cudaMalloc(&dxd, (size_t)E * sh.in / 256 * 4));
    CK(cudaMalloc(&dxs, (size_t)E * sh.in / 256 * 4));
    CK(cudaMalloc(&dpart, (size_t)32 * E * sh.out * 4));
    CK(cudaMalloc(&dy, (size_t)E * sh.out * 4));
    CK(cudaMemcpy(dx, hx.data(), sh.in * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsu, hsu.data(), hsu.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dsv, hsv.data(), hsv.size() * 2, cudaMemcpyHostToDevice));
    auto run = [&](int c) {
      prep<<<(E * sh.in / 256 * 32 + 127) / 128, 128>>>(dx, dsu, dxq, dxd, dxs, sh.in, E);
      const int warps = E * sns * S, blocks = (warps * 32 + 127) / 128;
      if (K2 == 3) strip_gemv<3><<<blocks, 128>>>(dw[c], dxq, dxd, dxs, dpart, sh.in, sh.out, E, S);
      if (K2 == 4) strip_gemv<4><<<blocks, 128>>>(dw[c], dxq, dxd, dxs, dpart, sh.in, sh.out, E, S);
      if (K2 == 5) strip_gemv<5><<<blocks, 128>>>(dw[c], dxq, dxd, dxs, dpart, sh.in, sh.out, E, S);
      post<<<(E * sh.out / 128 * 32 + 127) / 128, 128>>>(dpart, dsv, dy, sh.out, E, S);
    };
    run(0);
    CK(cudaDeviceSynchronize());
    // check expert 1 against the host decode with the same int8 activations
    std::vector<float> hy((size_t)E * sh.out), hd((size_t)E * sh.in / 256);
    std::vector<int8_t> hq((size_t)E * sh.in);
    CK(cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hq.data(), dxq, hq.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hd.data(), dxd, hd.size() * 4, cudaMemcpyDeviceToHost));
    double err = 0;
    {
      const int e = 1;
      std::vector<double> y(sh.out, 0.0);
      const uint32_t *we = hw.data() + (size_t)e * sh.in * sns * W;
      for (int k = 0; k < sh.in; k++) {
        const double xk = hq[(size_t)e * sh.in + k] * (double)hd[(size_t)e * sh.in / 256 + k / 256];
        for (int sn = 0; sn < sns; sn++)
          for (int t = 0; t < 256; t++) y[256 * sn + t] += xk * tval(we + ((size_t)k * sns + sn) * W, K2, t) / 104.51;
      }
      had128_host(y);
      double ymax = 0;
      for (int o = 0; o < sh.out; o++) ymax = fmax(ymax, fabs(y[o] * h2f(hsv[(size_t)e * sh.out + o])));
      for (int o = 0; o < sh.out; o++) err = fmax(err, fabs(hy[(size_t)e * sh.out + o] - y[o] * h2f(hsv[(size_t)e * sh.out + o])) / ymax);
    }
    for (int i = 0; i < 20; i++) run(i % nc);
    double us = 1e9;
    for (int t = 0; t < 5; t++) {
      CK(cudaEventRecord(e0));
      for (int i = 0; i < reps; i++) run(i % nc);
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      us = fmin(us, ms * 1e3 / reps);
    }
    printf("%-8s K=%.1f %5.1f MB S=%2d: %6.1f us per call (prep + GEMV + post) %5.0f GB/s  max err %.1e of max |y|\n", sh.what, K2 / 2.0, bytes / 1e6, S, us,
           bytes / us / 1e3, err);
    for (auto p : dw) cudaFree(p);
    cudaFree(dx); cudaFree(dsu); cudaFree(dsv); cudaFree(dxq); cudaFree(dxd); cudaFree(dxs); cudaFree(dpart); cudaFree(dy);
  }
  return 0;
}
