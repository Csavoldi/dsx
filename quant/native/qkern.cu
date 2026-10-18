// qkern.cu - native kernels for the V4.1 expert quantizer (v41_quant.py), batched over experts.
//
// gptq_block: one 128-column block of blocked GPTQ (Frantar et al.) for B experts at once. A warp owns a row; its 128
// columns live in registers (4 per lane) and the block of H^-1 in shared memory, so the column-sequential error
// feedback is register FMAs and one warp shuffle per column instead of ~8 tensor kernels per column. The trailing
// update W[:, i2:] -= E @ Hinv[i1:i2, i2:] is a batched GEMM done by the caller.
//
// search_scales: the static-group scale search (diag(H)-weighted, clip ratios x denominators x signs) in one pass:
// a thread per (row, group) evaluates every candidate on its group in registers.
//
// Layouts (row-major, fp32): W, S, Q [B][R][C]; E [B][R][128]; lo, hi [B][R]; Hinv [B][C][C] (upper Cholesky of H^-1).
#include <cuda_runtime.h>
#include <stdint.h>

#define GB 128          // GPTQ block width (columns)
#define ROWS 8          // rows (warps) per CTA

extern "C" __global__ void __launch_bounds__(32 * ROWS) gptq_block_k(const float *__restrict__ Hinv, int C, int i1, float *__restrict__ W,
                                                                     const float *__restrict__ S, const float *__restrict__ lo,
                                                                     const float *__restrict__ hi, float *__restrict__ Q,
                                                                     float *__restrict__ E, int R) {
  extern __shared__ float hb[];  // [GB][GB + 1]
  const int b = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float *Hb = Hinv + (size_t)b * C * C;
  for (int t = threadIdx.x; t < GB * GB; t += blockDim.x) {
    int ii = t / GB, jj = t % GB;
    hb[ii * (GB + 1) + jj] = Hb[(size_t)(i1 + ii) * C + i1 + jj];
  }
  __syncthreads();
  const int r = blockIdx.x * ROWS + warp;
  if (r >= R) return;
  const size_t row = ((size_t)b * R + r) * C + i1;
  float w[4], s[4];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    w[k] = W[row + lane + 32 * k];
    s[k] = S[row + lane + 32 * k];
  }
  const float l = lo[(size_t)b * R + r], h = hi[(size_t)b * R + r];
  float *er = E + ((size_t)b * R + r) * GB;
  for (int i = 0; i < GB; i++) {
    const int owner = i & 31, kk = i >> 5;
    float err = 0.f, q = 0.f;
    if (lane == owner) {
      float wi = w[kk], si = s[kk];
      q = fminf(fmaxf(rintf(wi / si), l), h) * si;
      err = (wi - q) / hb[i * (GB + 1) + i];
      Q[row + i] = q;
      er[i] = err;
    }
    err = __shfl_sync(0xffffffffu, err, owner);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      int c = lane + 32 * k;
      if (c > i) w[k] -= err * hb[i * (GB + 1) + c];
    }
  }
}

// one thread per (expert-row, group): best scale over sign x denominator x clip, error weighted by d (diag H)
extern "C" __global__ void search_scales_k(const float *__restrict__ W, const float *__restrict__ d, int R, int C, int G, float lo,
                                           float hi, const float *__restrict__ clip, int nclip, float *__restrict__ out) {
  const int ng = C / G;
  const long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= (long)R * ng) return;
  const int r = id / ng, g = id % ng;
  const float *wg = W + (size_t)r * C + (size_t)g * G;
  const float *dg = d + (size_t)g * G;
  float amax = 0.f;
  for (int j = 0; j < G; j++) amax = fmaxf(amax, fabsf(wg[j]));
  amax = fmaxf(amax, 1e-12f);
  // denominators: sorted unique {hi, -lo}; signs: +1 only for a symmetric grid
  float den[2];
  int nden;
  float a = hi, b2 = -lo;
  if (a == b2) { den[0] = a; nden = 1; } else { den[0] = fminf(a, b2); den[1] = fmaxf(a, b2); nden = 2; }
  const int nsign = (lo != -hi) ? 2 : 1;
  float best_e = 0.f, best_s = 0.f;
  bool first = true;
  for (int si = 0; si < nsign; si++) {
    const float sign = si == 0 ? 1.f : -1.f;
    for (int di = 0; di < nden; di++) {
      for (int ci = 0; ci < nclip; ci++) {
        const float sc = sign * amax * clip[ci] / den[di];
        float e = 0.f;
        for (int j = 0; j < G; j++) {
          float q = fminf(fmaxf(rintf(wg[j] / sc), lo), hi) * sc;
          float df = wg[j] - q;
          e += df * df * dg[j];
        }
        if (first || e < best_e) { best_e = e; best_s = sc; first = false; }
      }
    }
  }
  out[id] = best_s;
}

extern "C" int ds41q_gptq_block(const float *Hinv, int C, int i1, float *W, const float *S, const float *lo, const float *hi, float *Q,
                                float *E, int R, int B, uintptr_t stream) {
  static bool attr = false;
  const int smem = GB * (GB + 1) * 4;
  if (!attr) {
    cudaFuncSetAttribute(gptq_block_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  dim3 grid((R + ROWS - 1) / ROWS, B);
  gptq_block_k<<<grid, 32 * ROWS, smem, (cudaStream_t)stream>>>(Hinv, C, i1, W, S, lo, hi, Q, E, R);
  return (int)cudaGetLastError();
}

extern "C" int ds41q_search_scales(const float *W, const float *d, int R, int C, int G, float lo, float hi, const float *clip, int nclip,
                                   float *out, uintptr_t stream) {
  long n = (long)R * (C / G);
  search_scales_k<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(W, d, R, C, G, lo, hi, clip, nclip, out);
  return (int)cudaGetLastError();
}
