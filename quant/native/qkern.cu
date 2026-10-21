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

// warp per (row, group): lanes split the group's elements, each candidate's error is a warp sum (same candidates,
// same first-best rule as search_scales_k; 32x the parallelism of a thread per group)
extern "C" __global__ void search_scales_w(const float *__restrict__ W, const float *__restrict__ d, int R, int C, int G, float lo,
                                           float hi, const float *__restrict__ clip, int nclip, float *__restrict__ out) {
  const int ng = C / G, lane = threadIdx.x & 31;
  const long id = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (id >= (long)R * ng) return;
  const int r = id / ng, g = id % ng;
  const float *wg = W + (size_t)r * C + (size_t)g * G;
  const float *dg = d + (size_t)g * G;
  float wv[4], dv[4];  // G <= 128: up to 4 elements per lane
  int nv = 0;
  float amax = 0.f;
  for (int j = lane; j < G; j += 32, nv++) {
    wv[nv] = wg[j];
    dv[nv] = dg[j];
    amax = fmaxf(amax, fabsf(wv[nv]));
  }
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  amax = fmaxf(amax, 1e-12f);
  float den[2];
  int nden;
  if (hi == -lo) { den[0] = hi; nden = 1; } else { den[0] = fminf(hi, -lo); den[1] = fmaxf(hi, -lo); nden = 2; }
  const int nsign = (lo != -hi) ? 2 : 1;
  float best_e = 0.f, best_s = 0.f;
  bool first = true;
  for (int si = 0; si < nsign; si++) {
    const float sign = si == 0 ? 1.f : -1.f;
    for (int di = 0; di < nden; di++) {
      for (int ci = 0; ci < nclip; ci++) {
        const float sc = sign * amax * clip[ci] / den[di];
        float e = 0.f;
        for (int k = 0; k < nv; k++) {
          float q = fminf(fmaxf(rintf(wv[k] / sc), lo), hi) * sc;
          float df = wv[k] - q;
          e += df * df * dv[k];
        }
        for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
        if (first || e < best_e) { best_e = e; best_s = sc; first = false; }
      }
    }
  }
  if (lane == 0) out[id] = best_s;
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
  if (G >= 64 && G <= 128) {  // q3_k (16) keeps a thread per group: half a warp would idle
    search_scales_w<<<(unsigned)((n * 32 + 255) / 256), 256, 0, (cudaStream_t)stream>>>(W, d, R, C, G, lo, hi, clip, nclip, out);
  } else {
    search_scales_k<<<(unsigned)((n + 255) / 256), 256, 0, (cudaStream_t)stream>>>(W, d, R, C, G, lo, hi, clip, nclip, out);
  }
  return (int)cudaGetLastError();
}

// ---------------------------------------------------------------- IQ2_XXS (ggml's codebook: grid points of 8
// magnitudes from {8, 25, 43}, signs with even parity, scale db per 32 columns). Codebook GPTQ as in GPTVQ (van
// Baalen et al., arXiv 2402.15319) and QuIP# (Tseng et al., arXiv 2402.04396): each group of 8 columns is rounded
// jointly to the codebook, then its error is fed back column by column exactly as in scalar GPTQ.
// Group rounding (same rule as quant/experiments/quant_variants.py IQ2XXS.round_groups): signs from the targets; with
// an odd count of negatives the sign of the element with the smallest d |t| flips; then the grid point closest in
// d-weighted squared error to the (sign-corrected) magnitudes / db, first best on ties.
// grid: [256][8] float magnitudes (from v41_iqgrid.py).

__device__ __forceinline__ unsigned iq2_signs(const float (&t)[8], const float (&d)[8]) {
  unsigned neg = 0;
  float mc = 0.f;
  int mj = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    if (t[j] < 0.f) neg |= 1u << j;
    float c = d[j] * fabsf(t[j]);
    if (j == 0 || c < mc) { mc = c; mj = j; }
  }
  if (__popc(neg) & 1) neg ^= 1u << mj;
  return neg;
}

// one thread per (row, 32-column block): best db = amax / c over the candidates (first best), error = sum over the
// block's 4 groups of the d-weighted squared error of the rounded group
extern "C" __global__ void iq2_scales_k(const float *__restrict__ W, const float *__restrict__ d, int R, int C,
                                        const float *__restrict__ grid, const float *__restrict__ cand, int ncand,
                                        float *__restrict__ out) {
  __shared__ float gs[256 * 8];
  for (int i = threadIdx.x; i < 256 * 8; i += blockDim.x) gs[i] = grid[i];
  __syncthreads();
  const int nb = C / 32;
  const long id = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= (long)R * nb) return;
  const int r = id / nb, b = id % nb;
  const float *w = W + (size_t)r * C + (size_t)b * 32, *dd = d + (size_t)b * 32;
  float amax = 0.f;
  for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(w[j]));
  amax = fmaxf(amax, 1e-12f);
  float best_e = 0.f, best_db = 0.f;
  for (int ci = 0; ci < ncand; ci++) {
    const float db = amax / cand[ci];
    float e = 0.f;
    for (int g = 0; g < 4; g++) {
      float t[8], dv[8];
#pragma unroll
      for (int j = 0; j < 8; j++) { t[j] = w[8 * g + j]; dv[j] = dd[8 * g + j]; }
      unsigned neg = iq2_signs(t, dv);
#pragma unroll
      for (int j = 0; j < 8; j++) t[j] = (neg >> j & 1) ? -t[j] : t[j];
      float be = 0.f;
      for (int k = 0; k < 256; k++) {
        float ek = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) { float df = t[j] - db * gs[8 * k + j]; ek += dv[j] * df * df; }
        if (k == 0 || ek < be) be = ek;
      }
      e += be;
    }
    if (ci == 0 || e < best_e) { best_e = e; best_db = db; }
  }
  out[id] = best_db;
}

// one 128-column block of codebook GPTQ for B experts: a warp per row (columns lane + 32 k in registers), H^-1 block in
// shared memory. DB [B][R][C/32] effective scales; writes Q, E (as gptq_block_k) and per group the grid index and the
// 7-bit sign pattern (IDX, SGN [B][R][C/8], uint8).
extern "C" __global__ void __launch_bounds__(32 * ROWS) gptq_iq2_block_k(const float *__restrict__ Hinv, int C, int i1, float *__restrict__ W,
                                                                         const float *__restrict__ DB, const float *__restrict__ d,
                                                                         const float *__restrict__ grid, float *__restrict__ Q,
                                                                         float *__restrict__ E, uint8_t *__restrict__ IDX,
                                                                         uint8_t *__restrict__ SGN, int R) {
  extern __shared__ float hb[];  // [GB][GB + 1], then the grid [256][8]
  float *gs = hb + GB * (GB + 1);
  const int b = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float *Hb = Hinv + (size_t)b * C * C;
  for (int t = threadIdx.x; t < GB * GB; t += blockDim.x) {
    int ii = t / GB, jj = t % GB;
    hb[ii * (GB + 1) + jj] = Hb[(size_t)(i1 + ii) * C + i1 + jj];
  }
  for (int t = threadIdx.x; t < 256 * 8; t += blockDim.x) gs[t] = grid[t];
  __syncthreads();
  const int r = blockIdx.x * ROWS + warp;
  if (r >= R) return;
  const size_t rowi = (size_t)b * R + r, row = rowi * C + i1;
  float w[4];
#pragma unroll
  for (int k = 0; k < 4; k++) w[k] = W[row + lane + 32 * k];
  float *er = E + rowi * GB;
#pragma unroll
  for (int kk = 0; kk < 4; kk++) {
    for (int g = 0; g < 4; g++) {
      const int i0 = 32 * kk + 8 * g;  // first column of the group in the block
      const float db = DB[rowi * (C / 32) + (i1 + i0) / 32];
      float t[8], dv[8];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        t[j] = __shfl_sync(0xffffffffu, w[kk], 8 * g + j);
        dv[j] = __ldg(d + (size_t)b * C + i1 + i0 + j);
      }
      const unsigned neg = iq2_signs(t, dv);
#pragma unroll
      for (int j = 0; j < 8; j++) t[j] = (neg >> j & 1) ? -t[j] : t[j];
      // lane evaluates grid points lane + 32 m
      float be = 0.f;
      int bk = 0;
#pragma unroll
      for (int m = 0; m < 8; m++) {
        const int k = lane + 32 * m;
        float ek = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) { float df = t[j] - db * gs[8 * k + j]; ek += dv[j] * df * df; }
        if (m == 0 || ek < be) { be = ek; bk = k; }
      }
      for (int o = 16; o; o >>= 1) {  // argmin, smaller index on ties (first best)
        float oe = __shfl_xor_sync(0xffffffffu, be, o);
        int ok = __shfl_xor_sync(0xffffffffu, bk, o);
        if (oe < be || (oe == be && ok < bk)) { be = oe; bk = ok; }
      }
      if (lane == 0) {
        IDX[rowi * (C / 8) + (i1 + i0) / 8] = (uint8_t)bk;
        SGN[rowi * (C / 8) + (i1 + i0) / 8] = (uint8_t)(neg & 127);
      }
      // error feedback, column by column (the group's later columns are already rounded: only their errors change)
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const int i = i0 + j;
        const float q = ((neg >> j & 1) ? -db : db) * gs[8 * bk + j];
        float err = 0.f;
        if (lane == 8 * g + j) {
          err = (w[kk] - q) / hb[i * (GB + 1) + i];
          Q[row + i] = q;
          er[i] = err;
        }
        err = __shfl_sync(0xffffffffu, err, 8 * g + j);
#pragma unroll
        for (int k = 0; k < 4; k++) {
          int c = lane + 32 * k;
          if (c > i) w[k] -= err * hb[i * (GB + 1) + c];
        }
      }
    }
  }
}

extern "C" int ds41q_iq2_scales(const float *W, const float *d, int R, int C, const float *grid, const float *cand, int ncand,
                                float *out, uintptr_t stream) {
  long n = (long)R * (C / 32);
  iq2_scales_k<<<(unsigned)((n + 127) / 128), 128, 0, (cudaStream_t)stream>>>(W, d, R, C, grid, cand, ncand, out);
  return (int)cudaGetLastError();
}

extern "C" int ds41q_gptq_iq2_block(const float *Hinv, int C, int i1, float *W, const float *DB, const float *d, const float *grid,
                                    float *Q, float *E, uint8_t *IDX, uint8_t *SGN, int R, int B, uintptr_t stream) {
  static bool attr = false;
  const int smem = (GB * (GB + 1) + 256 * 8) * 4;
  if (!attr) {
    cudaFuncSetAttribute(gptq_iq2_block_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = true;
  }
  dim3 grid_((R + ROWS - 1) / ROWS, B);
  gptq_iq2_block_k<<<grid_, 32 * ROWS, smem, (cudaStream_t)stream>>>(Hinv, C, i1, W, DB, d, grid, Q, E, IDX, SGN, R);
  return (int)cudaGetLastError();
}
