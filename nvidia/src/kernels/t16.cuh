// T16 routed experts (core/src/t16.rs, docs/t16.md): exllamav3's bitshift trellis with the T16 codebook.
// A matrix (k inputs, n outputs) is trellis | suh | svh: a circular stream of 128 k2 bits per 16 x 16 tile (k2 = 2K
// = 3, 4, 5); position t holds tile row (input) t % 16 and tile column (output) t / 16, and its state is the 16-bit
// window ending after its own bits; suh f16 [k], svh f16 [n]. Two trellis layouts:
//   stored (GGUF, RAM, CPU kernels; type ids 1010-1012): tile (kb, nb) at nb * k/16 + kb, bytes in stream order;
//   repacked (GPU store; 2000 + id = 3010-3012): exllamav3's own, tile (kb, nb) at kb * n/16 + nb, little-endian u32 words read
//   MSB first (repack_t16).
// y = had128(had128(x * suh) @ W_inner) * svh. Prefill computes it on the activations: t16_pre (x * suh, Hadamard),
// a GEMM with W_inner^T from dequant_t16_f16, then t16_post (Hadamard, * svh).

__device__ __forceinline__ int t16_bit_end(int k2, int t) {
  return (k2 & 1) ? (t + 1) * (k2 >> 1) + ((t + 1) >> 1) : (t + 1) * (k2 >> 1);
}

// the 16-bit state of position t of a tile (w: its u32 words, 4 k2 of them)
__device__ __forceinline__ uint32_t t16_state(const uint32_t *w, int k2, int t) {
  const int nb = 128 * k2, nw = 4 * k2;
  const int b0 = (t16_bit_end(k2, t) - 16 + nb) % nb;
  const int i0 = b0 >> 5, i1 = (i0 + 1) % nw, s = b0 & 31;
  const uint64_t v = ((uint64_t)w[i0] << 32) | w[i1];
  return (uint32_t)(v >> (48 - s)) & 0xFFFFu;
}

// exllamav3's T16 decode: fp16(fma(fp16(1024 + b), 0x20e6, 0xca1e)), b = byte sum of low16(state * 0x3A2F)
__device__ __forceinline__ __half t16_value(uint32_t st) {
  const uint32_t p = (st * 0x3A2Fu) & 0xFFFFu;
  const int b = (int)(p & 0xFF) + (int)(p >> 8);
  return __hfma(__int2half_rn(1024 + b), __ushort_as_half((unsigned short)0x20e6), __ushort_as_half((unsigned short)0xca1e));
}

__device__ __forceinline__ size_t t16_trellis_bytes(int k2, int k, int n) { return (size_t)(k / 16) * (n / 16) * 16 * k2; }

// state of position t from a stored-layout tile (bytes in stream order)
__device__ __forceinline__ uint32_t t16_state_stored(const unsigned char *tb, int k2, int t) {
  const int nb = 128 * k2, nby = 16 * k2;
  const int b0 = (t16_bit_end(k2, t) - 16 + nb) % nb, y = b0 >> 3;
  const uint32_t v = (uint32_t)tb[y] << 16 | (uint32_t)tb[(y + 1) % nby] << 8 | tb[(y + 2) % nby];
  return (v >> (8 - (b0 & 7))) & 0xFFFFu;
}

// W_inner^T as f16 [n][k] (row = output) for cuBLAS; repacked = 0 stored layout, 1 GPU layout; grid (k / 256, n),
// block 256
extern "C" __global__ void dequant_t16_f16(int k2, const unsigned char *w, __half *out, int n, int k, int repacked) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
  if (i >= k) return;
  const int ti = i >> 4, tj = j >> 4, t = (j & 15) * 16 + (i & 15);
  uint32_t st;
  if (repacked) st = t16_state((const uint32_t *)(w + ((size_t)ti * (n / 16) + tj) * 16 * k2), k2, t);
  else st = t16_state_stored(w + ((size_t)tj * (k / 16) + ti) * 16 * k2, k2, t);
  out[(size_t)j * k + i] = t16_value(st);
}

// stored layout -> GPU layout (exllamav3's tile order, little-endian u32 words); suh / svh copied. One thread per
// 4 bytes of the matrix (trellis | suh | svh, the same size).
extern "C" __global__ void repack_t16(int k2, const unsigned char *src, unsigned char *dst, int k, int n) {
  const size_t tbytes = t16_trellis_bytes(k2, k, n), total = tbytes + 2 * (size_t)k + 2 * (size_t)n;
  const size_t o = 4 * ((size_t)blockIdx.x * blockDim.x + threadIdx.x);
  if (o >= total) return;
  if (o >= tbytes) {
    *(uint32_t *)(dst + o) = *(const uint32_t *)(src + o);
    return;
  }
  const int tb = 16 * k2, kt = k / 16, nt = n / 16;
  const size_t tile = o / tb, in = o % tb;  // destination tile index (kb * nt + nb), byte within it
  const int kb = (int)(tile / nt), nb = (int)(tile % nt);
  const unsigned char *s = src + ((size_t)nb * kt + kb) * tb + in;
  *(uint32_t *)(dst + o) = (uint32_t)s[0] << 24 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 8 | s[3];
}

// dst = had128(src * suh) per row, f16; grid (cols / 128, rows), block 128 (dst may be src)
extern "C" __global__ void t16_pre(const __half *src, __half *dst, const __half *suh, int cols) {
  __shared__ float fb[128];
  const int t = threadIdx.x, c = blockIdx.x * 128 + t;
  const size_t i = (size_t)blockIdx.y * cols + c;
  fb[t] = __half2float(src[i]) * __half2float(suh[c]);
  __syncthreads();
#pragma unroll
  for (int h = 1; h < 128; h <<= 1) {
    float own = fb[t], other = fb[t ^ h];
    __syncthreads();
    fb[t] = (t & h) ? other - own : own + other;
    __syncthreads();
  }
  dst[i] = __float2half(fb[t] * 0.08838834764831845f);
}

// y = had128(y) * svh * gain per row, f32; grid (cols / 128, rows), block 128
extern "C" __global__ void t16_post(float *y, const __half *svh, int cols, float gain) {
  __shared__ float fb[128];
  const int t = threadIdx.x, c = blockIdx.x * 128 + t;
  const size_t i = (size_t)blockIdx.y * cols + c;
  fb[t] = y[i];
  __syncthreads();
#pragma unroll
  for (int h = 1; h < 128; h <<= 1) {
    float own = fb[t], other = fb[t ^ h];
    __syncthreads();
    fb[t] = (t & h) ? other - own : own + other;
    __syncthreads();
  }
  y[i] = fb[t] * 0.08838834764831845f * __half2float(svh[c]) * gain;
}

// ---------------------------------------------------------------- decode (docs/t16.md "v4")
// Device-driven like moe_*_fast: pick j of T x K (token j / K) runs if its expert is resident here (tab entry). Per
// pick and matrix: t16d_prep (x * suh, had128, int8 per 256 written as pair words [x_k x_k x_k+1 x_k+1], group scale
// and sum), t16d_gemv (column-major tiles, one window + one IMAD per weight, two rows' products packed and dp4a'd
// with a pair word, exact affine value 0.009567260742 b - 2.4375), t16d_post_gu (had128, svh, swiglu, pick weight),
// t16d_post_down (had128, svh, sum of the token's picks in pick order).

__device__ __forceinline__ void t16_had128_warp(float (&v)[4], int lane) {
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

__device__ __forceinline__ const unsigned char *t16_mat(const ExpSlot &s, const unsigned char *base, int role) {
  return base + (role == 0 ? s.gate : role == 1 ? s.up : s.down);
}

// x: [T][k] (per token, xpp = 0) or [T K][k] (per pick, xpp = 1). Out: xp [T K][k / 2] u32, xd / xs [T K][k / 256].
// One warp per (pick, 256-group); block 128.
extern "C" __global__ void t16d_prep(const float *__restrict__ x, int xpp, const int *__restrict__ sel, const ExpSlot *__restrict__ tab,
                                     const unsigned char *__restrict__ base, int layer, int E, int K, int T, int role, int k, int n, int k2,
                                     uint32_t *__restrict__ xp, float *__restrict__ xd, int *__restrict__ xs) {
  const int lane = threadIdx.x & 31, wg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, groups = k / 256;
  if (wg >= T * K * groups) return;
  const int j = wg / groups, gr = wg % groups;
  const ExpSlot s = tab[layer * E + sel[j]];
  if (s.gate == ~0ull) return;
  const __half *suh = (const __half *)(t16_mat(s, base, role) + t16_trellis_bytes(k2, k, n));
  const float *xr = x + (size_t)(xpp ? j : j / K) * k;
  float v[2][4];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const int i = 256 * gr + 128 * h + 4 * lane;
#pragma unroll
    for (int q = 0; q < 4; q++) v[h][q] = xr[i + q] * __half2float(suh[i + q]);
    t16_had128_warp(v[h], lane);
  }
  float m = 0.f;
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int q = 0; q < 4; q++) m = fmaxf(m, fabsf(v[h][q]));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const float id = m > 0.f ? 127.f / m : 0.f;
  int sum = 0;
#pragma unroll
  for (int h = 0; h < 2; h++) {
    int q[4];
#pragma unroll
    for (int qq = 0; qq < 4; qq++) q[qq] = __float2int_rn(v[h][qq] * id), sum += q[qq];
    const size_t r = (size_t)j * k + 256 * gr + 128 * h + 4 * lane;
    const uint32_t b0 = q[0] & 255, b1 = q[1] & 255, b2 = q[2] & 255, b3 = q[3] & 255;
    xp[r / 2] = b0 | b0 << 8 | b1 << 16 | b1 << 24;
    xp[r / 2 + 1] = b2 | b2 << 8 | b3 << 16 | b3 << 24;
  }
  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if (lane == 0) xd[(size_t)j * groups + gr] = m / 127.f, xs[(size_t)j * groups + gr] = sum;
}

__device__ __forceinline__ uint32_t t16_fsr(uint32_t lo, uint32_t hi, int s) { return (uint32_t)((((uint64_t)hi << 32) | lo) >> s); }
__host__ __device__ constexpr int t16_hend(int K2, int t) { return K2 % 2 ? (t + 1) * (K2 / 2) + (t + 1) / 2 : (t + 1) * (K2 / 2); }

// the 16 windows of tile column c (positions 16c .. 16c + 15) from three consecutive words; products * 0x3A2F
template <int K2>
__device__ __forceinline__ void t16_prod16(const uint32_t *ptr, int i0, int i1, int i2, int sh, uint32_t (&p)[16]) {
  const uint32_t a = ptr[i0], b = ptr[i1], c = ptr[i2];
  const uint32_t lo = t16_fsr(c, b, sh), hi = t16_fsr(b, a, sh);
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int o = t16_hend(K2, 15) - t16_hend(K2, j);
    const uint32_t wj = o < 32 ? __funnelshift_r(lo, hi, o) : hi >> (o - 32);
    p[j] = wj * 0x3A2Fu;
  }
}

__device__ __forceinline__ int t16_dp4a(uint32_t a, int b, int c) {
  int d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

// part [S][T K][n]: split s of k's 256-groups. One warp per (pick, pair of tile columns, split); block 128.
template <int K2>
__device__ __forceinline__ void t16d_gemv_k(const uint4 *__restrict__ xp, const float *__restrict__ xd, const int *__restrict__ xs,
                                            const int *__restrict__ sel, const ExpSlot *__restrict__ tab, const unsigned char *__restrict__ base,
                                            int layer, int E, int K, int T, int role, int k, int n, int S, float *__restrict__ part) {
  const int lane = threadIdx.x & 31, warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nbs = n / 16, groups = k / 256, pairs = nbs / 2, picks = T * K;
  if (warp >= picks * pairs * S) return;
  const int s = warp / (picks * pairs), r = warp % (picks * pairs), j = r / pairs, nb = 2 * (r % pairs) + lane / 16, c = lane % 16;
  const ExpSlot sl = tab[layer * E + sel[j]];
  if (sl.gate == ~0ull) return;
  const int g0 = groups * s / S, g1 = groups * (s + 1) / S;
  constexpr int WT = 4 * K2, NB = 32 * WT;
  const int e15 = t16_hend(K2, 16 * c + 15) + NB, hw = (e15 - 1) / 32;
  const int i2 = hw % WT, i1 = (hw - 1) % WT, i0 = (hw - 2) % WT, sh = (hw + 1) * 32 - e15;
  const uint32_t *tp = (const uint32_t *)t16_mat(sl, base, role) + ((size_t)16 * g0 * nbs + nb) * WT;
  const uint32_t step = (uint32_t)nbs * WT;
  const uint4 *xe = xp + (size_t)j * (k / 8) + 2 * 16 * g0;
  float f = 0.f;
  for (int gr = g0; gr < g1; gr++) {
    int acc = 0;
#pragma unroll 4
    for (int i = 0; i < 16; i++, tp += step, xe += 2) {
      uint32_t p[16];
      t16_prod16<K2>(tp, i0, i1, i2, sh, p);
      const uint4 xa = __ldg(xe), xb = __ldg(xe + 1);
      const uint32_t xw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
      for (int q = 0; q < 8; q++) acc = t16_dp4a(__byte_perm(p[2 * q], p[2 * q + 1], 0x5410), (int)xw[q], acc);
    }
    f += __ldg(xd + (size_t)j * groups + gr) * (0.009567260742f * (float)acc - 2.4375f * (float)__ldg(xs + (size_t)j * groups + gr));
  }
  part[((size_t)s * picks + j) * n + 16 * nb + c] = f;
}
#define T16D_GEMV(K2)                                                                                                                   \
  extern "C" __global__ void __launch_bounds__(128) t16d_gemv_##K2(const uint4 *xp, const float *xd, const int *xs, const int *sel,   \
                                                                   const ExpSlot *tab, const unsigned char *base, int layer, int E,     \
                                                                   int K, int T, int role, int k, int n, int S, float *part) {          \
    t16d_gemv_k<K2>(xp, xd, xs, sel, tab, base, layer, E, K, T, role, k, n, S, part);                                                  \
  }
T16D_GEMV(3)
T16D_GEMV(4)
T16D_GEMV(5)

// h [T K][n] = bf16(silu(min(g, lim)) * clamp(u, +-lim) * w_j), g / u = had128(sum_s part) * svh. One warp per
// (pick, 128-block); block 128.
extern "C" __global__ void t16d_post_gu(const float *__restrict__ pg, const float *__restrict__ pu, int S, const int *__restrict__ sel,
                                        const float *__restrict__ wts, const ExpSlot *__restrict__ tab, const unsigned char *__restrict__ base,
                                        int layer, int E, int K, int T, int k, int n, int k2, float lim, float *__restrict__ h) {
  const int lane = threadIdx.x & 31, wb = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nblk = n / 128, picks = T * K;
  if (wb >= picks * nblk) return;
  const int j = wb / nblk, b = wb % nblk, i = 128 * b + 4 * lane;
  const ExpSlot s = tab[layer * E + sel[j]];
  if (s.gate == ~0ull) return;
  const size_t svo = t16_trellis_bytes(k2, k, n) + 2 * (size_t)k;
  const __half *svg = (const __half *)(base + s.gate + svo), *svu = (const __half *)(base + s.up + svo);
  float g[4] = {0, 0, 0, 0}, u[4] = {0, 0, 0, 0};
  for (int q = 0; q < S; q++)
#pragma unroll
    for (int t = 0; t < 4; t++) g[t] += pg[((size_t)q * picks + j) * n + i + t], u[t] += pu[((size_t)q * picks + j) * n + i + t];
  t16_had128_warp(g, lane);
  t16_had128_warp(u, lane);
  const float w = wts[j];
#pragma unroll
  for (int t = 0; t < 4; t++) {
    const float gv = fminf(g[t] * __half2float(svg[i + t]), lim), uv = fminf(fmaxf(u[t] * __half2float(svu[i + t]), -lim), lim);
    h[(size_t)j * n + i + t] = bf16r(gv / (1.f + expf(-gv)) * uv * w);
  }
}

// out [T][n] = sum over the token's resident picks (pick order) of had128(sum_s part) * svh; 0 if none. One warp per
// (token, 128-block); block 128.
extern "C" __global__ void t16d_post_down(const float *__restrict__ pd, int S, const int *__restrict__ sel, const ExpSlot *__restrict__ tab,
                                          const unsigned char *__restrict__ base, int layer, int E, int K, int T, int k, int n, int k2,
                                          float *__restrict__ out) {
  const int lane = threadIdx.x & 31, wb = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nblk = n / 128, picks = T * K;
  if (wb >= T * nblk) return;
  const int t = wb / nblk, b = wb % nblk, i = 128 * b + 4 * lane;
  const size_t svo = t16_trellis_bytes(k2, k, n) + 2 * (size_t)k;
  float o[4] = {0, 0, 0, 0};
  for (int kk = 0; kk < K; kk++) {
    const int j = t * K + kk;
    const ExpSlot s = tab[layer * E + sel[j]];
    if (s.gate == ~0ull) continue;
    float v[4] = {0, 0, 0, 0};
    for (int q = 0; q < S; q++)
#pragma unroll
      for (int r = 0; r < 4; r++) v[r] += pd[((size_t)q * picks + j) * n + i + r];
    t16_had128_warp(v, lane);
    const __half *sv = (const __half *)(base + s.down + svo);
#pragma unroll
    for (int r = 0; r < 4; r++) o[r] += v[r] * __half2float(sv[i + r]);
  }
#pragma unroll
  for (int r = 0; r < 4; r++) out[(size_t)t * n + i + r] = o[r];
}
