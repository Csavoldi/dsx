// trellis_tile.c - CPU decode cost of T16 experts in exllamav3's real layout (go/no-go next to UD's IQ mix).
// Layout (exllamav3 v1.6.4, exl3_dq.cuh / reconstruct.cu): a matrix W [in][out] is stored as 16x16 tiles, tile
// (kb, nb) at index kb * (out / 16) + nb, each a circular bit stream of 256 K bits read through u32 words
// MSB-first; position t's 16-bit state is the window ending at bit (t + 1) K. Positions map to (row, col) of the
// tile through the tensor-core fragment order (reconstruct.cu). T16 value = (lo + hi - 255) / 104.51 of the low 16
// bits of state * 0x3A2F (exllamav3 rounds this to fp16; here the exact linear value, int arithmetic).
// Forward per matrix: y = had128(had128(x * suh) @ W) * svh.
// Kernel: activations permuted once per 16-row block into tile-position order (one int32 lane per position:
// bytes 0, 1 = x, bytes 2, 3 = 0), so vpdpbusd(product, xperm) adds x * (lo + hi) per position; every position keeps
// its own accumulator across the k blocks of one column of tiles (8 at a time in registers), folded to the 16
// output columns at the end. The -255 * sum(x) term is the same for every output (each column holds every row once).
//
// build: gcc -O3 -mavx2 -mavxvnni -mfma -fopenmp trellis_tile.c -o trellis_tile -lm
// run:   ./trellis_tile [experts] [calls] [threads]
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HID 5120
#define INT 2304
#define MUL 0x3A2Fu
#define K 2
#define WORDS (256 * K / 32)  // u32 per tile

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int pos_r[256], pos_c[256];  // tile position -> (row = input, col = output)

// reconstruct.cu: lane L of the warp decodes positions 8L .. 8L + 7 as frag0 = {t0 t1, t2 t3}, frag1 = {t4 t5, t6 t7};
// lanes with (L & 4) == 0 pair their values with lane L + 4's into half2s written at tile[r][c]
static void build_map(void) {
  for (int L = 0; L < 32; L++) {
    if (L & 4) continue;
    const int q = L % 4, c0 = L / 8, c1 = c0 + 4, r[4] = {2 * q, 2 * q + 1, 2 * q + 8, 2 * q + 9};
    for (int j = 0; j < 8; j++) {
      // value j of lane L (own) and of lane L + 4 (partner): rows r[j % 4], half2 column c0 (j < 4) or c1
      const int row = r[j % 4], cc = j < 4 ? c0 : c1;
      pos_r[8 * L + j] = row, pos_c[8 * L + j] = 2 * cc;
      pos_r[8 * (L + 4) + j] = row, pos_c[8 * (L + 4) + j] = 2 * cc + 1;
    }
  }
}

// exl3_dq.cuh dq(): the 16-bit state of position t in a tile of K-bit positions
static uint32_t state_ref(const uint32_t *w, int t) {
  const int b0 = t * K + K - 16 + 256 * K, b1 = b0 + 16, i0 = b0 / 32, i1 = (b1 - 1) / 32, s0 = (i1 + 1) * 32 - b1;
  const uint64_t m = ((uint64_t)w[i0 % WORDS] << 32) | w[i1 % WORDS];
  return (uint32_t)(m >> s0) & 0xFFFF;
}

static int tval(uint32_t s) {  // lo + hi - 255 of the product: the T16 weight times 104.51
  const uint32_t p = (s * MUL) & 0xFFFF;
  return (int)(p & 0xFF) + (int)(p >> 8) - 255;
}

// in-place 128-wide Walsh-Hadamard on n floats (n % 128 == 0), scaled 1/sqrt(128)
static void had128(float *v, int n) {
  for (int b = 0; b < n; b += 128) {
    float *x = v + b;
    for (int h = 1; h < 128; h <<= 1)
      for (int i = 0; i < 128; i += 2 * h)
        for (int j = i; j < i + h; j++) { float a = x[j], c = x[j + h]; x[j] = a + c; x[j + h] = a - c; }
    for (int i = 0; i < 128; i++) x[i] *= 0.08838834764831845f;
  }
}

typedef struct { uint32_t *w; int in, out; } Mat;

static Mat mk(int in, int out) {
  Mat m = {0, in, out};
  const size_t n = (size_t)(in / 16) * (out / 16) * WORDS;
  m.w = aligned_alloc(64, n * 4 + 64);  // + padding for v3's last load
  for (size_t i = 0; i < n; i++) m.w[i] = (uint32_t)(i * 2654435761u) ^ 0x5bd1e995u ^ (uint32_t)(i >> 7);
  return m;
}

// xperm [in / 16][256] int32: bytes 0, 1 = x[16 kb + row(t)], bytes 2, 3 = 0
static void permute_x(const int8_t *xq, int in, int32_t *xperm) {
  for (int kb = 0; kb < in / 16; kb++)
    for (int t = 0; t < 256; t++) {
      const uint8_t v = (uint8_t)xq[16 * kb + pos_r[t]];
      xperm[kb * 256 + t] = (int32_t)(v | (uint32_t)v << 8);
    }
}

// columns [nb0, nb1) of tiles: out[16 nb + c] = sum over k of x * (lo + hi) (the -255 sum(x) term is added later)
static void gemv_tiles(const Mat *m, const int32_t *xperm, int32_t *out, int nb0, int nb1) {
  const int kbs = m->in / 16, nbs = m->out / 16;
  // windows of positions 8g .. 8g + 7: ends (8g + j + 1) K; with K = 2 they span 30 bits; v = 32 bits ending at the
  // last window's end, so position j's window = v >> (2 (7 - j)), low 16 bits
  const __m256i sh = _mm256_setr_epi32(14, 12, 10, 8, 6, 4, 2, 0), mul = _mm256_set1_epi32((int)MUL);
  for (int nb = nb0; nb < nb1; nb++) {
    int32_t colsum[16] = {0};
    for (int pass = 0; pass < 4; pass++) {
      __m256i acc[8];
      for (int i = 0; i < 8; i++) acc[i] = _mm256_setzero_si256();
      for (int kb = 0; kb < kbs; kb++) {
        const uint32_t *w = m->w + ((size_t)kb * nbs + nb) * WORDS;
        const int32_t *xp = xperm + kb * 256;
        for (int gi = 0; gi < 8; gi++) {
          const int g = pass * 8 + gi;
          const int e = (8 * g + 8) * K + 256 * K;  // end bit of the group's last window (+ one tile for the wrap)
          const int i1 = (e - 1) / 32, s = (i1 + 1) * 32 - e;
          const uint64_t m64 = ((uint64_t)w[(i1 - 1) % WORDS] << 32) | w[i1 % WORDS];
          const __m256i st = _mm256_srlv_epi32(_mm256_set1_epi32((int)(uint32_t)(m64 >> s)), sh);
          const __m256i pr = _mm256_mullo_epi32(st, mul);
          acc[gi] = _mm256_dpbusd_avx_epi32(acc[gi], pr, _mm256_loadu_si256((const __m256i *)(xp + 8 * g)));
        }
      }
      for (int gi = 0; gi < 8; gi++) {
        int32_t a[8];
        _mm256_storeu_si256((__m256i *)a, acc[gi]);
        for (int j = 0; j < 8; j++) colsum[pos_c[8 * (pass * 8 + gi) + j]] += a[j];
      }
    }
    memcpy(out + 16 * nb, colsum, sizeof colsum);
  }
}


// v2 (K = 2), tiles stored column-major (tile (kb, nb) at nb * kbs + kb: dsx would repack exllamav3's row-major tile
// order at load, bytes unchanged). Position t = 4m + r (r = 0..3) has its window end at bit 8m + 2r + 2, so its state is
// the big-endian 24-bit value of stream bytes m-2, m-1, m shifted right by 6 - 2r. Per 8 consecutive m: one vpshufb
// builds the 8 BE24 values (one per dword), then for each r one shift, one vpmullw (only the low 16 bits of the product
// matter; the upper half of each dword is garbage that meets zero activation bytes) and one vpdpbusd.
// xperm2 per k block: [m0 / 8][r][i] int32, position 4 (m0 + i) + r.
static int32_t *xperm2_from(const int32_t *xperm, int in) {  // reorder permute_x's [kb][t] layout
  int32_t *o = aligned_alloc(64, (size_t)in / 16 * 256 * 4);
  for (int kb = 0; kb < in / 16; kb++)
    for (int g = 0; g < 8; g++)
      for (int r = 0; r < 4; r++)
        for (int i = 0; i < 8; i++) o[kb * 256 + g * 32 + r * 8 + i] = xperm[kb * 256 + 4 * (8 * g + i) + r];
  return o;
}

static void gemv_tiles2(const Mat *m, const int32_t *xp2, int32_t *out, int nb0, int nb1) {
  const int kbs = m->in / 16;
  const __m256i bswap = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
  // dword i of each 128-bit lane: bytes (m, m-1, m-2, 0) for m = base + i, base byte at lane offset i
  const __m256i be24 = _mm256_setr_epi8(2, 1, 0, -1, 3, 2, 1, -1, 4, 3, 2, -1, 5, 4, 3, -1, 2, 1, 0, -1, 3, 2, 1, -1, 4, 3, 2, -1, 5, 4, 3, -1);
  const __m256i mul = _mm256_set1_epi16((short)MUL);
  for (int nb = nb0; nb < nb1; nb++) {
    int32_t colsum[16] = {0};
    for (int pass = 0; pass < 4; pass++) {  // m-ranges 2 pass, 2 pass + 1: 8 accumulators
      __m256i acc[8];
      for (int i = 0; i < 8; i++) acc[i] = _mm256_setzero_si256();
      const uint32_t *wt = m->w + (size_t)nb * kbs * WORDS;
      for (int kb = 0; kb < kbs; kb++, wt += WORDS) {
        // big-endian stream with the wrap: buf[2 + i] = stream byte i, buf[0..1] = stream bytes 62, 63
        uint8_t buf[96] __attribute__((aligned(32)));
        const __m256i a = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)wt), bswap);
        const __m256i b = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(wt + 8)), bswap);
        _mm256_storeu_si256((__m256i *)(buf + 2), a);
        _mm256_storeu_si256((__m256i *)(buf + 34), b);
        buf[0] = buf[64], buf[1] = buf[65];
        const int32_t *xp = xp2 + kb * 256;
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++) {
          const int g = 2 * pass + h, m0 = 8 * g;
          // lower lane: m0..m0+3 from buf + m0 (stream bytes m0-2..), upper: m0+4..m0+7 from buf + m0 + 4
          const __m256i src = _mm256_loadu2_m128i((const __m128i *)(buf + m0 + 4), (const __m128i *)(buf + m0));
          const __m256i v = _mm256_shuffle_epi8(src, be24);
          const int32_t *xg = xp + g * 32;
          acc[4 * h + 0] = _mm256_dpbusd_avx_epi32(acc[4 * h + 0], _mm256_mullo_epi16(_mm256_srli_epi32(v, 6), mul), _mm256_loadu_si256((const __m256i *)(xg)));
          acc[4 * h + 1] = _mm256_dpbusd_avx_epi32(acc[4 * h + 1], _mm256_mullo_epi16(_mm256_srli_epi32(v, 4), mul), _mm256_loadu_si256((const __m256i *)(xg + 8)));
          acc[4 * h + 2] = _mm256_dpbusd_avx_epi32(acc[4 * h + 2], _mm256_mullo_epi16(_mm256_srli_epi32(v, 2), mul), _mm256_loadu_si256((const __m256i *)(xg + 16)));
          acc[4 * h + 3] = _mm256_dpbusd_avx_epi32(acc[4 * h + 3], _mm256_mullo_epi16(v, mul), _mm256_loadu_si256((const __m256i *)(xg + 24)));
        }
      }
      for (int h = 0; h < 2; h++)
        for (int r = 0; r < 4; r++) {
          int32_t a[8];
          _mm256_storeu_si256((__m256i *)a, acc[4 * h + r]);
          for (int i = 0; i < 8; i++) colsum[pos_c[4 * (8 * (2 * pass + h) + i) + r]] += a[i];
        }
    }
    memcpy(out + 16 * nb, colsum, sizeof colsum);
  }
}

// v3 (K = 2): v2 without the staging buffer. Tiles are repacked at load (dsx would do this once): column-major tile
// order and each u32 byte-swapped, so a tile is its big-endian bit stream, 64 bytes, same size. The shuffle sources load
// straight from it; only m-range 0 (whose windows wrap to bytes 62, 63) takes one palignr. The matrix needs 16 bytes of
// padding after its last tile (the last upper-lane load reads past the tile; those bytes are unused).
static void gemv_tiles3(const Mat *m, const int32_t *xp2, int32_t *out, int nb0, int nb1) {
  const int kbs = m->in / 16;
  const __m256i be24 = _mm256_setr_epi8(2, 1, 0, -1, 3, 2, 1, -1, 4, 3, 2, -1, 5, 4, 3, -1, 2, 1, 0, -1, 3, 2, 1, -1, 4, 3, 2, -1, 5, 4, 3, -1);
  const __m256i mul = _mm256_set1_epi16((short)MUL);
  for (int nb = nb0; nb < nb1; nb++) {
    int32_t colsum[16] = {0};
    for (int pass = 0; pass < 4; pass++) {
      __m256i acc[8];
      for (int i = 0; i < 8; i++) acc[i] = _mm256_setzero_si256();
      const uint8_t *tb = (const uint8_t *)(m->w + (size_t)nb * kbs * WORDS);
      for (int kb = 0; kb < kbs; kb++, tb += 64) {
        const int32_t *xp = xp2 + kb * 256;
#pragma GCC unroll 2
        for (int h = 0; h < 2; h++) {
          const int g = 2 * pass + h, m0 = 8 * g;
          __m128i lo;
          if (g == 0) lo = _mm_alignr_epi8(_mm_loadu_si128((const __m128i *)tb), _mm_loadu_si128((const __m128i *)(tb + 48)), 14);
          else lo = _mm_loadu_si128((const __m128i *)(tb + m0 - 2));
          const __m256i src = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), _mm_loadu_si128((const __m128i *)(tb + m0 + 2)), 1);
          const __m256i v = _mm256_shuffle_epi8(src, be24);
          const int32_t *xg = xp + g * 32;
          acc[4 * h + 0] = _mm256_dpbusd_avx_epi32(acc[4 * h + 0], _mm256_mullo_epi16(_mm256_srli_epi32(v, 6), mul), _mm256_loadu_si256((const __m256i *)(xg)));
          acc[4 * h + 1] = _mm256_dpbusd_avx_epi32(acc[4 * h + 1], _mm256_mullo_epi16(_mm256_srli_epi32(v, 4), mul), _mm256_loadu_si256((const __m256i *)(xg + 8)));
          acc[4 * h + 2] = _mm256_dpbusd_avx_epi32(acc[4 * h + 2], _mm256_mullo_epi16(_mm256_srli_epi32(v, 2), mul), _mm256_loadu_si256((const __m256i *)(xg + 16)));
          acc[4 * h + 3] = _mm256_dpbusd_avx_epi32(acc[4 * h + 3], _mm256_mullo_epi16(v, mul), _mm256_loadu_si256((const __m256i *)(xg + 24)));
        }
      }
      for (int h = 0; h < 2; h++)
        for (int r = 0; r < 4; r++) {
          int32_t a[8];
          _mm256_storeu_si256((__m256i *)a, acc[4 * h + r]);
          for (int i = 0; i < 8; i++) colsum[pos_c[4 * (8 * (2 * pass + h) + i) + r]] += a[i];
        }
    }
    memcpy(out + 16 * nb, colsum, sizeof colsum);
  }
}

int main(int argc, char **argv) {
  const int ne = argc > 1 ? atoi(argv[1]) : 48, calls = argc > 2 ? atoi(argv[2]) : 200, nt = argc > 3 ? atoi(argv[3]) : 14;
  build_map();
  const int ver = getenv("T16V") ? atoi(getenv("T16V")) : 1, v2 = ver >= 2;
  // sanity: the map is a permutation of the 16 x 16 tile
  int seen[256] = {0};
  for (int t = 0; t < 256; t++) seen[pos_r[t] * 16 + pos_c[t]]++;
  for (int i = 0; i < 256; i++) if (seen[i] != 1) { printf("bad map\n"); return 1; }
  // correctness: the SIMD kernel against the scalar exllamav3-indexed decode on one small matrix
  {
    Mat t = mk(256, 64);
    int8_t xq[256];
    for (int i = 0; i < 256; i++) xq[i] = (int8_t)((i * 37) % 255 - 127);
    int32_t *xp = aligned_alloc(64, 16 * 256 * 4), got[64];
    permute_x(xq, 256, xp);
    gemv_tiles(&t, xp, got, 0, 4);
    int bad = 0, sx = 0;
    for (int i = 0; i < 256; i++) sx += xq[i];
    for (int n = 0; n < 64; n++) {
      long want = 0;
      for (int kb = 0; kb < 16; kb++)
        for (int pos = 0; pos < 256; pos++)
          if (pos_c[pos] == n % 16) want += (long)xq[16 * kb + pos_r[pos]] * tval(state_ref(t.w + ((size_t)kb * 4 + n / 16) * WORDS, pos));
      bad += (got[n] - 255L * sx) != want;
    }
    printf("check v1: %d of 64 outputs differ from the scalar exllamav3-indexed decode\n", bad);
    {
      Mat tc = t;  // same tiles, column-major
      tc.w = aligned_alloc(64, 16 * 4 * WORDS * 4);
      for (int kb = 0; kb < 16; kb++)
        for (int nb = 0; nb < 4; nb++) memcpy(tc.w + ((size_t)nb * 16 + kb) * WORDS, t.w + ((size_t)kb * 4 + nb) * WORDS, WORDS * 4);
      int32_t *x2 = xperm2_from(xp, 256), got2[64];
      gemv_tiles2(&tc, x2, got2, 0, 4);
      int bad2 = 0;
      for (int n = 0; n < 64; n++) bad2 += got2[n] != got[n];
      printf("check v2: %d of 64 outputs differ from v1%s\n", bad2, bad ? "" : " (v1 matches the scalar decode)");
      Mat t3 = t;
      t3.w = aligned_alloc(64, 16 * 4 * WORDS * 4 + 64);
      for (int i = 0; i < 16 * 4 * WORDS; i++) t3.w[i] = __builtin_bswap32(tc.w[i]);
      int32_t got3[64];
      gemv_tiles3(&t3, x2, got3, 0, 4);
      int bad3 = 0;
      for (int n = 0; n < 64; n++) bad3 += got3[n] != got[n];
      printf("check v3: %d of 64 outputs differ from v1\n", bad3);
      free(t3.w);
      free(tc.w);
      free(x2);
    }
    free(t.w);
    free(xp);
  }
  // timing: a pool of experts larger than the caches, calls of 6 experts (gate + up [5120 -> 2304], down [2304 -> 5120])
  Mat *gate = malloc(ne * sizeof(Mat)), *up = malloc(ne * sizeof(Mat)), *down = malloc(ne * sizeof(Mat));
  for (int e = 0; e < ne; e++) gate[e] = mk(HID, INT), up[e] = mk(HID, INT), down[e] = mk(INT, HID);
  const double bytes6 = 6.0 * (2.0 * HID * INT + INT * HID) * K / 8;
  printf("%d experts in the pool (%.1f GB), %d calls of 6, %d threads, K = %d\n", ne, ne * bytes6 / 6 / 1e9, calls, nt, K);
  float x[HID], xh[HID], h[6][INT], hd[INT], y[HID];
  for (int i = 0; i < HID; i++) x[i] = sinf(i * 0.37f);
  int8_t xq[HID], hq[6][INT];
  int32_t *xpe = aligned_alloc(64, HID / 16 * 256 * 4), *hpe[6];
  for (int e = 0; e < 6; e++) hpe[e] = aligned_alloc(64, INT / 16 * 256 * 4);
  static int32_t og[6][INT], ou[6][INT], od[6][HID];
  omp_set_num_threads(nt);
  double t0 = 0, tm[4] = {0};
  for (int c = -5; c < calls; c++) {
    if (c == 0) t0 = now();
    const double ta = now();
    // input side: x * suh (here 1), had128, int8, permute
    memcpy(xh, x, sizeof x);
    had128(xh, HID);
    float mx = 0;
    for (int i = 0; i < HID; i++) mx = fmaxf(mx, fabsf(xh[i]));
    for (int i = 0; i < HID; i++) xq[i] = (int8_t)lrintf(xh[i] * 127.f / mx);
    permute_x(xq, HID, xpe);
    int32_t *xpe2 = v2 ? xperm2_from(xpe, HID) : 0;
    const double tb = now();
    const int e0 = ((c + 5) * 6) % (ne - 5);  // c starts at -5 (warm-up)
    // gate + up: work items (expert, matrix, 16-column block range)
#pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < 6 * 2 * 18; it++) {
      const int e = it / 36, mtx = (it / 18) % 2, part = it % 18;
      if (v2) (ver == 3 ? gemv_tiles3 : gemv_tiles2)(mtx ? &up[e0 + e] : &gate[e0 + e], xpe2, mtx ? ou[e] : og[e], part * 8, part * 8 + 8);
      else gemv_tiles(mtx ? &up[e0 + e] : &gate[e0 + e], xpe, mtx ? ou[e] : og[e], part * 8, part * 8 + 8);
    }
    const double tc = now();
    for (int e = 0; e < 6; e++) {
      for (int i = 0; i < INT; i++) h[e][i] = (float)og[e][i] * (float)ou[e][i] * 1e-9f;
      had128(h[e], INT);
      float m2 = 0;
      for (int i = 0; i < INT; i++) m2 = fmaxf(m2, fabsf(h[e][i]));
      for (int i = 0; i < INT; i++) hq[e][i] = (int8_t)lrintf(h[e][i] * 127.f / (m2 + 1e-30f));
      permute_x(hq[e], INT, hpe[e]);
      if (v2) { int32_t *t2 = xperm2_from(hpe[e], INT); memcpy(hpe[e], t2, (size_t)INT / 16 * 256 * 4); free(t2); }
    }
    const double td = now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int it = 0; it < 6 * 40; it++) {
      const int e = it / 40, part = it % 40;
      if (v2) (ver == 3 ? gemv_tiles3 : gemv_tiles2)(&down[e0 + e], hpe[e], od[e], part * 8, part * 8 + 8);
      else gemv_tiles(&down[e0 + e], hpe[e], od[e], part * 8, part * 8 + 8);
    }
    for (int i = 0; i < HID; i++) y[i] = (float)(od[0][i] + od[1][i] + od[2][i] + od[3][i] + od[4][i] + od[5][i]);
    had128(y, HID);
    const double te = now();
    free(xpe2);
    if (c >= 0) tm[0] += tb - ta, tm[1] += tc - tb, tm[2] += td - tc, tm[3] += te - td;
  }
  const double dt = (now() - t0) / calls;
  printf("%.3f ms per 6-expert call (%.1f GB/s of weights): input had+quant+permute %.3f, gate+up %.3f, mid (swiglu stand-in, had, quant, permute) %.3f, down+out had %.3f  [%g]\n",
         dt * 1e3, bytes6 / dt / 1e9, tm[0] / calls * 1e3, tm[1] / calls * 1e3, tm[2] / calls * 1e3, tm[3] / calls * 1e3, (double)y[7] * 0);
  return 0;
}
