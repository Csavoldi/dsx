// gemv.h: weight-streaming matrix-vector products on MFMA, for 1 to 16 tokens at once (decode and verify).
//
// A block is 4 waves = 4 consecutive 16-row tiles sharing one k-slice. The caller stages the slice's activations in
// LDS (FP8 codes + one scale per 32, or bf16 hi/lo pairs), then each wave streams its tile's weights:
//   FP8   tile 16 rows x 64 k, [rowtile][kpair][lane][16 B]; lane l: row l%16, k 8*(l/16)..+8 of each 32-k half.
//         Scales: the checkpoint's E8M0 per 32x32 block [N/32][K/32].
//   MXFP4 tile 16 rows x 128 k, [rowtile][k128][lane][16 B]; dword m = k-block 4*k128+m, 8 codes for row l%16,
//         k 8*(l/16)..+8, code j at nibble 2j (j<4) / 2(j-4)+1. Scales E8M0 [rowtile][k128][16 rows][4].
//   BF16  tile 16 rows x 32 k, [rowtile][k32][lane][16 B]; lane l: row l%16, k 4*(l/16)..+4 of each 16-k half.
// Activations: FP8 codes are FNUZ (OCP value / 2; scales carry the 2), weights likewise (gemv results x2 per side).
#pragma once
#include "common.h"

// LDS layout of a staged FP8 activation slice of S k-pairs for TT tokens
template <int TT> struct ActLds {
    static constexpr int pad = 8;
    DEV static int xr(int S) { return S * 64 + pad; }
    DEV static int bytes(int S) { return ((TT * xr(S) + 15) & ~15) + TT * S * 2 * 4; }
};

// One wave of an FP8 x FP8 tile: S k-pairs; xr: this lane's LDS codes (token row + 8*(lane/16)), xsr: token scales
// [2S], wsr: weight scales per k-block [2S] (f32, 2^(e-127)).
template <int U = 4>
DEV f4 wave_fp8(const u4v* __restrict__ wp, int S, const uint8_t* xr, const float* xsr, const float* wsr, bool live) {
    f4 acc = {0, 0, 0, 0};
    u4v cur[U], nxt[U];
#pragma unroll
    for (int u = 0; u < U; u++) cur[u] = u < S ? __builtin_nontemporal_load(wp + (size_t)u * 64) : u4v{0, 0, 0, 0};
    for (int k = 0; k < S; k += U) {
#pragma unroll
        for (int u = 0; u < U; u++) nxt[u] = (k + U + u < S) ? __builtin_nontemporal_load(wp + (size_t)(k + U + u) * 64) : u4v{0, 0, 0, 0};
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int kk = k + u;
            if (kk < S) {
                const uint64_t b0 = live ? *reinterpret_cast<const uint64_t*>(xr + kk * 64) : 0;
                const uint64_t b1 = live ? *reinterpret_cast<const uint64_t*>(xr + kk * 64 + 32) : 0;
                f4 d0 = mfma_fp8((uint64_t)cur[u].y << 32 | cur[u].x, b0, f4{0, 0, 0, 0});
                f4 d1 = mfma_fp8((uint64_t)cur[u].w << 32 | cur[u].z, b1, f4{0, 0, 0, 0});
                acc += d0 * (wsr[2 * kk] * xsr[2 * kk]) + d1 * (wsr[2 * kk + 1] * xsr[2 * kk + 1]);
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) cur[u] = nxt[u];
    }
    return acc;
}

// e2m1 code nibbles -> FNUZ fp8 bytes (magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6; codes 8 (-0) are stored as 0)
DEV uint64_t fp4x8_to_fp8(uint32_t w) {
    const uint32_t lo = w & 0x0F0F0F0Fu, hi = (w >> 4) & 0x0F0F0F0Fu;
    const uint32_t a = perm(0x54504C48u, 0x44403800u, lo & 0x07070707u) | ((lo << 4) & 0x80808080u);
    const uint32_t b = perm(0x54504C48u, 0x44403800u, hi & 0x07070707u) | ((hi << 4) & 0x80808080u);
    return (uint64_t)b << 32 | a;
}

// One wave of an MXFP4 x FP8 tile over S k128-blocks; per-row scales: lane holds rows 4*(lane/16)+j.
// xr: lane's LDS codes (+8*(lane/16)), xsr: token scales per k-block [4S]. Returns acc[j] for rows 4*(lane/16)+j.
template <int U = 2>
DEV f4 wave_mxfp4(const u4v* __restrict__ wp, const u4v* __restrict__ sp, int S, const uint8_t* xr, const float* xsr, bool live) {
    f4 acc = {0, 0, 0, 0};
    u4v cw[U], cs[U], nw[U], ns[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
        cw[u] = u < S ? __builtin_nontemporal_load(wp + (size_t)u * 64) : u4v{0, 0, 0, 0};
        cs[u] = u < S ? sp[(size_t)u * 4] : u4v{0, 0, 0, 0};
    }
    for (int k = 0; k < S; k += U) {
#pragma unroll
        for (int u = 0; u < U; u++) {
            nw[u] = (k + U + u < S) ? __builtin_nontemporal_load(wp + (size_t)(k + U + u) * 64) : u4v{0, 0, 0, 0};
            ns[u] = (k + U + u < S) ? sp[(size_t)(k + U + u) * 4] : u4v{0, 0, 0, 0};
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int kk = k + u;
            if (kk < S) {
                const uint32_t ws[4] = {cw[u].x, cw[u].y, cw[u].z, cw[u].w};
                const uint32_t ss[4] = {cs[u].x, cs[u].y, cs[u].z, cs[u].w};  // ss[j]: row 4g+j, bytes = k-blocks 0..3
#pragma unroll
                for (int m = 0; m < 4; m++) {
                    const int kb = 4 * kk + m;
                    const uint64_t b = live ? *reinterpret_cast<const uint64_t*>(xr + kb * 32) : 0;
                    f4 d = mfma_fp8(fp4x8_to_fp8(ws[m]), b, f4{0, 0, 0, 0});
                    const float xsv = xsr[kb];
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[j] += d[j] * (__uint_as_float(((ss[j] >> (8 * m)) & 0xFFu) << 23) * xsv);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) cw[u] = nw[u], cs[u] = ns[u];
    }
    return acc;
}

// One wave of a codebook (3-bit: B=3, 2-bit: B=2) x FP8 tile over S k128-blocks. Per lane per k128: lo plane dwords
// D0 (k-blocks 0,1), D1 (2,3): byte b = [c(2p,b) c(2p,4+b) c(2p+1,b) c(2p+1,4+b)] 2 bits each; B=3 adds a hi-bit dword
// H: byte b bit (2m + half) = bit 2 of c(m, 4*half + b). Codes index a table of FNUZ fp8 values (tlo: entries 0-3,
// thi: 4-7). Scales: one e4m3 (FNUZ) byte per row per 32, laid out as for MXFP4.
template <int B, int U = 2>
DEV f4 wave_cb(const uint32_t* __restrict__ wp, const u4v* __restrict__ sp, int S, const uint8_t* xr, const float* xsr,
               uint32_t tlo, uint32_t thi, bool live) {
    constexpr int NDW = B;  // dwords per lane per k128 (2 lo + 1 hi for B=3)
    f4 acc = {0, 0, 0, 0};
    uint32_t cw[U][NDW], nw[U][NDW];
    u4v cs[U], ns[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
#pragma unroll
        for (int q = 0; q < NDW; q++) cw[u][q] = u < S ? __builtin_nontemporal_load(wp + (size_t)u * 64 * NDW + q) : 0u;
        cs[u] = u < S ? sp[(size_t)u * 4] : u4v{0, 0, 0, 0};
    }
    for (int k = 0; k < S; k += U) {
#pragma unroll
        for (int u = 0; u < U; u++) {
#pragma unroll
            for (int q = 0; q < NDW; q++)
                nw[u][q] = (k + U + u < S) ? __builtin_nontemporal_load(wp + (size_t)(k + U + u) * 64 * NDW + q) : 0u;
            ns[u] = (k + U + u < S) ? sp[(size_t)(k + U + u) * 4] : u4v{0, 0, 0, 0};
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int kk = k + u;
            if (kk < S) {
                const uint32_t ss[4] = {cs[u].x, cs[u].y, cs[u].z, cs[u].w};
#pragma unroll
                for (int m = 0; m < 4; m++) {
                    const uint32_t dl = cw[u][m >> 1];
                    const int sh = 4 * (m & 1);
                    uint32_t s0 = (dl >> sh) & 0x03030303u, s1 = (dl >> (sh + 2)) & 0x03030303u;
                    if (B == 3) {
                        const uint32_t hb = cw[u][2];
                        s0 |= ((hb >> (2 * m)) & 0x01010101u) << 2;
                        s1 |= ((hb >> (2 * m + 1)) & 0x01010101u) << 2;
                    }
                    const uint64_t a = (uint64_t)perm(thi, tlo, s1) << 32 | perm(thi, tlo, s0);
                    const int kb = 4 * kk + m;
                    const uint64_t b = live ? *reinterpret_cast<const uint64_t*>(xr + kb * 32) : 0;
                    f4 d = mfma_fp8(a, b, f4{0, 0, 0, 0});
                    const float xsv = xsr[kb];
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[j] += d[j] * (fp8_to_f32(ss[j], m) * xsv);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
#pragma unroll
            for (int q = 0; q < NDW; q++) cw[u][q] = nw[u][q];
            cs[u] = ns[u];
        }
    }
    return acc;
}

// One wave of a BF16 x BF16 tile over S k32-blocks: xh/xl: lane's LDS hi/lo bf16 (token row + 4*(lane/16)), as
// shorts; `lo` adds the low half (f32 activations split exactly into two bf16).
template <int U = 4>
DEV f4 wave_bf16(const u4v* __restrict__ wp, int S, const uint16_t* xh, const uint16_t* xl, bool lo, bool live) {
    f4 acc = {0, 0, 0, 0};
    u4v cur[U], nxt[U];
#pragma unroll
    for (int u = 0; u < U; u++) cur[u] = u < S ? __builtin_nontemporal_load(wp + (size_t)u * 64) : u4v{0, 0, 0, 0};
    for (int k = 0; k < S; k += U) {
#pragma unroll
        for (int u = 0; u < U; u++) nxt[u] = (k + U + u < S) ? __builtin_nontemporal_load(wp + (size_t)(k + U + u) * 64) : u4v{0, 0, 0, 0};
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int kk = k + u;
            if (kk < S) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    s4v a;
                    const uint32_t w0 = h ? cur[u].z : cur[u].x, w1 = h ? cur[u].w : cur[u].y;
                    a[0] = (short)(w0 & 0xFFFF); a[1] = (short)(w0 >> 16); a[2] = (short)(w1 & 0xFFFF); a[3] = (short)(w1 >> 16);
                    s4v b = {0, 0, 0, 0}, bl = {0, 0, 0, 0};
                    if (live) {
                        const uint64_t v = *reinterpret_cast<const uint64_t*>(xh + kk * 32 + 16 * h);
                        b = *reinterpret_cast<const s4v*>(&v);
                        if (lo) {
                            const uint64_t vl = *reinterpret_cast<const uint64_t*>(xl + kk * 32 + 16 * h);
                            bl = *reinterpret_cast<const s4v*>(&vl);
                        }
                    }
                    acc = mfma_bf16(a, b, acc);
                    if (lo) acc = mfma_bf16(a, bl, acc);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) cur[u] = nxt[u];
    }
    return acc;
}

// Quantize 4 values per thread to FP8 per 32 (DeepSeek's act_quant: scale = pow2_ceil(amax / 448), OCP e4m3 grid),
// 8 consecutive lanes = one block of 32. Writes 4 FNUZ codes and (lane%8 == 0) the block's scale (x2 for FNUZ).
DEV void quant32(const float v[4], uint32_t* q4, float* qs) {
    float a = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    a = group_max<8>(a);
    const float s = pow2_ceil(fmaxf(a, 1e-4f) / 448.f);
    const float r = 0.5f / s;
    *q4 = cvt_fp8x4(v[0] * r, v[1] * r, v[2] * r, v[3] * r);
    if ((threadIdx.x & 7) == 0) *qs = 2.f * s;
}
