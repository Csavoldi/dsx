// verify.h: decode-style matrix-vector products for 2..16 tokens at once (speculative verify, draft passes).
// Weights stream exactly as in decode (one wave per 16-row tile and k-slice); MFMA column c = token c, whose FP8
// activation row and scales the lane reads itself, so every per-k-block scale is per lane (token) and per row.
#pragma once
#include "gemv.h"

struct VJob {
    const void* w;      // tiles
    const void* sc;     // FP8: E8M0 [N/32][K/32]; experts: arena base (w) + offsets below
    long long woff, soff;
    float mul;
    int fmt;            // 0 FP8 dense, 1 MXFP4, 2 CB3, 3 CB2, 4 BF16
    int n, k, split;
    const uint8_t* xq;  // FP8 codes [row][ldx]
    const float* xs;    // scales [row][ldx/32]
    const uint16_t* xb; // bf16 rows [row][ldx] (fmt 4)
    int ldx;
    int ntok;           // tokens (<= 16)
    const int* rows;    // activation row per token (null: token index)
    float* y;           // output [orow][ldy]
    const int* orows;   // output row per token (null: token index)
    int ldy, atomic;    // atomic: += (split > 1 or scatter-add)
    int blocks, epi;    // epi 1/2: RoPE (q: 512-dim heads, idx: 128) at position pos0 + token
    int ropec, ndyn;    // ndyn: dyn points at ndyn consecutive device-built jobs (40 ints each)
    const int* dyn;     // device-built expert job (null: static): [fmt, ntok, woff_lo, woff_hi, soff_lo, soff_hi, mul_bits, ...]
    const uint16_t* xl; // fmt 4: optional bf16 low parts (x = xb + xl, as decode's head)
};

struct VJobs {
    VJob j[12];
    int n;
    uint32_t cb3lo, cb3hi, cb2;
};

// One wave: one 16-row tile over k-units [u0, u1) for up to 16 tokens.
template <int KIND>  // 0: any format, 1: dense only (FP8, BF16), 2: experts only
DEV f4 vwave(const VJob& J, int rt, int u0, int u1, int lane, uint32_t cb3lo, uint32_t cb3hi, uint32_t cb2) {
    const int col = lane & 15, g8 = 8 * (lane >> 4), g4 = 4 * (lane >> 4);
    const bool live = col < J.ntok;
    const int xrow = live ? (J.rows ? J.rows[col] : col) : 0;
    const uint8_t* xq = J.xq + (size_t)xrow * J.ldx;
    const float* xs = J.xs + (size_t)xrow * (J.ldx / 32);
    const float lm = live ? 1.f : 0.f;
    f4 acc = {0, 0, 0, 0};
    if (KIND != 2 && J.fmt == 0) {
        const int KP = J.k / 64, KB = J.k / 32;
        const u4v* wp = (const u4v*)J.w + (size_t)rt * KP * 64 + lane;
        const uint8_t* sp = (const uint8_t*)J.sc + (size_t)(rt >> 1) * KB;
        constexpr int U = 4;
        u4v cw[U], nw[U];
        uint64_t cx[U][2], nx[U][2];
        float cs[U][2], ns[U][2];
        auto ld = [&](int kp, u4v& w, uint64_t* x, float* s) {
            w = __builtin_nontemporal_load(wp + (size_t)kp * 64);
            x[0] = *reinterpret_cast<const uint64_t*>(xq + kp * 64 + g8);
            x[1] = *reinterpret_cast<const uint64_t*>(xq + kp * 64 + 32 + g8);
            s[0] = xs[2 * kp] * __uint_as_float((uint32_t)sp[2 * kp] << 23) * lm;
            s[1] = xs[2 * kp + 1] * __uint_as_float((uint32_t)sp[2 * kp + 1] << 23) * lm;
        };
        for (int u = 0; u < U; u++)
            if (u0 + u < u1) ld(u0 + u, cw[u], cx[u], cs[u]);
        for (int k = u0; k < u1; k += U) {
            for (int u = 0; u < U; u++)
                if (k + U + u < u1) ld(k + U + u, nw[u], nx[u], ns[u]);
#pragma unroll
            for (int u = 0; u < U; u++) {
                if (k + u < u1) {
                    const f4 d0 = mfma_fp8((uint64_t)cw[u].y << 32 | cw[u].x, cx[u][0], f4{0, 0, 0, 0});
                    const f4 d1 = mfma_fp8((uint64_t)cw[u].w << 32 | cw[u].z, cx[u][1], f4{0, 0, 0, 0});
                    acc += d0 * cs[u][0] + d1 * cs[u][1];
                }
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                cw[u] = nw[u];
                cx[u][0] = nx[u][0]; cx[u][1] = nx[u][1];
                cs[u][0] = ns[u][0]; cs[u][1] = ns[u][1];
            }
        }
        return acc * 2.f;
    }
    if (KIND != 2 && J.fmt == 4) {
        const int KQ = J.k / 32;
        const u4v* wp = (const u4v*)J.w + (size_t)rt * KQ * 64 + lane;
        const uint16_t* xb = J.xb + (size_t)xrow * J.ldx;
        for (int kq = u0; kq < u1; kq++) {
            const u4v w = __builtin_nontemporal_load(wp + (size_t)kq * 64);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint32_t w0 = h ? w.z : w.x, w1 = h ? w.w : w.y;
                s4v a;
                a[0] = (short)(w0 & 0xFFFF); a[1] = (short)(w0 >> 16); a[2] = (short)(w1 & 0xFFFF); a[3] = (short)(w1 >> 16);
                uint64_t bv = live ? *reinterpret_cast<const uint64_t*>(xb + kq * 32 + 16 * h + g4) : 0;
                acc = mfma_bf16(a, *reinterpret_cast<const s4v*>(&bv), acc);
                if (J.xl) {
                    uint64_t lv = live ? *reinterpret_cast<const uint64_t*>(J.xl + (size_t)xrow * J.ldx + kq * 32 + 16 * h + g4) : 0;
                    acc = mfma_bf16(a, *reinterpret_cast<const s4v*>(&lv), acc);
                }
            }
        }
        return acc;
    }
    if (KIND == 1) return acc;
    // experts: per-row scales (lane rows 4*(lane/16)+j), per-lane token scales. Weights, scales and the token's
    // activation fragments are loaded U units ahead (latency, not bandwidth, bounds a 16-row tile).
    const char* base = (const char*)J.w;
    const int KU = J.k / 128;
    const int NDW = J.fmt == 1 ? 4 : (J.fmt == 2 ? 3 : 2);
    const u4v* sp = (const u4v*)(base + J.soff) + (size_t)rt * KU * 4 + (lane >> 4);
    const uint32_t* wp = (const uint32_t*)(base + J.woff) + (size_t)rt * KU * 64 * NDW + lane * NDW;
    constexpr int U = 2;
    struct Unit { uint32_t w[4]; u4v s; uint64_t x[4]; float xs[4]; };
    Unit cu[U], nu[U];
    auto ld = [&](int u, Unit& o) {
#pragma unroll
        for (int q = 0; q < 4; q++) o.w[q] = q < NDW ? __builtin_nontemporal_load(wp + (size_t)u * 64 * NDW + q) : 0u;
        o.s = sp[(size_t)u * 4];
#pragma unroll
        for (int m = 0; m < 4; m++) {
            o.x[m] = live ? *reinterpret_cast<const uint64_t*>(xq + (4 * u + m) * 32 + g8) : 0ull;
            o.xs[m] = xs[4 * u + m] * lm;
        }
    };
#pragma unroll
    for (int i = 0; i < U; i++)
        if (u0 + i < u1) ld(u0 + i, cu[i]);
    for (int u = u0; u < u1; u += U) {
#pragma unroll
        for (int i = 0; i < U; i++)
            if (u + U + i < u1) ld(u + U + i, nu[i]);
#pragma unroll
        for (int i = 0; i < U; i++) {
            if (u + i >= u1) break;
            const Unit& c = cu[i];
            const uint32_t ss[4] = {c.s.x, c.s.y, c.s.z, c.s.w};
#pragma unroll
            for (int m = 0; m < 4; m++) {
                uint64_t a;
                if (J.fmt == 1) a = fp4x8_to_fp8(c.w[m]);
                else {
                    const uint32_t tlo = J.fmt == 2 ? cb3lo : cb2, thi = J.fmt == 2 ? cb3hi : 0u;
                    const uint32_t dl = c.w[m >> 1];
                    const int sh = 4 * (m & 1);
                    uint32_t s0 = (dl >> sh) & 0x03030303u, s1 = (dl >> (sh + 2)) & 0x03030303u;
                    if (J.fmt == 2) {
                        s0 |= ((c.w[2] >> (2 * m)) & 0x01010101u) << 2;
                        s1 |= ((c.w[2] >> (2 * m + 1)) & 0x01010101u) << 2;
                    }
                    a = (uint64_t)perm(thi, tlo, s1) << 32 | perm(thi, tlo, s0);
                }
                const f4 d = mfma_fp8(a, c.x[m], f4{0, 0, 0, 0});
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t bb = (ss[j] >> (8 * m)) & 0xFFu;
                    const float ws = J.fmt == 1 ? __uint_as_float(bb << 23) : fp8_to_f32(bb, 0);
                    acc[j] += d[j] * (ws * c.xs[m]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < U; i++) cu[i] = nu[i];
    }
    return acc * J.mul;
}

// Blocks enumerate (job, row-block of 4 tiles, k-slice). Dynamic expert jobs read fmt/offsets/tokens from `dyn`.
template <int KIND>
DEV void vgemv_impl(VJob J, int b, uint32_t cb3lo, uint32_t cb3hi, uint32_t cb2, const int* pos0p, const float* rope_f, const float* rope_fc) {
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    if (J.dyn) {
        // device-built expert job: [fmt, ntok, woff lo/hi, soff lo/hi, mul bits, rows[16], orows[16]]
        const int per = J.blocks / J.ndyn;
        const int* d = J.dyn + (size_t)(b / per) * 40;
        b %= per;
        J.fmt = d[0];
        J.ntok = d[1];
        if (J.ntok == 0) return;
        J.woff = (long long)(uint32_t)d[2] | (long long)d[3] << 32;
        J.soff = (long long)(uint32_t)d[4] | (long long)d[5] << 32;
        J.mul = __int_as_float(d[6]);
        J.rows = d + 8;
        J.orows = d + 24;
    }
    const int RB = (J.n + 63) / 64;
    const int rb = b % RB, sl = b / RB;
    const int rt = rb * 4 + wv;
    if (rt * 16 >= J.n) return;
    const int KU = J.fmt == 0 ? 64 : (J.fmt == 4 ? 32 : 128);
    const int KUs = J.k / KU;
    const int u0 = KUs * sl / J.split, u1 = KUs * (sl + 1) / J.split;
    f4 acc = vwave<KIND>(J, rt, u0, u1, lane, cb3lo, cb3hi, cb2);
    const int col = lane & 15;
    if (col >= J.ntok) return;
    const int r0 = rt * 16 + 4 * (lane >> 4);
    float v[4] = {acc[0], acc[1], acc[2], acc[3]};
    if (J.epi == 1 || J.epi == 2) {
        const int hd = J.epi == 1 ? HD : ID;
        const int dim = r0 % hd;
        if (dim >= hd - RD) {
            const float* f = J.ropec ? rope_fc : rope_f;
            const int pos = *pos0p + col;
            for (int p = 0; p < 2; p++) rope_pair(v[2 * p], v[2 * p + 1], (float)pos * f[(dim - (hd - RD)) / 2 + p]);
        }
    }
    const int orow = J.orows ? J.orows[col] : col;
    float* yp = J.y + (size_t)orow * J.ldy + r0;
    for (int j = 0; j < 4; j++) {
        if (J.atomic) atomicAdd(yp + j, v[j]);
        else yp[j] = v[j];
    }
}

extern "C" __global__ void __launch_bounds__(256) vgemv(VJobs js, const int* pos0p, const float* rope_f, const float* rope_fc) {
    int b = blockIdx.x, ji = 0;
    while (ji < js.n - 1 && b >= js.j[ji].blocks) b -= js.j[ji++].blocks;
    vgemv_impl<0>(js.j[ji], b, js.cb3lo, js.cb3hi, js.cb2, pos0p, rope_f, rope_fc);
}
extern "C" __global__ void __launch_bounds__(256) vgemv_d(VJobs js, const int* pos0p, const float* rope_f, const float* rope_fc) {
    int b = blockIdx.x, ji = 0;
    while (ji < js.n - 1 && b >= js.j[ji].blocks) b -= js.j[ji++].blocks;
    vgemv_impl<1>(js.j[ji], b, js.cb3lo, js.cb3hi, js.cb2, pos0p, rope_f, rope_fc);
}
extern "C" __global__ void __launch_bounds__(256) vgemv_x(VJobs js, const int* pos0p, const float* rope_f, const float* rope_fc) {
    int b = blockIdx.x, ji = 0;
    while (ji < js.n - 1 && b >= js.j[ji].blocks) b -= js.j[ji++].blocks;
    vgemv_impl<2>(js.j[ji], b, js.cb3lo, js.cb3hi, js.cb2, pos0p, rope_f, rope_fc);
}

// Expert grouping for a verify batch (one block): picks sel[t][6] -> one job per distinct expert with its tokens.
// dyn layout per job: [fmt, ntok, woff lo, hi, soff lo, hi, mul bits, pad, rows[16] (activation rows = token),
// orows[16] (output rows = compact pair index t*6+slot)], for gate/up mats 0/1 and down mat 2 (3 job arrays of 36).
// Also writes the pair weights wrow[t*6+slot].
extern "C" __global__ void vgroup(int T, int K, const int* sel, const float* wts, const ExpertMat* tab, int* dyn, float* wrow, int* npairs) {
    __shared__ int ex[96], cnt[96], nex;
    __shared__ int members[96][16];
    __shared__ int ss[96], lead[96], gidx[96];
    const int N = T * K, i = threadIdx.x;
    if (i < N) {
        ss[i] = sel[i];
        wrow[i] = wts[i];
    }
    if (i == 0) *npairs = N;
    __syncthreads();
    if (i < N) {
        int L = 0;
        while (ss[L] != ss[i]) L++;
        lead[i] = L;
    }
    __syncthreads();
    if (i < N && lead[i] == i) {
        int g = 0, c = 0;
        for (int j = 0; j < i; j++) g += lead[j] == j;
        for (int j = i; j < N; j++) c += lead[j] == i;
        gidx[i] = g;
        ex[g] = ss[i];
        cnt[g] = c < 16 ? c : 16;
    }
    if (i == 0) {
        int n = 0;
        for (int j = 0; j < N; j++) n += lead[j] == j;
        nex = n;
    }
    __syncthreads();
    if (i < N) {
        const int L = lead[i];
        int r = 0;
        for (int j = L; j < i; j++) r += lead[j] == L;
        if (r < 16) members[gidx[L]][r] = i;
    }
    __syncthreads();
    for (int q = threadIdx.x; q < 3 * 48; q += blockDim.x) {
        const int m = q / 48, j = q % 48;
        int* d = dyn + (size_t)(m * 48 + j) * 40;
        if (j >= nex) { d[1] = 0; continue; }
        const ExpertMat em = tab[ex[j] * 3 + m];
        d[0] = em.fmt == E_FP4 ? 1 : (em.fmt == E_CB3 ? 2 : 3);
        d[1] = cnt[j];
        d[2] = (int)(em.woff & 0xFFFFFFFF); d[3] = (int)(em.woff >> 32);
        d[4] = (int)(em.soff & 0xFFFFFFFF); d[5] = (int)(em.soff >> 32);
        d[6] = __float_as_int(em.mul);
        for (int i = 0; i < 16; i++) {
            const int pr = i < cnt[j] ? members[j][i] : 0;
            // gate/up: activation row = token, output row = pair; down: activation row = pair, output row = token
            d[8 + i] = m < 2 ? pr / K : pr;
            d[24 + i] = m < 2 ? pr : pr / K;
        }
    }
}
