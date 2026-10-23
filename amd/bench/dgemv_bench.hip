// dgemv_bench.hip: dense FP8 matrix x few-token products on the MI325X, variants for the runner's small-batch
// dense path (amd/RUNNER.md "Next items" 1). Weights in the runner's tile layout [rt][kpair][lane][16 B] with E8M0
// scales per 32x32 block, activations FP8 [tok][k] with f32 scales per 32, output f32 [tok][n].
//   A: wave per (16-row tile, k-slice), prefetch U k-pairs ahead, atomicAdd (the runner's vgemv_d structure)
//   B: wave per (tile, k-slice) loads its whole slice first, then computes, atomicAdd
//   C: as B, partial sums stored per slice, then a reduce kernel (no atomics)
// Build: hipcc --offload-arch=gfx942 -O3 -std=c++17 -I../kernels dgemv_bench.hip -o dgemv_bench
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Args {
    const u4v* w;        // tiles
    const uint8_t* sc;   // [n/32][k/32]
    const uint8_t* xq;   // [tok][k]
    const float* xs;     // [tok][k/32]
    float* y;            // [tok][n]
    float* part;         // C: [S][tok][n]
    int n, k, ntok, S;
};

// one k-pair (64 k) of a 16-row tile for this lane's token column: two FP8 MFMAs with block scales
DEV void kp_step(f4& acc, const u4v w, uint64_t x0, uint64_t x1, float s0, float s1) {
    const f4 d0 = mfma_fp8((uint64_t)w.y << 32 | w.x, x0, f4{0, 0, 0, 0});
    const f4 d1 = mfma_fp8((uint64_t)w.w << 32 | w.z, x1, f4{0, 0, 0, 0});
    acc += d0 * s0 + d1 * s1;
}

template <int U>
__global__ void __launch_bounds__(256) dg_a(Args a) {
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    const int RT = a.n / 16, KP = a.k / 64, KB = a.k / 32;
    const int b = blockIdx.x, rb = b % (RT / 4), sl = b / (RT / 4);
    const int rt = rb * 4 + wv;
    const int u0 = KP * sl / a.S, u1 = KP * (sl + 1) / a.S;
    const int col = lane & 15, g8 = 8 * (lane >> 4);
    const bool live = col < a.ntok;
    const uint8_t* xq = a.xq + (size_t)(live ? col : 0) * a.k;
    const float* xs = a.xs + (size_t)(live ? col : 0) * KB;
    const float lm = live ? 1.f : 0.f;
    const u4v* wp = a.w + (size_t)rt * KP * 64 + lane;
    const uint8_t* sp = a.sc + (size_t)(rt >> 1) * KB;
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
    f4 acc = {0, 0, 0, 0};
    for (int u = 0; u < U; u++)
        if (u0 + u < u1) ld(u0 + u, cw[u], cx[u], cs[u]);
    for (int kk = u0; kk < u1; kk += U) {
        for (int u = 0; u < U; u++)
            if (kk + U + u < u1) ld(kk + U + u, nw[u], nx[u], ns[u]);
#pragma unroll
        for (int u = 0; u < U; u++)
            if (kk + u < u1) kp_step(acc, cw[u], cx[u][0], cx[u][1], cs[u][0], cs[u][1]);
#pragma unroll
        for (int u = 0; u < U; u++) { cw[u] = nw[u]; cx[u][0] = nx[u][0]; cx[u][1] = nx[u][1]; cs[u][0] = ns[u][0]; cs[u][1] = ns[u][1]; }
    }
    if (!live) return;
    float* yp = a.y + (size_t)col * a.n + rt * 16 + 4 * (lane >> 4);
    for (int j = 0; j < 4; j++) atomicAdd(yp + j, 2.f * acc[j]);
}

// D: R adjacent 16-row tiles per wave share one set of activation loads (activation traffic / R), prefetch U ahead
template <int R, int U>
__global__ void __launch_bounds__(256) dg_d(Args a) {
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    const int RT = a.n / 16, KP = a.k / 64, KB = a.k / 32;
    const int RBN = RT / (4 * R);
    const int b = blockIdx.x, rb = b % RBN, sl = b / RBN;
    const int rt0 = (rb * 4 + wv) * R;
    const int u0 = KP * sl / a.S, u1 = KP * (sl + 1) / a.S;
    const int col = lane & 15, g8 = 8 * (lane >> 4);
    const bool live = col < a.ntok;
    const uint8_t* xq = a.xq + (size_t)(live ? col : 0) * a.k;
    const float* xs = a.xs + (size_t)(live ? col : 0) * KB;
    const float lm = live ? 1.f : 0.f;
    struct Unit { u4v w[R]; uint32_t sb[R]; uint64_t x[2]; float xs[2]; };
    auto ld = [&](int kp, Unit& o) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            o.w[r] = __builtin_nontemporal_load(a.w + ((size_t)(rt0 + r) * KP + kp) * 64 + lane);
            o.sb[r] = *reinterpret_cast<const uint16_t*>(a.sc + (size_t)((rt0 + r) >> 1) * KB + 2 * kp);
        }
        o.x[0] = *reinterpret_cast<const uint64_t*>(xq + kp * 64 + g8);
        o.x[1] = *reinterpret_cast<const uint64_t*>(xq + kp * 64 + 32 + g8);
        o.xs[0] = xs[2 * kp] * lm;
        o.xs[1] = xs[2 * kp + 1] * lm;
    };
    f4 acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = f4{0, 0, 0, 0};
    Unit cu[U], nu[U];
    for (int u = 0; u < U; u++)
        if (u0 + u < u1) ld(u0 + u, cu[u]);
    for (int kk = u0; kk < u1; kk += U) {
        for (int u = 0; u < U; u++)
            if (kk + U + u < u1) ld(kk + U + u, nu[u]);
#pragma unroll
        for (int u = 0; u < U; u++)
            if (kk + u < u1)
#pragma unroll
                for (int r = 0; r < R; r++) {
                    const float s0 = cu[u].xs[0] * __uint_as_float((cu[u].sb[r] & 0xFFu) << 23), s1 = cu[u].xs[1] * __uint_as_float((cu[u].sb[r] >> 8) << 23);
                    kp_step(acc[r], cu[u].w[r], cu[u].x[0], cu[u].x[1], s0, s1);
                }
#pragma unroll
        for (int u = 0; u < U; u++) cu[u] = nu[u];
    }
    if (!live) return;
#pragma unroll
    for (int r = 0; r < R; r++) {
        float* yp = a.y + (size_t)col * a.n + (rt0 + r) * 16 + 4 * (lane >> 4);
        for (int j = 0; j < 4; j++) atomicAdd(yp + j, 2.f * acc[r][j]);
    }
}

// B / C: the whole k-slice (L k-pairs, L = KP / S <= LMAX) loaded before any use
template <int LMAX, bool PART>
__global__ void __launch_bounds__(256) dg_b(Args a) {
    const int lane = threadIdx.x & 63, wv = threadIdx.x >> 6;
    const int RT = a.n / 16, KP = a.k / 64, KB = a.k / 32;
    const int b = blockIdx.x, rb = b % (RT / 4), sl = b / (RT / 4);
    const int rt = rb * 4 + wv;
    const int L = KP / a.S, u0 = L * sl;
    const int col = lane & 15, g8 = 8 * (lane >> 4);
    const bool live = col < a.ntok;
    const uint8_t* xq = a.xq + (size_t)(live ? col : 0) * a.k;
    const float* xs = a.xs + (size_t)(live ? col : 0) * KB;
    const float lm = live ? 1.f : 0.f;
    const u4v* wp = a.w + ((size_t)rt * KP + u0) * 64 + lane;
    const uint8_t* sp = a.sc + (size_t)(rt >> 1) * KB + 2 * u0;
    u4v w[LMAX];
    uint64_t x[LMAX][2];
#pragma unroll
    for (int u = 0; u < LMAX; u++)
        if (u < L) w[u] = __builtin_nontemporal_load(wp + (size_t)u * 64);
#pragma unroll
    for (int u = 0; u < LMAX; u++)
        if (u < L) {
            x[u][0] = *reinterpret_cast<const uint64_t*>(xq + (u0 + u) * 64 + g8);
            x[u][1] = *reinterpret_cast<const uint64_t*>(xq + (u0 + u) * 64 + 32 + g8);
        }
    f4 acc = {0, 0, 0, 0};
#pragma unroll
    for (int u = 0; u < LMAX; u++)
        if (u < L) {
            const float s0 = xs[2 * (u0 + u)] * __uint_as_float((uint32_t)sp[2 * u] << 23) * lm;
            const float s1 = xs[2 * (u0 + u) + 1] * __uint_as_float((uint32_t)sp[2 * u + 1] << 23) * lm;
            kp_step(acc, w[u], x[u][0], x[u][1], s0, s1);
        }
    if (!live) return;
    if (PART) {
        float* pp = a.part + ((size_t)sl * a.ntok + col) * a.n + rt * 16 + 4 * (lane >> 4);
        *reinterpret_cast<f4*>(pp) = 2.f * acc;
    } else {
        float* yp = a.y + (size_t)col * a.n + rt * 16 + 4 * (lane >> 4);
        for (int j = 0; j < 4; j++) atomicAdd(yp + j, 2.f * acc[j]);
    }
}
__global__ void dg_reduce(Args a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.ntok * a.n) return;
    float s = 0;
    for (int q = 0; q < a.S; q++) s += a.part[(size_t)q * a.ntok * a.n + i];
    a.y[i] = s;
}

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)

int main() {
    struct Shape { const char* name; int n, k; };
    const Shape shapes[] = {{"wo_b", 5120, 8192}, {"q_b", 32768, 1280}, {"sh_w1", 2304, 5120}, {"wq_a", 1280, 5120}};
    const int toks[] = {1, 8, 16};
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    for (const Shape& sh : shapes) {
        const int n = sh.n, k = sh.k;
        std::vector<uint8_t> hw((size_t)n * k), hs((size_t)(n / 32) * (k / 32)), hx((size_t)16 * k);
        std::vector<float> hxs((size_t)16 * k / 32, 0.01f);
        for (auto& v : hw) v = (uint8_t)(rand() & 0x77);
        for (auto& v : hs) v = 118;
        for (auto& v : hx) v = (uint8_t)(rand() & 0x77);
        Args a{};
        CK(hipMalloc((void**)&a.w, hw.size()));
        CK(hipMalloc((void**)&a.sc, hs.size()));
        CK(hipMalloc((void**)&a.xq, hx.size()));
        CK(hipMalloc((void**)&a.xs, hxs.size() * 4));
        CK(hipMalloc((void**)&a.y, (size_t)16 * n * 4));
        CK(hipMalloc((void**)&a.part, (size_t)64 * 16 * n * 4));
        CK(hipMemcpy((void*)a.w, hw.data(), hw.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy((void*)a.sc, hs.data(), hs.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy((void*)a.xq, hx.data(), hx.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy((void*)a.xs, hxs.data(), hxs.size() * 4, hipMemcpyHostToDevice));
        a.n = n;
        a.k = k;
        const double bytes = (double)n * k + hs.size();
        const int KP = k / 64;
        for (int t : toks) {
            a.ntok = t;
            auto run = [&](const char* label, int S, auto launch) {
                a.S = S;
                for (int i = 0; i < 20; i++) launch();
                CK(hipEventRecord(e0));
                const int it = 200;
                for (int i = 0; i < it; i++) launch();
                CK(hipEventRecord(e1));
                CK(hipEventSynchronize(e1));
                float ms;
                CK(hipEventElapsedTime(&ms, e0, e1));
                const double us = ms * 1e3 / it;
                printf("%-6s %5dx%-5d tok %2d  %-14s S=%-3d %7.1f us  %5.2f TB/s\n", sh.name, n, k, t, label, S, us, bytes / us / 1e6);
            };
            const int RT = n / 16;
            for (int S : {1, 2, 4, 8, 16, 32}) {
                if (KP % S || KP / S < 1) continue;
                const int grid = (RT / 4) * S;
                run("A U=2", S, [&] { hipLaunchKernelGGL(dg_a<2>, dim3(grid), dim3(256), 0, 0, a); });
                run("A U=4", S, [&] { hipLaunchKernelGGL(dg_a<4>, dim3(grid), dim3(256), 0, 0, a); });
                if (RT % 8 == 0) run("D R=2 U=2", S, [&] { hipLaunchKernelGGL((dg_d<2, 2>), dim3((RT / 8) * S), dim3(256), 0, 0, a); });
                if (RT % 16 == 0) run("D R=4 U=2", S, [&] { hipLaunchKernelGGL((dg_d<4, 2>), dim3((RT / 16) * S), dim3(256), 0, 0, a); });
                if (RT % 16 == 0) run("D R=4 U=1", S, [&] { hipLaunchKernelGGL((dg_d<4, 1>), dim3((RT / 16) * S), dim3(256), 0, 0, a); });
                if (RT % 32 == 0) run("D R=8 U=1", S, [&] { hipLaunchKernelGGL((dg_d<8, 1>), dim3((RT / 32) * S), dim3(256), 0, 0, a); });
                if (KP / S <= 16) {
                    run("B load-all", S, [&] { hipLaunchKernelGGL((dg_b<16, false>), dim3(grid), dim3(256), 0, 0, a); });
                    run("C partials", S, [&] {
                        hipLaunchKernelGGL((dg_b<16, true>), dim3(grid), dim3(256), 0, 0, a);
                        hipLaunchKernelGGL(dg_reduce, dim3((t * n + 255) / 256), dim3(256), 0, 0, a);
                    });
                }
                if (KP / S <= 32 && KP / S > 16) {
                    run("B load-all32", S, [&] { hipLaunchKernelGGL((dg_b<32, false>), dim3(grid), dim3(256), 0, 0, a); });
                }
            }
        }
        hipFree((void*)a.w); hipFree((void*)a.sc); hipFree((void*)a.xq); hipFree((void*)a.xs); hipFree(a.y); hipFree(a.part);
    }
    return 0;
}
