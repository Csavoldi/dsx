// gemv_lab: FP8 MFMA GEMV correctness against a CPU reference and bandwidth on the model's shapes.
#include "../kernels/gemv.h"
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)

__global__ void quant_k(const float* x, int n, uint8_t* q, float* qs) {
    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i4 * 4 >= n) return;
    float v[4] = {x[4 * i4], x[4 * i4 + 1], x[4 * i4 + 2], x[4 * i4 + 3]};
    quant32(v, i4, q, qs);
}

template <int TT, int U>
__global__ void __launch_bounds__(256) gemv_k(const u4v* W, const uint8_t* wsc, int N, int K, int split, const uint8_t* x, const float* xs, int T, float* y) {
    extern __shared__ char smem[];
    const int RB = (N / 16 + 3) / 4;
    const int rb = blockIdx.x % RB, sl = blockIdx.x / RB;
    const int KP = K / 64;
    const int kp0 = KP * sl / split, kp1 = KP * (sl + 1) / split;
    gemv_fp8_block<TT, U>(W, wsc, N, K, rb * 4, kp0, kp1, x, xs, T, y, 2.f, split > 1, smem);
}
static int gU = 4;
static void launch(const u4v* W, const uint8_t* wsc, int N, int K, int split, const uint8_t* x, const float* xs, int T, float* y) {
    const int RB = (N / 16 + 3) / 4;
    const int S = (K / 64 + split - 1) / split;
    size_t sm1 = ((S * 64 + 8 + 15) & ~15) + S * 2 * 4 + 2 * S * 2 * 4, sm16 = ((16 * (S * 64 + 8) + 15) & ~15) + 16 * S * 2 * 4 + 2 * S * 2 * 4;
#define L(U) if (gU == U) { if (T == 1) gemv_k<1, U><<<RB * split, 256, sm1>>>(W, wsc, N, K, split, x, xs, T, y); else gemv_k<16, U><<<RB * split, 256, sm16>>>(W, wsc, N, K, split, x, xs, T, y); }
    L(2) L(4) L(8)
}
static float ocp_e4m3(uint8_t b) {
    int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? m * ldexpf(1, -9) : (1 + m / 8.f) * ldexpf(1, e - 7);
    return s ? -v : v;
}
static float e4m3_round(float v) {  // nearest OCP e4m3 (RNE), saturating
    float best = 0, bd = 1e30f;
    for (int b = 0; b < 256; b++) {
        if ((b & 0x7f) == 0x7f) continue;
        float c = ocp_e4m3(b), d = fabsf(c - v);
        if (d < bd || (d == bd && (b & 1) == 0)) { bd = d; best = c; }
    }
    return best;
}

int main(int argc, char** argv) {
    std::mt19937 rng(1);
    struct Shape { int N, K, split; const char* name; };
    Shape shapes[] = {{1280, 5120, 16, "wq_a"}, {32768, 1280, 1, "wq_b"}, {8192, 4096, 2, "wo_a"}, {5120, 8192, 4, "wo_b"},
                      {4608, 5120, 4, "sh_gu"}, {5120, 2304, 4, "sh_dn"}, {512, 5120, 32, "wkv"}};
    for (auto& sh : shapes) {
        const int N = sh.N, K = sh.K;
        for (int T : {1, 16}) {
            std::vector<uint8_t> w((size_t)N * K), wt((size_t)N * K), sc((N / 32) * (K / 32));
            for (auto& b : w) { do b = rng() & 0xff; while ((b & 0x7f) == 0x7f); }
            for (auto& s : sc) s = 120 + rng() % 8;
            // tile + convert to FNUZ (0x80 -> 0)
            for (int rt = 0; rt < N / 16; rt++)
                for (int kp = 0; kp < K / 64; kp++)
                    for (int l = 0; l < 64; l++)
                        for (int i = 0; i < 16; i++) {
                            int r = rt * 16 + l % 16, k = kp * 64 + (i / 8) * 32 + 8 * (l / 16) + i % 8;
                            uint8_t b = w[(size_t)r * K + k];
                            wt[(((size_t)rt * (K / 64) + kp) * 64 + l) * 16 + i] = b == 0x80 ? 0 : b;
                        }
            std::vector<float> x((size_t)T * K);
            std::normal_distribution<float> nd(0, 1);
            for (auto& v : x) v = nd(rng) * (rng() % 16 == 0 ? 20 : 1);
            // CPU reference
            std::vector<float> xq(x);
            for (int t = 0; t < T; t++)
                for (int b = 0; b < K / 32; b++) {
                    float a = 1e-4f;
                    for (int i = 0; i < 32; i++) a = fmaxf(a, fabsf(x[t * K + b * 32 + i]));
                    float s = a / 448.f; int e; float m = frexpf(s, &e); s = m == 0.5f ? ldexpf(1, e - 1) : ldexpf(1, e);
                    for (int i = 0; i < 32; i++) xq[t * K + b * 32 + i] = e4m3_round(x[t * K + b * 32 + i] / s) * s;
                }
            std::vector<double> ref((size_t)T * N);
            for (int t = 0; t < T; t++)
                for (int r = 0; r < N; r++) {
                    double s = 0;
                    for (int k = 0; k < K; k++) s += (double)ocp_e4m3(w[(size_t)r * K + k]) * ldexp(1.0, sc[(r / 32) * (K / 32) + k / 32] - 127) * xq[t * K + k];
                    ref[t * N + r] = s;
                }
            uint8_t *dw, *dsc, *dq; float *dx, *dxs, *dy;
            CK(hipMalloc(&dw, wt.size())); CK(hipMalloc(&dsc, sc.size())); CK(hipMalloc(&dq, T * K)); CK(hipMalloc(&dx, T * K * 4));
            CK(hipMalloc(&dxs, T * K / 32 * 4)); CK(hipMalloc(&dy, T * N * 4));
            CK(hipMemcpy(dw, wt.data(), wt.size(), hipMemcpyHostToDevice)); CK(hipMemcpy(dsc, sc.data(), sc.size(), hipMemcpyHostToDevice));
            CK(hipMemcpy(dx, x.data(), T * K * 4, hipMemcpyHostToDevice));
            quant_k<<<(T * K / 4 + 255) / 256, 256>>>(dx, T * K, dq, dxs);
            CK(hipMemset(dy, 0, T * N * 4));
            launch((const u4v*)dw, dsc, N, K, sh.split, dq, dxs, T, dy);
            CK(hipDeviceSynchronize());
            std::vector<float> y(T * N);
            CK(hipMemcpy(y.data(), dy, T * N * 4, hipMemcpyDeviceToHost));
            double err = 0, mag = 0;
            for (size_t i = 0; i < y.size(); i++) { err = fmax(err, fabs(y[i] - ref[i])); mag = fmax(mag, fabs(ref[i])); }
            // timing: big buffer rotation to defeat the 256 MB cache: replicate weights across 1 GB
            const size_t bytes = wt.size();
            const int copies = (int)std::max<size_t>(1, (size_t)(1 << 30) / bytes);
            uint8_t* big; CK(hipMalloc(&big, bytes * copies));
            for (int c = 0; c < copies; c++) CK(hipMemcpy(big + c * bytes, dw, bytes, hipMemcpyDeviceToDevice));
            printf("%-6s N %5d K %5d T %2d: max err %.3g (ref max %.3g) |", sh.name, N, K, T, err, mag);
            hipStream_t st; CK(hipStreamCreate(&st));
            for (int U : {4, 8}) for (int split : {1, 2, 4, 8, 16, 32}) {
                if ((K / 64) / split < 2) continue;
                gU = U;
                hipGraph_t g; hipGraphExec_t ge;
                const int iters = 100;
                CK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
                for (int i = 0; i < iters; i++) {
                    const u4v* Wp = (const u4v*)(big + (i % copies) * bytes);
                    const int RB = (N / 16 + 3) / 4; const int S = (K / 64 + split - 1) / split;
                    size_t sm1 = ((S * 64 + 8 + 15) & ~15) + S * 2 * 4 + 2 * S * 2 * 4, sm16 = ((16 * (S * 64 + 8) + 15) & ~15) + 16 * S * 2 * 4 + 2 * S * 2 * 4;
#define LL(UU) if (U == UU) { if (T == 1) gemv_k<1, UU><<<RB * split, 256, sm1, st>>>(Wp, dsc, N, K, split, dq, dxs, T, dy); else gemv_k<16, UU><<<RB * split, 256, sm16, st>>>(Wp, dsc, N, K, split, dq, dxs, T, dy); }
                    LL(2) LL(4) LL(8)
                }
                CK(hipStreamEndCapture(st, &g)); CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                CK(hipGraphLaunch(ge, st)); CK(hipStreamSynchronize(st));
                hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
                CK(hipEventRecord(e0, st)); CK(hipGraphLaunch(ge, st)); CK(hipEventRecord(e1, st)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1));
                const float us = ms * 1000 / iters;
                printf(" U%d s%d %.1fus %.0f", U, split, us, bytes / us / 1e3);
                hipGraphExecDestroy(ge); hipGraphDestroy(g);
            }
            printf("\n");
            hipFree(big); hipFree(dw); hipFree(dsc); hipFree(dq); hipFree(dx); hipFree(dxs); hipFree(dy);
        }
    }
}
