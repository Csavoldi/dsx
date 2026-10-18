// dsxa.hip: decode kernels for DeepSeek-V4.1-Flash-1017 on one MI325X (gfx942).
//
// One decode step is a fixed sequence of kernels captured once in a HIP graph. Everything that depends on the
// position (ring slots, compressed-row counts, indexer lengths) is read from device memory (`st`), so the same graph
// replays for every token. Numerics follow the CPU reference (engine/src/model.rs, DeepSeek's model.py): bf16
// rounding points, FP8 activation quantization for FP8/FP4 weights, the training-time FP4 fake quantization of caches.
#include "gemv.h"

#define D 5120
#define HC 4
#define NH 64
#define HD 512
#define RD 64
#define QL 1280
#define NE 384
#define INTER 2304
#define WIN 128
#define IH 32
#define ID 128
#define ITOPK 512

struct State {
    int pos;     // position of the token being decoded
    int token;   // its id
};

// ------------------------------------------------------------------ hyper-connections

// model.py hc_split_sinkhorn for one token from raw dots: mix[0..24) = fn . h, mix[24] = sum h^2 (over hc*dim).
// pre[4], post[4], comb[16] (comb[j*4+k]: from stream j to stream k).
DEV void hc_finish(const float* mix, const float* scale, const float* base, float hc_eps, float eps, int iters,
                   float* pre, float* post, float* comb) {
    const float r = 1.f / sqrtf(mix[24] / (HC * D) + eps);
    for (int j = 0; j < HC; j++) {
        pre[j] = 1.f / (1.f + expf(-(mix[j] * r * scale[0] + base[j]))) + hc_eps;
        post[j] = 2.f / (1.f + expf(-(mix[j + HC] * r * scale[1] + base[j + HC])));
    }
    for (int j = 0; j < HC; j++) {
        float mx = -INFINITY;
        for (int k = 0; k < HC; k++) {
            comb[j * HC + k] = mix[j * HC + k + 2 * HC] * r * scale[2] + base[j * HC + k + 2 * HC];
            mx = fmaxf(mx, comb[j * HC + k]);
        }
        float s = 0;
        for (int k = 0; k < HC; k++) s += (comb[j * HC + k] = expf(comb[j * HC + k] - mx));
        for (int k = 0; k < HC; k++) comb[j * HC + k] = comb[j * HC + k] / s + hc_eps;
    }
    auto col = [&]() {
        for (int k = 0; k < HC; k++) {
            float s = 0;
            for (int j = 0; j < HC; j++) s += comb[j * HC + k];
            for (int j = 0; j < HC; j++) comb[j * HC + k] /= s + hc_eps;
        }
    };
    col();
    for (int it = 0; it < iters - 1; it++) {
        for (int j = 0; j < HC; j++) {
            float s = 0;
            for (int k = 0; k < HC; k++) s += comb[j * HC + k];
            for (int k = 0; k < HC; k++) comb[j * HC + k] /= s + hc_eps;
        }
        col();
    }
}

struct HcArgs {
    float scale[3];
    float base[24];
};

// Residual-stream step between sub-blocks, one block per 64 dims (grid D/64, 256 threads: thread = stream*64 + dim):
//   post (optional): h = bf16(post[k] * bf16?(y) + sum_j comb[j][k] h[j])   with mixes from `pmix` (previous sub-block)
//   engram (optional): h[k] = bf16(h[k] + gate[k] * value)                   gate from engram sums
//   x_raw = bf16(sum_k pre[k] h[k]); xss += sum x_raw^2                      pre from `premix` (or one-hot if null)
//   nmix[0..24) += fn . h; nmix[24] += sum h^2                              raw dots for the next sub-block's mixes
extern "C" __global__ void __launch_bounds__(256) hc_step(float* __restrict__ h, const float* __restrict__ y, int round_y,
                                               const float* pmix, HcArgs pa, const float* premix, HcArgs prea,
                                               const float* __restrict__ fn, float* nmix, float* x_raw, float* xss,
                                               const float* eng_val, const float* eng_sum, const float* eng_qk,
                                               float hc_eps, float eps, int iters) {
    __shared__ float hs[HC][64];
    __shared__ float post[4], comb[16], pre[4];
    const int k = threadIdx.x >> 6, i = threadIdx.x & 63, d = blockIdx.x * 64 + i;
    if (threadIdx.x == 0) {
        float p2[4], c2[16], pr[4];
        if (pmix) {
            hc_finish(pmix, pa.scale, pa.base, hc_eps, eps, iters, pr, p2, c2);
            for (int j = 0; j < 4; j++) post[j] = p2[j];
            for (int j = 0; j < 16; j++) comb[j] = c2[j];
        }
        if (premix) {
            hc_finish(premix, prea.scale, prea.base, hc_eps, eps, iters, pr, p2, c2);
            for (int j = 0; j < 4; j++) pre[j] = pr[j];
        } else {
            for (int j = 0; j < 4; j++) pre[j] = j == 0 ? 1.f : 0.f;
        }
    }
    float hv = h[k * D + d];
    hs[k][i] = hv;
    __syncthreads();
    if (pmix) {
        float yv = y[d];
        if (round_y) yv = round_bf16(yv);
        float v = post[k] * yv;
        for (int j = 0; j < HC; j++) v += comb[j * HC + k] * hs[j][i];
        hv = round_bf16(v);
    }
    if (eng_val) {
        // eng_sum[k*3 + {0,1,2}] = sum h^2, sum key^2, sum h*w*key over the stream (from engram_proj's epilogue)
        const float rh = 1.f / sqrtf(eng_sum[k * 3] / D + eps), rk = 1.f / sqrtf(eng_sum[k * 3 + 1] / D + eps);
        const float dt = eng_sum[k * 3 + 2] * rh * rk / sqrtf((float)D);
        const float g = 1.f / (1.f + expf(-copysignf(sqrtf(fmaxf(fabsf(dt), 1e-6f)), dt)));
        hv = round_bf16(hv + g * eng_val[d]);
    }
    __syncthreads();
    hs[k][i] = hv;
    h[k * D + d] = hv;
    __syncthreads();
    // x_raw and its sum of squares (stream-0 threads)
    if (k == 0 && x_raw) {
        float x = 0;
        for (int j = 0; j < HC; j++) x += pre[j] * hs[j][i];
        x = round_bf16(x);
        x_raw[d] = x;
        float s = wave_sum(x * x);
        if (i == 0) atomicAdd(xss, s);
    }
    // next sub-block's mix dots: 24 rows of fn over this block's 4x64 h values + sum h^2
    if (nmix) {
        float part[25];
        for (int j = 0; j < 24; j++) part[j] = fn[(size_t)j * HC * D + k * D + d] * hv;
        part[24] = hv * hv;
        for (int j = 0; j < 25; j++) part[j] = wave_sum(part[j]);
        if (i == 0)
            for (int j = 0; j < 25; j++) atomicAdd(&nmix[j], part[j]);
    }
}

// The final norm input: x = bf16(sum_k pre[k] h[k]) with pre from the last ffn mixes, into xf (f32, for the head).
extern "C" __global__ void __launch_bounds__(256) hc_final(const float* __restrict__ h, const float* premix, HcArgs prea,
                                                const float* norm_w, float* xout, float hc_eps, float eps, int iters) {
    __shared__ float pre[4];
    __shared__ float red[16];
    if (threadIdx.x == 0) {
        float p[4], q[4], c[16];
        hc_finish(premix, prea.scale, prea.base, hc_eps, eps, iters, p, q, c);
        for (int j = 0; j < 4; j++) pre[j] = p[j];
    }
    __syncthreads();
    float xv[D / 256];
    float ss = 0;
    for (int u = 0; u < D / 256; u++) {
        const int d = u * 256 + threadIdx.x;
        float x = 0;
        for (int j = 0; j < HC; j++) x += pre[j] * h[j * D + d];
        x = round_bf16(x);
        xv[u] = x;
        ss += x * x;
    }
    ss = block_sum(ss, red);
    const float r = 1.f / sqrtf(ss / D + eps);
    for (int u = 0; u < D / 256; u++) {
        const int d = u * 256 + threadIdx.x;
        xout[d] = norm_w[d] * (xv[u] * r);
    }
}

// token embedding (bf16 table) into all 4 streams
extern "C" __global__ void embed(const uint16_t* __restrict__ emb, const State* st, float* h) {
    const int tok = st->token;
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < D; d += gridDim.x * blockDim.x) {
        const float v = bf16_to_f32(emb[(size_t)tok * D + d]);
        for (int k = 0; k < HC; k++) h[k * D + d] = v;
    }
}

// ------------------------------------------------------------------ GEMV jobs

// Activation transforms staged into LDS by each GEMV block for its k-slice
enum XKind {
    X_NORM = 0,     // v = bf16(w[i] * (x[i] * rsqrt(ss/n + eps)))       (x_raw + sum of squares from the producer)
    X_RMS = 1,      // same, but the block computes ss itself over the whole vector (n small: q_a's 1280)
    X_PLAIN = 2,    // v = x[i]
    X_BF16R = 3,    // v = bf16(x[i])
    X_SWIGLU = 4,   // v = bf16(silu(min(g, lim)) * clamp(u, -lim, lim) * wt)  (x = g, x2 = u, wt from routing)
    X_ENGRAM = 5,   // v = bf16(fp8(row byte) * e8m0 scale), rows from the host-filled buffer
    X_Q = 6,        // already FP8: x = codes (uint8 FNUZ), x2 = scales per 32 (FP8 weights only)
};

enum WKind { W_FP8 = 0, W_BF16 = 1, W_MXFP4 = 2, W_EXPERT = 3 };
enum EFmt { E_FP4 = 0, E_CB3 = 1, E_CB2 = 2 };

// routed-expert matrix: weights at arena + woff, scales at arena + soff, result multiplier, format
struct ExpertMat {
    long long woff, soff;
    float mul;
    int fmt;
    int pad[2];
};

struct Job {
    const void* w;        // tiled weights
    const void* sc;       // scales
    float* y;             // output rows
    const float* x;       // activation source (meaning per xkind)
    const float* x2;      // second source (swiglu up)
    const float* nw;      // norm weight (bf16 values as f32) for X_NORM / X_RMS
    const float* ss;      // sum of squares for X_NORM
    int n, k, split;      // rows, cols, k-slices
    int wkind, xkind;
    int blocks;           // ceil(n/64) * split
    int xgroup;           // rows per activation group (wo_a: rows [g*xgroup, +xgroup) read x[g*k ...]); 0 = none
    int eslot;            // routed-expert slot (0..5) whose id picks the weights, -1 = fixed
    long long wstride, sstride;  // bytes per expert
    int lo;               // bf16 weights: also multiply the low bf16 half of f32 activations
    int epi;              // epilogue: 0 store/add, 1 rope (q heads: 512, rope dims last 64), 2 rope idx (128)
    int ropec;            // rope frequencies: 0 plain, 1 compressed (YaRN)
    int emat;             // routed experts: matrix 0 = w1 (gate), 1 = w3 (up), 2 = w2 (down); J.sc = ExpertMat[384][3]
};

struct Jobs {
    Job j[13];
    int n;
    float eps;
    float lim;
    uint32_t cb3[2];  // codebooks (FNUZ fp8 bytes): 3-bit lo/hi entries, 2-bit
    uint32_t cb2;
};

struct RopeTab {
    const float* freqs;  // [RD/2] per variant
};

DEV void rope_pair(float& a, float& b, float ang) {
    const float s = sinf(ang), c = cosf(ang);
    const float x = a * c - b * s, y = a * s + b * c;
    a = x;
    b = y;
}

// expert routing results shared by the expert kernels: ids[6], wts[6]
struct Route {
    int ids[8];
    float wts[8];
};

// Router top-6 (model.py Gate, sqrtsoftplus + bias, noaux_tc): by one wave; writes ids/weights into r.
DEV void route_top6(const float* logits, const float* bias, float route_scale, Route* r, int lane) {
    float sc[6], key[6];
    for (int j = 0; j < 6; j++) {
        const int e = lane + 64 * j;
        const float l = logits[e];
        const float sp = l > 20.f ? l : log1pf(expf(l));
        sc[j] = sqrtf(sp);
        key[j] = sc[j] + bias[e];
    }
    float sum = 0;
    for (int s = 0; s < 6; s++) {
        // best (key desc, index asc) among this lane's remaining picks, then across the wave
        float bk = -INFINITY;
        int bi = 1 << 30, bj = -1;
        for (int j = 0; j < 6; j++)
            if (key[j] > bk) { bk = key[j]; bi = lane + 64 * j; bj = j; }
        for (int o = 32; o > 0; o >>= 1) {
            const float ok = __shfl_xor(bk, o);
            const int oi = __shfl_xor(bi, o);
            if (ok > bk || (ok == bk && oi < bi)) { bk = ok; bi = oi; }
        }
        const float scv = __shfl(bj >= 0 ? sc[bj < 0 ? 0 : bj] : 0.f, bi & 63);
        if (lane == (bi & 63)) key[bi >> 6] = -INFINITY;
        if (lane == 0) r->ids[s] = bi;
        r->wts[s] = scv;
        sum += scv;
    }
    if (lane == 0)
        for (int s = 0; s < 6; s++) r->wts[s] = r->wts[s] / (sum + 1e-20f) * route_scale;
}

// One GEMV block of a multi-job launch. 256 threads.
// Stage this block's activation slice into LDS, then each wave computes one 16-row tile over the slice.
// route_mode: 1 = pick experts from the router logits (and block 0 publishes them in `route`), 2 = read `route`.
extern "C" __global__ void __launch_bounds__(256) gemv_jobs(Jobs js, const State* st, const float* rope_f, const float* rope_fc,
                                                 int route_mode, const float* router_logits, const float* router_bias,
                                                 float route_scale, Route* route, const uint8_t* engram_rows) {
    extern __shared__ char smem[];
    __shared__ float red[16];
    __shared__ Route rsh;
    int b = blockIdx.x, ji = 0;
    while (ji < js.n - 1 && b >= js.j[ji].blocks) b -= js.j[ji++].blocks;
    const Job& J = js.j[ji];
    const int tid = threadIdx.x, lane = tid & 63, wv = tid >> 6;
    const int RB = (J.n + 63) / 64;
    const int rb = b % RB, sl = b / RB;
    const int KU = J.wkind == W_FP8 ? 64 : (J.wkind == W_BF16 ? 32 : 128);  // k per weight unit
    const int KUs = J.k / KU;
    const int u0 = KUs * sl / J.split, u1 = KUs * (sl + 1) / J.split;
    const int S = u1 - u0;
    const int k0 = u0 * KU, kn = S * KU;
    const int rt = rb * 4 + wv;
    const bool rowok = rt * 16 < J.n;

    // routed experts: the block picks the expert's weights from the router logits
    const char* wbase = (const char*)J.w;
    const char* sbase = (const char*)J.sc;
    float ewt = 1.f, emul = 1.f;
    int efmt = E_FP4;
    if (J.eslot >= 0) {
        if (route_mode == 1) {
            if (wv == 0) route_top6(router_logits, router_bias, route_scale, &rsh, lane);
        } else if (tid < 6) {
            rsh.ids[tid] = route->ids[tid];
            rsh.wts[tid] = route->wts[tid];
        }
        __syncthreads();
        const int e = rsh.ids[J.eslot];
        ewt = rsh.wts[J.eslot];
        if (J.wkind == W_EXPERT) {
            const ExpertMat em = ((const ExpertMat*)J.sc)[e * 3 + J.emat];
            efmt = em.fmt;
            emul = em.mul;
            wbase = (const char*)J.w + em.woff;
            sbase = (const char*)J.w + em.soff;
        } else {
            wbase += (long long)e * J.wstride;
            sbase += (long long)e * J.sstride;
        }
        if (route_mode == 1 && blockIdx.x == 0 && tid < 6) { route->ids[tid] = rsh.ids[tid]; route->wts[tid] = rsh.wts[tid]; }
    }

    // ---- stage the activation slice [k0, k0+kn) for this job's x group
    int gx = 0;  // activation group offset
    if (J.xgroup) gx = (rb * 64 / J.xgroup) * J.k;
    float scale_all = 1.f;
    if (J.xkind == X_NORM) scale_all = 1.f / sqrtf(J.ss[0] / J.k + js.eps);
    if (J.xkind == X_RMS) {
        float s = 0;
        for (int i = tid; i < J.k; i += 256) { const float v = J.x[i]; s += v * v; }
        s = block_sum(s, red);
        scale_all = 1.f / sqrtf(s / J.k + js.eps);
    }
    auto act = [&](int i) -> float {  // i: absolute k index within the job's activation
        switch (J.xkind) {
            case X_NORM:
            case X_RMS: return round_bf16(J.nw[i] * (J.x[gx + i] * scale_all));
            case X_PLAIN: return J.x[gx + i];
            case X_BF16R: return round_bf16(J.x[gx + i]);
            case X_SWIGLU: {
                const float g = fminf(J.x[i], js.lim), u = fminf(fmaxf(J.x2[i], -js.lim), js.lim);
                return round_bf16(g / (1.f + expf(-g)) * u * ewt);
            }
            default: {  // X_ENGRAM: rows buffer = [24 rows][256 fp8] then [24][8] e8m0 scales
                const int row = i >> 8, j = i & 255;
                const uint8_t b8 = engram_rows[row * 256 + j];
                const uint8_t sc = engram_rows[24 * 256 + row * 8 + (j >> 5)];
                // OCP e4m3 byte -> f32
                const int s = b8 >> 7, e = (b8 >> 3) & 15, m = b8 & 7;
                float v = e == 0 ? ldexpf((float)m, -9) : ldexpf(1.f + m / 8.f, e - 7);
                v = s ? -v : v;
                return round_bf16(v * __uint_as_float((uint32_t)sc << 23));
            }
        }
    };
    if (J.wkind == W_BF16) {
        uint16_t* xh = (uint16_t*)smem;
        uint16_t* xl = xh + kn + 8;
        for (int i = tid; i < kn; i += 256) {
            const float v = act(k0 + i);
            const uint32_t hb = bf16_bits(v);
            xh[i] = (uint16_t)hb;
            xl[i] = (uint16_t)bf16_bits(v - __uint_as_float(hb << 16));
        }
    } else if (J.xkind == X_Q) {
        uint8_t* xq = (uint8_t*)smem;
        float* xs = (float*)(smem + ((kn + 8 + 15) & ~15));
        const uint8_t* src = (const uint8_t*)J.x + gx + k0;
        for (int i = tid; i < kn / 4; i += 256) reinterpret_cast<uint32_t*>(xq)[i] = reinterpret_cast<const uint32_t*>(src)[i];
        for (int i = tid; i < kn / 32; i += 256) xs[i] = J.x2[(gx + k0) / 32 + i];
    } else {
        uint8_t* xq = (uint8_t*)smem;
        float* xs = (float*)(smem + ((kn + 8 + 15) & ~15));
        // 4 values per thread, 8 threads per 32-block
        for (int i4 = tid; i4 < kn / 4; i4 += 256) {
            float v[4];
            for (int q = 0; q < 4; q++) v[q] = act(k0 + 4 * i4 + q);
            quant32(v, reinterpret_cast<uint32_t*>(xq) + i4, xs + (i4 >> 3));
        }
    }
    __syncthreads();
    if (!rowok) return;

    // ---- the wave's tile
    f4 acc;
    const int g16 = lane >> 4;
    float mul = 1.f;
    if (J.wkind == W_FP8) {
        const uint8_t* xq = (const uint8_t*)smem;
        const float* xs = (const float*)(smem + ((kn + 8 + 15) & ~15));
        // weight scales for this tile's 32-row block, k-blocks [2*u0, 2*u1)
        float* wsl = (float*)(smem + ((kn + 8 + 15) & ~15) + kn / 32 * 4) + wv * 2 * S;
        const uint8_t* sp = (const uint8_t*)sbase + (size_t)(rt >> 1) * (J.k / 32) + 2 * u0;
        for (int i = lane; i < 2 * S; i += 64) wsl[i] = __uint_as_float((uint32_t)sp[i] << 23);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
        const u4v* wp = (const u4v*)wbase + ((size_t)rt * KUs + u0) * 64 + lane;
        acc = wave_fp8<4>(wp, S, xq + 8 * g16, xs, wsl, true);
        mul = 2.f;
    } else if (J.wkind == W_MXFP4 || (J.wkind == W_EXPERT && efmt == E_FP4)) {
        const uint8_t* xq = (const uint8_t*)smem;
        const float* xs = (const float*)(smem + ((kn + 8 + 15) & ~15));
        const u4v* wp = (const u4v*)wbase + ((size_t)rt * KUs + u0) * 64 + lane;
        const u4v* sp = (const u4v*)sbase + ((size_t)rt * KUs + u0) * 4 + g16;
        acc = wave_mxfp4<2>(wp, sp, S, xq + 8 * g16, xs, true);
        mul = emul;  // the e2m1 table holds exact values (FP8 weights instead hold OCP/2: x2)
    } else if (J.wkind == W_EXPERT) {
        const uint8_t* xq = (const uint8_t*)smem;
        const float* xs = (const float*)(smem + ((kn + 8 + 15) & ~15));
        const u4v* sp = (const u4v*)sbase + ((size_t)rt * KUs + u0) * 4 + g16;
        if (efmt == E_CB3) {
            const uint32_t* wp = (const uint32_t*)wbase + ((size_t)rt * KUs + u0) * 64 * 3 + lane * 3;
            acc = wave_cb<3>(wp, sp, S, xq + 8 * g16, xs, js.cb3[0], js.cb3[1], true);
        } else {
            const uint32_t* wp = (const uint32_t*)wbase + ((size_t)rt * KUs + u0) * 64 * 2 + lane * 2;
            acc = wave_cb<2>(wp, sp, S, xq + 8 * g16, xs, js.cb2, 0u, true);
        }
        mul = emul;
    } else {
        const uint16_t* xh = (const uint16_t*)smem;
        const uint16_t* xl = xh + kn + 8;
        const u4v* wp = (const u4v*)wbase + ((size_t)rt * KUs + u0) * 64 + lane;
        acc = wave_bf16<4>(wp, S, xh + 4 * g16, xl + 4 * g16, J.lo != 0, true);
    }
    // ---- epilogue: lanes 0, 16, 32, 48 (token 0) hold rows 16*rt + 4*g16 + j
    if ((lane & 15) != 0) return;
    const int r0 = rt * 16 + 4 * g16;
    float v[4];
    for (int j = 0; j < 4; j++) v[j] = acc[j] * mul;
    if (J.epi == 1 || J.epi == 2) {
        // RoPE on the last RD dims of each head (head size 512 for q, 128 for the indexer's q); pairs (2i, 2i+1)
        const int hd = J.epi == 1 ? HD : ID;
        const int dim = r0 % hd;
        if (dim >= hd - RD) {
            const float* f = J.ropec ? rope_fc : rope_f;
            const int pos = st->pos;
            for (int p = 0; p < 2; p++) {
                const int pi = (dim - (hd - RD)) / 2 + p;
                rope_pair(v[2 * p], v[2 * p + 1], (float)pos * f[pi]);
            }
        }
    }
    if (J.split > 1) {
        for (int j = 0; j < 4; j++) atomicAdd(J.y + r0 + j, v[j]);
    } else {
        for (int j = 0; j < 4; j++) J.y[r0 + j] = v[j];
    }
}

// Engram sums from its projection output kv[25600] = [4 x 5120 key | 5120 value] and h:
// out[k*3] += sum h^2, out[k*3+1] += sum key^2, out[k*3+2] += sum h*w*key  (w = q_weight * k_weight)
extern "C" __global__ void engram_sums(const float* __restrict__ kv, const float* __restrict__ h, const float* __restrict__ qk, float* out) {
    __shared__ float red[16];
    const int k = blockIdx.y;
    float a = 0, b = 0, c = 0;
    for (int d = blockIdx.x * 256 + threadIdx.x; d < D; d += gridDim.x * 256) {
        const float hv = h[k * D + d], kv_ = kv[k * D + d];
        a += hv * hv;
        b += kv_ * kv_;
        c += hv * qk[k * D + d] * kv_;
    }
    a = block_sum(a, red);
    b = block_sum(b, red);
    c = block_sum(c, red);
    if (threadIdx.x == 0) { atomicAdd(out + k * 3, a); atomicAdd(out + k * 3 + 1, b); atomicAdd(out + k * 3 + 2, c); }
}

// ------------------------------------------------------------------ attention side

// e4m3 OCP rounding of a value in units of its block scale (|v| <= 448), RNE: via the FNUZ converter at half scale
DEV float e4m3_round(float v) {
    const uint32_t c = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(v * 0.5f, 0.f, 0, false);
    return 2.f * __builtin_amdgcn_cvt_f32_fp8(c, 0);
}
// e2m1 rounding (RNE on the grid 0, .5, 1, 1.5, 2, 3, 4, 6), |v| <= 6
DEV float e2m1_round(float v) {
    const float a = fabsf(v);
    float r;
    if (a <= 0.25f) r = 0.f;
    else if (a < 0.75f) r = 0.5f;
    else if (a <= 1.25f) r = 1.f;
    else if (a < 1.75f) r = 1.5f;
    else if (a <= 2.5f) r = 2.f;
    else if (a < 3.5f) r = 3.f;
    else if (a <= 5.f) r = 4.f;
    else r = 6.f;
    return copysignf(r, v);
}

// fake_fp8 over a 32-value group held 1 per lane in a 32-lane group
DEV float fake_fp8_lane(float v) {
    const float a = group_max<32>(fabsf(v));
    const float s = pow2_ceil(fmaxf(a, 1e-4f) / 448.f);
    return e4m3_round(fminf(fmaxf(v / s, -448.f), 448.f)) * s;
}
// fake_fp4 over a group of G lanes (16 or 32): E8M0 (pow2) or E4M3 scale
template <int G> DEV float fake_fp4_lane(float v, bool e4m3s) {
    const float a = group_max<G>(fabsf(v));
    float s;
    if (e4m3s) s = e4m3_round(fmaxf(a, 6.f * ldexpf(1.f, -9)) / 6.f);
    else s = pow2_ceil(fmaxf(a, 6.f * ldexpf(1.f, -126)) / 6.f);
    return e2m1_round(fminf(fmaxf(v / s, -6.f), 6.f)) * s;
}

// Per-layer attention bookkeeping after the projections (one block, 512 threads):
//   kv: rmsnorm(kv_raw, kv_norm), rope at pos, fake_fp8 per 32, store bf16 into the window ring.
//   compressor (kv-source layers): ratio 1: latent = rmsnorm(ckv); ratio 2: keep (ckv, cgate) in slot pos%2 and pool
//   on odd positions. With a latent: indexer key = fp4(rope(rmsnorm(wk . latent))) -> idx cache; latent ->
//   fp4(rope(latent)) -> comp cache.
extern "C" __global__ void __launch_bounds__(512) attn_kv(const State* st, const float* kv_raw, const float* kv_norm, uint16_t* ring,
                                               const float* rope_f, int ratio, const float* ckv, const float* cgate,
                                               float* cstate, const float* comp_norm, const uint16_t* wk,
                                               const float* knorm, uint16_t* comp_cache, uint16_t* idx_cache,
                                               const float* rope_fc, float eps) {
    __shared__ float red[16];
    __shared__ float lat[HD];
    __shared__ float kk[ID];
    const int t = st->pos, i = threadIdx.x;
    const float* rf = ratio > 0 ? rope_fc : rope_f;
    // window kv
    {
        float v = kv_raw[i];
        const float ss = block_sum(v * v, red);
        v = kv_norm[i] * (v * (1.f / sqrtf(ss / HD + eps)));
        // rope: pairs (2p, 2p+1) on dims >= HD-RD
        const float other = __shfl_xor(v, 1);
        if (i >= HD - RD) {
            const int p = (i - (HD - RD)) / 2;
            const float ang = (float)t * rf[p];
            const float s = sinf(ang), c = cosf(ang);
            v = (i & 1) ? other * s + v * c : v * c - other * s;
        }
        v = fake_fp8_lane(v);
        ring[(size_t)(t % WIN) * HD + i] = (uint16_t)bf16_bits(v);
    }
    if (!ckv) return;
    // compressor
    bool have = false;
    if (ratio == 1) {
        float v = ckv[i];
        const float ss = block_sum(v * v, red);
        lat[i] = comp_norm[i] * (v * (1.f / sqrtf(ss / HD + eps)));
        have = true;
    } else {
        const int slot = t % ratio;
        cstate[slot * 2 * HD + i] = ckv[i];
        cstate[slot * 2 * HD + HD + i] = cgate[i];
        __syncthreads();
        if ((t + 1) % ratio == 0) {
            float mx = -INFINITY;
            for (int s = 0; s < ratio; s++) mx = fmaxf(mx, cstate[s * 2 * HD + HD + i]);
            float den = 0, num = 0;
            for (int s = 0; s < ratio; s++) {
                const float e = expf(cstate[s * 2 * HD + HD + i] - mx);
                den += e;
            }
            for (int s = 0; s < ratio; s++) num += cstate[s * 2 * HD + i] * expf(cstate[s * 2 * HD + HD + i] - mx) / den;
            float v = round_bf16(num);
            const float ss = block_sum(v * v, red);
            lat[i] = comp_norm[i] * (v * (1.f / sqrtf(ss / HD + eps)));
            have = true;
        } else {
            __syncthreads();
        }
    }
    if (!have) return;
    __syncthreads();
    const int n = (t + 1) / ratio - 1;  // index of the new compressed row
    const int cpos = t + 1 - ratio;
    // indexer key: wk [128][512] bf16 . lat -> 4 threads per output (128 outputs x 4 = 512 threads)
    {
        const int o = i >> 2, q = i & 3;
        float s = 0;
        for (int j = q; j < HD; j += 4) s += bf16_to_f32(wk[o * HD + j]) * lat[j];
        s += __shfl_xor(s, 1);
        s += __shfl_xor(s, 2);
        if (q == 0) kk[o] = s;
    }
    __syncthreads();
    if (i < ID) {
        float v = kk[i];
        float ss = wave_sum(v * v);
        ss += __shfl_xor(0.f, 0);  // (two waves: combine below)
        red[8 + (i >> 6)] = ss;
    }
    __syncthreads();
    if (i < ID) {
        const float ss = red[8] + red[9];
        float v = knorm[i] * (kk[i] * (1.f / sqrtf(ss / ID + eps)));
        const float other = __shfl_xor(v, 1);
        if (i >= ID - RD) {
            const int p = (i - (ID - RD)) / 2;
            const float ang = (float)cpos * rope_fc[p];
            const float s = sinf(ang), c = cosf(ang);
            v = (i & 1) ? other * s + v * c : v * c - other * s;
        }
        v = fake_fp4_lane<32>(v, false);
        idx_cache[(size_t)n * ID + i] = (uint16_t)bf16_bits(v);
    }
    // compressed kv row
    {
        float v = lat[i];
        const float other = __shfl_xor(v, 1);
        if (i >= HD - RD) {
            const int p = (i - (HD - RD)) / 2;
            const float ang = (float)cpos * rope_fc[p];
            const float s = sinf(ang), c = cosf(ang);
            v = (i & 1) ? other * s + v * c : v * c - other * s;
        }
        v = fake_fp4_lane<16>(v, true);
        comp_cache[(size_t)n * HD + i] = (uint16_t)bf16_bits(v);
    }
}

// Indexer scores: score[s] = sum_h relu(q_h . k_s) * w_h over s in [0, end) (or the rows listed in `rows`).
// q: [32][128] f32 (roped), fake_fp4 per 32 here; w: raw weights * wscale, bf16-rounded.
// One wave per 16 keys, MFMA bf16: A = keys (16 x 128), B = q^T (128 x 16 heads) for two head tiles.
extern "C" __global__ void __launch_bounds__(256) idx_score(const State* st, int ratio, const float* q, const float* wraw, float wscale,
                                                 const uint16_t* __restrict__ kc, const int* rows, const int* nrows,
                                                 float* score) {
    __shared__ uint16_t qs[IH][ID + 8];
    __shared__ float ws[IH];
    const int t = st->pos;
    const int end = rows ? *nrows : (t + 1) / ratio;
    const int tid = threadIdx.x;
    for (int i = tid; i < IH * ID; i += 256) {
        float v = q[i];
        v = fake_fp4_lane<32>(v, false);
        qs[i / ID][i % ID] = (uint16_t)bf16_bits(v);
    }
    if (tid < IH) ws[tid] = round_bf16(wraw[tid] * wscale);
    __syncthreads();
    const int lane = tid & 63, g = lane >> 4, c = lane & 15;
    const int nw = gridDim.x * 4;
    for (int base = (blockIdx.x * 4 + (tid >> 6)) * 16; base < end; base += nw * 16) {
        const int s = base + c;  // key row for the A operand (lane row = lane%16)
        const int row = s < end ? (rows ? rows[s] : s) : -1;
        f4 acc0 = {0, 0, 0, 0}, acc1 = {0, 0, 0, 0};
#pragma unroll
        for (int kq = 0; kq < ID / 16; kq++) {
            s4v a = {0, 0, 0, 0};
            if (row >= 0) a = *reinterpret_cast<const s4v*>(kc + (size_t)row * ID + kq * 16 + 4 * g);
            const s4v b0 = *reinterpret_cast<const s4v*>(&qs[c][kq * 16 + 4 * g]);
            const s4v b1 = *reinterpret_cast<const s4v*>(&qs[16 + c][kq * 16 + 4 * g]);
            acc0 = mfma_bf16(a, b0, acc0);
            acc1 = mfma_bf16(a, b1, acc1);
        }
        // lane: head c (and 16+c), keys base + 4g + j
        float p[4];
        for (int j = 0; j < 4; j++) p[j] = fmaxf(acc0[j], 0.f) * ws[c] + fmaxf(acc1[j], 0.f) * ws[16 + c];
        for (int j = 0; j < 4; j++) p[j] = group_sum<16>(p[j]);
        if (c == 0)
            for (int j = 0; j < 4; j++) {
                const int sj = base + 4 * g + j;
                if (sj < end) score[sj] = p[j];
            }
    }
}

// ------------------------------------------------------------------ top-k (radix select on order-preserving keys)

DEV uint32_t fkey(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

struct TopCtl {
    int n;          // element count (resolved by topk_hist from st when n_src == null)
    int bin;        // threshold bin
    int need;       // how many to take from the threshold bin
    int nsel;       // selected so far (above the threshold bin)
    int ncand;      // candidates in the threshold bin
    int k;          // k actually taken (min(k, valid))
    int pad[2];
};

// histogram of the top 12 bits; n = (pos+1)/ratio (ratio>0) or *n_src; -inf entries are skipped
extern "C" __global__ void __launch_bounds__(256) topk_hist(const State* st, int ratio, const int* n_src, const float* sc, uint32_t* hist, TopCtl* ctl) {
    const int n = n_src ? *n_src : (st->pos + 1) / ratio;
    if (blockIdx.x == 0 && threadIdx.x == 0) ctl->n = n;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        const float v = sc[i];
        if (v == -INFINITY) continue;
        atomicAdd(&hist[fkey(v) >> 20], 1u);
    }
}

// one block: find the bin holding the k-th largest
extern "C" __global__ void __launch_bounds__(1024) topk_scan(const uint32_t* hist, int k, TopCtl* ctl) {
    __shared__ uint32_t part[1024];
    const int tid = threadIdx.x;
    // each thread owns 4 bins, from the top
    uint32_t c[4], s = 0;
    for (int q = 0; q < 4; q++) { c[q] = hist[4095 - (tid * 4 + q)]; s += c[q]; }
    part[tid] = s;
    __syncthreads();
    // inclusive scan (Hillis-Steele)
    for (int o = 1; o < 1024; o <<= 1) {
        const uint32_t v = tid >= o ? part[tid - o] : 0;
        __syncthreads();
        part[tid] += v;
        __syncthreads();
    }
    const uint32_t total = part[1023];
    const uint32_t kk = min((uint32_t)k, total);
    uint32_t before = tid ? part[tid - 1] : 0;
    if (kk == 0) {
        if (tid == 0) { ctl->bin = 4096; ctl->need = 0; ctl->k = 0; ctl->nsel = 0; ctl->ncand = 0; }
        return;
    }
    if (before < kk && part[tid] >= kk) {
        uint32_t acc = before;
        for (int q = 0; q < 4; q++) {
            if (acc + c[q] >= kk) {
                ctl->bin = 4095 - (tid * 4 + q);
                ctl->need = kk - acc;
                ctl->k = kk;
                ctl->nsel = 0;
                ctl->ncand = 0;
                break;
            }
            acc += c[q];
        }
    }
}

// split elements: above the bin -> sel; in the bin -> cand (with their keys)
extern "C" __global__ void __launch_bounds__(256) topk_collect(const float* sc, TopCtl* ctl, int* sel, int* cand, uint32_t* ckey, int cand_cap) {
    const int n = ctl->n, bin = ctl->bin;
    if (bin >= 4096) return;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        const float v = sc[i];
        if (v == -INFINITY) continue;
        const uint32_t key = fkey(v);
        const int b = key >> 20;
        if (b > bin) sel[atomicAdd(&ctl->nsel, 1)] = i;
        else if (b == bin) {
            const int p = atomicAdd(&ctl->ncand, 1);
            if (p < cand_cap) { cand[p] = i; ckey[p] = key; }
        }
    }
}

// one block (1024 threads): take `need` from the candidates (largest key, then smallest index), then sort the k
// selected indices ascending and write out[i] = idx (+ offset), -1 padding up to kmax; *nout = k.
// map: optional index -> row map (indexer rows from a candidate list). zero_hist: cleared for the next use.
extern "C" __global__ void __launch_bounds__(1024) topk_final(TopCtl* ctl, const int* sel, const int* cand, const uint32_t* ckey,
                                                   const int* map, int offset, int* out, int kmax, int* nout,
                                                   uint32_t* zero_hist) {
    __shared__ uint64_t buf[4096];
    __shared__ int idx[2048];
    __shared__ uint32_t h2[256];
    __shared__ int thr_bin, thr_need, taken;
    const int tid = threadIdx.x;
    for (int i = tid; i < 4096; i += 1024) zero_hist[i] = 0;
    const int k = ctl->k, nsel = ctl->nsel;
    int need = ctl->need;
    int nc = min(ctl->ncand, 1 << 30);
    for (int i = tid; i < nsel; i += 1024) idx[i] = sel[i];
    // narrow the candidates by the next bits until they fit in 4096 (keys 20 bits left)
    uint32_t mask = 0xFFF00000u, want = (ctl->bin << 20);
    int shift = 12;
    if (tid == 0) taken = 0;
    __syncthreads();
    // candidates beyond the 4096 buffer: radix on further 8-bit digits (rare: needs > 4096 near-equal scores)
    while (nc > 4096 && shift > 0 && need > 0) {
        for (int i = tid; i < 256; i += 1024) h2[i] = 0;
        __syncthreads();
        const int sh = 20 - (shift <= 4 ? 20 : (12 - shift + 8));
        (void)sh;
        const int lo = 20 - (12 - shift) - 8;  // bit position of this digit
        for (int i = tid; i < min(ctl->ncand, 1 << 30); i += 1024) {
            const uint32_t key = ckey[i];
            if ((key & mask) == want) atomicAdd(&h2[(key >> lo) & 255], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t acc = 0;
            for (int b = 255; b >= 0; b--) {
                if (acc + h2[b] >= (uint32_t)need) { thr_bin = b; thr_need = need - acc; break; }
                acc += h2[b];
            }
        }
        __syncthreads();
        // everything above the digit bin is selected
        for (int i = tid; i < min(ctl->ncand, 1 << 30); i += 1024) {
            const uint32_t key = ckey[i];
            if ((key & mask) == want && (int)((key >> lo) & 255) > thr_bin) idx[nsel + atomicAdd(&taken, 1)] = cand[i];
        }
        __syncthreads();
        mask |= 0xFFu << lo;
        want |= (uint32_t)thr_bin << lo;
        need = thr_need;
        nc = h2[thr_bin];
        shift -= 8;
        __syncthreads();
    }
    // gather remaining candidates (matching the narrowed prefix) into buf as (key << 32 | ~idx) for a descending sort
    __shared__ int nb;
    if (tid == 0) nb = 0;
    __syncthreads();
    const int ncand = min(ctl->ncand, 1 << 30);
    for (int i = tid; i < ncand; i += 1024) {
        const uint32_t key = ckey[i];
        if ((key & mask) == want) {
            const int p = atomicAdd(&nb, 1);
            if (p < 4096) buf[p] = (uint64_t)key << 32 | (uint32_t)(~cand[i]);
        }
    }
    __syncthreads();
    const int m = min(nb, 4096);
    int P = 1;
    while (P < m) P <<= 1;
    for (int i = m + tid; i < P; i += 1024) buf[i] = 0;
    __syncthreads();
    // bitonic sort descending
    for (int sz = 2; sz <= P; sz <<= 1)
        for (int st2 = sz >> 1; st2 > 0; st2 >>= 1) {
            for (int i = tid; i < P; i += 1024) {
                const int j = i ^ st2;
                if (j > i) {
                    const bool desc = (i & sz) == 0;
                    const uint64_t a = buf[i], b = buf[j];
                    if (desc ? a < b : a > b) { buf[i] = b; buf[j] = a; }
                }
            }
            __syncthreads();
        }
    const int base = nsel + taken;
    for (int i = tid; i < need && i < m; i += 1024) idx[base + i] = (int)(~(uint32_t)(buf[i] & 0xFFFFFFFFu));
    __syncthreads();
    // sort the k selected indices ascending (bitonic over P2 >= k)
    int P2 = 1;
    while (P2 < k) P2 <<= 1;
    for (int i = k + tid; i < P2; i += 1024) idx[i] = 0x7FFFFFFF;
    __syncthreads();
    for (int sz = 2; sz <= P2; sz <<= 1)
        for (int st2 = sz >> 1; st2 > 0; st2 >>= 1) {
            for (int i = tid; i < P2; i += 1024) {
                const int j = i ^ st2;
                if (j > i) {
                    const bool asc = (i & sz) == 0;
                    const int a = idx[i], b = idx[j];
                    if (asc ? a > b : a < b) { idx[i] = b; idx[j] = a; }
                }
            }
            __syncthreads();
        }
    for (int i = tid; i < kmax; i += 1024) out[i] = i < k ? (map ? map[idx[i]] : idx[i]) + offset : -1;
    if (tid == 0) *nout = k;
}

// Block scores for the candidate layer: bscore[b] = max over 8 positions; the newest block is pinned (+inf).
extern "C" __global__ void block_max8(const State* st, int ratio, int bs, const float* sc, float* bsc, int* nb_out) {
    const int end = (st->pos + 1) / ratio;
    const int nb = (end + bs - 1) / bs;
    if (blockIdx.x == 0 && threadIdx.x == 0) *nb_out = nb;
    for (int b = blockIdx.x * 256 + threadIdx.x; b < nb; b += gridDim.x * 256) {
        float m = -INFINITY;
        for (int i = b * bs; i < min((b + 1) * bs, end); i++) m = fmaxf(m, sc[i]);
        bsc[b] = (b == (end - 1) / bs) ? INFINITY : m;
    }
}

// Expand kept blocks (ascending block ids, count *nkb) into positions: rows[i], *nrows
extern "C" __global__ void expand_blocks(const State* st, int ratio, int bs, const int* kb, const int* nkb, int* rows, int* nrows) {
    const int end = (st->pos + 1) / ratio;
    const int n = *nkb;
    // block ids are ascending; all blocks are full except possibly the last one
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n * bs; i += gridDim.x * 256) {
        const int b = kb[i / bs], p = b * bs + i % bs;
        rows[i] = p < end ? p : -1;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // the last kept block may be partial: count valid rows
        const int lastb = n > 0 ? kb[n - 1] : 0;
        const int full = n > 0 ? (n - 1) * bs + min(bs, end - lastb * bs) : 0;
        *nrows = full;
    }
}

// ------------------------------------------------------------------ sparse attention

// Partial attention over a chunk of rows: block (chunk c, head group g of 8 heads). Rows: window ring slots (oldest
// first, only those <= pos) then compressed rows listed in cidx (>= 0, minus `woff`). Writes per (chunk, head):
// m, l and o[512] (unnormalized).
#define ACH 32
extern "C" __global__ void __launch_bounds__(256) attn_part(const State* st, const float* __restrict__ q, const uint16_t* __restrict__ ring,
                                                 const uint16_t* __restrict__ comp, const int* __restrict__ cidx, const int* ncomp,
                                                 int woff, float scale, float* part) {
    __shared__ uint16_t rows[ACH][HD + 8];
    __shared__ float qs[8][HD];
    __shared__ float p[8][ACH];
    __shared__ int rowsrc[ACH];
    const int t = st->pos;
    const int nwin = min(t + 1, WIN);
    const int nc = comp ? *ncomp : 0;
    const int total = nwin + nc;
    const int c = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const int r0 = c * ACH;
    float* outp = part + ((size_t)c * NH + g * 8) * (HD + 2);
    if (r0 >= total) {
        // empty chunk: m = -inf
        if (tid < 8) { outp[tid * (HD + 2)] = -INFINITY; outp[tid * (HD + 2) + 1] = 0.f; }
        return;
    }
    // row sources: window slot ids (ring order oldest first) or compressed rows
    if (tid < ACH) {
        const int r = r0 + tid;
        int src = -1;
        if (r < nwin) {
            const int oldest = (t + 1 > WIN) ? (t + 1) % WIN : 0;
            src = (oldest + r) % WIN;
        } else if (r < total) {
            const int ci = cidx[r - nwin];
            src = ci >= 0 ? 0x40000000 | (ci - woff) : -1;
        }
        rowsrc[tid] = src;
    }
    for (int i = tid; i < 8 * HD; i += 256) qs[i / HD][i % HD] = q[(g * 8 + i / HD) * HD + i % HD];
    __syncthreads();
    for (int i = tid; i < ACH * HD / 4; i += 256) {
        const int r = i / (HD / 4), j = (i % (HD / 4)) * 4;
        const int src = rowsrc[r];
        uint2 v = {0, 0};
        if (src >= 0) {
            const uint16_t* base = (src & 0x40000000) ? comp + (size_t)(src & 0x3FFFFFFF) * HD : ring + (size_t)src * HD;
            v = *reinterpret_cast<const uint2*>(base + j);
        }
        *reinterpret_cast<uint2*>(&rows[r][j]) = v;
    }
    __syncthreads();
    // scores: 8 heads x 32 rows = 256 dots of 512
    {
        const int h = tid >> 5, r = tid & 31;
        float s = 0;
        for (int j = 0; j < HD; j++) s += qs[h][j] * bf16_to_f32(rows[r][j]);
        p[h][r] = (rowsrc[r] >= 0 && r0 + r < total) ? s * scale : -INFINITY;
    }
    __syncthreads();
    // per head: max and exp
    {
        const int h = tid >> 5, r = tid & 31;
        const float v = p[h][r];
        const float m = group_max<32>(v);
        const float e = v == -INFINITY ? 0.f : expf(v - m);
        const float l = group_sum<32>(e);
        p[h][r] = e;
        if (r == 0) { outp[h * (HD + 2)] = m; outp[h * (HD + 2) + 1] = l; }
    }
    __syncthreads();
    // o[h][j] = sum_r p[h][r] row[r][j]: 8 x 512 outputs, 16 per thread
    for (int i = tid; i < 8 * HD; i += 256) {
        const int h = i / HD, j = i % HD;
        float s = 0;
        for (int r = 0; r < ACH; r++) s += p[h][r] * bf16_to_f32(rows[r][j]);
        outp[h * (HD + 2) + 2 + j] = s;
    }
}

// Merge chunks per head (+ sink), round to bf16, inverse RoPE on the last 64 dims, fake_fp8 per 32 and write the
// FP8 codes + scales for wo_a. One block per head, 512 threads.
extern "C" __global__ void __launch_bounds__(512) attn_merge(const State* st, const float* part, int nchunks, const float* sinks,
                                                  const float* rope_f, uint8_t* oq, float* os) {
    const int h = blockIdx.x, j = threadIdx.x;
    float M = -INFINITY;
    for (int c = 0; c < nchunks; c++) M = fmaxf(M, part[((size_t)c * NH + h) * (HD + 2)]);
    float L = expf(sinks[h] - M), O = 0;
    for (int c = 0; c < nchunks; c++) {
        const float* pp = part + ((size_t)c * NH + h) * (HD + 2);
        const float m = pp[0];
        if (m == -INFINITY) continue;
        const float w = expf(m - M);
        L += pp[1] * w;
        O += pp[2 + j] * w;
    }
    float v = round_bf16(O / L);
    const float other = __shfl_xor(v, 1);
    if (j >= HD - RD) {
        const int p = (j - (HD - RD)) / 2;
        const float ang = (float)st->pos * rope_f[p];
        const float s = -sinf(ang), c = cosf(ang);
        v = (j & 1) ? other * s + v * c : v * c - other * s;
    }
    // quantize to FP8 per 32 (wo_a's act_quant): 32-lane groups
    const float a = group_max<32>(fabsf(v));
    const float s = pow2_ceil(fmaxf(a, 1e-4f) / 448.f);
    const uint32_t code = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(v * (0.5f / s), 0.f, 0, false) & 0xFF;
    oq[h * HD + j] = (uint8_t)code;
    if ((j & 31) == 0) os[(h * HD + j) / 32] = 2.f * s;
}

// argmax over the logits (one block of 1024): writes the token id into st->token and out[step]
extern "C" __global__ void __launch_bounds__(1024) argmax(const float* logits, int n, State* st, int* out) {
    __shared__ float bv[16];
    __shared__ int bi[16];
    float best = -INFINITY;
    int besti = 0;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float v = logits[i];
        if (v > best) { best = v; besti = i; }
    }
    for (int o = 32; o > 0; o >>= 1) {
        const float ov = __shfl_xor(best, o);
        const int oi = __shfl_xor(besti, o);
        if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
    }
    if ((threadIdx.x & 63) == 0) { bv[threadIdx.x >> 6] = best; bi[threadIdx.x >> 6] = besti; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 16; w++)
            if (bv[w] > best || (bv[w] == best && bi[w] < besti)) { best = bv[w]; besti = bi[w]; }
        bv[0] = best;
        bi[0] = besti;
        out[0] = besti;
    }
}

// position bookkeeping at the end of a step: pos += 1, token = next
extern "C" __global__ void advance(State* st, const int* next) {
    st->pos += 1;
    st->token = next[0];
}

// ------------------------------------------------------------------ load-time re-quantization of MXFP4 experts

DEV float e2m1_val(uint32_t c) {
    const float m[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float v = m[c & 7];
    return (c & 8) ? -v : v;
}
DEV float fnuz_round_pos(float x, uint32_t* code) {
    const uint32_t c = (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(x, 0.f, 0, false) & 0xFF;
    *code = c;
    return __builtin_amdgcn_cvt_f32_fp8(c, 0);
}

// One thread per (row, 32-group): choose the e4m3 scale (relative to `mul`) minimizing squared error over a search of
// 24 candidates, then nearest codebook entries. cb: nl sorted levels (FNUZ-representable values).
extern "C" __global__ void cb_quant(const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, int n, int k,
                                    const float* cb, int nl, float mul, uint8_t* codes, uint8_t* sbytes) {
    const int gi = blockIdx.x * blockDim.x + threadIdx.x;
    const int G = k / 32;
    if (gi >= n * G) return;
    const int r = gi / G, g = gi % G;
    float v[32];
    const float s = __uint_as_float((uint32_t)sc[(size_t)r * G + g] << 23);
    float amax = 0;
    for (int i = 0; i < 16; i++) {
        const uint8_t b = w[(size_t)r * (k / 2) + g * 16 + i];
        v[2 * i] = e2m1_val(b & 15) * s;
        v[2 * i + 1] = e2m1_val(b >> 4) * s;
        amax = fmaxf(amax, fmaxf(fabsf(v[2 * i]), fabsf(v[2 * i + 1])));
    }
    float cbl[8], mid[7], cmax = 0;
    for (int c = 0; c < nl; c++) { cbl[c] = cb[c]; cmax = fmaxf(cmax, fabsf(cb[c])); }
    for (int c = 0; c + 1 < nl; c++) mid[c] = 0.5f * (cbl[c] + cbl[c + 1]);
    float best = INFINITY;
    uint32_t bcode = 0;
    float ba = 0;
    for (int f = 0; f < 24; f++) {
        const float fac = 0.5f + f * (0.8f / 23.f);
        uint32_t code;
        const float a = fnuz_round_pos(amax * fac / cmax / mul, &code) * mul;
        if (a <= 0.f) continue;
        const float ia = 1.f / a;
        float err = 0;
        for (int i = 0; i < 32; i++) {
            const float x = v[i] * ia;
            int q = 0;
            for (int c = 0; c + 1 < nl; c++) q += x > mid[c];
            const float e = v[i] - a * cbl[q];
            err += e * e;
        }
        if (err < best) { best = err; bcode = code; ba = a; }
    }
    if (amax == 0.f) { ba = mul; bcode = 0; }
    sbytes[(size_t)r * G + g] = (uint8_t)bcode;
    const float ia = ba > 0 ? 1.f / ba : 0.f;
    for (int i = 0; i < 32; i++) {
        const float x = v[i] * ia;
        int q = 0;
        for (int c = 0; c + 1 < nl; c++) q += x > mid[c];
        codes[(size_t)r * k + g * 32 + i] = (uint8_t)q;
    }
}

// Pack codes [n][k] + scale bytes [n][k/32] into the codebook tile layout (see wave_cb). One thread per
// (rowtile, k128, lane); lanes < 16 also write the row's 4 scale bytes.
extern "C" __global__ void cb_pack(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ sbytes, int n, int k, int bits,
                                   uint32_t* ow, uint8_t* os) {
    const int KU = k / 128;
    const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)(n / 16) * KU * 64;
    if (gid >= total) return;
    const int l = gid % 64, u = (gid / 64) % KU, rt = gid / (64 * KU);
    const int r = rt * 16 + l % 16, g = l / 16;
    auto c = [&](int m, int j) -> uint32_t { return codes[(size_t)r * k + (4 * u + m) * 32 + 8 * g + j]; };
    uint32_t d[3] = {0, 0, 0};
    for (int p = 0; p < 2; p++)
        for (int b = 0; b < 4; b++) {
            const uint32_t v = (c(2 * p, b) & 3) | (c(2 * p, 4 + b) & 3) << 2 | (c(2 * p + 1, b) & 3) << 4 | (c(2 * p + 1, 4 + b) & 3) << 6;
            d[p] |= v << (8 * b);
        }
    if (bits == 3)
        for (int b = 0; b < 4; b++)
            for (int m = 0; m < 4; m++)
                for (int h = 0; h < 2; h++) d[2] |= ((c(m, 4 * h + b) >> 2) & 1) << (8 * b + 2 * m + h);
    uint32_t* o = ow + gid * bits;
    for (int q = 0; q < bits; q++) o[q] = d[q];
    if (l < 16) {
        for (int m = 0; m < 4; m++) os[(((size_t)rt * KU + u) * 16 + l) * 4 + m] = sbytes[(size_t)(rt * 16 + l) * (k / 32) + 4 * u + m];
    }
}

// rows[i] -> dst[i] = src[rows[i]] for i < *n
extern "C" __global__ void gather_f(const float* src, const int* rows, const int* n, float* dst) {
    const int m = *n;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < m; i += gridDim.x * 256) {
        const int r = rows[i];
        dst[i] = r >= 0 ? src[r] : -INFINITY;
    }
}

// benchmark fill: pseudo-random bf16 values in [-1, 1)
extern "C" __global__ void fill_rand(uint16_t* p, int n, int seed) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        uint32_t x = (uint32_t)i * 2654435761u ^ (uint32_t)(seed * 40503 + 17);
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        const float v = (float)(x & 0xFFFF) / 32768.f - 1.f;
        p[i] = (uint16_t)bf16_bits(v);
    }
}

extern "C" __global__ void test_route(const float* logits, const float* bias, float scale, Route* r) {
    route_top6(logits, bias, scale, r, threadIdx.x);
}
