// batch.h: batched decode - B tokens of B independent sequences ("slots") per step. The prompt path's per-token
// kernels (norms, hyper-connections, GEMMs, routing) take the batch as a chunk unchanged; these are the kernels whose
// prompt versions assume one sequence (positions p0 + t, one set of state buffers): here each token t has its own
// position bpos[t] and slot bslot[t], and every state buffer is a slab of per-slot regions at a fixed stride
// (elements). Windows are read straight from each slot's ring: kv_b writes the token's row at pos % WIN first.
#pragma once

// RoPE (last RD dims of each head) at each token's own position
extern "C" __global__ void rope_b(float* x, int heads, int hd, const float* freqs, const int* bpos, int inverse) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;  // 32 pairs
    float* v = x + ((size_t)t * heads + h) * hd + hd - RD;
    const float ang = (float)bpos[t] * freqs[i];
    float sn = sinf(ang), cs = cosf(ang);
    if (inverse) sn = -sn;
    const float a = v[2 * i], b = v[2 * i + 1];
    v[2 * i] = a * cs - b * sn;
    v[2 * i + 1] = a * sn + b * cs;
}

// window kv row of each token: rmsnorm, rope(pos), fake_fp8 -> its slot's ring at pos % WIN (bf16)
extern "C" __global__ void __launch_bounds__(512) kv_b(const float* kv_raw, const float* kv_norm, const float* rf, const int* bpos,
                                                       const int* bslot, uint16_t* rings, long long ring_stride) {
    __shared__ float red[16];
    const int t = blockIdx.x, i = threadIdx.x, pos = bpos[t];
    float v = kv_raw[(size_t)t * HD + i];
    const float ss = block_sum(v * v, red);
    v = kv_norm[i] * (v * (1.f / sqrtf(ss / HD + 1e-20f)));
    const float other = __shfl_xor(v, 1);
    if (i >= HD - RD) {
        const int p = (i - (HD - RD)) / 2;
        const float ang = (float)pos * rf[p];
        const float sn = sinf(ang), cs = cosf(ang);
        v = (i & 1) ? other * sn + v * cs : v * cs - other * sn;
    }
    v = fake_fp8_lane(v);
    rings[(size_t)bslot[t] * ring_stride + (size_t)(pos % WIN) * HD + i] = (uint16_t)bf16_bits(v);
}

// Compressor + indexer keys, one token of its own sequence per block (comp_t with T = 1): the previous position's
// latent comes from the slot's cstate (ratio 2), the compressed row and index key go to the slot's caches.
extern "C" __global__ void __launch_bounds__(512) comp_b(const int* bpos, const int* bslot, int ratio, const float* ckv, const float* cgate,
                                                         float* cstate, long long cs_stride, const float* comp_norm, const uint16_t* wk,
                                                         const float* knorm, uint16_t* comp_cache, long long comp_stride, uint16_t* idx_cache,
                                                         long long idx_stride, const float* rope_fc, float eps) {
    __shared__ float red[16];
    __shared__ float lat[HD];
    __shared__ float kk[ID];
    const int t = blockIdx.x, i = threadIdx.x, p = bpos[t], s = bslot[t];
    float* cst = cstate + (size_t)s * cs_stride;
    if (ratio == 2 && (p + 1) % 2 != 0) {
        cst[i] = ckv[(size_t)t * HD + i];
        cst[HD + i] = cgate[(size_t)t * HD + i];
    }
    if ((p + 1) % ratio != 0) return;
    float v;
    if (ratio == 1) {
        v = ckv[(size_t)t * HD + i];
    } else {
        const float k0 = cst[i], s0 = cst[HD + i];
        const float k1 = ckv[(size_t)t * HD + i], s1 = cgate[(size_t)t * HD + i];
        const float mx = fmaxf(s0, s1);
        const float e0 = expf(s0 - mx), e1 = expf(s1 - mx);
        const float den = e0 + e1;
        v = round_bf16(k0 * e0 / den + k1 * e1 / den);
    }
    const float ss = block_sum(v * v, red);
    lat[i] = comp_norm[i] * (v * (1.f / sqrtf(ss / HD + eps)));
    __syncthreads();
    const int n = (p + 1) / ratio - 1, cpos = p + 1 - ratio;
    {
        const int o = i >> 2, q = i & 3;
        float sacc = 0;
        for (int j = q; j < HD; j += 4) sacc += bf16_to_f32(wk[o * HD + j]) * lat[j];
        sacc += __shfl_xor(sacc, 1);
        sacc += __shfl_xor(sacc, 2);
        if (q == 0) kk[o] = sacc;
    }
    __syncthreads();
    if (i < ID) {
        const float kv = kk[i];
        const float ssum = wave_sum(kv * kv);
        if ((i & 63) == 0) red[8 + (i >> 6)] = ssum;
    }
    __syncthreads();
    if (i < ID) {
        const float sk = red[8] + red[9];
        float x = knorm[i] * (kk[i] * (1.f / sqrtf(sk / ID + eps)));
        const float other = __shfl_xor(x, 1);
        if (i >= ID - RD) {
            const int pp = (i - (ID - RD)) / 2;
            const float ang = (float)cpos * rope_fc[pp];
            const float sn = sinf(ang), cs = cosf(ang);
            x = (i & 1) ? other * sn + x * cs : x * cs - other * sn;
        }
        x = fake_fp4_lane<32>(x, false);
        idx_cache[(size_t)s * idx_stride + (size_t)n * ID + i] = (uint16_t)bf16_bits(x);
    }
    {
        float x = lat[i];
        const float other = __shfl_xor(x, 1);
        if (i >= HD - RD) {
            const int pp = (i - (HD - RD)) / 2;
            const float ang = (float)cpos * rope_fc[pp];
            const float sn = sinf(ang), cs = cosf(ang);
            x = (i & 1) ? other * sn + x * cs : x * cs - other * sn;
        }
        x = fake_fp4_lane<16>(x, true);
        comp_cache[(size_t)s * comp_stride + (size_t)n * HD + i] = (uint16_t)bf16_bits(x);
    }
}

// Indexer scores, block (key chunk, token q) over the token's slot's index keys: score[q][s] for s < (pos+1)/ratio
// (or its candidate rows rows[q][..nrows[q]])
extern "C" __global__ void __launch_bounds__(256) iscore_b(const int* bpos, const int* bslot, int ratio, const float* iq, const float* iw,
                                                           float wscale, const uint16_t* __restrict__ kcs, long long kc_stride,
                                                           const int* rows, const int* nrows, int ldrows, float* score, int lds) {
    __shared__ uint16_t qs[IH][ID + 8];
    __shared__ float ws[IH];
    const int tq = blockIdx.y;
    const int end = rows ? nrows[tq] : (bpos[tq] + 1) / ratio;
    const uint16_t* kc = kcs + (size_t)bslot[tq] * kc_stride;
    const int tid = threadIdx.x;
    const int chunk0 = blockIdx.x * 1024;
    if (chunk0 >= end) return;
    for (int i = tid; i < IH * ID; i += 256) {
        const float v = fake_fp4_lane<32>(iq[(size_t)tq * IH * ID + i], false);
        qs[i / ID][i % ID] = (uint16_t)bf16_bits(v);
    }
    if (tid < IH) ws[tid] = round_bf16(iw[(size_t)tq * IH + tid] * wscale);
    __syncthreads();
    const int lane = tid & 63, g = lane >> 4, c = lane & 15;
    const int* rw = rows ? rows + (size_t)tq * ldrows : nullptr;
    float* out = score + (size_t)tq * lds;
    for (int base = chunk0 + (tid >> 6) * 16; base < min(end, chunk0 + 1024); base += 64) {
        const int sidx = base + c;
        const int row = sidx < end ? (rw ? rw[sidx] : sidx) : -1;
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
        float pj[4];
        for (int j = 0; j < 4; j++) pj[j] = group_sum<16>(fmaxf(acc0[j], 0.f) * ws[c] + fmaxf(acc1[j], 0.f) * ws[16 + c]);
        if (c == 0)
            for (int j = 0; j < 4; j++) {
                const int sj = base + 4 * g + j;
                if (sj < end) out[sj] = (rw && rw[sj] < 0) ? -INFINITY : pj[j];
            }
    }
}

// top-k per token (block = token): n = (pos+1)/ratio, or nrows[q] with map rows[q][..]
extern "C" __global__ void __launch_bounds__(1024) itopk_b(const int* bpos, int ratio, const float* score, int lds, const int* rows,
                                                           const int* nrows, int ldrows, int k, int offset, int* out, int ldo,
                                                           int* nout, int sorted) {
    const int tq = blockIdx.x;
    const int n = rows ? nrows[tq] : (bpos[tq] + 1) / ratio;
    block_topk(score + (size_t)tq * lds, n, k, rows ? rows + (size_t)tq * ldrows : nullptr, offset, out + (size_t)tq * ldo, k, nout + tq, sorted != 0);
}

// candidate layer per token: block maxima over bs positions (newest block pinned), top nblk blocks, expanded
extern "C" __global__ void __launch_bounds__(1024) icand_b(const int* bpos, int ratio, int bs, int nblk, const float* score, int lds,
                                                           float* bsc, int* kb, int* keep, int* nkeep) {
    __shared__ int nb_s;
    const int tq = blockIdx.x;
    const int end = (bpos[tq] + 1) / ratio;
    const int nb = (end + bs - 1) / bs;
    const float* sc = score + (size_t)tq * lds;
    float* b = bsc + (size_t)tq * (lds / bs + 1);
    for (int i = threadIdx.x; i < nb; i += 1024) {
        float m = -INFINITY;
        for (int j = i * bs; j < min((i + 1) * bs, end); j++) m = fmaxf(m, sc[j]);
        b[i] = (i == (end - 1) / bs) ? INFINITY : m;
    }
    __syncthreads();
    int* kbq = kb + (size_t)tq * nblk;
    block_topk(b, nb, nblk, nullptr, 0, kbq, nblk, &nb_s, true);
    const int n = nb_s;
    int* kq = keep + (size_t)tq * nblk * bs;
    for (int i = threadIdx.x; i < n * bs; i += 1024) {
        const int blk = kbq[i / bs], pp = blk * bs + i % bs;
        kq[i] = pp < end ? pp : -1;
    }
    if (threadIdx.x == 0) nkeep[tq] = n * bs;
}

// Sparse attention, block per token: the window from its slot's ring (kv_b wrote pos first), compressed rows from its
// slot's comp cache (cidx[t][..ncomp[t]] - woff)
extern "C" __global__ void __launch_bounds__(1024) attn_b(const int* bpos, const int* bslot, float* __restrict__ q,
                                                          const uint16_t* __restrict__ rings, long long ring_stride,
                                                          const uint16_t* __restrict__ comps, long long comp_stride,
                                                          const int* __restrict__ cidx, const int* ncomp, int ldc, int woff,
                                                          float scale, const float* sinks) {
    const int t = blockIdx.x, pos = bpos[t], s = bslot[t];
    const int lo = max(0, pos - (WIN - 1));
    attn_core(t, lo, pos - lo + 1, 0, true, q, rings + (size_t)s * ring_stride, comps ? comps + (size_t)s * comp_stride : nullptr,
              cidx, comps ? ncomp[t] : 0, ldc, woff, scale, sinks);
}

// after attn_b: inverse RoPE at each token's position, FP8 per 32 (+ bf16) for wo_a (attn_m_out per token)
extern "C" __global__ void __launch_bounds__(256) attn_out_b(const int* bpos, float* __restrict__ o, const float* rf, uint8_t* oq, float* os,
                                                             uint16_t* obq) {
    const int t = blockIdx.x, pos = bpos[t], h0 = blockIdx.y * 8;
    float* ot = o + (size_t)t * NH * HD + (size_t)h0 * HD;
    __shared__ float cs[32], sn[32];
    if (threadIdx.x < 32) {
        const float ang = (float)pos * rf[threadIdx.x];
        cs[threadIdx.x] = cosf(ang);
        sn[threadIdx.x] = -sinf(ang);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 8 * 32; i += 256) {
        const int h = i / 32, pi = i % 32;
        float* v = ot + h * HD + HD - RD + 2 * pi;
        const float a = v[0], b = v[1];
        v[0] = a * cs[pi] - b * sn[pi];
        v[1] = a * sn[pi] + b * cs[pi];
    }
    __syncthreads();
    const size_t off = (size_t)t * NH * HD + (size_t)h0 * HD;
    write_act(8 * HD, [&](int i) -> float { return ot[i]; }, oq + off, os + off / 32, nullptr, obq + off);
}

// greedy token of each of the batch's logit rows (block per row; ties to the lowest index, as argmax)
extern "C" __global__ void __launch_bounds__(1024) argmax_b(const float* logits, int n, int ld, int* out) {
    __shared__ float bv[16];
    __shared__ int bi[16];
    const float* lg = logits + (size_t)blockIdx.x * ld;
    float best = -INFINITY;
    int besti = 0x7FFFFFFF;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const float v = lg[i];
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
        out[blockIdx.x] = besti;
    }
}
