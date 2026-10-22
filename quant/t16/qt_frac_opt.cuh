#pragma once

// Fractional-rate trellis quantizer (KA + 0.5 bits per weight, mul1 / T16 codebook), built like exllamav3's dense
// integer-K kernels (quantize_tiles_optimized.cuh) instead of its generic fractional path. Same Viterbi, same fp16
// arithmetic and tie order as quantize_tiles_frac_kernel, so the indices are bit-identical; what changes:
//   - each thread owns the same 32768 / NT state pairs at every step (state = 2 thread + 2 NT m), whatever the step's
//     out-width, so the decoded codebook values are computed once and stay in registers;
//   - the costs live in shared memory: one buffer per node width (65536 >> KA and 65536 >> (KA + 1) halves), and with
//     alternating widths a step reads one buffer and writes the other, so a step needs one barrier;
//   - the traceback history stores each out-node's chosen branch (KOUT bits, packed by warp ballots: a fixed 4 KB per
//     position, 1 MB per tile) instead of a 16-bit in-node index per out-node at the widest stride (16 MB per tile).
// Needs decode_3inst<2> / decode_3inst_2<2> (codebook.cuh) and H_INF. Launch: NT threads, one tile per block,
// qt_frac_opt_shmem(KA) bytes of dynamic shared memory, hist = 1 MB (QTF_HIST_WORDS u32) per block of the launch.

#define QTF_HIST_POS_WORDS 1024                    // per position: KOUT bits x (65536 >> KOUT) out-nodes <= 32768 bits
#define QTF_HIST_WORDS (256 * QTF_HIST_POS_WORDS)  // per tile (L = 256)

__host__ __device__ constexpr int qt_frac_opt_shmem(int KA)
{
    return 256 * 2 + 32 * 4 + (65536 >> KA) * 2 + (65536 >> (KA + 1)) * 2;
}

template <int KIN, int KOUT, int NT, bool FIRST>
__device__ __forceinline__ void qt_frac_opt_step
(
    const half2 (&values)[32768 / NT],
    const half w,
    const half* __restrict__ cin,
    half* __restrict__ cout,
    uint32_t* __restrict__ hist_pos,
    const int pre_state,
    const bool write_hist,
    const int thread
)
{
    constexpr int EO = 65536 >> KOUT;   // out-nodes
    constexpr int NP = EO / (2 * NT);   // out-node pairs per thread
    constexpr int NC = 1 << KOUT;       // branches per out-node
    constexpr int NW = NT / 32;
    static_assert(NP * NC == 32768 / NT);
    const half2 w2 = __half2half2(w);
    const int lane = thread & 31, warp = thread >> 5;

    #pragma unroll
    for (int j = 0; j < NP; ++j)
    {
        half2 best;
        int b0 = 0, b1 = 0;
        #pragma unroll
        for (int c = 0; c < NC; ++c)
        {
            // state (c << (16 - KOUT)) | out, out = 2 (thread + NT j): values index j + NP c
            const int state = 2 * thread + 2 * NT * (j + NP * c);
            const int pred = state >> KIN;
            const half2 d = __hsub2(values[j + NP * c], w2);
            half2 err;
            if constexpr (FIRST)
            {
                err = __hmul2(d, d);
                if (pre_state >= 0 && pred != pre_state) err = __half2half2(H_INF);
            }
            else
                err = __hfma2(d, d, __half2half2(cin[pred]));
            if (c == 0) best = err;
            else
            {
                if (__hlt(__low2half(err), __low2half(best)))
                {
                    best = __halves2half2(__low2half(err), __high2half(best));
                    b0 = c;
                }
                if (__hlt(__high2half(err), __high2half(best)))
                {
                    best = __halves2half2(__low2half(best), __high2half(err));
                    b1 = c;
                }
            }
        }
        reinterpret_cast<half2*>(cout)[thread + NT * j] = best;
        if (write_hist)
        {
            // bit b of the branch of out-node e: word b * (EO / 32) + (e / 64) * 2 + (e & 1), bit (e / 2) & 31
            #pragma unroll
            for (int bit = 0; bit < KOUT; ++bit)
            {
                const uint32_t a = __ballot_sync(0xffffffffu, (b0 >> bit) & 1);
                const uint32_t b = __ballot_sync(0xffffffffu, (b1 >> bit) & 1);
                if (lane == 0)
                    *reinterpret_cast<uint2*>(hist_pos + bit * (EO / 32) + (warp + NW * j) * 2) = make_uint2(a, b);
            }
        }
    }
}

template <int KA, uint32_t MASK, int NT, int L = 256>
__global__ __launch_bounds__(NT, 1)
void quantize_tiles_frac_opt_kernel
(
    const float* __restrict__ input_tiles_ptr,
    float* __restrict__ output_tiles_ptr,
    uint16_t* __restrict__ output_indices_ptr,
    half* __restrict__ temp_costs_ptr,   // unused: costs live in shared memory
    uint16_t* __restrict__ temp_edges_ptr,
    const half2* __restrict__ lut        // unused
)
{
    static_assert(L == 256, "history and tie order are laid out for 16 x 16 tiles");
    static_assert(((MASK ^ ((MASK >> 1) | (MASK << 15))) & 0xffffu) == 0xffffu,
                  "a step reads one cost buffer and writes the other: widths must alternate");
    constexpr int KB = KA + 1;
    constexpr int NV = 32768 / NT;
    constexpr int NW = NT / 32;
    constexpr int EA = 65536 >> KA;                 // nodes after a KA-bit step
    constexpr int edges_last = 65536 >> (KA + (MASK & 1));
    (void) temp_costs_ptr; (void) lut;

    extern __shared__ uint8_t shbuf[];
    half* sh_input_tile = reinterpret_cast<half*>(shbuf);
    int* sh_idx = reinterpret_cast<int*>(shbuf + L * sizeof(half));
    half* costs_a = reinterpret_cast<half*>(shbuf + L * sizeof(half) + 32 * sizeof(int));  // width 16 - KA
    half* costs_b = costs_a + EA;                                                          // width 16 - KB

    const int tile_idx = blockIdx.x;
    const int thread = threadIdx.x;
    const float* input_tile = input_tiles_ptr + L * tile_idx;
    float* output_tile = output_tiles_ptr + L * tile_idx;
    uint16_t* output_indices = output_indices_ptr + L * tile_idx;
    uint32_t* hist = reinterpret_cast<uint32_t*>(temp_edges_ptr) + (size_t) QTF_HIST_WORDS * tile_idx;

    for (int i = thread; i < L; i += NT) sh_input_tile[i] = __float2half_rn(input_tile[i]);

    half2 values[NV];
    #pragma unroll
    for (int m = 0; m < NV; ++m)
    {
        const int state = 2 * thread + 2 * NT * m;
        values[m] = decode_3inst_2<2>(state, state + 1);
    }
    __syncthreads();

    auto ring = [&](int i, int roll)
    {
        int ri = i + roll;
        if (ri >= L) ri -= L;
        return ri;
    };
    // the cost buffer holding the out-costs of a step whose out-width is 16 - D bits
    auto buf = [&](int d) { return d == KA ? costs_a : costs_b; };

    auto forward = [&](int roll, int pre_state)
    {
        for (int i = 0; i < L; ++i)
        {
            const int ri = ring(i, roll);
            const int din = (MASK >> (ri & 15)) & 1, dout = (MASK >> ((ri + 1) & 15)) & 1;
            // The first pass only traces back to position 0, so its history for the rolled second half is never read
            const bool write_hist = pre_state >= 0 || ri < L / 2;
            uint32_t* hist_pos = hist + QTF_HIST_POS_WORDS * ri;
            const half w = sh_input_tile[ri];
            const half* cin = buf(KA + din);
            half* cout = buf(KA + dout);
            if (i == 0)
            {
                if (din) qt_frac_opt_step<KB, KA, NT, true>(values, w, cin, cout, hist_pos, pre_state, write_hist, thread);
                else qt_frac_opt_step<KA, KB, NT, true>(values, w, cin, cout, hist_pos, pre_state, write_hist, thread);
            }
            else
            {
                if (din) qt_frac_opt_step<KB, KA, NT, false>(values, w, cin, cout, hist_pos, pre_state, write_hist, thread);
                else qt_frac_opt_step<KA, KB, NT, false>(values, w, cin, cout, hist_pos, pre_state, write_hist, thread);
            }
            __syncthreads();
        }
    };

    // Same keys and tie order as quantize_tiles_frac_kernel (rank = the historical 512-thread reduction order)
    auto argmin_cost = [&]()
    {
        const half* costs = buf(KA + (MASK & 1));
        uint32_t best = 0x7c00ffffu;
        for (int e = thread; e < edges_last; e += NT)
        {
            unsigned v = e & 1023;
            unsigned rank = ((__brev(v >> 5) >> 27) << 10) | ((__brev(v & 31) >> 27) << 5) | (e >> 10);
            unsigned key = ((uint32_t) __half_as_ushort(costs[e]) << 16) | rank;
            best = min(best, key);
        }
        #pragma unroll
        for (int offset = 16; offset; offset >>= 1)
            best = min(best, __shfl_xor_sync(0xffffffff, best, offset));
        if ((thread & 31) == 0)
            ((uint32_t*) sh_idx)[thread >> 5] = best;
        __syncthreads();
        if (thread < 32)
        {
            best = thread < NW ? ((uint32_t*) sh_idx)[thread] : 0x7c00ffffu;
            #pragma unroll
            for (int offset = 16; offset; offset >>= 1)
                best = min(best, __shfl_xor_sync(0xffffffff, best, offset));
        }
        unsigned rank = best & 65535;
        unsigned v = ((__brev(rank >> 10) >> 27) << 5) | (__brev((rank >> 5) & 31) >> 27);
        return best >= 0x7c000000u ? 0 : (int) (((rank & 31) << 10) | v);
    };

    auto backward = [&](int roll, bool write, int edge)
    {
        if (thread == 0)
        {
            for (int i = L - 1; i >= 0; --i)
            {
                const int ri = ring(i, roll);
                const int kin = KA + ((MASK >> (ri & 15)) & 1), kout = KA + ((MASK >> ((ri + 1) & 15)) & 1);
                const uint32_t* hp = hist + QTF_HIST_POS_WORDS * ri + (edge >> 6) * 2 + (edge & 1);
                const int sh = (edge >> 1) & 31;
                int c = (hp[0] >> sh) & 1;
                if (kout >= 2) c |= ((hp[(65536 >> kout) / 32] >> sh) & 1) << 1;
                if (kout >= 3) c |= ((hp[2 * ((65536 >> kout) / 32)] >> sh) & 1) << 2;
                if (kout >= 4) c |= ((hp[3 * ((65536 >> kout) / 32)] >> sh) & 1) << 3;
                const int encoded = (c << (16 - kout)) | edge;   // = ((prev_edge << kin) | edge) & 0xffff
                edge = encoded >> kin;
                if (write)
                {
                    output_indices[ri] = (uint16_t) encoded;
                    output_tile[ri] = __half2float(decode_3inst<2>(encoded));
                }
                else if (ri == 0) break;
            }
        }
        if (thread == 0) sh_idx[0] = edge;
        __syncthreads();
        return sh_idx[0];
    };

    forward(L / 2, -1);
    int end_state = backward(L / 2, false, argmin_cost());
    forward(0, end_state);
    backward(0, true, end_state);
}
