// dequant_bench.cu - prefill's IQ -> f16 weight expansion (dsx.cu dequant_f16 / rp_weight: one thread per weight)
// next to dequant_f16_iq8 (one thread per 8-weight codebook group), GGUF and repacked layouts, for the codebook
// formats of Unsloth's UD-IQ1_M experts, on one gate matrix [2304 x 5120]. Checks that all give identical output.
//
// build: nvcc -O3 -arch=sm_86 -I nvidia/src/kernels -o dequant_bench bench/dequant_bench.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstring>
#include <cuda_fp16.h>
#include "dequant_iq8.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

__device__ __forceinline__ float ld_h(const uint8_t *p) { return __half2float(__ushort_as_half((unsigned short)(p[0] | p[1] << 8))); }

// as in dsx.cu: one thread per weight
__global__ void dq_old(int t, const uint8_t *w, __half *out, int rows, int k) {
  int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;
  float v;
  if (t == 16) {
    const uint8_t *blk = w + (size_t)r * (k / 256) * 66 + (c / 256) * 66;
    v = iq2_value(blk + 2 + ((c & 255) >> 5) * 8, ld_h(blk), c & 31);
  } else if (t == 19) {
    v = iq1s_value(w + (size_t)r * (k / 256) * 50 + (c / 256) * 50, c & 255);
  } else if (t == 17) {
    v = iq2xs_value(w + (size_t)r * (k / 256) * 74 + (c / 256) * 74, c & 255);
  } else {
    v = iq2s_value(w + (size_t)r * (k / 256) * 82 + (c / 256) * 82, c & 255);
  }
  out[(size_t)r * k + c] = __float2half(v);
}

// as rp_weight (expert_fast.cuh) for repacked IQ1_S / IQ2_XS / IQ2_S, and RP_IQ2 for IQ2_XXS
__global__ void dq_old_rp(int t, const uint8_t *w, __half *out, int rows, int k) {
  int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= k) return;
  float v;
  size_t nb = k / 256;
  if (t == 16) {
    size_t b = c / 256;
    const uint8_t *p = w + ((size_t)r * nb + b) * 64 + ((c & 255) >> 5) * 8;
    v = iq2_value(p, ld_h(w + rows * nb * 64 + ((size_t)r * nb + b) * 2), c & 31);
  } else {
    const int bs = t == 19 ? 50 : t == 17 ? 74 : 82, body = t == 19 ? 48 : 64, tail = bs - 2 - body;
    size_t b = (size_t)r * nb + c / 256;
    uint8_t blk[82];
    blk[0] = w[rows * nb * (body + tail) + b * 2];
    blk[1] = w[rows * nb * (body + tail) + b * 2 + 1];
    for (int j = 0; j < body; j++) blk[2 + j] = w[b * body + j];
    for (int j = 0; j < tail; j++) blk[2 + body + j] = w[rows * nb * body + b * tail + j];
    v = t == 19 ? iq1s_value(blk, c & 255) : t == 17 ? iq2xs_value(blk, c & 255) : iq2s_value(blk, c & 255);
  }
  out[(size_t)r * k + c] = __float2half(v);
}

int main() {
  const int rows = 2304, k = 5120, reps = 200;
  struct F { int t, bytes; const char *name; } fs[] = {{16, 66, "IQ2_XXS"}, {19, 50, "IQ1_S"}, {17, 74, "IQ2_XS"}, {22, 82, "IQ2_S"}};
  __half *o1, *o2, *o3, *o4;
  CK(cudaMalloc(&o1, (size_t)rows * k * 2));
  CK(cudaMalloc(&o2, (size_t)rows * k * 2));
  CK(cudaMalloc(&o3, (size_t)rows * k * 2));
  CK(cudaMalloc(&o4, (size_t)rows * k * 2));
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (auto f : fs) {
    const size_t nb = (size_t)rows * (k / 256) * f.bytes;
    std::vector<uint8_t> hw(nb);
    uint64_t z = 88172645463325252ull;
    for (auto &v : hw) { z ^= z << 13; z ^= z >> 7; z ^= z << 17; v = (uint8_t)z; }
    for (size_t i = 0; i < nb; i += f.bytes) hw[i] = 0x1e, hw[i + 1] = 0x21;  // d = fp16 0.0101
    uint8_t *dw;
    CK(cudaMalloc(&dw, nb));
    CK(cudaMemcpy(dw, hw.data(), nb, cudaMemcpyHostToDevice));
    // the same blocks repacked as repack_expert stores them: bodies, then tails, then d
    const int body = f.t == 19 ? 48 : 64, tail = f.bytes - 2 - body;
    const size_t nbt = (size_t)rows * (k / 256);
    std::vector<uint8_t> hr(nbt * f.bytes);
    for (size_t b = 0; b < nbt; b++) {
      const uint8_t *blk = hw.data() + b * f.bytes;
      memcpy(hr.data() + b * body, blk + 2, body);
      memcpy(hr.data() + nbt * body + b * tail, blk + 2 + body, tail);
      memcpy(hr.data() + nbt * (body + tail) + b * 2, blk, 2);
    }
    uint8_t *dr;
    CK(cudaMalloc(&dr, nb));
    CK(cudaMemcpy(dr, hr.data(), nb, cudaMemcpyHostToDevice));
    dim3 g1((k + 255) / 256, rows), g8((k / 8 + 255) / 256, rows);
    float ms[4];
    __half *outs[4] = {o1, o2, o3, o4};
    for (int pass = 0; pass < 2; pass++)
      for (int v = 0; v < 4; v++) {
        cudaEventRecord(a);
        for (int i = 0; i < reps; i++) {
          if (v == 0) dq_old<<<g1, 256>>>(f.t, dw, outs[v], rows, k);
          if (v == 1) dq_old_rp<<<g1, 256>>>(f.t, dr, outs[v], rows, k);
          if (v == 2) dequant_f16_iq8<<<g8, 256>>>(f.t, dw, nullptr, outs[v], rows, k);
          if (v == 3) dequant_f16_iq8<<<g8, 256>>>(f.t + 2000, dr, nullptr, outs[v], rows, k);
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms[v], a, b);
      }
    std::vector<uint16_t> h0((size_t)rows * k), hx((size_t)rows * k);
    CK(cudaMemcpy(h0.data(), o1, h0.size() * 2, cudaMemcpyDeviceToHost));
    size_t diff[4] = {0, 0, 0, 0};
    for (int v = 1; v < 4; v++) {
      CK(cudaMemcpy(hx.data(), outs[v], hx.size() * 2, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < h0.size(); i++) diff[v] += h0[i] != hx[i];
    }
    printf("%-8s per weight: GGUF %6.1f us, repacked %6.1f us | 8-wide: GGUF %5.1f us, repacked %5.1f us | outputs differing from the GGUF per-weight: %zu %zu %zu\n",
           f.name, ms[0] * 1e3 / reps, ms[1] * 1e3 / reps, ms[2] * 1e3 / reps, ms[3] * 1e3 / reps, diff[1], diff[2], diff[3]);
    cudaFree(dr);
    cudaFree(dw);
  }
  return 0;
}
