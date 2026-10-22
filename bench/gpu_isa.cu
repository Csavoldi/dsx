// gpu_isa.cu - per-SM instruction throughput on this GPU (lanes per clock per SM) for the operations a weight decoder
// can use: the inputs of the decode cost model (docs/t16.md, "Decode cost model"). Each test runs NCH independent
// dependency chains per thread over N iterations, one 1024-thread block per SM; cycles come from clock64 on each SM, so
// the 230 W power cap's clock changes do not matter. Check the SASS (cuobjdump -sass) to see each test is the intended
// instruction: the asm volatile bodies keep the compiler from folding or removing them.
//
// build: nvcc -O3 -arch=sm_86 -o gpu_isa gpu_isa.cu
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)
#define N 4096
#define NCH 8

__device__ unsigned long long g_cyc[1024];
__device__ uint32_t g_sink;

template <int OP>
__global__ void __launch_bounds__(1024) bench(uint32_t seed) {
  __shared__ uint32_t tab[1024];
  __shared__ unsigned long long t0s;
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) tab[i] = i * 2654435761u;
  __syncthreads();
  uint32_t r[NCH];
#pragma unroll
  for (int c = 0; c < NCH; c++) r[c] = seed + threadIdx.x * 7 + c;
  uint32_t k = seed | 1;
  __syncthreads();
  const unsigned long long t0 = clock64();
  for (int it = 0; it < N; it++) {
#pragma unroll
    for (int c = 0; c < NCH; c++) {
      if (OP == 0) asm volatile("shf.r.wrap.b32 %0, %0, %1, %2;" : "+r"(r[c]) : "r"(k), "r"(c + 1));  // funnel shift
      if (OP == 1) asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(r[c]) : "r"(k), "r"(c));
      if (OP == 2) asm volatile("add.u32 %0, %0, %1;" : "+r"(r[c]) : "r"(k));
      if (OP == 3) asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(r[c]) : "r"(k));
      if (OP == 4) asm volatile("mul.lo.u32 %0, %0, %1;" : "+r"(r[c]) : "r"(k));
      if (OP == 5) asm volatile("dp4a.u32.s32 %0, %0, %1, %0;" : "+r"(r[c]) : "r"(k));
      if (OP == 6) asm volatile("fma.rn.f16x2 %0, %0, %1, %0;" : "+r"(r[c]) : "r"(k));
      if (OP == 7) { float f = __uint_as_float(r[c]); asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(f) : "f"(__uint_as_float(k))); r[c] = __float_as_uint(f); }
      if (OP == 8) r[c] = tab[r[c] & 1023];                                           // LDS.32, data-dependent
      if (OP == 9) { uint2 v = ((const uint2 *)tab)[r[c] & 511]; r[c] = v.x ^ v.y; }  // LDS.64
      if (OP == 10) { uint4 v = ((const uint4 *)tab)[r[c] & 255]; r[c] = v.x ^ v.w; }  // LDS.128
      if (OP == 11) asm volatile("shr.u32 %0, %0, %1;" : "+r"(r[c]) : "r"(c + 1));     // plain shift
      if (OP == 12) asm volatile("bfe.u32 %0, %0, %1, 16;" : "+r"(r[c]) : "r"(c + 1));  // bit field extract
    }
    if (OP == 13) {  // IMMA m16n8k32 s8: 4 independent accumulators
      int acc[4][4];
#pragma unroll
      for (int j = 0; j < 4; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = r[j];
#pragma unroll
      for (int rep = 0; rep < 2; rep++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       : "+r"(acc[j][0]), "+r"(acc[j][1]), "+r"(acc[j][2]), "+r"(acc[j][3])
                       : "r"(k), "r"(k), "r"(k), "r"(k), "r"(k), "r"(k));
#pragma unroll
      for (int j = 0; j < 4; j++) r[j] ^= acc[j][0];
    }
    if (OP == 14) {  // HMMA m16n8k16 f16 -> f32
      float acc[4][4];
#pragma unroll
      for (int j = 0; j < 4; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = (float)r[j];
#pragma unroll
      for (int rep = 0; rep < 2; rep++)
#pragma unroll
        for (int j = 0; j < 4; j++)
          asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                       : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                       : "r"(k), "r"(k), "r"(k), "r"(k), "r"(k), "r"(k));
#pragma unroll
      for (int j = 0; j < 4; j++) r[j] ^= __float_as_uint(acc[j][0]);
    }
  }
  const unsigned long long t1 = clock64();
  uint32_t x = 0;
#pragma unroll
  for (int c = 0; c < NCH; c++) x ^= r[c];
  if (x == 0x12345678u) g_sink = x;
  if (threadIdx.x == 0) g_cyc[blockIdx.x] = t1 - t0;
  (void)t0s;
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  const int sms = p.multiProcessorCount, blocks_per_sm = 1, blocks = sms * blocks_per_sm, threads = 1024;  // one block per SM: its cycles are the SM's
  printf("%s, %d SMs; %d blocks x %d threads (%d warps per SM); %d chains x %d iterations per thread\n", p.name, sms, blocks, threads,
         blocks_per_sm * threads / 32, NCH, N);
  const char *names[] = {"SHF.R funnel shift", "LOP3", "IADD", "PRMT byte permute", "IMAD mul.lo", "IDP4A dp4a", "HFMA2", "FFMA", "LDS.32 (random)",
                         "LDS.64 (random)", "LDS.128 (random)", "SHR", "BFE", "IMMA m16n8k32 s8 (per mma)", "HMMA m16n8k16 f16 (per mma)"};
  for (int op = 0; op < 15; op++) {
    void (*k)(uint32_t) = nullptr;
    switch (op) {
      case 0: k = bench<0>; break; case 1: k = bench<1>; break; case 2: k = bench<2>; break; case 3: k = bench<3>; break;
      case 4: k = bench<4>; break; case 5: k = bench<5>; break; case 6: k = bench<6>; break; case 7: k = bench<7>; break;
      case 8: k = bench<8>; break; case 9: k = bench<9>; break; case 10: k = bench<10>; break; case 11: k = bench<11>; break;
      case 12: k = bench<12>; break; case 13: k = bench<13>; break; case 14: k = bench<14>; break;
    }
    k<<<blocks, threads>>>(1);
    CK(cudaDeviceSynchronize());
    k<<<blocks, threads>>>(3);
    CK(cudaDeviceSynchronize());
    unsigned long long cyc[1024];
    CK(cudaMemcpyFromSymbol(cyc, g_cyc, blocks * 8));
    double mean = 0;
    for (int b = 0; b < blocks; b++) mean += cyc[b];
    mean /= blocks;
    double ops_per_sm;
    if (op == 13 || op == 14) ops_per_sm = (double)blocks_per_sm * (threads / 32) * N * 8;  // warp-level mma per SM
    else ops_per_sm = (double)blocks_per_sm * threads * N * NCH;                           // lane ops per SM
    printf("  %-28s %8.1f %s per clock per SM\n", names[op], ops_per_sm / mean, (op >= 13) ? "warp mma" : "lanes");
  }
  return 0;
}
