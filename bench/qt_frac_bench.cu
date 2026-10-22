// qt_frac_bench.cu - exllamav3's fractional-K trellis quantizer (quantize_tiles_frac_kernel, the generic path) against
// quant/t16/qt_frac_opt.cuh on the same Gaussian tiles: indices must match bit for bit; times per kernel.
// Built against the T16-patched exllamav3 tree (quant/t16/make_mul16_exllamav3.sh): its codebook.cuh and
// quantize_tiles_frac_kernel.cuh, with their #include lines stripped (they pull in torch headers), see build.
//
// build: Q=~/src/t16q/exl3-t16/exllamav3/exllamav3_ext/quant
//        sed '/#include/d' $Q/codebook.cuh > qtf_codebook.cuh; sed '/#include/d' $Q/quantize_tiles_frac_kernel.cuh > qtf_stock.cuh
//        nvcc -O3 -arch=sm_86 -I. -I../quant/t16 -o qt_frac_bench qt_frac_bench.cu
// run:   ./qt_frac_bench [tiles] [KA 1|2] [which: 1 stock, 2 opt 1024 threads, 4 opt 512 threads; bits OR'd]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

union half2_uint32 {
  uint32_t as_uint32;
  half2 as_half2;
  __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
  __device__ half2_uint32(half2 val) : as_half2(val) {}
};
union half_uint16 {
  uint16_t as_uint16;
  half as_half;
  __device__ half_uint16(uint16_t val) : as_uint16(val) {}
  __device__ half_uint16(half val) : as_half(val) {}
};
#define H_INF __ushort_as_half(0x7c00)
#include "qtf_codebook.cuh"
#include "qtf_stock.cuh"
#include "qt_frac_opt.cuh"

typedef void (*fp_kernel)(const float *, float *, uint16_t *, half *, uint16_t *, const half2 *);

static float run(fp_kernel k, int threads, int shmem, int batch, const float *din, float *dout, uint16_t *didx, half *costs,
                 uint16_t *edges, int n) {
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  for (int i = 0; i < n; i += batch) {
    const int bs = n - i < batch ? n - i : batch;
    k<<<bs, threads, shmem>>>(din + 256L * i, dout + 256L * i, didx + 256L * i, costs, edges, nullptr);
    CK(cudaPeekAtLastError());
  }
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  float ms;
  CK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

int main(int argc, char **argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 4096, KA = argc > 2 ? atoi(argv[2]) : 1, which = argc > 3 ? atoi(argv[3]) : 7;
  std::vector<float> h(256L * n);
  uint64_t s = 0x9E3779B97F4A7C15ull;
  auto u = [&]() { s = s * 6364136223846793005ull + 1442695040888963407ull; return ((s >> 11) + 0.5) * (1.0 / 9007199254740992.0); };
  for (size_t i = 0; i < h.size(); i += 2) {
    const double r = sqrt(-2 * log(u())), t = 6.283185307179586 * u();
    h[i] = (float)(r * cos(t)), h[i + 1] = (float)(r * sin(t));
  }
  float *din, *dout;
  uint16_t *didx;
  CK(cudaMalloc(&din, h.size() * 4));
  CK(cudaMalloc(&dout, h.size() * 4));
  CK(cudaMalloc(&didx, h.size() * 2));
  CK(cudaMemcpy(din, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  const int edges_max = 65536 >> KA, sb = 128;  // stock scratch: 128 tiles, as exllamav3's get_temp_buffers_frac
  half *costs;
  uint16_t *edges;
  const size_t edge_bytes = (size_t)sb * 256 * edges_max * 2;  // 2 GB at KA = 1; the opt kernel reuses it (1 MB per tile)
  CK(cudaMalloc(&costs, (size_t)sb * 2 * edges_max * 2));
  CK(cudaMalloc(&edges, edge_bytes));
  const int ob = (int)(edge_bytes / (QTF_HIST_WORDS * 4));
  std::vector<uint16_t> ref, got(h.size());
  std::vector<float> refw, gotw(h.size());
  struct K { int bit; const char *name; fp_kernel k; int threads, shmem, batch; };
  std::vector<K> ks;
  const int stock_shmem = 256 * 2 + 32 * 4 + 128;
  if (KA == 1) {
    ks.push_back({1, "stock frac (generic)", quantize_tiles_frac_kernel<1, 0xaaaau>, 512, stock_shmem, sb});
    ks.push_back({2, "opt, 1024 threads", quantize_tiles_frac_opt_kernel<1, 0xaaaau, 1024>, 1024, qt_frac_opt_shmem(1), ob});
    ks.push_back({4, "opt, 512 threads", quantize_tiles_frac_opt_kernel<1, 0xaaaau, 512>, 512, qt_frac_opt_shmem(1), ob});
  } else {
    ks.push_back({1, "stock frac (generic)", quantize_tiles_frac_kernel<2, 0xaaaau>, 512, stock_shmem, sb});
    ks.push_back({2, "opt, 1024 threads", quantize_tiles_frac_opt_kernel<2, 0xaaaau, 1024>, 1024, qt_frac_opt_shmem(2), ob});
    ks.push_back({4, "opt, 512 threads", quantize_tiles_frac_opt_kernel<2, 0xaaaau, 512>, 512, qt_frac_opt_shmem(2), ob});
  }
  printf("%d Gaussian tiles, K = %d.5\n", n, KA);
  for (auto &k : ks) {
    if (!(which & k.bit)) continue;
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, k.k));
    run(k.k, k.threads, k.shmem, k.batch, din, dout, didx, costs, edges, n < 256 ? n : 256);  // warm-up
    const float ms = run(k.k, k.threads, k.shmem, k.batch, din, dout, didx, costs, edges, n);
    CK(cudaMemcpy(got.data(), didx, got.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gotw.data(), dout, gotw.size() * 4, cudaMemcpyDeviceToHost));
    double se = 0;
    for (size_t i = 0; i < h.size(); i++) se += (gotw[i] - h[i]) * (double)(gotw[i] - h[i]);
    long diff = -1, wdiff = -1;
    if (ref.empty()) ref = got, refw = gotw;
    else {
      diff = 0, wdiff = 0;
      for (size_t i = 0; i < got.size(); i++) diff += got[i] != ref[i], wdiff += gotw[i] != refw[i];
    }
    printf("  %-22s %8.1f ms  (%6.1f us per tile)  regs %3d  mse %.5f", k.name, ms, 1e3 * ms / n, fa.numRegs, se / h.size());
    if (diff >= 0) printf("  indices differing from the first kernel: %ld, weights: %ld", diff, wdiff);
    printf("\n");
  }
  return 0;
}
