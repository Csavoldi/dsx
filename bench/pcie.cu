// Host->device bandwidth per GPU (pinned, 64 MB and 10 MB transfers, one and both GPUs at once),
// empty-kernel launch latency and CUDA-graph launch latency, on each GPU.
// nvcc -O3 -arch=sm_86 pcie.cu -o pcie && ./pcie
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <chrono>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void empty() {}

static double h2d(int dev, void *host, size_t bytes, int reps) {
  CK(cudaSetDevice(dev));
  void *d; CK(cudaMalloc(&d, bytes));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaMemcpyAsync(d, host, bytes, cudaMemcpyHostToDevice, s));
  CK(cudaEventRecord(a, s));
  for (int i = 0; i < reps; i++) CK(cudaMemcpyAsync(d, host, bytes, cudaMemcpyHostToDevice, s));
  CK(cudaEventRecord(b, s));
  CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b));
  CK(cudaFree(d));
  return bytes * (double)reps / (ms / 1e3) / 1e9;
}

int main() {
  int n; CK(cudaGetDeviceCount(&n));
  size_t big = 64 << 20, small = 10 << 20;
  void *host; CK(cudaMallocHost(&host, big * 2));
  for (int d = 0; d < n; d++) {
    printf("GPU%d H2D 64MB: %.1f GB/s   10MB: %.1f GB/s\n", d, h2d(d, host, big, 40), h2d(d, host, small, 100));
  }
  {  // THP-backed host memory registered with CUDA: fewer IOMMU mappings than 4K-page cudaMallocHost
    size_t sz = 256 << 20;
    void *p = aligned_alloc(2 << 20, sz);
    madvise(p, sz, MADV_HUGEPAGE);
    memset(p, 1, sz);
    CK(cudaHostRegister(p, sz, cudaHostRegisterDefault));
    for (int d = 0; d < n; d++) printf("GPU%d H2D 64MB from THP-registered memory: %.1f GB/s\n", d, h2d(d, p, big, 40));
    CK(cudaHostUnregister(p));
  }
  if (n >= 2) {
    double r[2];
    std::vector<std::thread> t;
    for (int d = 0; d < 2; d++) t.emplace_back([&, d] { r[d] = h2d(d, (char *)host + d * big, big, 80); });
    for (auto &x : t) x.join();
    printf("both at once: GPU0 %.1f + GPU1 %.1f GB/s\n", r[0], r[1]);
  }
  for (int d = 0; d < n; d++) {
    CK(cudaSetDevice(d));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    empty<<<1, 32, 0, s>>>(); CK(cudaStreamSynchronize(s));
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a, s));
    for (int i = 0; i < 10000; i++) empty<<<1, 32, 0, s>>>();
    CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    // graph of 1000 empty kernels
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < 1000; i++) empty<<<1, 32, 0, s>>>();
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(a, s));
    for (int i = 0; i < 10; i++) CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b));
    float gms; CK(cudaEventElapsedTime(&gms, a, b));
    // round trip: launch + sync from host
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) { empty<<<1, 32, 0, s>>>(); CK(cudaStreamSynchronize(s)); }
    auto t1 = std::chrono::high_resolution_clock::now();
    double rt = std::chrono::duration<double, std::micro>(t1 - t0).count() / 1000;
    printf("GPU%d empty kernel back-to-back %.2f us, in a graph %.2f us, launch+sync round trip %.1f us\n",
           d, ms * 1e3 / 10000, gms * 1e3 / 10000, rt);
  }
  return 0;
}
