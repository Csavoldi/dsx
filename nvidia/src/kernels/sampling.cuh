// Temperature-one, top-p-one categorical sampling. The CPU supplies one 53-bit
// uniform variate. Weights use the CPU sampler's max-30 cutoff; any nonfinite
// input requests the existing CPU fallback. No model weights are changed.
__device__ __forceinline__ float dsx_sample_max256(float value) {
  __shared__ float warps[8];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  if (!lane) warps[warp] = value;
  __syncthreads();
  value = threadIdx.x < 8 ? warps[threadIdx.x] : -INFINITY;
  if (!warp) {
    for (int offset = 16; offset; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    if (!lane) warps[0] = value;
  }
  __syncthreads();
  return warps[0];
}
extern "C" __global__ void dsx_sample_max(const float *logits, int n, float *maxima, unsigned *bad) {
  const int index = blockIdx.x * 256 + threadIdx.x;
  const float value = index < n ? logits[index] : -INFINITY;
  const unsigned invalid = __syncthreads_or(index < n && !isfinite(value));
  const float maximum = dsx_sample_max256(value);
  if (!threadIdx.x) maxima[blockIdx.x] = maximum, bad[blockIdx.x] = invalid;
}
extern "C" __global__ void dsx_sample_weights(const float *logits, int n, const float *maxima, int blocks, float *weights, double *sums) {
  float maximum = -INFINITY;
  for (int i = threadIdx.x; i < blocks; i += 256) maximum = fmaxf(maximum, maxima[i]);
  maximum = dsx_sample_max256(maximum);
  const int index = blockIdx.x * 256 + threadIdx.x;
  float weight = 0.f;
  if (index < n) {
    const float value = logits[index];
    weight = isfinite(maximum) && value >= maximum - 30.f ? expf(value - maximum) : 0.f;
    weights[index] = weight;
  }
  double sum = weight;
  for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
  __shared__ double warp_sums[8];
  if (!(threadIdx.x & 31)) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (!threadIdx.x) {
    double total = 0.;
    for (int i = 0; i < 8; i++) total += warp_sums[i];
    sums[blockIdx.x] = total;
  }
}
extern "C" __global__ void dsx_sample_pick(const float *weights, int n, const double *sums, const unsigned *bad, int blocks,
                                           unsigned long long uniform_bits, unsigned *output) {
  __shared__ double block_sums[1024], residual;
  __shared__ float local_weights[256];
  __shared__ int selected, invalid;
  unsigned any_bad = 0;
  for (int i = threadIdx.x; i < blocks; i += 256) block_sums[i] = sums[i], any_bad |= bad[i];
  const unsigned invalid_input = __syncthreads_or(any_bad);
  if (!threadIdx.x) {
    invalid = invalid_input;
    if (!invalid) {
      double total = 0.;
      for (int i = 0; i < blocks; i++) total += block_sums[i];
      const double uniform = __longlong_as_double((long long)uniform_bits);
      invalid = !(total > 0.) || !isfinite(total) || !(uniform >= 0. && uniform < 1.);
      if (!invalid) {
        residual = uniform * total;
        selected = -1;
        int last_positive = 0;
        for (int i = 0; i < blocks; i++) {
          if (block_sums[i] > 0.) last_positive = i;
          if (residual < block_sums[i]) { selected = i; break; }
          residual -= block_sums[i];
        }
        if (selected < 0) { selected = last_positive; residual = block_sums[selected]; }
      }
    }
    output[1] = invalid;
  }
  __syncthreads();
  if (invalid) return;
  const int index = selected * 256 + threadIdx.x;
  local_weights[threadIdx.x] = index < n ? weights[index] : 0.f;
  __syncthreads();
  if (!threadIdx.x) {
    unsigned last = selected * 256;
    for (int i = 0; i < 256 && selected * 256 + i < n; i++) {
      const float weight = local_weights[i];
      if (weight > 0.f) last = selected * 256 + i;
      residual -= weight;
      if (residual < 0.) { output[0] = selected * 256 + i; return; }
    }
    output[0] = last;
  }
}
