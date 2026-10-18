// common.h: shared device helpers for the dsx-amd kernels (gfx942 / CDNA3).
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef float f4 __attribute__((ext_vector_type(4)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef uint32_t u2v __attribute__((ext_vector_type(2)));
typedef short s4v __attribute__((ext_vector_type(4)));

#define DEV __device__ __forceinline__

// D = A(16x32 fp8) * B(32x16 fp8) + C. Lane l: A row l%16, k 8*(l/16)..+8; B col l%16, same k; D col l%16, rows 4*(l/16)+j.
DEV f4 mfma_fp8(uint64_t a, uint64_t b, f4 c) { return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8((long)a, (long)b, c, 0, 0, 0); }
// D = A(16x16 bf16) * B(16x16 bf16) + C. Lane l: A row l%16, k 4*(l/16)..+4.
DEV f4 mfma_bf16(s4v a, s4v b, f4 c) { return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0); }

DEV uint32_t perm(uint32_t hi, uint32_t lo, uint32_t sel) { return __builtin_amdgcn_perm(hi, lo, sel); }

// f32 -> bf16 bits, round to nearest even (no NaN inputs here)
DEV uint32_t bf16_bits(float x) {
    uint32_t u = __float_as_uint(x);
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}
DEV float round_bf16(float x) { return __uint_as_float(bf16_bits(x) << 16); }
DEV float bf16_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

// fp8 e4m3 FNUZ (gfx942's fp8) from f32, RNE, saturating. The model's FP8 is OCP e4m3fn; an OCP value v is stored
// as the FNUZ code of v/2 (same bits, exact), so a quantizer that targets OCP's grid at scale s writes FNUZ codes at 2s.
DEV uint32_t cvt_fp8x4(float a, float b, float c, float d) {
    int lo = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    return (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(c, d, lo, true);
}
DEV float fp8_to_f32(uint32_t v, int byte) {
    switch (byte) {
        case 0: return __builtin_amdgcn_cvt_f32_fp8(v, 0);
        case 1: return __builtin_amdgcn_cvt_f32_fp8(v, 1);
        case 2: return __builtin_amdgcn_cvt_f32_fp8(v, 2);
        default: return __builtin_amdgcn_cvt_f32_fp8(v, 3);
    }
}

DEV float wave_sum(float v) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o);
    return v;
}
DEV float wave_max(float v) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
    return v;
}
// sum over the 32 lanes of each half-wave group of `w` lanes (w power of two <= 64)
template <int W> DEV float group_max(float v) {
#pragma unroll
    for (int o = W / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
    return v;
}
template <int W> DEV float group_sum(float v) {
#pragma unroll
    for (int o = W / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
    return v;
}

// block-wide sum; `red` is __shared__ float[16]; all threads get the result
DEV float block_sum(float v, float* red) {
    v = wave_sum(v);
    const int w = threadIdx.x >> 6, nw = (blockDim.x + 63) >> 6;
    __syncthreads();
    if ((threadIdx.x & 63) == 0) red[w] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < nw; i++) s += red[i];
    return s;
}
DEV float block_max(float v, float* red) {
    v = wave_max(v);
    const int w = threadIdx.x >> 6, nw = (blockDim.x + 63) >> 6;
    __syncthreads();
    if ((threadIdx.x & 63) == 0) red[w] = v;
    __syncthreads();
    float s = -INFINITY;
    for (int i = 0; i < nw; i++) s = fmaxf(s, red[i]);
    return s;
}

// power of two >= x (x > 0), as in DeepSeek's kernels (ceil of log2)
DEV float pow2_ceil(float x) {
    int e;
    float m = frexpf(x, &e);  // x = m * 2^e, m in [0.5, 1)
    return m == 0.5f ? ldexpf(1.f, e - 1) : ldexpf(1.f, e);
}

// relaxed device-scope (L2-bypassing) accesses for cross-block handoff without fences
DEV void st_dev(float* p, float v) { __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
DEV float ld_dev(const float* p) { return __hip_atomic_load(const_cast<float*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
DEV void st_dev_u(uint32_t* p, uint32_t v) { __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
DEV uint32_t ld_dev_u(const uint32_t* p) { return __hip_atomic_load(const_cast<uint32_t*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
