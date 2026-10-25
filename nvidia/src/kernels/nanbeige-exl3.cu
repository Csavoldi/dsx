// GPU-only EXL3 adapter. Upstream MIT headers and exact revision are under vendor/exllamav3.
#include "vendor/exllamav3/quant/exl3_gemv_int8_kernel.cuh"
#define NB_EXL3(K,H,M,NAME) \
extern "C" __global__ __launch_bounds__(256) void NAME(const half* A,const uint16_t* B,void* C,int m,int k,int n,int* locks,const half* su,half* ah,const half* sv){exl3_gemv_int8_sq_kernel<K,M,true,false,H>(A,B,C,m,k,n,locks,su,ah,sv);}
NB_EXL3(1,false,1,nb_exl3_sq2_m1)
NB_EXL3(1,false,4,nb_exl3_sq2_m4)
NB_EXL3(1,true,1,nb_exl3_sq3_m1)
NB_EXL3(1,true,4,nb_exl3_sq3_m4)
NB_EXL3(2,false,1,nb_exl3_sq4_m1)
NB_EXL3(2,false,4,nb_exl3_sq4_m4)
NB_EXL3(2,true,1,nb_exl3_sq5_m1)
NB_EXL3(2,true,4,nb_exl3_sq5_m4)
NB_EXL3(3,false,1,nb_exl3_sq6_m1)
NB_EXL3(3,false,4,nb_exl3_sq6_m4)
NB_EXL3(3,true,1,nb_exl3_sq7_m1)
NB_EXL3(3,true,4,nb_exl3_sq7_m4)
NB_EXL3(4,false,1,nb_exl3_sq8_m1)
NB_EXL3(4,false,4,nb_exl3_sq8_m4)
NB_EXL3(5,false,1,nb_exl3_sq10_m1)
NB_EXL3(5,false,4,nb_exl3_sq10_m4)
NB_EXL3(6,false,1,nb_exl3_sq12_m1)
NB_EXL3(6,false,4,nb_exl3_sq12_m4)
NB_EXL3(8,false,1,nb_exl3_sq16_m1)
NB_EXL3(8,false,4,nb_exl3_sq16_m4)
#undef NB_EXL3
extern "C" __global__ void nb_exl3_bf2h(const __nv_bfloat16* x,half* y,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=__float2half(__bfloat162float(x[i]));}
__device__ __forceinline__ int nb_exl3_position(int r,int c){int lane=4*(c%8)+(r%8)/2,j=r%2+2*(r/8)+4*(c/8);return 8*lane+j;}
__device__ __forceinline__ uint32_t nb_exl3_state(const uint32_t* w,int k2,int t){int bits=128*k2,nwords=4*k2,e=(t+1)*(k2/2)+(k2%2?(t+1)/2:0),b=(e+bits-16)%bits;uint64_t v=((uint64_t)w[b/32]<<32)|w[(b/32+1)%nwords];return (uint32_t)(v>>(48-b%32))&65535;}
extern "C" __global__ void nb_exl3_dequant(const uint32_t* w,__nv_bfloat16* out,int n,int k,int k2){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*k)return;int r=i/k,c=i%k,t=nb_exl3_position(c%16,r%16);const uint32_t* tile=w+((size_t)(c/16)*(n/16)+r/16)*(4*k2);uint32_t v=nb_exl3_state(tile,k2,t)*0x83DCD12Du;int b=(v&255)+((v>>8)&255)+((v>>16)&255)+(v>>24);half q=__hfma(__int2half_rn(1024+b),__ushort_as_half(0x1eee),__ushort_as_half(0xc931));out[i]=__float2bfloat16(__half2float(q));}
extern "C" __global__ void nb_exl3_pre(const __nv_bfloat16* x,const half* su,__nv_bfloat16* out,int k){int j=threadIdx.x;size_t off=(size_t)blockIdx.x*128;__shared__ float a[128];a[j]=__bfloat162float(x[off+j])*__half2float(su[(off+j)%k]);__syncthreads();for(int bit=1;bit<128;bit*=2){float own=a[j],other=a[j^bit];__syncthreads();a[j]=(j&bit)?other-own:own+other;__syncthreads();}out[off+j]=__float2bfloat16(a[j]*.08838834765f);}
extern "C" __global__ void nb_exl3_post(float* x,const half* sv,int n){int j=threadIdx.x;size_t off=(size_t)blockIdx.x*128;__shared__ float a[128];a[j]=x[off+j];__syncthreads();for(int bit=1;bit<128;bit*=2){float own=a[j],other=a[j^bit];__syncthreads();a[j]=(j&bit)?other-own:own+other;__syncthreads();}x[off+j]=a[j]*.08838834765f*__half2float(sv[(off+j)%n]);}

// Test direct cached loads against upstream's asynchronous shared-memory staging.
extern "C" __global__ __launch_bounds__(256) void nb_exl3_sq5_m1_narrow(const half* A,const uint16_t* B,void* C,int m,int k,int n,int* locks,const half* su,half* ah,const half* sv){exl3_gemv_int8_sq_kernel<2,1,true,false,true,true>(A,B,C,m,k,n,locks,su,ah,sv);}
extern "C" __global__ __launch_bounds__(256) void nb_exl3_sq5_m4_narrow(const half* A,const uint16_t* B,void* C,int m,int k,int n,int* locks,const half* su,half* ah,const half* sv){exl3_gemv_int8_sq_kernel<2,4,true,false,true,true>(A,B,C,m,k,n,locks,su,ah,sv);}
