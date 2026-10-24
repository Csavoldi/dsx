#include "../kernels/dsxa.hip"
#include <vector>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
// Frozen pre-change hc_step reference; the candidate is the production hc_step above.
extern "C" __global__ void __launch_bounds__(256) hc_step_reference(float* __restrict__ h0, const float* __restrict__ y0, int round_y,
                                               const float* pmix0, HcArgs pa, const float* premix0, HcArgs prea,
                                               const float* __restrict__ fn, float* nmix0, float* x_raw0, float* xss0,
                                               const float* eng_val0, const float* eng_sum0, const float* eng_qk_unused,
                                               float hc_eps, float eps, int iters, XOut xo) {
    __shared__ float hs[HC][64];
    __shared__ float post[4], comb[16], pre[4];
    __shared__ float m0[32], m1[32];
    __shared__ float red[4][32];
#ifdef HC_LDSRED
    __shared__ float prod[25][257];
#else
    __shared__ float red25[4][25];
#endif
    const int k = threadIdx.x >> 6, i = threadIdx.x & 63, d = blockIdx.x * 64 + i;
    // the next mixes' weights do not depend on h: loaded once, used for every token of the block
    float fw[24];
    if (nmix0) {
#pragma unroll
        for (int j = 0; j < 24; j++) fw[j] = fn[(size_t)j * HC * D + k * D + d];
    }
    const int tpb = xo.tpb > 0 ? xo.tpb : 1, ntok = xo.ntok > 0 ? xo.ntok : (int)gridDim.y * tpb;
    for (int tt = 0; tt < tpb; tt++) {
    // prompt path: token ty of a batch (decode: 0)
    const int ty = blockIdx.y * tpb + tt;
    if (ty >= ntok) break;
    __syncthreads();
    float* h = h0 + (size_t)ty * HC * D;
    const float* y = y0 ? y0 + (size_t)ty * D : nullptr;
    const float* pmix = pmix0 ? pmix0 + (size_t)ty * NPART * 32 : nullptr;
    const float* premix = premix0 ? premix0 + (size_t)ty * NPART * 32 : nullptr;
    float* nmix = nmix0 ? nmix0 + (size_t)ty * NPART * 32 : nullptr;
    float* x_raw = x_raw0 ? x_raw0 + (size_t)ty * D : nullptr;
    float* xss = xss0 ? xss0 + (size_t)ty * NPART : nullptr;
    const float* eng_val = eng_val0 ? eng_val0 + (size_t)ty * 25600 : nullptr;
    const float* eng_sum = eng_sum0 ? eng_sum0 + (size_t)ty * 16 : nullptr;
    {
        const int w = threadIdx.x >> 6, ln = threadIdx.x & 63;
#ifndef HC_NOFIN
        if (xo.mixd) {
            // prompt path: mixes finished once per token by hc_mix_t: [token][2][24] = (pre, post, comb) of pmix, premix
            const float* md = xo.mixd + (size_t)ty * 48;
            if (threadIdx.x < 16) comb[threadIdx.x] = md[8 + threadIdx.x];
            if (threadIdx.x < 4) { post[threadIdx.x] = md[4 + threadIdx.x]; pre[threadIdx.x] = premix ? md[24 + threadIdx.x] : (threadIdx.x == 0 ? 1.f : 0.f); }
        } else
        if (w == 0 && pmix) {
            reduce_mix(pmix, ln, m0);
            hc_finish16(m0, pa, hc_eps, eps, iters, ln, nullptr, post, comb);
        }
        if (w == 1) {
            if (premix) {
                reduce_mix(premix, ln, m1);
                hc_finish16(m1, prea, hc_eps, eps, iters, ln, pre, nullptr, nullptr);
            } else if (ln < 4) pre[ln] = ln == 0 ? 1.f : 0.f;
        }
#else
        if (threadIdx.x < 16) { comb[threadIdx.x] = 0.25f; if (threadIdx.x < 4) post[threadIdx.x] = pre[threadIdx.x] = 0.5f; }
#endif
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
        if (xo.cnt) st_dev(x_raw + d, x);
        else x_raw[d] = x;
        float s = wave_sum(x * x);
        if (i == 0) xss[blockIdx.x] = s;
    }
    // next sub-block's mix dots: 24 rows of fn over this block's 4x64 h values + sum h^2
    // next sub-block's mix dots: 24 rows of fn over this block's 4x64 h values + sum h^2, reduced through LDS
    if (nmix) {
#ifdef HC_LDSRED
        const int t = threadIdx.x;
#pragma unroll
        for (int j = 0; j < 24; j++) prod[j][t] = fw[j] * hv;
        prod[24][t] = hv * hv;
        __syncthreads();
        // 25 sums of 256: 8 threads per sum (t < 200), then a 3-step shuffle
        if (t < 200) {
            const int j = t >> 3, q = t & 7;
            float sacc = 0;
#pragma unroll
            for (int u = 0; u < 32; u++) sacc += prod[j][q * 32 + u];
            sacc = group_sum<8>(sacc);
            if (q == 0) nmix[blockIdx.x * 32 + j] = sacc;
        }
#else
        // 25 sums of 256: wave sums (one wave per stream k), then the 4 waves through LDS (small LDS: more blocks
        // per CU; this kernel is latency-bound)
#pragma unroll
        for (int j = 0; j < 25; j++) {
            const float sv = wave_sum(j < 24 ? fw[j] * hv : hv * hv);
            if (i == 0) red25[k][j] = sv;
        }
        __syncthreads();
        if (threadIdx.x < 25) nmix[blockIdx.x * 32 + threadIdx.x] = red25[0][threadIdx.x] + red25[1][threadIdx.x] + red25[2][threadIdx.x] + red25[3][threadIdx.x];
#endif
    }
    if (xo.cnt) {
        __shared__ int last;
        __shared__ float red2[16];
        __builtin_amdgcn_s_waitcnt(0);
        __syncthreads();
        if (threadIdx.x == 0) last = atomicAdd(xo.cnt, 1u) == gridDim.x - 1;
        __syncthreads();
        if (last) {
            float sq = 0;
            for (int j = threadIdx.x; j < D; j += 256) { const float v = ld_dev(x_raw + j); sq += v * v; }
            sq = block_sum(sq, red2);
            const float r = 1.f / sqrtf(sq / D + eps);
            write_act(D, [&](int j) -> float { return round_bf16(xo.nw[j] * (ld_dev(x_raw + j) * r)); }, xo.oq, xo.os, xo.ob);
        }
    }
    }  // tokens
}
#define HIPCHECK(x) do {hipError_t e=(x);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(2);}}while(0)
float* allocf(size_t n) {float*p;HIPCHECK(hipMalloc(&p,n*sizeof(float)));HIPCHECK(hipMemset(p,0,n*sizeof(float)));return p;}
int main(int argc, char** argv){
 const bool timing = !(argc > 1 && strcmp(argv[1], "--check") == 0);
 hipStream_t stream;HIPCHECK(hipStreamCreate(&stream));
 std::mt19937 rng(42); std::uniform_real_distribution<float> u(-0.2f,0.2f);
 for(int B: {1,4,8}) for(int mode: {0,1,2}) for(int tpb: {1,4}) for(int use_mixd: {0,1}) for(int graph: {0,1}) {
  size_t hn=(size_t)B*HC*D, yn=(size_t)B*D, mn=(size_t)B*NPART*32;
  std::vector<float> h(hn),y(yn),mix(mn),fn(24*HC*D);
  for(auto&v:h)v=u(rng);for(auto&v:y)v=u(rng);for(auto&v:mix)v=u(rng)*.01f;for(auto&v:fn)v=u(rng);
  for(int t=0;t<B;t++)for(int j=0;j<NPART;j++)mix[(t*NPART+j)*32+24]=256.f;
  float*dh=allocf(hn),*dy=allocf(yn),*dm=allocf(mn),*df=allocf(fn.size()),*md=allocf(B*48),*out=allocf(mn),*x=allocf(yn),*ss=allocf(B*NPART);
  HIPCHECK(hipMemcpy(dy,y.data(),yn*4,hipMemcpyHostToDevice));HIPCHECK(hipMemcpy(dm,mix.data(),mn*4,hipMemcpyHostToDevice));HIPCHECK(hipMemcpy(df,fn.data(),fn.size()*4,hipMemcpyHostToDevice));
  HcArgs a{};for(int j=0;j<3;j++)a.scale[j]=1.f;for(int j=0;j<24;j++)a.base[j]=u(rng);
  const float*pm=mode==1?nullptr:dm;const float*pre=mode==2?nullptr:dm;
  hipLaunchKernelGGL(hc_mix_t,dim3(B),dim3(128),0,stream,pm,a,pre,a,md,1e-6f,1e-6f,20);
  XOut xo{};xo.mixd=use_mixd?md:nullptr;xo.ntok=B;xo.tpb=tpb;
  auto raw_launch=[&](bool skip){
   if(skip)hipLaunchKernelGGL(hc_step,dim3(NPART,(B+tpb-1)/tpb),dim3(256),0,stream,dh,dy,1,pm,a,pre,a,df,out,x,ss,nullptr,nullptr,nullptr,1e-6f,1e-6f,20,xo);
   else hipLaunchKernelGGL(hc_step_reference,dim3(NPART,(B+tpb-1)/tpb),dim3(256),0,stream,dh,dy,1,pm,a,pre,a,df,out,x,ss,nullptr,nullptr,nullptr,1e-6f,1e-6f,20,xo);
  };
  HIPCHECK(hipStreamSynchronize(stream));
  hipGraph_t graphs[2]{};hipGraphExec_t execs[2]{};
  if(graph)for(int skip=0;skip<2;skip++){
   HIPCHECK(hipStreamBeginCapture(stream,hipStreamCaptureModeGlobal));raw_launch(skip);
   HIPCHECK(hipStreamEndCapture(stream,&graphs[skip]));HIPCHECK(hipGraphInstantiate(&execs[skip],graphs[skip],nullptr,nullptr,0));
  }
  auto launch=[&](bool skip){if(graph)HIPCHECK(hipGraphLaunch(execs[skip],stream));else raw_launch(skip);};
  auto run=[&](bool skip){HIPCHECK(hipMemcpy(dh,h.data(),hn*4,hipMemcpyHostToDevice));launch(skip);HIPCHECK(hipDeviceSynchronize());std::vector<float>v(hn+mn+yn+B*NPART);size_t k=0;for(auto z: {std::make_pair(dh,hn),std::make_pair(out,mn),std::make_pair(x,yn),std::make_pair(ss,(size_t)B*NPART)}){HIPCHECK(hipMemcpy(v.data()+k,z.first,z.second*4,hipMemcpyDeviceToHost));k+=z.second;}return v;};
  auto v0=run(false),v1=run(true);size_t dif=0;float maxd=0;for(size_t j=0;j<v0.size();j++){if(memcmp(&v0[j],&v1[j],4))dif++;maxd=fmaxf(maxd,fabsf(v0[j]-v1[j]));}
  printf("B=%d mode=%d tpb=%d mixd=%d graph=%d compared=%zu bit_differences=%zu maxabs=%g\n",B,mode,tpb,use_mixd,graph,v0.size(),dif,maxd);fflush(stdout);if(dif)return 3;
  // Event timing alternates baseline/candidate; immutable h is restored outside each timed launch group.
  if(timing)for(int round=0;round<3;round++)for(bool skip: {false,true}){
   HIPCHECK(hipMemcpy(dh,h.data(),hn*4,hipMemcpyHostToDevice));for(int j=0;j<5;j++)launch(skip);
   hipEvent_t a0,a1;HIPCHECK(hipEventCreate(&a0));HIPCHECK(hipEventCreate(&a1));HIPCHECK(hipEventRecord(a0,stream));for(int j=0;j<100;j++)launch(skip);HIPCHECK(hipEventRecord(a1,stream));HIPCHECK(hipEventSynchronize(a1));float ms;HIPCHECK(hipEventElapsedTime(&ms,a0,a1));printf("B=%d mode=%d tpb=%d mixd=%d graph=%d round=%d skip=%d us=%.3f\n",B,mode,tpb,use_mixd,graph,round,skip,ms*10);HIPCHECK(hipEventDestroy(a0));HIPCHECK(hipEventDestroy(a1));
  }
  if(graph)for(int skip=0;skip<2;skip++){HIPCHECK(hipGraphExecDestroy(execs[skip]));HIPCHECK(hipGraphDestroy(graphs[skip]));}
  for(auto p:{dh,dy,dm,df,md,out,x,ss})HIPCHECK(hipFree(p));
 }
 HIPCHECK(hipStreamDestroy(stream));
}
