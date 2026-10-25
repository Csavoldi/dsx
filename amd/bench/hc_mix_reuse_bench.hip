#include "../kernels/dsxa.hip"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CHECK(x) do{hipError_t e=(x);if(e!=hipSuccess){fprintf(stderr,"%d %s\n",__LINE__,hipGetErrorString(e));exit(2);}}while(0)
__global__ void hc_mix_reference(const float*pmix,HcArgs pa,const float*premix,HcArgs prea,float*out,float hc_eps,float eps,int iters){
 __shared__ float m0[32],m1[32];int t=blockIdx.x,ln=threadIdx.x;float*o=out+(size_t)t*48;
 if(pmix){reduce_mix(pmix+(size_t)t*NPART*32,ln,m0);hc_finish16(m0,pa,hc_eps,eps,iters,ln,o,o+4,o+8);}
 if(premix){reduce_mix(premix+(size_t)t*NPART*32,ln,m1);hc_finish16(m1,prea,hc_eps,eps,iters,ln,o+24,nullptr,nullptr);}
}
template<class T>T*alloc(size_t n){T*p;CHECK(hipMalloc(&p,n*sizeof(T)));CHECK(hipMemset(p,0,n*sizeof(T)));return p;}
int main(){
 hipStream_t st;CHECK(hipStreamCreate(&st));std::mt19937 rng(99);std::uniform_real_distribution<float>u(-.5,.5);
 for(int B:{1,4,8})for(int mode:{0,1,2,3,4,5})for(int graph:{0,1}){
  size_t n=(size_t)B*NPART*32;std::vector<float>a(n),b(n);for(auto&v:a)v=u(rng);for(auto&v:b)v=u(rng);
  for(int t=0;t<B;t++)for(int j=0;j<NPART;j++){a[(t*NPART+j)*32+24]=256.f;b[(t*NPART+j)*32+24]=256.f;}
  float*da=alloc<float>(n),*db=alloc<float>(n),*out=alloc<float>(B*48);CHECK(hipMemcpy(da,a.data(),n*4,hipMemcpyHostToDevice));CHECK(hipMemcpy(db,b.data(),n*4,hipMemcpyHostToDevice));
  HcArgs pa{},pre{};for(int j=0;j<3;j++){pa.scale[j]=1;pre.scale[j]=.9f;}for(int j=0;j<24;j++){pa.base[j]=u(rng);pre.base[j]=u(rng);}if(mode==0)pre=pa;
  const float*pm=(mode==3||mode==5)?nullptr:da;const float*pr=(mode==4||mode==5)?nullptr:(mode<=1?da:db);
  auto raw=[&](int opt){if(opt)hipLaunchKernelGGL(hc_mix_t,dim3(B),dim3(64),0,st,pm,pa,pr,pre,out,1e-6f,1e-6f,20);else hipLaunchKernelGGL(hc_mix_reference,dim3(B),dim3(64),0,st,pm,pa,pr,pre,out,1e-6f,1e-6f,20);};
  hipGraph_t gs[2]{};hipGraphExec_t ex[2]{};if(graph)for(int opt=0;opt<2;opt++){CHECK(hipStreamBeginCapture(st,hipStreamCaptureModeGlobal));raw(opt);CHECK(hipStreamEndCapture(st,&gs[opt]));CHECK(hipGraphInstantiate(&ex[opt],gs[opt],nullptr,nullptr,0));}
  auto launch=[&](int opt){if(graph)CHECK(hipGraphLaunch(ex[opt],st));else raw(opt);};
  auto run=[&](int opt){launch(opt);CHECK(hipStreamSynchronize(st));std::vector<float>v(B*48);CHECK(hipMemcpy(v.data(),out,B*48*4,hipMemcpyDeviceToHost));return v;};
  auto r0=run(0),r1=run(1);size_t diff=0;for(size_t j=0;j<r0.size();j++)diff+=(memcmp(&r0[j],&r1[j],4)!=0);printf("B=%d mode=%d graph=%d floats=%zu bit_differences=%zu\n",B,mode,graph,r0.size(),diff);fflush(stdout);if(diff)return 3;
  if(mode<=2)for(int round=0;round<4;round++)for(int opt:{0,1}){for(int j=0;j<5;j++)launch(opt);hipEvent_t e0,e1;CHECK(hipEventCreate(&e0));CHECK(hipEventCreate(&e1));CHECK(hipEventRecord(e0,st));for(int j=0;j<100;j++)launch(opt);CHECK(hipEventRecord(e1,st));CHECK(hipEventSynchronize(e1));float ms;CHECK(hipEventElapsedTime(&ms,e0,e1));printf("TIME B=%d mode=%d graph=%d round=%d opt=%d us=%.3f\n",B,mode,graph,round,opt,ms*10);CHECK(hipEventDestroy(e0));CHECK(hipEventDestroy(e1));}
  if(graph)for(int j=0;j<2;j++){CHECK(hipGraphExecDestroy(ex[j]));CHECK(hipGraphDestroy(gs[j]));}CHECK(hipFree(da));CHECK(hipFree(db));CHECK(hipFree(out));
 }
 CHECK(hipStreamDestroy(st));
}
