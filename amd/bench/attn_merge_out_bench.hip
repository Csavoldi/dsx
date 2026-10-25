#include "../kernels/dsxa.hip"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CHECK(x) do{hipError_t e=(x);if(e!=hipSuccess){fprintf(stderr,"%d %s\n",__LINE__,hipGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CHECK(hipMalloc(&p,n*sizeof(T)));CHECK(hipMemset(p,0,n*sizeof(T)));return p;}
int main(){
 hipStream_t st;CHECK(hipStreamCreate(&st));std::mt19937 rng(719);std::uniform_real_distribution<float> u(-1,1);
 for(int B:{1,4,8})for(int pos:{1,4000,129000})for(int masked:{0,1,2})for(int graph:{0,1}){
  size_t np=(size_t)B*ATTN_SPLITS*ATTN_PART,nq=(size_t)B*NH*HD;
  std::vector<float> part(np),sinks(NH),rf(RD/2);std::vector<int>bp(B,pos);
  for(auto&v:part)v=u(rng);for(auto&v:sinks)v=u(rng);for(auto&v:rf)v=.002f+.01f*u(rng);
  for(int t=0;t<B;t++)for(int sp=0;sp<ATTN_SPLITS;sp++)for(int h=0;h<NH;h++){
   size_t off=((size_t)t*ATTN_SPLITS+sp)*ATTN_PART+NH*HD;
   part[off+h]=(masked==2||(masked==1&&sp%3!=0))?-INFINITY:u(rng);
   part[off+64+h]=.1f+fabsf(u(rng));
  }
  float*dp=alloc<float>(np),*ds=alloc<float>(NH),*dr=alloc<float>(RD/2),*q=alloc<float>(nq),*sc=alloc<float>(nq/32);int*db=alloc<int>(B);
  uint8_t*code=alloc<uint8_t>(nq);uint16_t*bf=alloc<uint16_t>(nq);
  CHECK(hipMemcpy(dp,part.data(),np*4,hipMemcpyHostToDevice));CHECK(hipMemcpy(ds,sinks.data(),NH*4,hipMemcpyHostToDevice));CHECK(hipMemcpy(dr,rf.data(),RD/2*4,hipMemcpyHostToDevice));CHECK(hipMemcpy(db,bp.data(),B*4,hipMemcpyHostToDevice));
  auto raw=[&](int fused){
   if(fused)hipLaunchKernelGGL(attn_merge_out_b,dim3(B,NH),dim3(128),0,st,q,dp,ds,db,dr,code,sc,bf);
   else{
    hipLaunchKernelGGL(attn_merge_b,dim3(B,NH),dim3(512),0,st,q,dp,ds);
    hipLaunchKernelGGL(attn_out_b,dim3(B,NH/8),dim3(256),0,st,db,q,dr,code,sc,bf);
   }
  };
  hipGraph_t gs[2]{};hipGraphExec_t ex[2]{};
  if(graph)for(int fused=0;fused<2;fused++){CHECK(hipStreamBeginCapture(st,hipStreamCaptureModeGlobal));raw(fused);CHECK(hipStreamEndCapture(st,&gs[fused]));CHECK(hipGraphInstantiate(&ex[fused],gs[fused],nullptr,nullptr,0));}
  auto launch=[&](int fused){if(graph)CHECK(hipGraphLaunch(ex[fused],st));else raw(fused);};
  auto result=[&](int fused){
   launch(fused);CHECK(hipStreamSynchronize(st));std::vector<unsigned char>out(nq*4+nq+nq/32*4+nq*2);size_t off=0;
   auto copy=[&](void*p,size_t bytes){CHECK(hipMemcpy(out.data()+off,p,bytes,hipMemcpyDeviceToHost));off+=bytes;};copy(q,nq*4);copy(code,nq);copy(sc,nq/32*4);copy(bf,nq*2);return out;
  };
  auto base=result(0),candidate=result(1);size_t diff=0;for(size_t i=0;i<base.size();i++)diff+=(base[i]!=candidate[i]);
  size_t qdiff=0,cdiff=0,sdiff=0,bdiff=0;float qmax=0;int shown=0;
  for(size_t i=0;i<nq;i++){float x,y;memcpy(&x,base.data()+i*4,4);memcpy(&y,candidate.data()+i*4,4);if(memcmp(&x,&y,4)){qdiff++;qmax=fmaxf(qmax,fabsf(x-y));if(shown++<4)printf("QDIFF head=%zu d=%zu old=%.9g new=%.9g\n",i/HD,i%HD,x,y);}}
  for(size_t i=nq*4;i<nq*5;i++)cdiff+=(base[i]!=candidate[i]);
  for(size_t i=nq*5;i<nq*5+nq/32*4;i++)sdiff+=(base[i]!=candidate[i]);
  for(size_t i=nq*5+nq/32*4;i<base.size();i++)bdiff+=(base[i]!=candidate[i]);
  printf("B=%d pos=%d masked=%d graph=%d bytes=%zu differing_bytes=%zu q_values=%zu qmax=%g code_bytes=%zu scale_bytes=%zu bf16_bytes=%zu\n",B,pos,masked,graph,base.size(),diff,qdiff,qmax,cdiff,sdiff,bdiff);fflush(stdout);if(diff)return 3;
  if(pos==4000&&masked==0)for(int round=0;round<4;round++)for(int fused:{0,1}){
   for(int i=0;i<5;i++)launch(fused);hipEvent_t e0,e1;CHECK(hipEventCreate(&e0));CHECK(hipEventCreate(&e1));CHECK(hipEventRecord(e0,st));for(int i=0;i<100;i++)launch(fused);CHECK(hipEventRecord(e1,st));CHECK(hipEventSynchronize(e1));float ms;CHECK(hipEventElapsedTime(&ms,e0,e1));printf("TIME B=%d graph=%d round=%d fused=%d us=%.3f\n",B,graph,round,fused,ms*10);CHECK(hipEventDestroy(e0));CHECK(hipEventDestroy(e1));
  }
  if(graph)for(int i=0;i<2;i++){CHECK(hipGraphExecDestroy(ex[i]));CHECK(hipGraphDestroy(gs[i]));}
  for(void*p:{(void*)dp,(void*)ds,(void*)dr,(void*)q,(void*)sc,(void*)db,(void*)code,(void*)bf})CHECK(hipFree(p));
 }
 CHECK(hipStreamDestroy(st));
}
