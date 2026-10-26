#include "../kernels/dsxa.hip"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"line%d %s\n",__LINE__,hipGetErrorString(e));exit(2);}}while(0)
template<class T>T* alloc(size_t n){T*p;CK(hipMalloc(&p,n*sizeof(T)));CK(hipMemset(p,0,n*sizeof(T)));return p;}
struct Out {uint8_t*q;float*s;uint16_t*b,*bq;Out(int rows):q(alloc<uint8_t>(rows*INTER)),s(alloc<float>(rows*INTER/32)),b(alloc<uint16_t>(rows*INTER)),bq(alloc<uint16_t>(rows*INTER)){} ~Out(){hipFree(q);hipFree(s);hipFree(b);hipFree(bq);}};
void equal(void*a,void*b,size_t n){std::vector<char>x(n),y(n);CK(hipMemcpy(x.data(),a,n,hipMemcpyDeviceToHost));CK(hipMemcpy(y.data(),b,n,hipMemcpyDeviceToHost));if(x!=y){for(size_t i=0;i<n;i++)if(x[i]!=y[i]){fprintf(stderr,"mismatch byte%zu\n",i);break;}exit(3);}}
int main(){std::mt19937 rng(9283);std::normal_distribution<float> norm(0,12);int cases=0;
 for(int B:{1,4,8,32})for(int K:{3,6})for(int mode:{0,1,2,3})for(float lim:{7.f,20.f}){
  int R=B*K,N=(R+B)*INTER;std::vector<float>x(N),u(N),wt(R);for(auto&v:x)v=norm(rng);for(auto&v:u)v=norm(rng);for(int i=0;i<R;i++)wt[i]=i%5==0?0.f:.15f*(i%7);if(mode==3)for(int i=0;i<N;i++)x[i]=u[i]=0;
  float*dx=alloc<float>(N),*du=alloc<float>(N),*dw=alloc<float>(R);CK(hipMemcpy(dx,x.data(),N*4,hipMemcpyHostToDevice));CK(hipMemcpy(du,u.data(),N*4,hipMemcpyHostToDevice));CK(hipMemcpy(dw,wt.data(),R*4,hipMemcpyHostToDevice));
  Out a(R+B),b(R+B);ZeroList zl{};
  auto launch=[&](bool fused){
   auto bf=[&](Out&o,int row){return mode&1?o.b+(size_t)row*INTER:nullptr;};auto bq=[&](Out&o,int row){return mode&2?o.bq+(size_t)row*INTER:nullptr;};
   if(fused)hipLaunchKernelGGL(prep_swiglu_pair_t,dim3(R+B,3),dim3(256),0,0,R,dx,du,dw,b.q,b.s,bf(b,0),bq(b,0),dx+(size_t)R*INTER,du+(size_t)R*INTER,b.q+(size_t)R*INTER,b.s+(size_t)R*INTER/32,bf(b,R),bq(b,R),lim);
   else{
    hipLaunchKernelGGL(prep_t,dim3(R,3),dim3(256),0,0,3,dx,du,INTER,(float*)nullptr,(float*)nullptr,dw,INTER,1e-6f,lim,a.q,a.s,bf(a,0),bq(a,0),zl);
    hipLaunchKernelGGL(prep_t,dim3(B,3),dim3(256),0,0,3,dx+(size_t)R*INTER,du+(size_t)R*INTER,INTER,(float*)nullptr,(float*)nullptr,(float*)nullptr,INTER,1e-6f,lim,a.q+(size_t)R*INTER,a.s+(size_t)R*INTER/32,bf(a,R),bq(a,R),zl);
   }
  };
  launch(false);launch(true);CK(hipGetLastError());CK(hipDeviceSynchronize());equal(a.q,b.q,N);equal(a.s,b.s,N/32*4);equal(a.b,b.b,N*2);equal(a.bq,b.bq,N*2);cases++;
  if(K==6&&mode==2&&lim==7.f){for(bool f:{false,true}){hipEvent_t st,en;CK(hipEventCreate(&st));CK(hipEventCreate(&en));for(int i=0;i<10;i++)launch(f);CK(hipEventRecord(st));for(int i=0;i<400;i++)launch(f);CK(hipEventRecord(en));CK(hipEventSynchronize(en));float ms;CK(hipEventElapsedTime(&ms,st,en));printf("B%d fused%d %.3fus\n",B,f,ms*1000/400);hipEventDestroy(st);hipEventDestroy(en);}}
  hipFree(dx);hipFree(du);hipFree(dw);
 }
 printf("PASS %d bitwise cases\n",cases);
}
