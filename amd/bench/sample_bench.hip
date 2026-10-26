#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include "../kernels/sample.h"
#define CK(x) do {auto e=(x);if(e!=hipSuccess){fprintf(stderr,"line%d: %s\n",__LINE__,hipGetErrorString(e));exit(2);}}while(0)
template<class T>T* gpu(size_t n){T*p;CK(hipMalloc(&p,n*sizeof(T)));return p;}
int reference(const float* l,int n,int greedy,SampleDraw draw){
 if(!draw.active||!std::isfinite(l[greedy]))return greedy;
 float mx=l[greedy],th=mx-30.f*draw.temperature;double inv=1./draw.temperature;
 std::vector<long double> masses(1024);for(int tid=0;tid<1024;tid++)for(int i=tid;i<n;i+=1024)if(l[i]>=th)masses[tid]+=std::exp((double)(l[i]-mx)*inv);
 long double total=0;for(auto v:masses)total+=v;long double r=(long double)draw.uniform*total;int chosen=1023;
 for(int tid=0;tid<1024;tid++){if(r<masses[tid]){chosen=tid;break;}r-=masses[tid];}
 int last=greedy;for(int i=chosen;i<n;i+=1024)if(l[i]>=th){last=i;r-=std::exp((double)(l[i]-mx)*inv);if(r<0)return i;}return last;
}
double quantile_error(const float* l,int n,int greedy,SampleDraw draw,int got){
 if(got<0||got>=n)return INFINITY;
 float mx=l[greedy],th=mx-30.f*draw.temperature;double inv=1./draw.temperature;
 long double total=0,lo=0,hi=0;bool found=false;
 for(int tid=0;tid<1024;tid++)for(int i=tid;i<n;i+=1024)if(l[i]>=th){
  double w=std::exp((double)(l[i]-mx)*inv);
  if(i==got){lo=total;hi=total+w;found=true;}total+=w;
 }
 if(!found||total<=0)return INFINITY;
 long double u=draw.uniform;return (double)std::max((long double)0,std::max(lo/total-u,u-hi/total));
}
int main(){
 const int N=129280,B=32;std::mt19937 rng(792);std::normal_distribution<float> normal(0,6);std::uniform_real_distribution<double> uniform(0,1);
 std::vector<float> lg((size_t)N*B);std::vector<int> best(B),actual(B);std::vector<SampleDraw> draws(B);
 float*dlog=gpu<float>((size_t)N*B);int*dbest=gpu<int>(B),*dout=gpu<int>(B);SampleDraw*dd=gpu<SampleDraw>(B);
 int cases=0,diff=0,boundaries=0;double max_quantile_error=0.;
 for(int mode=0;mode<6;mode++)for(float T:{.1f,.7f,1.f,2.f}){
  for(int b=0;b<B;b++){
   float* l=lg.data()+(size_t)b*N;for(int i=0;i<N;i++)l[i]=mode==0?0.f:mode==1?-60.f:normal(rng);
   if(mode==1)l[(b*719)%N]=1.f;
   if(mode==3)for(int i=0;i<N;i+=31)l[i]=-INFINITY;
   if(mode==4)for(int i=0;i<N;i+=17)l[i]=NAN;
   if(mode==5)l[b*17]=INFINITY;
   int ix=0;float mx=-INFINITY;for(int i=0;i<N;i++)if(l[i]>mx){mx=l[i];ix=i;}best[b]=ix;
   double u=b==0?0.:b==1?std::nextafter(1.,0.):b==2?1e-15:uniform(rng);
   draws[b]={T,b%11!=10,u};
  }
  CK(hipMemcpy(dlog,lg.data(),lg.size()*4,hipMemcpyHostToDevice));CK(hipMemcpy(dbest,best.data(),B*4,hipMemcpyHostToDevice));CK(hipMemcpy(dd,draws.data(),B*sizeof(SampleDraw),hipMemcpyHostToDevice));
  hipLaunchKernelGGL(sample_logits_b,dim3(B),dim3(1024),0,0,dlog,N,dd,dbest,dout);CK(hipGetLastError());CK(hipDeviceSynchronize());CK(hipMemcpy(actual.data(),dout,B*4,hipMemcpyDeviceToHost));
  for(int b=0;b<B;b++){int want=reference(lg.data()+(size_t)b*N,N,best[b],draws[b]);cases++;if(want!=actual[b]){double error=quantile_error(lg.data()+(size_t)b*N,N,best[b],draws[b],actual[b]);max_quantile_error=std::max(max_quantile_error,error);if(error>2e-12)diff++;else boundaries++;printf("CDF_BOUNDARY mode%d T%.1f b%d u%.17g CPU%d GPU%d quantile_error%.3g\n",mode,T,b,draws[b].uniform,want,actual[b],error);}}
  printf("check mode%d T%.1f cases%d differences%d\n",mode,T,cases,diff);fflush(stdout);
 }
 for(int b:{1,4,8,32}){
  hipEvent_t e0,e1;CK(hipEventCreate(&e0));CK(hipEventCreate(&e1));
  // Fresh finite logits for timing, independent of the final nonfinite test.
  for(auto&v:lg)v=normal(rng);for(int i=0;i<B;i++){best[i]=std::max_element(lg.begin()+(size_t)i*N,lg.begin()+(size_t)(i+1)*N)-(lg.begin()+(size_t)i*N);draws[i]={1.f,1,uniform(rng)};}
  CK(hipMemcpy(dlog,lg.data(),lg.size()*4,hipMemcpyHostToDevice));CK(hipMemcpy(dbest,best.data(),B*4,hipMemcpyHostToDevice));CK(hipMemcpy(dd,draws.data(),B*sizeof(SampleDraw),hipMemcpyHostToDevice));
  for(int r=0;r<3;r++){
   CK(hipEventRecord(e0));for(int i=0;i<100;i++)hipLaunchKernelGGL(sample_logits_b,dim3(b),dim3(1024),0,0,dlog,N,dd,dbest,dout);CK(hipEventRecord(e1));CK(hipEventSynchronize(e1));float ms;CK(hipEventElapsedTime(&ms,e0,e1));printf("TIME B%d round%d us%.2f\n",b,r,ms*10);
  }CK(hipEventDestroy(e0));CK(hipEventDestroy(e1));
 }
 printf("FINAL cases%d failures%d boundary_differences%d max_quantile_error%.3g\n",cases,diff,boundaries,max_quantile_error);return diff?3:0;
}
