#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include "../kernels/reject.h"
#define CK(x) do{auto e=(x);if(e!=hipSuccess){fprintf(stderr,"line%d %s\n",__LINE__,hipGetErrorString(e));exit(2);}}while(0)
template<class T>T* gpu(size_t n){T*p;CK(hipMalloc(&p,n*sizeof(T)));return p;}
std::vector<long double> distribution(const float*l,int n,float T){float mx=-INFINITY;for(int i=0;i<n;i++)mx=fmaxf(mx,l[i]);std::vector<long double>w(n);long double sum=0;for(int i=0;i<n;i++)if(l[i]>=mx-30.f*T){w[i]=exp((double)(l[i]-mx)/(double)T);sum+=w[i];}for(auto&x:w)x/=sum;return w;}
int sample(const std::vector<long double>&w,double u){long double total=0;for(auto x:w)total+=x;long double r=(long double)u*total;int last=0;for(int tid=0;tid<1024;tid++)for(int i=tid;i<(int)w.size();i+=1024)if(w[i]>0){last=i;r-=w[i];if(r<0)return i;}return last;}
int reference(const float*p,const float*q,int n,int greedy,RejectDraw j){
 if(!j.active||!std::isfinite(p[greedy]))return greedy;
 auto pw=distribution(p,n,j.temperature);
 if(j.qrow>=0){float mx=-INFINITY;for(int i=0;i<n;i++)mx=fmaxf(mx,q[i]);if(std::isfinite(mx)){
  auto qw=distribution(q,n,j.temperature);auto pd=pw[j.draft],qd=qw[j.draft];
  if(qd>0){if(pd>=qd||j.accept_uniform<pd/qd)return j.draft;for(int i=0;i<n;i++)pw[i]=std::max((long double)0,pw[i]-qw[i]);}
 }}return sample(pw,j.recover_uniform);
}
int main(){
 const int B=32,MAXN=129280;std::mt19937 rng(781);std::normal_distribution<float>norm(0,6);std::uniform_real_distribution<double>u(0,1);
 float*dp=gpu<float>((size_t)B*MAXN),*dq=gpu<float>((size_t)B*MAXN);int*dg=gpu<int>(B),*dout=gpu<int>(B);RejectDraw*dj=gpu<RejectDraw>(B);
 int cases=0,diff=0;
 for(int N:{16,4096,MAXN})for(int mode=0;mode<4;mode++)for(float T:{.7f,1.f,2.f}){
  std::vector<float>p((size_t)B*N),q(p.size());std::vector<int>g(B),out(B);std::vector<RejectDraw>jobs(B);
  for(int b=0;b<B;b++){
   float*pl=p.data()+(size_t)b*N;float*ql=q.data()+(size_t)b*N;
   for(int i=0;i<N;i++){pl[i]=norm(rng);ql[i]=mode==0?pl[i]:pl[i]+(float)(mode*.7)*norm(rng);}
   g[b]=std::max_element(pl,pl+N)-pl;
   auto qw=distribution(ql,N,T);int d=sample(qw,u(rng));
   jobs[b]={T,b%13!=12,mode==3?-1:b,d,b==0?0.:b==1?std::nextafter(1.,0.):u(rng),b==2?0.:b==3?std::nextafter(1.,0.):u(rng)};
  }
  CK(hipMemcpy(dp,p.data(),p.size()*4,hipMemcpyHostToDevice));CK(hipMemcpy(dq,q.data(),q.size()*4,hipMemcpyHostToDevice));CK(hipMemcpy(dg,g.data(),B*4,hipMemcpyHostToDevice));CK(hipMemcpy(dj,jobs.data(),B*sizeof(RejectDraw),hipMemcpyHostToDevice));
  hipLaunchKernelGGL(reject_logits_b,dim3(B),dim3(1024),0,0,dp,dq,N,dj,dg,dout);CK(hipGetLastError());CK(hipDeviceSynchronize());CK(hipMemcpy(out.data(),dout,B*4,hipMemcpyDeviceToHost));
  for(int b=0;b<B;b++){int want=reference(p.data()+(size_t)b*N,q.data()+(size_t)b*N,N,g[b],jobs[b]);cases++;if(want!=out[b]){diff++;printf("DIFF N%d mode%d T%.1f b%d want%d got%d accept%.17g recover%.17g\n",N,mode,T,b,want,out[b],jobs[b].accept_uniform,jobs[b].recover_uniform);}}
  printf("check N%d mode%d T%.1f cases%d diff%d\n",N,mode,T,cases,diff);fflush(stdout);
 }
 // End-to-end proposal/rejection histogram, with independent proposal/accept/recover draws.
 const int N=16,rounds=4000;std::vector<float>p(B*N,-INFINITY),q(B*N,-INFINITY);std::vector<int>g(B,0),out(B);std::vector<RejectDraw>jobs(B);long hist[3]={},accepted=0;
 for(int b=0;b<B;b++){p[b*N]=log(.6);p[b*N+1]=log(.3);p[b*N+2]=log(.1);q[b*N]=log(.1);q[b*N+1]=log(.3);q[b*N+2]=log(.6);}
 CK(hipMemcpy(dp,p.data(),p.size()*4,hipMemcpyHostToDevice));CK(hipMemcpy(dq,q.data(),q.size()*4,hipMemcpyHostToDevice));CK(hipMemcpy(dg,g.data(),B*4,hipMemcpyHostToDevice));
 for(int r=0;r<rounds;r++){
  for(int b=0;b<B;b++){double v=u(rng);int d=v<.1?0:v<.4?1:2;jobs[b]={1.f,1,b,d,u(rng),u(rng)};}
  CK(hipMemcpy(dj,jobs.data(),B*sizeof(RejectDraw),hipMemcpyHostToDevice));hipLaunchKernelGGL(reject_logits_b,dim3(B),dim3(1024),0,0,dp,dq,N,dj,dg,dout);CK(hipMemcpy(out.data(),dout,B*4,hipMemcpyDeviceToHost));
  for(int b=0;b<B;b++){if(out[b]<0||out[b]>2){fprintf(stderr,"out of support\n");return 4;}hist[out[b]]++;accepted+=(out[b]==jobs[b].draft);}
 }
 const double total=B*rounds;printf("HIST draws%.0f p=[%.6f,%.6f,%.6f] acceptance%.6f expected[.6,.3,.1] acceptance.5\n",total,hist[0]/total,hist[1]/total,hist[2]/total,accepted/total);
 for(int i=0;i<3;i++)if(fabs(hist[i]/total-(i==0?.6:i==1?.3:.1))>.006)diff++;
 printf("FINAL cases%d failures%d\n",cases,diff);return diff?3:0;
}
