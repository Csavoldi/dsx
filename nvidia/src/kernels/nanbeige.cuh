// Nanbeige dense Llama path. Independent of DSX's MoE/MLA kernels.
#include <cuda_fp4.h>
#include <cuda_fp8.h>
__device__ __forceinline__ size_t nb_sf(int r,int c,int kc){return ((size_t)(r/128)*(kc/4)+c/4)*512+(r%32)*16+((r%128)/32)*4+c%4;}
extern "C" __global__ void nb_swizzle(const uint8_t* in,uint8_t* out,int m,int kc){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m*kc)out[nb_sf(i/kc,i%kc,kc)]=in[i];}
extern "C" __global__ void nb_embed(const __nv_bfloat16* e,const int* ids,float* x,int d,int t){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<t*d)x[i]=__bfloat162float(e[(size_t)ids[i/d]*d+i%d]);}
extern "C" __global__ void nb_norm(const float* x,const __nv_bfloat16* w,__nv_bfloat16* y,int d){int t=blockIdx.x;float s=0;for(int i=threadIdx.x;i<d;i+=blockDim.x){float v=x[t*d+i];s+=v*v;}__shared__ float sh[32];s=rsqrtf(block_sum(s,sh)/d+1.e-5f);for(int i=threadIdx.x;i<d;i+=blockDim.x)y[t*d+i]=__float2bfloat16(x[t*d+i]*s*__bfloat162float(w[i]));}
extern "C" __global__ void nb_quant(const __nv_bfloat16* x,uint8_t* q,uint8_t* sf,int k,int rows,int padded,float gs){
 int group=blockIdx.x*blockDim.x+threadIdx.x;int kc=k/16;if(group>=padded*kc)return;int row=group/kc,c=group%kc;float v[16],a=0;
 #pragma unroll
 for(int i=0;i<16;i++){v[i]=row<rows?__bfloat162float(x[(size_t)row*k+c*16+i]):0;a=fmaxf(a,fabsf(v[i]));}
 __nv_fp8_e4m3 sc(fminf(448.f,a*gs/6.f));float scale=(float)sc;sf[nb_sf(row,c,kc)]=sc.__x;float inv=scale>0?gs/scale:0;
 #pragma unroll
 for(int i=0;i<8;i++)q[(size_t)row*k/2+c*8+i]=__nv_cvt_float2_to_fp4x2(make_float2(v[2*i]*inv,v[2*i+1]*inv),__NV_E2M1,cudaRoundNearest);
}
extern "C" __global__ void nb_bf2f(const __nv_bfloat16* in,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__bfloat162float(in[i]);}
extern "C" __global__ void nb_add(float* x,const float* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=bf16r(x[i]+y[i]);}
extern "C" __global__ void nb_silu(const float* g,const float* u,__nv_bfloat16* y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=g[i];y[i]=__float2bfloat16(bf16r(v/(1+expf(-v)))*u[i]);}}
extern "C" __global__ void nb_rope(float* q,float* k,const float* v,__nv_bfloat16* kc,__nv_bfloat16* vc,const int* pos,int ctx,int rows){
 int i=blockIdx.x*blockDim.x+threadIdx.x;int t=i/(56*64),h=(i/64)%56,j=i%64;if(t>=rows)return;int p=*pos+t;float a=p*powf(120000000.f,-2.f*j/128.f),c=cosf(a),s=sinf(a);int b=t*7168+h*128;
 float x=q[b+j],y=q[b+j+64];q[b+j]=bf16r(x*c-y*s);q[b+j+64]=bf16r(y*c+x*s);
 if(h<8){b=t*1024+h*128;size_t dst=((size_t)h*ctx+p)*128; x=k[b+j];y=k[b+j+64];kc[dst+j]=__float2bfloat16(x*c-y*s);kc[dst+j+64]=__float2bfloat16(y*c+x*s);vc[dst+j]=__float2bfloat16(v[b+j]);vc[dst+j+64]=__float2bfloat16(v[b+j+64]);}
}
// Each warp computes online softmax over one 128-key partition. Partials merge independently.
extern "C" __global__ void nb_attn_part(const float* q,const __nv_bfloat16* kc,const __nv_bfloat16* vc,float* tmp,const int* pos,int ctx,int rows,int splits){
 int lane=threadIdx.x%32,w=threadIdx.x/32;int head=blockIdx.x*4+w;int t=blockIdx.y,part=blockIdx.z;if(head>=56||t>=rows)return;int end=min((*pos+t+1),(part+1)*128);int begin=part*128;float qq[4],acc[4]={0,0,0,0};
 #pragma unroll
 for(int j=0;j<4;j++)qq[j]=q[t*7168+head*128+lane+j*32];float mx=-INFINITY,den=0;
 for(int p=begin;p<end;p++){size_t off=((size_t)(head/7)*ctx+p)*128+lane;float dot=0;
 #pragma unroll
 for(int j=0;j<4;j++)dot+=qq[j]*__bfloat162float(kc[off+j*32]);dot=warp_sum(dot)*(1.f/sqrtf(128.f));float nm=fmaxf(mx,dot),a=expf(mx-nm),b=expf(dot-nm);den=den*a+b;
 #pragma unroll
 for(int j=0;j<4;j++)acc[j]=acc[j]*a+b*__bfloat162float(vc[off+j*32]);mx=nm;}
 size_t o=(((size_t)t*56+head)*splits+part)*130;if(lane==0){tmp[o]=mx;tmp[o+1]=den;}
 #pragma unroll
 for(int j=0;j<4;j++)tmp[o+2+lane+j*32]=acc[j];
}
extern "C" __global__ void nb_attn_merge(const float* tmp,__nv_bfloat16* out,int rows,int splits){int head=blockIdx.x,t=blockIdx.y,lane=threadIdx.x;float mx=-INFINITY;size_t base=((size_t)t*56+head)*splits*130;for(int s=0;s<splits;s++)mx=fmaxf(mx,tmp[base+s*130]);float sum=0,a=0;for(int s=0;s<splits;s++){float f=expf(tmp[base+s*130]-mx);sum+=f*tmp[base+s*130+1];a+=f*tmp[base+s*130+2+lane];}out[t*7168+head*128+lane]=__float2bfloat16(a/sum);}
extern "C" __global__ void nb_gemv(const uint8_t* w,const uint8_t* scales,const __nv_bfloat16* x,float* y,int m,int k,float inv){
 int r=blockIdx.x*4+threadIdx.x/32,l=threadIdx.x%32;if(r>=m)return;float sum=0;for(int i=l;i<k/2;i+=32){uint8_t a=w[(size_t)r*k/2+i];float sc=(float)(*reinterpret_cast<const __nv_fp8_e4m3*>(scales+nb_sf(r,i/8,k/16)))*inv;__half2 v=__nv_cvt_fp4x2_to_halfraw2(a,__NV_E2M1);sum+=sc*(__half2float(__low2half(v))*__bfloat162float(x[i*2])+__half2float(__high2half(v))*__bfloat162float(x[i*2+1]));}sum=warp_sum(sum);if(l==0)y[r]=bf16r(sum);
}
extern "C" __global__ void nb_unpack_activation(const uint8_t* q,const uint8_t* sf,__nv_bfloat16* x,int k){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=k/16)return;float scale=(float)(*reinterpret_cast<const __nv_fp8_e4m3*>(sf+nb_sf(0,c,k/16)));for(int j=0;j<8;j++){__half2 h=__nv_cvt_fp4x2_to_halfraw2(q[c*8+j],__NV_E2M1);x[c*16+2*j]=__float2bfloat16(scale*__half2float(__low2half(h)));x[c*16+2*j+1]=__float2bfloat16(scale*__half2float(__high2half(h)));}}
extern "C" __global__ void nb_gemv_fast(const uint8_t* w,const uint8_t* sf,const __nv_bfloat16* x,float* y,int m,int k,float alpha){
 int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(row>=m)return;float sum=0;int kc=k/16;
 for(int c=lane;c<kc;c+=32){uint64_t bits=*(const uint64_t*)(w+(size_t)row*k/2+c*8);float d=(float)(*reinterpret_cast<const __nv_fp8_e4m3*>(sf+nb_sf(row,c,kc)));float a=0;
 #pragma unroll
 for(int j=0;j<8;j++){__half2 h=__nv_cvt_fp4x2_to_halfraw2((bits>>(j*8))&255,__NV_E2M1);__nv_bfloat162 v=*(const __nv_bfloat162*)(x+c*16+j*2);float2 xx=__bfloat1622float2(v);a+=__half2float(__low2half(h))*xx.x+__half2float(__high2half(h))*xx.y;}sum+=d*a;
 }sum=warp_sum(sum);if(lane==0)y[row]=bf16r(sum*alpha);
}
