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
// Per-head orthogonal rotation + 3-bit Lloyd-Max codec. Not the full QJL TurboQuant algorithm.
__device__ __forceinline__ float nb_centroid(int code){int i=code<4?code:7-code;float v=i==0?2.1519457f:i==1?1.3439093f:i==2?.7560053f:.2450942f;return code<4?-v:v;}
__device__ __forceinline__ int nb_code(float x){return x<0?(x<-1.7479275f?0:x<-1.0499573f?1:x<-.50054975f?2:3):(x<.50054975f?4:x<1.0499573f?5:x<1.7479275f?6:7);}
__device__ __forceinline__ float nb_sign(int i){return (__popc((unsigned)(i*0x9e3779b9u)^0x3a2f91abu)&1)?-1.f:1.f;}
__device__ void nb_hadamard(float* a){int j=threadIdx.x;for(int bit=1;bit<128;bit*=2){float x=a[j],y=a[j^bit];__syncthreads();a[j]=(j&bit)?y-x:x+y;__syncthreads();}}
extern "C" __global__ void nb_rope_q3(float* q,const float* k,const float* v,uint8_t* kc,uint8_t* vc,__nv_bfloat16* kr,__nv_bfloat16* vr,const int* pos,int ctx,int rows){
 int h=blockIdx.x,t=blockIdx.y,j=threadIdx.x,p=*pos+t;__shared__ float a[128],b[128],c[128],reduce[32];
 int x=j%64;float angle=p*powf(120000000.f,-2.f*x/128.f),co=cosf(angle),si=sinf(angle);int base=t*7168+h*128;float lo=q[base+x],hi=q[base+x+64];a[j]=bf16r((j<64?lo*co-hi*si:hi*co+lo*si))*nb_sign(j);
 if(h<8){base=t*1024+h*128;lo=k[base+x];hi=k[base+x+64];b[j]=bf16r((j<64?lo*co-hi*si:hi*co+lo*si))*nb_sign(j);c[j]=v[base+j]*nb_sign(j);}__syncthreads();nb_hadamard(a);q[t*7168+h*128+j]=bf16r(a[j]*.08838834765f);
 if(h>=8)return;nb_hadamard(b);nb_hadamard(c);float kk=b[j]*.08838834765f,vv=c[j]*.08838834765f;size_t ring=((size_t)h*2592+p%2592)*128+j;kr[ring]=__float2bfloat16(kk);vr[ring]=__float2bfloat16(vv);
 float kn=sqrtf(block_sum(kk*kk,reduce)/128.f),vn=sqrtf(block_sum(vv*vv,reduce)/128.f);int ki=nb_code(kk/fmaxf(kn,1e-30f)),vi=nb_code(vv/fmaxf(vn,1e-30f));
 // Norm correction keeps the reconstructed vector norm equal to the input norm.
 float kcn=block_sum(nb_centroid(ki)*nb_centroid(ki),reduce),vcn=block_sum(nb_centroid(vi)*nb_centroid(vi),reduce);kn*=sqrtf(128.f/kcn);vn*=sqrtf(128.f/vcn);
 __shared__ uint8_t ki_sh[128],vi_sh[128];ki_sh[j]=ki;vi_sh[j]=vi;__syncthreads();size_t out=((size_t)h*ctx+p)*50;
 if(j==0){*reinterpret_cast<__nv_bfloat16*>(kc+out)=__float2bfloat16(kn);*reinterpret_cast<__nv_bfloat16*>(vc+out)=__float2bfloat16(vn);}
 if(j<16){unsigned ka=0,va=0;for(int z=0;z<8;z++){ka|=(unsigned)ki_sh[j*8+z]<<(z*3);va|=(unsigned)vi_sh[j*8+z]<<(z*3);}for(int z=0;z<3;z++){kc[out+2+j*3+z]=(ka>>(z*8))&255;vc[out+2+j*3+z]=(va>>(z*8))&255;}}
}
__device__ __forceinline__ float nb_load_q3(const uint8_t* p,int j){int group=j/8;unsigned bits=(unsigned)p[2+group*3]|((unsigned)p[3+group*3]<<8)|((unsigned)p[4+group*3]<<16);return bf16r(nb_centroid((bits>>(j%8*3))&7)*__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(p)));}
extern "C" __global__ void nb_attn_q3(const float* q,const uint8_t* kc,const uint8_t* vc,const __nv_bfloat16* kr,const __nv_bfloat16* vr,float* tmp,const int* pos,int ctx,int rows,int splits,int part_size){
 int lane=threadIdx.x%32,w=threadIdx.x/32,head=blockIdx.x*4+w,t=blockIdx.y,part=blockIdx.z;if(head>=56||t>=rows)return;int end=min(*pos+t+1,(part+1)*part_size),begin=part*part_size;float qq[4],acc[4]={0,0,0,0};
 #pragma unroll
 for(int j=0;j<4;j++)qq[j]=q[t*7168+head*128+lane+j*32];float mx=-INFINITY,den=0;
 for(int p=begin;p<end;p++){size_t off=((size_t)(head/7)*ctx+p)*50;size_t ring=((size_t)(head/7)*2592+p%2592)*128+lane;bool recent=p>=max(0,((*pos+t)/32)*32-2048);float dot=0;
 #pragma unroll
 for(int j=0;j<4;j++)dot+=qq[j]*(recent?__bfloat162float(kr[ring+j*32]):nb_load_q3(kc+off,lane+j*32));dot=warp_sum(dot)*.08838834765f;float nm=fmaxf(mx,dot),a=expf(mx-nm),b=expf(dot-nm);den=den*a+b;
 #pragma unroll
 for(int j=0;j<4;j++)acc[j]=acc[j]*a+b*(recent?__bfloat162float(vr[ring+j*32]):nb_load_q3(vc+off,lane+j*32));mx=nm;}
 size_t o=(((size_t)t*56+head)*splits+part)*130;if(lane==0){tmp[o]=mx;tmp[o+1]=den;}
 #pragma unroll
 for(int j=0;j<4;j++)tmp[o+2+lane+j*32]=acc[j];
}
extern "C" __global__ void nb_unrotate(__nv_bfloat16* out,int rows){int h=blockIdx.x,t=blockIdx.y,j=threadIdx.x;__shared__ float a[128];a[j]=__bfloat162float(out[t*7168+h*128+j]);__syncthreads();nb_hadamard(a);out[t*7168+h*128+j]=__float2bfloat16(a[j]*.08838834765f*nb_sign(j));}
extern "C" __global__ void nb_quant_head(const __nv_bfloat16* w,int8_t* q,__nv_bfloat16* scales,int groups){int group=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(group>=groups)return;float a=__bfloat162float(w[(size_t)group*64+lane*2]),b=__bfloat162float(w[(size_t)group*64+lane*2+1]);float mx=warp_max(fmaxf(fabsf(a),fabsf(b)));float d=__bfloat162float(__float2bfloat16(mx>0?mx/127.f:1.f));if(lane==0)scales[group]=__float2bfloat16(d);q[(size_t)group*64+lane*2]=(int8_t)max(-127,min(127,__float2int_rn(a/d)));q[(size_t)group*64+lane*2+1]=(int8_t)max(-127,min(127,__float2int_rn(b/d)));}
extern "C" __global__ void nb_head_q8(const int8_t* w,const __nv_bfloat16* scales,const __nv_bfloat16* x,float* y,int m,int k,int rows){int r=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32,t=blockIdx.y;if(r>=m||t>=rows)return;float sum=0;for(int c=lane;c<k/64;c+=32){const int8_t* p=w+(size_t)r*k+c*64;float dot=0;
 #pragma unroll
 for(int j=0;j<64;j++)dot+=(float)p[j]*__bfloat162float(x[(size_t)t*k+c*64+j]);sum+=dot*__bfloat162float(scales[(size_t)r*(k/64)+c]);}sum=warp_sum(sum);if(lane==0)y[(size_t)t*m+r]=sum;}
#include <mma.h>
// Dense prefill attention: BF16 Tensor Core QK, two-component BF16 softmax-P times V.
// Each CTA computes 32 queries for one head over a 2048-key partition.
extern "C" __global__ void nb_attn_q3_tc(const float* q,const uint8_t* kc,const uint8_t* vc,const __nv_bfloat16* kr,const __nv_bfloat16* vr,float* tmp,const int* pos,int ctx,int rows,int splits,int query_offset,int part_size,int full_rows){
 using namespace nvcuda;extern __shared__ __align__(32) unsigned char memory[];
 auto Q=(__nv_bfloat16*)memory;auto K=Q+32*144;auto V=K;auto S=(float*)(K+64*144);auto PH=(__nv_bfloat16*)(S+32*80);auto PL=PH+32*80;auto O=(float*)K;auto mx=(float*)(PL+32*80);auto den=mx+32;auto old=den+32;
 int tile=blockIdx.z; q+=(size_t)tile*32*7168; rows=min(32,rows-tile*32); query_offset+=tile*32;
 int h=blockIdx.x,part=blockIdx.y,tid=threadIdx.x,warp=tid/32,lane=tid%32,p0=*pos+query_offset,cutoff=max(max(0,(p0/32)*32-2048),*pos+full_rows-2592);
 for(int i=tid;i<32*128;i+=256){Q[(i/128)*144+i%128]=__float2bfloat16(i/128<rows?q[(size_t)(i/128)*7168+h*128+i%128]:0);}if(tid<32){mx[tid]=-INFINITY;den[tid]=0;}__syncthreads();
 wmma::fragment<wmma::accumulator,16,16,16,float> value[2]; for(int row=0;row<2;row++)wmma::fill_fragment(value[row],0.f);
 for(int begin=part*part_size;begin<min(p0+rows,(part+1)*part_size);begin+=64){
  for(int i=tid;i<64*128;i+=256){int p=begin+i/128,j=i%128;size_t off=((size_t)(h/7)*ctx+p)*50,ring=((size_t)(h/7)*2592+p%2592)*128+j;float kval=0;if(p<p0+rows){bool recent=p>=cutoff;kval=recent?__bfloat162float(kr[ring]):nb_load_q3(kc+off,j);}K[(i/128)*144+i%128]=__float2bfloat16(kval);}__syncthreads();
  wmma::fragment<wmma::accumulator,16,16,16,float> scores;wmma::fill_fragment(scores,0.f);int qr=warp/4,kcol=warp%4;
  #pragma unroll
  for(int d=0;d<128;d+=16){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> qa;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> kb;wmma::load_matrix_sync(qa,Q+qr*16*144+d,144);wmma::load_matrix_sync(kb,K+kcol*16*144+d,144);wmma::mma_sync(scores,qa,kb,scores);}wmma::store_matrix_sync(S+qr*16*80+kcol*16,scores,80,wmma::mem_row_major);__syncthreads();
  for(int r=warp;r<32;r+=8){float a=-INFINITY;for(int j=lane;j<64;j+=32){float v=(r<rows&&begin+j<=p0+r)?S[r*80+j]*.08838834765f:-INFINITY;S[r*80+j]=v;a=fmaxf(a,v);}a=warp_max(a);float nm=fmaxf(mx[r],a);float prior=isfinite(nm)?expf(mx[r]-nm):0;float sum=0;for(int j=lane;j<64;j+=32){float v=isfinite(nm)?expf(S[r*80+j]-nm):0;__nv_bfloat16 hi=__float2bfloat16(v);PH[r*80+j]=hi;PL[r*80+j]=__float2bfloat16(v-__bfloat162float(hi));sum+=v;}sum=warp_sum(sum);if(lane==0){old[r]=prior;den[r]=den[r]*prior+sum;mx[r]=nm;}}
  __syncthreads();
  // K is dead after QK. Reuse its storage for V to allow two CTAs per SM.
  for(int i=tid;i<64*128;i+=256){int p=begin+i/128,j=i%128;float val=0;if(p<p0+rows){size_t off=((size_t)(h/7)*ctx+p)*50,ring=((size_t)(h/7)*2592+p%2592)*128+j;val=p>=cutoff?__bfloat162float(vr[ring]):nb_load_q3(vc+off,j);}V[(i/128)*144+i%128]=__float2bfloat16(val);}__syncthreads();
  #pragma unroll
  for(int row=0;row<2;row++){
   // SM120 WMMA accumulator mapping, checked against load_matrix_sync on this GPU.
   for(int e=0;e<value[row].num_elements;e++)value[row].x[e]*=old[row*16+lane/4+((e/2)%2)*8];
   #pragma unroll
   for(int d=0;d<64;d+=16){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> pa;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> vb;wmma::load_matrix_sync(vb,V+d*144+warp*16,144);wmma::load_matrix_sync(pa,PH+row*16*80+d,80);wmma::mma_sync(value[row],pa,vb,value[row]);wmma::load_matrix_sync(pa,PL+row*16*80+d,80);wmma::mma_sync(value[row],pa,vb,value[row]);}}
  __syncthreads();
 }
 for(int row=0;row<2;row++)wmma::store_matrix_sync(O+row*16*136+warp*16,value[row],136,wmma::mem_row_major); __syncthreads();
 for(int i=tid;i<rows*128;i+=256){int r=i/128,j=i%128;size_t off=(((size_t)(tile*32+r)*56+h)*splits+part)*130;tmp[off+2+j]=O[r*136+j];if(j==0){tmp[off]=mx[r];tmp[off+1]=den[r];}}
}
extern "C" __global__ void nb_quant_head6(const __nv_bfloat16* w,uint8_t* q,__nv_bfloat16* scales,int groups){int group=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(group>=groups)return;float a=__bfloat162float(w[(size_t)group*64+lane*2]),b=__bfloat162float(w[(size_t)group*64+lane*2+1]);float mx=warp_max(fmaxf(fabsf(a),fabsf(b)));float d=__bfloat162float(__float2bfloat16(mx>0?mx/31.f:1.f));if(lane==0)scales[group]=__float2bfloat16(d);unsigned qa=max(-31,min(31,__float2int_rn(a/d)))+32,qb=max(-31,min(31,__float2int_rn(b/d)))+32;unsigned pair=qa|(qb<<6);unsigned next=__shfl_down_sync(0xffffffff,pair,1);if((lane&1)==0){unsigned bits=pair|(next<<12);size_t off=(size_t)group*48+(lane/2)*3;q[off]=bits&255;q[off+1]=(bits>>8)&255;q[off+2]=(bits>>16)&255;}}
extern "C" __global__ void nb_head_int(const uint8_t* w,const __nv_bfloat16* scales,const int8_t* x,const __nv_bfloat16* xs,float* y,int m,int k,int rows,int bits){int r=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(r>=m)return;float sum[9]={0};for(int c=0;c<k/128;c++){int off=c*128+lane*4;unsigned packed;if(bits==8){packed=*(const unsigned*)(w+(size_t)r*k+off);}else if(bits==4){unsigned v=*(const uint16_t*)(w+((size_t)r*k+off)/2);packed=0;for(int j=0;j<4;j++)packed|=((((v>>(j*4))&15)-8)&255)<<(j*8);}else{size_t b=((size_t)r*k+off)*3/4;unsigned v=(unsigned)w[b]|((unsigned)w[b+1]<<8)|((unsigned)w[b+2]<<16);packed=((v&63)-32)&255;packed|=((((v>>6)&63)-32)&255)<<8;packed|=((((v>>12)&63)-32)&255)<<16;packed|=((((v>>18)&63)-32)&255)<<24;}int group=off/64;float ws=__bfloat162float(scales[(size_t)r*(k/64)+group]);
 #pragma unroll
 for(int t=0;t<9;t++)if(t<rows){int xx=*(const int*)(x+(size_t)t*k+off);int dot=__dp4a((int)packed,xx,0);sum[t]+=(float)dot*ws*__bfloat162float(xs[(size_t)t*(k/64)+group]);}}
 #pragma unroll
 for(int t=0;t<9;t++)if(t<rows){float a=warp_sum(sum[t]);if(lane==0)y[(size_t)t*m+r]=a;}}
extern "C" __global__ void nb_attn_q3_parallel(const float* q,const uint8_t* kc,const uint8_t* vc,const __nv_bfloat16* kr,const __nv_bfloat16* vr,float* tmp,const int* pos,int ctx,int rows,int splits,int part_size){
 int h=blockIdx.x,t=blockIdx.y,part=blockIdx.z,tid=threadIdx.x,lane=tid%32,warp=tid/32,pq=*pos+t,begin=part*128,cutoff=max(0,(pq/32)*32-2048);__shared__ float scores[128],sh[32],values[256];float qq[4];
 #pragma unroll
 for(int j=0;j<4;j++)qq[j]=q[(size_t)t*7168+h*128+lane+j*32];
 for(int off=warp;off<128;off+=8){int p=begin+off;float dot=0;if(p<=pq){size_t a=((size_t)(h/7)*ctx+p)*50,b=((size_t)(h/7)*2592+p%2592)*128+lane;
 #pragma unroll
 for(int j=0;j<4;j++)dot+=qq[j]*(p>=cutoff?__bfloat162float(kr[b+j*32]):nb_load_q3(kc+a,lane+j*32));}dot=warp_sum(dot)*.08838834765f;if(lane==0)scores[off]=p<=pq?dot:-INFINITY;}
 __syncthreads();float mx=tid<128?scores[tid]:-INFINITY;mx=warp_max(mx);if(lane==0)sh[warp]=mx;__syncthreads();if(warp==0){mx=lane<8?sh[lane]:-INFINITY;mx=warp_max(mx);if(lane==0)sh[0]=mx;}__syncthreads();mx=sh[0];float v=tid<128&&isfinite(mx)?expf(scores[tid]-mx):0;if(tid<128)scores[tid]=v;float den=block_sum(v,sh);int feature=tid%128;float acc=0;
 for(int off=(tid/128)*64;off<(tid/128+1)*64;off++){int p=begin+off;float weight=scores[off];if(weight>0){size_t a=((size_t)(h/7)*ctx+p)*50,b=((size_t)(h/7)*2592+p%2592)*128+feature;acc+=weight*(p>=cutoff?__bfloat162float(vr[b]):nb_load_q3(vc+a,feature));}}
 values[tid]=acc;__syncthreads();size_t out=(((size_t)t*56+h)*splits+part)*130;if(tid<128)tmp[out+2+tid]=values[tid]+values[tid+128];if(tid==0){tmp[out]=mx;tmp[out+1]=den;}
}
// Experimental two-bit input-rotated weights. This is scalar quantization, not T16.
// Each 128-weight block uses levels {-3,-1,1,3}, a least-squares scale refit,
// and the same signed orthonormal Hadamard as its input activation.
extern "C" __global__ void nb_q2_weights(const uint8_t* w,const uint8_t* sf,uint8_t* out,uint8_t* scales,int k,float wg){
 int j=threadIdx.x,group=blockIdx.x,row=group/(k/128),g=group%(k/128),c=g*128+j;__shared__ float a[128],sh[32];
 uint8_t p=w[((size_t)row*k+c)/2];__half2 h=__nv_cvt_fp4x2_to_halfraw2(p,__NV_E2M1);float v=__half2float(j%2?__high2half(h):__low2half(h));float sc=(float)(*reinterpret_cast<const __nv_fp8_e4m3*>(sf+(size_t)row*(k/16)+c/16));a[j]=v*sc/wg*nb_sign(j);__syncthreads();nb_hadamard(a);v=a[j]*.08838834765f;
 float d=sqrtf(block_sum(v*v,sh)/128.f)*.498f;int level=(v<0?-1:1)*(fabsf(v)<2*d?1:3);float num=block_sum(v*level,sh),den=block_sum((float)(level*level),sh);__nv_fp8_e4m3 ds(fminf(448.f,num/fmaxf(den,1.f)*wg));if(j==0)scales[group]=ds.__x;
 unsigned code=(level+3)/2,bits=0;for(int q=0;q<4;q++)bits|=__shfl_sync(0xffffffff,code,(j%32&~3)+q)<<(2*q);if(j%4==0)out[(size_t)group*32+j/4]=(uint8_t)bits;
}
extern "C" __global__ void nb_rotate_input(const __nv_bfloat16* in,__nv_bfloat16* out,int k){int j=threadIdx.x;size_t off=(size_t)blockIdx.x*128;__shared__ float a[128];a[j]=__bfloat162float(in[off+j])*nb_sign(j);__syncthreads();nb_hadamard(a);out[off+j]=__float2bfloat16(a[j]*.08838834765f);}
extern "C" __global__ void nb_q2_expand(const uint8_t* in,uint8_t* out,int size){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size)return;unsigned v=in[i],z=0;for(int j=0;j<4;j++)z|=((0x52adu>>(((v>>(j*2))&3)*4))&15)<<(j*4);((uint16_t*)out)[i]=z;}
extern "C" __global__ void nb_q2_expand_scales(const uint8_t* in,uint8_t* out,int m,int k){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m*(k/16))return;int r=i/(k/16),c=i%(k/16);out[nb_sf(r,c,k/16)]=in[(size_t)r*(k/128)+c/8];}
extern "C" __global__ void nb_q2_gemv(const uint8_t* w,const uint8_t* scales,const int8_t* x,const __nv_bfloat16* xs,float* y,int m,int k,float inv){int r=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(r>=m)return;float sum=0;for(int g=0;g<k/128;g++){unsigned a=w[((size_t)r*k/4)+g*32+lane];unsigned z=(a&3)|((a&12)<<6)|((a&48)<<12)|((a&192)<<18);z=(z*2+0x7d7d7d7d)^0x80808080;unsigned xx=*(const unsigned*)(x+g*128+lane*4);float d=(float)(*reinterpret_cast<const __nv_fp8_e4m3*>(scales+(size_t)r*(k/128)+g))*__bfloat162float(xs[g*2+lane/16]);sum+=(float)__dp4a((int)z,(int)xx,0)*d;}sum=warp_sum(sum);if(lane==0)y[r]=bf16r(sum*inv);}

extern "C" __global__ void nb_quant_head4(const __nv_bfloat16* w,uint8_t* q,__nv_bfloat16* scales,int groups){int group=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(group>=groups)return;float a=__bfloat162float(w[(size_t)group*64+lane*2]),b=__bfloat162float(w[(size_t)group*64+lane*2+1]);float mx=warp_max(fmaxf(fabsf(a),fabsf(b)));float d=__bfloat162float(__float2bfloat16(mx>0?mx/7.f:1.f));if(lane==0)scales[group]=__float2bfloat16(d);unsigned qa=max(-7,min(7,__float2int_rn(a/d)))+8,qb=max(-7,min(7,__float2int_rn(b/d)))+8;q[(size_t)group*32+lane]=qa|(qb<<4);}

// GQA-aware attention shares each KV tile across the seven query heads.
extern "C" __global__ void nb_attn_q3_gqa(const float* q,const uint8_t* kc,const uint8_t* vc,const __nv_bfloat16* kr,const __nv_bfloat16* vr,float* tmp,const int* pos,int ctx,int rows,int splits,int query_offset){
 using namespace nvcuda;extern __shared__ __align__(32) unsigned char memory[];
 auto Q=(__nv_bfloat16*)memory;auto K=Q+32*144;auto V=K;auto S=(float*)(K+64*144);auto PH=(__nv_bfloat16*)(S+32*80);auto PL=PH+32*80;auto O=(float*)K;auto mx=(float*)(PL+32*80);auto den=mx+32;auto old=den+32;
 int tile=blockIdx.z; query_offset=tile*4; rows=min(4,rows-query_offset)*7;
 int h=blockIdx.x,part=blockIdx.y,tid=threadIdx.x,warp=tid/32,lane=tid%32,p0=*pos+query_offset,cutoff=max(0,(p0/32)*32-2048);
 for(int i=tid;i<32*128;i+=256){Q[(i/128)*144+i%128]=__float2bfloat16(i/128<rows?q[(size_t)(query_offset+i/128/7)*7168+(h*7+i/128%7)*128+i%128]:0);}if(tid<32){mx[tid]=-INFINITY;den[tid]=0;}__syncthreads();
 wmma::fragment<wmma::accumulator,16,16,16,float> value[2]; for(int row=0;row<2;row++)wmma::fill_fragment(value[row],0.f);
 for(int begin=part*512;begin<min(p0+(rows+6)/7,(part+1)*512);begin+=64){
  for(int i=tid;i<64*128;i+=256){int p=begin+i/128,j=i%128;size_t off=((size_t)h*ctx+p)*50,ring=((size_t)h*2592+p%2592)*128+j;float kval=0;if(p<p0+(rows+6)/7){bool recent=p>=cutoff;kval=recent?__bfloat162float(kr[ring]):nb_load_q3(kc+off,j);}K[(i/128)*144+i%128]=__float2bfloat16(kval);}__syncthreads();
  wmma::fragment<wmma::accumulator,16,16,16,float> scores;wmma::fill_fragment(scores,0.f);int qr=warp/4,kcol=warp%4;
  #pragma unroll
  for(int d=0;d<128;d+=16){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> qa;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> kb;wmma::load_matrix_sync(qa,Q+qr*16*144+d,144);wmma::load_matrix_sync(kb,K+kcol*16*144+d,144);wmma::mma_sync(scores,qa,kb,scores);}wmma::store_matrix_sync(S+qr*16*80+kcol*16,scores,80,wmma::mem_row_major);__syncthreads();
  for(int r=warp;r<32;r+=8){float a=-INFINITY;for(int j=lane;j<64;j+=32){float v=(r<rows&&begin+j<=p0+r/7)?S[r*80+j]*.08838834765f:-INFINITY;S[r*80+j]=v;a=fmaxf(a,v);}a=warp_max(a);float nm=fmaxf(mx[r],a);float prior=isfinite(nm)?expf(mx[r]-nm):0;float sum=0;for(int j=lane;j<64;j+=32){float v=isfinite(nm)?expf(S[r*80+j]-nm):0;__nv_bfloat16 hi=__float2bfloat16(v);PH[r*80+j]=hi;PL[r*80+j]=__float2bfloat16(v-__bfloat162float(hi));sum+=v;}sum=warp_sum(sum);if(lane==0){old[r]=prior;den[r]=den[r]*prior+sum;mx[r]=nm;}}
  __syncthreads();
  // K is dead after QK. Reuse its storage for V to allow two CTAs per SM.
  for(int i=tid;i<64*128;i+=256){int p=begin+i/128,j=i%128;float val=0;if(p<p0+(rows+6)/7){size_t off=((size_t)h*ctx+p)*50,ring=((size_t)h*2592+p%2592)*128+j;val=p>=cutoff?__bfloat162float(vr[ring]):nb_load_q3(vc+off,j);}V[(i/128)*144+i%128]=__float2bfloat16(val);}__syncthreads();
  #pragma unroll
  for(int row=0;row<2;row++){
   // SM120 WMMA accumulator mapping, checked against load_matrix_sync on this GPU.
   for(int e=0;e<value[row].num_elements;e++)value[row].x[e]*=old[row*16+lane/4+((e/2)%2)*8];
   #pragma unroll
   for(int d=0;d<64;d+=16){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> pa;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> vb;wmma::load_matrix_sync(vb,V+d*144+warp*16,144);wmma::load_matrix_sync(pa,PH+row*16*80+d,80);wmma::mma_sync(value[row],pa,vb,value[row]);wmma::load_matrix_sync(pa,PL+row*16*80+d,80);wmma::mma_sync(value[row],pa,vb,value[row]);}}
  __syncthreads();
 }
 for(int row=0;row<2;row++)wmma::store_matrix_sync(O+row*16*136+warp*16,value[row],136,wmma::mem_row_major); __syncthreads();
 for(int i=tid;i<rows*128;i+=256){int r=i/128,j=i%128;size_t off=(((size_t)(query_offset+r/7)*56+h*7+r%7)*splits+part)*130;tmp[off+2+j]=O[r*136+j];if(j==0){tmp[off]=mx[r];tmp[off+1]=den[r];}}
}
// Experimental INT8 QK / BF16 PV prefill. Its error is measured against the
// independent attention reference; the accurate BF16 path remains available.
extern "C" __global__ void nb_attn_q3_i8(const float* q,const uint8_t* kc,const uint8_t* vc,const __nv_bfloat16* kr,const __nv_bfloat16* vr,float* tmp,const int* pos,int ctx,int rows,int splits,int query_offset,int part_size,int full_rows){
 using namespace nvcuda;extern __shared__ __align__(32) unsigned char memory[];
 auto Q=(signed char*)memory;auto KV=(__nv_bfloat16*)(Q+32*144);auto K=(signed char*)KV;auto V=KV;auto S=(int*)(KV+64*144);auto P=(__nv_bfloat16*)S;auto mx=(float*)(S+32*64);auto den=mx+32;auto old=den+32;auto qs=old+32;auto ks=qs+32;auto O=(float*)KV;
 int tile=blockIdx.z; q+=(size_t)tile*32*7168;rows=min(32,rows-tile*32);query_offset+=tile*32;
 int h=blockIdx.x,part=blockIdx.y,tid=threadIdx.x,warp=tid/32,lane=tid%32,p0=*pos+query_offset,cutoff=max(max(0,(p0/32)*32-2048),*pos+full_rows-2592);
 for(int r=warp;r<32;r+=8){float v[4],a=0;for(int j=0;j<4;j++){v[j]=r<rows?q[(size_t)r*7168+h*128+lane+j*32]:0;a=fmaxf(a,fabsf(v[j]));}a=warp_max(a);float scale=a>0?a/127.f:1.f;if(lane==0){qs[r]=scale;mx[r]=-INFINITY;den[r]=0;}for(int j=0;j<4;j++)Q[r*144+lane+j*32]=max(-127,min(127,__float2int_rn(v[j]/scale)));}
 wmma::fragment<wmma::accumulator,16,16,16,float> value[2];for(int r=0;r<2;r++)wmma::fill_fragment(value[r],0.f);__syncthreads();
 for(int begin=part*part_size;begin<min(p0+rows,(part+1)*part_size);begin+=64){
  for(int r=warp;r<64;r+=8){int p=begin+r;float scale=1;int vals[4]={0,0,0,0};if(p<p0+rows){size_t off=((size_t)(h/7)*ctx+p)*50;if(p>=cutoff){float x[4],a=0;for(int j=0;j<4;j++){x[j]=__bfloat162float(kr[((size_t)(h/7)*2592+p%2592)*128+lane+j*32]);a=fmaxf(a,fabsf(x[j]));}a=warp_max(a);scale=a>0?a/127.f:1.f;for(int j=0;j<4;j++)vals[j]=max(-127,min(127,__float2int_rn(x[j]/scale)));}else{scale=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(kc+off))*(2.1519457f/127.f);for(int j=0;j<4;j++){int feature=lane+j*32,b=off+2+(feature/8)*3;unsigned bits=(unsigned)kc[b]|((unsigned)kc[b+1]<<8)|((unsigned)kc[b+2]<<16);int code=(bits>>(feature%8*3))&7,i=code<4?code:7-code;int a=(0x0e2d4f7f>>(i*8))&255;vals[j]=code<4?-a:a;}}}if(lane==0)ks[r]=scale;for(int j=0;j<4;j++)K[r*144+lane+j*32]=vals[j];}__syncthreads();
  wmma::fragment<wmma::accumulator,16,16,16,int> scores;wmma::fill_fragment(scores,0);int qr=warp/4,kcol=warp%4;
  #pragma unroll
  for(int d=0;d<128;d+=16){wmma::fragment<wmma::matrix_a,16,16,16,signed char,wmma::row_major> qa;wmma::fragment<wmma::matrix_b,16,16,16,signed char,wmma::col_major> kb;wmma::load_matrix_sync(qa,Q+qr*16*144+d,144);wmma::load_matrix_sync(kb,K+kcol*16*144+d,144);wmma::mma_sync(scores,qa,kb,scores);}wmma::store_matrix_sync(S+qr*16*64+kcol*16,scores,64,wmma::mem_row_major);__syncthreads();
  for(int r=warp;r<32;r+=8){float a=(r<rows&&begin+lane<=p0+r)?(float)S[r*64+lane]*qs[r]*ks[lane]*.08838834765f:-INFINITY,b=(r<rows&&begin+lane+32<=p0+r)?(float)S[r*64+lane+32]*qs[r]*ks[lane+32]*.08838834765f:-INFINITY;float nm=fmaxf(mx[r],warp_max(fmaxf(a,b))),prior=isfinite(nm)?expf(mx[r]-nm):0;float x=isfinite(nm)?expf(a-nm):0,y=isfinite(nm)?expf(b-nm):0;float sum=warp_sum(x+y);__syncwarp();P[r*128+lane]=__float2bfloat16(x);P[r*128+lane+32]=__float2bfloat16(y);if(lane==0){old[r]=prior;mx[r]=nm;den[r]=den[r]*prior+sum;}}__syncthreads();
  for(int i=tid;i<64*128;i+=256){int p=begin+i/128,j=i%128;float v=0;if(p<p0+rows){size_t off=((size_t)(h/7)*ctx+p)*50,ring=((size_t)(h/7)*2592+p%2592)*128+j;v=p>=cutoff?__bfloat162float(vr[ring]):nb_load_q3(vc+off,j);}V[(i/128)*144+i%128]=__float2bfloat16(v);}__syncthreads();
  #pragma unroll
  for(int r=0;r<2;r++){for(int e=0;e<value[r].num_elements;e++)value[r].x[e]*=old[r*16+lane/4+((e/2)%2)*8];
   #pragma unroll
   for(int d=0;d<64;d+=16){wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> pa;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> vb;wmma::load_matrix_sync(pa,P+r*16*128+d,128);wmma::load_matrix_sync(vb,V+d*144+warp*16,144);wmma::mma_sync(value[r],pa,vb,value[r]);}}
  __syncthreads();
 }
 for(int r=0;r<2;r++)wmma::store_matrix_sync(O+r*16*136+warp*16,value[r],136,wmma::mem_row_major);__syncthreads();
 for(int i=tid;i<rows*128;i+=256){int r=i/128,j=i%128;size_t off=(((size_t)(tile*32+r)*56+h)*splits+part)*130;tmp[off+2+j]=O[r*136+j];if(j==0){tmp[off]=mx[r];tmp[off+1]=den[r];}}
}

// Experimental fully INT8 matrix attention, with per-row probability scales.
extern "C" __global__ void nb_attn_q3_i8pv(const float* q,const uint8_t* kc,const uint8_t* vc,const __nv_bfloat16* kr,const __nv_bfloat16* vr,float* tmp,const int* pos,int ctx,int rows,int splits,int query_offset,int part_size,int full_rows){
 using namespace nvcuda;extern __shared__ __align__(32) unsigned char memory[];
 auto Q=(signed char*)memory;auto KV=(signed char*)(Q+32*144);auto K=KV;auto V=KV;auto S=(int*)(KV+64*144);auto P=(signed char*)S;auto mx=(float*)(S+32*64);auto den=mx+32;auto old=den+32;auto qs=old+32;auto ks=qs+32;auto vs=ks+64;auto ps=vs+64;auto O=(float*)KV;
 int tile=blockIdx.z; q+=(size_t)tile*32*7168;rows=min(32,rows-tile*32);query_offset+=tile*32;
 int h=blockIdx.x,part=blockIdx.y,tid=threadIdx.x,warp=tid/32,lane=tid%32,p0=*pos+query_offset,cutoff=max(max(0,(p0/32)*32-2048),*pos+full_rows-2592);
 for(int r=warp;r<32;r+=8){float v[4],a=0;for(int j=0;j<4;j++){v[j]=r<rows?q[(size_t)r*7168+h*128+lane+j*32]:0;a=fmaxf(a,fabsf(v[j]));}a=warp_max(a);float scale=a>0?a/127.f:1.f;if(lane==0){qs[r]=scale;mx[r]=-INFINITY;den[r]=0;}for(int j=0;j<4;j++)Q[r*144+lane+j*32]=max(-127,min(127,__float2int_rn(v[j]/scale)));}
 wmma::fragment<wmma::accumulator,16,16,16,float> value[2];for(int r=0;r<2;r++)wmma::fill_fragment(value[r],0.f);__syncthreads();
 for(int begin=part*part_size;begin<min(p0+rows,(part+1)*part_size);begin+=64){
  for(int r=warp;r<64;r+=8){int p=begin+r;float scale=1;int vals[4]={0,0,0,0};if(p<p0+rows){size_t off=((size_t)(h/7)*ctx+p)*50;if(p>=cutoff){float x[4],a=0;for(int j=0;j<4;j++){x[j]=__bfloat162float(kr[((size_t)(h/7)*2592+p%2592)*128+lane+j*32]);a=fmaxf(a,fabsf(x[j]));}a=warp_max(a);scale=a>0?a/127.f:1.f;for(int j=0;j<4;j++)vals[j]=max(-127,min(127,__float2int_rn(x[j]/scale)));}else{scale=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(kc+off))*(2.1519457f/127.f);for(int j=0;j<4;j++){int feature=lane+j*32,b=off+2+(feature/8)*3;unsigned bits=(unsigned)kc[b]|((unsigned)kc[b+1]<<8)|((unsigned)kc[b+2]<<16);int code=(bits>>(feature%8*3))&7,i=code<4?code:7-code;int a=(0x0e2d4f7f>>(i*8))&255;vals[j]=code<4?-a:a;}}}float vscale=0;if(p<p0+rows){if(p>=cutoff){float a=0;for(int j=0;j<4;j++)a=fmaxf(a,fabsf(__bfloat162float(vr[((size_t)(h/7)*2592+p%2592)*128+lane+j*32])));vscale=warp_max(a)/127.f;}else{size_t off=((size_t)(h/7)*ctx+p)*50;vscale=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(vc+off))*(2.1519457f/127.f);}}if(lane==0){ks[r]=scale;vs[r]=vscale;}for(int j=0;j<4;j++)K[r*144+lane+j*32]=vals[j];}__syncthreads();
  wmma::fragment<wmma::accumulator,16,16,16,int> scores;wmma::fill_fragment(scores,0);int qr=warp/4,kcol=warp%4;
  #pragma unroll
  for(int d=0;d<128;d+=16){wmma::fragment<wmma::matrix_a,16,16,16,signed char,wmma::row_major> qa;wmma::fragment<wmma::matrix_b,16,16,16,signed char,wmma::col_major> kb;wmma::load_matrix_sync(qa,Q+qr*16*144+d,144);wmma::load_matrix_sync(kb,K+kcol*16*144+d,144);wmma::mma_sync(scores,qa,kb,scores);}wmma::store_matrix_sync(S+qr*16*64+kcol*16,scores,64,wmma::mem_row_major);__syncthreads();
  for(int r=warp;r<32;r+=8){float a=(r<rows&&begin+lane<=p0+r)?(float)S[r*64+lane]*qs[r]*ks[lane]*.08838834765f:-INFINITY,b=(r<rows&&begin+lane+32<=p0+r)?(float)S[r*64+lane+32]*qs[r]*ks[lane+32]*.08838834765f:-INFINITY;float nm=fmaxf(mx[r],warp_max(fmaxf(a,b))),prior=isfinite(nm)?expf(mx[r]-nm):0;float x=isfinite(nm)?expf(a-nm):0,y=isfinite(nm)?expf(b-nm):0;float sum=warp_sum(x+y);x*=vs[lane];y*=vs[lane+32];float scale=warp_max(fmaxf(x,y))/127.f;float inv=scale>0?1.f/scale:0;__syncwarp();P[r*256+lane]=min(127,__float2int_rn(x*inv));P[r*256+lane+32]=min(127,__float2int_rn(y*inv));if(lane==0){old[r]=prior;mx[r]=nm;den[r]=den[r]*prior+sum;ps[r]=scale;}}__syncthreads();
  for(int i=tid;i<64*128;i+=256){int p=begin+i/128,j=i%128,v=0;if(p<p0+rows){if(p>=cutoff){float x=__bfloat162float(vr[((size_t)(h/7)*2592+p%2592)*128+j]);float inv=vs[i/128]>0?1.f/vs[i/128]:0;v=max(-127,min(127,__float2int_rn(x*inv)));}else{size_t off=((size_t)(h/7)*ctx+p)*50;int b=off+2+(j/8)*3;unsigned bits=(unsigned)vc[b]|((unsigned)vc[b+1]<<8)|((unsigned)vc[b+2]<<16);int code=(bits>>(j%8*3))&7,c=code<4?code:7-code;int val=(0x0e2d4f7f>>(c*8))&255;v=code<4?-val:val;}}V[(i/128)*144+i%128]=v;}__syncthreads();
  #pragma unroll
  for(int r=0;r<2;r++){wmma::fragment<wmma::accumulator,16,16,16,int> partial;wmma::fill_fragment(partial,0);
   #pragma unroll
   for(int d=0;d<64;d+=16){wmma::fragment<wmma::matrix_a,16,16,16,signed char,wmma::row_major> pa;wmma::fragment<wmma::matrix_b,16,16,16,signed char,wmma::row_major> vb;wmma::load_matrix_sync(pa,P+r*16*256+d,256);wmma::load_matrix_sync(vb,V+d*144+warp*16,144);wmma::mma_sync(partial,pa,vb,partial);}
   for(int e=0;e<value[r].num_elements;e++){int row=r*16+lane/4+((e/2)%2)*8;value[r].x[e]=value[r].x[e]*old[row]+(float)partial.x[e]*ps[row];}}
  __syncthreads();
 }
 for(int r=0;r<2;r++)wmma::store_matrix_sync(O+r*16*136+warp*16,value[r],136,wmma::mem_row_major);__syncthreads();
 for(int i=tid;i<rows*128;i+=256){int r=i/128,j=i%128;size_t off=(((size_t)(tile*32+r)*56+h)*splits+part)*130;tmp[off+2+j]=O[r*136+j];if(j==0){tmp[off]=mx[r];tmp[off+1]=den[r];}}
}
// E2M1 values multiplied by two are exact signed int8 values. Preserve the two
// E4M3 scales per 16-value dot product, using DP4A for the inner accumulation.
__device__ __forceinline__ unsigned nb_fp4_i8x4(unsigned bits){
 unsigned mag=__byte_perm(0x03020100u,0x0c080604u,bits&0x7777u);
 unsigned sign=((bits&0x8u)<<4)|((bits&0x80u)<<8)|((bits&0x800u)<<12)|((bits&0x8000u)<<16);
 unsigned mask=__vcmplts4(sign,0u);
 return __vsub4(mag^mask,mask);
}
extern "C" __global__ void nb_gemv_dp4a(const uint8_t* w,const uint8_t* sf,const uint8_t* x,const uint8_t* xs,float* y,int m,int k,float alpha){
 int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(row>=m)return;float sum=0;int kc=k/16;
 for(int c=lane;c<kc;c+=32){uint64_t wb=*(const uint64_t*)(w+(size_t)row*k/2+c*8),xb=*(const uint64_t*)(x+c*8);int dot=0;
  #pragma unroll
  for(int j=0;j<4;j++)dot=__dp4a((int)nb_fp4_i8x4((unsigned)(wb>>(j*16))),(int)nb_fp4_i8x4((unsigned)(xb>>(j*16))),dot);
  float sw=(float)(*reinterpret_cast<const __nv_fp8_e4m3*>(sf+nb_sf(row,c,kc))),sx=(float)(*reinterpret_cast<const __nv_fp8_e4m3*>(xs+nb_sf(0,c,kc)));
  sum+=dot*(sw*sx);
 }sum=warp_sum(sum);if(lane==0)y[row]=bf16r(sum*(alpha*.25f));
}

extern "C" __global__ void nb_h2bf(const half* input,__nv_bfloat16* output,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=__float2bfloat16(__half2float(input[i]));}
