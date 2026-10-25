import ctypes as C,json,time,torch
from pathlib import Path
torch.cuda.init();torch.empty(1,device='cuda');d=C.CDLL('libcuda.so.1');mod=C.c_void_p()
def ck(code):
 if code:raise RuntimeError(f'CUDA driver error {code}')
ck(d.cuModuleLoad(C.byref(mod),b'/root/dsx.fatbin'))
def function(name):
 f=C.c_void_p();ck(d.cuModuleGetFunction(C.byref(f),mod,name.encode()));return f
fn=function('nb_attn_q3_gqa');merge=function('nb_attn_merge');ck(d.cuFuncSetAttribute(fn,8,49152))
def launch(f,grid,block,smem,values):
 args=(C.c_void_p*len(values))(*[C.cast(C.pointer(v),C.c_void_p) for v in values]);ck(d.cuLaunchKernel(f,*map(C.c_uint,grid),C.c_uint(block),C.c_uint(1),C.c_uint(1),C.c_uint(smem),C.c_void_p(0),args,C.c_void_p(0)))
ptr=lambda t:C.c_uint64(t.data_ptr());i=C.c_int
ctx=262144;torch.manual_seed(379)
q=torch.randn((8,7168),device='cuda',dtype=torch.float32)
k=torch.randint(0,256,(8,ctx,50),dtype=torch.uint8,device='cuda');v=torch.randint(0,256,k.shape,dtype=torch.uint8,device='cuda')
k[:,:,0]=0;k[:,:,1]=63;v[:,:,0]=0;v[:,:,1]=63
recent=torch.randn((8,2592,128),device='cuda',dtype=torch.bfloat16)
tmp=torch.empty((8*56*((ctx+127)//128)*130,),device='cuda',dtype=torch.float32);out=torch.empty((8,7168),device='cuda',dtype=torch.bfloat16)
results=[]
for pos in [4096,32768,65536,131072,262135]:
 pd=torch.tensor([pos],device='cuda',dtype=torch.int32)
 for rows in [1,4,8]:
  baseline=None
  for part in [512,128,256,1024,2048]:
   splits=(pos+rows+part-1)//part
   def run():
    launch(fn,(8,splits,(rows+3)//4),256,49152,[ptr(q),ptr(k),ptr(v),ptr(recent),ptr(recent),ptr(tmp),ptr(pd),i(ctx),i(rows),i(splits),i(0),i(part)])
    launch(merge,(56,rows,1),128,0,[ptr(tmp),ptr(out),i(rows),i(splits)])
   for _ in range(24):run()
   torch.cuda.synchronize();got=out[:rows].float().clone()
   assert torch.isfinite(got).all()
   if baseline is None:baseline=got
   diff=(got-baseline).abs().max().item();assert diff<.01,diff
   times=[]
   for _ in range(7):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
    for _ in range(4):run()
    b.record();b.synchronize();times.append(a.elapsed_time(b)/4)
   r=dict(position=pos,rows=rows,partition_tokens=part,splits=splits,median_ms=sorted(times)[3],max_abs_difference_vs_512=diff);results.append(r)
r=dict(scope='synthetic isolated GQA attention plus merge; not model throughput',hardware=torch.cuda.get_device_name(),context_capacity=ctx,results=results)
Path('/root/attention-partitions.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
