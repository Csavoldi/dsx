import os,sys,pathlib,json,time,hashlib,gc
os.environ['OMP_NUM_THREADS']='8'
sys.path.insert(0,'/root/nanbeige-quant/exl3-source')
import torch
from exllamav3 import Config,Model
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles,get_temp_buffers_frac
root=pathlib.Path('/root/nanbeige-quant')
torch.set_num_threads(8)
@torch.inference_mode()
def main():
 torch.manual_seed(20261025)
 x=torch.randn((17,256),device='cuda');x[0]=0;x[1]=2;x[2]=torch.linspace(-4,4,256,device='cuda')
 os.environ['EXL3_QTF_OPT']='0'; y0,i0=quantize_tiles(x,{'K':2.5,'mul1':True})
 os.environ['EXL3_QTF_OPT']='1'; y1,i1=quantize_tiles(x,{'K':2.5,'mul1':True})
 check={'tiles':17,'rate':2.5,'seed':20261025,'indices_bit_identical':torch.equal(i0,i1),'decoded_values_bit_identical':torch.equal(y0,y1),'mean_squared_error':torch.mean((y1-x)**2).item()}
 (root/'fractional-quantizer-check.json').write_text(json.dumps(check,indent=2));print(check,flush=True)
 assert check['indices_bit_identical'] and check['decoded_values_bit_identical']
 del x,y0,i0,y1,i1;get_temp_buffers_frac.cache_clear();gc.collect();torch.cuda.empty_cache()
 ids=json.loads((root/'wiki-4096.tokens.json').read_text());assert len(ids)==4096
 inp=torch.tensor([ids[:-1]],dtype=torch.long);targets=torch.tensor(ids[-128:],dtype=torch.long)
 results={};saved={}
 for name,path in [('source','model'),('exl3_2.5bpw','result')]:
  print('REFERENCE LOAD',name,flush=True);cfg=Config.from_directory(str(root/path));m=Model.from_config(cfg)
  m.load(device='cuda:0',max_chunk_size=4096,max_output_size=128,progressbar=False)
  torch.cuda.synchronize();t=time.time()
  logits=m.forward(inp,params={'attn_mode':'flash_attn_nc','last_tokens_only':128})[0].float().cpu()
  torch.cuda.synchronize();elapsed=time.time()-t
  assert logits.shape==(128,166912),logits.shape
  assert torch.isfinite(logits).all()
  logs=logits.log_softmax(-1);loss=-logs.gather(-1,targets[:,None]).squeeze(-1)
  results[name]={'ppl':loss.mean().exp().item(),'mean_nll':loss.mean().item(),'top1_hits':(logits.argmax(-1)==targets).sum().item(),'forward_seconds':elapsed,'losses':loss.tolist()}
  saved[name]=logs
  torch.save(logits,root/f'{name}-tail128-logits.pt')
  m.unload();del m,cfg,logits;gc.collect();torch.cuda.empty_cache()
 base=saved['source'];quant=saved['exl3_2.5bpw'];kl=(base.exp()*(base-quant)).sum(-1).mean().item()
 out={'protocol':'4096-token WikiText-2 raw validation prefix, score final 128 tokens; uncached 4095-token forward in ExLlamaV3 1.6.4 using PyTorch SDPA; source checkpoint loaded by same runtime','fixture_sha256':hashlib.sha256((root/'wiki-4096.tokens.json').read_bytes()).hexdigest(),'gpu':torch.cuda.get_device_name(0),'results':results,'kl_source_to_quant':kl,'top1_agreement':(base.argmax(-1)==quant.argmax(-1)).float().mean().item()}
 (root/'reference-quality.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
main()
