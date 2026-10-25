"""Validate EXL3 release structure and bytes without importing a model runtime."""
import pathlib,json,struct,hashlib,math,numpy as np
root=pathlib.Path(__file__).parent/'result'
files=sorted(root.glob('*.safetensors')); assert files,'No completed model shards'
cloud_hashes={line.split()[1]:line.split()[0] for line in (root/'SHA256SUMS').read_text().splitlines()}

shards={}; tensors={}; total_payload=0
size={'I16':2,'I32':4,'F16':2,'BF16':2,'F32':4,'I64':8,'U8':1}
for p in files:
 with p.open('rb') as f:
  hlen=struct.unpack('<Q',f.read(8))[0]; assert 0<hlen<100_000_000
  hdr=json.loads(f.read(hlen)); data_start=8+hlen
  intervals=[]
  for name,t in hdr.items():
   if name=='__metadata__':continue
   assert name not in tensors, f'duplicate {name}'
   begin,end=t['data_offsets']; assert end-begin==math.prod(t['shape'])*size[t['dtype']],name
   assert 0<=begin<=end<=p.stat().st_size-data_start,name
   intervals.append((begin,end));tensors[name]=(p,data_start,t)
  cursor=0
  for begin,end in sorted(intervals):assert begin==cursor,(p,begin,cursor);cursor=end
  assert cursor==p.stat().st_size-data_start
  total_payload+=cursor
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(16*1024*1024):h.update(b)
 shards[p.name]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def read(name):
 p,start,t=tensors[name];a,b=t['data_offsets']
 with p.open('rb') as f:f.seek(start+a);return f.read(b-a)
projs={'self_attn.q_proj':(7168,7168),'self_attn.k_proj':(1024,7168),'self_attn.v_proj':(1024,7168),'self_attn.o_proj':(7168,7168),'mlp.gate_proj':(22016,7168),'mlp.up_proj':(22016,7168),'mlp.down_proj':(7168,22016)}
expected=set(); weights=0; coded_bytes=0
for layer in range(52):
 for proj,(n,k) in projs.items():
  prefix=f'model.layers.{layer}.{proj}'
  for suffix,shape,dtype in [('trellis',[k//16,n//16,40],'I16'),('suh',[k],'F16'),('svh',[n],'F16'),('mul1',[],'I32')]:
   name=prefix+'.'+suffix; expected.add(name); t=tensors[name][2]
   assert t['shape']==shape and t['dtype']==dtype,(name,t)
   if suffix in ['suh','svh']:
    values=np.frombuffer(read(name),dtype='<f2');assert np.isfinite(values).all(),name
   if suffix=='mul1':assert struct.unpack('<I',read(name))[0]==0x83DCD12D,name
  weights+=n*k;coded_bytes+=n*k*5//16
 for norm in ['input_layernorm','post_attention_layernorm']:
  name=f'model.layers.{layer}.{norm}.weight';expected.add(name);t=tensors[name][2]
  assert t['shape']==[7168] and t['dtype']=='BF16',(name,t)
  bits=np.frombuffer(read(name),dtype='<u2');assert ((bits&0x7f80)!=0x7f80).all(),name
for name,shape in [('model.embed_tokens.weight',[166912,7168]),('lm_head.weight',[166912,7168]),('model.norm.weight',[7168])]:
 expected.add(name);t=tensors[name][2];assert t['shape']==shape and t['dtype'] in ['BF16','F16'],(name,t)
for name in ['model.embed_tokens.weight','lm_head.weight']:
 p,start,t=tensors[name];a,b=t['data_offsets'];mask=0x7f80 if t['dtype']=='BF16' else 0x7c00
 data=np.memmap(p,dtype='<u2',mode='r',offset=start+a,shape=((b-a)//2,))
 for off in range(0,len(data),8*1024*1024):assert ((data[off:off+8*1024*1024]&mask)!=mask).all(),name
 del data
assert {name:v['sha256'] for name,v in shards.items()}==cloud_hashes,'Cloud/local checksum mismatch'
assert expected==set(tensors), {'missing':sorted(expected-set(tensors)),'extra':sorted(set(tensors)-expected)}
idx=json.loads((root/'model.safetensors.index.json').read_text()) if len(files)>1 else None
if idx:
 assert idx['metadata']['total_size']==total_payload
 assert idx['metadata'].get('total_parameters', weights+2*166912*7168+105*7168)==weights+2*166912*7168+105*7168
 assert idx['weight_map']=={name:p.name for name,(p,_,_) in tensors.items()}
q=json.loads((root/'config.json').read_text())['quantization_config'];assert q['quant_method']=='exl3' and q['codebook']=='mul1'
assert q['bits']==2.5, 'Retain upstream fractional EXL3 bits metadata'
storage_q=json.loads((root/'quantization_config.json').read_text())
assert all(storage_q.get(k)==v for k,v in q.items()), 'Root and detailed EXL3 metadata disagree'
manifest={'schema_version':1,'tensor_dtypes':{name:t['dtype'] for name,(_,_,t) in tensors.items() if name in ['lm_head.weight','model.embed_tokens.weight','model.norm.weight']},'tensor_count':len(tensors),'decoder_matrices':364,'decoder_weights':weights,'underlying_model_parameters':weights+2*166912*7168+105*7168,'decoder_trellis_bytes':coded_bytes,'decoder_bits_per_weight':coded_bytes*8/weights,'tensor_payload_bytes':total_payload,'model_shard_bytes':sum(x['bytes'] for x in shards.values()),'whole_model_effective_bits_per_weight':total_payload*8/(weights+2*166912*7168+105*7168),'checks':['all expected tensors present once','exact shapes and dtypes','contiguous offsets and exact payload sizes','finite rotation scales, norms, embedding and head','stock mul1 codebook markers','complete shard index','per-shard SHA256'],'shards':shards}
(root/'validation.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'SHA256SUMS').write_text(''.join(f'{v["sha256"]}  {name}\n' for name,v in shards.items()))
print(json.dumps(manifest,indent=2))
