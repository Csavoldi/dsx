import pathlib,json,hashlib,struct,math,time
r=pathlib.Path(__file__).parent;meta=json.loads((r/'receipts/quality-candidate-download.json').read_text());p=pathlib.Path(meta['local_directory']);q=json.loads((p/'quantization_config.json').read_text());tensors={};start=time.time()
for name,entry in meta['files'].items():
 f=p/name;h=hashlib.sha256()
 with f.open('rb') as s:
  while b:=s.read(16*1024*1024):h.update(b)
 assert h.hexdigest()==entry['sha256'],name
 with f.open('rb') as s:
  n=struct.unpack('<Q',s.read(8))[0];hdr=json.loads(s.read(n))
 for key,t in hdr.items():
  if key=='__metadata__':continue
  assert key not in tensors;tensors[key]=t
for i in range(52):
 for proj,(m,k) in {'self_attn.q_proj':(7168,7168),'self_attn.k_proj':(1024,7168),'self_attn.v_proj':(1024,7168),'self_attn.o_proj':(7168,7168),'mlp.gate_proj':(22016,7168),'mlp.up_proj':(22016,7168),'mlp.down_proj':(7168,22016)}.items():
  pref=f'model.layers.{i}.{proj}';assert tensors[pref+'.trellis']['shape']==[k//16,m//16,64];assert tensors[pref+'.suh']['shape']==[k];assert tensors[pref+'.svh']['shape']==[m];assert q['tensor_storage'][pref]['mul1_multiplier']==0x83DCD12D
out={**meta,'validation_seconds':time.time()-start,'verified_sha256':True,'validated_decoder_matrices':364,'tensor_count':len(tensors),'mode':'decoder-only override; original embedding and head stay unchanged'};(r/'receipts/quality-candidate-validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
