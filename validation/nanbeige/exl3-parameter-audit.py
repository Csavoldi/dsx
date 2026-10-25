import json,math,urllib.request,pathlib

def count(cfg):
 logical=stored=0;seen=set();rates={}
 for module,entry in cfg['tensor_storage'].items():
  tensors=entry['stored_tensors']
  for name,t in tensors.items():
   assert name not in seen;seen.add(name);stored+=math.prod(t['shape'])
  if entry.get('quant_format')=='exl3':
   tr=tensors[module+'.trellis'];assert len(tr['shape'])==3
   logical+=tr['shape'][0]*tr['shape'][1]*256
   bpw=tr['shape'][2]/16;assert bpw==entry['bits_per_weight'];rates[bpw]=rates.get(bpw,0)+1
   for name,t in tensors.items():
    if not name.endswith(('.trellis','.suh','.svh','.mul1','.mcg')):logical+=math.prod(t['shape'])
  else:logical+=sum(math.prod(t['shape']) for t in tensors.values())
 return {'logical_parameters':logical,'stored_elements':stored,'quantized_matrix_rates':rates,'stored_tensor_count':len(seen)}
root=pathlib.Path('/home/lxe/nanbeige-runner')
a=json.load(urllib.request.urlopen('https://huggingface.co/turboderp/Qwen3-32B-exl3/resolve/63c102648ff84be908577d25f25213c19ae13de9/quantization_config.json'));b=json.load(open(root/'quant-exl3/result/quantization_config.json'))
rows=[]
for name,cfg,source in [('turboderp/Qwen3-32B-exl3',a,'Qwen/Qwen3-32B'),('lxe/Nanbeige4.3-33B-EXL3-2.5bpw',b,'Nanbeige/Nanbeige4.3-33B')]:
 info=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+source))
 row={'pack':name,'source':source,**count(cfg),'source_api_parameters':info.get('safetensors',{}).get('total')};row['matches_source']=row['logical_parameters']==row['source_api_parameters'];assert row['matches_source'];rows.append(row)
print(json.dumps(rows,indent=2));(root/'receipts/exl3-logical-parameter-audit.json').write_text(json.dumps(rows,indent=2)+'\n')
