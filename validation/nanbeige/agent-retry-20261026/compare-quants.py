import json,os,pathlib,subprocess,time,hashlib
r=pathlib.Path(__file__).parent;out=r/'receipts/quant-comparison';out.mkdir(exist_ok=True);binary=r/'dsx-nanbeige-baseline';src=pathlib.Path('/home/lxe/dsx-nanbeige/target/release/dsx-nanbeige')
if not binary.exists():
 import shutil;shutil.copy2(src,binary)
env={**os.environ,'DSX_NB_BATCH':'4096','DSX_NB_ATTN8':'2','DSX_NB_PART_SIZE':'16384','DSX_NB_EXL3_NARROW':'1','DSX_NB_EXL3_EXPERIMENT':'pf4'};env.pop('DSX_NB_GQA_I8PV',None)
base=[str(binary),'--model','/home/lxe/nanbeige-runner/quant-exl3/result','--ctx','262144','--kv-bits','3','--host-embed','--head-bits','4']
code='<|im_start|>user\nWrite a Python function that tests whether a number is prime, and explain it.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
rows=[]
try:
 subprocess.run(['systemctl','--user','stop','dsx-nanbeige.service'],check=True)
 for name,overlay in [('local2.5',None),('luni4.0','/home/lxe/nanbeige-runner/models/Luni-Nanbeige-4.3-33B-EXL3-4.00bpw')]:
  e=env.copy();e.pop('DSX_NB_EXL3',None)
  if overlay:e['DSX_NB_EXL3']=overlay
  for mode,flags in [('quality',['--tokens','/home/lxe/nanbeige-runner/research-256k/wiki-4096.tokens.json','--ppl-tail','128','--ppl-logits',str(out/(name+'.f32'))]),('coding',['--prompt',code,'--max-tokens','512','--lookup','3'])]:
   stem=f'{name}-{mode}';receipt=out/(stem+'.json');t=time.time()
   with (out/(stem+'.log')).open('w') as f:p=subprocess.run(base+flags+['--receipt',str(receipt)],env=e,stdout=f,stderr=f)
   row={'quant':name,'mode':mode,'returncode':p.returncode,'wall_seconds':time.time()-t,'recipe':{'base':'local pack original embedding/head','decoder_override':overlay,'runtime_head_bits':4,'kv_bits':3,'capacity':262144,'batch':4096}}
   if p.returncode==0:row['result']=json.loads(receipt.read_text())
   else:row['error_tail']=(out/(stem+'.log')).read_text()[-1800:]
   rows.append(row);(out/'summary.json').write_text(json.dumps({'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'measurements':rows,'scope':'matched128-targetprecisiondiagnostic+one512-tokencodingprompt;notafulltaskcompletioneval'},indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='result'}),flush=True)
   if p.returncode:break
finally:subprocess.run(['systemctl','--user','start','dsx-nanbeige.service'],check=False)
