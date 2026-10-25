import pathlib,json,os,subprocess
r=pathlib.Path('/home/lxe/nanbeige-runner');b='/home/lxe/dsx-nanbeige/target/release/dsx-nanbeige';e={**os.environ,'DSX_NB_BATCH':'4096','DSX_NB_ATTN8':'2','DSX_NB_PART_SIZE':'16384','DSX_NB_EXL3_NARROW':'1'}
base=[b,'--model',str(r/'quant-exl3/result'),'--ctx','262144','--kv-bits','3','--host-embed','--head-bits','4']
code='<|im_start|>user\nWrite a Python function that tests whether a number is prime, and explain it.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
try:
 subprocess.run(['systemctl','--user','stop','dsx-nanbeige.service'],check=True)
 for variant in ['baseline','pf4']:
  env={**e,'DSX_NB_EXL3_EXPERIMENT':variant};env.pop('DSX_NB_EXL3_CARVEOUT',None)
  name='continue-full-'+variant+'-quality'
  with (r/'receipts'/(name+'.log')).open('w') as f:subprocess.run(base+['--tokens',str(r/'research-256k/wiki-4096.tokens.json'),'--ppl-tail','128','--ppl-logits',str(r/'receipts'/(name+'.f32')),'--receipt',str(r/'receipts'/(name+'.json'))],env=env,stdout=f,stderr=f,check=True)
  d=json.loads((r/'receipts'/(name+'.json')).read_text());print(name,{k:d[k] for k in ['ppl','prefill_seconds','decode_tps'] if k in d},flush=True)
finally:subprocess.run(['systemctl','--user','start','dsx-nanbeige.service'],check=True)
