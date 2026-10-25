import pathlib,json,os,subprocess
r=pathlib.Path('/home/lxe/nanbeige-runner');b='/home/lxe/dsx-nanbeige/target/release/dsx-nanbeige';e={**os.environ,'DSX_NB_BATCH':'4096','DSX_NB_ATTN8':'2','DSX_NB_PART_SIZE':'16384','DSX_NB_EXL3_NARROW':'1','DSX_NB_EXL3_EXPERIMENT':'pf4'}
base=[b,'--model',str(r/'quant-exl3/result'),'--ctx','262144','--kv-bits','3','--host-embed','--head-bits','4']
prompts={
 'rust':'Implement a Rust function that merges two sorted slices of i64 into a sorted Vec<i64>. Preserve duplicates, explain its complexity, and give three unit tests.',
 'sql':'Given orders(id, customer_id, created_at, amount), write a PostgreSQL query returning the top three customers by total spend per calendar month. Explain ties and include customers with refunds as negative amounts.',
 'reason':'A box contains 5 red, 4 blue, and 3 green balls. Draw 4 without replacement. Calculate the probability of drawing exactly two colors. Explain the counting, simplify the fraction, and verify the result by a second counting method.'}
def run(name,args):
 with (r/'receipts'/(name+'.log')).open('w') as f:subprocess.run(base+args+['--receipt',str(r/'receipts'/(name+'.json'))],env=e,stdout=f,stderr=f,check=True)
 d=json.loads((r/'receipts'/(name+'.json')).read_text());print(name,{k:d[k] for k in ['input_tokens','output_tokens','decode_tps','lookup','warm_matches_cold','retry_matches_cold'] if k in d},flush=True)
try:
 subprocess.run(['systemctl','--user','stop','dsx-nanbeige.service'],check=True)
 for name,content in prompts.items():
  prompt='<|im_start|>user\n'+content+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
  for mode in ['off','on']:
   e.pop('DSX_NB_SPEC_GRAPHS',None)
   if mode=='on':e['DSX_NB_SPEC_GRAPHS']='1'
   run('continue-graphs-'+name+'-'+mode,['--prompt',prompt,'--max-tokens','512','--lookup','3'])
 e['DSX_NB_SPEC_GRAPHS']='1'
 run('continue-graphs-prefix',['--tokens',str(r/'research-256k/wiki-4096.tokens.json'),'--cache-check'])
finally:subprocess.run(['systemctl','--user','start','dsx-nanbeige.service'],check=True)
