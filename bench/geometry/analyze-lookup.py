#!/usr/bin/env python3
"""Replay causal copy proposals from raw sampled IDs; no inference or speedup claim."""
import argparse,json,pathlib,hashlib

def audit(prompt,output,suffix=8,window=8192,width=4):
    k=width-1;ctx=list(prompt);latest={};next_start=0;proposal=[];matched=0;cooldown=False;attempts=0;hist=[0]*width
    def index():
        nonlocal next_start
        # A lookup must leave its whole continuation in the already known context.
        while next_start+suffix+k<=len(ctx):
            latest[tuple(ctx[next_start:next_start+suffix])]=next_start;next_start+=1
    index()
    for token in output:
        ctx.append(token);index()
        if proposal:
            if token==proposal[matched]:
                matched+=1
                if matched==k:hist[matched]+=1;proposal=[];cooldown=True
            else:hist[matched]+=1;proposal=[]
        elif cooldown:cooldown=False
        elif len(ctx)>suffix:
            start=latest.get(tuple(ctx[-suffix:]))
            if start is not None and start>=max(0,len(ctx)-suffix-window):
                proposal=ctx[start+suffix:start+suffix+k];attempts+=1;matched=0
    completed=sum(hist);future=sum(i*n for i,n in enumerate(hist))
    return {'tokens':len(output),'attempts':attempts,'completed_blocks':completed,'incomplete_blocks':attempts-completed,'accepted_prefix_histogram':hist,'matched_future_tokens':future,'mean_matched_future_tokens':future/completed if completed else 0}

def main():
    p=argparse.ArgumentParser();p.add_argument('raw',type=pathlib.Path);p.add_argument('--run',type=pathlib.Path);p.add_argument('--oracle-costs',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    records=[json.loads(s) for s in a.raw.read_text().splitlines() if s.strip()]
    live=None
    if a.run:
        seeds={json.loads(p.read_text())['seed'] for p in (a.run/'requests').glob('*.request.json')};records=[r for r in records if r['seed'] in seeds]
        live=json.loads((a.run/'summary.json').read_text()).get('lookup_totals')
    costs=json.loads(a.oracle_costs.read_text())['warm_oracle'] if a.oracle_costs else None
    result=[]
    for width in [4,8]:
        for suffix in [8,12,16,24,32]:
            rows=[audit(r['prompt_token_ids'],r['output_token_ids'][:r['processed_output_tokens']],suffix=suffix,width=width) for r in records]
            totals={key:sum(r[key] for r in rows) for key in ['tokens','attempts','completed_blocks','incomplete_blocks','matched_future_tokens']}
            totals['accepted_prefix_histogram']=[sum(r['accepted_prefix_histogram'][i] for r in rows) for i in range(width)]
            blocks=totals['completed_blocks'];totals['mean_matched_future_tokens']=totals['matched_future_tokens']/blocks if blocks else 0
            totals.update({'width':width,'suffix':suffix,'window':8192})
            if width==4 and suffix==8 and live:
                totals['matches_live_audit']=all(totals[k]==live[k] for k in ['tokens','attempts','completed_blocks','incomplete_blocks','matched_future_tokens','accepted_prefix_histogram']);assert totals['matches_live_audit'],(totals,live)
            if costs and totals['tokens']:
                cs=1/costs['0']['tps'];cv=width/costs[str(width)]['tps'];n=totals['tokens'];replaced=blocks+totals['matched_future_tokens'];assert replaced<=n
                serial=n*cs;modeled=serial-replaced*cs+blocks*cv
                totals['conditional_cost_model']={'serial_step_s':cs,'verify_cycle_s':cv,'serial_steps_replaced':replaced,'serial_decode_s':serial,'modeled_decode_s':modeled,'modeled_decode_gain_percent':100*(serial/modeled-1)}
            result.append(totals)
    out={'records':len(records),'raw_sha256':hashlib.sha256(a.raw.read_bytes()).hexdigest(),'cost_receipt_sha256':hashlib.sha256(a.oracle_costs.read_bytes()).hexdigest() if a.oracle_costs else None,'rows':result,'limit':'Causal copy opportunity on an unchanged sampled trajectory. No verifier or speculative sampler ran; sampled-path matching is not measured speculative acceptance or delivered throughput. Optional cost model borrows historical future-informed warm placement, assumes zero drafting, communication, rollback and residency cost, and ignores unfinished proposals. It applies only to decode, excludes prefill/tools, and is not a bound for other drafters, verifier implementations or trajectories.'}
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
