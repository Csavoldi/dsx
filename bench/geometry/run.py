#!/usr/bin/env python3
"""Run one isolated Pi geometry task against the existing DSX server."""
import argparse, hashlib, json, os, pathlib, re, secrets, shlex, signal, statistics, string, subprocess, sys, time, urllib.request
from judge import judge, EXPECTED
from status import snapshot, format_status
HERE=pathlib.Path(__file__).resolve().parent

def command(args,**kw): return subprocess.run(args,check=True,capture_output=True,text=True,**kw).stdout

def summarize(root):
    metrics=[json.loads(l) for l in (root/'request-metrics.jsonl').read_text().splitlines() if l.strip()]
    ends=[x for x in metrics if x['event']=='end']; tools=[]; errors=[]
    for path in (root/'sessions').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            entry=json.loads(line); msg=entry.get('message',{})
            if msg.get('role')=='assistant': tools.extend(x for x in msg.get('content',[]) if x.get('type')=='toolCall')
            if msg.get('role')=='toolResult' and msg.get('isError'): errors.append(msg)
    started=json.loads((root/'started.json').read_text())
    final=json.loads((root/'final-judge.json').read_text())
    totals={k:sum((x.get('usage') or {}).get(k,0) for x in ends) for k in ['prompt_tokens','completion_tokens','total_tokens']}
    first=[x['first_delta_s'] for x in ends if 'first_delta_s' in x]
    result={'shape':started['shape'],'arm':started['arm'],'contract_version':started.get('contract_version',1),'reasoning':started.get('reasoning','off'),'seed_base':started.get('seed_base'),'wall_s':started.get('agent_wall_s',time.time()-started['wall']),'requests':len(ends),'tool_calls':len(tools),'tool_errors':len(errors),'usage':totals,'ttft_median_s':statistics.median(first) if first else None,'automated_pass':final['pass'],'functional_pass':None,'final_judge':final}
    probes=sorted(root.glob('judge-*.json')); good=[json.loads(p.read_text()) for p in probes if json.loads(p.read_text()).get('pass') and json.loads(p.read_text()).get('judge_version',1)==final.get('judge_version',1)]
    result['first_observed_pass_s']=min((x['timestamp']-started['wall'] for x in good),default=None)
    review_path=root/'review.json'
    if review_path.exists():
        review=json.loads(review_path.read_text());result['review']=review;result['functional_pass']=final['pass'] and review['pass']
    result['source_revisions']=len(list((root/'revisions').glob('*.py')))
    log=(root/'server.log').read_text() if (root/'server.log').exists() else ''
    ids={x.get('response_id') for x in ends}; server=[]; pending={}; diagnostics={}
    for line in re.sub(r'\x1b\[[0-9;]*m','',log).splitlines():
        for label,key in [('runtime stats','runtime'),('lookup audit','lookup')]:
            if label in line and 'stats=' in line:
                try: pending[key]=json.loads(line.split('stats=',1)[1])
                except ValueError: pass
        if 'engine: done ' not in line: continue
        kv={k:v.strip('"') for k,v in re.findall(r'(\w+)=("[^"]*"|\S+)',line)}
        if kv.get('id') in ids and kv.get('id') not in {x['id'] for x in server}:
            server.append(kv)
            if pending: diagnostics[kv['id']]=pending
        pending={}
    result['server_request_count']=len(server)
    result['started_requests']=sum(x.get('event')=='start' for x in metrics)
    result['unfinished_requests']=result['started_requests']-len(ends)
    result['complete_server_counter_coverage']=len(server)==len(ends)
    for key in ['generated','prefill_ms','decode_ms','queue_ms','prompt_tokens','cached_prompt_tokens']:
        result[key]=sum(int(x.get(key,0)) for x in server)
    result['decode_tok_s']=result['generated']*1000/result['decode_ms'] if result['decode_ms'] else None
    fresh=result['prompt_tokens']-result['cached_prompt_tokens']
    result['new_prompt_tokens']=fresh
    result['new_prefill_tok_s']=fresh*1000/result['prefill_ms'] if result['prefill_ms'] else None
    result['diagnostics']=diagnostics
    runtime=[v['runtime'] for v in diagnostics.values() if 'runtime' in v]
    if runtime:
        result['runtime_totals']={key:sum(x.get(key,0) for x in runtime) for key in ['steps','gpu_samples','gpu_fallbacks','sample_s','emit_s','step_s','adapt_host_s','model_prefill_s','snapshot_s','prefix_replayed_tokens','routed_picks','cpu_picks']}
        result['gpu_sampling_requests']=sum(bool(x.get('gpu_sampling')) for x in runtime)
    lookup=[v['lookup'] for v in diagnostics.values() if 'lookup' in v]
    if lookup:
        totals={key:sum(x.get(key,0) for x in lookup) for key in ['tokens','attempts','completed_blocks','incomplete_blocks','matched_future_tokens','audit_seconds']}
        totals['accepted_prefix_histogram']=[sum(x['accepted_prefix_histogram'][i] for x in lookup) for i in range(4)]
        totals['mean_matched_future_tokens']=totals['matched_future_tokens']/totals['completed_blocks'] if totals['completed_blocks'] else 0
        totals['limit']='Serial copy audit only; no speculative verifier or drafter speedup measured.'
        result['lookup_totals']=totals
    result['acceptance_stop']=started.get('acceptance_stop',False)
    if (root/'accepted-stop.json').exists():result['accepted_stop']=json.loads((root/'accepted-stop.json').read_text())
    result['agent_exit']=started.get('agent_exit')
    result['timed_out']=started.get('timed_out',False)
    result['successful_agent_exit']=started.get('agent_exit')==0 and not result['timed_out']
    if result['functional_pass'] is True: result['functional_pass']=result['successful_agent_exit']
    result['server_request_ids']=[x['id'] for x in server]
    result['comparison_eligible']=not (root/'excluded.json').exists() and not (root/'interrupted.json').exists()
    if (root/'excluded.json').exists():result['excluded']=json.loads((root/'excluded.json').read_text())
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('shape',choices=EXPECTED);p.add_argument('root',type=pathlib.Path);p.add_argument('--arm',default='stock');p.add_argument('--timeout',type=int,default=360);p.add_argument('--reasoning',choices=['off','low'],default='off');p.add_argument('--port',type=int,default=8987);p.add_argument('--guidance',type=pathlib.Path);p.add_argument('--server-log',default='/home/lxe/ai/ds41/serve.out');p.add_argument('--acceptance-stop',action='store_true');p.add_argument('--judge-version',type=int,choices=[5,7,8],default=5);p.add_argument('--normalize-capture-blanks',action='store_true');a=p.parse_args()
    if a.normalize_capture_blanks and not a.acceptance_stop:p.error('--normalize-capture-blanks requires --acceptance-stop')
    if a.acceptance_stop and a.judge_version==8:
        from judge_v8 import judge as selected_judge
    elif a.acceptance_stop and a.judge_version==7:
        from judge_v7 import judge as selected_judge
    elif a.acceptance_stop:
        from judge_v5 import judge as selected_judge
    else:
        selected_judge=judge
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=False)
    for name in ['workspace','requests','sessions','config','revisions']: (root/name).mkdir()
    config=json.loads((HERE/'models.json').read_text());config['providers']['dsx-cube']['baseUrl']=f'http://127.0.0.1:{a.port}/v1'
    (root/'config/models.json').write_text(json.dumps(config,indent=2)+'\n')
    session='dsx-geometry-'+root.name
    pane=command(['tmux','new-session','-d','-s',session,'-x','110','-y','32','-P','-F','#{pane_id}']).strip()
    task=string.Template((HERE/'PROMPT.md').read_text()).substitute(shape=a.shape,pane=pane,workspace=str(root/'workspace'))
    if a.guidance: task+='\n'+a.guidance.read_text()
    (root/'TASK.md').write_text(task)
    health=json.load(urllib.request.urlopen('http://192.168.0.102:8090/health',timeout=10));assert health['status']=='ok' and health['pending']==0,health
    (root/'host-state.json').write_text(command(['ssh','-o','BatchMode=yes','192.168.0.102','python3 -'],input=(HERE/'host-state.py').read_text()))
    started={'wall':time.time(),'shape':a.shape,'arm':a.arm,'pane':pane,'contract_version':4,'reasoning':a.reasoning,'seed_base':secrets.randbits(53),'temperature':1.0,'top_p':1.0,'task_sha256':hashlib.sha256(task.encode()).hexdigest(),'health':health,'timeout_s':a.timeout,'server_log':a.server_log,'acceptance_stop':a.acceptance_stop,'judge_version':a.judge_version if a.acceptance_stop else 4,'normalize_capture_blanks':a.normalize_capture_blanks,'guidance_sha256':hashlib.sha256(a.guidance.read_bytes()).hexdigest() if a.guidance else None}
    (root/'started.json').write_text(json.dumps(started,indent=2)+'\n')
    env=dict(os.environ,GAUNTLET_RUN=str(root),GAUNTLET_PORT=str(a.port),PI_CODING_AGENT_DIR=str(root/'config'),PI_TELEMETRY='0',GAUNTLET_SEED_BASE=str(started['seed_base']),GAUNTLET_JUDGE_VERSION=str(a.judge_version),GAUNTLET_NORMALIZE_CAPTURE_BLANKS='1' if a.normalize_capture_blanks else '0');env.pop('NODE_OPTIONS',None)
    proxy_log=(root/'proxy.log').open('w');agent_log=(root/'agent.jsonl').open('w')
    proxy=subprocess.Popen([sys.executable,str(HERE/'proxy.py')],env=env,stdout=proxy_log,stderr=subprocess.STDOUT,start_new_session=True)
    agent=None
    try:
        for _ in range(50):
            try: urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health',timeout=1);break
            except OSError: time.sleep(.1)
        else: raise RuntimeError('proxy did not start')
        args=['pi','--offline','--provider','dsx-cube','--model','deepseek-v4.1-flash:'+a.reasoning,'--tools','read,bash,edit,write','--no-extensions','--no-mcp','--no-skills','--no-prompt-templates','--no-context-files','--session-dir',str(root/'sessions'),'--name',f'DSX {a.shape} {a.arm}','--mode','json','--print',task]
        if a.acceptance_stop: args[1:1]=['--extension',str(HERE/'acceptance-stop.ts')]
        agent=subprocess.Popen(args,cwd=root/'workspace',env=env,stdout=agent_log,stderr=subprocess.STDOUT,start_new_session=True)
        (root/'runtime.json').write_text(json.dumps({'runner_pid':os.getpid(),'agent_pid':agent.pid,'proxy_pid':proxy.pid,'argv':args},indent=2)+'\n')
        last=None;attempt=0;first_pass=False;last_status=0
        while agent.poll() is None and time.time()-started['wall']<a.timeout:
            if time.monotonic()-last_status>=5:
                status=snapshot(root);(root/'progress.txt').write_text(format_status(status))
                print(json.dumps(dict(status,event='progress')),flush=True);last_status=time.monotonic()
            source=root/'workspace/render.py';mesh=root/'workspace/mesh.json'
            if source.exists():
                digest=hashlib.sha256(source.read_bytes()+(mesh.read_bytes() if mesh.exists() else b'')).hexdigest()
                if digest!=last:
                    last=digest;attempt+=1;(root/'revisions'/f'{attempt:03}.py').write_bytes(source.read_bytes())
                    if mesh.exists(): (root/'revisions'/f'{attempt:03}.mesh.json').write_bytes(mesh.read_bytes())
                    result=selected_judge(root/'workspace',a.shape,root/f'judge-{attempt:03}.json')
                    print(json.dumps({'event':'judge','attempt':attempt,'elapsed_s':time.time()-started['wall'],'pass':result['pass'],'error':result.get('error')}),flush=True)
                    first_pass|=result['pass']
                    if a.acceptance_stop: (root/'latest-check.json').write_text(json.dumps(result,indent=2)+'\n')
                    if a.acceptance_stop and result['pass'] and hashlib.sha256(source.read_bytes()).hexdigest()==result.get('source_sha256'):
                        (root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
            time.sleep(2)
        if agent.poll() is None:
            os.killpg(agent.pid,signal.SIGTERM)
            try: agent.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(agent.pid,signal.SIGKILL);agent.wait()
            started['timed_out']=True
        started['agent_exit']=agent.returncode;started['agent_wall_s']=time.time()-started['wall']
        (root/'started.json').write_text(json.dumps(started,indent=2)+'\n')
        selected_judge(root/'workspace',a.shape,root/'final-judge.json')
        (root/'display.txt').write_text(command(['tmux','capture-pane','-p','-t',pane]))
        (root/'server.log').write_text(command(['ssh','-o','BatchMode=yes','192.168.0.102','cat '+shlex.quote(a.server_log)]))
        print(json.dumps(summarize(root),indent=2),flush=True)
        (root/'progress.txt').write_text(format_status(snapshot(root)))
    finally:
        if agent is not None and agent.poll() is None: os.killpg(agent.pid,signal.SIGTERM);agent.wait(timeout=10)
        command(['tmux','send-keys','-t',pane,'C-c']);time.sleep(.2)
        command(['tmux','kill-session','-t',session])
        proxy.terminate();proxy.wait(timeout=10);proxy_log.close();agent_log.close()

if __name__=='__main__':main()
