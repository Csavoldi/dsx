#!/usr/bin/env python3
"""Run one isolated Pi geometry task against the existing DSX server."""
import argparse, hashlib, json, os, pathlib, re, shlex, signal, statistics, subprocess, sys, time, urllib.request
from judge import judge, EXPECTED
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
    result={'shape':started['shape'],'arm':started['arm'],'wall_s':started.get('agent_wall_s',time.time()-started['wall']),'requests':len(ends),'tool_calls':len(tools),'tool_errors':len(errors),'usage':totals,'ttft_median_s':statistics.median(first) if first else None,'functional_pass':final['pass'],'final_judge':final}
    probes=sorted(root.glob('judge-*.json')); good=[json.loads(p.read_text()) for p in probes if json.loads(p.read_text()).get('pass')]
    result['first_observed_pass_s']=min((x['timestamp']-started['wall'] for x in good),default=None)
    result['source_revisions']=len(list((root/'revisions').glob('*.py')))
    log=(root/'server.log').read_text() if (root/'server.log').exists() else ''
    ids={x.get('response_id') for x in ends}; server=[]
    for line in re.sub(r'\x1b\[[0-9;]*m','',log).splitlines():
        if 'engine: done ' not in line: continue
        kv={k:v.strip('"') for k,v in re.findall(r'(\w+)=("[^"]*"|\S+)',line)}
        if kv.get('id') in ids and kv.get('id') not in {x['id'] for x in server}: server.append(kv)
    result['server_request_count']=len(server)
    for key in ['generated','prefill_ms','decode_ms','queue_ms','prompt_tokens','cached_prompt_tokens']:
        result[key]=sum(int(x.get(key,0)) for x in server)
    result['decode_tok_s']=result['generated']*1000/result['decode_ms'] if result['decode_ms'] else None
    fresh=result['prompt_tokens']-result['cached_prompt_tokens']
    result['new_prompt_tokens']=fresh
    result['new_prefill_tok_s']=fresh*1000/result['prefill_ms'] if result['prefill_ms'] else None
    result['server_request_ids']=[x['id'] for x in server]
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('shape',choices=EXPECTED);p.add_argument('root',type=pathlib.Path);p.add_argument('--arm',default='stock');p.add_argument('--timeout',type=int,default=1000);p.add_argument('--port',type=int,default=8987);p.add_argument('--guidance',type=pathlib.Path);a=p.parse_args()
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=False)
    for name in ['workspace','requests','sessions','config','revisions']: (root/name).mkdir()
    config=json.loads((HERE/'models.json').read_text());config['providers']['dsx-cube']['baseUrl']=f'http://127.0.0.1:{a.port}/v1'
    (root/'config/models.json').write_text(json.dumps(config,indent=2)+'\n')
    session='dsx-geometry-'+root.name
    pane=command(['tmux','new-session','-d','-s',session,'-x','110','-y','32','-P','-F','#{pane_id}']).strip()
    task=f'''Write render.py, a Python 3 program using only the standard library, that animates a solid regular 3D {a.shape} rotating about two axes. Shade its faces according to how much they face the viewer. Draw with Unicode braille characters (U+2800-U+28FF; each cell is a 2x4 dot grid), filling the terminal. Read its size every frame with shutil.get_terminal_size, adapt to resizing, aim for about 30 fps, and hide the cursor and restore it on Ctrl-C.

Save the actual mesh used by render.py in mesh.json, with fields "vertices" (lists of x,y,z coordinates) and "faces" (lists of zero-based vertex indices in perimeter order). Load this file from the script directory. Use a centered regular {a.shape}, with each face a convex polygon and every edge the same length.

Your display is tmux pane {pane}. Never run the animation in your own shell because it does not exit. Run it using: tmux send-keys -t {pane} 'cd {root}/workspace && python3 render.py' Enter. Inspect using: tmux capture-pane -p -t {pane}. Before restarting after an edit, send: tmux send-keys -t {pane} C-c. When it looks right, leave it running there.

Work only in this task directory and use only the named display pane. Do not access services, other agents, other tmux panes, credentials, or other project directories. Do not install packages. After checking the animation, stop making tool calls and give a brief completion summary.
'''
    if a.guidance: task+='\n'+a.guidance.read_text()
    (root/'TASK.md').write_text(task)
    health=json.load(urllib.request.urlopen('http://192.168.0.102:8090/health',timeout=10));assert health['status']=='ok' and health['pending']==0,health
    started={'wall':time.time(),'shape':a.shape,'arm':a.arm,'pane':pane,'task_sha256':hashlib.sha256(task.encode()).hexdigest(),'health':health,'timeout_s':a.timeout}
    (root/'started.json').write_text(json.dumps(started,indent=2)+'\n')
    env=dict(os.environ,GAUNTLET_RUN=str(root),GAUNTLET_PORT=str(a.port),PI_CODING_AGENT_DIR=str(root/'config'),PI_TELEMETRY='0');env.pop('NODE_OPTIONS',None)
    proxy_log=(root/'proxy.log').open('w');agent_log=(root/'agent.jsonl').open('w')
    proxy=subprocess.Popen([sys.executable,str(HERE/'proxy.py')],env=env,stdout=proxy_log,stderr=subprocess.STDOUT,start_new_session=True)
    agent=None
    try:
        for _ in range(50):
            try: urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health',timeout=1);break
            except OSError: time.sleep(.1)
        else: raise RuntimeError('proxy did not start')
        args=['pi','--offline','--provider','dsx-cube','--model','deepseek-v4.1-flash:off','--tools','read,bash,edit,write','--no-extensions','--no-mcp','--no-skills','--no-prompt-templates','--no-context-files','--session-dir',str(root/'sessions'),'--name',f'DSX {a.shape} {a.arm}','--mode','json','--print',task]
        agent=subprocess.Popen(args,cwd=root/'workspace',env=env,stdout=agent_log,stderr=subprocess.STDOUT,start_new_session=True)
        (root/'runtime.json').write_text(json.dumps({'runner_pid':os.getpid(),'agent_pid':agent.pid,'proxy_pid':proxy.pid,'argv':args},indent=2)+'\n')
        last=None;attempt=0;first_pass=False
        while agent.poll() is None and time.time()-started['wall']<a.timeout:
            source=root/'workspace/render.py';mesh=root/'workspace/mesh.json'
            if source.exists() and mesh.exists():
                digest=hashlib.sha256(source.read_bytes()+mesh.read_bytes()).hexdigest()
                if digest!=last:
                    last=digest;attempt+=1;(root/'revisions'/f'{attempt:03}.py').write_bytes(source.read_bytes());(root/'revisions'/f'{attempt:03}.mesh.json').write_bytes(mesh.read_bytes())
                    result=judge(root/'workspace',a.shape,root/f'judge-{attempt:03}.json')
                    print(json.dumps({'event':'judge','attempt':attempt,'elapsed_s':time.time()-started['wall'],'pass':result['pass'],'error':result.get('error')}),flush=True)
                    first_pass|=result['pass']
            time.sleep(2)
        if agent.poll() is None:
            os.killpg(agent.pid,signal.SIGTERM)
            try: agent.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(agent.pid,signal.SIGKILL);agent.wait()
            started['timed_out']=True
        started['agent_exit']=agent.returncode;started['agent_wall_s']=time.time()-started['wall']
        (root/'started.json').write_text(json.dumps(started,indent=2)+'\n')
        judge(root/'workspace',a.shape,root/'final-judge.json')
        (root/'display.txt').write_text(command(['tmux','capture-pane','-p','-t',pane]))
        (root/'server.log').write_text(command(['ssh','-o','BatchMode=yes','192.168.0.102','cat /home/lxe/ai/ds41/serve.out /home/lxe/ai/ds41/codex-gauntlet-server.log 2>/dev/null']))
        print(json.dumps(summarize(root),indent=2),flush=True)
    finally:
        if agent is not None and agent.poll() is None: os.killpg(agent.pid,signal.SIGTERM);agent.wait(timeout=10)
        command(['tmux','send-keys','-t',pane,'C-c']);time.sleep(.2)
        command(['tmux','kill-session','-t',session])
        proxy.terminate();proxy.wait(timeout=10);proxy_log.close();agent_log.close()

if __name__=='__main__':main()
