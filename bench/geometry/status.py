#!/usr/bin/env python3
"""Read recorded gauntlet activity without sending requests to the model."""
import argparse, json, os, pathlib, time

def read_json(path, default=None):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return default

def records(path):
    try: lines=path.read_text().splitlines()
    except OSError: return []
    result=[]
    for line in lines:
        try: result.append(json.loads(line))
        except ValueError: pass  # A streaming writer may not have finished its last line.
    return result

def snapshot(root):
    started=read_json(root/'started.json',{})
    summary=read_json(root/'summary.json',{})
    runtime=read_json(root/'runtime.json',{})
    live=False
    if runtime.get('agent_pid'):
        try: os.kill(runtime['agent_pid'],0);live=True
        except OSError: pass
    ended='agent_exit' in started
    metrics=records(root/'request-metrics.jsonl')
    ends=[x for x in metrics if x.get('event')=='end']
    last=metrics[-1] if metrics else {}
    active=live and not ended and last.get('event')=='start'
    events=records(root/'requests'/f"{last.get('number',0):03}.events.jsonl") if active else []
    deltas=[e for e in events if any(e.get(k+'_chars') for k in ['content','tool_calls','reasoning_content'])]
    latest=deltas[-1] if deltas else {}
    kind=next((label for key,label in [('tool_calls','code/tool arguments'),('content','text'),('reasoning_content','reasoning')] if latest.get(key+'_chars')),None)
    phase=('timed out' if started.get('timed_out') else 'completed') if ended else ('model streaming '+kind if kind else 'waiting for first model delta') if active else 'agent between requests' if live else 'agent not running'
    check=read_json(root/'latest-check.json',{})
    return {'shape':started.get('shape'),'arm':started.get('arm'),'phase':phase,
        'elapsed_s':round(started.get('agent_wall_s',time.time()-started.get('wall',time.time())),1),
        'completed_requests':len(ends),'responses_with_usage':sum(bool(x.get('usage')) for x in ends),'reported_output_tokens':sum((x.get('usage') or {}).get('completion_tokens',0) for x in ends),
        'active_request':last.get('number') if active else None,'active_stream_deltas':len(deltas),
        'last_delta_age_s':round(max(0,time.time()-last['start']-latest['t']),1) if latest else None,
        'last_automated_pass':check.get('pass'),'functional_pass':summary.get('functional_pass'),
        'decode_tok_s':summary.get('decode_tok_s'),'run_directory':str(root)}

def format_status(status):
    return '\n'.join(f'{key}: {value}' for key,value in status.items())+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=pathlib.Path);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        print(format_status(snapshot(a.root)),flush=True)
        if not a.watch:break
        time.sleep(5)
