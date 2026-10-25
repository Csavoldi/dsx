#!/usr/bin/env python3
"""Measure serving latency for saved dropped/retained-feedback request histories.

Run only in an owned serving window. This measures cache behavior, not task quality.
"""
import argparse,hashlib,json,pathlib,secrets,time,urllib.request

def main():
    p=argparse.ArgumentParser();p.add_argument('--server',required=True);p.add_argument('--previous',type=pathlib.Path,required=True);p.add_argument('--dropped',type=pathlib.Path,required=True);p.add_argument('--retained',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    paths={'setup':a.previous,'dropped':a.dropped,'retained':a.retained};fixtures={k:json.loads(v.read_text()) for k,v in paths.items()};seed=secrets.randbits(53);rows=[]
    manifest={'seed_base':seed,'fixtures':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in paths.items()},'order':['dropped','retained','retained','dropped'],'limit':'One-token request latency/cache replay on saved tool histories; not a coding completion or quality comparison. Setup requests are excluded from target latency.'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for case in manifest['order']:
        for role in ['setup',case]:
            number=len(rows)+1;body=dict(fixtures[role]);body.update(seed=seed+number,temperature=1.0,top_p=1.0,max_tokens=1,stream=False)
            (a.output/f'{number:03d}.request.json').write_text(json.dumps(body,ensure_ascii=False,indent=2)+'\n')
            request=urllib.request.Request(a.server.rstrip('/')+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            start=time.time();clock=time.perf_counter()
            with urllib.request.urlopen(request,timeout=240) as response:result=json.loads(response.read())
            row={'number':number,'case':case,'role':role,'start':start,'elapsed_s':time.perf_counter()-clock,'seed':body['seed'],'response_id':result.get('id'),'usage':result.get('usage')}
            assert result.get('id') and result.get('usage'),result
            (a.output/f'{number:03d}.response.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');rows.append(row)
            (a.output/'requests.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
