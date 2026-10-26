import json,pathlib,time,urllib.request
out=pathlib.Path(__file__).parent/'server-probe';out.mkdir(exist_ok=True)
request={'model':'nanbeige4.3-33b','messages':[{'role':'user','content':'Context data follows.\n'+('The quick brown fox jumps over the lazy dog.\n'*350)+'\nWrite a Python function named square that returns its argument multiplied by itself. Keep the answer short.'}],'temperature':0,'seed':42,'max_tokens':64,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
(out/'request.json').write_text(json.dumps(request))
rows=[]
for n in range(2):
 t=time.time();req=urllib.request.Request('http://127.0.0.1:8095/v1/chat/completions',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:d=json.load(r)
 (out/f'response-{n}.json').write_text(json.dumps(d,indent=2));rows.append(d)
 print(n,d['usage'],d['timings'],flush=True)
a,b=rows
result={'matched_response_message':a['choices'][0]['message']==b['choices'][0]['message'],'first_cached_tokens':a['usage']['prompt_tokens_details']['cached_tokens'],'second_cached_tokens':b['usage']['prompt_tokens_details']['cached_tokens'],'prompt_tokens':b['usage']['prompt_tokens'],'scope':'same deterministic request cold then reused prefix on deployed experimental kernel; no claim about all histories or full256K'}
(out/'summary.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
assert result['matched_response_message'];assert result['second_cached_tokens']>2000
