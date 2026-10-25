import json,time,urllib.request,pathlib
root=pathlib.Path('/home/lxe/nanbeige-runner/receipts')
q={'model':'nanbeige4.3-33b','temperature':0,'top_p':1,'max_tokens':2048,'stream':True,'chat_template_kwargs':{'enable_thinking':False},'tools':[{'type':'function','function':{'name':'write','description':'Write a file','parameters':{'type':'object','properties':{'path':{'type':'string'},'content':{'type':'string'}},'required':['path','content']}}}],'messages':[{'role':'user','content':'Call write exactly once. Set path to /tmp/dsx-stream-probe.py. Set content to a Python program containing 40 separate print statements, numbered 1 through 40. Include the string 世界 in every print statement. Do not explain anything.'}]}
req=urllib.request.Request('http://127.0.0.1:8095/v1/chat/completions',data=json.dumps(q).encode(),headers={'Content-Type':'application/json'})
start=time.monotonic();first=None;last=None;argparts=[];decl=[];content='';finish=None
with urllib.request.urlopen(req,timeout=120) as resp:
 for line in resp:
  if not line.startswith(b'data: '):continue
  if line.strip()==b'data: [DONE]':break
  d=json.loads(line[6:]);assert 'error' not in d,d
  for c in d.get('choices',[]):
   delta=c.get('delta',{});content+=delta.get('content','');finish=c.get('finish_reason') or finish
   for call in delta.get('tool_calls',[]):
    if call.get('id'):decl.append(call['id'])
    p=call.get('function',{}).get('arguments')
    if p:
     t=time.monotonic()-start;first=t if first is None else first;last=t;argparts.append(p)
args=json.loads(''.join(argparts));assert len(decl)==1,decl;assert finish=='tool_calls',finish;assert args['path']=='/tmp/dsx-stream-probe.py';assert len(args['content'].encode())>600;assert '世界' in args['content'];assert len(argparts)>20;assert '<tool_call>' not in content
receipt={'argument_chunks':len(argparts),'first_argument_seconds':first,'last_argument_seconds':last,'streaming_span_seconds':last-first,'declared_calls':len(decl),'finish_reason':finish,'arguments_parse_as_json':True,'unicode_survives':True,'content_bytes':len(args['content'].encode()),'raw_xml_leaked':False,'tool_executed':False}
(root/'tool-stream-live-smoke.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
