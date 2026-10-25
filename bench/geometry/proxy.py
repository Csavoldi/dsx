#!/usr/bin/env python3
import http.server,http.client,json,pathlib,time,threading,itertools,os
ROOT=pathlib.Path(os.environ['GAUNTLET_RUN'])
IDS=itertools.count(1);LOCK=threading.Lock()
def record(x):
 with LOCK:
  with (ROOT/'request-metrics.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
class Handler(http.server.BaseHTTPRequestHandler):
 protocol_version='HTTP/1.1'
 def log_message(self,*args):pass
 def do_GET(self):
  if self.path=='/health':
   b=json.dumps({'status':'ok','run':str(ROOT),'pid':os.getpid(),'nonce':os.environ.get('GAUNTLET_PROXY_NONCE')}).encode();self.send_response(200);self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
  c=http.client.HTTPConnection('192.168.0.102',8090,timeout=10);c.request('GET',self.path);r=c.getresponse();b=r.read();self.send_response(r.status);self.send_header('Content-Type',r.getheader('Content-Type','application/json'));self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);c.close()
 def do_POST(self):
  num=next(IDS);t0=time.time();mono=time.perf_counter();body=self.rfile.read(int(self.headers.get('Content-Length','0')));prefix=ROOT/'requests'/f'{num:03}'
  data=json.loads(body);data['seed']=(int(os.environ['GAUNTLET_SEED_BASE'])+num)%(2**53);data['temperature']=1.0;data['top_p']=1.0;body=json.dumps(data,ensure_ascii=False).encode();prefix.with_suffix('.request.json').write_bytes(body)
  meta={'number':num,'seed':data['seed'],'start':t0,'path':self.path,'request_bytes':len(body),'message_count':len(data.get('messages',[])),'max_tokens':data.get('max_tokens'),'temperature':data.get('temperature'),'top_p':data.get('top_p'),'reasoning_effort':data.get('reasoning_effort'),'stream':data.get('stream'),'content_deltas':0,'tool_deltas':0,'reasoning_deltas':0,'usage':None};prev=None;gaps=[];conn=None
  record(dict(meta,event='start'))
  try:
   conn=http.client.HTTPConnection('192.168.0.102',8090,timeout=600);conn.request('POST',self.path,body,{'Content-Type':'application/json'});response=conn.getresponse();meta['status']=response.status;meta['headers_s']=time.perf_counter()-mono
   self.send_response(response.status);self.send_header('Content-Type',response.getheader('Content-Type','application/json'));self.send_header('Connection','close');self.end_headers();self.close_connection=True
   with prefix.with_suffix('.response.sse').open('wb') as raw, prefix.with_suffix('.events.jsonl').open('w') as events:
    while True:
     line=response.readline()
     if not line:break
     now=time.perf_counter()-mono;raw.write(line);raw.flush()
     if line.startswith(b'data: ') and line[6:].strip()!=b'[DONE]':
      try:
       x=json.loads(line[6:]);ev={'t':now};meta.setdefault('response_id',x.get('id'))
       if x.get('usage'):meta['usage']=x['usage'];ev['usage']=x['usage']
       for choice in x.get('choices',[]):
        d=choice.get('delta',{});active=False
        for k,counter in [('content','content_deltas'),('tool_calls','tool_deltas'),('reasoning_content','reasoning_deltas')]:
         if d.get(k):
          meta[counter]+=1;ev[k+'_chars']=len(json.dumps(d[k],ensure_ascii=False));active=True
          meta.setdefault('first_'+k+'_s',now)
        if active:
         meta.setdefault('first_delta_s',now)
         if prev is not None:gaps.append(now-prev)
         prev=now
        if choice.get('finish_reason'):meta['finish_reason']=choice['finish_reason']
       events.write(json.dumps(ev)+'\n');events.flush()
      except (ValueError,TypeError):pass
     self.wfile.write(line);self.wfile.flush()
  except Exception as e:meta['error']=repr(e)
  finally:
   if conn:conn.close()
   meta['elapsed_s']=time.perf_counter()-mono;meta['end']=time.time()
   if gaps:
    gs=sorted(gaps);meta['delta_gap_p50_s']=gs[len(gs)//2];meta['delta_gap_p99_s']=gs[min(len(gs)-1,int(.99*len(gs)))];meta['delta_gap_max_s']=gs[-1]
   record(dict(meta,event='end'));prefix.with_suffix('.metrics.json').write_text(json.dumps(meta,indent=2)+'\n')
http.server.ThreadingHTTPServer(('127.0.0.1',int(os.environ.get('GAUNTLET_PORT','8987'))),Handler).serve_forever()
