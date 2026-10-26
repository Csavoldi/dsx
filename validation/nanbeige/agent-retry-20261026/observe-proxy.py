import http.server,http.client,json,pathlib,time,threading,datetime
root=pathlib.Path(__file__).parent/'receipts';lock=threading.Lock();counter=0
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):self.forward()
 def do_POST(self):self.forward()
 def forward(self):
  global counter
  with lock:counter+=1;number=counter
  body=self.rfile.read(int(self.headers.get('Content-Length',0)))
  state={'request':number,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path':self.path,'reasoning_characters':0,'content_characters':0,'argument_characters':0,'tool_names':[],'chunks':0,'done':False};start=time.monotonic();last=0;buf=b''
  if body:(root/f'request-{number:04}.json').write_bytes(body)
  def save():
   state['elapsed_seconds']=round(time.monotonic()-start,3);(root/f'stream-{number:04}.json').write_text(json.dumps(state,indent=2)+'\n')
  upstream=http.client.HTTPConnection('127.0.0.1',8095,timeout=600)
  try:
   upstream.request(self.command,self.path,body=body,headers={'Content-Type':'application/json'});r=upstream.getresponse();self.send_response(r.status)
   self.send_header('Content-Type',r.getheader('Content-Type','application/json'));self.send_header('Cache-Control','no-cache');self.send_header('Connection','close');self.end_headers()
   while True:
    data=r.read1(8192)
    if not data:break
    self.wfile.write(data);self.wfile.flush();buf+=data
    while b'\n' in buf:
     line,buf=buf.split(b'\n',1)
     if not line.startswith(b'data:'):continue
     try:d=json.loads(line[5:].strip())
     except:continue
     state['chunks']+=1
     if d.get('usage'):state['usage']=d['usage']
     for c in d.get('choices',[]):
      delta=c.get('delta',{});state['reasoning_characters']+=len(delta.get('reasoning_content') or '');state['content_characters']+=len(delta.get('content') or '')
      for call in delta.get('tool_calls',[]):
       f=call.get('function',{});state['argument_characters']+=len(f.get('arguments') or '')
       if f.get('name'):state['tool_names'].append(f['name'])
      if c.get('finish_reason'):state['finish_reason']=c['finish_reason']
    if time.monotonic()-last>1:save();last=time.monotonic()
   state['done']=True
  except (BrokenPipeError,ConnectionResetError):state['client_disconnected']=True
  except Exception as e:state['error']=str(e)
  finally:save();upstream.close();self.close_connection=True
http.server.ThreadingHTTPServer(('127.0.0.1',8097),Handler).serve_forever()
