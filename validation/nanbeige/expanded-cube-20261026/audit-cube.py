import importlib.util,json,pathlib,hashlib,math,subprocess
r=pathlib.Path(__file__).parent;p=r/'game/cube.py';out=r/'receipts';out.mkdir(exist_ok=True)
report={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'independent CPU geometry and Braille coordinate audit; no source edits or model feedback'}
try:
 spec=importlib.util.spec_from_file_location('candidate_cube',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 c=m.BrailleCanvas(10,20);c.set_dot(0,19);text=c.to_text();lines=text.splitlines()
 report["canvas_api"]="model-rewritten constructor takes character columns and dot rows; 20 dots yield5 character rows"
 report['bottom_dot_preserved']=len(lines)==5 and bool(lines[-1]) and lines[-1][0]==chr(0x2800+64)
 report['vertex_count']=len(m.CUBE_VERTICES);report['edge_count']=len(m.CUBE_EDGES)
 errs=[];frames=[];bounds=[]
 for a,b in [(.3,.7),(.9,1.3),(1.4,2.0)]:
  vs=[m.rotate(v,a,b,0) for v in m.CUBE_VERTICES]
  errs.extend(abs(sum(x*x for x in v)-sum(x*x for x in w)) for v,w in zip(vs,m.CUBE_VERTICES))
  f=m.render_frame(vs,100,26);frames.append(f);xy=[(x,y) for y,l in enumerate(f.splitlines()) for x,c in enumerate(l) if '\u2801'<=c<='\u28ff']
  bounds.append({'nonblank_cells':len(xy),'nonblank_rows':len(set(y for x,y in xy)),'bbox': [min(x for x,y in xy),max(x for x,y in xy),min(y for x,y in xy),max(y for x,y in xy)] if xy else None})
 report['rotation_norm_max_error']=max(errs);report['distinct_frames']=len(set(frames));report['frame_bounds']=bounds
except Exception as e:report['audit_error']=str(e)
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
