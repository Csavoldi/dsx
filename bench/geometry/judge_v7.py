#!/usr/bin/env python3
"""Judge geometry data and exercise the unmodified submitted renderer in a PTY."""
import argparse, ast, fcntl, hashlib, json, math, os, pathlib, pty, re, select, signal, struct, subprocess, sys, termios, time
from terminal import certify, Screen, split_frames
EXPECTED = {'cube': (8, 12, 6, 4), 'tetrahedron': (4, 6, 4, 3), 'octahedron': (6, 12, 8, 3)}

def mesh_check(path, shape):
    mesh = json.loads(path.read_text()); vertices = mesh['vertices']; faces = mesh['faces']
    nv, ne, nf, degree = EXPECTED[shape]
    assert len(vertices) == nv and len(faces) == nf, 'wrong vertex/face count'
    assert all(len(v) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v) for v in vertices), 'invalid coordinates'
    edges = {}; area = []
    sub = lambda a,b: [x-y for x,y in zip(a,b)]
    dot = lambda a,b: sum(x*y for x,y in zip(a,b))
    cross = lambda a,b: [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    scale = max(math.sqrt(dot(v,v)) for v in vertices)
    assert scale > 1e-8, 'collapsed mesh'
    center=[sum(v[i] for v in vertices)/nv for i in range(3)]
    assert max(map(abs,center)) < scale*1e-5, 'mesh is not centered'
    radii=[math.sqrt(dot(v,v)) for v in vertices]
    assert max(radii)-min(radii) < scale*1e-5, 'vertices are not on a common circumsphere'
    for f in faces:
        assert len(f) == degree and len(set(f)) == degree and all(type(i) is int and 0 <= i < nv for i in f), 'invalid face indices'
        normal = cross(sub(vertices[f[1]],vertices[f[0]]), sub(vertices[f[2]],vertices[f[0]]))
        norm = math.sqrt(dot(normal,normal)); assert norm > scale*scale*1e-6, 'degenerate face'
        distances = [dot(normal, sub(v,vertices[f[0]]))/norm for v in vertices]
        assert all(abs(distances[i]) < scale*1e-5 for i in f), 'nonplanar face'
        assert min(distances) > -scale*1e-5 or max(distances) < scale*1e-5, 'nonconvex face'
        area.append(norm)
        for a,b in zip(f, f[1:]+f[:1]):
            edge=tuple(sorted((a,b))); edges[edge]=edges.get(edge,0)+1
    assert len(edges)==ne and all(n==2 for n in edges.values()), 'mesh is not a closed manifold'
    lengths=[math.dist(vertices[a],vertices[b]) for a,b in edges]
    assert max(lengths)-min(lengths) < max(lengths)*1e-4, 'irregular edge lengths'
    assert nv-ne+nf == 2, 'wrong Euler characteristic'
    return {'pass':True, 'vertices':nv,'edges':ne,'faces':nf,'edge_length':sum(lengths)/len(lengths)}

def render_check(workspace, seconds=1.6):
    master,slave=pty.openpty()
    def size(cols,rows): fcntl.ioctl(slave,termios.TIOCSWINSZ,struct.pack('HHHH',rows,cols,0,0))
    size(80,24)
    proc=subprocess.Popen([sys.executable,'render.py'],stdin=slave,stdout=slave,stderr=slave,cwd=workspace,env=dict(os.environ,TERM='xterm-256color'),start_new_session=True)
    def read_for(duration):
        data=bytearray(); start=time.monotonic()
        while time.monotonic()-start<duration:
            if select.select([master],[],[],.04)[0]:
                try: chunk=os.read(master,262144)
                except OSError: break
                if not chunk: break
                data.extend(chunk)
            if proc.poll() is not None: break
        return bytes(data),time.monotonic()-start
    output=b''; phases=[]
    try:
        for cols,rows in [(80,24),(110,32),(45,15)]:
            size(cols,rows); data,elapsed=read_for(seconds); output+=data
            frames=split_frames(data.decode('utf-8','replace'))
            masks=[''.join(c for c in frame if 0x2800<=ord(c)<=0x28ff) for frame in frames]
            dots=[sum((ord(c)-0x2800).bit_count() for c in frame) for frame in masks]
            extents=[]
            row_masks=[]
            for frame in frames:
                screen=Screen(cols,rows);screen.feed('\x1b[H'+frame)
                row_masks.append([''.join(c for c,_ in row) for row in screen.cells])
            for rows_text in row_masks[2:-1]:
                points=[]
                for y,line in enumerate(rows_text[:rows]):
                    for x,c in enumerate(line[:cols]):
                        if 0x2801<=ord(c)<=0x28ff: points.append((x,y))
                if points:
                    xs,ys=zip(*points);extents.append(((max(xs)-min(xs)+1)/cols,(max(ys)-min(ys)+1)/rows,min(xs)>0 and max(xs)<cols-1 and min(ys)>0 and max(ys)<rows-1))
            typical=sorted(extents)[len(extents)//2] if extents else (0,0,False)
            phase={'columns':cols,'rows':rows,'fps':len(frames)/max(elapsed,.001),'frames':len(frames),'distinct_masks':len(set(masks)), 'max_dots':max(dots,default=0),'typical_width_fraction':typical[0],'typical_height_fraction':typical[1],'frames_with_margin':sum(e[2] for e in extents),'measured_extents':len(extents),'median_cells': sorted(map(len,masks))[len(masks)//2] if masks else 0}
            phase['frame_certification']=certify(frames[:-1],cols,rows)
            phase['pass']=15<=phase['fps']<=65 and phase['distinct_masks']>=8 and phase['max_dots']>=cols*rows*.12 and .2<=typical[0]<=.95 and .2<=typical[1]<=.95 and phase['frame_certification']['pass']
            phases.append(phase)
        if proc.poll() is None: os.killpg(proc.pid,signal.SIGINT)
        tail,_=read_for(.5); output+=tail
        try: proc.wait(timeout=2)
        except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
        result={'phases':phases,'exit_code':proc.returncode,'cursor_hidden':b'\x1b[?25l' in output,'cursor_restored':b'\x1b[?25h' in tail,'traceback':b'Traceback (most recent call last)' in output}
        result['pass']=all(p['pass'] for p in phases) and proc.returncode==0 and result['cursor_hidden'] and result['cursor_restored'] and not result['traceback']
        return result,output
    finally:
        if proc.poll() is None: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        os.close(master);os.close(slave)

def judge(workspace, shape, out):
    started=time.time(); result={'judge_version':7,'shape':shape,'timestamp':started,'pass':False}
    try:
        source=(workspace/'render.py').read_text(); tree=ast.parse(source)
        imports={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        imports.update((n.module or '').split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom))
        result['stdlib_only']=all(m in sys.stdlib_module_names for m in imports)
        result['source_sha256']=hashlib.sha256(source.encode()).hexdigest()
        result['mesh']=mesh_check(workspace/'mesh.json',shape) if (workspace/'mesh.json').exists() else {'pass':None,'status':'not_submitted','limit':'Geometry must be checked in source and display review.'}
        result['renderer'],data=render_check(workspace)
        out.with_suffix('.pty').write_bytes(data)
        result['pass']=result['stdlib_only'] and result['mesh']['pass'] is not False and result['renderer']['pass']
    except Exception as e: result['error']=repr(e)
    result['judge_s']=time.time()-started;out.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=pathlib.Path);p.add_argument('shape',choices=EXPECTED);p.add_argument('output',type=pathlib.Path);a=p.parse_args();print(json.dumps(judge(a.workspace,a.shape,a.output),indent=2))
