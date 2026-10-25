"""A small VT screen reader for ASCII/braille full-frame renderer certification.

Unsupported controls are reported. This is not a general terminal emulator.
"""
import collections,math,re
CSI=re.compile(r'\x1b\[([0-9;?]*)([A-Za-z])')
DEFAULT=(220,220,220)
class Screen:
    def __init__(self,cols,rows):
        self.cols,self.rows=cols,rows;self.x=self.y=0;self.wrap=False;self.color=DEFAULT;self.unknown=set();self.written=set();self.clear()
    def clear(self):self.cells=[[(' ',DEFAULT) for _ in range(self.cols)] for _ in range(self.rows)]
    def erase(self,mode,line=False):
        for y in ([self.y] if line else range(self.rows)):
            for x in range(self.cols):
                relative=(x-self.x) if line else (y-self.y)*self.cols+x-self.x
                if mode==2 or (mode==0 and relative>=0) or (mode==1 and relative<=0):self.cells[y][x]=(' ',self.color);self.written.add((x,y))
    def feed(self,text):
        i=0
        while i<len(text):
            c=text[i]
            if c=='\x1b':
                match=CSI.match(text,i)
                if not match:self.unknown.add(text[i:i+2]);i+=2;continue
                raw,cmd=match.groups();private=raw.startswith('?');values=[int(v or 0) for v in raw.lstrip('?').split(';')];n=values[0] or 1;i=match.end()
                if cmd!='m':self.wrap=False
                if cmd in 'Hf':self.y=min(self.rows-1,max(0,n-1));self.x=min(self.cols-1,max(0,(values[1] if len(values)>1 else 1)-1))
                elif cmd=='A':self.y=max(0,self.y-n)
                elif cmd=='B':self.y=min(self.rows-1,self.y+n)
                elif cmd=='C':self.x=min(self.cols-1,self.x+n)
                elif cmd=='D':self.x=max(0,self.x-n)
                elif cmd=='G':self.x=min(self.cols-1,max(0,n-1))
                elif cmd=='J':self.erase(values[0])
                elif cmd=='K':self.erase(values[0],True)
                elif cmd=='m':
                    j=0
                    while j<len(values):
                        v=values[j]
                        if v==0 or v==39:self.color=DEFAULT
                        elif v==38 and j+4<len(values) and values[j+1]==2:self.color=tuple(values[j+2:j+5]);j+=4
                        elif v==38 and j+2<len(values) and values[j+1]==5:
                            idx=values[j+2]
                            if idx>=232:self.color=(8+10*(idx-232),)*3
                            elif idx>=16:
                                z=idx-16;levels=[0,95,135,175,215,255];self.color=(levels[z//36],levels[z//6%6],levels[z%6])
                            else:self.color=((idx%8)&1 and 205 or 0,(idx%8)&2 and 205 or 0,(idx%8)&4 and 205 or 0)
                            j+=2
                        elif 30<=v<=37:self.color=tuple(205 if (v-30)&bit else 0 for bit in [1,2,4])
                        elif v not in [1,2,22]:self.unknown.add('SGR '+str(v))
                        j+=1
                elif private and cmd in 'hl' and values[0] in [25,1049,7]:
                    if values[0]==1049 and cmd=='h':self.clear();self.x=self.y=0
                else:self.unknown.add(raw+cmd)
                continue
            i+=1
            if c=='\r':self.x=0;self.wrap=False;continue
            if c=='\n':
                self.y+=1;self.wrap=False
                if self.y>=self.rows:self.cells.pop(0);self.cells.append([(' ',DEFAULT)]*self.cols);self.y=self.rows-1
                continue
            if ord(c)<32:continue
            if not (c.isascii() or 0x2800<=ord(c)<=0x28ff):self.unknown.add('non-ASCII/braille character')
            if self.wrap:self.x=0;self.y+=1;self.wrap=False
            if self.y>=self.rows:self.cells.pop(0);self.cells.append([(' ',DEFAULT)]*self.cols);self.y=self.rows-1
            self.cells[self.y][self.x]=(c,self.color);self.written.add((self.x,self.y));self.x+=1
            if self.x>=self.cols:self.x=self.cols-1;self.wrap=True
    def dots(self):return {(x,y) for y,row in enumerate(self.cells) for x,(c,_) in enumerate(row) if 0x2801<=ord(c)<=0x28ff}

def shade_check(screen):
    points=screen.dots();interior=[screen.cells[y][x] for x,y in points if all((x+dx,y+dy) in points for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)])]
    if len(interior)<12:return False
    colors=collections.Counter(rgb for _,rgb in interior);density=collections.Counter((ord(c)-0x2800).bit_count() for c,_ in interior)
    significant_colors=[rgb for rgb,count in colors.items() if count>=max(3,len(interior)*.05)]
    significant_density=[v for v,count in density.items() if count>=max(3,len(interior)*.05)]
    color_shading=any(math.dist(a,b)>=35 for a in significant_colors for b in significant_colors)
    density_shading=any(abs(a-b)>=2 for a in significant_density for b in significant_density)
    return color_shading or density_shading

def certify(frames,cols,rows):
    screen=Screen(cols,rows);checks=[]
    for frame in frames:
        screen.written.clear();screen.feed('\x1b[H'+frame);fresh=Screen(cols,rows);fresh.feed('\x1b[H'+frame)
        actual=screen.dots();clean=fresh.dots();ghost=len(actual-clean)/max(len(actual),1)
        checks.append({'ghost_fraction':ghost,'shading':shade_check(screen),'written_fraction':len(screen.written)/(cols*rows),'unknown_controls':sorted(screen.unknown|fresh.unknown)})
    checks=checks[2:]
    supported=bool(checks) and all(not c['unknown_controls'] and c['written_fraction']>=.2 for c in checks)
    ghost=max((c['ghost_fraction'] for c in checks),default=1)
    shaded=sum(c['shading'] for c in checks)/max(len(checks),1)
    return {'supported_full_frame':supported,'max_ghost_fraction':ghost,'shaded_frame_fraction':shaded,'frames':len(checks),'pass':supported and ghost<=.015 and shaded>=.2,'limit':'ASCII/braille full-frame redraw certification; shape identity and general rendering quality still require review.'}
