#!/usr/bin/env python3
"""Render saved braille terminal cells as a PNG dot mask, using only the standard library."""
import argparse,pathlib,re,struct,zlib

def preview(text,path):
    lines=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]','',text).splitlines()
    cols=max(map(len,lines),default=1);rows=max(len(lines),1);scale=4
    width,height=cols*2*scale,rows*4*scale
    image=bytearray(width*height)
    bits=[(0,0,0),(0,1,1),(0,2,2),(1,0,3),(1,1,4),(1,2,5),(0,3,6),(1,3,7)]
    for y,line in enumerate(lines):
        for x,char in enumerate(line):
            mask=ord(char)-0x2800
            if not 0<=mask<=255:continue
            for dx,dy,bit in bits:
                if mask&(1<<bit):
                    px=(2*x+dx)*scale;py=(4*y+dy)*scale
                    for j in range(scale-1): image[(py+j)*width+px:(py+j)*width+px+scale-1]=b'\xff'*(scale-1)
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    raw=b''.join(b'\0'+image[y*width:(y+1)*width] for y in range(height))
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);a=p.parse_args();preview(a.input.read_text(),a.output)
