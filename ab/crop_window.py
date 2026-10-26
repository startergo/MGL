#!/usr/bin/env python3
"""crop the glmark2 window out of a full screenshot (2x scale assumed) and dump
an ASCII luminance thumbnail + optional PNG crop."""
import zlib, struct, sys

def load(p):
    d = open(p, 'rb').read()
    pos = 8; idat = b''
    while pos < len(d):
        ln, = struct.unpack('>I', d[pos:pos+4]); typ = d[pos+4:pos+8]
        if typ == b'IHDR':
            w, h, bd, ct = struct.unpack('>IIBB', d[pos+8:pos+18])[0:4]
        if typ == b'IDAT': idat += d[pos+8:pos+8+ln]
        pos += 12 + ln
    raw = zlib.decompress(idat)
    stride = w * 4
    def paeth(a, b, c):
        pp = a+b-c; pa, pb, pc = abs(pp-a), abs(pp-b), abs(pp-c)
        return a if pa <= pb and pa <= pc else (b if pb <= pc else c)
    out = bytearray(); prev = bytearray(stride); i = 0
    for y in range(h):
        ft = raw[i]; i += 1
        line = bytearray(raw[i:i+stride]); i += stride
        for x in range(stride):
            a = line[x-4] if x >= 4 else 0; b = prev[x]; c = prev[x-4] if x >= 4 else 0
            if ft == 1: line[x] = (line[x]+a) & 255
            elif ft == 2: line[x] = (line[x]+b) & 255
            elif ft == 3: line[x] = (line[x]+((a+b)//2)) & 255
            elif ft == 4: line[x] = (line[x]+paeth(a,b,c)) & 255
        out += line; prev = line
    return w, h, bytes(out)

def crop(w, h, px, x, y, cw, ch):
    o = bytearray()
    for yy in range(y, min(y+ch, h)):
        o += px[(yy*w+x)*4:(yy*w+x+cw)*4]
    return cw, ch, o

def thumb(w, h, px, gx=60, gy=34, label=''):
    print(label)
    for yy in range(gy):
        row = ''
        for xx in range(gx):
            qx = int((xx+0.5)*w/gx); qy = int((yy+0.5)*h/gy)
            o = (qy*w+qx)*4
            r, g, b = px[o], px[o+1], px[o+2]
            lum = (r*299+g*587+b*114)//1000
            row += ' .:-=+*#%@'[min(9, lum*10//256)]
        print(row)

def write_png(path, w, h, rgba):
    def chunk(t, data):
        c = t + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    raw = b''
    for y in range(h):
        raw += b'\x00' + rgba[y*w*4:(y+1)*w*4]
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw))
    png += chunk(b'IEND', b'')
    open(path, 'wb').write(png)

if __name__ == '__main__':
    full = sys.argv[1]; winfile = sys.argv[2]; out = sys.argv[3]
    W, H, px = load(full)
    wf = open(winfile).read().split()
    x, y = int(wf[2].split(',')[0]), int(wf[2].split(',')[1])
    cw, ch = int(wf[3].split('x')[0]), int(wf[3].split('x')[1])
    # screenshot is in points at the display scale; detect scale
    scale = 2 if W >= 2000 else 1
    x, y, cw, ch = x*scale, y*scale, cw*scale, ch*scale
    cw, ch = min(cw, W-x), min(ch, H-y)
    cw2, ch2, cpx = crop(W, H, px, x, y, cw, ch)
    write_png(out, cw2, ch2, cpx)
    thumb(cw2, ch2, cpx, label=out)
