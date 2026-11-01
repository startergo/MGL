#!/usr/bin/env python3
"""Print every <InfoLog> and <Text> of the given case(s) in a qpa, plus compile errors.
usage: python3 ab/qpa_detail.py <qpa> [case-substring]"""
import re, sys
qpa, sub = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
body = open(qpa, errors="replace").read()
for chunk in body.split("#beginTestCaseResult ")[1:]:
    name = chunk.split()[0]
    if sub and sub not in name:
        continue
    m = re.search(r'<Result StatusCode="([^"]+)"[^>]*>([^<]*)', chunk)
    print("== %s: %s %s" % (name, m.group(1) if m else "?", (m.group(2).strip() if m else "")[:120]))
    for mm in re.finditer(r'<(InfoLog|Text|ShaderSource)>(.*?)</\1>', chunk, re.S):
        t = mm.group(2)
        if mm.group(1) == "ShaderSource" and len(t) > 400:
            t = t[:400] + " ...[%d bytes]" % len(t)
        t = t.strip()
        if t:
            print("   <%s>" % mm.group(1) + t[:3000])
    print()
