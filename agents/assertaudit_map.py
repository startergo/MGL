#!/usr/bin/env python3
"""Map each removed assert(0) to its enclosing function in the OLD tree and in the NEW tree."""
import subprocess, re, sys

BASE = "792cbab"

files = subprocess.run(["git", "diff", "--name-only", BASE, "HEAD", "--", "MGL/src"],
                       capture_output=True, text=True, check=True).stdout.split()

def blob_text(rev, path):
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None

def func_at(lines, n):
    """return function name for 1-based line n (C/ObjC heuristic)"""
    pat_c = re.compile(r'^[A-Za-z_][A-Za-z0-9_ \*]*\b([A-Za-z_][A-Za-z0-9_]*)\s*\(')
    pat_m = re.compile(r'^\s*-\s*\(([^)]*)\)\s*([A-Za-z_][A-Za-z0-9_]*)')
    best = None
    for i in range(min(n, len(lines)) - 1, -1, -1):
        l = lines[i]
        mm = pat_m.match(l)
        if mm:
            return f"- {mm.group(2)}"
        mc = pat_c.match(l)
        if mc and not l.lstrip().startswith(('if','for','while','switch','return','} else')):
            best = mc.group(1)
            # keep scanning for a better (nearer) match? take the nearest
            return best
    return best or "?"

out = []
for path in files:
    old = blob_text(BASE, path)
    diff = subprocess.run(["git", "diff", "-U0", BASE, "HEAD", "--", path],
                          capture_output=True, text=True, check=True).stdout
    if old is None:
        continue
    oldlines = old.split("\n")
    seen_hunk = False
    for line in diff.split("\n"):
        if not seen_hunk:
            if line.startswith("@@"):
                seen_hunk = True
            else:
                continue
        m = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
        if m:
            oldstart = int(m.group(1)); oldcnt = int(m.group(2) or 1)
            newstart = int(m.group(3))
            newline = newstart
            oldline = oldstart
            continue
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            body = line[1:]
            if re.search(r'\bassert\s*\(\s*(0|false)\s*\)', body):
                fn = func_at(oldlines, oldline)
                out.append((path, oldline, fn, body.strip()))
            oldline += 1
        elif line.startswith("+"):
            newline += 1
        else:
            oldline += 1
            newline += 1

for path, ln, fn, body in out:
    print(f"{path}:{ln}\t{fn}\t{body}")
print(f"TOTAL {len(out)}", file=sys.stderr)
