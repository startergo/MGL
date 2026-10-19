#!/usr/bin/env python3
"""For each function that used to contain assert(0), summarize the current implementation."""
import subprocess, re, sys, os

names = set()
for line in open("/tmp/map.txt"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        names.add(p[1])
names.discard("")

root = os.popen("git rev-parse --show-toplevel").read().strip()
cur = []
for line in subprocess.run(["git", "grep", "-n", "-E",
                            r"^(void|GLboolean|GLenum|GLint|GLuint|GLsizei|const char \*|char \*|bool|struct [A-Za-z_]+ \*)[[:space:]]+\*?("
                            + "|".join(sorted(re.escape(n) for n in names)) + r")[[:space:]]*\(",
                            "--", "MGL/src"], capture_output=True, text=True).stdout.split("\n"):
    if line:
        cur.append(line)
for line in cur:
    f, ln, _ = line.split(":", 2)
    # measure function length until closing brace at col 0
    txt = open(os.path.join(root, f)).read().split("\n")
    depth = 0
    started = False
    end = ln
    for i in range(int(ln) - 1, min(int(ln) + 300, len(txt))):
        l = txt[i]
        depth += l.count("{") - l.count("}")
        if "{" in l:
            started = True
        if started and depth <= 0:
            end = i + 1
            break
    body = "\n".join(txt[int(ln) - 1:end])
    nerr = len(re.findall(r"GL_INVALID_\w+", body))
    flags = []
    if end - int(ln) < 12:
        flags.append("SHORT")
    if nerr and end - int(ln) < 30:
        flags.append(f"ERRS({nerr})")
    if flags:
        m = re.match(r"[A-Za-z_ ][A-Za-z0-9_ \*]*?([A-Za-z_][A-Za-z0-9_]*)\s*\(", txt[int(ln) - 1])
        print(f"{f}:{ln}\t{m.group(1) if m else '?'}\t{end - int(ln) + 1} lines\t{' '.join(flags)}")
