#!/usr/bin/env python3
"""Audit: every entry point required by every ADVERTISED MGL extension must be
exported by libmgl.dylib (dlsym-visible).  Usage:
    python3 ab/ext_entry_audit.py [path/to/libmgl.dylib]
Reads the extension strings from MGL/src/get.c (+compat list in compat.c),
commands per extension from external/OpenGL-Registry/xml/gl.xml.
"""
import re, subprocess, sys, xml.etree.ElementTree as ET

LIB = sys.argv[1] if len(sys.argv) > 1 else "build/libmgl.dylib"
GLXML = "external/OpenGL-Registry/xml/gl.xml"

# ---- exported gl* symbols of the dylib -------------------------------------
out = subprocess.run(["nm", "-gU", LIB], capture_output=True, text=True).stdout
exported = set()
for line in out.splitlines():
    parts = line.split()
    if len(parts) >= 3 and parts[1] == "T" and parts[2].startswith("_gl"):
        exported.add(parts[2][1:])

# ---- extension list advertised by MGL --------------------------------------
src = open("MGL/src/get.c").read()
exts = []
for m in re.finditer(r'"(GL_[A-Za-z0-9_]+)"', src):
    exts.append(m.group(1))
# compat-profile extension additions live in compat.c / glm_params.c
for f in ("MGL/src/compat.c", "MGL/src/glm_params.c", "MGL/src/state_ext.c"):
    try:
        s = open(f).read()
    except OSError:
        continue
    for m in re.finditer(r'"(GL_[A-Za-z0-9_]+)"', s):
        exts.append(m.group(1))

# keep only real extension names (GL_X_Y_Z uppercase, no enums like GL_DEPTH...)
tree = ET.parse(GLXML)
cmds_by_ext = {}
for ext in tree.getroot().iter("extension"):
    name = ext.get("name")
    cmds = []
    for req in ext.findall("require"):
        # skip requires pulled in only under another feature/profile
        for c in req.findall("command"):
            cmds.append(c.get("name"))
        for ty in req.findall("type"):
            pass
    cmds_by_ext[name] = cmds

seen = set()
missing = []
for e in sorted(set(exts)):
    if e in seen:
        continue
    seen.add(e)
    cmds = cmds_by_ext.get(e)
    if cmds is None:
        continue  # enum-only or unknown
    miss = [c for c in cmds if c not in exported]
    if miss:
        missing.append((e, miss))

print(f"advertised strings scanned: {len(seen)}; exported gl* entry points: {len(exported)}")
if not missing:
    print("ALL advertised-extension entry points resolve")
for e, miss in missing:
    print(f"MISSING {e}: {', '.join(miss)}")
