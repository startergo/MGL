#!/bin/bash
# Run ONE CTS case (or a glob) through glcts and print the result + last lines.
# usage: bash agents/conformance/one.sh 'KHR-GL46.shaders.uniform_block.instance_array_basic_type.std140.bool' [extra glcts args...]
set -u
pat="$1"; shift
CTS="${MGL_CTS_DIR:-$HOME/MGL-cts}"
BIN="$CTS/build/external/openglcts/modules"
export MGL_LIBRARY="${MGL_LIBRARY:-$PWD/build/libmgl.dylib}"
export MGL_LOG_DIR=""   # keep MGL's own log out of the way
mkdir -p /tmp/mgl-cts-one
python3 - "$BIN" "$pat" /tmp/mgl-cts-one/one.qpa "$@" <<'EOF'
import os, subprocess, sys
bindir, pat, qpa = sys.argv[1], sys.argv[2], sys.argv[3]
extra = sys.argv[4:]
cmd = [os.path.join(bindir, "glcts"),
       "--deqp-case=" + pat,
       "--deqp-log-filename=" + qpa,
       "--deqp-surface-type=fbo",
       "--deqp-gl-context-type=mgl",
       "--deqp-visibility=hidden",
       "--deqp-surface-width=64", "--deqp-surface-height=64",
       "--deqp-log-images=disable", "--deqp-log-shader-sources=disable",
       "--deqp-log-decompiled-spirv=disable", "--deqp-log-empty-loginfo=disable",
       "--deqp-log-flush=enable", "--deqp-base-seed=0"] + extra
try:
    p = subprocess.run(cmd, cwd=bindir, capture_output=True, text=True, timeout=180)
except subprocess.TimeoutExpired:
    print("TIMEOUT"); sys.exit(0)
print(p.stdout[-2000:]); sys.stderr.write(p.stderr[-800:])
try:
    body = open(qpa).read()
    import re
    for m in re.finditer(r"#beginTestCaseResult (\S+).*?<Result StatusCode='?\"?(\w+)", body, re.S):
        print("== %s: %s" % (m.group(1), m.group(2)))
    # last non-empty lines of each case body
    for chunk in body.split("#beginTestCaseResult ")[1:]:
        name = chunk.split()[0]
        lines = [l.strip() for l in chunk.splitlines() if l.strip() and not l.startswith("<")]
        print("   %s", ) if False else None
        print("   [" + name + "] " + (lines[-1][:200] if lines else ""))
except FileNotFoundError:
    pass
EOF
