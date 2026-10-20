#!/bin/bash
# helper: run a whole CTS group (prefix) in one glcts invocation, print per-case status
# usage: bash tests/cts_group.sh 'KHR-GL46.geometry_shader' [outfile] [timeout]
set -u
GRP="$1"
OUT="${2:-/tmp/cts_group}"
T="${3:-3600}"
rm -f /tmp/mgl_cts_modules
ln -sfn ~/MGL-cts/build/external/openglcts/modules /tmp/mgl_cts_modules || exit 2
LIB=$(cd "$(dirname "$0")/../build" && pwd)/libmgl.dylib
cd /tmp/mgl_cts_modules || exit 2
MGL_LIBRARY="$LIB" perl -e 'alarm shift; exec @ARGV' "$T" ./glcts --deqp-case="$GRP.*" \
  --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
  --deqp-surface-width=256 --deqp-surface-height=256 \
  --deqp-log-filename="$OUT.qpa" --deqp-log-images=disable > "$OUT.out" 2> "$OUT.err"
rc=$?
/usr/bin/python3 - "$OUT.qpa" <<'PY'
import sys, re, collections
try:
    s = open(sys.argv[1], errors='replace').read()
except OSError:
    print("no log"); sys.exit(0)
res = re.findall(r'#BeginTestCaseResult\s*\n?(\S+)\n.*?StatusCode="(\w+)"', s, re.S)
c = collections.Counter(v for _, v in res)
print("cases:", len(res), dict(c))
fails = [k for k, v in res if v != 'Pass']
open('/tmp/cts_group_fails.txt', 'w').write('\n'.join(fails))
print("fails written to /tmp/cts_group_fails.txt:", len(fails))
PY
tail -2 "$OUT.err" 2>/dev/null
exit $rc
