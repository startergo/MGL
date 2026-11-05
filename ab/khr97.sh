#!/bin/bash
# ab/khr97.sh <lib> <outtag> <group>... — run KHR-GL46 groups against one lib, per-group tally
set -u
LIB="$1"; TAG="$2"; shift 2
rm -f /tmp/mgl_cts_modules
ln -sfn ~/MGL-cts/build/external/openglcts/modules /tmp/mgl_cts_modules || exit 2
mkdir -p /tmp/r97h
cd /tmp/mgl_cts_modules || exit 2
for G in "$@"; do
  OUT="/tmp/r97h/${TAG}_${G}"
  MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 perl -e 'alarm shift; exec @ARGV' 1500 ./glcts --deqp-case="KHR-GL46.${G}.*" \
    --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
    --deqp-surface-width=256 --deqp-surface-height=256 \
    --deqp-log-filename="$OUT.qpa" --deqp-log-images=disable > "$OUT.out" 2> "$OUT.err"
  rc=$?
  n=$(grep -c '^#beginTestCaseResult' "$OUT.qpa" || true)
  p=$(grep -c 'StatusCode="Pass"' "$OUT.qpa" || true)
  echo "== $G rc=$rc cases=$n pass=$p"
  python3 - "$OUT.qpa" <<'PY'
import sys,re
case=None; bad=[]
for line in open(sys.argv[1],errors='replace'):
    if line.startswith('#beginTestCaseResult '): case=line.split()[1]
    m=re.search(r'StatusCode="(\w+)"',line)
    if m and case:
        if m.group(1)!='Pass': bad.append((m.group(1),case))
        case=None
for s,c in bad: print("   ",s,c)
PY
done
