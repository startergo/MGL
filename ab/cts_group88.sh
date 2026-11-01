#!/bin/bash
# cts_group88.sh — run ONE KHR-GL46 group (or sub-pattern) against one libmgl.dylib.
#   usage: bash ab/cts_group88.sh <libmgl.dylib> <group-prefix>
# e.g. bash ab/cts_group88.sh $(pwd)/libmgl.dylib KHR-GL46.primitive_restart
set -u
LIB=${1:?libmgl.dylib}
PAT=${2:?KHR-GL46 group pattern}
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
OUT=/tmp/cts_group88
mkdir -p $OUT
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 perl -e 'alarm shift; exec @ARGV' 2400 \
  "$G/glcts" \
  --deqp-archive-dir=/tmp/mgl_cts_data \
  --deqp-case="$PAT.*" \
  --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
  --deqp-surface-width=64 --deqp-surface-height=64 \
  --deqp-log-images=disable --deqp-log-shader-sources=disable \
  --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
  --deqp-log-filename="$OUT/g.qpa" > $OUT/g.out 2> $OUT/g.err
rc=$?
p=$(grep -c '^#beginTestCaseResult' "$OUT/g.qpa")
f=$(grep -o 'StatusCode="Fail"' "$OUT/g.qpa" | wc -l | tr -d ' ')
ns=$(grep -o 'StatusCode="NotSupported"' "$OUT/g.qpa" | wc -l | tr -d ' ')
printf '%s: n=%d pass=%d fail=%d ns=%d rc=%d\n' "$PAT" "$p" "$((p-f-ns))" "$f" "$ns" "$rc"
grep -o '#beginTestCaseResult [^ ]*' "$OUT/g.qpa" | sed 's/#beginTestCaseResult //' > $OUT/all.txt
python3 - "$OUT/g.qpa" <<'EOF'
import re,sys
q=open(sys.argv[1]).read()
for m in re.finditer(r'#beginTestCaseResult (\S+)(.*?)(?=\n#beginTestCaseResult|\Z)', q, re.S):
    if 'StatusCode="Fail"' in m.group(2): print("  FAIL", m.group(1))
EOF
