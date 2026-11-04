#!/bin/bash
# cts_r101_re.sh <regex>... — run KHR-GL46 cases matching a regex from the
# gl46-main mustpass (same config as cts_gl46_90.sh), one glcts process.
#   usage: OUTDIR=/tmp/r101 bash ab/cts_r101_re.sh 'constant_expressions.*_geometry$'
set -u
LIB=${MGL_LIBRARY:-$(pwd)/libmgl.dylib}
OUTDIR=${OUTDIR:-/tmp/r101}
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
CL=$G/gl_cts/data/mustpass/gl/khronos_mustpass/main/gl46-main.txt
mkdir -p "$OUTDIR"
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
LST=$OUTDIR/cases.lst
: > $LST
for P in "$@"; do grep -E "$P" $CL >> $LST; done
N=re
sort -u $LST -o $LST
wc -l < $LST | sed 's/^/cases: /'
MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 nice -n 10 perl -e 'alarm shift; exec @ARGV' 3600 \
  "$G/glcts" \
  --deqp-archive-dir=/tmp/mgl_cts_data \
  --deqp-caselist-file=$LST \
  --deqp-gl-context-type=mgl --deqp-surface-type=fbo --deqp-visibility=hidden \
  --deqp-surface-width=64 --deqp-surface-height=64 --deqp-base-seed=1 \
  --deqp-watchdog=disable --deqp-log-images=disable --deqp-log-shader-sources=disable \
  --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
  --deqp-log-filename="$OUTDIR/$N.qpa" > "$OUTDIR/$N.out" 2> "$OUTDIR/$N.err"
rc=$?
awk '/^#beginTestCaseResult /{c=$2} /^ <Result StatusCode=/{s=$2; gsub(/StatusCode=|"|>/,"",s); if (s!="Pass") print s, c}' "$OUTDIR/$N.qpa" \
  | sed 's/^/  /' > "$OUTDIR/$N.fails"
p=$(grep -c '^#beginTestCaseResult' "$OUTDIR/$N.qpa")
f=$(grep -c '^  Fail' "$OUTDIR/$N.fails" || true)
ns=$(grep -c '^  NotSupported ' "$OUTDIR/$N.fails" || true)
cr=$(grep -c '^  Crash ' "$OUTDIR/$N.fails" || true)
printf '%-55s n=%-5d Pass=%-5d Fail=%-4d NS=%-3d Crash=%d rc=%d\n' \
  "$1" "$p" "$((p-f-ns-cr))" "$f" "$ns" "$cr" "$rc"
head -30 "$OUTDIR/$N.fails"
