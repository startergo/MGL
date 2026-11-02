#!/bin/bash
# cts_deqp90.sh <name-prefix>... — run dEQP-GL45-ES31 cases (the gl45-es31-main
# config: 256x256 fbo, base-seed -1), one glcts process for ALL prefixes.
#   usage: OUTDIR=/tmp/d90 bash ab/cts_deqp90.sh functional.separate_shader.random
set -u
LIB=${MGL_LIBRARY:-$(pwd)/libmgl.dylib}
OUTDIR=${OUTDIR:-/tmp/d90}
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
CL=$G/gl_cts/data/mustpass/gl/aosp_mustpass/main/gl45-es31-main.txt
mkdir -p "$OUTDIR"
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
LST=$OUTDIR/cases.lst
: > $LST
for P in "$@"; do grep "^dEQP-GL45-ES31.$P" $CL >> $LST; done
N=$(echo "$1" | tr '/.' '__')
sort -u $LST -o $LST
wc -l < $LST | sed 's/^/cases: /'
MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 nice -n 10 perl -e 'alarm shift; exec @ARGV' 3600 \
  "$G/glcts" \
  --deqp-archive-dir=/tmp/mgl_cts_data \
  --deqp-caselist-file=$LST \
  --deqp-gl-context-type=mgl --deqp-surface-type=fbo --deqp-visibility=hidden \
  --deqp-surface-width=256 --deqp-surface-height=256 --deqp-base-seed=-1 \
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
  "$*" "$p" "$((p-f-ns-cr))" "$f" "$ns" "$cr" "$rc"
