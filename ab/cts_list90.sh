#!/bin/bash
# cts_list90.sh <prefix> [prefix...] — run all KHR-Single-GL46 cases whose name
# starts with <prefix> in ONE glcts process (caselist-file driven), print a tally.
#   usage: OUTDIR=/tmp/l90 bash ab/cts_list90.sh enhanced_layouts.xfb_explicit_location
set -u
LIB=${MGL_LIBRARY:-$(pwd)/libmgl.dylib}
OUTDIR=${OUTDIR:-/tmp/l90}
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
CL=$G/gl_cts/data/mustpass/gl/khronos_mustpass_single/main/gl46-khr-single.txt
mkdir -p "$OUTDIR"
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
for P in "$@"; do
  N=$(echo "$P" | tr '/.' '__')
  LST=$OUTDIR/$N.lst
  grep "^KHR-Single-GL46\.$P" $CL > $LST
  if [ ! -s $LST ]; then echo "$P: no cases"; continue; fi
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
  awk '/^#beginTestCaseResult /{c=$2} /<Result StatusCode=/{match($0,/StatusCode="[A-Za-z]+"/); s=substr($0,RSTART+12,RLENGTH-13); if (s!="Pass") print s, c}' "$OUTDIR/$N.qpa" \
    | sed 's/^/  /' > "$OUTDIR/$N.fails"
  p=$(grep -c '^#beginTestCaseResult' "$OUTDIR/$N.qpa")
  f=$(grep -c '^  Fail' "$OUTDIR/$N.fails" || true)
  ns=$(grep -c '^  NotSupported' "$OUTDIR/$N.fails" || true)
  cr=$(grep -c '^  Crash' "$OUTDIR/$N.fails" || true)
  printf '%-55s n=%-5d Pass=%-5d Fail=%-4d NS=%-3d Crash=%d rc=%d\n' \
    "$P" "$p" "$((p-f-ns-cr))" "$f" "$ns" "$cr" "$rc"
done
