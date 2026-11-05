#!/bin/bash
# cts_el90.sh — KHR-Single-GL46.enhanced_layouts subgroups, one glcts process per
# subgroup (flat case names, so we drive glcts with --deqp-caselist-file subsets).
# KHR-Single config matches the official run: 64x64 fbo, base-seed 1.
#   usage: OUTDIR=/tmp/el90 bash ab/cts_el90.sh <subgroup> [...]   (default: all)
# Subgroups: ssb uniform xfb varying glsl output input vertex api fragment
set -u
LIB=${MGL_LIBRARY:-$(pwd)/libmgl.dylib}
OUTDIR=${OUTDIR:-/tmp/el90}
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
CL=$G/gl_cts/data/mustpass/gl/khronos_mustpass_single/main/gl46-khr-single.txt
mkdir -p "$OUTDIR"
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
SUBGROUPS=${*:-"ssb uniform xfb varying glsl output input vertex api fragment"}
for G_ in $SUBGROUPS; do
  LST=$OUTDIR/$G_.lst
  grep "^KHR-Single-GL46\.enhanced_layouts\.$G_" $CL > $LST
  if [ ! -s $LST ]; then echo "$G_: no cases"; continue; fi
  MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 nice -n 10 perl -e 'alarm shift; exec @ARGV' 3600 \
    "$G/glcts" \
    --deqp-archive-dir=/tmp/mgl_cts_data \
    --deqp-caselist-file=$LST \
    --deqp-gl-context-type=mgl --deqp-surface-type=fbo --deqp-visibility=hidden \
    --deqp-surface-width=64 --deqp-surface-height=64 --deqp-base-seed=1 \
    --deqp-watchdog=disable --deqp-log-images=disable --deqp-log-shader-sources=disable \
    --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
    --deqp-log-filename="$OUTDIR/$G_.qpa" > "$OUTDIR/$G_.out" 2> "$OUTDIR/$G_.err"
  rc=$?
  awk '/^#beginTestCaseResult /{c=$2} /<Result StatusCode=/{match($0,/StatusCode="[A-Za-z]+"/); s=substr($0,RSTART+12,RLENGTH-13); if (s!="Pass") print s, c}' "$OUTDIR/$G_.qpa" \
    | sed 's/^/  /' > "$OUTDIR/$G_.fails"
  p=$(grep -c '^#beginTestCaseResult' "$OUTDIR/$G_.qpa")
  f=$(grep -c '^  Fail' "$OUTDIR/$G_.fails" || true)
  ns=$(grep -c '^  NotSupported' "$OUTDIR/$G_.fails" || true)
  cr=$(grep -c '^  Crash' "$OUTDIR/$G_.fails" || true)
  printf '%-10s n=%-5d Pass=%-5d Fail=%-4d NS=%-3d Crash=%d rc=%d\n' \
    "$G_" "$p" "$((p-f-ns-cr))" "$f" "$ns" "$cr" "$rc"
done
