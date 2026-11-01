#!/bin/bash
# cts_case.sh — run individual KHR-GL46 cases against one libmgl.dylib (gl46-main
# config: 64x64 fbo surface).  Usage: bash ab/cts_case.sh <libmgl.dylib> <case>...
set -u
LIB=${1:?libmgl.dylib}
shift
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
for C in "$@"; do
  MGL_LIBRARY="$LIB" perl -e 'alarm shift; exec @ARGV' 900 \
    "$G/glcts" \
    --deqp-archive-dir=/tmp/mgl_cts_data \
    --deqp-case="$C" \
    --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
    --deqp-surface-width=64 --deqp-surface-height=64 \
    --deqp-log-images=disable --deqp-log-shader-sources=disable \
    --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
    --deqp-log-filename=/tmp/cts_case.qpa > /tmp/cts_case.out 2>/tmp/cts_case.err
  rc=$?
  r=$(grep -o 'StatusCode="[A-Za-z]*"' /tmp/cts_case.qpa | tail -1)
  printf '%-72s %-24s rc=%d\n' "$(basename "$C")" "${r:-none}" "$rc"
done
