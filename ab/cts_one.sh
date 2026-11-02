#!/bin/bash
# cts_one.sh <case-name>... — run individual CTS cases (any config) through glcts,
# print one status line each.  QPA/logs under /tmp/ctsone (reused per case).
set -u
LIB=${MGL_LIBRARY:-$(pwd)/libmgl.dylib}
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
D=/tmp/ctsone
mkdir -p "$D"
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
for C in "$@"; do
  Q=$D/$(echo "$C" | tr '/.' '__').qpa
  MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 nice -n 10 perl -e 'alarm shift; exec @ARGV' 600 \
    "$G/glcts" \
    --deqp-archive-dir=/tmp/mgl_cts_data \
    --deqp-case="$C" \
    --deqp-gl-context-type=mgl --deqp-surface-type=fbo --deqp-visibility=hidden \
    --deqp-surface-width=64 --deqp-surface-height=64 --deqp-base-seed=1 \
    --deqp-watchdog=disable --deqp-log-images=disable --deqp-log-shader-sources=disable \
    --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
    --deqp-log-filename="$Q" > "$D/out" 2> "$D/err"
  awk '/^#beginTestCaseResult /{c=$2} /^ <Result StatusCode=/{s=$2; gsub(/StatusCode=|"|>/,"",s); print s, c}' "$Q"
done
