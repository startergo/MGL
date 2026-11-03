#!/bin/bash
# cts_run_list.sh — run every case in a file (one per line), print StatusCode
# usage: bash ab/cts_run_list.sh <listfile> [lib]
set -u
LIB=${2:-$(pwd)/build/libmgl.dylib}
G=$HOME/MGL-cts-new/build/external/openglcts/modules
D=/tmp/cts_$(basename "$1" .txt)
mkdir -p $D
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
while read -r C; do
  [ -z "$C" ] && continue
  while pgrep -x glcts >/dev/null 2>&1; do sleep 10; done
  Q=$D/$(echo "$C" | sed 's/.*functional\.//; s/\./_/g').qpa
  MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 nice -n 10 perl -e 'alarm shift; exec @ARGV' 600 \
    "$G/glcts" \
    --deqp-archive-dir=/tmp/mgl_cts_data \
    --deqp-case="$C" \
    --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
    --deqp-surface-width=256 --deqp-surface-height=256 --deqp-base-seed=-1 \
    --deqp-watchdog=disable --deqp-log-images=disable --deqp-log-shader-sources=disable \
    --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
    --deqp-log-filename="$Q" > $D/out 2> $D/err
  r=$(grep -o 'StatusCode="[A-Za-z]*"' "$Q" 2>/dev/null | tail -1 | sed 's/StatusCode="//; s/"//')
  printf '%-70s %s\n' "$(echo "$C" | sed 's/.*functional\.//')" "${r:-none}"
done < "$1"
