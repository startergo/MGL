#!/bin/bash
# cts_piq8.sh — the 8 ES31 program_interface_query cases held from READY dbdf449.
# usage: bash ab/cts_piq8.sh <libmgl.dylib>
set -u
LIB=${1:?libmgl.dylib}
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
OUT=/tmp/piq8; mkdir -p $OUT; rm -f $OUT/*.qpa
P=0; F=0
while read -r C; do
  [ -z "$C" ] && continue
  MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 perl -e 'alarm shift; exec @ARGV' 300 \
    "$G/glcts" --deqp-archive-dir=/tmp/mgl_cts_data --deqp-case="$C" \
    --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
    --deqp-surface-width=256 --deqp-surface-height=256 \
    --deqp-log-images=disable --deqp-log-shader-sources=disable \
    --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
    --deqp-log-filename="$OUT/one.qpa" >/dev/null 2>$OUT/one.err
  rc=$?
  r=$(grep -o 'StatusCode="[A-Za-z]*"' $OUT/one.qpa | tail -1 | sed 's/StatusCode=//;s/"//g')
  case "$r" in Pass) P=$((P+1));; *) F=$((F+1)); cp $OUT/one.qpa "$OUT/$(echo "$C" | sed 's/.*\.//').qpa";; esac
  printf '%-8s rc=%-3d %s\n' "${r:-none}" "$rc" "${C##*program_interface_query.}"
done < /tmp/g88_piq8.txt
echo "== piq8: Pass=$P Fail=$F"
