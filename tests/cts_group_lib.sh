#!/bin/bash
# like cts_group.sh but runs an arbitrary libmgl.dylib (baseline A/B for regressions)
# usage: bash tests/cts_group_lib.sh <libmgl.dylib> 'CASE.PREFIX' [outfile] [timeout]
set -u
LIB="$1"; GRP="$2"; OUT="${3:-/tmp/cts_group}"; T="${4:-3600}"
rm -f /tmp/mgl_cts_modules
ln -sfn ~/MGL-cts/build/external/openglcts/modules /tmp/mgl_cts_modules || exit 2
cd /tmp/mgl_cts_modules || exit 2
MGL_LIBRARY="$LIB" perl -e 'alarm shift; exec @ARGV' "$T" ./glcts --deqp-case="$GRP.*" \
  --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
  --deqp-surface-width=256 --deqp-surface-height=256 \
  --deqp-log-filename="$OUT.qpa" --deqp-log-images=disable > "$OUT.out" 2> "$OUT.err"
grep -E 'Passed:|Failed:|Not supported:|Crashed:' "$OUT.out" | head -4
awk '/Test case/{name=$0} /\(Fail\)|\(InternalError\)|\(Crash\)/{print name}' "$OUT.out"
