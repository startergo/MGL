#!/bin/bash
# helper: run one CTS case (or a case-pattern) against the local build
# usage: bash tests/cts_one.sh '<case pattern>' [outbase]
# (outbase: path without extension; writes .qpa/.out/.err next to it)
set -u
CASE="$1"
OUT="${2:-/tmp/cts_one}"
rm -f /tmp/mgl_cts_modules
ln -sfn ~/MGL-cts/build/external/openglcts/modules /tmp/mgl_cts_modules || exit 2
LIB=$(cd "$(dirname "$0")/../build" && pwd)/libmgl.dylib
cd /tmp/mgl_cts_modules || exit 2
MGL_LIBRARY="$LIB" ./glcts --deqp-case="$CASE" \
  --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
  --deqp-surface-width=256 --deqp-surface-height=256 \
  --deqp-log-filename="$OUT.qpa" --deqp-log-images=disable \
  > "$OUT.out" 2> "$OUT.err"
rc=$?
grep -aE "^Test case '|StatusCode|ERROR" "$OUT.out" | tail -40
[ -s "$OUT.err" ] && tail -5 "$OUT.err"
exit $rc
