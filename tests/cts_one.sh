#!/bin/bash
# helper: run one CTS case, optionally with images enabled (arg2=images)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASE="$1"
IMG="${2:-disable}"
rm -f /tmp/mgl_cts_modules
ln -sfn ~/MGL-cts/build/external/openglcts/modules /tmp/mgl_cts_modules
cd /tmp/mgl_cts_modules || exit 2
MGL_LIBRARY="$ROOT/build/libmgl.dylib" perl -e 'alarm shift; exec @ARGV' 300 ./glcts \
  --deqp-case="$CASE" --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
  --deqp-surface-width=256 --deqp-surface-height=256 \
  --deqp-log-filename=/tmp/cts_one.qpa --deqp-log-images="$IMG" > /tmp/cts_one.out 2>/tmp/cts_one.err
rc=$?
grep -o 'StatusCode="[A-Za-z]*"' /tmp/cts_one.qpa | sort | uniq -c
tail -2 /tmp/cts_one.err 2>/dev/null
exit $rc
