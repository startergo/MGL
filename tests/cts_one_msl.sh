#!/bin/bash
# run one CTS case with MGL_DUMP_MSL=1 and print the dumped MSL around sample_compare
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASE="$1"
rm -f /tmp/mgl_cts_modules
ln -sfn ~/MGL-cts/build/external/openglcts/modules /tmp/mgl_cts_modules
cd /tmp/mgl_cts_modules || exit 2
MGL_LIBRARY="$ROOT/build/libmgl.dylib" MGL_DUMP_MSL=1 perl -e 'alarm shift; exec @ARGV' 300 ./glcts \
  --deqp-case="$CASE" --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
  --deqp-surface-width=256 --deqp-surface-height=256 \
  --deqp-log-filename=/tmp/cts_one.qpa --deqp-log-images=disable > /tmp/cts_one.out 2>/tmp/cts_one.err
grep -o 'StatusCode="[A-Za-z]*"' /tmp/cts_one.qpa | sort | uniq -c
grep -n "sample_compare" /tmp/cts_one.err | head
