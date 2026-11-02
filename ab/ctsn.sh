#!/bin/bash
# compat agent helper: run one CTS case (or prefix) against ~/MGL-cts-new build + this worktree's libmgl
# usage: bash ab/ctsn.sh '<case-prefix>' [timeout-s] [outtag] [extra glcts args...]
set -u
ROOT="$(pwd)"
CASE="$1"; T="${2:-120}"; TAG="${3:-mgl_cts}"
shift 3 2>/dev/null || shift $# 
D=/tmp/mgl_cts_modules
[ -d "$D" ] || { ln -sfn "$HOME/MGL-cts-new/build/external/openglcts/modules" "$D" || exit 2; }
export MGL_LIBRARY="$ROOT/build/libmgl.dylib"
perl -e 'alarm shift; my $d=shift; chdir $d or die; exec @ARGV' "$T" "$D" ./glcts \
  --deqp-case="$CASE" --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
  --deqp-surface-width=256 --deqp-surface-height=256 \
  --deqp-log-filename="/tmp/$TAG.qpa" --deqp-log-images=disable "$@" > "/tmp/$TAG.out" 2> "/tmp/$TAG.err"
rc=$?
grep -ao 'StatusCode="[A-Za-z]*"' "/tmp/$TAG.qpa" 2>/dev/null | sort | uniq -c
tail -2 "/tmp/$TAG.err" 2>/dev/null
exit $rc
