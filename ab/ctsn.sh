#!/bin/bash
# compat agent helper: run one CTS case (or prefix) against ~/MGL-cts build + this worktree's libmgl
# usage: bash ab/ctsn.sh '<case-prefix>' [timeout-s] [outtag] [extra glcts args...]
set -u
ROOT="$(pwd)"
CASE="$1"; T="${2:-120}"; TAG="${3:-mgl_cts}"
shift 3 2>/dev/null || shift $#
D=/tmp/mgl_cts_modules
[ -d "$D" ] || { ln -sfn "$HOME/MGL-cts/build/external/openglcts/modules" "$D" || exit 2; }
mkdir -p /tmp/cts_r96
export MGL_LIBRARY="$ROOT/build/libmgl.dylib"
perl -e 'alarm shift; my $d=shift; chdir $d or die; exec @ARGV' "$T" "$D" ./glcts \
  --deqp-case="$CASE" --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
  --deqp-surface-width=256 --deqp-surface-height=256 \
  --deqp-log-filename="/tmp/cts_r96/$TAG.qpa" --deqp-log-images=disable "$@" > "/tmp/cts_r96/$TAG.out" 2> "/tmp/cts_r96/$TAG.err"
rc=$?
grep -ao 'StatusCode="[A-Za-z]*"' "/tmp/cts_r96/$TAG.qpa" 2>/dev/null | sort | uniq -c
tail -2 "/tmp/cts_r96/$TAG.err" 2>/dev/null
exit $rc
