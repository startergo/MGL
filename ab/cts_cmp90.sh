#!/bin/bash
# cts_cmp90.sh <name> <caselist> — run a caselist under MY lib and the 1f4c16b
# baseline lib (KHR-Single config), then print the pass/fail DIFF.
#   usage: bash ab/cts_cmp90.sh el90 /tmp/el90_todo.lst
set -u
NAME=$1; LST=$2
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
BASE=${MGL_BASE:-$HOME/MGL-cts/baseline-1f4c16b/libmgl.dylib}
D=/tmp/cmp90/$NAME
mkdir -p "$D"
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
for v in mine base; do
  LIB=$(pwd)/build/libmgl.dylib; [ $v = base ] && LIB=$BASE
  MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 nice -n 10 perl -e 'alarm shift; exec @ARGV' 7200 \
    "$G/glcts" \
    --deqp-archive-dir=/tmp/mgl_cts_data \
    --deqp-caselist-file="$LST" \
    --deqp-gl-context-type=mgl --deqp-surface-type=fbo --deqp-visibility=hidden \
    --deqp-surface-width=64 --deqp-surface-height=64 --deqp-base-seed=1 \
    --deqp-watchdog=disable --deqp-log-images=disable --deqp-log-shader-sources=disable \
    --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
    --deqp-log-filename="$D/$v.qpa" > "$D/$v.out" 2> "$D/$v.err"
  echo "rc($v)=$?"
  awk '/^#beginTestCaseResult /{c=$2} /^ <Result StatusCode=/{s=$2; sub(/.*StatusCode="/,"",s); sub(/".*/,"",s); print s, c}' "$D/$v.qpa" > "$D/$v.status"
  echo "tally($v): $(sort $D/$v.status | awk '{print $1}' | uniq -c | tr '\n' ' ')"
done
echo "=== regressions (pass->fail / new crash):"
join -j 2 <(sort -k2 $D/base.status) <(sort -k2 $D/mine.status) | awk '$2=="Pass" && $3!="Pass" {print "  REGRESS", $1, $3}' | head -30
echo "=== fixes (fail->pass):"
join -j 2 <(sort -k2 $D/base.status) <(sort -k2 $D/mine.status) | awk '$2!="Pass" && $3=="Pass" {print "  FIX    ", $1}' | wc -l
echo "=== still failing (mine): $(awk '$1!="Pass"{print}' $D/mine.status | wc -l | tr -d ' ')"
