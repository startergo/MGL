#!/bin/bash
# cts_suite_ab.sh <lib> <case-pattern> <log-stem> — run one glcts suite against
# one library (sequential; glcts budget 1), print absolute Pass/Fail/other counts.
set -u
LIB=$1; PAT=$2; STEM=$3
G=$HOME/MGL-cts-new/build/external/openglcts/modules
D=/tmp/ctsab
mkdir -p "$D"
ln -sfn $HOME/MGL-cts-new/src/data /tmp/mgl_cts_data 2>/dev/null
Q="$D/$STEM.qpa"
rm -f "$Q"
MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 nice -n 10 perl -e 'alarm shift; exec @ARGV' 14400 \
    "$G/glcts" \
    --deqp-archive-dir=/tmp/mgl_cts_data \
    --deqp-case="$PAT" \
    --deqp-gl-context-type=mgl --deqp-surface-type=fbo --deqp-visibility=hidden \
    --deqp-surface-width=256 --deqp-surface-height=256 --deqp-base-seed=0 \
    --deqp-watchdog=disable --deqp-log-images=disable --deqp-log-shader-sources=disable \
    --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
    --deqp-log-filename="$Q" > "$D/$STEM.out" 2> "$D/$STEM.err"
echo "rc=$? cases: $(grep -c '#beginTestCaseResult' "$Q")"
grep -o 'StatusCode="[A-Za-z]*"' "$Q" | sort | uniq -c
# also list the failing case names for the A/B diff
awk '/^#beginTestCaseResult /{c=$2} /^ <Result StatusCode="(Fail|InternalError|Crash|Timeout|NotSupported)"/{print c}' "$Q" > "$D/$STEM.fails"
wc -l < "$D/$STEM.fails"
