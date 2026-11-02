#!/bin/bash
# compat r86: run one KHR-Single-GL46 group (extended-regex prefix) against this
# worktree's libmgl with crash/timeout resume, mirroring official/cts_resume.py
# (per-part .qpa/.out, crashes.txt, the in-flight case counted as done).
#   usage: bash ab/cts_group_new.sh 'KHR-Single-GL46\.enhanced_layouts\.' <tag>
set -u
ROOT="$(pwd)"
PREFIX="$1"; TAG="$2"
M=$HOME/MGL-cts-new/build/external/openglcts/modules
LIST=$M/gl_cts/data/mustpass/gl/khronos_mustpass_single/main/gl46-khr-single.txt
D=/tmp/mgl_cts_modules
rm -f "$D"; ln -sfn "$M" "$D"
mkdir -p /tmp/cts_$TAG
: > /tmp/cts_$TAG/pre.txt
grep -E "^$PREFIX" "$LIST" > /tmp/cts_$TAG/all.txt || true
echo "cases: $(grep -c . /tmp/cts_$TAG/all.txt)"
cd "$D" || exit 2   # script is invoked, never sourced; worktree guard is on the caller
export MGL_LIBRARY="$ROOT/build/libmgl.dylib"
part=0
while true; do
  todo=/tmp/cts_$TAG/todo-$part.txt
  if [ ! -f $todo ]; then
    : > /tmp/cts_$TAG/done.txt
    awk 'FILENAME==ARGV[1]{skip[$1];next} !($1 in skip)' /tmp/cts_$TAG/done.txt /tmp/cts_$TAG/all.txt > $todo
  fi
  n=$(grep -c . $todo)
  [ "$n" = 0 ] && { echo "ALL DONE parts=$part"; break; }
  perl -e 'alarm shift; exec @ARGV' 900 ./glcts \
    --deqp-caselist-file=$todo \
    --deqp-log-filename=/tmp/cts_$TAG/part-$part.qpa \
    --deqp-screen-rotation=unspecified --deqp-surface-width=64 --deqp-surface-height=64 \
    --deqp-watchdog=disable --deqp-base-seed=1 --deqp-visibility=hidden \
    --deqp-surface-type=fbo --deqp-gl-context-type=mgl \
    > /tmp/cts_$TAG/part-$part.out 2>&1
  rc=$?
  # finished = complete cases + the in-flight one (crash/timeout victim)
  awk '/^#beginTestCaseResult /{if(c)print c; c=$2}
       /^#endTestCaseResult|^#terminateTestCaseResult/{if(c){print c; c=""}}
       END{if(c)print c}' /tmp/cts_$TAG/part-$part.qpa >> /tmp/cts_$TAG/done.txt
  fin=$(grep -c . /tmp/cts_$TAG/done.txt)
  tot=$(grep -c . /tmp/cts_$TAG/all.txt)
  echo "part $part rc=$rc done=$fin/$tot"
  part=$((part+1))
  [ $part -gt 60 ] && { echo TOO_MANY_PARTS; break; }
done
