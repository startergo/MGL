#!/bin/bash
# compat r92: run one CTS group (extended-regex prefix) from ANY mustpass list
# against ANY libmgl.dylib, with crash/timeout resume (same scheme as
# cts_group_new.sh, which is hardwired to the khr-single list + this build).
#   usage: bash ab/cts_group_any.sh <list-file-abs> <ERE-prefix> <tag> <libmgl-abs>
set -u
LIST="$1"; PREFIX="$2"; TAG="$3"; LIB="$4"
M=$HOME/MGL-cts/build/external/openglcts/modules
D=/tmp/mgl_cts_modules
[ -e "$D" ] || ln -sfn "$M" "$D"
mkdir -p /tmp/cts_$TAG
: > /tmp/cts_$TAG/pre.txt
grep -E "^$PREFIX" "$LIST" > /tmp/cts_$TAG/all.txt || true
echo "cases: $(grep -c . /tmp/cts_$TAG/all.txt)"
cd "$D" || exit 2
export MGL_LIBRARY="$LIB"
part=0
while true; do
  todo=/tmp/cts_$TAG/todo-$part.txt
  if [ ! -f $todo ]; then
    # rebuild the done set from every part's qpa (idempotent; survives
    # restarts) — never start from an empty done.txt when parts exist
    : > /tmp/cts_$TAG/done.txt
    for q in /tmp/cts_$TAG/part-*.qpa; do
      [ -f "$q" ] || continue
      awk '/^#beginTestCaseResult /{if(c)print c; c=$2}
           /^#endTestCaseResult|^#terminateTestCaseResult/{if(c){print c; c=""}}
           END{if(c)print c}' "$q" >> /tmp/cts_$TAG/done.txt
    done
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
  awk '/^#beginTestCaseResult /{if(c)print c; c=$2}
       /^#endTestCaseResult|^#terminateTestCaseResult/{if(c){print c; c=""}}
       END{if(c)print c}' /tmp/cts_$TAG/part-$part.qpa >> /tmp/cts_$TAG/done.txt
  sort -u /tmp/cts_$TAG/done.txt > /tmp/cts_$TAG/done.tmp && mv /tmp/cts_$TAG/done.tmp /tmp/cts_$TAG/done.txt
  echo "part $part rc=$rc done=$(grep -c . /tmp/cts_$TAG/done.txt)/$(grep -c . /tmp/cts_$TAG/all.txt)"
  [ $rc -ne 0 ] && [ $rc -ne 142 ] && sleep 2
  part=$((part+1))
done
