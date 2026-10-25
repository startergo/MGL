#!/bin/bash
# Simultaneous A/B (identical machine state): this worktree build (A) vs the
# session-8 baseline 1335928 (B).  Alternate launch order across rounds to
# cancel frontmost-window effects.  Usage: run_ab2.sh <worktree> <scene> [rounds]
WT=$1; SCENE=$2; R=${3:-3}
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
A=$WT/build/nsgl/libmgl_nsgl.dylib
B=$WT/ab/1335928/build/nsgl/libmgl_nsgl.dylib
one() { DYLD_INSERT_LIBRARIES=$2 /opt/homebrew/bin/glmark2-macos --data-path $DATA -b $SCENE > /tmp/ab2_$1.log 2>&1 & }
for r in $(seq 1 $R); do
  if [ $((r % 2)) -eq 1 ]; then first=A second=B; else first=B second=A; fi
  one $first ${!first}; sleep 0.3; one $second ${!second}
  wait
  fa=$(grep "glmark2 Score" /tmp/ab2_A.log | tr -dc '0-9'); fb=$(grep "glmark2 Score" /tmp/ab2_B.log | tr -dc '0-9')
  echo "round$r(order:$first-first) $SCENE new=$fa old1335928=$fb ratio=$(python3 -c "print(f'{float('$fa')/max(float('$fb'),1):.3f}')")"
done
