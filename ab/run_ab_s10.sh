#!/bin/bash
# Simultaneous A/B (identical machine state): current worktree build (A) vs the
# bf0074e-era build (B).  Alternate launch order across rounds to cancel
# frontmost-window effects.  Usage: run_ab_s10.sh <worktree> <scene> [rounds]
WT=$1; SCENE=$2; R=${3:-3}
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
A=$WT/build/nsgl/libmgl_nsgl.dylib
B=$WT/ab/bf0074e/build/nsgl/libmgl_nsgl.dylib
one() { DYLD_INSERT_LIBRARIES=$2 /opt/homebrew/bin/glmark2-macos --data-path $DATA -b $SCENE > /tmp/ab10_$1.log 2>&1 & }
for r in $(seq 1 $R); do
  if [ $((r % 2)) -eq 1 ]; then first=A second=B; else first=B second=A; fi
  one $first ${!first}; sleep 0.3; one $second ${!second}
  wait
  fa=$(grep "glmark2 Score" /tmp/ab10_A.log | tr -dc '0-9'); fb=$(grep "glmark2 Score" /tmp/ab10_B.log | tr -dc '0-9')
  echo "round$r(order:$first-first) $SCENE new=$fa bf0074e=$fb ratio=$(python3 -c "print(f'{float('$fa')/max(float('$fb'),1):.3f}')")"
done
