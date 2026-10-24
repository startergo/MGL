#!/bin/bash
# Simultaneous A/B (identical machine state): bd31b86 (A) vs bf0074e (B).
# Alternate launch order across rounds to cancel frontmost-window effects.
WT=$1; SCENE=$2
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
A=$WT/build/nsgl/libmgl_nsgl.dylib
B=$WT/ab/bf0074e/build/nsgl/libmgl_nsgl.dylib
one() { DYLD_INSERT_LIBRARIES=$2 /opt/homebrew/bin/glmark2-macos --data-path $DATA -b $SCENE > /tmp/ab_$1.log 2>&1 & }
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then first=A second=B; else first=B second=A; fi
  one $first ${!first}; sleep 0.3; one $second ${!second}
  wait
  fa=$(grep "glmark2 Score" /tmp/ab_A.log | tr -dc '0-9'); fb=$(grep "glmark2 Score" /tmp/ab_B.log | tr -dc '0-9')
  echo "round$r(order:$first-first) $SCENE bd31b86=$fa bf0074e=$fb ratio=$(python3 -c "print(f'{float('$fa')/float('$fb'):.3f}')")"
done
