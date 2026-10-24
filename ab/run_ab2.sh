#!/bin/bash
# Simultaneous A/B: NEW (current tree, fixes) vs OLD (ab/aad7d91 = session-8 base).
# usage: run_ab2.sh <wt> <scene> [rounds]
WT=$1; SCENE=$2; R=${3:-3}
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
NEW=$WT/build/nsgl/libmgl_nsgl.dylib
OLD=$WT/ab/aad7d91/build/nsgl/libmgl_nsgl.dylib
one() { DYLD_INSERT_LIBRARIES=$2 /opt/homebrew/bin/glmark2-macos --data-path $DATA -b "$SCENE" > /tmp/ab2_$1.log 2>&1 & }
for r in $(seq 1 $R); do
  if [ $((r % 2)) -eq 1 ]; then first=NEW second=OLD; else first=OLD second=NEW; fi
  one $first ${!first}; sleep 0.3; one $second ${!second}
  wait
  fn=$(grep "glmark2 Score" /tmp/ab2_NEW.log | tr -dc '0-9'); fo=$(grep "glmark2 Score" /tmp/ab2_OLD.log | tr -dc '0-9')
  echo "round$r($first-first) $SCENE new=$fn old=$fo ratio=$(python3 -c "print(f'{float('$fn')/max(float('$fo'),1):.3f}')")"
done
