#!/bin/bash
# Interleaved A/B of the SAME build: MGL_ENCODER_THREAD=1 (A) vs off (B).
# Usage: run_cs_ab.sh <worktree> <scene> [rounds]
WT=$1; SCENE=$2; R=${3:-3}
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
LIB=$WT/build/nsgl/libmgl_nsgl.dylib
one() { env MGL_ENCODER_THREAD=$2 DYLD_INSERT_LIBRARIES=$LIB /opt/homebrew/bin/glmark2-macos --data-path $DATA -b "$SCENE" > /tmp/csab_$1.log 2>&1 & }
for r in $(seq 1 $R); do
  if [ $((r % 2)) -eq 1 ]; then first=1 second=0; else first=0 second=1; fi
  one cs $first; sleep 0.5; one off $second
  wait
  fa=$(grep "glmark2 Score" /tmp/csab_cs.log | tr -dc '0-9'); fb=$(grep "glmark2 Score" /tmp/csab_off.log | tr -dc '0-9')
  echo "round$r(order:cs=$first-first) $SCENE cs=$fa off=$fb delta=$(python3 -c "print(f'{(float('$fa')/max(float('$fb'),1)-1)*100:+.1f}%')")"
done
