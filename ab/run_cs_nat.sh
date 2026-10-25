#!/bin/bash
# Interleaved native vs MGL (MGL_ENCODER_THREAD=1) on one scene, alternating
# order. Usage: run_cs_nat.sh <worktree> <scene> [rounds]
WT=$1; SCENE=$2; R=${3:-3}
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
LIB=$WT/build/nsgl/libmgl_nsgl.dylib
n() { /opt/homebrew/bin/glmark2-macos --data-path $DATA -b "$SCENE" > /tmp/csnat_nat.log 2>&1; }
m() { env MGL_ENCODER_THREAD=1 DYLD_INSERT_LIBRARIES=$LIB /opt/homebrew/bin/glmark2-macos --data-path $DATA -b "$SCENE" > /tmp/csnat_mgl.log 2>&1; }
for r in $(seq 1 $R); do
  if [ $((r % 2)) -eq 1 ]; then m; n; else n; m; fi
  fn=$(grep "glmark2 Score" /tmp/csnat_nat.log | tr -dc '0-9'); fm=$(grep "glmark2 Score" /tmp/csnat_mgl.log | tr -dc '0-9')
  echo "round$r $SCENE native=$fn mgl-cs=$fm"
done
