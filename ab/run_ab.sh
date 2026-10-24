#!/bin/bash
# Interleaved back-to-back single-scene A/B: bd31b86 (A, current tree) vs bf0074e (B, ab tree)
WT=$1
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
LIB_A=$WT/build/nsgl/libmgl_nsgl.dylib          # aad7d91 = bd31b86 content
LIB_B=$WT/ab/bf0074e/build/nsgl/libmgl_nsgl.dylib
run() { DYLD_INSERT_LIBRARIES=$1 /opt/homebrew/bin/glmark2-macos --data-path $DATA -b $2 2>&1 | grep "glmark2 Score" | tr -dc '0-9'; }
for r in 1 2 3; do
  for scene in gl41-mrt gl41-texarray; do
    a=$(run $LIB_A $scene)
    b=$(run $LIB_B $scene)
    echo "round$r $scene bd31b86=$a bf0074e=$b load=$(sysctl -n vm.loadavg | cut -c9-13)"
  done
done
