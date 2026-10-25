#!/bin/bash
# A/B (session 12): current tree (A) vs base f5607b5 (B, ab/base12), alternating order
WT=$(cd "$(dirname "$0")/.." && pwd)
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
LIB_A=$WT/build/nsgl/libmgl_nsgl.dylib
LIB_B=$WT/ab/base12/build/nsgl/libmgl_nsgl.dylib
run() { perl -e 'alarm 120; exec @ARGV' env DYLD_INSERT_LIBRARIES=$1 /opt/homebrew/bin/glmark2-macos --data-path $DATA -b $2 2>&1 | grep "glmark2 Score" | tr -dc '0-9'; }
for r in 1 2 3; do
  for scene in ideas; do
    a=$(run $LIB_A $scene); b=$(run $LIB_B $scene)
    echo "round$r $scene A(cur)=$a B(base)=$b load=$(sysctl -n vm.loadavg | cut -c9-13)"
    a2=$(run $LIB_A $scene); b2=$(run $LIB_B $scene)
    echo "round$r $scene A(cur)=$a2 B(base)=$b2 load=$(sysctl -n vm.loadavg | cut -c9-13)"
  done
done
