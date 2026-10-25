#!/bin/bash
# interleaved MGL_PROFILE A/B: per-call medians of the hot buckets
WT=$(cd "$(dirname "$0")/.." && pwd)
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
A=$WT/build/nsgl/libmgl_nsgl.dylib
B=$WT/ab/base12/build/nsgl/libmgl_nsgl.dylib
run() { perl -e 'alarm 45; exec @ARGV' env MGL_PROFILE=1 DYLD_INSERT_LIBRARIES=$1 /opt/homebrew/bin/glmark2-macos --data-path $DATA -b ideas 2>&1 | grep -E "MGL PROFILE\(ms|glmark2 Score" | tail -2; }
for r in 1 2 3; do
  echo "== round $r"; echo "-- A:"; run $A; echo "-- B:"; run $B
done
