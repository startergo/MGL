#!/bin/sh
# abg83.sh — locked glmark2 A/B, 8 rounds, ideas+jellyfish+gl41-geometry medians.
# P = baseline-288e724, C = this branch's nsgl build (copied to glmark2-bench/geomC).
# Guarded by the shared lock so it never overlaps another agent's glmark2 run.
G=/opt/homebrew/bin/glmark2-macos
D=/opt/homebrew/share/glmark2
B=$HOME/MGL-cts/glmark2-bench
C=$B/geomC/libmgl_nsgl.dylib
P=$HOME/MGL-cts/baseline-d11d184/libmgl_nsgl.dylib
S=${SCENES:-"-b ideas -b jellyfish -b gl41-geometry"}
mkdir -p $B/geomC
R=${1:-8}
rm -f $B/abg83_done.flag
r=0
while [ $r -lt $R ]; do
  r=$((r+1))
  DYLD_INSERT_LIBRARIES=$P $G --data-path $D $S > $B/abg83P$r.log 2>&1
  DYLD_INSERT_LIBRARIES=$C $G --data-path $D $S > $B/abg83C$r.log 2>&1
done
echo AB_DONE > $B/abg83_done.flag
