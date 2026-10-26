#!/bin/bash
# xctx_diff.sh — capture native vs MGL at the same scene time, diff spatially
set -u
WT=$(cd "$(dirname "$0")/.." && pwd)
DATA=/opt/homebrew/share/glmark2
SCENE=${1:-jellyfish}
mkdir -p /tmp/xctx
run_native() {
  /opt/homebrew/bin/glmark2-macos --data-path $DATA -s 640x480 -b "$SCENE:duration=5" > /tmp/xctx/n.log 2>&1 &
  NPID=$!
  sleep 2.6; screencapture -x /tmp/xctx/nat_$SCENE.png 2>/dev/null
  kill $NPID 2>/dev/null; wait $NPID 2>/dev/null
}
run_mgl() {
  env DYLD_INSERT_LIBRARIES=$WT/build/nsgl/libmgl_nsgl.dylib /opt/homebrew/bin/glmark2-macos --data-path $DATA -s 640x480 -b "$SCENE:duration=5" > /tmp/xctx/m.log 2>&1 &
  MPID=$!
  sleep 2.6; screencapture -x /tmp/xctx/mgl_$SCENE.png 2>/dev/null
  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null
}
run_native; sleep 1; run_mgl
echo done
