#!/bin/bash
# xctx_color.sh — average colour of the glmark2 window region during a scene.
# usage: bash ab/xctx_color.sh <tag> <scene> [native|]
set -u
WT=$(cd "$(dirname "$0")/.." && pwd)
TAG=$1; SCENE=$2; NATIVE=${3:-mgl}
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
[ -d "$DATA" ] || DATA=/opt/homebrew/share/glmark2
mkdir -p /tmp/xctx
if [ "$NATIVE" = native ]; then INS=""; else INS="DYLD_INSERT_LIBRARIES=$WT/build/nsgl/libmgl_nsgl.dylib"; fi
env $INS /opt/homebrew/bin/glmark2-macos --data-path $DATA -s 640x480 -b "$SCENE:duration=6" > /tmp/xctx/$TAG.log 2>&1 &
pid=$!
sleep 4
screencapture -x /tmp/xctx/$TAG.png 2>/dev/null
sleep 1
screencapture -x /tmp/xTAG2_$TAG.png 2>/dev/null
kill $pid 2>/dev/null; wait $pid 2>/dev/null
echo captured $TAG
