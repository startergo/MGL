#!/bin/bash
# xctx_capture.sh — run glmark2 (MGL) with scenes, screenshot the glmark2 window
# during the LAST scene, once per second.  usage: bash ab/xctx_capture.sh out scenes...
set -u
WT=$(cd "$(dirname "$0")/.." && pwd)
LIB=$WT/build/nsgl/libmgl_nsgl.dylib
OUT=$1; shift
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
[ -d "$DATA" ] || DATA=/opt/homebrew/share/glmark2
mkdir -p /tmp/xctx; rm -f /tmp/xctx/$OUT.*.png
log=/tmp/xctx/$OUT.log; : > "$log"

args=(); n=$#
for s in "$@"; do
    n=$((n-1))
    if [ $n -eq 0 ]; then args+=(-b "${s}:duration=14"); else args+=(-b "${s}:duration=4"); fi
done

env DYLD_INSERT_LIBRARIES=$LIB /opt/homebrew/bin/glmark2-macos --data-path $DATA -s 640x480 \
    "${args[@]}" > "$log" 2>&1 &
pid=$!
sleep $((4 * ($# - 1) + 5))
for i in 1 2 3; do
    win=$(swift "$WT/ab/findwin.swift" | head -1)
    id=$(echo "$win" | awk '{print $2}')
    if [ -n "${id:-}" ] && [ "$id" != "0" ]; then
        echo "$win" > /tmp/xctx/$OUT.$i.win
        screencapture -x /tmp/xctx/$OUT.$i.full.png 2>>"$log"
        echo "shot $i: $win"
    else
        echo "shot $i: no window found"
    fi
    sleep 3
done
kill $pid 2>/dev/null; wait $pid 2>/dev/null
grep -cE "exceeds table" "$log" | sed 's/^/table-growth msgs: /'
ls /tmp/xctx/$OUT.*.png 2>/dev/null | wc -l | sed 's/^/pngs: /'
