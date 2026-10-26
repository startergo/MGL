#!/bin/bash
# usage: bash ab/xctx_glmark2.sh <out_prefix> [scenes...]
# runs glmark2 through MGL with the given scenes and screenshots the window
# during the LAST scene.
set -u
WT=$(cd "$(dirname "$0")/.." && pwd)
LIB=$WT/build/nsgl/libmgl_nsgl.dylib
OUT=$1; shift
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
[ -d "$DATA" ] || DATA=/opt/homebrew/share/glmark2
log=/tmp/xctx/$OUT.log; mkdir -p /tmp/xctx; : > "$log"

# last scene gets a long duration so we can screenshot inside it
args=(); n=$#
for s in "$@"; do
    n=$((n-1))
    if [ $n -eq 0 ]; then args+=(-b "${s}:duration=12"); else args+=(-b "${s}:duration=4"); fi
done

env DYLD_INSERT_LIBRARIES=$LIB /opt/homebrew/bin/glmark2-macos --data-path $DATA -s 640x480 \
    "${args[@]}" > "$log" 2>&1 &
pid=$!
# wait for the last scene: previous scenes take ~4s each + startup
sleep $((4 * ($# - 1) + 6))
screencapture -x -m /tmp/xctx/$OUT.png 2>>"$log"
sleep 2; screencapture -x -m /tmp/xOUT2_$OUT.png 2>>"$log"
kill $pid 2>/dev/null; wait $pid 2>/dev/null
echo "log lines: $(wc -l < "$log")"
grep -E "exceeds table|MGL ERROR|Assertion" "$log" | sort | uniq -c | head
