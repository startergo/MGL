#!/bin/bash
# Aggregate whole-run per-frame CPU cost (all MGL PROFILE dumps) for one scene, both sides.
WT=$1; SCENE=$2
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
run() { # $1=lib -> prints "score agg_ms_per_frame"
  local out
  out=$(MGL_PROFILE=1 DYLD_INSERT_LIBRARIES=$1 /opt/homebrew/bin/glmark2-macos --data-path $DATA -b $SCENE 2>&1)
  local score=$(echo "$out" | grep "glmark2 Score" | tr -dc '0-9')
  # sum "sum N ms" totals and frame counts (frames = slot total / slot per-frame, use dirtyState)
  python3 - "$out" "$score" <<'EOF'
import sys, re
out, score = sys.argv[1], sys.argv[2]
tot_ms = 0.0; frames = 0.0
for ln in out.splitlines():
    m = re.search(r'\(sum ([0-9.]+) ms\)', ln)
    if not m: continue
    tot_ms += float(m.group(1))
    d = re.search(r'dirtyState-block=([0-9.]+)\|([0-9.]+)\|', ln)
    if d and float(d.group(2)) > 0: frames += float(d.group(1)) / float(d.group(2))
print(f"{score} {tot_ms/max(frames,1):.4f}")
EOF
}
A=$WT/build/nsgl/libmgl_nsgl.dylib
B=$WT/ab/bf0074e/build/nsgl/libmgl_nsgl.dylib
for r in 1 2; do
  ra=($(run $A)); rb=($(run $B))
  echo "round$r $SCENE  bd31b86: fps=${ra[0]} cpu_ms/frame=${ra[1]}   bf0074e: fps=${rb[0]} cpu_ms/frame=${rb[1]}"
done
