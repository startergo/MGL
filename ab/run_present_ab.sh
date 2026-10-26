#!/bin/bash
# run_present_ab.sh — CPU-side A/B of the present path (s18): new single
# render-pass opaque copy (build/nsgl) vs the s16d kernel+blit pair
# (ab/prev = a036e29).  FPS is meaningless under load; the metrics here are
# load-independent CPU numbers:
#   1. MGL_PROFILE=1 "swap" bucket ms/frame   (present-path CPU)
#   2. /usr/bin/time user+sys CPU for a fixed 10 s [jellyfish] scene
#   3. mtlcount encoder deltas per interval   (encoders per present)
WT="$(cd "$(dirname "$0")/.." && pwd)"
A=$WT/build/nsgl/libmgl_nsgl.dylib
B=$WT/ab/prev/build/nsgl/libmgl_nsgl.dylib
DATA=/opt/homebrew/share/glmark2
run() { DYLD_INSERT_LIBRARIES="$1" /opt/homebrew/bin/glmark2-macos --data-path $DATA -s 640x480 -b jellyfish:duration=10 2>&1; }
for r in 1 2; do
  for who in A B; do
    L=$([ $who = A ] && echo $A || echo $B)
    out=$(MGL_PROFILE=1 run "$L")
    swap=$(echo "$out" | grep -o "swap=[0-9.]*|[0-9.]*" | tail -5 | cut -d'|' -f2 | paste -sd+ - | bc -l)
    n=$(echo "$out" | grep -o "swap=[0-9.]*|[0-9.]*" | tail -5 | wc -l | tr -d ' ')
    fps=$(echo "$out" | grep -o "FPS: [0-9]*" | tail -1)
    /usr/bin/time -p env DYLD_INSERT_LIBRARIES="$L" /opt/homebrew/bin/glmark2-macos --data-path $DATA -s 640x480 -b jellyfish:duration=10 >/dev/null 2>/tmp/mgl_present_ab_time.$who
    cpu=$(grep -E "user|sys" /tmp/mgl_present_ab_time.$who | paste -sd' ' -)
    echo "round$r $who swap_cpu_ms/frame=$(echo "scale=4; $swap / $n" | bc -l) (n=$n) $fps cpu: $cpu load=$(sysctl -n vm.loadavg | cut -c9-13)"
  done
done
# encoder counts: 5 s intervals, jellyfish
for who in A B; do
  L=$([ $who = A ] && echo $A || echo $B)
  MTL_COUNT_INTERVAL=5 DYLD_INSERT_LIBRARIES="$L:$WT/build/mtlcount.dylib" \
      /opt/homebrew/bin/glmark2-macos --data-path $DATA -s 640x480 -b jellyfish:duration=10 2>&1 \
      | grep "MTLCOUNT interval" | tail -4 > /tmp/mgl_present_ab_cnt.$who
done
echo "--- encoder deltas per 5 s (A=new, B=old) ---"
paste /tmp/mgl_present_ab_cnt.A /tmp/mgl_present_ab_cnt.B 2>/dev/null | tail -4
