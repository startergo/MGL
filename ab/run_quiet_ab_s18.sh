#!/bin/bash
# run_quiet_ab_s18.sh — quiet-machine A/B (supervisor protocol: one benchmark
# at a time, Terminal launch, 2 rounds): C-new (single-pass present copy) vs
# gl46-port 5178835 (s16d kernel+blit present path, has the same flash fix).
WT="$(cd "$(dirname "$0")/.." && pwd)"
NEW=$WT/build/nsgl/libmgl_nsgl.dylib
OLD=$WT/ab/gl46port/build/nsgl/libmgl_nsgl.dylib
DATA=/opt/homebrew/share/glmark2
run() { DYLD_INSERT_LIBRARIES=$2 /opt/homebrew/bin/glmark2-macos --data-path $DATA -b "$1" 2>&1 | grep -o "glmark2 Score: [0-9]*" | tr -dc '0-9'; }
for r in 1 2; do
  for who in new old new old; do :; done
  if [ $((r % 2)) -eq 1 ]; then o1=new o2=old; else o1=old o2=new; fi
  for scene in jellyfish ideas; do
    a=$(run $scene $NEW); b=$(run $scene $OLD)
    echo "round$r $scene new=$a gl46port=$b load=$(sysctl -n vm.loadavg | cut -c9-13)"
  done
done
