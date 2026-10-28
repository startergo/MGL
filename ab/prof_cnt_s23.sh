#!/bin/bash
# s23: fast-path accept/decline counters (pipeHit / pipeMiss-decline) per frame
# A = baseline-6f9b359 (prebuilt), B = ab/gl46port @ <commit given on cmdline>
# usage: prof_cnt_s23.sh <commitlabel> [scenes...]
WT="$(cd "$(dirname "$0")/.." && pwd)"
LBL=${1:?label}
shift
SCENES=${@:-ideas jellyfish}
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
A=$HOME/MGL-cts/baseline-6f9b359/libmgl_nsgl.dylib
B=$WT/ab/gl46port/build/nsgl/libmgl_nsgl.dylib
run() { # lib scene -> last PROFILE line + score
    /usr/bin/lockf -k /tmp/mgl_glmark2.lock perl -e 'alarm 150; exec @ARGV' \
        env MGL_PROFILE=1 DYLD_INSERT_LIBRARIES=$1 \
        /opt/homebrew/bin/glmark2-macos --data-path $DATA -b $2 2>&1 \
        | grep -E "MGL PROFILE\(ms|glmark2 Score" | tail -2
}
for s in $SCENES; do
    for side in A B; do
        lib=$(eval echo \$$side)
        out=$(run $lib $s)
        hit=$(echo "$out" | grep -o "pipeHit=[0-9.|]*" | tail -1)
        dec=$(echo "$out" | grep -o "pipeMiss-decline=[0-9.|]*" | tail -1)
        score=$(echo "$out" | grep -o "Score: [0-9]*" | tail -1)
        echo "$LBL $s $side: $hit $dec $score load=$(sysctl -n vm.loadavg | cut -c9-13)"
    done
done
