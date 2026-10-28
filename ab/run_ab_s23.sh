#!/bin/bash
# s23: interleaved A/B glmark2 scores — A = baseline-6f9b359 (prebuilt nsgl),
#      B = worktree build.  glmark2 always under the lockf lock.
# usage: run_ab_s23.sh <rounds> [scenes...]
WT="$(cd "$(dirname "$0")/.." && pwd)"
ROUNDS=${1:-3}; shift
SCENES=${@:-ideas jellyfish}
DATA=/opt/homebrew/share/glmark2
A=$HOME/MGL-cts/baseline-6f9b359/libmgl_nsgl.dylib
B=$WT/build/nsgl/libmgl_nsgl.dylib
run() {
    /usr/bin/lockf -k /tmp/mgl_glmark2.lock perl -e 'alarm 120; exec @ARGV' \
        env DYLD_INSERT_LIBRARIES=$1 \
        /opt/homebrew/bin/glmark2-macos --data-path $DATA -b $2 2>&1 \
        | grep -o "glmark2 Score: [0-9]*" | grep -o "[0-9]*$"
}
for r in $(seq 1 $ROUNDS); do
    for s in $SCENES; do
        a=$(run $A $s); b=$(run $B $s)
        echo "round$r $s A(base)=$a B(new)=$b load=$(sysctl -n vm.loadavg | cut -c9-13)"
    done
done
