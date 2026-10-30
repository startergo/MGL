#!/bin/bash
# s28 locked interleaved glmark2 A/B vs baseline-83e1777:
#   ideas (3) + jellyfish (4) + gl41-geometry (3), order alternates per round,
#   whole script under /tmp/mgl_glmark2.lock (one glmark2 on the machine at a time).
if [ -z "${MGL_AB_LOCK:-}" ]; then
    MGL_AB_LOCK=1 exec /usr/bin/lockf -k /tmp/mgl_glmark2.lock /bin/bash "$0" "$@"
fi
WT="$(cd "$(dirname "$0")/.." && pwd)"
A=$WT/build/nsgl/libmgl_nsgl.dylib                    # candidate (merged tip)
B=$HOME/MGL-cts/baseline-83e1777/libmgl_nsgl.dylib    # baseline (gl46-port 83e1777)
DATA=/opt/homebrew/share/glmark2
run() { perl -e 'alarm 150; exec @ARGV' env DYLD_INSERT_LIBRARIES=$1 \
        /opt/homebrew/bin/glmark2-macos --data-path $DATA -b $2 2>&1 \
        | grep -o "glmark2 Score: [0-9]*" | tr -dc '0-9'; }
for r in $(seq 1 ${1:-3}); do
    if [ $((r % 2)) -eq 1 ]; then o1=A o2=B; else o1=B o2=A; fi
    a=$(run ${!o1} ideas); b=$(run ${!o2} ideas)
    echo "r$r ideas $o1=$a $o2=$b load=$(sysctl -n vm.loadavg | cut -c9-13)"
done
for r in $(seq 1 ${2:-4}); do
    if [ $((r % 2)) -eq 1 ]; then o1=B o2=A; else o1=A o2=B; fi
    a=$(run ${!o1} jellyfish); b=$(run ${!o2} jellyfish)
    echo "r$r jellyfish $o1=$a $o2=$b load=$(sysctl -n vm.loadavg | cut -c9-13)"
done
for r in $(seq 1 ${3:-3}); do
    if [ $((r % 2)) -eq 1 ]; then o1=A o2=B; else o1=B o2=A; fi
    a=$(run ${!o1} gl41-geometry); b=$(run ${!o2} gl41-geometry)
    echo "r$r gl41-geometry $o1=$a $o2=$b load=$(sysctl -n vm.loadavg | cut -c9-13)"
done
