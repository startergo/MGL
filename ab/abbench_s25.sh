#!/bin/bash
# s25 headless per-draw gate vs the 1e840b4 baseline: ab_bench mode 0
# (drawArrays x2000) + mode 1 (blend-flip x600), min of 7, interleaved.
WT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=$HOME/MGL-cts/baseline-1e840b4
INCS="-I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include"
cc -std=gnu11 -O2 $INCS $WT/tests/headless/ab_bench.c -o /tmp/abb_base \
   -L$BASE -lmgl -Wl,-rpath,$BASE || exit 1
cc -std=gnu11 -O2 $INCS $WT/tests/headless/ab_bench.c -o /tmp/abb_new \
   -L$WT/build -lmgl -Wl,-rpath,$WT/build || exit 1
for mode in 0 1; do
    iters=$([ $mode = 0 ] && echo 2000 || echo 600)
    : > /tmp/abb_$mode.a; : > /tmp/abb_$mode.b
    for r in $(seq 1 7); do
        /tmp/abb_base $mode $iters >> /tmp/abb_$mode.a
        /tmp/abb_new  $mode $iters >> /tmp/abb_$mode.b
    done
    a=$(sort -n /tmp/abb_$mode.a | head -1)
    b=$(sort -n /tmp/abb_$mode.b | head -1)
    echo "mode$mode base=$a new=$b load=$(sysctl -n vm.loadavg | cut -c9-13)"
done
