#!/bin/bash
# s28 3-way ab_bench: (A) supervisor baseline-83e1777 dylib, (B) pristine
# gl46-port 83e1777 get.c/textures.c built in THIS tree, (C) merged tip.
# min of 7, interleaved A,B,C per round.
WT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=$HOME/MGL-cts/baseline-83e1777
INCS="-I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include"
cc -std=gnu11 -O2 $INCS $WT/tests/headless/ab_bench.c -o /tmp/abb3_a -L$BASE -lmgl -Wl,-rpath,$BASE || exit 1
cc -std=gnu11 -O2 $INCS $WT/tests/headless/ab_bench.c -o /tmp/abb3_b -L$WT/ab/pristine -lmgl -Wl,-rpath,$WT/ab/pristine || exit 1
cc -std=gnu11 -O2 $INCS $WT/tests/headless/ab_bench.c -o /tmp/abb3_c -L$WT/build -lmgl -Wl,-rpath,$WT/build || exit 1
for mode in 0 1; do
    iters=$([ $mode = 0 ] && echo 2000 || echo 600)
    for v in a b c; do : > /tmp/abb3_$mode.$v; done
    for r in $(seq 1 7); do
        /tmp/abb3_a $mode $iters >> /tmp/abb3_$mode.a
        /tmp/abb3_b $mode $iters >> /tmp/abb3_$mode.b
        /tmp/abb3_c $mode $iters >> /tmp/abb3_$mode.c
    done
    a=$(sort -n /tmp/abb3_$mode.a | head -1 | awk '{print $NF}')
    b=$(sort -n /tmp/abb3_$mode.b | head -1 | awk '{print $NF}')
    c=$(sort -n /tmp/abb3_$mode.c | head -1 | awk '{print $NF}')
    echo "mode$mode  A(base-83e1777)=$a  B(pristine@mytree)=$b  C(merged-tip)=$c  load=$(sysctl -n vm.loadavg | cut -c9-13)"
done
