#!/bin/bash
# s35 ab_bench: A = supervisor baseline-e8ff34b dylib, C = merged tip (this tree).
# min of 7, alternating order per round (load immunity), 2 modes.
WT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=$HOME/MGL-cts/baseline-e8ff34b
INCS="-I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include"
cc -std=gnu11 -O2 $INCS $WT/tests/headless/ab_bench.c -o /tmp/abbs35_a -L$BASE -lmgl -Wl,-rpath,$BASE || exit 1
cc -std=gnu11 -O2 $INCS $WT/tests/headless/ab_bench.c -o /tmp/abbs35_c -L$WT/build -lmgl -Wl,-rpath,$WT/build || exit 1
for mode in 0 1; do
    iters=$([ $mode = 0 ] && echo 2000 || echo 600)
    : > /tmp/abbs35_$mode.a; : > /tmp/abbs35_$mode.c
    for r in $(seq 1 7); do
        if [ $((r % 2)) = 1 ]; then
            /tmp/abbs35_a $mode $iters >> /tmp/abbs35_$mode.a
            /tmp/abbs35_c $mode $iters >> /tmp/abbs35_$mode.c
        else
            /tmp/abbs35_c $mode $iters >> /tmp/abbs35_$mode.c
            /tmp/abbs35_a $mode $iters >> /tmp/abbs35_$mode.a
        fi
    done
    a=$(sort -n /tmp/abbs35_$mode.a | head -1 | awk '{print $NF}')
    c=$(sort -n /tmp/abbs35_$mode.c | head -1 | awk '{print $NF}')
    echo "mode$mode  A(baseline-e8ff34b)=$a  C(merged-tip)=$c  load=$(sysctl -n vm.loadavg | cut -c9-13)"
done
