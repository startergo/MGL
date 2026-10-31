#!/bin/bash
# s36 ab_bench (interleave-within-round): A = baseline-1121e66 dylib, C = merged tip.
# min of 7, A/C order alternates within every round (kills monotonic-drift bias),
# modes: 0 drawArrays, 1 blend-flip, 31 blendFunc-flip, 32 blendEquation-flip
# (31/32 exercise the s36 DIRTY_ALPHA_STATE pipeline-gate change),
# 6 readPixels PBO (fbRead/fbSourceFor - the rgb9e5 s36 hook site).
WT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=$HOME/MGL-cts/baseline-1121e66
INCS="-I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include"
cc -std=gnu11 -O2 $INCS $WT/tests/headless/ab_bench.c -o /tmp/abbs36_a -L$BASE -lmgl -Wl,-rpath,$BASE || exit 1
cc -std=gnu11 -O2 $INCS $WT/tests/headless/ab_bench.c -o /tmp/abbs36_c -L$WT/build -lmgl -Wl,-rpath,$WT/build || exit 1
for mode in 0 1 31 32 6; do
    iters=$([ $mode = 0 ] && echo 2000 || ([ $mode = 6 ] && echo 40 || echo 600))
    : > /tmp/abbs36_$mode.a; : > /tmp/abbs36_$mode.c
    for r in $(seq 1 7); do
        if [ $((r % 2)) = 1 ]; then
            /tmp/abbs36_a $mode $iters >> /tmp/abbs36_$mode.a
            /tmp/abbs36_c $mode $iters >> /tmp/abbs36_$mode.c
        else
            /tmp/abbs36_c $mode $iters >> /tmp/abbs36_$mode.c
            /tmp/abbs36_a $mode $iters >> /tmp/abbs36_$mode.a
        fi
    done
    a=$(sort -n /tmp/abbs36_$mode.a | head -1 | awk '{print $NF}')
    c=$(sort -n /tmp/abbs36_$mode.c | head -1 | awk '{print $NF}')
    echo "mode$mode  A(baseline-1121e66)=$a  C(merged-tip)=$c  load=$(sysctl -n vm.loadavg | cut -c9-13)"
done
