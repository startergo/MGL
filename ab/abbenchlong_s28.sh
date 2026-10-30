#!/bin/bash
# s28 long-run ab_bench mode0: A = baseline-83e1777, C = merged tip.
# 20000 drawArrays per rep, 7 reps interleaved; min + median + all reps shown.
WT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=$HOME/MGL-cts/baseline-83e1777
INCS="-I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include"
cc -std=gnu11 -O2 $INCS $WT/tests/headless/ab_bench.c -o /tmp/abbl_a -L$BASE -lmgl -Wl,-rpath,$BASE || exit 1
cc -std=gnu11 -O2 $INCS $WT/tests/headless/ab_bench.c -o /tmp/abbl_c -L$WT/build -lmgl -Wl,-rpath,$WT/build || exit 1
: > /tmp/abbl.a; : > /tmp/abbl.c
for r in $(seq 1 7); do
    /tmp/abbl_a 0 20000 >> /tmp/abbl.a
    /tmp/abbl_c 0 20000 >> /tmp/abbl.c
done
for v in a c; do
    s=$(sort -n /tmp/abbl.$v | awk '{print $NF}')
    min=$(echo "$s" | head -1); med=$(echo "$s" | sed -n 4p)
    echo "$v: min=$min med=$med  all=$(echo $s | tr '\n' ' ')"
done
echo "load=$(sysctl -n vm.loadavg | cut -c9-13)"
