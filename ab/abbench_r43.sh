#!/bin/bash
# r43 ab_bench (tess domain): A = baseline-288e724 dylib, C = this tree.
# GL_PATCHES draw-path microbench (tests: drawArrays / noTCS / TCS+TES / 64-patch),
# min of 7, alternating order per round (load immunity), CPU-time accounting
# (MGL_AB_CPU=1) so the numbers are load-independent.
WT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=$HOME/MGL-cts/baseline-288e724
INCS="-I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include"
cc -std=gnu11 -O2 $INCS $WT/ab/ab_patches.c -o /tmp/abbr43_a -L$BASE -lmgl -Wl,-rpath,$BASE || exit 1
cc -std=gnu11 -O2 $INCS $WT/ab/ab_patches.c -o /tmp/abbr43_c -L$WT/build -lmgl -Wl,-rpath,$WT/build || exit 1
for mode in 0 1 2 3; do
    iters=1000
    : > /tmp/abbr43_$mode.a; : > /tmp/abbr43_$mode.c
    for r in $(seq 1 7); do
        if [ $((r % 2)) = 1 ]; then
            MGL_AB_CPU=1 /tmp/abbr43_a $iters 2>/dev/null >> /tmp/abbr43_$mode.a
            MGL_AB_CPU=1 /tmp/abbr43_c $iters 2>/dev/null >> /tmp/abbr43_$mode.c
        else
            MGL_AB_CPU=1 /tmp/abbr43_c $iters 2>/dev/null >> /tmp/abbr43_$mode.c
            MGL_AB_CPU=1 /tmp/abbr43_a $iters 2>/dev/null >> /tmp/abbr43_$mode.a
        fi
    done
    case $mode in
        0) k="drawArrays-tris submit";; 1) k="patches-noTCS submit";;
        2) k="patches-TCS-TES submit";; 3) k="patches-TCS-TES-64";;
    esac
    a=$(grep "$k" /tmp/abbr43_$mode.a | awk '{print $NF}' | sort -n | head -1)
    c=$(grep "$k" /tmp/abbr43_$mode.c | awk '{print $NF}' | sort -n | head -1)
    echo "[$k]  A(baseline-288e724)=${a}ms/1000  C(r43)=${c}ms/1000  load=$(sysctl -n vm.loadavg | cut -c9-13)"
done
