#!/bin/bash
# compat-domain A/B: A = baseline-d11d184 dylib, C = this tree.
# Legacy draw-path microbench (imm quads, attrib push/pop + light state,
# legacy glGet), min of 7, alternating order per round (load immunity),
# CPU-time accounting (MGL_AB_CPU=1).
WT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=$HOME/MGL-cts/baseline-d11d184
INCS="-I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include"
cc -std=gnu11 -O2 $INCS $WT/ab/ab_compat_draw.c -o /tmp/abcomp_a -L$BASE -lmgl -Wl,-rpath,$BASE || exit 1
cc -std=gnu11 -O2 $INCS $WT/ab/ab_compat_draw.c -o /tmp/abcomp_c -L$WT/build -lmgl -Wl,-rpath,$WT/build || exit 1
: > /tmp/abcomp.a; : > /tmp/abcomp.c
for r in $(seq 1 9); do
    if [ $((r % 2)) = 1 ]; then
        MGL_AB_CPU=1 /tmp/abcomp_a 2000 2>/dev/null >> /tmp/abcomp.a
        MGL_AB_CPU=1 /tmp/abcomp_c 2000 2>/dev/null >> /tmp/abcomp.c
    else
        MGL_AB_CPU=1 /tmp/abcomp_c 2000 2>/dev/null >> /tmp/abcomp.c
        MGL_AB_CPU=1 /tmp/abcomp_a 2000 2>/dev/null >> /tmp/abcomp.a
    fi
done
for k in "compat imm-quads submit" "compat attrib-lit submit" "compat legacy get"; do
    a=$(grep "$k" /tmp/abcomp.a | awk '{print $NF}' | sort -n | head -1)
    c=$(grep "$k" /tmp/abcomp.c | awk '{print $NF}' | sort -n | head -1)
    pct=$(python3 -c "print(f'{(float('$c')/float('$a')-1)*100:+.1f}%')" 2>/dev/null)
    echo "[$k]  A(baseline-d11d184)=${a}us/draw  C(tip)=${c}us/draw  delta=${pct}  load=$(sysctl -n vm.loadavg | cut -c9-13)"
done
