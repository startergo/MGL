#!/bin/bash
# advblend A/B: baseline-f600d55 (no advblend) vs tip (advblend merged on top
# of f600d55) — must show ZERO per-draw cost when no advanced equation is set.
#
# Same absolute-dylib discipline as abbench_abs.sh (copies + install_name_tool,
# DYLD_PRINT_LIBRARIES proof the sides differ), plus two noise controls:
#   * round order alternates (A,B / B,A / ...) so neither side systematically
#     runs second against a hot machine;
#   * min AND median of the rounds are reported.
# usage: abbench_advblend.sh <abs baseline dir> [rounds=12] [m0-iters=20000] [m1-iters=600]
set -u
WT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=${1:?abs baseline dir (a DIRECTORY containing libmgl.dylib; for a self-test noise-floor run pass a copy of build/)}
R=${2:-12}
IT0=${3:-20000}
IT1=${4:-600}
[ -d "$BASE" ] || { echo "no baseline dir $BASE"; exit 1; }
rm -rf /tmp/mglab; mkdir -p /tmp/mglab/base /tmp/mglab/new
cp "$BASE/libmgl.dylib" /tmp/mglab/base/ && install_name_tool -id /tmp/mglab/base/libmgl.dylib /tmp/mglab/base/libmgl.dylib && codesign -f -s - /tmp/mglab/base/libmgl.dylib >/dev/null 2>&1
cp "$WT/build/libmgl.dylib" /tmp/mglab/new/ && install_name_tool -id /tmp/mglab/new/libmgl.dylib /tmp/mglab/new/libmgl.dylib && codesign -f -s - /tmp/mglab/new/libmgl.dylib >/dev/null 2>&1
INCS="-I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include"
cc -std=gnu11 -O2 $INCS "$WT/tests/headless/ab_bench.c" -o /tmp/abbA -L/tmp/mglab/base -lmgl || exit 1
cc -std=gnu11 -O2 $INCS "$WT/tests/headless/ab_bench.c" -o /tmp/abbB -L/tmp/mglab/new  -lmgl || exit 1
a=$(DYLD_PRINT_LIBRARIES=1 /tmp/abbA 0 2 2>&1 | grep -m1 -o '/tmp/mglab/[a-z]*/libmgl.dylib')
b=$(DYLD_PRINT_LIBRARIES=1 /tmp/abbB 0 2 2>&1 | grep -m1 -o '/tmp/mglab/[a-z]*/libmgl.dylib')
echo "loaded: A=$a B=$b"
[ "$a" = /tmp/mglab/base/libmgl.dylib ] && [ "$b" = /tmp/mglab/new/libmgl.dylib ] || { echo "SIDES NOT DISTINCT - abort"; exit 1; }
for mode in 0 1; do
    iters=$([ $mode = 0 ] && echo $IT0 || echo $IT1)
    : > /tmp/abb_$mode.a; : > /tmp/abb_$mode.b
    for r in $(seq 1 "$R"); do
        if [ $((r % 2)) = 1 ]; then order="/tmp/abbA /tmp/abbB"; else order="/tmp/abbB /tmp/abbA"; fi
        set -- $order
        $1 $mode $iters | awk '{print $NF}' >> /tmp/abb_$mode.a
        $2 $mode $iters | awk '{print $NF}' >> /tmp/abb_$mode.b
    done
    python3 - "$mode" <<'EOF'
import sys, subprocess
mode = sys.argv[1]
out = {}
for v in ("a", "b"):
    vals = sorted(float(l) for l in open(f"/tmp/abb_{mode}.{v}") if l.strip())
    n = len(vals)
    med = vals[n//2] if n % 2 else (vals[n//2-1]+vals[n//2])/2
    out[v] = (min(vals), med, vals)
for stat, i in (("min", 0), ("median", 1)):
    x, y = out["a"][i], out["b"][i]
    print(f"mode{mode} {stat:6s} base={x:9.4f} new={y:9.4f} delta={(y-x)/x*100:+.1f}%")
for v in ("a", "b"):
    print(f"       {'base' if v=='a' else 'new'}[all]={' '.join(f'{t:.3f}' for t in out[v][2])}")
EOF
    echo "       load=$(sysctl -n vm.loadavg)"
done
