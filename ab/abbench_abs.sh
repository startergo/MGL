#!/bin/bash
# ab_bench A/B that ACTUALLY loads two different libraries.
#
# The old abbench_s2x.sh linked both sides with -L<dir> -lmgl, but libmgl.dylib's
# install name is the RELATIVE path "build/libmgl.dylib" (no @rpath), so dyld
# resolved it against the CWD at launch: run from the worktree root, BOTH sides
# loaded wt/compat/build/libmgl.dylib (verified 2026-10-28 with DYLD_PRINT_LIBRARIES:
# identical UUIDs) - every m0/m1 "delta" those runs printed was noise (plus a
# base-first/new-second warm-up asymmetry).  Fix: copy each side's dylib to a
# private dir, install_name_tool -id it to an ABSOLUTE path, link against the copy.
#
# usage: abbench_abs.sh <abs baseline dir> [rounds]   (default 7; min reported)
set -u
WT="$(cd "$(dirname "$0")/.." && pwd)"
BASE=${1:?abs baseline dir}
R=${2:-7}
[ -d "$BASE" ] || { echo "no baseline dir $BASE"; exit 1; }
rm -rf /tmp/mglab; mkdir -p /tmp/mglab/base /tmp/mglab/new
cp "$BASE/libmgl.dylib" /tmp/mglab/base/ && install_name_tool -id /tmp/mglab/base/libmgl.dylib /tmp/mglab/base/libmgl.dylib && codesign -f -s - /tmp/mglab/base/libmgl.dylib >/dev/null 2>&1
cp "$WT/build/libmgl.dylib" /tmp/mglab/new/ && install_name_tool -id /tmp/mglab/new/libmgl.dylib /tmp/mglab/new/libmgl.dylib && codesign -f -s - /tmp/mglab/new/libmgl.dylib >/dev/null 2>&1
INCS="-I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include"
cc -std=gnu11 -O2 $INCS "$WT/tests/headless/ab_bench.c" -o /tmp/abbA -L/tmp/mglab/base -lmgl || exit 1
cc -std=gnu11 -O2 $INCS "$WT/tests/headless/ab_bench.c" -o /tmp/abbB -L/tmp/mglab/new  -lmgl || exit 1
# proof the two sides differ before trusting any number
a=$(DYLD_PRINT_LIBRARIES=1 /tmp/abbA 0 2 2>&1 | grep -m1 -o '/tmp/mglab/[a-z]*/libmgl.dylib')
b=$(DYLD_PRINT_LIBRARIES=1 /tmp/abbB 0 2 2>&1 | grep -m1 -o '/tmp/mglab/[a-z]*/libmgl.dylib')
echo "loaded: A=$a B=$b"
[ "$a" = /tmp/mglab/base/libmgl.dylib ] && [ "$b" = /tmp/mglab/new/libmgl.dylib ] || { echo "SIDES NOT DISTINCT - abort"; exit 1; }
for mode in 0 1; do
    iters=$([ $mode = 0 ] && echo 2000 || echo 600)
    : > /tmp/abb_$mode.a; : > /tmp/abb_$mode.b
    for r in $(seq 1 "$R"); do
        /tmp/abbA $mode $iters | awk '{print $NF}' >> /tmp/abb_$mode.a
        /tmp/abbB $mode $iters | awk '{print $NF}' >> /tmp/abb_$mode.b
    done
    a=$(sort -n /tmp/abb_$mode.a | head -1); b=$(sort -n /tmp/abb_$mode.b | head -1)
    pct=$(python3 -c "print(f'{($b-$a)/$a*100:+.1f}%')")
    echo "mode$mode base(min)=$a new(min)=$b delta=$pct  base[all]=$(sort -n /tmp/abb_$mode.a | tr '\n' ' ')"
    echo "       new[all]=$(sort -n /tmp/abb_$mode.b | tr '\n' ' ') load=$(sysctl -n vm.loadavg)"
done
