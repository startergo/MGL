#!/bin/bash
# run_wall_s19.sh — interleaved A/B of CPU ms/frame on the [ideas]-shape bench
# (MGL_PROFILE off, wall clock around the measured 120 frames).
#   A = ab/base19 (9637b21)   B = worktree tip
WT="$(cd "$(dirname "$0")/.." && pwd)"
ROUNDS=${ROUNDS:-12}
SRC=$WT/tests/headless/test_compat_ideaspec.c

cc -std=gnu11 -O0 -g -w -I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include \
   $SRC -o /tmp/s19_a -L$WT/ab/base19/build -lmgl -Wl,-rpath,$WT/ab/base19/build -lm || exit 1
cc -std=gnu11 -O0 -g -w -I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include \
   $SRC -o /tmp/s19_b -L$WT/build -lmgl -Wl,-rpath,$WT/build -lm || exit 1

: > /tmp/s19_wall_a
: > /tmp/s19_wall_b
for r in $(seq 1 $ROUNDS); do
    /tmp/s19_a | grep WALL >> /tmp/s19_wall_a
    /tmp/s19_b | grep WALL >> /tmp/s19_wall_b
done
python3 - <<EOF
import statistics as st
a = [float(l.split()[1]) for l in open('/tmp/s19_wall_a')]
b = [float(l.split()[1]) for l in open('/tmp/s19_wall_b')]
print(f"A(base): min {min(a):7.3f}  med {st.median(a):7.3f}  mean {st.mean(a):7.3f}  (n={len(a)})")
print(f"B(new) : min {min(b):7.3f}  med {st.median(b):7.3f}  mean {st.mean(b):7.3f}  (n={len(b)})")
dmin, dmed = min(b)-min(a), st.median(b)-st.median(a)
print(f"delta  : min {dmin:+7.3f} ({dmin/min(a)*100:+.1f}%)   med {dmed:+7.3f} ({dmed/st.median(a)*100:+.1f}%)")
EOF
