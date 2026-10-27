#!/bin/bash
# run_ab_s19.sh — interleaved A/B of the per-draw CPU buckets on the
# [ideas]-shape bench (tests/headless/test_compat_ideaspec.c):
#   A = ab/base19   (merge 9637b21, pre-s19)
#   B = worktree    (compact VAO layout key etc.)
# Same bench source for both; medians over ROUNDS interleaved runs.
WT="$(cd "$(dirname "$0")/.." && pwd)"
ROUNDS=${ROUNDS:-9}
SRC=$WT/tests/headless/test_compat_ideaspec.c

cc -std=gnu11 -O0 -g -w -I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include \
   $SRC -o /tmp/s19_a -L$WT/ab/base19/build -lmgl -Wl,-rpath,$WT/ab/base19/build -lm || exit 1
cc -std=gnu11 -O0 -g -w -I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include \
   $SRC -o /tmp/s19_b -L$WT/build -lmgl -Wl,-rpath,$WT/build -lm || exit 1

: > /tmp/s19_a.out
: > /tmp/s19_b.out
for r in $(seq 1 $ROUNDS); do
    for side in a b; do
        MGL_PROFILE=1 /tmp/s19_$side 2>&1 | grep "MGL PROFILE(ms" | tail -1 >> /tmp/s19_${side}.out
    done
done
echo "load: $(uptime | sed 's/.*load averages/load averages/')"
python3 - <<'EOF'
import statistics as st
import re
def parse(path):
    rows = []
    for line in open(path):
        for m in re.finditer(r'([A-Za-z0-9_+-]+)=([0-9.]+)\|([0-9.]+)\|([0-9.]+)', line):
            rows.append((m.group(1), float(m.group(3)), float(m.group(4))))
    return rows
a, b = parse('/tmp/s19_a.out'), parse('/tmp/s19_b.out')
keys = [k for k, _, _ in a]
print(f"{'bucket':28s} {'A(base)':>10s} {'B(new)':>10s} {'delta':>9s} {'delta%':>8s}")
for k in dict.fromkeys(keys):
    av = [v for kk, v, c in a if kk == k]
    bv = [v for kk, v, c in b if kk == k]
    am, bm = min(av), min(bv)
    amd, bmd = st.median(av), st.median(bv)
    d = bm - am
    pct = (d / am * 100.0) if am else 0.0
    print(f"{k:28s} min {am:8.3f}->{bm:8.3f} ({pct:+6.1f}%)  med {amd:8.3f}->{bmd:8.3f}")
EOF
