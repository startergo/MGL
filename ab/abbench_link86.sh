#!/bin/bash
# abbench_link86.sh — link/introspection A/B for the introspect r86 round.
# old = first arg (library dir, default .dbg9f/build = 9f0c6f0), new = second
# (default build/).  Interleaved rounds, CPU clock, min + pair-ratio medians.
set -u
WT="$(cd "$(dirname "$0")/.." && pwd)"
OLD=${1:-$WT/.dbg9f/build}
NEW=${2:-$WT/build}
R=${3:-5}
N=${4:-40}
INCS="-I$WT/tests/headless -I$WT/MGL/include/GL -I$WT/MGL/include"
cc -std=gnu11 -O2 $INCS $WT/tests/headless/link_bench86.c -o /tmp/lb86_old -L$OLD -lmgl -Wl,-rpath,$OLD || exit 1
cc -std=gnu11 -O2 $INCS $WT/tests/headless/link_bench86.c -o /tmp/lb86_new -L$NEW -lmgl -Wl,-rpath,$NEW || exit 1
OUT=$(mktemp -d)
rep=0
while [ $rep -lt "$R" ]; do
  rep=$((rep+1))
  MGL_AB_CPU=1 /tmp/lb86_old "$N" 2>/dev/null | grep RESULT | sed 's/^/old /' >> "$OUT/raw"
  MGL_AB_CPU=1 /tmp/lb86_new "$N" 2>/dev/null | grep RESULT | sed 's/^/new /' >> "$OUT/raw"
done
python3 - "$OUT/raw" <<'PY'
import sys, collections, statistics
old, new = collections.defaultdict(list), collections.defaultdict(list)
for line in open(sys.argv[1]):
    side, rest = line.split(None, 1)
    w = rest.split()
    (old if side == 'old' else new)[' '.join(w[1:-1])].append(float(w[-1]))
for k in sorted(set(old) | set(new)):
    o, n = sorted(old.get(k, [])), sorted(new.get(k, []))
    q = min(len(o), len(n))
    r = sorted(o[i]/n[i] for i in range(q))
    print('%-34s old_min=%8.3f new_min=%8.3f  d(min)=%+6.1f%%  ratio_med=%.3f' %
          (k, o[0], n[0], (n[0]-o[0])/o[0]*100, statistics.median(r)))
PY
rm -rf "$OUT"
