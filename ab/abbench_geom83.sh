#!/bin/bash
# abbench_geom83.sh — locked A/B of the plain draw path (ab_bench modes 0/1/31).
# old = baseline copy in build/ab_old, new = the working build in build/ab_new.
# Contention-robust: old/new run back-to-back per round; reports the min per side
# AND the median of per-round PAIR RATIOS (sustained background load hits both
# sides of a pair, the ratio cancels it).
# Usage: ab/abbench_geom83.sh [rounds] [modes...]   (modes as m:iters)
set -u
R=${1:-7}; shift 2>/dev/null || true
MODES=${*:-"0:2000 1:1000 31:1000"}
OUT=$(mktemp -d)
rep=0
for spec in $MODES; do
  m=${spec%%:*}; n=${spec##*:}
  rep=0
  while [ $rep -lt "$R" ]; do
    rep=$((rep+1))
    ./build/ab_old/ab "$m" "$n" 2>/dev/null | grep RESULT | sed "s/^/old /" >> "$OUT/raw"
    ./build/ab_new/ab "$m" "$n" 2>/dev/null | grep RESULT | sed "s/^/new /" >> "$OUT/raw"
  done
done
python3 - "$OUT/raw" <<'PY'
import sys, collections, statistics
old, new = collections.defaultdict(list), collections.defaultdict(list)
for line in open(sys.argv[1]):
    side, rest = line.split(None, 1)
    w = rest.split()
    (old if side == 'old' else new)[' '.join(w[1:-1])].append((float(w[-1]), int(w[-2][1:]) if False else 0))
keys = sorted(set(old) | set(new))
print('%-30s %10s %10s %8s %12s' % ('case', 'old min', 'new min', 'd(min)', 'median ratio'))
for k in keys:
    o = [v for v, _ in old.get(k, [])]; n = [v for v, _ in new.get(k, [])]
    if not o or not n: continue
    m = min(len(o), len(n))
    ratios = sorted(o[i] / n[i] for i in range(m))   # >1 => new faster
    print('%-30s %10.3f %10.3f %7.1f%% %11.3f' % (k, min(o), min(n), (min(n) - min(o)) / min(o) * 100.0, statistics.median(ratios)))
PY
rm -rf "$OUT"
