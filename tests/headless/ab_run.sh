#!/bin/bash
# ab_run.sh — interleaved A/B timing (medians) of build/ab_old vs build/ab_new
set -u
runs=${1:-5}
modes="3:20 7:20 8:20 6:3"
for spec in $modes; do
  m=${spec%%:*}; n=${spec##*:}
  for rep in $(seq $runs); do
    a=$(./build/ab_old/ab $m $n 2>/dev/null | grep RESULT)
    b=$(./build/ab_new/ab $m $n 2>/dev/null | grep RESULT)
    echo "$a|$b"
  done
done | python3 -c "
import sys, statistics
old, new = {}, {}
for line in sys.stdin:
    parts = [p.strip() for p in line.strip().split('|')]
    if len(parts) != 2: continue
    def grab(p, d):
        w = p.split()
        lbl = ' '.join(w[1:-1]); d.setdefault(lbl, []).append(float(w[-1]))
    grab(parts[0], old); grab(parts[1], new)
print('%-34s %10s %10s %8s' % ('case', 'old(ms)', 'new(ms)', 'speedup'))
for k in old:
    if k not in new: continue
    o, n = statistics.median(old[k]), statistics.median(new[k])
    print('%-34s %10.3f %10.3f %7.2fx' % (k, o, n, o / n if n else 0))
"
