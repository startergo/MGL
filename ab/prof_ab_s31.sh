#!/bin/bash
# s31: interleaved MGL_PROFILE bucket comparison, current vs supervisor
# baseline-af20103 (P).  usage: prof_ab_s31.sh [scene] [N]
set -u
WT=$(cd "$(dirname "$0")/.." && pwd)
A=${LIB_A_OVERRIDE:-$WT/build/nsgl/libmgl_nsgl.dylib}
B=${LIB_B_OVERRIDE:-$HOME/MGL-cts/baseline-f973a81/libmgl_nsgl.dylib}
SCENE=${1:-jellyfish}
N=${2:-3}
rm -f /tmp/prof_A_all.txt /tmp/prof_B_all.txt
one() {  # $1 lib, $2 tag
    env MGL_PROFILE=1 DYLD_INSERT_LIBRARIES=$1 /opt/homebrew/bin/glmark2-macos \
        --data-path /opt/homebrew/share/glmark2 -b $SCENE 2>&1 | grep "ms total" | tail -3 > /tmp/prof_$2.txt
}
for i in $(seq $N); do
    one $A a; one $B b; one $B b2; one $A a2
    cat /tmp/prof_a.txt /tmp/prof_a2.txt >> /tmp/prof_A_all.txt
    cat /tmp/prof_b.txt /tmp/prof_b2.txt >> /tmp/prof_B_all.txt
done
python3 - <<'PY'
import re, statistics
def med(path):
    buckets = {}
    for line in open(path, errors='replace'):
        for m in re.finditer(r'(\S+?)=([\d.]+)\|([\d.]+)\|([\d.]+)', line):
            name, msframe = m.group(1), float(m.group(3))
            buckets.setdefault(name, []).append(msframe)
    return {k: statistics.median(v) for k, v in buckets.items()}
A, B = med('/tmp/prof_A_all.txt'), med('/tmp/prof_B_all.txt')
rows = []
for k in sorted(set(A) | set(B), key=lambda k: -(A.get(k, 0))):
    a, b = A.get(k, 0), B.get(k, 0)
    rows.append("%-24s cur %.4f  P %.4f  delta %+0.4f ms/f" % (k, a, b, a - b))
print("\n".join(rows))
print("sum: cur %.4f  P %.4f  (%+.1f%%)" % (sum(A.values()), sum(B.values()),
      100.0 * (sum(A.values()) / sum(B.values()) - 1)))
PY
