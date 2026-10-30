#!/bin/bash
# s31/s33 A/B — the supervisor's REJECTION-REPRO method (Oct 30 00:46):
#   ONE glmark2-macos process running `-b ideas -b jellyfish -b gl41-geometry`
#   (--data-path /opt/homebrew/share/glmark2), alternating rounds (A B B A per
#   round, cancels linear drift), each glmark2 process under
#   /usr/bin/lockf -k /tmp/mgl_glmark2.lock (mutual exclusion with other agents).
#   A = this tree's build/nsgl, B(P) = ~/MGL-cts/baseline-f973a81.
#   APPEND mode (default): every sample is tagged with the 1-min load average
#   right after the run, so the analysis can drop contended samples.
#   usage: ROUNDS=6 bash ab/run_ab_s31.sh        LOG=/tmp/ab_rounds.log
set -u
WT=$(cd "$(dirname "$0")/.." && pwd)
LIB_A=${LIB_A_OVERRIDE:-$WT/build/nsgl/libmgl_nsgl.dylib}
LIB_B=${LIB_B_OVERRIDE:-$HOME/MGL-cts/baseline-f973a81/libmgl_nsgl.dylib}
LOG=${LOG:-/tmp/ab_s31_rounds.log}
LOCK=/tmp/mgl_glmark2.lock
BENCH="-b ideas -b jellyfish -b gl41-geometry"
[ -n "${AB_APPEND:-}" ] || : > "$LOG"

run() { # $1 = nsgl dylib -> "ideas=N jellyfish=N gl41=N cpu=S" (cpu = child user+sys s)
  local out=/tmp/ab_run.$$.txt
  bash -c '/usr/bin/lockf -k '"$LOCK"' env DYLD_INSERT_LIBRARIES="'"$1"'" \
      /opt/homebrew/bin/glmark2-macos --data-path /opt/homebrew/share/glmark2 '"$BENCH"' 2>&1; times' > $out 2>&1
  awk '/^\[ideas\]/{i=$4} /^\[jellyfish\]/{j=$4} /^\[gl41-geometry\]/{g=$4}
       /^0m[0-9]/ { tn++; if (tn == 2) { u=$1; s=$2; sub(/^.*m/,"",u); sub(/^.*m/,"",s);
                                 sub(/s$/,"",u); sub(/s$/,"",s); cpu = u + s } }
       END{printf "ideas=%s jellyfish=%s gl41=%s cpu=%.2f\n", i, j, g, cpu}' $out
  rm -f $out
}

echo "A=$LIB_A"; echo "B=$LIB_B"
r0=${ROUND0:-0}
for r in $(seq ${ROUNDS:-6}); do
  rn=$(( r0 + r ))
  a=$(run "$LIB_A");  la=$(sysctl -n vm.loadavg | awk '{print $2}')
  b=$(run "$LIB_B");  lb=$(sysctl -n vm.loadavg | awk '{print $2}')
  b2=$(run "$LIB_B"); lb2=$(sysctl -n vm.loadavg | awk '{print $2}')
  a2=$(run "$LIB_A"); la2=$(sysctl -n vm.loadavg | awk '{print $2}')
  echo "round$rn A $a load=$la | $a2 load=$la2" | tee -a "$LOG"
  echo "round$rn B $b load=$lb | $b2 load=$lb2" | tee -a "$LOG"
done

python3 - "$LOG" <<'EOF'
import re, sys, statistics as st
A, B = {}, {}
CPUS = {}
for line in open(sys.argv[1]):
    m = re.match(r'round\d+ ([AB]) (.*)', line.strip())
    if not m: continue
    d = (A if m.group(1) == 'A' else B)
    for half in m.group(2).split('|'):
        mm = re.match(r'\s*ideas=(\d+) jellyfish=(\d+) gl41=(\d+)', half)
        lm = re.search(r'load=([\d.]+)', half)
        cm = re.search(r'cpu=([\d.]+)', half)
        if not mm: continue
        load = float(lm.group(1)) if lm else 99.0
        if cm: CPUS.setdefault(m.group(1), []).append((float(cm.group(1)), int(mm.group(1)) + int(mm.group(2)) + int(mm.group(3))))
        for k, i in [('ideas', 1), ('jellyfish', 2), ('gl41', 3)]:
            d.setdefault(k, []).append((int(mm.group(i)), load))
MAXLOAD = 8.0
for k in ['ideas', 'jellyfish', 'gl41']:
    af = [v for v, l in A.get(k, []) if l <= MAXLOAD]
    bf = [v for v, l in B.get(k, []) if l <= MAXLOAD]
    a_all = [v for v, _ in A.get(k, [])]
    b_all = [v for v, _ in B.get(k, [])]
    def rep(tag, xs, ys):
        if xs and ys:
            print(f"{k} [{tag} load<={MAXLOAD}]: A n={len(xs)} med {st.median(xs):.0f}  "
                  f"P n={len(ys)} med {st.median(ys):.0f}  A/P {st.median(xs)/st.median(ys)*100:+.2f}%")
        else:
            print(f"{k} [{tag}]: no samples under load {MAXLOAD} (A n={len(a_all)}, P n={len(b_all)})")
    rep('filtered', af, bf)
    if a_all and b_all:
        print(f"{k} [all]: A med {st.median(a_all):.0f}  P med {st.median(b_all):.0f}  "
              f"A/P {st.median(a_all)/st.median(b_all)*100:+.2f}%  (A spread {min(a_all)}-{max(a_all)}, P {min(b_all)}-{max(b_all)})")
for side in ('A', 'B'):
    if CPUS.get(side):
        # CPU seconds per 1000 frames over all three scenes (each scene runs 10 s)
        vals = [100.0 * c / max(1.0, f) for c, f in CPUS[side]]   # each scene runs 10 s -> frames = 10*f
        print(f"cpu-ms/1000frames [{side}] n={len(vals)} median {st.median(vals):.1f} "
              f"spread {min(vals):.1f}-{max(vals):.1f}")
EOF
