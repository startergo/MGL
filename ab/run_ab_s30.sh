#!/bin/bash
# s30 A/B: two nsgl builds interleaved under the glmark2 lock.
#   A = current tree build (LIB_A_OVERRIDE to point elsewhere)
#   B = supervisor baseline-d71ce14 (LIB_B_OVERRIDE to point elsewhere)
# usage: ROUNDS=6 bash ab/run_ab_s30.sh [scenes...]   (default: ideas jellyfish)
# order per round: A B B A (cancels linear drift); medians printed at the end.
set -u
WT=$(cd "$(dirname "$0")/.." && pwd)
DATA=/tmp/glmark2-native/opt/homebrew/share/glmark2
LIB_A=${LIB_A_OVERRIDE:-$WT/build/nsgl/libmgl_nsgl.dylib}
LIB_B=${LIB_B_OVERRIDE:-$HOME/MGL-cts/baseline-d71ce14/libmgl_nsgl.dylib}
SCENES="${@:-ideas jellyfish}"
run() { perl -e 'alarm 180; exec @ARGV' env DYLD_INSERT_LIBRARIES=$1 /opt/homebrew/bin/glmark2-macos --data-path $DATA -b $2 2>&1 | grep "glmark2 Score" | tr -dc '0-9'; }
echo "A=$LIB_A" ; echo "B=$LIB_B"
[ -n "${AB_APPEND:-}" ] || : > /tmp/ab_s30_rounds.log
for r in $(seq ${ROUNDS:-6}); do
  for scene in $SCENES; do
    a=$(run $LIB_A $scene); b=$(run $LIB_B $scene)
    b2=$(run $LIB_B $scene); a2=$(run $LIB_A $scene)
    echo "round$r $scene A=$a $a2 B=$b $b2" | tee -a /tmp/ab_s30_rounds.log
    echo "  load: $(sysctl -n vm.loadavg)"
  done
done
# medians + ratio per scene (A/B; 1.00 = parity, >1 = A faster)
grep '^round' /tmp/ab_s30_rounds.log 2>/dev/null | python3 $(dirname "$0")/ab_s30_med.py
