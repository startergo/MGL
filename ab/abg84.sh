#!/bin/bash
# abg84.sh — locked glmark2 A/B, 8 rounds, ideas+jellyfish+gl41-geometry medians.
# r84: baseline is f8be669 (gl46-port tip after geom r82+r83, tess r43b, compat s38);
# C = this branch's nsgl build.  Order alternates per round (P/C vs C/P) so thermal or
# background drift cannot favor a side.  Guarded by the shared lock (one glmark2 on the
# machine at a time).
set -u
if [ -z "${MGL_AB_LOCK:-}" ]; then
    MGL_AB_LOCK=1 exec /usr/bin/lockf -k /tmp/mgl_glmark2.lock /bin/bash "$0" "$@"
fi
G=/opt/homebrew/bin/glmark2-macos
D=/opt/homebrew/share/glmark2
B=$HOME/MGL-cts/glmark2-bench
C=$B/geomC/libmgl_nsgl.dylib
P=$HOME/MGL-cts/baseline-f8be669/libmgl_nsgl.dylib
S=${SCENES:-"-b ideas -b jellyfish -b gl41-geometry"}
mkdir -p $B/geomC
R=${1:-8}
rm -f $B/abg84.done
r=0
while [ $r -lt $R ]; do
  r=$((r+1))
  if [ $((r & 1)) -eq 1 ]; then ORDER="P C"; else ORDER="C P"; fi
  for side in $ORDER; do
    if [ "$side" = P ]; then LIB=$P; else LIB=$C; fi
    DYLD_INSERT_LIBRARIES=$LIB $G --data-path $D $S > $B/abg84_${side}$r.log 2>&1
  done
done
echo AB_DONE > $B/abg84.done

python3 - "$B" "$R" <<'PY'
import sys, re, statistics, collections
B, R = sys.argv[1], int(sys.argv[2])
for scene in ("ideas", "jellyfish", "gl41-geometry"):
    p, c = [], []
    for r in range(1, R + 1):
        for side, out in (("P", p), ("C", c)):
            try:
                txt = open(f"{B}/abg84_{side}{r}.log").read()
            except FileNotFoundError:
                continue
            fps = re.findall(r".*%s.*?FPS:\s*([0-9]+)" % scene, txt)
            if fps: out.append(int(fps[-1]))
    if p and c:
        print("%-16s P median %6.1f  C median %6.1f  d=%+.2f%%  (n=%d/%d)"
              % (scene, statistics.median(p), statistics.median(c),
                 (statistics.median(c) - statistics.median(p)) / statistics.median(p) * 100.0,
                 len(p), len(c)))
PY
