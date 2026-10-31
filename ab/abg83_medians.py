#!/usr/bin/env python3
"""abg83_medians.py — per-scene FPS medians of the locked glmark2 A/B logs."""
import re, glob, statistics, sys, os
B = os.path.expanduser("~/MGL-cts/glmark2-bench")
pre = sys.argv[1] if len(sys.argv) > 1 else "abg83"
rows = {}
for side, sfx in (("P", "P"), ("C", "C")):
    for f in sorted(glob.glob(f"{B}/{pre}{sfx}*.log")):
        txt = open(f, errors="ignore").read()
        for scene, fps in re.findall(r"\[([^\]]+)\] <default>: FPS: ([0-9.]+)", txt):
            rows.setdefault(scene, {}).setdefault(side, []).append(float(fps))
print("%-16s %5s %10s %10s %9s %8s" % ("scene", "n", "P median", "C median", "delta", "verdict"))
for scene in sorted(rows):
    p, c = rows[scene].get("P", []), rows[scene].get("C", [])
    if not p or not c:
        continue
    pm, cm = statistics.median(p), statistics.median(c)
    d = (cm - pm) / pm * 100.0
    print("%-16s %2d+%2d %10.0f %10.0f %8.2f%% %8s" %
          (scene, len(p), len(c), pm, cm, d, "OK" if d >= -1.0 else "REGRESSION"))
    print("   P:", " ".join("%.0f" % v for v in sorted(p)))
    print("   C:", " ".join("%.0f" % v for v in sorted(c)))
