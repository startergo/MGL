import sys, statistics, re
per = {}
for l in sys.stdin:
    m = re.match(r"round\d+\s+(\S+)\s+A=(\d*)\s+(\d*)\s+B=(\d*)\s+(\d*)$", l.strip())
    if not m:
        continue
    sc = m.group(1)
    a = [int(x) for x in m.group(2, 3) if x]
    b = [int(x) for x in m.group(4, 5) if x]
    per.setdefault(sc, [[], []])
    per[sc][0] += a
    per[sc][1] += b
for sc, (A, B) in per.items():
    ma, mb = statistics.median(A), statistics.median(B)
    print("%s: A %.0f  B %.0f  A/B %.4f (%+.2f%%)  n=%d/%d" %
          (sc, ma, mb, ma / mb, 100.0 * (ma / mb - 1), len(A), len(B)))
