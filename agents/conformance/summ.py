#!/usr/bin/env python3
"""Short summary of CTS results.json: status totals, per-group table, failure
reason clusters. usage: python3 agents/conformance/summ.py [pattern] [--fail-pattern RE]"""
import json, os, re, sys
from collections import Counter, defaultdict

d = os.path.join(os.environ.get("MGL_CTS_RESULTS",
               os.path.expanduser("~/MGL-cts/mgl-results")))
res = json.load(open(os.path.join(d, "results.json")))["results"]
pat = None; fail_pat = None
args = sys.argv[1:]
if "--fail-pattern" in args:
    i = args.index("--fail-pattern"); fail_pat = re.compile(args[i+1]); args = args[:i]+args[i+2:]
if args: pat = re.compile(args[0])

if pat: res = {k: v for k, v in res.items() if pat.search(k)}
tot = Counter(v["status"] for v in res.values())
print("cases=%d  " % len(res) + "  ".join("%s=%d" % (k, v) for k, v in tot.most_common()))
grp = defaultdict(Counter)
for k, v in res.items():
    grp[".".join(k.split(".")[:2])][v["status"]] += 1
for g, c in sorted(grp.items(), key=lambda kv: -(sum(kv[1].values()) - kv[1].get("Pass", 0))):
    n = sum(c.values())
    print("%4d/%-4d %-42s %s" % (n - c.get("Pass", 0), n, g,
          " ".join("%s=%d" % (k, v) for k, v in c.most_common() if k != "Pass")))
if fail_pat is None:
    sys.exit(0)
# reason clusters among failures
rc = Counter()
for k, v in res.items():
    if v["status"] in ("Fail", "Crash", "Timeout", "InternalError") and fail_pat.search(k):
        r = re.sub(r"\s+", " ", v.get("reason", ""))[:110]
        rc[(v["status"], r)] += 1
for (st, r), n in rc.most_common(30):
    print("%4d %-8s %s" % (n, st, r))
