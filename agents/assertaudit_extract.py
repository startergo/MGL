#!/usr/bin/env python3
"""Extract removed assert(0)/assert(false) sites from git diff and classify contexts."""
import subprocess, re, sys, json

BASE = "792cbab"
HEAD = "HEAD"

diff = subprocess.run(["git", "diff", BASE, HEAD, "--", "MGL/src"],
                      capture_output=True, text=True, check=True).stdout

sites = []
cur_file = None
hunk = None

def flush(hunk):
    if hunk is None:
        return
    removed = [l[1:] for l in hunk["lines"] if l.startswith("-") and not l.startswith("---")]
    added = [l[1:] for l in hunk["lines"] if l.startswith("+") and not l.startswith("+++")]
    ctx_before = [l[1:] for l in hunk["lines"] if l.startswith(" ") ]
    if any(re.search(r'\bassert\s*\(\s*(0|false)\s*\)', l) for l in removed):
        sites.append({
            "file": hunk["file"],
            "new_start": hunk["new_start"],
            "removed": removed,
            "added": added,
            "ctx_before": ctx_before,
        })

for line in diff.split("\n"):
    if line.startswith("diff --git"):
        flush(hunk); hunk = None
        m = re.match(r'diff --git a/(.*) b/(.*)', line)
        cur_file = m.group(2)
    elif line.startswith("@@"):
        flush(hunk)
        m = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
        hunk = {"file": cur_file, "new_start": int(m.group(1)), "lines": []}
    elif hunk is not None:
        if line.startswith("+") or line.startswith("-") or line.startswith(" "):
            hunk["lines"].append(line)
flush(hunk)

print(f"total sites: {len(sites)}", file=sys.stderr)
for s in sites:
    print("=" * 100)
    print(f"{s['file']}:{s['new_start']}")
    for l in s["ctx_before"][-6:]:
        print(f"  | {l}")
    for l in s["removed"]:
        print(f"  - {l}")
    for l in s["added"]:
        print(f"  + {l}")
