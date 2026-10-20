#!/usr/bin/env python3
"""Merge a cluster re-run results.json into the canonical CTS results.

    python3 agents/conformance/merge_results.py <cluster-results-dir> [<cluster-results-dir> ...]

The canonical results live in $MGL_CTS_RESULTS (default ~/MGL-cts/mgl-results); the per-group
summary.txt is rewritten afterwards.  Used by the cts agent to fold re-run groups back into
the checkpoint after a fix so later summaries reflect the current MGL state.
"""
import json
import os
import subprocess
import sys

def find_cts_dir():
    for c in [os.environ.get("MGL_CTS_DIR"), os.path.join(os.path.expanduser("~"), "MGL-cts")]:
        if c and os.path.isdir(os.path.join(c, "external", "openglcts")):
            return c
    sys.exit("cannot find the CTS tree")

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cts = find_cts_dir()
    results_dir = os.environ.get("MGL_CTS_RESULTS", os.path.join(cts, "mgl-results"))
    main_path = os.path.join(results_dir, "results.json")
    with open(main_path) as fh:
        results = json.load(fh)["results"]
    merged = 0
    for arg in sys.argv[1:]:
        with open(os.path.join(arg, "results.json")) as fh:
            other = json.load(fh)["results"]
        for case, r in other.items():
            results[case] = r
            merged += 1
    with open(main_path + ".tmp", "w") as fh:
        json.dump({"results": results}, fh, indent=0, sort_keys=True)
    os.replace(main_path + ".tmp", main_path)
    print("merged %d case results into %s" % (merged, main_path))
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "run.py"),
                    "--summarize-only"], check=False)

if __name__ == "__main__":
    main()
