#!/bin/bash
# Merge every gl46/<domain> branch into the current branch, then build and test.
#   agents/tools/merge_domains.sh [domain...]
# Stops at the first conflict so it can be resolved by hand (then re-run).
set -uo pipefail
ROOT="${MGL_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$ROOT"
DOMAINS=("$@")
[ ${#DOMAINS[@]} -gt 0 ] || DOMAINS=(debug buffers fb rstate vattrib texdata texparam texext program introspect xfb queries)
for d in "${DOMAINS[@]}"; do
    b="gl46/$d"
    git rev-parse -q --verify "$b" >/dev/null || { echo "skip $d (no branch)"; continue; }
    if git merge-base --is-ancestor "$b" HEAD; then echo "already merged: $d"; continue; fi
    echo "=== merging $b"
    if ! git merge --no-ff -m "Merge $b" "$b"; then
        echo "CONFLICT merging $b — resolve, commit, re-run"; exit 1
    fi
done
make -j"${MGL_MAKE_JOBS:-8}" lib >/dev/null 2>"${TMPDIR:-/tmp}/mgl-merge-build.log" || {
    grep -E 'error' "${TMPDIR:-/tmp}/mgl-merge-build.log" | head -40; exit 1; }
python3 agents/stub_scanner.py
tests/headless/run.sh
