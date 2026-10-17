#!/bin/bash
# Create one git worktree per domain for parallel porting work.
#   agents/tools/setup_worktrees.sh [base-branch] [domain...]
# Env: MGL_PROJECT_DIR (default: repo root), MGL_WORKTREE_ROOT (default: $MGL_PROJECT_DIR/wt)
set -euo pipefail
ROOT="${MGL_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
WT="${MGL_WORKTREE_ROOT:-$ROOT/wt}"
BASE="${1:-$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)}"; shift || true
DOMAINS=("$@")
[ ${#DOMAINS[@]} -gt 0 ] || DOMAINS=(fb queries debug introspect program xfb vattrib rstate texparam texdata texext buffers)

mkdir -p "$WT"
for d in "${DOMAINS[@]}"; do
    dir="$WT/$d"
    if [ -d "$dir" ]; then echo "exists: $d"; continue; fi
    git -C "$ROOT" worktree add -q -b "gl46/$d" "$dir" "$BASE"
    # share the prebuilt external deps instead of rebuilding them per worktree
    rm -rf "$dir/external" && ln -s "$ROOT/external" "$dir/external"
    echo "created: $d -> gl46/$d"
done
