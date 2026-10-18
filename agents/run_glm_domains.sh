#!/bin/bash
# Launch one GLM worker per domain in the background (nohup), each in its own worktree.
#   agents/run_glm_domains.sh introspect texdata tess geom      # start (or --resume if a transcript exists)
#   agents/run_glm_domains.sh [--status]                        # what each worker is doing
#   agents/run_glm_domains.sh --stop                            # stop all workers
# Config: agents/.env (git-ignored) or your shell env — see agents/.env.example.
set -uo pipefail
ROOT="${MGL_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
WT="${MGL_WORKTREE_ROOT:-$ROOT/wt}"
cd "$ROOT"
[ -f agents/.env ] && { set -a; . agents/.env; set +a; }

status() {
    for pidf in "$WT"/*/.agent/worker.pid; do
        [ -f "$pidf" ] || continue
        d="$(basename "$(dirname "$(dirname "$pidf")")")"; pid="$(cat "$pidf")"
        if kill -0 "$pid" 2>/dev/null; then st="running"; else st="stopped"; fi
        [ -f "$WT/$d/agent_report.md" ] && st="$st, report ready"
        echo "== $d ($st) — $(git -C "$WT/$d" log --oneline -1 | cut -c1-70)"
        tail -n 3 "$WT/$d/agent.log" 2>/dev/null | cut -c1-200 | sed 's/^/   /'
    done
}

case "${1:-}" in
    --status) status; exit 0 ;;
    --stop)
        for pidf in "$WT"/*/.agent/worker.pid; do
            [ -f "$pidf" ] && kill "$(cat "$pidf")" 2>/dev/null && echo "stopped $(basename "$(dirname "$(dirname "$pidf")")")"
        done; exit 0 ;;
    "") status; echo; echo "usage: $0 <domain>... (start/resume) | --status | --stop"; exit 0 ;;
    -h|--help) echo "usage: $0 <domain>... (start/resume) | --status | --stop"; exit 0 ;;
esac

# pick a Python that has the anthropic SDK (Homebrew upgrades can move python3 to a fresh version)
PY="${AGENT_PYTHON:-}"
if [ -z "$PY" ]; then
    for name in python3 python3.14 python3.13 python3.12 python3.15; do
        for c in $(which -a "$name" 2>/dev/null); do
            "$c" -c 'import anthropic' 2>/dev/null && { PY="$c"; break 2; }
        done
    done
fi
[ -n "$PY" ] || { echo "No Python with the anthropic SDK: pip install anthropic (or set AGENT_PYTHON)"; exit 1; }

if [ "${AGENT_PROVIDER:-glm}" = "glm" ] && [ -z "${ANTHROPIC_AUTH_TOKEN:-}${ANTHROPIC_API_KEY:-}" ]; then
    echo "No Z.AI key: put ANTHROPIC_AUTH_TOKEN=<key> in agents/.env (git-ignored) or export it."; exit 1
fi

for d in "$@"; do
    [ -d "$WT/$d/.git" ] || [ -f "$WT/$d/.git" ] || { echo "skip $d: no worktree (agents/tools/setup_worktrees.sh)"; continue; }
    [ -f "agents/tasks/$d.md" ] || { echo "skip $d: no agents/tasks/$d.md"; continue; }
    mkdir -p "$WT/$d/.agent"
    if [ -f "$WT/$d/.agent/worker.pid" ] && kill -0 "$(cat "$WT/$d/.agent/worker.pid")" 2>/dev/null; then
        echo "$d already running"; continue
    fi
    resume=""; [ -f "$WT/$d/.agent/transcript.json" ] && resume="--resume"
    nohup "$PY" agents/glm_worker.py --domain "$d" $resume > "$WT/$d/.agent/worker.out" 2>&1 &
    echo $! > "$WT/$d/.agent/worker.pid"
    echo "started $d (pid $!) $resume"
done
