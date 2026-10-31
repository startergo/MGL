#!/bin/bash
# helper: run several KHR-GL46 groups sequentially, one qpa per group, print per-group summary
# usage: bash agents/cts_audit.sh <outprefix> <timeout-per-group> <group1> [group2 ...]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PFX="$1"; T="$2"; shift 2
rm -f /tmp/mgl_cts_modules
ln -sfn ~/MGL-cts/build/external/openglcts/modules /tmp/mgl_cts_modules || exit 2
LIB=$(cd "$ROOT/build" && pwd)/libmgl.dylib
mkdir -p /tmp/cts_s35
cd /tmp/mgl_cts_modules || exit 2
for G in "$@"; do
  OUT="/tmp/cts_s35/${PFX}_${G}"
  MGL_LIBRARY="$LIB" perl -e 'alarm shift; exec @ARGV' "$T" ./glcts --deqp-case="KHR-GL46.${G}*" \
    --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
    --deqp-surface-width=256 --deqp-surface-height=256 \
    --deqp-log-filename="$OUT.qpa" --deqp-log-images=disable > "$OUT.out" 2> "$OUT.err"
  rc=$?
  tot=$(grep -c "^TEST: KHR-GL46.${G}" /tmp/khr46_cases.txt)
  echo "== $G (rc=$rc, caselist=$tot)"
  "$ROOT/agents/cts_qpa_summary.py" "$OUT.qpa" 2>/dev/null | sed 's/^/   /'
  tail -1 "$OUT.err" 2>/dev/null | cut -c1-120 | sed 's/^/   err: /'
done
