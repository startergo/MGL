#!/bin/bash
# helper: run a list of CTS cases (one glcts invocation each), print status per case
# usage: bash tests/cts_many.sh <casefile> [timeout-seconds]
# casefile: one full case name per line ('#' comments allowed)
set -u
LIST="$1"
T="${2:-60}"
rm -f /tmp/mgl_cts_modules
ln -sfn ~/MGL-cts/build/external/openglcts/modules /tmp/mgl_cts_modules || exit 2
LIB=$(cd "$(dirname "$0")/../build" && pwd)/libmgl.dylib
cd /tmp/mgl_cts_modules || exit 2
pass=0; fail=0; other=0
while IFS= read -r c; do
  case "$c" in ''|'#'*) continue;; esac
  out=/tmp/cts_many_$$
  MGL_LIBRARY="$LIB" perl -e 'alarm shift; exec @ARGV' "$T" ./glcts --deqp-case="$c" \
      --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
      --deqp-surface-width=256 --deqp-surface-height=256 \
      --deqp-log-filename="$out.qpa" --deqp-log-images=disable > "$out.out" 2> "$out.err"
  st=$(grep -ao 'StatusCode="[A-Za-z]*"' "$out.qpa" | tail -1 | sed 's/.*="//;s/"//')
  [ -z "$st" ] && st=NoResult
  case "$st" in
    Pass) pass=$((pass+1));;
    Fail) fail=$((fail+1));;
    *) other=$((other+1));;
  esac
  printf '%-8s %s\n' "$st" "$c"
  if [ "$st" != Pass ]; then
    grep -a -m1 -A2 '<Result StatusCode' "$out.qpa" | sed -e 's/^/    /' | sed -e 's/&apos;/'"'"'/g' | head -3
  fi
done < "$LIST"
echo "== pass=$pass fail=$fail other=$other"
