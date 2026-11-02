#!/bin/bash
# cts_lst90.sh <caselist-file> — run the cases of an arbitrary caselist file under
# the KHR-Single config (64x64 fbo, seed 1).  Slicing handled by the caller.
set -u
LIB=${MGL_LIBRARY:-$(pwd)/libmgl.dylib}
OUT=$1; LST=$2
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
mkdir -p "$(dirname "$OUT")"
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 nice -n 10 perl -e 'alarm shift; exec @ARGV' 5400 \
  "$G/glcts" \
  --deqp-archive-dir=/tmp/mgl_cts_data \
  --deqp-caselist-file="$LST" \
  --deqp-gl-context-type=mgl --deqp-surface-type=fbo --deqp-visibility=hidden \
  --deqp-surface-width=64 --deqp-surface-height=64 --deqp-base-seed=1 \
  --deqp-watchdog=disable --deqp-log-images=disable --deqp-log-shader-sources=disable \
  --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
  --deqp-log-filename="$OUT.qpa" > "$OUT.out" 2> "$OUT.err"
rc=$?
awk '/^#beginTestCaseResult /{c=$2} /^ <Result StatusCode=/{s=$2; gsub(/StatusCode=|"|>/,"",s); if (s!="Pass") print s, c}' "$OUT.qpa" \
  | sed 's/^/  /' > "$OUT.fails"
p=$(grep -c '^#beginTestCaseResult' "$OUT.qpa")
f=$(grep -c '^  Fail' "$OUT.fails" || true)
ns=$(grep -c '^  NotSupported ' "$OUT.fails" || true)
cr=$(grep -c '^  Crash ' "$OUT.fails" || true)
printf '%-30s n=%-5d Pass=%-5d Fail=%-4d NS=%-3d Crash=%d rc=%d\n' \
  "$(basename $OUT)" "$p" "$((p-f-ns-cr))" "$f" "$ns" "$cr" "$rc"
