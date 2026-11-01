#!/bin/bash
# cts_piq88.sh — dEQP-GL45-ES31 program_interface_query subgroups, one glcts
# invocation per subgroup so a crash cannot eat the group.  Writes .qpa + .fails
# into OUTDIR (default /tmp/piq88).
#   usage: bash ab/cts_piq88.sh <subgroup> [<subgroup>...]     (default: all)
# Subgroups: program_input program_output buffer_variable uniform
#            transform_feedback_varying shader_storage_block uniform_block
#            atomic_counter_buffer buffer_limited_query
set -u
LIB=${MGL_LIBRARY:-$(pwd)/libmgl.dylib}
OUTDIR=${OUTDIR:-/tmp/piq88}
G=${MGL_CTS_MODULES:-$HOME/MGL-cts-new/build/external/openglcts/modules}
mkdir -p "$OUTDIR"
ln -sfn $HOME/MGL-cts-new/src/external/openglcts/data /tmp/mgl_cts_data 2>/dev/null
SUBGROUPS=${*:-"program_input program_output buffer_variable uniform transform_feedback_varying shader_storage_block uniform_block atomic_counter_buffer buffer_limited_query"}
for G_ in $SUBGROUPS; do
  MGL_LIBRARY="$LIB" MGL_NO_GLSL_CACHE=1 perl -e 'alarm shift; exec @ARGV' 2400 \
    "$G/glcts" \
    --deqp-archive-dir=/tmp/mgl_cts_data \
    --deqp-case="dEQP-GL45-ES31.functional.program_interface_query.$G_.*" \
    --deqp-surface-type=fbo --deqp-gl-context-type=mgl --deqp-visibility=hidden \
    --deqp-surface-width=256 --deqp-surface-height=256 \
    --deqp-log-images=disable --deqp-log-shader-sources=disable \
    --deqp-log-decompiled-spirv=disable --deqp-log-empty-loginfo=disable \
    --deqp-log-filename="$OUTDIR/$G_.qpa" > "$OUTDIR/$G_.out" 2> "$OUTDIR/$G_.err"
  rc=$?
  awk '/^#beginTestCaseResult /{c=$2} /^ <Result StatusCode=/{s=$2; gsub(/StatusCode=|"|>/,"",s); if (s!="Pass") print s, c}' "$OUTDIR/$G_.qpa" \
    | sed 's/^/  /' > "$OUTDIR/$G_.fails"
  p=$(grep -c '^#beginTestCaseResult' "$OUTDIR/$G_.qpa")
  f=$(grep -c '^  Fail ' "$OUTDIR/$G_.fails" || true)
  ns=$(grep -c '^  NotSupported ' "$OUTDIR/$G_.fails" || true)
  cr=$(grep -c '^  Crash ' "$OUTDIR/$G_.fails" || true)
  printf '%-28s n=%-5d Pass=%-5d Fail=%-4d NS=%-3d Crash=%d rc=%d\n' \
    "$G_" "$p" "$((p-f-ns-cr))" "$f" "$ns" "$cr" "$rc"
done
