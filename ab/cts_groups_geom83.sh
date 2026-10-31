#!/bin/bash
# cts_groups_geom83.sh — the eight geom-domain CTS groups (the r82 audit set) against
# one libmgl.dylib.  Prints the per-group Pass/Fail/NotSupported tally.
# Usage: ab/cts_groups_geom83.sh <libmgl.dylib> [group ...]
set -u
LIB=${1:?libmgl.dylib}
shift || true
CTSGROUPS=${*:-"shader_image_size shader_image_load_store shader_texture_image_samples_tests texture_view geometry_shader sample_variables sample_shading shader_multisample_interpolation"}
rm -f /tmp/mgl_cts_modules
ln -sfn ~/MGL-cts/build/external/openglcts/modules /tmp/mgl_cts_modules
for G in $CTSGROUPS; do
  case $G in
    sample_variables) DEQP=sample_variables ;;
    *) DEQP=$G ;;
  esac
  MGL_LIBRARY="$LIB" perl -e 'alarm shift; exec @ARGV' 1800 \
    /tmp/mgl_cts_modules/glcts \
    --deqp-case="KHR-GL46.$DEQP*" --deqp-surface-type=fbo \
    --deqp-gl-context-type=mgl --deqp-visibility=hidden \
    --deqp-surface-width=256 --deqp-surface-height=256 \
    --deqp-log-filename=/tmp/cts_g83.qpa > /tmp/cts_g83.out 2>/tmp/cts_g83.err
  printf '%-34s ' "$G"
  grep -o 'StatusCode="[A-Za-z]*"' /tmp/cts_g83.qpa | sort | uniq -c | tr -d '\n' | sed 's/StatusCode=//g'
  echo
done
