#!/bin/bash
# run the KHR-GL46 blend_equation_advanced CTS group against a given libmgl.dylib
rm -f /tmp/mgl_cts_modules
ln -sfn ~/MGL-cts/build/external/openglcts/modules /tmp/mgl_cts_modules
MGL_LIBRARY="$1" perl -e 'alarm shift; exec @ARGV' 1800 \
  /tmp/mgl_cts_modules/glcts \
  --deqp-case="KHR-GL46.blend_equation_advanced*" --deqp-surface-type=fbo \
  --deqp-gl-context-type=mgl --deqp-visibility=hidden \
  --deqp-surface-width=256 --deqp-surface-height=256 \
  --deqp-log-filename=/tmp/cts_adv.qpa > /tmp/cts_adv.out 2>/tmp/cts_adv.err
grep -o 'StatusCode="[A-Za-z]*"' /tmp/cts_adv.qpa | sort | uniq -c
