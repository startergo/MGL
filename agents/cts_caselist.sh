#!/bin/bash
# helper: export the KHR-GL46 case list (group names + counts) without running tests
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rm -f /tmp/mgl_cts_modules
ln -sfn ~/MGL-cts/build/external/openglcts/modules /tmp/mgl_cts_modules || exit 2
LIB=$(cd "$ROOT/build" && pwd)/libmgl.dylib
cd /tmp/mgl_cts_modules || exit 2
MGL_LIBRARY="$LIB" perl -e 'alarm shift; exec @ARGV' 600 ./glcts \
  --deqp-case='KHR-GL46.*' --deqp-runmode=txt-caselist \
  --deqp-caselist-export-file=/tmp/khr46_cases.txt \
  --deqp-gl-context-type=mgl > /tmp/khr46_caselist.out 2>&1
rc=$?
echo "rc=$rc lines=$(wc -l < /tmp/khr46_cases.txt)"
awk -F. '{print $2}' /tmp/khr46_cases.txt | sort | uniq -c | sort -rn
