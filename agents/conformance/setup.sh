#!/bin/bash
# agents/conformance/setup.sh -- set up and build the Khronos GL CTS (VK-GL-CTS)
# with the MGL platform port against MGL's headless core-4.6 context.
#
#   MGL_CTS_DIR  where the CTS tree lives (default ~/MGL-cts)
#   MGL_LIB      path of libmgl.dylib to run against
#                (default <repo>/build/libmgl.dylib; override at run time
#                 with the MGL_LIBRARY env var instead)
#
# The CTS tree is NOT part of the repository; only the glue under
# agents/conformance/glue and this script are committed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"            # <repo>/agents/conformance
REPO="$(cd "$HERE/../.." && pwd)"
CTS_DIR="${MGL_CTS_DIR:-$HOME/MGL-cts}"
CTS_BRANCH="opengl-cts-4.6.1"
CTS_COMMIT="642964df5ee1319fe3330ebf54465f4c7367f3ab"   # GL CTS 4.6.1 release
BUILD_DIR="$CTS_DIR/build"
MGL_LIB="${MGL_LIB:-$REPO/build/libmgl.dylib}"

[ -f "$MGL_LIB" ] || { echo "missing $MGL_LIB -- run 'make lib' first" >&2; exit 2; }

# 1. clone (shallow, pinned to the GL CTS 4.6.1 release branch)
if [ ! -d "$CTS_DIR/.git" ]; then
    echo "== cloning VK-GL-CTS ($CTS_BRANCH) into $CTS_DIR"
    git clone --depth 1 --branch "$CTS_BRANCH" \
        https://github.com/KhronosGroup/VK-GL-CTS.git "$CTS_DIR"
fi
git -C "$CTS_DIR" rev-parse HEAD | grep -q "^$CTS_COMMIT" ||
    echo "note: $CTS_DIR is not at the pinned commit $CTS_COMMIT" >&2

# 2. external sources (glslang, spirv-tools/headers, amber, libpng, zlib)
if [ ! -d "$CTS_DIR/external/glslang/src" ]; then
    echo "== fetching external sources"
    (cd "$CTS_DIR" && python3 external/fetch_sources.py >/dev/null)
fi

# 3. install the MGL platform glue (target + framework/platform/mgl)
echo "== installing MGL platform glue"
mkdir -p "$CTS_DIR/targets/mgl"
cp "$HERE/glue/targets/mgl/mgl.cmake" "$CTS_DIR/targets/mgl/mgl.cmake"
mkdir -p "$CTS_DIR/framework/platform/mgl"
cp "$HERE/glue/framework/platform/mgl/"* "$CTS_DIR/framework/platform/mgl/"

# 4. configure
echo "== configuring (DEQP_TARGET=mgl)"
cmake -S "$CTS_DIR" -B "$BUILD_DIR" -G Ninja \
    -DDEQP_TARGET=mgl \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    "$@" > "$CTS_DIR/cmake-config.log" 2>&1 || {
        echo "cmake configure failed:" >&2; tail -30 "$CTS_DIR/cmake-config.log" >&2; exit 3; }

# 5. build the GL CTS binary
echo "== building glcts"
ninja -C "$BUILD_DIR" glcts > "$CTS_DIR/ninja.log" 2>&1 || {
    echo "ninja failed:" >&2; grep -E "error:" "$CTS_DIR/ninja.log" | head -30 >&2; exit 4; }

GLCTS="$BUILD_DIR/external/openglcts/modules/glcts"
[ -x "$GLCTS" ] || { echo "built glcts not found at $GLCTS" >&2; exit 5; }
echo "== ok: $GLCTS  (run it via $HERE/run.py)"
