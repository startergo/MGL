#!/bin/bash
# Build the MGL externals. Run from the external/ directory.
# Each external builds in its own subshell (cmake -S/-B), so one failure can no
# longer leave the script in the wrong directory for the next step (the old
# `cd ../..` chain made a later `make glfw` run where no glfw target exists).
# The script stops at the first failing step and names it.
set -eu
cd "$(dirname "$0")"
export SDKROOT="$(xcrun --show-sdk-path)"
# CMake 4 removed compatibility with cmake_minimum_required < 3.5 (SPIRV-Cross declares 3.0)
# Build type defaults to Release: with no build type CMake compiles without optimisation,
# and glslang / SPIRV-Cross run inside glLinkProgram, so an unoptimised build makes every
# link several times slower. MGL_EXT_BUILD_TYPE=Debug (or RelWithDebInfo) overrides it.
# An existing build dir is reconfigured with the new type on the next run.
CMAKE_FLAGS=(-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=${MGL_EXT_BUILD_TYPE:-Release})

step() { echo "== $*"; }
fail() { echo "build_external.sh: FAILED at: $1" >&2; exit 1; }

build() {   # build <dir> [make target]
    local d=$1 tgt=${2:-} b=$1/${BUILD_DIR:-build}   # BUILD_DIR: build into a fresh dir (default build)
    step "$d"
    ( cmake -S "$d" -B "$b" "${CMAKE_FLAGS[@]}" && make -C "$b" -j 4 $tgt ) || fail "$d"
}

cp ../MGL/include/MGLContext.h glfw/src
cp ../MGL/include/MGLRenderer.h glfw/src

build SPIRV-Tools
build SPIRV-Cross
build SPIRV-Headers
if [ -z "${SKIP_GLSLANG_UPDATE:-}" ]; then   # SKIP_GLSLANG_UPDATE=1: keep the fetched glslang deps as they are
    step "glslang sources"
    ( cd glslang && ./update_glslang_sources.py ) || fail "glslang/update_glslang_sources.py"
fi
build glslang
build glfw glfw
echo "== externals built"
