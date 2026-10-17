#!/bin/bash
# Build and run the headless MGL tests against build/libmgl.dylib.
#   tests/headless/run.sh              # all tests
#   tests/headless/run.sh test_queries # one test (file stem)
# Exit status is non-zero if any test fails or crashes.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/build/tests"
mkdir -p "$OUT"
LIB="$ROOT/build/libmgl.dylib"
[ -f "$LIB" ] || { echo "missing $LIB — run 'make lib' first"; exit 2; }

if [ $# -gt 0 ]; then
    SRCS=(); for t in "$@"; do SRCS+=("$ROOT/tests/headless/${t%.c}.c"); done
else
    SRCS=("$ROOT"/tests/headless/test_*.c)
fi

fail=0
for src in "${SRCS[@]}"; do
    name="$(basename "${src%.c}")"
    exe="$OUT/$name"
    if ! cc -std=gnu11 -O0 -g -Wall -Wno-unused-function -I"$ROOT/tests/headless" \
            -I"$ROOT/MGL/include/GL" -I"$ROOT/MGL/include" "$src" -o "$exe" \
            -L"$ROOT/build" -lmgl -Wl,-rpath,"$ROOT/build" -lm; then
        echo "BUILD-FAIL $name"; fail=1; continue
    fi
    echo "--- $name"
    # MGL logs heavily to stderr; keep only test diagnostics ("  [") visible.
    # hard per-test timeout (MGL_TEST_CASE_TIMEOUT seconds, default 120) so a GPU hang can't wedge the shell
    perl -e 'alarm shift; exec @ARGV' "${MGL_TEST_CASE_TIMEOUT:-120}" "$exe" 2> >(grep -E '^\s+\[|Assertion|assert|fault|abort' >&2)
    rc=$?
    if [ $rc -ne 0 ]; then echo "FAILED $name (rc=$rc)"; fail=1; fi
done
exit $fail
