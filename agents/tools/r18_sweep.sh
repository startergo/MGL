#!/bin/bash
# round-18 full KHR-GL46 mustpass sweep (one glcts at a time, sequential batches)
set -u
cd "$(dirname "$0")/../.."
export MGL_CTS_RESULTS="$HOME/MGL-cts/mgl-r18-sweep"
mkdir -p "$MGL_CTS_RESULTS"
ulimit -c 0          # no core dumps from a crashing glcts
exec nice -n 10 taskpolicy -b python3 agents/conformance/run.py \
    --pattern 'KHR-GL46.*' --batch 64 --per-case-timeout 20 --resume \
    --results-dir "$MGL_CTS_RESULTS" >> "$MGL_CTS_RESULTS/sweep.log" 2>&1
