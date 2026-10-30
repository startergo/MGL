#!/bin/bash
# s33: opportunistic verdict chunks — WAIT for a load window (1-min load <= WAITMAX,
# checked every 15 s for up to WAITSEC s), then run ROUNDS ABBA rounds of the
# supervisor's one-process method (run_ab_s31.sh, append mode).
#   usage: WAITMAX=6 WAITSEC=600 ROUNDS=2 LOG=/tmp/ab_s33_log.txt bash ab/run_ab_s31_q.sh
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
WAITMAX=${WAITMAX:-6}
WAITSEC=${WAITSEC:-600}
t0=$(date +%s)
while :; do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  now=$(date +%s)
  if (( $(echo "$l <= $WAITMAX" | bc -l) )); then
    echo "window: load=$l after $((now - t0))s"
    break
  fi
  if (( now - t0 >= WAITSEC )); then
    echo "no window (load=$l) — running anyway after ${WAITSEC}s"
    break
  fi
  sleep 15
done
AB_APPEND=1 ROUNDS=${ROUNDS:-2} LOG=${LOG:-/tmp/ab_s33_log.txt} bash "$DIR/run_ab_s31.sh"
