#!/bin/bash
# One-shot at-scale profiling of the doraemon JAXTPC dataset (run_0026628550,
# 100 shards / 20k events). Runs the three profilers sequentially so they
# don't contend for FS/CPU and skew each other's scaling numbers.
set -uo pipefail
cd "$(dirname "$0")"

# Was a hardcoded path under one developer's home that no longer exists.
OUT=${PROFILE_OUT:-${SCRATCH:-/tmp}/jaxtpc_profiling}
mkdir -p "$OUT"

echo "############ profile_scaling (FS vs IPC ceiling) ############"
python3 profile_scaling.py --dataset jaxtpc \
    --n-events-per-worker 100 \
    --worker-counts 1 2 4 8 16 24 \
    > "$OUT/profile_scaling.log" 2>&1
echo "  -> $OUT/profile_scaling.log (exit $?)"

echo "############ profile_loader (per-stage breakdown) ############"
python3 profile_loader.py --dataset jaxtpc \
    --n-events 200 --warmup 20 \
    > "$OUT/profile_loader.log" 2>&1
echo "  -> $OUT/profile_loader.log (exit $?)"

echo "############ benchmark_loader (workers x batch grid) ############"
python3 benchmark_loader.py --dataset jaxtpc \
    --workers 0 2 4 8 16 24 \
    --batches 1 8 32 \
    --n-warmup 3 --n-timed 20 \
    --prefetch-factor 1 \
    --prefetch-sweep \
    --csv "$OUT/benchmark_loader.csv" \
    --plot "$OUT/jaxtpc" \
    > "$OUT/benchmark_loader.log" 2>&1
echo "  -> $OUT/benchmark_loader.log (exit $?)"

echo "ALL DONE"
