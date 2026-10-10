#!/bin/sh
# Sweep Track at Pixel's median gate (`finish.min_zncc_median`) with
# `core_cascade`, full pass, over every query of the seoul_bull and Kerry Park
# ground truths, one run directory per gate value. The gate is off at -1.
#
#   sh scripts/track_at_pixel/sweep_median_gate.sh <out dir> [shards]
#
# Run from the repository root. Then:
#
#   pixi run python scripts/track_at_pixel/summarize_sweep.py <out dir>/*
#
# A run directory that already holds rows.jsonl is skipped, so an interrupted
# sweep resumes. The cache directory (SIFT, index and cluster files) is
# <out dir>/cache.
set -e
OUT=${1:?usage: sweep_median_gate.sh <out dir> [shards]}
SHARDS=${2:-14}
KERRY=test-data/images/kerry_park/kerry_park_ground_truth.sfmr
for ds in seoul_bull kerry; do
  if [ "$ds" = seoul_bull ]; then D=seoul_bull; else D=$KERRY; fi
  for g in -1 0.6 0.65 0.7 0.75 0.8; do
    run="$OUT/${ds}_$g"
    [ -f "$run/rows.jsonl" ] && continue
    echo "$(date +%T) $ds gate $g"
    pixi run python scripts/track_at_pixel/run_sharded.py --dataset "$D" \
      --candidate core_cascade --passes full --shards "$SHARDS" \
      --cache-dir "$OUT/cache" \
      --opt "core_options={\"finish.min_zncc_median\": $g}" \
      --out "$run"
  done
done
