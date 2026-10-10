#!/bin/sh
# Measure Add Image to Tracks' bars one at a time around the default (pooled
# bar k = 2, track bar 0.9, pair rule 0.9, floor 0.5) on the seoul_bull and
# kerry_park ground truths, every image, at the resected and the ground-truth
# pose.
#
#   sh scripts/add_image_to_tracks/sweep_bars.sh <out dir>
#
# Run from the repository root. Writes <out dir>/<dataset>.jsonl and prints
# the tables (summarize.py). The cache directory is <out dir>/cache.
set -e
OUT=${1:?usage: sweep_bars.sh <out dir>}
STRATEGIES=default,default_pooled_k3,default_pooled_k4,default_track0.85,default_track0.95,default_pair0.8,default_pair1.0,default_floor0.4,default_floor0.6
for ds in seoul_bull kerry_park; do
  echo "$(date +%T) $ds"
  pixi run python scripts/add_image_to_tracks/harness.py --dataset "$ds" \
    --cache "$OUT/cache" --out "$OUT" --measurements default --strategies "$STRATEGIES"
done
pixi run python scripts/add_image_to_tracks/summarize.py "$OUT/seoul_bull.jsonl" "$OUT/kerry_park.jsonl"
