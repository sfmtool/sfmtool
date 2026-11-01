# `sfm merge` Command

## Overview

Merges multiple aligned `.sfmr` reconstructions into a single unified reconstruction. Input
reconstructions should be pre-aligned to a common coordinate frame (e.g., via `sfm align`).

## Command Syntax

```bash
sfm merge <RECONSTRUCTION1.sfmr> <RECONSTRUCTION2.sfmr> [MORE...] --output <OUTPUT.sfmr> [OPTIONS...]
```

At least two reconstructions are required.

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output / -o` | path | required | Output `.sfmr` file path |
| `--merge-percentile` | float | 95.0 | Percentile of correspondence distances for merge threshold |

## Merge Process

1. **Deduplicate cameras** — Merge camera intrinsics across reconstructions:
   two cameras that took the same image become one, which keeps the first
   input's intrinsics. They become one only when they vary the same parameters
   per image (their `per_image_adjustments`, see
   [sfmr-file-format.md](../../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13))
   and have the same `rolling_shutter_scanlines` setting, since the merged
   camera's rolling-shutter rates are read on it; otherwise the merge stops
   with an error naming the image and both cameras. The merged camera varies
   what they vary and has their setting.
2. **Merge images** — Combine image sets, handling duplicates. An image in
   several inputs takes its pose and its intrinsic adjustment from the first:
   its focal scale, its principal-point offset, its radial correction and its
   rolling-shutter rates.
3. **Find correspondences** — Match 3D points across reconstructions via shared feature
   observations, then group the matched pairs transitively: each group is a
   connected component of the graph whose edges are the pairs.
4. **Filter by percentile** — Reject point correspondences with distances above the
   percentile-based threshold.
5. **Merge points and tracks** — Average corresponding point positions, combine observation
   lists. Points at infinity are kept: a merged point is finite when any
   point merged into it is finite, and at infinity otherwise
   ([point-correspondence.md](../../core/reconstruction/point-correspondence.md#merging-points-and-tracks-merge_points_and_tracks)
   gives the rule).
6. **Refine poses** — Run PnP+RANSAC on a pool of threads, one per CPU, to refine
   camera poses in the merged reconstruction, each image through its own intrinsics: its merged camera with
   its adjustment applied. Points at infinity are left out of PnP, which takes
   Euclidean positions only. pycolmap has no model for an `SFMTOOL_PINHOLE`
   camera, the only one whose images carry a radial correction or rates, so
   such an image's refinement fails and it keeps its pose, as any image whose
   refinement fails does.
7. **Combine tracks** — Build the final unified track data. Where an input
   carries [observation readings](../../formats/sfmr-file-format.md#observation-readings-optional-version-12),
   each merged observation takes the row of the input observation with the
   same image name and feature index, since the merge renders nothing; an
   observation no input read, or one from an input whose readings stand under
   other options than the first input's that has them, is not measured.
8. **Score the points** — Each point's RMS reprojection error is computed
   over its merged track at the refined poses (`recompute_point_errors`),
   replacing the average of the merged points' stored errors that step 5
   carries.

## Usage Examples

```bash
# Merge two aligned reconstructions
sfm merge aligned/part_a.sfmr aligned/part_b.sfmr -o merged.sfmr

# Stricter correspondence threshold
sfm merge aligned/a.sfmr aligned/b.sfmr aligned/c.sfmr -o merged.sfmr --merge-percentile 80
```
