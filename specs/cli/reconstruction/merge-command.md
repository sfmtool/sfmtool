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
   [sfmr-file-format.md](../../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13));
   otherwise the merge stops with an error naming the image and both cameras.
   The merged camera varies what they vary.
2. **Merge images** — Combine image sets, handling duplicates. An image in
   several inputs takes its pose and its intrinsic adjustment from the first.
3. **Find correspondences** — Match 3D points across reconstructions via shared feature
   observations using union-find.
4. **Filter by percentile** — Reject point correspondences with distances above the
   percentile-based threshold.
5. **Merge points and tracks** — Average corresponding point positions, combine observation
   lists.
6. **Refine poses** — Run parallel PnP+RANSAC to refine camera poses in the merged
   reconstruction, each image through its own intrinsics: its merged camera with
   its adjustment applied.
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
