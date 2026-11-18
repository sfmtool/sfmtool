# `sfm merge` Command

## Overview

Merges multiple aligned `.sfmr` reconstructions into a single unified reconstruction. Input
reconstructions should be pre-aligned to a common coordinate frame (e.g., via `sfm align`).

## Command Syntax

```bash
sfm merge <RECONSTRUCTION1.sfmr> <RECONSTRUCTION2.sfmr> [MORE...] --output <OUTPUT.sfmr> [OPTIONS...]
```

## Inputs

The merge matches and combines observations by feature index and refines the
poses from the `.sift` files, so it takes only reconstructions that carry
feature indexes into one set of `.sift` files. It checks every input once
they are loaded and before it prints or runs any step of the merge, and stops
with an error naming the inputs, writing nothing, when:

- there are fewer than two inputs;
- an input is an `embedded_patches` reconstruction, which stores each
  observation's keypoint in the file and has no feature indexes. Every input
  must be `sift_files`. No command turns an `embedded_patches` reconstruction
  back into a `sift_files` one, so the error says to merge the `sift_files`
  reconstructions it was made from, if there are any, and convert the result
  with `sfm xform --to-embedded-patches`;
- the inputs record different feature folders (`feature_prefix_dir` in their
  workspace metadata), such as one solved before `sfm ws init --force` and
  one after; the error names each input's folder. The merged reconstruction
  records the first input's folder;
- two cameras that took the same image vary different parameters per image
  (their `per_image_adjustments`, see
  [sfmr-file-format.md](../../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13))
  or have different `rolling_shutter_scanlines` settings, since step 1 below
  makes them one camera; the error names the image and both cameras;
- some inputs have rig frames and others do not, or an input's rig frames do
  not fit the reference's rigs. The reference is the first input with rigs:
  the merged reconstruction keeps its rigs, sensors and sensor cameras, and
  each image keeps its frame's rig index and its sensor index. So every rig an
  input's images use must exist in the reference with the same sensor count
  and sensor offset, and each image's camera must merge (step 1) into the
  same camera as the reference's camera for its sensor. Which sensor is a
  rig's reference sensor may differ: an image's sensor index counts over all
  sensors whatever the reference sensor, which only sets how the sensor poses
  are expressed, and the merge takes those from the reference while each
  image keeps its own pose. An input
  may define fewer rigs than the reference, or rigs no image of it uses: a
  solve of rig frames and single images, which adds a single-sensor rig for
  the single images after the `.camrig` rig, merges with a solve of the rig
  frames alone. The cameras of step 1 are merged with the checks, before any
  step prints;
- the first input's workspace is version 1 (the `sfm ws upgrade` error),
  since the poses are refined from the `.sift` files there.

Points at infinity are accepted (step 5 says how they merge). The checks live
in `_check_inputs` in
[merge/reconstructions.py](../../../src/sfmtool/merge/reconstructions.py),
which `merge_reconstructions` runs first; called from Python, it raises
`ValueError` (`Version1WorkspaceError` for the workspace) with the same
message, naming each input by number and, when `input_names` is given, by
name.

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output / -o` | path | required | Output `.sfmr` file path |
| `--merge-percentile` | float | 95.0 | Percentile of correspondence distances for merge threshold |

## Merge Process

1. **Deduplicate cameras** — Merge camera intrinsics across reconstructions:
   two cameras that took the same image become one, which keeps the first
   input's intrinsics. The input checks above have made sure such cameras
   vary the same parameters per image and have the same
   `rolling_shutter_scanlines` setting, since the merged camera's
   rolling-shutter rates are read on it. The merged camera varies what they
   vary and has their setting.
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
   its adjustment applied. The keypoints come from the `.sift` files in the
   feature folder the inputs record, not the workspace's current one. Points at infinity are left out of PnP, which takes
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
