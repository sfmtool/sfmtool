# `sfm compare` Command

## Overview

Compares two `.sfmr` reconstructions, reporting differences in alignment, camera intrinsics,
image poses, and feature usage.

## Command Syntax

```bash
sfm compare <RECONSTRUCTION1> <RECONSTRUCTION2>
             [--by-coordinate | --by-feature-index] [--pixel-threshold PX]
```

`RECONSTRUCTION1` is the reference; `RECONSTRUCTION2` is compared against it.

## Comparison Steps

1. **Alignment** — Computes a similarity transform (rotation, translation, scale) aligning
   the second reconstruction to the first from the shared images' camera poses.
2. **Camera intrinsics** — Compares focal length, principal point, and distortion parameters
   for matching cameras.
3. **Image poses** — For images present in both reconstructions, reports positional and
   angular differences after alignment. Position errors are reported relative to the reference
   **scene scale**: the root-mean-square distance of the reference's finite 3D points from
   their centroid. This makes the metric independent of each reconstruction's arbitrary SfM
   gauge, so a scene reconstructed 100× larger reports the same percentages. Rotation errors
   are in degrees and need no scaling.
4. **Feature usage** — Compares which features were triangulated in each reconstruction for
   shared images that use the same `.sift` file.
5. **3D points** — Puts the two reconstructions' 3D points in correspondence and reports how
   many match / are unique to each side, plus the post-alignment distance distribution of
   the matched pairs (also reported relative to scene scale). Correspondences where either
   point is at infinity (homogeneous `w = 0`) are excluded from the distance statistics
   (their finite-vs-infinity distance is undefined) and reported separately. The overall
   relationship conclusion (`IDENTICAL` / `VERY SIMILAR` / `SIGNIFICANT DIFFERENCES`) uses
   scale-relative thresholds.

## Point correspondence method

The 3D-point comparison (step 5) can correspond points two ways:

- **By feature index** (`--by-feature-index`) — keys observations on their feature index,
  which requires both reconstructions to reference the **same** `.sift` files (e.g. two solves
  over one workspace). Fast and exact, but yields no correspondences across different feature
  backends.
- **By keypoint coordinate** (`--by-coordinate`) — matches observations in shared images by
  *mutual nearest 2D keypoint* within `--pixel-threshold` pixels (default 2.0) and votes
  point-to-point. This works **across different SIFT backends** (e.g. COLMAP SIFT vs the
  sfmtool extractor), where the same scene keypoint lands at the same pixel under a different
  feature index.

The default is **auto**: coordinate matching is used when every shared image uses a different
SIFT file, otherwise feature-index matching.

## Side-by-side patch strips (`--strips`)

`--strips OUT.png` renders points where the two solves differ as a side-by-side patch-strip
montage — one point per row, the reference solve's views on the left and the target solve's on
the right. Each tile is the point's oriented surfel projected into one observing view, so a
clean column means the solve placed a real, photoconsistent surface point there. The point
normals are refined for cross-view photoconsistency first (`--strips-no-refine` to skip).

The strip view always puts points in correspondence by keypoint coordinate (the robust choice
across SIFT backends), independent of the `--by-coordinate`/`--by-feature-index` choice used for
the numeric comparison above.

By default (`--strips-rank overview`) the montage shows a few points from each of several
categories, each under a labeled divider: least aligned, narrowest/widest triangulation angle,
most peripheral, largest NCC gap, lowest NCC, and points **unique** to each solve. A unique
point appears in one column with the other left blank. `--strips-rank` can instead focus on a
single axis (below).

Images and `.sift` files are resolved from each reconstruction's `workspace_dir` (as with the
photometric `xform` filters), so the workspaces must still hold them. Points at infinity
(`w = 0`) are skipped — they have no finite surfel to render.

Each row is annotated with the post-alignment distance (as % of scene scale), and per solve
the patch photoconsistency NCC and the point's triangulation angle (`a=…°`, small = depth
weakly constrained by parallax).

`--strips-rank` selects what to surface: either `overview` (default, the multi-category sampler
described above) or a single quantity, ranked toward one end (`--strips-end high|low`, default
the axis's natural end):

| `--strips-rank` | quantity | natural end (default) |
|---|---|---|
| `distance` | post-alignment distance between the two solves' placement | high (least aligned) |
| `view-angle` | triangulation angle | low (parallax-starved) |
| `ncc` | per-solve patch photoconsistency (min of the two solves) | low (suspicious surfel) |
| `ncc-gap` | gap between the two solves' photoconsistency | high (quality differs) |
| `image-radius` | keypoint distance from the principal point | high (peripheral) |
| `feature-size` | keypoint feature size in image pixels | high (coarse/blurry) |
| `world-size` | feature size projected to a metric surface footprint (size·depth/focal) | high (large patch) |

`ncc` and `ncc-gap` render every candidate to score it, so they are slower than the
geometry-only axes.

The two columns are named `reference`/`target` by default (the per-row `R`/`T` and the
`reference-only`/`target-only` labels follow); `--strips-labels LEFT,RIGHT` renames them (e.g.
`--strips-labels OLD,NEW`).

Other options: `--strips-num` (rows, default 16), `--strips-views` (cap tiles per strip,
default 8), `--strips-context N` (render wider NxN-pixel context tiles with a green box on the
validated extent; default 96, must exceed the 32px patch, 0 = off),
`--strips-refine/--strips-no-refine`.

## Usage Examples

```bash
# Compare two reconstructions
sfm compare sfmr/solve_001.sfmr sfmr/solve_002.sfmr

# Compare against ground truth
sfm compare ground_truth.sfmr sfmr/my_result.sfmr

# Force coordinate matching with a tighter threshold (e.g. comparing a COLMAP-SIFT
# solve against an sfmtool-SIFT solve of the same images)
sfm compare old.sfmr new.sfmr --by-coordinate --pixel-threshold 1.5

# Render the points the two solves place most differently as side-by-side patch
# strips, with wider scene context around each point
sfm compare old.sfmr new.sfmr --strips diff.png --strips-num 12 --strips-context 96
```
