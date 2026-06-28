# `sfm match` Command

## Overview

Matches SIFT features between image pairs and writes a `.matches` file. Requires a workspace
with previously extracted SIFT features. Uses COLMAP to perform the matching, except for the
experimental "flow" mode, which has some promise for videos.

## Command Syntax

```bash
sfm match [PATHS...] --exhaustive | --sequential | --flow | --cluster [OPTIONS...]
sfm match --merge FILE1.matches FILE2.matches ... -o OUTPUT.matches
```

`PATHS` are image directories or files; when omitted, the current directory is used
(except with `--merge`, which requires explicit `.matches` paths). Exactly one matching
method must be specified, or `--merge` to combine existing `.matches` files.

## Matching Methods

| Method | Description |
|--------|-------------|
| `--exhaustive / -e` | Match every pair of images against every other |
| `--sequential / -s` | Match each image against its nearby neighbors in sequence order |
| `--flow` | Use dense optical flow to guide feature matching |
| `--cluster` | Cluster all images' descriptors at once (background-floor track-cluster matching) |
| `--merge` | Merge multiple `.matches` files into one |

Exactly one matching method must be given. Each method has its own tuning
options (`--sequential-overlap` for `--sequential`; `--flow-preset` /
`--flow-skip` for `--flow`; `--cluster-alpha` / `--cluster-d` /
`--cluster-preset` for `--cluster`). Passing a method-specific option without
its companion method is rejected with a `UsageError` rather than silently
ignored.

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sequential-overlap` | int | 10 | Number of overlapping neighbors for sequential matching |
| `--flow-preset` | `fast` \| `default` \| `high_quality` | `default` | Optical flow quality preset |
| `--flow-skip` | int | 5 | Sliding window size for flow matching |
| `--cluster-alpha` | float | 0.8 | Background-floor radius multiplier for cluster matching |
| `--cluster-d` | int | 10 | Background rank: the d-th-nearest distance sets the floor for cluster matching |
| `--cluster-preset` | `accurate` \| `balanced` \| `fast` | `accurate` | Kd-tree forest preset for cluster matching |
| `--max-features` | int | | Maximum features per image |
| `--output / -o` | path | auto | Output `.matches` file path (default: timestamped, required for `--merge`) |
| `--range / -r` | string | | Range expression for file numbers |
| `--camera-model` | choice | auto | Camera model override (e.g., `SIMPLE_RADIAL`, `OPENCV`). Accepts the same 11 COLMAP model names as `solve` and `camrig create`. |

## Process

1. Loads workspace config and SIFT features
2. Populates a COLMAP database with images, cameras, keypoints, and descriptors
3. Runs the selected matching strategy
4. Computes descriptor distances for matched pairs
5. Writes a timestamped `.matches` file

## Camera Intrinsics

If any image being processed resolves a `camera_config.json` (closest-ancestor walk from
its parent directory up to the workspace root), the file's intrinsics are used and
`--camera-model` is rejected with an error. See
[`../workspace/camera-config.md`](../workspace/camera-config.md).

## Cluster Matching

`--cluster` uses the background-floor track-cluster matcher: instead of
enumerating image pairs, it concatenates every image's descriptors into one
corpus, queries each descriptor's nearest neighbours over a randomized kd-tree
forest, and keeps the cross-image neighbours within `--cluster-alpha` × its
`--cluster-d`-th-nearest distance (its *background floor*). Those candidates
are materialized into track clusters and then expanded into per-image-pair
matches, which are geometrically verified — so the output `.matches` carries
two-view geometry and is written under `tvg-matches/`. Image pair selection
falls out of the clustering: only pairs that share a cluster are verified.

The clustering itself uses no intrinsics or poses. `--camera-model` is still
accepted with `--cluster` because it feeds the geometric verification step
(as it does for the other matchers): the chosen model is written into the
COLMAP database and used to estimate each clustered pair's two-view geometry.
As elsewhere, `--camera-model` is rejected only when a `camera_config.json`
resolves for one of the images (see [Camera Intrinsics](#camera-intrinsics)).

See [`../core/track-cluster-matching.md`](../core/track-cluster-matching.md)
for the algorithm design, empirical justification, and the production API.

## Merge

`--merge` combines multiple `.matches` files into a single file. This is useful
for combining results from different matching strategies (e.g., sequential + exhaustive)
before running a solve.

The merge process:
- Builds a unified image list from all input files
- Validates that images with the same name have identical SIFT content hashes
- Concatenates matches for each pair across all input files
- Deduplicates matches with the same feature index pair (keeps lowest descriptor distance)
- Preserves two-view geometry (TVG) data when present in any input file
  - For pairs with TVG in multiple inputs, keeps the TVG with the most inliers
  - Handles index remapping by transforming F/E/H matrices and inverting poses as needed
- Records source file names and methods in the output metadata

## Usage Examples

```bash
# Exhaustive matching (small datasets)
sfm match --exhaustive

# Sequential matching for ordered image sequences
sfm match --sequential --sequential-overlap 20

# Flow-based matching for video frames
sfm match --flow --flow-preset high_quality --flow-skip 10

# Background-floor track-cluster matching
sfm match --cluster images/

# Match a subset of images
sfm match --exhaustive --range 1:100 --max-features 4096

# Merge matches from different strategies
sfm match --merge seq.matches exhaustive.matches -o combined.matches
```
