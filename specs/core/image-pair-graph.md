# Image pair graph: covisibility and frustum-intersection pairs

**Status:** Implemented in
`crates/sfmtool-core/src/analysis/image_pair_graph.rs` (frustum geometry in
`crates/sfmtool-core/src/camera/frustum.rs`), exposed to Python as
`sfmtool._sfmtool.analysis.build_covisibility_pairs_py` /
`build_frustum_intersection_pairs_py` and wrapped by
`src/sfmtool/_image_pair_graph.py`.

## Overview

Several pipelines need to know **which image pairs see the same part of the
scene**: `sfm analyze --coviz` / `--frustum` report the graphs directly,
`sfm densify` sweeps covisible (or frustum-overlapping) pairs, the
motion-discontinuity analysis (`motion/recon_discontinuity.py`) checks
whether temporally-adjacent images remain covisible, and the COLMAP DB
export (`colmap/db_export.py`) seeds a match table from covisibility. This
module builds those pair lists from a posed reconstruction, two ways:

1. **Covisibility** — pairs that share reconstructed 3D points (uses the
   track structure; cheap and exact for what the solver already found).
2. **Frustum intersection** — pairs whose viewing volumes overlap in 3D
   (uses only poses, intrinsics, and stored depth statistics; finds pairs
   the solver *should* have connected, even with no shared points yet).

Both filter candidate pairs by camera viewing angle: pairs whose world-space
look directions differ by more than `angle_threshold_deg` (90° typical;
`db_export` passes 180° to disable the filter) are dropped, since
near-opposite views rarely produce usable matches. Look directions come
from `compute_camera_directions`: for the canonical −Z-forward camera,
`direction = R_world_from_cam · (0, 0, −1)`.

## Covisibility pairs (`build_covisibility_pairs`)

Input: per-observation `track_point_indexes` / `track_image_indexes` plus
the image quaternions.

1. Invert the tracks into `point → [images observing it]`.
2. For every point, count each unordered image pair once (images
   deduplicated per point).
3. Drop pairs failing the viewing-angle test.
4. Return `(img_i, img_j, shared_point_count)` sorted by count descending.

## Frustum-intersection pairs (`build_frustum_intersection_pairs`)

Input: poses, per-image pinhole intrinsics (`fx, fy, cx, cy, width,
height`), and the per-image depth histograms stored in the `.sfmr`
(`depth_statistics` + `depth_histogram_counts`). Images without depth
statistics (`min_z = NaN`) are skipped.

Per image, a **view frustum** is built:

- Near/far planes come from the depth histogram at `near_percentile` /
  `far_percentile` (5 / 95 typical), via `estimate_z_from_histogram`
  (cumulative-count bin search, bin-center value). If the estimate
  inverts (`near ≥ far`), it falls back to the histogram's full
  `[min_z, max_z]` range.
- `camera/frustum.rs` supplies the geometry: `compute_frustum_corners`
  (8 world-space corners from the intrinsics and pose),
  `compute_frustum_planes` (the 6 bounding planes), and
  `compute_frustum_volume` (exact truncated-pyramid volume).

Then for every candidate pair (angle filter first):

1. **Separating-plane rejection** (`frustums_can_intersect`) — a cheap
   conservative test; pairs that cannot intersect are skipped.
2. **Monte Carlo volume estimate**
   (`estimate_frustum_intersection_volume`): sample `num_samples` (100
   typical) uniform points inside frustum A and count the fraction inside
   frustum B's planes, scaled by A's volume; the estimate is averaged over
   both directions (A→B and B→A).

Returns `(img_i, img_j, intersection_volume)` for pairs with positive
estimated overlap, sorted by volume descending. The pair loop is
rayon-parallel over the first image index, with a deterministic per-row RNG
(`seed + i`), so results are reproducible for a fixed `seed`.

## CLI surface

`sfm analyze --coviz` and `--frustum` print these graphs;
`--near-percentile`, `--far-percentile`, and `--samples` map directly onto
the frustum-pair parameters (see
[`specs/cli/analyze-command.md`](../cli/analyze-command.md)). `sfm densify`
uses both builders to choose sweep pairs.

## Testing

Sibling `tests.rs` under `analysis/image_pair_graph/` covers direction-sign
conventions (identity pose looks down world −Z), covisibility counting and
ordering, histogram percentile estimation, frustum volume against the
closed form, and intersection estimates on known-overlap configurations.
