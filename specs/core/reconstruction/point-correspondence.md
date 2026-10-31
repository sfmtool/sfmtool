# Point correspondence and merging across reconstructions

## Overview

Several commands need to know that point `i` in one reconstruction and point
`j` in another are the same physical 3D point: `sfm align` fits a similarity
transform to such pairs, `sfm merge` collapses them into one merged point,
and `sfm compare` / `--align-to` measure against them. This module defines
how those pairs are found and how corresponding points and their tracks are
merged.

## Feature-index correspondence (`find_point_correspondences`)

The kernel lives in
[point_correspondence.rs](../../../crates/sfmtool-core/src/reconstruction/point_correspondence.rs),
bound as `sfmtool.analysis.find_point_correspondences` and
`merge_points_and_tracks`
([analysis/core.rs](../../../crates/sfmtool-py/src/analysis/core.rs)). The Python
side wraps it twice: [_point_correspondence.py](../../../src/sfmtool/_point_correspondence.py)
for the pairwise finders, and
[merge/correspondences.py](../../../src/sfmtool/merge/correspondences.py) plus
[merge/reconstructions.py](../../../src/sfmtool/merge/reconstructions.py) for
multi-way grouping and merging.

The primary correspondence key is **shared feature observations**: if source
image A and target image B are the same photograph, and both reconstructions
observe feature index `k` of that photograph, then the 3D points those two
observations belong to correspond.

`find_point_correspondences(...)` takes the two reconstructions' track
columns (`image_indexes`, `feature_indexes`, `point_indexes`) plus parallel
`shared_images_source` / `shared_images_target` arrays naming the shared
image pairs, and:

1. Builds a nested map `image → (feature_index → point_id)` for each side.
2. For every shared image pair, intersects the two feature maps; each common
   feature index yields a `(source_point, target_point)` pair.
3. Applies **first-occurrence semantics**: a source point observed in several
   shared images keeps the target point from the first shared image that
   matched it (no voting; the map is not guaranteed one-to-one on the target
   side).

Returns parallel `source_ids` / `target_ids` vectors. Because the key is the
feature *index*, both reconstructions must reference the **same `.sift`
files** — two solves of the same workspace qualify; reconstructions built
with different feature backends do not. `embedded_patches` reconstructions
have no feature-index column and cannot use this path; the keypoint-position
matcher below serves both cases.

### Python wrapper (`_point_correspondence.py`)

`find_point_correspondences(source_recon, target_recon, shared_images)`
calls the binding, then **drops any pair whose source or target point is at
infinity** (`w = 0` stores a unit bearing, not a metric position — it cannot
anchor a similarity fit or positional comparison). Returns the
`{source_id: target_id}` dict plus the two `(N, 3)` position arrays. Zero
correspondences (or zero finite ones) is a `ValueError`.

## Keypoint-position correspondence (`find_point_correspondences_by_coordinate`)

For reconstructions from **different feature backends** (e.g. COLMAP SIFT vs
the sfmtool extractor) the same scene keypoint lands at nearly the same
pixel but under a different feature index, and an `embedded_patches`
reconstruction has no feature index at all. This matcher keys on the pixel
instead. The kernel lives in
[point_correspondence/by_coordinate.rs](../../../crates/sfmtool-core/src/reconstruction/point_correspondence/by_coordinate.rs),
bound as `sfmtool.analysis.find_point_correspondences_by_coordinate`:

```rust
pub fn find_point_correspondences_by_coordinate(
    source_track_image_indexes: &[u32],
    source_keypoints_xy: &[[f64; 2]],
    source_track_point_indexes: &[u32],
    target_track_image_indexes: &[u32],
    target_keypoints_xy: &[[f64; 2]],
    target_track_point_indexes: &[u32],
    shared_images_source: &[u32],
    shared_images_target: &[u32],
    options: &CoordinateCorrespondenceOptions, // pixel_threshold 2.0, min_votes 2
) -> PointCorrespondenceResult
```

It takes the same layout as the feature-index finder, with one keypoint
position per observation in place of the feature index, so a caller holding
track columns passes them unchanged.

1. For every shared image pair, the two sides' observations of that image are
   matched by **mutual nearest keypoint**: each is the other's nearest
   keypoint, at most `pixel_threshold` px apart. The search sorts one image's
   keypoints by `x` and scans only those within the threshold in `x`, so an
   image with thousands of observations is not compared all against all.
2. Each match casts one vote for its `(source_point, target_point)` pair.
3. The votes are resolved greedily into a **one-to-one** mapping, most votes
   first, keeping pairs with at least `min_votes` votes and skipping a pair
   whose source or target point is already taken.

The result is deterministic: among equally near keypoints the one listed
first for its image wins, and pairs with equal votes are taken in the order
their first vote was cast (shared pairs in the order given, source
observations in row order). The pairs come back in the order they were
accepted. A non-finite position matches nothing. The function knows point ids
only, so a pair with a point at infinity is kept and the caller drops it.

### Python wrapper

`find_point_correspondences_by_coordinate(source_recon, target_recon,
shared_images, pixel_threshold=2.0, min_votes=2)` in `_point_correspondence.py`
gathers each side's observation positions per image (the embedded keypoints
of an `embedded_patches` reconstruction, else the positions in the workspace's
`.sift` files) and hands them to `_vote_point_correspondences`, which flattens
them into track columns and calls the binding. Then the same
points-at-infinity filter as the feature-index wrapper applies.

## Multi-way grouping for `sfm merge` (`merge/correspondences.py`)

`sfm merge` generalizes pairwise correspondences to N reconstructions:

1. For every reconstruction pair sharing images, run the Rust pairwise
   finder through `_finite_point_pairs` in `_point_correspondence.py`, the
   same helper the pairwise wrapper uses, which drops each pair with a point
   at infinity. Such a point's position is a unit direction, so it would
   distort the distance statistics in step 3.
2. Union the pairwise results transitively into **correspondence groups** —
   sets of `(recon_idx, point_id)` representing one physical point.
3. **Percentile-filter** the groups: for each multi-point group, compute the
   max distance of any member from the group centroid; groups above the
   `--merge-percentile` (default 95.0) percentile of that distribution are
   rejected as outliers (the inputs are pre-aligned, so a wide group means a
   bad correspondence).

## Merging points and tracks (`merge_points_and_tracks`)

`merge_points_and_tracks(reconstructions, correspondence_groups,
reverse_image_mapping)` produces the merged point set and track columns:

1. **Group points:** each correspondence group becomes one temp point with
   position/color/error **averaged** over its members (the averaged error is
   not the merged track's RMS, so `sfm merge` scores every point again once it
   has refined the poses), and the union of the
   members' observations remapped into merged-image indices via
   `reverse_image_mapping`.
2. **Unique points:** every point not in any group is carried over as its
   own temp point.
3. **Union-find over observations:** temp points that share an observation
   key `(merged_image_index, feature_index)` are transitively merged — this
   catches duplicates the pairwise finder missed (and duplicates *within*
   one reconstruction that share a feature).
4. Each final group is emitted as one merged point (averaged again over the
   temp points) with the union of observations as its track rows.

The merged points are returned as `positions_xyzw`, one homogeneous
`[x, y, z, w]` row per point, so points at infinity (`w = 0`) survive the
merge. Steps 1 and 4 average positions by one rule:

- When any point averaged is finite, the result is finite (`w = 1`), at the
  mean of the finite points' positions alone. A finite point carries a depth;
  a point at infinity sharing its observations does not contradict that
  depth, it only failed to measure one. The grouping step never puts a point
  at infinity into a correspondence group, so this case arises in step 3,
  when copies of a point share an observation, one finite and one at
  infinity. `sfm xform --classify-points-at-infinity` can decide such a point
  again on its merged track.
- When every point averaged is at infinity, the result is at infinity
  (`w = 0`), along the normalised sum of their unit directions; when the
  directions cancel to zero, along the first point's direction.

Colors and errors are averaged over every point, finite or not.

Because the merge keys observations on `(image, feature_index)` identity,
`embedded_patches` inputs are **rejected** with an error — they have no
feature indexes, and substituting a placeholder would collapse distinct
observations within an image.

`merge/reconstructions.py` stores the rows as homogeneous positions, and
its pose refinement ([merge/pose_refinement.py](../../../src/sfmtool/merge/pose_refinement.py))
leaves the observations of a point at infinity out of PnP, which takes
Euclidean positions only.

## Consumers

| Consumer | Path |
|----------|------|
| `sfm align --method points` | `align/by_points.py` → pairwise finder → least-squares/RANSAC ([reconstruction-alignment.md](../analysis/reconstruction-alignment.md)) |
| `sfm merge` | `merge/reconstructions.py` → grouping + the `merge_points_and_tracks` binding → PnP pose refinement |
| `sfm xform --align-to` | `xform/_align_to.py` → pairwise finder → similarity fit |
| `sfm compare` | `compare/_core.py` → pairwise finder (feature-index or coordinate-based) |
| SfM Explorer `Align to ▸` (Points) | `align_reconstructions` in [reconstructions.rs](../../../crates/sfmtool-core/src/analysis/alignment/reconstructions.rs) → feature-index finder when both sides have feature indexes, keypoint-position finder otherwise ([reconstruction-alignment.md](../analysis/reconstruction-alignment.md)) |

See [`specs/cli/reconstruction/align-command.md`](../../cli/reconstruction/align-command.md) and
[`specs/cli/reconstruction/merge-command.md`](../../cli/reconstruction/merge-command.md) for the CLI
surfaces.

## Testing

Sibling `tests.rs` under `reconstruction/point_correspondence/` covers the
nested-map construction, first-occurrence semantics, the rule for merging
points at infinity (with a finite point, with each other, with cancelling
directions, and inside a correspondence group), and the `embedded_patches`
rejection.
`point_correspondence/by_coordinate/tests.rs` covers the keypoint-position
matcher: the threshold, mutual nearest keypoints, vote counting, the
one-to-one resolution and its tie orders, non-finite positions, and agreement
with an all-pairs reference on about 1,500 observations a side.
Python-side behavior (infinity filtering, coordinate voting through the
binding, `sfm merge` keeping points at infinity) is covered in `tests/` via
the align/merge/compare test modules and
`tests/test_point_correspondence_coordinate.py`.
