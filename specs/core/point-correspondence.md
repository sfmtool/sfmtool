# Point correspondence and merging across reconstructions

**Status:** Implemented in
`crates/sfmtool-core/src/reconstruction/point_correspondence.rs`, exposed to
Python as `sfmtool._sfmtool.analysis.find_point_correspondences_py` /
`merge_points_and_tracks_py` (`crates/sfmtool-py/src/analysis/core.rs`).
Python wrappers: `src/sfmtool/_point_correspondence.py` (pairwise, used by
`sfm align --method points`, `sfm compare`, and `sfm xform --align-to`) and
`src/sfmtool/merge/correspondences.py` + `merge/reconstructions.py`
(multi-way grouping and merging, used by `sfm merge`).

## Overview

Several commands need to know that point `i` in one reconstruction and point
`j` in another are the same physical 3D point: `sfm align` fits a similarity
transform to such pairs, `sfm merge` collapses them into one merged point,
and `sfm compare` / `--align-to` measure against them. This module defines
how those pairs are found and how corresponding points and their tracks are
merged.

## Feature-index correspondence (`find_point_correspondences`)

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
with different feature backends do not (see the coordinate-based alternative
below). `embedded_patches` reconstructions have no feature-index column and
cannot use this path.

### Python wrapper (`_point_correspondence.py`)

`find_point_correspondences(source_recon, target_recon, shared_images)`
calls the binding, then **drops any pair whose source or target point is at
infinity** (`w = 0` stores a unit bearing, not a metric position — it cannot
anchor a similarity fit or positional comparison). Returns the
`{source_id: target_id}` dict plus the two `(N, 3)` position arrays. Zero
correspondences (or zero finite ones) is a `ValueError`.

### Coordinate-based alternative (`find_point_correspondences_by_coordinate`)

For reconstructions from **different feature backends** (e.g. COLMAP SIFT vs
the sfmtool extractor) the same scene keypoint lands at nearly the same
pixel but under a different feature index. The pure-Python fallback matches
observations in shared images by **mutual nearest 2D keypoint** within
`pixel_threshold` px (default 2.0), reading keypoint positions from each
workspace's `.sift` files. Each per-image match votes for its
`(source_point, target_point)` pair; votes are resolved greedily
(strongest-supported first) into a one-to-one mapping, keeping pairs with at
least `min_votes` supporting images (default 2). The same
points-at-infinity filter applies.

## Multi-way grouping for `sfm merge` (`merge/correspondences.py`)

`sfm merge` generalizes pairwise correspondences to N reconstructions:

1. For every reconstruction pair sharing images, run the Rust pairwise
   finder.
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
   position/color/error **averaged** over its members, and the union of the
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

Because the merge keys observations on `(image, feature_index)` identity,
`embedded_patches` inputs are **rejected** with an error — they have no
feature indexes, and substituting a placeholder would collapse distinct
observations within an image.

> Note: `merge/correspondences.py` still contains a pure-Python
> `merge_points_and_tracks` — the superseded reference implementation of the
> same algorithm. The live path (`merge/reconstructions.py`) calls the Rust
> binding `merge_points_and_tracks_py`.

## Consumers

| Consumer | Path |
|----------|------|
| `sfm align --method points` | `align/by_points.py` → pairwise finder → least-squares/RANSAC ([reconstruction-alignment.md](reconstruction-alignment.md)) |
| `sfm merge` | `merge/reconstructions.py` → grouping + `merge_points_and_tracks_py` → PnP pose refinement |
| `sfm xform --align-to` | `xform/_align_to.py` → pairwise finder → similarity fit |
| `sfm compare` | `_compare.py` → pairwise finder (feature-index or coordinate-based) |

See [`specs/cli/align-command.md`](../cli/align-command.md) and
[`specs/cli/merge-command.md`](../cli/merge-command.md) for the CLI
surfaces.

## Testing

Sibling `tests.rs` under `reconstruction/point_correspondence/` covers the
nested-map construction, first-occurrence semantics, union-find transitive
merging, observation-key collisions, and the `embedded_patches` rejection.
Python-side behavior (infinity filtering, coordinate voting) is covered in
`tests/` via the align/merge/compare test modules.
