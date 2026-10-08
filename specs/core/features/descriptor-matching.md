# Descriptor Matching

When the poses of two images are already known, the match for a feature in one
image must lie near its epipolar line in the other, so a matcher only has to
compare the feature's SIFT descriptor against the few features near that line
rather than against every feature in the image. This module is that matcher. It
sorts each image's features along the epipolar direction, slides a fixed-size
window over the other image's sorted features, takes the nearest descriptor in
the window, and keeps a match only when it is the nearest in both directions.
Optionally, before any descriptor is compared, it drops the window candidates
whose feature orientation and size disagree with what the two poses predict.
`sfm densify` uses it to find new correspondences between images of a solved
reconstruction, and `sfm epipolar --sweep-with-max-features` uses it to draw
them. The module also holds the descriptor distance and nearest-descriptor
search that other matchers in the crate build on.

The epipolar direction is made one-dimensional in one of two ways, chosen per
pair from where the epipoles lie:

- **Rectified sweep.** When both epipoles are outside their images, the two
  images are stereo-rectified, which turns every epipolar line into a
  horizontal line at the same Y in both images. Features are sorted by their
  rectified Y.
- **Polar sweep.** When an epipole is inside or near its image (the camera
  moved forward or backward), rectification is unstable, so each image's
  features are instead expressed in polar coordinates around its epipole. Every
  epipolar line is then a ray from the epipole, and features are sorted by
  angle.

The sweep is pinhole-only: it reads each camera's intrinsic matrix and treats
keypoint positions as undistorted pixels. There is no ratio test; the mutual
nearest-neighbour check is the only test of distinctiveness.

## Interface

The module is
[`features/feature_match/`](../../../crates/sfmtool-core/src/features/feature_match/mod.rs):
[`descriptor.rs`](../../../crates/sfmtool-core/src/features/feature_match/descriptor.rs)
holds the distance and best-match search,
[`sweep.rs`](../../../crates/sfmtool-core/src/features/feature_match/sweep.rs)
the rectified (Y) sweep,
[`polar.rs`](../../../crates/sfmtool-core/src/features/feature_match/polar.rs)
the polar sweep,
[`window.rs`](../../../crates/sfmtool-core/src/features/feature_match/window.rs)
the steps both sweeps share once a window is placed (the filtered nearest
descriptor in the window, and the mutual check), and
[`geometric_filter.rs`](../../../crates/sfmtool-core/src/features/feature_match/geometric_filter.rs)
the orientation and size filter. The stereo rectification itself is in
[`camera/rectification.rs`](../../../crates/sfmtool-core/src/camera/rectification.rs)
and the fundamental matrix and epipoles in
[`camera/epipolar.rs`](../../../crates/sfmtool-core/src/camera/epipolar.rs). The
bindings are in
[`sfmtool-py/src/matching/`](../../../crates/sfmtool-py/src/matching/mod.rs)
(`descriptor.rs`, `image.rs`, `sweep.rs`), registered on
`sfmtool._sfmtool.matching`, and the Python entry point is
[`feature_match/_core.py`](../../../src/sfmtool/feature_match/_core.py).

### Whole-pair matching

```rust
pub fn match_image_pair(
    k1: &Matrix3<f64>, k2: &Matrix3<f64>,
    r1: &Matrix3<f64>, t1: &Vector3<f64>,
    r2: &Matrix3<f64>, t2: &Vector3<f64>,
    width1: u32, height1: u32, width2: u32, height2: u32,
    positions1: &[f64], descriptors1: &[u8], n1: usize,
    positions2: &[f64], descriptors2: &[u8], n2: usize,
    desc_len: usize, window_size: usize, threshold: Option<f64>,
    rectification_margin: u32,
    affines1: Option<&[f64]>, affines2: Option<&[f64]>,
    geometric_config: Option<&GeometricFilterConfig>,
) -> Vec<(usize, usize, f64)>;

pub fn match_image_pairs_batch(/* per-image arrays, indexed by pair */)
    -> Vec<Vec<(usize, usize, f64)>>;
```

`match_image_pair` is what callers use. It computes the fundamental matrix from
the two poses, picks the polar sweep when either epipole lies inside its image
grown by `rectification_margin` pixels on every side, and the rectified sweep
otherwise. It returns `(index1, index2, distance)` for every mutual match, in
the caller's original feature indices. A pair whose intrinsic matrix is
singular has no fundamental matrix and returns no matches. The geometric filter
runs only when `affines1`, `affines2` and `geometric_config` are all `Some`.
`match_image_pairs_batch` runs `match_image_pair` over a list of pairs in
parallel with Rayon, taking per-camera intrinsics and image sizes and per-image
poses and features, and returns one result per pair in input order.

The poses are `cam_from_world` in the COLMAP/OpenCV camera convention (+Z
forward, Y down), because everything here is pixel-space algebra on `K`, `R`
and `t`. A caller holding `.sfmr` poses flips them first; see
[`geometry/convention.rs`](../../../crates/sfmtool-core/src/geometry/convention.rs).

All feature arrays are flat row-major: positions `N × 2`, descriptors
`N × desc_len` bytes, affine shapes `N × 4` (the 2×2 shape row-major). Flat
slices with explicit counts are the layout the NumPy arrays from the bindings
arrive in, so they cross the boundary without copying.

### The sweeps

```rust
pub fn mutual_best_match_sweep(keypoints1, descriptors1, n1,
    keypoints2, descriptors2, n2, desc_len, window_size, threshold)
    -> Vec<(usize, usize, f64)>;
pub fn mutual_best_match_sweep_geometric(/* same, plus */ affines1, affines2,
    geom: &StereoPairGeometry, config: &GeometricFilterConfig)
    -> Vec<(usize, usize, f64)>;

pub fn polar_mutual_best_match(positions1, descriptors1, n1,
    positions2, descriptors2, n2, desc_len, f_matrix: &[f64; 9],
    window_size, threshold, min_radius) -> Option<Vec<(usize, usize, f64)>>;
pub fn polar_mutual_best_match_geometric(/* same, plus */ affines1, affines2,
    geom, config) -> Option<Vec<(usize, usize, f64)>>;

pub fn match_one_way_sweep(sorted_kpts1, sorted_descs1, n1,
    sorted_kpts2, sorted_descs2, n2, window_size, threshold) -> SweepMatches;
pub fn match_one_way_sweep_geometric(/* … */) -> SweepMatches;
```

The `mutual_best_match_sweep` pair takes keypoints that are already rectified
(`match_image_pair` rectifies them) and does its own sorting. The polar pair
takes raw pixel positions and the row-major fundamental matrix, and returns
`None` when either epipole is at infinity, which is the caller's signal to use
the rectified sweep instead. The one-way functions take features already sorted
by Y and return a map from sorted index in image 1 to `(sorted index in image
2, distance)`; they are the single-direction step of the rectified sweep,
exposed for tests and the bindings.

Each `_geometric` variant runs the same sweep as its plain counterpart; the
only difference is that it narrows each window with the geometric filter before
comparing descriptors. Internally both share one body, with the filter inputs
passed as an `Option`.

### Descriptor distance and best match

```rust
pub fn descriptor_distance_l2_squared(a: &[u8], b: &[u8]) -> i64;
pub fn descriptor_distance_l2(a: &[u8], b: &[u8]) -> f64;
pub fn find_best_match(query: &[u8], candidates: &[&[u8]], threshold: Option<f64>)
    -> Option<(usize, f64)>;
pub fn find_best_match_contiguous(query: &[u8], candidates: &[u8], desc_len: usize,
    threshold: Option<f64>) -> Option<(usize, f64)>;
pub fn match_candidates_and_deduplicate(candidates: &[u32], in_bounds_idx: &[u32],
    desc1: &[u8], desc2: &[u8], n_queries: usize, k: usize, desc_len: usize,
    threshold: f64) -> Vec<[u32; 2]>;
```

The best-match functions scan every candidate and return the index and L2
distance of the nearest, or `None` when there are no candidates or the nearest
is farther than `threshold`. `match_candidates_and_deduplicate` takes, for each
query feature, a row of `k` candidate indices (empty slots are `u32::MAX`),
picks the nearest descriptor among them, and when several query features pick
the same target keeps only the nearest; it returns `[source, target]` pairs
sorted by source. Flow-based matching
([flow-based-matching.md](flow-based-matching.md)) fills the candidate rows with
the keypoints near each feature's flow-predicted position.
`descriptor_distance_l2_squared` is also the distance the k-d forest
([randomized-kdtree-forest.md](randomized-kdtree-forest.md)) and the
points-at-infinity search use.

### Example

```rust
use sfmtool_core::features::feature_match::{match_image_pair, GeometricFilterConfig};

let config = GeometricFilterConfig::default();
let matches = match_image_pair(
    &k1, &k2, &r1, &t1, &r2, &t2, w1, h1, w2, h2,
    &positions1, &descriptors1, n1, &positions2, &descriptors2, n2,
    128, 30, None, 50,
    Some(&affines1), Some(&affines2), Some(&config),
);
for (i1, i2, dist) in matches { /* feature i1 in image 1 matches i2 in image 2 */ }
```

## Theory

### Why a sweep

For a correct match, the feature in image 2 lies on the epipolar line of the
feature in image 1. After rectification that line is the horizontal line at the
same Y; in polar coordinates around the epipoles it is the ray at the
corresponding angle. Sorting both images by that one coordinate puts each
feature's possible matches next to it in the other sorted list, so a window of
the `window_size` nearest entries in the sort coordinate holds the candidates.
The window slides forward monotonically as the query advances through its own
sorted list, so one pass compares each query against at most `window_size`
candidates: `O((n1 + n2) log n + n1 · window_size)` descriptor comparisons per
direction, instead of `n1 · n2` for exhaustive matching. The window is a count
of features, not a distance, so it covers a narrow band where features are
dense and a wide one where they are sparse.

### Rectified path

`check_rectification_safe_from_f` computes each epipole from `F` (for image 2)
and `Fᵀ` (for image 1) and calls the pair safe to rectify when both lie outside
their image by more than `rectification_margin` pixels, or at infinity.
`compute_stereo_rectification` is Bouguet's method: it splits the relative
rotation equally between the two cameras, rotates both so the baseline is
horizontal, and uses the average of the two intrinsic matrices as the new
intrinsics. The keypoints are mapped through `K⁻¹`, the rectifying rotation and
the new projection, and sorted by Y with X breaking ties.

### Polar path

Features within `min_radius` (10 px, fixed in `match_image_pair`) of their
epipole are dropped, since their angle is poorly determined. The two images'
angles are measured in different frames, so image 2's angles are shifted onto
image 1's. A ray from `e1` at angle `θ` maps to the epipolar line
`l2 = F · p1 = r · (f0 cos θ + f1 sin θ)`, where `f0` and `f1` are the first two
columns of `F` and `r` cancels because `F e1 = 0`. The ray in image 2 runs
perpendicular to `l2`. The offset between the two angles varies with `θ` in
general, so `compute_angle_offset` samples 36 angles and uses the median
offset, choosing at each sample the one of the two opposite directions that is
within 90° of `θ`.

Angles wrap at ±π, so a query near one end of the sorted list needs candidates
from the other end. Before sweeping, the candidate list is extended with copies
of the entries within a threshold of each end, shifted by ∓2π so the angle
stays monotonic. The threshold is the smaller of π/4 and the angular span the
window covers on average, `window_size / n · 2π`. A match found on a copy is
mapped back to the original entry.

### Mutual best match

The sweep runs in both directions, image 1 to 2 and image 2 to 1, and a pair is
kept only when each feature is the other's nearest in its window. This rejects
a match to a feature that is nearer to some other query, which is the role a
ratio test would play, without needing a second-nearest distance. The returned
distance is the forward one. Within a window, the first of several equally near
candidates wins.

### Geometric filter

The filter predicts, from the poses, how a feature's affine shape should look in
the other image, and drops window candidates that disagree. It has two stages.

1. **Orientation, always.** The query's affine shape is rotated by the upper-left
   2×2 block of the relative rotation `R2 · R1ᵀ`. The candidate passes when the
   angle between the first column of that rotated shape and the first column of
   the candidate's shape is at most `max_angle_difference` degrees. A
   zero-length column fails.
2. **Size, when triangulation is reliable.** The ray angle between the two
   features' unprojected rays is computed. If it is below
   `min_triangulation_angle` degrees, the depth is poorly determined and the
   candidate passes without a size test. Otherwise the pair is triangulated by
   DLT; a degenerate solution or a point behind either camera fails. The rotated
   query shape is scaled by `d1 / d2` (its depths in camera 1 and camera 2), and
   the candidate passes when the ratio of its size to the scaled query size is
   within `[geometric_size_ratio_min, geometric_size_ratio_max]`. A shape's size
   is the mean length of its two columns.

The orientation stage is well conditioned for any motion. The size stage needs
a depth ratio, which only a reliable triangulation gives, so it is skipped for
nearly parallel rays (forward motion or a distant point) rather than letting a
bad depth reject good matches. For the backward sweep the filter is run with the
two cameras swapped (`StereoPairGeometry::swapped`), which transposes the 2×2
rotation.

## Implementation notes

- Every array that travels with a feature (positions, descriptors, affine
  shapes) is reordered through the same permutation by `gather_rows`, and the
  polar wraparound extends every one of them through one `Wraparound` plan. A
  window index therefore names the same feature in every array; reordering any
  one array separately would pair one feature's descriptor with another's
  shape.
- The threshold is compared in squared integer distance. Since a squared
  distance between `u8` descriptors is an integer, `d² ≤ ⌊t²⌋` is the same test
  as `d ≤ t`, and only the winning distance takes a square root.
- On the geometric path the filter runs before any descriptor comparison, and
  a window with no passing candidate yields no match for that query. Filtering
  first is what makes it cheaper than filtering matches afterwards: the size
  stage triangulates, but only for candidates that already passed orientation.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `window_size` | `30` | Candidates per window. Default of `match_image_pair` in [`_core.py`](../../../src/sfmtool/feature_match/_core.py) and of `sfm densify --sweep-window-size`. |
| `threshold` | `None` | Maximum L2 descriptor distance; `None` accepts any nearest. `sfm densify --distance-threshold`. |
| `rectification_margin` | `50` | Pixels beyond the image border an epipole must lie for the rectified sweep. Default in `_core.py`. |
| `min_radius` | `10.0` | Pixels around the epipole excluded from the polar sweep. Fixed in `match_image_pair`; a keyword of the `polar_mutual_best_match` binding. |
| `max_angle_difference` | `15.0` | Degrees, orientation stage. |
| `min_triangulation_angle` | `5.0` | Degrees; below it the size stage is skipped. |
| `geometric_size_ratio_min` | `0.8` | Size stage lower bound. `sfm densify` sets it to `1 / --geometric-size-ratio-max`. |
| `geometric_size_ratio_max` | `1.25` | Size stage upper bound. |

The four filter defaults are `GeometricFilterConfig::default()` in
`geometric_filter.rs` and the matching Python dataclass in
[`_geometric_filter.py`](../../../src/sfmtool/feature_match/_geometric_filter.py).

## Python bindings

`sfmtool._sfmtool.matching` exposes `descriptor_distance`,
`find_best_descriptor_match`, `match_candidates_by_descriptor`,
`match_image_pair`, `match_image_pairs_batch`, `match_one_way_sweep`,
`match_one_way_sweep_geometric`, `mutual_best_match_sweep`,
`mutual_best_match_sweep_geometric`, `polar_mutual_best_match` and
`polar_mutual_best_match_geometric`. They take NumPy arrays of shape
`(N, 2)` for positions, `(N, desc_len)` `uint8` for descriptors and `(N, 4)` for
affine shapes, and the counts come from the array shapes. Python code calls
`sfmtool.feature_match.match_image_pair`, which takes `pycolmap.Rigid3d` poses
and `pycolmap.Camera` objects, builds `K` from the camera, and passes a
`GeometricFilterConfig` through when filtering is enabled and both images have
affine shapes.

## Testing

Rust unit tests are beside each file:
[`descriptor/tests.rs`](../../../crates/sfmtool-core/src/features/feature_match/descriptor/tests.rs),
[`sweep/tests.rs`](../../../crates/sfmtool-core/src/features/feature_match/sweep/tests.rs),
[`polar/tests.rs`](../../../crates/sfmtool-core/src/features/feature_match/polar/tests.rs),
[`geometric_filter/tests.rs`](../../../crates/sfmtool-core/src/features/feature_match/geometric_filter/tests.rs),
[`gather/tests.rs`](../../../crates/sfmtool-core/src/features/feature_match/gather/tests.rs)
and, for `match_image_pair` with forward motion between images of different
sizes, [`tests.rs`](../../../crates/sfmtool-core/src/features/feature_match/tests.rs).
The Python side is
[`tests/matching/test_sweep_matching.py`](../../../tests/matching/test_sweep_matching.py)
and
[`tests/rust_bindings/matching/test_descriptor_rust_bindings.py`](../../../tests/rust_bindings/matching/test_descriptor_rust_bindings.py).

## Non-goals

- Matching without poses. `sfm match` finds pairs and matches with COLMAP's
  matchers, flow or track clusters ([match-command.md](../../cli/image-feature/match-command.md)),
  not with the sweep. Its `--flow` mode uses one function from this module,
  `match_candidates_and_deduplicate`.
- Distorted or non-pinhole cameras. Keypoints are not undistorted and the sweep
  follows straight epipolar lines; [epipolar-curves.md](../camera/epipolar-curves.md)
  describes the curved lines a fisheye pair has.
- Robust verification. Matches are not checked against a two-view model after
  the sweep; the poses are taken as correct.
