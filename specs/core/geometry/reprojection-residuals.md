# Batched reprojection residuals

## Purpose

A reprojection residual is the pixel offset between where a world point
projects under an image's camera pose and where that image observed it.
`reprojection_residuals` computes it, as projection minus observed, for every
observation at once over a set of images that share one camera model, and
`inlier_fraction` reports the share of observations whose offset is shorter
than a pixel threshold. Both only measure: neither changes a pose or a point.

`reprojection_residuals` has one production caller, the
[cluster census](../analysis/cluster-census.md), which takes the residual norm
of every observation of each cluster's triangulated point. Reconstruction
growth ([reconstruction-growth.md](reconstruction-growth.md)), pose refinement
([absolute-pose.md](absolute-pose.md) § "Pose refinement") and pose
verification ([pose-verification.md](pose-verification.md)) each project and
score their observations with their own code rather than calling this
function.

## Definitions

- `n_img` **images** that share one `CameraIntrinsics` (any supported model,
  including fisheye). One model covers every image; per-observation models are
  out of scope.
- Per-image world-to-camera pose `(R_img, t_img)` in the canonical convention
  (`x_cam = R·X + t`; the camera looks along `−Z`, a point in front has
  `z < 0`), `R` supplied as a WXYZ unit quaternion.
- `n_pt` world points `X_p` in the canonical world frame.
- `n_obs` observations, each a triple `(image, point, uv)`: image `i` observed
  point `p` at pixel `uv`.

## Residuals

```rust
pub fn reprojection_residuals(
    cam: &CameraIntrinsics,
    quats_wxyz: &[f64],      // n_img * 4, world-to-camera (WXYZ)
    translations: &[f64],    // n_img * 3, world-to-camera
    points: &[f64],          // n_pt * 3, world points
    uv: &[f64],              // n_obs * 2, observed pixels
    obs_img: &[u32],         // n_obs, image index per observation
    obs_pt: &[u32],          // n_obs, point index per observation
    invalid_residual: f64,
) -> Vec<f64>;               // n_obs * 2, (dx, dy) per observation
```

The poses, points and pixels are flat slices in row-major `(n, k)` order,
not slices of `UnitQuaternion` or `[f64; 3]`, so a caller passes the arrays it
already holds without converting them:

- The Python binding receives numpy arrays of shape `(n_img, 4)`, `(n_img, 3)`,
  `(n_pt, 3)` and `(n_obs, 2)`, and passes each C-contiguous one to this
  function as a borrowed slice of its buffer (`to_contiguous!` in
  [lib.rs](../../../crates/sfmtool-py/src/lib.rs) copies only an array that is
  not C-contiguous).
- An `.sfmr` reconstruction holds its image poses as `(N, 4)` and `(N, 3)`
  `Array2<f64>` (`quaternions_wxyz`, `translations_xyz`), which in standard
  layout are these slices.
- The cluster census builds its candidate poses, observed pixels and
  triangulated points as flat `Vec<f64>`s and passes them as they are
  ([cluster_census.rs](../../../crates/sfmtool-core/src/analysis/cluster_census.rs)).

The function converts each quaternion to a `UnitQuaternion` once per image,
not once per observation, and writes the output as one flat buffer that the
parallel pass fills two values per observation.

An example with two images that share a pinhole camera and observe one point.
The second image is translated 0.1 along `x`, so it projects the point to
`u = 330`, and its observation at `u = 330.5` gives a residual of `−0.5`:

```rust
use sfmtool_core::geometry::{inlier_fraction, reprojection_residuals};
use sfmtool_core::{CameraIntrinsics, CameraModel};

let cam = CameraIntrinsics {
    model: CameraModel::SimplePinhole {
        focal_length: 500.0,
        principal_point_x: 320.0,
        principal_point_y: 240.0,
    },
    width: 640,
    height: 480,
};
let quats_wxyz = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]; // two images
let translations = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0];
let points = [0.0, 0.0, -5.0]; // one point, 5 units in front of both
let uv = [320.0, 240.0, 330.5, 240.0]; // one observation per image
let obs_img = [0, 1];
let obs_pt = [0, 0];

let res = reprojection_residuals(
    &cam, &quats_wxyz, &translations, &points, &uv, &obs_img, &obs_pt, f64::INFINITY,
);
assert_eq!(res, [0.0, 0.0, -0.5, 0.0]);
assert_eq!(inlier_fraction(&res, 1.0), 1.0);
```

For each observation `k`: transform its point into the camera frame,
`x_cam = R_img · X_pt + t_img`, project it with the model-general
`CameraIntrinsics::ray_to_pixel`, and emit `(u − uv_x, v − uv_y)`. The output
is a flat `(dx, dy)` per observation, in input order — observation `k`'s result
depends only on `k`, and the pass is parallel over observations.

**Invalid observations.** A point that is non-finite, behind the camera, or
outside the model's valid domain (`ray_to_pixel` returns `None`) has no
meaningful pixel residual. Rather than drop it — which would silently shrink
the observation count a downstream trim or inlier tally reasons over — its
residual is set to `(invalid_residual, 0)`. The magnitude is the caller's
choice. [`inlier_fraction`](#inlier-fraction) counts the observation as an
outlier either way, as long as a finite value is at or above its threshold.
The choice matters to other uses of the residuals: `f64::INFINITY` lets a
caller drop the observation with a finiteness test, while a large finite value
keeps a sum of squares finite, as a least-squares cost needs.

Only the point is checked for finiteness, not the pose. A non-finite rotation
or translation goes into `ray_to_pixel` as it is, so its observations get
`(invalid_residual, 0)` only where the camera model's domain test rejects the
resulting ray; otherwise the residual is whatever the projection gives, usually
with NaN or infinite components. Under a pinhole model a NaN quaternion gives
`(NaN, NaN)`, while a translation of `−∞` along `z` puts the point at the
principal point and gives a finite residual. `inlier_fraction` counts a NaN
residual as an outlier, since a NaN norm is never below the threshold.

## Inlier fraction

```rust
pub fn inlier_fraction(residuals: &[f64], threshold_px: f64) -> f64;
```

The share of observations whose residual norm `hypot(dx, dy)` is below
`threshold_px`, computed over the flat `(dx, dy)` slice returned above. Empty
input is `0.0`. A caller can threshold a pose on it; no production code
calls it, since growth and pose verification count inliers with their own
helpers.

## Bindings

`reprojection_residuals` and `inlier_fraction` live in
[reprojection.rs](../../../crates/sfmtool-core/src/geometry/reprojection.rs),
bound under `sfmtool.geometry` by
[reprojection.rs](../../../crates/sfmtool-py/src/geometry/reprojection.rs).

```python
reprojection_residuals(
    camera,                   # CameraIntrinsics shared by all images
    quaternions_wxyz,         # (n_img, 4) world-to-camera (WXYZ)
    translations,             # (n_img, 3)
    points,                   # (n_pt, 3) world points
    uv,                       # (n_obs, 2) observed pixels
    obs_image,                # (n_obs,) uint32
    obs_point,                # (n_obs,) uint32
    invalid_residual=1e6,     # float('inf') to exclude invalids by norm
) -> (n_obs, 2) float64       # (dx, dy) per observation

inlier_fraction(residuals, threshold_px) -> float   # residuals: (n_obs, 2)
```

The binding validates its inputs and raises `ValueError` on any mismatch:
`quaternions_wxyz` must be `(n_img, 4)`, `translations` `(n_img, 3)` with the
same row count, `points` `(n_pt, 3)`, `uv` `(n_obs, 2)`, `obs_image` /
`obs_point` / `uv` the same length, every `obs_image` entry below `n_img`, and
every `obs_point` entry below `n_pt`. The Rust function checks only the
observation lengths, by assertion, and panics on an out-of-range index, so the
binding checks the indexes before calling it. The result is always
`(n_obs, 2)`, including `(0, 2)` for zero observations, which `inlier_fraction`
accepts and scores as `0.0`.

## Testing requirements

- **Zero residual**: observations generated by projecting known points through
  known poses reproject to `(0, 0)` within floating point.
- **Known offset**: a pose perturbed by a known pixel shift yields that shift.
- **Invalid handling**: a point behind the camera with a finite
  `invalid_residual`, and a non-finite point with `inf`, each return
  `(invalid_residual, 0)`, and `inlier_fraction` counts both as outliers.
- **Multi-image indexing**: observations spanning several images each pick up
  the correct per-image pose.

## Non-goals

- Per-observation camera models — all images share one `CameraIntrinsics`.
- Analytic Jacobians or any optimization step; this is the measurement
  function that a solver or gate consumes.
