# Absolute Pose from 2D-3D Correspondences (P3P + RANSAC)

## Purpose

Registers one camera against known 3D structure: given image bearings
paired with world points, most of which may be wrong matches, recover the
camera's rigid pose. Iterative pose refits (robust losses, trimmed
least-squares) need the true correspondences to be a substantial fraction
of the set before their basin of attraction contains the answer; a
minimal-sample estimator succeeds whenever *some* all-inlier 3-point
sample can be drawn, which keeps registration viable down to inlier
fractions of a few percent. The RANSAC success probability per draw is
`w³` for inlier fraction `w`, so the required trials grow as
`log(1 − p) / log(1 − w³)` — about 55,000 trials at `w = 0.05` and
`p = 0.999`, an inlier fraction at which a fit over the whole set does not
converge to the true pose.

Consumers in the tree:

- `resect_one` in
  [reconstruction_growth.rs](../../../crates/sfmtool-core/src/geometry/reconstruction_growth.rs)
  runs `estimate_absolute_pose` and then `refine_absolute_pose` on the
  consensus, and when that finds no large enough consensus it runs
  `refine_absolute_pose` alone from each of its initial poses. It is the
  single-image resection behind batch registration (`resect_images_batch`)
  and next-best-view growth
  ([reconstruction-growth.md](reconstruction-growth.md)).
- `repair_poses` in
  [pose_verification.rs](../../../crates/sfmtool-core/src/geometry/pose_verification.rs)
  calls `refine_absolute_pose` alone, starting a flagged image from the mean
  pose of its registered neighbours
  ([pose-verification.md](pose-verification.md)).
- The finite path of `resect_images`
  ([resect_images/finite.rs](../../../crates/sfmtool-core/src/geometry/resect_images/finite.rs),
  [resect-image.md](../../gui/edits/resect-image.md)) calls `p3p_solve` from
  its own RANSAC loop rather than `estimate_absolute_pose`: it scores every
  pair in pixels, including tracks whose point is at infinity and cluster
  pairs, and draws its samples from the finite track pairs when there are at
  least three, neither of which the angular estimator here does.

Reconstruction merging does not use this module; it refines poses with
pycolmap's `estimate_and_refine_absolute_pose`
([merge/pose_refinement.py](../../../src/sfmtool/merge/pose_refinement.py)).

## Definitions

- `N` correspondences `(b_i, X_i)`, `i = 0..N−1`:
  - `b_i` — **bearing**: the observed ray direction of the feature as a
    unit 3-vector in the **canonical camera frame** (a camera looks along
    `−Z`; a point in front has `z < 0`). Bearings come from
    `pixel_to_ray` and are camera-model-agnostic: the solver never sees
    pixels, focal lengths, or distortion.
  - `X_i` — the corresponding 3D point in world coordinates.
- Pose `(R, t)`: world-to-camera in the canonical convention,
  `x_cam = R·X + t`. `R` is returned as a unit quaternion.
- Predicted direction `d_i = normalize(R·X_i + t)`.
- **Angular residual** `θ_i = arccos(b_i · d_i)` — the inlier test is
  `θ_i ≤ max_angular_error`. Any threshold below `π/2` subsumes the
  cheirality check (a point behind the camera predicts a direction more
  than 90° from any observable bearing). Callers thinking in pixels
  convert as `max_angular_error = atan(px / f)`.

## Rust API

The minimal solver and the estimator live in
[absolute_pose.rs](../../../crates/sfmtool-core/src/geometry/absolute_pose.rs)
(module `sfmtool_core::geometry::absolute_pose`), bound as
`sfmtool.geometry.p3p_solve` and `sfmtool.geometry.estimate_absolute_pose`;
the pixel-reprojection refiner is in
[pose_refine.rs](../../../crates/sfmtool-core/src/geometry/pose_refine.rs),
re-exported as `sfmtool_core::geometry::refine_absolute_pose` and bound as
`sfmtool.geometry.refine_absolute_pose`.

```rust
/// Up to four world-to-camera poses from three correspondences.
pub fn p3p_solve(
    bearings: &[Vector3<f64>; 3],
    points: &[Point3<f64>; 3],
) -> Vec<(UnitQuaternion<f64>, Vector3<f64>)>;

pub struct AbsolutePoseOptions {
    /// Inlier bound on the bearing/prediction angle, radians.
    pub max_angular_error: f64,
    /// Adaptive-termination target: stop once the probability that an
    /// all-inlier sample was drawn reaches this (given the best inlier
    /// count so far).
    pub confidence: f64,
    /// Hard trial cap (the adaptive bound can exceed any budget when the
    /// inlier fraction is tiny).
    pub max_iterations: u32,
    /// Reject an estimate supported by fewer inliers than this.
    pub min_inliers: usize,
    /// SplitMix64 seed for the sampler: same inputs + same seed =>
    /// bit-identical output.
    pub seed: u64,
    /// Local optimization: after each new best consensus, refit the pose
    /// on its inliers and rescore, repeating while the inlier count grows.
    pub local_optimization: bool,
}

impl Default for AbsolutePoseOptions {
    // max_angular_error: 0.01, confidence: 0.999, max_iterations: 50_000,
    // min_inliers: 6, seed: 0, local_optimization: true
}

pub struct AbsolutePoseEstimate {
    pub rotation: UnitQuaternion<f64>,   // world-to-camera, canonical
    pub translation: Vector3<f64>,
    pub inliers: Vec<bool>,              // per input correspondence
    pub iterations: u32,                 // trials actually run
}

pub fn estimate_absolute_pose(
    bearings: &[Vector3<f64>],
    points: &[Point3<f64>],
    options: &AbsolutePoseOptions,
) -> Option<AbsolutePoseEstimate>;

pub struct PoseRefinement {
    pub rotation: UnitQuaternion<f64>,   // world-to-camera, canonical
    pub translation: Vector3<f64>,
    pub inlier_fraction: f64,            // within inlier_px, over all inputs
}

pub fn refine_absolute_pose(
    cam: &CameraIntrinsics,
    uv: &[[f64; 2]],                     // observed pixels, one per point
    points: &[[f64; 3]],                 // world points (canonical frame)
    init_rotation: &UnitQuaternion<f64>,
    init_translation: &Vector3<f64>,
    trim_rounds: usize,
    keep_fraction: f64,
    inlier_px: f64,
) -> PoseRefinement;
```

**Why the estimator takes bearings and the refiner takes pixels.** The
solver and the estimator work on rays: the P3P equations below are stated in
bearings, and the inlier test is an angle between two rays. Taking bearings
keeps both free of any camera model. The caller converts each pixel once with
`CameraIntrinsics::pixel_to_ray`, which handles every supported model,
including fisheye rays more than 90° off the optical axis, and one angular
threshold then applies to all of them. The refiner minimizes pixel
reprojection error, the quantity the callers' inlier gates are measured in
(`resect_one` and `repair_poses` both count inliers within a pixel bound), and
that needs the camera's projection and its Jacobian, so it takes the camera
and the pixels themselves.

**Why the refiner is a separate call.** Not every caller starts from a
RANSAC estimate. `repair_poses` refines from a pose built from an image's
registered neighbours, and `resect_one` refines from each of its initial
poses when the estimator finds no large enough consensus; both call
`refine_absolute_pose` without `estimate_absolute_pose`. The estimator returns
a per-input inlier mask so a caller that does run both can pass only the
consensus to the refiner.

`estimate_absolute_pose` returns `None` rather than an error when no
consensus reaches `min_inliers`, because an image with too few correct matches
is an expected outcome for a caller registering many images, not a fault.

A typical call, following `resect_one`: convert the pixels to bearings,
estimate with a 4 px threshold converted to an angle, then refine on the
consensus.

```rust
use nalgebra::{Point3, Vector3};
use sfmtool_core::geometry::absolute_pose::{estimate_absolute_pose, AbsolutePoseOptions};
use sfmtool_core::geometry::refine_absolute_pose;

// cam: &CameraIntrinsics, uv: &[[f64; 2]] observed pixels,
// world: &[[f64; 3]] the matching world points.
let bearings: Vec<Vector3<f64>> = uv
    .iter()
    .map(|p| Vector3::from(cam.pixel_to_ray(p[0], p[1])))
    .collect();
let points: Vec<Point3<f64>> = world.iter().map(|x| Point3::from(*x)).collect();
let (fx, fy) = cam.focal_lengths();
let options = AbsolutePoseOptions {
    max_angular_error: (4.0 / (0.5 * (fx + fy))).atan(),
    ..Default::default()
};
if let Some(est) = estimate_absolute_pose(&bearings, &points, &options) {
    let rows: Vec<usize> = (0..uv.len()).filter(|&k| est.inliers[k]).collect();
    let uv_c: Vec<[f64; 2]> = rows.iter().map(|&k| uv[k]).collect();
    let world_c: Vec<[f64; 3]> = rows.iter().map(|&k| world[k]).collect();
    let refined = refine_absolute_pose(
        cam, &uv_c, &world_c, &est.rotation, &est.translation, 5, 0.6, 3.0,
    );
    // refined.rotation, refined.translation: the world-to-camera pose.
}
```

## The minimal solver

The three unknown depths `λ_i` (distances from the camera center along
each bearing) satisfy one law-of-cosines constraint per pair:

```
λ_i² + λ_j² − 2 λ_i λ_j (b_i · b_j) = ‖X_i − X_j‖²    for (0,1), (0,2), (1,2)
```

`p3p_solve` solves this system with the **Lambda Twist** method (Persson &
Nordberg, ECCV 2018): the two-quadric intersection is parameterized so the
depths follow from the roots of a cubic and the eigendecomposition of a 3×3
symmetric matrix, avoiding the numerically fragile quartic of the
classical (Grunert 1841) formulation. Up to four real solutions survive
the positivity constraint `λ_i > 0`. For each, the camera-frame points
`λ_i b_i` and the world points `X_i` are related by a rigid motion, and
`p3p_solve` recovers `(R, t)` by a Kabsch alignment of the two triples
rather than by the paper's direct rotation recovery. The alignment is exact
for three points, so clean inputs reproduce the generating pose to
floating-point accuracy.

Degenerate inputs return an empty result rather than poses: collinear
`X_i` (alignment is rank-deficient), coincident or antipodal bearings,
and non-finite values. Collinearity is read off the **second** singular
value of the Kabsch cross-covariance rather than the third, because three
points are always coplanar: the third singular value is ~0 for every
triple — that free plane-normal direction is fixed by the determinant
correction `R = V · diag(1, 1, det(V·Uᵀ)) · Uᵀ` that keeps `R` proper — so
only the collapse of the *second* distinguishes a collinear triple from a
usable one. The test is relative: reject when `σ₁ < KABSCH_RANK_EPS · σ₀`,
with `KABSCH_RANK_EPS = 1e-9` in `absolute_pose.rs`.

The solver is a pure function with no randomness, so its output is the
same on every run.

## The robust estimator

Each trial draws three distinct indices with the seeded SplitMix64
sampler, skips a degenerate sample (the solver's empty result), scores every
candidate pose against all `N` correspondences with the angular test, and
keeps the candidate with the most inliers. Scoring accumulates in input
order — combined with the seeded sampler this makes the whole estimator
deterministic.

`resect_one` builds its options from `AbsolutePoseOptions::default()`,
overriding only `max_angular_error` (from the camera's mean focal length) and
`seed`, so the default confidence, trial cap, inlier floor and local
optimization are the ones registration and growth run with. The Python
binding sets every field from its own keyword defaults.

**Local optimization.** When enabled, each new best consensus triggers a
refit: minimize `Σ sin²θ_i = Σ (1 − (b_i · d_i)²)` over the current
inliers, the squared norm of the component of `d_i` perpendicular to `b_i`
(close to `θ_i²` at small angles), by Gauss-Newton with a local `SO(3) × R³`
parameterization (rotation updates composed from a rotation-vector
increment), and rescore.
A refit replaces the pose when it does not shrink the consensus: when it
has more inliers, or the same number and a lower value of the cost
Gauss-Newton minimizes, evaluated over the inliers it was fitted to.
Otherwise the previous pose and its inlier set stand. Refitting repeats only
while the inlier count strictly grows, bounded by a small fixed round limit,
because a refit on an unchanged
inlier set reproduces the same pose. The equal-count case is the common
one on clean data, where the 3-point pose already has every true
correspondence as an inlier; accepting the refit there is what makes the
returned pose a fit to its whole consensus rather than to three samples.
The residual comparison keeps a Gauss-Newton run that did not converge
from replacing a pose that fits better. This recovers most of the
accuracy gap to a full robust refinement at negligible cost.

**Termination.** After each trial with best inlier count `n_best`, the
required trial count is `log(1 − confidence) / log(1 − (n_best/N)³)`;
the estimator stops when the completed trials reach it, or at
`max_iterations`. It returns `None` when the best consensus is below
`min_inliers`.

## Pose refinement

The estimator scores and refits against **angular** residuals over bearings,
which is what keeps it model-agnostic and robust at tiny inlier fractions. A
caller that holds full pixel observations — for example after an estimate has
grown an image's inlier set — can then refine the pose against **pixel**
reprojection error over its six degrees of freedom with
`refine_absolute_pose`.

Refinement is **trimmed least-squares**, not a robust loss: from the initial
pose, each of `trim_rounds` rounds computes every observation's residual norm,
keeps those at or below the `keep_fraction`-quantile of the norms, and refits
on that subset by Levenberg–Marquardt. Each LM step is taken over a local
`SO(3) × ℝ³` perturbation of the pose (`R ← exp([δθ]ₓ)·R`, `t ← t + δt`), and
the Jacobian is analytic: the projection block from
[`ray_to_pixel_with_jacobian`](../camera/projection-jacobian.md) composed with the exact
`−[R·X]ₓ` rotation and identity translation blocks. The models with an analytic
projection Jacobian are the perspective family and the fisheye models
`EQUIDISTANT_FISHEYE`, `SIMPLE_RADIAL_FISHEYE` and `SFMTOOL_FISHEYE`
(`CameraModel::supports_pixel_jacobian`). The others, the multi-coefficient
fisheye models (`OPENCV_FISHEYE`, `RADIAL_FISHEYE`, `THIN_PRISM_FISHEYE`,
`RAD_TAN_THIN_PRISM_FISHEYE`) and `EQUIRECTANGULAR`, fall back to a central
difference of the projection only, keeping the pose block exact. Damping `λ` is
adapted down on a cost improvement, up on a rejected step. The trimmed-L2
choice is deliberate: a plain L2 fit over all correspondences is biased by gross
outliers, whose large residuals dominate the sum, while a robust loss seeded
from all-large residuals has near-zero gradient — trimmed L2 from a reasonable
init avoids both failure modes.

After the trim rounds, a final refit runs on the observations with residual
`< inlier_px` when at least six qualify, and the returned `inlier_fraction` is
the share of **all** supplied observations within `inlier_px` after
refinement. An observation that is behind the camera or outside the model
domain takes the residual `(1e6, 0)` px (`INVALID_RESIDUAL` in the first
component only) with a zero Jacobian row: it is trimmed out by its norm and
adds nothing to the normal equations, so it never steers a step.

Unlike the estimator, refinement does no sampling and has no notion of a
minimal set: it assumes the supplied pose is already in the basin of
attraction and improves it.

## Bindings

`sfmtool.geometry.estimate_absolute_pose`:

```python
estimate_absolute_pose(
    points2d_or_bearings,     # (N, 2) pixels or (N, 3) unit bearings
    points3d,                 # (N, 3) world points
    *,
    camera=None,              # CameraIntrinsics; required for (N, 2) input
    max_error_px=4.0,         # converted to angular via atan(px / f_mean)
    max_angular_error=None,   # overrides max_error_px when given
    confidence=0.999,
    max_iterations=50_000,
    min_inliers=6,
    seed=0,
    local_optimization=True,
) -> dict | None
# {"quaternion_wxyz", "translation", "inliers", "iterations"}
```

With `(N, 2)` input the binding converts pixels to bearings through the
camera's `pixel_to_ray` (any supported model, including fisheye) and
derives the angular threshold from the camera's mean focal length. With
`(N, 3)` input the caller supplies bearings, and the threshold is
`max_angular_error` when given, otherwise `atan(max_error_px / f_mean)` from
`camera`; with neither, the binding raises `ValueError`. The returned pose
is canonical world-to-camera, matching `.sfmr` reconstructions.

`sfmtool.geometry.refine_absolute_pose`:

```python
refine_absolute_pose(
    camera,                   # CameraIntrinsics for the observations
    uv,                       # (N, 2) observed pixels
    points,                   # (N, 3) world points (canonical frame)
    init_quaternion_wxyz,     # (4,) initial world-to-camera rotation (WXYZ)
    init_translation,         # (3,) initial world-to-camera translation
    trim_rounds=5,
    keep_fraction=0.6,
    inlier_px=3.0,
) -> dict
# {"quaternion_wxyz", "translation", "inlier_fraction"}
```

## Testing requirements

- **Exactness**: for random non-degenerate poses and points, the
  generating pose appears among `p3p_solve`'s solutions to
  floating-point accuracy, including near-planar point triples and
  wide/narrow bearing spreads.
- **Multiplicity**: configurations with more than one valid solution
  return all of them; the estimator disambiguates with a fourth point.
- **Degeneracy**: collinear points, repeated bearings, and non-finite
  inputs yield empty results, not NaN poses.
- **Contamination sweep**: synthetic sets at inlier fractions from 0.6
  down to 0.05 recover the true pose within tolerance; below
  `min_inliers` support the estimator returns `None`.
- **Determinism**: identical inputs and seed give bit-identical results;
  different seeds may differ only within tolerance of the true pose.
- **Differential**: agreement with an established implementation
  (pycolmap's `estimate_and_refine_absolute_pose`) on a synthetic
  contaminated set of 200 correspondences, 70 of them replaced by random
  pixels: consensus sizes within 20 % of each other, poses within tolerance.
  The test skips when pycolmap is missing, its API differs, or it finds no
  pose.
- **Refinement**: from a pose perturbed off a known solution,
  `refine_absolute_pose` recovers the generating pose and reports a high
  `inlier_fraction`; the trim rounds reject planted outliers a plain L2 fit
  would follow.

## Non-goals

- The module has no large-`N` linear solver (EPnP), and it reports no
  covariance or other uncertainty for a pose.
- It does no multi-view bundle adjustment: `refine_absolute_pose` refines a
  single camera's pose against fixed 3D points, and joint refinement over many
  cameras and the structure belongs to a caller's bundle adjustment.
- It has no variant that uses a known gravity direction or estimates the
  focal length (P2P+gravity, P4Pf).

## References

- M. Persson and K. Nordberg, "Lambda Twist: An Accurate Fast Robust
  Perspective Three Point (P3P) Solver," ECCV 2018.
- J. A. Grunert, "Das Pothenotische Problem in erweiterter Gestalt nebst
  über seine Anwendungen in der Geodäsie," 1841 — the original
  three-point resection.
- M. A. Fischler and R. C. Bolles, "Random Sample Consensus," CACM 1981.
- O. Chum, J. Matas, J. Kittler, "Locally Optimized RANSAC," DAGM 2003.
