# Absolute Pose from 2D-3D Correspondences (P3P + RANSAC)

**Status:** Implemented (2026-07-14) —
`crates/sfmtool-core/src/geometry/absolute_pose.rs` (solver + estimator,
tests in `absolute_pose/tests.rs`), PyO3 bindings in
`crates/sfmtool-py/src/geometry/absolute_pose.rs`
(`sfmtool._sfmtool.geometry.p3p_solve` / `estimate_absolute_pose`), Python
tests in `tests/rust_bindings/test_absolute_pose_rust_bindings.py`. Registers
one camera against known 3D structure from bearing-vector / 3D-point
correspondences that may be dominated by wrong matches.

> _Deviation (2026-07-14): `p3p_solve` returns
> `Vec<(UnitQuaternion<f64>, Vector3<f64>)>` (allocated with capacity 4), not
> the specified `ArrayVec<_, 4>` — the workspace has no `arrayvec` dependency
> and does not use fixed-capacity vectors elsewhere, so a plain `Vec` is the
> house-style fit. Behaviourally identical: at most four poses, allocated once._
>
> _Deviation (2026-07-14): the three-point Kabsch alignment necessarily has a
> rank-2 cross-covariance (three points are always coplanar), so the collinear
> degeneracy is detected from the collapse of the **second** singular value,
> not the third; the third (plane-normal) direction is resolved by the
> determinant correction. The spec's contract (collinear points ⇒ empty) is
> unchanged._

## Purpose

Given `N` correspondences between observed image bearings and known 3D
points, estimate the camera's rigid pose, robust to a heavily contaminated
correspondence set. Iterative pose refits (robust losses, trimmed
least-squares) need the true correspondences to be a substantial fraction
of the set before their basin of attraction contains the answer; a
minimal-sample estimator succeeds whenever *some* all-inlier 3-point
sample can be drawn, which keeps registration viable down to inlier
fractions of a few percent. The RANSAC success probability per draw is
`w³` for inlier fraction `w`, so the required trials grow as
`log(1 − p) / log(1 − w³)` — tractable at `w = 0.05` where any
full-set fit is hopeless.

Consumers: last-chance image registration in reconstruction growth,
relocalization of a new image against an existing reconstruction, and any
place a pose must be estimated from vetted-point support rather than
pairwise geometry.

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

## The minimal solver

```rust
/// Up to four world-to-camera poses from three correspondences.
pub fn p3p_solve(
    bearings: &[Vector3<f64>; 3],
    points: &[Point3<f64>; 3],
) -> ArrayVec<(UnitQuaternion<f64>, Vector3<f64>), 4>;
```

The three unknown depths `λ_i` (distances from the camera center along
each bearing) satisfy one law-of-cosines constraint per pair:

```
λ_i² + λ_j² − 2 λ_i λ_j (b_i · b_j) = ‖X_i − X_j‖²    for (0,1), (0,2), (1,2)
```

Solve this system with the **Lambda Twist** method (Persson & Nordberg,
ECCV 2018): the two-quadric intersection is parameterized so the depths
follow from the roots of a cubic and the eigendecomposition of a 3×3
symmetric matrix, avoiding the numerically fragile quartic of the
classical (Grunert 1841) formulation. Up to four real solutions survive
the positivity constraint `λ_i > 0`. For each, the camera-frame points
`λ_i b_i` and the world points `X_i` are related by a rigid motion;
recover `(R, t)` with an exact three-point rigid alignment. The paper's
direct rotation recovery and a Kabsch alignment are both acceptable —
the contract is that clean inputs reproduce the generating pose to
floating-point accuracy.

Degenerate inputs return an empty result rather than poses: collinear
`X_i` (alignment is rank-deficient), coincident or antipodal bearings,
and non-finite values.

The solver is a pure function: no allocation beyond the fixed-capacity
result, no randomness, bit-stable across runs.

## The robust estimator

```rust
pub struct AbsolutePoseOptions {
    /// Inlier bound on the bearing/prediction angle, radians.
    pub max_angular_error: f64,
    /// Adaptive-termination target: stop once the probability that an
    /// all-inlier sample was drawn exceeds this (given the best inlier
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
    /// on its inliers and rescore, repeating while the inlier set grows.
    pub local_optimization: bool,
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
```

Per trial: draw three distinct indices with the seeded SplitMix64
sampler, skip degenerate samples (the solver's empty result), score every
candidate pose against all `N` correspondences with the angular test, and
keep the candidate with the most inliers. Scoring accumulates in input
order — combined with the seeded sampler this makes the whole estimator
deterministic.

**Local optimization.** When enabled, each new best consensus triggers a
refit: minimize the sum of squared angular residuals over the current
inliers by Gauss-Newton with a local `SO(3) × R³` parameterization
(rotation updates composed from a rotation-vector increment), rescore,
and repeat while the inlier count strictly grows (bounded by a small
fixed round limit). This recovers most of the accuracy gap to a full
robust refinement at negligible cost, and keeps the returned pose the
best *refit* pose, not a raw 3-point solution.

**Termination.** After each trial with best inlier count `n_best`, the
required trial count is `log(1 − confidence) / log(1 − (n_best/N)³)`;
stop when the completed trials exceed it, or at `max_iterations`.
Return `None` when the best consensus is below `min_inliers`.

## Bindings

`sfmtool._sfmtool.geometry.estimate_absolute_pose`:

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
`(N, 3)` input the caller supplies bearings and an angular threshold
directly. The returned pose is canonical world-to-camera, matching
`.sfmr` reconstructions.

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
  (pycolmap's estimator) on real correspondence sets — same consensus
  size within sampling variation, poses within tolerance.

## Non-goals (v1)

- Large-`N` linear solvers (EPnP) and covariance/uncertainty output.
- Full robust refinement beyond the local-optimization refit — callers
  that own a bundle adjustment refine there.
- Gravity- or focal-estimating variants (P2P+gravity, P4Pf).

## References

- M. Persson and K. Nordberg, "Lambda Twist: An Accurate Fast Robust
  Perspective Three Point (P3P) Solver," ECCV 2018.
- J. A. Grunert, "Das Pothenotische Problem in erweiterter Gestalt nebst
  über seine Anwendungen in der Geodäsie," 1841 — the original
  three-point resection.
- M. A. Fischler and R. C. Bolles, "Random Sample Consensus," CACM 1981.
- O. Chum, J. Matas, J. Kittler, "Locally Optimized RANSAC," DAGM 2003.
