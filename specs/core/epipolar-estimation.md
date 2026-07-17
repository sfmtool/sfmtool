# Epipolar Geometry from 2D-2D Correspondences (7-point + RANSAC + Bougnoux)

**Status:** Proposed — target `crates/sfmtool-core/src/geometry/epipolar_estimation.rs`
(solvers + estimator, tests in `epipolar_estimation/tests.rs`), PyO3 bindings
in `crates/sfmtool-py/src/geometry/epipolar_estimation.rs`
(`sfmtool._sfmtool.geometry.estimate_fundamental` /
`focal_from_fundamental`), Python tests in
`tests/rust_bindings/test_epipolar_estimation_rust_bindings.py`. Estimates the
fundamental matrix relating two views from pixel correspondences that may be
contaminated by wrong matches, and extracts a focal-length estimate from it.

The complementary direction — the fundamental matrix **of two known
cameras**, for epipolar curve rendering — already exists in
`crates/sfmtool-core/src/camera/epipolar.rs` (`compute_fundamental_matrix`,
`compute_epipole`). This module is estimation from data; that one is
derivation from poses. `compute_epipole`'s null-space extraction is shared.

## Purpose

Given `N` pixel correspondences between two images, estimate the 3×3
fundamental matrix `F` (rank 2, defined up to scale) with `x₂ᵀ F x₁ = 0`
for true correspondences, robust to a contaminated set. A minimal-sample
estimator succeeds whenever some all-inlier 7-point sample can be drawn,
so verification stays viable at low inlier fractions where any full-set
fit is hopeless.

The fundamental matrix additionally determines the cameras' focal lengths
when the principal points are known (Bougnoux): each image pair then casts
an independent focal estimate, and a caller aggregating many pairs obtains
a focal consensus that requires no 3D structure — it cannot be biased by
the depth/focal compensation that afflicts structure-based focal
estimation on narrow-baseline image sets.

Consumers: two-view geometric verification of feature matches (inlier
masks over candidate matches), focal initialization by consensus over
wide-baseline pairs, and relative-pose seeding wherever pairwise geometry
is needed before any reconstruction exists.

## Definitions

- `N` correspondences `(x₁ᵢ, x₂ᵢ)`, `i = 0..N−1` — **pixel** positions of
  the same scene point in image 1 and image 2. Homogeneous forms
  `x̃ = (u, v, 1)ᵀ`.
- Fundamental matrix `F`: rank-2, `x̃₂ᵀ F x̃₁ = 0` exactly for noise-free
  true correspondences; scale is arbitrary (returned normalized to unit
  Frobenius norm, sign unspecified).
- Epipoles: `F ẽ₁ = 0`, `Fᵀ ẽ₂ = 0`.
- **Sampson distance** — the residual and inlier test, in squared pixels:

  ```
  d²(x₁, x₂; F) = (x̃₂ᵀ F x̃₁)² / ( (F x̃₁)₁² + (F x̃₁)₂² + (Fᵀ x̃₂)₁² + (Fᵀ x̃₂)₂² )
  ```

  the first-order approximation of the reprojection error; the inlier
  test is `d² ≤ max_error_px²`.
- **Hartley normalization**: each image's points are translated to zero
  mean and scaled so the mean distance from the origin is `√2` before any
  linear solve; the resulting `F̂` is denormalized as `F = T₂ᵀ F̂ T₁`.
  All linear solves below operate on normalized coordinates; Sampson
  scoring operates on pixels.

## The minimal solver (7-point)

```rust
/// One to three fundamental matrices from seven correspondences.
pub fn fundamental_7pt(
    x1: &[[f64; 2]; 7],
    x2: &[[f64; 2]; 7],
) -> Vec<Matrix3<f64>>;
```

Each correspondence contributes one row `(u₂u₁, u₂v₁, u₂, v₂u₁, v₂v₁, v₂,
u₁, v₁, 1)` of the 7×9 design matrix `A` with `A·vec(F) = 0`. The
null space of `A` is two-dimensional, spanned by `F_a, F_b`; the rank-2
constraint `det(α F_a + (1−α) F_b) = 0` is a cubic in `α` with one or
three real roots, each yielding a candidate `F`. Roots are found in
closed form (depressed cubic / trigonometric method), not iteratively.

Degenerate inputs return an empty result rather than candidates: a design
matrix whose null space exceeds two dimensions (fewer than seven
independent constraints — repeated points, or all points on a conic
through both images' configurations) and non-finite values.

The solver is a pure function: no randomness, bit-stable across runs.

## The refit solver (normalized 8-point)

```rust
/// Least-squares fundamental matrix from N >= 8 correspondences.
pub fn fundamental_8pt(x1: &[[f64; 2]], x2: &[[f64; 2]]) -> Option<Matrix3<f64>>;
```

The singular vector of the smallest singular value of the `N`×9 design
matrix (in normalized coordinates), followed by rank-2 enforcement:
replace `F̂` by the closest rank-2 matrix in Frobenius norm (zero the
smallest singular value). Used for local optimization inside the
estimator and available to callers polishing an inlier set. Returns
`None` for `N < 8`, non-finite input, or a rank-deficient design matrix.

## The robust estimator

```rust
pub struct FundamentalOptions {
    /// Inlier bound on the Sampson distance, pixels.
    pub max_error_px: f64,
    /// Adaptive-termination target: stop once the probability that an
    /// all-inlier sample was drawn exceeds this (given the best inlier
    /// count so far).
    pub confidence: f64,
    /// Hard trial cap.
    pub max_iterations: u32,
    /// Reject an estimate supported by fewer inliers than this.
    pub min_inliers: usize,
    /// SplitMix64 seed for the sampler: same inputs + same seed =>
    /// bit-identical output.
    pub seed: u64,
    /// Local optimization: after each new best consensus, refit with the
    /// normalized 8-point solver on its inliers and rescore, repeating
    /// while the inlier set grows.
    pub local_optimization: bool,
}

pub struct FundamentalEstimate {
    pub f_matrix: Matrix3<f64>,          // unit Frobenius norm, rank 2
    pub inliers: Vec<bool>,              // per input correspondence
    pub iterations: u32,                 // trials actually run
}

pub fn estimate_fundamental(
    x1: &[[f64; 2]],
    x2: &[[f64; 2]],
    options: &FundamentalOptions,
) -> Option<FundamentalEstimate>;
```

Per trial: draw seven distinct indices with the seeded SplitMix64 sampler,
skip degenerate samples (the solver's empty result), score every candidate
`F` against all `N` correspondences with the Sampson test, and keep the
candidate with the most inliers. Scoring accumulates in input order —
combined with the seeded sampler this makes the whole estimator
deterministic.

**Local optimization.** When enabled, each new best consensus triggers a
refit: `fundamental_8pt` on the current inliers, rescore, and repeat
while the inlier count strictly grows (bounded by a small fixed round
limit). The returned `F` is then the best refit matrix, not a raw
7-point solution.

**Termination.** After each trial with best inlier count `n_best`, the
required trial count is `log(1 − confidence) / log(1 − (n_best/N)⁷)`;
stop when the completed trials exceed it, or at `max_iterations`. Return
`None` when the best consensus is below `min_inliers`.

The estimator does not detect epipolar degeneracy of the *scene*: a
correspondence set explained by a homography (planar scene, or
rotation-only camera motion) yields an `F` whose inlier set is real but
whose epipolar geometry is one of a one-parameter family. Callers that
must reject such pairs test the inlier set against a homography and
compare support; that model selection is out of scope here (see
non-goals).

## Focal length from the fundamental matrix

```rust
/// Focal length of camera 1 in pixels, or None when the pair is
/// degenerate for focal recovery.
pub fn focal_from_fundamental(
    f_matrix: &Matrix3<f64>,
    pp1: [f64; 2],
    pp2: [f64; 2],
) -> Option<f64>;
```

The Bougnoux formula. With `p̃₁, p̃₂` the homogeneous principal points,
`ẽ₂` the epipole in image 2 (`Fᵀ ẽ₂ = 0`), `Ĩ = diag(1, 1, 0)`, and
`[ẽ₂]ₓ` the cross-product matrix:

```
f₁² = − ( p̃₂ᵀ [ẽ₂]ₓ Ĩ F p̃₁ ) · ( p̃₂ᵀ F p̃₁ )
      ─────────────────────────────────────────
        p̃₂ᵀ [ẽ₂]ₓ Ĩ F Ĩ Fᵀ p̃₂
```

Camera 2's focal follows from the same formula applied to `Fᵀ` with the
principal points swapped. Return `None` when the denominator vanishes or
`f₁² ≤ 0` — the classical degeneracies land there: optical axes
intersecting (both cameras fixating the same point), pure forward
translation along the optical axis, and rotation-dominant motion. The
formula assumes square pixels, zero skew, and known principal points.

Under noise the estimate is heavy-tailed and biased for near-degenerate
motion, so a single pair's focal is an initialization, not a
measurement; callers aggregate a robust statistic (median) over many
pairs and treat the result as a starting point for structure-based
refinement.

## Bindings

`sfmtool._sfmtool.geometry.estimate_fundamental`:

```python
estimate_fundamental(
    points1,                  # (N, 2) pixels in image 1
    points2,                  # (N, 2) pixels in image 2
    *,
    max_error_px=3.0,
    confidence=0.999,
    max_iterations=10_000,
    min_inliers=12,
    seed=0,
    local_optimization=True,
) -> dict | None
# {"f_matrix": (3, 3) float64, "inliers": (N,) bool, "iterations": int}
```

`sfmtool._sfmtool.geometry.focal_from_fundamental`:

```python
focal_from_fundamental(
    f_matrix,                 # (3, 3)
    principal_point1,         # (2,) pixels
    principal_point2,         # (2,) pixels
) -> float | None             # focal of camera 1, pixels
```

## Testing requirements

- **Exactness (7-point)**: for correspondences generated from random
  non-degenerate camera pairs, one of the returned candidates satisfies
  `x̃₂ᵀ F x̃₁ = 0` to floating-point accuracy on all seven inputs, and
  agrees (up to scale) with the F derived from the generating cameras via
  `compute_fundamental_matrix`.
- **Cubic multiplicity**: configurations with three real roots return all
  three; the estimator disambiguates with the remaining correspondences.
- **Rank**: every returned matrix has rank exactly 2 (smallest singular
  value zero to tolerance); `fundamental_8pt` output likewise after
  enforcement.
- **Degeneracy (solver)**: repeated points and non-finite values yield
  empty/None results, not NaN matrices.
- **Contamination sweep**: synthetic sets at inlier fractions from 0.9
  down to 0.2 recover the generating epipolar geometry (Sampson residuals
  of the true inliers within tolerance); below `min_inliers` support the
  estimator returns `None`.
- **Determinism**: identical inputs and seed give bit-identical results
  (matrix, mask, iteration count).
- **Differential**: agreement with an established implementation
  (OpenCV `findFundamentalMat` with its RANSAC flag) on synthetic and
  real correspondence sets — consensus sizes within sampling variation,
  inlier Sampson residuals comparable.
- **Focal recovery**: across randomized non-degenerate poses, scenes, and
  focal lengths, `focal_from_fundamental` on the exact F recovers the
  generating focal to floating-point accuracy; on RANSAC-estimated F from
  noisy correspondences the median over many pairs lands within a few
  percent. Fixating, forward-motion, and rotation-only configurations
  return `None` (or are excluded by the sign test) rather than arbitrary
  values.
- **Bindings**: dtype/shape validation, contiguity handling, and the
  documented dict layout, mirroring `test_absolute_pose_rust_bindings.py`.

## Non-goals (v1)

- Essential-matrix estimation and relative-pose decomposition (R, t from
  E) — a natural v2 once calibrated consumers exist.
- Homography estimation and F-vs-H model selection (GRIC/QDEGSAC); the
  estimator documents the planar/rotation-only caveat instead.
- Degeneracy-aware sampling (DEGENSAC) and PROSAC-style guided sampling.
- Radial-distortion-aware fundamental matrices.
- Uncertainty/covariance output.

## References

- R. Hartley and A. Zisserman, "Multiple View Geometry in Computer
  Vision," 2nd ed. — the 7-point and normalized 8-point algorithms,
  Sampson distance.
- R. Hartley, "In Defense of the Eight-Point Algorithm," PAMI 1997 — the
  normalization.
- S. Bougnoux, "From Projective to Euclidean Space under any Practical
  Situation, a Criticism of Self-Calibration," ICCV 1998 — the focal
  formula.
- M. A. Fischler and R. C. Bolles, "Random Sample Consensus," CACM 1981.
- O. Chum, J. Matas, J. Kittler, "Locally Optimized RANSAC," DAGM 2003.
