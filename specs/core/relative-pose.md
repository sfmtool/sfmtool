# Relative Pose from Ray Correspondences

**Status:** Implemented (2026-08-09) —
`crates/sfmtool-core/src/geometry/relative_pose.rs`, tests in
`relative_pose/tests.rs`; PyO3 bindings in
`crates/sfmtool-py/src/geometry/relative_pose.rs`
(`sfmtool._sfmtool.geometry.{estimate_essential_rays, fit_ray_rotation}`);
Python tests in `tests/rust_bindings/test_relative_pose_rust_bindings.py`.

## Overview

The setting: **two images of the same scene, taken with the same known
camera intrinsics, differing only in camera pose** — and a set of
point correspondences between them. Because the intrinsics are known,
each observed pixel converts to the unit ray the camera saw it along,
and the only unknown left is the motion between the two poses: a
rotation, plus a translation recoverable only up to scale (nothing in
two views says how far apart they are, only in which direction). This
module estimates that motion robustly from the ray correspondences —
the 2D–2D sibling of the absolute-pose module, which solves the
complementary problem of one camera against known 3D points.

Such a pair has two motion models, and the module exposes one estimator
per model:

- `estimate_essential_rays` — general motion (the cameras moved apart):
  a robust ray-space epipolar matrix, encoding the rotation and the
  translation direction jointly.
- `fit_ray_rotation` — rotation-only motion (the camera pivoted, or the
  scene is far enough that the baseline is negligible): a robust
  rotation taking image-1 rays onto image-2 rays.

These are the focal-vote camera-model columns' estimators
([focal-vote.md](focal-vote.md), Camera-Model Columns) evaluated for
**fixed camera intrinsics** instead of scanned over candidate focals,
returning the geometry instead of a focal. The sampling, consensus, local-optimization
and residual machinery is the column scan's, shared rather than
reimplemented.

Everything is on the sphere. A field of view past 180° puts a
substantial share of correspondences at `θ ≥ 90°`, where no `z = 1`
plane exists: rays with non-positive `z` are ordinary members of the
population, residuals are angles, and the cheirality a caller tests
after decomposing the epipolar matrix must be depth **along the ray**,
never `z > 0`.

## Inputs

Both estimators take two equal-length slices of unit rays (the caller
maps pixels to rays through its camera model) and an options struct:

| Option | Both | Meaning |
|---|---|---|
| `max_angle_rad` | ✓ | Consensus bound, radians. Derive from a keypoint localization tolerance through the camera map's local `dr/dθ` (equidistant: `tol_px / f`) |
| `min_inliers` | ✓ | Reject a consensus below this many correspondences (defaults 12 / 8) |
| `samples` | ✓ | Seeded minimal samples drawn (8-ray epipolar / 3-ray rotation) |
| `seed` | ✓ | SplitMix64 seed; identical input and seed give a bit-identical answer |
| `side` | essential only | Which epipolar residual the consensus scores (below) |

## Epipolar residual sides

The column scan's residuals are deliberately **one-sided** — its
direction-agreement certificate depends on the two directions being
separate measurements. A caller estimating geometry wants both
constraints at once, so the essential estimator adds a two-sided form
and defaults to it:

- `Both` (default) — the larger of the two one-sided angles per
  correspondence.
- `Two` — image-2 rays against the epipolar plane `E·x₁`.
- `One` — image-1 rays against the epipolar plane `Eᵀ·x₂`.

## Outputs

`estimate_essential_rays → RayEssential` (or none below `min_inliers`):
the consensus-refit epipolar matrix at unit Frobenius norm, the
consensus mask, per-correspondence angular residuals and their consensus
RMS, and the essentialness residual `(σ₁ − σ₂)/(σ₁ + σ₂)` — zero for a
perfectly essential matrix, and the column scan's cost. The matrix is
**not** projected onto the essential manifold: its distance from
essential is a measurement (a wrong camera model shows up here), and a
caller decomposing to a relative pose gets the projection for free from
the SVD it already needs.

`fit_ray_rotation → RayRotation` (or none): the rotation `R x₁ ≈ x₂`,
consensus mask, per-correspondence angular residuals, consensus RMS.

## Estimation

Both estimators: seeded minimal samples (linear 8-point on rays /
three-ray Kabsch), consensus by the angular bound, then local
optimization on the consensus set (refit + re-gate, up to 3 rounds,
keeping the last consensus that clears `min_inliers`).

## Determinism

All sampling derives from the input seed; identical inputs and seed
produce identical output on every platform.

## Tests

- Rust: planted essential and rotation recovery on synthetic scenes
  including beyond-hemisphere rays; consensus correctness under outlier
  contamination; rejection of degenerate inputs and starved consensus;
  the two-sided residual's relationship to the one-sided forms; seeded
  determinism; column-scan outputs unchanged by the machinery sharing
  (covered by the scan's existing tests).
- Python bindings: array round-trip, option plumbing incl. `side` and
  seeds, shape/type checks, agreement with the Rust results.
