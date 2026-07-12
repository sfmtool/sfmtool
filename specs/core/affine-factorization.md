# Affine Factorization

**Status: design, pre-implementation.** Production spec for promoting the
alternating-least-squares affine factorization from the pinhole bootstrap
experiments (`scripts/exp_pinhole_bootstrap.py`, notes in
`cluster-pinhole-bootstrap.md`) into `sfmtool-core` and the bindings.

## Purpose

Given 2D observations of clusters across a small group of images — with
most (cluster, image) combinations unobserved and some observations junk —
jointly estimate:

- an **affine camera** per image: a 2×3 matrix `M_i` and 2-vector `t_i`
  with `u ≈ M_i·X + t_i`,
- a **3D point** per cluster, in a shared affine coordinate system,
- a per-observation **keep mask** separating the consistent observations
  from the trimmed outliers,

and, as a second step, the **metric upgrade**: the 3×3 gauge that makes the
affine cameras rotation-times-scale, yielding a per-image rotation and
scale (both reflection hypotheses — the factorization cannot distinguish
them).

This is an operation that produces cameras and structure from 2D
positions alone: no poses, no intrinsics, no pairwise geometry. Its
output is an initialization, valid to the extent the affine camera model
holds (depth relief small relative to viewing distance — i.e. groups of
nearby viewpoints).

## Model

Throughout this spec:

- `N` — the number of images (equivalently, of affine cameras),
- `C` — the number of clusters (equivalently, of 3D points),
- `K` — the number of observations; each observation is the 2D position
  of one cluster in one image.

Stacking the x/y rows of all `N` images against all `C` cluster columns
gives the measurement matrix `W` (2N×C), holding the `K` observed
positions with the remaining entries missing. For a rigid scene under
affine viewing, after removing the per-image translations, `W = M·X` with
`M` (2N×3) stacking the camera matrices and `X` (3×C) the points —
rank ≤ 3 (Tomasi–Kanade). The factorization is defined only up to an
invertible 3×3 gauge `A`: `(M·A, A⁻¹·X)` fits identically. Missing
entries prevent a direct SVD, hence alternation.

## Algorithm

Inputs: parallel observation arrays `obs_clusters: &[u32]`,
`obs_images: &[u32]`, `obs_xy` (2 f64 per observation), plus
`num_images`, `num_clusters`, and params. No ordering is required of the
observation arrays. Coordinates are expected centered (the caller
subtracts its chosen image center); the algorithm is translation-invariant
beyond conditioning.

1. **Initialization.** Build the dense 2N×C matrix with observed entries
   filled in and missing entries set to the row mean; subtract row means;
   take the top-3 right singular vectors as the initial `X` (C×3).
2. **Rounds** (fixed count, default 25). Each round:
   a. **Camera sweep.** Per image, least-squares fit of `(M_i | t_i)`
      (2×4 unknowns as two rows) over that image's kept observations;
      images with fewer than 4 kept observations keep their previous
      values.
   b. **Point sweep.** Per cluster, least-squares fit of `X_c` (3
      unknowns) over that cluster's kept observations; clusters with fewer
      than 2 kept observations keep their previous values.
   c. **Residuals.** `r = u − (M_i·X_c + t_i)` for every observation.
   d. **Trimming**, only from round `rounds / 2` (integer division)
      onward: the kept set becomes the observations whose residual norm is
      strictly below the `(1 − trim_fraction)` quantile of the
      currently-kept residual norms. The quantile uses linear
      interpolation between order statistics (numpy's default), which is
      contractual — the first consumer's parity depends on it.
3. **Used images.** An image is *used* iff it has ≥ 4 kept observations
   after the final round.

Each sub-fit is an exact linear least-squares solve; the solver must be
deterministic. The whole algorithm is deterministic: fixed round count, no
randomness, no iteration-order dependence in the results (per-image and
per-cluster fits are independent within a sweep).

## Metric upgrade

Operates on the factorization's cameras and used-image mask.

1. Solve for the symmetric `Q = A·Aᵀ` (6 unknowns) by linear least
   squares over, for each used image with rows `m1, m2`:
   `m1ᵀQm1 − m2ᵀQm2 = 0` and `m1ᵀQm2 = 0` (equal-norm and orthogonal
   rows), plus one normalization row setting the mean of
   `m1ᵀQm1 + m2ᵀQm2` to 2 (mean squared row norm 1, excluding the trivial
   `Q = 0`).
2. Eigendecompose `Q`; clamp eigenvalues below `1e-8 × λ_max` up to that
   floor; `A = V·√Λ`.
3. The two reflection hypotheses are `A` and `A·diag(1, 1, −1)`.
4. Per hypothesis and used image: with `m = M_i·A`, the scale is the mean
   of the two row norms; the rotation is the orthonormalization (SVD, with
   determinant corrected to +1) of the 3×3 stack `[m/s; r3]` where
   `r3 = normalize(m1 × m2)`.

Returned per hypothesis: the gauge `A`, per-image rotations and scales
(defined only for used images), so consumers can also map points through
`A⁻¹` if they need them in the metric frame.

## Complexity and bounds

Per round, the camera sweep solves N tiny 8-unknown systems and the point
sweep C tiny 3-unknown systems — linear in observations with small
constants. The initialization's dense 2N×C matrix and its SVD dominate
memory and worst-case time; the implementation documents a bound on
`num_images × num_clusters` and errors clearly above it (the intended
inputs are small image groups against a few thousand clusters).

## Rust API

Module: `sfmtool_core::geometry::affine_factorization`. Core stays
I/O-free: raw slices in, result structs out.

```rust
pub struct AffineFactorizationParams {
    pub rounds: usize,       // default 25
    pub trim_fraction: f64,  // default 0.05
}

pub struct AffineFactorization {
    pub cameras: Vec<[[f64; 3]; 2]>,    // M_i, per image
    pub translations: Vec<[f64; 2]>,    // t_i, per image
    pub points: Vec<[f64; 3]>,          // X_c, per cluster (affine frame)
    pub residuals: Vec<[f64; 2]>,       // per observation, final round
    pub keep: Vec<bool>,                // per observation
    pub used_images: Vec<bool>,         // >= 4 kept observations
}

pub fn factorize_affine(
    obs_clusters: &[u32],
    obs_images: &[u32],
    obs_xy: &[[f64; 2]],
    num_images: usize,
    num_clusters: usize,
    params: &AffineFactorizationParams,
) -> Result<AffineFactorization, FactorizationError>;

pub struct MetricHypothesis {
    pub gauge: [[f64; 3]; 3],           // A
    pub rotations: Vec<[[f64; 3]; 3]>,  // per image; identity where unused
    pub scales: Vec<f64>,               // per image; 0 where unused
}

/// Both reflection hypotheses; `None` when no image is used or the
/// constraint system is degenerate.
pub fn metric_upgrade(
    factorization: &AffineFactorization,
) -> Option<[MetricHypothesis; 2]>;
```

Validation errors: non-parallel arrays, indexes out of range, and the
documented size bound.

## Bindings

Functions in `sfmtool._sfmtool.geometry` returning result pyclasses with
getters; no Python wrapper layer.

```python
fac = factorize_affine(obs_clusters, obs_images, obs_xy,
                       num_images, num_clusters,
                       rounds=25, trim_fraction=0.05)
fac.cameras        # numpy (N, 2, 3) float64
fac.translations   # numpy (N, 2)
fac.points         # numpy (C, 3)
fac.residuals      # numpy (K, 2)
fac.keep           # numpy (K,) bool
fac.used_images    # numpy (N,) bool

hyps = fac.metric_upgrade()   # None, or a pair of hypothesis objects
hyps[0].gauge       # numpy (3, 3)
hyps[0].rotations   # numpy (N, 3, 3)
hyps[0].scales      # numpy (N,)
```

Array arguments accept the dtypes the loading paths produce (u32 indexes,
f64 positions).

## Validation

- Unit tests (core): a synthetic affine scene recovered exactly up to
  gauge (compare `M·X + t` against the clean measurements, and the metric
  hypotheses' rotations against ground truth after global rotation
  alignment, one hypothesis matching); missing-data patterns; planted
  outliers rejected by trimming with inliers' fit unaffected; determinism
  (bitwise-identical repeat runs); sub-minimum images/clusters keep their
  previous values; error cases.
- Bindings test (`tests/rust_bindings/`): numpy round-trips and parity of
  the full pipeline (factorize + metric upgrade) against a numpy reference
  implementation on a synthetic fixture.
- First consumer: `exp_pinhole_bootstrap.py` swaps its `als_factorize()`,
  `metric_upgrade()`, and `weak_perspective_poses()` for the bindings —
  campaign parity on seoul is the acceptance test.

## Open questions

- Whether monotone descent of the kept-set squared error between rounds
  should be contractual or merely expected (trimming changes the objective
  between rounds, so the clean statement is per-sweep exactness).
- Rank-3 initialization alternatives to the mean-filled SVD for very
  sparse observation patterns.
- Whether the camera/point sub-solves should use normal equations or an
  orthogonal decomposition; parity with the first consumer constrains the
  results, not the method.
