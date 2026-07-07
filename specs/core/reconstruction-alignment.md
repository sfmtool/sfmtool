# Reconstruction alignment: Kabsch similarity fit + RANSAC outlier rejection

**Status:** Implemented in
`crates/sfmtool-core/src/analysis/alignment/{kabsch.rs,ransac.rs}`, exposed to
Python as `sfmtool._sfmtool.analysis.kabsch_algorithm_rs` /
`ransac_alignment_rs` (`crates/sfmtool-py/src/analysis/core.rs`). Driven by
`sfm align --method points` (`src/sfmtool/align/by_points.py`) and by the
`sfm xform` alignment operations `--align-to` / `--align-to-input`
(`src/sfmtool/xform/_align_to.py`, `_align_to_input.py`).

## Overview

Aligning one reconstruction to another means estimating a **similarity
transform** (rotation `R`, translation `t`, uniform scale `s`, together an
`Se3Transform`) that maps source-frame 3D points onto their target-frame
counterparts. The point-based alignment pipeline is:

1. Find corresponding 3D point pairs between the two reconstructions via
   shared feature observations — see
   [point-correspondence.md](point-correspondence.md). Pairs involving a
   point at infinity are dropped up front (a `w = 0` point stores a bearing
   direction, not a metric location, and would corrupt the fit).
2. Optionally run RANSAC over the correspondences to reject outlier pairs
   (mismatched or badly-triangulated points).
3. Fit the final similarity transform to the surviving inliers with the
   Kabsch algorithm.

Camera-based alignment (`sfm align --method cameras`) does **not** use this
module — it estimates the similarity from matched camera poses in pure
Python (`align/core.py::estimate_similarity_with_orientations`, weighted
quaternion averaging per Markley et al.).

## Kabsch algorithm (`kabsch.rs`)

`kabsch_algorithm(source_points, target_points, n_points) ->
Result<Se3Transform, String>` computes the least-squares similarity aligning
`n` uniformly-weighted source points to target points. Points are flat
`&[f64]` slices of length `3 * n` (row-major `x0, y0, z0, x1, …`).

The classical SVD construction:

1. **Centroids** of both sets; both sets are centered.
2. **Cross-covariance** `H = Σ w · (sᵢ − s̄)(tᵢ − t̄)ᵀ` with uniform
   `w = 1/n`.
3. **SVD** `H = U Σ Vᵀ`, then `R = V Uᵀ`.
4. **Proper-rotation correction:** if `det(R) < 0`, the third row of `Vᵀ` is
   negated before recomputing `R` (reflection → rotation).
5. **Scale** `s = Σ w · tᵢ' · (R sᵢ') / Σ w · sᵢ' · sᵢ'` (primes are the
   centered points). When the source points have zero variance (single point
   or all coincident), scale is undetermined and defaults to `1.0`.
6. **Translation** `t = t̄ − s · R · s̄`.

### Rank-deficient input

For degenerate configurations the SVD's nullspace columns are arbitrary
orthonormal completions, so the raw `V Uᵀ` could contain an arbitrary
rotation about the unconstrained axes. The implementation fixes only the
genuinely unconstrained cases, keyed off the numerical rank of `H`
(singular values relative to the largest, threshold `1e-10`):

- **Rank 0** (single point / all coincident): `R = I`.
- **Rank 1** (collinear points): the two nullspace columns of `V` are forced
  to match `U`'s, so the rotation about the constrained axis is identity.
- **Rank 2+**: fully determined by SVD + the det correction (the third axis
  is fixed by orthonormality); left untouched.

### Errors

Returns `Err` on `n_points < 1`, slice length ≠ `n_points * 3`, or SVD
failure. Note the Rust core accepts `n = 1` (yielding `R = I`, `s = 1`, pure
translation); the Python wrapper `align/core.py::kabsch_algorithm` demands
`n ≥ 2`.

## RANSAC (`ransac.rs`)

`ransac_alignment(source_points, target_points, n_points, max_iterations,
threshold, min_sample_size, seed) -> Vec<bool>` returns a per-correspondence
inlier mask. Standard hypothesize-and-verify:

1. Sample `min_sample_size` indices without replacement (seeded `StdRng` —
   deterministic for a given `seed`).
2. Fit a similarity to the sample with `kabsch_algorithm`; degenerate samples
   (Kabsch error) are skipped.
3. Transform **all** source points; a pair is an inlier when the Euclidean
   distance to its target is `< threshold`.
4. Keep the mask with the highest inlier count over `max_iterations`.

If no iteration produces a single inlier (or every sample is degenerate),
the initial all-`true` mask is returned — i.e. "no consensus found" degrades
to "keep everything" rather than "reject everything".

## Python surface and CLI parameterization

Binding defaults (`analysis/core.rs`): `max_iterations=1000`,
`threshold=0.1`, `min_sample_size=3`, `seed=42`.

`sfm align --method points`
(`align/by_points.py::estimate_alignment_from_points`) wires it up as:

- **Threshold is data-derived, not fixed:** a preliminary Kabsch fit over
  *all* correspondences is computed, and the threshold is the
  `--ransac-percentile` (default 95.0) percentile of its residual
  distances.
- `--ransac-iterations` (default 1000) → `max_iterations`; `seed` is fixed
  at 42 for reproducibility; `min_sample_size` stays at the binding default 3.
- RANSAC is skipped when `--no-ransac` is given or when the correspondence
  count is not above `min_points` (default 10). Fewer than `min_points`
  correspondences — before or after RANSAC — is an error.
- The final transform is a Kabsch re-fit over the inliers only; the reported
  RMS error and confidence are computed from the inlier residuals.

See [`specs/cli/align-command.md`](../cli/align-command.md) for the CLI
surface and multi-reconstruction ordering, and
[point-correspondence.md](point-correspondence.md) for how the input pairs
are found.

## Testing

Sibling `tests.rs` files under `analysis/alignment/kabsch/` and
`analysis/alignment/ransac/` cover exact recovery of known transforms,
rank-deficient inputs (identical points, collinear points), reflection
avoidance, scale recovery, and RANSAC inlier/outlier separation with
deterministic seeds.
