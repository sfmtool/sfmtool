# Structure-Free Focal Vote

**Status:** Implemented (2026-07-18) —
`crates/sfmtool-core/src/geometry/focal_vote.rs` (kernel + arbitration) and
`homography_estimation.rs` (4-point LO-RANSAC), tests in the respective
`*/tests.rs`; PyO3 bindings in `crates/sfmtool-py/src/geometry/{focal_vote,
homography_estimation}.rs` (`sfmtool._sfmtool.geometry.focal_vote` /
`estimate_homography`); Python tests in
`tests/rust_bindings/test_focal_vote_rust_bindings.py`. The reference
`scripts/exp_fast_pinhole.py` calls the native kernel.

> _Deviation (2026-07-18): the pair table accumulates the **true**
> shared-cluster count and mean displacement over every covisible member pair
> of each cluster, in one pass, rather than the single uniformly-sampled member
> pair per cluster the Pair-tables section describes. The sampled single-pair
> count undercounts covisibility so severely that the `30`/`25`-cluster
> thresholds never reach quorum on parallax-poor captures — the target capture
> `20240614_225938434` fell one vote short in each family and produced no
> consensus. The true count is what the original script gated on
> (`build_covisibility`), and restores the expected Rotation-family selection
> (structure refines to −0.6% of the ground-truth focal). The pass is
> deterministic, so the pair table no longer consumes the seed; the seed still
> drives the RANSAC estimators._

## Overview

`focal_vote` estimates a shared focal length from cluster-track observations
without building any reconstruction. Image pairs vote independently through
one of two estimators, chosen per pair by what the pair's geometry can
observe, and the consensus focal is the median of the winning vote family:

- **Epipolar votes** — pairs whose correspondences carry parallax vote the
  Bougnoux focal of a robustly estimated fundamental matrix.
- **Rotation votes** — pairs whose correspondences are dominated by a
  parallax-free (far-field) homography vote by rotation self-calibration:
  a parallax-free homography is conjugate to a rotation, `H = K R K⁻¹`, so
  the focal is the `f` that makes `K⁻¹ H K` orthogonal.

Each estimator is degenerate exactly where the other is informative: the
fundamental matrix collapses toward a homography on parallax-free pairs
(Bougnoux votes become arbitrary), and a homography fitted across genuine
parallax is not conjugate to any rotation. The per-pair split plus a
capture-level arbitration keeps each estimator on its own ground.

Because no structure is estimated, no bas-relief-type ambiguity can bias
the result: the vote is an independent witness that callers can hold
structure-based focal estimates against.

## Inputs

Flat observation arrays over track clusters (the same layout the patch and
matching modules use):

| Input | Type | Description |
|---|---|---|
| `cluster_indexes` | `u32 [n_obs]` | Cluster id per observation, nondecreasing |
| `image_indexes` | `u32 [n_obs]` | Image id per observation |
| `positions_xy` | `f64 [n_obs, 2]` | Full-pixel keypoint position |
| `width`, `height` | `u32` | Shared image size; the principal point is the image centre |
| `seed` | `u64` | RANSAC seed; identical inputs and seed reproduce identical output |

Observations must reference at least two images. Clusters with fewer than
two member images contribute nothing.

## Output

| Field | Type | Description |
|---|---|---|
| `focal_px` | `f64?` | Consensus focal, `None` when neither family reaches quorum |
| `family` | enum | `Epipolar` or `Rotation` — which family produced `focal_px` |
| `epipolar_focal_px` | `f64?` | Median of epipolar votes (diagnostic) |
| `rotation_focal_px` | `f64?` | Median of rotation votes (diagnostic) |
| `n_epipolar`, `n_rotation` | `usize` | Vote counts per family |
| `parallax_poverty` | `f64` | Median H/F inlier ratio over epipolar pairs (see Arbitration) |

## Pair tables

One sampled pass over the clusters produces, for every covisible image
pair, a shared-cluster count and a mean feature displacement: each cluster
with two or more member images contributes one uniformly sampled member
pair (skipping same-image pairs); displacements accumulate per image pair.
All later pair selection reads these tables; nothing depends on image
ordering.

## Epipolar votes

Candidate pairs: rank covisible pairs by shared-cluster count, descending,
keeping pairs with at least `min_shared` clusters (`30`, relaxing to `16`
when fewer than 6 pairs qualify) and mean displacement of at least
`0.02 × diagonal`; admit at most 2 pairs per image, up to 18 pairs.

Per pair, over the shared clusters' correspondences:

1. Estimate the fundamental matrix (existing `estimate_fundamental`,
   `max_error_px = 3.0`); record the inlier count `n_F`.
2. Fit a homography to the same correspondences (see Homography
   estimation) at the same 3 px gate; record the inlier count `n_H`.
   The ratio `n_H / n_F` feeds the arbitration. When
   `n_H ≥ max(16, 0.8 · n_F)` the pair is homography-dominated: it casts
   no epipolar vote (its F is collapsing toward H).
3. Otherwise both directions of the fundamental matrix cast a Bougnoux
   focal vote (existing `focal_from_fundamental`, principal point at the
   image centre); votes outside `[0.2, 4] × max(width, height)` are
   discarded.

## Rotation votes

Candidate pairs: for a sample of images spaced to visit at most 60, the
partner with the largest mean displacement among pairs sharing at least
25 clusters, when that displacement is at least `0.08 × diagonal`.
Small-displacement homographies are near identity and observe no focal.

Per pair, over the shared clusters' correspondences (centred on the
principal point):

1. Fit a homography; require at least 12 inliers.
2. Scan the orthogonality residual over `f` on a 48-point log grid
   spanning `[0.3, 4] × max(width, height)`:
   `cost(f) = ‖ G / (tr G / 3) − I ‖_F` with
   `G = M Mᵀ`, `M = K⁻¹ H K`, `K = diag(f, f, 1)`.
   The residual floor validates the homography as a conjugate rotation —
   a finite-plane homography carries a translation term and never gets
   orthogonal — and the residual's shape validates observability: a
   roll-only or too-small rotation is flat in `f`.
3. Reject when `cost(f*) > 0.15` or `2 · cost(f*)` exceeds the median
   cost over the grid. Otherwise refine `f*` by parabolic interpolation
   in `log f` over the bracketing grid points and cast the vote, subject
   to the same plausibility band as epipolar votes.

## Homography estimation

`estimate_homography` joins the geometry module as a public primitive
beside `estimate_fundamental`, with the same RANSAC shape: seeded minimal
sampling (4-point DLT), symmetric transfer error gating, local refit on
the consensus set, and a `{h_matrix, inliers, iterations}` result. Inputs
are two `f64 [n, 2]` correspondence arrays and `max_error_px`.

## Arbitration

`parallax_poverty` is the median `n_H / n_F` over the epipolar candidate
pairs. High poverty means most correspondences are explained without
parallax — the regime where Bougnoux votes are structurally degraded.

- `n_rotation ≥ 5` **and** `parallax_poverty ≥ 0.55` → the rotation
  median is the consensus.
- Otherwise the epipolar median, requiring `n_epipolar ≥ 8`.
- With fewer than 8 epipolar votes, the rotation median stands in when
  `n_rotation ≥ 6`; else no consensus.

Both thresholds are calibrated jointly: sparse rotation votes at marginal
poverty must not override a healthy epipolar consensus, and a rotation
consensus needs enough independent pairs for its median to be stable.

## Binding

`sfmtool._sfmtool.geometry.focal_vote(cluster_indexes, image_indexes,
positions_xy, width, height, seed=0)` returns a dict mirroring the output
table (`family` as a string, `None` for absent optionals).
`estimate_homography(points1, points2, max_error_px=3.0, seed=0)` is
exposed alongside `estimate_fundamental` and returns
`{"h_matrix", "inliers", "iterations"}` or `None`.

## Determinism

All sampling (pair tables, RANSAC) derives from the input seed; identical
inputs and seed produce identical output on every platform.

## Tests

- Rust: synthetic pure-rotation pairs recover a known focal through the
  orthogonality scan; a finite-plane homography with baseline is rejected
  by the residual floor; a roll-only rotation is rejected as flat; mixed
  synthetic scenes (near cloud + far cloud) arbitrate to the correct
  family on both sides of the poverty threshold; homography RANSAC
  recovers a planted H under outlier contamination; seeded determinism.
- Python bindings: array round-trip, dict shape, seed reproducibility,
  and an end-to-end vote on a small fixture agreeing with the Rust
  result.
