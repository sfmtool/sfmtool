# Structure-Free Focal Vote

**Status:** Implemented (2026-07-27) —
`crates/sfmtool-core/src/geometry/focal_vote.rs` (kernel + consensus) and
`homography_estimation.rs` (4-point LO-RANSAC), tests in the respective
`*/tests.rs`; PyO3 bindings in `crates/sfmtool-py/src/geometry/{focal_vote,
homography_estimation}.rs` (`sfmtool._sfmtool.geometry.focal_vote` /
`estimate_homography`); Python tests in
`tests/rust_bindings/test_focal_vote_rust_bindings.py`.

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
observe, and the consensus focal is the median of the pooled votes from both
families:

- **Epipolar votes** — pairs whose correspondences carry parallax vote the
  Bougnoux focal of a robustly estimated fundamental matrix. Both cameras
  of a pair share the focal, so the two directional Bougnoux focals (from
  `F` and `Fᵀ`) must agree; a pair whose directions disagree casts no
  vote (see Epipolar votes).
- **Rotation votes** — pairs whose correspondences are dominated by a
  parallax-free (far-field) homography vote by rotation self-calibration:
  a parallax-free homography is conjugate to a rotation, `H = K R K⁻¹`, so
  the focal is the `f` that makes `K⁻¹ H K` orthogonal.

Each estimator is degenerate exactly where the other is informative: the
fundamental matrix collapses toward a homography on parallax-free pairs
(Bougnoux votes become arbitrary), and a homography fitted across genuine
parallax is not conjugate to any rotation. Per-pair gates keep each
estimator on its own ground — homography domination and direction
agreement for epipolar pairs, the orthogonality residual for rotation
pairs — and every vote that survives its gate enters one pooled median
(see Consensus).

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
| `focal_px` | `f64?` | Consensus focal (see Consensus); `None` with fewer than 2 pooled votes |
| `family` | enum | Majority contributor to the pool, `Epipolar` or `Rotation` (ties → `Rotation`); `None` when there is no consensus |
| `epipolar_focal_px` | `f64?` | Median of the epipolar pair votes (diagnostic) |
| `rotation_focal_px` | `f64?` | Median of rotation votes (diagnostic) |
| `n_epipolar` | `usize` | Epipolar pair votes entering the pool (one per direction-consistent pair) |
| `n_rotation` | `usize` | Rotation votes entering the pool (one per unordered image pair) |
| `n_pool` | `usize` | Total certified votes (`n_epipolar + n_rotation`) |
| `pool_spread` | `f64` | Interquartile range, in log-focal space, of the votes behind `focal_px` |
| `family_disagreement` | `f64?` | `\|ln(epipolar_focal_px / rotation_focal_px)\|`; `None` unless both families voted |
| `parallax_poverty` | `f64` | Median H/F inlier ratio over the epipolar candidate pairs with at least 16 F inliers (capture-level parallax diagnostic) |
| `epipolar_spread`, `rotation_spread` | `f64` | Interquartile range of each family's pool contributions in log-focal space |
| `epipolar_votes` | list | Every in-band directional Bougnoux focal with its pair covariates: images, shared-cluster count, mean displacement (px), F and H inlier counts, F-vs-Fᵀ direction, focal |
| `rotation_votes` | list | Every accepted rotation vote with its pair covariates: image, partner, mean displacement (px), H inlier count, focal |
| `n_h_dominated`, `n_estimator_failed` | `usize` | Epipolar candidate pairs skipped as homography-dominated; candidate pairs with no usable F (pair-level counts) |
| `n_band_rejected`, `n_degenerate` | `usize` | Directional Bougnoux focals outside the plausibility band; directions where the Bougnoux extraction produced no value (direction-level counts) |
| `n_inconsistent_pairs` | `usize` | Pairs whose two directions disagree, or with only one usable direction, and so cast no vote (pair-level count) |

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
   When the pair has at least 16 F inliers, the ratio `n_H / n_F` feeds
   `parallax_poverty`. When `n_H ≥ max(16, 0.8 · n_F)` the pair is
   homography-dominated: it casts no epipolar vote (its F is collapsing
   toward H).
3. Otherwise compute the Bougnoux focal of both directions of the
   fundamental matrix (existing `focal_from_fundamental`, principal point
   at the image centre). A direction whose extraction is degenerate (no
   value) counts into `n_degenerate`; directional focals not strictly
   inside `(0.2, 4) × max(width, height)` are discarded into
   `n_band_rejected`.
4. The two cameras share the focal, so the two directional focals are two
   measurements of the same quantity. When both are in-band and agree
   within `0.05` in log-focal, the pair casts **one** vote: their
   geometric mean. Otherwise the pair casts no vote — a direction pair
   that disagrees (or has only one usable member) reveals a fundamental
   matrix that does not carry a consistent focal.

## Rotation votes

Candidate pairs: for a sample of images spaced to visit at most 60, the
partner with the largest mean displacement among pairs sharing at least
25 clusters, when that displacement is at least `0.08 × diagonal`.
Small-displacement homographies are near identity and observe no focal.
Each unordered image pair casts at most one rotation vote: when the scan
reaches a pair it has already voted (two images that are each other's
widest partner), the later occurrence is skipped — the inverse homography
over the same correspondences is the same measurement, not a second one.

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

## Consensus

The two families' votes pool into a single population: the epipolar pair
votes (one geometric-mean vote per direction-consistent pair) and the
rotation votes (one per unordered pair). All medians in this kernel are
taken in log-focal space — an even-length median is the geometric mean of
the two central votes, consistent with the agreement band, the spreads,
and the pair vote itself.

- With fewer than 2 pooled votes there is no consensus.
- When both families voted and their medians disagree by more than
  `0.25` in log-focal, the pool is bimodal and its blended median would
  be a value no pair voted for; the consensus is instead the median of
  the majority family (the family with more pool votes, ties →
  `Rotation`).
- Otherwise the consensus is the median of the whole pool.

`family` always reports the majority contributor, so under the
disagreement rule `focal_px` equals the reported family's median.
`pool_spread` is the log-focal interquartile range of the votes behind
`focal_px` (the whole pool, or the majority family's votes under the
disagreement rule), and `family_disagreement` exposes the inter-family
gap itself; callers that need a confidence signal read these two.

Per-pair gating replaces family-level quorums: a pair only enters the
pool when its own geometry certifies the vote — direction agreement for
epipolar pairs, the orthogonality-residual floor and shape for rotation
pairs — so a sparse pool of certified votes is still a consensus. Each
family's median, count, and log-focal spread remain available as
diagnostics, as does `parallax_poverty` (the median `n_H / n_F` over the
epipolar candidate pairs with at least 16 F inliers — high poverty means
most correspondences are explained without parallax, the regime where
callers should expect the pool to be rotation-dominated).

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
  synthetic scenes (near cloud + far cloud) produce a pooled consensus
  whose majority family matches the scene's parallax regime; a scene
  where BOTH families vote pools them into one median; a pair whose two
  directional Bougnoux focals disagree beyond the agreement band casts no
  vote, pinned by a disagreement placed between the band and 2× the band
  (so the test fails if the band loosens); mutual widest-partner images
  produce one rotation vote, not two; a bimodal capture (two
  sub-captures at different focals) triggers the family-disagreement rule
  and returns the majority family's median, never a between-modes blend;
  the tie→`Rotation` rule; homography RANSAC recovers a planted H under
  outlier contamination; seeded determinism.
- Python bindings: array round-trip, dict shape, seed reproducibility,
  and an end-to-end vote on a small fixture agreeing with the Rust
  result.
