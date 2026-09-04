# Structure-Free Focal Vote

## Overview

The focal vote estimates a shared focal length from cluster-track
observations before any reconstruction exists. Image pairs vote independently
through one of two estimators, chosen per pair by what the pair's geometry
can observe, and the consensus focal is the median of the pooled votes from
both families:

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

This document describes the diagnostic layer. A caller that wants one
camera rather than a table of columns and per-pair certificates calls
[estimate-intrinsics.md](estimate-intrinsics.md), which composes this vote
and returns the verdict, its corroboration, the focal and the votes that
belong to that verdict.

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

One pass over the cluster runs produces, for every covisible image pair, a
shared-cluster count and a mean feature displacement. Within a cluster each
member image contributes a single position — the last observation carrying
that image wins — and every pair of distinct member images adds one to that
image pair's count and its feature separation to the pair's displacement
sum. The counts are therefore the true shared-cluster covisibility over the
whole cluster set, enumerated exhaustively rather than estimated: one
uniformly sampled member pair per cluster undercounts covisibility far
enough that the downstream `30`-cluster epipolar and `25`-cluster rotation
thresholds cannot reach quorum on parallax-poor captures, which is exactly
where this estimator has to work. The pass consumes no randomness, so the
tables do not depend on the seed — the seed drives the RANSAC estimators
and the column scans only. All later pair selection reads these tables;
nothing depends on image ordering.

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

## Camera-Model Columns (Equidistant Fisheye)

Both estimator families generalize over the camera model through the
pixel→ray map. A **column** is a camera model hypothesis supplying an
invertible map from pixels to unit rays, parameterized by its own focal.
(The two ray-space estimators the fisheye cells scan are also exposed as
standalone single-camera primitives — see
[relative-pose.md](relative-pose.md).)

Validation coverage: two calibrated fisheye captures (one Insta360 lens
family at two scales, 211° FOV) and two pinhole controls — the model
verdict is correct on all four, and the pooled focal is within 1.2% of the
best-fit equidistant focal (kerry 138.3 px against 136.9; kp360 276.6 px
against 273.5) — plus three uncalibrated OmniPhotos rotating-rig captures
on a second Insta360 One X body at 1920², all three arbitrated fisheye,
voting 548–584 px (6.6% spread). The gate constants below are the
data-derived values from those captures.

- **Pinhole** — `ray ∝ ((x − cx)/f, (y − cy)/f, 1)`. The implemented
  kernel above is this column; its two cells (epipolar, rotation) have
  closed-form focal extraction (Bougnoux, `K⁻¹HK` orthogonality).
- **Equidistant fisheye** — `θ = r/f` for radial pixel distance `r` from
  the principal point; `ray = (sin θ · cos φ, sin θ · sin φ, cos θ)`.
  The map depends on the focal being estimated, so this column has no
  closed form: each cell scans candidate focals for self-consistency,
  in the same shape as the pinhole rotation cell's orthogonality scan
  (log grid over the plausibility band, floor and shape gates,
  parabolic refinement of the winning bracket).

The two fisheye cells:

- **Epipolar × fisheye.** Per candidate `f`: map the pair's
  correspondences to rays and robustly estimate the ray-space epipolar
  matrix. With the correct `f` that matrix is essential — its two
  non-zero singular values are equal — so the cost is the essentialness
  residual `(σ₁ − σ₂)/(σ₁ + σ₂)` on the consensus set. The floor gate
  rejects pairs whose best residual stays high (no essential explanation
  at any focal); the shape gate rejects flat scans (geometry with no
  focal opinion).

  The pinhole column's homography-domination gate has its analog here,
  and it carries the same meaning — fit the rotation family's model and
  abstain when it explains the pair. The pinhole form (a pixel-space
  homography) does not transfer: under a fisheye map a rotating camera
  induces no pixel homography at any focal. The fisheye form is the
  ray-rotation fit the rotation cell already computes over its coarse
  support sub-grid: when the pair's best rotation consensus over that
  sub-grid reaches `max(16, 0.8 × best essential consensus)`, the pair
  is rotation-dominated and casts no epipolar vote — a near-zero
  baseline makes `E = [t]×R` degenerate and its essentialness minima
  broad, exactly as the pinhole `F` collapses toward `H`. The same
  ratio feeds the column's parallax-poverty diagnostic, mirroring
  `n_H / n_F`.

  The sub-grid pass locates the winning bracket and then sweeps the
  grid points inside it at full resolution. That refinement is
  load-bearing for the gate rather than for the freeze: a parallax-free
  pair's rotation consensus peaks sharply at its own focal, the
  sub-grid's stride can straddle that peak by more than a tenth, and
  the essential consensus it is compared against carries no such bias,
  being read off the full grid. Uncorrected, a pair whose
  correspondences are *entirely* explained by a rotation reports a
  ratio near `0.74` and misses the `0.8` gate.

  The gate is a property of the cell, not of one column: every column
  applies it to its own scans, because certified masses are comparable
  only when the certificates come from the same machinery. It reads
  "does a rotation *under this column's map* explain the pair", so a
  parallax-free capture read through the wrong column is legitimately
  not gated — its rays are not a rotation of each other. On such a
  capture the right column's epipolar mass goes to zero and the model
  verdict rests on its rotation cell alone.

  The pinhole column's direction-agreement certificate has its analog
  here too: the scans of the two correspondence directions
  must locate minima within the epipolar agreement band, else the pair
  casts no vote; when they agree the pair casts **one** vote, the
  geometric mean of the two minimizing focals, exactly as the pinhole
  epipolar family does with its two directional Bougnoux focals. The
  per-direction residual must be **one-sided** — measured against
  `E·x₁` in the second image for one direction and against `Eᵀ·x₂` in
  the first for the other. A symmetric residual makes the certificate
  vacuous: the epipolar matrix of the swapped correspondences is
  exactly the transpose, with identical singular values, so a symmetric
  consensus scores the two directions as one measurement.

  Consensus selection and refinement live on the ray sphere with
  angular residuals (the ray-to-epipolar-plane angle
  `asin(|x₂ᵀ E x₁|)` for unit rays): a fisheye field of view can
  exceed 180°, and rays with `θ ≥ 90°` (backward of the image plane)
  have no planar projection. Hypothesis generation may equivalently run
  on the `z = 1` plane over the sub-hemisphere population — provided
  the population is frozen at the smallest candidate focal
  (`r < f_min · π/2`, so every candidate scores the same point set) and
  the consensus gate and refit remain angular on the sphere. Both
  conditions are load-bearing; with them the two estimators measure as
  equivalent at 211° FOV (with ~20% of correspondences beyond the
  hemisphere), without them the plane path becomes a population or
  periphery artifact. The angular gate's **value** derives per
  candidate focal from a pixel tolerance through the map's local scale
  `dr/dθ` — a fixed angular threshold does not transfer across lenses
  and resolutions (measured angular noise spans 0.03°–0.54° p90 across
  captures) and misclassifies narrow-FOV pinhole captures.
- **Rotation × fisheye.** Per candidate `f`: map both sides to rays and
  fit a rotation directly (robust orthogonal fit on unit rays); the
  cost is the fit's trimmed RMS angular residual. A parallax-free pair
  under the correct `f` is explained by a pure rotation of rays with no
  conjugacy construction needed, and the fit is valid over the full
  sphere — `θ ≥ 90°` rays participate like any others. The inlier
  support is frozen **once per pair** (the largest rotation consensus
  over a coarse sub-grid) and reused at every candidate: both columns'
  maps shrink every ray angle as `1/f`, so with a per-candidate support
  a bad focal buys a low cost by keeping fewer points and the scan has
  no interior minimum (it pins at the top of the grid). Floor and
  shape gates as above.

  The scan **locates** its minimum on the same residual carried through
  the map's local `dr/dθ` (a trimmed RMS in pixels), which removes the
  `1/f` drift both maps share, and the floor **gates** the angular value
  at that minimum, because the equivalent pixel floor doubles with
  resolution while the angular one holds. Frozen supports make the two
  agree to a fraction of a percent; the angular minimum is what an
  unfrozen support destroys.

  The scan band is FOV-derived rather than inherited from the pinhole
  band: under the equidistant map the focal and the field of view are
  tied by `f = r_edge / θ_edge`, so the grid must reach the focals a
  beyond-180° field of view implies at the image's own radius (a 480
  px-wide sensor at 200° FOV implies `f ≈ 137 px`) — the band is
  `[0.075, 3] × max(width, height)` on 64 log-spaced points, whose low
  end sits well under the pinhole plausibility band's `0.2` floor. Both
  columns scan that same band, because certified masses are comparable
  only when the certificates come from the same machinery; the credible
  half-FOV window at a pair's own edge radius (half-FOV in `[50°, 110°]`)
  is not a restriction on the grid but the covariate below.

**Radial coverage.** Pinhole and equidistant maps agree to first order
near the principal point, so a pair whose inliers hug the centre cannot
distinguish the columns regardless of how well it votes within one.
Each vote therefore carries a radial-coverage covariate (a high quantile
of its inliers' radial distance, as a fraction of the half-diagonal),
and only votes above a coverage floor (`0.50` of the half-diagonal at
the inliers' radial p90) are **model-informative**. Votes below the
floor still enter their column's focal pool; they are simply excluded
from the model verdict. Coverage is deliberately **radial, not
angular**: angular reach is what actually predicts discrimination, but
an angular floor disqualifies a narrow-FOV pinhole capture's own
legitimate votes by attrition and flips its model verdict — the radial
covariate penalizes centre-hugging votes without penalizing narrow
lenses.

The certification floors, following the same pattern as the pinhole
gates (each vote certified by its own geometry, no quorums): the
essentialness floor is `0.03` (a validity floor — correct-column costs
sit at p90 ≤ 0.023, and the floor is not the model discriminator, whose
distributions overlap); the rotation-fit floor is `0.02` rad trimmed
RMS (≈1.8× the observed correct-column p90, ~30× under the wrong
column's residuals, and stated in angle because the equivalent pixel
floor doubles with resolution while the angular one holds); the shape
gate (`2·cost(f*) ≤ median over the grid`) and the `0.05`
direction-agreement band carry over from the pinhole cells unchanged.
A certified vote also records whether its unconstrained wide-band
minimum falls inside the credible half-FOV window — on fisheye
captures it does for the large majority of pairs, on pinhole captures
it rarely does, so band containment is a model-evidence covariate
available to the verdict.

**Arbitration hierarchy.** Model precedes motion family: the model
verdict is the column with the greater certified mass of
model-informative votes, and the winning column then applies the
existing two-family consensus (pooled log-space median, the `0.25`
family-disagreement rule) unchanged over its own votes. Certified
masses are comparable **only when both columns' certificates come from
the same scan machinery**: for arbitration purposes the pinhole column
runs the same two self-consistency scans (essentialness, rotation fit)
under the identity ray map and the same gates, while its focal answer
keeps the closed forms above. Counting closed-form certificates against
scan certificates compares incommensurable quantities and is not
defined. Column focals are likewise not blended: a pinhole focal and an
equidistant focal parameterize different maps and only coincide near
the axis. The losing column's median, count, and spread remain as
diagnostics. The rotation cell abstains entirely on captures with no
far-field rotation pairs, so on such captures the model verdict rests
on the epipolar cell alone — the verdict's margin is structurally
thinner on pinhole input than on fisheye input.

Ties in the certified model-informative mass go to the pinhole column,
the narrower hypothesis.

**Compatibility.** The caller selects the column set; the default is
pinhole-only, which reproduces the implemented kernel's behavior
identically — no scan runs at all. Output gains `camera_model` (the
model verdict) and per-column diagnostics mirroring the per-family ones
plus the certificate counts the verdict reads. A single requested column
has nothing to arbitrate and is the verdict by construction; with
several, `camera_model` is `None` when no column has a
model-informative vote, and the top-level focal then falls back to the
pinhole column (or, absent it, the first requested column). The
equidistant column's focal
parameterizes the equidistant map **only**: against a polynomial
fisheye calibration it is the best-fit equidistant focal over the
observed radii, not the calibrated focal (measured ≈6% above a
Kannala-Brandt calibration's `f` on the validation lens). Callers must
not hand it to a consumer expecting another fisheye parameterization's
focal.

## Binding

The kernel and its consensus live in
[focal_vote.rs](../../../crates/sfmtool-core/src/geometry/focal_vote.rs), the
camera-model column scans in
[column_scan.rs](../../../crates/sfmtool-core/src/geometry/focal_vote/column_scan.rs),
and the 4-point LO-RANSAC homography in
[homography_estimation.rs](../../../crates/sfmtool-core/src/geometry/homography_estimation.rs),
bound as `sfmtool._sfmtool.geometry.focal_vote` and `estimate_homography`.

`sfmtool._sfmtool.geometry.focal_vote(cluster_indexes, image_indexes,
positions_xy, width, height, seed=0, epipolar_min_disp_frac=0.02,
columns=None)` returns a dict mirroring the output table (`family` and
`camera_model` as strings, `None` for absent optionals). `columns` is a
sequence of column names (`"pinhole"`, `"equidistant"` / `"fisheye"`);
`None` means the pinhole-only default, which reproduces the closed-form
kernel's dict exactly (`camera_model` is `"Pinhole"` and `columns` is
empty). `estimate_homography(points1, points2, max_error_px=3.0, seed=0)` is
exposed alongside `estimate_fundamental` and returns
`{"h_matrix", "inliers", "iterations"}` or `None`.

## Determinism

The pair tables and every pair selection built on them are exhaustive and
draw no randomness at all; all sampling that remains (the RANSAC
estimators and the column scans) derives from the input seed. Identical
inputs and seed produce identical output on every platform. The column
scans draw their minimal-sample index sets once per candidate pair from
the seed and the pair's position in the candidate list, then reuse them
at every candidate focal, in every cell direction and in every column —
so the cost curves carry no RANSAC jitter, the columns are directly
comparable, and the per-pair scans may run in parallel without affecting
the result.

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
- Rust, camera-model columns: synthetic equidistant scenes (pure
  rotation and parallax) recover a planted fisheye focal through each
  cell; the same fisheye pair read through the pinhole column survives
  only on a centre-hugging subset and so contributes no
  model-informative mass, while a narrow-FOV pinhole scene is the
  pinhole column's own ground; a synthetic pinhole capture is arbitrated
  `Pinhole` and a synthetic fisheye capture `EquidistantFisheye`, with
  the top-level focal equal to the winning column's consensus and never
  a blend; the one-sided direction residual is pinned twice — the two
  directions score a fifth of the points more than 10% apart and certify
  different consensus sets, while the swap onto the transpose reproduces
  the other direction bit for bit (so any symmetric residual, and the
  singular values themselves, are invariant under the swap and the
  certificate would be vacuous); the rotation scan with a deliberately
  unfrozen support pins its angular minimum at the top of the grid where
  the frozen one is interior and on the planted focal; the default
  pinhole-only column set reproduces the closed-form kernel bit for bit
  with the multi-column path present; seeded determinism of the scans.
- Rust, rotation domination: a parallax-free fisheye pair is gated out
  of the epipolar cell with a consensus ratio at the gate or above,
  while the same lens's parallax pair is not gated and still votes; and
  end to end, a capture with no parallax anywhere (the one whose
  ungated epipolar votes dragged the pooled median to 624 px against a
  planted 320) gates every epipolar candidate, leaves the rotation cell
  to carry the verdict, and lands on the planted focal.
- Python bindings: the default-argument dict is exactly the explicit
  pinhole-only dict; an unknown column name raises; a fisheye fixture is
  arbitrated `EquidistantFisheye` with the per-column diagnostic dict
  shape checked; a pinhole fixture's two-column result matches its
  pinhole-only result field by field; seed reproducibility with columns.
- Python bindings: array round-trip, dict shape, seed reproducibility,
  and an end-to-end vote on a small fixture agreeing with the Rust
  result.
