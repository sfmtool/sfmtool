# Point or Bearing by Likelihood Ratio

**Status:** Draft. Proposes moving the four rules that decide whether a track is
a finite point or a bearing onto one likelihood-ratio test. The test's core
primitives exist and are specified in
[core/reconstruction/batch-triangulation-api.md](../core/reconstruction/batch-triangulation-api.md)
§ "Point or bearing", as are the measured noise level, the test over a
reconstruction's points and the Python bindings. The reports that print the
test beside `inverse_depth_z` are specified in
[cli/reconstruction/analyze-command.md](../cli/reconstruction/analyze-command.md)
§ "Depth Reliability" and
[cli/reconstruction/inspect-command.md](../cli/reconstruction/inspect-command.md).
Reclassification (`classify_points_at_infinity`) decides on the test, and how
σ is measured is settled; both are in the standing spec (§ "Consumers" and §
"The measured noise level"). This draft covers what remains: discovery, the
bench and bundle adjustment. Amends
batch-triangulation-api.md (the inverse-depth z test, its pre-filter and its
noise floor),
[core/reconstruction/triangulation-rules.md](../core/reconstruction/triangulation-rules.md)
(the `floor` rule),
[core/geometry/bundle-adjustment.md](../core/geometry/bundle-adjustment.md)
(§ "Free points: crossing between representations") and
[core/bench/editable-track.md](../core/bench/editable-track.md)
(§ "Finite points and bearings"). Decided: the statistic, the split between
deciding and placing, the order of the steps, and σ for a stored
reconstruction. Not decided: the default threshold, σ in bundle adjustment,
and what discovery does with a track its cameras cannot resolve (see "Open
questions").

## Purpose

A track in a reconstruction is stored either as a finite point (`w = 1`) or as
a bearing, a direction with no depth (`w = 0`). Deciding which one to store is a
question about the photographs: does giving the track a depth explain where it
was seen better than a direction alone does, by more than measurement noise
could explain? This draft proposes answering that question directly. Fit both
models to the track's observations, measure how much the extra depth parameter
reduces the residual, and compare that reduction with what noise alone would
produce. That comparison is a likelihood-ratio test between two nested models.
It is one number per track, it needs one noise level, and it is the same
number whether it is read by a reclassification pass, by discovery, by the
bench or inside bundle adjustment.

## The problem

### Four rules answer the same question differently

| Where | Rule before this draft | Reads |
|---|---|---|
| `classify_rays_at_infinity` ([convert.rs](../../crates/sfmtool-core/src/analysis/infinity/convert.rs)), used by `find_points_at_infinity` and the bench (and by `classify_points_at_infinity` until it moved to the test) | condition number below `1e4` is finite; otherwise `resolvable_distance < finite_horizon` is indeterminate; otherwise `inverse_depth_z < 4` (or behind a camera) is a bearing | linear midpoint solve; per-ray noise `max(point error, 1 px) / f` |
| The `floor` rule in [triangulation-rules.md](../core/reconstruction/triangulation-rules.md), used by bundle adjustment's crossing and by demotion | the widest ray pair subtending less than `θ_floor` is a bearing | pairwise ray angles; `θ_floor = noise_floor_scale · s / f` with `s` the stage's `loss_scale` |
| `classify_track_rays` ([bench/classify.rs](../../crates/sfmtool-core/src/bench/classify.rs)) | the z rule above, then overridden when the midpoint's RMS reprojection error is not under `0.8×` the mean bearing's (and by more than the noise floor), or the reverse | the midpoint point and the mean ray, neither fitted to minimise pixel error |
| `COINCIDENT_CAMERA_FRACTION` in `classify_points_at_infinity` (removed with that pass's move) | observing cameras spanning under `1e-4` of the camera extent make the track a bearing | camera centres only |

Each rule is a proxy for "does a depth explain the pixels". The bench's override
exists because the z rule was seen to disagree with the pixels; its own comment
notes that neither candidate it compares is fitted, so its `0.8` ratio has to
cover the slack.

### A measured case

On the Kerry Park ground-truth candidate `tk117`, point `pt3d_a9665942_298`
(10 views) is stored as a bearing. Fitting both models to its keypoints by
minimising pixel error:

| | Bearing | Finite point (515 units from the centroid of all cameras) |
|---|---|---|
| Mean / max reprojection error | 0.62 / 1.55 px | 0.16 / 0.32 px |
| Sum of squared residuals | 6.53 px² | 0.37 px² |

The finite point explains every view to a third of a pixel, so the photographs
place the track at a depth. The z rule calls it a bearing, though. Its rays span
about 1.2°, so the condition number is 33,532 and it skips the pre-filter. Its
per-ray noise is the 1 px floor, which gives `z = 2.23` at the point fitted to
the pixel residuals above, under the cutoff of 4 (`sfm analyze
--depth-reliability` reports 2.18, at the plain least-squares fit of the sine
residuals, 511 units from the observing cameras' centroid).
The reconstruction's measured per-axis noise is 0.216 px, so the floor overstates
this track's noise roughly five-fold, and z by the same factor. The stored error
that feeds `max(error, floor)` is also the bearing's own residual once the point
has been demoted, so a demoted point stays demoted on a second pass.

The same computation over all twelve bearings in that file, with σ = 0.216 px
(the RMS per-axis residual over the 3,510 observations of finite points):

| Point | Views | Finite distance (from all cameras' centroid) | SSE bearing → finite (px²) | Λ |
|---|---|---|---|---|
| 298 | 10 | 515 | 6.53 → 0.37 | 132 |
| 294 | 21 | 1,078 | 4.90 → 1.02 | 84 |
| 295 | 18 | 1,159 | 2.53 → 1.09 | 31 |
| 269 | 7 | 2,810 | 0.89 → 0.23 | 14 |
| 270 | 17 | 4,847 | 1.16 → 0.62 | 12 |
| 153–157, 176, 268 | 21–26 | 4,600–13,000 | ≈1.0 → 0.5–1.1 | 1.3–9.5 |
| *finite points 10, 300, 200, for scale* | 5–9 | 3–81 | | *2,762 to 485,654* |

Λ here is from fits of both models to the pixel residuals through the camera
model. Three bearings are clearly finite, seven are clearly bearings, and two sit
between 10 and 25. Units are the file's own (it has no metric scale yet).
`sfm analyze --depth-reliability` measures the distance from the centroid of
the point's observing cameras instead, which gives 511, 1,076 and 1,160 for the
first three.

## The primitives

`bearing_score` / `bearing_score_batch` decide: they fit the bearing in closed
form and evaluate the score statistic for an inverse depth there, with no
iteration. `fit_point_and_bearing` / `fit_point_and_bearing_batch` place: they
run the iterative point fit and return Λ exactly. `is_finite` reads the verdict
off a score against a threshold. Their interface, the reasons for its shape, the
theory of the two nested models and the fit are in
[batch-triangulation-api.md](../core/reconstruction/batch-triangulation-api.md)
§ "Point or bearing"; the sections below cover the measurements that motivate
deciding on them and how each consumer moves.

A consumer holding a reconstruction builds its rays with `track_rays` (or
`observation_ray` for one sighting) from `(image, pixel)` observations, at the
noise level `SfmrReconstruction::reprojection_noise_px` measures. Discovery
builds its candidate tracks from `.sift` keypoints that way and the bench its
uncommitted sightings, then calls the batch functions on them; the reports use
`SfmrReconstruction::point_or_bearing_scores`, the same construction over
stored points, and reclassification builds the stored points' rays with
`track_rays` itself. All of these, and their Python bindings, are in the same
spec (§ "The measured noise level", § "Over a reconstruction", § "Consumers"
and § "Python bindings").

## Theory

### Three forms of one test

A test of a nested parameter has three classical forms, and each is evaluated
at a different place:

| Form | Evaluated at | Here |
|---|---|---|
| **Wald** | the full model's fit | `inverse_depth_z`, today's rule: the fitted depth over its linearised standard deviation |
| **Likelihood ratio** | both fits | Λ: needs the iterative point fit |
| **Score** | the restricted model's fit only | `depth_score`: the gradient of the point model's cost with respect to ρ at the bearing, normalised by the Gauss-Newton curvature there |

With the same σ the three agree asymptotically, and they differ in where their
approximations are accurate. Wald linearises around the fitted depth, which is
worst near infinity, where the depth's uncertainty is lopsided and the fitted
depth is itself mostly noise. That is the regime where the decision is close.
The score test linearises around ρ = 0, which is exactly that regime, and in
inverse depth the point model is close to linear there. Measured on `tk117`
with σ = 0.216 px, the score from `bearing_score` (sine residuals, per-ray
weights from `observed_ray`; see "The noise weight per ray") against Λ from a
full fit of both models to the pixel residuals through the camera model:

| Point | Views | Score (`bearing_score`) | Λ (pixel residuals) |
|---|---|---|---|
| 154 | 26 | 1.33 | 1.32 |
| 153, 155 | 25 | 4.19, 4.21 | 4.19, 4.21 |
| 157 | 25 | 6.76 | 6.76 |
| 268 | 26 | 7.94 | 7.95 |
| 176 | 21 | 8.27 | 8.27 |
| 156 | 24 | 9.47 | 9.46 |
| 270 | 17 | 11.63 | 11.62 |
| 269 | 7 | 14.34 | 14.34 |
| 295 | 18 | 30.98 | 30.93 |
| 294 | 21 | 83.30 | 83.23 |
| 298 | 10 | 131.97 | 132.04 |
| 10 (finite, 81 units) | 9 | 2,732 | 2,762 |
| 50 (finite) | 13 | 79,711 | 85,331 |

The two agree to the first decimal wherever the verdict is in question, and
part for near points, where both are thousands of times the threshold. They
part completely for tracks whose rays spread over a wide angle: there the
bearing describes nothing, and the score at `ρ = 0` can be 0 while `Λ` is in
the hundreds of thousands (tk117 point 91, 4 views over 105°: score 0, `Λ`
635,749 from `fit_point_and_bearing` on the sine residuals, and 1,267,297
from pixel residuals as in the table).
So the decision needs the score for near-parallel rays and the midpoint bound
below for wide ones, and both need only the bearing and one linear solve.

### What the score replaces

- **The condition-number pre-filter.** `Λ ≤ bearing_cost`, so a track whose
  `bearing_cost` is under the threshold is a bearing whatever the point fit
  would find. On `tk117` this alone settles 153, 176 and 269 without the score.
  It is the rigorous form of today's condition-number pre-filter: `λ_min` of
  the unweighted `A = Σ(I − dᵢdᵢᵀ)` is the best bearing's cost in square
  radians, and `λ_max ≈ K`, so `cond(A) = λ_max / λ_min` is roughly the view
  count over the bearing cost with no noise model. That is why its threshold
  drifts with track length.
- **The midpoint bound is the early exit for finite points.** A finite
  candidate's cost at the weighted linear midpoint gives a rigorous lower
  bound on Λ (`bearing_cost − cost`). For near-parallel rays the midpoint is a
  poor candidate (on `tk117` it costs more than the bearing for most far
  points), and the score decides. For rays spread over a wide angle the score
  says nothing and the bound is what decides: on `tk117` it calls 12 of the
  375 finite points finite that the score alone would call bearings, and on
  synthetic object-centric arcs it decides every one the score misses.
  Today's rule reaches those points through the condition-number pre-filter,
  which calls any well-conditioned track finite.

Per track, deciding therefore costs the same order as `triangulate_batch` plus
`depth_uncertainty_batch`, which today's rule already runs, and with no
iteration. Only the tracks that come out finite and need a refined position run
the point fit, warm-started from the position the caller holds.

### Relation to the inverse-depth z test

The current `inverse_depth_z` is the Wald form above. Its square approximates
`Λ` when both use the same σ: for point 298 at 1 px noise, `z² = 5.0` and
`Λ = 6.2`. Two things separate them in practice:

- **Noise.** The z rule's per-ray noise is `max(point error, 1 px)`. The floor
  exists because a short track's own error understates the noise: its few
  views are fitted almost exactly whatever the depth. A reconstruction-wide σ
  from the observations of all finite points has no such bias, so the floor is
  not needed. The threshold on the score already charges a short track for its
  extra parameter.
- **Where it linearises.** As above: at the fitted depth, the worst place for a
  far point.

### Where σ comes from

- **A stored reconstruction** (reclassification, discovery, the bench,
  `analyze`): `SfmrReconstruction::reprojection_noise_px`, the RMS per-axis
  pixel residual over the observations of its finite points with gross
  outliers gated out, and no degrees-of-freedom correction. The standing spec
  § "The measured noise level" has the estimator, the comparison with the
  plain RMS, trimmed RMS and robust spread that chose it, and the reasons;
  each consumer's step makes it that consumer's default.
- **Bundle adjustment**: the same RMS over the round's kept observations of
  finite points at the round's state, measured at re-estimation. Not
  `loss_scale`, which is a schedule constant chosen per stage and not a
  measurement.

### The noise weight per ray

Two questions this draft first left open are settled by measurement on
`tk117`, at σ = 0.216 px. The Kerry Park fisheyes stretch pixels per radian by
up to 2.1 to 1 between the radial and tangential directions, and one scalar
angular noise per ray misplaces the statistic: for point 298, 160.9 with the
geometric mean of the two stretches and 100.8 with σ_px over the focal length,
against 132 from pixel residuals through the camera model. The primitives
therefore take a 2×3 world-frame weight per ray, built by `observed_ray` from
the camera model's projection derivative, and with it the sine residuals and
the eigenvector bearing reproduce the pixel-residual table above:

| Point | Score | Λ | Pixel reference Λ |
|---|---|---|---|
| 298 | 132.0 | 132.0 | 132.0 |
| 294 | 83.3 | 83.3 | 83.2 |
| 295 | 31.0 | 30.9 | 30.9 |
| 269 | 14.3 | 14.4 | 14.3 |
| 270 | 11.6 | 11.6 | 11.6 |
| 10 (finite) | 2,732 | 2,746 | 2,762 |

Over the file, no finite point gets a bearing verdict, and of the twelve
bearings the three in the first rows are finite at a threshold of 25.

### Interaction with the `indeterminate` state

The batch triangulation spec drops (in discovery) a track whose observing
cameras are too close together to place a point even at the capture's own
scale; reclassification, which used to leave such a point alone, now has no
third state. `Λ` answers the case of near-parallel rays from one stop: a near
point and a bearing explain them equally well, `Λ` is near zero, and the track
is a bearing, which is an accurate description of what the rays say. The same
holds for rays from one centre that agree on a direction, as from a camera
panning in place. Rays from centres that a solver collapsed onto one point,
diverging because the poses kept their rotations, fit no bearing; their
centres differ only by round-off, which the test treats as one centre, so they
too get no depth score and a bearing verdict (standing spec § "Fitting").
Whether discovery should still
drop such tracks, rather than add them as bearings, is a policy question left
open below; `resolvable_distance` stays available to answer it.

## Bundle adjustment

Two changes, in order.

1. **Crossing on `Λ`.** At each inter-round re-estimation, the retriangulation
   operation gains a rule beside `floor`, `likelihood`, which runs
   `bearing_score_batch` over the free tracks with σ measured as above and
   writes the verdict of `is_finite` into the mask the next linearisation
   reads. A track that turns finite starts from the solve's own estimate, so
   the next round's solve does the placing and no separate point fit runs. It replaces `floor` in bundle adjustment's call, and
   `noise_floor_scale` is retired from `FreePointPolicy`. The `floor` rule stays
   for callers that want a pure geometric cut.
2. **Inverse-depth free points.** Free points are solved inside each round as
   `(u, ρ)` about a fixed anchor, so a point can move between near and infinity
   within a round instead of only between rounds, and far points stay
   well-conditioned. `ρ = 0` is a value the solve can reach, so the
   representation becomes a storage decision made by `is_finite` at the
   end, not a mode of the solve. This is a larger change to the kernel's
   parameter blocks (3 per point, as today, but in a different
   parametrisation), and it is a separate step.

## Migration

Each step is one PR and keeps the other rules unchanged until its turn. The
reports came first and are in place (see the status line), so the disagreements
each step resolves can be read off `sfm analyze --depth-reliability` before
and after it.

1. **Discovery and the bench.** (Reclassification went first:
   `classify_points_at_infinity` decides on the score and promotes as well as
   demotes, as the standing spec § "Consumers" describes, and
   `COINCIDENT_CAMERA_FRACTION` went with it.) `classify_rays_at_infinity`
   decides on the score. `CONDITION_NUMBER_PREFILTER`, `DEFAULT_INVERSE_DEPTH_Z_CUTOFF`
   and `DEFAULT_NOISE_FLOOR_PX` stop deciding anything. The bench's `0.8` RMS
   ratio and its override variants go, since the criterion now compares fitted
   candidates. The `--find-points-at-infinity` noise-floor component becomes a
   σ override. Two findings from reclassification carry over. A track from
   cameras collapsed onto one centre, with rays that diverge, got a verdict
   decided by the round-off between its centres, so the primitives now treat
   centres equal to round-off as one centre (standing spec § "Fitting"). And
   a finite verdict can come with a fit whose point is on top of a camera or
   whose `Λ` falls short of the score; reclassification applies the consumer
   rule of § "Fitting" (a minimum depth) and requires the fit's `Λ` to reach
   the threshold, and discovery and the bench need the same wherever they
   place a point.
2. **Bundle adjustment crossing** (step 1 of the section above).
3. **Inverse-depth free points in bundle adjustment** (step 2 of the section
   above).

## Parameters

| Parameter | Proposed default | Meaning |
|---|---|---|
| threshold | `25.0` (`DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`) | Score (and Λ) above which a track is a finite point. See "Open questions". |
| σ | measured | RMS per-axis pixel residual over finite points' observations; per round in bundle adjustment. |

## Testing

The primitives' own tests (nesting, calibration against the half-χ²₁ law,
recovery, robustness, the exact bearing, score against Λ, batch parity) are in
[point_or_bearing/tests.rs](../../crates/sfmtool-core/src/reconstruction/triangulation/point_or_bearing/tests.rs).
The measured noise level's tests (a known σ recovered, per camera, from every
observation source, with points at infinity left out) are in
[analysis/reprojection_noise/tests.rs](../../crates/sfmtool-core/src/analysis/reprojection_noise/tests.rs),
and those of the ray construction and the test over a reconstruction (agreement
with the primitives called by hand, alignment to the indexes asked for, tracks
with fewer than two rays) in
[analysis/point_or_bearing/tests.rs](../../crates/sfmtool-core/src/analysis/point_or_bearing/tests.rs).
The consumer steps add:

- **The bench's override cases** become ordinary outcomes: an ill-conditioned
  midpoint that reprojects worse than the bearing gets `Λ ≈ 0`, and a
  well-explained depth with a low z gets a large `Λ`.
- **Regression on real data.** The `tk117` table above, from a checked-in copy
  of the Kerry Park ground truth once it lands.

## Open questions

- **Threshold.** `25` puts 298, 294 and 295 finite and keeps 269 (Λ 14) and 270
  (Λ 12) as bearings. `10.8` makes those two finite as well, at 2,800 and 4,800
  units. Both are defensible on `tk117`; a larger capture with a known far
  field (KerryPark360) should decide it.
- **Per-camera σ.** One σ per reconstruction, or one per camera when the
  cameras differ (the two Kerry Park lenses are close, 0.2155 and 0.2157 px
  from `reprojection_noise`'s per-camera values; other rigs may not be).
- **σ in bundle adjustment.** The stored-reconstruction measure gates
  outliers at 30 robust spreads and applies no degrees-of-freedom correction
  (standing spec § "The measured noise level"). Bundle adjustment measures σ
  over a round's kept observations, after its own outlier handling; whether it
  uses the same gate is for its step to settle.
- **Discovery and single-stop tracks.** Whether `find_points_at_infinity`
  should still drop a track whose observing cameras cannot resolve a depth at
  the capture's scale, or add it as a bearing as `Λ` says.
- **Robust decision.** The decision tier is plain least squares. Whether a
  track whose score clears the threshold should also have to clear it under the
  soft-L1 point fit, so one bad sighting cannot make it finite, or whether
  reconstruction-level outlier trimming upstream is enough.
- **Naming.** `bearing_score`, `BearingScore`, `depth_score`,
  `fit_point_and_bearing`, `PointBearingFit`, `depth_likelihood_ratio` and
  `is_finite` are the names the primitives carry; whether they stand is
  settled, and given a glossary row, when this draft is filed.
