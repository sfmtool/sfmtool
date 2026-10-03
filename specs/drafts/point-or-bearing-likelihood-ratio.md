# Point or Bearing by Likelihood Ratio

**Status:** Draft. Proposes two core primitives, each in a single-track and a
batch form (a cheap decision and an iterative fit), and moving the four rules that decide whether a track is a finite point
or a bearing onto it. Amends
[core/reconstruction/batch-triangulation-api.md](../core/reconstruction/batch-triangulation-api.md)
(the inverse-depth z test, its pre-filter and its noise floor),
[core/reconstruction/triangulation-rules.md](../core/reconstruction/triangulation-rules.md)
(the `floor` rule),
[core/geometry/bundle-adjustment.md](../core/geometry/bundle-adjustment.md)
(§ "Free points: crossing between representations") and
[core/bench/editable-track.md](../core/bench/editable-track.md)
(§ "Finite points and bearings"). Decided: the statistic, the split between
deciding and placing, and the order of the steps. Not decided: the default threshold, how
σ is estimated in each caller, and what becomes of the `indeterminate` state
(see "Open questions").

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

| Where | Rule today | Reads |
|---|---|---|
| `classify_rays_at_infinity` ([convert.rs](../../crates/sfmtool-core/src/analysis/infinity/convert.rs)), used by `classify_points_at_infinity`, `find_points_at_infinity` and the bench | condition number below `1e4` is finite; otherwise `resolvable_distance < finite_horizon` is indeterminate; otherwise `inverse_depth_z < 4` (or behind a camera) is a bearing | linear midpoint solve; per-ray noise `max(point error, 1 px) / f` |
| The `floor` rule in [triangulation-rules.md](../core/reconstruction/triangulation-rules.md), used by bundle adjustment's crossing and by demotion | the widest ray pair subtending less than `θ_floor` is a bearing | pairwise ray angles; `θ_floor = noise_floor_scale · s / f` with `s` the stage's `loss_scale` |
| `classify_track_rays` ([bench/classify.rs](../../crates/sfmtool-core/src/bench/classify.rs)) | the z rule above, then overridden when the midpoint's RMS reprojection error is not under `0.8×` the mean bearing's (and by more than the noise floor), or the reverse | the midpoint point and the mean ray, neither fitted to minimise pixel error |
| `COINCIDENT_CAMERA_FRACTION` in `classify_points_at_infinity` | observing cameras spanning under `1e-4` of the camera extent make the track a bearing | camera centres only |

Each rule is a proxy for "does a depth explain the pixels". The bench's override
exists because the z rule was seen to disagree with the pixels; its own comment
notes that neither candidate it compares is fitted, so its `0.8` ratio has to
cover the slack.

### A measured case

On the Kerry Park ground-truth candidate `tk117`, point `pt3d_a9665942_298`
(10 views) is stored as a bearing. Fitting both models to its keypoints by
minimising pixel error:

| | Bearing | Finite point (515 units out) |
|---|---|---|
| Mean / max reprojection error | 0.62 / 1.55 px | 0.16 / 0.32 px |
| Sum of squared residuals | 6.53 px² | 0.37 px² |

The finite point explains every view to a third of a pixel, so the photographs
place the track at a depth. The z rule calls it a bearing, though. Its rays span
about 1.2°, so the condition number is 33,532 and it skips the pre-filter. Its
per-ray noise is the 1 px floor, which gives `z = 2.23`, under the cutoff of 4.
The reconstruction's measured per-axis noise is 0.216 px, so the floor overstates
this track's noise roughly five-fold, and z by the same factor. The stored error
that feeds `max(error, floor)` is also the bearing's own residual once the point
has been demoted, so a demoted point stays demoted on a second pass.

The same computation over all twelve bearings in that file, with σ = 0.216 px
(the RMS per-axis residual over the 3,510 observations of finite points):

| Point | Views | Finite distance | SSE bearing → finite (px²) | Λ |
|---|---|---|---|---|
| 298 | 10 | 515 | 6.53 → 0.37 | 132 |
| 294 | 21 | 1,078 | 4.90 → 1.02 | 84 |
| 295 | 18 | 1,159 | 2.53 → 1.09 | 31 |
| 269 | 7 | 2,810 | 0.89 → 0.23 | 14 |
| 270 | 17 | 4,847 | 1.16 → 0.62 | 12 |
| 153–157, 176, 268 | 21–26 | 4,600–13,000 | ≈1.0 → 0.5–1.1 | 1.3–9.5 |
| *finite points 10, 300, 200, for scale* | 5–9 | 3–81 | | *2,773 to 487,529* |

Three bearings are clearly finite, seven are clearly bearings, and two sit
between 10 and 25. Units are the file's own (it has no metric scale yet).

## Rust API

The primitives live in
[triangulation.rs](../../crates/sfmtool-core/src/reconstruction/triangulation.rs)
beside `triangulate_batch` and `depth_uncertainty_batch`, and take the same
CSR ray layout, so a caller that already builds rays for them builds nothing
new. Each comes as a single-track function and a batch form over `offsets`,
bound as `sfmtool._sfmtool.analysis.bearing_score_batch` and
`sfmtool._sfmtool.analysis.fit_point_and_bearing_batch`.

There are two tiers, because deciding and placing cost different amounts:

- **`bearing_score`** decides. It fits the bearing in closed form and
  evaluates the score statistic there (see "Theory: three forms of one test").
  Its cost is one pass over the rays and two 3×3 solves, the same order as
  `triangulate_batch`, with no iteration.
- **`fit_point_and_bearing`** places. It runs the iterative point fit, and is
  called only for tracks the decision made finite that need a position (and by
  callers that report the exact Λ, such as the bench and `analyze`).

```rust
/// The bearing that best explains one track's rays, and how strongly the rays
/// ask for a depth.
pub struct BearingScore {
    /// The unit direction minimising Σ wᵢ ‖(I − dᵢdᵢᵀ) u‖², wᵢ = 1/σᵢ²: the
    /// eigenvector of the smallest eigenvalue of M = Σ wᵢ (I − dᵢdᵢᵀ).
    pub bearing: Vector3<f64>,
    /// That smallest eigenvalue: the bearing's cost in units of the noise.
    /// Λ can never exceed it, so `bearing_cost < threshold` decides a bearing
    /// exactly.
    pub bearing_cost: f64,
    /// The score statistic for adding an inverse depth at the bearing,
    /// `gᵀ H⁻¹ g` with g and H the Gauss-Newton gradient and normal matrix of
    /// the point model at (bearing, ρ = 0). 0 when the gradient points to
    /// ρ < 0, which is behind the cameras. Approximates Λ closely wherever Λ
    /// is near any sensible threshold.
    pub depth_score: f64,
    pub num_views: usize,
}

/// Both fits of one track and the exact statistic that compares them.
pub struct PointBearingFit {
    pub bearing: Vector3<f64>,
    pub bearing_cost: f64,
    /// The point minimising Σ wᵢ ‖rᵢ‖² as an anchor plus a direction over an
    /// inverse depth `inverse_depth ≥ 0`. At `inverse_depth = 0` it is the
    /// bearing model, so `point_cost ≤ bearing_cost`.
    pub point: Point3<f64>,
    pub inverse_depth: f64,
    pub point_cost: f64,
    /// `bearing_cost − point_cost`, ≥ 0: Λ, exactly.
    pub depth_likelihood_ratio: f64,
    /// `point` lies in front of every observing camera.
    pub in_front_of_all_cameras: bool,
    pub num_views: usize,
}

pub struct PointBearingFitOptions {
    /// Per-component robust loss. `None` is plain least squares; `Some(c)` is
    /// soft-L1 with scale `c` noise units, the loss bundle adjustment uses.
    pub soft_l1_scale: Option<f64>,
    pub max_iterations: usize,
}

/// `dirs` unit world rays, `centers` camera centres, `sigma_rad` per-ray
/// angular noise (σ_px over the camera's local pixels per radian at that ray).
pub fn bearing_score(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    sigma_rad: &[f64],
) -> Option<BearingScore>;

pub fn bearing_score_batch(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    offsets: &[usize],
    sigma_rad: &[f64],
) -> Vec<Option<BearingScore>>;

/// `start`, when given, is a point to start the point fit from (the stored
/// position, or bundle adjustment's current estimate); otherwise it starts
/// from the bearing at ρ = 0 and from the linear midpoint.
pub fn fit_point_and_bearing(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    sigma_rad: &[f64],
    start: Option<Point3<f64>>,
    options: &PointBearingFitOptions,
) -> Option<PointBearingFit>;

pub fn fit_point_and_bearing_batch(
    dirs: &[Vector3<f64>],
    centers: &[Point3<f64>],
    offsets: &[usize],
    sigma_rad: &[f64],
    starts: Option<&[Point3<f64>]>,
    options: &PointBearingFitOptions,
) -> Vec<Option<PointBearingFit>>;

/// The decision. A bearing when `bearing_cost < threshold` (exact: Λ cannot
/// reach the threshold) or `depth_score < threshold`; otherwise finite.
pub fn is_finite(score: &BearingScore, threshold: f64) -> bool;
```

**Why it is shaped this way.**

- **Deciding is separate from placing.** Every caller needs the verdict for
  every track; only some need a refined position, and only for the tracks that
  came out finite. Most callers already hold a position to start from (the
  stored point, bundle adjustment's current estimate), so the point fit is a
  few warm-started iterations rather than a search.
- **Rays in, angles out, like the rest of the family.** The residual of ray
  `i` is the tangent-plane component `(I − dᵢdᵢᵀ) m` of the model's unit
  direction `m` from that camera, the sine of the angle between them. That
  keeps the primitives pure and IO-free, and independent of the camera model,
  as `triangulate_batch` is: un-projection is the caller's job. The sine
  rather than the tangent is what makes the bearing fit an eigenproblem.
  Converting pixel noise to angle per ray (`sigma_rad`) is where the camera
  model enters; `depth_uncertainty_batch` already takes that input.
- **σ is an input, not a constant.** The noise level is the one policy choice
  in the test, and the callers have different ways to know it: a stored
  reconstruction can measure it from its finite points, and bundle adjustment
  from the current round's residuals. A floor inside the primitive is what
  made the z rule disagree with the pixels.
- **The threshold is a separate argument** of `is_finite`, for the same reason
  `depth_uncertainty_batch` is kept out of `triangulate_batch`.
- **`Option` per track.** Fewer than two finite rays makes neither model
  meaningful; the caller's `few` rule decides what that track becomes.

**Example.**

```rust
use sfmtool_core::reconstruction::triangulation::{
    bearing_score_batch, fit_point_and_bearing_batch, is_finite, PointBearingFitOptions,
};

// dirs, centers, offsets as for triangulate_batch; sigma_px from the
// reconstruction's finite-point residuals; px_per_rad from each ray's camera.
let sigma_rad: Vec<f64> = px_per_rad.iter().map(|s| sigma_px / s).collect();
let scores = bearing_score_batch(&dirs, &centers, &offsets, &sigma_rad);
let finite: Vec<bool> = scores
    .iter()
    .map(|s| s.as_ref().is_some_and(|s| is_finite(s, DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD)))
    .collect();
// Bearings are final: store scores[t].bearing with w = 0. Finite tracks that
// need a position go through the point fit, started from what the caller holds.
```

## Theory

### Two nested models

Parametrise a finite point by an anchor `a` (the centroid of its observing
cameras), a unit direction `u` and an inverse depth `ρ ≥ 0`:
`X = a + u / ρ`. In homogeneous form the direction from camera `i` to the point
is proportional to `u + ρ (a − cᵢ)`, which is finite and smooth through
`ρ = 0`. At `ρ = 0` it is `u` from every camera, which is the bearing model.
The bearing model is the point model with one parameter held at its boundary,
so:

- the point fit, started at the bearing fit, can only lower the cost, and
  `Λ = bearing_cost − point_cost ≥ 0`;
- with Gaussian noise of the stated σ and a track truly at infinity, `Λ`
  follows a χ² distribution with one degree of freedom, halved by the boundary
  (half the time the unconstrained optimum would have `ρ < 0` and the
  constrained one sits at `ρ = 0` with `Λ = 0`). `P(Λ > t) = ½ P(χ²₁ > t)`, so
  `t = 10.8` is a 1-in-2,000 chance of calling a true bearing finite, and
  `t = 25` is roughly 1 in 3.5 million.

The residual of ray `i` is `(I − dᵢdᵢᵀ) m / σᵢ`: the tangent-plane component
of the model's unit direction `m` from that camera against the ray `dᵢ`, which
is the sine of the angle between them, over the ray's angular noise. Over a
track's views this is close to the pixel residual divided by σ_px, since `σᵢ`
is σ_px over each ray's local pixels per radian.

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
with σ = 0.216 px (pixel residuals through the camera model; see "Open
questions"), the score at the bearing against Λ from the full fit:

| Point | Views | Score | Λ |
|---|---|---|---|
| 154 | 26 | 1.3 | 1.3 |
| 153, 155 | 25 | 4.2, 4.2 | 4.2, 4.2 |
| 157 | 25 | 6.8 | 6.8 |
| 268 | 26 | 7.9 | 7.9 |
| 176 | 21 | 8.3 | 8.3 |
| 156 | 24 | 9.5 | 9.5 |
| 270 | 17 | 11.6 | 11.6 |
| 269 | 7 | 14.3 | 14.3 |
| 295 | 18 | 31.0 | 30.9 |
| 294 | 21 | 83.2 | 83.2 |
| 298 | 10 | 132.0 | 132.0 |
| 10 (finite, 81 units) | 9 | 2,749 | 2,762 |
| 50 (finite) | 13 | 79,529 | 85,331 |

The two agree to the first decimal wherever the verdict is in question, and
part only for near points, where both are thousands of times the threshold. So
the decision needs only the score, and the score needs only the bearing.

### Why the decision is cheap

- **The bearing is closed form.** With sine residuals the bearing's cost is
  `uᵀ M u` with `M = Σ wᵢ (I − dᵢdᵢᵀ)`, so the best bearing is the eigenvector
  of `M`'s smallest eigenvalue and the cost is that eigenvalue. `M` is the
  normal matrix `triangulate_batch` already builds, weighted per ray.
- **The score is one more pass.** At `(bearing, ρ = 0)` each ray contributes a
  2×3 Jacobian and its residual; `g` and `H` are 3-vectors and 3×3 matrices
  summed over the rays, and the score is one 3×3 solve.
- **An exact early exit for bearings.** The point model contains the bearing,
  and costs are non-negative, so `Λ ≤ bearing_cost`. A track whose
  `bearing_cost` is under the threshold is a bearing whatever the point fit
  would find. On `tk117` this alone settles 153, 176 and 269 without the score.
  It is the rigorous form of today's condition-number pre-filter: `λ_min` of
  the unweighted `A = Σ(I − dᵢdᵢᵀ)` is the best bearing's cost in square
  radians, and `λ_max ≈ K`, so `cond(A) = λ_max / λ_min` is roughly the view
  count over the bearing cost with no noise model. That is why its threshold
  drifts with track length.
- **No early exit for finite points is needed.** A finite candidate's cost at a
  linear midpoint gives a rigorous lower bound on Λ (`bearing_cost − cost`),
  but for near-parallel rays the midpoint is a poor candidate however it is
  weighted (on `tk117` it costs more than the bearing for most far points), so
  the bound decides only tracks the score decides just as cheaply.

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

### Fitting the point

- **Start** from the caller's position when it has one, converted to
  `(u, ρ)`; otherwise from both the bearing at `ρ = 0` and the linear midpoint
  from `triangulate_batch`, keeping the cheaper result. Gauss-Newton over
  `(u, ρ)` with `ρ` clamped at 0. Starting from the bearing guarantees
  `Λ ≥ 0`.
- **Cheirality.** A point fit that ends behind any observing camera is not a
  physical point, and the track is a bearing, which is today's `BEHIND` rule.
  The score's sign carries the same information early: a gradient pointing to
  `ρ < 0` means the rays converge behind the cameras, and the score is 0.
- **Robust loss.** With `soft_l1_scale` set, the point fit uses the soft-L1
  loss bundle adjustment uses, per residual component, at that scale in noise
  units, so one sighting off by many σ adds about linearly rather than
  quadratically to the cost. The decision tier is plain least squares, so that
  the bearing stays an eigenproblem; a track whose score clears the threshold
  by less than one bad sighting could account for is a case for the robust
  fit (see "Open questions").

### Where σ comes from

- **A stored reconstruction** (reclassification, discovery, the bench,
  `analyze`): the RMS per-axis pixel residual over the observations of its
  finite points. RMS rather than a robust spread, because σ has to cover pose,
  lens-model and keypoint error together, and those are heavier-tailed than a
  Gaussian. On `tk117` the robust spread is 0.137 px against an RMS of 0.216
  px, and at 0.137 px even the 13,000-unit bearings score `Λ` up to 35.
- **Bundle adjustment**: the same RMS over the round's kept observations of
  finite points at the round's state, measured at re-estimation. Not
  `loss_scale`, which is a schedule constant chosen per stage and not a
  measurement.

### Interaction with the `indeterminate` state

The batch triangulation spec drops (in discovery) or leaves alone (in
reclassification) a track whose observing cameras are too close together to
place a point even at the capture's own scale. `Λ` answers that case without a
third state: when the cameras are all at one stop, a near point and a bearing
explain the rays equally well, `Λ` is near zero, and the track is a bearing,
which is an accurate description of what the rays say. Whether discovery should
still drop such tracks, rather than add them as bearings, is a policy question
left open below; `resolvable_distance` stays available to answer it.

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

Each step is one PR and keeps the other rules unchanged until its turn.

1. **The primitives.** `bearing_score`, `fit_point_and_bearing`, their batch
   forms, `is_finite`, the Python bindings, and a reconstruction-level
   `reprojection_noise_px` that measures σ from finite points.
   `analyze --depth-reliability` and `inspect --verbose` report the score and
   `Λ` beside `inverse_depth_z`, so they can be compared on real files before
   anything decides on them.
2. **Reclassification, discovery and the bench.** `classify_rays_at_infinity`
   decides on the score. `CONDITION_NUMBER_PREFILTER`, `DEFAULT_INVERSE_DEPTH_Z_CUTOFF`
   and `DEFAULT_NOISE_FLOOR_PX` stop deciding anything. The bench's `0.8` RMS
   ratio and its override variants go, since the criterion now compares fitted
   candidates. `COINCIDENT_CAMERA_FRACTION` goes, since `Λ ≈ 0` covers that
   case. `classify_points_at_infinity` is no longer relabel-only towards
   infinity: it also promotes a bearing whose score clears the threshold,
   placing it with the point fit, which
   is the point-298 case. The `--find-points-at-infinity` noise-floor component
   becomes a σ override.
3. **Bundle adjustment crossing** (step 1 of the section above).
4. **Inverse-depth free points in bundle adjustment** (step 2 of the section
   above).

## Parameters

| Parameter | Proposed default | Meaning |
|---|---|---|
| `DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD` | `25.0` | Score (and Λ) above which a track is a finite point. See "Open questions". |
| `soft_l1_scale` | `3.0` noise units | Robust loss scale for the point fit; `None` for plain least squares. |
| `max_iterations` | `20` | Gauss-Newton iterations for the point fit. |
| σ | measured | RMS per-axis pixel residual over finite points' observations; per round in bundle adjustment. |

## Testing

- **Nesting.** For any track, `point_cost ≤ bearing_cost` and `Λ ≥ 0`; for
  rays exactly through one direction from every camera, `Λ = 0` and
  `inverse_depth = 0`.
- **Calibration.** Synthetic bearings with Gaussian noise of a known σ: the
  fraction with `Λ > t` matches `½ P(χ²₁ > t)` within sampling error.
- **Recovery.** Synthetic finite points at increasing distance: `Λ` falls as
  distance grows, and the fitted point matches the true point where `Λ` is
  large.
- **The bench's override cases** become ordinary outcomes: an ill-conditioned
  midpoint that reprojects worse than the bearing gets `Λ ≈ 0`, and a
  well-explained depth with a low z gets a large `Λ`.
- **Robustness.** One sighting displaced by 20σ on a true bearing does not lift
  `Λ` over the threshold with the soft-L1 point fit on.
- **The bearing is exact.** The eigenvector solution matches a Gauss-Newton
  bearing fit on the same sine residuals, and `bearing_cost` matches its cost.
- **Score against Λ.** Over synthetic tracks spanning near to infinity, the
  score and Λ agree within a small tolerance wherever either is below
  `4 × threshold`, and `bearing_cost < threshold` never accompanies
  `Λ ≥ threshold`.
- **Batch parity.** The batch form agrees with the single-track form track by
  track, bit for bit.
- **Regression on real data.** The `tk117` table above, from a checked-in copy
  of the Kerry Park ground truth once it lands.

## Open questions

- **Threshold.** `25` puts 298, 294 and 295 finite and keeps 269 (Λ 14) and 270
  (Λ 12) as bearings. `10.8` makes those two finite as well, at 2,800 and 4,800
  units. Both are defensible on `tk117`; a larger capture with a known far
  field (KerryPark360) should decide it.
- **Per-ray noise shape.** A fisheye's pixels per radian differ radially and
  tangentially and change across the field. A scalar `sigma_rad` per ray
  (the geometric mean of the projection Jacobian's two singular values) may be
  enough; a 2×2 per-ray weight is exact.
- **Per-camera σ.** One σ per reconstruction, or one per camera when the
  cameras differ (the two Kerry Park lenses are close; other rigs may not be).
- **Discovery and single-stop tracks.** Whether `find_points_at_infinity`
  should still drop a track whose observing cameras cannot resolve a depth at
  the capture's scale, or add it as a bearing as `Λ` says.
- **Score from pixels or from angles.** The `tk117` score table was measured
  with pixel residuals through the camera model and a Gauss-Newton bearing.
  The primitive uses sine residuals and the eigenvector bearing; the two should
  agree to within the per-ray noise-shape question above, and step 1 confirms
  it before anything decides on the score.
- **Robust decision.** The decision tier is plain least squares. Whether a
  track whose score clears the threshold should also have to clear it under the
  soft-L1 point fit, so one bad sighting cannot make it finite, or whether
  reconstruction-level outlier trimming upstream is enough.
- **Naming.** `bearing_score`, `BearingScore`, `depth_score`,
  `fit_point_and_bearing`, `PointBearingFit`, `depth_likelihood_ratio` and
  `is_finite` are proposals; the settled
  names get a glossary row when this is filed.
