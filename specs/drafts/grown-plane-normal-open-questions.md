# Grown Plane Normal: Open Questions

**Status:** Draft. Questions about the settings and rules of a built step, and
the tests it does not have yet; nothing here changes the step until a question
is settled.

Amends: [core/bench/grown-plane-normal.md](../core/bench/grown-plane-normal.md)
§ "Open questions" and § "Testing". The measurements cited are in
[core/bench/grown-plane-normal-measurements.md](../core/bench/grown-plane-normal-measurements.md).

## Purpose

`grown_plane_normal` estimates a patch's normal by fitting one plane to the
photographs, growing a window from the point outward in each of a few picked
views and combining the views' planes by their information. Most of its numbers
were set on a handful of points or on the same points that score them, and some
of its rules were the first that worked. This draft keeps the questions still
open about them in one place, with the evidence on each, so that a later
measurement can settle them. A question that is settled is deleted here and its
answer folded into the spec.

## Questions

- **The calibration of σ.** With the information read through the Sobel
  gradient, σ is somewhat large in the middle of the errors and still small in
  their tail: 63% of the dino's points and 54% of the points whose stored
  normal was judged right are within σ at the 2.75° target, against 39% for a
  calibrated σ, and 87% and 75% within 2σ, against 86%. Whether τ should be
  raised, read from the data, or σ reported with a floor, and how the pairs'
  reuse of views should enter it.
- **The agreement numbers.** Whether τ = 0.01 is too cautious: it scales every
  σ, so it sets how many estimates agree, when the selection stops and how often
  the report says low confidence. Whether 3σ and the 0.5 floor are the right
  bars; they were set on four points in a Python prototype.
- **The weak-turn rule's thresholds.** 70° and 3 were chosen on the same 62
  points that score them, and every combination near them also beat turning
  every time; whether they hold on points not yet judged, and whether σ (a) is
  the right σ once its calibration is settled.
- **Points where every view is weak.** On dino 1, six views within 5° of the
  reference, each view's own plane is so poorly determined (σ 15° to 171°) that
  their weighted mean is 4.6° to 5.4° off, where fitting the six together
  reached 0.48°: where each view's score is this flat in the tilt, the mean of
  their quadratic models is a poor stand-in for their summed score. One idea,
  not built: once each view's shift is fixed, fit the tilt alone to the sum of
  the views' robust scores. Whether to do that, and how the step would tell such
  a point from one where the mean is right.
- **Narrow-view points.** On a point whose views all lie within about 10° of
  the reference, the plane has low precision (dino 933: 2.62° off, σ 3.5°,
  against the 2×2 Grid Plane Normal's 1.16°). Whether to hand such points to a
  cell-based step.
- **The ladder's settings.** The first rung at 15° held the judged points where
  12° lost one, and the ratios 1.4 and 2.0 moved the dataset medians by 0.15° or
  less; whether a ratio read from the spread of the track's angles, or a ladder
  that ends at a fixed set of angles, does better.
- **The selection's model.** The predicted information is
  `g·(α d dᵀ + β e eᵀ)` with `α` and `β` read from the views matched so far;
  whether a model that reads the view's texture as well as its geometry, or the
  reference's information along each direction, picks better, and what the
  target σ and the budget (`max_selected_views`) should be once τ is settled.
- **The pairs' rules.** Whether the widest pairs, each view in one pair, up to
  three, are the right pairs, or whether a pair should lie on opposite sides of
  the reference or be wider than the widest view's angle from the reference;
  whether halving a pair's information is the right discount for its reuse of
  views, and whether the plane should be the mean with that discount rather than
  with the full information.
- **The pairs' grid.** Whether a pair is compared on the reference's working
  grid, as built, or on a grid of its own, for example one centred on the
  bisector of the two views' rays, where both renders are foreshortened alike.
- **The window schedule.** Whether the fractions `[1/8, 1/5, 1/3, 1/2]` of the
  grid's side are the right schedule.
- **The optimiser.** The prototype's pattern search took larger first steps, up
  to 0.08 tilt units and 1 px, and could leave a basin a local fit stays in.
  Whether the Levenberg–Marquardt fit reaches what the pattern search reached
  on the prototype's points is not measured, and its three settings were chosen
  on synthetic scenes only.
- **Blur matching the other way.** A view finer than the reference is read from
  the pyramid level the bilinear-mip rule picks at the starting plane, once per
  view. Whether one level chosen at the start is enough for a view whose
  footprint changes much as the plane turns, and whether the reference should
  instead be blurred to the view's sampling. In the prototype, blurring the
  reference to the view's sampling did not change the single-view results on
  dino point 810.
- **A view's keypoint in a wide view.** Image 62 of dino point 810, 80° from the
  reference, has a keypoint-implied shift about 6 px in y from where its
  single-view growth settled; whether its keypoint is off in that view is not
  known. The step has no search to cover it, and a fit of the keypoints before
  the step is what would.
- **What the step reports.** σ along the weakest and strongest directions and
  the low-confidence verdict are settled; which of the two σ the verdict reads
  is not. The rest of `GrownPlaneSequence` is what the step knows and records,
  and is open to change once a person reading it through the binding says what
  is missing or unused.

## Tests not written

- **A depth error.** A patch centre off along the reference ray, whose shifts
  the views absorb. A depth error large against the distance to the point
  changes each view's scale as well as its shift (on the tests' scene, half a
  unit at 4 units fitted a plane 7° off), so the scene needs a longer focal
  length than the tests' 160 px to keep the error small while the shift is
  several px.
- **σ against the spread** of the fitted normal under added noise.
- **A pair refused under its floor.**
