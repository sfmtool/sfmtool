# A released spline's weakly supported tail in bundle adjustment (amendment)

**Status:** Draft

Amends [`../core/geometry/bundle-adjustment.md`](../core/geometry/bundle-adjustment.md)
§ "Testing requirements" and
[`../core/reconstruction/bundle-adjust.md`](../core/reconstruction/bundle-adjust.md)
§ "Why it is shaped this way" (the distortion release). Both point back here.

Decided: the regression test below. The first two changes this draft proposed
are built and specified in
[`../core/geometry/bundle-adjustment.md`](../core/geometry/bundle-adjustment.md)
§ "The solve": the second-difference penalty on a released spline's
coefficients ("The tail is regularized"), and the step guard's retry with the
distortion held together with the round endings that go with it ("Step guards
and the held-distortion retry", "Termination"). Not decided: the penalty
weight's scaling under the robust loss, whether the reconstruction report
carries the held-at-the-guard state per camera, and whether the held retry
should also let the distortion take a damped step (see "Open questions").

## Purpose

Before the tail regularization, when bundle adjustment released a camera's
radial distortion spline, every coefficient whose basis span at least one kept
observation touched was free to move, and nothing but the data constrained it.
Observations thin out toward the image
corners, so the outermost coefficient is often held by one or two
observations. Such a coefficient is fitted to those few pixels' noise, and the
solve drives the end of the curve until its slope reaches the model's
monotonicity limit. There a second problem takes over: the step guard rejects
every candidate step whose spline would fold, and since the step is joint, it
rejects the pose and point updates with it. The damping rises until the steps
are so small that the convergence test fires, and the round ends with the poses
and points unsettled.

The first problem is addressed by the tail regularization the standing spec
now describes: a second-difference penalty on the released coefficients, the
same one the camera refit uses, so a coefficient with little support continues
its neighbours instead of following a few pixels. The second is addressed by
the held-distortion retry the standing spec now describes: a step whose only
fault is a camera's distortion is solved again with that distortion held, so
the poses, the points and the focal still move, and a round that ends with
the distortion held at the guard is not reported as converged. This draft
proposes the one remaining change, numbered as in the original three-step
plan:

3. **A regression test** of the whole case.

## Evidence

The case was first measured, before the tail regularization, on a reconstruction of the `dino_dog_toy` images taken
through one `SFMTOOL_PINHOLE` camera: `f ≈ 1459.3`, 2040 × 1536 pixels, 20
spline coefficients, `bspline_rho_max = 0.8635`. The adjustment ran the default
schedule `[(50, 5), (12, 2), (4, 1)]` (trim px, loss scale) through the Python
binding, `EditedReconstruction(SfmrReconstruction.load(path)).bundle_adjust(...)`,
which reproduces the viewer's Bundle Adjust result exactly.

**The tail has almost no support.** With 20 coefficients there are 19 knot spans
over `[0, ρ_max]`. Counting the kept observations whose span touches each
coefficient: the last, `c19`, is touched by 1 observation (at `ρ = 0.849`, the
outermost one); `c18` by 23; `c17` by 134; `c16` by 412. The outermost twentieth
of the radius holds 1 of the 17,575 observations, so the last span is nearly
empty at any knot count, and the end of the curve has behaved poorly across
several knot counts.

**Releasing the distortion makes the residuals worse.** Call the reconstruction
before the release A and the result of the release B.

| Start | Release | Schedule | Median residual (px), before → after |
|-------|---------|----------|--------------------------------------|
| A | held | default | 0.674 → 0.674 |
| A | focal | default | 0.674 → 0.674 |
| A | focal + distortion | default | 0.674 → 0.7928 |
| A | focal + distortion | `[(50, 5)]` only | 0.674 → 0.798 |
| A | focal + distortion | `[(12, 2)]` only | 0.674 → 0.751 |
| A | focal + distortion | `[(4, 1)]` only | 0.674 → 0.667 |
| B | focal + distortion | default | 0.7928 → 0.7928 |
| B | held | default | 0.7928 → 0.678 |

Released again, B does not move at all. Held, the same solve brings B back to
within a few thousandths of a pixel of A.

**The spline ends at the monotonicity limit.** A's tail was already wild: its
last three coefficients are 0.0163, −0.0075 and 0.0601, and the curve's slope
`1 + δ'` at `d_max` is about 5.4. Every released run, whatever the schedule,
ends with the last three coefficients near 0.0125, 0.0225 and 0.0073 and an end
slope `1 + δ'(d_max)` of about 0.01, where the limit is 0. Read on a grid, the
slope falls from about 1.06 at `ρ = 0.795` to about 0.57 near `ρ = 0.852`, which
is inside the imaged field (the far corner is at `ρ ≈ 0.875`).

**The poses and points were left unsettled.** B's residuals are worse than A's
at every radius, including the centre, where the spline cannot act: the median
in the innermost radial bin rises from 0.52 to 0.59 px. The mean radial bias got
smaller. So the release did not make the lens curve worse at the centre; it
stopped the poses and points from converging.

**Reading.** An unregularized tail supported by very few observations is driven
to the monotonicity limit. At the limit, rejecting the whole joint step
inflates the damping until the tiny-step convergence test fires, and that stalls
the poses and points too. Of the released runs, only the `[(4, 1)]` stage run
alone improves on A.

**After the tail regularization.** The same runs with the penalty in the
kernel (`SMOOTHING = 1e-6`, unchanged from the refit):

| Start | Release | Schedule | Median residual (px), before → after | Last three coefficients | End slope `1 + δ'(d_max)` |
|-------|---------|----------|--------------------------------------|-------------------------|---------------------------|
| A | focal + distortion | default | 0.674 → 0.6775 | 0.0130, 0.0160, 0.0156 | 0.98 |
| A | focal + distortion | `[(50, 5)]` only | 0.674 → 0.7987 | 0.0125, 0.0225, 0.0073 | 0.013 |
| A | focal + distortion | `[(12, 2)]` only | 0.674 → 0.7497 | 0.0123, 0.0225, 0.0073 | 0.013 |
| A | focal + distortion | `[(4, 1)]` only | 0.674 → 0.6664 | 0.0137, 0.0160, 0.0168 | 1.05 |
| B | focal + distortion | default | 0.7928 → 0.6782 | 0.0130, 0.0159, 0.0155 | 0.97 |
| B | held | default | 0.7928 → 0.6776 | unchanged | 0.013 |

Under the default schedule the release now ends with an end slope near 1 and a
median within 0.004 px of A's, and B released again moves back to within
0.005 px of A. The first stage alone, `[(50, 5)]`, and the second alone,
`[(12, 2)]`, still end at the monotonicity limit with the same coefficients as
before: the penalty does not keep those rounds off the limit, and once there
the whole-step rejection stalls them as before.

**The single-stage medians are the schedule's, not the release's.** The same
single-stage runs with nothing released end at nearly the same medians: A
held under `[(50, 5)]` alone goes 0.674 → 0.7991 and under `[(12, 2)]` alone
0.674 → 0.7500, and with the focal alone released 0.7990 and 0.7497. One round
at a wide trim and a large loss scale ends at a state whose median is above
A's whatever is released, so the medians of those rows above do not measure
the stall; their end slopes and their convergence do.

**After step 2.** The same runs with the held-distortion retry in the kernel
as well. "Final round" is how the final round ended (see the standing spec's
"Termination"); the end slope is `1 + δ'(d_max)`:

| Start | Release | Schedule | Median residual (px), before → after | Last three coefficients | End slope | Final round |
|-------|---------|----------|--------------------------------------|-------------------------|-----------|-------------|
| A | held | default | 0.674 → 0.674 | unchanged | 5.42 | converged |
| A | focal + distortion | default | 0.674 → 0.6761 | 0.0130, 0.0164, 0.0159 | 0.97 | converged |
| A | focal + distortion | `[(50, 5)]` only | 0.674 → 0.7990 | unchanged | 5.42 | settled at the guard |
| A | focal + distortion | `[(12, 2)]` only | 0.674 → 0.7496 | 0.0123, 0.0224, 0.0074 | 0.019 | settled at the guard |
| A | focal + distortion | `[(4, 1)]` only | 0.674 → 0.6664 | 0.0137, 0.0160, 0.0168 | 1.05 | converged |
| B | focal + distortion | default | 0.7928 → 0.6760 | 0.0131, 0.0163, 0.0158 | 0.97 | converged |
| B | held | default | 0.7928 → 0.6776 | unchanged | 0.013 | converged |

Under the default schedule the release from A ends 0.002 px above A's median,
against 0.0035 px before step 2, and B released again ends at 0.6760, against
0.6782 before step 2 and below B held. Neither reaches A's own median.

The single-stage runs show what the retry does and what it does not. Under
`[(12, 2)]` alone the round still ends at the monotonicity limit with the same
median, and now says so: it settles at the guard and is not reported as
converged. Under `[(50, 5)]` alone the spline does not move at all. The round
takes 8 held steps and no full step: A's tail starts far from the limit (end
slope 5.4), the full step at each iteration's `λ` overshoots past it, the held
step at that `λ` is accepted and halves `λ`, and the next full step overshoots
further. A damped full step that would move the spline part of the way toward
the limit, which is what step 1 alone took (ending that round at the limit
with a median of 0.7987), is never tried. The default schedule's first round
does the same (8 held steps, no full step); its second round takes 5 full and
13 held steps and settles at the guard, and its third converges on 23 full
steps.

## Step 3: a regression test of the whole case

### The synthetic fixture

A Rust test in
[`geometry/bundle_adjust/tests.rs`](../../crates/sfmtool-core/src/geometry/bundle_adjust/tests.rs)
(or in the reconstruction-level
[`reconstruction/bundle_adjust/tests.rs`](../../crates/sfmtool-core/src/reconstruction/bundle_adjust/tests.rs),
if the medians are easier to read there) builds the case from the evidence:

- An `SFMTOOL_PINHOLE` camera with enough coefficients that the last knot span
  is narrow (20, as in the evidence, or fewer if the test runs faster and still
  reproduces the stall), observed with a true spline that is zero or smooth.
- Points placed so that the observations thin out toward the domain end, with
  the last coefficient touched by one observation, or a few, near `d_max`, and
  pixel noise on every observation.
- A starting spline whose tail is wild in the way A's is: monotone, so the
  start is admissible, with the last three coefficients alternating as A's do
  and an end slope well above 1.

It runs the adjustment twice from the same start, once held and once with
focal and distortion released, under the default schedule, and asserts:

- the median residual after the release is not above the held adjustment's
  (to a small tolerance stated in the test);
- the end slope `1 + δ'(d_max)` of the released spline stays well away from 0
  (a bar such as 0.5, set from what the fixed solve reaches);
- the released solve does not report convergence with the spline at the
  monotonicity limit while the poses could still improve: the final round's
  `converged` is false whenever its ending was settled at the guard, and the
  pose error against the truth after the release is not above the held
  solve's by more than a stated tolerance.

The test is written to fail on the code before the tail regularization and
step 2. Its PR records that it was run against the code before the
regularization and against the code with it but without step 2, and which
assertions failed in each.

### The real case

The `dino_dog_toy` reconstruction behind the evidence is available locally but
is not checked in, and serves as a manual check of each step: rerun the table in
"Evidence" and expect the focal + distortion release from A at or below A's
median, and B released again to move. Whether to check in a 739 KB minimal
`.sfmr` fixture of it, for an end-to-end test at the Python or reconstruction
level, is a maintainer decision.

## What the standing specs say after this lands

- [`../core/geometry/bundle-adjustment.md`](../core/geometry/bundle-adjustment.md)
  § "Testing requirements" lists the regression test of step 3 beside the
  held-distortion retry's tests.
- [`../core/reconstruction/bundle-adjust.md`](../core/reconstruction/bundle-adjust.md)
  keeps its pointer here only while "Reporting the guard per camera" is open.

Then this draft is deleted.

## Open questions

- **The weight under the robust loss.** The refit's data rows are plain least
  squares. The adjustment's are soft-L1 with second-order scaling, so an
  observation's curvature is `(1 + z)^(−3/2)` of a plain row's, with
  `z = (r/s)²`, and at the last stage's loss scale of 1 px many inliers have
  `z` near or above one. Should the penalty weight `w_j` count each
  observation by its robust weight, or by `s²`, rather than as two plain
  rows? The penalty landed with the refit's count. On the real case it keeps
  the default schedule's release off the limit, but not the `[(12, 2)]` stage
  run alone, which with step 2 still ends at the limit, now reported as settled
  at the guard (see "After step 2"). Whether a heavier weight would keep it off
  is what the test of step 3 and the real case should show.
- **A damped distortion step alongside the held one.** The held retry pins the
  distortion outright, so when the full step crosses the guard by overshooting
  rather than because the optimum lies past the limit, the distortion does not
  move for the round (the `[(50, 5)]` run in "After step 2"). An alternative is
  to damp the inadmissible camera's distortion slots more, by their own factor
  on the diagonal, and retry at the same `λ` with that factor rising until the
  step is admissible, pinning being the limit. The poses and points would keep
  the step `λ` allows them while the distortion takes the largest admissible
  damped step. It costs more LU solves per rung and a second damping factor to
  carry between iterations; whether the real case needs it is open, since the
  default schedule's later rounds move the spline anyway.
- **Rows at a pinned coefficient.** Leaving out the rows that touch an
  unsupported coefficient lets the supported tail end freely next to it. The
  alternative, keeping every interior row with pinned values as constants, ties
  the tail to the unsupported values. The kernel does the first; a case where
  it bends the curve badly would reopen it.
- **Reporting the guard per camera.** The progress line names a camera whose
  distortion ended held. Whether `BundleAdjustment` and the reconstruction
  report also carry a per-camera flag, so the viewer's report can say "the
  spline of camera 0 ended at its monotonicity limit", is open.
