# A released spline's weakly supported tail in bundle adjustment (amendment)

**Status:** Draft

Amends [`../core/geometry/bundle-adjustment.md`](../core/geometry/bundle-adjustment.md)
§ "The solve" (the spline release's step guard and "Termination") and
[`../core/reconstruction/bundle-adjust.md`](../core/reconstruction/bundle-adjust.md)
§ "Why it is shaped this way" (the distortion release). Both point back here.

Decided: the two changes below, each landable on its own branch in the order
given. The first change this draft proposed, the second-difference penalty on a
released spline's coefficients, is built and specified in
[`../core/geometry/bundle-adjustment.md`](../core/geometry/bundle-adjustment.md)
§ "The solve" ("The tail is regularized"). Not decided: the penalty weight's
scaling under the robust loss, and whether the reconstruction report carries
the held-at-the-guard state per camera (see "Open questions").

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
its neighbours instead of following a few pixels. This draft proposes the two
remaining changes, numbered as in the original three-step plan:

2. **A step guard that does not stall the joint solve.** When only a camera's
   distortion is inadmissible, the same iteration is retried with that
   distortion held for the step, so poses, points and focal still move, and a
   round that ends with the distortion held at the guard is not reported as
   converged.
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
the whole-step rejection stalls them as before. That is what step 2 addresses.

## What the code does now

In [`bundle_adjust.rs`](../../crates/sfmtool-core/src/geometry/bundle_adjust.rs),
`solve_lm` runs one round's Levenberg–Marquardt solve:

- It accumulates the camera block `h_cc` and gradient `g_c` from the kept
  observations, then adds each released spline's `SplinePenalty`.
- Inside the damping ladder, after the Schur elimination of the points, it pins
  every lens slot that is not released and every spline slot whose data
  diagonal is exactly zero (`_ => data_diag[slot] > 0.0`), with an identity
  row and column and a zero gradient entry.
- It builds the candidate lens state camera by camera. A non-positive focal, a
  `k1` that fails `k1_step_admissible` or a spline that fails
  `bspline_step_admissible` sets `admissible = false`, and the whole step is
  rejected with `lambda *= 4.0; continue;`.
- An accepted step halves `lambda`. Two accepted steps in a row that each
  improve the cost by less than `1e-8` relative set `lambda = INFINITY`,
  and a ladder of 12 rungs with no accepted step sets
  `improved = false`. Either ends the round with `converged = true`.

The `converged` of the final round is what `FreePointDecision::converged`
reports, and what the "stopped on its iteration budget" clause of the progress
line reads.

Under Marquardt scaling `λ·diag(h_cc)` every slot's step shrinks with `λ`. So
when the spline's part of the step crosses the guard, the rejections raise `λ`
for all slots, the next accepted step is small for the poses and points as
well, and two such steps end the round.

## Step 2: the step guard does not stall the joint solve

### The held-distortion retry

When the candidate step fails only because a camera's distortion is
inadmissible, the same iteration retries the step with that distortion held:

1. Build the candidate lens state camera by camera, as today. Classify each
   camera's failure:
   - a focal that is non-finite or not above `1e-6`, or a non-finite `k1` or
     spline coefficient, fails the whole step as today;
   - a finite `k1` that fails `k1_step_admissible`, or a finite spline that
     fails the monotonicity half of `bspline_step_admissible`, marks camera `j`
     for holding and does not fail the step.
2. If no camera failed the whole step and some cameras are marked, pin the
   marked cameras' distortion slots (the `k1` slot, or every spline slot) in the
   reduced system and its gradient, with the same identity row and column and
   zero gradient entry the support pinning uses, and solve again at the same
   `λ`. The reduced system has to be kept for this: clone `s` and `g_red` before
   the first LU solve, or factor once and pin a copy. The cost is one extra
   `d × d` copy and one extra LU solve, paid only in an iteration whose
   distortion step was inadmissible.
3. Rebuild the candidate from the new solution, with the marked cameras'
   distortion at its current value. Their focals, the other cameras' lenses,
   the poses and the back-substituted points all move. Pinning a slot after
   the Schur elimination is the same as removing that variable from the
   damped system, so the back-substitution `δp = −V⁻¹(g_p + Wᵀδc)` with
   `δc = 0` in the pinned slots is the exact solution with the distortion held.
4. Check the rebuilt candidate. A held spline is the current one, which was
   admissible, and its check does not read the focal. A held `k1` is checked
   again, because `k1_step_admissible` reads the candidate focal, and a smaller
   focal can fold the imaged field at the same `k1`. A failure of a focal or of
   a held `k1` fails the whole step. If any other camera's distortion is now inadmissible, it is
   added to the marked set and step 2 repeats; the set only grows, so this ends
   within `n_cam` repeats.
5. Compare the cost as today. An accepted held step updates the state and halves
   `λ` like any accepted step. A held step that does not lower the cost raises
   `λ` as a cost failure does today, and the next rung tries the full step
   first again.

Each new iteration tries the full step first, so the distortion is released
again as soon as a step that keeps it monotone exists.

The held step is not the constrained optimum along the monotonicity limit. It
moves everything except the distortion, and the next iteration moves the
distortion if it can. That is enough to stop the stall, and it needs no
constrained solver.

The `lambda > 1e12` cap applies to every kind of rejection. Today the
inadmissible path `continue`s without checking it, and only the 12-rung count
ends the ladder.

### How the round ends

The round's ending must not say "converged" while a distortion is held against
its guard and the rest could still move. Each accepted step is either a **full**
step or a **held** step. The round ends in one of four ways:

- **Converged**: two accepted full steps in a row, each improving the cost by
  less than `1e-8` relative. This is today's test, counted over full steps
  only; an accepted held step resets the count.
- **Settled at the guard**: two accepted held steps in a row, each improving the
  cost by less than `1e-8` relative. The poses and points have settled with the
  distortion held, so the round stops, but it reports `converged = false`.
- **Ladder exhausted**: no rung of the ladder found a downhill step. If any rung
  of that ladder made a held retry, the round ends settled at the guard;
  otherwise it ends converged, as today.
- **Iteration budget**: `max_iters` accepted steps, `converged = false`, as
  today.

A round that ends settled at the guard says so on the progress stream, naming
each camera whose distortion was held in its last held step and that camera's
end slope `1 + δ'(d_max)` for a spline. Only the final round's ending reaches
`FreePointDecision::converged`. Its doc comment and the progress line's
"stopped on its iteration budget" clause need a wording that also covers
settling at the guard.

The tail regularization makes reaching the guard rarer; step 2 makes reaching
it harmless to the poses and points.

### The `k1` release takes the same treatment

`k1_step_admissible` guards the `SIMPLE_RADIAL_FISHEYE` release the way
`bspline_step_admissible` guards the spline, through the same `admissible` flag
and the same whole-step rejection. A `k1` pushed toward the fold of the imaged
field stalls the joint solve in the same way, so the held retry covers both: the
marked slot is `k1` on a `k1` camera and the spline block on a spline camera. No
`k1` case has been measured. The focal's guard is not given a retry: a focal at
or below zero is a wild step, not a parameter at a limit, and keeps today's
re-damping.

### Tests that come with step 2

- A constructed case whose full step is inadmissible for the spline and whose
  held step lowers the cost: the iteration accepts the held step, the spline
  comes back bit for bit, and the poses moved.
- Two cameras, one of whose spline crosses the guard: only that camera's spline
  is held, and the other camera's spline moves.
- The same for `k1` on a `SIMPLE_RADIAL_FISHEYE` camera.
- A round that settles at the guard reports `converged = false`.

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
  § "Step guard" and "Step guards" (under "Several cameras") describe the
  held-distortion retry for both `k1` and the spline; § "Termination"
  describes the four endings; its testing requirements list the tests of
  steps 2 and 3.
- [`../core/reconstruction/bundle-adjust.md`](../core/reconstruction/bundle-adjust.md)
  says how a round settled at the guard shows in what the adjustment reports.

Then this draft is deleted.

## Open questions

- **The weight under the robust loss.** The refit's data rows are plain least
  squares. The adjustment's are soft-L1 with second-order scaling, so an
  observation's curvature is `(1 + z)^(−3/2)` of a plain row's, with
  `z = (r/s)²`, and at the last stage's loss scale of 1 px many inliers have
  `z` near or above one. Should the penalty weight `w_j` count each
  observation by its robust weight, or by `s²`, rather than as two plain
  rows? The penalty landed with
  the refit's count. On the real case it keeps the default schedule's release
  off the limit, but not the `[(50, 5)]` or `[(12, 2)]` stage run alone (see
  "After the tail regularization"); whether a heavier weight would, or whether
  step 2 is enough there, is what the test of step 3 and the real case should
  show once step 2 lands.
- **Rows at a pinned coefficient.** Leaving out the rows that touch an
  unsupported coefficient lets the supported tail end freely next to it. The
  alternative, keeping every interior row with pinned values as constants, ties
  the tail to the unsupported values. The kernel does the first; a case where
  it bends the curve badly would reopen it.
- **Reporting the guard per camera.** The progress line names a camera whose
  distortion ended held. Whether `BundleAdjustment` and the reconstruction
  report also carry a per-camera flag, so the viewer's report can say "the
  spline of camera 0 ended at its monotonicity limit", is open.
