# A released spline's weakly supported tail in bundle adjustment (amendment)

**Status:** Draft

Amends [`../core/geometry/bundle-adjustment.md`](../core/geometry/bundle-adjustment.md)
§ "The solve" (the spline release, "Unsupported coefficients are pinned", the
step guard and "Termination") and
[`../core/reconstruction/bundle-adjust.md`](../core/reconstruction/bundle-adjust.md)
§ "Why it is shaped this way" (the distortion release). Both point back here.

Decided: the three changes below, each landable on its own branch in the order
given. Not decided: the penalty weight's scaling under the robust loss, and
whether the reconstruction report carries the held-at-the-guard state per camera
(see "Open questions").

## Purpose

When bundle adjustment releases a camera's radial distortion spline, every
coefficient whose basis span at least one kept observation touches is free to
move, and nothing else constrains it. Observations thin out toward the image
corners, so the outermost coefficient is often held by one or two
observations. Such a coefficient is fitted to those few pixels' noise, and the
solve drives the end of the curve until its slope reaches the model's
monotonicity limit. There a second problem takes over: the step guard rejects
every candidate step whose spline would fold, and since the step is joint, it
rejects the pose and point updates with it. The damping rises until the steps
are so small that the convergence test fires, and the round ends with the poses
and points unsettled.

This draft proposes three changes:

1. **Tail regularization.** A second-difference penalty on the released
   coefficients, the same one the camera refit uses, so a coefficient with
   little support continues its neighbours instead of following a few pixels.
2. **A step guard that does not stall the joint solve.** When only a camera's
   distortion is inadmissible, the same iteration is retried with that
   distortion held for the step, so poses, points and focal still move, and a
   round that ends with the distortion held at the guard is not reported as
   converged.
3. **A regression test** of the whole case.

## Evidence

The case was measured on a reconstruction of the `dino_dog_toy` images taken
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

## What the code does now

In [`bundle_adjust.rs`](../../crates/sfmtool-core/src/geometry/bundle_adjust.rs),
`solve_lm` runs one round's Levenberg–Marquardt solve:

- It accumulates the camera block `h_cc` and gradient `g_c` from the kept
  observations alone (the loop over `blocks` under "Accumulate the
  normal-equation blocks"). No row constrains the spline coefficients except
  the data.
- Inside the damping ladder, after the Schur elimination of the points, it pins
  every lens slot that is not released and every spline slot whose `h_cc`
  diagonal is exactly zero (`_ => h_cc[(slot, slot)] > 0.0`), with an identity
  row and column and a zero gradient entry.
- It builds the candidate lens state camera by camera. A non-positive focal, a
  `k1` that fails `k1_step_admissible` or a spline that fails
  `bspline_step_admissible` sets `admissible = false`, and the whole step is
  rejected with `lambda *= 4.0; continue;` (around lines 2546–2591 when this
  draft was written).
- An accepted step halves `lambda`. Two accepted steps in a row that each
  improve the cost by less than `1e-8` relative set `lambda = INFINITY`
  (around lines 2683–2694), and a ladder of 12 rungs with no accepted step sets
  `improved = false`. Either ends the round with `converged = true`.

The `converged` of the final round is what `FreePointDecision::converged`
reports, and what the "stopped on its iteration budget" clause of the progress
line reads.

Under Marquardt scaling `λ·diag(h_cc)` every slot's step shrinks with `λ`. So
when the spline's part of the step crosses the guard, the rejections raise `λ`
for all slots, the next accepted step is small for the poses and points as
well, and two such steps end the round.

## Step 1: tail regularization

### The penalty

When `opt_bspline` releases camera `j`'s spline, `c = (c₀, …, c_{N−1})`, the
round's cost gains one penalty row per interior coefficient:

```
E_pen,j = w_j · Σ_{i=1}^{N−2} (f_ref,j · (c_{i−1} − 2cᵢ + c_{i+1}))²
w_j     = SMOOTHING · rows_j / N
rows_j  = 2 · (camera j's kept observations in the round)
```

This is the camera refit's penalty
([`../core/camera/refit-camera-intrinsics.md`](../core/camera/refit-camera-intrinsics.md)
§ "The span past θ_fit is regularized" and its Parameters table): the same
second-difference stencil, the same `SMOOTHING = 1e-6`, the same squared weight
per row `SMOOTHING · rows / N`, with the data rows counted the same way, two per
observation. In
[`refit_intrinsics.rs`](../../crates/sfmtool-core/src/camera/refit_intrinsics.rs)
the refit's unknowns are `a = (f, f·c₀, …, f·c_{N−1})`, and `fit_spline` writes
the rows `√w·(a_{i−1} − 2aᵢ + a_{i+1})` for `i` in `1..n − 1`, so its row is
`f·(c_{i−1} − 2cᵢ + c_{i+1})` weighted as above. The adjustment's row is the
same at `f = f_ref`.

**The focal in the row is fixed for the round.** In the refit, `f·c` is one
unknown and the row is linear. In the adjustment, `f` and `c` are separate
parameters, and `f·Δ²c` would couple the penalty into the focal slot. The row
uses `f_ref,j`, camera `j`'s focal at the start of the round, so the penalty is
exactly quadratic in `c` and has no focal column. A focal that moves within the
round changes the penalty's scale only in proportion, and the weight is a tuning
constant anyway.

**The gauge.** The coefficient vector omits the two clamped basis functions the
center-anchored gauge drops (`δ(0) = δ'(0) = 0`); their coefficients are
implicitly zero. The refit writes no row that reaches them, and neither does
the adjustment: the first row is `c₀ − 2c₁ + c₂`. A row through the implicit
zeros would pull `c₀` and `c₁` toward zero near the centre, where the data are
densest and the gauge already fixes the value and slope. At the other end there
is no coefficient past `c_{N−1}`, so the last coefficient is held only by the
last row, `c_{N−3} − 2c_{N−2} + c_{N−1}`. With little data there, `c_{N−1}`
continues `c_{N−3}` and `c_{N−2}` in a straight line, which is the smoothest
continuation the basis offers.

**Rows that touch an unsupported coefficient are left out.** A row enters the
round only when its three coefficients are all supported, that is, touched by a
kept observation, at the round's first linearization. The row set is then fixed
for the round. Two reasons:

- A supported coefficient should not be pulled toward an unsupported one. An
  unsupported coefficient keeps its input value, which may be a wild value from
  an earlier model. A row coupling it to a supported neighbour would bend the
  curve near the last data toward that value.
  `unsupported_bspline_slots_hold_their_input_exactly` in
  [`tests.rs`](../../crates/sfmtool-core/src/geometry/bundle_adjust/tests.rs)
  plants exactly that case: sentinels in the two outermost coefficients, and the
  inner ones must stay near zero.
- The damping ladder compares the cost before and after a step, so the cost must
  be one function for the whole round. Support can change between
  linearizations within a round, because points and poses move. Fixing the row
  set at the start of the round keeps the cost function fixed. A coefficient
  that loses support later in the round is pinned in that linearization, and
  its rows then contribute a constant.

### In the normal equations

Write `P_j = √w_j · f_ref,j · D` for the round's rows, with `D` the
second-difference matrix restricted to the round's row set. The cost the LM
model approximates is `Σ r²` for small residuals (`s²·ρ(r²/s²)` with
`ρ(z) = 2(√(1 + z) − 1)`), so the penalty enters as residual rows `P_j·c` with
Jacobian `P_j`:

- `h_cc[spline block of j] += P_jᵀ·P_j`, added after the data accumulation and
  before the damping ladder;
- `g_c[spline block of j] += P_jᵀ·P_j·c`, at the current coefficients;
- the Marquardt damping `λ·diag(h_cc)` reads `h_cc` with the penalty in it, so a
  coefficient held mostly by the penalty is damped in proportion to that.

The points do not enter the penalty, so the Schur elimination is unchanged.

### Pinning

Pinned slots stay pinned, and the penalty must not move them.

- **The support test reads the data alone.** Today the ladder pins a spline slot
  when `h_cc[(slot, slot)]` is exactly zero. With the penalty added, a slot that
  shares a row with a supported slot has a positive diagonal even when no
  observation touches it, and the test would release it. So `solve_lm` records
  each slot's support from the data accumulation (the diagonal before the
  penalty, or a flag set while scattering the blocks), and the pinning reads
  that record.
- **Pinning removes the penalty's coupling too.** The pinning zeroes the slot's
  row and column of the reduced system and its gradient entry after the Schur
  elimination, which removes every penalty entry that involves it. The slot's
  step is exactly zero, and the existing `dv != 0.0` skip keeps its value bit
  for bit. The supported slots' gradient entries still include the pinned
  slot's value as a constant where a row touches it, which is correct for the
  cost being minimized.

### In the cost the ladder compares

`cost_at` adds `Σ_j E_pen,j`, evaluated at the candidate's coefficients over the
round's row set, to `robust_cost`. The initial `prev_cost` is computed through
the same `cost_at`, so the comparison `new_cost < prev_cost` and the relative
improvement `rel` both include the penalty. The per-observation residuals the
adjustment reports, `residual_norms`, stay data only.

### Sharing with the refit

`SMOOTHING` is a private constant of `refit_intrinsics.rs` today. Step 1 moves
it next to the spline basis in
[`bspline.rs`](../../crates/sfmtool-core/src/camera/distortion/bspline.rs) as a
`pub(crate)` constant, with a `pub(crate)` function for the squared row weight
(`SMOOTHING · rows / n`), and both the refit and the adjustment read them, so
the two agree on what a smooth tail is. The stencil itself is three numbers.
The refit writes it into design rows and the adjustment into the normal
equations, so a shared helper for the rows would not save anything. If the
constant is renamed, the refit spec's Parameters table follows.

### Tests that come with step 1

- The spline block of `g_c` and `h_cc` with the penalty matches a central
  difference of `cost_at` in each released coefficient, on a scene with a
  nonzero planted spline.
- `unsupported_bspline_slots_hold_their_input_exactly` and its pinhole twin
  still pass unchanged: the sentinels come back bit for bit, and the inner
  coefficients stay within the existing tolerance.
- The planted-spline recovery tests (`opt_bspline_recovers_a_planted_spline`,
  `opt_f_and_opt_bspline_recover_together` and their pinhole twins) still pass
  at their current tolerances. If one needs a looser tolerance, the PR says by
  how much, because that is the penalty's bias on a well-supported spline.
- `opt_bspline_holds_at_zero_on_an_equidistant_scene` is unaffected: a zero
  spline has zero penalty and zero penalty gradient.

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
4. Check the rebuilt candidate. The held distortion is the current one, which
   was admissible, so only the focal can fail now; a failure there fails the
   whole step. If any other camera's distortion is now inadmissible, it is
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

Step 1 makes reaching the guard rare; step 2 makes reaching it harmless to the
poses and points.

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

The test is written to fail on the code before steps 1 and 2. Its PR records
that it was run against that code and which assertions failed, and which
assertions each of steps 1 and 2 alone satisfies.

### The real case

The `dino_dog_toy` reconstruction behind the evidence is available locally but
is not checked in, and serves as a manual check of each step: rerun the table in
"Evidence" and expect the focal + distortion release from A at or below A's
median, and B released again to move. Whether to check in a 739 KB minimal
`.sfmr` fixture of it, for an end-to-end test at the Python or reconstruction
level, is a maintainer decision.

## What the standing specs say after this lands

- [`../core/geometry/bundle-adjustment.md`](../core/geometry/bundle-adjustment.md)
  § "The spline release" describes the second-difference penalty, the
  row set and its interaction with the pinning; § "Step guard" and "Step
  guards" (under "Several cameras") describe the held-distortion retry for
  both `k1` and the spline; § "Termination" describes the four endings; its
  testing requirements list the tests of steps 1 to 3.
- [`../core/reconstruction/bundle-adjust.md`](../core/reconstruction/bundle-adjust.md)
  says that a released spline is regularized the way the refit regularizes it,
  and how a round settled at the guard shows in what the adjustment reports.
- [`../core/camera/refit-camera-intrinsics.md`](../core/camera/refit-camera-intrinsics.md)
  says the smoothing constant is shared with the adjustment.

Then this draft is deleted.

## Open questions

- **The weight under the robust loss.** The refit's data rows are plain least
  squares. The adjustment's are soft-L1 with second-order scaling, so an
  observation's curvature is `(1 + z)^(−3/2)` of a plain row's, with
  `z = (r/s)²`, and at the last stage's loss scale of 1 px many inliers have
  `z` near or above one. Should `w_j` count each observation by its robust
  weight, or by `s²`, rather than as two plain rows? Step 1 lands with the
  refit's count and the open question; the test of step 3 and the real case are
  where a different scaling would show.
- **Rows at a pinned coefficient.** Leaving out the rows that touch an
  unsupported coefficient lets the supported tail end freely next to it. The
  alternative, keeping every interior row with pinned values as constants, ties
  the tail to the unsupported values. The draft chooses the first; a case where
  it bends the curve badly would reopen it.
- **Reporting the guard per camera.** The progress line names a camera whose
  distortion ended held. Whether `BundleAdjustment` and the reconstruction
  report also carry a per-camera flag, so the viewer's report can say "the
  spline of camera 0 ended at its monotonicity limit", is open.
