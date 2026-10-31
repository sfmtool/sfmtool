# Grown Plane Normal Measurements

This file records the measurements behind
[grown-plane-normal.md](grown-plane-normal.md), which specifies
`grown_plane_normal`: the bench step that estimates which way a point's patch
faces by fitting one plane directly to the photographs, growing a window from
the point outward in each of a few picked views and combining the views' planes.
The measurements bear on the step's design choices and defaults: growing every
view from the shift its keypoint implies, with no search; judging an estimate by
its distance from the combined plane scaled by precision; combining the planes
of a selection of views rather than fitting the views together; the near-to-far
ladder; the weak-turn rule; the optimiser's Sobel gradient, its damping tries
and its relative stop; the early stop, the second growth on the second rung and
the shift-only fit; the information read through the Sobel gradient with the σ
target of 2.75°; reading each view through its own slid patch; and writing no
keypoint.

## How the step is scored

Except where a section says otherwise, the step was run on every finite point of
three ground truths, 1609 points in all: `dino_dog_toy` (the candidate ground
truth `dino_dog_toy_ground_truth_candidate_v006.sfmr`, 961 points),
`kerry_park` (382) and `seoul_bull_sculpture` (266), each started from the mean
viewing direction and fitted once, in an optimised build on a 32-core machine.
The error of a point is the angle between the step's normal and the file's
stored normal. The stored normals are themselves estimates, and some were set to
fit the point's position rather than the surface: most of the dino file's were
set with the bench's Grid Plane Normal at 3×3, and the dino file is a
candidate, not an approved ground truth. "Judged right" counts the points, 62 or 63 depending on the round,
whose normals the maintainer judged by eye on the photographs, scored against
the normal judged right; 49 of them are points on which the judged run's own
grown normal was right, so a form that moves many points loses more of them
than it gains whether or not it is better, and the shares within 5° are read
over every point. "Moved" counts the points more than 1° better or worse than
the form compared against. Where several forms ran at once on the machine, only
the ratios of their times within that run hold.

Each measurement compared the step as it then stood with one change. Later
changes moved the absolute numbers, so a figure in one section is compared only
with the other figures of that section; the decision each section records still
holds in the code.

## Growing from the keypoint-implied shift (2026-10-16 and 2026-10-17)

**Question.** Does starting stage 1 from the shift the view's keypoint implies,
rather than from a search for the first window's shift, keep the first window
from picking a wrong peak?

**On one point.** In a Python prototype, not in the repository, on dino point
810: image 27's centre 7×7 window at the stored normal picked a shift of
(6.5, 3.0) at ZNCC 0.958, with a second peak at (3.0, 6.5) also at 0.958, where
the whole grid's best shift is (−0.5, −0.5); a 13×13 window scores 0.51 at the
wrong shift and 0.91 at the right one. Image 45's centre window picked
(−6.5, −6.5) at 0.967 against the right (1.0, 0.5). The keypoint-implied shifts
were (−0.44, −0.41) and (0.82, 0.65). On dino point 933, seen by three views
within 10° of the reference, the search found no shift reaching 0.7 in any view
and in every view picked a wrong peak scoring about 0.6, at (−5.0, 11.0),
(−10.0, 10.5) and (−7.0, −1.5) px against keypoint-implied shifts of
(1.74, −1.39), (1.77, −0.99) and (0.18, −1.35) px, each within 0.7 px of the
true one; no view joined. Started from the implied shifts, point 810 joined 17
of 17 views on their first try.

**On sixteen points.** The maintainer's sixteen hand-picked points (dino 0, 1,
528, 672, 810 and 933; Seoul bull 120, 50, 236 and 32; Kerry Park 85, 200, 228,
342, 91 and 242), through the binding, the step growing first from the implied
shift and searching after a refusal against the step always searching: the
median error fell from 2.63° to 2.02°, and the step beat the 2×2 Grid Plane
Normal on 13 points where it had on 12. On dino 933 all three views joined where
none had; on dino 528 a fourth view joined and the error fell from 5.36° to
1.49°.

**Decision.** Every growth starts from the keypoint-implied shift, and stage 1
is centred on the anchor, where that shift does not depend on the plane. The
search that followed a refused growth was later removed
(§ [Trusting the keypoints](#trusting-the-keypoints-2026-10-23)).

## The agreement rule (2026-10-16)

**Question.** Does judging a view by its agreement with the shared plane, scaled
by the plane's precision, join the views fixed convergence bars left out without
letting in wrong ones?

**Setup.** The prototype, on four hand-picked points started from the mean
viewing direction: dino 810 (17 views, 7° to 104° from the reference, start
22.8° off), Kerry Park 91 (three fisheye views at 3°, 26° and 104°, start
32.4°), dino 933 (three views at 3°, 8° and 10°, start 8.5°) and Seoul bull 236
(two views at 2° and 4°, start 80.1°).

| Rule | dino 810 | Kerry Park 91 | dino 933 | bull 236 |
|---|---|---|---|---|
| Convergence bars (whole-grid ZNCC ≥ 0.7, window ZNCC never 0.1 under its best) | 0.61°, 12 of 17 joined | 1.39°, 2 of 3 | 8.53°, 0 of 3 | 16.8°, 2 of 2 |
| A fixed loss of at most 0.03 when held to the shared plane, floor 0.5 | 0.47°, 17 of 17 | 1.12°, 3 of 3 | 1.36°, 3 of 3 | 16.8°, 2 of 2 |
| Mahalanobis distance at most 3, τ = 0.01 | 0.54°, 16 of 17, σ 0.38° | 1.12°, σ 1.43° | 1.36°, σ 7.1° | 16.8°, σ 390° |

The convergence bars left out views that ended near the stored normal and, on
dino 933, let no view join. Under the fixed loss, image 27 of dino 810 ended 26°
from the shared plane yet cost exactly 0.030 when held to it, so it passed, and
pulled the shared plane from 0.89° to 2.89° before later views brought it back;
the Mahalanobis rule refused it at 3.2σ.

**A gate on σ.** A 2° gate on σ left dino 933 and bull 236 at their starts, 8.5°
and 80.1° from the stored normal, where the planes it rejected were 1.36° and
16.8°; on bull 236 the first view alone had reached 5.6°. The maintainer chose
to turn to any plane a view joined and report its σ.

**Decision.** The agreement test is the Mahalanobis distance with
`max_agreement_distance` 3, `zncc_tolerance` 0.01 and `min_held_zncc` 0.5, the
values set here, and low precision is reported rather than gated.

## Combining the planes, and the selection (2026-10-17)

**Question.** Over every ground-truth point, how does the step compare with and
without a fit of the shared tilt over every joined view after each join, and
which way of combining the views' planes without it does best for its cost?

**Setup.** Six forms of the step, each at its defaults: the joint fit, and five
ways of combining the views' planes by their information without one, in
sequence or independently, with and without the wide pairs, and with a
selection of views by predicted information (`selected_pairs`). The 2×2 Grid
Plane Normal and `grid_plane_normal` were run from the same fitted track. All
18 runs shared one machine, so only the ratios of the times hold.

| Form | Median | Mean | < 2° | < 5° | < 10° | Beats 2×2 | Confident | Total time |
|---|---|---|---|---|---|---|---|---|
| start | 31.8° | 33.1° | 2% | 5% | 9% | – | – | – |
| 2×2 Grid Plane Normal | 24.7° | 27.0° | 6% | 13% | 22% | – | – | 353 s |
| `grid_plane_normal` | 30.9° | 31.3° | 7% | 12% | 16% | – | – | 93 s |
| joint fit | 2.77° | 11.6° | 41% | 61% | 74% | 79% | 60% | 4470 s |
| the four other combinations | 2.70° to 2.75° | 10.9° to 11.1° | 41% to 43% | 62% to 63% | 74% to 75% | 81% to 82% | 59% to 64% | 1140 s to 1806 s |
| `selected_pairs` | 2.74° | 11.2° | 42% | 62% | 74% | 81% | 62% | 745 s |

Paired against the joint fit, the median of each combination's error less the
joint fit's is −0.01° to −0.03°. On the sixteen hand-picked points,
`selected_pairs` grew 70 of the 114 candidate views and fitted the shift alone
of the other 44.

**Decision.** Without the joint fit the step is no worse, and the five
combinations lie within 0.05° of each other in median and 0.3° in mean.
`selected_pairs` takes about a sixth of the joint fit's time and two thirds of
the fastest other combination's, so the maintainer chose it, and the other
forms were removed.

## The weak-turn rule (2026-10-18)

**Question.** Where the step's normal and the starting or stored normal
disagree, which does a person judging the photographs find right, and does a
rule on the step's own readings pick the right one more often than turning
every time?

**Setup.** The maintainer was shown 70 points across four review rounds, each
with the grown normal beside the start or the stored normal, and decided 62 of
them. The quantities scored are the turn from the start and its ratio to σ (a),
the weakest-direction σ with no pair halved.

| Points | Grown normal right |
|---|---|
| All 62 decided | 49 |
| Turn under 65° | 30 of 32 |
| Turn of 65° or more | 19 of 30 |
| Turn of 65° or more, ratio under 3 | 4 of 13 |
| Turn of 65° or more, ratio of 3 or more | 15 of 17 |

**Rules tried.** A rule on the ratio alone never beat turning every time, at
any bar tried. Keeping the start when the turn is at least a bound and the ratio
under a bar, for bounds of 60°, 65° and 70° and bars from 2 to 3.5, beat turning
every time in every combination; at 70° and 3 it is right on 56 of 62, against
49: it fixes 8 wrong turns and loses 1 good one. The thresholds were chosen on
the points they are scored on, so 56 of 62 is optimistic for points not yet
judged; that every combination near 70° and 3 beats turning every time is the
stronger evidence. Enlarging small patches to 12 px before the step made it
worse (right on 12 of the 27 points it changed, against 19).

**On every ground-truth point.** With the rule at 70° and 3, the step keeps the
start on 8 of 266 Seoul bull points, 12 of 382 Kerry Park points and 1 of 961
dino points, and every other point ends where it ended without the rule.

**Decision.** `weak_turn_deg` 70 and `min_turn_sigma_ratio` 3, on σ (a).

## Fewer renders per fit (2026-10-18)

**Question.** The Levenberg–Marquardt fits make about 94% of the step's renders.
Can they make fewer without changing what the step finds?

**Changes.** Reusing the render that scores a kept step as the residual the next
iteration linearises at (bit for bit the same result on every point); at most
three damped tries an iteration, where it tried eight (more than half of all
fits end on an iteration whose tries all fail, which cost 11.5 renders on
average); and a stop once a kept step gains under 1% of the fit's first kept
step's gain.

| Form | Judged right | Bull median, < 5° | Kerry Park median, < 5° | Dino median, < 5° | Moved better / worse | Time alone |
|---|---|---|---|---|---|---|
| before | 55 of 63 | 19.04°, 16.2% | 6.87°, 40.8% | 1.46°, 85.1% | – | 460 s |
| residual reused, three tries | 55 of 63 | 19.06°, 16.2% | 6.88°, 40.8% | 1.45°, 85.1% | 24 / 18 | 387 s |
| and the 1% stop | 55 of 63 | 18.85°, 15.8% | 6.94°, 41.1% | 1.46°, 85.1% | 56 / 51 | 356 s |

The fits made 69.1 renders each before and 52.7 with the reuse and three tries;
the step's renders fell from 5.11 million to 3.92 million. Two tries gave the
same medians within 0.02°; a `min_improvement` of 10⁻⁵ or a 0.1% stop each lost
a judged point.

**Decision.** The reuse, three tries (`DAMPING_TRIES`) and the 1% stop
(`MIN_RELATIVE_GAIN`), together 23% less time than before.

## The near-to-far ladder (2026-10-18)

**Question.** The selection picked the widest view first and grew every pick
from the starting plane, and on slow points the wide picks failed one after
another (dino 100 refused 17 picks before 6 narrower views passed). Does picking
near to far, growing each later pick from the plane the narrower views reached,
keep the bar with fewer refused picks and less time?

**The angle evidence.** Over 9029 picks of the sweep above, every one grown from
the starting plane, by its angle from the reference:

| Angle from the reference | Passes the floor | Own-plane error, median | σ, median |
|---|---|---|---|
| 0–10° | 97–98% | 9–30° | 13–40° |
| 10–15° | 94% | 5.2° (dino 2.3°) | 6.7° |
| 15–20° | 88% | 3.6° | 3.9° |
| 20–60° | 74–82% | 2–3.6° | 1.6–3.2° |
| 60°+ | 65% | 1.3° | 1.2° |

A wide view fails mainly when the start is far off: with a start within 15° of
the stored normal, views at 20° to 60° pass 92% to 98% of the time, and with a
start 30° or more off, 61% to 76%.

**Forms.**

| Form | Judged right | Bull median, < 5° | Kerry Park median, < 5° | Dino median, < 5° | Moved better / worse | Time alone |
|---|---|---|---|---|---|---|
| widest first, from the start | 55 of 63 | 18.85°, 15.8% | 6.94°, 41.1% | 1.46°, 85.1% | – | 335 s |
| ladder, first rung 12°, ratio 1.6 | 54 of 63 | 18.73°, 16.5% | 6.64°, 42.7% | 1.38°, 85.7% | 98 / 85 | 321 s |
| ladder, first rung 15°, ratio 1.6 | 55 of 63 | 18.73°, 16.2% | 6.60°, 42.9% | 1.37°, 86.5% | 102 / 79 | 324 s |
| ladder, 12°, ratio 1.4 | 54 of 63 | 18.73°, 16.2% | 6.64°, 42.7% | 1.43°, 86.0% | 101 / 91 | – |
| ladder, 12°, ratio 2.0 | 54 of 63 | 18.75°, 16.2% | 6.49°, 44.5% | 1.38°, 86.3% | 98 / 78 | – |

The old selection refused 17.7% of 8473 picks; the ladder at 15° refused 11.1%
of 8249, and capped the worst points' picks at 17 or 18 rather than 29. At 12°
the ladder loses dino 10, whose first pick at 13.7° it grows from the plane of
an 11.8° view; at 15° that view is picked first from the starting plane, as
before.

**Decision.** The ladder at `first_rung_deg` 15 and `rung_ratio` 1.6. The ratio
changes little between 1.4 and 2.0.

## The Sobel gradient for the steps (2026-10-18)

**Question.** The fits' Jacobian was a forward difference, one render per
parameter per iteration. Can it be read from the render the residual is frozen
at, with no render of its own, and keep what the step finds?

**Forms.** The analytic Jacobian with the bilinear read's own derivative as the
image gradient is accurate (each column within 0.1% to 1.7% of a central
difference of renders on synthetic planes) and takes 26% less time, but it moved
127 points better and 138 worse, about twice as many as any of five small
changes of the forward difference (its steps 0.8 and 1.2 times as long, its tilt
step 1.5 times, the damping 10% higher and lower), and it lost two judged
points. The shift columns cause the difference: the bilinear read's slope jumps
at every pixel centre. The forms below read a gradient continuous across pixel
edges, with the analytic tilt columns in every one: a bilinear read of the
level's central-difference, Sobel or Scharr gradient images, a central
difference of the bilinear read over ±h px (the smoothed slope), or the value
read from a blurred level with its exact slope (the blurred render). In every
form the information is still read through the bilinear slope, because a
smoother gradient lowers the information and makes the selection pick more
views at the same σ target;
§ [Sobel information and the σ target](#sobel-information-and-the-σ-target-2026-10-21)
takes that up.

| Form | Judged right | Bull median, < 5° | Kerry Park median, < 5° | Dino median, < 5° | Moved better / worse |
|---|---|---|---|---|---|
| forward difference, before | 55 of 63 | 18.73°, 16.2% | 6.58°, 43.5% | 1.55°, 86.1% | – |
| the five changes of the forward difference | 55 of 63 in each | 18.42° to 18.66°, 16.2% to 16.9% | 6.44° to 6.63°, 42.4% to 44.0% | 1.54° to 1.57°, 85.5% to 86.1% | 42 to 78 / 35 to 81 |
| bilinear slope | 53 of 63 | 18.11°, 15.4% | 6.53°, 41.6% | 1.55°, 84.9% | 127 / 138 |
| **Sobel for steps, bilinear slope for information** | **52 of 63** | **18.77°, 16.5%** | **6.15°, 43.7%** | **1.54°, 86.0%** | **220 / 222** |
| central differences or Scharr for steps | 50, 51 of 63 | 17.40° to 18.71°, 15.8% to 16.2% | 6.33° to 6.39°, 42.1% to 42.7% | 1.55°, 85.7% to 86.4% | 216 to 223 / 188 to 221 |
| smoothed slope over ±0.25 px | 54 of 63 | 18.76°, 16.2% | 6.26°, 43.2% | 1.55°, 85.4% | 168 / 161 |
| blurred render | 44, 47 of 63 | 17.95° to 18.87°, 15.8% to 16.2% | 6.59° to 7.71°, 39.3% to 42.1% | 1.51° to 1.55°, 85.7% to 85.8% | 245 to 312 / 243 to 328 |

Under the Sobel gradient, changes of the optimiser's settings (the damping at
10⁻⁴, 10⁻² and 10% higher, two and five tries, the relative stop at 0.5% and 2%,
`min_improvement` at 10⁻⁵, step caps of 0.05 in the tilt and 1 px in the shift)
move a few to a few tens of points, except the tilt cap of 0.05, and none is
better than the defaults on every measure.

**Judged blind.** The Sobel form and the forward difference end more than 5°
apart on 246 points; 30 of them, 10 from each dataset, not judged before, were
shown to the maintainer blind, each point's two results labelled A and B at
random. The Sobel form was right on 8, the forward difference on 6, both fine on
14, the stored normal on 1, and 1 could not be told.

**Time.** Timed as a matched pair, each run alone in the order before, Sobel,
before, Sobel: 369 s and 360 s before, 226 s and 228 s with the Sobel form, 37%
less.

**Decision.** The fits step with the Sobel gradient. It keeps every share within
5° inside the range of the five changes of the forward difference, lowers Kerry
Park's median from 6.58° to 6.15°, is stable under small changes of the
optimiser's settings, is at least as good on the points judged blind, and takes
37% less time. Its 52 of 63 judged points, three fewer, come from a judged set
that counts against any form that moves many points.

## Cheaper growths (2026-10-20 and 2026-10-21)

**Question.** Three parts of the step cost time without adding to the plane:
growths that end refused (21% to 29% of the step's time), a second growth from
the starting plane after a pick is refused from the combined plane, and the
shift-only fits of the views not picked (27% of the time on the dino). Can each
be made cheaper while the judged points and the shares within 1°, 2° and 5°
stay where they are?

**Stopping a growth early.** The step made 7107 growths, 1562 of them refused
under the floor. Among the 5545 that passed, the lowest 0.5% of scores at the
four fitted windows are 0.22, 0.31, 0.39 and 0.42; the first fitted window is
too often low on a growth that passes. A rule that stops a growth at the first
window stage, from a given stage on, under a fixed score would have cut this
many growths that passed, and stopped this many refused ones:

| From the stage, score under | 0.10 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|
| first fitted window | 15 / 127 | 38 / 374 | 52 / 531 | 72 / 683 |
| second fitted window | 4 / 102 | 14 / 342 | 24 / 496 | 37 / 638 |
| third fitted window | 1 / 82 | 6 / 291 | 9 / 443 | 19 / 572 |

**The second growth.** A later pick refused from the combined plane was grown
again from the starting plane 463 times, and 27% of those growths passed,
about equally on every rung. Dropped on every rung, it saves about 8% of the
time, but on Kerry Park point 308 the first pick joins with a plane 75.7° off
the truth, the next two picks are refused from it, and the patch turns to 83.5°
where it ended at 23.5°; point 52 goes from 5.3° to 70.4°. Kept on the second
rung only, those two points end at 31.3° and 5.3°, for about one percentage
point of the time saved.

**The shift-only fit.** The fit over the whole grid, up to 20 iterations, took
4.2 s on 150 dino points. A fit over one window to the optimiser's usual stop,
then a few iterations over the whole grid, keeps the score: with the second
fitted stage's window and two whole-grid iterations it took 2.5 s, and the
keypoints it would have written lay a median of 0.07 px and a 95th percentile of
0.91 px from those before on the dino. The first fitted stage's window, at 1.9
s, ended 20% to 30% of the views more than 0.01 under the full fit's score.

**Forms.** Against the step before, in one batch:

| Form | Judged right | Bull median, < 5° | Kerry Park median, < 5° | Dino median, < 5° | Moved better / worse | Time |
|---|---|---|---|---|---|---|
| before | 52 of 63 | 18.77°, 16.5% | 6.15°, 43.7% | 1.54°, 86.0% | – | – |
| stop early at 0.25 from the second fitted window | 52 of 63 | 18.77°, 16.5% | 6.15°, 43.7% | 1.54°, 86.0% | 2 / 1 | −5% |
| **the early stop, the second growth on the second rung only, the cheaper shift-only fit** | **52 of 63** | **18.77°, 16.5%** | **6.20°, 43.5%** | **1.55°, 86.0%** | **11 / 11** | **−19%** |
| the same with no second growth | 52 of 63 | 18.77°, 16.5% | 6.26°, 43.5% | 1.54°, 86.0% | 15 / 14 | −20% |

**Decision.** The early stop at 0.25 under `min_held_zncc` from the second
fitted window (`EARLY_REFUSAL_MARGIN`, `EARLY_REFUSAL_FROM_STAGE`), the second
growth on the second rung only (`STARTING_PLANE_RETRY_RUNGS`), and the
shift-only fit over the second fitted window and two whole-grid iterations
(`SHIFT_ONLY_WINDOW`, `SHIFT_ONLY_GRID_ITERATIONS`).

## Sobel information and the σ target (2026-10-21)

**Question.** The fits stepped with the Sobel gradient, but the information that
gives σ, the robust mean's weights and the selection's stop was read through the
bilinear read's own slope, which adds information from detail a pixel or two
across that the fits cannot use. Does that make σ too small, and what does
reading the information through the Sobel gradient change, with the σ target
set again?

**Calibration.** If σ were calibrated, the angle between the step's normal and a
correct normal, over the weakest-direction σ, would have a median of about 1.18,
39% of points within σ and 86% within 2σ (the figures for a two-dimensional
error with σ along both directions; with σ along the weakest direction only they
are a bound). It is read on the dino's 949 points with a finite σ, whose stored
normals are mostly right, and on the 28 points whose stored normal the
maintainer judged right by eye. Each cell is the median ratio, then the shares
within σ and 2σ:

| Information, target σ | Dino | Judged reliable, 28 points |
|---|---|---|
| a calibrated σ | 1.18, 39%, 86% | 1.18, 39%, 86% |
| bilinear slope, 1.5° (before) | 1.42, 34%, 66% | 2.08, 39%, 50% |
| Sobel, 1.5° | 0.96, 53%, 82% | 0.92, 54%, 79% |
| Sobel, 2.25° | 0.85, 59%, 85% | 0.85, 57%, 79% |
| Sobel, 2.75° | 0.78, 63%, 87% | 0.92, 54%, 75% |
| Sobel, 3.0° | 0.74, 65%, 88% | 0.92, 54%, 75% |

**Forms.** Each with `low_confidence_sigma_deg` at 4/3 of the target, against
the step before in the same batch:

| Form | Judged right | Bull median, < 5° | Kerry Park median, < 5° | Dino median, < 5° | Moved better / worse | Time | Dino picks per point |
|---|---|---|---|---|---|---|---|
| bilinear information, 1.5° (before) | 52 of 63 | 18.77°, 16.5% | 6.20°, 43.5% | 1.55°, 86.0% | – | – | 3.53 |
| Sobel, 1.5° | 52 of 63 | 18.50°, 16.5% | 6.42°, 42.4% | 1.44°, 86.4% | 162 / 122 | +18% | 5.06 |
| Sobel, 2.25° | 52 of 63 | 18.50°, 16.5% | 6.39°, 42.4% | 1.52°, 86.3% | 134 / 118 | +4% | 4.01 |
| **Sobel, 2.75°** | **52 of 63** | **18.50°, 16.5%** | **6.38°, 42.9%** | **1.56°, 86.1%** | **124 / 112** | **−2%** | **3.48** |
| Sobel, 3.0° | 52 of 63 | 18.54°, 16.2% | 6.35°, 42.9% | 1.60°, 85.6% | 123 / 122 | −3% | 3.30 |

A turn ratio of 1.75 in place of 3, at 2.25°, changed almost nothing.

**Decision.** The information is read through the Sobel gradient, with
`target_sigma_deg` 2.75° and `low_confidence_sigma_deg` 3.67°. σ is no longer too
small in the middle of the errors, and the selection picks about as many views
as before at about the same time. `min_turn_sigma_ratio` stays 3.

## Trusting the keypoints (2026-10-23)

**Question.** A growth refused at the keypoint-implied shift was followed by an
exhaustive search for the first window's shift, within ±3 px of that shift and
widened to ±12 px about it when that found nothing; a view the combination left
out was grown once more from the search, and a shift-only fit under the floor
searched too. Does the step lose anything by trusting the keypoints and making
no search at all?

| Form | Judged right | Bull median, < 5° | Kerry Park median, < 5° | Dino median, < 5° | Moved better / worse | Step time |
|---|---|---|---|---|---|---|
| with the search | 52 of 63 | 18.50°, 16.5% | 6.38°, 42.9% | 1.56°, 86.1% | – | 146.3 s |
| **no search** | **52 of 63** | **18.07°, 16.5%** | **6.41°, 42.4%** | **1.56°, 85.5%** | **16 / 19** | **132.4 s** |

Of the 23 points that moved by more than 5°, 12 ended closer to the stored
normal and 11 further. On seven points (bull 61, 81, 172 and 182, Kerry Park
57, 360 and 375) a view joined from a searched first window and pulled the
result away from the stored normal; without it they end 10.4° to 62.5° closer.
Seven points whose keypoints were off by more than the growth recovers now join
no view and keep their starting normal (Kerry Park 229, 242 and 246, dino 483,
485, 500 and 575). A review by eye of the read-back localizer's far peaks found
the far match was usually on repeating texture: on 10 of the 11 items decided,
the near match was right.

**Decision.** No search, and no setting to bring it back; fitting a view's
keypoint is a separate step the caller runs before this one where its keypoints
may be off.

## Each view read through its own slid patch (2026-10-24)

**Question.** Every view was read through the one plane through the reference's
slid centre, with the keypoint's gap from the anchor's projection as the view's
starting image shift. The bench's tiles are cut through the patch slid along its
plane onto each keypoint's ray. Does reading each view through its own slid
patch, as its tile is cut, give better normals?

**Tiles first.** Cutting every view's tile from the patch's projection moved by
the keypoint's gap, a 2D shift, was tried against tiles cut through the slid
patch. A blind review of the 20 Kerry Park points whose grown normal moved by
more than 1° preferred the slid-patch tiles on 6 points and the shifted ones on
2, both fine on 9; of the 6 points whose two normals were more than 10° apart,
the slid-patch tiles won 5. Tiles stay cut through the slid patch.

**The step.** Measured on every point of the three ground truths, started from
the mean viewing direction with no fit, against the step reading every view
through the reference's plane: with keypoints left at 25 px median gaps by
rotating every camera about its centre, the grown normal is more than 1° better
on 134, 198 and 122 points and worse on 97, 143 and 112 (Seoul bull, Kerry Park,
dino), the median error falling from 21.4° to 19.0°, 21.3° to 19.0° and 1.99°
to 1.94°; with the same gaps made by sliding every point within its stored plane
it is better on 78, 121 and 35 and worse on 73, 163 and 56. On the refined
ground truths it is better on 22, 43 and 19 points and worse on 13, 41 and 17.

**Decision.** Each view reads the patch slid onto its own keypoint, and its
keypoint-implied shift is zero up to rounding.

## Writing no keypoint (2026-10-27)

**Question.** A turn wrote the keypoint of each view that joined the plane, and
of each shift-only view whose score reached `min_held_zncc`, at the anchor's
projection plus the shift the fit solved for. Are those shifts better keypoints
than the ones the step was given?

**Setup.** The 85-image `dino_dog_toy` ground-truth candidate, 1,095 points with
22,892 observations, in SfM Explorer over MCP: Refine All Normals (the step on
every point), then Refit All Points (the localizer) on its result. Each keypoint
set was then re-triangulated and bundle adjusted with the distortion and each
image's focal length and principal point released.

**What the step wrote.** The normals turned by a median 0.15°, and the step
wrote 22,357 keypoints, moving them a median 0.10 px, 3,704 of them by more than
0.25 px and up to 3.8 px, on points it turned 0.0° as well. With the centres
fixed, the observations whose reprojection error is over 2 px rose from 59 to
101, and the stored ZNCC scores below 0.7 from 104 to 190. The refit moved the
1,240 keypoints the step had moved by more than 0.5 px straight back, a median
direction cosine of −1.00 between the two moves, ending a median 0.07 px from
where they started.

| Keypoints, after a bundle adjustment | Observation error, p99 | Observations over 2 px | Median error where the two sets differ by 1 to 9 px |
|---|---|---|---|
| the refit's | 1.51 px | 91 | 0.59 px |
| the grown plane's | 1.58 px | 107 | 0.82 px |

**Decision.** The shifts are good enough to measure the tilt, and worse
keypoints than the localizer's; written into the track they make the normal step
and the refit undo each other. The step writes no keypoint: it turns the patch
about its centre, keeping every keypoint, and reports the shifts.
