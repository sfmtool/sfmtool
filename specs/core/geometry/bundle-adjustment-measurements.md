# Staged Bundle Adjustment Measurements

This file records the measurements behind the free-point choices of
[bundle-adjustment.md](bundle-adjustment.md#free-points-inverse-depth-and-the-storage-decision),
and, in its last three sections, those behind the regularization and the
step-guard retry of a released spline, those behind the per-image release
of each image's focal and principal point, and those behind each image's
radial correction and rolling-shutter rates.
Under `FreePointPolicy::CROSS` the bundle adjustment solves every free point in
inverse depth and decides once, at the end of the solve, whether it is stored as
a position or as a direction, by the point-or-bearing test at the noise level
the final round's residuals measure. The alternative measured here, called *the
crossing between rounds*, carried each free point as a position or a direction
for a whole round and re-decided it on the test between rounds; the kernel no
longer has it. The measurements were taken when the inverse-depth solve
replaced it (#709) and when it became the default (#739). They compare the two,
and the inverse-depth solve with the crossing off, on the seoul bull, seattle,
dino and Kerry Park reconstructions and ground truths and on synthetic scenes,
and they bear on where the decision is read, on the anchor, on what `converged`
reports, and on cost. Starts are named by the perturbation of the poses (for
example "1° and 5%", of the camera extents) and the keypoint noise added.

## The crossing between rounds

The crossing between rounds carried each free point as a position or a
direction for a whole round and re-decided it on the test between rounds. It
was rejected because the geometry of a round was then solved with the point in
the representation it had, and the solve fitted that representation: over ten
cameras and 300 points with 0.76 px of noise, a track
1,500 units out whose rays score 33 at the true poses scores 38 at the end of a
solve that starts it finite and 6 at the end of one that starts it as a
direction, so a borderline track keeps the representation it starts with, and
over six cameras and 40 points one round bends the poses by up to half a degree
to fit a far track as whichever it is given. A marginal track needs several
rounds to settle (on a Kerry Park solve with 1 px of keypoint noise the points
changing at each of the seven re-estimations of an eight-round schedule are
137, 39, 26, 14, 7, 6 and 7). And a round that stops on its iteration budget
far from convergence measures a level that is mostly pose error, makes points
with a depth directions at it, and the next round bends the poses to fit them:
on the Kerry Park solve with five iterations a round, from poses perturbed by
1° and 5% of the camera extents and 1 px of keypoint noise, the first
re-estimation measures 3.1 px and makes 633 of its 886 points directions, and
the solve ends with 658 directions and a median residual of 2.03 px, against
1.05 px with the crossing off.

The same happens on the synthetic scene of the kernel's test
`an_unconverged_round_keeps_points_with_depth`, whose thirty points 250 to 395
units out carry a depth at the converged level but not at three pixels: with
three iterations a round, the crossing between rounds made at least 25 of them
directions at its first re-estimation.

In inverse depth `ρ = 0` is a value the solve can reach, so a point carries no
representation through a round and the poses are not fitted to one chosen
before it; a far
point stays well conditioned because its inverse depth is close to linear in
the observations; and nothing is decided at a round boundary, so a rough round
cannot demote anything. Over 24 far tracks spanning the threshold in each of
four noise draws on the ten-camera scene above (96 tracks, 0.81 px of noise),
the verdict at the result agrees with the verdict at the true poses on 85
tracks solved in inverse depth, on 83 carried as positions and on 74 carried as
directions; the scores read at the result move from those at the true poses by
−7.2 to +13.7 on average per draw in inverse depth and by −11.8 to −13.6 when
carried as directions.

## Translation observability in inverse depth

**Translation observability needs only the pinning directions already had.** For
`ρ > 0` the translation column is the Euclidean point's (`ρ·J` at `p̃ = ρ·p_cam`
is `J` at `p_cam`), and the Schur complement is invariant under an invertible
reparametrisation of a point, so the reduced camera system is the Euclidean one
up to the Marquardt damping. Measured at the first linearization of the first
round, where both parametrisations read the same state, the translation blocks
of the reduced system agree to the printed digits for the images observing the
most far points (on the Kerry Park solve with 1 px of keypoint noise up to 30%
of an image's observations on points more than fifty times farther from their
anchor than the camera is), with smallest-to-largest eigenvalue
ratios of 0.09 to 0.65. Only `ρ = 0` exactly makes the column zero, and that is
the case the existing pin covers; no damping that reads the column's size is
needed.

## Re-reading the anchor every round

**The anchor is re-read at the start of every round.** The anchor only chooses
the parametrisation, but the bound at `ρ = 0` and the singularity at the anchor
are places a point cannot move through, and an anchor left at the input poses
can sit away from cameras a rough start has moved. Because the re-estimation
reads a position between rounds anyway, `ρ` keeping its meaning across rounds
buys nothing. Measured on the default schedule, the two choices store the same
representation for every point, and agree on the median residual to 0.001 px,
on every start up to 1° and 5% on the seoul bull ground truth, the Kerry Park
solve and the seoul bull `sift_files` solve. From 4° and 20% on the
Kerry Park solve, at 200 iterations a round, where both converge, the anchor at
the input poses stops the second round after 17 iterations and the result keeps
3,002 observations under 4 px, with 165 directions and 9 points disagreeing with
the test at the converged level; re-anchored, the second round runs 102
iterations further and the result keeps 3,120 (3,016 with the crossing off),
with 147 directions and none disagreeing.

## Deciding at the end of an unconverged solve

**The decision is read at the end of an unconverged solve too.** A final round
that stops on its iteration budget measures a level that still carries pose
error, which errs toward a direction, and `converged: false` says so. Not
deciding was measured as the alternative, storing each point as the solve left
it. On the Kerry Park solve from 1° and 5% with 1 px of keypoint noise,
counting the points whose stored representation disagrees with the test at the
converged level of 0.77 px:

| Iterations a round | Final level (px) | Decided: dirs, disagree, median (px) | Not decided | Crossing between rounds | Crossing off: disagree, median |
|---|---|---|---|---|---|
| 5 | 1.00 | 225, 74, 1.17 | 8, 155, 0.96 | 658, 434, 2.03 | 116, 1.05 |
| 10 | 0.85 | 185, 34, 1.01 | 8, 151, 0.87 | 537, 303, 1.54 | 115, 1.00 |
| 20 | 0.77 | 155, 1, 0.90 | 1, 155, 0.77 | 222, 17, 0.90 | 136, 0.78 |
| 40 (converges) | 0.77 | 158, 1, 0.90 | 1, 158, 0.77 | 218, 16, 0.90 | 155, 0.77 |
| 60, from 4° and 20% (converges) | 0.77 | 149, 5, 0.89 | 1, 151, 0.77 | 265, 53, 0.96 | 176, 0.84 |

"Not decided" is the same solve with the decision skipped, and the crossing
between rounds is the alternative of
[The crossing between rounds](#the-crossing-between-rounds). Deciding is
closer to the converged answer at every budget. The solve's own `ρ` makes few
points directions, because noise puts the best `ρ` of a far point above zero
about half the time, so leaving it undecided keeps nearly every far point
finite. The 20-iteration run does not meet the convergence test, though its
level has already settled at the converged one, so a rule that decided only
after convergence would leave it undecided. No budget brings back the mass
demotion of the crossing between rounds: at five iterations the decision makes
225 of the 886 points directions, against 658, and the median residual is
1.17 px, against 2.03.

## Unconverged ends at the default budget

**At the default budget an unconverged end is rare, and its level is the
converged one.** Every caller runs 60 iterations a round. Measured through the
reconstruction-level adjustment, which is what `sfm xform --bundle-adjust` and
the viewer's Bundle Adjust run, with each camera's focal and distortion released
where its model admits them, over thirteen inputs -- the incremental and global
solves of `seoul_bull_sculpture`, `seattle_backyard`, `kerry_park` and
`dino_dog_toy` converted to embedded patches and a spline camera
(`SFMTOOL_FISHEYE` for Kerry Park, `SFMTOOL_PINHOLE` for the others), the Kerry
Park global solve after `--find-points-at-infinity`, the Kerry Park ground
truth, the seoul bull
ground truth, the Kerry Park solve and the seoul bull `sift_files` solve -- each
clean and from poses perturbed by 0.2° and 1%, 1° and 5%, and 4° and 20% of the
camera extents with 1 px of keypoint noise: 48 of the 52 solves run (four from
4° and 20% exit degenerate), 6 of those end with an unconverged final round,
none of them from a clean start, and in 5 more an earlier round stops on its
budget and the final round converges. Each unconverged end is compared with the
same input solved at 3,000 iterations a round, where every round converges:

| Input | Start | Level, decided / converged (px) | Directions, decided / converged | Disagree | Disagree at the converged level |
|---|---|---|---|---|---|
| seoul bull global solve | 0.2° and 1% | 0.823 / 0.825 | 0 / 3 | 3 | 3 |
| dino incremental solve | 0.2° and 1% | 1.082 / 1.087 | 1 / 5 | 6 | 6 |
| Kerry Park global solve, points at infinity found | 1° and 5% | 0.733 / 0.733 | 735 / 737 | 2 | 2 |
| Kerry Park global solve, points at infinity found | 4° and 20% | 0.311 / 0.311 | 532 / 532 | 0 | 0 |
| seoul bull `sift_files` solve | 4° and 20% | 0.693 / 0.705 | 34 / 34 | 6 | 6 |
| seoul bull ground truth | 4° and 20% | 0.672 / 0.670 | 5 / 15 | 12 | 12 |

"Disagree" counts the free points whose stored representation differs from the
converged solve's, and the last column the same count when the unconverged state
is decided at the converged solve's level. The levels agree to within 2%, and
deciding at the converged level changes no point, so where the two answers
differ it is because the poses and points differ, not the level. The three from
4° and 20% are solves that failed: their final rounds keep 476 of 4,172, 732 of
3,007 and 552 of 1,277 observations, and their 3,000-iteration references
failed too (median residuals of 48.2 px on the seoul bull ground truth,
165 px on the seoul bull `sift_files` solve and 12.5 px on the Kerry Park
global solve with points at infinity), so those three rows compare two failed
states.

## Collapsed solves that report convergence

**`converged` reports the iteration budget, not success.** It says whether the
final round met its convergence test, and a solve that has collapsed meets it as
readily as one that has recovered. Of the 48 solves of
[Unconverged ends at the default budget](#unconverged-ends-at-the-default-budget), 8 more report
`converged: true` from a collapsed state, and their points are decided all the
same:

| Input | Start | Final round keeps | Median residual, crossing on (px) | Directions | Crossing off (px) |
|---|---|---|---|---|---|
| seoul bull incremental solve | 4° and 20% | 20 of 3,258 | 106.7 | 2 | 89.8, 227 points deleted |
| seoul bull global solve | 1° and 5% | 63 of 2,921 | 92.1 | 13 | 50.6, 243 points deleted |
| dino incremental solve | 1° and 5% | 587 of 85,805 | 322.7 | 0 | 249.9, 1,139 points deleted |
| dino global solve | 1° and 5% | 100 of 81,324 | 311.1 | 0 | degenerate exit |
| seattle incremental solve | 4° and 20% | 2,257 of 14,448 | 18.1 | 77 | 0.95, 205 points deleted |
| seattle global solve | 4° and 20% | 909 of 14,386 | 27.8 | 48 | 2.31, 233 points deleted |
| Kerry Park global solve | 4° and 20% | 724 of 2,797 | 9.2 | 163 | 51.3 |
| Kerry Park ground truth | 4° and 20% | 640 of 3,767 | 14.8 | 22 | 19.3 |

The Kerry Park incremental solve from 1° and 5% is borderline (30 of 70 kept,
3.69 px; 5.39 px off). A collapse shows in the final round's kept count and the
median residual, not in `converged`. It is not specific to the crossing: with
the crossing off the same starts collapse in six of the eight, and reach 0.95
and 2.31 px only on seattle, in a draw the five-draw table under
[Rough starts with the lenses released](#rough-starts-with-the-lenses-released)
shows to be the one of five where that happens.

Growth's adjustments end converged in every case measured, all 49 of them, on
the tracks of the seoul bull, seattle and dino incremental solves, each camera
of the Kerry Park global solve, the Kerry Park solve and the first camera of the
Kerry Park ground truth. `rotation_init` validates no rotation
edge on any of these inputs, which have no far field; on the binding tests'
synthetic far-field scene its finishing adjustment converges for each of four
seeds.

## Remedies at short budgets

**Remedies, measured where budgets are short.** An unconverged end changes the
answer only at budgets well under the default. From 1° and 5% with 1 px of
keypoint noise, at 5, 10 and 20 iterations a round, each candidate rule is
compared with the converged solve of the same input, counting disagreements as
above:

| Input | Iterations | Level / converged level | As decided | At the converged level | At the previous round's level | At a robust level | No demotion | One more final round | Not decided |
|---|---|---|---|---|---|---|---|---|---|
| Kerry Park solve | 5 | 1.30 | 106 | 91 | 236 | 104 | 161 | 87 | 165 |
| Kerry Park solve | 10 | 1.10 | 78 | 81 | 159 | 94 | 161 | 70 | 163 |
| Kerry Park solve | 20 | 1.00 | 14 | 14 | 11 | 79 | 161 | 13 | 160 |
| Kerry Park global solve | 5 | 1.51 | 192 | 160 | 356 | 160 | 137 | 181 | 150 |
| Kerry Park global solve | 10 | 1.18 | 121 | 114 | 168 | 115 | 137 | 115 | 145 |
| Kerry Park global solve | 20 | 1.01 | 44 | 42 | 52 | 65 | 137 | 27 | 135 |
| Kerry Park ground truth | 5 | 1.56 | 10 | 4 | 20 | 5 | 15 | 19 | 26 |
| Kerry Park ground truth | 10 | 1.81 | 12 | 2 | 18 | 5 | 15 | 14 | 25 |
| Kerry Park ground truth | 20 | 1.18 | 2 | 0 | 15 | 0 | 15 | 13 | 25 |
| seoul bull `sift_files` solve | 5 | 1.46 | 9 | 2 | 25 | 2 | 0 | 3 | 0 |
| seoul bull `sift_files` solve | 10 | 1.25 | 2 | 1 | 16 | 1 | 0 | 2 | 0 |
| seoul bull `sift_files` solve | 20 | 1.00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Total | | | 590 | 511 | 1,076 | 630 | 939 | 544 | 994 |

"As decided" is the kernel's rule, the final round's measured level. "At the converged level" decides the same state
at a level no solve can know before it converges, and shows how much of the
error the level accounts for; it is not a bound, and a rule that reads another
level does better than it in a few rows. "At the previous round's level" decides
at the level measured at the end of the second round, "at a robust level" at
1.4826 times the median absolute per-axis residual over the observations the
final round kept, "no demotion" keeps a position every free point handed in as a
position, "one more final round" runs the last schedule round again from where
the solve stopped, at the same budget, and decides there, and "not decided"
stores the solve's own `ρ`. What the table shows:

- **The level is the smaller part of the error.** Deciding at the converged
  level takes the total from 590 disagreements to 511. Those 511 come from
  deciding at poses and points that are not yet the converged ones, which no
  level corrects.
- **The previous round's level is higher**, its trim being wider and its poses
  rougher, and makes more directions: 1,076.
- **A robust level is lower than the converged one even where the level has
  settled** (0.59 to 0.61 px on the two Kerry Park inputs at 20 iterations,
  whose levels are within 1% of the converged 0.77, though the solves have not
  converged): the residuals have heavier tails than a Gaussian, whose spread
  the median-based estimate assumes. It makes far points finite that the
  converged solve stores as directions, 79 against 14 on the Kerry Park solve at
  20 iterations.
- **Not demoting is right only where the converged solve makes no direction.**
  It agrees on the seoul bull `sift_files` solve. On the two Kerry Park solves
  it leaves every one of the 161 and 137 converged directions finite at every
  budget, and on the Kerry Park ground truth, whose 9 bearings are handed in as
  directions, it leaves finite the 15 points the converged solve demotes. It
  is the asymmetric guard the point-or-bearing work measured and rejected, and
  it fails here for the same reason.
- **One more final round** removes 46 disagreements in all, at 0.01 to 0.30 s,
  converges in 1 of the 12 runs, and is worse than deciding on the Kerry Park
  ground truth. On the
  seoul bull global solve from 1° and 5%, which collapses to between 63 and 92
  kept observations at these budgets, it raises the disagreements from 0 to 3 to
  between 86 and 117, and at the default budget, on the Kerry Park global solve
  with points at infinity from 4° and 20%, from 0 to 441. A run to convergence
  is what it approximates, and that is a larger `max_iters`, which is the
  caller's to choose.

So the decision is read at the final round's measured level whether or not that
round converged, `converged: false` and the progress line saying when it did
not. At the default budget the level of an unconverged end is the converged
level; at budgets short enough that it is not, what remains is the unconverged
state, which only more iterations change.

## Rough starts with the lenses released

**Rough starts with the lenses released.** The crossing changes which rough
starts the reconstruction-level adjustment recovers from, in both directions,
and with every lens released neither setting recovers reliably from 4° and 20%.
Over five perturbation draws each, the median residual after the solve with the
crossing on and off:

| Input | Start | Crossing on (px) | Crossing off (px) |
|---|---|---|---|
| dino global solve | 0.2° and 1% | 1.62 to 2.71 | 5.20 to 32.98 |
| Kerry Park global solve | 1° and 5% | 0.89, 2.40, 2.13, 1.41, 2.13 | 3.80, 0.94, 1.07, 0.78, 3.39 |
| Kerry Park global solve | 4° and 20% | 3.18 to 16.03 | 28.20 to 151.61 |
| Kerry Park ground truth | 4° and 20% | 14.83 to 58.12, lower in four draws | 19.34 to 92.67 |
| seattle incremental and global solves | 4° and 20% | 18.06 and 27.81 in the first draw; 19.16 to 120.48 in the others | 0.95 and 2.31 in the first draw; 26.19 to 70.12 and one degenerate exit in the others |
| seoul bull incremental solve | 1° and 5% | 0.81 to 1.10 in three draws; 22.3 and 22.5 in two | 0.82 in three draws; 20.3 and 32.5 in two |

With the lenses held, the first seattle draw recovers with the crossing on as
well (1.11 and 0.99 px, against 0.93 and 0.90 off).

## Disagreements left at a converged end

**What the crossing left at a converged end, inverse depth does not.** From 4°
and 20% the Kerry Park result disagrees with the test at the converged level on
5 points at 60 iterations and on 3 at 200 (both converge, at the level of
0.77 px), where the crossing between rounds left 53 and 35. Read at the
result's own measured level, which is what `sfm analyze --depth-reliability`
lists, the counts are 35 and 50: that level is the stored measure over every
observation, and the 13 observations over 4 px, which the adjustment's last
trim left out and the stored measure's gate passes, raise it to 0.86 and
0.89 px, where the decision read the 0.77 px of the observations it solved on.

## Cost

**Cost is the solve's own.** The crossing changes how many iterations the
rounds take, not what an iteration costs. Seconds, best of five runs alternating
crossing off and on (three for the 1° and 5% starts and for dino growth, seven
for `rotation_init`), on the inputs above:

| Caller | Input | Off (s) | On (s) | On / off |
|---|---|---|---|---|
| reconstruction-level adjustment | seoul bull incremental solve | 0.169 | 0.151 | 0.89 |
| reconstruction-level adjustment | seoul bull global solve | 0.292 | 0.471 | 1.62 |
| reconstruction-level adjustment | seattle incremental solve | 0.904 | 0.746 | 0.83 |
| reconstruction-level adjustment | seattle global solve | 0.668 | 0.456 | 0.68 |
| reconstruction-level adjustment | Kerry Park global solve | 0.225 | 0.120 | 0.53 |
| reconstruction-level adjustment | Kerry Park global solve, points at infinity found | 0.180 | 0.255 | 1.42 |
| reconstruction-level adjustment | dino incremental solve | 11.903 | 8.852 | 0.74 |
| reconstruction-level adjustment | dino global solve | 9.965 | 5.480 | 0.55 |
| reconstruction-level adjustment | Kerry Park ground truth | 0.400 | 0.226 | 0.57 |
| reconstruction-level adjustment | Kerry Park solve | 0.110 | 0.053 | 0.48 |
| reconstruction-level adjustment | seoul bull `sift_files` solve | 0.042 | 0.030 | 0.72 |
| reconstruction-level adjustment | seoul bull ground truth | 0.033 | 0.021 | 0.64 |
| reconstruction-level adjustment, from 1° and 5% | Kerry Park solve | 0.516 | 0.394 | 0.77 |
| reconstruction-level adjustment, from 1° and 5% | Kerry Park ground truth | 1.216 | 1.397 | 1.15 |
| reconstruction-level adjustment, from 1° and 5% | Kerry Park global solve | 0.156 | 0.418 | 2.68 |
| reconstruction-level adjustment, from 1° and 5% | seattle global solve | 1.516 | 1.833 | 1.21 |
| `grow_reconstruction` | seoul bull incremental solve | 0.339 | 0.295 | 0.87 |
| `grow_reconstruction` | seattle incremental solve | 2.445 | 2.103 | 0.86 |
| `grow_reconstruction` | Kerry Park solve | 0.745 | 0.515 | 0.69 |
| `grow_reconstruction` | Kerry Park ground truth, first camera | 0.418 | 0.180 | 0.43 |
| `grow_reconstruction` | dino incremental solve | 46.658 | 27.771 | 0.60 |
| `rotation_init` | synthetic far-field scene, four seeds | 0.059 to 0.432 | 0.056 to 0.430 | 0.92 to 1.03 |

`rotation_init` has no opt-out, so its "off" column was timed through a switch
built for the measurement and not kept.

Timed round by round, an iteration costs the same in either parametrisation:
3.4 ms off and 3.6 ms on for the seoul bull global solve, and 1.198 s and
1.209 s for the 11 iterations of the first round of the dino global solve. The
re-estimation under the crossing costs 0.08 to 0.09 s a round on the dino global
solve, against 0.03 off, and the decision 0.015 s. The slower runs take more
iterations: the seoul bull global solve takes 59, 16 and 17 iterations a round
off and 60, 60 and 17 on; the Kerry Park global solve with points at infinity,
13, 11 and 11 off and 28, 6 and 9 on; the seattle global solve from 1° and 5%,
39, 11 and 36 off and 53, 22 and 24 on; the Kerry Park ground truth from 1°
and 5%, 29, 25 and 37 off and 25, 33 and 55 on. The dino global solve saves the same
way, with 11, 32 and 34 off and 11, 6 and 26 on. The Kerry Park global solve
from 1° and 5% takes longer because its second round runs the whole budget and
the solve recovers, to 0.886 px, where with the crossing off that round stops
after 15 iterations and the solve ends at 3.797 px. The time is spent on the
path the solve takes rather than on any step the crossing adds, so the
crossing's own work holds nothing to remove.

## The default schedule against the crossing between rounds

The default schedule over the four inputs of the point-or-bearing work, clean
and degraded (poses perturbed by 0.2° and 1% of the camera extents, 1 px of
keypoint noise, both, and 1° and 5% with 1 px), and the same 1° and 5% start
with five iterations a round, every free point crossing and the focal held.
"Disagree" counts the points whose stored representation the test at the
result's own measured noise disagrees with; "dirs" are the directions in and
out. Every default-iteration run converges.

| Input | Start | Between rounds: dirs, disagree, median (px) | Inverse depth: dirs, disagree, median (px) |
|---|---|---|---|
| Kerry Park ground truth | clean | 9 → 9, 0, 0.167 | 9 → 3, 0, 0.165 |
| Kerry Park ground truth | 1 px keypoints | 9 → 12, 0, 1.016 | 9 → 12, 0, 1.022 |
| Kerry Park ground truth | 0.2° and 1%, 1 px | 9 → 12, 1, 1.018 | 9 → 12, 1, 1.021 |
| Kerry Park ground truth | 1° and 5%, 1 px | 9 → 12, 1, 1.018 | 9 → 12, 1, 1.019 |
| seoul bull ground truth | clean | 14 → 14, 0, 0.256 | 14 → 13, 0, 0.254 |
| seoul bull ground truth | 1 px keypoints | 14 → 14, 0, 0.913 | 14 → 14, 0, 0.913 |
| seoul bull ground truth | 0.2° and 1%, 1 px | 14 → 14, 0, 0.911 | 14 → 14, 0, 0.912 |
| seoul bull ground truth | 1° and 5%, 1 px | 14 → 14, 1, 0.921 | 14 → 14, 0, 0.925 |
| Kerry Park solve | clean | 0 → 0, 0, 0.173 | 0 → 0, 0, 0.173 |
| Kerry Park solve | 1 px keypoints | 0 → 174, 27, 0.863 | 0 → 134, 7, 0.869 |
| Kerry Park solve | 0.2° and 1%, 1 px | 0 → 213, 34, 0.892 | 0 → 161, 2, 0.899 |
| Kerry Park solve | 1° and 5%, 1 px | 0 → 218, 33, 0.897 | 0 → 161, 10, 0.901 |
| seoul bull `sift_files` solve | every start | 0 → 0, 0 | 0 → 0, 0 |
| Kerry Park ground truth | 1° and 5%, 5 iterations | 9 → 43, 78, 3.75 | 9 → 27, 83, 1.39 |
| seoul bull ground truth | 1° and 5%, 5 iterations | 14 → 24, 114, 4.29 | 14 → 13, 87, 1.51 |
| Kerry Park solve | 1° and 5%, 5 iterations | 0 → 658, 58, 2.03 | 0 → 225, 272, 1.17 |
| seoul bull `sift_files` solve | 1° and 5%, 5 iterations | 0 → 75, 228, 6.29 | 0 → 9, 276, 2.81 |

On the converged Kerry Park runs the disagreements fall from 27 to 34 to 2 to
10, and the directions from 174 to 218 to 134 to 161: the crossing between
rounds left marginal far points still settling as directions when the three
rounds ended, and inverse depth settles them within the rounds. The
disagreements on the converged degraded runs are read at the result's own
level, a little above the level the decision read; at the decision's level
every converged run in the table agrees with the test on every point. On the
clean Kerry Park ground truth the solve makes 6 of its 9 bearings finite
(156, 157, 176, 268, 269 and 270), where the crossing between rounds makes
none. The six score 9 to 16 at the input, 9 to 16 after the solve that carries
them as directions, and 28 to 34 after the one that carries them in inverse
depth. They are depth
the capture supports. Rebuilt synthetically on the Kerry Park ground truth's
own geometry -- its poses, lenses and finite points, with the observation
patterns of its 9 bearings repeated five times (45 far tracks) and 0.21 px of
noise, over four draws (ten for the tracks truly at infinity) -- tracks that
are truly at infinity are made finite by the inverse-depth solve in 0 of 450
cases (mean score 0.5 to 1.8, against 0.2 to 1.1 at the true poses). Tracks
truly 2,500 m out score about 37.9 to 42.8 at the true poses and 41.3 to 46.1
after the inverse-depth solve, which makes 166 of 180 finite, but about 3.8 to
4.6 when carried as directions: the crossing between rounds makes 2 of 180
finite and the crossing off 0. At 5,000 m 9 of 180 are finite at the true
poses and 26 of 180 after the inverse-depth solve. A lens error moves the
scores of true bearings further in inverse depth than when they are carried as
directions: with the focal 1% off, the mean score of the truly infinite tracks
is about 58 to 66 against about 35 under the crossing between rounds, and 1 of
180 is made finite; with it 0.3% off, the mean score is about 7 to 9 against
about 4, and none is made finite. The five-iteration
disagreements are read at the result's own level, which unconverged poses put
at 3.1 to 12 px on the three inputs other than the Kerry Park solve, a level at
which the test calls most points bearings; the decision's level, read over the
observations the final round kept, is 1.0 to 1.3 px there. Those runs leave 28
to 315 points unscored, each in the representation it was handed in with. No input produced a
`NaN` point.

## A weakly supported spline tail

These measurements are behind the tail regularization of a released spline,
the rule that the penalty places a coefficient no observation supports, and
the held-distortion retry at the step guard
([bundle-adjustment.md](bundle-adjustment.md) § "The solve"). They were taken
on a reconstruction of the `dino_dog_toy` images through one `SFMTOOL_PINHOLE`
camera: `f ≈ 1459.3`, 2040 × 1536 pixels, 20 spline coefficients,
`bspline_rho_max = 0.8635`. The adjustment ran the default schedule
`[(50, 5), (12, 2), (4, 1)]` (trim px, loss scale) through the Python binding,
`EditedReconstruction(SfmrReconstruction.load(path)).bundle_adjust(...)`, which
gives the viewer's Bundle Adjust result exactly. The reconstruction is kept
locally and not checked in; rerunning these tables on it is the manual check
of a change to the spline release. A is the reconstruction before the
release, and B is what
releasing A's focal and distortion under the default schedule gave before
either change.

**The tail has almost no support.** With 20 coefficients there are 19 knot
spans over `[0, ρ_max]`. Of the 17,575 kept observations, the last coefficient,
`c19`, is touched by 1 (at `ρ = 0.849`, the outermost); `c18` by 23; `c17` by
134; `c16` by 412. The outermost twentieth of the radius holds 1 of the 17,575
observations, so the last span is nearly empty at any knot count, and the end
of the curve has behaved poorly across several knot counts. A's last three
coefficients are 0.0163, −0.0075 and 0.0601,
and its end slope `1 + δ'(d_max)` is 5.47.

**Without either change.**

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

Every released run, whatever the schedule, ends with the last three
coefficients near 0.0125, 0.0225 and 0.0073 and an end slope of about 0.01,
where the monotonicity limit is 0. Read on a grid, the slope falls from about
1.06 at `ρ = 0.795` to about 0.57 near `ρ = 0.852`, inside the imaged field
(the far corner is at `ρ ≈ 0.875`). B's residuals are worse than A's at every
radius, including the centre, where the spline cannot act: the median in the
innermost radial bin rises from 0.52 to 0.59 px, while the mean radial bias
got smaller. So the release did not make the lens curve worse at the centre;
it stopped the poses and points from converging. The coefficient fitted to
one observation drives the end of the curve to the limit, the guard then
rejects every joint step, the damping rises until the tiny-step test fires,
and the round reports convergence with the poses and points unsettled.

**With the tail regularization alone** (`SMOOTHING = 1e-6`, the refit's
weight):

| Start | Release | Schedule | Median residual (px), before → after | Last three coefficients | End slope |
|-------|---------|----------|--------------------------------------|-------------------------|-----------|
| A | focal + distortion | default | 0.674 → 0.6775 | 0.0130, 0.0160, 0.0156 | 0.98 |
| A | focal + distortion | `[(50, 5)]` only | 0.674 → 0.7987 | 0.0125, 0.0225, 0.0073 | 0.013 |
| A | focal + distortion | `[(12, 2)]` only | 0.674 → 0.7497 | 0.0123, 0.0225, 0.0073 | 0.013 |
| A | focal + distortion | `[(4, 1)]` only | 0.674 → 0.6664 | 0.0137, 0.0160, 0.0168 | 1.05 |
| B | focal + distortion | default | 0.7928 → 0.6782 | 0.0130, 0.0159, 0.0155 | 0.97 |
| B | held | default | 0.7928 → 0.6776 | unchanged | 0.000 |

The single-stage medians are the schedule's, not the release's: with nothing
released, A goes to 0.7991 under `[(50, 5)]` alone and to 0.7500 under
`[(12, 2)]` alone, and with the focal alone released to 0.7990 and 0.7497. So
those rows measure the stall by their end slopes, not their medians.

**With both changes.** "Final round" is how the final round ended (see
[bundle-adjustment.md](bundle-adjustment.md) § "The solve", "Termination"):

| Start | Release | Schedule | Median residual (px), before → after | Last three coefficients | End slope | Final round |
|-------|---------|----------|--------------------------------------|-------------------------|-----------|-------------|
| A | held | default | 0.674 → 0.674 | unchanged | 5.47 | converged |
| A | focal + distortion | default | 0.674 → 0.6761 | 0.0130, 0.0164, 0.0159 | 0.97 | converged |
| A | focal + distortion | `[(50, 5)]` only | 0.674 → 0.7990 | unchanged | 5.47 | settled at the guard |
| A | focal + distortion | `[(12, 2)]` only | 0.674 → 0.7496 | 0.0123, 0.0224, 0.0074 | 0.019 | settled at the guard |
| A | focal + distortion | `[(4, 1)]` only | 0.674 → 0.6664 | 0.0137, 0.0160, 0.0168 | 1.05 | converged |
| B | focal + distortion | default | 0.7928 → 0.6760 | 0.0131, 0.0163, 0.0158 | 0.97 | converged |
| B | held | default | 0.7928 → 0.6776 | unchanged | 0.000 | converged |

Under the default schedule the release from A ends 0.002 px above A's median,
and B released again ends at 0.6760, below B held; neither reaches A's own
median, so a release is not guaranteed to end at or below the held
adjustment's median. Under `[(12, 2)]` alone the round still ends at the
monotonicity limit, and now says so. Under `[(50, 5)]` alone the spline does
not move: the round takes 8 held steps and no full step, because from an end
slope of 5.47 the full step at each iteration's `λ` overshoots past the limit,
the held step at that `λ` is accepted and halves `λ`, and the next full step
overshoots further. The default schedule's first round does the same; its
second takes 5 full and 13 held steps and settles at the guard, and its third
converges on 23 full steps.

**The synthetic tail fixture.** The tests of
[bundle-adjustment.md](bundle-adjustment.md) § "Testing requirements" ("A
weakly supported spline tail") use this case scaled down: `f = 250`, 10 images,
a 20-coefficient spline over `[0, 0.835]`, at most 1, 3, 12 and 30
observations in the last four knot spans, ±0.3 px of noise, and A's last three
coefficients as the start (end slope 5.61), with the poses and points settled
with the lens held. The runs below release the focal and the spline under the
default schedule, over 20 noise draws (seeds 3, 7, 11, 19, 23, 31, 41, 53, 61,
77, 89, 101, 127, 151, 199, 211, 307, 401, 509 and 1009) at 1000, 1500 and
2000 points, 60 runs in all.

At 1500 points and seed 77, without either change the release ends with an
end slope of 0.000 and reports the final round converged, while the median
falls from 0.1568 to 0.1525 px, so on this scene the median does not show the
stall. With the tail regularization alone, and with both changes, it ends at
an end slope of 1.57, converged, at 0.1530 px.

*A coefficient that gained support within a round ran away.* With both
changes but with a coefficient released in every linearization where it had
support, the tail ran away on 5 of the 60 runs: the last coefficient to
32,152 (1000 points, seed 101), 69,425 (seed 199) and 5.3 million (2000
points, seed 307), and the last two to 1.74 and 4.16 (1000 points, seed 509,
end slope 166) and 0.596 and 1.342 (2000 points, seed 19, end slope 52). On
1000 points and seed 101 the outermost observation sits at `ρ = 0.7943`,
0.003 inside the last span, and the poses and points the round starts from
put it just short of that span: the last coefficient's data diagonal is
exactly 0 at the round's first linearization, so no penalty row includes it,
and 8.5·10⁻⁴ at the second, falling from there, against 6.5·10⁴ for the
coefficient two places before it. Released from the second linearization
with that column and no row, it rose on every accepted step, to 26,913 by the
round's last iteration and 32,152 at its end, and the next round, whose rows
include it, started from a penalty cost of 7.2·10⁹. With such a
coefficient kept pinned until the next round, as the kernel then did, no
coefficient of the 60 runs was larger than 0.112 in magnitude. The kernel now
gives every coefficient its penalty rows from the round's start, which holds
such a coefficient as it gains support.

*What the 60 runs showed with the coefficient held until the next round.*
These runs left out every penalty row that touched an unsupported
coefficient and held that coefficient at its input.
Every spline comes back monotone, and the median rises by at most 0.0004 px
(1500 points, seed 31) against a held adjustment from the same start. No
final round reports convergence with the end slope under 0.05: the two runs
that end under it, at 0.0001 (1000 points, seed 199) and 0.010 (2000 points,
seed 41), report not converged. The end slope is not bounded away from 0 or
from the start, for two reasons:

- On 12 runs the last coefficient was pinned for the whole solve and kept its
  starting 0.0601 while its neighbours moved, which left an end slope
  between 0.0001 and 9.3 (9.31 at 2000 points and seed 307, 9.12 at 1000
  points and seed 101): a coefficient no data reached held its value.
- On 9 runs the last coefficient had support from the one observation in the
  last span, and its penalty row did not keep the end slope near 1: it ended
  between 0.010 and 0.40 (for example 0.14 at 1500 points and seed 11, 0.18
  at 1000 points and seed 41, both converged). This is the penalty weight.

*With every row kept and the penalty placing an unsupported coefficient.*
This is the kernel's rule: every interior row enters, a coefficient with no
support at the round's start is placed by the penalty, and one that loses its
support within a round keeps its value until the next. Over the same 60 runs
every spline comes back monotone, no coefficient is larger than 0.112 in magnitude (0.1121 at 1000
points and seed 127), and the median rises by at most 0.0004 px against a held
adjustment from the same start (1500 points, seed 31). No final round reports
convergence with the end slope under 0.05: the three runs that end under it,
at 0.0024 (2000 points, seed 151), 0.0053 (2000 points, seed 401) and 0.037
(2000 points, seed 19), report not converged. The end slopes run from 0.0024
to 3.16, and the wild start survives on none of the runs:

- On 17 runs no observation reaches the last span at the end of the solve,
  and the last coefficient continues the two before it in a straight line,
  to a second difference below 10⁻⁸. The end slope is then the continuation's,
  between 0.0053 and 3.16 (3.16 at 1500 points and seed 23, a tail of
  0.0185, 0.0502 and 0.0819; 0.0053 at 2000 points and seed 401, where the
  continuation ends near the monotonicity limit and the round is not
  converged).
- On the 43 runs in which an observation reaches the last span, the end
  slope ends below 0.5 on 11, between 0.0024 and 0.42 (0.14 at 1500 points
  and seed 11, 0.18 at 1000 points and seed 41, both converged, as before).
  This is the penalty weight, kept as a decision for now
  ([bundle-adjustment.md](bundle-adjustment.md) § "The solve", "The weight is
  a decision for now").

The end slope's upper value of 3.16 is larger than the coefficients' slope
suggests because the clamped knots bunch at the domain end: the last two
Greville abscissae are a third of a knot span apart, against a full span
between interior ones, so a tail that is straight in the coefficients has
three times the slope departure `δ'` at the domain end that it has inside
the domain.

On the `dino_dog_toy` case every coefficient has support (the last from its
one observation), so the rule changes nothing there: under the default
schedule A goes from 0.674 to 0.6761 px with a tail of 0.0130, 0.0164 and
0.0159 and an end slope of 0.97, and B from 0.7928 to 0.6760 px with a tail
of 0.0131, 0.0163 and 0.0158 and an end slope of 0.96, both converged: the
medians and tails of the run with both changes above.

## Per-image intrinsic adjustments

These measurements are behind the per-image release
([bundle-adjustment.md](bundle-adjustment.md#the-per-image-release)) and the
`.sfmr` columns it writes
([sfmr-file-format.md](../../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13)).

### The evidence for adjusting per image

The first measurements were made by an experiment outside the tree, a joint
bundle adjustment of poses, points and intrinsics with a Huber loss and nothing
trimmed. Each is a **held-out** median reprojection error: the solve is fitted
on half the points; the other half are then triangulated with the poses and
intrinsics fixed, and their reprojection error is measured. A parameter that
only fits noise lowers the fitted half's error and not the held-out half's.

**`dino_dog_toy`** (85 photographs, a Pixel 7 Pro main camera, 6.81 mm f/1.85
with optical image stabilisation, 2040×1536, EXIF subject distance 0.15 to
0.45 m). Its candidate ground truth has one shared `SFMTOOL_PINHOLE` camera, and
its residuals grow radially toward the image edge in a way no single shared
camera fits.

| Released | Held-out median |
|----------|-----------------|
| shared focal and radial distortion (baseline) | 0.818 px |
| + shared principal point, tangential, thin prism, aspect and skew | 0.757 px |
| + per-image focal | 0.732 px |
| + per-image principal point | 0.322 px |
| + per-image focal and principal point | 0.280 px |

- The per-image principal-point offsets spread over about 21 px in x and 20 px
  in y (1.4% and 1.3% of the short side, 1536 px), about a shared offset of
  roughly (−7, +19) px. They drift along the capture sequence and correlate only
  weakly with the direction of gravity in the camera frame, which fits the
  stabilised lens's position at the moment of capture.
- The per-image focal scale spreads over 1.3 to 1.6%, and correlates at −0.89
  with the EXIF `SubjectDistance`: focus breathing.
- Without a prior, one image with 63 observations moved its principal point by
  −626 px, which is why the per-image release has a prior.

**`seoul_bull_sculpture`** (17 images, 270×480, `SIMPLE_RADIAL`, its checked-in
ground truth) shows the same effect at a smaller size: held-out median 0.233 px
at the baseline, 0.199 px with a shared principal point (a shared x offset of
about −10.6 px), 0.142 px with a per-image principal point. The per-image
offsets spread over about 3.2 px in x and 3.7 px in y (1.2% and 1.4% of the
short side, 270 px), and the focal scale over 0.5%.

Kerry Park (two fisheyes on a rig) was not measured. Both datasets measured are
hand-curated ground truths, cleaner than what a solve produces: their noise
level is low and their adjustments spread less than an ordinary
reconstruction's would. The spreads show that the effect is real and how large
it is in these two, not what width a prior should have in general, which is why
the release derives the width from each solve instead of fixing it. The
experiment is not reproducible in the tree: the dino ground truth is not checked
in, and a script that repeats the held-out measurement on the checked-in ground
truths is an [open question](bundle-adjustment.md#open-questions).

### The release on the dino candidate

The kernel's own release, run through `EditedReconstruction.bundle_adjust` on
`dino_dog_toy_ground_truth_candidate_v009.sfmr` (85 images, one
`SFMTOOL_PINHOLE` camera of 2040×1536 whose spline has four coefficients, 21557
observations with inline keypoints, not checked in). These are fitted medians,
the report's `median_residual_after`, not held-out ones, under the kernel's
soft-L1 rather than the experiment's Huber; the start's median is 0.898 px.

| Released | `[(inf, 1)]` | default schedule |
|----------|--------------|------------------|
| nothing (poses and points) | 0.758 px | 0.718 px |
| shared principal point | 0.715 px | 0.695 px |
| per-image principal point | 0.322 px | 0.320 px |
| per-image focal and principal point | 0.286 px | 0.285 px |
| the same, with the spline | 0.286 px | 0.284 px |

- Released per image alone, the principal point's derived width is 0.0131 of
  the short side (20.1 px) under `[(inf, 1)]` and 0.0130 (19.9 px) under the
  default schedule, and the solved offsets' standard deviation 19.5 px on each
  axis, against the experiment's 20 to 21 px. The camera's principal point moves
  to (1013.7, 789.6) px, an offset of (−6.3, +21.6) px from the centre. Six and
  seven images end beyond two widths.
- With the focal released per image as well, the focal scale's width is 1.2%
  under both schedules and the scales' standard deviation 1.2%, against the
  experiment's 1.3 to 1.6%; the principal point's width is 1.2 to 1.3%. Both
  focal figures are of the log focal scale `ln(1 + s)`, which the release
  solves; the stored scales `s` have the same standard deviation to 0.5%, and
  their arithmetic mean ends at `7.5·10⁻⁵` where the mean of `ln(1 + s)` is
  zero, about `var(s)/2`.
- No point is deleted in any of the runs.

The derivation first weighted each image by `1/v`, its own variance. On this
reconstruction every image's focal scale is determined to a variance far under
the spread (`√v` about 0.0004 against a spread of 0.012), and that weight let
the few best-determined images decide the focal width alone: 0.37 to 0.50% of
the focal, half the scales' standard deviation of 0.93 to 1.0%, with 38 to 46
of the 85 images beyond two widths. Weighting each image by `1/(s² + v)`, its
precision as a sample of the spread, gives the widths above, with six to ten
images beyond two widths.

### Recovery on synthetic scenes

On the per-image tests' scenes (60° pinhole, images looking from a cap of
directions at points of their own), a planted spread of offsets of 0.4% of the
short side over forty images is derived 21 to 31% above it, on three noise
draws, with sixty observations per image at 0.5 px of noise, and 4 to 47% above
it with 25 observations at 0.7 px, where each image's variance is about the
spread's square: high, as the derivation's leftover error is expected to be.
Deriving again on the same observations moves the width by under 2% over five
more rounds. Images that all look at the same point,
from one arc or from a cap of directions, leave the per-image focal and
principal point a family of exact solutions, the critical motion for a varying
focal: from noise-free observations such scenes end with residuals under
`10⁻⁶` px and offsets 0.4 to 0.9 px from the planted ones, which only the prior
chooses among.

## Per-image radial corrections and rolling-shutter rates

These measurements are behind the two further per-image corrections of an
`SFMTOOL_PINHOLE` camera, each image's radial correction and its
rolling-shutter rates
([../camera/sfmtool-pinhole-kernels.md § Per-image
corrections](../camera/sfmtool-pinhole-kernels.md#per-image-corrections)), and
their release
([bundle-adjustment.md § The per-image
corrections](bundle-adjustment.md#the-per-image-corrections)). They bear on
five decisions: the radial correction has two coefficients; the rolling
shutter is the full rotation field on a known scanlines setting rather than a
per-image shear; a camera whose setting is not known runs one solve with rows
and one with columns rather than solving for a weight between them; the mean
of each correction over a camera's images is held; and the prior widths' loose
cap and floor.

### Cross-validation with a scratch solver

These numbers come from an independent bundle adjustment written with scipy
in a scratch directory outside the tree, **not from the kernel**. They show
that the corrections predict held-out points better; they do not show the
physical cause.

**Data.** `dino_dog_toy_ground_truth_candidate_v012.sfmr` in the maintainer's
local `dino_dog_toy_ws` (not checked in): 85 handheld Pixel 7 Pro images at
2040×1536, 1176 points, 23731 observations; one `SFMTOOL_PINHOLE` camera with 8
spline coefficients, `bspline_rho_max` 0.84, focal 1447.8 px, with per-image
focal and principal point released. EXIF focus distance 0.15 to 0.48 m,
exposures mostly 1/120 s, every EXIF orientation 1.

**Method.** Plain least squares keeping every observation (no trim, no robust
loss), with the per-image focal and principal-point priors at the kernel's
derived widths for v012 (0.0134 and 0.0131). 5-fold cross-validation by point:
each fold's held-out points are re-triangulated through the trained cameras,
the corrections included, and their reprojection RMS over all observations is
the score. Solves were capped at 150 to 400 evaluations; 100 against 400
changed the held-out RMS by under 0.0003 px.

The scratch solver wrote the rolling-shutter term in the kernels' optical frame
(y down) with the readout position in focal units: `t_o = (v − cy_i)/f_i` for
rows and `t_o = (u − cx_i)/f_i` for columns. For each image the stored model's
readout position is a fixed multiple of it, `t = (f_i/m)·t_o` with
`m = min(width, height)`, and matching the two displacements gives the stored
rates as `ω = (m/f_i)·(ωx, −ωy, −ωz)` of the scratch solver's, for rows and for
columns. `m/f_i` is about 1.06 here (`1536/1447.8`), varying by the ±1.3% of the
per-image focal scale. So the two are the same family of maps image by image,
and every fit and held-out score here carries over unchanged. Two things
differ slightly: a prior width in the stored unit is about 1.06 times the
scratch solver's width quoted below, and the kernel's mean hold weights the
images by `m/f_i`, which varies too little to matter.

**Held-out RMS, px:**

| Terms added to v012 | Held-out RMS |
|---|---|
| none (baseline, as v012) | 0.386 |
| shear alone | 0.344 |
| rolling shutter, rows, alone | 0.331 (better than shear in all 5 folds) |
| 2-coefficient correction (σ 2·10⁻³) + rolling shutter, rows | **0.2304** |
| 2-coefficient correction + rolling shutter, columns | 0.2374 |
| 2-coefficient correction + shear | 0.2384 |
| 2-coefficient correction + shear + rolling shutter, rows | 0.2254 |

Rows beat both columns and shear in all 5 folds. Adding shear on top of
rolling shutter with rows is better in all 5 folds too, but the two terms
inflate each other: 1.55 px and 1.9 px at the image edge together, against
0.76 px and 1.34 px apart, so the bundle adjustment does not combine them. The
shear is `Δx = a·x + s·y`, `Δy = s·x − a·y` per image, two parameters, with no
scanlines setting: to first order, rolling shutter with rows or with columns
is a linear map of the image plane, and once the part a rotation of the pose
and a change of focal absorb is removed, what remains is this traceless
symmetric map. The full rolling-shutter field fits better in every fold and
has a physical reading, so it is the correction the kernel has; whether to
offer the shear as a fallback is an [open
question](../camera/sfmtool-pinhole-kernels.md#open-questions).

**One coefficient against two.** At a tighter prior (σ 5·10⁻⁴), a
one-parameter quadratic correction, δ ∝ ρ² (a multiple of `(ρ/ρ_max)²` added to
`δ`), with rolling shutter gives 0.257 against 0.255 for the two-coefficient
correction with rolling shutter, the two-coefficient form better in all 5
folds. The `k1`-type one-parameter form, δ ∝ ρ³, was not tested in a re-solve;
both δ ∝ ρ² and δ ∝ ρ³ lie in the span of the two-coefficient basis.

**Prior widths.** For the radial correction, σ is quoted as `δ` at the domain
end (`e_1`, since `B1(ρ_max) = 1`):

| Radial σ | Held-out RMS | Note |
|---|---|---|
| 2.5·10⁻⁴ | 0.281 | rate mean free |
| 5·10⁻⁴ | 0.255 | rate mean free |
| 1·10⁻³ | 0.238 | rate mean free |
| 2·10⁻³ | 0.230 | rate mean free |
| 2·10⁻³ | 0.2384 | mean held, with shear |
| 4·10⁻³ | 0.2374 | mean held, with shear |
| 8·10⁻³ | 0.2374 | mean held, with shear |

| Rate σ | 1·10⁻³ | 2·10⁻³ | 4·10⁻³ | 8·10⁻³ |
|---|---|---|---|---|
| Held-out RMS | 0.2553 | 0.2549 | 0.2548 | 0.2548 |

The held-out error is flat for the radial correction at σ ≥ 4·10⁻³ and for the
rates at σ ≥ 2·10⁻³: past those widths the priors do no work beyond the mean
hold. This is why `10⁻²` is loose enough for the derived widths' cap.

With the mean of the rates free, the mean `ωx` over the images grew from
−7·10⁻⁴ to −1.7·10⁻³ as the rate prior widened, with no change in held-out
error (in the scratch solver's units; in the stored unit that `ωx` has the same
sign and is about 6% larger, −7.4·10⁻⁴ to −1.8·10⁻³). The mean rate is a
near-gauge, which is why the bundle adjustment holds it.

**By distance from the centre** (held-out RMS, px):

| ρ | [0, 0.3) | [0.3, 0.5) | [0.5, 0.6) | [0.6, 0.7) | [0.7, 0.75) | [0.75, 0.83) |
|---|---|---|---|---|---|---|
| baseline | 0.330 | 0.395 | 0.436 | 0.549 | 0.670 | 0.612 |
| 2-coefficient + rolling shutter, rows | 0.196 | 0.242 | 0.255 | 0.307 | 0.362 | 0.576 |

The outermost band has about 85 observations.

**Fitted sizes.**

- The rolling-shutter displacement at the centre of the bottom edge has a
  median of 1.3 px and a maximum of 5.5 px; its lag-1 autocorrelation along the
  capture sequence is about 0.4, so it changes from one photograph to the next
  rather than following a slow drift.
- The per-image radial correction is several pixels at ρ = 0.75 and
  correlates at −0.98 with the EXIF focus distance in dioptres, which is what
  focus-dependent distortion would give.
- The poses rotate by a median of 0.15° to 0.24° when the terms are added.
- Earlier residual fits, with the poses and points held, found no missing
  shared lens term: decentering, thin prism, aspect ratio and skew were all at
  the noise level. Only per-image fields remained, which is why no other part
  of the lens is adjusted per image.

**A continuous readout blend.**
`t = λ·(v − cy_i)/m + (1 − λ)·(u − cx_i)/m`, the rows position at `λ = 1` and
the columns position at `λ = 0`, with one `λ ∈ [0, 1]` per camera and shared
rates. The scratch solver wrote it in focal units in the optical frame,
`λ·(v − cy_i)/f_i + (1 − λ)·(u − cx_i)/f_i`, which is this `t` times `m/f_i`,
the same fit with the rates rescaled as above. The solve's cost (px², priors
included) at fixed `λ`:

| λ | 0 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|
| cost | 1190.3 | 1196.1 | 1187.1 | 1150.5 | 1134.0 |

The cost is not convex in `λ`: the wrong choice (`λ = 0`, columns, here) is a
local minimum. Released from 0, `λ` stayed at 0; released from 0.5 and from 1,
it ended at 1, rows. Rows and columns differ by about 56 px², about 840 in χ²
units at the 0.258 px noise level. This is why a camera whose scanlines are
`UNKNOWN` runs two solves rather than solving for a weight
([../reconstruction/bundle-adjust.md § The rolling-shutter
scanlines](../reconstruction/bundle-adjust.md#the-rolling-shutter-scanlines)).

**A focus-tied radial correction.** One shared two-coefficient shape per
camera, scaled for each image by its centred EXIF focus distance in dioptres
(two parameters per camera rather than two per image). With shear it gives
0.2467, and 0.706 in the outermost band, worse than free per-image coefficients
there. With a small free per-image remainder (σ 5·10⁻⁴) and shear it gives
0.2361, better than free coefficients with shear in all 5 folds. It needs a
focus value for every image: EXIF `SubjectDistance`, or the solved per-image
focal scale as a stand-in, which correlates at 0.96 with the dioptres. Whether
to add it is an [open
question](../camera/sfmtool-pinhole-kernels.md#open-questions).

**Caveats.** One capture from one phone. A scratch solver with plain least
squares rather than the kernel's soft-L1 and trim schedule, and capped
evaluations. Cross-validation shows that the terms predict held-out points
better, not what physically causes the residual they remove.

### The kernel's release on the dino candidate

The kernel, through its binding, on
`dino_dog_toy_ground_truth_candidate_v012.sfmr` in the maintainer's local
`dino_dog_toy_ws` (not checked in), with 85 images, 1245 points and 24931
observations (the file gained points after the scratch experiments' 1176 and
23731), keeping every observation (one round of `(inf, 1000)`, which a derived
prior runs twice) with the free points kept in their representation, releasing
the focal, the spline, each image's focal and principal point, and with them
each image's radial correction and rates on `ROWS`. Training RMS per
component, with no held-out score:

| Solve | Training RMS, px | Time |
|---|---|---|
| as stored, no step | 0.2500 | |
| re-solved with no new terms | 0.2494 | 6.6 s |
| with both corrections, rows | **0.1443** | 13.5 to 20 s |
| with both corrections, columns | 0.1486 | 15 s |

against the scratch solver's 0.256 without the terms and 0.149 to 0.164 with
them (two-coefficient correction and rows, fixed widths). On rows the derived
widths were `2.96·10⁻³` for the radial correction and `1.28·10⁻³` for the rates
(the focal's `0.0150`, the principal point's `0.0131`). The fitted radial
correction has a median size of 1.9 px at `ρ = 0.5`, 4.9 px at `ρ = 0.75` and
6.6 px at `ρ_max`, with a largest of 22.5 px at `ρ = 0.75`; its coefficients
spread `2.0·10⁻³` in `e_0` and `6.9·10⁻³` in `e_1`, so the one width pooled over
both sits near `e_0`'s spread and holds `e_1` more tightly than its own spread
would (an [open question](bundle-adjustment.md#open-questions)). The rates
spread `1.1·10⁻³` to `1.5·10⁻³` per component, and their displacement at the
centre of the bottom edge has a median of 1.0 px and a largest of 5.3 px,
against the scratch solver's 1.3 and 5.5 px. The mean rates ended within
`2·10⁻⁷` of zero, their start, under `2·10⁻⁴` of their spread, and the mean
radial correction was zero to rounding after its move into the spline: the
mean hold of width `10⁻⁶` holds the mean far inside any width.

The reconstruction-level adjustment's race on the same file, which chose rows,
is in
[../reconstruction/bundle-adjust-measurements.md](../reconstruction/bundle-adjust-measurements.md#the-first-race-on-the-dino-candidate).
