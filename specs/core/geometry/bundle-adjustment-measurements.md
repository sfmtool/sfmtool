# Staged Bundle Adjustment Measurements

This file records the measurements behind the free-point choices of
[bundle-adjustment.md](bundle-adjustment.md#free-points-inverse-depth-and-the-storage-decision).
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

