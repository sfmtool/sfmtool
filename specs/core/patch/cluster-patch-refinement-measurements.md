# Cluster-Patch Piecewise Refinement Measurements

This file records the fleet measurements behind the piecewise refinement of [cluster-patch-refinement.md](cluster-patch-refinement.md#piecewise-refinement), and the ground-truth measurements behind [cell-plane-normals.md](cell-plane-normals.md). The piecewise refinement registers each kept member's nine cells against the cluster's template, starting at the member's cascade shape, fits a robust affine map to the cell shifts to find the cells that disagree with the others, and stores the shifts with a status per cell in the `.matches` file. By default it also applies the fitted map as an update of the shape, by a loop that keeps the cascade's ZNCC from falling (`PiecewiseParams::move_shape`, on by default); with `move_shape` off it is a measurement only and leaves the member's shape as the cascade found it. The measurements bear on whether the stage runs by default (`sfm cluster-patches --piecewise`), and on the defaults of `min_cell_zncc` (provisionally `0.8`) and of `min_cell_curvature` (provisionally `0.02`).

The first five sections were measured at commit `f787be3b` on 43 fleet entries, when the stage was a loop that applied every fitted update to the shape: agreement with the cascade's shapes, the cell statistics, the wall time, and the seed stage on the two checked-in ground truths. The later sections repeat parts of that on five entries: after the loop was given an acceptance rule that keeps the cascade's ZNCC, after the default was changed to the measurement with the loop behind `move_shape`, and after the robust fit's residual scale was changed to the factor for two-dimensional residuals. The next two sections measure the cell plane normals against the two checked-in ground truths. The two after them record a blind human review of the loop's moved shapes against the cascade's, which reversed the earlier decision to leave the shape to the cascade, and repeat the subset with the loop as the default. The next measures two gates that read each member's own patch again at its refined shape, against the ground truths of four of the five entries, and the last repeats the cell plane normals on files written at the current defaults. While these measurements were taken the stage was specified in a draft proposal, since folded into the spec; "the draft" in the sections below means that proposal.

## Setup of the fleet run (f787be3b)

- **Code.** Commit `f787be3b` (branch `bootstrap-core-migration`), with the extension rebuilt by `pixi run maturin develop --release`.
- **Machine.** Windows 11, Intel Core i9-14900HX (32 logical processors), 63.7 GB RAM. During the refinement runs another checkout on the same machine was running a `pytest -n auto` suite, so the machine was not idle. [Wall time](#wall-time) says how the timings account for that.
- **Data.** The 42 workspaces of `C:/DataSets/workspace-prep/wslist.txt` plus `C:/DataSets/SeoulBull` (the 17 `seoul_bull_sculpture` images): 43 entries. An entry under `PhotogrammetryVids/` is named by its capture id, and `SpainSoapmaker/ws2` and `SfmExperimentation/vid2` carry their parent directory's name. Each entry's input is the unrefined `*-clusters.matches` behind the first `*-clusters-patches.matches` in sorted order, the file the seed reads. Two workspaces hold more than one clusters file; only that one was refined.
- **Runs.** Each entry ran `sfm cluster-patches -i <clusters> -o <scratch> --patch-size 12` twice, sequentially, with `SFMTOOL_PROFILE=1` and the other options at their defaults (resolution 25, `min_zncc` 0.85, `max_shift` 3, self-similarity bar 2.5):
  - the *cascade* run, with `--no-piecewise`;
  - the *piecewise* run, with `--piecewise` at the default piecewise settings (bound 2.0, `min_cell_zncc` 0.8, `min_cell_curvature` 0.02, tolerance 0.05, cap 5).

  Outputs went to a scratch directory, and no workspace file was written.
- **Baseline.** The cascade run, not the `*-clusters-patches.matches` stored in each workspace. Those were refined by an older build. On `KerryPark480` and `MossyRailing` they differ from this commit's cascade in status and shape (18,321 against 15,986 and 712,733 against 691,998 kept members). On `SeoulBull`, refined the day before, they are identical.
- **Units.** A grid px is one template sample, `patch_size / resolution` = 0.48 keypoint-frame units. A displacement in photograph px is converted to grid px through the inverse of the member's cascade shape. The parity table gives each entry's median photograph px per grid px, which runs from 0.63 to 2.99.

## Parity with the cascade

_Measured at `f787be3b` on the loop that applied every fitted update to the shape, which the acceptance rule has since replaced._

_What the human review corrects: this section judges the loop by its agreement with the cascade, which counts a correction of the cascade's shape as an error. A blind human review preferred the loop's moved shapes to the cascade's 37 to 3, and the loop is now the default; see [Human review of moved shapes](#human-review-of-moved-shapes-2026-10-09)._

**Question.** On members both runs keep, how far does the piecewise loop move the cascade's shape? The draft expects agreement "within tolerance", and this measurement finds out what that tolerance is in practice.

**Method.** For every member with status `kept` in both files, three movements are measured in grid px and in photograph px:

- the change of refined position;
- the change of shape, measured as the largest movement of the nine cell centres (at ±4 keypoint units) under the shape change alone;
- the total, the largest movement of a cell centre under the position and shape changes together.

The measurement also records the change of `member_zncc`, the count of members whose status differs, and whether every member that neither run keeps is bit-identical in position, shape and ZNCC.

**Result.** No member's status differs between the two runs on any entry, and every member that is not kept is bit-identical. The stage touches only kept members, as specified.

The kept members do move. On each entry 80% to 99.6% of them move. Per entry, the total cell-centre movement has a median of 0.16 to 0.71 grid px, a 95th percentile of 0.64 to 2.19 grid px (1.5 to 6.3 photograph px), and a maximum of 5 to 10 grid px.

The movement has no bias. This was checked on five entries (`SeoulBull`, `KerryPark480`, `fleetws`, `OmniTemple1`, `20250712_204251146`). On them the mean position change is under 0.006 grid px, the mean log scale change is under 0.002, and the mean rotation is under 0.05°. Between the 5th and 95th percentiles, the log scale change spans up to about ±0.07 and the rotation up to about ±3°.

The whole-member windowed ZNCC falls. Per entry, the median change is −0.001 to −0.006, the 5th percentile is −0.009 to −0.037, and the 95th percentile is 0.000 on every entry. The cascade maximizes that score, so any move away from its optimum lowers it.

So the shapes the loop returned agreed with the cascade only to within about 2 grid px at the 95th percentile, which is the size of the cell search bound itself. On that loop the draft's check at the time, that "the converged affine shapes agree with the cascade's within tolerance", failed. The movement is scatter around the cascade's shape, not a shift in one direction. Without a pose-dependent truth for the shapes, this measurement cannot say whether the scatter is error or a real refinement. The ZNCC drop and the [seed result](#seed-stage-on-the-ground-truth-entries) both point to error. A test that would settle it: score both files' shapes against the `seoul_bull_sculpture` and `kerry_park` ground truths, by projecting each kept member's cell centres through the ground-truth poses and a plane.

| entry | kept | moved % | status differ | dpos grid p50/p95/max | dshape grid p50/p95/max | dtotal grid p50/p95/max | dtotal photo p50/p95/max | dzncc p5/p50/p95 | photo px per grid px p50 |
|---|---|---|---|---|---|---|---|---|---|
| 20240614_203547691 | 1111255 | 97.1 | 0 | 0.06/0.21/2.87 | 0.14/0.51/4.76 | 0.19/0.67/6.38 | 0.46/2.16/32.00 | -0.016/-0.002/0.000 | 2.25 |
| 20240614_224244438 | 1386390 | 97.9 | 0 | 0.05/0.20/2.40 | 0.11/0.48/4.91 | 0.16/0.64/6.27 | 0.35/1.80/38.87 | -0.015/-0.001/0.000 | 2.05 |
| 20240614_224422531 | 1245364 | 96.5 | 0 | 0.07/0.31/2.84 | 0.18/0.70/5.79 | 0.24/0.94/7.73 | 0.51/2.43/65.15 | -0.019/-0.002/0.000 | 1.96 |
| 20240614_225938434 | 1239391 | 97.5 | 0 | 0.05/0.25/2.96 | 0.12/0.56/5.59 | 0.17/0.76/6.11 | 0.37/2.26/29.79 | -0.014/-0.001/0.000 | 2.05 |
| 20240617_204133531 | 350437 | 89.6 | 0 | 0.10/0.43/3.00 | 0.25/0.99/5.31 | 0.35/1.29/7.01 | 0.82/3.26/35.19 | -0.027/-0.004/0.000 | 2.15 |
| 20240618_001255975~2 | 464832 | 93.3 | 0 | 0.11/0.44/3.08 | 0.29/1.07/5.37 | 0.40/1.38/6.15 | 0.85/2.96/39.00 | -0.021/-0.004/0.000 | 1.94 |
| 20240702_224718414 | 562400 | 89.1 | 0 | 0.16/0.71/3.89 | 0.34/1.45/7.60 | 0.52/1.91/8.04 | 0.71/3.60/50.20 | -0.034/-0.005/0.000 | 1.25 |
| 20240713_205804006 | 1736056 | 98.6 | 0 | 0.05/0.26/3.30 | 0.12/0.61/5.98 | 0.17/0.82/7.57 | 0.38/1.97/31.16 | -0.012/-0.001/-0.000 | 2.06 |
| 20240906_081206935 | 490841 | 87.6 | 0 | 0.15/0.65/3.65 | 0.35/1.43/6.06 | 0.52/1.87/6.91 | 0.95/4.18/43.87 | -0.035/-0.005/0.000 | 1.69 |
| 20240915_071403318 | 1096006 | 95.1 | 0 | 0.10/0.40/3.29 | 0.26/1.00/5.17 | 0.36/1.27/7.50 | 0.71/2.81/38.61 | -0.020/-0.003/0.000 | 2.04 |
| 20240915_073428267 | 610227 | 90.5 | 0 | 0.21/0.82/4.43 | 0.47/1.59/5.87 | 0.70/2.10/7.62 | 1.30/4.15/50.78 | -0.034/-0.005/0.000 | 1.67 |
| 20240918_074134864 | 547642 | 92.8 | 0 | 0.21/0.84/4.38 | 0.48/1.63/6.06 | 0.71/2.16/9.24 | 0.73/2.96/57.05 | -0.037/-0.006/0.000 | 0.92 |
| 20240919_040613535 | 729466 | 91.7 | 0 | 0.13/0.59/6.55 | 0.32/1.31/6.08 | 0.45/1.71/7.00 | 0.84/3.41/47.52 | -0.029/-0.004/0.000 | 1.74 |
| 20240919_062358681 | 642485 | 88.4 | 0 | 0.19/0.76/4.59 | 0.42/1.57/7.37 | 0.62/2.06/9.19 | 1.17/4.47/58.76 | -0.036/-0.005/0.000 | 1.67 |
| 20250425_135433677 | 1290338 | 96.1 | 0 | 0.07/0.36/3.23 | 0.17/0.98/5.40 | 0.24/1.24/7.32 | 0.56/3.20/89.10 | -0.021/-0.002/0.000 | 2.17 |
| 20250425_141305472 | 704765 | 93.2 | 0 | 0.09/0.42/5.61 | 0.23/1.09/7.29 | 0.32/1.39/7.82 | 0.66/3.51/51.75 | -0.023/-0.003/0.000 | 2.03 |
| 20250426_150733286 | 297275 | 85.5 | 0 | 0.18/0.72/4.22 | 0.41/1.51/4.94 | 0.60/1.96/7.11 | 1.79/6.29/42.97 | -0.030/-0.004/0.000 | 2.99 |
| 20250706_223513674 | 549279 | 90.3 | 0 | 0.09/0.40/3.58 | 0.23/1.02/5.20 | 0.32/1.30/6.46 | 0.70/3.41/39.18 | -0.026/-0.004/0.000 | 2.08 |
| 20250712_195736354 | 1315125 | 96.9 | 0 | 0.08/0.35/2.57 | 0.20/0.85/5.67 | 0.27/1.11/6.75 | 0.56/2.68/49.29 | -0.018/-0.002/-0.000 | 1.84 |
| 20250712_202131684 | 1519297 | 97.6 | 0 | 0.06/0.27/2.41 | 0.16/0.66/5.80 | 0.22/0.86/6.36 | 0.49/2.25/45.44 | -0.016/-0.002/-0.000 | 2.01 |
| 20250712_204251146 | 962616 | 97.7 | 0 | 0.05/0.22/3.19 | 0.13/0.52/4.11 | 0.17/0.69/5.19 | 0.40/2.04/29.63 | -0.016/-0.002/-0.000 | 2.10 |
| 20250906_211742965 | 648097 | 94.9 | 0 | 0.15/0.66/5.58 | 0.35/1.36/8.49 | 0.51/1.80/10.21 | 0.47/2.29/30.04 | -0.028/-0.004/0.000 | 0.88 |
| 20250907_000240907 | 324535 | 92.4 | 0 | 0.18/0.76/5.21 | 0.37/1.50/6.68 | 0.56/1.98/8.12 | 0.48/2.59/31.23 | -0.032/-0.004/0.000 | 0.80 |
| 20250907_000422554 | 101475 | 90.8 | 0 | 0.20/0.75/4.17 | 0.46/1.56/6.63 | 0.67/2.04/8.15 | 0.60/3.69/36.92 | -0.033/-0.006/0.000 | 0.80 |
| 20250907_000742129 | 279368 | 94.5 | 0 | 0.17/0.72/4.20 | 0.39/1.47/6.39 | 0.57/1.94/9.40 | 0.52/2.71/43.36 | -0.031/-0.004/0.000 | 0.81 |
| 20250907_001316663 | 596016 | 92.8 | 0 | 0.13/0.58/4.77 | 0.32/1.27/6.72 | 0.45/1.67/8.45 | 0.57/2.53/34.12 | -0.030/-0.004/0.000 | 1.22 |
| 20250907_211111559 | 830382 | 95.2 | 0 | 0.09/0.42/3.78 | 0.23/0.97/6.39 | 0.33/1.28/7.69 | 0.42/2.08/27.38 | -0.023/-0.003/0.000 | 1.23 |
| AltonaGalleryInTheParkBoyReading | 555268 | 91.4 | 0 | 0.10/0.48/5.67 | 0.26/1.16/5.65 | 0.37/1.49/6.94 | 0.81/3.76/50.79 | -0.026/-0.003/0.000 | 2.11 |
| BadlandPanorama | 1411007 | 99.6 | 0 | 0.08/0.28/2.37 | 0.20/0.64/4.46 | 0.27/0.84/5.45 | 0.62/1.87/19.00 | -0.009/-0.002/-0.000 | 2.04 |
| DaeguArtMuseumTreeStumpExhibit | 511252 | 90.2 | 0 | 0.10/0.51/5.03 | 0.25/1.19/6.38 | 0.36/1.53/7.25 | 0.70/3.54/47.05 | -0.025/-0.004/0.000 | 1.87 |
| DaeguMuseumMasks | 427172 | 87.2 | 0 | 0.23/0.93/4.54 | 0.45/1.63/6.09 | 0.71/2.19/8.87 | 1.15/4.38/78.07 | -0.035/-0.005/0.000 | 1.51 |
| DnDTabletop | 1375116 | 95.8 | 0 | 0.09/0.41/2.63 | 0.23/1.05/5.57 | 0.31/1.34/6.51 | 0.88/3.85/35.79 | -0.018/-0.002/0.000 | 2.60 |
| KerryPark360 | 40038 | 90.6 | 0 | 0.14/0.54/2.62 | 0.34/1.26/6.02 | 0.48/1.65/7.75 | 0.49/2.32/31.03 | -0.027/-0.005/0.000 | 0.97 |
| KerryPark480 | 15986 | 91.3 | 0 | 0.17/0.62/2.46 | 0.38/1.27/4.38 | 0.54/1.67/6.41 | 0.37/1.50/13.81 | -0.026/-0.005/0.000 | 0.64 |
| MossyRailing | 691998 | 92.0 | 0 | 0.09/0.43/2.87 | 0.25/1.08/6.89 | 0.35/1.37/7.05 | 0.95/3.82/65.14 | -0.022/-0.003/0.000 | 2.53 |
| MurdoSmallAntiqueCat | 625198 | 90.6 | 0 | 0.11/0.49/3.72 | 0.27/1.18/7.21 | 0.39/1.51/8.12 | 0.93/4.06/51.77 | -0.026/-0.004/0.000 | 2.30 |
| OmniCoast | 229085 | 94.9 | 0 | 0.13/0.50/3.21 | 0.32/1.17/6.10 | 0.45/1.52/7.39 | 0.62/2.41/23.54 | -0.023/-0.003/0.000 | 1.22 |
| OmniHilltop | 511372 | 96.1 | 0 | 0.16/0.56/2.85 | 0.40/1.30/6.06 | 0.55/1.68/7.50 | 0.62/2.08/50.34 | -0.025/-0.005/-0.000 | 1.00 |
| OmniTemple1 | 157914 | 91.9 | 0 | 0.16/0.65/5.65 | 0.36/1.37/5.49 | 0.52/1.81/6.59 | 0.62/2.74/39.22 | -0.031/-0.005/0.000 | 1.04 |
| SeoulBull | 2953 | 87.5 | 0 | 0.21/0.80/1.95 | 0.46/1.55/3.67 | 0.69/1.98/5.25 | 0.49/1.66/9.12 | -0.031/-0.006/0.000 | 0.63 |
| SfmExperimentation_vid2 | 453385 | 85.8 | 0 | 0.20/0.78/3.53 | 0.41/1.58/6.07 | 0.64/2.05/8.08 | 1.37/5.00/50.99 | -0.032/-0.005/0.000 | 2.05 |
| SpainSoapmaker_ws2 | 965487 | 95.0 | 0 | 0.15/0.61/3.72 | 0.38/1.37/6.77 | 0.53/1.78/7.52 | 1.05/4.03/37.25 | -0.028/-0.004/0.000 | 1.79 |
| fleetws | 68722 | 79.8 | 0 | 0.20/0.84/3.70 | 0.31/1.57/6.72 | 0.56/2.09/9.42 | 0.84/4.41/37.58 | -0.036/-0.005/0.000 | 1.37 |

### The loop does not settle

**Question.** Is the movement a product of the stopping rule? That is, does the loop stop at its iteration cap before it has converged?

**Method.** Two sources:

- the piecewise run's `member_cell_iterations`;
- a sweep through the binding on seven entries (`SeoulBull`, `KerryPark480`, `fleetws`, `OmniTemple1`, `20250907_000422554`, `20250426_150733286`, `20240617_204133531`). The sweep reruns the refinement with the cap raised to 20, with the tolerance raised to 0.1 and to 0.2 grid px, and with both changes together (tolerance 0.2, cap 20). It reads the same images and seeds as the CLI. The table is under [Gate sweep](#gate-sweep).

**Result.** Pooled over the fleet, 23.9% of kept members stop at the cap of 5. Of the rest, 2.7% stop after one iteration, 34.4% after two, 28.3% after three and 10.7% after four. Per entry, the share at the cap runs from 5.7% to 54.3%.

Raising the cap does not make the loop converge:

- with the cap at 20, 16% to 30% of the swept entries' members still reach it;
- with the tolerance at 0.2 grid px and the cap at 20, 5% to 12% still reach it.

These members do not converge slowly. They alternate between updates larger than 0.2 grid px.

The stopping rule does not change how far the shapes move. In every stopping configuration of the sweep, the median of the total movement stays within 0.05 grid px of the default's, and the 95th percentile within 0.15 grid px. Members that run more iterations do move further. On `SeoulBull` the median absolute log scale change is 0.000 after one or two iterations and 0.024 at the cap.

The 0.05 grid px tolerance was below the noise of the cell readings. This bears on the draft's open question of whether the loop replaces the cascade or follows it. As built at `f787be3b`, the loop did not converge for a quarter of the members, so it could not replace the cascade until its update was damped, or until it stopped when an update no longer improved the fit. A run of the loop from the detection's shape alone was not measured.

## Cell statistics

_Measured at `f787be3b` on the loop that applied every fitted update to the shape, which the acceptance rule has since replaced._

**Question.** How are the kept members' cells distributed over the five statuses, and where do the two provisional gates sit in the distributions they cut? The answer should show whether a data-derived default exists for `min_cell_zncc` and for `min_cell_curvature`.

**Method.** Every cell of every member the piecewise run keeps, 267,059,817 cells over 43 entries, read from `member_cell_status`, `member_cell_zncc`, `member_cell_shift_px` and `member_cell_iterations`. A cell's curvature is computed in `read_cell` but **not stored** in the file, so its distribution cannot be read from the output. The [gate sweep](#gate-sweep) stands in for it. With the ZNCC gate off (`min_cell_zncc = −1`), the share of cells refused for curvature at a threshold `t` approximates the share whose curvature is below `t`. The approximation is not exact, because the loop's path changes with the threshold.

**Result, pooled.**

| status | share of cells |
|---|---|
| fitted | 77.62% |
| refused_curvature | 4.26% |
| refused_zncc | 5.73% |
| not_attempted | 5.73% |
| refused_bound | 6.67% |

The number of fitted cells per kept member is distributed as follows:

| fitted cells | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| share of kept members | 5.4% | 1.4% | 2.7% | 2.7% | 4.4% | 5.4% | 8.3% | 10.8% | 16.0% | 42.9% |

A member with no fitted cell kept its cascade shape.

The mix varies widely by entry. The fitted share runs from 38.0% (`fleetws`) to 93.4%, and the refused_bound share from 0.8% to 24.9%. On the harder entries, refusal at the search bound is the largest refusal class. Those cells have a median ZNCC of 0.80. Many of them are cells whose optimum lies just outside ±1.5 grid px, not cells over another surface.

Cell ZNCC quantiles, pooled:

| cells | n | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| fitted | 207,283,658 | 0.812 | 0.844 | 0.869 | 0.912 | 0.949 | 0.973 | 0.986 | 0.991 | 0.997 |
| refused_zncc | 15,303,899 | 0.263 | 0.453 | 0.543 | 0.655 | 0.731 | 0.771 | 0.789 | 0.795 | 0.799 |
| refused_curvature (best whole-pixel shift) | 11,363,375 | 0.643 | 0.834 | 0.887 | 0.940 | 0.969 | 0.984 | 0.992 | 0.994 | 0.997 |
| refused_bound (best whole-pixel shift) | 17,813,351 | 0.040 | 0.258 | 0.381 | 0.600 | 0.799 | 0.925 | 0.968 | 0.981 | 0.992 |

For a cell that reaches a sub-pixel peak (fitted or refused_zncc), the share below each ZNCC threshold is:

| below | 0.5 | 0.6 | 0.7 | 0.75 | **0.8** | 0.85 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|---|
| share of peaked cells | 0.5% | 1.1% | 2.6% | 4.1% | **6.8%** | 12.4% | 25.1% | 53.5% |

The density of those cells in bins of 0.05 rises monotonically up to the top bin, with one mode at 1 and no valley:

| bin | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| share | 0.05% | 0.08% | 0.11% | 0.16% | 0.24% | 0.36% | 0.57% | 0.93% | 1.59% | 2.70% | 5.71% | 12.85% | 28.85% | 45.71% |

The residual `|shift|` of fitted cells, in grid px:

| p1 | p5 | p25 | p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| 0.009 | 0.023 | 0.062 | 0.117 | 0.220 | 0.394 | 0.557 | 0.949 |

For scale, the draft's synthetic tilted plane at `h = [0.003, −0.002]` has a second-order term of up to 0.52 grid px.

**Reading on `min_cell_zncc`.** The distribution gives no boundary to put a default on. It has one mode, at 1, and a smooth tail with no valley, so any threshold is a choice of tail fraction. At 0.8 the bar refuses 6.8% of peaked cells, which is 5.7% of all cells: a small fraction, not a large one.

The refused_curvature cells have a median ZNCC of 0.969. As the draft predicts, flat cells correlate well at any shift, so ZNCC alone would not catch them.

On the swept entries, raising the bar to 0.9 refuses a further 7 to 20 percentage points of cells and lowers the median shape movement by 13% to 48% (to 0.3 to 0.55 grid px). Lowering it to 0.7 moves the shapes slightly more. Fewer fitted cells mean a smaller move away from the cascade. The threshold therefore changes the stage's output, and the parity numbers cannot choose it.

A data-derived default needs an outcome measure as a function of the threshold. Candidates are the normal error against the `seoul_bull_sculpture` ground truth (the draft's last Testing sentence) and the shape error against the ground-truth poses. Until that measure exists, 0.8 sits at the 7th percentile of peaked cells and can stay.

**Reading on `min_cell_curvature`.** Curvature is not stored, so only the sweep's proxy is available. The proxy is smooth, with no knee, and it varies widely between entries. The share of cells refused for curvature at each threshold:

| entry | 0 | 0.005 | 0.01 | 0.02 | 0.04 | 0.08 |
|---|---|---|---|---|---|---|
| `SeoulBull` | 0.2% | 1.1% | 2.6% | 6.5% | 14.7% | 24.9% |
| `20240617_204133531` | 0.1% | 0.3% | 0.7% | 1.9% | 5.3% | 13.3% |
| `20250426_150733286` | 0.6% | 3.1% | 6.8% | 14.2% | 25.0% | 30.8% |

At 0.02 the bar refuses 2% to 14% of cells depending on the capture. Pooled over the fleet at the default settings, 4.3% of cells are refused for curvature. Nothing in the proxy marks 0.02 as a boundary. Doubling the bar to 0.04 roughly doubles to triples the refusals, and there is no plateau on either side.

Settling this default needs the same outcome measure as the ZNCC bar. A first step is to store the curvature, or write it from a debug path, so that its distribution can be read directly instead of through reruns.

| entry | cells | fitted % | ref_curv % | ref_zncc % | not_att % | ref_bound % | members 0 fitted % | at cap (5) % | fitted res p50/p95 | fitted zncc p5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20240614_203547691 | 10001295 | 88.9 | 0.7 | 6.2 | 3.3 | 0.8 | 2.9 | 8.8 | 0.08/0.27 | 0.841 |
| 20240614_224244438 | 12477510 | 93.4 | 0.6 | 2.7 | 2.5 | 0.8 | 2.1 | 5.7 | 0.07/0.28 | 0.867 |
| 20240614_224422531 | 11208276 | 86.1 | 1.9 | 6.0 | 3.8 | 2.2 | 3.5 | 14.3 | 0.10/0.38 | 0.843 |
| 20240614_225938434 | 11154519 | 89.4 | 1.3 | 4.5 | 2.9 | 1.9 | 2.5 | 9.6 | 0.07/0.28 | 0.862 |
| 20240617_204133531 | 3153933 | 74.8 | 1.9 | 7.4 | 10.8 | 5.1 | 10.4 | 28.9 | 0.15/0.61 | 0.836 |
| 20240618_001255975~2 | 4183488 | 75.6 | 3.5 | 8.2 | 6.9 | 5.7 | 6.7 | 25.5 | 0.18/0.66 | 0.831 |
| 20240702_224718414 | 5061600 | 53.1 | 8.2 | 8.5 | 11.1 | 19.0 | 10.9 | 44.9 | 0.19/0.77 | 0.829 |
| 20240713_205804006 | 15624504 | 91.5 | 1.3 | 3.4 | 1.9 | 2.0 | 1.4 | 7.9 | 0.08/0.32 | 0.877 |
| 20240906_081206935 | 4417569 | 56.1 | 5.9 | 8.5 | 12.6 | 16.9 | 12.4 | 48.0 | 0.18/0.77 | 0.833 |
| 20240915_071403318 | 9864054 | 83.0 | 2.1 | 6.3 | 5.1 | 3.4 | 4.9 | 19.1 | 0.16/0.62 | 0.835 |
| 20240915_073428267 | 5492043 | 55.3 | 13.0 | 4.9 | 9.7 | 17.0 | 9.5 | 51.4 | 0.24/0.86 | 0.839 |
| 20240918_074134864 | 4928778 | 55.2 | 14.0 | 4.3 | 7.3 | 19.2 | 7.2 | 50.9 | 0.23/0.85 | 0.846 |
| 20240919_040613535 | 6565194 | 65.3 | 5.2 | 9.6 | 8.6 | 11.3 | 8.3 | 38.1 | 0.15/0.67 | 0.828 |
| 20240919_062358681 | 5782365 | 50.3 | 10.3 | 7.7 | 11.8 | 20.0 | 11.6 | 54.1 | 0.20/0.82 | 0.831 |
| 20250425_135433677 | 11613042 | 86.9 | 1.8 | 4.2 | 4.3 | 2.8 | 3.9 | 14.7 | 0.10/0.43 | 0.848 |
| 20250425_141305472 | 6342885 | 76.7 | 2.3 | 8.9 | 7.2 | 5.0 | 6.8 | 23.7 | 0.12/0.52 | 0.830 |
| 20250426_150733286 | 2675475 | 54.8 | 14.3 | 2.8 | 14.9 | 13.2 | 14.5 | 50.4 | 0.22/0.79 | 0.855 |
| 20250706_223513674 | 4943511 | 72.0 | 0.9 | 11.6 | 10.0 | 5.5 | 9.7 | 24.0 | 0.12/0.52 | 0.826 |
| 20250712_195736354 | 11836125 | 85.4 | 1.9 | 6.3 | 3.3 | 3.0 | 3.1 | 18.1 | 0.11/0.43 | 0.840 |
| 20250712_202131684 | 13673673 | 89.5 | 1.2 | 4.6 | 2.8 | 1.9 | 2.4 | 10.6 | 0.09/0.36 | 0.853 |
| 20250712_204251146 | 8663544 | 91.8 | 0.5 | 3.9 | 2.8 | 1.0 | 2.3 | 6.5 | 0.08/0.28 | 0.856 |
| 20250906_211742965 | 5832873 | 67.4 | 8.8 | 5.9 | 5.4 | 12.5 | 5.1 | 37.0 | 0.18/0.72 | 0.841 |
| 20250907_000240907 | 2920815 | 55.6 | 11.8 | 5.3 | 7.8 | 19.5 | 7.6 | 43.9 | 0.21/0.80 | 0.842 |
| 20250907_000422554 | 913275 | 53.4 | 12.8 | 4.5 | 9.5 | 19.8 | 9.2 | 51.6 | 0.27/0.91 | 0.838 |
| 20250907_000742129 | 2514312 | 59.3 | 13.0 | 4.6 | 5.8 | 17.4 | 5.5 | 44.5 | 0.19/0.77 | 0.849 |
| 20250907_001316663 | 5364144 | 70.9 | 5.5 | 5.6 | 7.6 | 10.4 | 7.2 | 35.1 | 0.17/0.73 | 0.846 |
| 20250907_211111559 | 7473438 | 81.1 | 2.1 | 6.9 | 5.2 | 4.7 | 4.8 | 20.1 | 0.14/0.57 | 0.837 |
| AltonaGalleryInTheParkBoyReading | 4997412 | 73.4 | 3.0 | 8.5 | 9.0 | 6.1 | 8.6 | 26.5 | 0.14/0.60 | 0.831 |
| BadlandPanorama | 12699063 | 91.3 | 5.0 | 1.7 | 0.7 | 1.3 | 0.4 | 11.3 | 0.13/0.46 | 0.873 |
| DaeguArtMuseumTreeStumpExhibit | 4601268 | 67.6 | 3.2 | 10.1 | 10.0 | 9.0 | 9.8 | 28.3 | 0.15/0.63 | 0.822 |
| DaeguMuseumMasks | 3844548 | 47.5 | 12.2 | 5.1 | 12.9 | 22.3 | 12.8 | 53.4 | 0.25/0.89 | 0.836 |
| DnDTabletop | 12376044 | 83.7 | 5.1 | 2.8 | 4.7 | 3.7 | 4.2 | 20.0 | 0.13/0.54 | 0.864 |
| KerryPark360 | 360342 | 63.6 | 2.6 | 13.8 | 9.5 | 10.4 | 9.4 | 38.9 | 0.18/0.71 | 0.823 |
| KerryPark480 | 143874 | 61.3 | 3.8 | 12.8 | 8.9 | 13.1 | 8.7 | 34.9 | 0.22/0.76 | 0.825 |
| MossyRailing | 6227982 | 77.4 | 2.3 | 7.3 | 8.5 | 4.5 | 8.0 | 24.3 | 0.15/0.60 | 0.831 |
| MurdoSmallAntiqueCat | 5626782 | 71.5 | 3.1 | 8.2 | 9.8 | 7.3 | 9.4 | 28.5 | 0.15/0.64 | 0.832 |
| OmniCoast | 2061765 | 75.2 | 5.0 | 6.9 | 5.2 | 7.7 | 5.1 | 29.8 | 0.18/0.68 | 0.839 |
| OmniHilltop | 4602348 | 73.6 | 4.6 | 9.0 | 3.9 | 8.9 | 3.9 | 35.7 | 0.21/0.76 | 0.830 |
| OmniTemple1 | 1421226 | 58.8 | 7.1 | 8.8 | 8.2 | 17.1 | 8.1 | 42.2 | 0.19/0.75 | 0.831 |
| SeoulBull | 26577 | 50.4 | 6.7 | 11.8 | 13.1 | 18.0 | 12.5 | 46.2 | 0.26/0.90 | 0.821 |
| SfmExperimentation_vid2 | 4080465 | 45.9 | 11.4 | 6.5 | 14.5 | 21.7 | 14.2 | 50.4 | 0.24/0.85 | 0.834 |
| SpainSoapmaker_ws2 | 8689383 | 65.5 | 10.9 | 6.3 | 5.3 | 12.1 | 5.0 | 43.2 | 0.17/0.69 | 0.848 |
| fleetws | 618498 | 38.0 | 8.7 | 8.1 | 20.2 | 24.9 | 20.2 | 54.3 | 0.24/0.91 | 0.825 |

### Gate sweep

**Question.** How do the cell mix, the share of members at the cap and the movement from the cascade respond to the two gates and to the stopping rule?

**Method.** `refine_cluster_patches` was called through the binding on the seven entries named above. The calls load the data the way the CLI does (images decoded with OpenCV, detection seeds scattered to per-image rows) and use the CLI's settings. Each configuration varies one setting from the default:

- `z` is `min_cell_zncc`;
- `c` is `min_cell_curvature`;
- `it` is `max_iterations`;
- `tol` is `update_tolerance_px`.

Movement is the total cell-centre movement against a cascade run in the same process. The members-at-cap column counts members at the configuration's own cap.

**Result.**

- The gates set how many cells are fitted, and through that how far the shapes move.
- The stopping rule sets how many members reach the cap, but not how far the shapes move.
- With every gate open (`z-1_c0.0`), only 0.1% to 0.9% of cells are refused for curvature. A cell rarely fails to reach a peak at all, so the curvature bar is what refuses the flat cells.
- The bound refusals stay within about three points of the default in every configuration except the strictest curvature bar (`c` 0.08), because the search bound, not the gates, sets them.

| entry | config | fitted % | ref_curv % | ref_zncc % | not_att % | ref_bound % | members 0 fitted % | at cap % | dtotal grid p50/p95 | fitted res p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| 20240617_204133531 | default | 74.8 | 1.9 | 7.4 | 10.8 | 5.1 | 10.4 | 28.9 | 0.35/1.29 | 0.149 |
| 20240617_204133531 | z-1_c0.0 | 84.7 | 0.1 | 0.0 | 10.6 | 4.5 | 10.2 | 33.1 | 0.37/1.44 | 0.174 |
| 20240617_204133531 | z-1_c0.005 | 84.5 | 0.3 | 0.0 | 10.7 | 4.5 | 10.2 | 33.1 | 0.37/1.43 | 0.173 |
| 20240617_204133531 | z-1_c0.01 | 84.0 | 0.7 | 0.0 | 10.7 | 4.6 | 10.3 | 33.0 | 0.36/1.42 | 0.172 |
| 20240617_204133531 | z-1_c0.02 | 82.5 | 1.9 | 0.0 | 11.0 | 4.5 | 10.6 | 32.8 | 0.36/1.39 | 0.169 |
| 20240617_204133531 | z-1_c0.04 | 78.3 | 5.3 | 0.0 | 11.9 | 4.5 | 11.5 | 32.4 | 0.35/1.32 | 0.161 |
| 20240617_204133531 | z-1_c0.08 | 67.9 | 13.3 | 0.0 | 14.6 | 4.1 | 14.2 | 30.0 | 0.33/1.18 | 0.143 |
| 20240617_204133531 | z0.7_c0.02 | 80.0 | 1.9 | 2.6 | 10.6 | 4.9 | 10.1 | 30.3 | 0.36/1.34 | 0.161 |
| 20240617_204133531 | z0.9_c0.02 | 54.8 | 1.9 | 24.9 | 13.3 | 5.1 | 12.9 | 23.4 | 0.30/1.13 | 0.120 |
| 20240617_204133531 | default_it20 | 74.6 | 1.9 | 7.4 | 11.0 | 5.1 | 10.6 | 15.7 | 0.35/1.31 | 0.149 |
| 20240617_204133531 | tol0.1 | 75.0 | 1.9 | 7.4 | 10.6 | 5.1 | 10.1 | 19.4 | 0.34/1.29 | 0.148 |
| 20240617_204133531 | tol0.2 | 75.2 | 2.0 | 7.5 | 10.2 | 5.2 | 9.7 | 10.7 | 0.33/1.27 | 0.148 |
| 20240617_204133531 | tol0.2_it20 | 75.1 | 2.0 | 7.5 | 10.2 | 5.2 | 9.8 | 5.4 | 0.33/1.27 | 0.147 |
| 20250426_150733286 | default | 54.8 | 14.3 | 2.8 | 14.9 | 13.2 | 14.5 | 50.4 | 0.60/1.96 | 0.217 |
| 20250426_150733286 | z-1_c0.0 | 74.0 | 0.6 | 0.0 | 13.3 | 12.0 | 13.0 | 59.4 | 0.71/2.43 | 0.280 |
| 20250426_150733286 | z-1_c0.005 | 71.2 | 3.1 | 0.0 | 13.5 | 12.2 | 13.1 | 59.1 | 0.70/2.36 | 0.271 |
| 20250426_150733286 | z-1_c0.01 | 66.9 | 6.8 | 0.0 | 13.9 | 12.5 | 13.5 | 58.2 | 0.69/2.27 | 0.257 |
| 20250426_150733286 | z-1_c0.02 | 57.7 | 14.2 | 0.0 | 15.3 | 12.8 | 14.9 | 54.3 | 0.64/2.09 | 0.231 |
| 20250426_150733286 | z-1_c0.04 | 42.3 | 25.0 | 0.0 | 20.9 | 11.7 | 20.6 | 42.7 | 0.49/1.72 | 0.191 |
| 20250426_150733286 | z-1_c0.08 | 23.6 | 30.8 | 0.0 | 38.0 | 7.6 | 37.7 | 22.8 | 0.21/1.18 | 0.140 |
| 20250426_150733286 | z0.7_c0.02 | 56.7 | 14.3 | 1.0 | 15.0 | 13.0 | 14.7 | 52.4 | 0.62/2.03 | 0.225 |
| 20250426_150733286 | z0.9_c0.02 | 46.9 | 14.2 | 9.8 | 15.9 | 13.2 | 15.6 | 43.3 | 0.52/1.72 | 0.190 |
| 20250426_150733286 | default_it20 | 54.3 | 14.3 | 2.7 | 15.4 | 13.3 | 15.1 | 27.4 | 0.60/2.03 | 0.215 |
| 20250426_150733286 | tol0.1 | 54.9 | 14.4 | 2.8 | 14.7 | 13.2 | 14.3 | 38.1 | 0.60/1.95 | 0.216 |
| 20250426_150733286 | tol0.2 | 55.1 | 14.5 | 2.8 | 14.3 | 13.3 | 13.9 | 23.5 | 0.58/1.93 | 0.214 |
| 20250426_150733286 | tol0.2_it20 | 54.8 | 14.6 | 2.8 | 14.5 | 13.4 | 14.1 | 10.9 | 0.57/1.96 | 0.212 |
| 20250907_000422554 | default | 53.4 | 12.8 | 4.5 | 9.5 | 19.8 | 9.2 | 51.6 | 0.67/2.04 | 0.268 |
| 20250907_000422554 | z-1_c0.0 | 72.1 | 0.9 | 0.0 | 8.3 | 18.7 | 8.0 | 67.0 | 0.87/2.75 | 0.333 |
| 20250907_000422554 | z-1_c0.005 | 68.8 | 3.9 | 0.0 | 8.3 | 18.9 | 8.0 | 65.2 | 0.85/2.60 | 0.323 |
| 20250907_000422554 | z-1_c0.01 | 65.4 | 7.0 | 0.0 | 8.6 | 19.1 | 8.3 | 62.9 | 0.82/2.47 | 0.313 |
| 20250907_000422554 | z-1_c0.02 | 58.4 | 12.6 | 0.0 | 9.7 | 19.3 | 9.4 | 57.7 | 0.75/2.26 | 0.292 |
| 20250907_000422554 | z-1_c0.04 | 45.7 | 21.7 | 0.0 | 14.3 | 18.3 | 14.0 | 46.6 | 0.59/1.89 | 0.252 |
| 20250907_000422554 | z-1_c0.08 | 26.9 | 28.9 | 0.0 | 31.0 | 13.2 | 30.7 | 27.2 | 0.29/1.38 | 0.183 |
| 20250907_000422554 | z0.7_c0.02 | 56.5 | 12.7 | 1.7 | 9.4 | 19.6 | 9.1 | 54.4 | 0.70/2.13 | 0.281 |
| 20250907_000422554 | z0.9_c0.02 | 40.3 | 12.7 | 15.4 | 11.8 | 19.8 | 11.5 | 41.8 | 0.55/1.79 | 0.225 |
| 20250907_000422554 | default_it20 | 52.7 | 12.8 | 4.5 | 10.2 | 19.9 | 9.9 | 29.6 | 0.67/2.17 | 0.265 |
| 20250907_000422554 | tol0.1 | 53.4 | 12.9 | 4.5 | 9.4 | 19.8 | 9.1 | 39.8 | 0.67/2.03 | 0.267 |
| 20250907_000422554 | tol0.2 | 53.5 | 13.1 | 4.5 | 9.0 | 19.8 | 8.7 | 24.4 | 0.64/1.99 | 0.265 |
| 20250907_000422554 | tol0.2_it20 | 53.1 | 13.1 | 4.5 | 9.4 | 19.9 | 9.1 | 11.3 | 0.64/2.06 | 0.261 |
| fleetws | default | 38.0 | 8.7 | 8.1 | 20.2 | 24.9 | 20.2 | 54.3 | 0.56/2.09 | 0.237 |
| fleetws | z-1_c0.0 | 56.3 | 0.6 | 0.0 | 19.6 | 23.5 | 19.6 | 80.7 | 0.94/3.07 | 0.330 |
| fleetws | z-1_c0.005 | 54.6 | 2.2 | 0.0 | 19.8 | 23.5 | 19.7 | 79.6 | 0.91/3.00 | 0.323 |
| fleetws | z-1_c0.01 | 51.9 | 4.2 | 0.0 | 20.4 | 23.4 | 20.4 | 77.7 | 0.87/2.88 | 0.314 |
| fleetws | z-1_c0.02 | 46.4 | 8.4 | 0.0 | 21.9 | 23.3 | 21.9 | 72.9 | 0.78/2.64 | 0.293 |
| fleetws | z-1_c0.04 | 36.5 | 15.1 | 0.0 | 26.6 | 21.8 | 26.6 | 61.2 | 0.59/2.28 | 0.253 |
| fleetws | z-1_c0.08 | 23.2 | 21.4 | 0.0 | 38.6 | 16.8 | 38.6 | 38.9 | 0.27/1.66 | 0.194 |
| fleetws | z0.7_c0.02 | 42.3 | 8.7 | 4.3 | 20.0 | 24.7 | 20.0 | 61.7 | 0.65/2.31 | 0.261 |
| fleetws | z0.9_c0.02 | 24.0 | 7.6 | 16.9 | 30.9 | 20.6 | 30.9 | 33.5 | 0.29/1.57 | 0.177 |
| fleetws | default_it20 | 37.0 | 8.7 | 8.1 | 21.4 | 24.8 | 21.4 | 29.8 | 0.54/2.24 | 0.230 |
| fleetws | tol0.1 | 38.1 | 8.8 | 8.2 | 20.0 | 24.9 | 20.0 | 43.6 | 0.55/2.08 | 0.236 |
| fleetws | tol0.2 | 38.2 | 9.0 | 8.3 | 19.4 | 25.1 | 19.4 | 28.7 | 0.52/2.04 | 0.232 |
| fleetws | tol0.2_it20 | 37.5 | 9.1 | 8.3 | 20.0 | 25.2 | 20.0 | 12.3 | 0.51/2.12 | 0.226 |
| KerryPark480 | default | 61.3 | 3.8 | 12.8 | 8.9 | 13.1 | 8.7 | 34.9 | 0.54/1.67 | 0.222 |
| KerryPark480 | z-1_c0.0 | 79.7 | 0.2 | 0.0 | 8.3 | 11.8 | 8.1 | 47.9 | 0.63/2.18 | 0.283 |
| KerryPark480 | z-1_c0.005 | 79.2 | 0.7 | 0.0 | 8.4 | 11.8 | 8.2 | 47.7 | 0.63/2.16 | 0.282 |
| KerryPark480 | z-1_c0.01 | 78.1 | 1.6 | 0.0 | 8.5 | 11.8 | 8.3 | 47.4 | 0.62/2.13 | 0.279 |
| KerryPark480 | z-1_c0.02 | 75.4 | 3.9 | 0.0 | 8.9 | 11.8 | 8.7 | 46.6 | 0.62/2.02 | 0.272 |
| KerryPark480 | z-1_c0.04 | 68.1 | 9.8 | 0.0 | 10.5 | 11.6 | 10.3 | 45.1 | 0.59/1.87 | 0.254 |
| KerryPark480 | z-1_c0.08 | 51.9 | 22.0 | 0.0 | 15.8 | 10.3 | 15.6 | 37.3 | 0.51/1.61 | 0.216 |
| KerryPark480 | z0.7_c0.02 | 69.5 | 3.9 | 5.5 | 8.4 | 12.7 | 8.2 | 39.1 | 0.57/1.78 | 0.247 |
| KerryPark480 | z0.9_c0.02 | 37.4 | 3.6 | 32.4 | 14.7 | 11.9 | 14.5 | 22.8 | 0.40/1.33 | 0.169 |
| KerryPark480 | default_it20 | 61.2 | 3.9 | 12.7 | 9.1 | 13.1 | 8.9 | 19.5 | 0.54/1.70 | 0.221 |
| KerryPark480 | tol0.1 | 61.4 | 3.9 | 12.8 | 8.7 | 13.2 | 8.5 | 24.5 | 0.54/1.67 | 0.221 |
| KerryPark480 | tol0.2 | 61.4 | 3.9 | 12.9 | 8.5 | 13.3 | 8.3 | 13.2 | 0.53/1.64 | 0.220 |
| KerryPark480 | tol0.2_it20 | 61.4 | 3.9 | 12.9 | 8.6 | 13.3 | 8.4 | 6.7 | 0.52/1.65 | 0.220 |
| OmniTemple1 | default | 58.8 | 7.1 | 8.8 | 8.2 | 17.1 | 8.1 | 42.2 | 0.52/1.81 | 0.193 |
| OmniTemple1 | z-1_c0.0 | 75.5 | 0.6 | 0.0 | 8.0 | 15.9 | 8.0 | 57.3 | 0.67/2.49 | 0.248 |
| OmniTemple1 | z-1_c0.005 | 73.9 | 2.1 | 0.0 | 8.1 | 15.9 | 8.1 | 56.5 | 0.66/2.42 | 0.244 |
| OmniTemple1 | z-1_c0.01 | 72.0 | 3.7 | 0.0 | 8.3 | 16.0 | 8.3 | 55.4 | 0.64/2.34 | 0.239 |
| OmniTemple1 | z-1_c0.02 | 68.2 | 6.9 | 0.0 | 8.8 | 16.1 | 8.7 | 52.9 | 0.61/2.20 | 0.230 |
| OmniTemple1 | z-1_c0.04 | 61.1 | 12.6 | 0.0 | 10.3 | 16.0 | 10.2 | 47.2 | 0.55/1.98 | 0.212 |
| OmniTemple1 | z-1_c0.08 | 48.2 | 21.5 | 0.0 | 16.2 | 14.1 | 16.1 | 35.9 | 0.42/1.61 | 0.180 |
| OmniTemple1 | z0.7_c0.02 | 64.2 | 7.0 | 3.9 | 8.1 | 16.8 | 8.0 | 46.1 | 0.56/1.96 | 0.210 |
| OmniTemple1 | z0.9_c0.02 | 41.7 | 6.9 | 23.4 | 11.5 | 16.5 | 11.4 | 31.1 | 0.39/1.48 | 0.155 |
| OmniTemple1 | default_it20 | 58.4 | 7.1 | 8.7 | 8.6 | 17.2 | 8.5 | 22.6 | 0.52/1.90 | 0.191 |
| OmniTemple1 | tol0.1 | 58.9 | 7.1 | 8.8 | 8.1 | 17.1 | 8.0 | 31.3 | 0.51/1.80 | 0.192 |
| OmniTemple1 | tol0.2 | 58.9 | 7.2 | 8.8 | 7.8 | 17.2 | 7.7 | 18.9 | 0.49/1.77 | 0.191 |
| OmniTemple1 | tol0.2_it20 | 58.7 | 7.3 | 8.8 | 8.0 | 17.2 | 7.9 | 9.0 | 0.49/1.81 | 0.189 |
| SeoulBull | default | 50.4 | 6.7 | 11.8 | 13.1 | 18.0 | 12.5 | 46.2 | 0.69/1.98 | 0.261 |
| SeoulBull | z-1_c0.0 | 71.8 | 0.2 | 0.0 | 11.6 | 16.4 | 11.0 | 61.3 | 0.86/2.50 | 0.342 |
| SeoulBull | z-1_c0.005 | 71.0 | 1.1 | 0.0 | 11.5 | 16.5 | 10.9 | 60.8 | 0.85/2.46 | 0.340 |
| SeoulBull | z-1_c0.01 | 69.0 | 2.6 | 0.0 | 11.8 | 16.5 | 11.2 | 60.3 | 0.83/2.42 | 0.335 |
| SeoulBull | z-1_c0.02 | 63.7 | 6.5 | 0.0 | 13.6 | 16.2 | 13.0 | 57.9 | 0.79/2.30 | 0.318 |
| SeoulBull | z-1_c0.04 | 52.8 | 14.7 | 0.0 | 16.7 | 15.8 | 16.1 | 51.6 | 0.69/2.04 | 0.284 |
| SeoulBull | z-1_c0.08 | 34.7 | 24.9 | 0.0 | 27.3 | 13.1 | 26.8 | 34.9 | 0.45/1.66 | 0.219 |
| SeoulBull | z0.7_c0.02 | 58.4 | 6.6 | 4.9 | 13.0 | 17.1 | 12.4 | 51.0 | 0.73/2.12 | 0.290 |
| SeoulBull | z0.9_c0.02 | 30.0 | 6.2 | 25.6 | 21.4 | 16.8 | 20.8 | 27.6 | 0.42/1.64 | 0.209 |
| SeoulBull | default_it20 | 50.1 | 6.6 | 11.7 | 13.6 | 18.1 | 13.0 | 26.0 | 0.69/2.05 | 0.261 |
| SeoulBull | tol0.1 | 50.5 | 6.6 | 11.9 | 12.8 | 18.0 | 12.2 | 34.1 | 0.68/1.96 | 0.261 |
| SeoulBull | tol0.2 | 50.5 | 6.7 | 12.0 | 12.7 | 18.1 | 12.0 | 20.9 | 0.67/1.94 | 0.259 |
| SeoulBull | tol0.2_it20 | 50.2 | 6.7 | 12.0 | 12.8 | 18.2 | 12.2 | 10.4 | 0.67/1.98 | 0.260 |

## Wall time

_Measured at `f787be3b` on the loop that applied every fitted update to the shape, which the acceptance rule has since replaced._

**Question.** Does turning the stage on raise the refinement's wall time? The draft's test is that it does not. An entry above 1.5× is flagged.

**Method.** Each run has three clocks:

- the *kernel* wall is the `refine_cluster_patches` wall the profile prints, which covers the refinement alone;
- the *process* wall covers the whole `pixi run … sfm cluster-patches` process, including image decoding, file I/O and interpreter start-up;
- the *CPU ratio* is read from the piecewise run alone. It divides the profile's thread-summed `cluster_total` by `cluster_total` minus the `piecewise` phase, which gives the refinement's CPU cost with the stage over its cost without it.

The CPU ratio comes from a single process, so contention from the other workload on the machine affects its numerator and denominator alike. The kernel and process ratios compare two processes run minutes apart, so the contention adds noise to them.

**Result.** The stage raises the wall time on every entry. Summed over the fleet:

| clock | cascade | piecewise | ratio |
|---|---|---|---|
| kernel wall | 1,494 s | 2,445 s | **1.64×** |
| process wall | 2,410 s | 3,289 s | 1.36× |
| CPU, pooled | | | 1.61× |

The piecewise phase is 37.8% of the refinement's CPU time. Per entry, the kernel ratio runs from 1.13× to 2.64×, and **33 of the 43 entries are above 1.5×**. The per-entry CPU ratio, which contention does not affect, runs from 1.35× to 1.81×. The process ratio of `20240915_073428267` (0.57×) is an artefact: its piecewise run was rerun later, when the machine was less loaded (see [Crashes](#crashes)).

The draft's test failed on that loop: with the stage on, the refinement cost about 1.6 times as much. The profile's own split shows the reason. Per kept member, the piecewise phase costs about as much as the cascade's `refine_member` phase. The Theory section expects each iteration to be a cheap render and nine small searches, much cheaper than the cascade's simplex, and on this fleet the stage costs about as much as the cascade does. If the loop converged in the two or three iterations the draft expects, the 24% of members that run five iterations would cost less, and that would change this verdict.

| entry | proc cascade s | proc piecewise s | ratio | kernel cascade s | kernel piecewise s | ratio | CPU ratio | flag |
|---|---|---|---|---|---|---|---|---|
| 20240614_203547691 | 71.8 | 98.0 | 1.36 | 47.3 | 74.6 | 1.58 | 1.60 | **>1.5x** |
| 20240614_224244438 | 78.9 | 133.7 | 1.69 | 54.1 | 96.4 | 1.78 | 1.58 | **>1.5x** |
| 20240614_224422531 | 98.0 | 137.4 | 1.40 | 64.2 | 110.8 | 1.73 | 1.62 | **>1.5x** |
| 20240614_225938434 | 77.6 | 105.5 | 1.36 | 51.6 | 79.3 | 1.54 | 1.62 | **>1.5x** |
| 20240617_204133531 | 40.6 | 58.6 | 1.44 | 24.8 | 41.4 | 1.67 | 1.50 | **>1.5x** |
| 20240618_001255975~2 | 36.3 | 50.8 | 1.40 | 22.9 | 37.3 | 1.63 | 1.58 | **>1.5x** |
| 20240702_224718414 | 48.9 | 66.3 | 1.36 | 31.0 | 48.8 | 1.57 | 1.58 | **>1.5x** |
| 20240713_205804006 | 95.9 | 149.6 | 1.56 | 62.0 | 115.8 | 1.87 | 1.65 | **>1.5x** |
| 20240906_081206935 | 55.0 | 77.1 | 1.40 | 37.4 | 58.3 | 1.56 | 1.53 | **>1.5x** |
| 20240915_071403318 | 82.6 | 126.1 | 1.53 | 52.3 | 96.2 | 1.84 | 1.58 | **>1.5x** |
| 20240915_073428267 | 112.2 | 63.7 | 0.57 | 29.3 | 46.5 | 1.59 | 1.77 | **>1.5x** |
| 20240918_074134864 | 39.9 | 57.7 | 1.45 | 22.6 | 40.4 | 1.79 | 1.79 | **>1.5x** |
| 20240919_040613535 | 68.9 | 100.0 | 1.45 | 43.6 | 71.8 | 1.65 | 1.56 | **>1.5x** |
| 20240919_062358681 | 52.6 | 77.7 | 1.48 | 32.3 | 57.3 | 1.77 | 1.68 | **>1.5x** |
| 20250425_135433677 | 89.8 | 146.7 | 1.63 | 64.4 | 116.8 | 1.81 | 1.53 | **>1.5x** |
| 20250425_141305472 | 60.6 | 80.6 | 1.33 | 40.9 | 60.7 | 1.48 | 1.49 |  |
| 20250426_150733286 | 31.9 | 40.4 | 1.27 | 18.4 | 26.6 | 1.45 | 1.64 |  |
| 20250706_223513674 | 64.6 | 70.5 | 1.09 | 47.5 | 53.5 | 1.13 | 1.42 |  |
| 20250712_195736354 | 118.7 | 120.1 | 1.01 | 64.9 | 92.0 | 1.42 | 1.63 |  |
| 20250712_202131684 | 88.6 | 126.6 | 1.43 | 60.8 | 97.1 | 1.60 | 1.64 | **>1.5x** |
| 20250712_204251146 | 56.7 | 83.7 | 1.48 | 36.5 | 62.5 | 1.71 | 1.59 | **>1.5x** |
| 20250906_211742965 | 42.4 | 62.2 | 1.47 | 27.8 | 47.3 | 1.70 | 1.69 | **>1.5x** |
| 20250907_000240907 | 27.3 | 52.5 | 1.92 | 15.5 | 40.9 | 2.64 | 1.67 | **>1.5x** |
| 20250907_000422554 | 12.7 | 22.8 | 1.80 | 6.6 | 12.8 | 1.94 | 1.70 | **>1.5x** |
| 20250907_000742129 | 34.7 | 40.4 | 1.16 | 14.5 | 28.0 | 1.92 | 1.77 | **>1.5x** |
| 20250907_001316663 | 62.4 | 82.4 | 1.32 | 45.6 | 61.5 | 1.35 | 1.62 |  |
| 20250907_211111559 | 59.6 | 89.4 | 1.50 | 41.4 | 68.2 | 1.65 | 1.55 | **>1.5x** |
| AltonaGalleryInTheParkBoyReading | 58.6 | 76.0 | 1.30 | 37.6 | 53.7 | 1.43 | 1.45 |  |
| BadlandPanorama | 121.4 | 151.3 | 1.25 | 58.2 | 110.1 | 1.89 | 1.80 | **>1.5x** |
| DaeguArtMuseumTreeStumpExhibit | 50.2 | 65.3 | 1.30 | 31.5 | 48.2 | 1.53 | 1.47 | **>1.5x** |
| DaeguMuseumMasks | 37.2 | 65.3 | 1.76 | 21.3 | 48.5 | 2.28 | 1.69 | **>1.5x** |
| DnDTabletop | 103.3 | 122.0 | 1.18 | 71.0 | 94.9 | 1.34 | 1.67 |  |
| KerryPark360 | 6.5 | 20.0 | 3.08 | 4.1 | 8.6 | 2.06 | 1.35 | **>1.5x** |
| KerryPark480 | 2.4 | 3.0 | 1.25 | 1.0 | 1.6 | 1.58 | 1.57 | **>1.5x** |
| MossyRailing | 61.9 | 83.9 | 1.36 | 43.5 | 64.0 | 1.47 | 1.52 |  |
| MurdoSmallAntiqueCat | 59.8 | 79.6 | 1.33 | 40.7 | 59.0 | 1.45 | 1.48 |  |
| OmniCoast | 23.8 | 40.0 | 1.68 | 14.3 | 31.4 | 2.20 | 1.65 | **>1.5x** |
| OmniHilltop | 35.5 | 50.9 | 1.43 | 25.5 | 38.8 | 1.52 | 1.73 | **>1.5x** |
| OmniTemple1 | 23.0 | 29.1 | 1.27 | 11.0 | 15.4 | 1.40 | 1.62 |  |
| SeoulBull | 1.2 | 1.9 | 1.58 | 0.2 | 0.4 | 1.78 | 1.43 | **>1.5x** |
| SfmExperimentation_vid2 | 41.8 | 76.1 | 1.82 | 25.6 | 51.5 | 2.01 | 1.58 | **>1.5x** |
| SpainSoapmaker_ws2 | 64.3 | 91.4 | 1.42 | 43.6 | 69.6 | 1.60 | 1.81 | **>1.5x** |
| fleetws | 9.5 | 12.8 | 1.35 | 4.4 | 6.7 | 1.53 | 1.50 | **>1.5x** |
| total | 2410 | 3289 | 1.36 | 1494 | 2445 | 1.64 | | |

### Crashes

Three of the 86 runs aborted with exit status `0xC0000409`:

- the piecewise run of `MossyRailing`;
- the piecewise run of `20240915_073428267`;
- the **cascade** run of `BadlandPanorama`.

Each succeeded when rerun after the fleet pass finished, and the tables use the reruns. The `MossyRailing` log was read before its rerun replaced it. It shows an allocation failure after the refinement had finished, in the warp-consistency pass:

```text
memory allocation of 13538208 bytes failed
stack backtrace:
   0: std::alloc::rust_oom
   ...
   6: rayon::iter::collect::collect_with_consumer<…cluster_refine::consistency::warp_consistency_residuals…>
   ...
   9: sfmtool_core::patch::cluster_refine::consistency::warp_consistency_residuals
```

The reruns replaced the other two logs before they were read. A two-million-member entry holds about 20 GB of private memory during the refinement, and the other workload on the machine was running at the same time. These were out-of-memory aborts under shared load. One of them was a cascade-only run, so the stage did not cause them.

## Seed stage on the ground-truth entries

_Measured at `f787be3b` on the loop that applied every fitted update to the shape, which the acceptance rule has since replaced._

**Question.** If the seed reads the piecewise file instead of the cascade file, does that change which candidates agree with the ground truth, or which candidate the seed picks?

**Method.** The seed ran as `scripts/exp_fast_seed.py` with the environment of the earlier fleet seed run:

- `SFMTOOL_RELAX=0`, `SFMTOOL_SEED_RUNG1=3000` and `SFMTOOL_SEED_EVO_DUMP` set;
- `SFMTOOL_SEED_EVAL`, `SFMTOOL_PINHOLE_RAW_VOTE`, and the radius, coarse-admission, stage-1, snapshot and stage-dump variables unset.

It ran twice on each of `SeoulBull` and `KerryPark480`, once reading each file. Each run used a temporary copy of the workspace holding the images, `.sfm-workspace.json`, `rig_config.json`, and a `matches/` directory with only the scratch file in it.

Every finite released candidate was scored against the checked-in ground truth (`seoul_bull_sculpture_ground_truth.sfmr` or `kerry_park_ground_truth.sfmr`), after a similarity alignment of the camera centres. The scores are:

- the median and maximum rotation error;
- the median and maximum centre error, as a percentage of the ground truth's camera extent;
- the focal error, both from the manifest focal and as the equivalent focal recomputed over the ground truth's observed radii.

The pass rule is: median rotation under 1°, median centre error under 5%, and equivalent focal within 5%. The pick is `ladder_first`. A `-spline-refused` file is the same hypothesis released without its spline rung, and the tables list it as the seed releases it.

Both `KerryPark480` runs were repeated, and every number of the repeats was identical. The `SeoulBull` cascade run reproduces the earlier run of 2026-10-07 on the same file exactly.

**Result.** The piecewise file changes candidates on both entries, and it breaks the pick on `KerryPark480`.

On `KerryPark480`, the pick `h00` passes with the cascade file (rotation 0.71°, centre 0.91%, focal −0.1%). With the piecewise file it fails, with a rotation error of 1.13° and a centre error of 4.11% median and 34% maximum. Five of the eight finite hypotheses pass with the cascade file, and three with the piecewise file. Individual hypotheses:

- `h02` and `h05` pass with both files;
- `h03` passes with the cascade file, and with the piecewise file it has one camera rotated 178°;
- `h06` fails with both files, and with the piecewise file it also has a camera rotated 179°.

On `SeoulBull` the pick `h00` passes with both files, within 0.03° and 0.5 points of the bars: its rotation errors are 0.995° and 0.979°, and its focal errors +4.6% and +4.5%. With the piecewise file it is no longer `qualified`, and it is flagged `edge_scan`. `h01` passes with the cascade file. With the piecewise file its spline release is refused, and the released `h01` misses the focal bar at exactly −5.00%, while its spline-refused variant passes (+0.5% equivalent focal). The pick does not change.

The loop at `f787be3b` turned the passing pick on `KerryPark480` into a failing one, and it changed candidates' pass or fail on both captures. That agrees with the parity result, which found the loop moving shapes away from the cascade's ZNCC optimum.

The decision this informed was that the stage stayed off by default (`--no-piecewise`). A version of the stage whose shapes agreed with the cascade within a few tenths of a grid px at the 95th percentile would reopen the question; on such a version the per-cell readings would be the only new output, and this comparison should show no change. The [measurement](#subset-with-the-shape-left-to-the-cascade-2026-10-08) is that version.

### SeoulBull

**(a) cascade file** (pick `ladder_first` = 0)

| file | qualified | accepted | posed | rot med/max deg | centre med/max % | f err % (manifest / equiv) | pass |
|---|---|---|---|---|---|---|---|
| h00.sfmr | 1 | 1 | 8 | 0.995/1.97 | 0.15/0.24 | +4.68 / +4.57 | PASS |
| h01.sfmr | 0 | 1 | 8 | 0.419/1.00 | 0.18/0.38 | +3.19 / +3.06 | PASS |
| h02-spline-refused.sfmr | 0 | 0 | 3 | 7.233/7.47 | 0.39/0.71 | -11.16 / -15.67 | fail |
| h02.sfmr | 0 | 0 | 3 | 7.990/8.10 | 0.29/0.55 | -11.16 / -11.16 | fail |

**(b) piecewise file** (pick `ladder_first` = 0)

| file | qualified | accepted | posed | rot med/max deg | centre med/max % | f err % (manifest / equiv) | pass |
|---|---|---|---|---|---|---|---|
| h00.sfmr | 0 | 1 | 8 | 0.979/1.94 | 0.14/0.25 | +4.56 / +4.47 | PASS |
| h01-spline-refused.sfmr | 0 | 0 | 8 | 0.321/0.80 | 0.14/0.32 | -5.00 / +0.47 | PASS |
| h01.sfmr | 0 | 0 | 8 | 0.219/0.37 | 0.19/0.36 | -5.00 / -5.00 | fail |
| h02-spline-refused.sfmr | 0 | 0 | 3 | 7.162/7.38 | 0.39/0.70 | -17.40 / -16.58 | fail |
| h02.sfmr | 0 | 0 | 3 | 7.595/7.62 | 0.31/0.59 | -17.40 / -17.40 | fail |

### KerryPark480

**(a) cascade file** (pick `ladder_first` = 0)

| file | qualified | accepted | posed | rot med/max deg | centre med/max % | f err % (manifest / equiv) | pass |
|---|---|---|---|---|---|---|---|
| h00.sfmr | 1 | 1 | 18 | 0.710/1.27 | 0.91/1.67 | -0.19 / -0.11 | PASS |
| h01.sfmr | 1 | 1 | 14 | 0.423/2.31 | 1.46/6.25 | +0.05 / -0.03 | PASS |
| h02.sfmr | 1 | 1 | 14 | 0.930/2.32 | 2.47/7.03 | +0.64 / +0.36 | PASS |
| h03-spline-refused.sfmr | 1 | 0 | 14 | 0.591/5.81 | 1.58/7.80 | +0.65 / +0.38 | PASS |
| h03.sfmr | 1 | 0 | 14 | 0.963/6.07 | 3.03/10.48 | +0.65 / +0.65 | PASS |
| h04.sfmr | 1 | 1 | 14 | 1.764/6.27 | 16.68/40.51 | +1.14 / +1.20 | fail |
| h05.sfmr | 1 | 1 | 14 | 0.704/3.12 | 1.85/15.86 | +0.84 / +0.44 | PASS |
| h06.sfmr | 1 | 1 | 14 | 0.644/3.99 | 8.96/46.65 | +0.71 / +0.68 | fail |
| h07.sfmr | 1 | 1 | 10 | 0.891/2.56 | 18.47/50.58 | +1.32 / +0.78 | fail |

**(b) piecewise file** (pick `ladder_first` = 0)

| file | qualified | accepted | posed | rot med/max deg | centre med/max % | f err % (manifest / equiv) | pass |
|---|---|---|---|---|---|---|---|
| h00.sfmr | 1 | 1 | 19 | 1.134/4.98 | 4.11/34.13 | +0.49 / +0.62 | fail |
| h01.sfmr | 1 | 1 | 14 | 0.574/1.58 | 1.44/2.78 | -0.15 / -0.10 | PASS |
| h02.sfmr | 1 | 1 | 14 | 0.820/2.76 | 4.14/29.80 | +0.19 / +0.23 | PASS |
| h03.sfmr | 1 | 1 | 14 | 1.512/177.78 | 10.89/39.87 | +1.23 / +1.23 | fail |
| h04-spline-refused.sfmr | 1 | 0 | 14 | 4.650/170.38 | 22.36/38.19 | +1.62 / +1.61 | fail |
| h04.sfmr | 1 | 0 | 14 | 4.020/172.23 | 22.50/38.07 | +1.62 / +1.62 | fail |
| h05.sfmr | 1 | 1 | 14 | 0.894/2.01 | 4.65/24.01 | +0.04 / +0.18 | PASS |
| h06.sfmr | 1 | 1 | 13 | 0.562/179.29 | 10.50/28.03 | +0.71 / +0.74 | fail |
| h07.sfmr | 1 | 1 | 9 | 0.803/3.45 | 16.23/45.07 | +1.34 / +0.98 | fail |

## What these measurements decide

_Measured at `f787be3b` on the loop that applied every fitted update to the shape, which the acceptance rule has since replaced._

- **Default.** The stage stayed off (`--no-piecewise`). On this fleet the loop moved kept members' shapes by up to about 2 grid px at the 95th percentile with no bias, lowered their whole-member ZNCC, cost 1.6 times the refinement's time, and broke the seed's pick on `KerryPark480`.
- **`min_cell_zncc`.** The provisional 0.8 refuses 6.8% of peaked cells. The distribution has one mode and no valley, so it gives no data-derived boundary. The value stays provisional until an outcome measure, normal or shape error against a ground truth, is swept over it.
- **`min_cell_curvature`.** The curvature is not stored, so its distribution was read through reruns. That proxy is smooth, with no knee, and varies widely between captures (2% to 14% of cells refused at 0.02). The value stays provisional on the same terms. Storing the curvature would let the next measurement read it directly.
- **Agreement with the cascade, and convergence.** The loop's agreement with the cascade was about 2 grid px at the 95th percentile. A quarter of the members stopped at the cap, because their update alternated at a size above the 0.05 grid px tolerance. Both checks the draft's Testing section names, agreement within tolerance and no rise in wall time, failed at commit `f787be3b`.

## Subset with the acceptance rule (2026-10-08)

_What the human review corrects: this section judges the loop by its agreement with the cascade, which counts a correction of the cascade's shape as an error. A blind human review preferred the loop's moved shapes to the cascade's 37 to 3, and the loop is now the default; see [Human review of moved shapes](#human-review-of-moved-shapes-2026-10-09)._

**Question.** The fleet run above found the loop moving the cascade's shapes off its ZNCC optimum, a quarter of members oscillating at the cap, a 1.6× cost, and the seed's `KerryPark480` pick failing. The loop was changed in three ways:

- An update is applied only when the whole-member windowed ZNCC the cascade maximised does not fall below the current shape's by more than `1e-4`, and never below the starting shape's.
- The update is fitted by IRLS with a Tukey biweight, and a cell given no weight is stored as `refused_outlier`.
- The loop also stops when the update's largest cell-centre movement does not shrink.

After these changes, do the shapes stay with the cascade? Does the whole-member ZNCC never fall? Does the cost come down, and does the seed's pick pass again?

**Method.**

- **Code.** Branch `bootstrap-core-migration` at `b1b98e12`, with the changes of the commit that adds this section (`.matches` version 9). The extension was rebuilt with `pixi run maturin develop --release`. Same machine as above, with no other workload running.
- **Data.** Five entries of the fleet, each from the same clusters file as above: `SeoulBull`, `KerryPark480`, `fleetws` (the entry with the highest `refused_bound` share), `MurdoSmallAntiqueCat` and `DnDTabletop`.
- **Runs.** The cascade and piecewise CLI runs exactly as under [Setup](#setup-of-the-fleet-run-f787be3b), one after the other, with `SFMTOOL_PROFILE=1`. The file does not store why the loop stopped or whether its last update was applied. A third run, through the binding with `piecewise=True`, read those as `member_cell_loop_stop` and `member_cell_update_accepted`. Its statuses and cells are identical to the CLI file's on every entry.
- **Seed.** As under [Seed stage on the ground-truth entries](#seed-stage-on-the-ground-truth-entries), with the same environment, scoring and pass rule, on `SeoulBull` and `KerryPark480` with the new cascade and piecewise files.

A first version of the rule applied the `1e-4` tolerance to each update separately. On the same subset, members that moved over several updates ended with a whole-member ZNCC up to 0.0004 below the cascade's; on `DnDTabletop`, 5,247 members fell by more than 1e-4. The rule was changed in two ways: every update is floored at the starting shape's ZNCC, and the re-read after the loop must be at least the cascade's stored value. The tables are from the second version.

**Result: the stage now leaves almost every shape where the cascade put it.**

- **Loop stops.** On each entry, 98.5% to 99.8% of kept members stop at their first iteration: 0.2% to 5.0% because their render left no cell to fit (not run), 94.3% to 98.4% with the first update rejected, and under 0.1% converged. Among members whose loop ran, the last fitted update is applied to between 0% (`SeoulBull`) and 0.6% (`DnDTabletop`). No member reaches the cap, and no loop stops for oscillation.
- **Movement.** 0.2% to 1.4% of kept members move at all. These members either took one or more accepted updates before a later one was rejected, or converged. Their moves are not small: the median is 0.28 to 1.68 grid px, and the largest is 5.3.
- **ZNCC.** The whole-member ZNCC never falls: its change is ≥ 0 on every kept member of every entry. Where a member moves, its ZNCC rises, by a median of 0.0004 to 0.021 and at most 0.13. Since the cascade's simplex stopped at a lower ZNCC, this suggests it stopped at a local optimum for those members.

| entry | kept | moved % | dtotal grid p50/p95/max (moved) | stop at cap % | final update rejected % | dzncc min (all kept) | dzncc p5/p50/p95/max (moved) | kernel wall cascade/piecewise s | ratio | CPU ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| SeoulBull | 2953 | 0.17 | 1.68/3.45/3.77 | 0.0 | 100.0 | 0.0000 | 0.0005/0.0208/0.0404/0.0408 | 0.2/0.2 | 1.12 | 1.12 |
| KerryPark480 | 15986 | 0.34 | 0.28/2.05/3.32 | 0.0 | 99.9 | 0.0000 | 0.0000/0.0004/0.0253/0.0682 | 0.6/0.7 | 1.13 | 1.15 |
| fleetws | 68722 | 0.64 | 0.62/2.19/4.37 | 0.0 | 99.8 | 0.0000 | 0.0000/0.0008/0.0245/0.0733 | 4.2/5.1 | 1.21 | 1.14 |
| MurdoSmallAntiqueCat | 625198 | 1.44 | 1.00/2.29/5.32 | 0.0 | 99.5 | 0.0000 | 0.0001/0.0049/0.0404/0.1025 | 38.1/44.9 | 1.18 | 1.17 |
| DnDTabletop | 1375116 | 1.42 | 0.44/1.96/4.51 | 0.0 | 99.4 | 0.0000 | 0.0000/0.0007/0.0267/0.1289 | 53.3/68.8 | 1.29 | 1.25 |

Over all kept members, the median and the 95th percentile of the movement are 0.00 grid px on every entry. At `f787be3b`, the same entries moved 79.8% to 95.8% of kept members, by a median of 0.31 to 0.69 grid px, and 20% to 54% of members reached the cap.

The next table gives why each member's loop stopped, as a share of kept members, and the share at each iteration count:

| entry | not run | converged | cap | rejected | oscillation | iterations 1/2/3/4/5 % |
|---|---|---|---|---|---|---|
| SeoulBull | 3.7 | 0.0 | 0.0 | 96.3 | 0.0 | 99.8/0.1/0.1/0.0/0.0 |
| KerryPark480 | 1.8 | 0.1 | 0.0 | 98.1 | 0.0 | 99.7/0.3/0.1/0.0/0.0 |
| fleetws | 5.0 | 0.2 | 0.0 | 94.7 | 0.0 | 99.4/0.4/0.1/0.0/0.0 |
| MurdoSmallAntiqueCat | 0.9 | 0.4 | 0.0 | 98.6 | 0.0 | 98.5/0.8/0.5/0.2/0.1 |
| DnDTabletop | 0.2 | 0.6 | 0.0 | 99.2 | 0.0 | 98.6/0.9/0.4/0.1/0.0 |

A member counted as "not run" is one whose first render left no cell to fit, so all nine of its cells are `not_attempted`.

The next table gives the cell statuses, as a share of the cells of kept members, and the residual `|shift|` of fitted cells:

| entry | fitted | refused_curvature | refused_zncc | not_attempted | refused_bound | refused_outlier | fitted res p50/p95 grid px |
|---|---|---|---|---|---|---|---|
| SeoulBull | 52.9 | 7.8 | 14.4 | 4.6 | 19.8 | 0.6 | 0.36/1.31 |
| KerryPark480 | 63.9 | 4.4 | 14.7 | 2.0 | 14.4 | 0.6 | 0.28/1.13 |
| fleetws | 44.1 | 11.3 | 10.2 | 5.1 | 28.7 | 0.6 | 0.33/1.35 |
| MurdoSmallAntiqueCat | 75.5 | 3.9 | 9.7 | 1.4 | 8.8 | 0.7 | 0.20/0.98 |
| DnDTabletop | 85.8 | 5.7 | 3.1 | 0.7 | 4.2 | 0.5 | 0.17/0.80 |

The robust fit refuses 0.5% to 0.7% of cells as outliers. Because nearly every update is rejected, the cells are read at the cascade's shape, and their stored shifts are the residual to that shape. They still contain the first-order part that the rejected update would have removed. This is why the fitted residuals are larger than at `f787be3b`, where their median was 0.13 to 0.26. A consumer that wants only the second-order term must fit an affine to the nine shifts and remove it.

**Cost.** The kernel wall ratio is 1.12× to 1.29×. The CPU ratio, which contention does not affect, is 1.12× to 1.25×. At `f787be3b` these entries ran at 1.34× to 1.78× wall and 1.43× to 1.67× CPU. Almost every member now costs one render, nine cell searches and two readings of the whole-member ZNCC.

**Seed.** `SeoulBull` does not regress. With the piecewise file every hypothesis scores as it does with the cascade file, to the third decimal; the only difference is `h01`'s median rotation, 0.420° against 0.419°. The pick `h00` passes as before (0.995°, 0.15%, +4.57%).

`KerryPark480` still fails. With the cascade file, `h00` passes (0.710°, 0.91%, −0.11%), as it did on 2026-10-07. With the piecewise file, `h00` poses 26 cameras and fails, with a median rotation of 1.071° and a median centre error of 10.4%. Five hypotheses pass with the cascade file and two with the piecewise file.

| file (piecewise) | qualified | accepted | posed | rot med/max deg | centre med/max % | f err % (manifest / equiv) | pass |
|---|---|---|---|---|---|---|---|
| h00.sfmr | 1 | 1 | 26 | 1.071/4.83 | 10.37/39.32 | +0.52 / +0.60 | fail |
| h01.sfmr | 1 | 1 | 14 | 0.781/2.31 | 0.76/4.20 | +0.07 / +0.09 | PASS |
| h02.sfmr | 1 | 1 | 14 | 0.755/2.58 | 3.17/11.55 | +0.12 / +0.23 | PASS |
| h03.sfmr | 1 | 1 | 14 | 2.115/177.28 | 13.66/47.06 | +1.02 / +0.95 | fail |
| h04.sfmr | 1 | 0 | 13 | 2.450/178.93 | 18.01/50.04 | +1.69 / +1.69 | fail |
| h05.sfmr | 1 | 0 | 14 | 0.700/2.06 | 10.80/27.58 | +1.68 / +1.68 | fail |
| h06.sfmr | 1 | 0 | 14 | 14.605/116.86 | 21.28/34.60 | +1.54 / +1.54 | fail |
| h07.sfmr | 1 | 1 | 11 | 1.356/3.09 | 14.83/40.74 | +1.47 / +1.12 | fail |

The `-spline-refused` variants of `h04`, `h05` and `h06` also fail.

The two files differ in three ways:

- the shapes, positions, ZNCCs and shifts of 55 of 15,986 kept members (0.34%);
- the per-cell columns;
- the consistency residuals, which the joint factorization spreads over 24,765 members.

To separate these effects, three variant files were built and the seed was run on each:

- **File A:** the piecewise file, with the 55 members' four columns and every consistency residual set back to the cascade's, so that only the per-cell columns differ from the cascade file.
- **File B:** the cascade file, with the piecewise file's consistency residuals.
- **File C:** the cascade file, with the 55 members' shapes, positions, ZNCCs and shifts taken from the piecewise file.

Files A and B reproduce every number of the cascade run, so neither the cells nor the consistency residuals affect the seed here. File C reproduces every number of the piecewise run. The 55 moved members alone turn the passing pick into a failing one, and each of them moved to a shape with a higher whole-member ZNCC.

**What this decides.**

- **The acceptance rule does what it was meant to do on the shapes.** The whole-member ZNCC never falls, the loop no longer oscillates or reaches the cap, and the cost falls to about 1.2×. The stage is now close to a no-op on shapes: it reads the cells at the cascade's shape and stores them with their residuals, and it moves under 1.5% of members. The stage's product is the residuals, which are what the normal estimator needs.
- **The seed criterion still fails on `KerryPark480`.** The stage does not lower any score. The pick fails because it is sensitive to 0.34% of members moving to shapes that the cascade's own objective prefers. ZNCC alone cannot say whether those 55 shapes are actually worse. Two measurements could settle it: score the moved members' shapes against the `kerry_park` ground truth, or find which of them carry the seed's choice. Until then the stage stays off by default.
- **A variant that never moves the shape would leave the seed exactly as the cascade does.** Such a variant only stores the cells read at the cascade's shape; file A is its output. It is the conservative choice if the stage is wanted only for its residuals.
- **The loop's stop reason and acceptance are not stored.** Why the loop stopped and whether its last update was applied are available through the binding but not in the file. Storing them is a follow-up, if a consumer needs them.

## Subset with the shape left to the cascade (2026-10-08)

_What the human review corrects: this section made the measurement the default because the loop's moved shapes changed the seed's pick, and could not say whether they were worse. A blind human review found them better, and `move_shape` is now on by default; with it off the stage is the measurement this section describes. See [Human review of moved shapes](#human-review-of-moved-shapes-2026-10-09)._

_Measured with the robust fit's residual scale at 1.4826 times the median residual; the [next section](#subset-with-the-two-dimensional-residual-scale-2026-10-08) repeats the cell statuses with the two-dimensional factor._

**Question.** The section above found that moving 55 kept members on `KerryPark480` is enough to fail the seed's pick, and that the cells alone (file A) leave the seed exactly as the cascade does. The stage was changed to match: by default (`PiecewiseParams::move_shape` false) it renders each kept member once at the cascade's shape, registers the nine cells, fits the robust affine map only to find the outlier cells, and stores the shifts as measured. The member's shape, position, ZNCC and shift are not touched, and the whole-member ZNCC is not read. The loop is still available with `move_shape`. Does the default leave every member output bit-identical to the cascade-only run, what does it cost, and does the seed give the cascade-only result on both ground-truth entries?

**Method.**

- **Code.** Branch `bootstrap-core-migration` at `c494bf5b`, with the changes of the commit that adds this section, and the extension rebuilt with `pixi run maturin develop --release`. Same machine, with no other workload running.
- **Data and runs.** The five entries and the two CLI runs of the section above (`--no-piecewise` and `--piecewise`, `--patch-size 12`, `SFMTOOL_PROFILE=1`), the piecewise run now at the measuring default. A third run through the binding with `piecewise=True` read `member_cell_loop_stop` and `member_cell_update_accepted`; its statuses and cells are identical to the CLI file's on every entry.
- **Comparison.** `reference_members`, `member_status`, `member_positions`, `member_affine_shapes`, `member_zncc`, `member_shift_px` and `member_consistency_residual` compared byte for byte between the two files.
- **Seed.** As under [Seed stage on the ground-truth entries](#seed-stage-on-the-ground-truth-entries), with the same environment, scoring and pass rule, on `SeoulBull` and `KerryPark480`, each with its new cascade and piecewise files.

**Result: every member output is bit-identical to the cascade's on all five entries.** Each of the seven columns above is byte-for-byte equal between the two files. Every kept member reads one iteration, no update is applied anywhere, and the loop stop is `measured` for every kept member except those whose render left no cell to fit (`not run`, 0.1% to 5.0%).

The cells are the readings the loop of the section above stored for members whose first update it rejected: on the 2,838 to 1,353,196 such members of each entry, the statuses, shifts and ZNCCs are identical to that run's.

| entry | kept | member outputs bit-identical | kernel wall cascade/piecewise s | wall ratio | CPU ratio |
|---|---|---|---|---|---|
| SeoulBull | 2953 | yes | 0.16/0.17 | 1.09 | 1.09 |
| KerryPark480 | 15986 | yes | 0.70/0.99 | 1.42 | 1.13 |
| fleetws | 68722 | yes | 5.4/5.8 | 1.06 | 1.11 |
| MurdoSmallAntiqueCat | 625198 | yes | 44.8/45.2 | 1.01 | 1.12 |
| DnDTabletop | 1375116 | yes | 56.4/64.7 | 1.15 | 1.17 |

The CPU ratio, which contention does not affect, is 1.09× to 1.17×, against 1.12× to 1.25× for the loop with the acceptance rule; each kept member now costs one render, nine cell searches and the affine fit, and no reading of the whole-member ZNCC. The wall ratio is noisier: `KerryPark480`'s kernel runs for under a second, and its 1.42× is a difference of 0.3 s.

The next table gives the cell statuses, as a share of the cells of kept members, the length of the stored displacement of fitted cells, and the loop stop of kept members:

| entry | fitted | refused_curvature | refused_zncc | not_attempted | refused_bound | refused_outlier | fitted displacement p50/p95 grid px | measured / not run % |
|---|---|---|---|---|---|---|---|---|
| SeoulBull | 52.9 | 7.8 | 14.4 | 4.6 | 19.8 | 0.6 | 0.36/1.31 | 96.3 / 3.7 |
| KerryPark480 | 63.9 | 4.4 | 14.7 | 2.0 | 14.5 | 0.6 | 0.28/1.13 | 98.3 / 1.7 |
| fleetws | 44.1 | 11.3 | 10.2 | 5.0 | 28.8 | 0.6 | 0.33/1.36 | 95.0 / 5.0 |
| MurdoSmallAntiqueCat | 75.5 | 3.9 | 9.7 | 1.4 | 8.9 | 0.7 | 0.20/1.00 | 99.1 / 0.9 |
| DnDTabletop | 85.8 | 5.7 | 3.1 | 0.7 | 4.2 | 0.5 | 0.17/0.82 | 99.9 / 0.1 |

The shares are within 0.1 point of the section above, as they should be, since that run measured nearly every member at the cascade's shape too. The stored displacements are the measured shifts with no affine map removed, so their median of 0.17 to 0.36 grid px includes the first-order disagreement between the cells and the whole-patch fit.

**Seed.** On both entries the seed gives the cascade-only result exactly. Every released `.sfmr` (8 on `SeoulBull`, 18 on `KerryPark480`) is byte-for-byte identical between the two runs in every entry but the write timestamp, the content hash over it and the workspace's absolute path, and the manifests' hypotheses differ only in their timings. The scores therefore repeat the cascade-only tables of the [seed section](#seed-stage-on-the-ground-truth-entries) to the last digit: `SeoulBull`'s pick `h00` passes (0.995°, 0.15%, +4.57%), and `KerryPark480`'s pick `h00` passes (0.710°, 0.91%, −0.11%), with the same five of its finite hypotheses passing.

**What this decides.**

- **The default stage is a measurement.** With `move_shape` off, the cells are added to the file and nothing else in it changes, so a consumer that does not read the cells sees the cascade's file exactly. Its cost is about 1.1× the refinement's CPU time.
- **The seed is unaffected by the stage.** The piecewise file gives the seed the cascade file's result bit for bit, as file A of the section above predicted, so the stage can be turned on for its cells without changing any seed outcome.
- **The loop stays behind `move_shape`.** Whether the shapes it moves are better or worse than the cascade's is still open, and the question is now separate from whether the cells are stored.
- **The stage itself stays off by default** (`--no-piecewise`). Its cost is now small and it no longer changes the seed, so the remaining reason is that no consumer reads the cells yet.

## Subset with the two-dimensional residual scale (2026-10-08)

**Question.** The robust fit's residual scale was 1.4826 times the median residual, the factor that turns the median absolute value of one-dimensional Gaussian residuals into their standard deviation. A cell's residual to the fitted affine map is a two-dimensional length. Under isotropic Gaussian noise of per-axis standard deviation σ that length is Rayleigh-distributed with median σ·√(2 ln 2), so the matching factor is 1/√(2 ln 2) ≈ 0.849. With 1.4826 the scale was 1.75 times σ, and the Tukey cut-off of 4.685 scales sat at about 8.2σ instead of 4.685σ. The scale was changed to 0.849 times the median residual length, with the floor of 0.1 grid px kept. How many more cells does the fit refuse as outliers, and does the measurement still leave every member output bit-identical to the cascade-only run?

**Method.**

- **Code.** Branch `bootstrap-core-migration` at `ec210edc`, with the change of the commit that adds this section, and the extension rebuilt with `pixi run maturin develop --release`.
- **Runs.** The piecewise CLI run of the [section above](#subset-with-the-shape-left-to-the-cascade-2026-10-08) (`--piecewise`, `--patch-size 12`, the measuring default) repeated on the same five entries. The cascade-only files of that section are the baseline, since the change does not touch the cascade. "Before" is that section's piecewise file.

**Result.**

| entry | kept | refused_outlier before % | after % | fitted before % | after % | kept members with an outlier cell, before / after % | member outputs bit-identical to cascade-only |
|---|---|---|---|---|---|---|---|
| SeoulBull | 2953 | 0.57 | 3.01 | 52.92 | 50.48 | 4.7 / 23.2 | yes |
| KerryPark480 | 15986 | 0.59 | 3.30 | 63.89 | 61.18 | 5.0 / 25.7 | yes |
| fleetws | 68722 | 0.64 | 2.63 | 44.08 | 42.09 | 5.4 / 20.5 | yes |
| MurdoSmallAntiqueCat | 625198 | 0.73 | 3.38 | 75.50 | 72.85 | 6.1 / 25.7 | yes |
| DnDTabletop | 1375116 | 0.53 | 2.65 | 85.79 | 83.67 | 4.5 / 20.3 | yes |

Shares are of the cells of kept members. The robust fit now refuses 2.6% to 3.4% of cells as outliers, against 0.5% to 0.7% before, and one kept member in four to five has at least one outlier cell. Every changed cell moved between `fitted` and `refused_outlier`: 95% to 99% of the changes are `fitted` to `refused_outlier`, and the rest go the other way, because the refit without the newly refused cells moves the map. No other status changes, and the stored shifts, cell ZNCCs and iteration counts are identical to the run before. The fitted cells' displacement median falls by 0.01 to 0.02 grid px, to 0.16 to 0.35 grid px, and its 95th percentile by 0.03 to 0.06, to 0.76 to 1.32.

The seven member columns compared in the section above (`reference_members`, `member_status`, `member_positions`, `member_affine_shapes`, `member_zncc`, `member_shift_px`, `member_consistency_residual`) are byte-for-byte equal to the cascade-only file's on all five entries. The seed was not rerun: it does not read the cells, as file A of the [acceptance-rule section](#subset-with-the-acceptance-rule-2026-10-08) showed.

Under Gaussian noise alone a cut-off at 4.685σ would refuse a cell with probability exp(−4.685²/2), under 0.002%. The 2.6% to 3.4% refused here say that the cells' residuals have a heavier tail than Gaussian noise; this measurement does not say whether the newly refused cells are wrong.

## Cell plane normals against the ground truths (2026-10-08)

**Question.** With poses, the stored cell displacements give a patch normal per cluster through [cell-plane-normals.md](cell-plane-normals.md). How close is that normal to the ground truth's patch normal, compared with the mean viewing direction and with the same triangulation run on the members' affine shapes alone? How often does it fix both axes, one, or none, and does admitting `refused_outlier` cells change it?

**Data.**

- **Code.** Branch `bootstrap-core-migration` at `35442934`, with the commit that adds this section (the kernel and its binding), the extension rebuilt with `pixi run maturin develop --release`.
- **Machine.** Windows 11, Intel Core (family 6 model 183), 32 threads.
- **Cluster files.** `sfm cluster-patches --patch-size 12 --piecewise` (the measuring default, `move_shape` off, `resolution` 25) on the `SeoulBull` and `KerryPark480` cluster files of the [two-dimensional residual scale section](#subset-with-the-two-dimensional-residual-scale-2026-10-08), written to scratch.
- **Poses and truth.** Cameras and poses from `seoul_bull_sculpture_ground_truth.sfmr` and `kerry_park_ground_truth.sfmr`, read with `SfmrReconstruction.load`; every image of both cluster files is posed. The truth normal is `u × v` of the ground truth's stored patch frame.
- **Matching.** A cluster is mapped to a ground-truth point when its reference member's position lies within a tolerance of one of that point's stored keypoints in the same image (the nearest). The planned tolerance is 1 px; a second run at 3 px is reported because 1 px matches too few clusters to read.
- **Variants.** *cells*: the kernel at its defaults. *cells+outliers*: `include_refused_outlier`. *shapes only*: every kept member's nine cells taken as `fitted` with zero displacement, so the cells sit where the affine shapes place them; this is the whole-member warp tilt read through the same triangulation. *view_dir*: the mean unit direction from the cells toward their rays' cameras, the kernel's own prior.
- **Seed writer's tilt solve.** Not run. It is inline in `scripts/seed_camera.py`'s writer, reads the writer's local state (a `.sift` read per image, its poses flipped to the COLMAP frame) and is not callable on its own; *shapes only* stands in for it, without its fronto prior and its 80° cap.
- **Errors.** Both-axes clusters: the angle between the kernel's normal and the truth, as lines. One-axis clusters: the error of the component the cells fix, the difference of the two normals' tilts along the free axis.

**Matching.** The ground truths' keypoints are not this cluster file's detections: the nearest member of any status to a ground-truth observation is a median 4.4 px away on `SeoulBull` and 4.8 px on `KerryPark480`, and only 4.9% and 6.6% of the observations have a member within 1 px.

| entry | clusters with a reference | matched at 1 px | GT points hit | matched at 3 px | GT points hit |
|---|---|---|---|---|---|
| SeoulBull | 3934 | 27 (0.69%) | 23 / 280 | 164 (4.2%) | 116 / 280 |
| KerryPark480 | 11865 | 80 (0.67%) | 43 / 391 | 323 (2.7%) | 161 / 391 |

**Verdicts.** Over the clusters with a reference, at the defaults:

| entry | both axes % | one axis % | none % | cells in plane % | too few rays % | narrow baseline % | in-plane cells from two rays % |
|---|---|---|---|---|---|---|---|
| SeoulBull | 14.9 | 4.5 | 80.6 | 21.6 | 72.2 | 4.8 | 70.1 |
| KerryPark480 | 11.2 | 6.1 | 82.7 | 15.2 | 50.8 | 30.8 | 58.3 |

Of the clusters with a reference (3934 on `SeoulBull`, 11865 on `KerryPark480`), 80.6% and 82.7% get no normal, for three reasons. 48.5% and 25.3% of the clusters with a reference have no kept member, so each cell has only the reference's ray (`SeoulBull` keeps 2953 members over its 3934 references). 16.6% and 49.4% have a kept member but fewer than three triangulated cells; of these, 8.9 and 40.5 points (of the clusters with a reference) have at least one cell refused for a narrow baseline, which on `KerryPark480` is the distant city, and the rest have too few rays. The remaining 15.5% and 8.0% have three or more triangulated cells but fewer than three live ones: this run counted a cell toward the verdict only when its weight was at least a quarter of the cluster's largest, and a cell triangulated from two rays at the 0.05 px floor carries a small fraction of the weight of a cell with more rays, so it was left out. That share comes from the rule, not from the data, and the [re-measurement](#cell-plane-normals-after-the-audit-2026-10-08) removes it. Admitting `refused_outlier` cells raises the both-axes share by half a point (to 13.8% and 10.1% of all 4407 and 13699 clusters, against 13.3% and 9.7%). *Shapes only* raises it to 21.7% and 14.5% of all clusters, but this variant adds support as well as changing the cell positions: it marks all nine cells of every kept member `fitted`, including the cells the refinement refused or did not measure, so its higher share and its errors below are not a comparison of the displacements alone. The [re-measurement](#cell-plane-normals-after-the-audit-2026-10-08) compares with the same cells. The ray-intersection residual of in-plane cells is a median 0.17 px (`SeoulBull`) and 0.13 px (`KerryPark480`), p90 0.42 and 4.2 px, which says the grid-to-pixel map and the poses agree. The kernel takes 3 ms on `SeoulBull`'s 4407 clusters and 10 ms on `KerryPark480`'s 13699.

**Both-axes clusters**, median / p90 error in degrees:

| entry @ tolerance | n | cells | view_dir |
|---|---|---|---|
| SeoulBull @ 1 px | 7 | 17.6 / 24.6 | 29.9 / 36.1 |
| SeoulBull @ 3 px | 45 | 15.5 / 56.6 | 29.8 / 49.6 |
| KerryPark480 @ 1 px | 20 | 41.2 / 72.0 | 23.8 / 54.7 |
| KerryPark480 @ 3 px | 69 | 17.6 / 61.1 | 32.8 / 62.9 |

**Paired**, on the clusters every variant calls both-axes:

| entry @ tolerance | n | cells | cells+outliers | shapes only | view_dir |
|---|---|---|---|---|---|
| SeoulBull @ 1 px | 5 | 17.6 / 24.1 | 17.6 / 24.1 | 21.3 / 27.8 | 29.9 / 35.5 |
| SeoulBull @ 3 px | 33 | 16.6 / 56.7 | 16.6 / 56.7 | 17.6 / 58.4 | 29.3 / 43.4 |
| KerryPark480 @ 1 px | 14 | 39.0 / 69.3 | 39.0 / 70.1 | 25.7 / 66.6 | 28.9 / 58.4 |
| KerryPark480 @ 3 px | 55 | 15.8 / 56.6 | 17.8 / 59.8 | 14.3 / 49.0 | 33.5 / 63.5 |

**One-axis clusters**, error of the fixed component, median / p90:

| entry @ tolerance | n | cells | view_dir |
|---|---|---|---|
| SeoulBull @ 1 px | 1 | 7.4 | 23.7 |
| SeoulBull @ 3 px | 10 | 14.1 / 64.2 | 22.5 / 57.2 |
| KerryPark480 @ 1 px | 8 | 46.1 / 72.3 | 15.6 / 37.3 |
| KerryPark480 @ 3 px | 25 | 13.4 / 65.2 | 28.6 / 64.1 |

**Where the error is.** On the 3 px sets, split by the precision the cell weights predict for the normal, `1 / √(Σ wⱼ (dⱼ · e)²)` along the plane's weaker in-plane axis `e` (computed in the script from the kernel's outputs), and by how far the truth is from the viewing direction; medians in degrees:

| entry | predicted < 5° | 5° to 15° | over 15° | truth within 20° of view_dir | 20° to 40° | 40° to 60° | over 60° |
|---|---|---|---|---|---|---|---|
| SeoulBull, cells | 7.7 (n=28) | 30.0 (n=10) | 27.9 (n=7) | 25.1 (n=14) | 14.4 (n=23) | 8.4 (n=8) | none |
| SeoulBull, view_dir | 33.1 | 20.3 | 25.8 | 11.6 | 31.8 | 50.5 | none |
| KerryPark480, cells | 11.1 (n=36) | 20.8 (n=27) | 29.7 (n=6) | 47.8 (n=17) | 17.6 (n=31) | 10.5 (n=12) | 7.0 (n=9) |
| KerryPark480, view_dir | 35.4 | 29.3 | 29.2 | 13.2 | 31.7 | 51.6 | 67.7 |

66% of `SeoulBull`'s and 54% of `KerryPark480`'s both-axes clusters predict a precision under 5°.

**Sensitivity**, 3 px, both-axes clusters, median / p90 against view_dir on the same clusters:

| entry | `min_triangulation_angle_deg` 5: share of all clusters, n, cells, view_dir | `min_rays` 3: share, n, cells, view_dir |
|---|---|---|
| SeoulBull | 11.7%, 34, 10.7 / 31.2, 30.9 / 50.5 | 5.3%, 22, 12.5 / 58.3, 31.4 / 46.9 |
| KerryPark480 | 5.8%, 39, 10.8 / 48.8, 35.5 / 62.9 | 3.8%, 29, 25.0 / 72.2, 32.9 / 59.1 |

**Result.**

- On the 3 px sets the cell plane normal halves the median error of the mean viewing direction (15.5° against 29.8° on `SeoulBull`, 17.6° against 32.8° on `KerryPark480`), and on its one-axis clusters the fixed component improves the same way. Its p90 is not better: 56.6° and 61.1°, against 49.6° and 62.9°.
- The error follows the predicted precision: under 5° predicted, the median is 7.7° and 11.1°; above 15°, 27.9° and 29.7°. A minimum triangulation angle of 5° gives a similar subset (medians 10.7° and 10.8°) at the cost of the both-axes share (11.7% and 5.8% of all clusters).
- Where the truth faces the cameras (within 20° of the viewing direction) the cell normal is worse than the viewing direction itself (25.1° against 11.6°, 47.8° against 13.2°); where the truth is tilted 40° or more it is far better (7.0° to 10.5° against 50.5° to 67.7°). The measurement cannot separate a noisy cell normal from a truth normal held near the viewing direction by the estimator that made it.
- The displacements add little over the affine shapes here. On the paired sets *cells* and *shapes only* are within 1.5° in median (16.6° against 17.6°, 15.8° against 14.3°), and on `KerryPark480` the shapes are better at p90. *Shapes only* here also used cells that *cells* did not, so the two differ in support as well as in the displacements. The draft's test that the cell normals beat the whole-member warp tilt on the `seoul_bull_sculpture` ground truth is not met by a margin this sample can show.
- Admitting `refused_outlier` cells changes no median on `SeoulBull` and worsens `KerryPark480`'s paired median by 2.0°.
- At 1 px the samples are 5 to 20 clusters. `KerryPark480`'s 1 px set is the one where the cell normal loses to the viewing direction (41.2° against 23.8°); 8 of its 20 clusters have a truth within 20° of the viewing direction.

**What this decides.**

- **The kernel stays as built, with `refused_outlier` cells excluded by default.** Including them does not help.
- **No consumer reads it yet.** A cell normal is better than the viewing direction where it predicts itself precise, and worse where the surface faces the cameras, and the seed has no way to know the second case in advance. The next step is to return the predicted precision from the kernel and measure a gate on it; a minimum angle of 5° is the alternative gate.
- **The ground truth cannot decide between the cell normal and the warp tilt.** Its keypoints are not the cluster files' detections, so 1 px matches under 1% of the clusters with a reference, too few to separate two estimators a few degrees apart; and its normals are estimates, so where a cell normal disagrees with a truth near the viewing direction it cannot say which of the two is wrong. A ground truth with independently measured normals, or one built from the same detections, is needed before the cell normal is compared with the warp tilt solve again.

## Cell plane normals after the audit (2026-10-08)

**Question.** An audit of the [section above](#cell-plane-normals-against-the-ground-truths-2026-10-08) changed four things in the kernel ([cell-plane-normals.md](cell-plane-normals.md)): a cell is live when its final Tukey weight is above zero, not when its weight is at least a quarter of the cluster's largest; a cell's residual is floored at the precision of a measured cell shift, 0.1 grid px mapped into each ray's image (`0.1 · (patch_size / R) · ‖S‖` pixels, the largest singular value of the member's shape), with 0.05 px kept as an absolute floor; the rays are weighted by `(f/ρ)²` in the triangulation, so the point and its covariance come from the same estimator; and the both-axes verdict reads the live cells' spread within the fitted plane. How do the verdict shares and the errors move, and how does the cell normal compare with the affine shapes when both use the same cells?

**Data.**

- **Code.** Branch `bootstrap-core-migration` at `0da8d375`, with the commit that adds this section, the extension rebuilt with `pixi run maturin develop --release`.
- **Machine.** Windows 11, Intel Core (family 6 model 183), 32 threads.
- **Cluster files, poses, truth and matching.** As in the [section above](#cell-plane-normals-against-the-ground-truths-2026-10-08): the same `SeoulBull` and `KerryPark480` `--piecewise` cluster files, the ground truths' cameras, poses and patch-frame normals, and a cluster mapped to a ground-truth point when its reference member lies within 1 px or 3 px of one of that point's keypoints in the same image. One ground-truth point can be matched by several clusters, so the matched clusters are not independent samples; the tables give the number of distinct points.
- **Variants.** *cells*: the kernel at its defaults. *shapes only*: each cell that had a measured shift gets a zero displacement, and every cell status stays as stored, so the cells and their support are those of *cells* and only the displacement differs. This replaces the earlier variant, which marked all nine cells of every kept member `fitted`. *view_dir*: the mean unit direction from the cells toward their rays' cameras.
- **Errors.** As above: the angle to the truth normal for both-axes clusters, the error of the fixed component for one-axis clusters.

**Verdicts**, as a share of the clusters with a reference (3934 and 11865):

| entry | both axes % | one axis % | none % | none: no kept member % | none: kept member, < 3 cells triangulated % | none: ≥ 3 triangulated, < 3 live % | before the audit: both / one / none % |
|---|---|---|---|---|---|---|---|
| SeoulBull | 29.7 | 5.2 | 65.1 | 48.5 | 16.6 | 0.0 | 14.9 / 4.5 / 80.6 |
| KerryPark480 | 17.5 | 7.8 | 74.7 | 25.3 | 49.4 | 0.0 | 11.2 / 6.1 / 82.7 |

Of the clusters with at least one kept member (51.5% and 74.7% of the clusters with a reference), 57.7% and 23.4% get both axes and 10.0% and 10.5% one axis. *Shapes only* gives 29.6% and 17.6% both axes of the clusters with a reference. The in-plane cells' residual is unchanged, a median 0.17 px and 0.13 px, p90 0.42 and 4.0 px.

**Both-axes clusters**, median / p90 error in degrees. *All*: every matched cluster *cells* calls both-axes. *Paired*: those that *shapes only* also calls both-axes.

| entry @ tolerance | all: n | cells | view_dir | paired: n (GT points) | cells | shapes only | view_dir |
|---|---|---|---|---|---|---|---|
| SeoulBull @ 1 px | 14 | 20.0 / 28.7 | 28.3 / 43.7 | 14 (13) | 20.0 / 28.7 | 13.6 / 28.0 | 28.3 / 43.7 |
| SeoulBull @ 3 px | 89 | 19.1 / 60.1 | 29.3 / 49.3 | 88 (65) | 19.2 / 60.2 | 21.0 / 53.0 | 29.5 / 49.4 |
| KerryPark480 @ 1 px | 21 | 49.5 / 73.6 | 21.7 / 38.1 | 19 (16) | 48.6 / 66.7 | 36.0 / 64.9 | 21.7 / 41.3 |
| KerryPark480 @ 3 px | 92 | 20.6 / 60.6 | 30.0 / 58.5 | 89 (63) | 19.3 / 59.4 | 16.6 / 57.7 | 30.1 / 58.9 |

The matched counts are those of the section above: 27 clusters on 23 ground-truth points and 164 on 116 for `SeoulBull`, 80 on 43 and 323 on 161 for `KerryPark480`.

**One-axis clusters**, error of the fixed component, median / p90:

| entry @ tolerance | n | cells | view_dir |
|---|---|---|---|
| SeoulBull @ 1 px | 2 | 21.6 / 26.8 | 38.8 / 44.6 |
| SeoulBull @ 3 px | 13 | 28.1 / 63.8 | 26.7 / 45.8 |
| KerryPark480 @ 1 px | 15 | 36.2 / 73.2 | 29.3 / 59.1 |
| KerryPark480 @ 3 px | 39 | 22.2 / 72.2 | 35.1 / 65.8 |

**Result.**

- Under the new live rule no cluster with three or more triangulated cells is left without a verdict. The both-axes share doubles on `SeoulBull` (14.9% to 29.7% of the clusters with a reference) and rises from 11.2% to 17.5% on `KerryPark480`. Every remaining cluster without a normal lacks support: it has no kept member, or fewer than three cells triangulated, on `KerryPark480` mostly for narrow baselines.
- The clusters the verdict now admits are less accurate. At 3 px the both-axes median error rises from 15.5° to 19.1° on `SeoulBull` and from 17.6° to 20.6° on `KerryPark480`, still below the viewing direction's 29.3° and 30.0°; the p90 stays near 60°. The four changes were applied together, and this run does not split the change in error among them.
- With the same cells, the displacements do not beat the affine shapes. On the paired 3 px sets *cells* is 1.8° better in median on `SeoulBull` (19.2° against 21.0°) and 2.7° worse on `KerryPark480` (19.3° against 16.6°); at 1 px the shapes are better on both (13.6° against 20.0°, 36.0° against 48.6°), on 14 and 19 clusters.
- On one-axis clusters at 3 px the fixed component beats the viewing direction on `KerryPark480` (22.2° against 35.1°) and does not on `SeoulBull` (28.1° against 26.7°, 13 clusters).
- `KerryPark480` at 1 px is still the set where the cell normal loses to the viewing direction (49.5° against 21.7°).

**What this decides.**

- **The four changes stay.** They make the verdict say what it is defined to say, whether the cells the fit keeps spread in two directions within the plane, and make the weights match the noise model. That the newly admitted clusters are less accurate is a statement about their precision, which the verdict is not meant to carry.
- **A precision gate is now needed before any consumer reads the normal.** The verdict admits more low-precision clusters than before, so the next step of the section above, returning the predicted precision from the kernel and measuring a gate on it, comes first.
- **The displacements against the shapes stays open.** With the same cells the two are within 3° in median at 3 px, in opposite directions on the two entries, and this ground truth cannot separate them, for the reasons in the section above.

## Human review of moved shapes (2026-10-09)

**Question.** The sections above judged the shape-moving loop by its agreement with the cascade and by the seed's pick on `KerryPark480`. Both measure agreement with the cascade, not correctness: a correction of a shape the cascade left at a wrong local optimum counts as a disagreement on the first, and the seed's pick turned on 55 moved members without saying whether they moved the right way. Where the loop moves a member, is the moved shape or the cascade's shape the better alignment of the member's photograph with the reference's?

**Data.**

- **Files.** Branch `bootstrap-core-migration` at `ebce954f`. On `SeoulBull`, `KerryPark480`, `fleetws` and `MurdoSmallAntiqueCat`, each from the clusters file of the [Setup](#setup-of-the-fleet-run-f787be3b), two cluster-patches files at `--patch-size 12`, written through the command's pipeline: a cascade-only file, and a piecewise file with `move_shape` true. The piecewise files are identical, column for column, to the ones the new default writes ([next section](#subset-with-the-loop-as-the-default-2026-10-09)). The member statuses and references are identical between the two. A *mover* is a kept member whose shape or position differs between them; the four entries hold 6, 60, 417 and 8,556 movers. Its *movement* is the largest displacement of a template corner from the cascade's footprint to the moved one, in the cascade's grid px.
- **Cases.** 90 cases, drawn with a fixed seed (`20261008`):
  - 60 **movers**, stratified by movement: all movers sorted by movement and split into four equal-count bins, 15 drawn from each bin round-robin over the entries. All six `SeoulBull` movers are among them; the rest are 16 from `KerryPark480`, 18 from `fleetws` and 20 from `MurdoSmallAntiqueCat`. The quartile edges of the drawn movements are 0.53, 1.36 and 2.02 grid px, and the largest is 5.1.
  - 20 **controls**, five per entry: a kept member that did not move, shown as the cascade's shape against the piecewise file's identical shape. A preference on a control is a false preference.
  - 10 **sanity** pairs, two or three per entry: a kept member that did not move, shown as the cascade's shape against the same shape rotated by 5° and shifted by 1.5 grid px in a random direction, which moves its corners 2.8 to 3.0 grid px.
- **Renderer.** For each case the page showed the reference member's patch and the two candidate patches, each rendered as the kernel renders it: the template's 25 × 25 grid, at 4 display px per grid px, through the same pyramid-level rule, pixel-centre convention and bilinear sampling. Beside each patch was a crop of its photograph with the footprint outlined, and a crop of the member's photograph with both footprints outlined, one orange (A) and one blue (B).
- **Blind A/B.** Which side was A was drawn at random per case, and the cases were shown in a random order with nothing on the page naming the sides. The reviewer, the maintainer, chose A, B, *same* or *both wrong*, with an optional note, and could place a corrected footprint by hand on a full-resolution crop. The key was revealed once, behind a confirmation, after every case had been answered; no choice was changed after the reveal.
- **Limits of the review.** One reviewer judged every case, and the reviewer is the maintainer. The controls were recognisable as controls, since their two footprints coincide on the member's photograph, though which side was the cascade's stayed hidden.

**Result: the reviewer preferred the moved shape, and more strongly the further it moved.**

| movement quartile (grid px) | cases | moved preferred | cascade preferred | same | both wrong |
|---|---|---|---|---|---|
| Q1, under 0.53 | 15 | 1 | 1 | 11 | 2 |
| Q2, 0.53 to 1.36 | 15 | 8 | 1 | 6 | 0 |
| Q3, 1.36 to 2.02 | 15 | 13 | 1 | 1 | 0 |
| Q4, 2.02 and over | 15 | 15 | 0 | 0 | 0 |
| all movers | 60 | 37 | 3 | 18 | 2 |

Among the movers the moved shape was preferred 37 times and the cascade's 3 times. Below about half a grid px the two are mostly indistinguishable (11 of 15 *same*); from there the moved shape is preferred 36 to 2, and every one of the 15 largest movements was preferred. The three cascade preferences moved 0.15, 0.94 and 1.56 grid px. Every drawn mover's whole-member ZNCC rose, by a median of 0.003 and at most 0.067, so the ZNCC favours the moved shape in all 60 cases, the three the reviewer gave to the cascade included.

| control choice | cases |
|---|---|
| same | 14 |
| cascade side | 2 |
| piecewise side (identical) | 3 |
| both wrong | 1 |

The controls read *same* 14 times in 20. The five false preferences split 2 and 3 between the two identical sides, so they say how often a pair of identical patches gets a preference anyway, not that one side was favoured. The 37 to 3 among movers is far outside that.

The sanity pairs chose the unperturbed shape 10 times in 10, so a misalignment of about 3 grid px at the corners is visible on these renders.

**Spurious members.** Four cases were not misalignments but wrong members: the cluster kept a member whose photograph shows a different surface, and no shape of it is right. The reviewer's notes, paraphrased:

| case | entry | type | movement grid px | choice | note |
|---|---|---|---|---|---|
| c15 | KerryPark480 | mover | 0.19 | same | a street with cars and a bicycle matched to a dog's shadow and leash |
| c33 | SeoulBull | mover | 0.49 | both wrong | no note |
| c39 | fleetws | control | 0 | both wrong | the two photographs are not of the same subject |
| c71 | KerryPark480 | mover | 0.34 | both wrong | a building matched to a tree, with a similar cloud pattern in the sky behind both |

All four passed the cascade's gates, at ZNCCs of 0.85 to 0.93.

**The adjustment on c01.** One footprint was placed by hand, on control c01 (`MurdoSmallAntiqueCat`, member 649526, cluster 135995), which the reviewer marked *same* with the note "both wrong". There the cascade's shape, and the identical piecewise one, sat in a local optimum one period of a repeating texture off: the corrected footprint moves the centre 3.4 grid px and the corners 22.6 grid px from the cascade's, so it differs in rotation and skew as well as position. The reviewer confirms the corrected footprint is right. Its position and shape, in photograph px, with the footprint `pos + shape · u` for `u` in `[−6, 6]²`:

| footprint | position | shape |
|---|---|---|
| cascade | (2873.03, 55.14) | [[−0.0086, −4.1378], [2.9099, 1.1385]] |
| corrected by hand | (2868.89, 52.55) | [[−1.9909, −3.8026], [2.1223, −1.9894]] |

The cascade's whole-member ZNCC there is 0.895, past the 0.85 bar. The loop rejected its first update and left the member where it was: six of its nine cells were fitted, with shifts under 1.5 grid px that agree with the wrong optimum, since the right one lies 3.4 grid px away at the centre, past the cells' search bound of 2 grid px.

**Archive.** One row per case is in [cluster-patch-refinement-human-review-2026-10-09.csv](cluster-patch-refinement-human-review-2026-10-09.csv): the case id, entry, type, which side was A and which B, the member and cluster indexes in the cluster-patches files, the movement in grid px, the cascade's and the moved side's stored ZNCC (empty for a sanity pair, whose perturbed side is not in any file), the raw choice and the side it names, the note, and for c01 the corrected footprint and the side it was placed from. The renders and the review page are not kept.

**What this decides.**

- **The loop is the default.** The decision to leave the shape to the cascade by default rested on agreement with the cascade and on one seed pick, and both measure agreement, not correctness. Where the loop moves a shape by more than about half a grid px, the reviewer preferred the moved shape 36 times to 2. `PiecewiseParams::move_shape` is now true by default; off, the stage is the measurement of the [section above](#subset-with-the-shape-left-to-the-cascade-2026-10-08).
- **Agreement with the cascade is retired as a metric.** It counts every correction as an error. A shape metric for this stage needs a reference that is not the cascade: a human judgment like this one, or a ground truth's reprojected footprint.
- **Spurious members are a matching and gating defect.** Four of 90 cases kept a member of a different surface at a ZNCC over 0.85. Moving the shape cannot fix them, and the review says nothing about how often they occur in the files at large; detecting them belongs to the matcher or to a gate on the kept members.
- **The cascade can sit in a texture-period optimum the loop does not leave.** c01 shows a member kept 22.6 grid px off at the corners, on a control the loop did not move. How often this happens is not measured.

## Subset with the loop as the default (2026-10-09)

**Question.** After the [human review](#human-review-of-moved-shapes-2026-10-09), `PiecewiseParams::move_shape` is true by default, so `sfm cluster-patches --piecewise` moves shapes. On the five entries of the subset, how many kept members move and how far, does the whole-member ZNCC fall anywhere, what does the stage cost, and what does the seed give on the two ground truths?

**Method.**

- **Code.** Branch `bootstrap-core-migration` at `ebce954f`, with the change of the commit that adds this section (the default flipped), and the extension rebuilt with `pixi run maturin develop --release`. Same machine, with no other workload running.
- **Runs.** The five entries and the two CLI runs of the [acceptance-rule section](#subset-with-the-acceptance-rule-2026-10-08) (`--no-piecewise` and `--piecewise`, `--patch-size 12`, `SFMTOOL_PROFILE=1`), one after the other, the piecewise run now at the new default. A third run, through the binding with `piecewise=True` and no other piecewise setting, read why each loop stopped; its statuses and cells are identical to the CLI file's on every entry.
- **Checks.** The cascade-only files are identical in their seven member columns to those of the [section with the shape left to the cascade](#subset-with-the-shape-left-to-the-cascade-2026-10-08). On the four entries of the human review, the piecewise files are identical in every member and cell column to the moved files the review was drawn from.
- **Seed.** As under [Seed stage on the ground-truth entries](#seed-stage-on-the-ground-truth-entries), with the same environment, scoring and pass rule, on `SeoulBull` and `KerryPark480` with the new piecewise files. The cascade-only baseline is that section's cascade tables, which the cascade-only files reproduce. The `KerryPark480` run was repeated, and every number of the repeat was identical.

**Result: the loop moves 0.2% to 1.4% of kept members, and never lowers a whole-member ZNCC.**

| entry | kept | moved | moved % | movement grid p50/p95/max (moved) | dzncc min (all kept) | dzncc p50/p95/max (moved) | final update rejected % | kernel wall cascade/piecewise s | wall ratio | CPU ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| SeoulBull | 2953 | 6 | 0.20 | 0.29/3.25/3.78 | 0.0000 | 0.0018/0.0344/0.0389 | 100.0 | 0.3/0.4 | 1.13 | 1.11 |
| KerryPark480 | 15986 | 60 | 0.38 | 0.28/2.23/3.34 | 0.0000 | 0.0004/0.0177/0.0667 | 99.9 | 1.3/1.6 | 1.17 | 1.16 |
| fleetws | 68722 | 417 | 0.61 | 0.46/2.17/4.14 | 0.0000 | 0.0005/0.0237/0.0723 | 99.7 | 5.0/5.6 | 1.11 | 1.14 |
| MurdoSmallAntiqueCat | 625198 | 8556 | 1.37 | 0.98/2.25/5.25 | 0.0000 | 0.0049/0.0398/0.1026 | 99.5 | 40.1/47.3 | 1.18 | 1.17 |
| DnDTabletop | 1375116 | 18765 | 1.37 | 0.40/1.93/4.49 | 0.0000 | 0.0007/0.0266/0.1141 | 99.4 | 73.6/76.3 | 1.04 | 1.25 |

The movement is the largest displacement of a cell centre (at ±4 keypoint units, the total movement of [Parity with the cascade](#parity-with-the-cascade)) from the cascade's placement, in the cascade's grid px; the human review measured to the template's corners instead, at ±6, which reads larger; over all kept members its median and 95th percentile are 0.00 on every entry. No kept member's whole-member ZNCC falls; among the moved members it rises by a median of 0.0004 to 0.0049 and at most 0.114. "Final update rejected" is the share, among members whose loop ran, whose last fitted update was not applied.

| entry | not run | converged | cap | rejected | oscillation | iterations 1/2/3/4/5 % |
|---|---|---|---|---|---|---|
| SeoulBull | 3.7 | 0.0 | 0.0 | 96.3 | 0.0 | 99.8/0.1/0.1/0.0/0.0 |
| KerryPark480 | 1.8 | 0.1 | 0.0 | 98.1 | 0.0 | 99.7/0.2/0.1/0.0/0.0 |
| fleetws | 5.0 | 0.2 | 0.0 | 94.7 | 0.0 | 99.4/0.4/0.1/0.0/0.0 |
| MurdoSmallAntiqueCat | 0.9 | 0.4 | 0.0 | 98.6 | 0.0 | 98.6/0.8/0.4/0.2/0.1 |
| DnDTabletop | 0.2 | 0.6 | 0.0 | 99.2 | 0.0 | 98.6/0.9/0.3/0.1/0.0 |

The loop's behaviour matches the [acceptance-rule section](#subset-with-the-acceptance-rule-2026-10-08), which ran it with the one-dimensional residual scale: there it moved 0.17%, 0.34%, 0.64%, 1.44% and 1.42% of kept members, here 0.20%, 0.38%, 0.61%, 1.37% and 1.37%. No member reaches the cap or stops for oscillation. The cell statuses are within 0.1 point of the [two-dimensional residual scale section](#subset-with-the-two-dimensional-residual-scale-2026-10-08)'s: 42.1% to 83.7% of cells fitted and 2.6% to 3.4% refused as outliers. The CPU ratio, which contention does not affect, is 1.11× to 1.25×, against the acceptance-rule section's 1.12× to 1.25×; the wall ratio is noisier.

**Seed.** `SeoulBull` gives the cascade-only table to the last digit: the pick `h00` passes (0.995°, 0.15%, +4.57%), and `h01` passes. Its six moved members change no score.

`KerryPark480`'s pick `h00` passes, and by a wider margin than with the cascade file: it poses 26 cameras instead of 18, with a median rotation error of 0.452° instead of 0.710° and a median centre error of 0.50% instead of 0.91%.

| file (piecewise, `move_shape` default) | qualified | accepted | posed | rot med/max deg | centre med/max % | f err % (manifest / equiv) | pass |
|---|---|---|---|---|---|---|---|
| h00.sfmr | 1 | 1 | 26 | 0.452/1.18 | 0.50/1.56 | +0.12 / +0.10 | PASS |
| h01.sfmr | 1 | 1 | 14 | 0.711/1.63 | 1.70/5.02 | +0.32 / +0.34 | PASS |
| h02-spline-refused.sfmr | 1 | 0 | 14 | 5.898/158.12 | 14.84/46.54 | +1.72 / +1.01 | fail |
| h02.sfmr | 1 | 0 | 14 | 6.146/157.69 | 14.84/46.54 | +1.72 / +1.72 | fail |
| h03.sfmr | 1 | 1 | 14 | 0.444/1.81 | 14.47/30.77 | +0.61 / +0.37 | fail |
| h04.sfmr | 1 | 1 | 14 | 2.738/174.64 | 8.46/45.15 | +1.57 / +1.58 | fail |
| h05.sfmr | 1 | 1 | 10 | 0.886/2.06 | 17.12/43.77 | +1.35 / +0.81 | fail |
| h06-spline-refused.sfmr | 1 | 0 | 14 | 8.483/149.22 | 25.59/38.29 | +1.15 / +0.72 | fail |
| h06.sfmr | 1 | 0 | 14 | 8.499/150.15 | 25.56/38.28 | +1.15 / +1.15 | fail |
| h07.sfmr | 1 | 1 | 14 | 32.207/140.43 | 15.66/21.66 | -3.37 / -3.00 | fail |

The candidates themselves change. Two of the eight finite hypotheses pass, against five with the cascade file, and only two of the eight start from the same seed frames as a cascade-file hypothesis:

| seed frames | cascade file | piecewise file |
|---|---|---|
| none (the full admission) | `h00`, 18 posed, PASS | `h00`, 26 posed, PASS |
| 45, 46, 47 | `h05`, 14 posed, PASS | `h05`, 10 posed, fail (centre 17.1%) |
| 10, 11 | `h06`, 14 posed, fail (centre 9.0%) | `h03`, 14 posed, fail (centre 14.5%) |
| 24 to 28; 7, 8; 10, 34, 35; 12, 13, 42; 19, 43 | `h01` to `h04`, `h07`: four PASS, `h04` and `h07` fail | none |
| 7, 8, 30, 31; 38, 39; 6, 7; 15, 16; 4, 9 | none | `h01` PASS; `h02`, `h04`, `h06`, `h07` fail |

With the acceptance-rule section's file, which moved 55 members of `KerryPark480` under the one-dimensional residual scale, `h00` posed 26 cameras and failed (1.071°, 10.4%). The file here moves 60, 49 of them members that file moved too, and `h00` passes. Two files that differ in a few dozen members' shapes give a passing and a failing pick.

**What this decides.**

- **The loop holds its contract as the default.** The whole-member ZNCC never falls, no member reaches the cap or oscillates, and the CPU cost is 1.11× to 1.25× the refinement's.
- **The seed's pick on `KerryPark480` passes, and the set of candidates changes.** This is not a regression of the stage: the pick is better than with the cascade file, and the human review says the moved shapes are better. It is a seed-brittleness finding. Sub-percent changes in member shapes change which hypotheses the seed commits and which of them pass, and the pick turns pass or fail on them. A photometric score of the seed's candidates is what would have to make the pick robust to it.

## Gates at the refined shape (2026-10-09)

**Question.** The [human review](#human-review-of-moved-shapes-2026-10-09) found four spurious members, wrong correspondences the cluster kept. Two gates read each member's own patch again after the refinement, at its refined shape ([the member gate at the refined shape](cluster-patch-refinement.md#the-member-gate-at-the-refined-shape)): the whole grid against the up-front bar, and the count of its nine cells that read the largest radius, 3 (*capped*). How many kept members does each refuse, how often are those members wrong, which bar for the cell count do the data support, and does either refuse many right members?

**Method.**

- **Code.** Branch `bootstrap-core-migration` at `589ce099`, with the changes of the commit that adds this section, the extension rebuilt with `pixi run maturin develop --release`.
- **Readings.** On the five entries of the [subset](#subset-with-the-loop-as-the-default-2026-10-09), the binding was run twice with `piecewise=True`, `radius=6` and the other settings at their defaults: with both gates off, and with the whole grid's gate on and the cell gate off (`max_capped_cells=9`), so that every member passing the ZNCC and shift gates carries both readings. With both gates off, every status, ZNCC and stored position is identical to the piecewise files of the [previous section](#subset-with-the-loop-as-the-default-2026-10-09). The tables count over the members those files keep.
- **Ground truth.** Each entry's ground-truth reconstruction gives the poses of its images. A kept member is judged when its own image and its reference's image both have a ground-truth pose: all kept members on `SeoulBull` (2,953), `KerryPark480` (15,986), `fleetws` (68,722) and `DnDTabletop` (1,375,116), and 585,707 of the 625,198 on `MurdoSmallAntiqueCat`, whose ground truth poses 349 of the file's 373 images, so 6.3% of its kept members are not judged. A kept member is *wrong* when its refined position lies further than `max(3, 5·m)` px of its own photograph from the epipolar half-line of its reference's detection (the directions from the member's camera to the points of the reference's ray in front of the reference's camera), with `m` the entry's median of that distance over kept members, and *right* otherwise. The median is 0.21 px on `SeoulBull`, 0.22 on `KerryPark480`, 1.74 on `MurdoSmallAntiqueCat` and 0.94 on `DnDTabletop`. On `fleetws` it is 6.9 px: its reconstruction does not describe these images closely enough, and the entry is left out of every right and wrong count. A wrong correspondence along the epipolar line counts as right, so the wrong counts are lower bounds.
- **Files.** `sfm cluster-patches --piecewise --patch-size 12` at the chosen defaults wrote one file per entry. Every file verifies at format version 10, and every member it keeps the ungated file kept. The gates run before the per-image dedupe, so a refusal at the cascade's shape costs its image no member that would have passed. In the code measured here, a member whose moved shape failed a gate was refused after the dedupe: the whole grid's gate refused 0, 0, 1, 3 and 3 members after a piecewise move on the five entries in table order, each passed at its cascade shape, and each left its image with no member of the cluster; the cell rule refused none after a move. The stage now reverts such a move to the cascade's shape and keeps the member ([the member gate at the refined shape](cluster-patch-refinement.md#the-member-gate-at-the-refined-shape)), so at the defaults the whole grid's refusals below are smaller by those seven.
- **Seed.** As under [Seed stage on the ground-truth entries](#seed-stage-on-the-ground-truth-entries), with the same environment, scoring and pass rule, on `SeoulBull` and `KerryPark480` with the gated files, against the previous section's piecewise results.

**Result: the radius at the refined shape and the capped-cell count both separate wrong members, the cell count only at its top.**

Wrong share of kept members by radius at the refined shape, `SeoulBull` and `KerryPark480` pooled:

| radius, grid px | members | wrong |
|---|---|---|
| under 1 | 9,578 | 6.5% |
| 1 to 1.5 | 5,123 | 18.5% |
| 1.5 to 2 | 2,762 | 26.5% |
| 2 to 2.25 | 638 | 31.3% |
| 2.25 to 2.5 | 328 | 38.1% |
| 2.5 to 2.75 | 161 | 49.1% |
| 2.75 to 2.9 | 53 | 52.8% |
| 2.9 to under 3 | 36 | 69.4% |
| 3 | 260 | 86.2% |

Wrong share of judged kept members by capped cells, with the number of kept members in brackets, and on `MurdoSmallAntiqueCat` the number judged after it:

| capped cells | SeoulBull | KerryPark480 | MurdoSmallAntiqueCat | DnDTabletop |
|---|---|---|---|---|
| 0 | 2.2% (1,931) | 9.7% (9,126) | 8.4% (461,809; 433,208) | 8.5% (945,266) |
| 1 | 6.1% (506) | 20.5% (2,596) | 13.7% (69,805; 65,127) | 9.7% (186,626) |
| 2 | 10.8% (231) | 27.6% (1,626) | 18.1% (37,879; 35,525) | 11.1% (98,835) |
| 3 | 12.3% (122) | 29.7% (1,186) | 22.8% (25,673; 24,024) | 12.4% (64,061) |
| 4 | 30.5% (59) | 33.4% (686) | 29.3% (14,282; 13,331) | 14.6% (37,807) |
| 5 | 38.9% (36) | 38.6% (404) | 35.3% (8,284; 7,671) | 17.1% (22,296) |
| 6 | 50.0% (20) | 35.2% (193) | 40.6% (4,356; 4,023) | 21.1% (12,251) |
| 7 | 55.6% (9) | 48.8% (82) | 49.3% (1,930; 1,746) | 23.1% (5,820) |
| 8 | 83.3% (12) | 70.6% (34) | 64.4% (727; 663) | 30.4% (1,913) |
| 9 | 96.3% (27) | 92.5% (53) | 95.1% (453; 389) | 72.6% (241) |

The wrong share rises with the count on every entry except one step, `KerryPark480` from five capped cells to six (38.6% to 35.2%), and only nine capped cells is a majority wrong on all four. A bar of `2`, refusing three or more capped cells, would refuse 9.7%, 16.5%, 27.0%, 8.9% and 10.5% of the kept members of the five entries in table order, more right members than wrong on each of the four judged.

**Defaults.** The whole grid's gate shares the up-front bar, `2.5`; in the pooled bands above, the wrong share passes one half between 2.5 and 2.9, so the data put no other bar there. The cell bar is the smallest bar above which members are more often wrong than right on every judged entry: `8`.

**Refusals at the defaults.**

| entry | kept without the gates | refused, whole grid | refused, cells | cell rule also over, among the whole grid's | refused % | refused right / wrong | right refused, % of right | kept wrong % |
|---|---|---|---|---|---|---|---|---|
| SeoulBull | 2,953 | 133 | 0 | 27 | 4.50 | 57 / 76 | 2.07 | 6.7 |
| KerryPark480 | 15,986 | 377 | 0 | 53 | 2.36 | 97 / 280 | 0.74 | 17.4 |
| fleetws | 68,722 | 2,721 | 0 | 297 | 3.96 | not judged | not judged | not judged |
| MurdoSmallAntiqueCat | 625,198 | 5,144 | 14 | 439 | 0.83 | 1,930 / 2,730 | 0.37 | 11.4 |
| DnDTabletop | 1,375,116 | 4,572 | 53 | 188 | 0.34 | 3,004 / 1,621 | 0.24 | 9.6 |

The whole grid's gate is read first, so the cell rule refuses the members of the "cells" column and of the column after it, and the two gates overlap in that later column. The cell rule adds members to the whole grid's refusals only on `MurdoSmallAntiqueCat` (14) and `DnDTabletop` (53). The refused members are wrong at 35% to 74%, three to nine times the entry's share among all kept members. The two readings cost +9.5%, +12.0%, +5.1%, +3.5% and +2.5% of the binding call's process CPU time on the five entries in table order, against the call with both gates off.

**The review's spurious members.**

| case | entry | member | radius at refined shape | capped cells | distance from the epipolar half-line, px | status |
|---|---|---|---|---|---|---|
| c15 | KerryPark480 | 20286 | 2.60 | 1 | 257 | `rejected_unlocalizable_refined` |
| c39 | fleetws | 101521 | 1.68 | 1 | 406 | kept |
| c71 | KerryPark480 | 10868 | 1.07 | 0 | 227 | kept |
| c33 | SeoulBull | 7160 | 0.22 | 0 | 0.2 | kept |
| c01 | MurdoSmallAntiqueCat | 649526 | 1.28 | 0 | 2.5 | kept |

c15 reads 1.43 at its detection's shape and 2.60 at the refined shape, and the whole grid's gate refuses it. c39 has one capped cell, its bottom middle one, and its reference has two; the bar that would refuse it, `0`, refuses about a third of all kept members. c33 and c01 lie on the epipolar line within the ground truth's noise: c01 sits one period of a repeating texture off along it, and c33 is a wrong correspondence on distinctive texture, if it is one. No reading of the member's own patch can tell either from a right member. c71 and c39 are wrong by the ground truth and pin a position.

**Spot check.** For `SeoulBull` and `KerryPark480`, 12 members drawn at random (seed `20261009`) from those the whole grid's gate refuses and 12 from those the cell rule refuses (all also refused by the whole grid's gate on these two), and 12 of the 53 `DnDTabletop` members only the cell gate refuses, were rendered as the reference's tile, the member's tile at its refined shape and a crop of the member's photograph with the footprint drawn, for the maintainer to look at. The sheets are not kept.

**Seed.**

| entry | file | posed | rot med/max deg | centre med/max % | f err % (manifest / equiv) | pass |
|---|---|---|---|---|---|---|
| SeoulBull | ungated `h00` | 8 | 0.995/1.97 | 0.15/0.24 | +4.68 / +4.57 | PASS |
| SeoulBull | gated `h00` | 8 | 0.963/2.05 | 0.19/0.37 | +0.85 / +0.85 | PASS |
| KerryPark480 | ungated `h00` | 26 | 0.452/1.18 | 0.50/1.56 | +0.12 / +0.10 | PASS |
| KerryPark480 | gated `h00` | 28 | 0.432/1.27 | 0.67/1.54 | +0.04 / −0.03 | PASS |

Both picks pass. On `SeoulBull` the pick's focal error falls from 4.6% to 0.9%, but its released camera is no longer marked accepted, and the second finite hypothesis `h01` fails on its focal (+5.9%) where it passed before; its spline-refused variant passes. On `KerryPark480` the pick poses two more cameras, and none of the other hypotheses passes, where `h01` passed before. As in the previous section, which hypotheses the seed commits moves with small changes to the member set.

**What this decides.**

- **Both gates are on by default**, the whole grid's at the shared bar and the cells' at `8`. Each refuses at most 2.1% of the members the ground truth calls right, and the members they refuse are wrong several times as often as the kept ones.
- **The cell gate adds little at its bar.** Its refusals are almost all the whole grid's too, and the bar review case c39 would need refuses more right members than wrong on every judged entry.
- **Spurious members whose patch pins a position remain.** A wrong correspondence on distinctive texture, on the epipolar line or off it, passes both gates; refusing it needs a reading across the cluster's members, not of one member's patch.

## Cell plane normals at the current defaults (2026-10-09)

**Question.** The [cell plane normals after the audit](#cell-plane-normals-after-the-audit-2026-10-08) were measured on files written with `move_shape` off and before the [gates at the refined shape](#gates-at-the-refined-shape-2026-10-09). At the current defaults the loop moves a few members' shapes, and the two gates refuse some kept members. Do the verdicts and the errors against the ground truths change, and which of the two changes moves them?

**Data.**

- **Code.** Branch `bootstrap-core-migration` at `5de03f60`, the extension rebuilt with `pixi run maturin develop --release`. No code changes.
- **Machine.** Windows 11, Intel Core (family 6 model 183), 32 threads.
- **Cluster files.** Three per entry, from the same `SeoulBull` and `KerryPark480` cluster files: *old*, the `--piecewise` files of the [first cell plane normal section](#cell-plane-normals-against-the-ground-truths-2026-10-08) (`move_shape` off, no refined-shape gates); *defaults*, `sfm cluster-patches --patch-size 12 --piecewise` at the current defaults (`move_shape` on, both gates on); and *gates off*, the same with `--no-regate-at-refined-shape --max-capped-cells 9`. *Old* to *gates off* is the effect of the moved shapes alone, and *gates off* to *defaults* that of the gates alone. Every file has the same clusters, members and references.
- **Poses, truth, matching, variants and errors.** As in the [section after the audit](#cell-plane-normals-after-the-audit-2026-10-08), with its script; run on *old* it reproduces that section's tables to the last digit. The matched counts are unchanged, since the reference members do not move: 27 clusters on 23 ground-truth points and 164 on 116 for `SeoulBull`, 80 on 43 and 323 on 161 for `KerryPark480`.

**Members.**

| entry | kept, old | kept, defaults | moved (shape or position) | move refused, cells not attempted | refused by the gates | moved, and refused by a gate at the defaults |
|---|---|---|---|---|---|---|
| SeoulBull | 2953 | 2820 | 6 | 0 | 133 | 1 |
| KerryPark480 | 15986 | 15609 | 60 | 2 | 377 | 4 |

A member is *moved* when its shape or position in *gates off* differs from *old*. A move the loop refuses for its ZNCC or shift keeps the cascade's shape and stores every cell `not_attempted`, so those two members lose their cells. The five moved members a gate refuses at the defaults fail it at their cascade shape, since a moved shape that fails a gate is reverted rather than refused. The gates leave every member they keep with the shape, position and cells it has in *gates off*.

**Verdicts at the defaults**, as a share of the clusters with a reference (3934 and 11865), with the section after the audit's in brackets:

| entry | both axes % | one axis % | none % | none: no kept member % | none: kept member, < 3 cells triangulated % | of which ≥ 1 narrow-baseline cell / too few rays only % |
|---|---|---|---|---|---|---|
| SeoulBull | 29.7 (29.7) | 5.0 (5.2) | 65.3 (65.1) | 50.4 (48.5) | 14.9 (16.6) | 8.7 / 6.2 (8.9 / 7.7) |
| KerryPark480 | 17.4 (17.5) | 7.7 (7.8) | 74.9 (74.7) | 27.1 (25.3) | 47.8 (49.4) | 40.4 / 7.4 (40.5 / 8.9) |

No cluster with three or more triangulated cells is without a verdict, as before. Of the clusters with at least one kept member (49.6% and 72.9% of the clusters with a reference, against 51.5% and 74.7%), 59.9% and 23.9% get both axes and 10.1% and 10.6% one axis. *Shapes only* gives 29.6% and 17.5% both axes. The in-plane cells' residual is unchanged, a median 0.17 px and 0.13 px, p90 0.42 and 4.0 px.

**Both-axes clusters at the defaults**, median / p90 error in degrees, as in the section after the audit:

| entry @ tolerance | all: n | cells | view_dir | paired: n (GT points) | cells | shapes only | view_dir |
|---|---|---|---|---|---|---|---|
| SeoulBull @ 1 px | 14 | 20.0 / 28.7 | 28.3 / 43.7 | 14 (13) | 20.0 / 28.7 | 13.2 / 28.0 | 28.3 / 43.7 |
| SeoulBull @ 3 px | 88 | 19.0 / 58.6 | 29.1 / 49.4 | 87 (64) | 19.1 / 58.8 | 20.1 / 49.7 | 29.8 / 49.6 |
| KerryPark480 @ 1 px | 19 | 49.5 / 74.4 | 25.7 / 41.3 | 17 (14) | 48.6 / 69.4 | 36.0 / 65.7 | 25.7 / 44.5 |
| KerryPark480 @ 3 px | 89 | 20.3 / 61.3 | 30.1 / 58.9 | 86 (62) | 19.2 / 59.8 | 15.6 / 58.8 | 30.2 / 59.4 |

**One-axis clusters at the defaults**, error of the fixed component, median / p90:

| entry @ tolerance | n | cells | view_dir |
|---|---|---|---|
| SeoulBull @ 1 px | 2 | 21.6 / 26.8 | 38.8 / 44.6 |
| SeoulBull @ 3 px | 14 | 29.7 / 65.1 | 22.3 / 45.7 |
| KerryPark480 @ 1 px | 15 | 36.2 / 73.2 | 29.3 / 59.1 |
| KerryPark480 @ 3 px | 39 | 22.2 / 72.2 | 35.1 / 65.8 |

On *gates off*, every *cells* and *view_dir* figure of these three tables equals the section after the audit's, and the verdict shares differ by at most 0.1 point. Only *shapes only* moves, since a moved member's shape is where that variant places its cells: on the paired `SeoulBull` sets from 13.6° to 13.2° at 1 px and from 21.0° to 20.5° at 3 px.

**Per cluster.** Each cluster with a reference is compared with itself across two files: whether its verdict changes, and the angle between the two normals where both files call it both-axes (median / p90 / max, degrees). *Mover clusters* have a member that moved or whose move was refused; *refusal clusters* have a member a gate refuses. Verdicts are written 0 for none, 1 for one axis, 2 for both axes.

| entry | comparison | clusters | verdict changed | changes | both axes in both: n | angle between normals |
|---|---|---|---|---|---|---|
| SeoulBull | old to gates off, mover clusters | 6 | 0 | | 3 | 7.15 / 40.7 / 49.1 |
| SeoulBull | old to gates off, other clusters | 3928 | 0 | | 1167 | 0 / 0 / 0 |
| SeoulBull | gates off to defaults, refusal clusters | 115 | 11 | 2→0: 4; 2→1: 1; 1→0: 2; 1→2: 4 | 12 | 3.05 / 23.9 / 29.4 |
| SeoulBull | gates off to defaults, other clusters | 3819 | 0 | | 1153 | 0 / 0 / 0 |
| KerryPark480 | old to gates off, mover clusters | 62 | 2 | 2→0: 1; 2→1: 1 | 17 | 1.27 / 12.2 / 14.6 |
| KerryPark480 | old to gates off, other clusters | 11803 | 0 | | 2056 | 0 / 0 / 0 |
| KerryPark480 | gates off to defaults, refusal clusters | 340 | 27 | 2→0: 11; 2→1: 2; 1→0: 11; 1→2: 2; 0→2: 1 | 12 | 0.61 / 10.8 / 20.7 |
| KerryPark480 | gates off to defaults, other clusters | 11525 | 0 | | 2048 | 0 / 0 / 0 |

Old to defaults is the two steps together: the clusters with neither a mover nor a refusal (3814 and 11467) have the same verdict and the same normal, bit for bit. On one-axis clusters the moved shapes change the normal by 0.06° (`SeoulBull`, 1 cluster) and by a median of 0.08° (`KerryPark480`, 9 clusters, at most 10.6°). `KerryPark480`'s 2→0 among the mover clusters is a cluster whose refused move took away its member's cells.

Few of the changed clusters are matched to a ground-truth point. At 3 px one mover cluster is matched, on `SeoulBull`: its normal turns 5.2°, and its error goes from 3.4° to 3.5°. The gates take the both-axes verdict from four matched clusters, whose errors were 63.5° (`SeoulBull`, now one axis), 39.1°, 56.6° and 15.1° (`KerryPark480`, now none; the first two are also matched at 1 px), and turn one more `SeoulBull` cluster's normal by 7.3°, its error from 9.9° to 3.0°.

**Result.**

- **The moved shapes do not change the cell normals' accuracy.** They touch 6 and 62 clusters. On `SeoulBull` the three both-axes ones turn by up to 49°, on `KerryPark480` the 17 by a median of 1.3°, and only one of them is matched. Every *cells* and *view_dir* figure against the truths is unchanged.
- **The gates' refusals move the figures by a few tenths of a degree.** They change the verdict of 11 and 27 clusters, most of them to none, and leave 1.9 and 1.8 points of the clusters with a reference with no kept member. The both-axes verdicts they remove among the matched clusters had large errors, so the 3 px both-axes median falls by 0.1° and 0.3° and the `SeoulBull` p90 by 1.5°. That is four clusters, too few to read as a gain.
- **The comparison with the shapes and the viewing direction stands.** On the paired 3 px sets *cells* is 1.0° better than *shapes only* on `SeoulBull` (19.1° against 20.1°) and 3.6° worse on `KerryPark480` (19.2° against 15.6°). On the `KerryPark480` 1 px set, now 17 clusters, the viewing direction is still ahead (25.7° against 48.6°).

**What this decides.**

- **The cell plane normal figures hold at the current defaults.** [cell-plane-normals.md](cell-plane-normals.md#what-the-measurements-show) quotes this section's numbers. The question of the [precision gate draft](../../drafts/cell-plane-normal-precision-gate.md#open-questions) whether the moved shapes change the normals is answered: on these ground truths they do not, measurably.
- Nothing else changes. The displacements against the shapes, and the gate on the normal's precision, stay open for the reasons in the sections above.
