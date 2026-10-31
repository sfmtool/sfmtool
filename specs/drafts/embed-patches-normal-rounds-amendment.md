# Embed Patches: The Normal Step in Every Round

**Status:** Draft. Open: whether `sfm embed-patches` needs the grown-plane
normal step in every round, or once.

Amends: [cli/reconstruction/embed-patches-command.md](../cli/reconstruction/embed-patches-command.md)
§ "Behaviour Notes".

## Purpose

`sfm embed-patches` builds a reconstruction of oriented patches from SIFT
tracks in rounds of "refine normals, select views, refine keypoints". Each
round's normal step runs `sfmtool.bench.refine_normals`, the grown-plane normal
([core/bench/grown-plane-normal.md](../core/bench/grown-plane-normal.md)) on
every finite point. With it the command gives better normals than with the
photometric search it replaced, and takes three to five times as long, most of
that in the normal step of round 2, whose tracks hold the views the vetting
added. This draft records that measurement and the question it leaves: whether
every round needs the normal step, or whether one normal step, in round 1 or in
the last round, gives the same output for less time.

## Measurement (2026-10-30)

**Setup.** `sfm embed-patches` at its defaults (two rounds) on `sift_files`
inputs built as the `seoul_bull_workspace` and `kerry_park_workspace` fixtures
build theirs, by `build_reconstruction_at_poses`: sfmtool SIFT and cluster
matching, the tracks triangulated at the ground truth's poses and kept with
three or more views. Seoul bull has all 17 images (780 points); Kerry Park has
all 48 images rather than the fixture's 16 (1,507 points). "Photometric" is the
command with its normal step the photometric search, as of PR #1078; "grown
plane" is the command with the switch of PR #1079. Each output point is matched to
the nearest finite point of the ground truth
(`seoul_bull_sculpture_ground_truth_v01.sfmr`, `kerry_park_ground_truth_v01.sfmr`)
within that point's patch half-extent, and the error is the angle between the
two stored normals. "Start" is the mean-viewing frame `to_embedded_patches`
begins from, unrefined. The bar columns are the shares of non-reference
observations whose stored reading against the bitmap is at or above `0.70`
(the bench's `BENCH_MIN_ZNCC`), blur-matched and plain. Times are the command's
wall time, one run each, in one optimised build on one machine.

| Dataset | Normal step | Points | Observations | Matched | Median error | Mean error | Within 5° | Within 10° | Over 30° | Blur-matched ≥ 0.70 | Plain ≥ 0.70 | Time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Seoul bull | start | 780 | 2,700 | 443 | 29.6° | 31.0° | 5.0% | 13.5% | 49.2% | | | |
| Seoul bull | photometric | 734 | 3,651 | 423 | 19.8° | 22.4° | 6.6% | 19.9% | 25.1% | 97.2% | 96.9% | 1.9 s |
| Seoul bull | grown plane | 720 | 3,622 | 416 | 18.4° | 23.8° | 8.4% | 25.0% | 27.9% | 96.1% | 95.9% | 6.1 s |
| Kerry Park | start | 1,507 | 6,339 | 298 | 32.9° | 32.8° | 10.7% | 14.8% | 54.4% | | | |
| Kerry Park | photometric | 1,474 | 18,159 | 289 | 25.5° | 27.9° | 11.8% | 22.5% | 42.6% | 94.0% | 93.1% | 10.5 s |
| Kerry Park | grown plane | 1,454 | 18,066 | 293 | 22.6° | 30.2° | 18.4% | 32.1% | 41.6% | 93.0% | 92.0% | 47.9 s |

On Kerry Park the grown plane turned 1,024 of 1,507 normals in round 1 (mean
agreement 0.84) and 1,175 of 1,499 in round 2 (0.77, 44 refused); its two
normal steps took 11.2 s and 26.7 s, where the photometric ones took 0.4 s and
2.7 s.

**Reading.** On both datasets the grown plane gives the lower median error and
puts more points within 5° and within 10° of the ground truth: 8.4% and 25.0%
against 6.6% and 19.9% on the bull, 18.4% and 32.1% against 11.8% and 22.5% on
Kerry Park. Its mean error is higher, by 1.4° and 2.3°, because more of its
errors are large: on the bull 27.9% of the matched points are over 30° against
25.1%. It keeps 1 to 2% fewer points, and about one percentage point fewer
observations clear the 0.70 bar. The run takes three to five times as long,
nearly all of it in the normal step, and most of that in round 2.

On the 780-point seoul_bull test fixture with the self-similarity cull off, the
points round 1 aligns to no reference grow from 73 to 98 with the grown plane:
each is a point the localizer kept one view of, and a point with fewer than two
views has no reference, so more points are left with one view at the turned
normals; the final `min_views` cull drops them. For that reason
`test_embed_patches_rounds_keep_the_references_without_bitmaps` checks that
every reference the views were aligned to is recorded, rather than a share of
points with a reference over 0.9 (0.906 with the photometric search, 0.874 with
the grown plane).

## Open question

- **The normal step in every round, or once.** Round 2's normal step is the
  largest cost of the command (26.7 s of 47.9 s on Kerry Park). Measure, on the
  same inputs, the command with the normal step in round 1 only and in the last
  round only, against every round, by the same normal errors and bars, and drop
  a round's normal step where the output does not get worse.
