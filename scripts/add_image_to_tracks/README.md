# Add Image to Tracks: evaluation harness

A leave-one-image-out harness for the core operation
([specs/core/reconstruction/add-image-to-tracks.md](../../specs/core/reconstruction/add-image-to-tracks.md)):
how many of the tracks an image was in it rejoins, how close the rejoined
keypoints land to the originals, and how good the tracks it joins that it was
not in are. It is what chose the operation's default rule, and what any change
to the rule is measured with.

- `datasets.py`: the ground truths and their cluster-patches files. seoul_bull
  is copied into a cache directory and its index files built there, through
  the track-at-pixel harness's own preparation; kerry_park is read in place.
- `strategies.py`: the rules and measurement settings compared, as keyword
  sets for `EditedReconstruction.add_image_to_tracks`.
- `harness.py`: the sweep. For every image it removes the image's
  observations (and the points left with fewer than two, which are counted as
  lost), resects the image back in with `geometry.resect_images` over the
  tracks and the cluster-patches file, and runs every strategy at the resected
  pose and at the ground-truth pose, so pose error and matching error can be
  told apart. It measures recall on the tracks the image was in, each rejoined
  keypoint's distance from the original, and for the tracks it joins that it
  was not in ("extra"), their ZNCC, their distance from the projection and the
  new observation's residual after the point is retriangulated with it.
- `summarize.py`: the per-dataset, per-pose, per-strategy tables.
- `misses.py`: every known track a rule refused, with the numbers behind the
  refusal and the original observation's own distance from its projection.

```bash
pixi run -e test python scripts/add_image_to_tracks/harness.py     --dataset seoul_bull --cache <dir> --out <dir> [--images 0,3] [--strategies default]
pixi run -e test python scripts/add_image_to_tracks/summarize.py <dir>/seoul_bull.jsonl
pixi run -e test python scripts/add_image_to_tracks/misses.py     --dataset kerry_park --cache <dir> --out <dir>
```

The results below were recorded on 2026-09-25. All figures are summed over
every image of the dataset; measurement: rendered template, sub-pixel step,
localizability gate on. Until § "Why recall is not 100%", "the default" means the
first default, the pooled bar alone with the image-MAD positional gate; that
section's change made the current one.

Datasets:

- **seoul_bull** (checked-in ground truth, 17 images, 280 points, one
  SIMPLE_RADIAL camera): copied with its images into a cache directory, where
  `.sift` files and the index files are built by the track-at-pixel harness's
  own preparation. 1235 known observations to recover; 42 points lost on
  removal. Every resection accepted, rotation error median 0.107°, max 0.219°;
  centre error median 0.0019, max 0.0039 of the scene scale.
- **kerry_park** candidate `tk106` (48 images, 384 points, two OPENCV_FISHEYE
  cameras): read in place, with the cluster-patches file built for `tk105`,
  which names the same images (the resection matches images by name). 2225
  known observations; 64 points lost on removal. Every resection accepted. Over
  the 45 images that have observations in the ground truth: rotation error
  median 0.114°, p90 0.300°, max 1.83° (image 37, `fisheye_right/frame_14`);
  centre error median 0.0033, max 0.041 of the scene scale. Images 21 to 23
  (`fisheye_left/frame_22` to `frame_24`) have **no observations in the ground
  truth**, so their stored poses are not registrations; the resection moves
  them by 8 to 10° and about 19 scene scales, and the operation then adds 12 to
  19 observations to each at a median retriangulation residual of 0.19 to
  0.32 px, where at the stored pose it adds none. The stored poses of those
  three images look wrong, and resection followed by this operation registers
  them.

All figures are summed over every image of the dataset. Measurement: rendered
template, sub-pixel step, localizability gate on.

## seoul_bull, resected pose

| strategy | recall | err med / p90 px | >1 px | >2 px | extra | extra ZNCC | extra new res med / p90 px | extra res > 2 px |
|---|---|---|---|---|---|---|---|---|
| fixed 0.6 | 97.4% | 0.062 / 0.277 | 57 | 48 | 233 | 0.743 | 0.66 / 1.81 | 20 |
| fixed 0.7 | 94.6% | 0.060 / 0.269 | 56 | 47 | 162 | 0.781 | 0.59 / 1.63 | 11 |
| fixed 0.8 | 86.2% | 0.058 / 0.213 | 46 | 38 | 68 | 0.847 | 0.49 / 1.22 | 2 |
| fixed 0.9 | 55.1% | 0.052 / 0.182 | 23 | 18 | 15 | 0.944 | 0.50 / 0.66 | 0 |
| track min, pair mean ×1.0 | 75.9% | 0.062 / 0.323 | 52 | 43 | 73 | 0.734 | 0.90 / 2.74 | 11 |
| track min, pair min ×1.0 | 74.1% | 0.062 / 0.307 | 49 | 41 | 72 | 0.733 | 0.90 / 2.76 | 11 |
| track min, pair max ×1.0 | 80.2% | 0.062 / 0.316 | 52 | 43 | 73 | 0.734 | 0.90 / 2.74 | 11 |
| track min, pair mean ×0.9 | 85.1% | 0.062 / 0.303 | 54 | 45 | 76 | 0.758 | 0.80 / 2.68 | 11 |
| track med−3·MAD | 89.1% | 0.062 / 0.316 | 58 | 47 | 80 | 0.792 | 0.61 / 2.59 | 10 |
| track 0.9·median | 92.4% | 0.062 / 0.303 | 58 | 47 | 75 | 0.833 | 0.53 / 1.91 | 7 |
| pooled med−2·MAD | 90.8% | 0.059 / 0.248 | 51 | 42 | 106 | 0.821 | 0.55 / 1.52 | 5 |
| pooled med−3·MAD | 95.3% | 0.061 / 0.269 | 56 | 47 | 165 | 0.780 | 0.60 / 1.62 | 11 |
| track min, pair mean ×0.9 + image MAD | 79.8% | 0.058 / 0.200 | 23 | 18 | 44 | 0.801 | 0.54 / 1.10 | 0 |
| track 0.9·median + image MAD | 86.2% | 0.058 / 0.198 | 24 | 18 | 53 | 0.836 | 0.38 / 0.74 | 0 |
| fixed 0.7 + image MAD | 89.1% | 0.057 / 0.193 | 25 | 19 | 117 | 0.792 | 0.42 / 0.92 | 0 |
| **pooled med−3·MAD + image MAD (first default)** | **89.7%** | **0.058 / 0.193** | **25** | **19** | **119** | **0.792** | **0.44 / 0.96** | **0** |
| pooled med−3·MAD + image MAD, k = 5 | 91.4% | 0.058 / 0.200 | 32 | 25 | 129 | 0.789 | 0.48 / 1.10 | 0 |
| pooled med−3·MAD + image MAD, floor 2 px | 92.7% | 0.059 / 0.211 | 34 | 26 | 138 | 0.790 | 0.50 / 1.22 | 0 |
| pooled med−3·MAD + max 2 px | 92.7% | 0.059 / 0.211 | 34 | 26 | 137 | 0.792 | 0.50 / 1.21 | 0 |
| pooled med−3·MAD + max 3 px | 94.0% | 0.060 / 0.222 | 42 | 33 | 152 | 0.781 | 0.55 / 1.49 | 4 |

Rejoined known tracks retriangulate with a new-observation residual median of
0.26 to 0.28 px under every strategy. At the ground-truth pose the default
recovers 89.8% at 0.056 px median, with 13 rejoined keypoints over 2 px and 102
extra tracks: the resected pose costs almost nothing on this capture.

## kerry_park, resected pose

| strategy | recall | err med / p90 px | >1 px | >2 px | extra | extra ZNCC | extra new res med / p90 px | extra res > 2 px |
|---|---|---|---|---|---|---|---|---|
| fixed 0.6 | 96.8% | 0.068 / 0.444 | 94 | 65 | 2417 | 0.857 | 0.36 / 1.45 | 144 |
| fixed 0.7 | 95.6% | 0.067 / 0.436 | 87 | 59 | 2013 | 0.882 | 0.34 / 1.21 | 89 |
| fixed 0.8 | 92.3% | 0.065 / 0.434 | 79 | 53 | 1558 | 0.909 | 0.33 / 1.15 | 58 |
| fixed 0.9 | 76.4% | 0.060 / 0.411 | 60 | 44 | 859 | 0.941 | 0.30 / 0.87 | 19 |
| track min, pair mean ×1.0 | 84.0% | 0.061 / 0.411 | 74 | 53 | 1169 | 0.912 | 0.37 / 1.42 | 63 |
| track min, pair min ×1.0 | 80.5% | 0.060 / 0.397 | 71 | 51 | 1140 | 0.912 | 0.37 / 1.45 | 63 |
| track min, pair max ×1.0 | 85.7% | 0.062 / 0.414 | 78 | 55 | 1195 | 0.912 | 0.37 / 1.40 | 63 |
| track min, pair mean ×0.9 | 90.5% | 0.066 / 0.438 | 83 | 59 | 1258 | 0.912 | 0.36 / 1.35 | 63 |
| track med−3·MAD | 88.2% | 0.065 / 0.438 | 79 | 56 | 1179 | 0.914 | 0.34 / 1.18 | 44 |
| track 0.9·median | 93.0% | 0.065 / 0.433 | 81 | 57 | 1430 | 0.915 | 0.33 / 1.05 | 46 |
| pooled med−2·MAD | 88.4% | 0.063 / 0.417 | 71 | 51 | 1292 | 0.922 | 0.33 / 1.03 | 41 |
| pooled med−3·MAD | 92.4% | 0.065 / 0.434 | 79 | 53 | 1581 | 0.908 | 0.33 / 1.15 | 60 |
| track min, pair mean ×0.9 + image MAD | 86.0% | 0.060 / 0.363 | 29 | 15 | 1040 | 0.919 | 0.29 / 0.78 | 3 |
| track 0.9·median + image MAD | 88.6% | 0.060 / 0.359 | 28 | 14 | 1220 | 0.920 | 0.28 / 0.72 | 2 |
| fixed 0.7 + image MAD | 91.3% | 0.062 / 0.364 | 33 | 15 | 1705 | 0.892 | 0.30 / 0.79 | 6 |
| **pooled med−3·MAD + image MAD (first default)** | **88.0%** | **0.061 / 0.364** | **29** | **13** | **1356** | **0.913** | **0.29 / 0.74** | **2** |
| pooled med−3·MAD + image MAD, k = 5 | 89.8% | 0.062 / 0.378 | 41 | 18 | 1444 | 0.911 | 0.30 / 0.83 | 7 |
| pooled med−3·MAD + image MAD, floor 2 px | 90.1% | 0.063 / 0.385 | 47 | 22 | 1451 | 0.910 | 0.30 / 0.80 | 2 |
| pooled med−3·MAD + max 2 px | 89.8% | 0.063 / 0.385 | 47 | 22 | 1440 | 0.910 | 0.30 / 0.79 | 2 |
| pooled med−3·MAD + max 3 px | 90.8% | 0.063 / 0.390 | 51 | 25 | 1510 | 0.909 | 0.32 / 0.90 | 13 |

Rejoined known tracks retriangulate with a new-observation residual median of
0.21 to 0.24 px. At the ground-truth pose the default recovers 89.1% at
0.060 px median, with 9 rejoined keypoints over 2 px and 1280 extra tracks.

## Measurement settings

Against the first default rule at the resected pose:

- **Sub-pixel step.** Without it the rejoined keypoints' median error rises from
  0.058 to 0.088 px on seoul_bull and from 0.061 to 0.074 px on kerry_park, and
  the counts move by a few. It stays on.
- **Localizability gate.** It refuses 0 to 29 candidates per kerry_park image
  and almost none on seoul_bull. Turning it off adds 3 rejoined and 10 extra
  observations on kerry_park and 1 extra on seoul_bull, with no change in the
  far or bad counts: what it refuses mostly fails the photometric rule anyway.
  It stays on at the localizer's `τ = 0.35`.
- **Stored bitmap as the template** (kerry_park only; the seoul_bull ground
  truth is a minimal file with no bitmaps): recall 86.9% against 88.0% rendered,
  10 rejoined keypoints over 2 px against 13, 1247 extra tracks against 1356.
  Slightly lower ZNCC throughout, as a bitmap quantised to bytes and fused with
  its own weights is a different reference. The rendered consensus stays the
  default; the references have to be rendered for their leave-one-out ZNCCs in
  any case, so the bitmap saves no rendering.

## Readings

- **The positional gate is what separates the strategies.** Every photometric
  rule alone leaves 3 to 4% of rejoined keypoints more than 2 px from the
  original and several percent of extra tracks whose new observation does not
  agree with the retriangulated point. The image-MAD bound halves or better the
  first (seoul 47 → 19, kerry 53 → 13) and all but removes the second (seoul
  11 → 0, kerry 60 → 2), at a cost of 4 to 6 points of recall. Its bound came
  out at a median of 1.2 px on both captures (1.0 to 2.5 px per image).
- **The per-track leave-one-out basis, as first proposed, costs recall and buys
  no precision.** With the minimum of the references' leave-one-out ZNCCs as the
  bar, and the pair rule at factor 1, recall is 76% on seoul_bull and 84% on
  kerry_park, and the extra tracks it accepts are no better than a fixed bar's.
  The references' keypoints were fitted together and the new view's was not,
  so the new view's ZNCC runs lower than theirs for the same quality of match,
  and the minimum of several numbers is a demanding bar. A softer per-track
  statistic (0.9 × median) recovers most of the recall. Within the pair rule,
  `max` beats `mean` beats `min` on recall with the same precision, and a factor
  of 0.9 gains 6 to 9 points of recall over 1.0 at no cost in precision.
- **The pooled basis follows the capture.** The median−3·MAD bar over every
  candidate's references came out at 0.64 to 0.72 on seoul_bull and 0.75 to
  0.85 on kerry_park; a fixed bar right for one is wrong for the other (fixed
  0.8 recovers 86% on seoul_bull against the pooled bar's 95%; fixed 0.7 admits
  89 bad extra observations on kerry_park against its 60). Under the positional gate the pooled bar gives
  the best recall of the data-driven rules on seoul_bull and matches the
  per-track 0.9 × median on kerry_park.
- **What remains.** Of the 19 (seoul) and 13 (kerry) rejoined keypoints over
  2 px under the default, 13 and 10 sit where the ground truth's own keypoint
  is more than 2 px from the point's projection: the ground truth disagrees with
  its own point there, and the operation found the patch where the point says
  it is. The known observations the default refuses are mostly `too_far` (69
  seoul, 96 kerry) and `below_bar` (40, 104), then `peak_at_edge` and, on
  kerry_park, `shared_keypoint` (24).
- **One observation per place bites on kerry_park.** Across its 96 default
  calls, 290 candidates were refused as `shared_keypoint`: the candidate
  ground truth holds pairs of points at one place in one image. On seoul_bull
  it never fires.
- **Cost.** One call takes 8 ms on seoul_bull (280 candidates) and 16 ms on
  kerry_park (384 candidates), measuring every candidate in parallel.

## The first default and why

`PooledBasis { MedianMinusMad { k: 3 } }` with `ImageMad { k: 3, floor_px: 1 }`
and a 0.5 ZNCC floor. Both bars are read off the call's own data, so they
follow the capture's texture and the pose's error rather than a constant. It
has the fewest bad extra observations of any rule that keeps recall near 90%
on both captures (0 of 119 and 2 of 1356), and it halves the far rejoined
keypoints against every rule without a positional gate. A caller that wants
more recall at some cost in precision widens the positional bound
(`position_floor_px = 2`: +3 points of recall on seoul_bull and +2 on
kerry_park, for 7 and 9 more rejoined keypoints over 2 px).

## Why recall is not 100%

Under the first default at the resected pose, 127 of 1235 known observations
were refused on seoul_bull and 266 of 2225 on kerry_park (`misses.py`). By
reason, with the medians of the numbers behind each:

| reason | seoul | kerry | what the numbers say |
|---|---|---|---|
| `too_far` | 69 | 96 | offset 1.9 / 2.2 px against a bound of 1.1 px. Half land on the original pixel (within 0.5 px of it), where the original itself sits 1.8 to 2.0 px from its own projection, beyond the 95th percentile of the ground truth's own reprojection residuals (1.31 / 0.83 px): a sighting the ground truth holds at a large residual. The other half land 4.3 px from the original, a different place: correctly refused. |
| `below_bar` | 40 | 104 | ZNCC 0.62 / 0.74 against a pooled bar of 0.68 / 0.80. 35 of 40 and 85 of 104 land on the original pixel, and their tracks' own references agree less well with each other (median of the minimum leave-one-out ZNCC 0.60 / 0.71): good sightings on hard surfaces, refused by a bar set by the rest of the image. |
| `peak_at_edge` | 13 | 31 | the window's highest correlation is on its edge, mostly where the original is within 1 px of the projection: a repeated texture a period away. |
| `shared_keypoint` | 0 | 24 | the kerry_park candidate holds 13 pairs of points at one pixel of one image (28 observation pairs under 1 px apart): duplicate tracks in the ground truth, one of which is refused. |
| `below_floor` | 3 | 7 | ZNCC near 0 at a keypoint 5 px from the original: the search found another place. |
| other | 2 | 4 | `too_few_references`, `unlocalizable`, `unscorable`. |

Two data-derived changes were tried for the two classes that refuse good
sightings:

| strategy (resected pose) | seoul recall | >2 px | extra / bad | kerry recall | >2 px | extra / bad |
|---|---|---|---|---|---|---|
| first default (pooled bar + image MAD) | 89.7% | 19 | 119 / 0 | 88.0% | 13 | 1356 / 2 |
| + ascent on an edge peak | 89.8% | 20 | 119 / 0 | 88.9% | 14 | 1435 / 3 |
| **pooled bar or the track's own bar (0.9 × median; the current default)** | **91.2%** | **20** | **125 / 0** | **89.9%** | **14** | **1419 / 2** |
| the same + ascent | 91.4% | 21 | 126 / 0 | 90.4% | 15 | 1488 / 3 |
| the same with the track's minimum as its bar, + ascent | 91.5% | 21 | 133 / 0 | 91.4% | 18 | 1575 / 5 |

At the ground-truth pose the current default recovers 91.3% (seoul_bull) and
90.8% (kerry_park). Accepting what either bar accepts halves the `below_bar`
misses (40 to 19, 104 to 62) at no cost in bad extra observations, and is the
default. The ascent recovers a few `peak_at_edge` sightings on kerry_park and
none on seoul_bull, and adds a bad extra observation, so it stays an option,
off. The `too_far` class is left alone: loosening the positional bound (a 2 px
floor, above) recovers some of it and lets in as many far keypoints.
