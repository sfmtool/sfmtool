# Track at a pixel: evaluation harness

A development harness for the core operation behind bench track editing in SfM
Explorer: **given a reconstruction, an image and a pixel, build a high-quality
track centred at (or very near) that pixel, or explain why none could be
built.** Candidates are Python compositions of the bench steps and the patch
kernels. Once one wins here, it moves into `sfmtool-core`. The operation's
contract and its open questions are in
[`specs/drafts/track-at-pixel.md`](../../specs/drafts/track-at-pixel.md).

## How the evaluation works

1. Take a ground-truth reconstruction (today `seoul_bull`, meaning
   `test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr`).
2. Remove one point from it. It is deleted from the `EditedReconstruction` the
   candidate is given (index-stable) and filtered out of every neighbourhood
   query.
3. For each image that point was observed in, call the candidate at that
   observation's pixel.
4. Score the returned track against the removed one (`metrics.py`), or record
   the refusal's stage and reason.

Every query runs in two passes, which stand for two stages of building a
reconstruction:

- **`full`**: only the point under test is removed. The candidate can lean on
  the reconstructed points around the pixel for depth, normal and size. This
  is a reconstruction that already has many tracks and is being filled in.
- **`empty`**: every point is removed. The candidate has the cameras, the
  photographs, the descriptor index, the `.sift` keypoints and the
  cluster-patches clusters, and no reconstructed point:
  `observations_near`, `points_near` and `scene_depths` return nothing, and
  `edited` holds no points. This is a reconstruction early on, when a track has
  to be built from scratch.

Both passes are scored against the same ground truth. `--passes full` or
`--passes empty` runs one of them.

```bash
pixi run -e test python scripts/track_at_pixel/harness.py                       # every point
pixi run -e test python scripts/track_at_pixel/harness.py --points 20 --seed 1   # a sample
pixi run -e test python scripts/track_at_pixel/harness.py --point-ids 144,177 --raise
pixi run -e test python scripts/track_at_pixel/harness.py --opt max_query_offset_px=3 --opt normal_prior=false
```

`--dataset` also takes the path of any ground-truth `.sfmr` whose directory
holds its images and `.sfm-workspace.json`. A whole dataset runs faster in
parallel shards, one harness process per shard, merged into one run directory
that `compare.py` reads like any other. The shards' track files are merged too,
into one `tracks-full.sfmr` and one `tracks-empty.sfmr`, with each row's
`output_point` renumbered to match:

```bash
pixi run -e test python scripts/track_at_pixel/run_sharded.py --candidate renormal --shards 14 --out <run>
```

Each shard runs its patch kernels on one thread (`--threads`). A kernel thread
pool per process, sized to the machine, oversubscribes it many times over when
a dozen shards run at once, and made every query about five times slower.

The first run copies the dataset into a cache workspace
(`%TEMP%/sfmtool-track-at-pixel/<dataset>`, or `--cache-dir`), extracts SIFT
there with the marker's settings, and builds a `.kdf` over it in the
reconstruction's image order. It then clusters the features from that same
`.kdf` (`background_floor_clusters_kdf`, `sfm match --cluster`'s clustering)
and refines the clusters with `sfm cluster-patches`' defaults. The result is
`track_at_pixel_clusters-patches.matches`, which every candidate is handed.
Pass `--matches <file>` to hand candidates a different cluster-patches file
instead; its images are matched to the reconstruction's by name. Nothing is
written beside `test-data`. Each run
writes these files to `<cache>/runs/<candidate>-<time>/`, or to `--out`:

- `rows.jsonl`: one row per query and pass, with the candidate's diagnostics.
  A row's `pass` names its pass.
- `summary.txt`, with a section per pass, and `config.json`.
- `tracks-full.sfmr` and `tracks-empty.sfmr`: every track returned in that
  pass, committed into the ground truth's cameras with none of its points.
  The files store no patch bitmaps, so a track a candidate returns without one
  (any patch step after its last fit drops it) can still be written beside a
  ground truth that stores one per point. A sighting a fit left a fraction of
  a pixel outside its photograph is turned out in the written copy, because a
  `.sfmr` with such a keypoint does not load; the row is scored on the track
  as returned.

A row's `output_point` is its track's index in its pass's file. There is one
point per successful query, so a ground-truth point queried from five images
can appear up to five times. To compare the run with the ground truth, load
them into one Explorer window and toggle between them:

```bash
pixi run gui -- test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr <run>/tracks-full.sfmr <run>/tracks-empty.sfmr
```

## Results on seoul_bull

Every point of the ground truth seen in two or more images, queried from each
image it is seen in (1277 queries, 44 of them on the 14 points at infinity),
judged against the good-track bar. The ground-truth tracks themselves pass
that bar on 1230 queries: 47 of them miss it, mostly on the median ZNCC. That
is the number a candidate is measured against.

**Full pass** (only the point under test is removed):

| Candidate | Built | Good | Good at infinity | Not good | Good % | Median angle err (deg) | Median normal err (deg) | Median projection offset at the pixel (px) | s/query |
|---|---|---|---|---|---|---|---|---|---|
| ground truth | 1277 | 1230 | 44 | 47 | 96.3% | | | | |
| `baseline` | 289 | 254 | 11 | 35 | 19.9% | 0.102 | 13.3 | 0.67 | 0.23 |
| `baseline --opt finish=common` | 337 | 308 | 7 | 29 | 24.1% | 0.019 | 12.6 | 0.19 | 0.53 |
| `transfer` | 667 | 592 | 0 | 75 | 46.4% | 0.025 | 11.9 | 0.24 | 0.31 |
| `clusters` | 721 | 674 | 19 | 47 | 52.8% | 0.028 | 13.5 | 0.24 | 0.30 |
| `sweep` | 833 | 739 | 0 | 94 | 57.9% | 0.026 | 10.2 | 0.26 | 0.30 |
| `planesweep` | 786 | 697 | 11 | 89 | 54.6% | 0.021 | 13.4 | 0.24 | 1.11 |
| `cascade` | 1025 | 925 | 20 | 100 | 72.4% | 0.030 | 12.7 | 0.26 | 0.51 |
| `core_cascade` | 1025 | 925 | 20 | 100 | 72.4% | 0.030 | 12.7 | 0.26 | 0.37 |
| `ensemble` | 1240 | 1089 | 34 | 151 | 85.3% | 0.032 | 13.4 | 0.26 | 7.66 |
| `centred` | 1240 | 1100 | 34 | 140 | 86.1% | 0.031 | 13.4 | 0.25 | 8.02 |

**Empty pass** (every point is removed):

| Candidate | Built | Good | Good at infinity | Not good | Good % | Median angle err (deg) | Median normal err (deg) | Median projection offset at the pixel (px) | s/query |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 240 | 213 | 11 | 27 | 16.7% | 0.084 | 29.4 | 0.53 | 0.14 |
| `baseline --opt finish=common` | 264 | 243 | 7 | 21 | 19.0% | 0.020 | 29.9 | 0.20 | 0.29 |
| `transfer` | 0 | 0 | 0 | 0 | 0.0% | | | | |
| `clusters` | 694 | 647 | 19 | 47 | 50.7% | 0.029 | 30.2 | 0.24 | 0.20 |
| `sweep` | 0 | 0 | 0 | 0 | 0.0% | | | | |
| `planesweep` | 623 | 529 | 13 | 94 | 41.4% | 0.030 | 33.1 | 0.23 | 0.92 |
| `cascade` | 741 | 688 | 20 | 53 | 53.9% | 0.029 | 30.3 | 0.24 | 0.42 |
| `core_cascade` | 741 | 688 | 20 | 53 | 53.9% | 0.029 | 30.3 | 0.24 | 0.26 |
| `ensemble` | 1177 | 1003 | 36 | 174 | 78.5% | 0.035 | 34.3 | 0.24 | 3.96 |
| `centred` | 1177 | 1009 | 36 | 168 | 79.0% | 0.034 | 34.2 | 0.24 | 4.55 |

The medians are over the built tracks. Times are the median over the pass's
queries with 16 processes sharing the machine; for times measured with each
candidate alone, time it on its own with `--points`. The second candidate row
is the control: the baseline's own way of finding sightings, with the shared
finish. It shows how much of the gain is the finish and how much is where the
sightings come from. Anchoring puts the queried keypoint on the pixel by
construction, so the centring number to read is the point's projection offset,
not `query_keypoint_offset_px`.

In the full pass the cascade's tracks came from `clusters` (721), `transfer`
(183), `sweep` (110) and the descriptor route (11). `sweep` and `transfer`
build from finite neighbours only, so they return nothing at infinity. From the
cascade on, each candidate is measured by how much of the gap to the ground
truth's 1230 good queries it closes. The cascade leaves 305. The ensemble
leaves 141, 54% of the cascade's gap; its tracks came from `clusters` (876),
`transfer` (191), `sweep` (124), `planesweep` (48) and the descriptor route
(1). `centred` leaves 130, 8% of the ensemble's gap.

`core_cascade` is the cascade as `sfmtool_core::bench::build_track_at_pixel`
runs it ([`specs/core/bench/track-at-pixel.md`](../../specs/core/bench/track-at-pixel.md)).
Against `cascade` it has the same outcome on every query of both passes, from
the same member, with the same refusal stages. The tracks agree to about `1e-15` except
one, point 83 from image 8, whose median ZNCC differs by `1.3e-4`: a last-bit
change in the sweep's surface hypothesis decides which side of a discrete step
in the fit the track lands on, and perturbing the Python sweep's hypothesis by
one part in `1e15` moves it between the same two results. Each run alone on an
idle machine, over the `--points 15 --seed 3` sample, twice each, the Rust
cascade's median was 0.101 and 0.117 s a query against the Python cascade's
0.123 and 0.117 s: the time is in the bench kernels both call, not in the
composition around them.

The empty pass separates what finds sightings from the photographs and the
descriptors from what needs reconstructed neighbours. `sweep` and `transfer`
have nothing to start from and refuse every query. `clusters` loses little
(674 to 647): its sightings come from the cluster file, which holds nothing of
the reconstruction, and it only loses the neighbours' normal. The cascade falls
from 925 to 688, because two of its members are gone. The ensemble falls
least, to 1003, because `planesweep` needs only the poses and the photographs;
it supplies 279 of the ensemble's tracks in the empty pass against 48 in the
full one. Positions stay as accurate as in the full pass. Normals do not:
without the neighbours' normal to tilt toward, the median normal error rises
from about 13 degrees to about 30.

## Where the remaining gap is

These measurements were made on the full pass before the plane sweep took
its depth range from the points inside the queried photograph only, when
`centred` got 1107 good queries and left 123. They say where the remaining gap
is.

**The sightings are not the limit; the framing is.** A diagnostic built
tracks from the held-out point's own keypoints, the true sightings, with the
members' construction (size from the neighbours, the neighbours' normal, the
upgrade, the fit and the shared finish). A diagnostic reads the held-out
point, so it is not a candidate and is not in `candidates/`.

| Built from the true sightings with | Good |
|---|---|
| the members' size and normal | 1090 |
| the ground truth's size | 1110 |
| the ground truth's normal | 1134 |
| the ground truth's size and normal | 1156 |
| the ground-truth track itself, pinned, through the finish | 1175 |

`centred` gets 1107, more than the construction gets from the true sightings.
The normal decides where the keypoints settle. The neighbours' mean normal is
11 degrees from the ground truth's at the median. A fit through the directions
to the image-space neighbours is 27 degrees off. The photometric normal
refinement is 17 degrees off, and it moves even the ground truth's own normal
by 16 degrees.

**The misses are coherent.** The tracks that miss the bar are mostly 2 or 3
views, and all their views are off by a similar amount: 3.6 px from the
ground-truth projection at the median, against the bar's 3 px. Their median
ZNCC is within 0.012 of the ground truth's. At 40 of the 123, the ground
truth's own point projects 1.5 px or more from its keypoint at the queried
pixel. A track that holds that pixel then disagrees with the ground truth in
every other view.

**What was tried and not kept** (good queries, against `centred`'s 1107 unless
noted):

| Idea | Result |
|---|---|
| A cross-validated logistic selector over 13 or 21 member runs | 1115 to 1122 |
| Adding the photometric normal search to every member | 1107 |
| A plane through the congealed depths of a 3x3 grid of subpatches tiling the patch, as the normal | 1092 (+8 to +13 from the true sightings) |
| One unanchored refit of the chosen track, kept if the queried keypoint moves under 2 px | 1098 |
| No anchoring in any fit | 944 |
| Growing the chosen track into every view that should see it | +3 net, on the ensemble's not-good points |
| A fine photometric depth search around the chosen depth | moved 113 correct tracks off the truth |
| The neighbours' plane as a depth prior in the vote | 1063 to 1103 |

## Normals, gates and fallbacks on two ground truths

The second ground truth is a Kerry Park candidate, `tk113`: 48 fisheye images
from a two-lens rig, 370 points (12 at infinity), 3903 queries per pass. The
harness reads it through `--dataset <path to the .sfmr>`. The candidate
`renormal` was chosen on both ground truths together, scored in a way that
weights the normal more than the good-track count alone:

- `G`: good tracks per query (the good-track bar).
- `N`: over the good tracks, `max(0, 1 − normal error / 30°)` summed and
  divided by the queries; a good track at infinity counts 1.
- `S = (G + 2N) / 3`, so a perfect run scores 1. The two datasets' two passes
  are averaged with equal weight.

`goal_score.py` computes these for any run. `ground_truth_ceiling.py` scores
the ground-truth tracks themselves as if a candidate had returned them: they
meet the good-track bar on 96.3% of seoul_bull's queries and 98.4% of Kerry
Park's, so almost all of the distance to 1 is open to a candidate.

**Wide-angle lenses.** `Camera.project` in `context.py` returned nothing for
a point more than 90 degrees off the lens axis. A fisheye sees past that: 241
of Kerry Park's 3903 ground-truth observations lie between 90 and 107 degrees
off axis. The metrics counted every sighting there as wrong, so a three-view
track with one of them failed `view_precision` although its keypoints were
within half a pixel of the ground truth's, and the Python candidates could not
project into those views. A wide-angle lens now projects any ray through its
model. The Kerry Park numbers below are measured after that change.

`core_cascade` is the operation as it ships. Times are the median over the
pass's queries with every shard on one kernel thread; they compare only
within a dataset.

| Run | Pass | Good | G | N | S | Median normal err (deg) | Precision (good ÷ built) | s/query |
|---|---|---|---|---|---|---|---|---|
| seoul_bull `core_cascade` | full | 933 | 0.731 | 0.409 | 0.516 | 12.1 | 0.90 | 0.12 |
| seoul_bull `renormal` | full | 1044 | 0.818 | 0.559 | 0.645 | 7.0 | 0.86 | 0.12 |
| seoul_bull `core_cascade` | empty | 690 | 0.540 | 0.129 | 0.266 | 30.3 | 0.93 | 0.09 |
| seoul_bull `renormal` | empty | 1011 | 0.792 | 0.261 | 0.438 | 22.9 | 0.87 | 0.12 |
| Kerry Park `core_cascade` | full | 2699 | 0.692 | 0.489 | 0.557 | 3.6 | 0.86 | 0.37 |
| Kerry Park `renormal` | full | 3070 | 0.787 | 0.614 | 0.672 | 1.9 | 0.83 | 0.38 |
| Kerry Park `core_cascade` | empty | 1867 | 0.478 | 0.170 | 0.272 | 32.3 | 0.86 | 0.27 |
| Kerry Park `renormal` | empty | 2793 | 0.716 | 0.239 | 0.398 | 28.8 | 0.78 | 0.31 |

The mean `S` goes from 0.403 to 0.539, 23% of the way from `core_cascade` to
a perfect score. Each dataset's two candidates ran side by side under the same
load. The median time is the same in both full passes, and 35% (seoul_bull)
and 15% (Kerry Park) longer in the empty passes. The step that re-orients the
track costs 10 to 17 ms at the median; the rest is the Python fallbacks, which
run only on the queries the cascade refuses and take 0.5 to 1 s each.

**Why not further.** The empty pass holds the score down. Its good tracks
earn about a third of the normal credit the full pass's do, because with no
reconstructed neighbours the only normal source is photometry, and photometry
does not agree with these ground truths' normals to better than about 20
degrees at the median (see "Photometric normals"). If the empty pass's
normals earned the full pass's share of the credit, the mean `S` would be
about 0.64.

That also bounds what this route can reach. The best empty-pass normal found
here, photometric at the ground truth's own position and views
(`normal_sources/photo_bias.py`, each point weighted by its track length as the
harness weights queries), earns a normal credit of 0.34 on seoul_bull and 0.32
on Kerry Park. A candidate with every query good and that normal would score
about 0.54 in each empty pass, and with the full passes' current normals the
mean `S` would be about 0.68. Half the gap from `core_cascade` is 0.70. Past
that needs an empty-pass normal source better than photometry at the true
position, and none of those tried below is.

**Where the gain comes from**, in the order it was found on seoul_bull (mean `S`):

| Step | Mean S |
|---|---|
| `core_cascade` | 0.391 |
| the track re-oriented to its neighbours' normal (20 px, then 40 px), or else the photometric normal | 0.426 |
| gates at two views and a median ZNCC of 0.7, in the cascade and after the re-orientation | 0.477 |
| `planesweep` when the cascade refuses | 0.510 |
| the photometric search over 45 degrees from the mean viewing direction, with a 0.05 fronto-parallel prior | 0.514 |
| `clusters` at 1.5 and 2 times, then `planesweep` at 1 and 1.5 times, when the cascade refuses | 0.524 |
| the photometric normal measured on a patch twice the track's size | 0.537 |
| the tight-to-wide neighbour chain, then the 3D neighbours' normal | 0.542 |

**The gates are a trade.** Two views and a ZNCC of 0.7 return many more good
tracks, and more wrong ones: precision falls by 3 to 8 points. With the
finish's own gates (three views, 0.8) everywhere (`--opt min_in_views=3
--opt min_zncc_median=0.8 --opt core_options={}`, and the fallbacks' gates to
match), `renormal` scores a mean `S` of 0.500 rather than 0.539, with
precision near the cascade's: 0.89 and 0.89 on seoul_bull, 0.84 and 0.77 on
Kerry Park, in the full and empty passes.

**Neighbours' normals.** Measured alone (`nb_sweep`, every query of every
point, against the ground-truth normal), a tight neighbourhood on the pixel's
own surface is much better than a wide one, and covers fewer pixels. On
seoul_bull the neighbours within 10 px and 5% of the depth are 6.6 degrees
off at the median, against 11.1 for the shipped 40 px and 15%, but only 23% of
queries have one. So the neighbours are read as a chain from tight to wide
(10 px within 5% of the depth, 20 px within 10%, 30 px within 15%, 40 px
within 30%), and the first neighbourhood that holds a neighbour gives the
principal axis of their normals, weighted `1/(1+d)^2`. Over all queries, with
a pixel that has no estimate counted as zero, the chain scores 0.644 on
seoul_bull and 0.693 on Kerry Park, against 0.538 and 0.553 for the shipped
average. The 3D neighbours' mean normal covers the pixels with none.
Planes fitted through the neighbours' positions, in 2D or 3D, are 20 to 45
degrees off, much worse than their normals.

On Kerry Park the neighbours' normals match the ground truth's to about a
degree at the median, and exactly for a quarter of the queries. The ground
truth was probably curated on the bench, whose finish tilts every track toward
its neighbours' normal, which would make its normals partly copies of one
another. Numbers that lean on neighbours' normals there are likely optimistic.

**Photometric normals.** `PatchCloud.refine_normals` is the only normal
source in the empty pass. At the ground-truth position and views, from the
mean viewing direction, it is 19 degrees off at the median on seoul_bull and
27 on Kerry Park. Widening the search to 45 degrees with a 0.05 fronto-parallel
prior brings that to 18 and 16. A patch twice the track's size helps in the
pipeline, and three times helps seoul_bull more but costs Kerry Park. The
floor is the ground truth itself. Started from the ground truth's own normal,
the refinement moves 16 degrees on seoul_bull and 8 on Kerry Park, and on
some Kerry Park points it settles 22 degrees off whatever it starts from,
with the photoconsistency almost flat. Where there are neighbours,
photometry loses to them: started from the neighbours' normal over a narrow
range, it raised the full-pass median error from 7.6 to 17.4 degrees on
seoul_bull.

**What was tried and not kept** (mean `S`, both passes unless noted):

| Idea | Result |
|---|---|
| The normal from the cluster-patches members' affine shapes (`S_b S_a⁻¹` fitted by a plane through the point) | 24 degrees off on seoul_bull, against 19 for photometry; few points have a matching cluster |
| Pseudo-neighbours in the empty pass: nearby clusters triangulated, each given a photometric normal, averaged | 28 to 32 degrees off, worse than the point's own photometric normal |
| Averaging photometric normals over 1, 2 and 3 times the size; two photometric passes | no better than 2 times alone |
| The photometric normal over every view that sees the point rather than the track's own | worse on seoul_bull (occlusion) |
| The neighbours' normal inside the Rust finish, before the geometry search (20 px, k = 5, always kept), or no normal prior there | within ±0.003; without it, −0.007 |
| The geometry search again after the re-orientation | no change |
| A plane-sweep challenger when the cascade's track has under five views, the better-scoring kept | +0.014 on seoul_bull at twice the time, −0.004 on Kerry Park |
| More clusters (6 within 24 px), fewer constellation inliers (4), more lateral searches, larger cluster patches | within ±0.01, or worse |
| Two anchored refits; cleaning at 1.0 px; a 1.0 px projection gate | within ±0.003 |
| The depth modes' gap at 1.08 or 1.3; the sweep's prior over 20 px and five points; the transfer over 30 px | within ±0.005 |
| Pseudo-neighbours again, with the photometric settings `renormal` uses (45 degrees, the fronto-parallel prior, twice the size) | 24 to 35 degrees off, against 23 to 24 for the point's own estimate |
| An oracle for pseudo-neighbours (`photo_neighbour_oracle.py`): the tight-to-wide chain over the true neighbours, each with its photometric normal at its true position and views | no better than the point's own photometric normal: credit 0.319 and 0.304 against 0.344 and 0.318 (0.342 and 0.293 with the point's own added). The same neighbours' ground-truth normals score 0.667 and 0.840. Photometry's error is shared across a surface, so no pseudo-neighbours, however well built, would lift the empty pass |
| The photometric normal shrunk toward the mean viewing direction, or pushed further from it; the fronto-parallel prior off | worse in every case. Photometry tilts less than the ground truth (19 and 27 degrees from the mean viewing direction, against 34 and 39) but its error is in the direction of the tilt, not its size: with the prior off it tilts as far as the ground truth and is still 25 degrees off |
| Snapping the photometric normal to the vertical (the cameras' mean up direction) when within 10 to 25 degrees of it, and into the horizontal plane when within that of it | Kerry Park's normal credit 0.321 to 0.375 at 25 degrees (39% of its ground-truth normals are within 10 degrees of vertical and 40% of horizontal); none on seoul_bull. A prior about city scenes, worth about 0.01 of the mean `S`, not kept |
| A depth probe for tracks under six views: the patch moved along the ray to 0.7 to 1.45 times its distance, grown and refit there, the best-scoring kept | −0.022 and −0.028 (seoul_bull, the Kerry Park sample); −0.006 and −0.015 when a probe must score 25% more. The views a wrong depth gathers outscore the right depth's |

**A cheap vote.** `core_vote` runs each Rust member alone and lets their
tracks vote on the depth as the ensemble's do, returning the winning group's
track in the cascade's order. As `renormal`'s base (`--opt base=core_vote`),
it adds 0.009 on seoul_bull and 0.010 on the Kerry Park sample, at 3.5 to 5
times the full pass's median time. Running the members after the first only
when its track has under six views (`--opt 'base_options={"confirm_below_views":
6}'`) keeps 0.009 and 0.005 at 2.3 and 1.5 times. It is a small gain for its
cost, so `renormal` keeps `core_cascade` as its base.

**The slow ensemble.** Measured before the wide-angle change, `renormal`
with `centred` as its base (`--opt
base=centred`, the same normal and fallbacks) scores, over the same 90-point
Kerry Park sample, 0.714 and 0.393 against `renormal`'s 0.674 and 0.406 in
the full and empty passes, and 0.690 and 0.453 against 0.645 and 0.438 on all
of seoul_bull, at 18 to 25 times the time. Most of its gain is good tracks in
the full pass.

## Anchors: the first step on its own

Building a track at a pixel starts with no idea of the pixel's depth. The
first step is to find **anchors**: 3D points near the pixel that several
photographs agree on, each with the pixel it sits at in the queried image,
for later steps to start from and walk toward the pixel. The step is framed
in [`specs/drafts/surface-co-solve.md`](../../specs/drafts/surface-co-solve.md),
"The first step: anchors". `anchors.py` implements it, and the harness
measures it on its own with `--mode anchors`: no track is built, and each
query's anchors are scored against the ground truth, which the finder never
sees.

```bash
pixi run -e test python scripts/track_at_pixel/run_sharded.py --mode anchors --candidate anchors --shards 16 --out <run>
pixi run -e test python scripts/track_at_pixel/run_sharded.py --mode anchors --candidate anchors --opt stop=never --shards 16 --out <run>
```

### The sources

`find_anchors` tries its sources in order, strongest first, and by default
stops after the first that leaves two usable anchors (bounded or far, below)
within 20 px of the pixel. `--opt stop=never` runs every source, so each is
measured.

1. **Tracks** (`tracks`): the reconstruction's points observed within 40 px of
   the pixel in the queried image, finite, in two or more images, every
   observation within 2 px of the point's projection.
2. **Clusters** (`clusters`): the cluster-patches clusters with a member within
   48 px (up to 16), vetted with the posed cameras. By default
   (`cluster_members=any`) every member in a reconstruction image is used,
   preferring the reference and kept members and then the best-reading one
   per image; the triangulation drops the worst member while three or more
   remain, never the queried image's, until every member meets the point in
   front of its camera within 2 px. `cluster_members=kept` uses only the
   reference and kept members, and needs the queried image's member to be one
   of them.
3. **Guided matching** (`guided`): with the cameras posed, a keypoint's match in
   another image lies where that image's keypoint rays pass close to its own
   ray. The 8 keypoints nearest the pixel, within 24 px, are each compared with
   every other image's keypoints whose rays pass within 2 px of theirs. The
   nearest descriptor is the match when it passes the ratio test (0.8) and is
   within 250 of the query's descriptor. The matches are triangulated, dropping
   the worst while three or more remain. A second pass then adds each match
   that passed the ratio test but was over the cap, up to 400, if every view
   still meets the new point within 2 px. Two views are enough for an anchor.
4. **Constellation** (`constellation`): the SIFT index's constellation query
   from the pixel. The pixel carried into each matched image by the
   constellation's affine warp is triangulated, dropping the worst sighting
   while three or more remain, within 3 px. This anchor sits at the pixel.
   `constellation_at=keypoints` also queries from up to 4 keypoints within
   24 px of the pixel.
5. **The far-field sweep** (`farfield`, run after the others, below).
   `far_test=infinity` runs the infinity test in its place.
6. **Plane sweep** (`sweep`, not in the default sources): the sweep of
   `candidates/planesweep.py` at the pixel, keeping the best depth when two or
   more other photographs read ZNCC 0.8 or better there.

### Reading the pixel's patch, whole and in the middle

The far-field sweep, the infinity test and the layers' evidence read the
pixel's own patch in the other photographs with `read_patch`: an 11 by 11 grid of samples, 17 px across,
on a plane facing the queried camera at each distance asked for, sampled in
every image it lands in. Each read gives two ZNCCs from the same samples: of
the whole grid, and of its middle 5 by 5. When the whole patch matches and its
middle does not, the parts away from the pixel carry the match: a small near
object whose patch is mostly the background behind it, a pixel at a depth
edge, or a facade whose pattern repeats along the epipolar line.

### Ranges, support and layers

Photographs a few metres apart looking at a point a hundred metres away fix
its distance only loosely, and two views from adjacent frames may not fix it
at all. Each anchor therefore carries a **range**: the distances along its
pixel's ray at which every view stays within a pixel (or half a pixel more than
its own largest error, when that is larger). `distance_range` finds it by
doubling away from the triangulated distance until a view leaves its tolerance,
then bisecting. An anchor is **bounded** when its range is finite at both ends
and no wider than 3 times (`max_span`), far over near. It is **far** when its
range has no far end and its near end is at least 5 times the largest distance
between two cameras (`far_spread`): a reading that the point is further than
the photographs can tell from infinity. Bounded and far anchors are usable;
a range with no far end that starts nearer, such as two adjacent frames give,
is not.

Two anchors **support** each other when both are usable, their ranges
overlap, and neither's images are all among the other's, so the two readings
do not rest on the same photographs.

The scene near a pixel can hold surfaces at very different depths: at Kerry
Park point 33, within 40 px of the pixel there are trees at 0.3 to 0.4 times
its distance, the city at 3.3 to 3.5 times, and points at infinity. An anchor
near the pixel may be real geometry of another surface, so the anchors are
hypotheses. `find_anchors` also returns **layers**: the usable anchors grouped
by overlapping ranges, nearest first. Each layer carries what supports it
(`layer_evidence`): how many anchors, how many rest on different photographs,
their most views and widest ray angle, the nearest one's distance from the
pixel, and the pixel's own patch read at the layer's distances, whole and in
the middle, in every photograph (five distances across the range, from
infinity in for a far layer). Its `score` is the mean of the whole patch's
reading and the lesser of the whole and middle readings, each the mean of the
three best images. Its `key` adds the middle's reading alone and takes off
0.05 times `ln(1 + nearest_px)`, the distance from the pixel to the layer's
nearest anchor; `rank` orders the layers by the key (`layer_rank=score` by
the score alone). Walking to the pixel starts from rank 1 and has the others.

Each layer also carries a `confidence`, the chance that it is the pixel's: a
logistic of its margin over the best other layer's key (1 when it is the only
layer), `ln(1 + votes)`, `ln(1 + support)` and `ln(1 + nearest_px)`, with
coefficients fitted on the first-ranked layers of both ground truths. A
single layer can be wrong, and nothing ranks it below another; the
confidence is what a later step can refuse it on. The far-field anchors' own
metrics added nothing to the key or the confidence once support and votes
were in, and stay on the anchors.

### The far-field sweep

`from_farfield` reads the pixel's patch from infinity in along its ray. The
distances are counted as disparities `d`, the pixels the pixel shifts by in
the image that moves it most for a change of inverse distance: `d = 0` is
infinity, and `R / d` the distance, `R` that image's pixels per unit of
inverse distance. The sweep reads 0, 1, 2, 3, 4, 6, 8, 10, 12, 14 and 16 px
(`ff_disparities`) in every image the pixel lands in.

- **Wide images.** The reading at a disparity is over the images whose whole
  patch reads 0.8 somewhere in the sweep and that move the pixel at least half
  as far as the widest of them (`ff_wide`, `ff_wide_among=matching`). Images
  that barely move read alike at every disparity. Judging the width among the
  images that match keeps an image that sees something else at the pixel from
  setting the scale; `ff_wide_among=all` judges it against every image.
- **The reading.** At each disparity, the middle's mean over the three best
  wide images, or the whole patch's where the query's middle is flat (a grey
  standard deviation under 8), since a flat middle correlates with noise.
- **Peaks.** Every peak of the reading is a candidate, up to 3
  (`ff_max_peaks`): not only the highest. A peak needs the whole patch to read
  0.8 there and, unless the middle is flat, the middle 0.7; it must stand
  0.02 above the reading around it (`ff_min_prominence`), since a flat
  reading has no peak; and it must not be at 16 px, where a rising reading
  says the peak is further in. Its range runs to the midpoints with the
  neighbouring disparities.
- **Which images belong** (`_ff_group`). The sweep compares each image with
  the query only, so a reading can gather images of two surfaces. The query's
  patch and each agreeing image's, sampled at the peak, are compared pair by
  pair by the ZNCC of their middles (of the whole patch where the middle is
  flat), and grouped by average linkage while the closest groups' mean is 0.9
  or more (`ff_group_cut`), over at most 16 images (`ff_group_max`). The table
  grows as the square of the images, which the cap bounds; how it scales past
  a few dozen is not settled. The reading keeps the query's group. When the
  query stands alone and two or more other images form a group, that group is
  fit on the bench with the query turned out: where it lands more than 8 px
  from the pixel, within 48 px, with every image within 2 px, the anchor moves
  to that point, at the pixel where it lands (`refit="moved"`). When no two
  other images agree either, the reading stands as it was.
- **Metrics.** Each far-field anchor carries `farfield`: its `whole` and
  `middle` readings and their profiles over the sweep, the peak's
  `prominence`, `peak_rank` and the number of `peaks`, whether the middle was
  flat, how many `images` agree and the most parallax any of them has
  (`parallax_px`, against `widest_px` for the widest image that sees the
  pixel), the query's mean middle ZNCC with its group (`group_middle`) and how
  many images were `left_out`. A later comparison between candidates can
  weigh these; the sweep keeps readings it has no clear evidence against.

The sweep runs when the other sources gave no usable anchor, more than one
layer, or no usable anchor at the pixel (`infinity=needed`, the default). A
pixel at infinity often has one layer of nearer anchors beside it and nothing
at it.

### The infinity test

`from_infinity`, with `far_test=infinity`, reads the patch only where the
pixel lands at infinity, and the pixel is at infinity when one or more
images, and half of those it lands in, read 0.8 or better. Two more checks
follow:

- **The peak along the ray.** The agreeing images are read at the distances
  that move the pixel 1, 2, 4, up to 128 px from its position at infinity. If
  a finite distance reads 0.02 better than infinity, there is no reading.
  Otherwise the range's near end is where the reading first falls 0.05 below
  infinity's.
- **The middle.** When the patch's middle has texture (a grey standard
  deviation of 8 or more), it must also read 0.8 on average at infinity in the
  agreeing images.

### How the harness scores anchors

The ground truth does not hold every surface, so an anchor with no true point
near it is not necessarily wrong. The harness asks three questions:

- **On the pixel's layer**: is an anchor's range overlapping the true point's
  surface along the anchor's own ray? That is where the anchor's ray meets the
  true point's plane, scaled by the true point's own range at the pixel over
  its distance, so a neighbour on sloping ground is judged at its own
  distance, with the ground truth's own uncertainty. For a true point at
  infinity, its range itself.
- **At the pixel**: for a usable anchor within 1 px of the pixel, is its range
  overlapping the true point's range (right) or not (wrong)?
- **Checked**: when a true point is observed within 1.5 px of an anchor's own
  pixel in the queried image, do their ranges overlap? An anchor with no true
  point at its pixel is unchecked.

Each row also records, per layer, whether its range overlaps the true point's
(`layers_right`), which measures the ranking.

The summary's columns, per source and for all sources together, as shares of
all queries unless noted:

| Column | Meaning |
|---|---|
| has | With at least one anchor |
| mean n | Anchors a query |
| bnd | Share of the anchors that are usable |
| at px, right, wrong | With a usable anchor at the pixel; with one there that is right; with one there but none right |
| layer | With an anchor on the pixel's layer |
| lyr px | Median distance in the queried image from the pixel to the nearest anchor on its layer |
| l+sup | With an anchor on the pixel's layer that another anchor supports |
| chk, agree | Share of the anchors that are checked, and of those, the share that agree |
| s | Mean seconds a query |

### Results

Every source run (`stop=never`), the defaults otherwise:

| | Pass | Source | has | bnd | at px | right | wrong | layer | lyr px | l+sup | chk | agree | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seoul_bull | full | tracks | 0.927 | 1.000 | 0.000 | 0.000 | 0.000 | 0.741 | 12.7 | 0.372 | 1.000 | 1.000 | 0.014 |
| | | clusters | 0.987 | 0.999 | 0.040 | 0.034 | 0.005 | 0.942 | 5.6 | 0.803 | 0.027 | 0.922 | 0.017 |
| | | guided | 0.891 | 0.998 | 0.047 | 0.038 | 0.009 | 0.811 | 4.7 | 0.688 | 0.037 | 0.877 | 0.048 |
| | | constellation | 0.763 | 0.999 | 0.762 | 0.554 | 0.208 | 0.554 | 0.0 | 0.338 | 1.000 | 0.732 | 0.018 |
| | | farfield | 0.079 | 1.000 | 0.079 | 0.055 | 0.024 | 0.055 | 0.0 | 0.031 | 0.991 | 0.609 | 0.013 |
| | | all | 0.999 | 0.999 | 0.810 | 0.612 | 0.199 | 0.984 | 0.0 | 0.836 | 0.330 | 0.955 | 0.188 |
| | empty | all | 0.998 | 0.999 | 0.810 | 0.612 | 0.199 | 0.977 | 0.0 | 0.809 | 0.091 | 0.778 | 0.113 |
| Kerry Park | full | tracks | 0.955 | 1.000 | 0.006 | 0.003 | 0.003 | 0.687 | 8.4 | 0.360 | 1.000 | 1.000 | 0.055 |
| | | clusters | 0.946 | 0.875 | 0.048 | 0.044 | 0.004 | 0.851 | 7.4 | 0.691 | 0.063 | 0.958 | 0.017 |
| | | guided | 0.854 | 0.968 | 0.068 | 0.060 | 0.007 | 0.716 | 5.4 | 0.574 | 0.083 | 0.844 | 0.096 |
| | | constellation | 0.173 | 0.990 | 0.171 | 0.093 | 0.078 | 0.093 | 0.0 | 0.049 | 1.000 | 0.542 | 0.029 |
| | | farfield | 0.133 | 1.000 | 0.131 | 0.074 | 0.058 | 0.074 | 0.0 | 0.071 | 0.979 | 0.511 | 0.104 |
| | | all | 0.994 | 0.939 | 0.357 | 0.232 | 0.125 | 0.958 | 4.2 | 0.762 | 0.405 | 0.968 | 0.489 |
| | empty | all | 0.957 | 0.908 | 0.353 | 0.231 | 0.122 | 0.891 | 4.5 | 0.691 | 0.097 | 0.796 | 0.269 |

The "all" rows' seconds include reading the pixel's patch for the layers'
evidence: 30 to 110 ms a query at the median. The far-field sweep's wrong
readings at the pixel are candidates; most rank below the right layer (below),
but they count in "wrong" here.

The default, stopping once enough anchors are found:

| | Pass | has | at px | right | wrong | layer | lyr px | l+sup | agree | s/query (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| seoul_bull | full | 0.999 | 0.138 | 0.101 | 0.037 | 0.947 | 8.2 | 0.487 | 0.988 | 0.132 |
| | empty | 0.998 | 0.164 | 0.119 | 0.045 | 0.968 | 5.0 | 0.763 | 0.813 | 0.062 |
| Kerry Park | full | 0.994 | 0.159 | 0.092 | 0.066 | 0.879 | 6.7 | 0.374 | 0.986 | 0.379 |
| | empty | 0.957 | 0.191 | 0.124 | 0.068 | 0.876 | 6.4 | 0.621 | 0.831 | 0.187 |

The empty pass, rescored, as the sources were added (`stop=never`; the empty
pass has no tracks):

| Sources | Kerry Park layer | l+sup | seoul_bull layer | l+sup | Kerry Park s |
|---|---|---|---|---|---|
| Clusters (kept, 24 px) and constellation, the first version | 0.664 | 0.321 | 0.821 | 0.471 | 0.029 |
| + guided matching (two views), clusters of any member within 48 px | 0.813 | 0.594 | 0.946 | 0.800 | 0.120 |
| + guided matching's second pass, ranges up to 3 times | 0.829 | 0.624 | 0.944 | 0.796 | 0.129 |
| + far readings kept | 0.889 | 0.689 | 0.965 | 0.800 | 0.154 |
| + the infinity test, and the layers' evidence | 0.889 | 0.689 | 0.977 | 0.802 | 0.215 |
| + the far-field sweep in place of the infinity test (the defaults) | 0.891 | 0.691 | 0.977 | 0.809 | 0.269 |

The infinity test, empty pass. Kerry Park has 248 queries whose true point is
at infinity, seoul_bull 44:

| Infinity test | Kerry Park: right at the pixel, at infinity | wrong far readings on finite points | seoul_bull: right at the pixel, at infinity | wrong far readings on finite points |
|---|---|---|---|---|
| None (far readings from the other sources only) | 7 | 0 | 5 | 0 |
| Three agreeing images | 220 | 16 | 16 | 0 |
| One agreeing image | 220 | 16 | 42 | 25 |
| One image, and the peak along the ray | 191 | 1 | 42 | 2 |
| One image, the peak, and the middle | 191 | 0 | 40 | 0 |

The far tests compared, empty pass, every source run. "Right" counts the
queries with a right far reading at the pixel; "rank 1 right" the queries whose
first-ranked layer is the pixel's, of all 3903 on Kerry Park and 1277 on
seoul_bull; "wrong at rank 1" the queries whose first-ranked layer holds a
wrong far reading.

| Far test | Kerry Park: right at infinity (of 248) | right, 300 m or more (of 59) | rank 1 right | wrong at rank 1 | seoul_bull: right at infinity (of 44) | rank 1 right | Kerry Park s |
|---|---|---|---|---|---|---|---|
| The infinity test | 188 | 0 | 2937 | 0 | 37 | 1144 | 0.216 |
| Sweep to 10 px, best middle, reads at 16 to 128 px to reject | 211 | 30 | 2941 | 1 | 31 | 1140 | 0.293 |
| Sweep to 16 px, best middle, images grouped pairwise | 213 | 31 | 2941 | 1 | 31 | 1140 | 0.251 |
| Sweep to 16 px, every peak (the default) | 225 | 48 | 2943 | 15 | 38 | 1143 | 0.269 |

With the default stopping rule, rank 1 is right on 2839 of Kerry Park's empty
pass queries against 2829 with the infinity test, and 2103 against 2088 on the
full pass; on seoul_bull, 1108 against 1109 and 1010 against 1011.

Ranking the layers, over the queries with more than one layer and the pixel's
among them (1841 on Kerry Park and 734 on seoul_bull, empty pass): the share
where the pixel's layer is ranked first, and in the top two.

| Ranking | Kerry Park top 1 | top 2 | seoul_bull top 1 | top 2 |
|---|---|---|---|---|
| Nearest anchor to the pixel | 0.854 | 0.964 | 0.820 | 0.960 |
| Anchors' views, weighted by distance from the pixel | 0.866 | 0.964 | 0.883 | 0.952 |
| Photographs voting for the layer they read best | 0.884 | 0.971 | 0.895 | 0.960 |
| The pixel's patch, whole | 0.901 | 0.983 | 0.926 | 0.984 |
| The pixel's patch, middle | 0.926 | 0.988 | 0.899 | 0.980 |
| The pixel's patch, whole and middle (the `score`) | 0.929 | 0.989 | 0.939 | 0.988 |
| The `score` with the far-field sweep's candidates | 0.931 | 0.988 | 0.937 | 0.988 |
| The `key`: the score, the middle, and the nearest anchor's distance (the default) | 0.944 | 0.995 | 0.941 | 0.989 |

A layer chosen at random is first about 39% of the time.

The key was chosen by fitting a ranking over every layer feature on one ground
truth and testing it on the other; three features carried it, and the
middle's reading and the nearest anchor's distance carried the same weight
fitted on either. Adding the anchors' support ranked better with every source
run and worse with the default stopping rule, which stops after the
reconstruction's own tracks: their many views give a layer high support
whichever surface it is on. Queries whose first-ranked layer is the pixel's,
of all of them:

| Ranked by | Kerry Park, every source, empty | full | default stopping, empty | full | seoul_bull, every source, empty | full | default stopping, empty | full |
|---|---|---|---|---|---|---|---|---|
| The `score` | 2943 | 3008 | 2839 | 2103 | 1143 | 1149 | 1108 | 1010 |
| The `key` (the default) | 2969 | 3059 | 2843 | 2153 | 1146 | 1153 | 1115 | 1005 |

The confidence of the first-ranked layer tells a right one from a wrong one
with an area under the ROC curve of 0.91 to 0.94 with every source run, and
0.83 to 0.91 with the default stopping rule; the first layer's score alone
gives 0.74 to 0.77. Fitted on one ground truth, it gives 0.83 to 0.94 on the
other. The summary reports both on its `rank 1` line.

Each source's variants alone, empty pass, as the share of queries with an
anchor from it (which does not depend on the scoring):

| Variant | Kerry Park | seoul_bull | Kerry Park s |
|---|---|---|---|
| Clusters, kept members, 24 px | 0.833 | 0.862 | 0.002 |
| Clusters, kept members, 48 px | 0.930 | 0.962 | 0.005 |
| Clusters, any member, 24 px | 0.860 | 0.951 | 0.003 |
| Clusters, any member, 48 px | 0.946 | 0.987 | 0.008 |
| Constellation, target 50 (the default) | 0.173 | 0.763 | 0.021 |
| Constellation, target 25 | 0.404 | 0.691 | 0.022 |
| Constellation, target 100 | 0.030 | 0.652 | 0.048 |
| Constellation, target 200, 4 inliers | 0.000 | 0.404 | 0.075 |
| Constellation, also from 4 keypoints | 0.347 | 0.915 | 0.133 |
| Guided, three views | 0.659 | 0.614 | 0.090 |
| Guided, three views, skipping the pixel's own keypoint | 0.654 | 0.614 | 0.090 |
| Guided, two views | 0.854 | 0.891 | 0.091 |
| Guided, two views, 48 px and 16 keypoints | 0.915 | 0.956 | 0.197 |
| Guided, two views, no descriptor cap | 0.410 | 0.996 | 0.087 |
| Plane sweep at the pixel (a 60-point sample) | 0.715 | 0.536 | 0.632 |

- **Guided matching is the strongest new source.** Restricting each keypoint's
  candidates to those whose rays meet its own makes the descriptor comparison
  far more reliable than an unconstrained search. It does as well when the
  pixel's own keypoint is skipped, so it does not depend on the harness always
  querying a detected keypoint. Removing the descriptor cap lets wrong matches
  into the first triangulation on Kerry Park, which dropping the worst view
  cannot recover from; the second pass adds them only after a triangulation
  exists. At Kerry Park point 5, a pixel on a depth edge beside an apartment
  facade, its best anchor has 14 views whose rays meet within 0.1 px, up to 25
  degrees apart.
- **Clusters are the most accurate source.** Where the ground truth can check
  them, 96% of Kerry Park's cluster anchors and 92% of seoul_bull's agree with
  it. Using every member and letting the triangulation drop the bad ones, over
  a 48 px radius, gives 95% and 99% of queries a cluster anchor.
- **The constellation query is the one source that usually reads the pixel
  itself, and it is the least accurate.** On Kerry Park 9.3% of queries get a
  right reading at the pixel from it and 7.8% a wrong one; on seoul_bull 55%
  and 21%. It comes last, as a starting hit where the clusters and the guided
  matches give nothing; with `stop=enough` it runs on 1 to 2% of empty-pass
  queries. A wider search makes it worse: at a target of 100 it answers on 3%
  of Kerry Park's queries.
- **Points at infinity.** Kerry Park's 248 queries at infinity had an anchor on
  the pixel's layer 21 times when a range with no far end counted as
  unusable. The clusters and the guided matches had far readings all along (at
  point 272, from 354 to 948 m out to infinity); keeping them as far readings puts 240 of the
  248 on their layer. The infinity test adds a reading at the pixel itself:
  191 of them.
- **How many images see infinity depends on how the photographs were taken.**
  Kerry Park's paired fisheyes see the horizon in 20 or more images;
  seoul_bull's photographs walk around a sculpture facing it, and a pixel at
  infinity lands in only 1 to 5 of the others. With three agreeing images
  required, the test found 16 of seoul_bull's 44; with one, 42, but also 25
  finite points, most of them 11 to 14 m away behind the lawn, such as the
  hedge, whose leaves read well at any shift along the ray. The peak along the ray removes those.
- **The middle of the patch.** The last wrong far readings were a birdhouse on
  a pole in front of distant houses (seoul_bull point 180), whose 17 px patch
  is mostly the houses, and the balconies of an apartment building (Kerry Park
  point 5), which repeat along the epipolar line. The birdhouse's middle reads
  0.47 and below at infinity, and the balconies' 0.78 on average. A larger
  patch does not do as well: requiring the test to pass at 16 px too rejects
  the balconies, but 26 of Kerry Park's far readings at infinity then start
  too near to count as far.
- **The far-field sweep in place of the infinity test.** The infinity test
  asks whether the pixel is at infinity and, reading along the ray, rejects
  it when something nearer reads better. The sweep asks where in the far
  field the pixel reads, and returns each peak it finds. Kerry Park's points
  300 m and more away had no reading at the pixel from the infinity test and
  have 48 of 59 from the sweep.
- **Reading further in to reject a far pick did nothing for the answer.** At
  16, 32, 64 and 128 px the reads fell on either side of narrow peaks: at Kerry
  Park point 250, 93 m away, the tree reads 0.99 at 40 px, 0.79 at 32 and 0.46
  at 64. Of the 22 far readings those reads rejected, the 18 wrong ones were
  already ranked below the right layer (or, once, below a wrong one), and 4
  were right.
- **Every peak, not the best.** At seoul_bull point 88, from a single image,
  the whole patch reads best at infinity (0.98) and its middle best at 12 px
  (0.97); the true point is at infinity. With one image there is nothing to
  decide between them, so both go out. Picking the best reading returned
  only the wrong one.
- **Two surfaces in one reading.** At Kerry Park point 251 the pixel is on a
  distant tower. Three images read the tower, and three taken close together
  see a nearer tree in front of it, whose patch resembles the query's as a
  whole. Against a blend of all six, every image reads in between; pair by
  pair, by the middle, they form two groups (0.94 to 0.97 and 0.96 to 0.99,
  with 0.78 to 0.90 between them), and the reading keeps the tower's.
- **The query standing apart.** At Kerry Park point 5, a balcony edge, the
  query's observation reads 0.71 on the bench, the other images' 0.90 to 0.97.
  Fit without the query they meet on the next building's roofline, 30 to
  40 px from the pixel, where the anchor moves.
- **Lookalikes that do not separate.** At Kerry Park point 250 two dark trees
  stand side by side against the sky. From one frame the sweep pairs the
  query's tree (point 250, 93 m) with its twin (point 18, 100 m) in two
  neighbouring frames, which agree at any depth, and reads about 900 m. The
  query's middle reads 0.90 against them, so it joins their group. The
  reading goes out with its evidence, two images and little parallax, against
  the 19 images that read the tree at 91 m.
- **Which images are wide.** Seoul_bull point 93 is seen by two images; the one
  that moves it most sees something else there and matches nowhere. Judged
  against it, the image that does match was not wide enough to be read.
  Judging the width among the images that match gives the reading, and on
  Kerry Park 15 more of the points 300 m and more away.
- **Ranking the layers.** Near a pixel the layers are real surfaces, so which
  one the pixel is on takes its own patch. Read at each layer's distances,
  whole and in the middle, it ranks the pixel's layer first 93 to 94% of the
  time where there are several, and 94% weighted toward the middle and less a
  little for the nearest anchor's distance. The anchors' own features rank it
  first 82 to 88% of the time; they tell instead how far to trust the first
  layer, which the confidence carries.
- **The plane sweep at the pixel** reads some depth on 72% of Kerry Park's
  queries, but costs 0.6 s a query, and in the full pass it mostly picks a
  wrong depth. It is kept as an option.
- **Point 33.** Two sources gave anchors at the pixel that the first scoring
  called 10% too deep. Their sightings are within 0.5 px of the true point's
  projection in every view: the same matches. The true track itself only fixes
  the distance to between 0.90 and 1.08 times its value within a pixel, and the
  anchors' views, which lacked the true track's widest two, did not bound it
  from above. The first scoring, in fractions of a half-size, was tighter than
  the photographs can resolve; on Kerry Park, anchors whose rays meet at under
  2 degrees passed it only 38% of the time. That led to the ranges, the second
  pass of guided matching, and this scoring.
- **Checked anchors are few.** In the empty pass only 8.9% of Kerry Park's
  anchors have a true point at their own pixel, so "agree" describes a small
  sample, and the shares with an anchor on the pixel's layer are the main
  measure.

## Files

| File | Role |
|---|---|
| `api.py` | The contract: `build_track(ctx, image, pixel, options) -> TrackAtPixelResult`, or raise `TrackAtPixelError(stage, reason, diagnostics)` |
| `dataset.py` | Ground-truth registry, the cache workspace, SIFT, the `.kdf`, the cluster-patches `.matches` |
| `context.py` | `DatasetContext`, loaded once: photographs as an `ImagePyramidSet`, `LazyKdForest`, keypoints, cameras (−Z forward, depth = −z), per-point frames with 2D/3D indexes. `HoldoutContext` is what a candidate sees: `edited`, `observations_near(image, pixel, r)`, `clusters_near(image, pixel, r)`, `points_near(xyz)`, `texel_scales(track)`, `keypoints(image)`, `camera(image)` |
| `metrics.py` | The per-query score |
| `harness.py` | The loop, the JSONL rows, the summary. `--mode anchors` runs only the anchor step |
| `anchors.py` | The first step, on its own: `find_anchors(ctx, image, pixel)` returns the anchors near the pixel from the reconstruction's tracks, the vetted clusters, guided matching, the constellation query and the infinity test, each with its range of distances, grouped into depth layers ranked by the evidence for each; `distance_range` finds a range; `read_patch` reads the pixel's patch along its ray, whole and in the middle; `score_anchors` and `summarize` are the harness's side |
| `compare.py` | Several runs side by side, each re-judged against the current good-track bar |
| `candidates/baseline.py` | The first candidate. A local prior sets size and normal, then: cluster, constellation search plus lateral search, cluster evaluate, upgrade, tilt to the prior normal, geometry search, refit, gates. `--opt finish=common` swaps its last two steps for `common.finish` |
| `candidates/common.py` | What the other candidates share once a track stands: `finish` (anchored fit, neighbours' normal, geometry search, cleaning, gates), and `track_from_sightings`, which upgrades a pixel plus a list of `(image, pixel)` sightings straight to a track |
| `candidates/sweep.py` | No descriptors. The neighbours' depth modes each give a plane; the pixel's ray meets it; the guess is projected into the views that face it and are not occluded, and the fit corrects it |
| `candidates/transfer.py` | No descriptors, no depth guess. For each other image, a weighted affine map is fitted to the neighbours' own matched keypoints and carries the pixel across |
| `candidates/clusters.py` | Reads the cluster-patches `.matches`: the nearest clusters' kept members, with the pixel's offset from the member carried into each image through the members' affine shapes |
| `candidates/cascade.py` | Runs `clusters`, `transfer`, `sweep` and the descriptor route in that order and returns the first track that passes its own gates |
| `candidates/core_cascade.py` | The same cascade run in Rust by `bench.build_track_at_pixel`, with the dataset's SIFT index, keypoints and `.matches` clusters built once into a `bench.TrackAtPixelSources`. Its descriptor route is reported as the member `constellation` |
| `candidates/planesweep.py` | Needs only the poses and the photographs. Sweeps a patch facing the queried camera along the pixel's ray (uniform in inverse depth, down to infinity), scores each depth by the other views' ZNCC against the query, and fits the best-agreed depths |
| `candidates/centred.py` | The ensemble, preferring within the winning group the tracks whose queried view's correlation peak is within 0.5 px of the pixel |
| `candidates/renormal.py` | The core cascade with the gates at two views and 0.7, Python fallbacks when it refuses, and the track re-oriented by the tight-to-wide neighbour chain, the 3D neighbours or the photometric normal. The variants the harness measured and did not keep are options |
| `candidates/core_vote.py` | Each Rust cascade member run alone, their tracks voting on the depth as the ensemble's do |
| `ground_truth_ceiling.py` | Scores the ground-truth tracks themselves, as a run directory: the ceiling the good-track bar allows |
| `run_sharded.py` | Runs the harness over every point in parallel shards and merges them into one run directory |
| `goal_score.py` | The normal-weighted score `S` of "Normals, gates and fallbacks", with each pass's median time |
| `normal_sources/` | Each normal source measured alone against the ground-truth normal: `normal_diag.py` (neighbours in 2D and 3D, planes, photometric), `nb_sweep.py` (the neighbour estimator's parameters and chains), `photo_diag.py` (`refine_normals` settings), `affine_diag.py` (cluster members' affine shapes), `cluster_nb_diag.py` (pseudo-neighbours from clusters), `photo_bias.py` (how the photometric normal misses the ground truth's, and priors that might correct it), `photo_neighbour_oracle.py` (averaging over true neighbours' photometric normals) |
| `candidates/ensemble.py` | Runs every member above at 1, 1.5 and 2 times its own patch size, with the ray consensus in the finish and gates at the good-track bar's ZNCC. The members' tracks all lie on the pixel's ray, so they vote on the depth; the group most distinct members agree on wins, and its track is chosen in the cascade's order |

The baseline's `size_policy` option (`prior`, `largest_view`, `median_view`,
`smallest_view`, with `texel_scale_target`, default 1.0) resizes the patch so
the chosen view samples at the target ratio, and refits:

```bash
pixi run -e test python scripts/track_at_pixel/harness.py --opt size_policy=largest_view
```

A new candidate is a new module in `candidates/` with a `DEFAULTS` dict and
`build_track`; run it with `--candidate <name>`. To compare runs:

```bash
pixi run -e test python scripts/track_at_pixel/compare.py <run> <run> ...
```

## Metrics

All are computed over the built track's `in` observations.

**A good track.** Candidates have their own gates, so the number of tracks they
return does not compare them. The harness judges every returned track against
one bar (`metrics.GOOD_BAR`) and the summary counts the tracks that pass it
(`good`) and the ones that do not (`built but not good`). A track is good when
all of these hold:

- the queried sighting is `in` (its keypoint may settle away from the pixel:
  the track's job is to find the correspondences, and bundle adjustment
  reconciles a point with its keypoints);
- the point is within one ground-truth half-extent of the ground-truth point
  (0.5 degrees for a bearing);
- `view_precision` is at least 0.75;
- the median leave-one-out ZNCC is at least 0.7.

`view_precision` is the fraction of `in` views whose keypoint lies within 3 px
of where the ground-truth point projects in that image. Unlike
`image_precision` it credits a correct sighting in a photograph the
ground-truth track does not list (ground-truth tracks are not complete), and it
still charges one on another piece of surface, or an `in` view with no keypoint.

- **Contract**
  - `query_keypoint_offset_px`: the queried sighting's fitted keypoint versus
    the pixel asked about.
  - `query_projection_offset_px`: where the point projects in that image,
    versus the pixel.
  - `query_image_in`: whether the queried sighting is still `in`.
- **Geometry**
  - `position_err_angle_deg`: the angle the built point and the GT point
    subtend from the query camera.
  - `position_err` / `_rel_depth` / `_in_gt_halves` / `_along_ray` /
    `_lateral`: the 3D position error, raw and in other units.
  - `normal_err_deg`.
  - `half_extent_ratio`: world patch half-size, built ÷ GT.
  - `texel_scale_min` / `_median` / `_max`: image pixels per patch-bitmap
    texel over the `in` views, from the Jacobian of the render at the bench's
    24-texel resolution (`context.texel_scale`), with `gt_texel_scale_*` for the
    ground truth's frame over its own images. `texel_scale_aniso_max` is the
    largest ratio of the Jacobian's singular values.
  - finite versus infinity agreement.
- **Membership**
  - `image_precision` and `image_recall` against the GT track's images.
  - `view_precision`, described above.
  - `kp_err_median_px` / `kp_err_max_px`: keypoint error in the shared images.
- **Photometry**
  - `zncc_median` / `_min`: leave-one-out ZNCC, set against the same
    `evaluate` reading of the GT point (`gt_zncc_*`, `zncc_median_delta`).
  - `localizability_*` and `reproj_median`.
- **Cost**
  - `seconds` per query.

GT normals and sizes are what the ground truth's embedding pass produced. They
are good references, not exact truth: a built track can beat the GT ZNCC.

## Tools the candidates draw on

| Need | Call |
|---|---|
| Constellation search from a pixel | `bench.search_descriptors(track, obs, xy, affine, forest, radius_px=, min_inliers=)` (radius from `spatial.radius_for_feature_count`) |
| Geometry search | `bench.search_geometry(track, obs, edited, pyramids)` |
| Nearby observations in an image, and their depth, normal and size | `ctx.observations_near` |
| Clusters with a member near a pixel, with every member's image, refined position, shape, status and ZNCC | `ctx.clusters_near` (the raw arrays are on `ctx.dataset`: `cluster_starts`, `member_*`, `reference_members`, `cluster_radius`) |
| Nearby 3D patches, including ones with no image overlap | `ctx.points_near` |
| Evaluating a track without moving it | `bench.evaluate` |
| Fitting it (localize, refine, re-triangulate, re-fuse) | `bench.fit` |
| Moving between cluster and track stage | `bench.set_stage` |
| Hand moves | `bench.tilt_patch`, `translate_patch`, `translate_patch_to_pixel`, `resize_patch`, `spin_patch`, `sight_observation` |
| Congealing a subset of views | `PatchCloud.localize_keypoints(view_sets=…, basis_max_views=…)` |
| Sub-pixel refinement against the consensus | `PatchCloud.refine_keypoints` |
| Normal refinement | `PatchCloud.refine_normals` |
| Member coherence (a pairwise ZNCC matrix, and a split proposal) | `PatchCloud.validate_member_coherence` |
| Localizability | `PatchCloud.score_localizability_deprecated` |
| Adjacency-surfel normals | `analysis.estimate_adjacency_surfel_normals` |
| Cluster refinement | `matching.refine_cluster_patches` |

Two kernels are not bound yet: registering one bitmap directly against another,
and congealing a bare stack of bitmaps. The `PatchCloud` kernels reach both
indirectly, through a one-patch cloud built with `from_halfvec_arrays`.
