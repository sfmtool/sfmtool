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
"The first step: anchors". `anchors.py` is its first version, and the harness
measures it on its own with `--mode anchors`: no track is built, and each
query's anchors are scored against the ground truth, which the finder never
sees.

```bash
pixi run -e test python scripts/track_at_pixel/run_sharded.py --mode anchors --candidate anchors --shards 8 --out <run>
pixi run -e test python scripts/track_at_pixel/run_sharded.py --mode anchors --candidate anchors --opt stop=never --shards 8 --out <run>
```

`find_anchors` tries three sources, strongest first, and by default stops
after the first that leaves two anchors within 20 px of the pixel
(`--opt stop=never` runs all three, so each is measured):

1. **Tracks**: the reconstruction's points observed within 40 px of the pixel
   in the queried image, finite, in two or more images, every observation
   within 2 px of the point's projection.
2. **Clusters**: the cluster-patches clusters with a member within 24 px,
   vetted with the posed cameras. The queried image's member must be the
   reference or kept, and the reference and kept members must meet in front
   of every camera within 2 px. Every cluster that passes is used.
3. **Constellation**: the SIFT index's constellation query from the pixel. The
   pixel carried into each matched image by the constellation's affine warp is
   triangulated, dropping the worst sighting while three or more remain,
   within 3 px. This anchor sits at the pixel.

An anchor is **supported** when an anchor from another source, or another
cluster, lies within 10% of its depth of it in 3D.

The summary's columns, per source and for all sources together:

| Column | Meaning |
|---|---|
| has | Share of queries with at least one anchor |
| mean n | Anchors per query |
| near px | Median distance of the nearest anchor from the pixel, in the queried image |
| 3D h | Median distance of the nearest anchor from the true point, in the true point's half-sizes |
| <=2h | Share of queries with an anchor within 2 half-sizes of the true point |
| surface | Share with an anchor on the true point's surface: within a quarter of a half-size of its plane, and within 8 half-sizes of it |
| at px, <5% | Share with an anchor at the pixel, and with one there within 5% of the true depth |
| support | Share with an anchor another source supports |

The shares after "has" are of the queries whose ground truth is finite.

Every source run (`stop=never`):

| | Pass | Source | has | near px | 3D h | <=2h | surface | at px | <5% | support |
|---|---|---|---|---|---|---|---|---|---|---|
| seoul_bull | full | tracks | 0.927 | 13.4 | 1.40 | 0.737 | 0.713 | 0.000 | 0.000 | 0.899 |
| | | clusters | 0.862 | 6.0 | 0.77 | 0.773 | 0.612 | 0.032 | 0.032 | 0.850 |
| | | constellation | 0.763 | 0.0 | 0.42 | 0.650 | 0.359 | 0.785 | 0.634 | 0.729 |
| | | all | 0.958 | 0.0 | 0.48 | 0.926 | 0.852 | 0.786 | 0.637 | 0.924 |
| | empty | all | 0.912 | 0.0 | 0.43 | 0.853 | 0.671 | 0.786 | 0.637 | 0.796 |
| Kerry Park | full | tracks | 0.955 | 9.7 | 1.29 | 0.634 | 0.698 | 0.007 | 0.006 | 0.563 |
| | | clusters | 0.833 | 6.5 | 1.37 | 0.531 | 0.465 | 0.052 | 0.034 | 0.523 |
| | | constellation | 0.173 | 0.0 | 0.89 | 0.129 | 0.069 | 0.182 | 0.085 | 0.064 |
| | | all | 0.985 | 4.9 | 1.01 | 0.817 | 0.786 | 0.231 | 0.122 | 0.639 |
| | empty | all | 0.844 | 5.1 | 1.25 | 0.559 | 0.479 | 0.226 | 0.117 | 0.384 |

The default, stopping once enough anchors are found:

| | Pass | has | near px | 3D h | <=2h | surface | at px | support | s/query (median) |
|---|---|---|---|---|---|---|---|---|---|
| seoul_bull | full | 0.958 | 8.6 | 1.05 | 0.867 | 0.805 | 0.074 | 0.924 | 0.003 |
| | empty | 0.912 | 4.5 | 0.71 | 0.837 | 0.640 | 0.181 | 0.783 | 0.002 |
| Kerry Park | full | 0.985 | 7.2 | 1.14 | 0.749 | 0.758 | 0.036 | 0.535 | 0.008 |
| | empty | 0.844 | 6.1 | 1.30 | 0.548 | 0.475 | 0.095 | 0.372 | 0.002 |

- In the full pass nearly every query gets an anchor, mostly from the
  reconstruction's own tracks, and three quarters or more have one within 2
  half-sizes of the true point.
- On seoul_bull the clusters and the constellation are nearly as good as the
  tracks, so the empty pass loses little. The constellation is the only source
  that reads the pixel's own depth, and it is within 5% of the truth on 63% of
  queries.
- Kerry Park's empty pass is the gap. The constellation query answers for only
  17% of queries (fisheye frames give few keypoints around a pixel, and much of
  the scene is ground seen at a grazing angle), 16% of queries get no anchor at
  all, and only 38% get one that a second source supports.
- The finder costs a few milliseconds a query by default. The constellation
  query is most of the cost when it runs: with every source run, the median is
  0.012 s on seoul_bull and 0.028 s on Kerry Park.

## Files

| File | Role |
|---|---|
| `api.py` | The contract: `build_track(ctx, image, pixel, options) -> TrackAtPixelResult`, or raise `TrackAtPixelError(stage, reason, diagnostics)` |
| `dataset.py` | Ground-truth registry, the cache workspace, SIFT, the `.kdf`, the cluster-patches `.matches` |
| `context.py` | `DatasetContext`, loaded once: photographs as an `ImagePyramidSet`, `LazyKdForest`, keypoints, cameras (−Z forward, depth = −z), per-point frames with 2D/3D indexes. `HoldoutContext` is what a candidate sees: `edited`, `observations_near(image, pixel, r)`, `clusters_near(image, pixel, r)`, `points_near(xyz)`, `texel_scales(track)`, `keypoints(image)`, `camera(image)` |
| `metrics.py` | The per-query score |
| `harness.py` | The loop, the JSONL rows, the summary. `--mode anchors` runs only the anchor step |
| `anchors.py` | The first step, on its own: `find_anchors(ctx, image, pixel)` returns the anchors near the pixel from the reconstruction's tracks, the vetted clusters and the constellation query; `score_anchors` and `summarize` are the harness's side |
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
| Localizability | `PatchCloud.score_localizability` |
| Adjacency-surfel normals | `analysis.estimate_adjacency_surfel_normals` |
| Cluster refinement | `matching.refine_cluster_patches` |

Two kernels are not bound yet: registering one bitmap directly against another,
and congealing a bare stack of bitmaps. The `PatchCloud` kernels reach both
indirectly, through a one-patch cloud built with `from_halfvec_arrays`.
