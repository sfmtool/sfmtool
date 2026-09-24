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

```bash
pixi run -e test python scripts/track_at_pixel/harness.py                       # every point
pixi run -e test python scripts/track_at_pixel/harness.py --points 20 --seed 1   # a sample
pixi run -e test python scripts/track_at_pixel/harness.py --point-ids 144,177 --raise
pixi run -e test python scripts/track_at_pixel/harness.py --opt max_query_offset_px=3 --opt normal_prior=false
```

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

- `rows.jsonl`: one row per query, with the candidate's diagnostics.
- `summary.txt` and `config.json`.
- `tracks.sfmr`: every returned track, committed into the ground truth's
  cameras with none of its points.

A row's `output_point` is its track's index in `tracks.sfmr`. There is one
point per successful query, so a ground-truth point queried from five images
can appear up to five times. To compare the run with the ground truth, load
both into one Explorer window and toggle between them:

```bash
pixi run gui -- test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr <run>/tracks.sfmr
```

## Results on seoul_bull

Every point of the ground truth seen in two or more images, queried from each
image it is seen in (1277 queries, 44 of them on the 14 points at infinity),
judged against the good-track bar. The first row scores the ground-truth tracks
themselves against the same bar. It is the number a candidate is measured
against: 47 of the ground truth's own tracks miss the bar, mostly on the median
ZNCC.

| Candidate | Built | Good | Good at infinity | Not good | Good % | Median angle err (deg) | Median normal err (deg) | Median projection offset at the pixel (px) | s/query |
|---|---|---|---|---|---|---|---|---|---|
| ground truth | 1277 | 1230 | 44 | 47 | 96.3% | | | | |
| `baseline` | 289 | 254 | 11 | 35 | 19.9% | 0.102 | 13.3 | 0.67 | 0.30 |
| `baseline --opt finish=common` | 337 | 308 | 7 | 29 | 24.1% | 0.019 | 12.6 | 0.19 | 0.59 |
| `sweep` | 833 | 739 | 0 | 94 | 57.9% | 0.026 | 10.2 | 0.26 | 0.43 |
| `transfer` | 667 | 592 | 0 | 75 | 46.4% | 0.025 | 11.9 | 0.24 | 0.43 |
| `clusters` | 721 | 674 | 19 | 47 | 52.8% | 0.028 | 13.5 | 0.24 | 0.38 |
| `cascade` | 1025 | 925 | 20 | 100 | 72.4% | 0.030 | 12.7 | 0.26 | 0.62 |
| `core_cascade` | 1025 | 925 | 20 | 100 | 72.4% | 0.030 | 12.7 | 0.26 | 0.49 |
| `planesweep` | 699 | 614 | 14 | 85 | 48.1% | 0.024 | 13.5 | 0.25 | 1.43 |
| `ensemble` | 1239 | 1087 | 35 | 152 | 85.1% | 0.032 | 13.4 | 0.26 | 14.05 |
| `centred` | 1239 | 1107 | 35 | 132 | 86.7% | 0.031 | 13.3 | 0.25 | 7.11 |

The medians are over the built tracks. The second candidate row is the control:
the baseline's own way of finding sightings, with the shared finish. It shows
how much of the gain is the finish and how much is where the sightings come
from. The cascade's tracks came from `clusters` (721), `transfer` (183),
`sweep` (110) and the descriptor route (11). `sweep` and `transfer` build from
finite neighbours only, so they return nothing at infinity. Anchoring puts the
queried keypoint on the pixel by construction, so the centring number to read
for the new candidates is the point's projection offset, not
`query_keypoint_offset_px`. Times are with 32 processes sharing the machine
(22 for `core_cascade`).

`core_cascade` is the cascade as `sfmtool_core::bench::build_track_at_pixel`
runs it ([`specs/core/bench/track-at-pixel.md`](../../specs/core/bench/track-at-pixel.md)).
Against `cascade` it has the same outcome on all 1277 queries, from the same
member, with the same refusal stages. The tracks agree to about `1e-15` except
one, point 83 from image 8, whose median ZNCC differs by `1.3e-4`: a last-bit
change in the sweep's surface hypothesis decides which side of a discrete step
in the fit the track lands on, and perturbing the Python sweep's hypothesis by
one part in `1e15` moves it between the same two results. On the
`--points 15 --seed 3` sample, run one after the other on a loaded machine, the
Rust cascade's median was 0.31 to 0.37 s a query against the Python cascade's
0.34 to 0.39 s: the time is in the bench kernels both call, not in the
composition around them.

From the cascade on, each candidate is measured by how much of the gap to the
ground truth's 1230 good queries it closes. The cascade leaves 305. The
ensemble leaves 143, 53% of the cascade's gap; its tracks came from `clusters`
(878), `transfer` (197), `sweep` (123), `planesweep` (40) and the descriptor
route (1). `centred` leaves 123, 14% of the ensemble's gap.

## Where the remaining gap is

`centred` leaves 123 of the ground truth's 1230 good queries. These
measurements say where they are.

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

## Files

| File | Role |
|---|---|
| `api.py` | The contract: `build_track(ctx, image, pixel, options) -> TrackAtPixelResult`, or raise `TrackAtPixelError(stage, reason, diagnostics)` |
| `dataset.py` | Ground-truth registry, the cache workspace, SIFT, the `.kdf`, the cluster-patches `.matches` |
| `context.py` | `DatasetContext`, loaded once: photographs as an `ImagePyramidSet`, `LazyKdForest`, keypoints, cameras (−Z forward, depth = −z), per-point frames with 2D/3D indexes. `HoldoutContext` is what a candidate sees: `edited`, `observations_near(image, pixel, r)`, `clusters_near(image, pixel, r)`, `points_near(xyz)`, `texel_scales(track)`, `keypoints(image)`, `camera(image)` |
| `metrics.py` | The per-query score |
| `harness.py` | The loop, the JSONL rows, the summary |
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
