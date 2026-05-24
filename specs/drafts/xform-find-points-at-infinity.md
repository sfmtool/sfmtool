# Finding points at infinity in an existing solve

**Status:** Draft proposal. Adds a new `sfm xform` operation that *discovers*
points at infinity (and near-infinite distant points) in an already-solved
`.sfmr` by clustering the world-space directions of keypoints across all
images and confirming clusters with SIFT descriptors. Complements the existing
[`classify_points_at_infinity`](sfmr-v2-points-at-infinity.md), which only
*reclassifies* points that the solve already triangulated.

[v2 model]: sfmr-v2-points-at-infinity.md

## Motivation

A distant point carries a different kind of information than a nearby one. Its
track spans a tiny range of viewing angles (the rays from every camera are
nearly parallel), so it pins down depth poorly: triangulation needs parallax,
and a distant point has almost none. COLMAP's default incremental pipeline
detects and filters such tracks out: `triangulation.min_angle` /
`mapper.filter_min_tri_angle` reject tracks whose maximum viewing angle is below
1.5°, and `ignore_two_view_tracks` drops 2-view tracks outright (the
lowest-parallax tracks are predominantly 2-view).
The in-repo datasets produce **zero** points at
infinity, even outdoors where sky, ridgelines, and far rooflines clearly exist.

But that same near-constant direction makes a distant point a strong constraint
on the **relative rotation** between camera poses. A point at infinity is *pure*
rotational information: it has no depth to estimate, so it constrains
orientation without the position/scale coupling a finite point brings, the way
distant stars anchor a sextant. Near points fix translation and scale; far
points stiffen rotation. Recovering the far points lets them **complement** the
well-triangulated near ones.

We recently updated the `.sfmr` format to store [homogeneous points][v2 model],
so it can represent points at infinity. This document is about how to
efficiently search for points at infinity in an existing reconstruction, using a
spatial data structure over all the keypoints.

## The geometric insight

A finite 3-D point is seen along a *different* world-space ray from each camera;
the rays converge on the point, and the angle between them (the parallax) is
what triangulation needs. Matching finite points across images therefore
requires epipolar search: given a feature in image *i*, its match in image *j*
lies somewhere along an epipolar curve, because its depth is unknown. That is
what `feature_match/` does today (rectified / polar sweep along epipolar
curves).

A point at infinity is different. Its rays are **parallel**: every camera sees
it along the *same* world-space direction, independent of where the camera is.
So if we take each keypoint and un-project it to the world-space direction it
*would* have if it were at infinity, all the keypoints belonging to one infinite
point land on the **same spot on the unit sphere**, with no epipolar search and
no depth unknown. A single nearest-neighbour query replaces the per-pair
epipolar sweep.

That is the whole idea: un-project every keypoint in every image to a world
direction, drop all the directions into one KD-tree, and points that cluster
tightly on the unit sphere are infinite-point candidates. Descriptor distance
then confirms that co-directional keypoints are the *same* physical feature
rather than two unrelated things that happen to align.

### The `sfmtool-core` crate already implements un-projection

For image *i* with world→camera rotation `R_i` (from `quaternion_wxyz`) and a
keypoint at pixel `(u, v)`:

```
ray_cam   = camera.pixel_to_ray(u, v)     # unit ray in camera frame, all models
dir_world = R_iᵀ · ray_cam                # camera→world; unit because R is orthonormal
```

`CameraIntrinsics::pixel_to_ray` already handles every camera model including
fisheye beyond 180° (needed for the kerry_park rig), and `pixel_to_ray_batch`
is the vectorised form. `KdTree3d` is already exposed to Python. So the
machinery exists; this proposal is mostly about the clustering policy on top.

### "Distant" is the same query with a looser radius

The angular radius `ε` we cluster within is not just numerical slack. It is a
**physical knob for how distant a point must be to count as "at infinity."** A
finite point at distance `d`, viewed by two cameras separated by baseline `B`,
has parallax `≈ B/d`. Clustering directions within `ε` therefore captures the
points with parallax `≤ ε`, i.e. `d ≳ B/ε`. Tightening `ε` raises the distance
cutoff toward true infinity; loosening it sweeps in the "finite but distant"
points. They are the same search, and `ε` slides between them. (The prototype
prints `B_max/ε` next to each `ε` to make this cutoff concrete; see the findings
below.)

This also tells us what to *do* with a cluster once found. By construction its
members agree in direction to within `ε`. If their parallax is below the
keypoint-localisation noise floor, the depth is unrecoverable and the point is
genuinely `w = 0`. If `ε` is loose enough that a track's parallax clears the
floor, we triangulate it and decide per cluster: emit `w = 0` when the
triangulated result still fails the existing `α_max · f_max < noise` criterion,
otherwise emit a finite distant point (see Decisions).

## Approach

Build one KD-tree over all keypoint directions across all images, with a
parallel index array mapping each entry back to `(image_index, feature_index)`.
For each direction, query neighbours within the chord radius corresponding to
`ε`, keep neighbour pairs that (a) come from *different* images and (b) pass a
SIFT descriptor test, and assemble the surviving pairs into tracks. Assign each
track a direction by the bearing mean `normalise(Σ rᵢ)` (the same rule
`infinity.rs` already uses) and emit it as a `w = 0` point with a new track.

This needs only one global structure, `O(N log N)` to build and near-linear to
query, with no need to enumerate image pairs, and it naturally finds tracks
spanning many images at once. It reuses `pixel_to_ray_batch`, `KdTree3d`, the
descriptor L2 in `feature_match/descriptor.rs`, and the bearing-mean from
`infinity.rs`.

Tests show that mutual descriptor agreement and **at most one feature per image**
per track (a single infinite point cannot appear twice in one image) turn the
loose neighbour set into a clean cross-image track; without them, direction
coincidence is cheap enough that naive transitive grouping (union-find) chains
unrelated keypoints into runaway mega-clusters (see findings).

Other idea, not pursued here: exposing the existing
`classify_points_at_infinity` as its own `xform` flag is a cheap, composable
complement, but it only reclassifies already-triangulated points and finds
nothing new.

## Algorithm

1. **Un-project.** For every image, load its keypoints from the `.sift` file
   (`get_sift_path_for_image` + `SiftReader`), optionally capped to the largest
   `--max-features` per image, batch-un-project with `pixel_to_ray_batch`, and
   rotate to world with `R_iᵀ`. Accumulate `dirs (T,3)`, `descriptors (T,128)`,
   and the back-index `(image_index, feature_index)`.
2. **Build** one `KdTree3d` over `dirs`.
3. **Neighbour query** within chord radius `r = √(2(1−cos ε))` (this is the
   Euclidean distance on the unit sphere that corresponds to angular distance
   `ε`). Cap `k` per query.
4. **Pairwise confirm.** Keep a neighbour pair only if the two features come
   from different images, are **mutual** best descriptor matches within that
   image-to-image neighbourhood, and pass an L2 descriptor threshold (and
   ideally a Lowe ratio test against the second-best in the other image).
5. **Assemble tracks** from confirmed pairs with the **one-feature-per-image**
   constraint; drop tracks seen in fewer than `min_views` images (default 2,
   raise to 3 to suppress false positives).
6. **Classify.** Compute each track's parallax signal `α_max · f_max` in pixels.
   If it is below the noise floor the depth is unrecoverable, so the track stays
   `w = 0`. Otherwise triangulate it and apply the same `α_max · f_max < noise`
   criterion to the triangulated result: `w = 0` if it still fails, a finite
   distant point if it passes.
7. **Emit.** Each surviving track becomes a new point (direction
   `normalise(Σ rᵢ)` for `w = 0`, the triangulated position for finite), plus
   its observations, appended to the reconstruction via `clone_with_changes`.
   Deduplicate against existing tracks: if a candidate's observations already
   belong to one existing point, skip it (or reclassify that existing point in
   place instead of adding a duplicate).

Heavy lifting (steps 3–6) belongs in `sfmtool-core` behind a PyO3 entry point.
The prototype implements this policy in vectorised NumPy and it works, but the
final per-image-pair mutual matching is easier to get right (and parallelise) in
Rust next to the existing descriptor matchers, and avoids the prototype's
memory-hungry global edge arrays on the larger solves.

## CLI surface

Fits `sfm xform` as an ordered operation, consistent with the existing
filtering/optimisation ops:

```
sfm xform in.sfmr out.sfmr --find-points-at-infinity <eps_deg>[,<desc_thresh>[,<min_views>]]
```

e.g. `--find-points-at-infinity 0.1,300,2`. A `--max-features <N>` flag (the
standard cap many commands carry, taking each image's largest features) bounds
the per-image keypoint set: it caps memory and runtime on dense or many-image
solves, and the largest-scale features tend to be the most repeatable across the
wide viewpoint changes a distant point is seen under. The prototype caps at
2000/image. Optionally a companion `--classify-points-at-infinity
<noise_floor_px>` flag (reclassify existing points), which composes naturally
before or after. Because the operation *adds* points and tracks rather than
transforming existing geometry, note it as the first additive `xform`; the
`Transform.apply(recon) -> recon` protocol still holds. The op emits both kinds
of point on its own (per the classify step): a tight `ε` yields all `w = 0`, a
looser `ε` lets some tracks triangulate into finite distant points.

## Prototype findings

`tmp/infinity_search_prototype.py` (gitignored) runs **two** clustering policies
head-to-head so the false-positive cost is measurable:

- **NAIVE**: every cross-image neighbour pair under the descriptor threshold is
  an edge; transitive union-find. (The strawman.)
- **REFINED**: the recommended policy, per-image descriptor-best + Lowe ratio
  test + **mutual** match + **one-feature-per-image** tracks.

Both load all keypoints (capped 2000/image), un-project, share one `KdTree3d`,
and cross-check candidates against the existing solve: **%new** = tracks with no
member in any existing track (content the solve discarded); **%consistent** = of
tracks touching existing tracks, the fraction whose tracked members all belong
to a single existing point (agreement with COLMAP); **dirty%** = tracks with a
repeated image, which is impossible for one infinite point and so a pure-noise
tell.

Head-to-head (sweeping `ε` ∈ {0.1°, 0.5°}, descriptor threshold ∈ {200, 300}):

| dataset | ε | descT | method | cands | ≥3 img | biggest (#img) | dirty% | %new | %consistent |
|---|---|---|---|---|---|---|---|---|---|
| seoul_bull (17, indoor) | 0.5° | 300 | NAIVE | 4338 | 1797 | 7 | 60% | 69% | 65% |
| | | | REFINED | 3819 | 1065 | 6 | **0%** | 75% | **81%** |
| seattle_backyard (26, outdoor) | 0.5° | 300 | NAIVE | 5010 | 1544 | 16 | 17% | 91% | 93% |
| | | | REFINED | 4870 | 1309 | 15 | **0%** | 92% | **95%** |
| dino_dog_toy (85, turntable) | 0.5° | 300 | NAIVE | 4903 | 2111 | **69** | 54% | 61% | 63% |
| | | | REFINED | 11409 | 4168 | **14** | **0%** | 60% | **76%** |
| dino_dog_toy | 0.5° | 200 | NAIVE | 8944 | 3110 | **61** | 33% | 56% | 64% |
| | | | REFINED | 12796 | 4205 | **16** | **0%** | 59% | **74%** |
| kerry_park (48 fisheye, overlook) | 0.1° | 200 | NAIVE | 1067 | 119 | 12 | 2% | 85% | 97% |
| | | | REFINED | 1071 | 121 | 12 | **0%** | 86% | **97%** |
| kerry_park | 0.5° | 300 | NAIVE | 10716 | 5112 | 34 | 34% | 83% | 73% |
| | | | REFINED | 10308 | 4302 | **23** | **0%** | 84% | **85%** |

Context per dataset: scene radius p50 ≈ 5.0 / 5.5 / 1.7 / 20.8 world units; max
camera baseline ≈ 9.7 / 12.3 / 15.6 / 17.5; so `ε = 0.1°` ⇒ distance cutoff ≈
5500–10000 units (≈ 1000–5000× the scene radius, effectively infinite),
`ε = 0.5°` ⇒ ≈ 1100–2000 (~200–1000×, merely "distant").

What the numbers say:

- **The premise holds.** Every dataset yields co-directional, descriptor-
  consistent candidate tracks, and the large majority (**78–97%**) are *new*,
  not in the existing reconstruction. These are the low-parallax matches the
  default solve's 1.5° angle filter and 2-view drop threw away.
- **The refined policy is worth its cost.** Across the board it drives **dirty%
  to 0** (the one-per-image constraint) and lifts %consistent, most on the hard
  cases: seoul 65 to 81%, dino 63 to 76%, kerry 73 to 85% at `ε=0.5°, descT=300`.
  The effect is structural, not cosmetic: on the turntable the naive policy's
  "biggest track" is a **61–69-image** mega-cluster (a single
  descriptor-near-duplicate chain swallowing dozens of real tracks), which
  refined collapses to a sane **≤16 images**. The naive low candidate count at
  loose `ε` is an *artifact* of that swallowing: refined reports *more*
  candidates there because it splits the mega-clusters back into many clean,
  separate tracks.
- **Distant-content scenes are the sweet spot.** `kerry_park` (a scenic fisheye
  overlook of the Seattle skyline, genuinely distant content, scene radius p95
  ≈ 166) and `seattle_backyard` (outdoor) hold **95–97% consistency** with
  hundreds of multi-image tracks: distant landmarks tracked across many frames.
  The solve itself triangulated kerry_park's distant content as wildly far
  finite points (radius p95 ≈ 166 vs p50 ≈ 21), the ill-conditioned points that
  should be `w = 0`.
- **Fisheye works.** kerry_park is a back-to-back fisheye rig; `pixel_to_ray`
  un-projects it correctly (no special-casing needed), and the directions
  cluster cleanly. The approach is camera-model-agnostic.
- **Close-object scenes are the hazard, and the guardrails handle it.**
  `dino_dog_toy` (turntable, nothing actually distant) is where naive
  over-merges worst. Refined still produces 0 dirty tracks, but consistency
  caps at ~76%, so the operation should default to a **tight `ε`** (≈0.05–0.1°)
  and reward `min_views ≥ 3` to suppress the residual false positives on scenes
  with no real distant content.
- **`ε` is a distance dial, as predicted.** `~min dist = B_max/ε`: ≈5500–10000
  world units at `ε = 0.1°` (effectively infinite) down to ~1100–2000 at
  `ε = 0.5°` (merely "distant"). The "finite but distant" case is covered by
  loosening this one parameter.

## Decisions

- **`w = 0` vs distant-finite.** Decide per track from its parallax signal
  `α_max · f_max`. Below the noise floor the depth is unrecoverable, so emit
  `w = 0`. Above the floor, triangulate the track and judge the triangulated
  result against the same finite-vs-infinite criterion: keep `w = 0` if its noise
  still fails, otherwise emit a finite distant point. No separate flag; ε governs
  how many tracks reach the triangulation branch.
- **Mutual-match scope.** Per-image descriptor-best + ratio test + mutual edge,
  then transitive closure through mutual edges with a one-per-image constraint
  (validated by the prototype: 0 dirty tracks, higher consistency). When closure
  pulls two same-image features into one component via a chain, **split** rather
  than drop: keep the best feature per image, so a near miss does not discard an
  otherwise-good track.
- **Implementation.** The clustering, matching, and classification live in
  `sfmtool-core` Rust behind a PyO3 entry point, next to the existing descriptor
  matchers; the `xform` is a thin Python wrapper.
- **Out of `solve` for now.** Ship as an `xform`-only op. A later follow-up could
  run it as a supplementary augmentation pass after `solve`, recovering the
  low-parallax tracks the solve's own filters dropped.

## Reuse map

| Need | Existing piece |
|---|---|
| pixel → camera-frame unit ray (all models, fisheye) | `CameraIntrinsics::pixel_to_ray[_batch]` |
| camera → world rotation | `quaternion_wxyz`, `R_iᵀ` (`camera_to_world_rotation_flat`) |
| direction KD-tree, radius query | `KdTree3d` (PyO3) / `spatial.rs` `PointCloud3` |
| descriptor L2 / best-match | `feature_match/descriptor.rs` |
| all keypoints + descriptors per image | `get_sift_path_for_image` + `SiftReader` |
| bearing-mean direction for a track | `infinity.rs` (`normalise(Σ rᵢ)`) |
| emit new points/tracks | `SfmrReconstruction.clone_with_changes` |
| reclassify existing points | `classify_points_at_infinity` |
