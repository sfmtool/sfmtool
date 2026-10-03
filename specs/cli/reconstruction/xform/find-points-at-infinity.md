# Finding points at infinity in an existing solve

The `sfm xform --find-points-at-infinity` operation discovers points at
infinity in a reconstruction that has already been solved. It un-projects
every untracked keypoint to the world direction it would have if it were
infinitely far away, clusters those directions on the unit sphere, confirms
each cluster with SIFT descriptors, decides each resulting track with the
point-or-bearing likelihood-ratio test, and appends the tracks the test calls
bearings to the reconstruction as new `w = 0` points. It complements the
companion `--classify-points-at-infinity` operation, which decides the points
the reconstruction already has with the same test and so finds nothing new.

[v2 model]: ../../../formats/sfmr-file-format.md

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

The `.sfmr` format stores [homogeneous points][v2 model], so it can represent a
point at infinity directly. This document is about how the operation searches an
existing reconstruction for them, using one spatial data structure over all the
untracked keypoints.

## The geometric insight

A finite 3-D point is seen along a *different* world-space ray from each camera;
the rays converge on the point, and the angle between them (the parallax) is
what triangulation needs. Matching finite points across images therefore
requires epipolar search: given a feature in image *i*, its match in image *j*
lies somewhere along an epipolar curve, because its depth is unknown. That is
what `features/feature_match/` does today (rectified / polar sweep along epipolar
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

`CameraIntrinsics::pixel_to_ray` handles every camera model including fisheye
beyond 180° (needed for the kerry_park rig), and `pixel_to_ray_batch` is the
vectorised form; `KdTree3d` is exposed to Python. The un-projection is therefore
shared machinery, and what is particular to this operation is the clustering
policy on top of it.

### "Distant" is the same query with a looser radius

The angular radius `ε` the operation clusters within is not numerical slack. It
is a
**physical knob for how distant a point must be to count as "at infinity."** A
finite point at distance `d`, viewed by two cameras separated by baseline `B`,
has parallax `≈ B/d`. Clustering directions within `ε` therefore captures the
points with parallax `≤ ε`, i.e. `d ≳ B/ε`. Tightening `ε` raises the distance
cutoff toward true infinity; loosening it sweeps in "finite but distant"
candidates as well. They are the same search, and `ε` slides between them; the
calibration below reports `B_max/ε` beside each `ε` to make the cutoff
concrete.

A cluster's members agree in direction to within `ε`, which does not by
itself make them a point at infinity: whether a direction explains the rays as
well as a point does is a question about the measurement noise, not about `ε`.
Each cluster is therefore decided by the point-or-bearing test at the
reconstruction's measured noise. A bearing verdict is appended as a `w = 0`
point; a finite verdict, which a loose `ε` produces more of, is a distant
finite point and is left out (see Decisions).

## Approach

One KD-tree holds every keypoint direction across every image, with a parallel
index array mapping each entry back to `(image_index, feature_index)`. For each
direction the operation queries neighbours within the chord radius corresponding
to `ε`, keeps neighbour pairs that (a) come from *different* images and (b) pass
a SIFT descriptor test, and assembles the surviving pairs into tracks. Each track
the point-or-bearing test calls a bearing becomes a `w = 0` point, at the
test's closed-form bearing, with a new track.

That needs one global structure, `O(N log N)` to build and near-linear to query,
with no image-pair enumeration, and it naturally finds tracks spanning many
images at once. It reuses `pixel_to_ray_batch`, `KdTree3d`, the descriptor L2 in
`features/feature_match/descriptor.rs`, and the point-or-bearing test of
[batch-triangulation-api.md](../../../core/reconstruction/batch-triangulation-api.md).

Two guardrails turn the loose neighbour set into clean cross-image tracks: mutual
descriptor agreement, and **at most one feature per image** per track, since a
single infinite point cannot appear twice in one image. Without them, direction
coincidence is cheap enough that naive transitive grouping (union-find) chains
unrelated keypoints into runaway mega-clusters — measured below.

## Algorithm

1. **Un-project.** For every image, load its keypoints from the `.sift` file
   (`get_sift_path_for_image` + `SiftReader`), optionally capped to the largest
   `--max-features` per image, **skipping any keypoint already assigned to an
   existing 3D point** — discovery operates only on the features the solve left
   untracked. Batch-un-project the rest with `pixel_to_ray_batch` and rotate to
   world with `R_iᵀ`. Accumulate `dirs (T,3)`, `descriptors (T,128)`, and the
   back-index `(image_index, feature_index)`.
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
6. **Decide.** Each member keypoint's pixel becomes a ray and a 2×3 noise
   weight through `observation_ray`, at the per-axis pixel noise `σ`: the
   reconstruction's `reprojection_noise_px` (the RMS reprojection residual over
   its finite points' observations, measured before anything is appended),
   or the `sigma_px` the caller gives. A member whose pixel gives no ray
   (outside the camera model's domain, as at a fisheye's rim) is left out of
   the track, and a track left with fewer than `min_views` members is dropped
   as unscored. `decide_candidate_tracks` scores each track with
   `bearing_score_batch` and reads the verdict with `is_finite` at
   `DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD`, as reclassification does (see
   [batch-triangulation-api.md](../../../core/reconstruction/batch-triangulation-api.md)
   § "Consumers"). A bearing verdict whose bearing is in front of every
   observing camera is kept; a finite verdict and a bearing behind a camera
   are dropped.
7. **Emit.** Each kept track becomes a new `w = 0` point at the score's
   closed-form bearing, plus its observations, appended to the reconstruction.
   Each new point is assigned its mean reprojection error, measured inline
   against the member keypoints it was built from: the bearing projects
   through rotation + intrinsics only, so downstream filters (e.g.
   `--filter-by-reprojection-error`) score discovered points like any other.
   Because step 1 excluded already-tracked keypoints, no appended observation
   reuses a feature an existing point already owns — a 2D feature still observes
   exactly one 3D point, which COLMAP export and bundle adjustment require.

Steps 2–6 run in `sfmtool-core` behind a PyO3 entry point rather than in Python.
The policy is expressible in vectorised NumPy — the calibration below was
measured that way — but the per-image-pair mutual matching is easier to get right
and to parallelise in Rust next to the existing descriptor matchers, and Rust
avoids the memory-hungry global edge arrays a NumPy form needs on larger
solves.

## CLI surface

The clustering, matching and decision live in
[discover.rs](../../../../crates/sfmtool-core/src/analysis/infinity/discover.rs),
exposed as the PyO3 method `SfmrReconstruction.find_points_at_infinity` and
driven by the thin Python transforms in
[_find_points_at_infinity.py](../../../../src/sfmtool/xform/_find_points_at_infinity.py).

```rust
pub struct InfinityParams { pub eps_deg: f64, pub desc_thresh: f64, pub ratio: f64, pub min_views: usize }
pub struct InfinityTrack { pub members: Vec<(u32, u32)> }   // (image, feature), one per image

/// The candidate tracks, sorted by members. Pure: no .sift reads.
pub fn find_infinity_tracks(dirs: &[Vector3<f64>], descriptors: &[[u8; 128]],
    image_index: &[u32], feature_index: &[u32], params: &InfinityParams) -> Vec<InfinityTrack>;

pub enum CandidateDecision { Bearing(Vector3<f64>), Finite, BearingBehindCamera, Unscored }
/// One decision per track of `rays`, from bearing_score_batch and is_finite.
pub fn decide_candidate_tracks(rays: &RayBatch) -> Vec<CandidateDecision>;

pub struct InfinityDiscovery {
    pub sigma_px: Option<f64>,            // the caller's, or measured; None: nothing searched
    pub noise: Option<ReprojectionNoise>, // the measurement, when measured
    pub candidates: usize,
    pub bearings: usize,                  // appended
    pub short_baseline: usize,            // of those, unresolvable at the camera extents
    pub finite: usize,                    // dropped
    pub bearing_behind_camera: usize,     // dropped
    pub unscored: usize,                  // dropped
}

impl SfmrReconstruction {
    pub fn find_points_at_infinity(&self, eps_deg: f64, desc_thresh: f64, ratio: f64,
        min_views: usize, max_features: Option<usize>, sigma_px: Option<f64>)
        -> Result<(Self, InfinityDiscovery), PointOrBearingError>;
}
```

```rust
let (found, summary) = recon.find_points_at_infinity(0.1, 200.0, 0.8, 2, None, None)?;
println!("{} of {} candidates appended at {:?} px", summary.bearings,
    summary.candidates, summary.sigma_px);
```

The clustering is a function of its own so it can be tested without `.sift`
files, and the decision is one so that a caller holding rays from elsewhere
reads the same verdicts. With no `sigma_px` and no observation of a finite
point to measure one from, nothing is searched and the reconstruction comes
back unchanged with `sigma_px: None`, as reclassification does. A `sigma_px`
that is not finite and positive is `InvalidNoiseLevel`, and an
`embedded_patches` reconstruction (no `.sift` files to search) or an
unreadable `.sift` file is `Reconstruction`. The binding returns
`(reconstruction, summary)`, the summary a dict of the same fields
(`noise` as `reprojection_noise()`'s dict), and raises `ValueError` and
`OSError` for the two errors.

It is an ordered `sfm xform` operation, consistent with the existing
filtering and optimisation ops:

```
sfm xform in.sfmr out.sfmr --find-points-at-infinity <eps_deg>[,<desc_thresh>[,<min_views>[,<sigma_px>]]]
```

e.g. `--find-points-at-infinity 0.1,200,2` (defaults: `desc_thresh` 200,
`min_views` 2, `sigma_px` measured from the reconstruction; a value given must
be finite and positive). A fourth component written for older versions, which
read it as a keypoint noise floor for the inverse-depth z rule, is read as `σ`:
on the kerry_park solve, `0.5,300,2,1.0` weights the rays at 1 px instead of
the measured 0.231 px and appends 3,067 bearings instead of 2,829. The operation prints the noise level and where it
came from, the candidate and appended counts, the `short_baseline` count when it
is not zero, and each kind of drop that occurred. The Lowe ratio of the pairwise
confirm step is fixed at 0.8 by the Python transform and is not exposed on the
command line; only the core function and its PyO3 method take it.
`--max-features <N>` — the standard cap many commands carry, taking each image's
largest features — bounds the per-image keypoint set. It caps memory and runtime
on dense or many-image solves, and the largest-scale features tend to be the most
repeatable across the wide viewpoint changes a distant point is seen under.
Uncapped by default; `2000` is the value the calibration below used. The flag
applies to this operation alone and is rejected when no
`--find-points-at-infinity` is present.

The companion `--classify-points-at-infinity [<sigma_px>]` reclassifies
the points the reconstruction already has with the point-or-bearing test (see
[batch-triangulation-api.md](../../../core/reconstruction/batch-triangulation-api.md)
§ "Consumers"), and composes before or after this one. Run after it at the
measured noise, it changes none of the points discovery appended: bearings do
not enter the noise measure, so the reconstruction measures the same `σ`
after discovery as before, and a bearing's rays are built from the same pixels
at the same `σ`, so the score and its verdict are the same.

The operation is *additive*: it appends new points and tracks through the
`Transform.apply(recon) -> recon` protocol, and every point it appends is
`w = 0`.

## Calibration

The two guardrails above are what separate this from a much simpler policy, and
their cost was measured by running both head to head over the four in-tree
datasets:

- **NAIVE**: every cross-image neighbour pair under the descriptor threshold is
  an edge; transitive union-find. The strawman.
- **REFINED**: the shipped policy — per-image descriptor-best, Lowe ratio test,
  **mutual** match, **one-feature-per-image** tracks.

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
  caps at ~76%. `ε` has no default — it is the operation's one required
  argument — and this is why: on a scene with no genuinely distant content, a
  **tight `ε`** (≈0.05–0.1°) together with `min_views ≥ 3` is what suppresses the
  residual false positives.
- **`ε` is a distance dial, as predicted.** `~min dist = B_max/ε`: ≈5500–10000
  world units at `ε = 0.1°` (effectively infinite) down to ~1100–2000 at
  `ε = 0.5°` (merely "distant"). Loosening this one parameter brings in the
  "finite but distant" candidates, and the point-or-bearing test decides which
  of them are bearings.
- **A loose `ε` costs bundle adjustment.** On the kerry_park solve, bundle
  adjustment after discovery at `0.5,300,2` (2,829 bearings) ends without
  converging. Its camera rotations differ by a median 0.21° from the input
  poses and 0.231° from those of bundle adjustment with no discovery (which
  itself moves them 0.06° from the input); its camera centres move by about
  1 unit (6% of the camera extent); and it raises `σ` from 0.204 to 0.236 px.
  After discovery at `0.1,200,2` the rotations differ by a median 0.059° from
  the input poses and 0.017° from the no-discovery adjustment, which is
  negligible. A loose `ε` finds more candidates, but follow it with bundle
  adjustment only after checking that the poses hold.

## Decisions

- **Decide on the point-or-bearing test, at the measured noise.** The verdict
  is the one reclassification reads, with the rays weighted at the
  reconstruction's `reprojection_noise_px`, so discovery and reclassification
  agree on every track they both see. (Before, discovery decided on the
  inverse-depth z rule at a 1 px noise floor and dropped an *indeterminate*
  middle; on a `sift_files` solve of seoul bull at `0.5,300,2`, 56 of the 505
  points at infinity it appended had finite verdicts at the measured 0.36 px,
  which a reclassification pass then promoted.)
- **Append bearings only.** A finite verdict means the rays ask for a depth,
  which is a finite point, and finite points are the solve's: discovery leaves
  them out. Appending them was measured to disturb the noise level every later
  decision reads. On a 48-image `sift_files` kerry_park solve at `0.5,300,2`,
  appending the 239 finite verdicts (238 of which the point fit could place)
  at their fitted points raised the measured
  `σ` from 0.231 to 0.318 px (they were marginal: at 0.318 px, 106 of them and
  3 points of the solve itself scored under the threshold), so a
  reclassification pass after discovery would have demoted 109 points. With
  bearings only, the measured `σ` is unchanged and reclassification changes
  nothing. What the drop loses depends on the capture. On the seoul bull
  solve at `0.5,300,2` the 60 finite candidates are sound: placed at their
  fitted points, their RMS reprojection error is 0.484 px against the solve's
  0.509, adding them leaves `σ` unchanged, and they are mostly 2-view tracks
  at a distance of about 147 against a median scene distance of 5.2. On
  kerry_park the 239 finite verdicts, of which 238 could be placed, are
  marginal: 0.645 px against 0.327, with the effect on
  `σ` above. Whether a quality gate could keep the consistent finite
  candidates is an open question of the
  [amendment draft](../../../drafts/point-or-bearing-likelihood-ratio.md).
- **A track with a short baseline is appended as a bearing.** When the
  observing cameras are too close together to tell a point at the capture's own
  scale from infinity (`resolvable_distance` under the camera extents, at `σ`
  over the focal length), the score finds no depth to ask for, `Λ ≈ 0`, and the verdict
  is a bearing. Discovery appends it. The direction explains every sighting to
  within the noise, which is all a bearing claims; reclassification stores such
  a track as a bearing, so dropping it here would make the two passes disagree
  about what one set of rays earns; and the reason the z rule dropped these
  tracks was that `inverse_depth_z` divides by a solved depth that is noise
  when the rays are near-parallel, so it could call a near short-baseline
  track finite, which the score cannot do. Such a bearing describes the
  sightings that exist; a later image from a camera further to the side would
  show its depth, and reclassification would then promote it. The summary
  counts these bearings as `short_baseline`. On the kerry_park solve at
  `0.1,200,2`, the z rule dropped 174 tracks as indeterminate; the test appends 172 of them as bearings (the
  other 2 have too few usable rays), and at the measured 0.231 px only 17 of
  the 683 bearings appended are under the capture-scale gate (172 at 1 px: the
  floor overstated the noise 4.3-fold). At `0.5,300,2`, 484 of 493 are
  appended and 56 of 2,829 bearings are under the gate. The seoul bull solve
  has none.
- **A member without a ray is not appended.** The test reads only the members
  whose pixel gives a ray, so those are the members the appended track holds.
  On kerry_park the pixels that give none are keypoints more than about
  203 px from the centre of the 480 px fisheye images. That is where the
  un-projection's wide-angle blend starts for both solved `OPENCV_FISHEYE`
  cameras (90° of distorted radius, 202.2 to 205.0 px depending on the camera
  and the axis; `trustworthy_max_theta_deg` reports the incidence angle it
  corresponds to), and the appended observations reach 204.2 px. Past it
  `pixel_to_ray` deliberately moves toward the equidistant ray, which
  `ray_to_pixel` does not map back (for camera 0 along x the round trip
  misses by 0 px at 202 px, 0.02 px at 204, 2.5 px at 210 and 7 px at 220),
  so `observed_ray` declines the pixel, as it should. The solved
  lens model's own fold is farther out, at about 102° (231 px). At
  `0.5,300,2` this drops 418 candidates outright and 104 of the 7,193
  observations of the tracks appended.
- **Mutual-match scope.** Per-image descriptor-best + ratio test + mutual edge,
  then transitive closure through mutual edges with a one-per-image constraint
  (0 dirty tracks and higher consistency, measured below). When closure
  pulls two same-image features into one component via a chain, **split** rather
  than drop: keep the best feature per image, so a near miss does not discard an
  otherwise-good track.
- **Implementation.** The clustering, matching, and classification live in
  `sfmtool-core` Rust behind a PyO3 entry point, next to the existing descriptor
  matchers; the `xform` is a thin Python wrapper.
- **Out of `solve`.** It is an `xform`-only operation. Running it as a
  supplementary augmentation pass inside `solve`, to recover the low-parallax
  tracks the solve's own filters dropped, would be a larger change: `solve`
  drives pycolmap, and the discovery would have to fit between the mapper's
  passes rather than after them.

## Reuse map

| Need | Existing piece |
|---|---|
| pixel → camera-frame unit ray (all models, fisheye) | `CameraIntrinsics::pixel_to_ray[_batch]` |
| camera → world rotation | `quaternion_wxyz`, `R_iᵀ` (`camera_to_world_rotation_flat`) |
| direction KD-tree, radius query | `KdTree3d` (PyO3) / `spatial.rs` `PointCloud3` |
| descriptor L2 / best-match | `features/feature_match/descriptor.rs` |
| all keypoints + descriptors per image | `get_sift_path_for_image` + `SiftReader` |
| a candidate's rays and weights | `analysis/point_or_bearing.rs` (`observation_ray`) |
| the verdict and the bearing | `bearing_score_batch`, `is_finite` |
| the noise level | `SfmrReconstruction::reprojection_noise` |
| emit new points/tracks | `SfmrReconstruction.clone_with_changes` |
| reclassify existing points | `classify_points_at_infinity` |
