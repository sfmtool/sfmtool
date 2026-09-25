# Resect Image

An action of the image menu that re-estimates one image's pose against the rest
of its reconstruction and installs the answer as that reconstruction's next
**version**, which an undo steps back out of. Its 2D-3D correspondences are the
reconstruction's tracks and, beside them, the clusters of the node's
cluster-patches index file, each cluster used as a track of its own.

The shared primitive underneath takes a **set** of target images and holds all
of them out together, so a group whose members corroborate each other is
questioned as a group; the viewer's action is that primitive on a one-element
set.

Related specs: [../scene-graph.md](../scene-graph.md) (nodes, context
menus, `Align to…` as the template for a per-node action),
[../index-files.md](../index-files.md) (the cluster-patches file and its
none / current / stale state),
[../../core/geometry/reconstruction-growth.md](../../core/geometry/reconstruction-growth.md)
(`resect_images_batch`, the registration primitive whose consensus floor,
refinement schedule and 3 px bound the finite path shares),
[../document-model.md](../document-model.md) and
[../edit-history.md](../edit-history.md) (the version it pushes, and the map the
selection follows), [README.md](README.md) (the other edit families),
`seed-candidate-evaluation` (not yet written)
(the hold-out self-resection channel; same mechanism, read offline).

---

## Purpose

Whether one camera's pose is corroborated by the rest of the reconstruction is
a question the stored pose cannot answer: it was fit jointly with the points it
observes, so it always agrees with them. Re-estimating the pose from structure
that did **not** depend on that image is what answers it, and installing the
answer is what acts on it: the frustum moves, the points the image observes
re-triangulate, and the point track detail shows where the image's observations
land under the new pose. `Ctrl+Z` puts the stored pose back when the answer is
not the better one.

---

## Invocation

`Resect Image` is the first entry of the **image menu**, the context menu of one
image that opens on an image row of the Scene Graph tree (the rows under a
reconstruction's `Camera Images` group) and on a thumbnail of the Image Browser
strip ([../scene-graph.md](../scene-graph.md) § "Image menu"). The image's
reconstruction is the source node; the image is the target. Over MCP the same
step is `resect_camera_image` ([../mcp-server.md](../mcp-server.md)).

The action needs the node's **current** cluster-patches file
(`<stem>-cluster-patches.matches`, [../index-files.md](../index-files.md)). The
entry is greyed out, with a hover explanation, for the first of these that
holds:

1. the image is not posed in its reconstruction;
2. the source has fewer than three other posed images;
3. the node's cluster-patches file is missing (state `none`) or out of date
   (state `stale`). The explanation names the state, gives the stale file's own
   reason, and ends with what makes a current file: the Index Files entry under
   its present name (`Build Index Files` or `Rebuild Index Files`), or, for a
   node with no path on disk, that it has to be saved first.

The step and the MCP tool refuse with the same sentence
(`AppState::resect_image_refusal` in
[image_menu.rs](../../../crates/sfm-explorer/src/image_menu.rs)). A missing or
stale file is never replaced by the tracks alone: the rule is the same for every
node, whether its observations carry feature indexes or embedded patches.

---

## Mechanism

Everything below `## Invocation` lives in `sfmtool-core` as one function,
`geometry::resect_images` in
[resect_images.rs](../../../crates/sfmtool-core/src/geometry/resect_images.rs); the
GUI wraps it in [resect.rs](../../../crates/sfm-explorer/src/resect.rs), which adds
the refusal for a missing or stale file, the version and the status line. The
same function is what an offline caller uses, with as many targets as it wants
to hold out, through the `sfmtool._sfmtool.geometry.resect_images` binding.

```rust
pub enum ResectSource<'a> {
    Tracks,
    TracksAndClusters(&'a MatchesData),
}

pub fn resect_images(
    recon: &SfmrReconstruction,
    image_indexes: &[usize],
    source: ResectSource<'_>,
    options: &ResectImageOptions,
) -> Result<ResectedImages, ResectImageError>;

// The viewer's call: one image, the node's cluster-patches file.
let data = sfmtool_matches_format::read_matches(&cluster_patches_path)?;
let (next, report) = resect_image_in_place(
    &recon,
    image,
    ResectSource::TracksAndClusters(&data),
    &ResectImageOptions::default(),
)?;
```

The source is the tracks alone or the tracks plus one cluster-patches file,
because those are the two things a caller can have in hand: every
reconstruction has tracks, and the file is optional input. `TracksAndClusters`
refuses the whole call (`ResectImageError::Clusters`) when the file lacks the
clusters section, the cluster-patches section or the member positions.

The function takes a **target set**: one or more images of the source, named as
a set rather than resected one after another. Every step below is over that
set; the viewer's action passes a single image, which is the set of size one.

The whole call refuses, producing nothing, when the set is empty, names an
image twice, names an image that is not posed, or leaves fewer than three
**non-target** posed images behind. Nothing else fails the call: an outcome
that belongs to one target is that target's refusal.

### 1. Clone

The source reconstruction is deep-copied. All work happens on the copy.

### 2. Held-out structure

The whole target set's contribution to structure is removed before any pose is
estimated. "Non-target" below means a posed image that is not in the set.

- Every finite point any target observes that retains at least two non-target
  observations is **re-triangulated from those non-target observations only**,
  at their stored poses. The stored position is discarded for this purpose.
- A finite point with fewer than two non-target observations has no held-out
  position. It is excluded from every estimate.
- A point at infinity is a direction, which one rotation already fixes, so its
  held-out bearing is the mean of the world rays the non-target images see it
  along. A point at infinity no non-target image observes has no held-out
  bearing.

A point two targets share is therefore re-triangulated from neither of them:
holding a set out together asks whether the group is corroborated by the rest,
not whether each member is corroborated by the others.

The re-triangulated positions are used for the pose estimates and are **kept**
in the resected reconstruction (they are what the non-target images say about
those points). Points at infinity keep their stored directions; the held-out
bearings are the estimate's input only.

### 3. Pose estimates

Each target is estimated against that one shared held-out structure, and
accepted or refused on the primitive's own gate (`accept_gate`) independently
of the others.

- **Finite path.** A target's correspondences are its track pairs and its
  cluster pairs together (see "Correspondence sources"). A track pair is
  **finite** when its point has a held-out position, and a **bearing** when its
  point is at infinity and has a held-out bearing. A target with at least
  `ResectOptions::min_obs` finite pairs (finite tracks and clusters; bearings
  do not count toward it) is estimated by the finite path in
  [resect_images/finite.rs](../../../crates/sfmtool-core/src/geometry/resect_images/finite.rs):
  - **Residuals.** Every pair is scored in pixels through the image's own
    camera model. A finite pair's residual is its reprojection distance. A
    bearing's residual is the angle between its held-out direction, rotated
    into the camera, and the ray the target observed it along, times the
    camera's focal length (the larger of the two): the currency the
    rotation-only path uses. A bearing therefore constrains the rotation and
    not the translation.
  - **Minimal samples.** RANSAC P3P draws its three-pair samples from the
    **finite track pairs** when the target has at least three of them, and
    from every finite pair (tracks and clusters) when it has fewer. Bearings
    are never sampled: P3P needs three finite points. A pool with at most
    2000 triples is enumerated whole; a larger one is sampled with a SplitMix64
    seeded from `(seed, image index)`, for at most 2000 trials, stopping once
    an all-inlier triple has been drawn with 0.999 probability at the best
    inlier rate seen among the pool.
  - **Scoring.** Each hypothesis is scored over **every** pair, tracks,
    bearings and clusters alike: the count within the 3 px bound, ties broken
    by the sum of the residuals capped at 3 px. A new best hypothesis is refit
    once on its own inliers and kept refit when that scores better. Sampling
    from the tracks means every hypothesis fits three tracks exactly, so a pose
    under which the tracks are outliers is never proposed, however many
    clusters agree with it; the clusters and bearings choose between the poses
    the tracks allow and sharpen them. Below three finite tracks, the tracks
    are sampled with the clusters rather than set aside. No pair is filtered
    for being a track.
  - **Refinement.** A best hypothesis with fewer than 8 inliers (the
    batch-registration primitive's P3P consensus floor) refuses the estimate.
    Otherwise its inliers are refit by trimmed Levenberg-Marquardt, five
    rounds each keeping the best-fitting 60% of them, and the result is refit
    once more on every pair within 3 px when at least six are. A bearing's
    residual in the fit is `f · (ray × R·d)`, whose length is `f · sin(angle)`,
    so it has a derivative at zero angle.
  - The estimate is accepted when its inliers over all pairs are at least
    `accept_gate` of them. Each target runs on its own, seeded from its own
    index, so running targets in parallel changes no answer.
- **Rotation-only path.** Below that floor, or when the reconstruction is
  rotation-only, the rotation is estimated by closed-form absolute
  orientation between the target's observed ray directions and the held-out
  bearings of the points at infinity it observes (trimmed, iterated), and the
  translation is left at its stored value. It reads the tracks' bearings only;
  clusters are finite correspondences and do not reach it. Requires at least
  three bearings,
  spanning an angle the camera can resolve — the largest angle any bearing
  makes with their mean direction has to exceed one pixel's worth of angle at
  the camera's own focal, since a spread narrower than that is not a spread.
  The inlier bound is the finite path's 3 px bound in the same currency: the
  angle a pixel subtends on this camera.
- **Refusals.** A target that misses the gate, whose best pose hypothesis has
  fewer than 8 inliers, whose bearings span no resolvable angle, or that has
  support on neither path, is refused: it keeps
  its stored pose, its report carries the reason, and the rest of the set
  proceeds. The call still hands back its reconstruction, carrying the refused
  targets' stored poses and the held-out re-triangulation; what the viewer's
  single-target action does with that is step 6.

### 4. Re-triangulation at the new poses

With the accepted targets at their resected poses, every finite point an
accepted target observes is re-triangulated from **all** its observations,
including the targets' own. Points no accepted target observes are left at
their held-out positions, and points the set does not observe at all are
untouched. Clusters are not re-triangulated and create no points: they change
the result only through the poses. No bundle adjustment runs: the point of the
action is to show what the resection alone says, not what a joint refit would
smooth over.

A point fails re-triangulation when fewer than two observations survive, when
the solve puts it behind one of the cameras that observe it, or when its depth
is not observable at all (parallel rays, which leaves the triangulation's normal
matrix rank-deficient — the parallax floor stated in the solve's own
diagnostics rather than as a separate angle threshold). A point that fails keeps
its held-out position from step 2 when it has one, and is otherwise removed with
its observations.

### 5. Result

The resected reconstruction differs from the source only in the accepted
targets' poses and in the points the set observes. Its metadata records
`operation = "explorer_resect"`, the targets' relative paths, the correspondence
source, and the estimates' inlier fractions, so a later save carries
provenance.

### 6. Installing the answer

`sfmtool_core::geometry::resect_image_in_place` in
[resect_images.rs](../../../crates/sfmtool-core/src/geometry/resect_images.rs) is
`resect_images` on the one-element target set plus the one rule a caller
installing the answer needs: **a refused estimate yields no value.** The set
call keeps such an answer, because a held-out re-triangulation is worth looking
at beside the reconstruction it came from; installed *as* that reconstruction it
would be a version that moved the points and left the pose alone. So a refusal
pushes no version, and is reported as a failure.

This is the edit family of [README.md](README.md) and it is a **bulk** edit
([../document-model.md](../document-model.md), "Two kinds of edit"): a resection
re-poses an image and re-triangulates the points it observes, so the next
version is a whole new base with an empty overlay.

The viewer's wrapper is `AppState::resect_image` in
[state/edits.rs](../../../crates/sfm-explorer/src/state/edits.rs), which
re-derives the node's index-file states, refuses with
`AppState::resect_image_refusal` when the entry would be greyed, reads the
cluster-patches file, materialises the current value when its overlay is not
empty, runs the core function over it with `TracksAndClusters`, and pushes the
result with the row map `RowMap::by_scan` reads
off that call's input and output: the resection may drop a point it can neither
re-triangulate nor hold out, and says nothing about which. The selection follows
that map. The image table does not move, so image indexes, the image and camera
selections and the decoded pixels keyed by them all still mean what they meant;
what the panels cached *about* the geometry -- rendered patches, prepared track
rows, per-camera derived quantities -- is dropped, because it describes a
geometry the node no longer holds.

The same edit is available offline as
`EditedReconstruction.resect_image_in_place(image, cluster_patches_path=…)`,
which returns `(EditedReconstruction, report)` and raises on the refusal.

### Correspondence sources

A target's correspondences are the **union of two sets**, handed to the pose
estimate side by side. The two are never joined or merged: no cluster is
matched to a track, nothing is deduplicated, and a cluster that sees the same
physical point as a track is simply one more correspondence, scored and refit
like any other. The two sets differ in one place: with at least three finite
track pairs, only the tracks are drawn as RANSAC's minimal samples (step 3).
The tracks lead and the clusters support them.

- **Tracks** (both sources): the target's own observations, joined to the
  held-out positions and held-out bearings of step 2. A point any target
  observes is scored only against its held-out position or bearing, never the
  stored one it helped fit, and a point with neither contributes nothing.
- **Clusters** (`TracksAndClusters`): each cluster of the cluster-patches file,
  used as a track of its own. Its members are joined to the reconstruction's
  images by image name; a member in an image the reconstruction does not hold
  is ignored. Only **kept** members count — the cluster's `Reference` and the
  members refinement marked `Kept` — since a rejected or unevaluated member is
  not a claim about the world. A cluster gives a target one pair when:
  1. it has **exactly one** kept member in the target. A cluster with two or
     more does not say which of its pixels the point is at, and contributes
     nothing to that target;
  2. it has kept members in at least two **non-target** posed images **that
     have tracks**. Members in any target image never count, so the cluster
     is held out from the whole target set exactly as a track is. A member in
     an image with no track observation does not count either; see "Images
     without tracks" below. A cluster with members in two or more non-target
     posed images but in fewer than two with tracks is counted in
     `clusters_untracked`;
  3. those counted members, as rays through their **refined positions** at
     their images' stored poses, triangulate under step 2's rules: at least two
     usable rays, in front of every camera, and an observable depth. A cluster
     that fails contributes nothing;
  4. the triangulated position **agrees with its own members**: projected into
     each counted member's image at its stored pose, it lands within
     `ResectImageOptions::max_cluster_residual_px` (default 1.5 px) of that
     member's refined position. A position behind a member's camera (judged
     along the member's ray, as the triangulation judges "in front"), or one
     that projects outside the member's frame, does not agree. A cluster that
     fails is dropped whole and contributes nothing; no member is removed and
     the rest re-triangulated.

  The pair is the target member's refined position against that triangulated
  position. A cluster reaching several targets is triangulated once.

The fourth rule exists because a cluster's members can agree pairwise in
appearance and still not be one point in space: refinement keeps a member on
its patch match, not on the geometry. A cluster whose own members do not meet
at one position gives a position the target's pixel has no reason to agree
with. On the Kerry Park reconstruction (48 fisheye images, 14532 clusters),
scored against the stored poses of the 42 images that have tracks, 51.5% of
the clusters that triangulate put their target member within 3 px of the
stored pose; of those whose own worst member residual is at most 1.5 px, 88.5%
do. On the seoul_bull ground truth (17 images) the same numbers are 70.3% and
92.3%.

The threshold and the form were chosen on those two reconstructions, each image
resected alone through `resect_images`:

| Threshold (px) | Kerry clusters kept | within 3 px | seoul_bull kept | within 3 px | Kerry accepted |
|---|---|---|---|---|---|
| 0.5 | 3125 | 0.898 | 1611 | 0.935 | 45/48 |
| 1.0 | 3830 | 0.894 | 1737 | 0.926 | 48/48 |
| 1.5 | 4223 | 0.885 | 1754 | 0.923 | 48/48 |
| 2.0 | 4444 | 0.878 | 1772 | 0.915 | 48/48 |
| 3.0 | 4820 | 0.861 | 1792 | 0.908 | 48/48 |
| no rule | 9061 | 0.515 | 2319 | 0.703 | 42/48 |

"Kept" and "within 3 px" count the targets with tracks; "accepted" counts every
image, and every image with tracks is accepted at every threshold on both
reconstructions. Kerry has six images with no tracks, which the clusters alone
have to place. With no rule all six are refused; at 0.5 px three are refused,
keeping only 6 to 13 clusters each; from 1 px up all six are accepted. At 1 px
one of them (`fisheye_left/frame_24`) is accepted on 5 inliers of 14 with a
5.0° rotation away from its stored pose, against 0.8° at 1.5 px, and the 90th
percentile of the camera-centre move over the images with tracks drops from
0.0096 to 0.0057 scene units between 1 and 1.5 px. Above 1.5 px the agreement
keeps falling while those images gain few clusters. Removing the worst member
and re-triangulating instead of dropping the cluster keeps about 1.5 times as
many Kerry clusters, but only 78% of them agree with the stored pose at 1.5 px
and the images with tracks move further from their stored poses (median
rotation 0.12° against 0.10°), so the cluster is dropped whole. These
measurements count the members in every non-target posed image and sample the
tracks and the clusters together; the two rules below were measured on their
own.

#### Images without tracks

An image **has tracks** when at least one track observation of the
reconstruction is in it, whatever the point. A member in an image without
tracks does not count because nothing corroborates that image's pose: no
point it observes was triangulated together with any other image. Such images
can hold poses that agree with each other and with nothing else, and the
clusters they place then agree with each other too, however wrong the poses
are. One observation is the bar because it is the least that ties an image's
pose to the rest of the reconstruction; a higher count would be a threshold
with no measurement to choose it.

On the Kerry Park reconstruction five images, the left and right frames 22 to
24 other than `fisheye_right/frame_22`, have no tracks, and their stored poses
agree with each other and are far from the rig's path. With their members
counted, `fisheye_right/frame_22`, which has four finite tracks and one
bearing, was resected onto them: 0 of 4 tracks and 33 of 53 clusters were
inliers, with a 13.0° rotation and an 18.7 scene-unit move. With them not
counted and the tracks leading, 3 of 4 finite tracks, the bearing and 16 of 23
clusters are inliers, and the camera centre moves 0.056 scene units. Over the
42 other images with tracks the median rotation away from the stored pose is
0.088° (0.095° with those members counted and all pairs sampled together) and
the largest camera-centre move is the same, at
0.040 scene units; on the seoul_bull ground truth, where every image has
tracks, no cluster is affected and all 17 images stay within 0.14° and 0.0033
scene units. Every accepted image with at least three finite tracks keeps at
least three quarters of them as inliers on both reconstructions, so the
estimate carries no further rule requiring a majority of the tracks.

Clusters need no feature indexes and no position matching, so they work the same
on `sift_files` and `embedded_patches` reconstructions. A target keypoint that is
a cluster member but no observation of the reconstruction contributes to the
estimate and creates no track.

### Reported quantities

Each target gets its own report: the path taken; the correspondences the
estimate saw and how many were inliers, each split into the part from the
tracks and the part from the clusters, with the track part's bearings
(`bearing_correspondences`, `bearing_inliers`) counted within it; how many
clusters had a kept member in the image (`clusters_considered`), how many of
those the member rules set aside (`clusters_skipped`), how many were left with
members in fewer than two tracked images (`clusters_untracked`), how many
failed to triangulate (`clusters_failed`) and how many triangulated to a
position their own members disagree with (`clusters_inconsistent`); whether it
was accepted and why not; how far its pose moved; and its share of the
held-out, re-triangulated and removed points. On the rotation-only path the
correspondences are the tracks' bearings and the cluster counts of pairs and
inliers are zero. Over the set there are totals: how many targets were accepted
and refused, the summed correspondences and inliers with their ratio and with
their split by source and bearings, the summed `clusters_untracked` and
`clusters_inconsistent`, and the held-out, re-triangulated and removed point
counts with each point counted once however many targets observe it.

The rotation delta is the angle between the stored and resected world-to-camera
rotations and the translation delta is the distance the camera **centre** moved,
in units of the source's median-over-images of that image's median
camera-to-structure distance (the same unit the evaluation channels use). A
rotation-only reconstruction has no such distance, so it reports the
displacement in its own units and no ratio.

---

## The version

- **Label**: `Resected <image basename> (<node label>)`.
- **Action Log**: one entry of kind `Edit`, the label, the estimate's own
  quantities, and the serials:
  `Resected IMG_0007.jpg (bull): 214 pts (147 finite tracks, 3 tracks at
  infinity, 64 clusters), inliers 198/214 (0.93; 137 finite tracks, 3 tracks at
  infinity, 58 clusters), rotation 12.40°, translation 0.081 (scene-scale), 190
  re-triangulated; clusters 80 considered, 12 skipped, 2 untracked, 4 failed to
  triangulate, 6 inconsistent (v3 → v4)`. A refusal is one **failed** entry:
  `Resect <image> in <node> refused: <reason>`. One entry either way, and it is
  also what the status line (viewport overlay, as `Align to…` reports) shows.
- **Selection**: unchanged, but followed through the map: the node is the one
  already on screen, and the image the menu was opened on is where it was. A
  point the resection dropped clears the point selection.

---

## Performance

Held-out re-triangulation and the estimates touch only the observations of the
points the target set observes (hundreds to a few thousand rows for one
target). The finite path scores each of at most 2000 minimal samples against
every pair of its target, and the targets run in parallel. The viewer's
single-target action runs synchronously in tens of milliseconds on the
reconstructions the viewer targets. It reads the cluster-patches file on every
run, as *Create Track Here* does, so the file on disk is always the one used;
the read is its own row (`read cluster patches`) in the entry's timing. The
cluster pass is one walk over the file's members, one triangulation per
cluster that passes the member rules, and one reprojection per non-target
member of each cluster that triangulates. The current value is materialised first
when the overlay is not empty, and the
answer pushed is a whole base held by the history until the budget releases it
([../document-model.md](../document-model.md)).

---

## Testing

Core (`sfmtool-core`, headless):

- Perturb one image's stored pose in a synthetic reconstruction; the action
  recovers the original within the estimator's tolerance, and the held-out
  re-triangulation never reads the target's observations (a target with
  corrupted observation coordinates still yields correct held-out positions).
- Two targets held out together: both poses recover, and with **both** targets'
  observations corrupted the held-out positions of the points they share are
  still the truth — a hold-out that dropped only the image being estimated
  would read the other target's corrupted rows.
- A rotation-only synthetic reconstruction: the rotation-only path recovers a
  perturbed rotation and leaves the translation untouched.
- Per-target refusals: too few held-out points, too few bearings, degenerate
  bearings — each reported rather than failing the call.
- Whole-call refusals: an empty set, a target named twice, an unposed target, a
  set leaving fewer than three non-target posed images.
- Determinism: the same input gives a bit-identical resected reconstruction.
- Clusters: an image with no tracks is resected from clusters alone; the tracks
  and the clusters reach the estimate side by side, with the track pairs those
  the tracks alone give; a cluster with two kept members in the target is set
  aside; a cluster whose only non-target kept member is in one image is set
  aside when a second target holds the other member, and a rejected member does
  not count; clusters with parallel rays or rays that meet behind the cameras
  fail triangulation; a cluster whose members agree with its position is kept,
  one with a member 10 px off is counted in `clusters_inconsistent` and gives
  no pair (and is kept under a wider threshold); a member behind its camera,
  outside its frame or at a non-finite pixel does not agree; the rotation-only
  path reads no cluster; a file without the cluster sections is refused.
- Images without tracks: a member in an image with no track observation is not
  counted (a member 10 px off makes a cluster inconsistent while its image has
  tracks, and does not once the image has none); a cluster left with one
  tracked non-target image is counted in `clusters_untracked` and gives no
  pair; the cluster counts add up to `clusters_considered`, and the totals to
  the reports.
- Tracks lead: with ten finite tracks, fifteen clusters that agree with them
  and thirty clusters that agree with each other on a pose 4° away, the pose
  recovered is the tracks' and the thirty are outliers; with two tracks, the
  tracks are sampled with the clusters, and are inliers when the clusters agree
  with them and outliers when the thirty win.
- Bearings: a track at infinity is a correspondence of the finite path,
  counted in `bearing_correspondences` and `bearing_inliers`, and one observed
  30 px off is an outlier; a bearing's residual is unchanged by moving the
  camera and grows by the focal length times the angle the camera turns
  across its ray.

Bindings (`tests/rust_bindings/`): the name-to-index lookup and its
`ValueError`, the report dict and its per-image list, a refusal returning a
reconstruction rather than raising, a two-target call, that the input
reconstruction is unchanged, the source split with a written cluster-patches
file, the totals' split, a cluster with a moved member counted in
`clusters_inconsistent` at the default threshold and kept at
`max_cluster_residual_px=inf`, members in images that observe nothing counted
in `clusters_untracked` and giving no pair, and a file without the cluster sections
(`ValueError`) or that cannot be read (`OSError`). The installing variant is in
`test_edited_reconstruction_rust_bindings.py`, on the real reconstruction the
other edit bindings use: an image past the table raising, and an accepted
estimate moving that image alone while the value it came from stands.

Explorer (`sfm-explorer` lib tests, headless egui):

- The entry is on image rows and not on the reconstruction row, reports the
  image it was chosen on, and is greyed for an unposed image, too few posed
  images, and a missing or stale cluster-patches file, with the refusal as its
  hover text; it is live on an embedded-patches node with a current file. The
  Image Browser strip's copy of the menu is compared with the tree's
  (`image_menu/tests.rs`).
- The refusal names the file's state and the build entry, or tells an unsaved
  node to save first; an image's own reasons come first
  (`resect/tests.rs`).
- The version pushed and its new base, the image table standing still, the pose
  recovered from a perturbation, one Action Log entry naming the image, the
  source split and the serials, an undo putting the pose and the selection
  back, and the refusals -- one the call could not attempt, one the estimate
  declined, and a missing or stale file -- pushing no version and logging a
  failure. Over MCP, `resect_camera_image` without a file is refused in the
  same words.

---

## Non-goals

- No bundle adjustment after resection (see step 4). Adjusting the whole node
  afterwards is its own edit, in [README.md](README.md).
- No multi-image selection in the panel; one image per action, whatever the
  primitive underneath accepts.
