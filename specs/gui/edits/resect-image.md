# Resect Image

An action in the Scene Graph panel that re-estimates one image's pose against
the rest of its reconstruction and installs the answer as that reconstruction's
next **version**, which an undo steps back out of.

The shared primitive underneath takes a **set** of target images and holds all
of them out together, so a group whose members corroborate each other is
questioned as a group; the panel's action is that primitive on a one-element
set.

Related specs: [../scene-graph.md](../scene-graph.md) (nodes, context
menus, `Align to…` as the template for a per-node action),
[../../core/geometry/reconstruction-growth.md](../../core/geometry/reconstruction-growth.md)
(`resect_images_batch`, the registration primitive this reuses),
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

Context menu on an **image row** of the Scene Graph tree (the rows under a
reconstruction's `Camera Images` group): `Resect Image`. The row's
reconstruction is the source node; the row's image is the target.

The entry is greyed out, with a hover explanation, when the image is not posed
in its reconstruction, or when the source has fewer than three other posed
images.

A second entry, `Resect Image from Matches…`, opens a file chooser for a
`.matches` file and runs the matches-backed variant (see "Correspondence
sources"). It is greyed when the source's observations carry no feature
indexes (an `embedded_patches` file), since matches cannot be joined to them.
The chosen path is remembered per source node for the session.

---

## Mechanism

Everything below `## Invocation` lives in `sfmtool-core` as one function,
`geometry::resect_images` in
[resect_images.rs](../../../crates/sfmtool-core/src/geometry/resect_images.rs); the
GUI wraps it in [resect.rs](../../../crates/sfm-explorer/src/resect.rs), which adds
the menu entries, the version and the status line. The same function is what an
offline caller uses, with as many targets as it wants to hold out, through the
`sfmtool._sfmtool.geometry.resect_images` binding.

The function takes a **target set**: one or more images of the source, named as
a set rather than resected one after another. Every step below is over that
set; the panel's action passes a single image, which is the set of size one.

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
  bearing. Bearings are excluded from the finite set.

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

- **Finite path.** A target with at least the batch-registration observation
  floor of held-out finite correspondences (`ResectOptions::min_obs`) is
  estimated by `resect_images_batch`: RANSAC P3P polished by trimmed pose-only
  refinement, scored by the all-observation inlier fraction at the 3 px bound,
  seeded deterministically. The camera model is each image's own, so the
  targets run as one batch per camera model; each image's seed is a function of
  its own index, so the grouping changes no answer.
- **Rotation-only path.** Below that floor, or when the reconstruction is
  rotation-only, the rotation is estimated by closed-form absolute
  orientation between the target's observed ray directions and the held-out
  bearings of the points at infinity it observes (trimmed, iterated), and the
  translation is left at its stored value. Requires at least three bearings,
  spanning an angle the camera can resolve — the largest angle any bearing
  makes with their mean direction has to exceed one pixel's worth of angle at
  the camera's own focal, since a spread narrower than that is not a spread.
  The inlier bound is the finite path's 3 px bound in the same currency: the
  angle a pixel subtends on this camera.
- **Refusals.** A target that misses the gate, whose bearings span no
  resolvable angle, or that has support on neither path, is refused: it keeps
  its stored pose, its report carries the reason, and the rest of the set
  proceeds. The call still hands back its reconstruction, carrying the refused
  targets' stored poses and the held-out re-triangulation; what the panel's
  single-target action does with that is step 6.

### 4. Re-triangulation at the new poses

With the accepted targets at their resected poses, every finite point an
accepted target observes is re-triangulated from **all** its observations,
including the targets' own. Points no accepted target observes are left at
their held-out positions, and points the set does not observe at all are
untouched. No bundle adjustment runs: the point of the action is to show what
the resection alone says, not what a joint refit would smooth over.

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
materialises the current value when its overlay is not empty, runs the core
function over it, and pushes the result with the row map `RowMap::by_scan` reads
off that call's input and output: the resection may drop a point it can neither
re-triangulate nor hold out, and says nothing about which. The selection follows
that map. The image table does not move, so image indexes, the image and camera
selections and the decoded pixels keyed by them all still mean what they meant;
what the panels cached *about* the geometry -- rendered patches, prepared track
rows, per-camera derived quantities -- is dropped, because it describes a
geometry the node no longer holds.

The same edit is available offline as
`EditedReconstruction.resect_image_in_place(image, matches_path=…)`, which
returns `(EditedReconstruction, report)` and raises on the refusal.

### Correspondence sources

- **Stored observations** (default, `Resect Image`): a target's 2D-3D pairs are
  its own observations joined to the held-out positions of step 2.
- **Matches** (`Resect Image from Matches…`): a target's 2D-3D pairs come from
  the match graph of the chosen `.matches` file — the target's keypoints, to
  matched keypoints in the non-target posed images, to those observations'
  held-out positions. This admits points the reconstruction never assigned to
  the target (its observation set is a subset of what the matches offer) and is
  the same construction as the offline non-member resection. Match rows are
  joined through feature indexes, so this source requires a `sift_files`
  reconstruction. Either backbone serves — clusters, where every pair of
  members on distinct images is a match, or the pairwise sections. The
  target's pixel is the refined member position when the file carries
  `cluster_patches` (that is what the cluster claims the feature is at) and
  the target's own `.sift` detection otherwise; a rejected or unevaluated
  cluster member is not a claim and does not participate. A point the set
  observes is scored against its **held-out** position, never the stored one it
  helped fit, and is dropped from the pairs when it has none. Keypoints of the
  target that have no observation in the reconstruction contribute to the
  estimate but create no new track.

### Reported quantities

Each target gets its own report — the path taken, the correspondences the
estimate saw and how many were inliers, whether it was accepted and why not,
how far its pose moved, and its share of the held-out, re-triangulated and
removed points. Over the set there are totals: how many targets were accepted
and refused, the summed correspondences and inliers with their ratio, and the
held-out, re-triangulated and removed point counts with each point counted
once however many targets observe it.

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
  `Resected IMG_0007.jpg (bull): 214 pts, inliers 198/214 (0.93),
  rotation 12.40°, translation 0.081 (scene-scale), 190 re-triangulated
  (v3 → v4)`. A refusal is one **failed** entry:
  `Resect <image> in <node> refused: <reason>`. One entry either way, and it is
  also what the status line (viewport overlay, as `Align to…` reports) shows.
- **Selection**: unchanged, but followed through the map: the node is the one
  already on screen, and the image the menu was opened on is where it was. A
  point the resection dropped clears the point selection.

---

## Performance

Held-out re-triangulation and the estimates touch only the observations of the
points the target set observes (hundreds to a few thousand rows for one
target); the finite path is `resect_images_batch` over the targets. The panel's
single-target action runs synchronously in tens of milliseconds on the
reconstructions the viewer targets. The matches variant additionally parses the
`.matches` file once per session per source node; that parse is cached. The
current value is materialised first when the overlay is not empty, and the
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

Bindings (`tests/rust_bindings/`): the name-to-index lookup and its
`ValueError`, the report dict and its per-image list, a refusal returning a
reconstruction rather than raising, a two-target call, and that the input
reconstruction is unchanged. The installing variant is in
`test_edited_reconstruction_rust_bindings.py`, on the real reconstruction the
other edit bindings use: an image past the table raising, and an accepted
estimate moving that image alone while the value it came from stands.

Explorer (`sfm-explorer` lib tests, headless egui):

- The two entries are on image rows and not on the reconstruction row, they
  report the image and the correspondence source they were chosen for, and they
  grey on the same reasons.
- The version pushed and its new base, the image table standing still, the pose
  recovered from a perturbation, one Action Log entry naming the image and the
  serials, an undo putting the pose and the selection back, and all three
  refusals -- one the call could not attempt, one the estimate declined, and a
  matches source with no file chosen -- pushing no version and logging a
  failure.

---

## Non-goals

- No bundle adjustment after resection (see step 4). Adjusting the whole node
  afterwards is its own edit, in [README.md](README.md).
- No multi-image selection in the panel; one image per action, whatever the
  primitive underneath accepts.
