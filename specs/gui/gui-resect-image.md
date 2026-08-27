# Resect Image

*Status: implemented* — `sfmtool_core::geometry::resect_image`
(`crates/sfmtool-core/src/geometry/resect_image.rs`), surfaced by
`crates/sfm-explorer/src/resect.rs`, the image rows' context menu in
`scene_graph/mod.rs`, and `AppState::resect_image`.

A per-image action in the Scene Graph panel that re-estimates one image's pose
against the rest of its reconstruction and shows the result as a new node
beside the original. The original node is never modified; the derived node is
an ordinary loaded reconstruction in the same frame, so the two can be
compared with every existing affordance (tint, solo, per-node visibility,
point track detail).

Related specs: [gui-scene-graph.md](gui-scene-graph.md) (nodes, context
menus, `Align to…` as the template for a per-node action),
[../core/reconstruction-growth.md](../core/reconstruction-growth.md)
(`resect_images_batch`, the registration primitive this reuses),
[../core/seed-candidate-evaluation.md](../core/seed-candidate-evaluation.md)
(the hold-out self-resection channel; same mechanism, read offline).

---

## Purpose

Whether one camera's pose is corroborated by the rest of the reconstruction is
a question the stored pose cannot answer: it was fit jointly with the points it
observes, so it always agrees with them. Re-estimating the pose from structure
that did **not** depend on that image, and putting the result next to the
original, lets a reviewer see the disagreement directly — the frustum moves,
the points the image observes re-triangulate, and the point track detail shows
where the image's observations land under each pose.

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

Everything below `## Invocation` lives in `sfmtool-core` as one function —
`geometry::resect_image` — and the GUI adds the menu entries, the node
bookkeeping, and the status line. The same function is what an offline caller
uses to resect an image.

### 1. Clone

The source reconstruction is deep-copied. All work happens on the copy.

### 2. Held-out structure

The target image's contribution to structure is removed before its pose is
estimated:

- Every finite point the image observes that retains at least two other
  observations is **re-triangulated from those other observations only**, at
  the stored poses of the other images. The stored position is discarded for
  this purpose.
- A finite point the image observes that retains fewer than two other
  observations has no held-out position. It is excluded from the estimate.
- A point at infinity the image observes is a direction fixed by the other
  images' rotations; it is kept as a bearing and excluded from the finite set.

The re-triangulated positions are used for the pose estimate and are
**kept** in the derived reconstruction (they are what the other images say
about those points).

### 3. Pose estimate

- **Finite path.** With at least the batch-registration observation floor of
  held-out finite points (`ResectOptions::min_obs`), the pose is estimated by
  `resect_images_batch` for the one image: RANSAC P3P polished by trimmed
  pose-only refinement, scored by the all-observation inlier fraction at the
  3 px bound, seeded deterministically. The camera model is the image's own.
- **Rotation-only path.** Below that floor, or when the reconstruction is
  rotation-only, the rotation is estimated by closed-form absolute
  orientation between the image's observed ray directions and the bearings of
  the points at infinity it observes (trimmed, iterated), and the translation
  is left at its stored value. Requires at least three bearings, spanning an
  angle the camera can resolve — the largest angle any bearing makes with the
  set's mean direction has to exceed one pixel's worth of angle at the
  camera's own focal, since a spread narrower than that is not a spread.
  Below either, the action refuses. The inlier bound is the finite path's
  3 px bound in the same currency: the angle a pixel subtends on this
  camera.
- The estimate is accepted or refused on the primitive's own gate
  (`accept_gate`). A refused estimate still produces the derived node — with
  the stored pose retained and the refusal reported — so the reviewer can see
  the held-out re-triangulation on its own.

### 4. Re-triangulation at the new pose

With the target at its resected pose, every finite point the image observes is
re-triangulated from **all** its observations, including the target's. Points
the image does not observe are untouched. No bundle adjustment runs: the point
of the action is to show what the resection alone says, not what a joint refit
would smooth over.

A point fails re-triangulation when fewer than two observations survive, when
the solve puts it behind one of the cameras that observe it, or when its depth
is not observable at all (parallel rays, which leaves the triangulation's normal
matrix rank-deficient — the parallax floor stated in the solve's own
diagnostics rather than as a separate angle threshold). A point that fails keeps
its held-out position from step 2 when it has one, and is otherwise removed with
its observations.

### 5. Result

The derived reconstruction differs from the source only in the target's pose
and in the points the target observes. Its metadata records
`operation = "explorer_resect"`, the target image's relative path, the
correspondence source, and the estimate's inlier fraction, so a later save
carries provenance.

### Correspondence sources

- **Stored observations** (default, `Resect Image`): the 2D-3D pairs are the
  image's own observations joined to the held-out positions of step 2.
- **Matches** (`Resect Image from Matches…`): the 2D-3D pairs come from the
  match graph of the chosen `.matches` file — the target's keypoints, to
  matched keypoints in the other posed images, to those observations' held-out
  positions. This admits points the reconstruction never assigned to the
  target (its observation set is a subset of what the matches offer) and is
  the same construction as the offline non-member resection. Match rows are
  joined through feature indexes, so this source requires a `sift_files`
  reconstruction. Either backbone serves — clusters, where every pair of
  members on distinct images is a match, or the pairwise sections. The
  target's pixel is the refined member position when the file carries
  `cluster_patches` (that is what the cluster claims the feature is at) and
  the target's own `.sift` detection otherwise; a rejected or unevaluated
  cluster member is not a claim and does not participate. A point the target
  also observes is scored against its **held-out** position, never the stored
  one it helped fit. Keypoints of the target that have no observation in the
  reconstruction contribute to the estimate but create no new track.

### Reported quantities

The status line (viewport overlay, as `Align to…` reports) shows:

`Resected <image> in <node>: <n> pts, inliers <k>/<n> (<f>), rotation
<deg>°, translation <d> (scene-scale), <m> re-triangulated`

where the rotation delta is the angle between the stored and resected
world-to-camera rotations and the translation delta is the distance the camera
**centre** moved, in units of the source's median-over-images of that image's
median camera-to-structure distance (the same unit the evaluation channels
use). A rotation-only reconstruction has no such distance, so it reports the
displacement in its own units and no ratio. On refusal:
`Resect <image> in <node> refused: <reason>`.

---

## The derived node

- **Name**: `<source name> (resected <image basename>)`. A second resection of
  the same image from the same source replaces the earlier derived node rather
  than adding a third.
- **Frame**: the derived node inherits the source's current transform, so it
  lands exactly on top of the source in the viewport.
- **Selection**: the derived node becomes the selected reconstruction, and the
  target image is selected in it, so the point track detail opens on the
  resected image immediately.
- **Saving**: the derived node saves through whatever file action the
  application offers for a loaded node — today none, so it is a session
  artifact. This feature adds no writing of its own; what it reads is the
  `.matches` file of the matches variant, and, on a `sift_files`
  reconstruction, the `.sift` companions of the images that observe the
  target's points (that is where those observations' 2D coordinates live).
- The node's name is its provenance in the tree; the resected image is not
  marked anywhere else (its identity is in the name, and in the
  reconstruction's metadata).

---

## Performance

Held-out re-triangulation and the estimate touch only the target's
observations (hundreds to a few thousand rows); the finite path is
`resect_images_batch` on one image. The whole action runs synchronously in
tens of milliseconds on the reconstructions the viewer targets. The matches
variant additionally parses the `.matches` file once per session per source
node; that parse is cached.

---

## Testing

Core (`sfmtool-core`, headless):

- Perturb one image's stored pose in a synthetic reconstruction; the action
  recovers the original within the estimator's tolerance, and the held-out
  re-triangulation never reads the target's observations (a target with
  corrupted observation coordinates still yields correct held-out positions).
- A rotation-only synthetic reconstruction: the rotation-only path recovers a
  perturbed rotation and leaves the translation untouched.
- Refusal paths: too few held-out points, too few bearings, degenerate
  bearings.
- Determinism: the same input gives a bit-identical derived reconstruction.

Explorer (`sfm-explorer` lib tests, headless egui):

- The menu entry appears on image rows only, greys correctly, emits the
  action, and the app creates the derived node with the specified name,
  transform, and selection; a repeat replaces the earlier node.

---

## Non-goals

- No bundle adjustment after resection (see step 4).
- No batch resection of many images from the panel; one image per action.
- No modification of the source node under any outcome.
