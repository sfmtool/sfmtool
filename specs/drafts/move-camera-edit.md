# Move Camera: editing an image's pose by locking it to the viewport

**Status:** Draft

Proposes the next edit family for
[`sfm-explorer-editing.md`](sfm-explorer-editing.md) Part 5, filing as
`specs/gui/edits/move-camera.md` and `specs/core/reconstruction/move-camera.md`,
and amends [`../gui/camera-views.md`](../gui/camera-views.md) § "Persistent
camera view and free-look navigation", [`../gui/viewport-navigation.md`](../gui/viewport-navigation.md)
§ "Keyboard Shortcuts", [`../gui/viewport-hud.md`](../gui/viewport-hud.md),
[`../gui/edit-history.md`](../gui/edit-history.md) § "The Edit menu",
[`../gui/scene-graph.md`](../gui/scene-graph.md) (the image row's menu),
[`../gui/action-log.md`](../gui/action-log.md) and
[`../gui/mcp-server.md`](../gui/mcp-server.md).

---

## Purpose

An image whose pose is wrong is the commonest defect a reviewer can see and
cannot fix: the photograph, drawn behind the points in camera view, sits a few
degrees off the structure it should line up with, or a metre to one side of it.
Resection re-estimates the pose from correspondences, and when the
correspondences are what is wrong it lands the same answer again. What the
reviewer wants is to take hold of the camera and put it where the photograph
lines up, the way a hand would.

The viewer already has the hand. Camera view puts the viewport exactly at an
image's pose with its photograph rigidly attached, and every navigation control
moves the viewport. So moving a camera is camera view with one bit flipped:
instead of the viewport leaving the camera when it moves, the camera comes
along. The lock is the edit's whole interaction; releasing it is the version.

---

## Invocation

**Enter the lock.** In camera view of image `i` of node `n` (the viewport is
looking through the camera, its photograph is the background), press `M`, or
choose `Edit > Move Camera`, or choose `Move Camera` on the image row's context
menu in the Scene Graph (which enters camera view first, then locks). The Edit
menu entry is greyed with a hover text when the viewport is not in camera view,
when the node is not editable (the value refuses, see the core function's
refusals), or when a lock is already held.

On entry the viewport is **snapped to the camera's exact pose**, the same
placement entering camera view performs, so a free-look offset the reviewer
happened to have does not become an edit nobody made. The snap keeps the
viewport's field of view: the lens is not what is being moved.

**While locked**, every navigation input moves the pending pose rather than
the viewport alone. The table in `camera-views.md` § "Which navigation keeps
camera view" gains a fourth column, and every row of it reads **Yes**: no input
leaves camera view while the lock is held, because the camera and the viewport
are one thing.

| Input | Effect on the pending pose |
|-------|----------------------------|
| Unmodified drag, two-finger drag, gesture (nodal pan) | rotates about the camera centre |
| Alt-drag (orbit) | rotates about the target and translates the centre with it |
| Shift-drag, middle-drag (pan) | translates in the image plane |
| WASD, R/F (fly) | translates along the camera axes |
| Q/E (tilt) | rolls about the optical axis |
| Scroll, Ctrl-drag, right-drag, pinch (zoom) | viewport FOV only; the pose is untouched |
| Alt+Ctrl scroll (target push/pull) | the target distance only |
| Home / Shift+Home | refused with a status line while locked (both would move the viewport away from the camera by a reset, not a hand) |
| `,` / `.` | commit, then switch camera (see below) |

**Leave the lock.** `M` again, `Enter`, or `Edit > Commit Camera Move` commits:
one version is pushed carrying the pose the viewport is at. `Escape`, or
`Edit > Cancel Camera Move`, cancels: the viewport snaps back to the camera's
stored pose and no version is pushed. Anything else that ends the lock
implicitly (`,`/`.`, double-clicking another frustum, selecting another node,
closing the node, an MCP tool that edits the node) **commits** first when the
pose has changed by more than the dead band below, and drops the lock silently
otherwise. Committing rather than cancelling on an implicit exit follows the
rule the delete edits set: an edit has a history behind it, so undo is the
answer to an accidental one, whereas cancelled work has no undo.

**Dead band.** A commit whose pose differs from the stored one by under
0.01 degrees and under 1e-6 of the scene extent pushes nothing and says so in
the status line: the lock was held, nothing was moved.

---

## What the viewport shows while locked

This is the design question the family turns on: what is live while the camera
moves.

### The signal the reviewer is looking for

The point of moving a camera by hand is to make the photograph agree with the
structure. In camera view, the structure is the points drawn over the
photograph. The points this image **does not observe** are the honest signal:
they were placed by other images, they do not care where this camera is, and the
photograph either lands on them or does not. The points this image **does
observe** are a mixed signal: their positions were solved with this camera's
old ray in the mix, so they are partly a memory of the wrong pose.

If those observed points were re-triangulated live, every frame, they would
chase the camera: with the moved ray in the triangulation, each one is dragged
toward wherever its keypoint now points, and the reviewer watches the points
follow the photograph instead of the photograph reaching the points. The
live signal would be destroyed by the very update meant to help.

So the rule is:

> **While the lock is held, every point stays where the value has it.** The
> photograph and the frustum move; nothing else does. Re-triangulation is what
> the commit does, once.

### What is live

Three things update every frame, and all three are cheap:

1. **The frustum and the photograph** at the pending pose. The frustum is one
   instance's uniform; the photograph is the camera-view background, which
   already follows the viewport's view matrix.
2. **A residual readout** in the lock banner: the median and the 90th
   percentile reprojection residual of this image's observations under the
   pending pose, against their stored keypoints, beside the same two numbers
   under the stored pose. This is `O(K)` projections for a track count `K` in
   the hundreds to low thousands, well under a millisecond, and it is the
   number the reviewer is really steering by when the photograph lines up: it
   goes down as the observed points' keypoints come to sit on the points.
   Reads `sift_files` keypoints when the value carries them and shows `n/a`
   when it does not.
3. **Highlighting of the observed points.** The points this image observes
   are drawn with the selection tint's hover variant for the duration of the
   lock, so the reviewer can tell the two signals apart: tinted points are the
   ones that will move on commit, untinted ones are the fixed reference.

### The preview option, and why it is adaptive

A reviewer who wants to see where the observed points will land can turn on
**Preview re-triangulation** in the lock banner (off by default; remembered
for the session). With it on, the observed points are drawn at the position
the commit would give them, and only they: the fixed points never move under
any option, because nothing about the move changes them.

The preview is what needs a cost rule, because it is per-track work on every
frame the pose changes. Re-triangulating `K` tracks is `O(K)` ray assemblies
plus `K` fixed 3×3 eigensolves
([`../core/reconstruction/batch-triangulation-api.md`](../core/reconstruction/batch-triangulation-api.md)),
so it is a few microseconds a track, and pushing `K` new positions to the GPU
is one `write_buffer` over the affected rows (the base's buffer is left alone;
the preview rows are written over it and restored from the value when the lock
ends, the same rows-not-buffers discipline the additions buffer follows in
[`../gui/document-model.md`](../gui/document-model.md)). For a typical image
that is a fraction of a frame. It is not bounded, though: an image in a dense
embed can observe tens of thousands of tracks.

The rule is measured, not guessed:

> The first preview update after the lock is entered is **timed**. If it took
> under 2 ms, the preview runs **every frame** the pose changes. If not, it
> runs **on rest**: when no navigation input has arrived for 150 ms, and once
> more at commit. The banner says which mode it is in (`preview: live` or
> `preview: on rest`).

"On rest" is the adaptive form: the frustum and the photograph still move every
frame, so the hand never lags, and the observed points catch up whenever the
hand pauses, which is when the reviewer is looking at them anyway. There is no
middle setting (every Nth frame, a time-sliced subset), because a preview that
shows half the points moved is a lie about the other half, and a preview that
updates at 20 Hz behind a 60 Hz hand feels broken in a way one that updates on
rest does not. Patches in the preview translate with their points and keep
their frames; the frame's resize is a commit-time rule (below), and a surfel
drawn at a slightly wrong size for the duration of a drag is not a lie about
anything the reviewer is deciding.

Everything outside the viewport reads the value at the cursor and is not
told about the pending pose: the Image Detail panel, the Point Track Detail
panel, the Scene Graph's pose readout and the MCP `get_camera_image` reply all
show the stored pose until the commit. A lock is viewport state, not document
state, which is the same line camera view itself sits on.

---

## Mechanism

### The core function

```rust
pub struct MoveCameraReport {
    pub image: usize,
    pub rotation_deg: f64,
    pub translation: f64,
    pub translation_scene: Option<f64>,
    pub observed: usize,
    pub retriangulated: usize,
    pub kept: usize,
    pub rotated_bearings: usize,
    pub residual_before_px: Option<[f64; 2]>,
    pub residual_after_px: Option<[f64; 2]>,
}

pub fn move_camera(
    recon: &SfmrReconstruction,
    image: usize,
    world_from_camera: &Se3Transform,
) -> Result<(SfmrReconstruction, MoveCameraReport), MoveCameraError>;
```

Pure: the input is untouched, the output is the value with one pose replaced
and the consequences settled. The pose is given in the **node's own frame**,
never the displayed one: the viewer composes the viewport pose with the inverse
of the node's `Align to…` transform before calling, so a moved camera in an
aligned node lands where the reviewer put it on screen and the value never
learns the display transform existed. This is the one place a display
transform crosses into an edit short of bake-transform, and it crosses by
being divided out.

**Refusals.** The image is out of range; the image carries no pose; the pose
is not finite or its rotation is not unit.

**What the call does.**

1. Replaces image `image`'s world-to-camera rotation and translation with the
   inverse of the given pose.
2. Collects the tracks observing `image`. For each:
   - a **finite point with two or more observations that all carry a pixel**
     is re-triangulated from the value's own poses and lenses, the moved one
     included, through `triangulate_batch`, exactly as remove-observation
     re-solves a shortened track. A track whose triangulation fails (rays
     near parallel, the solve behind a camera) **keeps its position** and is
     counted in `kept`; one failed track does not refuse a whole move, since
     the move is what the reviewer decided and the points are its consequence.
     A re-triangulated point's patch frame is rescaled by its
     placement-distance ratio through `ImageTable::placement_scale`, so its
     angular size is preserved, and its `error` column is rewritten as the RMS
     of its residuals under the new geometry;
   - a **bearing (`w = 0`) whose only observation is this image** rotates
     with the camera: its direction becomes the moved camera's ray through its
     keypoint, frame and bitmap unchanged, counted in `rotated_bearings`;
   - a **bearing with two or more observations** keeps its direction, as it
     does under remove-observation, counted in `kept`;
   - a **track without inline keypoints** (`sift_files` with no keypoint
     column) keeps its position, counted in `kept`. The pose still moves: a
     `sift_files` node's camera can be moved, and what it cannot do is
     re-solve its points without pixels.
3. Rebuilds the derived indexes the other edits rebuild.

The report carries the rotation angle and translation of the move (the
translation also in scene units the way the resect report gives it, when the
node has an extent to measure against), the counts above, and the observed
tracks' median and 90th-percentile residual before and after.

### The version

A pose lives in the base, not the overlay, so this is a **bulk edit** in the
document model's sense and follows `delete_image`'s shape: materialise the
current value when its overlay is non-empty, run the pure function, install
the result as the next version's base with an empty overlay, and derive the
map with `RowMap::by_scan` over an identity image map. The move deletes and
creates no points, so the scan produces the identity map and the selection
follows into the same index; the scan is still run rather than assumed, so the
map is a fact about the two values and not about this spec.

Image indexes do not move, so the node keeps its image, camera and hover
selections and its decode caches; the panels' geometry-derived caches are
dropped the way the resect and bundle-adjust edits drop them.

**Label**: `Moved camera {basename} ({node}): {rotation_deg:.2} deg, {translation}`,
with the translation in scene units when the report has them, and
`, {retriangulated} points re-solved` appended when any were.

### The Action Log

One `Kind::Edit` entry per commit, in the label's words plus the residual
pair: `Moved camera IMG_0007.jpg (bull): 3.2 deg, 0.14 units, 212 points
re-solved, residual 4.1 → 1.2 px (v7 → v8)`. A cancel records nothing: the
value did not change. A dead-band exit records nothing either and says so only
in the status line.

### Undo

Ordinary. Undo restores the version before the commit, and since the lock is
viewport state, an undo while the viewport is in camera view of the same image
re-snaps the viewport to the restored pose, the way camera view follows a
`,`/`.` switch, so the photograph jumps back with the frustum rather than
staying where the hand left it.

### Bindings and the wire

`EditedReconstruction.move_camera(image, quaternion_wxyz, translation) ->
(EditedReconstruction, dict)`, the pose given as world-from-camera in the
value's frame, so the same edit is scriptable offline against a pose from
anywhere. Over MCP, `move_camera_image { reconstruction_label, camera_image,
world_from_camera: { quaternion_wxyz, translation } }` applies the commit
directly with no lock and the same reply shape the other edit tools use. The
lock itself is not on the wire: an agent has no hand, and the pose is the
whole input.

---

## Testing

Core: a synthetic scene where moving one camera by a known rigid motion and
re-triangulating brings every observed point to within tolerance of where the
same scene solved with that pose puts it; the input untouched; a single-view
bearing's direction equal to the moved ray; a multi-view bearing and a
`sift_files` track keeping position and being counted; a failed triangulation
counted in `kept` and the point unmoved; frames rescaled by the depth ratio;
the report's residual pair; every refusal.

Explorer, headless: the lock entered from camera view snaps the viewport to
the stored pose and keeps the FOV; each navigation input moves the pending
pose and none leaves camera view; the residual readout equals a direct
computation; commit pushes one version with the identity map and the label,
and the selection follows; cancel pushes none and restores the viewport; the
dead band pushes none; an implicit exit commits when moved and not otherwise;
the pose crosses the node's transform correctly (an aligned node's stored
pose is the viewport pose divided by the transform); the preview mode
decision from a stubbed timing; undo re-snaps the viewport; the menu and
banner gating.

Windowed: enter camera view on the demo reconstruction, press `M`, drag,
press `Enter`, and read the version and the Action Log line back over MCP.

---

## Non-goals

- **Editing the lens.** Zoom while locked changes the viewport's FOV and
  nothing in the value. A focal or distortion edit is its own family.
- **Moving several cameras at once**, or a rig. One lock, one image. A rig's
  cameras move as a unit through the rig's own transform, which is not this
  edit.
- **A live bundle adjustment.** Nothing in the value but this image's pose and
  its observed tracks changes at commit; running the adjustment afterwards is
  `Edit > Bundle Adjust...`, which is where the other cameras get to answer.
- **Snapping or constraints** (to a ground plane, to another camera's height,
  along a rig baseline). The hand is free; a constrained move is a later
  option on the same lock.
- **Persisting the lock or the preview option across sessions.**

---

## Open questions

- Whether the observed-point highlight should also draw each observed point's
  keypoint ray from the moving camera, which makes the residual visible per
  point rather than as a statistic. Cheap to draw (one line per observed
  track); the question is whether it helps or clutters.
- Whether `Move Camera` on the Scene Graph image row should exist at all, or
  whether camera view plus `M` is the only entry. The row entry saves a
  keystroke and gives the edit a discoverable name; it also puts a pose edit
  one click from a tree row, where every other entry is a view or a
  resection.
