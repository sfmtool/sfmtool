# Move Camera

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

Related specs: [`../camera-views.md`](../camera-views.md) (camera view itself,
and which navigation keeps it), [`../viewport-navigation.md`](../viewport-navigation.md)
(the controls the lock re-points), [`../document-model.md`](../document-model.md)
(the version, the two kinds of edit, and what a bulk edit owes the caches),
[`../edit-history.md`](../edit-history.md) (the cursor and the Edit menu),
[`../../core/reconstruction/move-camera.md`](../../core/reconstruction/move-camera.md)
(the core function this wraps), [`../resect-image.md`](../resect-image.md) (the
edit that re-*estimates* one image's pose rather than handing it over), and
[`../saving.md`](../saving.md).

---

## Invocation

**Enter the lock.** In camera view of image `i` of node `n` — the viewport is
looking through the camera, its photograph is the background — press `M`, or
choose `Edit > Move Camera`, or choose `Move Camera` on the image row's context
menu in the Scene Graph, which enters camera view first and then locks. The Edit
menu entry is greyed with a hover text when the viewport is not in camera view,
when the image carries no pose, or when a lock is already held.

On entry the viewport is **snapped to the camera's exact pose**, the same
placement entering camera view performs, so a free-look offset the reviewer
happened to have does not become an edit nobody made. The snap keeps the
viewport's field of view: the lens is not what is being moved.

**While locked**, every navigation input moves the pending pose rather than the
viewport alone. No input leaves camera view while the lock is held, because the
camera and the viewport are one thing — which is the fourth column of the table
in [`../camera-views.md`](../camera-views.md) § "Which navigation keeps camera
view".

| Input | Effect on the pending pose |
|-------|----------------------------|
| Unmodified drag, two-finger drag, gesture (nodal pan) | rotates about the camera centre |
| Alt-drag (orbit) | rotates about the target and translates the centre with it |
| Shift-drag, middle-drag (pan) | translates in the image plane |
| WASD, R/F (fly) | translates along the camera axes |
| Q/E (tilt) | rolls about the optical axis |
| Scroll, Ctrl-drag, right-drag, pinch (zoom) | viewport FOV only; the pose is untouched |
| Alt+Ctrl scroll (target push/pull) | the target distance only |
| Home / Shift+Home | refused, with a status line |
| `,` / `.` | commit, then switch camera |

**Leave the lock.** `M` again, `Enter`, or `Edit > Commit Camera Move` commits:
one version is pushed carrying the pose the viewport is at. `Escape`, or
`Edit > Cancel Camera Move`, cancels: the viewport snaps back to the camera's
stored pose and no version is pushed. Anything else that ends the lock
implicitly, which is `,`/`.`, `[`/`]`, double-clicking another frustum,
selecting another node, closing the node, an MCP tool that edits it and an
MCP `set_view` that leaves camera view (a fit, a look-through, a placement or
an explicit exit; a field-of-view change keeps camera view and so keeps the
lock, as the zoom controls do),
**commits** first when the pose has changed by more than the dead band below,
and drops the lock silently otherwise. Closing is the one that cannot commit
afterwards, so it commits before the close is even asked about; a lock left over
a node that is gone anyway is dropped rather than acted on. Committing rather than cancelling on an implicit exit follows the
rule the delete edits set: an edit has a history behind it, so undo is the
answer to an accidental one, whereas cancelled work has no undo.

**Dead band.** A commit whose pose differs from the stored one by under
0.01 degrees and under 1e-6 of the capture's extent — the radius of its camera
cloud — pushes nothing and says so in the status line: the lock was held,
nothing was moved.

The keys are gated on egui's own keyboard arbitration, exactly as the viewport's
other bindings are, so `M` typed into a HUD field is an `M`.

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
observe** are a mixed signal: their positions were solved with this camera's old
ray in the mix, so they are partly a memory of the wrong pose.

If those observed points were re-solved live, every frame, they would chase the
camera — each dragged toward wherever its keypoint now points — and the reviewer
would watch the points follow the photograph instead of the photograph reaching
the points. So the rule is:

> **While the lock is held, every point stays where the value has it.** The
> photograph and the frustum move; nothing else does. Re-triangulation is what
> the commit does, once.

### What is live

Three things update every frame, and all three are cheap:

1. **The photograph**, at the pending pose. The camera-view background is a mesh
   of the image's own rays, pre-rotated by the camera's stored rotation, so
   putting it at the pending pose is a rotation of the node's model matrix
   rather than a rebuilt mesh. It is drawn at infinity, which is also why
   *translating* the camera leaves the photograph alone: a background infinitely
   far away has no parallax to show.
2. **The residual readout**, in the lock banner: the median and 90th-percentile
   reprojection residual of this image's observations under the pending pose,
   against their stored keypoints, beside the same two numbers under the stored
   pose. It is `O(K)` projections for a track count `K` in the hundreds to low
   thousands, well under a millisecond, and it is the number the reviewer is
   really steering by when the photograph lines up. `n/a` where the value
   carries no inline keypoints.
3. **The observed points**, drawn in the selection tint's hover variant for the
   duration of the lock, so the reviewer can tell the two signals apart: tinted
   points are the ones that will move on commit, untinted ones are the fixed
   reference.

**The frustum is the viewport.** Camera view hides the frustum of the image it
looks through — it is drawn from inside — and the lock keeps it hidden, because
the pending pose *is* the viewport pose and a wireframe pinned to the eye says
nothing a reviewer can act on.

Everything outside the viewport reads the value at the cursor and is not told
about the pending pose: the Image Detail panel, the Point Track Detail panel,
the Scene Graph's pose readout and the MCP `get_camera_image` reply all show the
stored pose until the commit. A lock is viewport state, not document state,
which is the same line camera view itself sits on.

### The lock banner

Three lines, in the HUD's own vocabulary — 12 pt text on a dark ground — drawn
top-centre in the viewport, under the camera-position line and only while the
lock is held:

```
Moving IMG_0007.jpg (bull)
residual 4.12 / 6.50 px, stored 1.00 / 2.25 px
M or Enter commits, Esc cancels
```

---

## Mechanism

Everything below the lock is
[`../../core/reconstruction/move-camera.md`](../../core/reconstruction/move-camera.md):
`sfmtool_core::move_camera`, a pure function from the version's value plus a
pose to the next value and a report. What the viewer adds is the hand, the
version and the history entry, in
[camera_lock.rs](../../../crates/sfm-explorer/src/camera_lock.rs) and
[state/edits.rs](../../../crates/sfm-explorer/src/state/edits.rs).

No images are decoded. The re-triangulation reprojects through the poses and
lenses the value already carries, so this edit reads nothing off disk, and it
runs synchronously on the frame the commit was asked for.

### The pose crosses the node's transform by division

The core function takes the pose in the **node's own frame**, never the
displayed one. The viewport's pose is in the frame the node is *drawn* in, so
the viewer composes it with the inverse of the node's `Align to…` transform
before calling: a moved camera in an aligned node lands where the reviewer put
it on screen, and the value never learns the display transform existed. This is
the one place a display transform crosses into an edit short of bake-transform,
and it crosses by being divided out.

### The version

A pose lives in the base, so this is a **bulk edit** in the document model's
sense and follows `delete_image`'s shape: materialise the current value when its
overlay is non-empty, run the pure function, install the result as the next
version's base with an empty overlay, and derive the map with `RowMap::by_scan`
over the call's input and output. The move deletes and creates no points, so the
scan produces the identity map and the selection follows into the same index;
the scan is still run rather than assumed, so the map is a fact about the two
values and not about this spec.

Image indexes do not move, so the node keeps its image, camera and hover
selections and its decode caches; the panels' geometry-derived caches are dropped
the way the resect and bundle-adjust edits drop them.

The version's label is

`Moved camera <image> (<node>): <rotation> deg, <translation>`

with the translation in scene units where the report has them, and
`, <n> points re-solved` appended when any were.

### The Action Log

One `Kind::Edit` entry per commit, the label plus the residual pair:

`Moved camera IMG_0007.jpg (bull): 3.20 deg, 0.140 scene units, 212 points
re-solved, residual 4.1 → 1.2 px (v7 → v8)`

Entering the lock, cancelling one and a dead-band exit are `Kind::View` entries —
`Moving the camera of IMG_0007.jpg (bull)`, `Cancelled the camera move of
IMG_0007.jpg (bull)`, `IMG_0007.jpg (bull) was not moved` — because the lock is
viewport state and none of the three changed the value. A refused `Home` is a
failed entry of the same kind.

### Undo

Ordinary. Undo restores the version before the commit, and since the lock is
viewport state, a cursor move while the viewport is in camera view of the same
image re-snaps the viewport onto the restored pose, the way camera view follows
a `,`/`.` switch — so the photograph jumps back with the pose rather than
staying where the hand left it.

The rule is the cursor's and not the menu's: `undo`, `redo` and `jump_to_version`
on the wire re-snap it too, where the GUI thread applies them
([`../mcp-server.md`](../mcp-server.md)), so an agent stepping the history moves
the photograph in front of the reviewer exactly as their own Undo would. A
camera held in hand suppresses the re-snap either way: a commit is what releases
a lock, and until then the viewport is showing the pending pose.

### The wire

`move_camera_image { reconstruction_label, camera_image, world_from_camera:
{ quaternion_wxyz, translation } }` applies the commit directly, with no lock,
and answers as every edit tool there does: the version it pushed, its label, and
the sentence the edit recorded. The lock itself is not on the wire: an agent has
no hand, and the pose is the whole input. A lock held on the node an edit tool
names is ended first, by the same rule every other step away from a lock
follows, because an edit landing under one would leave the reviewer holding a
camera whose stored pose had moved beneath them.
See [`../mcp-server.md`](../mcp-server.md).

---

## Testing

Core (`sfmtool-core`, headless): the four ways a track is settled, the frame
rescale, the report and every refusal. See
[`../../core/reconstruction/move-camera.md`](../../core/reconstruction/move-camera.md).

Bindings (`tests/rust_bindings/test_edited_reconstruction_rust_bindings.py`): the
call's shape over a real reconstruction, and the refusals.

Explorer (`sfm-explorer` lib tests, headless):

- `camera_lock/tests.rs`: the entry snap and the FOV it keeps, the pose the
  viewport reads back as (through an aligned node's transform, and back), which
  navigation inputs move it and which do not, that no navigation path can drop
  camera view behind the lock's back, the residual readout against a direct
  computation, the banner's wording with and without keypoints, commit, cancel,
  the dead band, the implicit exit, the keys, the undo re-snap, the highlighted
  set, a lock whose node is closed under it, and that holding the lock leaves
  the value alone.
- `state/edits/tests.rs`: the version pushed and its new base, the image table
  standing still, the label, the selection following the identity map, the undo,
  and a refusal pushing no version and logging a failure.
- `scene_graph/tests.rs`: the image row's `Move Camera` entry and the image it
  reports.
- `mcp/tests.rs`: the wire tool's reply, its refusal, that it commits a lock
  held on the same node first, and that the wire's three cursor moves re-snap a
  camera view onto the pose the version they land on holds.

Windowed (`crates/sfm-explorer/tests/ui_basic.rs`, Windows): camera view on the
demo reconstruction, `Edit > Move Camera` pressed in a live window, the lock's
own line read back over MCP, the entry found turned into `Commit Camera Move`
beside an enabled `Cancel Camera Move`, and the cancel pressed and read back.
The part a headless frame cannot reach is whether the menu takes the camera in
hand at all and gives it back.

The pose is not moved there, and the reason is the platform rather than the
edit: Windows routes injected mouse input through `WM_POINTER`
([../architecture.md](../architecture.md) carries the same note for the
right-click path), and a moving contact arrives as a touch
rather than a drag, so a synthetic drag turns no camera in this app whether a
lock is held or not. The banner is painted rather than built of widgets, so it
is not in the accessibility tree either. Both are covered headlessly.

---

## Non-goals

- **Previewing where the observed points will land.** Every point stands where
  the value has it until the commit; drawing the observed ones at the positions
  the commit would give them is proposed in
  [`../../drafts/move-camera-preview-amendment.md`](../../drafts/move-camera-preview-amendment.md).
- **Editing the lens.** Zoom while locked changes the viewport's FOV and nothing
  in the value. A focal or distortion edit is its own family.
- **Moving several cameras at once**, or a rig. One lock, one image. A rig's
  cameras move as a unit through the rig's own transform, which is not this edit.
- **A live bundle adjustment.** Nothing in the value but this image's pose and
  its observed tracks changes at commit; running the adjustment afterwards is
  `Edit > Bundle Adjust...`, which is where the other cameras get to answer.
- **Snapping or constraints** — to a ground plane, to another camera's height,
  along a rig baseline. The hand is free.
- **Persisting the lock across sessions.** It ends with the frame that ends it.
