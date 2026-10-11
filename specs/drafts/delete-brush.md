# Delete brush in the 3D viewport

**Status:** Draft

Decided: a brush tool in the 3D viewport that deletes the points painted over
in one stroke, highlights them while the stroke is held, and deletes them on
release as one version. Before the press, the circle shows the points it would
paint. It has no depth modes and no unpaint gesture. A tool strip of two icons
in the viewport, chosen with the mouse only, switches between ordinary
navigation and the brush; the choice does not go in the HUD and has no keyboard
shortcut. The radius is fixed at 24 logical pixels. Nothing is open.

Amends [`../gui/viewport-navigation.md`](../gui/viewport-navigation.md) (a
tool that takes the unmodified left drag),
[`../gui/viewport-hud.md`](../gui/viewport-hud.md) (a second floating control
beside the HUD), [`../gui/document-model.md`](../gui/document-model.md) § "Two
kinds of edit" (a point edit that deletes many points), and
[`../gui/mcp-server.md`](../gui/mcp-server.md) (`delete_points`, the tool
state, and a `drag` input tool). When it is filed, the edit itself becomes
`gui/edits/delete-brush.md`.

## Purpose

A reconstruction often has clusters of bad points: floaters in front of the
subject, sky, a spray of points from one bad image, background nobody wants.
The SfM Explorer can delete only one point at a time: select it, press Delete.
Cleaning a few thousand points that way is not practical. The delete brush is
a paint tool: the cursor becomes a circle, the person drags it across the
points they want gone, the points it passes over turn red, and when the mouse
button is released those points are deleted from the reconstruction as one
edit, which one Undo reverses.

The brush is a cleanup tool, not a selection. It holds nothing once the stroke
ends, so it leaves the design of multi-select open.

## What the person does

### Choosing the tool

A **tool strip** floats at the left edge of the 3D viewport, centred
vertically. It holds two icon buttons, one above the other, and exactly one of
them is on:

| Icon | Tool | Tooltip |
|------|------|---------|
| Mouse cursor (top) | **Navigate**: everything the viewport does today | `Navigate: orbit, pan and zoom the view, and click to select.` |
| Eraser brush (below) | **Delete brush** | `Delete brush: drag over points to delete them from the selected reconstruction.` |

The strip holds nothing else: no radius control and no other settings. A tool is
chosen by clicking its icon, and there is no keyboard shortcut for either. The
viewer starts in Navigate, and the tool is remembered for the session and not
saved.

The strip follows the HUD's input rules
([`../gui/viewport-hud.md`](../gui/viewport-hud.md) § "Input arbitration"): it
is an `egui::Area` on a layer above the painter, so a click or drag on it never
reaches the viewport, and its rect joins the HUD's in the scroll and gesture
gates. The left edge is used because the four corners and the top centre are
already taken (`user-experience.md` § "Information overlays"), and because
paint programs put their tools there. The strip is not a HUD section: the HUD
holds display settings and collapses to a gear, and the current tool has to be
visible and one click away at all times, because it changes what a left drag
does.

Entering the Move Camera lock (M) switches the tool to Navigate and disables the
brush button until the lock ends. The lock commits a pose and is a different
edit in progress; a deletion pushed in the middle of it would put a version
between the lock's start and its commit.

### The stroke

With the brush on, the pointer over the viewport shows a circle outline of
radius 24 logical pixels, a constant (`BRUSH_RADIUS`), drawn on the viewport's
painter, and the crosshair cursor at
its centre. egui has no circle cursor, so the circle is painted the way the
lock banner is (`viewer_3d/mod.rs`).

- **Before the press, the circle previews.** The points a press at this spot
  would paint are drawn in a pale red, so the person sees what a stroke would
  take before starting it. The preview follows the pointer and also updates
  when the view moves under a still pointer (a scroll zoom, the fly keys, a
  navigation drag). It is cleared when the pointer leaves the viewport or is
  over the tool strip or the HUD. It replaces the single-point cyan hover while
  the brush is on.
- **An unmodified left press starts a stroke.** Every point the circle passes
  over while the button is held is **painted**: drawn red instead of its
  colour. A press and release without a move paints one circle and is a
  stroke like any other.
- **Release deletes the painted points** as one version of the selected
  reconstruction, and the painted set is cleared. A stroke that painted nothing
  creates no version and logs nothing.
- **Esc while the button is held cancels the stroke**: the painted points go
  back to their colours and nothing is deleted. This follows the bench handle
  drag (`viewer_3d/input.rs`), which Esc cancels in the same way.
- **Alt + left drag orbits.** Orbit is the unmodified left drag in Navigate,
  which the brush takes, so while the brush is on, Alt + left drag orbits
  around the target in its place. In Navigate, Alt + left drag is the nodal pan;
  the brush gives that up, because cleanup alternates between orbiting and
  erasing and the nodal pan is rarely wanted there. In camera view Alt + left
  drag already orbits, so there it means the same in both tools. Alt + click
  still sets the target, as in Navigate.
- **Everything else still navigates.** The brush claims only the unmodified
  left press. Middle drag, right drag, Shift and Ctrl left drags, Alt + Shift
  left drag, the scroll wheel, pinch, the precision touchpad's two-finger pan
  (which orbits when unmodified) and the fly keys work as in Navigate, so the
  person can orbit, pan and zoom without changing tools. A right *click* still
  opens the point menu. On
  Windows every button is reported as Primary
  (`platform::other_mouse_button_down`), so the brush checks that no other
  button is down, as the bench handle drag does; otherwise a right-drag zoom
  would also paint.
- **Bench handles do not respond** while the brush is on. A left press on one
  starts a stroke instead.
- **Double-click does nothing special.** It is two strokes.

### What the brush can reach

**The brush paints the points the person can see, and only those.** It reads
the same GPU pick buffer a click does, so the front-most splat at each pixel
inside the circle is what it paints, and a point hidden behind another is not
painted. The person sees nothing that is not a candidate, and every candidate
is in plain view.

Painted points stay drawn (red) for the rest of the stroke, so they keep hiding
whatever is behind them. Holding the brush in one place therefore never digs
through the cloud. After release the deleted points are gone, the points that
were behind them show, and the next stroke can take those. Clearing a thick
clump of floaters takes a few strokes over the same place, one layer each.

It follows that:

- **Hidden layers are untouched.** A reconstruction whose eye is off, one hidden
  by a solo, points at infinity with their toggle off, and a reconstruction
  marked non-interactive (its pixels pick as "none") are never painted.
- **Points at infinity are painted like any other point** when they are shown.
  They are drawn as billboards at the far plane and picked from the same
  buffer, so the brush needs no rule for `w = 0` rows and does no arithmetic on
  positions.
- **Points drawn as patches are painted through their patch's pixels**, which
  write the point's pick ID as the splats do.
- **Frustums, image quads and the bench layer** are not points. The brush skips
  their pixels.

### Only the selected reconstruction

**The brush edits the selected reconstruction and no other.** Points of other
reconstructions are never previewed or painted, though they still hide what is
behind them, since the brush reads what is on screen. A version belongs to one
node and Undo acts on the selected reconstruction (`app/menu.rs`), so the
reconstruction a stroke edits is always the one the next Ctrl+Z undoes, and the
brush never changes the selection.

The selection is fixed for the length of a stroke: the stroke records the
reconstruction at the press, and a selection change while the button is held
(by `]`, which steps it) cancels the stroke as Esc does.

With no reconstruction selected, the circle is drawn grey, nothing is
previewed, and a press paints nothing. The status line then says `Select a
reconstruction to use the delete brush.`

### What the version says

The version's label and its Action Log entry say `Deleted 3412 points in
<label>`, with `1 point` in the singular, and the Action Log entry carries the
version step text as every edit's does. The entry has no "brush" suffix,
because the version is the same whether the brush or the wire deleted the
points.

The edit follows the existing rules for a point edit: a selection on a deleted
point is cleared and an undo does not restore it
([`../gui/edit-history.md`](../gui/edit-history.md)), and the version's map is
`PointMap::Removed` of the painted indices, sorted ascending, which
`PointMap::is_live` requires.

A node that is busy with a background task refuses the edit when the stroke is
released. The refusal is the node's busy sentence in the status line, and the
painted points go back to their colours. Refusing at the press instead would
have to give a reason for a stroke the person has not finished making, so the
check is made once, at the commit.

## Interfaces

### Core: `EditedReconstruction::delete_points`

```rust
impl EditedReconstruction {
    /// Delete every point in `indices`, or none of them.
    ///
    /// Every index is checked before any is deleted, so a refusal leaves the
    /// value as it was. Duplicates are allowed and count once.
    pub fn delete_points(&mut self, indices: &[u32]) -> Result<(), EditError>;
}
```

It is the many-point form of `delete_point`, beside it in
[`crates/sfmtool-core/src/reconstruction/edited.rs`](../../crates/sfmtool-core/src/reconstruction/edited.rs),
and it is atomic because a version must not hold half a stroke. It is bound
through `sfmtool-py` as `EditedReconstruction.delete_points`, so a script can
make the same edit offline (`specs/gui/edits/README.md`: every edit family's
mechanism is a core function bound through `sfmtool-py`).

A point edit's unshared cost is its overlay, so a stroke of `N` points costs
`4N` bytes per version beyond the overlay it shares, and the overlay is cloned
once per version as it is today. A stroke of 100 000 points is 400 kB, small
against the 4 GiB history budget.

### Viewer: `AppState::delete_points`

```rust
impl AppState {
    /// Delete the points `points` of reconstruction `recon` as one version.
    ///
    /// `points` need not be sorted or unique. Returns the sentence the caller
    /// reports when there is nothing to delete or the node is busy.
    pub(crate) fn delete_points(&mut self, recon: ReconId, points: &[u32]) -> Result<(), String>;
}
```

It sits beside `delete_point` in `state/edits.rs`, which becomes the
one-element call of it, so the two cannot drift apart in their labels, their
maps or how they move the selection. The brush is reported out of the viewport
as a gesture, as every point gesture is (`PointGesture`), because the panel
holds the node borrowed while it draws.

### Wire

- **`delete_points`** `{reconstruction_label, points: [int]}`: the same edit,
  one version, with the same reply shape as `delete_point`. It is how an agent
  cleans a reconstruction and how a test makes a large delete without a pointer.
- **The tool state** is one field on `get_viewer_3d_display` /
  `set_viewer_3d_display`: `tool` (`"navigate"` or `"delete_brush"`).
- **`drag`** `{x, y, path: [[x, y], ...], button, modifiers}`, an input tool
  beside `click` and `hover`: press at the first point, move through the path
  one point per frame, release at the last. `click` presses and releases in one
  frame, so nothing on the wire can hold a button across frames, and without
  `drag` a stroke cannot be driven from MCP, so `gui-bug-bash` cannot exercise
  the brush. It is useful beyond the brush (orbit, bench handles), and it can
  ship separately.

## Implementation notes

### Reading the circle from the pick buffer

The pick for a click reads a 5×5 pixel patch around the cursor
(`scene_renderer/readback.rs`). A stroke reads a larger region of the same R32
pick texture: the bounding box of the area the circle swept since the last
frame, which is a capsule from the previous cursor position to the current one,
thickened by the radius in physical pixels. Each pixel inside the capsule whose
tag is `PICK_TAG_POINT` decodes to a node and point index through the
per-reconstruction pick ranges, as a click's pick does, and joins the painted
set if it belongs to the reconstruction the stroke recorded at the press.

Reading the swept capsule rather than the circle at each frame's position is
what stops a fast drag from leaving gaps: the pointer can move further than the
diameter between two frames.

The preview reads the circle's own bounding box at the pointer, with the same
decode. It needs a region read on every frame the brush is on and the pointer is
over the viewport, because the view can move under a still pointer; when neither
the pointer nor the view changed since the last read, the last result stands and
nothing is copied. The brush's region read replaces the 5×5 hover read on those
frames rather than adding to it.

The preview's region is small: the 24-pixel radius at a pixels-per-point of 2
is a 97 × 97 box, 37 kB of `u32`. A stroke's capsule is longer in proportion to
how far the pointer moved in a frame; a 400-pixel flick at the same scale is
about 900 × 100, 360 kB. Rows are padded to wgpu's 256-byte row alignment as the
existing copy pads them. The existing readback blocks on the buffer map each
frame (`readback.rs`); that wait should be measured for the largest capsules,
and the read moved to a mapping that resolves a frame later if it shows in the
frame time.

A readback resolves a frame after its copy, as a click's does. On release the
commit waits for the last capsule's readback, so the final segment of a stroke
is not lost; the release is held for that one frame and then the edit runs.

### Drawing the painted points

The point buffers already carry one `u32` flag per point beside the instance
data, written by `SceneRenderer::update_point_mask`
(`scene_renderer/upload/overlay.rs`): `0` deleted, `1` alive, `2` alive and
drawn in the hover cyan, which the Move Camera lock uses for the points its
camera observes. The brush adds `3`, **alive and painted**, drawn in a red the
fragment shader picks by the same branch. It is a new value in the existing
word, not a second buffer, following the comment there that keeps highlights
in that word. The preview adds `4`, **alive and under the brush**, drawn in the
pale red. The brush and the lock are never on together, but the brush gets its
own values and colours so that the highlights do not mean the same thing. The
preview set changes on most frames the pointer moves, which makes the run
coalescing below matter for the preview as much as for the stroke.

`update_point_mask` writes one 4-byte `write_buffer` per changed index. A stroke
can add thousands of points in one frame, so the change list should be written
as contiguous runs, one `write_buffer` per run, or the whole buffer rewritten
when the changes exceed some fraction of it. Which one is worth it should be
measured on a dense cloud (the 1.4M-point example in `scene-graph.md`) before
picking.

### Who owns the stroke

The stroke is viewport state, like `bench_drag`: a `BrushStroke` on `Viewer3D`
holding the node, the painted set (a `HashSet<u32>`), and the last cursor
position. It takes the pointer the way `bench_drag` does: when it is held,
`handle_drag` and `handle_click` are skipped. The painted set is passed to the
renderer each frame with the deleted set, and is reported as a gesture on
release.

## Testing

- **Core:** `delete_points` deletes each index once; a list holding an index
  out of range or already deleted refuses and leaves the value unchanged;
  duplicates count once; an empty list is a no-op. A Python test through the
  binding.
- **Viewer state** (headless, `state/edits/tests`): `delete_points` pushes one
  version whose map is the sorted unique list, its label in the singular and the
  plural, the selection cleared when it was one of the points, and one Ctrl+Z
  restores all of them; `delete_point` still produces what it did.
- **The stroke** (headless, `viewer_3d/tests.rs`): with a synthetic pick region,
  the capsule test paints the pixels within the radius of the segment and no
  others; pixels of a reconstruction other than the selected one are neither
  previewed nor painted; with none selected nothing is painted; a selection
  change while the button is held cancels the stroke; the preview at
  a position equals the set a press-and-release there paints; Esc while held paints nothing and pushes nothing; a release with an
  empty set pushes nothing; a press with the right button down on Windows does
  not start a stroke; with the brush on, an Alt + left drag orbits (in both the
  free view and camera view) and paints nothing.
- **The flag** (`scene_renderer/upload/tests.rs`, `noop` backend): previewed
  points get `4` and go back to `1` when the pointer moves off them; painted
  points get `3`, go back to `1` on cancel, and to `0` after the commit.
- **Wire** (`mcp/tests/edit.rs`): `delete_points` makes one version and its
  reply names it; `set_viewer_3d_display` switches the tool and refuses
  `delete_brush` while the Move Camera lock is held.
- **End to end** (`ui_basic`, once `drag` exists): a drag across a known
  region of the seoul bull ground truth deletes points, and their count matches
  the Action Log entry.

## Non-goals

- **Reaching through the cloud.** The brush paints only visible points, by
  design. An X-ray mode is a different tool.
- **Unpainting within a stroke**, and pressure or pen input.
- **Editing a reconstruction other than the selected one**, or more than one
  in a stroke.
- **Deleting by a point's properties** (reprojection error, track length,
  distance from the subject). That is a filter, not a brush.
- **Multi-select.** The painted set does not outlive the stroke.
