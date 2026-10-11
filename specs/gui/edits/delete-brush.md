# Delete Brush

A reconstruction often has clusters of bad points: floaters in front of the
subject, sky, a spray of points from one bad image, background nobody wants.
The delete brush is a paint tool in the SfM Explorer's 3D viewport for removing
them: the cursor becomes a circle, the person drags it across the points they
want gone, the points it passes over turn red, and when the mouse button is
released those points are deleted from the selected reconstruction as one
version, which one Undo reverses. Before the press, the circle shows in a pale
red the points a press there would paint.

The brush is a cleanup tool, not a selection. It holds nothing once the stroke
ends, so it leaves the design of multi-select open. It has no depth modes, no
unpaint gesture and no radius setting: the radius is fixed at 24 logical
pixels.

The brush lives in
[viewer_3d/brush.rs](../../../crates/sfm-explorer/src/viewer_3d/brush.rs), with
headless tests in
[viewer_3d/brush/tests.rs](../../../crates/sfm-explorer/src/viewer_3d/brush/tests.rs).
The edit it commits, `AppState::delete_points`, is specified with the other
point edit in [`../document-model.md`](../document-model.md) § "Two kinds of
edit", and its core form, `EditedReconstruction::delete_points`, in
[`../../core/reconstruction/edited-reconstruction.md`](../../core/reconstruction/edited-reconstruction.md)
§ "The overlay: a base plus its edits".

## Invocation

### Choosing the tool

A **tool strip** floats at the left edge of the 3D viewport, centred
vertically. It holds two icon buttons, one above the other, and exactly one of
them is on:

| Icon | Tool | Tooltip |
|------|------|---------|
| Mouse pointer (top) | **Navigate**: everything the viewport's navigation does ([`../viewport-navigation.md`](../viewport-navigation.md)) | `Navigate: orbit, pan and zoom the view, and click to select.` |
| Eraser (below) | **Delete brush** | `Delete brush: drag over points to delete them from the selected reconstruction.` |

The icons are drawn with the painter rather than taken from a font, so they do
not depend on which glyphs egui's bundled fonts carry. Each button reports
itself to accessibility as a radio button named for its tool, which is how
`get_widgets` lists it.

The strip holds nothing else. A tool is chosen by clicking its icon, and there
is no keyboard shortcut for either. The viewer starts in Navigate, and the tool
(`Viewer3D::tool`) is kept for the session and not saved. A choice writes a
`Display` entry to the Action Log, `Delete brush tool` or `Navigate tool`; two
choices less than a second apart fold into one entry holding the last.

The strip follows the HUD's input rules
([`../viewport-hud.md`](../viewport-hud.md) § "Input arbitration"): it is an
`egui::Area` on a layer above the viewport, so a click on it never reaches the
viewport, and its rect (`Viewer3D::tool_strip_rect`) joins the HUD's in the
scroll and gesture gates. The left edge is used because the four corners and
the top centre are already taken
([`../user-experience.md`](../user-experience.md) § "Information overlays"),
and because paint programs put their tools there. The strip is not a HUD
section: the HUD holds display settings and collapses to a gear, while the
current tool has to be visible and one click away at all times, because it
changes what a left drag does.

**The Move Camera lock leaves the brush.** Entering the lock
([`move-camera.md`](move-camera.md)) switches the tool to Navigate, and the
brush button is greyed until the lock ends, with the tooltip `A camera is being
moved; M or Enter commits it, Esc cancels it.` The lock commits a pose and is a
different edit in progress, and a deletion pushed in the middle of it would put
a version between the lock's start and its commit.

### The stroke

With the brush on, the pointer over the viewport shows a circle outline of
radius `BRUSH_RADIUS`, 24 logical pixels, with the crosshair cursor at its
centre. egui has no circle cursor, so the circle is painted on the viewport's
painter, as the lock banner is. It is light grey before a press and red while a
stroke is painting.

- **Before the press, the circle previews.** The selected reconstruction's
  points a press at this spot would paint are drawn in a pale red, so the
  person sees what a stroke would take before starting it. The preview follows
  the pointer and is read again when the view moves under a still pointer (a
  scroll zoom, the fly keys, an Alt drag) or the reconstruction's version
  changes. It is cleared when the pointer leaves the viewport or is over the
  tool strip or the HUD. It replaces the single-point cyan hover while the
  brush is on: the viewport hovers nothing.
- **An unmodified left press starts a stroke.** Every point the circle passes
  over while the button is held is **painted**: drawn red instead of its
  colour. A press and release without a move paints one circle and is a stroke
  like any other; a click therefore selects nothing while the brush is on.
- **Release deletes the painted points** as one version of the reconstruction
  selected at the press, and the painted set is cleared. A stroke that painted
  nothing creates no version and logs nothing.
- **Esc while the button is held cancels the stroke**: the painted points go
  back to their colours and nothing is deleted. The rest of the drag does
  nothing until the button comes up, as after Esc on a bench handle drag
  ([`../viewer-3d-bench-layer.md`](../viewer-3d-bench-layer.md)).
- **Alt + left drag orbits.** Orbit is the unmodified left drag in Navigate,
  which the brush takes, so while the brush is on, Alt + left drag orbits
  around the target in its place. In Navigate, Alt + left drag is the nodal pan
  in free view; the brush gives that up, because cleanup alternates between
  orbiting and erasing and the nodal pan is rarely wanted there. In camera view
  Alt + left drag already orbits, so there it means the same in both tools.
  Alt + click still sets the target.
- **Everything else still navigates.** The brush claims only the unmodified
  left press. Middle drag, right drag, Shift and Ctrl left drags, Alt + Shift
  left drag, the scroll wheel, pinch, the precision touchpad's two-finger pan
  (which orbits when unmodified) and the fly keys work as in Navigate, so the
  person can orbit, pan and zoom without changing tools. A right *click* still
  opens the point menu. On Windows every button is reported as Primary
  (`platform::other_mouse_button_down`), so the brush checks that no other
  button is down, as the bench handle drag does; otherwise a right-drag zoom
  would also paint.
- **Bench handles do not respond** while the brush is on. A left press on one
  starts a stroke instead.
- **Double-click does nothing special.** It is two strokes.

With the controls help on (the HUD's Debug section), its line names the
brush's bindings while the brush is on: `Drag: delete points | Esc: cancel
stroke | Alt+drag: orbit | Shift: pan | Scroll: zoom | WASD: fly`.

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

- **Hidden layers are untouched.** A reconstruction whose eye is off, one
  hidden by a solo, points at infinity with their toggle off, and a
  reconstruction marked non-interactive (its pixels pick as "none") are never
  painted.
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
node and Undo acts on the selected reconstruction, so the reconstruction a
stroke edits is always the one the next Ctrl+Z undoes, and the brush never
changes the selection. The viewport is drawn only while a reconstruction is
selected, so the brush always has one to edit.

The selection is fixed for the length of a stroke: the stroke records the
reconstruction at the press, and a selection change while the button is held
(by `]`, which steps it) cancels the stroke as Esc does. So does a change of
tool, by the wire or by the lock.

## The version

The version is `AppState::delete_points`'s, so its label and its Action Log
entry are that edit's: `Deleted 3412 points in <label>`, or `Deleted point 7 in
<label>` for a stroke that painted one point, with the version step text the
Action Log entry carries for every edit
([`../document-model.md`](../document-model.md) § "Two kinds of edit"). The
entry has no "brush" suffix, because the version is the same whether the brush
or the wire deleted the points.

The edit follows the rules for a point edit: a selection on a deleted point is
cleared and an undo does not restore it
([`../edit-history.md`](../edit-history.md)), and the version's map is
`PointMap::Removed` of the painted indices, sorted ascending, which
`PointMap::is_live` requires.

A node that is busy with a background task refuses the edit when the stroke is
released. The refusal is the node's busy sentence in the status line, and the
painted points go back to their colours. Refusing at the press instead would
have to give a reason for a stroke the person has not finished making, so the
check is made once, at the commit.

A point edit's unshared cost is its overlay, so a stroke of `N` points costs
`4N` bytes per version beyond the overlay it shares, and the overlay is cloned
once per version as for any point edit. A stroke of 100 000 points is 400 kB,
small against the 4 GiB history budget.

## The wire

- **The tool** is one field of `get_viewer_3d_display` /
  `set_viewer_3d_display`: `tool`, `"navigate"` or `"delete_brush"`
  ([`../mcp-server.md`](../mcp-server.md) § "`get_viewer_3d_display` /
  `set_viewer_3d_display`"). Setting `delete_brush` while a camera is being
  moved is refused, and the call changes nothing.
- **`drag`** `{panel_name, path_px, mouse_button, modifiers}` is how a stroke
  is driven from MCP, and so how `gui-bug-bash` exercises the brush
  ([`../mcp-server.md`](../mcp-server.md) § "`drag`"). `click` presses and
  releases in one frame, so it paints one circle. The stroke's version lands a
  frame or two after `drag` replies, when the release's last region read
  arrives.
- **`delete_points`** `{reconstruction_label, points: [int]}` is the same edit
  without the brush ([`../mcp-server.md`](../mcp-server.md) §
  "`delete_point` / `delete_points` / `delete_camera_image`").

## Mechanism

### Who owns the stroke

The stroke is viewport state, like the bench handle drag: a `BrushStroke` on
`Viewer3D` holding the node selected at the press, the painted set (a
`HashSet<u32>` of edited indexes) and the last cursor position. It takes the
pointer the way the bench drag does: while it is held, `handle_drag` and
`handle_click` are skipped. A cancelled stroke keeps the pointer until the
button comes up, so the rest of the drag does not orbit.

A released stroke is reported out of the viewport as a `BrushGesture`, as the
point menu's gestures are, because the panel holds the node borrowed while it
draws. `dock.rs` takes it after the frame and calls `AppState::delete_points`.

The frame splits in two around the viewport's own input. Before it,
`begin_brush_frame` takes an answer that arrived, starts a stroke on an
unmodified press and cancels one that Esc, a selection change or a change of
tool ended. After it, `end_brush_frame` asks for the next region, from the view
the input left, which is the picture the next frame draws.

### Reading the circle from the pick buffer

The brush reads what is on screen through the renderer's region read
([`../camera-views.md`](../camera-views.md) § "Region reads"): a rectangle of
the pick buffer requested a frame ahead, decoded per pixel to a point, and the
pure capsule test, `points_in_capsule`, that picks one reconstruction's points
out of it. The answer to a request made in frame N is seen in frame N + 2.

- **A stroke reads the swept capsule**: the segment from the previous cursor
  position to the current one, thickened by the radius in physical pixels,
  requested as `PixelRect::around_capsule` and filtered with
  `points_in_capsule` for the reconstruction the stroke recorded at the press.
  Reading the swept capsule rather than the circle at each frame's position is
  what stops a fast drag from leaving gaps: the pointer can move further than
  the diameter between two frames.
- **The preview reads the circle** (a capsule of zero length) at the pointer,
  and each answer replaces the preview.
- **A region is asked for only when something changed**: the pointer, the
  camera's position, orientation or field of view, the viewport's rect or
  pixels per point, or the node's version. Otherwise the last answer stands
  and nothing is requested, and the 5×5 hover read runs in its place.
- **Each request is kept with its capsule until its answer arrives**, and the
  answer is matched to it by the rectangle it named (`PickRegion::requested`).
  A request that names nothing left on the texture, such as a stroke dragged
  off the viewport, is never answered, and is given up after three frames.
- **On release the commit waits for the last capsule's answer**, so the final
  segment of a stroke is not lost; the stroke is held for the frames until it
  arrives and then reported. The viewport asks for repaints while any request
  is outstanding, so those frames come without input.

### The map blocks

`read_pick_region` blocks on the staging buffer's map, as the 5×5 read does
([`../camera-views.md`](../camera-views.md) § "Region reads"), and logs each
wait at `debug` level. Measured on the seoul bull ground truth in a debug build
on a Windows desktop, the wait was 3 to 5 ms for every region from a 48 × 48
circle to an 88 × 48 stroke segment, 4.2 ms for a 588 × 128 flick and 5.3 ms
for a 1148 × 548 one (2.5 MB). The wait is almost all the GPU finishing the
frame that was just submitted, which the 5×5 read waits for as well, rather
than the copy's size, so the read is not moved to a mapping that resolves a
frame later.

### Drawing the painted points

The per-point mask word `SceneRenderer::update_point_mask` writes carries `3`
for alive and painted (red) and `4` for alive and under the brush (pale red),
drawn on splats and on surfels
([`../point-cloud-rendering.md`](../point-cloud-rendering.md) § "Per-Point
Data"). Each frame `App::prepare_uploads` passes the stroke's painted set and
the preview's set (`Viewer3D::brush_marks`) as the `painted` and `under_brush`
sets of the `PointMarks` it hands `update_point_mask`, beside the deleted set.
The mask write goes out as runs of consecutive positions, one `write_buffer`
per run. The preview set changes on most frames the pointer moves, and whether
bridging gaps between runs or rewriting the whole buffer is cheaper on a dense
cloud (the 1.4M-point example in [`../scene-graph.md`](../scene-graph.md)) was
not measured ([`../document-model.md`](../document-model.md) § "Change
detection by identity").

## Testing

- **Core and viewer state:** `delete_points` is tested where it is specified,
  in `sfmtool-core`'s `reconstruction/edited/tests.rs` and in
  `state/edits/tests.rs`.
- **The region read and the mask words** are tested where they are built:
  `scene_renderer/pick_region/tests.rs` (the capsule selects the pixels within
  the radius of the segment and no others, and skips another reconstruction's)
  and `scene_renderer/upload/tests.rs` (words `4` and `3` written and reset to
  `1`, and `3` becoming `0` once the painted set is the deleted set).
- **The stroke** (headless, `viewer_3d/brush/tests.rs`), with a synthetic
  picture standing in for the pick buffer and answering each region request at
  the frame's real latency: the preview at a position equals the set a press
  and release there paints, and neither touches a reconstruction other than the
  selected one; a stroke paints the capsules it swept, marks them painted while
  held and reports them sorted; the release waits for the last capsule; Esc
  while held paints nothing, reports nothing and does not orbit; a selection
  change or a change of tool while held cancels the stroke; a stroke over
  background reports nothing; a press with another button down, as Windows
  reports a right press, starts no stroke; with the brush on, an Alt + left
  drag orbits in free view and in camera view and paints nothing, where in
  Navigate the free-view Alt drag keeps the camera in place; the Move Camera
  lock switches to Navigate and refuses the brush; and the tool strip's buttons
  choose the tool and record it.
- **Wire** (`mcp/tests/edit.rs`): `set_viewer_3d_display` switches the tool,
  refuses an unknown one, and refuses `delete_brush` while the Move Camera lock
  is held, applying none of the call's other fields.
- **End to end** (`ui_basic`): the tool strip's button turns the brush on in a
  real window, a `drag` across the middle of the seoul bull ground truth seen
  through its first camera deletes points, and their count in the Action Log
  entry is the count the reconstruction lost.

## Non-goals

- **Reaching through the cloud.** The brush paints only visible points, by
  design. An X-ray mode would be a different tool.
- **Unpainting within a stroke**, and pressure or pen input.
- **Editing a reconstruction other than the selected one**, or more than one
  in a stroke.
- **Deleting by a point's properties** (reprojection error, track length,
  distance from the subject). That is a filter, not a brush.
- **Multi-select.** The painted set does not outlive the stroke.
