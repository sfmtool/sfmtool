# Measurement View

**Status:** Draft

Amends [gui/measurement-view.md](../gui/measurement-view.md), which describes
what is built (steps 1 to 5 of § "Steps"): the selected measurement, the
per-version readings, the Scene tree's Measurements group with its row
gestures and menu, the Measurement View panel with its header, the distance
body and the endpoint gestures, its *Edit* box and edited mode with the live
readings, *Commit*, *Discard* and *Rename*, the header's label and stated
distance fields with their steps, each endpoint block's image drop-down and
pixel fields and its *Switch image…* with its popup and step, *Set Endpoint A
Here* and *Set Endpoint B Here* in Image Detail's menu and the 3D view's point
menu, a point's reference pixel, deleting a stored measurement, the *Bench
Measurements* group, and their tools. The measurement item itself, its origin,
the steps that set its label, its stated distance and an endpoint's image and
pixel, its commit and the direct delete are
[core/bench/editable-measurement.md](../core/bench/editable-measurement.md) and
[gui/bench.md](../gui/bench.md) § "Measurement items", and the offers of the
popup are `endpoint_views` in
[core/reconstruction/measurement-endpoint.md](../core/reconstruction/measurement-endpoint.md)
§ "The views of an endpoint". This draft keeps the rest: creating a
measurement, and drawing measurements in the 3D view and Image Detail. It also
amends [gui/bench.md](../gui/bench.md), [gui/scene-graph.md](../gui/scene-graph.md),
[gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md) and the
3D view's specs, as § "What changes in the standing specs" lists.

Decided by the maintainer:

- **A measurement is edited on the bench, as a track is.** Every step on an
  item is a bench version that leaves the reconstruction alone, and the commit
  is the one step that writes the measurement into the file. A new measurement
  is a bench item from its first endpoint on, with no state outside the
  history.
- **The bench of a `sift_files` reconstruction stays view-only** for
  measurements too, with the same sentence: every step that edits an item is
  refused.
- The first edit is switching an endpoint to another image: the patches whose
  square in the endpoint's image holds the endpoint's pixel are found as the
  resolver finds them, and every other observation of those tracks is offered,
  with a crop of where the endpoint falls in each, so a person can pick the
  image that shows the endpoint best.
- An endpoint's image and pixel can also be set directly. Its 3D position is
  never edited: it is always derived from the image and pixel.
- The 3D view draws each endpoint and a line between the two. Image Detail
  draws the same where both endpoints are in the image it shows, and the one
  endpoint that is in it otherwise.
- Only the selected measurement is drawn by default. A display option, off by
  default, draws every measurement. A selected measurement that is off screen
  gets an indicator at the edge of the view pointing to it.
- The checked-in reconstructions have no measurements, so the panel is built
  and tested on copies that are given artificial measurements
  ([gui/measurement-view.md](../gui/measurement-view.md) § "Artificial
  measurements"). Real measurements are not checked in yet.

Nothing in this draft is still open.

Related specs: [gui/measurement-view.md](../gui/measurement-view.md) (what is
built), [gui/bench.md](../gui/bench.md) and
[core/bench/editable-measurement.md](../core/bench/editable-measurement.md)
(the item the steps edit), [gui/track-view.md](../gui/track-view.md) (the panel
whose toolbar and fields Measurement View follows),
[gui/document-model.md](../gui/document-model.md) and
[gui/edit-history.md](../gui/edit-history.md) (the versions a step pushes),
[gui/mcp-server.md](../gui/mcp-server.md) (the tools).

## Measurement View's edited mode, with the steps

The edited mode is built: the toolbar, the header with its label and stated
distance fields ([gui/measurement-view.md](../gui/measurement-view.md) § "The
label and the stated distance"), the endpoint blocks with their image
drop-down and pixel fields (§ "Setting an endpoint by hand" there), *Switch
image…* and the right-click menu holding *Switch Image…* (§ "Switching an
endpoint's image" there), and the live readings. The remaining step adds one
thing to it:

- an endpoint not yet placed is a block saying how to place it: "Right-click a
  pixel in Image Detail, or a point in the 3D view, and choose Set Endpoint B
  Here." (As built, the block says only `Endpoint B is not placed.`, though the
  entries it names exist.)

## Steps on an item

Each is a bench step: one version through `History::push_bench`, listed in the
Edit History and undone by `Ctrl+Z`, with an Action Log row, refused while a
background task holds the node and on a view-only bench. None changes the
reconstruction or marks the node dirty. Each is a pure core function from one
`EditableMeasurement` to the next, which replaces the item's `Arc` under the
same `ItemId`. The steps built so far, setting the label, the stated distance
and an endpoint's image and pixel, are made this way
([gui/measurement-view.md](../gui/measurement-view.md) § "The label and the
stated distance", § "Switching an endpoint's image" and § "Setting an endpoint
by hand").

| Step | Label in the Edit History |
|---|---|
| Switch an endpoint's image | `Moved endpoint A of distance #2 from IMG_0004.jpg to IMG_0011.jpg` (built) |
| Set an endpoint's image and pixel | `Moved endpoint A of distance #2 to IMG_0004.jpg (412.3, 230.5)` (built: the endpoint block's fields, *Set Endpoint A Here* and *Set Endpoint B Here*, and the wire's `set_bench_measurement_endpoint`) |
| Place an endpoint not yet placed | `Set endpoint B of distance from IMG_0042@142,198 to IMG_0050.jpg (88.0, 301.5)` |

Placing an endpoint not yet placed is the built step too, labelled `Set
endpoint B of distance #2 to ...`, and the two menus' entries already place
one; what step 6 adds is the item with one endpoint that it starts from, and
its label.

## Creating a measurement

**Starting it.** Two menus offer `New Measurement from Here`:

- **Image Detail's context menu**, at the clicked pixel of the image shown.
  Any pixel inside the image will do; the endpoint does not have to resolve,
  since the pixel is the stored measurement.
- **The 3D view's point menu.** A right-click on a point selects it (G3) and
  offers the entry, which starts the measurement at the point's reference
  pixel ([gui/measurement-view.md](../gui/measurement-view.md) § "A point's
  reference pixel", `measurements::reference_pixel`, the helper *Set Endpoint
  A Here* reads). The entry is greyed with that section's three sentences, as
  *Set Endpoint A Here* is, where the point has no reference pixel.

Either puts a new measurement item on the node's bench with endpoint A at that
pixel and nothing else, focuses it, and raises Measurement View beside the
panel the gesture was made in (G9), in one version (`Put distance from
IMG_0042@142,198 on the bench`). The new item has no origin, and its bench
label is minted from where it started, `distance from IMG_0042@142,198`, with
the bench's collision suffix. Endpoint B is then placed with `Set Endpoint B
Here` from the same menus, and the stated distance and label are typed into
the panel. Each is a step, so a mistake is undone like any other, and a
half-made measurement waits on the bench, unfocused, until it is finished or
discarded. *Commit* is greyed, with the refusal naming what is missing, until
both endpoints and the distance are given, and appends the measurement as a
new row, as it does for an item with no origin today.

## Drawing measurements

**Which measurements are drawn.** The selected measurement, when its
reconstruction is visible, and the focused measurement item. A **Show all
measurements** option, off by default, draws every stored measurement of every
visible reconstruction as well. A reconstruction with many measurements stays
readable by default, and the person who wants to see them all together asks
for it. The option is in the 3D view's HUD and in Image Detail's display
controls, one for each view, as their other display options are. With it on,
the selected measurement is drawn wider and brighter than the others.

**In the 3D view**, a drawn measurement puts a mark at each endpoint that
resolves to a finite position, and a line between the two where both do. They
are drawn in the node's world space, through its display transform, as its
points are, in the node's tint, and over the cloud rather than hidden by it,
since a measurement is something a person looks for. An endpoint at infinity
or refused draws nothing, and the other endpoint still draws its mark.

**In Image Detail**, a drawn measurement with an endpoint in the image shown
puts a mark at that endpoint's pixel. Where both endpoints are in that image,
it also draws a line between them. The pixel is drawn whether or not the
endpoint resolves, since the pixel is the stored measurement; a refused
endpoint's mark is drawn in the warning colour.

**The focused item is drawn as the bench's layer** in Image Detail and the 3D
view ([gui/bench.md](../gui/bench.md)), in the bench's colours rather than the
node's tint, so a person can tell the measurement being edited from the stored
one it came from when both are drawn. Its endpoint marks in Image Detail are
handles: dragging one is the *Set an endpoint's image and pixel* step, one
version when the drag ends, and a view-only bench draws no handles. The
selected endpoint's mark is drawn larger, as the 3D view draws the selected
observation's circle larger.

Hovering a mark or a line in either view shows the measurement's row text. A
click on a line selects the measurement; a click on a mark selects the
measurement and that endpoint ([gui/measurement-view.md](../gui/measurement-view.md)
§ "Endpoint gestures"), and a double-click on it does what a double-click on
its block does. A mark owns its press (G7).

**A selected measurement off screen.** When no part of the selected
measurement's marks and line is inside the view, the view draws an indicator
at its edge: an arrow on the border pointing towards the measurement, with
its kind and index (`Distance #2`) beside it.

- In the 3D view the arrow points towards the projection of the midpoint of
  the two endpoints, or of the one finite endpoint. For a point behind the
  camera, the direction is the projection's reflected through the view's
  centre, so the arrow points the way to turn. A click on the arrow frames
  the measurement (`Viewer3D::frame_keeping_orientation`).
- In Image Detail, when an endpoint is in the image shown but outside the part
  the panel shows, the arrow points to that pixel, and a click centres the
  panel on it at the current zoom.

The indicator follows the selection only. With *Show all measurements* on,
the others are clipped as any scene geometry is.

## MCP

A new tool names the remaining step, refused in the panel's words:

- `create_bench_measurement(reconstruction_label, camera_image, pixel)`. An
  agent starts from a point as the 3D view's menu does by passing what the
  built `get_point_reference_pixel` gives it.

The built tools (`put_measurement_on_bench`, `set_bench_measurement_label`,
`set_bench_measurement_distance`, `get_measurement_endpoint_views`,
`set_bench_measurement_endpoint`, which takes any pixel inside the image as the
panel's fields do, `get_point_reference_pixel`, `commit_bench_measurement`,
`delete_measurement`,
`get_measurement` naming an item, and the bench tools on a measurement item)
are in [gui/mcp-server.md](../gui/mcp-server.md). No tool sets an endpoint's
3D position.

## Real measurements

`dino_dog_toy`'s four tape-measured distances, entered by hand once the panel
can place measurements, are the first real ones. They are not checked in yet.
When they are, the file keeps its name: the `_vNN` suffix versions the
reconstruction's cameras, poses, points, tracks and patch frames, and adding
measurements changes none of those.

## What changes in the standing specs

- [gui/measurement-view.md](../gui/measurement-view.md): the unplaced
  endpoint's instructions, creating, drawing, and the tools.
- [gui/bench.md](../gui/bench.md): a new measurement item's label.
- [gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md) and
  the 3D view's specs: *New Measurement from Here*, the marks, the handles, the
  indicator and the *Show all measurements* option.

## Steps

Each step is one PR through `implement-change`, with its MCP tools, its
`ui_basic` coverage and screenshots.

1. **Read-only.** Built; see [gui/measurement-view.md](../gui/measurement-view.md).
2. **The item on the bench.** Built; see
   [gui/measurement-view.md](../gui/measurement-view.md),
   [gui/bench.md](../gui/bench.md) § "Measurement items" and
   [core/bench/editable-measurement.md](../core/bench/editable-measurement.md).
   Two parts were fitted to the code: the commit's value shares its whole base
   and holds measurements of its own on the `EditedReconstruction`
   ([core/reconstruction/edited-reconstruction.md](../core/reconstruction/edited-reconstruction.md)
   § "Measurements of the value's own"), rather than a new base sharing the
   point columns, which are not shared pointers; and each version keeps a
   measurement row map beside its point map, which an item's origin is
   followed through ([gui/document-model.md](../gui/document-model.md)).
3. **Editing the label and the stated distance.** Built; see
   [gui/measurement-view.md](../gui/measurement-view.md) § "The label and the
   stated distance" and
   [core/bench/editable-measurement.md](../core/bench/editable-measurement.md)
   § "The label and the stated distance". Fitted to the code: the stated
   distance is read by `parse_stated_distance` in core, which
   `sfm xform --scale-by-measurements` calls too, and which takes a space
   between the number and the unit and an exponent, so the field reads back
   the `153 cm` it shows; and the fields commit only text other than what
   they held when they took focus.
4. **Switching an endpoint's image.** Built; see
   [gui/measurement-view.md](../gui/measurement-view.md) § "Switching an
   endpoint's image",
   [core/reconstruction/measurement-endpoint.md](../core/reconstruction/measurement-endpoint.md)
   § "The views of an endpoint" and
   [core/bench/editable-measurement.md](../core/bench/editable-measurement.md)
   § "Setting an endpoint". Fitted to the code: each view carries the
   observation's `ℓ`, its viewing angle and the point the resolver picks at
   its pixel (`resolves_to`), so the popup's readings and its mark come from
   core in one call rather than from the panel; the step that sets an
   endpoint's image and pixel, its `Set` label and
   `set_bench_measurement_endpoint` are built with the switch, since the wire
   tool belongs to this step, leaving step 5 the panel's fields and the menus;
   and the endpoint blocks stack one above the other where the panel is too
   narrow for two side by side, which also stops a block's lines drawing over
   the other's at the default layout's width.
5. **Setting an endpoint by hand.** Built; see
   [gui/measurement-view.md](../gui/measurement-view.md) § "Setting an
   endpoint by hand" and § "A point's reference pixel",
   [gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md) §
   "Image Detail: the context menu",
   [gui/viewport-navigation.md](../gui/viewport-navigation.md) § "The point
   context menu" and [gui/mcp-server.md](../gui/mcp-server.md) §
   "`get_point_reference_pixel`". Fitted to the code: picking an image in the
   drop-down is the step at once, with the pixel the endpoint holds, since a
   drop-down has no Enter; a pixel field's text that is not a number is a
   failed row in the panel's sentence rather than a step; an unplaced
   endpoint's block has no fields, since there is no image or pixel to start
   from, and the menus place it; the entries are greyed by the step's own
   refusal too, a busy node or a view-only bench; a point's reference is the
   one it holds in the version on screen, a pick only the display render made
   included, and no column counts as no reference; and the greying on another
   node is reachable only on the frame a right-click lands on another node's
   point, since that selection unfocuses the item.
6. **Creating a measurement.** `New Measurement from Here` in both menus, an
   item with one endpoint and its label, the unplaced endpoint's block with
   its instructions, and `create_bench_measurement`.
7. **Drawing.** The selected measurement's and the focused item's marks and
   lines in the 3D view and Image Detail, their hover and click, the handles,
   the off-screen indicator and its click, and the *Show all measurements*
   option in each view, with `get_viewer_3d_display`, `set_viewer_3d_display`
   and the Image Detail display tools carrying it as `show_all_measurements`.

## Testing

- Each step's label, undo, and refusal, including on a view-only bench; a step
  on the item changes its live readings and not the stored measurement's.
- Creating: starting from a pixel and from a point; the entry greyed, with its
  sentence, on a point with no patch, no reference observation, or at
  infinity; *Commit* greyed until B and the distance are given, then
  appending one row.
- Drawing: with the option off only the selected measurement and the focused
  item are drawn; with it on, every measurement of a visible reconstruction; a
  hidden reconstruction draws none either way; a handle drag is one version; a
  view-only bench draws no handles.
- The off-screen indicator: absent while any part of the measurement is in
  view; pointing the right way for a midpoint to each side of the view and
  behind the camera; a click framing the measurement so both endpoints project
  inside the view.
- `ui_basic`: a measurement made with the two menus and the panel's fields.
