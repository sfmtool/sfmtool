# Measurement View

**Status:** Draft

Amends [gui/measurement-view.md](../gui/measurement-view.md), which describes
what is built (steps 1 to 6 of § "Steps"): the selected measurement, the
per-version readings, the Scene tree's Measurements group, the Measurement
View panel with its edited mode and every step on a measurement item, the
menus' *Set Endpoint A Here*, *Set Endpoint B Here* and *New Measurement from
Here*, a point's reference pixel, the commit, deleting a stored measurement,
the *Bench Measurements* group, and their tools. The measurement item itself,
its steps, starting one from a pixel and its commit are
[core/bench/editable-measurement.md](../core/bench/editable-measurement.md)
and [gui/bench.md](../gui/bench.md) § "Measurement items". This draft keeps
the rest: drawing measurements in the 3D view and Image Detail. It also amends
[gui/bench.md](../gui/bench.md),
[gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md) and the
3D view's specs, as § "What changes in the standing specs" lists.

Decided by the maintainer:

- The 3D view draws each endpoint and a line between the two. Image Detail
  draws the same where both endpoints are in the image it shows, and the one
  endpoint that is in it otherwise.
- Only the selected measurement is drawn by default. A display option, off by
  default, draws every measurement. A selected measurement that is off screen
  gets an indicator at the edge of the view pointing to it.
- An endpoint's 3D position is never edited: it is always derived from the
  image and pixel, so a handle drag sets the pixel.
- The checked-in reconstructions have no measurements, so the drawing is built
  and tested on copies that are given artificial measurements
  ([gui/measurement-view.md](../gui/measurement-view.md) § "Artificial
  measurements"). Real measurements are not checked in yet.

Nothing in this draft is still open.

Related specs: [gui/measurement-view.md](../gui/measurement-view.md) (what is
built), [gui/bench.md](../gui/bench.md) (the bench's layer the focused item is
drawn as), [gui/edit-history.md](../gui/edit-history.md) (the version a handle
drag pushes), [gui/mcp-server.md](../gui/mcp-server.md) (the tools).

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
endpoint's mark is drawn in the warning colour. A measurement item with one
endpoint placed, such as a new one, draws that endpoint's mark alone.

**The focused item is drawn as the bench's layer** in Image Detail and the 3D
view ([gui/bench.md](../gui/bench.md)), in the bench's colours rather than the
node's tint, so a person can tell the measurement being edited from the stored
one it came from when both are drawn. Its endpoint marks in Image Detail are
handles: dragging one is the *Set an endpoint's image and pixel* step
([gui/measurement-view.md](../gui/measurement-view.md) § "Setting an endpoint
by hand"), one version when the drag ends, and a view-only bench draws no
handles. The selected endpoint's mark is drawn larger, as the 3D view draws the
selected observation's circle larger.

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

`get_viewer_3d_display`, `set_viewer_3d_display` and the Image Detail display
tools carry the option as `show_all_measurements`. The built tools are in
[gui/mcp-server.md](../gui/mcp-server.md). No tool sets an endpoint's 3D
position.

## Real measurements

`dino_dog_toy`'s four tape-measured distances, entered by hand with the panel,
are the first real ones. They are not checked in yet. When they are, the file
keeps its name: the `_vNN` suffix versions the reconstruction's cameras,
poses, points, tracks and patch frames, and adding measurements changes none
of those.

## What changes in the standing specs

- [gui/measurement-view.md](../gui/measurement-view.md): drawing, and the
  option on the wire.
- [gui/bench.md](../gui/bench.md): the measurement item in the bench's layer.
- [gui/multi-panel-image-browser.md](../gui/multi-panel-image-browser.md) and
  the 3D view's specs: the marks, the handles, the indicator and the *Show all
  measurements* option.

## Steps

Each step is one PR through `implement-change`, with its MCP tools, its
`ui_basic` coverage and screenshots. Steps 1 to 6 are built; see
[gui/measurement-view.md](../gui/measurement-view.md) (step 6 is its §
"Creating a measurement").

7. **Drawing.** The selected measurement's and the focused item's marks and
   lines in the 3D view and Image Detail, their hover and click, the handles,
   the off-screen indicator and its click, and the *Show all measurements*
   option in each view, with `get_viewer_3d_display`, `set_viewer_3d_display`
   and the Image Detail display tools carrying it as `show_all_measurements`.

## Testing

- Drawing: with the option off only the selected measurement and the focused
  item are drawn; with it on, every measurement of a visible reconstruction; a
  hidden reconstruction draws none either way; an item with one endpoint draws
  its one mark; a handle drag is one version; a view-only bench draws no
  handles.
- The off-screen indicator: absent while any part of the measurement is in
  view; pointing the right way for a midpoint to each side of the view and
  behind the camera; a click framing the measurement so both endpoints project
  inside the view.
- `ui_basic`: a measurement drawn in both views, and a handle dragged in Image
  Detail.
