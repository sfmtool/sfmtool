# Measurement View

**Status:** Draft

Amends [gui/measurement-view.md](../gui/measurement-view.md), which describes
what is built: the selected measurement, the per-version readings, the Scene
tree's Measurements group with its row gestures, the Measurement View panel
with its header, the distance body and the endpoint gestures, `get_scene`'s
measurement counts, `list_measurements`, `get_measurement`,
`select_measurement`, and the artificial-measurements script (step 1 of
§ "Steps"). This draft keeps the rest: the edits, creating a measurement, and
drawing measurements in the 3D view and Image Detail.

Decided by the maintainer:

- The panel edits a measurement. The first edit is switching an endpoint to
  another image: the patches whose square in the endpoint's image holds the
  endpoint's pixel are found as the resolver finds them, and every other
  observation of those tracks is offered, with a crop of where the endpoint
  falls in each, so a person can pick the image that shows the endpoint best.
- The panel also edits an endpoint's image and pixel directly. An endpoint's
  3D position is never edited: it is always derived from the image and pixel.
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
built), [gui/track-view.md](../gui/track-view.md),
[gui/document-model.md](../gui/document-model.md) and
[gui/edit-history.md](../gui/edit-history.md) (the versions an edit pushes),
[gui/mcp-server.md](../gui/mcp-server.md) (the tools).

## What the later steps add to what is built

**The selection follows an edit to a measurement.** An edit to a measurement
keeps its index. Deleting a measurement renumbers the rows after it, and the
selection moves with them, or is cleared if it was the one deleted, as it
already does when a deleted image drops measurements.

**The Scene tree's rows gain a menu.** A right-click on a Measurements row
opens the row's menu (*Delete*) and selects nothing (rule G3 of
[gui/interaction-rules.md](../gui/interaction-rules.md)).

**The header's label and stated distance become editable in place**
(§ "Editing").

**Each endpoint block gains two controls**: *Switch image…*, the edit in
§ "Switching an endpoint's image", and the image and pixel as editable fields,
the edit in § "Setting an endpoint by hand", with the 3D position beside them
staying read-only. The block's own controls keep their clicks, and a
right-click on the block opens a menu with *Switch Image…* and selects nothing.
The selected endpoint's mark is drawn larger in the 3D view and Image Detail,
as the 3D view draws the selected observation's circle larger.

**From outside the panel**, a click on an endpoint's mark in Image Detail or
the 3D view selects the measurement and that endpoint, as a click on a bench
mark selects that observation's row, and a double-click on it does what a
double-click on its block does. The marks are drawn only for the measurements
§ "Drawing measurements" draws.

**An endpoint's patch picture** shows the chosen point's stored bitmap; where
the point stores none, its display bitmap is to be shown instead.

## Switching an endpoint's image

An endpoint sits on one texel `(s, t)` of a point's patch. Every other
observation of that point holds the same texel at a pixel of its own image, in
the observation's square (`OrientedPatch::seen_at_keypoint` projected through
that image's own intrinsics), which is exactly how the resolver read the texel
in the first place, run backwards. So the endpoint can move to any of those
images without moving on the surface.

**Which observations are offered.** The resolver's candidates for the endpoint
as it stands: every observation in the endpoint's image whose square holds the
pixel, chosen or not, excluded or not. For each candidate's point, every other
observation of it, at the candidate's texel. An offer is dropped where the
texel's place projects outside its image or the square faces away from that
camera. Where the endpoint is refused, nothing can be offered, and the button
says why in its hover text.

**The list.** A popup grouped by point, the chosen candidate's point first,
then the others in the resolver's order. Each offer is a row:

- the image name;
- a crop of that image around that observation's square with the cross at the
  offered pixel, with the same hover view as the endpoint's crop;
- the observation's self-similarity length `ℓ` and viewing angle, the readings
  the resolver's cost reads, so the sharper and less oblique views stand out;
- a mark where moving there would make the endpoint resolve to a different
  point than it does now, or to none, because another square holds the new
  pixel there.

A click on a row moves the endpoint to that image and pixel. The point the
endpoint resolves to, and the texel, are those of the row's point, unless the
mark above says otherwise.

**Core support.** A new function beside the resolver:

```rust
/// The pixels in other images that hold the same texel of each candidate's
/// patch as `(image, pixel)` does.
pub fn endpoint_views(
    edited: &EditedReconstruction,
    image: u32,
    pixel: [f64; 2],
    options: &MeasurementEndpointOptions,
) -> Result<Vec<EndpointView>, EndpointRefusal>;

pub struct EndpointView {
    pub point: u32,
    pub observation: usize,  // within the point's track
    pub image: u32,
    pub pixel: [f64; 2],
    pub patch_coords: [f64; 2],
}
```

It shares the resolver's refusals, so a person sees the same sentence in both
places.

## Setting an endpoint by hand

The image and the pixel are the endpoint; everything else the block shows is
derived from them. So both can be set directly, in two places:

- **In the endpoint block.** The image name is a drop-down of the
  reconstruction's images, and the pixel is two number fields. A change
  commits on Enter or when the field loses focus, as one edit that sets both,
  and is refused with the format's sentence where the pixel is not inside the
  image (`[0, width) × [0, height)` of that image's camera).
- **In Image Detail.** With a distance measurement selected, Image Detail's
  context menu gains `Move Endpoint A Here` and `Move Endpoint B Here`, which
  set that endpoint to the image shown and the pixel clicked. The context menu
  is the viewer's way to name a pixel
  ([multi-panel-image-browser.md](../gui/multi-panel-image-browser.md) §
  "Image Detail: the context menu"), so this is where a person zoomed in on a
  photograph places an endpoint precisely.
- **In the 3D view.** With a distance measurement selected, a right-click on a
  point, which selects it (rule G3 of
  [gui/interaction-rules.md](../gui/interaction-rules.md)), offers `Move Endpoint A Here` and
  `Move Endpoint B Here`, which set the endpoint to the point's **reference
  pixel** (§ "A point's reference pixel").

Neither offers the 3D position, the point or the texel: they follow from the
image and pixel through the resolver every time, and a setting that disagreed
with them could not be kept.

## Editing

Each edit is a version of the node, listed in the Edit History panel, undone by
`Ctrl+Z` and recorded in the Action Log, and marks the node dirty, so a save
writes it. Measurements live in the image set, in the base of the value, so an
edit to them is a **bulk edit** in the document model's terms
([document-model.md](../gui/document-model.md) § "Two kinds of edit"). It
replaces the measurements and nothing else: the new base must share the
points, tracks and patch bitmaps with the old one rather than copy them, and
the overlay is carried across unchanged, since no point index moves. If
`SfmrReconstruction` has no way to replace its measurements while sharing
those columns, the first step adds one.

The edits, in the order they are built:

| Edit | Label in the Edit History | Notes |
|---|---|---|
| Switch an endpoint's image | `Moved endpoint A of distance #2 in dino from IMG_0004.jpg to IMG_0011.jpg` | § "Switching an endpoint's image" |
| Set an endpoint's image and pixel | `Moved endpoint A of distance #2 in dino to IMG_0004.jpg (412.3, 230.5)` | § "Setting an endpoint by hand" |
| Rename | `Renamed distance #2 in dino to "paw width"` | Enter commits, Escape cancels, as Track View's *Rename* |
| Set the stated distance | `Set distance #2 in dino to 4.2 cm` | Number and unit in one field, parsed as the CLI's YAML parses a distance (`12in`, `30.48cm`, or a bare number in the current unit); refuses a value that is not finite and positive with the format's sentence |
| Delete | `Deleted distance #2 in dino` | Renumbers the rows after it; the selection follows (§ "What the later steps add to what is built") |

| Add a distance measurement | `Added distance #4 in dino` | § "Creating a measurement" |

Dragging an endpoint in Image Detail is a later draft's.

## Creating a measurement

A measurement is made in two gestures, one per endpoint, each a right-click
that names a pixel, then a short prompt for what was measured.

**Starting it.** Two menus offer `New Measurement from Here`:

- **Image Detail's context menu**, at the clicked pixel of the image shown.
  Any pixel inside the image will do; the endpoint does not have to resolve,
  since the pixel is the stored measurement.
- **The 3D view's point menu.** A right-click on a point selects it (G3) and
  offers the entry, which starts the measurement at the point's reference
  pixel.

Either puts the new endpoint A in a **pending measurement**:

```rust
// on AppState: view state, not part of any version
pub pending_measurement: Option<PendingMeasurement>,

pub struct PendingMeasurement {
    pub recon: ReconId,
    pub a: (u32, [f64; 2]), // image and pixel of endpoint A
}
```

A pending measurement is not a version and is not saved. Measurement View shows
it in place of the selected measurement: endpoint A's block, as for any
measurement, and in place of endpoint B a line saying how to finish
("Right-click a point in the 3D view, or a pixel in Image Detail, and choose
End Measurement Here. Escape cancels."). Image Detail and the 3D view draw
endpoint A's mark. Starting another one replaces it, and Escape, closing the
node, or *Cancel* in the panel discards it.

**Finishing it.** While a measurement is pending, the same two menus offer
`End Measurement Here` instead, with the same gates, on the pending
measurement's reconstruction only (a point or an image of another node shows
the entry greyed, its hover text saying the measurement was started in the
other node). Choosing it opens a prompt for:

- the stated distance, a number and a unit in one field, parsed as the *Set
  the stated distance* edit parses it, with the reconstruction's
  `world_space_unit` as the unit of a bare number, or `m` where it has none;
- the label, which may be empty.

*Add* is enabled once the distance parses to a finite, positive length, and
adds the measurement as one version (`Added distance #4 in dino`), at the end
of the distance rows, and selects it. *Cancel* leaves the measurement pending,
so a mistyped prompt does not lose the two endpoints.

### A point's reference pixel

A point in the 3D view is not a pixel, and a measurement's endpoint must be
one. The pixel is taken from the point's **reference observation**
(`tracks/reference_observations`): the image of that observation, and its
keypoint. The keypoint is where the reference observation's square is centred,
so the endpoint resolves to texel `(0, 0)` of the point's own patch, which is
the point itself, through the observation its bitmap was rendered from.

The entry is greyed, with hover text saying why, where the point has no
reference pixel:

- it has no patch frame (a `sift_files` reconstruction, or a point the patch
  pass left without one): "This point has no patch, so no pixel stands for it";
- it has no reference observation (`-1`): "This point has no reference
  observation to take a pixel from";
- it is a point at infinity: "This point is at infinity, so no distance can be
  measured to it".

The entry does not check that the pixel resolves back to the same point: a
neighbouring square can hold the same pixel and win the ranking. The endpoint
block then shows which point it resolved to, as for any endpoint.

## Drawing measurements

**Which measurements are drawn.** The selected measurement, when its
reconstruction is visible. A **Show all measurements** option, off by default,
draws every measurement of every visible reconstruction as well. A
reconstruction with many measurements stays readable by default, and the
person who wants to see them all together asks for it. The option is in the
3D view's HUD and in Image Detail's display controls, one for each view, as
their other display options are. With it on, the selected measurement is
drawn wider and brighter than the others.

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

Hovering a mark or a line in either view shows the measurement's row text. A
click on a line selects the measurement; a click on a mark selects the
measurement and that endpoint ([gui/measurement-view.md](../gui/measurement-view.md)
§ "Endpoint gestures").

**A selected measurement off screen.** When no part of the selected
measurement's marks and line is inside the view, the view draws an indicator
at its edge: an arrow on the border pointing towards the measurement, with
its kind and index (`Distance #2`) beside it.

- In the 3D view the arrow points towards the projection of the midpoint of
  the two endpoints, or of the one finite endpoint. For a point behind the
  camera, the direction is the projection's reflected through the view's
  centre, so the arrow points the way to turn. A click on the arrow frames
  the measurement: the view keeps its orientation and moves so that both
  endpoints are inside it, as `Z` frames the whole scene.
- In Image Detail, when an endpoint is in the image shown but outside the part
  the panel shows, the arrow points to that pixel, and a click centres the
  panel on it at the current zoom.

The indicator follows the selection only. With *Show all measurements* on,
the others are clipped as any scene geometry is.

## MCP

The edits are reachable over MCP, in the shape of the existing tools
([mcp-server.md](../gui/mcp-server.md)), beside the reads and the selection
tool that are built:

- `get_measurement_endpoint_views(reconstruction_label, measurement_index,
  endpoint)`: the offers of § "Switching an endpoint's image".
- `add_distance_measurement(reconstruction_label, a, b, distance, label)`, each
  endpoint an image and a pixel, and `get_point_reference_pixel` with a
  reconstruction and a point, so an agent can start from a point as the 3D
  view's menu does. The pending measurement is a GUI state with no tool of its
  own.
- `set_measurement_endpoint(reconstruction_label, measurement_index, endpoint,
  camera_image, pixel)`, `rename_measurement`, `set_measurement_distance`,
  `delete_measurement`: the edits, each refused in the panel's words.

`set_measurement_endpoint` takes any pixel inside the image, not only an
offered one, as the panel's fields do. No tool sets an endpoint's 3D
position.

## Real measurements

`dino_dog_toy`'s four tape-measured distances, entered by hand once the panel
can place measurements, are the first real ones. They are not checked in yet.
When they are, the file keeps its name: the `_vNN` suffix versions the
reconstruction's cameras, poses, points, tracks and patch frames, and adding
measurements changes none of those.

## Steps

Each step is one PR through `implement-change`, with its MCP tools, its
`ui_basic` coverage and screenshots.

1. **Read-only.** Built; see [gui/measurement-view.md](../gui/measurement-view.md).
2. **Switching an endpoint's image.** `endpoint_views` in core with its
   binding, the share-preserving measurement replacement if it is missing, the
   switch popup, `get_measurement_endpoint_views` and
   `set_measurement_endpoint`.
3. **Setting an endpoint by hand.** The endpoint block's image and pixel
   fields, `Move Endpoint A Here` and `Move Endpoint B Here` in Image Detail's
   menu and the 3D view's point menu, the reference pixel, and
   `set_measurement_endpoint` taking any pixel.
4. **The other edits.** Rename, set the stated distance, delete, and their
   tools.
5. **Creating a measurement.** The pending measurement, `New Measurement from
   Here` and `End Measurement Here` in both menus, the prompt,
   `add_distance_measurement` and `get_point_reference_pixel`.
6. **Drawing.** The selected measurement's marks and line in the 3D view and
   Image Detail, their hover and click, the off-screen indicator and its
   click, and the *Show all measurements* option in each view, with
   `get_viewer_3d_display`, `set_viewer_3d_display` and the Image Detail
   display tools carrying it as `show_all_measurements`.

## Testing

- The selection: deleting the selected measurement clears the selection, and
  deleting an earlier row moves it.
- Switching: the offers of an endpoint on a known texel land at the pixels the
  other observations' squares put that texel at; a switch keeps the resolved
  point and texel; the "resolves elsewhere" mark appears where another square
  holds the offered pixel; the edit's version shares the base's point columns
  (pointer equality).
- Each edit's label, undo, and refusal.
- Drawing: with the option off only the selected measurement is drawn, and
  none with nothing selected; with it on, every measurement of a visible
  reconstruction; a hidden reconstruction draws none either way.
- The off-screen indicator: absent while any part of the measurement is in
  view; pointing the right way for a midpoint to each side of the view and
  behind the camera; a click framing the measurement so both endpoints project
  inside the view.
- A click on a mark selects the measurement and that endpoint.
- Creating: starting from a pixel and from a point; the entry greyed, with its
  sentence, on a point with no patch, no reference observation, or at
  infinity; a point's reference pixel resolving to texel `(0, 0)` of its
  patch where no other square holds it; finishing in another node greyed;
  *Cancel* in the prompt keeping the pending measurement; *Add* pushing one
  version and selecting the new row; Escape discarding a pending measurement
  without a version.
- `ui_basic`: a measurement made with the two menus and the prompt.
