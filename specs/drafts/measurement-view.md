# Measurement View

**Status:** Draft

Decided by the maintainer:

- A loaded reconstruction's measurements are listed in the Scene tree, under
  the reconstruction, one row per measurement.
- A new panel, **Measurement View**, shows one measurement, in the way Track
  View shows one track. Each kind of measurement has its own body in the
  panel. Distance measurements are the only kind today
  ([formats/sfmr-file-format.md](../formats/sfmr-file-format.md) § 10
  "Measurements").
- For a distance measurement the panel shows both endpoints, each with a crop
  of its photograph whose hover shows a larger view. When the file declares a
  `world_space_unit`, it shows the measurement's error. When it does not, it
  shows the length the reconstruction gives the measurement in scene units.
  When an endpoint has no 3D position, it says why instead.
- The panel edits a measurement. The first edit is switching an endpoint to
  another image: the patches whose square in the endpoint's image holds the
  endpoint's pixel are found as the resolver finds them, and every other
  observation of those tracks is offered, with a crop of where the endpoint
  falls in each, so a person can pick the image that shows the endpoint best.
- The panel also edits an endpoint's image and pixel directly. An endpoint's
  3D position is never edited: it is always derived from the image and pixel.
- The Scene tree lists measurements directly under the reconstruction's
  *Measurements* group, with no subgroup per kind while there is one kind. A
  row with no unit shows the reconstructed length in scene units.
- The 3D view draws each endpoint and a line between the two. Image Detail
  draws the same where both endpoints are in the image it shows, and the one
  endpoint that is in it otherwise.
- Only the selected measurement is drawn by default. A display option, off by
  default, draws every measurement. A selected measurement that is off screen
  gets an indicator at the edge of the view pointing to it.
- The checked-in reconstructions have no measurements, so the panel is built
  and tested on copies that are given artificial measurements
  (§ "Artificial measurements"). Real measurements are not checked in yet.

Nothing in this draft is still open.

A measurement in an `.sfmr` file is a real-world quantity tied to places in the
photographs. A distance measurement is a length, such as one taken with a
tape measure, between two **endpoints**, each a pixel in one image. SfM
Explorer shows these measurements so a person can see whether the
reconstruction agrees with them. If the reconstruction has a physical unit,
the question is how far off its distance is. If it has none, the question is
what scale the measurement would give it. Measurement View shows one
measurement: where each endpoint sits in its photograph, which point of the
reconstruction it lands on and how that was decided, and what the
reconstruction makes of the length between them. It also lets a person
improve an endpoint by moving it to another photograph of the same surface.
The 3D position of an endpoint comes from
[core/reconstruction/measurement-endpoint.md](../core/reconstruction/measurement-endpoint.md).

Related specs: [gui/track-view.md](../gui/track-view.md) (the panel this one
is modelled on, and the crops and hover views it reuses),
[gui/scene-graph.md](../gui/scene-graph.md) (the tree the rows go in),
[gui/document-model.md](../gui/document-model.md) and
[gui/edit-history.md](../gui/edit-history.md) (the versions an edit pushes),
[gui/panel-layout.md](../gui/panel-layout.md) (the new tab),
[gui/mcp-server.md](../gui/mcp-server.md) (the tools).

## The selected measurement

A measurement is selected the way a point is: one selected measurement in the
whole viewer, named by a typed reference.

```rust
pub enum MeasurementKind { Distance }

pub struct MeasurementRef {
    pub recon: ReconId,
    pub kind: MeasurementKind,
    pub index: u32, // row of that kind's section
}

pub enum EndpointSide { A, B } // endpoint 0 and 1 of the format

// on AppState, beside selected_point
pub selected_measurement: Option<MeasurementRef>,
/// The selected endpoint of the selected measurement, when one is.
pub selected_endpoint: Option<EndpointSide>,
```

The reference carries the kind because each kind's rows are numbered within
their own subsection of the file, and a later kind would otherwise reuse the
same indexes.

**Selecting a measurement selects nothing else.** It does not select either
endpoint's image or point: a person reading a measurement usually wants the
3D view and Image Detail to stay where they are. Selecting a measurement clears
`selected_endpoint`.

**Selecting an endpoint is selecting an observation.** An endpoint is a pixel
in one photograph, as an observation is, so the gestures on it are Track
View's gestures on an observation row, with the same effects in the other
panels (§ "Endpoint gestures"). It selects the endpoint's image, which the 3D
viewer, the Image Browser and Image Detail follow, and leaves the selected
point alone, as a Track View row click leaves the track's point alone.

**The selection follows an edit.** Measurements are never renumbered by a
point edit. An edit to a measurement keeps its index. Deleting a measurement
renumbers the rows after it, and the selection moves with them, or is cleared
if it was the one deleted. A bulk edit that drops images can drop
measurements (`subset_by_image_indices` drops a measurement whose endpoint
image is gone); the selection is cleared when its measurement is dropped, and
kept on the same row otherwise. Undo and redo keep it where the version
arrived at has that row, and clear it where it does not.

## The Scene tree's Measurements group

A fixed group node under each reconstruction that has measurements, after the
Points group and before the Bench groups:

```
├── Reconstruction "dino"
│   ├── Points (23,412)
│   ├── Measurements (4)
│   │   ├── Distance #0  ear to ear      12.0 cm  → 11.94 cm  −0.5%
│   │   ├── Distance #1  nose to tail    31.5 cm  → 31.71 cm  +0.7%
│   │   ├── Distance #2  paw width        4.2 cm  ⚠ endpoint B: no patch holds the pixel
│   │   └── Distance #3                   8.0 cm  → 7.96 cm   −0.5%
│   ├── Bench Points (2)
```

- The group's count is the number of measurements of every kind, and it is
  absent when there are none.
- Each row names its kind and index, then its label (blank when the label is
  empty), its stated distance, and the reconstruction's reading of it: with a
  unit, the reconstructed distance in the measurement's own unit and the
  signed error as a percentage; without one, the reconstructed distance in
  scene units (`0.317 su`). An endpoint with no position puts a warning mark
  and the refusal's short form in place of the reading; an endpoint at
  infinity shows `∞`.
- A click selects the measurement and does nothing else, as a click on a
  Bench row does. A double-click selects it, raises Measurement View and
  frames the measurement in the 3D view, as a double-click on a Bench row
  raises Track View and moves the 3D view onto the item's point: the view keeps
  its orientation and moves so both finite endpoints are inside it (the
  framing the off-screen indicator's click makes, § "Drawing measurements").
  A measurement with no finite endpoint leaves the view where it is. The
  selected row is highlighted as the selected point's row is. A right-click
  opens the row's menu (*Delete*) and selects nothing.
- The rows sit directly under the group. A subgroup per kind is for when a
  second kind exists, and is that kind's draft's to add.

The readings come from one call to `resolve_distance_measurements` per
version, cached on the node by the version's serial, since every row and the
panel read the same results and a version never changes. It is a pass over the
endpoints' images' observations, well under a second, so it runs on the GUI
thread when a version first needs it.

## Placement

A dock tab, `Tab::MeasurementView`, titled **Measurement View** and named
`measurement_view` in the layout file and on the wire. Its home is the
top-right group with Track View, after it: `[TrackView, MeasurementView,
IntrinsicsDetail]`. A measurement is read beside the picture, as a track is.
The layout file's list of panel names gains `measurement_view`; the name is a
value a leaf holds, so `LAYOUT_VERSION` does not change
([panel-layout.md](../gui/panel-layout.md) § "Validation").

## What the panel shows

**No reconstruction selected** or **no measurement selected**: one line saying
so. When the selected reconstruction has measurements, the second case also
lists them as the Scene tree does, each a link that selects it, so the panel
is usable without the tree.

**A measurement selected**: a header, then the body for its kind.

### The header

- The kind and index (`Distance #2`) and the reconstruction's label.
- The label, editable in place (§ "Editing").
- For a distance: the stated distance and its unit, editable in place.
- A line comparing this measurement with the others of its kind in the same
  reconstruction, when there are at least two with a reading: the median scale
  they imply and this measurement's deviation from it
  (`Median over 4: 1 su = 37.8 cm; this one +0.4%`). It is the check that
  catches one bad measurement among good ones, whether or not the file has a
  unit.

## The distance body

Two endpoint blocks side by side, **A** and **B** (endpoint 0 and 1 of the
format), then the result block.

### An endpoint

Each block shows:

- **The photograph's name** and the endpoint's pixel, `(412.3, 230.5)`.
- **The crop.** The photograph around the endpoint. Where the endpoint
  resolves, the crop is cut around the chosen candidate's square in that
  image, as Track View's *Crop* cell is cut around a row's square
  (`patch_pictures::crop::patch_outline` and `region`), with the square's
  outline over it and a cross at the endpoint's pixel. Where it does not, the
  crop is a fixed square of the photograph centred on the pixel
  (`UNRESOLVED_CROP_PX`, 64 photograph pixels), with the cross alone.
- **The crop's hover view.** The same crop in context, three times its width
  and height at 288 points (`marks::CONTEXT_FACTOR`, `CONTEXT_HOVER_SIDE`), as
  Track View's crop hover is, with the outline and the cross. The tooltip opens
  with the photograph's full name, and bullets under the picture give the
  pixel, the crop's size in photograph pixels and the square's two axes in
  photograph pixels.
- **The patch.** The chosen point's stored patch bitmap, or its display
  bitmap, with a cross at the endpoint's texel `(s, t)`, so the person sees
  which place on the surface the endpoint names. Absent where the endpoint does
  not resolve or the point has no bitmap.
- **The resolution.** One of:
  - `Alone`: the point's ID with Track View's *Copy Point ID* and *Go to Point*
    buttons, the texel `(s, t)`, and the 3D position in the file's unit or
    scene units. "One square holds the pixel."
  - `Ranked`: the same, and "Chosen from N squares", with a disclosure that
    lists every candidate: its point, standing, cost and costliest term, in the
    order the resolver ranked them.
  - At infinity: the point's ID and its direction, and "This endpoint lies on a
    point at infinity, so no distance can be measured to it."
  - A refusal: its sentence, which names the remedy.
- **Switch image…**, the edit in § "Switching an endpoint's image".
- **The image and pixel, editable**, the edit in § "Setting an endpoint by
  hand". The 3D position beside them is read-only.

The selected endpoint's block is drawn as Track View draws its selected row,
and its mark is drawn larger in the 3D view and Image Detail, as the 3D view
draws the selected observation's circle larger.

### Endpoint gestures

These are Track View's row gestures ([track-view.md](../gui/track-view.md) §
"Row gestures"), applied to an endpoint, and they follow rules G1, G2 (a
pixel of an image), G3, G5 and G6 of the proposed interaction rules
(`specs/drafts/gui-interaction-rules.md`). The block's own controls (the image
drop-down, the pixel fields, the buttons) keep their clicks; everywhere else in
the block, the name, the crop and the patch among it, is one target.

- **Click**: select the endpoint and its image, and **reveal** the endpoint's
  pixel: it rides along as `reveal_feature` does, and Image Detail pans, never
  zooms, to centre it when its current view is not showing it
  ([multi-panel-image-browser.md](../gui/multi-panel-image-browser.md) §
  "Revealing a feature named by another panel"). The 3D view does not move.
- **Double-click**: enter camera view for the endpoint's image, and turn the
  view until the endpoint's pixel is inside the middle 1/2 of the viewport on
  both axes (`Viewer3D::look_through_toward_feature`), in one animated
  transition, as a double-click on a Track View row does.
- **Hover**: set the cross-panel hover to the endpoint's image, which
  brightens its frustum and outlines its thumbnail.
- **Right-click**: a menu with *Switch Image…*, and selects nothing.

**From outside the panel**, a click on an endpoint's mark in Image Detail or
the 3D view selects the measurement and that endpoint, as a click on a bench
mark selects that observation's row, and a double-click on it does what a
double-click on its block does. The marks are drawn only for the measurements
§ "Drawing measurements" draws.

### The result

Exactly one of:

- **Both endpoints finite, the file has a unit.** The reconstructed distance
  in the measurement's own unit (the file's `world_space_unit` converted
  through metres), the error `reconstructed − stated` in that unit and as a
  percentage, and the scale the measurement implies (`stated / reconstructed`,
  as a factor). The error's sign is stated in words in the hover text
  ("the reconstruction is 0.6 mm longer than the tape").
- **Both endpoints finite, no unit.** The reconstructed distance in scene
  units, and the scale the measurement implies (`1 su = 3.78 cm`), which is
  what `sfm xform --scale-by-measurements` would apply from this measurement
  alone.
- **An endpoint at infinity.** "No distance: endpoint B lies on a point at
  infinity."
- **An endpoint refused.** "No distance: endpoint A has no 3D position", with
  the refusal's sentence above it in A's block.

Every distance is read from the node's value, in the file's own coordinates.
The node's display transform is view state and never scales a reading, so a
node drawn under a scaled alignment still reports the file's own error.

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
| Delete | `Deleted distance #2 in dino` | Renumbers the rows after it; the selection follows (§ "The selected measurement") |

Placing a new measurement, and dragging an endpoint in Image Detail, are a
later draft's.

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
measurement and that endpoint (§ "Endpoint gestures").

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

The panel's content and its edits are reachable over MCP, in the shape of the
existing tools ([mcp-server.md](../gui/mcp-server.md)):

- `get_scene` gives each reconstruction a `measurement_counts` block, by kind.
- `list_measurements(recon)`: every measurement of every kind, each with the
  reading the Scene tree row shows.
- `select_measurement(recon, kind, index, endpoint?)` and the selection
  block's `selected_measurement` and `selected_endpoint`. Naming an endpoint
  does what a click on its block does, including the reveal.
- `get_measurement(recon, kind, index)`: for a distance, both endpoints'
  images, pixels and resolutions (with every candidate), and the result block.
- `get_measurement_endpoint_views(recon, index, endpoint)`: the offers of
  § "Switching an endpoint's image".
- `set_measurement_endpoint(recon, index, endpoint, image, pixel)`,
  `rename_measurement`, `set_measurement_distance`, `delete_measurement`: the
  edits, each refused in the panel's words.
- `screenshot` accepts `measurement_view` as a panel name.

`set_measurement_endpoint` takes any pixel inside the image, not only an
offered one, as the panel's fields do. No tool sets an endpoint's 3D
position.

## Artificial measurements

No checked-in reconstruction has measurements, and the panel cannot be judged
or tested without them. They are made, never checked in, in two ways.

**In Rust tests** (`sfm-explorer` lib tests, headless): a small synthetic
reconstruction built in the test, as the resolver's tests build theirs, given
measurements whose endpoints are chosen to exercise each case below. These
test the selection, the Scene tree rows, the result block's numbers, the
switch offers and every edit.

**On copies of the ground truths**, for the `ui_basic` tests, for screenshots
and for a person or an agent trying the panel: a script,
`scripts/measurements/add_artificial_measurements.py`, that reads a
reconstruction, writes measurements into it through `sfmtool.fileio`, and
writes the result to a path it is given, refusing a path under `test-data/`.
Its endpoints are observations' keypoints, so they resolve to a texel near the
centre of a known point's patch, and its stated distances are the
reconstructed distances times a known factor, so the expected error is known.
It writes a set that covers:

| Case | How it is made |
|---|---|
| Both endpoints in one image | two observations' keypoints in the same image |
| Endpoints in different images | keypoints of observations in two images |
| A known error | stated distance = reconstructed × 1.02 |
| One bad measurement among good ones | one stated distance off by 20%, so the header's median line flags it |
| An endpoint that is refused | a pixel well away from every square in its image (sky, or a blank wall) |
| An endpoint held by several squares | a pixel between two observations whose squares overlap |
| No unit | the same set written to a second copy with `world_space_unit` cleared |

The ground truths are in metres (seoul_bull and kerry_park from GPS fixes,
dino_dog_toy from a tape measure), so the copies with a unit report errors in
real units. A point at infinity is not in every ground truth; the at-infinity
case is covered by the Rust tests.

`dino_dog_toy`'s four tape-measured distances, entered by hand once the panel
can place measurements, are the first real ones. They are not checked in yet.
When they are, the file keeps its name: the `_vNN` suffix versions the
reconstruction's cameras, poses, points, tracks and patch frames, and adding
measurements changes none of those.

## Steps

Each step is one PR through `implement-change`, with its MCP tools, its
`ui_basic` coverage and screenshots.

1. **Read-only.** `MeasurementRef` and the selection, the per-version
   resolution cache, the Scene tree group with its row gestures, Measurement
   View with the header, the distance body and the endpoint gestures,
   `get_scene`'s counts, `list_measurements`,
   `select_measurement`, `get_measurement`, and the artificial-measurements
   script.
2. **Switching an endpoint's image.** `endpoint_views` in core with its
   binding, the share-preserving measurement replacement if it is missing, the
   switch popup, `get_measurement_endpoint_views` and
   `set_measurement_endpoint`.
3. **Setting an endpoint by hand.** The endpoint block's image and pixel
   fields, Image Detail's `Move Endpoint A Here` and `Move Endpoint B Here`,
   and `set_measurement_endpoint` taking any pixel.
4. **The other edits.** Rename, set the stated distance, delete, and their
   tools.
5. **Drawing.** The selected measurement's marks and line in the 3D view and
   Image Detail, their hover and click, the off-screen indicator and its
   click, and the *Show all measurements* option in each view, with
   `get_viewer_3d_display`, `set_viewer_3d_display` and the Image Detail
   display tools carrying it as `show_all_measurements`.

## Testing

- The selection: a click on a row selects it; deleting it clears the
  selection; deleting an earlier row moves it; undo and redo keep or clear it
  as § "The selected measurement" says.
- The Scene tree: rows with and without a unit, a refused endpoint, an
  endpoint at infinity; the group absent with no measurements.
- The result block: the error and implied scale against hand-computed values
  for a unit and for scene units, a unit conversion (file in `m`, measurement
  in `in`), and a node under a scaled display transform reading the same as
  the node at the identity.
- The median line: absent with one reading, flagging the bad measurement among
  good ones.
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
- Endpoint gestures: a click selects the endpoint and its image, leaves the
  selected point and the 3D view alone, and pans Image Detail only when the
  pixel is out of its view; a double-click enters camera view with the pixel
  in the middle half; a right-click selects nothing; a click on a mark selects
  the measurement and that endpoint; selecting a measurement clears the
  endpoint.
- The Scene tree row: a click selects and moves nothing; a double-click frames
  the measurement's finite endpoints and leaves the view alone with none.
- `ui_basic`: the panel opens from the Panels menu and from a double-click on a
  row, on an artificial-measurements copy.
