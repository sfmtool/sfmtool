# Measurement View

A measurement in an `.sfmr` file is a real-world quantity tied to places in the
photographs. A distance measurement is a length, such as one taken with a tape
measure, between two **endpoints**, each a pixel in one image. SfM Explorer
shows these measurements so a person can see whether the reconstruction agrees
with them. If the reconstruction has a physical unit, the question is how far
off its distance is. If it has none, the question is what scale the
measurement would give it. The Scene tree lists a reconstruction's
measurements, one row each, and the **Measurement View** panel shows one of
them: where each endpoint sits in its photograph, which point of the
reconstruction it lands on and how that was decided, and what the
reconstruction makes of the length between them. The 3D position of an
endpoint comes from
[core/reconstruction/measurement-endpoint.md](../core/reconstruction/measurement-endpoint.md).

A measurement is edited on the bench, as a track is ([bench.md](bench.md)): a
double-click on its row puts it on the bench as an item and focuses it, the
panel's **Edit** box shows that item with its readings taken live from it, and
*Commit* writes it back into the file in one step. The edited mode's header
sets the item's label and its stated distance, and each endpoint block sets
its endpoint's image and pixel in fields or switches it to another photograph
that shows the same place. Image Detail's context menu and the 3D view's point
menu set an endpoint too, to a pixel of the photograph shown or to the pixel
that shows the spot right-clicked on a point's patch. Each is a bench step. A
new measurement starts from the same two menus, with its first endpoint at the
pixel or point they were opened on, and is finished with the same steps. Deleting a stored measurement is a direct
edit from its row's menu. The 3D view and Image Detail draw the selected
measurement and the item being edited, every measurement where a display option
asks for it, and an arrow to the selected one when it is off screen; the item's
marks in Image Detail are handles that set an endpoint's pixel
(§ "Drawing measurements").

Related specs: [track-view.md](track-view.md) (the panel this one is modelled
on, and the crops it reuses), [bench.md](bench.md) § "Measurement items" (the
item, its origin and its commit, in the viewer) and
[core/bench/editable-measurement.md](../core/bench/editable-measurement.md)
(in core), [scene-graph.md](scene-graph.md) (the tree the
rows go in), [panel-layout.md](panel-layout.md) (the tab),
[mcp-server.md](mcp-server.md) (the tools), and
[formats/sfmr-file-format.md](../formats/sfmr-file-format.md) § 10
"Measurements" (what the file stores).

## Interface

The code is in
[measurements.rs](../../crates/sfm-explorer/src/measurements.rs) (the
reference, the readings and their text, and an item's live readings),
[measurement_view.rs](../../crates/sfm-explorer/src/measurement_view.rs) (the
panel),
[bench/measurement.rs](../../crates/sfm-explorer/src/bench/measurement.rs)
(putting a measurement on the bench, the commit and the delete),
[scene_graph/measurements.rs](../../crates/sfm-explorer/src/scene_graph/measurements.rs)
(the Scene tree group),
[mcp/measurements.rs](../../crates/sfm-explorer/src/mcp/measurements.rs) (the
tools), and, for the drawing,
[measurement_marks.rs](../../crates/sfm-explorer/src/measurement_marks.rs)
(which measurements are drawn, the marks and lines on screen, their hit test,
the press and the indicator, shared by both views),
[viewer_3d/measurements.rs](../../crates/sfm-explorer/src/viewer_3d/measurements.rs)
(where the 3D view puts them, and the instances its scene pass draws them with) and
[image_detail/measurements.rs](../../crates/sfm-explorer/src/image_detail/measurements.rs)
(where Image Detail puts them, and the handle drag).

```rust
// Core's, re-exported: `title()` is `Distance`, `name()` is `distance`.
pub(crate) use sfmtool_core::bench::MeasurementKind;

/// One measurement of one loaded reconstruction.
pub(crate) struct MeasurementRef {
    pub(crate) recon: ReconId,
    pub(crate) kind: MeasurementKind,
    pub(crate) index: u32, // row of that kind's section
}

pub(crate) enum EndpointSide { A, B } // endpoint 0 and 1 of the format

// on AppState, beside selected_point
pub(crate) selected_measurement: Option<MeasurementRef>,
pub(crate) selected_endpoint: Option<EndpointSide>,

impl AppState {
    pub(crate) fn select_measurement(&mut self, measurement: MeasurementRef);
    pub(crate) fn select_endpoint(&mut self, measurement: MeasurementRef, side: EndpointSide);
    pub(crate) fn prune_measurement_selection(&mut self);

    // The bench (bench.md § "Measurement items").
    pub(crate) fn put_measurement_on_bench(&mut self, measurement: MeasurementRef)
        -> Result<PutMeasurement, String>;
    /// A new item with endpoint A at `pixel` of `image`, focused
    /// (§ "Creating a measurement"); its label comes back.
    pub(crate) fn start_measurement(&mut self, image: ImageRef, pixel: [f64; 2])
        -> Result<String, String>;
    /// Image Detail's New Measurement from Here: the start, then Measurement
    /// View raised beside `from`; a refusal one failed row.
    pub(crate) fn start_measurement_here(&mut self, image: ImageRef, pixel: [f64; 2],
                                         from: Tab);
    /// The 3D view's: the same, at the pixel that shows the spot right-clicked.
    pub(crate) fn start_measurement_at_point(&mut self, spot: PointSpot, from: Tab);
    /// Why the 3D view's entry is greyed, by node.
    pub(crate) fn measurement_start(&self) -> MeasurementStart;
    pub(crate) fn set_measurement_editing(&mut self, id: ReconId, on: bool)
        -> Result<(), String>;
    pub(crate) fn commit_bench_measurement(&mut self, id: ReconId, label: &str)
        -> Result<MeasurementCommitted, String>;
    /// The header's fields (§ "The label and the stated distance").
    pub(crate) fn set_bench_measurement_label(&mut self, id: ReconId, label: &str,
                                              to: &str) -> Result<bool, String>;
    pub(crate) fn set_bench_measurement_distance(&mut self, id: ReconId, label: &str,
                                                 text: &str)
        -> Result<StatedDistanceSet, String>;
    /// A row of the Switch image… popup (`EndpointMove::Switch`), and the
    /// fields, the menus and the wire's set_bench_measurement_endpoint
    /// (`EndpointMove::Set`) (§ "Switching an endpoint's image", § "Setting an
    /// endpoint by hand").
    pub(crate) fn set_bench_measurement_endpoint(&mut self, id: ReconId, label: &str,
                                                 side: EndpointSide, image: u32,
                                                 pixel: [f64; 2], how: EndpointMove)
        -> Result<bool, String>;
    /// The fields and Image Detail's Set Endpoint … Here: the step with
    /// `EndpointMove::Set`, a refusal one failed row.
    pub(crate) fn set_endpoint(&mut self, set: &EndpointSet);
    /// The 3D view's Set Endpoint … Here: the same, at the pixel that shows the
    /// spot right-clicked.
    pub(crate) fn set_endpoint_at_point(&mut self, spot: PointSpot, side: EndpointSide);
    /// The focused measurement item as the two menus see it, or `None`.
    pub(crate) fn endpoint_target(&self) -> Option<EndpointTarget>;
    pub(crate) fn delete_measurement(&mut self, measurement: MeasurementRef)
        -> Result<(), String>;
    /// A click on an endpoint block of the focused item.
    pub(crate) fn select_item_endpoint(&mut self, id: ReconId, side: EndpointSide);
}

/// The readings of the version a node shows, resolved once per version.
pub(crate) fn readings(node: &SceneNode) -> Arc<MeasurementReadings>;
pub(crate) fn distance_result(
    row: &DistanceRow,
    resolved: &ResolvedDistanceMeasurement,
    world_space_unit: Option<&str>,
) -> DistanceResult;
/// A version's stored distance rows, read from the value's measurements.
pub(crate) fn distance_rows(measurements: &MeasurementsData) -> Vec<DistanceRow>;
/// A measurement item's live readings against a version.
pub(crate) fn item_reading(edited: &EditedReconstruction,
                           item: &EditableMeasurement) -> ItemReading;
/// A point's reference pixel (§ "The spot under the pointer"), or the
/// sentence saying why it has none.
pub(crate) fn reference_pixel(edited: &EditedReconstruction, point: u32)
    -> Result<ReferencePixel, String>;
/// Where on a point the 3D view's point menu was opened: the point, and the
/// texel of its drawn square under the pointer, or `None`.
pub(crate) struct PointSpot { pub(crate) point: PointRef, pub(crate) texel: Option<[f64; 2]> }
/// The pixel of the reference image that shows `texel`, else the reference
/// pixel; refused as `reference_pixel` is.
pub(crate) fn spot_pixel(edited: &EditedReconstruction, point: u32,
                         texel: Option<[f64; 2]>) -> Result<ReferencePixel, String>;
/// The texel of a patch frame's square, drawn at `surfel_scale`, that a ray
/// meets, in the reconstruction's own coordinates; `None` off the square, on
/// its back, or grazing it.
pub(crate) fn texel_under_ray(patch_frame: &OrientedPatch, origin: Point3<f64>,
                              direction: Vector3<f64>, surfel_scale: f64)
    -> Option<[f64; 2]>;

// bench/measurement.rs
/// What a gesture that sets an endpoint outright asks for.
pub(crate) struct EndpointSet {
    pub(crate) node: ReconId,
    pub(crate) item: String, // the item's label on the bench
    pub(crate) side: EndpointSide,
    pub(crate) image: u32,
    pub(crate) pixel: [f64; 2],
}
/// The focused measurement item, as the two menus see it.
pub(crate) struct EndpointTarget { pub(crate) node: ReconId, pub(crate) item: String, .. }
impl EndpointTarget {
    /// Why the entries are greyed on `node`, or `None`.
    pub(crate) fn refusal_on(&self, node: ReconId) -> Option<String>;
    /// The 3D view's entries: the pixel that shows `spot`, or why not.
    pub(crate) fn at_point(&self, scene: &[SceneNode], spot: PointSpot)
        -> Result<ReferencePixel, String>;
}
/// "New Measurement from Here", in both menus.
pub(crate) const NEW_MEASUREMENT_HERE_LABEL: &str;
/// Why a step on each node's bench is refused, as the 3D view's New
/// Measurement from Here sees it.
pub(crate) struct MeasurementStart { .. }
impl MeasurementStart {
    /// The pixel that shows `spot`, or why the entry is greyed.
    pub(crate) fn at_point(&self, scene: &[SceneNode], spot: PointSpot)
        -> Result<ReferencePixel, String>;
}

// measurement_marks.rs
/// A stored row, or the focused item by its label on its node's bench.
pub(crate) enum DrawnId { Stored(MeasurementRef), Item { node: ReconId, label: String } }
/// What one view draws this frame, in draw order: the others, the selected
/// measurement, the focused item.
pub(crate) struct MeasurementLayer { pub(crate) drawn: Vec<DrawnMeasurement> }
impl AppState {
    /// With that view's Show all measurements.
    pub(crate) fn measurement_layer(&self, show_all: bool) -> MeasurementLayer;
    /// Everything that layer is built from.
    pub(crate) fn measurement_layer_key(&self, show_all: bool) -> MeasurementLayerKey;
}
/// A view's layer and the key it was built on.
pub(crate) struct MeasurementLayerCache { .. }
impl MeasurementLayerCache {
    /// Build the layer again only when the key differs from the last one.
    pub(crate) fn update(&mut self, state: &AppState, show_all: bool);
    pub(crate) fn layer(&self) -> &Arc<MeasurementLayer>;
}
/// What a click, a double-click or a handle drag asks of the app.
pub(crate) enum MeasurementGesture {
    Select(DrawnId),
    SelectEndpoint(DrawnId, EndpointSide),
    LookThrough(DrawnId, EndpointSide),
    SetEndpoint(EndpointSet),
}
```

Each view holds a `MeasurementLayerCache` (`Viewer3D::measurements`,
`ImageDetail::measurements`), and the dock brings it up to date every frame
with that view's *Show all measurements*, as it hands the 3D view its bench
figure, because the views are handed `&mut` into the state further down the
same call. The cache builds the layer again only when the layer's key
(`MeasurementLayerKey`) differs from the one it was last built on, so a frame
in which nothing it is drawn from has changed resolves no endpoint and formats
no hover text. The key holds *Show all measurements*, the selected measurement
and endpoint, the focused measurement item (its node, its label on the bench,
its `Arc`, which a step on it replaces, and whether its bench takes edits now,
which says whether its marks are handles), and each visible node and the
focused item's node: its visibility, its version's document serial (which the
stored rows, their readings, the item's reading and the reconstruction's
`world_space_unit` follow), its display transform and its tint. The label is
there because a rename keeps the item's `Arc`; the bench's editability because
a task starting or finishing on the node changes it with nothing else. The
camera is not in the key: the layer holds world positions and pixels, and
each view places them on screen every frame. The layer is view-independent
but for *Show all measurements*, so each view keeps its own cache rather than
the two sharing one that would build again whenever their options differ.
A view reports a gesture back and the dock applies
it (`TabContext::apply_measurement_gesture`), through the same calls an
endpoint block's click and double-click make, so a mark and a block cannot
disagree (G8):

```rust
// A click on endpoint B's mark of distance #2, in either view.
MeasurementGesture::SelectEndpoint(DrawnId::Stored(MeasurementRef::distance(id, 2)),
                                   EndpointSide::B)
// applied as state.select_endpoint(MeasurementRef::distance(id, 2), EndpointSide::B)
```

The reference pixel and the spot's pixel are in
[measurements.rs](../../crates/sfm-explorer/src/measurements.rs) rather than
beside the 3D view's menu because three things read the reference pixel: the
menu's *Set Endpoint … Here* entries, its *New Measurement from Here*, and the
wire's `get_point_reference_pixel`. The texel-to-pixel mapping itself is
core's `texel_pixel`, the one `endpoint_views` offers its views through.

The reference carries the kind because each kind's rows are numbered within
their own subsection of the file, and a later kind would otherwise reuse the
same indexes. A Scene tree row, a link in the panel's empty state and the
`select_measurement` tool all select through `AppState::select_measurement`, and
a click on an endpoint block and the tool's `endpoint` argument through
`AppState::select_endpoint`, so the gestures and the wire cannot disagree:

```rust
// A row click.
state.select_measurement(MeasurementRef::distance(id, 2));
// A click on endpoint A's block.
state.select_endpoint(MeasurementRef::distance(id, 2), EndpointSide::A);
```

## The selection

A measurement is selected the way a point is: one selected measurement in the
whole viewer, inside the selected reconstruction.

**Selecting a measurement selects it and its reconstruction, and nothing
more.** Selecting the reconstruction keeps the other selections inside it, as
selecting a reconstruction does. Neither endpoint's image nor its point is
selected, so the 3D view and Image Detail stay where they are.
It clears `selected_endpoint`. A measurement the version on screen does not
hold selects nothing.

**Selecting an endpoint is selecting an observation.** An endpoint is a pixel
in one photograph, as an observation is, so a click on its block does what a
click on a Track View row does: it selects the measurement, then the endpoint's
image with its pixel for Image Detail to reveal (`AppState::reveal_in_image`),
then the endpoint. The 3D viewer, the Image Browser and Image Detail follow the
image, and the selected point is left alone. Selecting another image clears the
endpoint.

**The selection follows the version.** Measurements are never renumbered by a
point edit. Deleting an image drops the measurements with an endpoint in it and
renumbers the rows after them, as `SfmrReconstruction::subset_by_image_indices`
does, and deleting a measurement drops its row and renumbers the rows after
it; the selection moves with its measurement, or is cleared where its
measurement was dropped (`AppState::follow_measurement_rows`). Each such
version keeps that renumbering as its measurement map
([document-model.md](document-model.md)), and **an undo or a redo carries the
selection across it** (`AppState::follow_measurement_step`): an undo of the
delete puts the selection back on the measurement that was selected, at the
row it holds again, and a redo moves it down once more. A row the step being
redone drops clears the selection. When the move ends, and again every frame
before the panels draw, `AppState::prune_measurement_selection` clears a
selection whose row the version on screen does not hold, such as one on a row
an undone commit had appended. A cleared selection stays cleared when a later
step brings the row back, except while a measurement item is focused and its
origin was the selection (or both were none, its row gone) before the move:
the selected measurement is then the item's origin at the cursor, so an undo
or a redo that brings the origin's row back selects it again, with no Action
Log row beside the move's own. A selection the person cleared while the item
stayed focused stays cleared across an undo or a redo.

## The readings

Every row, the panel and the wire read the same readings: one call to core's
`resolve_distance_measurements`, with the default options, per version. The
result is kept on the node (`SceneNode::measurement_readings`) keyed by the
version's document serial, so a bench step, which shares its predecessor's
document half, reads the same value, and a version is never resolved twice in
a row. It is a pass over the endpoints' images' observations, well under a
second, so it runs on the GUI thread when a version first needs it.

Every distance is read in the file's own coordinates. The node's display
transform is view state and never scales a reading, so a node drawn under a
scaled alignment still reports the file's own error.

A distance measurement's result (`DistanceResult`) is exactly one of:

- **Both endpoints finite, the file declares a `world_space_unit`.** The
  reconstructed distance in the measurement's own unit (the file's unit
  converted through metres), the error `reconstructed − stated` in that unit
  and as a percentage of the stated distance, and the scale the measurement
  implies, `stated / reconstructed`.
- **Both endpoints finite, no unit.** The reconstructed distance in scene units
  (`su`), and the length one scene unit stands for in the measurement's unit,
  `stated / reconstructed`, which is what `sfm xform --scale-by-measurements`
  would apply from this measurement alone.
- **An endpoint refused.** The first refused side, `A` before `B`.
- **An endpoint at infinity.** The first side at infinity, where neither is
  refused.

A refusal is reported before a point at infinity because the refusal is the one
a person can act on.

## The Scene tree's Measurements group

A fixed group node under each reconstruction whose version on screen stores
measurements, after the Points group and before the Bench groups
([scene-graph.md](scene-graph.md) § "Tree rows"):

```
├── Reconstruction "dino"
│   ├── Points (23,412)
│   ├── Measurements (4)
│   │   ├── Distance #0  ear to ear  12 cm  → 11.94 cm  −0.5%
│   │   ├── Distance #1  nose to tail  31.5 cm  → 31.71 cm  +0.7%
│   │   ├── Distance #2  paw width  4.2 cm  ⚠ endpoint B: no patch holds the pixel
│   │   └── Distance #3  8 cm  → 7.96 cm  −0.5%
│   ├── Bench Points (2)
```

- The group's count is the number of measurements of every kind, and the group
  is absent when there are none. It opens expanded.
- The rows sit directly under the group. A subgroup per kind is for when a
  second kind exists.
- Each row names its kind and index, then its label (left out when it is
  empty), its stated distance, and the reading: with a unit, the reconstructed
  distance in the measurement's own unit to four significant digits and the
  signed error as a percentage; without one, the reconstructed distance in
  scene units (`0.317 su`). An endpoint with no position puts a warning mark,
  the side and the refusal's short form in place of the reading; an endpoint at
  infinity shows `∞`. The row's hover text gives each endpoint's image and
  pixel, the whole refusal sentence for an endpoint that has none, and what
  `su` means where the file has no unit.

The row gestures follow the interaction rules of
[interaction-rules.md](interaction-rules.md):

- **Click** (G1): select the measurement and its reconstruction, and nothing
  more (§ "The selection").
- **Double-click** (G2, a stored measurement): put it on the bench and focus
  it, or focus its item where it is on the bench already, which selects it
  (`AppState::put_measurement_on_bench`, § "The Edit box"); raise Measurement
  View beside the tree through `show_panel_beside` (G9); and frame the
  measurement in the 3D view: the view keeps its orientation and moves so that
  its finite endpoints are inside the middle half of the viewport, the orbit
  target on their centroid (`Viewer3D::frame_keeping_orientation`). A
  measurement with one finite endpoint centres it at the orbit distance the
  view had, and one with none leaves the view where it is. The raise is a
  layout operation, so the double-click is kept for
  `SceneGraphPanel::take_measurement_open` and applied by `app.rs` once the
  dock is back in the state (`app::apply_measurement_open`). A put refused
  while a background task holds the node is one failed row, and nothing is
  raised or framed.
- **Right-click** (G3): selects nothing and opens the row's menu: *Edit on
  Bench*, which is the double-click, and *Delete*, which deletes the stored
  measurement (§ "Deleting a measurement"), greyed with the reason while a
  background task or a camera model proposal holds the node.

The selected measurement's row is drawn selected.

## Placement

A dock tab, `Tab::MeasurementView`, titled **Measurement View** and named
`measurement_view` in the layout file and on the wire. Its home is the
top-right group with Track View, after it: `[TrackView, MeasurementView,
IntrinsicsDetail]`, which is also the default layout's leaf, with Track View in
front. A measurement is read beside the picture, as a track is. The layout
file's list of panel names includes `measurement_view`; the name is a value a
leaf holds, so `LAYOUT_VERSION` does not change and a layout written before the
panel existed loads with it closed ([panel-layout.md](panel-layout.md)
§ "Validation").

## What the panel shows

**No reconstruction loaded**: one line saying so.

Otherwise the panel opens with the **Edit** box (§ "The Edit box"). With it
clear, the panel shows the selected stored measurement, read-only, as below;
with it ticked, the focused measurement item (§ "The edited mode").

**No measurement selected**: one line saying so, then, where the selected
reconstruction has measurements, each listed as the Scene tree lists it, as a
link that selects it, so the panel is usable without the tree.

**A measurement selected**: the header, then the body for its kind.

### The header

- The kind and index (`Distance #2`) and the reconstruction's label.
- The label, or *No label*.
- For a distance, the stated distance and its unit.
- A line comparing this measurement with the others of its kind in the same
  reconstruction, when at least two have a reading: the median length of one
  scene unit they imply (`sfmtool_core::numeric::median` of `stated /
  scene_distance`, in metres, shown in this measurement's unit) and this
  measurement's deviation from it (`Median over 4: 1 su = 37.8 cm; this one
  +0.4%`). It does not read the file's unit, so it is the same check with one
  and without one, and it is the one that catches a bad measurement among good
  ones. A measurement with no reading gets the line with *this one has no
  reading*.

### The distance body

Two endpoint blocks, **A** and **B** (endpoint 0 and 1 of the format), then
the result block. The blocks are side by side where the panel is wide enough
for two (`BLOCK_MIN_WIDTH` each, a crop and a patch with the frame's margins,
232 points), and one above the other where it is not, as at the panel's width
in the default layout of a 1280 x 720 window. Every line of a block wraps
inside it, so neither block draws over the other.

Nothing in the panel is wider than the panel, in either mode, so it
normally scrolls only up and down. Each row that holds several pieces of
text, such as the header's first line and the edited mode's bench label and
node, wraps between them, and a single word longer than the line breaks. The
edited mode's two header fields take the width beside their names, up to 180
points (`HEADER_FIELD_WIDTH`) and down to 60 (`HEADER_FIELD_MIN_WIDTH`); a
block's image drop-down takes the block's width and cuts a long name short;
the `x` and `y` pixel fields, each with its letter, go one under the other
where the block is too narrow for both; and the crop shrinks to the block's
width where that is under 128 points. Whether the blocks sit side by side, how
wide each block is, and how wide the header fields are, is decided from the
panel's width read before anything in it is drawn, since egui widens a `Ui` to
take in a row that overflows it, and a width read after such a row would be
wider than the panel. The header fields are two rows, each name at the wider
name's measured width, rather than an `egui::Grid`, which lays out from the
column widths of the frame before: on the frame the panel narrows, as on
every frame of a splitter drag, a grid would still be at the old width.
Only in the edited mode and in a panel narrower than about 150 points, too
narrow for a header field at its minimum beside its name, is the content
wider than the panel; egui_dock's tab body, which scrolls both ways for every
panel, then scrolls it sideways. Each block shows:

- **The photograph's name** and the endpoint's pixel, `(412.3, 230.5)`.
- **The crop.** The photograph around the endpoint, 128 points square. Where
  the endpoint resolves, the crop is cut around the chosen candidate's square
  in that image, as Track View's crop is cut around a row's square
  (`patch_pictures::crop::patch_outline` and `region`), with the square's
  outline over it and a cross at the endpoint's pixel. Where it does not, the
  crop is a square of the photograph `UNRESOLVED_CROP_PX` (64) photograph
  pixels across, centred on the pixel, with the cross alone. The crop and
  its hover view are cut in one render from a photograph made of the
  photograph tiles both need (`state::read_peeked`, as Track View's crops
  are read), so nothing is decoded on the GUI thread and no whole pyramid is
  read ([photograph-cache.md](../core/camera/photograph-cache.md) § "In the
  viewer"). While those tiles are loading the cell says *loading…*; while
  the photograph is an out-of-core one waiting for its mipmap
  (`PeekedRead::Building`, `Crop::Building`), the cell draws a spinner with
  *Building mipmap…* under it (`patch_pictures::paint_building_mipmap`),
  which its hover text says too, asks for no photograph tile, and queues
  the photograph for *Build mipmaps*
  ([background-tasks.md](background-tasks.md) § "Build mipmaps"), the crop
  asked for again on every frame, so the frame after the run that makes the
  mipmap reads its tiles; the spinner asks for the frames it turns in, so
  the panel repaints every frame only while it draws one. Where the
  photograph cannot be read, the cell says *no crop* and why in its hover
  text, and the crop is cut again on every frame, so it is drawn once the
  photograph can be read.
- **The crop's hover view.** The photograph's full name, then the same crop in
  context, three times its width and height at 288 points
  (`marks::CONTEXT_FACTOR`, `CONTEXT_HOVER_SIDE`), with the outline and the
  cross, then bullets giving the pixel, the crop's size in photograph pixels and
  the square's two axes in photograph pixels.
- **The patch.** The chosen point's patch bitmap (`PointView::patch_bitmap`),
  64 points square, with a cross at the endpoint's texel `(s, t)` (`s`
  across, `t` up, since a bitmap's rows step along `−v`), so the person sees
  which place on the surface the endpoint names. For a file that stores
  bitmaps it is the point's stored one; for a file that stores none it is the
  display bitmap the open rendered into the same column
  (`PointSet::patch_bitmaps_for_display`). Absent where the endpoint does not
  resolve, where the reconstruction carries no bitmap column, or where the
  point's row is all zero, as Track View and the Track Table treat a row with
  no patch.
- **The resolution.** Where the endpoint resolves: the chosen point's ID with
  Track View's *Copy Point ID* button and its *Go to Point* button, which
  opens the Go to Point dialog holding that ID; the texel `(s, t)`; and the 3D
  position in the file's unit or in scene units, or, for a point at infinity,
  its direction and "This endpoint lies on a point at infinity, so no distance
  can be measured to it." Then "One square holds the pixel." for a candidate
  that stood alone, or "Chosen from N squares", a disclosure listing every
  candidate in the resolver's order with its point, its cost and costliest
  term or why it was excluded, the chosen one marked. Where the endpoint is
  refused: "No position:" and the refusal's sentence, which names the remedy.

The selected endpoint's block is drawn as Track View draws its selected row.

The **result block** gives the result (§ "The readings"):

- With a unit: `Reconstructed: 11.94 cm`, `Error: −0.0600 cm (−0.50%)`, whose
  hover text says the sign in words ("the reconstruction is 0.0600 cm shorter
  than the stated distance"), and `Implied scale: 1.0050×`.
- Without one: `Reconstructed: 0.3175 su` and `Implied scale: 1 su = 37.80 cm`.
- `No distance: endpoint B lies on a point at infinity.`
- `No distance: endpoint A has no 3D position.`, with the refusal's sentence
  in A's block above it.

### Endpoint gestures

These are Track View's row gestures ([track-view.md](track-view.md)), applied
to an endpoint, and they follow rules G1, G2 (a pixel of an image), G3, G5 and
G6 of [interaction-rules.md](interaction-rules.md).
The block's buttons keep their clicks; the name, the pixel, the crop and the
patch are one target. In the edited mode the name and the pixel are fields
(§ "Setting an endpoint by hand"), which keep their own clicks, and the
endpoint's letter beside the image drop-down takes their place in the target.

- **Click** (G1, G5): select the endpoint and its image, and reveal the
  endpoint's pixel: Image Detail pans, never zooms, to show it when its view
  is not showing it (`Look::Reveal`). The 3D view does not move.
- **Double-click** (G2): the click, then camera view through the endpoint's
  image, turned until the pixel is inside the middle half of the viewport on
  both axes, in one animated transition
  (`Viewer3D::look_through_toward_feature`), as a double-click on a Track View
  row does.
- **Hover** (G6): the panel owns the hover while the pointer is over it; an
  endpoint block sets the hovered image to the endpoint's, which brightens its
  frustum and outlines its thumbnail, and clears the hovered point.
- **Right-click** (G3): selects nothing. On a stored measurement it opens no
  menu; in the edited mode it opens the block's menu, *Switch Image…*
  (§ "Switching an endpoint's image").

In the edited mode a click on a block selects the endpoint and its image with a
reveal of the item's pixel (`AppState::select_item_endpoint`), and a
double-click enters camera view toward it, as for a stored row.

## The Edit box

Track View's **Edit** box, with the same meaning: it is ticked exactly when the
focused item is a measurement on the selected node's bench
(`AppState::focused_measurement_label`). Track View and Measurement View each
tick their box only for their own kind of item, so with a measurement focused
Track View's box is clear, and the reverse.

- **Ticking it with a measurement selected** puts that measurement on the bench
  and focuses it (`AppState::set_measurement_editing`, which calls
  `put_measurement_on_bench`): one version, `Put distance #2 on the bench`,
  the item labelled by the measurement's kind and row. Where an item on the
  bench already came from that row (its origin, followed to the cursor), that
  item is focused instead and no version is pushed. Only the put is refused
  while a background task holds the node, and the box is greyed with the
  node's sentence then.
- **Ticking it with none selected** focuses the most recently focused
  measurement item still on a bench (`AppState::most_recent_item_of`), and with
  none of those the box is greyed, naming the ways in.
- **Clearing it** unfocuses the item, which stays on the bench; the selection
  is left as it is.

A double-click on a Measurements row and *Edit on Bench* in its menu do what a
tick does, and raise the panel. On a `sift_files` reconstruction the
measurement goes on the bench and is viewed as on any other, and its commit is
refused with the view-only sentence ([bench.md](bench.md) § "A view-only
bench").

**While a measurement item is focused**, the selected measurement is its origin
followed to the cursor, or none where its row is gone or the person cleared the
selection, and the selected point is left alone: selecting a point on the same
node keeps the item focused, and
selecting any other measurement, or anything on another node, unfocuses it
([bench.md](bench.md) § "The focused item").

## The edited mode

With the box ticked the panel shows the focused item:

- **The toolbar**: *Commit* (§ "The commit"), greyed with the refusal the step
  would give; *Discard*, which takes the item off the bench in one version;
  and *Rename*, which opens a field in place and commits on Enter or *Set*, as
  Track View's do. *Discard* and *Rename* are greyed only while a background
  task holds the node.
- **The header**: the item's label on the bench and its node; where it came
  from, `From Distance #2: Commit replaces it`, or `From no stored row: Commit
  adds it as a new row` where its origin resolves to none; the measurement's
  own label and its stated distance, each in a field (§ "The label and the
  stated distance"); and the median line (§ "Live readings").
- **The endpoint blocks**, as for a stored measurement, read from the item's
  endpoints, with the image name as a drop-down of the reconstruction's images
  and the pixel as two number fields (§ "Setting an endpoint by hand"), then
  *Switch image…* under the pixel, and a right-click menu holding *Switch
  Image…* (§ "Switching an endpoint's image"). The 3D position stays
  read-only, in the resolution, since it follows from the image and the pixel.
  An endpoint that is not placed, which a new measurement's endpoint B is and
  deleting the image an endpoint was in leaves, is a block saying `Endpoint B
  is not placed.` and how to place it, "Right-click a pixel in Image Detail, or
  a point in the 3D view, and choose Set Endpoint B Here."
  (`measurement_view::place_instruction`), with its *Switch image…* greyed.
- **The result block**, from the live readings, or `No distance: endpoint B is
  not placed.` where an endpoint is missing.

### The label and the stated distance

The header shows the measurement's own label and its stated distance as two
fields, **Label** and **Stated distance**, the second holding the number and
the unit in one, `153 cm`. An empty field shows *No label* or *No stated
distance* in grey. A field commits on Enter or when it loses focus, with text
other than what it held when it took focus, and Escape puts it back and
commits nothing, as Track View's *Rename* field reads Enter
([track-view.md](track-view.md)). A field reads whether it has the keyboard
focus from egui's memory, which is what the text input takes typing by, and
not from `Response::has_focus`, which is also false while the window is not
the operating system's front window, so typing that reaches a field, such as
an agent's `type_text` into a viewer behind another window, is committed. The
panel reports the text and the dock makes the step, so a refusal is a failed
Action Log row in the step's words.

Each commit is one bench step on the focused item, made by a pure core function
from one `EditableMeasurement` to the next
([core/bench/editable-measurement.md](../core/bench/editable-measurement.md) §
"The label and the stated distance"), which replaces the item's `Arc` under
the same `ItemId`: one version through `History::push_bench`, listed in the
Edit History and undone by `Ctrl+Z`, with one `Bench` row. Neither changes the
reconstruction or marks the node dirty; the stored row changes only when the
item is committed.

| Step | Label in the Edit History |
|---|---|
| Set the measurement's label (`AppState::set_bench_measurement_label`) | `Set the label of distance #2 to "paw width"` |
| Set the stated distance (`AppState::set_bench_measurement_distance`) | `Set the stated distance of distance #2 to 4.2 cm` |

- **The label** is any text, the empty one included; it is not the item's name
  on the bench, which *Rename* sets.
- **The stated distance** is read by core's `parse_stated_distance`, the parser
  `sfm xform --scale-by-measurements` reads a YAML distance with: a number with
  an optional unit suffix, `12in`, `30.48cm` or `30.48 cm`. A bare number is in
  the reconstruction's `world_space_unit`, or in metres where it has none, and
  the field's hover text names that unit. Text that does not read, a suffix
  other than `mm`, `cm`, `m`, `in` and `ft`, and a value that is not finite and
  strictly positive as the file's `float32` are refused with core's sentence:
  *Cannot set the stated distance of distance #2: the stated distance is -2,
  which must be finite and strictly positive.*
- **A value the item holds already**, the same label or the same distance and
  unit as the file would store them, pushes no version and writes the no-effect
  row (`Set the label of distance #2 to "paw width": no effect, it has that
  label already`).
- **Both are refused** while a background task holds the node and on a
  view-only bench, with the shared sentence (`AppState::bench_edit_refusal`).
  The fields are then greyed, with that sentence as their hover text.

### Switching an endpoint's image

An endpoint sits on one texel `(s, t)` of a point's patch, and every other
observation of that point holds the same texel at a pixel of its own image,
in its own square. So the endpoint can move to any of those photographs
without moving on the surface, and a person picks the one that shows the
endpoint best. Core's `endpoint_views` finds them
([core/reconstruction/measurement-endpoint.md](../core/reconstruction/measurement-endpoint.md)
§ "The views of an endpoint"): for each candidate the resolver finds at the
endpoint's image and pixel, chosen or not, every other observation of its
point, at the pixel where that observation's square holds the candidate's
texel, dropped where that pixel is outside its image or the square faces away
from that camera.

**The button and the menu.** Each endpoint block of the edited mode has
*Switch image…*, and a right-click on the block (G3: it selects nothing)
opens a menu with *Switch Image…*; both open the popup, anchored on the
button. Both are greyed, with the refusal as their hover text
(`measurement_view::switch_refusal`), while a step on the item would be
refused (a background task holding the node, or a view-only bench, in
`AppState::bench_edit_refusal`'s sentence), where the endpoint is not placed
(*Endpoint B is not placed, so it has no image to switch from.*), and where
the resolver refuses it (*Endpoint B has no position, so no other image can be
offered:* and the resolver's sentence), since then no square names a place for
another photograph to show.

**The popup** lists the offers, read when it opens and again after a step on
the item or an edit of the reconstruction. They are grouped by point, the
chosen candidate's point first (`Point pt3d_9f2c41d0_812: the point the
endpoint resolves to`), then the others' in the resolver's order (`another
square holding the pixel`). Each row is drawn as an endpoint block draws its
pictures:

- a crop of that photograph, 96 points square, cut around that observation's
  square as Track View cuts a row's crop, with the square's outline and a
  cross at the offered pixel, and the endpoint crop's hover view;
- the image name and the offered pixel;
- the observation's `ℓ`, in the file's unit, in `su` without one, or in
  degrees for a point at infinity (`ℓ not read` where it has no stored
  reading), and its viewing angle `θ`: the readings the resolver's cost reads,
  so the sharper and less oblique views stand out;
- a warning mark where the endpoint would resolve elsewhere there, because
  another square holds the offered pixel in that photograph and wins (`⚠
  resolves to point pt3d_9f2c41d0_915 there`), or where it would have no
  position there (`⚠ no position there`).

A click on a row's crop, name, pixel, readings or mark is the step, and closes
the popup; a click outside closes it with no step. The panel reports the row
and the dock makes the step, so a refusal is a failed Action Log row in the
step's words.

**The step** (`AppState::set_bench_measurement_endpoint` with
`EndpointMove::Switch`) is made by core's `set_measurement_endpoint`
([core/bench/editable-measurement.md](../core/bench/editable-measurement.md) §
"Setting an endpoint"): one version through `History::push_bench`, which
replaces the item's `Arc` under the same `ItemId`, listed in the Edit History
and undone by `Ctrl+Z`, with one `Bench` row, leaving the reconstruction alone
and the node clean. It sets the image and the pixel only; the point and the
texel the endpoint resolves to there are the row's, unless the row is marked.

| Step | Label in the Edit History |
|---|---|
| Switch an endpoint's image (a popup row) | `Moved endpoint A of distance #2 from IMG_0004.jpg to IMG_0011.jpg` |
| Set an endpoint's image and pixel (the block's fields, *Set Endpoint A Here* and *Set Endpoint B Here*, and the wire's `set_bench_measurement_endpoint`; § "Setting an endpoint by hand") | `Moved endpoint A of distance #2 to IMG_0004.jpg (412.3, 230.5)` |
| The same, for an endpoint not placed | `Set endpoint B of distance #2 to IMG_0050.jpg (88.0, 301.5)` |

An endpoint at that image and pixel already, as the file would store the
pixel, pushes no version and writes the no-effect row. The step is refused
while a background task holds the node, on a view-only bench, for a switch of
an endpoint not placed, and for an image past the image set or a pixel outside
`[0, width) × [0, height)` of its image, in the commit's sentence (*Cannot
move endpoint A of distance #2: endpoint A's pixel (640, 3) is outside image
0, which is [0, 640) x [0, 480).*).

### Setting an endpoint by hand

The image and the pixel are the endpoint; everything else a block shows is
derived from them. So both can be set directly, in three places, each the step
of § "Switching an endpoint's image" with `EndpointMove::Set` on the focused
item, which the wire's `set_bench_measurement_endpoint` also makes: one
version, labelled by the pixel, with a refusal written as one failed `Bench`
row in the step's words. None of them sets the 3D position, the point or the
texel: they follow from the image and the pixel through the resolver every
time, and a setting that disagreed with them could not be kept.

**In the endpoint block.** In the edited mode the block's first line is the
endpoint's letter beside a drop-down of the reconstruction's images, named
`endpoint A image` in the widget listing, and the next is the pixel as two
number fields, `x` and `y` (`endpoint A pixel x`, `endpoint A pixel y`). A
field shows its coordinate as the shortest text that reads back as the same
`float32`, which is how the file stores it (`148.00723`), so a field left as it
is commits nothing. Each field is wide enough for that text: for eleven
characters of digits and a decimal point, which hold any such coordinate from
0.1 up to 100,000 pixels (`117.847664`, `0.123456789`), or for the text it
holds where that is wider, as a smaller coordinate's leading zeros can make it
(`measurement_view::pixel_field_width`).

- **Picking another image** in the drop-down is one step that sets that image
  and the pixel the endpoint holds; picking the image it holds asks for
  nothing.
- **A pixel field** commits as the header's fields do (§ "The label and the
  stated distance"): on Enter or when it loses focus, with text other than what
  it held when it took focus, and Escape puts it back and commits nothing. Each
  field keeps its own draft, so moving straight from `x` to `y` commits the `x`
  once. A commit is one step that sets the image the endpoint holds and the
  pixel with the coordinate typed.
- **Text that is not a finite number** asks for no step and writes one failed
  row: *Cannot move endpoint A of distance #2: "abc" is not a number.* A pixel
  outside the image is refused by the step, in the format's sentence above,
  whether a field or the drop-down named it.
- The drop-down and the fields are **greyed** while a step on the item would be
  refused (`AppState::bench_edit_refusal`: a task holding the node, or a
  view-only bench), with that sentence as their hover text.

An endpoint that is not placed has no fields, since it has no image or pixel to
start from; the menus below place it.

**In Image Detail.** With a measurement item focused, Image Detail's context
menu gains `Set Endpoint A Here` and `Set Endpoint B Here` after its bench
entries, which set that endpoint of the focused item to the image shown and the
pixel the menu was opened at
([multi-panel-image-browser.md](multi-panel-image-browser.md) § "Image Detail:
the context menu"). The context menu is the viewer's way to name a pixel, so
this is where a person zoomed in on a photograph places an endpoint precisely.
Any pixel of the photograph is taken; the endpoint does not have to resolve,
since the pixel is what the file stores.

**In the 3D view.** With a measurement item focused, a right-click on a point,
which selects it (G3) and keeps the item focused since the point is on the
item's node, offers the same two entries after the point's own, which set the
endpoint to the pixel that shows the **spot right-clicked** on the point's
patch, or to the point's reference pixel where the right-click was not on its
drawn patch (§ "The spot under the pointer")
([viewport-navigation.md](viewport-navigation.md) § "The point context menu").

The entries of both menus are offered only while a measurement item is
focused, and either menu places an endpoint that is not placed (`Set endpoint B
of distance #2 to ...`). They are greyed with the reason as hover text
(`EndpointTarget::refusal_on`) while a step on the item would be refused, and
on a node other than the item's, naming the item's node (*The measurement
being edited, distance #2, is on dino, so its endpoints are set from that
reconstruction's images and points.*). A selection on another node unfocuses
the item ([bench.md](bench.md) § "The focused item") and Image Detail shows a
photograph of the selected node, so the menus meet this last case only on the
frame a right-click in the 3D view lands on a point of another node; once that
point is selected no measurement item is focused, and the entries are gone.

#### The spot under the pointer

A point in the 3D view is not a pixel, and a measurement's endpoint must be
one. The pixel is taken in the image of the point's **reference observation**
(`tracks/reference_observations`), the observation its bitmap on screen was
rendered from. The reference is the one the point holds in the version on
screen, a pick only the display render made included.

**The spot.** The viewport reads where the right-click landed on the point's
drawn patch (`Viewer3D::texel_under_pointer`): its pick ray, through the
click's place with the camera of the frame the click landed on, taken back
through the node's display transform as the bench handles' readings are, is met
with the plane of the point's patch frame (`measurements::texel_under_ray`).
The ray is kept from the click's frame because the point is known only once
the pick at the click's pixel is read back, a frame or more later
([viewport-navigation.md](viewport-navigation.md) § "The point context
menu"). The surfel is drawn at the HUD's patch size,
`2^patch_size_log2` times the stored patch, with its bitmap stretched over it,
so the hit is divided by that scale: the texel `(s, t)` is the one the bitmap
shows under the pointer, in the units the resolver and `OrientedPatch` use,
each in `[-1, 1]`. The menu records the texel with the point
(`measurements::PointSpot`) on the frame it opens, as Image Detail records its
`context_menu_pixel`, because the entries are chosen on later frames, after
the pointer has moved. An entry then maps the texel into the reference
image (`measurements::spot_pixel`): the pixel where the reference
observation's square, the patch slid onto its keypoint and seen through that
image's own intrinsics, holds the texel, core's `texel_pixel`, the mapping
`endpoint_views` offers its views through. The endpoint resolves back to the
same point at that texel, unless another square in that image holds the pixel
and wins the ranking.

**The reference pixel** is what the entries take where there is no spot: the
image of the reference observation and its keypoint
(`measurements::reference_pixel`). The keypoint is where the reference
observation's square is centred, so the endpoint resolves to texel `(0, 0)` of
the point's own patch, which is the point itself. The entries fall back to it
where:

- the pointer is not on the point's drawn square: the node draws its points as
  dots (its style is Dots, it has no patch bitmaps to draw surfels with, its
  points are hidden, or the patch opacity is 0), the point itself is drawn as
  a dot because its own patch frame or bitmap row is empty (written where the
  reference rule picks no observation), or the click landed on the point's dot
  outside the square;
- the ray sees the square's back, which the viewport culls, or meets its plane
  at under 5° (the bench's `MIN_PLANE_ANGLE_DEG`): a pixel of pointer travel
  then crosses more than eleven pixels' worth of the square, so the texel under
  the pointer is not the one aimed at, the reason the bench's plane handles
  refuse a press under the same angle;
- the texel's place falls outside the reference image, or the square faces
  away from that image's camera.

The 3D view's entries are greyed, with hover text saying why, where the point
has no reference pixel, asked in this order:

- it has no patch frame (a `sift_files` reconstruction, or a point the patch
  pass left without one): "This point has no patch, so no pixel stands for it.";
- it has no reference observation (`-1`, or no column): "This point has no
  reference observation to take a pixel from.";
- it is a point at infinity: "This point is at infinity, so no distance can be
  measured to it."

The entries do not check that the pixel resolves back to the same point: a
neighbouring square can hold the same pixel and win the ranking. The endpoint
block then shows which point it resolved to, as for any endpoint.

### Live readings

The focused item's readings are computed from the item, not read from the
stored row: each placed endpoint is resolved against the version at the cursor
with the resolver's default options (`measurements::item_reading`, which calls
core's `resolve_distance_endpoints`), and the result block and the median line
are computed from those. The node keeps them against the item's `Arc` and the
version's document serial (`SceneNode::item_reading`, beside
`SceneNode::measurement_readings`), so a step on the item, or an edit of the
reconstruction under it, reads them again on the next frame, and the panel and
every view drawing the focused item (§ "Drawing measurements") read one
resolution between them. The median line
compares the item with the stored measurements other than its origin, so a
person sees whether a change brings it into line before committing it.

### The commit

*Commit* writes the focused item into the reconstruction
(`AppState::commit_bench_measurement`, [bench.md](bench.md) § "Measurement
items"): the row its origin names at the cursor is replaced, or, where it names
none, a row is appended. It is one ordinary edit, as a track's commit is
([edits/commit-track.md](edits/commit-track.md)): one version, one Action Log
row of kind `Edit` (`Committed distance #2 in dino`, `Added distance #4 in
dino`), undoable like any other, and the node is marked dirty. The new value
shares the old one's base, so no point, track or patch bitmap is copied and no
point index moves. The item stays on the bench and focused, its origin
re-seated on the row written, which is selected. A commit onto a row that
already holds exactly what the item holds pushes no version and writes a
no-effect row, and still selects the row.

The button is greyed with the refusal as its hover text, asked of the item it
would commit (`AppState::measurement_commit_refusal`), so the button and the
step cannot disagree: a background task or a camera model proposal holding the
node, a view-only bench, and core's refusals, an endpoint not placed, no stated
distance, an endpoint's image past the image set, a pixel outside its image
([core/bench/editable-measurement.md](../core/bench/editable-measurement.md)
§ "The commit"). An endpoint the resolver refuses does not stop a commit, since
the pixel is what the file stores.

## Creating a measurement

A new measurement is started where its first endpoint is named, in the two
menus that set an endpoint, and finished with the steps above.

**Starting it.** `New Measurement from Here` is in Image Detail's context menu,
at the pixel the menu was opened at of the photograph shown
([multi-panel-image-browser.md](multi-panel-image-browser.md) § "Image Detail:
the context menu"), and in the 3D view's point menu, at the pixel that shows
the spot right-clicked on the point's patch, or its reference pixel off the
patch (§ "The spot under the pointer",
[viewport-navigation.md](viewport-navigation.md) § "The point context menu"),
whose right-click selects the point first (G3). Any pixel of the photograph is
taken; it does not have to resolve, since the pixel is the stored measurement.
Both entries are offered whether or not an item is focused, after a separator
and before *Set Endpoint A Here* and *Set Endpoint B Here*.

Either one, in one version (`Put distance from IMG_0042@142,198 on the
bench`), puts a new measurement item on the node's bench with endpoint A at
that pixel and nothing else (`AppState::start_measurement`, which calls core's
`start_measurement`), focuses it, and raises Measurement View beside the panel
the gesture was made in, through `show_panel_beside` (G9): where Measurement
View is a tab behind that panel, it stays behind. The item has no origin, so
focusing it clears the selected measurement, and the header says `From no
stored row: Commit adds it as a new row`. Its bench label is minted in core
from where it started, the image stem and the rounded pixel after `distance
from `, `distance from IMG_0042@142,198`, with the bench's collision suffix
(`distance from IMG_0042@142,198 (2)`) where another item holds that
([core/bench/bench.md](../core/bench/bench.md) § "Labels"). Its own label and
stated distance are empty, and the header's two fields show *No label* and *No
stated distance* in grey, the second's hover text naming the unit a bare
number is in.

**Refused**, with the sentence as the entries' hover text and as one failed
`Bench` row: while a background task holds the node, and on a view-only bench
(`AppState::bench_edit_refusal`), since every step that would finish the item
is refused there and it could never be committed; in the 3D view, where the
point has no reference pixel, with that section's three sentences, asked after
the node's; and for a pixel outside the image, in the commit's sentence
(*Cannot start a measurement there: endpoint A's pixel (640, 3) is outside
image 0, which is [0, 640) x [0, 480).*), which a menu opened over the
photograph does not name. A refused start pushes no version and raises
nothing. The 3D view's entry is greyed by `MeasurementStart::at_point`, which
the dock reads each frame for every node, so the entry on a point of any node
says what the step would.

**Finishing it.** Endpoint B's block says how to place it (§ "The edited
mode"), and *Set Endpoint B Here* in either menu places it (`Set endpoint B of
distance from IMG_0042@142,198 to IMG_0050.jpg (88.0, 301.5)`). The stated
distance and the label are typed into the header's fields. Each is a bench
step, so a mistake is undone like any other, and a half-made measurement waits
on the bench, in the Scene tree's Bench Measurements group, until it is
finished or discarded. *Commit* is greyed with the refusal naming what is
missing, first endpoint B and then the stated distance, until both endpoints
and the distance are given, and then appends the measurement as a new row
(`Added distance #5 in dino`), as it does for any item whose origin resolves to
none (§ "The commit").

## Deleting a measurement

*Delete* in a Measurements row's menu deletes the stored measurement
(`AppState::delete_measurement`): a direct edit of the reconstruction, as
deleting a point is, and not a bench step. One version (`Deleted distance #2 in
dino`), one `Edit` row, the node marked dirty. The rows after it move down by
one; the selected measurement follows, and every measurement item's origin
follows through the version graph. An item whose origin was the deleted row
stays on the bench with no row to replace, so committing it adds the
measurement back as a new row.

## Drawing measurements

**Which measurements are drawn.** The selected measurement, where its
reconstruction is visible, and the focused measurement item. A **Show all
measurements** option, off by default, draws every stored measurement of every
visible reconstruction as well, so a reconstruction with many measurements
stays readable, and the person who wants to see them all together asks for it.
Each view has its own: a checkbox in the 3D view's HUD, in its Layers section,
and one in Image Detail's toolbar ([viewport-hud.md](viewport-hud.md),
[multi-panel-image-browser.md](multi-panel-image-browser.md) § "Image Detail").
With it on, the selected measurement is drawn wider (3.5 points against 2) and
the others at 55% opacity, so the selected one is the brighter. A hidden
reconstruction draws none of its stored measurements; the focused item is the
bench's layer and is drawn whatever the node's eye says, as the bench layers
draw a focused track.

A stored measurement is drawn in its node's tint, or a light grey (`STORED_RGB`)
on a node drawn in its own colours. **The focused item is drawn as the bench's
layer**, in the bench's `in` violet and the bench layers' 2.5-point stroke, so a
person tells the measurement being edited from the stored one it came from when
both are drawn: with an item focused its origin is the selected measurement,
and the item is drawn over it. The selected endpoint's mark is drawn 1.6 times
larger, as the 3D view draws the selected observation's circle larger: on the
item while one is focused, and on the selected measurement otherwise. Every mark
and line has a dark halo so it reads over a bright photograph or cloud.

**In the 3D view** a drawn measurement puts a mark, a ring with a dot at its
centre, at each endpoint that resolves to a finite position, and a line between
the two where both do. The positions are the resolved ones put through the
node's display transform, as its points are. **The marks and lines are drawn
in the scene**, by the bench figure's shader into a layer of their own
([viewer-3d-bench-layer.md](viewer-3d-bench-layer.md) § "Depth against the
scene"), so a person sees how a measurement meets the geometry around it, as
the hidden part of a highlighted surfel's outline shows
([patch-rendering.md](patch-rendering.md)): a piece in front of the scene is
drawn at full strength, and a piece behind it at `BEHIND_OPACITY` (0.45) with
no fade with distance, so a measurement far behind the cloud is as findable as
one just behind it. Each piece is one instance of the overlay ribbon store,
the buffer the bench figure's segments share
([viewer-3d-bench-layer.md](viewer-3d-bench-layer.md) § "The overlay ribbon
store"), carrying its own colour and its sizes in points
(`viewer_3d::measurements::instances`): the line and its halo are segments,
and a mark is three instances, its ring's halo, its ring and its dot, the ring
and its halo each a ring and the dot a disc about the endpoint, which the pass
turns to face the eye, so a mark keeps its size on screen. Its edges, and the
line's, are anti-aliased over about one point. The line and its halo stop on
screen at the outer edge of each end's ring, inset there by the ring's radius
and half its stroke (`OverlayInstance::inset`): drawn to the endpoint, the
line's stroke would cross the inside of the ring to the dot, so the mark would
read as a spiral rather than a ring around a dot, and would lie over the ring's
stroke, where the two translucent strokes of a dimmed measurement add up to a
brighter spot. The ring's halo, drawn before every stroke, bounds the join. An
end clipped at the near plane has no ring and is not inset. Within the region every halo
comes first, then every stroke (each measurement's line and then its rings),
then every dot, each in the layer's order, so one measurement's halo never
crosses another's stroke. Nothing in an instance depends on the camera, so the
instances are built only when the layer is: the 3D view builds them when its
`MeasurementLayerCache` holds a layer other than the one they were built from
(by `Arc` pointer), which happens only when the layer's key changes, and writes
them into the measurements' region only then. Orbiting, zooming and
resizing the panel build and write nothing, even with *Show all measurements*
on over a thousand measurements. The layer is a texture of its own, cleared to transparent
every frame, that holds egui's premultiplied colours as egui's own textures
hold them; the view paints it over the scene, the grid and the axes, where
the overlay was painted, so egui composites a translucent colour (the dimmed
others, the halo) as it composited the overlay's, to within one level in 255,
on an sRGB window surface or an encoded one. Every piece
is drawn toward the eye by `PULL_POINTS`
(10.75: the selected ring's radius and its halo) as a world length at its own
depth, since an endpoint lies on the surface it was read from and a mark
exactly there would be half in front of that surface and half behind it.
Sized in points, the pull keeps a mark and its halo in front of a surface
turned up to 45 degrees from the view whatever the field of view and the
panel's height. One panel point at view depth `d` is
`d · 2·tan(vfov/2) / height` long, so the pull is the same fraction of every
piece's depth (`toward_eye_fraction`, at most a quarter, and none for a panel
with no height), a uniform scale toward the eye: each piece moves along its
own line of sight, which leaves its projection where it was. The 3D view
computes the fraction once a frame and hands it to the pass with the panel's
pixels per point, and the vertex shader moves each piece by it, so the
instances need not change when the field of view or the panel does. An
endpoint at infinity or refused draws nothing, and the other endpoint still
draws its mark; a finite endpoint behind the near plane draws no mark, and a
line with one end behind it is clipped at the near plane, both by the pass as
it clips the bench figure's segments. The pass writes no pick id and no depth,
and the hit test below reads the projected positions alone, so a measurement
behind the scene is hovered and clicked as one in front of it is. The hover
text and the off-screen indicator are drawn over the view.

**In Image Detail** a drawn measurement with an endpoint in the image shown
puts a mark at that endpoint's pixel, and a line between the two where both are
in that image. The mark is drawn whether or not the endpoint resolves, since the
pixel is the stored measurement; a refused endpoint's mark is in the warning
colour. A pixel that is not finite, which the file reader refuses but an
in-memory reconstruction can hold, draws no mark and so no line. An item with
one endpoint placed, such as a new one, draws that mark alone.

**Hover and click.** Hovering a mark or a line in either view shows the
measurement's row text beside the pointer: the Scene tree row's for a stored
measurement and, for the item, its label with the reading, or with where each
endpoint is while one is not placed or no stated distance is given. The text
sits right of and below the pointer where that fits and on another side of it
where it does not (G6), and the pointer takes the pointing hand. A click on a
stored measurement's line selects the measurement (G1); a click on a mark selects the measurement and that endpoint,
as a click on the endpoint block does (§ "Endpoint gestures"), and a
double-click on it does what a double-click on the block does, camera view
through the endpoint's image turned toward its pixel. On the focused item a
mark's click selects that endpoint of the item, and a click on its line asks
for nothing, the item being focused already. **A mark, a line and the indicator
own their press** (G7): the press is taken when it lands
(`measurement_marks::follow`), so the 3D view neither orbits nor picks a point
under it, and Image Detail neither pans nor selects a feature's point under it.
Only an unmodified primary press is taken; the bench figure's handles and the
eraser brush take a press first, although the 3D view draws the marks and
lines over the figure. The reach is 9 points
from a mark and 6 from a line, marks tried before lines and the topmost
measurement first.

**The handles.** In Image Detail the focused item's endpoint marks are handles:
the pointer takes the open hand over one and the closed hand while it is held.
Dragging one draws the mark at the pixel under the pointer, and the release is
the *Set an endpoint's image and pixel* step
(§ "Setting an endpoint by hand") with the image shown and that pixel, held
inside the image, so a drag past an edge lands on the edge: one version when
the drag ends, nothing while the button is down, and Escape abandons it. A
press that moves less than 3 points is a click. The item's marks are not
handles where a step on it would be refused, on a view-only bench or while a
background task holds the node (`AppState::bench_edit_refusal`), and a drag
there asks for nothing. The 3D view has no handles: an endpoint's 3D position
follows from its image and pixel and is never set.

**A selected measurement off screen.** When no part of the selected
measurement's marks and line is inside the view, the view draws an indicator 24
points inside its edge: an arrow pointing toward the measurement, with its kind
and index (`Distance #2`) behind it. A click on either is the indicator's.

- In the 3D view the arrow points toward the projection of the midpoint of the
  two finite endpoints, or of the one finite endpoint. The direction on screen
  is the point's view-space `(x, −y)`: in front of the eye that is the
  direction of its projection from the view's centre, and behind it, where a
  projection lands on the other side, it is that projection reflected through
  the centre, so the arrow points the way to turn. A measurement with no finite
  endpoint has none. A click frames the measurement with
  `Viewer3D::frame_keeping_orientation`, as the double-click on its row does.
- In Image Detail, when an endpoint is in the image shown but outside the part
  the panel shows, the arrow points to that pixel, or to the midpoint of the
  two where both are in the image, and a click centres the panel on it at the
  current zoom (`Look::Pixel` with no zoom).

The indicator follows the selection only. With *Show all measurements* on, the
others are clipped as any scene geometry is.

## MCP

The panel's content is reachable over MCP ([mcp-server.md](mcp-server.md)):

- `get_scene` gives each reconstruction a `measurement_counts` block, by kind,
  and its `selection` block carries `measurement` (its reconstruction label,
  `measurement_kind` and `measurement_index`) and `measurement_endpoint` (`a`,
  `b` or null).
- `list_measurements` lists every measurement of a reconstruction with the
  reading its Scene tree row shows and the result's numbers.
- `get_measurement` gives one measurement whole: each endpoint's resolution
  with every candidate, or its refusal, the result, and the median line.
- `select_measurement` selects a measurement as a row click does, and, naming
  an `endpoint`, that endpoint as a click on its block does, including the
  reveal.
- `put_measurement_on_bench` is a row's double-click without the raise and the
  framing, and where the item is on the bench already its reply's `report`
  says so and names it; `set_bench_measurement_label` and
  `set_bench_measurement_distance` are the header's two fields, and
  `commit_bench_measurement` is *Commit*, each on the item named or the focused
  one; `delete_measurement` is *Delete*.
- `get_measurement_endpoint_views` gives the offers of an endpoint's *Switch
  image…* popup, refused in the button's words, and
  `set_bench_measurement_endpoint` moves an endpoint to any pixel inside an
  image, not only an offered one, refusing a pixel outside it in the commit's
  sentence (§ "Switching an endpoint's image"); it is the step the block's
  fields and the two menus' *Set Endpoint … Here* make.
- `get_point_reference_pixel` gives a point's reference pixel, refused in the
  3D view's entries' words (§ "The spot under the pointer"), so an agent places
  an endpoint from a point as that menu does for a right-click off the point's
  drawn patch, passing its `camera_image` and `pixel` to
  `set_bench_measurement_endpoint`, or starts a measurement there, passing them
  to `create_bench_measurement`. It has no pointer, so it gives no spot on the
  patch; a right-click through `click` in the 3D view does, as a person's
  does.
- `create_bench_measurement` is *New Measurement from Here* without the raise:
  the item with endpoint A at a pixel of a camera image, put on the bench and
  focused, refused in the panel's words (§ "Creating a measurement").
- `get_measurement` naming an `item` gives a measurement item on the bench
  with its live readings and its `origin`, and `get_bench` lists measurement
  items beside tracks. `focus_bench_item`, `unfocus_bench_item`,
  `rename_bench_item` and `discard_bench_item` work on a measurement item as on
  a track.
- `screenshot` accepts `measurement_view` as a `panel_name`.
- `get_viewer_3d_display` and `set_viewer_3d_display` carry the 3D view's
  *Show all measurements* as `show_all_measurements`, and
  `get_image_detail_display` and `set_image_detail_display` carry Image
  Detail's under the same name. No tool sets an endpoint's 3D position.

## Artificial measurements

No checked-in reconstruction has measurements, and the panel cannot be judged
or tested without them. They are made, never checked in, in three ways. The
first real ones will be `dino_dog_toy`'s four tape-measured distances, entered
by hand with the panel; when they are checked in the file keeps its name, since
the `_vNN` suffix versions the reconstruction's cameras, poses, points, tracks
and patch frames, and adding measurements changes none of those.

**In Rust tests** (`sfm-explorer` lib tests, headless): a small synthetic
reconstruction built in the test
([measurements/fixture.rs](../../crates/sfm-explorer/src/measurements/fixture.rs)),
as the resolver's tests build theirs, whose five measurements cover two
endpoints in one image, endpoints in two images, a refused endpoint, an
endpoint at infinity, and a bad measurement among good ones, each with a
reading known by construction.

**In `ui_basic`**: a copy of the seoul bull ground truth written to a
temporary directory with two distance measurements, recording the dataset as
its workspace.

**For screenshots and for a person or an agent trying the panel**:
[scripts/measurements/add_artificial_measurements.py](../../scripts/measurements/add_artificial_measurements.py)
reads a reconstruction, gives it a set of distance measurements, and writes the
result to a path it is given, refusing a path under `test-data/`. Its endpoints
on points are observations' keypoints that resolve alone to their own point, so
each lands on texel `(0, 0)` of a known point's patch, and its stated distances
are the reconstructed distances times a known factor, so the expected error is
known. It writes, in order:

| Case | How it is made |
|---|---|
| Both endpoints in one image | two observations' keypoints in the image with the most observations, stated at `--factor` (default 1.02) times the reconstructed distance, in centimetres |
| Endpoints in different images | keypoints of observations in two images, at the same factor |
| A second one in one image | two more keypoints of the first image, at the same factor |
| One bad measurement among good ones | two keypoints of the second image, at `--bad-factor` (default 1.2), so the header's median line flags it |
| An endpoint that is refused | a pixel of the first image no square holds, from a grid over the image |
| An endpoint held by several squares | a pixel two squares hold, the midpoint of two nearby keypoints or a keypoint another square holds too, where one is found among the first observations of the images |

`--no-unit-copy` writes the same set to a second copy with `world_space_unit`
cleared. The copy is written through `SfmrReconstruction.save` with the
source's workspace, so the viewer finds the photographs wherever the copy is
written. The ground truths are in metres, so the copies with a unit report
errors in real units. A point at infinity is not in every ground truth, so the
at-infinity case is covered by the Rust tests.

## Testing

- **The readings** (`measurements/tests.rs`): with a unit, the error and
  implied scale against hand-computed values; without one, the scene distance
  and the length of a scene unit; a file in metres with a measurement in
  inches; a refused endpoint and one at infinity naming their side; a node
  under a scaled display transform reading the same as at the identity while
  its drawn endpoints scale; the row text of each case; the median line
  flagging the bad measurement and absent with one reading; the readings
  resolved once per version (the same `Arc` twice, a new one after an edit).
- **The selection** (`measurements/tests.rs`): selecting a measurement selects
  nothing else and clears the endpoint; selecting an endpoint selects its image
  with a reveal of its pixel and keeps the selected point, and another image
  clears it; deleting an image moves the selection with its measurement or
  clears it; undo and redo keep or clear it as § "The selection" says.
- **The Scene tree** (`scene_graph/tests.rs`): the group absent on a node
  with no measurements and present with one row per measurement; a row's click
  selects it and asks for nothing else; a double-click asks for it to be
  opened, which the panel keeps once; a right-click selects nothing.
- **The double-click's effect** (`app/tests.rs`): it selects the measurement,
  opens the panel, and frames the finite endpoints without turning, both inside
  the middle half; one finite endpoint keeps the orbit distance; a row past
  the rows moves nothing. The framing itself (`viewer_3d/framing/tests.rs`):
  every point inside the middle fraction, one on its edge, the orientation
  kept; and nothing moves while a camera is held for Move Camera
  (`camera_lock/tests.rs`).
- **The panel** (`measurement_view/tests.rs`, whole headless egui frames): the
  empty state and its links; the header, both blocks and the result's numbers;
  the result block's scene-unit, at-infinity and refused cases; a refused
  endpoint's sentence; a block's click, double-click, hover and right-click.
  (`dock/tests.rs`): a double-click's camera view is the one
  `look_through_toward_feature` gives the endpoint's pixel.
- **The crops** (`measurement_view/tests/crops.rs`): the two endpoints' crops
  and hover views, read from photographs on disk with a cache budget and
  with one that keeps nothing, equal the crops the whole pyramid gives; until
  they are drawn each says *loading…*, each photograph is asked for once
  over three frames, nothing is decoded on the GUI thread, and a crop and a
  hover view that need different tiles take no more reads than one render
  does; a photograph that cannot be read gives *no crop*, and the crop is
  drawn on the frame after the photograph lands in the cache; and crops
  whose out-of-core photographs wait for their mipmaps, under a budget
  smaller than a photograph and one of 0, each draw a spinner with
  *Building mipmap…*, ask for no photograph tile and queue each photograph
  once, and once *Build mipmaps* has made the mipmaps are cut with a few
  reads, after which the frames ask for nothing and repaint no more.
- **The tools** (`mcp/tests/measurements.rs`): `get_scene`'s counts,
  `list_measurements`' rows and results, `get_measurement`'s resolutions,
  refusal and median line, an index past the rows refused with the count, and
  `select_measurement` with and without an endpoint.
- **Undo and redo** (`bench/measurement/tests.rs`, `measurements/tests.rs`):
  the selection carried back onto its own measurement by an undo of a
  measurement delete and of an image delete, and down again by the redo; a
  redo that drops the selected row clearing it.
- **The save** (`state/save/tests.rs`): a save after a delete and an
  appending commit writes the value's own rows, read back from the file.
- **The bench** (`bench/measurement/tests.rs`): a put copies the row and
  focuses the item, selecting its row and leaving the point selection; a
  second put focuses the same item with no version; a point selection keeps a
  focused measurement and another measurement's unfocuses it; Track View's and
  Measurement View's labels each name their own kind; the *Edit* box's put,
  focus, unfocus and refusal; the origin following a measurement delete, its
  undo and redo, and a deleted origin's commit appending; the origin and the
  endpoints following an image delete; a commit replacing the row with the
  base shared, the node dirty and the row selected, and its undo; a commit that
  changes nothing; the commit refusal matching the button's; a view-only bench
  taking a measurement, renaming and discarding it, and refusing its commit;
  an item endpoint's click; a Bench row's double-click raising the item's
  panel beside the Scene tree, never over it.
- **The edited mode** (`measurement_view/tests.rs`): ticking the box asks for
  the put; the toolbar, the header and the live readings, which follow a step
  on the item while the stored row's do not; an unplaced endpoint's block, the
  result's line and *Commit* greyed with the refusal; the label field
  committing on Enter, the stated distance field when it loses focus, Escape
  putting a field back, a field left unchanged asking for nothing, and the
  fields greyed while a task holds the node; and an empty field taking typing
  and committing it on Enter while the window lacks the operating system's
  focus. (`dock/tests.rs`): a committed field reaching its step, and a
  refused one writing a failed row.
- **The layout** (`measurement_view/tests.rs`): at the width the panel has in
  the default layout of a 1280 x 720 window the blocks are one above the
  other, inside the panel and not overlapping, in both modes, and side by side
  where the panel is wide. (`measurement_view/tests/layout.rs`): drawn in a
  scroll area that scrolls both ways, with the tab body's margin, as
  egui_dock draws a tab body, at 200, 272, 428, 500 and 900 points, with long
  image names and a long node label, no widget reaches past the panel's right
  edge, neither pixel field of a block is squeezed narrower than the other,
  two pixel fields on one line are level, and the blocks stack below two
  blocks' width and sit side by side above it: read-only with nothing
  selected and with a measurement selected, edited with both endpoints
  placed, and a new item with endpoint B not placed. The same holds on the
  first frame at each width after frames at 900 points (or 200, for a width
  where the blocks sit side by side), read only and edited.
- **Switching an endpoint's image** (`measurement_view/tests.rs`): no button
  on a stored measurement; the button enabled for an endpoint that resolves
  and greyed with its sentence for one refused, one not placed, and while a
  task holds the node; a right-click selecting nothing and opening the menu,
  whose entry opens the popup; the button opening it too; a row's heading,
  readings and mark; a click on a row asking for the switch once and closing
  the popup. (`dock/tests.rs`): a row reaching its step.
  (`bench/measurement/tests.rs`): the step's label, its undo and redo, the
  point and texel kept, the stored row and the dirty flag left alone, the
  no-effect row, the labels of a pixel set outright and of an endpoint placed,
  and the refusals.
- **The label and the stated distance** (`bench/measurement/tests.rs`): each
  step's label in the Edit History, the stored row and the dirty flag left
  alone, the item's ID kept, undo and redo; a bare number in the file's unit,
  in feet, and in metres with no unit; the no-effect row; text that does not
  read, an unknown unit and a value that is not positive refused in core's
  sentence; both refused while busy, on a view-only bench, and for a track.
- **The bench tools** (`mcp/tests/measurements.rs`): `put_measurement_on_bench`
  once and twice, the second's `report` naming the item, `get_bench` listing
  the item, a track tool refusing it; `set_bench_measurement_label` and its
  no-effect answer; `set_bench_measurement_distance` with text, a JSON number
  and a refused value;
  `get_measurement` with an item; `delete_measurement` and
  `commit_bench_measurement`'s no-effect and appending commits; a track tool
  naming a measurement item saying what it is;
  `get_measurement_endpoint_views`' offers and `set_bench_measurement_endpoint`
  moving the endpoint to one, onto the same point, with their refusals;
  `get_point_reference_pixel` by index and by id, the endpoint set there
  resolving to the point, and its two refusals.
- **Setting an endpoint by hand** (`measurement_view/tests.rs`): a pixel field
  committing on Enter as one step that sets the image and the pixel, and once;
  a field committing when it loses focus; Escape putting it back; moving from
  `x` to `y` committing the `x` once; text that is not a number refused; the
  drop-down listing the images, picking another setting it with the pixel kept
  and picking the one held asking for nothing; the drop-down and the fields
  greyed while a task holds the node. (`dock/tests.rs`): a set reaching its
  step, a pixel outside the image a failed row in the format's sentence, and a
  refused number a failed row. (`image_detail/tests.rs`): *Set Endpoint A
  Here* and *Set Endpoint B Here* absent with no measurement item focused,
  drawn after the bench entries with one, and greyed with the dock's sentence.
  (`viewer_3d/tests.rs`): the right-click on a point asking for it to be
  selected and offering the two entries after the point's own with a
  measurement item focused, and not without; an entry reporting its endpoint
  and point; both greyed, asking for nothing, on a point with no reference
  pixel; each entry setting its endpoint to the pixel that shows the spot
  right-clicked under a display transform, resolving back to the point at that
  texel. (`bench/measurement/tests.rs`): the point menu's selection keeping
  the item focused; *Set Endpoint B Here* moving B to the reference pixel in
  one step that leaves the node clean; *Set Endpoint A Here* placing an
  endpoint not placed; a point with no reference pixel a failed row; the
  target absent with nothing or a track focused, and its refusals on another
  node, while busy and on a view-only bench.
- **Creating a measurement** (`bench/measurement/tests.rs`): *New Measurement
  from Here* in Image Detail as one version labelled by where it starts,
  endpoint A alone, the item focused with no origin, the selected measurement
  cleared, the node clean, Measurement View raised, a second at the same
  rounded pixel taking the collision suffix with a track focused, and its
  undo; the raise never over Image Detail or the 3D view where the panel is a
  tab behind it; the 3D view's start at a point's reference pixel, and a point
  with none a failed row with no version; the refusals while busy, on a
  view-only bench and of a pixel outside the image, the node's before the
  point's; and *Commit* refused until endpoint B and then the distance are
  given, then appending a row, with its undo. (`image_detail/tests.rs`): the
  entry after the bench entries and before *Set Endpoint A Here*, offered
  with nothing focused and greyed while busy and on a view-only bench.
  (`viewer_3d/tests.rs`): the entry after the point's own, a click reporting
  the gesture, and greyed, asking for nothing, on a point with no reference
  pixel and on a busy node; the texel of the spot right-clicked recorded on
  the frame the menu opens, carried by the entry chosen after the pointer has
  moved, and the measurement started at the pixel that shows it, resolving
  back to the point at that texel under a display transform; the texel read
  where the right-click landed though the pointer moved on before the pick
  at the click's pixel came back; the point's
  centre taken past the square's edge, in the Dots style and at zero patch
  opacity, and the texel read at the patch size drawn.
  (`measurement_view/tests.rs`): the new item's unplaced block with its
  instructions, the empty header fields, the result's line and *Commit*'s
  refusal following B and the distance; and a pixel field wide enough for
  `117.847664`. (`mcp/tests/measurements.rs`):
  `create_bench_measurement` starting an item, focused with no panel raised,
  that the other tools finish and commit as a new row, and its refusal of a
  pixel outside the image.
- **The reference pixel and the spot** (`measurements/tests.rs`): the
  keypoint of the reference observation in its own image, which resolves alone
  to texel `(0, 0)` of the point's patch; and the three refusals, with no
  column at all counted as no reference observation. The texel a ray meets on
  a tilted square at any drawn size; none off the square, from behind, away
  from it, at infinity, under the grazing bar and on degenerate input; the
  spot's pixel resolving to its point at its texel, and the reference pixel
  where the texel's place is off the reference image, on a square facing away
  from its camera, and outside the square, with the three refusals kept.
- **`ui_basic`**: the panel opens from the Panels menu; a double-click on a
  Measurements row puts the measurement on the bench and raises Measurement
  View on the item, with its image drop-down and pixel fields, and after its
  row is deleted a click on *Commit* adds it back as a new row, on a measured
  copy of the seoul bull ground truth; an endpoint switched to another
  photograph through *Switch image…* and its popup, then committed into its
  row; *Set Endpoint B Here* chosen from a right-click on the photograph in
  Image Detail, moving endpoint B to that photograph at the pixel clicked; and
  a measurement made from Image Detail's menu, *New Measurement from Here*
  raising Measurement View on an item whose block for B says how to place it,
  *Set Endpoint B Here* on another photograph, a distance typed into its field
  and Enter, and *Commit* adding a third row.
- **Drawing** (`measurement_marks/tests.rs`): with the option off only the
  selected measurement, with it on every one, the selected one wider and the
  others dimmer; a hidden reconstruction none either way, and the focused item
  whatever the eye; each endpoint's place, finite through the display
  transform, at infinity or refused; the item in the bench's colour with its
  one endpoint, its marks handles and the selected endpoint's larger; a tinted
  node's tint; the segment test, the indicator's place, the hit test taking
  marks before lines, and what a click on each target asks for.
  (`viewer_3d/measurements/tests.rs`, whole headless frames): the marks and the
  line where the camera projects the endpoints through a display transform; a
  refused endpoint and one at infinity drawing nothing; the hover text and
  cursor on a mark and on the line; a click on a mark, on the line and a
  double-click asking for their gestures and no pick (G7), while the same click
  off them picks; a drag from a mark orbiting nothing while one off it orbits;
  the indicator absent while the endpoints or only the line are in view, and
  pointing right, left, down, up and, for a midpoint behind the eye, the way to
  turn; a click on it framing both endpoints inside the view with the
  orientation kept; the option and a hidden node; a line with one end behind
  the eye clipped, the end in front keeping its mark; the instances the frame
  builds for the pass: the halos of the line and of each ring, then the line
  and each ring, then each dot, at the endpoints themselves, each ring and dot
  one instance with its radius in points whatever the distance and the
  selected ring 1.6 times larger, the colours and widths in points, every piece
  held behind the scene and pulled toward the eye; every halo before every
  stroke and every dot last across the measurements of *Show all*, with its
  emphasis and dimming; every line and its halo inset at each end by the
  radius and half the stroke of the ring there, the selected endpoint's larger
  ring and the emphasised stroke included, and no other piece inset; the
  item's violet; none for a refused endpoint or one
  at infinity, and a line with one end behind the eye handed to the pass
  whole; the pull a frame hands the pass equal to `toward_eye_fraction` of its
  camera and panel, which is the same number of points at any field of view
  and panel height, with its pixels per point; the instances built once and
  not again for a camera move, a field of view or a panel size, and once more
  for each change of their key (the item focused, a step on it, an undo,
  another measurement or an endpoint selected, *Show all* toggled, the node
  hidden and shown, its display transform changed, and its tint); and a thousand measurements with *Show all*
  on orbited with nothing built or handed to the upload.
  (`scene_renderer/bench_track/tests.rs`): the figure's segments fading while
  the measurements' instances hold at `BEHIND_OPACITY`, their upload, the
  camera block carrying the projection, the pixels per point and the pull,
  and the pass with both drawn validated on a headless device, a segment, a
  ring and a disc into the measurements' own layer; the layer's one format and
  its clear, and a translucent colour composited as egui painted it on an sRGB
  or an encoded surface.
  (`image_detail/measurements/tests.rs`):
  the marks at the endpoint pixels, a line only where both are in the image,
  the refused endpoint's warning colour; the hover, a click selecting no
  feature's point under the mark while the same click with nothing drawn does,
  a double-click and a click on the line; a handle drag asking once for the
  pixel under the release, which the step makes one version, without a pan;
  Escape abandoning a held handle with no step; a measurement with
  mismatched columns and a pixel that is not a number drawn without a panic,
  that pixel with no mark and no line;
  no handle on a view-only bench; the indicator pointing to the endpoints
  outside the view and a click centring them at the same zoom.
  (`mcp/tests/display.rs`): each view's `show_all_measurements` over its own
  tool, recording its own row. (`measurement_view/tests.rs`): the panel and the
  drawing reading one cached resolution of the focused item, and a step on it
  one more.
- **`ui_basic`, drawing**: on the measured copy, endpoint A of the focused item
  dragged in Image Detail moves to the pixel under the release in one version,
  and Image Detail's *Show all measurements* checkbox turns its option on.
- **The script** (`tests/test_add_artificial_measurements.py`): every case
  with its known scale read back through the resolver, the copy without a
  unit, the recorded workspace, and a path under `test-data/` refused with
  nothing written.

## Non-goals

An endpoint's 3D position, point and texel are never set directly: they always
follow from its image and pixel, so the 3D view draws no handle.
