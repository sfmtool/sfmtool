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

The panel reads measurements and does not edit them. Switching an endpoint to
another photograph, setting one by hand, renaming, setting the stated distance,
deleting and creating a measurement, and drawing measurements in the 3D view
and Image Detail are proposed in
[drafts/measurement-view.md](../drafts/measurement-view.md).

Related specs: [track-view.md](track-view.md) (the panel this one is modelled
on, and the crops it reuses), [scene-graph.md](scene-graph.md) (the tree the
rows go in), [panel-layout.md](panel-layout.md) (the tab),
[mcp-server.md](mcp-server.md) (the tools), and
[formats/sfmr-file-format.md](../formats/sfmr-file-format.md) § 10
"Measurements" (what the file stores).

## Interface

The code is in
[measurements.rs](../../crates/sfm-explorer/src/measurements.rs) (the
reference, the readings and their text),
[measurement_view.rs](../../crates/sfm-explorer/src/measurement_view.rs) (the
panel),
[scene_graph/measurements.rs](../../crates/sfm-explorer/src/scene_graph/measurements.rs)
(the Scene tree group) and
[mcp/measurements.rs](../../crates/sfm-explorer/src/mcp/measurements.rs) (the
tools).

```rust
pub(crate) enum MeasurementKind { Distance }

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
}

/// The readings of the version a node shows, resolved once per version.
pub(crate) fn readings(node: &SceneNode) -> Arc<MeasurementReadings>;
pub(crate) fn distance_result(
    row: &DistanceRow,
    resolved: &ResolvedDistanceMeasurement,
    world_space_unit: Option<&str>,
) -> DistanceResult;
```

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

**Selecting a measurement selects nothing else.** It selects the
measurement's reconstruction, which keeps the other selections inside that
reconstruction as selecting the reconstruction does, and does not select either
endpoint's image or point, so the 3D view and Image Detail stay where they are.
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
does; the selection moves with its measurement, or is cleared where its
measurement was dropped (`AppState::follow_measurement_rows`). Every frame,
before the panels draw, `AppState::prune_measurement_selection` clears a
selection whose row the version on screen does not hold, so an undo or a redo
keeps the selection where the version arrived at has that row and clears it
where it does not. A cleared selection stays cleared when a later step brings
the row back.

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

- **Click** (G1): select the measurement and nothing else, as a click on a
  Bench row does.
- **Double-click** (G2, a set of things): select it, raise Measurement View
  beside the tree through `show_panel_beside` (G9), and frame the measurement
  in the 3D view: the view keeps its orientation and moves so that its finite
  endpoints are inside the middle half of the viewport, the orbit target on
  their centroid (`Viewer3D::frame_keeping_orientation`). A measurement with
  one finite endpoint centres it at the orbit distance the view had, and one
  with none leaves the view where it is. The raise is a layout operation, so
  the double-click is kept for `SceneGraphPanel::take_measurement_open` and
  applied by `app.rs` once the dock is back in the state
  (`app::apply_measurement_open`).
- **Right-click** (G3): opens no menu, and selects nothing.

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

Two endpoint blocks side by side, **A** and **B** (endpoint 0 and 1 of the
format), then the result block. Each block shows:

- **The photograph's name** and the endpoint's pixel, `(412.3, 230.5)`.
- **The crop.** The photograph around the endpoint, 128 points square. Where
  the endpoint resolves, the crop is cut around the chosen candidate's square
  in that image, as Track View's crop is cut around a row's square
  (`patch_pictures::crop::patch_outline` and `region`), with the square's
  outline over it and a cross at the endpoint's pixel. Where it does not, the
  crop is a square of the photograph `UNRESOLVED_CROP_PX` (64) photograph
  pixels across, centred on the pixel, with the cross alone. While the
  photograph is decoding the cell says so; where it cannot be read, the cell
  says why in its hover text.
- **The crop's hover view.** The photograph's full name, then the same crop in
  context, three times its width and height at 288 points
  (`marks::CONTEXT_FACTOR`, `CONTEXT_HOVER_SIDE`), with the outline and the
  cross, then bullets giving the pixel, the crop's size in photograph pixels and
  the square's two axes in photograph pixels.
- **The patch.** The chosen point's stored patch bitmap, 64 points square,
  with a cross at the endpoint's texel `(s, t)` (`s` across, `t` up, since a
  bitmap's rows step along `−v`), so the person sees which place on the
  surface the endpoint names. Absent where the endpoint does not resolve or the
  point stores no bitmap.
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
patch are one target.

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
- **Right-click** (G3): opens no menu, and selects nothing.

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
- `screenshot` accepts `measurement_view` as a `panel_name`.

## Artificial measurements

No checked-in reconstruction has measurements, and the panel cannot be judged
or tested without them. They are made, never checked in, in three ways.

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
  opened, which the panel keeps once.
- **The double-click's effect** (`app/tests.rs`): it selects the measurement,
  opens the panel, and frames the finite endpoints without turning, both inside
  the middle half; one finite endpoint keeps the orbit distance; a row past
  the rows moves nothing. The framing itself (`viewer_3d/framing/tests.rs`):
  every point inside the middle fraction, one on its edge, the orientation
  kept.
- **The panel** (`measurement_view/tests.rs`, whole headless egui frames): the
  empty state and its links; the header, both blocks and the result's numbers;
  the result block's scene-unit, at-infinity and refused cases; a refused
  endpoint's sentence; a block's click, double-click, hover and right-click.
  (`dock/tests.rs`): a double-click's camera view is the one
  `look_through_toward_feature` gives the endpoint's pixel.
- **The tools** (`mcp/tests/measurements.rs`): `get_scene`'s counts,
  `list_measurements`' rows and results, `get_measurement`'s resolutions,
  refusal and median line, an index past the rows refused with the count, and
  `select_measurement` with and without an endpoint.
- **`ui_basic`**: the panel opens from the Panels menu, and from a
  double-click on a Measurements row, on a measured copy of the seoul bull
  ground truth.
- **The script** (`tests/test_add_artificial_measurements.py`): every case
  with its known scale read back through the resolver, the copy without a
  unit, the recorded workspace, and a path under `test-data/` refused with
  nothing written.

## Non-goals

The panel does not edit a measurement or create one, and the 3D view and Image
Detail do not draw measurements; those are proposed in
[drafts/measurement-view.md](../drafts/measurement-view.md). An endpoint's 3D
position is never set directly: it always follows from its image and pixel.
