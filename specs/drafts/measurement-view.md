# Measurement View

**Status:** Draft

Amends [gui/measurement-view.md](../gui/measurement-view.md), which describes
what is built: the selected measurement, the per-version readings, the Scene
tree's Measurements group with its row gestures, the Measurement View panel
with its header, the distance body and the endpoint gestures, `get_scene`'s
measurement counts, `list_measurements`, `get_measurement`,
`select_measurement`, and the artificial-measurements script (step 1 of
§ "Steps"). This draft keeps the rest: editing and creating measurements on the
bench, and drawing measurements in the 3D view and Image Detail. It also amends
[gui/bench.md](../gui/bench.md), [core/bench/bench.md](../core/bench/bench.md),
[gui/scene-graph.md](../gui/scene-graph.md) and G2 of
[gui/interaction-rules.md](../gui/interaction-rules.md), as § "What changes in
the standing specs" lists.

Decided by the maintainer:

- **A measurement is edited on the bench, as a track is.** A person puts a
  measurement on the bench, works on it there, and commits it back. Every step
  on it is a bench version that leaves the reconstruction alone, and the commit
  is the one step that writes the measurement into the file. A new measurement
  is a bench item from its first endpoint on, with no state outside the
  history. Renaming and setting the stated distance go through the bench as
  well, with no direct shortcut. Deleting a measurement stays a direct edit of
  the reconstruction, as deleting a point does.
- **A double-click on a Measurements row puts the measurement on the bench**,
  as a double-click on a point puts its track there (G2), and raises
  Measurement View on it.
- **The bench of a `sift_files` reconstruction stays view-only** for
  measurements too, with the same sentence: a measurement can be put on it and
  viewed, and every step that edits it is refused.
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
[core/bench/bench.md](../core/bench/bench.md) (the bench a measurement is put
on), [gui/track-view.md](../gui/track-view.md) (the panel whose *Edit* box and
toolbar Measurement View follows), [gui/edits/commit-track.md](../gui/edits/commit-track.md)
(the commit this one mirrors), [gui/document-model.md](../gui/document-model.md)
and [gui/edit-history.md](../gui/edit-history.md) (the versions a step pushes),
[gui/mcp-server.md](../gui/mcp-server.md) (the tools).

## Why the bench

The bench is the viewer's place for work that is not settled yet
([gui/bench.md](../gui/bench.md)): a track is taken out of the reconstruction,
argued over and committed back, and the history holds every step of it
alongside the reconstruction, so one Undo walks both. Placing a measurement's
endpoints well is the same kind of work. A person switches an endpoint between
photographs, nudges a pixel, compares the result with the other measurements,
and only then wants it written. Done as direct edits, each of those steps would
rewrite the file's measurements and mark the node dirty, a half-made
measurement would need a state of its own outside the history, and dragging an
endpoint would have no editable layer to live in. On the bench they are bench
steps, a new measurement is an item like a cluster started from a pixel, and
the bench's layer in Image Detail and the 3D view is where its handles go.

## The measurement item

Core's bench gains a second kind of item
([core/bench/bench.md](../core/bench/bench.md), which already says a second
kind joins by adding a variant):

```rust
pub enum BenchItem {
    Track(Arc<EditableTrack>),
    Measurement(Arc<EditableMeasurement>),
}

pub enum ItemKind {
    Track,
    Measurement,
}

/// A measurement being worked on: where it came from, and what it will write.
pub struct EditableMeasurement {
    /// The stored measurement it was taken from, or `None` for a new one. A
    /// commit replaces the row the origin resolves to, and appends a row where
    /// it resolves to none.
    pub origin: Option<MeasurementOrigin>,
    /// The measurement's own label, written into the file on commit. It may be
    /// empty and need not be unique, so it is not the item's bench label.
    pub label: String,
    pub body: EditableMeasurementBody,
}

pub enum EditableMeasurementBody {
    Distance(EditableDistance),
}

pub struct EditableDistance {
    /// Endpoints A and B. A new measurement starts with A alone.
    pub endpoints: [Option<MeasurementEndpoint>; 2],
    /// The stated length and its unit, once given.
    pub distance: Option<(f64, LengthUnit)>,
}

pub struct MeasurementEndpoint {
    pub image: u32,
    pub pixel: [f64; 2],
}

pub struct MeasurementOrigin {
    /// The version the measurement was read out of, as the caller numbers
    /// versions.
    pub version: u64,
    pub kind: MeasurementKind,
    pub index: u32,
}
```

`MeasurementKind`, today the viewer's (`crates/sfm-explorer/src/measurements.rs`),
moves into core with the item, and the viewer uses core's.

The item holds only what the file stores: the endpoints' images and pixels,
the stated distance and unit, and the label. Each endpoint's point, texel and
3D position are derived from these by the resolver, at the cursor, every time
(§ "Live readings"). An edit of an item is a pure function from one item to the
next, as every track step is, and each replaces the item's `Arc` under the same
`ItemId`.

**Labels.** A measurement put on the bench from a stored one is labelled by its
kind and index at the time it was put on, `distance #2`. A new one is labelled
by where it started, `distance from IMG_0042@142,198`. A collision takes ` (2)`
as every label does ([core/bench/bench.md](../core/bench/bench.md) § "Labels"),
and the label stays when rows are renumbered later, as a label stays when it no
longer describes its item.

**Putting a measurement on the bench twice focuses the item it already made**,
as for a point: the test is the origin, followed to the cursor, and that case
pushes no version.

**Following the origin.** A measurement has no portable id, so the origin is
followed by row. Two edits renumber the rows of a kind: deleting an image drops
the measurements with an endpoint in it, and deleting a measurement drops one
row. Each such version carries a measurement map, the measurements' analogue of
its point map, which the viewer follows an origin through as it follows a
track's origin and the selected measurement (`AppState::follow_measurement_rows`
already does the latter for an image delete). An origin whose row was dropped
resolves to none, and a commit then appends.

**Deleting an image** also reaches the bench's measurements, in the same
version, through core's `Bench::delete_image`
([gui/bench.md](../gui/bench.md) § "One history for the pair"): an endpoint in
the deleted image is cleared, an endpoint in a later image moves down by one,
and an item with neither endpoint left is discarded, as a track with no
observation left is.

## Putting a measurement on the bench

| Gesture | What happens |
|---|---|
| Double-click a Measurements row in the Scene tree (G2) | The measurement is selected, put on the bench and focused, in one version (`Put distance #2 on the bench`), or its item focused where it is on the bench already; Measurement View is raised beside the tree (G9), and the 3D view frames the measurement as the row's double-click does today. |
| Tick *Edit* in Measurement View with a measurement selected | The same, without the raise and the framing. |
| *Edit on Bench* in a Measurements row's menu (G3) | The same as the double-click. |
| *New Measurement from Here* (§ "Creating a measurement") | A new item with endpoint A alone, focused. |

On a `sift_files` reconstruction these put the measurement on the bench and
focus it as on any other, and every step that edits it is refused with the
view-only sentence ([gui/bench.md](../gui/bench.md) § "A view-only bench").

**One focused item for the viewer.** A measurement item is focused exactly as
a track is ([gui/bench.md](../gui/bench.md) § "The focused item"): focusing it
unfocuses any track, focusing pushes no version, and the recent items strip
holds measurement items beside tracks. While a measurement item is focused,
the selected measurement is its origin followed to the cursor, or none for a
new one, and the selected point is left alone.

## Measurement View with an item focused

Measurement View gains Track View's **Edit** box, with the same meaning: it is
ticked exactly when the focused item is a measurement on the selected node's
bench ([gui/track-view.md](../gui/track-view.md) § "The Edit checkbox"), and its
transitions are Track View's with "measurement" for "point". Clearing it
unfocuses the item and leaves it on the bench. Track View and Measurement View
each tick their box only for their own kind of item, so with a measurement
focused Track View's box is clear, and the reverse.

- **Edit clear**: the panel is what is built, the selected stored measurement,
  read-only.
- **Edit ticked**: the panel shows the focused item, with a toolbar and the
  editing controls:
  - the toolbar: *Commit*, *Discard* and *Rename*, as Track View's;
  - the header's label and stated distance as fields, each a step on Enter or
    when the field loses focus, and Escape cancelling (§ "Steps on an item");
  - in each endpoint block, *Switch image…*, and the image and pixel as fields
    (§ "Switching an endpoint's image", § "Setting an endpoint by hand"), with
    the 3D position beside them read-only;
  - an endpoint not yet placed is a block saying how to place it: "Right-click
    a pixel in Image Detail, or a point in the 3D view, and choose Set Endpoint
    B Here."

The endpoint gestures are those built (G1, G2, G3, G5, G6), on the item's
endpoints. A right-click on an endpoint block opens a menu with *Switch
Image…* and selects nothing.

### Live readings

The focused item's readings are computed from the item, not read from the
stored measurement: each placed endpoint is resolved against the
reconstruction at the cursor with `resolve_measurement_endpoint`, and the
result block and the median line are computed from those. They are kept per
item `Arc` and document serial, so a step on the item, or an edit of the
reconstruction under it, reads them again on the next frame. The median line
compares the item with the stored measurements other than its origin, so a
person sees whether the change brings it into line before committing it.

## Steps on an item

Each is a bench step: one version through `History::push_bench`, listed in the
Edit History and undone by `Ctrl+Z`, with an Action Log row, refused while a
background task holds the node and on a view-only bench. None changes the
reconstruction or marks the node dirty.

| Step | Label in the Edit History |
|---|---|
| Switch an endpoint's image | `Moved endpoint A of distance #2 from IMG_0004.jpg to IMG_0011.jpg` |
| Set an endpoint's image and pixel | `Moved endpoint A of distance #2 to IMG_0004.jpg (412.3, 230.5)` |
| Place an endpoint not yet placed | `Set endpoint B of distance from IMG_0042@142,198 to IMG_0050.jpg (88.0, 301.5)` |
| Set the measurement's label | `Set the label of distance #2 to "paw width"` |
| Set the stated distance | `Set the stated distance of distance #2 to 4.2 cm` |
| Rename the item | as Track View's *Rename* |
| Discard | as Track View's *Discard* |

The stated distance is a number and a unit in one field, parsed as the CLI's
YAML parses a distance (`12in`, `30.48cm`), with the reconstruction's
`world_space_unit` as the unit of a bare number, or `m` where it has none. A
value that is not finite and positive is refused with the format's sentence. A
pixel outside its image (`[0, width) × [0, height)` of that image's camera) is
refused with the format's sentence.

## The commit

*Commit*, in Measurement View's toolbar, writes the focused item into the
reconstruction. As for a track ([gui/edits/commit-track.md](../gui/edits/commit-track.md)),
it is one ordinary edit: one version, one label, one Action Log entry of kind
`Edit`, undoable like any other, and the node is marked dirty.

The write is a pure core function beside the track commit:

```rust
/// Write `item` into `base`: replace the row `origin_row` names, or append a row
/// where it names none.
pub fn commit_measurement(
    base: &EditedReconstruction,
    item: &EditableMeasurement,
    origin_row: Option<u32>,
) -> Result<MeasurementCommit, MeasurementCommitRefusal>;

pub struct MeasurementCommit {
    pub value: EditedReconstruction,
    /// The row written.
    pub index: u32,
    /// `false` where the row already held exactly what the item holds.
    pub wrote: bool,
}
```

The caller resolves the origin to a row at the cursor, as it does a track's.
The new value replaces the measurements and nothing else: its base shares the
points, tracks and patch bitmaps with the old one rather than copying them, and
the overlay is carried across unchanged, since no point index moves. If
`SfmrReconstruction` has no way to replace its measurements while sharing those
columns, the first step adds one.

**The button is greyed with the core refusal as its hover text**, asked of the
item it would commit, so the button and the step cannot disagree. The
refusals: an endpoint not placed, no stated distance, an endpoint's image past
the image set, a pixel outside its image, and the viewer's own refusals while
a background task holds the node and on a view-only bench. An endpoint that the
resolver refuses does not stop a commit, since the pixel is the stored
measurement and the format stores it either way.

**After the commit** the item's origin is re-seated on the row written, which is
selected, and the item stays on the bench and focused, as a committed track
does. Its labels: `Committed distance #2 in dino` for a replaced row, `Added
distance #4 in dino` for an appended one, and a no-effect row where nothing was
written.

## Deleting a measurement

*Delete* in a Measurements row's menu (G3) deletes the stored measurement as a
direct edit of the reconstruction, one version (`Deleted distance #2 in dino`),
as deleting a point is. It renumbers the rows after it, with a measurement map
the selection and every measurement item's origin follow (§ "Following the
origin"). An item whose origin was the deleted row stays on the bench with no
origin, so committing it adds the measurement back.

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

A click on a row is the step that moves the endpoint to that image and pixel.
The point the endpoint resolves to, and the texel, are those of the row's
point, unless the mark above says otherwise.

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
derived from them. So both can be set directly, each a step on the focused
item:

- **In the endpoint block.** The image name is a drop-down of the
  reconstruction's images, and the pixel is two number fields. A change is one
  step that sets both, on Enter or when the field loses focus.
- **In Image Detail.** With a measurement item focused, Image Detail's context
  menu gains `Set Endpoint A Here` and `Set Endpoint B Here`, which set that
  endpoint of the focused item to the image shown and the pixel clicked. The
  context menu is the viewer's way to name a pixel
  ([multi-panel-image-browser.md](../gui/multi-panel-image-browser.md) §
  "Image Detail: the context menu"), so this is where a person zoomed in on a
  photograph places an endpoint precisely.
- **In the 3D view.** With a measurement item focused, a right-click on a
  point, which selects it (G3), offers the same two entries, which set the
  endpoint to the point's **reference pixel** (§ "A point's reference pixel").

The entries are greyed on a node other than the focused item's, with hover
text naming the item's node. None of them offers the 3D position, the point or
the texel: they follow from the image and pixel through the resolver every
time, and a setting that disagreed with them could not be kept.

## Creating a measurement

**Starting it.** Two menus offer `New Measurement from Here`:

- **Image Detail's context menu**, at the clicked pixel of the image shown.
  Any pixel inside the image will do; the endpoint does not have to resolve,
  since the pixel is the stored measurement.
- **The 3D view's point menu.** A right-click on a point selects it (G3) and
  offers the entry, which starts the measurement at the point's reference
  pixel.

Either puts a new measurement item on the node's bench with endpoint A at that
pixel and nothing else, focuses it, and raises Measurement View beside the
panel the gesture was made in (G9), in one version (`Put distance from
IMG_0042@142,198 on the bench`). Endpoint B is then placed with `Set Endpoint B
Here` from the same menus, and the stated distance and label are typed into
the panel. Each is a step, so a mistake is undone like any other, and a
half-made measurement waits on the bench, unfocused, until it is finished or
discarded. *Commit* is greyed, with the refusal naming what is missing, until
both endpoints and the distance are given, and appends the measurement as a
new row.

### A point's reference pixel

A point in the 3D view is not a pixel, and a measurement's endpoint must be
one. The pixel is taken from the point's **reference observation**
(`tracks/reference_observations`): the image of that observation, and its
keypoint. The keypoint is where the reference observation's square is centred,
so the endpoint resolves to texel `(0, 0)` of the point's own patch, which is
the point itself, through the observation its bitmap was rendered from.

The entries are greyed, with hover text saying why, where the point has no
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

The bench tools ([gui/bench.md](../gui/bench.md) § "The wire") take the
measurement item as they take a track, and new tools name the measurement
steps, each refused in the panel's words:

- `put_measurement_on_bench(reconstruction_label, measurement_kind,
  measurement_index)`, the double-click without the raise and the framing.
- `create_bench_measurement(reconstruction_label, camera_image, pixel)` and
  `get_point_reference_pixel(reconstruction_label, point)`, so an agent can
  start from a point as the 3D view's menu does.
- `get_measurement_endpoint_views(reconstruction_label, item, endpoint)`: the
  offers of § "Switching an endpoint's image".
- `set_bench_measurement_endpoint(reconstruction_label, item, endpoint,
  camera_image, pixel)`, taking any pixel inside the image, not only an
  offered one, as the panel's fields do. No tool sets an endpoint's 3D
  position.
- `set_bench_measurement_label` and `set_bench_measurement_distance`.
- `commit_bench_measurement(reconstruction_label, item)`.
- `delete_measurement(reconstruction_label, measurement_kind,
  measurement_index)`.
- `get_measurement` names an item as well as a stored row, and gives the
  item's live readings.

Focusing, unfocusing, renaming and discarding go through the existing bench
tools, which name an item by its label whatever its kind.

## Real measurements

`dino_dog_toy`'s four tape-measured distances, entered by hand once the panel
can place measurements, are the first real ones. They are not checked in yet.
When they are, the file keeps its name: the `_vNN` suffix versions the
reconstruction's cameras, poses, points, tracks and patch frames, and adding
measurements changes none of those.

## What changes in the standing specs

- [core/bench/bench.md](../core/bench/bench.md): the second item kind,
  `EditableMeasurement`, its labels, and `Bench::delete_image` on it; a new
  `core/bench/editable-measurement.md` for the item's steps and its commit.
- [gui/bench.md](../gui/bench.md): measurement items in the focused item, the
  recent items strip, the view-only refusal, image deletion, the wire, and a
  *Bench Measurements* group beside *Bench Points* and *Bench Clusters*, whose
  rows behave as theirs do.
- [gui/measurement-view.md](../gui/measurement-view.md): the *Edit* box, the
  edited mode, the live readings, the commit, and the row's double-click.
- [gui/scene-graph.md](../gui/scene-graph.md): the Measurements row's
  double-click and menu, and the *Bench Measurements* group.
- [gui/interaction-rules.md](../gui/interaction-rules.md): G2's table gains a
  row, "A stored measurement (a Measurements row): put it on the bench when it
  is not there already, which pushes a version, raise Measurement View on it,
  and frame its endpoints in the 3D view", and the measurement's endpoints
  leave the "set of things" row.
- [gui/track-view.md](../gui/track-view.md): its *Edit* box ticks only for a
  track item.

## Steps

Each step is one PR through `implement-change`, with its MCP tools, its
`ui_basic` coverage and screenshots.

1. **Read-only.** Built; see [gui/measurement-view.md](../gui/measurement-view.md).
2. **The item on the bench.** `EditableMeasurement` and `BenchItem::Measurement`
   in core with its labels and `Bench::delete_image`; the measurement map and
   origin following; putting a measurement on the bench (the row's
   double-click and menu entry, the *Edit* box); the focused measurement item,
   the recent items strip and the *Bench Measurements* group; Measurement
   View's edited mode with the live readings, *Discard* and *Rename*;
   `commit_measurement` with the share-preserving measurement replacement and
   the *Commit* button; *Delete*; and their tools.
3. **Editing the label and the stated distance.** The header's fields and
   their steps and tools.
4. **Switching an endpoint's image.** `endpoint_views` in core with its
   binding, the switch popup, `get_measurement_endpoint_views` and
   `set_bench_measurement_endpoint`.
5. **Setting an endpoint by hand.** The endpoint block's image and pixel
   fields, `Set Endpoint A Here` and `Set Endpoint B Here` in Image Detail's
   menu and the 3D view's point menu, the reference pixel and
   `get_point_reference_pixel`.
6. **Creating a measurement.** `New Measurement from Here` in both menus, an
   item with one endpoint, the unplaced endpoint's block, and
   `create_bench_measurement`.
7. **Drawing.** The selected measurement's and the focused item's marks and
   lines in the 3D view and Image Detail, their hover and click, the handles,
   the off-screen indicator and its click, and the *Show all measurements*
   option in each view, with `get_viewer_3d_display`, `set_viewer_3d_display`
   and the Image Detail display tools carrying it as `show_all_measurements`.

## Testing

- The item: putting a measurement on the bench copies its endpoints, distance,
  unit and label; putting it on twice focuses the same item with no version; a
  view-only bench refuses every step with the sentence and allows the put,
  focus, rename and discard.
- Origin following: deleting an earlier measurement, and deleting an image that
  drops an earlier measurement, move the origin; deleting the origin's row
  leaves the item with no origin, and its commit appends.
- Image deletion on the bench: an endpoint in the deleted image is cleared, one
  in a later image moves down, and an item with no endpoint left is discarded.
- The commit: replacing a row and appending one; the no-effect case; each
  refusal, and the button's greying matching it; the version shares the base's
  point columns (pointer equality); undo restores the row and keeps the item.
- Live readings: a step on the item changes its readings and not the stored
  measurement's; the median line leaves out the item's origin.
- The *Edit* box: ticking puts the selected measurement on the bench; Track
  View's box stays clear while a measurement item is focused, and the reverse.
- Switching: the offers of an endpoint on a known texel land at the pixels the
  other observations' squares put that texel at; a switch keeps the resolved
  point and texel; the "resolves elsewhere" mark appears where another square
  holds the offered pixel.
- Each step's label, undo, and refusal.
- Creating: starting from a pixel and from a point; the entry greyed, with its
  sentence, on a point with no patch, no reference observation, or at
  infinity; a point's reference pixel resolving to texel `(0, 0)` of its
  patch where no other square holds it; *Commit* greyed until B and the
  distance are given, then appending one row.
- Drawing: with the option off only the selected measurement and the focused
  item are drawn; with it on, every measurement of a visible reconstruction; a
  hidden reconstruction draws none either way; a handle drag is one version; a
  view-only bench draws no handles.
- The off-screen indicator: absent while any part of the measurement is in
  view; pointing the right way for a midpoint to each side of the view and
  behind the camera; a click framing the measurement so both endpoints project
  inside the view.
- `ui_basic`: a measurement put on the bench by a double-click, an endpoint
  switched, and the commit; a measurement made with the two menus and the
  panel's fields.
