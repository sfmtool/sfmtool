# Track View

A 3D point in a reconstruction is only as good as the photographs that saw it.
Each point carries a **track**, the list of feature observations, one per image,
that were triangulated into it, and judging a suspect point means reading that
list sighting by sighting: where each one sits, how far it lies from where the
point projects, and whether the patch of surface it shows looks like the
others. Sometimes the answer is that the track is wrong, and then it has to be
worked on: sightings tried, measured, turned out, and the result written back.
Track View is the one panel of the SfM Explorer where both happen. With its
**Edit** box clear it shows the selected point's committed track and changes
nothing. With the box ticked it shows the one track the viewer is editing, which
is held beside the reconstruction on the **bench** until it is committed,
together with the controls that measure it, fit it and commit it.

The panel has an explicit mode so that it always says which of the two things is
on screen, and it does not list the bench: the Scene tree already lists each
node's bench in two groups that say which stage each item is at, and a
double-click there is the way into editing an item.

Related specs: [`bench.md`](bench.md) (the bench, the versions its steps push
and the Scene tree groups), [`edits/commit-track.md`](edits/commit-track.md)
(the Commit button's edit), [`scene-graph.md`](scene-graph.md) (the Bench rows'
gestures), [`multi-panel-image-browser.md`](multi-panel-image-browser.md) (the
Image Detail panel, which carries the gestures that name a pixel and draws the
active track as its bench layer),
[`viewer-3d-bench-layer.md`](viewer-3d-bench-layer.md) (the same track in the 3D
viewer), [`../core/bench/bench.md`](../core/bench/bench.md) and
[`../core/bench/editable-track.md`](../core/bench/editable-track.md) (the value
edit mode shows and every step it calls), [`panel-layout.md`](panel-layout.md)
(its tab and its home), [`goto-point.md`](goto-point.md) (the dialog both modes
reach), [`background-tasks.md`](background-tasks.md) (where Evaluate, Fit, the
stage change, both searches and the index build run), and
[`sift-index.md`](sift-index.md) (the `.kdf` a search queries).

---

## The interface

The panel is [track_view/](../../crates/sfm-explorer/src/track_view/):
[mod.rs](../../crates/sfm-explorer/src/track_view/mod.rs) holds the checkbox,
the dispatch on it and the selection notice, and the two bodies it is made of
are its children. [view/](../../crates/sfm-explorer/src/track_view/view/) is
view mode, split into `prepare` (the per-observation data, built when the
selection changes), `header`, `table` and `patch`; the numbers it displays come
from [metrics/](../../crates/sfm-explorer/src/metrics), at the crate root,
because the Image Detail overlay and the MCP surface read the same ones.
[edit/](../../crates/sfm-explorer/src/track_view/edit/) is edit mode: `mod.rs`
the header, the toolbar and the sliders, `table.rs` the observation table and
`tile.rs` the tile each row draws. The bench steps it reports are `AppState`
methods in [bench.rs](../../crates/sfm-explorer/src/bench.rs), and the dock
applies them in [dock.rs](../../crates/sfm-explorer/src/dock.rs).

```rust
pub(crate) struct TrackView {
    view: PointTrackView, // view mode's state
    edit: TrackEdit,      // edit mode's state
}

impl TrackView {
    pub(crate) fn new() -> Self;
    pub(crate) fn show(
        &mut self,
        ui: &mut egui::Ui,
        state: &AppState,
        gesture_events: &[GestureEvent],
        scroll_input: &ScrollInput,
    ) -> TrackViewResponse;
    pub(crate) fn forget_recon(&mut self, id: ReconId);
    /// Select one edit-mode row from outside the panel: what a click on a mark
    /// of either bench layer reports through.
    pub(crate) fn select_row(&mut self, id: ReconId, label: &str, observation: usize);
    /// The one edit-mode row selected on `label`'s track, when exactly one is.
    pub(crate) fn selected_row(&self, id: ReconId, label: &str) -> Option<usize>;
    /// Whether edit mode's *Lock* box is ticked: what the dock hands Image
    /// Detail, whose track-stage dot slides the patch when it is and moves one
    /// sighting's keypoint when it is not.
    pub(crate) fn lock(&self) -> bool;
}

pub(crate) struct TrackViewResponse {
    /// The box ticked (`true`) or cleared (`false`), or the notice's *View*.
    pub set_edit: Option<bool>,
    /// The selection notice's *Edit it*.
    pub edit_selected_point: bool,
    /// What view mode reported, on a frame it was drawn.
    pub view: Option<PointTrackViewResponse>,
    /// What edit mode reported, on a frame it was drawn.
    pub edit: Option<TrackEditResponse>,
}

pub struct PointTrackViewResponse {
    pub select_image: Option<usize>,
    pub reveal_feature: Option<[f32; 2]>,
    pub request_camera_view: Option<usize>,
    pub hovered_image: Option<usize>,
    pub has_pointer: bool,
    pub request_goto_point: bool,
}

pub struct TrackEditResponse {
    pub discard: Option<String>,
    pub rename: Option<(String, String)>,
    pub evaluate: Option<f64>,       // the search radius the control stands at
    pub fit: Option<f64>,            // the same, for the reading a fit ends with
    pub set_stage: Option<StageKind>,
    pub apply_thresholds: Option<Thresholds>,
    pub split: Option<Vec<usize>>,
    pub duplicate: bool,
    pub commit: bool,
    pub build_sift_index: bool,             // a row's Build/Rebuild SIFT Index to Search
    pub search_descriptors: Option<usize>,  // Find matches by SIFT query
    pub search_geometry: Option<usize>,     // Find matches by geometry; track stage only
    pub set_verdict: Option<(usize, Verdict)>,
    pub select_image: Option<usize>,
    pub request_camera_view: Option<usize>,
    pub reveal_feature: Option<[f32; 2]>,
    pub hovered_image: Option<usize>,
    pub has_pointer: bool,
}

impl AppState {
    /// What the Edit box asks: `true` puts the selected point on the bench
    /// (activating the item already from it), `false` deactivates.
    pub(crate) fn set_editing(&mut self, id: ReconId, on: bool) -> Result<(), String>;
    /// Leave every item on `id`'s bench and make none active. One version; no
    /// effect, and no version, with nothing active.
    pub(crate) fn deactivate_bench_item(&mut self, id: ReconId) -> Result<(), String>;
    /// A Scene tree double-click on a Bench row: select the node, activate the
    /// item, raise Track View.
    pub(crate) fn edit_bench_item_at(&mut self, id: ReconId, position: usize);
    /// Image Detail's *Start cluster on the bench here*, raising Track View.
    pub(crate) fn start_cluster_here(&mut self, image: ImageRef, pixel: [f32; 2]);
}
```

A frame in the dock is one call and three applications:

```rust
let response = track_view.show(ui, state, gesture_events, scroll_input);
if let Some(on) = response.set_edit.or(response.edit_selected_point.then_some(true)) {
    state.set_editing(id, on)?;         // one bench step, refused in its own words
}
// then the view or the edit response, whichever was drawn
```

### Why it is shaped this way

**Two bodies behind one tab, not one body.** The two modes' state is disjoint.
View mode caches thumbnails and patch tiles per image of a committed point;
edit mode caches tiles per observation of a bench track keyed on its `Arc`, and
the slider seeding, the painting and the commit refusal. Keeping each as the
struct it is means neither cache learns about the other, and each body's
headless tests read what that body drew. What the panel adds is the checkbox,
the dispatch on it and the notice.

**The mode is derived from the bench, so nothing about it is stored.** A panel
flag would have to be set by each of the places that can change the activation,
and an Undo would leave it disagreeing with the bench at the cursor. Reading the
active label each frame costs a map lookup.

**The panel decides nothing.** `show` takes `&AppState` and every gesture lands
in the response; the dock applies each through the `AppState` method that pushes
the version, as every other panel's response is applied, because the panel holds
the state immutably while it draws. `view` and `edit` are `Option`s because only
one body is drawn on a frame, and a response from the body that was not drawn
would be a set of defaults pretending to be a report.

**The response keeps both bodies' types.** The dock already knows how to apply
each; the merged response adds only the two gestures that are the panel's own.

**The file chooser is the dock's, not the panel's.** A search's build remedy
reports the gesture and nothing else. That keeps `show` a pure egui function a
headless frame can run, the same split the resection's `.matches` chooser
takes.

---

## Placement

A dock tab, `Tab::TrackView`, titled **Track View** and named `track_view` in
the layout file and on the wire. Its home is the top-right node, as the first
and active tab of that leaf with Camera Intrinsics behind it, so the stock grid
is

```
+--------+-------------------------+----------------------------------+
| Scene  |[3D Viewer][Image Detail]| [Track View][Camera Intrinsics]  |
+--------+-------------------------+----------------------------------+
|Backgr. |[Image Browser][Action Log][Edit History]                   |
+--------+------------------------------------------------------------+
```

and the group the layout's placement rules call home is
`[TrackView, IntrinsicsDetail]` ([`panel-layout.md`](panel-layout.md) § "Home
positions"). The narrow right-hand column suits the panel because it is a table
of rows about a selection rather than a picture of one, and reads beside the
picture rather than instead of it. It is a panel like any other: closeable,
ticked in the Panels menu, and saved in the layout file.

**A saved layout that names `point_track` or `track_edit` is refused whole.**
Those are unknown panel names like any other, and the refusal is the ordinary
one from [`panel-layout.md`](panel-layout.md) § "Validation", prefixed with the
path to the leaf and listing the nine names that exist: `unknown panel
"point_track"; the panels are scene, background_task, viewer_3d, image_browser,
image_detail, track_view, camera_intrinsics, action_log, edit_history`. The
reader carries no alias for either, and the at-most-once rule applies to
`track_view` as to every other name. Reset Layout is the way back, and saving the
layout afterwards writes a file the viewer reads.

A default-layout file naming either takes the path every refused startup load
takes (§ "The default layout file" in the panel-layout spec): nothing of the file
is applied, the viewer comes up on the stock grid with Track View in it, and the
failed `Layout` entry `Load layout from <path>: <reason>` goes on the viewport
status line, so the person sees why their layout did not come back.
`LAYOUT_VERSION` is `2`: the version tags the document's shape, and a panel name
is a value a leaf holds, not a key or a node kind.

---

## The Edit checkbox

The first row of the panel is a checkbox, **Edit**, and nothing else is drawn
above it in either mode.

**The box shows the state of the bench, not an independent setting.** It is
checked if and only if a track or cluster is active on the selected node's
bench (`crate::bench::active_track_label`). Every way the activation can change
therefore moves the box on the next frame with no code keeping the two in step:
a Scene tree double-click, a step that puts an item on the bench, a wire call,
and an Undo or Redo that walks back over an activation.

The bench holds **one active item at a time**, and a point track and a cluster
are not two kinds for that purpose. Both are editable tracks (`ItemKind::Track`)
at different stages, which is why the Scene tree draws them in two groups and
still marks one active row across both ([`bench.md`](bench.md) § "The Bench
groups in the Scene tree"). So a ticked box names exactly one item, whether it
sits under *Bench Points* or *Bench Clusters*, and activating a cluster while a
track is being edited leaves the track on the bench, no longer active.

### Transitions

| From | Gesture | What happens | Then shown |
|---|---|---|---|
| Viewing a selected point | Tick Edit | `put_point_on_bench(selected point)`: a new item, active, in one version; or, when an item on the bench already came from that point, that item is activated instead | Edit mode, on that item |
| Viewing, no point selected, nothing active | Tick Edit | Nothing: the box is greyed, its hover text naming the ways in | Unchanged |
| Editing an item | Clear Edit | `deactivate_bench_item`: the item stays on the bench and nothing is active, in one version | View mode, on the selection |
| Either | Double-click a Bench row in the Scene tree | That item becomes active, its node is selected, and the panel is raised | Edit mode, on that item |
| Either | *Edit on Bench* in the 3D viewport or Image Detail, or a double-click on a point or a feature | As the ticked box from that point, then the panel is raised | Edit mode |
| Either | *Start cluster on the bench here* in Image Detail | A cluster is put on the bench, active, and the panel is raised | Edit mode, on the cluster |
| Editing | *Duplicate*, *Split off N rows* | The new item is put on the bench and becomes active | Edit mode, on the new item |
| Editing | *Discard* | The item leaves the bench and nothing is active | View mode |
| Editing | *Commit* | The point is written and selected; the item stays on the bench and stays active | Edit mode, on the same item |
| Either | Undo or Redo over an activation, a deactivation, a put or a discard | The bench at the cursor says what is active | Whatever that bench says |

The box is greyed while a background task holds the node, with the node's own
busy sentence, because every transition above is a bench step and a bench step
is refused then. With no point selected and nothing active, its refusal is one
sentence naming the three ways in: *"Nothing to edit: select a point,
double-click an item in the Scene tree's Bench groups, or right-click a pixel in
Image Detail and choose "Start cluster on the bench here"."*, the last quoted
from that entry's own constant (`track_view::nothing_to_edit`).

### Why each transition is the one it is

**Clearing Edit only deactivates.** The box stands for "active for editing",
not for being on the bench, and a discard already has its own toolbar button and
Scene tree menu entry. An Edit-clear that discarded would make looking at the
committed point cost the person their verdicts and their fit.

**A deactivation is a version**, one version and one `Bench` row labelled
`Stopped editing IMG_0042@142,198; it stays on the bench`, as an activation is.
The activation is part of the bench value, and an Undo that brings back what was
being edited is useful. The price is that flipping Edit to compare a track with
its committed point pushes two versions per round trip.

**Ticking Edit on a point already on the bench reuses its item**: the item whose
origin, followed to the cursor, is that point is activated rather than a second
one put on, which is what `put_point_on_bench` does for every way in. Two items
for one point are two answers to one question, and with no list of items in the
panel a second one would be easy not to notice.

**Discarding the active item leaves nothing active**, so the panel returns to
view mode (`Bench::discard`, [`../core/bench/bench.md`](../core/bench/bench.md)).
Handing the activation to a neighbour would switch the panel to an item the
person did not ask for, and nothing on screen would say which one arrived.

**Commit leaves edit mode on**, the item on the bench and active and the written
point selected. A person commonly commits and keeps going, the written point is
one Edit-clear away, and clearing on commit would make Commit two bench steps.

**With no point selected and nothing active, the box is greyed** rather than
bringing back the last active item or the last item on the bench. "The last one"
is not recorded anywhere a person can see, and the Scene tree double-click names
the item exactly.

**Start cluster on the bench here raises Track View.** It turns edit mode on,
which makes it the same kind of gesture as *Edit on Bench*, and that one raises
because a track staged into a panel nobody can see is a gesture with no answer.

---

## The selection and the active item

The **selection** is the viewer's selected point, image and camera, which every
panel reads and which clicking in the 3D viewer or Image Detail moves. The
**active item** is the bench's, part of the version at the cursor. The two are
independent, and the box decides which one Track View shows: view mode shows the
selection, edit mode shows the active item.

**Editing is sticky.** Selecting another point while editing moves the
selection, and with it the 3D viewer's track rays, the Image Browser's borders
and Image Detail's highlighted feature, but it does not change what Track View
shows and pushes no version. The person may be clicking around the scene to see
what else the patch covers, and a panel that dropped the edit on every click
would make that impossible.

**The selection notice** is what says the two have parted. It is one line under
the box, drawn in edit mode exactly when the node has a live selected point that
is not the active item's origin followed to the cursor: *"Selected:
pt3d_a1b2c3d4_5120, not the track being edited."*, with *View*, which clears
Edit, and *Edit it*, which puts that point on the bench as the ticked box does.
Both buttons grey with the busy sentence while a task holds the node.

**The common case keeps them together without asking.** Every way into editing
from a point selects that point first, and a commit selects the point it wrote,
so for a track that came from a point the selection and the item's origin agree
until the person clicks somewhere else, and the notice is absent.

**Go to Point** moves the selection and raises Track View. In edit mode the panel
stays on the item and the notice names the point jumped to, with *View* one click
away. The dialog does not clear Edit on its own: it is a selection gesture, and a
dialog that pushes no version stays one that pushes none.

---

## What the panel shows

### No reconstruction selected

`No reconstruction loaded`, centred, and no checkbox.

### View mode

View mode reads the reconstruction and never the bench.

**No point selected**: `No point selected` above a **Go to Point...** button,
which is where a person with an ID in hand and no idea how to feed it in looks
([goto-point.md](goto-point.md)). Under the button one line says what the bench
holds when it holds anything, *"3 items on the bench: double-click one in the
Scene tree's Bench groups to edit it."*, and otherwise names the pixel gesture:
*"To work on a track from a pixel: right-click it in the Image Detail panel and
choose "Start cluster on the bench here"."*.

**What it reads.** Everything about the point, including its position, colour
and error, its whole track, each observation's keypoint, its patch frame and
stored bitmap, and its triangulation diagnostics, is read through the version's
overlay accessor rather than off the base, so a point an edit modified shows the
track it holds now. An index the version has deleted resolves to nothing and the
panel takes its empty state, even though the base still has a row at that index
([document-model.md](document-model.md)).

#### The header

A compact bar of the point's summary:

```
[RGB] pt3d_a1b2c3d4_12345 [copy][->] | xyzw: (1.234, -0.567, 2.891, 1) [copy] | error: 0.42px | track: 7 obs | max pair angle: 12.3° | depth z: 41.0 | cond: 12
```

| Field | Description |
|-------|-------------|
| Colour | A swatch of the point's RGB. |
| Point ID | The copyable `pt3d_{hash}_{index}` ID, in monospace, with *Copy Point ID* and the *Go to Point* arrow beside it. |
| Position | `xyzw`, with *Copy coordinates*; *at infinity* follows when `w` is `0`. |
| Error | RMS reprojection error in pixels. |
| Track length | The number of observing images. |
| Max pair angle | The largest angle between any pair of observation rays, the main indicator of triangulation quality; shown when greater than zero. |
| Depth z | The inverse-depth z-score `depth / σ_depth`, a scale-free observability diagnostic that stays correct near infinity; shown when finite. |
| Cond | The condition number of the triangulation's normal matrix; shown when finite. |

The max angle, depth z and condition number are computed once per selection
change and cached on the body.

**The Point ID** uniquely references the point across `.sfmr` files and
sessions, which a raw index cannot. It uses only `[a-zA-Z0-9_]`, so it selects
with one double-click in a terminal or a browser. It is minted over the node's
version graph by the rule **disk state first, earliest otherwise**: the hash is
the content the point sits in on disk when its identity reaches that version,
and the oldest content its identity reaches otherwise ([goto-point.md](goto-point.md)
§ "The ID forms and the version graph"), so in the ordinary case the ID copied
out of this header is one a reader of the file on disk resolves as it stands.
The caller mints it and hands it in, because the body sees one reconstruction
value and not the node behind it, and it is **recomputed every frame**: an edit,
an undo or a save can change which content the ID names without the selection
moving. The format is the
[sfmr file format spec's](../formats/sfmr-file-format.md#point-id-portable-3d-point-references).
The copy button and the Go to Point arrow sit together because they are the two
halves of one round trip: copy an ID out of this header, paste it back here, in a
later session, or after `sfm xform` produced a new file.

**The stored-patch tile.** On an embedded-patches reconstruction that stores
patch bitmaps, a second row shows the point's stored RGBA bitmap at a fixed
64 px square with nearest-neighbour filtering, the alpha channel forced opaque.
The row is absent when the reconstruction has no bitmaps or the point's bitmap is
all zero.

#### The observation table

One row per observing image, in image-index order (the Image Browser's order).
The column headings sit above the scroll area, so they stay put while the rows
scroll under them.

```
+-----+-------+-----------------+--------+-----------+--------+-------+----------------+
|     | Image | Name            | Feat # | Size      | Error  | Angle | Feature (x, y) |
+-----+-------+-----------------+--------+-----------+--------+-------+----------------+
| [t] |     3 | image_003.jpg   |    847 |   8.4x8.2 | 0.21px | 0.03° | (1024.3, 512.7)|
| [t] |    12 | image_012.jpg   |   1247 |   7.6x7.4 | 0.38px | 0.05° | ( 983.1, 498.2)|
+-----+-------+-----------------+--------+-----------+--------+-------+----------------+
```

| Column | Content |
|--------|---------|
| Thumbnail | The image's display thumbnail (the file's own, or the row the open built from its `.sift` or photograph; see [multi-panel-image-browser.md](multi-panel-image-browser.md) § "Thumbnail loading"), with a dot at the feature position tinted by that observation's reprojection error. An image with no picture draws the placeholder and is not cached. |
| Patch | *(embedded-patches only)* The point's patch rendered from this observation's full-resolution image (below). Absent, with the following columns keeping their offsets, when the point has no patch frame. |
| Image | The image index. |
| Name | The image file name, shortened by a cut out of the **middle** so the directory and the file name both survive (`images/seatt…yard_13.jpg`); the full path on hover. A path with a directory above its parent keeps a leading `…/` for what was left out. |
| Feat # | The feature index in the image's SIFT file, or the observation index for an embedded-keypoint reconstruction with no SIFT file. |
| Size | The feature's two full extents in pixels (below); `N/A` when there is no shape. |
| Error | The observation's reprojection error, `‖project(R_i P + t_i) - x_i‖`; `N/A` when undefined. A point at infinity has one too: its unit direction rotates into camera space without translating and projects like any homogeneous coordinate, so only a point or direction behind the camera is undefined. |
| Angle | The angle at the camera centre between the observation ray and the direction to the point, in degrees; for a point at infinity, to its direction. |
| Feature (x, y) | The feature position in image pixels. |

**The thumbnail dot's colour** is `colormap::error_color`, the green to yellow
to red ramp the Image Detail error overlay draws, over a **fixed 0 to 2 px**.
Fixed rather than fitted to the track, because the dots are read against each
other and against the number in the Error column, and a range fitted to seven
observations would paint a sub-pixel track in full red. An observation with no
error to show is grey, off the ramp: "no measurement" is not a position on a
green-to-red scale.

**Size** doubles the column norms of the observation's affine shape (the
columns are the projected patch half-vectors) and prints the two full extents as
`<larger>x<smaller>` with one decimal each (`20.3x7.7`, a circular feature
`14.0x14.0`), so an obliquely viewed patch reads as foreshortened rather than
smaller. It is the span the viewport's patch quad covers and the diameter
convention `embed-patches --patch-size` uses. The shape is the cached SIFT
`affine_shapes` for a SIFT observation and the patch frame projected into the
image for an embedded keypoint. An edge-on shape shows its collapse
(`9.0x0.0`).

**The Patch column** is drawn when the reconstruction stores patch frames and
the point's `u` half-vector is non-zero. Each tile is the point's oriented patch
re-rendered from that observation's full-resolution image: the stored
half-vectors become an `OrientedPatch` (marked `w = 0` for a point at infinity),
and the image is warped through it with `WarpMap::from_patch` and
`remap_bilinear` at 64 by 64, shown at the thumbnail's 48 px with nearest
filtering. Tiles are rendered lazily and cached per image; a patch not visible
in a view warps to an all-black tile, which is cached and drawn as such. Stored
bitmaps are not needed for this column, only the frame.

**The frame is re-anchored on the observation's stored keypoint**
(`OrientedPatch::anchored_at_keypoint`, [patch-cloud.md](../core/patch/patch-cloud.md)):
the patch centre slides within its own plane, on its tangent sphere for a point
at infinity, until it projects onto the keypoint this observation carries. This
is a photometric comparison, so it is anchored where the keypoint localizer
aligned the pixels rather than at the point's geometric projection. The tiles of
a well-localized track then match each other whatever discrepancy the geometry
carries, and that discrepancy is what the Error column reports: a column of
matching tiles beside a large error is a well-aligned patch whose position or
pose is off, not a bad match. The geometric frame is used as stored when the
reconstruction has no keypoints or the keypoint's ray cannot meet the patch. The
render is `patch_color_image` in
[view/patch.rs](../../crates/sfm-explorer/src/track_view/view/patch.rs), and
edit mode draws its track-stage tiles through it too, so a committed track and
the editable copy of it cannot show one surface two ways.

**The photographs** come from `AppState::full_res_cache`, the decoded
full-resolution images shared with the Image Detail panel, so no image is
decoded more than once. An entry is the `ImageU8Pyramid` built when the image
was decoded, whose level 0 is the photograph: the photometric readers sample
the lower levels. A failed decode is remembered as `None` so a missing file is
not reopened every frame. The dock fills the cache for every observing image of
the selected point before view mode draws, when the reconstruction carries patch
frames, and fills the SIFT cache for them on a `sift_files` reconstruction. The
cache has no eviction: it holds every image decoded in the session, with its
pyramid, until the reconstruction is closed.

#### Gestures

- **Click a row**: select that image, which the 3D viewer, the Image Browser and
  Image Detail follow. A row is an observation, so the click also **reveals the
  feature**: its pixel rides along in `reveal_feature`, and Image Detail pans
  (never zooms) to centre it when its current view is not showing it
  ([multi-panel-image-browser.md](multi-panel-image-browser.md) § "Revealing a
  feature named by another panel").
- **Double-click a row**: enter camera view for that image, as a double-click on
  a frustum or a thumbnail does.
- **Hover a row**: set the cross-panel hover, which brightens the frustum and
  outlines the thumbnail. The body produces no point hover, since every row is
  about the one selected point, so owning the pointer clears it.
- **Copy Point ID** and **Copy coordinates**: copy the ID
  (`pt3d_a1b2c3d4_12345`) or the coordinates (`1.234, -0.567, 2.891`).
- **Go to Point**, from the header arrow or the empty state's button: open the
  dialog.

### Edit mode

Edit mode shows the active item and nothing else on the bench; the bench as a
list is the Scene tree's. A gesture in it that names no item means the active
track. **A cluster and a track are the same body at two stages**: the header's
headline, the table's cells and the row menu follow the stage, and there is no
separate cluster layout. What differs is outside the panel: a cluster has no
geometry, so the 3D viewer's bench layer draws nothing for it and Image Detail
draws its parallelograms rather than an outline, and no ghost outline in the
images it has no sighting in. A cluster has no view mode
either, since there is no committed point behind it, so the way back to a
cluster after Edit has been cleared is its row under *Bench Clusters*.

Almost no state lives in the body. The bench is the node's, at its cursor, so a
step taken anywhere, in this panel, in the Scene tree or by an undo, shows here on
the next frame. What the body owns is about looking rather than about the track:
where the sliders stand, whether *Lock* is ticked, which rows are selected, the
tiles it has rendered, and the painting.

#### The header

The active track's label, its stage as a word, its origin as a point index or
`new`, and `N in · M candidates · K out`. Under it the stage's own headline: at
the cluster stage the reference observation and whether a template has been cut,
and at the track stage the coordinate with the last triangulation's condition
number, or the sentence saying nothing has triangulated it yet. The header is not
joined by view mode's summary: the label of an item put on from a point already
is the portable Point ID, the committed track is one Edit-clear away, and two
modes that both showed the point's summary would be harder to tell apart at a
glance.

**The word in front of the coordinate says which coordinate it is.** A track at
infinity carries a unit direction where a finite one carries a place
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "Finite
points and bearings"), and the same three numbers under the wrong rule read as a
point a metre from the world origin. So the line is `Bearing (x, y, z), at
infinity` for a bearing and `Position (x, y, z)` otherwise, *at infinity* being
view mode's word for the same thing. It comes from the track's own `at_infinity`
and not from its patch's `w`, so a point put on the bench from a reconstruction
with no patch frames reads as the bearing it is. A fit that crosses the boundary
changes the header's first word, which is how a person sees that it crossed.

#### The toolbar

Two rows. The first acts on the active track: *Evaluate*, *Fit*, the *Stage*
toggle (which names the stage it would move to), *Apply thresholds*, *Split off
N rows*, *Duplicate*, *Commit* and *Discard*. The second is the *Lock* box and
*Rename*, which opens a field in place and commits on Enter. Each entry is enabled, or greyed
with a hover text naming what is missing, and the refusal is the core step's own
sentence asked of the very track the button would act on, so the button and the
step cannot disagree. *Commit* asks the core commit; *Evaluate*, *Fit* and the
*Stage* toggle ask `evaluate_preconditions`, `fit_preconditions` and
`set_stage_preconditions`, the halves of those steps' validation that read no
photograph. A track with no patch therefore greys all three with *"this track
has no patch yet; fit it first"* rather than offering buttons whose only act
would be to decode a dozen images and fail. The commit refusal is cached against
the track's `Arc` and the node's version, since asking it builds a point record.

**Commit leaves the point it wrote selected.** The write is
[`edits/commit-track.md`](edits/commit-track.md)'s; the panel's part is that the
point the commit produced becomes the selection, whether it replaced a point or
created one, so the viewport's track rays, the observing frustums and view mode
all look at what was just written. The dock drops what the panels cached about
the node's points in the same breath. **A commit of a track the point already
holds writes nothing**, and the button is not greyed for it: the press pushes no
version and records the no-effect row *"Committed bull-nose: no effect, point
4211 already holds this track"*, with that point selected.

**Duplicate is how a second patch over neighbouring ground is started.** It puts
a copy of the active track on the bench and makes the copy active, so a patch
fitted to one piece of surface can be slid to the piece beside it rather than
built again from a pixel
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) §
"Duplicating"). The copy drops only the origin, which makes its commit create a
point, so its header reads *new*. Ctrl+D (Cmd+D on macOS) does the same from
anywhere in the window while a track is active on the selected node's bench;
with none active the key is left alone.

**Evaluate and Fit are two buttons because they are two questions.** *Evaluate*
measures every observation where it sits and **moves nothing**, so a person
asking whether a track is right gets an answer that does not change the thing
asked about, and no kernel's gate drops a row. *Fit* moves it: localize,
re-triangulate, re-fuse, and read the result back, which is why the numbers
after a fit are the numbers *Evaluate* would report. *Fit* greys with
`fit_preconditions`' sentence for a track stage with fewer than two `in`
observations, while *Evaluate* stays available for it, because one sighting is
something to report.

***Lock* says what Image Detail's handles do at the track stage.** Ticked, which
is how the panel starts, dragging a sighting's dot there slides the patch and
every sighting follows it, and the patch-wide handles are all live: the
outline's edges and corners, the normal's segment and arrowhead, and the ghost
outline in an image the track has no sighting in, which then takes the same
edits read against the patch as it stands. Cleared, the dot moves that one
sighting's keypoint and the patch and every other sighting stay where they are,
which is how a keypoint that settled on the wrong detail is fixed; the edges,
the corners, the normal's two handles and the whole ghost take no drag while it
is cleared, a track-stage sighting having no size, turn, depth or facing of its
own and the ghost no keypoint to move
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "The
handles"). At the cluster stage the box is drawn and greyed, its hover
text saying that every sighting there already moves on its own. It keeps its
state across the greyed stretch, so a track taken to the cluster stage and back
is edited with the lock it had.

**The lock is a tool setting, not bench state.** It says what the next drag will
mean and nothing about the track, so toggling it is no step: no version, no
Action Log row, nothing for Undo to walk, and it is not greyed by a busy node,
whose drags are refused on their own. It is panel state for the session and is
not saved with the layout, since a lock left cleared by one session would make
the next session's first drag an edit of one sighting that nobody asked for.
The wire has no copy of it: an agent says which it means by calling
`translate_bench_patch` or `sight_bench_observation`.

**The row selection is panel state, not a version.** It is what *Split off N
rows* reads and nothing else; a split names its observations explicitly, because
`out` says a sighting does not belong here and cannot say which of two tracks it
belongs to. A change of active item clears it.

#### The thresholds

Four sliders, one per bar of `Thresholds`: minimum ZNCC, maximum shift, maximum
keypoint uncertainty and minimum relative ZNCC, so no bar is one only the wire
can move. The first three are the bars the painting reads; the fourth is the
fraction of the track's own self-agreement a sweep candidate is scored by.
Moving a slider repaints the table and changes nothing about the track; *Apply
thresholds* turns the painting into verdicts in one version carrying both.

**The painting is `apply_thresholds` run over a copy**, the core step itself
with the sliders' bars, so a row can never be painted one way and turned the
other way when the button is pressed, and a pinned verdict comes back unchanged.
It is recomputed when the track's `Arc` or the bars move, and not per frame,
because a copy of a track carries its consensus bitmap.

**The sliders stand where the active track's own bars are**, seeded when the
active item changes and again whenever a step moves that track's bars (*Apply
thresholds*, `apply_bench_track_thresholds` over the wire, an undo of either). A
drag in progress is not re-seeded, since it moves the panel's copy and leaves the
track's alone. Sliders showing anything else would paint by a rule the track
does not hold and hand it to the next press.

**Beside them, one control that is not a threshold**: *search px*, how far from
each observation's own pixel the next reading looks for its correlation peak,
in patch-grid px, starting at `EvaluateOptions::default`'s radius. Moving it
repaints nothing and changes no number until *Evaluate* or *Fit* runs, and both
carry it, so a fit's numbers and a reading's are measured in one window.

#### The observation table

One row per observation, in index order, with the headings above the scroll
area.

| Column | Cluster stage | Track stage |
|---|---|---|
| Verdict | a three-state control, clicked to cycle `in` / `out` / `candidate`; a dot marks a verdict set by hand | same |
| Tile | the `R x R` grid the refinement kernel samples where the observation sits, at its shape | the patch re-rendered from this observation, re-anchored where it sits, through view mode's warp |
| Img, Name | as view mode | as view mode |
| ZNCC | against the reference template | leave-one-out against the consensus, at the correlation peak within *search px* of the observation |
| Seed sh. | how far the refinement moved off the seed, px | how far that peak sits from the observation's own keypoint, px |
| Proj. off | absent | how far the keypoint sits from the point's projection, px |
| σ_pos | the tile's localizability | the same |
| Error, Angle | absent | the reprojection error and the ray angle |
| Status | the kernel's `member_status` | `walked 19 px, kept at seed` where the last fit refused to move it, `localized` where the reading scored it, the reason's own sentence where it could not, `not evaluated` where nothing has been read |
| From | the provenance | the provenance |

A cell with nothing measured behind it reads `-`, which says the difference
between a number a round produced and a round that has not been run.

**The two distances are two columns because they are two questions.** *Seed sh.*
is the sighting's own evidence, where the correlation would rather sit, and is
what the `max shift px` bar paints on. *Proj. off* is a statement about the
point: a mis-triangulated track shows a column of large offsets beside a column
of near-zero shifts, the picture that says the position is wrong and the
sightings are not.

**The Status cell names the refusal.** A reading drops nothing, so a row without
a ZNCC has one of core's `Unmeasured` reasons behind it, and the cell prints that
sentence (`it sits off the photograph`, `its ray grazes the patch`, `its seed
sits 2,483 px from the projection, beyond the 64 px bound`) elided to its
column. **It also names the walk a fit refused**: a sighting the fit's kernels
wanted to carry further than the `max shift px` bar kept its seed
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "The
fit's walk is bounded by the person's bar"), which a person reading `localized`
would get wrong, so the walk comes first among a scored row's answers.

**The tile is the column the numbers are about.** A ZNCC is a number; the
picture that produced it is what a person can judge. So each row draws what its
stage registers, through the code that registers it: at the track stage the
patch warped into this view and re-anchored where the observation sits, by view
mode's own warp; at the cluster stage the grid the refinement kernel samples
(`sfmtool_core::patch::cluster_refine::sample_member_grid`) at that place and
shape, over the cluster's own radius
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "The
cluster stage's units"), on the template's resolution once one has been cut. A
row with nothing to render draws an empty frame of the same size, so the columns
beside it never shift.

**Where the observation sits is one rule** at either stage: the keypoint a
reading wrote, else the refined cluster position, else the seed it was proposed
at (`crate::bench::observation_site`, which the marks, the reveal and the wire
read too). A candidate a search has just added carries only that seed, and its
tile is cut around it, so nothing has to be evaluated for a fresh row to show its
patch. The photographs are the node's full-resolution cache, which the dock fills
for the active track's images before edit mode draws; the rendered tiles are kept
against the track's `Arc` and rebuilt when a step moves it.

**Each row is painted** by what the sliders propose for it: green for
would-pass, red for would-not, the panel's faint background for a row nothing
has measured.

#### Row gestures

- **Click a row**: select its image and **reveal** the observation, at the pixel
  the Image Detail bench layer draws its mark at, and take the row into the split
  selection; Ctrl-click or Shift-click extends the selection.
- **Double-click a row**: enter camera view for its image, as a view-mode row's
  double-click does. The rows of both modes are observations of one track in one
  table position, and a gesture that worked in one mode and did nothing in the
  other would be a trap.
- **Click the verdict control**: cycle the verdict. The row rect is registered
  first and the control after it, so the control keeps its own click.
- **Hover a row**: set the cross-panel hover.
- **Right-click a row**: the searches the stage offers. *Find matches by SIFT
  query* runs the descriptor search from that observation
  ([`../core/bench/editable-track.md`](../core/bench/editable-track.md) §
  "Searching the descriptor index") and adds what it finds as candidates with
  `search (N)` in *From*; it is a row gesture because what a search searches
  from is one sighting's patch. With no index beside the node, or a stale one,
  the entry is the remedy instead, *Build SIFT Index to Search* or *Rebuild SIFT
  Index to Search*, which starts the build and runs no search
  ([`sift-index.md`](sift-index.md) § "The search entry"). At the track stage
  the menu also carries *Find matches by geometry*, which projects the patch into
  every camera and appends each newly admitted image as an untouched `candidate`
  with `sweep` provenance
  ([`../core/bench/editable-track.md`](../core/bench/editable-track.md) §
  "Searching by geometry"); it is absent at the cluster stage, which has no
  geometry to project, and needs no index. Both searches run as cancellable
  background tasks, push one version labelled by their report sentence and write
  one `Bench` row.

A row is also selected from **outside** the panel: a click on a mark of Image
Detail's bench layer or on an observation circle of the 3D viewer's selects that
observation's row, replacing the selection as a plain click does
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "The bench
layer", [`viewer-3d-bench-layer.md`](viewer-3d-bench-layer.md)). The 3D viewer
draws the circle of the one selected row larger.

Track View asks the node to look for its SIFT index whenever it draws, in either
mode, so a session finds the index the last one built without anyone asking.

#### The gestures that name a pixel

Starting a cluster and adding a candidate sighting act at a pixel, and the
viewer's way to name a pixel is a right-click in **Image Detail**, so both are
entries in that panel's context menu, *Start cluster on the bench here* and
*Add observation to bench track here*
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "Image Detail:
the context menu"). *Start cluster* puts the cluster on the bench active and
raises Track View on it (`AppState::start_cluster_here`); its radius is the
node's own default patch radius in that image (`AppState::default_patch_radius`),
converted to the cluster stage's keypoint-frame units. *Add observation* greys
while nothing is active, with *"No track is being edited: tick Edit in Track
View, or double-click a Bench item in the Scene tree."*

---

## The Scene tree's Bench rows

The Bench group rows are one per item, by label with its `in` count, the active
item drawn as a selected row, *Discard* on the secondary click
([`scene-graph.md`](scene-graph.md)).

- **Single click** selects the node the row is under and does nothing to the
  bench, the node row's own pattern (click to select, double-click to act), so a
  pass of clicks down the tree pushes no versions.
- **Double-click** makes the item active, selects its node and brings Track View
  to the front, reopening it at its home when it is closed
  (`AppState::edit_bench_item_at`). The node is selected because Track View shows
  the selected node's bench. On an item already active it pushes no version and
  writes no row: the gesture asked for the panel, and the panel is what it gets.

**Every raise is applied after the dock is back in the state.** The frame swaps
the dock out of `AppState` while a tab body draws, so a raise from inside one
would land on the placeholder and be thrown away. The Scene tree keeps a
double-click for `SceneGraphPanel::take_bench_edit` and Image Detail keeps
*Start cluster* for `ImageDetail::take_cluster_start`, and `app.rs` drains both
after the `DockArea` call, beside the point gestures that carry *Edit on Bench*.
The Action Log's layout rows read `Raised Track View panel`, `Opened Track View
panel` and `Closed Track View panel`.

---

## Other panels

**The bench layers follow the box.** Image Detail's bench layer and the 3D
viewer's draw the active item and nothing else, so with Edit clear neither draws
anything and none of their handles can be grabbed. That is what "no track is
active for editing" means in the rest of the window. Image Detail's layer also
reads *Lock*, which the dock hands it beside the active track; the 3D viewer's
does not, its handles being the patch's own.

---

## The wire

- `track_view` is the name `show_panel`, `hide_panel`, `screenshot` and
  `get_window_layout` use; `point_track` and `track_edit` are refused with the
  unknown-panel sentence, which lists the names that exist. The wire name is the
  title lower-cased by the panel-layout rule, and an alias would be a second name
  for one panel on a surface whose error message lists the names.
- **`deactivate_bench_item`** `{ "reconstruction_label": "bull" }` is the Edit
  box cleared, answering as every bench step answers; with nothing active it is
  a no-effect reply, `changed: false`, and one row. It is named for what it acts
  on and sits beside `activate_bench_item` in a listing.
- **`get_bench`'s `active.track` can be `null` on a bench that has items.**
- **A track tool that names no track, with nothing active**, is refused with
  *"No track is active on bull's bench. Name one with track, activate one with
  activate_bench_item, or put one on with create_bench_track or
  create_bench_cluster."*
- `create_bench_track` on a point already on the bench activates that item,
  which is the Edit box's own rule.

The whole bench family is [`bench.md`](bench.md) § "The wire" and
[`mcp-server.md`](mcp-server.md).

---

## Testing

- **The panel**,
  [track_view/tests.rs](../../crates/sfm-explorer/src/track_view/tests.rs),
  headless through `Context::run_ui`: the box reads the bench, edit mode drawn
  with an active item and view mode without one, and following an undo of a
  deactivation with no panel call in between; ticking it over a selected point
  reporting `set_edit`, and applied, one version putting the point on the bench,
  with a second tick after a clear activating the existing item; clearing it
  reporting `set_edit`, one version labelled `Stopped editing ...`, and the next
  frame drawing the selected point's header; the box greyed with no point
  selected and nothing active, a click on it reporting nothing, the refusal
  naming the three ways in and the empty state carrying the bench line; the
  empty state counting the items on a bench with none active; a discard of the
  active item returning to view mode; edit mode on a cluster drawing the cluster
  headline; the selection notice drawn exactly when the selected point is not the
  item's origin, with *View* and *Edit it* reporting their two gestures; and no
  reconstruction drawing no box.
- **View mode**,
  [view/tests.rs](../../crates/sfm-explorer/src/track_view/view/tests.rs):
  preparing one row per observation and re-preparing on a selection change; the
  extents; the name column; the max pair angle; per-row hover; a row click
  selecting and revealing, a double-click entering camera view; the Go to Point
  way in from the empty state and the header; the pointer ownership; the cached
  thumbnails; the patch column's presence, its tiles rendered once per image and
  anchored on the observation's keypoint, with the geometric frame where there is
  no keypoint; a deleted or out-of-range index taking the empty state.
- **Edit mode**,
  [edit/tests.rs](../../crates/sfm-explorer/src/track_view/edit/tests.rs), with
  what the table drew recorded unconditionally so the assertions read the table
  the app draws: nothing drawn with nothing active; **no item tabs**, the labels
  of two other items on the bench appearing nowhere in what the frame painted; a
  row per observation in index order; a verdict under the same observation index,
  pinned; the painting matching what applying the bars produces and leaving a
  pinned verdict; the cells following the stage; every row's tile at both stages,
  and a fresh candidate's cut around its seed; the sliders keeping where they
  were left, following the track's bars when a step moves them and re-seating on
  another item; the Status cell's reading sentence and its `walked` form; the
  header's `Bearing (...)` and `Position (` lines; the row menu's search entries
  and their remedies; a row click reporting the image and the pixel, and a
  double-click asking for camera view; *Lock* starting ticked, a click clearing
  it and a second ticking it again with no version pushed and no gesture
  reported, and the box drawn greyed at the cluster stage, where a click leaves
  it as it was.
- **The Scene tree**,
  [scene_graph/tests.rs](../../crates/sfm-explorer/src/scene_graph/tests.rs): a
  single click on a Bench row selects its node and pushes no version; a
  double-click from either group activates, selects the node and raises the
  panel; a double-click on the active item pushes no version and writes no row;
  and the raise survives a double-click made while the dock is swapped out.
- **Layout and wire**: [`panel-layout.md`](panel-layout.md) § "Testing" (the
  stock grid, the retired names refused, the startup load of an old default
  file) and [`mcp-server.md`](mcp-server.md) § "Testing" (the panel name,
  `deactivate_bench_item`, the null `active.track`).
- **`ui_basic`**: the refusal for a panel behind another names "Track View". No
  windowed test of the box: what it decides is covered headlessly, and the tab's
  presence by the layout test that walks every tab.

---

## Non-goals

- **Showing a committed track and its bench copy side by side.** The panel shows
  one or the other; clearing and ticking Edit flips between them, and each flip
  is a version. A second Track View is not possible, since a panel is a
  singleton.
- **Listing the bench in the panel.** The Scene tree's two Bench groups are the
  list, per node, with the stage each item is at.
- **Deciding anything from a number.** The sliders propose and the person
  decides.
- **A second tile beside the first**, the cluster template and each member warped
  onto it side by side, and **the remaining searches**, both proposed in
  [`../drafts/sfm-explorer-track-editing.md`](../drafts/sfm-explorer-track-editing.md).
- **Keeping the SIFT index in step with the workspace.** *Build* is asked for;
  the viewer does not watch the `.sift` files.
- **Editing the patch by hand in this panel.** The hand edits that move it are
  the two bench layers' handles and the wire's patch tools
  ([`bench.md`](bench.md) § "The wire").
- **Crops for a `sift_files` reconstruction**, which carries no patch frame to
  define one; the stored bitmap's alpha channel as a tile; a per-row highlight of
  the track ray in the 3D viewer; and a positional uncertainty display.
