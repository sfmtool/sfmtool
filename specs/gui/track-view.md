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
together with the controls that fit it and commit it, and the measurements that
are kept current as it changes.

The panel has an explicit mode so that it always says which of the two things is
on screen, and it does not list the bench: the Scene tree already lists each
node's bench in two groups that say which stage each item is at, and a
double-click there is the way into editing an item.

Related specs: [`bench.md`](bench.md) (the bench, the versions its steps push
and the Scene tree groups), [`edits/commit-track.md`](edits/commit-track.md)
(the Commit button's edit), [`scene-graph.md`](scene-graph.md) (the Bench rows'
gestures), [`multi-panel-image-browser.md`](multi-panel-image-browser.md) (the
Image Detail panel, which carries the gestures that name a pixel and draws the
focused item as its bench layer),
[`viewer-3d-bench-layer.md`](viewer-3d-bench-layer.md) (the same track in the 3D
viewer), [`../core/bench/bench.md`](../core/bench/bench.md) and
[`../core/bench/editable-track.md`](../core/bench/editable-track.md) (the value
edit mode shows and every step it calls), [`panel-layout.md`](panel-layout.md)
(its tab and its home), [`goto-point.md`](goto-point.md) (the dialog both modes
reach), [`background-tasks.md`](background-tasks.md) (where Fit, the stage
change, both searches and the index build run), and
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
the header, the toolbar and the boxes, `table.rs` the observation table,
`tile.rs` the tile each row draws and its hover view, and `crop.rs` the crop
of the photograph beside the tile and its hover view. The bench steps it reports are `AppState`
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
    pub fit: bool,
    pub set_stage: Option<StageKind>,
    pub apply_thresholds: Option<Thresholds>, // a threshold box released
    pub accept_walk: Option<usize>,           // a kept-at-seed row's Accept walk
    pub split: Option<Vec<usize>>,
    pub duplicate: bool,
    pub commit: bool,
    pub build_index_files: bool,           // a row's Build/Rebuild Index Files
    pub search_descriptors: Option<usize>,  // Find matches by SIFT query
    pub search_geometry: Option<usize>,     // Find matches by geometry; track stage only
    pub set_verdict: Option<(usize, Verdict)>, // a row's Keep switch, or its pin pinning
    pub unpin_verdicts: Option<Vec<usize>>,    // a pin, Unpin in a menu, or the Keep heading's pin
    pub pin_verdicts: Option<Vec<usize>>,      // the Keep heading's pin when no row is pinned
    pub request_goto_point: bool,              // the header's go-to button
    pub select_image: Option<usize>,
    pub request_camera_view: Option<usize>,
    pub reveal_feature: Option<[f32; 2]>,
    pub hovered_image: Option<usize>,
    pub has_pointer: bool,
}

impl AppState {
    /// What the Edit box asks: `true` puts the selected point on the bench
    /// (focusing the item already from it), `false` unfocuses.
    pub(crate) fn set_editing(&mut self, id: ReconId, on: bool) -> Result<(), String>;
    /// The item Track View edits, one for the viewer; no version, one
    /// `Selection` row ([`bench.md`](bench.md) § "The focused item").
    pub(crate) fn focus_bench_item(&mut self, id: ReconId, label: &str) -> Result<(), String>;
    /// Leave no item focused, every item staying on its bench. No version.
    pub(crate) fn unfocus_bench_item(&mut self);
    /// The focused item's label on `id`'s bench, which the box reads.
    pub(crate) fn focused_item_label(&self, id: ReconId) -> Option<&str>;
    /// A Scene tree double-click on a Bench row: select the node, focus the
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
    state.set_editing(id, on)?;         // a put is a bench step, refused in its own words
}
// then the view or the edit response, whichever was drawn
```

### Why it is shaped this way

**Two bodies behind one tab, not one body.** The two modes' state is disjoint.
View mode caches thumbnails and patch tiles per image of a committed point;
edit mode caches tiles and their hover views per observation of a bench
track keyed on its `Arc`, and
the box seeding, the bars' judgement and the commit refusal. Keeping each as the
struct it is means neither cache learns about the other, and each body's
headless tests read what that body drew. What the panel adds is the checkbox,
the dispatch on it and the notice.

**The mode is derived from the focused item, so nothing about it is stored in
the panel.** A panel flag would have to be set by each of the places that can
focus or unfocus an item, and an Undo that takes the item off the bench would
leave it disagreeing with the bench at the cursor. Reading the focused item's
label each frame costs one lookup on the bench.

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

**The box shows the focused item, not an independent setting.** It is checked
if and only if the focused item is on the selected node's bench
(`AppState::focused_item_label`, [`bench.md`](bench.md) § "The focused item").
Every way the focused item can change therefore moves the box on the next frame
with no code keeping the two in step: a Scene tree double-click, a step that
puts an item on the bench, a wire call, and an Undo or Redo that lands on a
version without the item.

There is **one focused item for the viewer**, and a point track and a cluster
are not two kinds for that purpose. Both are editable tracks at different
stages, which is why the Scene tree draws them in two groups and still marks
one selected row across both ([`bench.md`](bench.md) § "The Bench groups in the
Scene tree"). So a ticked box names exactly one item, whether it sits under
*Bench Points* or *Bench Clusters*, and focusing a cluster while a track is
being edited leaves the track on the bench, unfocused.

### Transitions

| From | Gesture | What happens | Then shown |
|---|---|---|---|
| Viewing a selected point | Tick Edit | `put_point_on_bench(selected point)`: a new item, focused, in one version; or, when an item on the bench already came from that point, that item is focused instead, with no version | Edit mode, on that item |
| Viewing, no point selected, nothing focused | Tick Edit | Nothing: the box is greyed, its hover text naming the ways in | Unchanged |
| Editing an item | Clear Edit | `unfocus_bench_item`: the item stays on the bench and nothing is focused, with no version | View mode, on the selection |
| Either | Double-click a Bench row in the Scene tree | That item is focused, its node is selected, and the panel is raised | Edit mode, on that item |
| Either | *Edit on Bench* in the 3D viewport or Image Detail, or a double-click on a point or a feature | As the ticked box from that point, then the panel is raised | Edit mode |
| Either | *Start cluster on the bench here* in Image Detail | A cluster is put on the bench, focused, and the panel is raised | Edit mode, on the cluster |
| Either | *Create Track Here* in Image Detail, or its Control+Shift click, once its worker lands a track | The track is put on the bench, focused, and committed; the point it wrote is selected. No panel is raised | Edit mode, on the new item |
| Editing | *Duplicate*, *Split off N rows* | The new item is put on the bench and focused | Edit mode, on the new item |
| Editing | *Discard* | The item leaves the bench and is unfocused | View mode |
| Editing | *Commit* | The point is written and selected; the item stays on the bench and stays focused | Edit mode, on the same item |
| Either | Undo or Redo | The item stays focused while the version landed on holds it, and is unfocused when it does not | Edit mode on the item, or view mode |

The box is greyed while a background task holds the node, with the node's own
busy sentence, only when ticking it would put a point on the bench, since that
is a bench step and a bench step is refused then. Clearing it, and ticking it
over a point an item on the bench already came from, push no version and are
never greyed for a busy node. With no point selected and nothing focused, its
refusal is one
sentence naming the three ways in: *"Nothing to edit: select a point,
double-click an item in the Scene tree's Bench groups, or right-click a pixel in
Image Detail and choose "Start cluster on the bench here"."*, the last quoted
from that entry's own constant (`track_view::nothing_to_edit`).

### Why each transition is the one it is

**Clearing Edit only unfocuses.** The box stands for "being edited", not for
being on the bench, and a discard already has its own toolbar button and
Scene tree menu entry. An Edit-clear that discarded would make looking at the
committed point cost the person their verdicts and their fit.

**An unfocus is not a version**, and neither is a focus: each writes one
`Selection` row (`Stopped editing IMG_0042@142,198`, `Editing
IMG_0042@142,198`) and leaves the history alone. A person who turns an
observation out, clears Edit to look at the committed point and presses Ctrl+Z
gets the verdict back, rather than spending the undo on the change of what was
being edited; and flipping Edit to compare a track with its committed point
pushes nothing.

**Ticking Edit on a point already on the bench reuses its item**: the item whose
origin, followed to the cursor, is that point is focused rather than a second
one put on, which is what `put_point_on_bench` does for every way in. Two items
for one point are two answers to one question, and with no list of items in the
panel a second one would be easy not to notice.

**Discarding the focused item unfocuses it**, so the panel returns to view
mode. Focusing a neighbour instead would switch the panel to an item the person
did not ask for, and nothing on screen would say which one arrived. An undo of
the discard puts the item back without focusing it.

**Commit leaves edit mode on**, the item on the bench and focused and the written
point selected. A person commonly commits and keeps going, the written point is
one Edit-clear away, and clearing on commit would make Commit two bench steps.

**With no point selected and nothing focused, the box is greyed** rather than
bringing back the last focused item or the last item on the bench. "The last one"
is not recorded anywhere a person can see, and the Scene tree double-click names
the item exactly.

**Start cluster on the bench here raises Track View.** It turns edit mode on,
which makes it the same kind of gesture as *Edit on Bench*, and that one raises
because a track staged into a panel nobody can see is a gesture with no answer.

---

## The selection and the focused item

The **selection** is the viewer's selected point, image and camera, which every
panel reads and which clicking in the 3D viewer or Image Detail moves. The
**focused item** is held beside it, outside every version. The two are
independent, and the box decides which one Track View shows: view mode shows the
selection, edit mode shows the focused item.

**Editing is sticky.** Selecting another point while editing moves the
selection, and with it the 3D viewer's track rays, the Image Browser's borders
and Image Detail's highlighted feature, but it does not change what Track View
shows and pushes no version. The person may be clicking around the scene to see
what else the patch covers, and a panel that dropped the edit on every click
would make that impossible.

**The selection notice** is what says the two have parted. It is one line under
the box, drawn in edit mode exactly when the node has a live selected point that
is not the focused item's origin followed to the cursor: *"Selected:
pt3d_a1b2c3d4_5120, not the track being edited."*, with *View*, which clears
Edit, and *Edit it*, which puts that point on the bench as the ticked box does.
*Edit it* greys with the busy sentence while a task holds the node and the
point is not on the bench yet; *View* is never greyed for it.

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
| Infinity mark | ∞, left of the ID, for a point at infinity (`w` is `0`); absent otherwise. Its hover text says the point is a direction and its numbers a unit bearing. |
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
| Error | The observation's reprojection error, `‖project(R_i P + t_i) - x_i‖`; `N/A` when undefined. A point at infinity has one too: its unit direction rotates into camera space without translating and projects like any homogeneous coordinate. The projection is of the ray, through the camera's own model, so a fisheye observation more than 90° off the axis has an error like any other; it is undefined only where the model has no pixel for the ray, which for a perspective model means a point or direction behind the camera. |
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
- **Double-click a row**: enter camera view for that image, and turn the view
  until the row's feature is inside the middle 1/2 of the viewport on both axes
  (`Viewer3D::look_through_toward_feature`), in one animated transition. A
  double-click on a frustum or a thumbnail enters camera view the same way but
  names no feature, and looking straight through the camera can leave the
  feature near the edge of a photograph wider than the viewport, or off it; the
  row names an observation, so the view it opens shows where it is. The turn
  starts from the view the other double-clicks land on (the camera's own pose,
  or when already in camera view the relative orientation a switch keeps), is
  level with that view's up, and uses the ray the lens model maps the feature's
  pixel from. A feature already inside the middle 1/2 turns nothing. The result
  is camera view looked around in, as a free look leaves it.
- **Hover a row**: set the cross-panel hover, which brightens the frustum and
  outlines the thumbnail. The body produces no point hover, since every row is
  about the one selected point, so owning the pointer clears it.
- **Copy Point ID** and **Copy coordinates**: copy the ID
  (`pt3d_a1b2c3d4_12345`) or the coordinates (`1.234, -0.567, 2.891`).
- **Go to Point**, from the header arrow or the empty state's button: open the
  dialog.

### Edit mode

Edit mode shows the focused item and nothing else on the bench; the bench as a
list is the Scene tree's. A gesture in it that names no item means the focused
item. **A cluster and a track are the same body at two stages**: the header's
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
where the boxes stand, whether *Lock* is ticked, which rows are selected, the
tiles it has rendered, and the bars' judgement of each row.

#### The header

The focused item's label, its stage as a word, and `N kept · K out · P
pinned`. When the
point the track was read from is still in the version at the cursor, its
portable Point ID follows the label with the two icon buttons view mode's
header draws beside an ID: copy it, and open *Go to Point*, so an ID copied
here is one that dialog takes back. A track put on from a point is labelled with
that ID until it is renamed, and the ID is then printed once, as the label. A
track whose point is gone says `from point N`, the index it had, and a track
with no origin says `new`. Under it the stage's own headline: at
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
The same flag puts **an infinity mark** (∞, U+221E, which egui's bundled fonts
draw) left of the label, as view mode puts one left of its Point ID
(`track_view::infinity_mark`), so the header's first glyph says a direction
before any number is read; its hover text says the point is a direction and
its numbers a unit bearing.

**The track's own patch stands at the left, under the label**, at view mode's
stored-patch size (64 points) with nearest filtering and no label, since the
picture says what it is. The headline, both rows of the toolbar and the
threshold boxes stand to its right, and the table's separator runs directly
under it. At the track stage it is the consensus bitmap the observations were
fused into (`TrackPayload::bitmap`), the bitmap a commit writes as the point's
stored patch, so a *Fit* that re-fuses the track changes it and a person sees
what the commit would store before committing. At the cluster stage it is the
template every member registers onto, once an evaluation has cut one. With
neither -- a track not yet fused, as one put on from a reconstruction that
stores no bitmaps is until its first *Fit*, or a cluster with no template -- the
slot is an empty frame of the same size, so the controls beside it do not move
when a step fills it. Its hover text says which of the two it is, or what would
fill it. The bitmap is converted by view mode's own `stored_patch_image`, so the
two modes draw one stored patch one way: one channel repeated across RGB, three
as RGB, and a fourth, the confidence, dropped for an opaque alpha. The upload is
kept against the track's `Arc` and dropped with the tiles.

#### The toolbar

Two rows. The first opens with where the focused item's evaluation stands, and
then acts on the track: *Fit*, the *Stage* toggle (which names the stage it
would move to), *Split off N rows*, *Duplicate*, *Commit* and *Discard*. The
second is the *Lock* box and *Rename*, which opens a field in place and commits
on Enter. Each entry is enabled, or greyed with a hover text naming what is
missing, and the refusal is the core step's own sentence asked of the very
track the button would act on, so the button and the step cannot disagree.
*Commit* asks the core commit; *Fit* and the *Stage* toggle ask
`fit_preconditions` and `set_stage_preconditions`, the halves of those steps'
validation that read no photograph. A track with no patch therefore greys both
with *"this track has no patch yet; fit it first"* rather than offering buttons
whose only act would be to decode a dozen images and fail. The commit refusal is
cached against the track's `Arc` and the node's version, since asking it builds
a point record.

**There is no *Evaluate* button, because evaluation is live.** Every change to
an input of the evaluation evaluates the track again on a worker, and a track
put on the bench is evaluated first ([`bench.md`](bench.md) § "Live
evaluation"). What the panel owes the person is which state the numbers below
are in, and the head of the toolbar says it: *Evaluated* when they are the
evaluation of the track as it stands; a spinner and *Evaluating…* while an
evaluation of the current inputs is running or waiting to start; and, when core's
`evaluate_preconditions` refuses the track or the evaluation failed, that
sentence in the warning colour -- *"Cannot evaluate bull-nose: the track carries
no patch frame to read against; upgrade it from the cluster stage to build
one"* -- in place of any state.

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
a copy of the focused item on the bench and focuses the copy, so a patch
fitted to one piece of surface can be slid to the piece beside it rather than
built again from a pixel
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) §
"Duplicating"). The copy drops only the origin, which makes its commit create a
point, so its header reads *new*. Ctrl+D (Cmd+D on macOS) does the same from
anywhere in the window while an item on the selected node's bench is focused;
with none focused the key is left alone.

**Evaluating and fitting are two things because they are two questions.** The
evaluation measures every observation where it sits and **moves nothing**, so a
person asking whether a track is right gets an answer that does not change the
thing asked about, and no kernel's gate drops a row; that is why it can run on
its own after every change. *Fit* moves the track: localize, re-triangulate,
re-fuse, and read the result back, and it stays a button because it replaces
what the person placed. *Fit* greys with `fit_preconditions`' sentence for a
track stage with fewer than two `in` observations, while the evaluation still
runs for it, because one sighting is something to report.

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

**The row selection is the bench's selected observations**, held in the state
rather than in the panel ([bench.md](bench.md) § "The selected observations"),
so a click on a mark in either bench layer and the wire's
`select_bench_observations` set the rows the panel highlights, and
`get_bench_track` reports the rows a person clicked. A row click reports itself
as `TrackEditResponse::pick_row` and the dock applies it through
`AppState::pick_bench_observation`. It is not a version: undo, redo, a jump and
a change of focused item clear it; a rename keeps it. *Split off N rows* takes it; a split names its
observations explicitly, because `out` says a sighting does not belong here and
cannot say which of two tracks it belongs to.

#### The thresholds

Five boxes, one per bar of `Thresholds`: minimum ZNCC, minimum middle ZNCC,
maximum shift, maximum self-similarity radius and minimum relative ZNCC, so no
bar is one only the wire can move. Each is its label and a number box: dragging
the box left or right changes the bar (half a percent per point for the ZNCC
bars, 0.05 grid px per point for the shift, 0.02 grid px per point for the
self-similarity radius),
and clicking it takes a typed value. There is no slider rail beside it, since a
rail would say nothing the box does not. The first four are the bars the painting and the table's colours read;
the fifth is the fraction of the track's own self-agreement a geometry search's view is
scored by. The three ZNCC boxes read and take percent in whole steps, as the
table's ZNCC column reads, so *min ZNCC (%)* and *min middle ZNCC (%)* both show
70 on a new track while the track and the wire hold 0.7; a typed value may end
in `%`. A middle bar of 0 turns it off, and a row with no middle reading clears
it.
*shift px*, the maximum shift, is in patch-grid px, the unit of the
self-similarity radius, and 6 on a new track. It is three things at once, since
they are one question -- how far from where a sighting is the correlation may
put it: the bar the *Shift* column is judged by, the radius the evaluation
looks for each peak within, so moving it evaluates the track again, and the
bound on how far a *Fit* may move a sighting
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "The
fit's walk is bounded by the person's bar"). Its label's hover text says so.
There is no separate search radius.
*self-sim. px*, the largest self-similarity radius, is in patch-grid px, 2.5 on a
new track, and takes `0` to `3` to one decimal. It judges the whole tile's
radius, the upper reading in the *Self-similarity* column; `3`, the largest
radius read, turns nothing out, and a row with no reading clears it. Its label's
hover text says what the radius is, what a row past the bar is, and that `3`
turns nothing out.

**A box applies to the focused item when it is let go.** Dragging one
recolours the table live; releasing it sets the track's bars to where the five
boxes stand and turns the painting into verdicts, one version carrying both,
with the row `Applied the thresholds to …` in the Action Log, and Undo reverses
it. A typed value is the same gesture, applied when the field is left rather
than per keystroke, and an arrow key on a focused box is one step each. No
intermediate drag position pushes a version, and a release that leaves the bars
where the track has them pushes nothing. There is no separate *Apply* button:
a box that coloured the table while the track kept its old bar would let a
*Fit* run on a bar the person had already moved away from. The boxes are
greyed with the busy sentence while the node is busy, since a release there
would be refused.

**The colours are the core's judgement run over a copy** carrying the boxes'
bars: `bar_checks` for each reading and `verdicts_if_unpinned` for each row's
*Keep* cell ([`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "Growing and judging"). For an unpinned row the proposal is what `apply_thresholds`, the
step a release applies, gives it, so a row can never be shown one way and turned
the other way when the box is released; for a pinned row it is what unpinning
it would give, which a release does not apply, since the painting leaves a
pinned verdict unchanged. It is recomputed when the track's `Arc` or the bars
move, and not per frame, because a copy of a track carries its consensus
bitmap.

**The boxes show the focused item's own bars**, copied from it on every frame
no box is being dragged, so whatever moved them -- a release here,
`apply_bench_track_thresholds` over the wire, an undo or redo of either, another
item focused -- the boxes follow. Only during a drag do they hold a value
the track does not.

#### The observation table

One row per observation, in index order, with the headings above the scroll
area. The crop of the photograph around the patch's outline is the first
column, at the table's left edge, under *Crop*; the rendered tile is the
second, under *Patch*; and *Keep* follows them. The crop comes first because it
is the photograph as it is, and the tile beside it is that patch warped square,
so the eye reads from the raw pixels to the picture the numbers are read from.
The headings are drawn at the cells' own body size, in the weak text colour so
they still read as headings. Each heading has hover text over the width of its
column, running to where the next heading starts, saying what the column holds:
the *Crop* heading's says what the crop shows and what its hover view adds, the
*Patch* heading's what the tile is at each stage and that the numbers are read
from it, and the *Keep* heading's says
what a kept observation is used for, when the thresholds set the switch, what a
click on the switch, on the pin and on the heading's own pin does, and what the
cell's colour means.

**The *Keep* heading carries a pin** over the rows' pin column, and it
toggles. While any row of the focused item is pinned, clicking it unpins every
pinned row in one step (`AppState::unpin_bench_verdicts`, core's
`unpin_verdicts`). While none is, clicking it pins every row at the verdict it
has now (`AppState::pin_bench_verdicts`, core's `pin_verdicts`), moving no
verdict, so a person can fix what the bars decided and undo an *Unpin all*.
Either is one version and one `Bench` row, such as *Pinned 12 verdicts in
{label}: 5 in, 7 out*. It is drawn solid while any row is pinned and as an
outline when none is, its accessible name is *Unpin all* or *Pin all* to
match, and its hover text says what a click does with the count: *Unpin all 12
pinned verdicts and let the bars decide*, or *Pin all 12 verdicts as they
stand*. It is greyed on a track with no observations, and while the node is
busy with the same busy sentence the threshold boxes carry, since the step
would be refused. The *ZNCC* heading's says what a ZNCC is, that `whole` is over the
whole patch and `mid` over its middle half, what the two apart mean, and how
the grid beside them is coloured. The *Self-similarity* heading's
says what the radius is and what its two readings, its colours and its lines
mean, and ends by saying that the *self-sim. px* bar judges the whole tile's
radius. The *Proj. err* heading's says what the error is measured to before and
after the track is triangulated, and that the degrees are the same residual as
an angle.

**A cell with two readings prints them on two lines**, each with its unit, and
the whole patch's and the middle's each with its name: `93% whole` over
`89% mid`, `0.4 px whole` over `1.4 px mid`, and `0.65 px` over `0.08°` for
the reprojection error. The rows are tall enough for the tile, so the second
line costs no height, and a reading that names its own part and unit needs no
explanation in the heading, which carries the column's name alone. A reading
that is not there prints a bare `-`, with no unit.

| Column | Cluster stage | Track stage |
|---|---|---|
| Crop | the photograph around the observation's parallelogram, with the parallelogram over it; hovering it shows it in context, with the patch's two axes in pixels | the photograph around the patch's outline as Image Detail draws it, with the outline over it; hovering it shows it in context, with the projection and the patch's two axes in pixels |
| Patch (the tile) | the `R x R` grid the refinement kernel samples where the observation sits, at its shape; hovering it shows it in context | the patch re-rendered from this observation, re-anchored where it sits, through view mode's warp; hovering it shows it in context, with the projection |
| Keep | a switch, on for `in` and off for `out`, then a pushpin, solid on a verdict set by hand and a faint outline otherwise; each takes clicks over the whole height of the row; the cell is tinted by what the bars propose | same |
| Img, Name | as view mode; the name is elided in its middle to fit, and hovering it shows it whole | as view mode, and the same |
| ZNCC | against the reference template, over the middle ZNCC: `92% whole` over `61% mid`, then the ZNCC grid | leave-one-out against the consensus, at the correlation peak within *shift px* of the observation, over the middle ZNCC, then the ZNCC grid |
| Proj. err | absent | the reprojection error: how far the keypoint sits from the point's projection, or, before the track is triangulated, from its patch's centre's, over the same residual as the ray angle, comparable across lenses and depths: `0.65 px` over `0.08°` |
| Self-similarity | the tile's ZNCC self-similarity radius over its middle square's: `0.4 px whole` over `3+ px mid`, `3+` for the largest, then the self-similarity grid and the surface plot | the same |
| Shift | how far the refinement moved the member off its seed, in patch-grid px: `1.20 px` | how far the correlation peak, looked for within *shift px*, sits from the observation's own keypoint, in patch-grid px on the patch's plane; just before Status, which says what a fit did with a shift past the bar |
| Status | the kernel's `member_status` | `walked 19 grid px (ZNCC 87% / 41% there), kept at seed` where the last fit refused to move it, the ZNCC being the one the fit scored at the walked peak (left out where it scored none), `localized` where the evaluation scored it, the reason's own sentence where it could not, `not evaluated` where nothing has been read |
| From | the provenance | the provenance |

A cell with nothing measured behind it reads `-`, which says the difference
between a number a round produced and a round that has not been run.

**The ZNCC cell holds two readings of the same samples.** `whole` is the
whole-patch ZNCC, the number the *min ZNCC* bar judges. `mid` is the
middle ZNCC: the same samples, at the same peak and against the same reference,
correlated over only the centred square half the grid's width (the middle
`12 x 12` of a `24 x 24` grid). A whole-patch ZNCC can be high because of the
parts of the patch away from its centre: a small near object whose patch is
mostly the background behind it, a pixel at a depth edge, a texture that
repeats along the epipolar line. The middle reading is low in each of those
cases, so the pair shows whether a match is carried by the pixel's own
neighbourhood or by its surroundings. Both print in percent, `92%` for a
ZNCC of `0.92`, which says the same two digits in fewer characters and keeps
the column narrow. A sentence that quotes a pair, such as the walk's Status
cell, writes it on one line (`87% / 41%`). The bars and the wire keep ZNCC on
its own `0 .. 1` scale. A row with a ZNCC and no middle reading prints `- mid`
under it, as on a track read back from a committed point before its first
evaluation, since `.sfmr` stores the whole-patch score alone. The table has no column sort, so nothing is ordered by
either reading.

**Beside the two readings is the ZNCC grid**, drawn as three rows of three
boxes with a border: the same samples correlated over each ninth of the patch's
whole square, corners included, with every pixel weighted equally (see
[`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "The
ZNCC grid"). The boxes are laid out as the tile is, so each sits over the part
of the tile it read, and a whole-patch match with one red corner points at the
corner that disagrees. A box is red at a ZNCC of `0.5` and below, green at
`1`, and yellow halfway, and grey where the patch is flat over it. The grid is
absent where the cell prints `-`. Hovering it shows its nine numbers in
percent, in the same layout.

**The Self-similarity cell holds the ZNCC self-similarity radius** (see
[`../core/patch/zncc-self-similarity-radius.md`](../core/patch/zncc-self-similarity-radius.md)
and § "The ZNCC self-similarity radius" of
[`../core/bench/editable-track.md`](../core/bench/editable-track.md)): how far,
in patch-grid pixels, the tile's core can slide over itself and still match
itself as well as a true match between two views would, read where its ZNCC
against itself, interpolated between whole-pixel shifts, falls through that
level. `whole` is the whole core's and `mid` its middle square's, each to one
decimal (`0.4 px`, `1.4 px`), and `3+ px` for the largest radius searched,
which reads "3 or more". Beside them is the grid of each ninth of the core read
alone: a box is green under `1`, yellow from `1` to `2`, orange from `2` to
`3` and red at `3` or more, and carries a dark
line along its slide where the slide is at least `0.5` long, so the ninth's
matching shifts line up along one direction, as on an edge. Hovering it shows
its nine numbers as the cell prints them. The *self-sim. px* bar judges
`whole`; the middle and the grid are shown and judged by no bar.

**Beside the grid is the core's surface plot**: the whole core's ZNCC against
itself at every whole-pixel shift the radius searches, interpolated between
the shifts (Catmull-Rom, repeating the edge value past the square's edge)
and drawn over the whole square of shifts, with the contour at `1 - τ`, the
level the radius is read at, over it. The colour ramp is keyed to that level so
the region inside the contour reads as one shape: below the level a muted ramp
from dark to mid slate, and at it a jump to bright amber that lightens towards
pale yellow at `1`. A dot marks each whole-pixel shift inside the contour and a
ring marks the centre, and the radius is how far from the centre the contour
reaches, read up to `r`. A
small ring round the centre is a patch that locks; a long region is a patch
that slides along it; a region that runs to the edge of the square is one that
slides at least as far as the radius looks. Hovering the plot draws it large
with a sentence giving the contour's level, the tolerance and the radius. The
picture and the contour are computed once per reading and kept until the row's
surface or tolerance changes.

All three grids and the surface plot fade with the numbers while an evaluation is on its way.

**The cells follow where the evaluation stands.** While an evaluation of the
track's current inputs is on its way, the numbers are the previous
evaluation's: they are drawn greyed and every Status cell reads *Evaluating…*,
so a number is never presented as the track's when it was measured on inputs
the track has left. When the track cannot be evaluated or its evaluation
failed, every number cell reads `-` and every Status cell `not evaluated`, and
the toolbar says why.

**The two distances are two columns because they are two questions.** *Shift*
is the sighting's own evidence, where the correlation would rather sit, and is
what the *shift px* bar paints on. *Proj. err* is a statement about
the point: a mis-triangulated track shows a column of large errors beside a
column of near-zero shifts, the picture that says the position is wrong and the
sightings are not.

**One column holds the reprojection error.** The measurement carries it twice:
`projection_offset_px`, to where the patch's centre projects, and
`reprojection_error`, to where the triangulated point projects. The patch's
centre is kept on the point once there is one, so wherever both are measured
they are one number, and the column shows the error to the point, or to the
patch's centre before the track is triangulated. Both fields stay on the wire
and in the Python dicts. The same residual as an angle, `ray_angle_deg`, is the
cell's second line, so the error reads in pixels and in degrees together.

**The Status cell names the refusal.** An evaluation drops nothing, so a row without
a ZNCC has one of core's `Unmeasured` reasons behind it, and the cell prints that
sentence (`it sits off the photograph`, `its ray grazes the patch`, `its seed
sits 2,483 px from the projection, beyond the 64 px bound`) elided to its
column. **It also names the walk a fit refused**: a sighting the fit's kernels
wanted to carry further than the *shift px* bar kept its seed
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "The
fit's walk is bounded by the person's bar"), which a person reading `localized`
would get wrong, so the walk comes first among a scored row's answers. The
row's own *ZNCC* cell beside it is the evaluation's, taken with the sighting at its
seed, so the two numbers a person weighs the walk by sit on one row.

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

**Where the observation sits is one rule** at either stage: the track-stage
keypoint, else the refined cluster position, else the seed it was proposed
at (`crate::bench::observation_site`, which the marks, the reveal and the wire
read too). A row a search has just added carries only that seed, and its
tile is cut around it, so nothing has to be evaluated for a fresh row to show its
patch. The photographs are the node's full-resolution cache, which the dock fills
for the focused item's images before edit mode draws; the rendered tiles are kept
against the track's `Arc` and rebuilt when a step moves it.

**Hovering a tile shows it in context.** The tooltip draws the same picture
over three times the patch's width (`tile::CONTEXT_FACTOR`), 288 points across,
so a person can see what surrounds the patch and whether it sits on the surface
they meant. The wider picture is rendered at the tile's own sampling, so the
tile is its middle third texel for texel: at the track stage the tile's frame,
re-anchored where the observation sits, is widened three times about its centre
and warped at three times the tile's resolution; at the cluster stage the kernel's
sampler reads the member grid over three times the cluster's radius at three
times its resolution, which keeps the step between samples and the mip level it
reads. Over the picture are drawn:

- the patch's box, the part of the picture the row's tile shows;
- a dot at its centre, where the observation sits;
- at the track stage, a ring where the track's point projects into this
  photograph, or, before the track is triangulated, where its patch's centre
  projects: the other end of the distance the *Proj. err* cell reports. The
  projected pixel is read onto the widened frame's plane
  (`OrientedPatch::keypoint_plane_offset`, the reading that anchors the tile),
  so the ring sits on the part of the photograph the picture shows at that
  pixel;
- a dashed line from the dot to the ring.

The marks are egui shapes over the picture rather than pixels written into it,
each a light stroke over a wider dark one so that it reads over bright and dark
photographs alike, the ring in amber. They are clipped to the picture, so a
projection outside it shows as the line leaving the edge towards it. A caption
under the picture says what each mark is and how far the projection is, in the
photograph's pixels, and that a cluster has no point to project. At the cluster
stage there is no ring and no line.

The wider picture is rendered the first time the pointer rests on a tile and
kept per row beside the tiles, dropped with them when a step moves the track,
since it goes stale exactly when the tile does. The tooltip's hover region
takes no click, so a click or a double-click on the tile is still the row's.
The picture itself is `tile::context`, a pure function of the track, the
observation and the photograph, which returns the picture with the box, the
keypoint and the projection in its own texels, so the tests check the geometry
rather than the pixels on screen.

**The crop beside the tile shows the patch as the photograph holds it.** The
tile is the patch warped square, which hides how the lens and the view bend
it. The crop is the photograph itself, cut around the outline Image Detail's
bench layer strokes for the row, with that outline over it in the row's
verdict colour and none of the layer's other marks: no dot, no normal, no SIFT
features. At the track stage the outline is the patch re-anchored where the
observation sits (`crate::bench::geometry::anchored_frame`, the frame the tile
is rendered through) with its boundary projected through the camera's own
model (`geometry::project_outline`, the function Image Detail strokes), so on a
fisheye the edges show as the curves they are. At the cluster stage it is the
observation's parallelogram (`geometry::parallelogram`). The crop is the
outline's bounding box widened by one photograph pixel on every side
(`crop::CROP_MARGIN_PX`) and rounded out to whole pixels, so the whole outline
is in it with its stroke clear of the edge, and then widened on its shorter
side, evenly on either side, until it is square. The photograph is not
stretched: the extra is more of the photograph around the outline, which stays
in the middle, and the square crop fills the square cell. It is read one texel
per photograph pixel up to 128 texels a side, and past that from the pyramid
level nearest the step, so a patch that spans half a photograph costs a small
texture; texels off the photograph are transparent. It is magnified with
nearest filtering as the tile is, so the photograph's own resolution shows.

**Hovering the crop shows it in context**, as hovering the tile does: the same
crop centred in three times its width and height of the photograph
(`tile::CONTEXT_FACTOR`), at the crop's own sampling, so the crop is its middle
third texel for texel, drawn at 288 points with the outline over it. Over the
outline are the tile's hover view's marks, drawn by the same code
(`tile::paint_marks`): a dot where the observation sits and, at the track
stage, the amber ring where the track's point projects, or before the track is
triangulated its patch's centre, with a dashed line from the dot to the ring.
The picture is the photograph itself, so each mark sits at its own pixel. The
crop is not boxed: the outline already shows where it is. The caption says
what the marks are and how far the projection is, in the sentence the tile's
hover view uses (`tile::projection_sentence`), then gives the crop's size in
photograph pixels and the patch's two axes, the one across the tile and the one up it, as
lengths in the photograph's pixels. At the track stage each axis is measured
along its projection through the lens, from one edge's midpoint through the
centre to the opposite edge's, so a bent axis is measured along its bend; at
the cluster stage it is the shape's column times the template's width. The
crop and its hover view are cached per row beside the tiles and dropped with
them, and the hover view is rendered the first time the pointer rests on the
crop. Its hover region takes no click, so a click on the crop is the row's.

**Each judged reading is coloured by its bar.** Four readings are judged, one
per bar: the *whole* line of the ZNCC cell by *min ZNCC*, its *mid* line by
*min middle ZNCC*, the Shift cell by *shift px*, and the *whole* line of the
Self-similarity cell by *self-sim. px*. A reading that clears its bar is drawn
green and one that does not red, in a green and a red chosen for each of the
dark and light visuals so they read as text on the panel's background. Each line
of a two-line cell takes its own colour, so a whole ZNCC can be green over a red
middle one. Everything no bar judges keeps the plain text colour: *Proj. err*,
the self-similarity *mid* line, Status, From, the *mid* ZNCC while its bar is
off at 0, a reading that is not there (`-`, though it clears its bar), and every
reading of a row nothing at this stage has measured or whose evaluation was
refused. A `NaN` reading fails its bar and is red. While an evaluation is on its
way the colours fade with the numbers, as the grids do, so a stale number does
not read as judged. The colours follow the boxes live during a drag.

**The *Keep* cell's colour is what the bars say, and the switch is what the
person decided.** The switch-and-pin cell, the full height of the row, is
tinted green where the bars propose `in` and red where they propose `out`, and
left untinted for a row nothing has measured; the rest of the row carries only
the selection and hover highlights. For an unpinned row the proposal is what
applying the bars makes it. For a pinned row it is what unpinning it would
make it: its bars, and whether its image is free -- not already held by a
pinned `in` sighting of the same image, or by an unpinned one the painting
takes because it scores better. A switch that is on in a red cell, or off in a
green one, is a hand ruling against the bars. The switch's hover text says what
the switch and the pin say, then why the bars propose what they do: that the
row clears every bar, or which bars it fails, named as the headings name the
readings (`ZNCC whole is under the bar`, `ZNCC mid is under the bar`, `Shift is
over the bar`, `Self-similarity whole is over the bar`), or, for a row that
clears every bar and is still proposed `out`, that another sighting in the same
image is kept, naming the image.

**The Keep switch is the verdict.** On is `in`: the evaluation and a fit read
the track by the observation, and a commit writes it. Off is `out`. The
thresholds set it for every unpinned row when a threshold box is let go and on
every evaluation, so an unpinned row's switch follows its latest reading in
both directions: a row added by a search or a pixel gesture joins `out` and is
switched on by the evaluation that first measures it when it clears the bars, a
row moved past a bar is switched off, and one a *Fit* moves back within the bars
is switched on again
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) §
"Evaluating"). The switch and the cell's colour then agree on an unpinned row.
The one exception is the reading that follows a repaint, which does not repaint
again so the table settles; a row it leaves out of step shows as a switch
disagreeing with its colour until the next step. A track put on the bench from
a point arrives with every row pinned `in`, so its verdicts stand until they are
unpinned. A click sets the other verdict by hand and pins it. The switch's
part of the cell, the whole height of the row, takes the click, since a
switch's own few points are a small target. There is no third state: a row
nobody has ruled on is an unpinned `out`.

**The pin beside the switch says whether a hand set the verdict**, and is the
control for it: a solid pushpin on a pinned verdict, which the thresholds leave
alone, and a faint outline on one they set. Clicking a solid pin unpins the
verdict and gives the row the one the bars propose; clicking an outline pins
the verdict as it stands. A pushpin says "pinned" where a dot said only that
something was marked, and the control sits where the state is shown.

#### Row gestures

- **Click a row**: select its image and **reveal** the observation, at the pixel
  the Image Detail bench layer draws its mark at, and take the row into the split
  selection; Ctrl-click or Shift-click extends the selection.
- **Double-click a row**: enter camera view for its image, turned toward the
  row's observation at the pixel the bench layer draws its mark at, as a
  view-mode row's double-click does. An observation nothing has placed yet has
  no pixel, and its double-click enters camera view without the turn. The rows of both modes are observations of one track in one
  table position, and a gesture that worked in one mode and did nothing in the
  other would be a trap.
- **Click the Keep switch**: turn the observation `in` or `out` by hand, which
  pins it (`AppState::set_bench_verdict`). The row rect is registered first and
  the switch after it, so the switch keeps its own click. Turning a row `in`
  while another row of its image is `in` is refused with core's sentence.
- **Click the pin**: on a pinned row, clear the pin and give the row the
  verdict the bars propose (`AppState::unpin_bench_verdicts`, core's
  `unpin_verdicts`); on an unpinned row, pin the verdict it carries
  (`AppState::set_bench_verdict` with that verdict). One version and one
  `Bench` row either way.
- **Right-click the Keep switch**, or the row: *Unpin, let the thresholds
  decide*, the same unpinning as a click on a solid pin, on a pinned row only
  (greyed on the others). On a row that is one of several selected, the row's
  menu acts on the selection instead: *Unpin 4 verdicts, let the thresholds
  decide* unpins every pinned row among the selected in one step, so the bars
  decide them together, and is greyed when none of them is pinned. The *Keep*
  switch's own menu stays with its row.
- **Hover a row**: set the cross-panel hover.
- **Right-click a row**: the searches the stage offers. *Find matches by SIFT
  query* runs the descriptor search from that observation
  ([`../core/bench/editable-track.md`](../core/bench/editable-track.md) §
  "Searching the descriptor index") and adds what it finds, unpinned and `out`,
  with `search (N)` in *From*; it is a row gesture because what a search searches
  from is one sighting's patch. With no index beside the node, or a stale one,
  the entry is the remedy instead, *Build Index Files* or *Rebuild Search
  Files*, which starts the build of the node's index files and runs no search
  ([`sift-index.md`](sift-index.md) § "The search entry"). At the track stage
  the menu also carries *Find matches by geometry*, which projects the patch into
  every camera and appends each newly admitted image, unpinned and `out`, with
  `sweep` provenance
  ([`../core/bench/editable-track.md`](../core/bench/editable-track.md) §
  "Searching by geometry"); it is absent at the cluster stage, which has no
  geometry to project, and needs no index. Both searches run as cancellable
  background tasks, push one version labelled by their report sentence and write
  one `Bench` row.
- **Accept walk**, in the same menu on a track-stage row the last fit kept at
  its seed and on no other row: put the sighting where the fit's walk would have
  taken it (`walked_to`). Its hover text gives the distance, the pixel, and the
  ZNCC pair (whole, then middle) at the seed and at the walked peak. The step is core's
  `sight_observation` at that pixel (`AppState::accept_bench_walk`), so the
  observation is pinned and the measurements read at the seed are dropped, the
  walk's among them; one version, labelled `Accepted the walk of observation 3
  of pt3d_a1b2c3d4_1207: moved 11.2 px to (1050.8, 1702.4) in IMG_0042.jpg`,
  and one `Bench` row. Greyed with the busy sentence while the node is busy. The
  wire's form is `sight_bench_observation` with `walked_to` as its pixel.

A row is also selected from **outside** the panel: a click on a mark of Image
Detail's bench layer or on an observation circle of the 3D viewer's selects that
observation's row, replacing the selection as a plain click does
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "The bench
layer", [`viewer-3d-bench-layer.md`](viewer-3d-bench-layer.md)). The 3D viewer
draws the circle of the one selected row larger.

Track View asks the node to look for its SIFT index whenever it draws, in either
mode, so a session finds the index the last one built without anyone asking.

#### The gestures that name a pixel

Starting a cluster and adding a sighting act at a pixel, and the
viewer's way to name a pixel is a right-click in **Image Detail**, so both are
entries in that panel's context menu, *Start cluster on the bench here* and
*Add observation to bench track here*
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "Image Detail:
the context menu"). *Start cluster* puts the cluster on the bench focused and
raises Track View on it (`AppState::start_cluster_here`); its radius is the
node's own default patch radius in that image (`AppState::default_patch_radius`),
converted to the cluster stage's keypoint-frame units. *Add observation* adds
an unpinned `out` sighting at the clicked pixel, which its first evaluation
switches on when it clears the bars; on a track-stage track that pixel is its
keypoint, so the row reads from it and, once `in`, commits at it without a
*Fit* first, while on a cluster it is a seed. It greys while nothing is focused,
with *"No track is being edited: tick Edit in Track View, or double-click a Bench
item in the Scene tree."*

---

## The Scene tree's Bench rows

The Bench group rows are one per item, by label with its `in` count, the focused
item drawn as a selected row, *Discard* on the secondary click
([`scene-graph.md`](scene-graph.md)).

- **Single click** selects the node the row is under and does nothing to the
  bench, the node row's own pattern (click to select, double-click to act), so a
  pass of clicks down the tree pushes no versions.
- **Double-click** focuses the item, selects its node and brings Track View
  to the front, reopening it at its home when it is closed
  (`AppState::edit_bench_item_at`), with no version. The node is selected
  because Track View shows the selected node's bench. On an item already
  focused it writes no row: the gesture asked for the panel, and the panel is
  what it gets.

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
viewer's draw the focused item and nothing else, so with Edit clear neither draws
anything and none of their handles can be grabbed. That is what "no item is
being edited" means in the rest of the window. Image Detail's layer also
reads *Lock*, which the dock hands it beside the focused item; the 3D viewer's
does not, its handles being the patch's own.

---

## The wire

- `track_view` is the name `show_panel`, `hide_panel`, `screenshot` and
  `get_window_layout` use; `point_track` and `track_edit` are refused with the
  unknown-panel sentence, which lists the names that exist. The wire name is the
  title lower-cased by the panel-layout rule, and an alias would be a second name
  for one panel on a surface whose error message lists the names.
- **`unfocus_bench_item`** `{}` is the Edit box cleared. It pushes no version
  and names no reconstruction, since there is one focused item for the viewer;
  it answers with the item it unfocused and `changed`, and with nothing focused
  it is a no-effect reply, `changed: false`, and one row. It sits beside
  `focus_bench_item` in a listing.
- **`get_bench`'s `focused_item` can be `null` on a bench that has items.**
- **A track tool that names no track, with no item on that bench focused**, is
  refused with *"No item on bull's bench is focused. Name one with track, focus
  one with focus_bench_item, or put one on with create_bench_track or
  create_bench_cluster."*
- `create_bench_track` on a point already on the bench focuses that item,
  which is the Edit box's own rule, and pushes no version.

The whole bench family is [`bench.md`](bench.md) § "The wire" and
[`mcp-server.md`](mcp-server.md).

---

## Testing

- **The panel**,
  [track_view/tests.rs](../../crates/sfm-explorer/src/track_view/tests.rs),
  headless through `Context::run_ui`: the box reads the focused item, edit mode
  drawn with a focused item and view mode without one, and following an
  unfocus and a focus made outside the panel with no panel call in between;
  ticking it over a selected point reporting `set_edit`, and applied, one
  version putting the point on the bench, with a second tick after a clear
  focusing the existing item and pushing no version; clearing it reporting
  `set_edit`, no version and one `Selection` row `Stopped editing ...`, and the
  next frame drawing the selected point's header; the box greyed with no point
  selected and nothing focused, a click on it reporting nothing, the refusal
  naming the three ways in and the empty state carrying the bench line; a busy
  node greying the box only over a point not on the bench; the empty state
  counting the items on a bench with none focused; a discard of the focused
  item returning to view mode; edit mode on a cluster drawing the cluster
  headline; the selection notice drawn exactly when the selected point is not the
  item's origin, with *View* and *Edit it* reporting their two gestures; and no
  reconstruction drawing no box.
- **View mode**,
  [view/tests.rs](../../crates/sfm-explorer/src/track_view/view/tests.rs):
  preparing one row per observation and re-preparing on a selection change; the
  extents; the name column; the max pair angle; per-row hover; a row click
  selecting and revealing, a double-click entering camera view with the
  feature; the Go to Point way in from the empty state and the header; the infinity mark left of a point at infinity's ID and absent for a finite point; the
  pointer ownership; the cached
  thumbnails; the patch column's presence, its tiles rendered once per image and
  anchored on the observation's keypoint, with the geometric frame where there is
  no keypoint; a deleted or out-of-range index taking the empty state.
- **Edit mode**,
  [edit/tests.rs](../../crates/sfm-explorer/src/track_view/edit/tests.rs), with
  what the table drew recorded unconditionally so the assertions read the table
  the app draws: nothing drawn with nothing focused; **no *Evaluate* button**,
  the toolbar reading *Evaluating…* until the evaluation of a track just put on
  the bench lands and *Evaluated* after it; every Status cell reading
  *Evaluating…* until then and again after a verdict; a frameless bearing on the
  bench showing core's refusal sentence where the state would be; **no item tabs**, the labels
  of two other items on the bench appearing nowhere in what the frame painted; a
  row per observation in index order; a verdict under the same observation index,
  pinned; the *Keep* switch's cell taking a click the row behind it does not,
  and turning a kept row out; a click on the pin pinning an unpinned row's
  verdict as it stands and unpinning a pinned one, and *Unpin* offered from the
  switch's menu
  and the row's, a second unpin being no effect; the row menu on a row of a
  multi-row selection unpinning every pinned selected row with the count in its
  label, while the switch's menu names its row alone; the *Keep* heading's pin
  unpinning every pinned row as one version, and with none pinned asking to pin
  every row, which pins them as one version with one log row and keeps each
  verdict, undoes in one step, and makes the next click unpin them all again;
  its hover text in both states and its accessible name; the header counting kept, out
  and pinned rows; the header printing the point's
  ID once, beside a renamed label, and the old index for a point that is gone,
  and a track from a point resolving the ID its copy button copies; a hover on
  an elided name showing it whole, with the row still hovered; the *Keep*
  cells' proposals matching what applying the bars produces on unpinned rows and
  a drag leaving a pinned verdict; each line of the ZNCC cell judged by its own
  bar, whole passing while mid fails and the reverse; the projection error, the
  self-similarity middle, the status, the middle ZNCC with its bar off, a
  missing reading, an unmeasured row and a refused evaluation drawn plain; a
  drag of the boxes changing the judgement without stepping the track; a pinned
  `in` row ruled against the bars proposed `out` with the failing bar in its
  hover text, and unpinning it turning it `out`; of two sightings in one image
  that clear every bar, the one that loses the image proposed `out` with the
  image named in its hover text; the crop column at the table's left edge
  under *Crop*, the tile after it under *Patch* and *Keep* after that; the
  headings as tall as the cells; the
  cells following the stage; every row's tile at both stages,
  and a fresh row's cut around its seed; a tile's hover view at both stages
  holding the tile texel for texel in its middle third with the keypoint at the
  box's centre, its projection mark mapping back through the picture's own frame
  onto the projected pixel at the row's reprojection error, no mark at the
  cluster stage, and resting the pointer on a tile showing the view for that
  row alone while the row keeps its hover and its click; every row's crop at
  both stages, the crop square,
  holding every sample of the outline at least one pixel inside it, centred to
  within the rounding and less than two pixels from both edges on its longer
  side, its texels the photograph's own pixels, its hover view holding it
  texel for texel in the middle third with the outline and the dot moved by one
  crop, the hover view's dot on the observation's pixel and its ring on the
  point's projection at the row's reprojection error with the caption stating
  it, and no ring at the cluster stage, the axes
  matching the projected edge midpoints' distance at the track stage and the
  shape's columns at the cluster stage and printed in the caption, and resting
  the pointer on a crop showing the view for that row alone while the row keeps
  its hover and its click; the boxes showing the track's
  bars outside a drag, following them when a step, an undo or a redo moves them,
  and re-seating on another item; a drag of *shift px* pushing exactly one
  version and one row on its release, with the track's bar where it was let go
  and an undo taking bar and box back; no *Apply thresholds* button drawn; a
  fit after a release to a zero bar keeping sightings at their seeds, *Accept
  walk* absent from a row before that and offered on exactly the kept rows
  after, reporting the observation, and accepting it moving the keypoint to the
  walked pixel, pinned, in one version; the Status cell's reading sentence and
  its `walked` form with and without the walked ZNCC; the ZNCC cell's
  `whole / middle` percent form at both stages, and `-` for a missing middle; the
  header's `Bearing (...)` and `Position (` lines; the infinity mark first in a bearing's header and absent from a position's; the track's patch slot empty before a fit and filled with the consensus bitmap after it, with the toolbar to its right and the table's first heading not; the cluster stage's slot following whether a template is cut; a bitmap of one, three or four channels drawn opaque and an all-zero one not drawn; the row menu's search entries
  and their remedies; a row click reporting the image and the pixel, and a
  double-click asking for camera view with the pixel; *Lock* starting ticked, a
  click clearing it and a second ticking it again with no version pushed and no
  gesture reported, and the box drawn greyed at the cluster stage, where a click
  leaves it as it was.
- **The Scene tree**,
  [scene_graph/tests.rs](../../crates/sfm-explorer/src/scene_graph/tests.rs): a
  single click on a Bench row selects its node and pushes no version; a
  double-click from either group focuses, selects the node and raises the
  panel with no version and one `Editing ...` row; a double-click on the
  focused item pushes no version and writes no bench or selection row;
  and the raise survives a double-click made while the dock is swapped out.
- **Layout and wire**: [`panel-layout.md`](panel-layout.md) § "Testing" (the
  stock grid, the retired names refused, the startup load of an old default
  file) and [`mcp-server.md`](mcp-server.md) § "Testing" (the panel name,
  `unfocus_bench_item`, the null `focused_item`).
- **`ui_basic`**: a `screenshot` of `camera_intrinsics`, which the stock grid
  keeps behind Track View, is refused with a message naming "Track View". No
  windowed test of the box: what it decides is covered headlessly, and the
  tab's presence by `layout/tests.rs`'s
  `every_panel_appears_exactly_once_in_the_default`.

---

## Non-goals

- **Showing a committed track and its bench copy side by side.** The panel shows
  one or the other; clearing and ticking Edit flips between them, and each flip
  is a version. A second Track View is not possible, since a panel is a
  singleton.
- **Listing the bench in the panel.** The Scene tree's two Bench groups are the
  list, per node, with the stage each item is at.
- **Deciding anything from a number.** The boxes propose and the person
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
