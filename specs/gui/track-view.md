# Track View

Track View is the SfM Explorer panel that shows one **track**, the
observations of one point across the photographs, one row per observation with
the measurements of each: where it sits in its image, how far it lies from where
the point projects, and whether the patch of surface it shows looks like the
others. It draws one body in two modes. With its **Edit** box clear it shows the
selected 3D point read as a track, measured the way the bench measures a track,
and changes nothing. Ticking the box puts that point on the **bench**, where an
item is held beside the reconstruction and judged before it is committed to it
([`bench.md`](bench.md)), and makes the new track the focused item, the one item
the viewer is editing; a point already put on the bench has its existing track
focused instead, and with no point selected the tick focuses the item that was
focused most recently. With the box ticked the panel shows the focused item,
together with the controls that fit it, set which observations it keeps, and
commit it, and the measurements that are kept current as it changes. Clearing
the box stops the editing and leaves the item on the bench.

The panel has an explicit mode so that it always says which of the two things is
on screen, and it does not list the bench: the Scene tree already lists each
node's bench in two groups that say which stage each item is at, and a
double-click there is the way into editing an item. What the panel does keep,
beside the box, is a strip of the few items most recently edited, so the way
back to one of them is one click.

Related specs: [`bench.md`](bench.md) (the bench, the versions its steps push,
the viewed track and the Scene tree groups),
[`edits/commit-track.md`](edits/commit-track.md) (the Commit button's edit),
[`scene-graph.md`](scene-graph.md) (the Bench rows' gestures),
[`multi-panel-image-browser.md`](multi-panel-image-browser.md) (the Image Detail
panel, which carries the gestures that name a pixel and draws the focused item
as its bench layer), [`viewer-3d-bench-layer.md`](viewer-3d-bench-layer.md) (the
same track in the 3D viewer), [`../core/bench/bench.md`](../core/bench/bench.md)
and [`../core/bench/editable-track.md`](../core/bench/editable-track.md) (the
value the body shows and every step it calls), [`panel-layout.md`](panel-layout.md)
(its tab and its home), [`goto-point.md`](goto-point.md) (the dialog both modes
reach), [`background-tasks.md`](background-tasks.md) (where Fit, the stage
change, both searches and the index build run), and
[`sift-index.md`](sift-index.md) (the `.kdf` a search queries).

---

## The interface

The panel is [track_view/](../../crates/sfm-explorer/src/track_view/):
[mod.rs](../../crates/sfm-explorer/src/track_view/mod.rs) holds the checkbox,
the empty state and the choice between them and the body,
[recent.rs](../../crates/sfm-explorer/src/track_view/recent.rs) is the recent
items strip,
[header_buttons.rs](../../crates/sfm-explorer/src/track_view/header_buttons.rs)
holds the copy and go-to icon buttons the header draws beside a point ID in
either mode, and
[body/](../../crates/sfm-explorer/src/track_view/body/) is the one body that
draws both modes: `mod.rs` the header, the toolbar and the boxes, `table.rs` the
observation table, `tile.rs` the tile each row draws and its hover view,
`crop.rs` the crop of the photograph beside the tile and its hover view,
`surface_plot.rs` the self-similarity surface plot, `reference.rs` the
*Reference* column's cells, hover text and sort key,
[`zncc_hover.rs`](../../crates/sfm-explorer/src/track_view/body/zncc_hover.rs)
the *ZNCC* cell's hover at the track stage, and `patch.rs` the warp a
track-stage tile is rendered through and the picture of a track's own patch,
which the header and the strip both draw. The
viewed track is [bench/viewed.rs](../../crates/sfm-explorer/src/bench/viewed.rs).
The bench steps the body reports are `AppState` methods in
[bench.rs](../../crates/sfm-explorer/src/bench.rs), and the dock applies them in
[dock.rs](../../crates/sfm-explorer/src/dock.rs).

```rust
pub(crate) struct TrackView {
    body: TrackBody,
    recent: RecentStrip, // the chips' uploaded patches, keyed by item and track Arc
}

impl TrackView {
    pub(crate) fn new() -> Self;
    pub(crate) fn show(&mut self, ui: &mut egui::Ui, state: &AppState) -> TrackViewResponse;
    pub(crate) fn forget_recon(&mut self, id: ReconId);
    /// Whether Edited mode's *Lock* box is ticked: what the dock hands Image
    /// Detail, whose track-stage dot slides the patch when it is and moves one
    /// sighting's keypoint when it is not.
    pub(crate) fn lock(&self) -> bool;
}

pub(crate) struct TrackViewResponse {
    /// The box ticked (`true`) or cleared (`false`).
    pub set_edit: Option<bool>,
    /// What the body reported, or the empty state in its place; `None` only
    /// with no reconstruction selected.
    pub body: Option<TrackBodyResponse>,
    /// A chip in the recent items strip clicked: its item's node and label.
    pub focus_item: Option<(ReconId, String)>,
}

pub(crate) enum BodyMode {
    Viewed, // the viewed track, Edit clear
    Edited, // the focused item, Edit ticked
}

pub struct TrackBodyResponse {
    pub mode: Option<BodyMode>,               // the mode drawn; None for the empty state
    pub viewed_thresholds: Option<Thresholds>, // Viewed: a box moved the read-only bars
    pub discard: Option<String>,
    pub rename: Option<(String, String)>,
    pub fit: bool,
    pub(crate) normal: Option<NormalStep>,    // one of the three normal buttons, with its settings
    pub set_stage: Option<StageKind>,
    pub apply_thresholds: Option<Thresholds>, // Edited: a threshold box released
    pub accept_walk: Option<usize>,           // a kept-at-seed row's Accept walk
    pub split: Option<Vec<usize>>,
    pub pick_row: Option<(usize, bool)>,      // Edited: a row click, and whether it extends
    pub duplicate: bool,
    pub commit: bool,
    pub build_index_files: bool,           // a row's Build/Rebuild Index Files
    pub search_descriptors: Option<usize>,  // Find matches by SIFT query
    pub search_geometry: Option<usize>,     // Find matches by geometry; track stage only
    pub set_verdict: Option<(usize, Verdict)>, // a row's Keep switch, or its pin pinning
    pub unpin_verdicts: Option<Vec<usize>>,    // a pin, Unpin in a menu, or the Keep heading's pin
    pub pin_verdicts: Option<Vec<usize>>,      // the Keep heading's pin when no row is pinned
    pub request_goto_point: bool,              // the header's go-to button, or the empty state's
    pub select_image: Option<usize>,
    pub request_camera_view: Option<usize>,
    pub reveal_feature: Option<[f32; 2]>,
    pub hovered_image: Option<usize>,
    pub has_pointer: bool,
}

impl AppState {
    /// What the Edit box asks: `true` puts the selected point on the bench
    /// (focusing the item already from it), or with no point selected focuses
    /// the most recent item; `false` unfocuses.
    pub(crate) fn set_editing(&mut self, id: ReconId, on: bool) -> Result<(), String>;
    /// The selected point's index on `id` when the version at the cursor holds
    /// it: the selection the box and `set_editing` both read, so a deleted or
    /// out-of-range point counts as no point selected.
    pub(crate) fn selected_point_held_in(&self, id: ReconId) -> Option<usize>;
    /// The item Track View edits, one for the viewer, and its node and origin
    /// selected; no version, one `Selection` row ([`bench.md`](bench.md)
    /// § "The focused item").
    pub(crate) fn focus_bench_item(&mut self, id: ReconId, label: &str) -> Result<(), String>;
    /// Leave no item focused, every item staying on its bench, and select the
    /// item's origin or clear the point selection. No version.
    pub(crate) fn unfocus_bench_item(&mut self);
    /// The focused item's label on `id`'s bench, which the box reads.
    pub(crate) fn focused_item_label(&self, id: ReconId) -> Option<&str>;
    /// The focused item's origin followed to the cursor, which is the one
    /// point a selection may name while the item stays focused.
    pub(crate) fn focused_origin(&self) -> Option<PointRef>;
    /// The first entry of `recent_items` on its node's bench at the cursor:
    /// what a tick with no point selected focuses.
    pub(crate) fn most_recent_item(&self) -> Option<(ReconId, String)>;
    /// Build or take from its cache the viewed track for the selected point,
    /// which Viewed mode draws ([`bench.md`](bench.md) § "The viewed track").
    pub(crate) fn refresh_viewed_track(&mut self);
    /// The read-only bars Viewed mode's boxes hold, session state.
    pub(crate) fn set_viewed_thresholds(&mut self, bars: Thresholds);
    /// A Scene tree double-click on a Bench row: focus the item, which
    /// selects its node, and raise Track View.
    pub(crate) fn edit_bench_item_at(&mut self, id: ReconId, position: usize);
    /// Image Detail's *Start cluster on the bench here*, raising Track View.
    pub(crate) fn start_cluster_here(&mut self, image: ImageRef, pixel: [f32; 2]);
}
```

A frame in the dock asks for the viewed track, draws, and applies what came back
by the mode that was drawn:

```rust
state.refresh_viewed_track();
let response = track_view.show(ui, state);
if let Some(on) = response.set_edit {
    state.set_editing(id, on)?;         // a put is a bench step, refused in its own words
}
if let Some((node, label)) = &response.focus_item {
    state.focus_bench_item(*node, label)?; // a chip: no version, never busy-refused
}
match response.body.and_then(|b| b.mode) {
    Some(BodyMode::Edited) => { /* each step on the focused item */ }
    _ => { /* selection, hover, Go to Point, set_viewed_thresholds */ }
}
```

### Why it is shaped this way

**One body, drawn in two modes, rather than two bodies.** The two modes show the
same kind of thing, one track, and a person comparing a committed point with its
bench copy should be comparing two tracks, not two layouts. Reading the
committed point as an editable track also measures it the way the bench
measures a track, so its numbers mean what the same column means once the point
is put on. The differences between the modes are what can be changed, and they
are branches on the mode in one body.

**The mode is derived from the focused item, so nothing about it is stored in
the panel.** A panel flag would have to be set by each of the places that can
focus or unfocus an item, and an Undo that takes the item off the bench would
leave it disagreeing with the bench at the cursor. Reading the focused item's
label each frame costs one lookup on the bench.

**The panel decides nothing.** `show` takes `&AppState` and every gesture lands
in the response; the dock applies each through the `AppState` method that pushes
the version, as every other panel's response is applied, because the panel holds
the state immutably while it draws. The response carries the mode, and the dock
applies a Viewed frame's response by the few gestures that mode has, so a
field only Edited mode sets can never act on a track that is on no bench.

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

The first row of the panel is a checkbox, **Edit**, with the recent items strip
to its right on the same row (§ "The recent items strip"), and nothing else is
drawn above it in either mode.

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

The rules keep one invariant: **while an item is focused, the selected node is
the focused item's node, and the selected point is the item's origin followed
to the cursor, or no point.** An item with no origin (a cluster, a duplicate, a
split, a track whose origin was deleted) is focused with no point selected.

| From | Gesture | What happens | Then shown |
|---|---|---|---|
| Viewing a selected point | Tick Edit | `put_point_on_bench(selected point)`: a new item, focused, in one version; or, when an item on the bench already came from that point, that item is focused instead, with no version | Edited mode, on that item |
| Viewing, no point selected | Tick Edit | The most recently focused item still on a bench at the cursor is focused, with no version, and its node selected; with none, the box is greyed, its hover text naming the ways in | Edited mode, on that item |
| Editing | Clear Edit | `unfocus_bench_item`: the item stays on the bench and nothing is focused, with no version. Its origin is selected when it resolves at the cursor; otherwise the point selection is cleared. The node stays selected | Viewed mode, on the origin, or the empty state |
| Editing | Select another point: a click in the 3D viewer or Image Detail, *Go to Point*, the wire's `select_point` | The item is unfocused, with no version, and the new point is selected | Viewed mode, on that point |
| Editing | Select the item's own origin | Nothing changes | Edited mode, same item |
| Editing | Clear the point selection (a click on empty space) | The item stays focused | Edited mode, same item |
| Editing | Select an image or a camera of the focused item's node (a row click, a thumbnail, a frustum) | The item stays focused | Edited mode, same item |
| Editing | Select another node, or an image, camera or point of another node (a Scene tree click, the Image Browser, the 3D viewer, `[` and `]`, opening a file, the wire) | The item is unfocused, and the selection is what the gesture made it | Viewed mode, on the new selection |
| Either | Click a chip in the recent items strip | That item is focused, with no version, and its node and its origin are selected (or the point selection cleared) | Edited mode, on that item |
| Either | Double-click a Bench row in the Scene tree | The same as a chip, and the panel is raised | Edited mode, on that item |
| Either | *Edit on Bench* in the 3D viewport or Image Detail, or a double-click on a point or a feature | The point is selected (which unfocuses any other item), then as the ticked box from that point, and the panel is raised, unless it shares a node with the panel the gesture was made in ([panel-layout.md](panel-layout.md) § "A raise from a gesture") | Edited mode |
| Either | *Start cluster on the bench here* in Image Detail | A cluster is put on the bench and focused, the point selection is cleared, and the panel is raised, unless it shares a node with Image Detail | Edited mode, on the cluster |
| Either | *Create Track Here* in Image Detail, or its Control+Shift click, once its worker lands a track | The track is put on the bench and focused, then committed; the point it wrote is the item's re-seated origin and is selected, so the item stays focused. No panel is raised | Edited mode, on the new item |
| Editing | *Duplicate*, *Split off N rows* | The new item is put on the bench and focused; it has no origin, so the point selection is cleared | Edited mode, on the new item |
| Editing | *Discard* | The item leaves the bench (a version) and is unfocused, and the selection is left as a cleared box leaves it, from the origin as it resolved before the discard | Viewed mode, on the origin, or the empty state |
| Editing | *Commit* | The origin is re-seated on the written point and that point selected, so the item stays on the bench and focused | Edited mode, on the same item |
| Either | Undo or Redo | The item stays focused while the version landed on holds it, and is unfocused, with the selection unchanged, when it does not | Edited mode on the item, or Viewed mode |

The box is greyed while a background task holds the node, with the node's own
busy sentence, only when ticking it would put a point on the bench, since that
is a bench step and a bench step is refused then. Clearing it, ticking it over
a point an item on the bench already came from, ticking it with no point
selected and clicking a chip push no version and are never greyed for a busy
node.

**Ticking Edit with no point selected** focuses the first entry of the recent
list whose item is on its node's bench at the cursor
(`AppState::most_recent_item`), on any loaded node, and selects that node. That
is the strip's first chip when the row has room for one. The box's hover text
names it, so a panel too narrow to draw the chip still says what the tick does:
*"Tick to edit pt3d_a1b2c3d4_1207 again"* (`track_view::again_hint`). It is
greyed only when no entry is on a bench, and
its refusal is then one sentence naming the three ways in: *"Nothing to edit:
select a point, double-click an item in the Scene tree's Bench groups, or
right-click a pixel in Image Detail and choose "Start cluster on the bench
here"."*, the last quoted from that entry's own constant
(`track_view::nothing_to_edit`).

**The recent list** is `AppState::recent_items`: every item focused this
session, most recently focused first and each once. Every focus, whether a
gesture, a wire call or a step that put the item on, moves the item to the
front. An entry is dropped when its node is closed, and the list is emptied by
*Close All*. An entry whose item is not on its node's bench at the cursor is
skipped by a reader rather than dropped, so an undo of a discard brings it back.
The list is not cut to any length: the strip draws the first entries that are on
a bench, so skipped entries do not leave it short. It is session state: not in a
version, not saved with the layout, and not on the wire.

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

**Selecting another point leaves Edited mode.** A click on a point in the 3D
viewer is a request to look at that point, and a panel that went on showing a
different track would answer another question. Because focusing and unfocusing
push no version, the exit costs nothing: an undo after it takes back the last
step on the bench, not the exit. The rule lives in `AppState::select_point`,
which every point selection gesture goes through, so none of them needs one of
its own. The selection that follows the point maps after an edit assigns the
selected point directly rather than through `select_point`, so an edit that
moves the selection (a deletion of the origin, a commit's undo) leaves the item
focused.

**Selecting the item's origin, clearing the point, or selecting an image or a
camera of its node keeps it focused.** None of these names another track. An
image of the node is how a person moves between the photographs an item is
being worked on in, and a row click in the table is one.

**Anything on another node unfocuses.** There is one focused item for the
viewer, and the selected node is its node while it is focused, so a selection
on another node is a selection away from it.

**Focusing selects the item's node and origin**, or clears the point selection
when it has none, and **clearing Edit selects the origin**, or clears the point
selection. Either way the invariant holds after the gesture, and the panel
shows next what the person was just working on: the committed point the item
came from, or the empty state for an item with no point. The item is focused
before its origin is selected, so the exit rule sees the origin as the focused
item's.

**Discarding the focused item unfocuses it**, so the panel leaves Edited
mode. Focusing a neighbour instead would switch the panel to an item the person
did not ask for, and nothing on screen would say which one arrived. The
selection is what a cleared box leaves, with the origin resolved before the
discard. An undo of the discard puts the item back without focusing it.

**Commit leaves Edited mode on**, the item on the bench and focused and the written
point selected. The commit re-seats the item's origin on the point it wrote
before it selects it, so the selection is the origin and the item stays focused,
including an item that had no origin before. A person commonly commits and keeps
going, the written point is one Edit-clear away, and clearing on commit would
make Commit two bench steps. An undo of the commit takes the point and the
origin back; the selection follows the map to no point, and the item stays
focused.

**An undo or redo that unfocuses leaves the selection alone.** The item left
because the version no longer holds it, not because of a gesture about the
selection, and the selection has already followed the version's map.

**With no point selected, the tick goes back to the most recent item** rather
than being greyed. The Scene tree names every item, but the one a person has
just left is the one most often wanted, and the hover text names it before the
tick.

**Start cluster on the bench here raises Track View.** It turns Edited mode on,
which makes it the same kind of gesture as *Edit on Bench*, and that one raises
because a track staged into a panel that is not on screen gives the person
no sign of what the gesture did.
Neither raise covers the panel the gesture was made in: when Track View is a
tab in that panel's node, it stays behind it, and the bench layer there shows
the focused item ([panel-layout.md](panel-layout.md) § "A raise from a
gesture").

---

## The recent items strip

To the right of the *Edit* box, on the same row, the panel draws the items most
recently focused, most recent first, leaving out the focused item
([recent.rs](../../crates/sfm-explorer/src/track_view/recent.rs)). They are the
entries of the recent list (§ "Transitions") whose item is on its node's bench
at the cursor, from any loaded node. The strip is drawn in both modes and in the
empty state, so after the box is cleared its first chip is the item just left.

**What a chip shows.** The item's patch at 24 points square, the same picture
the header's patch slot draws for that item (the patch bitmap at the track
stage, the template at the cluster stage, an empty frame when there is neither,
through `body::track_patch_image`), drawn with nearest filtering, and the label
beside it, shortened by a cut out of the middle so that the start and the end
both survive (`pt3d_…_1207`, `IMG_0…@142,198`), by the same middle elision the
table's *Name* column uses (`crate::elide::middle`), to at most 72 points.

**How many.** As many as fit whole in the width left on the row, up to eight. A
chip that would not fit whole is not drawn, and neither is any chip after it, so
the chips drawn are always the most recent ones. On a narrow panel there may be
none, and the box's hover text still names the item a tick focuses.

**Hover** shows the patch at 64 points, then one line each: the whole label; the
node's name (`On bull`) when more than one reconstruction is loaded; the stage
(`The track stage`); `N kept · K out · P pinned`; the origin (`Point
pt3d_a1b2c3d4_1207` when it resolves at the cursor, `From point 1207, which is
gone` when it does not, `New: read from no point` for an item that never had
one); at the track stage `Position (x, y, z)` or `Bearing (x, y, z), at
infinity`, as the Edited headline writes them (`body::position_text`); where the
item's evaluation stands (*Evaluated*, *Evaluating…*, or the refusal or failure
sentence); and *Click to edit*.

**Click** reports the item in `TrackViewResponse::focus_item`, and the dock
focuses it through `AppState::focus_bench_item`, which selects the item's node
first when it is on another one, then its origin, or clears the point selection
for an item with no origin. A focus is no version, so a chip is never greyed
while a background task holds the node.

**The patches are uploaded once.** Each chip's texture is cached against the
item and the track's `Arc`, so a chip does not upload its patch again every
frame; a step on the item gives it a new `Arc`, and the next frame uploads the
new picture. An entry the frame did not draw is dropped.

**It is not a list of the bench.** The Scene tree's Bench groups remain the full
list, per node; the strip is only the way back to the last few, which are the
ones most often wanted after a point click has left Edited mode.

---

## The selection and the focused item

The **selection** is the viewer's selected point, image and camera, which every
panel reads and which clicking in the 3D viewer or Image Detail moves. The
**focused item** is held beside it, outside every version, and the box decides
which one Track View shows: Viewed mode shows the selection, Edited mode shows the
focused item.

**The two cannot disagree about the point.** While an item is focused the
selected point is its origin or no point (§ "Transitions"), so Edited mode never
shows one track while the 3D viewer's track rays, the Image Browser's borders
and Image Detail's highlighted feature show another. A selection that would
name another point unfocuses the item first, and the panel shows that point in
Viewed mode.

**Go to Point** selects the point and raises Track View. It goes through
`AppState::select_point`, so a point other than the focused item's origin
unfocuses the item by the ordinary rule, and the panel shows the point jumped
to. The jump pushes no version.

**A click on a bench-layer handle is the handle's.** In Image Detail and in the
3D viewer, a press a handle of the focused item's figure takes owns the whole
gesture: its release selects no point under the handle and stages none, so a
click on the item's own outline, corner, edge, normal or ghost cannot pick a
feature of another point and so unfocus the item
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "The bench
layer", [`viewer-3d-bench-layer.md`](viewer-3d-bench-layer.md)).

---

## What the panel shows

### No reconstruction selected

`No reconstruction loaded`, centred, and no checkbox.

### No point selected

With *Edit* clear and no point selected on the selected node, the panel draws
its empty state: `No point selected` above a **Go to Point...** button, which is
where a person who has an ID and does not know where to enter it looks
([goto-point.md](goto-point.md)). Under the button one line names the item a
tick of *Edit* goes back to when there is one, *"Tick Edit to edit
pt3d_a1b2c3d4_1207 again, click a recent item beside it, or double-click an item
in the Scene tree's Bench groups."*; otherwise it says what the bench holds when it holds anything, *"3
items on the bench: double-click one in the Scene tree's Bench groups to edit
it."*, and otherwise names the pixel gesture: *"To work on a track from a
pixel: right-click it in the Image Detail panel and choose "Start cluster on
the bench here"."*.

A selected index the version at the cursor does not hold -- a point an edit
deleted, or an index past the end of the points -- is no selection here, and
the panel takes the same empty state, even though the base may still have a row
at that index ([document-model.md](document-model.md)).

### One body in two modes

Below the box the panel draws one body, `TrackBody`, in one of two modes
(`BodyMode`):

- **Viewed**, with *Edit* clear and a point selected: the **viewed track**, the
  selected point read as an editable track and held off every bench. Nothing
  about it can be changed.
- **Edited**, with *Edit* ticked: the focused item, and every step that acts on
  it.

Every column, tile, crop, reading and hover view is drawn the same way in both,
so a committed point and its copy on the bench are read in one layout with one
set of numbers. The differences are these:

| Part | Edited (the focused item) | Viewed (the viewed track) |
|---|---|---|
| Header label | the item's label, then the point ID when it differs | the point's portable ID, with copy and *Go to Point* |
| Header summary | stage, `N kept · K out · P pinned`; at the track stage error, track length, max pair angle, depth z and cond from the track's own position | the point's colour swatch, `xyzw` with *Copy coordinates*, error, track length, max pair angle, depth z, cond |
| Under the header | the stage's headline | how to change the track, or the item on the bench from this point |
| Toolbar | evaluation state; *Fit*, *Stage*, *Split*, *Duplicate*, *Commit*, *Discard*; *Lock*, *Rename* | evaluation state only |
| Threshold boxes | the item's bars; a release applies them as one version | the session's read-only bars; they judge the readings and the *Verdict* column and change nothing |
| Verdict column | *Keep*: switch and pin, tinted by what the bars propose | *Verdict*: `in` or `out`, with how many bars an `out` row fails, in a cell tinted green or red, the verdict the bars give the row |
| Heading pin | on the *Keep* heading | absent |
| Row click | select image, reveal, pick into the split selection | select image, reveal |
| Row double-click | camera view toward the observation | same |
| Row context menu | searches, *Accept walk*, unpin | absent |
| *From* column | provenance | absent: every row came from the point |

**The viewed track** is core's `create_track` applied to the selected point in
the version at the cursor, held in `AppState` rather than on the bench
([`bench.md`](bench.md) § "The viewed track"). It is built as a put builds a
bench track, labelled with the point's portable ID, so its rows arrive `in` and
pinned and the score against the bitmap (`plain_zncc`) is read back from the point's stored
confidence column, and the live evaluation keeps it evaluated as it keeps a
bench track, so every number in the table means the same thing in both modes.
It is never written anywhere: it is in no version, the Scene tree does not list
it, the bench layers do not draw it, and no step accepts it. Ticking *Edit* over
it puts the same point on the bench, and the bench track starts from what the
panel was showing. The panel is handed `&AppState` and cannot build it, so the
dock asks for it before drawing the panel (`AppState::refresh_viewed_track`).

**What Viewed mode reads.** Everything about the point, including its position,
colour and error, its whole track, each observation's keypoint, its patch placement
and stored bitmap, and its triangulation diagnostics, is read through the
version's overlay accessor rather than off the base, so a point an edit modified
shows the track it holds now.

**A cluster and a track are the same body at two stages** in Edited mode: the
header's headline, the table's cells and the row menu follow the stage, and
there is no separate cluster layout. What differs is outside the panel: a
cluster has no geometry, so the 3D viewer's bench layer draws nothing for it and
Image Detail draws its parallelograms rather than an outline, and no ghost
outline in the images it has no sighting in. A cluster has no committed point
behind it and so no Viewed mode, and the way back to a cluster after Edit has
been cleared is its chip in the recent items strip, a tick of *Edit* while no
point is selected, or its row under *Bench Clusters*. A gesture in Edited mode that
names no item means the focused item, and the panel shows the focused item and
nothing else on the bench: the bench as a list is the Scene tree's.

**Almost no state lives in the body.** The bench is the node's, at its cursor,
and the viewed track and its bars are `AppState`'s, so a step taken anywhere, in
this panel, in the Scene tree or by an undo, shows here on the next frame. What
the body owns is about looking rather than about the track: where the boxes
stand during a drag, whether *Lock* is ticked, the tiles and crops it has
rendered, the header's summary, and the bars' judgement of each row.

#### The header

In **Viewed mode** the header is a compact bar of the point's summary:

```
[RGB] pt3d_a1b2c3d4_12345 [copy][->] | xyzw: (1.234, -0.567, 2.891, 1) [copy] | error: 0.42px | track: 7 obs | max pair angle: 12.3° | depth z: 41.0 | cond: 12
```

| Field | Description |
|-------|-------------|
| Colour | A swatch of the point's RGB. |
| Infinity mark | ∞, left of the ID, for a point at infinity (`w` is `0`); absent otherwise. Its hover text says the point is a direction and its numbers a unit bearing. |
| Point ID | The copyable `pt3d_{hash}_{index}` ID, in monospace, with *Copy Point ID* and the *Go to Point* arrow beside it. |
| Position | `xyzw`, with *Copy coordinates*; *at infinity* follows the track length when `w` is `0`. |
| Error | The point's stored RMS reprojection error in pixels. |
| Track length | The number of observing images. |
| Max pair angle | The largest angle between any pair of observation rays, the main indicator of triangulation quality; shown when greater than zero. |
| Depth z | The inverse-depth z-score `depth / σ_depth`, a scale-free observability diagnostic that stays correct near infinity; shown when finite. |
| Cond | The condition number of the triangulation's normal matrix; shown when finite. |

In **Edited mode** the header is the focused item's label, its stage as a word,
and `N kept · K out · P pinned`. When the point the track was read from is
still in the version at the cursor, its portable Point ID follows the label with
the same two icon buttons, copy it and open *Go to Point*, so an ID copied here
is one that dialog takes back. A track put on from a point is labelled with
that ID until it is renamed, and the ID is then printed once, as the label. A
track whose point is gone says `from point N`, the index it had, and a track
with no origin says `new`. At the track stage the same summary follows, from the
error on, read from the track's own position and its `in` observations, so the
header reads the same either way: the error is the root mean square of the kept
rows' measured reprojection errors, `error: -` before a reading has measured
one, since a bench track carries no stored error; the track length counts the
kept rows; and the angle and the two diagnostics are computed from the rays of
the kept rows' cameras to the position, and left out where the track has no
position yet. A cluster has no position and shows none of them.

The metrics are [metrics.rs](../../crates/sfm-explorer/src/metrics.rs), at the
crate root, because the Image Detail overlay and `get_point` read the same numbers.
The summary is computed once per key and cached on the body, since the
diagnostics triangulate: for the viewed track per point and document serial,
for a bench track per track `Arc` and version.

**The Point ID** uniquely references the point across `.sfmr` files and
sessions, which a raw index cannot. It uses only `[a-zA-Z0-9_]`, so it selects
with one double-click in a terminal or a browser. It is minted over the node's
version graph by the rule **disk state first, earliest otherwise**: the hash is
the content the point sits in on disk when its identity reaches that version,
and the oldest content its identity reaches otherwise ([goto-point.md](goto-point.md)
§ "The ID forms and the version graph"), so in the ordinary case the ID copied
out of this header is one a reader of the file on disk resolves as it stands.
The body mints it through `crate::scene::point_id` **every frame**: an edit, an
undo or a save can change which content the ID names without the selection
moving. The format is the
[sfmr file format spec's](../formats/sfmr-file-format.md#point-id-portable-3d-point-references).
The copy button and the Go to Point arrow sit together because they are the two
halves of one round trip: copy an ID out of this header, paste it back here, in a
later session, or after `sfm xform` produced a new file.

**Under the Viewed header one line says how to change the track**: *"Tick Edit
to work on this track."* When an item on the bench came from this point (an item
whose origin, followed to the cursor, is this point; `AppState::bench_item_from_point`),
the line names it instead: *"On the bench as pt3d_a1b2c3d4_1207. Tick Edit to
open it."* The two can differ, and this is the only place a person looking at a
point learns that a changed copy of it is waiting.

**Under the Edited header is the stage's own headline**: at the cluster stage
the reference observation and whether a template has been cut, and at the track
stage the coordinate with the last triangulation's condition number, or the
sentence saying nothing has triangulated it yet. A track at infinity shows its
bearing without the condition number: that number measures the finite
triangulation, whose failure is what put the track on a bearing, so beside the
bearing it would read as a fit that failed its bar.

**The word in front of the coordinate says which coordinate it is.** A track at
infinity carries a unit direction where a finite one carries a place
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "Finite
points and bearings"), and the same three numbers under the wrong rule read as a
point a metre from the world origin. So the line is `Bearing (x, y, z), at
infinity` for a bearing and `Position (x, y, z)` otherwise, *at infinity* being
the Viewed header's word for the same thing. It comes from the track's own
`at_infinity` and not from its patch's `w`, so a point put on the bench from a
reconstruction with no patch placements reads as the bearing it is. A fit that
crosses the boundary changes the headline's first word, which is how a person
sees that it crossed. The same flag puts **an infinity mark** (∞, U+221E, which
egui's bundled fonts draw) left of the label, as the Viewed header puts one left
of its Point ID (`track_view::infinity_mark`), so the header's first glyph says
a direction before any number is read.

**The track's own patch stands at the left, under the header**, 64 points
square (`STORED_PATCH_SIZE`) with nearest filtering and no label, since the
picture says what it is. The line under the header, the toolbar and the
geometry search box stand to its right, and the table's separator runs directly
under it. At the track stage it is the patch bitmap
(`TrackPayload::bitmap`): for the viewed track that is the point's stored patch,
and for a bench track the bitmap a commit writes as the point's stored patch,
so a *Fit* that re-renders the track's bitmap changes it and a person sees what the commit
would store before committing. At the cluster stage it is the template every
member registers onto, once an evaluation has cut one. With neither -- a point
that stores no patch, a track with no bitmap rendered yet, or a cluster with no template --
the slot is an empty frame of the same size, so the controls beside it do not
move when a step fills it. Its hover text (`track_patch_hover`) says which it
is, or what would fill it; for a track-stage bitmap, that it is the render of
the reference observation, the row the *Reference* column marks, or the mean
of the views where there is none. Two bitmaps a bench track can hold are not
that (`patch::BitmapKind`):

- **A bitmap for judging** (`TrackPayload::bitmap_for_judging`), which the
  live evaluation renders where fewer than two `in` rows carry a keypoint, is
  drawn at a third of its brightness, in the slot and in the track's chip in
  the recent items strip alike (`patch::track_patch_image`). Its hover says it
  is not the track's patch: it was rendered from the rows with a keypoint, `in`
  or `out`, only for the bars to score the rows against, no row is its
  reference, a commit does not write it, and the first evaluation after two
  `in` rows carry a keypoint renders the patch.
- **A patch an unpin left pending its render** (`EditableTrack::bitmap_pending`)
  is drawn as it is, and its hover says it is kept only until the next render
  replaces it, that no row is scored against it meanwhile, and that a commit
  before that render writes it.

The bitmap is converted by `patch::stored_patch_image`: one channel repeated
across RGB, three as RGB, and a fourth, the confidence, dropped for an opaque
alpha, and an all-zero bitmap, which is how a point with no stored patch is
written, is not drawn. The upload is kept against the track's `Arc` and dropped
with the tiles.

#### The toolbar

In Edited mode, two rows. The first opens with where the focused item's evaluation stands, and
then acts on the track: *Fit*, *Fit Normal*, *Finite Diff Normal*, *Grid Plane
Normal* with their *per axis* and *overlap* boxes, the *Stage* toggle (which
names the stage it would move to), *Split off N rows*, *Duplicate*, *Commit* and *Discard*. The
second is the *Lock* box and *Rename*, which opens a field in place and commits
on Enter. Each entry is enabled, or greyed with a hover text naming what is
missing, and the refusal is the core step's own sentence asked of the very
track the button would act on, so the button and the step cannot disagree.
*Commit* asks the core commit; *Fit*, the two normal entries and the *Stage*
toggle ask `fit_preconditions`, `normal_preconditions` and
`set_stage_preconditions`, the halves of those steps' validation that read no
photograph. A track with no patch therefore greys both
with *"this track has no patch yet; fit it first"* rather than offering buttons
whose only act would be to decode a dozen images and fail. The commit refusal is
cached against the track's `Arc` and the node's version, since asking it builds
a point record.

In **Viewed mode** the toolbar is the evaluation state alone. The viewed track
is on no bench, so there is nothing for *Fit*, *Stage*, *Split*, *Duplicate*,
*Commit*, *Discard*, *Lock* or *Rename* to act on, and ticking *Edit* is the way
to them.

**There is no *Evaluate* button, because evaluation is live.** Every change to
an input of the evaluation evaluates the track again on a worker; the viewed
track is evaluated first while *Edit* is clear, and the focused item first while
it is ticked ([`bench.md`](bench.md) § "Live evaluation"). What the panel owes
the person is which state the numbers below are in, in either mode, and the
head of the toolbar says it: *Evaluated* when they are the
evaluation of the track as it stands; a spinner and *Evaluating…* while an
evaluation of the current inputs is running or waiting to start; and, when core's
`evaluate_preconditions` refuses the track or the evaluation failed, that
sentence in the warning colour -- *"Cannot evaluate bull-nose: the track carries
no patch to read against, as a point put on the bench from a sift_files
reconstruction has none; convert the reconstruction to embedded patches and put
the point on the bench again"* -- in place of any state.

**Commit leaves the point it wrote selected.** The write is
[`edits/commit-track.md`](edits/commit-track.md)'s; the panel's part is that the
point the commit produced becomes the selection, whether it replaced a point or
created one, so the viewport's track rays and the observing frustums look at what
was just written. A commit that pushes a version also makes the dock drop
what the panels cached about the node's points, since that describes a value
the node no longer holds. **A commit of a track the point already
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
re-render the bitmap, and read the result back, and it stays a button because it replaces
what the person placed. *Fit* greys with `fit_preconditions`' sentence for a
track stage with fewer than two `in` observations, while the evaluation still
runs for it, because one sighting is something to report.

**The three normal entries turn the patch and leave its centre.** *Fit* moves
the patch along the sightings' rays and keeps the way it faces; *Fit Normal*,
*Finite Diff Normal* and *Grid Plane Normal* do the opposite, each estimating a
normal and turning
the patch to it by the same least rotation the 3D viewer's arrowhead drag makes,
then reading the track back, rendering its stored bitmap and scoring every row
against it, as a fit does
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) §
"Estimating the normal"). *Fit Normal* takes the normal at which the `in`
sightings' tiles agree best. *Finite Diff Normal* fits a row of smaller square
pieces through the centre along each of the patch's two axes and takes the
plane through the two lines their centres land on. *Grid Plane Normal* fits a
grid of pieces tiling the whole patch and takes the plane through all their
centres. The two boxes after them say how both cut: *per axis*, from 2 to 8
pieces along each axis (2 by default), which is a row of that many on each axis
for *Finite Diff Normal* and that many by that many for *Grid Plane Normal*; and
*overlap*, from 0% to 90% of a piece's side (0% by default); they read
`2 per axis` and `0% overlap`. The version each of the two pushes, and its
Action Log row, names the cut it was made with: `Finite-difference normal of
<label> (3 pieces along each axis, 25% overlap)` for the rows, `Grid-plane
normal of <label> (3x3 pieces, 25% overlap)` for the grid. All
three entries run on a worker like *Fit*, push one version, and grey
with `normal_preconditions`' sentence at the cluster stage, at infinity, and
with fewer than two `in` observations; the boxes grey with them. The boxes are
tool settings, as *Lock* is: changing one is no step, and they keep their values
for the session.

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
as `TrackBodyResponse::pick_row` and the dock applies it through
`AppState::pick_bench_observation`. It is not a version: undo, redo, a jump and
a change of focused item clear it; a rename keeps it. *Split off N rows* takes it; a split names its
observations explicitly, because `out` says a sighting does not belong here and
cannot say which of two tracks it belongs to.

#### The thresholds

Six boxes for the eight bars of `Thresholds`, so every bar the stage the
track is in judges has a box. The two ZNCC boxes edit the bars of that stage
(`Thresholds::zncc_bars`): `min_zncc` and `min_zncc_middle` at the track
stage, `cluster_min_zncc` and `cluster_min_zncc_middle` at the cluster stage,
which judges another score; the other stage's pair has no box while the
track is not at that stage, and only the wire moves it then. Five of them judge the table's readings, and they stand in **a threshold
row directly under the column headings**, each in the column of the readings it
judges and followed by the unit and name those readings print with:

```
Img  Crop  Patch  Keep  ZNCC          Self-similarity           Proj. err  Shift     Zoom  Reference  Status  ...  Name
Thresholds              [65]% whole         [2.5] px whole      [3.0] px   [6.0] px
                        [0]% mid
```

The row is drawn above the scroll area with the headings, so a bar stays beside
the heading of what it judges however far the table is scrolled, and the
columns no bar judges leave room for the word *Thresholds* at its left. The
minimum ZNCC and minimum middle ZNCC boxes stack in the *ZNCC* column, whole
over mid, as its cell stacks the two readings. They read and take percent in
whole steps, as that column reads, so on a new track they show 65 and 0 at
the track stage while the track and the wire hold 0.65 and 0, and 70 and 70
at the cluster stage; a typed value may end in `%`. At the track stage both
judge the plain score against the stored bitmap, and at the cluster stage the
achieved template ZNCC. A middle bar of 0 turns it off, and a row with no
middle reading clears it.
The self-similarity box, `px whole`, is the largest self-similarity radius, in
patch-grid px, 2.5 on a new track, and takes `0` to `3` to one decimal. It
judges the whole tile's radius, the upper reading in the *Self-similarity*
column; `3`, the largest radius read, turns nothing out, and a row with no
reading clears it.
The shift box, `px`, is the maximum shift, in patch-grid px, the unit of the
self-similarity radius, and 6 on a new track. It is three things at once, since
they are one question -- how far from where a sighting is the correlation may
put it: the bar the *Shift* column is judged by, the radius the evaluation
looks for each peak within, so moving it evaluates the track again, and the
bound on how far a *Fit* may move a sighting
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "The
fit's walk is bounded by the person's bar"). There is no separate search radius.
The projection error box, `px`, is the largest reprojection error, in the
photograph's pixels, 3 on a new track, and takes `0` to `100` to one decimal.
It judges the px line of the *Proj. err* cell, at the track stage only, and `0`
turns it off; a track made by *Create Track Here* arrives with it off. Unlike
the other four it judges the point as much as the sighting: a mis-triangulated
point fails it on every row, so a track whose rows fail this bar alone has a
point that is off rather than bad sightings.
The text after each box has hover text saying what the bar is and what a row
past it is.

The sixth box, *geometry search min relative ZNCC (%)*, is
`geometry_search_min_relative_zncc`: the fraction of the track's own
self-agreement a photograph's ZNCC has to reach for *Find matches by geometry*
to add it. It judges no reading and no verdict, so it has no column; it stands
above the table, to the right of the track's patch, after the toolbar in Edited
mode and after the evaluation line in Viewed mode, as its label and a box. It
reads in percent, 70 on a new track.

Each box is a number box: dragging it left or right changes the bar (half a
percent per point for the ZNCC bars, 0.05 grid px per point for the shift, 0.02
grid px per point for the self-similarity radius, 0.05 px per point for the
projection error), and clicking it takes a
typed value. There is no slider rail beside it, since a rail would say nothing
the box does not.

**A box applies to the focused item when it is let go.** Dragging one
recolours the table live; releasing it sets the track's bars to where the six
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
verdict cell ([`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "Growing and judging"). For an unpinned row the proposal is what `apply_thresholds`, the
step a release applies, gives it, so a row can never be shown one way and turned
the other way when the box is released; for a pinned row it is what unpinning
it would give, which a release does not apply, since the painting leaves a
pinned verdict unchanged. Every row of the viewed track is pinned, so its
proposals are what unpinning each row would give, which is the verdict the
bench's own evaluation would give the row once the point is on the bench and
the row unpinned. The judgement is kept on the body against the mode, the
track's `Arc` and the bars, and recomputed when one of them moves rather than
per frame, because a copy of a track carries its patch bitmap. An
evaluation of the viewed track landing gives it a new `Arc`, which is one such
move.

**The boxes show the focused item's own bars** in Edited mode, copied from it
on every frame no box is being dragged, so whatever moved them -- a release
here, `apply_bench_track_thresholds` over the wire, an undo or redo of either,
another item focused -- the boxes follow. Only during a drag do they hold a
value the track does not.

**In Viewed mode the boxes are drawn and editable, and change only what is
drawn.** They hold the **read-only bars**, `AppState::viewed_thresholds`, which
are session state: they start at the bench's default bars, keep their values
across selection changes, and are not saved. Dragging a box or typing into it
moves the bar, and every judged reading and every *Verdict* cell is recoloured
live, as a drag recolours the table in Edited mode. The body reports the new
bars on every frame a box moves them (`TrackBodyResponse::viewed_thresholds`)
and the dock sets them (`AppState::set_viewed_thresholds`). Nothing else
happens, on the drag or on its release: the viewed track's verdicts stay `in`,
no version is pushed and no Action Log row is written, and the busy state does
not grey the boxes. So a person can set a strict bar and click through points
to see which observations each one would lose. Their hover text says that they
judge and change nothing, and that a point put on the bench from here takes
them: a put of the viewed point carries the read-only bars onto the new track
when they differ from the defaults ([`bench.md`](bench.md) § "The viewed
track").

#### The observation table

One row per observation, with the headings above the scroll area.

**The rows are ordered by a column**, increasing by *Img* when the panel opens.
Clicking a heading orders the rows by that column, worst first where a bar
judges the column and increasing where none does, and clicking the same
heading again reverses the order. Worst first is decreasing for *Keep* and
*Verdict* (the most bars failed), *Self-similarity*, *Proj. err* and *Shift*,
and increasing for *ZNCC*; *Img*, *Zoom*, *Reference*, *Status* and *Name*
start increasing. *Zoom* and *Reference* are among them because no bar judges
them. The
heading the rows are ordered by carries a small triangle after its word,
pointing up for increasing and down for decreasing, and a heading that orders
the rows is drawn in the plain text colour under the pointer. Its hover text
ends by saying what a click does with the order as it stands, and its accessible
name is *Sort by …*. The keys are what the rows print:

| Heading | Ordered by |
|---|---|
| Img | the image's index |
| Keep, Verdict | how many bars the row fails; among rows that fail the same number, the ones the bars propose `in` come first, so a row that fails none and is `out` because another sighting holds its image comes after the `in` rows |
| ZNCC | the whole patch's ZNCC, at the track stage the plain score against the stored bitmap |
| Self-similarity | the whole tile's radius |
| Proj. err | the reprojection error in px |
| Shift | the shift in px |
| Zoom | the geometric mean of the two zooms, `1 / sqrt(abs(det J))` |
| Reference | the reference in use first, then the rule's pick, then the rows turned away for sharpness, agreement, a ninth, the angle, clipping and coverage, in that order, so the rows nearest to being picked come first; an `out` row that is not the reference has no key |
| Status | the Status cell's text, compared character by character |
| Name | the image's file name, compared character by character |

*Crop* and *Patch* are pictures and *From* is provenance, so they order
nothing. A row with no key in the column -- a reading that is not there or is
`NaN`, a row nothing has measured, or every reading of a viewed track whose
evaluation was refused -- sorts after every row with one, whichever way the
order runs. Rows that tie stay in increasing order of image, then of
observation index. The order is a tool setting on the body, as *Lock* is: it
moves nothing on the track, pushes no version, is kept for the session across
tracks and modes, and is not saved with the layout. Selecting rows does not
depend on it, since a Shift- or Ctrl-click adds or removes one row.

**The table scrolls both ways.** The rows scroll up and down under the
headings and the threshold row, which stay in place. The table is wider than a
narrow dock cell, so it also scrolls sideways, and the headings and the
threshold row move sideways with the rows, placed from the left edge the rows
are drawn from on the same frame. The controls above the table, the header,
the toolbar and the geometry search box, fit the panel's width and do not
scroll. A trackpad, a wheel (with Shift for sideways) and the scroll bars move
the table, and so does a drag with the left or the middle button begun
anywhere on the rows, the headings or the threshold row except on a control.
The *Keep* switches and the pins take a drag begun on them, which does
nothing, so a drag that starts on a switch does not scroll the table and
cannot toggle a switch it passes over. A drag on the rows keeps moving briefly
after the button is released, as egui's scroll area does on a touch screen.
The scroll area drags on a touch screen only unless it is asked to drag always
(`ScrollSource::ALL`), and the headings and the threshold row are outside it,
so a drag of either is added to its offset by the panel.

The image's index is the first column, at the table's left edge, under
*Img*; the crop of the photograph around the patch's outline is the second,
under *Crop*; the rendered tile is the third, under *Patch*; and the verdict
column follows them, *Keep* in Edited mode and *Verdict* in Viewed mode. The
crop comes before the tile because it is the photograph as it is, and the tile
beside it is that patch warped square, so the eye reads from the raw pixels to
the picture the numbers are read from. The two photometric columns come
straight after the verdict, *ZNCC* and then *Self-similarity*, since they are
the readings the verdict is most often decided by; the reprojection error, the
shift, the patch's zoom, the *Reference* column, the status and, in Edited
mode, the provenance follow them. The image's name is the last column, 220 points wide: hovering it or the
*Img* cell shows the name whole, so the room in the middle of the table goes to
the readings. The columns stand at the same offsets in both modes, and Viewed
mode has no *From* column, since every row of a committed point came from the
point; *Name* moves left into the room *From* leaves. The headings are drawn at
the cells' own body size, in the weak text colour so they still read as
headings. Each heading has hover text over the width of its
column, running to where the next heading starts, saying what the column holds:
the *Crop* heading's says what the crop shows and what its hover view adds, the
*Patch* heading's what the tile is at each stage and that the numbers are read
from it, the *Keep* heading's says
what a kept observation is used for, when the thresholds set the switch, what a
click on the switch, on the pin and on the heading's own pin does, and what the
cell's colour means, and the *Verdict* heading's says what the column shows,
that nothing here changes a verdict of the point, and that ticking *Edit* is the
way to work on it. The *ZNCC* heading's says that at the track stage the
number is the row's tile against the stored patch bitmap, read plain, which
the bars judge, and that a value after `⏵` is the same pair read blur-matched.

**The *Keep* heading carries a pin**, in Edited mode, over the rows' pin column, and it
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
mean, and ends by saying that the box under the heading is the bar that judges
the whole tile's radius. The *Proj. err* heading's says what the error is measured to before and
after the track is triangulated, and that the degrees are the same residual as
an angle, and ends by saying that the box under the heading is the bar that
judges the px.
The *Zoom* heading's says which side of `1×` enlarges, that the zoom does not
wait for the photograph, what the two numbers of a range are, when the cell
prints `-` and what the column orders by.

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
| Patch (the tile) | the `R x R` grid the refinement kernel samples where the observation sits, at its shape; hovering it shows it in context | the patch re-rendered from this observation, re-anchored where it sits, through `patch::patch_color_image`; hovering it shows it in context, with the projection |
| Keep (Edited) | a switch, on for `in` and off for `out`, then a pushpin, solid on a verdict set by hand and a faint outline otherwise; each takes clicks over the whole height of the row; the cell is tinted by what the bars propose | same |
| Verdict (Viewed) | absent: a cluster has no Viewed mode | `in` or `out`, the verdict the read-only bars give the row, with the number of bars an `out` row fails in brackets (`out (2)`), in a cell tinted green or red by it; `-` untinted where nothing has measured the row, or on the pinned reference row the bars cannot judge until the bitmap is rendered again |
| Img | the image's index; hovering it shows the file name whole, as hovering *Name* does | the same |
| ZNCC | against the reference template, over the middle ZNCC: `92% whole` over `61% mid`, then the ZNCC grid | against the stored patch bitmap, read plain, with the blur-matched score after an arrow where the bitmap was blurred and the two print differently (`50% ⏵ 53% whole`), over the middle ZNCC, then the ZNCC grid; the reference's own row reads `100%`; `-` where the row has no score, the Status cell saying why; hovering the numbers shows the stored bitmap, the bitmap as blurred for the row and the row's tile side by side, with the plain and blur-matched scores under them, and a note where the localizer could not read the row |
| Self-similarity | the surface plot, then the tile's ZNCC self-similarity radius over its middle square's: `0.4 px whole` over `3+ px mid`, `3+` for the largest, then the self-similarity grid | the same |
| Proj. err | absent | the reprojection error: how far the keypoint sits from the point's projection, or, before the track is triangulated, from its patch's centre's, over the same residual as the ray angle, comparable across lenses and depths: `0.65 px` over `0.08°` |
| Shift | how far the refinement moved the member off its seed, in patch-grid px: `1.20 px` | how far the correlation peak against the render of the track's reference, looked for within the shift bar, sits from the observation's own keypoint, in patch-grid px on the patch's plane; `0` on the reference's own row |
| Zoom | `-` | patch-grid px, at the reconstruction's patch resolution `R`, per photograph pixel at the patch's centre, the reciprocals of the two singular values of the Jacobian there of the warp from the patch grid to the photograph, least over most, each to two significant digits: `0.71/1.3×`, and both numbers even where the two print the same, `0.19/0.19×`; `-` for a track with no patch yet, an observation with nothing saying where it sits, a patch whose centre is behind the camera or outside the camera model's domain, and a patch seen edge on |
| Reference | `-` | `reference` over the viewing angle and pair ZNCC (`24°, 87%`) on the reference in use, on a green cell where the reference-view rule picks it too, a red one where the rule picks another row, and a cell with no fill where the rule picks no row; `pick` on a grey cell for the rule's pick where it is not the reference; on any other row the test that turned it away, `partial`, `clipped`, `oblique`, `ninth differs`, `agrees less` or `less sharp`; an `out` row, which the rule does not consider, prints `-` over its angle; hovering the cell says what the mark means and how to accept the pick, then the reason and every reading |
| Status | the kernel's `member_status` | `walked 19 grid px (ZNCC 87% / 41% there), kept at seed` where the last fit refused to move it, the ZNCC being the plain score against the stored bitmap of the tile at the walked peak (left out where there is none); the reason's own sentence where the localizer could not read the row or the row has no score against the bitmap (`there is no bitmap to score it against`, or `the bitmap is to be rendered again before the row is scored` while an unpin leaves it pending its render); `localized` where the evaluation read it; `not evaluated` where nothing has been read |
| From (Edited) | the provenance | the provenance |
| Name | the image's file name elided in its middle to fit, the start of the path and the end of the file name both kept; hovering the name shows it whole | the same |

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
evaluation, since `.sfmr` stores the whole-patch score alone. Ordering by the *ZNCC*
column orders by the whole reading.

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
in patch-grid pixels, the tile can slide over itself and still match
itself as well as a true match between two views would: the semi-major axis of
the ellipse fitted to the shifts where its ZNCC against itself, interpolated
between whole-pixel shifts, is at or above that level. `whole` is the whole tile's and `mid` its middle square's, each to one
decimal (`0.4 px`, `1.4 px`), and `3+ px` for the largest radius searched,
which reads "3 or more". Beside them is the grid of each ninth of the tile read
alone: a box is green under `1`, yellow from `1` to `2`, orange from `2` to
`3` and red at `3` or more, and carries a dark
line along its ellipse's major axis where the ellipse is long and thin, its
elongation `1 − (minor / major)²` at least `0.5` (a minor axis at most 0.71 of
the major), so the ninth's matching shifts line up along one direction, as on an
edge. Hovering it shows
its nine numbers as the cell prints them. The self-similarity bar judges
`whole`; the middle and the grid are shown and judged by no bar.

**Hovering the two numbers shows the ellipses they are the semi-major axes
of**, in three units, under a sentence saying what the table is. The
measurement's `zncc_self_similarity_ellipse` and `_middle` carry them, measured
in sfmtool-core
([`../core/patch/zncc-self-similarity-radius.md`](../core/patch/zncc-self-similarity-radius.md)
§ "The ellipse in other units"), and the hover only prints them, a column for
`whole` and one for `mid`:

```
           whole                  mid
grid px    0.42 × 0.31 at 35°     3+ × 0.80 at 0°
image px   0.85 × 0.60 at 41°     5.6+ × 1.6 at 2°
world      3.1 × 2.2 mm at 145°   29+ × 7.8 mm at 0°
```

The table is built only while the cell is hovered; the row keeps the two
ellipses, not the text. Each cell is the semi-major axis × the semi-minor axis
and the angle of the major axis in whole degrees, `0°` to `179°`; a circle has
no direction and prints no angle.

- **grid px** is the ellipse in patch-grid px, whose semi-major axis is the
  radius the cell prints, its angle from the grid's `x` (right) towards its `y`
  (down).
- **image px** is the ellipse in the photograph's pixels, through the Jacobian
  of the patch grid at its centre at the reconstruction's patch resolution `R`
  (the edge of its patch bitmaps, else the evaluation's 24), the grid the
  reading was taken on, not the 64-texel display tile; its angle runs from the
  photograph's `x` towards its `y` (down).
- **world** is the ellipse along the patch, its angle from `u` towards `v`, in
  world space from the reconstruction's `metadata.world_space_unit`, with the
  unit after the two lengths. Every length in the row, both axes for both parts,
  prints in one unit, chosen from the largest of them so they compare directly.
  A metric scene (`mm`, `cm`, `m`) keeps its own unit where that puts the
  largest in [1, 1000), and otherwise takes whichever of `µm`, `mm` and `m`
  does, so `0.0031 m` prints `3.1 mm` and `cm` appears only in a scene in
  `cm`; a length over 1000 m or under 1 µm takes the nearer end, `m` or `µm`.
  A scene in `ft` prints in `in` while the largest is under 1 ft, and a scene
  in `in` stays in `in`. Where the file names no unit the row is labelled
  **scene units** and the numbers are bare and not converted; where the
  largest is under 0.001, every number prints in scientific form to two
  significant digits (`5.4e-4`). Only the hover text is scaled: the ellipses
  the table row keeps, and the MCP and Python readings, stay in the scene's own
  unit. A patch at infinity has no length, so the row is labelled **angle** and
  each semi-axis reads in degrees.

Numbers carry two significant digits, chosen after rounding. A `+` after a
number marks a lower bound, as the cell's `3+` does: the region at the level ran
off the square the reading searched, met a shift with no reading that could hide
more of it, or reached the largest radius searched, so the true length may be
larger (the rule is § "Lower bounds" of the self-similarity spec). A grid length
at the largest radius prints `3+` as the cell does. A value that cannot be
computed prints `-`: the world row at the cluster stage, which has no patch (its
image row is read through the seed shape), and the image row where a point
beside the tile's centre does not project. A row whose evaluation is refused or
failed, or that has no reading, has no hover. The hover adds to the cell and
changes nothing about it: its text, its colours, its bar and the column's
ordering are the radius's.

**The column opens with the tile's surface plot**, before the two readings:
the whole tile's ZNCC against
itself at every whole-pixel shift the radius searches, interpolated between
the shifts (Catmull-Rom, repeating the edge value past the square's edge)
and drawn over the whole square of shifts, with the contour at `1 - τ`, the
level the region is read at, over it. The colour ramp is keyed to that level so
the region inside the contour reads as one shape: below the level a muted ramp
from dark to mid slate, and at it a jump to bright amber that lightens towards
pale yellow at `1`. A dot marks each whole-pixel shift inside the contour and a
ring marks the centre, and the radius is the semi-major axis of the ellipse
fitted to the region inside the contour, read up to `r`. A
small ring round the centre is a patch that locks; a long region is a patch
that slides along it; a region that runs to the edge of the square is one that
may slide further than the radius looks. Hovering the plot draws it large
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
what the shift bar paints on. *Proj. err* is a statement about
the point: a mis-triangulated track shows a column of large errors beside a
column of near-zero shifts, the picture that says the position is wrong and the
sightings are not. The projection error bar judges it all the same, so such a
track is painted red down the *Proj. err* column, and its rows proposed `out`,
by that bar alone.

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
wanted to carry further than the shift bar kept its seed
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "The
fit's walk is bounded by the person's bar"), which a person reading `localized`
would get wrong, so the walk comes first among a scored row's answers. The
row's own *ZNCC* cell beside it is the evaluation's, taken with the sighting at its
seed, so the two numbers a person weighs the walk by sit on one row.

**The *Zoom* column says how much the patch magnifies the photograph.** It
sits after *Shift*. A patch that samples a few photograph pixels many times
over reads interpolation rather than detail, and one that spans a large
stretch of the photograph with few samples averages away detail the
photograph has, so the zoom says how much of what the patch holds is the
photograph's own detail. It is the Jacobian of the warp from the patch grid to
the photograph, through the placement the row's tile is rendered through, the
patch re-anchored where the observation sits (`patch::render_frame`), so it
describes the warp the tile shows and not a separately derived placement.

**The zoom is in patch-grid px at the reconstruction's patch resolution `R`**,
per photograph pixel: the grid the bench's shift and self-similarity readings
are in, so the zoom, the shift and the ellipse beside it are in one unit. `R` is
`crate::bench::patch_resolution`, core's `EvaluateOptions::patch_resolution`:
the edge of the reconstruction's patch bitmaps, which an `.sfmr` declares as
`patch_bitmap_resolution`, and where it stores none the evaluation's own
resolution, 24. The tile is drawn at 64 texels a side (`patch::PATCH_RES`) only
so that it looks crisp in its cell; no number in the table, a hover or a reply
is read at that resolution.

The zoom is pure geometry: the patch, the observation's camera and pose, and
where the observation sits. `tile::patch_jacobian` reads it with core's
`camera::warp_map::patch_grid_jacobian` at `R` grid px a side, with no photograph,
and the body caches the answer per row beside the tiles, dropping it whenever it
drops them (a step on the track, or the reconstruction leaving the scene). So a
row whose photograph is still decoding, or cannot be read, prints its zoom all
the same and keeps its place when the rows are ordered by *Zoom*, and the table
and the wire, which has no photograph to warp, always give the same numbers. A
patch whose centre projects has a zoom even where the middle of its tile is off
the photograph: what the zoom describes is the warp, which extends past the
photograph's edge.

When `R` is even the grid's centre is the corner the four middle grid px
share rather than a grid px. `patch_grid_jacobian` reads the Jacobian there as
the finite difference across the four points half a grid px either side of the
centre, the four middle grid px centres, each projected with no test against the image's
bounds (`CameraIntrinsics::project_homogeneous`): each of its columns is the
mean of the two differences along that axis, which is the derivative at the
centre of the bilinear interpolation through the four. The per-texel Jacobians
`WarpMap::compute_svd` stores for the mip selection are not used, for two
reasons: they are central differences about a texel, so none of them is at the
centre, which lies between texels; and they fall back to the identity where a
neighbour has no source pixel, which would print as a `1×` zoom.

There is no Jacobian, and the cell prints `-`, in each case where there is no
warp: a row at the cluster stage, whose tile is the refinement kernel's grid
rather than a warp map; a track-stage track with no patch yet; an observation
with nothing saying where it sits; and a patch whose centre is behind the camera
or outside the camera model's domain. A Jacobian whose smaller singular value is
at most `1e-9` of the larger, a patch seen edge on, has no zoom either: its
plane holds the camera, so the four points project onto one line and the finite
difference leaves a rounding residue across it rather than an exact zero, which
would print as a zoom of about `10¹³`. A real oblique view stays far above that
ratio; a patch at 89.9 degrees to the line of sight reads about `2e-3`.

*Zoom* is `1 / s` for each singular value `s` of the Jacobian, the least zoom
over the most, so over `1×` the patch samples the photograph more finely than
its pixels and under `1×` more coarsely, and two different numbers say the warp
stretches one direction more than the other, as on a patch seen at a slant. The
cell prints both numbers even where they print the same, `0.19/0.19×`, so every
row reads in one format. Each
number carries two significant digits, judged after rounding, as the
self-similarity hover's do: `0.031` prints `0.031`, `9.96` prints `10`, `0.996`
prints `1.0`, and a zoom of 10 or more prints whole, keeping every whole digit
from 100 up, so `123.4` prints `123`. The cell is geometry rather than a reading
of the evaluation, so it is drawn in the plain text colour whatever the
evaluation stands at, as the tile is, and no bar judges it. Ordering by *Zoom*
orders by the geometric mean of the two zooms, `1 / sqrt(abs(det J))`; a row
with no zoom sorts last either way, as a row with no key does in any column.
`get_bench_track` and `get_point`'s evaluation block report the zoom for each
row as `patch_zoom`, and the Jacobian it is read from as `patch_jacobian`, a
diagnostic the table does not print; `get_bench_track`'s `stage_data` reports
`R` as `patch_resolution`.

**The zoom also decides which sampler draws the tile.** The bench evaluation's
sampler choice (`crate::bench::sampler_choice`), the sampler rule by default
([image-warping.md](../core/camera/image-warping.md) § "Choosing the sampler
per view"), reads the same Jacobian at `R` and renders the row's tile and its
hover view with the anisotropic sampler where one mip level for both axes would
read the less compressed axis at least 1.5 times too coarsely
(`PatchJacobian::sampler`, `tile::tile_sampler`), and with `bilinear_mip`
otherwise, so a view is drawn with the sampler the bench's evaluation and the
stored bitmap's render use. Hovering a *Zoom* cell names the sampler and why the
choice picked it (`table::zoom_sampler_text`): under the rule, the loss against
the rule's threshold, or that the view is compressed less than √2 along both
axes; under a fixed choice, that the bench renders every view with that
sampler. The *Zoom* heading's hover describes the same choice
(`table::zoom_tip`). `get_bench_track` and `get_point`'s evaluation
block report each row's `sampler` (`anisotropic` or `bilinear_mip`) and
`sampler_minor_axis_loss`, both null where `patch_jacobian` is, and the loss
null as well where it is not finite.

**The *Reference* column marks the reference and the reference-view rule's
view of it.** The reference is the row the track's patch bitmap is rendered
from (`TrackPayload::reference`, where the track has a bitmap;
`crate::bench::reference_in_use`), and the rule's pick is the row the
reference-view rule picks from the current readings
(`crate::bench::reference_view_pick`). While the reference's row is pinned
every render keeps it, so the two can differ ([`bench.md`](bench.md) § "The
reference"); they also differ on an unpinned reference row between an unpin
that hands the reference on and the render, and where the live evaluation's
pick alternated between rows. The marks (`reference::ReferenceRows`,
`ReferenceMark`):

- **The reference is the rule's pick:** its cell reads `reference` on a green
  fill.
- **The rule picks no row:** the reference's cell reads `reference` with no
  fill (`ReferenceMark::ReferenceWithoutPick`), since there is no pick to
  accept in its place.
- **It is not:** the reference's cell reads `reference` on a red fill, and the
  pick's own cell reads `pick` on a grey fill. Unpinning the reference's row,
  or *Set as reference* on the pick, makes the pick the reference, and the
  hover of each of the two cells says so, naming the other row by its image.
  Where the reference's row is already unpinned, the hovers say instead that
  the pick becomes the reference at the next evaluation after a step (an
  evaluation that stopped because the pick alternated between rows leaves it
  until the next step), and the reference's hover that pinning its row keeps it.
- **No reference:** where the bitmap is the fused mean of the `in` rows, the
  render of no row, is a bitmap for judging, or the track has no bitmap yet,
  only the pick is marked, `pick` on a grey fill. The pick's hover says which:
  for a bitmap for judging, that fewer than two `in` rows carry a keypoint and
  the bitmap is rendered from the rows with one, in or out, and never
  committed; for a fused mean, that *Set as reference* on the pick renders the
  bitmap from it, and either that the next evaluation that renders renders the
  bitmap from the pick and makes it the reference or, where the rule reached
  the pick only through its last fallback, that the mean stays
  (`ReferenceRows::pick_by_last_fallback`). So on a point stored at `-1` the
  pick is grey only until the first evaluation that reads a pick the rule
  reached other than through its last fallback, and from then on it is the
  green reference; with a pick reached only through that fallback it stays
  grey ([`../core/bench/editable-track.md`](../core/bench/editable-track.md) §
  "The stored bitmap's reference"). The same holds for the viewed point's
  evaluation, which renders the same way.

A track-stage evaluation runs the reference-view rule over the `in`
rows ([`../core/patch/reference-view.md`](../core/patch/reference-view.md)): a
candidate has at least 99% of its tile on the photograph, at most 5% of the
photograph under the tile clipped, a viewing angle of at most 65°, and no ninth
where it agrees with the other rows more than 0.3 below the track's **typical
agreement** there, the median over the `in` rows; of the candidates whose pair
ZNCC, the median of its ZNCCs with the other `in` rows, is within 15 points of
the best candidate's, the rule picks the one with the smallest self-similarity
radius. The agreements are read plain. When no row passes, the rule drops the
65° limit, then the check of the ninths, then coverage and clipping; a row
that sees the patch at 90° or more, edge on or from behind, is never picked.
Every other row's first line is the word for the first test that turned it
away, read against the rule's pick: `partial`, `clipped`, `oblique`,
`ninth differs`, `agrees less` or `less sharp`. The hover of a `less sharp` row with no self-similarity radius
says the rule could not compare its sharpness, rather than naming a sharper
row. The column says *ninth* for a cell of the ZNCC grid, since a *cell* in the
table is one row's entry in one column. Its second line is the viewing angle, the angle
between the patch's normal and the direction to the camera at the keypoint, and
the pair ZNCC the rule read, in percent.
Hovering the cell gives, on a marked row, what the mark means and how to
change it, and then (`reference::reference_hover`) the reason in a sentence
with the row's own reading against the threshold that goes with it, the tests
the rule dropped when no row passed them, and every reading: the viewing angle
and the tilt direction (the direction in the patch's plane the ray from the
camera leans along, from `u` towards `v`), the coverage, the clipped share,
the pair ZNCC, the cell deficit, and the pair ZNCC of each ninth. The cell
prints `-` at the cluster stage, before the track is evaluated, and where it
could not be evaluated. Where the rule reached its pick only through its last
fallback, a render the rule decides stores the fused mean of the `in` rows,
naming no row
([`../core/patch/reference-view.md`](../core/patch/reference-view.md) § "The
stored bitmap"). On a file with patch placements but no stored bitmaps, the bitmap
is the one SfM Explorer rendered for display when it opened the file, and the
reference is the row it was rendered from: the file's reference observation,
or the display render's own pick for a point the file stores at `-1`, held by
its row's pin like any other.

**The *ZNCC* column reads the stored bitmap at the track stage.** A row's ZNCC
is its tile's windowed ZNCC with the track's stored patch bitmap, read plain
over the samples both have (`TrackMeasurement::plain_zncc`), and the middle
reading and the ZNCC grid are read from the same pair (`plain_zncc_middle`,
`plain_zncc_grid`)
([`../core/patch/blur-matched-zncc.md`](../core/patch/blur-matched-zncc.md)
§ "Scores against the stored bitmap"). Where the bitmap was blurred to the row's
sharpness (the bitmap sharper than the row's tile along every direction by the
ratio of 1.25) and the blur-matched score prints differently, the first line
carries it after an arrow, `50% ⏵ 53% whole`; otherwise it is one number. The
reference's own row reads `100%`, which is not computed. The bars judge the
plain score, and the cell and the grid are coloured by them. The bars are not
meant to turn away an out-of-focus view: a blurred view of the right place is one to keep, and
the plain score is the one the bar was measured on
([`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "Parameters"). A row with no score prints `-`, its Status
cell says why, and the bars leave its verdict where it is. At the cluster
stage the ZNCC is the member's against the reference's template.

**Hovering the numbers shows the comparison behind the score**
([`zncc_hover.rs`](../../crates/sfm-explorer/src/track_view/body/zncc_hover.rs)),
at the track stage. A line says these are scores against the stored patch
bitmap and that the min ZNCC bars judge the plain ones. Under it are three
tiles side by side, each 96 points across, the size the tile's own hover view
draws the tile at: a third of its 288-point picture, which spans three times
the patch's width:

1. the stored bitmap, the reference's render, captioned *Stored bitmap*;
2. the bitmap as blurred for this row, captioned with the blur's width in grid
   px (*Blurred by σ 0.83 grid px*). It is blurred by the row's
   `bitmap_blur_sigma` with core's kernel (`TilePlanes::blurred`), the one the
   blur-matched score was read on, so the picture is the bitmap the score
   compared. Where the pair was read unblurred, a note takes its place and
   says why: the row's tile is sharper than the bitmap along every direction,
   so it could replace the reference, or the bitmap is not sharper than the
   row's tile along every direction by the ratio of 1.25;
3. the row's own tile, captioned *This row*.

Under the tiles a monospace table sets the plain and the blur-matched scores
side by side for the whole tile and the middle, in percent to one decimal, and
then each ninth as two 3×3 grids of whole-percent numbers side by side
(*ninths, plain* and *ninths, blur-matched*). The reference's own row shows the
bitmap alone and says its score is 1 (100%) and is not computed. A row with no
score shows no pictures and gives the reason in a sentence (*No score against
the stored patch bitmap: there is no bitmap to score it against.* before the
first render, *the bitmap is to be rendered again before the row is scored*
between an unpin that hands the reference on and the render). Where the
localizer could not align the row to the reference's render (no
`seed_shift_px`), the hover ends with the sentence that it could not, so the
bars do not judge the row. The blurred bitmaps are uploaded the first time a
row's hover needs one, cached per row, and dropped when the track moves, as
the tiles are.

**The tile is the column the numbers are about.** A ZNCC is a number; the
picture that produced it is what a person can judge. So each row draws what its
stage registers, through the code that registers it: at the track stage the
patch warped into this view and re-anchored where the observation sits
(`patch::patch_color_image`, `WarpMap::from_patch` and the sampler the rule
picks for the row, at
64 by 64, shown at 48 points with nearest filtering; the 64 is a display
resolution, `patch::PATCH_RES`, and no reading is taken from the tile); at the
cluster stage the
grid the refinement kernel samples
(`sfmtool_core::patch::cluster_refine::sample_member_grid`) at that place and
shape, over the cluster's own radius
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "The
cluster stage's units"), on the template's resolution once one has been cut. A
row with nothing to render draws an empty frame of the same size, so the columns
beside it never shift. A patch not visible in a view warps to an all-black tile,
which is drawn as such.

**Under `bilinear_mip` a track-stage tile is one bilinear sample per texel from
the mip level the warp picks there.** This is how every row the sampler rule
does not move is drawn. The warp map's SVD is computed (`WarpMap::compute_svd`), and
each texel reads the level `round(log2(s_major))` of the photograph's pyramid,
`s_major` being the larger singular value of the warp's Jacobian at that texel,
clamped to the pyramid's levels (`remap_bilinear_mip`). The pyramid is the one
the photograph cache builds at the decode. The level is chosen texel by texel,
and it is above 0 only where a texel shrinks the photograph by more than about
1.4 times (`s_major` of `sqrt(2)` or more). Such a texel reads a level averaged
down towards its own sampling instead of picking scattered full-resolution
pixels, which would alias a fine texture into a pattern the photograph does not
have. Where no texel spans more than about 1.4 photograph pixels every texel
reads level 0, and the tile is exactly plain bilinear. A row the rule moves is
drawn with the anisotropic sampler instead, whose level follows the smaller
singular value and which averages several samples along the more compressed
direction. The choice is made at `R`, not at the display resolution, so it is
the bench's choice for the same observation. The hover view renders
its wider patch the same way, with the row's sampler and at the tile's
sampling, so each texel picks the level the tile's own texel there does and the
tile is still the middle of the picture texel for texel.

**The track-stage tile's placement is re-anchored on the observation's keypoint**
(`OrientedPatch::anchored_at_keypoint`, [patch-cloud.md](../core/patch/patch-cloud.md)):
the patch centre slides within its own plane, on its tangent sphere for a point
at infinity, until it projects onto the keypoint this observation carries. The
tile is a photometric comparison, so it is anchored where the keypoint localizer
aligned the pixels rather than at the point's geometric projection. The tiles of
a well-localized track then match each other whatever discrepancy the geometry
carries, and that discrepancy is what the *Proj. err* column reports: a column
of matching tiles beside a large error is a well-aligned patch whose position or
pose is off, not a bad match. The stored placement is used as it is when the
observation has no keypoint or the keypoint's ray cannot meet the patch
(`patch::render_frame`).

**Where the observation sits is one rule** at either stage: the track-stage
keypoint, else the refined cluster position, else the seed it was proposed
at (`crate::bench::observation_site`, which the marks, the reveal and the wire
read too). A row a search has just added carries only that seed, and its
tile is cut around it, so nothing has to be evaluated for a fresh row to show its
patch. The rendered tiles are kept against the track's `Arc` and rebuilt when a
step moves it.

**The photographs** come from `AppState::photographs`, the viewer's photograph
cache ([../core/camera/photograph-cache.md](../core/camera/photograph-cache.md)),
shared with the Image Detail panel and read by the evaluations, so no image is
decoded more than once while the cache holds it. An entry is the
`ImageU8Pyramid` built when the image was decoded, whose level 0 is the
photograph: the tiles, their hover views and the photometric readers sample the
lower levels where a warp shrinks the photograph. A failed decode is
remembered so a missing file is not reopened every frame. Before the body
draws, the dock asks for every image the drawn track observes: the focused
item's in Edited mode, and in Viewed mode the selected point's, when the
reconstruction carries patch placements, with the SIFT cache filled for them on a
`sift_files` reconstruction. Asking (`state::display_photograph`) starts a
decode on the rayon pool for each photograph the cache does not hold, and
repaints when it lands; the body only peeks at the cache, and a row whose
photograph is not decoded yet keeps no tile or crop, rather than a remembered
empty one, so they appear on the frame after the decode's repaint. The cache is
bounded by its byte budget, which drops the least recently used photographs
first.

**Hovering a tile shows it in context.** The tooltip draws the same picture
over three times the patch's width (`tile::CONTEXT_FACTOR`), 288 points across,
so a person can see what surrounds the patch and whether it sits on the surface
they meant. The wider picture is rendered at the tile's own sampling, so the
tile is its middle third texel for texel: at the track stage the tile's placement,
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
  projected pixel is read onto the widened placement's plane
  (`OrientedPatch::keypoint_plane_offset`, the reading that anchors the tile),
  so the ring sits on the part of the photograph the picture shows at that
  pixel;
- a dashed line from the dot to the ring.

The marks are egui shapes over the picture rather than pixels written into it,
each a light stroke over a wider dark one so that it reads over bright and dark
photographs alike, the ring in amber. They are clipped to the picture, so a
projection outside it shows as the line leaving the edge towards it. The
tooltip opens with the photograph's full name, the text the *Name* column's
hover shows, and under the picture short bullets say what each mark is and how
far the projection is, in the photograph's pixels, or that a cluster has no
point to project. At the cluster stage there is no ring and no line.

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
observation sits (`crate::bench::geometry::anchored_frame`, the placement the tile
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
crop is not boxed: the outline already shows where it is. As on the tile,
the tooltip opens with the photograph's full name, and short bullets under the
picture say what the outline and the view are, with the crop's size in
photograph pixels, what the marks are and how far the projection is, in the
bullet the tile's hover view uses (`tile::projection_bullet`), and the patch's
two axes, the one across the tile and the one up it, as lengths in the
photograph's pixels. At the track stage each axis is measured
along its projection through the lens, from one edge's midpoint through the
centre to the opposite edge's, so a bent axis is measured along its bend; at
the cluster stage it is the shape's column times the template's width. Two
last bullets give the pixel the observation sits at and its feature index
(`crop_caption`): the `.sift` feature a row put on by index names, or, for a
row read from the point the track came from, that point's feature in the same
image; on a reconstruction that stores its keypoints there is no `.sift` feature
behind them, and the index given is the observation's place in the point's
track. A row a search, the view sweep or a hand placed has no feature index,
and the line says which of them placed it. The
crop and its hover view are cached per row beside the tiles and dropped with
them; like a tile, a crop whose photograph is not decoded yet is not cached,
so it appears once the decode ends. The hover view is rendered the first time the pointer rests on the
crop. Its hover region takes no click, so a click on the crop is the row's.

**Each judged reading is coloured by its bar.** Five readings are judged, one
per bar: the *whole* line of the ZNCC cell by the minimum ZNCC, its *mid* line
by the minimum middle ZNCC, the Shift cell by the shift bar, the *whole*
line of the Self-similarity cell by the self-similarity bar, and the px line of
the Proj. err cell by the projection error bar. A reading that clears its bar is drawn
green and one that does not red, in a green and a red chosen for each of the
dark and light visuals so they read as text on the panel's background. Each line
of a two-line cell takes its own colour, so a whole ZNCC can be green over a red
middle one. Everything no bar judges keeps the plain text colour: the degrees of
*Proj. err*, the self-similarity *mid* line, Status, From, the *mid* ZNCC while its bar is
off at 0, a reading that is not there (`-`, though it clears its bar), and every
reading of a row nothing at this stage has measured or whose evaluation was
refused. A `NaN` reading fails its bar and is red. While an evaluation is on its
way the colours fade with the numbers, as the grids do, so a stale number does
not read as judged. The colours follow the boxes live during a drag.

**The *Keep* cell's colour is what the bars say, and the switch is what the
person decided.** The switch-and-pin cell, the full height of the row, is
tinted green where the bars propose `in` and red where they propose `out`, and
left untinted for a row nothing has measured, and for the pinned row holding
the reference where unpinning it would render the bitmap again from another
row: that row scores 1 against its own render, which says nothing, and the
bars can judge it only against the bitmap the unpin leads to, so its hover
says that rather than that nothing has measured it; the rest of the row carries only
the selection and hover highlights. For an unpinned row the proposal is what
applying the bars makes it. For a pinned row it is what unpinning it would
make it: its bars, and whether its image is free -- not already held by a
pinned `in` sighting of the same image, or by an unpinned one the painting
takes because it scores better. A switch that is on in a red cell, or off in a
green one, is a verdict set by hand against what the bars propose. The switch's hover text says what
the switch and the pin say, then why the bars propose what they do: that the
row clears every bar, or which bars it fails, named as the headings name the
readings (`ZNCC whole is under the bar`, `ZNCC mid is under the bar`, `Shift is
over the bar`, `Self-similarity whole is over the bar`, `Proj. err is over the
bar`), or, for a row that
clears every bar and is still proposed `out`, that another sighting in the same
image is kept, naming the image.

**In Viewed mode the *Verdict* column takes the *Keep* column's place.** A
committed point's observations are all in its track, so a switch there would
say the same thing on every row, and nothing in Viewed mode could change it.
The column shows instead what the read-only bars say about each observation:
the word `in` in a green cell where the row clears every bar and holds its
image, `out` in a red cell where it does not, and an untinted `-` where nothing
has measured the row yet, where the viewed track's evaluation was refused or
failed, or on the pinned row holding the reference where unpinning it would
render the bitmap again from another row (its hover says the bars cannot judge
it until that render). It is the proposal the *Keep* cell is tinted by, in the same green and
red, which for the viewed track's pinned rows is the verdict the bench's own
evaluation would give each row once the point is on the bench and the row
unpinned. Its hover text is the *Keep* switch's second half: that the row
clears every bar, which bars it fails, named as above, or which other sighting
holds its image. An `out` row that fails bars says how many after the word,
`out (2)`, which is the number the column is ordered by; an `out` row that
fails none says `out` alone, since what turned it out is another sighting of
its image. The cell takes the pointer for its hover text and no click, so
a click on it is the row's. There is no switch, no pin and no heading pin.
*Keep* has no room for the count beside its switch and pin; the red readings
and the switch's hover text say which bars a row fails.

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
nobody has ruled on is an unpinned `out`. While the node is busy or its bench
is view-only, the switch and the pin are drawn greyed, at the opacity of a
disabled widget and with a switch that is on filled grey rather than green, and
a click on them does nothing.

**The pin beside the switch says whether a hand set the verdict**, and is the
control for it: a solid pushpin on a pinned verdict, which the thresholds leave
alone, and a faint outline on one they set. Clicking a solid pin unpins the
verdict and gives the row the one the bars propose; clicking an outline pins
the verdict as it stands. A pushpin says "pinned" where a dot said only that
something was marked, and the control sits where the state is shown.

#### Row gestures

In both modes:

- **Click a row**: select its image, which the 3D viewer, the Image Browser and
  Image Detail follow, and **reveal** the observation: its pixel, where the
  Image Detail bench layer draws its mark, rides along in `reveal_feature`, and
  Image Detail pans (never zooms) to centre it when its current view is not
  showing it ([multi-panel-image-browser.md](multi-panel-image-browser.md) §
  "Revealing a feature named by another panel"). In Edited mode the click also
  takes the row into the split selection; Ctrl-click or Shift-click extends the
  selection. The viewed track has no row selection, since there is no step it
  could be taken to.
- **Double-click a row**: enter camera view for its image, and turn the view
  until the row's observation is inside the middle 1/2 of the viewport on both
  axes (`Viewer3D::look_through_toward_feature`), in one animated transition. A
  double-click on a frustum or a thumbnail enters camera view the same way but
  names no feature, and looking straight through the camera can leave the
  observation near the edge of a photograph wider than the viewport, or off it;
  the row names an observation, so the view it opens shows where it is. An
  observation nothing has placed yet has no pixel, and its double-click enters
  camera view without the turn.
- **Hover a row**: set the cross-panel hover, which brightens the frustum and
  outlines the thumbnail. The body produces no point hover, since every row is
  about the one track, so owning the pointer clears it.
- **Copy Point ID**, **Copy coordinates** and **Go to Point**, in the header,
  and in the empty state *Go to Point...*: copy the ID or the coordinates
  (`1.234, -0.567, 2.891, 1`), or open the dialog.

In Edited mode only:

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
  plain ZNCC pair against the stored bitmap (whole, then middle) at the seed
  (`plain_zncc`) and at the walked peak (`walked_plain_zncc`), two readings against the same
  bitmap that compare. The step is core's
  `sight_observation` at that pixel (`AppState::accept_bench_walk`), so the
  observation is pinned and the measurements read at the seed are dropped, the
  walk's among them; one version, labelled `Accepted the walk of observation 3
  of pt3d_a1b2c3d4_1207: moved 11.2 px to (1050.8, 1702.4) in IMG_0042.jpg`,
  and one `Bench` row. Greyed with the busy sentence while the node is busy. The
  wire's form is `sight_bench_observation` with `walked_to` as its pixel.
- **Set as reference**, in the same menu on a track-stage row that is `in` and
  has a keypoint, the rows core's `set_reference` accepts, and on no other row
  (`table::set_reference_offer`): make the row the track's reference, the row
  its patch bitmap is rendered from, and pin it, so it stays the reference
  whichever row the reference-view rule picks. The step is
  `AppState::set_bench_reference`, one version and one `Bench` row (`Set
  IMG_0042.jpg as the reference of pt3d_a1b2c3d4_1207, in place of
  IMG_0040.jpg`); it drops the bitmap, and the live evaluation that follows
  renders the bitmap from the row and scores every row against it
  ([bench.md](bench.md) § "The reference"). Greyed on the reference the track
  holds on a pinned row, with a hover that says so, and with the busy sentence
  while the node is busy. The wire's form is `set_bench_track_reference`.

A row is also selected from **outside** the panel: a click on a mark of Image
Detail's bench layer or on an observation circle of the 3D viewer's selects that
observation's row, replacing the selection as a plain click does
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "The bench
layer", [`viewer-3d-bench-layer.md`](viewer-3d-bench-layer.md)). The 3D viewer
draws the circle of the one selected row larger.

Track View asks the node to look for its index files, the SIFT index and the
cluster-patches file ([`index-files.md`](index-files.md)), whenever it draws, in
either mode, and to re-derive their states when the node's image table, or the
index the cluster patches are judged against, has moved. A session so finds the
files the last one built without anyone asking, and a look that found nothing
is remembered, so a node with neither file is not checked on every frame.

#### A point with no patch placement

A reconstruction whose points carry no patch placement, which is the usual case for
a `sift_files` reconstruction imported from COLMAP, is drawn as any other, and
what that shows is less than the rest of this section describes. `create_track`
builds a track-stage track with no placement, and core's `evaluate_preconditions`
refuses it, so the toolbar shows the refusal sentence (*"Cannot evaluate
pt3d_…: the track carries no patch to read against; …"*) and every
number cell reads `-`, the reprojection error and ray angle included, and every
*Verdict* cell `-`. The *Crop* and *Patch* cells are empty frames, and there is
no crop hover view to carry the pixel and the feature index. The header still
carries the point's summary (its colour, `xyzw`, error, track length, max pair
angle, depth z and cond), which reads the point and needs no placement. So for
such a point the panel shows its header and no per-observation readings. Which
readings an evaluation without a placement should produce and what the *Crop*
cell would show without one are not decided here (§ "Non-goals").

**Such an item is not edited here.** A `sift_files` reconstruction's bench is
view-only ([`bench.md`](bench.md) § "A view-only bench"): every toolbar button
but *Discard* and *Rename*, the threshold boxes, the *Keep* switches and pins
and the row menus' editing entries are greyed and take no click, with the
sentence that names Convert to Embedded Patches as their hover. After the
conversion, putting the point on the bench again (*Edit*, *Edit on Bench* or a
double-click) rebuilds the item with the point's new patch placement, and from
then on it is edited like any other.

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
with *"No track is being edited: tick Edit in Track View, click a recent item
beside it, or double-click a Bench item in the Scene tree."*

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
  which is the Edit box's own rule, and pushes no version; a `label` other
  than the one the item has is refused, naming it.

The whole bench family is [`bench.md`](bench.md) § "The wire" and
[`mcp-server.md`](mcp-server.md).

---

## Testing

The panel is tested headless. A test draws it through egui's
`Context::run_ui` with no window, asks for the viewed track before each frame as
the dock does, and applies what the panel reports through the `AppState`
methods the dock calls, so one test covers a gesture and the step it leads to.
The body records what the table drew on every frame, not only under test, so an
assertion about a row reads the table the app draws. The hover views are pure
functions of the track, the observation and the photograph (`tile::context`)
that return the box, the keypoint and the projection in the picture's own
texels, so their geometry is checked without reading pixels off the screen.

- **The box, the empty state and the recent items strip**:
  [track_view/tests.rs](../../crates/sfm-explorer/src/track_view/tests.rs).
  These tests establish the rules of § "The Edit checkbox" and § "Transitions"
  that the box itself carries out: the mode follows the focused item, including
  a focus or an unfocus made outside the panel with no call into it in between;
  a tick over a point puts it on the bench in one version, and a second tick
  after a clear focuses the same item with no version; a clear writes one
  `Selection` row and no version, and the next frame shows the point in Viewed
  mode; a tick with no point selected goes back to the most recent item on any
  node, passing over a discarded one, and the hover text and the empty state
  name that item first; with nothing to go back to the box is greyed, its
  refusal names the three ways in, and a click on it reports nothing; and a busy node greys the box only over a point
  not on the bench. They also cover the three forms of the empty state's line
  and its *Go to Point...* button (§ "No point selected"), a deleted and an
  out-of-range point taking the empty state, a discard leaving Edited mode, a
  cluster drawn with the cluster headline, another point selected while editing
  drawn in Viewed mode with no version, and no box with no reconstruction. For
  the strip (§ "The recent items strip") they cover the order, the label cut in
  its middle, how many chips a width holds, a click focusing the item and
  selecting its node and origin with no version while a background task holds
  the node, every line of the hover text, a discarded item's chip hidden and
  brought back by the undo, and a chip keeping its place under its new label
  after a rename. A chip's patch is uploaded once and drawn from the same
  texture on the next frame, and a cluster with no template draws an empty
  slot and its hover text says the item is new. That `set_editing` reads a
  selected point the version at the cursor no longer holds as no point
  selected, as the box does, so that a tick goes back to the most recent item
  with no version rather than putting the missing point on the bench, is
  tested in [bench/tests.rs](../../crates/sfm-explorer/src/bench/tests.rs)
  § "The focused item".
- **The selection rules**:
  [bench/tests.rs](../../crates/sfm-explorer/src/bench/tests.rs) § "The
  selection rules". These tests establish the invariant of § "Transitions" for
  every gesture `AppState` decides rather than the panel: which selections keep
  the item focused and which unfocus it, with one `Stopped editing` row and no
  version; that a focus, every put, a clear and a discard select what § "Why each
  transition is the one it is" says they select, a focus of an item with no
  origin clearing the point and keeping a selected image of that node; that an item stays focused
  through the deletion of its origin, through a commit, which gives a duplicate
  its origin, and through the undo of that commit, with no point selected; that
  an undo past the put unfocuses and leaves the selection as it was; and the
  recent list's order, skipping and pruning. The same rules over the wire are
  in [mcp/tests/bench.rs](../../crates/sfm-explorer/src/mcp/tests/bench.rs).
  That a press on a bench-layer handle selects no point (§ "The selection and
  the focused item") is tested in
  [image_detail/tests.rs](../../crates/sfm-explorer/src/image_detail/tests.rs),
  for the ghost's centre over a feature of another point, and in
  [viewer_3d/tests.rs](../../crates/sfm-explorer/src/viewer_3d/tests.rs), for
  the dot, a corner and an edge.
- **Viewed mode**:
  [body/tests/viewed.rs](../../crates/sfm-explorer/src/track_view/body/tests/viewed.rs).
  These tests establish what § "One body in two modes" gives Viewed mode and
  withholds from it: the headings of Edited mode with *Verdict* in place of
  *Keep* and no *From*, no toolbar button and no *Lock*, and a click on a
  *Verdict* cell belonging to the row and stepping no verdict. They establish
  that each *Verdict* cell is the verdict the read-only bars give the row,
  tinted to match and naming the failing bar in its hover text, and `-`
  untinted on a row nothing has measured; that a drag of any of the five boxes
  recolours the readings and the *Verdict* cells and changes nothing else, with
  no version and no Action Log row; and that the bars survive a change of
  selection. They also cover the header and its summary, which a bench item
  shows at the track stage and not at the cluster stage, the go-to button, the
  infinity mark,
  the line under the header in both forms, the crop's pixel and feature index
  for a row put on by index, the evaluation state, a point with no patch
  placement (§ "A point with no patch placement"), a row click, which selects
  and reveals with no row selection, and a double-click, which asks for
  camera view.
- **Edited mode and the table**:
  [body/tests.rs](../../crates/sfm-explorer/src/track_view/body/tests.rs).
  These tests establish the rest of § "One body in two modes", section by
  section:
  - § "The toolbar": no *Evaluate* button, and the evaluation state in the
    toolbar and the Status cells as the evaluation runs, lands and runs again,
    with core's refusal sentence for a bearing that has no placement; no item
    tabs, other items on the bench appearing nowhere in what the panel drew;
    *Lock* starting ticked and toggling with no version and no gesture
    reported, greyed at the cluster stage, where a click leaves it as it was.
    With no item focused, Edited mode draws nothing.
  - § "The header": the counts of kept, `out` and pinned rows, the Point ID
    printed once beside a renamed label, the index of a point that is gone,
    the ID the copy button copies, the `Bearing (...)` and `Position (` lines
    with the infinity mark on a bearing alone, and the track's own patch slot:
    empty before a fit, filled after it with the toolbar to its right,
    following the template at the cluster stage, opaque for a bitmap of one,
    three or four channels, and not drawn for an all-zero bitmap; a bitmap
    kept only for judging drawn dimmed, its hover saying it is not the track's
    patch and a commit does not write it, and the hover of a bitmap the next
    render replaces saying so.
  - § "The thresholds": each box under the heading of what it judges and the
    geometry search box above the table; the boxes showing the track's bars
    outside a drag and following a step, an undo, a redo and another item
    focused; a release pushing exactly one version and one row, with the bar
    where it was let go, which an undo takes back; no *Apply thresholds*
    button; each judged line coloured by its own bar, so a whole ZNCC can pass
    while its middle fails and the reverse; the readings no bar judges drawn
    plain (the degrees of *Proj. err*, the self-similarity middle, the status,
    the middle ZNCC with its bar off, a missing reading, an unmeasured row and
    a refused evaluation); the *Keep* tint matching what applying the bars
    gives an unpinned row, a drag leaving a pinned verdict as it was, a pinned
    `in` row against the bars tinted `out` with the failing bar in its hover
    text and turning `out` when unpinned, and of two sightings of one image that clear every bar, the one that
    loses the image tinted `out` with the image named; and the row that holds
    the reference, where unpinning it would move the bitmap, given no proposal,
    its hover saying the bars cannot yet say what they would propose, rather
    than that nothing has measured it.
  - The *Keep* column and § "Row gestures": the switch taking a click the row
    behind it does not, turning a kept row `out` and pinning the verdict it
    sets; the pin pinning and unpinning; *Unpin* in the switch's menu and the
    row's, where unpinning a row that is no longer pinned has no effect; the row menu unpinning every
    pinned row of a selection with the count in its label, while the switch's
    menu acts on its own row; the heading pin unpinning all, or pinning every
    row at its verdict as one version and one row that one undo takes back,
    with its hover text and accessible name in both states; the searches and
    their remedies in the row menu; *Accept walk* offered on exactly the rows a
    fit kept at their seeds and moving that keypoint to the walked pixel,
    pinned, in one version; *Set as reference* in the row menu on an `in` row
    with a keypoint, greyed on the held reference and absent on an `out` row,
    making the row the reference, from which the live evaluation after it
    renders the bitmap; unpinning the row that holds the reference, where the
    rule picks another row, logging that the verdicts wait for the new bitmap,
    after which the live evaluation renders from the pick and the pick is the
    reference; and a row click and double-click reporting the image and the
    pixel.
  - § "The observation table": the order of the columns and the heading size;
    one row per observation, in index order, a verdict set on a row staying
    under that observation's index, pinned; the cells following the stage; the
    ZNCC cell's `whole` and `mid` percent form and `-` for a missing middle; the
    Status cell's sentences, including the walk with and without its ZNCC; the
    *Reference* cell's pick, its word for each test, the tests dropped, the
    angle and pair ZNCC on every row, its hover readings, and `-` at the
    cluster stage and for a refused evaluation; the column's two marks,
    `reference` on the reference and `pick` on a pick that is not the
    reference, green for a reference the rule picks, red for one it does not
    with the pick grey, the pick alone for a track whose bitmap is a fused mean
    or that has none, and the hover naming the other row and how to accept the
    pick; the track-stage *ZNCC* cell's plain score against the stored bitmap,
    with the blur-matched score after an arrow where the two print differently
    and one number otherwise, `100%` on the reference, `-` with the reason in
    the Status cell and the hover for a row with no score; the *ZNCC* hover
    drawing three tiles for a blurred row at the tile hover's third, the note
    in place of the blurred bitmap for an unblurred row, the bitmap alone on
    the reference, no picture on a row with no score, both scores in its
    table, and the blurred bitmap equal to the bitmap through core's kernel
    ([`tests/zncc_hover.rs`](../../crates/sfm-explorer/src/track_view/body/tests/zncc_hover.rs));
    after a fit, no
    *Bitmap* heading, exactly one row marked as the reference, the one the
    stored bitmap names, reading `100%`, and every other row scored; `out (2)`
    and `out` in the *Verdict* text; the *Zoom* cell's
    two significant digits, chosen after rounding, with both numbers printed
    where they agree; the self-similarity hover's table in its three units,
    each ellipse as major × minor axis and angle with no angle on a circle,
    the unit chosen, bare numbers under *scene units*, the `+` kept, scientific form under 0.001, degrees at infinity and
    `-` in the world row at the cluster stage, and the hover carrying exactly
    the ellipses of the row's measurement; and a long name cut in its middle,
    shown whole on hover while the row stays hovered.
  - The tiles and the crops: a tile on every row at both stages, a fresh row's
    cut around its seed; the track-stage tile re-anchored on each
    observation's keypoint, and kept as stored with no keypoint or where the
    keypoint's ray cannot meet the patch; a shrinking tile read from its mip
    level throughout and one that shrinks nothing identical to plain bilinear;
    the hover view holding the tile, or the crop, texel for texel in its middle
    third, with the dot on the observation's pixel and the projection mark at
    the row's reprojection error, which the crop's caption states, no mark at
    the cluster stage, and resting on
    one row showing that row's view alone while the row keeps its hover and its
    click; the crop square, every sample of the outline at least one pixel
    inside it, centred to within the rounding so the outline is less than two
    pixels from both edges along the longer side, its texels the photograph's
    own pixels, and its caption's two axes matching the projected edge
    midpoints at the track stage and the shape's columns at the cluster stage.
  - The zoom (§ "The *Zoom* column says how much the patch magnifies the
    photograph"): the centre Jacobian of a fronto-parallel patch is diagonal at
    its width over `R`, the fallback 24 and not the tile's 64 texels; a patch
    turned in its plane carries the turn in the Jacobian's off-diagonal
    entries, signs included, and changes no zoom; a slanted patch gives a range
    of zooms; a tile whose middle is off the photograph has the zoom it has on
    it; a patch behind the camera or seen edge on has none; a keypoint moved
    off the projection gives exactly the re-anchored placement's Jacobian; 48
    px patch bitmaps give a Jacobian half the size and zooms twice as large as
    no bitmaps; and the cell prints the zoom with no photograph decoded.
  - The order of the rows (§ "The rows are ordered by a column"): increasing
    by image at the start; each heading's first click worst first or
    increasing as the table there says and a second click reversing it;
    *Keep* by the number of bars failed, *Zoom* by the mean zoom and
    *Reference* with the reference first and the pick next; a row with
    no key last both ways and ties in increasing order of image; and *Crop*,
    *Patch* and *From* ordering nothing.
  - The scrolling (§ "The table scrolls both ways"): a sideways wheel, in both
    the point and the line units egui reports, moving the rows, the headings
    and the threshold row together; a middle and a left drag over the rows and
    over the headings; and a drag begun on a switch moving nothing.
- **The Scene tree**:
  [scene_graph/tests.rs](../../crates/sfm-explorer/src/scene_graph/tests.rs).
  These tests establish § "The Scene tree's Bench rows": a single click selects
  the node with no version; a double-click from either group focuses the item,
  selects the node and raises the panel with no version and one `Editing ...`
  row, and on the focused item writes no row; and the raise survives a
  double-click made while the dock is swapped out.
- **Layout and wire**: [`panel-layout.md`](panel-layout.md) § "Testing" (the
  stock grid, the retired names refused, the startup load of an old default
  file) and [`mcp-server.md`](mcp-server.md) § "Testing" (the panel name,
  `unfocus_bench_item`, the null `focused_item`). That the stock grid holds the
  tab exactly once is tested in
  [layout/tests.rs](../../crates/sfm-explorer/src/layout/tests.rs).
- **The windowed suite**,
  [tests/ui_basic.rs](../../crates/sfm-explorer/tests/ui_basic.rs): a real
  viewer refuses a `screenshot` of `camera_intrinsics`, which the stock grid
  keeps behind Track View, with a message naming "Track View". It does not test
  the box, since what the box decides is covered headlessly.

---

## Non-goals

- **Showing a committed track and its bench copy side by side.** The panel shows
  one or the other in one layout; clearing and ticking Edit flips between them,
  and no flip is a version. A second Track View is not possible, since a panel is a
  singleton.
- **Listing the bench in the panel.** The Scene tree's two Bench groups are the
  list, per node, with the stage each item is at; the recent items strip holds
  only the last few items focused.
- **Acting on the track from a number beyond its verdicts.** The bars set the
  verdict of every unpinned row (§ "The Keep switch is the verdict"), and a
  pinned verdict stands against them; every other step on the track, a fit, a
  normal, a stage change, a split, a search or a commit, is made only when the
  person asks for it.
- **A second tile beside the first**, the cluster template and each member warped
  onto it side by side, and **the remaining searches**, both proposed in
  [`../drafts/sfm-explorer-track-editing.md`](../drafts/sfm-explorer-track-editing.md).
- **Keeping the SIFT index in step with the workspace.** *Build* is asked for;
  the viewer does not watch the `.sift` files.
- **Editing the patch by hand in this panel.** The hand edits that move it are
  the two bench layers' handles and the wire's patch tools
  ([`bench.md`](bench.md) § "The wire").
- **Per-observation readings for a point with no patch placement** (§ "A point
  with no patch placement"); a projected outline of the viewed track in Image Detail or
  a figure of it in the 3D viewer, which would need its own way of reading as
  not editable; the stored bitmap's alpha channel as a tile; a per-row highlight of
  the track ray in the 3D viewer; and a positional uncertainty display.
