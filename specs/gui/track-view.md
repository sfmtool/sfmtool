# Track View

A 3D point in a reconstruction is only as good as the photographs that saw it.
Each point carries a **track**, the list of feature observations, one per image,
that were triangulated into it, and judging a suspect point means reading that
list sighting by sighting: where each one sits, how far it lies from where the
point projects, and whether the patch of surface it shows looks like the
others. Sometimes the answer is that the track is wrong, and then it has to be
worked on: sightings tried, measured, turned out, and the result written back.
Track View is the one panel of the SfM Explorer where both happen, in one body
drawn in two modes. With its **Edit** box clear it shows the selected point read
as a track, measured the way the bench measures a track, and changes nothing.
With the box ticked it shows the one track the viewer is editing, which is held
beside the reconstruction on the **bench** until it is committed, together with
the controls that fit it and commit it, and the measurements that are kept
current as it changes.

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
items strip, and
[body/](../../crates/sfm-explorer/src/track_view/body/) is the one body that
draws both modes: `mod.rs` the header, the toolbar and the boxes, `table.rs` the
observation table, `tile.rs` the tile each row draws and its hover view,
`crop.rs` the crop of the photograph beside the tile and its hover view,
`surface_plot.rs` the self-similarity surface plot, and `patch.rs` the warp a
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
because a track staged into a panel nobody can see is a gesture with no answer.
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
the header's patch slot draws for that item (the consensus bitmap at the track
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
where a person with an ID in hand and no idea how to feed it in looks
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
pinned and the leave-one-out ZNCC is read back from the point's stored
confidence column, and the live evaluation keeps it evaluated as it keeps a
bench track, so every number in the table means the same thing in both modes.
It is never written anywhere: it is in no version, the Scene tree does not list
it, the bench layers do not draw it, and no step accepts it. Ticking *Edit* over
it puts the same point on the bench, and the bench track starts from what the
panel was showing. The panel is handed `&AppState` and cannot build it, so the
dock asks for it before drawing the panel (`AppState::refresh_viewed_track`).

**What Viewed mode reads.** Everything about the point, including its position,
colour and error, its whole track, each observation's keypoint, its patch frame
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

The metrics are [metrics/](../../crates/sfm-explorer/src/metrics), at the crate
root, because the Image Detail overlay and `get_point` read the same numbers.
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
reconstruction with no patch frames reads as the bearing it is. A fit that
crosses the boundary changes the headline's first word, which is how a person
sees that it crossed. The same flag puts **an infinity mark** (∞, U+221E, which
egui's bundled fonts draw) left of the label, as the Viewed header puts one left
of its Point ID (`track_view::infinity_mark`), so the header's first glyph says
a direction before any number is read.

**The track's own patch stands at the left, under the header**, 64 points
square (`STORED_PATCH_SIZE`) with nearest filtering and no label, since the
picture says what it is. The line under the header, the toolbar and the
geometry search box stand to its right, and the table's separator runs directly
under it. At the track stage it is the consensus bitmap
(`TrackPayload::bitmap`): for the viewed track that is the point's stored patch,
and for a bench track the bitmap a commit writes as the point's stored patch,
so a *Fit* that re-fuses the track changes it and a person sees what the commit
would store before committing. At the cluster stage it is the template every
member registers onto, once an evaluation has cut one. With neither -- a point
that stores no patch, a track not yet fused, or a cluster with no template --
the slot is an empty frame of the same size, so the controls beside it do not
move when a step fills it. Its hover text says which it is, or what would fill
it. The bitmap is converted by `patch::stored_patch_image`: one channel repeated
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
no patch frame to read against, as a point put on the bench from a sift_files
reconstruction has none; convert the reconstruction to embedded patches and put
the point on the bench again"* -- in place of any state.

**Commit leaves the point it wrote selected.** The write is
[`edits/commit-track.md`](edits/commit-track.md)'s; the panel's part is that the
point the commit produced becomes the selection, whether it replaced a point or
created one, so the viewport's track rays and the observing frustums look at what
was just written. The dock drops what the panels cached about
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

**The three normal entries turn the patch and leave its centre.** *Fit* moves
the patch along the sightings' rays and keeps the way it faces; *Fit Normal*,
*Finite Diff Normal* and *Grid Plane Normal* do the opposite, each estimating a
normal and turning
the patch to it by the same least rotation the 3D viewer's arrowhead drag makes,
then reading the track back and fusing its bitmap as a fit does
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) §
"Estimating the normal"). *Fit Normal* takes the normal at which the `in`
sightings' tiles agree best. *Finite Diff Normal* fits a row of smaller square
pieces through the centre along each of the patch's two axes and takes the
plane through the two lines their centres land on. *Grid Plane Normal* fits a
grid of pieces tiling the whole patch and takes the plane through all their
centres. The two boxes after them say how both cut: *per axis*, from 2 to 8
pieces along each axis (2 by default), which is a row of that many on each axis
for *Finite Diff Normal* and that many by that many for *Grid Plane Normal*; and
*overlap*, from 0% to 90% of a piece's side (0% by default). The labels say
which: "3 pieces along each axis" for the rows, "3x3 pieces" for the grid. All
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

Six boxes, one per bar of `Thresholds`, so no bar is one only the wire can
move. Five of them judge the table's readings, and they stand in **a threshold
row directly under the column headings**, each in the column of the readings it
judges and followed by the unit and name those readings print with:

```
Img  Crop  Patch  Keep  ZNCC          Self-similarity           Proj. err  Shift     Zoom  Status  ...  Name
Thresholds              [70]% whole         [2.5] px whole      [3.0] px   [6.0] px
                        [70]% mid
```

The row is drawn above the scroll area with the headings, so a bar stays beside
the heading of what it judges however far the table is scrolled, and the
columns no bar judges leave room for the word *Thresholds* at its left. The
minimum ZNCC and minimum middle ZNCC boxes stack in the *ZNCC* column, whole
over mid, as its cell stacks the two readings. They read and take percent in
whole steps, as that column reads, so both show 70 on a new track while the
track and the wire hold 0.7; a typed value may end in `%`. A middle bar of 0
turns it off, and a row with no middle reading clears it.
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
per frame, because a copy of a track carries its consensus bitmap. An
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
and increasing for *ZNCC*; *Img*, *Zoom*, *Status* and *Name* start increasing.
*Zoom* is among them because no bar says one zoom is worse than another. The
heading the rows are ordered by carries a small triangle after its word,
pointing up for increasing and down for decreasing, and a heading that orders
the rows is drawn in the plain text colour under the pointer. Its hover text
ends by saying what a click does with the order as it stands, and its accessible
name is *Sort by …*. The keys are what the rows print:

| Heading | Ordered by |
|---|---|
| Img | the image's index |
| Keep, Verdict | how many bars the row fails; among rows that fail the same number, the ones the bars propose `in` come first, so a row that fails none and is `out` because another sighting holds its image comes after the `in` rows |
| ZNCC | the whole patch's ZNCC |
| Self-similarity | the whole tile's radius |
| Proj. err | the reprojection error in px |
| Shift | the shift in px |
| Zoom | the geometric mean of the two zooms, `1 / sqrt(abs(det J))` |
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
so a drag of either is added to its offset by the panel. The tests cover a
sideways wheel in both units, a middle and a left drag over the rows and the
headings, and a drag begun on a switch.

The image's index is the first column, at the table's left edge, under
*Img*; the crop of the photograph around the patch's outline is the second,
under *Crop*; the rendered tile is the third, under *Patch*; and the verdict
column follows them, *Keep* in Edited mode and *Verdict* in Viewed mode. The
crop comes before the tile because it is the photograph as it is, and the tile
beside it is that patch warped square, so the eye reads from the raw pixels to
the picture the numbers are read from. The two photometric columns come
straight after the verdict, *ZNCC* and then *Self-similarity*, since they are
the readings the verdict is most often decided by; the reprojection error, the
shift, the patch's zoom, the status and, in Edited mode, the provenance follow
them. The image's name is the last column, 220 points wide: hovering it or the
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
way to work on it.

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
| Verdict (Viewed) | absent: a cluster has no Viewed mode | `in` or `out`, the verdict the read-only bars give the row, with the number of bars an `out` row fails in brackets (`out (2)`), in a cell tinted green or red by it; `-` untinted where nothing has measured the row |
| Img | the image's index; hovering it shows the file name whole, as hovering *Name* does | the same |
| ZNCC | against the reference template, over the middle ZNCC: `92% whole` over `61% mid`, then the ZNCC grid | leave-one-out against the consensus, at the correlation peak within the shift bar of the observation, over the middle ZNCC, then the ZNCC grid |
| Self-similarity | the surface plot, then the tile's ZNCC self-similarity radius over its middle square's: `0.4 px whole` over `3+ px mid`, `3+` for the largest, then the self-similarity grid | the same |
| Proj. err | absent | the reprojection error: how far the keypoint sits from the point's projection, or, before the track is triangulated, from its patch's centre's, over the same residual as the ray angle, comparable across lenses and depths: `0.65 px` over `0.08°` |
| Shift | how far the refinement moved the member off its seed, in patch-grid px: `1.20 px` | how far the correlation peak, looked for within the shift bar, sits from the observation's own keypoint, in patch-grid px on the patch's plane |
| Zoom | `-` | patch-grid px, at the reconstruction's patch resolution `R`, per photograph pixel at the patch's centre, the reciprocals of the two singular values of the Jacobian there of the warp from the patch grid to the photograph, least over most, each to two significant digits: `0.71/1.3×`, and both numbers even where the two print the same, `0.19/0.19×`; `-` for a track with no patch yet, an observation with nothing saying where it sits, a patch whose centre is behind the camera or outside the camera model's domain, and a patch seen edge on |
| Status | the kernel's `member_status` | `walked 19 grid px (ZNCC 87% / 41% there), kept at seed` where the last fit refused to move it, the ZNCC being the one the fit scored at the walked peak (left out where it scored none), `localized` where the evaluation scored it, the reason's own sentence where it could not, `not evaluated` where nothing has been read |
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

**The zoom also decides which sampler draws the tile.** The sampler rule
([image-warping.md](../core/camera/image-warping.md) § "Choosing the sampler
per view") reads the same Jacobian at `R` and renders the row's tile and its
hover view with the anisotropic sampler where one mip level for both axes would
read the less compressed axis at least 1.5 times too coarsely
(`PatchJacobian::sampler`, `tile::tile_sampler`), and with `bilinear_mip`
otherwise, so a view is drawn with the sampler the bench's evaluation and the
fuse render it with. Hovering a *Zoom* cell names the sampler and that loss
(`table::zoom_sampler_text`). `get_bench_track` and `get_point`'s evaluation
block report each row's `sampler` (`anisotropic` or `bilinear_mip`) and
`sampler_minor_axis_loss`, both null where `patch_jacobian` is.

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

**The track-stage frame is re-anchored on the observation's keypoint**
(`OrientedPatch::anchored_at_keypoint`, [patch-cloud.md](../core/patch/patch-cloud.md)):
the patch centre slides within its own plane, on its tangent sphere for a point
at infinity, until it projects onto the keypoint this observation carries. The
tile is a photometric comparison, so it is anchored where the keypoint localizer
aligned the pixels rather than at the point's geometric projection. The tiles of
a well-localized track then match each other whatever discrepancy the geometry
carries, and that discrepancy is what the *Proj. err* column reports: a column
of matching tiles beside a large error is a well-aligned patch whose position or
pose is off, not a bad match. The stored frame is used as it is when the
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
reconstruction carries patch frames, with the SIFT cache filled for them on a
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
has measured the row yet, or where the viewed track's evaluation was refused or
failed. It is the proposal the *Keep* cell is tinted by, in the same green and
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

#### A point with no patch frame

A reconstruction whose points carry no patch frame, which is the usual case for
a `sift_files` reconstruction imported from COLMAP, is drawn as any other, and
what that shows is less than the rest of this section describes. `create_track`
builds a track-stage track with no frame, and core's `evaluate_preconditions`
refuses it, so the toolbar shows the refusal sentence (*"Cannot evaluate
pt3d_…: the track carries no patch frame to read against; …"*) and every
number cell reads `-`, the reprojection error and ray angle included, and every
*Verdict* cell `-`. The *Crop* and *Patch* cells are empty frames, and there is
no crop hover view to carry the pixel and the feature index. The header still
carries the point's summary (its colour, `xyzw`, error, track length, max pair
angle, depth z and cond), which reads the point and needs no frame. So for such
a point the panel shows its header and no per-observation readings. Which
readings a frameless evaluation should produce and what the *Crop* cell would
show without a frame are not decided here (§ "Non-goals").

**Such an item is not edited here.** A `sift_files` reconstruction's bench is
view-only ([`bench.md`](bench.md) § "A view-only bench"): every toolbar button
but *Discard* and *Rename*, the threshold boxes, the *Keep* switches and pins
and the row menus' editing entries are greyed and take no click, with the
sentence that names Convert to Embedded Patches as their hover. After the
conversion, putting the point on the bench again (*Edit*, *Edit on Bench* or a
double-click) rebuilds the item with the point's new patch frame, and from
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

- **The panel**,
  [track_view/tests.rs](../../crates/sfm-explorer/src/track_view/tests.rs),
  headless through `Context::run_ui`, each frame asking for the viewed track
  first as the dock does: the box reads the focused item, Edited mode
  drawn with a focused item and Viewed mode on the origin without one, and following an
  unfocus and a focus made outside the panel with no panel call in between;
  ticking it over a selected point reporting `set_edit`, and applied, one
  version putting the point on the bench, with a second tick after a clear
  focusing the existing item and pushing no version; clearing it reporting
  `set_edit`, no version and one `Selection` row `Stopped editing ...`, and the
  next frame drawing the selected point's header; the box greyed with no point
  selected and no item focused this session, a click on it reporting nothing,
  the refusal naming the three ways in and the empty state carrying the bench
  line; with no point selected and an earlier item, the box live, its hover
  text `Tick to edit {label} again`, the empty state naming the item, and the
  tick focusing it and selecting its origin with no version; the tick going
  back to an item on another node, selecting that node, and passing over a
  discarded item; a busy node greying the box only over a point not on the
  bench; the empty state counting the items on a bench with none focused and
  none recent; the empty state's *Go to Point...* asking for the dialog and an
  idle frame not asking; a deleted and an out-of-range point taking the empty
  state; a discard of the focused item leaving Edited mode; Edited mode
  on a cluster drawing the cluster headline; a point selected while editing
  drawing that point in Viewed mode with no version; and no
  reconstruction drawing no box. The recent items strip, in the same module:
  the chips most recent first with the focused item left out, the label cut in
  its middle, and the item just left first after a clear; at most eight on a
  wide panel, the most recent few on a narrow one, and none when the row has no
  room beside the box; a chip click reporting `focus_item`, and applied,
  focusing the item and selecting its origin with no version while a
  background task holds the node; a rename keeping the chip under the new
  label; a discard hiding the chip and its undo bringing it back; a chip for an
  item on another node naming that node in its hover text and, clicked,
  selecting that node, the item and its origin; the hover text's label, stage,
  counts, origin, position, evaluation state and *Click to edit*, and no node
  name with one reconstruction loaded; and the track's patch drawn from the
  same texture on a second frame, a cluster with no template drawing an empty
  frame, and its hover text saying it is new.
- **The selection rules**,
  [bench/tests.rs](../../crates/sfm-explorer/src/bench/tests.rs) § "The
  selection rules": another point selected unfocusing with one `Stopped
  editing` row and no version; the origin, a cleared point, an image or a
  camera of the node, the node row and a cleared selection keeping the item
  focused; an image, a camera, a point or the node row of another node
  unfocusing; focusing selecting the node and the origin, or clearing the point
  and keeping an image of that node for an item with no point; every put
  (from a point, a cluster, a duplicate, a split) selecting what it focused;
  clearing Edit and a discard selecting the origin or clearing the point, the
  node staying selected; an item whose origin was deleted staying focused
  through the deletion and focused with no point; a commit keeping the item
  focused, a duplicate gaining an origin by its commit, and the undo of that
  commit leaving it focused with no point selected; an undo past the put
  unfocusing with the selection unchanged; and the recent list most recent
  first, once each, skipping an item not on the bench and bringing it back on
  the undo, pruned when its node closes and emptied by *Close All*. The wire's
  half is in [mcp/tests/bench.rs](../../crates/sfm-explorer/src/mcp/tests/bench.rs)
  (`select_point` unfocusing with no version, `focus_bench_item` and
  `unfocus_bench_item` selecting the origin). The handle's claim on a click is
  in [image_detail/tests.rs](../../crates/sfm-explorer/src/image_detail/tests.rs)
  (a click on the ghost's centre over a feature of another point selecting no
  point) and [viewer_3d/tests.rs](../../crates/sfm-explorer/src/viewer_3d/tests.rs)
  (a click on the dot, a corner or an edge requesting no point pick).
- **Viewed mode**,
  [body/tests/viewed.rs](../../crates/sfm-explorer/src/track_view/body/tests/viewed.rs):
  the body drawing the point's ID, the line saying how to change the track, no
  toolbar button and no *Lock*, and a click on the verdict cell being the row's
  with no verdict step; the headings matching Edited mode's with *Verdict* in
  place of *Keep* and no *From*; each *Verdict* cell `in` or `out` as
  `verdicts_if_unpinned` gives it, tinted to match, with the failing bar in its
  hover text under strict bars, and `-` untinted on an unmeasured row; a drag
  of each of the five boxes reporting the bars, recolouring the ZNCC readings
  and the *Verdict* cells, and leaving every verdict `in`, the viewed track
  unchanged, no version and no row; the bars surviving a selection change; the
  "On the bench as …" line when an item from the point exists; the header's
  summary for the viewed track and for a bench item at the track stage, and
  none for a cluster; the crop caption's pixel and feature index, and the
  `.sift` feature a row put on by index names; a frameless point drawn with its
  refusal sentence and its header, empty *Crop* and *Patch* frames and `-`
  cells; *Evaluating…* until the viewed evaluation lands and *Evaluated* after;
  a row click selecting and revealing with no row pick, a double-click asking
  for camera view; the header's go-to button; and the infinity mark left of a
  point at infinity's ID and absent for a finite point.
- **Edited mode**,
  [body/tests.rs](../../crates/sfm-explorer/src/track_view/body/tests.rs), with
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
  image named in its hover text; the *Img* column at the table's left edge,
  the crop after it under *Crop*, the tile after that under *Patch* and *Keep*
  after them; the
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
  and re-seating on another item; each box of the threshold row standing
  under the heading of what it judges, whole over mid in the ZNCC column, and the
  geometry search box above the table; a drag of the shift box pushing exactly one
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
  leaves it as it was; the tile's frame re-anchored on each observation's own keypoint
  and kept as stored with no keypoint and where the keypoint's ray cannot meet
  the patch; a shrinking tile read from mip level 2 throughout and one that
  shrinks nothing plain bilinear to the bit; the centre Jacobian of a
  fronto-parallel patch diagonal at its width over the fallback `R` of 24, and
  not over the tile's 64 texels, of a patch turned 30
  degrees in its plane carrying the turn with its signs, and of a slanted patch
  giving a range of zooms; a zoom for a tile whose middle is off the photograph,
  the same as on it, none for a patch behind the camera, and none for a patch
  seen edge on; a zoom read from the patch re-anchored on a keypoint moved a few
  px off the projection, exactly that placement's Jacobian; a reconstruction
  with no patch bitmaps giving every row the Jacobian at 24, and the same rows
  with 48 px patch bitmaps a Jacobian half the size and zooms twice as large;
  each row's *Zoom*
  cell printing the zoom of its tile's Jacobian, and the same with no photograph
  decoded; a click on *Zoom* ordering the rows by mean zoom both ways, a row
  with no zoom last either way; the *Zoom* cell's two significant digits chosen
  after rounding, with both numbers printed where they agree; the
  *Self-similarity* hover's table of ellipses in grid px, image px and world
  space, each as major × minor axis and angle, with `+` on a lower bound and no
  angle on a circle, every world length scaled to the one
  unit its largest picks (metres to mm and µm, cm kept, feet to inches) with
  the `+` kept, bare numbers under *scene units* with no unit on the file, in
  scientific form under 0.001 and plain from there up, the whole table for a
  patch at infinity in degrees, and for the cluster stage with `-` in the world
  row, and an evaluated row carrying exactly the ellipses its measurement carries
  at both stages; and a long image name cut in its middle, keeping the start of
  the path and the end of the file name.
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
  one or the other in one layout; clearing and ticking Edit flips between them,
  and no flip is a version. A second Track View is not possible, since a panel is a
  singleton.
- **Listing the bench in the panel.** The Scene tree's two Bench groups are the
  list, per node, with the stage each item is at; the recent items strip holds
  only the last few items focused.
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
- **Per-observation readings for a point with no patch frame** (§ "A point with
  no patch frame"); a projected outline of the viewed track in Image Detail or
  a figure of it in the 3D viewer, which would need its own way of reading as
  not editable; the stored bitmap's alpha channel as a tile; a per-row highlight of
  the track ray in the 3D viewer; and a positional uncertainty display.
