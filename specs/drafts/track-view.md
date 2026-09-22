# Track View

**Status:** Draft. Proposes one panel, **Track View**, in place of the two the
viewer has now: the Point Track panel
([`../gui/point-track-detail.md`](../gui/point-track-detail.md)) and the Track
Edit panel ([`../gui/track-edit.md`](../gui/track-edit.md)). Every design
question is settled: the merge itself, the **Edit** checkbox at the top of the
panel as the switch between looking at a committed track and editing a bench
item, the Scene tree's double-click on a Bench row as the way into editing that
item, the panel no longer listing the bench, and each point of detail below.
What remains is the implementation and the filing of `gui/track-view.md`.

## Purpose

A 3D point in a reconstruction is only as good as the photographs that saw it.
Each point carries a **track**, the list of feature observations, one per image,
that were triangulated into it, and judging a suspect point means reading that
list sighting by sighting: where each one sits, how far it lies from where the
point projects, and whether the patch of surface it shows looks like the
others. Sometimes the answer is that the track is wrong, and then it has to be
worked on: sightings tried, measured, turned out, and the result written back.
Track View is the one panel where both happen. With its **Edit** box clear it
shows the selected point's committed track and changes nothing. With the box
ticked it shows the one track the viewer is editing, which is held off to the
side of the reconstruction on the **bench** until it is committed, together with
the controls that measure it, fit it and commit it.

The viewer has had these as two panels, one showing the selected point and one
showing the bench. Working with them apart has two costs. The person has to
keep two tabs in step with each other by hand, raising one after a gesture made
in the other; and the editing panel spends its top rows on a list of every item
on the bench, which the Scene tree already lists per node, in two groups that
say which stage each item is at. One panel with an explicit mode says which of
the two things is on screen, and leaves listing the bench to the tree.

---

## The panel

A dock tab, `Tab::TrackView`, titled **Track View**, saved in the layout file
and named on the wire as `track_view`. Its home is the top-right node, where
Point Track was, as the first and active tab of that leaf with Camera
Intrinsics behind it, so the stock grid is

```
+--------+-------------------------+----------------------------------+
| Scene  |[3D Viewer][Image Detail]| [Track View][Camera Intrinsics]  |
+--------+-------------------------+----------------------------------+
|Backgr. |[Image Browser][Action Log][Edit History]                   |
+--------+------------------------------------------------------------+
```

and the panel count drops from ten to nine. `Tab::PointTrackDetail` and
`Tab::TrackEdit` are removed, and with them their titles, their Panels menu
entries and their wire names. The group the layout's placement rules call
home becomes `[TrackView, IntrinsicsDetail]`
([`../gui/panel-layout.md`](../gui/panel-layout.md) § "Home positions").

**A saved layout that names either retired panel is refused whole**, which is
what [`../gui/panel-layout.md`](../gui/panel-layout.md) § "Validation" already
says of a renamed panel. `point_track` and `track_edit` are unknown panel names
like any other, and the refusal is the ordinary one: prefixed with the path to
the leaf, as every validation message is, it lists the nine names that exist,
`unknown panel "point_track"; the panels are scene, background_task, viewer_3d,
image_browser, image_detail, track_view, camera_intrinsics, action_log,
edit_history`. The reader carries no aliases and
no rule for a file that names both; the at-most-once rule applies to
`track_view` as to every other name. Reset Layout is the way back, and saving
the layout afterwards writes a file the viewer reads.

The default-layout file read at startup is the case that meets this most often,
since every such file saved before the merge names both retired panels. It takes
the path every refused startup load already takes (§ "The default layout file"
in the panel-layout spec): nothing of the file is applied, the viewer comes up
on the stock grid, which now holds Track View, and the failed `Layout` entry
`Load layout from C:/Users/mark/.sfm-explorer-default-layout.json: <reason>`
goes on the viewport status line, so the person sees why their layout did not
come back. No code is added for it; the implementation adds a test that drives
`load_layout_file` over a file naming `point_track` on a freshly built state and
asserts the stock grid, the failed entry and its status-line message.

`LAYOUT_VERSION` stays at `2`. The panel-layout spec's rule is that the version
tags the document's *shape*, and a rename changes no key and no node kind, only
the set of values a leaf's `tabs` may hold; its `LAYOUT_VERSION` row already
names renaming as the case the version does not cover, answered by the whole-file
refusal. So the rename leaves the version alone and relies on that refusal, as
the row says.

---

## The Edit checkbox

The first row of the panel is a checkbox, **Edit**, and nothing else is drawn
above it in either mode.

**The box shows the state of the bench, not an independent setting.** It is
checked if and only if a track or cluster is active on the bench. Nothing about
it is stored in the panel. So every way the activation can change moves the box
on the next frame without any code keeping the two in step: a Scene tree
double-click, a step that puts an item on the bench, a wire call, and an Undo
or Redo that walks back over an activation.

The bench holds **one active item at a time**, and a point track and a cluster
are not two kinds for that purpose. Both are editable tracks
(`ItemKind::Track`) at different stages, which is why the Scene tree draws them
in two groups and still marks one active row across both
([`../gui/bench.md`](../gui/bench.md) § "The Bench groups in the Scene tree").
So a ticked box always names exactly one item, whether it sits under *Bench
Points* or *Bench Clusters*, and activating a cluster while a track is being
edited leaves the track on the bench, no longer active.

### Transitions

| From | Gesture | What happens | Then shown |
|---|---|---|---|
| Viewing a selected point | Tick Edit | `put_point_on_bench(selected point)`: a new item, active, in one version; or, when an item on the bench already came from that point, that item is activated instead | Edit mode, on that item |
| Viewing, no point selected, nothing active | Tick Edit | Nothing: the box is greyed, its hover text naming the ways in | Unchanged |
| Editing an item | Clear Edit | `deactivate_bench_item`: the item stays on the bench and nothing is active, in one version | View mode, on the selection |
| Either | Double-click a Bench row in the Scene tree | That item becomes active and the panel is raised | Edit mode, on that item |
| Either | *Edit on Bench* in the 3D viewport or Image Detail, or a double-click on a point or a feature | As the ticked box from that point, then the panel is raised | Edit mode |
| Either | *Start cluster on the bench here* in Image Detail | A cluster is put on the bench, active; the panel is raised | Edit mode, on the cluster |
| Editing | *Duplicate*, *Split off N rows* | The new item is put on the bench and becomes active, as today | Edit mode, on the new item |
| Editing | *Discard* | The item leaves the bench and nothing is active | View mode |
| Editing | *Commit* | The point is written and selected; the item stays on the bench and stays active | Edit mode, on the same item |
| Either | Undo or Redo over an activation, a deactivation, a put or a discard | The bench at the cursor says what is active | Whatever that bench says |

The box is greyed while a background task holds the node, with the node's own
busy sentence, because every transition above is a bench step and a bench step
is refused then.

The ticked box's own refusal, with no point selected and nothing active, is one
sentence naming the three ways in: select a point, double-click an item in the
Scene tree's Bench groups, or right-click a pixel in Image Detail and choose
*Start cluster on the bench here* (the last quoted from that entry's constant,
as the Track Edit empty state quotes it now).

### Why each transition is the one it is

**Clearing Edit only deactivates.** The box stands for "active for editing",
not for being on the bench, and a discard already has its own toolbar button and
Scene tree menu entry. An Edit-clear that discarded would make looking at the
committed point cost the person their verdicts and their fit.

**A deactivation is a version**, one version and one `Bench` row, as an
activation is. The activation is part of the bench value, an Undo that brings
back what was being edited is useful, and keeping it there leaves the core
bench's equality and every "undo of a discard restores the activation" test as
they are. The price is that flipping Edit to compare a track with its committed
point pushes two versions per round trip.

**Ticking Edit on a point already on the bench reuses its item**: the item whose
origin, followed to the cursor, is that point is activated rather than a second
one put on, which is what `put_point_on_bench` does for every way in. Two items
for one point are two answers to one question, and with the item tabs gone a
second one would be easy not to notice.

**Discarding the active item leaves nothing active**, so Edit clears and the
panel returns to view mode. Handing the activation to a neighbour, as
`Bench::discard` does before this change, would switch the panel to an item the
person did not ask for, and with the list no longer in the panel nothing on
screen would say which one arrived. This changes core's `discard`, and with it
its Python binding and tests.

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

## What the panel shows

### No reconstruction selected

`No reconstruction loaded`, centred, and no checkbox. Both panels already
follow the selected node and each says this in its own words; the merged panel
says it once.

### View mode

View mode is the Point Track panel as it stands, below the checkbox, and it
reads the reconstruction and never the bench.

- **No point selected**: `No point selected` above *Go to Point...*, as now.
  Below the button, one line saying what the bench holds when it holds
  anything, *"3 items on the bench: double-click one in the Scene tree's Bench
  groups to edit it."*, and otherwise the line naming *Start cluster on the
  bench here* that the Track Edit empty state carries now.
- **Header**: the colour swatch, the Point ID in monospace with *Copy Point ID*
  and the *Go to Point* arrow beside it, `xyzw` with *Copy coordinates*, the
  error, the track length, *at infinity* when `w` is `0`, and the max pair
  angle, depth z and condition number where each is defined. Unchanged.
- **Stored-patch tile**, on a reconstruction that stores patch bitmaps.
  Unchanged.
- **The hint line** *"To work on this track: press ... in the Track Edit
  panel."* is **dropped**. The Edit checkbox is the thing it pointed at, and it
  is two rows above.
- **Observation table**: thumbnail with its error-coloured dot, the rendered
  patch tile on a reconstruction with patch frames, Image, Name, Feat #, Size,
  Error, Angle, Feature (x, y). Row click selects the image and reveals the
  feature in Image Detail; double-click enters camera view; hover drives the
  cross-panel hover. Unchanged.

A point at infinity is shown as it is now: `w = 0` in the coordinate, the
*at infinity* word, and the Error and Angle columns defined against the stored
direction.

### Edit mode

Edit mode is the Track Edit panel below the checkbox, **without its row of item
tabs**. It shows the active item and nothing else on the bench; the bench as a
list is the Scene tree's.

- **Header**: the item's label, its stage as a word, its origin as a point
  index or `new`, and `N in · M candidates · K out`. Under it the stage's own
  headline: at the cluster stage the reference observation and whether a
  template has been cut; at the track stage `Position (x, y, z)`, or `Bearing
  (x, y, z), at infinity` for a track at infinity, with the condition number,
  or the sentence saying nothing has triangulated it yet. Unchanged, and not
  joined by the view-mode summary line: the label of an item put on from a
  point already is the portable Point ID, the committed track is one Edit-clear
  away, and two modes that both showed the point's summary would be harder to
  tell apart at a glance, which is the thing the merge is for.
- **Toolbar, first row**: *Evaluate*, *Fit*, the *Stage* toggle, *Apply
  thresholds*, *Split off N rows*, *Duplicate*, *Commit*, *Discard*, each
  greyed with the step's own refusal as now. Unchanged.
- **Toolbar, second row**: *Rename*. *Put selected point on bench* is
  **dropped** from here: in edit mode it would stage a different point while the
  panel stays on this one, and the way to edit the selected point is the notice
  line below or the Edit box.
- **The selection notice**, new, drawn only when the node has a selected point
  that is not the active item's origin (followed to the cursor): one line,
  *"Selected: pt3d_a1b2c3d4_5120, not the track being edited."*, with *View*,
  which clears Edit, and *Edit it*, which puts that point on the bench.
- **Thresholds**: the four sliders and *search px*, seeded from the active
  track's own bars. Unchanged.
- **Observation table**: Verdict, tile, Img, Name, ZNCC, Seed sh., Proj. off,
  σ_pos, Error, Angle, Status, From, painted by what the sliders propose, with
  the row selection that *Split* reads and the row context menu carrying *Find
  matches by SIFT query* (or its build remedy) and, at the track stage, *Find
  matches by geometry*. Row click selects the image, reveals the observation
  and takes the row into the split selection; a mark clicked in Image Detail or
  in the 3D viewer selects its row. A **double-click** on a row now enters
  camera view for its image, as a view-mode row's does. The rows in both modes
  are observations of one track in one table position, and a gesture that works
  in one mode and silently does nothing in the other would be a trap; the
  verdict control keeps its own click, so nothing competes.

**A cluster and a track are the same panel at two stages.** The header's
headline, the table's cells and the row menu follow the stage, exactly as the
Track Edit panel does now; there is no separate cluster layout. What differs is
outside the panel: a cluster has no geometry, so the 3D viewer's bench layer
draws nothing for it and Image Detail draws its parallelograms rather than an
outline. A cluster has **no view mode** either, since there is no committed
point behind it, so the way back to a cluster after Edit has been cleared is its
row under *Bench Clusters*.

### Every feature, and where it goes

| Today | Where | In Track View |
|---|---|---|
| Point Track empty state and *Go to Point...* | Point Track | View mode empty state, plus the bench line |
| Point Track header, stored-patch tile, table, row gestures | Point Track | View mode, unchanged |
| *"To work on this track"* hint | Point Track | Dropped: the Edit checkbox replaces it |
| `remove_observation` in the response | Point Track spec only | Not carried: no code sets it, and the merged spec drops it |
| Item tabs, one per bench item, click to activate | Track Edit | **Dropped**: the Scene tree's Bench groups list the bench, and a double-click there activates |
| A tab's close mark (discard) | Track Edit | Dropped with the tabs; *Discard* stays in the toolbar and on the Scene tree row's menu |
| *No track on the bench* empty state | Track Edit | Gone as a state: with nothing active the panel is in view mode, whose empty state carries the same pointers |
| *Put selected point on bench* | Track Edit, both rows | The Edit checkbox in view mode; *Edit it* on the selection notice in edit mode |
| Header, toolbar, *Rename*, sliders, *search px*, table, row menu, row selection, painting | Track Edit | Edit mode, unchanged |
| Row click on a Track Edit row | Track Edit | Unchanged; double-click added |
| Looking for the node's SIFT index when the panel draws | Track Edit | Kept, in both modes, so a session finds the last one's index |
| Pre-caching photographs for the rows' tiles | dock, per panel | Kept, for whichever mode is drawn |

---

## The selection and the active item

The **selection** is the viewer's selected point, image and camera, which every
panel reads and which clicking in the 3D viewer or Image Detail moves. The
**active item** is the bench's, part of the version at the cursor. They were
independent before the merge and stay independent after it; what the merge
settles is which one Track View shows, and that is the box: view mode shows the
selection, edit mode shows the active item.

**Editing is sticky.** Selecting another point while editing moves the
selection, and with it the 3D viewer's track rays, the Image Browser's borders
and Image Detail's highlighted feature, but it does not change what Track View
shows and pushes no version: the person may be clicking around the scene to see
what else the patch covers, and a panel that dropped the edit every time would
make that impossible. The selection notice is what says the two have parted,
and its two buttons are the two things the person might want next.

**The common case keeps them together without asking.** Every way into editing
from a point selects that point first, and a commit selects the point it wrote,
so for a track that came from a point the selection and the item's origin agree
until the person clicks somewhere else, and the notice is absent.

**Go to Point** moves the selection and raises Track View, as it raises Point
Track now. In edit mode the panel stays on the item and the notice names the
point jumped to, with *View* one click away. The dialog does not clear Edit on
its own: it is a selection gesture, and making it a bench step would push a
version from a dialog that has never pushed one.

---

## The Scene tree

The Bench group rows keep their shape: one row per item, by label with its `in`
count, the active item drawn as a selected row, *Discard* on the secondary
click.

- **Double-click** makes the item active, turns edit mode on and brings Track
  View to the front (reopening it at its home when it is closed). It also
  selects the node the row is under, because Track View shows the selected
  node's bench and a raise onto another node's bench would show the wrong item.
  On an item already active it pushes no version and writes no no-effect row:
  the gesture asked for the panel, and the panel is what it gets.
- **Single click** selects the node the row is under and does nothing to the
  bench. A pass of single clicks down the tree then pushes no versions, and
  the node row's own pair, click to select and double-click to zoom, is the
  pattern the bench rows follow.

The raise is a layout operation, and the Scene tree is drawn inside a tab body,
where the dock is swapped out of the state and a raise would land on the
placeholder. So the tree's response carries the request and `app.rs` applies it
after the dock is back, the path Image Detail's *Edit on Bench* already takes.

---

## Other panels

**The bench layers follow the box.** The Image Detail panel's bench layer and
the 3D viewer's draw the active item and nothing else
([`../gui/multi-panel-image-browser.md`](../gui/multi-panel-image-browser.md)
§ "The bench layer",
[`../gui/viewer-3d-bench-layer.md`](../gui/viewer-3d-bench-layer.md)).
With Edit clear there is no active item, so neither draws anything and none of
their handles can be grabbed. This is what "no track is active for editing"
means in the rest of the window, and it needs no change in either layer: each
already draws nothing for a bench with no active track.

**Image Detail's *Add observation to bench track here*** greys while nothing is
active, with *"No track is being edited: tick Edit in Track View, or
double-click a Bench item in the Scene tree."* in place of today's sentence,
which assumes a non-empty bench always has an active track.

**Every raise names Track View.** *Edit on Bench* from either panel, Go to
Point, and *Start cluster on the bench here* raise `Tab::TrackView`, and
the Action Log's layout rows read `Raised Track View panel`, `Opened Track View
panel` and `Closed Track View panel`.

---

## The interface

The panel would live in `crates/sfm-explorer/src/track_view/`, holding the two
bodies it is made of as they are: `view/` from today's
[point_track_detail/](../../crates/sfm-explorer/src/point_track_detail/) and
`edit/` from today's [track_edit/](../../crates/sfm-explorer/src/track_edit/).
The activation step lives beside its siblings in
[bench.rs](../../crates/sfm-explorer/src/bench.rs), and the core half in
[bench/mod.rs](../../crates/sfmtool-core/src/bench/mod.rs).

```rust
// sfmtool_core::bench
impl Bench {
    /// The bench with no active item of `kind`, every item left where it is.
    pub fn deactivate(&self, kind: ItemKind) -> Bench;
}

// sfm-explorer
pub(crate) enum Tab { /* ... */ TrackView, /* ... */ }

pub struct TrackView {
    view: PointTrackView,   // today's PointTrackDetail, unchanged inside
    edit: TrackEdit,        // today's TrackEdit, minus the item tabs
}

impl TrackView {
    pub fn show(&mut self, ui: &mut egui::Ui, state: &AppState /* , ... */)
        -> TrackViewResponse;
    pub fn forget_recon(&mut self, id: ReconId);
    pub(crate) fn select_row(&mut self, id: ReconId, label: &str, observation: usize);
    pub(crate) fn selected_row(&self, id: ReconId, label: &str) -> Option<usize>;
}

pub struct TrackViewResponse {
    /// The box was ticked (from view mode) or cleared (from edit mode).
    pub set_edit: Option<bool>,
    /// The selection notice's *Edit it*.
    pub edit_selected_point: bool,
    pub view: PointTrackDetailResponse,
    pub edit: TrackEditResponse,   // without `activate` and
                                   // `put_selected_point_on_bench`
}

impl AppState {
    /// Leave every item on `id`'s bench and make none active: what clearing
    /// Edit does. One version; no effect, and no version, with nothing active.
    pub(crate) fn deactivate_bench_item(&mut self, id: ReconId) -> Result<(), String>;
}
```

### Why it is shaped this way

**Two bodies behind one tab, not one body.** The two panels' state is disjoint:
one caches thumbnails and patch tiles per image of a committed point, the other
caches tiles per observation of a bench track keyed on its `Arc`, and the
slider seeding, the painting and the commit refusal. Keeping each as the struct
it is means neither cache learns about the other and each panel's headless tests
move over nearly as they are. The merge is the checkbox, the dispatch on it and
the dropped rows.

**The mode is derived from the bench, so nothing about it is stored.** A panel
flag would have to be set by each of the half-dozen places that can change the
activation, and an Undo would leave it disagreeing with the bench at the cursor.
Reading `active_track_label` each frame costs a map lookup.

**Deactivation is a core function.** Today a non-empty bench always has an
active item, because `put` activates and `discard` hands the activation to a
neighbour, and nothing can take it away. The Edit box needs a bench that holds
items with none active, and the bench is core's value, so the state is made
there and the viewer's step is one call over it, like `activate`.

**The response keeps both halves' types.** The dock already knows how to apply
each; the merged response adds only the two gestures that are new.

---

## The wire

- **Panel names.** `track_view` replaces `point_track` and `track_edit` in
  `show_panel`, `hide_panel`, `screenshot`'s `panel_name` and
  `get_window_layout`'s `panels` object, which then has nine entries. A call
  naming a retired panel is refused with the unknown-panel sentence, which
  lists the names that exist. No surface takes the old names:
  `set_window_layout` validates with the layout reader, which refuses them as
  a saved file's load does. The wire name is the title lower-cased by the
  panel-layout rule, and an alias would be a second name for one panel on a
  surface whose error message lists the names that exist. The Scene tree's
  group names, *Bench Points* and *Bench Clusters*, stay as they are: they name
  the bench's contents, not the panel.
- **A new tool, `deactivate_bench_item`** `{ "reconstruction_label": "bull" }`:
  the Edit box cleared, answering as every bench step answers. With nothing
  active it is a no-effect reply, `changed: false`, and one row. It is named for
  what it acts on, as the glossary asks, and sits beside `activate_bench_item`
  in a listing; an `activate_bench_item` with a null `item` would give one tool
  two meanings chosen by a null.
- **`get_bench`'s `active.track` can be `null` on a bench that has items.** A
  reader that inferred "non-empty means something is active" was relying on
  today's invariant, which this removes.
- **A track tool that names no track, with nothing active**, is refused as it is
  now on an empty bench, the sentence gaining the remedy that now exists:
  *"No track is active on bull's bench. Name one with track, activate one with
  activate_bench_item, or put one on with create_bench_track or
  create_bench_cluster."*
- **Descriptions.** `activate_bench_item`, `get_bench_track` and the `track`
  argument's schema say "the item Track View is editing" where they say "the
  Track Edit panel" now. No tool's behaviour changes except by the invariant
  above.
- `create_bench_track` on a point already on the bench activates that item, as
  it does now, which is the Edit box's own rule.

---

## Standing specs that change

When this ships, [`../gui/point-track-detail.md`](../gui/point-track-detail.md)
and [`../gui/track-edit.md`](../gui/track-edit.md) are replaced by
`gui/track-view.md`, filed from this draft with both panels' content folded in
under the two modes, and both old files are deleted.

| Spec | Change |
|---|---|
| [`../core/bench/bench.md`](../core/bench/bench.md) | `Bench::deactivate`; a non-empty bench may have no active item; *discard* leaves nothing active; the Python binding gains `bench.deactivate(kind)` |
| [`../gui/bench.md`](../gui/bench.md) | `deactivate_bench_item` in the interface and its version label (`Stopped editing IMG_0042@142,198; it stays on the bench`); § "The Bench groups in the Scene tree" for the click and double-click; the wire tool count (this spec says twenty-eight and mcp-server.md twenty-nine today, which the filing reconciles); every "Track Edit panel" |
| [`../gui/scene-graph.md`](../gui/scene-graph.md) | The Bench group rows' gestures, the deferred raise, the response field for it |
| [`../gui/panel-layout.md`](../gui/panel-layout.md) | The panel table, the groups, the stock grid, the panel names listed in § "Validation"'s unknown-panel message, Go to Point's raise; the whole-file refusal of a renamed panel and the `LAYOUT_VERSION` row stay as they are |
| [`../gui/mcp-server.md`](../gui/mcp-server.md) | Panel names, the `get_window_layout` example (which lists eight panels today), the bench family's new tool and the `null` active |
| [`../gui/multi-panel-image-browser.md`](../gui/multi-panel-image-browser.md) | The layout diagram and panel list, *Edit on Bench*'s and *Start cluster*'s raise, the mark click selecting a Track View row, the *Add observation* refusal |
| [`../gui/viewer-3d-bench-layer.md`](../gui/viewer-3d-bench-layer.md), [`../gui/viewport-navigation.md`](../gui/viewport-navigation.md), [`../gui/goto-point.md`](../gui/goto-point.md), [`../gui/edits/commit-track.md`](../gui/edits/commit-track.md) | The panel's name at each mention, and the raise targets |
| [`../gui/README.md`](../gui/README.md) | One row for `track-view.md` in place of two |
| [`../gui/architecture.md`](../gui/architecture.md), [`../gui/user-experience.md`](../gui/user-experience.md), [`../gui/action-log.md`](../gui/action-log.md), [`../gui/background-tasks.md`](../gui/background-tasks.md), [`../gui/camera-intrinsics.md`](../gui/camera-intrinsics.md), [`../gui/document-model.md`](../gui/document-model.md), [`../gui/patch-rendering.md`](../gui/patch-rendering.md), [`../gui/sift-index.md`](../gui/sift-index.md), [`../gui/viewport-hud.md`](../gui/viewport-hud.md), the three other [`../gui/edits/`](../gui/edits/) specs that name either panel, [`../core/bench/editable-track.md`](../core/bench/editable-track.md), [`../core/patch/patch-cloud.md`](../core/patch/patch-cloud.md) | The panel's name and links |
| [`sfm-explorer-track-editing.md`](sfm-explorer-track-editing.md), [`sfm-explorer-editing.md`](sfm-explorer-editing.md), [`bench-inconsistent-fit-amendment.md`](bench-inconsistent-fit-amendment.md) | The panel's name and links |
| [`../../docs/tutorials/getting-started.md`](../../docs/tutorials/getting-started.md) | The panel's name |
| [`../GLOSSARY.md`](../GLOSSARY.md) | An entry under "The bench" for **active item** (the one item Track View edits; one per bench, whatever its stage) and **edit mode**, so "active" stops being used for a selected Scene tree row or a front tab in bench prose |

---

## Testing

- **Core** ([bench/tests.rs](../../crates/sfmtool-core/src/bench/tests.rs)):
  deactivating leaves every item the same `Arc` and no label active; activating
  afterwards restores one; discarding the active item leaves none active;
  the binding test covers `deactivate` and that it leaves the Python object it
  was called on unchanged.
- **The panel** (`track_view/tests.rs`, headless through `Context::run_ui`,
  absorbing both panels' tests): the box reads the bench, ticked with an active
  item and clear without one, and follows an undo of a deactivation with no
  panel call in between; ticking it over a selected point reports `set_edit`
  and, applied, is one version putting the point on the bench, and a second
  tick over the same point after a clear activates the existing item rather
  than putting a second one on; clearing it is one version and the next frame
  draws the view-mode header of the selected point; the box greyed with its
  sentence with no point selected and nothing active; edit mode drawing **no**
  item tabs, asserted by the labels of two other items on the bench appearing
  nowhere in what the frame painted; edit mode on a cluster drawing the cluster
  headline and cells; the selection notice drawn exactly when the selected point
  is not the item's origin, and its two buttons reporting their two gestures; a
  row double-click in edit mode asking for camera view; and every existing
  assertion of both panels, moved.
- **The Scene tree** (scene-graph tests): a single click on a Bench row selects
  the node and pushes no version; a double-click activates, selects the node and
  asks for the raise; a double-click on the active item pushes no version and
  writes no row; the raise survives the placeholder dock, as the existing test
  for *Edit on Bench*'s raise asserts in
  [state/edits/tests.rs](../../crates/sfm-explorer/src/state/edits/tests.rs).
- **Layout** ([layout/tests.rs](../../crates/sfm-explorer/src/layout/tests.rs)):
  the stock grid; `Tab::ALL` at nine; a file naming `point_track`, and one
  naming `track_edit`, refused whole with the unknown-panel message listing
  `track_view`, the dock untouched; the old stock layout, both names in one
  leaf, refused the same way; `track_view` twice refused; and the startup path,
  a default-layout file naming `point_track` loaded through `load_layout_file`
  on a freshly built state, leaving the stock grid with Track View in it and
  recording the failed `Layout` entry with its status-line message.
- **The wire** ([mcp/tests.rs](../../crates/sfm-explorer/src/mcp/tests.rs)):
  `show_panel` and `screenshot` with `track_view`, and a retired name refused
  with the list; `deactivate_bench_item` as one version and as a no-effect
  reply;
  `get_bench` reporting `null` active over a non-empty bench; a track tool with
  no `track` and nothing active refused in the new sentence.
- **`ui_basic`**
  ([tests/ui_basic.rs](../../crates/sfm-explorer/tests/ui_basic.rs)):
  the refusal for a panel behind another names "Track View" rather than "Point
  Track". No new windowed test: what the checkbox decides is covered headlessly,
  and the tab's presence by the layout test that walks every tab.

---

## Non-goals

- **Showing a committed track and its bench copy side by side.** The panel shows
  one or the other; clearing and ticking Edit flips between them, and each flip
  is a version. A second Track View is not possible, since a panel is a
  singleton.
- **Listing the bench in the panel.** The Scene tree's two Bench groups are the
  list, per node, with the stage each item is at.
- **Changing what either mode measures or draws.** The columns, the numbers, the
  tiles and every step are the two panels' as they stand.
