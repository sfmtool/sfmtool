# Track View with one body: the selected point read as an editable track (amendment)

**Status:** Draft. Decided: view mode's body is removed, and the edit body draws
both modes; with *Edit* clear, the panel shows the selected point as an
`EditableTrack` that is not on any bench; selecting another point leaves edit
mode; clearing *Edit* selects the item's point, or clears the point selection;
a strip of recently edited items sits to the right of the box; in read-only the
*Keep* column is a *Verdict* column coloured by the bars, and the threshold
boxes stay editable and judge without changing any verdict, and bars moved
there carry onto the bench when the viewed point is put on it; core's `Bench`
loses its active map, and the viewer's focused item is the only record of what is
being edited; there is one focused item for the viewer, named by a node and an item,
so an item with no point is focused like any other; with no point selected,
ticking *Edit* focuses the most recent item; view mode's leftover readings
move into the header and the crop's hover caption, and reconstructions with no
patch frames are left incomplete; `get_point` reports the viewed point's
evaluation; the bench layers draw only the focused item; the wire's
`activate_bench_item`, `deactivate_bench_item` and `get_bench`'s `active` are
renamed `focus_bench_item`, `unfocus_bench_item` and `focused_item`. Nothing is
open; § "Decisions" records each question and where its answer is.

Amends [`../gui/track-view.md`](../gui/track-view.md) (the panel),
[`../gui/bench.md`](../gui/bench.md) (activation, live evaluation, the selected
observations), [`../core/bench/bench.md`](../core/bench/bench.md) (item
identity, and the removal of the active map),
[`../core/bench/editable-track.md`](../core/bench/editable-track.md) (the steps
that no longer activate), [`../gui/goto-point.md`](../gui/goto-point.md),
[`../gui/scene-graph.md`](../gui/scene-graph.md),
[`../gui/edit-history.md`](../gui/edit-history.md),
[`../gui/mcp-server.md`](../gui/mcp-server.md) and the **active item** row of
[`../GLOSSARY.md`](../GLOSSARY.md).

## Purpose

Track View has two bodies behind one tab. View mode reads a committed point
through `metrics/` and draws thumbnails, a stored-patch tile and per-observation
errors. Edit mode reads the active bench item and draws crops, rendered tiles,
ZNCC and self-similarity readings, the bars' judgement and the verdict
switches. The two show the same kind of thing, one track, in two different
tables with two different sets of numbers. A person comparing a committed point
with its bench copy is comparing two layouts as well as two tracks, and a
committed point is never measured the way the bench measures a track until it
is put on the bench.

This draft keeps one body, the edit body, and makes *Edit* a switch between two
ways of drawing it:

- **Edit clear**: the selected point, read as an `EditableTrack` built with
  `create_track` and held outside any bench (the **viewed track**). It is
  evaluated live exactly as a bench track is, so every number in the table
  means the same thing in both modes. Nothing about it can be changed.
- **Edit ticked**: the active bench item, as today.

It also changes how the two modes follow the selection, so that clicking a
point in the 3D viewer always shows that point, and adds a strip of recently
edited items so the way back to an item is one click.

## What the user specified, and what this draft adds

The request fixed four things: the edit body is the only body; *Edit* clear
draws an off-bench `EditableTrack` for the selected point; a point selection
while editing leaves edit mode and shows the selected point; clearing *Edit*
selects the item's point or clears the selection; and the recent-items strip.

Working those through turns up the following, each covered in its own section:

1. **The activation has to leave the version** (§ "The focused item is not in the
   version"). Today activation and deactivation are bench steps. If a point
   click leaves edit mode, it would push a version, and Ctrl+Z would then undo
   "stopped editing" instead of the person's last verdict. This is the largest
   change in the draft.
2. **Bench items need a stable identity** (§ "Item identity"). The recent strip,
   the focused item and the selected observations are all keyed by label today,
   and a rename, or an undo across one, breaks a label key.
3. **A reconstruction with no patch frames loses its numbers**
   (§ "Reconstructions with no patch frames: incomplete"). View mode shows each
   observation's reprojection error on any reconstruction; the evaluation
   refuses a track-stage track with no frame, so the unified table shows `-`
   everywhere on a plain COLMAP import. This draft leaves that case incomplete.
4. **Some of view mode's readings have no place in the edit body**
   (§ "What view mode shows that the edit body does not").
5. **The verdict column and the bars mean something different in read-only**:
   *Keep* becomes a *Verdict* column showing what the bars say, and the boxes
   recolour the table without changing any verdict (§ "Drawing the viewed
   track").
6. **The live evaluation gets a second kind of subject** and a cache, since a
   person clicking through points would otherwise re-evaluate each one every
   time it is revisited (§ "Evaluating the viewed track").
7. **Several gestures that currently mention "tick Edit" or the selection
   notice change**, and the notice itself is removed (§ "Transitions").
8. **The wire** has two tools whose meaning changes and one reading it could
   gain (§ "The wire").

## The viewed track

When *Edit* is clear and a point is selected on the selected node, the panel
draws the **viewed track**: `create_track` applied to that point, in the
version at the cursor, held in `AppState` rather than on the bench.

```rust
/// The selected point read as an editable track, off every bench.
pub(crate) struct ViewedTrack {
    pub(crate) node: ReconId,
    pub(crate) point: u32,
    /// The document half it was read from. A point's content at a given
    /// document serial is fixed, so (node, point, document) is the key.
    pub(crate) document: VersionSerial,
    /// Labelled with the point's portable ID, as a put would label it.
    pub(crate) track: Arc<EditableTrack>,
    pub(crate) evaluation: Evaluation,
}

impl AppState {
    /// The viewed track for the selected point, built on first ask for a key
    /// and kept in a small cache (§ "Evaluating the viewed track").
    pub(crate) fn viewed_track(&self) -> Option<&ViewedTrack>;
}
```

**It is built the way a put builds a bench track**, so the rows arrive `in` and
pinned, the leave-one-out ZNCC is read back from `observation_confidence`, and
nothing else is measured until the evaluation lands. Ticking *Edit* over it puts
the same point on the bench, and the bench track starts from exactly what the
panel was showing.

**It is never written anywhere.** It is not in a version, the Scene tree does
not list it, the bench layers do not draw it, and no step accepts it. It is
rebuilt when the key changes: another point, another node, or a document edit
or an undo that moves the document serial.

**A deleted or missing point** gives no viewed track, and the panel takes the
empty state view mode takes today.

### Evaluating the viewed track

`drive_bench_evaluation` gains a second kind of subject. `Inputs` names either a
bench item or the viewed point:

```rust
enum Subject {
    Item { node: ReconId, item: ItemId },
    Viewed { node: ReconId, point: u32 },
}
```

The freshness rule is unchanged: the track's `Arc` and the document serial. A
landed result is installed into the `ViewedTrack`, not into the version, and
pushes nothing.

**Order.** The viewed track is evaluated first when *Edit* is clear, since it is
what the panel shows, and the focused item first when *Edit* is ticked. A
selection change while the viewed track is evaluating cancels that evaluation,
as a step on a bench track cancels its own.

**Cache.** A person clicking back and forth between points would otherwise
evaluate each one again on every visit. The last few viewed tracks (eight is
enough) are kept, keyed on `(node, point, document)`, with their evaluations.
The cache is dropped when its node is closed. A bench item's evaluation is not
shared with the viewed track even when they came from the same point, because
the bench copy may have been changed.

**Cost.** The evaluation needs the full-resolution photograph of every observing
image. View mode already decodes them for a reconstruction with patch frames,
so the decode cost does not change; the localization and self-similarity
readings are new work per selection. The panel shows *Evaluating…* while it
runs, as it does for a bench track.

**A busy node.** The bench's evaluation waits while a background task holds the
node. The viewed track follows the same rule; during a long bundle adjustment the
panel shows the point's stored numbers and *Evaluating…* until the task ends.

### Drawing the viewed track

The body draws the viewed track with the same header, table and hover views as
a bench track, with these differences:

| Part | Edit ticked (bench item) | Edit clear (viewed track) |
|---|---|---|
| Header label | the item's label, then the point ID when it differs | the point's portable ID, with copy and *Go to Point* |
| Header summary | stage, `N kept · K out · P pinned`, position or bearing | the point's colour swatch, `xyzw` with *Copy coordinates*, error, track length, max pair angle, depth z, cond (see below) |
| Toolbar | evaluation state; *Fit*, *Stage*, *Split*, *Duplicate*, *Commit*, *Discard*; *Lock*, *Rename* | evaluation state only |
| Threshold boxes | the item's bars; release applies them as one version | drawn and editable; they judge the readings and the *Verdict* column and change nothing (below) |
| Verdict column | *Keep*: switch, pin, tinted by what the bars propose | *Verdict*: the word `in` or `out`, in a cell tinted green or red by it, the verdict the boxes' bars give the row. No switch, no pin (below) |
| Heading pin | on the *Keep* heading | absent |
| Row click | select image, reveal, pick into the split selection | select image, reveal. No row selection |
| Row double-click | camera view toward the observation | same |
| Row context menu | searches, *Accept walk*, unpin | absent |
| *From* column | provenance | absent: every row reads `point` |

**The *Verdict* column takes the *Keep* column's place.** A committed point's
observations are all in the track, so a switch showing that would say the same
thing on every row, and nothing here could change it. The column shows instead
what the bars say about each observation: `in` in a green cell where the row
clears every bar and holds its image, `out` in a red cell where it does not, and
an untinted `-` where nothing has measured the row yet. It is the verdict
`verdicts_if_unpinned` gives the row, which is the verdict the bench's own
evaluation would give it once unpinned, so the green and red are the same
green and red the *Keep* cell is tinted with in edit mode. Its hover text is
the *Keep* switch's second half: that the row clears every bar, or which bars
it fails, named as the headings name the readings (`ZNCC whole is under the
bar`, `Shift is over the bar`), or which other sighting holds its image.

**The five threshold boxes are drawn and editable, and change only what is
drawn.** Dragging a box or typing into it moves the bar, and every judged
reading (the two ZNCC lines, *Shift*, the self-similarity *whole* line) and
every *Verdict* cell is recoloured live, as a drag recolours the table in edit
mode. Letting go applies nothing: the viewed track's verdicts stay `in`, no
version is pushed and no Action Log row is written, and the busy state does not
grey the boxes. The boxes keep their values for the session, across selection
changes, so a person can set a strict bar and click through points to see which
observations each one would lose. They start at the bench's default bars.

**Moved bars carry onto the bench with the point.** When the boxes hold bars
other than the defaults and the viewed point is put on the bench, by ticking
*Edit*, *Edit on Bench*, a double-click on the point or one of its features, or
the wire's `create_bench_track` naming it, the new track takes the boxes' bars
in place of the defaults. So the table keeps the colours it had at the moment
editing starts, and the first evaluation paints the unpinned rows by the bars
the person was reading by. The put's version label and the wire's reply name the
bars when they are not the defaults (`Put point 1207 on the bench as
pt3d_a1b2c3d4_1207, with min ZNCC 80%`). Only the viewed point carries them:
a put of any other point, and a cluster, start at the defaults. A point that
already has an item on the bench is focused rather than put on, and keeps that
item's own bars, which were set on it. The boxes keep their values after the
put, for the next point viewed.

A line under the header says how to change the track: *"Tick Edit to work on
this track."* When the point already has an item on the bench (an item whose
origin followed to the cursor is this point), the line names it instead: *"On
the bench as pt3d_a1b2c3d4_1207 (edited). Tick Edit to open it."* The two can
differ, and this is the only place a person looking at a point learns that a
changed copy of it is waiting.

## What view mode shows that the edit body does not

| View mode | In the unified body |
|---|---|
| Colour swatch, `xyzw` with copy, RMS error, track length, max pair angle, depth z, cond | In the header, in both modes: for the viewed track, and for a bench item at the track stage from the track's own position, so the header reads the same either way. A cluster has no position and shows none of them |
| Stored-patch tile (64 px) | Already the header's patch slot, through the same `stored_patch_image` |
| Thumbnail with an error-coloured dot | Replaced by the *Crop* column |
| *Patch* column (tile re-anchored on the keypoint) | Already the *Patch* column at the track stage, through the same `patch_color_image` |
| *Img*, *Name* | Same |
| *Feat #* | In the crop's hover caption |
| *Size* (the two extents of the affine shape) | Dropped: the crop's hover caption gives the patch's two axes in the photograph's pixels |
| *Error* | The *Proj. err* column's first line |
| *Angle* | The *Proj. err* column's second line (`ray_angle_deg` is the same angle) |
| *Feature (x, y)* | In the crop's hover caption |

The table also gains columns view mode never had for a committed point: *ZNCC*
with its grid, *Self-similarity* with its grid and surface plot, *Shift* and
*Status*.

## Reconstructions with no patch frames: incomplete

This draft does not finish Track View for a reconstruction whose points carry no
patch frame, which is the usual case for a `sift_files` reconstruction imported
from COLMAP. What the unified body does there is what it already does for a
frameless bench track, and it is less than view mode showed:

- `create_track` builds a track-stage track without a frame, and
  `evaluate_preconditions` refuses it, so the toolbar shows the refusal
  sentence (*"the track carries no patch frame to read against"*) and every
  number cell reads `-`, the reprojection error and ray angle included. View
  mode showed those two for any reconstruction, since they need only the pose,
  the camera and the keypoint.
- The *Crop* and *Patch* cells are empty frames, and there is no crop hover
  caption to carry the pixel and the feature index.
- The header still carries the point's summary (position, error, track length,
  max pair angle, depth z, cond), which reads the point and needs no frame.

Finishing it needs its own design: which readings a frameless evaluation should
produce, what the *Crop* cell shows without a frame (for instance the `.sift`
feature's affine shape), and how the cluster stage, which a frameless track is
taken through to get a frame, fits in. It is left to a later draft, and the
standing spec will say in one sentence that a frameless point shows its header
and no per-observation readings, linking that draft.

## The focused item is not in the version

Today the active item is part of the bench value. `activate` and `deactivate`
are bench steps: each pushes a version and writes a `Bench` row, and an undo
restores what was active.

The new rule, that selecting another point leaves edit mode, makes that
unworkable:

- **A selection would push a version.** Selecting is not a step anywhere else
  in the viewer, and the point selection follows the history rather than being
  recorded in it.
- **Undo would do the wrong thing.** A person turns an observation out, clicks
  a point in the 3D viewer to look at it, and presses Ctrl+Z to take the
  verdict back. With deactivation as a version, the undo restores edit mode and
  leaves the verdict where it is. A second press is needed, and nothing on
  screen says why the first did not work.
- **The strip would push a version per click**, since each chip is an
  activation.

So the activation becomes view state, held beside the selection:

```rust
/// What Track View edits while Edit is ticked. At most one in the viewer.
pub(crate) struct FocusedItem {
    pub(crate) node: ReconId,
    pub(crate) item: ItemId,
}

impl AppState {
    pub(crate) fn focused_item(&self) -> Option<&FocusedItem>;
    /// Focus the item and select its point, or clear the point
    /// selection when it has none. No version; one Selection row.
    pub(crate) fn focus_bench_item(&mut self, id: ReconId, item: ItemId) -> Result<(), String>;
    /// Unfocus the focused item, and select the item's point or clear the point
    /// selection (§ "Transitions"). No version; one Selection row.
    pub(crate) fn unfocus_bench_item(&mut self);
}
```

What this means elsewhere:

- **Steps that put an item on the bench still focus it**: a put from a point,
  *Start cluster*, *Create Track Here*, *Find Nearby Tracks* (its `1a`),
  *Duplicate*, *Split*. The step pushes its version; the item is focused beside it.
- **Undo and redo no longer restore the focused item.** They leave the item focused while it
  exists at the cursor, and unfocus it when the version landed on does not
  hold the item (an undo past its put, a redo past its discard). The panel then
  shows the selection.
- **The Edit History panel and the Action Log lose the `Made … the active
  track` and `Stopped editing …` versions.** The Action Log records a change
  of focused item as a `Selection` row (`Editing pt3d_a1b2c3d4_1207`, `Stopped editing
  pt3d_a1b2c3d4_1207`), folded like other selection rows.
- **There is one focused item for the viewer, not one per node.** Today each
  node's bench has its own active item and the panel shows the selected node's.
  With the focused item following the selection, one for the viewer is the
  simpler rule: selecting a point in another node leaves edit mode like any
  other point selection. A `FocusedItem` names a node and an item on that node's
  bench, and never a point, so an
  item that corresponds to no point in the reconstruction is focused the same
  way as one that does (§ "An item with no point").
- **The busy rule relaxes.** Focusing an item that is already on the bench, and
  unfocusing the item, are no longer bench steps, so a background task on the
  node does not grey them. Ticking *Edit* over a point that is not yet on the
  bench is still a put, and still greyed.
- **Core's active map is removed.** Core's `Bench` holds an active label per
  kind today, and `put` activates what it puts on. With the viewer keeping its
  own focused item, a second record of "which item is active" in the same process
  could disagree with it, which is the disagreement the current spec avoids by
  deriving the box from the bench. So `Bench` loses the `active` field,
  `active_label`, `active_track`, `activate` and `deactivate`, and `put`,
  `discard` and `rename` no longer touch an activation. A bench is then only
  its list of items, and two benches are equal when their items are.
  Which item a caller is working on is the caller's to hold: the viewer's focused
  item, or a Python script's own variable. The core steps that put an item on
  (`create_track`, `create_cluster`, `split`, `duplicate`) already return the
  label they minted, which is what the viewer focuses. `ItemKind` stays, since
  step reports carry it. Nothing in `src/sfmtool/` reads the activation; the
  Python bindings lose the four calls and
  `tests/rust_bindings/test_bench_rust_bindings.py` loses the tests of them,
  as do the core bench tests of activation.

### The words

*Focus* is only ever a verb, and the thing it produces is only ever the
**focused item**. So the actions are **focus** (make an item the one Track View
edits) and **unfocus** (leave no item being edited), and the state they change is the
**focused item**, of which there is at most one. No sentence, type, field or
tool uses *focus* as a noun: not "the focus", "clear the focus" or "edit
focus". The names follow the rule:

| Kind | Action | State |
|---|---|---|
| Prose | focus an item, unfocus it | the focused item; an item is focused |
| `AppState` | `focus_bench_item`, `unfocus_bench_item` | `focused_item()` returning `Option<&FocusedItem>` |
| Panel response | `focus_item: Option<(ReconId, ItemId)>` | |
| Wire | `focus_bench_item`, `unfocus_bench_item` | `get_bench`'s `focused_item` |
| Action Log | `Editing pt3d_a1b2c3d4_1207`, `Stopped editing pt3d_a1b2c3d4_1207` | |

The Action Log keeps the words the *Edit* box uses, since that is the control a
person sees.

## Transitions

The invariant the rules keep: **while Edit is ticked, the selected node is the
focused item's node, and the selected point is the item's origin followed to the
cursor, or no point.** That removes the case the selection notice exists for,
so the notice is removed.

| From | Gesture | What happens | Then shown |
|---|---|---|---|
| Viewing a point | Tick Edit | The point is put on the bench and focused, or its existing item is focused. The first is a version; the second is not | Edit, on that item |
| Viewing, no point selected | Tick Edit | The most recently focused item still on the bench at the cursor is focused, selecting its node; with none, the box is greyed | Edit, on that item |
| Editing | Clear Edit | The item is unfocused. Its origin is selected when it is live at the cursor; otherwise the point selection is cleared | The origin's viewed track, or the empty state |
| Editing | Select another point: a click in the 3D viewer or Image Detail, *Go to Point*, the wire's `select_point` | The item is unfocused and the new point is selected | That point's viewed track |
| Editing | Select the item's own origin | Nothing changes | Edit, same item |
| Editing | Clear the point selection (a click on empty space) | The item stays focused | Edit, same item |
| Editing | Select an image or a camera of the focused item's node (a row click, a thumbnail, a frustum) | The item stays focused | Edit, same item |
| Editing | Select another node, or an image, camera or point of another node (a Scene tree click, the Image Browser, the 3D viewer, the wire) | The item is unfocused, and the selection is what the gesture made it | The new selection's viewed track, or the empty state |
| Either | Click a chip in the strip | That item is focused, and its origin selected or the point selection cleared | Edit, on that item |
| Either | Double-click a Bench row in the Scene tree | The same as a chip, and the panel is raised | Edit, on that item |
| Either | *Edit on Bench*, double-click a point or a feature | The point is selected (which unfocuses any other item), then put on or focused | Edit, on that point's item |
| Either | *Start cluster on the bench here* | A cluster is put on and focused; the point selection is cleared | Edit, on the cluster |
| Either | *Create Track Here*, *Find Nearby Tracks* | As today; the commit selects the written point, which is the item's re-seated origin, so the item stays focused | Edit, on the new item |
| Editing | *Commit* | The origin is re-seated on the written point and that point selected, so the item stays focused | Edit, same item |
| Editing | *Discard* | The item leaves the bench (a version), and the item is unfocused as for a cleared box | The origin's viewed track, or the empty state |
| Editing | *Duplicate*, *Split off N rows* | The new item is focused; it has no origin, so the point selection is cleared | Edit, on the new item |
| Either | Undo or redo | The item stays focused while it exists at the cursor; otherwise it is unfocused | |

**Where the exit rule lives.** In `AppState::select_point`: selecting a point
other than the focused item's origin unfocuses the item. Every point selection
gesture already goes through it, and so does the commit, which re-seats the
origin before it selects, so the commit needs no exception. The node rule lives
in the calls that can move `selected_recon` (`select_point`, `select_image`,
`select_camera`, and the Scene tree's node selection): a move to a node other
than the focused item's unfocuses the item. The selection that follows
the point maps after an edit assigns `selected_point` directly and does not go
through `select_point`, so a map that moves the selection does not unfocus the
item.

**Accidental exits.** A person working on a track may click a SIFT feature in
Image Detail that belongs to another point and leave edit mode without meaning
to. The strip puts the item one click away, and the exit costs no version, so
nothing is lost. Whether a click on a bench-layer handle can also reach the
feature picking under it needs checking in Image Detail's input order before
this is built; if it can, the handle has to win.

### An item with no point

Several items correspond to no point in the reconstruction: a cluster, which
has no position; a track made by *Duplicate* or *Split*, which has no origin;
a track put on from a point that has since been deleted, whose origin no longer
resolves; and a commit that is undone, which takes the origin its commit gave
the item back off it. Each is still an item on one node's bench, and a
`FocusedItem` names the node and the item, so one focused item for the viewer
covers them:

- **Focusing one** (a chip, a Scene tree double-click, *Start cluster*, a
  duplicate or a split) selects its node and clears the point selection. The
  selected image stays when it belongs to that node, so a person starting a
  cluster in Image Detail keeps looking at the image they started it in.
- **While it is focused**, any point selection is a selection of a point other
  than its origin, so a click on any point leaves edit mode. Selecting an image
  or a camera of its node keeps the item focused, which is how a person moves between
  the photographs a cluster is being grown in. Selecting anything on another
  node clears it (the table above).
- **Clearing Edit** unfocuses the item and leaves the point selection empty; the
  node stays selected. The panel shows the empty state: *No point selected*,
  *Go to Point...*, and the strip, whose first chip is the item just left.
  Ticking *Edit* again focuses it (§ "Ticking Edit with no point selected").
- **Discarding it** does the same.
- **The item gains a point when it is committed**: the commit re-seats its
  origin on the point it wrote and selects that point, so the invariant holds
  from then on and clearing *Edit* selects the written point. An undo of the
  commit removes the point and takes the origin back off the item, the point
  selection follows the map to no point, and the item stays focused.
- **The 3D viewer draws no bench figure for a cluster**, as today, since a
  cluster has no world geometry. With no point selected there are no track
  rays either, so the 3D viewer shows nothing of the item; Image Detail's bench
  layer and Track View are where it is seen.

## Ticking Edit with no point selected

Today the box is greyed with no point selected and nothing active, because "the
last item" was recorded nowhere a person could see. The strip now shows it, so
the tick focuses it: the first entry of the recent list that is on the bench at
the cursor, which is the strip's first chip when the panel is wide enough to
draw one. It is on any loaded node, and focusing it selects that node. The
box's hover text names it (*"Tick to edit pt3d_a1b2c3d4_1207 again"*), so a
panel too narrow to draw the chip still says what the tick will do. Focusing an
item already on the bench is not a bench step, so a busy node does not grey
the tick.

The box is greyed only when the list has no entry on any bench, and its refusal
names the ways in, as now: *"Nothing to edit: select a point, double-click an
item in the Scene tree's Bench groups, or right-click a pixel in Image Detail
and choose "Start cluster on the bench here"."*

## The recent items strip

To the right of the *Edit* box, on the same row, the items most recently
focused, most recent first, excluding the focused item.

**What a chip shows.** The item's patch at 24 points square, the image the
header's patch slot draws (the consensus bitmap at the track stage, the
template at the cluster stage, an empty frame when there is neither), and the
label beside it, shortened by a cut out of the middle so that the start and the
end both survive (`pt3d_…_1207`, `IMG_0…@142,198`), using the Name column's
elision.

**How many.** As many as fit in the width left on the row, up to eight. A chip
that would not fit whole is not drawn. On a narrow panel that may be none.

**Hover** shows the patch at 64 points and: the whole label; the node's name
when more than one reconstruction is loaded; the stage; `N kept · K out · P
pinned`; the origin's point ID, or `from point N` when it is gone, or `new`;
`Position (…)` or `Bearing (…), at infinity`; the evaluation state; and *Click
to edit*.

**Click** focuses the item (§ "Transitions"), selecting its node first when it
is on another node.

**What goes on the list.** An item is added, or moved to the front, when it
is focused, whatever focused it. It is removed when its node is
closed. An item not on the bench at the cursor is skipped at draw time rather
than removed, so an undo of its discard brings its chip back. The list itself
is not cut at eight: the strip draws the first eight entries that are on the
bench at the cursor, so skipped entries do not leave it short.

**It is session state.** Not in a version, not saved with the layout, not on
the wire as a list (see § "The wire").

**Relation to the Scene tree.** The Bench groups remain the full list. The strip
is only a way back to the last few.

## Item identity

The focused item, the strip, `BenchRows` and the live evaluation's `Inputs` all
key bench items by label today. A label changes on rename, and an undo across a
rename changes it back, so each of those has special handling (`BenchRows`
carries the selection over a rename) or none (the strip would lose an entry).

**Proposal:** `BenchEntry` gains an `ItemId`, a `u64` minted by the bench when
an item is put on and never reused within that bench's history. `replace` and
`rename` keep it. `put`, `duplicate` and `split` mint a new one. The label stays
what every sentence and every wire call names an item by; the ID is what the
viewer holds on to between frames.

This is a core change in [`../core/bench/bench.md`](../core/bench/bench.md), and
the bindings would expose it read-only as `bench.id(label)`.

## The wire

- **`focus_bench_item`** `{ "reconstruction_label": "bull", "item":
  "pt3d_a1b2c3d4_1207" }`, the parameters `activate_bench_item` took, replaces `activate_bench_item`. It focuses the item:
  no version, `changed` true when the focused item changed, and a `Selection`
  row. It selects the item's node and origin, or clears the point selection, as
  a chip click does.
- **`unfocus_bench_item`** `{}` replaces `deactivate_bench_item`. It unfocuses
  the item, as the cleared box does: no version, and a no-effect reply when
  nothing is focused. It takes no reconstruction, since there is one focused
  item for the viewer.
- **The old names are refused** with the unknown-tool error, which lists the
  tools that exist; the catalog carries no alias. Both tools changed behaviour
  (neither pushes a version now), and an agent calling a changed tool under its
  old name would expect the old behaviour, where a refusal makes it look the
  new name up.
- **`select_point`** unfocuses the item when the point is not the focused item's
  origin. An agent that selects points while working on a track names the track
  on every bench call, which the bench tools already allow.
- **`get_bench`'s `focused_item`** replaces `active`: the focused item's label
  when it is on that node, and `null` otherwise.
- **A bench tool that names no track** acts on the focused item, and is refused
  when the focused item is on another node or there is none, with the current sentence.
- **Reading the viewed track.** `get_point` gains an `evaluation` block when
  the point asked for is the viewed point, so an agent reads the numbers a
  person sees without putting the point on the bench. The block has
  `get_bench_track`'s shape for the per-observation measurements, with
  `evaluation.state` (`current`, `evaluating`, `refused`, `failed`) and its
  reason as the bench reports them, and the read-only bars with each row's
  *Verdict* (`in` or `out` by the bars). While the state is `evaluating` the
  measurements are the last ones landed, as the panel greys them. For any other
  point the block is absent: there is no call that evaluates an arbitrary point
  on request, since that is a separate operation with a cost.

## Other panels

- **The bench layers** in Image Detail and the 3D viewer draw the focused item,
  bench track or cluster, and nothing while *Edit* is clear, as now. The viewed
  track gets no projected outline in Image Detail and no figure in the 3D
  viewer. With *Edit* clear, the photographs show the selected point as they do
  today (its highlighted feature in Image Detail, its track rays in the 3D
  viewer), and the patch as it lies in each photograph is seen in Track View's
  *Crop* column. A read-only bench layer would need its own way of reading as
  not editable at a glance, and is left out of this change.
- **Image Detail's *Add observation to bench track here*** greys with
  *"No track is being edited: tick Edit in Track View, click a recent item
  beside it, or double-click a Bench item in the Scene tree."*
- **Ctrl+D** duplicates the focused item.
- **Go to Point** selects the point, which unfocuses the item by the ordinary
  rule. Its spec's sentence that the dialog does not clear Edit is replaced.

## The code

- `track_view/view/` is removed: `prepare.rs`, `header.rs`, `table.rs`.
  `patch.rs`'s `patch_color_image` and `stored_patch_image` move into the body
  that draws both modes, since that body is their only caller.
- `track_view/edit/` becomes the one body and is renamed to say so
  (`track_view/body/`, or its files moved up into `track_view/`). It takes a
  mode, `Viewed` or `Edited`, and the differences in § "Drawing the viewed
  track" are branches on it.
- `PointTrackViewResponse` is removed. `TrackViewResponse` loses `view` and
  `edit_selected_point`, and gains `focus_item: Option<(ReconId, ItemId)>`, the item a chip click asks to focus
  click. `TrackEditResponse`'s gestures that name an item are only reported in
  edit mode.
- `metrics/` stays: Image Detail's overlay and `get_point` read it.
- The view-mode header summary (max pair angle, depth z, cond) moves into the
  body's header.
- `bench::live` takes a `Subject` and gains the viewed-track cache.
- `AppState` gains `focused_item` and `recent_items`, and `select_point` gains the
  exit rule. The bench's `active` is no longer read by the viewer.

## Testing

- **Panel**, headless: *Edit* clear with a point selected draws the unified
  body with the point's ID and no toolbar buttons, no *Keep* switch and no pin;
  the table's columns match edit mode's at the track stage with *Verdict* in
  place of *Keep*; each *Verdict* cell reading `in` or `out` as
  `verdicts_if_unpinned` gives it, green or red to match, and `-` untinted on an
  unmeasured row, with the failing bar in its hover text; a drag and a release
  of each box recolouring the readings and the *Verdict* cells while every
  verdict stays `in`, no version is pushed and no row is written; the boxes'
  values surviving a selection change; ticking *Edit* after moving a box
  putting the point on with the boxes' bars and naming them in the version
  label, with the table's colours unchanged across the tick; the same through
  *Edit on Bench* and `create_bench_track`; unmoved boxes putting the point on
  at the defaults; a put of a point other than the viewed one, and a cluster,
  taking the defaults; ticking *Edit* on a point with an existing item keeping
  that item's bars; *Evaluating…* until the viewed track's
  evaluation lands and *Evaluated* after; the crop's hover caption giving the
  observation's pixel and feature index; the header summary for the viewed
  track and for a bench item at the track stage, and absent for a cluster; a
  frameless point drawn with its refusal sentence, empty *Crop* and *Patch*
  frames and `-` cells; the "on the bench as …" line when an
  item from the point exists.
- **Bench layers**: with *Edit* clear and a viewed track drawn in Track View,
  Image Detail's bench layer and the 3D viewer's bench figure draw nothing and
  take no drag; ticking *Edit* on that point draws its outline and figure.
- **Transitions**: each row of the table in § "Transitions", including that a
  point click while editing unfocuses the item and pushes no version; that
  selecting the origin, clearing the point selection or selecting an image
  keeps it; that clearing *Edit* selects the origin or clears the point
  selection; that a commit keeps the item focused; that an undo past the item's put
  clears it and an undo of a verdict after a point click undoes the verdict;
  that selecting an image, camera, point or the node row of another node clears
  the item while an image of the focused item's node keeps it focused; and, for a cluster, a
  duplicate and an item whose origin was deleted, that focusing it selects its
  node, clears the point selection and keeps an image of that node selected,
  that clearing *Edit* leaves the node selected and no point, that a commit
  selects the written point and keeps the item focused, and that an undo of that
  commit leaves the item focused and no point selected.
- **Strip**: most recent first, the focused item excluded; at most eight and fewer on a
  narrow panel; a chip click focusing the item and selecting its origin; a
  rename keeping the chip; a discard hiding the chip and its undo bringing it
  back; a chip for an item on another node selecting that node; with no point
  selected, a tick focusing the most recent item on the bench, its name in the
  box's hover text, on a panel too narrow for any chip too, a discarded most
  recent item skipped for the next, and the box greyed with the ways-in
  sentence only when no entry is on any bench.
- **Live evaluation**: the viewed track evaluated first when *Edit* is clear; a
  selection change cancelling it; a revisit within the cache drawing current
  numbers without a new evaluation; a document edit rebuilding it.
- **Core**: `ItemId` kept by `replace` and `rename`, fresh on `put`,
  `duplicate` and `split`; `put`, `discard` and `rename` leaving nothing but
  the list changed. The activation tests in
  `bench/tests.rs` and `test_bench_rust_bindings.py` are removed.
- **Wire**: `focus_bench_item` and `unfocus_bench_item` pushing no version,
  and `unfocus_bench_item` with nothing focused replying no effect;
  `activate_bench_item` and `deactivate_bench_item` refused as unknown tools;
  `get_bench` reporting `focused_item` on the focused item's node and `null` on
  another; `select_point` unfocusing the item; `get_point` on the viewed point
  carrying the `evaluation` block, with `evaluating` and then `current`, the
  read-only bars and each row's verdict by them, and the same call on any other
  point carrying none.

## Decisions

The questions the first version of this draft left open, each now decided.

**Q1. Decided:** core's active map is removed (§ "The focused item is not in the
version", "Core's active map").

**Q2. Decided:** one focused item for the viewer, naming a node and an item, so
an item with no point is focused like any other (§ "An item with no point").
The glossary's **active item** row ("one per bench") is replaced by
**focused item** and the verbs **focus** / **unfocus** (§ "The words").

**Q3. Decided:** moved read-only bars carry onto the bench when the viewed
point is put on it (§ "Drawing the viewed track").

**Q4. Decided:** with no point selected, ticking *Edit* focuses the most
recent item (§ "Ticking Edit with no point selected").

**Q5. Decided:** the pixel and the feature index go in the crop's hover
caption, *Size* is dropped, and the header summary is shown for bench items at
the track stage too (§ "What view mode shows that the edit body does not").
Reconstructions with no patch frames are left incomplete (§ "Reconstructions
with no patch frames: incomplete").

**Q6. Decided:** `get_point` carries an `evaluation` block for the viewed
point only (§ "The wire").

**Q7. Decided:** the bench layers draw only the focused item; the viewed track
gets no projected outline in Image Detail and no figure in the 3D viewer
(§ "Other panels").

**Q8. Decided:** the wire's tools and field are renamed by § "The words", to
`focus_bench_item`, `unfocus_bench_item` and `get_bench`'s `focused_item`
(§ "The wire").

## Spec changes on filing

- [`../gui/track-view.md`](../gui/track-view.md): rewritten around one body;
  § "View mode" removed; § "The Edit checkbox" and § "The selection and the
  active item" replaced by this draft's transitions and strip; the selection
  notice removed; the *Why* paragraphs that argue for the versioned activation
  and for two bodies removed.
- [`../gui/bench.md`](../gui/bench.md): the activation and deactivation rows of
  § "What a step writes" removed; § "Live evaluation" gains the viewed track;
  § "The selected observations" keyed by `ItemId` and cleared when the focused item changes.
- [`../core/bench/bench.md`](../core/bench/bench.md): `ItemId`; the active map,
  its methods and the "A non-empty bench may have no active item" and discard
  paragraphs removed; the Python bindings section without the four calls.
- [`../core/bench/editable-track.md`](../core/bench/editable-track.md): the
  sentences saying a split, a duplicate or a create makes its item the active
  one, rewritten to say the step returns the label it minted.
- [`../gui/goto-point.md`](../gui/goto-point.md),
  [`../gui/scene-graph.md`](../gui/scene-graph.md),
  [`../gui/edit-history.md`](../gui/edit-history.md),
  [`../gui/mcp-server.md`](../gui/mcp-server.md),
  [`../gui/multi-panel-image-browser.md`](../gui/multi-panel-image-browser.md):
  the sentences named in § "Other panels" and § "The wire".
- [`../GLOSSARY.md`](../GLOSSARY.md): the **active item** and **activate** /
  **deactivate** rows replaced by **focused item** with **focus** / **unfocus**
  and the wire's `focus_bench_item`, `unfocus_bench_item` and `focused_item`;
  **Track View** rewritten; **viewed track** and **recent items strip** added.
