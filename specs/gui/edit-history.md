# The edit history: versions, a cursor, and the maps between them

Every change made to a loaded reconstruction in the viewer can be taken back,
and taken back again, without the viewer having to know how to reverse anything.
It works because a change does not modify what was there: it produces the next
value beside it, and the node remembers both. Undo is then a cursor moving one
step back along a list of values, and redo is the same cursor moving forward.

The list is not only values. Every step between two versions also carries a
small record of **what it did to point indexes** -- which indexes stopped
resolving, and where the survivors went. That record is what lets something
holding an index across an edit follow it: a selected point stays selected
through an edit that renumbered the cloud around it, and comes back to the same
point when the edit is undone.

This spec describes the cursor's behaviour, the maps, what follows them, and the
Edit History panel the list is read and walked in. What
a version *is*, and how an edit is applied, is
[document-model.md](document-model.md).

## The cursor

A node's history is a list of versions, oldest first, and a cursor into it. The
node shows the version at the cursor.

- **An edit** appends a version after the cursor and moves the cursor onto it.
- **Undo** moves the cursor one version back; **redo** moves it one forward.
  Neither adds, removes or reorders a version: walking the cursor is not an
  edit, so undoing and redoing a step leaves the list exactly as it was.
- **A new edit at a cursor that is not at the end discards the versions after
  it.** The redo tail's values are dropped; their maps are not. This is the
  Photoshop rule, and a branching history is a non-goal: a second dimension to
  navigate costs more to explain than the branch is worth.
- **Undo and redo refuse at the ends,** and refuse rather than step onto a
  version whose value the budget has released.

The history is **per node**. Each loaded reconstruction carries its own versions
and cursor; the Edit menu and its shortcuts act on the selected reconstruction.
A scene-wide cursor would make undoing a change in one node silently undo the
last change in another.

## The maps

Each version but the first was made from exactly one other, and the step between
them carries a `PointMap`
([document.rs](../../crates/sfm-explorer/src/document.rs)):

```rust
pub enum PointMap {
    /// Indexes are stable across this step; these ones stopped resolving.
    Removed(Vec<u32>),
    /// A whole-value edit's row map: a materialisation's, or the one
    /// `RowMap::by_scan` reads off a bulk edit's input and output.
    Rows(RowMap),
    /// The steps one edit took, applied in order.
    Chain(Vec<PointMap>),
}

impl PointMap {
    /// Where an index before this step lands after it.
    pub fn forward(&self, before: u32) -> Option<u32>;
    /// Where an index after this step came from.
    pub fn inverse(&self, after: u32) -> Option<u32>;
}
```

Each case is stored as what it is rather than as a pair of dense arrays, so a map
costs the size of the edit that made it: a list of the indexes one point edit
removed, a row map that is a sorted list of holes plus one entry per addition, or
the two or three steps a bulk edit took. A point deletion's map is one `u32`; an
image deletion's is a materialisation's row map chained with the one scanned off
the image subset's input and output
([`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md)).

**The maps are kept for every version ever minted**, including versions a new
edit after an undo discarded. They are small enough that a node holds the whole
graph of its session even after the budget has released the values, and keeping
them is what lets an index taken at any version -- a copied point id, a
constraint written against an earlier state -- still be resolved against the
version on screen.

Every serial is minted once and never reused, so a map keyed on a pair of serials
names one step for the life of the session, and a discarded version's map can
never be mistaken for a later one's.

## The version graph

The maps are not a list beside the versions but a **graph over them**. The
history keeps a step list with one entry per version ever minted, holding the
version, the version it was made from, the point map between them, and the
points that step created:

```rust
/// What a point-creating edit records about itself.
pub struct CreatedPoints {
    /// The edit's content hash (`EditedReconstruction::point_edit_hash`).
    pub hash: String,
    /// The indexes it created in that version, in creation order. Position `k`
    /// in this list is the `k` a point id carries.
    pub indexes: Vec<u32>,
}

impl History {
    /// Push a version whose edit created points, recording them.
    pub fn push_creating(&mut self, value: EditedReconstruction, map: PointMap,
                         label: impl Into<String>, created: CreatedPoints)
        -> VersionSerial;

    /// Where index `index` of version `from` sits in version `to`. Walks back
    /// to the last version the two share, inverting each step's map, then
    /// forward. `Err` carries the version the walk stopped at.
    pub fn follow(&self, from: VersionSerial, to: VersionSerial, index: u32)
        -> Result<u32, VersionSerial>;

    pub fn ancestry(&self, serial: VersionSerial) -> Vec<VersionSerial>;
    pub fn parent_of(&self, serial: VersionSerial) -> Option<VersionSerial>;
    pub fn created_by(&self, serial: VersionSerial) -> Option<&CreatedPoints>;
    pub fn all_serials(&self) -> impl Iterator<Item = VersionSerial> + '_;
}
```

**The step list is never pruned** -- not by the budget, which releases values
and never steps, and not by the truncation a new edit after an undo performs. So
a discarded redo tail is still walkable, and an index taken on it still has a
graph to be followed through.

**Every base has a hash, and so does every point-creating edit.** A base's
content hash is computed from the value without writing a file and equals what a
save of it writes, and a point edit that creates points is hashed over the base's
hash plus the records it adds
([`../core/reconstruction/edited-reconstruction.md`](../core/reconstruction/edited-reconstruction.md)
§ "Hashes"). Demo data and a resection therefore hash exactly like a loaded file.
Those hashes plus this graph are what a point id is minted against and resolved
through ([goto-point.md](goto-point.md) § "The ID forms and the version graph").

## What follows a map

**The selection.** A selected point is an index into the version it was chosen
in. Across an edit it is put through that step's map: a surviving point keeps
whatever index the map gives it, and a point the edit removed clears the
selection. Across an undo it goes through the same map backwards. So deleting a
point clears the selection (the point is gone), deleting a *different* point
leaves it exactly where it was (indexes are stable across a point edit), and
deleting an image moves it to wherever the renumbering put the same point --
which an undo then moves back.

An undo does not restore a selection the edit cleared. The map says where an
index went, not what the user was looking at, and inventing the second from the
first would be a guess.

**The image and camera selections** do not follow a map, because there is not
one for them: a bulk edit renumbers the image table wholesale, so they are
cleared, along with the caches keyed by them
([document-model.md](document-model.md), "What a bulk edit owes the rest of the
viewer").

## The Edit menu and its shortcuts

The menu bar's **Edit** menu, in
[app.rs](../../crates/sfm-explorer/src/app.rs), holds the four actions, each
greyed with a hover text saying why when it does not apply:

| Item | Shortcut | Enabled when |
|------|----------|--------------|
| Undo | `Ctrl/Cmd+Z` | the selected reconstruction has a version to step back to |
| Redo | `Ctrl/Cmd+Y`, also `Ctrl/Cmd+Shift+Z` | it has a version to step forward to |
| Delete Point | `Delete` | a 3D point is selected |
| Delete Image | -- | an image is selected |

Only one redo spelling is written beside the menu item, because a menu that
lists two spellings of one action reads as two actions; both are live.

The shortcuts are gated on egui's own keyboard arbitration, so a text field or a
`DragValue` being typed into keeps `Delete` and `Ctrl+Z` for its own editing.

`Delete Image` is also on the **image row's context menu** in the Scene Graph
panel, beside the two `Resect Image` entries, which is where a specific image is
addressed. It asks for no confirmation: it is an edit with a history behind it,
and undo is the answer to a mis-click. The resections beside it show their answer
as a second node precisely because they are not edits and cannot be undone.

## The Edit History panel

A dock tab, **Edit History**, registered with the panel layout like every other
panel ([panel-layout.md](panel-layout.md)): it is in the Panels menu, in the
stock grid behind the Image Browser and the Action Log, and a layout file spells
it `edit_history`. It shows the **selected** node's history, since that is the
node the Edit menu and its shortcuts act on.

The panel is [edit_history_panel.rs](../../crates/sfm-explorer/src/edit_history_panel.rs),
one function over `&AppState` that reports what was clicked:

```rust
/// What the panel reports back to the dock.
pub(crate) struct EditHistoryResponse {
    /// The version a click asked for, and the node it belongs to.
    pub jump: Option<(ReconId, VersionSerial)>,
}

pub(crate) fn show(ui: &mut egui::Ui, state: &AppState) -> EditHistoryResponse;
```

It keeps no state of its own, so there is nothing of the panel's to lose when it
is closed, and it decides nothing: the dock hands the click to
`AppState::jump_to_version` ([state/edits.rs](../../crates/sfm-explorer/src/state/edits.rs))
and that is what moves the cursor.

**A row per version, oldest first**, carrying the version's label, the time it
was made and its unshared bytes, written for a person to read (`2.00 KiB`,
`165 MiB`). A header above them names the node and counts its versions.

**Two marks.** The row at the cursor is marked with `▶` and drawn as the
selected row: it is what the node is showing. The row at the **disk state** --
the version the node's file on disk holds, which is the version it was loaded at
until a save moves it (`History::disk_serial`, [saving.md](saving.md)) -- is
marked with `●`. A cursor
mark anywhere but the disk mark is the panel's way of saying the node is dirty,
which is a fact about two rows rather than a badge of its own.

**A row whose value the budget released still lists**, says `(released)`, and
refuses the jump with a hover text saying why: the maps are kept for every
version ever minted, so the history still knows what happened there, and only
the value it would return to is gone.

**Clicking a row jumps the cursor to it**, in either direction and in one step
(§ "Jumping to a version"). The cursor's own row is not a jump and is disabled;
its hover says it is what the node shows.

**With no node selected**, or with a node whose history is the one version it
was loaded at, the panel says so rather than showing an empty list.

**Keyboard.** The panel adds no shortcut and holds no text field, so
`Ctrl/Cmd+Z` and `Ctrl/Cmd+Y` keep working with the focus in it: they are read
outside the dock, under egui's keyboard arbitration, and nothing here asks for
the keyboard. An undo or a redo moves the mark the panel draws, since the panel
reads the cursor rather than remembering it.

## Jumping to a version

`AppState::jump_to_version(id, serial)` moves a node's cursor straight to one
version:

```rust
pub fn jump_to_version(&mut self, id: ReconId, serial: VersionSerial) -> Result<(), String>;
```

**It is the run of undos or redos that separates the two versions.** The cursor
is walked one version at a time and the selection is put through each step's map
in turn, so a jump over three edits leaves the selection exactly where three
undos would have, including the case where one of the steps removed the point
and cleared it. Writing it as the composition rather than as a single map from
serial to serial is what keeps one rule for where a selection goes.

**It is refused as a whole** when the destination is not one of that node's
versions, when it is where the cursor already is, or when any version the walk
would pass through, the destination included, has had its value released: the
check runs over the whole span before the cursor moves, so a refusal leaves the
cursor where it was.

What a bulk edit owes the rest of the viewer is owed here too: the walk may pass
a version whose base renumbered the image table, so the image and camera
selections and the caches keyed by them are dropped, exactly as an undo drops
them.

## The Action Log

Every edit, undo and redo writes one `Edit` entry naming the node, what was done
and the step's serials
([action-log.md](action-log.md)):

```
Deleted point 12345 in run_a (v3 → v4)
Deleted image IMG_0007.jpg from run_a (v4 → v5)
Undo: Deleted image IMG_0007.jpg from run_a (v5 → v4)
Redo: Deleted image IMG_0007.jpg from run_a (v4 → v5)
Go to: Opened run_a (v5 → v3)
```

A jump is one entry however many versions it crossed, because it is one thing
the user asked for; its text is the label of the version it arrived at, and its
serials are where it started and where it stopped rather than every version in
between.

An undo's text is the label of the version it left, so the log says what was
undone rather than what is now showing. A refused edit writes a failed entry
carrying its reason.

The log and the history are not the same thing and neither is derived from the
other. The log is text, per session, and complete: it records selections,
display changes and view moves alongside the edits. The history is the replayable
record, per node, and holds values. The log's non-goal "an undo stack" stays true
of the log.

## Implementation notes

**Inverting a row map is core's problem, not this module's.** `RowMap::inverse`
is a binary search over a run rather than a point, for a reason its own spec
states; the history calls it and adds nothing.

**A chain inverts in reverse.** `Chain` applies its steps in order forward and in
reverse order backward, and both fold through `Option`, so a point that any step
removed resolves to nothing rather than to whatever the next step would have made
of a stale index.

## Testing

`crates/sfm-explorer/src/document/tests.rs` covers the cursor and the maps: that
undo and redo move the cursor without moving the versions, that an edit after an
undo truncates the tail and keeps its maps, that serials are not reused, that the
budget releases values and never maps, and that each map case is its own inverse
on the indexes that survive it, over maps built the way the edits build them.
The scan those maps come from is covered in core, in
`crates/sfmtool-core/src/reconstruction/edited/tests.rs`.

`crates/sfm-explorer/src/state/edits/tests.rs` covers what follows: a surviving
selection keeping its index across a point edit, a deleted selection clearing,
a selection following the renumbering of an image deletion onto the same point
and coming back through the undo, and the three log texts. The jump is there
too: a jump back landing where a run of undos would have, a jump forward
returning to the version it came from, the selection arriving at the same index
a run of undos leaves it at across a point edit and a renumbering, one log entry
naming the two serials, and the refusals -- a released version, a version behind
a released one, the cursor's own version, and a serial belonging to another
node -- each leaving the cursor and the log untouched.

`crates/sfm-explorer/src/edit_history_panel/tests.rs` runs the panel through
`Context::run_ui` and reads the strings it painted: the rows in oldest-first
order with the cursor and disk marks on the right ones, the mark following an
undo, a released row listing and saying so, the two empty states, and a
synthesized click on a row reporting that version -- which the test then jumps
to, so what the panel offers and what the jump does are asserted together. The
sizes are checked against the strings a row states them in.

`crates/sfm-explorer/tests/ui_basic.rs` covers what no headless frame can: the
panel drawn in a real window, brought to the front by a layout file naming it
alone, listing the loaded node's one version.

## Non-goals

- A branching history. A new edit after an undo discards the redo tail.
- History across sessions. The versions live as long as the viewer.
- Restoring a selection an edit removed.
- Editing a version from the panel: a row is a place to stand, not a thing to
  rename, delete or export.
