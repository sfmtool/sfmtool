# The bench in the viewer

Beside each loaded reconstruction the viewer keeps a **bench**: a place where
things that are not settled yet are held and worked on. Today the one kind of
thing put on it is an **editable track** -- a track taken out of the
reconstruction, or started from a pixel, with the measurements that judge each
sighting and the person's verdict on each -- and the one step that crosses back
is the commit. The bench is a value beside the reconstruction rather than part
of it: a save does not write it, the point count does not include it, and no
panel that reads the reconstruction sees it.

What makes it the viewer's rather than core's is the **history**. A version of a
node is a pair, the reconstruction value and the bench as it stood, so one Undo
walks both and a person editing a track never has to know which of their steps
touched the file. Everything else about the bench -- the list, the labels, the
activation, and every step over an item -- is `sfmtool_core::bench`
([`../core/bench/bench.md`](../core/bench/bench.md),
[`../core/bench/editable-track.md`](../core/bench/editable-track.md)), a value
and pure functions with no window in them.

Related specs: [`track-edit.md`](track-edit.md) (the panel that edits a track on
it), [`multi-panel-image-browser.md`](multi-panel-image-browser.md) (the Image
Detail panel, which carries the two steps that name a pixel and draws the active
track as its bench layer), [`edits/commit-track.md`](edits/commit-track.md) (the one step that writes
the reconstruction), [`document-model.md`](document-model.md) (the version the
bench is a half of), [`edit-history.md`](edit-history.md) (the cursor that walks
it), [`scene-graph.md`](scene-graph.md) (the tree the Bench group is a child
of), [`background-tasks.md`](background-tasks.md) (where an evaluation runs),
[`action-log.md`](action-log.md) (the row each step writes), and
[`../drafts/sfm-explorer-track-editing.md`](../drafts/sfm-explorer-track-editing.md)
(the proposal this is step 2 of, and where the searches, the remaining overlays
and the wire are still going).

---

## The interface

The bench half of the history is in
[document.rs](../../crates/sfm-explorer/src/document.rs); every step on it is an
`AppState` method in [bench.rs](../../crates/sfm-explorer/src/bench.rs).

```rust
pub struct Version {
    pub serial: VersionSerial,
    pub label: String,
    pub at: jiff::Timestamp,
    /// `None` once the budget has released it.
    pub value: Option<EditedReconstruction>,
    /// The other half of the pair, kept whether or not the value is.
    pub bench: Arc<Bench>,
    /// The version whose document half this one shares.
    pub document_serial: VersionSerial,
    pub unshared_bytes: u64,
}

impl History {
    /// A document edit: the bench at the cursor is carried along.
    pub fn push(&mut self, value: EditedReconstruction, map: PointMap,
                label: impl Into<String>) -> VersionSerial;
    /// A bench step: the document value is carried along, and the map is an
    /// empty `Removed` -- the identity.
    pub fn push_bench(&mut self, bench: Arc<Bench>, label: impl Into<String>)
        -> VersionSerial;
    /// Both halves at once, which only a commit states.
    pub fn push_pair(&mut self, value: Option<EditedReconstruction>,
                     bench: Arc<Bench>, map: PointMap, label: impl Into<String>,
                     created: Option<CreatedPoints>) -> VersionSerial;

    pub fn current_bench(&self) -> &Arc<Bench>;
    /// Whether the **document** half at the cursor is not the one on disk.
    pub fn is_dirty(&self) -> bool;
}

impl AppState {
    pub(crate) fn bench(&self, id: ReconId) -> Option<&Arc<Bench>>;
    pub(crate) fn bench_track(&self, id: ReconId, label: &str)
        -> Option<&Arc<EditableTrack>>;

    pub(crate) fn put_point_on_bench(&mut self, point: PointRef) -> Result<String, String>;
    pub(crate) fn start_bench_cluster(&mut self, image: ImageRef, pixel: [f32; 2],
                                      radius_px: f32) -> Result<String, String>;
    pub(crate) fn add_bench_observation(&mut self, label: &str, image: ImageRef,
                                        pixel: [f32; 2]) -> Result<(), String>;
    pub(crate) fn set_bench_verdict(&mut self, id: ReconId, label: &str,
                                    observation: usize, verdict: Verdict) -> Result<(), String>;
    pub(crate) fn apply_bench_thresholds(&mut self, id: ReconId, label: &str,
                                         thresholds: &Thresholds) -> Result<(), String>;
    pub(crate) fn split_bench_track(&mut self, id: ReconId, label: &str,
                                    observations: &[usize]) -> Result<String, String>;
    pub(crate) fn activate_bench_item(&mut self, id: ReconId, label: &str) -> Result<(), String>;
    pub(crate) fn discard_bench_item(&mut self, id: ReconId, label: &str) -> Result<(), String>;
    pub(crate) fn rename_bench_item(&mut self, id: ReconId, label: &str, to: &str)
        -> Result<(), String>;
    pub(crate) fn commit_bench_track(&mut self, id: ReconId, label: &str) -> Result<(), String>;
    pub(crate) fn start_bench_evaluate(&mut self, id: ReconId, label: &str) -> Result<(), String>;
    pub(crate) fn start_bench_stage(&mut self, id: ReconId, label: &str, stage: StageKind)
        -> Result<(), String>;
}
```

### Why it is shaped this way

**Three pushes rather than one, over one function.** A version always holds both
halves, and `push_pair` is the one that writes it; the other two are what a
caller says when it changed one half and not the other. Spelling a document edit
as "push this value" keeps every existing edit exactly as it was -- none of them
knows the bench exists -- while still producing a version that carries the bench
forward, and spelling a bench step as "push this bench" keeps a hundred verdicts
from having to restate the reconstruction they did not touch.

**The item is named by its label at every call.** A gesture in a bench panel
means "the active item of the kind this panel edits", and the panel is what
knows that; the steps take the label, so the question of which item is answered
in one place rather than inside each step. `active_track_label` is what a caller
resolves it with.

**The two steps that name a pixel are invoked from the Image Detail context
menu.** `start_bench_cluster` and `add_bench_observation` take a pixel, and the
viewer's one way to name a pixel is a right-click in that panel, where the two
point edits that need one already live
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "Image Detail:
the context menu"). Every other step names an item and is a button in the Track
Edit panel.

**Each step returns `Result<_, String>`.** The `String` is the sentence a
refusal shows, which is core's own wording behind a clause naming what was
being done. The caller is a menu entry or a button that has to say in one line
why nothing happened.

**The two steps that read photographs return as soon as the worker is running.**
They are `start_`-prefixed for that reason, and what they answer is whether the
operation could *begin*. The report lands frames or seconds later, through the
background machinery.

### Example

```rust
let label = state.put_point_on_bench(PointRef::new(id, 1207))?;   // one version
state.set_bench_verdict(id, &label, 3, Verdict::Out)?;            // one version
state.start_bench_evaluate(id, &label)?;                          // a task
// ... the report lands, which is one more version ...
state.commit_bench_track(id, &label)?;                            // one version, both halves
```

---

## One history for the pair

A version of the node is the reconstruction value, base plus overlay, **and**
the bench as it stood. The consequences are the ones the value model already
pays for, applied to one more field.

- **Undo and redo walk the pair.** The Edit menu's Undo, its shortcuts and the
  Edit History panel's jump move one cursor over one list, and whichever half a
  step changed comes back. A verdict, an evaluation, a stage change, a commit
  and then a deletion of some other point are five versions in one order, and
  undo retraces them in that order.
- **Truncation is one rule.** A new step after an undo discards the redo tail,
  whichever half the discarded versions had changed.
- **Dirty is about the document half.** A version is dirty when its document
  half is not the one on disk, which is what `document_serial` says: a version
  that changed the document carries its own serial there, and a bench step
  carries its parent's. So a run of bench steps over a clean value is clean --
  a save of any of them would write the same bytes -- and the `*` marker, the
  window title and the close prompt say so. The Edit History panel marks the
  version on disk as it does now, and a bench step above it is a row that does
  not move the mark. A disk version a truncation took with it leaves the node
  dirty, because what the file holds is then no version of this history.
- **The budget counts the bench.** A version's unshared bytes are the unshared
  half of its value plus the items its predecessor's bench does not share, each
  charged its observations and its consensus bitmap. Every item a step did not
  touch is the same `Arc` in both benches and costs nothing, so a step on one
  track costs that track. A bench is small against a bulk edit, so the budget's
  arithmetic does not change, only what it sums. A released version keeps its
  bench as it keeps its label: a bench is a few tracks, and the budget is about
  the reconstruction.
- **The maps are trivial for a bench step.** Point indexes are untouched, so the
  step's `PointMap` is an empty `Removed`, the selection stays where it is, and
  an id copied before the step resolves after it.

What this buys over a history of the bench's own is that there is one Undo. Two
stacks would leave the person guessing which the shortcut would act on, and a
commit, which changes both, would have to appear in both or in neither.

**Closing a node drops its bench with its history.** An item names images of one
node and is meaningless without it.

---

## What a step writes

Every bench step is **one version**, with one Action Log row of kind `Bench`,
and an Edit History row that does not move the on-disk mark. The commit is the
exception in one respect only: its row is of kind `Edit`, because it is one
([`edits/commit-track.md`](edits/commit-track.md)).

| Step | Version label |
|---|---|
| Put a point on the bench | `Put point 1207 on the bench as pt3d_a1b2c3d4_1207` |
| Start a cluster from a pixel | `Started IMG_0042@142,198 on the bench` |
| Add an observation | `Added image_012.jpg to pt3d_a1b2c3d4_1207` |
| A verdict | `Turned image_012.jpg out of pt3d_a1b2c3d4_1207` |
| Apply the thresholds | `Applied the thresholds to IMG_0042@142,198: 3 in, 1 out, 1 pinned, 0 unmeasured` |
| Evaluate | `Evaluated IMG_0042@142,198` |
| Set the stage | `Set IMG_0042@142,198 to the track stage` |
| Split | `Split 2 observations off pt3d_a1b2c3d4_1207 as pt3d_a1b2c3d4_1207-split` |
| Activate | `Made IMG_0042@142,198 the active track` |
| Discard | `Discarded IMG_0042@142,198 from the bench` |
| Rename | `Renamed IMG_0042@142,198 to bull-nose on the bench` |

The Action Log row is that sentence plus the version serials, exactly as an
edit's is: `Turned image_012.jpg out of pt3d_a1b2c3d4_1207 (v7 → v8)`. A
refusal is one failed row carrying the refusal's sentence.

**A step that changes nothing pushes no version.** Setting the verdict an
observation already has, setting the stage a track is already at, activating the
active item and renaming an item to the label it holds each report that nothing
happened and leave the history alone: a row that has to be undone for nothing is
worse than no row.

### Labels

A label is minted from what the item was made from and is how the item is named
everywhere -- in the Scene tree, on the panel's tabs, in every row and version
label. The minting is core's ([`../core/bench/bench.md`](../core/bench/bench.md)
§ "Labels") with one exception the viewer supplies: **a track put on the bench
from a point is labelled by that point's portable id**, `pt3d_a1b2c3d4_1207`,
because that id names the content the point is a row of and the version graph
that content sits in, and core has neither ([`goto-point.md`](goto-point.md)).

**Putting a point on the bench twice activates the track it already made.** The
person asked to work on that point, and there it is; a second item for one point
would be two answers to one question. The test is the origin, followed to the
cursor.

---

## The two steps that read photographs

An evaluation and a stage change run as **background tasks**
([`background-tasks.md`](background-tasks.md)), under `Evaluate track` and `Set
track stage`. Neither is cancellable: the patch kernels they run take the
`Progress` for their phases and never ask whether they should stop, and the
declaration is held to that by the background tests.

**The photographs are decoded on the GUI thread, through the node's own
full-resolution cache**, before the worker starts -- the same cache
`add_observation` decodes through, so an image a panel has already shown is not
read twice and nothing in the viewer holds a second copy of it. What crosses to
the worker is those decoded pyramids, a clone of the value at the cursor and a
clone of the track, so the worker holds no reference into the scene. The images
decoded are the ones the track's observations name; every other entry of the
view slice is a one-pixel placeholder, which no kernel samples.

**A report lands on the observations it measured.** Observations are appended
and never renumbered and a measurement is keyed by observation index, so a
report computed against the track as it stood when the task began still applies
when it finishes. What it cannot survive is the item leaving the bench at the
cursor: then it is discarded with one Action Log row saying so, and no version.
That row is a guard rather than an everyday outcome -- the node is locked for
the duration of its own task, so nothing can take the item off in the meantime
-- and it is what keeps a report from inventing an item to land on.

---

## The Bench group in the Scene tree

Each node gains a **Bench** child beside its Camera Intrinsics, Camera Images
and Points groups ([`scene-graph.md`](scene-graph.md)), drawn only when
something is on it. The header is `Bench (2)`; inside it is one row per item, by
label, with its `in` count, the active one marked as a selected row. There is no
eye: nothing on the bench is drawn from this row, and an item is not part of the
reconstruction.

Clicking a row makes that item the active one of its kind, which is a step like
any other; a secondary click offers *Discard*. The bench is in the tree because
the tree is where a node's parts are listed, and it is per node because an item
names that node's images and poses.

The row reports the item by its **position** on the bench rather than by its
label, because `SceneGraphResponse` is a `Copy` value and a label is a `String`;
the dock reads the label off the bench at that position before calling the step.

---

## Testing

[bench/tests.rs](../../crates/sfm-explorer/src/bench/tests.rs), headless, over
the demo reconstruction rewritten as `embedded_patches` with every keypoint its
point's exact projection and a photograph cached for every image:

- putting a second item on, activating it and discarding it are three versions,
  an undo of the discard puts it back with the activation as it stood, and the
  first item is the same `Arc` throughout;
- putting a point on the bench twice activates the track it already made;
- the two gestures the Image Detail context menu carries -- a cluster started at
  a pixel, then a candidate added at one in the same image -- are one version and
  one `Bench` row each, the second sighting joining as a candidate, and an undo
  walks them back one at a time;
- a verdict, a stage change and an evaluation are three versions, undo retraces
  them in order and redo replays them;
- a document edit between two bench steps is a version in its place, and undoing
  it leaves the bench alone;
- a commit replaces its origin point, the selection follows it, and an undo
  restores the pair;
- a commit's row is an `Edit` and every other bench step's is a `Bench`, one row
  per step;
- a step that changes nothing pushes no version;
- a run of bench steps over a clean value is clean, a commit is dirty, and
  undoing the commit is clean again;
- a report lands on the item it measured, and one for an item that is not there
  is discarded with one row and no version.

The steps themselves are core's and are tested there, over a synthetic textured
plane whose numbers are known to the pixel.

---

## Non-goals

- **A second kind of item.** The bench is shaped so one can be added; the
  editable track is the only kind there is.
- **Persisting the bench.** A save writes the document half; a commit is how
  bench work reaches the file.
- **A bench across nodes.** An item names one node's images.
- **The searches that propose observations.** The descriptor search, the view
  sweep and the pull-in are proposed in
  [`../drafts/sfm-explorer-track-editing.md`](../drafts/sfm-explorer-track-editing.md).
- **Drawing the bench in the 3D viewer and the Image Browser.** The preview
  buffer and the thumbnail borders are proposed in the same draft. The Image
  Detail panel does draw the active track, as its bench layer
  ([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "The bench
  layer").
- **The wire.** The MCP tools for the bench are proposed there too.
