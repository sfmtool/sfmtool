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
of), [`background-tasks.md`](background-tasks.md) (where a reading and a fit
run),
[`action-log.md`](action-log.md) (the row each step writes),
[`mcp-server.md`](mcp-server.md) (the surface § "The wire" is a family of), and
[`../drafts/sfm-explorer-track-editing.md`](../drafts/sfm-explorer-track-editing.md)
(the proposal this is step 2 of, and where the searches and the remaining
overlays are still going).

---

## The interface

The bench half of the history is in
[document.rs](../../crates/sfm-explorer/src/document.rs); every step on it is an
`AppState` method in [bench.rs](../../crates/sfm-explorer/src/bench.rs), and the
descriptor index a search queries is in
[descriptor_index.rs](../../crates/sfm-explorer/src/descriptor_index.rs).

```rust
/// Where a seed's position and shape come from.
pub(crate) enum Seed {
    /// A pixel, with the patch's half-width in that image's own pixels where
    /// the caller named one.
    Pixel { pixel: [f64; 2], radius_px: Option<f64> },
    /// A pixel with the keypoint-frame affine shape read at it.
    Affine { pixel: [f64; 2], shape: [[f64; 2]; 2] },
    /// A `.sift` feature, which carries its own position and shape.
    Feature { feature: u32 },
}

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
    pub(crate) fn start_bench_cluster(&mut self, image: ImageRef, seed: &Seed)
        -> Result<String, String>;
    pub(crate) fn add_bench_observation(&mut self, label: &str, image: ImageRef,
                                        seed: &Seed) -> Result<(), String>;
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
    /// The point the commit wrote: the index it took, and the index it
    /// replaced where it replaced one.
    pub(crate) fn commit_bench_track(&mut self, id: ReconId, label: &str)
        -> Result<Committed, String>;
    /// Read the track where it sits, moving nothing. `search_px` is how far
    /// around each observation the correlation peak is looked for.
    pub(crate) fn start_bench_evaluate(&mut self, id: ReconId, label: &str,
                                       search_px: Option<f64>) -> Result<(), String>;
    /// Move it: localize, re-triangulate, re-fuse, then read the result back.
    pub(crate) fn start_bench_fit(&mut self, id: ReconId, label: &str,
                                  search_px: Option<f64>) -> Result<(), String>;
    pub(crate) fn start_bench_stage(&mut self, id: ReconId, label: &str, stage: StageKind)
        -> Result<(), String>;
    /// Ask the node's descriptor index which other photographs hold the patch
    /// around one observation, and add each as a candidate.
    pub(crate) fn start_bench_descriptor_search(
        &mut self, id: ReconId, label: &str, observation: usize,
        radius_px: Option<f64>, min_inliers: Option<usize>) -> Result<(), String>;
    /// Why that search cannot run, or `None`: what greys the row's menu entry.
    pub(crate) fn bench_search_refusal(&self, id: ReconId, label: &str,
                                       observation: usize) -> Option<String>;
}

// The index the search queries, in
// [descriptor_index.rs](../../crates/sfm-explorer/src/descriptor_index.rs).
impl AppState {
    pub(crate) fn descriptor_index(&self, id: ReconId) -> Option<&DescriptorIndex>;
    pub(crate) fn default_descriptor_index_path(&self, id: ReconId) -> Option<PathBuf>;
    /// Open the default index if the file is there and none is open, and
    /// remember the look either way.
    pub(crate) fn open_default_descriptor_index(&mut self, id: ReconId);
    pub(crate) fn open_descriptor_index(&mut self, id: ReconId, path: Option<PathBuf>)
        -> Result<PathBuf, String>;
    pub(crate) fn build_descriptor_index_refusal(&self, id: ReconId) -> Option<String>;
    pub(crate) fn start_build_descriptor_index(&mut self, id: ReconId, path: Option<PathBuf>)
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

**One `Seed` for the two steps that place an observation.** A caller arrives
holding one of three things -- a pixel, a pixel with a size or a shape read at
it, or a `.sift` feature, which carries both -- and the step should not care
which. `Seed::Pixel` with no radius means "I have no shape to give you, use the
one you have": the radius the Create 3D Point prompt would offer when a cluster
is being started, and the track's reference shape when an observation is being
added to one, which is what a right-click means. A `Seed::Feature` is read
through `AppState::sift_cache`, the cache the Image Detail overlay draws its
ellipses from, so a feature seeded here is the mark the person is looking at;
its stored affine is already the cluster stage's own convention and is passed on
as it stands.

**The two steps that name a pixel are invoked from the Image Detail context
menu.** `start_bench_cluster` and `add_bench_observation` take a seed, and the
viewer's one way to name a pixel is a right-click in that panel, where the two
point edits that need one already live
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "Image Detail:
the context menu"). Every other step names an item and is a button in the Track
Edit panel.

**Each step returns `Result<_, String>`.** The `String` is the sentence a
refusal shows, which is core's own wording behind a clause naming what was
being done. The caller is a menu entry or a button that has to say in one line
why nothing happened.

**The commit hands back the point it wrote.** One row of the reconstruction is
the whole of what it produces, and both callers have to name it: the panel
selects it, and the wire reports its index and its portable id. It comes back
from the step rather than being looked up afterwards, because "the point this
commit wrote" is not a question the value can be asked once the version has
landed -- a replacement takes the index it replaced, and a creation takes
whatever index the overlay had free.

**The three steps that read photographs return as soon as the worker is running.**
They are `start_`-prefixed for that reason, and what they answer is whether the
operation could *begin*. The report lands frames or seconds later, through the
background machinery. What they refuse in the call is everything the track alone
decides (§ "The three steps that read photographs").

### Example

```rust
let label = state.put_point_on_bench(PointRef::new(id, 1207))?;   // one version
state.set_bench_verdict(id, &label, 3, Verdict::Out)?;            // one version
state.start_bench_evaluate(id, &label, None)?;                    // a task: measures, moves nothing
state.start_bench_fit(id, &label, None)?;                         // a task: moves it, then measures
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
| Evaluate | `Evaluated IMG_0042@142,198: measured 4 of 5 observations at (x, y, z)` |
| Fit | `Fitted IMG_0042@142,198: placed 4, measured 4 of 5 observations at (x, y, z)` |
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

## The three steps that read photographs

A reading, a fit and a stage change run as **background tasks**
([`background-tasks.md`](background-tasks.md)), under `Evaluate track`, `Fit
track` and `Set track stage`, beside `Search descriptors` and `Build descriptor
index`, which read a `.kdf` and a capture's `.sift` files rather than
photographs ([`track-edit.md`](track-edit.md) § "The descriptor index"). None of
them is cancellable: the patch kernels
they run take the `Progress` for their phases and never ask whether they should
stop, and the declaration is held to that by the background tests.

**What the track alone decides is decided before the task starts.** Core
publishes the half of each step's own validation that reads no photograph --
`bench::evaluate_preconditions`, `bench::fit_preconditions` and
`bench::set_stage_preconditions`
([`../core/bench/editable-track.md`](../core/bench/editable-track.md)) -- and
`start_bench_evaluate`, `start_bench_fit` and `start_bench_stage` ask it before
they build a job. So a track being fitted with fewer than two `in` observations
(which a *reading* of the same track permits), or one being taken down to the
cluster stage with no frame or no position, is a refusal of the **gesture**: a
sentence in the caller's own hand, no task, no version. The step itself calls
the same function first, so the two answers cannot drift. What is left for the
task is everything that needs the pixels, which is the rest.

**The photographs are decoded on the worker**, along with the kernel work that
reads them: the file reads and the pyramid builds are seconds of work in their
own right, and a step that did them on the GUI thread would freeze the frame --
and hold the wire's reply window shut -- for all of it before the task it defers
to had begun. So what the gesture does here is clone a handful of
handles: `ViewSources` carries the node's cameras and poses, a **shared** clone
of each photograph the node's own full-resolution cache already holds, and a
path for each one it does not, and `ViewSources::decode` turns that into the
pyramids on the worker, under a `decode images` phase. A photograph the viewer
had is therefore not decoded a second time and not copied; one the worker reads
itself is dropped with the task, because the cache is the GUI thread's and the
panels fill it for what they draw. A photograph that cannot be read is the
worker's refusal, arriving as the task's failed row rather than as a refusal of
the gesture -- which is honest: whether a file is readable is not a question the
gesture can answer without doing the read. Beside those views the worker gets a
clone of the value at the cursor and a clone of the track, so it holds no
reference into the scene. The images it decodes are the ones the track's
observations name; every other entry of the view slice is a one-pixel
placeholder, which no kernel samples.

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

## The wire

An agent gets the same bench a human does, through eighteen MCP tools
([mcp-server.md](mcp-server.md) § "The bench family"), in
[mcp/bench.rs](../../crates/sfm-explorer/src/mcp/bench.rs). **Each one is one of
the `AppState` methods above**, which is the whole of what makes an agent's
verdict, split or commit a version in the history the human is looking at.

Every tool takes `reconstruction_label`. The item tools take `item`; the track
tools take `track`, and **a call that names no track acts on the active one**,
resolved with `active_track_label`, which is what a gesture in the Track Edit
panel means when it names no item.

```jsonc
// The two creates are named by the stage they make, and where the first
// observation comes from is a parameter.
// create_bench_cluster { "reconstruction_label": "bull", "camera_image": 4,
//                        "pixel": [142.0, 197.5], "radius_px": 7.5 }
// create_bench_cluster { "reconstruction_label": "bull", "camera_image": 4,
//                        "pixel": [142.0, 197.5], "affine": [[7.1, -0.4], [0.4, 7.1]] }
// create_bench_cluster { "reconstruction_label": "bull", "camera_image": 4, "feature": 847 }
// create_bench_track   { "reconstruction_label": "bull", "point": 1207 }
//
// The list.
// get_bench            { "reconstruction_label": "bull" }
// activate_bench_item  { "reconstruction_label": "bull", "item": "IMG_0042@142,198" }
// rename_bench_item    { "reconstruction_label": "bull", "item": "IMG_0042@142,198",
//                        "label": "bull-nose" }
// discard_bench_item   { "reconstruction_label": "bull", "item": "bull-nose" }
//
// One track on it. "track" omitted means the active track.
// get_bench_track              { "reconstruction_label": "bull" }
// add_bench_track_observation  { "reconstruction_label": "bull", "track": "bull-nose",
//                                "camera_image": 7, "pixel": [88.5, 210.0] }
// set_bench_track_verdict      { "reconstruction_label": "bull", "observation": 3,
//                                "verdict": "in" }
// apply_bench_track_thresholds { "reconstruction_label": "bull", "min_zncc": 0.8 }
// evaluate_bench_track         { "reconstruction_label": "bull" }
// fit_bench_track           { "reconstruction_label": "bull" }
// set_bench_track_stage        { "reconstruction_label": "bull", "stage": "track" }
// split_bench_track            { "reconstruction_label": "bull", "observations": [3, 5, 8] }
// commit_bench_track           { "reconstruction_label": "bull" }
//
// The descriptor index, which is the node's rather than any track's, and the
// search through it, which is one observation's.
// open_descriptor_index           { "reconstruction_label": "bull" }
// build_descriptor_index          { "reconstruction_label": "bull" }
// search_bench_track_descriptors  { "reconstruction_label": "bull", "observation": 0 }
```

**The two reads have no panel gesture behind them**, because a panel shows what
they answer. `get_bench` is the Bench group as JSON: each item's label, kind,
stage, origin and counts, and the active label per kind. `get_bench_track` is
the Track Edit table: the stage and its data, the origin, the thresholds, and
every observation with its provenance, verdict, `pixel` and both stages'
measurements where they exist -- at the track stage, the two distances
(`seed_shift_px` and `projection_offset_px`) and, for a row the reading could
not score, the `reason` sentence in place of a ZNCC. **An observation is
addressed by its position in
that list**, which is stable for the life of the track, so an index an agent is holding after
a verdict or an evaluation still names the same observation. The template's
samples and the consensus bitmap are reported as present or absent rather than
sent: they are pictures, and that surface is not a data channel.

**`pixel` is where the observation sits, whatever said so**: the keypoint a
reading wrote, else the refined cluster position, else the seed the step that
proposed it left. It is a field of its own rather than something a caller
assembles out of the two measurement blocks, because that one answer is what
every reader of a sighting wants and a candidate a descriptor search has just
added has no keypoint at all -- an agent would otherwise have to know which slot
to fall back to before it could look at one. It is `crate::bench::observation_site`'s
rule, so the number an agent reads here is the pixel the Image Detail panel
marks, the place the Track Edit row click reveals, the centre of the tile that
row draws and what `set_image_detail_view`'s `bench_observation` aims. `null`
only for an observation nothing says the place of, which is the state core's
`Unmeasured::NoSeed` names.

**Every step answers as an edit answers**, with the version it pushed and the
sentence the Action Log recorded, plus the `item` it acted on. A create and a
split name what they made, a rename names the label the item now holds, and an
added observation names the index it took. **A commit names the point it
wrote** -- `{ "point": { "index": 4211, "id": "pt3d_95fe75db_0", "replaced":
1207 } }` -- because one row of the reconstruction is the whole of what a commit
produces, and neither index is derivable from the sentence: a commit that
replaces takes the index it replaced, and one that creates takes whatever index
the overlay had free. So the next call is a `get_point` rather than a search
through the counts for whichever row is new. So `undo`, `redo` and
`jump_to_version` need no bench variant: the history they walk already holds the
bench steps.

**The two index tools push no version.** An index is a file beside the
workspace and a handle on it, so `open_descriptor_index` and
`build_descriptor_index` change neither the reconstruction nor the bench, and
their reply is the index -- its path, whether it is open, and its descriptor
count -- rather than a version. `get_bench` reports the same shape under
`descriptor_index`, carrying the **default** path even when nothing is open, so
an agent can see where a build would put one. The search itself is an ordinary
bench step and answers as one.

**The steps that read a file answer in two levels**, as the bundle
adjustment does: with the version they pushed when they finish inside the reply
window, and with `running: true` and an `operation_id` to poll
`get_background_task` with when they do not. The deferral is taken before a
single photograph has been read (§ "The three steps that read photographs"), so
the window is measured against the operation rather than spent on the decode in
front of it. A step that finds nothing to do starts no task and answers with the
version the node stands at, and a step the **track** rules out starts no task
either: it is a tool error in the step's own sentence, arriving in the call
rather than through a task the agent would have had to poll to learn that
nothing was ever going to happen.

**A refusal is the step's own sentence and pushes nothing.** The wire wraps
nothing: what an agent reads is the sentence the panel's status line would show.

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
- a verdict, a stage change and a reading are three versions, undo retraces
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
  is discarded with one row and no version;
- a photometric step whose photographs are neither cached nor readable **starts
  its task all the same**, and the refusal comes home through it, which is what
  says the decode is the worker's;
- a photometric step the **track** rules out starts no task, pushes no version
  and writes one failed row in its own sentence, which is what says a refusal
  costs no decode;
- a descriptor search with no index open starts no task and writes one failed
  row naming the row that would give it one, and an index build on a node with
  no `.sift` files is refused the same way.

The steps themselves are core's and are tested there, over a synthetic textured
plane whose numbers are known to the pixel.

The wire is tested in
[mcp/tests.rs](../../crates/sfm-explorer/src/mcp/tests.rs), over the same
fixture with a label on the node, for what the boundary owes: each tool being
the `AppState` call the panel makes, an observation index surviving the steps
that follow it, a refusal arriving as the step's own sentence, and the two
photometric steps deferring to a worker and landing their version
([mcp-server.md](mcp-server.md) § "Testing").

---

## Non-goals

- **A second kind of item.** The bench is shaped so one can be added; the
  editable track is the only kind there is.
- **Persisting the bench.** A save writes the document half; a commit is how
  bench work reaches the file.
- **A bench across nodes.** An item names one node's images.
- **The remaining searches that propose observations.** The view
  sweep and the pull-in are proposed in
  [`../drafts/sfm-explorer-track-editing.md`](../drafts/sfm-explorer-track-editing.md).
- **Drawing the bench in the 3D viewer and the Image Browser.** The preview
  buffer and the thumbnail borders are proposed in the same draft. The Image
  Detail panel does draw the active track, as its bench layer
  ([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "The bench
  layer").
- **Wire tools for the searches.** The three tools that would drive a descriptor
  search, a view sweep and a pull-in wait on the core steps behind them, and are
  proposed in the same draft. The fourteen tools for the steps that exist are
  § "The wire".
