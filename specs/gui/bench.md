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
touched the file. Everything else about the bench -- the list, the labels and
every step over an item -- is `sfmtool_core::bench`
([`../core/bench/bench.md`](../core/bench/bench.md),
[`../core/bench/editable-track.md`](../core/bench/editable-track.md)), a value
and pure functions with no window in them.

Related specs: [`track-view.md`](track-view.md) (the panel that edits the focused
item with its *Edit* box ticked), [`multi-panel-image-browser.md`](multi-panel-image-browser.md) (the Image
Detail panel, which carries the three steps that name a pixel and draws the focused
item as its bench layer), [`edits/commit-track.md`](edits/commit-track.md) (the one step that writes
the reconstruction), [`document-model.md`](document-model.md) (the version the
bench is a half of), [`edit-history.md`](edit-history.md) (the cursor that walks
it), [`scene-graph.md`](scene-graph.md) (the tree the two Bench groups are
children of), [`background-tasks.md`](background-tasks.md) (where a fit and a
stage change run),
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
SIFT index a search queries is in
[sift_index.rs](../../crates/sfm-explorer/src/sift_index.rs), one of the node's
two index files ([index_files.rs](../../crates/sfm-explorer/src/index_files.rs)).

```rust
/// One hand edit of a track's geometry, in the form the core steps take.
///
/// Each word is said once along the path: the type already says this is an edit
/// of a patch, so the variants are the verbs alone.
pub(crate) enum PatchEdit {
    /// Slide the track-stage patch until its centre sits under this pixel of
    /// that observation's image. Every sighting follows. The track stage's dot
    /// while Track View's Lock is ticked.
    TranslateToPixel { observation: usize, pixel: [f64; 2] },
    /// Move the track-stage patch by this displacement on its own orthonormal
    /// axes `[u, v, n]`, in world units. Every sighting follows.
    Translate { by: [f64; 3] },
    /// Resize the track-stage patch to this world half-length, holding the far
    /// edge when `moved_edge` names one and the centre when it does not.
    Resize { half_length: f64, moved_edge: Option<Edge> },
    /// Put one edge of the outline drawn at this observation under this pixel,
    /// with the opposite edge left where it is. The one size gesture that spans
    /// both stages.
    ResizeToPixel { observation: usize, edge: Edge, pixel: [f64; 2] },
    /// Turn the track-stage patch about its normal.
    Spin { angle_rad: f64 },
    /// Turn one cluster-stage sighting's shape in its own image's pixels.
    SpinShape { observation: usize, angle_rad: f64 },
    /// Turn the track-stage patch about its centre until it faces this outward
    /// normal, by the least rotation.
    Tilt { normal: [f64; 3] },
    /// Put one observation's own sighting at this pixel, and leave every other
    /// where it is. The cluster stage's dot, and the track stage's with Track
    /// View's Lock cleared.
    Sight { observation: usize, pixel: [f64; 2] },
    /// Give one cluster-stage sighting this affine shape outright.
    Shape { observation: usize, shape: [[f64; 2]; 2] },
}

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
    /// As many halves as the step changed: the value and the bench for a
    /// commit, and the display transform as well for a bake. `None` carries
    /// the one at the cursor.
    pub fn push_pair(&mut self, value: Option<EditedReconstruction>,
                     bench: Arc<Bench>, transform: Option<Se3Transform>,
                     map: PointMap, label: impl Into<String>,
                     created: Option<CreatedPoints>) -> VersionSerial;

    pub fn current_bench(&self) -> &Arc<Bench>;
    /// Whether the **document** half at the cursor is not the one on disk.
    pub fn is_dirty(&self) -> bool;
}

impl AppState {
    pub(crate) fn bench(&self, id: ReconId) -> Option<&Arc<Bench>>;
    pub(crate) fn bench_track(&self, id: ReconId, label: &str)
        -> Option<&Arc<EditableTrack>>;

    /// `label` names the item, before the collision suffix; `None` mints one.
    /// The gestures pass `None`, and the wire passes the call's `label`.
    pub(crate) fn put_point_on_bench(&mut self, point: PointRef, label: Option<&str>)
        -> Result<String, String>;
    /// The pixel is brought inside the photograph before anything is seeded, and
    /// the answer says where the observation went and what was asked for.
    pub(crate) fn start_bench_cluster(&mut self, image: ImageRef, seed: &Seed,
                                      label: Option<&str>) -> Result<Seeded, String>;
    pub(crate) fn add_bench_observation(&mut self, label: &str, image: ImageRef,
                                        seed: &Seed) -> Result<Seeded, String>;

    /// What a seeding step made, and whether its pixel had to be clamped.
    pub(crate) struct Seeded {
        pub(crate) label: String,
        pub(crate) pixel: [f64; 2],
        pub(crate) clamped_from: Option<[f64; 2]>,
    }
    pub(crate) fn set_bench_verdict(&mut self, id: ReconId, label: &str,
                                    observation: usize, verdict: Verdict) -> Result<(), String>;
    /// One hand edit of the track's geometry: the patch slid, one edge of it
    /// put under a pixel, a turn, or one sighting placed. The one call behind
    /// every handle of the Image Detail panel's bench layer and behind the
    /// wire's eight patch tools. An edit that changed nothing pushes no version
    /// and writes the row that says so.
    pub(crate) fn edit_bench_patch(&mut self, id: ReconId, label: &str,
                                   edit: &PatchEdit) -> Result<PatchEdited, String>;

    /// What one patch edit did, as much of it as a reply needs.
    pub(crate) struct PatchEdited {
        pub(crate) changed: bool,
        pub(crate) pixel: Option<[f64; 2]>,
        pub(crate) clamped_from: Option<[f64; 2]>,
    }
    /// Set the track's bars and paint their verdicts, as one version: Track
    /// View's threshold box released, or `apply_bench_track_thresholds`.
    pub(crate) fn apply_bench_thresholds(&mut self, id: ReconId, label: &str,
                                         thresholds: &Thresholds) -> Result<(), String>;
    /// Track View's *Accept walk*: put a sighting the last fit kept at its seed
    /// at its `walked_to`, through `sight_observation`. One version.
    pub(crate) fn accept_bench_walk(&mut self, id: ReconId, label: &str,
                                    observation: usize) -> Result<(), String>;
    pub(crate) fn split_bench_track(&mut self, id: ReconId, label: &str,
                                    observations: &[usize]) -> Result<String, String>;
    /// The Edit box ticked (the selected point put on, or its item focused)
    /// or cleared (`unfocus_bench_item`).
    pub(crate) fn set_editing(&mut self, id: ReconId, on: bool) -> Result<(), String>;
    /// A Scene tree double-click on a Bench row: select the node, focus the
    /// item unless it is focused already, raise Track View.
    pub(crate) fn edit_bench_item_at(&mut self, id: ReconId, position: usize);
    /// Image Detail's *Start cluster on the bench here*: the cluster put on,
    /// focused, and Track View raised.
    pub(crate) fn start_cluster_here(&mut self, image: ImageRef, pixel: [f32; 2]);
    /// Image Detail's *Create Track Here*: a track built at the pixel on a
    /// worker, then put on the bench, focused, and committed. A refusal in
    /// front of the worker is one failed row and no task.
    pub(crate) fn start_create_track_at_pixel(&mut self, image: ImageRef,
                                              pixel: [f64; 2], label: Option<&str>)
        -> Result<(), String>;
    /// Why that cannot run on this image, or `None`: what greys the entry.
    pub(crate) fn create_track_here_refusal(&self, image: ImageRef) -> Option<String>;
    /// Image Detail's *Find Nearby Tracks*: the tracks near the pixel found on
    /// a worker, then every usable one put on the bench and, with `commit`,
    /// the new ones committed, as one version. `label` replaces the group
    /// label. A refusal in front of the worker is one failed row and no task.
    pub(crate) fn start_find_nearby_tracks(&mut self, image: ImageRef, pixel: [f64; 2],
                                           commit: bool, label: Option<&str>)
        -> Result<(), String>;
    /// Why that cannot run on this image, or `None`: with `commit`, Create
    /// Track Here's reasons word for word; without it, all but the one about
    /// `.sift` features.
    pub(crate) fn find_nearby_tracks_refusal(&self, image: ImageRef, commit: bool)
        -> Option<String>;
    /// Discarding the focused item unfocuses it.
    pub(crate) fn discard_bench_item(&mut self, id: ReconId, label: &str) -> Result<(), String>;
    /// A copy of the item beside it, focused, with no origin: the label it took.
    pub(crate) fn duplicate_bench_item(&mut self, id: ReconId, label: &str)
        -> Result<String, String>;
    pub(crate) fn rename_bench_item(&mut self, id: ReconId, label: &str, to: &str)
        -> Result<(), String>;
    /// The point the commit wrote, which it also selects: the index it took,
    /// the index it replaced where it replaced one, and whether it wrote at
    /// all -- a point that already holds the track is not written again.
    pub(crate) fn commit_bench_track(&mut self, id: ReconId, label: &str)
        -> Result<Committed, String>;
    /// Move it: localize, re-triangulate, re-fuse, then read the result back.
    /// The reading looks for each peak within the track's `max_shift_px`.
    pub(crate) fn start_bench_fit(&mut self, id: ReconId, label: &str) -> Result<(), String>;
    pub(crate) fn start_bench_stage(&mut self, id: ReconId, label: &str, stage: StageKind)
        -> Result<(), String>;
    /// Ask the node's SIFT index which other photographs hold the patch
    /// around one observation, and add each, unpinned and `out`.
    pub(crate) fn start_bench_descriptor_search(
        &mut self, id: ReconId, label: &str, observation: usize,
        radius_px: Option<f64>, min_inliers: Option<usize>) -> Result<(), String>;
    /// Why that search cannot run, or `None`: what greys the row's menu entry.
    pub(crate) fn bench_search_refusal(&self, id: ReconId, label: &str,
                                       observation: usize) -> Option<String>;
}

// The live evaluation, in
// [bench/live.rs](../../crates/sfm-explorer/src/bench/live.rs)
// (§ "Live evaluation").
impl AppState {
    /// Where the evaluation of one track stands: `Current`, `Evaluating`,
    /// `Refused(why)` or `Failed(why)`. `None` when there is no such track.
    pub(crate) fn bench_evaluation(&self, id: ReconId, item: &str) -> Option<Evaluation>;
    /// Whether an evaluation of that track's current inputs is on a worker now.
    pub(crate) fn bench_evaluation_running(&self, id: ReconId, item: &str) -> bool;
    /// Once per frame: land a finished evaluation, cancel one whose inputs
    /// have moved on, start the next. True when the next frame has to draw.
    pub(crate) fn drive_bench_evaluation(&mut self) -> bool;
}

// The focused item, held in `AppState::focused_item`, in
// [bench.rs](../../crates/sfm-explorer/src/bench.rs) (§ "The focused item").
pub(crate) struct FocusedItem {
    pub(crate) node: ReconId,
    pub(crate) item: ItemId,
}

impl AppState {
    pub(crate) fn focused_item(&self) -> Option<&FocusedItem>;
    /// The focused item's label on `id`'s bench at the cursor; `None` when
    /// nothing is focused, it is on another node, or it is not on that bench.
    pub(crate) fn focused_item_label(&self, id: ReconId) -> Option<&str>;
    /// No version; one `Selection` row. Selects the item's node and its
    /// origin, or clears the point selection. Refused when nothing on the
    /// bench is called `label`, and never for a busy node.
    pub(crate) fn focus_bench_item(&mut self, id: ReconId, label: &str) -> Result<(), String>;
    /// No version; one `Selection` row, or a no-effect row with nothing
    /// focused. Selects the item's origin, or clears the point selection.
    pub(crate) fn unfocus_bench_item(&mut self);
    /// The focused item's origin followed to the cursor.
    pub(crate) fn focused_origin(&self) -> Option<PointRef>;
    /// The first entry of `AppState::recent_items` on its node's bench at
    /// the cursor, as its node and label.
    pub(crate) fn most_recent_item(&self) -> Option<(ReconId, String)>;
}

// The selected observations, held in `AppState::bench_rows`
// (§ "The selected observations").
pub(crate) struct BenchRows {
    pub(crate) recon: ReconId,
    pub(crate) item: ItemId,
    /// Ascending, without repeats.
    pub(crate) observations: Vec<usize>,
}

impl AppState {
    /// Empty unless `label` is the focused item and the selection was made on it.
    pub(crate) fn selected_bench_observations(&self, id: ReconId, label: &str) -> &[usize];
    /// The one selected observation, when exactly one is: what the 3D viewer's
    /// bench figure draws larger.
    pub(crate) fn selected_bench_observation(&self, id: ReconId, label: &str)
        -> Option<usize>;
    /// Replace the selection; an empty list clears it. Refused for a track that
    /// is not the focused item and for an index past the end of the list.
    pub(crate) fn select_bench_observations(&mut self, id: ReconId, label: &str,
                                            observations: &[usize]) -> Result<(), String>;
    /// A click on one observation: alone, or with `extend` added to or taken out
    /// of the selection. A refusal is a failed row.
    pub(crate) fn pick_bench_observation(&mut self, id: ReconId, label: &str,
                                         observation: usize, extend: bool);
    /// What a move of the cursor does.
    pub(crate) fn clear_bench_rows(&mut self, id: ReconId);
}

// The index the search queries, in
// [sift_index.rs](../../crates/sfm-explorer/src/sift_index.rs), and the
// index files it is one of, in
// [index_files.rs](../../crates/sfm-explorer/src/index_files.rs); specified
// in [sift-index.md](sift-index.md) and [index-files.md](index-files.md).
impl AppState {
    pub(crate) fn sift_index(&self, id: ReconId) -> Option<&SiftIndex>;
    pub(crate) fn sift_index_state(&self, id: ReconId) -> IndexFileState;
    pub(crate) fn sift_index_path(&self, id: ReconId) -> Option<PathBuf>;
    /// Open the node's index files if they are there and nothing has looked
    /// yet, re-derive their states when a version has moved the image table,
    /// and remember the look either way.
    pub(crate) fn refresh_index_files(&mut self, id: ReconId);
    pub(crate) fn open_index_files(&mut self, id: ReconId, sift_index: Option<PathBuf>,
                                    cluster_patches: Option<PathBuf>) -> Result<(), String>;
    pub(crate) fn close_index_files(&mut self, id: ReconId) -> Result<(), String>;
    pub(crate) fn build_index_files_refusal(&self, id: ReconId) -> Option<String>;
    pub(crate) fn start_build_index_files(&mut self, id: ReconId) -> Result<(), String>;
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
means "the focused item", and the panel is what knows that; the steps take the
label, so the question of which item is answered in one place rather than
inside each step. `focused_item_label` is what a caller resolves it with.

**One `Seed` for the two steps that place an observation.** A caller arrives
holding one of three things -- a pixel, a pixel with a size or a shape read at
it, or a `.sift` feature, which carries both -- and the step should not care
which. `Seed::Pixel` with no radius means "I have no shape to give you, use the
one you have": the node's own default patch radius when a cluster
is being started, and the track's reference shape when an observation is being
added to one, which is what a right-click means. A `Seed::Feature` is read
through `AppState::sift_cache`, the cache the Image Detail overlay draws its
ellipses from, so a feature seeded here is the mark the person is looking at;
its stored affine is already the cluster stage's own convention and is passed on
as it stands.

**At the track stage the added pixel is the observation's keypoint.** Core's
`add_observation` writes it into the track slot as well as the seed, so a
right-click Add, the wire's `add_bench_track_observation` and a search all leave
a sighting that a reading measures where it was put and that, once `in`, a
commit writes without a fit first. At the cluster stage the same gesture is a
seed alone.

**The three steps that name a pixel are invoked from the Image Detail context
menu.** `start_bench_cluster` and `add_bench_observation` take a seed, and
`start_create_track_at_pixel` takes the pixel itself; the viewer's one way to
name a pixel is a right-click in that panel, where the two point edits that need
one already live
([`multi-panel-image-browser.md`](multi-panel-image-browser.md) § "Image Detail:
the context menu"). The third also answers a Control+Shift click there. Every
other step names an item and is a button in the Track Edit panel.

**Each step returns `Result<_, String>`.** The `String` is the sentence a
refusal shows, which is core's own wording behind a clause naming what was
being done. The caller is a menu entry or a button that has to say in one line
why nothing happened.

**The commit hands back the point it wrote.** One row of the reconstruction is
the whole of what it produces, and it has to be named: the step selects it, and
the wire reports its index and its portable id. It comes back
from the step rather than being looked up afterwards, because "the point this
commit wrote" is not a question the value can be asked once the version has
landed -- a replacement takes a **new** index and deletes the one it replaced,
and a creation takes whatever index the overlay had free.

**The steps that read photographs return as soon as the worker is running.**
They are `start_`-prefixed for that reason, and what they answer is whether the
operation could *begin*. The report lands frames or seconds later, through the
background machinery. What they refuse in the call is everything the track alone
decides (§ "The two steps that read photographs").

**Nothing asks for an evaluation.** It is not a step: the viewer keeps every
track's measurements the evaluation of the track as it stands, and the one call
a frame makes for it, `drive_bench_evaluation`, takes no item (§ "Live
evaluation").

### Example

```rust
let label = state.put_point_on_bench(PointRef::new(id, 1207), None)?; // one version
state.drive_bench_evaluation();                                   // the next frame: evaluates it
state.set_bench_verdict(id, &label, 3, Verdict::Out)?;            // one version, evaluated again
state.start_bench_fit(id, &label, None)?;                         // a task: moves it, then measures
// ... the report lands, which is one more version, and is evaluated again ...
state.commit_bench_track(id, &label)?;                            // one version, both halves
```

---

## One history for the pair

A version of the node is the reconstruction value, base plus overlay, **and**
the bench as it stood. The consequences are the ones the value model already
pays for, applied to one more field.

- **Undo and redo walk the pair.** The Edit menu's Undo, its shortcuts and the
  Edit History panel's jump move one cursor over one list, and whichever half a
  step changed comes back. A verdict, a stage change, a commit and then a
  deletion of some other point are four versions in one order, and undo
  retraces them in that order. The evaluations between them are none: they
  fill measurement slots in the version they were computed for.
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

**A bake puts the bench's placements through the same transform** as the value,
in the one version that states all three halves
([`edits/bake-transform.md`](edits/bake-transform.md)). A track-stage item's
placement and position are in the reconstruction's own coordinates, the same
coordinates the bake rewrites, so a bake that left them alone would stand the
focused item's square in the old frame and the 3D viewer's figure would jump.
The centre goes through the whole similarity, the axes are rotated and the
half-extent is scaled; a track at infinity keeps the rotation alone; a
cluster-stage item has no world geometry and keeps its `Arc`.

**Deleting an image renumbers the bench's observations with the image table**,
in the one version that deletes the image (`AppState::delete_image` pushes the
pair through `push_pair`). Every observation of every item names its
photograph by its index in the node's image table, and the delete moves each
later image down by one; carried along unchanged, every observation past the
deleted image would name the next photograph, a reading would score the wrong
pixels as `current`, and a commit would write them into the reconstruction.
So the bench goes through core's `Bench::delete_image`
([`../core/bench/bench.md`](../core/bench/bench.md) § "Deleting an image"):

- an observation in the deleted image is dropped, and one in a later image
  moves down by one with its place, verdict, pin and measurements;
- **an item whose observations were all in the deleted image is discarded**,
  in the same version, since nothing it was made of is left; when it was the
  focused item it is unfocused, as a discard of it would be. Undoing the delete
  brings the item back with the image;
- an item's origin is followed through the delete's own point map, like a
  selection, so a track from a point the delete kept still replaces that point
  when committed;
- the selected observations of the focused item follow their observations to
  their new indexes, and one in the deleted image is deselected
  (§ "The selected observations").

The changed items get a new `Arc` and the version a new document serial, so
the live evaluation reads them again on the next frame. The delete's Action Log
row says what it did to the bench when any bench observation was in the image:
`Deleted image IMG_0007.jpg from run_a; dropped 3 bench observations in it and
discarded IMG_0007@142,198, which had no other observations`. A delete that
reaches no bench observation says nothing about the bench, and a delete no
item observes an image at or after leaves the bench the same `Arc`.

---

## What a step writes

Every bench step is **one version**, with one Action Log row of kind `Bench`,
and an Edit History row that does not move the on-disk mark. The commit is the
exception in one respect only: its row is of kind `Edit`, because it is one
([`edits/commit-track.md`](edits/commit-track.md)).

| Step | Version label |
|---|---|
| Put a point on the bench | `Put point 1207 on the bench as pt3d_a1b2c3d4_1207`; a put of the viewed point that carries Track View's read-only bars names the bars that differ from the defaults: `Put point 1207 on the bench as pt3d_a1b2c3d4_1207, with min ZNCC 80%` (§ "Live evaluation") |
| Put a point on the bench again, once it has a patch frame its item lacks (§ "A view-only bench") | `Rebuilt pt3d_a1b2c3d4_1207 from point 1207, which now carries a patch frame` |
| Start a cluster from a pixel | `Started IMG_0042@142,198 on the bench` |
| Create a track at a pixel (the put; its commit is the commit's row) | `Created IMG_0042@142,198 at (142.0, 198.0) in IMG_0042.jpg with the clusters member` |
| Add an observation | `Added image_012.jpg to pt3d_a1b2c3d4_1207` |
| A verdict | `Turned image_012.jpg out of pt3d_a1b2c3d4_1207` |
| Unpin one verdict | `Handed image_012.jpg back to the thresholds in pt3d_a1b2c3d4_1207: in` |
| Unpin several, or all | `Handed 4 verdicts back to the thresholds in pt3d_a1b2c3d4_1207: 1 turned in and 2 turned out, leaving 5 in, 3 out` (what moved, then the track's totals; `none moved` when the bars kept every verdict) |
| Slide the patch | `Moved pt3d_a1b2c3d4_1207 by 0.123 units to (1.204, -0.318, 4.006)` |
| Place one sighting | `Moved observation 3 of pt3d_a1b2c3d4_1207 to (1041.6, 1702.9) in IMG_0042.jpg (2.3 px)` |
| Resize the patch | `Resized pt3d_a1b2c3d4_1207 to 7.4 px in IMG_0042.jpg` |
| Turn the patch | `Rotated pt3d_a1b2c3d4_1207 by 12.3 degrees` |
| Turn one sighting's shape | `Rotated observation 3 of IMG_0042@142,198 by 12.3 degrees` |
| Apply the thresholds | `Applied the thresholds to IMG_0042@142,198: 3 in, 1 out, 1 pinned, 0 unmeasured` |
| Accept a walk | `Accepted the walk of observation 3 of pt3d_a1b2c3d4_1207: moved 11.2 px to (1050.8, 1702.4) in IMG_0042.jpg` |
| Fit | `Fitted IMG_0042@142,198: finite at (x, y, z): depth score 5210.3 over the 25 threshold, likelihood ratio 5208.9, 4.012 from the observing cameras, at 0.468 px noise over 3 rays up to 15.204 deg apart` |
| Set the stage | `Set IMG_0042@142,198 to the track stage` |
| Split | `Split 2 observations off pt3d_a1b2c3d4_1207 as pt3d_a1b2c3d4_1207-split` |
| Discard | `Discarded IMG_0042@142,198 from the bench` |
| Duplicate | `Duplicated IMG_0042@142,198 as IMG_0042@142,198 copy` |
| Rename | `Renamed IMG_0042@142,198 to bull-nose on the bench` |

The Action Log row is that sentence plus the version serials, exactly as an
edit's is: `Turned image_012.jpg out of pt3d_a1b2c3d4_1207 (v7 → v8)`. A
refusal is one failed row carrying the refusal's sentence.

**A step that changes nothing pushes no version.** Setting the verdict an
observation already has, setting the stage a track is already at, renaming an
item to the label it holds, and a drag of a handle that
ends where it started each report that nothing happened and leave the history
alone: a row that has to be undone for nothing is worse than no row.

**A size is reported in the pixels of the sighting it was named at.** A world
half-length says nothing to someone looking at a photograph, so the resize's
sentence states the patch's half-width in that observation's own image -- the
patch re-anchored on it, projected -- and falls back to the world number only
when the patch does not project there.

**A fit's sentence says which representation the rays earned, and why.** The
finite-versus-bearing decision is the step's real outcome on a distant track
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "Finite
points and bearings"), so the **version label** carries it -- `Fitted
IMG_0042@142,198: at infinity along (0.553, -0.809, -0.198): depth score 4.2 and
midpoint bound 0.0 under the 25 threshold, at 0.216 px noise over 8 rays up to
0.312 deg apart` -- rather than naming the item and stopping there. A person scrolling the history can therefore see which fit
crossed the boundary, and on what evidence, without opening each version and
re-reading its coordinate. The **Action Log row** is the whole report, which is
that sentence with the counts around it: how many sightings the kernels placed
and how many the walk bound left at their seeds.

**A reconstruction holding only bearings has no noise level to classify at.**
The test weights the sightings at the reprojection noise measured over the
reconstruction's finite points, and with none there is nothing to measure, so
Track View's *Fit* and the upgrade to the track stage are refused with that
sentence in the Action Log rather than run at a guessed level. The panel offers
no way to give one: an agent passes `sigma_px` to `fit_bench_track` or
`set_bench_track_stage` ([mcp-server.md](mcp-server.md)), and a person adds or
keeps a finite point in the reconstruction first.

### Labels

A label is minted from what the item was made from and is how the item is named
everywhere -- in the Scene tree, in Track View's header, in every row and
version label. The minting is core's ([`../core/bench/bench.md`](../core/bench/bench.md)
§ "Labels") with one exception the viewer supplies: **a track put on the bench
from a point is labelled by that point's portable id**, `pt3d_a1b2c3d4_1207`,
because that id names the content the point is a row of and the version graph
that content sits in, and core has neither ([`goto-point.md`](goto-point.md)).

**Putting a point on the bench twice focuses the track it already made.** The
person asked to work on that point, and there it is; a second item for one point
would be two answers to one question. The test is the origin, followed to the
cursor. That case pushes no version.

---

## A view-only bench

**The bench of a `sift_files` reconstruction is view-only.** Such a
reconstruction keeps its patches in the `.sift` files rather than in the
`.sfmr`, so a point put on its bench carries no patch frame, and a track built
there could not be committed back. Its bench items can be viewed as fully as
the data allows: a point goes on the bench, is focused, and shows its header,
its keypoints (which the load fills from the `.sift` files) and whatever
evaluation its frame allows. Every step that edits an item is refused up front.

One question decides it, `AppState::bench_view_only_refusal`, and every editing
step asks `AppState::bench_edit_refusal`, which is the busy refusal followed by
it. The refusal is one sentence, `bench::view_only_sentence`:

> Bench editing needs embedded patches, and dino keeps its patches in .sift
> files (sift_files), so its bench is view-only. Convert it with Convert to
> Embedded Patches (the reconstruction's menu in the Scene panel, or
> convert_to_embedded_patches), then put the point on the bench again.

It names the one remedy, which is never itself refused on a `sift_files` node
that is not busy. The same sentence is given everywhere:

- **The steps.** Starting a cluster, adding an observation, a verdict, a pin
  or unpin, the thresholds, every patch edit, *Accept walk*, a duplicate, a
  split, a commit, a fit, a normal step, a stage change (including one to the
  stage the track is at, which is refused rather than answered as no effect),
  and both searches. They ask it in `bench_step_target` and the step methods
  that do not go through it, so the refusal comes before any other.
- **The wire.** Every bench tool that edits an item asks it before it reads
  anything else (`mcp/bench.rs`'s `edit_target`), and so does
  `create_bench_cluster`.
- **The panels.** Track View greys every toolbar button but *Discard* and
  *Rename*, the threshold boxes, the *Keep* heading's pin, the row menus'
  unpin, search and *Accept walk* entries, and draws a row's *Keep* switch
  and pin greyed, the switch grey rather than green when on, taking no click,
  each with the sentence as its hover. Image Detail greys
  *Start cluster on the bench here* and *Add observation to bench track here*
  with it (`BenchMenu::view_only`), and its bench layer draws the focused item
  with no handle on it. The 3D viewer's bench figure offers no handle either.

**Not refused**: putting a point on the bench (`create_bench_track`, *Edit on
Bench*, a double-click), focusing and unfocusing, selecting rows, renaming,
discarding and clearing the bench, none of which edits a track, and *Find
Nearby Tracks* without a commit, whose items are built with a frame and viewed
like any other. *Create Track Here* and a committing find are refused as before
(§ "Create Track Here").

**After Convert to Embedded Patches, putting the point on the bench again
rebuilds its item.** The conversion leaves the bench as it was, so an item put
on the bench before it still has no frame. A put of a point whose item is a
track-stage track with no frame, on a node that is no longer view-only and
whose point now has one, builds the track afresh from the point
(`AppState::rebuilt_with_frame`) and replaces the item under its own label and
`ItemId`, as one version (`Rebuilt … from point …, which now carries a patch
frame`). The bench is editable from the conversion on, so the item may have
taken edits before the re-bench, and the rebuild keeps them (`with_frame_of`):
the stage (frame, position, bitmap, colour) and the origin come from the fresh
build, and the item keeps its thresholds, every verdict and pin, and every
observation that was sighted elsewhere or added. An observation still at the
point's keypoint takes the fresh build's measurement. An item nobody edited
comes out exactly as a fresh put of the point would build it. Until the
re-bench, *Duplicate* is refused on such an item
(`AppState::duplicate_refusal`), because the copy drops the origin and could
never be given the frame. Rebuilding on the re-bench
was chosen over refreshing every item when the conversion lands because it is
one step in one place (`put_point_on_bench`), it needs nothing from the
background task that pushes the conversion's version, and the refusal sentence
already tells the person to put the point on the bench again. Undoing the
rebuild puts the frame-less item back; undoing the conversion makes the bench
view-only again.

## The focused item

The item Track View edits while its *Edit* box is ticked, and the item a bench
panel's gesture or a wire call means when it names none, is the **focused
item**: `AppState::focused_item`, a `FocusedItem` naming a node and an item on
that node's bench by its `ItemId`. There is **at most one for the whole
viewer**, not one per bench. A track and a cluster are focused alike, and an
item with no point in the reconstruction is focused like any other, since a
`FocusedItem` names an item and never a point.

**It is not part of any version.** It is held beside the selection, and
focusing and unfocusing are not bench steps: they push no version and write one
Action Log row of kind `Selection`, `Editing IMG_0042@142,198` and `Stopped
editing IMG_0042@142,198`, folded like other selection rows. Focusing the item
already focused writes the no-effect row `Editing IMG_0042@142,198: no effect,
it is being edited already`; unfocusing with nothing focused writes `Stopped
editing: no effect, no item is being edited`. Neither is refused while a
background task holds the node, since neither changes the bench. So an undo is
never spent on a change of focused item: a person who turns an observation out,
clears *Edit* and presses Ctrl+Z gets the verdict back.

**It is named by `ItemId`, not by label.** Core gives every item an ID when it
is put on the bench and keeps it across a rename and every step that replaces
the item's value ([`../core/bench/bench.md`](../core/bench/bench.md)), so a
rename leaves the item focused under its new label.

**It keeps the selection with it.** While an item is focused, the selected node
is its node and the selected point is its origin followed to the cursor
(`AppState::focused_origin`), or no point. Focusing and unfocusing move the
selection to keep that true, and a selection that would break it unfocuses the
item ([`track-view.md`](track-view.md) § "Transitions").

What changes it:

- **focusing** it: ticking *Edit* over a point an item already came from,
  ticking it with no point selected (the most recent item, below), a click on
  a chip in Track View's recent items strip, a Scene tree double-click on a
  Bench row, and the wire's `focus_bench_item`. Focusing an
  item on one node unfocuses whatever was focused on another. **Focusing
  selects**: the item's node, and its origin when that resolves at the cursor,
  or else the point selection is cleared; a selected image stays when it belongs
  to that node. The item is focused before its origin is selected, so the
  selection rule below sees the origin as the focused item's. These selection
  changes write no rows of their own;
- **unfocusing**: clearing *Edit* and the wire's `unfocus_bench_item`.
  **Unfocusing selects**: the item's origin when it resolves at the cursor, or
  else the point selection is cleared; the node stays selected. No rows beyond
  the `Stopped editing` one;
- **a selection that leaves the item**: `AppState::select_point` of any point
  but the focused item's origin, and `select_point`, `select_image`,
  `select_camera` or `select_recon` naming another node, unfocus it with its
  `Stopped editing` row before the selection's own, and push no version. The
  selection is what that gesture made it. Clearing the point selection, and an
  image or a camera of the item's node, keep it focused. The selection that
  follows the point maps after an edit assigns the point directly and does not
  unfocus;
- **a step that puts an item on the bench** focuses the item it put on, and
  selects as focusing does, with no row of its own beside the step's: a put from
  a point, *Start cluster*, *Create Track Here*, *Find Nearby Tracks* (its
  `1a`), *Duplicate* and *Split*. A cluster, a duplicate and a split have no
  origin, so the point selection is cleared. *Create Track Here* then commits
  the item, and a commit re-seats the item's origin on the point it wrote
  before selecting it, so the item stays focused;
- **a step that takes the focused item off the bench**, a discard or *Clear
  the Bench*, unfocuses it, so Track View leaves Edited mode instead of
  switching to an item nobody asked for. The selection is then what an unfocus
  leaves, from the origin as it resolved before the step;
- **undo, redo and a jump** leave it focused while the version landed on holds
  it, and unfocus it when that version does not (an undo past its put, a redo
  past its discard), with no row beyond the move's own and the selection left
  as the move's map left it. They never focus an item: an undo of a discard puts
  the item back unfocused;
- **closing its node** unfocuses it.

**The recent items.** `AppState::recent_items` lists every item focused this
session, most recently focused first and each once: `set_focused_item`, the
one place the focused item changes, moves a newly focused item to the front.
An entry is dropped when its node is closed and the list emptied by *Close
All*; an entry whose item is not on its node's bench at the cursor is skipped
by `most_recent_item` rather than dropped, so an undo of a discard brings it
back. It is session state, outside every version. Ticking *Edit* with no point
selected focuses its first resolving entry ([`track-view.md`](track-view.md)
§ "Transitions"), and Track View's recent items strip draws the first eight
resolving entries other than the focused item
([`track-view.md`](track-view.md) § "The recent items strip").

---

## The selected observations

The bench carries a selection of the focused item's observations: the rows
highlighted in Track View's Edited mode. *Split off N rows* takes them, and when
exactly one is selected the 3D viewer's bench figure draws its mark larger
([`viewer-3d-bench-layer.md`](viewer-3d-bench-layer.md)). It is held in
`AppState::bench_rows` as a `BenchRows`, which names the node and the track's
`ItemId` beside the observation indexes, so every gesture that reads or sets it
reads or sets one value:

- a click on a Track View row selects that observation alone, and a Ctrl-click
  or Shift-click adds it or takes it out (`pick_bench_observation`);
- a click on a mark in Image Detail's bench layer or on the 3D viewer's bench
  figure selects that observation alone, since the mark and the row are one
  observation;
- the wire's `select_bench_observations` replaces the whole set, and
  `get_bench_track` reports it as `selected_observations` (§ "The wire").

**Selecting is not a step.** It pushes no version, and a change writes one
Action Log row of kind `Selection` -- `Selected observations 0, 2 of bull-nose`
-- folded into the row before it when that was also a change of the selected
observations, so a run of Ctrl-clicks reads as one line.

**Only the focused item has selected observations.** The indexes mean
something only against one track's list, and Track View shows only the focused
item, so a selection on any other track is refused. What clears it:

- **a move of the cursor**: undo, redo or a jump. The version landed on may
  hold another list of observations for the same item, and an undo does not
  bring back a selection, because a selection is not in a version;
- **a change of focused item**: a focus of another item, an unfocus, a discard
  of the focused item, or a step that puts another item on the bench and
  focuses it. Coming back to the track later does not bring the selection back;
- **a split**, which renumbers the observations left on the track;
- **closing the node**.

A rename keeps the selection, since it keeps the item's ID and the observations
are the same ones. Every other step on the focused item keeps it, because those
steps append observations or change them in place and never renumber the list.
**Deleting an image** does renumber it, and the selection follows: each
selected row moves to the index its observation has after the delete
(`AppState::follow_bench_rows`), a row in the deleted image is deselected, and
a selection left with no rows is cleared. A delete that discards the focused
item clears its selection with the focus.

---

## Live evaluation

**Every track on a bench is kept evaluated.** The measurements a track shows --
in Track View's table, in the Image Detail bench layer and on the wire -- are
the evaluation of its current inputs, or an evaluation of those inputs is
running or about to start. There is no *Evaluate* button and no tool for it: a
track put on the bench is evaluated first, and every change to an input of the
evaluation evaluates it again. The code is
[bench/live.rs](../../crates/sfm-explorer/src/bench/live.rs).

**The inputs are what the evaluation job captures**, and the freshness test
holds the same three things:

| Input | What changes it | How the change is seen |
|---|---|---|
| The track value: observations, their seeds and keypoints, verdicts, pins, the stage and its patch frame, position and template | every bench step on the track -- a put, an add, a verdict, a patch or sighting edit, *Accept walk*, the thresholds (their painting moves verdicts), a fit, a stage change, a search, a split, a duplicate -- and an undo, redo or jump that lands on another version of it | the track is a new `Arc` |
| The document half of the version: the poses and camera intrinsics the kernels project with | every document edit under the track -- a bundle adjustment, a resection, a refit or switch of the camera model, a commit -- and an undo, redo or jump across one | the version's `document_serial` moves |

The radius the evaluation looks for each peak within is the track's own
`max_shift_px` bar, so it is part of the track value: moving it is a thresholds
step like any other. There is no viewer-wide radius.

So no step has to remember to ask for an evaluation, and none does. The item's
`ItemId` is part of the key too, so a rename, which keeps the ID and the track
value, leaves an evaluated track current. The thresholds are not an input on their own: the evaluation reads with
its gates off, and the bars reach it only through the verdicts they paint.

**One evaluation runs at a time, and a stale one is cancelled.** Once per frame,
after the frame's steps, `drive_bench_evaluation` lands a finished evaluation,
and then either cancels the running one when its inputs are no longer the
track's, or, when nothing is running, starts the next track whose evaluation is
`Evaluating`: the viewed track first (§ "The viewed track" below), then the
focused item, whichever node it is on, then the rest in bench order, in scene
order. The viewed track and the focused item do not exist together, since the
viewed point is defined only while no item is focused on its node, so whichever
of the two Track View shows goes first. It does not start the next until a cancelled one has
reported back, so a box drag that moves an input on every frame has at most
one worker behind it rather than a queue. It also waits while a background task
holds the node, since that task's answer replaces the inputs it would read.

**An answer is installed only if its inputs are still the track's.** An
evaluation that lands after a step has moved the track on -- cancelled, or
measured before it saw the flag -- is dropped, and the track reads `Evaluating`
until the evaluation of its new inputs lands. A result that matches is written
into the version at the cursor in place (`History::replace_current_bench`):
**no version is pushed and no Action Log row is written.** An evaluation fills
measurement slots and sets each unpinned verdict to what the bars propose from
them, moving nothing a person put there, so a version per
evaluation would put a step in the history for every edit that Undo would then
have to walk back over, each restoring numbers that no longer matched the
inputs beside them. The version keeps its serial and its label.

**A repaint that moved a verdict is read once more.** A track-stage reading is
scored against the `in` rows, so when the evaluation's repaint changes which
rows are `in`, the readings it brings back were taken under the rows before it.
The track it lands carries core's repaint mark
(`EditableTrack::repainted`, [../core/bench/editable-track.md](../core/bench/editable-track.md)
§ "Evaluating"), and it is not recorded as current: the next frame evaluates it
again. Core reads a marked track without repainting, so that second evaluation
settles the track, and a row it leaves out of step with the bars shows in Track
View as a switch disagreeing with its cell's colour until the next step.

**A track put on the bench from a point arrives with every row pinned `in`**,
so its evaluations leave the point's verdicts alone; rows added later by a
search, the far-field sweep or a pixel gesture arrive `out` and unpinned, and
their first evaluation takes them in when they clear the bars. A cluster
started from a pixel arrives with its one seed pinned `in` for the same reason:
it is what the person pointed at.

**A track-stage track with no bitmap gets one fused.** A patch step -- a move,
a resize, a spin or a tilt -- drops the consensus bitmap, because it was fused
over the square as it stood. When the evaluation reads a track-stage track that
has a placement and no bitmap, it also runs core's `fuse_bitmap_in_place` on
the photographs it has already decoded. That renders the `in` sightings through
the patch as it now lies, at their keypoints, and writes the bitmap and the
colour at its centre, moving nothing. So a tilted patch shows its texture again
as soon as the evaluation lands, and can be committed into a reconstruction
that stores a bitmap per point without a fit first. The bitmap is installed with
the measurements, under the same rule: no version and no Action Log row. A track with fewer than two
`in` sightings that carry a keypoint has nothing to fuse, and stays without one.

**The node is not locked by it.** Every step stays available while an
evaluation runs, and taking one is what cancels it. It is not a background task
either: it does not appear in the Background panel, is not what
`get_background_task` reports, and does not refuse another operation.

**Four states**, which Track View's toolbar shows and the wire reports under
`evaluation`:

- `Current` -- the numbers are the evaluation of the track as it stands.
- `Evaluating` -- the inputs have changed since, and an evaluation of the
  current ones is running or starts on the next frame. The numbers are the
  previous evaluation's, and are presented as that.
- `Refused(why)` -- core's `evaluate_preconditions` refuses the track as it
  stands, a track-stage track with no patch frame, in the sentence `Cannot
  evaluate {item}: …`. Nothing runs until a step changes that.
- `Failed(why)` -- the evaluation of the current inputs failed, for instance
  because a photograph could not be read. It is not retried until an input
  changes.

### The viewed track

The live evaluation has a second kind of subject beside the bench's items: the
**viewed track**, the **viewed point** read as an editable track and held off
every bench. The viewed point is the selected point on the selected node while
no item is focused on that node (`AppState::viewed_point`). The code is
[bench/viewed.rs](../../crates/sfm-explorer/src/bench/viewed.rs), and the live
evaluation names its subject as

```rust
enum Subject {
    Item { node: ReconId, item: ItemId },
    Viewed { node: ReconId, point: u32 },
}
```

**It is built the way a put builds a bench track**: core's `create_track`, from
the version at the cursor, under the label a put gives it (the point's portable
ID) and with the same options. Its rows arrive `in` and pinned, the
leave-one-out ZNCC is read back from the point's stored column, and the bars are
the defaults. It is held in `AppState::viewed_tracks` as a `ViewedTrack` (node,
point, document serial, label, track, evaluation state), never in a version: no
step accepts it, the Scene tree does not list it, and the bench layers do not
draw it. A deleted or out-of-range point gives none.

**Who asks for it.** Track View's body draws from `&AppState` and cannot build a
track, so the frame clears the current viewed track before the dock draws
(`hide_viewed_track`), and the dock asks for it before it draws Track View
(`refresh_viewed_track`), which builds it or takes it from the cache and makes
it current. A frame that does not draw Track View therefore leaves no current
viewed track. `AppState::viewed_track` answers only while the key it was asked
for is still the one the selection and the cursor give.

**The cache.** The last eight viewed tracks are kept, most recently viewed
first, keyed on `(node, point, document serial)`, each with its evaluation
state. A point's content at a given document serial is fixed, so a track built
for a key stays right for it, and another point, another node, or a document
edit or an undo that moves the document serial is another key. So clicking back
to a point whose evaluation has landed draws current numbers and starts nothing,
and an undo back across a document edit finds the track built before it. Closing
a node drops its entries, and *Close All* empties the cache. A bench item's
evaluation is not shared with the viewed track of the point it came from, since
the bench copy may have been changed.

**The freshness rule is the bench's**: the track's `Arc` and the document serial.
The viewed track carries its own state (`Evaluating` when built, or `Refused`
when core's `evaluate_preconditions` refuses it, in the sentence `Cannot evaluate
{point id}: …`). A change of the viewed track -- a selection change, another
node, a document edit or an undo moving the document serial, or Track View no
longer drawn -- cancels its running evaluation, as a step on a bench track
cancels that track's. The evaluation reads the photographs as a bench track's
does, and waits while a background task holds the node.

**A landed result is installed into the cached viewed track**, never into a
version: no version is pushed and no Action Log row is written. The rows are
pinned, so the evaluation moves no verdict; the repaint mark is handled as for a
bench track all the same.

**The read-only bars.** `AppState::viewed_thresholds` holds the bars Track
View's threshold boxes show while the panel draws the viewed track. They are
session state, starting at the bench's default bars, and `set_viewed_thresholds`
changes them without pushing a version or writing a row. They are never applied
to the viewed track: they judge its readings, and each row's verdict by them is
core's `verdicts_if_unpinned` over a copy of the track carrying them
(`AppState::viewed_verdicts`), the verdict the bench's own evaluation would give
the row once unpinned.

**Moved bars carry onto the bench with the point.** A put of the viewed point
(`put_point_on_bench` for the point that is `viewed_point`: ticking *Edit*, *Edit
on Bench*, a double-click on the point or one of its features, which select the
point before putting it, or the wire's `create_bench_track` naming it) while the
read-only bars differ from the defaults gives the new track those bars, in the
same version as the put. The rows arrive pinned `in` as always, so the bars
change no verdict until rows are unpinned. The version label, and so the wire's
reply, names the bars that differ, in percent for the three ZNCC bars and in
patch-grid px for the shift and self-similarity bars (§ "What a step writes").
A put of any other point, a cluster, and focusing a point's existing item (which
keeps that item's own bars) carry nothing. The read-only bars keep their values
after the put.

## The two steps that read photographs

A fit and a stage change run as **background tasks**
([`background-tasks.md`](background-tasks.md)), under `Fit track` and `Set
track stage`, beside `Geometry search`, which reads
photographs too, `Search descriptors`, which reads a `.kdf` and a capture's
`.sift` files rather than photographs, and `Build index files`, which reads
the `.sift` files and then the photographs ([`index-files.md`](index-files.md)),
and `Create track at pixel`, which reads the photographs and the index files
(§ "Create Track Here"), and `Find nearby tracks`, which reads the photographs,
the index files and the `.sift` files (§ "Find Nearby Tracks").
All of them are
**cancellable**: the kernels they run take the `Progress` for their phases and
poll its flag as well -- between the fit's rounds, between the views the
localizer or the geometry selector renders, in front of the forest query and
between the candidates for the descriptor search, between the candidates the
geometry search appends, and in every phase of the index-files build
([`index-files.md`](index-files.md)) -- so a cancelled step ends as a
cancellation, pushing no version. The declaration is held to that by the background tests,
which cancel each one over a fixture that can really run it.

**The two and the live evaluation are one family, and the geometry search is
beside them rather than in it.** What names them is the split validation below:
each publishes the half
of its own refusal that reads no photograph, so a caller can refuse in front of
the decode. The geometry search reads photographs and cancels the same way, but
its refusals are the panel's and the wire's
([`track-view.md`](track-view.md) § "Row gestures"), not a published
core precondition, because what it needs of a track -- the track stage, a
fitted patch, and a sighting to search from -- the viewer already holds.

**What the track alone decides is decided before the task starts.** Core
publishes the half of each step's own validation that reads no photograph --
`bench::evaluate_preconditions`, `bench::fit_preconditions` and
`bench::set_stage_preconditions`
([`../core/bench/editable-track.md`](../core/bench/editable-track.md)) -- and
`start_bench_fit` and `start_bench_stage` ask theirs before they build a job,
as the live evaluation asks `evaluate_preconditions` before it starts one
(§ "Live evaluation"). So a track being fitted with fewer than two `in`
observations (which an *evaluation* of the same track permits), or one being
taken down to the
cluster stage with no frame or no position, is a refusal of the **gesture**: a
sentence in the caller's own hand, no task, no version. The step itself calls
the same function first, so the two answers cannot drift. What is left for the
task is everything that needs the pixels, which is the rest.

**The photographs are decoded on the worker**, along with the kernel work that
reads them: the file reads and the pyramid builds are seconds of work in their
own right, and a step that did them on the GUI thread would freeze the frame --
and hold the wire's reply window shut -- for all of it before the task it defers
to had begun. So what the gesture does here is clone a handful of
handles: `ViewSources` carries the node's cameras and poses, the path of each
photograph the step reads, and a clone of the viewer's photograph cache
(`AppState::photographs`, [../core/camera/photograph-cache.md](../core/camera/photograph-cache.md)).
On the worker `ViewSources::decode` asks the cache for every path at once
(`PhotographCache::get_many`), which hands back what it holds and decodes the
rest in parallel, under a `decode images` phase whose note says how many it
read from disk, how many it reused from the cache, and how full the cache is
afterwards. The cache holds pyramids rather than bare photographs, so a
photograph the viewer had is neither decoded, nor copied, nor pyramided a
second time, and what the worker decodes stays in the cache for the next step
and the panels. A photograph that cannot be read is the
worker's refusal, arriving as the task's failed row rather than as a refusal of
the gesture -- which is honest: whether a file is readable is not a question the
gesture can answer without doing the read. Beside those views the worker gets a
clone of the value at the cursor and a clone of the track, so it holds no
reference into the scene. The images it decodes are the ones the track's
observations name; every other entry of the view slice is a one-pixel
placeholder, which no kernel samples.

**A report lands on the observations it measured.** No bench step renumbers
the observations and a measurement is keyed by observation index, so a report
computed against the track as it stood when the task began still applies when
it finishes. Deleting an image, the one edit that renumbers them, is refused
while a task holds the node. What it cannot survive is the item leaving the bench at the
cursor: then it is discarded with one Action Log row saying so, and no version.
That row is a guard rather than an everyday outcome -- the node is locked for
the duration of its own task, so nothing can take the item off in the meantime
-- and it is what keeps a report from inventing an item to land on.

---

## Create Track Here

*Create Track Here* builds a track at one pixel of a posed photograph and
commits it, as one gesture: Image Detail's context-menu entry, directly above
*Edit on Bench*, or a left click there with Control and Shift held
([`multi-panel-image-browser.md`](multi-panel-image-browser.md)), and the wire's
`create_track_at_pixel`. All three are `AppState::start_create_track_at_pixel`,
in [bench/track_at_pixel.rs](../../crates/sfm-explorer/src/bench/track_at_pixel.rs).
The track is core's `build_track_at_pixel`
([`../core/bench/track-at-pixel.md`](../core/bench/track-at-pixel.md)), the
cascade of four members that each find the pixel's sightings their own way.

```rust
// The menu entry and the Control+Shift click: a refusal is one failed row.
state.create_track_here(ImageRef::new(id, 0), [142.0, 198.0]);
// ... the worker lands: two versions, the put and the commit ...
```

**What stops it before the worker** is `create_track_here_refusal`, which the
greyed entry, the step and the wire all read: a background task holding the
node, an image with no pose, and a node whose observations are `.sift`
features, because the commit it ends in writes keypoints inline. A pixel off
the photograph is refused too, not clamped: the track is built at the pixel
named or not at all.

**It runs on a worker** as the background operation `Create track at pixel`,
cancellable. What crosses to the worker is what a photometric step's worker
gets (`ViewSources` for every image of the node, since the cascade may look in
any of them, and a clone of the value at the cursor), plus what the members
read beside the photographs: the SIFT index's forest handle and every image's
`.sift` path, read there into the keypoints the constellation member queries
with, and the cluster-patches `.matches` path, read there into the clusters
the clusters member searches. **An index file is handed over only when it is
`current`** ([`index-files.md`](index-files.md)). One that is missing or stale
is left out, and the member that reads it refuses and names what it lacked; the
transfer and sweep members read neither and run as usual. The node's files are
opened on sight and re-judged when the run starts
(`refresh_index_files`), so the states it reads are the node's as it stands.

**The track arrives with its bitmap.** `build_track_at_pixel` fuses the
consensus bitmap and colour where the track stands before it returns
([`../core/bench/track-at-pixel.md`](../core/bench/track-at-pixel.md) § "The
finish"), so the commit takes the track as the operation returned it, on a
reconstruction that stores a bitmap per point as on one that does not.

**A track that comes home is two versions**, in the order they happened. The
first is a bench step: the track put on the bench under the label a cluster
started at that pixel would take (`IMG_0042@142,198`), focused, one `Bench`
row whose sentence names the pixel, the member that built it, its `in` count and
median ZNCC with the median middle ZNCC beside it (`median ZNCC 93% / 71%`),
and any members that refused before it. The second is
`commit_bench_track`, the step Track View's *Commit* takes, so its version, its
`Edit` row, the selection of the written point and the item left seated on it
are that step's own. The operation's row is written first and the commit's
after it, both as whoever asked for the run. One Undo takes back the point and
leaves the track on the bench, focused, which is where a person who wants to
work on it would want it. A commit that is refused, say over a track with fewer
than two `in` sightings, is one failed `Edit` row naming the item the track
stays on the bench as.

**A run every member refuses pushes nothing** and writes one failed `Bench` row
in one sentence: the pixel, the last member tried with the stage it refused at
and its reason, and then what the index files lacked when the run started,
naming the member each would have fed and ending on *Build Index Files* (or on
why the node has nowhere to put them):

> Cannot create a track at (135.0, 6.0) in IMG_0042.jpg: every member refused;
> the last, constellation, at constellation: no indexed keypoint sits within 30
> px of the observation, out of 2352 in the image. No cluster patches file is
> open, so the clusters member had nothing to read. Build Index Files (the
> Index Files row in the Scene tree) makes both.

Every member's refusal is one of the row's detail lines, `clusters refused at
clusters: no cluster has a member within 16 px of the pixel` and so on in the
order tried, which is where the Action Log shows an operation's detail when the
row is expanded; the wire's refusal carries the same lines.

---

## Find Nearby Tracks

*Find Nearby Tracks* asks what the other photographs agree is at one pixel of a
posed photograph, or near it, and puts the answer on the bench: Image Detail's
context-menu entry, directly below *Edit on Bench*
([`multi-panel-image-browser.md`](multi-panel-image-browser.md)), and the
wire's `find_nearby_tracks`. Both are `AppState::start_find_nearby_tracks`, in
[bench/nearby_tracks.rs](../../crates/sfm-explorer/src/bench/nearby_tracks.rs).
The search is core's `find_nearby_tracks`
([`../core/bench/nearby-tracks.md`](../core/bench/nearby-tracks.md)): the
**nearby tracks**, 3D points near the pixel that several photographs see,
grouped into ranked **depth layers**, each track labelled with its layer's
rank and its place in the layer. Where *Create Track Here* builds the one
track at the pixel, this finds the geometry around it that the photographs
already agree on.

```rust
// The menu entry: at the pixel the menu was opened at, committing.
state.find_nearby_tracks_here(ImageRef::new(id, 13), [412.0, 230.0]);
// The wire, with the tracks put on the bench and nothing committed.
state.start_find_nearby_tracks(ImageRef::new(id, 13), [412.0, 230.0], false, None)?;
// ... the worker lands: one version, or one row when nothing usable was found ...
```

**What stops it before the worker** is `find_nearby_tracks_refusal`, which the
greyed entry, the step and the wire all read. With `commit` its reasons are
*Create Track Here*'s, word for word, since it ends in the same commit: a
background task holding the node, an image with no pose, and a node whose
observations are `.sift` features. Without `commit`, which only the wire
offers, the last does not apply. A pixel off the photograph is refused, not
clamped.

**It runs on a worker** as the background operation `Find nearby tracks`,
cancellable. It gets what *Create Track Here*'s worker gets -- `ViewSources`
for every image, a clone of the value at the cursor, the SIFT index's forest
handle and the cluster-patches path **only when each is `current`** -- and
every image's `.sift` path, which it reads there, when every image has one,
into the keypoints and descriptors guided matching reads and the keypoints the
constellation source queries the index with. A source whose input is missing is
skipped and named in core's report, not refused: a node with no index files and
no `.sift` files still gets its own points and the far-field sweep. The grey
images the far-field sweep and the layers sample, and the rays through the
keypoints, are built on the worker for the run and dropped with it, since
building them is a small part of a run. They are built from the pyramids the
viewer's photograph cache hands back, which decodes only the photographs it
does not already hold and keeps them for the next run. The options
are core's defaults, the harness's, with the caller's `label` as the group
label.

**What lands is one version.** Every track in core's `bench_order()` -- the
usable ones, best-ranked layer first and within a layer nearest the pixel first,
less the duplicates, whose built track repeats an existing point or a track
before it and which carry no label -- goes on the bench under its label, one after another, and the value and the
bench that result are pushed as one pair, so one Undo takes back the whole find
and one Redo puts it back:

- **an existing point** goes on as its own track, as *Edit on Bench* puts it
  (core's `create_track`, seated on the point), under the label with its
  point, `frame_13@412,230 1b pt 812`, and is never committed again. A point a
  bench item already came from keeps that item under the label it has, as
  *Edit on Bench* focuses the item it made rather than putting a second one
  on, so a second find at the same pixel puts no copies of its existing points
  on the bench;
- **a built track** goes on under its label, `frame_13@412,230 2a`, and with
  `commit` is committed as a new point by core's `commit`, then seated on that
  point as `commit_bench_track` seats a creation. The commits' point maps are
  chained into the version's, and the points they created are named by one
  content hash over them all, in the order they were made
  ([`goto-point.md`](goto-point.md)). A commit that refuses leaves its track
  on the bench and the rest are committed;
- **a track whose build failed** goes nowhere; the row counts it and the wire
  carries its reason.

A find that would change nothing -- every track it found on the bench already,
-- pushes no version and ends its row with *no effect, the bench holds them
already*; `1a` is focused all the same.

A label another item holds takes core's ` (n)` suffix, as every label on the
bench does, so a second find at the same pixel puts its new tracks beside the
first's (`frame_13@412,230 2a (2)`); a caller who wants them apart names a
group label. The version is labelled *Found 8 nearby tracks at
frame_13@412,230*. The row is an `Edit` row when the version wrote points and
a `Bench` row when it wrote only the bench, and its sentence says how many
tracks were found in how many layers, the first layer's confidence, what
became of the tracks, and which item is being edited:

> Found 8 nearby tracks in 2 layers at (412.0, 230.0) in frame_13.jpg, the
> first layer at 87% confidence: 5 committed as new points, 3 existing points
> put on the bench; editing frame_13@412,230 1a (v12 -> v13)

**`1a` is the focused item afterwards**, the track nearest the pixel on the
best-ranked layer, the best stand-in for the pixel on the surface the
photographs favour. Its point, existing or just committed, becomes the
selection after the row, as a commit selects the point it wrote, so Track View
opens on it.

**A find with nothing usable pushes no version** and writes one row saying so,
with the sources that were skipped for want of their input:

> Found no nearby tracks at (2.0, 125.0) in image_0.jpg; skipped clusters (no
> clusters), guided (no descriptors), constellation (no SIFT index)

A run core refuses (a pixel that is not a place on the photograph, inputs that
do not match the reconstruction) is one failed row in core's words.

---

## The Bench groups in the Scene tree

Each node gains two **Bench** children beside its Camera Intrinsics, Camera
Images and Points groups ([`scene-graph.md`](scene-graph.md)), one per stage:
`Bench Points (2)` for the track-stage items and `Bench Clusters (1)` for the
cluster-stage ones. The tree says which of the two an item is because the two
are different things -- a track stands at a position and a cluster is a set of
image patches with no geometry behind them -- and an item taken up or down moves
between the groups. A group with nothing in it is not drawn, so an empty bench
adds no row and a bench of tracks alone shows one group; each remembers its own
expansion.

Inside each is one row per item of that stage, in the bench's own order, by
label, with its `in` count, the focused item marked as a selected row. There is no
eye: nothing on the bench is drawn from these rows, and an item is not part of
the reconstruction.

A click on a row selects the node it is under and does nothing to the bench,
as a click on the node's own row selects it, so a pass of clicks down the tree
pushes no versions. A **double-click** focuses the item, selects the node and
raises Track View on it (`AppState::edit_bench_item_at`), with no version and
the one `Selection` row a focus writes; on the item that is focused already it
writes no row, since the gesture asked for the panel. A secondary click offers
*Discard*, and one on either group's header offers *Clear the Bench*, which
takes off every item of both groups in one version (`AppState::clear_bench`).
The bench is in the tree
because the tree is where a node's parts are listed, and it is per node because
an item names that node's images and poses.

The row reports the item by its **position** on the bench rather than by its
label, because `SceneGraphResponse` is a `Copy` value and a label is a `String`;
the step reads the label off the bench at that position. The double-click ends
in a raise, which is a layout operation, so the panel keeps it for
`SceneGraphPanel::take_bench_edit` and `app.rs` applies it once the dock is back
in the state, the path Image Detail's *Edit on Bench* takes.
The position is the item's place in the **whole** bench and not in the group it
is drawn under, so the two groups' rows reach one list.

---

## The wire

An agent gets the same bench a human does, through thirty-three MCP tools
([mcp-server.md](mcp-server.md) § "The bench family"), in
[mcp/bench.rs](../../crates/sfm-explorer/src/mcp/bench.rs). **Each one is one of
the `AppState` methods above**, which is the whole of what makes an agent's
verdict, split or commit a version in the history the human is looking at.

Every tool takes `reconstruction_label`. The item tools take `item`; the track
tools take `track`, and **a call that names no track acts on the focused
item**, resolved with `focused_item_label`, which is what a gesture in Track
View's Edited mode means when it names no item. When the focused item is on
another node's bench, or nothing is focused, such a call is refused: *"No item
on bull's bench is focused. Name one with track, focus one with
focus_bench_item, or put one on with create_bench_track or
create_bench_cluster."*

**`focus_bench_item` and `unfocus_bench_item` are not steps.** They push no
version, so they answer as the selection tools do, with what they left rather
than a version: `focus_bench_item` with the `reconstruction_label` and `item`
it focused, and `unfocus_bench_item`, which takes no argument since there is
one focused item for the viewer, with the `reconstruction_label` and `item` it
unfocused, both `null` when nothing was focused. Each carries `changed`, false
when the focused item was already what the call asked for.

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
// Each create takes an optional label, so an item needs no rename after it. A
// label another item holds takes the collision suffix: "bull-nose (2)".
// create_bench_cluster { "reconstruction_label": "bull", "camera_image": 4,
//                        "feature": 847, "label": "bull-nose" }
//
// Create Track Here: built at the pixel, put on the bench and committed.
// create_track_at_pixel { "reconstruction_label": "bull", "camera_image": 4,
//                         "pixel": [142.0, 197.5] }
//
// Find Nearby Tracks: every usable track near the pixel on the bench, the new
// ones committed, one version. "commit": false commits nothing.
// find_nearby_tracks { "reconstruction_label": "bull", "camera_image": 4,
//                      "pixel": [142.0, 197.5], "commit": false, "label": "nose" }
//
// The list.
// get_bench            { "reconstruction_label": "bull" }
// focus_bench_item     { "reconstruction_label": "bull", "item": "IMG_0042@142,198" }
// unfocus_bench_item   { }
// rename_bench_item    { "reconstruction_label": "bull", "item": "IMG_0042@142,198",
//                        "label": "bull-nose" }
// discard_bench_item   { "reconstruction_label": "bull", "item": "bull-nose" }
// duplicate_bench_item { "reconstruction_label": "bull", "item": "bull-nose" }
//
// One track on it. "track" omitted means the focused item.
// get_bench_track              { "reconstruction_label": "bull" }
// add_bench_track_observation  { "reconstruction_label": "bull", "track": "bull-nose",
//                                "camera_image": 7, "pixel": [88.5, 210.0] }
// set_bench_track_verdict      { "reconstruction_label": "bull", "observation": 3,
//                                "verdict": "in" }
//
// The patch, named for the part each tool acts on: the patch itself, one
// sighting, or a cluster sighting's parallelogram.
// translate_bench_patch      { "reconstruction_label": "bull", "observation": 3,
//                              "pixel": [1041.6, 1702.9] }
// translate_bench_patch      { "reconstruction_label": "bull",
//                              "by": [0.0, 0.0, 0.042] }   // along its normal
// resize_bench_patch         { "reconstruction_label": "bull", "observation": 3,
//                              "edge": "+u", "pixel": [1049.0, 1702.9] }
// resize_bench_patch         { "reconstruction_label": "bull", "half_length": 0.0184 }
// translate_bench_patch      { "reconstruction_label": "bull", "camera_image": 11,
//                              "pixel": [388.0, 502.5] }   // the ghost's centre
// spin_bench_patch           { "reconstruction_label": "bull", "degrees": 12.3 }
// tilt_bench_patch           { "reconstruction_label": "bull",
//                              "normal": [0.1, -0.2, 0.97] }
// sight_bench_observation    { "reconstruction_label": "bull", "observation": 3,
//                              "pixel": [1041.6, 1702.9] }
//
// The cluster stage's own, over one sighting's parallelogram.
// resize_bench_shape         { "reconstruction_label": "bull", "observation": 3,
//                              "edge": "+u", "pixel": [1049.0, 1702.9] }
// spin_bench_shape           { "reconstruction_label": "bull", "observation": 3,
//                              "degrees": 12.3 }
// shape_bench_observation    { "reconstruction_label": "bull", "observation": 3,
//                              "shape": [[7.1, -0.4], [0.4, 7.1]] }
// apply_bench_track_thresholds { "reconstruction_label": "bull", "min_zncc": 0.8 }
// fit_bench_track           { "reconstruction_label": "bull" }
// fit_bench_track_normal    { "reconstruction_label": "bull", "method": "photometric" }
// fit_bench_track_normal    { "reconstruction_label": "bull",
//                             "method": "finite_difference", "pieces": 3,
//                             "overlap_percent": 25 }
// fit_bench_track_normal    { "reconstruction_label": "bull",
//                             "method": "grid_plane", "pieces": 3 }
// set_bench_track_stage        { "reconstruction_label": "bull", "stage": "track" }
// select_bench_observations    { "reconstruction_label": "bull", "observations": [3, 5, 8] }
// split_bench_track            { "reconstruction_label": "bull", "observations": [3, 5, 8] }
// commit_bench_track           { "reconstruction_label": "bull" }
//
// The index files, which are the node's rather than any track's, and the
// search through its SIFT index, which is one observation's.
// open_index_files               { "reconstruction_label": "bull" }
// build_index_files              { "reconstruction_label": "bull" }
// close_index_files              { "reconstruction_label": "bull" }
// search_bench_track_descriptors  { "reconstruction_label": "bull", "observation": 0 }
//
// The other search, which asks the reconstruction's geometry rather than an
// index, and so needs no `.kdf`: track stage only.
// search_bench_track_geometry     { "reconstruction_label": "bull", "observation": 0 }
```

**The two reads have no panel gesture behind them**, because a panel shows what
they answer. `get_bench` is the bench as JSON: each item's label, kind, stage,
origin and counts, each with `focused`, and `focused_item`: the focused item's
label when it is on this node's bench, and `null` otherwise, which a bench
holding items can be. It is **one flat list** in the
bench's own order, with the stage on each item, rather than the tree's two
groups: a reader that wants them apart has the field to do it with, and a
grouping on the wire would be the panel's layout rather than the bench's own
state. Both carry each track's `evaluation`: its `state` (`current`, `evaluating`,
`refused` or `failed`, § "Live evaluation"), the `reason` sentence where it is
refused or failed, `running` for whether an evaluation of the current inputs is
on a worker now rather than waiting to start.
An agent that has just made a step reads `evaluating` and the previous numbers,
and reads again until it says `current`. `get_bench_track` is
Track View's Edited-mode table: the stage and its data, the origin, the thresholds, and
every observation with its provenance, verdict, `pixel` and both stages'
measurements where they exist -- at both stages `zncc_middle` and `zncc_grid`
beside `zncc` (the same samples read over the middle square of the patch and
over each ninth of it, § "The middle ZNCC" and § "The ZNCC grid" of
[`../core/bench/editable-track.md`](../core/bench/editable-track.md)), and the ZNCC
self-similarity radius `zncc_self_similarity_radius` with its `_middle` and
`_grid`, `zncc_self_similarity_surface`, `zncc_self_similarity_tolerance`,
`zncc_self_similarity_ellipse` with its `_middle`, the ellipse whose semi-major
axis is the radius, in grid px, image px and along the patch, and
`zncc_self_similarity_ellipse_grid`, each ninth's in grid px
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "The
ZNCC self-similarity radius"), at the track stage
the two distances
(`seed_shift_px` and `projection_offset_px`), `walked_px`, `walked_to`, `walked_zncc`, `walked_zncc_middle` and `walked_zncc_grid` for a row the last fit
refused to move (`sight_bench_observation` at `walked_to` accepts that walk), and, for a row the evaluation could
not score, the `reason` sentence in place of a ZNCC. Each row also carries
`patch_zoom`, the zoom Track View's *Zoom* column prints
([`track-view.md`](track-view.md) § "The observation table"), as `[least, most]`
patch-grid px per photograph pixel, and `patch_jacobian`, the Jacobian it is
read from: the Jacobian of the warp from the patch grid to the photograph, at
the patch's centre, as `[[dx/dcol, dx/drow], [dy/dcol, dy/drow]]` in photograph
pixels per patch-grid px, core's `camera::warp_map::patch_grid_jacobian` of the
patch re-anchored where the row sits, the placement its tile is rendered
through. Both are at the reconstruction's patch resolution `R`, which the track
stage's data reports as `patch_resolution`: the edge of its patch bitmaps, which
an `.sfmr` declares as `patch_bitmap_resolution`, and where it stores none the
evaluation's own 24 (core's `EvaluateOptions::patch_resolution`). The track
stage's shift and self-similarity are read on the same grid
([`../core/bench/editable-track.md`](../core/bench/editable-track.md)), so the
zoom, the shift and the self-similarity radius are in one unit; the 64 texels Track
View draws a tile at do not enter any of them. The table does not print the
Jacobian; it is reported as a diagnostic.
Both are geometry alone, read from the patch, the camera, the pose and where the
observation sits, so neither waits for a photograph, and a tile whose middle is
off the photograph has both. Both are null at the cluster stage, on a track with
no patch yet, for an observation with nothing saying where it sits, and for a
patch whose centre is behind the camera or outside the camera model's domain;
`patch_zoom` is null as well for a patch seen edge on. The track stage's own data
carries `at_infinity` with the coordinate under `direction` or `position`, the
other null, for the reason Track View's edit header carries a word in front of it:
the same three numbers are a place or a bearing depending on `w`, and an agent
that read `position` off a `w = 0` track would be holding a place one unit from
the world origin. It carries the patch too, as `placement` -- centre, unit
axes, outward `normal` and `half_extent`, or `null` before one is fitted -- in
the block `get_point` reports a committed point's patch in, so an agent can
read the normal it would `tilt_bench_patch` from and compare it with the point
the track was committed over. It also carries `selected_observations`, the rows selected in Track View
(§ "The selected observations"), which `select_bench_observations` replaces:
an agent reads the rows a person picked out, and picks out rows for a person to
look at. The list is empty on a track that is not focused. **An observation is
addressed by its position in
that list**, which no bench step renumbers, so an index an agent is holding after
a verdict or a fit still names the same observation. `delete_camera_image` is
the one call that renumbers it (§ "One history for the pair"), and its
description says so. The template's
samples and the consensus bitmap are reported as present or absent rather than
sent: they are pictures, and that surface is not a data channel.

**`pixel` is where the observation sits, whatever said so**: the keypoint the
track stage carries, else the refined cluster position, else the seed the step that
proposed it left. It is a field of its own rather than something a caller
assembles out of the two measurement blocks, because that one answer is what
every reader of a sighting wants, and a cluster's observations carry a seed and
no keypoint while a track's carry a keypoint -- an agent would otherwise have to
know which slot to fall back to before it could look at one. It is `crate::bench::observation_site`'s
rule, so the number an agent reads here is the pixel the Image Detail panel
marks, the place the Track View row click reveals, the centre of the tile that
row draws and what `set_image_detail_view`'s `bench_observation` aims. `null`
only for an observation nothing says the place of, which is the state core's
`Unmeasured::NoSeed` names.

**The patch tools are the panels' handles**, and each is one
`edit_bench_patch`, so a drag and a tool call are the same version carrying the
same sentence. Each is **named for the part it acts on** -- the patch, one
sighting, or a cluster sighting's parallelogram -- and **four verbs carry them,
no two of them synonyms**: `translate` moves the centre, `resize` changes the
half-length, `spin` turns the square about its normal and `tilt` turns the
normal itself.

`translate_bench_patch` and `resize_bench_patch` each take **exactly one of two
ways** to say what they want. A translation takes `by`, a displacement
`[u, v, n]` on the patch's own orthonormal axes in world units, or an
`pixel` with the photograph it is in, which its centre lands under. A resize
takes a world `half_length` with an optional `moved_edge`, or an `edge` and a
`pixel` with the photograph it is in. The pixel forms are the gestures, and they
are what make the answer exact: the pixel is unprojected onto the patch's own
plane, so the edge lands there through whatever distortion the lens has and the
opposite edge is left where it was. **The photograph is exactly one of
`observation` and `camera_image`**, and the two name different squares, which
is why a call carrying both is refused. An `observation` names that sighting's
outline, the patch re-anchored on its keypoint. A `camera_image`, by index or by
name, names the patch as it stands seen in that image, which is Image Detail's
ghost outline, and it is the only form that reaches an image the track has no
sighting in. The reply carries whichever it was. `spin_bench_patch` turns the
patch about its own normal and names no sighting, there being one patch.

**The cluster stage's own are tools of their own.** There is no shared geometry
there, only one affine shape per sighting, so `resize_bench_shape` and
`spin_bench_shape` take an `observation` and do that image's pixel arithmetic,
and `shape_bench_observation` states the whole 2x2 `shape` outright. Each of
them refuses a track-stage track and names the tool that belongs to it, and
`resize_bench_patch` and `spin_bench_patch` refuse a cluster the same way. Two
names rather than one with an optional observation, because a tool named for
the part it acts on cannot act on two different parts.

**The normal part of a `by`, and `tilt_bench_patch`, name no pixel**, because
no sighting can say what they say. The `n` of a displacement moves the patch
that many world units along its own outward normal, positive toward the face
the patch shows: a sighting names the ray the patch lies along and not how far
down it the surface is, so this is where a patch's depth is settled. A **mixed**
`by` that moves the patch across its plane and along its normal at once is
allowed. The tilt turns the patch to face the outward `normal` named, by the
least rotation and so with no spin about the normal, and stops 80 degrees from
any observation's camera -- where that photograph would be looking along the
surface rather than at it -- the sentence naming the image that stopped it: a
sighting says nothing about which way the surface under it faces either. Each is
settled by the tool or by the drag it shares a step with, the normal's segment
and the arrowhead at the end of it, in the 3D viewer or in Image Detail, where
the drag is read through the photograph's own camera. A track at infinity refuses a `by`
with a normal part and a tilt, a direction patch's normal being its own bearing,
and carries a purely tangential `by` like any other.
`sight_bench_observation` is the one that moves a single sighting: the cluster
stage's dot, the track stage's dot with Track View's *Lock* cleared, and a
script that means one keypoint. The lock is the panel's setting and the wire
carries no copy of it, because a tool call already says which of the two it
means by being one tool or the other.

**Every step answers as an edit answers**, with the version it pushed and the
sentence the Action Log recorded, plus the `item` it acted on. A create and a
split name what they made, a rename names the label the item now holds, and an
added observation names the index it took.

**The sentence is the step's own.** A step can set something else off -- putting
the first item on a bench looks for the node's SIFT index, and
finding one writes a row of its own, after the step's. The reply reads the log
back for what the call did, so a row nobody asked for would arrive under the
step's label. Those rows are the **viewer's** (`Actor::Viewer`), which is what
they are, and the reply skips them. It skips `Selection` rows for the same
reason: a commit selects the point it wrote, which is where the call left the
viewer looking rather than what the call did. **A commit names the point it
wrote** -- `{ "point": { "index": 4211, "id": "pt3d_95fe75db_0", "replaced":
1207 } }`, and a commit that wrote nothing names the point that already holds
the track the same way, with `replaced` null -- because one row of the
reconstruction is the whole of what a commit
produces, and neither index is derivable from the sentence: a commit that
replaces writes a **new** row and deletes the one it replaced, so `index` and
`replaced` are two different numbers, and one that creates takes whatever index
the overlay had free. So the next call is a `get_point` rather than a search
through the counts for whichever row is new. So `undo`, `redo` and
`jump_to_version` need no bench variant: the history they walk already holds the
bench steps.

**The three index-files tools push no version.** The index files sit beside
the node's `.sfmr`, so `open_index_files`, `build_index_files` and
`close_index_files` change neither the reconstruction nor the bench, and their
reply is the `index_files` object -- per file its path, which of the three
states it is in, its counts, and the sentence saying why it is stale -- rather
than a version. `get_bench` reports the same object under `index_files`,
carrying the node's own paths even when nothing is open, so an agent can see
where a build would put them; those paths are spelled in one convention, the
platform's own. A search against an index that is not `current` is refused
with that index's own sentence ([`sift-index.md`](sift-index.md),
[`index-files.md`](index-files.md)). The search itself is an ordinary bench step
and answers as one.

**The steps that read a file answer in two levels**, as the bundle
adjustment does: with the version they pushed when they finish inside the reply
window, and with `running: true` and an `operation_id` to poll
`get_background_task` with when they do not. The deferral is taken before a
single photograph has been read (§ "The two steps that read photographs"), so
the window is measured against the operation rather than spent on the decode in
front of it. A step that finds nothing to do starts no task and answers with the
version the node stands at, and a step the **track** rules out starts no task
either: it is a tool error in the step's own sentence, arriving in the call
rather than through a task the agent would have had to poll to learn that
nothing was ever going to happen.

**A step that had no effect pushes nothing either, and says so.** A refusal and
a no-effect are different answers and the surface keeps them apart: a refusal is
a tool error, and a step that was allowed to run and found there was nothing to
do answers successfully. The contract is one for every step:

- **No version.** The history is not moved, so there is no row to undo for
  nothing.
- **One Action Log row**, of kind `Bench` and not marked failed, carrying the
  step's own no-effect sentence -- *"Moved bull-nose: no effect, the patch
  already sits there"*, *"Set bull-nose to the track stage: no effect, it is at
  that stage already"*. The row has no `(v3 -> v4)` because there is no
  transition to name.
- **A reply that agrees with it.** `changed` is `false`, `serial` and `cursor`
  are the version the node still stands at, `label` is that version's label, and
  `report` is the row just written -- the step's **own** sentence, never the
  previous step's label, which is what a reply assembled from the cursor alone
  echoes.

`changed` is on **every** edit reply, not only the bench's: it is read off the
cursor before and after the call, so "did the history move" has one answer in one
place for every family.

**Whether a step had an effect is core's to decide, with a tolerance.** A pixel
named under a pointer or on the wire is turned into a ray, met with the patch's
own plane and projected back, and that round trip returns the place it started
from to within the arithmetic's last bits rather than bit for bit. An exact
comparison therefore reads every re-statement of where the patch already is as a
move, and the version it pushes says the patch moved by zero. So each step judges
in the units of the value it
moves: a centre within a millionth of the patch's own half-length, a half-length
within a millionth of itself, an affine coefficient within a millionth of the
shape's largest, a sighting within a thousandth of a pixel, a turn within a
nanoradian. The steps that compare something that is not a float -- a verdict, a
stage, a label -- compare it exactly, because there is
nothing to round.

**The commit is the one float comparison that is exact**, and for the same
reason: what it asks is not whether a gesture moved anything but whether writing
the record would leave the point's own columns as they stand, and a coordinate
that differs by a stored amount is a point that moved. So it compares the record
it would write against the one the origin holds, column for column, `NaN`
agreeing with `NaN`
([`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "The commit"), and answers `changed: false` when the two say the same thing.

**A pixel off the photograph is brought inside it rather than refused.** A
pointer can be dragged past the edge of the picture and a call can carry any two
finite numbers, and neither names a place on the photograph: `[-500, -500]` of a
480 px frame is no column and no row, and a patch whose centre is slid until it
meets that pixel's ray lands wherever the extrapolated ray happens to cross its
plane, which is an arbitrary distance from where it stood. Every step
that takes a pixel as a **gesture** -- `translate_bench_patch`,
`sight_bench_observation`, `resize_bench_patch`, `resize_bench_shape`,
`add_bench_track_observation`'s seed and `create_bench_cluster`'s pixel -- takes
the nearest pixel of `[0, width) x [0, height)` instead, and says that it did:
the reply carries `clamped: true`, `clamped_from` and the `pixel` it used, and
the Action Log row and the version label end *"clamped to the photograph from
(-500.0, -500.0) to (0.0, 0.0)"*. The consequence worth knowing is that a
resize's reach is the photograph: an edge cannot be dragged to a place the
sighting does not show, and a patch grown past the frame is grown in a photograph
that holds it. `sfmtool_core::bench::clamp_to_photograph` is the one rule, so the
panel's drag and the tool call land on the same pixel. A seed a **search** places
is not clamped -- a warp may put the patch off the edge of an image that only
half holds it, and saying so is the reading's job.

**A refusal is the step's own sentence and pushes nothing.** The wire wraps
nothing: what an agent reads is the sentence the panel's status line would show.

---

## Testing

[bench/tests.rs](../../crates/sfm-explorer/src/bench/tests.rs), headless, over
the demo reconstruction rewritten as `embedded_patches` with every keypoint its
point's exact projection and a photograph cached for every image:

- putting a second item on and discarding it are two versions and focusing the
  first between them none, an undo of the discard puts it back and leaves the
  focused item alone, and the first item is the same `Arc` throughout;
- putting a point on the bench twice focuses the track it already made, with no
  version;
- focusing and unfocusing push no version and write one `Selection` row each,
  and the undo after them takes back the last step; focusing the focused item
  and unfocusing nothing write no-effect rows; focusing a label not on the
  bench is refused in a sentence;
- every step that puts an item on focuses it: a put, a cluster, a duplicate
  and a split;
- discarding the focused item unfocuses it, and its undo does not focus it
  again;
- a rename keeps the item focused and its selected observations, and so does
  the undo of the rename;
- the selected observations clear when the focused item changes, and a
  selection on an item not focused is refused;
- an undo after an unfocus takes back the verdict before it, and an undo past
  the put unfocuses, the redo not focusing it again;
- there is one focused item for the viewer: a put on a second node unfocuses
  the first node's item, focusing the first's unfocuses the second's, and
  closing the node unfocuses;
- a busy node refuses neither a focus nor an unfocus, nor a tick of *Edit* over
  a point whose item is on the bench;
- the two gestures the Image Detail context menu carries -- a cluster started at
  a pixel, then a sighting added at one in the same image -- are one version and
  one `Bench` row each, the second sighting joining unpinned and `out`, and an
  undo walks them back one at a time;
- a verdict and a stage change are two versions and the evaluations after them
  none, undo retraces them in order and redo replays them;
- a document edit between two bench steps is a version in its place, and undoing
  it leaves the bench alone;
- a commit replaces its origin point, the selection lands on what it wrote from
  wherever it was standing, and an undo restores the pair;
- a commit that creates a point selects what it created, and an undo of that
  clears the selection;
- a commit's row is an `Edit` and every other bench step's is a `Bench`, one row
  per step, with the selection's row after the commit's;
- a step that changes nothing pushes no version, the commit onto a point that
  already holds the track included: repeated presses push no version, mint no
  index, keep the point selected and write one no-effect row each, while the
  press after a sighting is turned out or after an undo writes again;
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
  no `.sift` files is refused the same way;
- a fit's **version label** names the item and then says which representation
  the rays earned and on what test, while its **Action Log row** carries the
  whole report, counts and all: the two are different lengths on purpose, and a
  label that stopped at the item would hide the step's real outcome.

The live evaluation is tested in
[bench/live/tests.rs](../../crates/sfm-explorer/src/bench/live/tests.rs), over
the same fixture: a track put on the bench reads `Evaluating`, the next drive
starts its evaluation, and once it lands the track is `Current` and measured,
with no version pushed and no row written; a verdict makes it `Evaluating` again
and the next drive evaluates it; an answer for inputs a step has moved on from
is not installed, the drive after the step cancels it without starting a second
evaluation beside it, and the track is `Current` only once the evaluation of the
new inputs lands; an undo, a document edit under the track and a change of the
search radius each make it `Evaluating`; a track-stage track with no frame is
`Refused` with core's sentence and starts nothing; nothing starts while a
background task holds the node; the focused item is evaluated first when it is
on a node after another in the scene; and a rename keeps an evaluated track
current.

Deleting an image under the bench is tested over the wire in
[mcp/tests/bench.rs](../../crates/sfm-explorer/src/mcp/tests/bench.rs): a track
put on the bench from a point in images 0, 1 and 2, with rows 1 and 2
selected, keeps after `delete_camera_image` of image 1 the observations in the
photographs that were images 0 and 2, now at indexes 0 and 1, with row 1 alone
selected; committing it writes a point whose observations are those two
photographs, each reprojecting within a hundredth of a pixel; the delete's
label names the dropped observation; and two undos bring back the track's
three photographs. A cluster started in image 1 alone is discarded by the same
delete, in its one version, unfocused and named in the label, and an undo puts
it back.

The unfocus is tested where its two callers are: [track_view/tests.rs](../../crates/sfm-explorer/src/track_view/tests.rs) clears the *Edit* box and finds no version
and one `Selection` row with the item still on the bench, a focus bringing Edited mode back, and a
discard of the focused item leaving Edited mode; [mcp/tests/bench.rs](../../crates/sfm-explorer/src/mcp/tests/bench.rs) calls
`unfocus_bench_item` twice, no version and then a no-effect reply, with `get_bench`
reporting a `null` `focused_item` in between, and `focus_bench_item` pushes no
version while a task holds the node, `get_bench` reporting the item on its node
and `null` on another, and the old tool names are unknown tools.

The handles are tested in
[image_detail/tests.rs](../../crates/sfm-explorer/src/image_detail/tests.rs),
which drives real frames: a press on an edge followed by two panel pixels of
motion -- under egui's own drag threshold -- and then thirty resizes the patch
and **pans nothing**, which is what says the press and not the drag decides the
handle, while the same motion from a press on empty photograph pans as it always
did; hovering an edge asks for the resize cursor its orientation on screen
names, a corner for the resize cursor along the arc it spins on, and a dot for
`Move`; a drag of the dot publishes a move that `AppState` turns into exactly
one version whose label names it; with *Lock* cleared the same drag publishes a
`Sight` of that one observation, whose version moves its keypoint, pins it and
leaves the patch and every other sighting as they were, while a press on an edge
or a corner pans and edits nothing (and the cluster stage's dot and corner are
that sighting's own either way); the segment the held dot draws to the patch's
projection is none when locked and the length of the drag when not, which is
what each release then leaves; a drag of an edge resizes so that the outline's dragged edge
reprojects under the release point while the far edge holds; a drag of a corner
onto its neighbour is a quarter turn and one version; Escape leaves no edit
behind; and a drag that ends where it started pushes no version. The panel is
zoomed in for them, because the demo's patch is under three source pixels
across and every handle would otherwise sit inside every other one's reach.

The steps themselves are core's and are tested there, over a synthetic textured
plane whose numbers are known to the pixel.

*Create Track Here* is tested in
[bench/track_at_pixel/tests.rs](../../crates/sfm-explorer/src/bench/track_at_pixel/tests.rs),
over that plane rebuilt in the viewer (core's is private to its own tests):
pinhole cameras over a textured plane, a grid of points every camera sees at
its exact projection, less the middle one, and a bitmap column. At the middle's
pixel the run is two versions, a `Bench` row then the commit's `Edit` row and a
`Selection`; it creates one point, the transfer member builds it (the demo node
has no index files), the item is focused and seated on the point, and one undo
takes the point back and leaves the track on the bench. At a pixel further from
every point than any member looks, nothing is pushed, one failed row gives the
last member's stage and reason and names both missing index files, and its
detail lines carry each member's refusal. The greyed states are the unposed
image, the `sift_files` node and the busy node, and a pixel off the photograph
is refused with no task. The panel's side is in
[image_detail/tests.rs](../../crates/sfm-explorer/src/image_detail/tests.rs):
the entry is drawn first, directly above *Edit on Bench*, with its shortcut
beside it and in the same place when greyed; a Control+Shift click on a feature
asks for a track at the pixel clicked, not the feature's, and selects nothing,
while the same click without the modifiers selects the point and Control or
Shift alone asks for nothing; and a Control+Shift double-click is neither *Edit
on Bench* nor a zoom. The background tests cancel the operation over the same
plane, as they do every operation that says it is cancellable.

*Find Nearby Tracks* is tested in
[bench/nearby_tracks/tests.rs](../../crates/sfm-explorer/src/bench/nearby_tracks/tests.rs),
over the same plane and over the same capture of a plane ten million units out,
whose one point sits beyond the points source's reach so that only the
far-field sweep finds anything. At the held-out point's pixel of the near plane
the eight points around it land as their own tracks, labelled `1a` to `1h` with
their points, seated on them, nothing committed, one `Bench` version, `1a`
focused and its point selected; a second find there puts no copies on and
pushes no version. On the far plane the sweep's reading is built and committed as a new point in the same
version, whose row is an `Edit`, and one undo takes back the point and the
bench items together, one redo puts both back; without `commit` the track goes
on the bench under the caller's group label and nothing is written, and a
second find takes the ` (2)` suffix. At a pixel far from every point on the
near plane nothing usable is found: one row, no version, the bench untouched.
The refusals are *Create Track Here*'s, and a `sift_files` node refuses only a
find that commits. The panel's side, in
[image_detail/tests.rs](../../crates/sfm-explorer/src/image_detail/tests.rs),
draws the entry third, directly below *Edit on Bench*, and greys it with
*Create Track Here*'s sentences in the same place. The background tests cancel
it over the near plane.

The view-only bench is tested over the wire in
[mcp/tests/bench.rs](../../crates/sfm-explorer/src/mcp/tests/bench.rs), on a
`sift_files` node with `.sift` files beside it: a point goes on the bench, every
editing tool is refused with the view-only sentence and pushes nothing, the
conversion runs, and putting the point on the bench again rebuilds the item
with a frame in one version, after which it takes a verdict. Image Detail's
greyed entries are tested in
[image_detail/tests.rs](../../crates/sfm-explorer/src/image_detail/tests.rs).

The wire is tested in
[mcp/tests.rs](../../crates/sfm-explorer/src/mcp/tests.rs), over the same
fixture with a label on the node, for what the boundary owes: each tool being
the `AppState` call the panel makes, the reads reporting `evaluation.state` as
`evaluating` until the evaluation lands and `current` after it,
a move of the `max_shift_px` bar, which is the search radius, pushing one
version and making the track `evaluating`, an observation index surviving the
steps
that follow it, a refusal arriving as the step's own sentence, and the two
photometric steps deferring to a worker and landing their version
([mcp-server.md](mcp-server.md) § "Testing"). Two of its cases are about what a
reply says rather than what a step does: a create on a node whose default index
is there but not yet open reports the **create's** sentence and not the open's,
and an observation added at a pixel a long way from the point's projection comes
back from its evaluation with the reason on its row, the rows that could be read
still read. `create_track_at_pixel` is tested over the viewer's plane: it answers
with the commit's version, the item, the member and the point, which `get_point`
takes back by id; a pixel every member refuses is a tool error whose lines are
each member's stage in the order tried; and a pixel off the photograph is
refused in the call with no task.

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
- **Drawing the bench in the Image Browser.** The thumbnail borders that would
  mark the focused item's `in` observations are proposed in the same draft. Both
  of the panels that do draw the focused item draw it as handles: the Image
  Detail panel's bench layer marks it in each photograph that observes it, where
  a mark places a sighting and sizes, turns, moves along its normal and tilts
  the patch, and at the track stage draws the patch as a ghost outline in each
  photograph that does not, which takes the same patch-wide edits while Track
  View's *Lock* is ticked ([`multi-panel-image-browser.md`](multi-panel-image-browser.md)
  § "The bench layer"); the 3D viewer draws the patch where it stands in the
  world, with the same handles read through its own camera
  ([`viewer-3d-bench-layer.md`](viewer-3d-bench-layer.md)).
- **Wire tools for the searches.** The three tools that would drive a descriptor
  search, a view sweep and a pull-in wait on the core steps behind them, and are
  proposed in the same draft. The thirty-three tools for the steps that exist
  are § "The wire".
