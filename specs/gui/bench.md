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

Related specs: [`track-view.md`](track-view.md) (the panel that edits the active
track with its *Edit* box ticked), [`multi-panel-image-browser.md`](multi-panel-image-browser.md) (the Image
Detail panel, which carries the three steps that name a pixel and draws the active
track as its bench layer), [`edits/commit-track.md`](edits/commit-track.md) (the one step that writes
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
    /// View's threshold slider released, or `apply_bench_track_thresholds`.
    pub(crate) fn apply_bench_thresholds(&mut self, id: ReconId, label: &str,
                                         thresholds: &Thresholds) -> Result<(), String>;
    /// Track View's *Accept walk*: put a sighting the last fit kept at its seed
    /// at its `walked_to`, through `sight_observation`. One version.
    pub(crate) fn accept_bench_walk(&mut self, id: ReconId, label: &str,
                                    observation: usize) -> Result<(), String>;
    pub(crate) fn split_bench_track(&mut self, id: ReconId, label: &str,
                                    observations: &[usize]) -> Result<String, String>;
    pub(crate) fn activate_bench_item(&mut self, id: ReconId, label: &str) -> Result<(), String>;
    /// Every item left on the bench and none active: Track View's Edit box
    /// cleared. No version, and a no-effect row, with nothing active.
    pub(crate) fn deactivate_bench_item(&mut self, id: ReconId) -> Result<(), String>;
    /// The Edit box ticked (the selected point put on, or its item activated)
    /// or cleared (`deactivate_bench_item`).
    pub(crate) fn set_editing(&mut self, id: ReconId, on: bool) -> Result<(), String>;
    /// A Scene tree double-click on a Bench row: select the node, activate the
    /// item unless it is active already, raise Track View.
    pub(crate) fn edit_bench_item_at(&mut self, id: ReconId, position: usize);
    /// Image Detail's *Start cluster on the bench here*: the cluster put on,
    /// active, and Track View raised.
    pub(crate) fn start_cluster_here(&mut self, image: ImageRef, pixel: [f32; 2]);
    /// Image Detail's *Create Track Here*: a track built at the pixel on a
    /// worker, then put on the bench, active, and committed. A refusal in
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
    /// Discarding the active item leaves nothing active.
    pub(crate) fn discard_bench_item(&mut self, id: ReconId, label: &str) -> Result<(), String>;
    /// A copy of the item beside it, active, with no origin: the label it took.
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
    /// `search_px` is how far around each observation the correlation peak is
    /// looked for; `None` is the radius the live evaluation reads at.
    pub(crate) fn start_bench_fit(&mut self, id: ReconId, label: &str,
                                  search_px: Option<f64>) -> Result<(), String>;
    pub(crate) fn start_bench_stage(&mut self, id: ReconId, label: &str, stage: StageKind)
        -> Result<(), String>;
    /// Ask the node's SIFT index which other photographs hold the patch
    /// around one observation, and add each as a candidate.
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
    /// The radius every evaluation reads at, and its setter: Track View's
    /// *search px* slider and the wire's `set_bench_search_px`.
    pub(crate) fn bench_search_px(&self) -> f64;
    pub(crate) fn set_bench_search_px(&mut self, search_px: f64) -> Result<(), String>;
    /// Once per frame: land a finished evaluation, cancel one whose inputs
    /// have moved on, start the next. True when the next frame has to draw.
    pub(crate) fn drive_bench_evaluation(&mut self) -> bool;
}

// The selected observations, held in `AppState::bench_rows`
// (§ "The selected observations").
pub(crate) struct BenchRows {
    pub(crate) recon: ReconId,
    pub(crate) label: String,
    /// Ascending, without repeats.
    pub(crate) observations: Vec<usize>,
}

impl AppState {
    /// Empty unless `label` is the active track and the selection was made on it.
    pub(crate) fn selected_bench_observations(&self, id: ReconId, label: &str) -> &[usize];
    /// The one selected observation, when exactly one is: what the 3D viewer's
    /// bench figure draws larger.
    pub(crate) fn selected_bench_observation(&self, id: ReconId, label: &str)
        -> Option<usize>;
    /// Replace the selection; an empty list clears it. Refused for a track that
    /// is not active and for an index past the end of the list.
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
means "the active item of the kind this panel edits", and the panel is what
knows that; the steps take the label, so the question of which item is answered
in one place rather than inside each step. `active_track_label` is what a caller
resolves it with.

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
active track's square in the old frame and the 3D viewer's figure would jump.
The centre goes through the whole similarity, the axes are rotated and the
half-extent is scaled; a track at infinity keeps the rotation alone; a
cluster-stage item has no world geometry and keeps its `Arc`.

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
| Create a track at a pixel (the put; its commit is the commit's row) | `Created IMG_0042@142,198 at (142.0, 198.0) in IMG_0042.jpg with the clusters member` |
| Add an observation | `Added image_012.jpg to pt3d_a1b2c3d4_1207` |
| A verdict | `Turned image_012.jpg out of pt3d_a1b2c3d4_1207` |
| Slide the patch | `Moved pt3d_a1b2c3d4_1207 by 0.123 units to (1.204, -0.318, 4.006)` |
| Place one sighting | `Moved observation 3 of pt3d_a1b2c3d4_1207 to (1041.6, 1702.9) in IMG_0042.jpg (2.3 px)` |
| Resize the patch | `Resized pt3d_a1b2c3d4_1207 to 7.4 px in IMG_0042.jpg` |
| Turn the patch | `Rotated pt3d_a1b2c3d4_1207 by 12.3 degrees` |
| Turn one sighting's shape | `Rotated observation 3 of IMG_0042@142,198 by 12.3 degrees` |
| Apply the thresholds | `Applied the thresholds to IMG_0042@142,198: 3 in, 1 out, 1 pinned, 0 unmeasured` |
| Accept a walk | `Accepted the walk of observation 3 of pt3d_a1b2c3d4_1207: moved 11.2 px to (1050.8, 1702.4) in IMG_0042.jpg` |
| Fit | `Fitted IMG_0042@142,198: finite at (x, y, z): condition number 82 under the 10000 bar, rms 0.1 px finite against 48.3 px as a bearing, rays up to 15.204 deg apart` |
| Set the stage | `Set IMG_0042@142,198 to the track stage` |
| Split | `Split 2 observations off pt3d_a1b2c3d4_1207 as pt3d_a1b2c3d4_1207-split` |
| Activate | `Made IMG_0042@142,198 the active track` |
| Discard | `Discarded IMG_0042@142,198 from the bench` |
| Duplicate | `Duplicated IMG_0042@142,198 as IMG_0042@142,198 copy` |
| Rename | `Renamed IMG_0042@142,198 to bull-nose on the bench` |

The Action Log row is that sentence plus the version serials, exactly as an
edit's is: `Turned image_012.jpg out of pt3d_a1b2c3d4_1207 (v7 → v8)`. A
refusal is one failed row carrying the refusal's sentence.

**A step that changes nothing pushes no version.** Setting the verdict an
observation already has, setting the stage a track is already at, activating the
active item, renaming an item to the label it holds, and a drag of a handle that
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
IMG_0042@142,198: at infinity along (0.553, -0.809, -0.198): finite point would
have 15.1 px rms against the bearing's 5.2 px rms` -- rather than naming the item and
stopping there. A person scrolling the history can therefore see which fit
crossed the boundary, and on what evidence, without opening each version and
re-reading its coordinate. The **Action Log row** is the whole report, which is
that sentence with the counts around it: how many sightings the kernels placed
and how many the walk bound left at their seeds.

### Labels

A label is minted from what the item was made from and is how the item is named
everywhere -- in the Scene tree, in Track View's header, in every row and
version label. The minting is core's ([`../core/bench/bench.md`](../core/bench/bench.md)
§ "Labels") with one exception the viewer supplies: **a track put on the bench
from a point is labelled by that point's portable id**, `pt3d_a1b2c3d4_1207`,
because that id names the content the point is a row of and the version graph
that content sits in, and core has neither ([`goto-point.md`](goto-point.md)).

**Putting a point on the bench twice activates the track it already made.** The
person asked to work on that point, and there it is; a second item for one point
would be two answers to one question. The test is the origin, followed to the
cursor.

**The bench holds one active item, and may hold none.** A track and a cluster
share the one activation, so activating one leaves the other on the bench,
inactive. Clearing Track View's *Edit* box is `deactivate_bench_item`, one
version labelled `Stopped editing IMG_0042@142,198; it stays on the bench`, and
with nothing active it pushes nothing and writes the no-effect row `Stopped
editing: no effect, no track is active`. Discarding the active item leaves
nothing active rather than handing the activation to a neighbour, so Track View
returns to view mode instead of switching to an item nobody asked for. An undo
over any of these restores the activation the version held.

---

## The selected observations

The bench carries a selection of the active track's observations: the rows
highlighted in Track View's edit mode. *Split off N rows* takes them, and when
exactly one is selected the 3D viewer's bench figure draws its mark larger
([`viewer-3d-bench-layer.md`](viewer-3d-bench-layer.md)). It is held in
`AppState::bench_rows` as a `BenchRows`, which names the node and the track's
label beside the observation indexes, so every gesture that reads or sets it
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

**Only the active track has selected observations.** The indexes mean
something only against one track's list, and Track View shows only the active
track, so a selection on any other track is refused. What clears it:

- **a move of the cursor**: undo, redo or a jump. The version landed on may
  hold another list of observations under the same label, and an undo does not
  bring back a selection, because a selection is not in a version;
- **a step that leaves another item active, or none**: an activation, a
  deactivation, a discard, a duplicate, a create or a commit that activates
  something else. Coming back to the track later does not bring the selection
  back;
- **a split**, which renumbers the observations left on the track;
- **closing the node**.

A rename carries the selection to the new label, since the observations are the
same ones. Every other step on the active track keeps it, because those steps
append observations or change them in place and never renumber the list.

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
| The search radius | Track View's *search px* slider, the wire's `set_bench_search_px` | `AppState::bench_search_px` moves |

So no step has to remember to ask for an evaluation, and none does. The track's
label is part of the key too, so a renamed item is evaluated under its new
name. The thresholds are not an input on their own: the evaluation reads with
its gates off, and the bars reach it only through the verdicts they paint.

**One evaluation runs at a time, and a stale one is cancelled.** Once per frame,
after the frame's steps, `drive_bench_evaluation` lands a finished evaluation,
and then either cancels the running one when its inputs are no longer the
track's, or, when nothing is running, starts the next track whose evaluation is
`Evaluating`: the active track of each bench first, then the rest in bench
order, in scene order. It does not start the next until a cancelled one has
reported back, so a slider drag that moves an input on every frame has at most
one worker behind it rather than a queue. It also waits while a background task
holds the node, since that task's answer replaces the inputs it would read.

**An answer is installed only if its inputs are still the track's.** An
evaluation that lands after a step has moved the track on -- cancelled, or
measured before it saw the flag -- is dropped, and the track reads `Evaluating`
until the evaluation of its new inputs lands. A result that matches is written
into the version at the cursor in place (`History::replace_current_bench`):
**no version is pushed and no Action Log row is written.** An evaluation fills
measurement slots and moves nothing a person put there, so a version per
evaluation would put a step in the history for every edit that Undo would then
have to walk back over, each restoring numbers that no longer matched the
inputs beside them. The version keeps its serial and its label.

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
([`track-view.md`](track-view.md) § "Edit mode"), not a published
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
handles: `ViewSources` carries the node's cameras and poses, a **shared** clone
of the pyramid the node's own full-resolution cache already holds for each
photograph, and a path for each one it does not, and `ViewSources::decode` turns
that into one pyramid per view on the worker, under a `decode images` phase
whose note says how many it read from disk and how many it reused from the
cache. The cache holds pyramids rather than bare photographs, so a photograph
the viewer had is neither decoded, nor copied, nor pyramided a second time; one
the worker reads itself is decoded and pyramided there and dropped with the
task, because the cache is the GUI thread's and the panels fill it for what they
draw. A photograph that cannot be read is the
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
started at that pixel would take (`IMG_0042@142,198`), made active, one `Bench`
row whose sentence names the pixel, the member that built it, its `in` count and
median ZNCC with the median middle ZNCC beside it (`median ZNCC 93% / 71%`),
and any members that refused before it. The second is
`commit_bench_track`, the step Track View's *Commit* takes, so its version, its
`Edit` row, the selection of the written point and the item left seated on it
are that step's own. The operation's row is written first and the commit's
after it, both as whoever asked for the run. One Undo takes back the point and
leaves the track on the bench, active, which is where a person who wants to
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
keypoints, are built on the worker for the run and dropped with it: the viewer
keeps no decoded set of photographs between runs for a cache of them to sit
beside, and building them is a small part of a run, which decodes every
photograph it does not find in the node's full-resolution cache. The options
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
  *Edit on Bench* activates the item it made rather than putting a second one
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
and `1a` active already -- pushes no version and ends its row with *no effect,
the bench holds them already*.

A label another item holds takes core's ` (n)` suffix, as every label on the
bench does, so a second find at the same pixel puts its new tracks beside the
first's (`frame_13@412,230 2a (2)`); a caller who wants them apart names a
group label. The version is labelled *Found 8 nearby tracks at
frame_13@412,230*. The row is an `Edit` row when the version wrote points and
a `Bench` row when it wrote only the bench, and its sentence says how many
tracks were found in how many layers, the first layer's confidence, what
became of the tracks, and which item is active:

> Found 8 nearby tracks in 2 layers at (412.0, 230.0) in frame_13.jpg, the
> first layer at 87% confidence: 5 committed as new points, 3 existing points
> put on the bench; frame_13@412,230 1a is active (v12 -> v13)

**`1a` is the active item afterwards**, the track nearest the pixel on the
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
label, with its `in` count, the active one marked as a selected row. There is no
eye: nothing on the bench is drawn from these rows, and an item is not part of
the reconstruction.

A click on a row selects the node it is under and does nothing to the bench,
as a click on the node's own row selects it, so a pass of clicks down the tree
pushes no versions. A **double-click** makes the item active, selects the node
and raises Track View on it (`AppState::edit_bench_item_at`); on the item that
is active already it pushes no version and writes no row, since the gesture
asked for the panel. A secondary click offers *Discard*, and one on either
group's header offers *Clear the Bench*, which takes off every item of both
groups in one version (`AppState::clear_bench`). One item is active
across both groups, the kind being the item rather than the stage. The bench is in the tree
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
tools take `track`, and **a call that names no track acts on the active one**,
resolved with `active_track_label`, which is what a gesture in Track View's edit
mode means when it names no item. With nothing active such a call is refused:
*"No track is active on bull's bench. Name one with track, activate one with
activate_bench_item, or put one on with create_bench_track or
create_bench_cluster."*

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
// activate_bench_item  { "reconstruction_label": "bull", "item": "IMG_0042@142,198" }
// deactivate_bench_item { "reconstruction_label": "bull" }
// rename_bench_item    { "reconstruction_label": "bull", "item": "IMG_0042@142,198",
//                        "label": "bull-nose" }
// discard_bench_item   { "reconstruction_label": "bull", "item": "bull-nose" }
// duplicate_bench_item { "reconstruction_label": "bull", "item": "bull-nose" }
//
// One track on it. "track" omitted means the active track.
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
// set_bench_search_px          { "search_px": 8.0 }   // the viewer's, every track
// fit_bench_track           { "reconstruction_label": "bull" }
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
origin and counts, and the active label per kind, `null` when none is active,
which a bench holding items can be. It is **one flat list** in the
bench's own order, with the stage on each item, rather than the tree's two
groups: a reader that wants them apart has the field to do it with, and a
grouping on the wire would be the panel's layout rather than the bench's own
state. Both carry each track's `evaluation`: its `state` (`current`, `evaluating`,
`refused` or `failed`, § "Live evaluation"), the `reason` sentence where it is
refused or failed, `running` for whether an evaluation of the current inputs is
on a worker now rather than waiting to start, and the `search_px` it reads at.
An agent that has just made a step reads `evaluating` and the previous numbers,
and reads again until it says `current`. `get_bench_track` is
Track View's edit-mode table: the stage and its data, the origin, the thresholds, and
every observation with its provenance, verdict, `pixel` and both stages'
measurements where they exist -- at both stages `zncc_middle` and `zncc_grid`
beside `zncc` (the same samples read over the middle square of the patch and
over each ninth of it, § "The middle ZNCC" and § "The ZNCC grid" of
[`../core/bench/editable-track.md`](../core/bench/editable-track.md)), and
`localizability_middle_deprecated`, `localizability_grid_deprecated` and
`localizability_slide_deprecated` beside `localizability_deprecated`, and the ZNCC
self-similarity radius `zncc_self_similarity_radius` with its `_middle` and
`_grid`, `zncc_self_similarity_slide_grid`, `zncc_self_similarity_surface` and
`zncc_self_similarity_tolerance`
([`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "The
ZNCC self-similarity radius"), at the track stage
the two distances
(`seed_shift_px` and `projection_offset_px`), `walked_px`, `walked_to`, `walked_zncc`, `walked_zncc_middle` and `walked_zncc_grid` for a row the last fit
refused to move (`sight_bench_observation` at `walked_to` accepts that walk), and, for a row the evaluation could
not score, the `reason` sentence in place of a ZNCC. The track stage's own data
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
look at. The list is empty on a track that is not active. **An observation is
addressed by its position in
that list**, which is stable for the life of the track, so an index an agent is holding after
a verdict or a fit still names the same observation. The template's
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
stage, a label, which item is active -- compare it exactly, because there is
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

- putting a second item on, activating it and discarding it are three versions,
  an undo of the discard puts it back with the activation as it stood, and the
  first item is the same `Arc` throughout;
- putting a point on the bench twice activates the track it already made;
- the two gestures the Image Detail context menu carries -- a cluster started at
  a pixel, then a candidate added at one in the same image -- are one version and
  one `Bench` row each, the second sighting joining as a candidate, and an undo
  walks them back one at a time;
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
  the rays earned and on what residuals, while its **Action Log row** carries the
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
`Refused` with core's sentence and starts nothing; and nothing starts while a
background task holds the node.

The deactivation is tested where its two callers are: [track_view/tests.rs](../../crates/sfm-explorer/src/track_view/tests.rs) clears the *Edit* box and finds one version
labelled as above with the item still on the bench, an undo bringing edit mode back, and a
discard of the active item leaving view mode; [mcp/tests.rs](../../crates/sfm-explorer/src/mcp/tests.rs) calls
`deactivate_bench_item` twice, one version and then a no-effect reply, with `get_bench`
reporting `null` active in between.

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
has no index files), the item is active and seated on the point, and one undo
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
active and its point selected; a second find there puts no copies on and
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

The wire is tested in
[mcp/tests.rs](../../crates/sfm-explorer/src/mcp/tests.rs), over the same
fixture with a label on the node, for what the boundary owes: each tool being
the `AppState` call the panel makes, the reads reporting `evaluation.state` as
`evaluating` until the evaluation lands and `current` after it,
`set_bench_search_px` pushing no version and making a track `evaluating` at the
new radius, an observation index surviving the steps
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
  mark the active track's `in` observations are proposed in the same draft. Both
  of the panels that do draw the active track draw it as handles: the Image
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
