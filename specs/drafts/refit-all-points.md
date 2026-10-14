# Refit All Points

**Status:** Draft

Amends [`../gui/scene-graph.md`](../gui/scene-graph.md) § "Tree rows" (a new
entry on the reconstruction row's context menu) and
[`../gui/edits/retriangulate-point.md`](../gui/edits/retriangulate-point.md)
(the edit family it sits beside), and builds on
[`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "Fitting"
(the per-point step it runs in batch). Both standing specs point back here.

`Refit All Points` is a proposed entry on a reconstruction row's context menu in
the Scene Graph, directly under `Retriangulate All Points`. For every point of
the reconstruction it does what Track View's *Fit* button does to one track put
on the bench from that point, with every verdict `in` and pinned, followed by
the commit: the other observations are aligned to the render of the point's
reference observation, their keypoints are rewritten, the point is
re-triangulated (or, for a point at infinity, its direction is re-estimated),
the patch is placed at the answer and its stored bitmap is rendered again from
the reference observation, and each observation's readings are taken again
against that bitmap. The whole reconstruction comes back as one new version, on
the background worker, with a `Progress` and a cancel. The mechanism is one
function in `sfmtool-core`, as `Retriangulate All Points` is.

Where `Retriangulate All Points` re-solves each point from the keypoints it
already has, this reads the keypoints again from the photographs first. It is
what a reviewer wants after a run of camera moves or a bundle adjustment has
left the keypoints of many points a pixel or two from where the photographs put
the patch's content, and fitting each point on the bench by hand would take one
press per point.

---

## The interface

### The menu entry

A reconstruction row's context menu gains `Refit All Points` immediately after
`Retriangulate All Points`, so the whole-reconstruction entries read
`Bundle Adjust...`, `Retriangulate All Points`, `Refit All Points`,
`Prune Covered Observations`. It acts on the node the menu was opened on, not on
the selection. Its hover text:

> Fit every point to the render of its reference observation, as Track View's
> Fit does with every observation kept: rewrite each observation's keypoint,
> re-triangulate the point and render its patch bitmap again, as one version.
> Moves no camera. Reads every photograph the points observe. Runs on a worker
> thread and can be cancelled; Undo (Ctrl+Z) puts the points back.

It is greyed, with the reason as hover text, by one function,
`state::edits::refit_refusal`, which the menu entry, the wire's tool and the
operation itself all ask, as `retriangulate_refusal` is asked today. The
reasons, in the order they are checked:

1. an operation is already running on the node (the busy sentence);
2. a camera model proposal is open on the node (*"Apply or cancel the camera
   model proposal first."*), as for every other entry that edits the node;
3. the observations carry no inline keypoint (a `sift_files` reconstruction),
   with the sentence naming `Convert to Embedded Patches`, since the commit
   refuses such a value (`CommitError::NotEmbeddedPatches`);
4. the points carry no patch frame, since a fit registers the frame into every
   view and refuses a track without one (`FitError::NoFrame`);
5. no image carries a pose.

Whether every photograph can be read and whether the reconstruction's
reprojection noise can be measured are not asked by the gate: the first reads
files and the second walks every observation, and neither belongs on the GUI
thread every frame the menu is open. Both are refused by the operation on the
worker, as a failed entry (§ "What a run can refuse").

Over MCP the same action is `refit_all_points`, which defers to the worker and
comes back with the version, as `retriangulate_all_points` does
([`../gui/mcp-server.md`](../gui/mcp-server.md)).

### The core function

The batch is one function in the bench module of `sfmtool-core`,
`bench::refit_points`, in a new file `crates/sfmtool-core/src/bench/refit_points.rs`:

```rust
/// How a whole-reconstruction refit runs.
#[derive(Debug, Clone)]
pub struct RefitPointsOptions {
    /// The fit each point gets. Track View's Fit passes
    /// `FitOptions::default()` with the noise level its panel holds in
    /// `sigma_px`. `sigma_px: None` is measured once for the whole run from
    /// the value it was handed, not once per point.
    pub fit: FitOptions,
    /// How far, in grid px, a fit may move a sighting before it keeps its
    /// keypoint instead: the bound `Thresholds::max_shift_px` sets for a track
    /// on the bench. `BENCH_MAX_SHIFT_PX` (6) by default, the value every track
    /// put on the bench starts with.
    pub max_shift_px: f64,
}

impl Default for RefitPointsOptions { /* FitOptions::default(), BENCH_MAX_SHIFT_PX */ }

/// Refit every point of `edited` to the render of its reference observation
/// and hand back the value that holds the answers.
pub fn refit_points(
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &RefitPointsOptions,
    progress: &Progress<'_>,
) -> Result<(EditedReconstruction, PointMap, RefitPointsReport), RefitPointsError>;
```

**Why it is shaped this way.**

- **It takes the photographs decoded, as [`fit`](../../crates/sfmtool-core/src/bench/fit.rs)
  does.** `images` is one `ProjectedImage` per image of `edited`, indexed by
  image index, which is what every photometric step of the bench takes. Reading
  the files stays with the caller: the viewer reads through its photograph cache
  on the worker
  ([`ViewSources`](../../crates/sfm-explorer/src/state/edits.rs)), and Python
  resolves its `images` argument as `sfmtool.bench.fit` does. So core reads no
  file and holds no cache.
- **It returns the same triple as
  [`retriangulate_points`](../../crates/sfmtool-core/src/reconstruction/triangulation/retriangulate.rs)**:
  the next value, the `PointMap` from the caller's indexes to it, and a report
  with one status per point. The viewer then installs it the way it installs a
  retriangulation, and an agent reads it the same way.
- **There is no `which` argument.** The request is for every point, and a fit
  of one point already exists: put it on the bench, press *Fit*, press
  *Commit*. A list form is an open question (§ "Open questions").
- **The settings struct is `*Options`**, holding the per-point `FitOptions`
  rather than repeating its fields, so the two cannot drift: a refit of a point
  is the fit of that point.
- **It lives in `bench`, not in `reconstruction`.** It is built from three bench
  steps, and `reconstruction` does not depend on `bench`; putting it there
  would make it.

Example:

```rust
use sfmtool_core::bench::{refit_points, RefitPointsOptions};
use sfmtool_core::progress::Progress;

let (next, map, report) =
    refit_points(&edited, &views, &RefitPointsOptions::default(), &Progress::none())?;
println!(
    "{} of {} points refit, {} left as they were",
    report.refit(),
    report.read(),
    report.kept(),
);
```

**The report.** `RefitPointsReport::points` holds one `RefitPoint { index,
new_index, outcome }` per point of the value: `index` in the caller's indexing and
`new_index` in the returned value's. The two differ only where the caller's
value had pending point edits, whose deleted slots close up when the overlay is
folded in (§ "Writing the answers"); the refit itself moves no index.
`RefitOutcome` is:

- `Refit { placed, kept_at_seed, crossed, shift, classification, unchanged }`:
  the point was fitted and written. `placed` and `kept_at_seed` are the
  `FitReport`'s counts; `crossed` says it went from finite to a direction or
  back; `shift` is the distance a point that stayed finite moved, `None`
  otherwise; `classification` is the fit's `TrackClassification`;
  `unchanged` says the record it would write agrees with the stored one
  (`PointRecord::agrees_with`) and the stored reference is not one only the
  display render picked, so the point was not rewritten, the rule `commit`
  uses.
- `Held` and `Ranged`: the point's constraint kept it out of the run
  (§ "Constraints").
- `NotFitted(FitError)`: the fit refused this point, and the point keeps every
  column it had (§ "Points the fit cannot place").
- `NoBitmap`: the fit placed the point but rendered no bitmap, in a value that
  stores one per point, and the point keeps every column it had.

The counts (`read`, `refit`, `unchanged`, `crossed`, `kept`, `held`, `ranged`,
`kept_at_seed`, `with_error(kind)`, `median_shift`) are read off the statuses,
as `RetriangulateReport`'s are.

`RefitPointsError` is what refuses the run as a whole: `NoKeypoints` (a
`sift_files` value), `NoPatchFrames`, `NoPosedImage`, `ViewsMissing { got, expected }`,
`NoNoiseLevel(String)` (no `sigma_px` given and the value's reprojection noise
cannot be measured, the fit's own refusal) and `Cancelled`. The error and the
report carry the `RefitPoints` prefix of the options because `RefitError` and
`RefitOptions` already name `camera::refit_intrinsics`'s types.

---

## What one point's refit is

For each point the batch runs the steps the bench runs when a person puts the
point on the bench, presses *Fit* and presses *Commit*, and nothing else:

1. **The track.** The point becomes a track-stage `EditableTrack` the way
   [`create_track`](../../crates/sfmtool-core/src/bench/steps.rs) builds one:
   its own frame, bitmap, colour, keypoints and stored readings, its stored
   reference observation as the track's reference, and every observation `in`
   and **pinned**. Nothing is recomputed. The difference is that it is built
   without a `Bench`: `create_track` puts the track on a bench and mints a
   label, which a batch has no use for (§ "What moves into core").
2. **The fit.** [`bench::fit`](../../crates/sfmtool-core/src/bench/fit.rs) at
   the track stage, with the run's `FitOptions` and the track's
   `max_shift_px`:
   - every sighting is localized and refined against the **reference
     render**, the render of the reference observation at its own keypoint
     through the frame the point arrives with
     ([`localize_patch_keypoints`](../core/patch/patch-keypoint-localization.md)
     then `refine_patch_keypoints`). Because every row is pinned, the
     reference held is the stored one where its row has a keypoint
     (`EditableTrack::held_reference`), and its keypoint is not moved;
   - a sighting the kernels would move further than `max_shift_px` keeps its
     keypoint, and one they did not place (out of frame) keeps its pixel;
   - the keypoints are classified and triangulated by `classify_track_rays`,
     the point-or-bearing test, at the run's noise level, and the frame is
     placed at the answer at the apparent size it had;
   - the stored bitmap is rendered from the reference observation through that
     placed frame (`render_bitmap`, the same render
     [`render_bitmap_in_place`](../../crates/sfmtool-core/src/bench/fit.rs)
     and [`render_patch_bitmap`](../../crates/sfmtool-core/src/patch/stored_bitmap.rs)
     make), with the colour at its centre;
   - the fit ends by evaluating its result
     ([`evaluate`](../../crates/sfmtool-core/src/bench/evaluate.rs)): every
     row's tile is rendered again, its self-similarity reading, viewing angle,
     tilt direction and zoom are read, and it is scored against the new bitmap.
     These are the "track bitmaps" of the request, and they are what the
     record's observation readings and `observation_confidence` are written
     from.

   The request lists the bitmap update before the alignment. The fit does the
   alignment against the reference render at the frame the point arrived with,
   which is the same tile the bitmap would have been updated to; it then renders
   the stored bitmap once more through the frame the re-triangulation placed, so
   that the bitmap written describes the patch that is written. The two orders
   align the sightings to the same image.
3. **The record.** The `PointRecord` the commit builds from the fitted track
   ([`commit`](../../crates/sfmtool-core/src/bench/commit.rs)): the fitted
   position with the classification's `w`, the placed frame, the new bitmap and
   its reference observation, the colour at the bitmap's centre, the mean of the
   evaluated reprojection errors in `error`, the normal the frame states (zero
   for a direction), and per observation, in image order, the new keypoint, its
   blur-matched score as `observation_confidence` and its readings
   (`measurement_reading`). Built, compared with the stored record, and not
   applied: the batch writes every record at once (§ "Writing the answers").

What the batch adds to the three steps is the loop, the constraint rule, the
per-point failure rule and the in-place write. It sets no verdict and paints
none: the thresholds never run, because every row is pinned `in`.

### Points at infinity

A point stored with `w = 0` comes onto the track as a bearing
(`TrackPayload::at_infinity` from the point's own `w`, not from the frame), and
the fit registers its frame as it is, tangent to the direction sphere. The
classification then decides whether its keypoints now earn a depth: a bearing
whose rays have one becomes a finite point, and a finite point whose rays no
longer fix one becomes a bearing, each with the frame carried across at the
apparent size it had. The record is written with `w = 0`, a zero normal and a
`normal_confidence` of zero for a bearing, as the commit writes one. The batch
itself never divides by `w`: it copies the record's columns, and its one
derived number, `shift`, is measured only for a point finite both before and
after, with `crossed` counting the rest.

This is a difference from `Retriangulate All Points`, whose defaults turn the
angular floor and the likelihood rule off, so that no free point crosses
between the two representations on its rays' account alone. The fit's
classification is on, as it is for the bench's *Fit*, so a refit can move points
across; the report counts them (`crossed`).

### Constraints

A point the value **holds** or **ranges** is a statement a reviewer made by hand
(`POINT_CONSTRAINT_HELD`, `POINT_CONSTRAINT_RANGED`). The commit writes every
point free, because a track on the bench says nothing about a distance, so a
refit that committed such a point would silently drop the constraint. The batch
leaves both out: not fitted, not written, and reported as `Held` or `Ranged`.
This is the rule `Prune Covered Observations` follows; whether a ranged point
should instead be refit at its distance is an open question.

### Points the fit cannot place

A point the fit refuses keeps every column it had, and its status names the
refusal:

- fewer than two observations (`TooFewObservations`);
- no patch frame on this point (`NoFrame`), possible where the column exists
  but the row is empty;
- the keypoints give fewer than two usable rays (`TooFewRays`) or the linear
  solve is not a number (`Triangulation`);
- a round's tiles would pass `EvaluateOptions::max_cache_bytes` (`TooLarge`) or
  a buffer could not be allocated (`OutOfMemory`).

A sighting the kernels cannot place is not a failure of the point: it keeps its
keypoint, its ray still feeds the triangulation, and its readings say it was not
read. So a point some of whose observations fit and some do not is written with
the fitted ones moved and the others where they were, which is what *Fit* does
on the bench.

**A reference observation that cannot be rendered.** Where the stored reference
is held, the bitmap is that row's tile. Where the point has no reference (stored
`-1` with no display pick, or a reference row with no keypoint), the fit runs
the reference-view rule over the `in` sightings and the pick becomes the
reference, as it does on the bench. Where the rule finds no eligible sighting
the fit renders no bitmap. In a value that stores a bitmap per point the record
cannot be written (the commit's `NoBitmap`), so the point keeps every column it
had and its status is `NoBitmap`. In a value that stores no bitmaps the point is
written, with no reference observation where the track carries none.

A point the viewer's display render gave a reference (`display_only_reference`)
comes onto the track with that reference held, as on the bench, and the record
writes it as the point's own. After a refit, a save writes those references
rather than `-1`, as it does after a commit of the same point.

---

## Writing the answers

The run is a **bulk edit**, the shape `RetriangulateWhich::All` writes:

1. The overlay is folded in first (`EditedReconstruction::materialize`), in a
   `materialise` phase, where the value has pending point edits, so every point
   is a row of one base. The `PointMap` the call returns is the one
   `retriangulate_points` returns for `All`: the fold's rows chained with the
   write, which moves nothing, or an empty removal where there was nothing to
   fold.
2. Every record is written into a copy of that base (`clone_for_edit`) at its
   own row. The observations a record carries are the point's own, all `in`,
   in image order, which is the order the stored track is in, so each
   observation is written over its own row: the point's track keeps its length
   and its offsets, and no point and no observation row is added, removed or
   renumbered. The writer asserts that each record's image list equals the
   stored track's.
3. A point whose status is not `Refit`, or whose record agrees with the stored
   one while its reference is not a display-only pick, is not touched.

The writer is new. `retriangulate_points`'s own `write_in_place` writes
positions and frames only, and the commit writes one record through
`replace_point`, which is an overlay edit that gives the point a new index. A
batch of tens of thousands of `replace_point` calls would renumber every point
and rebuild the addition set's derived indexes once per point, which is the
reason `RetriangulateWhich::TheseInBulk` exists.

The columns written are the ones the commit's record carries, under the same
rules: `observation_confidence`, the observation readings, the normals, the
normal confidences and the reference observations only where the value has the
column; the patch bitmap only where the value stores bitmaps. A column the value
marks `patch_bitmaps_for_display` keeps that mark (§ "Open questions").

---

## The run

### Stages and progress

```text
materialise          (only where the overlay is not empty)
measure noise        the run's sigma_px, once, where none was given
fit points           counts "k of n points" and sets its fraction
write back
```

The viewer's job adds a `decode images` phase in front, reading every
photograph an observation names through the photograph cache, with the note
every `decode images` phase carries
([`../gui/background-tasks.md`](../gui/background-tasks.md) § "Reporting
progress").

Each point's fit is called with `&Progress::none()`. A fit opens a phase for
`localize`, `refine`, `bitmap` and each stage of its evaluation, and per-point
phases from a rayon loop are the per-item timing
[`../gui/operation-progress.md`](../gui/operation-progress.md) § "Non-goals"
refuses. The loop's own phase reports the count and the fraction instead, from
an atomic counter, as `render_patch_cloud_bitmaps` does.

### Cancellation

The run is cancellable. The flag is read in front of every stage, and in front
of every point's fit inside the loop (`progress.is_cancelled()`); a point whose
fit has begun runs to its end, so a cancel lands within one point's fit per
worker thread. A cancelled run returns `RefitPointsError::Cancelled` and writes
nothing: the value the caller holds is unchanged, and no partially refit value
is returned, because a reconstruction in which some points were refit and others
were not, chosen by thread scheduling, is not an answer anybody asked for. This
is `retriangulate_points`'s rule.

### Parallelism

The points are fitted in parallel over the rayon pool, one point per task, with
the results collected by point index. The fit's kernels are themselves
rayon-parallel over a point's views; rayon runs the nested work on the same
pool. Peak memory per point is bounded by `EvaluateOptions::max_cache_bytes`
per round, so the run's peak is that bound times the number of worker threads,
on top of the decoded photographs.

### Determinism

Every point's fit reads only the value the run was handed and the photographs,
never another point's answer: the batch fits every point against the input
value and writes all the answers at the end. So the answer depends on neither
the order the points are visited in nor the thread count, and the same value and
photographs give the same output bit for bit. A test runs the batch in a
one-thread pool and in the default pool and compares the values.

A refit is not promised to be idempotent: a second run fits against the
patches and bitmaps the first wrote, and can move keypoints again by small
amounts. The first refit of a point that was not moved at all is in general
still a change, because the commit's colour is read from the bitmap's centre
and its `error` is the evaluated reprojection error, neither carried from the
stored point ([`../core/bench/editable-track.md`](../core/bench/editable-track.md)
§ "The commit").

### What a run can refuse

Before any point is fitted, the run is refused as a whole, writing nothing,
when the value is `sift_files`, carries no patch frame, has no posed image, is
handed fewer views than images, or has no noise level. In the viewer, a
photograph an observation names that cannot be read fails the job in its
`decode images` phase with *"Cannot read <image>."*, as the bench's *Fit* does
(open question on this below).

---

## In the viewer

The viewer wraps the core function as `Retriangulate All Points` is wrapped
([`edits.rs`](../../crates/sfm-explorer/src/state/edits.rs)):

- **The operation.** A new
  [`Operation`](../../crates/sfm-explorer/src/background/mod.rs)
  `REFIT_ALL_POINTS { name: "Refit all points", cancellable: true, kind:
  Kind::Edit }`, beside `RETRIANGULATE_ALL_POINTS`, held to its cancellable
  claim by the test that cancels every operation that makes it.
- **The job.** `AppState::start_refit_all_points` and `refit_all_points_job`,
  shaped as `start_retriangulate_all_points` and its job: the gate is
  `refit_refusal`; what crosses to the worker is a clone of the value at the
  cursor and a `ViewSources` from `view_sources_for` naming every image some
  point observes; the worker decodes, builds the views and calls
  `refit_points`; the answer is a `Finished::Produced` with the folded base.
- **The menu.** `show_refit_entry` in
  [`menus.rs`](../../crates/sfm-explorer/src/scene_graph/menus.rs), with a
  `REFIT_ALL_POINTS` label constant and a `SceneGraphResponse` field, drawn
  right after `show_retriangulate_entry`.

### The version

| | Every point |
|---|---|
| Version label | `Refit seoul_bull` |
| Entry | that label, then the counts, then the serials |

One version, a bulk edit with a new base and an empty overlay, pushed by the
frame that polls the finished task. The refit moves no index (pending point
edits close up when the overlay is folded in, and the map carries the selection
across that), so the selection stays on its point, and what the panels cached about each point's geometry, keypoints
and bitmap is dropped off `Polled::installed`, as after a retriangulation. The
bench is not touched: a bench track whose origin is a refit point still
resolves, and committing it replaces the refit point.

**Undo** (Ctrl+Z) steps back to the version before, which holds every point's
keypoints, bitmap and geometry byte for byte; Redo steps forward again. That is
what the history already does for a bulk edit, and nothing here adds to it.

**The entry** says what the run did, each count read off the statuses, a count
of zero left out, in the shape `Retriangulate All Points` uses:

```text
Refit seoul_bull: 812 of 840 points refit, 3 crossed to or from infinity,
41 observations kept their keypoint past the shift bound, 9 left as they were
(5 with too few rays, 4 with no bitmap to render), 16 held, 3 ranged,
median shift 0.0021 (v3 → v4)
```

The words for each `FitError` a point can be left for come from one exhaustive
match in `edits.rs`, as the retriangulation's words for each `PointVerdict` do,
so a refusal added to the fit does not compile until the sentence has words for
it.

A **cancelled** run pushes no version and writes one failed entry,
`Refit all points of seoul_bull cancelled`, the sentence the background task
writes for every cancelled operation. A **refusal** pushes no version and writes
one failed entry in the gate's own sentence. A run in which every point was left as it was, or agrees
with what is stored, still pushes its version, whose entry says so, as a
retriangulation that moved nothing does.

---

## What moves into core

The per-point Fit already lives in `sfmtool-core`: `bench::fit`,
`bench::evaluate`, `bench::create_track` and `bench::commit` are core functions,
and the viewer's *Fit* button is a job that decodes the photographs and calls
`bench::fit` ([`bench.rs`](../../crates/sfm-explorer/src/bench.rs),
`bench_fit_job`). Nothing has to move out of the viewer. Two pieces of existing
core functions have to be callable on their own:

- **The track of a point without a bench.** The body of `create_track` that
  builds the `EditableTrack` from a point becomes a function of its own,
  `bench::track_of_point(edited, point, origin)`, which `create_track` calls and
  then puts on the bench. Public, because Python's batch callers have the same
  use for it.
- **The commit's record without the edit.** The part of `commit` that validates
  the track and builds its `PointRecord` becomes
  `bench::commit::point_record(edited, track) -> Result<PointRecord,
  CommitError>`, crate-visible, which `commit` calls before its
  `replace_point` / `add_point`. So the batch and the commit write one record
  for one track.

The new code is the loop, the constraint rule, the report and the in-place
writer, all in `bench/refit_points.rs`. Python gets `sfmtool.bench.refit_points`,
taking `images` as `sfmtool.bench.fit` does, because every edit family's
mechanism is bound ([`../gui/edits/README.md`](../gui/edits/README.md)).

---

## Standing specs the conversion changes

- [`../gui/scene-graph.md`](../gui/scene-graph.md) § "Tree rows": the menu
  list, a `Refit All Points` bullet after `Retriangulate All Points`'s, the list
  of entries a camera model proposal greys, and the sentence that
  `Prune Covered Observations` "sits directly under `Retriangulate All Points`",
  which becomes "under `Refit All Points`, after the two entries that re-read
  the points"; § "Testing" for the entry's position.
- A new edit spec, `gui/edits/refit-all-points.md`, with a row in
  [`../gui/edits/README.md`](../gui/edits/README.md); its Related specs and
  [`../gui/edits/retriangulate-point.md`](../gui/edits/retriangulate-point.md)
  link each other, and the latter's non-goal "Adding, removing or walking an
  observation" names the refit as the edit that moves keypoints.
- A new core spec, `core/bench/refit-points.md`, with a row in
  [`../core/bench/README.md`](../core/bench/README.md), and in
  [`../core/bench/editable-track.md`](../core/bench/editable-track.md) the
  `track_of_point` and `point_record` functions in § "Rust API", § "Putting a
  point on the bench" and § "The commit", and the binding in § "Python
  bindings".
- [`../gui/background-tasks.md`](../gui/background-tasks.md): the operation in
  the list of background operations and its phases.
- [`../gui/mcp-server.md`](../gui/mcp-server.md): the `refit_all_points` tool.
- [`../gui/track-view.md`](../gui/track-view.md) § "The toolbar": one sentence
  that *Fit* has a whole-reconstruction form on the Scene Graph's menu.
- [`../GLOSSARY.md`](../GLOSSARY.md): a **refit** (of a point) row, separating
  it from **refit camera intrinsics** and from **retriangulate**.

---

## Testing

Core, in `bench/tests/refit_points.rs`:

- **A refit of one point equals the bench's.** For each point of a small value,
  `refit_points`'s record equals the record from `create_track`, `fit` and
  `commit` on that point alone, column for column.
- **Keypoints nudged off the content come back.** On the seoul_bull ground truth
  with its photographs, keypoints displaced by a few pixels return to within
  the refine's tolerance of the stored ones, and the points move back toward
  their stored positions.
- **No index moves**, the point and observation counts are unchanged, and every
  track keeps its images.
- **A point at infinity** is fitted as a bearing and written with `w = 0`, a
  zero normal and a zero normal confidence; a bearing given a baseline crosses
  to finite and is counted.
- **Held and ranged points** are not fitted and come back byte for byte.
- **A point the fit refuses** (two observations whose rays are parallel, one
  observation) and **a point with no eligible reference** in a value that stores
  bitmaps come back byte for byte, with the status naming why.
- **Cancel** before the loop and part way through it returns `Cancelled` and
  leaves the input untouched.
- **Determinism**: a one-thread pool and the default pool give identical
  values.
- **The refusals**: `sift_files`, no frames, no posed image, too few views, no
  noise level.

Viewer, beside the retriangulation's tests:

- the entry is live on an embedded-patches node, drawn directly under
  `Retriangulate All Points` and above `Prune Covered Observations`, and dead
  with the right sentence on a `sift_files` node, a node with no frames and a
  busy node;
- the job pushes one version labelled `Refit <node>` with a new base, the same
  point count and the counts in the entry; Undo restores the value byte for
  byte;
- a cancelled run pushes no version and writes the failed entry;
- `the_prune_entry_is_live_on_a_node_with_patch_frames`, whose doc comment and
  failure message say Prune sits directly under `Retriangulate All Points`,
  measures Prune against the refit entry instead;
- the wire's `refit_all_points` defers to the worker and comes back with the
  version.

---

## Open questions

1. **Ranged points.** Recommended: leave held and ranged points out, as above.
   The alternative for a ranged point is to fit its keypoints and bitmap and
   then re-read only its direction at its distance with
   `retriangulate_points`' ranged rule, which keeps the constraint but makes the
   point's geometry the retriangulation's rather than the fit's. Worth doing
   only if ranged points are common enough for a refit to need them.
2. **An unreadable photograph.** Recommended: refuse the run, naming the image,
   as the bench's *Fit* does (`ViewSources::decode`). The alternative is
   `decode_available` with each point that observes an unreadable image left as
   it was and counted, which lets a run go ahead with a photograph missing, at
   the cost of a status kind and a rule for which points it covers.
3. **The classification.** Recommended: keep the fit's classification on, so a
   refit is the bench's *Fit* and points may cross to or from infinity. The
   alternative is to pass the retriangulation's rule (no crossing on the rays'
   account alone), which makes a refit and a retriangulation agree on which
   points are bearings but makes a refit differ from *Fit* on one point.
4. **A bitmap column rendered for display.** Recommended: the refit rewrites
   the rows and keeps the `patch_bitmaps_for_display` mark, so a save writes the
   columns the file had and no bitmap column the file lacked; adding one is
   `sfm xform --add-patch-bitmaps`. The alternative is to clear the mark, since
   the column is now the refit's answer, which makes every save of a refit node
   write bitmaps.
5. **A list form.** Recommended: none for now; the request is for every point,
   and one point is *Fit* and *Commit* on the bench. A `RetriangulateWhich`-like
   argument (`All`, `TheseInBulk`) can be added when a caller wants, for
   instance, every point one camera observes.
6. **Cost.** Unmeasured. A bench fit of one track is localize, refine and a full
   evaluation, so a run over a reconstruction of tens of thousands of points may
   take minutes, and holds every photograph decoded for its length. The
   implementation should measure the run on the seoul_bull and dino_dog_toy
   ground truths before the edit spec is filed, and record the numbers there.

---

## Non-goals

- **Moving a camera or a lens.** The poses and the cameras are the input, as for
  `Retriangulate All Points`; an answer that moved them would be an adjustment.
- **Changing which observations a point has.** Every verdict is pinned `in`, so
  no observation is turned out, added or searched for. Deciding which
  sightings a track should have is the bench's work.
- **Turning the patch.** The fit keeps the way the patch faces. *Fit Normal* and
  the finite-difference normals stay per-track bench steps.
