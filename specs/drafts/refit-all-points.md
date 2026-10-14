# Refit All Points

**Status:** Draft

Amends [`../gui/scene-graph.md`](../gui/scene-graph.md) § "Tree rows" (a new
entry on the reconstruction row's context menu) and
[`../gui/edits/retriangulate-point.md`](../gui/edits/retriangulate-point.md)
(the edit family it sits beside), and builds on
[`../core/bench/editable-track.md`](../core/bench/editable-track.md) § "Fitting"
(the per-point step it runs in batch) and
[`../core/reconstruction/triangulation-rules.md`](../core/reconstruction/triangulation-rules.md)
§ "A direction read at a held distance" (the rule it applies to a ranged point).
The first two standing specs point back here.

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

/// Refit every point of `recon` to the render of its reference observation
/// and hand back a copy of `recon` that holds the answers.
pub fn refit_points(
    recon: &SfmrReconstruction,
    images: &[ProjectedImage<'_>],
    options: &RefitPointsOptions,
    progress: &Progress<'_>,
) -> Result<(SfmrReconstruction, RefitPointsReport), RefitPointsError>;
```

**Why it is shaped this way.**

- **It takes the photographs decoded, as [`fit`](../../crates/sfmtool-core/src/bench/fit.rs)
  does.** `images` is one `ProjectedImage` per image of `recon`, indexed by
  image index, which is what every photometric step of the bench takes. Reading
  the files stays with the caller: the viewer reads through its photograph cache
  on the worker
  ([`ViewSources`](../../crates/sfm-explorer/src/state/edits.rs)), and Python
  resolves its `images` argument as `sfmtool.bench.fit` does. So core reads no
  file and holds no cache.
- **It takes a plain `SfmrReconstruction`, not an `EditedReconstruction`.**
  Given an `EditedReconstruction`, the run's first step would be to fold the
  overlay in (`EditedReconstruction::materialize`) and work on a plain base
  from then on, so the fold belongs to the caller that has an overlay. The
  refit moves no index, so with the fold at the caller there is no `PointMap`
  to return: the answer is the next value and a report with one status per
  point, each addressed by the one index the input and the output share. That
  makes it a whole-reconstruction transform like
  `materialize_points_at_infinity` and the `sfm xform` operations, which a
  command or Python can call with no knowledge of edit overlays.
  [`retriangulate_points`](../../crates/sfmtool-core/src/reconstruction/triangulation/retriangulate.rs)
  takes an `EditedReconstruction` and returns a `PointMap` because it has
  selection forms: `RetriangulateWhich::These` writes the named points into
  the overlay, which gives each one a new index, and `TheseInBulk` folds the
  overlay in and writes the named points at their rows. A refit has no
  selection form (§ "Non-goals").
- **It borrows the value and returns a copy.** The per-point bench steps take
  an `EditedReconstruction` (§ "What moves into core"), so the run makes one
  copy of `recon` (`clone_for_edit`), wraps it once with
  `EditedReconstruction::new(Arc::new(copy))`, and hands that one wrapper to
  every point's steps. After the loop it drops the wrapper, takes the copy
  back out of the `Arc` (`Arc::unwrap_or_clone`, a move, since nothing else
  holds it) and writes the answers into it. The copy is the one the write
  needs in any case, so the wrapper costs no second copy of the
  reconstruction, and a signature taking `Arc<SfmrReconstruction>` would save
  nothing.
- **There is no `which` argument.** The request is for every point, and a fit
  of one point already exists: put it on the bench, press *Fit*, press
  *Commit*. There is no list form (§ "Non-goals").
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

let (next, report) =
    refit_points(&recon, &views, &RefitPointsOptions::default(), &Progress::none())?;
println!(
    "{} of {} points refit, {} left as they were",
    report.refit(),
    report.read(),
    report.kept(),
);
```

**The report.** `RefitPointsReport::points` holds one `RefitPoint { index,
outcome }` per point of the value, in index order. The refit moves no index, so
`index` names the point in the input and in the returned value alike.
`RefitOutcome` is:

- `Refit { placed, kept_at_seed, crossed, shift, classification, range,
  unchanged }`: the point was fitted and written. `placed` and `kept_at_seed`
  are the `FitReport`'s counts; `crossed` says it went from finite to a
  direction or back; `shift` is the distance a point that stayed finite moved,
  `None` otherwise; `classification` is the fit's `TrackClassification`, `None`
  for a point whose coordinate the distance rule decided; `range` is a
  `RefitRange` saying how the point's constraint was applied (§ "Constraints");
  `unchanged` says the record it would write agrees with the stored one
  (`PointRecord::agrees_with`) and the stored reference is not one only the
  display render picked, so the point was not rewritten, the rule `commit`
  uses.
- `Held`: the value holds the point's coordinate, so it was left out of the
  run (§ "Constraints").
- `NotFitted(FitError)`: the fit refused this point, and the point keeps every
  column it had (§ "Points the fit cannot place").
- `NoBitmap`: the fit placed the point but rendered no bitmap, in a value that
  stores one per point, and the point keeps every column it had.

The counts (`read`, `refit`, `unchanged`, `crossed`, `kept`, `held`, `ranged`,
`released`, `kept_at_seed`, `with_error(is)`, `median_shift`) are read off the
statuses, as `RetriangulateReport`'s are. `with_error` counts the points
refused with a `FitError` the predicate `is` accepts. `ranged` counts the
refit points whose direction was read at their distance, and `released` the
ranged points whose constraint was released (§ "Constraints").

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
     placed at the answer at the apparent size it had. For a ranged point the
     coordinate comes from the distance rule instead (§ "Constraints"), and the
     frame is placed at that coordinate in the same way;
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
   (`measurement_reading`). The commit writes every record free; the batch
   replaces the record's constraint with the point's own (§ "Constraints"),
   so a ranged point is written ranged. Built, compared with the stored record,
   and not applied: the batch writes every record at once (§ "Writing the
   answers").

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
between the two representations on its rays' account alone. A refit uses the
fit's classification, as the bench's *Fit* does, so a free point can cross in
either direction; the report counts the points that do (`crossed`). A refit
and *Fit* then agree on every free point, and a refit and a retriangulation can
disagree on which points are bearings. A ranged point whose distance the rule
applies does not cross: its `w` follows its distance. One whose reference image
has no pose is fitted free and can cross (§ "Constraints").

### Constraints

A point the value **holds** or **ranges** is a statement a reviewer made by hand
(`POINT_CONSTRAINT_HELD`, `POINT_CONSTRAINT_RANGED`, in the columns
`PointRecord::constraint` carries as a `(kind, distance, reference image)`
triple). The batch treats the two differently.

**A held point is left out**: not fitted, not written, and reported as `Held`.
The value owns its coordinate, so the fit's answer for it could not be
written, and `Retriangulate All Points` and `Prune Covered Observations` leave
held points alone too.

**A ranged point is refit, and its range is applied to the answer.** The
caller owns the point's distance `r` from its reference image's camera centre
`O`, and the refit owns its direction `d`, which is the split the adjustment and
the retriangulation already make. Everything in § "What one point's refit is"
runs as for a free point, the alignment, the bitmap and the evaluation
included, except the step that turns the fitted keypoints into a coordinate:

- **The coordinate is the distance rule's.** In place of
  `classify_track_rays`, the fitted keypoints of the `in` rows go through the
  triangulation's distance rule
  ([`triangulation-rules.md`](../core/reconstruction/triangulation-rules.md)
  § "A direction read at a held distance"), the rule `retriangulate_points`
  applies to a ranged point: `triangulate_points_through_cameras` over the one
  track, with `PointRules { distance: Some(&[range]), ..PointRules::default() }`
  and the value's cameras and poses. At a finite `r` the answer is
  `X = O + r · d`, with `d` the direction from `O` that minimizes the summed
  squared reprojection error of the fitted keypoints, and `w = 1`. At `r = ∞`
  the answer is the normalized mean of the keypoints' rays, written as a
  direction with `w = 0`. So the record's `w` is `0` exactly when its distance
  is infinite, which is the format's rule for a ranged row
  ([`sfmr-file-format.md`](../formats/sfmr-file-format.md) § "Per-point
  constraints"), and a ranged point
  never crosses. The frame is placed at that coordinate, and the bitmap and
  the evaluation read the patch there, so the record describes the point
  that is written.
- **The origin is resolved as the retriangulation resolves it.** At a finite
  `r`, `O` is the reference image's camera centre at the value's poses, by the
  rule of `ranged_entry` in
  [`retriangulate.rs`](../../crates/sfmtool-core/src/reconstruction/triangulation/retriangulate.rs),
  resolved for every ranged point before the loop. A ranged point at `r = ∞`
  names no reference and needs no origin. `ranged_entry` gives such a point no entry,
  and the retriangulation reaches the normalized mean of its rays through the
  `marks` rule instead, since the stored `w` is `0`; the fit's classification
  reads no mark, so the refit passes the infinite entry itself,
  `PointDistance { distance: f64::INFINITY, origin: [f64::NAN; 3] }`, which the
  distance rule answers without reading the origin.
- **The constraint is kept.** The record carries the point's stored triple,
  unchanged, so the point is written ranged, at the same distance, from the
  same reference image. Its status is `Refit` with `range:
  RefitRange::Applied`.
- **A ranged point whose finite distance has no origin** (its reference image
  has no pose) is fitted free, by the fit's classification, which is
  `retriangulate_points`' rule for the same point ("the rule says nothing about
  it and it is solved free"). Where the answer stays finite the stored triple
  is kept, as the retriangulation keeps it, and the status is `Refit` with
  `range: RefitRange::Unresolved { released: false }`. Where the fit's
  classification makes it a direction, a finite distance would no longer
  describe the coordinate, and the format allows no `w = 0` row at a finite
  distance; the constraint is then released, written free, which is the
  format's rule for an edit that rewrites a point's representation
  ([`sfmr-file-format.md`](../formats/sfmr-file-format.md) § "Per-point
  constraints", "What an edit does to them"; `release_touched_constraints` in
  [`analysis/infinity/convert.rs`](../../crates/sfmtool-core/src/analysis/infinity/convert.rs)
  applies it to a reclassification), and the status is `Refit` with `range:
  RefitRange::Unresolved { released: true }`. The refit does not leave such a
  point as it was: its keypoints are aligned like any free point's, and a
  constraint ends only where the fit makes the point a direction.
  `retriangulate_points` treats such a point the same way: it keeps the stored
  triple where its free solve stays finite, and releases the constraint where
  the solve makes the point a direction.

A free point's status is `Refit` with `range: RefitRange::Free`.

Because the record's constraint is the stored triple, a ranged point whose
fitted record matches the stored one column for column is `unchanged` and not
rewritten, by the same `agrees_with` comparison as a free point
(`agrees_with` compares the constraint triples too).

**A refit of a ranged point differs from the bench's.** The bench's *Fit* has
no distance rule, and its commit writes every point free, so putting a ranged
point on the bench and committing it releases the constraint. That is the
bench's behaviour and stays so; the refit is the place that keeps the range.

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

The run is a **bulk write** into a copy of the value it was handed, the shape
`RetriangulateWhich::All` writes into its folded base:

1. The copy is the one the run took at its start (`clone_for_edit`, § "Why it
   is shaped this way"), taken back out of the wrapper the per-point steps
   read, once every point has been fitted. The fits read it before anything is
   written to it.
2. Every record is written into that copy at its own row. The observations a
   record carries are the point's own, all `in`, in image order, which is the
   order the stored track is in, so each observation is written over its own
   row: the point's track keeps its length and its offsets, and no point and
   no observation row is added, removed or renumbered. The writer asserts that
   each record's image list equals the stored track's. So the returned value
   has the input's points at the input's indexes, and there is no `PointMap`
   to return.
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
column; the patch bitmap only where the value stores bitmaps. The constraint
columns, where the value has them, take the record's triple, which is the
stored one except for a released point (§ "Constraints"); a value with no
constraint columns gains none.

A bitmap column the value marks `patch_bitmaps_for_display` (rendered for the
viewer, not read from the file) is rewritten row by row and keeps that mark. So
a save writes the columns the file had and no bitmap column the file lacked;
adding one to the file is `sfm xform --add-patch-bitmaps`. Clearing the mark
because the column now holds the refit's answer would make every save of a
refit node write bitmaps, which the person did not ask for.

---

## The run

### Stages and progress

```text
measure noise        the run's sigma_px, once, where none was given
fit points           counts "k of n points" and sets its fraction
write back
```

Where no `sigma_px` was given, the noise level is measured on `recon` itself
(`SfmrReconstruction::reprojection_noise_px`, the measure
`EditedReconstruction::base_reprojection_noise_px` keeps), before the copy is
taken, and passed to every point's fit as its `sigma_px`. A plain value keeps
no measure, so the run measures it once even where the caller's own
`EditedReconstruction` had measured it already: one walk over the
observations per run, and none per point. The copy and its wrapper are made
after this stage, in front of the loop.

The viewer's job adds two phases in front: `decode images`, reading every
photograph an observation names through the photograph cache, with the note
every `decode images` phase carries
([`../gui/background-tasks.md`](../gui/background-tasks.md) § "Reporting
progress"), and then `materialise`, folding the overlay in, only where the
overlay is not empty (§ "In the viewer").

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
worker thread. A cancelled run returns `RefitPointsError::Cancelled` and
drops its copy before writing to it: the value the caller lent is unchanged, as
it is after any run, and no partially refit value is returned, because a
reconstruction in which some points were refit and others were not, chosen by
thread scheduling, is not an answer anybody asked for. This is
`retriangulate_points`'s rule.

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

Before the copy is taken, the run is refused as a whole, writing nothing,
when the value is `sift_files`, carries no patch frame, has no posed image, is
handed fewer views than images, or has no noise level.

**A photograph that cannot be read is an error for the whole run.** In the
viewer, a photograph some point observes that cannot be read fails the job in
its `decode images` phase with *"Cannot read <image>."*, through
`ViewSources::decode`, as the bench's *Fit* does, and no version is pushed. The
run does not go ahead without it: `ViewSources::decode_available` would let the
points that do not observe the missing image be refit, but every point that does
would then need a status of its own and a rule for which of its observations
the missing photograph covers, and a refit of most points with a few silently
skipped is harder to read than a refusal that names the file.

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
  point observes. The worker decodes and builds the views, then folds the
  overlay in itself, as `Add Image to Tracks` and `Bundle Adjust...` do: where
  the overlay is not empty, `edited.materialize()` in a `materialise` phase
  gives the plain value and a `RowMap`; where it is empty, the job reads
  `edited.base` and folds nothing. It calls `refit_points` on that value. The
  answer is a `Finished::Produced` whose `value` is the refit value and whose
  `map` is `PointMap::Chain` of the fold's `PointMap::Rows` (where there was a
  fold) and `PointMap::Removed(Vec::new())`, the identity, since the refit
  moves no index.
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
frame that polls the finished task: the poll installs a `Finished::Produced`
as it installs every one, by `history.push(EditedReconstruction::new(Arc::new(value)),
map, version_label)`, and reports the node in `Polled::installed`. The refit
moves no index (pending point edits close up when the job folds the overlay
in, and the fold's row map in `map` carries the selection across that), so the
selection stays on its point, and what the panels cached about each point's
geometry, keypoints and bitmap is dropped off `Polled::installed`, as after a
retriangulation. The
bench is not touched: a bench track whose origin is a refit point still
resolves, and committing it replaces the refit point.

**Undo** (Ctrl+Z) steps back to the version before, which holds every point's
keypoints, bitmap and geometry byte for byte; Redo steps forward again. That is
what the history already does for a bulk edit, and nothing here adds to it.

**The entry** says what the run did, each count read off the statuses, a count
of zero left out, in the shape `Retriangulate All Points` uses:

```text
Refit seoul_bull: 815 of 840 points refit, 3 of them ranged at their distance,
3 crossed to or from infinity, 41 observations kept their keypoint past the
shift bound, 9 left as they were (5 with too few rays, 4 with no bitmap to
render), 16 held, median shift 0.0021 (v3 → v4)
```

A run that released a constraint adds *"1 ranged point released: its
reference image has no pose and it became a direction"* after the ranged
count, so a reviewer sees which statement of theirs the run ended.

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
  `bench::track_of_point(edited, point, origin: Option<Origin>)`, which `create_track` calls and
  then puts on the bench. Public, because Python's batch callers have the same
  use for it.
- **The commit's record without the edit.** The part of `commit` that validates
  the track and builds its `PointRecord` becomes
  `bench::commit::point_record(edited, track) -> Result<PointRecord,
  CommitError>`, crate-visible, which `commit` calls before its
  `replace_point` / `add_point`. So the batch and the commit write one record
  for one track.

**How the batch feeds them.** Every per-point step takes an
`EditedReconstruction`, and the batch hands each one the wrapper of its copy
(§ "Why it is shaped this way"). None of them reads an overlay, so a wrapper
with an empty overlay gives them all they read:

- `track_of_point` reads the point through `EditedReconstruction::point`,
  whose `PointView` is a row of the base when the overlay is empty: frame,
  bitmap, colour, keypoints, readings, confidences, reference observation,
  `display_only_reference` and the point's `w`. The `PointView` has no
  constructor outside `edited.rs`, so a version taking `&SfmrReconstruction`
  would need a new accessor for no gain.
- `point_record` reads only the base: which columns it carries
  (`has_feature_indexes`, `has_patch_frames`, `has_patch_bitmaps` and the
  rest, each a test of `edited.base`), `image_count` and
  `observation_reading_options`. The batch's `unchanged` test reads the stored
  record through `point(index).to_record()`, as `commit` does.
- `fit_track` and what it calls read only the base and its two cached
  measures: `check_views` (`image_count`), `FitOptions::at_patch_resolution`
  and `bitmap_layout` (`edited.base`'s bitmap column), `noise_level`
  (`base_reprojection_noise_px`, not reached, since the batch passes
  `sigma_px`), `classify` (`base_min_point_depth`, measured on the first
  point's fit and kept by the wrapper for the rest) and `evaluate`
  (`image_count`, the base's patch resolution).
- Nothing in the three steps reads `base_content_hash`; only `create_track`'s
  label does, and `track_of_point` leaves the label out.

So the two split-out functions keep the `&EditedReconstruction` argument the
bench's other functions take, and no per-point step needs overlay state that a
plain base cannot give.

The ranged rule needs two more changes to existing code, neither of which
changes what any current caller gets:

- **The fit takes a range.** `fit_track` in
  [`bench/fit.rs`](../../crates/sfmtool-core/src/bench/fit.rs), the
  bench-private body that `fit` and the cluster upgrade (`set_stage`) share,
  gains an argument `range: Option<PointDistance>`. With `None` it runs as it
  does today, and both existing callers pass `None`. With an entry, the step
  that turns the fitted keypoints into a coordinate runs the distance rule
  (`triangulate_points_through_cameras` with `PointRules::distance` set) in
  place of `classify`, and `FitReport::classification` is `None`. The
  keypoints are still triangulated free first (`triangulate_keypoints`), so a
  ranged point is refused for too few rays exactly as a free one is, and the
  report keeps its condition number. `refit_points` calls `fit_track` the way
  `fit` calls it at the track stage: after `check_views` and
  `fit_preconditions`, with the payload's own frame and the options at the
  value's patch resolution (`FitOptions::at_patch_resolution`, a private method
  of `fit.rs` today, which becomes `pub(super)` for it).
- **The origin rule is shared.** The part of `ranged_entry` in
  [`retriangulate.rs`](../../crates/sfmtool-core/src/reconstruction/triangulation/retriangulate.rs)
  that reads a reference image's camera centre, or nothing where the image is
  out of range or has no pose, becomes a crate-visible function of its own,
  which `ranged_entry` calls and the refit calls for a finite distance. The
  retriangulation's answers do not change.

The new code is the loop, the constraint rule, the report and the in-place
writer, all in `bench/refit_points.rs`. Python gets `sfmtool.bench.refit_points`,
taking an `sfmtool.reconstruction.SfmrReconstruction` and `images` as
`sfmtool.bench.fit` does, and returning the refit `SfmrReconstruction` and a
report dict with no point map, because every edit family's mechanism is bound
([`../gui/edits/README.md`](../gui/edits/README.md)). A caller holding an
`EditedReconstruction` folds it first with its `materialize`.

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
  point on the bench" and § "The commit", the range a fit can be given in
  § "Fitting", and the binding in § "Python bindings".
- [`../core/reconstruction/triangulation-rules.md`](../core/reconstruction/triangulation-rules.md)
  § "A direction read at a held distance": the refit as a second caller of the
  distance rule, and the origin rule shared with it.
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

- **A refit of one point equals the bench's.** For each free point of a small
  value, `refit_points`'s record equals the record from `create_track`, `fit`
  and `commit` on that point alone, column for column.
- **Keypoints nudged off the content come back.** On the seoul_bull ground truth
  with its photographs, keypoints displaced by a few pixels return to within
  the refine's tolerance of the stored ones, and the points move back toward
  their stored positions. The ground truth stores no reference observation
  (every row is `-1`), and the reference-view rule would pick a nudged row, so
  the test names each point's first observation as its reference and leaves
  that row where it is.
- **No index moves**, the point and observation counts are unchanged, and every
  track keeps its images.
- **A point at infinity** is fitted as a bearing and written with `w = 0`, a
  zero normal and a zero normal confidence; a bearing given a baseline crosses
  to finite and is counted.
- **Held points** are not fitted and come back byte for byte.
- **A ranged point at a finite distance** has its keypoints moved as a free
  point's are, comes back at exactly its distance from its reference image's
  camera centre with `w = 1`, keeps its constraint triple, and agrees with
  what `triangulate_points_through_cameras` under the distance rule gives for
  its fitted keypoints. On a value where the same point's free fit would land
  elsewhere, the two answers differ, so the test shows the range was applied.
- **A ranged point at an infinite distance** comes back with `w = 0`, the
  normalized mean of its fitted rays, and its triple, and is not counted as
  crossed even where its rays now have a baseline.
- **A ranged point whose reference image has no pose** is fitted free and keeps
  its triple while it stays finite; one whose fit makes it a direction is
  written free and counted as released.
- **`fit_track` with no range** gives the same answer as before the argument
  was added, for `fit` and for the cluster upgrade.
- **A point the fit refuses** (one observation; two whose rays are parallel are
  not a refusal, since the classification makes them a bearing) and **a point
  with no eligible reference** in a value that stores bitmaps come back byte
  for byte, with the status naming why.
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
- on a node with pending point edits, the job folds the overlay in before the
  refit and the selection follows its point through the fold's row map;
- a cancelled run pushes no version and writes the failed entry;
- a run with a photograph that cannot be read pushes no version and writes the
  failed entry naming the image;
- `the_prune_entry_is_live_on_a_node_with_patch_frames`, whose doc comment and
  failure message say Prune sits directly under `Retriangulate All Points`,
  measures Prune against the refit entry instead;
- the wire's `refit_all_points` defers to the worker and comes back with the
  version.

---

## Before conversion

**Measure the run time.** A bench fit of one track is localize, refine and a
full evaluation, so a run over a reconstruction of tens of thousands of points
may take minutes, and it holds every photograph decoded for its length. The
implementation measures the run on the seoul_bull and dino_dog_toy ground
truths, the wall time of each phase and the peak memory, before the edit spec
is filed, and records the numbers there.

---

## Non-goals

- **Moving a camera or a lens.** The poses and the cameras are the input, as for
  `Retriangulate All Points`; an answer that moved them would be an adjustment.
- **Changing which observations a point has.** Every verdict is pinned `in`, so
  no observation is turned out, added or searched for. Deciding which
  sightings a track should have is the bench's work.
- **Turning the patch.** The fit keeps the way the patch faces. *Fit Normal* and
  the finite-difference normals stay per-track bench steps.
- **Refitting a list of points.** The run is over every point, and one point is
  *Fit* and *Commit* on the bench. A `RetriangulateWhich`-like argument (`All`,
  `TheseInBulk`) can be added when a caller wants, for instance, every point
  one camera observes.
- **Refitting a held point.** The value owns its coordinate, so the run leaves
  it out (§ "Constraints"). Realigning a held point's keypoints and bitmap at the
  coordinate it holds would be a different operation from a fit, which places
  the point where its keypoints say.
