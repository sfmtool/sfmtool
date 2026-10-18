# Refitting every point

`bench::refit_points` does to every point of a reconstruction what the bench
does to one point when a person puts it on the bench, presses *Fit* and presses
*Commit*, with every observation kept: the point's other observations are
aligned to the render of its reference observation and their keypoints
rewritten, the point is re-triangulated (or, for a point at infinity, its
direction is re-estimated), its patch is placed at the answer and its stored
bitmap is rendered again from the reference observation, and each observation's
readings are taken again against that bitmap. It returns a new reconstruction
holding the answers, with every point at the index it had, and a report with one
status per point. It moves no camera and no lens.

It is what a reviewer wants after camera moves or a bundle adjustment have left
the keypoints of many points a pixel or two from where the photographs put the
patch's content. `reconstruction::retriangulate_points` re-solves each point
from the keypoints it already has; this reads the keypoints again from the
photographs first. The viewer runs it as **Refit All Points** on a
reconstruction row's context menu
([`../../gui/edits/refit-all-points.md`](../../gui/edits/refit-all-points.md)),
and Python calls it as `sfmtool.bench.refit_points`. The per-point steps are the
bench's own, specified in [`editable-track.md`](editable-track.md).

## Rust API

The function, its options, its report and its error are in
[`bench/refit_points.rs`](../../../crates/sfmtool-core/src/bench/refit_points.rs)
and re-exported from `sfmtool_core::bench`.

```rust
/// How a whole-reconstruction refit runs.
#[derive(Debug, Clone)]
pub struct RefitPointsOptions {
    /// The fit each point gets. `sigma_px: None` is measured once for the
    /// whole run from the value it was handed, not once per point.
    pub fit: FitOptions,
    /// How far, in grid px, a fit may move a sighting before it keeps its
    /// keypoint instead: the bound `Thresholds::max_shift_px` sets for a track
    /// on the bench. `BENCH_MAX_SHIFT_PX` (6) by default.
    pub max_shift_px: f64,
}

impl Default for RefitPointsOptions { /* FitOptions::default(), BENCH_MAX_SHIFT_PX */ }

pub fn refit_points(
    recon: &SfmrReconstruction,
    images: &[ProjectedImage<'_>],
    options: &RefitPointsOptions,
    progress: &Progress<'_>,
) -> Result<(SfmrReconstruction, RefitPointsReport), RefitPointsError>;

pub struct RefitPointsReport {
    pub points: Vec<RefitPoint>, // one per point, in index order
}

pub struct RefitPoint {
    pub index: u32,          // in the input and in the returned value alike
    pub outcome: RefitOutcome,
}

pub enum RefitOutcome {
    Refit {
        placed: usize,
        kept_at_seed: usize,
        crossed: bool,
        shift: Option<f64>,
        classification: Option<TrackClassification>,
        range: RefitRange,
        unchanged: bool,
    },
    Held,
    NotFitted(FitError),
    NoBitmap,
}

pub enum RefitRange {
    Free,
    Applied,
    Unresolved { released: bool },
}

pub enum RefitPointsError {
    NoKeypoints,
    NoPatchFrames,
    NoPosedImage,
    ViewsMissing { got: usize, expected: usize },
    ImageOutOfRange { point: u32, image: u32, image_count: usize },
    NoNoiseLevel(String),
    Cancelled,
}
```

### Why it is shaped this way

- **It takes the photographs decoded, as [`fit`](../../../crates/sfmtool-core/src/bench/fit.rs)
  does.** `images` is one `ProjectedImage` per image of `recon`, indexed by
  image index, which is what every photometric step of the bench takes. Reading
  the files is the caller's: the viewer reads through its photograph cache on
  the worker, and Python resolves its `images` argument as `sfmtool.bench.fit`
  does. So this reads no file and holds no cache.
- **It takes a plain `SfmrReconstruction`, not an `EditedReconstruction`.**
  Given an overlay, the run's first step would be to fold it in
  (`EditedReconstruction::materialize`), so the fold belongs to the caller that
  has one. The refit moves no index, so with the fold at the caller there is no
  `PointMap` to return: the answer is the next value and one status per point,
  each addressed by the one index the input and the output share. That makes it
  a whole-reconstruction transform like `materialize_points_at_infinity` and the
  `sfm xform` operations, which a command or Python can call with no knowledge
  of edit overlays. `retriangulate_points` takes an `EditedReconstruction` and
  returns a `PointMap` because it has selection forms, which a refit does not
  (§ "Non-goals").
- **It borrows the value and returns a copy.** The per-point bench steps take
  an `EditedReconstruction`, so the run makes one copy of `recon`
  (`clone_for_edit`), wraps it once with `EditedReconstruction::new`, and hands
  that one wrapper to every point's steps. After the loop it drops the wrapper,
  takes the copy back out of the `Arc` (a move, since nothing else holds it) and
  writes the answers into it. The copy is the one the write needs in any case,
  so the wrapper costs no second copy.
- **There is no `which` argument.** The run is over every point, and the fit of
  one point is the bench's *Fit* and *Commit*.
- **The settings struct holds the per-point `FitOptions`** rather than
  repeating its fields, so a refit of a point is the fit of that point and the
  two cannot drift.
- **It lives in `bench`, not in `reconstruction`.** It is built from three bench
  steps, and `reconstruction` does not depend on `bench`.
- **`RefitPoints` prefixes the report, the error and the options**, because
  `RefitError` and `RefitOptions` already name `camera::refit_intrinsics`'s
  types.

### Example

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

### The report

`RefitOutcome` says what the run did to one point:

- `Refit { .. }`: the point was fitted and written. `placed` and `kept_at_seed`
  are the `FitReport`'s counts; `crossed` says it went from finite to a
  direction or back; `shift` is the distance a point that stayed finite moved,
  `None` otherwise; `classification` is the fit's `TrackClassification`, `None`
  for a point whose coordinate the distance rule decided; `range` says how the
  point's constraint was applied (§ "Constraints"); `unchanged` says the record
  it would write agrees with the stored one (`PointRecord::agrees_with`) and the
  stored reference is not one only the display render picked, so the point was
  not rewritten, the rule `commit` uses.
- `Held`: the value holds the point's coordinate, so it was left out.
- `NotFitted(FitError)`: the fit refused this point, and it keeps every column
  it had (§ "Points the fit cannot place").
- `NoBitmap`: the fit placed the point but rendered no bitmap, in a value that
  stores one per point, and the point keeps every column it had.

The counts are read off the statuses, as `RetriangulateReport`'s are: `read`
(every status but `Held`), `refit`, `unchanged`, `crossed`, `kept` (`NotFitted`
and `NoBitmap`), `held`, `ranged` (`RefitRange::Applied`), `released`
(`RefitRange::Unresolved { released: true }`), `kept_at_seed` (summed over the
refit points), `with_error(is)` (the points refused with a `FitError` the
predicate accepts) and `median_shift` (`NaN` where no point stayed finite).

`RefitPointsError` refuses the run as a whole, before anything is copied:
`NoKeypoints` (a `sift_files` value, which a commit refuses too), `NoPatchFrames`
(a fit registers the frame into every view), `NoPosedImage`, `ViewsMissing`,
`ImageOutOfRange` (a track names an image the value does not hold; a load,
`from_sfmr_data` and `clone_with_changes` refuse such a value
(`PointSet::validate_track_image_indexes`), so only Rust code that writes the
public `point_set.tracks` field can hand one in; refused up front, since the
fit reads every observation's camera centre), `NoNoiseLevel` (no `sigma_px` was
given and the value's reprojection noise cannot be measured, or the `sigma_px`
given is not a finite positive number) and `Cancelled`.

## What one point's refit is

For each point the run does what the bench does when the point is put on the
bench, *Fit* is pressed and *Commit* is pressed, and nothing else:

1. **The track.** `bench::track_of_point` builds the point's track-stage
   `EditableTrack` the way `create_track` builds the one it puts on the bench:
   its own patch frame, bitmap, colour, keypoints and stored readings, its stored
   reference observation as the track's reference, and every observation `in`
   and **pinned**. Nothing is recomputed, and no bench or label is involved
   ([`editable-track.md`](editable-track.md) § "Putting a point on the bench").
2. **The fit.** The bench's fit at the track stage (`fit_track`, the body
   [`fit`](../../../crates/sfmtool-core/src/bench/fit.rs) and the cluster
   upgrade share), with the run's `FitOptions` at the value's patch resolution
   and `max_shift_px` as the track's bound
   ([`editable-track.md`](editable-track.md) § "Fitting"):
   - every sighting is localized and refined against the **reference render**,
     the render of the reference observation at its own keypoint through the
     frame the point arrives with. Because every row is pinned, the reference
     held is the stored one where its row has a keypoint, and its keypoint is
     not moved;
   - a sighting the kernels would move further than `max_shift_px` keeps its
     keypoint, and one they did not place (out of frame) keeps its pixel;
   - the keypoints are classified and triangulated by the point-or-bearing test
     at the run's noise level, and the frame is placed at the answer at the
     apparent size it had. For a ranged point the coordinate comes from the
     distance rule instead (§ "Constraints"), and the frame is placed there in
     the same way;
   - the stored bitmap is rendered from the reference observation through the
     placed frame, with the colour at its centre;
   - the fit ends by evaluating its result: every row's tile is rendered again,
     its self-similarity reading, viewing angle, tilt direction and zoom are
     read, and it is scored against the new bitmap. The record's observation
     readings and `observation_confidence` are written from these.

   The alignment is against the reference render at the patch frame the point
   arrived with; the stored bitmap is then rendered once more through the patch frame
   the re-triangulation placed, so the bitmap written describes the patch that
   is written.
3. **The record.** `bench::commit::point_record` builds the `PointRecord` the
   commit writes for the fitted track ([`editable-track.md`](editable-track.md)
   § "The commit"): the fitted position with the classification's `w`, the
   placed frame, the new bitmap and its reference observation, the colour at
   the bitmap's centre, the mean of the evaluated reprojection errors in
   `error`, the normal the frame states (zero for a direction), and per
   observation, in image order, the new keypoint, its blur-matched score as
   `observation_confidence` and its readings. The record's constraint is free,
   as the commit writes it, and the run replaces it with the point's own
   (§ "Constraints"). The record is compared with the stored one and not
   applied: every record is written at once at the end (§ "Writing the
   answers").

The run adds the loop, the constraint rule, the per-point failure rule and the
in-place write. It sets no verdict and paints none: the thresholds never run,
because every row is pinned `in`.

### Points at infinity

A point stored with `w = 0` comes onto the track as a bearing, from the point's
own `w`, and the fit registers its frame as it is, tangent to the direction
sphere. The classification decides whether its keypoints now earn a depth: a
bearing whose rays have one becomes a finite point, and a finite point whose
rays no longer fix one becomes a bearing, each with the frame carried across at
the apparent size it had. The record is written with `w = 0`, a zero normal and
a `normal_confidence` of zero for a bearing, as the commit writes one. The run
never divides by `w`: it copies the record's columns, and its one derived
number, `shift`, is measured only for a point finite before and after, with
`crossed` counting the rest.

This differs from `retriangulate_points`, whose defaults turn the angular floor
and the likelihood rule off so that no free point crosses between the two
representations on its rays' account alone. A refit uses the fit's
classification, as *Fit* does, so a free point can cross in either direction.
A refit and *Fit* agree on every free point; a refit and a retriangulation can
disagree on which points are bearings. A ranged point whose distance the rule
applies does not cross: its `w` follows its distance.

### Constraints

A point the value **holds** or **ranges** is a statement a reviewer made by
hand, in the columns `PointRecord::constraint` carries as a `(kind, distance,
reference image)` triple.

**A held point is left out**: not fitted, not written, and reported as `Held`.
The value owns its coordinate, so the fit's answer could not be written.
`retriangulate_points` and `prune_covered_observations` leave held points alone
too.

**A ranged point is refit, and its range is applied to the answer.** The caller
owns the point's distance `r` from its reference image's camera centre `O`, and
the refit owns its direction `d`. Everything above runs as for a free point,
the alignment, the bitmap and the evaluation included, except the step that
turns the fitted keypoints into a coordinate:

- **The coordinate is the distance rule's**
  ([`../reconstruction/triangulation-rules.md`](../reconstruction/triangulation-rules.md)
  § "A direction read at a held distance"), the rule `retriangulate_points`
  applies to a ranged point: in place of the classification, the fitted
  keypoints of the `in` rows go through `triangulate_points_through_cameras`
  with `PointRules::distance` set. At a finite `r` the answer is `X = O + r·d`
  with `w = 1`; at `r = ∞` it is the normalized mean of the keypoints' rays,
  written with `w = 0`. So a ranged point's `w` is `0` exactly when its
  distance is infinite, the format's rule for a ranged row, and it never
  crosses. The keypoints are still triangulated free first, so a ranged point
  is refused for too few rays exactly as a free one is.
- **The origin is resolved as the retriangulation resolves it.** At a finite
  `r`, `O` is the reference image's camera centre at the value's poses, read by
  `reference_camera_center` in
  [`retriangulate.rs`](../../../crates/sfmtool-core/src/reconstruction/triangulation/retriangulate.rs),
  the one rule both callers use, resolved for every ranged point before the
  loop. A ranged point at `r = ∞` needs no origin, and its entry carries none,
  which the distance rule answers without reading.
- **The constraint is kept.** The record carries the point's stored triple, so
  the point is written ranged, at the same distance, from the same reference
  image, with `range: RefitRange::Applied`.
- **A ranged point whose finite distance has no origin** (its reference image
  has no pose) is fitted free, by the fit's classification, as
  `retriangulate_points` solves the same point free. Where the answer stays
  finite the stored triple is kept (`RefitRange::Unresolved { released: false
  }`). Where the classification makes it a direction, a finite distance no
  longer describes the coordinate and the format allows no `w = 0` row at a
  finite distance, so the constraint is released and the point is written free
  (`RefitRange::Unresolved { released: true }`), the format's rule for an edit
  that rewrites a point's representation
  ([`../../formats/sfmr-file-format.md`](../../formats/sfmr-file-format.md)
  § "Per-point constraints"). `retriangulate_points` treats such a point the
  same way: it keeps the stored triple where its free solve stays finite, and
  releases the constraint where the solve makes the point a direction.

A free point's status carries `range: RefitRange::Free`. Because a ranged
point's record carries the stored triple, one whose fitted record matches the
stored one column for column is `unchanged` and not rewritten.

**The bench's own *Fit* has no distance rule**, and its commit writes every point
free, so committing a ranged point from the bench releases its constraint. The
refit is the place that keeps the range.

### Points the fit cannot place

A point the fit refuses keeps every column it had, and its status names the
refusal: fewer than two observations (`TooFewObservations`), no patch frame on
this point (`NoPatchFrame`, possible where the column exists but the row is empty),
fewer than two usable rays (`TooFewRays`) or a linear solve that is not a number
(`Triangulation`), or a round's tiles past `EvaluateOptions::max_cache_bytes`
(`TooLarge`) or an allocation that failed (`OutOfMemory`).

A sighting the kernels cannot place is not a failure of the point: it keeps its
keypoint, its ray still feeds the triangulation, and its readings say it was not
read. So a point some of whose observations fit and some do not is written with
the fitted ones moved and the others where they were, as *Fit* does on the
bench.

**A reference observation that cannot be rendered.** Where the stored reference
is held, the bitmap is that row's tile. Where the point has none (stored `-1`
with no display pick, or a reference row with no keypoint), the fit runs the
reference-view rule over the `in` sightings and the pick becomes the reference,
as on the bench. Where the rule finds no eligible sighting the fit renders no
bitmap: in a value that stores a bitmap per point the record cannot be written,
so the point keeps every column it had and its status is `NoBitmap`; in a value
that stores no bitmaps the point is written, with no reference observation.

A point the viewer's display render gave a reference (`display_only_reference`)
comes onto the track with that reference held, as on the bench, and the record
writes it as the point's own, so a save writes those references rather than
`-1`, as after a commit of the same point.

## Writing the answers

The run writes every answer into the copy it took at its start, once every
point has been fitted. The fits read the copy before anything is written to it.

- **Every record is written at its own row.** The observations a record carries
  are the point's own, all `in`, listed by image with ties in track order, as
  the commit sorts them. A stored track is in image order when it was read from
  a file, but a value edited in memory need not be (tracks replaced through
  `clone_with_changes` from Python keep the order they were given in). So the
  writer sorts the point's stored rows the same way and writes the record's
  `k`-th observation over the `k`-th of them, the row it came from, and writes
  the record's reference as the stored row it names. The point's track keeps
  its length, its order and its offsets; no point and no observation row is
  added, removed or renumbered. The writer asserts that each record's images are
  the stored rows' in that order.
- **A point not rewritten is not touched**: one whose status is not `Refit`, or
  whose record agrees with the stored one while its reference is not a
  display-only pick. The stored record is put in the record's order first, its
  reference renumbered with it, since `agrees_with` compares observations
  position by position. In a value with no readings column the stored rows
  read as "not measured", so a record that brings a measured reading does not
  agree and is rewritten, as the commit's own comparison decides.
- **The columns written are the ones the value has, and the readings**, under
  the commit's rules:
  `observation_confidence`, the normals, the normal confidences and the
  reference observations only where the value has the column, and the patch
  bitmap only where it stores bitmaps. A readings row is written only under
  the column's own options, and is "not measured" otherwise.
  **A value with no readings column gains one**, as a run of bench commits in
  point order adds it (`EditedReconstruction::push_record`): where a written
  record brings a measured reading, the column is created under the options of
  the first such record, every row "not measured", and the written records
  fill their rows. The rows the refit does not write (a held point, a point
  left as it was, a point not rewritten) stay "not measured". Every record of
  one run is read under the same options, those of the run's evaluation at the
  value's patch resolution, so no record's rows are dropped for standing under other
  options. Where no record brings a measured reading, no column is added. The
  readings are the one column the refit adds; it treats every other column as
  the value has it. The constraint columns, where the value has them, take the
  record's triple, which is the stored one except for a released point; a value
  with no constraint columns gains none. A written point's display-only mark is
  cleared, since its reference is now the record's own.
- **A bitmap column marked `patch_bitmaps_for_display`** (rendered for the
  viewer, not read from the file) is rewritten row by row and keeps that mark,
  so a save writes the columns the file had and no bitmap column the file
  lacked. Adding one to the file is `sfm xform --add-patch-bitmaps`.

`retriangulate_points`'s own in-place writer writes positions, frames and a
released constraint only, and the commit writes one record through
`replace_point`, an overlay edit that gives the point a new index. A batch of tens of thousands of `replace_point`
calls would renumber every point and rebuild the overlay's derived indexes once
per point, which is why the refit has a writer of its own.

## The run

### Stages and progress

```text
measure noise        the run's sigma_px, once, where none was given
fit points           counts "k of n points" and sets its fraction
write back
```

Where no `sigma_px` is given, the noise level is measured on `recon` itself
(`SfmrReconstruction::reprojection_noise_px`) before the copy is taken, and
passed to every point's fit as its `sigma_px`: one walk over the observations
per run, and none per point. The copy and its wrapper are made after this
stage, in front of the loop.

Each point's fit is called with `&Progress::none()`. A fit opens a phase for
each of its steps, and per-point phases from a parallel loop are the per-item
timing [`../../gui/operation-progress.md`](../../gui/operation-progress.md)
§ "Non-goals" refuses. The loop's own phase reports the count and the fraction
from an atomic counter instead.

### Cancellation

The flag is read in front of every stage and in front of every point's fit; a
point whose fit has begun runs to its end, so a cancel lands within one point's
fit per worker thread. A cancelled run returns `RefitPointsError::Cancelled` and
drops its copy before writing to it, so no partially refit value is returned:
a reconstruction in which some points were refit and others were not, chosen by
thread scheduling, is not an answer anybody asked for. This is
`retriangulate_points`' rule.

### Parallelism and determinism

The points are fitted in parallel over the rayon pool, one point per task, with
the results collected by point index; the fit's own kernels nest on the same
pool. Peak memory per point is bounded by `EvaluateOptions::max_cache_bytes` per
round, so the run's peak is that bound times the number of worker threads, on
top of the decoded photographs.

Every point's fit reads only the value the run was handed and the photographs,
never another point's answer, and the answers are written at the end. So the
answer depends on neither the order the points are visited in nor the thread
count, and the same value and photographs give the same output bit for bit.

A refit is not idempotent: a second run fits against the patches and bitmaps the
first wrote, and can move keypoints again by small amounts. The first refit of a
point the fit does not move is in general still a change, because the commit's
colour is read from the bitmap's centre and its `error` is the evaluated
reprojection error, neither carried from the stored point
([`editable-track.md`](editable-track.md) § "The commit").

### Run time

Measured through `sfmtool.bench.refit_points` in a release build with the
default options, on an Intel Core i9-14900HX (32 logical processors), on the two
checked-in ground truths with embedded patches. Decode is reading every
photograph and the pyramids are an `ImagePyramidSet` built from them, both timed
apart from the refit, which is then handed the set. The 32-thread refit is the
median of three runs; the one-thread refit (`RAYON_NUM_THREADS=1`) the first of
two, which agreed within 2%.

| | seoul_bull ground truth | kerry_park ground truth |
|---|---|---|
| Images | 17 at 270 × 480 | 48 at 480 × 480 |
| Points (at infinity) | 280 (0) | 391 (9) |
| Observations | 1277 | 3767 |
| Decode | 0.10 s | 0.17 s |
| Pyramids | 0.004 s | 0.013 s |
| Refit, 32 threads | 0.10 s | 0.40 s |
| Refit, 1 thread | 1.23 s | 4.94 s |
| Peak working set of the process | 99 MB | 169 MB |
| Points refit; observations kept at their keypoint | 280 of 280; 22 | 391 of 391; 34 |

A refit costs about 1 to 1.3 ms of one thread per observation, and runs about
twelve times faster on the 32-thread pool. A capture of 50 000 points and
250 000 observations would take on the order of 20 to 30 s on this machine, on
top of decoding its photographs; that figure is an extrapolation, since no
checked-in reconstruction that size carries embedded patches.

## Python binding

`sfmtool.bench.refit_points(recon, images, *, max_seed_offset_px=None,
max_cache_bytes=None, sigma_px=None, depth_likelihood_ratio_threshold=None,
max_shift_px=6)`
([`bench/refit_points.rs`](../../../crates/sfmtool-py/src/bench/refit_points.rs))
takes an `sfmtool.reconstruction.SfmrReconstruction` and `images` as
`sfmtool.bench.fit` does (one decoded image per image, or an
`ImagePyramidSet`), and the fit's keyword arguments. A caller holding an
`EditedReconstruction` folds it first with its `materialize`. It runs with the
GIL released and `Progress::none()`, and raises `ValueError` with the refusal's
sentence. It returns the refit `SfmrReconstruction` and a report dict with the
counts `read`, `refit`, `unchanged`, `crossed`, `kept`, `held`, `ranged`,
`released`, `kept_at_seed` and `median_shift`, and `points`, one dict per point
in index order with its `index` and `outcome` (`"refit"`, `"held"`,
`"not_fitted"` or `"no_bitmap"`). A refit point's dict adds `placed`,
`kept_at_seed`, `crossed`, `shift`, `classification` (as `fit` reports it, or
`None`), `range` (`"free"`, `"applied"` or `"unresolved"`), `released` and
`unchanged`; a refused one adds `error`, the `FitError`'s name in snake case,
and `reason`, its sentence. There is no point map, since no index moves.

## Testing

[`bench/tests/refit_points.rs`](../../../crates/sfmtool-core/src/bench/tests/refit_points.rs):

- **A refit of one point equals the bench's**: for each free point of a small
  value, the record written equals the one `create_track`, `fit` and `commit`
  write for that point alone, column for column, both in a value with a
  readings column and in one without, to which both add the column under the
  same options.
- **The readings column a refit adds** equals the one a run of commits of the
  same points in point order adds, and a held point's rows in it are "not
  measured".
- **No index moves**: the point and observation counts are unchanged and every
  track keeps its images.
- **Keypoints nudged off their content come back** on the seoul_bull ground
  truth with its photographs, to within the refine's tolerance, and the points
  move back toward their stored positions. The ground truth stores no reference
  observation, so the test names each point's first observation as its
  reference and leaves that row where it is.
- **A point at infinity** is refit as a bearing with `w = 0`, a zero normal and
  a zero normal confidence; a bearing whose rays now have a baseline crosses to
  a point and is counted.
- **A held point** is left out and comes back byte for byte.
- **A ranged point at a finite distance** comes back at exactly its distance
  from its reference image's camera centre, keeps its triple, agrees with the
  distance rule over its fitted keypoints, and differs from its free fit.
- **A ranged point at an infinite distance** stays a bearing at the mean of its
  rays and is not counted as crossed.
- **A ranged point whose reference image has no pose** is fitted free and keeps
  its triple while finite, and is released where it becomes a direction.
- **A point with one observation** and **a point with no eligible reference in
  a value with bitmaps** come back as they were, with the status naming why.
- **A track stored out of image order** is written back at its own rows.
- **Cancel** before the loop and part way through returns `Cancelled` and
  leaves the input untouched.
- **One thread and the default pool** give the same value.
- **The refusals**: `sift_files`, no frames, no posed image, too few views, a
  track naming an image past the table, no noise level.

[`tests/rust_bindings/bench/test_refit_points_rust_bindings.py`](../../../tests/rust_bindings/bench/test_refit_points_rust_bindings.py)
checks the binding on the seoul_bull ground truth: no index moves and the input
is unchanged, the report's statuses and counts, a held point, the refusals, a
refused point's `error` and `reason`, and a track stored out of image order.
From Python a track naming an image past the table cannot reach the run:
`clone_with_changes` refuses to build the value.

## Non-goals

- **Moving a camera or a lens.** The poses and the cameras are the input; an
  answer that moved them would be an adjustment.
- **Changing which observations a point has.** Every verdict is pinned `in`, so
  no observation is turned out, added or searched for. Deciding which sightings
  a track should have is the bench's work.
- **Turning the patch.** The fit keeps the way the patch faces. *Fit Normal* and
  the finite-difference normals are per-track bench steps.
- **Refitting a list of points.** The run is over every point; one point is
  *Fit* and *Commit* on the bench.
- **Refitting a held point.** The value owns its coordinate. Realigning a held
  point's keypoints and bitmap at the coordinate it holds would be a different
  operation from a fit, which places the point where its keypoints say.
