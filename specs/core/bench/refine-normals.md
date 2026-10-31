# Refining every point's normal

`bench::refine_normals` does to every finite point of a reconstruction, or to
the finite points of a list, what the bench does to one point when a person
puts it on the bench, presses *Grown Plane Normal* and presses *Commit*, with
every observation kept: one plane is fitted to the photographs the point is
seen in, the point's patch is turned about its centre to face along that
plane's normal, the stored bitmap is rendered again from the reference
observation and each observation's readings are taken again. The point's
centre does not move and no keypoint moves. It returns a new reconstruction
holding the answers, with every point at the index it had, and a report with
one status per point and, for each point turned, how well its views agree at
the plane. It moves no camera and no lens, and leaves points at infinity and
points not on the list as they are.

It is what a reviewer wants when the patches of a reconstruction face the wrong
way, for instance after a solve that stored each patch facing its cameras
rather than along its surface. The viewer runs it as **Refine All Normals** on a
reconstruction row's context menu
([`../../gui/edits/refine-all-normals.md`](../../gui/edits/refine-all-normals.md)),
and Python calls it as `sfmtool.bench.refine_normals` (§ "Python binding").
The commands that turn patch normals call that binding: `sfm xform
--refine-normals`, `inspect --strips`, `compare --strips` and each round of
`sfm embed-patches`
([`../../cli/reconstruction/embed-patches-command.md`](../../cli/reconstruction/embed-patches-command.md)
§ "Behaviour Notes").
The per-point step is the bench's `grown_plane_normal`
([`editable-track.md`](editable-track.md) § "Estimating the normal", designed
in [`../../drafts/grown-plane-normal.md`](../../drafts/grown-plane-normal.md)),
and the run is built as [`refit-points.md`](refit-points.md)'s refit of every
point is, sharing its preconditions and its writer.

## Rust API

The function, its options, its report and its error are in
[`bench/refine_normals.rs`](../../../crates/sfmtool-core/src/bench/refine_normals.rs)
and re-exported from `sfmtool_core::bench`.

```rust
/// How a whole-reconstruction normal refinement runs.
#[derive(Debug, Clone, Default)]
pub struct RefineNormalsOptions {
    /// The step each point gets; Track View's Grown Plane Normal runs the default.
    pub normal: GrownPlaneNormalOptions,
    /// The points to refine; None, the default, refines every finite point.
    pub point_indexes: Option<Vec<u32>>,
}

pub fn refine_normals(
    recon: &SfmrReconstruction,
    images: &[ProjectedImage<'_>],
    options: &RefineNormalsOptions,
    progress: &Progress<'_>,
) -> Result<(SfmrReconstruction, RefineNormalsReport), RefineNormalsError>;

pub struct RefineNormalsReport {
    pub points: Vec<RefinedNormal>, // one per point, in index order
}

pub struct RefinedNormal {
    pub index: u32, // in the input and in the returned value alike
    pub outcome: RefineNormalsOutcome,
}

pub enum RefineNormalsOutcome {
    Turned {
        turn_deg: f64,
        joined: usize,
        low_confidence: bool,
        unchanged: bool,
        agreement: f64,
    },
    LeftUnchanged(NormalLeftUnchanged),
    NotRefined(NormalError),
    NoBitmap,
    AtInfinity,
    NotAsked,
}

pub enum NormalLeftUnchanged {
    NoneJoined,
    WeakTurn { turn_deg: f64, agreement: f64 },
}

pub enum RefineNormalsError {
    NoKeypoints,
    NoPatchFrames,
    NoPosedImage,
    ViewsMissing { got: usize, expected: usize },
    ImageOutOfRange { point: u32, image: u32, image_count: usize },
    BadOption(&'static str),
    PointOutOfRange { point: u32, point_count: usize },
    Cancelled,
}
```

### Why it is shaped this way

It is shaped as `refit_points` is, for the reasons
[`refit-points.md`](refit-points.md) § "Why it is shaped this way" gives: it
takes the photographs decoded, one `ProjectedImage` per image, and reads no
file; it takes a plain `SfmrReconstruction`, so a caller holding an overlay
folds it first and no `PointMap` comes back; it borrows the value and writes
its answers into one copy, wrapped once for every point's steps. Unlike the
refit, it takes a list of the points to refine (below).

- **The options hold the step's `GrownPlaneNormalOptions`** rather than
  repeating its fields, so the refinement of a point is that step on that
  point. The default is the step's default, which is what Track View's *Grown
  Plane Normal* runs. They are checked once, up front
  (`GrownPlaneNormalOptions::check`), so a setting the step does not take
  refuses the run as `BadOption` rather than every point.
- **A list of points is an option, not a second function.** A caller that
  refines the points a command names (`inspect --strips`, `compare --strips`)
  gets the same answers for them as a run over every point, since no point's
  step reads another point's answer, and the report keeps one status per point
  of the value, so an index means the same point in the input, the output and
  the report. A point not listed is `NotAsked` and keeps every column it had.
  An index past the last point refuses the run (`PointOutOfRange`) before
  anything is copied, rather than being dropped, since a caller that names a
  point it does not hold has its indexes wrong. A point listed more than once
  is refined once.
- **No noise level is measured.** The step and the reading it ends with weight
  no rays, so there is no `measure noise` stage and no `NoNoiseLevel` refusal,
  which the refit has.
- **It is a sibling of `refit_points`, not a mode of it.** The two share the
  whole-run refusals and the write, and differ in what runs on each point and
  in the rules for constrained points and points at infinity. One function with
  a switch would carry a report whose fields half apply to each mode.
- **`NormalLeftUnchanged` is its own enum** because the step reports two ways
  of running to its end and keeping the normal, and a reviewer reads them
  differently: no view joined says the photographs gave no plane, a large turn
  on weak evidence says they gave one the step would not trust.

### Example

```rust
use sfmtool_core::bench::{refine_normals, RefineNormalsOptions};
use sfmtool_core::progress::Progress;

let (next, report) =
    refine_normals(&recon, &views, &RefineNormalsOptions::default(), &Progress::none())?;
println!(
    "{} of {} finite points turned, {} at infinity skipped",
    report.turned(),
    report.read(),
    report.at_infinity(),
);
```

### The report

`RefineNormalsOutcome` says what the run did to one point:

- `Turned { .. }`: the step turned the patch, and the record is the answer.
  `turn_deg` is the turn the patch was given (the tilt the step reached, which
  a view's tilt cap can stop short of the plane), `joined` the views that
  joined the plane, and `low_confidence` the step's flag that the plane's weakest
  `σ`, with each view counted once, is over its bar. `unchanged` says the
  record agrees with the stored one (`PointRecord::agrees_with`) and the stored
  reference is not one only the display render picked, so the point was not
  rewritten. `agreement` is how well the views agree at the plane (below).
- `LeftUnchanged(NoneJoined)`: no view joined the plane, or the reference-view
  rule found no reference, so there is no plane and no agreement. The point
  keeps every column it had.
- `LeftUnchanged(WeakTurn { turn_deg, agreement })`: a view joined, but the
  turn to the plane is large on weak evidence, so the step kept the normal.
  `agreement` is the joined views' at the plane found and not turned to, so it
  says how well they agree on that plane, not on the normal the point keeps.
  The point keeps every column it had.
- `NotRefined(NormalError)`: the step refused the point (fewer than two
  observations, for one), and it keeps every column it had.
- `NoBitmap`: the step turned the patch but rendered no bitmap, in a value that
  stores one per point, so there is no record to write and the point keeps
  every column it had.
- `AtInfinity`: the point is stored with `w = 0`, and was skipped.
- `NotAsked`: the run was given a list of points
  (`RefineNormalsOptions::point_indexes`) and the point is not on it. The run
  did not look at it, and it keeps every column it had. A point at infinity
  not on the list is `NotAsked`.

The counts are read off the statuses: `read` (every status but `AtInfinity`
and `NotAsked`), `turned`, `unchanged`, `low_confidence` (of the turned
points), `none_joined`, `weak_turn`, `kept` (`NotRefined` and `NoBitmap`),
`at_infinity`, `not_asked`, `with_error(is)`, `median_turn_deg` (`NaN` where
none was turned) and `mean_agreement`.

**The agreement** of a point is the mean robust grid ZNCC (`grid_zncc`) of
the views that joined the plane (`GrownPlaneView::Joined`) or had their shift
alone fitted at it (`GrownPlaneView::ShiftOnly`), each read against the
reference at the final plane, as `GrownPlaneSequence::agreement` computes it.
A view whose score is `NaN` is left out, and the agreement is `NaN` when no
view has a score. The reference view has no `grid_zncc` of its own, since it
is what every other view is scored against, so it does not count; a view left
out does not count either, since it is not in the plane. A shift-only view
does count, its score under the floor or not: it is a view of the point the
plane has to explain, and leaving the ones that agree badly out would flatter
the plane. The step computes no score at the starting normal, so there is no
before-and-after difference. Under `WeakTurn` no view has its shift fitted,
so the agreement is the joined views' alone.

`mean_agreement` is the mean of the turned points' agreements, a `NaN` one
left out, and `NaN` where no turned point has one. A `WeakTurn` point does
not count: its agreement is at a plane the run did not write, and the mean is
meant to describe the normals the run wrote.

`RefineNormalsError` refuses the run as a whole, before anything is copied:
`NoKeypoints`, `NoPatchFrames`, `NoPosedImage`, `ViewsMissing` and
`ImageOutOfRange` are the refit's own checks, run by the same function
([`refit-points.md`](refit-points.md) § "The report"); `BadOption` names the
setting the step does not take; `PointOutOfRange` names an index of the list
past the last point; `Cancelled` says the run was asked to stop.

## What one point's refinement is

For each point the run does what the bench does when the point is put on the
bench, *Grown Plane Normal* is pressed and *Commit* is pressed, and nothing
else:

1. **The track.** `bench::track_of_point` builds the point's track-stage
   `EditableTrack` as for the refit: its own patch frame, bitmap, colour,
   keypoints and stored readings, its stored reference observation as the
   track's reference, and every observation `in` and pinned.
2. **The step.** `grown_plane_normal` with the run's options
   ([`editable-track.md`](editable-track.md) § "Estimating the normal"). It
   fits one plane through the patch's centre to the photographs, view by view,
   and when a view joined and the turn is not large on weak evidence it turns
   the patch to that plane by the least rotation about its centre, renders the
   stored bitmap again from the reference observation, and reads the track
   back: every row's tile, its readings and its score against the new bitmap.
   It writes no keypoint: the turn keeps each keypoint's gap from the centre's
   projection, and the image shifts the plane fit solves for are reported, not
   written. It trusts the keypoints it is given, so on a reconstruction whose
   keypoints may be off, a refit comes first; a refit after the run
   (`refit_points`, the viewer's *Refit All Points*) relocalizes the keypoints
   against the turned patches.
3. **The record.** `bench::commit::point_record` builds the `PointRecord` the
   commit writes for the turned track, as for the refit: the stored position,
   the turned frame, the new bitmap and its reference observation, the colour
   at the bitmap's centre, the root mean square of the evaluated reprojection
   errors, the normal the frame states, and per observation, in image order,
   the keypoint, its blur-matched score and its readings. The record's
   constraint is free, as the commit writes it, and the run replaces it with
   the point's own (§ "Constraints").

Where the step ran and kept the normal (`LeftUnchanged`), the run writes
nothing for the point, although the step read its track back: the readings it
took at the stored normal are not what was asked for, and a point the run did
not turn keeps its columns as a point the refit cannot place keeps its own.

### Points at infinity

A point stored with `w = 0` is skipped before its track is built, counted as
`AtInfinity`, and keeps every column it had. Its patch frame is tangent to the
direction sphere and faces along its own bearing, which every normal step
refuses (`NormalError::AtInfinity`), so there is no plane to fit; the run says
so per point rather than reporting a refusal for each. Nothing in the run
divides by `w`, and the one number it derives, `turn_deg`, is the step's.

### Constraints

**Every finite point asked for is refined, held and ranged points included,
and keeps its constraint.** The refit leaves a held point out because the
fit places the point where its keypoints say and a held point's coordinate is
the reviewer's; it reads a ranged point's direction at its distance for the
same reason ([`refit-points.md`](refit-points.md) § "Constraints"). The
grown-plane step moves no centre: the position written is the stored one, bit
for bit, so a held coordinate and a ranged distance both still describe the
point. The record carries the point's stored constraint triple, so the point
is written held or ranged as it was. The step writes no keypoint either, so
the point's rays say what they said before the run.

A ranged point at an infinite distance is stored with `w = 0`, so it is
skipped as a point at infinity.

## Writing the answers

The run writes every record with `refit_points`' writer, into the copy it took
at its start, once every point's step has run
([`refit-points.md`](refit-points.md) § "Writing the answers"): each record at
its own row, the observations over the stored rows they came from, the columns
the value has and the readings, a readings column added under the options of
the first record that brings a measured reading, a display-marked bitmap
column kept marked, and nothing written for a point not rewritten. No point and
no observation row is added, removed or renumbered.

## The run

### Stages and progress

```text
refine normals       counts "k of n points" and sets its fraction
write back
```

Each point's step is called with `&Progress::none()`, for the reason the
refit's fits are ([`refit-points.md`](refit-points.md) § "Stages and
progress"); the loop's phase reports the count and the fraction from an atomic
counter. With a list of points, `n` is the number of points on it, each
counted once, and a point not on it is not counted.

### Cancellation

The flag is read in front of every stage and in front of every point's step; a
point whose step has begun runs to its end, so a cancel lands within one
point's step per worker thread, about 30 to 80 ms on the ground truths below. A
cancelled run returns `RefineNormalsError::Cancelled` and writes nothing, the
refit's rule.

### Parallelism and determinism

The points are refined in parallel over the rayon pool, one point per task,
with the results collected by point index. Every point's step reads only the
value the run was handed and the photographs, never another point's answer, so
the same value and photographs give the same output bit for bit for any thread
count.

A refinement is not idempotent: a second run starts each step from the normal
the first wrote. It starts from the same keypoints, since no run moves one.

### Run time

Measured in a release build with the default options, on an Intel Core
i9-14900HX (32 logical processors), on copies of the two checked-in ground
truths with embedded patches, by a test that loaded each file, decoded its
photographs through `PhotographCache` and called `refine_normals` on the
decoded views. The 32-thread time is the median of three runs; the one-thread
time (a one-thread rayon pool) is one run.

| | seoul_bull ground truth | kerry_park ground truth |
|---|---|---|
| Images | 17 at 270 × 480 | 48 at 480 × 480 |
| Points (at infinity) | 280 (14) | 391 (9) |
| Observations | 1277 | 3767 |
| Decode | 0.02 s | 0.02 s |
| Refinement, 32 threads | 0.44 s | 2.01 s |
| Refinement, 1 thread | 7.95 s | 30.0 s |
| Points turned, of the finite points; with low confidence | 258 of 266; 168 | 360 of 382; 172 |
| Normals left unchanged: no view joined; large turn on weak evidence | 2; 6 | 4; 18 |
| Median turn | 17.0° | 5.2° |

A refinement costs about 6 to 8 ms of one thread per observation, about six
times a refit's, and runs about 15 to 18 times faster on the 32-thread pool. A
capture of 250 000 observations would take on the order of two minutes on this
machine, on top of decoding its photographs; that figure is an extrapolation,
since no checked-in reconstruction that size carries embedded patches.

## Python binding

`sfmtool.bench.refine_normals(recon, images, *, point_indexes=None,
working_resolution=None, ..., depth_likelihood_ratio_threshold=None,
progress=None)`
([`bench/refine_normals.rs`](../../../crates/sfmtool-py/src/bench/refine_normals.rs))
takes an `sfmtool.reconstruction.SfmrReconstruction` and `images` as
`sfmtool.bench.refit_points` does (one decoded image per image, or an
`ImagePyramidSet`), `point_indexes` as `RefineNormalsOptions::point_indexes`,
and `sfmtool.bench.grown_plane_normal`'s keyword arguments, each `None` for the
Rust default, which make up `RefineNormalsOptions::normal`. A caller holding an
`EditedReconstruction` folds it first with its `materialize`. A setting the
step does not take is refused before the photographs are decoded. It runs with
the GIL released and raises `ValueError` with the refusal's sentence.
`progress`, an `sfmtool.ProgressCounter`, is set from the `refine normals`
stage's count, so it climbs to that stage's `n` (§ "Stages and progress"):
the points listed, each counted once, or every point, those at infinity
included.

It returns the new `SfmrReconstruction` and a report dict, as `refit_points`'
binding does ([`refit-points.md`](refit-points.md) § "Python binding"), with
the counts `read`, `turned`, `unchanged`, `low_confidence`, `none_joined`,
`weak_turn`, `kept`, `at_infinity` and `not_asked`, `median_turn_deg`,
`mean_agreement`, and `points`, one dict per point in index order with its
`index` and `outcome`: `"turned"`, `"none_joined"`, `"weak_turn"`,
`"not_refined"`, `"no_bitmap"`, `"at_infinity"` or `"not_asked"`. The two
kinds of `LeftUnchanged` are outcomes of their own, named as the grown plane
normal's report names its outcomes. A turned point's dict adds `turn_deg`,
`joined`, `low_confidence`, `unchanged` and `agreement`; a `"weak_turn"` one
adds `turn_deg` and `agreement`; a refused one adds `error`, the
`NormalError`'s name in snake case, and `reason`, its sentence. There is no
point map, since no index moves.

## Testing

[`bench/tests/refine_normals.rs`](../../../crates/sfmtool-core/src/bench/tests/refine_normals.rs),
on the grown plane normal's synthetic textured plane seen from six views out to
50 degrees:

- **A refinement of one point equals the bench's**: three points whose patches
  start 25 degrees off the plane are each turned to within 2 degrees of it,
  their centres unchanged bit for bit, and each record written agrees with the
  one `create_track`, `grown_plane_normal` and `commit` write for that point
  alone, the keypoints and the rendered bitmap included; no index moves, and
  every keypoint is the input's.
- **Keypoints off their content stay put**: with every keypoint but each
  point's first pushed 0.6 px and −0.4 px off its content, the three points
  are still turned more than 20 degrees, every keypoint comes back as given,
  and a second run on the first run's output moves none of them either.
- **A point at infinity** is skipped, counted, and comes back as it was.
- **Held and ranged points** are turned and keep their constraint triple and
  their position.
- **A flat wall** joins no view, so every point is left unchanged, comes back
  as it was, its keypoints included, and no readings column is added.
- **A point with one observation** is refused with `TooFewObservations` and
  comes back as it was.
- **The run's stages** are `refine normals` and `write back`, and the count
  reaches every point; **cancel** before the loop and part way through returns
  `Cancelled` and leaves the input untouched.
- **One thread and the default pool** give the same value.
- **The refusals**: `sift_files`, no frames, no posed image, too few views, an
  option the step does not take, a listed index past the last point.
- **A list of points**: with point 1 at infinity, the list `[2, 1, 2]` refines
  point 2 once, to the record and status a run over every point gives it,
  reports point 1 `AtInfinity` and point 0 `NotAsked`, and point 0 keeps its
  record bit for bit, its patch frame unturned, apart from the readings column
  the write added, which holds no reading for it. An empty list refines
  nothing and returns the value unchanged.
- **The agreement**: each of the three points turned on the textured plane has
  an agreement over 0.5 and at most 1, and `mean_agreement` is their mean; on
  the flat wall no point is turned and `mean_agreement` is `NaN`.
  `GrownPlaneSequence::agreement`'s own tests, in
  [`bench/tests/grown_plane_normal.rs`](../../../crates/sfmtool-core/src/bench/tests/grown_plane_normal.rs),
  check that it averages the joined and shift-only views, the reference, the
  views left out and a `NaN` score not counted, that it is `NaN` with no score,
  and that it is the joined views' alone under `WeakTurn`.

[`tests/rust_bindings/bench/test_refine_normals_rust_bindings.py`](../../../tests/rust_bindings/bench/test_refine_normals_rust_bindings.py)
checks the binding on the seoul_bull ground truth: no position, keypoint or
index moves, only the rewritten points' normals change, and the input is
unchanged; the report's statuses and counts; each turned point's agreement is
finite and `mean_agreement` is their mean; a list of points gets the statuses a
run over every point gives them, and the other points keep their normals and
are `"not_asked"`; the refusals; a refused point's `error` and `reason`; and the
progress counter.

## Non-goals

- **Moving a point, a camera or a lens.** The centre is kept and the poses and
  the cameras are the input; moving the points is
  [`refit-points.md`](refit-points.md)'s refit.
- **Changing which observations a point has.** Every verdict is pinned `in`.
- **Choosing the normal step.** The run uses the grown-plane normal only; the
  other normal steps are per-track bench steps.
