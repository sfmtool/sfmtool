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
points not on the list as they are. An option runs the pairwise-flow normal
as the per-point step instead (§ "Choosing the step").

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
([`editable-track.md`](editable-track.md) § "Estimating the normal",
specified in [`grown-plane-normal.md`](grown-plane-normal.md)), or, where the
options ask for it, `pairwise_flow_normal`
([`pairwise-flow-normal.md`](pairwise-flow-normal.md)),
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
    /// The step each point gets, with its options; the grown plane by default.
    pub step: RefineNormalsStep,
    /// The points to refine; None, the default, refines every finite point.
    pub point_indexes: Option<Vec<u32>>,
}

pub enum RefineNormalsStep {
    GrownPlane(GrownPlaneNormalOptions),     // the default, at its defaults
    PairwiseFlow(PairwiseFlowNormalOptions),
}

impl RefineNormalsStep {
    pub fn check(&self) -> Result<(), NormalError>; // the step's own check
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
        unchanged: bool,
        by: NormalTurned,
    },
    LeftUnchanged(NormalLeftUnchanged),
    NotRefined(NormalError),
    NoBitmap,
    AtInfinity,
    NotAsked,
}

pub enum NormalTurned {
    GrownPlane { joined: usize, low_confidence: bool, agreement: f64 },
    PairwiseFlow { single_pair: bool, chosen_radius: f64, sigma_deg: f64 },
}

pub enum NormalLeftUnchanged {
    NoneJoined,                                 // the grown plane's
    WeakTurn { turn_deg: f64, agreement: f64 }, // the grown plane's
    Undetermined,                               // the pairwise-flow normal's
    Unstable,                                   // the pairwise-flow normal's
}

pub enum RefineNormalsError {
    NoKeypoints,
    NoPatchFrames,
    NoPosedImage,
    ViewsMissing { got: usize, expected: usize },
    ImageOutOfRange { point: u32, image: u32, image_count: usize },
    BadOption(NormalError),
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

- **The options hold the step's own options** rather than repeating their
  fields, so the refinement of a point is that step on that point. The step
  is an enum, `RefineNormalsStep`, each variant carrying the options of the
  step it names, so a run carries only the settings of the step it runs and
  no setting of the other step can be set and then ignored. The default is
  the grown plane at its defaults, which is what Track View's *Grown Plane
  Normal* runs, so a caller that does not choose gets what it got before the
  choice existed. The options are checked once, up front
  (`RefineNormalsStep::check`, the step's own check), so a setting the step
  does not take refuses the run as `BadOption`, carrying the step's
  `NormalError`, rather than every point.
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
- **`NormalLeftUnchanged` is its own enum** because each step reports two ways
  of running to its end and keeping the normal, and a reviewer reads them
  differently: for the grown plane, no view joined says the photographs gave
  no plane, a large turn on weak evidence says they gave one the step would
  not trust; for the pairwise-flow normal, undetermined says the photographs
  carry too little evidence about the tilt, unstable that the evidence does
  not settle (the glossary's **turned** / **undetermined** / **unstable**).
- **`NormalTurned` carries what each step found** where it turned the patch.
  The turn and whether the record was rewritten mean the same for either
  step, so they are `Turned`'s own fields; the views that joined, the low
  confidence flag and the agreement are the grown plane's numbers, and the
  single pair, the chosen radius and the `σ` the pairwise-flow normal's, so
  each sits under its step rather than being filled with a placeholder for
  the other. A single pair is a turn, not an outcome of its own here: it
  moves the normal and writes the record as a turn does, so it is counted
  among the turned points, with `single_pair` saying which.

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

### Choosing the step

`RefineNormalsOptions::step` is `RefineNormalsStep::GrownPlane` by default.
`RefineNormalsStep::PairwiseFlow` runs `pairwise_flow_normal` on each point
instead, with the options it carries; the rest of the run is the same: the
track, the record, the skip of points at infinity, the constraints, the
write, the stages and the cancellation. The pairwise-flow normal costs far
more per point, a dense optical flow per pair of views per round: a median
of 1.3 s of one thread per point on kerry_park and 4.4 s on dino_dog_toy
([`pairwise-flow-normal.md`](pairwise-flow-normal.md) § "Implementation
notes"), against 6 to 8 ms per observation for the grown plane (§ "Run
time"). The viewer's *Refine All Normals* and `sfm xform
--refine-normals` run the grown plane; only the Rust option and the Python
binding's `method` choose the other step.

### The report

`RefineNormalsOutcome` says what the run did to one point:

- `Turned { .. }`: the step turned the patch, and the record is the answer.
  `turn_deg` is the turn the patch was given (the tilt the step reached, which
  a view's tilt cap can stop short of the normal it found). `unchanged` says
  the record agrees with the stored one (`PointRecord::agrees_with`) and the
  stored reference is not one only the display render picked, so the point
  was not rewritten. `by` is what the step found:
  - `NormalTurned::GrownPlane`: `joined` the views that joined the plane,
    `low_confidence` the step's flag that the plane's weakest `σ`, with each
    view counted once, is over its bar, and `agreement` how well the views
    agree at the plane (below).
  - `NormalTurned::PairwiseFlow`: `single_pair` says the point has one pair,
    read with no agreement to test (`PairwiseFlowOutcome::SinglePair`; a
    `Turned` outcome otherwise), `chosen_radius` the radius the answer was
    read at, in patch half-extents, and `sigma_deg` its `σ` along its largest
    axis.
- `LeftUnchanged(NoneJoined)`: no view joined the plane, or the reference-view
  rule found no reference, so there is no plane and no agreement. The point
  keeps every column it had.
- `LeftUnchanged(WeakTurn { turn_deg, agreement })`: a view joined, but the
  turn to the plane is large on weak evidence, so the step kept the normal.
  `agreement` is the joined views' at the plane found and not turned to, so it
  says how well they agree on that plane, not on the normal the point keeps.
  The point keeps every column it had.
- `LeftUnchanged(Undetermined)`: the pairwise-flow normal found too little
  evidence about the tilt (`PairwiseFlowOutcome::Undetermined`). The point
  keeps every column it had.
- `LeftUnchanged(Unstable)`: the pairwise-flow normal's pairs are precise but
  disagree at every radius, or its answer is too far from the mean viewing
  direction (`PairwiseFlowOutcome::Unstable`). The point keeps every column
  it had.
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
and `NotAsked`), `turned`, `unchanged`, `low_confidence` (of the points the
grown plane turned), `single_pair` (of the points the pairwise-flow normal
turned), `none_joined`, `weak_turn`, `undetermined`, `unstable`, `kept`
(`NotRefined` and `NoBitmap`), `at_infinity`, `not_asked`, `with_error(is)`,
`median_turn_deg` (over every turned point, either step's; `NaN` where none
was turned) and `mean_agreement`.

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
meant to describe the normals the run wrote. The pairwise-flow normal scores
no agreement, so a run of it has a `NaN` mean.

`RefineNormalsError` refuses the run as a whole, before anything is copied:
`NoKeypoints`, `NoPatchFrames`, `NoPosedImage`, `ViewsMissing` and
`ImageOutOfRange` are the refit's own checks, run by the same function
([`refit-points.md`](refit-points.md) § "The report"); `BadOption` carries the
step's check's `NormalError`, which names the setting the step does not take
(`BadGrownPlaneOption`, `BadPairwiseFlowOption`, or `Flow` for the
pairwise-flow normal's flow settings); `PointOutOfRange` names an index of
the list past the last point; `Cancelled` says the run was asked to stop.

## What one point's refinement is

For each point the run does what the bench does when the point is put on the
bench, *Grown Plane Normal* is pressed and *Commit* is pressed, and nothing
else. With the pairwise-flow normal as the step, step 2 is
`pairwise_flow_normal` instead, which also turns the patch about its centre,
renders the bitmap again and reads the track back, writing no keypoint
([`pairwise-flow-normal.md`](pairwise-flow-normal.md)):

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
   The bitmap is the render of the turned frame as the value stores it
   ([reference-view.md](../patch/reference-view.md#the-stored-bitmap)
   § "The stored bitmap").
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
nothing for the point, although the step read its track back (the
pairwise-flow normal reads it back at the normal it was given under
`Undetermined` and `Unstable`): the readings it took at the stored normal
are not what was asked for, and a point the run did not turn keeps its
columns as a point the refit cannot place keeps its own.

### Points at infinity

A point stored with `w = 0` is skipped before its track is built, counted as
`AtInfinity`, and keeps every column it had, whichever the step. Its patch
frame is tangent to the direction sphere and faces along its own bearing,
which every normal step refuses (`NormalError::AtInfinity`), so there is no
plane to fit; the run says so per point rather than reporting a refusal for
each. Nothing in the run divides by `w`, and the one number it derives,
`turn_deg`, is the step's.

### Constraints

**Every finite point asked for is refined, held and ranged points included,
and keeps its constraint.** The refit leaves a held point out because the
fit places the point where its keypoints say and a held point's coordinate is
the reviewer's; it reads a ranged point's direction at its distance for the
same reason ([`refit-points.md`](refit-points.md) § "Constraints"). Neither
step moves a centre: the position written is the stored one, bit
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

Each point's step is called with a `Progress` that carries the run's cancel
flag and drops every event it reports (`Progress::tapped` with a sink that
does nothing): a step opens phases, and per-point phases from a parallel loop
are what the refit keeps out of its report for the reason
[`refit-points.md`](refit-points.md) § "Stages and progress" gives. The
loop's phase reports the count and the fraction from an atomic counter. With
a list of points, `n` is the number of points on it, each counted once, and a
point not on it is not counted.

### Cancellation

The flag is read in front of every stage and in front of every point's step,
and the step reads it too, at points inside its work (the pairwise-flow
normal before each pair's flow and each radius of a pair's fits). A step that
stops returns `NormalError::Cancelled`, which the run takes
as its own cancellation, not as a refusal of the point. A cancel therefore
lands within a part of one point's step per worker thread rather than after
a whole step, which for the pairwise-flow normal takes seconds. A cancelled run
returns `RefineNormalsError::Cancelled` and writes nothing, the refit's
rule.

### Parallelism and determinism

The points are refined in parallel over the rayon pool, one point per task,
with the results collected by point index, for either step. The
pairwise-flow normal's own parallel work, its crops over views and its pairs,
runs on the same pool inside the point's task, and its answer does not depend
on the thread count either. Every point's step reads only the
value the run was handed and the photographs, never another point's answer, so
the same value and photographs give the same output bit for bit for any thread
count.

A refinement is not idempotent: a second run starts each step from the normal
the first wrote. It starts from the same keypoints, since no run moves one.
The grown-plane step's turned support, which lays its working grid at each
stage's plane, makes a second pass turn less: on seoul_bull v02 point 255,
passes that turned 15.6° and 3.9° on one grid turn 4.4° and 0.3°
([grown-plane-normal-measurements.md](grown-plane-normal-measurements.md#a-working-grid-turned-with-the-plane-2026-11-10)).

### Run time

Measured with the grown plane as the step, in a release build with the
default options of the time, before
the grown-plane step laid its working grid again at each stage's plane (the
step as `turned_support = false` runs it; the turned support's cost is in
[grown-plane-normal-measurements.md](grown-plane-normal-measurements.md#a-working-grid-turned-with-the-plane-2026-11-10)), on an Intel Core
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
resolution=None, working_resolution=None, ...,
depth_likelihood_ratio_threshold=None, progress=None)`
([`bench/refine_normals.rs`](../../../crates/sfmtool-py/src/bench/refine_normals.rs))
takes an `sfmtool.reconstruction.SfmrReconstruction` and `images` as
`sfmtool.bench.refit_points` does (one decoded image per image, or an
`ImagePyramidSet`), `point_indexes` as `RefineNormalsOptions::point_indexes`,
and `sfmtool.bench.grown_plane_normal`'s keyword arguments, each `None` for the
Rust default, which make up `RefineNormalsOptions::normal`; `resolution`, the
patch resolution `R` set on its `fit`, counts only where `recon` stores no
patch bitmaps, such as the values `sfm embed-patches` refines. A caller
holding an `EditedReconstruction` folds it first with its `materialize`. A
setting the step does not take is refused before the photographs are
decoded. It runs with the GIL released and raises `ValueError` with the
refusal's sentence.
`progress`, an `sfmtool.ProgressCounter`, is set from the `refine normals`
stage's count, so it climbs to that stage's `n` (§ "Stages and progress"):
the points listed, each counted once, or every point, those at infinity
included.

`method` names the step: `"grown_plane"` (the default) or `"pairwise_flow"`,
the `kind` each step's report gives its estimate. With
`"pairwise_flow"` the step's settings are `sfmtool.bench.pairwise_flow_normal`'s
keyword arguments, the `flow_` ones included and `gpu_min_pixels` not taken
([`pairwise-flow-normal.md`](pairwise-flow-normal.md) § "Python binding"),
which make up `RefineNormalsStep::PairwiseFlow`. `max_pairs`, `resolution`
and `fit`'s four keywords are taken by both steps and set the chosen step's.
A keyword only the other step takes is refused with `ValueError` naming it,
rather than ignored, and so is an unknown `method`.

It returns the new `SfmrReconstruction` and a report dict, as `refit_points`'
binding does ([`refit-points.md`](refit-points.md) § "Python binding"), with
the counts `read`, `turned`, `unchanged`, `low_confidence`, `single_pair`,
`none_joined`, `weak_turn`, `undetermined`, `unstable`, `kept`, `at_infinity`
and `not_asked`, `median_turn_deg`, `mean_agreement`, and `points`, one dict
per point in index order with its `index` and `outcome`: `"turned"`,
`"none_joined"`, `"weak_turn"`, `"undetermined"`, `"unstable"`,
`"not_refined"`, `"no_bitmap"`, `"at_infinity"` or `"not_asked"`. The kinds
of `LeftUnchanged` are outcomes of their own, named as each step's report
names its outcomes. A turned point's dict adds `turn_deg` and `unchanged`,
and `joined`, `low_confidence` and `agreement` where the grown plane turned
it, or `single_pair`, `chosen_radius` and `sigma_deg` where the pairwise-flow
normal did; a `"weak_turn"` one adds `turn_deg` and `agreement`; a refused
one adds `error`, the `NormalError`'s name in snake case, and `reason`, its
sentence. There is no point map, since no index moves.

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
- **The pairwise-flow normal as the step** turns a point listed alone onto
  the textured plane to within 2 degrees, its record the one the bench's
  `pairwise_flow_normal` and `commit` write for that point alone, with no
  single pair, a chosen radius among the step's radii and a `σ` under its
  bar, no low confidence count and a `NaN` mean agreement; a setting the step
  does not take refuses the run with the step's `NormalError`.
- **Undetermined and unstable points**: with the pairwise-flow normal on the
  flat wall every point asked for is `Undetermined`, and on the textured
  plane, with the bar on the mean viewing direction lowered to 1 degree,
  `Unstable`; each keeps its record and no readings column is added.
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
are `"not_asked"`; the refusals; a refused point's `error` and `reason`; the
progress counter; and `method="pairwise_flow"` on seoul_bull v02, where
point 2 is turned with the step's numbers and, with the bar on the mean
viewing direction lowered to 20 degrees, about half its normal's angle from
that direction, is `"unstable"` and keeps its normal, a flow keyword
reaching the step (a flow patch wider than every crop makes point 2
`"undetermined"`), every keyword of the step taken, and an unknown method, a
keyword of the other step and the step's own refusals raised as `ValueError`.

## Non-goals

- **Moving a point, a camera or a lens.** The centre is kept and the poses and
  the cameras are the input; moving the points is
  [`refit-points.md`](refit-points.md)'s refit.
- **Changing which observations a point has.** Every verdict is pinned `in`.
- **Choosing among every normal step.** The run uses the grown-plane normal
  or the pairwise-flow normal; the finite-difference and grid-plane normals
  are per-track bench steps.
