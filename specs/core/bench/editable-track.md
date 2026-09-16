# The editable track

A track is the record of which photographs saw one point on a surface and where
in each photograph it appears. An **editable track** is a track being worked on
rather than one that exists: a list of sightings under consideration, what has
been measured about each, what the person has decided about each, and the
representation the whole thing is currently in. It is the first kind of item on
the bench ([bench.md](bench.md)), and it is a plain value with pure functions
over it, so a script that wants to grow a track under its own thresholds, or
split one by a rule of its own, does so without a window.

Two things make it more than a list of image indexes. The first is that a
sighting carries **what was measured** and **what was decided** separately: a
number is a report and never a decision, and the verdict is always the person's.
The second is that a track passes through two **stages** on the way from a set
of image patches to a reconstructed point, and which stage it is in says which
kernels apply to it and whether it can be written back at all.

Related specs: [bench.md](bench.md) (the place it sits and where its label comes
from),
[`../reconstruction/edited-reconstruction.md`](../reconstruction/edited-reconstruction.md)
(the reconstruction value a commit writes into, and `PointRecord`),
[`../reconstruction/add-observation.md`](../reconstruction/add-observation.md)
and [`../reconstruction/create-point.md`](../reconstruction/create-point.md)
(the one-step track edits this is the worked-on counterpart of),
[`../patch/cluster-patches.md`](../patch/cluster-patches.md) and
[`../patch/cluster-patch-refinement.md`](../patch/cluster-patch-refinement.md)
(the cluster stage's representation and the kernel that measures it),
[`../patch/patch-keypoint-localization.md`](../patch/patch-keypoint-localization.md),
[`../patch/patch-localizability.md`](../patch/patch-localizability.md) and
[`../patch/candidate-track-spawning.md`](../patch/candidate-track-spawning.md)
(the track stage's kernels and the pipeline an upgrade runs),
[`../patch/patch-cloud.md`](../patch/patch-cloud.md) (the frame an upgrade
builds and the shape a downgrade derives),
[`../../formats/matches-file-format.md`](../../formats/matches-file-format.md)
(the `member_status` legend a cluster measurement carries), and
[`../../drafts/sfm-explorer-track-editing.md`](../../drafts/sfm-explorer-track-editing.md)
(the proposal for the searches and the pull-in), and
[`../../gui/track-edit.md`](../../gui/track-edit.md) (the panel it is edited in).

## Rust API

The value lives in
[bench/track.rs](../../../crates/sfmtool-core/src/bench/track.rs), the steps
that read no photograph in
[bench/steps.rs](../../../crates/sfmtool-core/src/bench/steps.rs), the
reading in
[bench/evaluate.rs](../../../crates/sfmtool-core/src/bench/evaluate.rs), the
fit in [bench/fit.rs](../../../crates/sfmtool-core/src/bench/fit.rs), the
stage change in
[bench/stage.rs](../../../crates/sfmtool-core/src/bench/stage.rs), and the
commit in [bench/commit.rs](../../../crates/sfmtool-core/src/bench/commit.rs),
bound as `sfmtool._sfmtool.bench`.

```rust
pub struct EditableTrack {
    pub observations: Vec<Observation>,
    pub stage: Stage,
    pub origin: Option<Origin>,
    pub thresholds: Thresholds,
}

pub struct Observation {
    pub image: u32,
    pub provenance: Provenance,
    pub verdict: Verdict,
    pub pinned: bool,
    pub cluster: Option<ClusterMeasurement>,
    pub track: Option<TrackMeasurement>,
}

pub struct TrackMeasurement {
    pub keypoint: Option<[f32; 2]>,
    pub zncc: Option<f64>,
    pub seed_shift_px: Option<f64>,          // the peak's move from the sighting
    pub projection_offset_px: Option<f64>,   // the sighting's distance from the point
    pub reprojection_error: Option<f64>,
    pub ray_angle_deg: Option<f64>,
    pub localizability: Option<f64>,
    pub reason: Option<Unmeasured>,          // present exactly when zncc is not
}

pub enum Unmeasured {
    NoSeed,
    OffSensor,
    NoProjection,
    Grazing { cosine: f64 },
    NoConsensus,
    Unscorable,
}

pub enum Verdict { In, Out, Candidate }

pub enum Provenance {
    Origin,
    Descriptor { feature: u32 },
    Sweep,
    Pixel,
    Point { point: u32 },
}

pub enum Stage {
    Cluster(ClusterPayload),
    Track(TrackPayload),
}

pub struct Origin { pub version: u64, pub point: u32 }

pub struct Thresholds {
    pub min_zncc: f64,
    pub max_shift_px: f64,
    pub max_keypoint_uncertainty: f64,
    pub min_relative_zncc: f64,
}

// Putting one on the bench.
pub fn create_track(
    bench: &Bench,
    edited: &EditedReconstruction,
    point: u32,
    options: &CreateTrackOptions,
) -> Result<(Bench, CreateReport), CreateTrackError>;

pub fn create_cluster(
    bench: &Bench,
    seed: &ClusterSeed,
) -> Result<(Bench, CreateReport), CreateClusterError>;

// Steps on one track.
pub fn add_observation(
    track: &EditableTrack,
    seed: &ObservationSeed,
) -> Result<(EditableTrack, AddObservationReport), TrackEditError>;

pub fn set_verdict(
    track: &EditableTrack,
    observation: usize,
    verdict: Verdict,
) -> Result<(EditableTrack, VerdictReport), TrackEditError>;

pub fn apply_thresholds(track: &EditableTrack) -> (EditableTrack, ThresholdReport);

pub fn split(
    bench: &Bench,
    edited: &EditedReconstruction,
    label: &str,
    observations: &[usize],
) -> Result<(Bench, SplitReport), SplitError>;

// The steps that read photographs, and the half of each one's validation that
// does not.
pub fn evaluate_preconditions(track: &EditableTrack) -> Result<(), EvaluateError>;

pub fn fit_preconditions(track: &EditableTrack) -> Result<(), FitError>;

pub fn set_stage_preconditions(
    track: &EditableTrack,
    stage: StageKind,
) -> Result<(), StageError>;

// Read the track as it stands, and move nothing.
pub fn evaluate(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &EvaluateOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, EvaluateReport), EvaluateError>;

// Move it: localize, re-triangulate, re-fuse -- then read the result back.
pub fn fit(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &FitOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, FitReport), FitError>;

pub fn set_stage(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    stage: StageKind,
    options: &FitOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, StageReport), StageError>;

/// The localizer with every per-view gate off and the basis cap lifted, which
/// is what both steps run.
pub fn open_localizer() -> KeypointLocalizeParams;

pub struct EvaluateOptions {
    pub cluster: ClusterRefineParams,
    pub localize: KeypointLocalizeParams,   // open_localizer, one round
    pub search_px: f64,                     // patch-grid px
}

pub struct FitOptions {
    pub localize: KeypointLocalizeParams,   // open_localizer
    pub refine: KeypointSubpixelParams,
    pub evaluate: EvaluateOptions,          // the reading a fit ends with
}

pub struct EvaluateReport {
    pub stage: StageKind,
    pub measured: usize,
    pub unmeasured: usize,
    pub reference: Option<usize>,        // the cluster stage's
    pub position: Option<Point3<f64>>,   // where the track stands
    pub condition_number: Option<f64>,
}

pub struct FitReport {
    pub evaluate: EvaluateReport,        // the reading of its own result
    pub placed: usize,
    pub position: Option<Point3<f64>>,
    pub condition_number: Option<f64>,
}

pub struct StageReport {
    pub from: StageKind,
    pub to: StageKind,
    pub changed: bool,
    pub fit: Option<FitReport>,            // the upgrade's
    pub reference: Option<usize>,          // the downgrade's
}

impl StageReport {
    /// What the change did beyond moving the stage, as the clause that follows
    /// the stage phrase, its own separator included: a caller that writes that
    /// phrase in its own words adds this rather than printing the whole report
    /// behind its own sentence and stating the stage twice.
    pub fn detail(&self) -> String;
}

// The one step that writes the reconstruction.
pub fn commit(
    edited: &EditedReconstruction,
    track: &EditableTrack,
) -> Result<(EditedReconstruction, CommitReport), CommitError>;

pub struct CommitReport {
    pub point: u32,
    pub replaced: Option<u32>,
    pub map: PointMap,
    pub observation_count: usize,
}

impl CommitReport {
    /// The points the commit deleted because `in` observations had been pulled
    /// from them, ascending. Read off `map`.
    pub fn absorbed(&self) -> &[u32];
    pub fn label(&self, node: &str) -> String;
}
```

### Why it is shaped this way

**A value in, a value out, for every step.** There is no `&mut` anywhere, so a
refused step leaves nothing half-applied and a caller holding both the before
and the after can go back. The creating steps take the bench because they mint a
label and make the new item active, which is the bench's business; the steps on
one track take the track alone, and the caller installs the result with
`Bench::replace`. That split is what stops a verdict from touching the list.

**The measurement slots are per stage and optional.** A track put on the bench
from a point has a keypoint and no cluster seed; one started from a pixel has a
seed and no keypoint. Two slots, each `Option`, say which of those a given
sighting is without a third state to keep coherent, and they say it per
observation rather than per track, so a report computed against the track as it
stood when a task began still applies when it finishes.

**The verdict and the pin are separate fields.** `verdict` is what the person
has decided and `pinned` is whether they decided it by hand. Without the second,
a threshold slider would either be unable to propose anything or would silently
overwrite a judgement, and the whole difference between the bench and the batch
pipeline is that here the numbers are shown and the person decides.

**Reading and moving are two steps, not one.** `evaluate` is a measurement of
the track as it stands: it fills the measurement slots and leaves the position,
the frame, the bitmap, every keypoint and every verdict exactly as they were.
`fit` is the modification, and it **ends by calling `evaluate` on its own
result**, so the numbers a fit leaves behind are a reading's numbers and the two
steps can never give two accounts of one track. A person asking "is this track
right?" and a person saying "make it right" are asking different questions, and
a single step that measured by refitting could only ever answer the second.

**Each photometric step publishes its photograph-free refusals.** A caller that
runs `evaluate`, `fit` or `set_stage` somewhere expensive -- on a worker, after
decoding a dozen images -- wants the refusals that were knowable from the track
alone *before* that work, not behind it. So the conditions that need no pixels
are their own functions: whether the track stage has a surfel to read against
(`evaluate_preconditions`), that plus enough `in` observations for the consensus
a fit registers against (`fit_preconditions`), and whether an upgrade has two
sightings to triangulate from or a downgrade has the frame and the position it
projects (`set_stage_preconditions`). The two-`in` rule is a fit's and not a
reading's: fitting one sighting against nothing would move it to wherever a
template of itself sits, while reading one sighting reports a sighting with
nothing to correlate against, which is an answer. Setting the stage a track is
already at is not among them -- that is the change that does nothing, reported
as `StageReport::changed = false`. Each step calls its own before anything else,
so the answer a caller gets in advance and the answer the step would have given
cannot drift, and the conditions that *do* need the views stay inside the step
where the views are.

**The commit reports an index map.** `map` is the `PointMap` the write made --
the pair `replaced -> point`, or the created index, chained with a `Removed` of
the points it absorbed when there are any
([`../reconstruction/edited-reconstruction.md`](../reconstruction/edited-reconstruction.md)
§ "The point map"). A caller carrying a selection, a point id or an undo across
the commit reads that one field, in the same vocabulary every other edit answers
in. `point` and `replaced` stand beside it for the caller that wants the write
itself rather than the mapping -- re-seating a track with `with_origin` takes
`point` -- and `absorbed()` reads the removal step back off the map, which is
what the label's sentence counts.

**The origin's version serial is opaque.** It is a `u64` the caller numbers its
own versions with. Core neither mints nor interprets it; what core does with the
origin is decide whether a commit replaces a point or creates one.

**The decoded views are a named input.** `evaluate`, `fit` and `set_stage` take one
[`ProjectedImage`](../../../crates/sfmtool-core/src/patch/normal_refine/params.rs)
per image of the reconstruction -- a camera, a pose and a pyramid -- indexed by
image index, exactly as
[`add_observation`](../reconstruction/add-observation.md) takes them. A
reconstruction carries poses and lenses rather than photographs, so decoding and
caching stay the caller's, and the same call serves a viewer with a warm cache
and a script that just read the files.

**The kernel parameters are not the track's thresholds.** `EvaluateOptions` and
`FitOptions` carry what the kernels are allowed to do; the track's `Thresholds`
are what the *painting* judges the result against. Keeping them apart is what
makes a slider a question about verdicts rather than about numbers: moving one
repaints, and it cannot change what was measured.

**Neither step lets a kernel drop a sighting.** Both default their localizer to
`open_localizer`: the four per-view gates
(`max_shift_px`, `min_absolute_zncc`, `min_relative_zncc`,
`max_member_keypoint_uncertainty`) off, and the consensus-basis cap lifted
(`basis_max_views = 0`). A gate deletes a view from the kernel's answer, and a
deleted view is a row the bench would show empty for a reason nothing recorded;
on the bench the deleting is the person's, through a verdict or a threshold they
can see and move. The cap is lifted for the same reason: with the batch default
of eight, a twelve-sighting track would have four of its views registered once
against a template the other eight built, and be reported in terms that are not
the terms the eight were reported in. `max_shift_px` goes to a large finite bar
rather than to infinity, because the kernel reports "this keypoint left the
photograph" as an infinite shift and that signal still has to land.

### Example

```rust
use sfmtool_core::bench::{
    apply_thresholds, commit, create_track, set_verdict, Bench, CreateTrackOptions, Verdict,
};

let bench = Bench::new();
let (bench, report) = create_track(&bench, &edited, 1207, &CreateTrackOptions::default())?;
let track = bench.track(&report.label).expect("just put on");

let (track, _) = set_verdict(track, 3, Verdict::Out)?;   // this photograph is not it
let (track, painted) = apply_thresholds(&track);          // the rest, from the numbers
println!("{} in, {} out", painted.turned_in, painted.turned_out);

let (next, commit_report) = commit(&edited, &track)?;
println!("{}", commit_report.label("bull"));
```

## Observations, provenance and verdicts

An observation names one image of the reconstruction and one place in it.
Observations are **appended and never renumbered**, so an index into the list is
stable for the life of the track: an agent holding an index after a verdict
still holds the same observation, and an evaluation that finishes after the
track has moved on still lands on the observations it measured.

**Provenance** is where the sighting came from, and it is shown rather than
used, with exactly one exception: a commit deletes the points that `Point`
observations were pulled from. No kernel reads it.

**A verdict** is `in`, `out` or `candidate`. `in` observations are what the
kernels run over and what a commit writes. `out` is a sighting the person
refused, kept in the list so a later search does not propose it again and so the
refusal stays visible. `candidate` is something proposed and not yet ruled on.

**One `in` observation per image.** A track cannot observe an image twice. A
second candidate in an image already held is allowed, and is shown and scored
like any other, but turning it `in` while the other is `in` is refused naming
the observation that holds the image. This is the same rule the cluster kernel
spells `duplicate_image`.

## The two stages

**The cluster stage** is a `.matches` cluster with its cluster-patches section,
in memory: a **reference** observation, a **radius**, a template cut around the
reference, and per observation a seed (a position and a 2x2 affine shape in that
image's pixels), the refined absolute position and shape, the achieved ZNCC, the
shift from the seed, the observation's own tile localizability and a status in
the `member_status` legend. No pose, no position, no normal. It is what a track
is when it starts from a pixel or from a search hit. The template is `Option`
because cutting it reads the reference's pixels: a track carries one once an
evaluation has run, and none before that.

**The track stage** is an `embedded_patches` point that is not in the
reconstruction yet: a position, an
[`OrientedPatch`](../../../crates/sfmtool-core/src/patch/cloud.rs) frame, a
consensus bitmap and a colour, and per observation a keypoint with its
leave-one-out ZNCC, its reprojection error, its ray angle and its tile
localizability. It is what a track is when put on the bench from a committed
point.

### The cluster stage's units

The affine shape a cluster observation carries, seeded or refined, is the same
`S` the `.matches` cluster-patches section stores: the map from the detector's
canonical **keypoint frame** onto that image's pixels. It is a scale and not a
size. What makes it a size is the cluster's **radius**: the patch is the square
`[-radius, radius]` of keypoint-frame units, so a sighting's pixel half-width
along a column is `radius * ||column||`, and a shape read without the radius
says nothing about how large the patch is.

The radius lives on the cluster payload, and it is the one place the size is
written down:

- **An evaluation runs the refinement kernel at it**, in place of the radius in
  the kernel parameters it is otherwise handed. A round run at another radius
  would register a different square from the one the person put on the bench, so
  a seed's meaning cannot change under it.
- **A seed named in pixels is divided by it.** `ClusterSeed::from_pixel` takes a
  half-width in source-image pixels, exactly as creating a point from a pixel
  does, and stores `radius_px / radius` per unit, so the patch spans the pixels
  that were asked for. A `.sift` feature's own shape is already in these units
  and is stored as it is.
- **Both stage transitions convert through it.** The `.sfmr` rule for a
  keypoint's shape states a patch's half-axes in pixels, so a downgrade divides
  the projected columns by the radius and an upgrade multiplies the reference's
  by it before framing the surfel. A track taken down and put back up is the
  size it was.
- **Everything that draws reads it**: the bench layer's parallelogram in the
  Image Detail panel and the Track Edit panel's tile. A cluster carries its
  radius from the moment it is started, so a seed is drawn at the size it was
  named before anything has read a photograph, and an evaluation that finds the
  same scale leaves the outline where it is.

The template carries no radius of its own. Two copies of one number could
disagree, and a template that disagreed with the seeds it was cut from would be
a cut of a square nothing else was measured over.

## The steps

### Putting a point on the bench

`create_track` reads the point through the reconstruction's overlay and builds a
track-stage track from it: the point's own frame, bitmap, colour and keypoints,
its origin set to that point, and every observation `in` and unpinned. **Nothing
is recomputed.** The leave-one-out ZNCC is `observation_confidence` read back out
of its byte scale where the column exists, and every measurement an evaluation
would produce is left unmeasured, so putting a track on the bench and doing
nothing shows the numbers the reconstruction already holds plus the verdict
column.

A `sift_files` reconstruction is put on the bench like any other: inspecting a
track is allowed everywhere, and it is the commit that refuses to write one
back.

### Starting a cluster

`create_cluster` puts a new cluster-stage track on the bench with one
observation, which is also its reference and is `in`. The seed is a pixel with a
half-width in pixels (`ClusterSeed::from_pixel`, for the patch every detector
missed, where nothing says how large the patch is so the caller names it) or a
`.sift` feature with its own position and shape (`ClusterSeed::from_feature`).
The cluster takes the default radius, which is the refinement kernel's own, and
a pixel seed's shape is sized against that radius, so the patch spans the pixels
the gesture asked for. The seed's image stem is carried because the label is
minted from it.

### Growing and judging

`add_observation` appends a `candidate`, unpinned, with a cluster seed at the
named pixel. The shape defaults to the reference observation's own, so a pixel
gesture on a track that already has a scale needs no radius prompt and lands at
that track's size. With no reference to copy it is the identity, which is one
pixel to the keypoint-frame unit: a patch of `[-radius, radius]` pixels, and
what a track with nothing to say about its own scale is worth.

`set_verdict` sets one verdict **by hand** and pins it. `apply_thresholds`
paints the proposed verdicts from the stored measurements onto the unpinned
observations, and leaves a pinned one where it is. An observation nothing has
measured at the track's current stage is left alone: there is no proposal to
apply. An observation that *was* measured and failed is turned `out`, because a
measured refusal is something the person should see.

The painting cannot produce a track that observes an image twice. It walks the
observations best score first, and where several unpinned sightings of one image
would pass, the best takes the `in` and the rest stay candidates.

### Splitting

`split` moves the named observations off one track into a second track beside it
on the bench, labelled `<parent>-split`. The observations are **named
explicitly**, by index, rather than read off the verdicts, because `out` says a
sighting does not belong *here*: a sighting the localizer refused and one that
belongs to the track next door are both `out`, and a verdict cannot say which of
them the new track should take. The moved observations keep their verdicts,
their provenance and both stages' measurements. The second track has no origin,
so a commit of it creates a point while a commit of the first still replaces the
one it came from. An empty list, or every observation, is refused: neither
leaves two tracks.

**The second track is a cluster.** A split is the step for a track that is two
surfaces, and the half being taken off is a set of sightings that agree with
each other and not with a 3D hypothesis fitted to both; carrying that hypothesis
onto it would state as fact the thing the split is questioning. So a
track-stage half is put down to the cluster stage by the same downgrade
`set_stage` runs, which is why `split` takes the reconstruction: the downgrade
projects the frame through each observation's camera. The first track keeps its
stage, its origin and everything it was.

### Evaluating

`evaluate` fills the measurement slots of every observation at the stage the
track is in, whatever its verdict, and **moves nothing else**: the position, the
frame, the bitmap, every keypoint and every verdict come back as they went in.
An `out` observation is scored the way a candidate is, so a refusal is shown
beside the number it would have been judged on and a slider can propose taking
it back.

**At the cluster stage** every observation's seed is a member of an in-memory
`.matches` cluster, and
[`refine_cluster_patches`](../patch/cluster-patch-refinement.md) is run over it
through its borrowed-pyramid entry. The kernel picks the reference -- its
largest-scale usable member -- cuts the template there and warps every other
seed onto it; what lands in each observation's slot is the refined position and
shape, the achieved ZNCC, the drift from the seed, the observation's own tile
localizability and the kernel's `member_status`. The kernel runs at the
cluster's own radius rather than the one in the evaluation's options, so the
square it registers is the square the seeds were written against. The payload's
reference is set to where the kernel cut, and `ClusterPayload::template` to the
reference's own tile on the template grid, sampled by the kernel's own sampler.
No pose is read, so a cluster evaluates on a node whose images have none.

The tile localizability is scored at each observation's **seed** geometry, which
is where the kernel's own gate scores it, so the column and a
`RejectedUnlocalizable` status are the same measurement rather than two.

**At the track stage** the reading is **one round** of
[`localize_patch_keypoints`](../patch/patch-keypoint-localization.md) over the
observations *where they sit*: the cores are cut at the pixels the observations
already carry, each is scored against the leave-one-out consensus of the round's
others, and the peak of its shift search says where the correlation would rather
be. The peak is reported as a distance and never taken, so a reading is a
reading. What lands in each slot is:

| Slot | What it says |
|------|--------------|
| `zncc` | The leave-one-out agreement at the peak, within `search_px` of where the sighting is. With `seed_shift_px` near zero it is the agreement at the keypoint itself. |
| `seed_shift_px` | How far that peak sits from the observation's own keypoint, in source-image px. The **sighting's** own evidence, and what `max_shift_px` paints on. |
| `projection_offset_px` | How far the observation's keypoint sits from the point's projection. A statement about the **point**: a mis-triangulated track shows a column of large offsets beside a column of zero shifts. |
| `reprojection_error`, `ray_angle_deg` | The same residual in px and in degrees, against the position the track carries and the pixel the observation sits at. |
| `localizability` | The observation's own tile `sigma_pos`, through the frame anchored at its keypoint. |
| `reason` | Why there is no ZNCC, when there is none. Present exactly when `zncc` is absent. |

The last four rows are filled for every observation that has a pixel at all,
scored or not: a row the correlation could not reach still has a distance from
the projection, and that distance is often the thing that explains the row.

**The two distances are two questions**, and one column could not carry both.
Anchoring the shift at the projection -- which is what the localizer's own
`offsets_px` and its `max_shift_px` gate do -- makes a sighting five px from a
mis-triangulated point look like a sighting that moved five px, and turns out
the very observation that would pull the point back.

**A row without a score names its refusal.** `Unmeasured` is that name, one
short sentence each: `NoSeed` (nothing says where it sits), `OffSensor` (it sits
off the photograph), `NoProjection` (the point misses this view), `Grazing` (its
ray grazes the patch plane, with the cosine), `NoConsensus` (fewer than two
observations of its round could be read together) and `Unscorable` (its tile
could not be scored: it runs off the photograph, or no channel of it carries
texture). The first four are decided before any correlation, from the
observation and the geometry, which is what lets the row carry the reason
instead of simply going missing from the kernel's answer.

**The search window is widened to reach the furthest seed.** The kernel anchors
its window at the point's projection and clips the integer part of a seed beyond
`search` back onto that bound, so a window sized for `search_px` alone would
start a far-out sighting short of where it actually is and report the
correlation of a place the sighting is not. Each round therefore runs at
`search_px` plus the furthest seed's own offset
([`keypoint_grid_offset`](../patch/patch-keypoint-localization.md) is that
offset). In return, an observation in a round that holds a far-out seed can
report a peak further than `search_px` from itself, which is the honest reading
of a window that had to be that wide.

**One localization holds one observation per image**, because it registers a
point's sighting in a view and two sightings in one view are two hypotheses
about that view. So the first round is the `in` observations plus every other
observation in an image none of them holds -- the shape `add_observation` runs
-- and an observation in an image that is already spoken for gets a round of its
own, against the `in` observations minus the one whose image it wants, which
asks the same leave-one-out question about the other hypothesis. A row read in
the first round keeps that round's numbers: a contested round re-reads the `in`
observations against a consensus one of them was left out of, and that answers a
question about the *other* hypothesis.

### Fitting

`fit` is the step that moves the track, and it runs the same rounds. At the
**track stage** the surfel is localized into every view by the two kernels the
embed pass and `add_observation` chain --
[`localize_patch_keypoints`](../patch/patch-keypoint-localization.md) then
`refine_patch_keypoints` -- the `in` results are re-triangulated, the frame is
re-centred at that position and the consensus bitmap is fused over them. The
payload takes the position, the frame, the fused bitmap, the colour at that
bitmap's centre and the triangulation's condition number.

**An observation the kernels did not place keeps its pixel.** Out of frame, or
in no round at all, its keypoint stays wherever it already sat -- the one it
arrived with, or the cluster stage's refined position for a track just upgraded
-- so the column still says where the sighting is and the re-triangulation still
has its ray.

The fuse is the sub-pixel kernel's own `render_bitmaps` path, run over the `in`
views alone with no Gauss-Newton step, so it moves nothing and only renders and
blends the keypoints the fit settled. Its grid is the reconstruction's own
bitmap grid where it stores one, so what is fused is a tile the column can hold
and a commit can write.

**A fit ends by evaluating its result**, and that reading is where every
per-observation number and every count in the `FitReport` comes from. `placed`
is the fit's own: how many observations the kernels moved. At the **cluster
stage** a fit *is* the refinement, which is what a reading is too -- a cluster
has no geometry behind it to move -- so the two steps run the one kernel and
report the same thing.

### Moving between the stages

`set_stage` is one operation in both directions. Setting the stage a track is
already at gives it back unchanged with `changed` false, so a caller can wire a
toggle straight to it and push no version for a step that did not happen.

**Up, cluster to track**, is the spawn pipeline's own steps over one candidate:

1. **Triangulate** the `in` observations' refined cluster positions through
   their cameras.
2. **Frame** the patch at that position: the in-plane axes and half-extents are
   what the reference observation's affine shape, scaled by the cluster's
   radius, unprojects to on the plane at the triangulated depth
   ([`OrientedPatch::from_affine_shape_at_depth`](../patch/patch-cloud.md)), and
   the normal is the mean viewing direction, which is how `to_embedded_patches`
   frames a surfel from the views that see it. The reference is the cluster's
   own when it is `in`, and otherwise the largest-scale `in` observation, which
   is what the cluster kernel would have picked among them.
3. **Localize, refine, re-triangulate, fuse and read back**, which is the
   track-stage fit above over seeds that are the cluster's refined positions.

The cluster-stage measurements are dropped with the stage: each describes a
registration against a reference and a template the track no longer has, and
the track stage measures every observation afresh against the surfel.

**Down, track to cluster**, is always possible and lossy on purpose. Each
observation is re-seeded at its keypoint with the affine shape the format
derives by projecting the frame at that observation's anchor
([`../../formats/sfmr-file-format.md`](../../formats/sfmr-file-format.md)
§ "Deriving keypoint shape, scale, and orientation", the inverse of the framing
the upgrade does, so the two directions state one relationship), divided by the
new cluster's radius because that rule's columns are pixel half-axes and a
cluster seed is per keypoint-frame unit. The reference
becomes the `in` observation with the largest projected patch scale, which is
the one showing the most of the patch, and the position, the frame and the
bitmap are dropped, and the track-stage measurements with them, since each was
made against that position and frame. This is the
step for a track whose observations were right and whose 3D hypothesis was the
problem: the cluster kernel then judges the observations on appearance alone,
and an upgrade builds the 3D afresh from whatever survives.

A track therefore carries the measurements of its current stage and no other.
What crosses a transition is what the next stage starts from: the refined
cluster positions become the seeds of the triangulation going up, and the
localized keypoints become the cluster seeds going down, which is the point of
taking a track down -- the sightings are what is being kept, and the 3D
hypothesis is what goes.

### The commit

`commit` is the only step that touches the reconstruction, and it is one
ordinary point edit. It builds a `PointRecord` from the track-stage payload and
the `in` observations: the position the track carries, the frame it stands on,
the consensus bitmap, the colour read from that bitmap's centre, the normal the
frame states, the mean of what the last evaluation measured as each `in`
sighting's reprojection error in the point's `error` column (zero where nothing
was measured, which is what a point no observation could be scored for carries
anywhere else), and one observation per `in` sighting with its keypoint and its
leave-one-out ZNCC in `observation_confidence` where the column exists. The
observations are written in image order, which is the order a stored track is in
and every reader of one relies on.

- **With no origin that resolves**, `EditedReconstruction::add_point`.
  `replaced` is `None` and the map is a `Created` naming the index it took.
- **With an origin that resolves** in the value being committed into,
  `replace_point` on that index. `replaced` names it and the map is a
  `Replaced` of the one pair.
- **With `in` observations whose provenance names a point** other than the
  origin, those points are deleted as well, because a track cannot observe an
  image twice and a reconstruction should not hold two points for one surface.
  Only `in` observations count: a candidate or an `out` sighting pulled from a
  point leaves that point alone. This is the merge, and the map becomes a
  `Chain` of the write and a `Removed` of what it absorbed, which is what
  `absorbed()` reads back.

An origin whose point the value no longer holds names nothing, and the commit
creates rather than refusing: the person is looking at a track whose point was
deleted under it.

**The commit does not triangulate.** A track commits with the position it
carries, so the record that is written is one the numbers on screen describe,
and a track that carries no position refuses naming the fit as the step
that is missing. After a commit the track is unchanged; `EditableTrack::with_origin`
is what a caller re-seats it with so a second commit replaces what the first
wrote.

The refusals are: the cluster stage (*"upgrade it before committing"*), a
`sift_files` reconstruction (an observation the localizer placed is a keypoint
and not a feature index), fewer than two `in` observations, no position, no
frame or no bitmap where the reconstruction stores one per point, an `in`
observation with no keypoint, and an image past the image table.

`CommitReport::label(node)` writes the sentence a log records, which needs the
name the caller knows the reconstruction by: `Committed track: 5 observations in
bull, replacing point 1207, absorbing 2 points`.

## Parameters

The thresholds default to the kernels' own bars, read from those kernels'
parameter types rather than written out again, so the bench and the batch pass
start from the same bar and moving one is the person choosing to differ.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_zncc` | `0.85` | The ZNCC an observation has to reach: the achieved template ZNCC at the cluster stage, the leave-one-out ZNCC at the track stage. From `ClusterRefineParams::default`. |
| `max_shift_px` | `3.0` | How far the correlation peak may sit from where the observation sits: the drift from its seed at the cluster stage, `seed_shift_px` at the track stage, both in source-image px. From `ClusterRefineParams::default`. The other track-stage distance, `projection_offset_px`, is deliberately **not** judged: it is a verdict on the point, and painting sightings by it would turn out the observations that would move a mis-triangulated point back. |
| `max_keypoint_uncertainty` | `0.35` | The largest tile localizability an observation may have, in grid px. From `KeypointLocalizeParams::default`'s `max_member_keypoint_uncertainty`. |
| `min_relative_zncc` | `0.7` | The fraction of the track's own self-agreement a sweep candidate has to reach. From `ViewSelectParams::default`. |

## Implementation notes

**A split re-seats the reference of both halves and drops both templates.** A
cluster's reference is an index into its own observation list, and a split
renumbers both lists; a reference that moved out is pointed at the first
observation the half has left. The template goes with it, because a template is a
cut around one particular reference.

**A `NaN` score clears no bar and is not "unmeasured".** The painting reads a
`NaN` ZNCC as a measured failure and proposes `out` for it, because a round that
produced a `NaN` did run; only an absent measurement is unmeasured. An
evaluation therefore writes `None` where a kernel reported `NaN`: a kernel's
`NaN` means it did not score the row, and storing it as a number would turn a
silence into a refusal.

**The cluster stage's kernel reaches through a borrowed-pyramid entry.** A
pyramid is a decoded image and cloning one copies every pixel, while the bench's
views are borrowed ([`ProjectedImage`]); `refine_cluster_patches_borrowed` is
the same kernel over `&[&ImageU8Pyramid]`, and `refine_cluster_patches` is a
wrapper that collects the references. Nothing about the refinement differs.

[`ProjectedImage`]: ../../../crates/sfmtool-core/src/patch/normal_refine/params.rs

**A track at infinity is promoted before it is fitted against.** A `w = 0` frame
is tangent to the direction sphere, and a fit against one would pull every
sighting back onto the bearing's projection and undo the depth the other
sightings carry. So a track-stage fit of one triangulates the seeds first and
promotes the frame to that depth -- the two-pass shape
[`add_observation`](../reconstruction/add-observation.md) takes, and the same
rescale from the camera-cloud centroid. A *reading* promotes nothing: it reports
the track as it stands, bearing and all.

**A commit's absorb list is filtered by what is still live.** A point another
version already deleted is nothing to absorb, and a commit is not the place to
complain about it, so the delete is attempted and a refusal drops the entry
rather than the commit.

## Python bindings

`sfmtool._sfmtool.bench`. The steps are module-level functions with the same
names and the same shape as the Rust ones; `EditableTrack` is a read-only value
class whose observations cross as dicts, with each stage's measurements under
`"cluster"` and `"track"` and a key present exactly when something has measured
it. Verdicts and provenance kinds are the lowercase words (`"in"`, `"out"`,
`"candidate"`; `"origin"`, `"descriptor"`, `"sweep"`, `"pixel"`, `"point"`).
Refusals are `ValueError` carrying the core sentence.

`create_cluster` takes either `radius_px`, a half-width in that image's pixels,
or `shape`, a 2x2 in keypoint-frame units; `EditableTrack.radius` is the
cluster's radius those units are read over, and is `None` at the track stage.

`apply_thresholds` takes each bar as a keyword and moves only the ones given, so
a script can differ from the pipeline's default in one number without restating
the others. `commit` takes the reconstruction's name as `node`, which is what
the report's `label` reads.

`evaluate`, `fit` and `set_stage` take `images` the way every patch kernel does
-- a list of `HxW[xC]` `uint8` arrays, one per image of the reconstruction, or a
prebuilt `ImagePyramidSet`. `evaluate` and `fit` take an optional keyword
`search_px`, the radius the reading looks for each peak in; `set_stage` takes
the stage as the word `"cluster"` or `"track"`. Their reports are dicts:
`stage`, `measured` and `unmeasured`, with `reference` at the cluster stage and
`position` and `condition_number` at the track stage; `placed`, `position`,
`condition_number` and the reading's own `evaluate` report for a fit; and
`from`, `to`, `changed`, the upgrade's `fit` report and the downgrade's
`reference` for a stage change. An observation's `"track"` dict carries
`reason`, the sentence, exactly when it carries no `zncc`.

```python
from sfmtool._sfmtool import bench as bench_module
from sfmtool._sfmtool.reconstruction import EditedReconstruction

edited = EditedReconstruction(recon)
bench = bench_module.Bench()
bench, track = bench_module.create_track(bench, edited, point=1207)
track, read = bench_module.evaluate(track, edited, images)   # measures, moves nothing
print(read["measured"], "of", track.observation_count, "sightings scored")
track, fitted = bench_module.fit(track, edited, images)      # moves it, then reads it back
print(fitted["placed"], "sightings placed at", fitted["position"])
track, painted = bench_module.apply_thresholds(track, min_zncc=0.9)
edited, report = bench_module.commit(edited, track, node="bull")
print(report["label"])
```

## Testing

[bench/tests.rs](../../../crates/sfmtool-core/src/bench/tests.rs) runs over the
synthetic textured-plane scene the add-observation tests build, wrapped as an
`embedded_patches` reconstruction whose stored keypoints are the exact
projections, so what a commit should have written is known to the pixel. It
covers: a point put on the bench being at the track stage with every observation
`in` and the stored numbers carried; two observations in one image not both
being `in`; the painting proposing from the measurements, leaving a pinned
verdict alone and giving one image one `in`; a split taking exactly the named
observations, handing the half it takes off back as a cluster, and refusing an
empty list or all of them; and every commit path -- appending, replacing,
absorbing a pulled-from point, the map each of those reports, and each refusal
naming why.

The fit is tested against the kernels themselves: a track put on the bench from
a point fits to what a direct call of the same two kernels on the same frame and
seeds produces, keypoint for keypoint. The reading is tested as a reading: the
position, the frame, the bitmap and every keypoint are the same values before
and after, and every row comes back with both distances. Beside them: a
downgrade re-seeding every sighting at its keypoint and an upgrade triangulating
back to within a pixel's worth of where the point was; a candidate placed on the
plane clearing the bar and one placed off every image coming back unmeasured
with `OffSensor` and its sentence; a lone sighting read as `NoConsensus` rather
than refused, while a fit of the same track is refused; a pinned `out` scored
and left `out`; a cluster started from two pixels refining, upgrading and
committing a point onto the planted surface; setting the current stage reporting
`changed` false; each refusal naming what did not hold; and the three
precondition functions giving their step's own answer when they are asked alone,
which is what makes them safe to ask in front of a decode.
[tests/rust_bindings/test_bench_rust_bindings.py](../../../tests/rust_bindings/test_bench_rust_bindings.py)
covers the same surface through the bindings, over the 17-image seoul_bull solve
converted to `embedded_patches`.

## Non-goals

- **Searching for observations to add.** The view sweep over the images that
  see the surfel, and the descriptor search over a `.kdf` forest, are what
  propose candidates; both are proposed in
  [`../../drafts/sfm-explorer-track-editing.md`](../../drafts/sfm-explorer-track-editing.md).
  An evaluation scores the candidates something else put on the track.
- **The pairwise coherence matrix.** An evaluation scores each observation
  against the others' consensus; the `k x k` matrix that shows a track made of
  two surfaces as two blocks is
  [`member_zncc_matrix`](../patch/member-coherence-validation.md), and reading
  it into the track is proposed in the same draft.
- **Pulling observations in from another point or another item.** The
  `Provenance::Point` a commit absorbs is set by the caller today; the step that
  reads a point's track and adds it is proposed in the same draft.
- **Editing the surfel's frame or normal by hand.** The frame is what the
  kernels fit.
- **Bundle adjustment after a commit.** The commit writes a record and nothing
  settles around it.
