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
[`../../formats/matches-file-format.md`](../../formats/matches-file-format.md)
(the `member_status` legend a cluster measurement carries), and
[`../../drafts/sfm-explorer-track-editing.md`](../../drafts/sfm-explorer-track-editing.md)
(the proposal for the evaluations, the stage transitions, the searches and the
viewer's panels).

## Rust API

The value lives in
[bench/track.rs](../../../crates/sfmtool-core/src/bench/track.rs), the steps in
[bench/steps.rs](../../../crates/sfmtool-core/src/bench/steps.rs), and the
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
    label: &str,
    observations: &[usize],
) -> Result<(Bench, SplitReport), SplitError>;

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
in memory: a **reference** observation, a template cut around it, and per
observation a seed (a position and a 2x2 affine shape in that image's pixels),
the refined absolute position and shape, the achieved ZNCC, the shift from the
seed, the observation's own tile localizability and a status in the
`member_status` legend. No pose, no position, no normal. It is what a track is
when it starts from a pixel or from a search hit. The template is `Option`
because cutting it reads the reference's pixels, and the steps in this spec read
no photograph.

**The track stage** is an `embedded_patches` point that is not in the
reconstruction yet: a position, an
[`OrientedPatch`](../../../crates/sfmtool-core/src/patch/cloud.rs) frame, a
consensus bitmap and a colour, and per observation a keypoint with its
leave-one-out ZNCC, its reprojection error, its ray angle and its tile
localizability. It is what a track is when put on the bench from a committed
point.

The affine shape a cluster seed carries is the same `S` the `.matches`
cluster-patches section stores: the map from the detector's canonical unit frame
onto that image's pixels, so its column norms are the sighting's image-space
extent.

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
radius (`ClusterSeed::from_pixel`, for the patch every detector missed, where
nothing says how large the patch is so the caller names it) or a `.sift` feature
with its own position and shape (`ClusterSeed::from_feature`). The seed's image
stem is carried because the label is minted from it.

### Growing and judging

`add_observation` appends a `candidate`, unpinned, with a cluster seed at the
named pixel. The shape defaults to the reference observation's own, so a pixel
gesture on a track that already has a scale needs no radius prompt.

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

### The commit

`commit` is the only step that touches the reconstruction, and it is one
ordinary point edit. It builds a `PointRecord` from the track-stage payload and
the `in` observations: the position the track carries, the frame it stands on,
the consensus bitmap, the colour read from that bitmap's centre, the normal the
frame states, and one observation per `in` sighting with its keypoint and its
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
and a track that carries no position refuses naming the evaluation as the step
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
| `max_shift_px` | `3.0` | How far an observation may sit from its seed (cluster) or the surfel's projection (track), in source-image px. From `ClusterRefineParams::default`. |
| `max_keypoint_uncertainty` | `0.35` | The largest tile localizability an observation may have, in grid px. From `KeypointLocalizeParams::default`'s `max_member_keypoint_uncertainty`. |
| `min_relative_zncc` | `0.7` | The fraction of the track's own self-agreement a sweep candidate has to reach. From `ViewSelectParams::default`. |

## Implementation notes

**A split re-seats the reference of both halves and drops both templates.** A
cluster's reference is an index into its own observation list, and a split
renumbers both lists; a reference that moved out is pointed at the first
observation the half has left. The template goes with it, because a template is a
cut around one particular reference.

**A split preserves the stage.** The measurements the moved observations carry
are measurements of the representation the track is in, and moving a track-stage
half down to the cluster stage means projecting the frame through each
observation's camera, which is the stage step's work and not the split's.

**A `NaN` score clears no bar and is not "unmeasured".** The painting reads a
`NaN` ZNCC as a measured failure and proposes `out` for it, because a round that
produced a `NaN` did run; only an absent measurement is unmeasured.

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

`apply_thresholds` takes each bar as a keyword and moves only the ones given, so
a script can differ from the pipeline's default in one number without restating
the others. `commit` takes the reconstruction's name as `node`, which is what
the report's `label` reads.

```python
from sfmtool._sfmtool import bench as bench_module
from sfmtool._sfmtool.reconstruction import EditedReconstruction

edited = EditedReconstruction(recon)
bench = bench_module.Bench()
bench, track = bench_module.create_track(bench, edited, point=1207)
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
observations and refusing an empty list or all of them; and every commit path --
appending, replacing, absorbing a pulled-from point, the map each of those
reports, and each refusal naming why.
[tests/rust_bindings/test_bench_rust_bindings.py](../../../tests/rust_bindings/test_bench_rust_bindings.py)
covers the same surface through the bindings, over the 17-image seoul_bull solve
converted to `embedded_patches`.

## Non-goals

- **Reading a photograph.** Every step here is decided by what the
  reconstruction and the person already say. The evaluations that register
  pixels -- the cluster refinement, the localizer, the view sweep, the
  descriptor search -- are proposed in
  [`../../drafts/sfm-explorer-track-editing.md`](../../drafts/sfm-explorer-track-editing.md).
- **Moving between the stages.** Upgrading a cluster to a track triangulates and
  frames a surfel, and downgrading a track to a cluster projects the frame
  through each observation's camera; both are proposed in the same draft.
- **Pulling observations in from another point or another item.** The
  `Provenance::Point` a commit absorbs is set by the caller today; the step that
  reads a point's track and adds it is proposed in the same draft.
- **Editing the surfel's frame or normal by hand.** The frame is what the
  kernels fit.
- **Bundle adjustment after a commit.** The commit writes a record and nothing
  settles around it.
