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
[`../patch/cluster-patches.md`](../patch/cluster-patches.md) and
[`../patch/cluster-patch-refinement.md`](../patch/cluster-patch-refinement.md)
(the cluster stage's representation and the kernel that measures it),
[`../patch/patch-keypoint-localization.md`](../patch/patch-keypoint-localization.md),
[`../patch/patch-localizability.md`](../patch/patch-localizability.md) and
[`../patch/candidate-track-spawning.md`](../patch/candidate-track-spawning.md)
(the track stage's kernels and the pipeline an upgrade runs),
[`../patch/patch-cloud.md`](../patch/patch-cloud.md) (the frame an upgrade
builds and the shape a downgrade derives),
[`../patch/patch-view-selection.md`](../patch/patch-view-selection.md) (the
geometric and photometric gates the geometry search shares with the batch
pipeline),
[`../../formats/matches-file-format.md`](../../formats/matches-file-format.md)
(the `member_status` legend a cluster measurement carries),
[`../features/kdf-constellation-query.md`](../features/kdf-constellation-query.md)
(the query the descriptor search is one of), and
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
finite-versus-bearing criterion the fit and the upgrade share in
[bench/classify.rs](../../../crates/sfmtool-core/src/bench/classify.rs), the
stage change in
[bench/stage.rs](../../../crates/sfmtool-core/src/bench/stage.rs), the
descriptor search in
[bench/search.rs](../../../crates/sfmtool-core/src/bench/search.rs), and the
geometry search in
[bench/geometry_search.rs](../../../crates/sfmtool-core/src/bench/geometry_search.rs), and the
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

impl Observation {
    /// Where it sits: the track keypoint, else the cluster's refined position
    /// or its seed, else nothing.
    pub fn site(&self) -> Option<[f64; 2]>;
    /// The affine shape it is read at, or `None` when only a keypoint says
    /// where it is.
    pub fn shape(&self) -> Option<[[f64; 2]; 2]>;
}

pub struct TrackMeasurement {
    pub keypoint: Option<[f32; 2]>,
    pub zncc: Option<f64>,
    pub seed_shift_px: Option<f64>,          // the peak's move from the sighting
    pub projection_offset_px: Option<f64>,   // the sighting's distance from the point
    pub reprojection_error: Option<f64>,
    pub ray_angle_deg: Option<f64>,
    pub localizability: Option<f64>,
    pub walked_px: Option<f64>,              // set when a fit refused the walk and kept the seed
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
    Descriptor { feature: u32 },     // a detected keypoint, named directly
    Search { inliers: u32 },         // an image a descriptor search found
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

pub fn duplicate(
    bench: &Bench,
    label: &str,
) -> Result<(Bench, DuplicateReport), DuplicateError>;

pub struct DuplicateReport {
    pub label: String,               // the copy's, after the collision suffix
    pub from: String,
    pub observation_count: usize,
}

pub fn split(
    bench: &Bench,
    edited: &EditedReconstruction,
    label: &str,
    observations: &[usize],
) -> Result<(Bench, SplitReport), SplitError>;

// Placing a sighting, and moving, sizing and turning the patch, by hand.
pub fn translate_patch_to_pixel(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    observation: usize,                  // whose image the pixel is in
    pixel: [f64; 2],                     // where the centre should land in it
) -> Result<(EditableTrack, TranslateToPixelReport), TrackEditError>;

/// The move itself: `by` is read on the patch's **own orthonormal axes**
/// `[u, v, n]`, in world units. `u` and `v` slide it across its own plane and
/// `n` along its outward normal, which is the one direction a pixel cannot
/// name; a mixed `by` does both at once.
pub fn translate_patch(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    by: Vector3<f64>,                    // [u, v, n], world units
) -> Result<(EditableTrack, TranslateReport), TrackEditError>;

pub fn sight_observation(
    track: &EditableTrack,
    edited: &EditedReconstruction,       // the photograph the pixel is clamped to
    observation: usize,
    pixel: [f64; 2],
) -> Result<(EditableTrack, SightReport), TrackEditError>;

/// One pixel brought inside `[0, width) x [0, height)`, with the pixel that was
/// asked for when it had to move. The rule the three steps above share, and the
/// one a caller holding a seed applies before `create_cluster` or
/// `add_observation`.
pub fn clamp_to_photograph(
    camera: &CameraIntrinsics,
    pixel: [f64; 2],
) -> (Option<[f64; 2]>, [f64; 2]);

/// The resize itself, to one world half-length on both axes. `moved_edge`
/// names the edge that moves, the far one held, or `None` for both about a held
/// centre -- which is the discriminator for what becomes of the sightings.
pub fn resize_patch(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    half_length: f64,                    // world, both axes
    moved_edge: Option<Edge>,            // None holds the centre
) -> Result<(EditableTrack, ResizeReport), TrackEditError>;

/// The one size gesture that spans both stages, a pixel being meaningful at
/// either where a world half-length is not.
pub fn resize_patch_to_pixel(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    observation: usize,                  // whose outline is being dragged
    edge: Edge,
    pixel: [f64; 2],                     // where that edge's midpoint lands
) -> Result<(EditableTrack, ResizeReport), TrackEditError>;

/// Turn the patch about its centre, by the least rotation, toward the outward
/// normal named -- the other place a pixel cannot name -- stopping
/// `MAX_TILT_DEG` from any observation's camera.
pub fn tilt_patch(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    normal: Vector3<f64>,                // any non-zero length; the direction is read
) -> Result<(EditableTrack, TiltReport), TrackEditError>;

/// How far from an observation's line of sight a tilt may carry the normal.
pub const MAX_TILT_DEG: f64 = 80.0;

pub fn spin_patch(
    track: &EditableTrack,
    angle_rad: f64,                      // about the outward normal
) -> Result<(EditableTrack, SpinReport), TrackEditError>;

pub fn shape_observation(
    track: &EditableTrack,
    observation: usize,
    shape: [[f64; 2]; 2],                // cluster stage only
) -> Result<(EditableTrack, ShapeReport), TrackEditError>;

/// `radius * ||column 0||`: a shape's half-width along `u`, in that image's px.
pub fn half_width_px(shape: [[f64; 2]; 2], radius: f64) -> f64;

pub enum Axis { U, V }

pub enum Edge { PlusU, MinusU, PlusV, MinusV }
impl Edge {
    pub fn axis(self) -> Axis;
    pub fn sign(self) -> f64;            // -1.0 or +1.0
    pub fn name(self) -> &'static str;   // "+u", "-u", "+v", "-v"
    pub const ALL: [Edge; 4];
}

pub struct TranslateToPixelReport {
    pub observation: usize,
    pub image: u32,
    pub pixel: [f64; 2],                 // where the centre now projects in it
    pub center: Point3<f64>,
    pub moved: f64,                      // world units
    pub placed: usize,                   // keypoints written
    pub changed: bool,
    pub clamped_from: Option<[f64; 2]>,  // the pixel asked for, when off the picture
}

/// `TranslateToPixelReport` without the photograph: a caller that names a
/// displacement names no image and no pixel. `by` is reported as it was asked
/// for rather than as a length, because the **sign** of its normal part carries
/// meaning where a tangential move has nothing but its length to report.
pub struct TranslateReport {
    pub by: Vector3<f64>,                // [u, v, n], as it was asked for
    pub center: Point3<f64>,
    pub moved: f64,                      // world units
    pub placed: usize,                   // keypoints written
    pub changed: bool,
}

pub struct SightReport {
    pub observation: usize,
    pub image: u32,
    pub was: Option<[f64; 2]>,
    pub pixel: [f64; 2],
    pub moved_px: Option<f64>,
    pub changed: bool,
    pub clamped_from: Option<[f64; 2]>,
}

pub struct ResizeReport {
    pub observation: Option<usize>,      // the sighting the size was named at
    pub image: Option<u32>,
    pub half: f64,                       // world at the track stage, px at the cluster stage
    pub was: f64,
    pub changed: bool,
    pub pixel: Option<[f64; 2]>,         // the edge's pixel; none for resize_patch
    pub clamped_from: Option<[f64; 2]>,
}

/// What was asked for and what happened are separate fields, the two parting
/// company whenever an observation's own view of the patch caps the turn.
pub struct TiltReport {
    pub degrees: f64,                    // how far it turned, never negative
    pub asked: Vector3<f64>,             // the unit normal named
    pub normal: Vector3<f64>,            // the one the patch now shows
    pub stopped: Option<TiltStop>,       // the observation whose cap ended it
    pub placed: usize,                   // keypoints written
    pub changed: bool,
}

/// One `Option` over the pair, "it was stopped" and "by this observation"
/// being the same fact.
pub struct TiltStop { pub observation: usize, pub image: u32 }

pub struct SpinReport { pub degrees: f64, pub changed: bool }

pub struct ShapeReport {
    pub observation: usize,
    pub image: u32,
    pub shape: [[f64; 2]; 2],
    pub half_px: f64,
    pub was_half_px: f64,
    pub changed: bool,
}

// Grow it from a descriptor index, which reads a file and no photograph.
pub fn search_descriptors(
    track: &EditableTrack,
    observation: usize,
    keypoints: &ImageKeypoints,
    forest: &LazyKdForestU8,
    options: &SearchOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, SearchReport), SearchError>;

pub struct SearchOptions {
    pub constellation: ConstellationParams,  // its `min_inliers` is not read
    pub radius_px: f32,                      // source-image px
    pub min_inliers: usize,
}
pub const DEFAULT_RADIUS_PX: f32;

pub struct SearchReport {
    pub observation: usize,
    pub observation_count: usize,
    pub image: u32,
    pub center: [f64; 2],
    pub constellation: usize,        // keypoints the query asked about
    pub matches: Vec<SearchMatch>,   // most inliers first
}
impl SearchReport {
    pub fn added(&self) -> usize;
    pub fn already_in_track(&self) -> usize;
}

pub struct SearchMatch {
    pub image: u32,
    pub inliers: usize,
    pub correspondences: usize,
    pub affine: [[f64; 3]; 2],       // searched image's px to this image's
    pub pixel: [f64; 2],             // where the warp puts the observation
    pub found: Found,
}

pub enum Found {
    Added { observation: usize },
    AlreadyInTrack { observation: usize },
    OwnImage,
}

pub enum SearchError {
    NoSuchObservation { observation: usize, observation_count: usize },
    NoPlace { observation: usize },
    NoConstellation { radius_px: f32, keypoint_count: usize },
    Index(String),
}

// Grow a track-stage item from its reconstruction geometry and photographs.
pub fn search_geometry(
    track: &EditableTrack,
    observation: usize,                  // the explicit reference appearance
    views: &[ProjectedImage<'_>],         // one per reconstruction image
    options: &GeometrySearchOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, GeometrySearchReport), GeometrySearchError>;

pub struct GeometrySearchReport {
    pub observation: usize,
    pub observation_count: usize,
    pub image: u32,
    pub reference_views: usize,
    pub self_agreement: f64,
    pub matches: Vec<GeometryMatch>,      // ascending image order
}

pub struct GeometryMatch {
    pub image: u32,
    pub zncc: f64,
    pub pixel: [f64; 2],                  // the patch centre's projection
    pub found: Found,
}

pub enum GeometrySearchError {
    NoSuchObservation { observation: usize, observation_count: usize },
    WrongStage { is: StageKind },
    NoFrame,
    NoPlace { observation: usize },
    NoSuchImage { image: u32, view_count: usize },
    Cancelled,
}

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
    pub max_seed_offset_px: f64,            // how far a seed may sit, 64
    pub max_cache_bytes: usize,             // one round's tiles, 256 MiB
}

pub struct FitOptions {
    pub localize: KeypointLocalizeParams,   // open_localizer
    pub refine: KeypointSubpixelParams,
    pub evaluate: EvaluateOptions,          // the reading a fit ends with
    pub noise_floor_px: f64,                // the classification's, 1.0
    pub inverse_depth_z_cutoff: f64,        // the classification's, 4.0
    pub residual_margin: f64,               // the classification's, 0.8
}

pub struct EvaluateReport {
    pub stage: StageKind,
    pub measured: usize,
    pub unmeasured: usize,
    pub reference: Option<usize>,        // the cluster stage's
    pub position: Option<Point3<f64>>,   // where the track stands; a bearing at w = 0
    pub at_infinity: bool,               // which of the two that coordinate is
    pub condition_number: Option<f64>,
}

pub struct FitReport {
    pub evaluate: EvaluateReport,        // the reading of its own result
    pub placed: usize,
    pub kept_at_seed: usize,             // rows the walk bound left at their seeds
    pub position: Option<Point3<f64>>,   // the coordinate: a place, or a bearing
    pub condition_number: Option<f64>,
    pub classification: Option<TrackClassification>,
}

// The one rule every step that triangulates goes through.
pub fn classify_track_rays(
    rays: &TrackRays,
    images: &[ProjectedImage<'_>],
    noise_floor_px: f64,
    z_cutoff: f64,
    residual_margin: f64,
) -> TrackClassification;

pub struct TrackClassification {
    pub at_infinity: bool,
    pub coordinate: Point3<f64>,         // a world point, or a unit direction
    pub reason: ClassificationReason,
    pub condition_number: f64,
    pub inverse_depth_z: f64,
    pub inverse_depth_z_cutoff: f64,
    pub resolvable_distance: f64,
    pub finite_horizon: f64,
    pub max_pair_angle_deg: f64,
    pub finite_rms_px: f64,              // the point reprojected against the sightings
    pub bearing_rms_px: f64,             // the bearing, against the same sightings
    pub residual_margin: f64,
}

impl TrackClassification {
    /// The call and the numbers behind it, as one sentence. A candidate whose
    /// rms is `NaN` reprojected into none of the sightings, and the sentence
    /// names that instead of printing the word.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;   // Display
}

pub enum ClassificationReason {
    WellConditioned,      // the pre-filter settled it: finite
    DepthResolved,        // the z-score reached the bar: finite
    DepthUnresolved,      // it did not, or the solve was degenerate: a bearing
    BaselineTooShort,     // no depth is resolvable at the capture's scale: a bearing
    // The residual check, which overturns either of the criterion's answers.
    FiniteDoesNotExplainTheSightings,
    BearingDoesNotExplainTheSightings,
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
    pub changed: bool,
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
are their own functions: whether the track stage has a patch to read against
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
image index, exactly as every other photometric kernel takes them. A
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

`Descriptor` and `Search` are two kinds because they name two things.
`Descriptor` names **one detected feature**, which the observation sits exactly
on: it is what a cluster started on a `.sift` keypoint carries. A search's
observation sits wherever that image's affine warp puts the pixel that was
searched from, which is in general no feature at all; what stands behind it is
the number of correspondences that agreed on the warp, so that is the number
`Search` carries, and it is what ranks a search's candidates against one
another.

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
reconstruction yet: a position, a flag saying whether that position is a place or
a bearing, an
[`OrientedPatch`](../../../crates/sfmtool-core/src/patch/cloud.rs) frame, a
consensus bitmap and a colour, and per observation a keypoint with its
leave-one-out ZNCC, its reprojection error, its ray angle and its tile
localizability. It is what a track is when put on the bench from a committed
point.

**`TrackPayload::at_infinity` is where that one bit lives, and not the frame's
`w`.** The same three numbers are a world place at `w = 1` and a unit bearing at
`w = 0`, and the frame carries a `w` of its own for the kernels that project its
corners -- but a track can carry the bit with no frame at all. A point put on the
bench from a reconstruction with no `patch_u_halfvec` column has no patch to
read a `w` off, and a bearing it came from is still a bearing: a reader that went
to the frame would call it a place one unit from the world origin, publish it
under `position`, and commit it back at `w = 1`. So `create_track` takes the
point's own `w`, `fit` takes the classification's, and everything that states
which of the two it is holding -- the Track Edit header's word, the wire's choice
of `direction` over `position`, the commit's `w`, the reading's report -- reads
the flag. Where a frame exists the two agree, and every step that writes one
writes both.

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
  by it before framing the patch. A track taken down and put back up is the
  size it was.
- **Everything that draws reads it**: the bench layer's parallelogram in the
  Image Detail panel and the Track Edit panel's tile. A cluster carries its
  radius from the moment it is started, so a seed is drawn at the size it was
  named before anything has read a photograph, and an evaluation that finds the
  same scale leaves the outline where it is.

The template carries no radius of its own. Two copies of one number could
disagree, and a template that disagreed with the seeds it was cut from would be
a cut of a square nothing else was measured over.

### Finite points and bearings

A track-stage track stands on one of two things, and its frame's `w` says
which: a **place** at `w = 1`, whose coordinate is a world point, or a
**bearing** at `w = 0`, whose coordinate is a unit direction and whose frame is
tangent to the direction sphere. That is the `.sfmr` rule for a point row
([`../../formats/sfmr-file-format.md`](../../formats/sfmr-file-format.md)
§ "Points at infinity"), and the payload's `position` holds whichever the frame
says, so the coordinate and the frame's centre are one thing in both.

**Every step that triangulates decides which, on one criterion.** The criterion
is not the bench's: `classify_track_rays`
([`classify.rs`](../../../crates/sfmtool-core/src/bench/classify.rs)) calls
`classify_rays_at_infinity`, the same per-track test
`classify_points_at_infinity` reclassifies a whole reconstruction with and
`find_points_at_infinity` admits discovered tracks on
([`../reconstruction/batch-triangulation-api.md`](../reconstruction/batch-triangulation-api.md)
§ "Consumers"), with the same defaults. It reads the rays and answers with the
coordinate the track takes, a flag, and the number that settled it:

- a well-conditioned in-front solve is **finite** on the condition number
  alone, before any noise model is consulted (`well_conditioned`);
- otherwise the inverse-depth z-score decides: at or above the cutoff the track
  is **finite** (`depth_resolved`), below it a **bearing**
  (`depth_unresolved`), and a degenerate or behind-a-camera solve is a bearing
  too;
- a baseline that cannot place a point even at the capture's own scale
  (`resolvable_distance` short of the camera cloud's extent) is a **bearing**
  (`baseline_too_short`). The reconstruction passes leave such a track alone,
  because leaving it alone is an option when the pass is relabel-only; a fit has
  to write something, and what the numbers say is that the depth is not
  observable.

The bearing a `w = 0` answer carries is the normalised mean of the rays, which
is the robust direction those sightings agree on.

**And then the sightings have the last word.** The criterion above is a
statement about *observability* -- whether this geometry could resolve a depth --
and on an ill-conditioned solve it can clear its own bar on a depth nothing in
the photographs supports: the least-squares midpoint of near-parallel, slightly
inconsistent rays lands wherever the inconsistency throws it, and then reprojects
nowhere near the sightings it was solved from. So both candidates are scored on
the one thing a person looking at the photographs can check -- the rms distance,
in px, from each sighting to where the candidate projects in its own image,
which is
[`observation_metrics`](../../../crates/sfmtool-core/src/bench/evaluate.rs)'
own first number and so the same residual the *Error* column shows -- and:

- a **finite** answer stands only where the point's residual comes under
  `residual_margin` of the bearing's **and** under it by more than
  `noise_floor_px`; otherwise the bearing stands, with the reason
  `FiniteDoesNotExplainTheSightings`;
- a **bearing** answer stands unless the point clears that same bar, in which
  case the photographs place the track at a depth whatever the conditioning
  says, with the reason `BearingDoesNotExplainTheSightings`.

One comparison, asked in both directions, so the two answers cannot be settled
by different rules. The margin is a fraction because the comparison has no
natural scale -- a scene metre is a pixel count that depends on the lens and the
depth -- and the absolute term is there because a fraction alone would believe a
0.05 px residual over a 0.07 px one, which is two roundings of the same answer.
Both residuals and the margin are on every classification, and in every sentence
it writes, because they are the evidence a person reads the call by.

The finite candidate has three degrees of freedom against the bearing's two and
neither is fitted to minimise pixel error, so a point that fits *slightly* better
has bought that with its extra freedom while one that fits clearly better has
found a depth. That is what the margin's default is set by; the constant carries
the argument.

**A fit registers against the frame the track has.** A `w = 0` patch is
tangent to the direction sphere and a fit of one registers against *that*: no
promotion to a provisional depth, because a frame promoted to a depth the rays
do not carry re-warps every view, and the consensus rounds then walk sightings
onto whatever detail the re-warped template happens to match. What comes out of
the fit is the classification's, applied to the frame the fit ran against:

| Was | Is | The frame |
|-----|----|-----------|
| bearing | bearing | the refined direction, the tangent frame re-pinned on it, at the angular half-extents it had |
| bearing | place | the angular half-extents become world ones at the placement distance from the camera-cloud centroid, which is what keeps the patch the apparent size it had when a second sighting gives a bearing its depth |
| place | bearing | the world half-extents become angular by the distance the frame stood at, the rescale `classify_points_at_infinity` applies to a demoted point, and the frame is re-expressed as the tangent one |
| place | place | the centre moves and nothing else does |

All four keep the patch the apparent size it had, so the next round registers
the square the person has been looking at.

**The cluster-to-track upgrade goes through the same criterion**, over the
refined cluster positions: a capture that only ever stated a direction becomes a
`w = 0` track rather than a point at a depth its rays never carried. There the
reference observation's shape is unprojected at unit distance, where a
fronto-parallel half-axis *is* the tangent of the angle it subtends, which is
what an infinity patch's half-extent is.

**A commit writes `w` as the track has it**, and nothing there re-decides: that
was settled by the fit that wrote the frame. A bearing's row carries the unit
direction as its coordinate and a zero normal with a zero normal confidence,
which is what the format states for `w = 0`; the point map and
[`../reconstruction/edited-reconstruction.md`](../reconstruction/edited-reconstruction.md)'s
`replace_point` / `add_point` take either, and the materialised value counts the
row in `infinity_point_count`.

**The hand edits keep a bearing a bearing.** A slide and an edge drag move the
centre, which for `w = 0` is a direction, so the moved centre is renormalised
and the half-extents are divided by the same factor -- scaling a bearing and its
tangent frame together leaves every corner the same direction, which is what
keeps the far edge where it was. A turn is about the frame's own normal, which
for a bearing is minus the direction, so it keeps the tangency. Every one of
them carries the payload's coordinate with the centre.

**A reading reports a bearing as one.** `EvaluateReport` carries `at_infinity`
beside its coordinate for the reason the payload does, and the per-observation
*ray angle* at `w = 0` is the angle between the sighting's ray and the direction
itself. Both the projection and that angle go through the homogeneous transform,
which folds out the pose's translation at `w = 0`: applying it to a unit
direction would measure against a phantom point one unit from the world origin,
which is the one thing a bearing is not.

### The fit's walk is bounded by the person's bar

The kernels a fit runs stay gate-free and cap-free, for the reason
`open_localizer` gives: a sighting that does not belong is turned out by the
person or by a threshold, not deleted from the evidence by a kernel. What *is*
bounded is where a fit may put a sighting. A row whose refined keypoint lands
further than `max_shift_px` from its seed keeps the seed, records how far the
peak sat in `walked_px`, and still casts its ray -- from the seed. The
`FitReport` counts them in `kept_at_seed`.

The bound is not a verdict and turns nothing out: a correlation that jumped onto
a similar detail elsewhere in the photograph would otherwise hand the
re-triangulation a place the person never pointed at, and on a near-parallel
track that is the difference between a bearing and a scrambled point at some
invented depth. The reading that follows scores the row where it sits, like any
other. Only a fit sets and clears the flag; an evaluation leaves it alone,
because the statement is about what a fit did rather than about what the
photographs show.
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

**A half whose every row is `out` still splits.** The rows a person cuts off are
usually the ones the thresholds just rejected, so the common case is a half with
no `in` observation in it at all. The cluster it becomes needs an observation to
cut its template around, and that reference is a **seed and not a judgement**:
the downgrade takes the `in` observation the patch is largest in where there is
one, and otherwise the largest of whatever the half carries, verdicts and all.
The verdicts travel with the rows either way. Only a half with no seed anywhere
in it is refused, with `StageError::NoReference`.

### Duplicating

`duplicate` puts a copy of one item on the bench beside it, labelled
`<label> copy` through the bench's own collision rule, and makes the copy the
active one, because it is the thing about to be worked on.

**What a second patch over neighbouring ground is started from.** A patch slid,
turned and sized until it covers one piece of surface is most of the work of
covering the piece beside it, so the copy carries everything that describes the
geometry and the judgements made about it: the stage and all of its data (the
patch, the consensus bitmap, the cluster's template and its radius), every
observation with its keypoint, its seed, its shape, its verdict and its pin, and
the thresholds. The measurements come too, because they were read against this
geometry and still describe it -- and the moment the copy is moved, the steps
that move it drop the ones that no longer hold.

**The copy has no origin**, and that is the whole of the difference between the
two items. An origin is what makes a commit *replace* a point; a copy is a new
patch over new ground and has to create one, or the second commit would delete
what the first wrote.

### Placing, sizing and turning by hand

Eight steps put a person's own hand on the track's geometry, and they are the
steps behind the bench handles of both panels
([`../../gui/multi-panel-image-browser.md`](../../gui/multi-panel-image-browser.md)
§ "The bench layer") and the wire's eight patch tools.

**Four verbs, no two of them synonyms.** `translate` moves the centre, `resize`
changes the half-length, `spin` turns the square about its normal and `tilt`
turns the normal itself. Each word is said once along the path: a step sits in
`sfmtool_core::bench` and takes an `&EditableTrack`, so it is `tilt_patch`, the
enclosing namespace carrying the rest; nothing is called `bench_track_frame`,
and `frame` is not a third noun the gestures need. It
stays only where the geometry alone is meant, which is the
[`OrientedPatch`](../../../crates/sfmtool-core/src/patch/cloud.rs) type and
`TrackPayload::placement`.

**At the track stage a track has one patch and every observation is a view of
it**, so the gestures over the outline are gestures over *the patch*: it slides,
resizes and turns, and each photograph shows where it lands. That is what makes
them worth having -- a patch can be worked until it covers the piece of surface
a person means. The cluster stage has no shared geometry at all, only one affine
shape per sighting, so there each gesture is that sighting's own.

**A sighting follows the centre, and is rebuilt only when the plane turns under
it.** The track's position *is* the patch's centre, so every sighting of that
point is carried by the centre's own displacement and keeps its in-plane offset
`(a_i, b_i)`: that is what a translation does, and it is equally what a resize
from an edge does, which moves the centre because it holds the far one. A spin
and a resize about the centre move no point of the plane at all and so touch no
sighting -- a spin leaves every keypoint where it is and lets `(a_i, b_i)` turn
under it, which is what re-aligning the sampling square over a surface that did
not move means. A tilt moves no point either, but takes the plane out from under
them, so there is nothing to carry and the offsets are what is kept, each point
rebuilt as `c + a_i u' + b_i v'`. That one rule is most of why `spin_patch` and
`tilt_patch` stay two steps rather than becoming one rotation about two axes:
a single step would have to choose a single rule and would be wrong for the
other axis.

**`translate_patch` moves the patch by `by`**, read on the patch's **own
orthonormal axes** `[u, v, n]` in world units. `u` and `v` slide it across its
own plane and `n` moves it along its outward normal -- the one direction no
photograph can name, a sighting saying which ray the patch lies along and
nothing about how far down it the surface is -- and a **mixed** `by` does both
at once. One step for the two, because they are the same body with a different
displacement: what differs is only how the sightings *look* afterwards. Along
the plane they all move by the same amount in their photographs; along the
normal the plane travels with the patch and they move by *different* amounts,
and that spread is the parallax the old depth was wrong by. `u` and `v` are unit
vectors and the half-extent is stated separately, so `by` is in the world's own
units rather than in half-lengths.

**Every** observation's keypoint is carried by that displacement and keeps its
own in-plane offset from the centre: that offset is where the photograph sees the
patch's content against where the geometry puts its middle, and it is what the
tile is cut on, so resetting the keypoints to the centre's projection would
scramble the correlation the next reading scores. A sighting that has never been
localized has no offset to keep and takes the projection of the new centre; one
the moved centre no longer projects into is left with no keypoint and
`Unmeasured::NoProjection` as its reason, which is the truth about it. The axes,
the normal and the size are untouched. Nothing is pinned: where the patch is
says nothing about whether any sighting belongs to it.

Two refusals, each attached to the part of `by` that earns it. A **bearing**
(`w == 0`) has its moved centre renormalized, which leaves every corner the
direction it was, so a tangential `by` is carried like any other; a `by` with a
**normal** part is refused as `TrackEditError::AtInfinity`, a direction patch's
normal being its own bearing, so there is no line standing off it to move along.
A `by` that is not finite is refused as `BadDisplacement`, the way a pixel that
is not one is refused as `BadPixel`.

**`translate_patch_to_pixel` is that step named as a pixel.** A person dragging the
outline in a photograph names a place by pointing at it, so the pixel is read
against the outline as drawn -- the patch re-anchored on that observation's
sighting -- and the offset of the meeting from *that* centre, read on `u` and `v`
alone, is the `by` `translate_patch` is then given. The consequence is the one
the gesture wants: the sighting the drag came through lands under the pointer,
because its plane point plus the displacement *is* the plane point under the
pixel, and every other sighting moves with the patch. There is one
implementation of the move and both forms reach it.

**`sight_observation` places one sighting**, and one only. At the track
stage it writes that observation's keypoint -- the pixel a commit writes and the
place every reading is anchored at -- and at the cluster stage it re-seeds the
observation at that pixel with the shape it is being read at, because a person
moving a cluster's mark is saying where that patch is and not how large it is.
Either way **every measurement read at the old pixel is dropped**: the
leave-one-out ZNCC, both distances, the reprojection residual, the
localizability and the reason were all computed for a pixel that is not this
one, and an evaluation recomputes all of them from the track as it stands. **The
observation is pinned**, at both stages: a sighting a person placed is a sighting
they have ruled on, so `apply_thresholds` leaves its verdict where it is rather
than painting over a placement by hand. Nothing else on the track moves -- which
is the difference from `translate_patch_to_pixel`, and why the two are separate steps: the
viewer's dot is the translation at the track stage and this at the cluster stage,
and this is also what a script that really means one keypoint asks for.

**`resize_patch` sizes the patch to one half-length on both axes.** One scalar
and not two, because a patch is square: the stored half-vector pair has
`|u| == |v|` and the tile grid is square with it, so a resize that moved one axis
alone would be a stretched template rather than a larger one.

**`moved_edge` is the discriminator for what becomes of the sightings**, which
is why it is a parameter of the step rather than a convenience left to the
caller. `None` moves **both** edges about a held centre: nothing on the patch
moves but its size, so no sighting is touched at all, and this is the form a
caller that knows the size it wants asks for. `Some(edge)` moves that edge and
holds the **opposite** one, which is the gesture -- a person pulling an edge
expects the other three where the geometry puts them, not the far edge running
away -- so with the dragged edge at `+h` from the centre and the far one at `-h`,
the new half-length `h'` moves the centre by `h' - h` along that edge's own
direction, and every sighting is carried along the plane by that same
displacement, keeping its own offset, exactly as a translation's are. The
track's position follows the centre. Nothing is pinned either way. A **bearing**
(`w == 0`) with an edge named is handled by renormalizing the moved centre and
dividing the half-length by the same factor, which leaves every corner the same
direction, so the far edge is held there too.

**`resize_patch_to_pixel` is that step named as a pixel**, and it is the gesture
and the **one step that spans both stages**, a pixel being meaningful at either
where a world half-length is not. *What the outline shows is what is resized*: at
the track stage the outline is the patch re-anchored on `observation`'s own
sighting, which is where a person sees the patch in that photograph, so that is
the square the pixel is read against, and the offset of the meeting along the
dragged edge's axis, read from the patch's own centre, gives the half-length
`(p + h) / 2` that `resize_patch` is then handed with that edge. So the dot and
the outline move together in the image the edge was dragged in, the far edge
really does hold still there, and the outline in every other image moves with the
patch. The arithmetic is core's rather than each caller's, and there is one
implementation of the resize, so a tool call, a drag in a photograph and a drag
in the 3D viewer cannot resize differently.

**`tilt_patch` turns the patch to face a new outward normal**, and is the
other hand step no pixel can name: a sighting says which ray the patch lies
along and nothing about which way the surface under it faces, so the patch's
orientation is settled out in the world too. The turn is the **least rotation**
taking the normal the patch shows onto the one named -- the rotation about
`n_old x n_new` -- and `u` and `v` are both carried by it, so a tilt adds no
spin about the normal; spin is `spin_patch`'s. Where the two normals are
collinear there is no such rotation to speak of: at zero none is needed, and at
a half turn every rotation about an axis in the patch's plane is equally least,
so the patch's own `u` is taken as the direction the arc leaves in, which holds
`v` and reverses `u` and `n`.

The centre and the half-lengths do not move, and each keypoint becomes the
projection of `c + a_i u' + b_i v'`: every observation keeps the in-plane offset
`(a_i, b_i)` it was measured at, read on the patch as it stood **before** the
turn and rebuilt on the turned axes. That is not the rigid carry a translation makes
-- the pair is kept and the point is built again -- but it preserves the same
thing, which is where each photograph sees the patch's content against where
the geometry puts its middle. A sighting the turned patch no longer projects
into is left with no keypoint and `Unmeasured::NoProjection`. Nothing is
pinned: which way the patch faces says nothing about whether a sighting belongs
to it.

**A tilt stops `MAX_TILT_DEG` from any observation.** With `e_i` the unit
vector from the centre to observation `i`'s camera centre, a normal is *allowed*
when the angle between it and `e_i` is at most that, for every observation,
whatever its verdict. Past it a photograph sees the patch so obliquely that its
tile is a smear of a few pixels stretched over the square, and the correlation
that tile is scored by says nothing. The allowed normals are the intersection of
one spherical cap per observation, so the step walks the great arc from the
current normal toward the one named and stops at the last allowed normal on it,
and a caller held against the limit traces the edge of what the existing
sightings can see. An observation **already** past the cap before the turn
constrains nothing, since otherwise a track that starts outside the region could
not be turned back into it. The report carries the normal asked for beside the
one reached, and names the observation that stopped the turn, so a caller can
say so in its own sentence.

The angle the report states is the arc's own, taken as `atan2` of the sine
against the cosine rather than from the rotation's reading of itself: a turn is
judged no turn below a nanoradian, and an `acos` of a cosine alone cannot
resolve that -- a turn of a nanoradian leaves the cosine `1.0` to the last bit,
and near the half turn the error in a cosine that *is* resolved lands on the
answer divided by its sine.

A **bearing** (`w == 0`) is refused as `TrackEditError::AtInfinity`, a direction
patch's normal being its own bearing, so there is no normal standing off it to
turn; a `normal` that is not a finite direction, or is too short to name one, is
refused as `BadNormal`, the way a displacement that is not one is refused as
`BadDisplacement`.

**`spin_patch` turns the patch about its own outward normal.** The axes are
rotated as a pair by a rotation whose axis *is* the normal, so both keep their
lengths, the patch keeps its handedness and it keeps the plane and the face it
had; what changes is which way up the square sits. The centre is untouched, so a
spin moves no sighting.

**`shape_observation` sets one cluster-stage sighting's affine shape**,
which is what a corner drag there hands it: the shape turned about the sighting.
The observation is re-seeded where it is already drawn and its refinement is
dropped, for the reason a move drops one -- the ZNCC, the drift and the status
were the refinement's answer about the shape it was run at. The verdict is
**not** pinned here: a size or a turn is not a ruling on whether the sighting
belongs, which is what a pin protects from the painting.

**What a change to the patch invalidates is cleared.** Every patch step drops
the consensus bitmap and every track measurement but its keypoint: the
bitmap is the observations fused over the square as it stood, and every number
beside a keypoint was read over that square and against that position. Where
each sighting sits is not one of those things, so it stays; an evaluation
restores the rest, and the next fit fuses a new bitmap at the size and turn the
patch now has.

**The stage decides which step applies.** `translate_patch_to_pixel`,
`translate_patch`, `resize_patch`, `tilt_patch` and `spin_patch` are the track
stage's and refuse a cluster; `shape_observation` is the cluster stage's and
refuses a track. `sight_observation` and `resize_patch_to_pixel` work at either
and do the stage's own arithmetic.

**A moved bearing that renormalizes to nothing is refused** as
`TrackEditError::BadPlace`, the way a pixel that is not one is refused as
`BadPixel`: it is the one place a step's own arithmetic can leave a centre that
is no place at all.

**A step that would change nothing reports `changed: false`.** The patch steps
hand the track back exactly as it was; the two that **pin** -- a verdict set again
by hand, a sighting placed where it already sits -- still pin, because a pin is
the person's ruling on the row and not the thing the tolerance is about. The
comparison is not exact, and cannot be: a pixel is turned
into a ray, met with the patch's own plane and projected back, and the round trip
returns the place it started from to within the arithmetic's last bits. An exact
`!=` therefore reads every re-statement of where the patch already is as a move.
So each step judges in the units of the value it moves -- a centre within `1e-6`
of the patch's own half-length, an offset within `1e-6` of that half-length too,
a half-length within `1e-6` of itself, an affine
coefficient within `1e-6` of the shape's largest, a sighting within `1e-3` px, a
turn within `1e-9` rad, a tilt within that same `1e-9` rad of the normal it
already shows -- while the steps that compare a verdict, a stage or a
label compare those exactly, there being nothing to round. What the caller does
with the flag is the caller's: the viewer pushes no version and records the row
that says so ([`../../gui/bench.md`](../../gui/bench.md) § "The wire").

**A pixel is brought inside the photograph it names.** `translate_patch_to_pixel`,
`resize_patch_to_pixel` and `sight_observation` take the nearest pixel of
`[0, width) x [0, height)` -- the sensor's own half-open extent, and the far end
is the largest double **under** the width, a pixel at the width naming a column
the sensor does not have -- and report the pixel that was asked for as
`clamped_from` where it had to move. A patch whose centre is slid until it meets
an extrapolated ray's crossing of its plane lands an arbitrary distance from where
it stood, which is not what a pointer past the edge of the picture or a call
carrying two numbers of its own means. The consequence worth knowing is that a
resize's reach is the photograph: an edge cannot be put where that sighting does
not show it. `clamp_to_photograph` is the rule, public so a caller that holds a
**seed** rather than a gesture can apply it before `create_cluster` or
`add_observation` -- those two do not, because a descriptor search places its
candidates by the warp it found and a warp may put the patch off the edge of an
image that only half holds it, which the reading refuses as `OffSensor` rather
than the step inventing a sighting inside the frame. The two world-point forms
have no photograph to be brought inside and clamp nothing: a place is bounded by
the frame's own plane, which it is projected onto, and by nothing else.

### Searching the descriptor index

`search_descriptors` is the third way an observation reaches a track, beside the
pixel someone pointed at and the point a track was put on the bench from, and it
is the only one that proposes several at a time. It is a **constellation query**
([`../features/kdf-constellation-query.md`](../features/kdf-constellation-query.md)),
not a lookup of one descriptor: a pixel someone pointed at is an extremum of
nothing, so a descriptor computed there matches nothing a detector produced for
the same surface elsewhere. What is stable is the neighbourhood of detected
keypoints around it, which another view of the same surface carries under a
locally affine warp. So the keypoints within `radius_px` of the observation are
looked up in the forest, the hits are grouped by image, and an image whose hits
agree on a single warp with at least `min_inliers` of them is found.

**The warp is what makes a found image worth anything.** Applied to the
observation's own pixel it gives that image a seed position, and its linear part
applied to the observation's own keypoint-frame shape gives that seed a shape,
so a candidate arrives at the place *and* the size the warp says the patch has
there, whether the observation searched from was a detected feature or a
hand-placed pixel. Because both halves of it are read, and read hardest at the
observation's own pixel, the search hands the query that pixel as the
constellation's centre and takes the query's default warp: a least-squares fit
to the consensus, weighted towards the centre
([`../features/kdf-constellation-query.md`](../features/kdf-constellation-query.md)
§ "Three points find a model; the consensus reports one"), rather than the
three-point model RANSAC drew. Both are the cluster stage's own convention (§ "The cluster
stage's units"), which is what the next evaluation reads at either stage: at the
cluster stage the refinement registers the seed, and at the track stage the
candidate is a row the reading measures and the thresholds propose a verdict
for. The step sets no verdict and moves nothing that was already on the track.

**Where the search runs from** is one observation, named by index, and its pixel
is the one everything that draws an observation uses: the track stage's keypoint,
else the cluster stage's refined position or its seed. There is no third source
-- projecting the track's point would need the reconstruction, which this step
does not take -- so an observation with neither is refused rather than guessed
at. The shape falls through the same order `add_observation` does: the
observation's own, else the cluster's reference's, else the identity.

**An image the track already names is left alone**, whatever the verdict on it.
`out` is a decision the person made and a search does not overturn it; a
`candidate` is already on the table. Those images are reported rather than
dropped, and the count of them is in the report's sentence, so "the search found
nothing new" and "the search found nothing" read differently.

**The corpus indexes the reconstruction's images, in its order.** A match names
a corpus image index and the observation it becomes names a node image index,
and this step takes no reconstruction to compare a name table against, so the
two are stated to be one number. The caller that adopts a forest is where that
is checked -- in the viewer, at the moment a `.kdf` is opened
([`../../gui/track-edit.md`](../../gui/track-edit.md)).

The keypoints are the caller's: nothing here opens a `.sift` file, because a
window already holds every image's keypoints to draw them over the photograph
and a read per gesture would be a second copy of what is on screen. The only
file this step reads is the index, and it reads it as file I/O rather than as
pixels, which is why it is a background task in the viewer and takes a
`Progress` with one phase, `query index`.

`SearchOptions::min_inliers` is the bar, and it is the *only* place the bar is
written: it is what the report states a refusal against, and it is written into
the query's own params so an image the search would discard is never fitted.
`SearchOptions::constellation.min_inliers` is not read.

### Searching by geometry

`search_geometry` is the track-stage counterpart to `search_descriptors`: it
asks which other cameras see the patch the track already carries, using the
same patch-view selection that `sfm embed-patches` runs per point. It projects
the finite patch or direction patch (`w = 0`) into every supplied camera,
requires the existing front-facing, cheirality and image-support gates, and
admits a view only when its rendered patch clears the relative-ZNCC bar against
a trustworthy reference appearance. The bar is the editable track's own
`thresholds.min_relative_zncc`; changing the panel slider therefore changes the
next geometry search by the same rule it changes a batch view selection.

**The row names the appearance being searched from.** Its observation is first
in the reference basis even when its verdict is `candidate` or `out`; the
track's other `in` observations follow in observation order. Duplicate images
are removed first-seen, so the selected row wins. Each basis render is anchored
at that observation's own `site()`, while a candidate has no sighting yet and is
scored at the patch's projection. This is patch-view selection's anchored
reference mode: the row gesture chooses real source appearance without giving
up the robust consensus of the observations already accepted.

**An admitted image arrives as a seed and no decision.** Its pixel is the
patch centre's projection. Its shape is the projected `u`/`v` half-frame,
converted from the negative-determinant patch-frame convention into the
positive-determinant cluster/SIFT convention and divided by the cluster radius,
exactly as a track-to-cluster stage change seeds an observation. The row is a
`candidate` with `Provenance::Sweep`, carries no evaluation, and the next
Evaluate or Fit judges it. An image the track already names is reported and
left byte-for-byte alone, including an `out` verdict or a pin; the source image
and all other reference images are excluded by the selector itself. Repeating
the same search is therefore idempotent.

The operation accepts finite and infinity patches because patch-view selection
already renders and projects both homogeneous forms. It refuses a cluster-stage
track, a track with no frame, a source observation with no site, or a reference
image beyond the supplied view table rather than inferring geometry the track
does not carry.

The work reports `build reference`, `score views`, and `add candidates` through
`Progress`. It polls before and after reference construction and between views
and additions; cancellation returns no grown track, so a caller never installs
a partial candidate list. Candidate and report order is deterministic: newly
admitted views are in ascending image index, after the selector's reference
basis.

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
ray grazes the patch plane, with the cosine), `SeedTooFar` (its seed sits
further from the projection than the reading will widen its window for, with the
offset and the bound), `NoConsensus` (fewer than two observations of its round
could be read together) and `Unscorable` (its tile could not be scored: it runs
off the photograph, or no channel of it carries texture). The first five are
decided before any correlation, from the observation and the geometry, which is
what lets the row carry the reason instead of simply going missing from the
kernel's answer.

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

**And the widening is bounded, because it is a memory bound.** Every view of a
round renders a tile `resolution + 4 · window` on a side, so the memory one
round costs is the widening **squared**, per view: a seed a couple of thousand
px from the projection asks for gigabytes of tile in each photograph of the
track. Two numbers keep that from being asked for.

- `max_seed_offset_px`, 64 patch-grid px, is how far a seed may sit from the
  projection and still be read. A seed past it is left out of the round carrying
  `SeedTooFar`, with its own offset and the bound in the sentence, and every
  other observation of the round is read as it always was. Sixty-four is a
  little under three tile-widths at the default `resolution` of 24 -- a sighting
  a couple of patches from the projection is still read -- and a view's tile at
  the bound is about a megabyte and a half rather than a gigabyte. Past it, the
  correlation at the projection would say nothing about the sighting and the
  correlation at the seed is a question about a different surface, so naming the
  row is the whole of the answer.
- `max_cache_bytes`, 256 MiB, is what one round's tiles may take **together**:
  the offset bound holds one observation and this holds the round, which a track
  with enough views at the bound would otherwise exceed. A round past it is
  refused with `EvaluateError::TooLarge`, naming both numbers, **before**
  anything is allocated.

The order matters. An allocation the global allocator cannot make aborts the
process where it stands, taking the window and everything unsaved in it, so
these are decisions made from the parameters -- `plan_rounds` sizes each view's
tile from its seed's offset, and the round's total is read off the same formula
the localizer allocates by -- rather than attempts made and recovered from. What
is left is the unforeseen size, and the localizer reserves its own tiles and
shift grids through `try_reserve_exact`
([`patch-keypoint-localization.md`](../patch/patch-keypoint-localization.md)),
so even that comes back as a refusal.

**A fit runs the same rounds as the reading it ends with**, the bound included:
an observation the reading will refuse to read is one the kernels do not move
either, so the two cannot come to disagree about which sightings were in play.
The fit's own localization is not widened -- it registers each view within
`localize.search` of where it sits -- so the budget bites where the widening is,
which is the reading.

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
**track stage** the patch is localized into every view by the two kernels the
embed pass chains --
[`localize_patch_keypoints`](../patch/patch-keypoint-localization.md) then
`refine_patch_keypoints` -- against the frame the track carries, bearing and
all; the `in` results are re-triangulated, the frame is placed at what they
resolve to (§ "Finite points and bearings") and the consensus bitmap is fused
over them. The payload takes the coordinate, the frame, the fused bitmap, the
colour at that bitmap's centre and the triangulation's condition number, and the
`FitReport` carries the classification: which representation the rays earned,
which of the criterion's tests settled it, and the numbers behind that.

**An observation the kernels did not place keeps its pixel.** Out of frame, or
in no round at all, its keypoint stays wherever it already sat -- the one it
arrived with, or the cluster stage's refined position for a track just upgraded
-- so the column still says where the sighting is and the re-triangulation still
has its ray. One the kernels placed further than `max_shift_px` from its seed
keeps its pixel too, and says so (§ "The fit's walk is bounded by the person's
bar").

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

A fit writes the placement it finds whether or not the observations can be
measured against it, so observations that disagree with the camera poses come
back unmeasured at a moved point. A fit that refuses and reports instead is
proposed in
[`bench-inconsistent-fit-amendment.md`](../../drafts/bench-inconsistent-fit-amendment.md).

### Moving between the stages

`set_stage` is one operation in both directions. Setting the stage a track is
already at gives it back unchanged with `changed` false, so a caller can wire a
toggle straight to it and push no version for a step that did not happen.

**Up, cluster to track**, is the spawn pipeline's own steps over one candidate:

1. **Triangulate and classify** the `in` observations' refined cluster
   positions through their cameras, on the criterion of § "Finite points and
   bearings": the rays answer with a place or a bearing.
2. **Frame** the patch at that coordinate: the in-plane axes and half-extents
   are what the reference observation's affine shape, scaled by the cluster's
   radius, unprojects to on the plane at the triangulated depth
   ([`OrientedPatch::from_affine_shape_at_depth`](../patch/patch-cloud.md)), and
   the normal is the mean viewing direction, which is how `to_embedded_patches`
   frames a patch from the views that see it. For a bearing the same
   unprojection runs at unit distance and the frame becomes the tangent one, per
   that section. The reference is the cluster's
   own when it is `in`, and otherwise the largest-scale `in` observation, which
   is what the cluster kernel would have picked among them.
3. **Localize, refine, re-triangulate, fuse and read back**, which is the
   track-stage fit above over seeds that are the cluster's refined positions.

The cluster-stage measurements are dropped with the stage: each describes a
registration against a reference and a template the track no longer has, and
the track stage measures every observation afresh against the patch.

**Down, track to cluster**, is always possible and lossy on purpose. Each
observation is re-seeded at its keypoint with the affine shape the format
derives by projecting the frame at that observation's anchor
([`../../formats/sfmr-file-format.md`](../../formats/sfmr-file-format.md)
§ "Deriving keypoint shape, scale, and orientation", the inverse of the framing
the upgrade does, so the two directions state one relationship), divided by the
new cluster's radius because that rule's columns are pixel half-axes and a
cluster seed is per keypoint-frame unit, and **read in the other chirality**.

That last is a convention, and it is the whole of the difference between the two
stages' shapes. A patch-frame shape has the projections of `u` and `v` as its
columns, and `v` points image-*up* while pixel rows count *down*, so a patch
facing the camera projects to a **negative** determinant. A keypoint-frame shape
is the `.sift` convention, a scaled rotation of **positive** determinant, which
is what a descriptor search seeds and what the cluster stage rasters its
template with. The two are one negation of the `v` column, applied on the way
down here and on the way up by `OrientedPatch::from_affine_shape_at_depth`,
which negates the second column of a positive-determinant shape so the patch it
builds faces the camera. A seed taken straight from the projection is the patch
*mirrored*, and no size tells the two apart: `det.abs().sqrt()`, which is how
both stages measure how big a patch is in a view, is the same number either way
round.

The reference
becomes the `in` observation with the largest projected patch scale, which is
the one showing the most of the patch -- and, where the track has no `in`
observation left, the largest of whatever it does carry, because a reference is
a seed to cut a template around and not a judgement about the sighting (§ "The
split"). The position, the frame and the
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
the `in` observations: the coordinate the track carries with the frame's own `w`
(§ "Finite points and bearings"), the frame it stands on,
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

**A commit that would change nothing writes nothing.** Where the origin
resolves, already holds exactly the record the commit would write, and nothing
is left to absorb, the value comes back as it stands with `changed: false`,
`point` naming the point that already holds the track, `replaced` `None` and the
map an empty `Chain`; every other commit reports `changed: true`. Otherwise
committing one track twice deletes a point and re-adds an identical one at a new
index for each press, and a value's history fills with versions that say
nothing. The comparison is `PointRecord::agrees_with`, which is exact on the
stored representation -- the `f64` coordinate as stored, the `f32` keypoints, the
`u8` confidences, the whole bitmap -- because what it answers is whether writing
the record would leave the point as it is, and a position that moved by a stored
amount is a point that moved. Its one concession is that two `NaN`s in a column
agree, a free point's constraint distance being `NaN` by definition. The
observations are compared in the order they are stored, which the commit sorts
into, so a bench holding the same sightings in another order is the same track.

Three things are therefore changes even where the record is the one the point
holds: a track with **no origin** (it creates, which a second copy of one
landmark is meant to do -- nothing here looks for an identical point elsewhere in
the value), an origin whose point has **gone** since, and a track with a sighting
still to **absorb** from a live point.

Two columns the commit writes are not carried across from the point, so the
*first* commit of a point put on the bench and left alone is in general a change:
the colour, which is read from the consensus bitmap's centre rather than from the
stored byte, and the `error`, which is the mean of what the last evaluation
measured and is zero for a track nothing has read. Everything else round-trips
exactly, and after that first write the two agree.

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
bull, replacing point 1207, absorbing 2 points`, or, for the commit that wrote
nothing, `Committed track: no effect, point 1207 of bull already holds it`.

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

The reading's two memory bounds are not thresholds either: nothing about them is
a verdict on a sighting, and what they decide is what may be asked of the
machine, so they live on `EvaluateOptions` beside the search radius.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `search_px` | `6.0` | How far from each observation's own pixel the correlation peak is looked for, in patch-grid px. From `KeypointLocalizeParams::default`'s `search`. |
| `max_seed_offset_px` | `64.0` | How far from the point's projection a seed may sit and still be read, in patch-grid px. Past it the row carries `SeedTooFar` and is left out of the round. |
| `max_cache_bytes` | `256 MiB` | What one round's per-view tiles may take together. A round past it is refused with `TooLarge`, before anything is allocated. |

The finite-versus-bearing criterion's two knobs are not thresholds either: what
they move is which representation the geometry earns rather than which sightings
are kept, so they live on `FitOptions` and both default to the values the
reconstruction's own passes use.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `noise_floor_px` | `1.0` | The measurement noise the classification assumes at each sighting, in source-image px; the per-ray angular noise is this over the observing camera's focal length. From `DEFAULT_NOISE_FLOOR_PX`. |
| `inverse_depth_z_cutoff` | `4.0` | The inverse-depth z-score a depth has to reach to be written as a finite point rather than a bearing. From `DEFAULT_INVERSE_DEPTH_Z_CUTOFF`. |
| `residual_margin` | `0.8` | The fraction of the bearing's rms reprojection residual the triangulated point has to come under, on top of beating it by more than `noise_floor_px`, before the depth is believed. From `RESIDUAL_MARGIN`; the constant carries the argument for the value. |

The criterion's third number, the condition-number pre-filter that settles a
well-conditioned solve before any noise model is consulted, is not an option
here: it is the criterion's own constant, and a bench that could move it would be
a second classifier.

The descriptor search has bars of its own, which are not the track's: they say
what the *index* is asked, and nothing about them is a verdict, so they are
`SearchOptions` and not `Thresholds`.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `radius_px` | `DEFAULT_RADIUS_PX`, 128.0 | The constellation's radius around the observation, in source-image px. The constant is what the radius rule gives a full-frame capture at fifty features; a caller that knows its frame and its keypoint count computes its own with `radius_for_feature_count`, which is what the viewer does. |
| `min_inliers` | `8` | Fewest agreeing correspondences an image needs to be found. From `ConstellationParams::DEFAULT`. |
| `constellation` | `ConstellationParams::DEFAULT` | The query's own tunables: `k` neighbours per keypoint, the RANSAC budget and pixel threshold, the scale change a warp may claim, and the `refit` the reported warp is fitted by. Its own `min_inliers` is not read. |

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

**The triangulation refuses only a coordinate that is not a number.** An
infinite condition number and a point behind one of the observing cameras used
to be refusals here, as they are for every other caller that re-solves one
track; on the bench they are exactly what a track at infinity looks like, and
the classification reads them as the evidence for a bearing (§ "Finite points
and bearings"). Refusing them would take the answer away from the step whose job
it is to give one.

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
`"candidate"`; `"origin"`, `"descriptor"`, `"search"`, `"sweep"`, `"pixel"`,
`"point"`). Refusals are `ValueError` carrying the core sentence.

`create_cluster` takes either `radius_px`, a half-width in that image's pixels,
or `shape`, a 2x2 in keypoint-frame units; `EditableTrack.radius` is the
cluster's radius those units are read over, and is `None` at the track stage.

`duplicate` takes the bench and the label and hands back `(Bench, report)`, as
`split` does; the report carries `label`, `from` and `observation_count`.

`apply_thresholds` takes each bar as a keyword and moves only the ones given, so
a script can differ from the pipeline's default in one number without restating
the others. `commit` takes the reconstruction's name as `node`, which is what
the report's `label` reads; its report carries `changed`, and `replaced` is
absent for the commit that wrote nothing, as it is for one that created.

`evaluate`, `fit` and `set_stage` take `images` the way every patch kernel does
-- a list of `HxW[xC]` `uint8` arrays, one per image of the reconstruction, or a
prebuilt `ImagePyramidSet`. `evaluate` and `fit` take three optional keywords --
`search_px`, the radius the reading looks for each peak in, and
`max_seed_offset_px` and `max_cache_bytes`, the two memory bounds above, each
defaulting to the reading's own. `fit` and `set_stage` take the
classification's three knobs as well, `noise_floor_px`,
`inverse_depth_z_cutoff` and `residual_margin`, each defaulting to core's own
value;
`set_stage` takes
the stage as the word `"cluster"` or `"track"`. Their reports are dicts:
`stage`, `measured` and `unmeasured`, with `reference` at the cluster stage and
`at_infinity` with `position` or `direction` and `condition_number` at the track
stage; `placed`, `kept_at_seed`, `at_infinity` with `position` or `direction`,
`condition_number`, `classification` and the reading's own `evaluate` report for
a fit; and
`from`, `to`, `changed`, the upgrade's `fit` report and the downgrade's
`reference` for a stage change. An observation's `"track"` dict carries
`reason`, the sentence, exactly when it carries no `zncc`, and `walked_px`
exactly when the last fit refused to walk that sighting and kept its seed.

**The coordinate crosses under the name of whichever it is**, in the reports and
on the track: `EditableTrack.at_infinity` says which, `position` is the world
point and is `None` for a bearing, and `direction` is the unit bearing and is
`None` for a place. A caller that read `position` off a `w = 0` track would be
holding a place one unit from the world origin. A fit's `classification` dict
carries `at_infinity`, `position` or `direction`, `reason` (the lowercase words
`"well_conditioned"`, `"depth_resolved"`, `"depth_unresolved"`,
`"baseline_too_short"`, `"finite_does_not_explain_the_sightings"`,
`"bearing_does_not_explain_the_sightings"`), `condition_number`,
`inverse_depth_z`,
`inverse_depth_z_cutoff`, `resolvable_distance`, `finite_horizon`,
`max_pair_angle_deg`, `finite_rms_px`, `bearing_rms_px`, `residual_margin` and
`text`, the sentence the Action Log shows.

`spin_patch` and `shape_observation` take their numbers directly;
`sight_observation`, `resize_patch`, `translate_patch_to_pixel` and
`resize_patch_to_pixel` take the reconstruction too, because they read an
observation's camera to unproject a pixel or to carry a sighting, and the two
that name an edge name it as the word `"+u"`, `"-u"`, `"+v"` or `"-v"`.
Their reports are dicts of the fields above, with `shape` as a 2x2 array.
`EditableTrack.placement` is what the patch steps are read back through: the
patch as `center`, `u_halfvec`, `v_halfvec` and `w`, the half-vectors being the
axes scaled by the half-extents the way a `.sfmr` stores them, and `None` at the
cluster stage or before anything has fitted one.

`search_descriptors` takes the searched image's keypoints as the two arrays a
`.sift` read gives -- an `(N, 2)` float32 of positions and an `(N, 2, 2)` of
affine shapes, in that file's own row order -- and an open
`sfmtool._sfmtool.spatial.LazyKdForest`. `radius_px` and `min_inliers` are
keywords beside the constellation query's own; its report is a dict carrying
`observation`, `observation_count`, `image`, `center`, `constellation`, `added`,
`already_in_track`, `sentence` and `matches`, one dict per found image with its
`image`, `inliers`, `correspondences`, `affine`, `pixel` and `found` --
`"added"` or `"already_in_track"` with the observation index that goes with it,
or `"own_image"`.

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
synthetic textured-plane scene
[bench/tests/scene.rs](../../../crates/sfmtool-core/src/bench/tests/scene.rs)
builds, wrapped as an
`embedded_patches` reconstruction whose stored keypoints are the exact
projections, so what a commit should have written is known to the pixel. It
covers: a point put on the bench being at the track stage with every observation
`in` and the stored numbers carried; two observations in one image not both
being `in`; the painting proposing from the measurements, leaving a pinned
verdict alone and giving one image one `in`; a split taking exactly the named
observations, handing the half it takes off back as a cluster, and refusing an
empty list or all of them; a duplicate carrying every observation and all of the
stage's data, dropping the origin so its commit creates a point rather than
replacing the original's, taking the collision suffix on a second copy of the
same track, becoming the active item, and leaving the original exactly as it
was; and every commit path -- appending, replacing,
absorbing a pulled-from point, the map each of those reports, and each refusal
naming why.

The commit that writes nothing has a slice of its own: ten presses after the one
that wrote the point leave the value, the indexes and the point count where the
first left them and report the same point each time; the same sightings in
another order are the same track; a sighting turned out, a point taken back, a
track with no origin and a sighting still to absorb each write again; every
column the commit writes is moved in turn and each is seen; and the two columns a
first commit of an untouched point rewrites -- its colour and its error -- are
stated as what they are.
[edited/tests.rs](../../../crates/sfmtool-core/src/reconstruction/edited/tests.rs)
holds the comparison itself: a record agrees with itself, `NaN` columns
included, and disagrees with every one-column move of it, each by the smallest
step the stored representation holds.

The hand steps are tested for what makes them worth having. A **slide** is run
under a pinhole and under a distorting lens: the centre lands under the pointer
in the image it was dragged in (exactly, and within the lens's inverse-map
tolerance), the offset lies in the patch's own plane with the normal, the axes
and the size untouched, every sighting keeps its own in-plane offset from where
the centre projects, nothing is pinned, and a slide to the place the
patch already sits moves it nowhere. A single-keypoint move writes the keypoint,
pins that observation and leaves nothing that was read at the old pixel, and
moving a cluster sighting keeps the shape it is read at. A resize
from an edge is checked **through a lens**, over the same fixture with its
camera swapped for one with real radial distortion: the dragged edge reprojects
onto the pixel the call named, the far edge reprojects onto the pixel it was
already on, and the frame stays square -- exactly under a pinhole, and within
the lens's own inverse-map tolerance under distortion, which is the only error
there is, since the resize itself is arithmetic in the patch's plane. The same
is asserted for a **bearing** (`w == 0`), whose centre is renormalized. A turn
keeps both axes' lengths and the normal, moves no sighting, and lands the corner
under the pixel a drag of that corner would have released on. At the cluster
stage the edge drag scales the shape by one scalar -- so the detector's
anisotropy survives -- moves the sighting by half the change, and holds the far
edge of the parallelogram.

The **world-point forms** are tested for the two claims that are theirs alone.
A place off the plane moves the centre by its in-plane part and by nothing else,
and a place naming where the centre already stands is not a move; an edge given
a place lands on it with the far edge held and the frame square; and a bearing
comes back on the unit sphere from both, with its far edge still reprojecting
where it was. Beside them the reduction itself: the same pointer, put through
the pixel form and through the world-point form with the place it names, leaves
the patch at one centre and at one half-length, which is what having a single
implementation of each edit means. A place that is not a finite point and a
cluster are each refused in their own words.

The **offset** is tested for the pair of claims that make it the step no pixel
can name. The centre travels exactly the distance asked for along the outward
normal while the axes, the half-length and the plane's orientation stay where
they are; and, with every sighting first put somewhere of its own on the plane
so the claim is not vacuous, each keeps its `(a_i, b_i)` and each keypoint comes
back as the projection of `c' + a_i u + b_i v` -- computed in the test from the
pair read *before* the move, so the step is held to the arithmetic rather than
to its own. The sightings move by different amounts, which is the gesture: a
step that read each ray against the plane it had already reached would preserve
neither. Beside them: a patch pushed out past the cameras leaves every sighting
with no keypoint and `NoProjection`; the bitmap and the measurements go and
nothing is pinned; an offset inside the patch's own tolerance reports
`changed: false` and hands the track back; and a distance that is not one, a
track at infinity and a cluster are each refused in their own words.

The **tilt** is tested for the three claims that make it a tilt rather than
some other turn. It is the **least** rotation: the axis is perpendicular to
both normals, so `u` and `v` keep their own components along it and no spin
comes with the turn. The centre and both half-lengths are untouched. And, with
every sighting first put somewhere of its own on the plane so the claim is not
vacuous, each keeps its `(a_i, b_i)` and each keypoint comes back as the
projection of `c + a_i u' + b_i v'` -- computed in the test from the pair read
*before* the turn and the axes read after it, so the step is held to the
arithmetic rather than to its own.

The cap is tested from both sides. A turn asked 89 degrees over stops with its
normal exactly `MAX_TILT_DEG` from the observation the report names, with no
observation past the cap and the normal asked for still in the report; and a
track built facing 85 degrees off one camera and 69 off the other -- so it
starts outside one cap -- is turned back inside without being stopped at all,
which is the rule that keeps the region reachable from outside it. Beside them:
a sighting whose own piece of the patch swings out behind its camera is left
with no keypoint and `NoProjection` while the centred one survives; the bitmap
and the measurements go and nothing is pinned; a turn onto the normal the patch
already shows reports `changed: false`; and a direction that is not one, a
track at infinity and a cluster are each refused in their own words.

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

The two memory bounds are tested as the bounds they are: an observation seeded
past `max_seed_offset_px` comes back named with `SeedTooFar`, carrying its own
offset and the bound, while the rest of the round is read as it always was and
raising the bound past that offset puts the row back in; and a budget nothing
fits in refuses with `TooLarge` naming both numbers, where the same track reads
at the default budget. The allocation itself is tested where it is made
([keypoint_localize/tests.rs](../../../crates/sfmtool-core/src/patch/keypoint_localize/tests.rs)):
a buffer of 256 TB comes back as `LocalizeError::OutOfMemory` rather than
aborting the process, and the shift grids refuse a span no machine has the
memory for.

The split of a half whose every row is `out` is tested too: it splits, the half
that comes off is a cluster cut around the one row it has, and the verdicts
travel with the rows.

**The finite/infinity boundary is tested by turning the capture, not the
picture.** The fixture's scene takes its camera centres and its plane depth as
arguments, with the texture's frequency scaled by that depth so a plane two
hundred units out photographs as the same detail a plane four units out does.
Eight cameras stepping by five centimetres at that far plane give a third of a
pixel of parallax over the whole run, and one ninth camera twenty units off to
the side gives real baseline, so promotion and demotion are two settings of one
dial. Over that: a bearing fits and stays a bearing, at the half-extents it had,
with every sighting inside a pixel of where a *reading* of the same track put it
and scoring what the reading scored -- which is the claim, since a fit that
promoted the frame to a provisional depth re-warps every view and the consensus
rounds then walk sightings onto some other facade detail; a bearing's per-row ray
angle agrees with the direction to a thousandth of a degree, where measuring it
against a phantom point one unit from the origin reads tens of degrees; the same
sightings plus the offset camera's promote to a point on the plane with the frame
grown by the placement distance; the same eight sightings stored as a *finite*
point demote to a bearing with the frame shrunk by the distance it stood at; a
sighting moved four pixels off with the bar at two keeps its seed, carries
`walked_px` and is still scored there while the other seven move; a bearing taken
down to the cluster stage and back up comes back a bearing at the size it was and
commits as a `w = 0` row with a unit direction, a zero normal and a zero normal
confidence, counted in the materialised value's `infinity_point_count`; and a
slide, an edge drag, a centred resize and a turn each leave a bearing's
coordinate on the unit sphere with the payload's coordinate following the frame's
centre. The near scene's own track is asserted to come back **finite**, on the
condition-number pre-filter, so the two answers are both pinned.

**The residual check has a fixture whose criterion answer is wrong.** Eight
sightings on the far scene, tilted across the run by 0.2 px per camera and thrown
off that tilt by 1.5 px alternating, are consistent with a bearing to 1.6 px rms
while the midpoint solve reads a depth of 3.4 units out of the inconsistency. The
test asserts the criterion's own answer first -- `Finite`, with the
condition-number pre-filter *not* firing and the z-score clearing the bar -- so
what it then checks is the check and not the criterion: the point reprojects at
5.1 px rms, over three times the bearing's, the call comes back at infinity with
`FiniteDoesNotExplainTheSightings`, and the sentence names both residuals. The
near scene's track is run through the same comparison and survives it, the point
beating the bearing by far more than the margin. And the whole fit over the
skewed shape writes the bearing, at the frame size it arrived with.

**The sentence is also read back over a candidate with no residual at all**, a
classification built in the test with one side's rms `NaN` and then both: a
candidate that reprojects into none of the sightings is named in words, in every
one of the sentence's four shapes, and the word `NaN` appears in none of them.

The search is tested over a corpus built in the test
([bench/search/tests.rs](../../../crates/sfmtool-core/src/bench/search/tests.rs)):
a patch planted in three images under two warps the test states, written to a
`.kdf` and reopened, so the seed a candidate takes is a number the assertions
can name rather than merely something that appeared. It covers the searched
image never being a candidate of its own search; the found image's candidate
landing at the observation's pixel and shape under the planted warp, as a
`candidate` with the search's own provenance and inlier count; an image the
track already names being reported and left exactly as it was, pin and `out`
verdict included; a bar no image reaches leaving the track untouched; and the
three refusals -- an observation past the end, one with no place in its
photograph, and a radius holding no indexed keypoint.

[tests/rust_bindings/test_bench_rust_bindings.py](../../../tests/rust_bindings/test_bench_rust_bindings.py)
covers the same surface through the bindings, over the 17-image seoul_bull solve
converted to `embedded_patches`.

The geometry search is covered in
[bench/tests.rs](../../../crates/sfmtool-core/src/bench/tests.rs) over the
textured-plane scene: a matching third view lands at the exact patch
projection with positive-chirality seed geometry; existing observations are
unchanged; repeating the search leaves an `out`, pinned candidate untouched;
an explicitly selected `out` row remains part of the reference basis; the
cluster stage is refused; and the real call reports all three phases and stops
on cancellation without returning a partial track.

## Non-goals

- **Building the descriptor index.** `search_descriptors` takes an open forest;
  making one out of a capture's `.sift` files is the viewer's
  ([`../../gui/track-edit.md`](../../gui/track-edit.md)) or a script's.
- **The pairwise coherence matrix.** An evaluation scores each observation
  against the others' consensus; the `k x k` matrix that shows a track made of
  two surfaces as two blocks is
  [`member_zncc_matrix`](../patch/member-coherence-validation.md), and reading
  it into the track is proposed in the same draft.
- **Pulling observations in from another point or another item.** The
  `Provenance::Point` a commit absorbs is set by the caller today; the step that
  reads a point's track and adds it is proposed in the same draft.
- **Editing the patch's frame or normal by hand.** The frame is what the
  kernels fit.
- **Bundle adjustment after a commit.** The commit writes a record and nothing
  settles around it.
