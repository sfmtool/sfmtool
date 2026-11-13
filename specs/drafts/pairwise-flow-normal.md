# Pairwise Flow Normal

**Status:** Draft. A proposal for a new bench normal step; nothing here is
built. The design was prototyped in Python outside the repository and measured
against reference normals on real points and on synthetic scenes; the
measurements are in
[pairwise-flow-normal-measurements.md](pairwise-flow-normal-measurements.md).
Parts of the design were measured and parts were not: each default in
§ [Parameters](#parameters) says which.

Complements: [core/bench/grown-plane-normal.md](../core/bench/grown-plane-normal.md),
the bench's grown-plane normal step, which it stands beside and does not replace.
Builds on: [core/features/optical-flow.md](../core/features/optical-flow.md), the
DIS optical flow it reads; [core/camera/image-warping.md](../core/camera/image-warping.md)
§ "Choosing the sampler per view", the sampler its crops are rendered with.
Later steps would touch [core/bench/refine-normals.md](../core/bench/refine-normals.md)
and [core/bench/editable-track.md](../core/bench/editable-track.md) § "Estimating
the normal" (§ [Later steps](#later-steps)).

## Purpose

A patch in a patch-based reconstruction is a small oriented square standing in
the world at a 3D point, and its normal says which way the square faces. The
bench's grown-plane normal step
([grown-plane-normal.md](../core/bench/grown-plane-normal.md)) estimates the
normal cheaply: it registers a few views against one reference view through the
plane, growing a window from the point out to the patch. It has three
properties that this step is meant to avoid:

- its answer can depend on the reference view it picks and on the normal it
  starts from (the ground truths' points are labelled `ok`, `A1` and `A2` by
  whether the step returns to the same plane from every start;
  [grown-plane-normal-measurements.md](../core/bench/grown-plane-normal-measurements.md#σ-in-the-planes-own-frame-2026-11-11));
- it fits exactly the patch, so the support the plane is read over is whatever
  size the patch was given, which is 3.4 photograph px in half-size at the
  median on a fresh solve of seoul_bull;
- it turns the patch whenever a view joins, unless the turn is large on weak
  evidence; its only outcome with no plane is that no view joined, and none
  says that the views joined but do not determine the normal.

This draft proposes a deliberately expensive estimator for one point,
`pairwise_flow_normal`. It renders every observation's photograph onto one
plane through the point, computes dense optical flow between the renders of
every pair of observations, and fits the plane through the point to each pair's
flow over discs of several radii. It chooses the support itself: the largest
radius at which the pairs agree with each other within their own precision. It
does this in a few rounds, each rendering on the plane the previous round found,
the first on the normal it is given. It reads no reference view, and it says
when it has no normal, for one of two reasons kept apart by the glossary's
**turned** / **undetermined** / **unstable** entry ([GLOSSARY.md](../GLOSSARY.md)):
*undetermined* when the photographs carry too little evidence about the tilt at
any radius (no radius has two pairs whose fits are precise enough), and
*unstable* when pairs are precise but disagree at every radius. On either
outcome the step leaves the given normal unchanged, as `grid_plane_normal` does
on its `GridPlaneOutcome::Undetermined` and `GridPlaneOutcome::Unstable`, and the
report says which.

The step starts from the normal it is given, which in a reconstruction's
pipeline is the mean viewing direction and in the viewer whatever the patch
faces, possibly a normal set by hand or by another step. It must not assume the
mean viewing direction, and a point it does not turn keeps the normal it was
given. Its starting normal only places the plane the first round renders on;
each pair's fit searches the tilt afresh. Whether the answer depends on the
start beyond that has been measured only from the mean viewing direction
(§ [Open questions](#open-questions)).

It spends far more compute than the grown-plane step: one dense flow per pair
of observations per round, and many small fits. The uses proposed for it are:

- a refinement step on the bench, for a point a person is working on;
- a check on the grown-plane step's output, point by point or over a
  reconstruction;
- the test of whether a point's normal is determined at all.

## Rust API

Proposed in `crates/sfmtool-core/src/bench/normal/pairwise_flow.rs`, beside
[grown_plane.rs](../../crates/sfmtool-core/src/bench/normal/grown_plane.rs),
re-exported from `sfmtool_core::bench`, with the signature every bench normal
step has ([bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs)), so
Track View and the wire could call it the way they call the others.

```rust
/// How pairwise_flow_normal renders its crops, computes its flows, fits its
/// planes and chooses its support. Defaults and their sources in § Parameters.
pub struct PairwiseFlowNormalOptions {
    /// How many rounds the step runs, each rendering on the plane the last
    /// found, the first on the given normal; a point with one pair runs one.
    /// 3.
    pub rounds: u32,
    /// A view more than this from the round's plane normal, at the point, is
    /// left out of that round. 80.
    pub max_view_angle_deg: f64,
    /// A normal more than this from the mean viewing direction of the point's
    /// views is no answer: the views cannot see such a plane. 80.
    pub max_normal_from_mean_view_deg: f64,
    /// The radius whose combined normal, over the pairs with a finite σ, is
    /// the next round's plane when a round has no answer, in patch
    /// half-extents. 1.
    pub fallback_radius: f64,
    /// The grid is one sample per photograph pixel of the finest view, but at
    /// least min_samples_per_patch and at most max_samples_per_patch along one
    /// side of the patch. 24, 128.
    pub min_samples_per_patch: u32,
    pub max_samples_per_patch: u32,
    /// The crop's half-width in the world, in patch half-extents. 3.
    pub crop_half_extents: f64,
    /// The least crop half-width, in photograph px of the point's finest
    /// view. 48.
    pub min_crop_half_width_px: f64,
    /// The most samples along one side of a rendered crop; past it the grid
    /// is coarsened. 512.
    pub max_crop_samples: u32,
    /// The disc radii the planes are fitted over, in patch half-extents,
    /// ascending. [0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12].
    pub radii: Vec<f64>,
    /// The flow between two crops. high_quality with finest_scale Some(0)
    /// and patch_stride 1.
    pub flow: DisFlowParams,
    /// The coarse grid of normals each pair's fit starts from: tilts up to
    /// max_grid_tilt_deg in steps of grid_tilt_step_deg, at azimuths
    /// grid_azimuth_step_deg apart. 85, 10, 30.
    pub max_grid_tilt_deg: f64,
    pub grid_tilt_step_deg: f64,
    pub grid_azimuth_step_deg: f64,
    /// Tukey's constant for the pair fit's robust weights, in units of the
    /// whitened residuals' robust scale. 4.685.
    pub tukey_c: f64,
    /// How far apart two flow errors are before they are independent, in
    /// pixels of the pair's coarser view as its crop reads them (one sample
    /// per pixel for a view read through its pyramid); also the side of the
    /// square the crops' structure tensors are summed over. 8.
    pub correlation_length_px: f64,
    /// The share of pure-noise windows whose larger tensor eigenvalue may
    /// pass the significance threshold. 0.001.
    pub noise_false_alarm: f64,
    /// A floor on the precision of one correlated square's flow, in pixels of
    /// the pair's coarser view as its crop reads them: its covariance is the
    /// texture's plus this squared. 0.03.
    pub flow_floor_px: f64,
    /// The largest σ a normal may have and still be a determined answer: a
    /// pair's fit, a radius's combined normal, or a single pair's. 10.
    pub max_sigma_deg: f64,
    /// A pair joins the combined normal at a radius when its normal is within
    /// agreement_sigmas times its sigma plus agreement_margin_deg of the
    /// pairs' medoid. 3, 5.
    pub agreement_sigmas: f64,
    pub agreement_margin_deg: f64,
    /// The pairs agree at a radius when their mean chi-squared is at most
    /// this. 2.
    pub max_mean_chi2: f64,
    /// The radius a point with one pair is read at, where there is no
    /// agreement to test, in patch half-extents. 1.
    pub single_pair_radius: f64,
    /// The reading the step ends with, as for the other normal steps.
    pub fit: FitOptions,
}

impl PairwiseFlowNormalOptions {
    /// Refuses a setting the step cannot run with, naming the first that
    /// fails: NormalError::BadPairwiseFlowOption("radii") for an empty,
    /// unsorted or non-positive list, ("rounds") of 0, ("min_samples_per_patch")
    /// under 2 or over max_samples_per_patch, ("crop_half_extents") under 1,
    /// and so on. The flow's own settings are checked here as
    /// compute_optical_flow checks them; a pair whose square at its level is
    /// narrower than the flow's patch_size is not refused but has no flow and
    /// no fits.
    pub fn check(&self) -> Result<(), NormalError>;
}

/// What the step did, carried by NormalEstimate::PairwiseFlow(Box<_>).
pub struct PairwiseFlowRounds {
    pub rounds: Vec<PairwiseFlowRound>,    // in the order run
    pub outcome: PairwiseFlowOutcome,      // the last round's
    pub chosen_radius: Option<f64>,        // the last round's, patch half-extents; None: Undetermined or Unstable
    pub normal: Option<[f64; 3]>,          // the last round's combined normal at its chosen radius
    pub sigma_deg: Option<f64>,            // its sigma, largest axis
    pub turn_deg: Option<f64>,             // from the given normal to it
}

/// One round: the renders on one plane, the flows and fits, and its answer.
pub struct PairwiseFlowRound {
    pub plane_normal: [f64; 3],            // the plane the crops were rendered on
    pub sample_spacing: f64,               // world units per grid sample
    pub samples_per_patch: f64,            // the grid used: by the rule, or fewer under the cap
    pub crop_half_width: u32,              // the rendered crops' half-width, grid samples
    pub crops: Vec<PairwiseFlowCrop>,      // one per observation of the track
    pub pairs: Vec<PairwiseFlowPair>,      // every pair of rendered crops
    pub radii: Vec<PairwiseFlowRadius>,    // one per options.radii entry
    pub outcome: PairwiseFlowOutcome,      // by the rule, on this round alone
    pub chosen_radius: Option<f64>,        // patch half-extents
    pub normal: Option<[f64; 3]>,          // the answer; None: no answer by the rule
    pub sigma_deg: Option<f64>,
    pub from_mean_view_deg: Option<f64>,   // the answer's angle from the views' mean viewing direction
    pub next_plane: Option<[f64; 3]>,      // the next round's plane: the answer, else the fallback; None: the step stopped here
}

/// Turned: the patch was turned to the last round's normal at its chosen
/// radius.
/// SinglePair: one pair, read in one round at single_pair_radius with no
/// agreement test, its σ at most max_sigma_deg and its normal within
/// max_normal_from_mean_view_deg of the mean viewing direction, and turned.
/// Undetermined: too little evidence. No radius has two determined pairs
/// (each with σ at most max_sigma_deg), so no agreement can be tested, or a
/// point with one pair has no fit at single_pair_radius or its σ there is
/// over the bar; the track as given.
/// Unstable: some radius has two or more determined pairs, but at every such
/// radius they disagree (mean χ² over max_mean_chi2), or the answer is more
/// than max_normal_from_mean_view_deg from the mean viewing direction; the
/// track as given.
pub enum PairwiseFlowOutcome { Turned, SinglePair, Undetermined, Unstable }

pub enum PairwiseFlowCrop {
    /// angle_deg: between the view's line of sight and the round's plane
    /// normal, at the point. samples_per_pixel: how many grid samples one of
    /// the view's own pixels spans at the point, along its coarser axis on
    /// the plane; over 1 the view was upsampled, under 1 it was read through
    /// the photograph's pyramid. coverage: the share of the crop's samples
    /// that fell on the photograph.
    Rendered { image: u32, angle_deg: f64, samples_per_pixel: f64, coverage: f64 },
    /// More than max_view_angle_deg from the round's plane normal.
    TooOblique { image: u32, angle_deg: f64 },
    /// Not in, a second row in one image, or a camera centre at the point.
    NotAttempted,
}

pub struct PairwiseFlowPair {
    pub observations: [usize; 2],
    pub images: [u32; 2],
    pub angle_deg: f64,                    // between the two views, from the point
    pub level: u32,                        // the crops' pyramid level the pair is read at
    pub half_width: u32,                   // the central square read, samples at that level
    pub fits: Vec<Option<PairwiseFlowFit>>, // one per radius; None: past the square read, too few samples on both crops, or a singular Hessian
}

pub struct PairwiseFlowFit {
    pub normal: [f64; 3],
    pub sigma_deg: f64,                    // largest axis: the larger of the two below
    pub sigma_texture_deg: f64,            // from the texture-weighted Hessian
    pub sigma_residual_deg: f64,           // from the fit's residuals
    pub offset: [f64; 2],                  // the pair's free shift, grid samples
    pub rms_residual: f64,                 // grid samples, weighted
    pub determined: bool,                  // sigma_deg at most max_sigma_deg
    pub combined: bool,                    // within the agreement window at this radius
}

pub struct PairwiseFlowRadius {
    pub radius: f64,                       // patch half-extents
    pub normal: Option<[f64; 3]>,          // the combined normal; None: no determined pair
    pub sigma_deg: f64,
    pub mean_chi2: f64,                    // over the determined pairs; NaN with fewer than two
    pub pairs_fitted: usize,
    pub pairs_determined: usize,
    pub pairs_combined: usize,
}

pub fn pairwise_flow_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &PairwiseFlowNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError>;
```

`NormalError` would gain two variants: `BadPairwiseFlowOption(&'static str)`,
as `BadGrownPlaneOption` is for the grown-plane step, and `Flow(OpticalFlowError)`
for a flow refusal the options check could not foresee.

The report's sentence (`Display`) would read, for example, "3 rounds; 21 of 21
pairs agree at 1.5 half-extents (χ² 1.4), σ 1.8°". A point the step does not
turn ends with the words `grid_plane_normal` uses for the same two outcomes: "3
rounds; 15 pairs, no radius with two pairs under σ 10°, normal left unchanged:
undetermined", or "3 rounds; 15 pairs, no radius where they agree (least χ² 3.7
at 0.75 half-extents), normal left unchanged: unstable".

### Why it is shaped this way

**The signature is the other normal steps'.** The step ends as they do: it
turns the patch with `tilt_patch`
([bench/steps.rs](../../crates/sfmtool-core/src/bench/steps.rs)) and reads the
track back through `turn_and_read` in
[bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs), so a caller can
swap one step for another and compare `NormalReport`s. Under `Undetermined`
and `Unstable` it reads back the track as given, as `grid_plane_normal` does.
The two are kept apart, as there, because a caller acts on them differently:
missing evidence calls for more views or a larger patch, and evidence that
does not settle calls for a look at the point (a fold, another layer). The
read-back renders the bitmap from the track's reference observation as every
normal step's does; the estimate itself reads no reference view.

**It refuses what every normal step refuses**, through `normal_preconditions`: a
track at the cluster stage, one with no patch, a track at infinity and one with
fewer than two `in` observations. A point at infinity is a direction; its patch
faces along its own bearing, there is no plane through it to render on, and
the step refuses it as `NormalError::AtInfinity`, as the other normal steps do.
`refine_normals` already skips such points before building their tracks and
counts them, so a later option there that runs this step would inherit that.
Every other quantity is computed from the finite point `X`, so no division by
`w` occurs. A camera whose centre is at `X` has no line of sight; its row is
`NotAttempted`.

**It reads no keypoint.** The crops are sampled around the 3D point, not around
the keypoints, and each pair's free shift absorbs the difference between the
point's projection and where the photographs see it. A track whose keypoints are
off still gets a normal; one whose point is off by more than the crop's half
width does not.

**The settings are `*Options`** (STANDARDS § "A settings struct is named
*Options"). The flow's settings are a `DisFlowParams` field, which keeps its
name as backlog from before that ruling.

**It takes a `Progress`** (STANDARDS § "Long-running functions take a
Progress"): a point with 20 views has 190 pairs, each a dense flow, in every
round. It would report a phase per round, and within it one for the crops, one
for the flows with a count per pair and one for the fits, and check for
cancellation between pairs and between radii.

**The report is per round** because the rounds are where the step moves: a
person checking a point needs the plane each round rendered on, what it found,
and, when a round had no answer, the fallback plane passed on.

**`SinglePair` is an outcome of its own** because a caller treats it
differently: the normal is a measurement, but nothing checked that the support
was not too large. It is turned only when its σ is at most `max_sigma_deg`,
and is `Undetermined` otherwise, so a two-view point on weak texture is not
turned to a normal its one pair cannot fix. Whether a precise single pair
should be turned at all is an open question
(§ [Open questions](#open-questions)).

**The name** follows the glossary's rule for the bench normal steps
(**finite-difference normal** / **grid-plane normal** / **grown-plane normal**),
which names each step for the evidence it reads. This step reads dense optical
flow between every pair of the point's observations, so it is the
*pairwise-flow normal*: `pairwise_flow_normal`, Track View's *Pairwise Flow
Normal* and the wire's `fit_bench_track_normal` `method` `pairwise_flow` when
those are built. *Flow normal* alone would not say that the flow is read between
views rather than along a sequence, which is how
[optical-flow.md](../core/features/optical-flow.md) uses it everywhere else.
The report types are named, like `GridPlaneCells` and `GrownPlaneSequence`, for
what they hold: `PairwiseFlowRounds` the rounds, each `PairwiseFlowRound` one
reading per radius. The glossary entry is added when the step is built.

### Example

```rust
use sfmtool_core::bench::{pairwise_flow_normal, PairwiseFlowNormalOptions};
use sfmtool_core::progress::Progress;

let (turned, report) = pairwise_flow_normal(
    &track,
    &edited,
    &images,
    &PairwiseFlowNormalOptions::default(),
    &Progress::none(),
)?;
println!("{report}"); // "3 rounds; 21 of 21 pairs agree at 1.5 half-extents (χ² 1.4), σ 1.8°, turned 12.3°, …"
```

## Theory

### Crops on a common plane

In each round every view is rendered onto one plane through the point `X`, the
round's **plane**: the first round's faces the normal the step is given, and
each later round's faces the normal the round before found
(§ [Rounds](#rounds)). The crops share one grid and one pair of in-plane axes:
the patch's `u` projected onto the plane and normalised, and the plane's normal
crossed with it (where `u` is within a few degrees of the plane's normal, `v` is
projected instead). A view whose line of sight at `X` is more than
`max_view_angle_deg` (80°) from the plane's normal is left out of the round
(`TooOblique`): so near edge on, the plane is compressed in its photograph by
more than 5 times along one axis.

Each crop is rendered as an `OrientedPatch` with that frame, centred on `X`,
through `WarpMap::from_patch`
([camera/warp_map.rs](../../crates/sfmtool-core/src/camera/warp_map.rs)), which
maps each sample through the camera model onto an arbitrary oriented plane.
Lens distortion, including a fisheye's edges, is therefore removed, and every
crop shows the photograph as it would look on that plane. Each view is rendered
once per round, on the grid all the point's views share, so the flow is in one
unit across pairs; there are no per-pair renders.

**Why one common plane and not a plane per view.** An earlier form of the
design rendered each view on its own fronto-parallel plane, the plane through
`X` facing that camera. Two such crops of a wide pair then differ by a large
projective warp, a rotation and a shear, which DIS recovers poorly. On a
kerry_park point whose pairs are 89° to 113° apart the flow between fronto crops
was chaotic and its leftover over 1 px almost everywhere; on a dino point whose
pairs are 30° to 50° apart the crops were rotated and sheared by several pixels
and the flow was reliable only near the centre. On a common plane near the
answer the crops of every pair line up, and the flow measures only the small
parallax the plane's remaining error leaves. On six points traced both ways,
per-view fronto planes gave 4 `Unstable` and 2 `Turned`; the common plane turned
all 6, from 13.6° to 59.9° off at the mean viewing direction to 1.1° to 4.3°
after 3 rounds (measurements, § "Real points with the revised design").

### The grid

**The grid follows the finest view.** The sample spacing is one photograph pixel
of the finest view, the view with the most pixels across the patch, but the
patch's side is never fewer than `min_samples_per_patch` (24) samples nor more
than `max_samples_per_patch` (128). At one sample per pixel of the finest view
no view's detail is lost to the grid until the upper bound: in the prototype, a
fixed 24 samples per side lost detail on the dino ground truth's patches, about
43 px across, and its error at a fixed radius of 1 half-extent rose from 1.7°
with the earlier pixel-based crops to 2.9°; at native resolution it was 1.3°.
The lower bound keeps a tiny patch on enough samples for the flow and the fits.
The upper bound bounds the cost: a patch that covers 1024 photograph pixels
across in its finest view gets 128 samples per side, each spanning 8 pixels, and
a crop of 3 times the patch is 384 samples a side rather than 3072.

**Views finer than the grid are prefiltered.** Where the grid is held at its
upper bound a view can have pixels finer than the grid; it is read through the photograph's
pyramid down to the grid, so every such view is filtered alike and its texture
finer than the grid cannot alias into false structure that the flow matches. A
view coarser than the grid is upsampled, which only blurs; blur that both crops
of a pair share is harmless to the flow, and a pair whose crops differ in blur
is read at a coarser level (§ [The pyramid level of a pair](#the-pyramid-level-of-a-pair)).
Its σ counts the correlation of the flow's errors in that view's own pixels
(§ [Fitting each pair at each radius](#fitting-each-pair-at-each-radius)).

The renderers already sample this way. With `SamplerChoice::PerView`
([sampler.rs](../../crates/sfmtool-core/src/camera/sampler.rs)) a view is read
with `Sampler::BilinearMip` at the level `bilinear_mip_level` gives,
`round(log2 max(σ_major, 1))` for `σ_major` the larger singular value of the
grid's Jacobian into the photograph in photograph px per sample, or with the
anisotropic sampler where the compression is much stronger along one axis
(`rule_sampler`), as it is for a view oblique to the plane. The levels are
those of the `ImageU8Pyramid` each `ProjectedImage` carries, built by 2×
box-filter downsampling
([camera/image.rs](../../crates/sfmtool-core/src/camera/image.rs)). A view with
about one pixel per sample (`σ_major` under √2) reads level 0, a view with finer
pixels reads a coarser level, about one of its pixels per sample there, and a
view coarser than the grid reads level 0 bilinearly, which upsamples it.
Because the level is rounded, two views on either side of a level boundary are
filtered up to √2 apart in scale; that is "alike" to within one rounding. The
crops are then converted to the flow's grey (`matcher_gray_from_rgb` in
[optical_flow/image.rs](../../crates/sfmtool-core/src/features/optical_flow/image.rs)).

### The extent

Every crop of a round is rendered over a square of half-width
`max(crop_half_extents · h, min_crop_half_width_px · p_f)` in the world, where
`h` is the patch's half-extent and `p_f` the size, at `X`, of one photograph
pixel of the point's finest view on the plane: 3 times the patch by default,
but never fewer than 48 of the finest view's pixels, and every pair reads that
whole square at its own level. The floor is there for tiny patches: a fresh
solve of seoul_bull has a median patch half-size of 3.4 px, so 3 times the
patch would be about 10 px, while the first prototype's best fixed support on
those points was 16 px, about 4.7 times the patch, and the support chooser
needs room above that to find where the surface stops being one plane. The
finest view's pixels still cover a tiny patch, so the floor counts them.

**Why the finest view and not the pair's coarser view.** An earlier form set
the floor per pair, at 48 pixels of the pair's coarser view. A point with one
far view then got pair squares, and so allowed radii, that were large in the
world: on seoul_bull points 70, 173 and 255 the rule picked 4 to 12
half-extents, about 76 to 152 pixels of the finest view, where the first
prototype, with its crops set in pixels, never went past 45 px
(measurements, § "Real points with the revised design"). With the floor in the
finest view's pixels, the radii a pair can reach are the same for every pair,
and a pair with a far view simply reads fewer of that view's own pixels: at its
level its square is `2^ℓ` times narrower in samples, and its σ larger.

**The cap and the cost.** `max_crop_samples` (512) bounds the one render per
view. A crop's half-width in grid samples is `max(3h, 48 · k_f)`, `h` here in
grid samples and `k_f` the `samples_per_pixel` of the finest view: 1 where the
grid is at the finest view's pixel, over 1 where the grid is held at its lower
bound on a tiny patch, under 1 where it is held at its upper bound on a large
one. The grid is coarsened for the whole round, with the report carrying the
`samples_per_patch` used, only when the render would have more than
`max_crop_samples` along a side. `3h` is at most 192 samples, under the upper
bound on the grid, so with the defaults that happens only when the floor
exceeds the cap, `2 · 48 · k_f > 512`: a finest view whose pixel spans more
than about 5.3 grid samples, that is a patch under about 4.5 pixels across in
its finest view. The coarsening may take the grid under
`min_samples_per_patch`; the extent is kept and the resolution given up,
because the floor is what leaves the support chooser room. A 3.4 px half-size
patch is 6.8 px across in a view at that scale, so if that is its finest view
its grid is the lower bound, 24 samples, 3.5 samples per pixel; 48 px of that
view is a half-width of 170 samples, a render of 340 a side, under the cap.

What bounds each pair's flow is the render and its level. A pair read at level
`ℓ` reads the render's square at `2^ℓ` times fewer samples a side, so its
square is at most `2 · max(3h, 48 · k_f) / 2^ℓ` samples a side: at level 0 at
most 384 where the grid is at the finest view's pixel and up to the render's
512 where it is held at its lower bound. A pair whose square at its level is
narrower than the flow's `patch_size` has no flow and no fits; it is reported,
not refused.

### The pyramid level of a pair

The scale only matters within a pair when one of its views is coarser than the
grid. Then that view has been upsampled by `k`, its `samples_per_pixel` (over
1), and its crop is blurred over `k` samples while the other crop is sharp; DIS
would match a sharp crop against a blurred one. So each rendered crop gets a
**crop pyramid**, 2× downsamplings with one Gaussian kernel for every view: the
flow's own `ImagePyramid` in
[optical_flow/pyramid.rs](../../crates/sfmtool-core/src/features/optical_flow/pyramid.rs),
whose levels are a blur by the 6-tap σ = 1 kernel `GAUSS6_1D` and a 2×
downsample (the module is private to the flow today, reachable outside it only
through the hidden `bench` re-export, so the step needs it made visible within
the crate). A pair is read at level `ℓ = floor(log2 k)` of both crops'
pyramids, `k` being the coarser view's; at that level the coarser view is
upsampled by `m = k / 2^ℓ`, between 1 and 2, and the two crops carry about the
same blur. A pair whose views are both at or finer than the grid is read at
level 0. The flow at level `ℓ` is in level samples, `2^ℓ` grid samples each.
The level is chosen per pair from the renders already made, so no view is
rendered again for a pair, and reading a pair of unequal views at the coarser
one's level also cuts the flow's cost by `4^ℓ`. With the grid at the finest
view's pixel, every other view is at or coarser than the grid, so a pair's
level is set by the scale ratio of its coarser view to the finest view.

### Radii in patch half-extents

The radii are multiples of `h`, so every pair is compared at the same radius in
the world. That is what the χ² rule needs: it detects the radius at which the
surface stops being one plane, which is a distance on the surface, not a count
of any one view's pixels. A radius that covers few pixels of a pair's coarser
view gives that pair a large σ and so little weight, with no special case; a
radius past the square a pair reads gives that pair no fit there.

### Flow

For every unordered pair `(i, j)` of rendered crops the step computes the DIS
flow from crop `i` to crop `j` with `compute_optical_flow`
([optical_flow/mod.rs](../../crates/sfmtool-core/src/features/optical_flow/mod.rs)),
on the central square the pair reads at its pyramid level, at that level's
full resolution: `finest_scale` `Some(0)`, `patch_stride` 1, and otherwise
`DisFlowParams::high_quality()` (patch 12, 16 gradient-descent iterations,
variational refinement on).

`high_quality` itself is not enough. It fixes `finest_scale` at 1, so it never
solves at full resolution; on the prototype's crops of 96 samples the automatic
coarsest scale is `floor(log2(2 · 96 / (5 · 12)))`, which is 1, so the whole
flow is solved at half resolution and upsampled bilinearly
([optical-flow.md](../core/features/optical-flow.md#presets)). On synthetic
pairs, a real crop warped by the flow of a known plane, the prototype measured a
median flow error of 0.093 px with `high_quality` and 0.040 px at full
resolution with stride 1 and variational refinement, and a median normal error
of 2.0° against 0.9° (§ "Flow settings" in the measurements).

### The pair model

For a candidate normal `n` of the plane through `X`, the map from a sample of
crop `i` to crop `j` is exact ray-plane geometry:

1. the sample's position on the round's plane gives a ray from `C_i`;
2. the ray meets the candidate plane, through `X` with normal `n`, at `Y`;
3. the ray from `C_j` to `Y` meets the round's plane at a point whose in-plane
   coordinates, in grid samples, minus the sample's own coordinates, are the
   predicted flow.

Each pair also has a free 2D offset, in grid samples, added to the predicted
flow: the point's projection and the true position of the surface in each
photograph differ by the keypoint and triangulation inconsistency, and the
offset takes that up so it does not bias the tilt. The pair's unknowns are the
normal's two tilt angles and the offset. A sample whose ray at step 2 is
parallel to the candidate plane, or meets it behind a camera, is left out of the
fit. When the candidate plane is the round's plane, every predicted flow is
zero: the flow measures only the parallax the plane's error leaves.

In the prototype, fitting this model to the exact predicted flow of a known
plane recovered the normal to 0.00°, so the geometry is right.

### Fitting each pair at each radius

At each radius `r` in `radii`, the step fits the normal to pair `(i, j)`'s
measured flow at the samples within `r · h` of the crop's centre that lie on
both photographs, at the pair's pyramid level:

1. **A coarse grid.** Every normal on a grid of tilts from 0° to
   `max_grid_tilt_deg` in `grid_tilt_step_deg` steps, at azimuths
   `grid_azimuth_step_deg` apart, about the pair's mean viewing direction, with
   the offset at each solved in closed form; the grid's best is the start. The
   fit therefore searches the tilt afresh in every round, whatever plane the
   crops are on. Centring it on the pair's mean viewing direction is this
   draft's choice, not a measured one.
2. **Gauss-Newton with texture and robust weights** from that start, over the
   two tilt angles and the offset. Each sample's flow residual `r_s` (a
   2-vector, measured minus predicted) enters as `r_sᵀ W_s r_s`, where `W_s` is
   the sample's flow information from the image structure (below). Each sample
   also carries a Tukey weight on its Mahalanobis distance `√(r_sᵀ W_s r_s)`,
   in units of the whitened residuals' robust scale with `tukey_c`; the
   weights are recomputed after every step, and they multiply the texture
   weights. Without them a strong edge, often the outline of the surface
   against another layer, gets the full weight its contrast earns even where
   its flow belongs to the other layer.
3. **σ from the texture.** The covariance of the tilt is the inverse of the
   Hessian `H = Σ J_sᵀ W_s J_s` (with `J_s` the derivative of the predicted flow
   at sample `s` with respect to the two tilt angles and the offset, and the
   Tukey weights folded in), read in the plane's own angular frame as the
   grown-plane step reads its σ; `sigma_texture_deg` is the square root of its
   larger eigenvalue, in degrees.
4. **Correlated samples.** `H` sums over samples, but DIS flow errors are
   correlated over a distance of the order of the DIS patch (12 samples); the
   prototype took `correlation_length_px`, 8 photograph pixels, and the step
   counts them in the coarser view's pixels as its crop reads them. With `m`
   the factor by which that view is upsampled at the level the pair is read at
   (`m = k / 2^ℓ`, between 1 and 2), or `m = 1` where neither view is coarser
   than the grid, that is a square of side `L = 8m` samples at that level. Each
   sample's `W_s` (below) is the information of the `L × L` square about it, so
   summing it over every sample counts each square's information `L²` times;
   `H` is therefore divided by `L²`, and each correlated square counts once.
5. **σ from the residuals.** `sigma_residual_deg` is read the same way from
   the covariance `s² · L² · (Σ J_sᵀ J_s)⁻¹`, the sums over the samples that
   carry weight and `s²` the weighted residual variance of the fit per flow
   component there. This is the σ the prototype first used, before the texture
   term.
6. **σ.** `sigma_deg` is the larger of the two. The texture σ alone was far too
   small on real photographs: on good points the combined normal was close to
   the reference (on one seoul_bull point, 1.6° to 8° at every radius) while
   the mean χ² was 44 to 2,500, and 600 to 50,000 on the points whose grid had
   90 or more samples per patch side. Real errors (in the calibration, a slight
   curvature, keypoint offsets beyond a shift) are correlated across the whole
   patch, not independent per correlated square, and the texture σ cannot see
   them. The texture σ is what makes a flat crop `Undetermined`; the residual σ
   is what keeps χ² honest where the texture is strong.

A pair whose `H` is singular has no fit at that radius (`None`), and a fit whose
`sigma_deg` is over `max_sigma_deg` is not **determined**: it is reported but
does not enter the combination or χ².

**The flow information of a sample.** The geometric derivative `J_s` alone
says nothing about texture: on a textureless crop the smoothed DIS flow is
near zero with small residuals, and a Hessian of geometry alone times that
small residual variance would give each pair a confident normal, whichever
normal predicts zero flow slope for the pair's geometry. The information must
depend on how well the image pins the flow at each sample. The step takes it
from the Lucas-Kanade measurement model: a flow read over a window is measured
precisely only along the image gradients in it, with covariance
`σ_I² · T⁻¹`, where `T` is the window's **structure tensor**, the sum of
`∇I ∇Iᵀ` over the window, and `σ_I²` is the variance of the photometric
residual. So:

- `T_i(s)` is the structure tensor of crop `i`'s grey values over the
  `L × L` square centred on `s`, the correlated square of step 4, so the window
  the information is read over and the area it is counted once per are the
  same; `T_j(s)` is crop `j`'s at `s` plus the measured flow. There is no
  separate window option.
- Each tensor keeps only what is significant against noise. Removing the
  expected noise contribution from the diagonal is not enough: it removes the
  noise only on average, and a sum of `n = L²` noise gradients fluctuates
  about its mean. For gradients `g` of independent Gaussian noise with
  per-component variance `v` (proportional to the crop's noise variance
  `σ_I²/2`, with a factor set by the gradient operator), each diagonal entry
  of the tensor has mean `n·v` and standard deviation `v·√(2n)`; for `L = 8`
  that is `64v ± 11v`. Neighbouring gradients share pixels and an upsampled
  crop is smooth over `k` samples, so the true spread is larger than that
  formula. The step therefore takes the threshold from a simulation: for each
  window side `L` and each way a crop is rendered and read (the crop's own
  `samples_per_pixel`, which can differ from the pair's `k`, and the pyramid
  level the pair reads it at), it renders pure white noise the way that crop
  is rendered, takes it to that level, computes the tensor's larger eigenvalue
  over many windows, and takes its `1 − noise_false_alarm` quantile (99.9%),
  in units of `v`. That threshold `τ·v` is subtracted from both eigenvalues of
  each tensor, and eigenvalues below zero are set to zero. A window on a flat
  crop then has zero information except in about one window in a thousand, and
  there only the small excess over the threshold. Since the noise adds about
  `n·v` to each eigenvalue on average and `τ·v` is larger than that, a
  textured window also loses the margin `(τ − n)·v`: its information is
  understated, so its σ errs large. The loss is small beside strong texture and
  matters only for texture near the noise level.
- The noise variance is read from `σ_I²`, which includes the flow's
  registration error as well as the image noise. That makes `v`, and with it
  the threshold, larger than the image noise alone would, which removes more
  information than necessary, never less: the error is on the safe side.
- The two are combined as the **parallel sum** `T_i (T_i + T_j)⁺ T_j`, the
  inverse of the sum of the two covariances. The flow is the difference of two
  positions, one read in each crop, so its error is the sum of the two
  measurement errors and its information the parallel sum: never more than the
  weaker crop's tensor along any direction, and zero where either crop is flat.
  The plain sum `T_i + T_j` is rejected because it would credit a pair with
  texture only one of its crops has, as where one crop is blurred, occluded or
  off the surface.
- `W_s = T_s / (σ_I²/2)`, `T_s` being that parallel sum: each crop's
  position is read with covariance `(σ_I²/2) · T⁻¹`, its own noise variance
  times its tensor's inverse, so where the two tensors are equal `W_s` is
  `T / σ_I²`, the Lucas-Kanade information of the flow. `σ_I²` is the pair's
  photometric residual variance over the disc (crop `j` warped by the measured
  flow against crop `i`), floored at the 8-bit quantization variance so a
  perfect match does not claim infinite precision.

**A floor on the flow's precision.** Strong texture makes `W_s` large, but the
flow DIS returns has errors the structure tensor does not model, which do not
shrink as the contrast rises. `flow_floor_px` is set, as `correlation_length_px`
is, in pixels of the pair's coarser view as its crop reads them, so at the
pair's level it is `f = flow_floor_px · m` samples. Each correlated square's
flow measurement has the covariance `W_s⁻¹ + f² · I`, and `W_s` is replaced by
`W_s (I + f² W_s)⁻¹` before `H` is divided by `L²`. Written per sample, with
`w = W_s / L²` the information one sample carries, that is
`w (I + f² L² w)⁻¹`. On weak texture the floor changes nothing; on strong
texture it caps each square's information at `1/f²`. In a prototype of the
texture σ and the outcome rules on synthetic scenes, σ without the floor was
too small where the texture was strong, the error 2 to 4 times σ at the highest
contrasts, and one clean plane in 24 came out `Unstable` because χ² was
inflated; with a floor of 0.02 px the error was 1 to 2 times σ, with 0.05 px
under σ, and no clean plane was `Unstable` with either (measurements,
§ "Synthetic scenes"). The default, 0.03, is between the two. Those synthetic
crops were at one sample per pixel, where samples and pixels are the same.

Because `W_s` is absolute, not scaled by the flow residual's own variance, and
counts only structure significant against noise, a flat crop gives an `H` that
is zero or nearly so: of the 380 samples in a disc of radius 11 samples, about
one in a thousand has a window over the threshold, each with information only
just above zero, so the texture σ is far above `max_sigma_deg` or infinite and
no pair is determined, whatever the residual σ. Weak texture gives a σ that
grows as the texture weakens.

### Choosing the support

At each radius the step combines the normals of the determined pairs (σ at
most `max_sigma_deg`):

1. **The medoid** of the pair normals: the pair normal with the least sum of
   angles to the others.
2. **The combined normal**: the σ-weighted mean (weights `1/σ²`) of the pairs
   whose normal is within `agreement_sigmas · σ + agreement_margin_deg` (3σ +
   5°) of the medoid, `σ` being that pair's own. Its σ is the inverse square
   root of the summed weights, in degrees.
3. **χ²**: the mean over the determined pairs of (angle from the combined
   normal / that pair's σ)², taken only where at least two pairs are
   determined; with fewer it is `NaN`.

A radius has **enough evidence** when at least two pairs are determined there,
so that their agreement can be tested; its combined normal's σ is then at most
`max_sigma_deg`, since only determined pairs enter it. The chosen radius is the
largest radius with enough evidence whose mean χ² is at most `max_mean_chi2`
(2): the largest support on which the pairs agree within their own precision.
The round's answer is the combined normal at that radius.

A radius with only one determined pair, on a point with several pairs, is not
used at all. Its χ² over one pair would be 0, so counting it would let the
largest such radius win with no test of agreement, which is the rejected
"radius of smallest σ" rule (one pair's σ keeps falling as the disc grows)
without even the single-pair bar. It is not read as `SinglePair` either:
when a point has several pairs and only one of them fixes the tilt, the other
pairs' failure is itself evidence that the views see the surface poorly, and
the one fit has nothing to check it against. `SinglePair` stays the allowance
for a point that has only one pair to offer.

The reason the agreement between pairs, and not any one pair's readings, picks
the support: a support that is too large, one that runs over a fold of the
surface, a curve or another layer, affects each viewpoint differently, so the
pairs' normals spread apart once the disc reaches it. A single pair's σ keeps
falling as the radius grows, even across a fold, because more samples fit a
plane at some normal; a rule on one pair's σ or its steps cannot see the fold.
The rules the prototype tried and rejected are in the measurements
(§ "Choosing the support" and § "Rejected alternatives").

### Rounds

The step runs `rounds` (3) rounds. Round 1 renders on the plane facing the
normal the step is given; round `k + 1` renders on round `k`'s answer, so each
round's crops line up more closely than the last and its flow measures a
smaller parallax. When a round has no answer by the rule, the next round's
plane is the **fallback**: the combined normal at `fallback_radius` (1
half-extent) over the pairs with a finite σ there, determined or not, which is
the prototype's choice. Every round is reported.

**A plane the views cannot see is no answer.** A normal more than
`max_normal_from_mean_view_deg` (80°) from the mean viewing direction of the
point's views faces a plane the views see nearly edge on or from behind. A
round whose answer is that far does not pass it on: the step stops after that
round, and its outcome is `Unstable`, since the evidence settled on a plane
that cannot be the surface the photographs show. A fallback plane that far is
not passed on either. Where a round has no plane to pass on, because it has no
answer and its fallback is missing or too far, the step stops, as rendering on
the same plane again would repeat the same round. The bar does not catch every
drift: on seoul_bull point 207, a bush with the bull in front and trees behind,
the step ends 112° from the reference while within 80° of the mean viewing
direction.

**A point with one pair runs one round.** With one pair there is no agreement
to test, so nothing stops a round from moving the plane further along that
pair's error, and the next round renders on that error. In the prototype a
two-view point of seoul_bull, 160, drifted 14° → 40° → 55° → 68° from the
reference over the rounds. Its one round renders on the given normal, as
round 1 always does.

The step's outcome, chosen radius, normal and σ are those of the last round it
ran. A last round with no answer leaves the given normal unchanged, even where
an earlier round had one; which round's answer to keep, and whether to stop
early when a round's answer moves less than its σ, are open questions. In the
prototype, with the crop floor in the finest view's pixels and the bar on the
mean viewing direction, 3 rounds from the mean viewing direction
brought the ground-truth dino points from 28.8° to 1.2° median and the
kerry_park points from 48.9° to 3.7°, the kerry_park points still improving at
round 3; the seoul_bull points stopped improving after round 1, at about 10° to
18° (measurements, § "Real points with the revised design"). 3 is the most
rounds measured.

### Undetermined and unstable

When no radius has enough evidence, the outcome is `Undetermined`: at no
support do two pairs fix the tilt to within `max_sigma_deg`, which is a lack of
evidence. When some radius has enough evidence but the pairs disagree (mean χ²
over `max_mean_chi2`) at every such radius, the outcome is `Unstable`: the
evidence is there and does not settle on one plane. An answer more than
`max_normal_from_mean_view_deg` from the mean viewing direction is also
`Unstable` (§ [Rounds](#rounds)). Both outcomes leave the given normal
unchanged, and the report's outcome says which.

A point with one pair has no agreement to test; the first prototype read it at
a fixed 8 photograph pixels, and this draft reads it in one round at a fixed
`single_pair_radius` (1 half-extent, not measured) as `SinglePair`. It is turned
only when the pair's σ at that radius is at most `max_sigma_deg` and its normal
is within `max_normal_from_mean_view_deg` of the mean viewing direction;
otherwise the outcome is `Undetermined`, or `Unstable` for a normal too far
from the mean viewing direction.

### Output

`PairwiseFlowRounds` carries the outcome, the normal and its σ and the chosen
radius, and every round: its plane, its crops (with the views left out as too
oblique), the mean χ² and the combined normal at every radius, every pair's
angle, level, normals, both σs and offsets at every radius, its answer and the
plane it passed on. That is in the style of the grown-plane step's report:
enough to tell, for a point a person disagrees with, which round went wrong and
which pairs disagreed at which radius.

## Implementation notes

- **Order of work.** In each round, crops are rendered once per observation,
  and each crop's pyramid is built once, to the deepest level any of its pairs
  reads; flows are computed once per pair, at its level, in parallel over pairs
  with rayon; the fits are per pair and radius, independent of one another, and
  also parallel. The combined normal and χ² are taken in pair order, so the
  result does not depend on the thread count.
- **Precision.** The crops and the flow are `f32`, as `GrayImage` and
  `FlowField` are: flow errors are 0.04 px or more, far above `f32`'s rounding
  at crop coordinates under `max_crop_samples` (512). The fits accumulate a few
  thousand samples into 4×4 normal equations and run in `f64`, which costs
  nothing at that size and keeps the Hessian's inverse well conditioned near
  edge-on tilts.
- **Cost.** The first Python prototype took about 59 s for 153 points with 12
  processes, about 4.6 CPU-seconds per point, most of it in the fits; the flows
  were one full-resolution flow on a 96×96 crop per pair. The prototype of the
  revised design took about 4 to 8 minutes for 189 points with 12 processes per
  configuration, more with the native grid and 3 rounds. A Rust version is
  expected to be far cheaper, chiefly in the fits. The crops bound each flow: a
  render is at most `max_crop_samples` a side, and each pair reads it at its
  own level, `2^ℓ` times fewer samples a side (§ [The extent](#the-extent)).
  The prototype read at most 10 pairs per point, chosen at random. The step stays expensive by design all the
  same: the number of pairs grows with the square of the number of views, and
  every round repeats the flows. How to bound it is an open question.

## Parameters

| Option | Default | Source |
|--------|---------|--------|
| `rounds` | 3 | measured on real points up to 3 rounds (measurements, § "Real points with the revised design"); not beyond |
| `max_view_angle_deg` | 80° | the prototype's; not measured against other values |
| `max_normal_from_mean_view_deg` | 80° | the prototype's, in the runs after the two changes of § "Real points with the revised design"; not measured against other values |
| `fallback_radius` | 1 half-extent | the prototype's choice; not measured against other values |
| `min_samples_per_patch`, `max_samples_per_patch` | 24, 128 | the prototype's native grid with these bounds; the bounds themselves not measured against other values |
| `crop_half_extents` | 3 | used in the real-point runs of the revised design; not measured against other values |
| `min_crop_half_width_px` | 48 px of the finest view | used in the real-point runs after the two changes (before them, 48 px of the pair's coarser view); not measured against other values |
| `max_crop_samples` | 512 | used in the real-point runs; bounds the render per view only, so the grid is coarsened only for a patch under about 4.5 px across in its finest view; not measured against other values |
| `radii` | 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12 half-extents | used in the real-point runs; not measured against other lists |
| `flow` | `high_quality`, `finest_scale` 0, `patch_stride` 1 | measured, § "Flow settings" |
| `max_grid_tilt_deg`, `grid_tilt_step_deg`, `grid_azimuth_step_deg` | 85°, 10°, 30° | prototype |
| `tukey_c` | 4.685 | the usual 95%-efficiency constant; not measured here |
| `correlation_length_px` | 8 px | used in the real-point runs; a guess, open question |
| `noise_false_alarm` | 0.001 | used in the real-point and synthetic runs; not measured against other values |
| `flow_floor_px` | 0.03 px | measured only on synthetic scenes at one sample per pixel (measurements, § "Synthetic scenes"): 0.02 left the error 1–2 times σ, 0.05 under 1 times σ; used in the real-point runs |
| `max_sigma_deg` | 10° | measured on real points with the revised design: the best of the settings tried |
| `agreement_sigmas`, `agreement_margin_deg` | 3, 5° | prototype |
| `max_mean_chi2` | 2 | measured on real points with the revised design: 1 and 2 were the best of the settings tried, 2 turning more points (measurements, § "Real points with the revised design") |
| `single_pair_radius` | 1 half-extent | proposed; the first prototype's best fixed radius was 8 photograph px, which has no fixed value in half-extents; open question |

## Testing

Proposed tests for the step when it is built:

- **Exact geometry.** Crops rendered from a synthetic textured plane at a known
  normal, seen by three or more cameras at different angles, from a given
  normal 20° off: the step returns `Turned` within a fraction of a degree of
  the true normal.
- **Rounds converge on a clean plane.** The same scene with wide baselines,
  given a normal 40° off: each round's plane is closer to the true normal than
  the last, and the last round's answer is within 3 times the largest σ among
  its combined pairs. The same scene given a normal 40° off in another
  direction ends within the same bound of the true normal.
- **The common plane lines up the crops.** A textured plane seen by two
  cameras 90° apart, given its true normal: both crops show the same texture at
  the same place, so the measured flow is under 0.1 sample over the disc, and
  the pair's normal is within its σ of the true one.
- **A view too oblique is left out.** A camera whose line of sight is 85° from
  the given normal: its crop is `TooOblique` in round 1, and no pair includes
  it.
- **The grid follows the finest view.** Patches covering about 10, 43 and 1000
  photograph pixels across in their finest view get 24, 43 and 128 samples per
  side. At 128 the crop is 384 samples a side, and the 1000 px view is read
  through its photograph's pyramid at the level `bilinear_mip_level` gives,
  `round(log2 7.8)` = 3; the step returns `Turned` near the true normal.
- **A fold.** Two textured planes meeting at a crease near the point: the chosen
  radius stops short of the crease, and the normal is the near plane's.
- **Undetermined.** A textureless plane, its photographs a constant grey with
  white noise added: after the significance threshold, about one window in a
  thousand keeps any information, and only a little, so every pair's `H` is
  zero or nearly so and its texture σ is far over `max_sigma_deg`; no radius has
  two determined pairs in any round, the outcome is `Undetermined`, and the
  track's normal is exactly the one given. The same with two views, one pair,
  is `Undetermined` and not `SinglePair`.
- **Weak texture.** A plane whose texture contrast is lowered step by step: the
  pairs' σ grows as the contrast falls, and below some contrast the outcome
  changes from `Turned` to `Undetermined`. It is never `Unstable`, since the
  scene is one plane: an `Unstable` reading would mean the pairs' σ
  understates their error. At every contrast where it is `Turned`, the normal
  is within 3 times the largest σ among the pairs combined at the chosen
  radius. The combined σ is not the bound, because the pairs reuse views and
  so understate it (§ [Open questions](#open-questions)).
- **The residual σ.** A textured plane with a slight curvature over the disc,
  strong texture: the residual σ exceeds the texture σ at the larger radii, and
  the mean χ² there is lower than it is with the texture σ alone.
- **One determined pair among several.** The support rule given pair fits at
  every radius in which one pair is determined and the others are not: no
  radius has enough evidence, and the outcome is `Undetermined`, not `Turned`
  at the largest radius.
- **A crease through the point.** Two textured planes meeting at a 90°
  dihedral along a line through `X`, with the cameras in front of the edge and
  spread to both sides, so each sees both faces. Every disc straddles the
  crease, and each pair's fit is a compromise between the two faces that
  depends on how that pair's viewpoints foreshorten each face. The pairs still
  agree on small discs, where the two faces barely differ in the flow, and
  disagree more as the disc grows. With wide baselines the outcome is `Turned`
  at a small radius, near the bisector of the two faces, and the mean χ²
  rises steeply at the radii beyond it; the test asserts both. With narrow
  baselines every pair sees about the same compromise, and the patch may be
  turned at a large radius, also near the bisector. The outcome is `Unstable`
  only where no radius passes the χ² rule. A prototype of these rules gave
  exactly this on the scene (measurements, § "Synthetic scenes"); the radius
  at which it stops depends on the χ² calibration and the flow floor.
- **Two views 4× apart in scale.** A textured plane seen by a near camera and
  by one four times as far with the same focal length, the patch 128 pixels
  across in the near view, so the grid is 128 samples per side at the near
  view's pixel and every crop has a half-width of `3h` = 192 samples, 384 a
  side, under `max_crop_samples`: the far crop's `samples_per_pixel` is about
  4, the pair is read at level 2 of both crops' pyramids, on a square 96
  samples a side there, and the normal is recovered; a third, near camera pairs
  with the first at level 0, on the whole 384-sample square.
- **One pair runs one round.** A two-view scene: the report holds one round,
  rendered on the given normal, and the outcome is `SinglePair` or
  `Undetermined`.
- **An answer the views cannot see.** The round logic given a round whose
  answer is 85° from the mean viewing direction: the outcome is `Unstable`,
  no later round runs, and the track's normal is exactly the one given.
- **Refusals.** A track at infinity is refused as `NormalError::AtInfinity`; an
  empty or unsorted `radii`, `rounds` of 0, or a `min_samples_per_patch` under
  2 or over `max_samples_per_patch` is refused by `check` naming the option;
  cancellation through `Progress` returns `NormalError::Cancelled`.

The synthetic scenes are small (a few cameras, crops of at most a few hundred
samples a side), so the tests take well under a second each.

## Non-goals

- **Moving the centre or the keypoints.** Like the grown-plane step, it turns
  the patch about its centre and moves neither the centre nor any keypoint.
- **Replacing the grown-plane step.** That step stays the per-point step of
  `refine_normals` and Surface Tiling Mode; this one is the expensive check and
  refinement beside it.
- **A curved surface model.** The step fits a plane and chooses the support on
  which a plane fits; it does not fit curvature.

## Later steps

Each a separate change after the core step:

1. The Python binding, `sfmtool.bench.pairwise_flow_normal`, taking every
   option as a keyword, as `sfmtool.bench.grown_plane_normal` does.
2. Track View's *Pairwise Flow Normal* entry and the wire's
   `fit_bench_track_normal` `method` `pairwise_flow`.
3. An option of `refine_normals` choosing this step as the per-point step,
   with the per-point outcomes it reports extended by `Undetermined` and
   `Unstable`.
4. The glossary entry for the step's name, and the **turned** /
   **undetermined** / **unstable** entry widened from `grid_plane_normal`'s
   outcomes to this step's as well.

## Open questions

The question whether to render each view on its own fronto plane or all views
on a common plane is settled: a common plane, iterated from the given normal
(§ [Crops on a common plane](#crops-on-a-common-plane)).

- **seoul_bull stops improving after round 1.** On the seoul_bull points the
  rounds stop at about 10° to 18° from the reference after round 1, with pairs
  2° to 5° apart, against 6.9° on the curated seoul_bull points for the first
  prototype. Point by point, the remaining errors are concentrated on layered
  and narrow-baseline points, not on the `ok` points: of the ten `ok` points,
  1, 2, 109, 143 and 157 end 1.2° to 7.1° from the reference, 10 ends 16°,
  18, 120 and 167 are `Unstable`, and 160, a single pair, ends 68° (before the
  one-round rule); the median is set by the judged and hand points (63 at 33°,
  70 at 22°, 173 at 25°, 207 at 112°, 255 at 46°, 238 at 9°, 192 and 260
  `Undetermined`) and by 160. What the step should
  do on a point with another layer in its crops is the open part.
- **kerry_park is still improving at round 3.** Whether to run more rounds, or
  take a stronger step per round.
- **Which round's answer.** The step reports the last round's; whether to keep
  an earlier round's answer when the last has none, and whether to stop early
  when a round's answer moves less than its σ.
- **Start dependence.** The rounds were measured only from the mean viewing
  direction. Starts from the stored normal and from a hand-edited normal are not
  measured.
- **Separating undetermined points.** The outcomes do not yet tell the `A2`
  points (those whose grown-plane answer depends on its start) from good ones:
  8 of 18 kerry_park `A2` points turned, down from 11 to 13 before the floor
  counted the finest view and the bar on the mean viewing direction. They may
  be determined, with wide
  baselines; the `A2` label comes from the grown plane's instability, not from
  the truth.
- **The thresholds.** `max_sigma_deg` of 10° with `max_mean_chi2` of 1 or 2
  are the best settings tried so far on real points; neither was swept finely.
  `max_mean_chi2`, the agreement window and the radius list were also tuned
  against reference normals that the grown-plane step made, apart from two hand
  normals, and need checking against normals that step did not make.
- **Very wide pairs.** Whether to leave out pairs more than some angle apart
  (90° and more): on a common plane they line up, but their crops see the
  surface from very different sides.
- **The crops' defaults.** `crop_half_extents`, `min_crop_half_width_px`,
  `max_crop_samples` and the radius list are the values the real-point runs
  used, and `single_pair_radius` is proposed; none was measured against other
  values.
- **Points with one pair.** There is no agreement to test; the first prototype
  read them at 8 photograph pixels. This draft reads one in one round and turns
  it only when its σ there is at most `max_sigma_deg`, and calls it
  `Undetermined` otherwise, so weak texture is not turned. The one-round rule
  rests on one point, seoul_bull 160, which drifted over the rounds. Open:
  whether a precise single pair should be turned at all, read at a fixed
  radius, or have its radius chosen by another rule.
- **Drift that stays in view.** The bar on the mean viewing direction stops an
  answer the views cannot see, but not every drift: seoul_bull point 207 ends
  112° from the reference within 80° of the mean viewing direction.
- **The noise threshold.** The noise model the tensors are thresholded with
  (white Gaussian noise of the photometric residual's variance) and the
  significance level `noise_false_alarm`. The structure window has no setting
  of its own: it is the correlated square, so it is settled with the
  correlation length.
- **Many views.** A dino point can have 20 or more views, which is 190 pairs.
  Whether to cap the pairs, and how to choose them: widest angles first, a
  spread of baselines, each view in a fixed number of pairs, or the pairs whose
  geometry predicts the most information about the tilt, as the grown-plane
  step's selection does. A cap should prefer pairs whose two views have
  similar scale: a very unequal pair is read at the coarser view's level and
  carries little information, which its σ shows. The prototype read at most 10
  pairs per point, chosen at random.
- **The correlation length.** 8 photograph px is a guess at how far flow errors
  stay correlated. It sets the scale of the texture σ and, through `L²`, of the
  residual σ; it could be measured from the residuals of the synthetic pairs.
- **The combined σ.** The pairs reuse views, so their information is not
  independent and the combined σ is too small; whether to discount it, as the
  grown-plane step halves each kept pair's information.
- **A crease through the point.** On a crease through the point the rule
  shrinks the support to where the pairs still agree and turns the patch to
  about the bisector of the two faces, rather than reporting `Unstable`.
  Whether the bisector is the wanted answer for such a point, or whether it
  should be `Unstable`, or have its normal from one face, is open.
- **Layered points.** A foreground object against a background makes the flow
  see two layers. The chosen radius should stop before the other layer; the χ²
  rule does so only where the layers make the pairs disagree, which a narrow
  baseline may not. The robust weights reduce the weight of an outline whose
  flow belongs to the other layer, but do not remove the other layer from a
  disc.
- **GPU flow.** `compute_optical_flow` takes a `GpuFlowContext`
  ([gpu-optical-flow.md](../core/features/gpu-optical-flow.md)); whether
  batches of small crops gain from it. A pair's square under 224 samples a side
  is below the presets' `gpu_min_pixels` of 50,000; only pairs read at level 0
  on a large patch, up to 384 a side, or on a tiny one whose grid is held at
  its lower bound, up to 512, reach it.
