# Pairwise Flow Normal

**Status:** Draft. A proposal for a new bench normal step; nothing here is
built. The design was prototyped in Python outside the repository and measured
against reference normals; the measurements are in
[pairwise-flow-normal-measurements.md](pairwise-flow-normal-measurements.md).

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

- its answer can depend on the normal it starts from and on the reference view
  it picks (the ground truths' points are labelled `ok`, `A1` and `A2` by
  whether the step returns to the same plane from every start;
  [grown-plane-normal-measurements.md](../core/bench/grown-plane-normal-measurements.md#σ-in-the-planes-own-frame-2026-11-11));
- it fits exactly the patch, so the support the plane is read over is whatever
  size the patch was given, which is 3.4 photograph px in half-size at the
  median on a fresh solve of seoul_bull;
- it turns the patch whenever a view joins, unless the turn is large on weak
  evidence; its only outcome with no plane is that no view joined, and none
  says that the views joined but do not determine the normal.

This draft proposes a deliberately expensive estimator for one point,
`pairwise_flow_normal`. It renders a fronto-parallel crop of the photograph
around the point for every observation, computes dense optical flow between the
crops of every pair of observations at full resolution, and fits the plane
through the point to each pair's flow over discs of several radii. It then
chooses the support itself: the largest radius at which the pairs agree with
each other within their own precision. It needs no start normal and no
reference view for its estimate, and it says when the normal is undetermined:
no radius at which the pairs agree. On that outcome the step leaves the given
normal unchanged, which is what the bench's normal steps do when they have no
estimate (`GridPlaneOutcome::Undetermined`, the glossary's **turned** /
**undetermined** / **unstable** entry, [GLOSSARY.md](../GLOSSARY.md)). The step
must not assume the given normal is the mean viewing direction: in the viewer it
may have been set by hand or by another step, and an undetermined point keeps
it.

It spends far more compute than the grown-plane step: one full-resolution flow
per pair of observations, and many small fits. The uses proposed for it are:

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
/// planes and chooses its support. Defaults are the prototype's (§ Parameters).
pub struct PairwiseFlowNormalOptions {
    /// Half the crop's side, in samples; one sample is about one photograph
    /// pixel at the point. 48.
    pub crop_half_width: u32,
    /// The disc radii the planes are fitted over, in samples, ascending.
    /// [4, 6, 8, 11, 16, 22, 32, 45].
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
    /// Tukey's constant for the Gauss-Newton weights, in units of the
    /// residuals' robust scale.
    pub tukey_c: f64,
    /// How far apart two flow errors are before they are independent, in
    /// photograph px. 8.
    pub correlation_length_px: f64,
    /// A pair joins the combined normal at a radius when its normal is within
    /// agreement_sigmas times its sigma plus agreement_margin_deg of the
    /// pairs' medoid. 3, 5.
    pub agreement_sigmas: f64,
    pub agreement_margin_deg: f64,
    /// The pairs agree at a radius when their mean chi-squared is at most
    /// this. 2.
    pub max_mean_chi2: f64,
    /// The radius a point with one pair is read at, where there is no
    /// agreement to test. 8.
    pub single_pair_radius: f64,
    /// The reading the step ends with, as for the other normal steps.
    pub fit: FitOptions,
}

impl PairwiseFlowNormalOptions {
    /// Refuses a setting the step cannot run with, naming the first that
    /// fails: NormalError::BadPairwiseFlowOption("radii") for an empty,
    /// unsorted or non-positive list or a radius past crop_half_width, and so
    /// on. The flow settings are checked against the crop's size here, so a
    /// patch_size larger than the crop is refused before any render.
    pub fn check(&self) -> Result<(), NormalError>;
}

/// What the step did, carried by NormalEstimate::PairwiseFlow(Box<_>).
pub struct PairwiseFlowRadii {
    pub sample_spacing: f64,               // world units per crop sample
    pub crops: Vec<PairwiseFlowCrop>,      // one per observation of the track
    pub pairs: Vec<PairwiseFlowPair>,      // every pair of rendered crops
    pub radii: Vec<PairwiseFlowRadius>,    // one per options.radii entry
    pub outcome: PairwiseFlowOutcome,
    pub chosen_radius: Option<f64>,        // None: Undetermined
    pub normal: Option<[f64; 3]>,          // the combined normal at the chosen radius
    pub sigma_deg: Option<f64>,            // its sigma, largest axis
    pub turn_deg: Option<f64>,             // from the given normal to it
}

/// Turned: the patch was turned to the normal at the chosen radius.
/// Undetermined: no radius at which the pairs agree, or no pair fitted;
/// the track as given. SinglePair: one pair, read at single_pair_radius
/// with no agreement test, and turned.
pub enum PairwiseFlowOutcome { Turned, Undetermined, SinglePair }

pub enum PairwiseFlowCrop {
    /// samples_per_pixel: how many crop samples one of the view's own pixels
    /// spans at the point; over 1 the view was upsampled, under 1 it was read
    /// from a coarser pyramid level. coverage: the share of the crop's samples
    /// that fell on the photograph.
    Rendered { image: u32, samples_per_pixel: f64, coverage: f64 },
    /// Not in, a second row in one image, or a camera centre at the point.
    NotAttempted,
}

pub struct PairwiseFlowPair {
    pub observations: [usize; 2],
    pub images: [u32; 2],
    pub angle_deg: f64,                    // between the two views, from the point
    pub fits: Vec<Option<PairwiseFlowFit>>, // one per radius; None: too few samples on both crops
}

pub struct PairwiseFlowFit {
    pub normal: [f64; 3],
    pub sigma_deg: f64,                    // largest axis
    pub offset: [f64; 2],                  // the pair's free shift, samples
    pub rms_residual: f64,                 // samples, Tukey-weighted
    pub combined: bool,                    // within the agreement window at this radius
}

pub struct PairwiseFlowRadius {
    pub radius: f64,
    pub normal: Option<[f64; 3]>,          // the combined normal; None: no pair fitted
    pub sigma_deg: f64,
    pub mean_chi2: f64,
    pub pairs_fitted: usize,
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

The report's sentence (`Display`) would read, for example, "21 of 21 pairs
agree at 11 samples (χ² 1.4), σ 1.8°", and for an undetermined point "15
pairs, no radius where they agree (least χ² 3.7 at 6 samples), normal left
unchanged: undetermined", the words `grid_plane_normal` uses for the same
outcome.

### Why it is shaped this way

**The signature is the other normal steps'.** The step ends as they do: it
turns the patch with `tilt_patch`
([bench/steps.rs](../../crates/sfmtool-core/src/bench/steps.rs)) and reads the
track back through `turn_and_read` in
[bench/normal.rs](../../crates/sfmtool-core/src/bench/normal.rs), so a caller can
swap one step for another and compare `NormalReport`s. Under `Undetermined` it
reads back the track as given, as `grid_plane_normal` does. The read-back
renders the bitmap from the track's reference observation as every normal step's
does; the estimate itself reads no reference view.

**It refuses what every normal step refuses**, through `normal_preconditions`: a
track at the cluster stage, one with no patch, a track at infinity and one with
fewer than two `in` observations. A point at infinity is a direction; its patch
faces along its own bearing, the fronto plane through it is not defined, and
the step refuses it as `NormalError::AtInfinity`, as the other normal steps do.
`refine_normals` already skips such points before building their tracks and
counts them, so a later option there that runs this step would inherit that.
Every other quantity is computed from the finite point `X`, so no division by
`w` occurs. A camera whose centre is at `X` gives no fronto plane; its row is
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
Progress"): a point with 20 views has 190 pairs, each a full-resolution flow.
It would report a phase for the crops, one for the flows with a count per pair,
and one for the fits, and check for cancellation between pairs and between
radii.

**`SinglePair` is an outcome of its own** because a caller treats it
differently: the normal is a measurement, but nothing checked that the support
was not too large. Whether such a point should be `Undetermined` instead is an
open question (§ [Open questions](#open-questions)).

**The name** follows the glossary's rule for the bench normal steps
(**finite-difference normal** / **grid-plane normal** / **grown-plane normal**),
which names each step for the evidence it reads. This step reads dense optical
flow between every pair of the point's observations, so it is the
*pairwise-flow normal*: `pairwise_flow_normal`, Track View's *Pairwise Flow
Normal* and the wire's `fit_bench_track_normal` `method` `pairwise_flow` when
those are built. *Flow normal* alone would not say that the flow is read between
views rather than along a sequence, which is how
[optical-flow.md](../core/features/optical-flow.md) uses it everywhere else.
The report type is named, like `GridPlaneCells` and `GrownPlaneSequence`, for
what it holds: one reading per radius. The glossary entry is added when the
step is built.

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
println!("{report}"); // "21 of 21 pairs agree at 11 samples (χ² 1.4), σ 1.8°, turned 12.3°, …"
```

## Theory

### Fronto crops

For each `in` observation `j`, with camera centre `C_j` and the point at `X`,
the step samples the photograph on the plane through `X` facing that camera,
whose normal is `unit(C_j − X)`. The crop's in-plane axes are the patch's `u`
projected onto that plane and normalised, and the normal crossed with it, so
every crop has the same orientation about the line of sight and two crops
differ, before the plane's effect, by little more than a shift. Where `u` is
within a few degrees of the line of sight, `v` is projected instead.

The crop is rendered as an `OrientedPatch` with that frame and a half-extent of
`crop_half_width` samples, through `WarpMap::from_patch`
([camera/warp_map.rs](../../crates/sfmtool-core/src/camera/warp_map.rs)), which
maps each sample through the camera model. Lens distortion, including a
fisheye's edges, and the view's angle to the surface are therefore removed: a
crop is what a distortion-free camera looking straight at `X` from `C_j`'s side
would see.

**The sample spacing** is one photograph pixel at the point: for each
observation, the patch's half-extent divided by that half-extent's length in the
photograph in pixels (the half-side of the square of equal area, read from the
patch's Jacobian, `patch_grid_jacobian`), and the median of that over the
observations (`sfmtool_core::numeric::median`). All crops of one point share
the spacing, so the flow is in one unit across pairs, and radii are in samples,
which are close to photograph pixels.

**The crop's size is a fixed number of photograph pixels**, not a multiple of
the patch. What a disc of flow can say about the tilt is bounded by how many
real image pixels it covers, and the patches a solve stores vary widely in that:
a fresh solve of seoul_bull has a median patch half-size of 3.4 px, and the
curated ground truths 12.7 to 21.6 px. A crop of a multiple of the patch would
be a few pixels across on the first and needlessly large on the others.

**Aliasing.** A view whose own pixels are finer than the grid (it is closer to
the point, or has a longer focal length) must be sampled through a pyramid or
an area filter: point sampling aliases fine texture into false structure that
the flow then matches with confidence. A view coarser than the grid is
upsampled, which only blurs, and blur is harmless to the flow. The sampler rule
([sampler.rs](../../crates/sfmtool-core/src/camera/sampler.rs),
`SamplerChoice::PerView`) already does both: it reads a finer view from the mip
level the warp's compression picks, or with the anisotropic sampler where the
compression is much stronger along one axis, and a coarser one bilinearly from
level 0, through the `ImageU8Pyramid` each `ProjectedImage` carries. The crops are then
converted to the flow's grey (`matcher_gray_from_rgb` in
[optical_flow/image.rs](../../crates/sfmtool-core/src/features/optical_flow/image.rs)).

### Flow

For every unordered pair `(i, j)` of rendered crops the step computes the DIS
flow from crop `i` to crop `j` with `compute_optical_flow`
([optical_flow/mod.rs](../../crates/sfmtool-core/src/features/optical_flow/mod.rs)),
at full resolution: `finest_scale` `Some(0)`, `patch_stride` 1, and otherwise
`DisFlowParams::high_quality()` (patch 12, 16 gradient-descent iterations,
variational refinement on).

`high_quality` itself is not enough. It fixes `finest_scale` at 1, and on a crop
of 96 samples the automatic coarsest scale is
`floor(log2(2 · 96 / (5 · 12)))`, which is 1, so the whole flow is solved at
half resolution and upsampled bilinearly
([optical-flow.md](../core/features/optical-flow.md#presets)). On synthetic
pairs, a real crop warped by the flow of a known plane, the prototype measured a
median flow error of 0.093 px with `high_quality` and 0.040 px at full
resolution with stride 1 and variational refinement, and a median normal error
of 2.0° against 0.9° (§ "Flow settings" in the measurements).

### The pair model

For a candidate normal `n` of the plane through `X`, the map from a sample of
crop `i` to crop `j` is exact ray-plane geometry:

1. the sample's position on crop `i`'s fronto plane gives a ray from `C_i`;
2. the ray meets the candidate plane, through `X` with normal `n`, at `Y`;
3. the ray from `C_j` to `Y` meets crop `j`'s fronto plane at a point whose
   in-plane coordinates, in samples, minus the sample's own coordinates, are
   the predicted flow.

Each pair also has a free 2D offset, in samples, added to the predicted flow:
the point's projection and the true position of the surface in each photograph
differ by the keypoint and triangulation inconsistency, and the offset takes
that up so it does not bias the tilt. The pair's unknowns are the normal's two
tilt angles and the offset. A sample whose ray at step 2 is parallel to the
plane, or meets it behind a camera, is left out of the fit.

In the prototype, fitting this model to the exact predicted flow of a known
plane recovered the normal to 0.00°, so the geometry is right.

### Fitting each pair at each radius

At each radius `r` in `radii`, the step fits the normal to pair `(i, j)`'s
measured flow at the samples within `r` of the crop's centre that lie on both
photographs:

1. **A coarse grid.** Every normal on a grid of tilts from 0° to
   `max_grid_tilt_deg` in `grid_tilt_step_deg` steps, at azimuths
   `grid_azimuth_step_deg` apart, about the pair's mean viewing direction, with
   the offset at each solved in closed form; the grid's best is the start. The
   grid is why the result does not depend on the given normal. Centring it on
   the pair's mean viewing direction is this draft's choice, not a measured
   one: any centre that does not depend on the given normal would do.
2. **Gauss-Newton with Tukey weights** from that start, over the two tilt
   angles and the offset, the weights from the residuals' robust scale and
   `tukey_c`.
3. **σ.** The covariance of the tilt is the inverse of the Gauss-Newton Hessian
   times the weighted residual variance, read in the plane's own angular frame
   as the grown-plane step reads its σ; `sigma_deg` is the square root of its
   larger eigenvalue, in degrees. The residual variance is divided among the
   effective number of independent measurements, not the number of samples:
   DIS flow errors are correlated over about one DIS patch, taken as
   `correlation_length_px` (8) photograph pixels, and where the coarser of the
   two views is upsampled `k` times, that is `8k` samples. Counting every sample
   as independent would make σ too small by about `k`.

### Choosing the support

At each radius the step combines the pairs' normals:

1. **The medoid** of the pair normals: the pair normal with the least sum of
   angles to the others.
2. **The combined normal**: the σ-weighted mean (weights `1/σ²`) of the pairs
   whose normal is within `agreement_sigmas · σ + agreement_margin_deg` (3σ +
   5°) of the medoid, `σ` being that pair's own. Its σ is the inverse square
   root of the summed weights, in degrees.
3. **χ²**: the mean over the fitted pairs of (angle from the combined normal /
   that pair's σ)².

The chosen radius is the largest radius whose mean χ² is at most
`max_mean_chi2` (2): the largest support on which the pairs agree within their
own precision. The step turns the patch to the combined normal at that radius.

The reason the agreement between pairs, and not any one pair's readings, picks
the support: a support that is too large, one that runs over a fold of the
surface, a curve or another layer, affects each viewpoint differently, so the
pairs' normals spread apart once the disc reaches it. A single pair's σ keeps
falling as the radius grows, even across a fold, because more samples fit a
plane at some normal; a rule on one pair's σ or its steps cannot see the fold.
The rules the prototype tried and rejected are in the measurements
(§ "Choosing the support" and § "Rejected alternatives").

### Undetermined

When no radius has a mean χ² of at most `max_mean_chi2`, or no pair fitted at
any radius, the outcome is `Undetermined`: the photographs do not give a normal
the pairs agree on at any support, and the step leaves the given normal
unchanged. A point with one pair has no agreement to test; the prototype read
it at a fixed 8 samples, which this draft keeps as `SinglePair` until the open
question is settled.

### Output

`PairwiseFlowRadii` carries, beside the outcome, the normal and its σ, the
chosen radius, the mean χ² and the combined normal at every radius, and every
pair's angle, normals, σ and offsets at every radius, in the style of the
grown-plane step's report: enough to tell, for a point a person disagrees
with, which pairs disagreed and at which radius.

## Implementation notes

- **Order of work.** Crops are rendered once per observation; flows are
  computed once per pair, in parallel over pairs with rayon; the fits are per
  pair and radius, independent of one another, and also parallel. The combined
  normal and χ² are taken in pair order, so the result does not depend on the
  thread count.
- **Precision.** The crops and the flow are `f32`, as `GrayImage` and
  `FlowField` are: flow errors are 0.04 px or more, far above `f32`'s rounding
  at crop coordinates under 100. The fits accumulate a few thousand samples into
  4×4 normal equations and run in `f64`, which costs nothing at that size and
  keeps the Hessian's inverse well conditioned near edge-on tilts.
- **Cost.** The Python prototype took about 59 s for 153 points with 12
  processes, about 4.6 CPU-seconds per point, most of it in the fits; the flows
  were one full-resolution flow on a 96×96 crop per pair. A Rust version is
  expected to be far cheaper, chiefly in the fits, but the step stays expensive
  by design: the number of pairs grows with the square of the number of views.
  How to bound it is an open question.

## Parameters

| Option | Default | Source |
|--------|---------|--------|
| `crop_half_width` | 48 samples | prototype |
| `radii` | 4, 6, 8, 11, 16, 22, 32, 45 samples | prototype |
| `flow` | `high_quality`, `finest_scale` 0, `patch_stride` 1 | measured, § "Flow settings" |
| `max_grid_tilt_deg`, `grid_tilt_step_deg`, `grid_azimuth_step_deg` | 85°, 10°, 30° | prototype |
| `tukey_c` | 4.685 | the usual 95%-efficiency constant; not measured here |
| `correlation_length_px` | 8 px | a guess; open question |
| `agreement_sigmas`, `agreement_margin_deg` | 3, 5° | prototype |
| `max_mean_chi2` | 2 | measured, § "Choosing the support"; tuned on references the grown plane made |
| `single_pair_radius` | 8 samples | the best fixed radius measured; open question |

## Testing

Proposed tests for the step when it is built:

- **Exact geometry.** Crops rendered from a synthetic textured plane at a known
  normal, seen by three or more cameras at different angles: the step returns
  `Turned` within a fraction of a degree of the true normal, at the largest
  radius.
- **Start independence.** The same track given two normals 40° apart: the two
  answers are the same to within the fits' convergence tolerance.
- **A fold.** Two textured planes meeting at a crease near the point: the chosen
  radius stops short of the crease, and the normal is the near plane's.
- **Undetermined.** A textureless plane: `Undetermined`, and the track's normal
  is exactly the one given.
- **Refusals.** A track at infinity is refused as `NormalError::AtInfinity`; an
  empty `radii` or a radius past `crop_half_width` is refused by `check` naming
  the option; cancellation through `Progress` returns `NormalError::Cancelled`.

The synthetic scenes are small (a few cameras, crops of 96 samples), so the
tests take well under a second each.

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
   with the per-point outcomes it reports extended by `Undetermined`.
4. The glossary entry for the step's name.

## Open questions

- **Points with one pair.** There is no agreement to test; the prototype read
  them at 8 samples. Whether they should be `Undetermined`, read at a fixed
  radius, or chosen by another rule.
- **Many views.** A dino point can have 20 or more views, which is 190 pairs.
  Whether to cap the pairs, and how to choose them: widest angles first, a
  spread of baselines, each view in a fixed number of pairs, or the pairs whose
  geometry predicts the most information about the tilt, as the grown-plane
  step's selection does.
- **The thresholds' references.** `max_mean_chi2`, the agreement window and the
  radius list were tuned against reference normals that the grown-plane step
  made, apart from two hand normals. They need checking against normals that
  step did not make.
- **The correlation length.** 8 photograph px is a guess at how far flow errors
  stay correlated. It sets the scale of every σ and so the χ² threshold with it;
  it could be measured from the residuals of the synthetic pairs.
- **The combined σ.** The pairs reuse views, so their information is not
  independent and the combined σ is too small; whether to discount it, as the
  grown-plane step halves each kept pair's information.
- **Layered points.** A foreground object against a background makes the flow
  see two layers. The chosen radius should stop before the other layer; the χ²
  rule does so only where the layers make the pairs disagree, which a narrow
  baseline may not.
- **The fronto plane per view or a common plane.** Each crop faces its own
  camera. A common plane, such as the given patch's or the bisector of each
  pair's views, would make the two crops of a pair more alike before the flow,
  at the cost of reintroducing a dependence on a start.
- **GPU flow.** `compute_optical_flow` takes a `GpuFlowContext`
  ([gpu-optical-flow.md](../core/features/gpu-optical-flow.md)); whether batches
  of 96×96 crops gain from it, below its `gpu_min_pixels` default of 50,000.
