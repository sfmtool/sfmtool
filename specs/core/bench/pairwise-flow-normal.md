# Pairwise Flow Normal

A patch in a patch-based reconstruction is a small oriented square standing in
the world at a 3D point, and its normal says which way the square faces. The
pairwise-flow normal estimates that normal from dense optical flow. It renders
every photograph that sees the point onto one plane through the point, computes
the optical flow between the renders of pairs of photographs, and fits the plane
through the point to each pair's flow over discs of several radii. It then
chooses the support itself: the largest disc on which the pairs agree with each
other within their own precision. It does this in a few rounds, each rendering
on the plane the round before found. When the photographs carry too little
evidence about the tilt it says the normal is **undetermined**, and when the
pairs are precise but disagree at every disc it says the normal is
**unstable**; either way it leaves the normal as it was given. It moves neither
the patch's centre nor any of the point's image positions.

It is a bench step, `sfmtool_core::bench::pairwise_flow_normal`: it takes an
editable track at the track stage, as the bench's other normal steps do
([editable-track.md](editable-track.md) § "Estimating the normal"), and gives
the track back turned, with a report of every round. It is deliberately
expensive, one dense flow per pair of photographs per round, and stands beside
the cheap per-point step, the grown-plane normal
([grown-plane-normal.md](grown-plane-normal.md)), which it does not replace.
Track View, the wire and `refine_normals` do not run it, and its Python
binding carries only how it ended; the steps that would reach it from those
are proposed in
[pairwise-flow-normal-amendment.md](../../drafts/pairwise-flow-normal-amendment.md),
with the questions about its settings and rules that are still open. The
measurements behind its design and defaults are in
[pairwise-flow-normal-measurements.md](pairwise-flow-normal-measurements.md).

Builds on: [../features/optical-flow.md](../features/optical-flow.md), the DIS
optical flow it reads; [../camera/image-warping.md](../camera/image-warping.md)
§ "Choosing the sampler per view", the sampler its crops are rendered with.

## Why a separate step

The grown-plane step registers a few views against one reference view through
the plane, growing a window from the point out to the patch. It has three
properties that this step avoids:

- its answer can depend on the reference view it picks and on the normal it
  starts from (the ground truths' points are labelled `ok`, `A1` and `A2` by
  whether the step returns to the same plane from every start;
  [grown-plane-normal-measurements.md](grown-plane-normal-measurements.md#σ-in-the-planes-own-frame-2026-11-11));
- it fits exactly the patch, so the support the plane is read over is whatever
  size the patch was given, which is 3.4 photograph px in half-size at the
  median on a fresh solve of seoul_bull;
- it turns the patch whenever a view joins, unless the turn is large on weak
  evidence; its only outcome with no plane is that no view joined, and none
  says that the views joined but do not determine the normal.

This step reads no reference view, chooses its support, and separates the two
reasons for having no normal, as the glossary's **turned** / **undetermined**
/ **unstable** entry ([GLOSSARY.md](../../GLOSSARY.md)) keeps them apart. Its
uses are a refinement on the bench for a point a person is working on, a check
on the grown-plane step's output, and the test of whether a point's normal is
determined at all.

The step starts from the normal it is given, which in a reconstruction's
pipeline is the mean viewing direction and in the viewer whatever the patch
faces, possibly a normal set by hand or by another step. It does not assume the
mean viewing direction, and a point it does not turn keeps the normal it was
given. The given normal only places the plane the first round renders on; each
pair's fit searches the tilt afresh.

## Rust API

In [bench/normal/pairwise_flow.rs](../../../crates/sfmtool-core/src/bench/normal/pairwise_flow.rs),
with its pieces under
[bench/normal/pairwise_flow/](../../../crates/sfmtool-core/src/bench/normal/pairwise_flow/),
re-exported from `sfmtool_core::bench`, with the signature every bench normal
step has ([bench/normal.rs](../../../crates/sfmtool-core/src/bench/normal.rs)).

```rust
pub fn pairwise_flow_normal(
    track: &EditableTrack,
    edited: &EditedReconstruction,
    images: &[ProjectedImage<'_>],
    options: &PairwiseFlowNormalOptions,
    progress: &Progress<'_>,
) -> Result<(EditableTrack, NormalReport), NormalError>;

/// How the step renders its crops, computes its flows, fits its planes and
/// chooses its support. Defaults and their sources in § Parameters.
pub struct PairwiseFlowNormalOptions {
    pub rounds: u32,                        // 3; a point with two observations runs 1
    pub max_view_angle_deg: f64,            // 80
    pub max_normal_from_mean_view_deg: f64, // 80
    pub fallback_radius: f64,               // 1 half-extent; one of radii
    pub min_samples_per_patch: u32,         // 24
    pub max_samples_per_patch: u32,         // 128
    pub crop_half_extents: f64,             // 3
    pub min_crop_half_width_px: f64,        // 48 px of the finest view
    pub max_crop_samples: u32,              // 512
    pub radii: Vec<f64>,                    // [0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12] half-extents
    pub max_pairs: usize,                   // 10
    pub flow: DisFlowParams,                // high_quality, finest_scale Some(0), patch_stride 1
    pub max_grid_tilt_deg: f64,             // 85
    pub grid_tilt_step_deg: f64,            // 10
    pub grid_azimuth_step_deg: f64,         // 30
    pub tukey_c: f64,                       // 4.685
    pub correlation_length_px: f64,         // 8
    pub noise_false_alarm: f64,             // 0.001
    pub flow_floor_px: f64,                 // 0.03
    pub max_sigma_deg: f64,                 // 10
    pub agreement_sigmas: f64,              // 3
    pub agreement_margin_deg: f64,          // 5
    pub max_mean_chi2: f64,                 // 2
    pub single_pair_radius: f64,            // 1 half-extent; one of radii
    pub fit: FitOptions,                    // the reading the step ends with
}

impl PairwiseFlowNormalOptions {
    pub fn check(&self) -> Result<(), NormalError>;
}

/// What the step did, carried by NormalEstimate::PairwiseFlow(Box<_>).
pub struct PairwiseFlowRounds {
    pub rounds: Vec<PairwiseFlowRound>,  // in the order run
    pub outcome: PairwiseFlowOutcome,    // the last round's
    pub chosen_radius: Option<f64>,      // the last round's, where the patch was turned
    pub normal: Option<[f64; 3]>,        // the normal turned to, facing as the given one
    pub sigma_deg: Option<f64>,          // its σ, largest axis
    pub turn_deg: Option<f64>,           // from the given normal to it
}

pub struct PairwiseFlowRound {
    pub plane_normal: [f64; 3],          // the plane rendered on, facing the views
    pub sample_spacing: f64,             // world units per grid sample; NaN: no view saw the plane
    pub samples_per_patch: f64,          // by the rule, or fewer under the crop cap
    pub crop_half_width: u32,            // grid samples
    pub crops: Vec<PairwiseFlowCrop>,    // one per observation of the track
    pub pairs: Vec<PairwiseFlowPair>,    // the pairs read, in observation order
    pub radii: Vec<PairwiseFlowRadius>,  // one per options.radii entry
    pub outcome: PairwiseFlowOutcome,    // by the rule, on this round alone
    pub chosen_radius: Option<f64>,      // where the answer was read, also when refused as too far
    pub normal: Option<[f64; 3]>,        // the answer; None: no answer by the rule
    pub sigma_deg: Option<f64>,
    pub from_mean_view_deg: Option<f64>, // the normal read at chosen_radius from the mean viewing direction
    pub next_plane: Option<[f64; 3]>,    // the answer, else the fallback; None: no round follows
}

pub enum PairwiseFlowOutcome { Turned, SinglePair, Undetermined, Unstable }

pub enum PairwiseFlowCrop {
    Rendered { image: u32, angle_deg: f64, samples_per_pixel: f64, coverage: f64 },
    TooOblique { image: u32, angle_deg: f64 },
    NotAttempted, // not in, a second row in one image, a camera centre at the point
}

pub struct PairwiseFlowPair {
    pub observations: [usize; 2],        // the flow is read from the first's crop to the second's
    pub images: [u32; 2],
    pub angle_deg: f64,                  // between the two views, from the point
    pub level: u32,                      // the crops' pyramid level the pair is read at
    pub half_width: u32,                 // the square read, samples at that level
    pub fits: Vec<Option<PairwiseFlowFit>>, // one per radius
}

pub struct PairwiseFlowFit {
    pub normal: [f64; 3],                // facing the pair's cameras
    pub sigma_deg: f64,                  // the larger of the two below
    pub sigma_texture_deg: f64,
    pub sigma_residual_deg: f64,
    pub offset: [f64; 2],                // the pair's free shift, grid samples
    pub rms_residual: f64,               // grid samples, Tukey-weighted
    pub residual_variance: f64,          // σ_I², the photometric residual's variance over the disc
    pub determined: bool,                // sigma_deg at most max_sigma_deg
    pub combined: bool,                  // within the agreement window at this radius
}

pub struct PairwiseFlowRadius {
    pub radius: f64,                     // patch half-extents
    pub normal: Option<[f64; 3]>,        // the combined normal; None: no determined pair
    pub sigma_deg: f64,                  // NaN with no combined normal
    pub mean_chi2: f64,                  // NaN with fewer than two determined pairs
    pub pairs_fitted: usize,
    pub pairs_determined: usize,
    pub pairs_combined: usize,
}
```

**The outcomes.** `Turned`: the patch is turned to the last round's combined
normal at its chosen radius. `SinglePair`: a point with exactly two
observations, one pair, read in one round at `single_pair_radius` with no
agreement to test, its σ at most `max_sigma_deg` and its normal within
`max_normal_from_mean_view_deg` of the mean viewing direction, and turned.
`Undetermined`: too little evidence; no radius has two determined pairs, or a
two-observation point has no determined fit at `single_pair_radius`; the track
as given. `Unstable`: some radius has two or more determined pairs but at every
such radius they disagree, or the answer is more than
`max_normal_from_mean_view_deg` from the mean viewing direction; the track as
given. Only `Turned` and `SinglePair` move the normal; neither of the other two
is a refusal.

**The refusals.** `normal_preconditions` refuses a track at the cluster stage,
one with no patch, one at infinity (`NormalError::AtInfinity`) and one with
fewer than two `in` observations, and `check_views` too few views, as for every
normal step. `check` names the first setting the step does not take as
`NormalError::BadPairwiseFlowOption(name)`: at least one round; view and normal
angles above 0° and at most 90°; finite radii above zero, `radii` non-empty
and strictly ascending, `fallback_radius` and `single_pair_radius` each one of
its entries; `min_samples_per_patch` from 2 to `max_samples_per_patch`;
`crop_half_extents` of at least 1; `max_crop_samples` from twice
`min_samples_per_patch` to 4,096; `max_pairs` of at least 2; a coarse grid of
tilts under 90° in steps above 0 and azimuth steps above 0° and at most 360°,
holding at most 4,096 normals (refused as `grid_tilt_step_deg` where its tilts
alone pass that, as `grid_azimuth_step_deg` otherwise); a `correlation_length_px`
above 0 and at most 16; `noise_false_alarm` strictly between 0 and 1; finite
floors and agreement settings of at least 0; `max_sigma_deg` and
`max_mean_chi2` above 0, infinite allowed; and a read-back fit at a patch
resolution of at least 2. The flow's own settings are checked as
`compute_optical_flow` checks them, except against an image's size, and refused
as `NormalError::Flow`; a flow refusal for a particular crop size, which the
check cannot foresee, is `NormalError::Flow` from the step. Cancellation through
`progress` is `NormalError::Cancelled`.

**The report's sentence** (`Display`), followed by the turn and the reading as
for every normal step, reads for example "3 rounds; 6 of 6 pairs agree at 4
half-extents (χ² 0.3), σ 0.01°", or for one pair "1 round; one pair at 1
half-extents, σ 0.18°, single pair". A point the step does not turn ends with
the words `grid_plane_normal` uses for the same two outcomes: "3 rounds; 6
pairs, no radius with two determined pairs, normal left unchanged:
undetermined", "3 rounds; 10 pairs, no radius where they agree (least χ² 5.2
at 0.5 half-extents), normal left unchanged: unstable", or "1 round; 6 pairs,
the answer at 4 half-extents is 40° from the mean viewing direction, normal
left unchanged: unstable".

### Why it is shaped this way

**The signature is the other normal steps'.** The step ends as they do: it
turns the patch with `tilt_patch`
([bench/steps.rs](../../../crates/sfmtool-core/src/bench/steps.rs)) and reads the
track back through `turn_and_read`, so a caller can swap one step for another
and compare `NormalReport`s. Under `Undetermined` and `Unstable` it reads back
the track as given, as `grid_plane_normal` does. The two are kept apart, as
there, because a caller acts on them differently: missing evidence calls for
more views or a larger patch, and evidence that does not settle calls for a
look at the point (a fold, another layer). The read-back renders the bitmap from
the track's reference observation as every normal step's does; the estimate
itself reads no reference view.

**A point at infinity is refused**, as the other normal steps refuse it. A
point at infinity is a direction; its patch faces along its own bearing and
there is no plane through it to render on. Every other quantity is computed
from the finite point `X`, so no division by `w` occurs. A camera whose centre
is at `X` has no line of sight; its row is `NotAttempted`.

**It reads no keypoint.** The crops are sampled around the 3D point, not
around the keypoints, and each pair's free shift absorbs the difference between
the point's projection and where the photographs see it. A track whose
keypoints are off still gets a normal; one whose point is off by more than the
crop's half width does not.

**The settings are `*Options`** (STANDARDS § "A settings struct is named
*Options"). The flow's settings are a `DisFlowParams` field, which keeps its
name as backlog from before that ruling. The fallback and single-pair radii are
entries of `radii`, so the fits they read are in the report.

**It takes a `Progress`** (STANDARDS § "Long-running functions take a
Progress"): a point with 20 views reads 10 pairs, each a dense flow, in every
round. It opens a phase `pairwise flow normal`, inside it one `round` per
round, and inside each `crops`, `pyramids` and `pairs`, the last with a count
per pair. It checks for cancellation before each pair and before each radius
of a pair's fits.

**The report is per round** because the rounds are where the step moves: a
person checking a point needs the plane each round rendered on, what it found,
and, when a round had no answer, the fallback plane passed on.

**`SinglePair` is an outcome of its own** because a caller treats it
differently: the normal is a measurement, but nothing checked that the support
was not too large. It is turned only when its σ is at most `max_sigma_deg`,
and is `Undetermined` otherwise, so a two-view point on weak texture is not
turned to a normal its one pair cannot fix.

**The name** follows the glossary's rule for the bench normal steps
(**finite-difference normal** / **grid-plane normal** / **grown-plane normal**),
which names each step for the evidence it reads. This step reads dense optical
flow between pairs of the point's observations, so it is the *pairwise-flow
normal*. *Flow normal* alone would not say that the flow is read between views
rather than along a sequence, which is how
[optical-flow.md](../features/optical-flow.md) uses it everywhere else. The
report types are named, like `GridPlaneCells` and `GrownPlaneSequence`, for
what they hold: `PairwiseFlowRounds` the rounds, each `PairwiseFlowRound` one
reading per radius.

### Example

```rust
use sfmtool_core::bench::{pairwise_flow_normal, PairwiseFlowNormalOptions, PairwiseFlowOutcome};
use sfmtool_core::bench::NormalEstimate;
use sfmtool_core::progress::Progress;

let (turned, report) = pairwise_flow_normal(
    &track,
    &edited,
    &images,
    &PairwiseFlowNormalOptions::default(),
    &Progress::none(),
)?;
println!("{report}"); // "3 rounds; 10 of 10 pairs agree at 1.5 half-extents (χ² 1.4), σ 1.83°, turned 12.3°, …"
if let NormalEstimate::PairwiseFlow(rounds) = &report.estimate {
    if rounds.outcome == PairwiseFlowOutcome::Undetermined {
        // more views, or a larger patch
    }
}
```

### Python binding

No Python function runs the step. A `NormalReport` that carries its estimate
becomes, in the dict form every normal step's report has, an `estimate` of
`kind` `"pairwise_flow"` with `outcome` (`"turned"`, `"single_pair"`,
`"undetermined"` or `"unstable"`), `rounds` (how many ran), `chosen_radius`,
`normal`, `sigma_deg` and `turn_deg`, the rounds themselves left out
([sfmtool-py/src/bench/normal.rs](../../../crates/sfmtool-py/src/bench/normal.rs)).

## Theory

### Crops on a common plane

In each round every view is rendered onto one plane through the point `X`, the
round's **plane**: the first round's faces the normal the step is given, and
each later round's faces the normal the round before found
(§ [Rounds](#rounds)). The plane has no sign; it is rendered facing the point's
views. The crops share one grid and one pair of in-plane axes: the patch's `u`
projected onto the plane and normalised, and the plane's normal crossed with it
(where `u` is within about 6° of the plane's normal, `v` is projected instead).
A view whose line of sight at `X` is more than `max_view_angle_deg` (80°) from
the plane's normal is left out of the round (`TooOblique`): so near edge on,
the plane is compressed in its photograph by more than 5 times along one axis.

The views the step renders are the track's `in` observations, the first in each
image; any other row is `NotAttempted`. Each crop is rendered as an
`OrientedPatch` with the round's frame, centred on `X`, through
`WarpMap::from_patch`
([camera/warp_map.rs](../../../crates/sfmtool-core/src/camera/warp_map.rs)), which
maps each sample through the camera model onto an arbitrary oriented plane.
Lens distortion, including a fisheye's edges, is therefore removed, and every
crop shows the photograph as it would look on that plane. Each view is rendered
once per round, on the grid all the point's views share, so the flow is in one
unit across pairs; there are no per-pair renders.

**Why one common plane and not a plane per view.** Rendering each view on its
own fronto-parallel plane, the plane through `X` facing that camera, makes two
crops of a wide pair differ by a large projective warp, a rotation and a shear,
which DIS recovers poorly. On a kerry_park point whose pairs are 89° to 113°
apart the flow between fronto crops was chaotic and its leftover over 1 px
almost everywhere; on a dino point whose pairs are 30° to 50° apart the crops
were rotated and sheared by several pixels and the flow was reliable only near
the centre. On a common plane near the answer the crops of every pair line up,
and the flow measures only the small parallax the plane's remaining error
leaves. On six points traced both ways, per-view fronto planes gave 4
`Unstable` and 2 `Turned`; the common plane turned all 6, from 13.6° to 59.9°
off at the mean viewing direction to 1.1° to 4.3° after 3 rounds
(measurements, § "Real points with the revised design").

### The grid

**The grid follows the finest view.** The sample spacing is one photograph pixel
of the finest view, the view with the smallest pixel on the plane at the point,
but the patch's side is never fewer than `min_samples_per_patch` (24) samples
nor more than `max_samples_per_patch` (128). A view's pixel is read along its
coarser axis on the plane: the reciprocal of the smaller singular value of the
plane-to-photograph Jacobian at the point. At one sample per pixel of the finest
view no view's detail is lost to the grid until the upper bound: in the
prototype, a fixed 24 samples per side lost detail on the dino ground truth's
patches, about 43 px across, and its error at a fixed radius of 1 half-extent
rose from 1.7° with the earlier pixel-based crops to 2.9°; at native resolution
it was 1.3°. The lower bound keeps a tiny patch on enough samples for the flow
and the fits. The upper bound bounds the cost: a patch that covers 1024
photograph pixels across in its finest view gets 128 samples per side, each
spanning 8 pixels, and a crop of 3 times the patch is 384 samples a side rather
than 3072.

**Views finer than the grid are prefiltered.** Where the grid is held at its
upper bound a view can have pixels finer than the grid; it is read through the
photograph's pyramid down to the grid, so every such view is filtered alike and
its texture finer than the grid cannot alias into false structure that the flow
matches. A view coarser than the grid is upsampled, which only blurs; blur that
both crops of a pair share is harmless to the flow, and a pair whose crops
differ in blur is read at a coarser level
(§ [The pyramid level of a pair](#the-pyramid-level-of-a-pair)).

The renderers already sample this way. With `SamplerChoice::PerView`
([sampler.rs](../../../crates/sfmtool-core/src/camera/sampler.rs)) a view is read
with `Sampler::BilinearMip` at the level `bilinear_mip_level` gives,
`round(log2 max(σ_major, 1))` for `σ_major` the larger singular value of the
grid's Jacobian into the photograph in photograph px per sample, or with the
anisotropic sampler where the compression is much stronger along one axis, as
it is for a view oblique to the plane. The levels are those of the
`ImageU8Pyramid` each `ProjectedImage` carries, built by 2× box-filter
downsampling ([camera/image.rs](../../../crates/sfmtool-core/src/camera/image.rs)).
Because the level is rounded, two views on either side of a level boundary are
filtered up to √2 apart in scale. The crops are then converted to the flow's
grey (`matcher_gray_from_rgb`), and a sample off the photograph holds the mean
of the samples on it, so the flow's pyramid blurs no edge into the crop.

### The extent

Every crop of a round is rendered over a square of half-width
`max(crop_half_extents · h, min_crop_half_width_px · p_f)` in the world, where
`h` is the patch's half-extent and `p_f` the size, at `X`, of one photograph
pixel of the point's finest view on the plane: 3 times the patch by default,
but never fewer than 48 of the finest view's pixels, rounded up to whole
samples. The floor is there for tiny patches: a fresh solve of seoul_bull has a
median patch half-size of 3.4 px, so 3 times the patch would be about 10 px,
while the first prototype's best fixed support on those points was 16 px, about
4.7 times the patch, and the support chooser needs room above that to find
where the surface stops being one plane.

**Why the finest view and not the pair's coarser view.** With the floor set per
pair, at 48 pixels of the pair's coarser view, a point with one far view got
pair squares, and so allowed radii, that were large in the world: on seoul_bull
points 70, 173 and 255 the rule picked 4 to 12 half-extents, about 76 to 152
pixels of the finest view, where the first prototype, with its crops set in
pixels, never went past 45 px (measurements, § "Real points with the revised
design"). With the floor in the finest view's pixels, the radii a pair can reach
are the same for every pair, and a pair with a far view simply reads fewer of
that view's own pixels: at its level its square is `2^ℓ` times narrower in
samples, and its σ larger.

**The cap and the cost.** `max_crop_samples` (512) bounds the one render per
view. A crop's half-width in grid samples is `max(3h, 48 · k_f)`, `h` here in
grid samples and `k_f` the `samples_per_pixel` of the finest view: 1 where the
grid is at the finest view's pixel, over 1 where the grid is held at its lower
bound on a tiny patch, under 1 where it is held at its upper bound on a large
one. The grid is coarsened for the whole round, with the report carrying the
`samples_per_patch` used, only when the render would have more than
`max_crop_samples` along a side, keeping the world half-width. `3h` is at most
192 samples, so with the defaults that happens only when the floor exceeds the
cap, `2 · 48 · k_f > 512`: a finest view whose pixel spans more than about 5.3
grid samples, that is a patch under about 4.5 pixels across in its finest view.
The coarsening may take the grid under `min_samples_per_patch`; the extent is
kept and the resolution given up, because the floor is what leaves the support
chooser room.

What bounds each pair's flow is the render and its level. A pair read at level
`ℓ` reads the render's whole square at `2^ℓ` times fewer samples a side. A pair
whose square at its level is narrower than the flow's `patch_size` has no flow
and no fits; it is reported, not refused.

### The pyramid level of a pair

The scale only matters within a pair when one of its views is coarser than the
grid. Then that view has been upsampled by `k`, its `samples_per_pixel` (over
1), and its crop is blurred over `k` samples while the other crop is sharp; DIS
would match a sharp crop against a blurred one. So each rendered crop gets a
**crop pyramid**, the flow's own `ImagePyramid`
([optical_flow/pyramid.rs](../../../crates/sfmtool-core/src/features/optical_flow/pyramid.rs)),
whose levels are a blur by the 6-tap σ = 1 kernel and a 2× downsample, built
once per round to the deepest level any of the crop's pairs reads. A pair is
read at level `ℓ = floor(log2 k)` of both crops' pyramids, `k` being the
coarser view's; at that level the coarser view is upsampled by `m = k / 2^ℓ`,
between 1 and 2, and the two crops carry about the same blur. A pair whose
views are both at or finer than the grid is read at level 0. The flow at level
`ℓ` is in level samples, `2^ℓ` grid samples each. Reading a pair of unequal
views at the coarser one's level also cuts the flow's cost by `4^ℓ`. A sample
of a level is valid where every sample under its kernel was.

### Radii in patch half-extents

The radii are multiples of `h`, so every pair is compared at the same radius in
the world. That is what the χ² rule needs: it detects the radius at which the
surface stops being one plane, which is a distance on the surface, not a count
of any one view's pixels. A radius that covers few pixels of a pair's coarser
view gives that pair a large σ and so little weight, with no special case; a
radius past the square a pair reads gives that pair no fit there.

### Choosing the pairs

A round reads every pair of its rendered crops, up to `max_pairs` (10). A round
with more pairs reads `max_pairs` of them, chosen by a partial Fisher-Yates
shuffle of the pairs in order with a fixed seed: the prototype's random choice
of 10, made repeatable, so the same number of rendered crops always gives the
same pairs. The pairs read are reported in observation order. A point with 20
views has 190 pairs, so without the cap the cost would grow with the square of
the views. How to choose the pairs better is an open question
([pairwise-flow-normal-amendment.md](../../drafts/pairwise-flow-normal-amendment.md)).

### Flow

For every pair `(i, j)` read, `i` before `j` in observation order, the step
computes the DIS flow from crop `i` to crop `j` with `compute_optical_flow`
([optical_flow/mod.rs](../../../crates/sfmtool-core/src/features/optical_flow/mod.rs)),
on the square the pair reads at its pyramid level, at that level's full
resolution: `finest_scale` `Some(0)`, `patch_stride` 1, and otherwise
`DisFlowParams::high_quality()` (patch 12, 16 gradient-descent iterations,
variational refinement on).

`high_quality` itself is not enough. It fixes `finest_scale` at 1, so it never
solves at full resolution; on the prototype's crops of 96 samples the automatic
coarsest scale is 1, so the whole flow is solved at half resolution and
upsampled bilinearly ([optical-flow.md](../features/optical-flow.md#presets)).
On synthetic pairs, a real crop warped by the flow of a known plane, the
prototype measured a median flow error of 0.093 px with `high_quality` and
0.040 px at full resolution with stride 1 and variational refinement, and a
median normal error of 2.0° against 0.9° (measurements, § "Flow settings").
The flow runs on the CPU; whether the larger crops gain from a
`GpuFlowContext` is an open question
([pairwise-flow-normal-amendment.md](../../drafts/pairwise-flow-normal-amendment.md)).

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
normal's two tilt angles and the offset. A sample whose ray at step 2 or 3 is
parallel to the plane it should meet, or meets it behind the camera it leaves
from, is left out of the fit. When the candidate plane is the round's plane,
every predicted flow is zero: the flow measures only the parallax the plane's
error leaves. In the prototype, fitting this model to the exact predicted flow
of a known plane recovered the normal to 0.00°.

### Fitting each pair at each radius

At each radius `r` in `radii`, the step fits the normal to pair `(i, j)`'s
measured flow at the samples within `r · h` of the crop's centre that lie on
both photographs, at the pair's pyramid level:

1. **A coarse grid.** Every normal on a grid of tilts from 0° to
   `max_grid_tilt_deg` in `grid_tilt_step_deg` steps, at azimuths
   `grid_azimuth_step_deg` apart, about the pair's mean viewing direction, with
   the offset at each solved in closed form, scored on at most 4,096 of the
   disc's samples at an even stride; the grid's best is the start. The fit
   therefore searches the tilt afresh in every round, whatever plane the crops
   are on.
2. **Gauss-Newton with texture and robust weights** from that start, damped as
   Levenberg-Marquardt, over the two tilt angles and the offset, reading every
   sample. Each sample's flow residual `r_s` (a 2-vector, measured minus
   predicted) enters as `r_sᵀ W_s r_s`, where `W_s` is the sample's flow
   information from the image structure (below). Each sample also carries a
   Tukey weight on its Mahalanobis distance `√(r_sᵀ W_s r_s)`, in units of the
   whitened residuals' robust scale with `tukey_c`; the weights are recomputed
   after every step, and they multiply the texture weights. Without them a
   strong edge, often the outline of the surface against another layer, gets
   the full weight its contrast earns even where its flow belongs to the other
   layer.
3. **σ from the texture.** The covariance of the tilt is the inverse of the
   Hessian `H = Σ J_sᵀ W_s J_s` (with `J_s` the derivative of the predicted flow
   at sample `s` with respect to the two tilt angles and the offset, and the
   Tukey weights folded in), read about two orthonormal axes in the fitted
   plane's own tangent plane; `sigma_texture_deg` is the square root of its
   larger eigenvalue, in degrees.
4. **Correlated samples.** `H` sums over samples, but DIS flow errors are
   correlated over a distance of the order of the DIS patch (12 samples). The
   step takes `correlation_length_px`, 8 pixels of the pair's coarser view as
   its crop reads them: with `m` the factor by which that view is upsampled at
   the pair's level, or 1 where neither view is coarser than the grid, a square
   of side `L = round(8m)` samples at that level. Each sample's `W_s` is the
   information of the `L × L` square about it, so summing it over every sample
   counts each square's information `L²` times; `H` is therefore divided by
   `L²`, and each correlated square counts once.
5. **σ from the residuals.** `sigma_residual_deg` is read the same way from the
   covariance `s² · L² · (Σ J_sᵀ J_s)⁻¹`, the sums over the samples where the
   Tukey weight and the information are both non-zero, and `s²` the
   Tukey-weighted mean squared residual per flow component there.
6. **σ.** `sigma_deg` is the larger of the two. The texture σ alone was far too
   small on real photographs: on good points the combined normal was close to
   the reference (on one seoul_bull point, 1.6° to 8° at every radius) while
   the mean χ² was 44 to 2,500, and 600 to 50,000 on the points whose grid had
   90 or more samples per patch side. Real errors (in the calibration, a slight
   curvature, keypoint offsets beyond a shift) are correlated across the whole
   patch, not independent per correlated square, and the texture σ cannot see
   them. The texture σ is what makes a flat crop `Undetermined`; the residual σ
   is what keeps χ² honest where the texture is strong.

A pair whose disc holds no information, whose grid has no normal that predicts
the flow at 95% of the scored samples, or whose tilt the disc does not fix (a
singular Hessian, as for a disc whose texture runs one way everywhere, such as
parallel stripes) has no fit at that radius (`None`). A fit whose `sigma_deg`
is over `max_sigma_deg` is not **determined**: it is reported but does not
enter the combination or χ².

**The flow information of a sample.** The geometric derivative `J_s` alone
says nothing about texture: on a textureless crop the smoothed DIS flow is near
zero with small residuals, and a Hessian of geometry alone times that small
residual variance would give each pair a confident normal. The information must
depend on how well the image pins the flow at each sample. The step takes it
from the Lucas-Kanade measurement model: a flow read over a window is measured
precisely only along the image gradients in it, with covariance
`σ_I² · T⁻¹`, where `T` is the window's **structure tensor**, the sum of
`∇I ∇Iᵀ` over the window, and `σ_I²` is the variance of the photometric
residual. So:

- `T_i(s)` is the structure tensor of crop `i`'s grey values, from central
  differences that are zero where a neighbour is off the photograph, over the
  `L × L` square centred on `s`, the correlated square of step 4, so the window
  the information is read over and the area it is counted once per are the
  same; `T_j(s)` is crop `j`'s at the sample nearest `s` plus the measured flow.
- Each tensor keeps only what is significant against noise. Removing the
  expected noise contribution from the diagonal is not enough: it removes the
  noise only on average, and a sum of `n = L²` noise gradients fluctuates about
  its mean. For gradients of independent Gaussian noise with per-component
  variance `v`, each diagonal entry of the tensor has mean `n·v` and standard
  deviation `v·√(2n)`, `64v ± 11v` for `L = 8`; neighbouring gradients share
  pixels and an upsampled crop is smooth over `k` samples, so the true spread
  is larger than that formula. The step therefore takes the threshold from a
  simulation: for each window side, each crop `samples_per_pixel` and each
  pyramid level the pair reads it at, it renders pure white noise the way that
  crop is rendered, takes it to that level, computes the tensor's larger
  eigenvalue over many windows, and takes its `1 − noise_false_alarm` quantile
  (99.9%), in units of the noise variance. That threshold, times the crop's
  noise variance `σ_I²/2`, is subtracted from both eigenvalues of each tensor,
  and eigenvalues below zero are set to zero. A window on a flat crop then has
  zero information except in about one window in a thousand, and there only the
  small excess over the threshold. A textured window also loses the margin
  between the threshold and the noise's mean contribution: its information is
  understated, so its σ errs large, which matters only for texture near the
  noise level.
- The noise variance is read from `σ_I²`, which includes the flow's
  registration error as well as the image noise. That makes the threshold
  larger than the image noise alone would, which removes more information than
  necessary, never less.
- The two are combined as the **parallel sum** `T_i (T_i + T_j)⁺ T_j`, the
  inverse of the sum of the two covariances. The flow is the difference of two
  positions, one read in each crop, so its error is the sum of the two
  measurement errors and its information the parallel sum: never more than the
  weaker crop's tensor along any direction, and zero where either crop is flat.
  The plain sum `T_i + T_j` would credit a pair with texture only one of its
  crops has, as where one crop is blurred, occluded or off the surface.
- `W_s = T_s / (σ_I²/2)`, `T_s` being that parallel sum: each crop's position
  is read with covariance `(σ_I²/2) · T⁻¹`, so where the two tensors are equal
  `W_s` is `T / σ_I²`, the Lucas-Kanade information of the flow. `σ_I²` is the
  pair's photometric residual variance over the disc (crop `j` read at the end
  of each sample's flow against crop `i`), floored at the 8-bit quantization
  variance so a perfect match does not claim infinite precision; the report
  carries it as `residual_variance`.

**A floor on the flow's precision.** Strong texture makes `W_s` large, but the
flow DIS returns has errors the structure tensor does not model, which do not
shrink as the contrast rises. `flow_floor_px` is set, as
`correlation_length_px` is, in pixels of the pair's coarser view as its crop
reads them, so at the pair's level it is `f = flow_floor_px · m` samples. Each
correlated square's flow measurement has the covariance `W_s⁻¹ + f² · I`, so the
information one sample carries is `w (I + f² L² w)⁻¹` with `w = W_s / L²`. On
weak texture the floor changes nothing; on strong texture it caps each square's
information at `1/f²`. In a prototype on synthetic scenes, σ without the floor
was too small where the texture was strong, the error 2 to 4 times σ at the
highest contrasts, and one clean plane in 24 came out `Unstable`; with a floor
of 0.02 px the error was 1 to 2 times σ, with 0.05 px under σ, and no clean
plane was `Unstable` with either (measurements, § "Synthetic scenes"). The
default, 0.03, is between the two.

Because `W_s` is absolute, not scaled by the flow residual's own variance, and
counts only structure significant against noise, a flat crop gives an `H` that
is zero or nearly so, so the texture σ is far above `max_sigma_deg` or there is
no fit, and no pair is determined, whatever the residual σ. Weak texture gives a
σ that grows as the texture weakens.

### Choosing the support

At each radius the step combines the normals of the determined pairs, each
turned to face the point's mean viewing direction:

1. **The medoid** of the pair normals: the pair normal with the least sum of
   angles to the others, the first in pair order on a tie.
2. **The combined normal**: the σ-weighted mean (weights `1/σ²`), normalised,
   of the pairs whose normal is within `agreement_sigmas · σ +
   agreement_margin_deg` (3σ + 5°) of the medoid, `σ` being that pair's own.
   Its σ is the inverse square root of the summed weights, in degrees.
3. **χ²**: the mean over the determined pairs, those outside the window
   included, of (angle from the combined normal / that pair's σ)², taken only
   where at least two pairs are determined; with fewer it is `NaN`.

A radius has **enough evidence** when at least two pairs are determined there,
so that their agreement can be tested. The chosen radius is the largest radius
with enough evidence whose mean χ² is at most `max_mean_chi2` (2): the largest
support on which the pairs agree within their own precision. The round's answer
is the combined normal at that radius.

A radius with only one determined pair, on a point with several pairs, is not
used at all. Its χ² over one pair would be 0, so counting it would let the
largest such radius win with no test of agreement, which is the rejected
"radius of smallest σ" rule (one pair's σ keeps falling as the disc grows)
without even the single-pair bar. It is not read as `SinglePair` either: when a
point has several pairs and only one of them fixes the tilt, the other pairs'
failure is itself evidence that the views see the surface poorly, and the one
fit has nothing to check it against.

The reason the agreement between pairs, and not any one pair's readings, picks
the support: a support that is too large, one that runs over a fold of the
surface, a curve or another layer, affects each viewpoint differently, so the
pairs' normals spread apart once the disc reaches it. A single pair's σ keeps
falling as the radius grows, even across a fold, because more samples fit a
plane at some normal; a rule on one pair's σ or its steps cannot see the fold.
The rules the prototype tried and rejected are in the measurements
(§ "Choosing the support" and § "Rejected alternatives").

**A crease through the point.** On two faces meeting at a crease through `X`,
every disc straddles the crease and each pair's fit is a compromise between the
faces that depends on how its viewpoints foreshorten each face. The pairs agree
on small discs, where the two faces barely differ in the flow, and disagree
more as the disc grows, so the rule shrinks the support and turns the patch to
about the bisector of the faces. With narrow baselines every pair sees about
the same compromise and the support can be large; the outcome is `Unstable`
only where no radius passes the χ² rule.

### Rounds

The step runs `rounds` (3) rounds. Round 1 renders on the plane facing the
normal the step is given; round `k + 1` renders on round `k`'s answer, so each
round's crops line up more closely than the last and its flow measures a
smaller parallax. When a round has no answer by the rule, the next round's
plane is the **fallback**: the combined normal at `fallback_radius` (1
half-extent) over the pairs with a finite σ there, determined or not. Every
round is reported, and the last round passes on no plane.

**A plane the views cannot see is no answer.** The point's **mean viewing
direction** is the normalised sum of the unit vectors from `X` to the camera
centres of the views the step renders (the given normal where they cancel). A
normal more than `max_normal_from_mean_view_deg` (80°) from it faces a plane
the views see nearly edge on or from behind. A round whose answer is that far
does not pass it on: the step stops after that round, and its outcome is
`Unstable`, since the evidence settled on a plane that cannot be the surface
the photographs show. A fallback plane that far is not passed on either. Where
a round has no plane to pass on, because it has no answer and its fallback is
missing or too far, the step stops, as rendering on the same plane again would
repeat the same round. The bar does not catch every drift: on seoul_bull point
207, a bush with the bull in front and trees behind, the prototype ended 112°
from the reference while within 80° of the mean viewing direction.

**A point with one pair runs one round.** A point with one pair is a point with
exactly two observations the step renders, whichever views a round leaves out
as too oblique. With one pair there is no agreement to test, so nothing stops a
round from moving the plane further along that pair's error, and the next round
renders on that error. In the prototype a two-view point of seoul_bull, 160,
drifted 14° → 40° → 55° → 68° from the reference over the rounds. Its one round
renders on the given normal and reads the pair at `single_pair_radius` (1
half-extent): `SinglePair` where its σ there is at most `max_sigma_deg` and its
normal within `max_normal_from_mean_view_deg` of the mean viewing direction,
`Unstable` for a normal too far from it, `Undetermined` otherwise.

A point with more observations follows the ordinary rule in every round, even
when a round's too-oblique views leave it only one rendered pair. That round
cannot have two determined pairs, so it has no answer of its own, and the
fallback decides whether there is a next plane; the next plane may bring the
views it left out back within `max_view_angle_deg`. A round with fewer than two
rendered crops has no pair, so it has neither an answer nor a fallback, and the
step stops there.

The step's outcome, chosen radius, normal and σ are those of the last round it
ran. A last round with no answer leaves the given normal unchanged, even where
an earlier round had one. The answer the patch is turned to faces as the given
normal faces. In the prototype, 3 rounds from the mean viewing direction
brought the ground-truth dino points from 28.8° to 1.2° median and the
kerry_park points from 48.9° to 3.7°, the kerry_park points still improving at
round 3; the seoul_bull points stopped improving after round 1, at about 10° to
18° (measurements, § "Real points with the revised design").

### Output

`PairwiseFlowRounds` carries the outcome, the normal and its σ and the chosen
radius, and every round: its plane, its crops (with the views left out as too
oblique), the mean χ² and the combined normal at every radius, every pair's
angle, level, normals, both σs, offsets and photometric variance at every
radius, its answer and the plane it passed on. That is in the style of the
grown-plane step's report: enough to tell, for a point a person disagrees with,
which round went wrong and which pairs disagreed at which radius.

## Implementation notes

- **Order of work.** In each round every crop is rendered once, in parallel
  over views; the pairs are chosen from the rendered crops; each crop's pyramid
  is built once, to the deepest level any of its pairs reads, in parallel over
  crops; then each pair's flow and its fits at every radius are read, in
  parallel over pairs with rayon. A pair's reading depends on its two crops
  alone, and the combination and χ² are taken in pair order, so the result does
  not depend on the thread count (the measurements' example gave the same
  answers on one thread and on 32).
- **The pieces.** `crops` sizes the grid and renders the crops and their
  pyramids, `model` is the pair model and its derivative, `flow` the DIS call,
  `texture` the information of each sample, `fit` one pair's fit at one radius,
  `support` the combination, the support rule and a round's decision, and
  `rounds` one round composed from them; `report` holds the public report
  types.
- **Precision.** The crops, their pyramids and the flow are `f32`, as
  `GrayImage` and `FlowField` are: flow errors are 0.04 px or more, far above
  `f32`'s rounding at crop coordinates under 4,096. The crop's fill mean and
  the noise simulation's field variance are summed in `f32`, each row and then
  the row sums, whose rounding is bounded by about `(rows + columns) · 2⁻²⁴` of
  the sum: `5 · 10⁻⁴` over the largest crop the check allows, for a fill that
  only has to be near the crop's grey, and `1.5 · 10⁻⁵` over the simulated
  field, far below the sampling error of its 0.999 quantile. The structure
  tensors are `f32` box sums of at most 32² products. The per-sample
  information, the disc's sums and the fits are `f64`: they accumulate a few
  thousand samples into 4×4 normal equations, which costs nothing at that size
  and keeps the Hessian's inverse well conditioned near edge-on tilts. The
  combination reads a handful of pair normals in `f64`.
- **The correlated square.** Its side `L = correlation_length_px · m` is
  rounded to whole samples at the pair's level, and that whole side is both the
  structure tensor's window and the `L²` the information is shared over.
- **The noise threshold.** It is simulated per window side, crop
  `samples_per_pixel` (read in quarter steps, at least 1 and at most 64) and
  level: white noise drawn at the photograph's pixels and sampled bilinearly
  onto the grid where the crop is upsampled, at the grid's samples where it is
  not, taken down the flow's pyramid and scaled to unit variance, four fields
  of 128 × 128 samples at the level read, about 50,000 windows. A level deeper
  than 3 is simulated at level 3 with `samples_per_pixel` scaled down by the
  levels left out, which bounds the simulated field at 1024 samples a side. The
  draws come from a fixed SplitMix64 sequence per setting, so the threshold is
  the same on every machine. Each setting's threshold is kept for the life of
  the process. Two threads that ask for a setting not yet kept at the same time
  may both simulate it; they compute the same value from the same draws, and
  the second store replaces the first with an equal one.
- **The pair choice.** The shuffle draws from `numeric::splitmix64` with a
  fixed seed, so it is the same on every machine.
- **Cost.** On the prototype's curated ground-truth points, with every option
  at its default, the step took a median of 4.4 s per point on one thread on
  dino_dog_toy and 1.3 s on kerry_park, and 0.9 to 1.5 s and 0.3 to 0.5 s on
  32 threads (measurements, § "The Rust step on real points"). The parallel
  speedup is held to about 4 by the 10 pairs a round reads. Python prototypes
  of the design took several CPU-seconds or more per point.

## Parameters

| Option | Default | Source |
|--------|---------|--------|
| `rounds` | 3 | measured on real points up to 3 rounds (measurements, § "Real points with the revised design"); not beyond |
| `max_view_angle_deg` | 80° | the prototype's; not measured against other values |
| `max_normal_from_mean_view_deg` | 80° | the prototype's, in the runs after the two changes of § "The crop floor and the bar on the mean viewing direction"; not measured against other values |
| `fallback_radius` | 1 half-extent | the prototype's choice; not measured against other values |
| `min_samples_per_patch`, `max_samples_per_patch` | 24, 128 | the prototype's native grid with these bounds; the bounds themselves not measured against other values |
| `crop_half_extents` | 3 | used in the real-point runs of the revised design; not measured against other values |
| `min_crop_half_width_px` | 48 px of the finest view | used in the real-point runs after the two changes (before them, 48 px of the pair's coarser view); not measured against other values |
| `max_crop_samples` | 512 | used in the real-point runs; bounds the render per view only, so the grid is coarsened only for a patch under about 4.5 px across in its finest view; not measured against other values |
| `radii` | 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12 half-extents | used in the real-point runs; not measured against other lists |
| `max_pairs` | 10 | the prototype's cap, there chosen at random; not measured against other values |
| `flow` | `high_quality`, `finest_scale` 0, `patch_stride` 1 | measured, § "Flow settings" |
| `max_grid_tilt_deg`, `grid_tilt_step_deg`, `grid_azimuth_step_deg` | 85°, 10°, 30° | prototype |
| `tukey_c` | 4.685 | the usual 95%-efficiency constant; not measured here |
| `correlation_length_px` | 8 px | used in the real-point runs; a guess |
| `noise_false_alarm` | 0.001 | used in the real-point and synthetic runs; not measured against other values |
| `flow_floor_px` | 0.03 px | measured only on synthetic scenes at one sample per pixel (measurements, § "Synthetic scenes"): 0.02 left the error 1–2 times σ, 0.05 under 1 times σ; used in the real-point runs |
| `max_sigma_deg` | 10° | measured on real points with the revised design: the best of the settings tried |
| `agreement_sigmas`, `agreement_margin_deg` | 3, 5° | prototype |
| `max_mean_chi2` | 2 | measured on real points: 1 and 2 were the best of the settings tried, 2 turning more points (measurements, § "Real points with the revised design" and § "The Rust step on real points") |
| `single_pair_radius` | 1 half-extent | proposed; the first prototype's best fixed radius was 8 photograph px, which has no fixed value in half-extents |

## Testing

The pieces are tested in their own modules under
[bench/normal/pairwise_flow/](../../../crates/sfmtool-core/src/bench/normal/pairwise_flow/):
the options' check names the first setting that fails, the bounds on the crop
size, the coarse grid and the correlation length among them; the grid follows the
finest view and the crop cap coarsens it keeping the extent; views too oblique
or at the point are left out; crops of the photographed plane line up and the
pyramid's mask shrinks with its kernel; a pair is read at its coarser view's
level; the pair model predicts no flow on the round's plane, ends where the
second camera sees the candidate, and its derivative matches finite
differences; a fit recovers a plane from its exact flow, has none on a disc
with no information or on stripes, down-weights another layer's flow, and its
residual σ matches the spread of noisy fits; the structure tensors, the
significance threshold, the parallel sum and the floor; a flat crop carries no
information; and the combination, the support rule, the decision with its bar
and fallback, the single pair and the pair choice, given pair fits directly,
including one determined pair among several (undetermined) and an answer 85°
from the mean viewing direction (unstable, no next plane).

The step itself is tested end to end in
[bench/tests/pairwise_flow_normal.rs](../../../crates/sfmtool-core/src/bench/tests/pairwise_flow_normal.rs),
on synthetic surfaces photographed by pinholes around the point, with noise of
2 grey levels:

- **A textured plane given a normal 40° off**, in two directions: each round's
  plane is closer to the truth than the last, and the patch is turned to
  within a degree of it, within 3 times the largest σ among the pairs combined
  at the chosen radius.
- **A flat plane**, constant grey: no pair is determined in any round, the
  outcome is `Undetermined` and the normal is the one given; with two views it
  is `Undetermined` too, after one round, not `SinglePair`.
- **Weak texture**, contrast 12 down to 0.5 grey levels: never `Unstable`;
  `Turned` within 3 times the largest combined pair σ where turned; σ grows as
  the contrast falls, and every undetermined contrast's σ is larger than every
  turned one's.
- **A crease through the point**, a ridge of two faces at 90° with five
  cameras in front of its edge: `Turned` at a radius of at most 0.75
  half-extents within 3° of the bisector, every larger radius with two
  determined pairs over the χ² bar, the largest over 10.
- **A view left out as too oblique** to the first round's plane: round 1 has
  one pair and no answer and passes on its fallback; round 2 renders all three
  views; the step runs every round and turns the patch onto the plane.
- **One pair runs one round**, rendered on the given normal, `SinglePair`.
- **An answer too far from the mean viewing direction**, the bar lowered to
  20° on cameras to one side of the plane: `Unstable` after one round with the
  normal as given; at the default bar the same scene is turned.
- **A view four times as far** as three others: each of its pairs is read at
  level 2 on a square a quarter as wide, fits and enters the combination, and
  the patch is turned onto the plane.
- **Refusals and cancellation**: a track at infinity is refused as
  `NormalError::AtInfinity`, `rounds` of 0 as a bad option, and a cancel flag
  set as the first pair is counted stops the step with `NormalError::Cancelled`
  after that one pair.

Each scene is a few cameras of 160 × 160 px and crops of about 100 samples a
side, so each run of the step takes well under a second; the weak-texture
test, five runs, takes about 3 s, and each of the others under 1 s.

## Non-goals

- **Moving the centre or the keypoints.** Like the grown-plane step, it turns
  the patch about its centre and moves neither the centre nor any keypoint.
- **Replacing the grown-plane step.** That step stays the per-point step of
  `refine_normals` and Surface Tiling Mode; this one is the expensive check and
  refinement beside it.
- **A curved surface model.** The step fits a plane and chooses the support on
  which a plane fits; it does not fit curvature.
