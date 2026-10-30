# Bundle-adjusting a reconstruction

A reconstruction is a set of camera poses and a set of 3D points, and the only
evidence for either is where the points landed in the photographs. Bundle
adjustment is the joint refinement that takes all of that evidence at once: it
moves every pose and every point together until the pixels they predict sit as
close as they can to the pixels that were actually measured. This spec describes
the function that runs one over a whole reconstruction value: what it gathers,
what it refuses, what it writes back, and what it does with a point the solve can
no longer support.

It is not the solver. The solver is the staged robust array kernel in
[`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md), which
takes poses, points and observations as plain arrays over a list of cameras.
What this adds is the layer between that kernel and a reconstruction: which
images are posed, which cameras they were taken through, which observations
carry a pixel, what each point's stored constraint means, and how the answer goes back into a value whose patch frames,
colours and tracks have to come out the other side still describing the same
scene.

The function is pure. It takes the reconstruction value and its options, and
returns a new value plus a report; the input is left exactly as it was whether
the adjustment succeeds or is refused.

Related specs:
[`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md) (the
kernel, its schedule, its point constraints and its points at infinity),
[`edited-reconstruction.md`](edited-reconstruction.md) (the value, and the row
map a caller follows an index through),
[`../bench/editable-track.md`](../bench/editable-track.md) (the bench fit, which
rescales a patch frame the same way when a track crosses between a place and a
bearing), and
[`../../gui/edits/bundle-adjust.md`](../../gui/edits/bundle-adjust.md) (the
viewer's edit over it).

## Rust API

The function lives in
[bundle_adjust.rs](../../../crates/sfmtool-core/src/reconstruction/bundle_adjust.rs),
re-exported as `sfmtool_core::{bundle_adjust, BundleAdjustOptions,
BundleAdjustReport, CameraAdjustment, BundleAdjustError}` and bound as
`EditedReconstruction.bundle_adjust`. `focal_is_releasable` and
`distortion_is_releasable` are bound as functions of the same names in
`sfmtool.reconstruction`
([bundle_adjust.rs](../../../crates/sfmtool-py/src/reconstruction/bundle_adjust.rs)).
`CameraRelease`, `ImagePriorOptions`, `ImagePriorWidths` and `ImagePrior` are
the kernel's types,
[`../../../crates/sfmtool-core/src/geometry/bundle_adjust.rs`](../../../crates/sfmtool-core/src/geometry/bundle_adjust.rs),
re-exported from this module.

```rust
pub fn bundle_adjust(
    recon: &SfmrReconstruction,
    options: &BundleAdjustOptions,
    progress: &Progress<'_>,
) -> Result<(SfmrReconstruction, BundleAdjustReport), BundleAdjustError>;

pub struct BundleAdjustOptions {
    /// One per camera of the image set, or empty for every camera held.
    pub releases: Vec<CameraRelease>,
    pub schedule: Vec<BaSchedule>,
    pub max_iters: usize,
    pub min_track: usize,
    pub min_obs: usize,
    /// Default `FreePointPolicy::CROSS`; `FreePointPolicy::NO_CROSS` opts out.
    pub free_points: FreePointPolicy,
    /// The widths of the prior on a per-image release; derived by default.
    pub image_prior: ImagePriorOptions,
    /// One per camera of the image set, or empty: the scanlines a camera that
    /// releases its rates is solved on. `None` is the stored setting, or
    /// `Unknown` where the camera has none.
    pub rolling_shutter_scanlines: Vec<Option<RollingShutterScanlinesChoice>>,
    /// How the trailing solve of a race under `Unknown` is stopped early.
    pub scanlines_early_stop: ScanlinesEarlyStopOptions,
}

/// The early stop of a scanlines race. The default is on, 100, 3.
pub struct ScanlinesEarlyStopOptions {
    pub enabled: bool,      // off: both solves run to the end, one after the other
    pub gap_threshold: f64, // the gap D the trailing solve must trail by
    pub count: usize,       // consecutive pairs the rule must hold for
}

impl BundleAdjustOptions {
    /// The defaults with `release` for each of `camera_count` cameras.
    pub fn uniform(camera_count: usize, release: CameraRelease) -> Self;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RollingShutterScanlinesChoice {
    Rows,
    Columns,
    Unknown, // solve with each, keep the lower cost; never stored
}

impl RollingShutterScanlinesChoice {
    pub fn name(self) -> &'static str;                  // "ROWS", "COLUMNS", "UNKNOWN"
    pub fn parse(name: &str) -> Result<Self, String>;
    pub fn setting(self) -> Option<RollingShutterScanlines>; // None for Unknown
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CameraRelease {
    pub focal: bool,
    pub distortion: bool,      // k1, or the radial spline
    pub principal_point: bool, // (cx, cy), on any model
    pub image_focal: bool,           // each image's focal scale; moves the focal too
    pub image_principal_point: bool, // each image's offset; moves the principal point too
    pub image_radial_bspline: bool,  // each image's radial correction; SFMTOOL_PINHOLE
    pub image_rolling_shutter: bool, // each image's rolling-shutter rates; SFMTOOL_PINHOLE
}

impl CameraRelease {
    pub const HELD: CameraRelease;                 // nothing
    pub const FOCAL: CameraRelease;                // the focal alone
    pub const FOCAL_AND_DISTORTION: CameraRelease; // the focal and the distortion
    pub const ALL: CameraRelease;                  // all seven
}

pub struct BundleAdjustReport {
    pub images: usize,
    pub points: usize,
    pub observations: usize,
    pub points_deleted: usize,
    pub median_residual_before: f64,
    pub median_residual_after: f64,
    pub cameras: Vec<CameraAdjustment>, // one per camera in the solve
    /// The kernel's storage decision; `None` under `FreePointPolicy::NO_CROSS`.
    pub free_point_decision: Option<FreePointDecision>,
}

pub struct CameraAdjustment {
    pub camera: usize,        // its index among the reconstruction's cameras
    pub images: usize,        // posed images taken through it in the solve
    pub focal_before: f64,
    pub focal_after: f64,
    pub focal_released: bool,      // shared or per image: what its release asked for
    pub distortion_released: bool,
    pub principal_point_released: bool, // shared or per image
    pub image_focal_released: bool,
    pub image_principal_point_released: bool,
    pub image_radial_bspline_released: bool,
    pub image_rolling_shutter_released: bool,
    pub rolling_shutter_scanlines: Option<RollingShutterScanlines>, // the rates' setting; None unless released
    pub scanlines_race: Option<ScanlinesRace>, // under Unknown
    pub principal_point_before: [f64; 2], // [cx, cy], px
    pub principal_point_after: [f64; 2],  // the before unless released
    pub outermost_observed: Option<KeypointReach>, // under the solved camera
    pub image_adjustments: Option<ImageAdjustmentSpread>, // with a per-image release
}

/// The two solves that chose one camera's scanlines.
pub struct ScanlinesRace {
    pub rows_cost: f64,      // in units of sigma_px²: final, or at the stop
    pub columns_cost: f64,
    pub observations: usize, // both final rounds kept, or both rounds at the stop
    pub sigma_px: f64,       // the pooled σ_n, or at the stop √σ̄²
    pub stopped_early: bool, // stop is Some
    pub stop: Option<ScanlinesStop>, // None: both solves ran to the end
}

/// Where the early stop stopped the trailing solve.
pub struct ScanlinesStop {
    pub stopped: RollingShutterScanlines, // the setting not chosen
    pub round: usize,       // counted from 1, as the kernel runs the schedule
    pub iteration: usize,   // within the round, counted from 1
    pub gap: f64,           // D there, in units of sigma_px²
    pub winner_cost: f64,   // the winner's at the end, over its own final round
}

/// How a camera's images' adjustments ended, over its posed images.
pub struct ImageAdjustmentSpread {
    pub images: usize,
    pub focal_mean: f64,                 // of ln(1 + s), read as fractions;
    pub focal_std: f64,                  // the mean zero after the re-centring
    pub principal_point_mean: [f64; 2],  // px
    pub principal_point_std: [f64; 2],   // px
    pub radial_bspline_mean: [f64; 2],   // [e_0, e_1], in units of ρ
    pub radial_bspline_std: [f64; 2],
    pub rolling_shutter_mean: [f64; 3],  // [ωx, ωy, ωz], rad per short side
    pub rolling_shutter_std: [f64; 3],
    pub prior: ImagePriorWidths,         // the final round's; NaN where not released
    pub prior_derived: bool,
    pub spread: ImagePriorWidths,        // the last derivation's, before the clamp
    pub near_prior: Vec<NearPrior>,      // beyond two widths from the mean
}

pub struct NearPrior {
    pub image: usize,                       // its index in the reconstruction
    pub parameter: ImageAdjustmentParameter, // Focal, PrincipalPoint, RadialBspline or RollingShutter
    pub widths: f64,                        // |t − t̄|/σ_t, t = ln(1 + s), ‖a − ā‖/σ_pp, ‖e − ē‖/σ_e or ‖ω − ω̄‖/σ_ω
}

pub enum BundleAdjustError {
    NoKeypoints,
    ReleaseCount { releases: usize, cameras: usize },
    ScanlinesCount { choices: usize, cameras: usize },
    FocalNotReleasable { camera: usize, camera_model: &'static str },
    ImageFocalNotReleasable { camera: usize, camera_model: &'static str },
    ImageRadialBsplineNotReleasable { camera: usize, camera_model: &'static str },
    ImageRollingShutterNotReleasable { camera: usize, camera_model: &'static str },
    DistortionWithoutFocal { camera: usize },
    DistortionNotReleasable { camera: usize, camera_model: &'static str },
    NoPosedImages,
    NoObservations,
    EmptySchedule,
    ScheduleTrim { round: usize, trim_px: f64 },          // round counted from 1
    ScheduleLossScale { round: usize, loss_scale: f64 },
    Constraints(PointConstraintsError),
    Cancelled,
    Degenerate { observations: usize, min_obs: usize },
}

/// The schedule refusals alone: `EmptySchedule`, `ScheduleTrim`,
/// `ScheduleLossScale`, or `Ok`.
pub fn check_schedule(schedule: &[BaSchedule]) -> Result<(), BundleAdjustError>;

/// Whether the kernel's analytic focal column is exact for this camera.
pub fn focal_is_releasable(camera: &CameraIntrinsics) -> bool;

/// Whether this camera has lens distortion the adjustment can release.
pub fn distortion_is_releasable(camera: &CameraIntrinsics) -> bool;

/// Whether each image's radial correction, or its rolling-shutter rates, can
/// be released on this camera.
pub fn image_radial_bspline_is_releasable(camera: &CameraIntrinsics) -> bool;
pub fn image_rolling_shutter_is_releasable(camera: &CameraIntrinsics) -> bool;
```

### Why it is shaped this way

**A value in, a value out.** The adjustment is a function from one
reconstruction to the next, which is what lets a caller keep both and undo by
pointing at the one it had ([`edited-reconstruction.md`](edited-reconstruction.md)).
There is no `&mut` on the input, so a refusal cannot leave a half-adjusted
reconstruction behind, and the caller decides whether the answer replaces
anything.

**One options struct, and a small one.** The kernel takes eighteen arguments
because it is an array kernel with every switch exposed. A caller here is
adjusting a reconstruction, and the only decision that is genuinely theirs is
which lenses move; everything else is the schedule the rest of the toolkit
runs. So `releases` is the first field and the rest are the kernel's own
defaults, stated rather than hidden so a caller that does need to tighten a trim
can.

**The releases are per camera.** Each camera has its own model, and a model
decides what can be released at all, so the choice is made camera by camera: a
rig that mixes an `OPENCV_FISHEYE` camera with spline cameras releases the
spline cameras and holds the other. `releases` holds one `CameraRelease` per
camera of the image set, in camera index order, because the camera index is the name every
other surface gives a camera; an entry for a camera no posed image uses is
ignored, since that camera is not in the solve. An empty list holds every
camera, which is what `Default` gives, and any other length than the camera count is
refused (`ReleaseCount`) rather than read as a prefix, because a list one entry
short would otherwise hold the last camera without saying so.
`BundleAdjustOptions::uniform(n, release)` builds the case where every camera
is released alike, as `BaCameras::shared` builds the one-camera case of the
kernel's camera list.

**A camera with nothing released is held, and a solve with every camera held
still runs.** Holding a lens is a choice, not an error: the solve refines the
poses and the points against the lenses as they are, and the report's entries
say `focal_released: false`, `distortion_released: false` and
`principal_point_released: false`. That is also the
default, because a lens that moves is a different claim about the capture than
a pose that does, and the caller should be the one making it.

**A release a model cannot take is refused, naming the camera.** The kernel
holds a parameter its model is not exact for and releases the rest. That is
right for a kernel with a parity requirement and wrong for a caller who asked a
question: a report saying the focal was released when it was held would be
false. So each camera's release is checked against its own model, and the first
camera that fails refuses the call with its camera index and its model:
`FocalNotReleasable` for a focal release on a model the kernel's analytic focal
column is not exact for, and `DistortionNotReleasable` for a distortion release
on a model with no distortion the kernel can free.
[`focal_is_releasable`] and [`distortion_is_releasable`] are public and per
camera for the same reason, so a caller offering the releases as choices can
grey each choice on its own camera instead of taking it and refusing. The focal
answer depends on the model alone. The list of the five models
(`SIMPLE_PINHOLE`, `EQUIDISTANT_FISHEYE`, `SIMPLE_RADIAL_FISHEYE`,
`SFMTOOL_FISHEYE`, `SFMTOOL_PINHOLE`) is written once, as the crate-private
`focal_is_releasable` in the kernel module
[`geometry/bundle_adjust.rs`](../../../crates/sfmtool-core/src/geometry/bundle_adjust.rs),
because it is a property of that kernel's analytic focal column, not of the
camera model: the kernel's focal gate reads it, and the public
`focal_is_releasable` here delegates to it, so the viewer and the Python
binding's `opt_f` check get the same answer.

**The distortion is released only with the focal, per camera.** The kernel has
two distortion releases, each exact on its own models: `opt_k1` frees `k1` on
`SIMPLE_RADIAL_FISHEYE`, and `opt_bspline` frees the spline on `SFMTOOL_FISHEYE`
and `SFMTOOL_PINHOLE`. No model carries both, so a caller here asks one
question per camera, whether its lens distortion moves, and
`CameraRelease::distortion` reaches both. Neither `k1` nor the spline can
change the scale at the centre of the image, which is the focal's job:
`θ·(1 + k1·θ²)` has slope one on the axis, and the spline's gauge pins its value
and slope there. A distortion released against a held focal could only bend the
periphery around a scale it cannot fix, so a camera whose release has the
distortion without the focal is refused (`DistortionWithoutFocal`, naming it).

**The principal point is released on any model, and needs no focal.** Every
model projects as `u = fx·x_d + cx` with a distorted coordinate that does not
read the principal point, so the kernel's columns for it are exact everywhere,
and it is not a scale, so it is not tied to the focal: a `PINHOLE` or an
`OPENCV_FISHEYE`, whose focal cannot be released, releases its principal point
all the same. It has no refusal of its own. It carries no prior: it is one pair
of numbers fitted from every image of the camera, kept on the image by the
kernel's step guard. On a narrow field it is barely told apart from a turn of
every image ([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)
§ "The principal-point release"), so the report states how far it moved,
`principal_point_before` and `principal_point_after`, for a person to judge.

**Each image's radial correction and rolling-shutter rates are read, and
released per camera.** An image of an `SFMTOOL_PINHOLE` camera may carry both
([../camera/sfmtool-pinhole-kernels.md § Per-image
corrections](../camera/sfmtool-pinhole-kernels.md#per-image-corrections)). The
adjustment passes them to the kernel with each posed image's adjustment, and
each camera's scanlines setting in `BaCameras::rolling_shutter_scanlines`, so
every observation is read through them. A camera that does not release them
holds them, and its images keep the ones they came in with.
`image_radial_bspline` and `image_rolling_shutter` release them
([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)
§ "The per-image corrections"): each image's correction and rates are held
toward the camera's mean by the prior, that mean is held, and the mean radial
correction moves into the camera's spline after every round. Unlike the focal
and the principal point, they hold no shared value of their kind, so neither
moves `focal_released` or `distortion_released`. They are refused on any model
but `SFMTOOL_PINHOLE`, and the radial correction also where the spline domain
end is not positive and finite or the spline has one coefficient
(`ImageRadialBsplineNotReleasable`, `ImageRollingShutterNotReleasable`, each
naming the camera); `image_radial_bspline_is_releasable` and
`image_rolling_shutter_is_releasable` answer per camera, as
`focal_is_releasable` does. The rates are solved on the scanlines
[The rolling-shutter scanlines](#the-rolling-shutter-scanlines) chooses.
[per-image-lens-corrections.md](../../drafts/per-image-lens-corrections.md#choosing-rows-or-columns)
keeps the proposed rule for stopping the losing solve of that choice early.

**Each image's focal and principal point are released per image, and the
camera's value moves to their mean.** `image_focal` and `image_principal_point`
solve each posed image's intrinsic adjustment, held toward the mean of the
camera's images by a prior whose widths the kernel derives from the solve
([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)
§ "The per-image release"). The kernel holds the camera's shared value of the
same kind and re-centres after every round: the camera takes the images' mean
and each image keeps its difference, the focal's as its log focal scale
`ln(1 + s)`, so the camera's focal takes the geometric mean of its images'.
So the shared value moves with a per-image
release, a release of both is the per-image release alone, and the report says
the shared value was released (`focal_released`, `principal_point_released`)
beside the per-image flags. A per-image focal is refused on the models a shared
one is (`ImageFocalNotReleasable`), and stands for the focal a distortion
release needs, since the images' mean scale is the camera's central scale.

**The kernel is handed the choice, not re-told it.** The kernel's eight flags
are requests to every camera; `BaCameras::releases` narrows them per camera
([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)). This
function passes every flag on and each camera's checked release in that list,
so the kernel frees exactly what the report says it did.

**The coefficient count and the domain are not the adjustment's.** How many
coefficients a spline has, and where its domain ends, choose how the lens curve
is parameterized; the solve refines the coefficients it is given and cannot
change either. Changing them is a refit of one camera, a switch of that camera
to its own spline model
([`switch-camera-model.md`](switch-camera-model.md)), after which an
adjustment with its distortion released brings the new coefficients to the
observations.

**A released spline is regularized the way the refit regularizes it.** The
kernel adds the refit's second-difference penalty
([`../camera/refit-camera-intrinsics.md`](../camera/refit-camera-intrinsics.md))
to the coefficients it releases, with the same `SMOOTHING` weight per data row,
so a coefficient near the domain end that few observations reach continues its
neighbours instead of following those few pixels, and one that no observation
reaches is placed by the penalty alone, as the straight-line continuation of
the supported coefficients before it
([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)
§ "The solve"). The residuals this function reports are the data's alone.

**A distortion at its step guard does not stop the poses and points.** When
a released `k1` or spline would fold under a step, the kernel solves that step
again with the camera's distortion held, so the focal, the poses and the
points still move
([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)
§ "Step guards and the held-distortion retry"). A final round that settles
that way is not converged: `report.free_point_decision` carries
`converged: false`, and the kernel's progress names the held camera. Whether
the report carries the held state per camera is an
[open question](#open-questions).

**The report carries each camera's outermost observation.** Measured under the
camera the solve returned, as a radius and an incidence angle, because that is
the camera whose domain the next adjustment would edit; each keypoint is
measured through its image's intrinsics, that camera with the image's
adjustment applied. Only the observations:
the adjustment reads nothing off disk, so the features detected in the images'
`.sift` files are for a caller to read with `outermost_keypoints`.

**The report is per camera.** Each camera in the solve has its own focal and
principal point, so the report carries one `CameraAdjustment` per camera rather
than one of each for the whole solve. Per-camera residual medians are not in it: the
overall medians are the ones a caller reports in one line, and a caller wanting
more runs the kernel.

**The report names three populations.** `images`, `points` and `observations`
are what went into the solve, which is not the whole reconstruction: an unposed
image is not in it, and neither is a point nothing posed observes.
`points_deleted` is what came out short. A caller reporting the edit in one line
needs all four to write a true sentence.

**The errors are an enum.** Every variant is a property of the value or the
options that a caller could have checked, and two of them (`NoKeypoints`,
`NoPosedImages`) are exactly what a menu entry greys itself on. A
`Result<_, String>` would make that gate a string comparison.

### Example

```rust
use sfmtool_core::progress::Progress;
use sfmtool_core::{bundle_adjust, BundleAdjustOptions};

use sfmtool_core::reconstruction::bundle_adjust::CameraRelease;

// A two-camera rig: release camera 1's lens and hold camera 0's.
let options = BundleAdjustOptions {
    releases: vec![CameraRelease::HELD, CameraRelease::FOCAL_AND_DISTORTION],
    ..BundleAdjustOptions::default()
};
let (next, report) = bundle_adjust(&recon, &options, &Progress::none())?;
println!(
    "{} px -> {} px, {} points dropped",
    report.median_residual_before, report.median_residual_after, report.points_deleted,
);
for camera in &report.cameras {
    println!(
        "camera {}: focal {} -> {} over {} images",
        camera.camera, camera.focal_before, camera.focal_after, camera.images
    );
}
```

## What the call does

### 1. The refusals

In order, cheapest first, so a caller greying a menu entry gets the same answers
without solving anything:

- A schedule the kernel cannot run, `check_schedule`: no rounds
  (`EmptySchedule`), a round whose trim is not a positive number of pixels or
  `f64::INFINITY` (`ScheduleTrim`), or one whose loss scale is not a positive
  finite number of pixels (`ScheduleLossScale`), each naming the round counted
  from 1. A NaN fails both: it compares false against everything, so a NaN trim
  would trim every observation. An infinite trim is allowed and leaves out only
  the observations that do not project. `check_schedule` is public so a caller
  that takes a schedule from a person -- the viewer's dialog and its wire tool
  -- refuses the same schedules in the same words before starting anything.
- Observations with no pixel behind them (`NoKeypoints`): a `sift_files` value
  without the format's optional inline keypoint column states no coordinate for
  a sighting, and the residual has nothing to be measured against. The other
  observation mode always carries one.
- No image with a finite pose (`NoPosedImages`).
- A release list whose length is neither zero nor the camera count
  (`ReleaseCount`), and the same of the scanlines list (`ScanlinesCount`).
- Then, for each camera the posed images use, in camera index order, the first of: a
  distortion release without the focal, shared or per image
  (`DistortionWithoutFocal`); a
  focal release on a model the kernel's focal column is not exact for
  (`FocalNotReleasable`); a per-image focal release on such a model
  (`ImageFocalNotReleasable`); and a distortion release on a camera that is neither a
  `SIMPLE_RADIAL_FISHEYE` nor a spline model whose spline is defined, at least
  two coefficients on a positive finite domain end (`DistortionNotReleasable`);
  a per-image radial correction on a camera that is not an `SFMTOOL_PINHOLE`
  whose spline domain end is positive and finite and whose spline is empty or
  has at least two coefficients (`ImageRadialBsplineNotReleasable`); and
  per-image rolling-shutter rates on a camera that is not an `SFMTOOL_PINHOLE`
  (`ImageRollingShutterNotReleasable`). Each names the camera by its camera
  index, and every one but the first its model.
- No observation of any point in a posed image (`NoObservations`).
- Constraint columns stating something the adjustment cannot honour
  (`Constraints`), by the rules of
  [`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md).

The last refusal is the solve's own, in step 5.

### 2. What goes into the solve

Every image with a finite pose, in image index order, and every observation whose image
is one of them. An unposed image is not a camera a residual can be written
against, so it sits the solve out and its observations do not enter; a point left
with no observation in the solve is not in the solve either, and comes back
exactly as it went in.

The cameras of the solve are the image set's cameras the posed images use, in
camera index order, and each posed image's index into that list goes to the kernel as
its image-to-camera column. The posed images may use any number of cameras; each
keeps its own lens in the solve. A camera only unposed images use, or none, is
not in the solve and comes back exactly as it went in.

Every live point goes in at the coordinate it holds, with the representation it
holds: a `w = 0` row is a world-frame direction and enters the kernel's
`point_at_infinity` mask as the representation it starts in. Under the default
`free_points`, `FreePointPolicy::CROSS`, every free point is solved in inverse
depth and stored as the kernel's storage decision says at the end of the solve
([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)
§ "Free points: inverse depth and the storage decision"), so a point can come
back in the other representation: a bearing whose rays carry a depth at the
noise the solve measures comes back a position, and a position whose rays do
not comes back a bearing. Deciding that during the solve rather than in a pass
after it is what lets the poses be fitted with no representation imposed on a
far point. `report.free_point_decision` says at what level the points were
decided, whether the final round converged (rather than stopping on its
iteration budget or settling with a distortion held at its step guard), and how
many crossed each way.
Under `FreePointPolicy::NO_CROSS` the mask is honoured for the **whole** solve, and
re-deciding the representation is left to
[`../../cli/reconstruction/xform/find-points-at-infinity.md`](../../cli/reconstruction/xform/find-points-at-infinity.md)
and `--classify-points-at-infinity`. Ranged and held points keep their
representation either way.

A point's constraint is the one its constraint columns state, converted through
`PointConstraints::from_arrays`, which is the same bridge the Python binding's
own constraint arguments go through. A reference image is re-indexed onto the
solve's image list first, and a point whose reference is not in it comes back
free rather than carrying a distance from a camera the solve does not hold, which
is the rule
[`../../formats/sfmr-file-format.md`](../../formats/sfmr-file-format.md) states
for a consumer that drops an image.

### 3. The runs

The kernel runs twice, or, where a camera's scanlines are unknown, twice per
such camera and once more. The **first** run measures: it runs the kernel over
an **empty** schedule, with the crossing off, which executes no round, decides
nothing and reports the residuals at the state it was handed: the "before"
median is then measured by the same projection the "after" one is, through the
same camera model, rather than by a second spelling of the reprojection in this
module. It reads the value as it holds it, so the "before" median describes the
input value. The rest are the adjustment: one solve, or the races of
[The rolling-shutter scanlines](#the-rolling-shutter-scanlines), whose last
winner is the answer.

Both medians are taken over the finite residuals of the points that survive, so
the two numbers describe one population and their difference is the improvement
on it.

### The rolling-shutter scanlines

A camera that releases its images' rates (`image_rolling_shutter`) needs to
know whether its sensor reads rows or columns, and the rates are solved on one
setting. `BundleAdjustOptions::rolling_shutter_scanlines` chooses it per camera,
as `RollingShutterScanlinesChoice::Rows`, `Columns` or `Unknown`. Where the
caller does not choose (an empty list or a `None` entry), it is the camera's
stored `rolling_shutter_scanlines`, used as stored, or `Unknown` where the
camera has none: an absent setting means the scanlines are not known, and no
setting is assumed. A camera that does not release its rates ignores its entry,
and its images' held rates are read on its stored setting. `Unknown` is a value
of the option only: the setting the rates were solved on is written into the
camera.

`Rows` or `Columns` runs one solve on that setting. A camera whose rates are
solved on another setting than the one they were stored on starts its images'
rates at zero, because a rate read on rows means something else on columns.

`Unknown` runs two solves, one with each setting, and keeps the one with the
lower cost, stopping the trailing one once it cannot plausibly catch up
([Stopping the trailing solve early](#stopping-the-trailing-solve-early)). A
continuous weight between the two readout positions is not
solved for instead, because the wrong choice is a local minimum
([per-image-lens-corrections.md](../../drafts/per-image-lens-corrections.md#alternatives-considered)).
The two are one **race**. Where both solves run to the end, the cost compared
is the robust cost the kernel's
damping ladder compares (the data, the per-image prior with its mean holds,
and the spline penalty), in units of `σ_n²`, read from each solve's
`BundleAdjustment::final_round`
([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)
§ "The per-image corrections"). Each solve's final round trims from its own
state, so the data are summed over the observations both final rounds kept,
each at its own final state, and both costs are divided by one noise level:
the gated RMS of both solves' residuals of finite points there, measured as the
kernel measures a round's noise, never under the keypoint resolution. A tie
between the two computed costs goes to rows; that breaks an exact tie and is
not a setting assumed in place of the race. A solve that exits degenerate costs `+∞`, and a race in which both
do is the call's `Degenerate` refusal.

Several `Unknown` cameras are raced one at a time, in camera index order. While
one camera's race runs, the cameras already raced use the setting their race
chose, and those not yet raced have their rates held at zero, where the setting
has no effect, so no setting is assumed for them; a camera given `Rows` or
`Columns` releases its rates at that setting in every race. The first race
starts from the caller's value. Each later race starts from the state the
previous winner reached, its poses, points, cameras and image adjustments, so
the earlier races' work is carried forward. Both solves of a race start from
that same state, with the raced camera's rates at zero, and the last race's
winner is the answer. `k` such cameras cost `2k` solves; the cases with more
than one camera are rare enough not to justify a search over all `2ᵏ`
combinations. Whether a decided camera's choice would change once the later
cameras' rates are released is not tested
([open question](#open-questions)).

#### Stopping the trailing solve early

`BundleAdjustOptions::scanlines_early_stop`, a `ScanlinesEarlyStopOptions`,
is on by default with a threshold `T` of 100 and a count `K` of 3, the values
[bundle-adjust-measurements.md](bundle-adjust-measurements.md#stopping-the-trailing-solve-of-a-scanlines-race)
recommends. It is a field of the options so that a caller measuring the race
can turn it off and record both solves to the end, as the `scanlines_race`
example does; the viewer and the Python binding run the default. The rule is
`EarlyStopRule` and the two threads are `race`, in
[race.rs](../../../crates/sfmtool-core/src/reconstruction/bundle_adjust/race.rs).

The two solves of a race run on two threads, each through its own `Progress`
(a `Progress::tapped` copy of the race's, with its own cancel flag), in
lockstep per LM iteration: after each of its iterations a solve waits in its
progress sink until the other has finished the same iteration or the round,
and at the first value of each round after the first, its kept count after
its trim, until the other has finished the round before, so neither runs an
iteration of round `r + 1` before both have finished round `r`. The rule reads what the kernel reports
([../geometry/bundle-adjustment.md](../geometry/bundle-adjustment.md)): each
round's kept count (`KEPT_VALUE`), its noise level at its start
(`NOISE_VALUE`) and the cost its damping ladder compares before its first
iteration and after each (`COST_VALUE`). After each pair of iterations it reads
the gap `D = (C_trailing − C_leading)/σ̄²`, `σ̄²` the mean of the two solves'
round-start noise levels squared; a solve that has finished the round keeps
its last cost. The trailing solve is stopped, through its cancel flag, when for
`K` consecutive pairs with the same solve trailing both `D > T` and `D`
exceeds the **budget term**: the mean of the trailing solve's last three
decreases within the round (fewer early in a round) times the iterations it has
left, those left in the round plus `max_iters` for each later round of the
schedule as the kernel runs it (a one-round schedule with a derived prior runs
as two). The pair at a round's start is skipped, neither counting nor breaking
a run; no pair holds while the two rounds kept different counts of
observations, nor once the trailing solve has finished its last round and has
no iteration left to save. These are the definitions
[`scripts/scanlines_race_early_stop.py`](../../../scripts/scanlines_race_early_stop.py)
replays, and the race's tests reproduce the measurement's stops on its
synthetic scenes. The leading solve then runs to the end alone; if the rule
never fires, the final costs decide as above.

The rule is a function of the reported values alone, read pair by pair in
order, so where it fires never depends on the threads' timing. A solve that
is not stopped computes what it would compute alone: being held in its
progress sink and having a cancel flag that nobody sets change nothing in its
arithmetic, so the winner's answer is, to the bit, the one two full solves
give, and a race the rule never stops is two full solves to the bit. The
stopped solve returns the state it reached, which is discarded. Should the
winner go on to exit degenerate, the stopped solve runs again to the end alone
and the final costs decide, as two full solves would have. Several `Unknown`
cameras each run their race under the rule, in the order and from the states
above. With `enabled` off, the two solves run one after the other, each to the
end of its schedule.

`progress` gives the solve phase one share per race. Under the early stop both
solves report over that share, each in a phase named `solve with rows` or
`solve with columns`; the columns solve's events are held until the rows solve
has closed its phase, so that a collector that tracks one open phase per depth
reads each solve's phase whole, rows first. A stopped solve's phase ends at the
round it was stopped in and carries the note `stopped at round 2, iteration 4`.
The solve phase says each race's two costs and the setting kept, and for a
stopped race which solve was stopped, where, the gap and the winner's final
cost. A cancellation of the caller's `progress` stops both solves, and the call
returns `Cancelled`, writing nothing back; with the early stop off it stops the
solve that is running and the other does not start.

The report's camera entry states the setting the rates were solved on
(`rolling_shutter_scanlines`) and, under `Unknown`, the race
(`scanlines_race`). Where both solves ran to the end, `stop` is `None`,
`stopped_early` is `false`, and the costs, the observations and `sigma_px` are
the final comparison's above. Where the rule fired, `stopped_early` is `true`
and `stop` says which solve was stopped, at which round and iteration, and the
gap `D` there; the two costs are each solve's cost at that pair, over the
observations both rounds kept (`observations`, the same count for both), in
units of `σ̄²` (`sigma_px` is `√σ̄²`), so they compare with each other and their
difference is the gap; and `stop.winner_cost` is the winner's cost at the end
of its schedule, measured as the final comparison measures a race with one
solve, over the observations its final round kept and in units of the `σ_n²` of
its own residuals there, which compares with neither cost at the stop.

On the maintainer's `dino_dog_toy_ground_truth_candidate_v012.sfmr` (one
`SFMTOOL_PINHOLE` camera with no stored setting, 85 images, 24931
observations), releasing the focal, the spline, each image's focal and
principal point and both corrections over one `(inf, 1000)` round with the
crossing off, the race chose rows at a cost of 49350 against 52349 for columns
over all 24931 observations at a pooled `σ_n` of 0.146 px, and the training
RMS per component went from 0.2500 px to 0.1443 px, the kernel's figure for
rows, in 28 s for both solves and the measuring run, with both solves run to
the end. Under the default schedule with the crossing on and the same
releases, the early stop stopped columns at round 1, iteration 12, at a gap of
953 units over 24918 observations, and rows ran on to the end; the call took
25.5 s against 38.7 s for two full solves, and the rows solve's cost after
each iteration and the final median residual were the full solves' to the bit.
On the seoul bull switched to `SFMTOOL_PINHOLE`, a near tie of 7.4 units, the
rule did not fire, both trajectories and the final costs were the full solves'
to the bit, and the call took 0.7 s against 1.1 s, the two solves running side
by side.

### 4. Writing the answer back

- **Poses.** Each posed image takes the rotation and translation the solve
  ended with.
- **The cameras.** Each camera in the solve is replaced by the camera the
  kernel returned for it: itself at its solved focal where its release has the
  focal, with its solved `k1` or spline where its release has the distortion,
  at its solved principal point where its release has the principal point, and
  itself unchanged where it was held. Nothing else about any lens moves: every
  other model's distortion stays where it is.
- **Intrinsic adjustments.** Each posed image's adjustment goes to the kernel as
  `BaCameras::image_adjustments`, so every observation, both runs' residuals and
  the stored errors are read through the image's intrinsics, its camera with its
  adjustment applied
  ([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)
  § "Held image adjustments"). A camera without a per-image release holds them,
  and each of its images keeps the adjustment it came in with; a value whose
  adjustments are all zero and that releases nothing per image passes none, and
  the solve is the one without them bit for bit. A camera with a per-image
  release writes back each posed image's solved adjustment of the released
  kinds; each of its unposed images takes the shift its camera's shared value
  took, `a −= (cx_after − cx_before)/m` per axis and
  `1 + s ← (1 + s)·f_before/f_after`, so that its intrinsics are the ones it
  had (the focal's is the shift the kernel's re-centring gave every image of
  the camera, `ln(1 + s) −= ln(f_after/f_before)`, the logs of the geometric
  means it moved the focal by), and `e −= Δ` for the radial correction, with
  `Δ` the mean correction the kernel moved into the spline over the solves the
  answer descends from (`BundleAdjustment::moved_radial_bspline`), so that its
  spline `c + K·e` is the one it had where the shared spline is held. An
  unposed image's rates are kept, or set to zero where the camera's setting
  changed, since they were read on the old one. The camera's
  `per_image_adjustments` gains the released parameters, so a solved zero is
  recorded as solved, and, where the rates were released, its
  `rolling_shutter_scanlines` becomes the setting they were solved on.
- **Positions and representations.** Each point in the solve takes its solved
  coordinate, and `w` follows the kernel's returned representation.
- **The stored error.** Each point's error column becomes the RMS of that
  point's own finite residuals at the final state. The column describes how well
  the point is reconciled with its observations, and after a solve that moved
  everything the stored value would otherwise describe a geometry the value no
  longer holds.
- **Patch frames.** See "The frame follows the depth" below.
- **Everything else per point** -- the bitmap, the normal, the colour, the
  constraint, the track and every keypoint in it -- is the point's own and is
  kept. The adjustment moves geometry; it does not re-fit appearance and it moves
  no observation.

The derived indexes are rebuilt, so the observation offsets, the feature maps and
the infinity-point count describe the value that comes back.

### 5. The points that do not come back

A point in the solve is **deleted** when its coordinate came back non-finite, or
when every one of its observations is invalid at the final state. Both are the
same statement: nothing that saw the point can still be reconciled with it. A
track the trim wears below `min_track` arrives here by the first route, because
the kernel's inter-round re-estimation cannot rebuild a track with fewer than two
usable rays and returns it non-finite.

Deleting is the honest outcome. A point kept at a coordinate no observation
supports is a claim about the scene with nothing behind it, and one kept at a
non-finite coordinate is not a claim at all. The points that survive keep their
order, so the caller reads what happened off
`RowMap::by_scan` ([`edited-reconstruction.md`](edited-reconstruction.md)) over
the input and the output.

Every residual coming back infinite is not a per-point verdict but the kernel's
**degenerate exit**: fewer than `min_obs` observations survived a trim, and the
state passed through untouched. Read as a verdict it would delete the whole point
set, so it is the call's refusal (`Degenerate`) instead, and no value comes back.

### The frame follows the depth

A point's patch frame is two half-vectors: world-space extents on a finite point,
and angular extents tangent to the direction sphere on a bearing. The distance
that converts between them is the **placement distance**, the distance from the
camera-cloud centroid, which is where
`SfmrReconstruction::materialize_points_at_infinity` places a bearing and what a
bench fit multiplies by and divides by when a track crosses
([`../bench/editable-track.md`](../bench/editable-track.md)). Here it is
`ImageSet::placement_scale`, measured from the same reference so the
conversions cancel exactly.

An adjustment moves points in depth without changing what they look like, so a
frame left alone would change the patch's apparent size in every view that sees
it. Each point's frame is therefore multiplied by the ratio of its placement
distances, after over before, measured against the poses each state holds. A
point that stayed where it was scales by one; a point that halved its distance
from the centroid halves its frame; a degenerate distance (a point at the
centroid itself, which subtends no angle) leaves the frame alone rather than
destroying it.

The centroid is the mean of the posed images' camera centres. An unposed image
(a non-finite rotation or translation) has no centre, so it is left out rather
than turning every distance into `NaN`, and an image set with no posed image measures
from the origin. It is taken once per image set, not once per point:
`ImageSet::camera_centroid` gives it and `ImageSet::placement_scale_from`
measures a position from it, so a write-back over every point costs one pass
over the images. `placement_scale` is the same two steps for a caller with one
point to measure. The ratio itself is one public function,
[`patch_frame_factor`](../../../crates/sfmtool-core/src/reconstruction/bundle_adjust.rs),
shared by every edit that moves a point (this write-back, the camera move's and
the retriangulation's) and by a caller that draws a point between two positions
and wants the size the edit will write:

```rust
pub fn patch_frame_factor(
    centroid: &Point3<f64>,      // of the image set the point now stands in
    before: Option<f64>,         // its placement distance, None for a bearing
    after: Option<&Point3<f64>>, // its new position, None for a bearing
) -> Option<f32>;                // None: leave the frame as it is
```

`None` is a degenerate distance on either side or a ratio of exactly one.

## Implementation notes

**The "before" residuals come from the kernel, not from a second projection.**
Running the kernel over an empty schedule is the whole of it: the staged loop
iterates the schedule, so with no rounds it falls straight through to the final
residual pass. That pass is model-aware in ways a re-implementation here would
have to copy exactly -- the ray-path in-front measure, the invalid-observation
sentinel, the direction rows projecting through rotation alone -- and a copy
would drift.

**Points not in the solve are not candidates for deletion.** The deletion rule
reads a point's residuals, and a point no posed image observes has none; treating
"no finite residual" as the verdict without that guard would delete every point
of an unposed corner of the capture rather than leaving it alone.

**The frame rescale reads two different image sets.** The "before" distance is
measured against the input's poses and the "after" against the output's, because
the centroid itself moves when the cameras do. Measuring both against one image set
would leave the frame carrying the gauge drift of the solve.

**The race blocks threads.** Each solve of a race waits in its progress sink
for the other, and the calling thread, which runs the rows solve, then waits
for the columns solve's thread, while the solves' parallel work runs on the
rayon pool. Called from the only worker of a
one-thread rayon pool, the race would deadlock: that worker would wait on
solves whose work waits on it. No caller does: the viewer runs the adjustment
on a `std` thread and the Python binding on the calling thread with the
interpreter released. Each solve also sees the caller's cancel flag only
through its tap, which passes it on at the solve's next event.

**A stop is judged only while the trailer has work left.** The rule does not
count a pair at which the trailing solve has finished its last round, so every
stop is judged while that solve still waits in its sink for the pair, and the
stopped solve's note and `stopped_early` are set before its thread can end.
Counting such pairs would record a stop that saves nothing, at a moment that
depends on which thread reads it. For the same reason no pair is judged once
the caller's cancellation has reached the race, since each solve's trace then
ends wherever the cancellation found it, and a cancelled race notes no stop on
either phase.

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `releases` | empty: every camera held | One `CameraRelease` per camera of the image set: `focal` releases that camera's focal, `distortion` its `k1` or spline and needs `focal` or `image_focal`, `principal_point` its principal point on any model, `image_focal` each image's focal scale where `focal` is accepted, `image_principal_point` each image's principal-point offset on any model, and `image_radial_bspline` and `image_rolling_shutter` each image's radial correction and rolling-shutter rates on `SFMTOOL_PINHOLE`. |
| `schedule` | `DEFAULT_SCHEDULE`, `[(50, 5), (12, 2), (4, 1)]` | The staged trim schedule, `(trim_px, loss_scale)` per round; `trim_px` may be `f64::INFINITY`. |
| `max_iters` | `60` | LM iteration budget per round. |
| `min_track` | `2` | Trim survivors a point needs to stay in a round's solve. |
| `min_obs` | `12` | Trim survivors below which a round exits degenerate. |
| `free_points` | `FreePointPolicy::CROSS` | Whether free points cross between a position and a direction by the storage decision; `NO_CROSS` honours the value's representation. |
| `image_prior` | `ImagePriorOptions::default()`: loose 0.1, 0.1, 0.01, 0.01, floor 0.001, 0.001, 1e-4, 1e-4, derived | The widths of the prior on a per-image release, the focal's of the log focal scale `ln(1 + s)`, read as a fraction, the principal point's in the short-side unit, the radial correction's in units of `ρ` and the rates' in radians per short side: the first round's and the cap, the floor, and `fixed` widths that replace the derivation. |
| `rolling_shutter_scanlines` | empty: each camera's stored setting, or `Unknown` | One `Option<RollingShutterScanlinesChoice>` per camera of the image set: the scanlines a camera that releases its rates is solved on ([The rolling-shutter scanlines](#the-rolling-shutter-scanlines)). |

The defaults after `releases` are the kernel's own
([`../geometry/bundle-adjustment.md`](../geometry/bundle-adjustment.md)), stated
here so a caller sees what it is getting.

## Python bindings

`EditedReconstruction.bundle_adjust(*, opt_f=False, opt_distortion=False,
opt_pp=False, opt_image_f=False, opt_image_pp=False,
opt_image_radial_bspline=False, opt_image_rolling_shutter=False,
image_prior=None, releases=None, schedule=None, max_iters=60, min_track=2,
min_obs=12, free_points_cross=True)` returns `(EditedReconstruction, report)`.
`free_points_cross=False` is `FreePointPolicy::NO_CROSS`. `opt_f`,
`opt_distortion`, `opt_pp`, `opt_image_f`, `opt_image_pp`,
`opt_image_radial_bspline` and `opt_image_rolling_shutter` are the release
given to every camera of the image set, each camera's rates solved on its
default scanlines. `releases`, when given, replaces them: a
list with one dict per camera in the image set, in camera index order, each
`{"focal": bool, "distortion": bool, "principal_point": bool, "image_focal":
bool, "image_principal_point": bool, "image_radial_bspline": bool,
"image_rolling_shutter": bool, "rolling_shutter_scanlines": str}`, a missing
flag meaning `False` and a missing or `None` `rolling_shutter_scanlines` the
camera's default; `"ROWS"`, `"COLUMNS"` and `"UNKNOWN"` are the
`RollingShutterScanlinesChoice` values. An unknown key is a `ValueError` naming
the eight keys an entry takes, and a flag that is not a bool, or a scanlines
value that is not one of the three names or `None`, is one too. `image_prior`
is `ImagePriorOptions` as a dict of `loose`, `floor` and `fixed`, each a dict
of `focal`, `principal_point`, `radial_bspline` and `rolling_shutter` (`fixed`
also `None`), a missing key at its default; an unknown key, or a width that is
not positive and finite, is a `ValueError`. It materialises the version's value when its
overlay is not empty, runs the function over it, and wraps the answer as a new
base with an empty overlay, so the Python surface is the viewer's edit exactly.
The report is the fields above as a dict, `cameras` a list with one dict per
camera in the solve carrying the `CameraAdjustment` fields
(`principal_point_before` and `principal_point_after` each a `(cx, cy)` pair,
`outermost_observed` a dict of `radius_px`, `theta_deg`, `image` and `xy`, or
`None`, `rolling_shutter_scanlines` `"ROWS"`, `"COLUMNS"` or `None`,
`scanlines_race` a dict of `rows_cost`, `columns_cost`, `observations`,
`sigma_px`, `stopped_early` and `stop` (`None`, or a dict of `stopped`,
`"ROWS"` or `"COLUMNS"`, `round`, `iteration`, `gap` and `winner_cost`), or
`None`, and `image_adjustments` the
`ImageAdjustmentSpread` as a dict, `None` without a per-image release: its
pairs as `(x, y)`, its radial correction's as `(e_0, e_1)` and its rates' as
`(wx, wy, wz)`, `prior` and `spread` as dicts of `focal`, `principal_point`,
`radial_bspline` and `rolling_shutter`, and `near_prior` a list of dicts of
`image`, `parameter` (`"focal"`, `"principal_point"`, `"radial_bspline"` or
`"rolling_shutter"`) and `widths`),
`free_point_decision`
the dict the kernel binding reports (`sigma_px`, `observation_count`,
`outlier_count`, `decided`, `converged`, `to_finite`, `to_direction`,
`unscored`) or
`None` with the crossing off, and every refusal is a
`ValueError` carrying the sentence the error writes.

```python
adjusted, report = value.bundle_adjust(opt_f=True)
# A two-camera rig: camera 0 held, camera 1's focal and spline released.
adjusted, report = value.bundle_adjust(
    releases=[{}, {"focal": True, "distortion": True}]
)
print(report["median_residual_before"], "->", report["median_residual_after"])
for camera in report["cameras"]:
    print(camera["camera"], camera["focal_before"], "->", camera["focal_after"])
```

`focal_is_releasable(camera)` and `distortion_is_releasable(camera)` in
`sfmtool.reconstruction` take a `CameraIntrinsics` and return the core
functions' answers, so a caller can build `releases` without asking for a
release the call would refuse. They are module functions beside the
adjustment rather than properties of `CameraIntrinsics`, because what the
adjustment can release is a fact about the adjustment, not about the camera.

```python
from sfmtool.reconstruction import distortion_is_releasable, focal_is_releasable

releases = [
    {"focal": focal_is_releasable(c), "distortion": distortion_is_releasable(c)}
    for c in recon.cameras
]
```

## Testing

`crates/sfmtool-core/src/reconstruction/bundle_adjust/tests.rs` builds cameras on
a shallow arc looking at a cloud of points, with every observation at the exact
projection, so a perturbed copy of it has a known place to converge to; the
fixture needs no pixels, because the adjustment reads none. What it pins:

- The poses and the points converging, with three points held at the truth to
  pin the gauge the solve is otherwise free in, and the residual median coming
  down.
- The input value untouched.
- A released focal found from 6 % off, reported, and written into the camera.
- The same with every image carrying its own intrinsic adjustment and its
  keypoints made through it: the solve holds the adjustments, its residual
  median ends under `10⁻³` px and every adjustment comes back as it went in,
  while the same pixels with the adjustments dropped end with a median over
  0.05 px.
- A value whose images are taken through two cameras adjusted rather than
  refused, each image read through its own lens: the poses converge, and each
  camera's report entry counts its images and holds its focal.
- A released spline moving toward a planted one with the focal, and a released
  `k1` on a `SIMPLE_RADIAL_FISHEYE` doing the same, each with the residual median
  falling by an order of magnitude and the report marking the camera's
  distortion released; a spline camera, a `k1` camera and a pinhole in one solve,
  the first two releasing their distortion and the third its focal alone; a
  spline camera released beside a pinhole held, which comes back unchanged.
- A rig of an `OPENCV_FISHEYE` camera and a spline camera, the first held and
  coming back unchanged while the spline moves, and a focal release on the
  `OPENCV_FISHEYE` refused naming it.
- Each per-camera refusal naming its camera in a one-line sentence: a distortion
  release without the focal, a distortion release on a `SIMPLE_PINHOLE`, a focal
  release on a `PINHOLE`.
- Every camera held, by an empty list and by an explicit one, giving the same
  solve bit for bit, the residual median falling and no lens moving.
- A release list one camera short and one camera long, refused.
- Each camera's outermost observation one of its own images', its radius
  measured from its principal point.
- The focal released on two cameras finding each its own planted focal and
  moving nothing else about either lens.
- A released principal point, planted 9 px off, moving more than halfway back,
  written into the camera, reported before and after, and nothing else about
  the lens moving; a held one reported unchanged and coming back exactly; two
  cameras, one releasing its principal point alone and one with its focal,
  each moving back along its own planted axis to within 3 px.
- The per-image release, on the fixture with every image's keypoints made
  through its own adjustment and handed in with none, image 5 unposed with an
  adjustment of its own: each posed image written back with a solved
  adjustment, the five averaging zero to `10⁻¹⁴` (the focal in `ln(1 + s)`), the camera's
  `per_image_adjustments` marking both parameters, the unposed image's
  intrinsics unchanged to `10⁻¹²` while its camera moved, the residual median
  halved, and the report's flags, spread (its standard deviation in pixels the
  adjustments' own, and the focal's that of `ln(1 + s)`), prior and derivation; under fixed widths a fifth of the
  adjustments, the images beyond two widths each at its own distance from the
  mean and the unposed image not among them, and none under wide ones; the
  principal point released per image alone keeping the focal, every focal
  scale and the focal flag, and reporting no focal width; a per-image focal on
  an `OPENCV_FISHEYE` refused naming it; a distortion beside a per-image focal
  accepted; and a solve with no per-image release reporting no spread and
  marking no camera.
- The per-image corrections
  ([corrections.rs](../../../crates/sfmtool-core/src/reconstruction/bundle_adjust/tests/corrections.rs)),
  on a scene of 30 `SFMTOOL_PINHOLE` images from a general motion, each image's
  keypoints made through its own planted radial correction and rates with
  0.1 px of noise: with the rates planted on rows and on columns and no stored
  setting, the fallback to `Unknown` racing both and choosing the planted
  scanlines at the lower cost, writing them into the camera, marking both
  names, recovering the rates and naming the comparison on `progress`; a
  choice of `Rows` run as one solve on rows whatever the camera stores, a
  stored `Columns` used with no choice, and an explicit `Unknown` racing beside
  a stored setting; two `Unknown` cameras raced in camera index order, the
  answer equal to the bit to two adjustments in sequence (the first racing
  camera 0 with camera 1's rates held, the second started from its answer), so
  each race starts from the previous winner's state; an unposed image's spline
  unchanged to `10⁻¹²` while the mean correction moved into the camera's, and
  the report's means, standard deviations and widths of both corrections; an
  unposed image's rates kept when its camera's stored `Columns` is solved on
  and set to zero when the rates are solved on `Rows` instead;
  under narrow fixed widths, `near_prior` naming both corrections at their own
  distances; the refusals on a `SIMPLE_PINHOLE`, an `SFMTOOL_FISHEYE` and an
  `SFMTOOL_PINHOLE` whose domain end is zero or infinite, and of a scanlines
  list one camera short; a scanlines choice on a camera that releases no rates
  changing no bit of the answer; and a cancellation stopping both solves of a
  race, and with the early stop off ending the call before the second solve
  opens.
- The early stop (`tests/corrections.rs`): on the planted scenes the rule
  stops the solve on the other scanlines where the measurement's replay did
  (columns at round 2, iteration 4 with rows planted, rows at round 2,
  iteration 2 with columns planted), the costs at the stop differing by the
  gap and the stopped phase's note saying where; the stopped race's answer
  equal to the bit to two full solves' and to one solve on the winner's
  setting, the reports differing in the race alone; a race the rule never
  stops (an infinite threshold) equal to two full solves to the bit, report
  included, with the columns solve's phase reported after the rows solve's;
  repeated runs giving the same answer and stop to the bit; two `Unknown`
  cameras equal to the bit to two full solves per race. In `race.rs`, the rule
  on scripted trajectories: the count, the threshold, the budget term with its
  later rounds, a change of trailing solve restarting the count, a round's
  start skipped, no stop under different kept counts, a finished solve keeping
  its last cost, the gap's units, the rate's window, and the same stop
  whichever solve reports first.
- A camera no posed image uses (one nothing references, one only an unposed
  image uses) coming back untouched, left out of the report, and its entry in
  the release list ignored even where its model could not take it.
- A point at infinity coming back a unit direction, and the infinity count with
  it, with the crossing off; and by default the same bearing, whose rays carry
  plenty of parallax, coming back finite where its truth sits among the other
  points, with the decision converged and reporting one point to finite.
- The progress messages: the size of the problem, the two medians, the
  schedule, and the storage decision's line between them.
- A held point coming back at exactly the coordinate it went in at, with its
  constraint.
- A point worn below `min_track` deleted, and `RowMap::by_scan` reporting the
  deletion and the survivors closing up.
- A patch frame scaling by the same ratio its placement distance did.
- Every refusal, by its own variant.

Bindings (`tests/rust_bindings/reconstruction/test_edited_reconstruction_rust_bindings.py`): the
call's shape over a real reconstruction -- the poses moving, the residual median
not getting worse, the report's populations and its per-camera entries, the value
that came back being a new base with no overlay, and the object it came from
untouched -- the no-keypoints refusal, a released principal point reported
and written back, a principal point released on an `OPENCV_FISHEYE`, the
per-image focal refused on the fixture's `SIMPLE_RADIAL` and released on a
`SFMTOOL_PINHOLE` switch of it, with both per-image releases written back
averaging zero and reported, the per-image principal point released alone
under a fixed prior, a malformed `image_prior` refused, the unknown-key
refusal naming the eight keys, both corrections refused on the fixture's
`SIMPLE_RADIAL` and released on its `SFMTOOL_PINHOLE` switch with the
scanlines raced, written back and reported, a `"COLUMNS"` choice run as one
solve under a fixed corrections prior and then used as stored, an explicit
`"UNKNOWN"` racing again, and a misspelt or mistyped scanlines value refused.

## Non-goals

- Releasing the distortion of the multi-coefficient models (`RADIAL`,
  `OPENCV`, `OPENCV_FISHEYE`, …). The kernel has no release exact for them; switch
  the camera to a spline model first
  ([`switch-camera-model.md`](switch-camera-model.md)).
- Finding new points at infinity. The storage decision re-decides the
  representation of the points the value holds; appending bearings for tracks
  the value does not hold is
  [`../../cli/reconstruction/xform/find-points-at-infinity.md`](../../cli/reconstruction/xform/find-points-at-infinity.md).
- Adjusting a rig as a rig. Every image keeps its own free pose, whichever
  camera took it.
- Adding, removing or moving observations. The adjustment reads the track it is
  given.
- Running in the background. The function is synchronous, and a caller that
  needs a responsive window runs it where a stall is acceptable.

## Open questions

- **The CLI.** `sfm xform --bundle-adjust` releases neither the shared principal
  point nor the per-image releases, and has no scanlines choice. Whether it
  gains keys for them is open.
- **Several `Unknown` cameras.** Each race is decided with the later cameras'
  rates held at zero, which their poses absorb in part. Whether releasing them
  could change an earlier choice, and whether a second pass over the cameras is
  worth its cost, is untested: every measured capture has one camera.
- **Reporting the guard per camera.** When the final round settles with a
  camera's distortion held at its step guard, the kernel's progress line names
  the camera, and `report.free_point_decision.converged` is `false`. Whether
  `BundleAdjustment` and this function's report also carry a per-camera flag,
  so the viewer's report can say "the spline of camera 0 ended at its
  monotonicity limit", is open.
- **An end-to-end fixture of a weakly supported spline tail.** The
  `dino_dog_toy` reconstruction behind the kernel's tail measurements
  ([`../geometry/bundle-adjustment-measurements.md`](../geometry/bundle-adjustment-measurements.md#a-weakly-supported-spline-tail))
  is not checked in, and the kernel's regression test of the case is
  synthetic. Whether to check in a 739 KB minimal `.sfmr` of it, for an
  end-to-end test at this level or through the Python binding, is a
  maintainer decision.
