# Switching the cameras of a reconstruction to another model

A camera model is the function that maps a ray leaving the camera to a pixel.
Switching a camera of a reconstruction to another model replaces that function
with one from a different family, fitted to the old one where the old one is
trusted, and leaves everything else as it is: poses, points, the measured
keypoints, patches and tracks. What changes is how rays map to those keypoints,
so reprojection errors change, and the operation reports by how much, measured
on one fixed set of observations. The main use is moving a fisheye camera from a
COLMAP polynomial model, whose inverse fails a little past 90° off the axis, to
`SFMTOOL_FISHEYE`, whose spline and linear tail are defined out to 180°, so that
a later bundle adjustment can refine the lens out to the image circle. The same
operation, asked to switch a spline camera to its own spline model, gives it a
new coefficient count or a new domain end: that is how a spline camera's
parameterization changes, since bundle adjustment only refines the coefficients
it has.

The lens fit itself is [`../camera/refit-camera-intrinsics.md`](../camera/refit-camera-intrinsics.md). This spec is
the layer that adds the reconstruction: which observations belong to a camera,
how the fit's largest angle is chosen from them, what is written back, and the
before-and-after comparison.

## Rust API

The function lives in
[switch_camera_model.rs](../../../crates/sfmtool-core/src/reconstruction/switch_camera_model.rs),
as `sfmtool_core::reconstruction::switch_camera_model`, and is bound as
`EditedReconstruction.switch_camera_model` and
`SfmrReconstruction.switch_camera_model`. `sfm xform --camera-model`, the
viewer's Switch Camera Model edit
([`../../gui/edits/switch-camera-model.md`](../../gui/edits/switch-camera-model.md)) and the MCP
tool `switch_camera_model` reach it; the viewer reaches it from the **Switch
Model…** proposal's Apply.

```rust
pub fn switch_camera_model(
    recon: &SfmrReconstruction,
    cameras: &[usize],
    target: &RefitTarget,
    options: &SwitchCameraModelOptions,
) -> Result<(SfmrReconstruction, SwitchCameraModelReport), SwitchCameraModelError>;

pub fn fit_camera_switch(
    recon: &SfmrReconstruction,
    camera: usize,
    target: &RefitTarget,
    options: &RefitOptions,
) -> Result<CameraIntrinsicsRefit, SwitchCameraModelError>;

pub fn default_fit_angle(
    recon: &SfmrReconstruction,
    camera: usize,
) -> Result<(f64, ThetaFitSource), SwitchCameraModelError>;

pub fn refit_radial_corrections(
    recon: &SfmrReconstruction,
    camera: usize,
    refit: &CameraIntrinsicsRefit, // fit_camera_switch's
) -> Result<Option<RefittedRadialCorrections>, SwitchCameraModelError>;

pub fn varied_image_adjustments(image_set: &ImageSet, camera: usize) -> Vec<ImageAdjustmentParameter>;

pub fn image_adjustments_carried(
    recon: &SfmrReconstruction,
    camera: usize,
    target: &RefitTarget,
    options: &RefitOptions,
) -> Result<Vec<CarriedImageAdjustment>, SwitchCameraModelError>;

pub struct CarriedImageAdjustment {
    pub parameter: ImageAdjustmentParameter,
    pub carried: bool,
    pub refitted: bool, // carried by refitting each image's value to a moved domain end
}

pub struct RefittedRadialCorrections {
    pub images: Vec<(usize, RadialCorrectionRefit)>, // by image index, ascending
    pub rms_px: f64,                                 // over every image's samples
    pub max_px: f64,
    pub max_image: usize,
    pub monotone_constrained_images: usize,
}

pub struct SwitchCameraModelOptions {
    pub drop_image_adjustments: Vec<ImageAdjustmentParameter>, // default empty: every image keeps its own
    pub refit: RefitOptions,       // the lens fit's: theta_fit_deg, spline_domain_deg
    pub outermost_keypoint: bool,  // default true
}

pub struct SwitchCameraModelReport {
    pub cameras: Vec<CameraSwitch>, // one per switched camera, in camera index order
}

pub struct CameraSwitch {
    pub camera: usize,
    pub source: CameraIntrinsics,              // before any drop
    pub dropped_image_adjustments: Vec<ImageAdjustmentParameter>,
    pub refit: CameraIntrinsicsRefit,          // carries the new camera
    pub radial_corrections: Option<RefittedRadialCorrections>, // None: none refitted
    pub images: usize,
    pub observations: ObservationComparison,
    pub outermost: Option<OutermostKeypoints>, // under the switched camera; None when left out
}

pub struct ObservationComparison {
    pub observations: usize,         // the fixed set
    pub unmeasured: usize,
    pub max_theta_deg: f64,
    pub before: ErrorSummary,
    pub after: ErrorSummary,
    pub changed_over_1px: usize,
    pub trusted_deg: Option<f64>,
    pub past_trusted: usize,
    pub past_trusted_before: ErrorSummary,
    pub past_trusted_after: ErrorSummary,
}

pub struct ErrorSummary { pub median_px: f64, pub p90_px: f64, pub max_px: f64 }

pub enum SwitchCameraModelError {
    NoCameras,
    UnknownCamera { camera: usize, count: usize },
    Refit { camera: usize, error: RefitError },
    Observations(String),
    LensCorrections { camera: usize, target: &'static str, kept: Vec<ImageAdjustmentParameter> },
    Drop(DropImageAdjustmentsError),
    RadialCorrection { camera: usize, image: usize, error: RefitError },
}
```

### Why it is shaped this way

**A value in, a value out**, like every bulk edit
([`bundle-adjust.md`](bundle-adjust.md), [`move-camera.md`](move-camera.md)): the
input is never written, so a refusal leaves nothing half-switched, and a caller
keeps both values to compare or undo. The viewer's edit and the Python binding
reach the same function with the same value.

**The fit is separate from the switch.** [`refit_camera_intrinsics`](../camera/refit-camera-intrinsics.md)
takes only the lens, so a caller comparing targets or coefficient counts can fit
many times without touching the reconstruction. This function adds only what
needs the observations.

**The fit a switch makes is callable on its own.** `fit_camera_switch` is the
fit `switch_camera_model` makes for one camera, with the same defaults and
refusals, and returns the fit's report without building the switched value or
reading any pixel. It reads the poses and the point positions, one pass over
the observations, because the fit angle of a source without a trusted bound
and the refusal of a perspective target both come from them. The viewer's
camera model proposal calls it on every change of a control, so the camera it
shows is the one Apply installs for the same request. `default_fit_angle` is
the angle such a fit takes when none is given, and where it comes from
(§ "The fit's largest angle"), reported whatever the target and whether or not
the fit at that angle succeeds, so the proposal shows it beside a refused fit.

**A refusal names the camera.** Every camera is fitted before anything is
written, and before any `.sift` file is read for the observations' pixels, and
the first refusal refuses the whole switch with the camera's index and the
fit's own reason.

**The drops are an option of the switch, not a step before it.** A caller
asking for a new model often wants some per-image parameters gone with it, and
has to have them gone where the target cannot carry them (§ "What a switch
carries"). With the drops in `SwitchCameraModelOptions`, one call drops and
fits, and its comparison measures `before` on the value as it was handed in, so
it says what the drops and the switch cost together; a caller that dropped
first with [`drop_image_adjustments`](drop-image-adjustments.md) and switched
the result would see only the switch's share. One list serves every camera
named, and a parameter a camera does not vary is passed over, since the
parameters are the same four words for every camera. `fit_camera_switch` fits
the camera as the value holds it, so a caller showing the fit of a request
with drops, as the viewer's proposal does, calls `drop_image_adjustments` and
hands it the value that returns. Each entry's `dropped_image_adjustments` is
what that camera dropped, for a caller's label.

**Which drops a target forces is callable on its own.**
`image_adjustments_carried` lists each per-image parameter a camera varies
(`varied_image_adjustments`: its flag, or a non-zero value on one of its
images, the parameters a drop changes something for), whether a switch to a
target with given fit options carries it, and whether it carries it by
refitting it. The viewer's proposal reads it to tick and grey the drops its
target forces, and to say which correction it refits, before it fits anything.

**The images' radial corrections are refitted beside the camera's fit, by a
call of their own.** `fit_camera_switch` fits the shared camera, as before;
`refit_radial_corrections` takes that fit and refits each image's correction
to it (§ "Refitting the images' radial corrections"), without building the
switched value or reading any pixel. `switch_camera_model` calls both, and the
viewer's proposal calls both on every change of a control, so it shows the
images' refit the switch would install.

**A spline camera switched to its own model is a refit, in the same call.** A
new coefficient count or domain end is a change of how the lens curve is
parameterized, and is a choice about one camera, like a change of model. One
operation covers both, so every surface that switches a model also refits a
spline, and the report is the same shape either way.

### Example

```rust
use sfmtool_core::camera::refit_intrinsics::RefitTarget;
use sfmtool_core::reconstruction::switch_camera_model::{
    switch_camera_model, SwitchCameraModelOptions,
};

let target = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(8))?;
let options = SwitchCameraModelOptions::default();
let (switched, report) = switch_camera_model(&recon, &[0, 1], &target, &options)?;
for camera in &report.cameras {
    let o = &camera.observations;
    println!(
        "camera {}: median {:.3} -> {:.3} px, {} past the trusted bound",
        camera.camera, o.before.median_px, o.after.median_px, o.past_trusted,
    );
}
```

## What the call does

### The drops

Before any fit, each camera named drops the parameters of
`SwitchCameraModelOptions::drop_image_adjustments` it varies, by
`drop_image_adjustments`: its flag for each is cleared, its images' values are
set to zero, and their mean is folded into the camera (the geometric mean for
the focal, linear for the principal point and the radial correction; the
rolling-shutter rates fold nothing and the scanlines setting is kept). The
target is then fitted to the folded camera, so the fit's `rms_px` and `max_px`
are the new camera's distance from the folded one, and the value written back
is the dropped one with its cameras replaced. A refused drop refuses the switch
(`Drop`), before any fit. With the list empty, as by default, nothing is
dropped and the switch is the one it was before the option existed.

### What a switch carries

An image keeps whatever per-image parameter it is not asked to drop, on top of
the switched camera. The focal scale and the principal-point offset are relative
to whatever the camera becomes, so every switch carries them. The other two are
not:

| Parameter | Carried by |
|-----------|------------|
| `rolling_shutter` | an `SFMTOOL_PINHOLE` target, the one model the rates are defined on ([../camera/sfmtool-pinhole-kernels.md § Per-image corrections](../camera/sfmtool-pinhole-kernels.md#per-image-corrections)) |
| `radial_bspline` | an `SFMTOOL_PINHOLE` target: as it is where the spline's domain end stays put, refitted to the new domain where the switch moves it |

The radial correction is a curve on the camera's own domain, `τ = ρ/ρ_max`,
added to the camera's spline, so only an `SFMTOOL_PINHOLE` camera has it. A
switch to `SFMTOOL_PINHOLE` (whose source, since it has corrections, is an
`SFMTOOL_PINHOLE` too) keeps each image's correction as it is, bit for bit,
where `ρ_max` stays the same number: a refit at another coefficient count
keeps the domain end bit for bit, and so does a given domain end equal to it.
A given domain end that differs, or a fit over a given angle (whose domain end
is the far image corner), moves it, and each image's correction is then
refitted to the new domain (below), so it is carried either way;
`CarriedImageAdjustment::refitted` says which. `fit_camera_switch` refuses a
camera that keeps a parameter its switch does not carry, with
`SwitchCameraModelError::LensCorrections` naming the camera, the target and
the parameters kept, before it fits anything; `switch_camera_model` checks
after its drops, so dropping them lifts the refusal. A parameter counts as
kept where the camera lists it or any of its images holds a non-zero value of
it.

### Refitting the images' radial corrections

Where a switch of an `SFMTOOL_PINHOLE` camera to `SFMTOOL_PINHOLE` moves
`ρ_max`, the two coefficients each image holds would describe another curve on
the new domain. So after the camera's fit, `refit_radial_corrections` refits
each image's correction the way the camera's curve was refitted: the new
camera's spline plus the image's new correction is fitted to the old camera's
spline plus its old correction, linear tail included, over the angles the
camera's fit sampled (the whole new domain for a refit of the spline as
itself, `θ_fit` for a fit over a given angle), with the same weights and under
the same monotonicity floor on the same grid. The correction is linear in its
two coefficients for a fixed camera, so each image is one small least-squares
solve, constrained where the floor binds; the math is
[`../camera/refit-camera-intrinsics.md` § An image's radial correction on a new
domain](../camera/refit-camera-intrinsics.md#an-images-radial-correction-on-a-new-domain).
An image whose old curve is flatter than the floor is held at the floor, as
the camera's fit holds the camera, and the report says on how many images
that happened; a curve that still fails the monotonicity check refuses the
switch (`RadialCorrection`, naming the camera and the image), which only a
failed solve can give.

Every image of the camera is refitted where the camera lists the radial
correction, a zero correction included, since the camera's own refit leaves a
residual the image's curve takes up; where the camera does not list it, only
the images that hold a non-zero one are, and a zero stays zero. The focal
scale, the principal-point offset and the rates are not refitted: they are
relative to whatever the camera becomes. The refit is a fraction of a
millisecond per image, so it takes no `Progress`.

The entry's `radial_corrections` reports it: each image's new coefficients
with the rms and largest pixel distance between its curve before and after
over the fit's samples, and over all of them the rms (`rms_px`), the largest
(`max_px`) and its image (`max_image`), and `monotone_constrained_images`.
These sit beside the camera fit's `rms_px` and `max_px`, and say what the
conversion cost each image's own lens. On the seoul_bull ground truth
(`seoul_bull_sculpture_ground_truth_v02.sfmr`, 17 images, domain end 37.8°):

| Domain end | Camera fit rms / max | Images' refit rms / max | Median error before → after | p90 before → after | Median, p90 after with the correction dropped |
|---|---|---|---|---|---|
| 32.8° | 0.016 / 0.057 px | 0.016 / 0.057 px | 0.101 → 0.105 px | 0.275 → 0.288 px | 0.231, 0.538 px |
| 42.8° | 0.147 / 0.426 px | 0.149 / 0.696 px | 0.101 → 0.149 px | 0.275 → 0.322 px | 0.222, 0.574 px |

The images' curves move about as far as the camera's: what a two-coefficient
correction cannot follow on the new domain is what the camera's spline cannot
follow either. Dropping the corrections instead costs about twice the median
error.

### The observations of a camera

A camera's observations are the track rows of every image that uses it, posed
or not. Each has an **incidence angle**: the angle between the camera's axis
(`−Z`) and its point's direction in the camera frame, from the pose and the
point alone. It does not depend on either model, so it is the same number before
and after the switch; for a point at infinity it is the bearing's angle.

Each row's pixel is its inline keypoint where the value carries the column, and
otherwise the position the image's `.sift` file holds at the row's feature
index, each file read once. A value with neither is refused
(`Observations`), since the comparison is the point of the report.

### The fit's largest angle

A spline camera switched to its own spline model is refitted as described in
"A spline refitted as itself" below. For every other camera, with no
`theta_fit_deg` given, each camera's fit reaches:

1. its trusted bound, where the model has one;
2. otherwise the largest incidence angle among its observations;
3. otherwise, for a camera with no observations, its far image corner.

A caller's value is used as given, and the fit refuses one past the trusted
bound. A perspective target is refused for a camera one of whose points lies
90° or more off its axis (`ObservationsPast90`), whatever `θ_fit` is, because
those observations would have no pixel under it. The angle is the one rule 2
reads: between the camera's axis and the ray to the point's triangulated
position, from the pose alone, not through the lens to the keypoint. A badly
placed point can therefore reach far past any keypoint the camera detected, so
the refusal names the point and says which angle it is: *"camera 1: a
perspective model has no pixel for a ray at 90° or more, and the ray to point
5121 is 144.91° off the camera's axis, measured from the point's position
rather than through the lens"*. The point is an index into the value the
switch was given; the viewer, which switches a materialised value when the
version carries an overlay, names it by the version's own index instead.

### A spline refitted as itself

An `SFMTOOL_FISHEYE` camera switched to `SFMTOOL_FISHEYE`, or an
`SFMTOOL_PINHOLE` switched to `SFMTOOL_PINHOLE`, with no `theta_fit_deg` given,
is fitted by `refit_spline`
([`../camera/refit-camera-intrinsics.md`](../camera/refit-camera-intrinsics.md))
at the target's coefficient count: over the **whole** new domain rather than
the observations' extent, because a spline states its curve at every angle of
its domain and along its linear tail, and constrained to stay monotone. The
domain end is the `spline_domain_deg` given, or the camera's own, kept bit for
bit, when none is; so a caller changing only the count keeps the domain, and
one changing only the domain names the count the camera has. The domain may be
shorter or longer than the old one.

The fit's report says what the refit did: `theta_fit_source` is
`spline_domain`, `rms_px` and `max_px` are the refitted camera's distance from
the old one over the whole domain, and `monotone_constraint` gives the range of
incidence angles where the monotonicity constraint held the slope at its floor,
which is where the refit departs from the old curve. An `SFMTOOL_PINHOLE`
refit given a new domain end also refits its images' radial corrections to it
(§ "Refitting the images' radial corrections"), and the entry's
`radial_corrections` reports that beside the fit. The observation
comparison is taken as for any switch. A domain end the model cannot have (past
180°, or at 90° or more for `SFMTOOL_PINHOLE`) is refused, naming the camera; a count
outside 0 and 2 to 32 is refused by `RefitTarget::from_name`, before any camera.

With a `theta_fit_deg` given, the camera is fitted like any other source, over
that angle, with the domain end at the far image corner unless one is given.
Switching a spline camera to the *other* spline model is a switch between
models, and is fitted like any other source too.

### What is written back

Each switched camera is replaced by its fitted camera. Nothing else in the image
set, the point set or the observation columns changes beyond the drops and the
refitted radial corrections: each image keeps what it was not asked to drop of
its adjustment as it is, since it is relative to whatever its camera is,
except a radial correction on a domain end the switch moved, which takes its
refit. Every
reprojection error below, before and after, is read through each image's
intrinsics, its camera with its adjustment applied: before through the value
handed in, after through the dropped and switched one. The stored error of
every point observed in an image that uses a switched camera is recomputed as
the root mean square of its reprojection errors over all its observations, the convention
`SfmrReconstruction::recompute_point_errors` writes; a point with none that
project gets `0.0`. Points only other cameras see keep their stored error.

### The comparison

For each camera, the **fixed set** is its observations with a finite
reprojection error under both models. Every figure is over that set, so the
change it shows is the lens and nothing else: `before` and `after` are the
median, 90th percentile (linear interpolation, both from
[`numeric`](../../STANDARDS.md#medians-and-quantiles-come-from-one-module)) and maximum error;
`changed_over_1px` counts observations whose error moved by more than a pixel;
and `past_trusted` and its two summaries cover the observations whose incidence
angle is past the source's trusted bound, which are the observations the switch
is for. `unmeasured` counts the rest: an unposed image, a point behind the
camera, a row with no pixel.

### The outermost keypoint

Each entry also carries the camera's outermost keypoint
([`outermost-keypoint.md`](outermost-keypoint.md)): among its images'
observations, and among every feature detected in their `.sift` files where
those can be read, each as a radius from the principal point and an incidence
angle under the **switched** camera, the model whose spline domain the fit just
placed. It is reported beside that domain so a person can see how far out the
photographs reach; the domain itself still defaults to the far image corner. A
`.sift` file that cannot be read leaves the detected keypoint empty and does not
refuse the switch.

`SwitchCameraModelOptions::outermost_keypoint` is on by default. Off, each
entry's `outermost` is `None` and no `.sift` file is read for it, so a caller
that already holds the keypoint, or does not show it, does not pay for a read
of every image's features. The viewer's switch turns it off, and so does the
switch its camera model proposal's job runs for the observation comparison and
the preview
([`../../gui/edits/switch-camera-model.md`](../../gui/edits/switch-camera-model.md)
§ "The proposal's job"), since the proposal reads the keypoint when it opens. The observations' pixels are still
read from the `.sift` files of a value without the inline keypoint column,
since the comparison and the points' stored errors are measured against them.

Comparing on a fixed set is deliberate. A following bundle adjustment or Add
Image to Tracks changes the set, and a metric over a moving set cannot separate
the lens from the population.

## Python bindings

`EditedReconstruction.switch_camera_model(camera_model, *, cameras=None,
coeff_count=None, theta_fit_deg=None, spline_domain_deg=None)` returns
`(EditedReconstruction, report)`, a new base with no overlay; `cameras=None`
switches every camera. `SfmrReconstruction.switch_camera_model` takes the same
arguments and returns `(SfmrReconstruction, report)`; it is what `sfm xform
--camera-model` calls
([`../../cli/reconstruction/xform/xform-command.md`](../../cli/reconstruction/xform/xform-command.md)).
The bindings drop nothing; a caller drops first with
`SfmrReconstruction.drop_image_adjustments`
([drop-image-adjustments.md](drop-image-adjustments.md)), which `sfm xform
--drop-image-adjustments` does before a `--camera-model` later in the same
command
([../../cli/reconstruction/xform/xform-command.md](../../cli/reconstruction/xform/xform-command.md#--drop-image-adjustments-parametersnamename-camerasij)).

The report's `cameras` is a list with one dict per switched camera: `camera`,
`images`, `source` and `target` (the two `CameraIntrinsics`), `fit` (the dict
`CameraIntrinsics.refit` reports), `radial_corrections` (`None`, or a dict of
`rms_px`, `max_px`, `max_image`, `monotone_constrained_images` and `images`,
one dict per image refitted with `image`, `radial_bspline`, `rms_px`, `max_px`
and `monotone_constraint`), `observations` (the comparison's fields,
each summary a dict of `median_px`, `p90_px` and `max_px`) and `outermost` (as
`SfmrReconstruction.outermost_keypoints` reports it; the bindings always ask
for it). A refusal is a `ValueError` carrying the error's sentence.

```python
switched, report = edited.switch_camera_model("SFMTOOL_FISHEYE", coeff_count=8)
for camera in report["cameras"]:
    o = camera["observations"]
    print(camera["camera"], o["before"]["median_px"], o["after"]["median_px"])
```

## Testing

[switch_camera_model/tests.rs](../../../crates/sfmtool-core/src/reconstruction/switch_camera_model/tests.rs)
builds a scene through the `kerry_park` first lens with every keypoint at its
exact projection, plus one point 95° off the first image's axis:

- poses, points, tracks and keypoints bit-identical after the switch, the point
  errors recomputed, the input untouched;
- the report's fixed set covering every observation, and the one past the
  trusted bound counted;
- the outermost observation being the wide one, at its angle under the switched
  camera, with nothing detected beside no `.sift` file;
- only the named cameras changing;
- with every image carrying its own intrinsic adjustment and its keypoints made
  through it, every adjustment kept when nothing is dropped, the `before` errors
  under `10⁻³` px and the stored errors the ones a recompute gives;
- `fit_camera_switch` giving the camera and report the switch installs, for a
  change of model, a source with no trusted bound and a refit of a spline, and
  the switch's refusals; `default_fit_angle` giving the angle and source such a
  fit uses, and giving it for a target whose fit is refused;
- a source with no trusted bound fitted out to its observations' extent;
- a perspective target refused for a camera observed past 90°, and every
  refusal naming its camera;
- each of the focal and the principal point dropped before the fit: the
  camera installed being the fit of the camera `drop_image_adjustments`
  leaves, the parameter's flag cleared and its values zero, every other value
  kept, and `before` the value as it was handed in; nothing dropped, or only
  parameters the camera does not vary, giving the switch without the option;
- `image_adjustments_carried` on an `SFMTOOL_PINHOLE` camera varying all four:
  an `SFMTOOL_FISHEYE` target carrying the focal and principal point only, the
  camera's own refit, and one given its own domain end, all four as they are,
  and a given domain end or fit angle all four with the radial correction
  refitted;
- a kept radial correction or kept rates on a target that cannot carry them
  refused (`LensCorrections` naming what was kept), the same camera switching
  once they are dropped with its scanlines setting kept, and a camera listing
  either at zero refused alike;
- a same-model refit whose only change is a drop, on an `SFMTOOL_PINHOLE`
  camera that keeps its corrections and on an `SFMTOOL_FISHEYE` spline;
- the radial corrections refitted to a moved domain end: on the seoul_bull
  ground truth (read only) shortened and lengthened by 5°, every image
  refitted, its correction written and valid and monotone, nothing else of its
  adjustment changed, its curve over the shared domain moving no more than
  the camera's does plus 0.3 px, its report's rms the distance over the fit's
  angles and no larger than the old coefficients read on the new domain give,
  and the comparison closer than dropping the corrections leaves it; on the
  synthetic scene, a given domain end and a given fit angle both refitting,
  as `refit_radial_corrections` reports alone, an image with a zero
  correction refitted near zero, and on a camera that does not list the
  correction a zero kept exactly; a new count alone, and the camera's own
  domain end given as a number, carrying every correction bit for bit; and an
  image whose curve is flatter than the floor held at it, its curve monotone;
- a spline camera switched to its own model at a new count and domain giving
  exactly `refit_spline`'s camera and report, over the whole domain; the domain
  end kept bit for bit when none is given; a given fit angle taking the general
  fit instead; and a domain end past 180° refused, naming the camera;
- an empty or out-of-range camera list refused;
- beside a `.sift` file per image, the outermost keypoint detected from every
  file when asked for, and, left out, no entry, the same camera, fit,
  comparison and point errors, and the same report once every file is gone.

Bindings: `tests/rust_bindings/reconstruction/test_edited_reconstruction_rust_bindings.py`
(`TestSwitchCameraModel`) and `tests/xform/test_switch_camera_model.py`, which
includes a `sift_files` value read through its `.sift` files and one refused
without them.

## Non-goals

- Moving poses or points. The switch is a lens change; a bundle adjustment with
  the spline released ([`bundle-adjust.md`](bundle-adjust.md)) is the step that
  refines the new lens against the observations.
- Adding observations. Add Image to Tracks
  ([`add-image-to-tracks.md`](add-image-to-tracks.md)) adds those the old
  model's inverse kept out.
- Showing the change before it is applied. That is the viewer's camera model
  proposal ([`../../gui/edits/switch-camera-model.md`](../../gui/edits/switch-camera-model.md)),
  which calls `fit_camera_switch` while it is open and this function when it is
  applied.
