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
a later bundle adjustment can refine the lens out to the image circle.

The lens fit itself is [`../camera/refit-camera-intrinsics.md`](../camera/refit-camera-intrinsics.md). This spec is
the layer that adds the reconstruction: which observations belong to a camera,
how the fit's largest angle is chosen from them, what is written back, and the
before-and-after comparison.

## Rust API

The function lives in
[switch_camera_model.rs](../../../crates/sfmtool-core/src/reconstruction/switch_camera_model.rs),
as `sfmtool_core::reconstruction::switch_camera_model`, and is bound as
`EditedReconstruction.switch_camera_model` and
`SfmrReconstruction.switch_camera_model`.

```rust
pub fn switch_camera_model(
    recon: &SfmrReconstruction,
    cameras: &[usize],
    target: &RefitTarget,
    options: &RefitOptions,
) -> Result<(SfmrReconstruction, SwitchCameraModelReport), SwitchCameraModelError>;

pub struct SwitchCameraModelReport {
    pub cameras: Vec<CameraSwitch>, // one per switched camera, in table order
}

pub struct CameraSwitch {
    pub camera: usize,
    pub source: CameraIntrinsics,
    pub refit: CameraIntrinsicsRefit,          // carries the new camera
    pub images: usize,
    pub observations: ObservationComparison,
    pub outermost: OutermostKeypoints,         // under the switched camera
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

**A refusal names the camera.** Every camera is fitted before anything is
written, and the first refusal refuses the whole switch with the camera's index
and the fit's own reason.

### Example

```rust
use sfmtool_core::camera::refit_intrinsics::{RefitOptions, RefitTarget};
use sfmtool_core::reconstruction::switch_camera_model::switch_camera_model;

let target = RefitTarget::from_name("SFMTOOL_FISHEYE", Some(8))?;
let (switched, report) = switch_camera_model(&recon, &[0, 1], &target, &RefitOptions::default())?;
for camera in &report.cameras {
    let o = &camera.observations;
    println!(
        "camera {}: median {:.3} -> {:.3} px, {} past the trusted bound",
        camera.camera, o.before.median_px, o.after.median_px, o.past_trusted,
    );
}
```

## What the call does

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

With no `theta_fit_deg` given, each camera's fit reaches:

1. its trusted bound, where the model has one;
2. otherwise the largest incidence angle among its observations;
3. otherwise, for a camera with no observations, its far image corner.

A caller's value is used as given, and the fit refuses one past the trusted
bound. A perspective target is refused for a camera observed at 90° or more
(`ObservationsPast90`), whatever `θ_fit` is, because those observations would
have no pixel under it.

### What is written back

Each switched camera is replaced by its fitted camera. Nothing else in the image
table, the point set or the observation columns changes. The stored error of
every point observed in an image that uses a switched camera is recomputed as
the mean of its reprojection errors over all its observations, the convention
`SfmrReconstruction::recompute_point_errors` writes; a point with none that
project gets `0.0`. Points only other cameras see keep their stored error.

### The comparison

For each camera, the **fixed set** is its observations with a finite
reprojection error under both models. Every figure is over that set, so the
change it shows is the lens and nothing else: `before` and `after` are the
median, 90th percentile (linear interpolation) and maximum error;
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

Comparing on a fixed set is deliberate. A following bundle adjustment or Add
Image to Tracks changes the set, and a metric over a moving set cannot separate
the lens from the population.

## Python bindings

`EditedReconstruction.switch_camera_model(target, *, cameras=None,
coeff_count=None, theta_fit_deg=None, spline_domain_deg=None)` returns
`(EditedReconstruction, report)`, a new base with no overlay; `cameras=None`
switches every camera. `SfmrReconstruction.switch_camera_model` takes the same
arguments and returns `(SfmrReconstruction, report)`; it is what `sfm xform
--camera-model` calls
([`../../cli/reconstruction/xform/xform-command.md`](../../cli/reconstruction/xform/xform-command.md)).

The report's `cameras` is a list with one dict per switched camera: `camera`,
`images`, `source` and `target` (the two `CameraIntrinsics`), `fit` (the dict
`CameraIntrinsics.refit` reports), `observations` (the comparison's fields,
each summary a dict of `median_px`, `p90_px` and `max_px`) and `outermost` (as
`SfmrReconstruction.outermost_keypoints` reports it). A refusal is a
`ValueError` carrying the error's sentence.

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
- a source with no trusted bound fitted out to its observations' extent;
- a perspective target refused for a camera observed past 90°, and every
  refusal naming its camera;
- an empty or out-of-range camera list refused.

Bindings: `tests/rust_bindings/test_edited_reconstruction_rust_bindings.py`
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
- A proposal the viewer shows before the switch is applied, and the MCP tools
  for it. Both are proposed in
  [`../../drafts/switch-camera-model.md`](../../drafts/switch-camera-model.md).
