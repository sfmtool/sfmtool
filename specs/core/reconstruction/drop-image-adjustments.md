# Dropping per-image parameters from a camera

A reconstruction's camera can let some of its intrinsics vary from one
photograph to the next: each image's focal, principal point, radial distortion
correction and rolling-shutter rates can carry a small adjustment of its own on
top of the shared camera, which a phone's autofocus, stabiliser and readout
call for. Dropping one of those per-image parameters from a camera goes the
other way: the camera stops varying it, every image of the camera is read
through the shared value again, and the shared value moves to the images' mean
so that the lens changes as little as it can. It is the step a person takes
before switching the camera to another model, which cannot carry a radial
correction or rates over, or to see what a per-image parameter was worth by
removing it.

The per-image parameters themselves, their units and how an image's
intrinsics are built from them are in
[sfmr-file-format.md § Per-image intrinsic
adjustments](../../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13)
and [sfmtool-pinhole-kernels.md § Per-image
corrections](../camera/sfmtool-pinhole-kernels.md#per-image-corrections). Only
the bundle adjustment turns a per-image parameter on, since it is what solves
it ([bundle-adjust.md](bundle-adjust.md)); this operation only turns one off.

## Rust API

The function lives in
[drop_image_adjustments.rs](../../../crates/sfmtool-core/src/reconstruction/drop_image_adjustments.rs),
as `sfmtool_core::reconstruction::drop_image_adjustments`, and is bound as
`SfmrReconstruction.drop_image_adjustments`
([drop_image_adjustments.rs](../../../crates/sfmtool-py/src/reconstruction/drop_image_adjustments.rs)).
The viewer and the CLI do not call it yet; how they will is proposed in
[per-image-adjustments-ui.md](../../drafts/per-image-adjustments-ui.md).

```rust
pub fn drop_image_adjustments(
    recon: &SfmrReconstruction,
    camera: usize,
    parameters: &[ImageAdjustmentParameter], // Focal, PrincipalPoint, RadialBspline, RollingShutter
) -> Result<SfmrReconstruction, DropImageAdjustmentsError>;

pub enum DropImageAdjustmentsError {
    UnknownCamera { camera: usize, count: usize },
    Fold { camera: usize, message: String },   // the mean cannot be applied, or the focal is not positive
    FoldedSplineNotMonotone { camera: usize },
    Intrinsics(String),                        // ImageSet::check_image_intrinsics refused the result
}
```

### Why it is shaped this way

**A pure function on a whole reconstruction**, like
[`switch_camera_model`](switch-camera-model.md) and
[`move_camera`](move-camera.md): the value goes in, a new one comes out, and the
caller decides whether to keep it. The viewer's Switch Camera Model proposal
can then drop a camera's per-image parameters and fit the new model to the
value that comes back, without either operation knowing about the other.

**One camera per call.** The parameters a camera varies, and the mean folded
into it, are that camera's alone, so a caller dropping from several cameras
calls once for each; nothing is shared between the calls that one call over a
list would save.

**The parameters are [`ImageAdjustmentParameter`](../../../crates/sfmtool-core/src/reconstruction/bundle_adjust.rs)**,
the bundle adjustment's name for one part of an intrinsic adjustment, whose
`name()` is the word a camera's `per_image_adjustments` lists. A
`PerImageAdjustments` would also have carried a scanlines setting that means
nothing here.

**An error enum, not a string.** A caller such as the viewer shows a refused
spline differently from a camera index that went stale, so each refusal is its
own variant with the camera it concerns.

**No report.** What the drop costs is a reprojection comparison, and the
caller that shows one already has its own: Switch Camera Model's proposal
compares the observations before and after over one fixed set. The stored
per-point errors are kept current (below), so a caller can compare those too.

### Example

```rust
use sfmtool_core::reconstruction::bundle_adjust::ImageAdjustmentParameter;
use sfmtool_core::reconstruction::drop_image_adjustments::drop_image_adjustments;

let dropped = drop_image_adjustments(
    &recon,
    0,
    &[ImageAdjustmentParameter::RadialBspline, ImageAdjustmentParameter::RollingShutter],
)?;
assert!(!dropped.image_set.camera_per_image_adjustments(0).radial_bspline);
```

## What the call does

For each parameter named, the camera's flag for it is cleared, every image of
the camera has that part of its adjustment set to zero, and the images' mean
is folded into the camera:

| Parameter | The camera after |
|-----------|------------------|
| `focal` | each focal (both on a two-focal model) times the geometric mean of `1 + s`, `exp(mean ln(1 + s))` |
| `principal_point` | the principal point moved by the mean offset `ā` times the short side, `min(width, height)` |
| `radial_bspline` | the spline coefficients `c` replaced by `c + K·ē`, or `ē` on an empty spline (`SFMTOOL_PINHOLE` only) |
| `rolling_shutter` | unchanged: a mean rate describes no photograph, so nothing is folded |

The fold is `CameraIntrinsics::with_adjustment` with the mean adjustment, so it
is the same arithmetic an image holding the mean was read through: such an
image's intrinsics are identical before and after, to the last bit for the
principal point and the radial correction and to rounding for the focal, whose
mean is taken through `ln(1 + s)` and back.

The means are the ones the bundle adjustment re-centres a camera to after each
round of a per-image release
([bundle-adjust.md § Why it is shaped this way](bundle-adjust.md#why-it-is-shaped-this-way)), so a drop straight
after such a solve folds nothing and leaves the shared camera as it was. The
principal point and the radial correction are linear in the camera's
parameters, and their mean is the shared value whose change sums to zero over
the images. The focal is the exception: the scale `1 + s` multiplies the
camera's focal, so its mean is taken in log space, where the images'
`ln(1 + s)` sum to zero about the folded camera. An arithmetic mean of
`1 + s` would sit above the geometric one by about half the variance of
`ln(1 + s)`, and would move a freshly solved camera's focal by that much.

**The mean is over the camera's posed images**, the ones whose observations the
drop changes. An unposed image's adjustment is bookkeeping that keeps its
intrinsics where they were, and is left out. A camera with no posed image takes
the mean over all its images, and a camera with no image folds nothing.

**The folded camera must be a valid lens.** The camera must take the mean as
an adjustment (`check_adjustment`), the folded focal must be positive and
finite and the principal point finite (`Fold`), and a folded spline must be
finite and monotone by `bspline_is_monotone` over its whole domain, the test
the bundle adjustment's spline step guard applies (`FoldedSplineNotMonotone`).
The mean of monotone splines is monotone, since `1 + δ'(ρ) > 0` is linear in
the coefficients, so in practice the last refusal is reached only by a value whose images
were not valid lenses to begin with. The result is then checked by
`ImageSet::check_image_intrinsics` (`Intrinsics`).

**The scanlines setting is kept** when the rolling shutter is dropped. It says
whether the sensor reads out by rows or by columns, which is a fact about the
camera rather than about the solve, and the format allows it on a camera that
lists no `"rolling_shutter"`, so a later solve of the rates starts from the
known setting.

**A parameter the camera does not vary**, whose values on its images are all
zero, is left as it is: dropping it changes nothing, and a call that drops
nothing, an empty `parameters` included, returns the value unchanged with its
content hashes. This makes the call safe to repeat, which a caller offering one
checkbox per parameter relies on. A parameter the camera does not list but
whose images carry non-zero values, which a file cannot hold but a value built
in memory can, is dropped and folded like any other. When no camera has a flag
or a scanlines setting left, `ImageSet::per_image_adjustments` is emptied, as
it is for a file without adjustments.

**Nothing else moves.** Other cameras and their images, every pose and every
point are carried over unchanged; a caller that wants the poses and points to
follow the new lens bundle-adjusts or re-triangulates afterwards. The stored
error of every point the camera's images observe is recomputed from the inline
keypoints, as `SfmrReconstruction::recompute_point_errors` scores it, where the
value carries them; a `sift_files` value without them keeps its stored errors,
as [move-camera.md](move-camera.md) does. Points at infinity need nothing of
their own: the drop changes only intrinsics, and the rescoring projects a
direction through its bearing like any other point.

## Python bindings

```python
dropped = recon.drop_image_adjustments(0, ["radial_bspline", "rolling_shutter"])
```

`camera` is the camera's index and `parameters` a list of the names
`per_image_adjustments` lists. It returns a new `SfmrReconstruction` and leaves
this one as it was. An unknown name, an unknown camera and every refusal above
raise `ValueError` with the error's message.

## Testing

[`drop_image_adjustments/tests.rs`](../../../crates/sfmtool-core/src/reconstruction/drop_image_adjustments/tests.rs)
builds two `SFMTOOL_PINHOLE` cameras with every parameter varying, and checks
that an image holding the mean is read as before after each fold, that the
folded camera is the mean over the posed images (the geometric one for the
focal), that a drop of the focal right after re-centring leaves the camera's
focal bit-identical, that dropping the rates keeps the camera and its
scanlines setting, that the other camera and every pose and point are
untouched, that dropping what a camera does not vary changes nothing, that
the stored errors match a full rescoring, and each refusal.
[`test_drop_image_adjustments_rust_bindings.py`](../../../tests/rust_bindings/reconstruction/test_drop_image_adjustments_rust_bindings.py)
drops from the seoul_bull ground truth's camera, which varies all four, and
saves and reloads the result.
