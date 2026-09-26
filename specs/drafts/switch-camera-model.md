# Switching a camera to another camera model

**Status:** Draft

Decided in outline:

- The switch is a fit: the new model is fitted to the old one over the angles
  where the old one is trusted.
- It is one core operation, reached from the Camera Intrinsics panel, from MCP
  and from `sfm xform --camera-model`.
- In the viewer, the change is shown as a proposal, drawn against the current
  model, before it is applied.

Not decided: see [Open questions](#open-questions).

Amends:

- [`../gui/camera-intrinsics.md`](../gui/camera-intrinsics.md). It lists
  "Editing intrinsics" as out of scope, which this draft replaces for the model
  switch.
- [`../cli/reconstruction/xform/xform-command.md`](../cli/reconstruction/xform/xform-command.md)
  § "Camera Model". Its `--camera-model` copies parameters by name and sets the
  new ones to zero; this draft makes it a fit.
- [`../gui/mcp-server.md`](../gui/mcp-server.md), which gains two tools.

A camera model is the function that maps a ray leaving the camera to a pixel.
Switching a camera to another model replaces that function with one from a
different family. The goal is to change the pixels as little as possible where
the old model was right, while letting the new model describe the parts of the
image the old one could not. Everything else in the reconstruction stays as it
is: poses, points and the measured keypoints. What changes is how rays are
mapped to those keypoints, so reprojection errors change, and the operation
reports by how much. The main use is moving a fisheye camera from a COLMAP
polynomial model, which fails a little past 90° off-axis, to `SFMTOOL_FISHEYE`,
whose spline and linear tail are defined out to 180°
([`../formats/sfmtool-camera-models.md`](../formats/sfmtool-camera-models.md)).

## Why: the Kerry Park lenses

`kerry_park_ground_truth_candidate_tk107.sfmr` has two cameras, one per rig
sensor. Both are `OPENCV_FISHEYE`, 480 × 480, with the principal point held at
(240, 240):

| | fx | fy | k1 | k2 | k3 | k4 |
|---|---|---|---|---|---|---|
| cam0 | 129.718 | 129.430 | 0.02865 | −0.00228 | 0.00902 | −0.00355 |
| cam1 | 128.886 | 129.747 | 0.04008 | −0.00428 | 0.00288 | −0.00098 |

Averaged over ten frames of each sensor, the image content fills a circle of
radius about 245 px. At f ≈ 129.6 that is about 108° off-axis. The observations
reach 101.4° (cam0) and 89.4° (cam1), but the 99th percentile is 85.7° and
84.6°: almost nothing is observed past 86°, and the models show why:

| θ (deg) | cam0 r (px) | cam0 ray error (deg) | cam1 r (px) | cam1 ray error (deg) |
|---|---|---|---|---|
| 80 | 192.5 | 0.000 | 192.4 | 0.000 |
| 85 | 205.0 | 0.044 | 205.6 | 0.330 |
| 88 | 211.9 | 1.641 | 213.4 | 3.292 |
| 92 | 220.1 | 4.218 | 223.7 | 7.352 |
| 98 | 228.6 | 2.978 | 238.2 | 7.893 |
| 102 | 230.3 | 0.280 | 247.0 | 7.782 |

The table gives the pixel radius the model assigns to a ray at θ off-axis. The
ray error is the angle between that ray and the ray `pixel_to_ray` returns for
the pixel.

Two failures show:

- **The inverse is wrong past about 86°.** The polynomial fisheye inverse blends
  toward the identity ray once the distorted angle passes 90°
  (`FISHEYE_BLEND_START_RAD` in `camera/distortion.rs`). For these lenses that
  point is at about 86° of incidence. Past it, every keypoint unprojects to a ray
  that is wrong by degrees. Any operation that goes from pixel to ray is affected
  there: triangulation, seeding a track, matching by rays.
- **cam0's forward map folds.** Its polynomial stops increasing at 101.6°, at
  230.3 px. So the ring of real image from 230 px to 245 px has no ray at all
  under cam0. From 95° to 102° the whole curve covers only 5 px, so the model
  there is fitted noise, not a lens.

`SFMTOOL_FISHEYE` has neither problem. Its inverse is a monotone 1-D solve with
no blend, and past `bspline_theta_max` it continues along a straight line.

A fit shows the switch loses almost nothing where the old model is right. The
probe fitted an 8-coefficient `SFMTOOL_FISHEYE` to each camera by the linear
least squares in [The fit](#the-fit), over θ ≤ 85° and 64 azimuths:

| | f | radial rms | rms over all azimuths | max |
|---|---|---|---|---|
| cam0 | 129.557 | 0.007 px | 0.092 px | 0.247 px |
| cam1 | 129.313 | 0.002 px | 0.275 px | 0.691 px |

The radial profile fits to within a hundredth of a pixel. All of the remaining
error is the difference between fx and fy (0.2 % for cam0, 0.7 % for cam1),
which a single-focal model cannot represent. The switch has to report that
loss; see [Open questions](#open-questions).

## What exists today

- **`sfm xform --camera-model NAME`** (`xform/_switch_camera_model.py`):
  - copies parameters that have the same name in both models;
  - sets target-only parameters to zero;
  - splits or averages the focal lengths.

  It does no fitting. Across a change of radial coordinate (tan θ vs θ) it
  produces a different lens. It only accepts the COLMAP models in
  `_CAMERA_PARAM_NAMES`; the spline models are deliberately left out, as
  "produced by refinement, never user-specified". Its following
  `--bundle-adjust` goes through pycolmap, which knows neither spline model.
- **Bundle adjustment frees the spline** (`opt_bspline` in
  `geometry/bundle_adjust.rs`), with each step gated on monotonicity. The
  release is reachable from the low-level Python binding `geometry.bundle_adjust`
  only. The reconstruction-level `BundleAdjustOptions`,
  `EditedReconstruction.bundle_adjust`, the viewer's Bundle adjust… dialog and
  the MCP `bundle_adjust` tool free the focal length only. None of them can free
  the distortion of an `OPENCV_FISHEYE` camera.
- **A spline refit prototype** exists on the unmerged `bootstrap-core-migration`
  branch, in `scripts/seed_relax/lens.py`. Its `refit_knots` fits spline
  coefficients to another camera's r(θ), `equivalent_focal` gives an equidistant
  focal, and `observed_field` places `d_max`. The fit here is the same idea,
  moved into core, with the focal solved jointly and the unconstrained span
  regularized.
- **`trustworthy_max_theta_deg(cam)`** (`camera/report.rs`) gives the angle where
  a polynomial fisheye's forward map folds or its inverse starts to blend. The
  intrinsics panel and the Image Detail overlay already shade past it. It is the
  natural default for the fit's domain.
- **Viewer displays of a model.**
  - The intrinsics panel's r(θ) and Δr(θ) plots against the family's ideal map
    (`intrinsics_detail/projection_plot.rs`), with a 32-azimuth band.
  - The Image Detail overlay's axes, iso-angle rings and arrow field of each
    pixel's displacement from the ideal map (`image_detail/intrinsics/`).

  Neither can show two models.
- **The Move Camera lock** ([`../gui/edits/move-camera.md`](../gui/edits/move-camera.md))
  is the one edit with a pending state the reviewer judges before committing.
  The proposal below follows its rules for how that state ends.

## The operation

### What changes and what does not

For each chosen camera, the operation:

1. fits the target model to the source over the fit domain (next section);
2. replaces the camera's intrinsics with the fitted camera;
3. recomputes the reprojection errors of the observations in images that use
   the camera.

Poses, points, keypoints, patches and track membership are untouched. Since
cameras belong to the reconstruction's base
([`../gui/document-model.md`](../gui/document-model.md)), in the viewer this is a
bulk edit: it builds a new base and pushes one version, the same way bundle
adjustment does.

A rig with one camera per sensor switches the cameras it is asked to. The viewer
offers "this camera" or "all cameras of this node", and the CLI takes a camera
list; see [The CLI](#the-cli).

### The fit

The fit samples rays over the domain θ ∈ [0, θ_fit] and 64 azimuths. It projects
each ray with the source model, and chooses the target's parameters to minimize
the pixel distance between the source's pixels and the target's. The principal
point is copied, not fitted: bundle adjustment never frees it, and a fit that
moved it would move every keypoint's ray for a reason the data did not give.

**θ_fit** defaults to the source's `trustworthy_max_theta_deg`, about 86° for the
Kerry lenses. For a model with no trusted bound (the perspective models, the
spline models), it defaults to the largest incidence angle among the camera's
observations. A caller may give a smaller θ_fit, never a larger one. A fit that
followed the source past its trusted bound would copy the fold into the new
model: fitting the Kerry cam0 to 100° instead of 85° matched the flattening
between 95° and 102°.

**Spline targets are a linear fit.** For `SFMTOOL_FISHEYE`,
r(θ) = f·(θ + Σ cᵢ Bᵢ(θ)), where the Bᵢ are the fixed basis functions on
[0, `bspline_theta_max`]. So r is linear in (f, f·c₀, …, f·c_{N−1}). One
least-squares solve gives the focal and the coefficients together, with no
starting point and no iteration. `SFMTOOL_PINHOLE` is the same fit on ρ = tan θ.
The zero-spline identity makes `EQUIDISTANT_FISHEYE` the N = 0 case.

**The span past θ_fit is regularized, not left free.**

- `bspline_theta_max` is placed where the format spec asks: at the incidence
  angle of the far image corner, estimated as corner radius / f. For Kerry that
  is about 150°. It is chosen once and held.
- Coefficients whose support lies wholly past θ_fit have no data. The fit adds a
  small penalty on the second difference of the coefficients, so δ continues past
  θ_fit as the smoothest curve that meets the data. Without the penalty the solve
  is rank-deficient, and a minimum-norm solution bends toward zero there.
- The result is checked with `bspline_is_monotone`. A fit that is not monotone is
  refused, not repaired.

**Polynomial targets are a small nonlinear fit.** For the COLMAP models
(`OPENCV_FISHEYE`, `RADIAL`, …), the same samples are fitted by Levenberg–Marquardt
from a start that copies same-named parameters, which is today's copy step. A
source that the target represents exactly (widening `SIMPLE_RADIAL` to `RADIAL`)
fits to zero error, and gives the answer today's `--camera-model` gives. A fitted
fisheye polynomial is checked the way the viewer checks one: its trusted bound
must reach θ_fit, or the fit is refused.

**Refusals.**

- A perspective target (`SFMTOOL_PINHOLE` or any COLMAP pinhole model) when θ_fit
  is 90° or more, or when the camera has observations at 90° or more. A
  perspective model has no pixel for such a ray.
- A non-monotone spline.
- A polynomial whose trusted bound falls short of θ_fit.
- An unknown or unsupported target.

A refusal names the camera, the rule and the measured value, and changes nothing.

### The report

Each camera's report says:

- **Source and target:** both models and their parameters.
- **Fit:**
  - θ_fit and where it came from (trusted bound, observations, or given);
  - rms and max pixel error over the samples;
  - the **radial rms**: the error of the azimuth-averaged radius alone;
  - what the target cannot represent, named. For Kerry that is "fx/fy aspect
    0.9978 dropped (single focal)", so a large overall error with a small radial
    rms reads as what it is.
- **Extent:** the angle the new model assigns to the image edge and corner, and
  for the source, its trusted bound and, if it has one, its fold.
- **Observations**, compared on the same set of observations: every observation
  of an image that uses the camera. Its median, 90th percentile and maximum
  reprojection error before and after. How many observations change by more than
  a pixel. How many observations lie past the source's trusted bound, and their
  errors before and after, since those are the observations the switch is for.

Comparing on a fixed set is deliberate. Afterwards, bundle adjustment and Add
Image to Tracks change the set, and a metric over a moving set can't separate
the lens from the population.

### Rust interface

```rust
// crates/sfmtool-core/src/camera/refit.rs
pub enum RefitTarget {
    SfmtoolFisheye { coeff_count: usize },
    SfmtoolPinhole { coeff_count: usize },
    EquidistantFisheye,
    Colmap(CameraModelName),
}

pub struct RefitOptions {
    /// Largest incidence angle the fit samples. `None` takes the source's
    /// trusted bound, or its observations' extent.
    pub theta_fit: Option<f64>,
    /// `bspline_theta_max` / `bspline_rho_max` for a spline target. `None`
    /// takes the far image corner.
    pub spline_domain_max: Option<f64>,
}

pub struct CameraRefit {
    pub camera: CameraIntrinsics,
    pub theta_fit: f64,
    pub theta_fit_source: ThetaFitSource,
    pub rms_px: f64,
    pub max_px: f64,
    pub radial_rms_px: f64,
    pub dropped: Vec<DroppedTerm>,
    pub extent: ModelExtent,
}

pub fn refit_camera(
    source: &CameraIntrinsics,
    target: &RefitTarget,
    options: &RefitOptions,
) -> Result<CameraRefit, RefitError>;
```

`refit_camera` knows only the lens. The reconstruction-level operation adds the
observations:

```rust
// crates/sfmtool-core/src/reconstruction/switch_camera_model.rs
impl EditedReconstruction {
    pub fn switch_camera_model(
        &self,
        cameras: &[usize],
        target: &RefitTarget,
        options: &RefitOptions,
    ) -> Result<(EditedReconstruction, SwitchCameraModelReport), RefitError>;
}
```

This keeps the lens fit separate so the viewer can run it alone, many times, as
the reviewer changes the target or the coefficient count; see
[The proposal](#the-proposal). A fit is a few hundred samples and one small
solve, so it runs well within a frame. Both functions are bound in `sfmtool-py`:

- `CameraIntrinsics.refit(target, *, coeff_count, theta_fit, spline_domain_max)`;
- `EditedReconstruction.switch_camera_model(cameras, target, ...)`, and the same
  method on `SfmrReconstruction` for xform.

## What follows the switch

The switch alone changes little, since it moves pixels by under a pixel where
there are observations. What it gives is a model that can be refined out to the
image circle, and three existing steps then do that:

1. **Bundle adjustment with the spline freed.** This is a companion change.
   `opt_bspline` is carried up to:
   - `BundleAdjustOptions` and `EditedReconstruction.bundle_adjust`;
   - the Bundle adjust… dialog, as a "Release lens distortion" checkbox under
     "Release focal length", enabled when a camera of the node has a spline;
   - MCP `bundle_adjust` as `release_distortion`.

   Without it, the viewer and xform can switch to a spline and never refine it.
2. **Add Image to Tracks** on the images, now that tracks project past 86° to the
   right pixel. It adds the observations the old model's inverse kept out.
3. **Bundle adjustment again**, now with observations where the spline had none.

The report's observation counts past the old trusted bound show whether step 2
reached the periphery. As
[`add-image-to-tracks`](../gui/edits/add-image-to-tracks.md) established, whether
the richer connectivity makes a better reconstruction is judged by inspection,
not by the residual metrics alone.

## The viewer

### Invocation

- A **Switch Model…** button in the Camera Intrinsics panel header, beside
  `Copy ▾`.
- A **Switch Camera Model…** entry on a new context menu on the camera rows of the
  Scene tree. The rows have no context menu today.
- `Edit > Switch Camera Model…`, greyed when no camera is selected.

Each opens the proposal for the selected camera.

### The proposal

While a proposal is open, the node has a **proposed camera**: a fitted
`CameraIntrinsics` and its report, held as viewer state the way the Move Camera
lock holds a pending pose. It is not in the version history until it is applied.

The panel shows a strip of controls at the top:

- **Model**: the targets the source can be fitted to, with `SFMTOOL_FISHEYE`
  first for a fisheye source.
- **Coefficients** (2–16, default 8), for a spline target.
- **Fit to θ**: shows the default and where it came from; can be lowered.
- **Spline domain**: shows the default corner angle; can be edited.
- **Cameras**: this camera, or all cameras of this node.
- **Apply** and **Cancel**.

Every change refits at once. The fit report is one line under the strip, for
example: "rms 0.09 px, radial 0.007 px, max 0.25 px over θ ≤ 85.0°; fx/fy aspect
0.9978 dropped". The observation comparison, which needs the reconstruction, is
computed on a worker when the controls rest.

While the proposal is open, the rest of the panel compares the two models:

- **Parameters and Derived**: two columns, current and proposed.
- **Projection plot, r(θ) and Δr(θ)**: both models drawn, current in the
  neutral stroke and proposed in the accent. Each model's own untrusted region
  is shaded, so the current model's shading at 86° and the proposed model's
  reach to the image circle are both visible.
- **A third plot, "Change (px)"**: the proposed radius minus the current one,
  against θ, with the azimuth band. The fx ≠ fy loss shows as the band's width.
  A fitting error shows as the centre line's departure from zero inside θ_fit.
  Past θ_fit it shows how far the new model departs from where the old one was
  going.
- **An observation rug** under the shared θ axis: one tick per observation of
  the camera's images, at its incidence angle under the current model. It shows
  where the data is: the Kerry rug thins out at 86°. Beside the rug is the count
  of observations past the current model's trusted bound.

**Apply** pushes one version, labelled "Switch camera model", and writes an Action
Log row carrying the report's summary. **Cancel** and `Escape` discard the
proposal. The implicit endings follow the Move Camera lock: selecting another
node, closing it, or an MCP edit tool naming it applies the proposal when the fit
succeeded, since an edit has a history and undo can reverse it. The proposal is
dropped when the fit was refused. Another edit on the node from the menus is
greyed while a proposal is open.

### What else shows the proposal

**The Image Detail intrinsics overlay** gains a second field mode, **Change to
proposed**. It is selected automatically while a proposal is open for the
image's camera, and restored afterwards.

- **Arrows over the image grid.** Each arrow's tail is the pixel the current
  model gives a ray, and its head is the pixel the proposed model gives the same
  ray, with the overlay's usual exaggeration and legend. The arrows are drawn
  only where the current model is trusted.
- **Past the trusted bound**, the current model has no reliable pixel, so the
  overlay instead draws the proposed model's iso-angle rings in that zone (every
  5° from θ_fit to the edge). For Kerry that shows the band from 86° to the image
  circle and how it is now divided.
- **An observation layer.** For each observation of the image, a tick at the
  keypoint and a line to its reprojection under the current model and under the
  proposed one. This is the fixed-set comparison, shown where it happens.

The axes and rings follow the proposed model while the proposal is open, and a
checkbox in the ⚙ popup puts them back on the current model.

**The 3D view is left on the current model.** The distorted frustums and
background mesh would change by fractions of a pixel, which are not visible at
viewport scale.

### Caches

Two caches key on `CameraRef` alone and must not serve the proposed camera as the
current one:

- the panel's `derived` map (`intrinsics_detail/mod.rs`);
- the Image Detail layer cache (`image_detail/mod.rs`).

Both gain a second key for the model they were computed from: current, or the
proposal's generation. Apply goes through `forget_recon` the way every bulk
edit does.

## MCP

The switch differs from Move Camera in one way that matters for the wire: its
inputs are a few discrete choices an agent can state. So the proposal is on the
wire too, and an agent can open it, screenshot the panel and the overlay, and
then apply or cancel. This is the check a reviewer would make.

- **`propose_camera_model { reconstruction_label, camera, model, coeff_count?,
  theta_fit?, spline_domain_max?, all_cameras? }`** opens or replaces the
  proposal exactly as the panel does, and answers with the fit report. It pushes
  nothing.
- **`switch_camera_model { reconstruction_label, camera?, model?, ... }`** applies:
  - the open proposal, when no model is named;
  - a direct switch, when one is named. The direct form ends an open proposal
    first, by the rule above.

  It answers like every edit tool: the version, its label, the sentence recorded,
  and the report, including the observation comparison.

`get_camera_intrinsics` gains a `proposed` block while a proposal is open, and
`get/set_image_detail_display` gains the field mode.

## The CLI

`--camera-model` keeps its name and becomes the fit:

```bash
sfm xform in.sfmr out.sfmr --camera-model SFMTOOL_FISHEYE,coeffs=8
sfm xform in.sfmr out.sfmr --camera-model SFMTOOL_FISHEYE,coeffs=8,fit_to=80,cameras=0
```

- The first comma-separated field is the model.
- The rest are `coeffs=`, `fit_to=` (degrees), `spline_domain=` (degrees) and
  `cameras=` (indexes separated by `+`, default all).

The report is printed per camera. `_CAMERA_PARAM_NAMES`'s exclusion of the spline
models is lifted for this option only: they are user-chosen targets now, but
still never a `solve` or `match` camera model.

For the COLMAP models, the change of behaviour is intended. A switch that stays
in the same radial coordinate fits to what the copy gave. A switch across
coordinates, which today silently gives a different lens, becomes correct.

`--bundle-adjust` goes through pycolmap and cannot adjust a spline camera. After
a switch to a spline model it will run sfmtool's own bundle adjustment, with
`opt_f` and `opt_bspline`, when any camera is a spline. Whether that becomes a
separate option or the default for every camera is an open question.

## Testing

- **Core, exact cases:**
  - `EQUIDISTANT_FISHEYE` to `SFMTOOL_FISHEYE` fits to zero error with zero
    coefficients;
  - `SIMPLE_RADIAL` to `RADIAL` reproduces the copy.
- **Core, recovery:** a synthetic `SFMTOOL_FISHEYE` rendered as samples and fitted
  back recovers its coefficients inside θ_fit.
- **Core, refusals:** perspective target past 90°; non-monotone spline; θ_fit
  beyond the trusted bound.
- **Core, domain:** an `OPENCV_FISHEYE` fold at 101° does not reach the fit when
  θ_fit defaults to the trusted bound.
- **Reconstruction:** poses, points and keypoints are bit-identical after the
  switch. Reprojection errors are recomputed.
- **Viewer lib tests:** a proposal leaves the history unchanged; Apply pushes one
  version; Cancel restores; the two caches do not cross.
- **A Kerry evaluation run**, kept as a script rather than a test:
  1. switch tk107 → Add Image to Tracks over all images;
  2. bundle adjust with the spline freed;
  3. report the observation counts by θ band before and after, and the fitted
     spline against the image circle at 245 px.

  The result is judged in the viewer.

## Order of work

1. `refit_camera` and `switch_camera_model` in core, the bindings, and
   `--camera-model` on the fit.
2. `opt_bspline` through `BundleAdjustOptions`, the dialog and MCP. Spline-aware
   `--bundle-adjust` in xform.
3. The proposal in the Camera Intrinsics panel: the controls, the two-model
   plots, the change plot and the rug. Apply as an edit.
4. The Image Detail change field and observation layer.
5. The MCP tools.

The Kerry evaluation run can start after 2, from the CLI.

## Open questions

- **The dropped aspect.** `SFMTOOL_FISHEYE` has one focal. The Kerry fx/fy
  differences of 0.2 % and 0.7 % are most likely fitting noise on square pixels,
  and the switch drops them visibly. If a real lens needs them, an aspect
  parameter is a format change to the spline models. Measure after the Kerry run
  whether bundle adjustment's residuals show an azimuthal pattern.
- **The spline domain for a circular fisheye.** The far corner is at about 150°,
  but the Kerry image content ends at about 108°. So two of eight knot spans
  cover black pixels. Placing `bspline_theta_max` at the image circle instead
  needs the circle, which nothing detects today. It is fixed once chosen, so the
  choice matters.
- **The regularization weight** past θ_fit, and whether the Kerry run prefers a
  smooth continuation or the linear tail starting at θ_fit.
- **Whether the switch should free the principal point** for a model whose
  source never fitted it. Today nothing frees it anywhere, and this draft keeps
  that.
- **Spline-aware `--bundle-adjust`**: a new option, or the Rust path for every
  camera.
