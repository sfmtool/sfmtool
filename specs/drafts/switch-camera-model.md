# Switching a camera to another camera model in the viewer

**Status:** Draft

Decided in outline:

- The switch is a fit, one core operation, reached from the Camera Intrinsics
  panel, from MCP and from `sfm xform --camera-model`. The core operation, the
  CLI and the direct switch are built: the lens fit is
  [`../core/camera/refit-camera-intrinsics.md`](../core/camera/refit-camera-intrinsics.md), the reconstruction-level
  switch is
  [`../core/reconstruction/switch-camera-model.md`](../core/reconstruction/switch-camera-model.md),
  the CLI is
  [`../cli/reconstruction/xform/xform-command.md`](../cli/reconstruction/xform/xform-command.md)
  § "Camera Model", and the viewer's direct switch is
  [`../gui/edits/switch-camera-model.md`](../gui/edits/switch-camera-model.md):
  the "Refit spline…" action in the Camera Intrinsics panel header, which
  gives a spline camera a new coefficient count or domain, and the MCP tool
  `switch_camera_model`, which applies a switch or a refit at once.
- In the viewer, a change of model is shown as a proposal, drawn against the
  current model, before it is applied. That is what remains, with the MCP tool
  `propose_camera_model` and the proposal form of `switch_camera_model`.

Not decided: see [Open questions](#open-questions).

Amends:

- [`../gui/camera-intrinsics.md`](../gui/camera-intrinsics.md), whose header
  gains the Switch Model… button beside "Refit spline…".
- [`../gui/mcp-server.md`](../gui/mcp-server.md), which gains
  `propose_camera_model`, and whose `switch_camera_model` gains the proposal
  form.
- [`../gui/edits/switch-camera-model.md`](../gui/edits/switch-camera-model.md),
  the direct switch the proposal's Apply pushes.
- [`../core/reconstruction/switch-camera-model.md`](../core/reconstruction/switch-camera-model.md),
  whose non-goals name the proposal this draft describes.

A camera model is the function that maps a ray leaving the camera to a pixel.
Switching a camera to another model replaces that function with one fitted to
it over the angles where the old one is trusted, and leaves poses, points and
keypoints alone. The main use is moving a fisheye camera from a COLMAP
polynomial model, which fails a little past 90° off the axis, to
`SFMTOOL_FISHEYE`, whose spline and linear tail are defined out to 180°.

## Why: the Kerry Park lenses

`kerry_park_ground_truth_candidate_tk107.sfmr` has two cameras, one per rig
sensor, both `OPENCV_FISHEYE`, 480 × 480, with the principal point held at
(240, 240). Averaged over ten frames of each sensor, the image content fills a
circle of radius about 245 px, which at f ≈ 129.6 is about 108° off the axis. The
polynomials' inverse blends toward the identity ray from about 84° to 86°, and
cam0's forward map folds at 101.6°, so the ring of real image from 230 px to
245 px has no ray at all under cam0.

`sfm xform --camera-model SFMTOOL_FISHEYE,coeffs=8` on that file reports:

| | f | radial rms | rms | max | θ_fit | dropped |
|---|---|---|---|---|---|---|
| cam0 | 129.523 | 0.013 px | 0.130 px | 0.292 px | 84.5° | fx/fy aspect 0.9978 |
| cam1 | 129.301 | 0.003 px | 0.384 px | 0.690 px | 83.8° | fx/fy aspect 1.0067 |

The radial profile fits to about a hundredth of a pixel. The rest of the error
is the difference between fx and fy, which a single-focal model cannot
represent. Deciding whether that loss is acceptable for a given lens is the
reviewer's judgment, and the viewer is where it is made: the proposal below
exists so the reviewer can see the change, not only read its numbers.

## What follows the switch

The switch alone changes little, since it moves pixels by under a pixel where
there are observations. What it gives is a model that can be refined out to the
image circle: bundle adjustment with the camera's spline released (its row's
"Release lens distortion" in the Bundle Adjust dialog, MCP `bundle_adjust`'s
`release_distortion` or a `cameras` entry naming it, or `sfm xform
--bundle-adjust` on a spline camera), then Add Image to Tracks on
the images, now that tracks project past 86° to the right pixel, then bundle
adjustment again with observations where the spline had none. The proposal's
observation counts past the old trusted bound show whether the second step
reached the periphery. As
[`add-image-to-tracks`](../gui/edits/add-image-to-tracks.md) established,
whether the richer connectivity makes a better reconstruction is judged by
inspection, not by the residual metrics alone.

The first release on tk107 shows why. `sfm xform --camera-model
SFMTOOL_FISHEYE,coeffs=8 --bundle-adjust` brought the median residual from
0.298 px to 0.274 px, and moved the coefficient whose support starts at about
86° from 0.09 to −0.48 on cam0 and from 0.25 to −0.43 on cam1. At most the 58
and 39 observations past the old trusted bound reach that coefficient, and some of
them are off by hundreds of pixels under either model. The two coefficients no
observation reaches came back unchanged. Whether the periphery that fit
produces describes the lens or those observations is judged in the viewer.

## The viewer

### Invocation

- A **Switch Model…** button in the Camera Intrinsics panel header, beside
  "Refit spline…" and `Copy ▾`.
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
- **Spline domain**: shows the default corner angle; can be edited. Beside it,
  the outermost keypoint of the camera's images
  ([`../core/reconstruction/outermost-keypoint.md`](../core/reconstruction/outermost-keypoint.md)),
  detected where the `.sift` files can be read and observed otherwise, labelled
  by its source, with the same button the Bundle Adjust dialog's domain row has,
  which sets the domain to its angle. The default stays the corner.
- **Cameras**: this camera, or all cameras of this node.
- **Apply** and **Cancel**.

Every change refits at once. The fit report is one line under the strip, for
example: "rms 0.13 px, radial 0.013 px, max 0.29 px over θ ≤ 84.5°; fx/fy aspect
0.9978 dropped". A spline fit is constrained to stay monotone, so it has an
inverse; where that constraint bound, the line adds its range, for example
"monotone constraint bound at 1 angle, 113.2°", because there the proposed
curve is the closest invertible one rather than the current one. The observation comparison, which needs the reconstruction, is
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
  The range where the monotonicity constraint bound is marked on the θ axis,
  so a departure there reads as the constraint rather than as a poor fit.
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
- **`switch_camera_model`** exists and applies a direct switch or a spline
  refit ([`../gui/mcp-server.md`](../gui/mcp-server.md)). It gains a
  `proposal: true` form that applies the open proposal instead of fitting one;
  the direct form ends an open proposal first, by the rule above. Either answers
  like every edit tool: the version, its label, the sentence recorded, and the
  report, including the observation comparison.

`get_camera_intrinsics` gains a `proposed` block while a proposal is open, and
`get/set_image_detail_display` gains the field mode.


## Testing

- **Viewer lib tests:** a proposal leaves the history unchanged; Apply pushes one
  version; Cancel restores; the two caches do not cross.
- **MCP tests:** `propose_camera_model` pushes nothing and answers with the fit
  report; `switch_camera_model { proposal: true }` applies the open proposal,
  and a direct switch ends one.
- **A Kerry evaluation run**, kept as a script rather than a test:
  1. switch tk107 → Add Image to Tracks over all images;
  2. bundle adjust with the spline freed;
  3. report the observation counts by θ band before and after, and the fitted
     spline against the image circle at 245 px.

  The result is judged in the viewer.

## Order of work

1. The proposal in the Camera Intrinsics panel: the controls, the two-model
   plots, the change plot and the rug. Apply as an edit.
2. The Image Detail change field and observation layer.
3. `propose_camera_model`, and the proposal form of `switch_camera_model`.

## Open questions

- **The dropped aspect.** `SFMTOOL_FISHEYE` has one focal. The Kerry fx/fy
  differences of 0.2 % and 0.7 % are most likely fitting noise on square pixels,
  and the switch drops them visibly. If a real lens needs them, an aspect
  parameter is a format change to the spline models. Measure after the Kerry run
  whether bundle adjustment's residuals show an azimuthal pattern.
- **Whether the switch should free the principal point** for a model whose
  source never fitted it. Today nothing frees it anywhere, and the switch keeps
  that.

The regularization weight is an open question of the fit itself, in
[`../core/camera/refit-camera-intrinsics.md`](../core/camera/refit-camera-intrinsics.md).
The spline domain is settled there: the default is the far image corner, the
model's own reach, and the outermost keypoint with its button is shown wherever
the domain is edited, so a circular fisheye is trimmed to its image circle by
choice.
