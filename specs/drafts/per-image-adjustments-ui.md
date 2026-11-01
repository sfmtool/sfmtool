# Showing and dropping per-image adjustments in the viewer and the CLI

**Status:** Draft. Decided: the five steps below, each its own pull request,
and the rule that only the bundle adjustment turns a per-image parameter on
while every other operation only drops one. Step 1 is built and specified in
[core/reconstruction/drop-image-adjustments.md](../core/reconstruction/drop-image-adjustments.md);
this draft proposes steps 2 to 5. Open: the exact layout of the statistics and
histograms, whether the rates are shown per frame of readout or in their
stored unit, and whether `sfm xform` gets an option to drop per-image
parameters.

A camera in a `.sfmr` file can vary its focal, principal point, radial
correction and rolling-shutter rates per image
([sfmr-file-format.md § Per-image intrinsic
adjustments](../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13)).
Today the viewer shows one image's values at a time, in Image Detail's legend,
and nothing in the viewer or the CLI can drop a per-image parameter once the
bundle adjustment has turned it on. This draft proposes how a person sees all
of a camera's per-image values at once, drops the ones they do not want, and
releases the parameters the CLI cannot release yet.

## Step 2: statistics in the Camera Intrinsics panel

The panel's Parameters section
([gui/camera-intrinsics.md § 2. Parameters](../gui/camera-intrinsics.md#2-parameters))
shows, beside each shared parameter the camera also varies per image, the
statistics of its images' values: the image count, the mean, the standard
deviation, the minimum and maximum, and the number of images more than two
standard deviations from the mean. Units are chosen to read at a glance:

| Parameter | Shown as |
|-----------|----------|
| focal scale `s` | a percentage, `100·s` |
| principal-point offset | pixels on each axis, `a·min(width, height)` |
| radial correction | its two coefficients `e₀`, `e₁` |
| rolling-shutter rates | rad per frame of readout on each axis: the stored rate, in rad per short side of readout, times the scanline count over the short side; with the scanlines setting |

The statistics are over the camera's posed images, the set step 1 folds over.

This revises the 2026-10-25 decision in [gui/camera-intrinsics.md §
Decisions](../gui/camera-intrinsics.md#decisions), which kept per-image values
out of the panel so that no per-image line would sit among the shared
parameters as if it were one of them. Statistics of the whole camera are a
statement about the camera, not about one image, and are labelled *per image*
so they do not read as shared values. Image Detail's legend keeps stating one
image's values.

A Scene Graph camera row gains a short tag naming its per-image parameters,
for example `per image: f, pp`, after the row's image count
([gui/camera-intrinsics.md § Scene Graph](../gui/camera-intrinsics.md#scene-graph-the-camera-intrinsics-group)).

MCP `get_camera_intrinsics` returns the same statistics per parameter, and a
new read returns every image's values for one camera, so an agent can check
the panel's numbers against the stored ones.

## Step 3: a histogram per component

Beside the statistics, each component of each parameter the camera varies
gets a small histogram of its images' values: one for the focal scale, two
each for the principal point and the radial correction, three for the rates.
They are hand-painted with the projection plot's helpers
([projection_plot.rs](../../crates/sfm-explorer/src/intrinsics_detail/projection_plot.rs)),
since the viewer has no plotting dependency and is not to gain one for this.
Hovering a bin lists the images in it; clicking it selects them, so an outlier
is one click from Image Detail.

## Step 4: dropping a parameter in Switch Camera Model

Switch Camera Model's proposal
([gui/edits/switch-camera-model.md § The proposal](../gui/edits/switch-camera-model.md#the-proposal)),
and other camera operations where it applies, offers one checkbox per
per-image parameter the camera varies: *drop per-image focal*, and so on. A
ticked box drops the parameter with step 1's `drop_image_adjustments` before
the fit, so the fit is of the folded camera, and the strip's existing
observation comparison shows what the drop costs together with the switch.
Dropping the radial correction and the rates lifts the refusal Switch Camera
Model makes today for a camera that carries them (`LensCorrections`); with
either kept, the refusal stands.

MCP `propose_camera_model` and `switch_camera_model` take the same choice as
an argument, the Action Log line names the parameters dropped, `ui_basic`
covers the checkboxes, and the pull request carries before and after
screenshots of the strip.

## Step 5: the CLI and the Bundle Adjust dialog

`sfm xform --bundle-adjust` gains a key that releases a camera's shared
principal point
([cli/reconstruction/xform/xform-command.md](../cli/reconstruction/xform/xform-command.md)).
Today the CLI moves it only through `image_principal_point=on`, while
`BundleAdjustTransform(release_principal_point=True)` and the viewer's dialog
can release it directly. Whether `sfm xform` also gains an option that drops
per-image parameters through step 1's operation is open.

The viewer's Bundle Adjust dialog
([gui/edits/bundle-adjust.md § The dialog](../gui/edits/bundle-adjust.md#the-dialog))
opens each camera's per-image release boxes ticked where the camera already
varies that parameter, so a second solve keeps solving what the first one
turned on unless the person clears the box. Today they always open clear.
