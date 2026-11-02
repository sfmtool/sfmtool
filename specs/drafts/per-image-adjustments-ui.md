# Showing and dropping per-image adjustments in the viewer and the CLI

**Status:** Draft. Decided: the five steps below, each its own pull request,
and the rule that only the bundle adjustment turns a per-image parameter on
while every other operation only drops one. Step 1 is built and specified in
[core/reconstruction/drop-image-adjustments.md](../core/reconstruction/drop-image-adjustments.md),
and step 2 in [gui/camera-intrinsics.md § 2.
Parameters](../gui/camera-intrinsics.md#2-parameters), whose Decisions record
that the rates are shown in their stored unit; this draft proposes steps 3 to
5. Open: the exact layout of the histograms, and whether `sfm xform` gets an
option to drop per-image parameters.

A camera in a `.sfmr` file can vary its focal, principal point, radial
correction and rolling-shutter rates per image
([sfmr-file-format.md § Per-image intrinsic
adjustments](../formats/sfmr-file-format.md#per-image-intrinsic-adjustments-optional-version-13)).
The viewer shows one image's values in Image Detail's legend, and the
statistics of all of a camera's images' values in the Camera Intrinsics panel,
and nothing in the viewer or the CLI can drop a per-image parameter once the
bundle adjustment has turned it on. This draft proposes how a person sees the
spread of a camera's per-image values, drops the ones they do not want, and
releases the parameters the CLI cannot release yet.

## Step 2: statistics in the Camera Intrinsics panel

Built. The panel's Parameters table states, beside each shared parameter the
camera also varies per image, the statistics of its images' values, a Scene
Graph camera row carries a tag naming its per-image parameters, and MCP
`get_camera_intrinsics` and `get_intrinsic_adjustments` report the statistics
and every image's values: [gui/camera-intrinsics.md §
2. Parameters](../gui/camera-intrinsics.md#2-parameters), [§ Scene
Graph](../gui/camera-intrinsics.md#scene-graph-the-camera-intrinsics-group)
and [gui/mcp-server.md](../gui/mcp-server.md#list_camera_images--get_camera_image--get_camera_intrinsics--get_intrinsic_adjustments--get_point).

## Step 3: a histogram per component

Beside its statistics in the Parameters table, each component of each
parameter the camera varies gets a small histogram of its images' values: one for the focal scale, two
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
