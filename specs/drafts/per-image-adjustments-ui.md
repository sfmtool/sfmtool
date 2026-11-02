# Showing and dropping per-image adjustments in the viewer and the CLI

**Status:** Draft. Decided: the five steps below, each its own pull request,
and the rule that only the bundle adjustment turns a per-image parameter on
while every other operation only drops one. Step 1 is built and specified in
[core/reconstruction/drop-image-adjustments.md](../core/reconstruction/drop-image-adjustments.md),
and steps 2 and 3 in [gui/camera-intrinsics.md § 2.
Parameters](../gui/camera-intrinsics.md#2-parameters), whose Decisions record
that the rates are shown in milliradians per short side, the unit of Image
Detail's legend, rather than per frame of readout, and that a click on a
histogram's bin selects one of its images at a time, and step 4 in
[gui/edits/switch-camera-model.md § Dropping per-image
parameters](../gui/edits/switch-camera-model.md#dropping-per-image-parameters)
and [core/reconstruction/switch-camera-model.md § The
drops](../core/reconstruction/switch-camera-model.md#the-drops); this draft
proposes step 5. Open: whether `sfm xform` gets an option to drop per-image
parameters.

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

Built. Each `per image` row of the Parameters table draws a histogram of its
images' values, beside its statistics where the table leaves room and under
them otherwise; hovering a bin lists its images, a click selects one of them,
stepping through the bin on repeated clicks, and MCP `get_camera_intrinsics`
reports the bins: [gui/camera-intrinsics.md § 2.
Parameters](../gui/camera-intrinsics.md#2-parameters) and
[gui/mcp-server.md](../gui/mcp-server.md#list_camera_images--get_camera_image--get_camera_intrinsics--get_intrinsic_adjustments--get_point).

## Step 4: dropping a parameter in Switch Camera Model

Built. Switch Camera Model's proposal has a Drop per image row, one checkbox
per per-image parameter the camera varies, cleared by default; a ticked box
drops the parameter with step 1's `drop_image_adjustments` before the fit, and
the observation comparison shows what the drop costs together with the switch.
A parameter the target cannot carry (the rolling-shutter rates off
`SFMTOOL_PINHOLE`, the radial correction outside a refit of an
`SFMTOOL_PINHOLE` spline that keeps its domain end) is ticked and greyed, so a
camera carrying them opens a fitted proposal rather than the `LensCorrections`
refusal; kept on such a target, the core switch still refuses. MCP
`propose_camera_model` and `switch_camera_model` take
`drop_per_image_adjustments`, and the version label and the Action Log line
name the parameters dropped:
[gui/edits/switch-camera-model.md § Dropping per-image
parameters](../gui/edits/switch-camera-model.md#dropping-per-image-parameters),
[core/reconstruction/switch-camera-model.md § The
drops](../core/reconstruction/switch-camera-model.md#the-drops) and
[gui/mcp-server.md](../gui/mcp-server.md#switch_camera_model).

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
