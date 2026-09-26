# Switch Camera Model

An edit that replaces one camera's lens model with one fitted to it and
installs the answer as the node's next **version**, which an undo steps back
out of. Its one entry in the window is the Camera Intrinsics panel's
`Refit spline…`, which switches a spline camera to its own spline model with
another coefficient count or domain end; the MCP tool `switch_camera_model`
reaches the same edit with any target model.

Related specs:
[../../core/reconstruction/switch-camera-model.md](../../core/reconstruction/switch-camera-model.md)
(the core function this wraps, and its report),
[../../core/camera/refit-camera-intrinsics.md](../../core/camera/refit-camera-intrinsics.md)
(the lens fit, and `refit_spline`),
[../camera-intrinsics.md](../camera-intrinsics.md) (the panel whose header
carries the action), [bundle-adjust.md](bundle-adjust.md) (the edit that then
refines the new coefficients), [../mcp-server.md](../mcp-server.md)
(`switch_camera_model` on the wire), and
[../../drafts/switch-camera-model.md](../../drafts/switch-camera-model.md) (the
proposal the viewer is to show before a switch is applied).

---

## Purpose

A spline camera's coefficient count and domain end say how finely its lens
curve can bend and where the bending stops. Bundle adjustment refines the
coefficients a camera has; it cannot change how many there are or where the
domain ends, because those are the camera's parameterization. Changing them is
a refit of that one camera: the old curve described on the new scheme, over the
whole new domain, which a later adjustment then fits to the observations. The
reason to do it in the viewer is the reason to adjust there: the outermost
keypoint of the camera's images is on screen beside the domain it should reach.

---

## Invocation

**`Refit spline…`** in the **Camera Intrinsics panel's header**, to the left of
`Copy ▾` ([../camera-intrinsics.md](../camera-intrinsics.md) § "Header"). It
acts on the camera the panel is showing. It is **greyed**, with the reason as
its hover text, for a camera whose model has no spline: *"A OPENCV_FISHEYE
camera has no spline to refit; only SFMTOOL_FISHEYE and SFMTOOL_PINHOLE carry
one. Switch the camera to one of those first (sfm xform --camera-model, or the
MCP tool switch_camera_model)."* The panel reports the click as
`IntrinsicsDetailResponse::refit_spline`, and `dock.rs` answers it with
`AppState::open_refit_spline`, which puts up the dialog below.

No menu entry and no shortcut. A switch to another model family, with the
proposal that shows the change before it is applied, is the draft's.

### The dialog

A small window, **Refit Spline**, in
[refit_spline_prompt.rs](../../../crates/sfm-explorer/src/refit_spline_prompt.rs):

- **Coefficients**: the count, from 2 to 32 (`SPLINE_COEFF_COUNT_RANGE`), and
  `now 8`. It starts at the camera's count.
- **Spline domain (°)**: the domain end as an incidence angle (up to 180°, and
  below 90° for `SFMTOOL_PINHOLE`), and `now 150.1°`. It starts at the camera's
  domain end, and a field left there keeps the domain end exactly rather than
  taking it through degrees and back. Under it is the **outermost keypoint** of
  the camera's images
  ([../../core/reconstruction/outermost-keypoint.md](../../core/reconstruction/outermost-keypoint.md)):
  `outermost keypoint: 230.3 px, 95.8° observed; 259.2 px, 108.8° detected`,
  and a **Use 108.8°** button that sets the domain to that angle. The button
  takes the detected keypoint, and the observed one where no `.sift` file could
  be read, in which case the text names only the observed one. The default
  domain is not changed: a circular fisheye is trimmed to its image circle by
  choice. The keypoints are read once, when the dialog opens, on the GUI thread.
- **Apply** and **Cancel**. `Enter` applies, `Escape` and the window's close
  button cancel.

Asking twice while it is up does not stack a second dialog.

---

## Mechanism

Everything below the dialog is
[../../core/reconstruction/switch-camera-model.md](../../core/reconstruction/switch-camera-model.md):
`sfmtool_core::reconstruction::switch_camera_model` over one camera. The dialog's
answer becomes a `SwitchCameraModelRequest` with no model named, which is the
camera's own, and the core function fits a spline camera switched to its own
model with `refit_spline`: over the whole new domain, constrained to stay
monotone. The viewer adds the invocation, the version and the history entry, in
[state/edits/switch_camera_model.rs](../../../crates/sfm-explorer/src/state/edits/switch_camera_model.rs).

`AppState::switch_camera_model` takes the request the wire takes too:

| Field | Default |
|-------|---------|
| `camera` | required: the camera's index in the node's table |
| `camera_model` | the camera's own model |
| `coeff_count` | the camera's own count when the target is its own spline model, else the core default, 8 |
| `spline_domain_deg` | the camera's own domain end in a refit, else the far image corner |
| `theta_fit_deg` | the core default; given, even a refit is an ordinary fit over that angle |

It runs **on the GUI thread**: a fit is a few milliseconds, and the only file
read is the `.sift` positions for the report's outermost keypoint.

### The version

A bulk edit. The value is the whole new base the core function produced, with
an empty overlay; the current value is materialised first when its overlay is
not empty. The switch moves no pose, point, keypoint or track and deletes
nothing, so the `RowMap::by_scan` read off its input and output is the
identity, and the selection stays where it is. The camera table changes and the
stored errors of the points the camera's images observe are recomputed, so the
panels drop what they cached about the node, the Camera Intrinsics panel's
derived report and the Image Detail overlay among it.

The version's label is, for a refit of a spline,

`Refit spline of camera 0 of kerry_park: 8 → 12 coefficients, domain 150.1° → 108.8°`

with a part that did not move named as kept (`8 coefficients kept`, `domain
150.1° kept`), and, for a change of model,

`Switched camera 0 of kerry_park from OPENCV_FISHEYE to SFMTOOL_FISHEYE`.

### The Action Log

One entry, of kind `Edit`: the label, then the fit's report:

`Refit spline of camera 0 of kerry_park: 8 → 12 coefficients, domain 150.1° →
108.8°: fit rms 0.004 px, max 0.013 px over θ ≤ 108.8° (spline_domain);
median error 0.298 → 0.301 px over 4210 observations (v5 → v6)`

`rms` and `max` are the new camera's distance from the old over the fit, which
says how much of a later change is the refit rather than the adjustment. Where
the monotonicity constraint bound, `; monotone constraint bound at 3 angles,
112.0°–118.7°` follows, since over that range the refit is the closest
invertible curve rather than the old one. The median is over the camera's
observations with an error under both models.

A refusal is one **failed** entry, `Switch camera model of <node> refused:
camera 0: <reason>`, carrying the core function's sentence: a camera the table
does not have, a count on a model without a spline, a domain end the model
cannot have. Nothing is pushed.

---

## Testing

Core: [../../core/reconstruction/switch-camera-model.md](../../core/reconstruction/switch-camera-model.md)
§ "Testing", which pins the refit of a spline to `refit_spline`.

Explorer (`sfm-explorer` lib tests, headless):

- `refit_spline_prompt/tests.rs`: the fields starting at the camera's count and
  domain, an untouched domain answered as kept, the answer becoming a request
  with no model and no fit angle, `Escape` cancelling, a second ask not
  stacking, the outermost keypoint's button taking the detected angle, the
  observed one without a detected and nothing without either, the keypoint text
  labelled by its source, and a camera with no spline refused.
- `state/edits/tests.rs`: the dialog's gates read off a spline camera, and a
  pinhole and a missing camera refused; a refit pushing one version with the
  label naming both changes, only the named camera changing, the log row
  carrying the fit's numbers, and undo bringing the old camera back; a domain
  end past 90° on `SFMTOOL_PINHOLE` and a count on a `SIMPLE_PINHOLE` refused
  naming the camera, pushing nothing.
- `mcp/tests/edit.rs`: `switch_camera_model` parsing, a refit with the
  camera's own count and domain kept by default, a named domain keeping the
  count, a named model switching models, and a missing camera and an
  impossible domain refused.

---

## Non-goals

- A proposal shown before the switch is applied, with the two models compared
  in the panel and the Image Detail overlay. That is
  [../../drafts/switch-camera-model.md](../../drafts/switch-camera-model.md).
- A dialog for a change of model family. The wire and `sfm xform
  --camera-model` make one; the window's is the proposal above.
- Refitting several cameras at once. A rig's cameras are refitted one by one,
  each its own version.
